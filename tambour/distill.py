"""Knowledge distillation: a large teacher (tambour-l/-b) -> a small student (tambour-n).

The student is trained with hard-label CTC *and* a temperature-softened KL toward the
teacher's distribution. All variants share the same patch/merge strides, so for a given
input size the timestep count T matches between teacher and student and the per-(B, T)
distributions align directly. This recovers most of the teacher's accuracy at the
student's latency (the PP-OCR distillation recipe).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from .config import NUM_CLASSES
from .data import build_loaders
from .engine import (ModelEMA, _amp_dtype, _save, build_scheduler, ctc_loss_inputs,
                     evaluate, resolve_device, seed_all)
from .infer import load_checkpoint
from .models import build_model
from .text import CTCCodec


class KDLoss(nn.Module):
    """Temperature-softened KL divergence between student and teacher log-probs."""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.t = temperature

    def forward(self, student_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor) -> torch.Tensor:
        c = student_logprobs.size(-1)
        s = F.log_softmax(student_logprobs.reshape(-1, c) / self.t, dim=-1)
        t = F.softmax(teacher_logprobs.reshape(-1, c) / self.t, dim=-1)
        return F.kl_div(s, t, reduction="batchmean") * (self.t ** 2)


def distill_fit(teacher_ckpt: str, student: str, data_dir: str, cfg: Dict,
                project: str = "runs", name: str = "distill",
                alpha: float = 0.5, temperature: float = 2.0) -> str:
    """Train ``student`` from a frozen teacher checkpoint. alpha weights hard CTC vs KD."""
    device = resolve_device(cfg.get("device"))
    seed_all(cfg.get("seed", 42))
    codec = CTCCodec()

    teacher, tck = load_checkpoint(teacher_ckpt, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    cfg["img_h"], cfg["img_w"] = tck["img_h"], tck["img_w"]  # align T with the teacher

    train_loader, val_loader, _, meta = build_loaders(data_dir, cfg, codec)
    id2dom = {v: k for k, v in meta["domain_to_id"].items()}
    net = build_model(student, NUM_CLASSES, num_domains=len(meta["domain_to_id"])).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.05))
    sched = build_scheduler(opt, cfg, len(train_loader))
    ctc, kd = nn.CTCLoss(blank=0, zero_infinity=True), KDLoss(temperature)
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp)
    ema = ModelEMA(net, cfg.get("ema_decay", 0.9995))

    run_dir = Path(project) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in net.parameters())
    print(f"  distill {tck['model_name']} ({teacher_params:,}) -> {student} ({student_params:,}) "
          f"| alpha {alpha} T {temperature} | device {device}")

    best, best_path = -1.0, run_dir / "best.pth"
    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        for imgs, targets, lengths, _, dom in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            lengths, dom = lengths.to(device), dom.to(device)
            with autocast(device.type, dtype=_amp_dtype(), enabled=use_amp):
                s_lp = net(imgs, dom)
                with torch.no_grad():
                    t_lp = teacher(imgs, dom)
                lp, in_len = ctc_loss_inputs(s_lp, device)
                loss = alpha * ctc(lp, targets, in_len, lengths) + (1 - alpha) * kd(s_lp, t_lp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), cfg.get("grad_clip", 5.0))
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= before:
                ema.update(net)
        sched.step()
        val = evaluate(net, val_loader, ctc, device, codec, id2dom, use_amp)
        print(f"  distill epoch {epoch}/{cfg['epochs']} | val_exact {val['exact']:.4f} "
              f"| char {val['char_acc']:.4f}")
        _save(run_dir / "last.pth", net, ema, student, cfg, meta, val, epoch)
        if val["exact"] > best or not best_path.exists():
            _save(best_path, net, ema, student, cfg, meta, val, epoch)
        best = max(best, val["exact"])
    return str(best_path)
