"""Load a checkpoint and predict readings with calibrated confidence + abstention."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch

from .confidence import decode_conf, tta_logits
from .data import build_transforms
from .models import TambourNet
from .text import CTCCodec

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def load_checkpoint(path: str, device="cpu", use_ema: bool = True) -> Tuple[TambourNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = TambourNet(**ckpt["model_config"])
    state = ckpt.get("ema_state") if use_ema and ckpt.get("ema_state") else ckpt["model_state"]
    net.load_state_dict(state)
    return net.to(device).eval(), ckpt


def _preprocess(img_bgr, transform):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return transform(image=img)["image"]


@torch.no_grad()
def predict_image(net, image_path, codec, transform, device, T: float = 1.0,
                  tau: float = 0.0, domain_id: Optional[int] = None, tta: bool = False):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    x = _preprocess(img, transform).unsqueeze(0).to(device)
    dom = torch.tensor([domain_id], device=device) if domain_id is not None else None
    if tta:
        views = [x] + [_preprocess(cv2.convertScaleAbs(img, alpha=a, beta=b), transform).unsqueeze(0).to(device)
                       for a, b in ((0.9, 10), (1.1, -10))]
        log_probs = tta_logits(net, torch.cat(views), dom.repeat(len(views)) if dom is not None else None)[0]
    else:
        log_probs = net(x, dom)[0]
    text, conf, _ = decode_conf(log_probs, codec, T)
    abstain = conf < tau
    return text, conf, abstain


@torch.no_grad()
def predict_dir(net, source_dir, codec, transform, device, T=1.0, tau=0.0, tta=False):
    files = sorted(f for f in Path(source_dir).iterdir() if f.suffix.lower() in IMG_EXT)
    results: List[Tuple[str, str, float, bool]] = []
    for f in files:
        text, conf, ab = predict_image(net, f, codec, transform, device, T, tau, tta=tta)
        results.append((f.name, text, conf, ab))
    return results


def build_eval_transform(ckpt: dict):
    return build_transforms(ckpt["img_h"], ckpt["img_w"], training=False)


def run_predict(ckpt_path: str, source: str, device="cpu", T: float = 1.0,
                tau: float = 0.0, tta: bool = False):
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    net, ckpt = load_checkpoint(ckpt_path, dev)
    codec = CTCCodec()
    transform = build_eval_transform(ckpt)
    p = Path(source)
    if p.is_dir():
        rows = predict_dir(net, p, codec, transform, dev, T, tau, tta)
        for name, text, conf, ab in rows:
            flag = "  [ABSTAIN]" if ab else ""
            print(f"  {name} -> {text}  (conf {conf:.3f}){flag}")
        return rows
    text, conf, ab = predict_image(net, p, codec, transform, dev, T, tau, tta=tta)
    print(f"  {p.name} -> {text}  (conf {conf:.3f}){'  [ABSTAIN]' if ab else ''}")
    return text, conf, ab
