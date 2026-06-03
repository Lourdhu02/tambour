"""Tambour command-line interface."""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import VARIANTS, load_config


def _train(a):
    from .engine import fit
    cfg = load_config(a.config, epochs=a.epochs, batch=a.batch, lr=a.lr, device=a.device,
                      workers=a.workers, aug_loss=a.aug_loss or None, sgm_weight=a.sgm or None)
    fit(a.model, a.data, cfg, project=a.project, name=a.name)


def _val(a):
    import torch.nn as nn
    from .data import build_loaders
    from .engine import evaluate, resolve_device
    from .infer import load_checkpoint
    from .text import CTCCodec
    dev = resolve_device(a.device)
    net, ckpt = load_checkpoint(a.ckpt, dev)
    cfg = load_config(None, img_h=ckpt["img_h"], img_w=ckpt["img_w"], batch=a.batch,
                      workers=0, domain_balanced=False)
    _, val_loader, test_loader, meta = build_loaders(a.data, cfg, CTCCodec())
    id2dom = {v: k for k, v in meta["domain_to_id"].items()}
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    for nm, ld in (("val", val_loader), ("test", test_loader)):
        if ld is not None:
            m = evaluate(net, ld, crit, dev, CTCCodec(), id2dom)
            print(f"  {nm}: exact {m['exact']:.4f} | char {m['char_acc']:.4f} | n {m['n']}")


def _predict(a):
    from .infer import run_predict
    run_predict(a.ckpt, a.source, a.device, T=a.temperature, tau=a.tau, tta=a.tta)


def _export(a):
    from .export import run_export
    run_export(a.ckpt, a.output, a.opset)


def _calibrate(a):
    import json
    import numpy as np
    from .confidence import TemperatureScaler, choose_threshold, decode_conf
    from .data import build_loaders
    from .engine import resolve_device
    from .infer import load_checkpoint
    from .text import CTCCodec
    import torch
    dev = resolve_device(a.device)
    net, ckpt = load_checkpoint(a.ckpt, dev)
    codec = CTCCodec()
    cfg = load_config(None, img_h=ckpt["img_h"], img_w=ckpt["img_w"], batch=a.batch,
                      workers=0, domain_balanced=False)
    _, val_loader, _, _ = build_loaders(a.data, cfg, codec)
    logps, labels = [], []
    with torch.no_grad():
        for imgs, _, _, lbls, dom in val_loader:
            lp = net(imgs.to(dev), dom.to(dev)).cpu().numpy()
            logps.extend(lp)
            labels.extend(lbls)
    scaler = TemperatureScaler().fit(logps, labels, codec)
    confs, correct = [], []
    for lp, gt in zip(logps, labels):
        text, mn, _ = decode_conf(np.asarray(lp), codec, scaler.T)
        confs.append(mn); correct.append(int(text == gt))
    tau = choose_threshold(confs, correct, a.precision)
    coverage = float(np.mean(np.asarray(confs) >= tau))
    print(f"  temperature T = {scaler.T:.3f} | tau = {tau:.3f} for >={a.precision:.1%} precision "
          f"| coverage {coverage:.1%}")
    out = {"temperature": scaler.T, "tau": tau, "target_precision": a.precision, "coverage": coverage}
    with open(a.out or "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved: {a.out or 'calibration.json'}")


def _mine(a):
    """Active-learning: flag low-confidence / abstained reads for relabeling."""
    from .infer import run_predict
    rows = run_predict(a.ckpt, a.source, a.device, T=a.temperature, tau=a.tau)
    if isinstance(rows, list):
        flagged = [(n, t, c) for (n, t, c, ab) in rows if ab or c < a.tau]
        with open(a.out, "w", encoding="utf-8") as f:
            for n, t, c in sorted(flagged, key=lambda r: r[2]):
                f.write(f"{n}\t{t}\t{c:.4f}\n")
        print(f"  flagged {len(flagged)} reads for review -> {a.out}")


def _sweep(a):
    """Take all 4 variants through build -> forward -> a few train steps -> export."""
    import tempfile
    import torch
    import torch.nn as nn
    from .config import NUM_CLASSES
    from .engine import resolve_device
    from .export import export_onnx
    from .models import build_model
    dev = resolve_device(a.device)
    print(f"  SWEEP smoke | device {dev} | variants {list(VARIANTS)}")
    h, w = 48, 320
    rows = []
    for v in VARIANTS:
        net = build_model(v, NUM_CLASSES).to(dev)
        x = torch.randn(4, 3, h, w, device=dev)
        tgt = torch.randint(1, 11, (4 * 6,), device=dev)
        lens = torch.full((4,), 6, device=dev)
        crit = nn.CTCLoss(blank=0, zero_infinity=True)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        losses = []
        for _ in range(5):
            lp = net(x)
            il = torch.full((4,), lp.size(1), dtype=torch.long, device=dev)
            loss = crit(lp.permute(1, 0, 2), tgt, il, lens)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        with tempfile.TemporaryDirectory() as d:
            ok = bool(export_onnx(net, h, w, f"{d}/{v}.onnx", check=False))
        n = sum(p.numel() for p in net.parameters())
        rows.append((v, n, losses[0], losses[-1], losses[-1] < losses[0], ok))
        print(f"    {v:<10} {n:>11,} params | loss {losses[0]:.2f}->{losses[-1]:.2f} "
              f"| {'learns' if losses[-1] < losses[0] else 'NO'} | export {'ok' if ok else 'FAIL'}")
    print(f"  all {len(rows)} variants {'PASSED' if all(r[4] and r[5] for r in rows) else 'FAILED'}")


def _shards(a):
    from .data.manifest import parse_manifest
    from .data.shards import write_shards
    from pathlib import Path
    samples = parse_manifest(a.data)
    paths = write_shards(Path(a.data) / "images", samples, a.out, a.shard_size, a.resize_h)
    print(f"  wrote {len(paths)} shards ({len(samples)} samples) -> {a.out}")


def _agents(a):
    from .agents import run_agents
    run_agents(a.model, a.data, a.device)


def _info(a):
    import torch
    from .config import NUM_CLASSES, DEFAULTS
    from .models import build_model
    net = build_model(a.model, NUM_CLASSES)
    n = sum(p.numel() for p in net.parameters())
    with torch.no_grad():
        out = net(torch.randn(1, 3, DEFAULTS["img_h"], DEFAULTS["img_w"]))
    print(f"  {a.model} | {n:,} params | input 3x{DEFAULTS['img_h']}x{DEFAULTS['img_w']} "
          f"| output {out.shape[1]}x{out.shape[2]}")


def main():
    p = argparse.ArgumentParser(prog="tambour", description=f"Tambour OCR v{__version__} (analog meter counters)")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train"); t.set_defaults(fn=_train)
    t.add_argument("--model", default="tambour-b", choices=list(VARIANTS))
    t.add_argument("--data", required=True); t.add_argument("--config", default=None)
    t.add_argument("--epochs", type=int); t.add_argument("--batch", type=int)
    t.add_argument("--lr", type=float); t.add_argument("--device"); t.add_argument("--workers", type=int)
    t.add_argument("--project", default="runs"); t.add_argument("--name", default="exp")
    t.add_argument("--aug-loss", action="store_true", dest="aug_loss")
    t.add_argument("--sgm", type=float, default=None)

    v = sub.add_parser("val"); v.set_defaults(fn=_val)
    v.add_argument("--ckpt", required=True); v.add_argument("--data", required=True)
    v.add_argument("--batch", type=int, default=128); v.add_argument("--device")

    pr = sub.add_parser("predict"); pr.set_defaults(fn=_predict)
    pr.add_argument("--ckpt", required=True); pr.add_argument("--source", required=True)
    pr.add_argument("--device"); pr.add_argument("--temperature", type=float, default=1.0)
    pr.add_argument("--tau", type=float, default=0.0); pr.add_argument("--tta", action="store_true")

    e = sub.add_parser("export"); e.set_defaults(fn=_export)
    e.add_argument("--ckpt", required=True); e.add_argument("--output", default=None)
    e.add_argument("--opset", type=int, default=18)

    c = sub.add_parser("calibrate"); c.set_defaults(fn=_calibrate)
    c.add_argument("--ckpt", required=True); c.add_argument("--data", required=True)
    c.add_argument("--batch", type=int, default=128); c.add_argument("--device")
    c.add_argument("--precision", type=float, default=0.995); c.add_argument("--out", default=None)

    m = sub.add_parser("mine"); m.set_defaults(fn=_mine)
    m.add_argument("--ckpt", required=True); m.add_argument("--source", required=True)
    m.add_argument("--out", default="to_relabel.txt"); m.add_argument("--tau", type=float, default=0.9)
    m.add_argument("--device"); m.add_argument("--temperature", type=float, default=1.0)

    sw = sub.add_parser("sweep"); sw.set_defaults(fn=_sweep); sw.add_argument("--device")

    sh = sub.add_parser("shards"); sh.set_defaults(fn=_shards)
    sh.add_argument("--data", required=True); sh.add_argument("--out", default="shards")
    sh.add_argument("--shard-size", type=int, default=10000, dest="shard_size")
    sh.add_argument("--resize-h", type=int, default=0, dest="resize_h")

    ag = sub.add_parser("agents"); ag.set_defaults(fn=_agents)
    ag.add_argument("--model", default="tambour-b"); ag.add_argument("--data"); ag.add_argument("--device")

    inf = sub.add_parser("info"); inf.set_defaults(fn=_info)
    inf.add_argument("--model", default="tambour-b", choices=list(VARIANTS))

    args = p.parse_args()
    if not args.cmd:
        p.print_help(); sys.exit(0)
    print(f"\n  Tambour v{__version__}\n")
    args.fn(args)


if __name__ == "__main__":
    main()
