"""Portable CPU test suite (synthetic data -> no dataset dependency).

    python tests/test_engine.py     # standalone runner
    python -m pytest tests          # or pytest
"""
from __future__ import annotations

import atexit
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tambour.agents import data_agent, model_agent           # noqa: E402
from tambour.config import NUM_CLASSES, VARIANTS             # noqa: E402
from tambour.confidence import (TemperatureScaler,           # noqa: E402
                                choose_threshold, decode_conf)
from tambour.data import build_loaders, build_transforms      # noqa: E402
from tambour.data.dataset import MeterDataset, collate_fn     # noqa: E402
from tambour.data.manifest import group_split, parse_manifest  # noqa: E402
from tambour.data.sampler import make_domain_balanced_sampler  # noqa: E402
from tambour.data.shards import ShardDataset, write_shards     # noqa: E402
from tambour.data.transforms import fit_pad, synthetic_rolling  # noqa: E402
from tambour.export import export_onnx                        # noqa: E402
from tambour.losses import AugLoss                            # noqa: E402
from tambour.models import build_model                        # noqa: E402
from tambour.text import CTCCodec, billed_reading             # noqa: E402

H, W = 32, 128


def _render(label, h=H, w=W):
    img = np.full((h, w, 3), 20, np.uint8)
    cv2.putText(img, label, (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
    return img


def _build_dataset(meters=16, per=4, domains=("rrf", "mrc")):
    root = Path(tempfile.mkdtemp(prefix="tambour_"))
    atexit.register(lambda: shutil.rmtree(root, ignore_errors=True))
    imgdir = root / "images"
    imgdir.mkdir(parents=True)
    rng = random.Random(0)
    lines = []
    for mi in range(meters):
        dom = domains[mi % len(domains)]
        mid = f"{1000 + mi}{dom.upper()}{mi}"
        for k in range(per):
            label = "".join(rng.choice("0123456789") for _ in range(rng.choice([5, 6])))
            fname = f"{mid}-2025-01-{k + 1:02d}-10-00-{mi:02d}.png"
            cv2.imwrite(str(imgdir / fname), _render(label))
            lines.append(f"{fname}\t{label}\t{dom}")
    (root / "labels.txt").write_text("\n".join(lines))
    return root


_DATA = _build_dataset()


def test_codec():
    c = CTCCodec()
    assert c.num_classes == NUM_CLASSES == 12
    assert c.decode(c.encode("01234")) == "01234"
    assert c.decode([1, 1, 0, 2, 2]) == c.i2c[1] + c.i2c[2]
    assert billed_reading("01234.5") == "01234"


def test_manifest_and_split():
    samples = parse_manifest(str(_DATA))
    assert len(samples) == 64
    assert {s.domain for s in samples} == {"rrf", "mrc"}
    splits = group_split(samples, (0.9, 0.05, 0.05), seed=42)
    g = {k: {s.meter_id for s in v} for k, v in splits.items()}
    assert g["train"].isdisjoint(g["test"]) and g["train"].isdisjoint(g["val"])
    assert sum(len(v) for v in splits.values()) == 64


def test_transforms():
    padded = fit_pad(np.zeros((20, 200, 3), np.uint8), H, W)
    assert padded.shape == (H, W, 3)
    assert synthetic_rolling(np.zeros((H, W, 3), np.uint8)).shape == (H, W, 3)
    tf = build_transforms(H, W, training=True)
    out = tf(image=np.zeros((30, 90, 3), np.uint8))["image"]
    assert tuple(out.shape) == (3, H, W)


def test_domain_balanced_sampler():
    ids = [0] * 90 + [1] * 10
    sampler = make_domain_balanced_sampler(ids, temperature=0.5)
    torch.manual_seed(0)
    drawn = [ids[i] for _ in range(30) for i in sampler]  # ~3000 draws, deterministic
    frac = sum(d == 1 for d in drawn) / len(drawn)
    assert frac > 0.2, frac  # minority (true rate 0.10) oversampled toward ~0.25


def test_model_all_variants():
    x = torch.randn(2, 3, H, W)
    for v in VARIANTS:
        net = build_model(v, NUM_CLASSES).eval()
        with torch.no_grad():
            out = net(x)
        assert out.shape[0] == 2 and out.shape[2] == NUM_CLASSES and out.shape[1] >= 13
        assert torch.isfinite(out).all()
    # domain conditioning path
    net = build_model("tambour-n", NUM_CLASSES, num_domains=2).eval()
    with torch.no_grad():
        assert net(x, torch.tensor([0, 1])).shape[2] == NUM_CLASSES
        assert net(x).shape[2] == NUM_CLASSES  # domain_ids=None skips adapter


def test_losses_backward():
    net = build_model("tambour-n", NUM_CLASSES)
    x = torch.randn(2, 3, H, W)
    lp = net(x)
    il = torch.full((2,), lp.size(1), dtype=torch.long)
    targets, lengths = torch.randint(1, 11, (12,)), torch.tensor([6, 6])
    for crit in (torch.nn.CTCLoss(blank=0, zero_infinity=True), AugLoss(positions=2)):
        loss = crit(lp.permute(1, 0, 2), targets, il, lengths)
        assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in net.parameters())


def test_build_loaders():
    cfg = dict(img_h=H, img_w=W, batch=8, workers=0, seed=42, split=(0.7, 0.15, 0.15),
               group_split=True, domain_balanced=True, domain_temperature=0.5,
               aug_level="light", rolling_aug=0.1)
    train, val, test, meta = build_loaders(str(_DATA), cfg, CTCCodec())
    assert len(meta["domain_to_id"]) == 2
    imgs, targets, lengths, labels, dom = next(iter(train))
    assert imgs.shape[1:] == (3, H, W) and len(labels) == imgs.shape[0]
    assert dom.shape[0] == imgs.shape[0]


def test_overfit():
    torch.manual_seed(0)
    labels = ["12345", "67890", "024681", "13579"]
    tf = build_transforms(H, W, training=False)
    x = torch.stack([tf(image=cv2.cvtColor(_render(l), cv2.COLOR_BGR2RGB))["image"] for l in labels])
    targets = torch.cat([torch.tensor(CTCCodec().encode(l)) for l in labels])
    lengths = torch.tensor([len(l) for l in labels])
    net = build_model("tambour-n", NUM_CLASSES)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    crit = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    first = None
    for step in range(400):
        lp = net(x)
        il = torch.full((4,), lp.size(1), dtype=torch.long)
        loss = crit(lp.permute(1, 0, 2), targets, il, lengths)
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    last = loss.item()
    c = CTCCodec()
    net.eval()
    with torch.no_grad():
        preds = [c.decode(r) for r in net(x).argmax(2).numpy()]
    correct = sum(p == g for p, g in zip(preds, labels))
    assert last < 0.5 * first and correct >= 3, f"{correct}/4 last={last:.3f} preds={preds}"


def test_confidence():
    c = CTCCodec()
    logp = [torch.randn(20, NUM_CLASSES).log_softmax(-1).numpy() for _ in range(20)]
    labels = [c.decode(np.asarray(l).argmax(-1)) for l in logp]
    scaler = TemperatureScaler().fit(logp, labels, c)
    assert scaler.T > 0
    text, conf, _ = decode_conf(torch.tensor(logp[0]), c, scaler.T)
    assert 0.0 <= conf <= 1.0
    tau = choose_threshold([0.9, 0.8, 0.4], [1, 1, 0], target_precision=1.0)
    assert tau >= 0.8


def test_export_onnx():
    net = build_model("tambour-n", NUM_CLASSES)
    with tempfile.TemporaryDirectory() as d:
        out = export_onnx(net, H, W, str(Path(d) / "m.onnx"))
        assert Path(out).exists() and Path(out).stat().st_size > 0


def test_shards_roundtrip():
    samples = parse_manifest(str(_DATA))
    with tempfile.TemporaryDirectory() as d:
        paths = write_shards(_DATA / "images", samples, d, shard_size=20)
        assert len(paths) >= 1
        ds = ShardDataset(paths, build_transforms(H, W, False), CTCCodec(), {"rrf": 0, "mrc": 1})
        count = sum(1 for _ in ds)
    assert count == len(samples)


def test_agents():
    assert model_agent("tambour-n", torch.device("cpu"), H, W)["status"] == "ok"
    rep = data_agent(str(_DATA))
    assert rep["status"] == "ok" and rep["meter_id_leakage"] == 0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
