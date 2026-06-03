"""Deployment tests (CPU, dataset-free)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tambour.config import NUM_CLASSES                       # noqa: E402
from tambour.deploy import benchmark, quantize_dynamic_int8  # noqa: E402
from tambour.models import build_model                       # noqa: E402

H, W = 32, 128


def test_quantize_int8_forward():
    net = build_model("tambour-n", NUM_CLASSES)
    q = quantize_dynamic_int8(net)
    with torch.no_grad():
        out = q(torch.randn(1, 3, H, W))
    assert out.shape[0] == 1 and out.shape[2] == NUM_CLASSES
    assert torch.isfinite(out).all()


def test_benchmark_positive():
    net = build_model("tambour-n", NUM_CLASSES)
    tput = benchmark(net, H, W, "cpu", batch=2, iters=3, warmup=1)
    assert tput > 0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
