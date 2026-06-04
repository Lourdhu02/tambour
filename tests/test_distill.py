"""Knowledge-distillation tests (CPU, dataset-free)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tambour.config import NUM_CLASSES          # noqa: E402
from tambour.distill import KDLoss              # noqa: E402
from tambour.models import build_model          # noqa: E402

H, W = 32, 128


def test_kd_loss_backward():
    torch.manual_seed(0)
    teacher = build_model("tambour-n", NUM_CLASSES).eval()
    student = build_model("tambour-n", NUM_CLASSES)
    x = torch.randn(2, 3, H, W)
    with torch.no_grad():
        t = teacher(x)
    loss = KDLoss(2.0)(student(x), t)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())


def test_kd_pulls_student_to_teacher():
    torch.manual_seed(0)
    teacher = build_model("tambour-n", NUM_CLASSES).eval()
    student = build_model("tambour-n", NUM_CLASSES)
    x = torch.randn(4, 3, H, W)
    with torch.no_grad():
        t = teacher(x)
    kd = KDLoss(2.0)
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3)
    first = kd(student(x), t).item()
    for _ in range(60):
        loss = kd(student(x), t)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first, f"KD did not reduce: {first:.4f} -> {loss.item():.4f}"


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
