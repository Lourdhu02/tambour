"""Deployment helpers: quantization and throughput benchmarking.

Dynamic INT8 quantization of the Linear-heavy transformer blocks typically gives a
~2-4x CPU speedup for well under ~1% accuracy loss; FP16 halves GPU latency near
losslessly. Benchmark torch (FP32/INT8) and ONNX Runtime to pick a serving target.
"""
from __future__ import annotations

import time
from typing import Union

import torch
import torch.nn as nn


def quantize_dynamic_int8(net: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization of Linear layers for CPU inference."""
    net = net.cpu().eval()
    return torch.quantization.quantize_dynamic(net, {nn.Linear}, dtype=torch.qint8)


def to_fp16(net: nn.Module) -> nn.Module:
    """Half precision for GPU inference (near-lossless, ~2x)."""
    return net.eval().half()


@torch.no_grad()
def benchmark(net: nn.Module, img_h: int, img_w: int, device: Union[str, torch.device] = "cpu",
              batch: int = 8, iters: int = 20, warmup: int = 3, fp16: bool = False) -> float:
    """Return throughput in images/second."""
    device = torch.device(device)
    net = net.to(device).eval()
    x = torch.randn(batch, 3, img_h, img_w, device=device)
    if fp16 and device.type == "cuda":
        net, x = net.half(), x.half()
    for _ in range(warmup):
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batch * iters / (time.perf_counter() - t0)


def onnx_throughput(onnx_path: str, img_h: int, img_w: int, batch: int = 8, iters: int = 20) -> float:
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, 3, img_h, img_w).astype("float32")
    for _ in range(3):
        sess.run(None, {"image": x})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {"image": x})
    return batch * iters / (time.perf_counter() - t0)
