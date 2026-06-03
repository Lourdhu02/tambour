"""Diagnostic agents: model health, latency, and dataset audit (multi-domain aware)."""
from __future__ import annotations

import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import NUM_CLASSES
from .data.manifest import group_split, parse_manifest
from .engine import resolve_device
from .models import build_model


def model_agent(variant: str, device: torch.device, img_h=48, img_w=320) -> Dict:
    checks: List[Dict] = []
    try:
        net = build_model(variant, NUM_CLASSES).to(device).eval()
        n = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            out = net(torch.randn(2, 3, img_h, img_w, device=device))
        ok = out.ndim == 3 and out.shape[0] == 2 and out.shape[2] == NUM_CLASSES and torch.isfinite(out).all()
        checks.append({"name": "forward", "status": "ok" if ok else "error",
                       "details": f"{variant} {n:,} params -> {tuple(out.shape)}"})
        return {"status": "ok" if ok else "error", "checks": checks, "net": net, "params": n}
    except Exception as exc:
        return {"status": "error", "checks": [{"name": "forward", "status": "error", "details": str(exc)}]}


@torch.no_grad()
def latency_agent(net, device, img_h=48, img_w=320, batch=8, warmup=3, steps=10) -> Dict:
    net = net.to(device).eval()
    x = torch.randn(batch, 3, img_h, img_w, device=device)
    for _ in range(warmup):
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mean = statistics.fmean(times)
    return {"status": "ok", "mean_ms": round(mean, 2),
            "throughput_img_s": round(batch * 1000 / mean, 1), "batch": batch}


def data_agent(data_dir: str) -> Dict:
    samples = parse_manifest(data_dir)
    if not samples:
        return {"status": "error", "details": f"no valid samples in {data_dir}"}
    domains = Counter(s.domain for s in samples)
    lengths = Counter(len(s.label) for s in samples)
    splits = group_split(samples)
    groups = {k: {s.meter_id for s in v} for k, v in splits.items()}
    leak = (groups["train"] & groups["test"]) | (groups["train"] & groups["val"]) | (groups["val"] & groups["test"])
    return {"status": "ok" if not leak else "error", "n": len(samples),
            "domains": dict(domains), "length_dist": dict(sorted(lengths.items())),
            "split_counts": {k: len(v) for k, v in splits.items()},
            "meter_id_leakage": len(leak)}


def run_agents(variant: str = "tambour-b", data_dir: Optional[str] = None,
               device: Optional[str] = None) -> Dict:
    dev = resolve_device(device)
    print(f"\n  Tambour diagnostics | variant {variant} | device {dev}")
    m = model_agent(variant, dev)
    for c in m.get("checks", []):
        print(f"    [model]   {c['name']}: {c['status']} | {c['details']}")
    report = {"model": m.get("status")}
    if m.get("net") is not None:
        lat = latency_agent(m["net"], dev)
        print(f"    [latency] {lat['mean_ms']} ms/batch | {lat['throughput_img_s']} img/s (batch {lat['batch']})")
        report["latency"] = lat
    if data_dir:
        d = data_agent(data_dir)
        if d["status"] == "error":
            print(f"    [data]    error: {d.get('details', 'meter_id leakage in split!')}")
        else:
            print(f"    [data]    {d['n']} samples | domains {d['domains']} | "
                  f"splits {d['split_counts']} | leakage {d['meter_id_leakage']}")
        report["data"] = d
    return report
