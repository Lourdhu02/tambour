"""Data layer: manifest, transforms, dataset, domain-balanced sampling, shards."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..text import CTCCodec
from .dataset import MeterDataset, collate_fn
from .manifest import (MeterSample, domain_vocab, group_split, infer_domain,
                       meter_id_of, parse_manifest)
from .sampler import make_domain_balanced_sampler
from .transforms import build_transforms, fit_pad, synthetic_rolling

__all__ = [
    "MeterSample", "parse_manifest", "domain_vocab", "group_split", "infer_domain",
    "meter_id_of", "MeterDataset", "collate_fn", "build_transforms", "fit_pad",
    "synthetic_rolling", "make_domain_balanced_sampler", "build_loaders",
]


def build_loaders(data_dir: str, cfg: Dict, codec: Optional[CTCCodec] = None,
                  rank: int = 0, world_size: int = 1
                  ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    """Build train/val/test loaders with group-split + optional domain balancing.

    Under DDP (``world_size > 1``) the non-balanced path uses a DistributedSampler;
    the domain-balanced path replicates the weighted sampler per rank (each rank
    draws independently with replacement — fine at million-image scale).
    """
    codec = codec or CTCCodec()
    samples = parse_manifest(data_dir)
    if not samples:
        raise FileNotFoundError(
            f"No valid samples in {data_dir} (need images/ + labels.txt with existing images).")

    dom2id = domain_vocab(samples)
    splits = group_split(samples, tuple(cfg.get("split", (0.9, 0.05, 0.05))), cfg.get("seed", 42),
                         by_group=cfg.get("group_split", True))
    images_dir = Path(data_dir) / "images"
    train_tf = build_transforms(cfg["img_h"], cfg["img_w"], True,
                                cfg.get("aug_level", "analog"), cfg.get("rolling_aug", 0.15))
    eval_tf = build_transforms(cfg["img_h"], cfg["img_w"], False)

    workers = min(cfg.get("workers", 4), os.cpu_count() or 1)
    kw = dict(num_workers=workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    if workers > 0:
        kw["persistent_workers"] = True

    def make(split, tf):
        return MeterDataset(images_dir, splits[split], tf, codec, dom2id)

    train_ds = make("train", train_tf)
    if cfg.get("domain_balanced", True) and len(dom2id) > 1:
        domids = [dom2id[s.domain] for s in splits["train"]]
        sampler = make_domain_balanced_sampler(domids, cfg.get("domain_temperature", 0.5))
        train_loader = DataLoader(train_ds, batch_size=cfg["batch"], sampler=sampler, drop_last=True, **kw)
    elif world_size > 1:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch"], sampler=sampler, drop_last=True, **kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, drop_last=True, **kw)

    eb = cfg["batch"] * 2
    val_loader = DataLoader(make("val", eval_tf), batch_size=eb, shuffle=False, **kw)
    test_loader = (DataLoader(make("test", eval_tf), batch_size=eb, shuffle=False, **kw)
                   if splits["test"] else None)

    meta = dict(domain_to_id=dom2id, counts={k: len(v) for k, v in splits.items()})
    return train_loader, val_loader, test_loader, meta
