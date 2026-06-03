"""Parse the dataset manifest and build leakage-safe, domain-stratified splits."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..text import is_valid_label


@dataclass(frozen=True)
class MeterSample:
    fname: str
    label: str
    domain: str
    meter_id: str


def infer_domain(fname: str) -> str:
    """Heuristic domain from the filename when no explicit domain column exists."""
    up = fname.upper()
    for tag in ("RRF", "MRC", "GAS", "ELEC", "WATER"):
        if tag in up:
            return tag.lower()
    return "default"


def meter_id_of(fname: str) -> str:
    """All photos of one physical meter share this id (text before the timestamp)."""
    return fname.split("-", 1)[0]


def parse_manifest(data_dir: str) -> List[MeterSample]:
    """Read ``labels.txt``: ``fname<TAB>label[<TAB>domain]`` per line.

    Domain falls back to a filename heuristic when the 3rd column is absent.
    """
    data_path = Path(data_dir)
    labels_file = data_path / "labels.txt"
    images_dir = data_path / "images"
    out: List[MeterSample] = []
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split("\t") if "\t" in raw else raw.split()
            if len(parts) < 2:
                continue
            fname, label = parts[0].strip(), parts[1].strip()
            domain = parts[2].strip() if len(parts) >= 3 else infer_domain(fname)
            if not is_valid_label(label):
                continue
            if not (images_dir / fname).exists():
                continue
            out.append(MeterSample(fname, label, domain, meter_id_of(fname)))
    return out


def domain_vocab(samples: Sequence[MeterSample]) -> Dict[str, int]:
    return {d: i for i, d in enumerate(sorted({s.domain for s in samples}))}


def group_split(
    samples: Sequence[MeterSample],
    ratios: Tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42,
    by_group: bool = True,
    stratify_domain: bool = True,
) -> Dict[str, List[MeterSample]]:
    """Split into train/val/test.

    ``by_group`` keeps every photo of a ``meter_id`` in one split (no leakage from
    repeat shots). ``stratify_domain`` guarantees each split covers every domain.
    """
    rng = random.Random(seed)
    buckets: Dict[str, List[MeterSample]] = defaultdict(list)
    for s in samples:
        buckets[s.domain if stratify_domain else "_all_"].append(s)

    splits: Dict[str, List[MeterSample]] = {"train": [], "val": [], "test": []}
    for items in buckets.values():
        if by_group:
            groups: Dict[str, List[MeterSample]] = defaultdict(list)
            for s in items:
                groups[s.meter_id].append(s)
            units = list(groups.values())
        else:
            units = [[s] for s in items]
        rng.shuffle(units)
        n = len(items)
        n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
        c_train = c_val = 0
        for u in units:
            if c_train < n_train:
                splits["train"].extend(u); c_train += len(u)
            elif c_val < n_val:
                splits["val"].extend(u); c_val += len(u)
            else:
                splits["test"].extend(u)
    return splits
