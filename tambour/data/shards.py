"""WebDataset-style tar shards for million-image scale.

Loose small files murder throughput at scale (random reads, inode pressure). We
pack samples into sequential ``.tar`` shards. Writer/reader use only the stdlib
(``tarfile``) so there is no hard dependency; if ``webdataset`` is installed you
can point its loader at the same shards.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import cv2
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .manifest import MeterSample


def write_shards(images_dir, samples: Sequence[MeterSample], out_dir,
                 shard_size: int = 10000, resize_h: int = 0) -> List[str]:
    """Pack (image, label, domain) into tar shards. Returns shard paths.

    Optionally pre-resizes to a fixed height (recommended at scale: decode once,
    store small JPEGs, keep aspect ratio).
    """
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: List[str] = []
    tar = None
    for i, s in enumerate(samples):
        if i % shard_size == 0:
            if tar is not None:
                tar.close()
            path = out_dir / f"shard-{i // shard_size:06d}.tar"
            shards.append(str(path))
            tar = tarfile.open(path, "w")
        img = cv2.imread(str(images_dir / s.fname), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if resize_h and img.shape[0] != resize_h:
            scale = resize_h / img.shape[0]
            img = cv2.resize(img, (max(1, round(img.shape[1] * scale)), resize_h))
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            continue
        key = f"{i:09d}"
        for ext, data in ((".jpg", buf.tobytes()),
                          (".txt", s.label.encode()),
                          (".dom", s.domain.encode())):
            info = tarfile.TarInfo(key + ext)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    if tar is not None:
        tar.close()
    return shards


class ShardDataset(IterableDataset):
    """Stream samples from tar shards sequentially; shards are split across workers."""

    def __init__(self, shard_paths: Sequence[str], transform, codec, domain_to_id):
        self.shard_paths = list(shard_paths)
        self.transform = transform
        self.codec = codec
        self.domain_to_id = domain_to_id

    def _my_shards(self) -> List[str]:
        info = get_worker_info()
        if info is None:
            return self.shard_paths
        return self.shard_paths[info.id::info.num_workers]

    def __iter__(self) -> Iterator[Tuple]:
        for path in self._my_shards():
            with tarfile.open(path, "r") as tar:
                pending: dict = {}
                for member in tar:
                    key, ext = member.name.rsplit(".", 1)
                    pending.setdefault(key, {})[ext] = tar.extractfile(member).read()
                    rec = pending[key]
                    if {"jpg", "txt", "dom"} <= rec.keys():
                        img = cv2.imdecode(np.frombuffer(rec["jpg"], np.uint8), cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        label = rec["txt"].decode()
                        domain = rec["dom"].decode()
                        import torch
                        yield (self.transform(image=img)["image"],
                               torch.tensor(self.codec.encode(label), dtype=torch.long),
                               len(label), label, self.domain_to_id.get(domain, 0))
                        del pending[key]
