"""Domain-balanced sampling so the largest domain can't dominate training."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def make_domain_balanced_sampler(domain_ids: Sequence[int], temperature: float = 0.5
                                 ) -> WeightedRandomSampler:
    """Sample domains with probability ~ N_domain ** temperature.

    temperature=1 -> proportional (no rebalancing); 0 -> uniform across domains;
    ~0.3-0.5 is the usual sweet spot. A sample's weight is the chosen per-domain
    probability divided by the domain's size, i.e. N_domain ** (temperature - 1).
    """
    counts = Counter(int(d) for d in domain_ids)
    weight_per_domain = {d: n ** (temperature - 1.0) for d, n in counts.items()}
    weights = torch.tensor([weight_per_domain[int(d)] for d in domain_ids], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
