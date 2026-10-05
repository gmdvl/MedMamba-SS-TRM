# -*- coding: utf-8 -*-
"""
training/samplers.py
=====================
Main Development Plan, Phase 12 - Test Classification Strategies (options
D and E).

  D. Balanced sampling - a weighted `DataLoader` sampler that visits every
     class roughly equally often per epoch (full balance, WITH
     replacement, dataset size unchanged - unlike Phase 2's
     `--balance_classes undersample/oversample` in
     `prepare_histologyhsi_bc_v4.py`, which physically rewrites
     X_train.npy/y_train.npy at prep time; this is a training-time-only
     alternative that needs no re-preprocessing).

  E. Moderate oversampling - a gentler alternative to (D): only
     under-represented classes are topped up (via repeated indices, TRAIN
     split only), up to a moderate target (default: the MEDIAN class
     count), while majority classes are left completely alone. This is
     deliberately less aggressive than full balancing - the plan's Phase 12
     lists it as a distinct option to compare against (D), not a synonym
     for it.

Per Phase 12, use ONE of these at a time (or neither) - `train_example_v8.py`
exposes them as a single `--sampler {none,balanced,moderate_oversample}`
choice for exactly this reason.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_balanced_sampler(y_train: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    y = np.asarray(y_train)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    class_weight = 1.0 / counts
    sample_weights = class_weight[y]
    return WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double),
                                  num_samples=len(y), replacement=True)


def build_moderate_oversample_indices(y_train: np.ndarray, num_classes: int, seed: int = 0,
                                       target_percentile: float = 50.0) -> np.ndarray:
    """Returns a shuffled array of TRAIN-split row indices (with repeats for
    classes below the target) suitable for `torch.utils.data.Subset`.
    `target_percentile=50` (median) is the "moderate" default; raise it
    (e.g. towards 100, the max class count) to move this option closer to
    full balancing, or lower it for a gentler top-up."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_train)
    counts = np.bincount(y, minlength=num_classes)
    present = counts[counts > 0]
    if len(present) == 0:
        return np.arange(len(y))
    target = int(np.percentile(present, target_percentile))

    parts = []
    for c in range(num_classes):
        idx = np.flatnonzero(y == c)
        if len(idx) == 0:
            continue
        if len(idx) < target:
            extra = rng.choice(idx, size=target - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        parts.append(idx)

    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices
