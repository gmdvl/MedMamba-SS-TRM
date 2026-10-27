# -*- coding: utf-8 -*-
"""
training/minority_augmentation.py
====================================
Minority-class augmentation: more, and more varied, training views of the
class the split has too few of. Off unless `train.py --minority_augment on`.

Why
---
On `data/hsi_v8-80_10_10_importance` DCIS is 5.0% of the training patches
(122,850 of 2,452,086), and they come from only 5 patients (107, 152, 25, 45,
85 - 5 captures and 24,570 patches each). Healthy and IDC come from 30+
patients. On PAD-UFES-20 the same shape holds for MEL (39 of the 1,626 training
images; 52 of 2,298 in the whole dataset).
`training/augmentation.py` treats every class the same, and
`--sampler moderate_oversample` repeats minority rows; neither adds variation
BETWEEN the few minority patients, which is what the class lacks.

What
----
Three pieces, train split only:

  1. Oversampling with distinct views - `build_minority_oversample_indices`
     repeats the minority rows until the class has `factor` x its count. The
     repeats sit at different positions of the `Subset`, and the per-item seed
     (`item_augmentation_seed`) is keyed on the position, so every copy gets
     its own augmentation draw and its own mixing partner.
  2. Intra-class mixing - with probability `mix_prob` a minority patch is mixed
     with another patch of the SAME class, drawn from a different patient when
     `groups_train.npy` exists:
       'stats' (default) the anchor keeps its spatial pattern; its per-band
               mean and std move part of the way to the partner's (MixStyle,
               Zhou et al., ICLR 2021, applied to the input bands). Inter-patient
               variation in these cubes is mostly per-band gain/offset.
       'pixel' a convex combination of the two patches (intra-class mixup).
     The anchor's weight is max(l, 1 - l), l ~ Beta(alpha, alpha), so the anchor
     always dominates. The label never changes: both patches carry it.
  3. Stronger augmentation - the base augmentation's magnitudes (noise, per-band
     scale and offset, crop) times `strength`, for minority samples only.
     1.0 = the augmentation every class already gets.

Majority-class samples go through exactly the path `AugmentedPatchDatasetV16`
takes - same seed, same draws, same output (pinned by
`tests/test_minority_augmentation.py`).

Mixing operates on the NORMALIZED patch. Under `--normalization global_zscore`
(every HSI profile) the `norm_stats` a reconstruction run carries are the same
for every patch, so the mixed patch still de-normalizes correctly. Under the
per-patch modes the anchor's stats would no longer describe the mixed patch;
`check_normalization` refuses that combination when reconstruction is on.

The frozen `training/augmentation.py` and `training/losses.py` are not edited.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from training.augmentation import AugmentationConfig, item_augmentation_seed  # frozen
from training.augmentation_v16 import AugmentedPatchDatasetV16

MIX_MODES = ("stats", "pixel")

# `--minority_classes auto`: a class with fewer training samples than this
# fraction of the median class count. Picks DCIS alone on the HSI split
# (122,850 < 0.5 x 722,358) and MEL alone on PAD-UFES-20 (39 < 0.5 x 165, the
# median of the train counts 508/613/39/172/136/158).
AUTO_FRACTION_OF_MEDIAN = 0.5

# AugmentationConfig magnitudes `strength` multiplies. Probabilities are left
# alone (the paper recipe already flips/rotates at 0.5), and so are the band
# dropout / spectral mask widths, which are off on every profile.
STRENGTH_FIELDS = ("spectral_noise_std", "spectral_scale_range", "spectral_offset_std",
                   "crop_scale_max_frac")
MAX_CROP_FRAC = 0.5

# Rejection draws for a partner from another patient. With 5 DCIS patients a
# draw lands on the anchor's own patient 1 time in 5, so 8 tries miss with
# probability 0.2^8 ~ 3e-6; after that the same-patient partner is kept.
PARTNER_TRIES = 8
_STD_EPS = 1e-6


# ============================================================================
# Which classes, and how many rows
# ============================================================================

def resolve_minority_classes(spec: Optional[str], y_train: np.ndarray, num_classes: int,
                             class_names: Optional[Sequence[str]] = None) -> List[int]:
    """`spec` is 'auto' or a comma list of class names (case-insensitive) or
    indices, e.g. 'DCIS' or '1' or 'MEL,SCC'."""
    counts = np.bincount(np.asarray(y_train), minlength=num_classes)
    if spec is None or str(spec).strip().lower() == "auto":
        present = counts[counts > 0]
        if len(present) == 0:
            return []
        cut = AUTO_FRACTION_OF_MEDIAN * float(np.median(present))
        return [c for c in range(num_classes) if 0 < counts[c] < cut]

    lowered = [str(n).lower() for n in (class_names or [])]
    out: List[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() in lowered:
            c = lowered.index(tok.lower())
        elif tok.lstrip("-").isdigit():
            c = int(tok)
        else:
            raise ValueError(f"unknown class {tok!r}; class names are {list(class_names or [])}")
        if not 0 <= c < num_classes:
            raise ValueError(f"class index {c} out of range for {num_classes} classes")
        if counts[c] == 0:
            raise ValueError(f"class {tok!r} has no training samples")
        if c not in out:
            out.append(c)
    return sorted(out)


def build_minority_oversample_indices(y_train: np.ndarray, minority_classes: Sequence[int],
                                      factor: float, seed: int = 0) -> np.ndarray:
    """Every training row once, then extra copies of the minority rows, so each
    minority class ends with round(factor x count) rows. Whole copies first; the
    fractional remainder is drawn without replacement. Factor 1.0 returns
    `np.arange(n)`."""
    if factor < 1.0:
        raise ValueError(f"oversample factor must be >= 1.0, got {factor}")
    y = np.asarray(y_train)
    rng = np.random.default_rng(seed)
    parts = [np.arange(len(y))]
    for c in minority_classes:
        idx = np.flatnonzero(y == c)
        if len(idx) == 0:
            continue
        extra = int(round(factor * len(idx))) - len(idx)
        whole, rest = divmod(extra, len(idx))
        parts.extend([idx] * whole)
        if rest:
            parts.append(np.sort(rng.choice(idx, size=rest, replace=False)))
    return np.concatenate(parts)


@dataclass
class MinorityPlan:
    classes: List[int]
    factor: float
    indices: Optional[np.ndarray]      # None when factor == 1.0 (no Subset needed)
    effective_labels: np.ndarray       # the label of every row the train loader can visit
    counts_before: List[int]
    counts_after: List[int]

    def summary(self, class_names: Sequence[str]) -> str:
        names = [class_names[c] for c in self.classes]
        per = ", ".join(f"{class_names[c]} {self.counts_before[c]}->{self.counts_after[c]}"
                        for c in self.classes)
        return f"[minority-aug] classes={names} factor={self.factor:g}: {per}"

    def as_dict(self, class_names: Sequence[str]) -> Dict:
        return {"classes": [class_names[c] for c in self.classes], "oversample_factor": self.factor,
                "counts_before": dict(zip(class_names, self.counts_before)),
                "counts_after": dict(zip(class_names, self.counts_after))}


def plan_minority_augmentation(y_train: np.ndarray, num_classes: int, class_names: Sequence[str],
                               spec: Optional[str], factor: float, seed: int = 0) -> MinorityPlan:
    y = np.asarray(y_train)
    classes = resolve_minority_classes(spec, y, num_classes, class_names)
    if not classes:
        raise ValueError(f"no class has fewer than {AUTO_FRACTION_OF_MEDIAN:g} x the median class "
                         f"count; name the class(es) with --minority_classes")
    indices = None if factor == 1.0 else build_minority_oversample_indices(y, classes, factor, seed)
    effective = y if indices is None else y[indices]
    return MinorityPlan(classes=classes, factor=float(factor), indices=indices, effective_labels=effective,
                        counts_before=np.bincount(y, minlength=num_classes).tolist(),
                        counts_after=np.bincount(effective, minlength=num_classes).tolist())


def check_normalization(normalization: str, recon_on: bool, mix_prob: float) -> None:
    if mix_prob > 0 and recon_on and normalization != "global_zscore":
        raise ValueError(f"mixing under --normalization {normalization} would leave the reconstruction "
                         f"target de-normalized with the anchor's per-patch stats; use global_zscore, "
                         f"--minority_mix_prob 0, or --recon_mode none")


# ============================================================================
# Transforms
# ============================================================================

def scaled_config(cfg: AugmentationConfig, strength: float) -> AugmentationConfig:
    """`cfg` with the `STRENGTH_FIELDS` magnitudes times `strength` (crop capped
    at `MAX_CROP_FRAC`)."""
    if strength < 0:
        raise ValueError(f"strength must be >= 0, got {strength}")
    changes = {f: float(getattr(cfg, f)) * strength for f in STRENGTH_FIELDS}
    changes["crop_scale_max_frac"] = min(changes["crop_scale_max_frac"], MAX_CROP_FRAC)
    return replace(cfg, **changes)


def mix_weight(alpha: float, rng: random.Random) -> float:
    """The anchor's share, in [0.5, 1]."""
    lam = rng.betavariate(alpha, alpha)
    return max(lam, 1.0 - lam)


def mix_pixel(anchor: torch.Tensor, partner: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * anchor + (1.0 - lam) * partner


def mix_stats(anchor: torch.Tensor, partner: torch.Tensor, lam: float) -> torch.Tensor:
    """Re-standardize `anchor` [C,H,W] to per-band statistics interpolated
    between its own and `partner`'s. lam = 1 returns the anchor."""
    mu_a = anchor.mean(dim=(-2, -1), keepdim=True)
    sd_a = anchor.std(dim=(-2, -1), keepdim=True, unbiased=False)
    mu_p = partner.mean(dim=(-2, -1), keepdim=True)
    sd_p = partner.std(dim=(-2, -1), keepdim=True, unbiased=False)
    mu = lam * mu_a + (1.0 - lam) * mu_p
    sd = lam * sd_a + (1.0 - lam) * sd_p
    return (anchor - mu_a) / sd_a.clamp_min(_STD_EPS) * sd + mu


MIX_FNS: Dict[str, Callable] = {"stats": mix_stats, "pixel": mix_pixel}


# ============================================================================
# Dataset
# ============================================================================

def root_indices(ds: Dataset) -> Tuple[Dataset, np.ndarray]:
    """Unwrap nested `Subset`s: the innermost dataset, and for every position of
    `ds` the row of that dataset it reads."""
    idx = None
    while isinstance(ds, Subset):
        cur = np.asarray(ds.indices)
        idx = cur if idx is None else cur[idx]
        ds = ds.dataset
    return ds, (np.arange(len(ds)) if idx is None else idx)


class MinorityAugmentedDataset(AugmentedPatchDatasetV16):
    """`AugmentedPatchDatasetV16` that sends minority-class samples through
    `minority_fn` (after optional intra-class mixing) instead of `augment_fn`.

    `labels` / `groups` are aligned with the ROOT dataset under any `Subset`
    wrappers (the `NpyDatasetV16` over X_train) - i.e. `y_train.npy` and
    `groups_train.npy` as they are on disk. Partners are read from that root
    dataset, so they carry the same normalization as the anchor.
    """

    def __init__(self, base_dataset: Dataset, augment_fn: Callable, minority_fn: Callable,
                 minority_classes: Sequence[int], labels: Optional[np.ndarray] = None,
                 groups: Optional[np.ndarray] = None, mix_mode: str = "stats", mix_prob: float = 0.0,
                 mix_alpha: float = 0.4, seed: Optional[int] = None):
        super().__init__(base_dataset, augment_fn, seed=seed)
        if mix_mode not in MIX_FNS:
            raise ValueError(f"mix_mode must be one of {MIX_MODES}, got {mix_mode!r}")
        if not 0.0 <= mix_prob <= 1.0:
            raise ValueError(f"mix_prob must be in [0, 1], got {mix_prob}")
        if mix_prob > 0 and not mix_alpha > 0:
            raise ValueError(f"mix_alpha must be > 0, got {mix_alpha}")
        self.minority_fn = minority_fn
        self.minority_classes = frozenset(int(c) for c in minority_classes)
        self.mix_mode, self.mix_prob, self.mix_alpha = mix_mode, float(mix_prob), float(mix_alpha)

        self._root, self._root_idx = root_indices(base_dataset)
        self._groups = None
        self._pool: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        if self.mix_prob > 0:
            if labels is None:
                raise ValueError("mixing needs `labels` (y_train aligned with the root dataset)")
            y = np.asarray(labels)
            if len(y) != len(self._root):
                raise ValueError(f"labels has {len(y)} rows, the root dataset {len(self._root)}")
            if groups is not None:
                self._groups = np.asarray(groups)
                if len(self._groups) != len(y):
                    raise ValueError(f"groups has {len(self._groups)} rows, labels {len(y)}")
            for c in sorted(self.minority_classes):
                idx = np.flatnonzero(y == c)
                if len(idx):
                    self._pool[c] = (idx, None if self._groups is None else self._groups[idx])

    def _partner_row(self, c: int, idx: int, rng: random.Random) -> int:
        rows, groups = self._pool[c]
        j = rng.randrange(len(rows))
        if groups is not None:
            own = self._groups[self._root_idx[idx]]
            for _ in range(PARTNER_TRIES - 1):
                if groups[j] != own:
                    break
                j = rng.randrange(len(rows))
        return int(rows[j])

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        patch, label = item[0], item[1]
        rest = item[2:]

        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        s = item_augmentation_seed(self.seed, self._current_epoch(), idx, worker_id)
        rng, gen = random.Random(s), torch.Generator().manual_seed(s)

        c = int(label)
        if c in self.minority_classes:
            if c in self._pool and rng.random() < self.mix_prob:
                partner = self._root[self._partner_row(c, idx, rng)][0]
                patch = MIX_FNS[self.mix_mode](patch, partner, mix_weight(self.mix_alpha, rng))
            augmented = self.minority_fn(patch, rng, gen)
        else:
            augmented = self.augment_fn(patch, rng, gen)

        if rest:
            return (augmented, label, *rest)
        return augmented, label

    def describe(self) -> Dict:
        pool = {int(c): int(len(rows)) for c, (rows, _g) in self._pool.items()}
        patients = ({int(c): int(len(np.unique(g))) for c, (_r, g) in self._pool.items()}
                    if self._groups is not None else None)
        return {"mix_mode": self.mix_mode, "mix_prob": self.mix_prob, "mix_alpha": self.mix_alpha,
                "cross_patient": self._groups is not None, "partner_pool_rows": pool,
                "partner_pool_patients": patients}
