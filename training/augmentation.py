# -*- coding: utf-8 -*-
"""
training/augmentation.py
==========================
Main Development Plan, Phase 8 - Train-Only Augmentation, and Phase 9 -
Add HSI-Specific Augmentation.

Before this file, no augmentation code existed anywhere in this repo. Every
transform here is a pure function of a single already-loaded patch tensor
`[C,H,W]` (matching what `NpyDataset.__getitem__` returns), so
`AugmentedPatchDataset` can wrap ANY existing Dataset (a plain `NpyDataset`,
a `Subset` from Stage B's moderate oversampling, ...) and apply them.

Phase 8 is a hard requirement: wrap ONLY the training dataset. Never wrap
validation/test - `train_example_v11.py` enforces this by only ever passing
`AugmentedPatchDataset` to the train `DataLoader`.

v15 R6.3 - FIXED: the multi-worker RNG duplication this file used to
document as "a minor diversity loss, not a correctness bug".

The old design had `AugmentationConfig.build()` close over one shared
`random.Random(seed)` instance. `num_workers > 0` forks the dataset object per
worker, so every worker inherited that instance at the same state and drew the
same flip/rot90 sequence. The suggested remedy - a `worker_init_fn` reseeding
`torch`/`random` per worker - could not work: `make_worker_init_fn` reseeds the
MODULE-LEVEL `random`, and a private `random.Random` instance does not read
from it. Both v15-evidence runs used `--augment_preset custom` with
`flip_h/flip_v/rotate90 = 0.5`, all three driven by that instance, so all four
workers applied the same spatial augmentation to every batch they produced.

Now every draw - `random` and `torch` alike - is a deterministic function of
`(seed, epoch, sample index, worker id)`, derived per item in
`AugmentedPatchDataset.__getitem__`. See `item_augmentation_seed`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============================================================================
# Spectral augmentations (Phase 9)
# ============================================================================

def spectral_noise(patch: torch.Tensor, std: float, rng: random.Random,
                    gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Small additive Gaussian perturbation to every spectral value."""
    if std <= 0:
        return patch
    noise = torch.empty_like(patch).normal_(generator=gen) if gen is not None else torch.randn_like(patch)
    return patch + noise * std


def spectral_scale(patch: torch.Tensor, scale_range: float, rng: random.Random,
                    gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Small multiplicative change, one factor per band, shared across the
    patch's spatial extent - simulates per-wavelength sensor gain drift."""
    if scale_range <= 0:
        return patch
    C = patch.shape[0]
    factors = 1.0 + (torch.rand(C, 1, 1, generator=gen) * 2 - 1) * scale_range
    return patch * factors


def spectral_offset(patch: torch.Tensor, offset_std: float, rng: random.Random,
                     gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Small additive change, one offset per band."""
    if offset_std <= 0:
        return patch
    C = patch.shape[0]
    return patch + torch.randn(C, 1, 1, generator=gen) * offset_std


def band_dropout(patch: torch.Tensor, prob: float, max_frac: float, rng: random.Random,
                  gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Zeroes a small number of individual (scattered, not necessarily
    contiguous) spectral bands."""
    if prob <= 0 or rng.random() > prob:
        return patch
    C = patch.shape[0]
    n_drop = max(1, int(round(C * max_frac * rng.random())))
    idx = torch.randperm(C, generator=gen)[:n_drop]
    patch = patch.clone()
    patch[idx, :, :] = 0.0
    return patch


def spectral_masking(patch: torch.Tensor, prob: float, max_width_frac: float, rng: random.Random,
                      gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Masks ONE contiguous wavelength region (unlike `band_dropout`'s
    scattered bands) - simulates a dead detector segment or a strong,
    spectrally-localized artifact."""
    if prob <= 0 or rng.random() > prob:
        return patch
    C = patch.shape[0]
    width = max(1, int(round(C * max_width_frac * rng.random())))
    start = rng.randint(0, max(0, C - width))
    patch = patch.clone()
    patch[start:start + width, :, :] = 0.0
    return patch


# ============================================================================
# Spatial augmentations (Phase 9)
# ============================================================================

def spatial_flip(patch: torch.Tensor, p_h: float, p_v: float, rng: random.Random,
                  gen: Optional[torch.Generator] = None) -> torch.Tensor:
    if p_h > 0 and rng.random() < p_h:
        patch = torch.flip(patch, dims=[-1])
    if p_v > 0 and rng.random() < p_v:
        patch = torch.flip(patch, dims=[-2])
    return patch


def spatial_rotate90(patch: torch.Tensor, prob: float, rng: random.Random,
                      gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """90-degree-increment rotation only - arbitrary-angle rotation needs
    interpolation and can introduce out-of-range values at patch corners,
    which is a bigger risk for small HSI patches than the rotation is worth."""
    if prob <= 0 or rng.random() > prob:
        return patch
    k = rng.choice([1, 2, 3])
    return torch.rot90(patch, k=k, dims=[-2, -1])


def spatial_crop_scale(patch: torch.Tensor, max_crop_frac: float, rng: random.Random,
                        gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Crops a random sub-region (up to `max_crop_frac` smaller per spatial
    dimension) and resizes back to the original patch size (nearest-
    neighbour, dependency-free) - a "limited scaling" augmentation."""
    if max_crop_frac <= 0:
        return patch
    C, H, W = patch.shape
    crop_h = max(1, int(round(H * (1 - rng.random() * max_crop_frac))))
    crop_w = max(1, int(round(W * (1 - rng.random() * max_crop_frac))))
    if crop_h >= H and crop_w >= W:
        return patch
    top = rng.randint(0, H - crop_h)
    left = rng.randint(0, W - crop_w)
    cropped = patch[:, top:top + crop_h, left:left + crop_w]
    return F.interpolate(cropped.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)


# ============================================================================
# Config + composition
# ============================================================================

@dataclass
class AugmentationConfig:
    """Every probability/magnitude defaults to 0/off - augmentation is
    entirely opt-in, per Phase 9's own caution: "Augmentations must
    preserve the underlying pathology/class semantics." Tune conservatively
    and verify held-out validation metrics don't regress before trusting a
    non-default setting."""
    spectral_noise_std: float = 0.0
    spectral_scale_range: float = 0.0
    spectral_offset_std: float = 0.0
    band_dropout_prob: float = 0.0
    band_dropout_max_frac: float = 0.1
    spectral_mask_prob: float = 0.0
    spectral_mask_max_width_frac: float = 0.1
    flip_h_prob: float = 0.0
    flip_v_prob: float = 0.0
    rotate90_prob: float = 0.0
    crop_scale_max_frac: float = 0.0
    seed: int = 0

    def build(self) -> "Compose":
        """v15 R6.3 - the steps now take their RNGs as ARGUMENTS instead of
        closing over one shared `random.Random(self.seed)` instance. See the
        `Compose`/`AugmentedPatchDataset` docstrings for why that mattered."""
        steps: List[Callable] = [
            lambda x, rng, gen: spectral_noise(x, self.spectral_noise_std, rng, gen),
            lambda x, rng, gen: spectral_scale(x, self.spectral_scale_range, rng, gen),
            lambda x, rng, gen: spectral_offset(x, self.spectral_offset_std, rng, gen),
            lambda x, rng, gen: band_dropout(x, self.band_dropout_prob,
                                             self.band_dropout_max_frac, rng, gen),
            lambda x, rng, gen: spectral_masking(x, self.spectral_mask_prob,
                                                 self.spectral_mask_max_width_frac, rng, gen),
            lambda x, rng, gen: spatial_flip(x, self.flip_h_prob, self.flip_v_prob, rng, gen),
            lambda x, rng, gen: spatial_rotate90(x, self.rotate90_prob, rng, gen),
            lambda x, rng, gen: spatial_crop_scale(x, self.crop_scale_max_frac, rng, gen),
        ]
        return Compose(steps, seed=self.seed)


PRESETS: Dict[str, AugmentationConfig] = {
    "none": AugmentationConfig(),
    "light": AugmentationConfig(
        spectral_noise_std=0.01, band_dropout_prob=0.1, band_dropout_max_frac=0.05,
        flip_h_prob=0.5, flip_v_prob=0.5,
    ),
    "medium": AugmentationConfig(
        spectral_noise_std=0.02, spectral_scale_range=0.05, spectral_offset_std=0.01,
        band_dropout_prob=0.2, band_dropout_max_frac=0.08,
        spectral_mask_prob=0.1, spectral_mask_max_width_frac=0.1,
        flip_h_prob=0.5, flip_v_prob=0.5, rotate90_prob=0.5, crop_scale_max_frac=0.1,
    ),
}


class Compose:
    """v15 R6.3 - `__call__` takes the RNGs, it does not own them.

    Before: `AugmentationConfig.build()` closed over ONE `random.Random(seed)`
    instance and every step drew from it. With `num_workers > 0` PyTorch forks
    the dataset object per worker, so all N workers inherited that instance at
    the SAME state and produced the same flip/rot90 sequence.
    `training/dataloader_config.make_worker_init_fn` reseeds the MODULE-LEVEL
    `random`, which cannot reach a private instance, so the documented
    "fix it with a worker_init_fn" advice did not work either. Both evidence
    runs used `--augment_preset custom` with `flip_h/flip_v/rotate90 = 0.5`,
    all three driven by that instance.

    `rng`/`gen` default to a `Compose`-owned pair seeded from the config, so
    calling `compose(patch)` still works for anything that used the old
    single-argument form.
    """

    def __init__(self, steps: Sequence[Callable], seed: int = 0):
        self.steps = list(steps)
        self.seed = seed
        self._fallback_rng = random.Random(seed)
        self._fallback_gen = torch.Generator().manual_seed(int(seed))

    def __call__(self, patch: torch.Tensor, rng: Optional[random.Random] = None,
                  gen: Optional[torch.Generator] = None) -> torch.Tensor:
        rng = self._fallback_rng if rng is None else rng
        gen = self._fallback_gen if gen is None else gen
        for step in self.steps:
            patch = step(patch, rng, gen)
        return patch


def item_augmentation_seed(base_seed: int, epoch: int, index: int, worker_id: int = 0) -> int:
    """A deterministic per-(item, epoch, worker) seed.

    Mixed rather than added so that neighbouring indices, adjacent epochs and
    consecutive worker ids cannot collide onto the same stream - `base_seed +
    worker_id`, the scheme `make_worker_init_fn` used, makes worker 1 of a
    seed-42 run draw exactly what worker 0 of a seed-43 run draws.
    """
    h = (int(base_seed) & 0xFFFFFFFF) * 0x9E3779B1
    h ^= (int(epoch) & 0xFFFF) * 0x85EBCA6B
    h ^= (int(worker_id) & 0xFF) * 0xC2B2AE35
    h ^= (int(index) & 0xFFFFFFFF) * 0x27D4EB2F
    return h & 0x7FFFFFFF


class AugmentedPatchDataset(Dataset):
    """Wraps a dataset returning `(patch [C,H,W], label)` and applies
    `augment_fn` to the patch on every `__getitem__` call. Phase 8: use ONLY
    for the training split.

    v15 R6.3 - every draw is now a deterministic function of
    `(seed, epoch, index, worker_id)`, so:
      * different workers produce different augmentations (they did not);
      * the same index gets a different augmentation in a different epoch
        (call `set_epoch`);
      * the whole augmentation stream is reproducible from `--seed` alone,
        independently of how many workers the DataLoader happens to use.

    `set_epoch(epoch)` must be called on the MAIN-process dataset object
    before each epoch. With `persistent_workers=False` (this repo's default in
    every loader mode except `performance`) the workers are re-forked each
    epoch and pick the new value up; with persistent workers they do not, and
    the epoch component stays at whatever it was when the workers started.
    """

    def __init__(self, base_dataset: Dataset, augment_fn: Callable, seed: Optional[int] = None):
        self.base_dataset = base_dataset
        self.augment_fn = augment_fn
        # `seed=None` (the default) inherits the seed the `Compose` was built
        # with, so the callers that predate this argument -
        # `train_example_v11/v12/v13/v14.py`, which all pass
        # `AugmentationConfig(..., seed=args.seed).build()` - keep deriving
        # their augmentation stream from `--seed` rather than silently
        # dropping to 0.
        self.seed = int(seed if seed is not None else getattr(augment_fn, "seed", 0))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        patch, label = self.base_dataset[idx]
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        s = item_augmentation_seed(self.seed, self.epoch, idx, worker_id)
        return self.augment_fn(patch, random.Random(s), torch.Generator().manual_seed(s)), label
