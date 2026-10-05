# -*- coding: utf-8 -*-
"""
training/augmentation_v16.py
===============================
MedMamba-SS-TRM v16 plan, Stage 3.4 (E-4). `training/augmentation.py` is imported
at `train_example_v15.py:111` and is therefore FROZEN.

`AugmentedPatchDataset.set_epoch` (`augmentation.py:270-274`) sets a plain
Python attribute on the MAIN-process dataset object. That is invisible to
DataLoader worker processes once they exist: with `persistent_workers=False`
(most loader modes) workers are re-forked every epoch and pick the new value
up by inheriting the parent's memory at fork time, but the documented run
command (`documentations/10_commands_v15.md:113`) uses `--loader_mode
performance`, which sets `persistent_workers=True`
(`training/dataloader_config.py:46`) - so workers are never re-forked, the
epoch value they hold is frozen at whatever it was when they started, and the
per-(seed, epoch, index, worker) augmentation stream (R6.3) has never actually
varied across epochs in any run that used `performance` mode.

`AugmentedPatchDatasetV16` fixes this the standard way: the epoch counter
lives in a `torch.Tensor` allocated with `.share_memory_()` BEFORE the
DataLoader forks its workers, so a write from the main process (`set_epoch`)
is visible to every worker process without re-forking. Falls back to a plain
attribute (logged) when shared-memory allocation is unavailable (e.g. some
sandboxed/CI environments), so this degrades to the frozen module's own
(known) limitation rather than crashing.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from training.augmentation import AugmentedPatchDataset, item_augmentation_seed  # frozen

_logger = logging.getLogger("AugmentedPatchDatasetV16")


class AugmentedPatchDatasetV16(AugmentedPatchDataset):
    """`set_epoch` that survives `persistent_workers=True`, and passes a
    3-tuple `(patch, label, norm_stats)` through when the wrapped dataset
    returns one (`training.npy_data.NpyDatasetV16`,
    `return_norm_stats=True`) - `norm_stats` describes the UN-augmented raw
    patch's offset/scale and must not be touched by `augment_fn`."""

    def __init__(self, base_dataset: Dataset, augment_fn: Callable, seed: Optional[int] = None):
        super().__init__(base_dataset, augment_fn, seed=seed)
        self._shared_epoch_ok = True
        try:
            self._epoch_shared = torch.zeros(1, dtype=torch.int64).share_memory_()
        except (RuntimeError, OSError) as e:
            self._shared_epoch_ok = False
            _logger.warning(f"[E-4] shared-memory epoch counter unavailable ({e}); falling back to "
                             f"the plain-attribute epoch (same limitation as the frozen "
                             f"AugmentedPatchDataset under persistent_workers=True).")

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)  # keeps self.epoch as a fallback/introspection value
        if self._shared_epoch_ok:
            self._epoch_shared[0] = int(epoch)

    def _current_epoch(self) -> int:
        if self._shared_epoch_ok:
            return int(self._epoch_shared[0].item())
        return self.epoch

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        patch, label = item[0], item[1]
        rest = item[2:]

        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        s = item_augmentation_seed(self.seed, self._current_epoch(), idx, worker_id)
        augmented = self.augment_fn(patch, random.Random(s), torch.Generator().manual_seed(s))

        if rest:
            return (augmented, label, *rest)
        return augmented, label


# ---------------------------------------------------------------------------
# v17 S8 - drop the steps that are switched off
# ---------------------------------------------------------------------------
# `AugmentationConfig.build()` (frozen, training/augmentation.py:179-191) always returns
# all EIGHT steps, and each one re-checks its own "am I off" guard on every sample. Under
# OPTIMAL_DEFAULTS, `band_dropout_prob` and `spectral_mask_prob` are both 0.0 - deliberately,
# because zeroing a whole channel of a 3-channel RGB image is not an augmentation - so two
# lambda calls per sample do nothing, forever, on every run.
#
# This is bit-identical, and the reason is worth stating because it is the whole
# justification: every guard in the frozen module short-circuits BEFORE it touches `rng`.
#
#     spectral_noise / spectral_scale / spectral_offset / spatial_crop_scale
#         `if <magnitude> <= 0: return patch`                      - no draw
#     band_dropout / spectral_masking / spatial_rotate90
#         `if prob <= 0 or rng.random() > prob:`                   - `or` short-circuits
#     spatial_flip
#         `if p_h > 0 and rng.random() < p_h:`                     - `and` short-circuits
#
# So a disabled step consumes ZERO random draws, and removing it cannot shift the stream
# the remaining steps see. Had any of them drawn first and discarded after, this would
# silently change every augmented sample and would not be worth doing.
_STEP_MAGNITUDES = (
    ("spectral_noise_std",),          # 0
    ("spectral_scale_range",),        # 1
    ("spectral_offset_std",),         # 2
    ("band_dropout_prob",),           # 3
    ("spectral_mask_prob",),          # 4
    ("flip_h_prob", "flip_v_prob"),   # 5 - one step, two magnitudes
    ("rotate90_prob",),               # 6
    ("crop_scale_max_frac",),         # 7
)


def build_active(cfg):
    """`cfg.build()` with the steps whose magnitudes are all <= 0 removed.

    Falls back to the unfiltered `Compose` if the frozen `build()` ever stops returning
    exactly `len(_STEP_MAGNITUDES)` steps in this order - a silently mis-aligned filter
    would drop the wrong augmentation, which is far worse than not filtering at all.
    """
    composed = cfg.build()
    steps = list(getattr(composed, "steps", []))
    if len(steps) != len(_STEP_MAGNITUDES):
        _logger.warning(
            f"[v17 S8] AugmentationConfig.build() returned {len(steps)} steps, expected "
            f"{len(_STEP_MAGNITUDES)}; keeping all of them rather than guessing which is which.")
        return composed
    keep = [s for s, names in zip(steps, _STEP_MAGNITUDES)
            if any(float(getattr(cfg, n, 0.0) or 0.0) > 0.0 for n in names)]
    composed.steps = keep
    return composed
