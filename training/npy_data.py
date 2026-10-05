# -*- coding: utf-8 -*-
"""
training/npy_data.py
=====================
The `.npy` data layer every training run uses, copied verbatim from
`archive/train_example_v6.py` when v6, v7 and v15 were archived (2026-09-29):

    compute_global_channel_stats   train-split per-channel mean/std   (v6 lines 60-75)
    NpyDataset                     the memory-mapped patch loader      (78-154)
    discover_data                  finds X_/y_{train,val,test}.npy     (157-184)
    setup_experiment_dir           creates the run directory           (187-204)

and, below them, the class runs actually construct:

    NpyDatasetV16                  `NpyDataset` + `per_patch_zscore` + optional `norm_stats`
                                   (was `training/npy_data.py`, merged here 2026-10-01)

The originals stay frozen and byte-identical in `archive/`, where the archived entry
points still import them. `tests/test_live_copies_match_archive.py` requires each copy's
source to match its original exactly. Change a copy on purpose and you drop it from that
test's list in the same commit.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.npy_integrity import load_npy_array, resolve_storage_mode
from training.normalization import normalize_patch, stack_norm_stats


def compute_global_channel_stats(x_path: str, sample_cap: int = 5000, seed: int = 0):
    """Phase 0.6 ablation axis: 'global_zscore' normalization needs
    per-channel mean/std fit ONCE on a bounded sample of the TRAINING set
    (never on val/test - that would leak test-set statistics into
    normalization) and then applied identically to every split."""
    X = np.load(x_path, mmap_mode="r")
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_cap, n), replace=False)
    idx.sort()
    sample = np.asarray(X[idx], dtype=np.float32)
    sample = np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0)
    mean = sample.mean(axis=(0, 1, 2))
    std = sample.std(axis=(0, 1, 2))
    std = np.where(std > 1e-6, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


class NpyDataset(Dataset):
    """Memory-mapped dataset loader.

    `normalization`:
      - "per_sample_minmax" (default, original behaviour): each patch is
        independently min-max scaled to [0,1] using ITS OWN min/max.
      - "global_zscore": each patch is standardized with per-channel
        mean/std fit once on a training-set sample (`global_mean`/
        `global_std`, see `compute_global_channel_stats`) and applied
        identically to every split - a Phase 0.6 ablation axis for the
        validation-loss investigation (per-sample min-max normalization
        means every patch is rescaled by ITS OWN range, which can distort
        relative intensities across patches in a way a fixed, dataset-level
        normalization does not).

    `augment` (train split only - NEVER set True for a validation/test
    Dataset; Phase 0.3's pipeline-consistency audit checks that the
    train/val Dataset *configuration* matches, so this flag being the one
    deliberate difference between them is by design, not an oversight):
      - "none" (default): no augmentation.
      - "flip": random horizontal/vertical flip, each independently with
        p=0.5. Flips are a safe augmentation for HSI/RGB tissue patches
        (no canonical up/down orientation), unlike e.g. color jitter, which
        would need per-band-physically-meaningful handling.
    """

    def __init__(self, x_path: str, y_path: str, normalization: str = "per_sample_minmax",
                 global_mean=None, global_std=None, augment: str = "none", seed: int = 0,
                 storage_mode: str = "mmap"):
        """`storage_mode` (v12/v13 SIGBUS-remediation plan, Phase 5):
        "mmap" (default, unchanged legacy behaviour) | "ram" | "auto".
        Callers that have already run `training.npy_integrity.validate_dataset`
        upstream (e.g. `train_example_v13.py`) can safely pass "ram"/"auto";
        this constructor itself does NOT re-validate - see
        `training/npy_integrity.py` for why validation must happen before
        any DataLoader worker exists, not lazily on first access.
        """
        self.storage_mode = resolve_storage_mode(x_path, storage_mode) if storage_mode == "auto" else storage_mode
        self.X = load_npy_array(x_path, self.storage_mode)
        self.y = load_npy_array(y_path, self.storage_mode)
        assert len(self.X) == len(self.y), f"X/y length mismatch: {len(self.X)} vs {len(self.y)}"
        assert normalization in ("per_sample_minmax", "global_zscore")
        self.normalization = normalization
        self.global_mean = None if global_mean is None else np.asarray(global_mean, dtype=np.float32)
        self.global_std = None if global_std is None else np.asarray(global_std, dtype=np.float32)
        if normalization == "global_zscore":
            assert self.global_mean is not None and self.global_std is not None, \
                "global_zscore normalization requires global_mean/global_std"
        assert augment in ("none", "flip")
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        patch = np.array(self.X[idx], dtype=np.float32, copy=True)
        patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

        if self.normalization == "per_sample_minmax":
            p_min, p_max = patch.min(), patch.max()
            if p_max - p_min > 1e-6:
                patch = (patch - p_min) / (p_max - p_min)
            else:
                patch = np.zeros_like(patch)
        else:  # global_zscore
            patch = (patch - self.global_mean) / self.global_std

        if self.augment == "flip":
            if self._rng.random() < 0.5:
                patch = np.ascontiguousarray(patch[::-1, :, :])
            if self._rng.random() < 0.5:
                patch = np.ascontiguousarray(patch[:, ::-1, :])

        patch = torch.from_numpy(patch).permute(2, 0, 1).contiguous()
        label = int(self.y[idx])
        return patch, label


def discover_data(data_dir: str):
    base = Path(data_dir)
    x_train, y_train = base / "X_train.npy", base / "y_train.npy"
    x_test, y_test = base / "X_test.npy", base / "y_test.npy"
    wl_path = base / "wavelengths.npy"
    for p in [x_train, y_train, x_test, y_test]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    x_val, y_val = base / "X_val.npy", base / "y_val.npy"
    if x_val.exists() and y_val.exists():
        val_x_path, val_y_path = x_val, y_val
    else:
        val_x_path, val_y_path = x_test, y_test

    y_tr = np.load(y_train, mmap_mode="r")
    y_va = np.load(val_y_path, mmap_mode="r")
    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    wavelengths = np.load(wl_path).astype("float32") if wl_path.exists() else None

    class_names = None
    class_names_path = base / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path) as f:
            class_names = json.load(f)

    return (str(x_train), str(y_train), str(val_x_path), str(val_y_path),
            wavelengths, num_classes, class_names, len(y_tr), len(y_va))


def setup_experiment_dir(base_dir="experiments", exp_name="medmamba_ss_trm_run", resume_path=None,
                          append_timestamp=True):
    """`append_timestamp=True` (default, unchanged behaviour for every caller
    that predates this argument): the directory is `{exp_name}_{YYYYmmdd_HHMMSS}`.

    `append_timestamp=False`: `exp_name` is used verbatim. This exists for
    `train_example_v14.py`, whose naming scheme puts the timestamp FIRST so
    `ls experiments/` still sorts chronologically - see
    `training/run_naming.build_run_name`, which has already embedded it.
    """
    if resume_path:
        return Path(resume_path).resolve().parent.parent
    if append_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{exp_name}_{timestamp}"
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# ---------------------------------------------------------------------------------------
# NpyDatasetV16 (MedMamba-SS-TRM v16 plan, Stage 0.2). Lived in `training/npy_data.py`
# until 2026-10-01; the class is unchanged. It subclasses the verbatim `NpyDataset` above
# instead of editing it, because the original is frozen in `archive/train_example_v6.py`.
#
# Adds:
#   - a third normalization mode, `per_patch_zscore` (A-2), via
#     `training/normalization.py` (Stage 0.1);
#   - an optional third `__getitem__` return value, `norm_stats` ([2, C] float32:
#     `(offset_c, scale_c)`, `raw == normalized * scale + offset`), so the trainer can
#     report reconstruction metrics in reflectance units (R-3/R-7) regardless of which
#     normalization mode a run used.
#
# `__getitem__` is a FULL override (does not call `super().__getitem__`) because the
# parent normalizes and converts to a tensor in one pass - there is no way to recover
# `norm_stats` from its return value. The operations are the same, with the math in
# `training/normalization.normalize_patch`, so the two legacy modes stay bit-identical
# (asserted in `tests/test_normalization.py`).
# ---------------------------------------------------------------------------------------

class NpyDatasetV16(NpyDataset):
    """Adds `per_patch_zscore` and the optional third return value.
    Everything else - storage_mode, mmap, the length assert, the flip
    augmentation - is inherited unchanged from `NpyDataset.__init__`."""

    def __init__(self, *a, normalization: str = "per_sample_minmax",
                 return_norm_stats: bool = False, **kw):
        # The parent asserts a 2-mode whitelist (train_example_v6.py:119),
        # so construct it with a legal placeholder mode and set the real one
        # afterwards - `__init__` never actually normalizes anything itself.
        super().__init__(*a, normalization=("per_sample_minmax"
                                             if normalization == "per_patch_zscore"
                                             else normalization), **kw)
        if normalization not in ("per_sample_minmax", "global_zscore", "per_patch_zscore"):
            raise ValueError(f"unknown normalization mode {normalization!r}")
        self.normalization = normalization
        self.return_norm_stats = bool(return_norm_stats)

    def __getitem__(self, idx):
        patch = np.array(self.X[idx], dtype=np.float32, copy=True)
        patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

        normalized, offset_c, scale_c = normalize_patch(
            patch, self.normalization, global_mean=self.global_mean, global_std=self.global_std)

        if self.augment == "flip":
            if self._rng.random() < 0.5:
                normalized = np.ascontiguousarray(normalized[::-1, :, :])
            if self._rng.random() < 0.5:
                normalized = np.ascontiguousarray(normalized[:, ::-1, :])

        patch_t = torch.from_numpy(normalized).permute(2, 0, 1).contiguous()
        label = int(self.y[idx])

        if not self.return_norm_stats:
            return patch_t, label
        norm_stats = torch.from_numpy(stack_norm_stats(offset_c, scale_c))
        return patch_t, label, norm_stats
