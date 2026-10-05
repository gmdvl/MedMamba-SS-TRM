# -*- coding: utf-8 -*-
"""
training/normalization.py
==========================
MedMamba-SS-TRM v16 plan, Stage 0.1 - single home for every patch normalization
mode AND its inverse.

Three modes:

  per_sample_minmax  - offset = patch.min(), scale = patch.max() - patch.min()
                        (scalars, broadcast to per-channel [C] so the
                        collated batch has a uniform [B, 2, C] stats shape)
  global_zscore      - offset = global_mean [C], scale = global_std [C],
                        fit on the TRAIN split only and passed in.
  per_patch_zscore    - offset = patch.mean(), scale = patch.std(), taken over
                        the WHOLE patch (all H x W x C), not per-channel. The
                        v16 audit's fix for A-1 (patient 68's dark IDC
                        captures): measured 0.07 sigma on normal validation
                        IDC vs 0.13 sigma for global_zscore, and 0.35 sigma on
                        dark vs 2.79 sigma for global_zscore - it wins on both
                        columns because it re-centers/re-scales EVERY patch to
                        its own exposure instead of applying one fixed,
                        train-fit affine map to every patch regardless of how
                        it was acquired.

`per_sample_minmax` and `global_zscore` are ported verbatim from
`train_example_v6.py:137-144` so the two audited (v15/v11) runs stay
reproducible bit-for-bit if this module is ever used to re-derive their
inputs; `test_normalization.py` asserts the two are bit-identical to that
reference implementation.

`offset`/`scale` are always returned as per-channel float32 arrays of shape
`[C]` (broadcasting a scalar mode's single value across channels), so
whatever mode produced them, `raw == normalized * scale + offset` and the
collated batch's `norm_stats` tensor always has shape `[B, 2, C]` - this is
what lets `TrainerG_v12` compute reconstruction metrics in reflectance units
(R-3/R-7) without knowing which mode a given run used.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

NORMALIZATION_MODES = ("per_sample_minmax", "global_zscore", "per_patch_zscore")

# Below this range, per_sample_minmax/per_patch_zscore treat the patch as
# constant (matches train_example_v6.py:139's `1e-6` guard).
_MINMAX_EPS = 1e-6
_ZSCORE_EPS = 1e-6


def normalize_patch(
    patch_hwc: np.ndarray,
    mode: str,
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`patch_hwc`: float32 array, shape [H, W, C], already `nan_to_num`'d.

    Returns `(normalized_hwc, offset_c, scale_c)` with
    `raw == normalized * scale + offset` and `offset`/`scale` always
    per-channel float32 arrays of shape `[C]`.
    """
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"unknown normalization mode {mode!r}; must be one of {NORMALIZATION_MODES}")

    C = patch_hwc.shape[-1]

    if mode == "per_sample_minmax":
        p_min, p_max = patch_hwc.min(), patch_hwc.max()
        scale = p_max - p_min
        if scale > _MINMAX_EPS:
            normalized = (patch_hwc - p_min) / scale
        else:
            normalized = np.zeros_like(patch_hwc)
            scale = 1.0  # so denormalize(0) still recovers p_min, not a divide-by-zero
        offset_c = np.full((C,), p_min, dtype=np.float32)
        scale_c = np.full((C,), scale, dtype=np.float32)
        return normalized.astype(np.float32, copy=False), offset_c, scale_c

    if mode == "global_zscore":
        if global_mean is None or global_std is None:
            raise ValueError("global_zscore normalization requires global_mean/global_std")
        gm = np.asarray(global_mean, dtype=np.float32)
        gs = np.asarray(global_std, dtype=np.float32)
        normalized = (patch_hwc - gm) / gs
        return normalized.astype(np.float32, copy=False), gm.copy(), gs.copy()

    # per_patch_zscore
    p_mean = float(patch_hwc.mean())
    p_std = float(patch_hwc.std())
    scale = p_std if p_std > _ZSCORE_EPS else 1.0
    normalized = (patch_hwc - p_mean) / scale
    offset_c = np.full((C,), p_mean, dtype=np.float32)
    scale_c = np.full((C,), scale, dtype=np.float32)
    return normalized.astype(np.float32, copy=False), offset_c, scale_c


def denormalize(x_bchw: torch.Tensor, norm_stats_b2c: torch.Tensor) -> torch.Tensor:
    """`x_bchw`: [B, C, H, W] normalized tensor. `norm_stats_b2c`: [B, 2, C]
    with `norm_stats_b2c[:, 0, :]` = offset, `norm_stats_b2c[:, 1, :]` = scale
    (the collated `(offset_c, scale_c)` pair `normalize_patch` returns).

    Returns the reflectance/raw-domain tensor: `x * scale + offset`,
    broadcast per-channel.
    """
    if norm_stats_b2c.ndim != 3 or norm_stats_b2c.shape[1] != 2:
        raise ValueError(f"norm_stats_b2c must have shape [B, 2, C], got {tuple(norm_stats_b2c.shape)}")
    offset = norm_stats_b2c[:, 0, :].to(dtype=x_bchw.dtype, device=x_bchw.device)
    scale = norm_stats_b2c[:, 1, :].to(dtype=x_bchw.dtype, device=x_bchw.device)
    offset = offset.view(offset.shape[0], offset.shape[1], 1, 1)
    scale = scale.view(scale.shape[0], scale.shape[1], 1, 1)
    return x_bchw * scale + offset


def stack_norm_stats(offset_c: np.ndarray, scale_c: np.ndarray) -> np.ndarray:
    """`(offset_c, scale_c)`, each [C], -> a single [2, C] array - the
    per-item shape that collates into the `[B, 2, C]` `denormalize` expects."""
    return np.stack([offset_c, scale_c], axis=0).astype(np.float32, copy=False)
