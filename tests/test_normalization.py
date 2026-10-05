# -*- coding: utf-8 -*-
"""
test_normalization.py
======================
MedMamba-SS-TRM v16 plan, Stage 0.1 verification.

  1. denormalize(normalize(x)) == x, to float32 tolerance, for all three modes.
  2. per_sample_minmax / global_zscore are bit-identical to the reference
     implementation ported from train_example_v6.py:137-144.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from training.normalization import (
    NORMALIZATION_MODES, normalize_patch, denormalize, stack_norm_stats,
)
from training.npy_data import NpyDataset, NpyDatasetV16


def _random_patch(seed=0, H=11, W=11, C=8):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=500.0, scale=120.0, size=(H, W, C)).astype(np.float32)


def _reference_v6(patch, mode, global_mean=None, global_std=None):
    """Verbatim port of train_example_v6.py:137-144."""
    patch = patch.copy()
    if mode == "per_sample_minmax":
        p_min, p_max = patch.min(), patch.max()
        if p_max - p_min > 1e-6:
            patch = (patch - p_min) / (p_max - p_min)
        else:
            patch = np.zeros_like(patch)
    else:
        patch = (patch - global_mean) / global_std
    return patch


def test_round_trip_all_modes():
    patch = _random_patch()
    C = patch.shape[-1]
    global_mean = np.full((C,), 480.0, dtype=np.float32)
    global_std = np.full((C,), 100.0, dtype=np.float32)

    for mode in NORMALIZATION_MODES:
        normalized, offset_c, scale_c = normalize_patch(
            patch, mode, global_mean=global_mean, global_std=global_std)
        norm_stats = torch.from_numpy(stack_norm_stats(offset_c, scale_c)).unsqueeze(0)
        x = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        recon = denormalize(x, norm_stats)
        raw = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
        assert torch.allclose(recon, raw, atol=1e-4, rtol=1e-4), f"round trip failed for mode={mode}"


def test_degenerate_patch_round_trip():
    """A constant patch (the p_max - p_min <= 1e-6 branch) must still round-trip."""
    patch = np.full((5, 5, 4), 42.0, dtype=np.float32)
    for mode in NORMALIZATION_MODES:
        gm = np.full((4,), 42.0, dtype=np.float32)
        gs = np.full((4,), 1.0, dtype=np.float32)
        normalized, offset_c, scale_c = normalize_patch(patch, mode, global_mean=gm, global_std=gs)
        assert np.all(normalized == 0.0)
        norm_stats = torch.from_numpy(stack_norm_stats(offset_c, scale_c)).unsqueeze(0)
        x = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        recon = denormalize(x, norm_stats)
        raw = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
        assert torch.allclose(recon, raw, atol=1e-4), f"degenerate round trip failed for mode={mode}"


def test_legacy_modes_bit_identical_to_v6():
    patch = _random_patch(seed=1)
    C = patch.shape[-1]
    global_mean = np.full((C,), 480.0, dtype=np.float32)
    global_std = np.full((C,), 100.0, dtype=np.float32)

    for mode in ("per_sample_minmax", "global_zscore"):
        normalized, _, _ = normalize_patch(patch, mode, global_mean=global_mean, global_std=global_std)
        reference = _reference_v6(patch, mode, global_mean=global_mean, global_std=global_std)
        assert np.array_equal(normalized, reference.astype(np.float32)), \
            f"{mode} diverged from the train_example_v6.py reference"


def test_per_patch_zscore_matches_expected_stats():
    patch = _random_patch(seed=2)
    normalized, offset_c, scale_c = normalize_patch(patch, "per_patch_zscore")
    assert np.allclose(offset_c, patch.mean(), atol=1e-4)
    assert np.allclose(scale_c, patch.std(), atol=1e-4)
    assert abs(float(normalized.mean())) < 1e-3
    assert abs(float(normalized.std()) - 1.0) < 1e-3


def test_unknown_mode_raises():
    patch = _random_patch(seed=3)
    try:
        normalize_patch(patch, "bogus_mode")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown normalization mode")


def _write_fake_dataset(tmpdir, seed=7, n=6, H=5, W=5, C=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=500.0, scale=80.0, size=(n, H, W, C)).astype(np.float32)
    y = rng.integers(0, 3, size=(n,)).astype(np.int64)
    xp, yp = Path(tmpdir) / "X.npy", Path(tmpdir) / "y.npy"
    np.save(xp, X)
    np.save(yp, y)
    return str(xp), str(yp)


def test_npy_dataset_v16_bit_identical_to_parent_for_legacy_modes():
    with tempfile.TemporaryDirectory() as tmp:
        xp, yp = _write_fake_dataset(tmp)
        C = 4
        gm = np.full((C,), 480.0, dtype=np.float32)
        gs = np.full((C,), 90.0, dtype=np.float32)

        for mode in ("per_sample_minmax", "global_zscore"):
            parent = NpyDataset(xp, yp, normalization=mode, global_mean=gm, global_std=gs)
            child = NpyDatasetV16(xp, yp, normalization=mode, global_mean=gm, global_std=gs)
            assert len(parent) == len(child)
            for i in range(len(parent)):
                p_patch, p_label = parent[i]
                c_patch, c_label = child[i]
                assert p_label == c_label
                assert torch.equal(p_patch, c_patch), f"mode={mode} idx={i} diverged from NpyDataset"


def test_npy_dataset_v16_per_patch_zscore_and_norm_stats():
    with tempfile.TemporaryDirectory() as tmp:
        xp, yp = _write_fake_dataset(tmp)
        ds = NpyDatasetV16(xp, yp, normalization="per_patch_zscore", return_norm_stats=True)
        patch_t, label, norm_stats = ds[0]
        assert norm_stats.shape == (2, patch_t.shape[0])
        recon = denormalize(patch_t.unsqueeze(0), norm_stats.unsqueeze(0))[0]
        raw = torch.from_numpy(np.load(xp)[0]).permute(2, 0, 1)
        assert torch.allclose(recon, raw, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_round_trip_all_modes()
    test_degenerate_patch_round_trip()
    test_legacy_modes_bit_identical_to_v6()
    test_per_patch_zscore_matches_expected_stats()
    test_unknown_mode_raises()
    test_npy_dataset_v16_bit_identical_to_parent_for_legacy_modes()
    test_npy_dataset_v16_per_patch_zscore_and_norm_stats()
    print("OK")
