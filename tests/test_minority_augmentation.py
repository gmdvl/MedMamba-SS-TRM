# -*- coding: utf-8 -*-
"""
test_minority_augmentation.py
================================
`training/minority_augmentation.py` and its `train.py --minority_augment` wiring.

The property everything else rests on: a MAJORITY-class sample comes out of
`MinorityAugmentedDataset` bit-identical to what `AugmentedPatchDatasetV16`
returns for it, and with `--minority_augment off` (the default) the run name,
and therefore the pipeline, is the one the parity tests pin.

Run with: pytest tests/test_minority_augmentation.py -q
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from training.augmentation import AugmentationConfig
from training.augmentation_v16 import AugmentedPatchDatasetV16, build_active
from training.minority_augmentation import (
    MAX_CROP_FRAC, MinorityAugmentedDataset, build_minority_oversample_indices, check_normalization,
    mix_pixel, mix_stats, mix_weight, plan_minority_augmentation, resolve_minority_classes,
    root_indices, scaled_config,
)
from training.train_cli import resolve
from training.train_pipeline import build_run_name

REPO_ROOT = Path(__file__).resolve().parent.parent

# the measured train-split label counts of the two datasets
HSI_Y = np.repeat([0, 1, 2], [722358, 122850, 1606878])
PAD_Y = np.repeat(np.arange(6), [508, 613, 39, 172, 136, 158])


# ============================================================================
# class resolution / oversampling
# ============================================================================

def test_auto_picks_dcis_on_hsi_and_mel_on_pad():
    assert resolve_minority_classes("auto", HSI_Y, 3) == [1]
    assert resolve_minority_classes("auto", PAD_Y, 6) == [2]


def test_explicit_classes_by_name_or_index():
    names = ["healthy", "DCIS", "IDC"]
    assert resolve_minority_classes("dcis", HSI_Y, 3, names) == [1]
    assert resolve_minority_classes("1", HSI_Y, 3, names) == [1]
    assert resolve_minority_classes("IDC, DCIS,1", HSI_Y, 3, names) == [1, 2]
    for bad in ("tumour", "3", "-1"):
        with pytest.raises(ValueError):
            resolve_minority_classes(bad, HSI_Y, 3, names)
    with pytest.raises(ValueError, match="no training samples"):
        resolve_minority_classes("2", np.array([0, 0, 1]), 3, names)


@pytest.mark.parametrize("factor", [1.0, 2.0, 2.5, 3.0])
def test_oversampling_hits_the_factor_and_keeps_every_row_once(factor):
    y = np.repeat([0, 1, 2], [40, 7, 60])
    idx = build_minority_oversample_indices(y, [1], factor, seed=0)
    np.testing.assert_array_equal(idx[:len(y)], np.arange(len(y)))
    counts = np.bincount(y[idx], minlength=3)
    assert counts.tolist() == [40, int(round(factor * 7)), 60]
    with pytest.raises(ValueError):
        build_minority_oversample_indices(y, [1], 0.5)


def test_plan_reports_effective_counts_and_refuses_an_empty_minority():
    y = np.repeat([0, 1, 2], [40, 7, 60])
    plan = plan_minority_augmentation(y, 3, ["a", "b", "c"], "auto", 2.0)
    assert plan.classes == [1]
    assert plan.counts_after == [40, 14, 60]
    assert np.bincount(plan.effective_labels).tolist() == plan.counts_after
    assert plan_minority_augmentation(y, 3, ["a", "b", "c"], "b", 1.0).indices is None
    with pytest.raises(ValueError, match="--minority_classes"):
        plan_minority_augmentation(np.repeat([0, 1], [10, 10]), 2, ["a", "b"], "auto", 2.0)


def test_mixing_with_per_patch_normalization_and_reconstruction_is_refused():
    check_normalization("global_zscore", True, 0.5)
    check_normalization("per_patch_zscore", False, 0.5)
    check_normalization("per_patch_zscore", True, 0.0)
    with pytest.raises(ValueError):
        check_normalization("per_patch_zscore", True, 0.5)


# ============================================================================
# transforms
# ============================================================================

def test_strength_scales_magnitudes_only_and_caps_the_crop():
    cfg = AugmentationConfig(spectral_noise_std=0.02, spectral_scale_range=0.15, spectral_offset_std=0.05,
                             crop_scale_max_frac=0.3, flip_h_prob=0.5, rotate90_prob=0.5, seed=7)
    s = scaled_config(cfg, 2.0)
    assert (s.spectral_noise_std, s.spectral_scale_range, s.spectral_offset_std) == (0.04, 0.3, 0.1)
    assert s.crop_scale_max_frac == MAX_CROP_FRAC
    assert (s.flip_h_prob, s.rotate90_prob, s.seed) == (0.5, 0.5, 7)
    assert scaled_config(cfg, 1.0) == cfg


def test_mix_stats_moves_band_statistics_and_keeps_the_spatial_pattern():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(4, 5, 5, generator=g) * 2 + 3
    p = torch.randn(4, 5, 5, generator=g) * 0.5 - 1
    torch.testing.assert_close(mix_stats(a, p, 1.0), a)
    out = mix_stats(a, p, 0.7)
    dims = (-2, -1)
    torch.testing.assert_close(out.mean(dims), 0.7 * a.mean(dims) + 0.3 * p.mean(dims))

    def z(t):
        return (t - t.mean(dims, keepdim=True)) / t.std(dims, keepdim=True, unbiased=False)
    torch.testing.assert_close(z(out), z(a))


def test_mix_pixel_and_mix_weight():
    a, p = torch.ones(2, 3, 3), torch.zeros(2, 3, 3)
    torch.testing.assert_close(mix_pixel(a, p, 0.8), torch.full((2, 3, 3), 0.8))
    rng = random.Random(0)
    lams = [mix_weight(0.4, rng) for _ in range(500)]
    assert all(0.5 <= lam <= 1.0 for lam in lams)


def test_root_indices_composes_nested_subsets():
    base = _Rows(np.zeros(10, dtype=int))
    inner = Subset(base, [9, 8, 7, 6, 5])
    outer = Subset(inner, np.array([4, 0, 0]))
    root, idx = root_indices(outer)
    assert root is base
    assert idx.tolist() == [5, 9, 9]
    assert root_indices(base)[1].tolist() == list(range(10))


# ============================================================================
# the dataset
# ============================================================================

class _Rows(Dataset):
    """Patch [2,3,3] whose every value is the row index; records reads."""

    def __init__(self, labels, with_stats=False):
        self.labels = np.asarray(labels)
        self.with_stats = with_stats
        self.reads = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        self.reads.append(int(i))
        patch = torch.full((2, 3, 3), float(i)) + torch.arange(9.0).reshape(1, 3, 3) * 0.01
        if self.with_stats:
            return patch, int(self.labels[i]), torch.tensor([float(i)])
        return patch, int(self.labels[i])


def _noise(x, rng, gen):
    return x + torch.randn(x.shape, generator=gen) + rng.random()


def _minority_marker(x, rng, gen):
    return x + 1000.0


LABELS = np.repeat([0, 1, 2], [6, 4, 6])
GROUPS = np.array([0, 0, 1, 1, 2, 2,  10, 11, 12, 13,  3, 3, 4, 4, 5, 5])


def test_majority_samples_are_bit_identical_to_the_v16_wrapper():
    ref = AugmentedPatchDatasetV16(_Rows(LABELS), _noise, seed=3)
    ds = MinorityAugmentedDataset(_Rows(LABELS), _noise, _minority_marker, [1], labels=LABELS,
                                  groups=GROUPS, mix_prob=1.0, seed=3)
    for epoch in (0, 2):
        ref.set_epoch(epoch)
        ds.set_epoch(epoch)
        for i in np.flatnonzero(LABELS != 1):
            assert torch.equal(ds[i][0], ref[i][0])


def test_minority_samples_take_the_minority_path_and_a_cross_patient_partner():
    base = _Rows(LABELS)
    ds = MinorityAugmentedDataset(base, _noise, _minority_marker, [1], labels=LABELS, groups=GROUPS,
                                  mix_mode="pixel", mix_prob=1.0, seed=0)
    for i in np.flatnonzero(LABELS == 1):
        base.reads.clear()
        x, y = ds[i]
        assert y == 1 and float(x.min()) > 999
        anchor, partner = base.reads
        assert anchor == i and LABELS[partner] == 1 and GROUPS[partner] != GROUPS[i]


def test_without_groups_the_partner_is_any_same_class_row():
    ds = MinorityAugmentedDataset(_Rows(LABELS), _noise, _minority_marker, [1], labels=LABELS,
                                  mix_prob=1.0, seed=0)
    for s in range(50):
        assert LABELS[ds._partner_row(1, 6, random.Random(s))] == 1


def test_no_mixing_reads_no_partner():
    base = _Rows(LABELS)
    ds = MinorityAugmentedDataset(base, _noise, _minority_marker, [1], mix_prob=0.0, seed=0)
    base.reads.clear()
    ds[7]
    assert base.reads == [7]


def test_each_oversampled_copy_gets_its_own_view_and_epochs_differ():
    root = _Rows(LABELS)
    idx = build_minority_oversample_indices(LABELS, [1], 3.0, seed=0)
    ds = MinorityAugmentedDataset(Subset(root, idx), _noise, _noise, [1], labels=LABELS, groups=GROUPS,
                                  mix_prob=0.5, seed=0)
    copies = [p for p in range(len(idx)) if idx[p] == 6]
    assert len(copies) == 3
    views = [ds[p][0] for p in copies]
    assert not torch.equal(views[0], views[1]) and not torch.equal(views[1], views[2])
    assert torch.equal(ds[copies[0]][0], views[0])          # deterministic
    ds.set_epoch(1)
    assert not torch.equal(ds[copies[0]][0], views[0])


def test_norm_stats_pass_through_untouched():
    ds = MinorityAugmentedDataset(_Rows(LABELS, with_stats=True), _noise, _minority_marker, [1],
                                  labels=LABELS, groups=GROUPS, mix_prob=1.0, seed=0)
    x, y, stats = ds[6]
    assert y == 1 and stats.tolist() == [6.0]


def test_constructor_validates():
    with pytest.raises(ValueError):
        MinorityAugmentedDataset(_Rows(LABELS), _noise, _noise, [1], labels=LABELS, mix_mode="cutmix")
    with pytest.raises(ValueError):
        MinorityAugmentedDataset(_Rows(LABELS), _noise, _noise, [1], labels=LABELS, mix_prob=1.5)
    with pytest.raises(ValueError):
        MinorityAugmentedDataset(_Rows(LABELS), _noise, _noise, [1], mix_prob=0.5)          # no labels
    with pytest.raises(ValueError):
        MinorityAugmentedDataset(_Rows(LABELS), _noise, _noise, [1], labels=LABELS[:-1], mix_prob=0.5)


def test_describe_counts_partner_patients():
    ds = MinorityAugmentedDataset(_Rows(LABELS), _noise, _noise, [1], labels=LABELS, groups=GROUPS,
                                  mix_prob=0.5, seed=0)
    d = ds.describe()
    assert d["partner_pool_rows"] == {1: 4} and d["partner_pool_patients"] == {1: 4}
    assert d["cross_patient"] is True


# ============================================================================
# CLI + run name
# ============================================================================

HSI = "data/hsi_v8-80_10_10_importance-new/hsi"


def test_off_by_default_and_the_run_name_only_changes_when_on():
    off = resolve(["--data_dir", HSI, "--profile", "paper_recipe"])[0].args
    assert off.minority_augment == "off"
    on = resolve(["--data_dir", HSI, "--profile", "paper_recipe", "--minority_augment", "on"])[0].args
    ts = "20261022_000000"
    assert build_run_name(on, ts, modality="x") == build_run_name(off, ts, modality="x") + "_minaug"
    tagged = resolve(["--data_dir", HSI, "--profile", "paper_recipe", "--minority_augment", "on",
                      "--run_tag", "t1"])[0].args
    assert build_run_name(tagged, ts, modality="x").endswith("_minaug_t1")


# ============================================================================
# end to end
# ============================================================================

TINY = ["--epochs", "1", "--batch_size", "16", "--trm_dim", "16", "--trm_n_latent", "1",
        "--trm_n_improve", "1", "--trm_deep_supervision_steps", "1", "--compile", "off",
        "--loader_mode", "safe", "--drift_check_patches", "64", "--min_logit_std", "1e-4"]


@pytest.fixture(scope="module")
def tiny_imbalanced_hsi(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tiny_minority") / "hsi"
    out.mkdir()
    rng = np.random.default_rng(0)
    counts = {"train": [40, 8, 56], "val": [16, 16, 16], "test": [16, 16, 16]}
    for split, per in counts.items():
        y = np.repeat(np.arange(3), per).astype(np.int64)
        x = rng.random((len(y), 8, 8, 8), dtype=np.float32) * 0.5 + (y[:, None, None, None] * 0.15)
        np.save(out / f"X_{split}.npy", x.astype(np.float32))
        np.save(out / f"y_{split}.npy", y)
        np.save(out / f"groups_{split}.npy", (np.arange(len(y)) % 4 + 10 * y).astype(np.int32))
    np.save(out / "wavelengths.npy", np.linspace(450, 900, 8).astype(np.float32))
    (out / "class_names.json").write_text(json.dumps(["healthy", "DCIS", "IDC"]))
    return out


def test_train_py_runs_with_minority_augmentation(tiny_imbalanced_hsi, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    argv = ["--data_dir", str(tiny_imbalanced_hsi), "--profile", "paper_recipe", "--minority_augment", "on",
            "--minority_oversample_factor", "3"] + TINY
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "train.py")] + argv, cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert "[minority-aug] classes=['DCIS'] factor=3: DCIS 8->24" in proc.stdout

    (run_dir,) = list((tmp_path / "experiments").iterdir())
    assert run_dir.name.endswith("_minaug")
    cfg = json.loads((run_dir / "config.json").read_text())
    m = cfg["minority_augmentation"]
    assert m["classes"] == ["DCIS"] and m["counts_after"] == {"healthy": 40, "DCIS": 24, "IDC": 56}
    assert m["cross_patient"] is True and m["partner_pool_patients"] == {"1": 4}
    assert cfg["train_subsample"]["total"] == 40 + 24 + 56
