# -*- coding: utf-8 -*-
"""
test_split_drift_gate.py
==========================
MedMamba-SS-TRM v16 plan, Stage 1.2 verification (gate G8).

  - G8 fails a synthetic split shifted 2.8 sigma (patient 68's shape).
  - G8 passes a matched split.
  - per_patch_zscore trivially passes drift by construction (the fix).
  - `enforce_split_drift` raises on "abort", only warns on "warn", does
    nothing on "off".
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from training.gates import check_split_drift, enforce_split_drift


def _write_split(tmp, name, n, H, W, C, mean, std, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=mean, scale=std, size=(n, H, W, C)).astype(np.float32)
    path = Path(tmp) / f"X_{name}.npy"
    np.save(path, X)
    return str(path)


def test_g8_fails_a_shifted_split():
    with tempfile.TemporaryDirectory() as tmp:
        C = 6
        gm = np.full((C,), 500.0, dtype=np.float32)
        gs = np.full((C,), 100.0, dtype=np.float32)
        x_paths = {
            "train": _write_split(tmp, "train", 4000, 11, 11, C, mean=500.0, std=100.0, seed=0),
            # shifted 2.8 sigma below train's mean, in raw units - global_zscore
            # will carry that shift straight through, same as patient 68.
            "validation": _write_split(tmp, "val", 1000, 11, 11, C, mean=500.0 - 2.8 * 100.0, std=100.0, seed=1),
        }
        report = check_split_drift(x_paths, "global_zscore", global_mean=gm, global_std=gs,
                                    n_patches=2000, seed=0)
        assert report["passed"] is False
        assert report["splits"]["validation"]["passed"] is False
        assert abs(report["splits"]["validation"]["mean_sigma_vs_train"] - (-2.8)) < 0.15


def test_g8_passes_a_matched_split():
    with tempfile.TemporaryDirectory() as tmp:
        C = 6
        gm = np.full((C,), 500.0, dtype=np.float32)
        gs = np.full((C,), 100.0, dtype=np.float32)
        x_paths = {
            "train": _write_split(tmp, "train", 4000, 11, 11, C, mean=500.0, std=100.0, seed=0),
            "validation": _write_split(tmp, "val", 1000, 11, 11, C, mean=500.0, std=100.0, seed=2),
        }
        report = check_split_drift(x_paths, "global_zscore", global_mean=gm, global_std=gs,
                                    n_patches=2000, seed=0)
        assert report["passed"] is True
        assert report["splits"]["validation"]["passed"] is True


def test_g8_per_patch_zscore_absorbs_the_shift():
    with tempfile.TemporaryDirectory() as tmp:
        C = 6
        x_paths = {
            "train": _write_split(tmp, "train", 4000, 11, 11, C, mean=500.0, std=100.0, seed=0),
            "validation": _write_split(tmp, "val", 1000, 11, 11, C, mean=500.0 - 2.8 * 100.0, std=100.0, seed=1),
        }
        report = check_split_drift(x_paths, "per_patch_zscore", n_patches=2000, seed=0)
        assert report["passed"] is True


def test_enforce_split_drift_modes():
    failing_report = {"passed": False, "splits": {"validation": {"passed": False, "reason": "x"}}}
    passing_report = {"passed": True, "splits": {}}

    with pytest.raises(SystemExit):
        enforce_split_drift(failing_report, "abort")
    enforce_split_drift(failing_report, "warn")   # must not raise
    enforce_split_drift(failing_report, "off")    # must not raise
    enforce_split_drift(passing_report, "abort")  # must not raise - nothing failed


def test_g8_writes_report_when_exp_dir_given():
    with tempfile.TemporaryDirectory() as tmp:
        C = 4
        x_paths = {
            "train": _write_split(tmp, "train", 500, 5, 5, C, mean=0.0, std=1.0, seed=0),
        }
        report = check_split_drift(x_paths, "per_patch_zscore", n_patches=200, seed=0, exp_dir=tmp)
        assert (Path(tmp) / "split_drift_report.json").is_file()
        assert report["passed"] is True


if __name__ == "__main__":
    test_g8_fails_a_shifted_split()
    test_g8_passes_a_matched_split()
    test_g8_per_patch_zscore_absorbs_the_shift()
    test_enforce_split_drift_modes()
    test_g8_writes_report_when_exp_dir_given()
    print("OK")
