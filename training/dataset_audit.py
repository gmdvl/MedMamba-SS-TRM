# -*- coding: utf-8 -*-
"""
training/dataset_audit.py
==========================
Phase 0.1 of improve-promp_v8.plan.md - "Complete Dataset Pipeline Audit".

This module is the thing that actually produces `dataset_diagnostics.json`.
It is deliberately independent of `training.dataset_integrity` (Phase 7,
patient/slide leakage) - `run_dataset_audit()` *calls into* that module for
the leakage section, but adds everything else the Phase 0.1 checklist asks
for that Phase 7 doesn't cover:

  - per-split dataset statistics (count, shape, dtype)
  - per-split normalization / spectral statistics (mean, std, min, max,
    per-band mean/std) - used to check train vs. val are on the same scale
  - class distribution per split
  - patient / slide distribution per split (via the manifest, if given)
  - duplicate detection: exact byte-identical patches (within AND across
    splits - training.dataset_integrity only reports cross-split)
  - mirrored-duplicate detection (patch equals another patch after a
    horizontal and/or vertical flip)
  - rotated-duplicate detection (patch equals another patch after a 90/180/
    270 degree rotation)
  - corruption report: NaN count, Inf count, zero-valued spectra (a patch
    whose every pixel/band is exactly 0 - usually a sign of a bad ROI mask
    or a calibration divide-by-zero clipped to 0)

Everything here reads `.npy` files via `mmap_mode="r"` and only ever
materializes small slices at a time, so it stays usable on datasets much
larger than RAM. All of the "duplicate" checks are bounded by
`max_patches_per_split` for the same reason (see `check_patch_leakage` in
`training.dataset_integrity`, which this module mirrors in spirit).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------
# Per-split numeric statistics
# ----------------------------------------------------------------------

def _spectral_stats(X: np.ndarray, sample_cap: int = 20_000, seed: int = 0) -> Dict:
    """Per-band mean/std/min/max plus overall mean/std/min/max, computed on
    a bounded random sample (without replacement) of patches so this stays
    cheap on huge datasets. `X` is `[N,H,W,C]`, memory-mapped."""
    n = len(X)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_cap, n), replace=False)
    idx.sort()  # sequential-ish access is friendlier to the mmap page cache
    sample = np.asarray(X[idx], dtype=np.float32)  # [n_s,H,W,C]

    finite = np.isfinite(sample)
    safe = np.where(finite, sample, 0.0)

    per_band_mean = safe.mean(axis=(0, 1, 2)).tolist()
    per_band_std = safe.std(axis=(0, 1, 2)).tolist()
    per_band_min = np.where(finite, sample, np.inf).min(axis=(0, 1, 2))
    per_band_max = np.where(finite, sample, -np.inf).max(axis=(0, 1, 2))
    per_band_min = np.where(np.isfinite(per_band_min), per_band_min, np.nan).tolist()
    per_band_max = np.where(np.isfinite(per_band_max), per_band_max, np.nan).tolist()

    return {
        "n_sampled": int(len(idx)),
        "overall_mean": float(safe.mean()),
        "overall_std": float(safe.std()),
        "overall_min": float(np.nanmin(per_band_min)) if per_band_min else None,
        "overall_max": float(np.nanmax(per_band_max)) if per_band_max else None,
        "per_band_mean": per_band_mean,
        "per_band_std": per_band_std,
        "per_band_min": per_band_min,
        "per_band_max": per_band_max,
    }


def _corruption_report(X: np.ndarray, sample_cap: int = 20_000, seed: int = 0) -> Dict:
    """NaN / Inf / all-zero-spectrum counts on a bounded sample."""
    n = len(X)
    if n == 0:
        return {"n_sampled": 0, "n_nan_patches": 0, "n_inf_patches": 0, "n_zero_patches": 0}
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_cap, n), replace=False)
    idx.sort()
    sample = np.asarray(X[idx], dtype=np.float32)

    nan_mask = np.isnan(sample).any(axis=(1, 2, 3))
    inf_mask = np.isinf(sample).any(axis=(1, 2, 3))
    zero_mask = (sample == 0).all(axis=(1, 2, 3))

    return {
        "n_sampled": int(len(idx)),
        "n_nan_patches": int(nan_mask.sum()),
        "n_inf_patches": int(inf_mask.sum()),
        "n_zero_patches": int(zero_mask.sum()),
        "nan_patch_fraction": float(nan_mask.mean()),
        "inf_patch_fraction": float(inf_mask.mean()),
        "zero_patch_fraction": float(zero_mask.mean()),
    }


def _class_distribution(y: np.ndarray, class_names: Optional[List[str]] = None) -> Dict:
    y = np.asarray(y)
    if len(y) == 0:
        return {}
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).tolist()
    names = class_names if class_names and len(class_names) == n_classes else [str(i) for i in range(n_classes)]
    return {"n_classes": n_classes, "class_names": names, "counts": counts,
            "fractions": [c / max(1, len(y)) for c in counts]}


# ----------------------------------------------------------------------
# Duplicate / mirrored / rotated detection
# ----------------------------------------------------------------------

def _hash_bytes(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _canonical_transform_hashes(patch: np.ndarray) -> List[str]:
    """All 8 dihedral-group transforms (identity, 2 mirrors, 3 rotations,
    and their mirror-of-rotation compositions) of a single [H,W,C] patch.
    Two patches sharing ANY of these hashes are geometric duplicates of
    each other (exact, not near-duplicate - deliberately conservative)."""
    variants = [
        patch,
        np.flip(patch, axis=0),                    # vertical mirror
        np.flip(patch, axis=1),                    # horizontal mirror
        np.rot90(patch, k=1, axes=(0, 1)),
        np.rot90(patch, k=2, axes=(0, 1)),          # 180 deg
        np.rot90(patch, k=3, axes=(0, 1)),
        np.flip(np.rot90(patch, k=1, axes=(0, 1)), axis=0),
        np.flip(np.rot90(patch, k=1, axes=(0, 1)), axis=1),
    ]
    return [_hash_bytes(v) for v in variants]


def _duplicate_and_geometric_scan(X: np.ndarray, max_patches: int = 20_000,
                                   seed: int = 0) -> Dict:
    """Single pass over a bounded, randomly-sampled subset that reports:
      - exact within-sample duplicate groups
      - geometric duplicates (patch A equals some flip/rotation of patch B)
    Only within the sampled subset - not exhaustive over the whole split -
    to keep this tractable; `max_patches` can be raised for a more thorough
    (slower) pass."""
    n = len(X)
    if n == 0:
        return {"n_sampled": 0, "exact_duplicate_groups": 0, "exact_duplicate_patches": 0,
                "geometric_duplicate_pairs": 0}
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_patches, n), replace=False)
    idx.sort()

    exact_hash_to_indices: Dict[str, List[int]] = {}
    canon_hash_to_indices: Dict[str, List[int]] = {}  # min over the 8 transform hashes -> "canonical" identity
    identity_hash_of: Dict[int, str] = {}

    for i in idx:
        patch = np.asarray(X[int(i)])
        exact_h = _hash_bytes(patch)
        exact_hash_to_indices.setdefault(exact_h, []).append(int(i))
        identity_hash_of[int(i)] = exact_h

        transform_hashes = _canonical_transform_hashes(patch)
        canonical = min(transform_hashes)  # any consistent choice works as a group key
        canon_hash_to_indices.setdefault(canonical, []).append(int(i))

    exact_dup_groups = {h: v for h, v in exact_hash_to_indices.items() if len(v) > 1}
    n_exact_dup_patches = sum(len(v) for v in exact_dup_groups.values())

    # geometric duplicates = same canonical group, but NOT already an exact
    # duplicate of each other (otherwise this is double-counting Phase 0.1's
    # plain "duplicate patches" bullet).
    geometric_pairs = 0
    geometric_examples = []
    for canonical, members in canon_hash_to_indices.items():
        if len(members) < 2:
            continue
        # partition members by their own exact hash; if >1 distinct exact
        # hash share a canonical group, those pairs are mirror/rotation
        # duplicates of each other rather than byte-identical duplicates.
        by_exact: Dict[str, List[int]] = {}
        for m in members:
            by_exact.setdefault(identity_hash_of[m], []).append(m)
        if len(by_exact) > 1:
            groups = list(by_exact.values())
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    geometric_pairs += len(groups[gi]) * len(groups[gj])
            if len(geometric_examples) < 10:
                geometric_examples.append([g[0] for g in groups])

    return {
        "n_sampled": int(len(idx)),
        "exact_duplicate_groups": len(exact_dup_groups),
        "exact_duplicate_patches": int(n_exact_dup_patches),
        "example_exact_duplicate_indices": list(exact_dup_groups.values())[:10],
        "geometric_duplicate_pairs": int(geometric_pairs),
        "example_geometric_duplicate_indices": geometric_examples,
        "note": "Geometric duplicates are patches that are exact mirrors/rotations of another "
                "patch (not byte-identical). On a regular sliding-window grid this can legitimately "
                "happen near symmetric tissue structures; a high count is a signal to inspect "
                "--stride vs --patch_size for excessive overlap, not automatically a bug.",
    }


# ----------------------------------------------------------------------
# Wavelength ordering check
# ----------------------------------------------------------------------

def _wavelength_report(wavelengths: Optional[np.ndarray]) -> Dict:
    if wavelengths is None:
        return {"present": False}
    wl = np.asarray(wavelengths, dtype=np.float64)
    is_monotonic_increasing = bool(np.all(np.diff(wl) > 0))
    is_monotonic_nondecreasing = bool(np.all(np.diff(wl) >= 0))
    return {
        "present": True,
        "n_bands": int(len(wl)),
        "min_nm": float(wl.min()) if len(wl) else None,
        "max_nm": float(wl.max()) if len(wl) else None,
        "strictly_increasing": is_monotonic_increasing,
        "non_decreasing": is_monotonic_nondecreasing,
        "has_duplicate_wavelengths": bool(len(wl) != len(np.unique(wl))),
    }


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def _discover_npy_dataset_dir(data_dir: str) -> str:
    """Deliberately independent of `training.data.discover_npy_dataset` (which
    imports torch at module level) so this audit module stays usable in a
    torch-free environment - it only ever touches `.npy`/`.json` files."""
    required = ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy")
    candidates = [data_dir, os.path.join(data_dir, "hsi"), os.path.join(data_dir, "rgb")]
    for d in candidates:
        if os.path.isdir(d) and all(os.path.isfile(os.path.join(d, f)) for f in required):
            return d
    raise FileNotFoundError(
        f"Could not find a complete .npy dataset ({required}) under {data_dir!r}. Checked: {candidates}.")


def run_dataset_audit(data_dir: str, manifest_path: Optional[str] = None,
                       out_path: Optional[str] = None,
                       max_patches_per_split: int = 20_000, seed: int = 0) -> Dict:
    """Phase 0.1 entry point. `data_dir` must contain X_train.npy/y_train.npy
    and X_test.npy/y_test.npy (X_val.npy/y_val.npy optional - the layout
    `prepare_histologyhsi_bc_v3.py` writes to `<out_dir>/hsi/` or
    `<out_dir>/rgb/`). Writes and returns `dataset_diagnostics.json`."""
    chosen_dir = _discover_npy_dataset_dir(data_dir)
    x_train, y_train = os.path.join(chosen_dir, "X_train.npy"), os.path.join(chosen_dir, "y_train.npy")
    x_test, y_test = os.path.join(chosen_dir, "X_test.npy"), os.path.join(chosen_dir, "y_test.npy")
    x_val_path = os.path.join(chosen_dir, "X_val.npy")
    y_val_path = os.path.join(chosen_dir, "y_val.npy")
    has_val = os.path.isfile(x_val_path) and os.path.isfile(y_val_path)

    split_files = {
        "train": (x_train, y_train),
        "test": (x_test, y_test),
    }
    if has_val:
        split_files["validation"] = (x_val_path, y_val_path)

    class_names_file = os.path.join(chosen_dir, "class_names.json")
    class_names = None
    if os.path.isfile(class_names_file):
        with open(class_names_file) as f:
            class_names = json.load(f)

    wl_path = os.path.join(chosen_dir, "wavelengths.npy")
    wavelengths = np.load(wl_path) if os.path.isfile(wl_path) else None

    report: Dict = {
        "data_dir": chosen_dir,
        "splits_found": list(split_files.keys()),
        "wavelengths": _wavelength_report(wavelengths),
        "per_split": {},
    }

    per_split_X = {}
    for split, (x_path, y_path) in split_files.items():
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        per_split_X[split] = X
        report["per_split"][split] = {
            "n_samples": int(len(X)),
            "patch_shape_hwc": list(X.shape[1:]),
            "dtype": str(X.dtype),
            "class_distribution": _class_distribution(y, class_names),
            "spectral_statistics": _spectral_stats(X, sample_cap=max_patches_per_split, seed=seed),
            "corruption": _corruption_report(X, sample_cap=max_patches_per_split, seed=seed),
            "duplicates": _duplicate_and_geometric_scan(X, max_patches=max_patches_per_split, seed=seed),
        }

    # ------------------------------------------------------------------
    # Cross-split consistency: identical preprocessing implies the raw
    # per-band mean/std of train vs. validation/test should be in the same
    # ballpark (same calibration, same wavelength ordering, same padding/
    # interpolation). Large divergence is a red flag worth surfacing here
    # even though it's not proof of a bug on its own (class-distribution
    # shift between splits can also cause it).
    # ------------------------------------------------------------------
    consistency = {}
    if "train" in report["per_split"] and len(report["per_split"]) > 1:
        train_stats = report["per_split"]["train"]["spectral_statistics"]
        train_mean = np.array(train_stats.get("per_band_mean", []))
        for split, info in report["per_split"].items():
            if split == "train":
                continue
            other_mean = np.array(info["spectral_statistics"].get("per_band_mean", []))
            if len(train_mean) and len(train_mean) == len(other_mean):
                diff = np.abs(train_mean - other_mean)
                denom = np.maximum(np.abs(train_mean), 1e-8)
                rel_diff = diff / denom
                consistency[f"train_vs_{split}"] = {
                    "max_per_band_mean_abs_diff": float(diff.max()),
                    "mean_per_band_mean_abs_diff": float(diff.mean()),
                    "max_per_band_mean_relative_diff": float(rel_diff.max()),
                    "flag_large_shift": bool(rel_diff.mean() > 0.25),
                }
    report["train_vs_other_split_consistency"] = consistency

    # ------------------------------------------------------------------
    # Patient / slide leakage - delegate to Phase 7's dataset_integrity.
    # ------------------------------------------------------------------
    patient_slide_report = None
    if manifest_path and os.path.isfile(manifest_path):
        try:
            from training.dataset_integrity import check_split_integrity
            patient_slide_report = check_split_integrity(manifest_path)
        except Exception as e:
            patient_slide_report = {"error": str(e)}
    report["patient_slide_leakage"] = patient_slide_report

    # ------------------------------------------------------------------
    # Cross-split duplicate/leakage at the patch level (exact byte-identical
    # patches shared between splits) - reuses Phase 7's hasher directly so
    # this and leakage_report.json never disagree.
    # ------------------------------------------------------------------
    try:
        from training.dataset_integrity import check_patch_leakage
        report["cross_split_patch_leakage"] = check_patch_leakage(
            chosen_dir, max_patches_per_split=max_patches_per_split)
    except Exception as e:
        report["cross_split_patch_leakage"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Overall summary / verdict
    # ------------------------------------------------------------------
    any_corruption = any(
        info["corruption"]["n_nan_patches"] or info["corruption"]["n_inf_patches"]
        for info in report["per_split"].values()
    )
    any_leakage = bool(
        (patient_slide_report or {}).get("leakage_detected")
        or report["cross_split_patch_leakage"].get("leakage_detected")
    )
    any_shift = any(v.get("flag_large_shift") for v in consistency.values())
    report["summary"] = {
        "any_corruption_detected": bool(any_corruption),
        "any_leakage_detected": any_leakage,
        "any_large_train_val_distribution_shift": bool(any_shift),
        "wavelengths_monotonic": report["wavelengths"].get("strictly_increasing", None),
    }

    out_path = out_path or os.path.join(chosen_dir, "dataset_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  [dataset-audit] wrote {out_path}")
    return report


def main():
    import argparse
    p = argparse.ArgumentParser(description="Phase 0.1 - dataset pipeline audit -> dataset_diagnostics.json")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--manifest_path", default=None,
                    help="prepare_histologyhsi_bc_v3.py's _progress.json (for patient/slide leakage)")
    p.add_argument("--out_path", default=None)
    p.add_argument("--max_patches_per_split", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_dataset_audit(args.data_dir, manifest_path=args.manifest_path, out_path=args.out_path,
                       max_patches_per_split=args.max_patches_per_split, seed=args.seed)


if __name__ == "__main__":
    main()
