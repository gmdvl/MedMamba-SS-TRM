# -*- coding: utf-8 -*-
"""
training/dataset_integrity.py
==============================
Main Development Plan, Phase 7 - Dataset Integrity.

Automatically verifies:
  - no duplicated patients across train/validation/test
  - no duplicated slides (captures) across train/validation/test
  - no duplicated spectra / patches across splits (content-hash based)
  - no duplicated patches *within* a split (informational only - exact
    duplicate patches within one split aren't leakage, but are worth
    flagging since they usually indicate a bug upstream)

Two entry points:
  - `check_split_integrity(manifest_path)` - patient/slide-level check from
    `prepare_histologyhsi_bc_v3.py`'s `manifest.json` (has train/validation/
    test patient sets by construction). Writes `dataset_split_report.json`.
  - `check_patch_leakage(data_dir)` - patch-level check straight from the
    final `.npy` files (`X_train.npy` / `X_test.npy` / optionally
    `X_val.npy`), by hashing each patch. Writes `leakage_report.json`.

`run_full_integrity_check(...)` runs both and raises `DatasetLeakageError`
if `abort_on_leakage=True` and any leakage was found - "Abort training if
leakage exists" per the plan.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np


class DatasetLeakageError(RuntimeError):
    """Raised by `run_full_integrity_check` when leakage is detected and
    `abort_on_leakage=True`."""


# ----------------------------------------------------------------------
# Patient / slide (capture) level - from the prep-script manifest
# ----------------------------------------------------------------------

def check_split_integrity(manifest_path: str) -> Dict:
    """Verifies `train_patients` / `validation_patients` / `test_patients`
    (written by `prepare_histologyhsi_bc_v3.py`'s `new_manifest`) are
    pairwise disjoint, and does the same for captures ("slides") if the
    manifest records per-capture split assignment."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    train_p = set(manifest.get("train_patients", []))
    val_p = set(manifest.get("validation_patients", []))
    test_p = set(manifest.get("test_patients", []))

    patient_overlaps = {
        "train_validation": sorted(train_p & val_p),
        "train_test": sorted(train_p & test_p),
        "validation_test": sorted(val_p & test_p),
    }

    slide_overlaps = {"train_validation": [], "train_test": [], "validation_test": []}
    completed = manifest.get("completed_captures", {})
    if completed:
        split_of = {}
        for cap_key, info in completed.items():
            split_of.setdefault(info.get("split"), set()).add(cap_key)
        train_c, val_c, test_c = split_of.get("train", set()), split_of.get("validation", set()), split_of.get("test", set())
        slide_overlaps = {
            "train_validation": sorted(train_c & val_c),
            "train_test": sorted(train_c & test_c),
            "validation_test": sorted(val_c & test_c),
        }

    any_patient_overlap = any(patient_overlaps.values())
    any_slide_overlap = any(slide_overlaps.values())

    report = {
        "manifest_path": manifest_path,
        "n_train_patients": len(train_p),
        "n_validation_patients": len(val_p),
        "n_test_patients": len(test_p),
        "patient_overlaps": patient_overlaps,
        "slide_overlaps": slide_overlaps,
        "patient_level_leakage": any_patient_overlap,
        "slide_level_leakage": any_slide_overlap,
        "leakage_detected": any_patient_overlap or any_slide_overlap,
    }
    return report


# ----------------------------------------------------------------------
# Patch / spectrum level - content-hash based, works on the final .npy
# ----------------------------------------------------------------------

def _hash_patches(x_path: str, max_patches: Optional[int] = None, sample_stride: int = 1) -> Dict[str, List[int]]:
    """Memory-maps `x_path` ([N,H,W,C]) and returns {sha1_hex: [indices]}
    for a (possibly strided/capped) subset of patches. Hashing raw bytes is
    exact - two patches hash equal iff they are bit-identical.

    Reads only the selected patches, via `npy_integrity.read_npy_rows`
    (`open()/seek()/readinto()`, never mmap), so a multi-GB - or corrupt /
    failing-disk - X is neither fully read into RAM nor able to SIGBUS the
    main process here: a physically-unreadable extent raises a catchable
    `OSError(5)/EIO`. The authoritative structural check is
    `npy_integrity.validate_dataset`, which the caller runs BEFORE this."""
    from training.npy_integrity import _read_npy_header, read_npy_rows
    shape, _dtype, _fortran, _off = _read_npy_header(x_path)
    n = int(shape[0]) if shape else 0
    indices = list(range(0, n, sample_stride))
    if max_patches is not None:
        indices = indices[:max_patches]
    hashes: Dict[str, List[int]] = {}
    CHUNK = 4096
    for start in range(0, len(indices), CHUNK):
        batch = indices[start:start + CHUNK]
        rows = read_npy_rows(x_path, batch)
        for i, row in zip(batch, rows):
            h = hashlib.sha1(np.ascontiguousarray(row).tobytes()).hexdigest()
            hashes.setdefault(h, []).append(int(i))
    return hashes


def check_patch_leakage(data_dir: str, max_patches_per_split: Optional[int] = 50_000,
                         sample_stride: int = 1) -> Dict:
    """Content-hashes patches in each split found under `data_dir`
    (X_train.npy / X_val.npy / X_test.npy - val is optional) and reports any
    hash that appears in more than one split (= a literal duplicate patch
    leaked across the split boundary) as well as within-split duplicates
    (informational, not leakage).

    `max_patches_per_split` caps work on very large datasets; set to None to
    hash every patch. `sample_stride` can be raised (e.g. 5) to hash every
    Nth patch for a fast approximate check on huge datasets.
    """
    split_files = {
        "train": os.path.join(data_dir, "X_train.npy"),
        "validation": os.path.join(data_dir, "X_val.npy"),
        "test": os.path.join(data_dir, "X_test.npy"),
    }
    split_files = {s: p for s, p in split_files.items() if os.path.isfile(p)}
    if "train" not in split_files or "test" not in split_files:
        return {
            "data_dir": data_dir, "checked": False,
            "reason": "X_train.npy and/or X_test.npy not found - skipping patch-level check.",
            "leakage_detected": False,
        }

    per_split_hashes = {
        split: _hash_patches(path, max_patches=max_patches_per_split, sample_stride=sample_stride)
        for split, path in split_files.items()
    }

    splits = list(per_split_hashes.keys())
    cross_split_duplicates = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            shared = set(per_split_hashes[a]) & set(per_split_hashes[b])
            if shared:
                cross_split_duplicates[f"{a}_{b}"] = {
                    "n_duplicate_hashes": len(shared),
                    "example_hashes": list(shared)[:10],
                }

    within_split_duplicate_counts = {
        split: sum(1 for idxs in hashes.values() if len(idxs) > 1)
        for split, hashes in per_split_hashes.items()
    }

    report = {
        "data_dir": data_dir,
        "checked": True,
        "splits_checked": splits,
        "n_patches_hashed": {s: sum(len(v) for v in h.values()) for s, h in per_split_hashes.items()},
        "sample_stride": sample_stride,
        "max_patches_per_split": max_patches_per_split,
        "cross_split_duplicate_patches": cross_split_duplicates,
        "within_split_duplicate_patch_groups": within_split_duplicate_counts,
        "leakage_detected": len(cross_split_duplicates) > 0,
    }
    return report


# ----------------------------------------------------------------------
# Combined entry point
# ----------------------------------------------------------------------

def run_full_integrity_check(out_dir: str, manifest_path: Optional[str] = None,
                              npy_data_dir: Optional[str] = None,
                              abort_on_leakage: bool = True,
                              max_patches_per_split: Optional[int] = 50_000,
                              sample_stride: int = 1) -> Dict:
    """Runs whichever checks are possible given the paths provided, writes
    `dataset_split_report.json` and `leakage_report.json` into `out_dir`,
    and (if `abort_on_leakage`) raises `DatasetLeakageError` when either
    check finds leakage - "Abort training if leakage exists" (Phase 7)."""
    os.makedirs(out_dir, exist_ok=True)

    split_report = None
    if manifest_path and os.path.isfile(manifest_path):
        split_report = check_split_integrity(manifest_path)
        with open(os.path.join(out_dir, "dataset_split_report.json"), "w") as f:
            json.dump(split_report, f, indent=2)

    patch_report = None
    if npy_data_dir and os.path.isdir(npy_data_dir):
        patch_report = check_patch_leakage(npy_data_dir, max_patches_per_split=max_patches_per_split,
                                            sample_stride=sample_stride)

    leakage_report = {
        "split_level": split_report,
        "patch_level": patch_report,
        "leakage_detected": bool(
            (split_report and split_report.get("leakage_detected")) or
            (patch_report and patch_report.get("leakage_detected"))
        ),
    }
    with open(os.path.join(out_dir, "leakage_report.json"), "w") as f:
        json.dump(leakage_report, f, indent=2)

    if leakage_report["leakage_detected"]:
        msg = f"Dataset leakage detected - see {os.path.join(out_dir, 'leakage_report.json')}"
        print(f"  [dataset-integrity] {msg}")
        if abort_on_leakage:
            raise DatasetLeakageError(msg)
    else:
        print("  [dataset-integrity] no leakage detected.")

    return leakage_report
