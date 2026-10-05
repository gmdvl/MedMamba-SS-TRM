# -*- coding: utf-8 -*-
"""
training/class_coverage.py
===========================
Main Development Plan, Phase 3 - Enforce Class Coverage.

"The preprocessing script must fail before training if a required class is
missing. For every split, verify that every expected class is represented."

This is a standalone, dependency-free check so it can be called both from a
prep script (right after writing X/y .npy files) and from a train entry
point (right before training starts - `train_example_v7.py` does the
latter, as a fail-fast safety net independent of whatever prepared the
data).

Also includes a best-effort Phase 11 helper (`class_distribution_breakdown`)
for patient-/capture-level class counts from `prepare_histologyhsi_bc_v4.py`'s
`_progress.json` manifest, so "thousands of patches from one patient" isn't
mistaken for thousands of independent observations.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


def verify_class_coverage(splits: Dict[str, Iterable[int]], num_classes: int,
                           class_names: Optional[List[str]] = None,
                           fail_hard: bool = True) -> Dict:
    """`splits`: e.g. {"train": y_train, "validation": y_val, "test": y_test}.

    Returns a JSON-serializable report. If `fail_hard` and any split is
    missing a class that appears in `num_classes`, raises ValueError with a
    message identifying exactly which split/class combinations are missing -
    per the plan, this must be a hard failure, not a silent gap, unless the
    caller has explicitly opted out (`fail_hard=False`, e.g. because the
    source dataset genuinely lacks enough independent patients for a class
    in a particular split and this is a known/accepted limitation).
    """
    names = class_names if class_names and len(class_names) == num_classes else \
        [str(i) for i in range(num_classes)]

    report = {"num_classes": num_classes, "class_names": names, "per_split_counts": {}, "missing": {}}
    all_missing: Dict[str, List[str]] = {}

    for split_name, y in splits.items():
        y = np.asarray(y)
        counts = np.bincount(y, minlength=num_classes).tolist()
        report["per_split_counts"][split_name] = {names[i]: counts[i] for i in range(num_classes)}
        missing = [names[i] for i, c in enumerate(counts) if c == 0]
        if missing:
            all_missing[split_name] = missing

    report["missing"] = all_missing
    report["coverage_ok"] = len(all_missing) == 0

    if fail_hard and not report["coverage_ok"]:
        lines = [f"  - {split}: missing {missing}" for split, missing in all_missing.items()]
        raise ValueError(
            "Class coverage check failed (Main Development Plan, Phase 3) - at least one "
            "split has ZERO samples for a required class:\n" + "\n".join(lines) + "\n"
            "Either the source dataset genuinely lacks enough independent patients for these "
            "classes in this split (in which case pass --allow_missing_classes / "
            "fail_hard=False and treat this as a documented dataset limitation, per the plan's "
            "own escape hatch), or --split/--seed needs adjusting so every class survives the "
            "patient-level split. Preprocessing must not silently produce an incomplete split."
        )
    return report


def class_distribution_breakdown(manifest: Dict, labels: Dict[str, int],
                                  class_names: List[str]) -> Dict:
    """Main Development Plan, Phase 11 (best-effort). `manifest` is the dict
    loaded from `_progress.json` (prepare_histologyhsi_bc_v4.py); `labels`
    maps capture-key -> class id (same mapping `build_labels()` produces).
    Returns capture-level and patient-level counts per class per split -
    patch-level counts are already in dataset_statistics.json and are not
    duplicated here."""
    completed = manifest.get("completed_captures", {})
    train_patients = set(manifest.get("train_patients", []))
    val_patients = set(manifest.get("validation_patients", []))
    test_patients = set(manifest.get("test_patients", []))

    def which_split(patient):
        if patient in test_patients:
            return "test"
        if patient in val_patients:
            return "validation"
        return "train"

    capture_counts: Dict[str, Dict[str, int]] = {"train": {}, "validation": {}, "test": {}}
    patient_by_class_split: Dict[str, Dict[str, set]] = {"train": {}, "validation": {}, "test": {}}

    for cap_key, info in completed.items():
        label = labels.get(cap_key)
        if label is None:
            continue
        cname = class_names[label] if label < len(class_names) else str(label)
        split = info.get("split", "train")
        capture_counts[split][cname] = capture_counts[split].get(cname, 0) + 1

    return {
        "capture_count_per_class_per_split": capture_counts,
        "note": "Patient-level counts require patient id per capture beyond what "
                "_progress.json stores per-entry; cross-reference train/validation/test_patients "
                "with your capture-discovery output if patient-level granularity is needed.",
    }
