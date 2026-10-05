# -*- coding: utf-8 -*-
"""
training/class_imbalance.py
============================
Main Development Plan, Phase 11 - Analyze Class Imbalance.

"Before training, calculate class_count / class_percentage for train /
validation / test." This module does the patch-level half of that (the
patch counts already visible in X_*.npy/y_*.npy); capture-/patient-level
counts ("thousands of patches from one patient should not automatically be
treated as thousands of independent observations") are covered separately
by `training.class_coverage.class_distribution_breakdown`, which needs the
prep script's manifest and so lives in that module instead of duplicating
manifest-parsing here.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np


def compute_split_imbalance_report(splits: Dict[str, np.ndarray], num_classes: int,
                                    class_names: Optional[List[str]] = None) -> Dict:
    """`splits`: e.g. {"train": y_train, "validation": y_val, "test": y_test}."""
    names = class_names if class_names and len(class_names) == num_classes else \
        [str(i) for i in range(num_classes)]
    report = {"class_names": names, "per_split": {}}

    for split_name, y in splits.items():
        y = np.asarray(y)
        counts = np.bincount(y, minlength=num_classes)
        total = int(counts.sum())
        pct = (counts / total * 100.0) if total > 0 else counts.astype(np.float64)
        nonzero = counts[counts > 0]
        imbalance_ratio = float(counts.max() / nonzero.min()) if len(nonzero) else None

        report["per_split"][split_name] = {
            "class_count": {names[i]: int(counts[i]) for i in range(num_classes)},
            "class_percentage": {names[i]: round(float(pct[i]), 2) for i in range(num_classes)},
            "total": total,
            "imbalance_ratio_max_over_min": round(imbalance_ratio, 2) if imbalance_ratio is not None else None,
        }
    return report


def write_imbalance_report(out_path: str, splits: Dict[str, np.ndarray], num_classes: int,
                            class_names: Optional[List[str]] = None) -> Dict:
    report = compute_split_imbalance_report(splits, num_classes, class_names)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    worst = max(
        (v["imbalance_ratio_max_over_min"] for v in report["per_split"].values()
         if v["imbalance_ratio_max_over_min"] is not None),
        default=None,
    )
    print(f"[class-imbalance] wrote {out_path}"
          + (f" (worst max/min class ratio = {worst:.1f}x)" if worst is not None else ""))
    return report
