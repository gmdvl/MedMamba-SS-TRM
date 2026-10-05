#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/shallow_baseline_v16.py
==================================
MedMamba-SS-TRM v16 plan, Stage 1.1 - gate G4's ceiling, measured under
`per_patch_zscore` too. `scripts/shallow_baseline.py` is imported at
`train_example_v15.py:130`... no - it is a standalone CLI (nothing on the
v15/v11 import path imports it), but its own `--normalization` choices
(`shallow_baseline.py:108-119`, i.e. its `normalize()`/argparse choices) are
a SECOND copy of the 2-mode whitelist alongside `train_example_v6.NpyDataset`'s,
and it is simplest - and safest for the frozen script's own reproducibility
table in its docstring - to add the third mode in a thin wrapper rather than
edit the values documented there.

Imports the frozen script as a MODULE and substitutes its normalization
function for one call, rather than duplicating ~260 lines: `load_subset`,
`fit_global_stats`, `run_probe`, `format_table`, `FEATURE_SETS` all come from
`scripts.shallow_baseline` unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import shallow_baseline as _sb  # frozen, imported not edited
from training.normalization import normalize_patch


def normalize_v16(x: np.ndarray, mode: str, mean=None, std=None) -> np.ndarray:
    """`scripts.shallow_baseline.normalize`, plus `per_patch_zscore`."""
    if mode != "per_patch_zscore":
        return _sb.normalize(x, mode, mean=mean, std=std)
    out = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        out[i], _off, _scale = normalize_patch(x[i], "per_patch_zscore")
    return out


def evaluate_dataset_v16(data_dir: Path, n_train: int, n_val: int, seed: int, normalization: str,
                          feature_sets: List[str], max_iter: int) -> dict:
    t0 = time.time()
    x_tr, y_tr = _sb.load_subset(data_dir, "train", n_train, seed)
    x_va, y_va = _sb.load_subset(data_dir, "val", n_val, seed + 1)

    mean, std = _sb.fit_global_stats(x_tr)   # TRAIN ONLY (unused by per_patch_zscore, harmless to compute)
    x_tr = normalize_v16(x_tr, normalization, mean, std)
    x_va = normalize_v16(x_va, normalization, mean, std)

    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    results = {}
    for kind in feature_sets:
        t = time.time()
        results[kind] = _sb.run_probe(kind, x_tr, y_tr, x_va, y_va, max_iter, seed)
        results[kind]["seconds"] = round(time.time() - t, 1)

    return {
        "data_dir": str(data_dir), "normalization": normalization,
        "n_train": int(len(x_tr)), "n_val": int(len(x_va)), "num_classes": num_classes,
        "patch_shape": list(x_tr.shape[1:]), "seed": seed,
        "train_class_counts": np.bincount(y_tr, minlength=num_classes).tolist(),
        "val_class_counts": np.bincount(y_va, minlength=num_classes).tolist(),
        "results": results, "total_seconds": round(time.time() - t0, 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, nargs="+", type=Path)
    p.add_argument("--n_train", type=int, default=25000)
    p.add_argument("--n_val", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalization", choices=["global_zscore", "per_sample_minmax", "per_patch_zscore"],
                    default="global_zscore")
    p.add_argument("--feature_sets", nargs="+", choices=_sb.FEATURE_SETS, default=list(_sb.FEATURE_SETS))
    p.add_argument("--max_iter", type=int, default=1500)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    reports = []
    for d in args.data_dir:
        report = evaluate_dataset_v16(d, args.n_train, args.n_val, args.seed, args.normalization,
                                       args.feature_sets, args.max_iter)
        print(_sb.format_table(report), flush=True)
        reports.append(report)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(reports, f, indent=2, default=str)
        print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
