#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/export_val_subset_indices.py
========================================
Write the validation-subset indices that `train_example_v16.py` (and
`scripts/train_hsi_baseline.py`) select checkpoints on, so that an
external trainer - the MedMamba baseline's `train_hsi_v20.py` - can select on
exactly the same patches.

The draw is `_subsample_indices_v16(y_val, frac, "stratified_group", K,
seed + 1, groups=groups_val)`: note the `seed + 1`, mirrored here.

Light: reads only y_val.npy and groups_val.npy (a few MB).

Usage:
    python scripts/export_val_subset_indices.py --data_dir data/hsi_v9-trainsel/hsi \
        --seeds 1 7 13 23 42
    -> <data_dir>/val_subset_s{seed}.npy
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--val_subsample_frac", type=float, default=0.0986)
    args = p.parse_args(argv)

    from training.train_pipeline import _subsample_indices_v16

    y = np.load(args.data_dir / "y_val.npy")
    g = np.load(args.data_dir / "groups_val.npy", allow_pickle=True)
    k = int(y.max()) + 1
    for s in args.seeds:
        idx = _subsample_indices_v16(y, args.val_subsample_frac, "stratified_group", k, s + 1, groups=g)
        out = args.data_dir / f"val_subset_s{s}.npy"
        np.save(out, idx.astype(np.int64))
        print(f"seed {s}: {len(idx)} of {len(y)} validation patches -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
