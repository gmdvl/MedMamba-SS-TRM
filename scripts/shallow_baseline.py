#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/shallow_baseline.py
=============================
MedMamba-SS-TRM v15 plan, R0.0 - "the cheapest instrument in this plan".

Fits `sklearn.linear_model.LogisticRegression(class_weight='balanced')` on a
few thousand patches' worth of hand-made features and reports accuracy,
balanced accuracy and macro-F1. No GPU, no torch, ~2 minutes per dataset.

The point is not the probe. The point is the CEILING: a 442,215-parameter
spatial-spectral network that cannot beat per-band mean and standard
deviation has not demonstrated anything, and "the model is bad" only becomes
a thesis result once it is "the model is bad RELATIVE TO A KNOWN CEILING".
Re-run this on every re-prepped dataset and keep the JSON next to the run.

Feature sets (all fit on train only; the z-score is fit on train only too):

  majority       constant prediction of the most frequent training class.
  mean           per-band mean over the patch                      (C features)
  mean_std       per-band mean AND standard deviation             (2C features)
  flat           the whole patch, flattened                     (H*W*C features)

Reproduced on this tree at the defaults (`--seed 0`, 25,000 train / 15,000
val, `--normalization global_zscore`). The v15 plan's section 2.1 table is in
brackets - the small differences are the sampling seed, nothing else:

  data/hsi_v7-80_10_10/hsi          accuracy   balanced   f1_macro
    majority                     0.6361 [.6329]  0.3333  -
    mean                (32)     0.5147 [.5179]  0.4857 [.4863]  0.4510 [.4532]
    mean_std            (64)     0.6719 [.6590]  0.6078 [.6007]  0.5612 [.5539]
    flat              (3872)     0.5865 [.5811]  0.4472 [.4427]  0.4412 [.4379]
    -> gate G4 ceiling: balanced 0.6078, F1-macro 0.5612

  data/pad_v6                       accuracy   balanced   f1_macro
    majority                     0.3684 [.3612]  0.1667  -
    mean                 (3)     0.2260 [.2284]  0.2541 [.2605]  0.1895 [.1918]
    mean_std             (6)     0.2595 [.2689]  0.2899 [.2997]  0.2217 [.2319]
    flat               (363)     0.2169 [.2175]  0.2387 [.2397]  0.1892 [.1914]  (hit max_iter)
    -> gate G4 ceiling: balanced 0.2899, F1-macro 0.2217

Read those two ceilings against the model. The HSI run reached balanced
accuracy 0.3333 and F1-macro 0.0456 - far BELOW 64 numbers. The RGB run
reached 0.2571 / 0.2419 - at parity with SIX numbers, after 442,602
parameters and 17.5 GPU-hours. The +0.12 jump from `mean` to `mean_std` on
HSI says a large part of the signal is within-patch spectral VARIANCE, which
is exactly the spatial-spectral structure the model exists to exploit.

Usage:

    python scripts/shallow_baseline.py --data_dir data/hsi_v7-80_10_10/hsi
    python scripts/shallow_baseline.py --data_dir data/pad_v6 --out probe.json

    # R2.1's measurement: what per-sample min-max normalization costs.
    python scripts/shallow_baseline.py --data_dir data/hsi_v7-80_10_10/hsi \\
        --normalization per_sample_minmax
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

FEATURE_SETS = ("majority", "mean", "mean_std", "flat")


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def _find_split(data_dir: Path, split: str):
    """(X_path, y_path) for `split`, falling back from val to test the same
    way `train_example_v6.discover_data` does (the HSI dataset has no separate
    validation split)."""
    x, y = data_dir / f"X_{split}.npy", data_dir / f"y_{split}.npy"
    if x.is_file() and y.is_file():
        return x, y
    if split == "val":
        return _find_split(data_dir, "test")
    raise FileNotFoundError(f"missing {x} / {y}")


def load_subset(data_dir: Path, split: str, n: int, seed: int):
    """`n` patches drawn without replacement, memory-mapped so a 46 GB
    `X_train.npy` never has to fit in RAM."""
    x_path, y_path = _find_split(data_dir, split)
    X = np.load(x_path, mmap_mode="r")
    y = np.asarray(np.load(y_path, mmap_mode="r"))
    n = min(n, len(X))
    idx = np.sort(np.random.default_rng(seed).choice(len(X), size=n, replace=False))
    return np.nan_to_num(np.asarray(X[idx], dtype=np.float32)), y[idx].astype(np.int64)


def fit_global_stats(x: np.ndarray):
    """Per-channel mean/std over a [N,H,W,C] array. TRAIN ONLY - fitting these
    on val or test leaks their statistics into the normalization."""
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))
    return mean, np.where(std > 1e-6, std, 1.0)


def normalize(x: np.ndarray, mode: str, mean=None, std=None) -> np.ndarray:
    if mode == "global_zscore":
        return (x - mean) / std
    # per_sample_minmax - `train_example_v6.NpyDataset`'s default: every patch
    # rescaled by ITS OWN scalar min and max over all H*W*C values, both driven
    # by outlier pixels.
    flat = x.reshape(len(x), -1)
    lo = flat.min(axis=1)[:, None, None, None]
    hi = flat.max(axis=1)[:, None, None, None]
    span = np.maximum(hi - lo, 1e-6)
    out = (x - lo) / span
    return np.where(hi - lo > 1e-6, out, 0.0)


def featurize(x: np.ndarray, kind: str) -> np.ndarray:
    """x: [N,H,W,C] -> [N,D]."""
    if kind == "mean":
        return x.mean(axis=(1, 2))
    if kind == "mean_std":
        return np.concatenate([x.mean(axis=(1, 2)), x.std(axis=(1, 2))], axis=1)
    if kind == "flat":
        return x.reshape(len(x), -1)
    raise ValueError(f"unknown feature set {kind!r}")


# ----------------------------------------------------------------------
# probes
# ----------------------------------------------------------------------

def score(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_probe(kind, x_tr, y_tr, x_va, y_va, max_iter, seed) -> dict:
    from sklearn.linear_model import LogisticRegression

    if kind == "majority":
        majority = int(np.bincount(y_tr).argmax())
        out = score(y_va, np.full_like(y_va, majority))
        out.update(dim=0, majority_class=majority, converged=True)
        # F1-macro of a constant predictor is not a meaningful comparison point.
        out["f1_macro"] = None
        return out

    f_tr, f_va = featurize(x_tr, kind), featurize(x_va, kind)
    clf = LogisticRegression(class_weight="balanced", max_iter=max_iter, n_jobs=-1,
                              random_state=seed)
    import warnings as _w
    from sklearn.exceptions import ConvergenceWarning
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always", ConvergenceWarning)
        clf.fit(f_tr, y_tr)
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in caught)
    out = score(y_va, clf.predict(f_va))
    out.update(dim=int(f_tr.shape[1]), converged=bool(converged))
    return out


def evaluate_dataset(data_dir: Path, n_train: int, n_val: int, seed: int, normalization: str,
                      feature_sets, max_iter: int) -> dict:
    t0 = time.time()
    x_tr, y_tr = load_subset(data_dir, "train", n_train, seed)
    x_va, y_va = load_subset(data_dir, "val", n_val, seed + 1)

    mean, std = fit_global_stats(x_tr)          # TRAIN ONLY
    x_tr = normalize(x_tr, normalization, mean, std)
    x_va = normalize(x_va, normalization, mean, std)

    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    results = {}
    for kind in feature_sets:
        t = time.time()
        results[kind] = run_probe(kind, x_tr, y_tr, x_va, y_va, max_iter, seed)
        results[kind]["seconds"] = round(time.time() - t, 1)

    return {
        "data_dir": str(data_dir),
        "normalization": normalization,
        "n_train": int(len(x_tr)),
        "n_val": int(len(x_va)),
        "num_classes": num_classes,
        "patch_shape": list(x_tr.shape[1:]),
        "seed": seed,
        "train_class_counts": np.bincount(y_tr, minlength=num_classes).tolist(),
        "val_class_counts": np.bincount(y_va, minlength=num_classes).tolist(),
        "results": results,
        "total_seconds": round(time.time() - t0, 1),
    }


def format_table(report: dict) -> str:
    lines = [
        f"{report['data_dir']}  ({report['num_classes']} classes, "
        f"{report['n_train']} train / {report['n_val']} val, "
        f"normalization={report['normalization']})",
        f"  {'feature set':<14}{'dim':>6}{'accuracy':>11}{'balanced':>11}{'f1_macro':>11}",
    ]
    for kind, r in report["results"].items():
        f1 = "-" if r["f1_macro"] is None else f"{r['f1_macro']:.4f}"
        flag = "" if r.get("converged", True) else "  (hit max_iter - lower bound)"
        lines.append(f"  {kind:<14}{r['dim']:>6}{r['accuracy']:>11.4f}"
                      f"{r['balanced_accuracy']:>11.4f}{f1:>11}{flag}")
    best = max((r for k, r in report["results"].items() if r["f1_macro"] is not None),
               key=lambda r: r["balanced_accuracy"], default=None)
    if best is not None:
        lines.append(f"  -> ceiling to beat (gate G4): balanced accuracy {best['balanced_accuracy']:.4f}, "
                      f"F1-macro {best['f1_macro']:.4f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, nargs="+",
                    help="one or more prepared dataset directories (each holding X_train.npy etc.)")
    p.add_argument("--n_train", type=int, default=25000)
    p.add_argument("--n_val", type=int, default=15000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normalization", choices=["global_zscore", "per_sample_minmax"],
                    default="global_zscore",
                    help="global_zscore (default) is what v15 uses for HSI. Run both to measure "
                         "what per-sample min-max costs (v15 plan R2.1: -0.044 balanced accuracy "
                         "on HSI).")
    p.add_argument("--feature_sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    p.add_argument("--max_iter", type=int, default=1500)
    p.add_argument("--out", default=None,
                    help="write the full report as JSON here (recommended: into the run directory)")
    args = p.parse_args(argv)

    reports = []
    for d in args.data_dir:
        report = evaluate_dataset(Path(d), args.n_train, args.n_val, args.seed,
                                   args.normalization, args.feature_sets, args.max_iter)
        reports.append(report)
        print(format_table(report), flush=True)

    if args.out:
        payload = reports[0] if len(reports) == 1 else {"datasets": reports}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
