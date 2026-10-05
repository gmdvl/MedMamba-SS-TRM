# -*- coding: utf-8 -*-
"""
scripts/tune_class_weight_power.py
=====================================
MedMamba-SS-TRM v16 plan, Stage 3.3 (E-3).

`train_example_v15.py`'s `--class_weight_power 0.75` default provenance
(`train_example_v15.py:185-197`, `training/losses.py:49-66`) was chosen by
re-deciding a run's SAVED TEST-SPLIT probabilities - a test-set-informed
hyperparameter choice that contaminates every number reported from that
split. This script is the identical sweep, restricted to saved VALIDATION
probabilities.

"Re-deciding" a saved probability array under a different `class_weight_power`
means moving the decision boundary WITHOUT retraining: `argmax(probs *
class_weight ** power)` instead of plain `argmax(probs)`. This is a lower
bound on what actually training at that weighting would achieve (the decision
boundary the network learned never moves) - good enough to pick a power to
train with, not a substitute for training at it and re-validating.

Class weights come from `training.losses.compute_class_weights` (frozen,
imported not reimplemented) - the exact function `--loss weighted_ce`/
`focal_weighted` use at train time, so the powers swept here mean the same
thing the `--class_weight_power` CLI flag does.

Input: an `.npz` with `true_label` [N] and `probs` [N, num_classes] - e.g.
`test_predictions.npz`'s VALIDATION-split analogue (a full-probability dump
from a validation pass; `training.eval_utils.save_prediction_distribution`
does not save one today, only confidence/entropy - point this script at
whatever full-probability validation dump the run produced, and pass
`--y_train_npy` for the class counts the weights are computed from).

Usage::

    python scripts/tune_class_weight_power.py \\
        --val_probs_npz experiments/<run>/val_probs.npz \\
        --y_train_npy data/hsi_v7-80_10_10_importance/hsi/y_train.npy \\
        --out experiments/<run>/class_weight_power_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.losses import compute_class_weights  # frozen, imported not reimplemented


def _macro_f1(true_label: np.ndarray, pred_label: np.ndarray, num_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(true_label, pred_label, labels=list(range(num_classes)),
                           average="macro", zero_division=0))


def _per_class_f1(true_label: np.ndarray, pred_label: np.ndarray, num_classes: int) -> List[float]:
    from sklearn.metrics import f1_score
    return f1_score(true_label, pred_label, labels=list(range(num_classes)),
                     average=None, zero_division=0).tolist()


def sweep(true_label: np.ndarray, probs: np.ndarray, y_train: np.ndarray, num_classes: int,
          powers: List[float], method: str = "balanced") -> dict:
    results = []
    baseline_pred = probs.argmax(axis=1)
    for power in powers:
        weights = compute_class_weights(y_train, num_classes, method=method, power=power).numpy()
        reweighted_pred = (probs * weights[None, :]).argmax(axis=1)
        results.append({
            "power": power,
            "macro_f1": _macro_f1(true_label, reweighted_pred, num_classes),
            "per_class_f1": _per_class_f1(true_label, reweighted_pred, num_classes),
            "weights": weights.tolist(),
            "agrees_with_argmax_probs": bool(np.array_equal(reweighted_pred, baseline_pred)) if power == 0.0 else None,
        })
    results.sort(key=lambda r: -r["macro_f1"])
    return {
        "method": method, "powers": powers, "n_samples": int(len(true_label)),
        "num_classes": num_classes, "results": results,
        "best_power": results[0]["power"], "best_macro_f1": results[0]["macro_f1"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val_probs_npz", required=True,
                     help="npz with true_label [N] and probs [N, num_classes], from the "
                          "VALIDATION split only. Never point this at test_predictions.npz.")
    ap.add_argument("--y_train_npy", required=True, help="y_train.npy - class counts for the weights")
    ap.add_argument("--powers", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--method", choices=["balanced", "inverse"], default="balanced")
    ap.add_argument("--out", default=None, help="default: class_weight_power_sweep.json next to --val_probs_npz")
    args = ap.parse_args(argv)

    if "test" in os.path.basename(args.val_probs_npz).lower():
        raise SystemExit(
            f"--val_probs_npz {args.val_probs_npz!r} looks like a TEST-split file by name. This "
            f"script exists specifically because tuning on test contaminates the reported test "
            f"number (E-3) - point it at a validation-split probability dump instead.")

    data = np.load(args.val_probs_npz)
    true_label, probs = data["true_label"], data["probs"]
    y_train = np.load(args.y_train_npy, mmap_mode="r")
    num_classes = probs.shape[1]

    report = sweep(true_label, probs, np.asarray(y_train), num_classes, args.powers, method=args.method)
    report["tuned_on"] = "validation"
    report["source_npz"] = os.path.abspath(args.val_probs_npz)

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.val_probs_npz)),
                                         "class_weight_power_sweep.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[tune_class_weight_power] {len(args.powers)} power(s), n={report['n_samples']} "
          f"validation samples", flush=True)
    for r in report["results"]:
        print(f"  power={r['power']:.2f}  macro_f1={r['macro_f1']:.4f}", flush=True)
    print(f"  best: power={report['best_power']} macro_f1={report['best_macro_f1']:.4f} -> {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
