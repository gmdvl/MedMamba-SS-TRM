#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/shallow_probe_heldout.py
====================================
The shallow probes of `scripts/shallow_baseline.py` were only ever scored on a
15,000-patch VALIDATION sample, so no network's test-split margin over them
could be stated. This refits them exactly as that script does (same 25,000
training patches, same seed, same train-only global z-score, same
class-balanced LogisticRegression), checks that the validation numbers
reproduce `shallow_baseline.json`, and then scores the FULL test and
validation splits, chunked so the arrays never have to fit in memory.

Writes <data_dir>/shallow_probe_heldout_v19.json and, per feature set,
<data_dir>/shallow_probe_heldout_v19_<kind>.npz with full-split predictions
(for per-patient analysis next to the networks).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import shallow_baseline as sb  # noqa: E402


def fit(kind, x_tr, y_tr, max_iter, seed):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight="balanced", max_iter=max_iter, n_jobs=-1, random_state=seed)
    clf.fit(sb.featurize(x_tr, kind), y_tr)
    return clf


def predict_split(clf, kind, x_path, mean, std, chunk=40000):
    X = np.load(x_path, mmap_mode="r")
    probs = []
    for i in range(0, len(X), chunk):
        x = np.nan_to_num(np.asarray(X[i:i + chunk], dtype=np.float32))
        probs.append(clf.predict_proba(sb.featurize(sb.normalize(x, "global_zscore", mean, std), kind)))
    return np.concatenate(probs).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--n_train", type=int, default=25000)
    p.add_argument("--n_val", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_iter", type=int, default=1500)
    p.add_argument("--feature_sets", nargs="+", default=["mean", "mean_std", "flat"])
    args = p.parse_args(argv)
    d = args.data_dir

    x_tr, y_tr = sb.load_subset(d, "train", args.n_train, args.seed)
    x_va, y_va = sb.load_subset(d, "val", args.n_val, args.seed + 1)
    mean, std = sb.fit_global_stats(x_tr)
    x_tr = sb.normalize(x_tr, "global_zscore", mean, std)
    x_va = sb.normalize(x_va, "global_zscore", mean, std)
    stored = {}
    if (d / "shallow_baseline.json").is_file():
        s = json.load(open(d / "shallow_baseline.json"))
        stored = (s[0] if isinstance(s, list) else s).get("results", {})

    from sklearn.metrics import balanced_accuracy_score, f1_score
    out = {"data_dir": str(d), "n_train": len(x_tr), "seed": args.seed, "normalization": "global_zscore",
           "results": {}}
    for kind in args.feature_sets:
        t = time.time()
        clf = fit(kind, x_tr, y_tr, args.max_iter, args.seed)
        pv = clf.predict(sb.featurize(x_va, kind))
        repro = {"balanced_accuracy": float(balanced_accuracy_score(y_va, pv)),
                 "f1_macro": float(f1_score(y_va, pv, average="macro"))}
        ref = stored.get(kind, {})
        print(f"[{kind}] val-sample reproduction: bal {repro['balanced_accuracy']:.4f} "
              f"(stored {ref.get('balanced_accuracy')}), F1 {repro['f1_macro']:.4f} "
              f"(stored {ref.get('f1_macro')})", flush=True)
        res = {"dim": int(sb.featurize(x_tr[:1], kind).shape[1]), "val_sample_reproduction": repro,
               "stored_val_sample": ref}
        for split in ("test", "val"):
            probs = predict_split(clf, kind, d / f"X_{split}.npy", mean, std)
            y = np.load(d / f"y_{split}.npy")
            np.savez_compressed(d / f"shallow_probe_heldout_v19_{kind}_{split}.npz", probs=probs,
                                true_label=y, predicted_label=probs.argmax(1))
            res[split] = {"balanced_accuracy": float(balanced_accuracy_score(y, probs.argmax(1))),
                          "f1_macro": float(f1_score(y, probs.argmax(1), average="macro")),
                          "accuracy": float((probs.argmax(1) == y).mean()), "n": int(len(y))}
            print(f"   full {split}: bal {res[split]['balanced_accuracy']:.4f} "
                  f"F1 {res[split]['f1_macro']:.4f} acc {res[split]['accuracy']:.4f}", flush=True)
        res["seconds"] = round(time.time() - t, 1)
        out["results"][kind] = res
    json.dump(out, open(d / "shallow_probe_heldout_v19.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
