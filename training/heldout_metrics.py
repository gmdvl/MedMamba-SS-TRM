# -*- coding: utf-8 -*-
"""
training/heldout_metrics.py
===============================
One metric function for every model in the paper, computed from saved
probabilities, so that MedMamba-SS-TRM, MedMamba, HybridSN, SpectralFormer
and the probes are scored by identical code. Calibration comes from the
pipeline's own `training.metrics.calibration_metrics` (15 bins), so ECE here
reproduces the value in each run's `test_report.json`.

Also: per-patient macro recall, and a patient-level bootstrap for the
difference between two models scored on the same patches.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from training.metrics import calibration_metrics


def classification_metrics(labels: np.ndarray, probs: Optional[np.ndarray] = None,
                           preds: Optional[np.ndarray] = None, num_classes: int = 3) -> Dict:
    labels = np.asarray(labels).astype(np.int64)
    if preds is None:
        preds = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=list(range(num_classes)),
                                                 zero_division=0)
    out = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(),
        "per_class_support": s.tolist(),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum(axis=1, keepdims=True)
        onehot = np.eye(num_classes)[labels]
        out["roc_auc_macro"] = float(roc_auc_score(onehot, probs, average="macro"))
        out["pr_auc_macro"] = float(average_precision_score(onehot, probs, average="macro"))
        out.update(calibration_metrics(probs, labels))
        out.update(mce_details(probs, labels))
    return out


MCE_MIN_FRAC = 0.001


def mce_details(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15,
                min_frac: float = MCE_MIN_FRAC) -> Dict:
    """With the binning of `calibration_metrics`: the size of the bin that sets
    the MCE (`mce_bin_n`), and the MCE over bins holding at least `min_frac` of
    the samples (`mce_floor`), since an MCE set by a near-empty bin is noise."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    best_gap, best_n, floor = -1.0, 0, 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        if gap > best_gap:
            best_gap, best_n = gap, int(mask.sum())
        if mask.sum() >= min_frac * len(labels):
            floor = max(floor, float(gap))
    return {"mce_bin_n": best_n, "mce_floor": floor}


def per_patient(labels: np.ndarray, preds: np.ndarray, groups: np.ndarray,
                num_classes: int = 3) -> Dict[str, Dict]:
    """Macro recall over the classes a patient actually has, plus accuracy."""
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        y, p = labels[m], preds[m]
        present = [c for c in range(num_classes) if (y == c).any()]
        rec = [float((p[y == c] == c).mean()) for c in present]
        out[str(g)] = {"n": int(m.sum()), "classes_present": present,
                       "macro_recall": float(np.mean(rec)), "accuracy": float((p == y).mean()),
                       "per_class_recall": dict(zip(map(str, present), rec))}
    return out


def patient_bootstrap_delta(labels, preds_a, preds_b, groups, n_boot: int = 2000, seed: int = 0,
                            metric: str = "balanced_accuracy") -> Dict:
    """Resample PATIENTS with replacement (the unit of independence here), pool
    their patches, and recompute metric(a) - metric(b). Returns the observed
    difference, a percentile 95 % interval and the fraction of resamples > 0."""
    fn = {"balanced_accuracy": balanced_accuracy_score,
          "f1_macro": lambda y, p: f1_score(y, p, average="macro", zero_division=0)}[metric]
    uniq = np.array(sorted(set(groups.tolist())))
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    obs = fn(labels, preds_a) - fn(labels, preds_b)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(fn(y, preds_a[idx]) - fn(y, preds_b[idx]))
    deltas = np.asarray(deltas)
    return {"metric": metric, "observed": float(obs), "n_patients": int(len(uniq)),
            "n_boot": int(len(deltas)),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "frac_positive": float((deltas > 0).mean())}
