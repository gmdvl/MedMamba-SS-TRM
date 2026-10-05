# -*- coding: utf-8 -*-
"""
training/metrics.py
====================
Phase 6 (classification metrics) of the .npy-centric plan / Phase 10-11 of
the original v1 plan. Pure NumPy (no sklearn dependency, mirroring the
style of the original `evaluate.py`) so this keeps working in minimal
environments.

Computes: accuracy, balanced accuracy, macro/weighted precision-recall-F1,
confusion matrix, Cohen's kappa, Matthews correlation coefficient (multiclass
generalization), macro one-vs-rest ROC-AUC and PR-AUC, plus calibration
metrics (NLL, Brier score, Expected/Maximum Calibration Error) when
probabilities are available.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# numpy >= 2.0 renamed trapz -> trapezoid; keep working on either version.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def _prf_from_cm(cm: np.ndarray):
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = cm.sum(axis=1)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) > 0 else 0.0)
    return precision, recall, f1, support


def cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    pe = float((row_marg * col_marg).sum())
    return float((po - pe) / (1 - pe)) if pe != 1.0 else 0.0


def matthews_corrcoef_multiclass(cm: np.ndarray) -> float:
    """Multiclass generalization of MCC (Gorodkin 2004)."""
    cm = cm.astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0
    t = cm.sum(axis=1)   # true class totals
    p = cm.sum(axis=0)   # predicted class totals
    c = np.trace(cm)
    num = c * n - float(t @ p)
    den = np.sqrt((n ** 2 - float(p @ p)) * (n ** 2 - float(t @ t)))
    return float(num / den) if den > 0 else 0.0


def balanced_accuracy(cm: np.ndarray) -> float:
    support = cm.sum(axis=1)
    recalls = np.array([cm[c, c] / support[c] if support[c] > 0 else 0.0 for c in range(cm.shape[0])])
    return float(recalls.mean())


def _one_vs_rest_auc(scores: np.ndarray, labels_binary: np.ndarray) -> float:
    """AUC via the rank-sum (Mann-Whitney U) formula - no sklearn needed."""
    pos = scores[labels_binary == 1]
    neg = scores[labels_binary == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = ranks[labels_binary == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _one_vs_rest_pr_auc(scores: np.ndarray, labels_binary: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    y = labels_binary[order]
    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(_trapz(precision, recall))


def macro_roc_pr_auc(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict:
    aucs, pr_aucs = [], []
    for c in range(num_classes):
        y_bin = (labels == c).astype(np.int64)
        aucs.append(_one_vs_rest_auc(probs[:, c], y_bin))
        pr_aucs.append(_one_vs_rest_pr_auc(probs[:, c], y_bin))
    valid_auc = [a for a in aucs if not np.isnan(a)]
    valid_pr = [a for a in pr_aucs if not np.isnan(a)]
    return {
        "per_class_roc_auc": aucs,
        "per_class_pr_auc": pr_aucs,
        "macro_roc_auc": float(np.mean(valid_auc)) if valid_auc else float("nan"),
        "macro_pr_auc": float(np.mean(valid_pr)) if valid_pr else float("nan"),
    }


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict:
    """NLL, Brier score, Expected/Maximum Calibration Error (ECE/MCE)."""
    n = len(labels)
    eps = 1e-12
    true_probs = np.clip(probs[np.arange(n), labels], eps, 1.0)
    nll = float(-np.log(true_probs).mean())

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    brier = float(((probs - one_hot) ** 2).sum(axis=1).mean())

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)

    return {"nll": nll, "brier_score": brier, "ece": float(ece), "mce": float(mce)}


def compute_classification_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int,
                                    probs: Optional[np.ndarray] = None) -> Dict:
    cm = confusion_matrix(preds, labels, num_classes)
    precision, recall, f1, support = _prf_from_cm(cm)
    weights = support / max(support.sum(), 1)
    # Main Development Plan, Phase 5 - per-class Accuracy: for class c this is
    # ordinary binary accuracy on the one-vs-rest problem (c vs. not-c), i.e.
    # (TP+TN)/N - distinct from per-class recall, which only looks at rows
    # where the true label is c.
    n_total = cm.sum()
    per_class_accuracy = np.array([
        (cm[c, c] + (n_total - cm[c, :].sum() - cm[:, c].sum() + cm[c, c])) / n_total if n_total > 0 else 0.0
        for c in range(num_classes)
    ])

    result = {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": balanced_accuracy(cm),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
        "cohen_kappa": cohen_kappa(cm),
        "matthews_corrcoef": matthews_corrcoef_multiclass(cm),
        "per_class": {
            "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "support": support.tolist(),
            "accuracy": per_class_accuracy.tolist(),
        },
        "confusion_matrix": cm.tolist(),
    }
    if probs is not None:
        result.update(macro_roc_pr_auc(probs, labels, num_classes))
        result["calibration"] = calibration_metrics(probs, labels)
    return result
