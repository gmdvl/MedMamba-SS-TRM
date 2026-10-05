# -*- coding: utf-8 -*-
"""
training/eval_utils.py
=======================
Phase 4 of improve-prompt_v5.plan.md: fixes the evaluation pipeline so every
reported metric is internally consistent and reproducible.

  1. Validation mode - callers are responsible for `model.eval()` /
     `torch.no_grad()` (done once per validation pass in the trainer, not
     per-batch here); this module only ever consumes already-collected
     numpy arrays, never touches the model.
  2. Validation dataset determinism - enforced by the caller's DataLoader
     (shuffle=False, no augmentation); out of scope for this module.
  3. Metric calculation - every metric below is computed with sklearn
     (accuracy, balanced accuracy, precision/recall/F1 in both macro and
     weighted flavors), replacing the old hand-rolled versions.
  4. Per-class metrics - `write_classification_report`.
  5. Confusion matrix - `save_confusion_matrix` (.npy + .png via
     training.plots.plot_confusion_matrix).
  6. Prediction distribution - `save_prediction_distribution` (true label,
     predicted label, confidence, softmax entropy, per sample).
  7. Metric consistency sanity checks - `sanity_check_metrics`, returns a
     list of warning strings (never raises - these are diagnostics, not
     hard failures, per the plan: "generate warnings in the logs rather
     than silently accepting suspicious values").
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix as sk_confusion_matrix, classification_report,
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    labels: Optional[Sequence[int]] = None) -> Dict:
    """Phase 4 item 3: every metric computed via sklearn from (y_true, y_pred)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        # Main Development Plan, Phase 5 - per-class Precision/Recall/F1/Support/Accuracy
        "per_class_precision": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "per_class_support": compute_per_class_support(y_true, labels=labels),
        "per_class_accuracy": compute_per_class_accuracy(y_true, y_pred, labels=labels),
    }


def compute_per_class_support(y_true: np.ndarray, labels: Optional[Sequence[int]] = None) -> List[int]:
    """v15 R0.4f - support counts aligned with `labels`, so every entry of
    `per_class_support` corresponds to the same class as the entry at the same
    index of `per_class_precision`/`recall`/`f1`.

    Previously this was `np.bincount(y_true, minlength=max(labels)+1 if labels
    else y_true.max()+1)`, which (a) sized the array from the largest label
    PRESENT when `labels` was not given, silently dropping a trailing class
    with zero validation support, and (b) indexed by raw label value rather
    than by position in `labels`, so a non-contiguous `labels` subset produced
    a support vector misaligned with every other per-class array.
    """
    y_true = np.asarray(y_true)
    if labels is not None:
        counts = np.bincount(y_true, minlength=int(max(labels)) + 1) if y_true.size else \
            np.zeros(int(max(labels)) + 1, dtype=np.int64)
        return [int(counts[int(c)]) if int(c) < len(counts) else 0 for c in labels]
    n = int(y_true.max()) + 1 if y_true.size else 0
    return np.bincount(y_true, minlength=n).tolist()


def compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                labels: Optional[Sequence[int]] = None) -> Dict[int, float]:
    """Main Development Plan, Phase 5 - per-class Accuracy (one-vs-rest
    binary accuracy per class, distinct from per-class recall)."""
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    n_total = cm.sum()
    if n_total == 0:
        return {}
    out = {}
    label_ids = labels if labels is not None else list(range(cm.shape[0]))
    for i, c in enumerate(label_ids):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = n_total - tp - fn - fp
        out[int(c)] = float((tp + tn) / n_total)
    return out


def write_classification_report(y_true: np.ndarray, y_pred: np.ndarray, out_path: str,
                                  class_names: Optional[List[str]] = None,
                                  labels: Optional[Sequence[int]] = None) -> str:
    """Phase 4 item 4: per-class precision/recall/F1/support -> classification_report.txt."""
    target_names = class_names if class_names else None
    report_txt = classification_report(y_true, y_pred, labels=labels, target_names=target_names,
                                        zero_division=0)
    with open(out_path, "w") as f:
        f.write(report_txt)
    return report_txt


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str, epoch: int,
                           class_names: Optional[List[str]] = None,
                           labels: Optional[Sequence[int]] = None) -> np.ndarray:
    """Phase 4 item 5: confusion_matrix_epoch_XX.png (via training.plots) + confusion_matrix.npy."""
    from training.plots import plot_confusion_matrix

    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)
    plot_confusion_matrix(cm, class_names, out_dir, normalize=False)
    png_path = os.path.join(out_dir, "confusion_matrix.png")
    epoch_png_path = os.path.join(out_dir, f"confusion_matrix_epoch_{epoch:02d}.png")
    if os.path.exists(png_path):
        os.replace(png_path, epoch_png_path)
    return cm


def save_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                                  out_path: str) -> None:
    """Phase 4 item 6: per-sample true label, predicted label, confidence
    (max softmax probability) and softmax entropy - exposes class-collapse
    or overconfidence immediately (e.g. `predicted_label` never varying, or
    `confidence` sitting at ~1.0 for a model that's actually wrong a lot)."""
    eps = 1e-12
    confidence = probs.max(axis=1)
    entropy = -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1)
    np.savez(out_path, true_label=y_true, predicted_label=y_pred,
             confidence=confidence, entropy=entropy)


def sanity_check_metrics(metrics: Dict, history: Optional[List[Dict]] = None,
                          const_streak: int = 3) -> List[str]:
    """Phase 4 item 7: automatic sanity checks, returned as warning strings
    (never raised) so the trainer can log them without aborting training.

    v15 R0.4e - this is no longer a second, overlapping rule set.
    `training.metric_validation.MetricValidator` is the single rule set;
    `TrainerG_v5.fit` calls both, so every rule implemented there was emitted
    TWICE per epoch. Only the two rules unique to this function remain:

      * near-perfect weighted precision (>=0.99) with low accuracy (<0.6).
      * `accuracy == balanced_accuracy` for `const_streak` epochs in a row.

    Deleted here (and kept, corrected, in `MetricValidator`): the
    harmonic-mean F1 check (R0.4a - macro-F1 is the MEAN of the per-class F1s
    and is never the harmonic mean of macro precision and macro recall; it
    fired on 32 of the RGB run's 33 epochs) and the constant-metric streak
    check (R0.4e - duplicated verbatim).
    """
    warnings: List[str] = []

    acc, prec_w = metrics.get("accuracy"), metrics.get("precision_weighted")
    if acc is not None and prec_w is not None and prec_w >= 0.99 and acc < 0.6:
        warnings.append(f"Near-perfect weighted precision ({prec_w:.4f}) with low accuracy "
                         f"({acc:.4f}) is an impossible/suspicious combination - check for a "
                         f"class-collapse or a metric-computation bug.")

    if metrics.get("accuracy") is not None and metrics.get("balanced_accuracy") is not None:
        if abs(metrics["accuracy"] - metrics["balanced_accuracy"]) < 1e-9 and history:
            same_streak = 1
            for h in reversed(history):
                if abs(h.get("accuracy", -1) - h.get("balanced_accuracy", -2)) < 1e-9:
                    same_streak += 1
                else:
                    break
            if same_streak >= const_streak:
                warnings.append(f"accuracy == balanced_accuracy for {same_streak} epochs in a row "
                                 f"- suspicious unless classes are already perfectly balanced; "
                                 f"verify balanced_accuracy is computed independently (per-class "
                                 f"recall averaged), not aliased to accuracy.")

    return warnings
