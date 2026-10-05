# -*- coding: utf-8 -*-
"""
training/per_class_metrics.py
==============================
improve-promp_v8.plan.md, Phase 0.5 - "Per-Class Validation Analysis".

`training.eval_utils.compute_classification_metrics` already reports
per-class precision/recall/F1/support/accuracy (Main Development Plan,
Phase 5). This module adds exactly what Phase 0.5 asks for on top of that:

  - per-class Loss (mean per-sample cross-entropy for samples of that
    true class)
  - per-class Specificity (true-negative rate, one-vs-rest)
  - per-class Sensitivity (== recall, reported explicitly under its own
    name since the plan lists it as a separate bullet)
  - per-class MCC (one-vs-rest binary Matthews correlation coefficient)
  - per-class Cohen's Kappa (one-vs-rest binary kappa)

...and a heuristic (`identify_dominant_class`) for "Determine whether a
single class is responsible for the validation loss behavior": flags a
class whose share of total validation loss is disproportionate to its
share of validation samples.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def _binary_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float(num / den) if den > 0 else 0.0


def _binary_kappa(tp: int, tn: int, fp: int, fn: int) -> float:
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((fn + tn) / n) * ((fp + tn) / n)
    pe = p_yes + p_no
    return float((po - pe) / (1 - pe)) if pe != 1.0 else 0.0


def compute_per_class_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                        num_classes: int,
                                        per_sample_loss: Optional[np.ndarray] = None,
                                        class_names: Optional[List[str]] = None) -> Dict:
    """Phase 0.5: per-class Loss/Accuracy/Precision/Recall/Specificity/
    Sensitivity/F1/MCC/Cohen's Kappa. `per_sample_loss` (optional), if given,
    must align 1:1 with `y_true`/`y_pred` (e.g. per-sample cross-entropy from
    the validation pass) - without it, per-class Loss is omitted."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    names = class_names if class_names and len(class_names) == num_classes else [str(c) for c in labels]

    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    n_total = int(cm.sum())

    per_class = {}
    for i, c in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(n_total - tp - fn - fp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0          # == sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n_total if n_total > 0 else 0.0

        entry = {
            "class_name": names[i],
            "support": tp + fn,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "mcc": _binary_mcc(tp, tn, fp, fn),
            "cohen_kappa": _binary_kappa(tp, tn, fp, fn),
        }
        if per_sample_loss is not None:
            mask = (y_true == c)
            entry["loss"] = float(np.mean(per_sample_loss[mask])) if mask.any() else None
        per_class[str(c)] = entry

    return {"num_classes": num_classes, "per_class": per_class}


def identify_dominant_class(per_class_extended: Dict, min_share_ratio: float = 1.5) -> Optional[Dict]:
    """Phase 0.5: "Determine whether a single class is responsible for the
    validation loss behavior." Flags the class with the largest ratio of
    (share of total validation loss) to (share of validation samples),
    when that ratio exceeds `min_share_ratio` (i.e. the class contributes
    disproportionately more loss than its sample share would predict).
    Returns None if no per-class loss is available or nothing stands out."""
    per_class = per_class_extended.get("per_class", {})
    entries = [(cid, e) for cid, e in per_class.items() if e.get("loss") is not None]
    if not entries:
        return None

    total_support = sum(e["support"] for _, e in entries)
    total_loss_weighted = sum(e["loss"] * e["support"] for _, e in entries)
    if total_support == 0 or total_loss_weighted == 0:
        return None

    ratios = {}
    for cid, e in entries:
        support_share = e["support"] / total_support
        loss_share = (e["loss"] * e["support"]) / total_loss_weighted
        ratios[cid] = loss_share / support_share if support_share > 0 else float("inf")

    dominant_cid = max(ratios, key=ratios.get)
    if ratios[dominant_cid] < min_share_ratio:
        return None

    return {
        "class_id": dominant_cid,
        "class_name": per_class[dominant_cid]["class_name"],
        "loss_to_support_share_ratio": float(ratios[dominant_cid]),
        "mean_loss": per_class[dominant_cid]["loss"],
        "support": per_class[dominant_cid]["support"],
        "message": (f"Class '{per_class[dominant_cid]['class_name']}' contributes "
                    f"{ratios[dominant_cid]:.2f}x more validation loss than its sample share "
                    f"would predict - worth inspecting in isolation (label noise, ambiguous "
                    f"tissue category, insufficient training examples, etc.)."),
    }
