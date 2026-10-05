# -*- coding: utf-8 -*-
"""
training/metric_validation.py
==============================
Main Development Plan, Phase 8 - Metric Validation.

Automatically detects, per epoch:
  - NaN metrics
  - Infinite metrics
  - Constant metrics (streak-based, any metric)
  - Impossible F1 (F1 outside [0,1], or macro-F1 disagreeing with the mean of
    the per-class F1s)
  - Balanced Accuracy outside [0,1]
  - Sudden validation jumps (a metric moving by more than a configurable
    threshold in a single epoch, which is usually a bug - e.g. a shuffled
    validation set - rather than genuine learning)
  - Missing classes (a class with zero support in y_true or never predicted)
  - Empty predictions (no predictions produced at all for the epoch)

v15 R0.4 - this module is now the SINGLE rule set. `TrainerG_v5.fit` also
calls `training.eval_utils.sanity_check_metrics`, which has been stripped
down to the two rules unique to it, so each message is emitted once per epoch
instead of twice. Two rules here were removed outright because they were not
valid:

  R0.4a/b  macro-F1 was compared against the harmonic mean of MACRO precision
           and recall. That identity holds per class; after averaging it does
           not. It fired on 32 of the RGB evidence run's 33 epochs on entirely
           correct metrics. Replaced by `f1_macro == mean(per_class_f1)`.
  R0.4c    `balanced_accuracy > accuracy on an imbalanced set` was flagged as
           an inconsistency. It is not one: balanced accuracy is the unweighted
           mean of per-class recall and legitimately exceeds accuracy when a
           minority class is over-predicted - exactly the collapse case it kept
           firing on, where `class_never_predicted` already says the true thing.

`MetricValidator.validate_epoch(...)` returns a list of issue dicts for the
epoch and accumulates them; `write_report(path)` dumps everything collected
so far to `validation_report.json`.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence

import numpy as np


class MetricValidator:
    def __init__(self, jump_threshold: float = 0.35, const_streak: int = 3):
        self.jump_threshold = jump_threshold
        self.const_streak = const_streak
        self.epoch_issues: Dict[int, List[Dict]] = {}
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    def validate_epoch(self, epoch: int, metrics: Dict, preds: Optional[np.ndarray] = None,
                        labels: Optional[np.ndarray] = None, num_classes: Optional[int] = None) -> List[Dict]:
        issues: List[Dict] = []

        # 1/2 - NaN / Infinite metrics
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and math.isnan(value):
                issues.append({"type": "nan_metric", "metric": key, "epoch": epoch,
                                "message": f"'{key}' is NaN."})
            elif isinstance(value, float) and math.isinf(value):
                issues.append({"type": "infinite_metric", "metric": key, "epoch": epoch,
                                "message": f"'{key}' is infinite."})

        # 6 - empty predictions
        if preds is not None and len(preds) == 0:
            issues.append({"type": "empty_predictions", "epoch": epoch,
                            "message": "No predictions were produced this epoch."})

        # 5 - missing classes (zero support in labels, or never predicted)
        if preds is not None and labels is not None and num_classes and len(labels) > 0:
            label_support = np.bincount(labels, minlength=num_classes)
            pred_support = np.bincount(preds, minlength=num_classes)
            missing_in_labels = np.where(label_support == 0)[0].tolist()
            never_predicted = np.where(pred_support == 0)[0].tolist()
            if missing_in_labels:
                issues.append({"type": "missing_classes_in_labels", "epoch": epoch,
                                "classes": missing_in_labels,
                                "message": f"Classes {missing_in_labels} have zero support in the "
                                           f"validation labels this epoch."})
            if never_predicted:
                issues.append({"type": "class_never_predicted", "epoch": epoch,
                                "classes": never_predicted,
                                "message": f"Classes {never_predicted} were never predicted this "
                                           f"epoch (possible class collapse)."})

        # 3 - constant metric across a streak of epochs
        collapsed_this_epoch = any(i["type"] == "class_never_predicted" for i in issues)
        for key in ("precision_macro", "precision_weighted", "recall_macro", "f1_macro",
                    "balanced_accuracy", "accuracy"):
            if key not in metrics or metrics[key] is None:
                continue
            streak_vals = [h.get(key) for h in self._history[-(self.const_streak - 1):]] + [metrics[key]]
            if len(streak_vals) >= self.const_streak and all(v is not None for v in streak_vals) \
                    and len(set(round(v, 6) for v in streak_vals)) == 1:
                # v15 R0.4d - when `class_never_predicted` also fired this epoch
                # the cause is known and it is not a metric-computation bug, so
                # say what it actually is instead of sending the reader off to
                # check whether the metric is being recomputed.
                cause = ("the model produced the same prediction for every validation sample."
                         if collapsed_this_epoch else
                         "verify the metric is actually being recomputed each epoch.")
                issues.append({"type": "constant_metric", "metric": key, "epoch": epoch,
                                "value": metrics[key],
                                "class_collapse_concurrent": collapsed_this_epoch,
                                "message": f"'{key}' has been constant ({metrics[key]:.4f}) for "
                                           f"{self.const_streak} epochs in a row - {cause}"})

        # 4 - impossible F1.
        #
        # v15 R0.4a/b - the old rule compared macro-F1 against
        # `2*p*r/(p+r)` on the MACRO precision and recall. That identity holds
        # per class, not after averaging: macro-F1 is the mean of the per-class
        # F1s, and the harmonic mean of the means is a different (generally
        # smaller) number. The check therefore fired on 32 of the RGB run's 33
        # epochs on entirely correct metrics. Replaced with the identity that
        # IS true by construction, plus the range check (which was always valid).
        f1 = metrics.get("f1_macro")
        if f1 is not None and not (0.0 <= f1 <= 1.0):
            issues.append({"type": "impossible_f1", "epoch": epoch, "value": f1,
                            "message": f"macro F1 ({f1}) is outside the valid [0,1] range."})
        per_class_f1 = metrics.get("per_class_f1")
        if f1 is not None and per_class_f1 is not None and len(per_class_f1) > 0:
            expected_f1 = float(np.mean(np.asarray(per_class_f1, dtype=float)))
            if abs(expected_f1 - f1) > 1e-6:
                issues.append({"type": "impossible_f1", "epoch": epoch,
                                "f1": f1, "expected_f1": expected_f1,
                                "message": f"macro F1 ({f1:.6f}) is not the mean of the per-class "
                                           f"F1 scores ({expected_f1:.6f}) - the macro average and "
                                           f"the per-class breakdown disagree."})

        # Balanced-accuracy range check.
        #
        # v15 R0.4c - the `balanced_accuracy > accuracy on an imbalanced set`
        # rule is DELETED. Balanced accuracy is the unweighted mean of per-class
        # recall and legitimately exceeds accuracy whenever a minority class is
        # over-predicted - precisely the HSI collapse case it kept firing on.
        # The condition it was groping for is already covered, correctly, by
        # `class_never_predicted` above.
        bal_acc = metrics.get("balanced_accuracy")
        if bal_acc is not None and not (0.0 <= bal_acc <= 1.0):
            issues.append({"type": "balanced_accuracy_inconsistency", "epoch": epoch, "value": bal_acc,
                            "message": f"balanced_accuracy ({bal_acc}) is outside the valid [0,1] range."})

        # Sudden validation jumps vs. the previous epoch
        if self._history:
            prev = self._history[-1]
            for key in ("val_loss", "accuracy", "balanced_accuracy", "f1_macro"):
                cur_v, prev_v = metrics.get(key), prev.get(key)
                if cur_v is None or prev_v is None:
                    continue
                if abs(cur_v - prev_v) > self.jump_threshold:
                    issues.append({"type": "sudden_validation_jump", "metric": key, "epoch": epoch,
                                    "previous": prev_v, "current": cur_v,
                                    "message": f"'{key}' jumped from {prev_v:.4f} to {cur_v:.4f} "
                                               f"(> {self.jump_threshold} in one epoch) - verify this "
                                               f"is real learning and not e.g. a data-loading bug or "
                                               f"LR spike."})

        self.epoch_issues[epoch] = issues
        self._history.append(metrics)
        return issues

    # ------------------------------------------------------------------
    def write_report(self, path: str) -> Dict:
        all_issues = [issue for issues in self.epoch_issues.values() for issue in issues]
        by_type: Dict[str, int] = {}
        for issue in all_issues:
            by_type[issue["type"]] = by_type.get(issue["type"], 0) + 1
        report = {
            "n_epochs_checked": len(self.epoch_issues),
            "n_issues_total": len(all_issues),
            "issue_counts_by_type": by_type,
            "issues_by_epoch": {str(e): issues for e, issues in self.epoch_issues.items() if issues},
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return report
