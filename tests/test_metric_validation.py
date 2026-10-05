# -*- coding: utf-8 -*-
"""
test_metric_validation.py
===========================
MedMamba-SS-TRM v15 plan, R0.4 "Verify" - the metric validators must stop crying
wolf on correct metrics, and must keep firing on genuinely inconsistent ones.

The RGB evidence run
(`experiments/20260901_195001_pad_v6_recursive_rgb_mlp_sup3_bs256_bf16`)
emitted `impossible_f1` on 32 of its 33 epochs against entirely correct
sklearn metrics, because the rule compared macro-F1 with the harmonic mean of
MACRO precision and recall. `test_correct_multiclass_metrics_emit_no_impossible_f1`
is that case, reduced.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from training.eval_utils import (
    compute_classification_metrics, compute_per_class_support, sanity_check_metrics,
)
from training.metric_validation import MetricValidator


def _issue_types(issues):
    return [i["type"] for i in issues]


def _metrics(y_true, y_pred, num_classes):
    return compute_classification_metrics(np.asarray(y_true), np.asarray(y_pred),
                                           labels=list(range(num_classes)))


# ----------------------------------------------------------------------
# R0.4a/b - the harmonic-mean rule is gone; the mean-of-per-class rule works
# ----------------------------------------------------------------------

def test_correct_multiclass_metrics_emit_no_impossible_f1():
    """A perfectly ordinary 6-class result. Under the old rule this produced
    an `impossible_f1` every epoch."""
    rng = np.random.default_rng(0)
    # PAD-UFES's shape: 6 classes at 16.3:1 imbalance. The imbalance matters -
    # on a perfectly balanced set the harmonic mean of the macro averages
    # happens to land close to macro-F1, which is why the broken rule survived
    # every synthetic check until a real imbalanced run hit it.
    y_true = np.concatenate([np.full(n, c) for c, n in enumerate([1630, 900, 500, 300, 200, 100])])
    y_pred = np.where(rng.random(len(y_true)) < 0.35, y_true,
                      rng.integers(0, 6, size=len(y_true)))
    metrics = _metrics(y_true, y_pred, 6)

    # The old rule, restated, to show it really would have fired here.
    p, r, f1 = metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"]
    assert abs(2 * p * r / (p + r) - f1) > 1e-3, "this fixture no longer exercises the old rule"

    issues = MetricValidator().validate_epoch(1, metrics, preds=y_pred, labels=y_true, num_classes=6)
    assert "impossible_f1" not in _issue_types(issues)


def test_genuinely_inconsistent_f1_still_emits_impossible_f1():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 4, size=500)
    y_pred = rng.integers(0, 4, size=500)
    metrics = _metrics(y_true, y_pred, 4)
    metrics["f1_macro"] = metrics["f1_macro"] + 0.2      # corrupt the macro average only
    issues = MetricValidator().validate_epoch(1, metrics, preds=y_pred, labels=y_true, num_classes=4)
    assert "impossible_f1" in _issue_types(issues)


def test_out_of_range_f1_still_emits_impossible_f1():
    issues = MetricValidator().validate_epoch(1, {"f1_macro": 1.4})
    assert "impossible_f1" in _issue_types(issues)


def test_macro_f1_is_the_mean_of_per_class_f1_by_construction():
    """The identity the new rule checks - true for every sklearn result, which
    is why the rule is safe to run at 1e-6."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 5, size=1000)
    y_pred = rng.integers(0, 5, size=1000)
    metrics = _metrics(y_true, y_pred, 5)
    assert metrics["f1_macro"] == pytest.approx(float(np.mean(metrics["per_class_f1"])), abs=1e-12)


# ----------------------------------------------------------------------
# R0.4c - balanced_accuracy > accuracy is legitimate
# ----------------------------------------------------------------------

def test_balanced_accuracy_above_accuracy_emits_nothing():
    """The HSI case: a heavily imbalanced set where the minority classes are
    over-predicted, so balanced accuracy legitimately exceeds accuracy."""
    y_true = np.array([0] * 900 + [1] * 50 + [2] * 50)
    y_pred = np.concatenate([
        np.array([0] * 300 + [1] * 300 + [2] * 300),   # class 0 mostly wrong
        np.array([1] * 45 + [0] * 5),                  # class 1 mostly right
        np.array([2] * 45 + [0] * 5),                  # class 2 mostly right
    ])
    metrics = _metrics(y_true, y_pred, 3)
    assert metrics["balanced_accuracy"] > metrics["accuracy"], "fixture does not exercise the rule"
    issues = MetricValidator().validate_epoch(1, metrics, preds=y_pred, labels=y_true, num_classes=3)
    assert "balanced_accuracy_inconsistency" not in _issue_types(issues)


def test_out_of_range_balanced_accuracy_still_flagged():
    issues = MetricValidator().validate_epoch(1, {"balanced_accuracy": -0.1})
    assert "balanced_accuracy_inconsistency" in _issue_types(issues)


# ----------------------------------------------------------------------
# R0.4d - constant_metric wording
# ----------------------------------------------------------------------

def _run_streak(validator, metrics, preds, labels, num_classes, epochs=3):
    issues = []
    for e in range(1, epochs + 1):
        issues = validator.validate_epoch(e, dict(metrics), preds=preds, labels=labels,
                                           num_classes=num_classes)
    return issues


def test_constant_metric_says_class_collapse_when_collapse_also_fired():
    y_true = np.array([0] * 60 + [1] * 20 + [2] * 20)
    y_pred = np.zeros(100, dtype=np.int64)               # everything predicted as class 0
    metrics = _metrics(y_true, y_pred, 3)
    issues = _run_streak(MetricValidator(), metrics, y_pred, y_true, 3)
    constant = [i for i in issues if i["type"] == "constant_metric"]
    assert constant, "a fully constant prediction history should trip constant_metric"
    for i in constant:
        assert i["class_collapse_concurrent"] is True
        assert "same prediction for every validation sample" in i["message"]
        assert "not actually being recomputed" not in i["message"]


def test_constant_metric_keeps_the_recompute_wording_without_a_collapse():
    y_true = np.array([0, 1, 2] * 40)
    y_pred = np.array([0, 1, 2] * 40)                    # every class predicted -> no collapse
    metrics = _metrics(y_true, y_pred, 3)
    issues = _run_streak(MetricValidator(), metrics, y_pred, y_true, 3)
    constant = [i for i in issues if i["type"] == "constant_metric"]
    assert constant
    for i in constant:
        assert i["class_collapse_concurrent"] is False
        assert "recomputed" in i["message"]


# ----------------------------------------------------------------------
# R0.4e - de-duplication
# ----------------------------------------------------------------------

def test_sanity_check_metrics_no_longer_duplicates_the_validator_rules():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 6, size=800)
    y_pred = np.where(rng.random(800) < 0.4, y_true, rng.integers(0, 6, size=800))
    metrics = _metrics(y_true, y_pred, 6)
    history = [dict(metrics) for _ in range(4)]
    warnings = sanity_check_metrics(metrics, history=history)
    joined = " ".join(warnings).lower()
    assert "harmonic mean" not in joined
    assert "constant" not in joined


def test_sanity_check_metrics_keeps_its_two_unique_rules():
    warnings = sanity_check_metrics({"accuracy": 0.4, "precision_weighted": 0.995})
    assert any("near-perfect weighted precision" in w.lower() for w in warnings)

    balanced = {"accuracy": 0.5, "balanced_accuracy": 0.5}
    history = [dict(balanced) for _ in range(4)]
    warnings = sanity_check_metrics(balanced, history=history)
    assert any("accuracy == balanced_accuracy" in w for w in warnings)


# ----------------------------------------------------------------------
# R0.4f - a zero-support class keeps its slot
# ----------------------------------------------------------------------

def test_zero_support_class_keeps_its_slot_in_the_macro_average():
    """Class 3 has no validation samples and is never predicted. With
    `labels=range(4)` it stays in the macro average (as a 0.0), so macro-F1
    reports over 4 classes; without it, sklearn would average over 3 and
    report a flatteringly higher number."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])

    with_labels = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2, 3])
    without_labels = compute_classification_metrics(y_true, y_pred)

    assert len(with_labels["per_class_f1"]) == 4
    assert len(with_labels["per_class_support"]) == 4
    assert with_labels["per_class_support"][3] == 0
    assert with_labels["f1_macro"] == pytest.approx(0.75)
    assert without_labels["f1_macro"] == pytest.approx(1.0)


def test_per_class_support_is_aligned_with_the_label_list():
    y_true = np.array([0, 0, 0, 2, 2, 5])
    assert compute_per_class_support(y_true, labels=[0, 1, 2]) == [3, 0, 2]
    assert compute_per_class_support(y_true, labels=[5, 0]) == [1, 3]
    assert compute_per_class_support(y_true) == [3, 0, 2, 0, 0, 1]


def test_per_class_support_handles_an_empty_split():
    assert compute_per_class_support(np.array([], dtype=np.int64), labels=[0, 1, 2]) == [0, 0, 0]


# ----------------------------------------------------------------------
# The regression the whole item exists for
# ----------------------------------------------------------------------

def test_rgb_evidence_run_shape_produces_zero_impossible_f1_over_33_epochs():
    """33 epochs of a 6-class imbalanced result of the kind the RGB run
    produced. Old behaviour: 32 `impossible_f1`. Required: 0."""
    rng = np.random.default_rng(4)
    y_true = np.concatenate([np.full(n, c) for c, n in enumerate([1200, 800, 400, 300, 200, 100])])
    validator = MetricValidator()
    total = 0
    for epoch in range(1, 34):
        noise = rng.random(len(y_true))
        y_pred = np.where(noise < 0.25 + epoch * 0.005, y_true,
                          rng.integers(0, 6, size=len(y_true)))
        metrics = _metrics(y_true, y_pred, 6)
        issues = validator.validate_epoch(epoch, metrics, preds=y_pred, labels=y_true, num_classes=6)
        total += sum(1 for i in issues if i["type"] == "impossible_f1")
    assert total == 0
