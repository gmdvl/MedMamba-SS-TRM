# -*- coding: utf-8 -*-
"""tests/test_numerical_stability.py - Plan Phase 33/34/12/22.

Run with: pytest tests/test_numerical_stability.py -v
"""
import torch
import torch.nn as nn

from training.numerical_stability import (
    NumericalStabilityController, StabilityConfig, classify_skip_ratio, inspect_gradients,
    check_parameters_finite, check_loss_components, run_smoke_test,
)


def test_classify_skip_ratio_buckets():
    assert classify_skip_ratio(0.0) == "healthy"
    assert classify_skip_ratio(0.005) == "warning"
    assert classify_skip_ratio(0.03) == "degraded"
    assert classify_skip_ratio(0.07) == "critical"
    assert classify_skip_ratio(0.5) == "invalid"
    assert classify_skip_ratio(1.0) == "fatal"


def test_inspect_gradients_all_finite():
    model = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    report = inspect_gradients(model)
    assert report.all_finite
    assert report.first_bad_parameter is None


def test_inspect_gradients_detects_nan():
    model = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    model.weight.grad[0, 0] = float("nan")
    report = inspect_gradients(model)
    assert not report.all_finite
    assert report.first_bad_parameter == "weight"
    assert report.nan_count >= 1


def test_check_parameters_finite():
    model = nn.Linear(4, 2)
    ok, bad = check_parameters_finite(model)
    assert ok and not bad
    with torch.no_grad():
        model.weight[0, 0] = float("inf")
    ok, bad = check_parameters_finite(model)
    assert not ok and "weight" in bad


def test_check_loss_components():
    status = check_loss_components({
        "classification": torch.tensor(1.0),
        "SAM": torch.tensor(float("nan")),
        "GAN": None,
        "total": torch.tensor(float("nan")),
    })
    assert status["classification"] == "finite"
    assert status["SAM"] == "NONFINITE"
    assert status["GAN"] == "not_used"
    assert status["total"] == "NONFINITE"


def test_controller_isolated_bad_batch_does_not_abort():
    cfg = StabilityConfig(max_consecutive_bad_batches=3, max_gradient_skip_ratio=0.5)
    ctrl = NumericalStabilityController(cfg)
    ctrl.start_epoch()
    keep_going = ctrl.record_batch(loss_finite=False, grad_inspection=None, grad_norm=None,
                                    optimizer_stepped=False)
    assert keep_going
    keep_going = ctrl.record_batch(loss_finite=True, grad_inspection=None, grad_norm=1.0,
                                    optimizer_stepped=True)
    assert keep_going
    summary = ctrl.finish_epoch()
    assert ctrl.should_abort_run() is None
    assert summary.valid_updates == 1


def test_controller_aborts_on_consecutive_bad_batches():
    cfg = StabilityConfig(max_consecutive_bad_batches=3)
    ctrl = NumericalStabilityController(cfg)
    ctrl.start_epoch()
    keep_going = True
    for _ in range(3):
        keep_going = ctrl.record_batch(loss_finite=False, grad_inspection=None, grad_norm=None,
                                        optimizer_stepped=False)
    assert not keep_going
    assert ctrl.should_abort_run() is not None


def test_controller_epoch_skip_ratio_invalidates():
    cfg = StabilityConfig(max_gradient_skip_ratio=0.5, max_consecutive_bad_batches=100)
    ctrl = NumericalStabilityController(cfg)
    ctrl.start_epoch()
    # 10 batches, 8 skipped -> skip_ratio 0.8 > 0.5
    for _ in range(8):
        ctrl.record_batch(loss_finite=False, grad_inspection=None, grad_norm=None, optimizer_stepped=False)
    for _ in range(2):
        ctrl.record_batch(loss_finite=True, grad_inspection=None, grad_norm=1.0, optimizer_stepped=True)
    summary = ctrl.finish_epoch()
    assert not summary.is_valid_epoch
    assert ctrl.should_abort_run() is not None


def test_smoke_test_passes_for_healthy_model():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(5, 4)
    y = torch.randint(0, 2, (5,))

    def forward_and_loss(m, x, y):
        logits = m(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss, {"classification": loss, "total": loss}

    report = run_smoke_test(model, x, y, forward_and_loss, opt)
    assert report["passed"], report


def test_smoke_test_fails_on_nan_input():
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.full((3, 4), float("nan"))
    y = torch.randint(0, 2, (3,))

    def forward_and_loss(m, x, y):
        logits = m(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss, {"classification": loss, "total": loss}

    report = run_smoke_test(model, x, y, forward_and_loss, opt)
    assert not report["passed"]
    assert report["stage"] == "input"
