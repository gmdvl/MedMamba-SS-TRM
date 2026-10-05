# -*- coding: utf-8 -*-
"""
test_class_weight_power.py
============================
`--class_weight_power` (v15 R2.4) - the strength knob on inverse-frequency
class weighting.

Context, because the default is deliberately NOT the recommended value: the
v15 `recursive` HSI run used full (power=1.0) weighting, which on train counts
of healthy 722,358 / DCIS 122,850 / IDC 1,606,878 hands DCIS 13.1x the weight
of IDC. That run labels 32.4% of genuinely healthy test patches DCIS and its
test macro-F1 stalls at 0.804. Re-deciding its saved probabilities at an
effective power of 0.75 yields 0.847. The knob exists to reach that regime;
the default stays at 1.0 so no existing run's numbers move.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.losses import build_criterion, compute_class_weights

# The real HSI split's train counts.
COUNTS = (722358, 122850, 1606878)


def _y():
    return np.concatenate([np.full(n, i) for i, n in enumerate(COUNTS)])


def test_default_power_reproduces_the_pre_v15_1_weights():
    y = _y()
    torch.testing.assert_close(compute_class_weights(y, 3),
                               compute_class_weights(y, 3, power=1.0))


def test_power_zero_is_uniform():
    """power=0 must be exactly plain CE, not merely close to it."""
    w = compute_class_weights(_y(), 3, power=0.0)
    torch.testing.assert_close(w, torch.ones(3))


def test_power_monotonically_softens_the_correction():
    y = _y()
    ratios = [(compute_class_weights(y, 3, power=p)[1]
               / compute_class_weights(y, 3, power=p)[2]).item()
              for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(1.0)
    assert ratios[-1] == pytest.approx(COUNTS[2] / COUNTS[1], rel=1e-3)


@pytest.mark.parametrize("power", [0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
def test_weights_stay_mean_one_so_the_effective_lr_does_not_move(power):
    """The whole point of the mean-normalization: changing --class_weight_power
    must not silently rescale the loss (and with it the effective LR)."""
    assert compute_class_weights(_y(), 3, power=power).mean().item() == pytest.approx(1.0)


def test_negative_power_is_rejected():
    with pytest.raises(ValueError, match="power"):
        compute_class_weights(_y(), 3, power=-0.5)


@pytest.mark.parametrize("loss_type", ["weighted_ce", "focal_weighted"])
def test_power_reaches_the_criterion(loss_type):
    c = build_criterion(loss_type, y_train=_y(), num_classes=3, class_weight_power=0.5)
    torch.testing.assert_close(c.weight.cpu(), compute_class_weights(_y(), 3, power=0.5))


@pytest.mark.parametrize("loss_type", ["ce", "focal"])
def test_power_is_inert_for_the_unweighted_losses(loss_type):
    c = build_criterion(loss_type, y_train=_y(), num_classes=3, class_weight_power=0.25)
    assert getattr(c, "weight", None) is None


def test_weight_method_is_a_no_op_as_documented():
    """`balanced` and `inverse` are both proportional to 1/count, and the
    mean-normalization removes the constant - so they are the SAME weights.
    Documented in the --weight_method help; asserted here so a future change
    to either formula has to notice."""
    y = _y()
    torch.testing.assert_close(compute_class_weights(y, 3, method="balanced"),
                               compute_class_weights(y, 3, method="inverse"))
