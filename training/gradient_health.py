# -*- coding: utf-8 -*-
"""
training/gradient_health.py
============================
Main Development Plan, Phase 16 - Fix Gradient/Training Instability, and
Phase 17 - Invalid Epoch Protection.

Tracks, per epoch:
    total_batches
    valid_updates
    skipped_updates
    nonfinite_loss
    nonfinite_gradients
    skip_ratio  (= skipped_updates / total_batches)

An epoch with `valid_updates == 0` is INVALID (Phase 17): the caller
(training/trainerg_v5.py) is responsible for making sure an invalid epoch
never becomes the best epoch, never overwrites best_model.pt, and is
excluded from early-stopping bookkeeping.

--------------------------------------------------------------------------
v12/v13 stability plan, Phase 12 - extended (additive, backward compatible)
--------------------------------------------------------------------------
`record_batch(...)` gained OPTIONAL keyword arguments so every existing
caller (`TrainerG_v5`/`v6`, which only ever pass the original three
positional/keyword args) keeps working unmodified, while a caller that
wants Phase 12's parameter-level detail (which parameter went non-finite
first, max/min |gradient|, nan/inf element counts) can now get it recorded
alongside the plain counters. See `training/numerical_stability.py` for the
richer skip-ratio CLASSIFICATION (healthy/warning/degraded/critical/
invalid/fatal) and persistent-failure abort policy built on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EpochGradientHealth:
    total_batches: int = 0
    valid_updates: int = 0
    skipped_updates: int = 0
    nonfinite_loss: int = 0
    nonfinite_gradients: int = 0

    # Phase 12 - optional parameter-level detail (None until first populated)
    first_bad_parameter: Optional[str] = None
    max_gradient_norm: Optional[float] = None
    min_gradient_norm: Optional[float] = None
    gradient_nan_elements: int = 0
    gradient_inf_elements: int = 0

    @property
    def skip_ratio(self) -> float:
        return self.skipped_updates / self.total_batches if self.total_batches else 0.0

    @property
    def is_valid_epoch(self) -> bool:
        # Phase 17 - an epoch in which every batch's loss/gradient was
        # non-finite (so the optimizer never actually stepped) is invalid.
        return self.valid_updates > 0

    def as_dict(self) -> Dict:
        return {
            "total_batches": self.total_batches,
            "valid_updates": self.valid_updates,
            "skipped_updates": self.skipped_updates,
            "nonfinite_loss": self.nonfinite_loss,
            "nonfinite_gradients": self.nonfinite_gradients,
            "skip_ratio": round(self.skip_ratio, 4),
            "is_valid_epoch": self.is_valid_epoch,
            "first_bad_parameter": self.first_bad_parameter,
            "max_gradient_norm": self.max_gradient_norm,
            "min_gradient_norm": self.min_gradient_norm,
            "gradient_nan_elements": self.gradient_nan_elements,
            "gradient_inf_elements": self.gradient_inf_elements,
        }


class GradientHealthTracker:
    """Call `start_epoch()` once per epoch, `record_batch(...)` once per
    training batch (whether or not the optimizer actually stepped), and
    `finish_epoch()` to get + store that epoch's health summary."""

    def __init__(self):
        self.history: List[Dict] = []
        self._cur = EpochGradientHealth()

    def start_epoch(self) -> None:
        self._cur = EpochGradientHealth()

    def record_batch(self, loss_is_finite: bool, grad_is_finite: bool, optimizer_stepped: bool,
                      grad_norm: Optional[float] = None,
                      first_bad_parameter: Optional[str] = None,
                      gradient_nan_elements: int = 0,
                      gradient_inf_elements: int = 0) -> None:
        self._cur.total_batches += 1
        if not loss_is_finite:
            self._cur.nonfinite_loss += 1
        if not grad_is_finite:
            self._cur.nonfinite_gradients += 1
            if self._cur.first_bad_parameter is None and first_bad_parameter is not None:
                self._cur.first_bad_parameter = first_bad_parameter
            self._cur.gradient_nan_elements += gradient_nan_elements
            self._cur.gradient_inf_elements += gradient_inf_elements
        if optimizer_stepped:
            self._cur.valid_updates += 1
        else:
            self._cur.skipped_updates += 1

        if grad_norm is not None:
            import math
            if math.isfinite(grad_norm):
                self._cur.max_gradient_norm = grad_norm if self._cur.max_gradient_norm is None \
                    else max(self._cur.max_gradient_norm, grad_norm)
                self._cur.min_gradient_norm = grad_norm if self._cur.min_gradient_norm is None \
                    else min(self._cur.min_gradient_norm, grad_norm)

    def finish_epoch(self) -> Dict:
        d = self._cur.as_dict()
        self.history.append(d)
        return d
