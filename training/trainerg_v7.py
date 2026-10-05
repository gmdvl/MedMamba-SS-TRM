# -*- coding: utf-8 -*-
"""
training/trainerg_v7.py
========================
Stage E ("Generalization") on top of `training/trainerg_v6.py`
(TrainerG_v6): adds Phase 38 - Early Stopping.

TrainerG_v4's inherited `_check_stopping_rules()` (still used unchanged by
v5/v6) is heuristic divergence/stall detection (val loss+acc both getting
worse for N epochs in a row, or val_accuracy exactly repeating). That's a
useful safety net, but it isn't what Phase 38 asks for: "Use validation
macro-F1 (or balanced accuracy) ... with patience. best checkpoint =
highest validation macro-F1."

`TrainerG_v7` keeps the inherited heuristics AND adds real patience-based
early stopping on `self.checkpoint_metric` (the same metric Phase 13's
best-checkpoint selection already uses, from TrainerG_v5 - so "best
checkpoint" and "what early stopping watches" are guaranteed to be the same
metric, per Phase 38's own example).
"""

from __future__ import annotations

from typing import Optional

from training.trainerg_v6 import TrainerG_v6


class _StopReasonStr(str):
    """A minimal stand-in for `training.trainerg_v4.StopReason` members.
    `StopReason` is a closed `Enum` (can't be subclassed to add a new
    member without editing trainerg_v4.py), but every place that reads
    `self.stop_reason` only ever accesses `.value` - so a plain `str`
    subclass that also exposes `.value` (returning itself) is a fully
    compatible drop-in without touching trainerg_v4.py."""

    @property
    def value(self) -> str:
        return str(self)


class TrainerG_v7(TrainerG_v6):
    def __init__(self, *args, early_stop_patience: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop_patience = early_stop_patience
        self._epochs_without_improvement = 0

    def _check_stopping_rules(self) -> None:
        # Keep TrainerG_v4's divergence/stall heuristics as a safety net.
        super()._check_stopping_rules()

        if self.early_stop_patience is None or self.is_stopped or not self.history:
            return

        last = self.history[-1]
        # Phase 17 - an INVALID epoch (no optimizer step succeeded) shouldn't
        # count as "no improvement" against patience; it's not a real
        # attempt at improving, and shouldn't accelerate stopping training
        # that's otherwise progressing fine.
        if not last.get("is_valid_epoch", True):
            return

        candidate = last.get(self.checkpoint_metric)
        if candidate is None:
            return

        # NOTE: TrainerG_v5.fit() calls _check_stopping_rules() BEFORE it
        # updates self.best_metric_value for this epoch, so at this point
        # self.best_metric_value still holds the PREVIOUS best - this
        # comparison correctly asks "did THIS epoch improve on the prior best".
        if candidate > self.best_metric_value + 1e-9:
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self.early_stop_patience:
            self.stop_reason = _StopReasonStr(f"EARLY_STOPPED_PATIENCE_{self.checkpoint_metric.upper()}")
            self.is_stopped = True
            self.logger.info(
                f"Early stopping (Phase 38): no improvement in {self.checkpoint_metric!r} for "
                f"{self.early_stop_patience} epochs (best={self.best_metric_value:.4f}).")
