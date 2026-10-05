# -*- coding: utf-8 -*-
"""
training/trainerg_v8.py
========================
Stage F ("Scientific Validation") on top of `training/trainerg_v7.py`
(TrainerG_v7): adds Cohen's Kappa and MCC to the per-epoch validation
metrics dict (Phase 48 - see `training/extra_metrics.py` for why these were
missing from the live training loop despite existing elsewhere in the
repo).

This is a minimal one-method override BECAUSE `TrainerG_v7`'s
`_validate_one_epoch` (inherited unchanged from `TrainerG_v6`) already
returns `preds`/`labels` alongside its `classification` metrics dict - no
need to duplicate the whole validation loop again (unlike earlier
`TrainerG_vN` subclasses, which had to re-copy the loop because the thing
being changed - the loss function, gradient handling - was computed
INSIDE the loop body). Here the change is purely additive post-processing.
"""

from __future__ import annotations

import torch

from training.trainerg_v7 import TrainerG_v7
from training.extra_metrics import compute_extra_metrics


class TrainerG_v8(TrainerG_v7):
    @torch.no_grad()
    def _validate_one_epoch(self):
        result = super()._validate_one_epoch()
        result["classification"].update(compute_extra_metrics(result["labels"], result["preds"]))
        return result
