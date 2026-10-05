# -*- coding: utf-8 -*-
"""
training/extra_metrics.py
============================
Main Development Plan, Phase 48 - Final Evaluation Metrics requires Cohen's
Kappa and MCC (Matthews Correlation Coefficient). Neither is actually
computed anywhere in the trainer pipeline that's live during training:
`training/eval_utils.py` (what `TrainerG_v4`/`v5`/`v6`/`v7`'s
`_validate_one_epoch` calls every epoch) doesn't have them.
`training/metrics.py` DOES implement both (`cohen_kappa`,
`matthews_corrcoef_multiclass`) - but that module is a separate,
sklearn-free implementation used only by the standalone `evaluate.py`/
`training/evaluator.py` path, not by the per-epoch validation loop.

Rather than duplicate `training/metrics.py`'s pure-numpy implementations
here, this uses `sklearn.metrics` directly (sklearn is already a hard
dependency of `training/eval_utils.py`, so this adds nothing new).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef


def compute_extra_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"cohen_kappa": 0.0, "matthews_corrcoef": 0.0}
    return {
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }
