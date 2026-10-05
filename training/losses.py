# -*- coding: utf-8 -*-
"""
training/losses.py
===================
Main Development Plan, Phase 12 - Test Classification Strategies (options
B and C; option A "Standard Cross Entropy" is just `nn.CrossEntropyLoss()`,
already the implicit baseline everywhere in this repo).

  B. Weighted Cross Entropy - `nn.CrossEntropyLoss(weight=...)` using
     training-set class frequencies.
  C. Focal Loss - down-weights easy/majority examples so gradient signal
     concentrates on hard/underrepresented ones.

Per the plan: "Do not combine all strategies immediately. Identify which
intervention actually solves class collapse." `build_criterion()` is
therefore deliberately a single-choice factory (one loss type at a time),
not a way to stack every option together.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(y_train: np.ndarray, num_classes: int, method: str = "balanced",
                           power: float = 1.0) -> torch.Tensor:
    """`method="balanced"`: sklearn's `class_weight='balanced'` formula,
    `n_samples / (n_classes * class_count)`. `method="inverse"`: plain
    `1 / class_count`. Either way the result is re-normalized to mean 1.0
    across classes present, so switching --loss doesn't also silently
    rescale the overall loss magnitude (and therefore the effective
    learning rate) relative to plain CE.

    NOTE both methods are proportional to `1 / class_count`, so after the
    mean-normalization below they produce IDENTICAL weights. `--weight_method`
    is a no-op knob; `power` is the one that actually changes the strength of
    the correction.

    `power` (v15 R2.4) interpolates between no correction and the full one:
    `w ** power`, renormalized. 0.0 = uniform (plain CE), 1.0 = the classic
    full inverse-frequency weighting, 0.5 = the "square-root inverse
    frequency" that the long-tail literature generally prefers over 1.0.

    Full weighting is not automatically the right answer, and on this repo's
    HSI dataset it demonstrably is not. Train counts are healthy 722,358 /
    DCIS 122,850 / IDC 1,606,878, which at `power=1.0` hands DCIS 13.1x the
    weight of IDC and 5.9x that of healthy. The v15 recursive run trained that
    way reached DCIS recall 0.946 at precision 0.456 on the held-out test
    split - it labels 32.4% of all genuinely healthy patches DCIS - and
    macro-F1 stalls at 0.804 because of that one cell. Re-deciding that same
    run's saved test probabilities under a softer effective weighting (which
    is a lower bound on what training at that weighting would do, since the
    decision boundary itself never moved) gives:

        power  1.00 (as trained)   macro-F1 0.8038   DCIS F1 0.615
        power  0.75                macro-F1 0.8474   DCIS F1 0.672
        power  0.50                macro-F1 0.8391   DCIS F1 0.625
        power  0.00 (plain CE)     macro-F1 0.6657   DCIS F1 0.126

    So the correction is real and necessary - plain CE collapses the minority
    class - but 1.0 overshoots it.
    """
    counts = np.bincount(np.asarray(y_train), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)  # avoid div-by-zero for an absent class; Phase 3 should catch this upstream

    if method == "inverse":
        weights = 1.0 / counts
    elif method == "balanced":
        weights = counts.sum() / (num_classes * counts)
    else:
        raise ValueError(f"Unknown class-weight method: {method!r} (use 'balanced' or 'inverse')")

    if power != 1.0:
        if power < 0:
            raise ValueError(f"class-weight power must be >= 0, got {power!r}")
        # Applied to the raw 1/count weights, BEFORE the mean-normalization, so
        # the result is still mean-1 and the effective learning rate does not
        # move with `power`.
        weights = weights ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017):
    FL(p_t) = -(1 - p_t)^gamma * log(p_t), optionally combined with a
    per-class `weight` (so focal loss and class weighting CAN be combined
    within this one class if you deliberately want that combined
    experiment - but `build_criterion()` below keeps them as separate
    single-variable options by default, per Phase 12's guidance)."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None, persistent=False)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(log_probs, targets, weight=self.weight, reduction="none")
        pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(loss_type: str, y_train: Optional[np.ndarray] = None, num_classes: Optional[int] = None,
                     focal_gamma: float = 2.0, weight_method: str = "balanced",
                     device: str = "cpu", class_weight_power: float = 1.0) -> nn.Module:
    """`loss_type`: 'ce' | 'weighted_ce' | 'focal' | 'focal_weighted'.
    `y_train`/`num_classes` are required for anything involving weights.
    `class_weight_power` softens the weighting - see `compute_class_weights`."""
    weight = None
    if loss_type in ("weighted_ce", "focal_weighted"):
        if y_train is None or num_classes is None:
            raise ValueError(f"loss_type={loss_type!r} needs y_train and num_classes to compute class weights")
        weight = compute_class_weights(y_train, num_classes, method=weight_method,
                                        power=class_weight_power).to(device)

    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weight)
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma)
    if loss_type == "focal_weighted":
        return FocalLoss(gamma=focal_gamma, weight=weight)
    raise ValueError(f"Unknown loss_type: {loss_type!r} (use ce | weighted_ce | focal | focal_weighted)")
