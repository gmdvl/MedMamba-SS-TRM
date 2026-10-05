# -*- coding: utf-8 -*-
"""
training/recursive_features.py
=================================
MedMamba-SS-TRM v16 plan, Stage 2.1 (R-2) - the reconstruction decoder must read a
LIVE feature map, not a detached one.

`medmamba_ss_trm.py` is FROZEN (every trainer imports it). The fix needs no patch
to it at all: `MedMambaSSTRM.forward_deep_supervision`
(`medmamba_ss_trm.py:1942-1982`) is built entirely from PUBLIC attributes -
`backbone.embed`, `backbone.core.{init_states, one_segment}`, `model.head`,
`model.cfg` - so this module duplicates that loop as a free function and
returns the segment-recursion's LIVE final feature map as a third value.

Why the frozen method detaches (`medmamba_ss_trm.py:1981`,
`self.last_feature_map = y.detach()`): it's a plain module attribute, and
`EMAHelper.ema_copy` calls `copy.deepcopy(module)` every epoch to build the
EMA validation copy - `copy.deepcopy` refuses a tensor that is still part of
a live autograd graph. Storing `y` there un-detached would break EMA. This
module sidesteps that entirely: nothing is stored on the module, `y` is
returned directly from the call and consumed (decoded) within the SAME
forward, so a caller that needs reconstruction gradient into the core no
longer has to choose between EMA-safety and a live feature map.

Drift risk, and its guard: this duplicates ~25 lines of ACT/halting logic.
`test_recursive_features_parity.py` asserts that, in eval mode with a fixed
seed, `forward_deep_supervision_with_features(m, x)[:2]` is elementwise equal
to `m.forward_deep_supervision(x)`. If someone edits the frozen loop, that
test fails loudly instead of the two silently diverging.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def forward_deep_supervision_with_features(model, x: torch.Tensor,
                                            wavelengths: Optional[torch.Tensor] = None,
                                            sensor_range: Optional[Tuple[float, float]] = None):
    """`MedMambaSSTRM.forward_deep_supervision`, returning the LIVE final
    feature map `y` as a third value. `model` is a `medmamba_ss_trm.MedMambaSSTRM`
    (or anything exposing the same public surface: `.backbone.embed`,
    `.backbone.core`, `.head`, `.cfg`).

    Returns `(logits_list, q_list, y)` - `logits_list`/`q_list` are
    elementwise-identical (in eval mode) to `model.forward_deep_supervision(x,
    ...)`'s own return value; `y` is `[B, H, W, d]`, channels-last, STILL
    ATTACHED to the graph that produced it.
    """
    x_emb, _ctx_map, _band_weights = model.backbone.embed(x, wavelengths, sensor_range)
    core = model.backbone.core
    y, z = core.init_states(x_emb)
    n = model.cfg.trm_deep_supervision_steps
    logits_list, q_list = [], []

    tau = model.cfg.trm_halt_threshold if model.cfg.trm_act_halting else None
    explore = bool(model.training and tau is not None
                   and float(torch.rand(())) < model.cfg.trm_halt_exploration_prob)

    for i in range(n):
        y, z = core.one_segment(x_emb, y, z)
        logits, q = model.head(y)
        logits_list.append(logits)
        q_list.append(q)
        if i < n - 1:
            y, z = y.detach(), z.detach()
            if tau is not None and not explore and bool((torch.sigmoid(q) > tau).all()):
                break

    return logits_list, q_list, y
