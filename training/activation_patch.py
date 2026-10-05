# -*- coding: utf-8 -*-
"""
training/activation_patch.py
=============================
Main Development Plan, Phase 19 - Replace ReLU with LeakyReLU, and
Phase 20 - Do Not Replace SiLU Blindly.

`medmamba_ss_trm.py` is architecture-only and intentionally untouched by the rest
of this training framework (see `training/grad_checkpoint.py`'s docstring
for the same design choice re: gradient checkpointing) - so this swaps
`nn.ReLU` -> `nn.LeakyReLU(negative_slope)` IN PLACE, post-construction,
rather than editing the source.

Only plain `nn.ReLU` modules are touched (that's `GBlock.conv_branch`'s
three ReLUs, the spatial/conv branch per Phase 20's "Conv/spectral branch ->
LeakyReLU, SSM branch -> SiLU" first experiment). `nn.SiLU` (the SSM path)
and `nn.GELU` (tokenizer/FFN/spectral encoder) are left untouched.

For the Phase 21 "LeakyReLU everywhere" follow-up ablation, pass
`also_silu=True` to additionally swap SiLU -> LeakyReLU (GELU is still left
alone, since it isn't the activation the plan names).
"""

from __future__ import annotations

import torch.nn as nn


def replace_relu_with_leakyrelu(model: nn.Module, negative_slope: float = 0.01,
                                 also_silu: bool = False, verbose: bool = True) -> int:
    """Returns the number of activation modules replaced."""
    targets = (nn.ReLU, nn.SiLU) if also_silu else (nn.ReLU,)
    n_replaced = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, targets):
                inplace = getattr(child, "inplace", False)
                setattr(module, child_name, nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))
                n_replaced += 1
    if verbose:
        kind = "nn.ReLU + nn.SiLU" if also_silu else "nn.ReLU"
        print(f"[activation-patch] replaced {n_replaced} {kind} module(s) with "
              f"nn.LeakyReLU(negative_slope={negative_slope})"
              + ("" if also_silu else " (SiLU/GELU left untouched, Phase 20)"))
    return n_replaced
