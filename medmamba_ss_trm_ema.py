# -*- coding: utf-8 -*-
#
# `EMAHelper` below is ported from `models/ema.py` of
# https://github.com/SamsungSAILMontreal/TinyRecursiveModels, which is distributed
# under the following licence. The notice is kept as that licence requires.
#
# MIT License
#
# Copyright (c) 2025. Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
medmamba_ss_trm_ema.py
==================
Exponential Moving Average of model weights, ported from
`SamsungSAILMontreal/TinyRecursiveModels`'s `models/ema.py` (the TRM
reference implementation). Not previously present anywhere in this repo
(`training/trainerg_v9.py` and its ancestors have no EMA/SWA path).

Used by `training/trainerg_v10.py` for the recursive (TRM-style) variant:
small medical datasets + deep supervision benefit from evaluating/
checkpointing an averaged copy of the weights rather than the raw
end-of-step weights (the TRM paper reports this as important for
stability). Generic - works on any `nn.Module`, not just
`MedMambaSSTRM`.
"""

from __future__ import annotations

import copy

import torch.nn as nn


class EMAHelper:
    """Tracks a shadow copy of every trainable parameter, updated as
    `shadow = mu * shadow + (1 - mu) * param` after every optimizer step."""

    def __init__(self, mu: float = 0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module: nn.Module) -> None:
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module) -> None:
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module: nn.Module) -> None:
        """In-place: overwrite `module`'s parameters with the shadow copy."""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module: nn.Module) -> nn.Module:
        """Returns a deep copy of `module` with EMA weights applied, leaving
        `module` itself untouched (use this for validation/checkpointing)."""
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = state_dict
