# -*- coding: utf-8 -*-
"""
training/grad_checkpoint.py
============================
Optional gradient checkpointing (`--grad_checkpointing`), trading compute
for activation memory on large patches / deep configs.

`medmamba_ss_trm.py` is architecture-only and intentionally untouched, so this
wraps each `GStage.forward` at runtime with
`torch.utils.checkpoint.checkpoint` instead of editing the source - the
model's forward pass is unaffected when the flag is off, and this has no
effect on `evaluate.py`/eval mode (checkpointing only helps when gradients
are being computed).

Call `enable_gradient_checkpointing(model)` once, right after building the
model and before wrapping in DDP/compile.
"""

from __future__ import annotations

import types

import torch
import torch.utils.checkpoint as torch_ckpt


def enable_gradient_checkpointing(model, strict: bool = False) -> int:
    """Wraps every `GStage` in the backbone. Returns how many modules were
    wrapped.

    v15 R3.3 - two problems with the original version, both fixed here:

      * `MedMambaSSTRM` has no `.backbone.stages`, so this returned 0 and
        printed "wrapped 0 backbone stage(s)" while the preflight cheerfully
        reported `gradient_checkpointing: True`. The recursive backbone IS
        checkpointable - `RecursiveCore.f` does it per call - so this now
        reaches it through `cfg.trm_checkpoint_core` and counts the core.
      * a silent no-op on a flag the caller explicitly passed is how that went
        unnoticed for 19.7 GPU-hours. `strict=True` (what
        `train_example_v15.py` passes) RAISES when nothing was wrapped.
    """
    backbone = getattr(model, "backbone", None)
    stages = getattr(backbone, "stages", None)
    if stages is None:
        n = _enable_recursive_core_checkpointing(model)
        if n == 0 and strict:
            raise ValueError(
                "enable_gradient_checkpointing() found nothing to wrap on a "
                f"{type(model).__name__}: it has neither `.backbone.stages` (the hierarchical "
                "architectures) nor a `RecursiveCore` (--architecture recursive). Gradient "
                "checkpointing was explicitly requested, so this is an error rather than a "
                "silent no-op - pass strict=False if a no-op is genuinely intended.")
        return n
    n = 0
    for stage in stages:
        if getattr(stage, "_gc_checkpointed", False):
            continue
        # Capture the UNBOUND class function, not `stage.forward` (a bound
        # method). A bound method pins this exact pre-copy instance; after
        # copy.deepcopy(model) the copy's patched forward would still call into
        # the ORIGINAL stage - which .to(device) never moved - causing a
        # CPU/CUDA device mismatch. The unbound function re-binds to whatever
        # `self` the MethodType carries, so it follows the deepcopy.
        orig_forward = type(stage).forward
        stage.forward = types.MethodType(
            lambda self, x, ctx_map, _orig=orig_forward: (
                torch_ckpt.checkpoint(_orig, self, x, ctx_map, use_reentrant=False)
                if self.training and (x.requires_grad or ctx_map.requires_grad)
                else _orig(self, x, ctx_map)
            ),
            stage,
        )
        stage._gc_checkpointed = True
        n += 1
    return n


def _enable_recursive_core_checkpointing(model) -> int:
    """v15 R3.3 - `medmamba_ss_trm.RecursiveCore.f` already gradient-checkpoints each
    application of the shared core; since v15 that is gated on
    `MedMambaSSTRMConfig.trm_checkpoint_core`. Turning the flag on here is the
    recursive architecture's equivalent of wrapping the hierarchical backbone's
    stages. Returns the number of cores switched on (they may already be on,
    which still counts - the requested state is reached either way)."""
    n = 0
    for module in model.modules():
        if type(module).__name__ == "RecursiveCore" and hasattr(module, "checkpoint_core"):
            module.checkpoint_core = True
            cfg = getattr(module, "cfg", None)
            if cfg is not None and hasattr(cfg, "trm_checkpoint_core"):
                cfg.trm_checkpoint_core = True
            n += 1
    return n
