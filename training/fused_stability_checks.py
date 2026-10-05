# -*- coding: utf-8 -*-
"""
training/fused_stability_checks.py
==============================
MedMamba-SS-TRM v17, Stage 5. Fused replacements for
`training.numerical_stability.inspect_gradients` / `check_parameters_finite`, which are in
a FROZEN file and cannot be edited.

The problem
-----------
`inspect_gradients` (`numerical_stability.py:99-117`) runs, per parameter tensor, on the
HEALTHY path:

    bool(torch.isfinite(g).all())   -> 1 device-to-host sync
    float(g.abs().max().item())     -> 1 sync
    float(g.abs().min().item())     -> 1 sync

and `check_parameters_finite` (`:125-134`) adds a fourth. The MedMamba-SS-TRM model has **73
parameter tensors**, so a single training step performs **292 full pipeline drains** plus
~300 tiny kernels. Its own docstring says "one boolean reduction per parameter tensor";
the implementation does four, and none of them is batched.

Measured context (`plan/v17_baseline.md`): the PAD step is 0.852 s against 0.437 s for
forward+backward+`opt.step()` alone, and ~19.4 s of a 43.4 s epoch is spent inside the
training loop but outside the optimiser - roughly 380 ms per step.

The fix
-------
`torch._foreach_norm` reduces every gradient in ONE fused kernel launch; stacking the
result gives a single small tensor that answers all three questions with **one** sync.

This is not an approximation of the old check, it is the same check:
`isfinite(norm)` is False exactly when the tensor contains a NaN or an Inf, because both
propagate through the L2 reduction. What it does NOT give for free is *which* parameter
and *how many* elements - so the per-tensor walk is kept verbatim and runs only on the
failure branch, where a handful of extra syncs cost nothing and the diagnostic is the
entire point.

`max_abs_gradient` / `min_abs_gradient` change meaning slightly and deliberately: they were
the max/min absolute ELEMENT across all gradients; they are now the max/min per-tensor
NORM. Both are logged diagnostics only - `training.gradient_health` records them, nothing
branches on them - and the norm is arguably the more useful of the two for spotting a
layer that is exploding or dead. `GradientInspection` is returned unchanged, so every
downstream consumer keeps working.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from training.numerical_stability import (        # frozen - imported, not edited
    GradientInspection, inspect_gradients as _inspect_gradients_per_tensor,
    check_parameters_finite as _check_parameters_finite_per_tensor,
)


# `torch._foreach_norm` is a private API, and `torch.stack` refuses a mixed-dtype list.
# Either would raise at the FIRST optimizer step and take a multi-hour run with it, so
# both helpers below degrade to the frozen per-tensor implementation instead of failing.
# Same policy as `training/torch_compile.py` and `training/prep_progress.py`: an
# optimisation must never be what kills a run.
_FUSED_OK = hasattr(torch, "_foreach_norm")
_WARNED = False


def _fused_norms(tensors):
    """Per-tensor L2 norms as ONE stacked tensor, or None if this torch/dtype mix cannot."""
    global _FUSED_OK, _WARNED
    if not _FUSED_OK:
        return None
    try:
        return torch.stack(torch._foreach_norm(tensors))
    except Exception as e:                                   # pragma: no cover
        _FUSED_OK = False
        if not _WARNED:
            _WARNED = True
            print(f"[v17 S5] fused gradient-health check unavailable ({type(e).__name__}: {e}); "
                  f"falling back to the per-tensor path for the rest of this run. Correct, "
                  f"just slower.", flush=True)
        return None


def inspect_gradients_fused(model: torch.nn.Module) -> GradientInspection:
    """One sync instead of 3 per parameter tensor. Same `GradientInspection` contract."""
    named = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    if not named:
        return GradientInspection(all_finite=True)

    norms = _fused_norms([g.detach() for _n, g in named])
    if norms is None:
        return _inspect_gradients_per_tensor(model)
    if bool(torch.isfinite(norms).all()):                    # THE one sync
        return GradientInspection(
            all_finite=True,
            max_abs_gradient=float(norms.max()),
            min_abs_gradient=float(norms.min()),
        )

    # Failure path: fall back to the frozen per-tensor walk for `first_bad_parameter`,
    # `nan_count` and `inf_count`. A run is aborting here; syncs are free.
    return _inspect_gradients_per_tensor(model)


def check_parameters_finite_fused(model: torch.nn.Module) -> Tuple[bool, List[str]]:
    """Post-`optimizer.step()` NaN guard, one sync instead of one per tensor."""
    params = [p.detach() for p in model.parameters()]
    if not params:
        return True, []
    norms = _fused_norms(params)
    if norms is None:
        return _check_parameters_finite_per_tensor(model)
    if bool(torch.isfinite(norms).all()):                    # THE one sync
        return True, []
    bad = [n for n, p in model.named_parameters()
           if not bool(torch.isfinite(p.detach()).all())]
    return False, bad
