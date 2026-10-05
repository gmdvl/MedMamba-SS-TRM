# -*- coding: utf-8 -*-
"""
training/torch_compile.py
============================
Speeds up the two hot loops of the `recursive` architecture without touching
`medmamba_ss_trm.py`, using the same runtime-wrap pattern as
`training/grad_checkpoint.py` and `training/spectral_checkpoint.py`.

Where the time actually goes (RTX 5060 Ti, bf16, batch 256, 11x11x32 HSI
patches, `--architecture recursive` with the v15 defaults - measured, not
estimated):

    SpectralPathway + stem   1733 ms/step   (61%)
    RecursiveCore + head     1140 ms/step   (39%)
    ---------------------------------------------
    full training step       2873 ms/step   (89.5 samples/s)

The spectral half dominates because `medmamba_ss_trm._selective_scan_pure_pytorch`
is a *Python* loop over the sequence dimension - and in `SpectralPathway` that
dimension is the BAND count, so every call runs 32 iterations of four small
`einsum`s. `spectral_depth=3` SS2D layers x `ceil(N / spectral_chunk_size)`
chunks (31 for a 256-patch batch at `patch_size=1`) x 32 steps is ~3,000
sequential micro-kernel groups per forward, each far too small to saturate the
GPU, and `--spectral_checkpointing auto` runs all of it a second time during
backward. Raising `--spectral_chunk_size` does not help (measured: 1024 is the
optimum; 4096 and 16384 are 1.19x and 1.37x SLOWER because the chunk stops
fitting in cache), and neither does raising `--batch_size` (89.5 / 80.2 / 78.6
samples/s at 256 / 512 / 1024) - the loop is bandwidth- and launch-bound, not
occupancy-bound.

`torch.compile` is the right tool for exactly this shape of problem: Inductor
fuses each scan iteration's einsums into one kernel and keeps the running
`x_state` in registers across the fused region. Measured on the same machine:

    scan compiled                    1733 -> 843 ms   (2.06x on the spectral half)
    scan + core compiled             2873 -> 1412 ms  (2.03x overall, 181 samples/s)

Numerics are unchanged: max relative error vs the eager scan is 1.7e-07 in
fp32 (`_SelectiveScanParams.run` upcasts to fp32 before calling the backend,
so this is the path that matters), well inside float roundoff.

Compilation is OPT-IN (`--compile off` by default) so no existing run's
numbers move without the caller asking. Both helpers degrade to a warning and
a no-op if compilation is unavailable or fails - a `torch.compile` failure
must never cost a multi-hour run.
"""

from __future__ import annotations

import types
from typing import Optional

import torch

import medmamba_ss_trm
from medmamba_ss_trm import RecursiveCore, _SelectiveScanParams


# One compiled callable shared by every `_SelectiveScanParams` instance: the
# scan is a pure function, so compiling it once and reusing the artifact keeps
# us to a single set of Inductor kernels for the whole model.
_COMPILED_SCAN = None
_COMPILED_F_FORWARD = None


def _compile(fn, mode: Optional[str] = None):
    # `dynamic=False` (shape-specialized), NOT the default automatic-dynamic.
    # This is measured, not stylistic. `SpectralPathway` feeds the scan a full
    # `spectral_chunk_size` chunk plus one REMAINDER chunk of `N % chunk_size`,
    # and `N = batch * Hp * Wp` changes on the last, short batch of an epoch -
    # so the scan sees a handful of distinct lengths per run. Under automatic
    # dynamic shapes Dynamo reacts to the second length by recompiling the
    # region with a SYMBOLIC sequence dimension, and a symbolic-length kernel
    # cannot unroll or specialize this loop: measured end-to-end over a step
    # sequence ending in a short batch, that path ran at 61.3 samples/s against
    # 70.6 eager - a 13% REGRESSION, the opposite of the intent. Specializing
    # per shape costs a few extra Inductor compiles (seconds each, once) and
    # keeps the fast kernels. `cache_size_limit` is raised in
    # `enable_torch_compile` to leave room for them.
    return torch.compile(fn, mode=mode, dynamic=False)


def enable_compiled_scan(model: torch.nn.Module, mode: Optional[str] = None) -> int:
    """Point every `_SelectiveScanParams.scan_fn` in `model` at a compiled
    build of the pure-PyTorch selective scan. Returns how many were rebound.

    Note this rebinds the per-INSTANCE attribute rather than mutating
    `medmamba_ss_trm.SCAN_BACKENDS`: `_SelectiveScanParams.__init__` resolves
    `get_backend(backend)` once and stores the result, so a registry swap after
    the model is built would silently do nothing.
    """
    global _COMPILED_SCAN

    targets = [m for m in model.modules() if isinstance(m, _SelectiveScanParams)]
    if not targets:
        return 0
    if _COMPILED_SCAN is None:
        _COMPILED_SCAN = _compile(medmamba_ss_trm._selective_scan_pure_pytorch, mode=mode)
    n = 0
    for m in targets:
        # Only the pure-PyTorch backend is ours to compile. A caller who
        # registered a real CUDA/Triton kernel via `register_scan_backend` has
        # already solved this problem better than Inductor can.
        if m.scan_fn is medmamba_ss_trm._selective_scan_pure_pytorch:
            m.scan_fn = _COMPILED_SCAN
            n += 1
    return n


def enable_compiled_recursive_core(model: torch.nn.Module, mode: Optional[str] = None) -> int:
    """Compile `RecursiveCore._f_forward` - the `trm_core_layers`-deep stack
    that `RecursiveCore.f` calls `trm_n_improve * (trm_n_latent + 1)` times per
    segment. Returns how many cores were compiled.

    `_f_forward` (not `f`) is the compile target on purpose: `f` wraps it in
    `torch.utils.checkpoint` under `--trm_checkpoint_core`, and compiling the
    inside of the checkpoint keeps both the recompute and the backward on the
    fused kernels.

    The UNBOUND function is compiled once and re-bound per instance with
    `types.MethodType`. Compiling the bound method instead
    (`core._f_forward = torch.compile(core._f_forward)`) is a silent
    correctness bug, not a style preference: the resulting object holds a
    reference to the original `core`, and `copy.deepcopy` treats it as opaque
    rather than re-binding it. `EMAHelper` deep-copies the model every epoch to
    validate the EMA weights, so the copy's core would have kept calling the
    LIVE core - i.e. every validation metric, checkpoint selection and
    early-stopping decision in the run would have been computed from the wrong
    weights. A `types.MethodType` goes through `copy._deepcopy_method`, which
    rebinds `__self__` to the copy and shares the compiled `__func__`;
    verified by perturbing a copy's core and confirming only the copy's output
    moves.
    """
    global _COMPILED_F_FORWARD

    cores = [m for m in model.modules() if isinstance(m, RecursiveCore)]
    if not cores:
        return 0
    if _COMPILED_F_FORWARD is None:
        _COMPILED_F_FORWARD = _compile(RecursiveCore._f_forward, mode=mode)
    for core in cores:
        core._f_forward = types.MethodType(_COMPILED_F_FORWARD, core)
    return len(cores)


def enable_torch_compile(model: torch.nn.Module, mode: Optional[str] = None) -> str:
    """Both of the above, tolerant of failure. Returns a one-line summary."""
    if not hasattr(torch, "compile"):
        return "unavailable (this torch has no torch.compile)"
    try:
        n_scan = enable_compiled_scan(model, mode=mode)
        n_core = enable_compiled_recursive_core(model, mode=mode)
    except Exception as e:                                    # pragma: no cover
        return f"FAILED ({type(e).__name__}: {e}) - continuing uncompiled"
    # `_compile` specializes per shape, so raise Dynamo's default limit of 8:
    # the scan sees a full chunk, this batch's remainder chunk, and both of
    # those again for the short last batch of an epoch, plus whatever the
    # validation loader's batch shape adds. Exceeding the limit silently falls
    # back to eager for the rest of the run, which is exactly the failure this
    # flag exists to avoid.
    torch._dynamo.config.cache_size_limit = max(
        32, int(getattr(torch._dynamo.config, "cache_size_limit", 8)))
    return (f"compiled {n_scan} selective scan(s) and {n_core} recursive core(s) "
            f"(mode={mode or 'default'}); first step pays the Inductor compile")
