# -*- coding: utf-8 -*-
"""
training/spectral_checkpoint.py
==================================
Fixes a real OOM, not a hypothetical one: `SpectralPathway.forward()`
(`medmamba_ss_trm.py`) processes the batch's patchified spatial positions
(`N = batch_size * (H/patch_size) * (W/patch_size)`) in chunks of 1024 -
its own comment says this is "to limit peak memory". That's true for
INFERENCE. During TRAINING, autograd must retain every chunk's activations
simultaneously so `.backward()` has something to differentiate through -
the chunking loop discards nothing, it just changes when the memory is
allocated, not how much total memory is needed. Peak memory during
backward scales with the FULL `N`, not one chunk.

For an RGB image resized to 128x128 with `patch_size=4` (`medmamba_ss_tiny`,
`--architecture split`'s default for the `rgb` modality),
`N = batch_size * 32 * 32`. At `--batch_size 150`, `N = 153,600` - roughly
150 chunks' worth of activations all held at once. That is what actually
exhausted a 16GB GPU, not the 27M-parameter model itself (which is a
perfectly normal size).

This extends `training/grad_checkpoint.py`'s existing pattern (wrap at
runtime; `medmamba_ss_trm.py` stays untouched) to fix it:
`enable_spectral_gradient_checkpointing(model)` wraps
`SpectralPathway._process_patch_chunk` with
`torch.utils.checkpoint.checkpoint`, so each chunk's activations are
DISCARDED right after that chunk's forward pass and recomputed on demand
during backward, instead of every chunk being held in memory
simultaneously. This turns peak `SpectralPathway` memory from
`O(N)` into `O(chunk_size)` (a constant, 1024, regardless of batch size or
image resolution) - trading some recompute time for a large memory
reduction, exactly like `training/grad_checkpoint.py` already does for the
spatial backbone.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.utils.checkpoint as torch_ckpt


def enable_spectral_gradient_checkpointing(model: torch.nn.Module, strict: bool = False) -> int:
    """Finds every `SpectralPathway` inside `model` - there's exactly one
    per backbone (`model.backbone.spectral_pathway`), whether `model` is a
    plain `MedMambaSS`, `MedMambaSSFullChannel`, `MedMambaSSEfficient`, or one
    of this repo's reconstruction wrappers around any of those - and
    checkpoints its per-chunk processing in place. Returns how many were
    newly wrapped.

    v15 R3.3 - `strict=True` RAISES when the model has no `SpectralPathway` at
    all, instead of returning 0 and letting the caller print a reassuring line
    about a checkpointing pass that never happened. `strict=False` keeps the
    original silent-no-op convention. An ALREADY-wrapped pathway is not
    re-wrapped and so is not counted, which is why the strict check looks at
    whether a pathway was FOUND rather than at the return value."""
    n = 0
    found = 0
    for module in model.modules():
        if type(module).__name__ != "SpectralPathway" or not hasattr(module, "_process_patch_chunk"):
            continue
        found += 1
        if not getattr(module, "_spectral_gc_checkpointed", False):
            _wrap_spectral_pathway(module)
            n += 1
    if found == 0 and strict:
        raise ValueError(
            f"enable_spectral_gradient_checkpointing() found no SpectralPathway on a "
            f"{type(model).__name__}. Spectral gradient checkpointing was explicitly requested, "
            f"so this is an error rather than a silent no-op - pass strict=False if a no-op is "
            f"genuinely intended.")
    return n


def _wrap_spectral_pathway(pathway) -> None:
    if getattr(pathway, "_spectral_gc_checkpointed", False):
        return
    # Capture the UNBOUND class function, not `pathway._process_patch_chunk` (a
    # bound method): see training/grad_checkpoint.py for the full rationale. A
    # bound method would survive copy.deepcopy still pointing at the pre-copy
    # CPU pathway instance, causing a CPU/CUDA device mismatch on the copy.
    orig = type(pathway)._process_patch_chunk

    def checkpointed_process_chunk(self, band_values: torch.Tensor,
                                    wl: Optional[torch.Tensor], sensor_range):
        if not self.training:
            return orig(self, band_values, wl, sensor_range)

        # torch.utils.checkpoint threads TENSOR positional args through its
        # save/recompute machinery; `wl`/`sensor_range` are small, shared
        # across every chunk in one forward call (not per-chunk data), so
        # they're closed over rather than passed through checkpoint's arg list.
        def run(bv: torch.Tensor):
            return orig(self, bv, wl, sensor_range)

        return torch_ckpt.checkpoint(run, band_values, use_reentrant=False)

    pathway._process_patch_chunk = types.MethodType(checkpointed_process_chunk, pathway)
    pathway._spectral_gc_checkpointed = True
