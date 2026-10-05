# -*- coding: utf-8 -*-
"""
training/flops_report.py
==============================
v18 F2 - the FLOP accounting that has never once produced a number.

The defect
----------
`training/trainerg_v11.py:717` calls

    report["flops"] = count_flops(self.model, tuple(sample_x.shape[1:]), ...)

`sample_x` is `[B, C, H, W]`, so `shape[1:]` is the 3-tuple `(C, H, W)`.
`training.flops_counter.count_flops` then builds `torch.randn(*input_shape)` -
a 3-D tensor - and the model's forward raises

    ValueError: not enough values to unpack (expected 4, got 3)

which the `except Exception` around the call turns into the stored string
`{"error": "ValueError: ..."}`. EVERY `test_report.json` in this repository
carries that error and no run has ever reported FLOPs, which is why the whole
of manuscript section 7.6 - the cost inversion, the paper's most transferable
result - is tagged *(prior)* rather than *(verified)*.

`training/trainerg_v11.py` and `training/flops_counter.py` are both on
`tests/test_frozen_files_untouched.py:FROZEN_GLOBS`. Neither is edited. The
arithmetic in `flops_counter` is correct and is imported and reused verbatim;
only the *call* was wrong, and this module is the corrected caller. It is
imported by BOTH `scripts/flops_report.py` (retro-fit over finished runs)
and `TrainerG_v18` (every new run), so there is one implementation.

What is counted
---------------
Two terms, reported separately because they are measured differently and a
reader is entitled to know which is which:

`conv_linear` - exact, from `flops_counter`'s forward hooks on every
`nn.Conv1d`/`nn.Conv2d`/`nn.Linear`, at the corrected 4-D input shape.

`selective_scan` - the term `flops_counter` documents as excluded. It is
NOT estimated from hand-picked constants here. Every
`medmamba_ss_trm._SelectiveScanParams` instance holds its scan backend as a plain
per-instance attribute (`self.scan_fn`, bound at construction), so it can be
wrapped for the duration of one forward pass and the REAL `[B, K, D, L]`
shape read off each actual call. The per-call arithmetic is then the frozen
`flops_counter.estimate_selective_scan_flops`. This matters for the recursive
variant specifically: with `trm_mixer="mlp"` the backbone has no scan at all
and the only selective scan in the model is `SpectralMamba`'s, at
`k_directions=2` - a fact no analytical guess would have got right.

`core_applications` is recorded alongside, because manuscript section 7.6's
whole argument is that a parameter count without it is half a result:
`(trm_n_latent + 1) * trm_n_improve * trm_deep_supervision_steps`, which is
63 at the reported configuration and is paid at inference as well as in
training.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

# Frozen, imported for reuse - not reimplemented.
from training.flops_counter import count_flops, estimate_selective_scan_flops


def unwrap_model(model: nn.Module) -> nn.Module:
    """The module whose `forward` the counter should call.

    `torch.compile` wraps in `OptimizedModule._orig_mod`. Counting through the
    compiled wrapper works but reports the dynamo graph module's name, so peel
    it. The reconstruction wrapper's `base_model` is deliberately NOT peeled:
    with `--recon_mode latent` the decoder runs on every forward and its
    arithmetic is part of the model's cost.
    """
    return getattr(model, "_orig_mod", model)


def core_applications(cli_args: Dict[str, Any]) -> Optional[int]:
    """`(n_latent + 1) * n_improve * deep_supervision_steps`, or None for a
    non-recursive architecture where the notion does not apply."""
    if cli_args.get("architecture") != "recursive":
        return None
    try:
        n_latent = int(cli_args["trm_n_latent"])
        n_improve = int(cli_args["trm_n_improve"])
        n_sup = int(cli_args["trm_deep_supervision_steps"])
    except (KeyError, TypeError, ValueError):
        return None
    return (n_latent + 1) * n_improve * n_sup


@contextlib.contextmanager
def eager_recursive_core(model: nn.Module):
    """Temporarily undo `--compile on`'s rebinding of `RecursiveCore._f_forward`.

    THIS IS LOAD-BEARING, not a tidy-up. `training/torch_compile.py:143` does

        core._f_forward = types.MethodType(_COMPILED_F_FORWARD, core)

    which shadows the class method with a compiled one. Inductor fuses the
    core's `nn.Linear` calls into a single graph, so their forward hooks NEVER
    FIRE - and the core is where 126 of the model's mixer calls live. Counting a
    compiled model therefore reports only the one-time spectral/stem/head work:
    measured on the ref20 run, Linear MACs came out **71x low** and the total
    read 0.204 GF against the true 6.235 GF.

    Deleting the instance attribute restores the eager class method; the compiled
    binding is put back in the `finally`, so the model is unchanged afterwards and
    the run keeps its compiled speed.

    The scan side needs no equivalent: `count_scan_flops` replaces `scan_fn`
    itself, so it measures the call shapes whether or not the backend is
    compiled.
    """
    from medmamba_ss_trm import RecursiveCore  # noqa: PLC0415

    saved = []
    for core in (m for m in model.modules() if isinstance(m, RecursiveCore)):
        bound = core.__dict__.get("_f_forward")
        if bound is not None:
            saved.append((core, bound))
            del core.__dict__["_f_forward"]
    try:
        yield len(saved)
    finally:
        for core, bound in saved:
            core._f_forward = bound


def count_scan_flops(model: nn.Module, input_shape: Sequence[int], device: str = "cpu",
                     wavelengths=None, sensor_range=None) -> Dict[str, Any]:
    """Selective-scan FLOPs, from the shapes of the calls that actually happen.

    Wraps `scan_fn` on every `_SelectiveScanParams` instance, runs ONE forward,
    and restores the originals in a `finally`. A model with no scan modules
    returns zeros and `n_scan_calls: 0` rather than failing - that is the
    correct answer for `trm_mixer='mlp'` with an RGB stem, not an error.
    """
    from medmamba_ss_trm import _SelectiveScanParams  # noqa: PLC0415 - avoid a heavy import at module load

    model = unwrap_model(model).eval().to(device)
    tally = {"flops": 0, "calls": 0, "shapes": []}
    originals = []

    def make_counter(module, original):
        def counting_scan_fn(xs, *a, **kw):
            # `_SelectiveScanParams.run` takes [B, K, D, L] but FLATTENS the two
            # middle axes before the call (`medmamba_ss_trm.py:473`:
            # `xs.float().reshape(Bsz, K * D, L)`), so what arrives here is 3-D.
            # Take K and D from the module rather than the tensor; L is the last
            # axis and B the first either way.
            b, length = int(xs.shape[0]), int(xs.shape[-1])
            k, d = int(module.K), int(module.d_inner)
            tally["flops"] += estimate_selective_scan_flops(
                batch_size=b, k_directions=k, d_inner=d,
                d_state=int(module.d_state), seq_len=length)
            tally["calls"] += 1
            if len(tally["shapes"]) < 8:          # a sample, not a log
                tally["shapes"].append({"B": b, "K": k, "d_inner": d,
                                        "d_state": int(module.d_state), "L": length,
                                        "arrived_as": list(int(v) for v in xs.shape)})
            return original(xs, *a, **kw)
        return counting_scan_fn

    for module in model.modules():
        if isinstance(module, _SelectiveScanParams):
            originals.append((module, module.scan_fn))
            module.scan_fn = make_counter(module, module.scan_fn)

    try:
        with torch.no_grad(), eager_recursive_core(model):
            x = torch.randn(*input_shape, device=device)
            _forward(model, x, wavelengths=wavelengths, sensor_range=sensor_range, device=device)
    finally:
        for module, original in originals:
            module.scan_fn = original

    return {
        "selective_scan_flops": int(tally["flops"]),
        "n_scan_calls": int(tally["calls"]),
        "n_scan_modules": len(originals),
        "scan_call_shapes_sample": tally["shapes"],
        "method": ("wrapped _SelectiveScanParams.scan_fn over one forward pass; per-call arithmetic "
                    "from the frozen flops_counter.estimate_selective_scan_flops"),
    }


def _forward(model: nn.Module, x: torch.Tensor, wavelengths=None, sensor_range=None,
             device: str = "cpu"):
    """`model(x)`, forwarding wavelengths when the model accepts them.

    Mirrors `TrainerG_v11._model_forward` rather than inventing a second
    convention. The positional encoding branch this selects
    (`medmamba_ss_trm.py:827`) changes no Conv/Linear shape, so the conv/linear
    count is identical either way - but running the same code path the trainer
    runs is what makes the number attributable to the run.
    """
    if wavelengths is None:
        return model(x)
    wl = torch.as_tensor(wavelengths, dtype=torch.float32, device=device)
    try:
        return model(x, wavelengths=wl, sensor_range=sensor_range)
    except TypeError:
        return model(x)


def build_flops_report(model: nn.Module, in_channels: int, patch_hw: Sequence[int],
                       cli_args: Optional[Dict[str, Any]] = None, device: str = "cpu",
                       wavelengths=None, sensor_range=None,
                       backbone_num_params: Optional[int] = None) -> Dict[str, Any]:
    """The full report, for one instantiated model at batch 1.

    `patch_hw` is `(H, W)`. The input shape is `(1, C, H, W)` - the batch
    dimension the frozen caller dropped.
    """
    cli_args = cli_args or {}
    h, w = int(patch_hw[0]), int(patch_hw[1])
    input_shape = (1, int(in_channels), h, w)

    # Unwrap ONCE, so the conv/linear pass and the scan pass count the same
    # module. Counting through a `torch.compile` wrapper also works, but it
    # triggers a recompilation on the fresh random input and attributes the
    # layers to the dynamo graph module.
    model = unwrap_model(model)

    with eager_recursive_core(model) as n_decompiled:
        conv_linear = count_flops(model, input_shape, device=device)
    scan = count_scan_flops(model, input_shape, device=device,
                            wavelengths=wavelengths, sensor_range=sensor_range)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = int(conv_linear["conv_linear_flops_approx"]) + int(scan["selective_scan_flops"])

    return {
        "input_shape": list(input_shape),
        "conv_linear_macs": conv_linear["conv_linear_macs"],
        "conv_linear_flops_approx": conv_linear["conv_linear_flops_approx"],
        "macs_by_layer_type": conv_linear["macs_by_layer_type"],
        "selective_scan_flops": scan["selective_scan_flops"],
        "n_scan_calls": scan["n_scan_calls"],
        "n_scan_modules": scan["n_scan_modules"],
        "scan_call_shapes_sample": scan["scan_call_shapes_sample"],
        "scan_method": scan["method"],
        "total_flops_approx": total,
        "total_gflops_approx": round(total / 1e9, 6),
        "core_applications": core_applications(cli_args),
        "decompiled_cores_for_counting": n_decompiled,
        "architecture": cli_args.get("architecture"),
        "trm": {k: cli_args.get(k) for k in
                ("trm_dim", "trm_core_layers", "trm_n_latent", "trm_n_improve",
                 "trm_deep_supervision_steps", "trm_mixer")}
               if cli_args.get("architecture") == "recursive" else None,
        "wrapped_num_params": n_params,
        "wrapped_num_trainable_params": n_trainable,
        "backbone_num_params": backbone_num_params,
        "param_memory_mb": round(n_params * 4 / (1024 ** 2), 4),
        "note": ("conv_linear is exact (forward hooks); selective_scan is an analytical estimate at "
                  "the measured call shapes. Counted at batch 1 - the recursive variant pays its "
                  "core applications at inference too, so this is an inference cost, not a "
                  "training-only one."),
    }
