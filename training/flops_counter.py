# -*- coding: utf-8 -*-
"""
training/flops_counter.py
============================
Main Development Plan, Phase 0/48 requires FLOPs/MACs among the recorded
metrics. Nothing in this repo measures them - `evaluator.py`/`evaluate.py`
report parameter counts and wall-clock latency/throughput, never FLOPs.

SCOPE / HONEST LIMITATION
---------------------------
This counts multiply-accumulate ops for `nn.Conv1d`/`nn.Conv2d`/`nn.Linear`
via forward hooks (the vast majority of parameters and compute in every
architecture variant here). It does NOT count the custom selective-scan
einsum operations inside `medmamba_ss_trm._SelectiveScanParams.run()`
(`_selective_scan_pure_pytorch`'s per-timestep loop) - those are the actual
core Mamba/SSM computation and are NOT negligible, especially at the
patch-level resolutions this model runs at (the `L` sequence length passed
into the scan can be large: H*W for spatial SS2D, H+W-1 for diagonal scans).

Rather than silently under-report "total FLOPs" as if this were complete
(the exact failure mode `training/spectral_metrics.py`'s
`spectral_reconstruction_metrics()` already refuses to do elsewhere in this
repo - "no fake values"), `count_flops()` returns BOTH the measured
Conv/Linear MACs AND a clearly-labeled `selective_scan_flops_counted=False`
flag plus the analytical formula for what the scan actually costs
(`estimate_selective_scan_flops`), so a caller can add it in rather than
being misled by a partial number presented as the whole.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _conv_macs(module, output_shape) -> int:
    out_elements = 1
    for d in output_shape[2:]:
        out_elements *= d
    kernel_elements = 1
    for k in module.kernel_size:
        kernel_elements *= k
    in_channels_per_group = module.in_channels // module.groups
    return output_shape[0] * output_shape[1] * out_elements * kernel_elements * in_channels_per_group


def _linear_macs(module, input_shape) -> int:
    n_rows = 1
    for d in input_shape[:-1]:
        n_rows *= d
    return n_rows * module.in_features * module.out_features


def count_flops(model: nn.Module, input_shape, device: str = "cpu") -> Dict:
    """`input_shape`: e.g. `(1, 128, 11, 11)` for a batch-1 HSI patch. Runs
    ONE forward pass with hooks attached; returns MACs (multiply-
    accumulates) and FLOPs (~2x MACs) for Conv1d/Conv2d/Linear layers only -
    see module docstring for what's excluded."""
    model = model.eval().to(device)
    macs_by_type: Dict[str, int] = {"Conv1d": 0, "Conv2d": 0, "Linear": 0}
    total_macs = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal total_macs
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            m = _conv_macs(module, tuple(output.shape))
            macs_by_type[type(module).__name__] += m
            total_macs += m
        elif isinstance(module, nn.Linear):
            m = _linear_macs(module, tuple(inputs[0].shape))
            macs_by_type["Linear"] += m
            total_macs += m

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))

    try:
        with torch.no_grad():
            x = torch.randn(*input_shape, device=device)
            model(x)
    finally:
        for h in handles:
            h.remove()

    return {
        "input_shape": list(input_shape),
        "conv_linear_macs": total_macs,
        "conv_linear_flops_approx": total_macs * 2,
        "macs_by_layer_type": macs_by_type,
        "selective_scan_flops_counted": False,
        "selective_scan_note": (
            "Excluded - see estimate_selective_scan_flops() for the analytical formula. "
            "Not negligible at this model's typical patch resolutions."
        ),
    }


def estimate_selective_scan_flops(batch_size: int, k_directions: int, d_inner: int,
                                   d_state: int, seq_len: int) -> int:
    """Analytical FLOP estimate for ONE `_selective_scan_pure_pytorch` call
    (`medmamba_ss_trm.py`), matching its per-timestep recurrence:
    `x_state = dA*x_state + dB*x`, `y = einsum(x_state, C)`, each O(D*N) per
    timestep, run `L` times, across `K` scan directions and the batch.
    Returns an order-of-magnitude FLOP count (each multiply-add counted as
    2 FLOPs), NOT a cycle-accurate hardware estimate."""
    per_step_per_direction = 4 * d_inner * d_state  # dA + dB + state update + y einsum, each ~D*N
    return int(2 * batch_size * k_directions * seq_len * per_step_per_direction)
