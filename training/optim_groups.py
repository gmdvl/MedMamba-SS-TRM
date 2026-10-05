# -*- coding: utf-8 -*-
"""
training/optim_groups.py
==========================
MedMamba-SS-TRM v15 plan, R1.2 - weight-decay parameter groups.

`train_example_v6..v14` all build the optimizer as

    torch.optim.AdamW(model.parameters(), lr=..., weight_decay=0.05)

which applies decay to EVERY parameter: biases, `LayerNorm`/`RMSNorm` gains,
`LayerScale` alphas, the recursive core's `z_init`/`y_init`, the tokenizer's
`pe_gain`, and - critically - `value_embed.weight`, the one weight that has
to GROW for the model to become input-sensitive at all (see R1.1). Decaying a
1-D norm gain toward zero is a well-known no-op-at-best/harmful-at-worst
practice; decaying the value embedding actively fights the representation fix.

`build_param_groups(model, weight_decay)` returns the two-group list AdamW
expects:

  * decay      - parameters with `ndim >= 2` that are not norm weights.
  * no_decay   - everything else: biases, all 1-D parameters, anything whose
                 qualified name matches a norm / gain / init-state pattern,
                 and `value_embed` / the concat-MLP fusion weights explicitly.

Pure bookkeeping over `model.named_parameters()`: no torch ops, no model
mutation, and `sum(len(g["params"]) for g in groups)` always equals the number
of parameters that require grad, so nothing can be dropped silently.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch.nn as nn

# Substrings that mark a parameter as "never decay" regardless of its rank.
# Matched against the parameter's fully-qualified name, lowercased.
NO_DECAY_NAME_PARTS: Tuple[str, ...] = (
    "bias",
    "norm",            # LayerNorm / RMSNorm / _GroupNormChannelsLast gains
    "alpha",           # medmamba_ss_trm.LayerScale
    "pe_gain",         # v15 R1.1 tokenizer PE gain
    "spatial_pe_gain",  # v15 R1.1c 2-D PE gain
    "z_init",          # RecursiveCore state buffers, if ever made parameters
    "y_init",
    "threshold",       # BandGate (sparse mode) learnable cutoff
    "direction_weights",   # SS2D direction mixing logits
)

# Parameters that are 2-D (so the rank rule would decay them) but must not be
# decayed, because decaying them re-creates the defect R1.1 exists to fix.
NO_DECAY_MODULE_PARTS: Tuple[str, ...] = (
    "value_embed",     # SpectralTokenizer, "add"/"scaled" fusion
    "tokenizer.fuse_in",   # SpectralTokenizer, "concat_mlp" fusion
    "tokenizer.fuse_out",
)


def _is_norm_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d)) \
        or type(module).__name__ in ("RMSNorm", "_GroupNormChannelsLast")


def classify_parameter(name: str, param, norm_param_names: Sequence[str] = ()) -> str:
    """'decay' | 'no_decay' for one named parameter. Split out from
    `build_param_groups` so a test can assert the rule directly."""
    lowered = name.lower()
    if not param.requires_grad:
        return "no_decay"
    if param.ndim < 2:
        return "no_decay"
    if name in norm_param_names:
        return "no_decay"
    if any(part in lowered for part in NO_DECAY_NAME_PARTS):
        return "no_decay"
    if any(part in name for part in NO_DECAY_MODULE_PARTS):
        return "no_decay"
    return "decay"


def build_param_groups(model: nn.Module, weight_decay: float,
                        extra_no_decay: Iterable[str] = ()) -> List[Dict]:
    """AdamW parameter groups for `model` (v15 R1.2).

    `extra_no_decay` is an iterable of additional name substrings to exclude
    from decay. Returns a two-entry list; the `no_decay` group is present even
    when empty so callers can index it positionally in a report.
    """
    extra = tuple(extra_no_decay)
    norm_param_names = set()
    for mod_name, module in model.named_modules():
        if not _is_norm_module(module):
            continue
        for p_name, _ in module.named_parameters(recurse=False):
            norm_param_names.add(f"{mod_name}.{p_name}" if mod_name else p_name)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        bucket = classify_parameter(name, param, norm_param_names)
        if bucket == "decay" and any(part in name for part in extra):
            bucket = "no_decay"
        (decay if bucket == "decay" else no_decay).append(param)

    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def summarize_param_groups(model: nn.Module, weight_decay: float) -> Dict:
    """A JSON-serializable description of the split, for `config.json`. Lists
    the no-decay parameter NAMES (there are few of them and they are exactly
    what a reader wants to check), plus element counts for both groups."""
    groups = build_param_groups(model, weight_decay)
    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_names, decay_elems, no_decay_elems = [], 0, 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in decay_ids:
            decay_elems += param.numel()
        else:
            no_decay_names.append(name)
            no_decay_elems += param.numel()
    return {
        "weight_decay": float(weight_decay),
        "n_decay_tensors": len(groups[0]["params"]),
        "n_no_decay_tensors": len(groups[1]["params"]),
        "n_decay_elements": int(decay_elems),
        "n_no_decay_elements": int(no_decay_elems),
        "no_decay_parameter_names": no_decay_names,
    }
