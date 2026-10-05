# -*- coding: utf-8 -*-
"""
training/lightweight_attention.py
===================================
Main Development Plan, Phase 24 - Lightweight Spectral Attention.

"If the current model uses spectral inter-channel attention, compare: Full
attention vs Lightweight attention vs No attention. Potential lightweight
alternatives: SE-style gating, ECA-style channel weighting, simple learned
channel gate. Keep attention only if it provides measurable benefit
relative to its parameter/computational cost."

REINTERPRETATION NOTE (same spirit as medmamba_ss_fullchannel.py's docstring)
----------------------------------------------------------------------------
`medmamba_ss_trm.py` doesn't have an expensive full spectral-inter-channel
attention module to begin with: `SqueezeExcite1D` (inside
`ResidualSpectralBlock`, gated by `cfg.squeeze_excitation`) is already
SE-style gating over spectral tokens, and `BandGate` is already a simple
learned per-band gate. The one place genuinely "full attention" exists is
`CrossAttentionFusion` - one of `medmamba_ss_trm.py`'s five interchangeable
per-stage FUSION strategies (how the spectral context map is merged into
the spatial features at each stage), selectable via
`MedMambaSSTRMConfig.fusion_type`.

This module adds the requested comparison POINTS at that same fusion slot:
an ECA-style option (`ECAFusion`), an SE-gate-style option (`SEGateFusion`,
cheaper than the existing `GatedAdditiveFusion` - purely multiplicative, no
additive context injection), and a "no attention" floor (`NoFusion`, which
drops spectral context entirely - useful ONLY as an ablation baseline, not
a recommended default, since it throws away Phase 6/7's bidirectional
spectral<->spatial interaction).

`medmamba_ss_trm.py` is not edited; `medmamba_ss_fullchannel.py`'s `_fusion_factory`
hook is how these get wired in (see `medmamba_ss_efficient.py`).
"""

from __future__ import annotations

import math
from typing import Set

import torch
import torch.nn as nn

from medmamba_ss_trm import (
    SpectralSpatialFusion, FUSION_TYPES, ALL_SCAN_DIRECTIONS, get_backend, make_fusion,
)


class ECAFusion(SpectralSpatialFusion):
    """Efficient Channel Attention (Wang et al., 2020) style: a single 1D
    conv over the channel axis (kernel size derived from channel count,
    not a learned hidden dimension) produces per-channel gates from a
    pooled context summary - O(C) parameters instead of
    `CrossAttentionFusion`'s O(C^2) (Q/K/V/out projections) and O((HW)^2)
    compute."""

    def __init__(self, spatial_dim: int, ctx_dim: int, gamma: int = 2, b: int = 1):
        super().__init__(spatial_dim, ctx_dim)
        k = int(abs((math.log2(spatial_dim) + b) / gamma))
        k = k if k % 2 == 1 else k + 1
        k = max(k, 3)
        self.ctx_proj = nn.Linear(ctx_dim, spatial_dim)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        ctx_summary = ctx.mean(dim=(1, 2)) if ctx.dim() == 4 else ctx   # [B, ctx_dim]
        ctx_summary = self.ctx_proj(ctx_summary)                        # [B, spatial_dim]
        gate = self.conv(ctx_summary.unsqueeze(1)).squeeze(1)           # [B, spatial_dim]
        gate = torch.sigmoid(gate)[:, None, None, :]
        return spatial * gate


class SEGateFusion(SpectralSpatialFusion):
    """Squeeze-and-excite-style: pool spatial features, mix with a pooled
    context summary, produce a per-channel gate via a small bottleneck MLP.
    Purely multiplicative (cheaper than `medmamba_ss_trm.GatedAdditiveFusion`,
    which also additively injects a full projected context vector)."""

    def __init__(self, spatial_dim: int, ctx_dim: int, reduction: int = 8):
        super().__init__(spatial_dim, ctx_dim)
        hidden = max(spatial_dim // reduction, 8)
        self.ctx_proj = nn.Linear(ctx_dim, spatial_dim)
        self.fc = nn.Sequential(
            nn.Linear(spatial_dim, hidden), nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, spatial_dim),
        )

    def forward(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        ctx_summary = ctx.mean(dim=(1, 2)) if ctx.dim() == 4 else ctx
        pooled = spatial.mean(dim=(1, 2)) + self.ctx_proj(ctx_summary)
        gate = torch.sigmoid(self.fc(pooled))[:, None, None, :]
        return spatial * gate


class NoFusion(SpectralSpatialFusion):
    """Phase 24's "no attention" ablation floor: context is ignored
    entirely; spatial features pass through unchanged. Only meaningful as a
    comparison point to show whether spectral<->spatial fusion helps at
    all - not a recommended architecture choice."""

    def __init__(self, spatial_dim: int, ctx_dim: int):
        super().__init__(spatial_dim, ctx_dim)

    def forward(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return spatial


_EXTENDED_TABLE = {"eca": ECAFusion, "se_gate": SEGateFusion, "none": NoFusion}
EXTENDED_FUSION_TYPES: Set[str] = set(FUSION_TYPES) | set(_EXTENDED_TABLE)


def make_fusion_extended(fusion_type: str, spatial_dim: int, ctx_dim: int) -> SpectralSpatialFusion:
    ft = fusion_type.lower()
    if ft in _EXTENDED_TABLE:
        return _EXTENDED_TABLE[ft](spatial_dim, ctx_dim)
    return make_fusion(fusion_type, spatial_dim, ctx_dim)  # falls back to film/gated/cross_attention/...


def validate_extended_config(cfg) -> None:
    """Same checks as `medmamba_ss_trm.MedMambaSSTRMConfig.validate()`, except
    `fusion_type` is checked against `EXTENDED_FUSION_TYPES` instead of the
    original `FUSION_TYPES`, so 'eca'/'se_gate'/'none' don't fail
    validation. Used by `medmamba_ss_efficient.py`'s backbone in place of
    `cfg.validate()`."""
    assert len(cfg.dims) == len(cfg.depths) > 0, "dims and depths must be non-empty and same length"
    for i in range(len(cfg.dims) - 1):
        assert cfg.dims[i + 1] == 2 * cfg.dims[i], (
            f"dims[{i + 1}]={cfg.dims[i + 1]} must equal 2*dims[{i}]={2 * cfg.dims[i]}")
    dirs = cfg.resolved_scan_directions()
    assert len(dirs) > 0, "at least one scan direction must be enabled"
    bad = set(dirs) - ALL_SCAN_DIRECTIONS
    assert not bad, f"unknown scan direction(s): {bad}"
    for f in cfg.resolved_fusion_types():
        assert f in EXTENDED_FUSION_TYPES, f"unknown fusion_type {f!r}. Valid: {sorted(EXTENDED_FUSION_TYPES)}"
    assert cfg.band_selection_mode in ("soft", "sparse")
    assert cfg.norm_type in ("layernorm", "rmsnorm", "groupnorm")
    get_backend(cfg.scan_backend)
