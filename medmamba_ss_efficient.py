# -*- coding: utf-8 -*-
"""
medmamba_ss_efficient.py
========================
Main Development Plan, Stage D ("Efficiency") - Phases 21, 24, 25, 26.

Builds on `medmamba_ss_fullchannel.py` (Stage C) using the extension hooks
added there (`_fusion_factory`, `_block_factory`, `_stage_factory`,
`_backbone_factory`, `_classification_head_factory`) - each class below
overrides exactly ONE hook and inherits everything else unchanged:

    EfficientGBlock(FullChannelGBlock)             overrides _fusion_factory
    EfficientGStage(FullChannelGStage)              overrides _block_factory
    EfficientMedMambaSSBackbone(FullChannelMedMambaSSBackbone)
                                                     overrides _stage_factory, _validate_config
    MedMambaSSEfficient(MedMambaSSFullChannel)        overrides _backbone_factory,
                                                     _classification_head_factory

Phase-by-phase:
  Phase 21 - "Replace dense Conv3D": N/A here - this codebase never had a
    dense Conv3D spectral branch (see `medmamba_ss_fullchannel.py`'s Phase
    24-C note); `FullChannelGBlock`'s local branch is already
    depthwise+pointwise.
  Phase 24 - fusion_type defaults to `training.lightweight_attention`'s
    `"se_gate"` (a lightweight alternative to `cross_attention`/`film`/
    `gated`), with `"eca"` and `"none"` (ablation floor) also available via
    `cfg.fusion_type`.
  Phase 25 - `_classification_head_factory` swapped to
    `training.compact_head.CompactClassificationHead`.
  Phase 26/40/41 - see `training/capacity_search.py` for parameter-target
    sweeps over this class specifically (pass `model_cls=MedMambaSSEfficient`).

Phase 22 (spectral width) and Phase 23 (fewer expensive spectral blocks per
stage) are already directly configurable on the SAME `MedMambaSSTRMConfig`
object via `d_token`/`compression_dims` and `spectral_depth` respectively -
no new class needed, see the module-level `scale_spectral_width()` helper
below and the docstring note on Phase 23's stage-allocation idea.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from medmamba_ss_trm import MedMambaSSTRMConfig
from medmamba_ss_fullchannel import (
    FullChannelGBlock, FullChannelGStage, FullChannelMedMambaSSBackbone, MedMambaSSFullChannel,
)
from training.lightweight_attention import make_fusion_extended, validate_extended_config
from training.compact_head import CompactClassificationHead


class EfficientGBlock(FullChannelGBlock):
    _fusion_factory = staticmethod(make_fusion_extended)


class EfficientGStage(FullChannelGStage):
    _block_factory = staticmethod(EfficientGBlock)


class EfficientMedMambaSSBackbone(FullChannelMedMambaSSBackbone):
    _stage_factory = staticmethod(EfficientGStage)

    def _validate_config(self, cfg: MedMambaSSTRMConfig) -> None:
        validate_extended_config(cfg)


class MedMambaSSEfficient(MedMambaSSFullChannel):
    _backbone_factory = staticmethod(EfficientMedMambaSSBackbone)
    _classification_head_factory = staticmethod(CompactClassificationHead)


# ----------------------------------------------------------------------------
# Phase 22 - spectral branch width scaling (100%/75%/50%/25% ablation grid)
# ----------------------------------------------------------------------------

def scale_spectral_width(cfg: MedMambaSSTRMConfig, fraction: float) -> MedMambaSSTRMConfig:
    """Returns a NEW config with `d_token` and `compression_dims` scaled by
    `fraction` (e.g. 0.75/0.5/0.25 for the Phase 22 ablation grid), leaving
    everything else - including `dims`/`depths` (spatial capacity) -
    untouched, so the resulting parameter delta is attributable to spectral
    width alone."""
    new_d_token = max(4, round(cfg.d_token * fraction))
    new_compression = tuple(max(4, round(d * fraction)) for d in cfg.compression_dims)
    return replace(cfg, d_token=new_d_token, compression_dims=new_compression)


# ----------------------------------------------------------------------------
# Convenience constructors, mirroring medmamba_ss_trm.py's naming.
# fusion_type defaults to the Phase 24 lightweight recommendation.
# ----------------------------------------------------------------------------

def medmamba_ss_efficient_tiny(num_classes=1000, fusion_type: str = "se_gate", **kwargs) -> MedMambaSSEfficient:
    cfg = MedMambaSSTRMConfig(dims=(96, 192, 384, 768), depths=(2, 2, 4, 2), patch_size=4, fusion_type=fusion_type)
    return MedMambaSSEfficient(cfg, task="classification", num_classes=num_classes, **kwargs)


def medmamba_ss_efficient_hsi_small(num_classes=1000, fusion_type: str = "se_gate", **kwargs) -> MedMambaSSEfficient:
    cfg = MedMambaSSTRMConfig(dims=(64, 128, 256), depths=(2, 2, 2), d_state=8, d_ctx=64,
                           patch_size=1, spectral_depth=3, fusion_type=fusion_type)
    return MedMambaSSEfficient(cfg, task="classification", num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from medmamba_ss_trm import medmamba_ss_hsi_small
    from medmamba_ss_fullchannel import medmamba_ss_fullchannel_hsi_small

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    torch.manual_seed(0)
    x = torch.randn(2, 128, 11, 11)
    y = torch.randint(0, 6, (2,))

    print("=== Param comparison: split (Stage A/B baseline) vs fullchannel (Stage C) vs efficient (Stage D) ===")
    m_split = medmamba_ss_hsi_small(num_classes=6)
    m_full = medmamba_ss_fullchannel_hsi_small(num_classes=6)
    m_eff = medmamba_ss_efficient_hsi_small(num_classes=6)
    for name, m in (("split", m_split), ("fullchannel", m_full), ("efficient/se_gate", m_eff)):
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 6)
        print(f"  {name:<20} params={count_params(m)/1e6:.3f}M  out={tuple(out.shape)}")

    print("\n=== Fusion-type sweep on the efficient architecture (Phase 24) ===")
    for ft in ("se_gate", "eca", "none", "film"):
        m = medmamba_ss_efficient_hsi_small(num_classes=6, fusion_type=ft)
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 6)
        print(f"  fusion_type={ft:<10} params={count_params(m)/1e6:.3f}M")

    print("\n=== Backward pass / gradient-flow check (efficient, se_gate) ===")
    m_eff.train()
    out = m_eff(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    n_missing = sum(1 for p in m_eff.parameters() if p.requires_grad and p.grad is None)
    print(f"  loss={loss.item():.4f}  params_without_grad={n_missing}")
    assert n_missing == 0

    print("\n=== Phase 22 spectral-width scaling ===")
    base_cfg = MedMambaSSTRMConfig(dims=(64, 128, 256), depths=(2, 2, 2), d_state=8, d_ctx=64,
                                patch_size=1, spectral_depth=3)
    for frac in (1.0, 0.75, 0.5, 0.25):
        cfg = scale_spectral_width(base_cfg, frac)
        m = MedMambaSSEfficient(cfg, num_classes=6)
        print(f"  fraction={frac:<5} d_token={cfg.d_token} compression_dims={cfg.compression_dims} "
              f"params={count_params(m)/1e6:.3f}M")

    print("\nAll smoke tests passed.")
