# -*- coding: utf-8 -*-
"""
medmamba_ss_fullchannel.py
=========================
Main Development Plan, Stage C ("Architecture") - Phases 22-33:
"Remove the Hard Channel Split" / "Full-Channel Spectral-Spatial Block" /
"Full-Channel Fusion" / "Cross-Branch Interaction" / "Reduce Conv3D
Parameters" (Phase 24 Option C).

IMPORTANT REINTERPRETATION NOTE - read before using this module
------------------------------------------------------------------
The plan's Phase 22 describes removing a split of the form "C spectral
channels + 1 spatial channel", and frames Phases 22-27 as fixing the
model's spectral/spatial channel allocation. That literal split does not
exist in `medmamba_ss_trm.py`: `SpectralPathway` already tokenizes and processes
EVERY wavelength band densely (Phase 1/2 of the *original* MedMamba-SS plan
this codebase already implements), and its output is fused into the
backbone at every stage via a pluggable per-stage fusion strategy
(FiLM/gated/cross-attention/...), not by permanently carving off a fixed
subset of channels.

The one place a hard, fixed 50/50 channel split genuinely exists is inside
`GBlock` itself (see `medmamba_ss_trm.py`), between its TWO SPATIAL operators - a
long-range `SS2D` selective scan on one half of the channels, and a local
depthwise-ish conv branch on the other half - a "SS-Conv-SSM" design
borrowed from MedMamba, unrelated to spectral processing. This module
removes THAT split, translating Phase 22-27's actual mechanism (both
branches see the full channel width, combined via lightweight cross-branch
interaction and learned gating instead of concatenation + channel-shuffle)
onto the two operators that exist here. If your intent was instead to
change how `SpectralPathway` itself allocates channels, that code already
satisfies Phase 22's stated goal (dense, full-channel spectral processing)
and doesn't need this module.

WHAT'S IN HERE
--------------
`FullChannelGBlock` - drop-in alternative to `medmamba_ss_trm.GBlock`:
  - Branch A: full-channel `SS2D`/`MultiScaleSS2D` (long-range operator),
    same class `medmamba_ss_trm.py` already uses, just no longer given only half
    the channels.
  - Branch B: full-channel depthwise-separable conv (Phase 24 Option C:
    depthwise conv + pointwise projection, NOT a dense Conv2d(hidden_dim,
    hidden_dim) - Phase 28/41 explicitly flag dense convolutions as a
    parameter-reduction target, so this avoids reintroducing that cost
    just because channel width doubled per branch).
  - Phase 27 cross-branch interaction: `Fa' = Fa + A(Fb)`,
    `Fb' = Fb + B(Fa)` with lightweight `nn.Linear` `A`/`B`.
  - Phase 26 fusion: a small gate (global-pool -> 2-way softmax) produces
    per-sample weights `g_a, g_b` and the branches are combined as
    `out = residual + g_a*Fa' + g_b*Fb'` - a weighted sum, not
    concatenation + `channel_shuffle`.

`FullChannelGStage` / `FullChannelMedMambaSSBackbone` / `MedMambaSSFullChannel`
mirror `medmamba_ss_trm.py`'s `GStage`/`MedMambaSSBackbone`/`MedMambaSS` exactly
(same `forward_features()` output dict, same task heads, same config
object) by SUBCLASSING them and overriding only `__init__` to swap the
block type - `forward()` on every level above `FullChannelGBlock` is
inherited UNCHANGED, so this is a fully compatible drop-in: every wrapper
built in Stage A/B (`MedMambaSSTRMLatentReconWrapper`, the activation patch,
TrainerG_v5/v6, ...) works with it without modification.

`medmamba_ss_trm.py` is not edited.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from medmamba_ss_trm import (
    MedMambaSSTRMConfig, MedMambaSS, MedMambaSSBackbone, GStage, SpectralPathway,
    ClassificationHead, RegressionHead, EmbeddingHead, SegmentationHead,
    make_norm, make_fusion, SS2D, MultiScaleSS2D, DropPath, LayerScale, GatedMLP,
    PatchMerging2D, SpectralContextUpdater, StageBandSelector,
)


class FullChannelGBlock(nn.Module):
    """See module docstring. `forward(x, ctx_map) -> x`, matching
    `medmamba_ss_trm.GBlock`'s signature exactly (channels-last `x`: [B,H,W,D])."""

    def __init__(self, hidden_dim: int, cfg: MedMambaSSTRMConfig, drop_path: float, ctx_dim: int,
                 fusion_type: str):
        super().__init__()
        self.fusion = make_fusion(fusion_type, hidden_dim, ctx_dim)

        # Branch A - long-range operator, FULL channel width (was `half` in GBlock)
        self.ln_a = make_norm(cfg.norm_type, hidden_dim)
        if cfg.spatial_multiscale:
            self.branch_a = MultiScaleSS2D(hidden_dim, cfg.d_state, cfg.spatial_scales,
                                            scan_directions=cfg.scan_directions, backend=cfg.scan_backend)
        else:
            self.branch_a = SS2D(hidden_dim, d_state=cfg.d_state, scan_directions=cfg.scan_directions,
                                  backend=cfg.scan_backend)
        self.scale_a = LayerScale(hidden_dim, cfg.layerscale_init, enabled=cfg.layerscale)
        self.drop_path_a = DropPath(drop_path)

        # Branch B - local operator, FULL channel width, depthwise + pointwise (Phase 24-C)
        self.branch_b = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),  # depthwise
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),                                 # pointwise
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.drop_path_b = DropPath(drop_path)

        # Phase 27 - lightweight cross-branch interaction
        self.interact_a_from_b = nn.Linear(hidden_dim, hidden_dim)
        self.interact_b_from_a = nn.Linear(hidden_dim, hidden_dim)

        # Phase 26 - learned gate instead of concat + channel_shuffle
        gate_hidden = max(hidden_dim // 8, 8)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden), nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(gate_hidden, 2),
        )

        self.use_ffn = cfg.use_ffn
        if cfg.use_ffn:
            self.ln_ffn = make_norm(cfg.norm_type, hidden_dim)
            self.ffn = GatedMLP(hidden_dim, mult=2.0)
            self.ffn_scale = LayerScale(hidden_dim, cfg.layerscale_init, enabled=cfg.layerscale)
            self.ffn_drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, ctx_map: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fusion(x, ctx_map)

        fa = self.drop_path_a(self.scale_a(self.branch_a(self.ln_a(x))))

        xb = x.permute(0, 3, 1, 2).contiguous()
        fb = self.branch_b(xb).permute(0, 2, 3, 1).contiguous()
        fb = self.drop_path_b(fb)

        fa2 = fa + self.interact_a_from_b(fb)   # Phase 27
        fb2 = fb + self.interact_b_from_a(fa)

        pooled = x.mean(dim=(1, 2))                        # [B, D]
        g = torch.softmax(self.gate(pooled), dim=-1)        # [B, 2]
        g_a = g[:, 0][:, None, None, None]
        g_b = g[:, 1][:, None, None, None]
        out = residual + g_a * fa2 + g_b * fb2               # Phase 26

        if self.use_ffn:
            out = out + self.ffn_drop_path(self.ffn_scale(self.ffn(self.ln_ffn(out))))
        return out


class FullChannelGStage(GStage):
    """Identical to `medmamba_ss_trm.GStage` except its blocks are
    `FullChannelGBlock` instead of `GBlock`. `forward()` is inherited
    UNCHANGED from `GStage` - it only calls `blk(x, ctx_aligned)`
    generically, so nothing else needs to change."""

    def __init__(self, dim: int, depth: int, cfg: MedMambaSSTRMConfig, drop_path,
                 ctx_dim: int, downsample: bool, fusion_type: str):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList([
            FullChannelGBlock(dim, cfg, drop_path[i], ctx_dim, fusion_type) for i in range(depth)
        ])
        self.spectral_refinement = cfg.stage_spectral_refinement
        if self.spectral_refinement:
            self.ctx_updater = SpectralContextUpdater(dim, ctx_dim, norm_type=cfg.norm_type)
        self.dynamic_band_selection = cfg.dynamic_band_selection
        if self.dynamic_band_selection:
            self.stage_band_selector = StageBandSelector(ctx_dim)
        self.downsample = PatchMerging2D(dim, norm_type=cfg.norm_type) if downsample else None
        self.ctx_downsample = downsample


class FullChannelMedMambaSSBackbone(MedMambaSSBackbone):
    """Identical to `medmamba_ss_trm.MedMambaSSBackbone` except its stages are
    `FullChannelGStage`. `forward_features()`/`forward()` are inherited
    UNCHANGED - both only iterate `self.stages` generically, returning the
    exact same output dict shape (`feature_map`, `pooled`, `stage_pools`,
    `stage_feature_maps`, `stage_ctx_maps`, `spectral_context`,
    `band_weights`)."""

    def __init__(self, cfg: MedMambaSSTRMConfig):
        nn.Module.__init__(self)
        cfg.validate()
        self.cfg = cfg
        self.num_stages = len(cfg.dims)

        self.spectral_pathway = SpectralPathway(cfg)
        self.stem = nn.Linear(cfg.d_ctx, cfg.dims[0])
        self.stem_norm = make_norm(cfg.norm_type, cfg.dims[0])

        total_depth = sum(cfg.depths)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_depth)]
        fusion_types = cfg.resolved_fusion_types()

        self.stages = nn.ModuleList()
        cursor = 0
        for i, (dim, depth) in enumerate(zip(cfg.dims, cfg.depths)):
            stage = FullChannelGStage(dim, depth, cfg, dpr[cursor:cursor + depth], cfg.d_ctx,
                                       downsample=(i < self.num_stages - 1), fusion_type=fusion_types[i])
            self.stages.append(stage)
            cursor += depth

        self.apply(self._init_weights)
        # v15 R1.1 - see medmamba_ss_trm.MedMambaSSBackbone.__init__ (this backbone
        # re-implements __init__ rather than calling super(), so the tokenizer
        # re-init has to be repeated here or `spectral_value_init_std` would be
        # silently ignored for --architecture fullchannel/efficient).
        self.spectral_pathway.tokenizer.reinit_value_branch()


class MedMambaSSFullChannel(MedMambaSS):
    """Identical to `medmamba_ss_trm.MedMambaSS` except `self.backbone` is a
    `FullChannelMedMambaSSBackbone`. `forward()` is inherited UNCHANGED, so
    every task head (classification/regression/embedding/segmentation) and
    every Stage A/B wrapper (`MedMambaSSTRMLatentReconWrapper`,
    `replace_relu_with_leakyrelu`, `TrainerG_v5`/`v6`, ...) works with this
    exactly as it does with plain `MedMambaSS` - `base_model.cfg`,
    `base_model.backbone.forward_features(...)`, and `base_model.head(...)`
    all exist with the same names/shapes."""

    def __init__(self, cfg: MedMambaSSTRMConfig = None, task: str = "classification",
                 num_classes: int = 1000, **head_kwargs):
        nn.Module.__init__(self)
        self.cfg = cfg or MedMambaSSTRMConfig()
        self.task = task
        self.backbone = FullChannelMedMambaSSBackbone(self.cfg)

        if task == "classification":
            self.head = ClassificationHead(self.cfg, num_classes)
        elif task == "regression":
            self.head = RegressionHead(self.cfg, head_kwargs.get("out_dim", 1))
        elif task == "embedding":
            self.head = EmbeddingHead(self.cfg, head_kwargs.get("embed_dim", 256))
        elif task == "segmentation":
            self.head = SegmentationHead(self.cfg, num_classes)
        else:
            raise ValueError(f"Unknown task: {task!r}")


# ----------------------------------------------------------------------------
# Convenience constructors, mirroring medmamba_ss_trm.py's naming
# ----------------------------------------------------------------------------

def medmamba_ss_fullchannel_tiny(num_classes=1000, **kwargs) -> MedMambaSSFullChannel:
    cfg = MedMambaSSTRMConfig(dims=(96, 192, 384, 768), depths=(2, 2, 4, 2), patch_size=4)
    return MedMambaSSFullChannel(cfg, task="classification", num_classes=num_classes, **kwargs)


def medmamba_ss_fullchannel_hsi_small(num_classes=1000, **kwargs) -> MedMambaSSFullChannel:
    cfg = MedMambaSSTRMConfig(dims=(64, 128, 256), depths=(2, 2, 2), d_state=8, d_ctx=64,
                           patch_size=1, spectral_depth=3)
    return MedMambaSSFullChannel(cfg, task="classification", num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    import torch.nn.functional as F

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    torch.manual_seed(0)

    print("=== Shape/param smoke test: fullchannel vs original (Phase 46 spirit) ===")
    from medmamba_ss_trm import medmamba_ss_hsi_small
    orig = medmamba_ss_hsi_small(num_classes=6)
    full = medmamba_ss_fullchannel_hsi_small(num_classes=6)
    x = torch.randn(2, 128, 11, 11)
    y = torch.randint(0, 6, (2,))

    orig.eval(); full.eval()
    with torch.no_grad():
        out_orig = orig(x)
        out_full = full(x)
    assert out_orig.shape == out_full.shape == (2, 6)
    print(f"  original      params={count_params(orig)/1e6:.3f}M  out={tuple(out_orig.shape)}")
    print(f"  full-channel  params={count_params(full)/1e6:.3f}M  out={tuple(out_full.shape)}")

    print("\n=== Backward pass / gradient-flow check (full-channel) ===")
    full.train()
    out = full(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    n_missing = sum(1 for p in full.parameters() if p.requires_grad and p.grad is None)
    print(f"  loss={loss.item():.4f}  params_without_grad={n_missing}")
    assert n_missing == 0

    print("\n=== Multi-level backbone outputs still match the original dict shape ===")
    feats = full.backbone.forward_features(x)
    print("  keys:", sorted(feats.keys()))
    assert set(feats.keys()) == {"feature_map", "pooled", "stage_pools", "stage_feature_maps",
                                  "stage_ctx_maps", "spectral_context", "band_weights"}

    print("\nAll smoke tests passed.")
