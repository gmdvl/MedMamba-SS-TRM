# -*- coding: utf-8 -*-
"""
training/reconstruction_head_v2.py
=====================================
MedMamba-SS-TRM v16 plan, Stage 2.3 (R-3). `training/reconstruction_head.py` is
imported at `train_example_v15.py:108` and is therefore FROZEN.

`LatentReconstructionDecoderV2` subclasses `LatentReconstructionDecoder` -
`proj`/`out_conv` inherited verbatim - and changes only the final line of
`forward`: a configurable output activation instead of an always-on sigmoid.

Under any z-score normalization mode the reconstruction TARGET is
approximately `N(0, 1)` (unbounded, can be negative), while a sigmoid output
lives in `(0, 1)`. That floors `F.mse_loss` near 1.0 no matter how good the
decoder is - not merely a scale mismatch but a saturating nonlinearity the
decoder cannot get gradient through to fix. `out_activation="linear"` removes
the floor; `"sigmoid"` is kept (and remains the default via
`MedMambaSSTRMLatentReconWrapperV2`'s own default argument) for `per_sample_minmax`,
whose target genuinely lives in `[0, 1]`.

v16 selects `linear` for any z-score mode and `sigmoid` for min-max, and
RAISES at startup on a mismatch (`validate_out_activation_for_normalization`)
rather than training into a floor again.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.reconstruction_head import LatentReconstructionDecoder, SpectralDropout  # frozen

_ZSCORE_MODES = ("global_zscore", "per_patch_zscore")
_BOUNDED_MODES = ("per_sample_minmax",)


class LatentReconstructionDecoderV2(LatentReconstructionDecoder):
    def __init__(self, *a, out_activation: str = "linear", **kw):
        super().__init__(*a, **kw)  # proj / out_conv inherited verbatim
        if out_activation not in ("linear", "sigmoid"):
            raise ValueError(f"out_activation must be 'linear' | 'sigmoid', got {out_activation!r}")
        self.out_activation = out_activation

    def forward(self, feature_map: torch.Tensor, target_hw) -> torch.Tensor:
        x = feature_map.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.out_conv(self.proj(x))
        return torch.sigmoid(x) if self.out_activation == "sigmoid" else x


def resolve_out_activation(normalization: str) -> str:
    """`linear` for any z-score mode (target is ~N(0,1), unbounded, can be
    negative), `sigmoid` for `per_sample_minmax` (target is genuinely
    `[0, 1]`)."""
    if normalization in _ZSCORE_MODES:
        return "linear"
    if normalization in _BOUNDED_MODES:
        return "sigmoid"
    raise ValueError(f"unknown normalization mode {normalization!r}")


def validate_out_activation_for_normalization(out_activation: str, normalization: str) -> None:
    """Raises at startup on a mismatch instead of training into the MSE floor
    R-3 identified (`out_activation='sigmoid'` under a z-score target added
    ~1.0 to the reported loss - the v15 plan's "adds ~1.0 to the loss" note,
    train_example_v15.py:678, was the detach AND this, stacked)."""
    expected = resolve_out_activation(normalization)
    if out_activation != expected:
        raise ValueError(
            f"--recon_out_activation {out_activation!r} is inconsistent with "
            f"--normalization {normalization!r} (expected {expected!r}). Under a z-score "
            f"normalization the reconstruction target is ~N(0,1) and unbounded; a sigmoid output "
            f"in (0,1) floors F.mse_loss near 1.0 regardless of decoder quality (R-3). Pass "
            f"--recon_out_activation {expected!r}, or omit the flag and let it resolve "
            f"automatically from --normalization.")


class MedMambaSSTRMLatentReconWrapperV2(torch.nn.Module):
    """`forward(x) -> (logits, x_recon)` - same contract as
    `MedMambaSSTRMLatentReconWrapper`, built with the V2 decoder (configurable
    output activation) and, for a RECURSIVE base, decoding via
    `training.recursive_features.forward_deep_supervision_with_features`
    (Stage 2.2 / R-5) instead of `backbone.forward_features` - so the decode
    path is the SAME one `TrainerG_v12._forward_with_recon` uses in both
    training and validation.

    For a non-recursive base (`MedMambaSSBackbone`), decoding still goes
    through `backbone.forward_features`, exactly as the frozen V1 wrapper
    does - R-2/R-5 are recursive-architecture-specific defects (deep
    supervision's per-segment detach; the two different segment recursions).
    """

    def __init__(self, base_model, in_channels: int, out_activation: str = "linear",
                 spectral_dropout: float = 0.0):
        super().__init__()
        self.base_model = base_model
        latent_channels = base_model.cfg.dims[-1]
        self.decoder = LatentReconstructionDecoderV2(latent_channels, in_channels,
                                                       out_activation=out_activation)
        self.spectral_dropout = SpectralDropout(drop_prob=spectral_dropout) if spectral_dropout > 0 else None
        self.is_recursive = bool(getattr(base_model.cfg, "recursive", False))

    def forward(self, x: torch.Tensor, wavelengths=None, sensor_range=None):
        x_in = self.spectral_dropout(x) if self.spectral_dropout is not None else x

        if self.is_recursive:
            from training.recursive_features import forward_deep_supervision_with_features
            logits_list, _q_list, feat = forward_deep_supervision_with_features(
                self.base_model, x_in, wavelengths=wavelengths, sensor_range=sensor_range)
            logits = logits_list[-1]
            x_recon = self.decoder(feat, target_hw=x.shape[-2:])
            return logits, x_recon

        backbone_out = self.base_model.backbone.forward_features(
            x_in, wavelengths=wavelengths, sensor_range=sensor_range)
        head = self.base_model.head
        logits = (head.forward_features(backbone_out) if hasattr(head, "forward_features")
                  else head(backbone_out))
        x_recon = self.decoder(backbone_out["feature_map"], target_hw=x.shape[-2:])
        return logits, x_recon


# `--recon_mode raw_input`, copied verbatim from `archive/train_example_v7.py:56-74`
# (see `tests/test_live_copies_match_archive.py`).
class MedMambaSSTRMRawReconWrapper(nn.Module):
    """The OLD (Phase-34-violating) reconstruction path: the decoder sees
    only the raw input, never the backbone's learned representation. Kept
    only so `--recon_mode raw_input` can be A/B-ablated against the fix."""

    def __init__(self, base_model, in_channels):
        super().__init__()
        self.base_model = base_model
        hidden_channels = max(16, in_channels // 2)
        self.recon_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels), nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x, **_ignored):
        logits = self.base_model(x)
        x_recon = self.recon_head(x)
        return logits, x_recon


# Name before the 2026-10-02 rename; the frozen archive/train_example_v16.py imports it.
GMedMambaLatentReconWrapperV2 = MedMambaSSTRMLatentReconWrapperV2
