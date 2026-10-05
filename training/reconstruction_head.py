# -*- coding: utf-8 -*-
"""
training/reconstruction_head.py
================================
Main Development Plan, Phase 34 - Correct the Reconstruction Pathway, and
Phase 35 - Reconstruction Should Be Auxiliary. Also Phase 39 (Spectral
Dropout), since it naturally lives alongside the wrapper that owns the
model's input.

WHY THIS FILE EXISTS
---------------------
`GMedMambaReconWrapper` (train_example-gemini-v5.py) and
`GMedMambaGANWrapper` (train_example_v6.py) both build the reconstruction
like this:

    logits    = self.base_model(x)          # runs the WHOLE backbone
    x_recon   = self.recon_head(x)           # a tiny conv net on the RAW input

`x_recon` never actually looks at anything the backbone learned - the
"reconstruction" pathway and the classification pathway don't share any
computation beyond both consuming `x`. That contradicts the plan's intended
architecture:

    Input -> MedMamba-SS-TRM -> latent representation -> reconstruction decoder
             -> reconstructed HSI

`MedMambaSSTRMLatentReconWrapper` below fixes this: the decoder consumes the
backbone's `feature_map` (the same multi-stage spatial-spectral
representation the classification head reads from `stage_pools`), so the
reconstruction loss actually backpropagates into - and can improve - the
shared encoder, not just a bolted-on decoder.

It keeps the exact `forward(x) -> (logits, x_recon)` contract that
TrainerG_v3/v4/v5 already expect, so it's a drop-in replacement for the two
older wrappers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralDropout(nn.Module):
    """Main Development Plan, Phase 39. Train-time-only: randomly zeroes a
    random subset of spectral bands on the model's input, to discourage the
    network from memorizing exact spectral signatures rather than learning
    generalizable spectral-shape features. Off (no-op) whenever the model is
    in eval mode, and off entirely if `drop_prob <= 0`.

    Caution (per the plan): excessive masking can remove diagnostically
    useful spectral information - keep `max_frac_bands` small and validate
    that classification/reconstruction metrics don't regress before using
    this beyond an ablation.
    """

    def __init__(self, drop_prob: float = 0.05, max_frac_bands: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.max_frac_bands = max_frac_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0:
            return x
        if torch.rand(1).item() > self.drop_prob:
            return x
        C = x.shape[1]
        n_drop = max(1, int(round(C * self.max_frac_bands)))
        idx = torch.randperm(C, device=x.device)[:n_drop]
        x = x.clone()
        x[:, idx, :, :] = 0.0
        return x


class LatentReconstructionDecoder(nn.Module):
    """Reconstructs the input spectral cube from the backbone's LEARNED
    spatial feature map (`MedMambaSSBackbone.forward_features()["feature_map"]`),
    upsampling it back to the input's spatial resolution (backbone stages
    downsample via PatchMerging2D, so the feature map is smaller than the
    input whenever `patch_size > 1` or there's more than one stage)."""

    def __init__(self, latent_channels: int, out_channels: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or max(32, latent_channels // 2)
        self.proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.out_conv = nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)

    def forward(self, feature_map: torch.Tensor, target_hw) -> torch.Tensor:
        # feature_map: [B, H', W', D] channels-last (as MedMambaSSBackbone returns it)
        x = feature_map.permute(0, 3, 1, 2).contiguous()          # [B, D, H', W']
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)                                   # normalized [0,1] reconstruction


class MedMambaSSTRMLatentReconWrapper(nn.Module):
    """`forward(x) -> (logits, x_recon)` - same contract as
    GMedMambaReconWrapper/GMedMambaGANWrapper, but `x_recon` is decoded from
    the backbone's learned `feature_map` (Phase 34) instead of from `x`
    directly, and reconstruction is wired as a genuinely auxiliary loss
    (Phase 35: `lambda_mse`/`lambda_sam` in the trainer control how much it
    matters, and can be set to 0 to disable it without changing this class).
    """

    def __init__(self, base_model, in_channels: int, spectral_dropout: float = 0.0):
        super().__init__()
        self.base_model = base_model  # a medmamba_ss_trm.MedMambaSS, task="classification"
        latent_channels = base_model.cfg.dims[-1]
        self.decoder = LatentReconstructionDecoder(latent_channels, in_channels)
        self.spectral_dropout = SpectralDropout(drop_prob=spectral_dropout) if spectral_dropout > 0 else None

    def forward(self, x: torch.Tensor, wavelengths=None, sensor_range=None):
        x_in = self.spectral_dropout(x) if self.spectral_dropout is not None else x
        backbone_out = self.base_model.backbone.forward_features(
            x_in, wavelengths=wavelengths, sensor_range=sensor_range)
        # v15 R4.4 - `medmamba_ss_trm.RecursiveHead` now has an explicit
        # `forward_features(dict) -> logits` instead of overloading `forward`'s
        # return type on its argument type. `ClassificationHead` (split/
        # fullchannel/efficient) has no such method and still takes the dict
        # through `__call__`.
        head = self.base_model.head
        logits = (head.forward_features(backbone_out) if hasattr(head, "forward_features")
                  else head(backbone_out))
        x_recon = self.decoder(backbone_out["feature_map"], target_hw=x.shape[-2:])
        return logits, x_recon


# Name before the 2026-10-02 rename; the frozen archive/train_example_v7/v14/v15.py import it.
GMedMambaLatentReconWrapper = MedMambaSSTRMLatentReconWrapper
