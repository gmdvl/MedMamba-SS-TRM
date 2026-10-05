# -*- coding: utf-8 -*-
"""
training/gan.py
================
Phase 2 of improve-prompt_v5.plan.md: the pieces needed to turn the existing
classification + spectral-reconstruction model into

    HSI -> Encoder -> Classification Head
                    -> Spectral Generator -> Reconstructed Spectrum -> Discriminator

  * SAMLoss           - differentiable Spectral Angle Mapper, used alongside
                         MSE for the reconstruction objective (spectral SHAPE
                         vs. spectral INTENSITY, per the plan's rationale).
  * SpectralDiscriminator - small patch-level conv discriminator that scores
                         whether a spectral cube "looks real"; trained with a
                         standard non-saturating GAN loss against the
                         generator (= the model's reconstruction head).

--------------------------------------------------------------------------
v12/v13 stability plan, Phase 15 - SAMLoss stabilization
--------------------------------------------------------------------------
The original formulation clamped `cos_sim` to `[-1+eps, 1-eps]` with
`eps=1e-8` before `acos`. `d/dx acos(x) = -1/sqrt(1-x^2)`, which blows up
as `x -> +-1` - with `eps=1e-8` the clamped boundary is only ~1.4e-4 away
from +-1, where the derivative is already ~7000. That is large enough to
produce enormous (though individually finite) gradients purely from the
SAM term, which is a very plausible contributor to "gradient norm becomes
non-finite" even when every forward value was itself finite.

`SAMLoss` below fixes this without changing its public contract
(`forward(x_true, x_pred) -> scalar radians`, same as before):
  - always computes in FP32 (`.float()`), regardless of the caller's AMP
    context;
  - clamps each vector's OWN norm to a floor (`clamp_min`, not
    `norm_true*norm_pred + eps`) so a genuinely zero/near-zero spectrum
    doesn't inflate the product's denominator error and produces a well-
    defined (not NaN) cosine similarity instead;
  - uses a much larger `angle_eps` (default `1e-3`) for the `acos` clamp,
    trading a small amount of angular resolution at the extremes for a
    bounded derivative (`1/sqrt(2*1e-3) =~ 22`, not `~7000`);
  - never returns NaN/Inf - `nan_to_num` is applied as a final defensive
    floor, so even an adversarial/degenerate input can't propagate a
    non-finite value into the total loss.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_cosine_similarity(a: torch.Tensor, b: torch.Tensor, norm_eps: float = 1e-6) -> torch.Tensor:
    """Cosine similarity along the last dim, with each vector's norm
    individually floored (not the product) so a zero/near-zero vector
    produces a well-defined (0-ish, not NaN/Inf) result rather than
    dividing by an arbitrarily small product of two tiny norms."""
    a = a.float()
    b = b.float()
    a_norm = torch.norm(a, p=2, dim=-1).clamp_min(norm_eps)
    b_norm = torch.norm(b, p=2, dim=-1).clamp_min(norm_eps)
    dot = torch.sum(a * b, dim=-1)
    return dot / (a_norm * b_norm)


class SAMLoss(nn.Module):
    """Mean Spectral Angle Mapper (in radians) between two [B, C, H, W]
    spectral cubes, differentiable so it can be backpropagated as a loss
    term (as opposed to trainerg_v3's compute_sam(), which is metrics-only).
    Lower is better; a perfect reconstruction has SAM == 0.

    `angle_eps` bounds how close `cos_sim` is allowed to get to +-1 before
    `acos` (Phase 15 - see module docstring for why the old `1e-8` default
    was unsafe); `norm_eps` floors each spectral vector's own L2 norm.
    """

    def __init__(self, eps: Optional[float] = None, angle_eps: float = 1e-3, norm_eps: float = 1e-6):
        super().__init__()
        # backward-compat: a caller passing the old positional/keyword `eps`
        # is treated as `angle_eps` (the parameter it actually controlled).
        self.angle_eps = eps if eps is not None else angle_eps
        self.norm_eps = norm_eps

    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_true.shape
        true_vec = x_true.float().permute(0, 2, 3, 1).reshape(-1, c)
        pred_vec = x_pred.float().permute(0, 2, 3, 1).reshape(-1, c)
        cos_sim = safe_cosine_similarity(true_vec, pred_vec, norm_eps=self.norm_eps)
        cos_sim = torch.clamp(cos_sim, -1.0 + self.angle_eps, 1.0 - self.angle_eps)
        angle = torch.acos(cos_sim)
        # Defensive floor: never let a degenerate input (e.g. every band
        # dropped to exactly 0 by spectral augmentation/dropout on BOTH
        # x_true and x_pred simultaneously) propagate NaN/Inf into the
        # total loss - replace any surviving non-finite element with 0
        # (angle 0 == "no penalty" for a pathological all-zero comparison,
        # which is the least-surprising fallback for an ill-defined angle).
        angle = torch.nan_to_num(angle, nan=0.0, posinf=math.pi, neginf=0.0)
        return angle.mean()


class SpectralDiscriminator(nn.Module):
    """PatchGAN-style discriminator over spectral cubes. Operates on small
    HSI patches (e.g. 11x11), so it deliberately uses few downsampling
    steps and falls back to a 1x1 conv "classifier" over whatever spatial
    size is left instead of assuming a fixed input resolution.
    """

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()

        def block(c_in, c_out, stride):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
                nn.InstanceNorm2d(c_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            block(in_channels, base_channels, stride=1),
            block(base_channels, base_channels * 2, stride=2),
            block(base_channels * 2, base_channels * 4, stride=1),
        )
        self.head = nn.Conv2d(base_channels * 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        logits = self.head(feat)          # [B, 1, H', W']
        return logits.mean(dim=(1, 2, 3))  # [B] - one real/fake logit per sample


def discriminator_loss(disc: nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
    """Standard non-saturating discriminator loss: real -> 1, fake -> 0.
    `x_fake` should already be detached by the caller before this is called
    so gradients don't flow back into the generator during the D step."""
    real_logits = disc(x_real)
    fake_logits = disc(x_fake)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)


def generator_adversarial_loss(disc: nn.Module, x_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: wants the discriminator to score the
    reconstruction as real (logit -> 1)."""
    fake_logits = disc(x_fake)
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
