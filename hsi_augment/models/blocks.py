# -*- coding: utf-8 -*-
"""
hsi_augment/models/blocks.py
============================
Building blocks shared by the CVAE (#23), the autoencoder and the latent
diffusion denoiser (#24). The patches are tiny (11x11, then 6x6 and 3x3 after
one and two stride-2 convolutions), so the blocks are plain 3x3 convolutions
with padding 1, which work at every one of those sizes.

Why GroupNorm and never BatchNorm
---------------------------------
MedMamba-SS already showed a BatchNorm artefact on this HSI data. For a
generator BN is worse still: its batch statistics couple the samples of a
batch (a class-balanced batch is not the distribution generated at export
time, one class at a time), and its running statistics make train and eval
mode compute different functions. GroupNorm normalises each sample on its own
and behaves identically in both modes, so `assert_no_batchnorm` is part of
every generator's tests.

What is here
------------
  group_norm(C)               GroupNorm with the largest group count <= 32 that divides C
  FiLM(cond_dim, C)           x * (1 + gamma(c)) + beta(c); zero-initialised, so the
                              conditioning starts as the identity
  ResBlock(in, out, cond_dim) GN -> SiLU -> conv3x3 -> GN -> FiLM -> SiLU -> dropout
                              -> conv3x3, plus a 1x1 skip when the width changes
  Downsample / Upsample       stride-2 conv (11 -> 6 -> 3) / ConvTranspose (3 -> 6 -> 12)
  crop_to(x, 11)              centre crop (the decoder's 12 -> 11)
  timestep_embedding(t, dim)  sinusoidal embedding (Vaswani et al. 2017; DDPM/ADM layout:
                              [cos | sin], frequencies max_period^(-i/half))
  TimestepEmbedder(dim)       sinusoidal -> Linear -> SiLU -> Linear
  assert_no_batchnorm(model)  raises if any BatchNorm/SyncBatchNorm is inside

ConvTranspose rather than bilinear upsampling for the decoder: its backward
pass is deterministic on CUDA (bilinear `interpolate`'s is not), which the
plan's bitwise-resume and `--deterministic` requirements need.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_GROUPS = 32


def group_norm(channels: int, max_groups: int = MAX_GROUPS, eps: float = 1e-5) -> nn.GroupNorm:
    """GroupNorm over `channels` with the largest group count <= `max_groups`
    that divides it (32 for 64/96/192/256 channels; 4 for a 4-channel latent)."""
    if channels <= 0:
        raise ValueError("channels must be > 0")
    groups = max(g for g in range(1, min(max_groups, channels) + 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels, eps=eps)


class FiLM(nn.Module):
    """Feature-wise linear modulation (Perez et al. 2018): x * (1 + gamma) + beta,
    with (gamma, beta) a linear function of the conditioning vector, broadcast
    over the spatial dims. The projection is zero-initialised, so an untrained
    FiLM is the identity and conditioning is learned as a deviation from it."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.proj = nn.Linear(cond_dim, 2 * self.channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        shape = (x.shape[0], self.channels) + (1,) * (x.ndim - 2)
        return x * (1.0 + gamma.reshape(shape)) + beta.reshape(shape)


class ResBlock(nn.Module):
    """Pre-activation residual block, GroupNorm + SiLU, optional FiLM conditioning.

        h = conv1(SiLU(GN(x)))
        h = conv2(dropout(SiLU(FiLM(GN(h), cond))))        FiLM only if cond_dim
        out = skip(x) + h                                  skip = 1x1 conv if in != out

    `zero_init=True` zero-initialises conv2, so every block starts as the
    identity (the ADM/Fixup trick): deep stacks train from step one without a
    warm-up of the normalisation scales."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None, *,
                 cond_dim: Optional[int] = None, dropout: float = 0.0, zero_init: bool = True):
        super().__init__()
        out_channels = int(out_channels or in_channels)
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = group_norm(out_channels)
        self.film = FiLM(cond_dim, out_channels) if cond_dim else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if zero_init:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.film is not None:
            if cond is None:
                raise ValueError("this ResBlock was built with cond_dim and needs a conditioning vector")
            h = self.film(h, cond)
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class Downsample(nn.Module):
    """3x3 stride-2 conv, padding 1: 11 -> 6 -> 3 (ceil(n / 2))."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, int(out_channels or in_channels), kernel_size=3, stride=2,
                              padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """4x4 stride-2 ConvTranspose, padding 1: exactly 2n (3 -> 6 -> 12; crop
    12 -> 11 with `crop_to`). Deterministic backward on CUDA, unlike bilinear."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, int(out_channels or in_channels), kernel_size=4,
                                       stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def crop_to(x: torch.Tensor, size: int) -> torch.Tensor:
    """Centre crop of the last two dims to size x size (12 -> 11 drops the
    last row and column: offset floor((12 - 11) / 2) = 0)."""
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if size > h or size > w:
        raise ValueError(f"cannot crop {h}x{w} to {size}x{size}")
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding of (fractional) timesteps t, shape (B,) -> (B, dim):
    [cos(t f_0..f_{h-1}) | sin(t f_0..f_{h-1})], f_i = max_period^(-i / h),
    h = dim // 2; an odd `dim` gets one zero column. Computed in float32."""
    if t.ndim != 1:
        t = t.reshape(-1)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device)
                      / max(half, 1))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SinusoidalTimeEmbedding(nn.Module):
    """`timestep_embedding` as a module (no parameters)."""

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        self.dim, self.max_period = int(dim), float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t, self.dim, self.max_period)


class TimestepEmbedder(nn.Module):
    """Sinusoidal(t) -> Linear -> SiLU -> Linear, the usual diffusion time MLP.
    Add a class embedding to its output to condition on both."""

    def __init__(self, dim: int, freq_dim: int = 128, max_period: float = 10_000.0):
        super().__init__()
        self.freq = SinusoidalTimeEmbedding(freq_dim, max_period)
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.freq(t))


_BATCHNORM_TYPES = (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)


def assert_no_batchnorm(model: nn.Module) -> None:
    """Raise TypeError naming every BatchNorm inside `model` (see module docstring)."""
    found = [name or "<root>" for name, m in model.named_modules() if isinstance(m, _BATCHNORM_TYPES)]
    if found:
        raise TypeError(f"BatchNorm is not allowed in the generators; found at {found}")
