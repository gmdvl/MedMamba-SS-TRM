# -*- coding: utf-8 -*-
"""
training/hsi_baselines.py
=============================
Two published hyperspectral classifiers, re-implemented for the paper's
11 x 11 x C patches so the primary dataset has a comparison against methods
that actually read the spectral axis (MedMamba does not).

HybridSN (Roy et al., IEEE GRSL 2020)
    Three 3-D convolutions (8, 16, 32 filters; spectral kernels 7, 5, 3;
    spatial 3 x 3), one 2-D convolution (64 filters), then 256 -> 128 ->
    classes with dropout 0.4. The original works on 25 x 25 windows of 30 PCA
    components; here the window is 11 x 11 and the input is the 32 selected
    bands. No padding, as in the original: 11 -> 9 -> 7 -> 5 -> 3 spatially.

SpectralFormer (Hong et al., IEEE TGRS 2022), patch-wise CAF variant
    One token per band built by group-wise spectral embedding (the band and
    its `near_band - 1` neighbours, flattened over the spatial window), a class
    token, a learned position embedding, a pre-norm Transformer (dim 64,
    depth 5, 4 heads of 16, MLP 8, dropout 0.1) whose layers from the third on
    receive the output of the layer two below through a 1 x 2 convolution
    (cross-layer adaptive fusion, CAF), and a LayerNorm + linear head on the
    class token. Hyperparameters are the official implementation's defaults.
    Neighbouring bands wrap around at the ends of the spectrum, as in the
    official `gain_neighborhood_band`.

Both take `[B, C, H, W]` and return logits `[B, num_classes]`. Neither is
band-count agnostic: HybridSN's 2-D convolution and SpectralFormer's position
embedding both have a shape that depends on C.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridSN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, patch: int = 11):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3)), nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3)), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)), nn.ReLU(inplace=True),
        )
        spec = in_channels - 6 - 4 - 2
        if spec < 1:
            raise ValueError(f"HybridSN needs at least 13 bands, got {in_channels}")
        side = patch - 6
        self.conv2d = nn.Sequential(nn.Conv2d(32 * spec, 64, kernel_size=3), nn.ReLU(inplace=True))
        side -= 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * side * side, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B, C, H, W]
        h = self.conv3d(x.unsqueeze(1))                         # [B, 32, C', H', W']
        b, f, c, hh, ww = h.shape
        h = self.conv2d(h.reshape(b, f * c, hh, ww))
        return self.head(h)


class _PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        return self.to_out((attn @ v).transpose(1, 2).reshape(b, n, -1))


class SpectralFormer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, patch: int = 11, near_band: int = 3,
                 dim: int = 64, depth: int = 5, heads: int = 4, dim_head: int = 16,
                 mlp_dim: int = 8, dropout: float = 0.1, emb_dropout: float = 0.1):
        super().__init__()
        if near_band % 2 != 1:
            raise ValueError("near_band must be odd")
        self.near_band = near_band
        n_tok = in_channels + 1
        self.embed = nn.Linear(patch * patch * near_band, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_tok, dim))
        self.drop = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([nn.ModuleList([
            _PreNormResidual(dim, _Attention(dim, heads, dim_head, dropout)),
            _PreNormResidual(dim, nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                                nn.Linear(mlp_dim, dim), nn.Dropout(dropout))),
        ]) for _ in range(depth)])
        self.skipcat = nn.ModuleList([nn.Conv2d(n_tok, n_tok, kernel_size=(1, 2))
                                      for _ in range(max(0, depth - 2))])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def _group_embed(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        r = self.near_band // 2
        groups = [torch.roll(flat, shifts=s, dims=1) for s in range(r, -r - 1, -1)]
        return torch.cat(groups, dim=2)                          # [B, C, near_band * H * W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B, C, H, W]
        t = self.embed(self._group_embed(x))
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1) + self.pos
        t = self.drop(t)
        history = []
        for i, (attn, ff) in enumerate(self.layers):
            history.append(t)
            if i > 1:
                pair = torch.stack([t, history[i - 2]], dim=3)  # [B, N, dim, 2]
                t = self.skipcat[i - 2](pair).squeeze(3)
            t = ff(attn(t))
        return self.head(t[:, 0])


BASELINES = {"hybridsn": HybridSN, "spectralformer": SpectralFormer}


def build_baseline(name: str, in_channels: int, num_classes: int, patch: int = 11) -> nn.Module:
    return BASELINES[name](in_channels, num_classes, patch=patch)
