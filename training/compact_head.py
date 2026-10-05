# -*- coding: utf-8 -*-
"""
training/compact_head.py
=========================
Main Development Plan, Phase 25 - Reduce Classifier Parameters, and
Phase 33 - Multi-Level Feature Aggregation Without Parameter Explosion.

`medmamba_ss_trm.ClassificationHead` concatenates EVERY stage's pooled feature
plus the spectral summary (`sum(cfg.dims) + cfg.d_ctx` inputs - e.g.
96+192+384+768+64 = 1504 for `medmamba_ss_tiny`) into one big vector, then
runs a 2-layer MLP down to `pooled_dim // 2` before the final linear. That's
exactly the "very large raw concatenation followed by a huge projection"
Phase 33 asks to avoid.

`CompactClassificationHead` instead: GAP (already done for you -
`backbone_out["pooled"]` IS the final stage's global-average-pooled
feature) -> two small per-branch projections (final spatial feature,
spectral summary) -> add -> LayerNorm -> Dropout -> a single Linear to
`num_classes`. No hidden MLP layer, no multi-stage concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactClassificationHead(nn.Module):
    def __init__(self, cfg, num_classes: int, dropout: float = 0.1):
        super().__init__()
        final_dim = cfg.dims[-1]
        self.spatial_proj = nn.Linear(final_dim, final_dim)
        self.spectral_proj = nn.Linear(cfg.d_ctx, final_dim)
        self.norm = nn.LayerNorm(final_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_dim, num_classes)

    def forward(self, backbone_out) -> torch.Tensor:
        feat = self.spatial_proj(backbone_out["pooled"]) + self.spectral_proj(backbone_out["spectral_context"])
        feat = self.norm(feat)
        feat = self.dropout(feat)
        return self.fc(feat)
