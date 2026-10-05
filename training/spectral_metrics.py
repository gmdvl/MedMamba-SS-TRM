# -*- coding: utf-8 -*-
"""
training/spectral_metrics.py
=============================
Phase 7 of the .npy plan: spectral / band-selection metrics.

MedMamba-SS-TRM's classifier has no spectral-reconstruction head, so SAM/SID/
reconstruction-RMSE/MAE genuinely don't apply here - `spectral_reconstruction_metrics()`
reports that explicitly (status: "not_applicable") instead of inventing
numbers, per the plan's "no fake values" requirement.

What the model *does* expose is `band_weights` from
`MedMambaSSBackbone.forward_features` (dynamic band selection, cfg.dynamic_band_selection).
When present, `band_selection_metrics()` turns per-patch band weights into
research-relevant summaries: which wavelengths are selected, how stable
selection is across the val set, spectral coverage, and channel sparsity.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def spectral_reconstruction_metrics() -> Dict:
    """Always "not applicable" for the current classification-only model."""
    return {
        "status": "not_applicable",
        "reason": "The trained model is a classifier (ClassificationHead) with no "
                  "spectral-reconstruction head, so SAM/SID/reconstruction RMSE/MAE "
                  "have no ground truth to compare against. Add a reconstruction head "
                  "to MedMambaSS (task='regression' predicting the input spectrum, or a "
                  "dedicated decoder) to enable these metrics.",
    }


@torch.no_grad()
def collect_band_weights(model, loader, device: str, wavelengths=None, sensor_range=None,
                          max_batches: Optional[int] = None) -> Optional[np.ndarray]:
    """Runs the backbone (not the full model, so we can reach `band_weights`)
    over `loader` and returns a [N, C] array of mean per-patch band weights,
    or None if the model's config has band selection disabled."""
    model.eval()
    backbone = model.backbone
    all_weights = []
    for i, (x, _y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        wl = wavelengths.to(device) if wavelengths is not None else None
        out = backbone.forward_features(x, wavelengths=wl, sensor_range=sensor_range)
        bw = out.get("band_weights")
        if bw is None:
            return None
        # bw: [B, Hp, Wp, C] -> mean over spatial dims -> [B, C]
        all_weights.append(bw.mean(dim=(1, 2)).float().cpu().numpy())
    if not all_weights:
        return None
    return np.concatenate(all_weights, axis=0)


def band_selection_metrics(band_weights: np.ndarray, wavelengths: Optional[np.ndarray] = None,
                            top_k_frac: float = 0.25) -> Dict:
    """`band_weights`: [N, C] mean weight per band per patch (already
    collected via `collect_band_weights`)."""
    n, c = band_weights.shape
    mean_weight = band_weights.mean(axis=0)                      # [C]
    std_weight = band_weights.std(axis=0)                        # [C]

    k = max(1, int(round(c * top_k_frac)))
    top_k_idx = np.argsort(-mean_weight)[:k]

    # selection stability: how consistently the same bands land in each
    # patch's own top-k, across the whole set (1.0 = perfectly stable)
    per_patch_topk = np.argsort(-band_weights, axis=1)[:, :k]     # [N, k]
    selection_freq = np.zeros(c)
    for row in per_patch_topk:
        selection_freq[row] += 1
    selection_freq /= n
    stability = float(selection_freq[top_k_idx].mean())

    # channel sparsity: effective number of bands carrying weight (inverse
    # participation ratio), normalized to [0,1] where 1 = maximally sparse
    p = mean_weight / max(mean_weight.sum(), 1e-12)
    effective_bands = 1.0 / max(float((p ** 2).sum()), 1e-12)
    sparsity = float(1.0 - effective_bands / c)

    # energy retained by the top-k bands
    energy_retained = float(mean_weight[top_k_idx].sum() / max(mean_weight.sum(), 1e-12))

    result = {
        "n_bands": int(c),
        "top_k": int(k),
        "top_k_fraction": top_k_frac,
        "selected_band_indices": top_k_idx.tolist(),
        "band_importance": mean_weight.tolist(),
        "band_importance_std": std_weight.tolist(),
        "selection_frequency": selection_freq.tolist(),
        "selection_stability": stability,
        "channel_sparsity": sparsity,
        "energy_retained_top_k": energy_retained,
        "redundancy_score": float(1.0 - stability) if k > 1 else 0.0,
    }

    if wavelengths is not None:
        wl = np.asarray(wavelengths)
        selected_wl = wl[top_k_idx]
        result["selected_wavelengths_nm"] = selected_wl.tolist()
        result["average_selected_wavelength_nm"] = float(selected_wl.mean())
        full_range = wl.max() - wl.min()
        coverage = (selected_wl.max() - selected_wl.min()) / full_range if full_range > 0 else 0.0
        result["spectral_coverage"] = float(coverage)

    return result
