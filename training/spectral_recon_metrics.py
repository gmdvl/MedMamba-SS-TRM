# -*- coding: utf-8 -*-
"""
training/spectral_recon_metrics.py
===================================
Phase 3 of improve-prompt_v5.plan.md: expands validation-time spectral
reconstruction metrics from {spectral_recon_loss, SAM, RMSE} to the full set

    RMSE, MAE, SAM, SID, Pearson Correlation, Cosine Similarity, PSNR

All functions take x_true/x_pred as [B, C, H, W] float tensors (matching how
GMedMambaReconWrapper/GMedMambaGANWrapper reconstruct the input cube) and
return plain Python floats, so they're cheap to accumulate per-batch and log.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _flatten_spectra(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B*H*W, C] (one spectrum per pixel)."""
    x = x.float()
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(-1, c)


def rmse(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    return float(torch.sqrt(F.mse_loss(x_pred.float(), x_true.float())).item())


def mae(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    return float(F.l1_loss(x_pred.float(), x_true.float()).item())


def psnr(x_true: torch.Tensor, x_pred: torch.Tensor, data_range: float = 1.0, eps: float = 1e-12) -> float:
    mse = F.mse_loss(x_pred.float(), x_true.float()).item()
    if mse <= eps:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range ** 2 / mse)).item())


def sam_deg(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Spectral Angle Mapper, in degrees, averaged over all pixels."""
    true_vec = _flatten_spectra(x_true)
    pred_vec = _flatten_spectra(x_pred)
    dot = torch.sum(true_vec * pred_vec, dim=-1)
    norm_true = torch.norm(true_vec, p=2, dim=-1)
    norm_pred = torch.norm(pred_vec, p=2, dim=-1)
    cos_sim = torch.clamp(dot / (norm_true * norm_pred + eps), -1.0 + eps, 1.0 - eps)
    return float(torch.rad2deg(torch.acos(cos_sim)).mean().item())


def sid(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Spectral Information Divergence: treats each pixel's spectrum as a
    probability distribution over bands (after clamping to positive values
    and renormalizing) and computes the symmetric KL divergence between the
    true and reconstructed distributions, averaged over pixels."""
    true_vec = _flatten_spectra(x_true).clamp(min=eps)
    pred_vec = _flatten_spectra(x_pred).clamp(min=eps)
    p = true_vec / true_vec.sum(dim=-1, keepdim=True)
    q = pred_vec / pred_vec.sum(dim=-1, keepdim=True)
    kl_pq = (p * (p / q).log()).sum(dim=-1)
    kl_qp = (q * (q / p).log()).sum(dim=-1)
    return float((kl_pq + kl_qp).mean().item())


def pearson_correlation(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Per-pixel Pearson correlation across the spectral dimension, averaged
    over pixels (1.0 = perfectly correlated spectral shape)."""
    true_vec = _flatten_spectra(x_true)
    pred_vec = _flatten_spectra(x_pred)
    true_c = true_vec - true_vec.mean(dim=-1, keepdim=True)
    pred_c = pred_vec - pred_vec.mean(dim=-1, keepdim=True)
    num = (true_c * pred_c).sum(dim=-1)
    den = torch.norm(true_c, p=2, dim=-1) * torch.norm(pred_c, p=2, dim=-1) + eps
    corr = torch.clamp(num / den, -1.0, 1.0)
    return float(corr.mean().item())


def cosine_similarity(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-8) -> float:
    true_vec = _flatten_spectra(x_true)
    pred_vec = _flatten_spectra(x_pred)
    return float(F.cosine_similarity(true_vec, pred_vec, dim=-1, eps=eps).mean().item())


def peak_position_error(x_true: torch.Tensor, x_pred: torch.Tensor,
                         wavelengths: "torch.Tensor | None" = None) -> float:
    """Main Development Plan, Phase 4: 'Peak Position Error'.

    For every pixel spectrum, finds the band index (or wavelength, if
    `wavelengths` - a 1D tensor of length C - is given) at which the true
    spectrum peaks, and the band index/wavelength at which the reconstructed
    spectrum peaks, and returns the mean absolute difference between them
    averaged over all pixels. Reported in wavelength units (nm) when
    `wavelengths` is given, otherwise in band-index units.
    """
    true_vec = _flatten_spectra(x_true)   # [N, C]
    pred_vec = _flatten_spectra(x_pred)
    true_peak_idx = torch.argmax(true_vec, dim=-1)
    pred_peak_idx = torch.argmax(pred_vec, dim=-1)

    if wavelengths is not None:
        wl = wavelengths.to(true_vec.device).float()
        true_peak = wl[true_peak_idx]
        pred_peak = wl[pred_peak_idx]
    else:
        true_peak = true_peak_idx.float()
        pred_peak = pred_peak_idx.float()

    return float(torch.abs(true_peak - pred_peak).mean().item())


def ssim(x_true: torch.Tensor, x_pred: torch.Tensor, data_range: float = 1.0,
         c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> float:
    """Optional metric (Main Development Plan, Phase 4). Structural
    Similarity computed per-pixel across the spectral dimension (treating
    each pixel's spectrum as a 1D signal, matching how SAM/SID/Pearson are
    computed here) and averaged over pixels. This is a lightweight
    dependency-free approximation, not the windowed 2D SSIM used for
    natural images - appropriate here since the signal of interest is the
    spectral shape, not spatial texture."""
    true_vec = _flatten_spectra(x_true)
    pred_vec = _flatten_spectra(x_pred)
    c1 = (c1 * data_range ** 2)
    c2 = (c2 * data_range ** 2)

    mu_t = true_vec.mean(dim=-1)
    mu_p = pred_vec.mean(dim=-1)
    var_t = true_vec.var(dim=-1, unbiased=False)
    var_p = pred_vec.var(dim=-1, unbiased=False)
    cov_tp = ((true_vec - mu_t.unsqueeze(-1)) * (pred_vec - mu_p.unsqueeze(-1))).mean(dim=-1)

    numerator = (2 * mu_t * mu_p + c1) * (2 * cov_tp + c2)
    denominator = (mu_t ** 2 + mu_p ** 2 + c1) * (var_t + var_p + c2)
    ssim_per_pixel = numerator / denominator
    return float(ssim_per_pixel.mean().item())


def compute_all_spectral_metrics(x_true: torch.Tensor, x_pred: torch.Tensor,
                                  data_range: float = 1.0,
                                  wavelengths=None,
                                  include_optional: bool = True) -> Dict[str, float]:
    """Convenience wrapper computing the full Phase 4 metric set in one call
    (Main Development Plan: SAM, RMSE, MAE, SID, Pearson Correlation, Cosine
    Similarity, Peak Position Error, plus optional SSIM/PSNR)."""
    out = {
        "rmse": rmse(x_true, x_pred),
        "mae": mae(x_true, x_pred),
        "sam_deg": sam_deg(x_true, x_pred),
        "sid": sid(x_true, x_pred),
        "pearson_correlation": pearson_correlation(x_true, x_pred),
        "cosine_similarity": cosine_similarity(x_true, x_pred),
        "peak_position_error": peak_position_error(x_true, x_pred, wavelengths=wavelengths),
    }
    if include_optional:
        out["psnr"] = psnr(x_true, x_pred, data_range=data_range)
        out["ssim"] = ssim(x_true, x_pred, data_range=data_range)
    return out


def compute_per_sample_spectral_metrics(x_true: torch.Tensor, x_pred: torch.Tensor,
                                         wavelengths=None,
                                         eps: float = 1e-8):
    """Per-pixel (not batch-averaged) SAM/RMSE/MAE/SID/Peak-Position-Error,
    used to build the reconstruction visualizations and metric histograms in
    `training.plots` (Main Development Plan, Phase 4 'Reconstruction
    Visualizations' / 'Histograms') and to pick best/worst/median examples."""
    true_vec = _flatten_spectra(x_true)
    pred_vec = _flatten_spectra(x_pred)

    rmse_per = torch.sqrt(((pred_vec - true_vec) ** 2).mean(dim=-1))
    mae_per = (pred_vec - true_vec).abs().mean(dim=-1)

    dot = torch.sum(true_vec * pred_vec, dim=-1)
    norm_true = torch.norm(true_vec, p=2, dim=-1)
    norm_pred = torch.norm(pred_vec, p=2, dim=-1)
    cos_sim = torch.clamp(dot / (norm_true * norm_pred + eps), -1.0 + eps, 1.0 - eps)
    sam_per = torch.rad2deg(torch.acos(cos_sim))

    t = true_vec.clamp(min=eps)
    p = pred_vec.clamp(min=eps)
    pn = t / t.sum(dim=-1, keepdim=True)
    qn = p / p.sum(dim=-1, keepdim=True)
    sid_per = (pn * (pn / qn).log()).sum(dim=-1) + (qn * (qn / pn).log()).sum(dim=-1)

    true_peak_idx = torch.argmax(true_vec, dim=-1)
    pred_peak_idx = torch.argmax(pred_vec, dim=-1)
    if wavelengths is not None:
        wl = wavelengths.to(true_vec.device).float()
        ppe_per = (wl[true_peak_idx] - wl[pred_peak_idx]).abs()
    else:
        ppe_per = (true_peak_idx - pred_peak_idx).abs().float()

    return {
        "rmse": rmse_per.tolist(),
        "mae": mae_per.tolist(),
        "sam_deg": sam_per.tolist(),
        "sid": sid_per.tolist(),
        "peak_position_error": ppe_per.tolist(),
    }
