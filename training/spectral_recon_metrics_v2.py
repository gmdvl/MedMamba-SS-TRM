# -*- coding: utf-8 -*-
"""
training/spectral_recon_metrics_v2.py
========================================
MedMamba-SS-TRM v16 plan, Stage 2.4 (R-3/R-7) - reconstruction metrics computed in
REFLECTANCE units, with a guard against feeding them z-scored (possibly
negative) data.

`training/spectral_recon_metrics.py` is imported by `trainerg_v4/v6/v11`, so
it is FROZEN. `rmse`, `mae`, `psnr`, `ssim`, `pearson_correlation` and
`cosine_similarity` are imported UNCHANGED from it - they are well-defined in
any domain. `sam_deg`, `sid`, `peak_position_error` and
`compute_all_spectral_metrics` are redefined here:

  - spectral angle (SAM) is only meaningful for non-negative reflectance
    spectra - a "direction" in a space where negative coordinates are
    physically meaningless produces an angle that does not mean what SAM is
    supposed to mean;
  - SID renormalizes each spectrum to a probability distribution
    (`clamp(min=eps)` then divide by the sum) - on z-scored input this
    silently discards most of the signal and reports on whatever survived
    the clamp, which is worse than merely undefined.

`require_nonnegative=True` (the default here) RAISES if handed data with a
meaningful negative fraction, instead of the frozen module's silent
`clamp(min=eps)`. Small negatives from decoder error (as opposed to feeding
this module normalized rather than denormalized data by mistake) are
clamped at zero and the clamped fraction is reported alongside the metric.

This module also drops the frozen module's product-of-norms `+eps` cosine
form (`spectral_recon_metrics.sam_deg:44-53`, which `training/gan.py:20-45`
documents as numerically unsafe near +-1) and imports
`safe_cosine_similarity` from `training/gan.py` (frozen, imported not
reimplemented) instead.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from training.gan import safe_cosine_similarity  # frozen, imported not reimplemented
from training.spectral_recon_metrics import (  # frozen - correct in any domain, reused as-is
    rmse, mae, psnr, ssim, pearson_correlation, cosine_similarity, _flatten_spectra,
)

DEFAULT_NEGATIVE_FRACTION_LIMIT = 0.02  # > 2% meaningfully-negative pixels -> not reflectance data
_MEANINGFUL_NEGATIVE = 1e-3             # ignore decoder-error-scale negative noise


class NonReflectanceDataError(ValueError):
    """Raised by `require_nonnegative=True` when the input is not, in fact,
    non-negative reflectance-like data - e.g. it is still z-scored."""


def _negative_fraction(x: torch.Tensor) -> float:
    return float((x < -_MEANINGFUL_NEGATIVE).float().mean().item())


def _clamp_nonneg(x: torch.Tensor, require_nonnegative: bool, name: str, limit: float):
    frac = _negative_fraction(x)
    if require_nonnegative and frac > limit:
        raise NonReflectanceDataError(
            f"{name}: {frac:.1%} of values are meaningfully negative (< -{_MEANINGFUL_NEGATIVE}) - "
            f"this does not look like reflectance data (R-3/R-7). Denormalize with "
            f"training.normalization.denormalize before computing spectral-angle metrics, or pass "
            f"require_nonnegative=False if this is intentional.")
    return x.clamp(min=0.0), frac


def sam_deg(x_true: torch.Tensor, x_pred: torch.Tensor,
            require_nonnegative: bool = True,
            negative_fraction_limit: float = DEFAULT_NEGATIVE_FRACTION_LIMIT) -> float:
    """Spectral Angle Mapper, in degrees, averaged over all pixels. Requires
    (by default) non-negative reflectance-domain input; small negatives are
    clamped to zero rather than raised on."""
    true_vec, _ = _clamp_nonneg(_flatten_spectra(x_true), require_nonnegative, "sam_deg(x_true)",
                                 negative_fraction_limit)
    pred_vec, _ = _clamp_nonneg(_flatten_spectra(x_pred), require_nonnegative, "sam_deg(x_pred)",
                                 negative_fraction_limit)
    cos_sim = safe_cosine_similarity(true_vec, pred_vec)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-3, 1.0 - 1e-3)
    return float(torch.rad2deg(torch.acos(cos_sim)).mean().item())


def sid(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-8,
        require_nonnegative: bool = True,
        negative_fraction_limit: float = DEFAULT_NEGATIVE_FRACTION_LIMIT) -> float:
    """Spectral Information Divergence. Requires (by default) non-negative
    reflectance-domain input, since each spectrum is renormalized into a
    probability distribution over bands."""
    true_vec, _ = _clamp_nonneg(_flatten_spectra(x_true), require_nonnegative, "sid(x_true)",
                                 negative_fraction_limit)
    pred_vec, _ = _clamp_nonneg(_flatten_spectra(x_pred), require_nonnegative, "sid(x_pred)",
                                 negative_fraction_limit)
    true_vec = true_vec.clamp(min=eps)
    pred_vec = pred_vec.clamp(min=eps)
    p = true_vec / true_vec.sum(dim=-1, keepdim=True)
    q = pred_vec / pred_vec.sum(dim=-1, keepdim=True)
    kl_pq = (p * (p / q).log()).sum(dim=-1)
    kl_qp = (q * (q / p).log()).sum(dim=-1)
    return float((kl_pq + kl_qp).mean().item())


def peak_position_error(x_true: torch.Tensor, x_pred: torch.Tensor,
                         wavelengths: Optional[torch.Tensor] = None) -> float:
    """Byte-identical logic to the frozen module's version - peak-index
    argmax is well-defined regardless of sign, so no non-negativity guard
    is needed here. Ported (not imported) only because the frozen module
    isn't meant to be depended on piecemeal by this one; behavior matches."""
    true_vec = _flatten_spectra(x_true)
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


def compute_all_spectral_metrics(x_true: torch.Tensor, x_pred: torch.Tensor,
                                  data_range: float = 1.0,
                                  wavelengths=None,
                                  include_optional: bool = True,
                                  require_nonnegative: bool = True,
                                  negative_fraction_limit: float = DEFAULT_NEGATIVE_FRACTION_LIMIT
                                  ) -> Dict[str, float]:
    """The full metric set, in reflectance units. `rmse`/`mae`/`pearson_
    correlation`/`cosine_similarity`/`psnr`/`ssim` come from the frozen
    module unchanged; `sam_deg`/`sid`/`peak_position_error` come from this
    one, with the non-negativity guard on the two that need it."""
    true_neg_frac = _negative_fraction(x_true)
    pred_neg_frac = _negative_fraction(x_pred)
    out = {
        "rmse": rmse(x_true, x_pred),
        "mae": mae(x_true, x_pred),
        "sam_deg": sam_deg(x_true, x_pred, require_nonnegative=require_nonnegative,
                            negative_fraction_limit=negative_fraction_limit),
        "sid": sid(x_true, x_pred, require_nonnegative=require_nonnegative,
                   negative_fraction_limit=negative_fraction_limit),
        "pearson_correlation": pearson_correlation(x_true, x_pred),
        "cosine_similarity": cosine_similarity(x_true, x_pred),
        "peak_position_error": peak_position_error(x_true, x_pred, wavelengths=wavelengths),
        "negative_fraction_true": true_neg_frac,
        "negative_fraction_pred": pred_neg_frac,
    }
    if include_optional:
        out["psnr"] = psnr(x_true, x_pred, data_range=data_range)
        out["ssim"] = ssim(x_true, x_pred, data_range=data_range)
    return out


def compute_per_sample_spectral_metrics(x_true: torch.Tensor, x_pred: torch.Tensor,
                                         wavelengths=None, eps: float = 1e-8,
                                         require_nonnegative: bool = True,
                                         negative_fraction_limit: float = DEFAULT_NEGATIVE_FRACTION_LIMIT):
    """Per-pixel SAM/RMSE/MAE/SID/Peak-Position-Error in reflectance units,
    for the reconstruction visualizations/histograms (R-6)."""
    true_vec_raw = _flatten_spectra(x_true)
    pred_vec_raw = _flatten_spectra(x_pred)
    true_vec, _ = _clamp_nonneg(true_vec_raw, require_nonnegative, "per_sample(x_true)",
                                 negative_fraction_limit)
    pred_vec, _ = _clamp_nonneg(pred_vec_raw, require_nonnegative, "per_sample(x_pred)",
                                 negative_fraction_limit)

    rmse_per = torch.sqrt(((pred_vec_raw - true_vec_raw) ** 2).mean(dim=-1))
    mae_per = (pred_vec_raw - true_vec_raw).abs().mean(dim=-1)

    cos_sim = safe_cosine_similarity(true_vec, pred_vec)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-3, 1.0 - 1e-3)
    sam_per = torch.rad2deg(torch.acos(cos_sim))

    t = true_vec.clamp(min=eps)
    p = pred_vec.clamp(min=eps)
    pn = t / t.sum(dim=-1, keepdim=True)
    qn = p / p.sum(dim=-1, keepdim=True)
    sid_per = (pn * (pn / qn).log()).sum(dim=-1) + (qn * (qn / pn).log()).sum(dim=-1)

    true_peak_idx = torch.argmax(true_vec_raw, dim=-1)
    pred_peak_idx = torch.argmax(pred_vec_raw, dim=-1)
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
