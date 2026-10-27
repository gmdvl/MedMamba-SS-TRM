# -*- coding: utf-8 -*-
"""
hsi_augment/tissue_figures.py
=============================
The sign-off figures of the `tissue-mask` stage (issue #13; Plan §Phase 2,
item 6: "the user approves tau before Phase 4's pool is built"). Kept apart
from `tissue.py` so the numeric library (which QC and the report import to
score val/test and synthetic patches) never imports matplotlib.

    tissue_maps_<class>.png   per class, one row per capture: an RGB composite
                              of the per-patch mean spectra (bands nearest
                              640/550/460 nm), the score t on the capture grid,
                              and the mask t >= tau
    tissue_score_hist.png     log t of the threshold sample, the two GMM
                              components, tau (GMM) and tau (Otsu)
    s_glass_spectrum.png      s_glass, every capture's glass candidate and each
                              class's mean unit spectrum; lines are drawn per
                              spectral segment, never across the gap
    glass_rule.png            the minority glass fraction per train patient and
                              pooled, against the pre-registered 10%
    grid_check.png            edge correlations of true vs shuffled grid
                              neighbours, and every capture's two deltas

Why the gap matters here: trainsel has no band between 628.9 and 850.9 nm. A
line drawn across it shows an interpolation that was never measured, so the
segments come from `BuildAdapter.spectral_segments()` (gaps detected from the
wavelengths) and the gap is shaded. Nothing here imports
`scripts/make_figures_v19.py`, which loads v8 data at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402

# Categorical slots in fixed order (one per class, then extras); text and
# surface tokens. Sequential magnitude = one hue, light -> dark.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4")
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
GLASS_COLOR = "#e4e3df"
TISSUE_COLOR = "#1f5fae"
SCORE_CMAP = "Blues"
DPI = 110


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


# ============================================================================
# capture mosaics
# ============================================================================

@dataclass(frozen=True)
class MosaicPanel:
    title: str
    rgb: np.ndarray      # (rows, cols, 3) in [0, 1]
    score: np.ndarray    # (rows, cols) degrees
    mask: np.ndarray     # (rows, cols) bool, True = tissue


def plot_capture_mosaics(path: Path, class_name: str, panels: Sequence[MosaicPanel], tau_deg: float,
                         score_vmax: float, rgb_label: str = "") -> Path:
    n = len(panels)
    rows, cols = panels[0].score.shape
    row_h = 3.0 * rows / cols + 0.9
    fig, axes = plt.subplots(n, 3, figsize=(14.0, row_h * n + 0.8), squeeze=False, layout="constrained")
    norm = Normalize(vmin=0.0, vmax=float(score_vmax))
    mask_cmap = ListedColormap([GLASS_COLOR, TISSUE_COLOR])
    for i, pnl in enumerate(panels):
        a_rgb, a_score, a_mask = axes[i]
        a_rgb.imshow(pnl.rgb, interpolation="nearest")
        im = a_score.imshow(pnl.score, cmap=SCORE_CMAP, norm=norm, interpolation="nearest")
        a_mask.imshow(pnl.mask.astype(np.int8), cmap=mask_cmap, vmin=0, vmax=1, interpolation="nearest")
        for ax in (a_rgb, a_score, a_mask):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(MUTED)
        a_rgb.set_ylabel(pnl.title, fontsize=8, color=TEXT, rotation=0, ha="right", va="center", labelpad=8)
        cb = fig.colorbar(im, ax=a_score, fraction=0.035, pad=0.02)
        cb.ax.axhline(tau_deg, color=TEXT, linewidth=1.5)
        cb.ax.tick_params(labelsize=7, colors=TEXT_2)
        cb.set_label("t (deg)", fontsize=7, color=TEXT_2)
        tissue = float(pnl.mask.mean())
        a_mask.text(1.02, 0.5, f"tissue {tissue:.0%}\nglass {1 - tissue:.0%}", transform=a_mask.transAxes,
                    fontsize=8, color=TEXT_2, va="center")
    axes[0, 0].set_title("RGB composite of per-patch mean spectra" + (f"\n{rgb_label}" if rgb_label else ""),
                         fontsize=8, color=TEXT)
    axes[0, 1].set_title("tissue score t = SAM(patch, s_glass)\nblack line on the colour bar = tau",
                         fontsize=8, color=TEXT)
    axes[0, 2].set_title(f"mask t >= tau = {tau_deg:.3f} deg\ntissue dark, glass light", fontsize=8, color=TEXT)
    fig.suptitle(f"{class_name}: {n} train capture(s) on the {rows} x {cols} patch grid", fontsize=11,
                 color=TEXT)
    return _save(fig, path)


# ============================================================================
# the threshold
# ============================================================================

def plot_score_histogram(path: Path, sample_deg: np.ndarray, fit: Any) -> Path:
    """`fit` is a `tissue.ThresholdFit`. Densities are per unit ln t, so the
    histogram and the GMM components share one axis."""
    v = np.log(np.maximum(sample_deg[np.isfinite(sample_deg)], fit.log_floor_deg))
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    _style(ax)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 81)
    counts, edges = np.histogram(v, bins=bins)
    width = edges[1] - edges[0]
    ax.bar(np.exp(edges[:-1]), counts / (counts.sum() * width), width=np.diff(np.exp(edges)), align="edge",
           color=MUTED, edgecolor="white", linewidth=0.5, label=f"train sample (n = {v.size:,})")
    if fit.gmm is not None:
        xs = np.linspace(lo, hi, 400)
        total = np.zeros_like(xs)
        for k, (w, m, var) in enumerate(zip(fit.gmm["weights"], fit.gmm["means_log"], fit.gmm["variances_log"])):
            pdf = w * np.exp(-(xs - m) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
            total += pdf
            ax.plot(np.exp(xs), pdf, color=SERIES[k], linewidth=2,
                    label=f"GMM {'glass' if k == 0 else 'tissue'} (w = {w:.2f}, mean {np.exp(m):.2f} deg)")
        ax.plot(np.exp(xs), total, color=TEXT, linewidth=1, linestyle=":", label="GMM total")
        if fit.gmm.get("tau_deg"):
            ax.axvline(fit.gmm["tau_deg"], color=TEXT, linewidth=1.5,
                       label=f"tau GMM = {fit.gmm['tau_deg']:.3f} deg")
    ax.axvline(fit.otsu["tau_deg"], color=TEXT_2, linewidth=1.5, linestyle="--",
               label=f"tau Otsu = {fit.otsu['tau_deg']:.3f} deg")
    ax.set_xscale("log")
    ax.set_xlabel("tissue score t (degrees, log scale)", color=TEXT, fontsize=9)
    ax.set_ylabel("density per unit ln t", color=TEXT, fontsize=9)
    ax.set_title(f"Threshold: tau = {fit.tau_deg:.3f} deg ({fit.method_used}"
                 + (f"; fallback: {fit.fallback_reason}" if fit.fallback_reason else "") + ")",
                 fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_2)
    return _save(fig, path)


# ============================================================================
# the glass reference
# ============================================================================

def plot_glass_spectrum(path: Path, wavelengths: np.ndarray, segments: Sequence[Tuple[int, int]],
                        s_glass: np.ndarray, candidates: np.ndarray,
                        class_means: Mapping[str, np.ndarray]) -> Path:
    wl = np.asarray(wavelengths, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    _style(ax)

    def segmented(y, **kw):
        label = kw.pop("label", None)
        for j, (a, b) in enumerate(segments):
            ax.plot(wl[a:b], y[a:b], label=label if j == 0 else None, **kw)

    for k, cand in enumerate(np.asarray(candidates)):
        segmented(cand, color=MUTED, linewidth=0.6, alpha=0.35,
                  label=f"capture glass candidates ({len(candidates)})" if k == 0 else None)
    for k, (name, m) in enumerate(class_means.items()):
        segmented(np.asarray(m), color=SERIES[k % len(SERIES)], linewidth=2, label=f"{name} mean (all patches)")
    segmented(np.asarray(s_glass), color=TEXT, linewidth=2.5, label="s_glass")
    for (a0, b0), (a1, _) in zip(segments[:-1], segments[1:]):
        ax.axvspan(wl[b0 - 1], wl[a1], color=GRID, alpha=0.7, linewidth=0)
        ax.text(0.5 * (wl[b0 - 1] + wl[a1]), 0.98, f"no bands\n{wl[b0 - 1]:.0f}-{wl[a1]:.0f} nm",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=TEXT_2)
    ax.set_xlabel("wavelength (nm)", color=TEXT, fontsize=9)
    ax.set_ylabel("unit-normalised intensity", color=TEXT, fontsize=9)
    ax.set_title("Glass reference s_glass: band-wise median of each train capture's brightest 1%",
                 fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_2, loc="lower right")
    return _save(fig, path)


# ============================================================================
# the glass rule
# ============================================================================

def plot_glass_rule(path: Path, rule: Mapping[str, Any]) -> Path:
    per = rule.get("per_patient") or {}
    names = list(per)
    fracs = [100.0 * per[p]["glass_fraction"] for p in names]
    fig, ax = plt.subplots(figsize=(max(4.5, 1.0 + 0.7 * len(names)), 3.8), layout="constrained")
    _style(ax)
    ax.grid(False, axis="x")
    if names:
        ax.bar(range(len(names)), fracs, color=SERIES[0], width=0.6)
        for i, f in enumerate(fracs):
            ax.text(i, f, f"{f:.1f}%", ha="center", va="bottom", fontsize=7, color=TEXT_2)
        ax.set_xticks(range(len(names)), [f"patient {p}" for p in names], fontsize=7, rotation=30)
    thr = 100.0 * float(rule["threshold"])
    ax.axhline(thr, color=TEXT, linewidth=1.5, linestyle="--", label=f"pre-registered threshold {thr:.0f}%")
    overall = rule["overall"]["glass_fraction"]
    if overall is not None:
        ax.axhline(100.0 * overall, color=SERIES[1], linewidth=2, label=f"pooled {100.0 * overall:.1f}%")
    cls = "/".join(rule.get("minority_class_names") or ["minority"])
    ax.set_ylabel(f"{cls} rows the mask calls glass (%)", color=TEXT, fontsize=9)
    ax.set_title(f"Glass rule: use_copy_mask = {rule['use_copy_mask']}", fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_2)
    return _save(fig, path)


# ============================================================================
# the grid check
# ============================================================================

def plot_grid_check(path: Path, grid: Mapping[str, Any], hists: Mapping[str, np.ndarray],
                    bins: np.ndarray, rebin: int = 10) -> Path:
    """`hists` are counts over `bins`; they are summed `rebin` bins at a time
    for display (the pooled medians in grid_check.json use the fine bins)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained")
    for ax in (ax1, ax2):
        _style(ax)
    nb = (len(bins) - 1) // rebin
    coarse = bins[: nb * rebin + 1: rebin]
    centers = 0.5 * (coarse[:-1] + coarse[1:])
    width = coarse[1] - coarse[0]
    for k, (key, name) in enumerate((("h", "(r, c)/(r, c+1)"), ("v", "(r, c)/(r+1, c)"))):
        for shuffled in (False, True):
            fine = np.asarray(hists[f"{key}_shuffled" if shuffled else key], dtype=np.float64)
            counts = fine[: nb * rebin].reshape(nb, rebin).sum(axis=1)
            if counts.sum() <= 0:
                continue
            ax1.plot(centers, counts / (counts.sum() * width), color=SERIES[k], linewidth=2,
                     linestyle="--" if shuffled else "-",
                     label=f"{name} {'shuffled grid' if shuffled else 'derived grid'}")
    ax1.set_xlabel("edge correlation of facing patch edges", color=TEXT, fontsize=9)
    ax1.set_ylabel("density", color=TEXT, fontsize=9)
    ax1.set_title("All tested pairs, pooled", fontsize=10, color=TEXT)
    ax1.legend(fontsize=7, frameon=False, labelcolor=TEXT_2)

    per = grid.get("per_capture") or []
    dmin = float(grid["min_delta"])
    dh = np.array([np.nan if r.get("delta_h") is None else r["delta_h"] for r in per], dtype=np.float64)
    dv = np.array([np.nan if r.get("delta_v") is None else r["delta_v"] for r in per], dtype=np.float64) \
        if grid.get("vertical") else np.zeros(len(per))
    ok = np.array([bool(r["passed"]) for r in per], dtype=bool)
    if len(per):
        ax2.scatter(dh[ok], dv[ok], s=18, color=SERIES[0], edgecolors=SURFACE, linewidths=0.8,
                    label=f"passed ({int(ok.sum())})")
        if (~ok).any():
            ax2.scatter(dh[~ok], dv[~ok], s=22, color=SERIES[1], marker="x", label=f"failed ({int((~ok).sum())})")
    ax2.axvline(dmin, color=TEXT_2, linewidth=1, linestyle="--")
    if grid.get("vertical"):
        ax2.axhline(dmin, color=TEXT_2, linewidth=1, linestyle="--", label=f"min delta {dmin:g}")
    ax2.set_xlabel("delta, horizontal pairs (median true - median shuffled)", color=TEXT, fontsize=9)
    ax2.set_ylabel("delta, vertical pairs" if grid.get("vertical") else "(vertical pairs not tested)",
                   color=TEXT, fontsize=9)
    frac = grid.get("pass_fraction")
    ax2.set_title(f"Per capture: {grid['n_captures_passed']}/{grid['n_captures_tested']} pass"
                  + (f" ({frac:.1%}; needed {grid['min_pass_fraction']:.0%})" if frac is not None else ""),
                  fontsize=10, color=TEXT)
    ax2.legend(fontsize=7, frameon=False, labelcolor=TEXT_2)
    return _save(fig, path)
