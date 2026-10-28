# -*- coding: utf-8 -*-
"""
hsi_augment/export_figures.py
=============================
The visual QC of an export (Plan §Phase 8b "Visual QC", §Phase 9 `figures/`),
written by the export stage into `export/<source>/figures/` and nowhere else:

    spectra_<class>.png        class-mean +- sd of the per-patch mean spectra, real
                               train tissue vs the exported synthetic rows; lines per
                               spectral segment, the gap shaded
    contact_sheet_<class>.png  false-colour composites (bands nearest 640/550/460 nm)
                               of real reference patches and synthetic ones, ALL
                               stretched with the real reference's range
    nn_gallery_<class>.png     per synthetic sample: it | its nearest real train patch
                               (R6's search) | mean |difference| over bands

Why one stretch: stretching every panel to its own percentiles makes a
washed-out or saturated sample look like a real one, which is exactly what the
sheet is meant to show. Why per-segment lines: trainsel has no band between
628.9 and 850.9 nm, and a line across the gap would draw values that were
never measured; the segments come from `BuildAdapter.spectral_segments()`.

Kept apart from `export.py` so nothing that only reads an export imports
matplotlib. Nothing here imports `scripts/make_figures_v19.py`, which loads
v8 data at import time.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hsi_augment.tissue_figures import DPI, GRID, MUTED, SERIES, SURFACE, TEXT, TEXT_2  # noqa: E402

STRETCH_PERCENTILES = (1.0, 99.0)


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


def _suptitle(fig, text: str, width_in: float) -> None:
    """A suptitle wrapped to the figure's width (fontsize 8 fits ~15 characters per inch)."""
    width = max(30, int(15 * width_in))
    lines = [textwrap.fill(line, width) for line in str(text).split("\n")]
    fig.suptitle("\n".join(lines), fontsize=8, color=TEXT)


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)


# ============================================================================
# false colour
# ============================================================================

def rgb_stretch(reference: np.ndarray, bands: Sequence[int],
                percentiles: Tuple[float, float] = STRETCH_PERCENTILES) -> Tuple[float, float]:
    """(lo, hi) of the REAL reference patches' three RGB bands, pooled."""
    sel = np.asarray(reference, dtype=np.float64)[..., list(bands)]
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(sel, percentiles)
    if not hi > lo:
        lo, hi = float(sel.min()), float(sel.max())
    return float(lo), float(hi if hi > lo else lo + 1.0)


def to_rgb(x_nhwc: np.ndarray, bands: Sequence[int], lo: float, hi: float) -> np.ndarray:
    """(n, H, W, 3) in [0, 1] from NHWC patches, with a fixed (lo, hi)."""
    sel = np.asarray(x_nhwc, dtype=np.float64)[..., list(bands)]
    return np.clip((sel - lo) / (hi - lo), 0.0, 1.0)


# ============================================================================
# the figures
# ============================================================================

def plot_class_spectra(path: Path, wavelengths: np.ndarray, segments: Sequence[Tuple[int, int]],
                       real_spectra: np.ndarray, syn_spectra: np.ndarray, class_name: str, source: str) -> Path:
    """Mean +- sd of per-patch mean spectra, real vs synthetic."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    _style(ax)
    for k, (label, spec) in enumerate(((f"real {class_name} (train tissue, n={len(real_spectra)})", real_spectra),
                                       (f"{source} synthetic (n={len(syn_spectra)})", syn_spectra))):
        s = np.asarray(spec, dtype=np.float64)
        if s.size == 0:
            continue
        mu, sd = s.mean(axis=0), s.std(axis=0)
        color = SERIES[k % len(SERIES)]
        for j, (a, b) in enumerate(segments):
            ax.fill_between(wl[a:b], (mu - sd)[a:b], (mu + sd)[a:b], color=color, alpha=0.18, linewidth=0)
            ax.plot(wl[a:b], mu[a:b], color=color, linewidth=2, label=label if j == 0 else None)
    for (a0, b0), (a1, _) in zip(segments[:-1], segments[1:]):
        ax.axvspan(wl[b0 - 1], wl[a1], color=GRID, alpha=0.7, linewidth=0)
        ax.text(0.5 * (wl[b0 - 1] + wl[a1]), 0.98, f"no bands\n{wl[b0 - 1]:.0f}-{wl[a1]:.0f} nm",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=TEXT_2)
    ax.set_xlabel("wavelength (nm)", color=TEXT, fontsize=9)
    ax.set_ylabel("patch mean (build value scale)", color=TEXT, fontsize=9)
    ax.set_title(f"{class_name}: class mean +- sd of patch mean spectra, real vs {source}", fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_2, loc="lower right")
    return _save(fig, path)


def plot_contact_sheet(path: Path, real: np.ndarray, syn: np.ndarray, bands: Sequence[int],
                       stretch: Tuple[float, float], rgb_note: str, title: str, cols: int = 8) -> Path:
    """Real reference patches on top, synthetic ones below, one stretch."""
    lo, hi = stretch
    real_rgb, syn_rgb = to_rgb(real, bands, lo, hi), to_rgb(syn, bands, lo, hi)
    cols = max(1, min(int(cols), max(len(real_rgb), len(syn_rgb), 1)))
    r_rows = int(np.ceil(len(real_rgb) / cols)) if len(real_rgb) else 0
    s_rows = int(np.ceil(len(syn_rgb) / cols)) if len(syn_rgb) else 0
    rows = max(1, r_rows + s_rows)
    fig, axes = plt.subplots(rows, cols, figsize=(1.15 * cols, 1.2 * rows + 0.6), squeeze=False,
                             layout="constrained")
    for ax in axes.ravel():
        _bare(ax)
        ax.set_visible(False)
    for i, img in enumerate(real_rgb):
        ax = axes[i // cols][i % cols]
        ax.set_visible(True)
        ax.imshow(img, interpolation="nearest")
        if i % cols == 0:
            ax.set_ylabel("real", fontsize=8, color=TEXT_2)
    for i, img in enumerate(syn_rgb):
        ax = axes[r_rows + i // cols][i % cols]
        ax.set_visible(True)
        ax.imshow(img, interpolation="nearest")
        if i % cols == 0:
            ax.set_ylabel("synthetic", fontsize=8, color=TEXT_2)
    _suptitle(fig, f"{title}\n{rgb_note}; stretch [{lo:.3g}, {hi:.3g}] from the real reference",
              fig.get_size_inches()[0])
    return _save(fig, path)


def plot_nn_gallery(path: Path, syn: np.ndarray, nearest: np.ndarray, distances: Sequence[float],
                    bands: Sequence[int], stretch: Tuple[float, float], rgb_note: str, title: str) -> Path:
    """Rows: synthetic | nearest real train patch | mean |difference| over bands."""
    lo, hi = stretch
    n = len(syn)
    fig, axes = plt.subplots(max(n, 1), 3, figsize=(5.4, 1.45 * max(n, 1) + 0.6), squeeze=False,
                             layout="constrained")
    diff = np.abs(np.asarray(syn, dtype=np.float64) - np.asarray(nearest, dtype=np.float64)).mean(axis=-1)
    vmax = float(np.nanmax(diff)) if diff.size and np.isfinite(diff).any() else 1.0
    for r in range(n):
        axes[r][0].imshow(to_rgb(syn[r:r + 1], bands, lo, hi)[0], interpolation="nearest")
        axes[r][1].imshow(to_rgb(nearest[r:r + 1], bands, lo, hi)[0], interpolation="nearest")
        im = axes[r][2].imshow(diff[r], cmap="magma", vmin=0.0, vmax=vmax if vmax > 0 else 1.0,
                               interpolation="nearest")
        d = distances[r]
        axes[r][0].set_ylabel(f"d={d:.3g}" if d is not None and np.isfinite(d) else "d=n/a", fontsize=7,
                              color=TEXT_2)
        for c in range(3):
            _bare(axes[r][c])
        if r == 0:
            for c, t in enumerate(("synthetic", "nearest real", "mean |diff|")):
                axes[r][c].set_title(t, fontsize=8, color=TEXT)
    if n:
        fig.colorbar(im, ax=axes[:, 2].tolist(), fraction=0.05, pad=0.02)
    _suptitle(fig, f"{title}\n{rgb_note}; d = R6's exact distance (band-z-scored patch)", fig.get_size_inches()[0])
    return _save(fig, path)
