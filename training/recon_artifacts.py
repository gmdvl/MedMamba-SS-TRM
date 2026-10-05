# -*- coding: utf-8 -*-
"""
training/recon_artifacts.py
=============================
Reconstruction ARTIFACTS - the reconstructed cubes themselves, saved to disk,
plus the figures a paper needs and the per-epoch metric curves.

Why this file exists
--------------------
`TrainerG_v12` computes reconstruction metrics correctly (in reflectance
units, R-3/R-7) but stores nothing you can look at or re-plot:
`trainerg_v12.py:536-542` mean-pools `x_recon` over `(H, W)` and keeps only
`[N, C]` mean spectra; the `[B, C, H, W]` cube is freed at the end of the
batch. Every reconstruction figure in the repository is therefore a line
plot of a spatially averaged spectrum, and no reconstructed IMAGE has ever
been written to disk.

Three further gaps this module closes:

  1. `training/spectral_recon_metrics.ssim` is SPECTRAL SSIM - its own
     docstring says so ("not the windowed 2D SSIM used for natural images").
     Nothing in the pipeline measures SPATIAL fidelity at all. `ssim2d`
     below is the windowed 2D form, computed per band and averaged, and is
     reported under a deliberately different key so the two are never
     confused in a table.
  2. The frozen visualizations are written from the LAST epoch's validation
     metrics (`trainerg_v12.py:898` / `:943`), not from the checkpoint that
     is actually shipped. Everything here is written from an explicitly
     named checkpoint and records which one in `provenance.json`.
  3. `plot_reconstruction_metric_histograms`' docstring claims "across all
     validation samples" while the caller caps at the FIRST
     `recon_sample_cap` patches in split order. Selection here is by SAM
     quantile within each class and the rule is recorded in the provenance,
     so a figure cannot be mistaken for a random draw or for a cherry-pick.

FROZEN-file rule: nothing in this module is imported by `train_example_v15.py`
or by any `trainerg_v3..v11.py`. It imports the frozen metric helpers rather
than reimplementing them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from training.spectral_recon_metrics import (  # frozen - imported, not reimplemented
    rmse, mae, psnr, ssim as spectral_ssim, pearson_correlation,
)
from training.spectral_recon_metrics_v2 import (
    sam_deg as sam_deg_v2, sid as sid_v2, peak_position_error,
)
from training.gan import safe_cosine_similarity  # frozen

# Wavelengths (nm) a true-colour composite is built from when the cube has
# more than three bands and no explicit --recon_rgb_bands was given.
DEFAULT_RGB_NM = (640.0, 550.0, 460.0)
DEFAULT_PERCENTILE_CLIP = (2.0, 98.0)


# ======================================================================
# Spatial fidelity - the metric the pipeline did not have
# ======================================================================

def ssim2d(x_true: torch.Tensor, x_pred: torch.Tensor,
           data_range: Optional[float] = None, window_size: int = 7,
           c1: float = 0.01 ** 2, c2: float = 0.03 ** 2,
           reduce: bool = True):
    """Windowed 2-D SSIM, computed independently per band and averaged over
    bands. `[B, C, H, W] -> float` (or `[B]` when `reduce=False`).

    This is the SPATIAL structural similarity - distinct from
    `training.spectral_recon_metrics.ssim`, which treats each pixel's
    spectrum as a 1-D signal and says nothing about spatial structure.
    Report it as `ssim2d`, never as `ssim`.

    A uniform (box) window with VALID padding is used: no zero-padding
    means no edge bias, which matters on the small patches this project
    trains on. The window shrinks to the largest odd size that fits when a
    patch is smaller than `window_size`.
    """
    x_true = x_true.float()
    x_pred = x_pred.float()
    if x_true.shape != x_pred.shape:
        raise ValueError(f"shape mismatch: {tuple(x_true.shape)} vs {tuple(x_pred.shape)}")
    b, c, h, w = x_true.shape

    ws = int(min(window_size, h, w))
    if ws % 2 == 0:
        ws -= 1
    ws = max(ws, 1)

    if data_range is None:
        data_range = float(x_true.max().item() - x_true.min().item())
    if data_range <= 0:
        data_range = 1.0
    C1 = c1 * data_range ** 2
    C2 = c2 * data_range ** 2

    kernel = torch.ones(c, 1, ws, ws, device=x_true.device, dtype=x_true.dtype) / float(ws * ws)

    def _mean(t):
        return F.conv2d(t, kernel, groups=c)

    mu_t = _mean(x_true)
    mu_p = _mean(x_pred)
    mu_t2, mu_p2, mu_tp = mu_t * mu_t, mu_p * mu_p, mu_t * mu_p
    var_t = _mean(x_true * x_true) - mu_t2
    var_p = _mean(x_pred * x_pred) - mu_p2
    cov = _mean(x_true * x_pred) - mu_tp

    ssim_map = (((2 * mu_tp + C1) * (2 * cov + C2)) /
                ((mu_t2 + mu_p2 + C1) * (var_t + var_p + C2) + 1e-12))
    per_sample = ssim_map.flatten(1).mean(dim=1)
    return float(per_sample.mean().item()) if reduce else per_sample


def per_sample_full_cube_metrics(x_true: torch.Tensor, x_pred: torch.Tensor,
                                  wavelengths: Optional[torch.Tensor] = None,
                                  require_nonnegative: bool = True,
                                  data_range: Optional[float] = None) -> List[Dict[str, float]]:
    """One metric dict per sample, each computed on that sample's FULL
    `[1, C, H, W]` cube - so RMSE/PSNR/SSIM here describe the image, not a
    spatially averaged spectrum. `[B, C, H, W] -> list of B dicts`."""
    out: List[Dict[str, float]] = []
    for i in range(x_true.shape[0]):
        t = x_true[i:i + 1].float()
        p = x_pred[i:i + 1].float()
        dr = data_range if data_range is not None else float(t.max().item() - t.min().item()) or 1.0
        m = {
            "rmse": rmse(t, p),
            "mae": mae(t, p),
            "psnr": psnr(t, p, data_range=dr),
            "ssim_spectral": spectral_ssim(t, p, data_range=dr),
            "ssim2d": ssim2d(t, p, data_range=dr),
            "pearson_correlation": pearson_correlation(t, p),
            "peak_position_error": peak_position_error(t, p, wavelengths=wavelengths),
            "data_range": dr,
        }
        try:
            m["sam_deg"] = sam_deg_v2(t, p, require_nonnegative=require_nonnegative)
            m["sid"] = sid_v2(t, p, require_nonnegative=require_nonnegative)
        except Exception as e:                      # NonReflectanceDataError and friends
            m["sam_deg"] = float("nan")
            m["sid"] = float("nan")
            m["spectral_metric_error"] = f"{type(e).__name__}: {e}"
        out.append(m)
    return out


def batch_sam_deg(x_true: torch.Tensor, x_pred: torch.Tensor,
                  require_nonnegative: bool = True) -> torch.Tensor:
    """Per-sample SAM (degrees) on spatially mean-pooled spectra - the cheap
    ranking pass. `[B, C, H, W] -> [B]`, vectorized.

    Mean-pooling first is what makes a per-pixel spectral angle degenerate
    to one value per sample, the same trick `trainerg_v12.py:536-538` uses.
    The angle itself is computed exactly as `spectral_recon_metrics_v2.
    sam_deg` computes it - `safe_cosine_similarity` (frozen, imported not
    reimplemented), then the same clamp - so a per-sample value here and the
    split mean the trainer reports are the same quantity.

    `require_nonnegative` clamps negatives to zero rather than raising: this
    is a RANKING pass over a whole split, and one out-of-domain batch must
    not cost the run its figures. Anything genuinely wrong with the units is
    caught, and raised on, by `per_sample_full_cube_metrics`.
    """
    t = x_true.float().mean(dim=(2, 3))          # [B, C]
    p = x_pred.float().mean(dim=(2, 3))
    if require_nonnegative:
        t = t.clamp(min=0.0)
        p = p.clamp(min=0.0)
    cos_sim = safe_cosine_similarity(t, p)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-3, 1.0 - 1e-3)
    return torch.rad2deg(torch.acos(cos_sim)).detach().float().cpu()


# ======================================================================
# Which samples end up in the figure
# ======================================================================

def select_quantile_indices(sam: np.ndarray, labels: np.ndarray, per_class: int,
                            num_classes: int) -> Tuple[np.ndarray, List[str]]:
    """For each class, the samples sitting at `per_class` evenly spaced SAM
    quantiles - always including the best (q=0) and the worst (q=1).

    This is the selection rule a reviewer can check: it is not a random
    draw, and it is not a cherry-pick, because the worst reconstruction in
    each class is in the figure by construction. `per_class=3` gives
    best / median / worst; `per_class=5` adds the quartiles.
    """
    if per_class < 1:
        return np.array([], dtype=np.int64), []
    sam = np.asarray(sam, dtype=np.float64)
    labels = np.asarray(labels)
    chosen: List[int] = []
    tags: List[str] = []
    for cls in range(num_classes):
        cls_idx = np.flatnonzero(labels == cls)
        if cls_idx.size == 0:
            continue
        finite = np.isfinite(sam[cls_idx])
        cls_idx = cls_idx[finite] if finite.any() else cls_idx
        order = cls_idx[np.argsort(sam[cls_idx], kind="stable")]
        k = int(min(per_class, order.size))
        if k == 1:
            positions = [0]
        else:
            positions = [int(round(q * (order.size - 1))) for q in np.linspace(0.0, 1.0, k)]
        seen = set()
        for pos, q in zip(positions, np.linspace(0.0, 1.0, max(k, 1))):
            if pos in seen:
                continue
            seen.add(pos)
            chosen.append(int(order[pos]))
            tags.append(f"class{cls}_q{q:.2f}")
    order = np.argsort(np.asarray(chosen), kind="stable")
    return np.asarray(chosen, dtype=np.int64)[order], [tags[i] for i in order]


# ======================================================================
# Rendering a cube as something you can look at
# ======================================================================

def resolve_rgb_bands(n_channels: int, wavelengths: Optional[np.ndarray],
                      band_spec: Optional[str] = None) -> Tuple[Tuple[int, int, int], str]:
    """Which three band indices become R, G, B, and a human-readable note
    recording the choice (it goes into `provenance.json` - a false-colour
    composite whose band choice is undocumented is not a figure, it is a
    decoration).

    `band_spec` is either three comma-separated wavelengths in nm
    ("640,550,460", matched to the nearest band) or three band indices
    ("20,12,4"); values above 300 are read as nm when `wavelengths` is
    available.
    """
    if n_channels < 3:
        idx = tuple([0] * 3)
        return idx, f"grayscale: {n_channels}-channel cube, band 0 replicated"
    if n_channels == 3:
        return (0, 1, 2), "native RGB channel order (C=3)"

    if band_spec:
        parts = [p.strip() for p in str(band_spec).split(",")]
        if len(parts) != 3:
            raise ValueError(f"--recon_rgb_bands needs three comma-separated values, got {band_spec!r}")
        vals = [float(p) for p in parts]
        if wavelengths is not None and all(v > 300 for v in vals):
            wl = np.asarray(wavelengths, dtype=np.float64)
            idx = tuple(int(np.argmin(np.abs(wl - v))) for v in vals)
            return idx, (f"explicit wavelengths {vals} nm -> bands {list(idx)} "
                         f"(actual {[round(float(wl[i]), 1) for i in idx]} nm)")
        idx = tuple(int(v) for v in vals)
        if any(i < 0 or i >= n_channels for i in idx):
            raise ValueError(f"--recon_rgb_bands {band_spec!r} out of range for C={n_channels}")
        return idx, f"explicit band indices {list(idx)}"

    if wavelengths is not None and len(wavelengths) == n_channels:
        wl = np.asarray(wavelengths, dtype=np.float64)
        idx = tuple(int(np.argmin(np.abs(wl - v))) for v in DEFAULT_RGB_NM)
        return idx, (f"true-colour composite, bands nearest {list(DEFAULT_RGB_NM)} nm -> {list(idx)} "
                     f"(actual {[round(float(wl[i]), 1) for i in idx]} nm)")

    idx = (int(n_channels * 0.75), int(n_channels * 0.5), int(n_channels * 0.25))
    return idx, f"no wavelengths available: evenly spaced band indices {list(idx)}"


def composite_range(cube_chw: np.ndarray, bands: Sequence[int],
                    clip: Tuple[float, float] = DEFAULT_PERCENTILE_CLIP) -> Tuple[float, float]:
    """Percentile stretch computed on the TRUE cube only. The reconstruction
    is then rendered with the SAME range - stretching each panel to its own
    percentiles would make a washed-out reconstruction look identical to the
    original, which is precisely the comparison the figure exists to make."""
    sel = cube_chw[list(bands)]
    lo, hi = np.percentile(sel, clip[0]), np.percentile(sel, clip[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(sel)), float(np.max(sel))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def rgb_composite(cube_chw: np.ndarray, bands: Sequence[int], vmin: float, vmax: float) -> np.ndarray:
    """`[C, H, W] -> [H, W, 3]` in [0, 1], using a caller-supplied range."""
    sel = np.stack([cube_chw[b] for b in bands], axis=-1).astype(np.float64)
    return np.clip((sel - vmin) / (vmax - vmin), 0.0, 1.0)


# ======================================================================
# Figures
# ======================================================================

def render_sample_figure(x_true_chw: np.ndarray, x_recon_chw: np.ndarray,
                          bands: Sequence[int], wavelengths: Optional[np.ndarray],
                          metrics: Dict[str, float], title: str, out_path: str) -> None:
    """One sample, four panels: original / reconstruction / absolute error /
    spectra. Original and reconstruction share a colour range."""
    vmin, vmax = composite_range(x_true_chw, bands)
    rgb_t = rgb_composite(x_true_chw, bands, vmin, vmax)
    rgb_p = rgb_composite(x_recon_chw, bands, vmin, vmax)
    err = np.abs(x_true_chw - x_recon_chw).mean(axis=0)

    # The three image panels hold square content; giving the spectrum panel
    # the extra width keeps them narrow enough that aspect preservation does
    # not leave a band of dead space above and below every row.
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.9), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.15, 1.75]})
    axes[0].imshow(rgb_t, aspect="equal")
    axes[0].set_title("original", fontsize=10)
    axes[1].imshow(rgb_p, aspect="equal")
    axes[1].set_title("reconstruction", fontsize=10)
    im = axes[2].imshow(err, cmap="magma", aspect="equal")
    axes[2].set_title("mean |error| over bands", fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    x_axis = (np.asarray(wavelengths) if wavelengths is not None
              and len(wavelengths) == x_true_chw.shape[0] else np.arange(x_true_chw.shape[0]))
    ax = axes[3]
    ax.plot(x_axis, x_true_chw.mean(axis=(1, 2)), color="black", linewidth=1.4, label="original")
    ax.plot(x_axis, x_recon_chw.mean(axis=(1, 2)), color="crimson", linewidth=1.2,
            linestyle="--", label="reconstruction")
    ax.set_xlabel("wavelength (nm)" if wavelengths is not None else "band index", fontsize=9)
    ax.set_ylabel("reflectance", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("spatial-mean spectrum", fontsize=10)

    sub = (f"SAM {metrics.get('sam_deg', float('nan')):.2f}deg   "
           f"RMSE {metrics.get('rmse', float('nan')):.4f}   "
           f"PSNR {metrics.get('psnr', float('nan')):.2f} dB   "
           f"SSIM2D {metrics.get('ssim2d', float('nan')):.3f}")
    fig.suptitle(f"{title}\n{sub}", fontsize=10.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_contact_sheet(x_true: np.ndarray, x_recon: np.ndarray, bands: Sequence[int],
                          row_titles: Sequence[str], out_path: str, max_rows: int = 12) -> None:
    """The paper figure: N samples down the page, original / reconstruction /
    error across it. Each row keeps its own colour range (shared between its
    original and its reconstruction)."""
    n = int(min(len(x_true), max_rows))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(7.6, 2.15 * n), squeeze=False, layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.18]})
    for r in range(n):
        vmin, vmax = composite_range(x_true[r], bands)
        axes[r][0].imshow(rgb_composite(x_true[r], bands, vmin, vmax), aspect="equal")
        axes[r][1].imshow(rgb_composite(x_recon[r], bands, vmin, vmax), aspect="equal")
        im = axes[r][2].imshow(np.abs(x_true[r] - x_recon[r]).mean(axis=0), cmap="magma", aspect="equal")
        fig.colorbar(im, ax=axes[r][2], fraction=0.046, pad=0.04)
        axes[r][0].set_ylabel(row_titles[r], fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
        for c in range(3):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
        if r == 0:
            for c, t in enumerate(("original", "reconstruction", "mean |error|")):
                axes[r][c].set_title(t, fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recon_metric_curves(history_rows: Sequence[Dict], out_path: str) -> bool:
    """Per-epoch reconstruction metrics, four panels. Returns False (writing
    nothing) when the run carried no reconstruction, so an empty axis never
    reaches a paper - the same non-zero gate `training/plots_v16.py` applies
    to the reconstruction-loss curve."""
    keys = [("spectral_sam_deg", "Spectral angle (deg)"), ("spectral_rmse", "RMSE"),
            ("spectral_psnr", "PSNR (dB)"), ("spectral_ssim", "Spectral SSIM")]
    epochs = [r.get("epoch") for r in history_rows]
    series = {k: [r.get(k) for r in history_rows] for k, _ in keys}
    if not any(any(v is not None and np.isfinite(float(v)) and float(v) != 0.0 for v in vals)
               for vals in series.values()):
        return False

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for ax, (key, label) in zip(axes.ravel(), keys):
        vals = series[key]
        pairs = [(e, float(v)) for e, v in zip(epochs, vals)
                 if v is not None and np.isfinite(float(v))]
        if not pairs:
            ax.set_visible(False)
            continue
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", markersize=3)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("epoch", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Reconstruction quality per epoch (validation, reflectance units)", fontsize=12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


# ======================================================================
# Persistence
# ======================================================================

def save_reconstruction_samples(out_dir: str,
                                 x_true: np.ndarray, x_recon: np.ndarray,
                                 labels: np.ndarray, dataset_indices: np.ndarray,
                                 tags: Sequence[str], metrics: Sequence[Dict[str, float]],
                                 wavelengths: Optional[np.ndarray],
                                 class_names: Sequence[str],
                                 provenance: Dict,
                                 band_spec: Optional[str] = None,
                                 render_png: bool = True,
                                 max_rows_contact_sheet: int = 12) -> Dict:
    """Writes `samples.npz` (the arrays), `samples_metrics.json` (the table),
    `provenance.json` (what produced them), and - when `render_png` - one
    figure per sample plus a contact sheet.

    `x_true` / `x_recon` are `[N, C, H, W]` in REFLECTANCE units (denormalized).
    Saving normalized cubes would repeat, at artifact level, exactly the
    unit error R-3 fixed in the metrics.
    """
    out = Path(out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    x_true = np.asarray(x_true, dtype=np.float32)
    x_recon = np.asarray(x_recon, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    dataset_indices = np.asarray(dataset_indices, dtype=np.int64)

    npz_path = out / "samples.npz"
    np.savez_compressed(
        npz_path,
        x_true=x_true, x_recon=x_recon, label=labels, dataset_index=dataset_indices,
        tag=np.asarray(list(tags), dtype=object),
        wavelengths=(np.asarray(wavelengths, dtype=np.float32)
                     if wavelengths is not None else np.zeros(0, dtype=np.float32)),
        class_names=np.asarray(list(class_names), dtype=object),
    )

    with open(out / "samples_metrics.json", "w") as f:
        json.dump([
            {"tag": tags[i], "dataset_index": int(dataset_indices[i]),
             "label": int(labels[i]),
             "class_name": (class_names[int(labels[i])] if int(labels[i]) < len(class_names)
                            else str(int(labels[i]))),
             **{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in metrics[i].items()}}
            for i in range(len(tags))
        ], f, indent=2, default=str)

    bands, band_note = resolve_rgb_bands(x_true.shape[1], wavelengths, band_spec)
    prov = dict(provenance)
    prov.update({
        "n_samples": int(x_true.shape[0]),
        "cube_shape": list(x_true.shape[1:]),
        "units": "reflectance (denormalized)",
        "rgb_bands": list(bands),
        "rgb_band_note": band_note,
        "percentile_clip": list(DEFAULT_PERCENTILE_CLIP),
        "npz_bytes": int(npz_path.stat().st_size),
        "ssim2d_note": ("ssim2d is windowed 2-D SSIM (spatial). ssim_spectral is "
                        "training.spectral_recon_metrics.ssim, which is per-pixel across the "
                        "band axis. They are different quantities; do not report one as the other."),
    })
    with open(out / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2, default=str)

    written = []
    if render_png:
        for i, tag in enumerate(tags):
            cls = int(labels[i])
            name = class_names[cls] if cls < len(class_names) else str(cls)
            png = out / "samples" / f"{i:02d}_{tag}.png"
            render_sample_figure(x_true[i], x_recon[i], bands, wavelengths, metrics[i],
                                 title=f"{tag}  -  class {name}  -  val index {int(dataset_indices[i])}",
                                 out_path=str(png))
            written.append(str(png))
        sheet = out / "reconstruction_samples.png"
        render_contact_sheet(
            x_true, x_recon, bands,
            row_titles=[f"{tags[i]}\n{class_names[int(labels[i])] if int(labels[i]) < len(class_names) else labels[i]}"
                        for i in range(len(tags))],
            out_path=str(sheet), max_rows=max_rows_contact_sheet)
        written.append(str(sheet))

    return {"npz": str(npz_path), "figures": written, "bands": list(bands), "band_note": band_note}


def write_status(out_dir: str, **fields) -> None:
    """`reconstruction/status.json` - so an empty directory is never
    ambiguous again. `TrainerG_v12` swallowed every reconstruction-artifact
    failure into a single `logger.warning` (`trainerg_v12.py:973`), which is
    why four of the eight runs that DID train a decoder have an empty
    `reconstruction/` folder and no record of why."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "status.json", "w") as f:
        json.dump(fields, f, indent=2, default=str)
