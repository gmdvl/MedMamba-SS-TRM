# -*- coding: utf-8 -*-
"""
prepare_histologyhsi_bc.py
==========================
HistologyHSI-BC-Recurrence (TCIA, DOI 10.7937/6kpy-yt49) -> flat
``<out>/{hsi,rgb}/X_{train,val,test}.npy`` via the generic ``training.prep`` core.

The single HistologyHSI prep script. It unifies the former
``prepare_histologyhsi_bc_v3.py`` .. ``prepare_histologyhsi_bc_v8.py`` and behaves
exactly like ``prepare_histologyhsi_bc_v8.py`` (same CLI, same defaults, same
outputs, same adapter ``name`` - so resuming a dataset built by v8 still passes the
compat check). The superseded versions are kept, unmodified, in ``archive/``.

It is assembled, verbatim, from the three layers v8 used to import:

  1. HSI helpers from ``prepare_histologyhsi_bc_v6.py``: ENVI discovery,
     flat-field calibration, ROI masking, band statistics/selection, capture
     discovery and labels. Only the helpers the adapters reach are carried over;
     v6's own stand-alone pipeline (manifest/shard/unify/``main``) was already
     replaced by ``training.prep`` in v7 and stays in ``archive/``.
  2. ``HistologyHsiAdapter`` from ``prepare_histologyhsi_bc_v7.py``.
  3. ``HistologyHsiAdapterV8`` and ``main`` from ``prepare_histologyhsi_bc_v8.py``.

The only mechanical changes: ``_v6.<name>`` became ``<name>`` (one module now), and
the ``--rgb_source`` rebind of ``_find_rgb_image`` is a ``global`` assignment instead
of a module-attribute assignment - the same lookup ``_rgb_stream`` performs at call
time, so the effect is identical.

Usage::

    python prepare_histologyhsi_bc.py --root /path/to/HistologyHSI --out_dir ./data_hsi \\
        --band_selection importance --num_bands 32 --num_workers 8

----------------------------------------------------------------------------
Design notes carried over from ``prepare_histologyhsi_bc_v8.py``
----------------------------------------------------------------------------
MedMamba-SS-TRM v16 plan, Stage 1.5 (A-1 at the source) - gain-corrected
reflectance HSI prep. `prepare_histologyhsi_bc_v7.py` is FROZEN (its adapter
is on the v15 import-... no, it isn't imported by v15 at all, but it is the
generation `data/hsi_v7*` was built with, and the freezing rule applies to
every earlier generation). This subclasses `HistologyHsiAdapter` and
delegates everything unchanged except the three things below.

**Design: store per-capture-gain-corrected reflectance, not raw DN.** This
closes A-1 at the source (patient 68's IDC captures acquired at ~50%
intensity - the same acquisition fault `per_patch_zscore`/gate G8 catch at
train time, fixed here before the data ever reaches a Dataset), gives R-3/
R-7 the reflectance units they need, and roughly halves storage once
`--hsi_value_scale` brings the corpus into float16's range.

  --capture_gain {none, median_ratio}  (default median_ratio)
      Pass 1 already walks the captures for band statistics; this extends it
      to sample ROI pixels per capture and record a median intensity. Per
      capture, `gain = corpus_median / capture_median`, clipped to
      `[--gain_clip_lo, --gain_clip_hi]` (default `[0.5, 2.0]`) - anything
      clipped is logged as a SUSPECT ACQUISITION rather than silently
      stretched arbitrarily. Applied in `_hsi_stream`, between the flat-field
      calibration call and the band slice.

  --hsi_value_scale <float|auto>
      Divide by a recorded constant so cubes land in reflectance-like
      `[0, ~1.5]`. `auto` (default) picks the 99th percentile of the
      GAIN-CORRECTED pass-1 sample so `float16` becomes safe
      (`_FLOAT16_SAFE_MAX = 60000.0`, from `prepare_histologyhsi_bc_v7.py`).
      Recorded in `dataset_manifest.json` via `provenance()`.

  --emit_group_sidecars  (default on)
      Writes `groups_*.npy` / `captures_*.npy` / `capture_index.json`
      directly during `finalize_modality`, so `scripts/derive_patch_groups.py`
      (Stage 0.3) becomes a fallback rather than the only source - this
      adapter already has patient/tissue/capture per item, no key-regex
      reconstruction needed.

Two bugs fixed while in here (both A-1-adjacent, both in the frozen v6/v7
calibration probe, NOT edited there - re-derived correctly here instead):

  - `hsi_all_calibrated` was inverted: `get_calibration_means` returns
    `(None, None)` precisely BECAUSE the cube has no reference captures to
    calibrate against (frequently because it is raw sensor DN, not because
    it is already a bounded reflectance), and the v6/v7 probe read that as
    "calibrated". This adapter instead ties `hsi_all_calibrated` (and hence
    the dtype decision) to whether the ACTUAL stored values (post gain
    correction and `--hsi_value_scale`) fit under `_FLOAT16_SAFE_MAX` -
    which is what determines float16 safety, not the probe's boolean.
  - Per-capture stats (`gain`, `median_intensity`, `n_sampled`) are written
    to `capture_gain_report.json` next to the arrays via
    `write_pass1_outputs` (the same adapter hook `selected_band_indices.npy`
    already uses), NOT `ItemResult.extra` - `training/prep/parallel.py`
    (frozen) hardcodes `ItemResult(..., extra={})` with no adapter hook to
    populate it, so that field stays a documented but presently-inert part
    of the schema; this hook already exists precisely to write per-dataset
    (not per-item) artifacts, and per-capture stats are exactly that.

----------------------------------------------------------------------------
Design notes carried over from ``prepare_histologyhsi_bc_v7.py``
----------------------------------------------------------------------------
HistologyHSI-BC-Recurrence (TCIA, DOI 10.7937/6kpy-yt49) -> flat
``<out>/{hsi,rgb}/X_{train,val,test}.npy`` via the generic
``training.prep`` core.

``prepare_histologyhsi_bc_v6.py`` stays FROZEN for reproducing existing
datasets. v7 = a thin :class:`training.prep.DatasetAdapter` that DELEGATES
the HSI-specific work (ENVI discovery, flat-field calibration, ROI
masking, spectral band selection) to v6's helper functions, so there is
no logic fork - only the pipeline shell changes:

  * multi-process per-capture extraction (``--num_workers``), shards that
    span captures, batched manifest checkpoints, cheaper finalize.
  * ``--store_dtype`` (default: float16 for HSI when the sampled cubes are
    calibrated and in range, else float32; float32 for RGB), halving the
    on-disk size of a calibrated HSI dataset.
  * flexible ``--split`` (``75/15/10`` too), grouped ``--kfold``,
    reusable ``--split_file``; ``class_names.json`` is written.
"""

from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from training.prep import DatasetAdapter, ModalitySpec, PrepItem, run_prep
from training.prep_progress import install as install_prep_progress
from training.npy_atomic import save_npy_atomic, save_json_atomic
from training.progress import ProgressReporter


# ############################################################################
# Part 1 - HSI helpers (from prepare_histologyhsi_bc_v6.py)
# ############################################################################

CAPTURE_RE = re.compile(r"HSI?_VNIR_(?P<patient>[^_]+)_(?P<tissue>IDC|Healthy|DCIS)_x10_C(?P<capture>\d+)",
                         re.IGNORECASE)
TISSUE_TO_ID = {"Healthy": 0, "DCIS": 1, "IDC": 2}
_TISSUE_CANON = {"idc": "IDC", "healthy": "Healthy", "dcis": "DCIS"}


def canonicalize_tissue(name: str) -> str:
    """`CAPTURE_RE` matches IDC/Healthy/DCIS case-INSENSITIVELY, so a folder
    literally named ``..._healthy_...`` / ``..._idc_...`` yields a lowercase
    ``tissue`` that then KeyErrors on ``TISSUE_TO_ID``. Normalize it here."""
    key = str(name).strip().lower()
    if key not in _TISSUE_CANON:
        raise KeyError(f"unrecognized tissue type {name!r} (expected one of IDC/Healthy/DCIS)")
    return _TISSUE_CANON[key]


# ============================================================================
# File discovery inside a capture folder (adjust these patterns if your
# actual download uses different filenames - see module docstring)
# ============================================================================

def _find_envi_cube(capture_dir: str):
    """Returns the path to the .hdr file to load, preferring a file that
    looks pre-calibrated over raw/white/dark reference captures."""
    hdrs = [f for f in os.listdir(capture_dir) if f.lower().endswith(".hdr")]
    if not hdrs:
        return None

    def score(name):
        n = name.lower()
        if any(k in n for k in ("calib", "reflectance")):
            return 0
        if any(k in n for k in ("white", "whiteref", "wr_", "_wr")):
            return 2
        if any(k in n for k in ("dark", "darkref", "dr_", "_dr")):
            return 2
        if "raw" in n:
            return 1
        return 1

    hdrs.sort(key=score)
    return os.path.join(capture_dir, hdrs[0])


def _find_reference_cubes(capture_dir: str):
    """Best-effort discovery of white/dark reference .hdr files, for the
    fallback raw->reflectance calibration path."""
    hdrs = [f for f in os.listdir(capture_dir) if f.lower().endswith(".hdr")]
    white = next((f for f in hdrs if "white" in f.lower() or re.search(r"(^|_)wr(_|\.)", f.lower())), None)
    dark = next((f for f in hdrs if "dark" in f.lower() or re.search(r"(^|_)dr(_|\.)", f.lower())), None)
    return (os.path.join(capture_dir, white) if white else None,
            os.path.join(capture_dir, dark) if dark else None)


def _find_rgb_image(capture_dir: str):
    """The synthetic RGB rendering that ships alongside each HSI capture."""
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    candidates = [f for f in os.listdir(capture_dir) if f.lower().endswith(exts)]
    if not candidates:
        return None
    candidates.sort(key=lambda n: 0 if "rgb" in n.lower() or "synt" in n.lower() else 1)
    return os.path.join(capture_dir, candidates[0])


def open_envi_lazy(hdr_path: str):
    """Opens an ENVI cube WITHOUT loading it into memory (`spectral` memory-
    maps the .dat/.raw file); returns (img, wavelengths). Slice `img[y0:y1,
    x0:x1, :]` to pull just that block off disk - never call `img.load()`
    on a whole-slide cube."""
    try:
        import spectral
    except ImportError as e:
        raise ImportError(
            "reading ENVI (.hdr/.dat) hyperspectral cubes requires the `spectral` "
            "package: pip install spectral --break-system-packages") from e

    img = spectral.io.envi.open(hdr_path)
    wavelengths = None
    md = getattr(img, "bands", None)
    if md is not None and getattr(md, "centers", None):
        wavelengths = np.asarray(md.centers, dtype=np.float32)
    return img, wavelengths


def load_reference_mean(hdr_path: str):
    """White/dark reference captures are small (a quick flat-field shot),
    so it's fine to load these fully and reduce to a per-band mean vector -
    only the whole-slide capture itself needs the lazy path above."""
    img, _ = open_envi_lazy(hdr_path)
    arr = np.asarray(img.load(), dtype=np.float32)  # [h,w,C], small
    return arr.mean(axis=(0, 1), keepdims=True)      # [1,1,C]


def calibrate_patch(patch, white_mean, dark_mean):
    """Standard flat-field calibration: R = (raw - dark) / (white - dark)."""
    denom = np.clip(white_mean - dark_mean, 1e-6, None)
    refl = (patch - dark_mean) / denom
    return np.clip(refl, 0.0, 1.5).astype(np.float32)


def get_calibration_means(cap_dir: str, cube_path: str):
    """Shared by both the main extraction pass and the band-selection Pass 1
    sampler, so band statistics are computed on the SAME (raw vs calibrated)
    representation that ends up on disk. Returns (white_mean, dark_mean) or
    (None, None) if the cube already looks pre-calibrated or no reference
    captures could be found."""
    if "calib" in os.path.basename(cube_path).lower() or "reflect" in cube_path.lower():
        return None, None
    wp, dp = _find_reference_cubes(cap_dir)
    if wp and dp:
        return load_reference_mean(wp), load_reference_mean(dp)
    return None, None


# ============================================================================
# Band Selection Utilities (Phase 1) - isolated from patch-extraction code so
# that leaving --band_selection at its default ("none") touches nothing else.
# ============================================================================

def compute_band_statistics(captures, roi_root, labels, sample_fraction, seed,
                             max_pixels_per_capture=2000):
    """Pass 1: opens a SUBSET of cubes (sample_fraction of all captures,
    at least one), draws up to `max_pixels_per_capture` random pixel spectra
    from each (calibrated the same way Pass 2 will calibrate patches), and
    returns (pixels [N,C] float32, pixel_labels [N] int64, wavelengths).
    Bounded memory: never opens more than one whole-slide cube at a time,
    and only ever reads single-pixel spectra off the memmap (not patches)."""
    rng = np.random.default_rng(seed)
    hsi_captures = [c for c in captures if c["key"] in labels]
    if not hsi_captures:
        raise RuntimeError("Band selection: no labeled captures available to sample from.")

    n_sample = max(1, int(round(len(hsi_captures) * sample_fraction)))
    sample_idx = rng.choice(len(hsi_captures), size=min(n_sample, len(hsi_captures)), replace=False)
    sampled = [hsi_captures[i] for i in sorted(sample_idx)]

    pixel_chunks, label_chunks = [], []
    wavelengths = None
    n_bands_ref = None
    n_opened = 0
    for cap in sampled:
        cube_path = _find_envi_cube(cap["dir"])
        if cube_path is None:
            continue
        try:
            cube_img, wl = open_envi_lazy(cube_path)
        except Exception as e:
            print(f"    [band-selection warn] failed to open {cap['key']}: {e}")
            continue

        # F11 - band selection is computed on one spectrum and then applied BY
        # INDEX to every cube in Pass 2, so all cubes must share a band count.
        if n_bands_ref is None:
            n_bands_ref = int(cube_img.shape[2])
        elif int(cube_img.shape[2]) != n_bands_ref:
            raise RuntimeError(
                f"Band selection: cube {cap['key']} has {cube_img.shape[2]} bands but an "
                f"earlier cube had {n_bands_ref}. Cubes in this collection do not share a "
                f"spectrum - band selection by index would corrupt the data. Split the "
                f"dataset by sensor, or run with --band_selection none.")
        if wavelengths is None and wl is not None:
            wavelengths = wl

        H, W, _C = cube_img.shape
        white_mean, dark_mean = get_calibration_means(cap["dir"], cube_path)

        n_px = int(min(max_pixels_per_capture, H * W))
        ys = rng.integers(0, H, size=n_px)
        xs = rng.integers(0, W, size=n_px)
        spectra = np.empty((n_px, cube_img.shape[2]), dtype=np.float32)
        for i, (y, x) in enumerate(zip(ys, xs)):
            px = np.asarray(cube_img[int(y), int(x), :], dtype=np.float32).reshape(1, 1, -1)
            if white_mean is not None and dark_mean is not None:
                px = calibrate_patch(px, white_mean, dark_mean)
            spectra[i] = px.reshape(-1)

        pixel_chunks.append(spectra)
        label_chunks.append(np.full(n_px, labels[cap["key"]], dtype=np.int64))
        n_opened += 1
        del cube_img
        gc.collect()

    if not pixel_chunks:
        raise RuntimeError("Band selection: couldn't open any cube in the sampled subset - "
                            "try increasing --band_selection_sample_fraction.")

    print(f"  [band-selection] sampled {sum(c.shape[0] for c in pixel_chunks)} pixels "
          f"from {n_opened}/{len(hsi_captures)} cubes (sample_fraction={sample_fraction})")
    pixels = np.concatenate(pixel_chunks, axis=0)
    pixel_labels = np.concatenate(label_chunks, axis=0)
    return pixels, pixel_labels, wavelengths


def select_bands_variance(pixels, num_bands):
    """Keep the `num_bands` wavelengths with the highest variance across the
    sampled pixels - removes near-constant / uninformative bands."""
    num_bands = min(num_bands, pixels.shape[1])
    variances = pixels.var(axis=0)
    idx = np.argsort(-variances)[:num_bands]
    return np.sort(idx)


def select_bands_correlation(pixels, threshold=0.98, num_bands=None):
    """Greedy correlation pruning: visit bands in descending-variance order,
    keep a band only if it isn't highly correlated (> threshold) with any
    band already kept. This removes redundant near-duplicate neighbouring
    wavelengths rather than targeting an exact count; if `num_bands` is
    given and MORE bands survive pruning than that, the highest-variance
    survivors are kept. If fewer survive, all survivors are kept (the
    threshold, not num_bands, is the primary control for this method)."""
    C = pixels.shape[1]
    variances = pixels.var(axis=0)
    order = np.argsort(-variances)

    # guard against degenerate (zero-variance / constant) columns blowing up corrcoef
    corr = np.corrcoef(pixels.T)
    corr = np.nan_to_num(corr, nan=0.0)

    kept = []
    for b in order:
        if all(abs(corr[b, k]) <= threshold for k in kept):
            kept.append(int(b))
    kept = np.array(sorted(kept))

    if num_bands is not None and len(kept) > num_bands:
        kept_variances = variances[kept]
        top = kept[np.argsort(-kept_variances)[:num_bands]]
        kept = np.sort(top)
    if len(kept) == 0:
        kept = np.array([int(order[0])])
    return kept


def select_bands_mutual_information(pixels, pixel_labels, num_bands, seed=0):
    """Supervised: ranks bands by mutual information with the (per-capture,
    weakly-supervised) label and keeps the top `num_bands`."""
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError as e:
        raise ImportError(
            "--band_selection mutual_information (and hybrid/topk) require scikit-learn: "
            "pip install scikit-learn --break-system-packages") from e
    num_bands = min(num_bands, pixels.shape[1])
    mi = mutual_info_classif(pixels, pixel_labels, discrete_features=False, random_state=seed)
    idx = np.argsort(-mi)[:num_bands]
    return np.sort(idx)


def select_bands_hybrid(pixels, pixel_labels, num_bands, corr_threshold=0.98, seed=0):
    """Phase 1.4E - variance filter -> correlation pruning -> MI ranking.
    Also used for the CLI's `topk` choice (same strong hybrid recipe)."""
    C = pixels.shape[1]
    pool_size = min(C, max(num_bands * 4, num_bands + 16))
    pool = select_bands_variance(pixels, pool_size)

    pruned = select_bands_correlation(pixels[:, pool], threshold=corr_threshold, num_bands=None)
    pruned_global = pool[pruned]

    if len(pruned_global) <= num_bands:
        return np.sort(pruned_global)

    mi_local = select_bands_mutual_information(pixels[:, pruned_global], pixel_labels, num_bands, seed=seed)
    return np.sort(pruned_global[mi_local])


def select_bands(method, pixels, pixel_labels, num_bands, corr_threshold, seed):
    if method == "uniform":
        return select_bands_uniform(pixels.shape[1], num_bands)
    if method == "variance":
        return select_bands_variance(pixels, num_bands)
    if method == "correlation":
        return select_bands_correlation(pixels, threshold=corr_threshold, num_bands=num_bands)
    if method == "mutual_information":
        return select_bands_mutual_information(pixels, pixel_labels, num_bands, seed=seed)
    if method in ("hybrid", "topk"):
        return select_bands_hybrid(pixels, pixel_labels, num_bands, corr_threshold=corr_threshold, seed=seed)
    raise ValueError(f"Unknown band selection method: {method}")


# ----------------------------------------------------------------------------
# --num_bands / --importance_threshold (Phase 1, improve-prompt_v5.plan.md)
# ----------------------------------------------------------------------------
# Unlike the fixed-count methods above, `--band_selection importance` derives a
# single normalized [0, 1] "importance score" per wavelength and lets the user
# cap the result with --num_bands, floor it with --importance_threshold, both,
# or neither. Neither argument is mandatory (see module docstring / plan Phase 1).

def compute_band_importance(pixels, pixel_labels, seed=0):
    """Returns a per-band importance score normalized to [0, 1].

    The raw score is the per-band variance (always available, unsupervised)
    blended 50/50 with per-band mutual information against `pixel_labels`
    when scikit-learn is available and the labels carry more than one class
    (skipped otherwise, in which case the score falls back to variance
    alone). Blending keeps a single scalar "importance" per band while still
    rewarding wavelengths that are both spectrally informative and
    label-discriminative.
    """
    variance = pixels.var(axis=0).astype(np.float64)
    v_max = variance.max()
    norm_variance = variance / v_max if v_max > 0 else np.zeros_like(variance)

    norm_mi = None
    if pixel_labels is not None and len(np.unique(pixel_labels)) > 1:
        try:
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(pixels, pixel_labels, discrete_features=False,
                                      random_state=seed).astype(np.float64)
            mi_max = mi.max()
            norm_mi = mi / mi_max if mi_max > 0 else np.zeros_like(mi)
        except ImportError:
            norm_mi = None

    if norm_mi is not None:
        importance = 0.5 * norm_variance + 0.5 * norm_mi
    else:
        importance = norm_variance

    # re-normalize the blended score itself to [0, 1] so the threshold's
    # meaning ("keep wavelengths at or above X% of the most-important band")
    # stays consistent regardless of whether MI contributed.
    i_max = importance.max()
    if i_max > 0:
        importance = importance / i_max
    return importance.astype(np.float32)


def select_bands_uniform(n_total, num_bands):
    """v15 R2.2b - evenly spaced band indices across the FULL sensor range.

    Cheap, deterministic, needs no Pass-1 scan, and is a strictly better
    baseline than what `importance` produced on this dataset: its top-k walked
    off the front of the importance array and returned bands [0..31], i.e.
    400.5-423.0 nm out of 400.5-938.2 nm - 4.3% of the sensor, missing the
    entire haemoglobin absorption region (~540-580 nm) and all NIR scatter.
    """
    num_bands = int(min(max(1, num_bands), n_total))
    if num_bands == 1:
        return np.array([n_total // 2], dtype=np.int64)
    idx = np.linspace(0, n_total - 1, num=num_bands)
    return np.unique(np.round(idx).astype(np.int64))


def _decorrelate_ranking(ranking, pixels=None, max_corr=None, min_gap=0, target=None):
    """v15 R2.2a - walk a descending-importance `ranking` and keep a candidate
    only if it is not near-duplicate of something already kept.

    A band is rejected when its |Pearson correlation| with any kept band
    exceeds `max_corr`, or when its index is within `min_gap` of a kept one.
    `compute_band_importance` blends normalized variance with mutual
    information, and on this sensor that score falls almost monotonically with
    band index - so a plain top-k selects a contiguous, near-perfectly
    correlated run off the front of the array and calls it "the 32 most
    important bands".

    Returns `(kept, rejected)` in rank order. If fewer than `target` bands
    survive, the best rejected ones are appended back (the caller warns), so
    this can never return fewer bands than the plain top-k would have.
    """
    if (max_corr is None or pixels is None) and not min_gap:
        return list(int(b) for b in ranking), []

    corr = None
    if max_corr is not None and pixels is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(np.corrcoef(np.asarray(pixels, dtype=np.float64).T), nan=0.0)

    kept, rejected = [], []
    for cand in (int(b) for b in ranking):
        if min_gap and any(abs(cand - k) < min_gap for k in kept):
            rejected.append(cand)
            continue
        if corr is not None and any(abs(corr[cand, k]) > max_corr for k in kept):
            rejected.append(cand)
            continue
        kept.append(cand)
        if target is not None and len(kept) >= target:
            break
    return kept, rejected


def select_bands_by_importance(importance, num_bands, importance_threshold, max_bands=None,
                                pixels=None, max_corr=None, min_gap=0):
    """Implements the Main Development Plan's Phase 1 selection algorithm:

      1. Rank all wavelengths by normalized importance (descending).
      2. If --importance_threshold is given, drop everything below it -> the
         "informative" pool. Otherwise every band is in the pool.
      3/4. If --num_bands is also given: return the top --num_bands of the
         pool by rank, UNLESS the pool is already <= --num_bands, in which
         case return the whole pool as-is (never invents extra wavelengths -
         "Option A").
      5. If no threshold was given, --num_bands (if any) simply takes the top
         N bands overall.
      6. If neither is given, every band is kept (selection is a no-op).
      7. --max_bands (if given) is then applied as a hard, never-exceeded
         cap on top of whatever steps 1-6 produced: if the current selection
         is already <= max_bands, it is left untouched; otherwise it is
         trimmed down to the top `max_bands` bands by importance rank. This
         is a pure upper bound - it never *adds* bands and never overrides
         --importance_threshold's floor.

    v15 R2.2a - `pixels`/`max_corr`/`min_gap` add a decorrelation pass between
    steps 1 and 2: the ranking is walked in order and a candidate is skipped
    when it duplicates an already-selected band (see `_decorrelate_ranking`).
    All three default to off, so an existing call reproduces the old selection
    exactly.

    Returns (selected_indices [sorted ascending], ranking [band indices,
    best-to-worst], informative_count).
    """
    n_total = len(importance)
    ranking = np.argsort(-importance)  # band indices, most important first

    if importance_threshold is not None:
        informative_mask = importance >= importance_threshold
        pool = ranking[informative_mask[ranking]]
    else:
        pool = ranking

    n_informative = len(pool)

    if max_corr is not None or min_gap:
        kept, rejected = _decorrelate_ranking(pool, pixels=pixels, max_corr=max_corr,
                                               min_gap=min_gap, target=num_bands)
        if num_bands is not None and len(kept) < min(num_bands, n_informative):
            shortfall = min(num_bands, n_informative) - len(kept)
            print(f"  [band-selection] decorrelation (max_corr={max_corr}, min_gap={min_gap}) left "
                  f"only {len(kept)} of the requested {num_bands} bands; falling back to the plain "
                  f"top-k for the remaining {shortfall}. Loosen --band_max_corr or lower "
                  f"--num_bands if the selection matters.")
            kept = kept + rejected[:shortfall]
        else:
            print(f"  [band-selection] decorrelation kept {len(kept)} of {n_informative} ranked "
                  f"bands (max_corr={max_corr}, min_gap={min_gap})")
        pool = np.array(kept, dtype=np.int64)

    if num_bands is not None:
        if n_informative <= num_bands:
            selected = pool
            if importance_threshold is not None:
                print(f"  [band-selection] Requested {num_bands} bands.")
                print(f"  [band-selection] Only {n_informative} informative band(s) "
                      f"exceeded the threshold ({importance_threshold}).")
                print(f"  [band-selection] Using all informative bands.")
        else:
            selected = pool[:num_bands]
    else:
        selected = pool  # keep all informative bands (or all bands if no threshold either)

    if max_bands is not None and len(selected) > max_bands:
        print(f"  [band-selection] Selection has {len(selected)} bands, exceeding "
              f"--max_bands {max_bands}. Capping to the top {max_bands} by importance.")
        # `selected` is a subset of `ranking`, but not necessarily still in rank
        # order (e.g. after np.sort below on a previous call) - rebuild rank
        # order for this subset before truncating, so we keep the *best*
        # max_bands bands, not an arbitrary max_bands of them.
        selected_set = set(int(i) for i in selected)
        rank_ordered = [i for i in ranking if int(i) in selected_set]
        selected = np.array(rank_ordered[:max_bands], dtype=np.int64)

    selected = np.sort(selected.astype(np.int64))
    return selected, ranking.astype(np.int64), n_informative


class BandCoverageError(RuntimeError):
    """v15 R2.2c - the selected bands cover too little of the sensor."""


def band_coverage_report(indices, wavelengths, original_bands=None) -> dict:
    """v15 R2.2d - what fraction of the SENSOR the selection actually spans,
    in nm when wavelengths are known and in band indices otherwise.

    `data/hsi_v7`'s `selected_band_indices.npy` is exactly [0..31]: 400.5-423.0
    nm out of 400.5-938.2 nm, `coverage_frac = 0.042`. This is the number that
    would have caught it at prep time instead of two datasets later.
    """
    idx = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
    report = {
        "selected_bands": int(len(idx)),
        "original_bands": int(original_bands) if original_bands is not None else None,
        "index_min": int(idx.min()) if idx.size else None,
        "index_max": int(idx.max()) if idx.size else None,
        "wavelength_min_nm": None, "wavelength_max_nm": None, "wavelength_span_nm": None,
        "sensor_min_nm": None, "sensor_max_nm": None, "sensor_span_nm": None,
        "coverage_frac": None, "coverage_basis": None,
    }
    if wavelengths is not None and len(wavelengths) and idx.size:
        wl = np.asarray(wavelengths, dtype=np.float64)
        sel = wl[idx]
        sensor_span = float(wl.max() - wl.min())
        report.update(
            wavelength_min_nm=float(sel.min()), wavelength_max_nm=float(sel.max()),
            wavelength_span_nm=float(sel.max() - sel.min()),
            sensor_min_nm=float(wl.min()), sensor_max_nm=float(wl.max()),
            sensor_span_nm=sensor_span, coverage_basis="wavelength_nm",
            coverage_frac=(float((sel.max() - sel.min()) / sensor_span) if sensor_span > 0 else 1.0),
        )
    elif original_bands and idx.size:
        span = float(idx.max() - idx.min())
        denom = float(int(original_bands) - 1)
        report.update(coverage_basis="band_index",
                       coverage_frac=(span / denom if denom > 0 else 1.0))
    return report


def enforce_band_coverage(coverage: dict, min_coverage=None, allow_narrow: bool = False,
                           out_dir=None) -> None:
    """v15 R2.2c - fail the prep when the selection spans less than
    `min_coverage` of the sensor. Writes the coverage report next to the other
    band artifacts first, so the failure is diagnosable without re-running the
    Pass-1 scan."""
    if out_dir is not None:
        try:
            save_json_atomic(os.path.join(out_dir, "band_coverage_report.json"), coverage)
        except OSError:
            pass
    frac = coverage.get("coverage_frac")
    if min_coverage is None or frac is None or frac >= float(min_coverage):
        return
    detail = (f"{coverage['wavelength_min_nm']:.1f}-{coverage['wavelength_max_nm']:.1f} nm out of "
              f"{coverage['sensor_min_nm']:.1f}-{coverage['sensor_max_nm']:.1f} nm"
              if coverage.get("coverage_basis") == "wavelength_nm"
              else f"band indices {coverage['index_min']}-{coverage['index_max']} "
                   f"of {coverage['original_bands']}")
    msg = (f"BAND_COVERAGE_TOO_NARROW: the {coverage['selected_bands']} selected bands span "
           f"{frac:.1%} of the sensor ({detail}), below --band_min_coverage {min_coverage}. "
           f"A selection this narrow throws away most of the spectrum before the model ever sees "
           f"it. Use --band_selection uniform, raise --band_max_corr, or pass "
           f"--allow_narrow_bands to proceed deliberately.")
    if allow_narrow:
        print(f"  [band-selection] WARNING: {msg}")
        return
    raise BandCoverageError(msg)


def save_importance_selection_report(out_dir, importance, ranking, selected_indices,
                                      n_informative, num_bands_requested, importance_threshold,
                                      original_bands, wavelengths, max_bands=None,
                                      min_coverage=None, allow_narrow=False,
                                      max_corr=None, min_gap=0):
    """Writes the Phase 1 metadata bundle: selected_wavelengths.npy,
    band_importance.npy, band_ranking.npy and band_selection_report.json
    (also written as the legacy name selection_report.json, which older
    copies of train_example_v6.py look for)."""
    save_npy_atomic(os.path.join(out_dir, "band_importance.npy"), importance.astype(np.float32))
    save_npy_atomic(os.path.join(out_dir, "band_ranking.npy"), ranking.astype(np.int64))
    save_npy_atomic(os.path.join(out_dir, "selected_band_indices.npy"), selected_indices.astype(np.int64))
    if wavelengths is not None:
        save_npy_atomic(os.path.join(out_dir, "selected_wavelengths.npy"),
                        np.asarray(wavelengths, dtype=np.float32)[selected_indices])

    report = {
        "selection_method": "importance",
        "original_bands": int(original_bands),
        "bands_after_threshold": int(n_informative),
        "final_bands": int(len(selected_indices)),
        "importance_threshold": importance_threshold,
        "num_bands_requested": num_bands_requested,
        "max_bands": max_bands,
        "max_bands_cap_applied": bool(max_bands is not None and n_informative > max_bands
                                       and (num_bands_requested is None
                                            or num_bands_requested > max_bands)),
        "importance_statistics": {
            "min": float(importance.min()),
            "max": float(importance.max()),
            "mean": float(importance.mean()),
            "median": float(np.median(importance)),
            "std": float(importance.std()),
        },
        "selected_indices": selected_indices.tolist(),
        # v15 R2.2a - what the decorrelation pass was asked to do.
        "band_max_corr": max_corr,
        "band_min_gap": int(min_gap),
    }
    # v15 R2.2c/d - coverage is recorded, and enforced, before anything is
    # written that a later training run would trust.
    coverage = band_coverage_report(selected_indices, wavelengths, original_bands)
    report["coverage"] = coverage
    for fname in ("band_selection_report.json", "selection_report.json"):
        save_json_atomic(os.path.join(out_dir, fname), report)
    enforce_band_coverage(coverage, min_coverage, allow_narrow, out_dir=out_dir)
    return report


def save_selected_bands(out_dir, method, indices, wavelengths, original_bands,
                         num_bands_requested, sample_fraction,
                         min_coverage=None, allow_narrow=False):
    """Writes selected_band_indices.npy, selected_wavelengths.npy (if the
    full spectrum's wavelengths are known) and band_selection_config.json."""
    indices = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
    save_npy_atomic(os.path.join(out_dir, "selected_band_indices.npy"), indices)
    if wavelengths is not None:
        save_npy_atomic(os.path.join(out_dir, "selected_wavelengths.npy"),
                        np.asarray(wavelengths, dtype=np.float32)[indices])
    config = {
        "method": method,
        "original_bands": int(original_bands) if original_bands is not None else None,
        "selected_bands": int(len(indices)),
        "num_bands_requested": num_bands_requested,
        "sample_fraction": sample_fraction,
        "selected_indices": indices.tolist(),
    }
    coverage = band_coverage_report(indices, wavelengths, original_bands)   # v15 R2.2d
    config["coverage"] = coverage
    save_json_atomic(os.path.join(out_dir, "band_selection_config.json"), config)
    enforce_band_coverage(coverage, min_coverage, allow_narrow, out_dir=out_dir)   # v15 R2.2c
    return indices


# ============================================================================
# Optional ROI masking (01_03_HSI_ROI_Annotations/*.geojson)
# ============================================================================

def rasterize_roi_mask(geojson_path: str, height: int, width: int):
    """Rasterizes the first polygon feature in a GeoJSON file into a boolean
    [H,W] mask. Uses PIL only (no shapely dependency)."""
    from PIL import Image, ImageDraw

    with open(geojson_path) as f:
        gj = json.load(f)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    feats = gj.get("features", [gj]) if isinstance(gj, dict) else gj
    for feat in feats:
        geom = feat.get("geometry", feat)
        coords = geom.get("coordinates")
        gtype = geom.get("type", "Polygon")
        rings = coords[0] if gtype == "Polygon" else coords[0][0] if gtype == "MultiPolygon" else None
        if rings is None:
            continue
        pts = [(float(x), float(y)) for x, y in rings]
        draw.polygon(pts, outline=1, fill=1)
    return np.array(mask, dtype=bool)


def find_roi_geojson(roi_dir: str, patient: str, tissue: str, capture: str):
    if not roi_dir or not os.path.isdir(roi_dir):
        return None
    needle = f"{patient}_{tissue}_x10_C{capture}".lower()
    for f in os.listdir(roi_dir):
        if f.lower().endswith(".geojson") and needle in f.lower():
            return os.path.join(roi_dir, f)
    return None


# ============================================================================
# Clinical recurrence labels (optional --label_source recurrence)
# ============================================================================

def load_recurrence_labels(xlsx_path: str):
    """Returns {patient_id: 0/1} using whatever column looks like the
    distant-recurrence outcome. Inspect your actual sheet once and adjust
    the column-name matching below if it doesn't find the right column."""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip().lower() if h is not None else "" for h in rows[0]]

    def find_col(*keywords):
        for i, h in enumerate(header):
            if all(k in h for k in keywords):
                return i
        return None

    id_col = find_col("patient") or find_col("id")
    rec_col = find_col("recur")
    if id_col is None or rec_col is None:
        raise ValueError(
            f"Couldn't find patient-id / recurrence columns in {xlsx_path}. "
            f"Found headers: {header}. Edit load_recurrence_labels() to match your sheet.")

    labels = {}
    for row in rows[1:]:
        pid, rec = row[id_col], row[rec_col]
        if pid is None:
            continue
        rec_str = str(rec).strip().lower()
        labels[str(pid).strip()] = 1 if rec_str in ("1", "yes", "y", "true", "recurrence", "recurrent") else 0
    return labels


# ============================================================================
# Patch coordinates
# ============================================================================

def patch_coords(H, W, patch_size, stride, mask=None, roi_min_frac=0.8):
    """Returns the list of (y, x) top-left corners on the stride grid that
    (if `mask` given) are at least `roi_min_frac` inside the annotated ROI.
    Computing coordinates once and reusing them for every modality is what
    keeps HSI patch i and RGB patch i pointing at the same tissue location."""
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            if mask is not None:
                m = mask[y:y + patch_size, x:x + patch_size]
                if m.mean() < roi_min_frac:
                    continue
            coords.append((y, x))
    return coords


def resize_nn(rgb, target_h, target_w):
    """Nearest-neighbour resize so RGB patch coordinates line up with the
    (usually lower-resolution) HSI cube grid, without extra dependencies."""
    from PIL import Image
    img = Image.fromarray(rgb.astype(np.uint8))
    img = img.resize((target_w, target_h), Image.NEAREST)
    return np.asarray(img, dtype=np.float32)


# ============================================================================
# Discovery (cheap: directory-name parsing only, no image data touched)
# ============================================================================

def _find_hsi_root(root):
    """The paper calls this folder `02_01_HSI_Images`, but real downloads
    have been seen using slightly different names (e.g. `02_01_HS_Images`).
    Try the documented name first, then fall back to a fuzzy search."""
    exact = os.path.join(root, "02_01_HSI_Images")
    if os.path.isdir(exact):
        return exact
    exact2 = os.path.join(root, "02_01_HS_Images")
    if os.path.isdir(exact2):
        return exact2

    candidates = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        n = name.lower()
        if "hs" in n and "image" in n:
            candidates.append(full)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # prefer one that starts with "02" (matches the documented numbering)
        for c in candidates:
            if os.path.basename(c).startswith("02"):
                return c
        return candidates[0]
    return None


def _find_roi_root(root):
    exact = os.path.join(root, "01_03_HSI_ROI_Annotations")
    if os.path.isdir(exact):
        return exact
    for name in os.listdir(root) if os.path.isdir(root) else []:
        full = os.path.join(root, name)
        n = name.lower()
        if os.path.isdir(full) and "roi" in n and "annot" in n:
            return full
    return None  # fine to be missing - ROI masking is optional


TISSUE_ALIASES = {"idc": "IDC", "healthy": "Healthy", "dcis": "DCIS"}


def _capture_dirs_by_content(hsi_root):
    """Any directory that actually contains a .hdr file is a capture folder,
    regardless of what it (or its parents) are named. Far more robust than
    matching folder names, since real downloads nest things differently
    than the paper describes (e.g. tissue-type folders containing capture
    folders, rather than flat HSI_VNIR_... folders)."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(hsi_root):
        if any(f.lower().endswith(".hdr") for f in filenames):
            out.append(dirpath)
    return out


def discover_captures(root):
    hsi_root = _find_hsi_root(root)
    if hsi_root is None:
        top_level = sorted(os.listdir(root)) if os.path.isdir(root) else []
        raise FileNotFoundError(
            f"Couldn't find the HS-images folder under {root}. Top-level contents found: "
            f"{top_level}. Pass the correct folder name to _find_hsi_root() or rename it, "
            f"then re-run.")

    # 1) try the flat, documented naming first: hsi_root/HSI_VNIR_<patient>_<tissue>_x10_C<n>/
    flat_dirs = sorted(
        d for d in os.listdir(hsi_root) if os.path.isdir(os.path.join(hsi_root, d)) and CAPTURE_RE.match(d)
    )
    out = []
    if flat_dirs:
        for d in flat_dirs:
            m = CAPTURE_RE.match(d)
            out.append({"key": d, "dir": os.path.join(hsi_root, d),
                        "patient": m["patient"], "tissue": canonicalize_tissue(m["tissue"]),
                        "capture": m["capture"]})
        return out

    # 2) fall back to content-based discovery for nested layouts (e.g.
    #    hsi_root/<DCIS|Healthy|IDC>/<patient>/<capture>/*.hdr)
    capture_dirs = _capture_dirs_by_content(hsi_root)
    if not capture_dirs:
        sample = []
        for name in sorted(os.listdir(hsi_root))[:5]:
            sub = os.path.join(hsi_root, name)
            if os.path.isdir(sub):
                children = sorted(os.listdir(sub))[:5]
                sample.append(f"  {name}/ -> {children}")
        raise RuntimeError(
            f"Found the HS-images folder at {hsi_root} but couldn't find any subfolder "
            f"containing a .hdr file anywhere inside it. Sample of what's actually there:\n"
            + "\n".join(sample) +
            "\nCheck --root, or share this structure so the discovery logic can be adjusted.")

    n_no_tissue = 0
    for d in capture_dirs:
        rel_parts = os.path.relpath(d, hsi_root).split(os.sep)
        leaf_name = rel_parts[-1]
        m = CAPTURE_RE.match(leaf_name)
        if m:
            patient, tissue, capture = m["patient"], canonicalize_tissue(m["tissue"]), m["capture"]
        else:
            tissue = next((TISSUE_ALIASES[p.lower()] for p in rel_parts if p.lower() in TISSUE_ALIASES), None)
            if tissue is None:
                n_no_tissue += 1
                continue  # can't safely label this capture - skip it (reported below)
            tissue_idx = next(i for i, p in enumerate(rel_parts) if p.lower() in TISSUE_ALIASES)
            # convention assumed: <tissue>/<patient>/<capture .hdr files here>
            patient = rel_parts[tissue_idx + 1] if tissue_idx + 1 < len(rel_parts) else leaf_name
            cap_match = re.search(r"(\d+)$", leaf_name)
            capture = cap_match.group(1) if cap_match else "1"
        key = os.path.relpath(d, hsi_root).replace(os.sep, "_")
        out.append({"key": key, "dir": d, "patient": patient, "tissue": tissue, "capture": capture})

    if n_no_tissue:
        print(f"  [warn] {n_no_tissue} capture folder(s) skipped - couldn't tell their tissue type "
              f"from the path (none of the ancestor folder names matched IDC/healthy/DCIS)")

    print(f"  [note] used content-based discovery ({len(out)} folders containing .hdr files). "
          f"Patient IDs were INFERRED from folder structure - verify this sample looks right "
          f"(patient grouping matters for the train/test split):")
    for c in out[:8]:
        print(f"    key={c['key']!r}  patient={c['patient']!r}  tissue={c['tissue']!r}  capture={c['capture']!r}")
    if len(out) > 8:
        print(f"    ... and {len(out) - 8} more")

    return out


def build_labels(captures, label_source, clinical_xlsx):
    if label_source == "tissue":
        labels = {c["key"]: TISSUE_TO_ID[c["tissue"]] for c in captures}
        return labels, ["healthy", "DCIS", "IDC"]
    elif label_source == "recurrence":
        rec_labels = load_recurrence_labels(clinical_xlsx)
        labels = {}
        for c in captures:
            if c["patient"] not in rec_labels:
                print(f"  [warn] no recurrence label for patient {c['patient']}, skipping capture {c['key']}")
                continue
            labels[c["key"]] = rec_labels[c["patient"]]
        return labels, ["no_recurrence", "recurrence"]
    else:
        raise ValueError(label_source)


# ############################################################################
# Part 2 - HistologyHsiAdapter (from prepare_histologyhsi_bc_v7.py)
# ############################################################################

_FLOAT16_SAFE_MAX = 60000.0


class HistologyHsiAdapter(DatasetAdapter):
    name = "histology_hsi_bc"
    flat_output = False

    def __init__(self):
        self._args = None

    # ---- static description -----------------------------------------
    @property
    def class_names(self) -> List[str]:
        src = getattr(self._args, "label_source", "tissue") if self._args else "tissue"
        return ["healthy", "DCIS", "IDC"] if src == "tissue" else ["no_recurrence", "recurrence"]

    def modalities(self) -> List[str]:
        m = getattr(self._args, "modality", "both") if self._args else "both"
        return ["hsi", "rgb"] if m == "both" else [m]

    def add_cli_args(self, p) -> None:
        p.add_argument("--modality", choices=["hsi", "rgb", "both"], default="both")
        p.add_argument("--label_source", choices=["tissue", "recurrence"], default="tissue")
        p.add_argument("--clinical_xlsx", default=None)
        p.add_argument("--patch_size", type=int, default=11)
        p.add_argument("--stride", type=int, default=11)
        p.add_argument("--roi_min_frac", type=float, default=0.8)
        p.add_argument("--rgb_normalize", action="store_true",
                        help="divide RGB by 255 (v6 stores raw 0-255 float; off by default "
                             "to match v6)")
        p.add_argument("--band_selection", default="none",
                        choices=["none", "uniform", "variance", "correlation", "mutual_information",
                                  "manual", "hybrid", "topk", "importance"])
        p.add_argument("--num_bands", type=int, default=None)
        p.add_argument("--importance_threshold", type=float, default=None)
        p.add_argument("--max_bands", type=int, default=None)
        p.add_argument("--band_file", default="selected_bands.npy")
        p.add_argument("--band_selection_sample_fraction", type=float, default=0.10)
        p.add_argument("--band_corr_threshold", type=float, default=0.98)
        # v15 R2.2 - shared with prepare_histologyhsi_bc_v6.py, which owns the
        # implementations this adapter delegates to.
        p.add_argument("--band_max_corr", type=float, default=0.95,
                        help="[v15 R2.2a] skip a ranked candidate whose |correlation| with an "
                             "already-selected band exceeds this. >= 1.0 disables.")
        p.add_argument("--band_min_gap", type=int, default=0,
                        help="[v15 R2.2a] minimum index distance between selected bands (0 = off)")
        p.add_argument("--band_min_coverage", type=float, default=0.30,
                        help="[v15 R2.2c] minimum fraction of the sensor's wavelength range the "
                             "selection must span; the prep FAILS below it. 0.0 disables.")
        p.add_argument("--allow_narrow_bands", action="store_true",
                        help="[v15 R2.2c] downgrade the coverage failure to a warning")

    def preprocess_args(self, args) -> None:
        self._args = args
        if not getattr(args, "clinical_xlsx", None) and args.root:
            cand = os.path.join(args.root, "HistologyHSI-BC-Recurrence-Clinical-Standardized.xlsx")
            args.clinical_xlsx = cand if os.path.exists(cand) else None
        args.roi_root = _find_roi_root(args.root) if args.root else None
        # v6's auto-selection rules
        if args.band_selection == "none" and (args.num_bands is not None
                                               or args.importance_threshold is not None):
            args.band_selection = "importance"
        if (args.band_selection in ("uniform", "variance", "correlation", "mutual_information",
                                     "hybrid", "topk")
                and args.num_bands is None):
            print("  [band-selection] no --num_bands given; defaulting to 64")
            args.num_bands = 64

    def compat_keys(self, args) -> dict:
        return {
            "modality": args.modality, "label_source": args.label_source,
            "patch_size": args.patch_size, "stride": args.stride,
            "roi_min_frac": args.roi_min_frac, "rgb_normalize": args.rgb_normalize,
            "band_selection": args.band_selection,
            "num_bands": args.num_bands if args.band_selection != "none" else None,
            "importance_threshold": (args.importance_threshold
                                      if args.band_selection == "importance" else None),
            "max_bands": args.max_bands if args.band_selection == "importance" else None,
        }

    # ---- discovery -------------------------------------------------
    def discover(self, args) -> Iterator[PrepItem]:
        caps = discover_captures(args.root)
        labels, _names = build_labels(caps, args.label_source, args.clinical_xlsx)
        for c in caps:
            if c["key"] not in labels:
                continue
            yield PrepItem(
                key=c["key"], group_id=c["patient"], label=int(labels[c["key"]]),
                payload={"dir": c["dir"], "tissue": c["tissue"], "capture": c["capture"],
                          "patient": c["patient"]})

    # ---- Pass 1: cube probe + optional band selection -------------
    def pass1(self, items, args) -> Optional[dict]:
        caps = [{"key": it.key, "dir": it.payload["dir"], "tissue": it.payload["tissue"],
                  "capture": it.payload["capture"], "patient": it.group_id} for it in items]
        labels = {it.key: it.label for it in items}
        state: Dict = {"wavelengths_full": None, "selected_band_indices": None,
                        "hsi_all_calibrated": True, "hsi_value_max": 1.5}

        if args.band_selection == "manual":
            idx = np.load(args.band_file)
            state["selected_band_indices"] = np.asarray(sorted(int(i) for i in idx),
                                                         dtype=np.int64).tolist()

        need_stats = args.band_selection not in ("none", "manual")
        if need_stats or True:
            try:
                pixels, pix_labels, wl = compute_band_statistics(
                    caps, args.roi_root, labels, args.band_selection_sample_fraction, args.seed)
                state["wavelengths_full"] = None if wl is None else np.asarray(wl, np.float32).tolist()
                state["hsi_value_max"] = float(np.abs(pixels).max()) if pixels.size else 1.5
                if need_stats:
                    if args.band_selection == "importance":
                        imp = compute_band_importance(pixels, pix_labels, seed=args.seed)
                        max_corr = (None if getattr(args, "band_max_corr", 1.0) >= 1.0
                                    else args.band_max_corr)
                        sel, ranking, n_inf = select_bands_by_importance(
                            imp, args.num_bands, args.importance_threshold, args.max_bands,
                            pixels=pixels, max_corr=max_corr,
                            min_gap=getattr(args, "band_min_gap", 0))
                        state["selected_band_indices"] = np.sort(sel).astype(np.int64).tolist()
                        state["_importance"] = (imp, ranking, n_inf, pixels.shape[1])
                    else:
                        idx = select_bands(args.band_selection, pixels, pix_labels,
                                                args.num_bands, args.band_corr_threshold, args.seed)
                        state["selected_band_indices"] = np.asarray(
                            sorted(int(i) for i in idx), dtype=np.int64).tolist()
            except Exception as e:  # band-count mismatch etc. -> fail loud only if we needed it
                if need_stats:
                    raise
                print(f"  [pass1] cube probe failed ({e}); wavelengths.npy may be absent")

        # calibration probe (v6 semantics: uncalibrated cubes are stored raw)
        for c in caps[: max(3, math.ceil(len(caps) * args.band_selection_sample_fraction))]:
            cube = _find_envi_cube(c["dir"])
            if cube is None:
                continue
            w, d = get_calibration_means(c["dir"], cube)
            if w is None or d is None:
                state["hsi_all_calibrated"] = False
                break
        return state

    def write_pass1_outputs(self, state, out_dir) -> List[str]:
        imp_bundle = state.pop("_importance", None) if state else None
        if not state or state.get("selected_band_indices") is None:
            return []
        idx = np.asarray(state["selected_band_indices"], dtype=np.int64)
        wl = state.get("wavelengths_full")
        wl = None if wl is None else np.asarray(wl, np.float32)
        a = self._args
        min_cov = getattr(a, "band_min_coverage", None)
        allow_narrow = getattr(a, "allow_narrow_bands", False)
        if imp_bundle is not None:
            imp, ranking, n_inf, orig = imp_bundle
            max_corr = None if getattr(a, "band_max_corr", 1.0) >= 1.0 else a.band_max_corr
            save_importance_selection_report(
                out_dir, np.asarray(imp), np.asarray(ranking), idx, n_inf,
                a.num_bands, a.importance_threshold, orig, wl, max_bands=a.max_bands,
                min_coverage=min_cov, allow_narrow=allow_narrow,
                max_corr=max_corr, min_gap=getattr(a, "band_min_gap", 0))
        else:
            save_selected_bands(out_dir, a.band_selection, idx, wavelengths=wl,
                                     original_bands=(len(wl) if wl is not None else None),
                                     num_bands_requested=a.num_bands,
                                     sample_fraction=a.band_selection_sample_fraction,
                                     min_coverage=min_cov, allow_narrow=allow_narrow)
        return ["selected_band_indices.npy"]

    def finalize_modality(self, modality, modality_dir, state, manifest) -> List[str]:
        if modality != "hsi" or not state:
            return []
        wl_full = state.get("wavelengths_full")
        if wl_full is None:
            return []
        wl_full = np.asarray(wl_full, dtype=np.float32)
        written = []
        sel = state.get("selected_band_indices")
        from training.npy_atomic import save_npy_atomic
        if sel is not None:
            sel = np.asarray(sel, dtype=np.int64)
            save_npy_atomic(os.path.join(modality_dir, "wavelengths_full.npy"), wl_full)
            save_npy_atomic(os.path.join(modality_dir, "wavelengths.npy"), wl_full[sel])
            written = ["wavelengths.npy", "wavelengths_full.npy"]
        else:
            save_npy_atomic(os.path.join(modality_dir, "wavelengths.npy"), wl_full)
            written = ["wavelengths.npy"]
        return written

    # ---- per-item extraction -------------------------------------
    def modality_spec(self, modality, args, state) -> ModalitySpec:
        sel = (state or {}).get("selected_band_indices")
        c = len(sel) if sel is not None else None
        if modality == "rgb":
            return ModalitySpec(store_dtype="float32",
                                 sample_shape=(args.patch_size, args.patch_size, 3))
        calibrated = bool((state or {}).get("hsi_all_calibrated", True))
        vmax = float((state or {}).get("hsi_value_max", 1.5))
        dtype = "float16" if (calibrated and vmax <= _FLOAT16_SAFE_MAX) else "float32"
        return ModalitySpec(store_dtype=dtype,
                             sample_shape=(args.patch_size, args.patch_size, c) if c else None)

    def load_item(self, item, modality, args, state) -> Iterator[Tuple[np.ndarray, int]]:
        streams = self.load_item_multi(item, [modality], args, state)
        yield from streams[modality]

    def load_item_multi(self, item, modalities, args, state):
        from PIL import Image

        cap_dir = item.payload["dir"]
        sel = (state or {}).get("selected_band_indices")
        sel = None if sel is None else np.asarray(sel, dtype=np.int64)
        ps, step = args.patch_size, (args.stride or args.patch_size)

        cube_img = coords = cube_hw = None
        white = dark = None
        if "hsi" in modalities:
            cube_path = _find_envi_cube(cap_dir)
            if cube_path is not None:
                cube_img, _wl = open_envi_lazy(cube_path)
                cube_hw = cube_img.shape[:2]
                white, dark = get_calibration_means(cap_dir, cube_path)
                mask = None
                gj = find_roi_geojson(args.roi_root, item.payload["patient"],
                                           item.payload["tissue"], item.payload["capture"])
                if gj:
                    try:
                        mask = rasterize_roi_mask(gj, cube_hw[0], cube_hw[1])
                    except Exception as e:  # noqa: BLE001
                        print(f"    [warn] ROI rasterize failed for {item.key}: {e}")
                coords = patch_coords(cube_hw[0], cube_hw[1], ps, step, mask, args.roi_min_frac)

        def _hsi_stream():
            if cube_img is None or not coords:
                return
            for (y, x) in coords:
                patch = np.asarray(cube_img[y:y + ps, x:x + ps, :], dtype=np.float32)
                if white is not None and dark is not None:
                    patch = calibrate_patch(patch, white, dark)
                if sel is not None:
                    patch = patch[:, :, sel]
                yield patch, item.label

        def _rgb_stream():
            rgb_path = _find_rgb_image(cap_dir)
            if rgb_path is None:
                return
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
            if cube_hw is not None and rgb.shape[:2] != tuple(cube_hw):
                rgb = resize_nn(rgb, cube_hw[0], cube_hw[1])
            local_coords = coords
            if local_coords is None:
                local_coords = patch_coords(rgb.shape[0], rgb.shape[1], ps, step, None,
                                                 args.roi_min_frac)
            if args.rgb_normalize:
                rgb = rgb / 255.0
            for (y, x) in local_coords:
                yield rgb[y:y + ps, x:x + ps, :].copy(), item.label

        out = {}
        if "hsi" in modalities:
            out["hsi"] = _hsi_stream()
        if "rgb" in modalities:
            out["rgb"] = _rgb_stream()
        return out

    def provenance(self, args, state) -> dict:
        return {"label_source": args.label_source, "modality": args.modality,
                "band_selection": args.band_selection,
                "selected_band_indices": (state or {}).get("selected_band_indices")}


# ############################################################################
# Part 3 - HistologyHsiAdapterV8, the live adapter (from prepare_histologyhsi_bc_v8.py)
# ############################################################################

_DEFAULT_GAIN_CLIP = (0.5, 2.0)


def _sample_capture_median(cap_dir: str, roi_root, patient, tissue, capture, roi_min_frac: float,
                            sel: Optional[np.ndarray], seed: int, n_pixels: int = 400) -> Optional[float]:
    """Median intensity of up to `n_pixels` random (ROI-respecting when
    available) pixels from one capture, AFTER the same flat-field
    calibration `_hsi_stream` applies (so the gain this produces composes
    correctly with it), and after band selection if `sel` is given."""
    cube_path = _find_envi_cube(cap_dir)
    if cube_path is None:
        return None
    cube_img, _wl = open_envi_lazy(cube_path)
    h, w = cube_img.shape[:2]
    white, dark = get_calibration_means(cap_dir, cube_path)

    mask = None
    gj = find_roi_geojson(roi_root, patient, tissue, capture) if roi_root else None
    if gj:
        try:
            mask = rasterize_roi_mask(gj, h, w)
        except Exception:
            mask = None

    rng = np.random.default_rng(seed)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        if len(ys) > n_pixels:
            keep = rng.choice(len(ys), size=n_pixels, replace=False)
            ys, xs = ys[keep], xs[keep]
    else:
        n_px = int(min(n_pixels, h * w))
        ys = rng.integers(0, h, size=n_px)
        xs = rng.integers(0, w, size=n_px)
    if len(ys) == 0:
        return None

    # `spectral`'s SpyFile.__getitem__ does not support numpy-style
    # advanced (paired coordinate array) indexing - single-pixel scalar
    # indexing, looped, matches `prepare_histologyhsi_bc_v6.
    # compute_band_statistics`'s own sampling pattern (frozen, reused here).
    vals = []
    for y, x in zip(ys, xs):
        px = np.asarray(cube_img[int(y), int(x), :], dtype=np.float32).reshape(1, 1, -1)
        if white is not None and dark is not None:
            px = calibrate_patch(px, white, dark)
        px = px.reshape(-1)
        if sel is not None:
            px = px[sel]
        vals.append(px)
    pix = np.stack(vals, axis=0) if vals else np.empty((0,), dtype=np.float32)
    return float(np.median(pix)) if pix.size else None


# ---------------------------------------------------------------------------
# --rgb_source: which of the two PNGs beside each cube becomes the RGB arm
# ---------------------------------------------------------------------------
# Every capture folder ships TWO renderings, and they are not interchangeable.
# Measured over all 644 captures of the HistologyHSI-BC-Recurrence download:
#
#   SyntheticRGBImage.png   cube-aligned in 644/644 (100%)
#   RGBImage.png            cube-aligned in 416/644  (65%) - the other 228 are
#                           a wide-field 3648x5472 camera frame of a DIFFERENT
#                           field of view, and some carry the green ROI
#                           annotation burned into the pixels
#
# `prepare_histologyhsi_bc_v6._find_rgb_image` sorts candidates by
# `0 if "rgb" in n or "synt" in n else 1` - and BOTH names contain "rgb", so
# both score 0 and the winner is whatever `os.listdir` happened to return
# first. On this machine that resolves to RGBImage.png for 565 captures and
# SyntheticRGBImage.png for 79, i.e. the RGB arm is built from a MIXTURE of two
# optical sources, 190 of them (29.5%) NN-resized from a different field of
# view onto the cube's patch coordinates and ROI mask. Nothing is aligned to
# the HSI there, and the mixture is not reproducible across filesystems.
#
# That matters because the RGB arm exists to be the CONTROLLED comparison
# against the HSI arm: same tissue, same coordinates, fewer channels.
#
#   synthetic (default)  SyntheticRGBImage.png always. Pixel-registered with
#                        the cube by construction, carries no annotation
#                        overlay, and is the only choice that makes
#                        "HSI vs RGB" a single-variable comparison.
#   camera               RGBImage.png always. An INDEPENDENT sensor rather
#                        than a rendering of the cube - the more realistic
#                        "would an RGB microscope do?" question - but 35% of
#                        captures are then resized from a different FOV.
#   legacy               v6's arbitrary os.listdir order. Only for reproducing
#                        a dataset built before this flag existed.
RGB_SOURCE_PREFERENCE = {
    "synthetic": ("syntheticrgbimage", "synthetic", "synt"),
    "camera": ("rgbimage",),
}
_RGB_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def make_rgb_finder(mode: str):
    """A deterministic replacement for `_find_rgb_image`.

    Returns v6's own function for 'legacy'. Otherwise ranks by an explicit
    preference list and breaks every remaining tie by SORTED FILENAME, so the
    result never depends on directory iteration order.
    """
    if mode == "legacy":
        return _find_rgb_image
    prefs = RGB_SOURCE_PREFERENCE[mode]

    def _find(capture_dir: str):
        names = sorted(f for f in os.listdir(capture_dir) if f.lower().endswith(_RGB_EXTS))
        if not names:
            return None

        def rank(n: str) -> int:
            low = os.path.splitext(n.lower())[0]
            for i, key in enumerate(prefs):
                if key in low:
                    return i
            return len(prefs)

        names.sort(key=lambda n: (rank(n), n))
        return os.path.join(capture_dir, names[0])

    return _find


def _capture_median_task(task):
    """Top-level (picklable) wrapper around `_sample_capture_median` for the pass-1 pool."""
    key, cap_dir, roi_root, patient, tissue, capture, roi_min_frac, sel, seed, n_pixels = task
    try:
        return key, _sample_capture_median(cap_dir, roi_root, patient, tissue, capture,
                                           roi_min_frac, sel, seed=seed, n_pixels=n_pixels)
    except Exception as e:                                   # one bad cube must not kill pass 1
        print(f"  [capture-gain warn] {key}: {type(e).__name__}: {e}", flush=True)
        return key, None


def _capture_medians(items, args, sel_arr) -> Dict[str, float]:
    """Per-capture median intensity for every item, over `--num_workers` processes.

    Pass 2 extraction has used a process pool since v6 (`training/prep/parallel.py:104`);
    pass 1 never did, so this walked all 644 captures SERIALLY in the parent, and
    `_sample_capture_median` reads pixels one at a time in a Python loop - 400 random
    seeks per capture, ~257,600 single-pixel ENVI reads for the full corpus.

    Parallelising is exactly value-preserving here, which is why it is safe to do at all:
    `_sample_capture_median` is called with `seed=args.seed` - the SAME seed for every
    capture - so each capture's RNG stream is already independent of the others and of
    iteration order. Every median comes back bit-identical to the serial version.
    `fork` matches what `training/prep/parallel.py` uses.
    """
    tasks = [(it.key, it.payload["dir"], args.roi_root, it.payload["patient"],
              it.payload["tissue"], it.payload["capture"], args.roi_min_frac, sel_arr,
              args.seed, args.gain_sample_pixels) for it in items]
    n_workers = max(1, int(getattr(args, "num_workers", 1) or 1))
    reporter = ProgressReporter("  [capture-gain]", min_interval=5.0, stream_prefix="")
    medians: Dict[str, float] = {}
    done = 0

    def _take(key, m):
        nonlocal done
        done += 1
        if m is not None and m > 1e-6:
            medians[key] = m
        reporter.emit(done, len(tasks), f"{len(medians)} usable")

    if n_workers > 1 and len(tasks) > 1:
        ctx = get_context("fork") if "fork" in mp.get_all_start_methods() else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            for fut in as_completed([ex.submit(_capture_median_task, t) for t in tasks]):
                _take(*fut.result())
    else:
        for t in tasks:
            _take(*_capture_median_task(t))
    return medians


class HistologyHsiAdapterV8(HistologyHsiAdapter):
    name = "histology_hsi_bc_v8"

    def add_cli_args(self, p) -> None:
        super().add_cli_args(p)
        p.add_argument("--capture_gain", choices=["none", "median_ratio"], default="median_ratio",
                        help="[v16 A-1] per-capture flat-field residual correction, applied between "
                             "calibration and band selection.")
        p.add_argument("--gain_clip_lo", type=float, default=_DEFAULT_GAIN_CLIP[0])
        p.add_argument("--gain_clip_hi", type=float, default=_DEFAULT_GAIN_CLIP[1])
        p.add_argument("--hsi_value_scale", default="auto",
                        help="[v16 R-3] float, 'auto' (99th pct of the gain-corrected pass-1 "
                             "sample), or 'none'.")
        p.add_argument("--emit_group_sidecars", action="store_true", default=True)
        p.add_argument("--no_emit_group_sidecars", dest="emit_group_sidecars", action="store_false")
        p.add_argument("--gain_sample_pixels", type=int, default=400,
                        help="pixels sampled per capture for the median-ratio gain estimate")
        p.add_argument("--rgb_source", choices=["synthetic", "camera", "legacy"], default="synthetic",
                        help="which PNG beside each cube becomes the RGB arm. 'synthetic' "
                             "(default) = SyntheticRGBImage.png, cube-aligned in 644/644 captures "
                             "and carrying no burned-in ROI overlay - the only choice that makes "
                             "HSI-vs-RGB a single-variable comparison. 'camera' = RGBImage.png, an "
                             "independent sensor, but 228/644 are a different field of view and get "
                             "NN-resized onto the cube's coordinates. 'legacy' = v6's os.listdir "
                             "order, which MIXES the two (565/79 on this machine) and is not "
                             "reproducible; use it only to rebuild a pre-existing dataset.")

    def compat_keys(self, args) -> dict:
        keys = dict(super().compat_keys(args))
        keys.update({
            "capture_gain": args.capture_gain,
            "hsi_value_scale": str(args.hsi_value_scale),
            "gain_clip": [args.gain_clip_lo, args.gain_clip_hi],
            # Without this, resuming a 'camera' prep with --rgb_source synthetic
            # would append cube-aligned patches to misaligned ones inside one
            # X_train.npy, and nothing downstream could tell.
            "rgb_source": args.rgb_source,
        })
        return keys

    # ---- Pass 1: v7's band-selection/calibration probe, plus per-capture gain ----
    def pass1(self, items, args) -> Optional[dict]:
        state = super().pass1(items, args) or {}
        sel = state.get("selected_band_indices")
        sel_arr = None if sel is None else np.asarray(sel, dtype=np.int64)

        if args.capture_gain == "none":
            state["capture_gain"] = {}
        else:
            medians = _capture_medians(items, args, sel_arr)      # v17 S6: parallel + ETA
            corpus_median = float(np.median(list(medians.values()))) if medians else 1.0
            gains, suspects = {}, []
            # sorted(): with the v17 S6 pool, `medians` is built in completion order, so
            # iterating it raw would make `capture_gain_suspects` (written to
            # capture_gain_report.json) and the warning below come out in a different
            # order run to run. The VALUES were already order-independent; this makes the
            # report byte-identical too.
            for key, m in sorted(medians.items()):
                raw_gain = corpus_median / m
                gain = float(np.clip(raw_gain, args.gain_clip_lo, args.gain_clip_hi))
                gains[key] = {"gain": gain, "median_intensity": m, "n_sampled": args.gain_sample_pixels,
                              "raw_gain_before_clip": raw_gain}
                if not (args.gain_clip_lo <= raw_gain <= args.gain_clip_hi):
                    suspects.append(key)
            state["capture_gain"] = gains
            state["capture_gain_corpus_median"] = corpus_median
            state["capture_gain_suspects"] = suspects
            if suspects:
                print(f"  [capture-gain] {len(suspects)} capture(s) needed a raw gain outside "
                      f"[{args.gain_clip_lo}, {args.gain_clip_hi}] (clipped) - logged as SUSPECT "
                      f"ACQUISITION: {suspects[:10]}"
                      + (f" (+{len(suspects) - 10} more)" if len(suspects) > 10 else ""), flush=True)

        # --hsi_value_scale resolution, on the GAIN-CORRECTED sample.
        if str(args.hsi_value_scale) == "none":
            state["hsi_value_scale_resolved"] = 1.0
        elif str(args.hsi_value_scale) == "auto":
            gains = state.get("capture_gain", {})
            samples = []
            for it in items[: max(3, math.ceil(len(items) * args.band_selection_sample_fraction))]:
                m = gains.get(it.key, {}).get("median_intensity")
                g = gains.get(it.key, {}).get("gain", 1.0)
                if m is not None:
                    samples.append(m * g)
            base_max = state.get("hsi_value_max", 1.5)
            p99 = float(np.percentile(samples, 99)) if samples else base_max
            # Land the 99th percentile at ~1.0 reflectance-like units; guard
            # against a degenerate (near-zero) estimate.
            state["hsi_value_scale_resolved"] = max(p99, 1e-3)
        else:
            state["hsi_value_scale_resolved"] = float(args.hsi_value_scale)

        # A-1 calibration-flag fix: tie "is this safely float16-able" to the
        # ACTUAL post-scale value range, not the frozen (inverted-reading)
        # probe boolean.
        scale = state["hsi_value_scale_resolved"]
        post_scale_max = state.get("hsi_value_max", 1.5) / max(scale, 1e-9)
        state["hsi_all_calibrated"] = bool(post_scale_max <= _FLOAT16_SAFE_MAX)
        state["hsi_value_max_post_scale"] = post_scale_max
        return state

    def write_pass1_outputs(self, state, out_dir) -> List[str]:
        written = list(super().write_pass1_outputs(state, out_dir))
        if not state:
            return written
        report = {
            "capture_gain_method": "median_ratio" if state.get("capture_gain") else "none",
            "corpus_median": state.get("capture_gain_corpus_median"),
            "suspect_captures": state.get("capture_gain_suspects", []),
            "hsi_value_scale_resolved": state.get("hsi_value_scale_resolved"),
            "hsi_value_max_pre_scale": state.get("hsi_value_max"),
            "hsi_value_max_post_scale": state.get("hsi_value_max_post_scale"),
            "hsi_all_calibrated_v8": state.get("hsi_all_calibrated"),
            "per_capture": state.get("capture_gain", {}),
        }
        save_json_atomic(os.path.join(out_dir, "capture_gain_report.json"), report)
        written.append("capture_gain_report.json")
        return written

    # ---- dtype selection, on the v8-corrected calibration signal ----
    def modality_spec(self, modality, args, state) -> ModalitySpec:
        if modality != "hsi":
            return super().modality_spec(modality, args, state)
        sel = (state or {}).get("selected_band_indices")
        c = len(sel) if sel is not None else None
        calibrated = bool((state or {}).get("hsi_all_calibrated", False))
        vmax = float((state or {}).get("hsi_value_max_post_scale", (state or {}).get("hsi_value_max", 1.5)))
        dtype = "float16" if (calibrated and vmax <= _FLOAT16_SAFE_MAX) else "float32"
        return ModalitySpec(store_dtype=dtype,
                             sample_shape=(args.patch_size, args.patch_size, c) if c else None)

    # ---- extraction: gain correction + value scale, between calibration and band slice ----
    def load_item_multi(self, item, modalities, args, state):
        # Install the --rgb_source choice HERE, not in pass1/preprocess_args.
        # `training/prep/parallel.py` extracts items in worker PROCESSES; under
        # spawn (and on any restart) a rebind done in the parent would not be
        # present in the worker, and the run would silently fall back to v6's
        # os.listdir order for part of the corpus - the exact non-determinism
        # this flag exists to remove. `load_item_multi` runs in whichever
        # process does the extraction, so setting it here always takes effect.
        # `_find_rgb_image` is looked up as a module GLOBAL by
        # `HistologyHsiAdapter.load_item_multi._rgb_stream` at call time, so
        # rebinding the global is enough.
        global _find_rgb_image
        _find_rgb_image = make_rgb_finder(getattr(args, "rgb_source", "synthetic"))
        streams = super().load_item_multi(item, modalities, args, state)
        if "hsi" not in streams:
            return streams

        gain = (state or {}).get("capture_gain", {}).get(item.key, {}).get("gain", 1.0)
        scale = (state or {}).get("hsi_value_scale_resolved", 1.0)

        def _corrected(inner):
            for patch, label in inner:
                if gain != 1.0:
                    patch = patch * gain
                if scale != 1.0:
                    patch = patch / scale
                yield patch, label

        streams["hsi"] = _corrected(streams["hsi"])
        return streams

    def provenance(self, args, state) -> dict:
        base = super().provenance(args, state)
        base.update({
            "capture_gain_method": args.capture_gain,
            "gain_clip": [args.gain_clip_lo, args.gain_clip_hi],
            "hsi_value_scale_resolved": (state or {}).get("hsi_value_scale_resolved"),
            "hsi_all_calibrated_v8": (state or {}).get("hsi_all_calibrated"),
            "rgb_source": args.rgb_source,
        })
        return base

    # ---- Stage 0.3's sidecars, written directly (this adapter already
    # knows patient/tissue/capture per item; no key-regex reconstruction). ----
    def finalize_modality(self, modality, modality_dir, state, manifest) -> List[str]:
        written = list(super().finalize_modality(modality, modality_dir, state, manifest) or [])
        if not getattr(self._args, "emit_group_sidecars", True):
            return written

        capture_index: Dict[int, dict] = {}
        key_to_idx: Dict[str, int] = {}
        for i, key in enumerate(sorted(manifest.get("completed_items", {}))):
            it = manifest["completed_items"][key]
            key_to_idx[key] = i

        for split, out_name in (("train", "train"), ("validation", "val"), ("test", "test")):
            entries = [(k, v) for k, v in sorted(manifest.get("completed_items", {}).items())
                       if v.get("split") == split]
            y_path = os.path.join(modality_dir, f"y_{out_name}.npy")
            if not os.path.isfile(y_path):
                continue
            n_total = len(np.load(y_path, mmap_mode="r"))
            groups, captures = [], []
            cursor = 0
            for key, it in entries:
                n = int((it.get("counts") or {}).get(modality, 0))
                if n <= 0:
                    continue
                gain_info = (state or {}).get("capture_gain", {}).get(key, {})
                idx = key_to_idx[key]
                # patient/tissue/capture: recovered from `manifest["args"]`
                # is not enough (per-item), so fall back to the same
                # CAPTURE_RE the frozen adapter's own keys satisfy - this
                # adapter's `key` format is unchanged from v6/v7.
                m = CAPTURE_RE.search(key)
                patient = m["patient"] if m else None
                capture_index[idx] = {
                    "key": key, "patient": patient, "tissue": (m["tissue"] if m else None),
                    "capture": (m["capture"] if m else None), "split": split, "n": n,
                    **({"gain": gain_info.get("gain"), "median_intensity": gain_info.get("median_intensity")}
                       if gain_info else {}),
                }
                try:
                    patient_int = int(patient) if patient is not None else idx
                except ValueError:
                    patient_int = idx
                groups.extend([patient_int] * n)
                captures.extend([idx] * n)
                cursor += n
            if cursor == n_total and groups:
                save_npy_atomic(os.path.join(modality_dir, f"groups_{out_name}.npy"),
                                np.asarray(groups, dtype=np.int32))
                save_npy_atomic(os.path.join(modality_dir, f"captures_{out_name}.npy"),
                                np.asarray(captures, dtype=np.int32))
                written += [f"groups_{out_name}.npy", f"captures_{out_name}.npy"]
            elif groups:
                print(f"  [warn] group sidecar count mismatch for {modality}/{split} "
                      f"({cursor} != {n_total}) - not writing groups_{out_name}.npy/"
                      f"captures_{out_name}.npy", flush=True)

        if capture_index:
            # NOT added to `written`: `finalize_modality`'s return value feeds
            # `finalize_dataset(..., extra_files=...)` (training/prep/core.py
            # `_unify_all`), which runs every entry through `_describe_npy`
            # (shape/dtype/sha256) for `dataset_manifest.json` - a JSON file
            # there raises trying to read an .npy magic header. Written to
            # disk directly instead, same as `capture_gain_report.json`.
            save_json_atomic(os.path.join(modality_dir, "capture_index.json"),
                             {str(k): v for k, v in capture_index.items()})
        return written


def main() -> int:
    # v17 S6 - rate + ETA on the extraction loop. training/prep/core.py is FROZEN
    # and discards the rate it already computes (core.py:400); this rebinds
    # run_extraction instead of editing it.
    install_prep_progress()
    return run_prep(HistologyHsiAdapterV8())


if __name__ == "__main__":
    raise SystemExit(main())
