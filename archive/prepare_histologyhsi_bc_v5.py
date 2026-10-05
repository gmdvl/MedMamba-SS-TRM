# -*- coding: utf-8 -*-
"""
prepare_histologyhsi_bc.py
===========================
Turns a raw HistologyHSI-BC-Recurrence download (TCIA collection
"HistologyHSI-BC-Recurrence", DOI 10.7937/6kpy-yt49) into the flat
X_train.npy / y_train.npy / X_test.npy / y_test.npy (+ wavelengths.npy)
files that `train_example.py` / `evaluate.py` already know how to load.

--------------------------------------------------------------------------
Dataset layout this script expects (per the TCIA/Scientific Data description)
--------------------------------------------------------------------------
<root>/
  HistologyHSI-BC-Recurrence-Clinical-Standardized.xlsx   (optional, for --label_source recurrence)
  01_01_Histological_Images/           (WSIs, .mrxs - not used here)
  01_02_Tissue_Annotations/            (not used here)
  01_03_HSI_ROI_Annotations/           (GeoJSON polygons, one per capture - optional)
  02_01_HSI_Images/
    HSI_VNIR_<patient>_<tissue>_x10_C<capture>/
      *.hdr / *.dat / *.raw       <- ENVI cube(s): raw capture + white/dark
                                      reference + (usually) a calibrated cube
      *.png / *.jpg / *.tif       <- synthetic RGB rendering of the same cube
                                      (pixel-aligned with the HSI cube)

`<tissue>` is one of {IDC, healthy, DCIS} and is read directly out of the
folder name - this is the per-capture label used by --label_source tissue
(the default). --label_source recurrence instead looks up each <patient> in
the clinical XLSX and uses their distant-recurrence outcome as the label,
so every patch from a given patient shares one weakly-supervised label.

Because the exact calibration/reference filenames aren't pinned down in the
public docs, this script is deliberately permissive about file discovery
(see `_find_envi_cube` / `_find_rgb_image` below) and prints what it picked
for each capture the first time you run it - **check that output once**
against your actual download and tweak the glob patterns if it picked the
wrong file (e.g. if you have already-registered TIFFs instead of raw ENVI).

--------------------------------------------------------------------------
Memory-bounded, resumable, batch-streamed processing
--------------------------------------------------------------------------
Whole-slide HSI cubes at hundreds of bands are big, and a dataset has many
captures, so this script never (a) holds a whole cube fully in RAM or
(b) accumulates patches from more than one shard in RAM at a time:

  1. ENVI cubes are opened LAZILY (`spectral` memory-maps the .dat/.raw file)
     and each patch is read directly off disk at `img[y:y+p, x:x+p, :]` -
     the full cube is never materialized in memory.
  2. Patches are buffered in small shards (`--batch_size` patches, default
     256) and flushed straight to disk under
     `<out_dir>/<hsi|rgb>/_batches/<train|test>/shard_XXXXXX_X.npy` (+ a
     matching `..._y.npy` for labels) as soon as a shard fills up - so peak
     memory is ~one shard, not the whole dataset.
  3. Progress is tracked in `<out_dir>/_progress.json`. A capture is only
     marked "completed" once ALL of its patches have been flushed; if the
     process is killed (e.g. OOM-killed) mid-capture, that capture's
     partially-written shards are discarded and it is simply reprocessed
     from scratch next run - every OTHER already-completed capture is
     skipped, so you never lose more than the one in-flight capture.
     Just re-run the exact same command to resume.
  4. Once every capture has been processed (fresh run or resumed), all the
     per-capture/per-shard files for a given split are UNIFIED into the
     final `X_train.npy` / `y_train.npy` / `X_test.npy` / `y_test.npy` (and
     `wavelengths.npy`) via `np.lib.format.open_memmap`, which writes the
     unified array directly to disk shard-by-shard without ever holding the
     full unified array in RAM either. Shards are then deleted by default
     (pass `--keep_batches` to keep them).

If a run finishes processing but dies during the unify step (rare, since
unify itself is memory-bounded), just re-run with `--finalize_only` to
redo unify without reprocessing anything.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    pip install spectral openpyxl pillow --break-system-packages

    python prepare_histologyhsi_bc.py \\
        --root /path/to/HistologyHSI-BC-Recurrence \\
        --out_dir ./data \\
        --patch_size 11 --stride 11 \\
        --label_source tissue \\
        --modality both \\
        --batch_size 256

    # if the process gets killed for any reason (OOM or otherwise), just
    # re-run the SAME command - already-completed captures are skipped:
    python prepare_histologyhsi_bc.py --root ... --out_dir ./data --patch_size 11 --stride 11 ...

    # if all captures finished but the final unify step didn't complete:
    python prepare_histologyhsi_bc.py --root ... --out_dir ./data --finalize_only

Then:
    python train_example.py hsi --x_train data/hsi/X_train.npy --y_train data/hsi/y_train.npy \\
        --x_test data/hsi/X_test.npy --y_test data/hsi/y_test.npy \\
        --wavelengths_file data/hsi/wavelengths.npy --ckpt gmedmamba_hsi_best.pt

    python train_example.py hsi --x_train data/rgb/X_train.npy --y_train data/rgb/y_train.npy \\
        --x_test data/rgb/X_test.npy --y_test data/rgb/y_test.npy \\
        --ckpt gmedmamba_rgb_best.pt
    # (yes, `hsi` mode - the RGB patches are just C=3 patch cubes, so the
    #  same HSIPatchDataset / gmedmamba_hsi_small path handles them fine;
    #  ImageFolder-based `train_example.py rgb` expects whole images, not
    #  small aligned patches, so it isn't the right mode here.)

--------------------------------------------------------------------------
v2 additions (see train-improve-prompt_v2.plan.md)
--------------------------------------------------------------------------
This version adds three optional, backward-compatible features on top of
the pipeline described above. With all of them left at their defaults,
this script produces byte-for-byte the same outputs/behaviour as before.

  1. Spectral band selection (HSI only, --band_selection)
     A cheap "Pass 1" opens a small sample of cubes, computes per-band
     statistics (variance / correlation / mutual information with the
     label / a manual index file / a "hybrid"/"topk" combination of all
     three), and picks a subset of wavelengths. "Pass 2" (the normal
     capture loop) then keeps only those bands when it slices each patch
     off the memory-mapped cube, so smaller cubes are written to disk -
     no changes to GMedMamba itself are required, since `wavelengths.npy`
     is simply written to match the reduced band count.

  2. Class balancing (--balance_classes {undersample,oversample})
     Applied only to the TRAIN split, only at the final unify step (after
     every shard has already been written), so it never complicates the
     streaming/resumable extraction path. Validation/test are always left
     untouched so evaluation stays realistic.

  3. Flexible dataset splits (--split {80_20,80_10_10,70_15_15,70_30,
     60_20_20}) - replaces the old --test_size float. Splitting is still
     done at the PATIENT level. `80_20` (the default) reproduces the old
     default (test_size=0.2) exactly; validation is only created for the
     three-way presets.

A `dataset_statistics.json` summarizing all of the above (band counts,
class counts before/after balancing, split sizes, seed, patch/stride,
ROI settings, ...) is written to `--out_dir` once processing finishes.
"""

import argparse
import bisect
import gc
import json
import os
import re
import random
from collections import defaultdict

import numpy as np

from training.dataset_prep_common import stratified_patient_split_three, check_prep_class_coverage

CAPTURE_RE = re.compile(r"HSI?_VNIR_(?P<patient>[^_]+)_(?P<tissue>IDC|Healthy|DCIS)_x10_C(?P<capture>\d+)",
                         re.IGNORECASE)
TISSUE_TO_ID = {"Healthy": 0, "DCIS": 1, "IDC": 2}
PROGRESS_FILE = "_progress.json"

# Phase 3 - flexible dataset splits: preset name -> (train_frac, val_frac, test_frac)
SPLIT_PRESETS = {
    "80_20":    (0.80, 0.00, 0.20),
    "80_10_10": (0.80, 0.10, 0.10),
    "70_15_15": (0.70, 0.15, 0.15),
    "70_30":    (0.70, 0.00, 0.30),
    "60_20_20": (0.60, 0.20, 0.20),
}
SPLITS_FOR = lambda has_val: ("train", "validation", "test") if has_val else ("train", "test")
BAND_SELECTION_METHODS = ["none", "variance", "correlation", "mutual_information", "manual", "hybrid", "topk",
                           "importance"]
BALANCE_METHODS = ["none", "undersample", "oversample"]


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


def select_bands_by_importance(importance, num_bands, importance_threshold, max_bands=None):
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


def save_importance_selection_report(out_dir, importance, ranking, selected_indices,
                                      n_informative, num_bands_requested, importance_threshold,
                                      original_bands, wavelengths, max_bands=None):
    """Writes the Phase 1 metadata bundle: selected_wavelengths.npy,
    band_importance.npy, band_ranking.npy and band_selection_report.json
    (also written as the legacy name selection_report.json, which older
    copies of train_example_v6.py look for)."""
    np.save(os.path.join(out_dir, "band_importance.npy"), importance.astype(np.float32))
    np.save(os.path.join(out_dir, "band_ranking.npy"), ranking.astype(np.int64))
    np.save(os.path.join(out_dir, "selected_band_indices.npy"), selected_indices.astype(np.int64))
    if wavelengths is not None:
        np.save(os.path.join(out_dir, "selected_wavelengths.npy"),
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
                                       and (num_bands is None or num_bands > max_bands)),
        "importance_statistics": {
            "min": float(importance.min()),
            "max": float(importance.max()),
            "mean": float(importance.mean()),
            "median": float(np.median(importance)),
            "std": float(importance.std()),
        },
        "selected_indices": selected_indices.tolist(),
    }
    for fname in ("band_selection_report.json", "selection_report.json"):
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(report, f, indent=2)
    return report


def save_selected_bands(out_dir, method, indices, wavelengths, original_bands,
                         num_bands_requested, sample_fraction):
    """Writes selected_band_indices.npy, selected_wavelengths.npy (if the
    full spectrum's wavelengths are known) and band_selection_config.json."""
    indices = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
    np.save(os.path.join(out_dir, "selected_band_indices.npy"), indices)
    if wavelengths is not None:
        np.save(os.path.join(out_dir, "selected_wavelengths.npy"),
                np.asarray(wavelengths, dtype=np.float32)[indices])
    config = {
        "method": method,
        "original_bands": int(original_bands) if original_bands is not None else None,
        "selected_bands": int(len(indices)),
        "num_bands_requested": num_bands_requested,
        "sample_fraction": sample_fraction,
        "selected_indices": indices.tolist(),
    }
    with open(os.path.join(out_dir, "band_selection_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return indices


def load_selected_bands(out_dir):
    p = os.path.join(out_dir, "selected_band_indices.npy")
    if os.path.exists(p):
        return np.load(p)
    return None


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
# Streaming shard writer - bounds memory to ~one shard at a time
# ============================================================================

class ShardWriter:
    """Buffers patches in RAM only up to `batch_size`, then flushes them to
    a small .npy shard on disk and clears the buffer. Call `finalize()` at
    the end of a capture to flush any remainder."""

    def __init__(self, batches_dir: str, split: str, shard_counter_start: int, batch_size: int):
        self.batches_dir = batches_dir
        self.split = split
        self.batch_size = batch_size
        self.shard_id = shard_counter_start
        self.buf_x, self.buf_y = [], []
        self.written_shards = []  # list of (x_path, y_path, n)
        os.makedirs(os.path.join(batches_dir, split), exist_ok=True)

    def add(self, patch, label):
        self.buf_x.append(patch)
        self.buf_y.append(label)
        if len(self.buf_x) >= self.batch_size:
            self._flush()

    def _flush(self):
        if not self.buf_x:
            return
        name = f"shard_{self.shard_id:06d}"
        x_path = os.path.join(self.batches_dir, self.split, name + "_X.npy")
        y_path = os.path.join(self.batches_dir, self.split, name + "_y.npy")
        # write to temp names first so a kill mid-write never leaves a half
        # -written shard that looks complete
        # np.save silently appends ".npy" to bare string paths that don't
        # already end in it, which would corrupt the ".tmp" rename target -
        # write through an explicit file handle instead to avoid that.
        x_tmp, y_tmp = x_path + ".tmp", y_path + ".tmp"
        with open(x_tmp, "wb") as f:
            np.save(f, np.stack(self.buf_x).astype(np.float32))
        with open(y_tmp, "wb") as f:
            np.save(f, np.array(self.buf_y, dtype=np.int64))
        os.replace(x_tmp, x_path)
        os.replace(y_tmp, y_path)
        self.written_shards.append((x_path, y_path, len(self.buf_x)))
        self.shard_id += 1
        self.buf_x, self.buf_y = [], []
        gc.collect()

    def finalize(self):
        self._flush()
        return self.written_shards, self.shard_id


# ============================================================================
# Resumable progress manifest
# ============================================================================

def manifest_path(out_dir):
    return os.path.join(out_dir, PROGRESS_FILE)


def load_manifest(out_dir):
    p = manifest_path(out_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_manifest(out_dir, manifest):
    p = manifest_path(out_dir)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, p)


def new_manifest(args, class_names, train_patients, val_patients, test_patients, selected_band_indices):
    return {
        "args": {
            "modality": args.modality, "label_source": args.label_source,
            "patch_size": args.patch_size, "stride": args.stride,
            "roi_min_frac": args.roi_min_frac, "split": args.split, "seed": args.seed,
            # Phase 5 - persisted so a resume can't silently mix incompatible settings
            "band_selection": args.band_selection,
            "num_bands": args.num_bands if args.band_selection != "none" else None,
            "importance_threshold": args.importance_threshold if args.band_selection == "importance" else None,
            "max_bands": args.max_bands if args.band_selection == "importance" else None,
            "balance_classes": args.balance_classes,
        },
        "class_names": class_names,
        # Phase 3 - train/validation/test are all patient-level and persisted (validation_patients
        # is [] for two-way split presets)
        "train_patients": sorted(train_patients),
        "validation_patients": sorted(val_patients),
        "test_patients": sorted(test_patients),
        "wavelengths": None,                       # full-spectrum wavelengths, set once known
        "selected_band_indices": selected_band_indices.tolist() if selected_band_indices is not None else None,
        "next_shard_id": {f"{mod}_{split}": 0 for mod in ("hsi", "rgb") for split in ("train", "validation", "test")},
        "completed_captures": {},   # capture_key -> {"split", "hsi": [[x,y,n],...] or None, "rgb": [...] or None}
        "finalized": False,
    }


def check_manifest_compatible(manifest, args):
    saved = manifest["args"]
    current = {
        "modality": args.modality, "label_source": args.label_source,
        "patch_size": args.patch_size, "stride": args.stride,
        "roi_min_frac": args.roi_min_frac, "split": args.split, "seed": args.seed,
        # band_selection changes what's actually written into each patch (Pass 2 slices
        # cube_img[..., selected_band_indices]) and split changes which capture goes into
        # which writer, so both MUST match to safely resume already-written shards.
        # balance_classes only affects the final unify step, but is checked too so a resume
        # can't silently change the eventual balanced dataset without the user noticing.
        "band_selection": args.band_selection,
        "num_bands": args.num_bands if args.band_selection != "none" else None,
        "importance_threshold": args.importance_threshold if args.band_selection == "importance" else None,
        "max_bands": args.max_bands if args.band_selection == "importance" else None,
        "balance_classes": args.balance_classes,
    }
    mismatches = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
    if mismatches:
        raise ValueError(
            f"Found an existing progress file at --out_dir with DIFFERENT settings than this "
            f"run: {mismatches}. Use a fresh --out_dir, delete _progress.json to start over, "
            f"or match the original settings to resume it.")


def cleanup_orphan_shards(out_dir, modality):
    """Deletes any shard files on disk that aren't referenced by the
    manifest - these are leftovers from a capture that was interrupted
    mid-flush before it could be marked completed."""
    manifest = load_manifest(out_dir)
    if manifest is None:
        return
    referenced = set()
    for cap in manifest["completed_captures"].values():
        for mod in ("hsi", "rgb"):
            for entry in (cap.get(mod) or []):
                referenced.add(entry[0])
                referenced.add(entry[1])

    for mod in (["hsi", "rgb"] if modality == "both" else [modality]):
        batches_dir = os.path.join(out_dir, mod, "_batches")
        if not os.path.isdir(batches_dir):
            continue
        for split in ("train", "validation", "test"):
            split_dir = os.path.join(batches_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for fname in os.listdir(split_dir):
                fpath = os.path.join(split_dir, fname)
                if fname.endswith(".tmp") or fpath not in referenced:
                    os.remove(fpath)


# ============================================================================
# Per-capture processing (memory-bounded: one shard's worth of patches at a
# time, patches read directly off the memory-mapped ENVI cube)
# ============================================================================

def process_one_capture(cap_dir, patient, tissue, capture, roi_root, modality,
                         patch_size, stride, roi_min_frac,
                         hsi_writer, rgb_writer, label, selected_band_indices=None):
    """Streams every patch of this single capture straight into the given
    ShardWriter(s). Returns (wavelengths_or_None, n_hsi_patches, n_rgb_patches).

    `selected_band_indices` (Phase 1, optional): if given, each HSI patch is
    sliced down to just those bands before being written to the shard - RGB
    patches are never affected by band selection."""
    wavelengths = None
    cube_img, mask, cube_hw = None, None, None
    white_mean = dark_mean = None

    if hsi_writer is not None:
        cube_path = _find_envi_cube(cap_dir)
        if cube_path is None:
            print(f"    [skip HSI] no .hdr found in {os.path.basename(cap_dir)}")
        else:
            cube_img, wavelengths = open_envi_lazy(cube_path)
            cube_hw = cube_img.shape[:2]

            white_mean, dark_mean = get_calibration_means(cap_dir, cube_path)

            geojson = find_roi_geojson(roi_root, patient, tissue, capture)
            if geojson:
                try:
                    mask = rasterize_roi_mask(geojson, cube_hw[0], cube_hw[1])
                except Exception as e:
                    print(f"    [warn] failed to rasterize ROI: {e}")

    coords = patch_coords(cube_hw[0], cube_hw[1], patch_size, stride, mask, roi_min_frac) if cube_hw else None

    n_hsi = n_rgb = 0

    if hsi_writer is not None and cube_img is not None and coords:
        for (y, x) in coords:
            patch = np.asarray(cube_img[y:y + patch_size, x:x + patch_size, :], dtype=np.float32)
            if white_mean is not None and dark_mean is not None:
                patch = calibrate_patch(patch, white_mean, dark_mean)
            if selected_band_indices is not None:
                patch = patch[:, :, selected_band_indices]
            hsi_writer.add(patch, label)
            n_hsi += 1

    if rgb_writer is not None:
        rgb_path = _find_rgb_image(cap_dir)
        if rgb_path is None:
            print(f"    [skip RGB] no rgb image found in {os.path.basename(cap_dir)}")
        else:
            from PIL import Image
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
            if cube_hw is not None and rgb.shape[:2] != cube_hw:
                rgb = resize_nn(rgb, cube_hw[0], cube_hw[1])
            if coords is None:
                coords = patch_coords(rgb.shape[0], rgb.shape[1], patch_size, stride, None, roi_min_frac)
            for (y, x) in coords:
                rgb_writer.add(rgb[y:y + patch_size, x:x + patch_size].copy(), label)
                n_rgb += 1
            del rgb

    del cube_img
    gc.collect()
    return wavelengths, n_hsi, n_rgb


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


TISSUE_ALIASES = {"idc": "IDC", "healthy": "healthy", "dcis": "DCIS"}


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
                        "patient": m["patient"], "tissue": m["tissue"], "capture": m["capture"]})
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
            patient, tissue, capture = m["patient"], m["tissue"], m["capture"]
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


def resolve_split_fractions(split_preset):
    """Phase 3.2 - CLI preset name -> (train_frac, val_frac, test_frac)."""
    if split_preset not in SPLIT_PRESETS:
        raise ValueError(f"Unknown --split preset {split_preset!r}. Choices: {sorted(SPLIT_PRESETS)}")
    return SPLIT_PRESETS[split_preset]


def patient_split_three(captures, labels, split_preset, seed):
    """Phase 3.3 - splits by PATIENT (never by patch), same as before.

    Deliberately shuffles/slices in the SAME order as the old two-way
    `patient_split()` (shuffle once, take the first N as held-out) so that
    the default `--split 80_20` reproduces the exact same test set as the
    old `--test_size 0.2` for the same --seed. Returns
    (train_patients, val_patients, test_patients) as sets; val_patients is
    empty for two-way presets.
    """
    train_frac, val_frac, test_frac = resolve_split_fractions(split_preset)
    patients = sorted({c["patient"] for c in captures if c["key"] in labels})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_test = max(1, int(round(len(patients) * test_frac)))
    test_patients = patients[:n_test]
    remaining = patients[n_test:]

    if val_frac > 0:
        n_val = max(1, int(round(len(patients) * val_frac))) if remaining else 0
        n_val = min(n_val, len(remaining))
        val_patients = remaining[:n_val]
        train_patients = remaining[n_val:]
    else:
        val_patients = []
        train_patients = remaining

    return set(train_patients), set(val_patients), set(test_patients)


# ============================================================================
# Phase 2 - Class Balancing Utilities (applied to TRAIN only, at unify time)
# ============================================================================

def _class_counts(y, class_names):
    counts = {}
    for c in range(len(class_names)):
        counts[class_names[c]] = int((y == c).sum())
    return counts


def build_balance_selection(y_all, method, seed):
    """Returns an array of GLOBAL row indices into y_all to keep (with
    repeats for oversampling), shuffled, plus per-class before/after counts.
    Deterministic: always uses np.random.default_rng(seed) (Phase 2.6)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_all)
    class_indices = {int(c): np.flatnonzero(y_all == c) for c in classes}
    before = {int(c): len(idx) for c, idx in class_indices.items()}

    if method == "undersample":
        target = min(before.values())
        selected = []
        for c, idx in class_indices.items():
            take = rng.choice(idx, size=target, replace=False) if len(idx) > target else idx
            selected.append(take)
    elif method == "oversample":
        target = max(before.values())
        selected = []
        for c, idx in class_indices.items():
            if len(idx) < target:
                extra = rng.choice(idx, size=target - len(idx), replace=True)
                take = np.concatenate([idx, extra])
            else:
                take = idx
            selected.append(take)
    else:
        raise ValueError(f"Unknown balance method: {method}")

    selected = np.concatenate(selected)
    rng.shuffle(selected)
    after = {int(c): int((y_all[selected] == c).sum()) for c in classes}
    return selected, before, after


# ============================================================================
# Unify: stitch every shard for a split into one final .npy, memmap-to-
# -memmap so the full unified array is never held in RAM at once
# ============================================================================

def unify_split(shard_entries, out_x_path, out_y_path, balance_method=None, seed=0, class_names=None):
    """shard_entries: list of (x_path, y_path, n). Writes out_x_path/out_y_path.
    If balance_method is set ("undersample"/"oversample"), the written split
    is resampled per-class (Phase 2) - intended for the TRAIN split only.
    Returns (n_written, balance_report_or_None)."""
    if not shard_entries:
        return 0, None
    total_n = sum(n for _, _, n in shard_entries)
    sample = np.load(shard_entries[0][0], mmap_mode="r")
    patch_shape = sample.shape[1:]
    dtype = sample.dtype
    del sample

    # First pass: gather all labels (cheap - labels are tiny) and build a
    # global-row -> (shard_index, local_index) lookup via cumulative offsets.
    all_y = np.empty((total_n,), dtype=np.int64)
    cum_offsets = [0]
    offset = 0
    for x_path, y_path, n in shard_entries:
        all_y[offset:offset + n] = np.load(y_path)
        offset += n
        cum_offsets.append(offset)

    balance_report = None
    if balance_method and balance_method != "none":
        selected, before, after = build_balance_selection(all_y, balance_method, seed)
        if class_names:
            before = {class_names[c]: n for c, n in before.items()}
            after = {class_names[c]: n for c, n in after.items()}
        balance_report = {"method": balance_method, "before": before, "after": after}
        row_order = selected  # may repeat rows (oversampling)
    else:
        row_order = np.arange(total_n)

    out_n = len(row_order)
    out_x = np.lib.format.open_memmap(out_x_path, mode="w+", dtype=dtype, shape=(out_n,) + patch_shape)
    out_y = np.empty((out_n,), dtype=np.int64)

    # Group by shard so each shard is opened once and read sequentially where possible.
    shard_mmaps = {}

    def get_shard_mmap(shard_idx):
        if shard_idx not in shard_mmaps:
            shard_mmaps[shard_idx] = np.load(shard_entries[shard_idx][0], mmap_mode="r")
        return shard_mmaps[shard_idx]

    for out_row, global_row in enumerate(row_order):
        shard_idx = bisect.bisect_right(cum_offsets, global_row) - 1
        local_idx = int(global_row) - cum_offsets[shard_idx]
        out_x[out_row] = get_shard_mmap(shard_idx)[local_idx]
        out_y[out_row] = all_y[global_row]

    out_x.flush()
    del out_x, shard_mmaps
    np.save(out_y_path, out_y)
    return out_n, balance_report


def unify_all(out_dir, modality, manifest, delete_batches, balance_method, seed):
    """Returns a stats dict (per-modality patch counts + balance reports)
    used to build dataset_statistics.json - or None if already finalized."""
    if manifest.get("finalized"):
        print("  Already finalized in a previous run (shards were deleted after unifying); "
              "the X/y .npy files in --out_dir are the final output. Delete _progress.json "
              "and rerun with --fresh if you want to redo everything from scratch.")
        return None

    wavelengths = manifest.get("wavelengths")
    selected_band_indices = manifest.get("selected_band_indices")
    has_val = bool(manifest.get("validation_patients"))
    splits = SPLITS_FOR(has_val)
    class_names = manifest.get("class_names")

    stats = {"modalities": {}}

    for mod in (["hsi", "rgb"] if modality == "both" else [modality]):
        shards_by_split = {s: [] for s in splits}
        for cap in manifest["completed_captures"].values():
            entries = cap.get(mod)
            if not entries:
                continue
            split = cap["split"]
            if split not in shards_by_split:
                continue  # e.g. rgb has no validation shards in some edge case
            shards_by_split[split].extend(entries)

        if not any(shards_by_split.values()):
            print(f"  [{mod}] no completed shards found - nothing to unify")
            continue

        sub = os.path.join(out_dir, mod)
        os.makedirs(sub, exist_ok=True)

        mod_stats = {"patches": {}, "balance": None}
        for split in splits:
            out_name = "val" if split == "validation" else split
            x_path = os.path.join(sub, f"X_{out_name}.npy")
            y_path = os.path.join(sub, f"y_{out_name}.npy")
            # Phase 2.7 - balancing is applied to TRAIN only; validation/test are untouched.
            bm = balance_method if split == "train" else None
            n_written, report = unify_split(shards_by_split[split], x_path, y_path,
                                              balance_method=bm, seed=seed, class_names=class_names)
            mod_stats["patches"][split] = n_written
            if report is not None:
                mod_stats["balance"] = report

        # Phase 1.6 - wavelengths.npy must match the band count actually written to X_*.npy.
        if mod == "hsi" and wavelengths is not None:
            wl = np.asarray(wavelengths, dtype=np.float32)
            if selected_band_indices:
                wl_full = wl
                wl = wl[np.asarray(selected_band_indices, dtype=np.int64)]
                np.save(os.path.join(sub, "wavelengths_full.npy"), wl_full)
            np.save(os.path.join(sub, "wavelengths.npy"), wl)

        split_summary = " ".join(f"{s}={mod_stats['patches'].get(s, 0)}" for s in splits)
        print(f"  [{mod}] unified: {split_summary} patches -> {sub}/")
        stats["modalities"][mod] = mod_stats

        if delete_batches:
            batches_dir = os.path.join(sub, "_batches")
            if os.path.isdir(batches_dir):
                for split in splits:
                    split_dir = os.path.join(batches_dir, split)
                    if os.path.isdir(split_dir):
                        for f in os.listdir(split_dir):
                            os.remove(os.path.join(split_dir, f))
                        os.rmdir(split_dir)
                if not os.listdir(batches_dir):
                    os.rmdir(batches_dir)

    return stats


def write_dataset_statistics(out_dir, manifest, unify_stats, args):
    """Phase 4 - dataset_statistics.json, combining band selection, class
    balancing, split sizes, and general preprocessing metadata."""
    has_val = bool(manifest.get("validation_patients"))
    train_frac, val_frac, test_frac = resolve_split_fractions(manifest["args"]["split"])

    stats = {
        "band_selection": {
            "method": manifest["args"]["band_selection"],
            "num_bands_requested": manifest["args"].get("num_bands"),
            "importance_threshold": manifest["args"].get("importance_threshold"),
            "original_bands": len(manifest["wavelengths"]) if manifest.get("wavelengths") else None,
            "selected_bands": len(manifest["selected_band_indices"]) if manifest.get("selected_band_indices") else None,
        },
        "class_balancing": {
            "method": manifest["args"]["balance_classes"],
            "per_modality": {mod: v.get("balance") for mod, v in (unify_stats or {}).get("modalities", {}).items()},
        },
        "split": {
            "preset": manifest["args"]["split"],
            "train_fraction": train_frac, "validation_fraction": val_frac, "test_fraction": test_frac,
            "train_patients": len(manifest["train_patients"]),
            "validation_patients": len(manifest["validation_patients"]) if has_val else 0,
            "test_patients": len(manifest["test_patients"]),
            "patch_counts_per_modality": {
                mod: v.get("patches") for mod, v in (unify_stats or {}).get("modalities", {}).items()
            },
        },
        "class_names": manifest.get("class_names"),
        "patch_size": manifest["args"]["patch_size"],
        "stride": manifest["args"]["stride"],
        "roi_min_frac": manifest["args"]["roi_min_frac"],
        "seed": manifest["args"]["seed"],
        "modality": manifest["args"]["modality"],
        "label_source": manifest["args"]["label_source"],
    }
    with open(os.path.join(out_dir, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Main Development Plan, Phase 2 - class_distribution_before.png,
    # class_distribution_after.png, balancing_report.json. Only meaningful
    # when balancing actually ran (unify_stats carries a per-modality
    # "balance" dict with {method, before, after} - see unify_split()).
    _write_balancing_outputs(out_dir, unify_stats, manifest.get("class_names"))

    return stats


def _write_balancing_outputs(out_dir, unify_stats, class_names):
    """Writes balancing_report.json plus class_distribution_before.png /
    class_distribution_after.png (train split only - balancing never
    touches validation/test, per Phase 2)."""
    if not unify_stats:
        return
    modalities = unify_stats.get("modalities", {})
    balance_reports = {mod: v.get("balance") for mod, v in modalities.items() if v.get("balance")}
    if not balance_reports:
        return

    with open(os.path.join(out_dir, "balancing_report.json"), "w") as f:
        json.dump(balance_reports, f, indent=2)

    try:
        from training.plots import plot_class_distribution
    except Exception as e:
        print(f"  [class-balancing] could not import training.plots for distribution plots: {e}")
        return

    # Plot the first modality that has a balance report (train split is
    # balanced identically in spirit across hsi/rgb when --modality both).
    mod, report = next(iter(balance_reports.items()))
    before, after = report["before"], report["after"]
    names = sorted(before.keys())
    before_counts = [before[n] for n in names]
    after_counts = [after[n] for n in names]
    plot_class_distribution(before_counts, names, os.path.join(out_dir, "class_distribution_before.png"),
                             title=f"Class distribution before balancing ({mod}, train split)")
    plot_class_distribution(after_counts, names, os.path.join(out_dir, "class_distribution_after.png"),
                             title=f"Class distribution after balancing ({report['method']}, {mod}, train split)")
    print(f"  [class-balancing] wrote balancing_report.json, class_distribution_before.png, "
          f"class_distribution_after.png -> {out_dir}/")


def _run_dataset_integrity_check(out_dir, manifest):
    """Main Development Plan, Phase 7 - runs the patient/slide-level leakage
    check (patch-level hashing is comparatively expensive and is instead run
    on-demand at training start via training.dataset_integrity, guarded by
    --check_leakage / --abort_on_leakage in train_example_v6.py). Never
    aborts dataset *preparation* - this is informational at prep time so the
    dataset_split_report.json / leakage_report.json exist for training to
    read; training itself is what aborts on leakage detected."""
    try:
        from training.dataset_integrity import run_full_integrity_check
    except Exception as e:
        print(f"  [dataset-integrity] could not import training.dataset_integrity: {e}")
        return
    manifest_path = os.path.join(out_dir, PROGRESS_FILE)
    if not os.path.isfile(manifest_path):
        return
    try:
        run_full_integrity_check(out_dir, manifest_path=manifest_path, npy_data_dir=None,
                                  abort_on_leakage=False)
    except Exception as e:
        print(f"  [dataset-integrity] check failed: {e}")


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="top-level HistologyHSI-BC-Recurrence download directory")
    p.add_argument("--out_dir", default="./data")
    p.add_argument("--modality", choices=["hsi", "rgb", "both"], default="both")
    p.add_argument("--label_source", choices=["tissue", "recurrence"], default="tissue")
    p.add_argument("--clinical_xlsx", default=None,
                    help="required if --label_source recurrence; defaults to "
                         "<root>/HistologyHSI-BC-Recurrence-Clinical-Standardized.xlsx")
    p.add_argument("--patch_size", type=int, default=11)
    p.add_argument("--stride", type=int, default=11)
    p.add_argument("--roi_min_frac", type=float, default=0.8,
                    help="minimum fraction of a patch that must fall inside the annotated ROI "
                         "to be kept (only applies when ROI geojsons are found)")
    # Phase 3 - flexible dataset splits (replaces --test_size)
    p.add_argument("--split", choices=sorted(SPLIT_PRESETS), default="80_20",
                    help="patient-level split preset. '80_20'/'70_30' are two-way (no validation "
                         "set); the others are three-way. Default '80_20' reproduces the old "
                         "--test_size 0.2 default exactly.")
    # Phase 3 (v5) - class-stratified split, fixing a real bug where the old
    # unstratified patient shuffle could leave validation/test with zero
    # samples of a class that only had a few distinct patients.
    p.add_argument("--split_strategy", choices=["stratified", "legacy_random"], default="stratified",
                    help="'stratified' (default, v5): guarantees every class with enough distinct "
                         "patients appears in every split (Phase 3). 'legacy_random': the old v4 "
                         "behavior (plain shuffle-and-slice, no per-class guarantee) - use only for "
                         "reproducing a v4-generated dataset byte-for-byte with the same --seed.")
    p.add_argument("--allow_missing_classes", action="store_true",
                    help="warn instead of aborting when the post-split class-coverage check (Phase 3) "
                         "finds a split with zero samples of a required class. Without this flag, an "
                         "uncoverable class aborts the run immediately, before any patches are "
                         "extracted, rather than after (which is what used to happen when this was "
                         "only caught by train_example_v11.py's own coverage check).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256,
                    help="patches per shard file - lower this if you're still short on memory")
    p.add_argument("--keep_batches", action="store_true",
                    help="don't delete the per-capture shard files after unifying")
    p.add_argument("--fresh", action="store_true",
                    help="ignore/overwrite any existing progress file and start over")
    p.add_argument("--finalize_only", action="store_true",
                    help="skip processing entirely and just (re-)run the unify step on whatever "
                         "shards already exist in --out_dir")
    # Phase 1 - spectral band selection (HSI only; defaults preserve old behaviour)
    p.add_argument("--band_selection", choices=BAND_SELECTION_METHODS, default="none",
                    help="reduce the spectral dimension before patches are written. 'manual' "
                         "loads indices from --band_file; the rest run a Pass-1 sample scan.")
    p.add_argument("--num_bands", type=int, default=None,
                    help="maximum number of bands to keep. For --band_selection importance this "
                         "is an optional CAP on top of --importance_threshold - never invented, "
                         "just truncated to the top-N most important bands (default: keep every "
                         "band that clears the threshold, or every band if no threshold either). "
                         "For variance/mutual_information/hybrid/topk it's the target count (and "
                         "a cap for correlation); those methods fall back to 64 with a warning if "
                         "left unset, to preserve old behaviour.")
    p.add_argument("--importance_threshold", type=float, default=None,
                    help="[--band_selection importance] minimum normalized (0-1) importance score "
                         "a wavelength must reach to be kept. Not mandatory - if omitted, no "
                         "wavelength is dropped on importance grounds and --num_bands (if given) "
                         "simply takes the top-N bands overall. Passing --num_bands and/or "
                         "--importance_threshold without --band_selection auto-selects "
                         "'importance'.")
    p.add_argument("--max_bands", type=int, default=None,
                    help="[Main Development Plan, Phase 1] hard upper bound on the number of bands "
                         "--band_selection importance is allowed to select, applied AFTER "
                         "--importance_threshold and --num_bands (whichever combination was given). "
                         "Purpose: prevent an excessively large number of wavelengths during "
                         "automated ablation experiments. The cap is never exceeded, but is also "
                         "never used to invent extra bands beyond what --num_bands/--importance_threshold "
                         "selected. Example: original_bands=275, threshold selects 210, num_bands=None, "
                         "max_bands=128 -> final output is the 128 highest-ranked bands. If instead the "
                         "threshold only selects 61, max_bands=128 has no effect and the output is 61 bands.")
    p.add_argument("--band_file", default="selected_bands.npy",
                    help="for --band_selection manual: path to a .npy of band indices to load. "
                         "For other methods this filename is informational only - the actual "
                         "output is always <out_dir>/selected_band_indices.npy")
    p.add_argument("--band_selection_sample_fraction", type=float, default=0.10,
                    help="fraction of captures Pass 1 opens to compute band statistics from")
    p.add_argument("--band_corr_threshold", type=float, default=0.98,
                    help="correlation above which a band is pruned as redundant "
                         "(used by --band_selection correlation/hybrid/topk)")
    # Phase 2 - class balancing (train split only, applied at the final unify step)
    p.add_argument("--balance_classes", choices=BALANCE_METHODS, default="none",
                    help="resample the TRAIN split to equalize class counts. Validation/test "
                         "are never touched.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.finalize_only:
        manifest = load_manifest(args.out_dir)
        if manifest is None:
            raise RuntimeError(f"No {PROGRESS_FILE} found in {args.out_dir} - nothing to finalize")
        print("Finalizing (unifying shards) only ...")
        unify_stats = unify_all(args.out_dir, manifest["args"]["modality"], manifest,
                                 delete_batches=not args.keep_batches,
                                 balance_method=manifest["args"]["balance_classes"],
                                 seed=manifest["args"]["seed"])
        manifest["finalized"] = True
        save_manifest(args.out_dir, manifest)
        if unify_stats is not None:
            write_dataset_statistics(args.out_dir, manifest, unify_stats, args)
        _run_dataset_integrity_check(args.out_dir, manifest)
        print("Done.")
        return

    if not args.root:
        raise SystemExit("--root is required unless --finalize_only is passed")

    clinical_xlsx = args.clinical_xlsx or os.path.join(
        args.root, "HistologyHSI-BC-Recurrence-Clinical-Standardized.xlsx")
    roi_root = _find_roi_root(args.root)
    if roi_root is None:
        print("  [note] no ROI-annotations folder found - proceeding without ROI masking "
              "(patches will be taken from the full capture instead of just the annotated region)")

    if args.band_selection == "manual" and not os.path.exists(args.band_file):
        raise SystemExit(f"--band_selection manual requires --band_file to point at an existing "
                          f".npy of band indices; {args.band_file!r} not found")

    # Phase 1 - passing --num_bands and/or --importance_threshold without an explicit
    # --band_selection auto-activates the "importance" method (neither flag is mandatory
    # on its own; either one is enough to turn selection on).
    if args.band_selection == "none" and (args.num_bands is not None or args.importance_threshold is not None):
        args.band_selection = "importance"
        print("  [band-selection] --num_bands/--importance_threshold given without "
              "--band_selection - defaulting to --band_selection importance")

    # legacy fixed-count methods still need a target count; preserve the old default (64)
    # for them specifically now that --num_bands itself defaults to None (Phase 1 item 3).
    if args.band_selection in ("variance", "correlation", "mutual_information", "hybrid", "topk") \
            and args.num_bands is None:
        print(f"  [band-selection] --band_selection {args.band_selection} needs a target band "
              f"count; --num_bands not given, defaulting to 64")
        args.num_bands = 64

    print(f"Scanning {args.root} ...")
    captures = discover_captures(args.root)
    labels, class_names = build_labels(captures, args.label_source, clinical_xlsx)
    captures = [c for c in captures if c["key"] in labels]
    print(f"Found {len(captures)} labeled captures across "
          f"{len(sorted({c['patient'] for c in captures}))} patients. Classes: {class_names}")

    manifest = None if args.fresh else load_manifest(args.out_dir)
    if manifest is not None:
        check_manifest_compatible(manifest, args)
        train_patients = set(manifest["train_patients"])
        val_patients = set(manifest["validation_patients"])
        test_patients = set(manifest["test_patients"])
        selected_band_indices = (np.asarray(manifest["selected_band_indices"], dtype=np.int64)
                                  if manifest.get("selected_band_indices") else None)
        n_done = len(manifest["completed_captures"])
        print(f"Resuming from existing progress file: {n_done} capture(s) already completed and will be skipped.")
    else:
        # Phase 3 fix (v5): the OLD patient_split_three() below shuffles ALL
        # patients globally and slices, with no per-class guarantee - with a
        # class that has few distinct patients, that CAN (and, per the bug
        # report this version was written for, DID) put every patient of a
        # class into train, leaving validation/test with zero samples of it.
        # stratified_patient_split_three() guarantees every class with enough
        # distinct patients appears in every split; --split_strategy
        # legacy_random restores the old (unstratified) behavior if you need
        # byte-for-byte reproducibility with a v4-generated dataset.
        if args.split_strategy == "legacy_random":
            train_patients, val_patients, test_patients = patient_split_three(
                captures, labels, args.split, args.seed)
            split_warnings = []
        else:
            patient_to_classes = defaultdict(set)
            for c in captures:
                if c["key"] in labels:
                    patient_to_classes[c["patient"]].add(labels[c["key"]])
            train_patients, val_patients, test_patients, split_warnings = stratified_patient_split_three(
                dict(patient_to_classes), args.split, args.seed)

        for w in split_warnings:
            print(f"  [split-warning] {w}")

        # Phase 3 - class coverage MUST be checked before any patch extraction
        # happens (not just at train time, where this was previously only
        # discovered after the fact). Uses the exact same check
        # train_example_v11.py runs, applied to capture-level labels.
        def _assign_split_preview(patient):
            if patient in test_patients:
                return "test"
            if patient in val_patients:
                return "validation"
            return "train"

        item_labels = [(labels[c["key"]], c["patient"]) for c in captures if c["key"] in labels]
        coverage_report = check_prep_class_coverage(
            item_labels, _assign_split_preview, num_classes=len(class_names), class_names=class_names,
            fail_hard=not args.allow_missing_classes)
        with open(os.path.join(args.out_dir, "class_coverage_report.json"), "w") as f:
            json.dump(coverage_report, f, indent=2)
        if not coverage_report["coverage_ok"]:
            print(f"  [WARNING] class coverage check found missing classes: {coverage_report['missing']} "
                  f"(continuing because --allow_missing_classes was passed - training scripts will need "
                  f"--allow_missing_classes too, or they'll refuse this dataset for the same reason)")
        else:
            print("  [class-coverage] every class present in every split - OK")


        # Phase 1 - Pass 1: determine selected_band_indices ONCE, before any patches are
        # extracted, so Pass 2 (the capture loop below) can slice every HSI patch down to
        # just those bands as it writes it to a shard.
        selected_band_indices = None
        if args.band_selection != "none" and args.modality in ("hsi", "both"):
            if args.band_selection == "manual":
                manual_idx = np.load(args.band_file)
                print(f"  [band-selection] manual: loaded {len(manual_idx)} band indices from {args.band_file}")
                # wavelengths aren't known yet (no cube opened) - save the config now and
                # fill in selected_wavelengths.npy once the first cube is opened, at unify time.
                selected_band_indices = save_selected_bands(
                    args.out_dir, "manual", manual_idx, wavelengths=None,
                    original_bands=None, num_bands_requested=len(manual_idx),
                    sample_fraction=None)
            elif args.band_selection == "importance":
                print(f"  [band-selection] running Pass 1 (importance) ...")
                pixels, pixel_labels, wl = compute_band_statistics(
                    captures, roi_root, labels, args.band_selection_sample_fraction, args.seed)
                importance = compute_band_importance(pixels, pixel_labels, seed=args.seed)
                selected_band_indices, ranking, n_informative = select_bands_by_importance(
                    importance, args.num_bands, args.importance_threshold, max_bands=args.max_bands)
                save_importance_selection_report(
                    args.out_dir, importance, ranking, selected_band_indices, n_informative,
                    num_bands_requested=args.num_bands, importance_threshold=args.importance_threshold,
                    original_bands=pixels.shape[1], wavelengths=wl, max_bands=args.max_bands)
                print(f"  [band-selection] selected {len(selected_band_indices)}/{pixels.shape[1]} bands "
                      f"({n_informative} cleared the threshold) -> {args.out_dir}/selected_band_indices.npy")
                del pixels, pixel_labels
            else:
                print(f"  [band-selection] running Pass 1 ({args.band_selection}) ...")
                pixels, pixel_labels, wl = compute_band_statistics(
                    captures, roi_root, labels, args.band_selection_sample_fraction, args.seed)
                idx = select_bands(args.band_selection, pixels, pixel_labels,
                                    args.num_bands, args.band_corr_threshold, args.seed)
                selected_band_indices = save_selected_bands(
                    args.out_dir, args.band_selection, idx, wavelengths=wl,
                    original_bands=pixels.shape[1], num_bands_requested=args.num_bands,
                    sample_fraction=args.band_selection_sample_fraction)
                print(f"  [band-selection] selected {len(selected_band_indices)}/{pixels.shape[1]} bands "
                      f"-> {args.out_dir}/selected_band_indices.npy")
                del pixels, pixel_labels

        manifest = new_manifest(args, class_names, train_patients, val_patients, test_patients,
                                 selected_band_indices)
        msg = f"Starting fresh. Train={len(train_patients)}"
        if val_patients:
            msg += f" Validation={len(val_patients)}"
        msg += f" Test={len(test_patients)} patients (split={args.split})."
        print(msg)
        print(f"  Held-out test patients: {sorted(test_patients)}")

    for mod in (["hsi", "rgb"] if args.modality == "both" else [args.modality]):
        cleanup_orphan_shards(args.out_dir, mod)

    def assign_split(patient):
        if patient in test_patients:
            return "test"
        if patient in val_patients:
            return "validation"
        return "train"

    n_printed = 0
    for cap in captures:
        if cap["key"] in manifest["completed_captures"]:
            continue  # already fully processed in a previous run

        split = assign_split(cap["patient"])
        label = labels[cap["key"]]

        hsi_writer = rgb_writer = None
        if args.modality in ("hsi", "both"):
            hsi_writer = ShardWriter(os.path.join(args.out_dir, "hsi", "_batches"), split,
                                      manifest["next_shard_id"][f"hsi_{split}"], args.batch_size)
        if args.modality in ("rgb", "both"):
            rgb_writer = ShardWriter(os.path.join(args.out_dir, "rgb", "_batches"), split,
                                      manifest["next_shard_id"][f"rgb_{split}"], args.batch_size)

        wavelengths, n_hsi, n_rgb = process_one_capture(
            cap["dir"], cap["patient"], cap["tissue"], cap["capture"], roi_root, args.modality,
            args.patch_size, args.stride, args.roi_min_frac, hsi_writer, rgb_writer, label,
            selected_band_indices=selected_band_indices)

        entry = {"split": split, "hsi": None, "rgb": None}
        if hsi_writer is not None:
            shards, next_id = hsi_writer.finalize()
            entry["hsi"] = shards
            manifest["next_shard_id"][f"hsi_{split}"] = next_id
        if rgb_writer is not None:
            shards, next_id = rgb_writer.finalize()
            entry["rgb"] = shards
            manifest["next_shard_id"][f"rgb_{split}"] = next_id

        if wavelengths is not None and manifest.get("wavelengths") is None:
            manifest["wavelengths"] = wavelengths.tolist()
            # manual band selection didn't know wavelengths at selection time - fill them in now
            if manifest["args"]["band_selection"] == "manual" and selected_band_indices is not None:
                save_selected_bands(args.out_dir, "manual", selected_band_indices, wavelengths=wavelengths,
                                     original_bands=len(wavelengths),
                                     num_bands_requested=len(selected_band_indices), sample_fraction=None)

        manifest["completed_captures"][cap["key"]] = entry
        save_manifest(args.out_dir, manifest)  # persist after EVERY capture

        if n_printed < 5:
            print(f"  [{cap['key']}] hsi_patches={n_hsi} rgb_patches={n_rgb} (split={split})")
            n_printed += 1
        elif n_printed == 5:
            print("  ... (further per-capture logs suppressed; see _progress.json for full state)")
            n_printed += 1

    print(f"All {len(captures)} captures processed. Unifying shards into final arrays ...")
    unify_stats = unify_all(args.out_dir, args.modality, manifest, delete_batches=not args.keep_batches,
                             balance_method=args.balance_classes, seed=args.seed)
    manifest["finalized"] = True
    save_manifest(args.out_dir, manifest)
    if unify_stats is not None:
        write_dataset_statistics(args.out_dir, manifest, unify_stats, args)
        print(f"  Wrote {args.out_dir}/dataset_statistics.json")
    _run_dataset_integrity_check(args.out_dir, manifest)
    print("Done.")


if __name__ == "__main__":
    main()
