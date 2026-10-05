# -*- coding: utf-8 -*-
"""
prepare_histologyhsi_bc_v8.py
================================
GMedMamba v16 plan, Stage 1.5 (A-1 at the source) - gain-corrected
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
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import prepare_histologyhsi_bc_v6 as _v6                       # frozen, imported not edited
from prepare_histologyhsi_bc_v7 import HistologyHsiAdapter, _FLOAT16_SAFE_MAX  # frozen, imported not edited
from training.prep import ModalitySpec, run_prep
from training.prep_progress_v16 import install as install_prep_progress
from training.npy_atomic import save_npy_atomic, save_json_atomic
from training.progress_v16 import ProgressReporter

_DEFAULT_GAIN_CLIP = (0.5, 2.0)


def _sample_capture_median(cap_dir: str, roi_root, patient, tissue, capture, roi_min_frac: float,
                            sel: Optional[np.ndarray], seed: int, n_pixels: int = 400) -> Optional[float]:
    """Median intensity of up to `n_pixels` random (ROI-respecting when
    available) pixels from one capture, AFTER the same flat-field
    calibration `_hsi_stream` applies (so the gain this produces composes
    correctly with it), and after band selection if `sel` is given."""
    cube_path = _v6._find_envi_cube(cap_dir)
    if cube_path is None:
        return None
    cube_img, _wl = _v6.open_envi_lazy(cube_path)
    h, w = cube_img.shape[:2]
    white, dark = _v6.get_calibration_means(cap_dir, cube_path)

    mask = None
    gj = _v6.find_roi_geojson(roi_root, patient, tissue, capture) if roi_root else None
    if gj:
        try:
            mask = _v6.rasterize_roi_mask(gj, h, w)
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
            px = _v6.calibrate_patch(px, white, dark)
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
    """A deterministic replacement for `_v6._find_rgb_image`.

    Returns v6's own function for 'legacy'. Otherwise ranks by an explicit
    preference list and breaks every remaining tie by SORTED FILENAME, so the
    result never depends on directory iteration order.
    """
    if mode == "legacy":
        return _v6._find_rgb_image
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
        # `_v6._find_rgb_image` is looked up as a module ATTRIBUTE by
        # `prepare_histologyhsi_bc_v7._rgb_stream` (v7:285), so rebinding the
        # attribute is enough - the frozen file itself is not edited.
        _v6._find_rgb_image = make_rgb_finder(getattr(args, "rgb_source", "synthetic"))
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
                from prepare_histologyhsi_bc_v6 import CAPTURE_RE
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
