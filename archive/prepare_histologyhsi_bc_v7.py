# -*- coding: utf-8 -*-
"""
prepare_histologyhsi_bc_v7.py
=============================
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

Usage::

    python prepare_histologyhsi_bc_v7.py --root /path/to/HistologyHSI --out_dir ./data_hsi \\
        --band_selection importance --num_bands 32 --num_workers 8
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import prepare_histologyhsi_bc_v6 as _v6
from training.prep import DatasetAdapter, ModalitySpec, PrepItem, run_prep

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
        args.roi_root = _v6._find_roi_root(args.root) if args.root else None
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
        caps = _v6.discover_captures(args.root)
        labels, _names = _v6.build_labels(caps, args.label_source, args.clinical_xlsx)
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
                pixels, pix_labels, wl = _v6.compute_band_statistics(
                    caps, args.roi_root, labels, args.band_selection_sample_fraction, args.seed)
                state["wavelengths_full"] = None if wl is None else np.asarray(wl, np.float32).tolist()
                state["hsi_value_max"] = float(np.abs(pixels).max()) if pixels.size else 1.5
                if need_stats:
                    if args.band_selection == "importance":
                        imp = _v6.compute_band_importance(pixels, pix_labels, seed=args.seed)
                        max_corr = (None if getattr(args, "band_max_corr", 1.0) >= 1.0
                                    else args.band_max_corr)
                        sel, ranking, n_inf = _v6.select_bands_by_importance(
                            imp, args.num_bands, args.importance_threshold, args.max_bands,
                            pixels=pixels, max_corr=max_corr,
                            min_gap=getattr(args, "band_min_gap", 0))
                        state["selected_band_indices"] = np.sort(sel).astype(np.int64).tolist()
                        state["_importance"] = (imp, ranking, n_inf, pixels.shape[1])
                    else:
                        idx = _v6.select_bands(args.band_selection, pixels, pix_labels,
                                                args.num_bands, args.band_corr_threshold, args.seed)
                        state["selected_band_indices"] = np.asarray(
                            sorted(int(i) for i in idx), dtype=np.int64).tolist()
            except Exception as e:  # band-count mismatch etc. -> fail loud only if we needed it
                if need_stats:
                    raise
                print(f"  [pass1] cube probe failed ({e}); wavelengths.npy may be absent")

        # calibration probe (v6 semantics: uncalibrated cubes are stored raw)
        for c in caps[: max(3, math.ceil(len(caps) * args.band_selection_sample_fraction))]:
            cube = _v6._find_envi_cube(c["dir"])
            if cube is None:
                continue
            w, d = _v6.get_calibration_means(c["dir"], cube)
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
            _v6.save_importance_selection_report(
                out_dir, np.asarray(imp), np.asarray(ranking), idx, n_inf,
                a.num_bands, a.importance_threshold, orig, wl, max_bands=a.max_bands,
                min_coverage=min_cov, allow_narrow=allow_narrow,
                max_corr=max_corr, min_gap=getattr(a, "band_min_gap", 0))
        else:
            _v6.save_selected_bands(out_dir, a.band_selection, idx, wavelengths=wl,
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
            cube_path = _v6._find_envi_cube(cap_dir)
            if cube_path is not None:
                cube_img, _wl = _v6.open_envi_lazy(cube_path)
                cube_hw = cube_img.shape[:2]
                white, dark = _v6.get_calibration_means(cap_dir, cube_path)
                mask = None
                gj = _v6.find_roi_geojson(args.roi_root, item.payload["patient"],
                                           item.payload["tissue"], item.payload["capture"])
                if gj:
                    try:
                        mask = _v6.rasterize_roi_mask(gj, cube_hw[0], cube_hw[1])
                    except Exception as e:  # noqa: BLE001
                        print(f"    [warn] ROI rasterize failed for {item.key}: {e}")
                coords = _v6.patch_coords(cube_hw[0], cube_hw[1], ps, step, mask, args.roi_min_frac)

        def _hsi_stream():
            if cube_img is None or not coords:
                return
            for (y, x) in coords:
                patch = np.asarray(cube_img[y:y + ps, x:x + ps, :], dtype=np.float32)
                if white is not None and dark is not None:
                    patch = _v6.calibrate_patch(patch, white, dark)
                if sel is not None:
                    patch = patch[:, :, sel]
                yield patch, item.label

        def _rgb_stream():
            rgb_path = _v6._find_rgb_image(cap_dir)
            if rgb_path is None:
                return
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
            if cube_hw is not None and rgb.shape[:2] != tuple(cube_hw):
                rgb = _v6.resize_nn(rgb, cube_hw[0], cube_hw[1])
            local_coords = coords
            if local_coords is None:
                local_coords = _v6.patch_coords(rgb.shape[0], rgb.shape[1], ps, step, None,
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


def main() -> int:
    return run_prep(HistologyHsiAdapter())


if __name__ == "__main__":
    raise SystemExit(main())
