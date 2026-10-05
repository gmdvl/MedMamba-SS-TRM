#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/make_band_subset_build.py
=======================================
v18 E10 - a real dataset directory at a reduced band count, by SLICING the
existing 32-band build rather than re-extracting from the raw cubes.

Why slice instead of re-prepare
-------------------------------
`documentations/12_commands_datasets.md` section G specifies the band sweep as
four fresh `prepare_histologyhsi_bc_v8.py --num_bands N` passes: ~1.8 h each,
~43 GB, and each one re-selects the best N bands out of all 740. It then adds,
without providing the mechanism: "If disk is tight, prep only --num_bands 32
and subset channels at training time instead." There is no flag that does that
- no `--band_subset`, `--keep_bands` or equivalent anywhere in the v15/v16
parser or in `training/npy_data.py`. This script is that mechanism,
moved to preparation time so the training entry point needs no change at all.

Re-selection and decimation answer DIFFERENT questions, and this script does
decimation on purpose:

  re-selection  "what are the best 16 bands out of 740?"  - a sensor-design
                question, more deployable, and future work.
  decimation    "what happens when you remove bands from the set this model
                was trained on?"  - the behavioural half of RQ1, and the same
                question `scripts/eval_band_decimation.py` asks zero-shot.

Using the SAME band subsets as the zero-shot script is the point: the two then
read as one curve with two conditions (trained at C vs transferred to C),
rather than as two unrelated experiments.

Cost: I/O-bound, a few minutes per split on NVMe, ~11.5 GB at C=16 and ~5.8 GB
at C=8, against the 23 GB parent build.

Usage
-----
    python scripts/make_band_subset_build.py \
        --src data/hsi_v8-80_10_10_importance-new/hsi \
        --keep 16 --out data/hsi_v8-bands16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_band_decimation import decimate_indices  # one definition of "which bands"
from training.npy_atomic import write_dataset_manifest

# Copied verbatim: labels, group/capture ids and the small JSON sidecars are
# band-independent, and a build missing them is not loadable by NpyDatasetV16
# or scoreable per patient.
_COPY_AS_IS = (
    "y_train.npy", "y_val.npy", "y_test.npy",
    "groups_train.npy", "groups_val.npy", "groups_test.npy",
    "captures_train.npy", "captures_val.npy", "captures_test.npy",
    "images_train.npy", "images_val.npy", "images_test.npy",
    "class_names.json", "capture_index.json",
    # dataset_manifest.json is deliberately NOT here: it is REGENERATED below,
    # because the parent's records 32-band shapes, sizes and sha256 digests.
)
_SPLITS = ("train", "val", "test")


def slice_array(src_path: Path, dst_path: Path, band_idx: np.ndarray, chunk: int) -> dict:
    """Stream `(N, H, W, C) -> (N, H, W, len(band_idx))` without loading either
    array whole. `np.lib.format.open_memmap` writes a real .npy header, so the
    result is indistinguishable from a prepared build.
    """
    src = np.load(src_path, mmap_mode="r")
    if src.ndim != 4:
        raise ValueError(f"{src_path} has shape {src.shape}; expected (N, H, W, C)")
    n, height, width, c_src = src.shape
    if int(band_idx.max()) >= c_src:
        raise ValueError(f"band index {int(band_idx.max())} out of range for C={c_src}")

    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype,
                                    shape=(n, height, width, len(band_idx)))
    try:
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            dst[start:stop] = src[start:stop][..., band_idx]
            done = stop / n
            print(f"\r  {src_path.name}: {stop:,}/{n:,} ({done:6.1%})", end="", flush=True)
        dst.flush()
    finally:
        del dst
    print()
    return {"n": int(n), "hw": [int(height), int(width)],
            "c_src": int(c_src), "c_dst": int(len(band_idx)),
            "dtype": str(src.dtype), "bytes": int(dst_path.stat().st_size)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Slice a prepared HSI build down to fewer bands.")
    p.add_argument("--src", required=True, help="an existing build directory holding X_*.npy")
    p.add_argument("--keep", type=int, required=True, help="how many bands to keep")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=20000, help="patches per write (memory knob)")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = p.parse_args(argv)

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in _SPLITS:
            p.error(f"unknown split {s!r}")

    wl_path = src / "wavelengths.npy"
    if not wl_path.is_file():
        print(f"ERROR: {wl_path} not found - this is not an HSI build.")
        return 2
    wavelengths = np.load(wl_path)
    n_source = len(wavelengths)

    band_idx = decimate_indices(n_source, args.keep)
    if len(band_idx) == n_source:
        print(f"ERROR: --keep {args.keep} >= source band count {n_source}; nothing to do.")
        return 2

    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: {out} exists and is not empty. Pass --force to overwrite.")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    kept_nm = np.asarray(wavelengths)[band_idx]
    print(f"[src]   {src}   C={n_source}")
    print(f"[keep]  {len(band_idx)} bands, indices {band_idx.tolist()}")
    print(f"[nm]    {[round(float(v), 1) for v in kept_nm]}")
    print(f"[out]   {out}")

    report = {"source_build": str(src), "source_bands": int(n_source),
              "kept_bands": int(len(band_idx)),
              "band_indices": band_idx.tolist(),
              "wavelengths_nm": [round(float(v), 3) for v in kept_nm],
              "selection": "uniform in index over the source band set (NOT uniform in nm)",
              "rationale": ("decimation of the parent build, so these are the SAME subsets "
                             "scripts/eval_band_decimation.py evaluates zero-shot; this is "
                             "NOT a re-selection of the best N of 740 bands"),
              "splits": {}}

    for split in splits:
        src_x = src / f"X_{split}.npy"
        if not src_x.is_file():
            print(f"  [skip] {src_x.name} not present")
            continue
        report["splits"][split] = slice_array(src_x, out / f"X_{split}.npy", band_idx, args.chunk)

    np.save(out / "wavelengths.npy", kept_nm.astype(wavelengths.dtype))
    if (src / "wavelengths_full.npy").is_file():
        shutil.copy2(src / "wavelengths_full.npy", out / "wavelengths_full.npy")

    copied = []
    for name in _COPY_AS_IS:
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)
            copied.append(name)
    report["copied_verbatim"] = copied

    # Regenerate rather than copy. The parent's manifest records 32-band shapes,
    # byte sizes and sha256 digests, so copying it makes every sliced array look
    # TRUNCATED to `verify_against_manifest` and the run aborts at startup with
    # DATASET_TRUNCATED before epoch 1 - which is exactly what the gate is for.
    written = sorted(q.name for q in out.glob("*.npy"))
    write_dataset_manifest(str(out), written, sha256=True, extra={
        "derived_from": str(src),
        "derivation": "band subset (scripts/make_band_subset_build.py)",
        "source_bands": int(n_source),
        "kept_bands": int(len(band_idx)),
        "band_indices": band_idx.tolist(),
    })
    report["manifest"] = {"regenerated": True, "n_files": len(written), "sha256": True}

    with open(out / "band_subset_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    total = sum(v["bytes"] for v in report["splits"].values())
    print(f"[copied] {len(copied)} band-independent files")
    print(f"[manifest] regenerated over {len(written)} .npy files with sha256 "
          f"(copying the parent's would abort the run as DATASET_TRUNCATED)")
    print(f"[written] {out}/band_subset_report.json   ({total/1e9:.2f} GB of X_*.npy)")
    print(f"[next]   train on it with --data_dir {out}  (the entry point reads C from the data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
