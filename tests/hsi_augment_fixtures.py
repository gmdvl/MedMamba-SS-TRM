# -*- coding: utf-8 -*-
"""
tests/hsi_augment_fixtures.py
=============================
A tiny build with the SAME file layout and JSON schemas as
`data/hsi_v9-trainsel` (Plan §Phase 11, "Fixture"), plus the ground truth the
pipeline's tests check against. Every `test_hsi_augment_*.py` builds on it:

    from tests.hsi_augment_fixtures import make_tiny_build

    @pytest.fixture(scope="module")
    def tiny(tmp_path_factory):
        return make_tiny_build(tmp_path_factory.mktemp("tiny"))

    tiny.data_dir        # pass as --data_dir (the "hsi" dir); tiny.root is the build root
    tiny.is_glass["train"], tiny.duplicate_row, tiny.grid_rc["val"], ...

What it contains
----------------
* 3 classes (healthy, DCIS, IDC); 8 patients x 2 captures; patch 5; 8 bands;
  float16 NHWC; captures cut row-major from one continuous image each, on a
  4 x 5 grid (`grid_cols` = 5, recorded as `extra.grid_cols` in the manifest,
  the override the adapter reads; the real build relies on the class constant 91).
* Split like the real one, scaled down: 6 train patients (2 with DCIS), 1 val
  and 1 test patient, each held-out patient with one DCIS capture, so there is
  "one DCIS patient per held-out split".
* Real-build quirks on purpose: the val IDC capture has 3 grid rows instead of
  4 (as patient 65's 22-row captures do); the test healthy capture's gain is
  clipped and listed in `suspect_captures` (as patient 68's are);
  capture_index.json and split_assignment.json spell the split "validation".
* Bands 540-600 nm and 825-855 nm, 15 nm apart: an 225 nm gap after band 4,
  15 x the median step (trainsel: 222 nm, ~34x).

Ground truth (all deterministic for a given seed)
-------------------------------------------------
* Glass vs tissue. Glass patches are spectrally flat (only the patient's
  +-1.5% band gain), tissue follows Beer-Lambert, I = G exp(-d A_class(lambda)),
  so its spectrum is absorbing and class-shaped. `is_glass[split]` is per row.
  Glass cells form a smooth blob covering 15-40% of each capture.
* Spatial continuity. One smooth texture field per capture image (Gaussian,
  sigma 1.5 px) modulates the tissue density, so the facing edges of grid
  neighbours (r, c) / (r, c + 1) are correlated and a shuffled grid's are not.
* `duplicate_row` is a train row whose X is a byte-exact copy of
  `duplicate_of` (both DCIS tissue rows, from two different patients), for the
  memorization test.
* `grid_rc[split]` holds each row's true (r, c), to check the adapter's
  derivation against.

The manifest's `created_utc` is the build time, so two builds differ in that
file's sha256 even when every array is identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from training.npy_atomic import write_dataset_manifest

CLASS_NAMES = ("healthy", "DCIS", "IDC")
TISSUE_SPELLING = ("Healthy", "DCIS", "IDC")      # how capture keys / capture_index spell them
WAVELENGTHS_FULL = np.arange(500.0, 905.0, 5.0, dtype=np.float32)    # 81 "sensor" bands
SELECTED_BAND_INDICES = np.array([8, 11, 14, 17, 20, 65, 68, 71], dtype=np.int64)
WAVELENGTHS = WAVELENGTHS_FULL[SELECTED_BAND_INDICES]                # 540..600, 825..855
GAP_AFTER = 4                                                         # 600 -> 825 nm
PATCH = 5
GRID_ROWS, GRID_COLS = 4, 5
SHORT_GRID_ROWS = 3
N_BANDS = len(SELECTED_BAND_INDICES)
CORPUS_MEDIAN = 8640.0
GAIN_CLIP = (0.5, 2.0)

# Absorbance per class at the 8 bands (A = -ln T at unit density). Visible
# bands absorb (haematoxylin/eosin-like), NIR barely; DCIS and IDC are denser
# and differently shaped than healthy.
ABSORBANCE = np.array([
    [0.45, 0.55, 0.50, 0.40, 0.30, 0.08, 0.07, 0.06],    # healthy
    [0.70, 0.85, 0.75, 0.55, 0.40, 0.12, 0.11, 0.10],    # DCIS
    [0.80, 0.90, 0.85, 0.70, 0.55, 0.20, 0.18, 0.16],    # IDC
], dtype=np.float64)
GLASS_LEVEL = 1.45          # build value units (raw x gain / corpus median)
NOISE_STD = 0.004

# (patient, split as the build spells it, tissue of capture 1, tissue of capture 2)
PATIENTS: Tuple[Tuple[int, str, str, str], ...] = (
    (101, "train", "DCIS", "Healthy"),
    (102, "train", "DCIS", "IDC"),
    (103, "train", "Healthy", "IDC"),
    (104, "train", "Healthy", "IDC"),
    (105, "train", "IDC", "IDC"),
    (106, "train", "Healthy", "Healthy"),
    (201, "validation", "DCIS", "IDC"),
    (301, "test", "DCIS", "Healthy"),
)
SHORT_CAPTURE_KEY = "IDC_HS_VNIR_201_IDC_x10_C01"        # 3 grid rows, like patient 65's
SUSPECT_CAPTURE_KEY = "Healthy_HS_VNIR_301_Healthy_x10_C01"   # gain clipped, like patient 68's
SPLITS = ("train", "val", "test")
_SPLIT_FILE = {"train": "train", "validation": "val", "test": "test"}


@dataclass(frozen=True, eq=False)
class TinyBuild:
    root: Path                                   # build root (like data/hsi_v9-trainsel)
    data_dir: Path                               # root / "hsi": what --data_dir points at
    class_names: Tuple[str, ...]
    wavelengths: np.ndarray
    gap_after: int                               # the gap is wavelengths[gap_after] -> [gap_after + 1]
    patch: int
    grid_rows: int
    grid_cols: int
    n_bands: int
    minority_class: int                          # DCIS = 1
    patients: Dict[str, Tuple[int, ...]]         # split -> patient ids
    dcis_patients: Dict[str, Tuple[int, ...]]
    captures: Dict[str, Tuple[int, ...]]         # split -> capture indexes, in row order
    n_rows: Dict[str, int]
    is_glass: Dict[str, np.ndarray]              # split -> bool per row
    grid_rc: Dict[str, np.ndarray]               # split -> (n, 2) true (r, c) per row
    duplicate_row: int                           # train row: X_train[duplicate_row] == X_train[duplicate_of]
    duplicate_of: int
    short_captures: Tuple[int, ...]              # capture indexes with fewer grid rows
    suspect_captures: Tuple[str, ...]            # capture keys with a clipped gain
    absorbance: np.ndarray = field(repr=False)   # (3, n_bands) class absorbance at unit density
    glass_level: float = GLASS_LEVEL


def _capture_image(rng: np.random.Generator, rows: int, label: int,
                   band_gain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One capture as a continuous (rows*PATCH, GRID_COLS*PATCH, N_BANDS) image
    and its (rows, GRID_COLS) glass-cell mask."""
    h, w = rows * PATCH, GRID_COLS * PATCH
    cells = gaussian_filter(rng.normal(size=(rows, GRID_COLS)), sigma=1.0)
    k = max(1, int(round(rng.uniform(0.15, 0.40) * cells.size)))
    glass_cells = np.zeros(cells.size, dtype=bool)
    glass_cells[np.argsort(cells, axis=None, kind="stable")[:k]] = True
    glass_cells = glass_cells.reshape(rows, GRID_COLS)
    glass_px = np.kron(glass_cells, np.ones((PATCH, PATCH), dtype=bool))

    tex = gaussian_filter(rng.normal(size=(h, w)), sigma=1.5)
    tex /= tex.std()
    illum = GLASS_LEVEL * (1.0 + 0.02 * tex)                                   # glass: flat spectrum
    density = np.clip(1.0 + 0.35 * tex, 0.2, None)                             # tissue texture
    tissue = illum[..., None] * np.exp(-density[..., None] * ABSORBANCE[label])
    img = np.where(glass_px[..., None], illum[..., None] * np.ones(N_BANDS), tissue)
    img = img * band_gain + rng.normal(scale=NOISE_STD, size=img.shape)
    return img, glass_cells


def _patches(img: np.ndarray, rows: int) -> np.ndarray:
    """Row-major non-overlapping PATCH x PATCH cut, (rows*cols, P, P, B)."""
    b = img.shape[-1]
    x = img.reshape(rows, PATCH, GRID_COLS, PATCH, b).transpose(0, 2, 1, 3, 4)
    return x.reshape(rows * GRID_COLS, PATCH, PATCH, b)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def make_tiny_build(root: Union[str, Path], *, seed: int = 0) -> TinyBuild:
    """Write the tiny build under `root` (created if missing) and return its description."""
    root = Path(root)
    data_dir = root / "hsi"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # capture keys, sorted like the real capture_index (by key string)
    entries = []
    for patient, split, *tissues in PATIENTS:
        seen: Dict[str, int] = {}
        for t in tissues:
            seen[t] = seen.get(t, 0) + 1
            key = f"{t}_HS_VNIR_{patient}_{t}_x10_C{seen[t]:02d}"
            entries.append((key, patient, split, t, f"{seen[t]:02d}"))
    entries.sort(key=lambda e: e[0])

    band_gain = {p: 1.0 + 0.015 * rng.normal(size=N_BANDS) for p, *_ in PATIENTS}
    per_split: Dict[str, Dict[str, List]] = {s: {"X": [], "y": [], "g": [], "c": [], "glass": [], "rc": []}
                                             for s in SPLITS}
    capture_index: Dict[str, Dict] = {}
    gain_report: Dict[str, Dict] = {}
    short_caps, suspects = [], []
    for idx, (key, patient, split, tissue, capno) in enumerate(entries):
        label = TISSUE_SPELLING.index(tissue)
        rows = SHORT_GRID_ROWS if key == SHORT_CAPTURE_KEY else GRID_ROWS
        img, glass_cells = _capture_image(rng, rows, label, band_gain[patient])
        n = rows * GRID_COLS
        median_intensity = 3300.0 if key == SUSPECT_CAPTURE_KEY else float(np.round(rng.uniform(7500, 9500) * 2) / 2)
        raw_gain = CORPUS_MEDIAN / median_intensity
        gain = float(np.clip(raw_gain, *GAIN_CLIP))
        if key == SHORT_CAPTURE_KEY:
            short_caps.append(idx)
        if raw_gain != gain:
            suspects.append(key)
        s = _SPLIT_FILE[split]
        per_split[s]["X"].append(_patches(img, rows))
        per_split[s]["y"].append(np.full(n, label, dtype=np.int64))
        per_split[s]["g"].append(np.full(n, patient, dtype=np.int32))
        per_split[s]["c"].append(np.full(n, idx, dtype=np.int32))
        per_split[s]["glass"].append(glass_cells.reshape(-1))
        per_split[s]["rc"].append(np.stack(np.divmod(np.arange(n), GRID_COLS), axis=1))
        capture_index[str(idx)] = {"key": key, "patient": str(patient), "tissue": tissue, "capture": capno,
                                   "split": split, "n": n, "gain": gain, "median_intensity": median_intensity}
        gain_report[key] = {"gain": gain, "median_intensity": median_intensity, "n_sampled": 400,
                            "raw_gain_before_clip": raw_gain}

    arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for s in SPLITS:
        d = per_split[s]
        arrays[s] = {"X": np.concatenate(d["X"]).astype(np.float16), "y": np.concatenate(d["y"]),
                     "groups": np.concatenate(d["g"]), "captures": np.concatenate(d["c"]),
                     "glass": np.concatenate(d["glass"]), "rc": np.concatenate(d["rc"]).astype(np.int64)}

    # the duplicate: first tissue row of patient 101's DCIS capture copied over
    # the first tissue row of patient 102's DCIS capture
    tr = arrays["train"]
    first_tissue = {}
    for p in (101, 102):
        cap = next(i for i, e in enumerate(entries) if e[1] == p and e[3] == "DCIS")
        first_tissue[p] = int(np.flatnonzero((tr["captures"] == cap) & ~tr["glass"])[0])
    dup_of, dup_row = first_tissue[101], first_tissue[102]
    tr["X"][dup_row] = tr["X"][dup_of]

    # sidecars and arrays, under the real names
    files = []
    for s in SPLITS:
        for kind in ("X", "y", "groups", "captures"):
            name = f"{kind}_{s}.npy"
            np.save(data_dir / name, arrays[s][kind])
            files.append(name)
    np.save(data_dir / "wavelengths.npy", WAVELENGTHS)
    np.save(data_dir / "wavelengths_full.npy", WAVELENGTHS_FULL)
    files += ["wavelengths.npy", "wavelengths_full.npy"]
    _write_json(data_dir / "class_names.json", list(CLASS_NAMES))
    order = {"train": 0, "validation": 1, "test": 2}
    _write_json(data_dir / "capture_index.json",
                dict(sorted(capture_index.items(), key=lambda kv: (order[kv[1]["split"]], int(kv[0])))))

    groups = {str(p): split for p, split, *_ in PATIENTS}
    _write_json(root / "split_assignment.json", {
        "split_algo_version": 2, "kfold_algo_version": None, "strategy": "file", "seed": 42,
        "spec": "80_10_10", "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "class_names": list(CLASS_NAMES), "generated_utc": "2026-01-01T00:00:00+00:00", "groups": groups})
    train_patients = sorted(str(p) for p, split, *_ in PATIENTS if split == "train")
    heldout = sorted(str(p) for p, split, *_ in PATIENTS if split != "train")
    n_train_caps = sum(1 for e in entries if e[2] == "train")
    _write_json(root / "pass1_scope_report.json", {
        "rule": "band selection, band statistics, capture-gain corpus median, value scale and "
                "calibration probe computed on TRAINING captures only",
        "train_patients": train_patients, "n_train_captures": n_train_caps,
        "n_heldout_captures": len(entries) - n_train_caps, "heldout_patients": heldout,
        "band_selection_sample_fraction": 0.1, "seed": 42})
    np.save(root / "selected_wavelengths.npy", WAVELENGTHS)
    np.save(root / "selected_band_indices.npy", SELECTED_BAND_INDICES)
    x_max = max(float(arrays[s]["X"].max()) for s in SPLITS)
    _write_json(root / "capture_gain_report.json", {
        "capture_gain_method": "median_ratio", "corpus_median": CORPUS_MEDIAN, "suspect_captures": suspects,
        "hsi_value_scale_resolved": CORPUS_MEDIAN, "hsi_value_max_pre_scale": x_max * CORPUS_MEDIAN,
        "hsi_value_max_post_scale": x_max, "hsi_all_calibrated_v8": True, "per_capture": gain_report})
    write_dataset_manifest(str(data_dir), files, sha256=True, extra={
        "prep_core": "training.prep.core", "adapter": "histology_hsi_bc_v8_trainsel", "git_sha": "fixture",
        "cli_args": {"split": "80_10_10", "seed": 42, "patch_size": PATCH, "stride": PATCH,
                     "band_selection": "importance", "num_bands": N_BANDS, "capture_gain": "median_ratio",
                     "gain_clip_lo": GAIN_CLIP[0], "gain_clip_hi": GAIN_CLIP[1], "label_source": "tissue"},
        "split": {"spec": "80_10_10", "strategy": "file", "split_algo_version": 2},
        "class_names": list(CLASS_NAMES), "store_dtype": "float16",
        "provenance": {"label_source": "tissue", "band_selection": "importance",
                       "selected_band_indices": SELECTED_BAND_INDICES.tolist(),
                       "capture_gain_method": "median_ratio", "gain_clip": list(GAIN_CLIP),
                       "hsi_value_scale_resolved": CORPUS_MEDIAN},
        "grid_cols": GRID_COLS})

    def _by_split(fn) -> Dict[str, Tuple[int, ...]]:
        return {s: tuple(sorted({p for p, split, *t in PATIENTS if _SPLIT_FILE[split] == s and fn(t)}))
                for s in SPLITS}

    return TinyBuild(
        root=root, data_dir=data_dir, class_names=CLASS_NAMES, wavelengths=WAVELENGTHS.copy(),
        gap_after=GAP_AFTER, patch=PATCH, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, n_bands=N_BANDS,
        minority_class=1, patients=_by_split(lambda t: True),
        dcis_patients=_by_split(lambda t: "DCIS" in t),
        captures={s: tuple(int(c) for c in dict.fromkeys(arrays[s]["captures"].tolist())) for s in SPLITS},
        n_rows={s: int(len(arrays[s]["y"])) for s in SPLITS},
        is_glass={s: arrays[s]["glass"].copy() for s in SPLITS},
        grid_rc={s: arrays[s]["rc"].copy() for s in SPLITS},
        duplicate_row=dup_row, duplicate_of=dup_of, short_captures=tuple(short_caps),
        suspect_captures=tuple(suspects), absorbance=ABSORBANCE.copy())
