# -*- coding: utf-8 -*-
"""
training/synthetic_mix.py
==========================
A synthetic export, read next to the real build it was generated from: the
train-side half of the HSI augmentation pipeline (plan, Phase 9). Nothing here
runs unless `train.py --synthetic_dir` is given.

Why
---
The generators (CVAE, LDM) write extra DCIS patches. Adding them to a run must
change nothing else about it:
  - the real files are referenced, never copied (a merged copy of the v9 build
    would be 19 GB, and an invitation to fit statistics on synthetic rows);
  - normalization stays the one fitted on the REAL X_train, so a synthetic row
    goes through exactly the arithmetic a real row does;
  - the minority machinery (`training/minority_augmentation.py`, not edited)
    copies and mixes REAL rows only - a synthetic row is a sample, never a
    partner and never a source of copies.
And an export must never be used on a build it was not made from: a v8 export
on the v9 build would train on patches whose value scale, band set or split
are not the run's. The export records the sha256 of the real build's
`dataset_manifest.json`, which changes whenever any file of the build does.

What
----
  1. The export schema (file names, dtypes, the generators' group ids) and
     `write_synthetic_export`, the one writer. The export stage imports both
     from here, so the two sides cannot drift apart.
  2. `load_synthetic_export` - refuses an export that is not bound to this
     build, that does not match its own manifest, or whose arrays would load
     silently wrong: not float16, the wrong patch shape, a non-finite value
     (`NpyDatasetV16.__getitem__` turns +inf into 1.0 and NaN into 0.0 without
     a word), a label outside the classes, other wavelengths or class names.
  3. `attach_synthetic` - `ConcatDataset([real, synthetic])`, both built by the
     pipeline's own `_build_dataset`, so the synthetic rows share the real
     `global_mean/std` and return the same tuple (the `paper_recipe`
     `(x, y, norm_stats)` included). `pool_labels` is the real labels followed
     by -1 for every synthetic row; `MinorityAugmentedDataset` builds its
     partner pool with `flatnonzero(labels == c)`, which never matches -1.
  4. `plan_minority_with_synthetic` - the `MinorityPlan` over that
     concatenation: classes resolved and rows copied on the real part only,
     each synthetic row appended once. `copy_mask` (the glass rule) narrows
     which real rows may be copied without changing how many copies are added,
     so every arm keeps the same number of rows per epoch.

Export directory
----------------
    X_syn.npy              float16 [N, H, W, C], the real build's value space
    y_syn.npy              int64   [N]
    groups_syn.npy         int32   [N], the source's id (-1 cvae, -2 ldm); patient
                           ids are >= 0, so a synthetic row is never anyone's patient
    gen_index.npy          int64   [N], each row's position in generation order
    wavelengths.npy        byte copies of the real build's files (only those the
    class_names.json       real build has)
    dataset_manifest.json  `write_dataset_manifest(sha256=True)` over the .npy files
    provenance.json        written LAST - an export without one is incomplete

`provenance.json` holds at least `schema_version`, `export_id` (the first 12
hex digits of sha256(X_syn.npy)), `source`, `group_id`,
`source_manifest_sha256`, `n_rows`, `hwc` and `class_counts`; the export stage
adds its own keys (generator checkpoint sha, seeds, QC thresholds, ...)
through `extra_provenance`. An empty export (0 rows) is valid.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from training.minority_augmentation import (
    AUTO_FRACTION_OF_MEDIAN, MinorityPlan, build_minority_oversample_indices, resolve_minority_classes,
)
from training.npy_atomic import (
    MANIFEST_NAME, _sha256_of_file, save_json_atomic, save_npy_atomic, verify_against_manifest,
    write_dataset_manifest,
)
from training.npy_integrity import _read_npy_header

# ============================================================================
# Schema
# ============================================================================

# Bump on any change a reader must know about (a file, a dtype, a required
# provenance key). The loader accepts exactly this version.
SCHEMA_VERSION = 1

X_SYN_NAME = "X_syn.npy"
Y_SYN_NAME = "y_syn.npy"
GROUPS_SYN_NAME = "groups_syn.npy"
GEN_INDEX_NAME = "gen_index.npy"
PROVENANCE_NAME = "provenance.json"
WAVELENGTHS_NAME = "wavelengths.npy"
CLASS_NAMES_NAME = "class_names.json"
# Written by the QC/export stage next to the files above; nothing here reads them.
QC_SCORES_NAME = "qc_scores.npz"
QC_REPORT_NAME = "qc_report.json"
FIGURES_DIR = "figures"

X_SYN_DTYPE = np.dtype(np.float16)
Y_SYN_DTYPE = np.dtype(np.int64)
GROUPS_SYN_DTYPE = np.dtype(np.int32)
GEN_INDEX_DTYPE = np.dtype(np.int64)

# The group id every row of a source carries. Negative, so it never equals a
# patient id (`scripts/derive_patch_groups.py` writes patient numbers >= 0).
SYNTHETIC_GROUP_IDS: Dict[str, int] = {"cvae": -1, "ldm": -2}

# The .npy files the export's manifest lists (wavelengths.npy too when present).
EXPORT_ARRAYS = (X_SYN_NAME, Y_SYN_NAME, GROUPS_SYN_NAME, GEN_INDEX_NAME)

EXPORT_ID_LEN = 12

# Keys the writer sets itself; `extra_provenance` may not override them.
PROVENANCE_KEYS = ("schema_version", "export_id", "source", "group_id", "source_manifest_sha256",
                   "n_rows", "hwc", "class_counts", "created_utc")

# `pool_labels` value of a synthetic row: matches no class, so no partner pool.
NOT_A_PARTNER = -1

# Rows per read when scanning X_syn for non-finite values: ~64 MB at a time,
# never the whole array (122,850 x 11 x 11 x 32 float16 is ~950 MB).
_SCAN_CHUNK_BYTES = 64 << 20

MANIFEST_CHECK_MODES = ("auto", "off", "strict")
BUILD_CHECK_MODES = ("on", "off")


class SyntheticExportError(ValueError):
    """An export that must not be trained on. The message says which check
    failed and how to fix it."""


# ============================================================================
# Small helpers
# ============================================================================

def _npy_header(path: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Shape and dtype from the header alone, after checking the file holds all
    the bytes the header promises - a short file read through mmap would SIGBUS
    the process instead of raising."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        raise SyntheticExportError(f"{name}: missing")
    try:
        shape, dtype, _fortran, offset = _read_npy_header(path)
    except Exception as e:
        raise SyntheticExportError(f"{name}: unreadable .npy header ({e})") from e
    dtype = np.dtype(dtype)
    expected = int(offset) + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    size = os.path.getsize(path)
    if size < expected:
        raise SyntheticExportError(f"{name}: truncated ({size} of {expected} bytes)")
    return tuple(int(s) for s in shape), dtype


def _rows_per_chunk(X: np.ndarray) -> int:
    row_bytes = int(np.prod(X.shape[1:], dtype=np.int64)) * X.dtype.itemsize
    return max(1, _SCAN_CHUNK_BYTES // max(1, row_bytes))


def _nonfinite_rows(X: np.ndarray) -> Tuple[int, Optional[int]]:
    """(number of rows holding a NaN/inf, the first such row), read in chunks."""
    n_bad, first = 0, None
    step = _rows_per_chunk(X)
    for start in range(0, len(X), step):
        block = np.asarray(X[start:start + step])
        ok = np.isfinite(block).reshape(len(block), -1).all(axis=1)
        if not ok.all():
            bad = np.flatnonzero(~ok)
            n_bad += len(bad)
            first = int(start + bad[0]) if first is None else first
    return n_bad, first


def _nonfinite_message(n_bad: int, first: int) -> str:
    return (f"{X_SYN_NAME}: {n_bad} row(s) hold a NaN/inf after the float16 cast (first: row {first}); "
            f"NpyDatasetV16 would silently train on them as 0.0/1.0")


def _real_manifest_sha256(real_data_dir: str) -> Optional[str]:
    p = os.path.join(real_data_dir, MANIFEST_NAME)
    return _sha256_of_file(p) if os.path.isfile(p) else None


def _read_class_names(data_dir: str) -> Optional[List[str]]:
    p = os.path.join(data_dir, CLASS_NAMES_NAME)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def _read_wavelengths(data_dir: str) -> Optional[np.ndarray]:
    p = os.path.join(data_dir, WAVELENGTHS_NAME)
    return np.load(p) if os.path.isfile(p) else None


def _copy_or_remove(src_dir: str, dst_dir: str, name: str) -> bool:
    """Byte-copy `src_dir/name` into `dst_dir` (tmp + rename), or delete a stale
    copy when the source has none. Returns whether the file now exists."""
    src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
    if not os.path.isfile(src):
        if os.path.isfile(dst):
            os.remove(dst)
        return False
    shutil.copyfile(src, f"{dst}.tmp")
    os.replace(f"{dst}.tmp", dst)
    return True


def _class_counts(y: np.ndarray, class_names: Optional[Sequence[str]]) -> Dict[str, int]:
    n = len(class_names) if class_names else (int(y.max()) + 1 if len(y) else 0)
    counts = np.bincount(y, minlength=n)
    keys = list(class_names) if class_names else [str(c) for c in range(len(counts))]
    return {k: int(v) for k, v in zip(keys, counts)}


# ============================================================================
# Writer
# ============================================================================

def write_synthetic_export(out_dir: str, X_syn: np.ndarray, y_syn: np.ndarray, source: str,
                           real_data_dir: str, gen_index: Optional[np.ndarray] = None,
                           extra_provenance: Optional[Dict] = None) -> Dict:
    """Write an export `load_synthetic_export` accepts, and return its provenance.

    `X_syn` is [N, H, W, C] in the real build's value space; it is cast to
    float16 and refused if any value is then non-finite. `y_syn` holds class
    indices. `source` is a key of `SYNTHETIC_GROUP_IDS`. `gen_index` defaults
    to `arange(N)` (rows already in generation order). `extra_provenance` adds
    keys to provenance.json; it may not override `PROVENANCE_KEYS`.

    The real build must have a `dataset_manifest.json` - its sha256 is what
    binds the export to the build. N = 0 writes a valid empty export.
    """
    if source not in SYNTHETIC_GROUP_IDS:
        raise SyntheticExportError(f"unknown source {source!r}; expected one of {sorted(SYNTHETIC_GROUP_IDS)}")
    clash = sorted(set(extra_provenance or {}) & set(PROVENANCE_KEYS))
    if clash:
        raise SyntheticExportError(f"extra_provenance may not set {clash}; the writer owns them")
    if os.path.realpath(out_dir) == os.path.realpath(real_data_dir):
        # The export's manifest would replace the real build's own.
        raise SyntheticExportError(f"out_dir is the real build {real_data_dir}; an export lives in its own "
                                   f"directory and references the real files")
    source_sha = _real_manifest_sha256(real_data_dir)
    if source_sha is None:
        raise SyntheticExportError(
            f"the real build {real_data_dir} has no {MANIFEST_NAME}, so an export cannot be bound to it; "
            f"write one first (training.npy_atomic.write_dataset_manifest)")

    X = np.asarray(X_syn)
    if X.ndim != 4:
        raise SyntheticExportError(f"X_syn must be [N, H, W, C], got shape {X.shape}")
    if X.dtype != X_SYN_DTYPE:
        # Values above float16's max (65504) become inf here; the scan below refuses them.
        with np.errstate(over="ignore", invalid="ignore"):
            X = X.astype(X_SYN_DTYPE)
    n_bad, first = _nonfinite_rows(X)
    if n_bad:
        raise SyntheticExportError(_nonfinite_message(n_bad, first))
    real_x = os.path.join(real_data_dir, "X_train.npy")
    if os.path.isfile(real_x):
        real_hwc = _npy_header(real_x)[0][1:]
        if tuple(X.shape[1:]) != real_hwc:
            raise SyntheticExportError(f"X_syn patches are {tuple(X.shape[1:])}, the real build's {real_hwc}")

    n = len(X)
    y = np.asarray(y_syn)
    if y.shape != (n,) or (n and not np.issubdtype(y.dtype, np.integer)):
        raise SyntheticExportError(f"y_syn must be {n} integer labels, got shape {y.shape} dtype {y.dtype}")
    y = y.astype(Y_SYN_DTYPE)
    class_names = _read_class_names(real_data_dir)
    if n and (y.min() < 0 or (class_names and y.max() >= len(class_names))):
        raise SyntheticExportError(f"y_syn holds labels {sorted(set(y.tolist()))} outside the real build's "
                                   f"classes {class_names}")
    gen = np.arange(n, dtype=GEN_INDEX_DTYPE) if gen_index is None else np.asarray(gen_index)
    if gen.shape != (n,):
        raise SyntheticExportError(f"gen_index must have shape ({n},), got {gen.shape}")
    groups = np.full(n, SYNTHETIC_GROUP_IDS[source], dtype=GROUPS_SYN_DTYPE)

    os.makedirs(out_dir, exist_ok=True)
    # provenance.json goes first and comes back last: a crash in between leaves
    # an export the loader calls incomplete, never one that mixes two writes.
    prov_path = os.path.join(out_dir, PROVENANCE_NAME)
    if os.path.isfile(prov_path):
        os.remove(prov_path)
    for name, arr in ((X_SYN_NAME, X), (Y_SYN_NAME, y), (GROUPS_SYN_NAME, groups),
                      (GEN_INDEX_NAME, gen.astype(GEN_INDEX_DTYPE))):
        save_npy_atomic(os.path.join(out_dir, name), arr)
    arrays = list(EXPORT_ARRAYS)
    if _copy_or_remove(real_data_dir, out_dir, WAVELENGTHS_NAME):
        arrays.append(WAVELENGTHS_NAME)
    _copy_or_remove(real_data_dir, out_dir, CLASS_NAMES_NAME)

    x_sha = _sha256_of_file(os.path.join(out_dir, X_SYN_NAME))
    write_dataset_manifest(out_dir, arrays, sha256=True, precomputed_sha256={X_SYN_NAME: x_sha})
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "export_id": x_sha[:EXPORT_ID_LEN],
        "source": source,
        "group_id": SYNTHETIC_GROUP_IDS[source],
        "source_manifest_sha256": source_sha,
        "n_rows": int(n),
        "hwc": [int(s) for s in X.shape[1:]],
        "class_counts": _class_counts(y, class_names),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        **(extra_provenance or {}),
    }
    save_json_atomic(prov_path, provenance)
    return provenance


# ============================================================================
# Loader
# ============================================================================

@dataclass
class SyntheticExport:
    """A validated export. `x_path`/`y_path` go to `build_ds`; the small arrays
    (labels, groups, generation order) are held in memory, X_syn is not."""
    path: str
    x_path: str
    y_path: str
    y: np.ndarray            # int64 [n_syn]
    groups: np.ndarray       # int32 [n_syn], all equal to the source's group id
    gen_index: np.ndarray    # int64 [n_syn]
    hwc: Tuple[int, int, int]
    provenance: Dict = field(default_factory=dict)

    @property
    def n_syn(self) -> int:
        return int(len(self.y))

    @property
    def export_id(self) -> str:
        return self.provenance["export_id"]

    @property
    def source(self) -> str:
        return self.provenance["source"]


def _check_export_manifest(path: str, manifest_check: str) -> None:
    """`train_preflight._check_manifest`'s `--manifest_check` semantics, applied
    to the export directory: 'off' skips it, 'auto' accepts an export without a
    manifest (structural checks only), 'strict' refuses one. A manifest that is
    present must list the export's arrays, match them, and record the X_syn
    sha256 that `export_id` was taken from."""
    if manifest_check == "off":
        return
    manifest_path = os.path.join(path, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        if manifest_check == "strict":
            raise SyntheticExportError(f"--manifest_check strict but no {MANIFEST_NAME} in {path}")
        print(f"[synthetic] no {MANIFEST_NAME} in {path} - structural validation only.", flush=True)
        return
    problems = verify_against_manifest(path)
    if problems:
        raise SyntheticExportError("export does not match its manifest:\n  " + "\n  ".join(problems))
    with open(manifest_path) as f:
        files = json.load(f).get("files") or {}
    missing = [n for n in EXPORT_ARRAYS if n not in files]
    if missing:
        raise SyntheticExportError(f"{MANIFEST_NAME} does not list {missing}")
    with open(os.path.join(path, PROVENANCE_NAME)) as f:
        export_id = json.load(f).get("export_id")
    x_sha = files[X_SYN_NAME].get("sha256")
    if x_sha and export_id != x_sha[:EXPORT_ID_LEN]:
        raise SyntheticExportError(f"provenance export_id {export_id!r} is not the manifest's X_syn sha256 "
                                   f"({x_sha[:EXPORT_ID_LEN]}): provenance.json belongs to another export")
    print(f"[synthetic] all files match {MANIFEST_NAME}.", flush=True)


def load_synthetic_export(path: str, real_data_dir: str, num_classes: int,
                          hwc: Optional[Sequence[int]] = None, manifest_check: str = "auto",
                          build_check: str = "on") -> SyntheticExport:
    """Validate the export at `path` against the real build in `real_data_dir`
    and return it. Every refusal raises `SyntheticExportError`.

    `hwc` is the real patches' (H, W, C); None reads it from the real
    X_train.npy header. `manifest_check` takes `--manifest_check`'s values
    (auto/off/strict) and applies them to the export's own manifest.
    `build_check` 'on' requires `provenance.source_manifest_sha256` to equal the
    sha256 of `<real_data_dir>/dataset_manifest.json`; a real build WITHOUT a
    manifest is refused too, since nothing then ties the export to it. 'off'
    downgrades both to a printed warning.

    Cheap checks run first; the manifest's sha256 pass over the export and the
    chunked finite scan of X_syn run last.
    """
    if manifest_check not in MANIFEST_CHECK_MODES:
        raise ValueError(f"manifest_check must be one of {MANIFEST_CHECK_MODES}, got {manifest_check!r}")
    if build_check not in BUILD_CHECK_MODES:
        raise ValueError(f"build_check must be one of {BUILD_CHECK_MODES}, got {build_check!r}")
    path = os.fspath(path)
    if not os.path.isdir(path):
        raise SyntheticExportError(f"no export directory at {path}")
    prov_path = os.path.join(path, PROVENANCE_NAME)
    if not os.path.isfile(prov_path):
        raise SyntheticExportError(f"{path} has no {PROVENANCE_NAME}: the export is incomplete "
                                   f"(the writer saves it last)")
    try:
        with open(prov_path) as f:
            prov = json.load(f)
    except (OSError, ValueError) as e:
        raise SyntheticExportError(f"{PROVENANCE_NAME} is unreadable: {e}") from e
    missing = [k for k in PROVENANCE_KEYS if k not in prov]
    if prov.get("schema_version") != SCHEMA_VERSION:
        raise SyntheticExportError(f"schema_version {prov.get('schema_version')!r}, this reader expects "
                                   f"{SCHEMA_VERSION}; re-export with the current pipeline")
    if missing:
        raise SyntheticExportError(f"{PROVENANCE_NAME} lacks {missing}")
    source = prov["source"]
    if source not in SYNTHETIC_GROUP_IDS or prov["group_id"] != SYNTHETIC_GROUP_IDS[source]:
        raise SyntheticExportError(f"source {source!r} / group_id {prov['group_id']!r} is not one of "
                                   f"{SYNTHETIC_GROUP_IDS}")

    # ---- bound to THIS build ------------------------------------------------
    real_sha = _real_manifest_sha256(real_data_dir)
    if real_sha is None or prov["source_manifest_sha256"] != real_sha:
        why = (f"the real build {real_data_dir} has no {MANIFEST_NAME}, so the export cannot be matched to it"
               if real_sha is None else
               f"the export was made from a different build: source_manifest_sha256 "
               f"{str(prov['source_manifest_sha256'])[:12]}... != sha256({real_data_dir}/{MANIFEST_NAME}) "
               f"{real_sha[:12]}...")
        if build_check == "on":
            raise SyntheticExportError(why + " (--synthetic_build_check off overrides)")
        print(f"[synthetic] WARNING (build check off): {why}", flush=True)

    # ---- shapes and dtypes, from the headers --------------------------------
    if hwc is None:
        hwc = _npy_header(os.path.join(real_data_dir, "X_train.npy"))[0][1:]
    hwc = tuple(int(s) for s in hwc)
    x_path, y_path = os.path.join(path, X_SYN_NAME), os.path.join(path, Y_SYN_NAME)
    want = {X_SYN_NAME: X_SYN_DTYPE, Y_SYN_NAME: Y_SYN_DTYPE, GROUPS_SYN_NAME: GROUPS_SYN_DTYPE,
            GEN_INDEX_NAME: GEN_INDEX_DTYPE}
    headers = {name: _npy_header(os.path.join(path, name)) for name in EXPORT_ARRAYS}
    for name, (shape, dtype) in headers.items():
        if dtype != want[name]:
            raise SyntheticExportError(f"{name}: dtype {dtype}, the schema says {want[name]}")
    x_shape = headers[X_SYN_NAME][0]
    if len(x_shape) != 4 or x_shape[1:] != hwc:
        raise SyntheticExportError(f"{X_SYN_NAME}: shape {x_shape}, expected [N, H, W, C] with "
                                   f"(H, W, C) = {hwc} (the real patches)")
    n = x_shape[0]
    for name in (Y_SYN_NAME, GROUPS_SYN_NAME, GEN_INDEX_NAME):
        if headers[name][0] != (n,):
            raise SyntheticExportError(f"{name}: shape {headers[name][0]}, expected ({n},) like {X_SYN_NAME}")
    if prov["n_rows"] != n:
        raise SyntheticExportError(f"provenance n_rows {prov['n_rows']} != {n} rows in {X_SYN_NAME}")

    # ---- the real build's bands and classes ---------------------------------
    real_wl, syn_wl = _read_wavelengths(real_data_dir), _read_wavelengths(path)
    if (real_wl is None) != (syn_wl is None) or (
            real_wl is not None and (real_wl.shape != syn_wl.shape or not np.array_equal(real_wl, syn_wl))):
        raise SyntheticExportError(f"{WAVELENGTHS_NAME} differs from the real build's")
    real_names, syn_names = _read_class_names(real_data_dir), _read_class_names(path)
    if real_names != syn_names:
        raise SyntheticExportError(f"{CLASS_NAMES_NAME} {syn_names} differs from the real build's {real_names}")

    # ---- the export's own manifest (sha256 pass over the export) ------------
    _check_export_manifest(path, manifest_check)

    # ---- values ---------------------------------------------------------------
    y = np.load(y_path)
    if n and (y.min() < 0 or y.max() >= num_classes):
        bad = sorted(set(y[(y < 0) | (y >= num_classes)].tolist()))
        raise SyntheticExportError(f"{Y_SYN_NAME}: labels {bad} outside the {num_classes} classes")
    groups = np.load(os.path.join(path, GROUPS_SYN_NAME))
    if n and not np.all(groups == SYNTHETIC_GROUP_IDS[source]):
        raise SyntheticExportError(f"{GROUPS_SYN_NAME}: every row must be {SYNTHETIC_GROUP_IDS[source]} "
                                   f"(source {source!r}), found {sorted(set(groups.tolist()))}")
    n_bad, first = _nonfinite_rows(np.load(x_path, mmap_mode="r"))
    if n_bad:
        raise SyntheticExportError(_nonfinite_message(n_bad, first))

    return SyntheticExport(path=path, x_path=x_path, y_path=y_path, y=y, groups=groups,
                           gen_index=np.load(os.path.join(path, GEN_INDEX_NAME)), hwc=hwc, provenance=prov)


# ============================================================================
# Real + synthetic training set
# ============================================================================

# `NpyDatasetV16` attributes that decide what an item is. Real and synthetic
# must agree on all of them, or the two halves would be normalized differently.
_ITEM_ATTRS = ("normalization", "return_norm_stats", "augment", "global_mean", "global_std")


def _differs(a, b) -> bool:
    if a is None or b is None:
        return (a is None) != (b is None)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return not np.array_equal(a, b)
    return a != b


def _item_settings_diff(a: Dataset, b: Dataset) -> List[str]:
    return [attr for attr in _ITEM_ATTRS if _differs(getattr(a, attr, None), getattr(b, attr, None))]


@dataclass
class MixedTrainSet:
    """Row i < n_real is real row i; row n_real + j is synthetic row j.

    `y`/`groups`/`pool_labels` are aligned with `dataset`, which is what
    `MinorityAugmentedDataset` calls the ROOT dataset (`root_indices` stops at
    the `ConcatDataset`), so pass `labels=pool_labels, groups=groups` there.
    """
    dataset: ConcatDataset
    y: np.ndarray                 # int64, y_real ++ y_syn: class weights, the minority plan
    groups: Optional[np.ndarray]  # int32, groups_real ++ groups_syn; None without real groups
    pool_labels: np.ndarray       # int64, y_real ++ [-1] * n_syn: mixing partners are real rows
    n_real: int
    n_syn: int
    export: SyntheticExport

    def describe(self, class_names: Optional[Sequence[str]] = None) -> Dict:
        """The `config.json["synthetic"]` block (JSON-serializable)."""
        return {
            "export_id": self.export.export_id,
            "source": self.export.source,
            "schema_version": self.export.provenance.get("schema_version"),
            "export_dir": self.export.path,
            "source_manifest_sha256": self.export.provenance.get("source_manifest_sha256"),
            "n_real": int(self.n_real), "n_syn": int(self.n_syn), "n_total": int(self.n_real + self.n_syn),
            "syn_class_counts": _class_counts(self.y[self.n_real:], class_names),
            "group_id": self.export.provenance.get("group_id"),
            "normalization": "real X_train statistics (synthetic rows share the real dataset's)",
            "mixing_partners": "real rows only",
            "cross_patient_groups": self.groups is not None,
            "provenance": self.export.provenance,
        }


def attach_synthetic(real_ds: Dataset, y_real: np.ndarray, groups_real: Optional[np.ndarray],
                     export: SyntheticExport, build_ds: Callable[[str, str], Dataset]) -> MixedTrainSet:
    """Real rows first, then synthetic. `build_ds(x_path, y_path)` must be the
    callable that built `real_ds` (the pipeline's `_build_dataset`), so the
    synthetic `NpyDatasetV16` gets the same normalization, the same real
    `global_mean/std` and the same return tuple; a dataset whose settings differ
    from `real_ds`'s is refused.

    `groups_real` None (no `groups_train.npy`): `groups` is None as well. The
    real rows have no patient ids, so there is nothing for the synthetic ids to
    sit next to, and `MinorityAugmentedDataset` then draws partners from any
    patient - what it already does on such a build. Real ids must be >= 0 so
    they cannot collide with the synthetic ones.
    """
    n_real = len(real_ds)
    y_real = np.asarray(y_real)
    if len(y_real) != n_real:
        raise ValueError(f"y_real has {len(y_real)} rows, the real dataset {n_real}")
    real_x = getattr(real_ds, "X", None)
    if real_x is not None and tuple(real_x.shape[1:]) != tuple(export.hwc):
        raise ValueError(f"real patches are {tuple(real_x.shape[1:])}, the export's {tuple(export.hwc)}")

    syn_ds = build_ds(export.x_path, export.y_path)
    if len(syn_ds) != export.n_syn:
        raise ValueError(f"build_ds gave {len(syn_ds)} synthetic rows, the export has {export.n_syn}")
    diff = _item_settings_diff(real_ds, syn_ds)
    if diff:
        raise ValueError(f"the synthetic dataset differs from the real one in {diff}; build both with the "
                         f"same callable so they share the real X_train statistics")

    groups = None
    if groups_real is not None:
        groups_real = np.asarray(groups_real)
        if len(groups_real) != n_real:
            raise ValueError(f"groups_real has {len(groups_real)} rows, the real dataset {n_real}")
        if n_real and groups_real.min() < 0:
            raise ValueError(f"groups_real holds negative ids {sorted(set(groups_real[groups_real < 0].tolist()))}"
                             f", which collide with the synthetic group ids {SYNTHETIC_GROUP_IDS}")
        groups = np.concatenate([groups_real.astype(GROUPS_SYN_DTYPE), export.groups.astype(GROUPS_SYN_DTYPE)])

    n_syn = export.n_syn
    return MixedTrainSet(
        dataset=ConcatDataset([real_ds, syn_ds]),
        y=np.concatenate([y_real.astype(Y_SYN_DTYPE), export.y.astype(Y_SYN_DTYPE)]),
        groups=groups,
        pool_labels=np.concatenate([y_real.astype(Y_SYN_DTYPE), np.full(n_syn, NOT_A_PARTNER, Y_SYN_DTYPE)]),
        n_real=int(n_real), n_syn=int(n_syn), export=export)


# ============================================================================
# Minority plan over real + synthetic
# ============================================================================

def _masked_oversample_indices(y: np.ndarray, minority_classes: Sequence[int], factor: float, seed: int,
                               copy_mask: np.ndarray) -> np.ndarray:
    """`build_minority_oversample_indices` with the copies drawn from the rows
    `copy_mask` allows. The NUMBER of copies is still taken from the whole
    class - round(factor x n_class) - n_class - so the row total does not
    depend on the mask. Same RNG use, so an all-True mask gives the same
    indices."""
    if factor < 1.0:
        raise ValueError(f"oversample factor must be >= 1.0, got {factor}")
    rng = np.random.default_rng(seed)
    parts = [np.arange(len(y))]
    for c in minority_classes:
        idx = np.flatnonzero(y == c)
        if len(idx) == 0:
            continue
        extra = int(round(factor * len(idx))) - len(idx)
        pool = idx[copy_mask[idx]]
        if extra and len(pool) == 0:
            raise ValueError(f"copy_mask allows no row of class {c} to be copied ({len(idx)} rows, "
                             f"{extra} copies needed)")
        if not extra:
            continue
        whole, rest = divmod(extra, len(pool))
        parts.extend([pool] * whole)
        if rest:
            parts.append(np.sort(rng.choice(pool, size=rest, replace=False)))
    return np.concatenate(parts)


def plan_minority_with_synthetic(y_real: np.ndarray, n_syn: int, num_classes: int,
                                 class_names: Sequence[str], spec: Optional[str], factor: float,
                                 seed: int = 0, copy_mask: Optional[np.ndarray] = None,
                                 y_syn: Optional[np.ndarray] = None) -> MinorityPlan:
    """`plan_minority_augmentation` for a `MixedTrainSet` (rows: real, then
    `n_syn` synthetic).

      - classes come from the REAL counts (`--minority_classes auto` must not
        stop picking DCIS because synthetic DCIS rows were added);
      - `indices` = the real-only oversampling (`build_minority_oversample_indices`'s
        whole-copies + remainder rule), then `arange(n_real, n_real + n_syn)`:
        every synthetic row once, never copied. None when factor == 1.0 - the
        concatenation is then visited as is;
      - `copy_mask` (bool [n_real], the glass rule): only these rows are copied,
        but each class still gets round(factor x n_class) - n_class copies;
      - `effective_labels` are the labels of the visited rows, synthetic
        included; `counts_before` are the real counts.

    With `n_syn == 0` and no `copy_mask` this IS `plan_minority_augmentation`
    (pinned by `tests/test_synthetic_mix.py`). The labels are never masked to
    -1 and handed to it: its `np.bincount` refuses negative values.
    """
    y = np.asarray(y_real)
    n_real, n_syn = len(y), int(n_syn)
    if n_syn < 0:
        raise ValueError(f"n_syn must be >= 0, got {n_syn}")
    if y_syn is None:
        if n_syn:
            raise ValueError(f"y_syn is needed for the effective labels of {n_syn} synthetic rows")
        y_all = y
    else:
        y_syn = np.asarray(y_syn)
        if len(y_syn) != n_syn:
            raise ValueError(f"y_syn has {len(y_syn)} rows, n_syn is {n_syn}")
        y_all = y if n_syn == 0 else np.concatenate([y, y_syn.astype(y.dtype)])
    if copy_mask is not None:
        copy_mask = np.asarray(copy_mask)
        if copy_mask.dtype != np.bool_ or copy_mask.shape != (n_real,):
            raise ValueError(f"copy_mask must be a boolean array over the {n_real} real rows, "
                             f"got dtype {copy_mask.dtype} shape {copy_mask.shape}")

    classes = resolve_minority_classes(spec, y, num_classes, class_names)
    if not classes:
        raise ValueError(f"no class has fewer than {AUTO_FRACTION_OF_MEDIAN:g} x the median class "
                         f"count; name the class(es) with --minority_classes")
    indices = None
    if factor != 1.0:
        real_idx = (build_minority_oversample_indices(y, classes, factor, seed) if copy_mask is None
                    else _masked_oversample_indices(y, classes, factor, seed, copy_mask))
        indices = real_idx if n_syn == 0 else np.concatenate([real_idx, np.arange(n_real, n_real + n_syn)])
    effective = y_all if indices is None else y_all[indices]
    return MinorityPlan(classes=classes, factor=float(factor), indices=indices, effective_labels=effective,
                        counts_before=np.bincount(y, minlength=num_classes).tolist(),
                        counts_after=np.bincount(effective, minlength=num_classes).tolist())
