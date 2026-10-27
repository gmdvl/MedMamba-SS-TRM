# -*- coding: utf-8 -*-
"""
hsi_augment/adapters/base.py
============================
`BuildAdapter`: what the pipeline needs to know about a prepared build, behind
one thin class, so the stages never hard-code a file name or a dataset fact
(Plan, "Dataset scope": HistologyHSI-BC only, behind a thin adapter).

The build layout (written by `prepare_*.py` through `training.prep.core`):

    <build root>/                      e.g. data/hsi_v9-trainsel
        split_assignment.json, pass1_scope_report.json, ...   dataset-specific sidecars
        hsi/                           the "data dir" train.py's --data_dir points at
            X_{train,val,test}.npy         float16, NHWC (N, H, W, C)
            y_*.npy  int64   groups_*.npy  int32 patient id   captures_*.npy  int32
            capture_index.json         {idx: {key, patient, tissue, capture, split, n, ...}}
            class_names.json, wavelengths.npy, dataset_manifest.json

Either directory may be passed; the other is derived.

What the adapter does NOT do: read rows of a split. That goes through
`io.SplitGuard`, which logs it and enforces the leakage policy. The grid
helpers here therefore take the `captures` array as an argument (from
`SplitArrays.captures`) instead of loading it, because `captures_test.npy`
plus `capture_index.json` reveal the test labels.

Patch coordinates
-----------------
They are not stored. A capture is cut into patches row-major, and the rows of
one capture are contiguous in each split, so a row's ordinal within its
capture's run gives `(r, c) = divmod(ordinal, grid_cols)`. A capture whose
patch count is not a multiple of `grid_cols` gets no coordinates (r = c = -1)
rather than wrong ones.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from hsi_augment.io import ARRAY_KINDS, SPLITS, canonical_split
from training.npy_atomic import MANIFEST_NAME

PathLike = Union[str, os.PathLike]

CAPTURE_INDEX_NAME = "capture_index.json"
CLASS_NAMES_NAME = "class_names.json"
WAVELENGTHS_NAME = "wavelengths.npy"
DATA_SUBDIR = "hsi"

# `spectral_gaps`: a step counts as a gap when it exceeds this multiple of the
# median band spacing. On trainsel the median step is ~6.5 nm and only
# 628.9 -> 850.9 nm (222 nm, ratio ~34) qualifies; the next largest step,
# 575.1 -> 628.9 nm (54 nm, ratio ~8), does not.
GAP_MIN_RATIO = 10.0


class NonContiguousCaptureError(ValueError):
    """A capture's rows are split into more than one run, so ordinals (and
    therefore grid coordinates) are undefined."""


@dataclass(frozen=True)
class CaptureInfo:
    index: int                        # the value stored in captures_*.npy
    key: str                          # e.g. DCIS_HS_VNIR_107_DCIS_x10_C01
    patient: int                      # the value stored in groups_*.npy
    tissue: str                       # spelling of the source ("Healthy", "DCIS", "IDC")
    label: Optional[int]              # class index of `tissue`; None if it maps to no class
    capture: str                      # capture number within (patient, tissue), e.g. "01"
    split: str                        # canonical: train / val / test
    n: int                            # patches the build says this capture holds
    gain: Optional[float] = None
    median_intensity: Optional[float] = None


@dataclass(frozen=True)
class CaptureRun:
    """Rows [start, stop) of one split all belonging to `capture`."""
    capture: int
    start: int
    stop: int

    @property
    def n(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class GridCoordinates:
    """Per-row grid position for one split (aligned with its rows)."""
    ordinal: np.ndarray                                  # int64, position within the capture's run
    r: np.ndarray                                        # int64 grid row, -1 without a grid
    c: np.ndarray                                        # int64 grid column, -1 without a grid
    has_grid: np.ndarray                                 # bool
    grid_shape: Dict[int, Optional[Tuple[int, int]]]     # capture -> (rows, cols), or None
    runs: Tuple[CaptureRun, ...]


def resolve_build_dirs(path: PathLike) -> Tuple[Path, Path]:
    """(data_dir, build_root) from either of them."""
    p = Path(path)
    if (p / MANIFEST_NAME).is_file() or (p / "X_train.npy").is_file():
        return p, p.parent
    if (p / DATA_SUBDIR / MANIFEST_NAME).is_file() or (p / DATA_SUBDIR / "X_train.npy").is_file():
        return p / DATA_SUBDIR, p
    raise FileNotFoundError(f"{p} is not a prepared build: no {MANIFEST_NAME} or X_train.npy "
                            f"in it or in {p / DATA_SUBDIR}")


def spectral_gaps(wavelengths: Sequence[float], min_ratio: float = GAP_MIN_RATIO) -> List[int]:
    """Indices i where the step wavelengths[i] -> wavelengths[i + 1] is a gap
    (more than `min_ratio` x the median step). Interpolation, curvature and
    plotted lines must not cross these."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    if wl.size < 3:
        return []
    d = np.diff(wl)
    med = float(np.median(d))
    return [int(i) for i in np.flatnonzero(d > min_ratio * med)] if med > 0 else []


def spectral_segments(wavelengths: Sequence[float], min_ratio: float = GAP_MIN_RATIO) -> List[Tuple[int, int]]:
    """Contiguous band ranges [start, stop) between gaps."""
    n = len(wavelengths)
    cuts = [g + 1 for g in spectral_gaps(wavelengths, min_ratio)]
    edges = [0, *cuts, n]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class BuildAdapter:
    """Generic `prepare_*` build. Subclasses set the dataset facts as class
    attributes; everything else is read from the build's own files (cached)."""

    name = "base"
    GRID_COLS: Optional[int] = None            # patches per grid row; None = no coordinates
    STRIDE: Optional[int] = None               # default when the manifest does not record it
    MINORITY_CLASS_NAMES: Tuple[str, ...] = ()
    BUILD_SIDECARS: Dict[str, str] = {}        # name -> file under the build root

    def __init__(self, data_dir: PathLike, *, grid_cols: Optional[int] = None):
        self.data_dir, self.build_root = resolve_build_dirs(data_dir)
        self._grid_cols_override = grid_cols

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.data_dir)!r})"

    # -- paths -------------------------------------------------------------------
    def array_path(self, kind: str, split: str) -> Path:
        if kind not in ARRAY_KINDS:
            raise ValueError(f"unknown array kind {kind!r}; expected {ARRAY_KINDS}")
        return self.data_dir / f"{kind}_{canonical_split(split)}.npy"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / MANIFEST_NAME

    @property
    def capture_index_path(self) -> Path:
        return self.data_dir / CAPTURE_INDEX_NAME

    @property
    def class_names_path(self) -> Path:
        return self.data_dir / CLASS_NAMES_NAME

    @property
    def wavelengths_path(self) -> Path:
        return self.data_dir / WAVELENGTHS_NAME

    def sidecar_path(self, name: str) -> Path:
        if name not in self.BUILD_SIDECARS:
            raise KeyError(f"{self.name} has no sidecar {name!r}; known: {sorted(self.BUILD_SIDECARS)}")
        return self.build_root / self.BUILD_SIDECARS[name]

    def files(self) -> Dict[str, Path]:
        """Every file the adapter knows about, by role (existing or not)."""
        out = {"manifest": self.manifest_path, "capture_index": self.capture_index_path,
               "class_names": self.class_names_path, "wavelengths": self.wavelengths_path}
        for split in SPLITS:
            for kind in ARRAY_KINDS:
                out[f"{kind}_{split}"] = self.array_path(kind, split)
        out.update({n: self.sidecar_path(n) for n in self.BUILD_SIDECARS})
        return out

    # -- small metadata (cached) ---------------------------------------------------
    @staticmethod
    def _read_json(p: Path) -> Any:
        if not p.is_file():
            raise FileNotFoundError(f"{p} is missing")
        with open(p) as f:
            return json.load(f)

    def sidecar(self, name: str) -> Any:
        """A build-root sidecar: parsed JSON, or the array of a `.npy`."""
        p = self.sidecar_path(name)
        if p.suffix == ".npy":
            if not p.is_file():
                raise FileNotFoundError(f"{p} is missing")
            return np.load(p)
        return self._read_json(p)

    @cached_property
    def manifest(self) -> Dict[str, Any]:
        return self._read_json(self.manifest_path)

    @cached_property
    def class_names(self) -> Tuple[str, ...]:
        return tuple(self._read_json(self.class_names_path))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_index(self, name: str) -> int:
        """Case-insensitive: capture_index.json spells 'Healthy', class_names.json 'healthy'."""
        lowered = [c.lower() for c in self.class_names]
        try:
            return lowered.index(str(name).lower())
        except ValueError:
            raise KeyError(f"{name!r} is not a class; classes are {list(self.class_names)}") from None

    @property
    def minority_classes(self) -> Tuple[int, ...]:
        """The declared minority class(es) as indices (DCIS -> (1,) here). This is
        the dataset fact; `training.minority_augmentation.resolve_minority_classes
        ('auto')` derives the same from the train counts."""
        return tuple(self.class_index(n) for n in self.MINORITY_CLASS_NAMES)

    @cached_property
    def wavelengths(self) -> np.ndarray:
        if not self.wavelengths_path.is_file():
            raise FileNotFoundError(f"{self.wavelengths_path} is missing")
        return np.load(self.wavelengths_path)

    def spectral_gaps(self, min_ratio: float = GAP_MIN_RATIO) -> List[int]:
        return spectral_gaps(self.wavelengths, min_ratio)

    def spectral_segments(self, min_ratio: float = GAP_MIN_RATIO) -> List[Tuple[int, int]]:
        return spectral_segments(self.wavelengths, min_ratio)

    @cached_property
    def capture_index(self) -> Dict[int, CaptureInfo]:
        raw = self._read_json(self.capture_index_path)
        out: Dict[int, CaptureInfo] = {}
        for k, v in raw.items():
            tissue = str(v.get("tissue", ""))
            try:
                label: Optional[int] = self.class_index(tissue)
            except KeyError:
                label = None
            out[int(k)] = CaptureInfo(
                index=int(k), key=str(v.get("key", "")), patient=int(v["patient"]), tissue=tissue,
                label=label, capture=str(v.get("capture", "")), split=canonical_split(v["split"]),
                n=int(v["n"]), gain=None if v.get("gain") is None else float(v["gain"]),
                median_intensity=None if v.get("median_intensity") is None else float(v["median_intensity"]))
        return dict(sorted(out.items()))

    def captures_in_split(self, split: str) -> List[CaptureInfo]:
        s = canonical_split(split)
        return [c for c in self.capture_index.values() if c.split == s]

    def array_header(self, kind: str, split: str) -> Tuple[Tuple[int, ...], np.dtype]:
        """(shape, dtype) from the `.npy` header only; no rows are read."""
        p = self.array_path(kind, split)
        if not p.is_file():
            raise FileNotFoundError(f"{p} is missing")
        a = np.load(p, mmap_mode="r")
        return tuple(int(s) for s in a.shape), a.dtype

    def n_rows(self, split: str) -> int:
        return self.array_header("y", split)[0][0]

    @cached_property
    def x_shape(self) -> Tuple[int, ...]:
        """Header shape of X_train, (N, H, W, C) for these NHWC builds."""
        return self.array_header("X", "train")[0]

    @property
    def patch_size(self) -> int:
        return int(self.x_shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.x_shape[-1])

    @property
    def stride(self) -> int:
        cli = (self.manifest.get("extra") or {}).get("cli_args") or {}
        if cli.get("stride") is not None:
            return int(cli["stride"])
        return int(self.STRIDE) if self.STRIDE is not None else self.patch_size

    @property
    def grid_cols(self) -> Optional[int]:
        """Constructor override > manifest `extra.grid_cols` > the class constant."""
        if self._grid_cols_override is not None:
            return int(self._grid_cols_override)
        extra = self.manifest.get("extra") or {}
        if extra.get("grid_cols") is not None:
            return int(extra["grid_cols"])
        return self.GRID_COLS

    # -- the capture grid ------------------------------------------------------------
    def grid_shape(self, n: int) -> Optional[Tuple[int, int]]:
        """(rows, cols) for a capture of `n` patches, or None if `n` is not a
        whole number of grid rows."""
        cols = self.grid_cols
        if not cols or n <= 0 or n % cols:
            return None
        return n // cols, cols

    @staticmethod
    def capture_runs(captures: np.ndarray) -> List[CaptureRun]:
        """Maximal runs of equal values in `captures`, in row order."""
        cap = np.asarray(captures)
        if cap.size == 0:
            return []
        starts = np.flatnonzero(np.r_[True, cap[1:] != cap[:-1]])
        stops = np.r_[starts[1:], cap.size]
        return [CaptureRun(int(cap[a]), int(a), int(b)) for a, b in zip(starts, stops)]

    @classmethod
    def non_contiguous_captures(cls, captures: np.ndarray) -> List[int]:
        """Captures whose rows form more than one run (empty list = all contiguous)."""
        counts = Counter(r.capture for r in cls.capture_runs(captures))
        return sorted(c for c, k in counts.items() if k > 1)

    def coordinates(self, captures: np.ndarray) -> GridCoordinates:
        """Per-row (ordinal, r, c) for a split, from its `captures` sidecar.
        Raises NonContiguousCaptureError if any capture's rows are not one run."""
        cap = np.asarray(captures)
        runs = self.capture_runs(cap)
        broken = self.non_contiguous_captures(cap)
        if broken:
            raise NonContiguousCaptureError(f"captures {broken[:10]} are not contiguous; "
                                            f"grid coordinates are undefined")
        n = int(cap.size)
        starts = np.array([r.start for r in runs], dtype=np.int64)
        lengths = np.array([r.n for r in runs], dtype=np.int64)
        ordinal = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
        has_grid = np.zeros(n, dtype=bool)
        grid_shape: Dict[int, Optional[Tuple[int, int]]] = {}
        for run in runs:
            gs = self.grid_shape(run.n)
            grid_shape[run.capture] = gs
            if gs is not None:
                has_grid[run.start:run.stop] = True
        cols = self.grid_cols or 1
        r = np.where(has_grid, ordinal // cols, -1)
        c = np.where(has_grid, ordinal % cols, -1)
        return GridCoordinates(ordinal=ordinal, r=r, c=c, has_grid=has_grid, grid_shape=grid_shape,
                               runs=tuple(runs))

    # -- summary -------------------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """Small JSON-able summary for logs and reports (headers and JSON only)."""
        return {"adapter": self.name, "data_dir": str(self.data_dir), "build_root": str(self.build_root),
                "class_names": list(self.class_names), "minority_classes": list(self.minority_classes),
                "n_bands": self.n_bands, "patch_size": self.patch_size, "stride": self.stride,
                "grid_cols": self.grid_cols, "wavelength_gaps": self.spectral_gaps(),
                "n_rows": {s: self.n_rows(s) for s in SPLITS if self.array_path("y", s).is_file()},
                "n_captures": dict(Counter(c.split for c in self.capture_index.values()))}
