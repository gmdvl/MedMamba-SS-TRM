# -*- coding: utf-8 -*-
"""
hsi_augment/io.py
=================
Reading and writing the build without running out of RAM, disk, or rigour.

Why each piece exists
---------------------
* `iter_chunks`: `X_train.npy` is 19 GB of float16 and the host has 32 GB of
  RAM, so a full `X_*` is never loaded (Plan rule 5). Rows are read from the
  memmap in chunks of `DEFAULT_CHUNK_ROWS` (20,000 rows = 155 MB of float16 at
  11x11x32). A row subset is read in SORTED order, so the page cache sees
  forward sequential reads; a chunk whose rows are contiguous is read as a
  slice, which is much faster than fancy indexing.
* `RunningBandStats`: exact per-band count/mean/variance/min/max in float64,
  merged across chunks with the Chan et al. pairwise update. A float16 chunk is
  widened to float64 in small row blocks, so even a 20k-row chunk never
  materialises as 620 MB of float64.
* `SplitGuard`: "test is sealed; val is reserved for the classifier" (Plan
  rule 3) enforced where the data is read, not by convention. Every split is
  opened through it with a declared purpose; val and test are refused for every
  purpose except P1's structural checks and the final evaluation; every access,
  allowed or refused, is appended to `split_access_log.jsonl`.
* `save_npy` / `save_json` / `npy_writer`: atomic writes (temp file, fsync,
  `os.replace`) via `training.npy_atomic` and `training.dataset_prep_unify`, so
  an interrupted stage leaves no half-written file under its final name.
* `disk_preflight`: `/data` is 98% full. Any write over 1 GB first checks
  `shutil.disk_usage`, so a stage fails before writing instead of half-way.
* `ram_guard`: refuses work whose projected peak is above 50% of available RAM
  (Plan §Memory and time), using `training.system_memory.get_ram_info` (psutil
  is not installed; that function falls back to /proc/meminfo).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple, Union)

import numpy as np

from hsi_augment.config import to_jsonable
from training.npy_atomic import save_json_atomic, save_npy_atomic

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.adapters.base import BuildAdapter, GridCoordinates

PathLike = Union[str, os.PathLike]

DEFAULT_CHUNK_ROWS = 20_000
# Values (not rows) widened to float64 at a time inside RunningBandStats:
# 2**22 values = 32 MB, a few temporaries of that size at most.
STATS_BLOCK_VALUES = 1 << 22

DISK_PREFLIGHT_BYTES = 1_000_000_000      # "any write over 1 GB"
DISK_HEADROOM_BYTES = 1_000_000_000       # left free after the write
RAM_FRACTION = 0.5                        # of AVAILABLE RAM

SPLITS = ("train", "val", "test")
ARRAY_KINDS = ("X", "y", "groups", "captures")
ACCESS_LOG_NAME = "split_access_log.jsonl"


class ResourceError(RuntimeError):
    """Not enough disk or RAM for the requested work."""


class SplitAccessError(PermissionError):
    """A split was opened for a purpose that may not read it."""


# ============================================================================
# SplitGuard
# ============================================================================

# purpose -> splits it may read. val/test only for P1's structural checks and
# the final evaluation (Plan rules 3-4); everything that fits, selects,
# generates or quality-controls reads train only. Extending this table is a
# deliberate, reviewed change: it is the leakage policy.
PURPOSE_SPLITS: Dict[str, Tuple[str, ...]] = {
    "structural": SPLITS,     # P1: counts, NaN/Inf, hashes, header/provenance consistency
    "final_eval": SPLITS,     # P10 report, only after selection.json is frozen
    "fit": ("train",),        # any fitted quantity: norm/log stats, tau, s_glass, sigma_b, PCA
    "describe": ("train",),   # descriptive train tables/figures: index, imbalance, mosaics
    "pool": ("train",),       # P4 generator pool and LOPO folds
    "generator": ("train",),  # P6/P7 generator training and sampling
    "qc": ("train",),         # P8b QC references, thresholds and classifiers
    "lopo": ("train",),       # P8c LOPO utility test
}
PURPOSE_DOCS: Dict[str, str] = {
    "structural": "P1 integrity: counts, NaN/Inf, hashes, provenance consistency (any split)",
    "final_eval": "P10 final evaluation after selection.json exists (any split)",
    "fit": "fitting statistics, thresholds and feature spaces (train only)",
    "describe": "descriptive tables and figures of the train split (train only)",
    "pool": "generator pool and LOPO folds (train only)",
    "generator": "generator training and sampling (train only)",
    "qc": "QC references, thresholds and classifiers (train only)",
    "lopo": "LOPO utility test (train only)",
}
TRAIN_ONLY_PURPOSES: Tuple[str, ...] = tuple(p for p, s in PURPOSE_SPLITS.items() if s == ("train",))
HELD_OUT_PURPOSES: Tuple[str, ...] = tuple(p for p, s in PURPOSE_SPLITS.items() if s != ("train",))


def canonical_split(split: str) -> str:
    """'validation' (the spelling in capture_index.json / split_assignment.json)
    and 'val' (the spelling in the array file names) are the same split."""
    s = str(split).strip().lower()
    s = {"validation": "val", "valid": "val"}.get(s, s)
    if s not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return s


@dataclass
class SplitArrays:
    """One split, opened read-only. Every array is a memmap (`mmap_mode='r'`),
    so opening costs no RAM; read rows with `iter_chunks`. `None` for a kind
    that was not requested or whose file is missing."""
    split: str
    purpose: str
    X: Optional[np.ndarray]
    y: Optional[np.ndarray]
    groups: Optional[np.ndarray]
    captures: Optional[np.ndarray]
    adapter: "BuildAdapter" = field(repr=False)

    @property
    def n(self) -> int:
        for a in (self.y, self.captures, self.groups, self.X):
            if a is not None:
                return int(a.shape[0])
        return 0

    def coordinates(self) -> "GridCoordinates":
        """Per-row (ordinal, r, c) on the capture grid; see
        `BuildAdapter.coordinates`. Needs the `captures` sidecar."""
        if self.captures is None:
            raise ValueError(f"{self.split}: captures sidecar not opened/present")
        return self.adapter.coordinates(np.asarray(self.captures))


class SplitGuard:
    """The only way `hsi_augment` code reads a split's arrays.

        guard = SplitGuard(adapter, out_root / "split_access_log.jsonl", stage="tissue-mask",
                           allowed_purposes=("fit",))
        train = guard.open("train", purpose="fit")        # ok
        guard.open("val", purpose="fit")                   # SplitAccessError

    `allowed_purposes` is what the stage declared at registration
    (`StageSpec.purposes`): a stage cannot pick a val/test-capable purpose at
    the call site unless it declared it, which keeps the policy reviewable in
    one line per stage. `None` allows every purpose (for scripts outside the
    stage runner, which still get the purpose -> split check and the log)."""

    def __init__(self, adapter: "BuildAdapter", log_path: PathLike, *, stage: str = "",
                 allowed_purposes: Optional[Sequence[str]] = None):
        self.adapter = adapter
        self.log_path = Path(log_path)
        self.stage = stage
        if allowed_purposes is not None:
            unknown = sorted(set(allowed_purposes) - set(PURPOSE_SPLITS))
            if unknown:
                raise ValueError(f"unknown purpose(s) {unknown}; known: {sorted(PURPOSE_SPLITS)}")
        self.allowed_purposes = tuple(allowed_purposes) if allowed_purposes is not None else None

    def _log(self, record: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(to_jsonable(record), sort_keys=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
            f.flush()

    def check(self, split: str, purpose: str, *, files: Sequence[str] = ()) -> str:
        """Returns the canonical split name or raises SplitAccessError. Both
        outcomes are logged."""
        split = canonical_split(split)
        reason = None
        if purpose not in PURPOSE_SPLITS:
            reason = f"unknown purpose {purpose!r}; known: {sorted(PURPOSE_SPLITS)}"
        elif self.allowed_purposes is not None and purpose not in self.allowed_purposes:
            reason = (f"stage {self.stage!r} did not declare purpose {purpose!r} "
                      f"(declared: {list(self.allowed_purposes)})")
        elif split not in PURPOSE_SPLITS[purpose]:
            reason = (f"purpose {purpose!r} may read only {list(PURPOSE_SPLITS[purpose])}; "
                      f"{split!r} is reserved (val for the classifier, test for the final report)")
        self._log({"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   "stage": self.stage, "split": split, "purpose": purpose,
                   "allowed": reason is None, "reason": reason, "files": list(files),
                   "data_dir": str(self.adapter.data_dir), "pid": os.getpid()})
        if reason is not None:
            raise SplitAccessError(f"SplitGuard refused {split!r} for {self.stage or 'caller'}: {reason}")
        return split

    def open(self, split: str, purpose: str, *, kinds: Sequence[str] = ARRAY_KINDS) -> SplitArrays:
        bad = sorted(set(kinds) - set(ARRAY_KINDS))
        if bad:
            raise ValueError(f"unknown array kind(s) {bad}; expected {ARRAY_KINDS}")
        split_c = canonical_split(split)
        paths = {k: self.adapter.array_path(k, split_c) for k in kinds}
        split_c = self.check(split_c, purpose, files=[p.name for p in paths.values()])
        arrays: Dict[str, Optional[np.ndarray]] = {k: None for k in ARRAY_KINDS}
        for k, p in paths.items():
            if p.is_file():
                arrays[k] = np.load(p, mmap_mode="r")
            elif k in ("X", "y"):
                raise FileNotFoundError(f"{p} is missing")
        return SplitArrays(split=split_c, purpose=purpose, adapter=self.adapter, **arrays)


def read_access_log(path: PathLike) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================================
# chunked reads
# ============================================================================

class Chunk(NamedTuple):
    rows: np.ndarray   # int64 row numbers, ascending
    data: np.ndarray   # arr[rows]; may be a read-only view of the memmap


def _as_sorted_index(index: Any, n: int, assume_sorted: bool) -> np.ndarray:
    idx = np.asarray(index)
    if idx.dtype == bool:
        if idx.shape != (n,):
            raise ValueError(f"boolean index has shape {idx.shape}, expected ({n},)")
        return np.flatnonzero(idx).astype(np.int64)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("index must be a 1-D integer or boolean array")
    idx = idx.astype(np.int64, copy=False)
    if len(idx) and (idx.min() < 0 or idx.max() >= n):
        raise IndexError(f"index out of range for {n} rows")
    if not assume_sorted:
        idx = np.sort(idx, kind="stable")
    return idx


def iter_index_chunks(index: Any, n: int, *, chunk_rows: int = DEFAULT_CHUNK_ROWS,
                      assume_sorted: bool = False) -> Iterator[np.ndarray]:
    """Sorted row numbers in pieces of at most `chunk_rows`. `index=None` means
    every row. Duplicates are kept (an oversampling index reads a row twice)."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")
    if index is None:
        for a in range(0, n, chunk_rows):
            yield np.arange(a, min(a + chunk_rows, n), dtype=np.int64)
        return
    idx = _as_sorted_index(index, n, assume_sorted)
    for a in range(0, len(idx), chunk_rows):
        yield idx[a:a + chunk_rows]


def iter_chunks(arr: np.ndarray, index: Any = None, *, chunk_rows: int = DEFAULT_CHUNK_ROWS,
                dtype: Optional[np.dtype] = None, assume_sorted: bool = False) -> Iterator[Chunk]:
    """`Chunk(rows, arr[rows])` over `index` (default: all rows) in sorted order.

    A chunk whose rows are contiguous and unique is read as a slice; otherwise
    with a sorted fancy index. `dtype` casts each chunk (a copy); without it a
    sliced chunk is a read-only view of the memmap, so copy before modifying.
    Rows come back SORTED even if `index` was not: use `Chunk.rows` to map back."""
    n = int(arr.shape[0])
    for rows in iter_index_chunks(index, n, chunk_rows=chunk_rows, assume_sorted=assume_sorted):
        if len(rows) == 0:
            continue
        a, b = int(rows[0]), int(rows[-1]) + 1
        # The span check alone is not enough: sorted [3, 3, 5] also spans 3 rows.
        if b - a == len(rows) and (len(rows) == 1 or bool(np.all(np.diff(rows) == 1))):
            data = arr[a:b]
        else:
            data = arr[rows]
        if dtype is not None:
            data = np.asarray(data, dtype=dtype)
        yield Chunk(rows, data)


# ============================================================================
# float64 running statistics
# ============================================================================

class RunningBandStats:
    """Exact per-band statistics over any number of chunks, channel-last.

        stats = RunningBandStats(n_bands=32)
        for ch in iter_chunks(X, idx):
            stats.update(ch.data)            # (rows, H, W, C) -> every pixel of every row
        stats.mean, stats.std(), stats.to_dict()

    Count, mean and M2 are float64 and merged with the pairwise update of Chan,
    Golub & LeVeque (1979), which is exact up to rounding however the data is
    split; `merge` combines two accumulators the same way (e.g. per-class into
    per-split). min/max are tracked too. Non-finite values are COUNTED
    (`n_nonfinite`) and EXCLUDED from every other statistic, so a single NaN
    is reported instead of silently turning a band's mean into NaN."""

    def __init__(self, n_bands: int):
        self.n_bands = int(n_bands)
        self.count = np.zeros(self.n_bands, dtype=np.float64)
        self.mean = np.zeros(self.n_bands, dtype=np.float64)
        self.m2 = np.zeros(self.n_bands, dtype=np.float64)
        self.min = np.full(self.n_bands, np.inf, dtype=np.float64)
        self.max = np.full(self.n_bands, -np.inf, dtype=np.float64)
        self.n_nonfinite = np.zeros(self.n_bands, dtype=np.int64)

    # -- accumulation ---------------------------------------------------------
    def _merge_moments(self, n_b: np.ndarray, mean_b: np.ndarray, m2_b: np.ndarray) -> None:
        n_a = self.count
        n = n_a + n_b
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = mean_b - self.mean
            frac = np.where(n > 0, n_b / np.where(n > 0, n, 1.0), 0.0)
            self.mean = self.mean + delta * frac
            self.m2 = self.m2 + m2_b + delta * delta * n_a * frac
        self.count = n

    def update(self, x: np.ndarray) -> "RunningBandStats":
        x = np.asarray(x)
        if x.shape[-1] != self.n_bands:
            raise ValueError(f"last axis has {x.shape[-1]} bands, expected {self.n_bands}")
        flat = x.reshape(-1, self.n_bands)
        step = max(1, STATS_BLOCK_VALUES // self.n_bands)
        for a in range(0, flat.shape[0], step):
            b = flat[a:a + step].astype(np.float64)
            finite = np.isfinite(b)
            if finite.all():
                n_b = np.full(self.n_bands, float(b.shape[0]))
                mean_b = b.mean(axis=0)
                m2_b = ((b - mean_b) ** 2).sum(axis=0)
                lo, hi = b.min(axis=0), b.max(axis=0)
            else:
                self.n_nonfinite += (~finite).sum(axis=0)
                n_b = finite.sum(axis=0).astype(np.float64)
                safe = np.where(finite, b, 0.0)
                mean_b = safe.sum(axis=0) / np.maximum(n_b, 1.0)
                m2_b = (np.where(finite, safe - mean_b, 0.0) ** 2).sum(axis=0)
                lo = np.where(finite, b, np.inf).min(axis=0)
                hi = np.where(finite, b, -np.inf).max(axis=0)
            self._merge_moments(n_b, mean_b, m2_b)
            self.min = np.minimum(self.min, lo)
            self.max = np.maximum(self.max, hi)
        return self

    def merge(self, other: "RunningBandStats") -> "RunningBandStats":
        if other.n_bands != self.n_bands:
            raise ValueError("band counts differ")
        self._merge_moments(other.count.copy(), other.mean.copy(), other.m2.copy())
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)
        self.n_nonfinite = self.n_nonfinite + other.n_nonfinite
        return self

    # -- results ----------------------------------------------------------------
    def var(self, ddof: int = 0) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > ddof, self.m2 / np.maximum(self.count - ddof, 1.0), np.nan)

    def std(self, ddof: int = 0) -> np.ndarray:
        return np.sqrt(self.var(ddof))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able; floats round-trip exactly through `json`."""
        return {"n_bands": self.n_bands, "count": self.count.tolist(), "mean": self.mean.tolist(),
                "var": self.var().tolist(), "std": self.std().tolist(), "m2": self.m2.tolist(),
                "min": self.min.tolist(), "max": self.max.tolist(),
                "n_nonfinite": self.n_nonfinite.tolist()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunningBandStats":
        s = cls(int(d["n_bands"]))
        s.count = np.asarray(d["count"], dtype=np.float64)
        s.mean = np.asarray(d["mean"], dtype=np.float64)
        s.m2 = np.asarray(d["m2"], dtype=np.float64)
        s.min = np.asarray(d["min"], dtype=np.float64)
        s.max = np.asarray(d["max"], dtype=np.float64)
        s.n_nonfinite = np.asarray(d.get("n_nonfinite", [0] * s.n_bands), dtype=np.int64)
        return s


def band_stats(arr: np.ndarray, index: Any = None, *,
               chunk_rows: int = DEFAULT_CHUNK_ROWS) -> RunningBandStats:
    """Per-band float64 stats of `arr[index]` (all rows by default), chunked."""
    stats = RunningBandStats(int(arr.shape[-1]))
    for ch in iter_chunks(arr, index, chunk_rows=chunk_rows):
        stats.update(ch.data)
    return stats


# ============================================================================
# resource guards
# ============================================================================

def _fmt_bytes(n: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{int(n)} B"


def _existing_ancestor(path: Path) -> Path:
    p = Path(os.path.abspath(path))
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def disk_preflight(path: PathLike, nbytes: int, *, threshold: int = DISK_PREFLIGHT_BYTES,
                   headroom: int = DISK_HEADROOM_BYTES,
                   usage_fn: Callable[[str], Any] = shutil.disk_usage) -> Optional[Dict[str, Any]]:
    """Before writing `nbytes` to `path`: writes at or below `threshold` (1 GB)
    pass unchecked (None); larger ones need `nbytes + headroom` free on the
    target filesystem, else ResourceError. Returns what it checked."""
    nbytes = int(nbytes)
    if nbytes <= threshold:
        return None
    anchor = _existing_ancestor(Path(path))
    free = int(usage_fn(str(anchor)).free)
    need = nbytes + int(headroom)
    report = {"path": str(path), "bytes": nbytes, "free": free, "need": need}
    if free < need:
        raise ResourceError(f"disk preflight: writing {_fmt_bytes(nbytes)} to {path} needs "
                            f"{_fmt_bytes(need)} free (incl. {_fmt_bytes(headroom)} headroom), "
                            f"only {_fmt_bytes(free)} free on {anchor}")
    return report


def ram_guard(projected_bytes: int, *, what: str = "work", fraction: float = RAM_FRACTION,
              ram_info: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Refuse (ResourceError) work whose projected PEAK is above `fraction` of
    the RAM available right now. `ram_info` overrides the measurement (tests).
    If availability cannot be measured the work is allowed, and the returned
    dict says `checked: False`."""
    if ram_info is None:
        # Lazy: training.system_memory imports torch at module level.
        from training.system_memory import get_ram_info
        ram_info = get_ram_info()
    avail_mb = ram_info.get("available_mb")
    report = {"what": what, "projected_bytes": int(projected_bytes), "fraction": fraction,
              "available_mb": avail_mb, "checked": avail_mb is not None}
    if avail_mb is None:
        return report
    limit = fraction * float(avail_mb) * 1e6
    report["limit_bytes"] = int(limit)
    if projected_bytes > limit:
        raise ResourceError(f"RAM guard: {what} projects a peak of {_fmt_bytes(projected_bytes)}, above "
                            f"{fraction:.0%} of the {_fmt_bytes(float(avail_mb) * 1e6)} available "
                            f"({_fmt_bytes(limit)}). Lower the chunk/sample size.")
    return report


# ============================================================================
# atomic writes and hashing
# ============================================================================

def sha256_file(path: PathLike, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: PathLike, obj: Any) -> Path:
    """Atomic JSON write; numpy scalars/arrays become plain JSON numbers/lists
    (not the strings `save_json_atomic`'s `default=str` would give)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    save_json_atomic(str(p), to_jsonable(obj))
    return p


def save_npy(path: PathLike, array: np.ndarray) -> Path:
    """Atomic `.npy` write, with the disk preflight when it is over 1 GB."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    disk_preflight(p, arr.nbytes + 128)
    save_npy_atomic(str(p), arr)
    return p


class NpyStreamWriter:
    """Appends row blocks to a `.npy` being written by `npy_writer`."""

    def __init__(self, fh, holder, dtype: np.dtype, shape: Tuple[int, ...]):
        self._fh, self._holder = fh, holder
        self.dtype, self.shape = np.dtype(dtype), tuple(shape)
        self.rows_written = 0
        self.sha256: Optional[str] = None   # set when the file is committed

    def write(self, block: np.ndarray) -> None:
        arr = np.ascontiguousarray(block, dtype=self.dtype)
        if arr.shape[1:] != self.shape[1:]:
            raise ValueError(f"block rows have shape {arr.shape[1:]}, expected {self.shape[1:]}")
        if self.rows_written + arr.shape[0] > self.shape[0]:
            raise ValueError(f"writing past the declared {self.shape[0]} rows")
        buf = memoryview(arr).cast("B")
        self._fh.write(buf)
        if self._holder.hasher is not None:
            self._holder.hasher.update(buf)
        self.rows_written += int(arr.shape[0])


@contextlib.contextmanager
def npy_writer(path: PathLike, dtype: Any, shape: Sequence[int], *,
               sha256: bool = True) -> Iterator[NpyStreamWriter]:
    """Stream a large array to `path` front to back, block by block, without
    holding it in RAM:

        with npy_writer(out / "X_pool.npy", np.float16, (n, 11, 11, 32)) as w:
            for ch in iter_chunks(X, rows):
                w.write(transform(ch.data))
        w.sha256   # of the whole file, computed while writing (no read-back)

    Disk preflight first; then `training.dataset_prep_unify.atomic_npy_stream`
    (temp file, length check, fsync, `os.replace`). If fewer rows than declared
    were written, or anything raises, nothing appears under `path`."""
    from training.dataset_prep_unify import atomic_npy_stream
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    disk_preflight(p, int(np.prod(shape)) * dtype.itemsize + 128)
    with atomic_npy_stream(str(p), dtype, shape, sha256=sha256) as (fh, holder):
        writer = NpyStreamWriter(fh, holder, dtype, shape)
        yield writer
    writer.sha256 = holder.sha256
