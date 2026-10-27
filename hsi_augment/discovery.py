# -*- coding: utf-8 -*-
"""
hsi_augment/discovery.py
========================
Phase 1 (#11, Plan §Phase 1): dataset discovery and validation. Every later
stage trusts the build this one has checked, so it reads all of it once and
writes down what it found. The stage module is `hsi_augment/stages/discovery.py`;
this module holds the logic.

What it writes (stage directory `prep/validate/`)
-------------------------------------------------
discovery_report.json   the inventory: class names, wavelengths and their gap,
                        the manifest, every `.npy` header (shape, dtype, size),
                        the JSON sidecars (size, sha256), the capture index,
                        split assignment and Pass-1 scope, per-split row,
                        class, capture and grid counts, and the content scan
                        (per-band min/max/mean/std/percentiles per split and
                        per class, defect counts), the duplicate-patch scan and
                        the process's peak RSS.
integrity_report.json   one entry per named check:
                        {status: pass | warn | fail | skip, message, details}.

A "fail" is a hard failure. Both reports are written FIRST and the stage then
raises `StageError` (the CLI exits 1), so a failed run leaves its evidence in
the stage directory next to `failed.json`, and no `done.json`. A "warn" is
recorded and does not stop the pipeline (a partial scan, a within-split
duplicate, a float16-saturated value).

Why it is built this way
------------------------
* Val and test are read for structure only (Plan rules 3-4). Every split is
  opened through the stage's SplitGuard with purpose "structural", and the
  manifest sha256 pass is announced to the guard (`SplitGuard.check`) before it
  hashes the files, so `split_access_log.jsonl` names every held-out file this
  stage touched. The val/test band statistics in the report are structural
  checks; nothing downstream may fit on them (`content.splits.<s>.fit_allowed`).
* Exact percentiles in bounded memory. X is float16, which has only 65,536 bit
  patterns, so a per-band histogram over the patterns IS the value
  distribution: 16 MB per (split, class) at 32 bands, mergeable across chunks
  and classes. p0.05 / p99.95 therefore come out exactly (numpy's
  "inverted_cdf" order statistic over the finite values), not from a sample.
  A build stored in another float dtype is rounded to float16 first, and the
  report says so.
* Duplicate patches through the guard, over every row.
  `training.dataset_integrity.check_patch_leakage` opens the files itself (no
  access log) and hashes the FIRST N rows of each split, which on trainsel is
  about 10 of the 499 train captures. Here each row the content scan reads is
  sha1-hashed as it goes by (about 10 s on trainsel, 20 bytes per row), so the
  cross-split check is exhaustive at no extra I/O. `leakage_rows=N` hashes a
  seeded random sample instead. The all-zero patch is reported on its own:
  two splits sharing it is not leakage.
* Page release. A memmap scan of the 19 GB X_train leaves every page it
  touched resident in this process, so the peak RSS would read as ~19 GB of
  reclaimable page cache and say nothing about the plan's "peak RAM < 2 GB".
  Each chunk's pages are dropped (`madvise(MADV_DONTNEED)`) once processed, so
  the reported `peak_rss_mb` is the working set.
* Contiguity is checked vectorised (`capture_runs_summary`). The adapter's
  `capture_runs` builds one Python object per run: fine for 499 runs, not for
  a shuffled split with millions.
"""

from __future__ import annotations

import hashlib
import logging
import math
import mmap
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hsi_augment import SCHEMA_VERSION
from hsi_augment.io import ARRAY_KINDS, SPLITS, STATS_BLOCK_VALUES, RunningBandStats, iter_chunks, sha256_file
from hsi_augment.stage import StageError

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.adapters.base import BuildAdapter, CaptureInfo
    from hsi_augment.stage import StageContext

DISCOVERY_REPORT = "discovery_report.json"
INTEGRITY_REPORT = "integrity_report.json"

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
STATUSES = (PASS, WARN, FAIL, SKIP)

F16_MAX = 65504.0                  # largest finite float16: a value clipped at the dtype's ceiling
N_EXAMPLES = 10                    # offending rows / captures listed per check and split
DIGEST_DTYPE = np.dtype("V20")     # raw sha1 digests, as check_patch_leakage hashes rows

_F16_PATTERNS = 1 << 16


# ============================================================================
# checks
# ============================================================================

@dataclass
class Check:
    name: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "message": self.message, "details": self.details}


class CheckLog:
    """Named checks in the order they ran. `on_add(check)` mirrors each one to
    the console as it is recorded, so a failure shows before the long scan."""

    def __init__(self, on_add: Optional[Callable[[Check], None]] = None):
        self.checks: Dict[str, Check] = {}
        self._on_add = on_add

    def add(self, name: str, status: str, message: str, **details: Any) -> Check:
        if status not in STATUSES:
            raise ValueError(f"unknown check status {status!r}")
        if name in self.checks:
            raise ValueError(f"check {name!r} recorded twice")
        check = Check(name, status, message, details)
        self.checks[name] = check
        if self._on_add is not None:
            self._on_add(check)
        return check

    def verdict(self, name: str, ok: str, *, fail: Sequence[str] = (), warn: Sequence[str] = (),
                **details: Any) -> Check:
        """FAIL if `fail` lists a problem, else WARN if `warn` does, else PASS with `ok`."""
        problems = list(fail) or list(warn)
        if not problems:
            return self.add(name, PASS, ok, **details)
        msg = "; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
        return self.add(name, FAIL if fail else WARN, msg, failures=list(fail), warnings=list(warn),
                        **details)

    def skip(self, name: str, why: str, **details: Any) -> Check:
        return self.add(name, SKIP, why, **details)

    def names(self, status: str) -> List[str]:
        return [n for n, c in self.checks.items() if c.status == status]

    @property
    def overall(self) -> str:
        if self.names(FAIL):
            return FAIL
        return WARN if self.names(WARN) else PASS

    def counts(self) -> Dict[str, int]:
        return {s: len(self.names(s)) for s in STATUSES}

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {n: c.to_dict() for n, c in self.checks.items()}


# ============================================================================
# small helpers
# ============================================================================

def _floats(values: Any) -> List[Optional[float]]:
    """A float list for JSON, non-finite values as null (strict JSON has no NaN)."""
    return [float(v) if math.isfinite(v) else None for v in np.asarray(values, dtype=np.float64).ravel()]


def _ints(values: Any) -> List[int]:
    return [int(v) for v in np.asarray(values).ravel()]


def _pkey(percentile: float) -> str:
    return f"p{percentile:g}"            # 0.05 -> "p0.05", 99.95 -> "p99.95"


def _sorted_ints(values: Any) -> List[int]:
    return sorted(int(v) for v in values)


def _file_split(name: str) -> Optional[str]:
    """'X_train.npy' -> 'train'; files that belong to no split -> None."""
    stem = Path(name).stem
    if "_" not in stem:
        return None
    kind, split = stem.rsplit("_", 1)
    return split if kind in ARRAY_KINDS and split in SPLITS else None


def peak_rss_mb() -> Optional[float]:
    """Peak resident set of this process (ru_maxrss: kB on Linux), or None."""
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except (ImportError, OSError, ValueError):  # pragma: no cover
        return None


def release_pages(arr: np.ndarray, start_row: int, stop_row: int) -> bool:
    """Drop this process's resident pages for rows [start_row, stop_row) of a
    read-only `np.memmap` (`madvise(MADV_DONTNEED)`; the data stays in the
    file and re-faults on the next access). Returns False where that is not
    possible (not a memmap, a view of one, no madvise on this platform)."""
    mm = getattr(arr, "_mmap", None)
    if mm is None or not hasattr(mm, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        return False
    if stop_row <= start_row or not arr.flags.c_contiguous:
        return False
    row_bytes = int(arr.strides[0])
    # np.memmap maps from the allocation-granularity boundary below the
    # header, so the array starts `offset % ALLOCATIONGRANULARITY` into the map.
    base = int(getattr(arr, "offset", 0)) % mmap.ALLOCATIONGRANULARITY
    a = base + int(start_row) * row_bytes
    b = min(base + int(stop_row) * row_bytes, len(mm))
    a -= a % mmap.PAGESIZE
    if b <= a:
        return False
    try:
        mm.madvise(mmap.MADV_DONTNEED, a, b - a)
    except (OSError, ValueError):  # pragma: no cover
        return False
    return True


def npy_header(path: Path) -> Dict[str, Any]:
    """Shape, dtype and sizes from the `.npy` header alone. No payload is read
    and no memmap is made, so a truncated file still reports (and says so)."""
    from training.npy_integrity import _read_npy_header
    p = Path(path)
    out: Dict[str, Any] = {"exists": p.is_file()}
    if not out["exists"]:
        return out
    out["size_bytes"] = int(p.stat().st_size)
    try:
        shape, dtype, fortran, offset = _read_npy_header(str(p))
    except Exception as e:
        out["error"] = f"header unreadable ({e})"
        return out
    dtype = np.dtype(dtype)
    expected = int(offset) + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    out.update(shape=[int(s) for s in shape], dtype=str(dtype), fortran_order=bool(fortran),
               header_offset=int(offset), expected_size_bytes=expected,
               size_matches_header=out["size_bytes"] == expected)
    return out


def _header_problem(name: str, head: Mapping[str, Any]) -> Optional[str]:
    if not head.get("exists"):
        return f"{name} is missing"
    if head.get("error"):
        return f"{name}: {head['error']}"
    if not head.get("size_matches_header"):
        got, want = head["size_bytes"], head["expected_size_bytes"]
        return f"{name}: {got} bytes on disk, header implies {want} ({'truncated' if got < want else 'trailing bytes'})"
    return None


# ============================================================================
# exact float16 percentiles
# ============================================================================

def _float16_tables() -> Tuple[np.ndarray, np.ndarray]:
    """The finite float16 bit patterns sorted by value, and those values."""
    values = np.arange(_F16_PATTERNS, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite], kind="stable")]
    return order, values[order]


_F16_ORDER, _F16_SORTED_VALUES = _float16_tables()


def inverted_cdf_rank(n: int, q: float) -> int:
    """0-based rank of the q-quantile among `n` sorted values, computed exactly
    as numpy's `quantile(..., method="inverted_cdf")` does."""
    v = n * q - 1.0
    lo = math.floor(v)
    r = lo if v == lo else lo + 1
    return int(min(max(r, 0), n - 1))


class Float16Histogram:
    """Per-band counts of every float16 bit pattern, (n_bands, 65536) int64.

    float16 has 65,536 bit patterns, so this IS the value distribution, not an
    approximation: any quantile of the finite values is exact, the counts of
    NaN / Inf patterns fall out of it, and two histograms merge by addition.
    Input in another float dtype is rounded to float16 (`rounded` is then
    True); values beyond float16's range become Inf and drop out."""

    def __init__(self, n_bands: int):
        self.n_bands = int(n_bands)
        self.counts = np.zeros((self.n_bands, _F16_PATTERNS), dtype=np.int64)
        self._offsets = np.arange(self.n_bands, dtype=np.int32) * _F16_PATTERNS
        self.rounded = False

    def update(self, x: np.ndarray) -> "Float16Histogram":
        x = np.asarray(x)
        if x.shape[-1] != self.n_bands:
            raise ValueError(f"last axis has {x.shape[-1]} bands, expected {self.n_bands}")
        if x.dtype != np.float16:
            with np.errstate(over="ignore", invalid="ignore"):
                x = x.astype(np.float16)
            self.rounded = True
        bits = np.ascontiguousarray(x).view(np.uint16).reshape(-1, self.n_bands)
        step = max(1, STATS_BLOCK_VALUES // self.n_bands)
        for a in range(0, bits.shape[0], step):
            idx = bits[a:a + step].astype(np.int32)
            idx += self._offsets                     # band b's patterns live at b * 65536 + bits
            self.counts += np.bincount(idx.ravel(), minlength=self.counts.size).reshape(self.counts.shape)
        return self

    def merge(self, other: "Float16Histogram") -> "Float16Histogram":
        if other.n_bands != self.n_bands:
            raise ValueError("band counts differ")
        self.counts += other.counts
        self.rounded = self.rounded or other.rounded
        return self

    def n_finite(self) -> np.ndarray:
        return self.counts[:, _F16_ORDER].sum(axis=1)

    def quantiles(self, qs: Sequence[float]) -> np.ndarray:
        """(len(qs), n_bands) quantiles of the finite values, NaN for a band without any."""
        cum = np.cumsum(self.counts[:, _F16_ORDER], axis=1)
        out = np.full((len(qs), self.n_bands), np.nan)
        for b in range(self.n_bands):
            n = int(cum[b, -1])
            if n == 0:
                continue
            for i, q in enumerate(qs):
                rank = inverted_cdf_rank(n, float(q))
                out[i, b] = _F16_SORTED_VALUES[int(np.searchsorted(cum[b], rank + 1, side="left"))]
        return out


# ============================================================================
# the content scan
# ============================================================================

def _note_rows(store: List[int], rows: np.ndarray) -> None:
    if len(store) < N_EXAMPLES:
        store.extend(int(r) for r in rows[:N_EXAMPLES - len(store)])


class ContentAccumulator:
    """What the content scan keeps for one (split, class): exact float64
    moments and min/max (`RunningBandStats`), the float16 histogram for the
    percentiles, and the defect counts (NaN, +Inf, -Inf, float16-saturated
    values, all-zero patches) with the first offending rows."""

    def __init__(self, n_bands: int):
        self.n_bands = int(n_bands)
        self.stats = RunningBandStats(self.n_bands)
        self.hist = Float16Histogram(self.n_bands)
        self.rows = 0
        self.nan = np.zeros(self.n_bands, dtype=np.int64)
        self.posinf = np.zeros(self.n_bands, dtype=np.int64)
        self.neginf = np.zeros(self.n_bands, dtype=np.int64)
        self.saturated = np.zeros(self.n_bands, dtype=np.int64)
        self.rows_nonfinite = 0
        self.zero_patches = 0
        self.first_nonfinite_rows: List[int] = []
        self.first_saturated_rows: List[int] = []
        self.first_zero_rows: List[int] = []

    def update(self, x: np.ndarray, rows: np.ndarray) -> "ContentAccumulator":
        """`x` (n, H, W, C) and its global row numbers `rows`."""
        x = np.asarray(x)
        rows = np.asarray(rows)
        if len(x) == 0:
            return self
        per_row = int(np.prod(x.shape[1:]))
        step = max(1, STATS_BLOCK_VALUES // per_row)
        for a in range(0, len(x), step):
            blk = np.ascontiguousarray(x[a:a + step])
            r = rows[a:a + step]
            m = len(blk)
            self.stats.update(blk)
            self.hist.update(blk)
            flat = blk.reshape(m, -1, self.n_bands)               # (rows, pixels, bands)
            nan, pinf, ninf = np.isnan(flat), np.isposinf(flat), np.isneginf(flat)
            self.nan += nan.sum(axis=(0, 1))
            self.posinf += pinf.sum(axis=(0, 1))
            self.neginf += ninf.sum(axis=(0, 1))
            bad = (nan | pinf | ninf).any(axis=(1, 2))
            if bad.any():
                self.rows_nonfinite += int(bad.sum())
                _note_rows(self.first_nonfinite_rows, r[bad])
            sat = np.abs(flat) == F16_MAX
            self.saturated += sat.sum(axis=(0, 1))
            sat_rows = sat.any(axis=(1, 2))
            if sat_rows.any():
                _note_rows(self.first_saturated_rows, r[sat_rows])
            zero = ~flat.reshape(m, -1).any(axis=1)                # NaN counts as non-zero
            if zero.any():
                self.zero_patches += int(zero.sum())
                _note_rows(self.first_zero_rows, r[zero])
            self.rows += m
        return self

    def merge(self, other: "ContentAccumulator") -> "ContentAccumulator":
        self.stats.merge(other.stats)
        self.hist.merge(other.hist)
        self.rows += other.rows
        for name in ("nan", "posinf", "neginf", "saturated"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.rows_nonfinite += other.rows_nonfinite
        self.zero_patches += other.zero_patches
        for name in ("first_nonfinite_rows", "first_saturated_rows", "first_zero_rows"):
            setattr(self, name, sorted(getattr(self, name) + getattr(other, name))[:N_EXAMPLES])
        return self

    def to_dict(self, percentiles: Sequence[float]) -> Dict[str, Any]:
        qv = self.hist.quantiles([p / 100.0 for p in percentiles])
        per_band: Dict[str, Any] = {
            "count": _ints(self.stats.count), "min": _floats(self.stats.min), "max": _floats(self.stats.max),
            "mean": _floats(self.stats.mean), "std": _floats(self.stats.std())}
        per_band.update({_pkey(p): _floats(qv[i]) for i, p in enumerate(percentiles)})
        per_band.update({"nan": _ints(self.nan), "posinf": _ints(self.posinf), "neginf": _ints(self.neginf),
                         "float16_saturated": _ints(self.saturated)})
        return {"rows": self.rows, "nan": int(self.nan.sum()), "posinf": int(self.posinf.sum()),
                "neginf": int(self.neginf.sum()), "float16_saturated": int(self.saturated.sum()),
                "rows_with_nonfinite": self.rows_nonfinite, "all_zero_patches": self.zero_patches,
                "first_nonfinite_rows": self.first_nonfinite_rows,
                "first_saturated_rows": self.first_saturated_rows, "first_zero_rows": self.first_zero_rows,
                "per_band": per_band}


@dataclass
class RowDigests:
    """sha1 of the raw bytes of some rows of one split."""
    rows: np.ndarray       # int64, ascending
    digests: np.ndarray    # DIGEST_DTYPE, aligned with `rows`
    mode: str              # "all" | "sample"


@dataclass
class SplitScan:
    split: str
    n_rows: int
    rows_scanned: int
    buckets: Dict[int, ContentAccumulator]     # label value -> accumulator
    digests: Optional[RowDigests]
    seconds: float


def digest_rows(data: np.ndarray) -> np.ndarray:
    """sha1 of each row's raw bytes; byte-identical rows, and only those, share a digest."""
    if len(data) == 0:
        return np.empty(0, dtype=DIGEST_DTYPE)
    flat = np.ascontiguousarray(data).reshape(len(data), -1)
    return np.frombuffer(b"".join(hashlib.sha1(r).digest() for r in flat), dtype=DIGEST_DTYPE)


def scan_split(split: str, X: np.ndarray, y: np.ndarray, *, scan_rows: int, chunk_rows: int,
               hash_rows: Optional[np.ndarray],
               progress: Optional[Callable[[int], None]] = None) -> SplitScan:
    """Content scan of the first `scan_rows` rows of one split, in chunks.

    `hash_rows` selects the rows to digest for the duplicate check: None =
    every row, an empty array = none, else sorted row numbers. Rows inside the
    scanned range are hashed as the scan reads them; any others are read in a
    second sorted pass. Each chunk's memmap pages are released afterwards."""
    t0 = time.time()
    n = int(X.shape[0])
    scan_rows = int(min(max(scan_rows, 0), n))
    n_bands = int(X.shape[-1])
    buckets: Dict[int, ContentAccumulator] = {}
    hash_all = hash_rows is None
    dig_rows: List[np.ndarray] = []
    dig_vals: List[np.ndarray] = []

    if scan_rows:
        index = None if scan_rows == n else np.arange(scan_rows, dtype=np.int64)
        for ch in iter_chunks(X, index, chunk_rows=chunk_rows, assume_sorted=True):
            a, b = int(ch.rows[0]), int(ch.rows[-1]) + 1
            labels = np.asarray(y[a:b])
            ks = np.unique(labels)
            for k in ks:
                if len(ks) == 1:            # the usual case: a chunk inside one capture's class
                    sub, sub_rows = ch.data, ch.rows
                else:
                    sel = labels == k
                    sub, sub_rows = ch.data[sel], ch.rows[sel]
                if int(k) not in buckets:
                    buckets[int(k)] = ContentAccumulator(n_bands)
                buckets[int(k)].update(sub, sub_rows)
            if hash_all:
                dig_rows.append(ch.rows)
                dig_vals.append(digest_rows(ch.data))
            elif len(hash_rows):
                lo, hi = np.searchsorted(hash_rows, [a, b])
                if hi > lo:
                    sel_rows = hash_rows[lo:hi]
                    dig_rows.append(sel_rows)
                    dig_vals.append(digest_rows(ch.data[sel_rows - a]))
            release_pages(X, a, b)
            if progress is not None:
                progress(b)

    rest = np.arange(scan_rows, n, dtype=np.int64) if hash_all else hash_rows[hash_rows >= scan_rows]
    if len(rest):
        for ch in iter_chunks(X, rest, chunk_rows=chunk_rows, assume_sorted=True):
            dig_rows.append(ch.rows)
            dig_vals.append(digest_rows(ch.data))
            release_pages(X, int(ch.rows[0]), int(ch.rows[-1]) + 1)

    digests = None
    if hash_all or len(hash_rows):
        digests = RowDigests(
            rows=np.concatenate(dig_rows) if dig_rows else np.empty(0, dtype=np.int64),
            digests=np.concatenate(dig_vals) if dig_vals else np.empty(0, dtype=DIGEST_DTYPE),
            mode="all" if hash_all else "sample")
    return SplitScan(split=split, n_rows=n, rows_scanned=scan_rows, buckets=buckets, digests=digests,
                     seconds=round(time.time() - t0, 3))


def leakage_hash_rows(n: int, leakage_rows: Optional[int], rng: np.random.Generator) -> Optional[np.ndarray]:
    """None (every row), an empty array (off), or a sorted seeded sample."""
    if leakage_rows is None:
        return None
    if leakage_rows <= 0:
        return np.empty(0, dtype=np.int64)
    if leakage_rows >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=int(leakage_rows), replace=False)).astype(np.int64)


def duplicate_scan(digests: Mapping[str, RowDigests], zero_digest: bytes) -> Dict[str, Any]:
    """Byte-identical patches across splits (leakage) and within each split
    (informational). The all-zero patch is counted apart: sharing it is not
    leakage, and `content.all_zero_patches` already reports it."""
    zero = np.frombuffer(zero_digest, dtype=DIGEST_DTYPE)
    cross: Dict[str, Any] = {}
    names = list(digests)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = digests[a], digests[b]
            shared = np.intersect1d(da.digests, db.digests)
            zero_shared = bool(np.isin(zero, shared)[0])
            shared = shared[shared != zero[0]]
            examples = [{"sha1": shared[j:j + 1].tobytes().hex(),
                         f"{a}_rows": _ints(da.rows[da.digests == shared[j]][:3]),
                         f"{b}_rows": _ints(db.rows[db.digests == shared[j]][:3])}
                        for j in range(min(len(shared), N_EXAMPLES))]
            cross[f"{a}-{b}"] = {"n_shared_patches": int(len(shared)), "all_zero_patch_shared": zero_shared,
                                 "examples": examples}
    within: Dict[str, Any] = {}
    for s, d in digests.items():
        if len(d.digests) == 0:
            within[s] = {"n_groups": 0, "n_rows": 0, "all_zero_rows": 0, "examples": []}
            continue
        uniq, inverse, counts = np.unique(d.digests, return_inverse=True, return_counts=True)
        is_zero = uniq == zero[0]
        dup = np.flatnonzero((counts > 1) & ~is_zero)
        examples = [{"sha1": uniq[g:g + 1].tobytes().hex(), "rows": _ints(d.rows[inverse == g][:5]),
                     "count": int(counts[g])} for g in dup[:N_EXAMPLES]]
        within[s] = {"n_groups": int(len(dup)), "n_rows": int(counts[dup].sum()),
                     "all_zero_rows": int(counts[is_zero].sum()), "examples": examples}
    return {"cross_split": cross, "within_split": within}


# ============================================================================
# provenance
# ============================================================================

@dataclass(frozen=True)
class CaptureTable:
    """capture_index.json as sorted arrays, for vectorised per-row lookups."""
    keys: np.ndarray       # int64 capture index values, ascending
    label: np.ndarray      # int64 class of the capture's tissue, -1 if none
    patient: np.ndarray    # int64
    split: np.ndarray      # int8 position in SPLITS
    n: np.ndarray          # int64 patches the index says the capture holds

    @classmethod
    def from_index(cls, index: Mapping[int, "CaptureInfo"]) -> "CaptureTable":
        items = sorted(index.items())
        return cls(keys=np.array([k for k, _ in items], dtype=np.int64),
                   label=np.array([-1 if c.label is None else c.label for _, c in items], dtype=np.int64),
                   patient=np.array([c.patient for _, c in items], dtype=np.int64),
                   split=np.array([SPLITS.index(c.split) for _, c in items], dtype=np.int8),
                   n=np.array([c.n for _, c in items], dtype=np.int64))

    def locate(self, captures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(position in the table, known) for every row's capture value."""
        cap = np.asarray(captures, dtype=np.int64)
        if len(self.keys) == 0:
            return np.zeros(cap.shape, dtype=np.int64), np.zeros(cap.shape, dtype=bool)
        pos = np.minimum(np.searchsorted(self.keys, cap), len(self.keys) - 1)
        return pos, self.keys[pos] == cap


def capture_runs_summary(captures: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(run start rows, run capture values, distinct captures, runs per capture),
    vectorised. A capture is contiguous iff it has exactly one run."""
    cap = np.asarray(captures)
    if cap.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e, e
    starts = np.flatnonzero(np.r_[True, cap[1:] != cap[:-1]])
    run_caps = cap[starts]
    uniq, n_runs = np.unique(run_caps, return_counts=True)
    return starts, run_caps, uniq, n_runs


def _examples(mask: np.ndarray, **columns: np.ndarray) -> List[Dict[str, int]]:
    rows = np.flatnonzero(mask)[:N_EXAMPLES]
    return [{"row": int(r), **{k: int(v[r]) for k, v in columns.items()}} for r in rows]


def split_provenance(split: str, y: np.ndarray, groups: np.ndarray, captures: np.ndarray,
                     table: CaptureTable, adapter: "BuildAdapter") -> Dict[str, Any]:
    """Per-row consistency of one split's sidecars with capture_index.json.
    Returns {"checks": {check: {n_bad, examples, ...}}, "inventory": {...}}."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    captures = np.asarray(captures)
    code = SPLITS.index(split)
    pos, known = table.locate(captures)

    unknown = ~known
    bad_split = known & (table.split[pos] != code)
    exp_label = table.label[pos]
    bad_label = known & (y != exp_label)
    exp_patient = table.patient[pos]
    bad_patient = known & (groups.astype(np.int64) != exp_patient)

    caps, counts = np.unique(captures, return_counts=True)
    cpos, cknown = table.locate(caps)
    want = np.where(cknown, table.n[cpos], -1)
    bad_count = cknown & (counts != want)
    in_split = set(table.keys[table.split == code].tolist())
    missing = sorted(in_split - set(caps.tolist()))

    starts, run_caps, run_uniq, n_runs = capture_runs_summary(captures)
    broken = run_uniq[n_runs > 1]
    broken_examples = []
    for c in broken[:N_EXAMPLES]:
        st = starts[run_caps == c]
        broken_examples.append({"capture": int(c), "runs": int(len(st)), "run_starts": _ints(st[:5])})

    sizes = Counter(int(c) for c in counts)
    grids = Counter("x".join(map(str, g)) if g else "none"
                    for g in (adapter.grid_shape(int(c)) for c in counts))
    n_classes = len(adapter.class_names)
    valid = (y >= 0) & (y < n_classes)
    class_counts = np.bincount(y[valid].astype(np.int64), minlength=n_classes)
    inventory = {
        "n_rows": int(len(y)), "n_captures": int(len(caps)),
        "class_counts": {adapter.class_names[k]: int(class_counts[k]) for k in range(n_classes)},
        "rows_with_invalid_label": int((~valid).sum()),
        "patients": _sorted_ints(np.unique(groups)),
        "capture_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "grid_shapes": dict(sorted(grids.items())),
        "non_contiguous_captures": _ints(broken),
    }
    return {
        "checks": {
            "captures_known": {"n_bad": int(unknown.sum()), "examples": _examples(unknown, capture=captures)},
            "split": {"n_bad": int(bad_split.sum()),
                      "examples": [{**e, "capture_split": SPLITS[int(table.split[pos[e["row"]]])]}
                                   for e in _examples(bad_split, capture=captures)]},
            "labels": {"n_bad": int(bad_label.sum()),
                       "examples": _examples(bad_label, y=y, expected=exp_label, capture=captures)},
            "patients": {"n_bad": int(bad_patient.sum()),
                         "examples": _examples(bad_patient, groups=groups, expected=exp_patient,
                                               capture=captures)},
            "counts": {"n_bad": int(bad_count.sum()) + len(missing),
                       "examples": [{"capture": int(c), "rows": int(k), "index_n": int(w)}
                                    for c, k, w in zip(caps[bad_count], counts[bad_count],
                                                       want[bad_count])][:N_EXAMPLES],
                       "missing_captures": missing[:N_EXAMPLES], "n_missing": len(missing)},
            "contiguous": {"n_bad": int(len(broken)), "examples": broken_examples},
        },
        "inventory": inventory,
    }


PROVENANCE_CHECKS: Dict[str, Tuple[str, str]] = {
    # check -> (what a bad row/capture is, what passing proves)
    "captures_known": ("rows whose capture is not in capture_index.json", "every row's capture is indexed"),
    "split": ("rows whose capture belongs to another split", "every row sits in its capture's split"),
    "labels": ("rows with y != label(capture.tissue)", "y == label(capture.tissue) on every row"),
    "patients": ("rows with groups != capture.patient", "groups == capture.patient on every row"),
    "counts": ("captures whose row count differs from n (or that have no rows)",
               "every capture has exactly its n rows"),
    "contiguous": ("captures whose rows are split into several runs", "every capture's rows are contiguous"),
}


# ============================================================================
# the checks that read only JSON and headers
# ============================================================================

def check_manifest(adapter: "BuildAdapter", checks: CheckLog, *, verify_sha256: bool,
                   announce: Optional[Callable[[str, List[str]], None]] = None) -> Dict[str, Any]:
    """`verify_against_manifest` (shape/dtype/size, plus the streamed sha256
    when asked). `announce(split, files)` is called before a split's files are
    hashed, so the SplitGuard log records the read."""
    from training.npy_atomic import verify_against_manifest
    try:
        manifest = adapter.manifest
    except (OSError, ValueError) as e:
        checks.add("manifest.structure", FAIL, f"dataset_manifest.json is unreadable: {e}")
        checks.skip("manifest.sha256", "no readable manifest")
        return {"path": adapter.manifest_path.name, "readable": False}
    files = manifest.get("files") or {}
    recorded = bool(manifest.get("sha256")) and any((e or {}).get("sha256") for e in files.values())
    hash_now = verify_sha256 and recorded
    nbytes = int(sum(int((e or {}).get("size_bytes") or 0) for e in files.values()))
    if hash_now and announce is not None:
        for split in SPLITS:
            names = sorted(n for n in files if _file_split(n) == split)
            if names:
                announce(split, names)
    t0 = time.time()
    problems = verify_against_manifest(str(adapter.data_dir), check_sha256=hash_now)
    seconds = round(time.time() - t0, 3)
    sha = [p for p in problems if "sha256" in p or "while hashing" in p]
    structural = [p for p in problems if p not in sha]
    checks.verdict("manifest.structure", f"{len(files)} files match the manifest's shape, dtype and size",
                   fail=structural, n_files=len(files))
    if not verify_sha256:
        checks.skip("manifest.sha256", "verify_sha256 is off")
    elif not recorded:
        checks.add("manifest.sha256", WARN, "the manifest records no sha256, so nothing was verified")
    else:
        checks.verdict("manifest.sha256", f"{len(files)} files ({nbytes / 1e9:.2f} GB) re-hashed, all equal "
                                          f"to the manifest", fail=sha, n_files=len(files), bytes=nbytes,
                       seconds=seconds)
    return {"path": adapter.manifest_path.name, "readable": True,
            "file_sha256": sha256_file(adapter.manifest_path), "created_utc": manifest.get("created_utc"),
            "sha256_recorded": recorded, "n_files": len(files), "files": files,
            "extra": manifest.get("extra")}


def check_headers(adapter: "BuildAdapter", checks: CheckLog,
                  manifest_files: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], bool]:
    """Every split array present and complete, the four arrays of a split the
    same length, the NHWC layout, and each header equal to its manifest entry.
    Returns ({split: {kind: header}}, row-level work may proceed)."""
    heads = {s: {k: npy_header(adapter.array_path(k, s)) for k in ARRAY_KINDS} for s in SPLITS}

    present = [p for s in SPLITS for k in ARRAY_KINDS
               if (p := _header_problem(adapter.array_path(k, s).name, heads[s][k]))]
    checks.verdict("headers.present", f"all {len(SPLITS) * len(ARRAY_KINDS)} split arrays exist and are complete",
                   fail=present)
    ok = not present

    lengths = {s: {k: (heads[s][k].get("shape") or [None])[0] for k in ARRAY_KINDS} for s in SPLITS}
    unequal = [f"{s}: lengths {lengths[s]}" for s in SPLITS if ok and len(set(lengths[s].values())) != 1]
    checks.verdict("headers.row_counts", "len(X) == len(y) == len(groups) == len(captures) in every split",
                   fail=unequal, lengths=lengths)
    ok = ok and not unequal

    layout_fail: List[str] = []
    layout_warn: List[str] = []
    if ok:
        row_shapes = {s: tuple(heads[s]["X"]["shape"][1:]) for s in SPLITS}
        for s in SPLITS:
            hx = heads[s]["X"]
            if len(hx["shape"]) != 4 or hx["shape"][1] != hx["shape"][2]:
                layout_fail.append(f"X_{s}: shape {hx['shape']} is not NHWC with square patches")
            if not np.issubdtype(np.dtype(hx["dtype"]), np.floating):
                layout_fail.append(f"X_{s}: dtype {hx['dtype']} is not floating point")
            elif hx["dtype"] != "float16":
                layout_warn.append(f"X_{s}: dtype {hx['dtype']}, not float16 (percentiles are float16-rounded)")
            for k in ("y", "groups", "captures"):
                h = heads[s][k]
                if len(h["shape"]) != 1:
                    layout_fail.append(f"{k}_{s}: shape {h['shape']} is not 1-D")
                if not np.issubdtype(np.dtype(h["dtype"]), np.integer):
                    layout_fail.append(f"{k}_{s}: dtype {h['dtype']} is not an integer type")
        if len(set(row_shapes.values())) != 1:
            layout_fail.append(f"patch shapes differ between splits: {row_shapes}")
    if ok:
        checks.verdict("headers.layout", "X is float16 NHWC with one patch shape; y/groups/captures are 1-D "
                                         "integers", fail=layout_fail, warn=layout_warn)
    else:
        checks.skip("headers.layout", "headers.present / headers.row_counts failed")
    ok = ok and not layout_fail

    listed: List[str] = []
    for s in SPLITS:
        for k in ARRAY_KINDS:
            name, head = adapter.array_path(k, s).name, heads[s][k]
            entry = manifest_files.get(name)
            if entry is None:
                listed.append(f"{name} is not listed in the manifest")
            elif head.get("shape") is not None and (list(entry.get("shape") or []) != head["shape"]
                                                    or entry.get("dtype") != head["dtype"]):
                listed.append(f"{name}: header {head['shape']} {head['dtype']} != manifest "
                              f"{entry.get('shape')} {entry.get('dtype')}")
    checks.verdict("headers.match_manifest", "every split array is listed in the manifest with its header's "
                                             "shape and dtype", fail=listed)
    return heads, ok


def _manifest_extra(adapter: "BuildAdapter") -> Dict[str, Any]:
    """The manifest's `extra` block, {} if the manifest is missing or unreadable
    (`manifest.structure` reports that)."""
    try:
        return dict(adapter.manifest.get("extra") or {})
    except (OSError, ValueError, AttributeError):
        return {}


def check_classes(adapter: "BuildAdapter", checks: CheckLog) -> Dict[str, Any]:
    fail: List[str] = []
    try:
        names = list(adapter.class_names)
    except (OSError, ValueError, TypeError) as e:
        checks.add("classes.consistent", FAIL, f"class_names.json is unreadable: {e}")
        return {}
    if not names or len(set(str(n).lower() for n in names)) != len(names):
        fail.append(f"class_names.json must list distinct names, got {names}")
    sources = {"manifest": _manifest_extra(adapter).get("class_names")}
    split_assignment = _optional_attr(adapter, "split_assignment")
    if isinstance(split_assignment, dict):
        sources["split_assignment"] = split_assignment.get("class_names")
    for src, other in sources.items():
        if other is not None and list(other) != names:
            fail.append(f"{src} lists classes {list(other)}, class_names.json {names}")
    minority: List[int] = []
    try:
        minority = list(adapter.minority_classes)
    except KeyError as e:
        fail.append(f"minority class not among the classes: {e}")
    checks.verdict("classes.consistent", f"{len(names)} classes {names}, the same in every file that lists them",
                   fail=fail)
    return {"class_names": names, "minority_classes": minority,
            "minority_class_names": [names[i] for i in minority if 0 <= i < len(names)]}


def _optional_attr(adapter: "BuildAdapter", name: str) -> Any:
    """An adapter attribute that may not exist (or whose sidecar is missing): None then."""
    try:
        return getattr(adapter, name, None)
    except (OSError, ValueError, KeyError):
        return None


def check_wavelengths(adapter: "BuildAdapter", checks: CheckLog, *, n_bands: Optional[int],
                      expected_bands: Optional[int]) -> Dict[str, Any]:
    try:
        wl = np.asarray(adapter.wavelengths, dtype=np.float64)
    except (OSError, ValueError) as e:
        checks.add("wavelengths.valid", FAIL, f"wavelengths.npy is unreadable: {e}")
        checks.skip("wavelengths.selected_match", "no wavelengths")
        checks.skip("wavelengths.band_provenance", "no wavelengths")
        return {}
    fail: List[str] = []
    if wl.ndim != 1 or wl.size == 0:
        fail.append(f"wavelengths.npy has shape {list(wl.shape)}, expected 1-D")
    elif not np.isfinite(wl).all():
        fail.append("wavelengths.npy holds non-finite values")
    elif not (np.diff(wl) > 0).all():
        fail.append(f"wavelengths are not strictly increasing (first bad step after band "
                    f"{int(np.flatnonzero(np.diff(wl) <= 0)[0])})")
    cli = _manifest_extra(adapter).get("cli_args") or {}
    wants = {"X's band axis": n_bands, "the manifest's num_bands": cli.get("num_bands"),
             "expected_bands": expected_bands}
    for what, want in wants.items():
        if want is not None and wl.size != int(want):
            fail.append(f"{wl.size} wavelengths, {what} is {int(want)}")

    steps = np.diff(wl) if wl.ndim == 1 and wl.size > 1 else np.empty(0)
    med = float(np.median(steps)) if steps.size else None
    gaps = [{"after_band": int(i), "from_nm": float(wl[i]), "to_nm": float(wl[i + 1]),
             "width_nm": float(wl[i + 1] - wl[i]),
             "x_median_step": float((wl[i + 1] - wl[i]) / med) if med else None}
            for i in (adapter.spectral_gaps() if not fail else [])]
    info = {"n": int(wl.size), "values_nm": _floats(wl), "min_nm": float(wl.min()) if wl.size else None,
            "max_nm": float(wl.max()) if wl.size else None, "median_step_nm": med, "gaps": gaps,
            "segments": [list(s) for s in (adapter.spectral_segments() if not fail else [])]}
    gap_txt = ", ".join(f"{g['from_nm']:.1f}->{g['to_nm']:.1f} nm" for g in gaps) or "none"
    checks.verdict("wavelengths.valid", f"{wl.size} finite, strictly increasing wavelengths; gap {gap_txt}",
                   fail=fail, gaps=gaps)

    selected = _optional_attr(type(adapter), "selected_wavelengths")
    if selected is None:
        checks.skip("wavelengths.selected_match", f"{adapter.name} has no selected_wavelengths sidecar")
        info["selected_wavelengths_match"] = None
    else:
        try:
            sw = np.asarray(adapter.selected_wavelengths, dtype=np.float64)
            same = sw.shape == wl.shape and bool(np.array_equal(sw, wl))
            diff = float(np.max(np.abs(sw - wl))) if sw.shape == wl.shape and wl.size else None
            checks.verdict("wavelengths.selected_match", "wavelengths.npy == the build's selected_wavelengths.npy",
                           fail=[] if same else [f"wavelengths.npy differs from selected_wavelengths.npy "
                                                 f"(shapes {list(wl.shape)} / {list(sw.shape)}, max |diff| {diff})"],
                           max_abs_diff_nm=diff)
            info["selected_wavelengths_match"] = same
        except (OSError, ValueError) as e:
            checks.add("wavelengths.selected_match", FAIL, f"selected_wavelengths.npy is unreadable: {e}")
            info["selected_wavelengths_match"] = False

    # The build records which sensor bands it kept; check they are the ones stored.
    prov = _manifest_extra(adapter).get("provenance") or {}
    full_path = adapter.data_dir / "wavelengths_full.npy"
    idx = prov.get("selected_band_indices")
    if idx is None or not full_path.is_file():
        checks.skip("wavelengths.band_provenance", "no wavelengths_full.npy or manifest selected_band_indices")
    else:
        full = np.load(full_path)
        idx = np.asarray(idx, dtype=np.int64)
        okay = idx.size == wl.size and (idx.size == 0 or (idx.min() >= 0 and idx.max() < full.size)) and \
            bool(np.array_equal(np.asarray(full, dtype=np.float64)[idx], wl))
        checks.verdict("wavelengths.band_provenance",
                       f"wavelengths == wavelengths_full[selected_band_indices] ({full.size} sensor bands)",
                       fail=[] if okay else ["wavelengths.npy is not wavelengths_full.npy at the manifest's "
                                             "selected_band_indices"], n_sensor_bands=int(full.size))
    return info


def check_capture_index(adapter: "BuildAdapter", checks: CheckLog) -> Tuple[Optional[CaptureTable], Dict[str, Any]]:
    """capture_index.json entries well-formed: a class for every tissue, n > 0, unique keys."""
    try:
        index = adapter.capture_index
    except (OSError, ValueError, KeyError, TypeError) as e:
        checks.add("captures.index_valid", FAIL, f"capture_index.json is unreadable or malformed: {e}")
        return None, {}
    fail: List[str] = []
    no_class = [c.key for c in index.values() if c.label is None]
    if no_class:
        fail.append(f"{len(no_class)} captures have a tissue that is no class: {no_class[:5]}")
    empty = [c.key for c in index.values() if c.n <= 0]
    if empty:
        fail.append(f"{len(empty)} captures have n <= 0: {empty[:5]}")
    dup_keys = [k for k, v in Counter(c.key for c in index.values()).items() if v > 1]
    if dup_keys:
        fail.append(f"capture keys appear more than once: {dup_keys[:5]}")
    checks.verdict("captures.index_valid", f"{len(index)} captures, each with a class, n > 0 and a unique key",
                   fail=fail)
    by_split = Counter(c.split for c in index.values())
    info = {"path": adapter.capture_index_path.name, "sha256": sha256_file(adapter.capture_index_path),
            "n_captures": len(index), "by_split": {s: by_split.get(s, 0) for s in SPLITS},
            "by_split_tissue": {s: dict(sorted(Counter(c.tissue for c in index.values() if c.split == s).items()))
                                for s in SPLITS},
            "n_by_split": {s: {str(k): v for k, v in sorted(Counter(c.n for c in index.values()
                                                                     if c.split == s).items())}
                           for s in SPLITS},
            "patients_by_split": {s: _sorted_ints({c.patient for c in index.values() if c.split == s})
                                  for s in SPLITS}}
    return CaptureTable.from_index(index), info


def check_patients(adapter: "BuildAdapter", checks: CheckLog,
                   array_patients: Optional[Mapping[str, Sequence[int]]]) -> Dict[str, Any]:
    """Patient-level split integrity, from every source that names a split:
    split_assignment.json, capture_index.json and the groups arrays."""
    sources: Dict[str, Dict[str, set]] = {}
    split_of: Dict[int, str] = {}
    sp = _optional_attr(adapter, "split_patients")
    fail: List[str] = []
    if callable(sp):
        try:
            sources["split_assignment"] = {s: set(v) for s, v in sp().items()}
            split_of = {p: s for s, ps in sources["split_assignment"].items() for p in ps}
        except (OSError, ValueError, KeyError) as e:
            fail.append(f"split_assignment.json is unreadable: {e}")
    try:
        index = adapter.capture_index
        sources["capture_index"] = {s: {c.patient for c in index.values() if c.split == s} for s in SPLITS}
    except (OSError, ValueError, KeyError, TypeError):
        index = {}
    if array_patients is not None:
        sources["groups arrays"] = {s: set(array_patients.get(s, ())) for s in SPLITS}

    for src, by in sources.items():
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                both = by.get(a, set()) & by.get(b, set())
                if both:
                    fail.append(f"{src}: patients {sorted(both)[:10]} are in both {a} and {b}")
    if split_of and index:
        wrong = sorted({(c.patient, c.split, split_of.get(c.patient)) for c in index.values()
                        if split_of.get(c.patient) != c.split}, key=str)
        for p, s, want in wrong[:10]:
            fail.append(f"patient {p}: a capture in {s}, split_assignment says {want}")
    if "groups arrays" in sources and "capture_index" in sources:
        for s in SPLITS:
            a, c = sources["groups arrays"][s], sources["capture_index"][s]
            if a != c:
                fail.append(f"{s}: groups patients differ from capture_index's (only in groups {sorted(a - c)[:10]}, "
                            f"only in the index {sorted(c - a)[:10]})")
    warn: List[str] = []
    if "split_assignment" in sources and "capture_index" in sources:
        for s in SPLITS:
            idle = sources["split_assignment"][s] - sources["capture_index"][s]
            if idle:
                warn.append(f"{s}: split_assignment lists patients without captures {sorted(idle)[:10]}")
    counts = {src: {s: len(by.get(s, ())) for s in SPLITS} for src, by in sources.items()}
    checks.verdict("splits.patients_disjoint", f"patients disjoint across splits in {', '.join(sources)}, and "
                                               f"each capture in its patient's split", fail=fail, warn=warn,
                   n_patients=counts)
    out: Dict[str, Any] = {"n_patients": counts}
    if "split_assignment" in sources:
        out["patients_by_split"] = {s: sorted(v) for s, v in sources["split_assignment"].items()}
    return out


def check_scope(adapter: "BuildAdapter", checks: CheckLog,
                array_patients: Optional[Mapping[str, Sequence[int]]]) -> Dict[str, Any]:
    """Pass 1 (band selection, band statistics, gain corpus median, value scale)
    was fitted on exactly the train patients: the proof of leak-free band selection."""
    if not callable(_optional_attr(adapter, "pass1_train_patients")) or \
            not callable(_optional_attr(adapter, "split_patients")):
        checks.skip("scope.pass1_train_patients", f"{adapter.name} records no Pass-1 scope")
        return {}
    try:
        scope = dict(adapter.pass1_scope)
        pass1 = set(adapter.pass1_train_patients())
        assigned = {s: set(v) for s, v in adapter.split_patients().items()}
    except (OSError, ValueError, KeyError, TypeError) as e:
        checks.add("scope.pass1_train_patients", FAIL, f"pass1_scope_report.json / split_assignment.json "
                                                       f"unreadable: {e}")
        return {}
    heldout = assigned["val"] | assigned["test"]
    try:
        index = adapter.capture_index
        heldout |= {c.patient for c in index.values() if c.split != "train"}
        n_train_caps = sum(1 for c in index.values() if c.split == "train")
    except (OSError, ValueError, KeyError, TypeError):
        n_train_caps = None
    if array_patients is not None:
        heldout |= set(array_patients.get("val", ())) | set(array_patients.get("test", ()))

    fail: List[str] = []
    warn: List[str] = []
    if not pass1:
        fail.append("pass1_scope_report.json lists no train patients")
    if pass1 != assigned["train"]:
        fail.append(f"Pass-1 train patients differ from split_assignment's train patients (only in Pass 1: "
                    f"{sorted(pass1 - assigned['train'])[:10]}, only in the split: "
                    f"{sorted(assigned['train'] - pass1)[:10]})")
    if pass1 & heldout:
        fail.append(f"Pass 1 was fitted on held-out patients {sorted(pass1 & heldout)[:10]}")
    reported_heldout = {int(p) for p in scope.get("heldout_patients", [])}
    if reported_heldout & assigned["train"]:
        fail.append(f"pass1_scope_report.json calls train patients held out: "
                    f"{sorted(reported_heldout & assigned['train'])[:10]}")
    if "heldout_patients" in scope and reported_heldout != heldout:
        warn.append(f"pass1_scope_report.json heldout_patients ({len(reported_heldout)}) != the val+test "
                    f"patients ({len(heldout)})")
    if array_patients is not None and set(array_patients.get("train", ())) != pass1:
        warn.append("the train rows' patients (groups_train) differ from the Pass-1 train patients")
    if scope.get("n_train_captures") is not None and n_train_caps is not None and \
            int(scope["n_train_captures"]) != n_train_caps:
        warn.append(f"pass1_scope_report.json says {scope['n_train_captures']} train captures, the index "
                    f"has {n_train_caps}")
    checks.verdict("scope.pass1_train_patients", f"Pass 1 was fitted on exactly the {len(pass1)} train patients "
                                                 f"and no held-out patient", fail=fail, warn=warn,
                   n_pass1_patients=len(pass1), n_heldout_patients=len(heldout))
    return scope


# ============================================================================
# inventory of files
# ============================================================================

def inventory_files(adapter: "BuildAdapter", manifest_files: Mapping[str, Any]) -> Dict[str, Any]:
    """Every `.npy` header of the data dir and the build root, and the JSON
    sidecars the adapter reads (size, sha256), keyed by path under the build root."""
    def rel(p: Path) -> str:
        try:
            return p.relative_to(adapter.build_root).as_posix()
        except ValueError:
            return p.as_posix()

    npy: Dict[str, Any] = {}
    for p in sorted(set(adapter.data_dir.glob("*.npy")) | set(adapter.build_root.glob("*.npy"))):
        head = npy_header(p)
        head["in_manifest"] = p.parent == adapter.data_dir and p.name in manifest_files
        npy[rel(p)] = head
    sidecars: Dict[str, Any] = {}
    for role, p in adapter.files().items():
        if p.suffix == ".json":
            sidecars[role] = {"path": rel(p), "exists": p.is_file(),
                              **({"size_bytes": p.stat().st_size, "sha256": sha256_file(p)} if p.is_file() else {})}
    return {"npy": npy, "json": sidecars}


# ============================================================================
# reports of the row-level passes
# ============================================================================

def report_provenance(results: Mapping[str, Dict[str, Any]], checks: CheckLog) -> None:
    for name, (bad_what, ok_what) in PROVENANCE_CHECKS.items():
        per_split = {s: r["checks"][name] for s, r in results.items()}
        fail = [f"{s}: {d['n_bad']} {bad_what}" for s, d in per_split.items() if d["n_bad"]]
        checks.verdict(f"provenance.{name}", f"{ok_what} ({', '.join(results)})", fail=fail, splits=per_split)


def summarize_scan(scan: SplitScan, class_names: Sequence[str], percentiles: Sequence[float]) -> Dict[str, Any]:
    """One split's content-scan JSON: the whole split ("all") and each class.
    Empties `scan.buckets`, so the split's histograms (16 MB per class at 32
    bands) are freed before the next split is scanned."""
    total: Optional[ContentAccumulator] = None
    classes: Dict[str, Any] = {}
    rounded = False
    for k in sorted(scan.buckets):
        acc = scan.buckets[k]
        name = class_names[k] if 0 <= k < len(class_names) else f"invalid_label_{k}"
        classes[name] = acc.to_dict(percentiles)
        rounded = rounded or acc.hist.rounded
        total = acc if total is None else total.merge(acc)   # merging into the first bucket: no extra histogram
    scan.buckets = {}
    return {"rows": scan.n_rows, "rows_scanned": scan.rows_scanned, "complete": scan.rows_scanned == scan.n_rows,
            "fit_allowed": scan.split == "train", "seconds": scan.seconds,
            "quantile_method": "float16_rounded" if rounded else "float16_exact",
            "all": total.to_dict(percentiles) if total is not None else {"rows": 0}, "classes": classes}


def report_content(summaries: Mapping[str, Dict[str, Any]], checks: CheckLog,
                   percentiles: Sequence[float]) -> Dict[str, Any]:
    """The three content checks over the per-split summaries."""
    totals = {s: d["all"] for s, d in summaries.items()}
    partial = [f"{s}: {d['rows_scanned']:,} of {d['rows']:,} rows scanned" for s, d in summaries.items()
               if not d["complete"]]
    keys = ("nan", "posinf", "neginf", "rows_with_nonfinite", "first_nonfinite_rows")
    nonfinite = {s: {k: d.get(k, [] if k.startswith("first") else 0) for k in keys} for s, d in totals.items()}
    fail = [f"{s}: {d['nan']} NaN, {d['posinf']} +Inf, {d['neginf']} -Inf in {d['rows_with_nonfinite']} rows "
            f"(first rows {d['first_nonfinite_rows'][:3]})" for s, d in nonfinite.items() if d["rows_with_nonfinite"]]
    scanned = sum(d["rows_scanned"] for d in summaries.values())
    checks.verdict("content.nonfinite", f"0 NaN / +-Inf in {scanned:,} rows", fail=fail, warn=partial,
                   rows_scanned={s: d["rows_scanned"] for s, d in summaries.items()}, splits=nonfinite)
    sat = {s: {"values": d.get("float16_saturated", 0), "first_rows": d.get("first_saturated_rows", [])}
           for s, d in totals.items()}
    checks.verdict("content.float16_saturation", f"no value at the float16 ceiling |x| = {F16_MAX:g}",
                   warn=[f"{s}: {d['values']} values at |x| = {F16_MAX:g}" for s, d in sat.items() if d["values"]],
                   splits=sat)
    zero = {s: {"patches": d.get("all_zero_patches", 0), "first_rows": d.get("first_zero_rows", [])}
            for s, d in totals.items()}
    checks.verdict("content.all_zero_patches", "no all-zero patch",
                   warn=[f"{s}: {d['patches']} all-zero patches" for s, d in zero.items() if d["patches"]],
                   splits=zero)
    methods = {d["quantile_method"] for d in summaries.values()}
    return {"percentiles": list(percentiles), "percentile_method": "inverted_cdf over finite values",
            "quantile_method": "float16_rounded" if "float16_rounded" in methods else "float16_exact",
            "splits": dict(summaries)}


def report_leakage(scans: Mapping[str, SplitScan], checks: CheckLog, zero_digest: bytes) -> Dict[str, Any]:
    digests = {s: sc.digests for s, sc in scans.items() if sc.digests is not None}
    hashed = {s: int(len(d.rows)) for s, d in digests.items()}
    mode = sorted({d.mode for d in digests.values()})
    dup = duplicate_scan(digests, zero_digest)
    cross_fail = [f"{pair}: {d['n_shared_patches']} byte-identical patches" for pair, d in dup["cross_split"].items()
                  if d["n_shared_patches"]]
    checks.verdict("leakage.cross_split", f"no byte-identical patch in two splits ({', '.join(mode)} rows hashed: "
                                          f"{hashed})", fail=cross_fail, rows_hashed=hashed, mode=mode,
                   pairs=dup["cross_split"])
    within_warn = [f"{s}: {d['n_groups']} groups of identical patches ({d['n_rows']} rows)"
                   for s, d in dup["within_split"].items() if d["n_groups"]]
    checks.verdict("leakage.within_split", "no duplicate patch within a split", warn=within_warn,
                   splits=dup["within_split"])
    return {"rows_hashed": hashed, "mode": mode, **dup}


# ============================================================================
# the stage
# ============================================================================

def sidecar_summaries(adapter: "BuildAdapter") -> Dict[str, Any]:
    """split_assignment.json (without the patient map, which `patients` holds),
    pass1_scope_report.json (whole) and capture_gain_report.json (without the
    per-capture table) for the inventory; absent sidecars are left out."""
    out: Dict[str, Any] = {}
    sa = _optional_attr(adapter, "split_assignment")
    if isinstance(sa, dict):
        out["split_assignment"] = {k: v for k, v in sa.items() if k != "groups"}
    scope = _optional_attr(adapter, "pass1_scope")
    if isinstance(scope, dict):
        out["pass1_scope"] = scope
    gains = _optional_attr(adapter, "capture_gain_report")
    if isinstance(gains, dict):
        out["capture_gain_report"] = {**{k: v for k, v in gains.items() if k != "per_capture"},
                                      "n_per_capture": len(gains.get("per_capture") or {})}
    return out


def projected_peak_bytes(X: np.ndarray, n_rows: Mapping[str, int], n_classes: int, cfg: Any) -> int:
    """What the content scan and the duplicate hashes hold at their peak
    (for `ctx.ram_guard`); on trainsel about 0.5 GB."""
    row_bytes = int(np.prod(X.shape[1:])) * X.dtype.itemsize
    n_max = max(n_rows.values())
    if cfg.leakage_rows == 0:
        n_hash = 0
    else:
        n_hash = sum(n if cfg.leakage_rows is None else min(int(cfg.leakage_rows), n) for n in n_rows.values())
    histograms = (n_classes + 1) * int(X.shape[-1]) * _F16_PATTERNS * 8    # one split's classes + cumsum
    return int(min(cfg.chunk_rows, n_max) * row_bytes                       # one chunk
               + histograms
               + n_max * 8                                                  # y of the split being scanned
               + n_hash * (DIGEST_DTYPE.itemsize + 8) * 3                   # digests, row numbers, np.unique
               + 200_000_000)                                               # float64 blocks and temporaries


def _log_check(ctx: "StageContext", check: Check) -> None:
    if check.status == WARN:                 # the console already tags it "WARNING:"
        ctx.warn(f"{check.name}: {check.message}")
    else:
        ctx.log(f"{check.status.upper():<4} {check.name}: {check.message}",
                logging.ERROR if check.status == FAIL else logging.INFO)


def _progress(ctx: "StageContext", split: str, total: int) -> Callable[[int], None]:
    state = {"next": 0.1, "t0": time.time()}

    def report(done: int) -> None:
        ctx.log_metrics(split=split, rows_done=done, rows_total=total)
        if total and (done >= state["next"] * total or done == total):
            el = max(time.time() - state["t0"], 1e-9)
            ctx.log(f"scan {split}: {done:,}/{total:,} rows ({100.0 * done / total:.0f}%), {done / el:,.0f} rows/s")
            state["next"] = math.floor(10.0 * done / total) / 10.0 + 0.1
    return report


def run(ctx: "StageContext", cfg: Any) -> Dict[str, Any]:
    """The `discovery` stage: see the module docstring. `cfg` is a
    `hsi_augment.stages.discovery.DiscoveryConfig`."""
    t_start = time.time()
    adapter = ctx.adapter
    checks = CheckLog(on_add=lambda c: _log_check(ctx, c))
    report: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "stage": ctx.name, "adapter": adapter.name,
                              "data_dir": str(adapter.data_dir), "build_root": str(adapter.build_root),
                              "config": cfg.to_dict()}

    # -- JSON and headers -------------------------------------------------------------
    with ctx.timer("manifest"):
        if cfg.verify_sha256:
            ctx.log("verifying the manifest with a streamed sha256 of every file (minutes on a full build)")
        report["manifest"] = check_manifest(
            adapter, checks, verify_sha256=cfg.verify_sha256,
            announce=lambda split, names: ctx.split_guard.check(split, "structural", files=names))
    manifest_files = report["manifest"].get("files") or {}
    with ctx.timer("inventory"):
        heads, headers_ok = check_headers(adapter, checks, manifest_files)
        report["files"] = inventory_files(adapter, manifest_files)
        report.update(check_classes(adapter, checks))
        n_bands = heads["train"]["X"]["shape"][-1] if headers_ok else None
        report["wavelengths"] = check_wavelengths(adapter, checks, n_bands=n_bands,
                                                  expected_bands=cfg.expected_bands)
        table, report["capture_index"] = check_capture_index(adapter, checks)
        report.update(sidecar_summaries(adapter))

    class_names = list(report.get("class_names") or [])
    row_level = headers_ok and table is not None and bool(class_names)
    why_not = "skipped: header, capture-index or class checks failed"
    need_x = cfg.content_scan or cfg.leakage_rows != 0
    opened: Dict[str, Any] = {}
    array_patients: Optional[Dict[str, List[int]]] = None

    # -- provenance: y / groups / captures against capture_index.json --------------------
    if row_level:
        kinds = ("X", "y", "groups", "captures") if need_x else ("y", "groups", "captures")
        opened = {s: ctx.open_split(s, "structural", kinds=kinds) for s in SPLITS}
        results: Dict[str, Dict[str, Any]] = {}
        with ctx.timer("provenance"):
            for s in SPLITS:
                arr = opened[s]
                results[s] = split_provenance(s, np.asarray(arr.y), np.asarray(arr.groups),
                                              np.asarray(arr.captures), table, adapter)
        report_provenance(results, checks)
        report["splits"] = {s: r["inventory"] for s, r in results.items()}
        array_patients = {s: r["inventory"]["patients"] for s, r in results.items()}
        del results
    else:
        for name in PROVENANCE_CHECKS:
            checks.skip(f"provenance.{name}", why_not)

    report["patients"] = check_patients(adapter, checks, array_patients)
    check_scope(adapter, checks, array_patients)

    # -- content scan and duplicate hashes --------------------------------------------------
    if row_level and need_x:
        X0 = opened["train"].X
        n_rows = {s: int(opened[s].X.shape[0]) for s in SPLITS}
        report["ram_guard"] = ctx.ram_guard(projected_peak_bytes(X0, n_rows, len(class_names), cfg),
                                            "content scan + duplicate hashes")
        scans: Dict[str, SplitScan] = {}
        summaries: Dict[str, Dict[str, Any]] = {}
        for s in SPLITS:
            arr, n = opened[s], n_rows[s]
            scan_rows = 0 if not cfg.content_scan else (n if cfg.scan_max_rows is None else min(cfg.scan_max_rows, n))
            hash_rows = leakage_hash_rows(n, cfg.leakage_rows, ctx.rng("leakage", s))
            with ctx.timer(f"scan_{s}"):
                scans[s] = scan_split(s, arr.X, np.asarray(arr.y), scan_rows=scan_rows, chunk_rows=cfg.chunk_rows,
                                      hash_rows=hash_rows, progress=_progress(ctx, s, scan_rows))
            if cfg.content_scan:
                summaries[s] = summarize_scan(scans[s], class_names, cfg.percentiles)
        if cfg.content_scan:
            report["content"] = report_content(summaries, checks, cfg.percentiles)
        if cfg.leakage_rows != 0:
            zero = hashlib.sha1(np.zeros(X0.shape[1:], dtype=X0.dtype).tobytes()).digest()
            report["leakage"] = report_leakage(scans, checks, zero)
        del scans, summaries
    if not (row_level and cfg.content_scan):
        reason = why_not if not row_level else "content_scan is off"
        for name in ("content.nonfinite", "content.float16_saturation", "content.all_zero_patches"):
            checks.skip(name, reason)
    if not (row_level and cfg.leakage_rows != 0):
        reason = why_not if not row_level else "leakage_rows is 0"
        for name in ("leakage.cross_split", "leakage.within_split"):
            checks.skip(name, reason)
    opened.clear()

    # -- reports ------------------------------------------------------------------------------
    report["resources"] = {"peak_rss_mb": peak_rss_mb(), "wall_s": round(time.time() - t_start, 3),
                           "pid": os.getpid(),
                           "note": "peak RSS of the whole process (interpreter and imported libraries "
                                   "included); scanned memmap pages are released after each chunk"}
    failed, warned = checks.names(FAIL), checks.names(WARN)
    integrity = {"schema_version": SCHEMA_VERSION, "stage": ctx.name, "data_dir": str(adapter.data_dir),
                 "status": checks.overall, "counts": checks.counts(), "failed": failed, "warnings": warned,
                 "skipped": checks.names(SKIP), "checks": checks.to_dict()}
    ctx.save_json(DISCOVERY_REPORT, report)
    ctx.save_json(INTEGRITY_REPORT, integrity)
    ctx.log(f"{checks.overall.upper()}: {checks.counts()}; peak RSS {report['resources']['peak_rss_mb']} MB")
    if failed:
        raise StageError(f"{len(failed)} hard failure(s): {', '.join(failed)}; "
                         f"see {ctx.stage_dir / INTEGRITY_REPORT}")

    splits = report.get("splits") or {}
    content = (report.get("content") or {}).get("splits") or {}
    return {
        "status": checks.overall, "counts": checks.counts(), "warnings": warned,
        "rows": {s: d["n_rows"] for s, d in splits.items()},
        "captures": {s: d["n_captures"] for s, d in splits.items()},
        "train_capture_sizes": (splits.get("train") or {}).get("capture_sizes"),
        "rows_scanned": {s: d["rows_scanned"] for s, d in content.items()} or None,
        "nonfinite": sum(d["all"].get("rows_with_nonfinite", 0) for d in content.values()) if content else None,
        "scope_verified": checks.checks["scope.pass1_train_patients"].status == PASS,
        "wavelength_gaps": [(g["from_nm"], g["to_nm"]) for g in report["wavelengths"].get("gaps", [])],
        "peak_rss_mb": report["resources"]["peak_rss_mb"],
    }
