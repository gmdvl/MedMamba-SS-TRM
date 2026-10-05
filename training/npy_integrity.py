# -*- coding: utf-8 -*-
"""
training/npy_integrity.py
==========================
MedMamba-SS-TRM v12/v13 stability plan, Phases 4-6.

The training loop's `NpyDataset` (`train_example_v6.py`) memory-maps its
backing `.npy` files (`np.load(path, mmap_mode="r")`). A damaged, truncated,
or otherwise inaccessible backing file turns every later page-fault into a
GENUINE `SIGBUS` in the MAIN process - a completely different failure mode
from a DataLoader WORKER dying from `/dev/shm` exhaustion, even though both
can present as "Job ... terminated by signal SIGBUS" to the user.

This module validates a dataset's `.npy` files BEFORE any DataLoader worker
or memory-map is created, so a bad file is caught as an explicit
`DatasetStorageError` (Phase 6: "training must terminate with
DATASET_STORAGE_ERROR rather than proceeding to a guaranteed SIGBUS") -
never silently caught-and-continued.

It also implements Phase 5's explicit dataset storage modes:
    "mmap" - `np.load(path, mmap_mode="r")` (current/legacy behaviour)
    "ram"  - `np.load(path)` (fully materialized, no mmap in the hot path)
    "auto" - validate, then pick "ram" if the file comfortably fits in
             available RAM (see `resolve_storage_mode`), else validated
             "mmap"
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class DatasetStorageError(RuntimeError):
    """Raised when a `.npy` file fails integrity validation. Callers MUST
    NOT catch this and fall back to memory-mapping the file anyway - that
    is exactly the guaranteed-SIGBUS path this module exists to prevent."""


# Explicit per-file validation states (v13 plan, Workstream 1 - "one
# authoritative dataset-validation result", never INVALID -> VALID).
STATUS_NOT_CHECKED = "NOT_CHECKED"
STATUS_VALID = "VALID"
STATUS_MISSING = "MISSING"
STATUS_EMPTY = "EMPTY"
STATUS_HEADER_INVALID = "HEADER_INVALID"
STATUS_TRUNCATED = "TRUNCATED"
STATUS_SIZE_MISMATCH = "SIZE_MISMATCH"
STATUS_UNREADABLE = "UNREADABLE"
# Header + on-disk size are correct, but the payload bytes are physically
# unreadable (OSError(5)/EIO - bad sectors / failing disk / fs corruption).
# Distinct from TRUNCATED (file genuinely short) and UNREADABLE (format broken).
STATUS_IO_ERROR = "IO_ERROR"

VALIDATION_LEVELS = ("structural", "probe", "deep")
# "structural" / "probe" - header + on-disk size + a SIGBUS-free seek()/read()
#   probe of ~19 rows (identical at the single-file level; "probe" is the name
#   the prep core uses to signal "structural check + the caller's own all-zero /
#   truncation row probe, but no end-to-end payload read").
# "deep" - additionally read every payload byte back (`full_read_check`).


@dataclass
class NpyFileReport:
    path: str
    exists: bool = False
    size_bytes: int = 0
    valid_header: bool = False
    dtype: Optional[str] = None
    shape: Optional[List[int]] = None
    n_samples: Optional[int] = None
    expected_element_count: Optional[int] = None
    expected_byte_size: Optional[int] = None
    header_offset: Optional[int] = None
    expected_file_size: Optional[int] = None
    readable_elements: Optional[int] = None
    status: str = STATUS_NOT_CHECKED
    first_sample_ok: bool = False
    middle_sample_ok: bool = False
    last_sample_ok: bool = False
    random_sample_ok: bool = False
    finite_sample_values: bool = False
    # Full-file sha256 of what is ACTUALLY ON DISK, folded during the deep
    # read pass (level="deep" + compute_sha256=True). None whenever the deep
    # pass did not run, the file has trailing bytes after the payload, or the
    # read failed - callers must treat None as "no digest", never as a match.
    sha256: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        base = (self.exists and self.valid_header and self.status == STATUS_VALID
                and not self.errors)
        if base and self.n_samples:
            base = (base and self.first_sample_ok and self.middle_sample_ok
                    and self.last_sample_ok and self.random_sample_ok)
        return base

    def as_dict(self) -> Dict:
        d = dict(self.__dict__)
        return d


def _read_npy_header(path: str):
    """Parses ONLY the `.npy` header - no array data is read, no mmap is
    created. Returns `(shape, dtype, fortran_order, data_offset_bytes)`
    where `data_offset_bytes` is the absolute byte offset at which the array
    payload begins (== header length). Raises on a malformed/absent header."""
    import numpy.lib.format as _fmt
    with open(path, "rb") as f:
        version = _fmt.read_magic(f)
        if version == (1, 0):
            shape, fortran_order, dtype = _fmt.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran_order, dtype = _fmt.read_array_header_2_0(f)
        else:
            shape, fortran_order, dtype = _fmt._read_array_header(f, version)
        data_offset = f.tell()
    return shape, dtype, fortran_order, data_offset


def _classify_os_error(e: OSError) -> str:
    """OSError(5)/EIO -> STATUS_IO_ERROR (physically unreadable), anything
    else -> STATUS_UNREADABLE."""
    if getattr(e, "errno", None) == errno.EIO or "input/output error" in str(e).lower():
        return STATUS_IO_ERROR
    return STATUS_UNREADABLE


def full_read_check(path: str, *, chunk_bytes: int = 16 << 20,
                     hasher=None) -> Tuple[str, Optional[str]]:
    """Plain buffered read of the ENTIRE array payload - no mmap, no
    `np.load`. Used by the opt-in `level="deep"` validation. Returns
    `(status, detail)`:
      STATUS_IO_ERROR  - OSError(5)/EIO while reading (correct size on disk,
                         bytes physically unreadable)
      STATUS_TRUNCATED - a clean short read before the payload end
      STATUS_HEADER_INVALID / STATUS_UNREADABLE - header or other OSError
      STATUS_VALID     - every payload byte was read back

    `hasher`: optional `hashlib` object fed EVERY byte of the file in order -
    the header bytes first, then the payload. Because this pass already reads
    the whole file, a caller gets a digest of what is ACTUALLY ON DISK for
    free (no second read). The digest is comparable to
    `training.npy_atomic._sha256_of_file` only when the file has no trailing
    bytes after the payload, so callers must gate on
    `size == data_offset + payload_bytes` before trusting it (see
    `validate_npy_file`). Only meaningful when STATUS_VALID is returned.
    """
    try:
        shape, dtype, _fortran, data_offset = _read_npy_header(path)
    except Exception as e:
        return STATUS_HEADER_INVALID, f"invalid/unreadable .npy header: {e}"
    dtype = np.dtype(dtype)
    expected = (int(np.prod(shape)) * dtype.itemsize) if shape else 0
    read_total = 0
    try:
        with open(path, "rb", buffering=0) as fh:
            if hasher is not None:
                # Hash the header bytes so the digest covers the whole file,
                # matching _sha256_of_file. Reading them also positions the
                # handle exactly where the seek() below would have.
                head = fh.read(int(data_offset))
                if len(head) != int(data_offset):
                    return (STATUS_TRUNCATED,
                            f"short read in header: {len(head)} of {data_offset} bytes present")
                hasher.update(head)
            else:
                fh.seek(int(data_offset))
            while read_total < expected:
                want = min(chunk_bytes, expected - read_total)
                buf = fh.read(want)
                if not buf:
                    return (STATUS_TRUNCATED,
                            f"short read: {read_total} of {expected} payload bytes present")
                if hasher is not None:
                    hasher.update(buf)
                read_total += len(buf)
    except OSError as e:
        st = _classify_os_error(e)
        return st, (f"OSError(5)/EIO at byte {read_total} of {expected} payload bytes ({e})"
                    if st == STATUS_IO_ERROR
                    else f"OSError reading payload at byte {read_total}: {e}")
    return STATUS_VALID, None


def validate_npy_file(path: str, sample_indices: Optional[List[int]] = None,
                       check_finite: bool = True, finite_sample_cap: int = 4096,
                       level: str = "structural",
                       compute_sha256: bool = False) -> NpyFileReport:
    """Validates one `.npy` file per the Phase 4 checklist: exists, size>0,
    valid header, dtype/shape/sample-count, expected element/byte size,
    first/middle/last/random sample reads succeed, and (optionally) a
    bounded sample of values is finite. Never raises for an invalid file -
    returns a report with `.ok == False` and `.errors` populated; raising
    is the CALLER's decision (`validate_dataset` below raises).

    `level`: "structural" (default) - header + on-disk size + a SIGBUS-free
    seek()/read() probe of ~19 rows. "deep" - additionally read the whole
    payload end-to-end (`full_read_check`), catching a physically-unreadable
    region the row probes happen to miss.

    `compute_sha256`: only honoured at `level="deep"`. Folds a full-file
    sha256 into `report.sha256` during the read the deep pass already does,
    so a caller gets a digest of the bytes ON DISK at no extra I/O. Skipped
    (left None) when the file carries trailing bytes past the payload, since
    such a digest would not match `npy_atomic._sha256_of_file`."""
    report = NpyFileReport(path=path)

    if not os.path.isfile(path):
        report.status = STATUS_MISSING
        report.errors.append("file does not exist")
        return report
    report.exists = True

    size = os.path.getsize(path)
    report.size_bytes = size
    if size <= 0:
        report.status = STATUS_EMPTY
        report.errors.append("file size is 0 bytes")
        return report

    # ------------------------------------------------------------------
    # Deterministic header + file-size check (v13 plan, Workstream 1b).
    # This catches a byte-truncated but header-intact file WITHOUT any
    # mmap or full np.load - and short-circuits BEFORE the sample-probe
    # below, because a probe read past EOF on a mapping that did manage
    # to open raises an uncatchable SIGBUS in this (main) process.
    # ------------------------------------------------------------------
    try:
        shape, dtype, _fortran, data_offset = _read_npy_header(path)
    except Exception as e:  # malformed magic/header, unpickleable dtype, ...
        report.status = STATUS_HEADER_INVALID
        report.errors.append(f"invalid/unreadable .npy header: {e}")
        return report

    dtype = np.dtype(dtype)
    report.valid_header = True
    report.dtype = str(dtype)
    report.shape = list(shape)
    report.n_samples = int(shape[0]) if len(shape) >= 1 else None
    report.header_offset = int(data_offset)

    expected_elements = int(np.prod(shape)) if shape else 0
    expected_bytes = expected_elements * dtype.itemsize
    expected_file_size = int(data_offset) + expected_bytes
    report.expected_element_count = expected_elements
    report.expected_byte_size = expected_bytes
    report.expected_file_size = expected_file_size
    report.readable_elements = max(0, (size - int(data_offset))) // max(1, dtype.itemsize)

    if size < expected_file_size:
        report.status = STATUS_TRUNCATED
        pct = 100.0 * size / expected_file_size if expected_file_size else 0.0
        report.errors.append(
            f"TRUNCATED: {size} bytes on disk, header promises {expected_file_size} "
            f"({pct:.1f}% present); expected_elements={expected_elements}, "
            f"readable_elements={report.readable_elements}")
        return report
    if size > expected_file_size:
        report.warnings.append(
            f"file is {size - expected_file_size} byte(s) larger than the array payload "
            f"({expected_file_size}) - trailing bytes after the .npy data segment")

    # ------------------------------------------------------------------
    # SIGBUS-free sample probe (v13.1 remediation). Instead of faulting
    # mmap pages (which return zeros - or an *uncatchable* SIGBUS - on a
    # physically-unreadable region on some filesystems), seek()+read() the
    # bytes for a handful of rows with a plain file object. A bad region
    # then surfaces as a catchable OSError(5)/EIO right here, in the main
    # process, before any Dataset/DataLoader/mmap is built.
    # ------------------------------------------------------------------
    n = report.n_samples or 0
    if n <= 0:
        report.status = STATUS_UNREADABLE
        report.errors.append("array has 0 samples along axis 0")
        return report

    row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    row_nbytes = row_elems * dtype.itemsize

    def _row_offset(i: int) -> int:
        return int(data_offset) + int(i) * row_nbytes

    rng = np.random.default_rng(0)
    rand = (sample_indices if sample_indices is not None
            else rng.choice(n, size=min(16, n), replace=False).tolist())
    probe_indices = [0, n // 2, n - 1] + [int(i) for i in rand]

    sample_values: List[np.ndarray] = []
    try:
        with open(path, "rb", buffering=0) as fh:
            for pos, i in enumerate(probe_indices):
                fh.seek(_row_offset(i))
                buf = fh.read(row_nbytes)
                if len(buf) != row_nbytes:
                    report.status = STATUS_TRUNCATED
                    report.errors.append(
                        f"short read for row {i}: got {len(buf)} of {row_nbytes} bytes at "
                        f"offset {_row_offset(i)} (file size {size})")
                    return report
                if pos == 0:
                    report.first_sample_ok = True
                elif pos == 1:
                    report.middle_sample_ok = True
                elif pos == 2:
                    report.last_sample_ok = True
                if row_nbytes:
                    sample_values.append(np.frombuffer(buf, dtype=dtype))
        report.random_sample_ok = True
    except OSError as e:
        report.status = _classify_os_error(e)
        if report.status == STATUS_IO_ERROR:
            report.errors.append(
                f"IO_ERROR: OSError(5)/EIO reading array bytes ({e}). The file is the correct "
                f"size on disk but its bytes are physically unreadable (bad sectors / failing "
                f"disk / filesystem corruption) - not a truncated write, not a DataLoader issue.")
        else:
            report.errors.append(
                f"failed reading sample data (possible truncated/corrupt file): {e}")
        return report
    except (ValueError, BufferError, MemoryError) as e:
        report.status = STATUS_UNREADABLE
        report.errors.append(f"failed reading sample data: {e}")
        return report
    except Exception as e:  # never let an unexpected reader error escape as a raw exception
        report.status = STATUS_UNREADABLE
        report.errors.append(f"UNREADABLE: unexpected error reading sample data: {e!r}")
        return report

    if check_finite and np.issubdtype(dtype, np.floating) and sample_values:
        flat = np.concatenate([v.reshape(-1)[:finite_sample_cap] for v in sample_values])
        report.finite_sample_values = bool(np.all(np.isfinite(flat)))
        if not report.finite_sample_values:
            report.errors.append("sampled values contain NaN/Inf")
    else:
        report.finite_sample_values = True

    # Opt-in deep pass: read the whole payload end-to-end (no mmap, no np.load).
    if level == "deep":
        # Only hash when the file is exactly header+payload; a file with
        # trailing slack would produce a digest that no whole-file hash of it
        # can reproduce, and a wrong digest is worse than none.
        hasher = (hashlib.sha256()
                  if (compute_sha256 and size == expected_file_size) else None)
        st, detail = full_read_check(path, hasher=hasher)
        if st != STATUS_VALID:
            report.status = st
            report.errors.append(f"deep read failed: {detail}")
            return report
        if hasher is not None:
            report.sha256 = hasher.hexdigest()

    if not report.errors and report.status == STATUS_NOT_CHECKED:
        report.status = STATUS_VALID
    return report


@dataclass
class DatasetIntegrityReport:
    files: Dict[str, Dict]
    x_y_length_matches: Dict[str, bool]
    all_ok: bool
    level: str = "structural"


def validate_dataset(paths: Dict[str, str], pairs: Optional[List[tuple]] = None,
                      abort_on_failure: bool = True,
                      level: str = "structural",
                      compute_sha256: bool = False) -> DatasetIntegrityReport:
    """`paths`: e.g. {"X_train": ".../X_train.npy", "y_train": "...",
    "X_val": ..., "y_val": ...} - any subset; missing/optional entries
    (e.g. no val split) are simply skipped. `pairs`: list of (x_key, y_key)
    tuples whose sample counts must match (default: every "X_*"/"y_*" with
    the same suffix present in `paths`).

    Raises `DatasetStorageError` if `abort_on_failure` and any file fails
    validation or any X/y pair mismatches - per Phase 6, this must happen
    BEFORE any DataLoader/mmap is used for training.

    `compute_sha256`: at `level="deep"`, fold a full-file digest of the bytes
    on disk into each `NpyFileReport.sha256` during the read the deep pass
    already performs (see `validate_npy_file`)."""
    if level not in VALIDATION_LEVELS:
        raise ValueError(f"level must be one of {VALIDATION_LEVELS}, got {level!r}")
    file_reports: Dict[str, Dict] = {}
    parsed: Dict[str, NpyFileReport] = {}
    for key, path in paths.items():
        try:
            r = validate_npy_file(path, level=level, compute_sha256=compute_sha256)
        except Exception as e:  # validator itself must never crash the caller
            r = NpyFileReport(path=path, status=STATUS_UNREADABLE)
            r.errors.append(f"UNREADABLE: validate_npy_file raised: {e!r}")
        parsed[key] = r
        file_reports[key] = r.as_dict()

    if pairs is None:
        pairs = []
        for key in paths:
            if key.startswith("X_"):
                suffix = key[len("X_"):]
                y_key = f"y_{suffix}"
                if y_key in paths:
                    pairs.append((key, y_key))

    length_matches: Dict[str, bool] = {}
    mismatch_errors: List[str] = []
    for x_key, y_key in pairs:
        rx, ry = parsed.get(x_key), parsed.get(y_key)
        if rx is None or ry is None or not rx.ok or not ry.ok:
            length_matches[f"{x_key}/{y_key}"] = False
            continue
        matches = rx.n_samples == ry.n_samples
        length_matches[f"{x_key}/{y_key}"] = matches
        if not matches:
            mismatch_errors.append(f"{x_key} has {rx.n_samples} samples but {y_key} has {ry.n_samples}")

    all_ok = all(r.ok for r in parsed.values()) and all(length_matches.values()) and not mismatch_errors
    report = DatasetIntegrityReport(files=file_reports, x_y_length_matches=length_matches,
                                     all_ok=all_ok, level=level)

    if abort_on_failure and not all_ok:
        problems = []
        io_error_hit = False
        for key, r in parsed.items():
            if not r.ok:
                if r.status == STATUS_IO_ERROR:
                    io_error_hit = True
                    problems.append(f"{key} ({r.path}): PHYSICALLY UNREADABLE - {r.errors}")
                else:
                    problems.append(f"{key} ({r.path}): {r.errors}")
        problems.extend(mismatch_errors)
        tail = ""
        if io_error_hit:
            tail = ("\nAt least one file is the correct size on disk but its bytes cannot be read "
                    "(OSError(5)/EIO - bad sectors / failing disk / filesystem corruption). This is "
                    "NOT a truncated write and NOT a DataLoader/`/dev/shm` problem: more workers or "
                    "more shared memory will not help. Re-copy the dataset from a good source. "
                    "Re-run with --dataset_validation_level deep to scan every file end-to-end.")
        raise DatasetStorageError(
            "DATASET_STORAGE_ERROR: one or more .npy files failed integrity validation before "
            "any DataLoader/mmap was created (Phase 4/6 of the SIGBUS remediation plan). "
            "This is a genuine data-storage problem, not a DataLoader worker /dev/shm issue - "
            "do not retry with more workers or more shared memory. Problems found:\n  "
            + "\n  ".join(problems) + tail
        )
    return report


def write_integrity_report(report: DatasetIntegrityReport, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump({"all_ok": report.all_ok, "level": getattr(report, "level", "structural"),
                    "files": report.files, "x_y_length_matches": report.x_y_length_matches},
                   f, indent=2, default=str)


# ----------------------------------------------------------------------
# Phase 5 - explicit dataset storage modes
# ----------------------------------------------------------------------

def resolve_storage_mode(x_path: str, requested: str = "auto", ram_headroom_frac: float = 0.5,
                          verbose: bool = True) -> str:
    """`requested`: "mmap" | "ram" | "auto". "auto" picks "ram" only if the
    array's HEADER-DECLARED payload size (not merely the current on-disk
    size - a partially-written file is *smaller* on disk, which would
    wrongly bias "auto" toward a full RAM load) is comfortably smaller than
    a fraction of currently-available system RAM. Never raises - a RAM-info
    lookup failure just falls back to "mmap" (safer, lower peak memory).

    Call only on a file that has already passed `validate_dataset` -
    this function does not re-validate."""
    if requested in ("mmap", "ram"):
        return requested
    assert requested == "auto", f"Unknown dataset storage mode: {requested!r}"

    try:
        from training.system_memory import get_ram_info
        ram = get_ram_info()
        available_mb = ram.get("available_mb")
        if available_mb is None:
            return "mmap"
        try:
            shape, dtype, _fortran, data_offset = _read_npy_header(x_path)
            payload_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize + int(data_offset)
        except Exception:
            payload_bytes = os.path.getsize(x_path)
        size_mb = payload_bytes / 1e6
        chosen = "ram" if size_mb <= available_mb * ram_headroom_frac else "mmap"
        if verbose:
            print(f"[dataset-storage] auto -> {chosen}: array={size_mb:.0f} MB, "
                  f"available RAM={available_mb:.0f} MB, headroom_frac={ram_headroom_frac}",
                  flush=True)
        return chosen
    except Exception:
        return "mmap"


def _load_ram_buffered(path: str, chunk_bytes: int = 32 << 20) -> np.ndarray:
    """Fully materialize a `.npy` array WITHOUT `np.load`'s single large
    `readinto` - read the payload with a plain `open(...).readinto()` loop so
    a physically-unreadable extent surfaces as a catchable `OSError(5)/EIO`
    in THIS process instead of a SIGBUS (v13.1 remediation). A clean short
    read (genuine truncation) raises `ValueError`."""
    shape, dtype, fortran_order, data_offset = _read_npy_header(path)
    dtype = np.dtype(dtype)
    count = int(np.prod(shape)) if shape else 0
    raw = np.empty(count * dtype.itemsize, dtype=np.uint8)
    mv = memoryview(raw)
    got = 0
    with open(path, "rb", buffering=0) as fh:
        fh.seek(int(data_offset))
        while got < raw.nbytes:
            end = got + min(chunk_bytes, raw.nbytes - got)
            n = fh.readinto(mv[got:end])
            if not n:
                raise ValueError(
                    f"truncated: read {got} of {raw.nbytes} payload bytes from {path}")
            got += n
    arr = raw.view(dtype)
    return arr.reshape(shape, order="F" if fortran_order else "C")


def read_npy_rows(path: str, indices) -> np.ndarray:
    """SIGBUS-free read of specific rows (axis-0 slices) of a `.npy` file via
    `open()/seek()/readinto()` - never mmap. Returns an array of shape
    `(len(indices), *row_shape)`. A physically-unreadable extent raises
    `OSError(5)/EIO`; a short read raises `ValueError`. For code that needs a
    handful of patches out of a multi-GB X without risking a main-process
    SIGBUS on a corrupt/failing-disk file."""
    shape, dtype, fortran_order, data_offset = _read_npy_header(path)
    if fortran_order:
        raise ValueError("read_npy_rows does not support fortran_order .npy files")
    dtype = np.dtype(dtype)
    row_shape = tuple(shape[1:])
    row_elems = int(np.prod(row_shape)) if row_shape else 1
    row_nbytes = row_elems * dtype.itemsize
    idx = [int(i) for i in indices]
    out = np.empty((len(idx), *row_shape), dtype=dtype)
    flat = out.reshape(-1).view(np.uint8).reshape(len(idx), row_nbytes)
    with open(path, "rb", buffering=0) as fh:
        for k, i in enumerate(idx):
            fh.seek(int(data_offset) + i * row_nbytes)
            mv = memoryview(flat[k])
            got = 0
            while got < row_nbytes:
                n = fh.readinto(mv[got:])
                if not n:
                    raise ValueError(f"truncated: row {i} of {path} short by "
                                     f"{row_nbytes - got} of {row_nbytes} bytes")
                got += n
    return out


def copy_npy_payload(src_path: str, dst_array: np.ndarray, row_offset: int, *,
                      chunk_bytes: int = 32 << 20, hasher=None) -> int:
    """Sequentially copy the ENTIRE axis-0 payload of the `.npy` at `src_path`
    into `dst_array` starting at row `row_offset`, via `open()/readinto()` -
    never mmap of the source, so a corrupt / short source raises a catchable
    `ValueError`/`OSError` instead of SIGBUS-ing this process.

    `dst_array` must be C-contiguous with a row shape (`dst_array.shape[1:]`)
    and dtype matching the source. Used by `dataset_prep_unify.unify_split`'s
    identity fast path to stitch in-order shards without the per-row `seek`
    loop of `read_npy_rows`.

    If `hasher` is given (a `hashlib` object), every byte written is fed to it
    IN FILE ORDER - so a caller that seeds it with the destination `.npy`
    header bytes and then copies shards in order gets the full-file sha256 for
    free (no separate read-back pass)."""
    shape, dtype, fortran_order, data_offset = _read_npy_header(src_path)
    if fortran_order:
        raise ValueError(f"copy_npy_payload does not support fortran_order .npy files: {src_path}")
    dtype = np.dtype(dtype)
    if np.dtype(dst_array.dtype) != dtype:
        raise ValueError(f"dtype mismatch: source {dtype} vs destination {dst_array.dtype}")
    n_rows = int(shape[0]) if len(shape) else 0
    row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    row_nbytes = row_elems * dtype.itemsize
    total = n_rows * row_nbytes

    dst_u8 = dst_array.reshape(-1).view(np.uint8)
    start = int(row_offset) * row_nbytes
    if start + total > dst_u8.nbytes:
        raise ValueError(
            f"destination too small: need {total} bytes at offset {start}, "
            f"have {dst_u8.nbytes}")
    mv = memoryview(dst_u8)
    got = 0
    with open(src_path, "rb", buffering=0) as fh:
        fh.seek(int(data_offset))
        while got < total:
            end = got + min(chunk_bytes, total - got)
            n = fh.readinto(mv[start + got:start + end])
            if not n:
                raise ValueError(
                    f"truncated: read {got} of {total} payload bytes from {src_path}")
            if hasher is not None:
                hasher.update(mv[start + got:start + got + n])
            got += n
    return n_rows


def stream_npy_payload(src_path: str, dst_fh, *, chunk_bytes: int = 32 << 20,
                        expect_dtype=None, expect_row_shape: Optional[Sequence[int]] = None,
                        hasher=None) -> int:
    """Sequentially copy the ENTIRE axis-0 payload of the `.npy` at `src_path`
    into the writable binary file object `dst_fh`, via `open()/readinto()` into
    a reusable bounce buffer + `dst_fh.write()` - never mmap on EITHER side.

    This is the streaming sibling of `copy_npy_payload`, which reads into a
    memoryview over an mmap'd destination. Writing through `write()` instead
    hands each page to the kernel once and never touches it again from
    userspace, so kernel writeback cannot race the fill. On btrfs (checksum
    verified on read, no redundancy on a single-device profile) that race
    surfaces later as `OSError(5)/EIO` on perfectly-sized files - which is
    exactly the failure this function exists to avoid. Shards written with
    plain buffered `np.save` never showed it; multi-GB mmap-filled arrays did.

    `expect_dtype` / `expect_row_shape`: when given, the source must match, so
    a mismatched shard is rejected before a single byte is appended to an
    output whose header has already been committed.

    Returns the number of rows copied. Raises `ValueError` on a short/corrupt
    source and lets `OSError` propagate for a physically unreadable one."""
    shape, dtype, fortran_order, data_offset = _read_npy_header(src_path)
    if fortran_order:
        raise ValueError(f"stream_npy_payload does not support fortran_order .npy files: {src_path}")
    dtype = np.dtype(dtype)
    if expect_dtype is not None and np.dtype(expect_dtype) != dtype:
        raise ValueError(f"dtype mismatch: source {src_path} is {dtype}, expected {np.dtype(expect_dtype)}")
    row_shape = tuple(int(s) for s in shape[1:])
    if expect_row_shape is not None and tuple(int(s) for s in expect_row_shape) != row_shape:
        raise ValueError(
            f"row-shape mismatch: source {src_path} has rows {row_shape}, "
            f"expected {tuple(int(s) for s in expect_row_shape)}")

    n_rows = int(shape[0]) if len(shape) else 0
    row_elems = int(np.prod(row_shape)) if row_shape else 1
    total = n_rows * row_elems * dtype.itemsize

    buf = bytearray(min(int(chunk_bytes), total) or 1)
    mv = memoryview(buf)
    got = 0
    with open(src_path, "rb", buffering=0) as fh:
        fh.seek(int(data_offset))
        while got < total:
            want = min(len(mv), total - got)
            n = fh.readinto(mv[:want])
            if not n:
                raise ValueError(
                    f"truncated: read {got} of {total} payload bytes from {src_path}")
            dst_fh.write(mv[:n])
            if hasher is not None:
                hasher.update(mv[:n])
            got += n
    return n_rows


def load_npy_array(path: str, storage_mode: str = "mmap"):
    """The single place that actually decides mmap-vs-materialized for a
    validated `.npy` file. Callers should validate the file with
    `validate_npy_file`/`validate_dataset` FIRST - this function does not
    re-validate, it only loads."""
    if storage_mode == "ram":
        return _load_ram_buffered(path)
    return np.load(path, mmap_mode="r")


# ----------------------------------------------------------------------
# v13 plan, Workstream 3 - dataset failure artifact
# ----------------------------------------------------------------------

def write_dataset_failure_artifacts(out_dir: str, integrity_report: Optional[DatasetIntegrityReport],
                                     exc: BaseException, system_snapshot: Optional[Dict] = None,
                                     extra: Optional[Dict] = None) -> str:
    """Writes `<out_dir>/dataset_failure/{failure.json,validation_report.json,
    system_memory.json}` - the dataset-side analogue of the trainer's
    `numerical_failure/` artifacts. Returns the failure directory path.
    Never raises (best-effort; logs a warning on its own failure)."""
    import datetime as _dt

    failure_dir = os.path.join(out_dir, "dataset_failure")
    try:
        os.makedirs(failure_dir, exist_ok=True)

        try:
            from training.failure_taxonomy import classify_npy_status, classify_exception
        except Exception:
            classify_npy_status = classify_exception = None

        roles = []
        files = (integrity_report.files if integrity_report is not None else {}) or {}
        for role, fr in files.items():
            status = fr.get("status")
            if status in (None, STATUS_VALID):
                continue
            roles.append({
                "role": role,
                "path": fr.get("path"),
                "status": status,
                "expected_shape": fr.get("shape"),
                "dtype": fr.get("dtype"),
                "expected_elements": fr.get("expected_element_count"),
                "readable_elements": fr.get("readable_elements"),
                "expected_bytes": fr.get("expected_byte_size"),
                "expected_file_size": fr.get("expected_file_size"),
                "actual_bytes": fr.get("size_bytes"),
                "errors": fr.get("errors"),
                "warnings": fr.get("warnings"),
                "failure_class": (getattr(classify_npy_status(status), "value", status)
                                  if classify_npy_status else status),
            })

        failure_class = None
        if roles:
            failure_class = roles[0]["failure_class"]
        elif classify_exception:
            failure_class = getattr(classify_exception(exc), "value", None)

        failure = {
            "failure_class": failure_class,
            "exception": f"{type(exc).__name__}: {exc}",
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "all_ok": bool(integrity_report.all_ok) if integrity_report is not None else False,
            "x_y_length_matches": (integrity_report.x_y_length_matches
                                    if integrity_report is not None else {}),
            "failing_roles": roles,
        }
        if extra:
            failure["extra"] = extra
        with open(os.path.join(failure_dir, "failure.json"), "w") as f:
            json.dump(failure, f, indent=2, default=str)

        if integrity_report is not None:
            write_integrity_report(integrity_report, os.path.join(failure_dir, "validation_report.json"))

        if system_snapshot is None:
            try:
                from training.system_memory import collect_system_memory_snapshot
                system_snapshot = collect_system_memory_snapshot()
            except Exception:
                system_snapshot = {}
        with open(os.path.join(failure_dir, "system_memory.json"), "w") as f:
            json.dump(system_snapshot, f, indent=2, default=str)
    except Exception as e:  # pragma: no cover - artifact writer must not mask the real error
        print(f"WARNING: could not write dataset_failure artifacts: {e!r}", flush=True)

    return failure_dir
