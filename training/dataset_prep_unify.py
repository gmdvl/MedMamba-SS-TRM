# -*- coding: utf-8 -*-
"""
training/dataset_prep_unify.py
==============================
Crash-safe "unify shards -> final X_*/y_*.npy" for every `prepare_*.py`
dataset script, plus a read-back verification + provenance manifest step.

WHY THIS EXISTS
---------------
Both prep scripts used to build the final array with
`np.lib.format.open_memmap(path, mode="w+")`, which `ftruncate`s the file to
its full, header-consistent size *before the first patch is written*. If the
process is then killed (SIGKILL / OOM-killer / disk-full / power loss)
anywhere in the fill loop, `X_train.npy` is left at exactly the right size
with an unwritten (zero) tail - a file that passes every structural check
and only surfaces as a SIGBUS or a misleading "file seems not fully
written?" error days later inside training. On top of that the shards were
deleted immediately after unify with no verification, so there was no way
to re-unify.

This module fixes both:

  * `atomic_memmap_array(final_path, ...)` builds into `<final>.tmp`, then
    `flush()` + `fsync()` + `os.replace()` - the real filename never exists
    until every byte is on disk. A crash leaves only the `.tmp`.
  * `unify_split(...)` reads source shards with
    `training.npy_integrity.read_npy_rows` (`open()/seek()/readinto()`,
    never mmap), so a corrupt shard raises a catchable error instead of
    SIGBUS-ing the prep process.
  * `verify_unified_dataset(...)` re-reads every produced file end-to-end
    (`validate_dataset(level="deep")`) and rejects an all-zero X (the
    killed-`open_memmap` signature that a deep read alone would accept).
  * `finalize_dataset(...)` = verify + write a sha256 `dataset_manifest.json`
    (via `training.npy_atomic.write_dataset_manifest`). The caller only
    deletes shards / marks the run finalized once this returns True.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from training.npy_integrity import (
    DatasetStorageError, _read_npy_header, read_npy_rows, stream_npy_payload, validate_dataset,
)
from training.npy_atomic import save_npy_atomic, write_dataset_manifest

ShardEntry = Tuple[str, str, int]  # (x_path, y_path, n_rows)


# ----------------------------------------------------------------------
# atomic final-array writer
# ----------------------------------------------------------------------

@contextlib.contextmanager
def atomic_memmap_array(final_path: str, dtype, shape: Sequence[int], *, fortran_order: bool = False):
    """Context manager yielding a writable `np.memmap` backed by
    `<final_path>.tmp`. On clean exit: `flush()` -> drop the mapping ->
    `fsync` the tmp file -> `os.replace(tmp, final_path)` (atomic on POSIX).
    On ANY exception the `.tmp` is removed and `final_path` is left untouched
    (an existing previous version stays intact)."""
    final_path = os.fspath(final_path)
    tmp = final_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.dtype(dtype),
                                    shape=tuple(int(s) for s in shape), fortran_order=fortran_order)
    ok = False
    try:
        yield arr
        ok = True
    finally:
        try:
            arr.flush()
        except Exception:
            pass
        del arr
        gc.collect()
        if ok:
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, final_path)
        else:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


class _ArrayWriteHolder:
    """Filled in by `atomic_array_writer`: `sha256` is the full-file hexdigest
    (header + payload) once the write commits cleanly, or None if hashing was
    not requested / the payload was not written strictly in file order."""
    __slots__ = ("hasher", "sha256")

    def __init__(self):
        self.hasher = None
        self.sha256 = None


def _write_npy_header(fh, dtype, shape: Sequence[int], fortran_order: bool) -> int:
    """Writes a `.npy` header for `(dtype, shape)` to the binary handle `fh`
    using numpy's own writer, so the resulting `header_offset` matches what
    `open_memmap`/`np.save` produce (128 for every array in this project) and
    older manifests stay comparable. Returns the header length in bytes."""
    d = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
         "fortran_order": bool(fortran_order),
         "shape": tuple(int(s) for s in shape)}
    try:
        np.lib.format.write_array_header_1_0(fh, d)
    except ValueError:
        # header too long for the 1.0 (uint16) length field
        np.lib.format.write_array_header_2_0(fh, d)
    return int(fh.tell())


@contextlib.contextmanager
def atomic_npy_stream(final_path: str, dtype, shape: Sequence[int], *,
                       fortran_order: bool = False, sha256: bool = False):
    """Yields `(fh, holder)`: a binary handle on `<final_path>.tmp` positioned
    just past a fully-written `.npy` header. The caller appends the payload
    strictly front-to-back with `fh.write()`. On clean exit the payload length
    is checked, then `flush()` -> `fsync()` -> `os.replace(tmp, final_path)`.
    On ANY exception the `.tmp` is removed and `final_path` is untouched.

    WHY THIS EXISTS (and why it is preferred over `atomic_array_writer`):
    filling a multi-GB array through `open_memmap` leaves userspace holding
    dirty mmap pages that the kernel writes back CONCURRENTLY with the fill
    once the mapping outgrows RAM. On btrfs that race has been observed to
    commit bytes that disagree with the checksum computed at writeback, and
    the file then returns `OSError(5)/EIO` on read despite being the right
    size. Handing every page to the kernel once via `write()`, and never
    touching it again, removes the race. Measured on this project: 45 GiB of
    buffered-`np.save` shards were clean in the same run where a 35 GiB
    mmap-filled `X_train.npy` had six unreadable regions.

    It also strengthens the crash-safety this module was written for: a
    killed `open_memmap` fill leaves a FULL-SIZE file with a zero tail that
    passes structural checks, whereas a killed stream leaves a short `.tmp`
    that never gets renamed - and the length check below catches a short fill
    that somehow reaches a clean exit.

    `sha256=True` seeds `holder.hasher` with the header bytes; a caller that
    feeds it the payload (e.g. `stream_npy_payload(..., hasher=holder.hasher)`)
    gets `holder.sha256` on commit with no read-back pass."""
    final_path = os.fspath(final_path)
    tmp = final_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    expected_payload = int(np.prod(shape)) * dtype.itemsize if shape else 0

    holder = _ArrayWriteHolder()
    fh = open(tmp, "wb")
    ok = False
    try:
        # Serialise the header once into memory so the identical bytes seed the
        # hasher and land in the file (no re-read of the header, unlike
        # atomic_array_writer).
        hdr_buf = io.BytesIO()
        _write_npy_header(hdr_buf, dtype, shape, fortran_order)
        hdr = hdr_buf.getvalue()
        fh.write(hdr)
        data_offset = len(hdr)
        if sha256:
            holder.hasher = hashlib.sha256()
            holder.hasher.update(hdr)

        yield fh, holder

        written = fh.tell() - data_offset
        if written != expected_payload:
            raise DatasetStorageError(
                f"DATASET_STORAGE_ERROR: {final_path}: streamed {written} payload bytes but the "
                f"header declares {expected_payload} (shape={shape}, dtype={dtype}) - refusing to "
                f"commit a partially-filled array.")
        fh.flush()
        os.fsync(fh.fileno())
        ok = True
    finally:
        try:
            fh.close()
        except Exception:
            pass
        if ok:
            os.replace(tmp, final_path)
            if holder.hasher is not None:
                holder.sha256 = holder.hasher.hexdigest()
        else:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


@contextlib.contextmanager
def atomic_array_writer(final_path: str, dtype, shape: Sequence[int], *,
                         fortran_order: bool = False, sha256: bool = False):
    """DEPRECATED for large arrays - prefer `atomic_npy_stream`, which does not
    expose the mmap-writeback race described there. Kept because it is the
    documented API, is exercised by the test suite, and makes reverting the
    streaming switch a one-line change.

    Like `atomic_memmap_array`, but yields `(memmap, holder)` and can hash
    the file as it is written. When `sha256=True` the holder's `hasher` is a
    `hashlib.sha256` pre-seeded with the destination `.npy` header bytes; a
    caller that then fills the payload strictly front-to-back (feeding the same
    hasher, e.g. via `copy_npy_payload(..., hasher=holder.hasher)`) gets
    `holder.sha256` for free on clean exit - no read-back pass."""
    final_path = os.fspath(final_path)
    tmp = final_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.dtype(dtype),
                                    shape=tuple(int(s) for s in shape), fortran_order=fortran_order)
    holder = _ArrayWriteHolder()
    if sha256:
        _sh, _dt, _fo, data_offset = _read_npy_header(tmp)
        h = hashlib.sha256()
        with open(tmp, "rb") as fh:
            h.update(fh.read(int(data_offset)))
        holder.hasher = h

    ok = False
    try:
        yield arr, holder
        ok = True
    finally:
        try:
            arr.flush()
        except Exception:
            pass
        del arr
        gc.collect()
        if ok:
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, final_path)
            if holder.hasher is not None:
                holder.sha256 = holder.hasher.hexdigest()
        else:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


# ----------------------------------------------------------------------
# shard -> final unify
# ----------------------------------------------------------------------

def _load_shard_labels(y_path: str, expected_n: int) -> np.ndarray:
    try:
        yb = np.load(y_path)
    except (OSError, ValueError) as e:
        raise DatasetStorageError(
            f"DATASET_STORAGE_ERROR: unreadable shard label file {y_path}: {e}") from e
    yb = np.asarray(yb).reshape(-1)
    if len(yb) != expected_n:
        raise DatasetStorageError(
            f"DATASET_STORAGE_ERROR: shard {y_path} has {len(yb)} labels but the manifest "
            f"records {expected_n} - shard set is inconsistent, refusing to unify.")
    return yb.astype(np.int64, copy=False)


def unify_split(shard_entries: Sequence[ShardEntry], out_x_path: str, out_y_path: str, *,
                 row_order: Optional[Sequence[int]] = None, balance_method: Optional[str] = None,
                 seed: int = 0, class_names: Optional[List[str]] = None,
                 read_batch: int = 8192, sha_out: Optional[Dict[str, str]] = None
                 ) -> Tuple[int, Optional[Dict]]:
    """Stitch every shard listed in `shard_entries` (each `(x_path, y_path,
    n)`) into `out_x_path` / `out_y_path`, written atomically.

    `row_order`  - explicit global-row order/selection (may repeat rows).
    `balance_method` - "undersample"/"oversample": derive `row_order` from the
                       concatenated labels via
                       `training.dataset_prep_common.build_balance_selection`
                       (TRAIN split only - callers pass None for val/test).
    `sha_out`    - optional dict; when given AND the identity fast path is
                   taken (no `row_order`, no balancing) it is populated with
                   `{basename(out_x_path): hexdigest}` computed during the
                   write, so `finalize_dataset(precomputed_sha256=...)` can
                   skip re-reading the array to hash it.
    Returns `(n_written, balance_report_or_None)`.
    """
    if not shard_entries:
        return 0, None

    shape0, dtype0, fortran0, _off = _read_npy_header(shard_entries[0][0])
    if fortran0:
        raise DatasetStorageError(
            f"DATASET_STORAGE_ERROR: shard {shard_entries[0][0]} is fortran-order - unsupported.")
    patch_shape = tuple(int(s) for s in shape0[1:])
    dtype = np.dtype(dtype0)

    total_n = int(sum(n for _, _, n in shard_entries))
    all_y = np.empty((total_n,), dtype=np.int64)
    cum: List[int] = [0]
    off = 0
    for x_path, y_path, n in shard_entries:
        all_y[off:off + n] = _load_shard_labels(y_path, n)
        off += n
        cum.append(off)

    # ------------------------------------------------------------------
    # Identity fast path: no explicit row_order and no balancing -> the
    # output is just the shards concatenated in order. Stream each shard's
    # whole payload straight through to the output file with a chunked
    # readinto/write (SIGBUS-safe on the source, no mmap on the destination -
    # see `atomic_npy_stream` for why that matters -, no O(out_n*n_shards)
    # np.nonzero scan), optionally hashing as we go.
    # ------------------------------------------------------------------
    identity = row_order is None and (balance_method is None or balance_method == "none")
    if identity:
        want_sha = sha_out is not None
        with atomic_npy_stream(out_x_path, dtype, (total_n,) + patch_shape,
                                sha256=want_sha) as (out_fh, holder):
            off = 0
            for x_path, _y_path, n in shard_entries:
                copied = stream_npy_payload(x_path, out_fh, expect_dtype=dtype,
                                             expect_row_shape=patch_shape, hasher=holder.hasher)
                if copied != n:
                    raise DatasetStorageError(
                        f"DATASET_STORAGE_ERROR: shard {x_path} holds {copied} rows but the "
                        f"manifest records {n} - shard set is inconsistent, refusing to unify.")
                off += n
        save_npy_atomic(out_y_path, all_y.astype(np.int64, copy=False))
        if want_sha and holder.sha256 is not None:
            sha_out[os.path.basename(out_x_path)] = holder.sha256
        return total_n, None

    balance_report: Optional[Dict] = None
    if balance_method and balance_method != "none":
        from training.dataset_prep_common import build_balance_selection
        selected, before, after = build_balance_selection(all_y, balance_method, seed)
        if class_names:
            before = {class_names[c]: v for c, v in before.items()}
            after = {class_names[c]: v for c, v in after.items()}
        balance_report = {"method": balance_method, "before": before, "after": after}
        row_order = selected

    row_order = np.arange(total_n) if row_order is None else np.asarray(row_order, dtype=np.int64)
    out_n = int(len(row_order))
    out_y = all_y[row_order].astype(np.int64)

    cum_arr = np.asarray(cum)
    src_shard = np.searchsorted(cum_arr, row_order, side="right") - 1

    # Scatter/balanced path: output rows are a permutation/selection of the
    # concatenated shards, so unlike the identity path above we cannot just
    # stream shards through in order. Still write the destination
    # sequentially (never mmap - see `atomic_npy_stream`) by walking the
    # OUTPUT in `read_batch`-sized windows and, for each window, gathering
    # the (possibly several) source shards it draws from via the SIGBUS-safe
    # `read_npy_rows`. Peak memory per iteration is unchanged from the
    # pre-existing mmap version: one `read_batch`-sized row array.
    with atomic_npy_stream(out_x_path, dtype, (out_n,) + patch_shape) as (out_fh, _holder):
        for b in range(0, out_n, read_batch):
            sel = row_order[b:b + read_batch]
            shard_of = src_shard[b:b + read_batch]
            batch = np.empty((len(sel),) + patch_shape, dtype=dtype)
            for s in np.unique(shard_of):
                m = shard_of == s
                x_path = shard_entries[int(s)][0]
                local_idx = (sel[m] - cum_arr[s]).astype(np.int64)
                batch[m] = read_npy_rows(x_path, local_idx)  # SIGBUS-safe
            out_fh.write(batch.tobytes())

    save_npy_atomic(out_y_path, out_y)
    return out_n, balance_report


# ----------------------------------------------------------------------
# read-back verification + provenance manifest
# ----------------------------------------------------------------------

def verify_unified_dataset(data_dir: str, roles: Dict[str, str], *, level: str = "deep",
                            collect_sha256: bool = False) -> Optional[Dict[str, str]]:
    """Re-read every file just written. Raises `DatasetStorageError` on any
    structural/IO failure (via `validate_dataset(level=...)`) OR if an `X_*`
    file is entirely zero across a spread of probed rows - the signature of
    an `open_memmap` fill that was killed before writing any data.

    `collect_sha256`: only takes effect at `level="deep"`, which already reads
    every payload byte back - folding a sha256 into that same pass costs no
    extra I/O. Returns `{basename(path): hexdigest}` for every role hashed
    this way, or `None` when not requested / not at deep level. This digest
    reflects the bytes ACTUALLY ON DISK, unlike a digest computed in RAM
    while writing (which can never detect a write that landed wrong)."""
    report = validate_dataset(roles, abort_on_failure=True, level=level,
                               compute_sha256=collect_sha256)

    sha_map: Optional[Dict[str, str]] = None
    if collect_sha256 and level == "deep":
        sha_map = {}
        for role, path in roles.items():
            digest = report.files.get(role, {}).get("sha256")
            if digest:
                sha_map[os.path.basename(path)] = digest

    for role, path in roles.items():
        if not os.path.basename(role).lower().startswith("x"):
            continue
        shp, _dt, _fo, _off = _read_npy_header(path)
        n = int(shp[0]) if len(shp) else 0
        if n == 0:
            continue
        probe = sorted(set(int(i) for i in np.linspace(0, n - 1, num=min(n, 64)).astype(int)))
        rows = read_npy_rows(path, probe)
        if not np.any(rows != 0):
            raise DatasetStorageError(
                f"DATASET_STORAGE_ERROR: {role} ({path}) is entirely zero across {len(probe)} "
                f"probed rows spanning the whole file - this is the signature of a unify that "
                f"was killed after the file was pre-sized but before data was written. "
                f"Treating as corrupt; the source shards were NOT deleted - re-run --finalize_only.")

    return sha_map


def finalize_dataset(data_dir: str, roles: Dict[str, str], *, extra_files: Sequence[str] = (),
                      manifest_extra: Optional[Dict] = None, verify: bool = True,
                      verify_level: str = "deep", manifest_sha256: bool = True,
                      precomputed_sha256: Optional[Dict[str, str]] = None) -> bool:
    """verify (unless `verify=False`) -> write `<data_dir>/dataset_manifest.json`
    (sha256) covering `roles` + `extra_files` (e.g. wavelengths.npy). Returns
    True on success; raises `DatasetStorageError` otherwise. The caller must
    only delete shards / set `finalized=True` when this returns True.

    `verify_level`: "deep" (full read-back), "probe"/"structural" (header +
    size + row probe + the all-zero check, no end-to-end read).
    `precomputed_sha256`: `{basename: hexdigest}` for arrays hashed during the
    write (see `unify_split(sha_out=...)`) - used as a FALLBACK only. When
    `verify_level="deep"` this function re-hashes during the read-back it
    already performs and prefers THAT digest, because a digest computed in
    RAM while writing describes what was intended, not what is on disk, and
    so can never catch a bad write - only a digest read back from disk can."""
    on_disk_sha256: Optional[Dict[str, str]] = None
    if verify:
        on_disk_sha256 = verify_unified_dataset(data_dir, roles, level=verify_level,
                                                  collect_sha256=manifest_sha256)

    files = [os.path.basename(p) for p in roles.values() if os.path.isfile(p)]
    for f in extra_files:
        p = f if os.path.isabs(f) else os.path.join(data_dir, f)
        if os.path.isfile(p):
            files.append(os.path.basename(p))

    sha_map = dict(precomputed_sha256 or {})
    if on_disk_sha256:
        sha_map.update(on_disk_sha256)

    write_dataset_manifest(data_dir, files, sha256=manifest_sha256, extra=manifest_extra,
                            precomputed_sha256=sha_map)
    return True


def dataset_is_finalized_and_intact(data_dir: str, roles: Dict[str, str], *,
                                     level: str = "structural") -> bool:
    """Cheap guard for the `--finalize_only` / already-finalized fast path:
    True only if every file in `roles` exists AND passes validation. If it
    returns False while shards are still on disk, the caller should re-unify
    rather than trusting a stale `finalized: true`."""
    for p in roles.values():
        if not os.path.isfile(p):
            return False
    try:
        verify_unified_dataset(data_dir, roles, level=level)
    except DatasetStorageError:
        return False
    return True
