# -*- coding: utf-8 -*-
"""
training/npy_atomic.py
=======================
MedMamba-SS-TRM v13 stability plan, Workstream 6 - dataset-generation hardening.

The v13 incident (`X_train.npy` only ~19.6% written, yet present under its
final filename) is the classic symptom of a preprocessing job that was
interrupted, ran out of disk, or crashed *while* `np.save` was streaming
the array to its final path. Two cheap guarantees prevent it:

  1. `save_npy_atomic(path, array)` - write to `<path>.tmp`, flush + fsync,
     then `os.replace` (atomic on POSIX). A crash mid-write leaves only the
     `.tmp`; the real filename never exists until the bytes are all on disk.

  2. `write_dataset_manifest(data_dir, {...})` / `verify_against_manifest(
     data_dir)` - a `dataset_manifest.json` recording each file's shape,
     dtype, byte size (and optionally sha256). The training entry point
     cross-checks it before building any Dataset, so a silently
     half-replaced dataset is caught immediately.

`training/npy_integrity.py` still does the header/size/probe structural
validation at train time - this module is the *write-side* complement.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

MANIFEST_NAME = "dataset_manifest.json"


def save_npy_atomic(path: str, array: np.ndarray, *, do_fsync: bool = True) -> str:
    """`np.save` to `<path>.tmp`, flush (+ fsync), then atomically rename to
    `path`. Returns `path`. On any exception the partial `.tmp` is removed."""
    path = os.fspath(path)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "wb") as f:
            np.save(f, array, allow_pickle=False)
            f.flush()
            if do_fsync:
                os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX; replaces any existing file
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def _sha256_of_file(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _describe_npy(path: str, *, sha256: bool = False, precomputed_sha256: Optional[str] = None) -> Dict:
    from training.npy_integrity import _read_npy_header
    shape, dtype, fortran_order, data_offset = _read_npy_header(path)
    entry = {
        "file": os.path.basename(path),
        "shape": list(shape),
        "dtype": str(np.dtype(dtype)),
        "fortran_order": bool(fortran_order),
        "header_offset": int(data_offset),
        "size_bytes": int(os.path.getsize(path)),
        "expected_file_size": int(data_offset) + int(np.prod(shape)) * np.dtype(dtype).itemsize,
        "complete": True,
    }
    entry["complete"] = entry["size_bytes"] >= entry["expected_file_size"]
    if sha256:
        # A caller that hashed the bytes while writing the file (see
        # `dataset_prep_unify.atomic_array_writer`) passes the digest here so
        # we don't re-read a multi-GB array just to hash it.
        entry["sha256"] = precomputed_sha256 or _sha256_of_file(path)
    return entry


def write_dataset_manifest(data_dir: str, files: Iterable[str], *, sha256: bool = False,
                            extra: Optional[Dict] = None,
                            precomputed_sha256: Optional[Dict[str, str]] = None) -> str:
    """`files`: iterable of `.npy` basenames (or absolute paths) under
    `data_dir`. Writes `<data_dir>/dataset_manifest.json`. Returns its path.

    `precomputed_sha256`: optional `{basename: hexdigest}` for files whose
    sha256 was already computed during the write - those are not re-hashed."""
    entries: Dict[str, Dict] = {}
    precomputed_sha256 = precomputed_sha256 or {}
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(data_dir, f)
        if not os.path.isfile(p):
            continue
        entries[os.path.basename(p)] = _describe_npy(
            p, sha256=sha256, precomputed_sha256=precomputed_sha256.get(os.path.basename(p)))
    manifest = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sha256": bool(sha256),
        "files": entries,
    }
    if extra:
        manifest["extra"] = extra
    out = os.path.join(data_dir, MANIFEST_NAME)
    save_json_atomic(out, manifest)
    return out


def save_json_atomic(path: str, obj) -> str:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def verify_against_manifest(data_dir: str, *, check_sha256: Optional[bool] = None) -> List[str]:
    """Returns a list of human-readable mismatch strings (empty == OK, or no
    manifest present). Compares shape / dtype / byte size for every file the
    manifest lists; verifies sha256 too when the manifest recorded it (unless
    `check_sha256=False` forces a fast structural-only check)."""
    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return []
    problems: List[str] = []
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        return [f"{MANIFEST_NAME} is unreadable: {e}"]

    want_sha = manifest.get("sha256", False) if check_sha256 is None else check_sha256
    for name, entry in (manifest.get("files") or {}).items():
        p = os.path.join(data_dir, name)
        if not os.path.isfile(p):
            problems.append(f"{name}: listed in manifest but missing on disk")
            continue
        # Header parse (cheap: reads only the .npy header, ~128 bytes) and
        # payload hashing (reads the WHOLE file) are kept in separate try
        # blocks: a payload that is physically unreadable (OSError(5)/EIO -
        # bad sectors / a bad write on a large mmap-filled array, see
        # `dataset_prep_unify.atomic_npy_stream`) used to surface here as
        # "header unreadable", which is misleading - the header parses fine
        # and points investigation at the wrong layer entirely.
        try:
            actual = _describe_npy(p, sha256=False)
        except Exception as e:
            problems.append(f"{name}: header unreadable ({e})")
            continue
        if entry.get("shape") is not None and actual["shape"] != list(entry["shape"]):
            problems.append(f"{name}: shape {actual['shape']} != manifest {entry['shape']}")
        if entry.get("dtype") is not None and actual["dtype"] != entry["dtype"]:
            problems.append(f"{name}: dtype {actual['dtype']} != manifest {entry['dtype']}")
        if entry.get("size_bytes") is not None and actual["size_bytes"] != entry["size_bytes"]:
            problems.append(
                f"{name}: size {actual['size_bytes']} bytes != manifest {entry['size_bytes']} "
                f"(file is {'truncated' if actual['size_bytes'] < entry['size_bytes'] else 'larger than'} expected)")
        if want_sha and entry.get("sha256"):
            try:
                actual_sha256 = _sha256_of_file(p)
            except Exception as e:
                problems.append(f"{name}: payload unreadable while hashing ({e})")
                continue
            if actual_sha256 != entry["sha256"]:
                problems.append(f"{name}: sha256 mismatch (content differs from manifest)")
    return problems
