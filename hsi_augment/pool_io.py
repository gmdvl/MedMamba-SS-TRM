# -*- coding: utf-8 -*-
"""
hsi_augment/pool_io.py
======================
The generator pool's file contract (Plan §Phase 4, item 3), read by every
generator stage (#23 CVAE, #24 AE/LDM) and written by the pool stage (#16):

    <out>/pool/
        X_pool.npy        float16 (N, H, W, C), NHWC, ALREADY ENCODED by the codec
                          below (log-space z-score, `LogSpaceCodec.encode_np`)
        y_pool.npy        int64 (N,)   class index
        patient_pool.npy  int32 (N,)   patient id (the build's groups_train value)
        row_pool.npy      int64 (N,)   the row in X_train it was copied from
        capture_pool.npy  int32 (N,)   capture index (the build's captures_train value)
        codec.json        `LogSpaceCodec.to_dict()`

    pool = load_pool(out / "pool")          # X memmapped, sidecars in RAM, codec rebuilt
    train_mask, held_mask = splits.fold_masks(pool.patient, fold)

Why a separate module
---------------------
The pool stage and the generator stages are written in parallel branches; a
loader both sides import is the contract between them, checked in one place.
`load_pool` validates every dtype and shape (and that the codec has the
pool's band count) before a multi-hour fit starts, so a malformed pool fails
in a second, not after the first checkpoint. `write_pool` writes exactly
this layout atomically (the test fixtures use it; #16 may too).

What it does NOT check
----------------------
That the pool belongs to the build at `--data_dir`: that needs the train
split's sidecars, so the generator stage does it (`check_against_train`,
called with arrays it opened through its SplitGuard). And `X_pool` is not
scanned for NaN here (a 3 GB read); the pool stage writes it from finite,
encoded values and the trainers refuse a non-finite loss.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

from hsi_augment.codec import LogSpaceCodec
from hsi_augment.io import save_json, save_npy, sha256_file

PathLike = Union[str, os.PathLike]

POOL_FILES: Dict[str, str] = {"X": "X_pool.npy", "y": "y_pool.npy", "patient": "patient_pool.npy",
                              "row": "row_pool.npy", "capture": "capture_pool.npy"}
CODEC_NAME = "codec.json"
POOL_DTYPES: Dict[str, np.dtype] = {"X": np.dtype(np.float16), "y": np.dtype(np.int64),
                                    "patient": np.dtype(np.int32), "row": np.dtype(np.int64),
                                    "capture": np.dtype(np.int32)}
SIDECARS = ("y", "patient", "row", "capture")


class PoolError(ValueError):
    """The pool directory does not follow the contract in the module docstring."""


@dataclass(frozen=True, eq=False)
class Pool:
    """A loaded pool. `X` is a read-only memmap (unless loaded with
    `mmap=False`); the int sidecars are in RAM (a few MB)."""
    pool_dir: Path
    X: np.ndarray
    y: np.ndarray
    patient: np.ndarray
    row: np.ndarray
    capture: np.ndarray
    codec: LogSpaceCodec

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def patch(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.X.shape[3])

    def describe(self) -> Dict[str, Any]:
        classes, counts = np.unique(self.y, return_counts=True)
        return {"pool_dir": str(self.pool_dir), "n": self.n, "x_shape": [int(s) for s in self.X.shape],
                "x_dtype": str(self.X.dtype), "rows_per_class": {int(c): int(k) for c, k in zip(classes, counts)},
                "patients": int(np.unique(self.patient).size), "captures": int(np.unique(self.capture).size)}

    def fingerprint(self) -> str:
        """sha256 over the sha256 of every pool file (X included, streamed):
        what a fit records, so a later fit on another pool is told apart."""
        return pool_fingerprint(self.pool_dir)


def _check_arrays(X: np.ndarray, side: Mapping[str, np.ndarray], codec: LogSpaceCodec,
                  where: str) -> None:
    """The dtype/shape contract. Raises PoolError naming every problem."""
    problems: List[str] = []
    if X.dtype != POOL_DTYPES["X"]:
        problems.append(f"X_pool is {X.dtype}, expected float16")
    if X.ndim != 4:
        problems.append(f"X_pool has shape {X.shape}, expected NHWC (N, H, W, C)")
    else:
        if X.shape[1] != X.shape[2]:
            problems.append(f"X_pool patches are {X.shape[1]}x{X.shape[2]}, expected square")
        if X.shape[3] != codec.n_bands:
            problems.append(f"X_pool has {X.shape[3]} bands, codec.json {codec.n_bands}")
        if X.shape[0] == 0:
            problems.append("X_pool has no rows")
    n = int(X.shape[0]) if X.ndim else -1
    for k in SIDECARS:
        a = side[k]
        if a.dtype != POOL_DTYPES[k]:
            problems.append(f"{POOL_FILES[k]} is {a.dtype}, expected {POOL_DTYPES[k]}")
        if a.ndim != 1 or a.shape[0] != n:
            problems.append(f"{POOL_FILES[k]} has shape {a.shape}, expected ({n},)")
    if not problems:
        for k in SIDECARS:
            if side[k].size and int(side[k].min()) < 0:
                problems.append(f"{POOL_FILES[k]} has negative values")
        if np.unique(side["row"]).size != n:
            problems.append("row_pool.npy has duplicate rows (a train row pooled twice)")
    if problems:
        raise PoolError(f"{where}: " + "; ".join(problems))


def load_pool(pool_dir: PathLike, *, mmap: bool = True) -> Pool:
    """Open and validate a pool directory (contract in the module docstring).
    FileNotFoundError if the directory or a file is missing, PoolError if a
    file breaks the contract."""
    d = Path(pool_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"no pool directory at {d} (run the pool stage, #16, first)")
    missing = [n for n in (*POOL_FILES.values(), CODEC_NAME) if not (d / n).is_file()]
    if missing:
        raise FileNotFoundError(f"pool {d} is missing {missing}")
    try:
        with open(d / CODEC_NAME) as f:
            codec = LogSpaceCodec.from_dict(json.load(f))
    except (ValueError, KeyError, TypeError) as e:
        raise PoolError(f"{d / CODEC_NAME}: not a LogSpaceCodec dict ({e})") from e
    X = np.load(d / POOL_FILES["X"], mmap_mode="r" if mmap else None)
    side = {k: np.load(d / POOL_FILES[k]) for k in SIDECARS}
    _check_arrays(X, side, codec, str(d))
    return Pool(pool_dir=d, X=X, codec=codec, **side)


def write_pool(pool_dir: PathLike, X: np.ndarray, y: np.ndarray, patient: np.ndarray, row: np.ndarray,
               capture: np.ndarray, codec: LogSpaceCodec) -> Path:
    """Write a pool in the contract's layout, each file atomically, after
    validating it. `X` must already be encoded (float16 log-z); the sidecars
    are cast to their contract dtypes. Returns the directory."""
    d = Path(pool_dir)
    side = {"y": np.asarray(y).astype(np.int64), "patient": np.asarray(patient).astype(np.int32),
            "row": np.asarray(row).astype(np.int64), "capture": np.asarray(capture).astype(np.int32)}
    X = np.asarray(X)
    _check_arrays(X, side, codec, f"write_pool({d})")
    d.mkdir(parents=True, exist_ok=True)
    (d / POOL_FILES["X"]).unlink(missing_ok=True)   # an old X next to new sidecars would look complete
    save_json(d / CODEC_NAME, codec.to_dict())
    for k in SIDECARS:
        save_npy(d / POOL_FILES[k], side[k])
    save_npy(d / POOL_FILES["X"], X)          # last: a pool with X present is complete
    return d


def pool_fingerprint(pool_dir: PathLike) -> str:
    """sha256 of the sorted (name, sha256) of the pool's six files."""
    d = Path(pool_dir)
    parts = {n: sha256_file(d / n) for n in sorted((*POOL_FILES.values(), CODEC_NAME))}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def check_against_train(pool: Pool, y_train: np.ndarray, groups_train: np.ndarray,
                        captures_train: Optional[np.ndarray] = None) -> List[str]:
    """Problems (empty = none) with the pool as a subset of THIS build's train
    split: every `row` exists, and its class, patient and capture agree with
    the train sidecars. A pool from another build (or another split) fails."""
    n_train = int(np.asarray(y_train).shape[0])
    if pool.row.size and int(pool.row.max()) >= n_train:
        return [f"row_pool.npy points past the {n_train} train rows: the pool is from another build"]
    out = []
    checks = [("y", pool.y, y_train), ("patient", pool.patient, groups_train)]
    if captures_train is not None:
        checks.append(("capture", pool.capture, captures_train))
    for name, have, ref in checks:
        bad = int(np.count_nonzero(np.asarray(ref)[pool.row].astype(np.int64) != have.astype(np.int64)))
        if bad:
            out.append(f"{bad} pool rows disagree with the train split's {name} at row_pool")
    return out
