# -*- coding: utf-8 -*-
"""
training/prep/shard_writer.py
=============================
Buffered -> atomic `.npy` shard flush. Unlike the per-item writers the
old prep scripts created, ONE `ShardWriter` is owned by a worker for a
whole `(modality, split)` and spans many items, so `--batch_size` actually
bounds the shard count. No `gc.collect()` per flush; shard bodies are
written without `fsync` (the fsync that matters is on the final unified
array).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from training.npy_atomic import save_npy_atomic
from training.prep.adapter import ModalitySpec


class Float16OverflowError(RuntimeError):
    """A sample value did not fit float16 (|x| > 65504) - the modality must
    be stored as float32 (pass --store_dtype float32)."""


class ShardWriter:
    def __init__(self, split_dir: str, prefix: str, batch_size: int,
                 spec: Optional[ModalitySpec] = None):
        self.split_dir = split_dir
        os.makedirs(split_dir, exist_ok=True)
        self.prefix = prefix                       # per-worker namespace, e.g. "w03_9f1c2a"
        self.batch_size = max(1, int(batch_size))
        self.spec = spec or ModalitySpec()
        self.dtype = np.dtype(self.spec.store_dtype)
        self.shard_id = 0
        self.buf_x: List[np.ndarray] = []
        self.buf_y: List[int] = []
        self.written: List[Tuple[str, str, int]] = []   # (x_path, y_path, n)

    def add(self, arr: np.ndarray, label: int) -> None:
        self.buf_x.append(arr)
        self.buf_y.append(int(label))
        if len(self.buf_x) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buf_x:
            return
        x = np.stack(self.buf_x)
        if self.spec.value_clip is not None:
            lo, hi = self.spec.value_clip
            np.clip(x, lo, hi, out=x)
        if x.dtype != self.dtype:
            if self.dtype == np.float16 and np.isfinite(x).all():
                if float(np.abs(x, dtype=np.float64).max() if x.size else 0.0) > 65504.0:
                    raise Float16OverflowError(
                        f"value {np.abs(x).max():.1f} exceeds float16 range in "
                        f"{os.path.basename(self.split_dir)}; use --store_dtype float32")
            x = x.astype(self.dtype)
        name = f"{self.prefix}_s{self.shard_id:06d}"
        xp = os.path.join(self.split_dir, name + "_X.npy")
        yp = os.path.join(self.split_dir, name + "_y.npy")
        save_npy_atomic(xp, x, do_fsync=False)
        save_npy_atomic(yp, np.asarray(self.buf_y, dtype=np.int64), do_fsync=False)
        self.written.append((xp, yp, len(self.buf_x)))
        self.shard_id += 1
        self.buf_x, self.buf_y = [], []

    def finalize(self) -> List[Tuple[str, str, int]]:
        self._flush()
        return self.written
