# -*- coding: utf-8 -*-
"""
training/prep/parallel.py
=========================
Multi-process per-item extraction. Workers write their OWN shard files
(per-worker prefix) into `<out>/<mod>/_batches/<split>/` and return only
shard-path metadata - large arrays never cross the pipe. Out-of-order
completion and mid-run kills are safe: the parent records an item as
completed only after its worker returns, and `cleanup_orphan_shards`
sweeps shards from interrupted items on the next run.
"""

from __future__ import annotations

import os
import re
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from training.prep.adapter import DatasetAdapter, ModalitySpec, PrepItem
from training.prep.shard_writer import Float16OverflowError, ShardWriter

_WORKER: dict = {}


@dataclass
class ItemResult:
    key: str
    split: str
    modalities: Dict[str, List[tuple]] = field(default_factory=dict)   # mod -> [(x,y,n), ...]
    counts: Dict[str, int] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class SkipRecord:
    key: str
    reason: str
    stage: str = "load_item"


@dataclass
class FatalItemError:
    key: str
    message: str


def _init_worker(adapter, args, state):
    _WORKER["adapter"] = adapter
    _WORKER["args"] = args
    _WORKER["state"] = state


def _batches_dir(out_dir: str, flat: bool, modality: str) -> str:
    return os.path.join(out_dir, "_batches") if flat else os.path.join(out_dir, modality, "_batches")


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(key))[:120]


def _process_item(task):
    (item, split, modalities, out_dir, flat, specs, batch_size, run_nonce) = task
    adapter: DatasetAdapter = _WORKER["adapter"]
    args = _WORKER["args"]
    state = _WORKER["state"]
    # Prefix carries the item key (globally unique) so shards never collide
    # across workers OR items; run_nonce keeps a re-run from reusing a killed
    # run's half-written shard names (those are also swept as orphans).
    prefix = f"{run_nonce}_{_safe_key(item.key)}"
    try:
        writers = {
            m: ShardWriter(os.path.join(_batches_dir(out_dir, flat, m), split), prefix,
                            batch_size, specs[m])
            for m in modalities
        }
        streams = adapter.load_item_multi(item, modalities, args, state)
        if streams is None:
            streams = {m: adapter.load_item(item, m, args, state) for m in modalities}

        counts: Dict[str, int] = {}
        for m in modalities:
            c = 0
            for arr, label in streams[m]:
                writers[m].add(np.asarray(arr, dtype=np.float32), label)
                c += 1
            counts[m] = c

        mods_entries = {m: writers[m].finalize() for m in modalities}
        return ItemResult(item.key, split, mods_entries, counts, {})
    except Float16OverflowError as e:
        return FatalItemError(item.key, str(e))
    except Exception as e:  # noqa: BLE001 - one bad item must not kill a resumable run
        return SkipRecord(item.key, repr(e), "load_item")


def run_extraction(*, adapter: DatasetAdapter, args, state: Optional[dict],
                    tasks: List[tuple], num_workers: int,
                    on_result: Callable[[object], None]) -> None:
    """Dispatch `tasks` (built by the core), calling `on_result` once per
    finished item with an `ItemResult` / `SkipRecord` / `FatalItemError`.
    `num_workers <= 1` runs inline (no pool)."""
    if num_workers and num_workers > 1:
        ctx = get_context("fork") if "fork" in _mp_methods() else None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx,
                                  initializer=_init_worker,
                                  initargs=(adapter, args, state)) as ex:
            futs = [ex.submit(_process_item, t) for t in tasks]
            for fut in as_completed(futs):
                on_result(fut.result())
    else:
        _init_worker(adapter, args, state)
        for t in tasks:
            on_result(_process_item(t))


def _mp_methods():
    try:
        import multiprocessing as _mp
        return set(_mp.get_all_start_methods())
    except Exception:
        return set()


def make_run_nonce() -> str:
    return uuid.uuid4().hex[:8]


# ----------------------------------------------------------------------
# --writer direct
# ----------------------------------------------------------------------

def _direct_process_item(task):
    _tag, item, plan, args, run_nonce = task
    adapter: DatasetAdapter = _WORKER["adapter"]
    state = _WORKER["state"]
    mods = list(plan)
    try:
        streams = adapter.load_item_multi(item, mods, args, state)
        if streams is None:
            streams = {m: adapter.load_item(item, m, args, state) for m in mods}
        for m in mods:
            tmp_path, offset, count, _split = plan[m]
            mm = np.lib.format.open_memmap(tmp_path, mode="r+")
            i = 0
            for arr, _label in streams[m]:
                if i >= count:
                    raise RuntimeError(f"{item.key}/{m}: yielded more than expected_sample_count={count}")
                mm[offset + i] = np.asarray(arr, dtype=mm.dtype)
                i += 1
            del mm
            if i != count:
                raise RuntimeError(f"{item.key}/{m}: yielded {i} samples, expected {count}")
        return ItemResult(item.key, plan[mods[0]][3], {}, {m: plan[m][2] for m in mods}, {})
    except Exception as e:  # any failure in direct mode is fatal (offsets would desync)
        return FatalItemError(item.key, f"{type(e).__name__}: {e}")


def run_direct_extraction(*, adapter, args, state, tasks, num_workers, specs) -> int:
    n_ok = 0

    def _handle(res):
        nonlocal n_ok
        if isinstance(res, FatalItemError):
            raise SystemExit(
                f"--writer direct aborted at item {res.key}: {res.message}. "
                f"Re-run with the default --writer shards (crash-safe + fail-soft).")
        n_ok += 1

    if num_workers and num_workers > 1:
        ctx = get_context("fork") if "fork" in _mp_methods() else None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx,
                                  initializer=_init_worker, initargs=(adapter, args, state)) as ex:
            for fut in as_completed([ex.submit(_direct_process_item, t) for t in tasks]):
                _handle(fut.result())
    else:
        _init_worker(adapter, args, state)
        for t in tasks:
            _handle(_direct_process_item(t))
    return n_ok
