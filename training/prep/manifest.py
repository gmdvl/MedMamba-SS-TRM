# -*- coding: utf-8 -*-
"""
training/prep/manifest.py
=========================
`_progress.json` v2 lifecycle for the generic prep core: resumable state,
compatibility locking, and orphan-shard cleanup.

The v2 document is a SUPERSET of the v5/v6 layout - `train_patients` /
`validation_patients` / `test_patients` stay at the top level so the
external readers (`train_example_v*` leakage check, `training/data.py`)
keep working; everything else is namespaced.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from training.npy_atomic import save_json_atomic

PROGRESS_FILE = "_progress.json"


def manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, PROGRESS_FILE)


def load_manifest(out_dir: str) -> Optional[dict]:
    p = manifest_path(out_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_manifest(out_dir: str, manifest: dict) -> None:
    save_json_atomic(manifest_path(out_dir), manifest)


def new_manifest(*, adapter_name: str, args_locked: Dict[str, Any], split_meta: Dict[str, Any],
                  class_names: Iterable[str], train, val, test, modalities: Iterable[str],
                  pass1_state: Optional[dict]) -> dict:
    return {
        "schema": 2,
        "adapter": adapter_name,
        "args": dict(args_locked),
        "split": dict(split_meta),
        "class_names": list(class_names),
        # top-level, verbatim, for external consumers:
        "train_patients": sorted(train),
        "validation_patients": sorted(val),
        "test_patients": sorted(test),
        "pass1_state": pass1_state,
        "modalities": list(modalities),
        "completed_items": {},   # key -> {"split", "modalities": {mod: [[x,y,n],...]}, "counts": {...}, "extra": {}}
        "skipped_items": {},     # key -> {"reason", "stage"}
        "finalized": False,
    }


# keys that are advisory only (never abort a resume over them)
_ADVISORY_KEYS = {"batch_size", "num_workers", "manifest_flush_every", "manifest_flush_seconds",
                  "verify_level", "deep_verify_max_gb", "keep_batches", "writer"}


def check_manifest_compatible(manifest: dict, args_locked: Dict[str, Any]) -> None:
    saved = manifest.get("args", {})
    mism = {k: (saved.get(k), v) for k, v in args_locked.items()
            if k not in _ADVISORY_KEYS and saved.get(k) != v}
    if mism:
        raise ValueError(
            f"Found an existing {PROGRESS_FILE} with DIFFERENT settings: {mism}. "
            f"Use a fresh --out_dir, delete {PROGRESS_FILE} to start over, pass --fresh, "
            f"or match the original settings to resume.")
    for k in _ADVISORY_KEYS:
        if k in args_locked and saved.get(k) not in (None, args_locked[k]):
            print(f"  [warn] {k} changed on resume ({saved.get(k)} -> {args_locked[k]}); "
                  f"not correctness-relevant, continuing.")


def cleanup_orphan_shards(out_dir: str, subdir: str = "") -> int:
    """Delete every shard file under `<out_dir>/<subdir>/_batches/**` that is
    not referenced by a completed item (leftovers from a worker killed
    mid-flush), plus any `.tmp`. Returns the number of files removed."""
    manifest = load_manifest(out_dir)
    referenced = set()
    if manifest is not None:
        for it in manifest.get("completed_items", {}).values():
            for entries in (it.get("modalities") or {}).values():
                for e in entries:
                    referenced.add(os.path.abspath(e[0]))
                    referenced.add(os.path.abspath(e[1]))
    base = os.path.join(out_dir, subdir, "_batches") if subdir else os.path.join(out_dir, "_batches")
    if not os.path.isdir(base):
        return 0
    removed = 0
    for split in os.listdir(base):
        sdir = os.path.join(base, split)
        if not os.path.isdir(sdir):
            continue
        for fname in os.listdir(sdir):
            fpath = os.path.join(sdir, fname)
            if fname.endswith(".tmp") or os.path.abspath(fpath) not in referenced:
                try:
                    os.remove(fpath)
                    removed += 1
                except OSError:
                    pass
    return removed


def collect_shard_entries(manifest: dict, modality: str, split: str, *,
                           deterministic: bool = True) -> List[list]:
    """Gather `(x, y, n)` shard entries for one `(modality, split)` from every
    completed item. With `deterministic=True` they are ordered by item key so
    the unified array's row order does not depend on worker completion
    order."""
    out: List[tuple] = []
    for key in sorted(manifest.get("completed_items", {})) if deterministic \
            else manifest.get("completed_items", {}):
        it = manifest["completed_items"][key]
        if it.get("split") != split:
            continue
        for e in (it.get("modalities") or {}).get(modality, []):
            out.append(tuple(e))
    return [list(e) for e in out]
