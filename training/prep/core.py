# -*- coding: utf-8 -*-
"""
training/prep/core.py
=====================
`run_prep(adapter, argv)` - the dataset-agnostic prep pipeline:

  discover -> split -> class-coverage gate -> (optional Pass 1) ->
  parallel per-item extraction -> spanning shards -> crash-safe unify ->
  cheap finalize (folded sha256 + probe verify) -> stats.

Every concrete `prepare_*.py` script is just a `DatasetAdapter` + a
2-line `main()` that calls this.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from training.dataset_prep_common import (
    SkipRegistry, check_prep_class_coverage, resolve_split_fractions,
)
from training.dataset_prep_unify import dataset_is_finalized_and_intact, finalize_dataset, unify_split
from training.npy_atomic import save_json_atomic
from training.npy_integrity import DatasetStorageError
from training.prep import splits as _splits
from training.prep.adapter import DatasetAdapter
from training.prep.dryrun import dry_run_report
from training.prep.manifest import (
    check_manifest_compatible, cleanup_orphan_shards, collect_shard_entries, load_manifest,
    new_manifest, save_manifest,
)
from training.prep.parallel import (
    FatalItemError, ItemResult, SkipRecord, make_run_nonce, run_extraction,
)

_SPLITS3 = ("train", "validation", "test")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser(adapter: DatasetAdapter) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Generic dataset prep ({adapter.name})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", help="raw dataset root (adapter-specific; not needed with --finalize_only)")
    p.add_argument("--out_dir", default="./data")

    g = p.add_argument_group("split")
    g.add_argument("--split", default="80_20",
                    help="preset (80_20, 80_10_10, ...) OR an A/B / A/B/C fraction spec (75/15/10)")
    g.add_argument("--split_strategy", choices=["stratified", "legacy_random"], default="stratified")
    g.add_argument("--split_file", default=None,
                    help="reuse a frozen split_assignment.json (group_id -> split)")
    g.add_argument("--kfold", type=int, default=None, help="grouped stratified k-fold: number of folds")
    g.add_argument("--fold", type=int, default=0, help="which fold is the test set (with --kfold)")
    g.add_argument("--cv_val_frac", type=float, default=0.0, help="carve a val split of this fraction (with --kfold)")
    g.add_argument("--cv_repeat", type=int, default=0, help="repeat index (perturbs the seed) (with --kfold)")
    g.add_argument("--allow_missing_classes", action="store_true")
    g.add_argument("--seed", type=int, default=42)

    e = p.add_argument_group("extraction")
    e.add_argument("--num_workers", type=int, default=min(os.cpu_count() or 1, 8),
                    help="parallel worker processes (0/1 = inline)")
    e.add_argument("--batch_size", type=int, default=256, help="samples per shard file")
    e.add_argument("--store_dtype", choices=["auto", "float32", "float16", "uint8"], default="auto",
                    help="override the adapter's per-modality on-disk dtype")
    e.add_argument("--balance_classes", choices=["none", "undersample", "oversample"], default="none",
                    help="applied to the TRAIN split only, at unify time")
    e.add_argument("--writer", choices=["shards", "direct"], default="shards",
                    help="'direct' (experimental) writes final arrays without shards; "
                         "requires fixed sample counts and aborts on any skipped item")
    e.add_argument("--deterministic_order", dest="deterministic_order", action="store_true", default=True)
    e.add_argument("--no_deterministic_order", dest="deterministic_order", action="store_false")
    e.add_argument("--limit", type=int, default=None, help="process at most N items (debugging)")
    e.add_argument("--manifest_flush_every", type=int, default=64)
    e.add_argument("--manifest_flush_seconds", type=float, default=30.0)

    v = p.add_argument_group("finalize / verify")
    v.add_argument("--verify_level", choices=["auto", "probe", "structural", "deep"], default="auto",
                    help="'auto' = always deep (full read-back of every produced array). "
                         "A digest computed while writing can only certify what was INTENDED; "
                         "only a read-back after the write commits can catch bytes that landed "
                         "wrong on disk. Pass 'probe'/'structural' to opt OUT of that check.")
    v.add_argument("--deep_verify_max_gb", type=float, default=4.0,
                    help="DEPRECATED, unused: 'auto' used to mean deep only below this size and "
                         "probe above it, which is how a corrupt multi-GB array reached training "
                         "undetected. 'auto' is now always deep regardless of size. Kept only so "
                         "old manifests' recorded cli_args stay parseable.")
    v.add_argument("--no_verify_output", action="store_true")
    v.add_argument("--keep_batches", action="store_true")
    v.add_argument("--fresh", action="store_true")
    v.add_argument("--finalize_only", action="store_true")
    v.add_argument("--dry_run", action="store_true")

    adapter.add_cli_args(p)
    return p


_LOCK_KEYS = ("split", "split_strategy", "split_file", "kfold", "fold", "cv_val_frac", "cv_repeat",
              "seed", "store_dtype", "balance_classes", "batch_size", "writer")


def _locked_args(args, adapter: DatasetAdapter) -> Dict:
    d = {k: getattr(args, k, None) for k in _LOCK_KEYS}
    d.update(adapter.compat_keys(args))
    d["modalities"] = list(adapter.modalities())
    return d


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _log(out_dir: str, msg: str) -> None:
    print(msg, flush=True)
    try:
        with open(os.path.join(out_dir, "prep_log.txt"), "a") as f:
            f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    except OSError:
        pass


def _split_out_name(split: str) -> str:
    return "val" if split == "validation" else split


def _modality_dir(out_dir: str, adapter: DatasetAdapter, modality: str) -> str:
    return out_dir if adapter.flat_output else os.path.join(out_dir, modality)


def _resolve_specs(adapter: DatasetAdapter, args, state) -> Dict:
    specs = {}
    for m in adapter.modalities():
        spec = adapter.modality_spec(m, args, state)
        if args.store_dtype != "auto":
            spec = type(spec)(store_dtype=args.store_dtype, value_clip=spec.value_clip,
                               sample_shape=spec.sample_shape, extra_outputs=spec.extra_outputs)
        specs[m] = spec
    return specs


# ----------------------------------------------------------------------
# unify
# ----------------------------------------------------------------------

def _modality_roles(out_dir: str, adapter: DatasetAdapter, modality: str, has_val: bool) -> Dict[str, str]:
    sub = _modality_dir(out_dir, adapter, modality)
    roles = {}
    for split in ("train", "validation", "test"):
        if split == "validation" and not has_val:
            continue
        on = _split_out_name(split)
        roles[f"X_{on}"] = os.path.join(sub, f"X_{on}.npy")
        roles[f"y_{on}"] = os.path.join(sub, f"y_{on}.npy")
    return roles


def _all_finalized_and_intact(out_dir: str, adapter: DatasetAdapter, manifest: dict) -> bool:
    if not manifest.get("finalized"):
        return False
    has_val = bool(manifest.get("validation_patients"))
    for m in adapter.modalities():
        roles = _modality_roles(out_dir, adapter, m, has_val)
        if not dataset_is_finalized_and_intact(_modality_dir(out_dir, adapter, m), roles):
            return False
    return True


def _resolve_verify_level(args) -> str:
    if args.no_verify_output:
        return "structural"
    if args.verify_level != "auto":
        return args.verify_level
    return "deep"


def _unify_all(out_dir: str, adapter: DatasetAdapter, manifest: dict, args, state) -> Dict:
    has_val = bool(manifest.get("validation_patients"))
    active_splits = [s for s in _SPLITS3 if s != "validation" or has_val]
    class_names = manifest["class_names"]
    counts: Dict[str, Dict[str, int]] = {}

    for m in adapter.modalities():
        sub = _modality_dir(out_dir, adapter, m)
        os.makedirs(sub, exist_ok=True)
        roles = {}
        sha_all: Dict[str, str] = {}
        m_counts = {}
        for split in active_splits:
            entries = collect_shard_entries(manifest, m, split, deterministic=args.deterministic_order)
            out_name = _split_out_name(split)
            xp = os.path.join(sub, f"X_{out_name}.npy")
            yp = os.path.join(sub, f"y_{out_name}.npy")
            bal = args.balance_classes if (split == "train" and args.balance_classes != "none") else None
            n, _rep = unify_split(entries, xp, yp, balance_method=bal, seed=args.seed,
                                   class_names=class_names, sha_out=sha_all)
            m_counts[split] = n
            roles[f"X_{out_name}"] = xp
            roles[f"y_{out_name}"] = yp
        counts[m] = m_counts

        extra_files = list(adapter.finalize_modality(m, sub, state, manifest))
        vlevel = _resolve_verify_level(args)
        manifest_extra = {
            "prep_core": "training.prep.core", "adapter": adapter.name,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "git_sha": _git_sha(),
            "cli_args": {k: v for k, v in vars(args).items()},
            "split": manifest["split"], "class_names": class_names,
            "store_dtype": {mm: _resolve_specs(adapter, args, state)[mm].store_dtype
                            for mm in adapter.modalities()}.get(m),
            "provenance": adapter.provenance(args, state),
        }
        try:
            finalize_dataset(sub, roles, extra_files=extra_files, manifest_extra=manifest_extra,
                             verify=not args.no_verify_output, verify_level=vlevel,
                             precomputed_sha256=sha_all)
        except DatasetStorageError as ex:
            _log(out_dir, f"\nFAILURE_CLASS=DATASET_STORAGE_ERROR ({ex})")
            _log(out_dir, "  Unified arrays did NOT verify; shards kept. Fix the cause and "
                          "re-run with --finalize_only.")
            raise SystemExit(2)

        # class_names.json next to the arrays (documented consumer gap)
        save_json_atomic(os.path.join(sub, "class_names.json"), class_names)
        _log(out_dir, f"  [{m}] unified " + " ".join(f"{s}={m_counts[s]}" for s in active_splits)
             + f"  verify={vlevel} -> {sub}/")

    if not args.keep_batches:
        _delete_batches(out_dir, adapter)
    return counts


def _delete_batches(out_dir: str, adapter: DatasetAdapter) -> None:
    dirs = [os.path.join(out_dir, "_batches")] if adapter.flat_output else \
           [os.path.join(out_dir, m, "_batches") for m in adapter.modalities()]
    for base in dirs:
        if not os.path.isdir(base):
            continue
        for root, _d, files in os.walk(base, topdown=False):
            for fn in files:
                try:
                    os.remove(os.path.join(root, fn))
                except OSError:
                    pass
            try:
                os.rmdir(root)
            except OSError:
                pass


# ----------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------

def run_prep(adapter: DatasetAdapter, argv: Optional[List[str]] = None) -> int:
    args = build_parser(adapter).parse_args(argv)
    adapter.preprocess_args(args)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    _log(out_dir, f"prep start: adapter={adapter.name} argv={' '.join(sys.argv[1:])}")

    # ---- finalize-only fast path ----
    if args.finalize_only:
        manifest = load_manifest(out_dir)
        if manifest is None:
            raise SystemExit(f"--finalize_only: no _progress.json in {out_dir}")
        state = manifest.get("pass1_state")
        counts = _unify_all(out_dir, adapter, manifest, args, state)
        manifest["finalized"] = True
        save_manifest(out_dir, manifest)
        _write_statistics(out_dir, adapter, manifest, counts, SkipRegistry(manifest.get("skipped_items")))
        _log(out_dir, "Done (finalize_only).")
        return 0

    if not args.root:
        raise SystemExit("--root is required unless --finalize_only is passed")

    # ---- discover ----
    _log(out_dir, f"scanning {args.root} ...")
    items = list(adapter.discover(args))
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit("adapter.discover() returned no labelled items")
    class_names = list(adapter.class_names)
    group_to_meta = {}
    for it in items:
        cls, strat = group_to_meta.get(it.group_id, (set(), tuple(it.strat_keys)))
        cls = set(cls); cls.add(it.label)
        group_to_meta[it.group_id] = (cls, tuple(it.strat_keys))
    _log(out_dir, f"found {len(items)} items across {len(group_to_meta)} groups, "
                  f"{len(class_names)} classes")

    # ---- split ----
    manifest = None if args.fresh else load_manifest(out_dir)
    if manifest is not None:
        check_manifest_compatible(manifest, _locked_args(args, adapter))
        train = set(manifest["train_patients"])
        val = set(manifest["validation_patients"])
        test = set(manifest["test_patients"])
        state = manifest.get("pass1_state")
        if _all_finalized_and_intact(out_dir, adapter, manifest):
            _log(out_dir, "already finalized and every X/y still verifies - nothing to do.")
            return 0
        _log(out_dir, f"resuming: {len(manifest['completed_items'])} item(s) already done")
    else:
        train, val, test, warns, split_meta = _splits.compute_split(group_to_meta, args)
        for w in warns:
            _log(out_dir, f"  [split-warning] {w}")
        _splits.write_assignment(out_dir, train=train, val=val, test=test, split_meta=split_meta,
                                  seed=args.seed, class_names=class_names)

        def _assign(g):
            return "test" if g in test else "validation" if g in val else "train"

        pre_cov = check_prep_class_coverage(
            [(it.label, it.group_id) for it in items], _assign, num_classes=len(class_names),
            class_names=class_names, fail_hard=not args.allow_missing_classes)
        save_json_atomic(os.path.join(out_dir, "class_coverage_report.pre.json"), pre_cov)
        _log(out_dir, "  [class-coverage] " + ("OK" if pre_cov["coverage_ok"]
                                               else f"missing {pre_cov['missing']}"))

        if args.dry_run:
            print(dry_run_report(adapter=adapter, args=args, items=items, train=train, val=val,
                                  test=test, split_meta=split_meta, warnings=warns))
            return 0

        state = adapter.pass1(items, args)
        adapter.write_pass1_outputs(state, out_dir)
        manifest = new_manifest(
            adapter_name=adapter.name, args_locked=_locked_args(args, adapter), split_meta=split_meta,
            class_names=class_names, train=train, val=val, test=test,
            modalities=adapter.modalities(), pass1_state=state)
        save_manifest(out_dir, manifest)
        _log(out_dir, f"fresh start: train={len(train)} val={len(val)} test={len(test)} groups "
                      f"(split={split_meta.get('spec')}, {split_meta.get('strategy')})")

    if args.dry_run:
        _log(out_dir, "(dry_run on a resumed run: nothing to show)")
        return 0

    # ---- extraction ----
    for m in adapter.modalities():
        cleanup_orphan_shards(out_dir, "" if adapter.flat_output else m)

    def assign_split(g):
        return "test" if g in test else "validation" if g in val else "train"

    specs = _resolve_specs(adapter, args, state)
    if args.writer == "direct":
        return _run_direct(adapter, args, state, items, manifest, out_dir, specs, assign_split)

    skips = SkipRegistry(manifest.get("skipped_items"))
    done = set(manifest["completed_items"])
    todo = [it for it in items if it.key not in done and it.key not in skips]
    _log(out_dir, f"extracting {len(todo)} item(s) with {args.num_workers} worker(s) ...")

    run_nonce = make_run_nonce()
    tasks = [(it, assign_split(it.group_id), list(adapter.modalities()), out_dir,
              adapter.flat_output, specs, args.batch_size, run_nonce) for it in todo]

    state_box = {"n": 0, "t": time.time(), "ok": 0}

    def on_result(res):
        if isinstance(res, FatalItemError):
            save_manifest(out_dir, manifest)
            raise SystemExit(f"FATAL for item {res.key}: {res.message}")
        if isinstance(res, SkipRecord):
            skips.add(res.key, res.reason, res.stage)
            manifest["skipped_items"] = skips.as_dict()
        else:  # ItemResult
            manifest["completed_items"][res.key] = {
                "split": res.split, "modalities": res.modalities,
                "counts": res.counts, "extra": res.extra}
            state_box["ok"] += 1
        state_box["n"] += 1
        now = time.time()
        if (state_box["n"] % max(1, args.manifest_flush_every) == 0
                or now - state_box["t"] >= args.manifest_flush_seconds):
            save_manifest(out_dir, manifest)
            state_box["t"] = now
            rate = state_box["n"] / max(1e-6, now - state_box.get("t0", now))
            _log(out_dir, f"  ... {state_box['n']}/{len(todo)} done "
                          f"({state_box['ok']} ok, {len(skips)} skipped)")

    state_box["t0"] = time.time()
    try:
        run_extraction(adapter=adapter, args=args, state=state, tasks=tasks,
                        num_workers=args.num_workers, on_result=on_result)
    finally:
        save_manifest(out_dir, manifest)

    if len(skips):
        _log(out_dir, f"  [ingest] {skips.summary()}")

    # realized coverage (skips can empty a slot)
    done = set(manifest["completed_items"])
    realized = check_prep_class_coverage(
        [(it.label, it.group_id) for it in items if it.key in done], assign_split,
        num_classes=len(class_names), class_names=class_names,
        fail_hard=not args.allow_missing_classes)
    save_json_atomic(os.path.join(out_dir, "class_coverage_report.json"), realized)
    if not realized["coverage_ok"]:
        _log(out_dir, f"  [WARNING] realized class coverage incomplete: {realized['missing']}")

    _log(out_dir, "unifying shards ...")
    counts = _unify_all(out_dir, adapter, manifest, args, state)
    manifest["finalized"] = True
    save_manifest(out_dir, manifest)
    _write_statistics(out_dir, adapter, manifest, counts, skips)
    _log(out_dir, "Done.")
    return 0


# ----------------------------------------------------------------------
# --writer direct (experimental)
# ----------------------------------------------------------------------

def _run_direct(adapter, args, state, items, manifest, out_dir, specs, assign_split) -> int:
    """Experimental: workers write patches straight into the final X_*.npy
    (into a `.tmp`, atomically committed), skipping the shard->array copy.
    Requires a known `expected_sample_count` for every item + a fixed
    `sample_shape` per modality, and aborts on ANY skipped item (offsets
    would be invalid).

    WARNING: unlike the default `--writer shards` path (see
    `dataset_prep_unify.atomic_npy_stream`), this fills the final arrays via
    `open_memmap` from parallel workers writing at scattered offsets, which
    cannot be converted to a sequential `write()` - the workers need random
    access. It therefore retains the mmap-writeback-race hazard that produced
    `OSError(5)/EIO` on a multi-GB array on this project's hardware (a
    mapping much larger than available RAM forces the kernel to flush dirty
    pages WHILE they are still being written). Prefer `--writer shards`
    (the default) for anything larger than a small fraction of RAM."""
    print("WARNING: --writer direct fills final arrays via mmap from parallel workers - "
          "it retains the writeback-race hazard that --writer shards (the default) no "
          "longer has. See training/prep/core.py:_run_direct for details.", flush=True)
    import numpy as _np
    from training.prep.parallel import make_run_nonce

    mods = list(adapter.modalities())
    for m in mods:
        if specs[m].sample_shape is None:
            raise SystemExit(f"--writer direct: modality {m!r} has no fixed sample_shape")

    # deterministic plan: item order = sorted key
    ordered = sorted(items, key=lambda it: it.key)
    plan: Dict = {}   # (mod, split) -> list[(item_key, label, count, offset)]
    for it in ordered:
        sp = assign_split(it.group_id)
        for m in mods:
            c = adapter.expected_sample_count(it, m, args, state)
            if c is None:
                raise SystemExit(
                    f"--writer direct: expected_sample_count is unknown for {it.key!r}/{m}; "
                    f"re-run with the default --writer shards")
            plan.setdefault((m, sp), []).append([it.key, it.label, int(c), 0])
    totals: Dict = {}
    for k, rows in plan.items():
        off = 0
        for r in rows:
            r[3] = off
            off += r[2]
        totals[k] = off

    has_val = any(sp == "validation" for (_m, sp) in plan)
    active_splits = [s for s in _SPLITS3 if s != "validation" or has_val]

    # pre-size every .tmp and write the y arrays now (labels are known)
    tmp_paths: Dict = {}
    for m in mods:
        sub = _modality_dir(out_dir, adapter, m)
        os.makedirs(sub, exist_ok=True)
        for sp in active_splits:
            rows = plan.get((m, sp), [])
            n = totals.get((m, sp), 0)
            on = _split_out_name(sp)
            xt = os.path.join(sub, f"X_{on}.npy") + ".tmp"
            if os.path.exists(xt):
                os.remove(xt)
            _np.lib.format.open_memmap(xt, mode="w+", dtype=_np.dtype(specs[m].store_dtype),
                                        shape=(n,) + tuple(specs[m].sample_shape))
            tmp_paths[(m, sp)] = xt
            y = _np.empty((n,), dtype=_np.int64)
            for (_k, lab, cnt, o) in rows:
                y[o:o + cnt] = lab
            from training.npy_atomic import save_npy_atomic as _sv
            _sv(os.path.join(sub, f"y_{on}.npy"), y)

    # dispatch: one task per item carries its (mod,split)->(tmp_path, offset)
    run_nonce = make_run_nonce()
    done = set(manifest.get("completed_items", {}))
    tasks = []
    key_to_plan: Dict = {}
    for (m, sp), rows in plan.items():
        for (k, lab, cnt, o) in rows:
            key_to_plan.setdefault(k, {})[m] = (tmp_paths[(m, sp)], o, cnt, sp)
    todo = [it for it in ordered if it.key not in done]
    for it in todo:
        tasks.append(("__direct__", it, {m: key_to_plan[it.key][m] for m in mods},
                       args, run_nonce))

    from training.prep.parallel import run_direct_extraction
    run_direct_extraction(adapter=adapter, args=args, state=state, tasks=tasks,
                           num_workers=args.num_workers, specs=specs)
    for it in todo:
        manifest["completed_items"][it.key] = {"split": assign_split(it.group_id),
                                                "modalities": {}, "counts": {}, "extra": {"direct": True}}
    save_manifest(out_dir, manifest)

    # commit each .tmp
    counts: Dict = {}
    for m in mods:
        sub = _modality_dir(out_dir, adapter, m)
        roles = {}
        mc = {}
        for sp in active_splits:
            on = _split_out_name(sp)
            xt = tmp_paths[(m, sp)]
            final = os.path.join(sub, f"X_{on}.npy")
            fd = os.open(xt, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(xt, final)
            roles[f"X_{on}"] = final
            roles[f"y_{on}"] = os.path.join(sub, f"y_{on}.npy")
            mc[sp] = totals.get((m, sp), 0)
        counts[m] = mc
        extra_files = list(adapter.finalize_modality(m, sub, state, manifest))
        vlevel = _resolve_verify_level(args)
        try:
            finalize_dataset(sub, roles, extra_files=extra_files, verify=not args.no_verify_output,
                             verify_level=vlevel,
                             manifest_extra={"prep_core": "training.prep.core", "adapter": adapter.name,
                                              "writer": "direct", "split": manifest["split"],
                                              "class_names": manifest["class_names"]})
        except DatasetStorageError as ex:
            _log(out_dir, f"\nFAILURE_CLASS=DATASET_STORAGE_ERROR ({ex})")
            raise SystemExit(2)
        save_json_atomic(os.path.join(sub, "class_names.json"), manifest["class_names"])
        _log(out_dir, f"  [{m}] direct-wrote " + " ".join(f"{s}={mc[s]}" for s in active_splits))

    manifest["finalized"] = True
    save_manifest(out_dir, manifest)
    _write_statistics(out_dir, adapter, manifest, counts, SkipRegistry(manifest.get("skipped_items")))
    _log(out_dir, "Done (--writer direct).")
    return 0


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------

def _write_statistics(out_dir, adapter, manifest, counts, skips: SkipRegistry) -> None:
    split_meta = manifest.get("split", {})
    spec = split_meta.get("spec")
    try:
        tr, va, te = resolve_split_fractions(spec) if isinstance(spec, str) else (None, None, None)
    except Exception:
        tr = va = te = None
    stats = {
        "adapter": adapter.name,
        "split": {
            "spec": spec, "strategy": split_meta.get("strategy"),
            "split_algo_version": split_meta.get("split_algo_version"),
            "kfold_algo_version": split_meta.get("kfold_algo_version"),
            "target_fractions": {"train": tr, "validation": va, "test": te},
            "group_counts": {"train": len(manifest["train_patients"]),
                              "validation": len(manifest["validation_patients"]),
                              "test": len(manifest["test_patients"])},
            "realized_sample_counts": counts,
        },
        "class_names": manifest["class_names"],
        "modalities": manifest.get("modalities"),
        "n_completed_items": len(manifest.get("completed_items", {})),
        "skipped_items": skips.as_dict(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    save_json_atomic(os.path.join(out_dir, "dataset_statistics.json"), stats)
