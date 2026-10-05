# -*- coding: utf-8 -*-
"""
training/prep/splits.py
=======================
Resolve the shared `--split*` CLI into concrete group-id sets, delegating
the algorithms to `training.dataset_prep_common`. Also owns
`split_assignment.json` (write on a fresh run; read for `--split_file`).
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Iterable, List, Set, Tuple

from training.dataset_prep_common import (
    KFOLD_ALGO_VERSION, SPLIT_ALGO_VERSION, kfold_split, load_split_assignment,
    parse_split_spec, patient_split_three, resolve_split_fractions,
    stratified_group_split_three, stratified_patient_split_three, write_split_assignment,
)

SPLIT_ASSIGNMENT_FILE = "split_assignment.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compute_split(group_to_meta: Dict[str, Tuple[Iterable[int], tuple]], args
                   ) -> Tuple[Set[str], Set[str], Set[str], List[str], dict]:
    """Returns `(train, val, test, warnings, split_meta)` group-id sets.
    `group_to_meta`: {group_id: (set_of_class_ids, strat_key_tuple)}."""
    g2c = {g: set(m[0]) for g, m in group_to_meta.items()}

    if getattr(args, "split_file", None):
        train, val, test, obj = load_split_assignment(args.split_file)
        known = set(group_to_meta)
        train &= known; val &= known; test &= known
        missing = known - train - val - test
        if missing:
            raise ValueError(
                f"--split_file {args.split_file} does not assign {len(missing)} group(s) present "
                f"in this dataset, e.g. {sorted(missing)[:5]}")
        meta = {
            "spec": obj.get("spec", "file"), "strategy": "file",
            "split_algo_version": obj.get("split_algo_version"),
            "kfold_algo_version": obj.get("kfold_algo_version"),
            "split_file": os.path.abspath(args.split_file),
            "split_file_sha256": _sha256(args.split_file),
            "fractions": _fractions(train, val, test),
        }
        return train, val, test, [], meta

    if getattr(args, "kfold", None):
        tr, va, te, warns = kfold_split(
            g2c, int(args.kfold), int(args.fold), int(args.seed),
            cv_val_frac=float(getattr(args, "cv_val_frac", 0.0) or 0.0),
            cv_repeat=int(getattr(args, "cv_repeat", 0) or 0))
        meta = {
            "spec": {"kfold": int(args.kfold), "fold": int(args.fold),
                      "cv_val_frac": float(getattr(args, "cv_val_frac", 0.0) or 0.0),
                      "cv_repeat": int(getattr(args, "cv_repeat", 0) or 0)},
            "strategy": "kfold", "split_algo_version": SPLIT_ALGO_VERSION,
            "kfold_algo_version": KFOLD_ALGO_VERSION, "fractions": _fractions(tr, va, te),
        }
        return tr, va, te, warns, meta

    spec = args.split
    if args.split_strategy == "legacy_random":
        tr, va, te, warns = patient_split_three(list(group_to_meta), spec, int(args.seed))
    else:
        tr, va, te, warns = stratified_group_split_three(
            {g: (m[0], tuple(m[1])) for g, m in group_to_meta.items()}, spec, int(args.seed))
    meta = {
        "spec": spec, "strategy": args.split_strategy,
        "split_algo_version": SPLIT_ALGO_VERSION, "kfold_algo_version": None,
        "target_fractions": list(resolve_split_fractions(spec)),
        "fractions": _fractions(tr, va, te),
    }
    return tr, va, te, warns, meta


def _fractions(tr, va, te) -> dict:
    n = max(1, len(tr) + len(va) + len(te))
    return {"train": len(tr) / n, "validation": len(va) / n, "test": len(te) / n}


def write_assignment(out_dir: str, *, train, val, test, split_meta: dict, seed: int,
                      class_names) -> str:
    groups = {}
    for g in train:
        groups[g] = "train"
    for g in val:
        groups[g] = "validation"
    for g in test:
        groups[g] = "test"
    spec = split_meta.get("spec")
    spec_str = spec if isinstance(spec, str) else str(spec)
    path = os.path.join(out_dir, SPLIT_ASSIGNMENT_FILE)
    write_split_assignment(
        path, groups=groups, strategy=split_meta.get("strategy", "stratified"), seed=seed,
        spec=spec_str, fractions=tuple(resolve_split_fractions(spec)) if isinstance(spec, str)
        else (split_meta["fractions"]["train"], split_meta["fractions"]["validation"],
              split_meta["fractions"]["test"]),
        class_names=class_names, split_algo_version=split_meta.get("split_algo_version") or SPLIT_ALGO_VERSION,
        kfold_algo_version=split_meta.get("kfold_algo_version"))
    return path
