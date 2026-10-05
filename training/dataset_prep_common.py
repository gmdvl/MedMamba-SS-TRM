# -*- coding: utf-8 -*-
"""
training/dataset_prep_common.py
==================================
Shared preprocessing utilities for ANY `prepare_*.py` dataset script in
this repo - built to fix a real bug and to let a second dataset
(PAD-UFES-20) reuse the same, now-correct, splitting logic instead of
duplicating it.

THE BUG THIS FIXES
--------------------
`prepare_histologyhsi_bc_v4.py`'s `patient_split_three()` (and the plain
`patient_split()` before it) does a single global shuffle of ALL patients,
then slices off the first N as test/val. That's patient-level (good -
Phase 2), but it is NOT stratified by class. When a class has a small
number of distinct patients, plain random slicing can - and did, per the
error report this module was written in response to - put every single
patient carrying that class into `train`, leaving `validation`/`test` with
ZERO samples of it. `training/class_coverage.py`'s Phase-3 check catches
this correctly, but catching it isn't the fix - the SPLIT needs to stop
producing it in the first place, and the check needs to run at PREP time
(before the expensive patch/image extraction), not just at train time
where the failure was previously only discovered.

`stratified_patient_split_three()` below fixes the split. `check_prep_class_
coverage()` runs the exact same `training/class_coverage.verify_class_
coverage()` check `train_example_v11.py` runs, but against the split's
CAPTURE/IMAGE-level labels (known immediately after the split, before any
extraction), and is meant to be called by every `prepare_*.py` script
right after computing its split.
"""

from __future__ import annotations

import datetime as _dt
import json
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from training.class_coverage import verify_class_coverage

# Same presets as prepare_histologyhsi_bc_v4.py's SPLIT_PRESETS - centralized
# here so a second dataset script doesn't redefine them differently.
SPLIT_PRESETS = {
    "80_20":    (0.80, 0.00, 0.20),
    "80_10_10": (0.80, 0.10, 0.10),
    "70_15_15": (0.70, 0.15, 0.15),
    "70_30":    (0.70, 0.00, 0.30),
    "60_20_20": (0.60, 0.20, 0.20),
}

# Bumped whenever `stratified_patient_split_three` changes which patients it
# puts in which split for a fixed seed. Written into each prep script's
# dataset_manifest.json so a dataset's split is traceable to an algorithm.
#   1 = prepare_histologyhsi_bc_v5.py (grouping-by-rarest-class only)
#   2 = + coverage fill pass (prepare_*_v6/v5, this module)
SPLIT_ALGO_VERSION = 2


# Bumped whenever `grouped_stratified_kfold` / `kfold_split` change which
# groups land in which fold for a fixed seed. Recorded next to
# SPLIT_ALGO_VERSION in split_assignment.json / _progress.json.
KFOLD_ALGO_VERSION = 1


def resolve_split_fractions(split_preset: str) -> Tuple[float, float, float]:
    """Preset name -> its `(train, val, test)` fractions. Also accepts an
    arbitrary spec string ('75/15/10', '80/20', '0.7/0.15/0.15') via
    `parse_split_spec`, so every caller that took a preset name now also takes
    a free-form split without any change."""
    if split_preset in SPLIT_PRESETS:
        return SPLIT_PRESETS[split_preset]
    return parse_split_spec(split_preset)


def parse_split_spec(spec: str) -> Tuple[float, float, float]:
    """Turn `spec` into normalized `(train, val, test)` fractions.

    Accepts: a preset key ('80_20'); a 2- or 3-part '/' (or '_') separated
    string of numbers, interpreted as PERCENTAGES if the parts sum to > ~1.5,
    otherwise as fractions ('75/15/10', '80/20', '0.7/0.15/0.15'). A 2-part
    spec is train/test with no validation. Raises `ValueError` on anything
    malformed or a sum that isn't ~1 (or ~100)."""
    if spec in SPLIT_PRESETS:
        return SPLIT_PRESETS[spec]
    parts = str(spec).replace("_", "/").split("/")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(
            f"Unrecognised --split {spec!r}: not a preset {sorted(SPLIT_PRESETS)} and not an "
            f"'A/B' or 'A/B/C' fraction/percentage spec")
    if len(vals) not in (2, 3):
        raise ValueError(f"--split {spec!r} must have 2 or 3 parts, got {len(vals)}")
    if any(v < 0 for v in vals):
        raise ValueError(f"--split {spec!r} has a negative part")
    if len(vals) == 2:
        vals = [vals[0], 0.0, vals[1]]
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"--split {spec!r} sums to {total}")
    scale = 100.0 if total > 1.5 else 1.0
    if abs(total - scale) > 0.02 * scale:
        raise ValueError(f"--split {spec!r} parts sum to {total:g}, expected ~{scale:g}")
    return tuple(v / total for v in vals)  # type: ignore[return-value]


def stratified_patient_split_three(
    patient_to_classes: Dict[str, Iterable[int]], split_preset: str, seed: int,
) -> Tuple[Set[str], Set[str], Set[str], List[str]]:
    """Patient-level (Phase 2) AND class-stratified (Phase 3) split.

    `patient_to_classes`: {patient_id: iterable of distinct class ids that
    patient contributes at least one sample of}. A patient with samples of
    more than one class (e.g. multiple lesions/captures) is fine.

    Algorithm: each patient is assigned to a stratification group keyed by
    their RAREST class (the class with the fewest total patients) - this
    means a patient who has both a common and a rare class still gets
    grouped by the rare one, so the rare class's coverage guarantee isn't
    undermined by that patient "belonging" to the common class instead.
    Groups are then processed rarest-class-first, and within each group
    `test`/`val`/`train` patient COUNTS are computed directly (not by
    slicing a globally-shuffled list), so a class is left out of a split
    only when its patient pool is too small for that to be avoidable - and
    that gets reported, not silently produced.

    Returns `(train_patients, val_patients, test_patients, warnings)`.
    `warnings` lists every class whose patient pool was too small to
    guarantee full coverage (Phase 3's own escape hatch: "if this is
    impossible because the source dataset genuinely lacks enough
    independent patients, ... explicitly report the limitation").
    """
    train_frac, val_frac, test_frac = resolve_split_fractions(split_preset)
    has_val = val_frac > 0
    rng = random.Random(seed)

    class_patient_counts: Dict[int, int] = defaultdict(int)
    for classes in patient_to_classes.values():
        for c in set(classes):
            class_patient_counts[c] += 1

    class_to_patients: Dict[int, List[str]] = defaultdict(list)
    for patient, classes in patient_to_classes.items():
        classes = list(set(classes))
        if not classes:
            continue
        rarest = min(classes, key=lambda c: class_patient_counts[c])
        class_to_patients[rarest].append(patient)

    train_patients: Set[str] = set()
    val_patients: Set[str] = set()
    test_patients: Set[str] = set()
    warnings: List[str] = []

    # Rarest (smallest patient pool) classes processed first.
    for c in sorted(class_to_patients, key=lambda c: len(class_to_patients[c])):
        patients = sorted(class_to_patients[c])
        rng.shuffle(patients)
        n = len(patients)

        if has_val:
            if n >= 3:
                n_test = max(1, round(n * test_frac))
                n_test = min(n_test, n - 2)
                remaining = n - n_test
                n_val = max(1, round(n * val_frac))
                n_val = min(n_val, remaining - 1)
            elif n == 2:
                n_test, n_val = 1, 0
                warnings.append(f"class {c}: only 2 patients - validation split will have NO samples "
                                 f"of this class (1 patient -> train, 1 -> test)")
            else:  # n == 1
                n_test, n_val = 0, 0
                warnings.append(f"class {c}: only 1 patient - validation AND test splits will have NO "
                                 f"samples of this class (assigned to train only)")
        else:
            if n >= 2:
                n_test = max(1, round(n * test_frac))
                n_test = min(n_test, n - 1)
            else:
                n_test = 0
                warnings.append(f"class {c}: only 1 patient - test split will have NO samples of this "
                                 f"class (assigned to train only)")
            n_val = 0

        test_patients.update(patients[:n_test])
        val_patients.update(patients[n_test:n_test + n_val])
        train_patients.update(patients[n_test + n_val:])

    # ------------------------------------------------------------------
    # Coverage fill pass (SPLIT_ALGO_VERSION 2). The grouping-by-rarest-class
    # step above guarantees the *rarest* class per patient is spread across
    # splits, but a COMMON class whose every carrier happens to be a
    # multi-class patient grouped under some other (rarer) class can still
    # end up with zero patients in validation/test - which is exactly the
    # "validation: missing ['0', '2']" symptom this version was written for.
    # Here we walk every (non-train split, class) slot that is still empty
    # and, if that class has enough distinct patients to make coverage
    # possible, move one carrier out of train into the deficient split -
    # preferring a carrier whose removal doesn't uncover any class in train.
    # ------------------------------------------------------------------
    _coverage_fill_pass(train_patients, val_patients, test_patients,
                        patient_to_classes, class_patient_counts, has_val, warnings)

    return train_patients, val_patients, test_patients, warnings


def _classes_covered_by(patients: Iterable[str], patient_to_classes: Dict[str, Iterable[int]]) -> Set[int]:
    covered: Set[int] = set()
    for p in patients:
        covered |= set(patient_to_classes.get(p, ()))
    return covered


def _coverage_fill_pass(train: Set[str], val: Set[str], test: Set[str],
                         patient_to_classes: Dict[str, Iterable[int]],
                         class_patient_counts: Dict[int, int], has_val: bool,
                         warnings: List[str]) -> None:
    """Mutates `train` / `val` / `test` in place so that every class with
    enough distinct patients is represented in every non-train split."""
    active = [("test", test)] + ([("validation", val)] if has_val else [])
    min_patients = 3 if has_val else 2
    for split_name, split_set in active:
        for c in sorted(class_patient_counts):
            if any(c in set(patient_to_classes.get(p, ())) for p in split_set):
                continue
            if class_patient_counts[c] < min_patients:
                warnings.append(
                    f"class {c}: only {class_patient_counts[c]} distinct patient(s) - the "
                    f"{split_name} split cannot be guaranteed a sample of it (kept in train)")
                continue
            donors = [p for p in sorted(train) if c in set(patient_to_classes.get(p, ()))]
            safe = []
            for p in donors:
                rest_cov = _classes_covered_by(train - {p}, patient_to_classes)
                if all(cc in rest_cov for cc in patient_to_classes.get(p, ())):
                    safe.append(p)
            pick = safe[0] if safe else (donors[0] if donors else None)
            if pick is None:
                warnings.append(f"class {c}: no train patient carries it - cannot fill {split_name}")
                continue
            train.discard(pick)
            split_set.add(pick)
            if not safe:
                warnings.append(
                    f"class {c}: moved patient {pick!r} into {split_name} to guarantee coverage; "
                    f"train now relies on other patients for classes "
                    f"{sorted(set(patient_to_classes.get(pick, ())))}")


def patient_split_three(patient_ids: Iterable[str], split_preset: str, seed: int
                         ) -> Tuple[Set[str], Set[str], Set[str], List[str]]:
    """`--split_strategy legacy_random`: the pre-v5 unstratified behaviour -
    one global shuffle of all patients, then slice off test / val / train by
    count. NO per-class coverage guarantee. Kept only so a v4/v5-generated
    dataset can be reproduced byte-for-byte with the same `--seed`. Returns
    the same 4-tuple shape as `stratified_patient_split_three` (warnings is
    always empty here)."""
    train_frac, val_frac, test_frac = resolve_split_fractions(split_preset)
    patients = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_test = max(1, round(n * test_frac))
    n_test = min(n_test, n - 1) if n > 1 else n_test
    test = patients[:n_test]
    remaining = patients[n_test:]
    if val_frac > 0 and remaining:
        n_val = min(max(1, round(n * val_frac)), max(0, len(remaining) - 1))
        val = remaining[:n_val]
        train = remaining[n_val:]
    else:
        val, train = [], remaining
    return set(train), set(val), set(test), []


# ============================================================================
# Grouped, class-stratified k-fold cross-validation
# ============================================================================

def _class_group_counts(group_to_classes: Dict[str, Iterable[int]]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for classes in group_to_classes.values():
        for c in set(classes):
            counts[c] += 1
    return counts


def grouped_stratified_kfold(group_to_classes: Dict[str, Iterable[int]], k: int, seed: int
                              ) -> List[Set[str]]:
    """Split GROUP ids (patients) into `k` disjoint folds. Each group is
    bucketed by its rarest class (fewest distinct groups); buckets are dealt
    rarest-first, shuffled with `random.Random(seed)`, round-robin across the
    `k` folds - so every class's groups are spread as evenly as the pool
    allows. Returns a list of `k` sets whose union is every group in
    `group_to_classes`."""
    if k < 2:
        raise ValueError(f"kfold k must be >= 2, got {k}")
    rng = random.Random(seed)
    cgc = _class_group_counts(group_to_classes)
    bucket: Dict[int, List[str]] = defaultdict(list)
    for g, classes in group_to_classes.items():
        cs = list(set(classes))
        if not cs:
            continue
        bucket[min(cs, key=lambda c: cgc[c])].append(g)

    folds: List[Set[str]] = [set() for _ in range(k)]
    for c in sorted(bucket, key=lambda c: len(bucket[c])):
        gs = sorted(bucket[c])
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            folds[i % k].add(g)
    return folds


def kfold_split(group_to_classes: Dict[str, Iterable[int]], k: int, fold_i: int, seed: int, *,
                 cv_val_frac: float = 0.0, cv_repeat: int = 0
                 ) -> Tuple[Set[str], Set[str], Set[str], List[str]]:
    """test = fold `fold_i` of `grouped_stratified_kfold`; optionally carve a
    stratified `val` of ~`cv_val_frac` (of all groups) out of the remaining
    folds; the rest is train. Runs the same `_coverage_fill_pass` as
    `stratified_patient_split_three`. `cv_repeat` perturbs the seed for
    repeated CV. Returns `(train, val, test, warnings)`."""
    if not (0 <= fold_i < k):
        raise ValueError(f"fold_i must be in [0, {k}), got {fold_i}")
    eff_seed = int(seed) + 100003 * int(cv_repeat)
    folds = grouped_stratified_kfold(group_to_classes, k, eff_seed)
    test: Set[str] = set(folds[fold_i])
    rest = [g for i, f in enumerate(folds) for g in f if i != fold_i]

    warnings: List[str] = []
    val: Set[str] = set()
    if cv_val_frac and cv_val_frac > 0:
        sub = {g: group_to_classes[g] for g in rest}
        kk = max(2, round(1.0 / cv_val_frac))
        val = set(grouped_stratified_kfold(sub, kk, eff_seed + 7)[0])

    train: Set[str] = set(group_to_classes) - test - val
    _coverage_fill_pass(train, val, test, group_to_classes,
                        _class_group_counts(group_to_classes), bool(val), warnings)
    return train, val, test, warnings


def stratified_group_split_three(
    group_to_meta: Dict[str, Tuple[Iterable[int], Tuple]], split_preset: str, seed: int,
) -> Tuple[Set[str], Set[str], Set[str], List[str]]:
    """Like `stratified_patient_split_three`, but each group also carries a
    tuple of extra stratification keys (e.g. body-site). Grouping is keyed by
    `(rarest_class, *strat_keys)` so coverage is defended per stratum. When
    every group's strat tuple is empty this delegates verbatim to
    `stratified_patient_split_three` (byte-identical result)."""
    if all(not meta[1] for meta in group_to_meta.values()):
        return stratified_patient_split_three(
            {g: meta[0] for g, meta in group_to_meta.items()}, split_preset, seed)

    train_frac, val_frac, test_frac = resolve_split_fractions(split_preset)
    has_val = val_frac > 0
    rng = random.Random(seed)

    g2c = {g: set(meta[0]) for g, meta in group_to_meta.items()}
    cgc = _class_group_counts(g2c)

    strata: Dict[Tuple, List[str]] = defaultdict(list)
    for g, (classes, strat) in group_to_meta.items():
        cs = list(set(classes))
        if not cs:
            continue
        rarest = min(cs, key=lambda c: cgc[c])
        strata[(rarest, *tuple(strat))].append(g)

    train: Set[str] = set()
    val: Set[str] = set()
    test: Set[str] = set()
    warnings: List[str] = []
    for key in sorted(strata, key=lambda kk: (len(strata[kk]), repr(kk))):
        patients = sorted(strata[key])
        rng.shuffle(patients)
        n = len(patients)
        if has_val:
            if n >= 3:
                n_test = min(max(1, round(n * test_frac)), n - 2)
                n_val = min(max(1, round(n * val_frac)), n - n_test - 1)
            elif n == 2:
                n_test, n_val = 1, 0
                warnings.append(f"stratum {key}: only 2 groups - no validation sample for it")
            else:
                n_test, n_val = 0, 0
                warnings.append(f"stratum {key}: only 1 group - train only")
        else:
            n_test = min(max(1, round(n * test_frac)), n - 1) if n >= 2 else 0
            n_val = 0
            if n < 2:
                warnings.append(f"stratum {key}: only 1 group - train only")
        test.update(patients[:n_test])
        val.update(patients[n_test:n_test + n_val])
        train.update(patients[n_test + n_val:])

    _coverage_fill_pass(train, val, test, g2c, cgc, has_val, warnings)
    return train, val, test, warnings


# ============================================================================
# Split assignment export / import (reuse one split across modalities/datasets)
# ============================================================================

_SPLIT_NAME_CANON = {"val": "validation", "valid": "validation", "validation": "validation",
                     "train": "train", "test": "test"}


def write_split_assignment(path: str, *, groups: Dict[str, str], strategy: str, seed: int,
                            spec: str, fractions: Tuple[float, float, float],
                            class_names: Optional[Sequence[str]] = None,
                            split_algo_version: int = SPLIT_ALGO_VERSION,
                            kfold_algo_version: Optional[int] = None) -> Dict:
    """Write `split_assignment.json` (atomically): the full `{group_id: split}`
    map plus provenance. `groups` values are 'train' | 'validation' | 'test'."""
    from training.npy_atomic import save_json_atomic
    obj = {
        "split_algo_version": split_algo_version,
        "kfold_algo_version": kfold_algo_version,
        "strategy": strategy,
        "seed": seed,
        "spec": spec,
        "fractions": {"train": fractions[0], "validation": fractions[1], "test": fractions[2]},
        "class_names": list(class_names) if class_names else None,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "groups": {str(g): _SPLIT_NAME_CANON.get(str(s), str(s)) for g, s in groups.items()},
    }
    save_json_atomic(path, obj)
    return obj


def load_split_assignment(path: str) -> Tuple[Set[str], Set[str], Set[str], Dict]:
    """Read a `split_assignment.json` back into `(train, val, test, meta)`
    group-id sets - to reuse one frozen split across modalities / datasets /
    reproductions via `--split_file`."""
    with open(path) as f:
        obj = json.load(f)
    groups = obj.get("groups") or {}
    train = {g for g, s in groups.items() if _SPLIT_NAME_CANON.get(s, s) == "train"}
    val = {g for g, s in groups.items() if _SPLIT_NAME_CANON.get(s, s) == "validation"}
    test = {g for g, s in groups.items() if _SPLIT_NAME_CANON.get(s, s) == "test"}
    return train, val, test, obj


# ============================================================================
# Class balancing (TRAIN split only, applied at the final unify step) -
# shared by every prep script's unify path via
# training.dataset_prep_unify.unify_split.
# ============================================================================

def class_counts(y: "np.ndarray", class_names: Sequence[str]) -> Dict[str, int]:
    y = np.asarray(y)
    return {class_names[c]: int((y == c).sum()) for c in range(len(class_names))}


def build_balance_selection(y_all: "np.ndarray", method: str, seed: int
                             ) -> Tuple["np.ndarray", Dict[int, int], Dict[int, int]]:
    """Returns `(row_indices_to_keep, before_counts, after_counts)`. Row
    indices are GLOBAL indices into `y_all`, shuffled, with repeats for
    oversampling. Deterministic: always `np.random.default_rng(seed)`."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_all)
    class_indices = {int(c): np.flatnonzero(y_all == c) for c in classes}
    before = {int(c): len(idx) for c, idx in class_indices.items()}

    selected = []
    if method == "undersample":
        target = min(before.values())
        for _c, idx in class_indices.items():
            selected.append(rng.choice(idx, size=target, replace=False) if len(idx) > target else idx)
    elif method == "oversample":
        target = max(before.values())
        for _c, idx in class_indices.items():
            if len(idx) < target:
                extra = rng.choice(idx, size=target - len(idx), replace=True)
                selected.append(np.concatenate([idx, extra]))
            else:
                selected.append(idx)
    else:
        raise ValueError(f"Unknown balance method: {method}")

    selected = np.concatenate(selected)
    rng.shuffle(selected)
    after = {int(c): int((y_all[selected] == c).sum()) for c in classes}
    return selected, before, after


# ============================================================================
# Fail-soft ingestion bookkeeping
# ============================================================================

class SkipRegistry:
    """Records inputs (images / captures) that were skipped during a
    resumable prep run so one corrupt file can't abort the whole job or be
    retried forever. Persisted into `_progress.json` and surfaced in
    `dataset_statistics.json`."""

    def __init__(self, existing: Optional[Dict[str, Dict]] = None):
        self._skips: Dict[str, Dict] = dict(existing or {})

    def add(self, key: str, reason: str, stage: str = "") -> None:
        self._skips[str(key)] = {"reason": str(reason), "stage": str(stage)}

    def __contains__(self, key) -> bool:
        return str(key) in self._skips

    def __len__(self) -> int:
        return len(self._skips)

    def as_dict(self) -> Dict[str, Dict]:
        return dict(self._skips)

    def summary(self, limit: int = 5) -> str:
        if not self._skips:
            return "no inputs skipped"
        items = list(self._skips.items())
        head = "; ".join(f"{k} ({v['stage']}): {v['reason']}" for k, v in items[:limit])
        more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
        return f"skipped {len(items)} input(s): {head}{more}"


def check_prep_class_coverage(
    item_labels: Sequence[Tuple[int, str]],
    assign_split_fn,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    fail_hard: bool = True,
) -> Dict:
    """Runs `training.class_coverage.verify_class_coverage` against
    ITEM-level (capture/image) labels and a `patient -> split` assignment
    function - i.e. BEFORE any patch/image extraction happens, so a bad
    split is caught at prep time (Phase 3's "must fail before training"),
    not only later when `train_example_v11.py` loads the finished .npy
    files.

    `item_labels`: `[(class_id, patient_id), ...]`, one entry per labeled
    capture/image that will be extracted.
    """
    splits: Dict[str, List[int]] = defaultdict(list)
    for label, patient in item_labels:
        splits[assign_split_fn(patient)].append(label)
    splits_nonempty = {k: v for k, v in splits.items() if v}
    return verify_class_coverage(splits_nonempty, num_classes=num_classes,
                                  class_names=class_names, fail_hard=fail_hard)
