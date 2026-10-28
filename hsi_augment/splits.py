# -*- coding: utf-8 -*-
"""
hsi_augment/splits.py
=====================
Phase 3 (#14): the fixed split verified, the leakage ledger, and the
leave-one-DCIS-patient-out (LOPO) folds inside train. The stage is
`augment_pipeline.py leakage` (`stages/leakage.py`, group `preprocess`); it
writes `<out>/prep/leakage/`:

    split_report.json     per split: rows, patients, captures, per-class row,
                          patient and capture counts; every check and its outcome
    folds.json            the LOPO folds (schema below). #16 pool, #28 QC and
                          #30 LOPO utility read it with `load_folds` and turn a
                          fold into rows with `fold_masks`, so they share one
                          definition of a fold
    leakage_ledger.json   one row per fitted quantity: what it is fitted on, who
                          fits it, the evidence, and its status

Why the fixed split is re-verified
----------------------------------
The split is fixed (seed 42, patient-grouped; 35/5/5 patients, DCIS 5/1/1 on
trainsel) and is NOT re-drawn: the A0 baseline and every paper number use it.
What can still go wrong is that the build's files disagree with each other: a
patient whose rows sit in two splits, a capture whose rows sit in two splits,
one capture key indexed twice, a row whose patient id is not its capture's
patient, a split_assignment.json that no longer describes the arrays, or a
Pass-1 report fitted on other patients than the split's train patients (band
selection, gain and value scale would then have seen held-out labels; an older
build did exactly that). Each is a hard failure (StageError). Only the int
sidecars (y, groups, captures) are read, under the `structural` purpose; never X.
Byte-identical patches across splits are P1's `check_patch_leakage` (#11).

The LOPO folds
--------------
Val has one DCIS patient (197), and it already selects checkpoints and arms, so
generator tuning (early stopping, beta/tau/w, QC thresholds, the A4 choice)
needs held-out DCIS patients from inside train. Fold k holds out the k-th DCIS
train patient (ascending id: 25, 45, 85, 107, 152 on trainsel) plus one
healthy and one IDC "companion" train patient.

* Holding out a patient removes ALL of their rows, every tissue. 29 of the 35
  trainsel train patients have captures of more than one tissue (every DCIS
  patient has all three), so a (patient, class) mask would leave the held-out
  patient's other tissues on the fold's train side: a patient leak. Masks are
  by patient only (`fold_masks`).
* A companion for class c is a train patient with class-c rows and NO DCIS
  rows. A DCIS patient drawn as a companion would be held out twice, once in
  its own fold and once in another.
* Companions are a seeded draw (`ctx.rng("lopo-folds")`). Companions disjoint
  across folds are preferred: a patient held out in two folds counts twice in
  every across-fold mean. A greedy draw can paint itself into a corner (it
  spends the only remaining healthy candidate as an IDC companion), so the
  draw is a maximum bipartite matching of (fold, class) slots to patients,
  with the slots visited in a random order over randomly permuted candidates.
  Only a slot no matching can fill reuses a patient (the least used one,
  random tie-break), and folds.json then says `companions_disjoint: false`.
* Every held-out set must hold rows of every class, and so must every fold's
  train side (a fold without DCIS train rows cannot fit a DCIS generator).

folds.json (`FOLDS_SCHEMA_VERSION` 1)
-------------------------------------
    {"schema_version": 1, "seed": 42,
     "seed_stream": {"stage": "leakage", "keys": ["lopo-folds"], "entropy": [...]},
     "rule": "...", "minority_class": "DCIS", "companion_classes": ["healthy", "IDC"],
     "companions_per_class": 1, "train_patients": [...], "dcis_patients": [25, ...],
     "candidates": {"healthy": [...], "IDC": [...]},
     "companions_disjoint": true, "reused_companions": [],
     "n_train_rows": 2452086, "n_folds": 5,
     "folds": [{"fold": 0, "dcis_patient": 25,
                "companions": {"healthy": [p], "IDC": [q]},
                "heldout_patients": [25, p, q],                  # sorted
                "heldout_rows": n, "train_rows": N - n,
                "heldout_rows_per_class": {"healthy": .., "DCIS": .., "IDC": ..},
                "train_rows_per_class": {...},
                "heldout_captures": c}, ...]}

Folds are numbered from 0 in DCIS-patient order. Any per-row patient-id array
works as `groups` (groups_train, the pool's `patient` sidecar, the provenance
index's `patient` column):

    folds = load_folds(ctx.upstream_dir("leakage"))
    train_mask, heldout_mask = fold_masks(groups_train, folds["folds"][k])

The leakage ledger
------------------
One row per quantity in the plan's Phase 3 table, plus the LOPO folds
themselves: `{quantity, fitted_on, fitted_on_split, stage, issue, evidence,
status}`. `fitted_on` is the plan's wording; `fitted_on_split` is the split
the quantity may be fitted on: train for everything the pipeline fits, val
only for the classifier's checkpoint and arm selection (Phase 10), and None
for test, which is used for the final report only. Status is one of
`LEDGER_STATUSES`: "verified" (checked against the build here), "enforced"
(guaranteed by SplitGuard and the stages' declared purposes, checked here) or
"pending" (fitted by the later issue named in `issue`).

The split-access policy
-----------------------
`io.SplitGuard` refuses val and test to every train-only purpose. What it
cannot see is a stage that DECLARES a held-out purpose it should not have, so
`registry_policy_violations` checks every registered stage:
  * `final_eval` (reads test) only for the Phase 10 stages in
    `FINAL_EVAL_STAGES` (`report-tissue`, #34): no stage before Phase 10 reads test;
  * a stage that pools, generates, runs QC or LOPO (`GENERATOR_SIDE_PURPOSES`)
    declares no held-out purpose at all, so it never sees val either;
  * `structural` only for the data phases, whose directories are under `prep/`.
The stage runs it on the live registry and fails on a violation, and
tests/test_hsi_augment_splits.py runs it on every registered stage, so a stage
added later is covered without editing either.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple,
                    Union)

import numpy as np

from hsi_augment.io import HELD_OUT_PURPOSES, PURPOSE_SPLITS, SPLITS, TRAIN_ONLY_PURPOSES
from hsi_augment.stage import StageError, stage_seed_sequence

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.adapters.base import CaptureInfo
    from hsi_augment.registry import Registry
    from hsi_augment.stage import StageContext

SPLIT_REPORT_NAME = "split_report.json"
FOLDS_NAME = "folds.json"
LEDGER_NAME = "leakage_ledger.json"
FOLDS_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
FOLDS_RNG_KEY = "lopo-folds"           # ctx.rng(FOLDS_RNG_KEY): the companion draw's stream
LABEL_KINDS = ("y", "groups", "captures")

# Phase 10 stages that may declare `final_eval` (read test for the final
# report). No stage before Phase 10 reads test. Adding a name here is a one-line
# reviewed change, like an edit of `io.PURPOSE_SPLITS`. `report-tissue` (#34)
# scores val/test X with the train-fitted tissue reference, without labels, for
# the report's tissue-vs-glass breakdown; it reads test only once selection.json exists.
FINAL_EVAL_STAGES: frozenset = frozenset({"report-tissue"})
# Purposes of the generator side. A stage declaring any of them may declare no
# held-out purpose: generator, QC and pool code never see val or test.
GENERATOR_SIDE_PURPOSES: Tuple[str, ...] = ("pool", "generator", "qc", "lopo")
# `structural` (counts, ids, hashes of any split) is for the data phases only.
STRUCTURAL_DIR_PREFIX = "prep/"

CHECK_SEVERITIES = ("error", "warning")
PathLike = Union[str, os.PathLike]


# ============================================================================
# the int sidecars of a split
# ============================================================================

@dataclass(frozen=True)
class SplitLabels:
    """The per-row int sidecars of one split: all this phase reads (never X)."""
    split: str
    y: np.ndarray            # int64 class index
    groups: np.ndarray       # int64 patient id
    captures: np.ndarray     # int64 capture index (capture_index.json key)

    @classmethod
    def from_arrays(cls, split: str, y: Any, groups: Any, captures: Any) -> "SplitLabels":
        arrays = [np.asarray(a, dtype=np.int64).reshape(-1) for a in (y, groups, captures)]
        if len({a.shape[0] for a in arrays}) != 1:
            raise StageError(f"{split}: y, groups and captures have different lengths "
                             f"{[a.shape[0] for a in arrays]}")
        return cls(split, *arrays)

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


def open_split_labels(ctx: "StageContext", split: str, purpose: str = "structural") -> SplitLabels:
    """y, groups and captures of `split`, through the stage's SplitGuard."""
    arrays = ctx.open_split(split, purpose, kinds=LABEL_KINDS)
    missing = [f"{k}_{arrays.split}.npy" for k in LABEL_KINDS if getattr(arrays, k) is None]
    if missing:
        raise StageError(f"{arrays.split}: {', '.join(missing)} missing; patients and captures "
                         f"cannot be verified")
    return SplitLabels.from_arrays(arrays.split, arrays.y, arrays.groups, arrays.captures)


def _check(name: str, ok: bool, detail: str, *, severity: str = "error", **data: Any) -> Dict[str, Any]:
    assert severity in CHECK_SEVERITIES
    return {"check": name, "ok": bool(ok), "severity": severity, "detail": detail, **data}


def failed_checks(checks: Iterable[Mapping[str, Any]], severity: str = "error") -> List[Mapping[str, Any]]:
    return [c for c in checks if not c["ok"] and c["severity"] == severity]


def _per_class(counts: Sequence[int], class_names: Sequence[str]) -> Dict[str, int]:
    return {name: int(counts[i]) for i, name in enumerate(class_names)}


def _head(values: Iterable[Any], n: int = 10) -> List[Any]:
    """The first `n` values, so a report stays readable when everything breaks."""
    out = []
    for v in values:
        if len(out) == n:
            break
        out.append(v)
    return out


# ============================================================================
# split verification
# ============================================================================

def patient_class_rows(groups: np.ndarray, y: np.ndarray, num_classes: int) -> Dict[int, np.ndarray]:
    """{patient: rows per class (int64, length num_classes)}, patients ascending."""
    g = np.asarray(groups, dtype=np.int64)
    yy = np.asarray(y, dtype=np.int64)
    patients, inv = np.unique(g, return_inverse=True)
    counts = np.bincount(inv * num_classes + yy, minlength=len(patients) * num_classes)
    counts = counts.reshape(len(patients), num_classes)
    return {int(p): counts[i] for i, p in enumerate(patients)}


def _split_summary(lab: SplitLabels, class_names: Sequence[str]) -> Dict[str, Any]:
    C = len(class_names)
    pcr = patient_class_rows(lab.groups, lab.y, C)
    caps, cap_inv = np.unique(lab.captures, return_inverse=True)
    cap_cls = np.unique(cap_inv * C + lab.y) % C
    by_class = {name: [p for p, r in pcr.items() if r[i] > 0] for i, name in enumerate(class_names)}
    return {
        "rows": lab.n,
        "n_patients": len(pcr),
        "patients": sorted(pcr),
        "n_captures": int(len(caps)),
        "rows_per_class": _per_class(np.bincount(lab.y, minlength=C), class_names),
        "patients_per_class": {name: len(v) for name, v in by_class.items()},
        "captures_per_class": _per_class(np.bincount(cap_cls, minlength=C), class_names),
        "patients_by_class": by_class,
        # holding one of these out must remove every tissue of theirs (see fold_masks)
        "multi_class_patients": [p for p, r in pcr.items() if int((r > 0).sum()) > 1],
    }


def verify_splits(labels: Mapping[str, SplitLabels], capture_index: Mapping[int, "CaptureInfo"],
                  class_names: Sequence[str], minority_classes: Sequence[int], *,
                  assigned: Optional[Mapping[str, Sequence[int]]] = None) -> Dict[str, Any]:
    """The split report: per-split summaries and every disjointness /
    consistency check (`report["checks"]`, `report["ok"]`). Nothing raises
    here; `run` turns failed "error" checks into a StageError.

    `labels` maps a canonical split name to its sidecars; `assigned` is
    split_assignment.json as {split: patients} (None if the build has none)."""
    C = len(class_names)
    splits = [s for s in SPLITS if s in labels]
    checks: List[Dict[str, Any]] = []

    bad_labels = {s: int(((labels[s].y < 0) | (labels[s].y >= C)).sum()) for s in splits}
    checks.append(_check("labels_in_range", not any(bad_labels.values()),
                         f"rows with y outside 0..{C - 1}: {bad_labels}"))
    if any(bad_labels.values()):    # the per-class tables below would be meaningless
        return {"splits": {}, "checks": checks, "ok": False}

    per_split = {s: _split_summary(labels[s], class_names) for s in splits}
    patients = {s: set(per_split[s]["patients"]) for s in splits}
    captures = {s: set(np.unique(labels[s].captures).tolist()) for s in splits}

    # -- disjointness: the leak checks proper ----------------------------------
    overlap_p = {f"{a}&{b}": sorted(patients[a] & patients[b]) for a, b in combinations(splits, 2)}
    overlap_p = {k: v for k, v in overlap_p.items() if v}
    checks.append(_check("patients_disjoint", not overlap_p,
                         f"patients in two splits: {overlap_p}" if overlap_p else
                         "no patient has rows in two splits", overlaps=overlap_p))
    overlap_c = {f"{a}&{b}": sorted(captures[a] & captures[b]) for a, b in combinations(splits, 2)}
    overlap_c = {k: _head(v) for k, v in overlap_c.items() if v}
    checks.append(_check("captures_disjoint", not overlap_c,
                         f"captures with rows in two splits: {overlap_c}" if overlap_c else
                         "no capture has rows in two splits", overlaps=overlap_c))
    key_splits: Dict[str, Set[str]] = {}
    key_count: Counter = Counter()
    for info in capture_index.values():
        key_splits.setdefault(info.key, set()).add(info.split)
        key_count[info.key] += 1
    dup_keys = {k: sorted(key_splits[k]) for k, n in key_count.items() if n > 1}
    checks.append(_check("capture_keys_unique", not dup_keys,
                         f"capture keys indexed more than once (key: splits): {dict(_head(dup_keys.items()))}"
                         if dup_keys else f"{len(key_count)} capture keys, each indexed once",
                         duplicates=len(dup_keys)))

    # -- every row's capture is indexed, in this split, for this patient ---------
    ids = np.array(sorted(capture_index), dtype=np.int64)
    cap_patient = np.array([capture_index[int(i)].patient for i in ids], dtype=np.int64)
    cap_split = np.array([SPLITS.index(capture_index[int(i)].split) for i in ids], dtype=np.int64)
    unknown: Dict[str, List[int]] = {}
    wrong_split: Dict[str, List[int]] = {}
    wrong_patient: Dict[str, Dict[str, Any]] = {}
    unlisted: Dict[str, List[int]] = {}
    for s in splits:
        lab = labels[s]
        listed = {i.index for i in capture_index.values() if i.split == s}
        if not len(ids):
            if lab.n:
                unknown[s] = _head(np.unique(lab.captures).tolist())
            continue
        pos = np.minimum(np.searchsorted(ids, lab.captures), len(ids) - 1)
        known = ids[pos] == lab.captures
        if (~known).any():
            unknown[s] = _head(np.unique(lab.captures[~known]).tolist())
        ws = np.unique(lab.captures[known & (cap_split[pos] != SPLITS.index(s))])
        if len(ws):
            wrong_split[s] = _head(ws.tolist())
        wp = known & (cap_patient[pos] != lab.groups)
        if wp.any():
            wrong_patient[s] = {"rows": int(wp.sum()),
                                "captures": _head(np.unique(lab.captures[wp]).tolist())}
        missing = sorted(listed - captures[s])
        if missing:
            unlisted[s] = _head(missing)
    checks.append(_check("captures_indexed", not unknown,
                         f"capture ids missing from capture_index.json: {unknown}" if unknown else
                         "every row's capture is in capture_index.json"))
    checks.append(_check("capture_split_matches_rows", not wrong_split,
                         f"rows of captures that capture_index.json puts in another split: {wrong_split}"
                         if wrong_split else "every capture's rows are in its indexed split"))
    checks.append(_check("row_patient_matches_capture", not wrong_patient,
                         f"rows whose groups id is not their capture's patient: {wrong_patient}"
                         if wrong_patient else "groups == capture_index patient on every row"))
    checks.append(_check("indexed_captures_have_rows", not unlisted,
                         f"captures indexed for a split but with no rows in it: {unlisted}" if unlisted
                         else "every indexed capture has rows in its split", severity="warning"))

    # -- split_assignment.json describes the arrays -------------------------------
    if assigned is None:
        checks.append(_check("split_assignment_matches_rows", False,
                             "the build has no split_assignment.json to check the rows against"))
    else:
        diff = {}
        for s in splits:
            a = {int(p) for p in assigned.get(s, ())}
            if a != patients[s]:
                diff[s] = {"only_assignment": sorted(a - patients[s]),
                           "only_rows": sorted(patients[s] - a)}
        checks.append(_check("split_assignment_matches_rows", not diff,
                             f"split_assignment.json disagrees with the rows: {diff}" if diff else
                             "split_assignment.json lists exactly each split's row patients"))

    # -- the minority class in every split (a fact of the build, not a leak) -----
    minority_names = [class_names[c] for c in minority_classes]
    no_minority = [s for s in splits
                   if not any(per_split[s]["patients_per_class"][n] for n in minority_names)]
    checks.append(_check("minority_patients_in_every_split", not no_minority,
                         f"splits without a {'/'.join(minority_names)} patient: {no_minority}" if no_minority
                         else f"every split has a {'/'.join(minority_names)} patient", severity="warning"))

    def ratio(fn) -> str:
        return "/".join(str(fn(s)) for s in splits)

    return {
        "splits_order": splits,
        "patients": ratio(lambda s: per_split[s]["n_patients"]),
        "patients_per_class": {n: ratio(lambda s, n=n: per_split[s]["patients_per_class"][n])
                               for n in class_names},
        "captures": ratio(lambda s: per_split[s]["n_captures"]),
        "minority_classes": minority_names,
        "splits": per_split,
        "checks": checks,
        "ok": not failed_checks(checks),
    }


def check_pass1_scope(scope: Optional[Mapping[str, Any]], split_patients: Mapping[str, Iterable[int]],
                      n_train_captures: int, *, missing_reason: str = "") -> Dict[str, Any]:
    """The band-selection scope (Plan P1.6): Pass 1 (band selection, gain,
    value scale) must have been fitted on exactly the split's train patients,
    so it saw no held-out label. `scope` is pass1_scope_report.json."""
    if scope is None:
        return _check("pass1_scope", False, "no Pass-1 scope report: band selection, gain and value scale "
                      f"cannot be shown to be fitted on train only{': ' + missing_reason if missing_reason else ''}")
    train = {int(p) for p in split_patients.get("train", ())}
    held = {int(p) for s in SPLITS if s != "train" for p in split_patients.get(s, ())}
    p1_train = {int(p) for p in scope.get("train_patients") or ()}
    problems = []
    if p1_train != train:
        problems.append(f"Pass-1 train patients differ from the split's: only in Pass 1 "
                        f"{sorted(p1_train - train)}, only in the split {sorted(train - p1_train)}")
    if scope.get("heldout_patients") is not None:
        p1_held = {int(p) for p in scope["heldout_patients"]}
        if p1_held != held:
            problems.append(f"Pass-1 held-out patients differ from val+test: only in Pass 1 "
                            f"{sorted(p1_held - held)}, only in val+test {sorted(held - p1_held)}")
    if scope.get("n_train_captures") is not None and int(scope["n_train_captures"]) != int(n_train_captures):
        problems.append(f"Pass 1 used {scope['n_train_captures']} train captures, capture_index.json "
                        f"has {n_train_captures}")
    detail = "; ".join(problems) or (
        f"pass1_scope_report.json: Pass-1 train patients == the split's {len(train)} train patients; "
        f"held-out patients == the {len(held)} val+test patients; n_train_captures "
        f"{scope.get('n_train_captures')} == train captures in capture_index.json")
    return _check("pass1_scope", not problems, detail, rule=scope.get("rule"),
                  pass1_train_patients=len(p1_train), split_train_patients=len(train))


# ============================================================================
# LOPO folds
# ============================================================================

def fold_heldout_patients(fold: Union[Mapping[str, Any], Iterable[int]]) -> Tuple[int, ...]:
    """A fold record's `heldout_patients` (or an iterable of patient ids), sorted."""
    ids = fold["heldout_patients"] if isinstance(fold, Mapping) else fold
    return tuple(sorted(int(p) for p in ids))


def fold_masks(groups: np.ndarray, fold: Union[Mapping[str, Any], Iterable[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """(train_mask, heldout_mask): bool per row of `groups` (any per-row
    patient-id array). A held-out patient loses ALL of their rows, every
    tissue; the masks are complements. `fold` is a folds.json fold record or
    the held-out patient ids."""
    held = np.isin(np.asarray(groups), np.asarray(fold_heldout_patients(fold), dtype=np.int64))
    return ~held, held


def _draw_companions(slots: Sequence[Tuple[int, int]], candidates: Mapping[int, Sequence[int]],
                     rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    """A patient per (fold, class) slot, and the patients used by more than
    one slot. A maximum matching (Kuhn's augmenting paths) over random slot
    order and randomly permuted candidates, so companions are disjoint across
    folds whenever that is possible; unmatched slots reuse the least used
    eligible patient not already in their fold (random tie-break: the
    candidates are already in random order)."""
    order = [int(i) for i in rng.permutation(len(slots))]
    prefs = [[int(p) for p in rng.permutation(np.asarray(candidates[c], dtype=np.int64))] for _, c in slots]
    owner: Dict[int, int] = {}                      # patient -> slot

    def augment(i: int, seen: Set[int]) -> bool:
        for p in prefs[i]:
            if p in seen:
                continue
            seen.add(p)
            if p not in owner or augment(owner[p], seen):
                owner[p] = i
                return True
        return False

    for i in order:
        augment(i, set())
    chosen: List[Optional[int]] = [None] * len(slots)
    for p, i in owner.items():
        chosen[i] = p
    uses: Counter = Counter(list(owner))            # each matched patient once (not Counter(dict))
    for i in order:
        if chosen[i] is not None:
            continue
        fold = slots[i][0]
        in_fold = {chosen[j] for j, (f, _) in enumerate(slots) if f == fold and chosen[j] is not None}
        options = [p for p in prefs[i] if p not in in_fold]
        if not options:
            raise StageError(f"LOPO fold {fold}: no companion left for class index {slots[i][1]} "
                             f"(every candidate is already held out in this fold)")
        least = min(uses[p] for p in options)
        pick = next(p for p in options if uses[p] == least)
        chosen[i] = pick
        uses[pick] += 1
    return [int(p) for p in chosen], sorted(p for p, n in uses.items() if n > 1)


def folds_rule(minority: Sequence[str], companions: Sequence[str], per_class: int) -> str:
    return (f"Fold k (from 0) holds out the k-th {'/'.join(minority)} train patient in ascending id order, "
            f"plus {per_class} companion train patient(s) per class in {list(companions)}: a train patient "
            f"with rows of that class and no {'/'.join(minority)} rows, so every {'/'.join(minority)} train "
            f"patient is held out exactly once. Companions are a seeded draw: a maximum bipartite matching "
            f"of (fold, class) slots to patients, slots in random order over randomly permuted candidates, "
            f"so no companion is held out in two folds when that is possible; a slot no matching can fill "
            f"reuses the least used eligible patient. Holding out a patient removes ALL of their rows, "
            f"every tissue (splits.fold_masks). Every held-out set and every fold's train side holds rows "
            f"of every class.")


def make_lopo_folds(groups: np.ndarray, y: np.ndarray, *, class_names: Sequence[str],
                    minority_classes: Sequence[int], rng: np.random.Generator, companions_per_class: int = 1,
                    captures: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """The folds.json document, without the seed fields (`run` adds them).
    `groups`/`y`/`captures` are the TRAIN split's per-row sidecars."""
    C = len(class_names)
    g = np.asarray(groups, dtype=np.int64)
    yy = np.asarray(y, dtype=np.int64)
    minority = sorted({int(c) for c in minority_classes})
    if not minority:
        raise StageError("the adapter declares no minority class; the LOPO folds hold out minority patients")
    if companions_per_class < 1:
        raise StageError("companions_per_class must be >= 1")
    m_names = [class_names[c] for c in minority]
    others = [c for c in range(C) if c not in minority]
    pcr = patient_class_rows(g, yy, C)
    dcis = [p for p, r in pcr.items() if r[minority].sum() > 0]
    if len(dcis) < 2:
        raise StageError(f"LOPO needs at least 2 {'/'.join(m_names)} train patients (one held out, one to "
                         f"fit on); train has {dcis}")
    dcis_set = set(dcis)
    candidates = {c: [p for p, r in pcr.items() if p not in dcis_set and r[c] > 0] for c in others}
    for c, cand in candidates.items():
        if len(cand) < companions_per_class:
            raise StageError(f"{len(cand)} train patient(s) without {'/'.join(m_names)} rows have "
                             f"{class_names[c]} rows; each fold needs {companions_per_class}")

    slots = [(k, c) for k in range(len(dcis)) for c in others for _ in range(companions_per_class)]
    chosen, reused = _draw_companions(slots, candidates, rng)
    folds = []
    for k, d in enumerate(dcis):
        companions = {class_names[c]: sorted(p for (f, cc), p in zip(slots, chosen) if f == k and cc == c)
                      for c in others}
        held = sorted({d, *(p for v in companions.values() for p in v)})
        train_mask, held_mask = fold_masks(g, held)
        rec: Dict[str, Any] = {
            "fold": k, "dcis_patient": d, "companions": companions, "heldout_patients": held,
            "heldout_rows": int(held_mask.sum()), "train_rows": int(train_mask.sum()),
            "heldout_rows_per_class": _per_class(np.bincount(yy[held_mask], minlength=C), class_names),
            "train_rows_per_class": _per_class(np.bincount(yy[train_mask], minlength=C), class_names),
        }
        if captures is not None:
            rec["heldout_captures"] = int(len(np.unique(np.asarray(captures)[held_mask])))
        folds.append(rec)
    return {
        "rule": folds_rule(m_names, [class_names[c] for c in others], companions_per_class),
        "minority_class": "/".join(m_names),
        "companion_classes": [class_names[c] for c in others],
        "companions_per_class": int(companions_per_class),
        "train_patients": sorted(pcr),
        "dcis_patients": dcis,
        "candidates": {class_names[c]: v for c, v in candidates.items()},
        "companions_disjoint": not reused,
        "reused_companions": reused,
        "n_train_rows": int(g.shape[0]),
        "n_folds": len(folds),
        "folds": folds,
    }


def check_folds(doc: Mapping[str, Any], groups: np.ndarray, y: np.ndarray, *, class_names: Sequence[str],
                minority_classes: Sequence[int], captures: Optional[np.ndarray] = None) -> List[str]:
    """Problems with a folds document against the train sidecars (empty =
    valid). Downstream stages can call it to refuse a folds.json from
    another build."""
    C = len(class_names)
    g = np.asarray(groups, dtype=np.int64)
    yy = np.asarray(y, dtype=np.int64)
    minority = sorted({int(c) for c in minority_classes})
    pcr = patient_class_rows(g, yy, C)
    dcis = [p for p, r in pcr.items() if r[minority].sum() > 0]
    folds = list(doc.get("folds") or [])
    problems: List[str] = []
    if [f.get("fold") for f in folds] != list(range(len(folds))):
        problems.append(f"fold numbers {[f.get('fold') for f in folds]} are not 0..{len(folds) - 1}")
    if doc.get("n_train_rows") is not None and int(doc["n_train_rows"]) != int(g.shape[0]):
        problems.append(f"folds were drawn on {doc['n_train_rows']} train rows, this train split has {g.shape[0]}")
    held_count = Counter(p for f in folds for p in fold_heldout_patients(f))
    as_dcis = Counter(int(f["dcis_patient"]) for f in folds)
    for p in dcis:
        if held_count[p] != 1 or as_dcis[p] != 1:
            problems.append(f"DCIS train patient {p} is held out {held_count[p]} time(s), "
                            f"{as_dcis[p]} as a fold's DCIS patient; expected exactly once")
    not_dcis = sorted(set(as_dcis) - set(dcis))
    if not_dcis:
        problems.append(f"fold DCIS patient(s) {not_dcis} have no DCIS train rows")
    for f in folds:
        k = f.get("fold")
        held = fold_heldout_patients(f)
        if int(f["dcis_patient"]) not in held:
            problems.append(f"fold {k}: its DCIS patient {f['dcis_patient']} is not held out")
        foreign = sorted(set(held) - set(pcr))
        if foreign:
            problems.append(f"fold {k}: held-out patient(s) {foreign} have no train rows")
        train_mask, held_mask = fold_masks(g, f)
        leaked = sorted(set(np.unique(g[train_mask]).tolist()) & set(held))
        if leaked:
            problems.append(f"fold {k}: held-out patient(s) {leaked} have rows in the fold's train mask")
        h = np.bincount(yy[held_mask], minlength=C)
        t = np.bincount(yy[train_mask], minlength=C)
        lacking = [class_names[c] for c in range(C) if h[c] == 0]
        if lacking:
            problems.append(f"fold {k}: the held-out set has no {lacking} rows")
        lacking = [class_names[c] for c in range(C) if t[c] == 0]
        if lacking:
            problems.append(f"fold {k}: the train side has no {lacking} rows")
        if f.get("heldout_rows_per_class") is not None and \
                dict(f["heldout_rows_per_class"]) != _per_class(h, class_names):
            problems.append(f"fold {k}: recorded held-out rows per class {f['heldout_rows_per_class']} "
                            f"!= {_per_class(h, class_names)} in these arrays (folds from another build?)")
        if captures is not None:
            cap = np.asarray(captures)
            both = np.intersect1d(cap[held_mask], cap[train_mask])
            if len(both):
                problems.append(f"fold {k}: captures {_head(both.tolist())} have rows on both sides")
    return problems


def load_folds(path: PathLike) -> Dict[str, Any]:
    """folds.json, from its path or from the leakage stage directory."""
    p = Path(path)
    if p.is_dir():
        p = p / FOLDS_NAME
    with open(p) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("schema_version") != FOLDS_SCHEMA_VERSION:
        raise ValueError(f"{p}: not a folds.json of schema_version {FOLDS_SCHEMA_VERSION}")
    for key in ("seed", "rule", "dcis_patients", "folds"):
        if key not in doc:
            raise ValueError(f"{p}: missing key {key!r}")
    return doc


# ============================================================================
# the split-access policy over registered stages
# ============================================================================

def registry_policy_violations(registry: "Registry") -> List[str]:
    """Registrations that break the split-access policy (module docstring)."""
    out = []
    for spec in registry.specs():
        held = [p for p in spec.purposes if p in HELD_OUT_PURPOSES]
        gen = [p for p in spec.purposes if p in GENERATOR_SIDE_PURPOSES]
        if "final_eval" in spec.purposes and spec.name not in FINAL_EVAL_STAGES:
            out.append(f"stage {spec.name!r} declares 'final_eval' (reads test) but is not a Phase 10 stage "
                       f"in splits.FINAL_EVAL_STAGES {sorted(FINAL_EVAL_STAGES)}")
        if held and gen:
            out.append(f"stage {spec.name!r} declares {gen} and also {held}: generator, QC, pool and LOPO "
                       f"code may read train only")
        if "structural" in spec.purposes and not (spec.dirname + "/").startswith(STRUCTURAL_DIR_PREFIX):
            out.append(f"stage {spec.name!r} declares 'structural' but its directory {spec.dirname!r} is not "
                       f"under {STRUCTURAL_DIR_PREFIX!r} (data phases only)")
    return out


def held_out_readers(registry: "Registry") -> Dict[str, List[str]]:
    """{stage: its declared purposes that may read val/test}, for the ledger."""
    return {s.name: [p for p in s.purposes if p in HELD_OUT_PURPOSES]
            for s in registry.specs() if s.reads_held_out()}


# ============================================================================
# the leakage ledger
# ============================================================================

LEDGER_STATUSES: Dict[str, str] = {
    "verified": "checked against the build's own files by the leakage stage",
    "enforced": "guaranteed by io.SplitGuard and the registered stages' declared purposes; checked by "
                "the leakage stage",
    "pending": "fitted by a later stage (see `issue`), which must fit it on `fitted_on_split` only",
}


@dataclass(frozen=True)
class LedgerRow:
    quantity: str
    fitted_on: str                    # the plan's wording
    fitted_on_split: Optional[str]    # "train" | "val" | None (nothing is fitted on test)
    stage: str
    issue: str
    note: str = ""                    # the evidence a pending row will have


CLASSIFIER_SELECTION = "Classifier checkpoint + arm selection"
TEST_QUANTITY = "Test"
BAND_SELECTION = "Band selection, gain, value scale"
LOPO_FOLDS = "LOPO folds"

# Plan §Phase 3, item 2, plus the LOPO folds this stage draws (rows 5-7 rely on them).
LEDGER_TABLE: Tuple[LedgerRow, ...] = (
    LedgerRow(BAND_SELECTION, "train", "train", "Pass 1 of prepare_histologyhsi_bc.py (the build)", "#14",
              "verified here against pass1_scope_report.json"),
    LedgerRow("Linear and log stats, tau, s_glass, sigma_b", "train", "train",
              "P2a spectral-audit (stats, sigma_b); P2b tissue-mask (tau, s_glass)", "#12, #13",
              "fitted on train rows opened with SplitGuard purpose 'fit'"),
    LedgerRow("Feature spaces (PCA-64, mean/std) and QC references", "train (LOPO for thresholds)", "train",
              "P4b pool (real-vs-real references); P8b qc (feature spaces, thresholds)", "#16, #28",
              "references from each fold's held-out patients against the rest, folds from "
              "prep/leakage/folds.json; QC thresholds calibrated on LOPO held-out DCIS"),
    LedgerRow("CVAE / AE / LDM weights", "train", "train", "P6 fit-cvae; P7 fit-ae, fit-ldm", "#23, #24",
              "generator stages read train only (SplitGuard purpose 'generator')"),
    LedgerRow("Generator early stopping, beta/tau/w choices", "train LOPO folds", "train",
              "P6 fit-cvae; P7 fit-ldm; P8a generate", "#23, #24, #29",
              "chosen on the LOPO folds of prep/leakage/folds.json, never on val; tau_s / w in "
              "synth/<source>/choice.json"),
    LedgerRow("QC classifier", "train (cross-fitted LOPO)", "train", "P8b qc", "#28",
              "cross-fitted over the LOPO folds of prep/leakage/folds.json"),
    LedgerRow("A4 generator choice", "LOPO utility (train)", "train", "P8c lopo-utility", "#30",
              "lopo_utility/decision.json, decided on the LOPO folds before the screen starts"),
    LedgerRow(CLASSIFIER_SELECTION, "val", "val",
              "P10 train.py checkpoint selection; augmentation_select.py arm selection", "#33",
              "val selects checkpoints and arms in the classifier only"),
    LedgerRow(TEST_QUANTITY, "final report only", None, "P10 report", "#34",
              "test is read for the final report only, after selection.json is frozen"),
    LedgerRow(LOPO_FOLDS, "train", "train", "P3 leakage (this stage)", "#14", "verified here"),
)
LEDGER_QUANTITIES: Tuple[str, ...] = tuple(r.quantity for r in LEDGER_TABLE)


def folds_evidence(doc: Mapping[str, Any]) -> str:
    folds = doc["folds"]
    sizes = [len(f["heldout_patients"]) for f in folds]
    return (f"folds.json: {len(folds)} folds; every DCIS train patient {doc['dcis_patients']} held out exactly "
            f"once; held-out sets of {sorted(set(sizes))} train patient(s), all their rows; every held-out set "
            f"and train side has every class; companions disjoint across folds: {doc['companions_disjoint']}; "
            f"seed {doc.get('seed')}")


def build_ledger(*, pass1: Mapping[str, Any], folds_doc: Mapping[str, Any],
                 readers: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """The ledger document. Called only after every check passed: a failed
    check is a StageError, never a ledger row with a bad status."""
    reader_txt = ", ".join(f"{k} ({'/'.join(v)})" for k, v in sorted(readers.items())) or "none"
    final = sorted(k for k, v in readers.items() if "final_eval" in v)
    test_evidence = (f"registered stages declaring 'final_eval': {final or 'none'} (Phase 10 stages allowed: "
                     f"{sorted(FINAL_EVAL_STAGES) or 'none yet'}); SplitGuard refuses test to "
                     f"{list(TRAIN_ONLY_PURPOSES)}; stages that may read held-out splits: {reader_txt}; "
                     f"'structural' is for counts and ids only")
    val_evidence = (f"no hsi_augment stage reads val except structurally or for the final report (stages that "
                    f"may: {reader_txt}); generator, QC, pool and LOPO purposes refuse val. Selection itself "
                    f"is Phase 10's")
    dynamic = {
        BAND_SELECTION: ("verified", pass1["detail"]),
        LOPO_FOLDS: ("verified", folds_evidence(folds_doc)),
        TEST_QUANTITY: ("enforced", test_evidence),
        CLASSIFIER_SELECTION: ("pending", val_evidence),
    }
    entries = []
    for row in LEDGER_TABLE:
        status, evidence = dynamic.get(row.quantity, ("pending", row.note))
        entries.append({"quantity": row.quantity, "fitted_on": row.fitted_on,
                        "fitted_on_split": row.fitted_on_split, "stage": row.stage, "issue": row.issue,
                        "evidence": evidence, "status": status})
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "rule": "One row per quantity the pipeline fits or selects (Plan section Phase 3). Everything the "
                "pipeline fits is fitted on train rows only; val selects the classifier's checkpoint and "
                "arms only; test is used for the final report only.",
        "statuses": dict(LEDGER_STATUSES),
        "split_access": {"purpose_splits": {p: list(s) for p, s in PURPOSE_SPLITS.items()},
                         "held_out_readers": {k: list(v) for k, v in sorted(readers.items())},
                         "final_eval_stages": sorted(FINAL_EVAL_STAGES)},
        "entries": entries,
    }


def ledger_problems(ledger: Mapping[str, Any]) -> List[str]:
    """Unknown or ill-formed ledger entries (empty = valid)."""
    problems = []
    entries = list(ledger.get("entries") or [])
    names = [e.get("quantity") for e in entries]
    unknown = sorted(set(names) - set(LEDGER_QUANTITIES), key=str)
    missing = [q for q in LEDGER_QUANTITIES if q not in names]
    dupes = sorted(q for q, n in Counter(names).items() if n > 1)
    if unknown or missing or dupes:
        problems.append(f"unknown {unknown}, missing {missing}, duplicated {dupes} quantities")
    for e in entries:
        q = e.get("quantity")
        if e.get("status") not in LEDGER_STATUSES:
            problems.append(f"{q!r}: unknown status {e.get('status')!r}")
        want = "val" if q == CLASSIFIER_SELECTION else None if q == TEST_QUANTITY else "train"
        if e.get("fitted_on_split") != want:
            problems.append(f"{q!r}: fitted_on_split {e.get('fitted_on_split')!r}, expected {want!r}")
        if not e.get("evidence") or not e.get("issue"):
            problems.append(f"{q!r}: no evidence or issue")
    return problems


# ============================================================================
# the stage
# ============================================================================

def run(ctx: "StageContext", cfg: Any) -> Dict[str, Any]:
    """`augment_pipeline.py leakage`: split report, folds, ledger (in that
    order; the first failed check raises StageError before the next file)."""
    adapter = ctx.adapter
    class_names = list(adapter.class_names)
    minority = list(adapter.minority_classes)

    # 1. The int sidecars of every split. Peak RAM is these as int64 (~120 MB
    #    on trainsel), so there is no ram_guard (it would import torch).
    with ctx.timer("read"):
        labels = {s: open_split_labels(ctx, s) for s in SPLITS}

    # 2. Disjointness and consistency, then the Pass-1 scope.
    with ctx.timer("verify"):
        try:
            assigned: Optional[Dict[str, Tuple[int, ...]]] = adapter.split_patients()
        except (AttributeError, FileNotFoundError) as e:
            ctx.warn(f"no split assignment to check against: {e}")
            assigned = None
        report = verify_splits(labels, adapter.capture_index, class_names, minority, assigned=assigned)
        if report["splits"]:
            scope, why = None, ""
            try:
                scope = adapter.pass1_scope
            except (AttributeError, FileNotFoundError) as e:
                why = str(e)
            pass1 = check_pass1_scope(scope, {s: report["splits"][s]["patients"] for s in SPLITS},
                                      len(adapter.captures_in_split("train")), missing_reason=why)
            report["checks"].append(pass1)
            report["ok"] = not failed_checks(report["checks"])
    ctx.save_json(SPLIT_REPORT_NAME, report)     # also on failure: it says what broke
    for c in failed_checks(report["checks"], "warning"):
        ctx.warn(f"{c['check']}: {c['detail']}")
    errors = failed_checks(report["checks"])
    if errors:
        raise StageError("split verification failed: " +
                         "; ".join(f"{c['check']}: {c['detail']}" for c in errors))
    ctx.log(f"split ok: patients {report['patients']} (train/val/test), "
            + ", ".join(f"{n} patients {r}" for n, r in report["patients_per_class"].items())
            + f", captures {report['captures']}; patients and captures disjoint; {pass1['detail']}")
    multi = report["splits"]["train"]["multi_class_patients"]
    ctx.log(f"{len(multi)} of {report['splits']['train']['n_patients']} train patients have rows of more "
            f"than one class; a held-out patient loses all of them")

    # 3. LOPO folds inside train.
    train = labels["train"]
    with ctx.timer("folds"):
        body = make_lopo_folds(train.groups, train.y, class_names=class_names, minority_classes=minority,
                               rng=ctx.rng(FOLDS_RNG_KEY), companions_per_class=cfg.companions_per_class,
                               captures=train.captures)
        stream = stage_seed_sequence(ctx.seed, ctx.name, FOLDS_RNG_KEY)
        folds = {"schema_version": FOLDS_SCHEMA_VERSION, "seed": ctx.seed,
                 "seed_stream": {"stage": ctx.name, "keys": [FOLDS_RNG_KEY],
                                 "entropy": [int(v) for v in stream.entropy]},
                 **body}
        problems = check_folds(folds, train.groups, train.y, class_names=class_names,
                               minority_classes=minority, captures=train.captures)
    if problems:
        raise StageError("LOPO folds invalid: " + "; ".join(problems))
    ctx.save_json(FOLDS_NAME, folds)
    for f in folds["folds"]:
        comp = ", ".join(f"{n} {v}" for n, v in f["companions"].items())
        ctx.log(f"fold {f['fold']}: DCIS {f['dcis_patient']} + {comp} -> {f['heldout_rows']} held-out rows "
                f"{f['heldout_rows_per_class']}")
    if not folds["companions_disjoint"]:
        ctx.warn(f"companions reused across folds: {folds['reused_companions']} (no disjoint draw exists)")

    # 4. The split-access policy over every registered stage, then the ledger.
    from hsi_augment.registry import load_builtin_stages
    registry = load_builtin_stages()
    violations = registry_policy_violations(registry)
    if violations:
        raise StageError("split-access policy violated: " + "; ".join(violations))
    ledger = build_ledger(pass1=pass1, folds_doc=folds, readers=held_out_readers(registry))
    bad = ledger_problems(ledger)
    if bad:                                      # a bug in LEDGER_TABLE, not in the data
        raise StageError("leakage ledger malformed: " + "; ".join(bad))
    ctx.save_json(LEDGER_NAME, ledger)
    statuses = dict(Counter(e["status"] for e in ledger["entries"]))
    ctx.log(f"ledger: {len(ledger['entries'])} entries {statuses}")

    return {
        "patients": report["patients"],
        "patients_per_class": report["patients_per_class"],
        "captures": report["captures"],
        "n_folds": folds["n_folds"],
        "dcis_patients": folds["dcis_patients"],
        "companions_disjoint": folds["companions_disjoint"],
        "ledger_statuses": statuses,
        "warnings": [c["check"] for c in failed_checks(report["checks"], "warning")],
    }
