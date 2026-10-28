# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_splits.py
================================
Phase 3 (#14): split verification, the leakage ledger, the LOPO folds and the
split-access policy (`hsi_augment/splits.py`, stage `leakage`).

* Disjointness passes on the fixture and every constructed leak or
  inconsistency (a patient in train and val, a capture in two splits, a capture
  key indexed twice, a stale split assignment or Pass-1 scope) is a hard
  failure that leaves no done.json.
* The ledger has exactly the plan's quantities, known statuses, and fits on
  train only, except the classifier-selection row (val); test appears only as
  "final report only".
* Each DCIS train patient is held out exactly once; every held-out set holds
  all three classes and every row of its patients; no held-out patient's rows
  are in the fold's train mask; the draw is seeded and deterministic.
* The SplitGuard tests iterate the REGISTRY, so a stage registered later is
  covered automatically.

CPU only, on `tests/hsi_augment_fixtures.make_tiny_build`; leak variants are
built here, never in the shared fixture. The real-data smoke test runs only
with HSI_AUGMENT_REAL_DATA set, and reads JSON and the int sidecars, never X.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hsi_augment import splits
from hsi_augment.adapters import get_adapter
from hsi_augment.config import ConfigError
from hsi_augment.io import (HELD_OUT_PURPOSES, PURPOSE_SPLITS, SPLITS, SplitAccessError, SplitGuard,
                            read_access_log)
from hsi_augment.registry import Registry, load_builtin_stages
from hsi_augment.stage import DONE_NAME, FAILED_NAME, RunOptions, StageError, run_stage, stage_seed_sequence
from hsi_augment.stages.leakage import LeakageConfig
from tests.hsi_augment_fixtures import make_tiny_build

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = "prep/leakage"
CLASSES = {0, 1, 2}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def registry():
    return load_builtin_stages()


def _run(registry, data_dir, out, *, seed=42, **cfg):
    return run_stage(registry.get("leakage"), LeakageConfig(**cfg),
                     RunOptions(data_dir=str(data_dir), out=str(out), seed=seed), registry)


def _read(stage_dir: Path, name: str):
    return json.loads((stage_dir / name).read_text())


@pytest.fixture(scope="module")
def ran(tiny, registry, tmp_path_factory):
    """One run of the stage on the fixture, shared by the read-only tests."""
    out = tmp_path_factory.mktemp("aug")
    result = _run(registry, tiny.data_dir, out)
    d = out / STAGE_DIR
    return SimpleNamespace(out=out, dir=d, result=result, report=_read(d, splits.SPLIT_REPORT_NAME),
                           folds=_read(d, splits.FOLDS_NAME), ledger=_read(d, splits.LEDGER_NAME))


def _train_arrays(data_dir: Path):
    return (np.load(data_dir / "groups_train.npy"), np.load(data_dir / "y_train.npy"),
            np.load(data_dir / "captures_train.npy"))


def _verify(data_dir: Path):
    adapter = get_adapter("histologyhsi_bc", data_dir)
    labels = {s: splits.SplitLabels.from_arrays(s, *(np.load(data_dir / f"{k}_{s}.npy") for k in splits.LABEL_KINDS))
              for s in SPLITS}
    return splits.verify_splits(labels, adapter.capture_index, adapter.class_names, adapter.minority_classes,
                                assigned=adapter.split_patients())


def _failed(report) -> set:
    return {c["check"] for c in splits.failed_checks(report["checks"])}


# ============================================================================
# leak variants of the fixture (built here; the shared fixture is not edited)
# ============================================================================

def _edit_npy(path: Path, fn) -> None:
    a = np.load(path)
    fn(a)
    np.save(path, a)


def _edit_json(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj, indent=2))


def _patient_in_train_and_val(b):
    _edit_npy(b.data_dir / "groups_val.npy", lambda g: g.__setitem__(slice(0, 5), 101))


def _capture_in_train_and_val(b):
    _edit_npy(b.data_dir / "captures_val.npy", lambda c: c.__setitem__(slice(0, 5), b.captures["train"][0]))


def _capture_key_twice(b):
    def fn(ci):
        val = next(k for k, v in ci.items() if v["split"] == "validation")
        ci[val]["key"] = ci[str(b.captures["train"][0])]["key"]
    _edit_json(b.data_dir / "capture_index.json", fn)


def _stale_split_assignment(b):
    _edit_json(b.root / "split_assignment.json", lambda sa: sa["groups"].__setitem__("106", "validation"))


def _row_patient_not_capture_patient(b):
    caps = np.load(b.data_dir / "captures_train.npy")
    groups = np.load(b.data_dir / "groups_train.npy")
    cap = int(caps[groups == 103][0])
    _edit_npy(b.data_dir / "groups_train.npy", lambda g: g.__setitem__(caps == cap, 104))


def _label_out_of_range(b):
    _edit_npy(b.data_dir / "y_test.npy", lambda y: y.__setitem__(0, 7))


def _pass1_fitted_on_other_patients(b):
    _edit_json(b.root / "pass1_scope_report.json", lambda s: s["train_patients"].remove("106"))


LEAKS = {
    "patient_in_train_and_val": (_patient_in_train_and_val,
                                 {"patients_disjoint", "row_patient_matches_capture",
                                  "split_assignment_matches_rows"}),
    "capture_in_train_and_val": (_capture_in_train_and_val, {"captures_disjoint", "capture_split_matches_rows",
                                                             "row_patient_matches_capture"}),
    "capture_key_twice": (_capture_key_twice, {"capture_keys_unique"}),
    "stale_split_assignment": (_stale_split_assignment, {"split_assignment_matches_rows"}),
    "row_patient_not_capture_patient": (_row_patient_not_capture_patient, {"row_patient_matches_capture"}),
    "label_out_of_range": (_label_out_of_range, {"labels_in_range"}),
    "pass1_fitted_on_other_patients": (_pass1_fitted_on_other_patients, set()),   # caught by the stage
}


# ============================================================================
# split verification
# ============================================================================

def test_split_verification_passes_on_the_fixture(tiny):
    report = _verify(tiny.data_dir)
    assert report["ok"] and not _failed(report)
    assert report["patients"] == "6/1/1" and report["patients_per_class"]["DCIS"] == "2/1/1"
    assert report["captures"] == "/".join(str(len(tiny.captures[s])) for s in SPLITS)
    for s in SPLITS:
        sp = report["splits"][s]
        assert tuple(sp["patients"]) == tiny.patients[s]
        assert tuple(sp["patients_by_class"]["DCIS"]) == tiny.dcis_patients[s]
        assert sum(sp["rows_per_class"].values()) == sp["rows"] == tiny.n_rows[s]
    # patients with captures of several tissues: holding one out must remove all of them
    assert report["splits"]["train"]["multi_class_patients"] == [101, 102, 103, 104]


@pytest.mark.parametrize("leak", sorted(LEAKS))
def test_constructed_leaks_are_hard_failures(leak, registry, tmp_path):
    mutate, expected = LEAKS[leak]
    b = make_tiny_build(tmp_path / "build")
    mutate(b)
    assert _failed(_verify(b.data_dir)) == expected
    out = tmp_path / "aug"
    check = next(iter(sorted(expected)), "pass1_scope")
    with pytest.raises(StageError, match=check):
        _run(registry, b.data_dir, out)
    d = out / STAGE_DIR
    assert not (d / DONE_NAME).exists() and (d / FAILED_NAME).exists()
    assert not (d / splits.FOLDS_NAME).exists() and not (d / splits.LEDGER_NAME).exists()
    report = _read(d, splits.SPLIT_REPORT_NAME)      # written anyway: it says what broke
    assert report["ok"] is False
    if leak == "patient_in_train_and_val":
        overlap = next(c for c in report["checks"] if c["check"] == "patients_disjoint")["overlaps"]
        assert overlap == {"train&val": [101]}


def test_pass1_scope_check():
    patients = {"train": [1, 2, 3], "val": [4], "test": [5]}
    scope = {"train_patients": ["1", "2", "3"], "heldout_patients": ["4", "5"], "n_train_captures": 6}
    assert splits.check_pass1_scope(scope, patients, 6)["ok"]
    assert not splits.check_pass1_scope({**scope, "train_patients": ["1", "2", "3", "4"]}, patients, 6)["ok"]
    assert not splits.check_pass1_scope({**scope, "heldout_patients": ["4"]}, patients, 6)["ok"]
    assert not splits.check_pass1_scope(scope, patients, 7)["ok"]
    assert not splits.check_pass1_scope(None, patients, 6)["ok"]


# ============================================================================
# the leakage ledger
# ============================================================================

def test_ledger_has_no_unknown_entries(ran):
    entries = ran.ledger["entries"]
    assert [e["quantity"] for e in entries] == list(splits.LEDGER_QUANTITIES)
    assert splits.ledger_problems(ran.ledger) == []
    assert {e["status"] for e in entries} <= set(splits.LEDGER_STATUSES)
    status = {e["quantity"]: e["status"] for e in entries}
    assert status.pop(splits.BAND_SELECTION) == "verified" and status.pop(splits.LOPO_FOLDS) == "verified"
    assert status.pop(splits.TEST_QUANTITY) == "enforced"
    assert set(status.values()) == {"pending"}
    for e in entries:
        assert re.fullmatch(r"#\d+(, #\d+)*", e["issue"]) and e["evidence"] and e["stage"]
    band = next(e for e in entries if e["quantity"] == splits.BAND_SELECTION)
    assert "6 train patients" in band["evidence"]
    assert ran.ledger["split_access"]["held_out_readers"] == {"discovery": ["structural"], "index": ["structural"],
                                                              "leakage": ["structural"],
                                                              "report-tissue": ["final_eval"]}


def test_ledger_fits_on_train_only_except_classifier_selection(ran):
    for e in ran.ledger["entries"]:
        if e["quantity"] == splits.CLASSIFIER_SELECTION:
            assert e["fitted_on"] == e["fitted_on_split"] == "val"
        elif e["quantity"] == splits.TEST_QUANTITY:
            assert e["fitted_on"] == "final report only" and e["fitted_on_split"] is None
        else:
            assert e["fitted_on_split"] == "train"
            assert not re.search(r"\b(val|validation|test)\b", e["fitted_on"].lower()), e
    # tampering is caught: an unknown row, an unknown status, a quantity fitted on test
    bad = json.loads(json.dumps(ran.ledger))
    bad["entries"].append({**bad["entries"][0], "quantity": "Mystery threshold"})
    bad["entries"][1]["status"] = "unknown"
    bad["entries"][2]["fitted_on_split"] = "test"
    problems = splits.ledger_problems(bad)
    assert len(problems) == 3 and "Mystery threshold" in problems[0]


# ============================================================================
# LOPO folds
# ============================================================================

def test_each_dcis_train_patient_is_held_out_exactly_once(ran, tiny):
    f = ran.folds
    dcis = list(tiny.dcis_patients["train"])
    assert f["dcis_patients"] == dcis and f["n_folds"] == len(f["folds"]) == len(dcis)
    assert [x["fold"] for x in f["folds"]] == list(range(len(dcis)))
    assert [x["dcis_patient"] for x in f["folds"]] == dcis
    held = Counter(p for x in f["folds"] for p in x["heldout_patients"])
    assert all(held[p] == 1 for p in dcis)
    for x in f["folds"]:
        # one healthy and one IDC companion, neither a DCIS patient, both eligible
        assert {k: len(v) for k, v in x["companions"].items()} == {"healthy": 1, "IDC": 1}
        for cls, ps in x["companions"].items():
            assert set(ps) <= set(f["candidates"][cls]) and not set(ps) & set(dcis)
        assert sorted({x["dcis_patient"], *x["companions"]["healthy"], *x["companions"]["IDC"]}) == \
            x["heldout_patients"]
    # a disjoint draw exists on the fixture, so the matching finds one
    assert f["companions_disjoint"] and f["reused_companions"] == []
    assert f["seed"] == 42 and f["seed_stream"]["keys"] == [splits.FOLDS_RNG_KEY]
    assert "every tissue" in f["rule"]


def test_heldout_sets_have_all_classes_and_no_patient_leaks_into_train(ran, tiny):
    groups, y, caps = _train_arrays(tiny.data_dir)
    for f in ran.folds["folds"]:
        train_mask, held_mask = splits.fold_masks(groups, f)
        assert train_mask.dtype == bool and (train_mask ^ held_mask).all()
        assert set(np.unique(y[held_mask]).tolist()) == CLASSES
        assert set(np.unique(y[train_mask]).tolist()) == CLASSES
        assert not set(np.unique(groups[train_mask]).tolist()) & set(f["heldout_patients"])
        assert not np.intersect1d(caps[held_mask], caps[train_mask]).size
        for p in f["heldout_patients"]:              # every row of a held-out patient, every tissue
            assert held_mask[groups == p].all()
        assert f["heldout_rows"] == int(held_mask.sum())
        assert f["heldout_rows_per_class"] == {n: int((y[held_mask] == c).sum())
                                               for c, n in enumerate(tiny.class_names)}
    # patient 101 has DCIS and healthy captures: both leave with them
    f0 = ran.folds["folds"][0]
    assert f0["dcis_patient"] == 101 and f0["heldout_rows_per_class"]["healthy"] > 0
    assert splits.check_folds(ran.folds, groups, y, class_names=tiny.class_names, minority_classes=(1,),
                              captures=caps) == []


def test_fold_masks_api(tiny):
    groups, _, _ = _train_arrays(tiny.data_dir)
    t1, h1 = splits.fold_masks(groups, {"heldout_patients": [103, 101]})
    t2, h2 = splits.fold_masks(groups, (101, 103))
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(h1, np.isin(groups, [101, 103]))
    np.testing.assert_array_equal(t1, ~h1)
    t, h = splits.fold_masks(groups, [])
    assert t.all() and not h.any() and h.shape == groups.shape
    assert splits.fold_heldout_patients({"heldout_patients": [3, 1]}) == (1, 3)


def test_folds_are_deterministic_for_a_seed(ran, tiny, registry, tmp_path):
    again = tmp_path / "again"
    _run(registry, tiny.data_dir, again)
    assert (again / STAGE_DIR / splits.FOLDS_NAME).read_bytes() == (ran.dir / splits.FOLDS_NAME).read_bytes()

    groups, y, _ = _train_arrays(tiny.data_dir)

    def draw(seed):
        rng = np.random.default_rng(stage_seed_sequence(seed, "leakage", splits.FOLDS_RNG_KEY))
        doc = splits.make_lopo_folds(groups, y, class_names=tiny.class_names, minority_classes=(1,), rng=rng)
        return doc, [tuple(f["heldout_patients"]) for f in doc["folds"]]

    assert draw(42)[1] == [tuple(f["heldout_patients"]) for f in ran.folds["folds"]]
    assert draw(5)[0] == draw(5)[0]
    outcomes = {tuple(draw(s)[1]) for s in range(16)}
    assert len(outcomes) > 1                                       # it is a draw, not a fixed choice
    assert all(draw(s)[0]["companions_disjoint"] for s in range(16))


def _synthetic(patient_classes):
    """groups / y with 4 rows per (patient, class)."""
    g, y = [], []
    for p, classes in patient_classes.items():
        for c in classes:
            g += [p] * 4
            y += [c] * 4
    return np.array(g, dtype=np.int32), np.array(y, dtype=np.int64)


def test_the_matching_avoids_the_greedy_dead_end():
    # 2 folds need 2 healthy + 2 IDC companions out of A (healthy+IDC), B (healthy), C, D (IDC).
    # A greedy draw that spends A as an IDC companion has no second healthy patient left.
    groups, y = _synthetic({1: [1, 0, 2], 2: [1, 0, 2], 10: [0, 2], 11: [0], 12: [2], 13: [2]})
    for seed in range(40):
        doc = splits.make_lopo_folds(groups, y, class_names=("healthy", "DCIS", "IDC"), minority_classes=(1,),
                                     rng=np.random.default_rng(seed))
        assert doc["companions_disjoint"], seed
        assert sorted(p for f in doc["folds"] for p in f["companions"]["healthy"]) == [10, 11]


def test_companions_are_reused_only_when_no_disjoint_draw_exists():
    # 3 folds but only 2 healthy candidates: one healthy patient must serve twice, least used first
    groups, y = _synthetic({1: [1, 0, 2], 2: [1, 0, 2], 3: [1, 0, 2], 10: [0], 11: [0], 12: [2], 13: [2], 14: [2]})
    for seed in range(10):
        doc = splits.make_lopo_folds(groups, y, class_names=("healthy", "DCIS", "IDC"), minority_classes=(1,),
                                     rng=np.random.default_rng(seed))
        healthy = Counter(p for f in doc["folds"] for p in f["companions"]["healthy"])
        idc = [p for f in doc["folds"] for p in f["companions"]["IDC"]]
        assert not doc["companions_disjoint"] and len(doc["reused_companions"]) == 1
        assert set(healthy) == {10, 11} and sorted(healthy.values()) == [1, 2]
        assert sorted(idc) == [12, 13, 14]
        assert splits.check_folds(doc, groups, y, class_names=("healthy", "DCIS", "IDC"),
                                  minority_classes=(1,)) == []


def test_fold_preconditions():
    names = ("healthy", "DCIS", "IDC")
    groups, y = _synthetic({1: [1, 0, 2], 10: [0], 12: [2]})
    with pytest.raises(StageError, match="at least 2"):
        splits.make_lopo_folds(groups, y, class_names=names, minority_classes=(1,), rng=np.random.default_rng(0))
    groups, y = _synthetic({1: [1, 2], 2: [1, 2], 12: [2], 13: [2]})
    with pytest.raises(StageError, match="healthy rows"):
        splits.make_lopo_folds(groups, y, class_names=names, minority_classes=(1,), rng=np.random.default_rng(0))
    with pytest.raises(ConfigError):
        LeakageConfig(companions_per_class=0)


def test_a_fold_whose_train_side_lacks_a_class_fails(tiny, registry, tmp_path):
    # 2 companions per class need all four non-DCIS fixture patients in each fold,
    # so fold 0's train side is patient 102 alone, who has no healthy rows.
    with pytest.raises(StageError, match="train side has no"):
        _run(registry, tiny.data_dir, tmp_path / "aug", companions_per_class=2)
    assert not (tmp_path / "aug" / STAGE_DIR / DONE_NAME).exists()


def test_check_folds_catches_bad_folds(ran, tiny):
    groups, y, caps = _train_arrays(tiny.data_dir)

    def problems(edit):
        doc = json.loads(json.dumps(ran.folds))
        edit(doc)
        return " | ".join(splits.check_folds(doc, groups, y, class_names=tiny.class_names, minority_classes=(1,),
                                             captures=caps))

    assert problems(lambda d: None) == ""
    assert "held out 2 time(s)" in problems(lambda d: d["folds"][1]["heldout_patients"].append(101))
    assert "no train rows" in problems(lambda d: d["folds"][0]["heldout_patients"].append(999))
    assert "train rows, this train split" in problems(lambda d: d.__setitem__("n_train_rows", 7))
    assert "another build" in problems(lambda d: d["folds"][0]["heldout_rows_per_class"].__setitem__("DCIS", 1))
    assert "not 0.." in problems(lambda d: d["folds"].reverse())
    assert "no ['DCIS'] rows" in problems(lambda d: d["folds"][0]["heldout_patients"].remove(101))


def test_load_folds(ran, tmp_path):
    assert splits.load_folds(ran.dir) == splits.load_folds(ran.dir / splits.FOLDS_NAME) == ran.folds
    bad = tmp_path / "folds.json"
    bad.write_text(json.dumps({**ran.folds, "schema_version": 99}))
    with pytest.raises(ValueError, match="schema_version"):
        splits.load_folds(bad)


# ============================================================================
# SplitGuard and the test seal, over every registered stage
# ============================================================================

def test_no_registered_stage_reads_test_before_phase_10(registry):
    spec = registry.get("leakage")
    assert (spec.group, spec.dirname, spec.purposes) == ("preprocess", STAGE_DIR, ("structural",))
    assert splits.FINAL_EVAL_STAGES == frozenset({"report-tissue"})       # Phase 10 (#34)
    for s in registry.specs():
        assert "final_eval" not in s.purposes or s.name in splits.FINAL_EVAL_STAGES, s.name
    assert splits.registry_policy_violations(registry) == []


def test_every_registered_stage_reads_val_and_test_only_structurally(registry, tiny, tmp_path):
    adapter = get_adapter("histologyhsi_bc", tiny.data_dir)
    log = tmp_path / "split_access_log.jsonl"
    attempts = refused = 0
    for spec in registry.specs():
        guard = SplitGuard(adapter, log, stage=spec.name, allowed_purposes=spec.purposes)
        for purpose in spec.purposes:
            for split in SPLITS:
                attempts += 1
                if split == "train" or purpose in HELD_OUT_PURPOSES:
                    guard.open(split, purpose, kinds=("y",))
                else:
                    refused += 1
                    with pytest.raises(SplitAccessError, match="reserved"):
                        guard.open(split, purpose, kinds=("y",))
        if set(spec.purposes) & set(splits.GENERATOR_SIDE_PURPOSES):
            assert not spec.reads_held_out(), spec.name
    rec = read_access_log(log)
    assert len(rec) == attempts and sum(not r["allowed"] for r in rec) == refused
    assert all(r["reason"] for r in rec if not r["allowed"])


def test_generator_qc_pool_and_lopo_purposes_refuse_val_and_test(tiny, tmp_path):
    adapter = get_adapter("histologyhsi_bc", tiny.data_dir)
    log = tmp_path / "split_access_log.jsonl"
    for purpose in splits.GENERATOR_SIDE_PURPOSES:
        assert PURPOSE_SPLITS[purpose] == ("train",)
        guard = SplitGuard(adapter, log, stage=f"{purpose}-stage", allowed_purposes=(purpose,))
        for split in ("val", "validation", "test"):
            with pytest.raises(SplitAccessError, match="reserved"):
                guard.open(split, purpose, kinds=("y",))
        with pytest.raises(SplitAccessError, match="did not declare"):
            guard.open("test", "structural", kinds=("y",))
        assert guard.open("train", purpose, kinds=("y",)).n == tiny.n_rows["train"]
    rec = read_access_log(log)
    assert len(rec) == 5 * len(splits.GENERATOR_SIDE_PURPOSES)
    assert {(r["split"], r["allowed"]) for r in rec} == {("val", False), ("test", False), ("train", True)}


def test_registry_policy_flags_bad_registrations(registry, tiny, tmp_path, monkeypatch):
    bad = Registry()
    noop = lambda ctx, cfg: None  # noqa: E731
    bad.stage("peek-test", config=LeakageConfig, dirname="prep/peek", purposes=("final_eval",))(noop)
    bad.stage("fit-gen", config=LeakageConfig, dirname="gen/x", purposes=("generator", "structural"))(noop)
    bad.stage("qc-count", config=LeakageConfig, dirname="qc", purposes=("structural",))(noop)
    bad.stage("fine", config=LeakageConfig, dirname="prep/fine", purposes=("structural",))(noop)
    bad.stage("fit-ok", config=LeakageConfig, dirname="gen/y", purposes=("generator",))(noop)
    v = splits.registry_policy_violations(bad)
    assert len(v) == 4
    assert any("'peek-test' declares 'final_eval'" in m for m in v)
    assert any("'fit-gen' declares ['generator'] and also ['structural']" in m for m in v)
    assert sum("is not under 'prep/'" in m for m in v) == 2          # fit-gen, qc-count
    assert not any("'fine'" in m or "'fit-ok'" in m for m in v)
    assert set(splits.held_out_readers(bad)) == {"peek-test", "fit-gen", "qc-count", "fine"}

    # the stage itself refuses to write a ledger over a registry that breaks the policy
    import hsi_augment.registry as reg_mod
    monkeypatch.setattr(reg_mod, "load_builtin_stages", lambda: bad)
    out = tmp_path / "aug"
    with pytest.raises(StageError, match="split-access policy violated"):
        _run(registry, tiny.data_dir, out)
    assert not (out / STAGE_DIR / splits.LEDGER_NAME).exists() and not (out / STAGE_DIR / DONE_NAME).exists()


def test_the_stage_reads_only_int_sidecars_and_logs_every_access(ran):
    rec = read_access_log(ran.out / "split_access_log.jsonl")
    assert {(r["stage"], r["split"], r["purpose"], r["allowed"]) for r in rec} == \
        {("leakage", s, "structural", True) for s in SPLITS}
    files = {f for r in rec for f in r["files"]}
    assert files == {f"{k}_{s}.npy" for k in splits.LABEL_KINDS for s in SPLITS}
    assert not any(f.startswith("X_") for f in files)


# ============================================================================
# CLI
# ============================================================================

def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)


def test_cli_runs_the_leakage_stage(tiny, tmp_path):
    out = tmp_path / "aug"
    base = ["leakage", "--data_dir", tiny.data_dir, "--out", out]
    r = _cli(*base)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "leakage: ran" in r.stdout and "split ok: patients 6/1/1" in r.stdout and "fold 1: DCIS 102" in r.stdout
    done = _read(out / STAGE_DIR, DONE_NAME)
    assert set(done["outputs"]) == {splits.SPLIT_REPORT_NAME, splits.FOLDS_NAME, splits.LEDGER_NAME}
    assert done["summary"]["patients"] == "6/1/1" and done["summary"]["n_folds"] == 2
    assert done["summary"]["patients_per_class"]["DCIS"] == "2/1/1"
    r = _cli(*base)
    assert r.returncode == 0 and "leakage: skipped" in r.stdout
    r = _cli(*base, "--companions_per_class", "0")
    assert r.returncode == 2 and "companions_per_class" in r.stderr
    r = _cli("--help")
    assert r.returncode == 0 and re.search(r"leakage\s+P3: .*\[group preprocess\]", r.stdout)


# ============================================================================
# the real build (opt-in): JSON and the int sidecars only, never X
# ============================================================================

REAL_DATA = os.environ.get("HSI_AUGMENT_REAL_DATA")


@pytest.mark.skipif(not REAL_DATA, reason="set HSI_AUGMENT_REAL_DATA to the trainsel build's hsi dir")
def test_real_build_split_ledger_and_folds(registry, tmp_path):
    _run(registry, REAL_DATA, tmp_path)
    d = tmp_path / STAGE_DIR
    report, folds, ledger = (_read(d, n) for n in (splits.SPLIT_REPORT_NAME, splits.FOLDS_NAME, splits.LEDGER_NAME))
    assert report["patients"] == "35/5/5" and report["patients_per_class"]["DCIS"] == "5/1/1"
    assert folds["dcis_patients"] == [25, 45, 85, 107, 152] and folds["n_folds"] == 5
    assert folds["companions_disjoint"]
    for f in folds["folds"]:
        assert all(n > 0 for n in f["heldout_rows_per_class"].values())
        assert all(n > 0 for n in f["train_rows_per_class"].values())
    assert splits.ledger_problems(ledger) == []
    files = {f for r in read_access_log(tmp_path / "split_access_log.jsonl") for f in r["files"]}
    assert files and not any(f.startswith("X_") for f in files)
