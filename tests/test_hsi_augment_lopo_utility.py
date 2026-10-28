# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_lopo_utility.py
======================================
The LOPO utility test (#30; Plan §Phase 8c): the stage `lopo-utility` on the
tiny fixture, through the stage API, and its pieces on constructed inputs.

The chain is real and is the one `tests/test_hsi_augment_export.py` builds:
tissue-mask -> leakage -> pool, the CVAE (cvae-lopo, cvae-final) and the
AE/LDM (ae-lopo, ae-final, ldm-lopo, ldm-final), all tiny models fitted for 6
steps, then qc-reference and generate-cvae / generate-ldm, with tau_s / w
chosen on LOPO QC. The fixture has 2 LOPO folds, and both generators have a
fit in each. Checked:
  * decision.json follows its schema (`check_decision`), and `a4_source` is
    what the rule gives on the per-fold metrics it records;
  * the readers parse it: hsi_augment/report.py's (via
    scripts/augmentation_select.read_decision), and the sweep's `a4_source()`,
    `A4_SOURCE` and its `lopo-utility` job;
  * the count-matched copies equal the synthetic rows each generator adds,
    in every fold;
  * the stage reads train only (the SplitGuard access log), and its declared
    purposes could not read val or test anyway;
  * a re-run is skipped, and a forced re-run writes the same bytes;
  * the decision rule, its tie-break, the copy draw and the metrics, on
    constructed inputs.

As in the export test, the 6-step generators make flat patches that the
calibrated rules reject (`test_real_qc_rejects_the_tiny_generators_loudly`).
So the happy path runs under `oracle_qc`: the REAL `qc.apply_rules` computes
every score, and only its bitmask is replaced, by a deterministic function of
each patch's bytes.

CPU only; every model and array is tiny.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import run_experiments as rx
from hsi_augment import lopo_utility as L
from hsi_augment.io import ACCESS_LOG_NAME, SplitAccessError, SplitGuard, read_access_log
from hsi_augment.pool_io import load_pool, pool_fingerprint
from hsi_augment.registry import PLANNED_COMMANDS, load_builtin_stages
from hsi_augment.splits import fold_masks, held_out_readers, registry_policy_violations
from hsi_augment.stage import DONE_NAME, FAILED_NAME, RunOptions, StageError, run_stage
from hsi_augment.stages.fit_ae import AEFinalConfig, AELopoConfig
from hsi_augment.stages.fit_cvae import CVAEFinalConfig, CVAELopoConfig
from hsi_augment.stages.fit_ldm import LDMFinalConfig, LDMLopoConfig
from hsi_augment.stages.generate import GenerateCVAEConfig, GenerateLDMConfig
from hsi_augment.stages.leakage import LeakageConfig
from hsi_augment.stages.lopo_utility import LopoUtilityConfig
from hsi_augment.stages.pool import PoolConfig
from hsi_augment.stages.qc_reference import QCReferenceConfig
from hsi_augment.stages.tissue_mask import TissueMaskConfig
from tests.hsi_augment_fixtures import make_tiny_build
from tests.test_hsi_augment_export import ACCEPT, AE_LOPO, CVAE_LOPO, GEN, LDM_LOPO, LOOP, oracle_qc

DCIS = 1
NAMES = ("healthy", "DCIS", "IDC")
STAGE = "lopo-utility"
CFG = dict(device="cpu", qc_chunk_rows=32, chunk_rows=64)
REQUIRES = {"leakage", "pool", "qc-reference", "cvae-lopo", "cvae-final", "ldm-lopo", "generate-cvae", "generate-ldm"}
SWEEP = Path(__file__).resolve().parent.parent / "sweeps" / "augmentation_pipeline.py"
SWEEP_OUT = "data/hsi_v9-trainsel-aug"          # the sweep's default AUG_OUT


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(registry, tiny, out: Path, name: str, cfg, **kw):
    return run_stage(registry.get(name), cfg, RunOptions(data_dir=str(tiny.data_dir), out=str(out), **kw), registry)


# ============================================================================
# fixtures
# ============================================================================

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def registry():
    return load_builtin_stages()


@pytest.fixture(scope="module")
def chain(tiny, registry, tmp_path_factory) -> Path:
    """<out> with every upstream of lopo-utility done (generate-* under the oracle)."""
    out = tmp_path_factory.mktemp("aug")
    for name, cfg in (("tissue-mask", TissueMaskConfig()), ("leakage", LeakageConfig()),
                      ("pool", PoolConfig(rows_per_class=30)),
                      ("cvae-lopo", CVAELopoConfig(**LOOP, **CVAE_LOPO)),
                      ("cvae-final", CVAEFinalConfig(enforce_gates=False, device="cpu", prior_samples=16)),
                      ("ae-lopo", AELopoConfig(**LOOP, **AE_LOPO)),
                      ("ae-final", AEFinalConfig(device="cpu", metric_rows=20)),
                      ("ldm-lopo", LDMLopoConfig(**LOOP, **LDM_LOPO)), ("ldm-final", LDMFinalConfig(device="cpu")),
                      ("qc-reference", QCReferenceConfig(device="cpu"))):
        assert run(registry, tiny, out, name, cfg).status == "ran", name
    with oracle_qc(ACCEPT):
        assert run(registry, tiny, out, "generate-cvae", GenerateCVAEConfig(**GEN)).status == "ran"
        assert run(registry, tiny, out, "generate-ldm", GenerateLDMConfig(**GEN)).status == "ran"
    return out


@pytest.fixture(scope="module")
def ran(tiny, registry, chain):
    """lopo-utility on the chain; the access log as it was before the run."""
    log_before = read_access_log(chain / ACCESS_LOG_NAME)
    with oracle_qc(ACCEPT):
        result = run(registry, tiny, chain, STAGE, LopoUtilityConfig(**CFG))
    doc = json.loads((chain / "lopo_utility" / L.DECISION_NAME).read_text())
    return {"out": chain, "result": result, "doc": doc, "log_before": log_before}


@pytest.fixture
def chain_copy(chain, ran, tmp_path) -> Path:
    dst = tmp_path / "aug"
    shutil.copytree(chain, dst)
    return dst


# ============================================================================
# registration
# ============================================================================

def test_stage_is_registered_train_only_and_out_of_all(registry):
    from hsi_augment import cli
    s = registry.get(STAGE)
    assert (s.dirname, s.group, s.purposes, set(s.requires)) == ("lopo_utility", None, ("lopo",), REQUIRES)
    assert not s.in_all and not s.reads_held_out()
    assert STAGE not in {x.name for x in registry.resolve(["all"])}
    assert [x.name for x in registry.resolve([STAGE])] == [STAGE]
    # registering the stage retires the planned command; the plan's list keeps it
    assert STAGE in PLANNED_COMMANDS and STAGE not in registry.planned_only()
    assert STAGE not in held_out_readers(registry) and registry_policy_violations(registry) == []
    helptext = cli.build_stage_parser(registry, [STAGE]).format_help()
    for flag in ("--syn_ratio", "--max_added_rows", "--tie_tolerance", "--overgenerate", "--min_acceptance"):
        assert flag in helptext


def test_its_purposes_cannot_open_val_or_test(tiny, registry, tmp_path):
    from hsi_augment.adapters import get_adapter
    s = registry.get(STAGE)
    guard = SplitGuard(get_adapter("histologyhsi_bc", tiny.data_dir), tmp_path / "log.jsonl", stage=STAGE,
                       allowed_purposes=s.purposes)
    assert guard.open("train", "lopo", kinds=("y",)).y is not None
    for split in ("val", "test"):
        with pytest.raises(SplitAccessError, match="reserved"):
            guard.open(split, "lopo", kinds=("y",))
        for purpose in ("structural", "final_eval"):
            with pytest.raises(SplitAccessError, match="did not declare"):
                guard.open(split, purpose, kinds=("y",))


# ============================================================================
# the pieces, on constructed inputs
# ============================================================================

def test_the_decision_rule_and_its_tie_break():
    assert L.decide({"cvae": 0.50, "ldm": 0.61})["a4_source"] == "ldm"
    d = L.decide({"cvae": 0.70, "ldm": 0.61})
    assert d["a4_source"] == "cvae" and not d["tie"] and d["ldm_minus_cvae"] == pytest.approx(-0.09)
    # an exact tie goes to cvae, the cheaper generator, whatever the order of the keys
    d = L.decide({"ldm": 0.6, "cvae": 0.6})
    assert d["a4_source"] == "cvae" and d["tie"] and d["contenders"] == ["cvae", "ldm"]
    # within the tolerance is a tie too; just outside it is not
    assert L.decide({"cvae": 0.59, "ldm": 0.60}, 0.02)["a4_source"] == "cvae"
    assert L.decide({"cvae": 0.57, "ldm": 0.60}, 0.02)["a4_source"] == "ldm"
    assert L.TIE_ORDER[0] == "cvae" and set(L.TIE_ORDER) == set(L.SOURCES)
    for bad in ({"cvae": 0.5}, {"cvae": 0.5, "ldm": None}, {"cvae": float("nan"), "ldm": 0.5}):
        with pytest.raises(ValueError):
            L.decide(bad)
    with pytest.raises(ValueError):
        L.decide({"cvae": 0.5, "ldm": 0.5}, -0.1)


def _set(f1, recall=0.5, macro=0.6):
    return {"target_f1": f1, "target_recall": recall, "macro_f1": macro, "balanced_accuracy": macro,
            "targets": {"DCIS": {"f1": f1, "recall": recall}}}


def test_the_means_use_the_folds_every_set_was_scored_on():
    """ldm has no fit in fold 2 (ldm-lopo fits folds 0 and 1 by default): the
    decision compares the generators on folds 0 and 1 only."""
    per_fold = [
        {"fold": 0, "sets": {"real": _set(0.40), "copies": _set(0.42), "cvae": _set(0.50), "ldm": _set(0.60)}},
        {"fold": 1, "sets": {"real": _set(0.20), "copies": _set(0.22), "cvae": _set(0.30), "ldm": _set(0.30)}},
        {"fold": 2, "sets": {"real": _set(0.90), "copies": _set(0.90), "cvae": _set(1.00)}},
    ]
    s = L.summarize(per_fold, ["DCIS"])
    assert s["decision_folds"] == [0, 1]
    assert s["mean"]["cvae"]["target_f1"] == pytest.approx(0.40)
    assert s["mean"]["ldm"]["target_f1"] == pytest.approx(0.45)
    assert s["mean"]["real"]["folds"] == [0, 1] and s["mean_all"]["real"]["folds"] == [0, 1, 2]
    assert s["mean_all"]["cvae"]["target_f1"] == pytest.approx(0.60) and s["mean_all"]["ldm"]["n_folds"] == 2
    assert s["mean"]["ldm"]["per_target"]["DCIS"]["f1"] == pytest.approx(0.45)
    assert L.decide({g: s["mean"][g]["target_f1"] for g in L.SOURCES})["a4_source"] == "ldm"
    # on every fold cvae would look better (0.60): the rule must not mix fold sets
    c = L.compare(per_fold, "cvae", "copies", s["decision_folds"])
    assert c["per_fold"] == {"0": pytest.approx(0.08), "1": pytest.approx(0.08)} and c["better"] == 2 and c["beats"]
    assert L.compare(per_fold, "ldm", "cvae", [0, 1])["equal"] == 1


def test_copies_are_count_matched_and_as_even_as_possible():
    rng = np.random.default_rng
    assert np.array_equal(L.copy_positions(7, 7, rng(0)), np.arange(7))           # 1:1 = every row once
    for n_source, n_add in ((7, 17), (7, 3), (5, 1), (1, 4), (40, 40), (13, 100)):
        pos = L.copy_positions(n_source, n_add, rng(1))
        counts = np.bincount(pos, minlength=n_source)
        q = n_add // n_source
        assert len(pos) == n_add and set(np.unique(counts)) <= {q, q + 1}
        assert int((counts == q + 1).sum()) == n_add % n_source
        assert np.array_equal(pos, L.copy_positions(n_source, n_add, rng(1)))        # seeded
    assert len(L.copy_positions(5, 0, rng(0))) == 0
    with pytest.raises(ValueError):
        L.copy_positions(0, 3, rng(0))
    # n_add: round(ratio x n_train), at least 1, capped
    assert [L.added_rows(n, 1.0) for n in (0, 1, 15, 122_850)] == [0, 1, 15, 122_850]
    assert (L.added_rows(15, 0.5), L.added_rows(15, 0.01), L.added_rows(15, 2.0, cap=20)) == (8, 1, 20)


def test_heldout_metrics_match_sklearn():
    from sklearn.metrics import f1_score, precision_score, recall_score
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 300)
    p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 3, 300))
    s = L.score_heldout(y, p, NAMES, [DCIS])
    f1 = f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    assert s["targets"]["DCIS"]["f1"] == pytest.approx(f1[1]) == s["target_f1"]
    assert s["target_recall"] == pytest.approx(recall_score(y, p, labels=[1], average="macro"))
    assert s["targets"]["DCIS"]["precision"] == pytest.approx(precision_score(y, p, labels=[1], average="macro"))
    assert s["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert np.array_equal(np.array(s["confusion"]).sum(axis=1), np.bincount(y, minlength=3))
    # a class absent from the held-out rows has no F1 and is not in the macro-F1
    y2, p2 = np.array([0, 0, 1, 1]), np.array([0, 2, 1, 0])
    s2 = L.score_heldout(y2, p2, NAMES, [DCIS])
    assert s2["per_class"]["IDC"]["f1"] is None and s2["macro_f1_classes"] == ["healthy", "DCIS"]
    assert s2["macro_f1"] == pytest.approx(f1_score(y2, p2, labels=[0, 1], average="macro"))
    assert s2["target_recall"] == 0.5 and s2["targets"]["DCIS"]["f1"] == pytest.approx(2 / 3)


# ============================================================================
# end to end on the fixture
# ============================================================================

def test_end_to_end_writes_a_valid_decision(ran, chain):
    r, doc = ran["result"], ran["doc"]
    assert r.status == "ran" and L.check_decision(doc) == []
    assert doc["a4_source"] in ("cvae", "ldm") and doc["schema_version"] == L.DECISION_SCHEMA_VERSION
    assert r.done["summary"]["a4_source"] == doc["a4_source"]
    folds = json.loads((chain / "prep" / "leakage" / "folds.json").read_text())["folds"]
    assert doc["decision_folds"] == [0, 1] == [f["fold"] for f in doc["per_fold"]] and doc["n_folds"] == len(folds)
    assert doc["target_classes"] == ["DCIS"] and doc["training_sets"] == ["real", "copies", "cvae", "ldm"]
    # per fold and mean: DCIS recall and F1, fold macro-F1, for all four sets
    for f, want in zip(doc["per_fold"], folds):
        assert f["heldout_patients"] == sorted(want["heldout_patients"]) and f["dcis_patient"] == want["dcis_patient"]
        assert set(f["sets"]) == set(L.TRAINING_SETS)
        for rec in f["sets"].values():
            dc = rec["targets"]["DCIS"]
            assert 0 <= dc["recall"] <= 1 and 0 <= dc["f1"] <= 1 and 0 <= rec["macro_f1"] <= 1
            assert rec["target_f1"] == dc["f1"] and np.array(rec["confusion"]).sum() == f["n_heldout"]
    for s in L.TRAINING_SETS:
        m = doc["mean"][s]
        assert m["folds"] == [0, 1]
        assert m["target_f1"] == pytest.approx(np.mean([f["sets"][s]["target_f1"] for f in doc["per_fold"]]))
        assert m["macro_f1"] == pytest.approx(np.mean([f["sets"][s]["macro_f1"] for f in doc["per_fold"]]))
    # the rule, re-applied to the recorded metrics, gives the recorded decision
    again = L.decide({s: doc["mean"][s]["target_f1"] for s in L.SOURCES}, doc["tie_tolerance"])
    assert again == doc["decision"] and again["a4_source"] == doc["a4_source"]
    assert doc["rule"] == L.DECISION_RULE and doc["tie_break"] == L.TIE_BREAK and "cvae" in doc["tie_break"]
    assert doc["hybridsn"]["run"] is False


def test_decision_records_its_provenance(ran, chain):
    doc = ran["doc"]
    prov = doc["provenance"]
    assert prov["folds_sha256"] == sha(chain / "prep" / "leakage" / "folds.json")
    assert prov["pool_fingerprint"] == pool_fingerprint(chain / "pool")
    assert prov["thresholds_sha256"] == sha(chain / "qc" / "reference" / "thresholds.json")
    done = {n: json.loads((chain / d / DONE_NAME).read_text()) for n, d in
            (("pool", "pool"), ("qc-reference", "qc/reference"), ("leakage", "prep/leakage"))}
    assert prov["pool_stage_fingerprint"] == done["pool"]["fingerprint"]
    assert prov["qc_reference_fingerprint"] == done["qc-reference"]["fingerprint"]
    assert set(prov["upstream"]) == REQUIRES and prov["upstream"] == ran["result"].done["upstream"]
    for src in L.SOURCES:
        g = doc["generators"][src]
        choice = json.loads((chain / "synth" / src / "choice.json").read_text())
        assert g["value"] == choice["value"] and g["choice_source"] == choice["source"] == "lopo_qc"
        assert g["choice_sha256"] == sha(chain / "synth" / src / "choice.json")
        assert g["param"] == ("tau_s" if src == "cvae" else "guidance") and set(g["folds"]) == {"0", "1"}
        for k, fr in g["folds"].items():
            assert fr["checkpoint_sha256"] == sha(chain / fr["checkpoint"])
            assert fr["pool_sha256"] == prov["pool_fingerprint"]
            assert fr["heldout_patients"] == doc["per_fold"][int(k)]["heldout_patients"]
            syn = doc["per_fold"][int(k)]["synthetic"][src]["DCIS"]
            assert syn["qc_fold"] == int(k) and syn["n_selected"] == syn["n_target"]
    assert not any(str(v).startswith("/") for v in _strings(doc)), "decision.json holds an absolute path"


def _strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def test_count_matched_copies_equal_the_synthetic_count(ran, chain):
    doc = ran["doc"]
    pool = load_pool(chain / "pool")
    folds = json.loads((chain / "prep" / "leakage" / "folds.json").read_text())["folds"]
    for f, fold in zip(doc["per_fold"], folds):
        tr, ho = fold_masks(pool.patient, fold)
        n_dcis = int((pool.y[tr] == DCIS).sum())
        n_add = f["n_added"]["DCIS"]
        assert n_add == L.added_rows(n_dcis, 1.0) == n_dcis > 0                        # 1:1 by default
        assert f["copies"]["DCIS"] == {"n_source_rows": n_dcis, "n_added": n_add, "repeats": 1, "extra_draw": 0}
        for src in L.SOURCES:
            syn = f["synthetic"][src]["DCIS"]
            assert syn["n_selected"] == n_add and syn["n_accepted"] >= n_add and syn["n_generated"] <= 2 * n_add
        sets = f["sets"]
        assert f["n_train_real"] == int(tr.sum()) == sets["real"]["n_train"]
        assert f["n_heldout"] == int(ho.sum())
        for s in ("copies", "cvae", "ldm"):
            assert sets[s]["n_train"] == f["n_train_real"] + n_add
            assert sets[s]["train_rows_per_class"] == {**sets["real"]["train_rows_per_class"],
                                                       "DCIS": sets["real"]["train_rows_per_class"]["DCIS"] + n_add}


def test_report_and_sweep_read_the_decision(ran, chain, tmp_path, monkeypatch):
    from hsi_augment import report as R
    from scripts import augmentation_select as S
    a4 = ran["doc"]["a4_source"]
    # hsi_augment/report.py's reader (the first string among DECISION_KEYS)
    assert R.DECISION_JSON == "lopo_utility/" + L.DECISION_NAME and R.DECISION_KEYS == L.READER_KEYS
    assert S.read_decision(R.Inputs({"aug": chain}, reveal_test=False), chain) == a4
    # its cost row: a "lopo" stage (the report's LOPO wall time), charged to no generator
    assert R._stage_kind(STAGE) == "lopo" and R._stage_source(STAGE, "lopo_utility") is None
    # the sweep, loaded in a repo root holding this decision.json and done.json
    root = tmp_path / "repo"
    dst = root / SWEEP_OUT / "lopo_utility"
    dst.mkdir(parents=True)
    for name in (L.DECISION_NAME, DONE_NAME):
        shutil.copyfile(chain / "lopo_utility" / name, dst / name)
    monkeypatch.setattr(rx, "ROOT", root)
    for k in ("AUG_OUT", "AUG_SELECTION", "AUG_TAU", "AUG_PHYS"):
        monkeypatch.delenv(k, raising=False)
    sweep = rx.load_sweep(str(SWEEP))
    assert sweep["A4_SOURCE"] == sweep["a4_source"]() == a4
    (job,) = sweep["STAGES"]["lopo-utility"]
    assert job.requires() is None and job.done()          # registered, so no longer a planned command
    a4_arm = sweep["ARMS"]["A4"]
    assert sweep["source"](a4_arm) == a4 and a4_arm["flags"]["synthetic_dir"] == f"{SWEEP_OUT}/export/{a4}"


def test_the_stage_reads_train_only(ran, chain):
    before = len(ran["log_before"])
    new = read_access_log(chain / ACCESS_LOG_NAME)[before:]
    assert new and {r["stage"] for r in new} == {STAGE}                 # the QC reference opened nothing itself
    assert all(r["allowed"] and r["split"] == "train" and r["purpose"] == "lopo" for r in new)


def test_a_rerun_is_skipped_and_a_forced_one_is_identical(ran, registry, tiny, chain_copy):
    path = chain_copy / "lopo_utility" / L.DECISION_NAME
    before = sha(path)
    with oracle_qc(ACCEPT):
        assert run(registry, tiny, chain_copy, STAGE, LopoUtilityConfig(**CFG)).status == "skipped"
        r = run(registry, tiny, chain_copy, STAGE, LopoUtilityConfig(**CFG), force=True)
    assert r.status == "ran" and sha(path) == before
    assert r.done["outputs"][L.DECISION_NAME]["sha256"] == before


def test_a_changed_config_reruns_and_the_cap_shrinks_every_added_set(ran, registry, tiny, chain_copy):
    with oracle_qc(ACCEPT):
        r = run(registry, tiny, chain_copy, STAGE, LopoUtilityConfig(**CFG, max_added_rows=5))
    assert r.status == "ran"
    doc = json.loads((chain_copy / "lopo_utility" / L.DECISION_NAME).read_text())
    assert L.check_decision(doc) == [] and doc["max_added_rows"] == 5
    for f in doc["per_fold"]:
        assert f["n_added"]["DCIS"] == 5 and f["copies"]["DCIS"]["n_added"] == 5
        assert {f["synthetic"][s]["DCIS"]["n_selected"] for s in L.SOURCES} == {5}
        assert all(f["sets"][s]["n_train"] == f["n_train_real"] + 5 for s in ("copies", "cvae", "ldm"))


def test_real_qc_rejects_the_tiny_generators_loudly(ran, registry, tiny, chain_copy):
    """No oracle: the calibrated rules reject the 6-step generators' flat
    patches. The stage fails, and leaves neither done.json nor decision.json
    (the sweep would otherwise read a stale decision)."""
    d = chain_copy / "lopo_utility"
    assert (d / L.DECISION_NAME).is_file()
    with pytest.raises(StageError, match="below the 40% floor"):
        run(registry, tiny, chain_copy, STAGE, LopoUtilityConfig(**CFG), force=True)
    assert not (d / DONE_NAME).exists() and not (d / L.DECISION_NAME).exists() and (d / FAILED_NAME).exists()


def test_check_decision_catches_schema_errors(ran):
    doc = ran["doc"]
    assert L.check_decision(doc) == []
    assert L.check_decision([]) and L.check_decision({})
    for change, msg in (({"a4_source": "gan"}, "a4_source"), ({"schema_version": 2}, "schema_version"),
                        ({"source": "ldm"}, "'source' is a string"), ({"decision_folds": []}, "decision_folds"),
                        ({"training_sets": ["real"]}, "training_sets")):
        problems = L.check_decision({**doc, **change})
        assert any(msg in p for p in problems), (change, problems)
    broken = json.loads(json.dumps(doc))
    del broken["per_fold"][0]["sets"]["ldm"]
    assert any("lacks sets ['ldm']" in p for p in L.check_decision(broken))
    del broken["provenance"]["folds_sha256"]
    assert any("provenance" in p for p in L.check_decision(broken))
