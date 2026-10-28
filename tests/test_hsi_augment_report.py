# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_report.py
================================
Phase 10 (#34): `hsi_augment/stats.py`, `hsi_augment/report.py`, the
`report-tissue` stage and `scripts/augmentation_report.py`.

* The statistics: paired per-seed deltas and sign counts, the pre-registered
  minimum effect, the Wilcoxon reported only at 5/5, and the patient bootstrap
  with a FIXED class set that counts the resamples missing a class.
* The report, on synthetic run directories that copy the real formats
  (config.json with cli_args.run_tag, history.json as a list of epoch dicts,
  test_predictions.npz {true_label, predicted_label, probs}, test_report.json
  {sklearn_metrics, ...}, heldout_v19/validation/test_predictions.npz,
  per_patient_metrics.json, stage done.json files) with a KNOWN effect: A3
  fixes 40% of A0's errors on every seed, A1 helps on 2 seeds and hurts on 3.
  Every delta is recomputed here with sklearn and compared.
* The test seal: no test column, and no test file opened (open / np.load are
  patched to fail on `test_*` files), without selection.json or without
  --reveal_test; the screen reads config.json and history.json only; a code
  path pointed at a test file by mistake is refused by the reading layer
  (train.py always writes test_predictions.npz), and `report-tissue` is the
  only registered `final_eval` stage and refuses test before selection.json.
* Byte-identical regeneration; the `report-tissue` stage on the tiny build;
  the CLI.

CPU only, seconds.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from hsi_augment import report as R
from hsi_augment import splits, stats
from hsi_augment.registry import load_builtin_stages
from hsi_augment.stage import DONE_NAME, RunOptions, StageError, run_stage
from hsi_augment.stages.report_tissue import ReportTissueConfig
from hsi_augment.stages.tissue_mask import TissueMaskConfig
from tests.hsi_augment_fixtures import make_tiny_build

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["healthy", "DCIS", "IDC"]
SEEDS = (1, 7, 13, 23, 42)
DATE = "2026-11-02"
RUN_DATA = "data/hsi_v9-trainsel/hsi"
# patient -> classes it holds (one DCIS patient per held-out split, as in the real build)
VAL_PATIENTS = {65: (0, 2), 68: (0, 2), 197: (0, 1, 2), 238: (2,), 250: (0,)}
TEST_PATIENTS = {136: (0, 1, 2), 140: (0, 2), 150: (2,), 160: (0,), 170: (0, 2)}
ROWS_PER_CLASS = 60
SCREEN_SCORES = {"A0": 0.80, "A0os": 0.81, "A1": 0.83, "A2": 0.82, "A3": 0.85, "A4": 0.847}
SELECTED = ["A3", "A1"]
A1_HELPS = {1, 7}                       # A1 improves on these seeds only


# ============================================================================
# the synthetic study
# ============================================================================

def _wj(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _split(patients):
    y, g = [], []
    for pid, classes in patients.items():
        for c in classes:
            y += [c] * ROWS_PER_CLASS
            g += [pid] * ROWS_PER_CLASS
    return np.asarray(y, dtype=np.int64), np.asarray(g, dtype=np.int32)


def _a0_pred(y, seed, split):
    """A0: 25% of rows wrong; a healthy row that is wrong is called DCIS."""
    rng = np.random.default_rng([seed, 0 if split == "val" else 1])
    pred = y.copy()
    wrong = rng.random(len(y)) < 0.25
    pred[wrong] = (y[wrong] + rng.integers(1, 3, wrong.sum())) % 3
    pred[wrong & (y == 0)] = 1
    return pred


def _arm_pred(arm, y, seed, split):
    a0 = _a0_pred(y, seed, split)
    rng = np.random.default_rng([seed, 7, 0 if split == "val" else 1])
    if arm == "A0":
        return a0
    pred = a0.copy()
    if arm == "A3" or (arm == "A1" and seed in A1_HELPS):     # fix 40% of A0's errors: never worse per class
        fix = (a0 != y) & (rng.random(len(y)) < 0.4)
        pred[fix] = y[fix]
    else:                                                      # A1 elsewhere: break 10% of A0's correct rows
        brk = (a0 == y) & (rng.random(len(y)) < 0.10)
        pred[brk] = (y[brk] + 1) % 3
    return pred


def _probs(pred, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.2, (len(pred), 3))
    p[np.arange(len(pred)), pred] += 0.7
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def _history(n_epochs, score_epochs_value=None, base=0.6, dcis_f1=0.4):
    out = []
    for e in range(1, n_epochs + 1):
        f1 = score_epochs_value if (score_epochs_value is not None and e >= 10) else base + 0.01 * e
        out.append({"epoch": e, "f1_macro": f1, "balanced_accuracy": f1 + 0.02, "epoch_time": 300.0 + e,
                    "GPU_memory_MB": 840.0, "per_class_f1": [0.9, dcis_f1, 0.99],
                    "per_class_precision": [0.92, 0.3, 1.0], "per_class_recall": [0.88, 0.6, 0.98],
                    "per_class_support": [9692, 2423, 20870]})
    return out


def _config(tag, seed, epochs, arm):
    cfg = {"cli_args": {"run_tag": tag, "seed": seed, "epochs": epochs, "data_dir": RUN_DATA,
                        "trm_ema_rate": 0.9995, "profile": "paper_recipe", "batch_size": 256},
           "classes": 3, "class_names": CLASSES, "architecture": "recursive", "profile": "paper_recipe",
           "trainer": "TrainerG_v18Fast", "entry_point": "train.py",
           "train_subsample": {"frac": 0.102, "mode": "per_epoch", "kept": 250113, "total": 2452086},
           "val_subsample": {"frac": 0.0986, "kept": 32985, "total": 334516}}
    if arm in ("A2", "A3", "A4"):
        src = "cvae" if arm == "A2" else "ldm"
        cfg["synthetic"] = {"export_id": "stale0000000" if arm == "A2" else "ldm123456789", "source": src,
                            "n_syn": 122850}
    if arm in ("A0os", "A1", "A4"):
        cfg["minority_augmentation"] = {"factor": 2.0, "mix_prob": 0.0 if arm == "A0os" else 0.5}
    return cfg


class _Runs:
    def __init__(self, exp: Path):
        self.exp, self.n = exp, 0

    def new(self, tag, seed, epochs, arm, history):
        self.n += 1
        rd = self.exp / f"20261101_{self.n:06d}_hsi_v9-trainsel-hsi_recursive_{tag}"
        _wj(rd / "config.json", _config(tag, seed, epochs, arm))
        _wj(rd / "history.json", history)
        return rd


def _predictions(rd, arm, seed, y, g, split, path_name):
    pred = _arm_pred(arm, y, seed, split)
    np.savez(rd / path_name, true_label=y, predicted_label=pred, probs=_probs(pred, seed))
    return pred


def make_study(root: Path, *, selection=True, a4_decision="ldm"):
    root = Path(root)
    exp, data, aug = root / "experiments", root / "data/hsi_v9-trainsel/hsi", root / "aug"
    data.mkdir(parents=True)
    truth = {}
    for split, pts in (("val", VAL_PATIENTS), ("test", TEST_PATIENTS)):
        y, g = _split(pts)
        np.save(data / f"y_{split}.npy", y)
        np.save(data / f"groups_{split}.npy", g)
        truth[split] = (y, g)
    runs = _Runs(exp)
    tags = {a.key: a for a in R.DEFAULT_ARMS}
    for arm, score in SCREEN_SCORES.items():          # A1p never ran: "missing"
        runs.new(tags[arm].screen_tag, 42, 12, arm, _history(12, score, dcis_f1=0.3 + score / 10))
    preds = {}
    for arm in ["A0", *SELECTED]:
        for s in SEEDS:
            tag = tags[arm].tag_for_seed(s)
            rd = runs.new(tag, s, 20, arm, _history(20))
            for split, name in (("test", "test_predictions.npz"), ("val", R.VAL_FULL_PRED)):
                y, g = truth[split]
                (rd / name).parent.mkdir(parents=True, exist_ok=True)
                preds[(arm, s, split)] = _predictions(rd, arm, s, y, g, split, name)
            yt = truth["test"][0]
            pt = preds[(arm, s, "test")]
            _wj(rd / "test_report.json", {
                "checkpoint_epoch": 5, "checkpoint_weights": "ema", "best_epoch": 5, "n_test_samples": len(yt),
                "class_names": CLASSES, "sklearn_metrics": {
                    "f1_macro": f1_score(yt, pt, average="macro"),
                    "balanced_accuracy": balanced_accuracy_score(yt, pt)}})
            _wj(rd / "per_patient_metrics.json", {"epoch": 20, "class_names": CLASSES, "per_patient": {
                "197": {"n": 180, "recall_macro": 0.7, "per_class": {"1": {"recall": 0.5, "n": 60}}}}})
    # an unfinished, NEWER A0 run with the same tag: no test_predictions.npz, so it must be ignored
    runs.new(tags["A0"].tag_for_seed(1), 1, 20, "A0", _history(3))
    for arm, t in (("A0", 360.0), ("A3", 400.0)):     # bench: one-epoch solo runs
        runs.new(tags[arm].bench_tag, 42, 1, arm, [{"epoch": 1, "epoch_time": t, "GPU_memory_MB": 900.0}])

    def done(rel, stage, wall_s, summary=None, outputs=None):
        _wj(aug / rel / "done.json", {"status": "ok", "stage": stage, "fingerprint": f"fp-{stage}",
                                      "timings": {"wall_s": wall_s, "started_utc": "2026-11-01T00:00:00+00:00"},
                                      "summary": summary or {}, "outputs": outputs or {}})
    done("prep/tissue", "tissue-mask", 100.0)
    done("gen/cvae", "fit-cvae", 3600.0, {"peak_vram_mb": 2048.0})
    done("gen/ae", "fit-ae", 1800.0, {"peak_vram_mb": 3000.0})
    done("gen/ldm", "fit-ldm", 5400.0, {"train": {"peak_vram_mb": 4096.0}, "samples_per_s": 512.0})
    done("export/ldm", "export-ldm", 600.0, {"n_generated": 245700, "n_accepted": 122850},
         {"X_syn.npy": {"sha256": "x", "size_bytes": 2 * 2 ** 30}})
    done("export/cvae", "export-cvae", 300.0, {"n_accepted": 122850}, {"X_syn.npy": {"size_bytes": 2 ** 30}})
    done("lopo_utility", "lopo-utility", 1200.0)
    _wj(aug / "lopo_utility/decision.json", {"a4_source": a4_decision})
    _wj(aug / "export/ldm/provenance.json", {"export_id": "ldm123456789"})
    _wj(aug / "export/cvae/provenance.json", {"export_id": "cvae00000000"})
    tissue = {}
    (aug / "report/tissue").mkdir(parents=True)
    for split in ("val", "test"):
        tissue[split] = np.random.default_rng(3).random(len(truth[split][0])) < 0.7
        np.save(aug / "report/tissue" / f"is_tissue_{split}.npy", tissue[split])
    done("report/tissue", "report-tissue", 5.0, {"val": {"n": len(truth["val"][0])}})
    sel = root / "reports/augmentation/selection.json"
    if selection:
        _wj(sel, {"schema_version": 1, "rule": "best per family", "selected": SELECTED,
                  "scores": {k: SCREEN_SCORES[k] for k in SELECTED}})
    return SimpleNamespace(root=root, exp=exp, data=data, aug=aug, selection=sel, truth=truth, preds=preds,
                           tissue=tissue)


def _cfg(st, **kw):
    base = dict(experiments=str(st.exp), data_dir=str(st.data), run_data=RUN_DATA, aug_out=str(st.aug),
                selection=str(st.selection), out_root=str(st.root / "reports/augmentation"), date=DATE,
                n_boot=300)
    return R.ReportConfig(**{**base, **kw})


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return make_study(tmp_path_factory.mktemp("study"))


@pytest.fixture(scope="module")
def revealed(study):
    doc, md, figs = R.build_report(_cfg(study, reveal_test=True))
    return SimpleNamespace(doc=doc, md=md, figs=figs)


def _f1(y, p):
    return f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)


# ============================================================================
# statistics
# ============================================================================

def test_paired_deltas_and_sign_count():
    arm = {1: 0.82, 7: 0.80, 13: 0.85, 23: 0.79, 42: 0.90}
    ref = {1: 0.80, 7: 0.81, 13: 0.80, 23: 0.79, 42: 0.85, 99: 0.5}
    d = stats.paired_deltas(arm, ref)
    exp = np.array([0.02, -0.01, 0.05, 0.0, 0.05])
    assert d["seeds"] == [1, 7, 13, 23, 42] and d["n"] == 5
    np.testing.assert_allclose([d["per_seed"][str(s)] for s in d["seeds"]], exp, atol=1e-12)
    assert d["mean"] == pytest.approx(exp.mean()) and d["sd"] == pytest.approx(exp.std(ddof=1))
    from scipy import stats as st
    lo, hi = st.t.interval(0.95, 4, loc=exp.mean(), scale=exp.std(ddof=1) / np.sqrt(5))
    assert d["ci95"] == pytest.approx([lo, hi])
    assert (d["n_positive"], d["n_negative"], d["n_zero"], d["sign_count"]) == (3, 1, 1, "3/5")
    assert d["missing_seeds"] == {"arm": [99], "ref": []}
    # lower-is-better metrics (ECE, FPR): an improvement is a negative delta
    low = stats.paired_deltas({1: 0.05, 7: 0.04}, {1: 0.06, 7: 0.03}, higher_is_better=False)
    assert low["n_improved"] == 1 and low["sign_count"] == "1/2"
    assert stats.paired_deltas({1: None}, {1: 0.5})["n"] == 0


@pytest.mark.parametrize("deltas, met", [
    ([0.012, 0.011, 0.02, -0.001, 0.009], True),        # mean 0.0102, 4/5 positive
    ([0.05, 0.05, -0.01, -0.01, 0.01], False),          # 3/5 positive
    ([0.008, 0.009, 0.009, 0.009, 0.009], False),       # 5/5 positive but mean < 0.01
    ([0.05, 0.05, 0.05, 0.05], None),                    # 4 seeds: incomplete
])
def test_minimum_effect_rule(deltas, met):
    seeds = SEEDS[:len(deltas)]
    d = stats.paired_deltas(dict(zip(seeds, deltas)), dict(zip(seeds, [0.0] * len(deltas))))
    m = stats.minimum_effect(d)
    assert m["met"] is met
    assert m["delta_threshold"] == 0.01 and m["min_positive"] == 4 and m["reason"]


def test_wilcoxon_is_reported_only_when_all_five_seeds_improve():
    from scipy.stats import wilcoxon
    five = [0.02, 0.01, 0.03, 0.005, 0.04]
    w = stats.wilcoxon_one_sided(five)
    assert w["reported"] and w["p"] == 0.03125
    assert wilcoxon(five, alternative="greater").pvalue == pytest.approx(0.03125)
    for d in ([0.02, 0.01, 0.03, 0.005, -0.001], [0.02, 0.01, 0.03, 0.04]):
        w = stats.wilcoxon_one_sided(d)
        assert not w["reported"] and w["p"] is None and "not reported" in w["note"]
    # direction-aware through paired_deltas
    lower = stats.paired_deltas(dict(zip(SEEDS, [0.1] * 5)), dict(zip(SEEDS, [0.2] * 5)), higher_is_better=False)
    assert stats.wilcoxon_one_sided(lower)["p"] == 0.03125


def test_confusion_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    for t in range(40):
        n = int(rng.integers(3, 120))
        y, p = rng.integers(0, 3, n), rng.integers(0, 3, n)
        if t % 3 == 0:
            y[y == 1] = 2                       # a class absent from y
        cm = stats.confusion(y, p)
        np.testing.assert_array_equal(cm, confusion_matrix(y, p, labels=[0, 1, 2]))
        assert stats.metric_from_confusion(cm, "f1_macro") == pytest.approx(_f1(y, p), abs=1e-12)
        assert stats.metric_from_confusion(cm, "balanced_accuracy") == pytest.approx(_bal(y, p), abs=1e-12)
    with pytest.raises(ValueError, match="outside labels"):
        stats.confusion(np.array([0, 3]), np.array([0, 1]))


def _bal(y, p):
    """sklearn's balanced accuracy, its 'y_pred contains classes not in y_true' warning silenced."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return balanced_accuracy_score(y, p)


def _five_patients():
    """Five patients of 100 rows, ONE holding DCIS (as each held-out split)."""
    g = np.repeat([10, 20, 30, 40, 50], 100)
    y = np.zeros(500, dtype=np.int64)
    y[:40] = 1                                        # patient 10: DCIS + healthy
    y[100:200] = 2
    y[300:360] = 2
    a = y.copy()
    b = y.copy()
    b[::6] = (b[::6] + 1) % 3
    a[::23] = (a[::23] + 1) % 3
    return y, g, a, b


def test_patient_bootstrap_fixes_the_labels_and_counts_resamples_missing_a_class():
    y, g, a, b = _five_patients()
    r = stats.patient_bootstrap(y, a, b, g, labels=[0, 1, 2], n_boot=400, seed=3)
    assert r["labels"] == [0, 1, 2] and r["missing_policy"] == "skip"
    assert r["observed"] == pytest.approx(_f1(y, a) - _f1(y, b), abs=1e-12)
    assert r["patients_per_class"] == {"0": 4, "1": 1, "2": 2}       # healthy: 10, 30, 40, 50
    # replay the draw: a resample misses DCIS exactly when patient 10 (index 0) is not drawn
    rng = np.random.default_rng(3)
    miss_dcis = miss_any = 0
    has = {c: np.array([(y[g == p] == c).any() for p in (10, 20, 30, 40, 50)]) for c in (0, 1, 2)}
    for _ in range(400):
        w = np.bincount(rng.integers(0, 5, 5), minlength=5)
        absent = [not (w[has[c]] > 0).any() for c in (0, 1, 2)]
        miss_dcis += absent[1]
        miss_any += any(absent)
    assert r["missing_by_class"]["1"] == miss_dcis and r["n_resamples_missing_a_class"] == miss_any
    assert r["n_boot_used"] == 400 - miss_any and r["n_boot_requested"] == 400
    assert 0.2 < miss_dcis / 400 < 0.45                      # (4/5)^5 = 0.33
    lo, hi = r["ci95"]
    assert lo <= r["observed"] <= hi
    # "keep" scores every resample with the fixed class set: a missing class counts as F1 0
    k = stats.patient_bootstrap(y, a, b, g, n_boot=400, seed=3, missing="keep")
    assert k["n_boot_used"] == 400 and k["n_resamples_missing_a_class"] == miss_any


def test_patient_bootstrap_pools_seeds_and_checks_inputs():
    y, g, a, b = _five_patients()
    r = stats.patient_bootstrap(y, [a, a], [b, y], g, metric="balanced_accuracy", n_boot=50)
    assert r["n_seeds"] == 2
    assert r["observed"] == pytest.approx(np.mean([_bal(y, a) - _bal(y, b), _bal(y, a) - 1.0]))
    with pytest.raises(ValueError, match="pair them by seed"):
        stats.patient_bootstrap(y, [a, a], [b], g)
    with pytest.raises(ValueError, match="metric"):
        stats.patient_bootstrap(y, a, b, g, metric="dcis_f1")


def test_no_patient_level_ci_for_a_one_patient_class():
    y, g, _, _ = _five_patients()
    pol = stats.class_ci_policy(y, g, 1)
    assert not pol["claimable"] and pol["patients"] == [10] and "no patient-level CI" in pol["reason"]
    assert stats.class_ci_policy(y, g, 2)["claimable"]


# ============================================================================
# the report: numbers
# ============================================================================

def test_seeds_deltas_match_an_independent_computation(study, revealed):
    sd = revealed.doc["seeds"]
    assert sd["arms"] == ["A0", "A3", "A1"] and set(sd["splits"]) == {"val", "test"}
    for split in ("val", "test"):
        sp = sd["splits"][split]
        assert sp["source_kind"] == "full"
        y = study.truth[split][0]
        for arm in SELECTED:
            d = sp["deltas_vs_A0"][arm]["f1_macro"]
            exp = {s: _f1(y, study.preds[(arm, s, split)]) - _f1(y, study.preds[("A0", s, split)]) for s in SEEDS}
            assert d["seeds"] == list(SEEDS)
            for s in SEEDS:
                assert d["per_seed"][str(s)] == pytest.approx(exp[s], abs=1e-12)
            n_pos = sum(v > 0 for v in exp.values())
            assert d["n_positive"] == n_pos and d["sign_count"] == f"{n_pos}/5"
            assert d["mean"] == pytest.approx(np.mean(list(exp.values())))
        assert sp["deltas_vs_A0"]["A3"]["f1_macro"]["n_positive"] == 5
        assert sp["deltas_vs_A0"]["A1"]["f1_macro"]["n_positive"] == len(A1_HELPS)
        assert sp["decisions"]["A3"]["minimum_effect"]["met"] is True
        assert sp["decisions"]["A3"]["wilcoxon"]["p"] == 0.03125
        assert sp["decisions"]["A1"]["minimum_effect"]["met"] is False
        assert sp["decisions"]["A1"]["wilcoxon"]["p"] is None
        # the bootstrap: fixed labels, missing-class resamples counted, no DCIS CI
        b = sp["patient_bootstrap"]["A3"]["f1_macro"]
        assert b["labels"] == [0, 1, 2] and b["seeds"] == list(SEEDS) and b["n_patients"] == 5
        assert b["n_resamples_missing_a_class"] > 0 and b["missing_by_class"]["1"] > 0
        assert b["n_boot_used"] == b["n_boot_requested"] - b["n_resamples_missing_a_class"]
        assert sp["dcis_ci"]["claimable"] is False and "dcis_f1" not in sp["patient_bootstrap"]["A3"]


def test_per_run_metrics_fpr_patients_and_tissue(study, revealed):
    sp = revealed.doc["seeds"]["splits"]["test"]
    y, g = study.truth["test"]
    p = study.preds[("A0", 7, "test")]
    row = sp["per_arm"]["A0"]["per_seed"]["7"]
    assert row["f1_macro"] == pytest.approx(_f1(y, p))
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    assert row["healthy_to_dcis_fpr"] == pytest.approx(cm[0, 1] / cm[0].sum()) and row["healthy_to_dcis_fpr"] > 0
    assert row["dcis_recall"] == pytest.approx(cm[1, 1] / cm[1].sum())
    assert row["dcis_precision"] == pytest.approx(cm[1, 1] / cm[:, 1].sum())
    for k in ("roc_auc_macro", "pr_auc_macro", "ece", "nll"):
        assert row[k] is not None
    # DCIS patients: 136 on test, 197 on val, recall per seed
    assert sp["dcis_patients"] == [136]
    assert revealed.doc["seeds"]["splits"]["val"]["dcis_patients"] == [197]
    rec = sp["dcis_patient_recall_per_seed"]["A3"]["136"]
    m = g == 136
    pa = study.preds[("A3", 13, "test")]
    assert rec["13"] == pytest.approx(((pa[m] == 1) & (y[m] == 1)).sum() / (y[m] == 1).sum())
    assert set(sp["per_patient"]["A0"]) == {str(k) for k in TEST_PATIENTS}
    # summed confusion over seeds
    tot = sum(confusion_matrix(y, study.preds[("A1", s, "test")], labels=[0, 1, 2]) for s in SEEDS)
    assert sp["confusion"]["A1"]["summed_over_seeds"] == tot.tolist()
    # tissue vs glass, joined by row index
    t = study.tissue["test"]
    tg = sp["tissue_glass"]["A0"]
    assert tg["tissue"]["n"] == int(t.sum()) and tg["glass"]["n"] == int((~t).sum())
    exp = np.mean([_f1(y[t], study.preds[("A0", s, "test")][t]) for s in SEEDS])
    assert tg["tissue"]["f1_macro"]["mean"] == pytest.approx(exp)
    md = revealed.md
    assert "Tissue vs glass" in md and "### test" in md and "136" in md


def test_screen_scores_ranks_and_selection_check(revealed):
    sc = revealed.doc["screen"]
    rows = {r["arm"]: r for r in sc["rows"]}
    assert rows["A1p"]["status"] == "missing"
    for arm, v in SCREEN_SCORES.items():
        assert rows[arm]["score"] == pytest.approx(v) and rows[arm]["status"] == "ok"
    assert rows["A3"]["family_rank"] == 1 and rows["A4"]["family_rank"] == 2
    assert rows["A1"]["family_rank"] == 1 and rows["A1"]["delta_score_vs_A0"] == pytest.approx(0.03)
    sel = {s["arm"]: s for s in sc["selection"]}
    assert sel["A3"]["within_tie_gap"] and sel["A3"]["score_matches"]
    assert sc["score_epochs"] == [10, 11, 12]


def test_cost_table_and_checks(revealed):
    cost = revealed.doc["cost"]
    assert cost["a4_generator"] == "ldm"
    assert cost["per_source"]["ldm"]["gpu_h"] == pytest.approx((1800 + 5400) / 3600)
    assert cost["per_source"]["cvae"]["gpu_h"] == pytest.approx(1.0)
    assert cost["per_source"]["ldm"]["peak_vram_mb"] == 4096.0 and cost["per_source"]["ldm"]["samples_per_s"] == 512
    assert cost["per_source"]["ldm"]["export_bytes"] == 2 * 2 ** 30
    assert cost["lopo_wall_min"] == pytest.approx(20.0)
    arms = {a["arm"]: a for a in cost["arms"]}
    a3 = arms["A3"]
    assert a3["generator_shared_by"] == ["A3", "A4"] and a3["generator_gpu_h_amortised"] == pytest.approx(1.0)
    assert a3["s_per_epoch"] == 400.0 and a3["classifier_gpu_h_screen"] == pytest.approx(400 * 12 / 3600)
    assert a3["classifier_gpu_h_seeds"] == pytest.approx(400 * 20 * 5 / 3600)
    assert a3["gpu_h_total"] == pytest.approx(400 * 12 / 3600 + 400 * 100 / 3600 + 1.0)
    assert arms["A4"]["generator"] == "ldm" and arms["A1"]["bench_run"] is None
    checks = {c["check"]: c for c in revealed.doc["checks"]}
    assert all(c["ok"] for k, c in checks.items() if k.startswith(("row_order", "test_report_matches")))
    assert sum(k.startswith("row_order") for k in checks) == 2 * 15
    assert checks["train_subsample_kept"]["ok"] and checks["trm_ema_rate_equal"]["ok"]
    stale = [c for k, c in checks.items() if k.startswith("export_id") and c["ok"] is False]
    assert len(stale) == 1 and "stale0000000" in stale[0]["detail"]          # A2's export changed since its run
    assert any(s["qc"].get("n_accepted") == 122850 for s in cost["stages"])


def test_report_records_configs_shas_and_notes(revealed, study):
    doc = revealed.doc
    assert R.SEED42_NOTE in doc["notes"] and "seed 42" in revealed.md.lower()
    assert "date" not in doc["report_config"] and "out_root" not in doc["report_config"]
    assert doc["git_sha"]
    runs = doc["runs"]
    assert sum(r["role"] == "seeds" for r in runs.values()) == 15
    assert all(r["cli_args"]["run_tag"] == r["tag"] for r in runs.values())
    ins = doc["inputs_sha256"]
    assert all(len(v) == 64 for v in ins.values())
    assert "data/y_test.npy" in ins and "aug/report/tissue/is_tissue_val.npy" in ins
    assert "selection/selection.json" in ins
    assert not any(str(study.root) in k for k in ins)            # labels, never machine paths
    # the unfinished newer A0 run was skipped
    a0s1 = [r for r in runs.values() if r["tag"] == "v20-head-s1"]
    assert len(a0s1) == 1 and a0s1[0]["run"].endswith("v20-head-s1")


def test_val_falls_back_to_history_when_full_val_predictions_are_missing(tmp_path):
    st = make_study(tmp_path / "s")
    victim = next(p for p in sorted(st.exp.glob("*aug-ldm-s7/" + R.VAL_FULL_PRED)))
    victim.unlink()
    doc, _, _ = R.build_report(_cfg(st))
    sp = doc["seeds"]["splits"]["val"]
    assert sp["source_kind"] == "history" and "history.json" in sp["source"]
    assert set(sp["per_arm"]["A0"]["summary"]) == set(R.HISTORY_METRICS)
    assert sp["patient_bootstrap"] == {}


def test_selection_errors(tmp_path):
    st = make_study(tmp_path / "s")
    for bad, msg in (({"selected": ["A0"]}, "control"), ({"selected": ["A9"]}, "not one of"),
                     ({"selected": []}, "no selected")):
        _wj(st.selection, bad)
        with pytest.raises(R.ReportError, match=msg):
            R.build_report(_cfg(st))
    _wj(st.selection, {"per_family": {"generative": "aug-ldm", "non-generative": "A1"}})    # tags accepted too
    doc, _, _ = R.build_report(_cfg(st))
    assert doc["selection"] == ["A3", "A1"]


# ============================================================================
# the test seal
# ============================================================================

def _guard_test_reads(monkeypatch, root: Path, opened: list):
    """open / np.load fail on any `test_*` / `*_test.npy` file under `root`
    except heldout_eval's validation files (`heldout_v19/validation/test_*`)."""
    real_open, real_load = builtins.open, np.load

    def is_test(f) -> bool:
        if not isinstance(f, (str, os.PathLike)):
            return False
        p = Path(os.fspath(f))
        return str(p).startswith(str(root)) and R._is_test_path(p)

    def g_open(file, *a, **k):
        if is_test(file):
            raise AssertionError(f"test file opened: {file}")
        if isinstance(file, (str, os.PathLike)) and str(file).startswith(str(root)):
            opened.append(Path(os.fspath(file)))
        return real_open(file, *a, **k)

    def g_load(file, *a, **k):
        if is_test(file):
            raise AssertionError(f"test file loaded: {file}")
        return real_load(file, *a, **k)

    monkeypatch.setattr(builtins, "open", g_open)
    monkeypatch.setattr(np, "load", g_load)


def test_test_columns_absent_without_reveal_flag(study, monkeypatch):
    opened: list = []
    _guard_test_reads(monkeypatch, study.root, opened)
    doc, md, figs = R.build_report(_cfg(study, reveal_test=False))
    assert not doc["test_seal"]["revealed"] and "--reveal_test was not passed" in doc["test_seal"]["detail"]
    assert set(doc["seeds"]["splits"]) == {"val"}
    assert "### test" not in md and "**Test columns: sealed.**" in md
    assert not any(n.endswith("_test.png") for n in figs)
    assert not any(R._is_test_path(p) for p in opened)
    assert not any(k.endswith(("y_test.npy", "groups_test.npy", "test_report.json")) for k in doc["inputs_sha256"])
    assert any(p.name == "test_predictions.npz" and p.parent.parts[-2:] == ("heldout_v19", "validation")
               for p in opened)                                                                # val is read


def test_test_columns_absent_without_selection(tmp_path, monkeypatch):
    st = make_study(tmp_path / "s", selection=False)
    opened: list = []
    _guard_test_reads(monkeypatch, st.root, opened)
    doc, md, _ = R.build_report(_cfg(st, reveal_test=True))      # --reveal_test alone opens nothing
    assert not doc["test_seal"]["revealed"] and "no selection.json" in doc["test_seal"]["detail"]
    assert doc["seeds"] is None and doc["selection"] is None
    assert "no selection.json yet" in md and "### test" not in md
    assert {p.name for p in opened} <= {"config.json", "history.json", "done.json", "decision.json",
                                        "provenance.json"}


def test_screen_reads_only_config_and_history(study, monkeypatch):
    sel = {"selected": SELECTED, "document": {}}
    c = R._Ctx(cfg=_cfg(study, reveal_test=True), inp=R.Inputs({"experiments": study.exp}, reveal_test=True),
               arms=R.DEFAULT_ARMS, index=R.index_runs(study.exp), sealed_reason=None)
    opened: list = []
    _guard_test_reads(monkeypatch, study.root, opened)
    screen = R._screen(c, sel)
    assert {p.name for p in opened} == {"config.json", "history.json"}
    assert {k.rsplit("/", 1)[-1] for k in c.inp.sha256} == {"config.json", "history.json"}
    assert screen["rows"][0]["score"] == pytest.approx(SCREEN_SCORES["A0"])


def test_inputs_refuse_test_files_while_sealed(study):
    inp = R.Inputs({"data": study.data}, reveal_test=False)
    with pytest.raises(R.TestSealError, match="test is sealed"):
        inp.npy(study.data / "y_test.npy")
    assert inp.npy(study.data / "y_val.npy").shape == study.truth["val"][0].shape
    with pytest.raises(R.ReportError, match="outside"):
        inp.json(study.root / "reports/augmentation/selection.json")
    assert R._is_test_path(Path("run/test_predictions.npz"))
    assert not R._is_test_path(Path("run/heldout_v19/validation/test_predictions.npz"))
    # `test=` can only ADD the seal: a file the caller knows holds test rows is
    # refused, and a `test_*` file cannot be declared something else
    with pytest.raises(R.TestSealError):
        inp.npy(study.data / "y_val.npy", test=True)
    inp = R.Inputs({"experiments": study.exp}, reveal_test=False)
    rd = next(study.exp.glob("*_v20-head-s7"))
    for read in (inp.npz, inp.json):
        with pytest.raises(R.TestSealError):
            read(rd / (R.TEST_PRED if read is inp.npz else R.TEST_REPORT), test=False)
    assert inp.sha256 == {}                                   # refused before hashing


def test_a_code_path_pointed_at_a_test_file_fails_while_sealed(study, monkeypatch):
    """train.py always writes test_predictions.npz (`--eval_test best`). If the
    seeds section's validation path pointed at it by mistake, the reading layer
    refuses it while the seal is closed: the seal does not rely on code order."""
    monkeypatch.setattr(R, "VAL_FULL_PRED", R.TEST_PRED)
    with pytest.raises(R.TestSealError, match="test is sealed.*test_predictions.npz"):
        R.build_report(_cfg(study, reveal_test=False))
    r = subprocess.run([sys.executable, "-c", "import sys; from hsi_augment import report as R; "
                        "R.VAL_FULL_PRED = R.TEST_PRED; from scripts import augmentation_report as A; "
                        "sys.exit(A.main(sys.argv[1:]))", "--experiments", str(study.exp), "--data_dir",
                        str(study.data), "--run_data", RUN_DATA, "--aug_out", str(study.aug), "--selection",
                        str(study.selection), "--out_root", str(study.root / "sealed"), "--date", DATE],
                       cwd=str(REPO_ROOT), env=dict(os.environ, PYTHONPATH=str(REPO_ROOT), MPLBACKEND="Agg"),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 1 and "test is sealed" in r.stderr, r.stdout + r.stderr
    assert not (study.root / "sealed").exists()               # nothing written


def test_a_validation_directory_above_the_runs_does_not_unseal_test(tmp_path, monkeypatch):
    """Only `heldout_v19/validation/` right above a file marks its `test_*`
    name as val. An experiments root nested under a directory named
    `validation` keeps every run's real test files sealed."""
    st = make_study(tmp_path / "validation" / "study")
    rd = next(st.exp.glob("*_v20-head-s7"))
    assert "validation" in rd.parts
    inp = R.Inputs({"experiments": st.exp}, reveal_test=False)
    with pytest.raises(R.TestSealError):
        inp.npz(rd / R.TEST_PRED)
    with pytest.raises(R.TestSealError):
        inp.json(rd / R.TEST_REPORT)
    assert inp.npz(rd / R.VAL_FULL_PRED)["true_label"].shape == st.truth["val"][0].shape     # val stays readable
    assert R._is_test_path(Path("validation/test_predictions.npz"))                         # no heldout_v19 above
    assert R._is_test_path(Path("heldout_v19/validation/sub/test_predictions.npz"))         # not the parent
    assert not R._is_test_path(Path("validation/exp/run/heldout_v19/validation/test_predictions.npz"))
    # end to end: the val path pointed at the test file is still refused under the nested root
    monkeypatch.setattr(R, "VAL_FULL_PRED", R.TEST_PRED)
    with pytest.raises(R.TestSealError, match="test is sealed"):
        R.build_report(_cfg(st, reveal_test=False))


# ============================================================================
# determinism, CLI
# ============================================================================

def _tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_report_regenerates_byte_identically(study, tmp_path):
    a = R.generate_report(_cfg(study, reveal_test=True, out_root=str(tmp_path / "a")))
    b = R.generate_report(_cfg(study, reveal_test=True, out_root=str(tmp_path / "b"), date="2027-01-01"))
    ta, tb = _tree(a), _tree(b)
    assert a.name == DATE and b.name == "2027-01-01"            # the date is the directory name only
    assert set(ta) == {"report.md", "report.json", "figures/forest_delta_f1_macro_val.png",
                       "figures/forest_delta_f1_macro_test.png", "figures/per_class_f1_val.png",
                       "figures/per_class_f1_test.png", "figures/per_class_f1_screen.png"}
    assert ta == tb
    assert DATE not in ta["report.md"].decode() and DATE not in ta["report.json"].decode()
    # in place: the same bytes again, and a stale figure from another configuration is removed
    (a / "figures/old.png").write_bytes(b"x")
    R.generate_report(_cfg(study, reveal_test=True, out_root=str(tmp_path / "a")))
    assert _tree(a) == ta
    assert ta["figures/per_class_f1_test.png"][:8] == b"\x89PNG\r\n\x1a\n"


def test_cli(study, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), MPLBACKEND="Agg")
    base = [sys.executable, str(REPO_ROOT / "scripts/augmentation_report.py"), "--experiments", study.exp,
            "--data_dir", study.data, "--run_data", RUN_DATA, "--aug_out", study.aug, "--selection", study.selection,
            "--out_root", tmp_path / "rep", "--date", DATE, "--n_boot", "100"]
    r = subprocess.run([*map(str, base)], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads((tmp_path / "rep" / DATE / "report.json").read_text())
    assert set(doc["seeds"]["splits"]) == {"val"}
    r = subprocess.run([*map(str, base), "--reveal_test"], cwd=str(REPO_ROOT), env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads((tmp_path / "rep" / DATE / "report.json").read_text())
    assert set(doc["seeds"]["splits"]) == {"val", "test"} and doc["report_config"]["n_boot"] == 100
    r = subprocess.run([*map(str, base[:-6]), "--selection", tmp_path / "none.json", "--out_root", tmp_path / "r2",
                        "--date", DATE, "--reveal_test"], cwd=str(REPO_ROOT), env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0 and "--reveal_test ignored" in r.stdout
    r = subprocess.run([*map(str, base), "--date", "27-10-2026"], cwd=str(REPO_ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 1 and "YYYY-MM-DD" in r.stderr


# ============================================================================
# the report-tissue stage
# ============================================================================

def test_report_tissue_stage_is_a_registered_final_eval_stage():
    reg = load_builtin_stages()
    spec = reg.get("report-tissue")
    assert (spec.group, spec.dirname, spec.purposes, spec.requires) == \
        ("report", "report/tissue", ("final_eval",), ("tissue-mask",))
    assert not spec.in_all and "report-tissue" not in {s.name for s in reg.resolve(["all"])}
    assert "report-tissue" in splits.FINAL_EVAL_STAGES
    assert splits.registry_policy_violations(reg) == []
    assert splits.held_out_readers(reg)["report-tissue"] == ["final_eval"]
    with pytest.raises(Exception, match="splits"):
        ReportTissueConfig(splits=("train",))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


def test_report_tissue_stage_scores_val_test_with_the_train_reference(tiny, tmp_path):
    from hsi_augment import tissue
    from hsi_augment.io import read_access_log
    reg = load_builtin_stages()
    out = tmp_path / "aug"
    opts = RunOptions(data_dir=str(tiny.data_dir), out=str(out))
    run_stage(reg.get("tissue-mask"), TissueMaskConfig(), opts, reg)
    sel = tmp_path / "selection.json"
    d = out / "report/tissue"
    # test is sealed until selection.json exists: refused before anything is read
    with pytest.raises(StageError, match="sealed"):
        run_stage(reg.get("report-tissue"), ReportTissueConfig(selection=str(sel)), opts, reg)
    assert not (d / DONE_NAME).exists()
    assert not [r for r in read_access_log(out / "split_access_log.jsonl") if r["stage"] == "report-tissue"]
    # val alone runs before the selection
    res = run_stage(reg.get("report-tissue"), ReportTissueConfig(splits=("val",), selection=str(sel)), opts, reg)
    assert res.status == "ran" and (d / "is_tissue_val.npy").is_file() and not (d / "is_tissue_test.npy").exists()
    _wj(sel, {"selected": ["A3"]})
    res = run_stage(reg.get("report-tissue"), ReportTissueConfig(selection=str(sel), chunk_rows=7), opts, reg)
    ref = tissue.load_reference(out / "prep/tissue")
    for split in ("val", "test"):
        X = np.load(tiny.data_dir / f"X_{split}.npy")
        m = np.load(d / f"is_tissue_{split}.npy")
        assert m.dtype == bool and len(m) == tiny.n_rows[split]
        np.testing.assert_array_equal(m, ref.is_tissue(X))           # chunked == whole, row order kept
        np.testing.assert_array_equal(np.load(d / f"tissue_score_{split}.npy"),
                                      ref.score(X).astype(tissue.SCORE_DTYPE))
        assert (m == ~tiny.is_glass[split]).mean() > 0.9             # the fixture's ground truth
    doc = json.loads((d / R.TISSUE_JSON).read_text())
    assert doc["labels_read"] is False and doc["tau_deg"] == ref.tau_deg
    assert set(doc["splits"]) == {"val", "test"} and doc["splits"]["test"]["n"] == tiny.n_rows["test"]
    done = json.loads((d / DONE_NAME).read_text())
    assert set(done["outputs"]) == {"is_tissue_val.npy", "is_tissue_test.npy", "tissue_score_val.npy",
                                    "tissue_score_test.npy", R.TISSUE_JSON}
    assert "tissue-mask" in done["upstream"]
    recs = [r for r in read_access_log(out / "split_access_log.jsonl") if r["stage"] == "report-tissue"]
    assert {(r["split"], r["purpose"], r["allowed"], tuple(r["files"])) for r in recs} == \
        {("val", "final_eval", True, ("X_val.npy",)), ("test", "final_eval", True, ("X_test.npy",))}
