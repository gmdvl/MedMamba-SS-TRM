# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_index.py
===============================
The `index` stage (#15, Plan §Phase 4 items 1, 2 and 5): the provenance index
`index_train.npz` and `imbalance_report.json`, on the tiny fixture.

  * the index has one entry per train row, and its columns equal the build's
    sidecars, the fixture's true grid (r, c), and the tissue-mask outputs;
  * `patch_feats` equals a direct numpy per-band mean/std, and
    `scripts.shallow_baseline.featurize(x, "mean_std")`;
  * the class / patient / capture tables reproduce the fixture's known counts in
    every split, and on train tissue rows equal a direct computation;
  * the effective number of samples is Cui et al.'s closed form;
  * the access log holds X reads of train only; val/test are opened for their
    int sidecars, purpose "structural";
  * a partial (`max_rows`) index, a missing upstream, mismatched tissue outputs;
  * the CLI runs, skips and is in `preprocess`;
  * an optional real-data smoke test (HSI_AUGMENT_REAL_DATA): 3 tissue captures
    and 2,000 indexed rows, never the whole X_train.

CPU only, a few seconds in total.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hsi_augment import provenance as pv
from hsi_augment import tissue
from hsi_augment.io import read_access_log
from hsi_augment.registry import load_builtin_stages
from hsi_augment.stage import DONE_NAME, RunOptions, StageError, StageNotReady, run_stage
from hsi_augment.stages.index import IndexConfig
from hsi_augment.stages.tissue_mask import TissueMaskConfig
from tests import hsi_augment_fixtures as fx
from tests.hsi_augment_fixtures import make_tiny_build

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = Path("prep") / "index"
TISSUE_DIR = Path("prep") / "tissue"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
SPLIT_OF = {"train": "train", "validation": "val", "test": "test"}


def _run(build, out, cfg=None, *, tissue_cfg=None, **opts):
    reg = load_builtin_stages()
    options = RunOptions(data_dir=str(build.data_dir), out=str(out), **opts)
    run_stage(reg.get("tissue-mask"), tissue_cfg or TissueMaskConfig(), options, reg)
    return run_stage(reg.get("index"), cfg or IndexConfig(), options, reg)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def ran(tiny, tmp_path_factory):
    """tissue-mask then index with defaults, once, shared by the read-only tests."""
    out = tmp_path_factory.mktemp("aug")
    res = _run(tiny, out)
    d = out / STAGE_DIR
    return SimpleNamespace(out=out, dir=d, result=res, index=pv.load_index(d, require_complete=True),
                           meta=json.loads((d / pv.INDEX_META_NAME).read_text()),
                           report=json.loads((d / pv.REPORT_NAME).read_text()))


def _expected_counts():
    """{split: {"rows": {class: n}, "patients": {class: set}, "captures": {class: n},
    "patient_rows": {patient: {class: n}}}} from the fixture's PATIENTS table."""
    out = {s: {"rows": Counter(), "patients": {c: set() for c in fx.CLASS_NAMES}, "captures": Counter(),
               "patient_rows": {}} for s in fx.SPLITS}
    for patient, split, *tissues in fx.PATIENTS:
        s = SPLIT_OF[split]
        seen = Counter()
        for t in tissues:
            seen[t] += 1
            key = f"{t}_HS_VNIR_{patient}_{t}_x10_C{seen[t]:02d}"
            n = (fx.SHORT_GRID_ROWS if key == fx.SHORT_CAPTURE_KEY else fx.GRID_ROWS) * fx.GRID_COLS
            name = fx.CLASS_NAMES[fx.TISSUE_SPELLING.index(t)]
            out[s]["rows"][name] += n
            out[s]["captures"][name] += 1
            out[s]["patients"][name].add(patient)
            out[s]["patient_rows"].setdefault(patient, Counter())[name] += n
    return out


# ============================================================================
# the index
# ============================================================================

def test_index_has_one_entry_per_train_row_with_the_sidecars(tiny, ran):
    assert ran.result.status == "ran" and (ran.dir / DONE_NAME).is_file()
    idx, n = ran.index, tiny.n_rows["train"]
    assert list(idx) == list(pv.INDEX_COLUMNS)
    for col, dtype in pv.INDEX_DTYPES.items():
        assert idx[col].dtype == dtype, col
        assert idx[col].shape[0] == n, col
    np.testing.assert_array_equal(idx["row"], np.arange(n))
    np.testing.assert_array_equal(idx["y"], np.load(tiny.data_dir / "y_train.npy"))
    np.testing.assert_array_equal(idx["patient"], np.load(tiny.data_dir / "groups_train.npy"))
    np.testing.assert_array_equal(idx["capture"], np.load(tiny.data_dir / "captures_train.npy"))
    m = ran.meta
    assert m["complete"] is True and m["n_rows"] == m["n_train_rows"] == n and m["max_rows"] is None
    assert m["index_schema_version"] == pv.INDEX_SCHEMA_VERSION
    assert {c: v["dtype"] for c, v in m["columns"].items()} == {c: str(d) for c, d in pv.INDEX_DTYPES.items()}
    assert m["columns"]["patch_feats"]["shape"] == [n, 2 * tiny.n_bands]
    assert "No val/test provenance index" in m["val_test"]
    done = json.loads((ran.dir / DONE_NAME).read_text())
    assert set(done["outputs"]) == {pv.INDEX_NAME, pv.INDEX_META_NAME, pv.REPORT_NAME,
                                    "figures/class_balance.png", "figures/patient_contributions.png"}


def test_rc_and_ordinal_equal_the_fixture_grid(tiny, ran):
    idx = ran.index
    rc = tiny.grid_rc["train"]
    np.testing.assert_array_equal(idx["r"], rc[:, 0])
    np.testing.assert_array_equal(idx["c"], rc[:, 1])
    np.testing.assert_array_equal(idx["ordinal"], rc[:, 0] * tiny.grid_cols + rc[:, 1])
    assert ran.meta["grid"]["rows_without_grid"] == 0 and ran.meta["grid"]["grid_cols"] == tiny.grid_cols


def test_tissue_columns_equal_the_tissue_mask_outputs(ran):
    t = ran.out / TISSUE_DIR
    score = np.load(t / tissue.SCORE_NAME)
    np.testing.assert_array_equal(ran.index["tissue_score"].view(np.uint16), score.view(np.uint16))
    np.testing.assert_array_equal(ran.index["is_tissue"], np.load(t / tissue.MASK_NAME))
    tau = json.loads((t / tissue.TAU_NAME).read_text())["tau_deg"]
    assert ran.meta["tissue"]["tau_deg"] == tau and ran.meta["tissue"]["complete"] is True
    assert ran.report["tissue"]["tau_deg"] == tau


def test_patch_feats_equal_a_direct_numpy_computation(tiny, ran):
    X = np.load(tiny.data_dir / "X_train.npy").astype(np.float64)
    direct = np.concatenate([X.mean(axis=(1, 2)), X.std(axis=(1, 2))], axis=1).astype(np.float32)
    np.testing.assert_array_equal(ran.index["patch_feats"], direct)
    from scripts.shallow_baseline import featurize
    np.testing.assert_array_equal(ran.index["patch_feats"], featurize(X, "mean_std").astype(np.float32))
    # the library function on any block size, and on synthetic-looking input (QC reuses it)
    np.testing.assert_array_equal(pv.patch_feats(X, block_rows=7), direct)
    with pytest.raises(ValueError, match=r"\(n, H, W, C\)"):
        pv.patch_feats(X[0])
    assert ran.meta["patch_feats"]["n_nonfinite_rows"] == 0
    assert len(ran.meta["patch_feats"]["differences_from_shallow_probe"]) == 3


def test_load_index_reads_selected_columns(ran):
    sub = pv.load_index(ran.dir / pv.INDEX_NAME, ["y", "patient"])
    assert list(sub) == ["y", "patient"]
    np.testing.assert_array_equal(sub["y"], ran.index["y"])
    with pytest.raises(KeyError, match="no column"):
        pv.load_index(ran.dir, ["nope"])


# ============================================================================
# the imbalance report
# ============================================================================

def test_class_patient_and_capture_tables_reproduce_the_fixture(ran):
    exp = _expected_counts()
    rep = ran.report["splits"]
    for s in fx.SPLITS:
        t, e = rep[s], exp[s]
        assert t["rows_per_class"] == {c: e["rows"][c] for c in fx.CLASS_NAMES}, s
        assert t["rows"] == sum(e["rows"].values())
        assert t["patients_per_class"] == {c: len(e["patients"][c]) for c in fx.CLASS_NAMES}, s
        assert t["captures_per_class"] == {c: e["captures"][c] for c in fx.CLASS_NAMES}, s
        assert t["n_patients"] == len(e["patient_rows"])
        for c in fx.CLASS_NAMES:
            share = t["share_per_class"][c]
            assert share == pytest.approx(e["rows"][c] / t["rows"])
        # per-patient contributions
        assert {p["patient"]: p["rows_per_class"] for p in t["per_patient"]} == \
            {p: {c: r[c] for c in fx.CLASS_NAMES} for p, r in e["patient_rows"].items()}
        for p in t["per_patient"]:
            for c in fx.CLASS_NAMES:
                want = p["rows_per_class"][c] / t["rows_per_class"][c] if t["rows_per_class"][c] else None
                assert p["share_of_class_rows"][c] == (pytest.approx(want) if want is not None else None)
    tr = rep["train"]
    # the fixture's anchors: 6 train patients, 4 / 2 / 4 per class, 100 / 40 / 100 rows, ratio 2.5
    assert tr["n_patients"] == 6 and tr["patients_per_class"] == {"healthy": 4, "DCIS": 2, "IDC": 4}
    assert tr["rows_per_class"] == {"healthy": 100, "DCIS": 40, "IDC": 100}
    assert tr["imbalance"]["max_over_min"] == 2.5 and tr["imbalance"]["smallest_class"] == "DCIS"
    assert tr["imbalance"]["ratio_to_minority"]["DCIS"] == {"healthy": 2.5, "DCIS": 1.0, "IDC": 2.5}
    assert tr["imbalance"]["patients_max_over_min"] == 2.0
    conc = tr["patient_concentration"]["DCIS"]
    assert conc["n_patients"] == 2 and conc["effective_patients"] == pytest.approx(2.0)
    assert sorted(conc["patients"]) == [101, 102] and conc["largest_share"] == 0.5
    # healthy: 106 has two captures (40 rows), 101/103/104 one each (20): n_eff = 100^2 / (40^2 + 3 * 20^2)
    h = tr["patient_concentration"]["healthy"]
    assert h["patients"][0] == 106 and h["effective_patients"] == pytest.approx(100 ** 2 / (40 ** 2 + 3 * 20 ** 2))
    # val holds one DCIS patient, test one, like the real build
    assert rep["val"]["patients_per_class"]["DCIS"] == rep["test"]["patients_per_class"]["DCIS"] == 1
    assert rep["val"]["rows_per_class"]["IDC"] == fx.SHORT_GRID_ROWS * fx.GRID_COLS


def test_tissue_restricted_tables_equal_a_direct_computation(tiny, ran):
    mask = np.load(ran.out / TISSUE_DIR / tissue.MASK_NAME)
    y = np.load(tiny.data_dir / "y_train.npy")
    g = np.load(tiny.data_dir / "groups_train.npy")
    t = ran.report["tissue"]["train"]
    names = fx.CLASS_NAMES
    assert t["rows"] == int(mask.sum())
    assert t["rows_per_class"] == {c: int(((y == k) & mask).sum()) for k, c in enumerate(names)}
    assert t["patients_per_class"] == {c: len(np.unique(g[(y == k) & mask])) for k, c in enumerate(names)}
    for p in t["per_patient"]:
        for k, c in enumerate(names):
            assert p["rows_per_class"][c] == int(((g == p["patient"]) & (y == k) & mask).sum())
    # the fixture's glass is 15-40% of every capture, so every class keeps tissue rows and loses some
    frac = ran.report["tissue"]["tissue_fraction"]
    for k, c in enumerate(names):
        assert frac["per_class"][c] == pytest.approx(((y == k) & mask).sum() / (y == k).sum())
        assert 0.5 < frac["per_class"][c] < 0.9
    assert frac["overall"] == pytest.approx(mask.mean())
    assert frac["per_patient"]["101"]["IDC"] is None and frac["per_patient"]["101"]["DCIS"] is not None
    assert ran.report["tissue"]["complete"] is True and ran.report["tissue"]["n_unscored_rows"] == 0


def test_effective_number_is_cui_et_al_closed_form():
    for beta in (0.0, 0.5, 0.9, 0.999, 0.9999):
        for n in (0, 1, 2, 7, 50):
            want = sum(beta ** k for k in range(n))           # E_n = sum_{k<n} beta^k
            assert pv.effective_number(n, beta) == pytest.approx(want, rel=1e-12, abs=1e-15)
            if beta > 0:
                assert pv.effective_number(n, beta) == pytest.approx((1 - beta ** n) / (1 - beta), rel=1e-12)
    # saturation at 1 / (1 - beta) for counts of the real build's size
    e = pv.effective_number([122_850, 722_358, 1_606_878], 0.999)
    np.testing.assert_allclose(e, 1000.0, rtol=1e-12)
    assert pv.effective_number(122_850, 0.9999) == pytest.approx(10_000 * (1 - 0.9999 ** 122_850), rel=1e-12)
    # precision where 1 - beta^n is tiny: E_1 = 1 exactly, E_2 = 1 + beta
    assert pv.effective_number(1, 1 - 1e-12) == pytest.approx(1.0, rel=1e-9)
    assert pv.effective_number(2, 1 - 1e-12) == pytest.approx(2.0, rel=1e-9)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="beta"):
            pv.effective_number(10, bad)
    # class-balanced weights: 1 / E_n normalised to the number of classes present
    w = pv.class_balanced_weights([10, 0, 40], 0.9)
    e10, e40 = pv.effective_number(10, 0.9), pv.effective_number(40, 0.9)
    assert w[1] is None and w[0] + w[2] == pytest.approx(2.0) and w[0] / w[2] == pytest.approx(e40 / e10)
    assert pv.kish_effective_n([5, 5, 5, 5]) == pytest.approx(4.0) and pv.kish_effective_n([0, 0]) is None


def test_report_effective_numbers_match_the_formula(ran):
    tr = ran.report["splits"]["train"]
    assert ran.report["betas"] == [0.999, 0.9999]
    for beta in (0.999, 0.9999):
        block = tr["effective_number"][f"{beta:g}"]
        for c, n in tr["rows_per_class"].items():
            assert block["per_class"][c] == pytest.approx((1 - beta ** n) / (1 - beta), rel=1e-12)
            assert block["saturation"][c] == pytest.approx(1 - beta ** n, rel=1e-9)
        assert block["upper_bound"] == pytest.approx(1 / (1 - beta))
        assert sum(block["class_balanced_weights"].values()) == pytest.approx(3.0)


def test_report_carries_the_limitation_and_the_val_test_note(ran):
    lim = ran.report["limitations"]
    assert len(lim) == 1 and "one cell" in lim[0] and "F13" in lim[0] and "#40" in lim[0]
    assert "final_eval" in ran.report["tissue"]["val_test"] and "load_reference" in ran.report["tissue"]["val_test"]
    assert ran.report["split_access"]["val"].startswith("y, groups, captures only")


def test_figures_are_written(ran):
    for name in ("class_balance.png", "patient_contributions.png"):
        p = ran.dir / pv.FIGURES_DIR / name
        assert p.is_file() and p.read_bytes()[:8] == PNG_MAGIC and p.stat().st_size > 5_000


# ============================================================================
# split access
# ============================================================================

def test_access_log_shows_x_reads_on_train_only(ran):
    log = read_access_log(ran.out / "split_access_log.jsonl")
    mine = [r for r in log if r["stage"] == "index"]
    assert {(r["split"], r["purpose"], r["allowed"]) for r in mine} == \
        {("train", "describe", True), ("val", "structural", True), ("test", "structural", True)}
    for r in mine:
        if r["split"] == "train":
            assert set(r["files"]) == {f"{k}_train.npy" for k in ("X", "y", "groups", "captures")}
        else:
            assert set(r["files"]) == {f"{k}_{r['split']}.npy" for k in ("y", "groups", "captures")}
    assert not any(f.startswith("X_") and not f.endswith("_train.npy") for r in log for f in r["files"])


def test_stage_registration_declares_its_purposes():
    spec = load_builtin_stages().get("index")
    assert (spec.group, spec.dirname, spec.requires) == ("preprocess", "prep/index", ("tissue-mask",))
    assert spec.purposes == ("describe", "structural") and spec.reads_held_out()


# ============================================================================
# partial runs and failures
# ============================================================================

def test_max_rows_gives_a_partial_index_and_a_full_report(tiny, ran, tmp_path):
    res = _run(tiny, tmp_path, IndexConfig(max_rows=37, chunk_rows=10))
    d = tmp_path / STAGE_DIR
    assert res.done["summary"]["complete"] is False and res.done["summary"]["n_index_rows"] == 37
    idx = pv.load_index(d)
    assert all(a.shape[0] == 37 for a in idx.values())
    for col in pv.INDEX_COLUMNS:          # the first 37 entries of the full index, exactly
        np.testing.assert_array_equal(idx[col], ran.index[col][:37], err_msg=col)
    meta = json.loads((d / pv.INDEX_META_NAME).read_text())
    assert meta["complete"] is False and meta["n_rows"] == 37 and meta["n_train_rows"] == tiny.n_rows["train"]
    with pytest.raises(ValueError, match="partial index"):
        pv.load_index(d, require_complete=True)
    # the report reads no X: it is the same as the full run's
    assert json.loads((d / pv.REPORT_NAME).read_text())["splits"] == ran.report["splits"]


def test_partial_tissue_mask_is_carried_into_the_report(tiny, tmp_path):
    _run(tiny, tmp_path, tissue_cfg=TissueMaskConfig(max_captures=3))
    d = tmp_path / STAGE_DIR
    meta = json.loads((d / pv.INDEX_META_NAME).read_text())
    rep = json.loads((d / pv.REPORT_NAME).read_text())
    unscored = tiny.n_rows["train"] - 3 * tiny.grid_rows * tiny.grid_cols
    assert meta["tissue"]["complete"] is False and meta["tissue"]["n_unscored_rows"] == unscored
    assert rep["tissue"]["complete"] is False and rep["tissue"]["n_unscored_rows"] == unscored
    assert rep["tissue"]["train"]["rows"] <= 3 * tiny.grid_rows * tiny.grid_cols


def test_index_needs_the_tissue_mask_stage(tiny, tmp_path):
    reg = load_builtin_stages()
    with pytest.raises(StageNotReady, match="tissue-mask"):
        run_stage(reg.get("index"), IndexConfig(), RunOptions(data_dir=str(tiny.data_dir), out=str(tmp_path)), reg)


def test_mismatched_tissue_outputs_fail_without_done_json(tiny, ran, tmp_path):
    out = tmp_path / "aug"
    shutil.copytree(ran.out / TISSUE_DIR, out / TISSUE_DIR)
    reg = load_builtin_stages()
    opts = RunOptions(data_dir=str(tiny.data_dir), out=str(out))
    mask_path = out / TISSUE_DIR / tissue.MASK_NAME
    mask = np.load(mask_path)
    np.save(mask_path, ~mask)                                 # not the rule applied to the score
    with pytest.raises(StageError, match="different runs"):
        run_stage(reg.get("index"), IndexConfig(), opts, reg)
    np.save(mask_path, mask[:-1])                             # wrong length
    with pytest.raises(StageError, match="do not describe this train split"):
        run_stage(reg.get("index"), IndexConfig(), opts, reg)
    assert not (out / STAGE_DIR / DONE_NAME).exists() and not (out / STAGE_DIR / pv.INDEX_NAME).exists()


def test_config_validation():
    from hsi_augment.config import ConfigError
    for bad in ({"chunk_rows": 0}, {"betas": ()}, {"betas": (0.999, 1.0)}, {"betas": (-0.1,)},
                {"betas": (0.9, 0.9)}, {"max_rows": 0}):
        with pytest.raises(ConfigError):
            IndexConfig(**bad)
    assert IndexConfig().betas == (0.999, 0.9999) and IndexConfig().max_rows is None


# ============================================================================
# CLI
# ============================================================================

def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)


def test_cli_runs_skips_and_belongs_to_preprocess(tiny, tmp_path):
    r = _cli("index", "--help")
    assert r.returncode == 0 and "--betas" in r.stdout and "--max_rows" in r.stdout
    r = _cli("preprocess", "--help")
    assert r.returncode == 0 and "prep/index" in r.stdout and "requires tissue-mask" in r.stdout
    out = tmp_path / "aug"
    r = _cli("index", "--data_dir", tiny.data_dir, "--out", out)
    assert r.returncode == 1 and "tissue-mask" in (r.stdout + r.stderr)        # upstream not done
    r = _cli("tissue-mask", "--data_dir", tiny.data_dir, "--out", out, "--sample_rows", "200")
    assert r.returncode == 0, r.stdout + r.stderr
    base = ["index", "--data_dir", tiny.data_dir, "--out", out, "--betas", "0.99", "0.999"]
    r = _cli(*base)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "index: ran" in r.stdout and "[index] train rows per class healthy 100 / DCIS 40 / IDC 100" in r.stdout
    rep = json.loads((out / STAGE_DIR / pv.REPORT_NAME).read_text())
    assert rep["betas"] == [0.99, 0.999] and set(rep["splits"]["train"]["effective_number"]) == {"0.99", "0.999"}
    r = _cli(*base)
    assert r.returncode == 0 and "index: skipped" in r.stdout
    r = _cli(*base[:5], "--betas", "1.0")
    assert r.returncode == 2 and "betas" in r.stderr


# ============================================================================
# real data (opt-in)
# ============================================================================

REAL_DATA = os.environ.get("HSI_AUGMENT_REAL_DATA")


@pytest.mark.skipif(not REAL_DATA, reason="set HSI_AUGMENT_REAL_DATA to the trainsel hsi dir to run")
def test_real_data_smoke(tmp_path):
    """tissue-mask on 3 captures (~115 MB of X read twice), then the index of
    the first 2,000 train rows (~15 MB of X). Never the whole X_train. The
    imbalance report reads no X, so it must reproduce the build's known tables
    exactly."""
    from hsi_augment.adapters import get_adapter
    real = get_adapter("histologyhsi_bc", REAL_DATA)
    reg = load_builtin_stages()
    opts = RunOptions(data_dir=REAL_DATA, out=str(tmp_path))
    run_stage(reg.get("tissue-mask"), TissueMaskConfig(max_captures=3, sample_rows=5_000, mosaics_per_class=1,
                                                       grid_strict=False), opts, reg)
    res = run_stage(reg.get("index"), IndexConfig(max_rows=2_000), opts, reg)
    d = tmp_path / STAGE_DIR
    assert res.done["summary"]["n_index_rows"] == 2_000 and res.done["summary"]["complete"] is False
    idx = pv.load_index(d)
    assert idx["patch_feats"].shape == (2_000, 2 * real.n_bands) and np.isfinite(idx["patch_feats"]).all()
    assert idx["r"].max() < 54 and idx["c"].max() < 91 and idx["r"].min() >= 0
    rep = json.loads((d / pv.REPORT_NAME).read_text())["splits"]
    assert rep["train"]["rows_per_class"] == {"healthy": 722_358, "DCIS": 122_850, "IDC": 1_606_878}
    assert rep["val"]["rows_per_class"] == {"healthy": 98_280, "DCIS": 24_570, "IDC": 211_666}
    assert rep["test"]["rows_per_class"] == {"healthy": 83_538, "DCIS": 24_570, "IDC": 240_786}
    assert rep["train"]["patients_per_class"] == {"healthy": 30, "DCIS": 5, "IDC": 34}
    assert round(rep["train"]["imbalance"]["max_over_min"], 1) == 13.1
    assert rep["train"]["imbalance"]["label"] == "IDC:DCIS"
    assert rep["train"]["patient_concentration"]["DCIS"]["patients"] and \
        sorted(rep["train"]["patient_concentration"]["DCIS"]["patients"]) == [25, 45, 85, 107, 152]
    files = {f for r in read_access_log(tmp_path / "split_access_log.jsonl") for f in r["files"]}
    assert "X_train.npy" in files and "X_val.npy" not in files and "X_test.npy" not in files
