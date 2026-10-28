# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_qc.py
============================
The QC library `hsi_augment/qc/` and the `qc-reference` stage (#28, Plan
§Phase 8b).

On the tiny fixture, end to end (tissue-mask -> leakage -> qc-reference):
  * outputs, schemas and split access: the stage reads train with purpose
    "qc" only, and a loaded reference reads train through its own guard,
    logged as `qc-apply`;
  * the calibration targets are met on real LOPO held-out DCIS (each rule
    >= 99%, all rules >= 95%), re-checked by re-scoring those patches through
    the LOADED reference; the planted duplicate is excluded as a data duplicate;
  * crafted rejects: a duplicated train patch and a near copy (R6); values out
    of range, and values that become non-finite only in the float16 cast (R1);
    glass (R4); flat and blurred patches (R3); wrong-class patches (all rules);
  * a calibration that cannot meet the targets raises CalibrationError and
    leaves no done.json; a rerun writes byte-identical outputs;
  * set-level metrics: real cross-patient ranges, bootstrap CIs, the
    nearest-neighbour attribution flag, the reporting-only A0 block.

In the test itself, on data whose answer is known by construction:
  * R3: sharp textured 11 x 11 patches vs the same patches Gaussian- or
    box-blurred, with thresholds from `rules.calibrate` on other sharp patches;
  * R5: a three-class problem with four patients per class whose classes
    differ in spectral shape, cross-fitted probes and kNN agreement, thresholds
    from `rules.calibrate` on the held-out target patients;
  * the primitives: the export cast, exact float16 quantiles, the PCA space,
    curvature within segments, the kNN search and exact re-rank, the hash
    index, calibration mechanics, set metrics, the probe vs sklearn.

Fixture limitations (why R3 and R5 are also tested on in-test data)
-------------------------------------------------------------------
* R5. The fixture has two DCIS train patients, so each LOPO fold's probe sees
  one DCIS patient; and the fixture's DCIS absorbance is about 1.5 x
  healthy's (the same shape at a higher density, and density varies inside
  every capture). Real held-out DCIS patches then reach p(DCIS) ~ 0.03 and kNN
  agreement 0, so a calibration that must accept >= 99% of them puts R5's
  thresholds there: on the fixture, wrong-class patches are rejected mostly by
  R2, not R5. The thresholds are NOT changed for that; the fixture test asserts
  that QC as a whole rejects wrong-class held-out patches, and R5's own power
  is tested on the in-test problem.
* R3. The fixture's texture is smoothed (sigma 1.5 px) and its patches are
  5 x 5, so its real patches already sit at the smooth end of the range and
  5 x 5 statistics are noisy. A sigma-1.5 blur is still caught there (on the
  DCT high-frequency fraction); the sharp-vs-blurred test with box and
  Gaussian blur runs on in-test 11 x 11 patches, the real patch size.
* The 40% attribution flag. With two DCIS train patients, any set has one
  patient at >= 50%, so the unflagged branch is tested on `attribution` directly.

CPU only, a few seconds in total; every array is tiny.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, uniform_filter

from hsi_augment import splits
from hsi_augment.adapters import spectral_segments
from hsi_augment.config import ConfigError
from hsi_augment.io import ACCESS_LOG_NAME, read_access_log
from hsi_augment.qc import (CHECKS, RULE_BITS, RULES, CalibrationError, QCReference, apply_rules,
                            decode_bitmask, set_metrics, summarize_bitmask)
from hsi_augment.qc import consistency as K
from hsi_augment.qc import features as F
from hsi_augment.qc import memorization as M
from hsi_augment.qc import rules as R
from hsi_augment.qc import setmetrics as SM
from hsi_augment.qc.reference import (REFERENCE_NAME, SET_REFERENCE_NAME, THRESHOLDS_NAME, calibration_file)
from hsi_augment.registry import load_builtin_stages
from hsi_augment.stage import DONE_NAME, RunOptions, run_stage
from hsi_augment.stages.leakage import LeakageConfig
from hsi_augment.stages.qc_reference import QCReferenceConfig
from hsi_augment.stages.tissue_mask import TissueMaskConfig
from tests.hsi_augment_fixtures import GLASS_LEVEL, WAVELENGTHS, make_tiny_build

STAGE_DIR = Path("qc") / "reference"
DCIS = 1
R3_CHECKS = [c for c in CHECKS if c.rule == "R3"]
R5_CHECKS = [c for c in CHECKS if c.rule == "R5"]


# ============================================================================
# fixtures
# ============================================================================

def _run_chain(build, out, qc_cfg=None):
    """tissue-mask and leakage (the declared upstreams), then qc-reference."""
    reg = load_builtin_stages()
    opts = RunOptions(data_dir=str(build.data_dir), out=str(out))
    for name, cfg in (("tissue-mask", TissueMaskConfig()), ("leakage", LeakageConfig())):
        assert run_stage(reg.get(name), cfg, opts, reg).status == "ran"
    return run_stage(reg.get("qc-reference"), qc_cfg or QCReferenceConfig(device="cpu"), opts, reg)


@dataclass
class Ran:
    res: object
    out: Path
    d: Path
    ref: QCReference
    X: np.ndarray            # X_train memmap
    y: np.ndarray
    groups: np.ndarray
    mask: np.ndarray         # tissue mask of train
    thresholds: dict


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def ran(tiny, tmp_path_factory):
    out = tmp_path_factory.mktemp("aug")
    res = _run_chain(tiny, out)
    d = out / STAGE_DIR
    X = np.load(tiny.data_dir / "X_train.npy", mmap_mode="r")
    return Ran(res=res, out=out, d=d, ref=QCReference.load(d, train_X=X), X=X,
               y=np.load(tiny.data_dir / "y_train.npy"), groups=np.load(tiny.data_dir / "groups_train.npy"),
               mask=np.load(out / "prep" / "tissue" / "tissue_mask_train.npy"),
               thresholds=json.loads((d / THRESHOLDS_NAME).read_text()))


def _heldout_dcis(ran, tiny, k):
    """Fold k's real held-out DCIS tissue rows, minus the planted duplicate pair."""
    rows = np.flatnonzero((ran.y == DCIS) & ran.mask & np.isin(ran.groups, ran.ref.heldout[k]))
    return rows[~np.isin(rows, [tiny.duplicate_row, tiny.duplicate_of])]


def _rule(bitmask, rule):
    return (np.asarray(bitmask) & RULE_BITS[rule]) != 0


# ============================================================================
# registration and config
# ============================================================================

def test_stage_is_registered_train_only():
    reg = load_builtin_stages()
    spec = reg.get("qc-reference")
    assert spec.group == "qc" and spec.dirname == "qc/reference"
    assert spec.purposes == ("qc",) and not spec.reads_held_out()
    assert set(spec.requires) == {"tissue-mask", "leakage"}
    assert "qc-reference" not in splits.held_out_readers(reg)
    assert splits.registry_policy_violations(reg) == []
    for bad in ({"r3_level": 1.5}, {"hf_cutoff": 0.8}, {"knn_k": 0}, {"rule_accept": 0.0}, {"target_rows": 3}):
        with pytest.raises(ConfigError):
            QCReferenceConfig(**bad)
    assert RULE_BITS == {"R1": 1, "R2": 2, "R3": 4, "R4": 8, "R5": 16, "R6": 32}
    assert decode_bitmask(0b100001) == ["R1", "R6"] and decode_bitmask(0) == []


# ============================================================================
# the stage on the fixture
# ============================================================================

def test_stage_outputs_schemas_and_split_access(tiny, ran):
    assert ran.res.status == "ran" and (ran.d / DONE_NAME).is_file()
    meta = json.loads((ran.d / REFERENCE_NAME).read_text())
    for name in meta["files"]:
        assert (ran.d / name).is_file(), name
    done = json.loads((ran.d / DONE_NAME).read_text())
    assert set(meta["files"]) <= set(done["outputs"]) | {REFERENCE_NAME}
    assert meta["n_train_rows"] == tiny.n_rows["train"]
    assert meta["spectral_segments"] == [[0, tiny.gap_after + 1], [tiny.gap_after + 1, tiny.n_bands]]
    assert [t["name"] for t in meta["target_classes"]] == ["DCIS"]
    assert meta["folds"]["n_folds"] == 2 == ran.ref.n_folds          # one fold per train DCIS patient

    ref = ran.ref
    d = tiny.patch * tiny.patch * tiny.n_bands
    assert ref.spaces.patch_shape == (tiny.patch, tiny.patch, tiny.n_bands)
    assert ref.spaces.pca_components.shape == (ref.spaces.n_components, d) and ref.spaces.n_components <= 64
    assert len(ref.hashes) == tiny.n_rows["train"]                     # R6 hashes EVERY train row
    # the reference sample is train tissue only, every DCIS tissue row
    assert ran.mask[ref.sample["row"]].all()
    np.testing.assert_array_equal(np.sort(ref.sample["row"][ref.sample["y"] == DCIS]),
                                  np.flatnonzero((ran.y == DCIS) & ran.mask))

    th = ran.thresholds
    assert th["rule_bits"] == RULE_BITS and th["bitmask_dtype"] == "uint8"
    assert set(th["classes"]["DCIS"]["checks"]) == {c.name for c in CHECKS}
    sref = json.loads((ran.d / SET_REFERENCE_NAME).read_text())["DCIS"]
    assert {"frechet_mean_std", "frechet_pca", "mmd_pca", "wasserstein1_band_mean",
            "coverage_mean_std"} <= set(sref["ranges"])
    assert set(sref["bank_patient_share"]) == {str(p) for p in tiny.dcis_patients["train"]}

    # QC fits on train only: every access the stage made is train / "qc"
    log = [r for r in read_access_log(ran.out / ACCESS_LOG_NAME) if r["stage"] == "qc-reference"]
    assert log and all(r["split"] == "train" and r["purpose"] == "qc" and r["allowed"] for r in log)


def test_calibration_targets_met_on_real_heldout(tiny, ran):
    doc = ran.thresholds["classes"]["DCIS"]
    assert doc["ok"] and doc["problems"] == []
    assert doc["n_calibration_patients"] == 2
    assert all(doc["rule_accept"][r] >= 0.99 for r in RULES) and doc["joint_accept"] >= 0.95
    # the planted byte-identical pair is a DATA duplicate, not natural variation
    assert {e["row"] for e in doc["excluded_duplicates"]} == {tiny.duplicate_row, tiny.duplicate_of}
    # R1's out-of-range threshold is never stricter than the issue's 1%
    assert doc["checks"]["R1.out_of_range"]["threshold"] >= 0.01
    # thresholds are observed real values (R6: of the cross-capture distances)
    assert doc["checks"]["R6.nn_dist"]["reference"] == "cross_capture"

    # re-scored through the LOADED reference, fold by fold: the same scores
    # and the targets hold
    cal = np.load(ran.d / calibration_file("DCIS"))
    masks = []
    for k in range(ran.ref.n_folds):
        rows = _heldout_dcis(ran, tiny, k)
        bm, sc = apply_rules(ran.ref, ran.X[rows], DCIS, fold=k, device="cpu")
        assert bm.dtype == np.uint8
        masks.append(bm)
        sel = np.flatnonzero(cal["fold"] == k)
        sel = sel[np.isin(cal["row"][sel], rows)]
        assert len(sel) == len(rows)
        order = np.argsort(cal["row"][sel])
        for key in ("out_of_range_frac", "mahalanobis", "curvature", "lag1_autocorr", "hf_power_frac",
                    "p_class", "knn_agreement", "nn_dist"):
            np.testing.assert_allclose(sc[key], cal[key][sel][order], rtol=1e-6, err_msg=key)
    allm = np.concatenate(masks)
    summary = summarize_bitmask(allm)
    assert summary["n"] == doc["n_calibration"]
    assert all(1 - summary["rejected_by"][r] / summary["n"] >= 0.99 for r in RULES)
    assert summary["acceptance"] >= 0.95


def test_r6_duplicate_and_near_copy_are_rejected(tiny, ran):
    ref = ran.ref
    # byte-exact copies of train patches: the planted duplicate, DCIS tissue, glass
    glass = np.flatnonzero(tiny.is_glass["train"])[:2]
    rows = np.r_[tiny.duplicate_of, np.flatnonzero((ran.y == DCIS) & ran.mask)[:3], glass]
    bm, sc = apply_rules(ref, ran.X[rows], DCIS, device="cpu")
    assert _rule(bm, "R6").all() and sc["exact_match"].all()
    for i, r in enumerate(rows):
        assert bytes(ran.X[sc["exact_match_row"][i]]) == bytes(ran.X[r])
    # a near copy (sensor-level noise) is not byte-identical: the NN distance catches it
    # (not the planted pair: its real cross-capture NN distance is 0, its twin)
    src = np.setdiff1d(np.flatnonzero((ran.y == DCIS) & ran.mask), [tiny.duplicate_row, tiny.duplicate_of])[:4]
    near = ran.X[src].astype(np.float32) + np.random.default_rng(0).normal(scale=0.003, size=(4, 5, 5, 8))
    bm, sc = apply_rules(ref, near.astype(np.float32), DCIS, device="cpu")
    thr = ran.thresholds["classes"]["DCIS"]["checks"]["R6.nn_dist"]["threshold"]
    assert not sc["exact_match"].any() and (sc["nn_dist"] < thr).all() and _rule(bm, "R6").all()
    np.testing.assert_array_equal(sc["nn_row"], src)
    assert not sc["authentic"].any()


def test_fold_scoring_never_sees_the_folds_heldout_patients(tiny, ran):
    ref = ran.ref
    for k in range(ref.n_folds):
        rows = _heldout_dcis(ran, tiny, k)[:3]
        _, full = apply_rules(ref, ran.X[rows], DCIS, device="cpu")
        assert full["exact_match"].all()                         # a train patch, for the full reference
        _, sc = apply_rules(ref, ran.X[rows], DCIS, fold=k, device="cpu")
        assert not sc["exact_match"].any()                       # unseen by fold k's references
        assert not np.isin(sc["nn_patient"], ref.heldout[k]).any()
    with pytest.raises(ValueError, match="out of range"):
        apply_rules(ref, ran.X[:1], DCIS, fold=ref.n_folds)
    with pytest.raises(KeyError, match="no QC reference"):
        apply_rules(ref, ran.X[:1], 0)


def test_r1_out_of_range_and_nonfinite_after_the_float16_cast(tiny, ran):
    rows = _heldout_dcis(ran, tiny, 0)[:3]
    x = ran.X[rows].astype(np.float32)
    bm, _ = apply_rules(ran.ref, x, DCIS, fold=0, device="cpu")
    assert not _rule(bm, "R1").any()
    over = x.copy()
    over[0, 0, 0, 0] = 7.0e4             # finite in float32, +inf in the float16 export
    over[1, 2, 2, 3] = -1.0e5
    assert np.isfinite(over).all()
    nan = x[:1].copy()
    nan[0, 1, 1, 1] = np.nan
    scaled = 3.0 * x                     # every value far above train DCIS tissue
    bm, sc = apply_rules(ran.ref, np.concatenate([over[:2], nan, scaled]), DCIS, fold=0, device="cpu")
    assert _rule(bm, "R1").all()
    np.testing.assert_allclose(sc["nonfinite_frac"][:3], 1.0 / x[0].size)
    assert (sc["nonfinite_frac"][3:] == 0).all() and (sc["out_of_range_frac"][3:] == 1.0).all()


def test_r4_glass_is_rejected(tiny, ran):
    tau = ran.ref.tissue.tau_deg
    glass_rows = np.flatnonzero(tiny.is_glass["train"] & (ran.y == DCIS))[:4]
    copies = ran.X[glass_rows].astype(np.float32) * 1.01            # not byte-identical to train
    rng = np.random.default_rng(0)
    flat = GLASS_LEVEL * (1.0 + 0.02 * rng.normal(size=(3, 5, 5, 1))) * np.ones(tiny.n_bands)
    bm, sc = apply_rules(ran.ref, np.concatenate([copies, flat]).astype(np.float32), DCIS, device="cpu")
    assert _rule(bm, "R4").all() and (sc["tissue_score"] < tau).all() and sc["not_tissue"].all()
    bm, _ = apply_rules(ran.ref, ran.X[_heldout_dcis(ran, tiny, 1)], DCIS, fold=1, device="cpu")
    assert not _rule(bm, "R4").any()


def test_r3_flat_and_blurred_fixture_patches_are_rejected(tiny, ran):
    rows = np.flatnonzero((ran.y == DCIS) & ran.mask)
    mean_spec = ran.X[rows].astype(np.float32).mean(axis=(0, 1, 2))
    flat = np.broadcast_to(mean_spec, (2, 5, 5, tiny.n_bands)).astype(np.float32)
    bm, sc = apply_rules(ran.ref, flat, DCIS, device="cpu")
    assert _rule(bm, "R3").all() and np.isnan(sc["lag1_autocorr"]).all() and np.isnan(sc["hf_power_frac"]).all()
    # the fixture's texture is already smooth (see the module docstring); a
    # sigma-1.5 blur of its real held-out patches still falls below the real
    # high-frequency range. Sharp vs box/Gaussian blur: test_r3_* below.
    for k in range(ran.ref.n_folds):
        x = ran.X[_heldout_dcis(ran, tiny, k)].astype(np.float32)
        bm, _ = apply_rules(ran.ref, x, DCIS, fold=k, device="cpu")
        assert not _rule(bm, "R3").any()
        bm, sc = apply_rules(ran.ref, gaussian_filter(x, sigma=(0, 1.5, 1.5, 0), mode="reflect"), DCIS,
                             fold=k, device="cpu")
        assert _rule(bm, "R3").all() and sc["fail.R3.hf_low"].all()


def test_wrong_class_heldout_patches_are_rejected_on_the_fixture(tiny, ran):
    """Healthy and IDC patches of fold k's held-out companions, scored as DCIS
    by fold k's references, which never saw them. See the module docstring for
    why R5 alone is weak on this fixture."""
    r5_any = 0
    for k in range(ran.ref.n_folds):
        for c in (0, 2):
            rows = np.flatnonzero((ran.y == c) & ran.mask & np.isin(ran.groups, ran.ref.heldout[k]))
            assert len(rows) >= 10
            bm, _ = apply_rules(ran.ref, ran.X[rows], DCIS, fold=k, device="cpu")
            assert (bm != 0).mean() >= 0.8, (k, c)
            r5_any += int(_rule(bm, "R5").sum())
    assert r5_any > 0


def test_loaded_reference_reads_train_through_its_own_guard(tiny, ran):
    ref = QCReference.load(ran.d)                           # no train_X: opened on first use
    rows = np.flatnonzero((ran.y == DCIS) & ran.mask)[:2]
    near = ran.X[rows].astype(np.float32) * 1.001
    bm, sc = apply_rules(ref, near, DCIS, device="cpu")
    assert sc["nn_row"].tolist() == rows.tolist()
    log = read_access_log(ran.out / ACCESS_LOG_NAME)
    apply = [r for r in log if r["stage"] == "qc-apply"]
    assert apply and all(r["split"] == "train" and r["purpose"] == "qc" and r["allowed"] for r in apply)
    assert not any(r["split"] in ("val", "test") for r in log if r["stage"] in ("qc-reference", "qc-apply"))


def test_rerun_is_byte_identical(tiny, ran, tmp_path):
    res = _run_chain(tiny, tmp_path / "aug")
    want = json.loads((ran.d / DONE_NAME).read_text())["outputs"]
    got = res.done["outputs"]
    assert set(got) == set(want) and all(got[k]["sha256"] == want[k]["sha256"] for k in want)


def test_failed_calibration_is_loud_and_leaves_no_done(tiny, tmp_path, monkeypatch):
    from hsi_augment import tissue
    reg = load_builtin_stages()
    out = tmp_path / "aug"
    opts = RunOptions(data_dir=str(tiny.data_dir), out=str(out))
    for name, cfg in (("tissue-mask", TissueMaskConfig()), ("leakage", LeakageConfig())):
        run_stage(reg.get(name), cfg, opts, reg)
    # R4 rejecting every real patch: no threshold can fix a flag
    monkeypatch.setattr(tissue.TissueReference, "is_tissue", lambda self, x: np.zeros(len(x), dtype=bool))
    with pytest.raises(CalibrationError, match="R4.tissue"):
        run_stage(reg.get("qc-reference"), QCReferenceConfig(device="cpu"), opts, reg)
    d = out / STAGE_DIR
    assert not (d / DONE_NAME).exists()
    doc = json.loads((d / THRESHOLDS_NAME).read_text())["classes"]["DCIS"]
    assert not doc["ok"] and doc["rule_accept"]["R4"] == 0.0


def test_set_metrics_on_the_fixture(tiny, ran):
    rng = np.random.default_rng(0)
    rows = np.flatnonzero((ran.y == DCIS) & ran.mask & (ran.groups == 101))
    rows = rows[~np.isin(rows, [tiny.duplicate_row, tiny.duplicate_of])]    # its twin is patient 102's
    syn = (ran.X[rows].astype(np.float32) + rng.normal(scale=0.01, size=(len(rows), 5, 5, 8))).astype(np.float32)
    bm, sc = apply_rules(ran.ref, syn, DCIS, device="cpu")
    rep = set_metrics(ran.ref, syn, DCIS, scores=sc, n_boot=10, device="cpu")
    assert rep["n_synthetic"] == len(rows) and rep["n_real"] == int(((ran.y == DCIS) & ran.mask).sum())
    for name in ("wasserstein1_band_mean", "band_corr_frobenius", "class_mean_sam_deg", "frechet_mean_std",
                 "frechet_pca", "mmd_mean_std", "mmd_pca", "precision_pca", "recall_pca", "density_pca",
                 "coverage_pca", "precision_mean_std", "coverage_mean_std"):
        m = rep["metrics"][name]
        assert np.isfinite(m["value"]), name
        assert set(m["real_range"]) == {"min", "median", "max", "n_patients"}
        lo, hi = m["ci"]
        assert lo is not None and lo <= hi, name
    assert len(rep["metrics"]["wasserstein1_band_mean"]["per_band"]) == tiny.n_bands
    # all samples are copies of patient 101: the attribution histogram flags it
    att = rep["nn_attribution"]
    assert att["flagged"] and att["max_patient"] == 101 and att["max_share"] == 1.0 and rep["flags"]
    # the scores must belong to the same samples
    with pytest.raises(ValueError, match="same samples"):
        set_metrics(ran.ref, syn[:3], DCIS, scores=sc, n_boot=0, device="cpu")
    # A0 is reporting only
    a0 = {"real_features": rng.normal(size=(50, 4)), "synthetic_features": rng.normal(size=(40, 4)),
          "synthetic_pred": np.r_[np.ones(30, int), np.zeros(10, int)]}
    rep = set_metrics(ran.ref, syn, DCIS, scores=sc, n_boot=0, device="cpu", a0=a0)
    assert rep["a0"]["reporting_only"] and rep["a0"]["predicted_class_mix"] == {"0": 0.25, "1": 0.75}
    assert rep["metrics"]["frechet_pca"]["ci"] == [None, None]          # n_boot = 0: no CIs


def test_attribution_flag_threshold():
    assert not SM.attribution(np.repeat([1, 2, 3], 10), 0.4)["flagged"]
    a = SM.attribution(np.r_[np.repeat([1], 5), np.repeat([2, 3], 2), [-1, -1]], 0.4)
    assert a["flagged"] and a["max_patient"] == 1 and a["n"] == 9 and abs(a["max_share"] - 5 / 9) < 1e-12
    assert SM.attribution(np.array([-1]), 0.4)["n"] == 0


# ============================================================================
# R3 on in-test patches of known sharpness
# ============================================================================

ABS = np.array([0.70, 0.85, 0.75, 0.55, 0.40, 0.12, 0.11, 0.10])


def _sharp_patches(rng, n, p=11, sigma=0.7):
    """Tissue-like patches whose sharpness is known: Beer-Lambert spectra
    over a density field that is white noise smoothed at `sigma` px (fine
    texture), plus sensor noise. float32 NHWC."""
    pad = 4
    out = np.empty((n, p, p, len(ABS)), np.float32)
    for i in range(n):
        f = gaussian_filter(rng.normal(size=(p + 2 * pad, p + 2 * pad)), sigma)[pad:-pad, pad:-pad]
        dens = 1.0 + 0.3 * (f - f.mean()) / f.std()
        out[i] = 1.45 * np.exp(-dens[..., None] * ABS) + rng.normal(scale=0.004, size=(p, p, len(ABS)))
    return out


def _r3_fails(doc, x):
    lag1, hf = F.spatial_stats(F.as_export(x).astype(np.float32))
    return R.evaluate(doc, {"lag1_autocorr": lag1, "hf_power_frac": hf})


def test_r3_flags_blurred_and_flat_and_passes_sharp_with_calibrated_thresholds():
    rng = np.random.default_rng(0)
    real = _sharp_patches(rng, 300)                         # the "real held-out" calibration patches
    lag1, hf = F.spatial_stats(real)
    level = QCReferenceConfig().r3_level
    doc = R.calibrate({"lag1_autocorr": lag1, "hf_power_frac": hf},
                      {c.name: lag1 if c.score == "lag1_autocorr" else hf for c in R3_CHECKS},
                      {c.name: level for c in R3_CHECKS}, {}, checks=R3_CHECKS)
    assert doc["ok"] and doc["rule_accept"]["R3"] >= 0.99

    sharp = _sharp_patches(rng, 200)                        # other patches of the same process
    f_sharp = np.any(list(_r3_fails(doc, sharp).values()), axis=0)
    assert f_sharp.mean() <= 0.05
    blur_side = ("R3.autocorr_high", "R3.hf_low")
    for blurred in (gaussian_filter(sharp, sigma=(0, 1.0, 1.0, 0), mode="reflect"),
                    gaussian_filter(sharp, sigma=(0, 1.5, 1.5, 0), mode="reflect"),
                    uniform_filter(sharp, size=(1, 3, 3, 1), mode="reflect")):
        fails = _r3_fails(doc, blurred)
        assert np.any([fails[c] for c in blur_side], axis=0).all()          # every blurred copy is caught
        assert not np.any([fails[c] for c in ("R3.autocorr_low", "R3.hf_high")], axis=0).any()
    flat = np.broadcast_to(sharp.mean(axis=(1, 2), keepdims=True), sharp.shape)
    assert np.all(list(_r3_fails(doc, flat[:5]).values()), axis=0).all()     # NaN fails every check
    noise = 1.0 + 0.05 * rng.normal(size=(50, 11, 11, len(ABS)))            # no spatial correlation at all
    fails = _r3_fails(doc, noise)
    assert np.any([fails["R3.autocorr_low"], fails["R3.hf_high"]], axis=0).all()


def test_spatial_stats_are_gain_invariant():
    rng = np.random.default_rng(1)
    x = _sharp_patches(rng, 20).astype(np.float64)
    gain = rng.uniform(0.5, 2.0, size=(20, 1, 1, len(ABS)))
    for a, b in zip(F.spatial_stats(x), F.spatial_stats(x * gain)):
        np.testing.assert_allclose(a, b, rtol=1e-9)
    x[0, 0, 0, 0] = np.inf
    lag1, hf = F.spatial_stats(x[:2])
    assert np.isnan(lag1[0]) and np.isnan(hf[0]) and np.isfinite(lag1[1])


# ============================================================================
# R5 on an in-test problem whose classes are known
# ============================================================================

CLASS_ABS = np.array([[0.45, 0.55, 0.50, 0.40, 0.30, 0.08, 0.07, 0.06],
                      [0.40, 0.85, 0.45, 0.60, 0.25, 0.20, 0.06, 0.12],
                      [0.80, 0.50, 0.85, 0.35, 0.55, 0.06, 0.18, 0.05]])


def _class_patches(rng, label, n, p=5):
    """Patches of one patient of class `label`: the class's absorbance SHAPE
    over a random density (0.7-1.3, a nuisance shared by all classes) and a
    textured field, times the patient's band gain, plus noise."""
    gain = 1.0 + 0.015 * rng.normal(size=CLASS_ABS.shape[1])
    out = np.empty((n, p, p, CLASS_ABS.shape[1]), np.float32)
    for i in range(n):
        f = gaussian_filter(rng.normal(size=(p + 8, p + 8)), 0.7)[4:-4, 4:-4]
        dens = rng.uniform(0.7, 1.3) * (1.0 + 0.3 * (f - f.mean()) / f.std())
        out[i] = gain * 1.45 * np.exp(-dens[..., None] * CLASS_ABS[label]) + rng.normal(scale=0.004, size=out[i].shape)
    return out


def test_r5_rejects_wrong_class_with_cross_fitted_probe_and_knn():
    rng = np.random.default_rng(0)
    X, y, pid = [], [], []
    for c in range(3):
        for j in range(4):
            X.append(_class_patches(rng, c, 40))
            y += [c] * 40
            pid += [100 * (c + 1) + j] * 40
    X, y, pid = np.concatenate(X), np.asarray(y), np.asarray(pid)
    raw = np.asarray(F.mean_std_features(X), dtype=np.float64)
    mu, sd = raw.mean(axis=0), raw.std(axis=0)
    feats = ((raw - mu) / sd).astype(np.float32)
    cfg = QCReferenceConfig()
    # LOPO folds as the leakage stage draws them: one target patient plus one
    # companion patient of each other class held out per fold
    folds = [np.array([200 + j, 100 + j, 300 + j]) for j in range(4)]
    real, wrong = {"p_class": [], "knn_agreement": []}, {"p_class": [], "knn_agreement": []}
    for held in folds:
        train = ~np.isin(pid, held)
        probe = K.fit_probe(feats[train], y[train], seed=0)
        assert probe.converged
        for sel, dst in (((y == DCIS) & ~train, real), ((y != DCIS) & ~train, wrong)):
            dst["p_class"].append(probe.predict_proba(feats[sel], 3)[:, DCIS])
            dst["knn_agreement"].append(K.knn_label_agreement(feats[sel], feats, y, DCIS, cfg.knn_k,
                                                              bank_valid=train, device="cpu")[0])
    real = {k: np.concatenate(v) for k, v in real.items()}
    wrong = {k: np.concatenate(v) for k, v in wrong.items()}
    doc = R.calibrate(real, {c.name: real[c.score] for c in R5_CHECKS}, {c.name: cfg.r5_level for c in R5_CHECKS},
                      {}, checks=R5_CHECKS)
    assert doc["ok"] and doc["rule_accept"]["R5"] >= 0.99
    # every wrong-class patch of an unseen patient, scored as DCIS, is rejected
    fails = R.evaluate(doc, wrong)
    assert np.any(list(fails.values()), axis=0).all()
    assert fails["R5.p_class"].all() and fails["R5.knn_agreement"].all()

    # new patients never seen by any fit, against the probe on all rows
    full = K.fit_probe(feats, y, seed=0)
    new_dcis, new_idc = _class_patches(rng, DCIS, 60), _class_patches(rng, 2, 60)
    for x, want_reject in ((new_dcis, False), (new_idc, True)):
        f = ((F.mean_std_features(x) - mu) / sd).astype(np.float32)
        sc = {"p_class": full.predict_proba(f, 3)[:, DCIS],
              "knn_agreement": K.knn_label_agreement(f, feats, y, DCIS, cfg.knn_k, device="cpu")[0]}
        rejected = np.any(list(R.evaluate(doc, sc).values()), axis=0)
        assert rejected.mean() >= 0.95 if want_reject else rejected.mean() <= 0.05


@pytest.mark.filterwarnings("ignore:.*n_jobs.*:FutureWarning")      # the probe's n_jobs=-1, sklearn >= 1.8
def test_linear_probe_reproduces_sklearn():
    from scripts import shallow_probe_heldout as sp
    rng = np.random.default_rng(0)
    for n_classes in (2, 3):
        f = rng.normal(size=(300, 6)).astype(np.float32)
        y = rng.integers(0, n_classes, size=300)
        f[:, 0] += y
        probe = K.fit_probe(f, y, seed=3)
        clf = sp.fit("mean", f[:, None, None, :], y, 1500, 3)
        np.testing.assert_allclose(probe.predict_proba(f, n_classes), clf.predict_proba(f), atol=1e-10)
        back = K.LinearProbe.from_arrays(probe.to_arrays("p_"), "p_")
        np.testing.assert_array_equal(back.predict_proba(f), probe.predict_proba(f))
    with pytest.raises(ValueError, match="two classes"):
        K.fit_probe(f, np.zeros(len(f), int))


# ============================================================================
# primitives
# ============================================================================

def test_export_cast():
    x = np.full((1, 2, 2, 3), 1.0, np.float32)
    x[0, 0, 0, 0], x[0, 0, 0, 1] = 7.0e4, 6.0e4
    e = F.as_export(x)
    assert e.dtype == np.float16 and e.flags.c_contiguous
    assert np.isinf(e[0, 0, 0, 0]) and np.isfinite(e[0, 0, 0, 1])
    assert F.nonfinite_fraction(e)[0] == 1 / 12
    with pytest.raises(ValueError, match="NHWC"):
        F.as_export(np.zeros((2, 3)))
    # hashes are of the float16 export, whatever dtype comes in
    np.testing.assert_array_equal(F.patch_hashes(x), F.patch_hashes(e))


def test_histogram_quantiles_are_exact():
    rng = np.random.default_rng(0)
    v = rng.normal(0.3, 1.0, size=(400, 3, 1, 4)).astype(np.float16)
    v[0, 0, 0, :] = -0.0
    v[1, 0, 0, :] = 0.0
    counts = F.code_histogram(v)
    assert counts.shape == (4, F.N_CODES) and counts.sum() == v.size
    flat = v.reshape(-1, 4).astype(np.float64)
    for tail in (0.0, 0.0005, 0.01, 0.25):
        lo, hi = F.histogram_quantiles(counts, tail)
        np.testing.assert_array_equal(lo, np.quantile(flat, tail, axis=0, method="inverted_cdf"))
        np.testing.assert_array_equal(hi, np.quantile(flat, 1.0 - tail, axis=0, method="inverted_cdf"))
    lo, hi = F.histogram_quantiles(counts, 0.0)
    np.testing.assert_array_equal(lo, flat.min(axis=0))
    np.testing.assert_array_equal(hi, flat.max(axis=0))


def test_pca_space_matches_explicit_band_zscore():
    rng = np.random.default_rng(0)
    n, h, w, c = 400, 3, 3, 4
    base = rng.normal(size=(n, 1, 1, c)) @ rng.normal(size=(c, c))
    x = (5.0 + base + 0.3 * rng.normal(size=(n, h, w, c))) * np.array([1.0, 3.0, 0.5, 2.0])
    acc = F.CovarianceAccumulator(h * w * c)
    for a, b in F.blocks(n, 37):
        acc.update(x[a:b].reshape(b - a, -1))
    flat = x.reshape(n, -1)
    np.testing.assert_allclose(acc.cov(), np.cov(flat, rowvar=False), atol=1e-9)
    band_mean = x.reshape(-1, c).mean(axis=0)
    band_std = x.reshape(-1, c).std(axis=0)
    spaces = F.fit_feature_spaces((h, w, c), band_mean, band_std, acc, F.mean_std_features(x), 5)
    z = ((x - band_mean) / band_std).reshape(n, -1)
    vals, vecs = np.linalg.eigh(np.cov(z, rowvar=False))
    top = vecs[:, ::-1][:, :5]
    np.testing.assert_allclose(spaces.pca_explained_variance, vals[::-1][:5], rtol=1e-8)
    proj = (z - z.mean(axis=0)) @ top
    got = spaces.pca(x)
    for j in range(5):
        sign = np.sign(np.dot(got[:, j], proj[:, j]))
        np.testing.assert_allclose(got[:, j], sign * proj[:, j], atol=1e-3)
    back = F.FeatureSpaces.from_arrays(spaces.to_arrays())
    np.testing.assert_array_equal(back.pca(x), got)


def test_curvature_is_within_segments_never_across_the_gap():
    segs = spectral_segments(WAVELENGTHS)
    assert segs == [(0, 5), (5, 8)]
    smooth = np.broadcast_to(np.exp(-np.linspace(0.2, 1.0, 8)), (1, 2, 2, 8)).copy()   # log-linear
    assert F.curvature_energy(smooth, segs)[0] < 1e-6
    step = smooth.copy()
    step[..., 5:] *= 3.0                                  # a jump across the gap only
    assert F.curvature_energy(step, segs)[0] < 1e-6
    assert F.curvature_energy(step, [(0, 8)])[0] > 0.1   # it WOULD count across the gap
    jag = smooth.copy()
    jag[..., 1:4:2] *= 1.1                                # jagged within the visible segment
    assert F.curvature_energy(jag, segs)[0] > 1e-3
    shaped = np.broadcast_to(np.exp(-ABS), (1, 2, 2, 8)).copy()
    base = F.curvature_energy(shaped, segs)[0]
    tilt = 2.0 * shaped * np.exp(0.1 * np.arange(8))      # a gain and a tilt linear in the band index
    assert base > 1e-3 and F.curvature_energy(tilt, segs)[0] == pytest.approx(base, rel=1e-2)
    assert np.isnan(F.curvature_energy(smooth, [(0, 2)])[0])   # no segment of 3 bands


def test_knn_search_and_exact_rerank_match_brute_force():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(40, 6)).astype(np.float32)
    b = rng.normal(size=(300, 6)).astype(np.float32)
    b[7] = q[3]                                           # an exact copy
    valid = rng.random(300) > 0.2
    valid[7] = True
    qg, bg = rng.integers(0, 4, 40), rng.integers(0, 4, 300)
    bg[7] = (qg[3] + 1) % 4
    d_all = np.sqrt(((q[:, None, :].astype(np.float64) - b[None]) ** 2).sum(-1))
    d_all[:, ~valid] = np.inf
    d_all[qg[:, None] == bg[None, :]] = np.inf
    want = np.argsort(d_all, axis=1, kind="stable")[:, :5]
    dist, idx = M.knn_search(q, b, 5, device="cpu", block_elems=64, bank_valid=valid, query_groups=qg,
                             bank_groups=bg)
    np.testing.assert_array_equal(idx, want)
    # cdist's matrix-product form is approximate, worst near 0 (the copy scores
    # ~5e-4, not 0): the reason R6 re-ranks with the exact distance
    np.testing.assert_allclose(dist, np.take_along_axis(d_all, want, 1), atol=2e-3)
    # exact re-rank in the full space: RMS distance, exactly 0 for the copy
    nn = M.nearest_neighbours(q, lambda i: q[i], b, lambda i: b[i], candidates=5, query_chunk=7, device="cpu",
                              block_elems=64, bank_valid=valid, query_groups=qg, bank_groups=bg)
    np.testing.assert_array_equal(nn.idx, want[:, 0])
    np.testing.assert_allclose(nn.dist, d_all[np.arange(40), want[:, 0]] / np.sqrt(6), rtol=1e-6)
    assert nn.idx[3] == 7 and nn.dist[3] == 0.0
    # nothing eligible: inf / -1
    d, i = M.knn_search(q[:2], b, 3, device="cpu", bank_valid=np.zeros(300, bool))
    assert np.isinf(d).all() and (i == -1).all()


def test_hash_index_lookup_and_exclusions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 2, 2, 3)).astype(np.float16)
    patient = np.repeat([1, 2], 5)
    capture = np.repeat([10, 11, 20, 21, 22], 2)
    idx = M.HashIndex.build(F.patch_hashes(x), np.arange(100, 110), patient, capture)
    q = x[[0, 6]].copy()
    np.testing.assert_array_equal(idx.lookup(F.patch_hashes(q)), [100, 106])
    np.testing.assert_array_equal(idx.lookup(F.patch_hashes(q), exclude_patients=[2]), [100, -1])
    np.testing.assert_array_equal(idx.lookup(F.patch_hashes(q), exclude_capture=np.array([10, 0])), [-1, 106])
    off = q.copy()
    off.view(np.uint16)[0, 0, 0, 0] ^= 1                  # one ulp: no longer byte-identical
    assert idx.lookup(F.patch_hashes(off))[0] == -1
    back = M.HashIndex.from_arrays(idx.to_arrays())
    np.testing.assert_array_equal(back.lookup(F.patch_hashes(q)), [100, 106])


def test_calibration_loosens_to_targets_keeps_bounds_and_fails_on_flags():
    rng = np.random.default_rng(0)
    n = 200
    s = rng.normal(size=n)
    small = rng.uniform(0.0, 0.002, size=n)
    flag = np.zeros(n)
    checks = [R.Check("T.low", "R2", "s", "lower", "lvl"),
              R.Check("T.bounded", "R1", "small", "upper", "lvl", bound_param="b"),
              R.Check("T.flag", "R4", "flag", "flag", None)]
    scores = {"s": s, "small": small, "flag": flag}
    refs = {"T.low": s, "T.bounded": small}
    doc = R.calibrate(scores, refs, {"T.low": 0.05, "T.bounded": 0.01}, {"T.bounded": 0.01}, checks=checks)
    assert doc["ok"] and all(doc["rule_accept"][r] >= 0.99 for r in RULES)
    low = doc["checks"]["T.low"]
    assert "T.low" in doc["loosened"] and low["level"] < 0.05 and low["threshold"] in s
    assert doc["checks"]["T.bounded"]["threshold"] == 0.01            # never stricter than the bound
    # the thresholds evaluate as calibrated
    fails = R.evaluate(doc, scores)
    assert 1 - np.any(list(fails.values()), axis=0).mean() == pytest.approx(doc["joint_accept"])
    # a flag that rejects 2% of real patches cannot be loosened: not ok, and why
    flag[:4] = 1
    bad = R.calibrate(scores, refs, {"T.low": 0.05, "T.bounded": 0.01}, {"T.bounded": 0.01}, checks=checks)
    assert not bad["ok"] and any("T.flag (4)" in p for p in bad["problems"])
    # a stricter joint target loosens across rules
    flag[:] = 0
    j = R.calibrate(scores, refs, {"T.low": 0.05, "T.bounded": 0.01}, {}, checks=checks, joint_accept=1.0)
    assert j["ok"] and j["joint_accept"] == 1.0
    assert R.threshold_at(np.array([1.0, 2.0, np.nan]), "lower", 0.0) == 1.0
    assert R.check_fails(np.array([np.nan, 1.0]), "upper", 2.0).tolist() == [True, False]


def test_set_metric_primitives():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 4))
    assert SM.frechet(a, a) == pytest.approx(0.0, abs=1e-9)
    assert SM.frechet_from_moments(np.zeros(1), np.eye(1), np.ones(1), 4 * np.eye(1)) == pytest.approx(2.0)
    assert SM.corr_frobenius(a, a) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(SM.wasserstein1_per_band(a + 0.5, a), 0.5, atol=1e-12)
    ind = SM.prdc_indicators(a, a, 5, device="cpu")
    assert all(ind[k].mean() == 1.0 for k in ("precision", "recall", "coverage"))
    b = rng.normal(size=(300, 4))
    sigma = SM.median_bandwidth(a, rng)
    same, shifted = SM.mmd2(a, b, sigma), SM.mmd2(a + 1.0, b, sigma)
    assert abs(same) < 0.01 < shifted
    far = SM.prdc_indicators(a, a + 10.0, 5, device="cpu")
    assert all(far[k].mean() == 0.0 for k in ("precision", "recall", "coverage", "density"))
