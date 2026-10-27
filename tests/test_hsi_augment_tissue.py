# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_tissue.py
================================
The `tissue-mask` stage (#13, Plan §Phase 2 items 4-7) on the tiny fixture:

  * the scale-invariant score separates the fixture's flat glass from its
    absorbing tissue, and the fitted mask matches `is_glass`;
  * tau is fitted on train only: the access log holds train/"fit" reads only,
    and rewriting val/test leaves s_glass and tau bit-identical;
  * the grid edge-continuity check passes on the fixture, fails on a copy whose
    patches are shuffled within each capture (no done.json), and catches a
    wrong grid width through the vertical pairs;
  * `dcis_copy_mask.npy` is bool with one entry per train row and is accepted
    by `training.synthetic_mix.plan_minority_with_synthetic`;
  * the glass rule decides correctly above, at and below 10%;
  * the figures are written; the CLI runs, skips and is in `preprocess`;
  * an optional real-data smoke test (HSI_AUGMENT_REAL_DATA) on 3 captures.

CPU only, a few seconds in total.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hsi_augment import tissue
from hsi_augment.io import read_access_log
from hsi_augment.registry import load_builtin_stages
from hsi_augment.stage import DONE_NAME, RunOptions, StageError, run_stage
from hsi_augment.stages.tissue_mask import TissueMaskConfig
from tests.hsi_augment_fixtures import make_tiny_build

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = Path("prep") / "tissue"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _run(build, out, cfg=None, **opts):
    reg = load_builtin_stages()
    return run_stage(reg.get("tissue-mask"), cfg or TissueMaskConfig(),
                     RunOptions(data_dir=str(build.data_dir), out=str(out), **opts), reg)


def _rewrite_manifest(data_dir: Path) -> None:
    """After editing arrays of a fixture build: a manifest that matches them."""
    from training.npy_atomic import write_dataset_manifest
    m = json.loads((data_dir / "dataset_manifest.json").read_text())
    write_dataset_manifest(str(data_dir), list(m["files"]), sha256=True, extra=m.get("extra"))


def _shuffle_within_captures(data_dir: Path, seed: int = 0) -> None:
    """X_train with every capture's rows permuted: same patches, same labels,
    same capture runs, so only the derived (r, c) become wrong."""
    X = np.load(data_dir / "X_train.npy")
    caps = np.load(data_dir / "captures_train.npy")
    rng = np.random.default_rng(seed)
    for cap in np.unique(caps):
        rows = np.flatnonzero(caps == cap)
        X[rows] = X[rows[rng.permutation(len(rows))]]
    np.save(data_dir / "X_train.npy", X)
    _rewrite_manifest(data_dir)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def ran(tiny, tmp_path_factory):
    """The stage run once with defaults; (result, out root, stage dir)."""
    out = tmp_path_factory.mktemp("aug")
    res = _run(tiny, out)
    return res, out, out / STAGE_DIR


# ============================================================================
# the score and the mask
# ============================================================================

def test_score_separates_glass_from_tissue(tiny, ran):
    res, _, d = ran
    assert res.status == "ran" and (d / DONE_NAME).is_file()
    n = tiny.n_rows["train"]
    score = np.load(d / tissue.SCORE_NAME)
    mask = np.load(d / tissue.MASK_NAME)
    assert score.dtype == np.float16 and score.shape == (n,) and np.isfinite(score).all()
    assert mask.dtype == np.bool_ and mask.shape == (n,)
    s_glass = np.load(d / tissue.S_GLASS_NAME)
    assert s_glass.shape == (tiny.n_bands,) and abs(np.linalg.norm(s_glass) - 1.0) < 1e-12

    tau = json.loads((d / tissue.TAU_NAME).read_text())
    glass = tiny.is_glass["train"]
    t = score.astype(np.float64)
    # flat glass is within a degree of s_glass, absorbing tissue several degrees away
    assert t[glass].max() < tau["tau_deg"] < t[~glass].min()
    assert t[glass].max() < 1.5 and t[~glass].min() > 3.0
    bacc = 0.5 * (mask[~glass].mean() + (~mask[glass]).mean())
    assert bacc >= 0.98
    assert tau["method_used"] == "gmm" and tau["fallback_reason"] is None
    assert tau["complete"] is True and tau["n_scored_rows"] == n
    # the saved mask is exactly the documented rule applied to the saved float16 scores
    np.testing.assert_array_equal(mask, tissue.apply_threshold(score, tau["tau_deg"]))
    # consumers (#28 QC, #33 report) re-score patches with the train reference
    ref = tissue.load_reference(d)
    X = np.load(tiny.data_dir / "X_train.npy")
    np.testing.assert_array_equal(ref.is_tissue(X), mask)
    # the secondary feature is recorded, summarised, and not thresholded
    cv = np.load(d / tissue.CV_NAME)
    assert cv.dtype == np.float16 and cv.shape == (n,)
    assert tau["spatial_cv"]["thresholded"] is False and tau["spatial_cv"]["auc_tissue_vs_glass"] > 0.5


def test_score_is_scale_invariant_and_accurate_for_small_angles(tiny):
    X = np.load(tiny.data_dir / "X_train.npy").astype(np.float32)
    s = tissue.unit(np.ones(tiny.n_bands))
    a = tissue.tissue_score(X, s)
    gains = np.random.default_rng(0).uniform(0.3, 3.0, size=(len(X), 1, 1, 1)).astype(np.float32)
    np.testing.assert_allclose(tissue.tissue_score(X * gains, s), a, atol=1e-3)
    # atan2 form: a 0.01 deg angle is resolved where arccos(cos) would round to ~0
    ref = np.array([1.0, 0.0, 0.0])
    ang = np.radians(0.01)
    v = np.array([[np.cos(ang), np.sin(ang), 0.0], [0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]])
    out = tissue.sam_deg(v, ref)
    assert abs(out[0] - 0.01) < 1e-9 and np.isnan(out[1]) and np.isnan(out[2])
    assert not tissue.apply_threshold(np.array([np.nan], np.float16), 1.0)[0]


def test_glass_reference_is_robust_to_captures_without_glass():
    rng = np.random.default_rng(1)
    glass = tissue.unit(np.linspace(1.0, 1.1, 8))
    cands = tissue.unit(glass + 0.002 * rng.normal(size=(9, 8)))
    tissue_like = tissue.unit(np.exp(-np.linspace(0.8, 0.1, 8)))
    s = tissue.glass_reference(np.vstack([cands, np.tile(tissue_like, (4, 1))]))   # 4 of 13 captures: no glass
    assert tissue.sam_deg(s[None], glass)[0] < 0.5
    # the top 1% of a capture: at least one patch, deterministic on ties
    assert tissue.top_k(20, 0.01) == 1 and tissue.top_k(4914, 0.01) == 50
    m = np.vstack([np.full(8, 2.0), np.full(8, 1.0), np.full(8, 2.0)])
    u, k, b = tissue.capture_glass_candidate(m, m.mean(1), 0.5)
    assert k == 2 and b == 2.0 and np.allclose(u, tissue.unit(np.ones(8)))


# ============================================================================
# the threshold
# ============================================================================

def test_gmm_crossing_matches_sklearn_posteriors():
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(-1.0, 0.3, 3000), rng.normal(2.0, 0.6, 9000)])
    gm = GaussianMixture(2, random_state=0).fit(v[:, None])
    x = tissue.gmm_crossing(gm.weights_, gm.means_.ravel(), gm.covariances_.ravel())
    hi = int(np.argmax(gm.means_.ravel()))
    p = gm.predict_proba(np.array([[x - 1e-4], [x], [x + 1e-4]]))[:, hi]
    assert p[0] < 0.5 < p[2] and abs(p[1] - 0.5) < 1e-3
    assert tissue.gmm_crossing([0.5, 0.5], [0.0, 0.0], [1.0, 1.0]) is None       # no separation
    assert tissue.gmm_crossing([0.999, 0.001], [0.0, 0.1], [1.0, 1.0]) is None   # one side dominates


def test_threshold_fit_methods_and_fallback(monkeypatch):
    rng = np.random.default_rng(3)
    t = np.exp(np.concatenate([rng.normal(np.log(0.5), 0.2, 400), rng.normal(np.log(12.0), 0.3, 1600)]))
    gmm = tissue.fit_threshold(t, seed=0)
    assert gmm.method_used == "gmm" and 0.5 < gmm.tau_deg < 12.0 and gmm.gmm["converged"]
    otsu = tissue.fit_threshold(t, method="otsu")
    assert otsu.method_used == "otsu" and otsu.gmm is None and 0.5 < otsu.tau_deg < 12.0
    assert otsu.tau_deg == pytest.approx(gmm.otsu["tau_deg"])
    # the GMM failing to cross falls back to Otsu, and says why
    monkeypatch.setattr(tissue, "gmm_crossing", lambda *a, **k: None)
    fb = tissue.fit_threshold(t, seed=0)
    assert fb.method_used == "otsu" and fb.method_requested == "gmm" and "do not cross" in fb.fallback_reason
    assert fb.tau_deg == pytest.approx(otsu.tau_deg)
    # Otsu sits inside an empty gap, not on the edge of the lower mode
    thr, _ = tissue.otsu_threshold(np.r_[np.zeros(50), np.ones(50)], bins=100)
    assert 0.3 < thr < 0.7


def test_stratified_sample_is_proportional_exact_and_deterministic():
    strata = np.repeat(np.arange(5), [100, 200, 300, 50, 350])
    valid = np.ones(len(strata), dtype=bool)
    valid[:10] = False
    a = tissue.stratified_sample(strata, valid, 99, np.random.default_rng(0))
    b = tissue.stratified_sample(strata, valid, 99, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)
    assert len(a) == 99 and len(np.unique(a)) == 99 and valid[a].all() and np.all(np.diff(a) > 0)
    counts = np.bincount(strata[a], minlength=5)
    expect = 99 * np.bincount(strata[valid], minlength=5) / valid.sum()
    assert np.all(np.abs(counts - expect) < 1.0)
    every = tissue.stratified_sample(strata, valid, 10_000, np.random.default_rng(0))
    np.testing.assert_array_equal(every, np.flatnonzero(valid))


# ============================================================================
# train only
# ============================================================================

def test_tau_is_fitted_on_train_only(tiny, ran, tmp_path):
    _, out, d = ran
    rec = read_access_log(out / "split_access_log.jsonl")
    assert rec and {(r["stage"], r["split"], r["purpose"], r["allowed"]) for r in rec} == \
        {("tissue-mask", "train", "fit", True)}
    # the stage cannot even ask for val: it declared the train-only purpose "fit" alone
    assert load_builtin_stages().get("tissue-mask").purposes == ("fit",)

    # a build with the same train split and different val/test gives the same fit
    other = make_tiny_build(tmp_path / "b")
    for split in ("val", "test"):
        X = np.load(other.data_dir / f"X_{split}.npy")
        np.save(other.data_dir / f"X_{split}.npy", (X.astype(np.float32) * 3.0 + 0.5).astype(np.float16))
    _rewrite_manifest(other.data_dir)
    _run(other, tmp_path / "aug")
    d2 = tmp_path / "aug" / STAGE_DIR
    np.testing.assert_array_equal(np.load(d2 / tissue.S_GLASS_NAME), np.load(d / tissue.S_GLASS_NAME))
    for name in (tissue.SCORE_NAME, tissue.MASK_NAME, tissue.COPY_MASK_NAME):
        assert (d2 / name).read_bytes() == (d / name).read_bytes()
    t1 = json.loads((d / tissue.TAU_NAME).read_text())
    t2 = json.loads((d2 / tissue.TAU_NAME).read_text())
    assert t1["tau_deg"] == t2["tau_deg"] and t1["gmm"] == t2["gmm"]


# ============================================================================
# the grid check
# ============================================================================

def test_grid_check_passes_on_the_fixture(ran):
    _, _, d = ran
    g = json.loads((d / tissue.GRID_CHECK_NAME).read_text())
    assert g["passed"] is True and g["pass_fraction"] >= 0.95 and g["n_captures_tested"] == 12
    assert g["vertical"] is True and g["failed_captures"] == []
    pooled = g["pooled"]
    assert pooled["median_h"] > 0.8 and pooled["median_h"] > pooled["shuffled_median_h"] + 0.4
    assert pooled["median_v"] > 0.8 and pooled["median_v"] > pooled["shuffled_median_v"] + 0.4
    for c in g["per_capture"]:
        assert c["grid"] == [4, 5] and c["n_pairs_h"] == 16 and c["n_pairs_v"] == 15
        assert c["n_shuffles"] * 15 >= g["baseline_pairs"]


def test_grid_check_fails_on_a_shuffled_copy(tmp_path):
    build = make_tiny_build(tmp_path / "shuf")
    _shuffle_within_captures(build.data_dir)
    out = tmp_path / "aug"
    with pytest.raises(StageError, match="grid edge-continuity check FAILED"):
        _run(build, out)
    d = out / STAGE_DIR
    assert not (d / DONE_NAME).exists() and (d / "failed.json").is_file()
    g = json.loads((d / tissue.GRID_CHECK_NAME).read_text())
    assert g["passed"] is False and g["pass_fraction"] < 0.5
    assert abs(g["pooled"]["median_h"] - g["pooled"]["shuffled_median_h"]) < 0.3

    # grid_strict off: the stage finishes and records the failure
    res = _run(build, out, TissueMaskConfig(grid_strict=False))
    assert res.status == "ran" and res.done["summary"]["grid_passed"] is False


def test_grid_check_catches_a_wrong_grid_width(tiny):
    """Horizontal pairs survive a wrong width (they are consecutive rows);
    the vertical pairs do not. Width 4 instead of 5 makes every "vertical"
    pair a diagonal one; on the fixture's 5-px patches and smooth texture a
    diagonal neighbour still correlates a little, so a few captures pass,
    but far fewer than the stage's 95%."""
    X = np.load(tiny.data_dir / "X_train.npy")
    caps = np.load(tiny.data_dir / "captures_train.npy")
    right = wrong = 0
    for cap in np.unique(caps):
        rows = np.flatnonzero(caps == cap)
        e = tissue.patch_edges(X[rows])
        ordinal = np.arange(len(rows))
        res, _ = tissue.capture_grid_check(e, ordinal // 5, ordinal % 5, (4, 5), np.random.default_rng(0))
        right += res["passed"]
        bad, _ = tissue.capture_grid_check(e, ordinal // 4, ordinal % 4, (5, 4), np.random.default_rng(0))
        wrong += bad["passed"]
        assert bad["delta_v"] < res["delta_v"]
    assert right == 12 and wrong <= 6
    with pytest.raises(ValueError, match="exactly once"):
        tissue.capture_grid_check(e, np.zeros(20, int), np.arange(20) % 5, (4, 5), np.random.default_rng(0))


# ============================================================================
# the glass rule and the copy mask
# ============================================================================

def _rule_case(glass_per_patient, n_per_patient=100):
    """Minority (class 1) rows of len(glass_per_patient) patients, each with the
    given number of glass rows, plus 50 class-0 rows that are all glass."""
    tissue_rows, y, g = [], [], []
    for p, k in enumerate(glass_per_patient):
        tissue_rows += [False] * k + [True] * (n_per_patient - k)
        y += [1] * n_per_patient
        g += [p] * n_per_patient
    tissue_rows += [False] * 50
    y += [0] * 50
    g += [99] * 50
    return np.array(tissue_rows), np.array(y), np.array(g)


@pytest.mark.parametrize("glass, use", [((5, 5), False), ((15, 15), True), ((10, 10), False),
                                        ((0, 21), True), ((0, 20), False), ((30, 0), True)])
def test_glass_rule_decision(glass, use):
    m, y, g = _rule_case(glass)
    rule = tissue.glass_rule(m, y, g, (1,), ("healthy", "DCIS", "IDC"), threshold=0.10)
    assert rule["use_copy_mask"] is use
    assert rule["overall"]["glass_fraction"] == pytest.approx(sum(glass) / (100 * len(glass)))
    assert rule["per_patient"]["0"]["glass_fraction"] == pytest.approx(glass[0] / 100)
    assert rule["threshold"] == 0.10 and rule["minority_class_names"] == ["DCIS"]
    assert rule["n_copyable_minority_rows"] == 100 * len(glass) - sum(glass)
    assert "99" not in rule["per_patient"]          # the all-glass class-0 rows do not count


def test_glass_rule_without_minority_rows():
    rule = tissue.glass_rule(np.ones(4, bool), np.zeros(4, int), np.zeros(4, int), (1,), ("a", "b"))
    assert rule["overall"]["glass_fraction"] is None and rule["use_copy_mask"] is False and rule["note"]


def test_copy_mask_and_glass_rule_outputs(tiny, ran):
    _, _, d = ran
    n = tiny.n_rows["train"]
    cm = np.load(d / tissue.COPY_MASK_NAME)
    assert cm.dtype == np.bool_ and cm.shape == (n,)
    np.testing.assert_array_equal(cm, np.load(d / tissue.MASK_NAME))
    rule = json.loads((d / tissue.GLASS_RULE_NAME).read_text())
    y = np.load(tiny.data_dir / "y_train.npy")
    dcis = y == 1
    truth = tiny.is_glass["train"][dcis].mean()
    assert rule["overall"]["glass_fraction"] == pytest.approx(truth)       # mask == ground truth here
    assert rule["use_copy_mask"] is bool(truth > 0.10) and set(rule["per_patient"]) == {"101", "102"}
    assert rule["copy_mask_file"] == tissue.COPY_MASK_NAME and rule["complete"] is True

    # the consumer (#8): train.py --minority_copy_mask goes through this function
    from training.synthetic_mix import plan_minority_with_synthetic
    plan = plan_minority_with_synthetic(y, 0, 3, ("healthy", "DCIS", "IDC"), "DCIS", 3.0, seed=0, copy_mask=cm)
    extra = plan.indices[n:]
    assert len(extra) == 2 * int(dcis.sum()) and cm[extra].all() and (y[extra] == 1).all()


# ============================================================================
# figures, mosaics, subset runs
# ============================================================================

def test_figures_are_written(ran):
    _, _, d = ran
    done = json.loads((d / DONE_NAME).read_text())
    figs = {Path(k).name for k in done["outputs"] if k.startswith("figures/")}
    assert {"tissue_maps_healthy.png", "tissue_maps_dcis.png", "tissue_maps_idc.png", "tissue_score_hist.png",
            "s_glass_spectrum.png", "glass_rule.png", "grid_check.png"} <= figs
    for f in figs:
        p = d / "figures" / f
        assert p.stat().st_size > 5_000 and p.read_bytes()[:8] == PNG_MAGIC
    info = json.loads((d / tissue.CAPTURE_TISSUE_NAME).read_text())
    assert info["rgb_note"] and len(info["rgb_bands"]) == 3 and len(info["captures"]) == 12


def test_mosaic_selection_covers_every_minority_patient():
    caps = []
    for i, (label, patient, gf) in enumerate([(1, 25, 0.1), (1, 25, 0.3), (1, 45, 0.2), (1, 85, 0.0),
                                              (1, 107, 0.5), (1, 152, 0.05), (0, 1, 0.1), (0, 2, 0.4),
                                              (0, 3, 0.2), (0, 4, 0.0), (0, 5, 0.9), (2, 7, 0.3)]):
        caps.append({"capture": i, "label": label, "patient": patient, "glass_fraction": gf})
    sel = tissue.select_mosaic_captures(caps, 3, (1,))
    assert sel[1] == [1, 2, 3, 4, 5]                       # one per DCIS patient, its most-glass capture
    assert sel[0] == [9, 8, 10]                            # least glass (0.0), median (0.2), most (0.9)
    assert sel[2] == [11]                                  # as many as exist


def test_max_captures_subset_run(tiny, tmp_path):
    res = _run(tiny, tmp_path, TissueMaskConfig(max_captures=3))
    d = tmp_path / STAGE_DIR
    assert res.done["summary"]["complete"] is False
    n = tiny.n_rows["train"]
    score = np.load(d / tissue.SCORE_NAME)
    mask = np.load(d / tissue.MASK_NAME)
    assert score.shape == (n,) and mask.shape == (n,) and np.load(d / tissue.COPY_MASK_NAME).shape == (n,)
    done_rows = np.isfinite(score)
    assert done_rows.sum() == 3 * tiny.grid_rows * tiny.grid_cols and not mask[~done_rows].any()
    y = np.load(tiny.data_dir / "y_train.npy")
    assert set(np.unique(y[done_rows])) == {0, 1, 2}       # round-robin over the classes
    for name in (tissue.TAU_NAME, tissue.GLASS_RULE_NAME, tissue.CAPTURE_TISSUE_NAME):
        assert json.loads((d / name).read_text())["complete"] is False


def test_config_validation():
    from hsi_augment.config import ConfigError
    for bad in ({"top_fraction": 0.0}, {"sample_rows": 0}, {"method": "kmeans"}, {"glass_rule_threshold": 1.5},
                {"grid_min_pass_fraction": 0.0}, {"max_captures": 0}, {"rgb_nm": (640.0, -1.0, 460.0)}):
        with pytest.raises(ConfigError):
            TissueMaskConfig(**bad)
    assert TissueMaskConfig().sample_rows == 500_000 and TissueMaskConfig().glass_rule_threshold == 0.10


# ============================================================================
# CLI
# ============================================================================

def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)


def test_cli_runs_skips_and_belongs_to_preprocess(tiny, tmp_path):
    r = _cli("tissue-mask", "--help")
    assert r.returncode == 0 and "--sample_rows" in r.stdout and "--grid_min_delta" in r.stdout
    out = tmp_path / "aug"
    base = ["tissue-mask", "--data_dir", tiny.data_dir, "--out", out, "--sample_rows", "200"]
    r = _cli(*base)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[tissue-mask] tau = " in r.stdout and "tissue-mask: ran" in r.stdout
    done = json.loads((out / STAGE_DIR / DONE_NAME).read_text())
    assert done["config"]["sample_rows"] == 200 and done["summary"]["grid_passed"] is True
    assert json.loads((out / STAGE_DIR / tissue.TAU_NAME).read_text())["sample"]["n_rows"] == 200
    r = _cli(*base)
    assert r.returncode == 0 and "tissue-mask: skipped" in r.stdout
    r = _cli("preprocess", "--data_dir", tiny.data_dir, "--out", tmp_path / "aug2")
    assert r.returncode == 0 and "tissue-mask: ran" in r.stdout, r.stdout + r.stderr


# ============================================================================
# real data (opt-in)
# ============================================================================

REAL_DATA = os.environ.get("HSI_AUGMENT_REAL_DATA")


@pytest.mark.skipif(not REAL_DATA, reason="set HSI_AUGMENT_REAL_DATA to the trainsel hsi dir to run")
def test_real_data_smoke(tmp_path):
    """3 captures (one per class): two passes over ~15k rows (~115 MB of X
    read twice), never the whole X_train. Checks shapes and that the derived
    grid of real captures passes."""
    from hsi_augment.adapters import get_adapter
    real = get_adapter("histologyhsi_bc", REAL_DATA)
    reg = load_builtin_stages()
    cfg = TissueMaskConfig(max_captures=3, sample_rows=5_000, mosaics_per_class=1, grid_strict=False)
    res = run_stage(reg.get("tissue-mask"), cfg, RunOptions(data_dir=REAL_DATA, out=str(tmp_path)), reg)
    d = tmp_path / STAGE_DIR
    n = real.n_rows("train")
    score = np.load(d / tissue.SCORE_NAME, mmap_mode="r")
    assert score.dtype == np.float16 and score.shape == (n,)
    assert np.load(d / tissue.COPY_MASK_NAME, mmap_mode="r").shape == (n,)
    s = res.done["summary"]
    assert s["complete"] is False and s["n_scored_rows"] == 3 * 4914
    tau = json.loads((d / tissue.TAU_NAME).read_text())
    assert np.isfinite(tau["tau_deg"]) and tau["spectral_gaps_after_band"] == [17]
    g = json.loads((d / tissue.GRID_CHECK_NAME).read_text())
    assert g["n_captures_tested"] == 3 and all(c["grid"] == [54, 91] for c in g["per_capture"])
    assert g["pass_fraction"] >= 2 / 3, g["per_capture"]
