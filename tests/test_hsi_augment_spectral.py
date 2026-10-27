# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_spectral.py
==================================
Phase 2a (#12): `augment_pipeline.py spectral-audit` and `hsi_augment/preprocess.py`.

  * the exact float64 chunked stats equal numpy, whatever the chunking;
  * train.py's sample estimate is replicated bit for bit, agrees within
    sampling error, and the predicted standard errors match the empirical
    spread of the estimate over many seeds;
  * sigma_b recovers a known noise level from smooth signal + Gaussian noise;
  * the clipped-gain capture of the fixture is flagged;
  * saturation, zero variance, zeros, negatives and NaN are reported on
    constructed variants of the fixture, never dropped;
  * the correlation matrix is the seeded sample's;
  * only the train split is read, for purpose `fit`;
  * the CLI runs, skips when done, and is deterministic.

CPU only, on `tests/hsi_augment_fixtures.make_tiny_build` (variants are built
here; the fixture module is not edited). The real-data smoke test runs only
when `HSI_AUGMENT_REAL_DATA` points at a build's `hsi` dir, and reads 2,000 rows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import hsi_augment.io as hio
from hsi_augment import preprocess as pp
from hsi_augment.io import RunningBandStats, read_access_log
from hsi_augment.registry import load_builtin_stages
from hsi_augment.stage import DONE_NAME, RunOptions, run_stage, stage_seed_sequence
from hsi_augment.stages.spectral_audit import SpectralAuditConfig
from tests.hsi_augment_fixtures import SUSPECT_CAPTURE_KEY, make_tiny_build

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE = "spectral-audit"
STAGE_DIR = Path("prep") / "spectral"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


def _run(data_dir, out, **cfg):
    reg = load_builtin_stages()
    opts = RunOptions(data_dir=str(data_dir), out=str(out))
    return run_stage(reg.get(STAGE), SpectralAuditConfig(**cfg), opts, reg)


def _outputs(out):
    d = Path(out) / STAGE_DIR
    load = lambda name: json.loads((d / name).read_text())  # noqa: E731
    return (load(pp.NORM_STATS_NAME), load(pp.BAND_REPORT_NAME), load(pp.CALIBRATION_REPORT_NAME),
            np.load(d / pp.NOISE_SIGMA_NAME))


def _variant(root, edit):
    """A fresh tiny build whose X_train is replaced by `edit(X_train float16 copy)`."""
    t = make_tiny_build(root)
    X = np.load(t.data_dir / "X_train.npy").copy()
    X = edit(X)
    np.save(t.data_dir / "X_train.npy", X)
    return t, X


def _kinds(problems):
    return {p["kind"]: p for p in problems}


# ============================================================================
# exact stats
# ============================================================================

def test_norm_stats_equal_numpy_whatever_the_chunking(tiny, tmp_path, monkeypatch):
    X = np.load(tiny.data_dir / "X_train.npy")
    ref = X.astype(np.float64).reshape(-1, tiny.n_bands)
    results = []
    for i, chunk_rows in enumerate((7, 100_000)):
        _run(tiny.data_dir, tmp_path / f"o{i}", chunk_rows=chunk_rows, compare_sample_cap=60)
        results.append(_outputs(tmp_path / f"o{i}"))
    monkeypatch.setattr(hio, "STATS_BLOCK_VALUES", 200)      # one row per float64 block
    _run(tiny.data_dir, tmp_path / "o2", chunk_rows=13, compare_sample_cap=60)
    results.append(_outputs(tmp_path / "o2"))

    for norm, band, _, _ in results:
        assert norm["complete"] and norm["rows_scanned"] == norm["rows_total"] == tiny.n_rows["train"]
        np.testing.assert_allclose(norm["mean"], ref.mean(0), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(norm["var"], ref.var(0), rtol=1e-10)
        np.testing.assert_allclose(norm["std"], ref.std(0), rtol=1e-10)
        np.testing.assert_array_equal(norm["min"], ref.min(0))
        np.testing.assert_array_equal(norm["max"], ref.max(0))
        assert norm["count"] == [float(ref.shape[0])] * tiny.n_bands and norm["n_nonfinite"] == [0] * tiny.n_bands
        again = RunningBandStats.from_dict(norm["running_stats"])
        np.testing.assert_array_equal(again.mean, norm["mean"])
        assert norm["units"]["corpus_median"] == 8640.0 and "NOT reflectance" in norm["units"]["note"]
        assert band["problem_counts"]["error"] == 0
    # the chunking changes nothing but rounding; the samples are identical
    first = results[0][0]["sample_estimate"]["seeds"]
    for norm, band, _, sigma in results[1:]:
        np.testing.assert_allclose(norm["mean"], results[0][0]["mean"], rtol=1e-13)
        np.testing.assert_array_equal(sigma, results[0][3])
        for seed, est in norm["sample_estimate"]["seeds"].items():
            assert (est["mean"], est["std"]) == (first[seed]["mean"], first[seed]["std"])


def test_row_moments_and_running_covariance_equal_numpy():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(500, 4)), rng.normal(size=(500, 4)) ** 2
    valid = rng.random((500, 4)) > 0.1
    rm = pp.RowMoments(4)
    for a in range(0, 500, 37):
        rm.update(u[a:a + 37], v[a:a + 37], valid[a:a + 37])
    for b in range(4):
        uu, vv = u[valid[:, b], b], v[valid[:, b], b]
        assert rm.count[b] == uu.size
        np.testing.assert_allclose([rm.mean_u[b], rm.mean_v[b]], [uu.mean(), vv.mean()], rtol=1e-12)
        np.testing.assert_allclose(rm.m_uu[b], ((uu - uu.mean()) ** 2).sum(), rtol=1e-10)
        np.testing.assert_allclose(rm.m_uv[b], ((uu - uu.mean()) * (vv - vv.mean())).sum(), rtol=1e-10)

    z = rng.normal(size=(1000, 5)) @ rng.normal(size=(5, 5))
    z[[3, 400], [1, 4]] = [np.nan, np.inf]
    cov = pp.RunningCovariance(5)
    for a in range(0, 1000, 77):
        cov.update(z[a:a + 77])
    ok = np.isfinite(z).all(1)
    assert cov.count == ok.sum() and cov.n_dropped == 2
    np.testing.assert_allclose(cov.cov(ddof=1), np.cov(z[ok].T), rtol=1e-10)
    np.testing.assert_allclose(cov.corr(), np.corrcoef(z[ok].T), rtol=1e-10)


# ============================================================================
# train.py's 5,000-row estimate
# ============================================================================

def test_train_py_estimate_is_replicated_and_agrees_within_sampling_error(tiny, tmp_path):
    from training.npy_data import compute_global_channel_stats
    x_path = str(tiny.data_dir / "X_train.npy")
    _run(tiny.data_dir, tmp_path, compare_sample_cap=60, compare_seeds=(1, 7, 13, 23, 42))
    norm, _, _, _ = _outputs(tmp_path)
    est = norm["sample_estimate"]
    assert est["sample_rows"] == 60 and est["population_rows"] == tiny.n_rows["train"]
    assert not est["exhaustive"] and est["replicates_train_py"]
    for seed in (1, 7, 13, 23, 42):
        mean, std = compute_global_channel_stats(x_path, sample_cap=60, seed=seed)
        got = est["seeds"][str(seed)]
        np.testing.assert_array_equal(np.float32(got["mean"]), mean)     # bit for bit
        np.testing.assert_array_equal(np.float32(got["std"]), std)
        np.testing.assert_allclose(got["diff_mean"], np.float64(mean) - norm["mean"], rtol=1e-12)
    # the acceptance item: within sampling error
    assert est["agree"] and est["max_abs_z"] <= est["z_tol"] and est["n_z"] == 5 * 2 * tiny.n_bands

    # The standard errors are the right ones: the spread of the estimate over
    # many seeds matches them (cluster SE of a row sample, without replacement).
    X = np.load(x_path, mmap_mode="r")
    draws = [pp.train_py_sample_stats(X, 60, s) for s in range(600)]
    emp_mean = np.std([d[0] for d in draws], axis=0)
    emp_std = np.std([d[1] for d in draws], axis=0)
    ratio_mean = emp_mean / np.asarray(est["se_mean"])
    ratio_std = emp_std / np.asarray(est["se_std"])
    assert np.all((ratio_mean > 0.85) & (ratio_mean < 1.15)), ratio_mean
    assert np.all((ratio_std > 0.8) & (ratio_std < 1.2)), ratio_std
    # ... and a pixel-level (i.i.d.) SE would be far too small: rows are clusters
    naive = np.asarray(norm["std"]) / np.sqrt(60 * tiny.patch ** 2)
    assert np.all(emp_mean > 2 * naive)


def test_exhaustive_estimate_and_disagreement_are_reported(tiny, tmp_path):
    _run(tiny.data_dir, tmp_path, compare_sample_cap=5000, compare_seeds=(42,))   # cap > rows: every row
    est = _outputs(tmp_path)[0]["sample_estimate"]
    assert est["exhaustive"] and est["agree"] and est["seeds"]["42"]["within_float32_tolerance"]
    assert est["max_abs_z"] is None and est["se_mean"] == [0.0] * tiny.n_bands

    mean = np.ones(3)
    cmp = pp.compare_sample_estimates(mean, np.ones(3), np.full(3, 0.01), np.full(3, 0.01),
                                      {0: (mean + [0.0, 0.05, 0.0], np.ones(3))},
                                      n_sample=10, n_rows=100, z_tol=4.0)
    assert not cmp["agree"] and cmp["max_abs_z"] == pytest.approx(5.0)


# ============================================================================
# noise sigma_b
# ============================================================================

def _smooth_plus_noise(rng, n, P, sigma, level=1.0):
    """Per patch and band: a steep random ramp plus a faint curvature (signal),
    plus Gaussian noise of std `sigma[b]`."""
    C = len(sigma)
    i, j = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    a = level + 0.2 * rng.random((n, 1, 1, C))
    gi, gj = rng.normal(scale=0.05, size=(2, n, 1, 1, C))
    curv = rng.uniform(-2e-5, 2e-5, size=(n, 1, 1, C))
    signal = a + gi * i[None, :, :, None] + gj * j[None, :, :, None] + curv * (i ** 2 + j ** 2)[None, :, :, None]
    return signal + rng.normal(size=signal.shape) * np.asarray(sigma), signal


def test_noise_sigma_recovers_a_known_noise_level():
    rng = np.random.default_rng(3)
    sigma = np.array([0.002, 0.005, 0.01, 0.03])
    x, signal = _smooth_plus_noise(rng, 4000, 11, sigma)
    got = pp.laplacian_noise_sigma(x)
    np.testing.assert_allclose(got, sigma, rtol=0.03)
    assert np.all(x.std(axis=(1, 2)).mean(0) > 5 * sigma)        # the signal dominates the raw std
    assert np.all(pp.laplacian_noise_sigma(signal) < 0.05 * sigma)   # and the residual removes it
    # float16 storage at values ~1 adds only its quantisation (~0.0003)
    np.testing.assert_allclose(pp.laplacian_noise_sigma(x.astype(np.float16)), sigma, rtol=0.05, atol=3e-4)
    # robust: 0.5% gross outlier pixels (each taints 5 residuals) barely move it
    y = x.copy()
    hit = rng.random(y.shape) < 0.005
    y[hit] += rng.choice([-1.0, 1.0], size=hit.sum())
    np.testing.assert_allclose(pp.laplacian_noise_sigma(y), sigma, rtol=0.06)
    with pytest.raises(ValueError):
        pp.laplacian_residual(np.zeros((2, 2, 5, 1)))


def test_stage_noise_sigma_on_a_build_with_known_noise(tmp_path):
    sigma = np.array([0.005, 0.01, 0.02, 0.03, 0.005, 0.01, 0.02, 0.03])
    rng = np.random.default_rng(5)
    t, X = _variant(tmp_path / "build", lambda X: _smooth_plus_noise(rng, X.shape[0], X.shape[1], sigma)[0]
                    .astype(np.float16))
    _run(t.data_dir, tmp_path / "out", noise_sample_rows=10**6, compare_sample_cap=60)
    _, band, _, noise = _outputs(tmp_path / "out")
    assert noise.dtype == np.float32 and noise.shape == (t.n_bands,)
    np.testing.assert_allclose(noise, sigma, rtol=0.10)
    assert band["noise"]["sample_rows"] == t.n_rows["train"]
    assert band["noise"]["residuals_per_band"] == [t.n_rows["train"] * (t.patch - 2) ** 2] * t.n_bands
    np.testing.assert_allclose(band["per_band"]["residual_robust_std"], noise * np.sqrt(1.25), rtol=1e-6)


# ============================================================================
# calibration audit
# ============================================================================

def test_suspect_capture_is_flagged(tiny, tmp_path):
    _run(tiny.data_dir, tmp_path, compare_sample_cap=60)
    _, _, cal, _ = _outputs(tmp_path)
    assert cal["reads_arrays"] is False
    sus = {s["key"]: s for s in cal["suspect_captures"]}
    assert set(tiny.suspect_captures) == {SUSPECT_CAPTURE_KEY} <= set(sus)
    s = sus[SUSPECT_CAPTURE_KEY]
    assert (s["split"], s["patient"], s["class"], s["gain"]) == ("test", 301, "healthy", 2.0)
    assert {"listed_suspect", "gain_clipped", "at_clip_bound", "gain_outlier"} <= set(s["reasons"])
    assert s["raw_gain_before_clip"] == pytest.approx(8640.0 / 3300.0)
    assert cal["suspect_summary"]["by_split"]["test"] >= 1 and 301 in cal["suspect_summary"]["patients"]
    assert cal["gain_outlier_reference"]["fitted_on"] == "train"
    assert _kinds(cal["problems"])["suspect_captures"]["split"] == "test"

    # distributions per split, class and patient; median intensity per patient
    assert cal["n_captures"] == {"train": 12, "val": 2, "test": 2}
    assert cal["gain_clip"] == [0.5, 2.0] and cal["corpus_median"] == 8640.0
    assert set(cal["gain"]["by_split_class"]["train"]) == {"healthy", "DCIS", "IDC"}
    assert cal["gain"]["by_split_class"]["train"]["DCIS"]["n"] == 2
    assert set(cal["by_patient"]) == {str(p) for ps in tiny.patients.values() for p in ps}
    p301 = cal["by_patient"]["301"]
    assert p301["split"] == ["test"] and p301["n_suspect"] == 1 and p301["median_intensity"]["min"] == 3300.0
    assert cal["gain"]["by_split"]["test"]["max"] == 2.0


def test_calibration_audit_flags_mismatches_and_missing_suspects(tiny, monkeypatch):
    from hsi_augment.adapters import get_adapter
    a = get_adapter("histologyhsi_bc", tiny.data_dir)
    rep = json.loads(json.dumps(a.capture_gain_report))
    first = next(iter(rep["per_capture"]))
    rep["per_capture"][first]["gain"] += 0.1
    rep["suspect_captures"].append("Ghost_HS_VNIR_999_IDC_x10_C01")
    monkeypatch.setattr(type(a), "capture_gain_report", property(lambda self: rep))
    cal = pp.calibration_audit(a)
    k = _kinds(cal["problems"])
    assert k["gain_mismatch"]["severity"] == "error" and k["gain_mismatch"]["keys"] == [first]
    assert k["listed_suspect_not_in_index"]["keys"] == ["Ghost_HS_VNIR_999_IDC_x10_C01"]


# ============================================================================
# band validity
# ============================================================================

def test_saturation_zero_variance_and_bad_values_are_reported(tmp_path):
    def edit(X):
        X[3, 0, 0, 2] = X[50, 1, 2, 2] = X[90, 4, 4, 2] = 65504.0       # float16 maximum
        X[..., 5] = 0.7                                                   # zero variance
        X[10, 0, 0, 0] = X[11, 0, 0, 0] = X[12, 2, 2, 0] = X[200, 1, 1, 0] = 0.0
        X[20, 0, 1, 1] = X[21, 3, 3, 1] = -0.5
        X[30:35, 0, 0, 3] = 1e-4
        X[40, 2, 2, 4] = np.nan
        return X

    t, X = _variant(tmp_path / "build", edit)
    _run(t.data_dir, tmp_path / "out", compare_sample_cap=60)
    norm, band, _, noise = _outputs(tmp_path / "out")
    pb = band["per_band"]
    assert pb["saturated_count"] == [0, 0, 3, 0, 0, 0, 0, 0]
    assert pb["raw_near_ceiling_count"] == [0, 0, 3, 0, 0, 0, 0, 0]
    assert pb["raw_max_counts"][2] > 65504 * 8640 / 2.0 and max(pb["raw_max_counts"][:2]) < 30_000
    assert pb["zero_count"] == [4, 0, 0, 0, 0, 0, 0, 0]
    assert pb["near_zero_count"] == [4, 0, 0, 5, 0, 0, 0, 0]
    assert pb["negative_count"] == [0, 2, 0, 0, 0, 0, 0, 0]
    assert pb["n_nonfinite"] == [0, 0, 0, 0, 1, 0, 0, 0]
    assert pb["variance_positive"] == [True] * 5 + [False] + [True] * 2
    n_values = t.n_rows["train"] * t.patch ** 2
    assert pb["saturated_fraction"][2] == pytest.approx(3 / n_values)

    k = _kinds(band["problems"])
    assert k["zero_variance"]["bands"] == [5] and k["zero_variance"]["severity"] == "error"
    assert k["float16_saturated"]["bands"] == [2] and k["float16_saturated"]["counts"] == [3]
    assert k["raw_near_ceiling"]["bands"] == [2]
    assert k["exact_zero"]["bands"] == [0] and k["near_zero"]["bands"] == [0, 3]
    assert k["negative"]["bands"] == [1] and k["negative"]["counts"] == [2]
    assert k["nonfinite"]["bands"] == [4] and k["nonfinite"]["severity"] == "error"
    assert band["problem_counts"]["error"] >= 2

    # nothing is dropped: the exact stats keep every finite value, NaN excluded
    ref = X.astype(np.float64).reshape(-1, t.n_bands)
    np.testing.assert_allclose(norm["mean"][:4], ref[:, :4].mean(0), rtol=1e-12)
    np.testing.assert_allclose(norm["mean"][4], np.nanmean(ref[:, 4]), rtol=1e-12)
    assert norm["std"][5] == 0.0 and noise[5] == 0.0
    # a constant band has no correlation, and train.py's floored std is not compared
    r = np.array(band["correlation"]["matrix"], dtype=float)
    assert np.isnan(r[5]).all() and np.isnan(r[:, 5]).all() and r[0, 0] == 1.0
    assert band["correlation"]["pixels_dropped_nonfinite"] == 1
    assert norm["sample_estimate"]["std_bands_skipped"] == [5]
    # the reports are strict JSON (NaN written as null)
    for name in (pp.NORM_STATS_NAME, pp.BAND_REPORT_NAME):
        text = (tmp_path / "out" / STAGE_DIR / name).read_text()
        assert "NaN" not in text and "Infinity" not in text


def test_correlation_matrix_is_the_seeded_samples(tiny, tmp_path):
    _run(tiny.data_dir, tmp_path, corr_sample_rows=100, compare_sample_cap=60)
    corr = _outputs(tmp_path)[1]["correlation"]
    rows = pp.sample_rows(np.random.default_rng(stage_seed_sequence(42, STAGE, "correlation")),
                          tiny.n_rows["train"], 100)
    X = np.load(tiny.data_dir / "X_train.npy")
    pix = X[rows].astype(np.float64).reshape(-1, tiny.n_bands)
    np.testing.assert_allclose(corr["matrix"], np.corrcoef(pix.T), rtol=1e-10, atol=1e-12)
    assert (corr["sample_rows"], corr["pixels"]) == (100, 100 * tiny.patch ** 2)
    assert corr["across_gaps"] == [{"after_band": tiny.gap_after, "r": corr["adjacent"][tiny.gap_after]}]
    assert all(abs(p["r"]) > corr["flag_threshold"] for p in corr["pairs_above"])


def test_partial_scan_is_reported(tiny, tmp_path):
    _run(tiny.data_dir, tmp_path, max_rows=50, compare_sample_cap=20)
    norm, band, _, _ = _outputs(tmp_path)
    assert not norm["complete"] and norm["rows_scanned"] == 50
    assert not norm["sample_estimate"]["replicates_train_py"]
    X = np.load(tiny.data_dir / "X_train.npy")[:50].astype(np.float64).reshape(-1, tiny.n_bands)
    np.testing.assert_allclose(norm["mean"], X.mean(0), rtol=1e-12)
    assert _kinds(band["problems"])["partial_scan"]["severity"] == "warning"


# ============================================================================
# leakage policy, CLI
# ============================================================================

def test_only_the_train_split_is_read(tiny, tmp_path, monkeypatch):
    spec = load_builtin_stages().get(STAGE)
    assert spec.purposes == ("fit",) and not spec.reads_held_out()
    assert spec.group == "preprocess" and spec.dirname == "prep/spectral"

    real_load = np.load

    def guarded_load(path, *a, **kw):
        name = os.path.basename(str(path))
        if name.endswith(("_val.npy", "_test.npy")):
            raise AssertionError(f"spectral-audit opened held-out array {name}")
        return real_load(path, *a, **kw)

    monkeypatch.setattr(np, "load", guarded_load)
    _run(tiny.data_dir, tmp_path, compare_sample_cap=60)
    monkeypatch.undo()
    rec = read_access_log(tmp_path / hio.ACCESS_LOG_NAME)
    assert rec and {(r["stage"], r["split"], r["purpose"], r["allowed"]) for r in rec} == {(STAGE, "train", "fit", True)}
    assert set(rec[0]["files"]) == {"X_train.npy", "captures_train.npy"}


def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)


def test_cli_runs_skips_and_is_deterministic(tiny, tmp_path):
    r = _cli(STAGE, "--help")
    assert r.returncode == 0 and "--compare_seeds" in r.stdout and "--noise_sample_rows" in r.stdout
    r = _cli("preprocess", "--help")
    assert r.returncode == 0 and STAGE in r.stdout and "prep/spectral" in r.stdout

    out = tmp_path / "aug"
    base = [STAGE, "--data_dir", tiny.data_dir, "--out", out, "--chunk_rows", "7",
            "--compare_sample_cap", "60", "--compare_seeds", "3", "42"]
    r = _cli(*base)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"{STAGE}: ran" in r.stdout and "suspect" in r.stdout
    done = json.loads((out / STAGE_DIR / DONE_NAME).read_text())
    assert set(done["outputs"]) == {pp.NORM_STATS_NAME, pp.BAND_REPORT_NAME, pp.CALIBRATION_REPORT_NAME,
                                    pp.NOISE_SIGMA_NAME}
    assert done["config"]["compare_seeds"] == [3, 42] and done["summary"]["sample_estimate_agrees"]
    assert done["summary"]["suspect_captures"] >= 1 and done["summary"]["complete"]

    r = _cli(*base)
    assert r.returncode == 0 and "already done" in r.stdout
    r = _cli(*base, "--force")
    assert r.returncode == 0 and f"{STAGE}: ran" in r.stdout
    again = json.loads((out / STAGE_DIR / DONE_NAME).read_text())
    for name in (pp.NOISE_SIGMA_NAME, pp.NORM_STATS_NAME, pp.BAND_REPORT_NAME, pp.CALIBRATION_REPORT_NAME):
        assert again["outputs"][name]["sha256"] == done["outputs"][name]["sha256"]

    r = _cli(STAGE, "--data_dir", tiny.data_dir, "--out", out, "--noise_flat_fraction", "0")
    assert r.returncode == 2


# ============================================================================
# real data (opt-in, tiny)
# ============================================================================

REAL_DATA = os.environ.get("HSI_AUGMENT_REAL_DATA")


@pytest.mark.skipif(not REAL_DATA, reason="set HSI_AUGMENT_REAL_DATA to a build's hsi dir to run")
def test_real_data_smoke(tmp_path):
    """2,000 rows (about 15 MB) of the real X_train; never the whole 19 GB."""
    _run(REAL_DATA, tmp_path, max_rows=2000, chunk_rows=1000, corr_sample_rows=1000,
         noise_sample_rows=500, compare_sample_cap=500, compare_seeds=(42,))
    norm, band, cal, noise = _outputs(tmp_path)
    C = norm["n_bands"]
    assert not norm["complete"] and norm["rows_scanned"] == 2000
    assert noise.dtype == np.float32 and noise.shape == (C,) and np.all(np.isfinite(noise)) and np.all(noise >= 0)
    assert np.array(band["correlation"]["matrix"]).shape == (C, C)
    assert _kinds(band["problems"])["partial_scan"]
    assert norm["units"]["corpus_median"] == 8640.0
    assert cal["n_captures"]["train"] == 499
    assert all(s["patient"] == 68 and s["split"] != "train" for s in cal["suspect_captures"]
               if "listed_suspect" in s["reasons"])
