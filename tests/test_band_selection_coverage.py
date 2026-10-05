# -*- coding: utf-8 -*-
"""
test_band_selection_coverage.py
=================================
MedMamba-SS-TRM v15 plan, R2.2 "Verify" - band selection must span the sensor.

`data/hsi_v7/selected_band_indices.npy` is exactly `[0..31]`: the FIRST 32 of
740 bands, 400.5-423.0 nm out of 400.5-938.2 nm. That is 4.2% of the sensor,
missing the entire haemoglobin absorption region (~540-580 nm) and all NIR
scatter, and it happened because `compute_band_importance`'s blended
variance/MI score falls almost monotonically with band index while
`select_bands_by_importance` took a plain top-k - so the selection walked off
the front of the array.

These tests pin the three fixes: a decorrelated greedy walk (R2.2a), a
`uniform` method (R2.2b), and a coverage guardrail that FAILS the prep
(R2.2c/d).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import prepare_histologyhsi_bc as v6

SENSOR_BANDS = 740
SENSOR_WAVELENGTHS = np.linspace(400.5, 938.2, SENSOR_BANDS).astype(np.float32)


def smooth_spectra(n=800, C=SENSOR_BANDS, seed=0):
    """Spectra with the property that matters here: neighbouring bands are
    near-perfectly correlated, which is true of every real spectrometer and is
    what makes a contiguous top-k so nearly worthless."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, C)) * 0.05, axis=1).astype(np.float32)


def monotone_importance(C=SENSOR_BANDS):
    """The measured shape of `compute_band_importance` on this dataset:
    falling almost monotonically with band index."""
    return np.linspace(1.0, 0.0, C).astype(np.float32)


# ----------------------------------------------------------------------
# The defect, restated as a test
# ----------------------------------------------------------------------

def test_plain_topk_reproduces_the_hsi_v7_selection():
    sel, _, _ = v6.select_bands_by_importance(monotone_importance(), 32, None)
    assert sel.tolist() == list(range(32))
    cov = v6.band_coverage_report(sel, SENSOR_WAVELENGTHS, SENSOR_BANDS)
    assert cov["coverage_frac"] < 0.05
    assert cov["wavelength_max_nm"] < 425.0


# ----------------------------------------------------------------------
# R2.2a - decorrelated greedy selection
# ----------------------------------------------------------------------

def test_decorrelation_widens_the_selection():
    pixels = smooth_spectra()
    plain, _, _ = v6.select_bands_by_importance(monotone_importance(), 32, None)
    decor, _, _ = v6.select_bands_by_importance(monotone_importance(), 32, None,
                                                 pixels=pixels, max_corr=0.95)
    plain_span = plain.max() - plain.min()
    decor_span = decor.max() - decor.min()
    assert decor_span > plain_span * 3


def test_decorrelated_bands_are_not_near_duplicates():
    pixels = smooth_spectra()
    sel, _, _ = v6.select_bands_by_importance(monotone_importance(), 24, None,
                                               pixels=pixels, max_corr=0.95)
    corr = np.nan_to_num(np.corrcoef(pixels.T.astype(np.float64)), nan=0.0)
    for i, a in enumerate(sel):
        for b in sel[i + 1:]:
            assert abs(corr[a, b]) <= 0.95 + 1e-9, f"bands {a} and {b} correlate at {corr[a, b]:.3f}"


def test_min_gap_enforces_index_spacing():
    sel, _, _ = v6.select_bands_by_importance(monotone_importance(), 16, None, min_gap=20)
    assert np.all(np.diff(np.sort(sel)) >= 20)


def test_decorrelation_falls_back_rather_than_returning_too_few(capsys):
    """A constraint that cannot be satisfied must warn and top up from the
    plain ranking, never silently return a shorter selection."""
    C = 40
    pixels = np.repeat(smooth_spectra(n=200, C=4)[:, :1], C, axis=1)   # all bands identical
    sel, _, _ = v6.select_bands_by_importance(monotone_importance(C), 12, None,
                                               pixels=pixels, max_corr=0.5)
    assert len(sel) == 12
    assert "falling back to the plain top-k" in capsys.readouterr().out


def test_defaults_reproduce_the_old_selection_exactly():
    """No `pixels`/`max_corr`/`min_gap` => byte-for-byte the pre-v15 result,
    so re-running an old prep command reproduces the old dataset."""
    imp = monotone_importance()
    a, ra, na = v6.select_bands_by_importance(imp, 32, None, max_bands=64)
    assert a.tolist() == list(range(32))
    b, rb, nb = v6.select_bands_by_importance(imp, None, 0.5)
    assert b.tolist() == sorted(int(i) for i in np.where(imp >= 0.5)[0])
    assert na == len(imp) and nb == len(b)


# ----------------------------------------------------------------------
# R2.2b - the uniform method
# ----------------------------------------------------------------------

def test_uniform_spans_the_whole_sensor():
    sel = v6.select_bands_uniform(SENSOR_BANDS, 32)
    assert sel[0] == 0 and sel[-1] == SENSOR_BANDS - 1
    cov = v6.band_coverage_report(sel, SENSOR_WAVELENGTHS, SENSOR_BANDS)
    assert cov["coverage_frac"] == pytest.approx(1.0)


def test_uniform_is_deterministic_and_sorted_and_unique():
    a = v6.select_bands_uniform(SENSOR_BANDS, 32)
    b = v6.select_bands_uniform(SENSOR_BANDS, 32)
    assert np.array_equal(a, b)
    assert np.array_equal(a, np.unique(a))


@pytest.mark.parametrize("n_total,k,expected_len", [(740, 32, 32), (10, 32, 10), (740, 1, 1)])
def test_uniform_never_invents_bands(n_total, k, expected_len):
    sel = v6.select_bands_uniform(n_total, k)
    assert len(sel) == expected_len
    assert sel.min() >= 0 and sel.max() < n_total


def test_uniform_is_reachable_through_select_bands():
    pixels = smooth_spectra(n=50, C=SENSOR_BANDS)
    sel = v6.select_bands(
        "uniform", pixels, np.zeros(len(pixels), dtype=np.int64), 32, 0.98, 0)
    assert np.array_equal(sel, v6.select_bands_uniform(SENSOR_BANDS, 32))


# ----------------------------------------------------------------------
# R2.2c/d - the coverage guardrail
# ----------------------------------------------------------------------

def test_coverage_report_reproduces_the_hsi_v7_numbers():
    cov = v6.band_coverage_report(np.arange(32), SENSOR_WAVELENGTHS, SENSOR_BANDS)
    assert cov["coverage_basis"] == "wavelength_nm"
    assert cov["wavelength_min_nm"] == pytest.approx(400.5, abs=0.1)
    assert cov["wavelength_max_nm"] == pytest.approx(423.0, abs=0.5)
    assert cov["sensor_span_nm"] == pytest.approx(537.7, abs=0.5)
    assert cov["coverage_frac"] == pytest.approx(0.042, abs=0.005)


def test_coverage_falls_back_to_band_indices_without_wavelengths():
    cov = v6.band_coverage_report(np.arange(32), None, SENSOR_BANDS)
    assert cov["coverage_basis"] == "band_index"
    assert cov["coverage_frac"] == pytest.approx(31 / 739, abs=1e-6)


def test_narrow_selection_fails_the_prep(tmp_path):
    cov = v6.band_coverage_report(np.arange(32), SENSOR_WAVELENGTHS, SENSOR_BANDS)
    with pytest.raises(v6.BandCoverageError) as excinfo:
        v6.enforce_band_coverage(cov, 0.30, allow_narrow=False, out_dir=str(tmp_path))
    assert "BAND_COVERAGE_TOO_NARROW" in str(excinfo.value)
    # the report is written BEFORE the raise, so the failure is diagnosable
    written = json.loads((tmp_path / "band_coverage_report.json").read_text())
    assert written["coverage_frac"] == pytest.approx(cov["coverage_frac"])


def test_allow_narrow_bands_downgrades_to_a_warning(tmp_path, capsys):
    cov = v6.band_coverage_report(np.arange(32), SENSOR_WAVELENGTHS, SENSOR_BANDS)
    v6.enforce_band_coverage(cov, 0.30, allow_narrow=True, out_dir=str(tmp_path))
    assert "BAND_COVERAGE_TOO_NARROW" in capsys.readouterr().out


def test_wide_selection_passes(tmp_path):
    cov = v6.band_coverage_report(v6.select_bands_uniform(SENSOR_BANDS, 32),
                                   SENSOR_WAVELENGTHS, SENSOR_BANDS)
    v6.enforce_band_coverage(cov, 0.30, allow_narrow=False, out_dir=str(tmp_path))


def test_guardrail_is_off_when_min_coverage_is_none(tmp_path):
    cov = v6.band_coverage_report(np.arange(32), SENSOR_WAVELENGTHS, SENSOR_BANDS)
    v6.enforce_band_coverage(cov, None, allow_narrow=False, out_dir=str(tmp_path))


# ----------------------------------------------------------------------
# The artifacts a training run reads
# ----------------------------------------------------------------------

def test_save_selected_bands_records_coverage(tmp_path):
    sel = v6.select_bands_uniform(SENSOR_BANDS, 32)
    v6.save_selected_bands(str(tmp_path), "uniform", sel, SENSOR_WAVELENGTHS, SENSOR_BANDS,
                            num_bands_requested=32, sample_fraction=None,
                            min_coverage=0.30, allow_narrow=False)
    cfg = json.loads((tmp_path / "band_selection_config.json").read_text())
    assert cfg["coverage"]["coverage_frac"] == pytest.approx(1.0)
    assert cfg["coverage"]["wavelength_span_nm"] == pytest.approx(537.7, abs=0.5)


def test_save_selected_bands_refuses_a_narrow_selection(tmp_path):
    with pytest.raises(v6.BandCoverageError):
        v6.save_selected_bands(str(tmp_path), "importance", np.arange(32), SENSOR_WAVELENGTHS,
                                SENSOR_BANDS, num_bands_requested=32, sample_fraction=0.1,
                                min_coverage=0.30, allow_narrow=False)


def test_importance_report_records_coverage_and_decorrelation_settings(tmp_path):
    imp = monotone_importance()
    sel = v6.select_bands_uniform(SENSOR_BANDS, 32)
    report = v6.save_importance_selection_report(
        str(tmp_path), imp, np.argsort(-imp), sel, n_informative=SENSOR_BANDS,
        num_bands_requested=32, importance_threshold=None, original_bands=SENSOR_BANDS,
        wavelengths=SENSOR_WAVELENGTHS, min_coverage=0.30, allow_narrow=False,
        max_corr=0.95, min_gap=0)
    assert report["coverage"]["coverage_frac"] == pytest.approx(1.0)
    assert report["band_max_corr"] == 0.95
    on_disk = json.loads((tmp_path / "band_selection_report.json").read_text())
    assert on_disk["coverage"]["sensor_max_nm"] == pytest.approx(938.2, abs=0.1)


def test_the_prep_that_produced_hsi_v7_would_now_fail(tmp_path):
    """End to end: the exact call the v7 prep made, with the v15 defaults."""
    imp = monotone_importance()
    sel, ranking, n_inf = v6.select_bands_by_importance(imp, 32, None)
    with pytest.raises(v6.BandCoverageError):
        v6.save_importance_selection_report(
            str(tmp_path), imp, ranking, sel, n_inf, num_bands_requested=32,
            importance_threshold=None, original_bands=SENSOR_BANDS,
            wavelengths=SENSOR_WAVELENGTHS, min_coverage=0.30, allow_narrow=False)
