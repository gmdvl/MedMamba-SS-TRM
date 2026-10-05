# -*- coding: utf-8 -*-
"""
test_recon_artifacts.py
=========================
`training/recon_artifacts.py` - the module that finally writes reconstructed
cubes to disk. Synthetic tensors only: no dataset, no checkpoint, no GPU.

What is actually being pinned here:

  * `ssim2d` is the SPATIAL metric the pipeline never had. The frozen
    `spectral_recon_metrics.ssim` is per-pixel across the band axis (its own
    docstring says so), so a run can score 0.85 "SSIM" while being spatially
    wrong. The two must not agree by accident, and the test asserts they are
    computed on different axes.
  * The sample selection must include each class's WORST reconstruction.
    A figure that silently shows only good cases is worse than no figure.
  * Cubes must round-trip through `samples.npz` bit-exact at float32 - the
    npz exists so a figure can be redrawn without re-running the model.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from training import recon_artifacts as ra


# ----------------------------------------------------------------------
# ssim2d - the spatial metric
# ----------------------------------------------------------------------

def test_ssim2d_is_one_for_identical_cubes():
    x = torch.rand(2, 5, 16, 16)
    assert ra.ssim2d(x, x.clone()) == pytest.approx(1.0, abs=1e-4)


def test_ssim2d_drops_when_spatial_structure_is_destroyed():
    torch.manual_seed(0)
    x = torch.rand(1, 4, 16, 16)
    blurred = torch.full_like(x, float(x.mean()))          # same mean, no structure
    assert ra.ssim2d(x, blurred) < 0.5


def test_ssim2d_is_not_the_frozen_spectral_ssim():
    """The failure mode this metric exists to catch: a reconstruction that
    gets every SPECTRUM right and puts them in the wrong PLACE.

    The cube is one spectral shape scaled by a spatially varying amplitude.
    Shuffling the pixels leaves each pixel's spectral shape intact - the
    frozen spectral SSIM stays high (~0.73) - while the spatial structure is
    destroyed, and `ssim2d` collapses (~0.05). A run reporting only "SSIM"
    would call that reconstruction good.
    """
    from training.spectral_recon_metrics import ssim as spectral_ssim
    torch.manual_seed(0)
    base = torch.linspace(0.2, 1.0, 6).view(1, 6, 1, 1)     # one spectral shape
    amp = torch.rand(1, 1, 12, 12) * 0.8 + 0.2              # spatial structure
    x = base * amp
    flat = x.flatten(2)
    shuffled = flat[:, :, torch.randperm(flat.shape[2])].reshape_as(x)

    spectral = spectral_ssim(x, shuffled)
    spatial = ra.ssim2d(x, shuffled)
    assert spectral > 0.5, f"spectral SSIM should stay high, got {spectral}"
    assert spatial < 0.2, f"spatial SSIM should collapse, got {spatial}"
    assert spectral > spatial * 3


def test_ssim2d_shrinks_window_on_small_patches():
    x = torch.rand(1, 3, 5, 5)                              # smaller than the default 7x7
    assert ra.ssim2d(x, x.clone(), window_size=7) == pytest.approx(1.0, abs=1e-4)


# ----------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------

def test_selection_always_contains_best_and_worst_of_each_class():
    sam = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    idx, tags = ra.select_quantile_indices(sam, labels, per_class=3, num_classes=2)
    chosen = set(int(i) for i in idx)
    assert 1 in chosen and 2 in chosen                      # class 0 best (1.0) and worst (9.0)
    assert 5 in chosen and 4 in chosen                      # class 1 best (2.0) and worst (7.0)
    assert len(tags) == len(idx)
    assert all(t.startswith("class") for t in tags)


def test_selection_handles_a_class_with_fewer_samples_than_requested():
    sam = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 0, 1])
    idx, _ = ra.select_quantile_indices(sam, labels, per_class=5, num_classes=2)
    assert sorted(int(i) for i in idx) == [0, 1, 2]         # no duplicates, no crash


def test_selection_ignores_nan_sam_when_finite_values_exist():
    sam = np.array([np.nan, 4.0, 1.0])
    labels = np.zeros(3, dtype=int)
    idx, _ = ra.select_quantile_indices(sam, labels, per_class=2, num_classes=1)
    assert 0 not in set(int(i) for i in idx)


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------

def test_rgb_bands_native_for_three_channel_data():
    bands, note = ra.resolve_rgb_bands(3, None, None)
    assert bands == (0, 1, 2) and "native RGB" in note


def test_rgb_bands_match_nearest_wavelengths():
    wl = np.linspace(400.0, 700.0, 31)                      # 10 nm steps
    bands, note = ra.resolve_rgb_bands(31, wl, None)
    assert [round(float(wl[b])) for b in bands] == [640, 550, 460]
    assert "640" in note


def test_rgb_bands_accept_explicit_indices_and_reject_out_of_range():
    bands, note = ra.resolve_rgb_bands(10, None, "7,4,1")
    assert bands == (7, 4, 1) and "band indices" in note
    with pytest.raises(ValueError):
        ra.resolve_rgb_bands(10, None, "7,4,99")
    with pytest.raises(ValueError):
        ra.resolve_rgb_bands(10, None, "7,4")


def test_composite_shares_one_range_between_original_and_reconstruction():
    """A washed-out reconstruction must LOOK washed out. Stretching each
    panel to its own percentiles would hide exactly the error the figure
    exists to show."""
    true = np.random.default_rng(0).uniform(0.0, 1.0, size=(3, 8, 8))
    recon = true * 0.25
    vmin, vmax = ra.composite_range(true, (0, 1, 2))
    rgb_t = ra.rgb_composite(true, (0, 1, 2), vmin, vmax)
    rgb_p = ra.rgb_composite(recon, (0, 1, 2), vmin, vmax)
    assert rgb_p.mean() < rgb_t.mean() * 0.6
    assert rgb_t.min() >= 0.0 and rgb_t.max() <= 1.0


def test_metric_curves_write_nothing_when_the_run_had_no_reconstruction():
    rows = [{"epoch": e, "spectral_sam_deg": 0.0, "spectral_rmse": 0.0,
             "spectral_psnr": 0.0, "spectral_ssim": 0.0} for e in (1, 2, 3)]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "curves.png")
        assert ra.plot_recon_metric_curves(rows, out) is False
        assert not os.path.exists(out)


def test_metric_curves_are_written_when_a_decoder_actually_trained():
    rows = [{"epoch": e, "spectral_sam_deg": 20.0 - e, "spectral_rmse": 0.4 / e,
             "spectral_psnr": 9.0 + e, "spectral_ssim": 0.1 * e} for e in (1, 2, 3)]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "curves.png")
        assert ra.plot_recon_metric_curves(rows, out) is True
        assert os.path.getsize(out) > 0


# ----------------------------------------------------------------------
# persistence - the actual point of the module
# ----------------------------------------------------------------------

def _fake_export(tmp, n=4, c=8, hw=12, render_png=True):
    rng = np.random.default_rng(7)
    x_true = rng.uniform(0.05, 0.9, size=(n, c, hw, hw)).astype(np.float32)
    x_recon = (x_true + rng.normal(0, 0.02, x_true.shape)).astype(np.float32)
    labels = np.array([0, 1, 0, 1])[:n]
    idx = np.arange(n) * 3
    tags = [f"class{int(l)}_q{i/max(n-1,1):.2f}" for i, l in enumerate(labels)]
    wl = np.linspace(450.0, 700.0, c).astype(np.float32)
    metrics = ra.per_sample_full_cube_metrics(
        torch.from_numpy(x_true), torch.from_numpy(x_recon), wavelengths=torch.from_numpy(wl))
    result = ra.save_reconstruction_samples(
        tmp, x_true, x_recon, labels, idx, tags, metrics, wl,
        ["healthy", "DCIS"], {"run_dir": "unit-test", "checkpoint": "none"},
        render_png=render_png)
    return x_true, x_recon, labels, idx, tags, metrics, result


def test_cubes_round_trip_through_the_npz():
    with tempfile.TemporaryDirectory() as d:
        x_true, x_recon, labels, idx, tags, _, result = _fake_export(d)
        z = np.load(result["npz"], allow_pickle=True)
        assert np.array_equal(z["x_true"], x_true)
        assert np.array_equal(z["x_recon"], x_recon)
        assert np.array_equal(z["label"], labels)
        assert np.array_equal(z["dataset_index"], idx)
        assert list(z["tag"]) == tags
        assert list(z["class_names"]) == ["healthy", "DCIS"]
        assert z["wavelengths"].shape == (8,)


def test_export_writes_metrics_provenance_status_and_figures():
    with tempfile.TemporaryDirectory() as d:
        _fake_export(d)
        ra.write_status(d, written=True, n_samples=4)
        for name in ("samples.npz", "samples_metrics.json", "provenance.json",
                     "status.json", "reconstruction_samples.png"):
            assert os.path.getsize(os.path.join(d, name)) > 0, name
        assert len(os.listdir(os.path.join(d, "samples"))) == 4

        rows = json.load(open(os.path.join(d, "samples_metrics.json")))
        assert len(rows) == 4
        for r in rows:
            assert r["class_name"] in ("healthy", "DCIS")
            assert r["sam_deg"] is not None and r["sam_deg"] < 20.0
            assert r["ssim2d"] is not None and r["ssim2d"] > 0.0
            assert "ssim_spectral" in r and "psnr" in r and "rmse" in r


def test_provenance_records_the_band_choice_and_warns_about_the_two_ssims():
    with tempfile.TemporaryDirectory() as d:
        _fake_export(d, render_png=False)
        prov = json.load(open(os.path.join(d, "provenance.json")))
        assert prov["units"] == "reflectance (denormalized)"
        assert len(prov["rgb_bands"]) == 3
        assert "nm" in prov["rgb_band_note"]
        assert "ssim2d" in prov["ssim2d_note"] and "spectral" in prov["ssim2d_note"]
        assert prov["cube_shape"] == [8, 12, 12]
        assert not os.path.isdir(os.path.join(d, "samples")) or \
            len(os.listdir(os.path.join(d, "samples"))) == 0


def test_status_is_written_even_on_the_skip_path():
    """The defect this replaces: `trainerg_v12.py:967-974` swallows every
    artifact failure into one `logger.warning`, which is why four of the
    eight runs that trained a decoder have an empty `reconstruction/`
    directory and no record of why."""
    with tempfile.TemporaryDirectory() as d:
        ra.write_status(d, written=False, reason="no reconstruction decoder attached")
        s = json.load(open(os.path.join(d, "status.json")))
        assert s["written"] is False and "decoder" in s["reason"]


def test_batch_sam_matches_the_v2_metric_it_ranks_by():
    """The ranking pass must measure the same quantity the trainer reports,
    or the 'worst' sample in a figure is not the worst sample."""
    from training.spectral_recon_metrics_v2 import sam_deg as sam_deg_v2
    torch.manual_seed(0)
    x = torch.rand(3, 6, 4, 4) + 0.1
    y = x + torch.randn_like(x) * 0.05
    per_sample = ra.batch_sam_deg(x, y)
    for i in range(3):
        pooled_t = x[i:i + 1].mean(dim=(2, 3), keepdim=True)
        pooled_p = y[i:i + 1].mean(dim=(2, 3), keepdim=True)
        assert float(per_sample[i]) == pytest.approx(sam_deg_v2(pooled_t, pooled_p), abs=1e-3)
