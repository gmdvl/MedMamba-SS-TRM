# -*- coding: utf-8 -*-
"""tests/test_prep_hsi_fixture.py - prepare_histologyhsi_bc (v7 adapter layer) on synthetic
ENVI cubes (Stage 4). Skipped where the `spectral` package is absent.

Run with: pytest test_prep_hsi_fixture.py -v  (or the repo's pytest-free runner)
"""
import json
import os

import numpy as np
import pytest

spectral = pytest.importorskip("spectral")

from PIL import Image                                    # noqa: E402
from prepare_histologyhsi_bc import HistologyHsiAdapter  # noqa: E402
from training.prep import run_prep                        # noqa: E402

_TISSUES = ["Healthy", "DCIS", "IDC"]
_BANDS = 6
_HW = 8


def _make_fixture(root):
    hsi_root = os.path.join(root, "02_01_HSI_Images")
    os.makedirs(hsi_root)
    rng = np.random.default_rng(0)
    for pid in range(6):
        tissue = _TISSUES[pid % 3]
        cap = os.path.join(hsi_root, f"HSI_VNIR_p{pid:02d}_{tissue}_x10_C1")
        os.makedirs(cap)
        cube = rng.random((_HW, _HW, _BANDS), dtype=np.float64).astype(np.float32)
        md = {"wavelength": [str(400 + 40 * b) for b in range(_BANDS)],
              "wavelength units": "nm"}
        spectral.envi.save_image(os.path.join(cap, "cube.hdr"), cube, metadata=md,
                                  dtype=np.float32, force=True, interleave="bil")
        Image.fromarray(rng.integers(0, 256, (_HW, _HW, 3), dtype=np.uint8)).save(
            os.path.join(cap, "rgb.png"))
    return root


def _run(root, out, *extra):
    return run_prep(HistologyHsiAdapter(),
                    ["--root", str(root), "--out_dir", str(out), "--seed", "0",
                     "--patch_size", "4", "--stride", "4", *extra])


def test_hsi_and_rgb_contract(tmp_path):
    root = _make_fixture(tmp_path / "raw")
    out = tmp_path / "out"
    assert _run(root, out, "--modality", "both", "--split", "80_20", "--num_workers", "2") == 0

    Xh = np.load(out / "hsi" / "X_train.npy")
    Xr = np.load(out / "rgb" / "X_train.npy")
    assert Xh.shape[1:] == (4, 4, _BANDS)              # no band selection -> all bands, NHWC
    assert Xr.shape[1:] == (4, 4, 3)
    assert np.load(out / "hsi" / "wavelengths.npy").shape == (_BANDS,)
    assert json.load(open(out / "hsi" / "class_names.json")) == ["healthy", "DCIS", "IDC"]
    from training.npy_atomic import verify_against_manifest
    assert verify_against_manifest(str(out / "hsi")) == []
    assert verify_against_manifest(str(out / "rgb")) == []


def test_band_selection_reduces_channels(tmp_path):
    root = _make_fixture(tmp_path / "raw")
    out = tmp_path / "out"
    assert _run(root, out, "--modality", "hsi", "--split", "80_20",
                "--band_selection", "variance", "--num_bands", "3", "--num_workers", "0") == 0
    Xh = np.load(out / "hsi" / "X_train.npy")
    assert Xh.shape[1:] == (4, 4, 3)
    assert np.load(out / "hsi" / "wavelengths.npy").shape == (3,)
    assert np.load(out / "hsi" / "wavelengths_full.npy").shape == (_BANDS,)
    assert os.path.exists(out / "selected_band_indices.npy")
