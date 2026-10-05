# -*- coding: utf-8 -*-
"""tests/test_prepare_hmi_lusc_v20.py - HMI-LUSC preparation on a tiny synthetic
tree: calibration, mask purity, glass filter, patient-disjoint folds."""
import json

import numpy as np
from PIL import Image

import prepare_hmi_lusc_v20 as lusc

B, H, W = 5, 22, 33


def _envi(path, arr):
    arr.astype(np.uint8).tofile(path)
    path.with_name(path.name + ".hdr").write_text(
        f"ENVI\nsamples = {W}\nlines   = {H}\nbands   = {B}\nheader offset = 0\n"
        f"data type = 1\ninterleave = bsq\nbyte order = 0\n"
        f"wavelength = {{{', '.join(str(450 + 5 * i) for i in range(B))}}}\n")


def _tree(root, n_patients=5):
    for k in range(1, n_patients + 1):
        pd = root / f"P{k}"
        (pd / "LUSC_ROI_1").mkdir(parents=True)
        _envi(pd / "whiteReference", np.full((B, H, W), 200))
        _envi(pd / "darkReference", np.full((B, H, W), 10))
        raw = np.full((B, H, W), 105)            # reflectance 0.5: tissue
        raw[:, :, 22:] = 199                     # right third: glass (~0.99)
        _envi(pd / "LUSC_ROI_1" / "Raw", raw)
        m = np.zeros((H, W), np.uint8)
        m[:, :11] = 255                          # left third: tumour
        Image.fromarray(m).save(pd / "LUSC_ROI_1" / "Label.png")


def test_extract_and_split(tmp_path):
    _tree(tmp_path / "raw")
    pool = tmp_path / "pool"
    assert lusc.main(["extract", "--root", str(tmp_path / "raw"), "--pool", str(pool),
                      "--workers", "1"]) == 0
    X, y = np.load(pool / "X_pool.npy"), np.load(pool / "y_pool.npy")
    # per patient: 2 rows x 3 cols of 11x11 patches; col 0 tumour, col 1 non-tumour, col 2 glass (dropped)
    assert X.shape == (5 * 4, 11, 11, B) and y.sum() == 5 * 2
    assert np.allclose(X.astype(np.float32), 0.5, atol=1e-3)          # (105-10)/(200-10)
    assert np.load(pool / "wavelengths.npy").tolist() == [450, 455, 460, 465, 470]

    seen_test = set()
    for f in range(5):
        out = tmp_path / f"f{f}"
        assert lusc.main(["split", "--pool", str(pool), "--out", str(out), "--fold", str(f)]) == 0
        g = {s: set(np.load(out / "hsi" / f"groups_{s}.npy").tolist()) for s in ("train", "val", "test")}
        assert not (g["train"] & g["test"]) and not (g["train"] & g["val"]) and not (g["val"] & g["test"])
        seen_test |= g["test"]
        assert json.loads((out / "hsi" / "class_names.json").read_text()) == lusc.CLASS_NAMES
    assert seen_test == {1, 2, 3, 4, 5}                                  # every patient tested once
