# -*- coding: utf-8 -*-
"""tests/test_prep_pad_fixture.py - prepare_pad_ufes_20 on a synthetic
PAD-UFES-20 layout (Stage 3).

Run with: pytest test_prep_pad_fixture.py -v  (or the repo's pytest-free runner)
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from prepare_pad_ufes_20 import PadUfes20Adapter
from training.npy_integrity import validate_dataset
from training.prep import run_prep

_CLASSES = ["ACK", "BCC", "MEL"]          # subset of the 6


def _make_fixture(root):
    imgs = os.path.join(root, "imgs")
    os.makedirs(imgs)
    rows = []
    rng = np.random.default_rng(0)
    # 6 patients x 3 images, class = patient % 3  -> every class has 2 patients
    for pid in range(6):
        for j in range(3):
            stem = f"PAT_{pid}_{j}"
            w, h = 60 + 4 * j, 50 + 3 * j          # varied native sizes
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(
                os.path.join(imgs, stem + ".png"))
            rows.append((stem + ".png", f"pat{pid}", _CLASSES[pid % 3]))
    with open(os.path.join(root, "metadata.csv"), "w") as f:
        f.write("img_id,patient_id,diagnostic\n")
        for a, b, c in rows:
            f.write(f"{a},{b},{c}\n")
    return root


def _run(root, out, *extra):
    # the fixture only exercises 3 of the 6 PAD classes
    return run_prep(PadUfes20Adapter(),
                    ["--root", str(root), "--out_dir", str(out), "--seed", "0",
                     "--allow_missing_classes", *extra])


def test_whole_image_contract(tmp_path):
    root = _make_fixture(tmp_path / "raw")
    out = tmp_path / "out"
    assert _run(root, out, "--split", "80_20", "--img_size", "32", "--num_workers", "2") == 0

    Xtr, ytr = np.load(out / "X_train.npy"), np.load(out / "y_train.npy")
    Xte = np.load(out / "X_test.npy")
    assert Xtr.dtype == np.float32 and Xtr.shape[1:] == (32, 32, 3)      # NHWC
    assert ytr.dtype == np.int64 and 0.0 <= Xtr.min() and Xtr.max() <= 1.0
    assert len(Xtr) + len(Xte) == 18                                    # 1 sample / image
    assert set(np.unique(ytr).tolist()) <= {0, 1, 2}
    assert json.load(open(out / "class_names.json")) == ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    validate_dataset({"X_train": str(out / "X_train.npy"), "y_train": str(out / "y_train.npy"),
                      "X_test": str(out / "X_test.npy"), "y_test": str(out / "y_test.npy")},
                     level="deep")


def test_patch_grid_and_kfold(tmp_path):
    root = _make_fixture(tmp_path / "raw")
    out = tmp_path / "out"
    assert _run(root, out, "--kfold", "3", "--fold", "0", "--img_size", "22",
                "--patch_size", "11", "--stride", "11", "--num_workers", "0") == 0
    Xtr = np.load(out / "X_train.npy")
    assert Xtr.shape[1:] == (11, 11, 3)
    # 22x22 img, 11 patch, stride 11 -> 2x2 = 4 patches per image
    assert len(Xtr) % 4 == 0
    meta = json.load(open(out / "split_assignment.json"))
    assert meta["strategy"] == "kfold" and meta["kfold_algo_version"] == 1


def test_resume_is_idempotent(tmp_path):
    root = _make_fixture(tmp_path / "raw")
    out = tmp_path / "out"
    _run(root, out, "--split", "80_20", "--img_size", "24", "--num_workers", "2")
    a = np.load(out / "X_train.npy").copy()
    assert _run(root, out, "--split", "80_20", "--img_size", "24", "--num_workers", "2") == 0
    assert np.array_equal(a, np.load(out / "X_train.npy"))
