# -*- coding: utf-8 -*-
"""
test_group_sidecars.py
========================
MedMamba-SS-TRM v16 plan, Stage 0.3 verification.

Builds a tiny synthetic `histology_hsi_bc`-shaped `_progress.json` + unified
`y_*.npy`, runs `scripts/derive_patch_groups.py`'s HSI path against it, and
asserts:

  - derived `groups_*.npy` / `captures_*.npy` length matches `y_*.npy`;
  - per-capture (per-group) labels are single-valued;
  - patient ids are recovered correctly from the item key.

Also exercises the real, already-derived sidecars under
`data/hsi_v7-80_10_10_importance/hsi` and `data/pad_v6` when present, since
those are the datasets Stage 0.3 exists to unblock.
"""

import json
import os
import tempfile

import numpy as np

from scripts.derive_patch_groups import _derive_hsi, _derive_pad
from training.prep.manifest import save_manifest, load_manifest


def _build_synthetic_manifest(out_dir, modality="hsi"):
    """3 captures: two from patient 12 (different tissue), one from patient 7."""
    os.makedirs(os.path.join(out_dir, modality), exist_ok=True)
    items = {
        "HSI_VNIR_12_IDC_x10_C01": {"split": "train", "modalities": {modality: []},
                                     "counts": {modality: 4}, "extra": {}},
        "HSI_VNIR_12_Healthy_x10_C02": {"split": "train", "modalities": {modality: []},
                                         "counts": {modality: 3}, "extra": {}},
        "HSI_VNIR_7_DCIS_x10_C01": {"split": "validation", "modalities": {modality: []},
                                     "counts": {modality: 5}, "extra": {}},
    }
    manifest = {
        "schema": 2, "adapter": "histology_hsi_bc", "args": {}, "split": {},
        "class_names": ["healthy", "DCIS", "IDC"],
        "train_patients": ["12"], "validation_patients": ["7"], "test_patients": [],
        "pass1_state": None, "modalities": [modality],
        "completed_items": items, "skipped_items": {}, "finalized": True,
    }
    save_manifest(out_dir, manifest)

    # y_train.npy: 4 rows label=2 (IDC), 3 rows label=0 (healthy) - sorted key
    # order is "HSI_VNIR_12_Healthy_x10_C02" < "HSI_VNIR_12_IDC_x10_C01" (H<I),
    # so match that order here.
    y_train = np.array([0, 0, 0, 2, 2, 2, 2], dtype=np.int64)
    y_val = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    np.save(os.path.join(out_dir, modality, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, modality, "y_val.npy"), y_val)
    return manifest


def test_derive_hsi_group_sidecars_synthetic():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build_synthetic_manifest(tmp)
        reports = _derive_hsi(tmp, manifest, "hsi", dry_run=False)

        m_dir = os.path.join(tmp, "hsi")
        y_train = np.load(os.path.join(m_dir, "y_train.npy"))
        groups_train = np.load(os.path.join(m_dir, "groups_train.npy"))
        captures_train = np.load(os.path.join(m_dir, "captures_train.npy"))
        assert len(groups_train) == len(y_train) == 7
        assert set(groups_train.tolist()) == {12}
        # two distinct captures for patient 12
        assert len(set(captures_train.tolist())) == 2
        # every capture's labels are single-valued
        for cap in set(captures_train.tolist()):
            labels_here = y_train[captures_train == cap]
            assert len(np.unique(labels_here)) == 1

        y_val = np.load(os.path.join(m_dir, "y_val.npy"))
        groups_val = np.load(os.path.join(m_dir, "groups_val.npy"))
        assert len(groups_val) == len(y_val) == 5
        assert set(groups_val.tolist()) == {7}

        assert reports["train"]["written"] and reports["validation"]["written"]

        with open(os.path.join(m_dir, "capture_index.json")) as f:
            cap_idx = json.load(f)
        assert len(cap_idx) == 3
        patients = {v["patient"] for v in cap_idx.values()}
        assert patients == {"12", "7"}


def test_derive_hsi_aborts_on_count_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build_synthetic_manifest(tmp)
        # corrupt: y_train.npy now has 8 rows instead of 7
        np.save(os.path.join(tmp, "hsi", "y_train.npy"), np.zeros(8, dtype=np.int64))
        reports = _derive_hsi(tmp, manifest, "hsi", dry_run=False)
        assert reports["train"]["written"] is False
        assert not os.path.exists(os.path.join(tmp, "hsi", "groups_train.npy"))


def test_real_dataset_sidecars_if_present():
    real_dir = "data/hsi_v7-80_10_10_importance"
    if not os.path.isdir(real_dir):
        return
    m_dir = os.path.join(real_dir, "hsi")
    groups_path = os.path.join(m_dir, "groups_val.npy")
    if not os.path.isfile(groups_path):
        return
    y_val = np.load(os.path.join(m_dir, "y_val.npy"), mmap_mode="r")
    groups_val = np.load(groups_path)
    assert len(groups_val) == len(y_val)
    # the audit's headline fact: patient 68 sits in validation.
    assert 68 in set(groups_val.tolist())


def test_real_pad_image_sidecars_if_present():
    real_dir = "data/pad_v6"
    if not os.path.isdir(real_dir):
        return
    images_path = os.path.join(real_dir, "images_train.npy")
    if not os.path.isfile(images_path):
        return
    y_train = np.load(os.path.join(real_dir, "y_train.npy"), mmap_mode="r")
    images_train = np.load(images_path)
    assert len(images_train) == len(y_train)
    # ~400 patches per clinical image (11x11 patchification, C-1).
    n_images = len(set(images_train.tolist()))
    assert 300 < (len(images_train) / n_images) < 500


if __name__ == "__main__":
    test_derive_hsi_group_sidecars_synthetic()
    test_derive_hsi_aborts_on_count_mismatch()
    test_real_dataset_sidecars_if_present()
    test_real_pad_image_sidecars_if_present()
    print("OK")
