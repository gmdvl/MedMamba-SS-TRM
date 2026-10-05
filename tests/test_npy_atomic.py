# -*- coding: utf-8 -*-
"""tests/test_npy_atomic.py - MedMamba-SS-TRM v13 plan, Workstream 6.

Run with: pytest test_npy_atomic.py -v
"""
import os

import numpy as np
import pytest

from training.npy_atomic import (
    save_npy_atomic, write_dataset_manifest, verify_against_manifest, MANIFEST_NAME,
)


@pytest.fixture()
def arr():
    return np.random.randn(16, 3, 5, 5).astype(np.float32)


def test_save_atomic_leaves_no_tmp(tmp_path, arr):
    p = tmp_path / "X_train.npy"
    save_npy_atomic(str(p), arr)
    assert p.is_file()
    assert not (tmp_path / "X_train.npy.tmp").exists()
    assert np.array_equal(np.load(p), arr)


def test_save_atomic_replaces_existing(tmp_path, arr):
    p = tmp_path / "X_train.npy"
    np.save(p, np.zeros((2, 2), dtype=np.float32))
    save_npy_atomic(str(p), arr)
    assert np.array_equal(np.load(p), arr)


def test_interrupted_write_leaves_original_intact(tmp_path, arr):
    """A `.tmp` that never got `os.replace`d must not shadow the real file."""
    p = tmp_path / "X_train.npy"
    save_npy_atomic(str(p), arr)
    # simulate a crashed writer: a stray .tmp from a partial run
    (tmp_path / "X_train.npy.tmp").write_bytes(b"partial")
    assert np.array_equal(np.load(p), arr)  # original still the good one


def test_manifest_roundtrip_ok(tmp_path, arr):
    save_npy_atomic(str(tmp_path / "X_train.npy"), arr)
    save_npy_atomic(str(tmp_path / "y_train.npy"), np.zeros(16, dtype=np.int64))
    write_dataset_manifest(str(tmp_path), ["X_train.npy", "y_train.npy"])
    assert (tmp_path / MANIFEST_NAME).is_file()
    assert verify_against_manifest(str(tmp_path)) == []


def test_manifest_detects_size_change(tmp_path, arr):
    p = tmp_path / "X_train.npy"
    save_npy_atomic(str(p), arr)
    write_dataset_manifest(str(tmp_path), ["X_train.npy"])
    with open(p, "ab") as f:
        f.write(b"\x00" * 64)
    problems = verify_against_manifest(str(tmp_path))
    assert problems and "size" in problems[0].lower()


def test_manifest_detects_missing_file(tmp_path, arr):
    save_npy_atomic(str(tmp_path / "X_train.npy"), arr)
    write_dataset_manifest(str(tmp_path), ["X_train.npy"])
    os.remove(tmp_path / "X_train.npy")
    problems = verify_against_manifest(str(tmp_path))
    assert problems and "missing" in problems[0].lower()


def test_verify_no_manifest_is_empty(tmp_path):
    assert verify_against_manifest(str(tmp_path)) == []


def test_manifest_sha256(tmp_path, arr):
    p = tmp_path / "X_train.npy"
    save_npy_atomic(str(p), arr)
    write_dataset_manifest(str(tmp_path), ["X_train.npy"], sha256=True)
    assert verify_against_manifest(str(tmp_path)) == []
    # rewrite same shape/dtype/size but different content
    save_npy_atomic(str(p), np.random.randn(*arr.shape).astype(arr.dtype))
    problems = verify_against_manifest(str(tmp_path))
    assert problems and "sha256" in problems[0].lower()
