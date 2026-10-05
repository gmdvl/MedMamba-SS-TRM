# -*- coding: utf-8 -*-
"""tests/test_dataset_prep_unify.py - MedMamba-SS-TRM data-integrity remediation, Part B Stage 1/2.

Run with: pytest test_dataset_prep_unify.py -v
"""
import glob
import json
import os

import numpy as np
import pytest

from training.npy_atomic import save_npy_atomic, verify_against_manifest, _sha256_of_file
from training.npy_integrity import DatasetStorageError
from training.dataset_prep_unify import (
    atomic_array_writer, atomic_memmap_array, atomic_npy_stream, dataset_is_finalized_and_intact,
    finalize_dataset, unify_split, verify_unified_dataset,
)


def _make_shards(tmp_path, n_shards=3, rows=7, shape=(2, 2, 2), start_label=0):
    shards, chunks = [], []
    for s in range(n_shards):
        x = np.random.randn(rows, *shape).astype(np.float32)
        y = ((np.arange(rows) + start_label) % 3).astype(np.int64)
        xs = str(tmp_path / f"s{s}_X.npy")
        ys = str(tmp_path / f"s{s}_y.npy")
        save_npy_atomic(xs, x)
        save_npy_atomic(ys, y)
        shards.append((xs, ys, rows))
        chunks.append(x)
    return shards, np.concatenate(chunks)


# ----------------------------------------------------------------------
# atomic_memmap_array
# ----------------------------------------------------------------------

def test_atomic_writer_success_leaves_no_tmp(tmp_path):
    fp = str(tmp_path / "out.npy")
    with atomic_memmap_array(fp, np.float32, (5, 3)) as a:
        a[:] = 1.0
    assert os.path.isfile(fp)
    assert not glob.glob(fp + ".tmp")
    assert np.array_equal(np.load(fp), np.ones((5, 3), np.float32))


def test_atomic_writer_failure_keeps_prior_file_and_cleans_tmp(tmp_path):
    fp = str(tmp_path / "out.npy")
    with atomic_memmap_array(fp, np.float32, (5, 3)) as a:
        a[:] = 7.0
    prior = np.load(fp).copy()

    with pytest.raises(RuntimeError):
        with atomic_memmap_array(fp, np.float32, (5, 3)) as a:
            a[:] = 9.0
            raise RuntimeError("simulated kill mid-fill")

    assert not glob.glob(fp + ".tmp"), "the .tmp must be removed on failure"
    assert np.array_equal(np.load(fp), prior), "the previous good file must be untouched"


# ----------------------------------------------------------------------
# unify_split
# ----------------------------------------------------------------------

def test_unify_split_preserves_row_order(tmp_path):
    shards, expected = _make_shards(tmp_path)
    ox, oy = str(tmp_path / "X_train.npy"), str(tmp_path / "y_train.npy")
    n, report = unify_split(shards, ox, oy)
    assert n == 21 and report is None
    assert np.allclose(np.load(ox), expected)
    assert not glob.glob(str(tmp_path / "*.tmp"))


def test_unify_split_oversample_equalizes_classes(tmp_path):
    y = np.array([0] * 15 + [1] * 4 + [2] * 2, dtype=np.int64)
    save_npy_atomic(str(tmp_path / "s_X.npy"), np.random.randn(21, 3).astype(np.float32))
    save_npy_atomic(str(tmp_path / "s_y.npy"), y)
    n, report = unify_split(
        [(str(tmp_path / "s_X.npy"), str(tmp_path / "s_y.npy"), 21)],
        str(tmp_path / "X_train.npy"), str(tmp_path / "y_train.npy"),
        balance_method="oversample", seed=0, class_names=["a", "b", "c"])
    counts = np.bincount(np.load(str(tmp_path / "y_train.npy")))
    assert len(set(counts.tolist())) == 1, f"oversample should equalize, got {counts}"
    assert n == counts.sum()
    assert report["method"] == "oversample"


def test_unify_split_rejects_inconsistent_shard(tmp_path):
    shards, _ = _make_shards(tmp_path, n_shards=2)
    # claim a wrong row count for the 2nd shard
    shards[1] = (shards[1][0], shards[1][1], 999)
    with pytest.raises(DatasetStorageError):
        unify_split(shards, str(tmp_path / "X.npy"), str(tmp_path / "y.npy"))


# ----------------------------------------------------------------------
# verify_unified_dataset / finalize_dataset
# ----------------------------------------------------------------------

def test_verify_rejects_all_zero_array(tmp_path):
    z = str(tmp_path / "X_train.npy")
    save_npy_atomic(z, np.zeros((10, 2, 2, 2), np.float32))
    with pytest.raises(DatasetStorageError) as ei:
        verify_unified_dataset(str(tmp_path), {"X_train": z}, level="structural")
    assert "entirely zero" in str(ei.value)


def test_verify_rejects_truncated_array(tmp_path):
    shards, _ = _make_shards(tmp_path)
    ox, oy = str(tmp_path / "X_train.npy"), str(tmp_path / "y_train.npy")
    unify_split(shards, ox, oy)
    with open(ox, "r+b") as f:
        f.truncate(os.path.getsize(ox) // 2)
    with pytest.raises(DatasetStorageError):
        verify_unified_dataset(str(tmp_path), {"X_train": ox, "y_train": oy}, level="structural")


def test_finalize_writes_manifest_and_roundtrips(tmp_path):
    shards, _ = _make_shards(tmp_path)
    ox, oy = str(tmp_path / "X_train.npy"), str(tmp_path / "y_train.npy")
    unify_split(shards, ox, oy)
    roles = {"X_train": ox, "y_train": oy}
    assert finalize_dataset(str(tmp_path), roles, manifest_extra={"prep": "test"}) is True
    assert verify_against_manifest(str(tmp_path)) == []
    assert dataset_is_finalized_and_intact(str(tmp_path), roles)

    with open(ox, "r+b") as f:
        f.seek(200)
        f.write(b"\x00\x00\x00\x00")
    assert verify_against_manifest(str(tmp_path)), "a 1-word flip must break the sha256 manifest"


# ----------------------------------------------------------------------
# Stage 1 additions: identity fast path, folded sha256, probe verify level
# ----------------------------------------------------------------------

def test_identity_fast_path_matches_scatter_path(tmp_path):
    shards, expected = _make_shards(tmp_path, n_shards=4, rows=6, shape=(3, 3, 5))
    ox_fast, oy_fast = str(tmp_path / "Xf.npy"), str(tmp_path / "yf.npy")
    ox_sc, oy_sc = str(tmp_path / "Xs.npy"), str(tmp_path / "ys.npy")
    n1, r1 = unify_split(shards, ox_fast, oy_fast)                       # identity path
    n2, r2 = unify_split(shards, ox_sc, oy_sc, row_order=list(range(24)))  # scatter path
    assert n1 == n2 == 24 and r1 is None and r2 is None
    assert np.array_equal(np.load(ox_fast), np.load(ox_sc))
    assert np.array_equal(np.load(ox_fast), expected)
    assert np.array_equal(np.load(oy_fast), np.load(oy_sc))


def test_folded_sha256_equals_file_hash(tmp_path):
    shards, _ = _make_shards(tmp_path, n_shards=3, rows=5, shape=(2, 2, 4))
    ox, oy = str(tmp_path / "X_train.npy"), str(tmp_path / "y_train.npy")
    sha = {}
    unify_split(shards, ox, oy, sha_out=sha)
    assert sha["X_train.npy"] == _sha256_of_file(ox)
    # and finalize with the precomputed digest still produces a verifiable manifest
    finalize_dataset(str(tmp_path), {"X_train": ox, "y_train": oy},
                     verify_level="probe", precomputed_sha256=sha)
    assert verify_against_manifest(str(tmp_path)) == []


def test_probe_level_still_catches_all_zero_and_truncation(tmp_path):
    z = str(tmp_path / "X_train.npy")
    save_npy_atomic(z, np.zeros((8, 2, 2, 2), np.float32))
    with pytest.raises(DatasetStorageError):
        verify_unified_dataset(str(tmp_path), {"X_train": z}, level="probe")

    shards, _ = _make_shards(tmp_path)
    ox, oy = str(tmp_path / "X2.npy"), str(tmp_path / "y2.npy")
    unify_split(shards, ox, oy)
    with open(ox, "r+b") as f:
        f.truncate(os.path.getsize(ox) // 2)
    with pytest.raises(DatasetStorageError):
        verify_unified_dataset(str(tmp_path), {"X_train": ox}, level="probe")


def test_atomic_array_writer_commits_and_folds_hash(tmp_path):
    # Mirrors how unify_split uses it: fill front-to-back, feeding the same
    # hasher, so holder.sha256 == the full-file digest with no read-back.
    payload = np.arange(12, dtype=np.float32).reshape(4, 3)
    fp = str(tmp_path / "out.npy")
    with atomic_array_writer(fp, np.float32, (4, 3), sha256=True) as (arr, holder):
        arr[:] = payload
        holder.hasher.update(payload.tobytes())
    assert os.path.isfile(fp) and not glob.glob(fp + ".tmp")
    assert holder.sha256 == _sha256_of_file(fp)
    assert np.array_equal(np.load(fp), payload)


# ----------------------------------------------------------------------
# atomic_npy_stream / streaming unify (mmap-writeback-race remediation)
#
# The unify writer was switched from open_memmap fills to plain write()
# after two consecutive prep runs on this project produced X_train.npy
# files with unreadable byte ranges (OSError(5)/EIO). In the same run,
# 45 GiB of shards written with buffered np.save were completely clean
# while a 35 GiB mmap-filled X_train.npy had six bad regions - the mapping
# outgrew RAM, so kernel writeback raced the fill. See
# dataset_prep_unify.atomic_npy_stream's docstring for the full account.
# ----------------------------------------------------------------------

def test_atomic_npy_stream_commits_and_folds_hash(tmp_path):
    payload = np.arange(12, dtype=np.float32).reshape(4, 3)
    fp = str(tmp_path / "out.npy")
    with atomic_npy_stream(fp, np.float32, (4, 3), sha256=True) as (fh, holder):
        fh.write(payload.tobytes())
        holder.hasher.update(payload.tobytes())
    assert os.path.isfile(fp) and not glob.glob(fp + ".tmp")
    assert holder.sha256 == _sha256_of_file(fp)
    assert np.array_equal(np.load(fp), payload)


def test_atomic_npy_stream_short_fill_raises_and_leaves_no_files(tmp_path):
    fp = str(tmp_path / "out.npy")
    with pytest.raises(DatasetStorageError):
        with atomic_npy_stream(fp, np.float32, (4, 3)) as (fh, _holder):
            fh.write(np.zeros((2, 3), dtype=np.float32).tobytes())  # half the declared rows
    assert not os.path.isfile(fp), "a short fill must never be renamed into place"
    assert not glob.glob(fp + ".tmp"), "the .tmp must be removed on a short fill"


def test_scatter_path_streams_shuffled_and_repeated_row_order(tmp_path):
    # Exercises the rewritten scatter/balanced path (now streaming, gathering
    # per output-batch instead of scattering into an mmap) with a row_order
    # that is neither identity nor a single pass over each shard: shuffled,
    # and with some rows repeated - so a batch draws from multiple shards
    # out of their original order.
    shards, expected = _make_shards(tmp_path, n_shards=3, rows=5, shape=(2, 2, 3))
    rng = np.random.default_rng(0)
    row_order = rng.permutation(15)
    row_order = np.concatenate([row_order, row_order[:4]])  # repeat 4 rows
    ox, oy = str(tmp_path / "X.npy"), str(tmp_path / "y.npy")
    n, report = unify_split(shards, ox, oy, row_order=row_order.tolist(), read_batch=4)
    assert n == len(row_order) and report is None
    assert np.array_equal(np.load(ox), expected[row_order])
    assert not glob.glob(str(tmp_path / "*.tmp"))


def test_finalize_deep_verify_overrides_wrong_precomputed_sha256(tmp_path):
    # The manifest sha256 used to come ONLY from a digest computed in RAM
    # while writing (`unify_split(sha_out=...)`), which describes what was
    # INTENDED and can never detect a write that landed differently on disk -
    # exactly the failure this whole change addresses. finalize_dataset now
    # re-hashes during the deep read-back it already performs and prefers
    # THAT digest. Simulate "a write that landed differently than what was
    # hashed in RAM" with a deliberately wrong precomputed digest and check
    # deep verify overrides it with the real one.
    shards, _ = _make_shards(tmp_path, n_shards=3, rows=6, shape=(2, 2, 3))
    ox, oy = str(tmp_path / "X_train.npy"), str(tmp_path / "y_train.npy")
    unify_split(shards, ox, oy)  # no sha_out: this is the real on-disk file
    real_digest = _sha256_of_file(ox)
    wrong_precomputed = {"X_train.npy": "0" * 64}

    finalize_dataset(str(tmp_path), {"X_train": ox, "y_train": oy},
                     verify_level="deep", precomputed_sha256=wrong_precomputed)

    manifest = json.load(open(str(tmp_path / "dataset_manifest.json")))
    assert manifest["files"]["X_train.npy"]["sha256"] == real_digest, (
        "deep verify must re-hash from disk and override a wrong in-RAM digest")
    assert verify_against_manifest(str(tmp_path)) == []


def test_finalize_non_deep_verify_trusts_precomputed_sha256_verbatim(tmp_path):
    # Companion to the test above: without a deep read-back there is no
    # on-disk digest to prefer, so a wrong precomputed one is (as before)
    # written through unchecked - this is exactly the gap deep verify closes.
    shards, _ = _make_shards(tmp_path, n_shards=2, rows=4, shape=(2, 2, 2))
    ox, oy = str(tmp_path / "X_train.npy"), str(tmp_path / "y_train.npy")
    unify_split(shards, ox, oy)
    wrong_precomputed = {"X_train.npy": "0" * 64}

    finalize_dataset(str(tmp_path), {"X_train": ox, "y_train": oy},
                     verify_level="structural", precomputed_sha256=wrong_precomputed)

    manifest = json.load(open(str(tmp_path / "dataset_manifest.json")))
    assert manifest["files"]["X_train.npy"]["sha256"] == "0" * 64
    assert any("sha256 mismatch" in p for p in verify_against_manifest(str(tmp_path)))


def test_verify_against_manifest_distinguishes_header_vs_payload_errors(tmp_path, monkeypatch):
    # A payload that is physically unreadable (OSError(5)/EIO - the failure
    # mode this whole fix targets) used to be reported as "header unreadable",
    # which sent debugging toward the wrong layer entirely (see
    # npy_atomic.verify_against_manifest). Simulate an EIO specifically
    # during hashing (the header parse itself must still succeed) and check
    # the message names the payload, not the header.
    shards, _ = _make_shards(tmp_path, n_shards=2, rows=4, shape=(2, 2, 2))
    ox, oy = str(tmp_path / "X_train.npy"), str(tmp_path / "y_train.npy")
    unify_split(shards, ox, oy)
    finalize_dataset(str(tmp_path), {"X_train": ox, "y_train": oy}, verify_level="structural")

    import training.npy_atomic as npy_atomic

    def _boom(path, chunk=8 << 20):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(npy_atomic, "_sha256_of_file", _boom)
    problems = npy_atomic.verify_against_manifest(str(tmp_path))
    assert any("payload unreadable while hashing" in p for p in problems)
    assert not any("header unreadable" in p for p in problems)
