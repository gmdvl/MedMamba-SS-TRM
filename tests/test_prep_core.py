# -*- coding: utf-8 -*-
"""tests/test_prep_core.py - the generic training.prep pipeline (Stage 2).

Run with: pytest test_prep_core.py -v   (or the repo's pytest-free runner)
"""
import json
import os

import numpy as np
import pytest

from training.npy_atomic import verify_against_manifest
from training.prep import DatasetAdapter, ModalitySpec, PrepItem, run_prep


class _FakeAdapter(DatasetAdapter):
    name = "fake"
    flat_output = True
    RAISE_ON = set()          # item keys whose load_item raises (skip path)

    @property
    def class_names(self):
        return ["a", "b", "c"]

    def add_cli_args(self, p):
        p.add_argument("--n_items", type=int, default=60)
        p.add_argument("--patches", type=int, default=4)

    def compat_keys(self, args):
        return {"patches": args.patches}

    def discover(self, args):
        for i in range(args.n_items):
            yield PrepItem(key=f"it{i:03d}", group_id=f"g{i % 15}", label=i % 3,
                            payload={"seed": i})

    def modality_spec(self, modality, args, state):
        return ModalitySpec(store_dtype="float16", value_clip=(0.0, 1.0), sample_shape=(4, 4, 5))

    def expected_sample_count(self, item, modality, args, state):
        return args.patches

    def load_item(self, item, modality, args, state):
        if item.key in self.RAISE_ON:
            raise RuntimeError(f"boom {item.key}")
        rng = np.random.default_rng(item.payload["seed"])
        for _ in range(args.patches):
            yield rng.random((4, 4, 5), dtype=np.float64).astype(np.float32), item.label


def _run(d, *extra):
    return run_prep(_FakeAdapter(), ["--root", "x", "--out_dir", str(d), "--split", "70/15/15",
                                      "--seed", "1", *extra])


def _load(d, name):
    return np.load(os.path.join(str(d), name))


def test_end_to_end_artifacts_and_contract(tmp_path):
    assert _run(tmp_path, "--num_workers", "3", "--batch_size", "16") == 0
    for f in ("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy",
              "class_names.json", "dataset_manifest.json", "split_assignment.json",
              "dataset_statistics.json", "_progress.json", "class_coverage_report.json",
              "class_coverage_report.pre.json", "prep_log.txt"):
        assert os.path.exists(os.path.join(str(tmp_path), f)), f"missing {f}"
    X, y = _load(tmp_path, "X_train.npy"), _load(tmp_path, "y_train.npy")
    assert X.dtype == np.float16 and X.shape[1:] == (4, 4, 5)      # NHWC, float16
    assert y.dtype == np.int64 and len(y) == len(X)
    assert json.load(open(os.path.join(str(tmp_path), "class_names.json"))) == ["a", "b", "c"]
    assert verify_against_manifest(str(tmp_path)) == []


def test_worker_count_does_not_change_bytes(tmp_path):
    d1, d4 = tmp_path / "w1", tmp_path / "w4"
    d1.mkdir(); d4.mkdir()
    _run(d1, "--num_workers", "1")
    _run(d4, "--num_workers", "4")
    for nm in ("X_train.npy", "X_val.npy", "X_test.npy", "y_train.npy", "y_test.npy"):
        assert np.array_equal(_load(d1, nm), _load(d4, nm)), f"{nm} depends on worker count"


def test_resume_after_partial_completion(tmp_path):
    # full run keeping the shards, then drop half the completed items and re-run
    assert _run(tmp_path, "--num_workers", "2", "--keep_batches") == 0
    full = _load(tmp_path, "X_train.npy").copy()
    mpath = os.path.join(str(tmp_path), "_progress.json")
    man = json.load(open(mpath))
    keys = sorted(man["completed_items"])
    for k in keys[: len(keys) // 2]:
        del man["completed_items"][k]
    man["finalized"] = False
    json.dump(man, open(mpath, "w"))
    # remove the finalized arrays so the fast path can't short-circuit
    for on in ("train", "val", "test"):
        for pre in ("X_", "y_"):
            os.remove(os.path.join(str(tmp_path), f"{pre}{on}.npy"))
    assert _run(tmp_path, "--num_workers", "2", "--keep_batches") == 0
    assert np.array_equal(_load(tmp_path, "X_train.npy"), full), "resume did not reproduce the data"


def test_skip_is_fail_soft_and_recorded(tmp_path):
    _FakeAdapter.RAISE_ON = {"it005", "it017"}
    try:
        assert _run(tmp_path, "--num_workers", "0") == 0
    finally:
        _FakeAdapter.RAISE_ON = set()
    man = json.load(open(os.path.join(str(tmp_path), "_progress.json")))
    assert set(man["skipped_items"]) == {"it005", "it017"}
    assert "it005" not in man["completed_items"]
    # 58 of 60 items x 4 patches = 232 train+val+test samples total
    total = sum(len(_load(tmp_path, f"y_{s}.npy")) for s in ("train", "val", "test"))
    assert total == 58 * 4


def test_dry_run_writes_nothing_heavy(tmp_path):
    assert _run(tmp_path, "--dry_run") == 0
    assert not os.path.exists(os.path.join(str(tmp_path), "X_train.npy"))
    assert os.path.exists(os.path.join(str(tmp_path), "split_assignment.json"))


def test_writer_direct_matches_shard_mode(tmp_path):
    ds, dd = tmp_path / "shards", tmp_path / "direct"
    ds.mkdir(); dd.mkdir()
    _run(ds, "--num_workers", "2", "--writer", "shards")
    _run(dd, "--num_workers", "2", "--writer", "direct")
    for nm in ("X_train.npy", "X_val.npy", "X_test.npy", "y_train.npy", "y_test.npy"):
        a, b = _load(ds, nm), _load(dd, nm)
        assert a.shape == b.shape and a.dtype == b.dtype
        # rows are the same multiset (direct orders strictly by item key; shard mode
        # orders by item key too via --deterministic_order) -> exact match
        assert np.array_equal(a, b), f"{nm} differs between shard and direct writers"
    from training.npy_atomic import verify_against_manifest
    assert verify_against_manifest(str(dd)) == []


def test_resolve_verify_level_auto_is_always_deep_regardless_of_size():
    # Regression test: 'auto' used to mean "deep below --deep_verify_max_gb,
    # else probe", which silently skipped the end-to-end read-back for any
    # dataset bigger than the 4 GB default - the gap that let a corrupt
    # multi-GB X_train.npy reach training undetected. 'auto' must now mean
    # deep unconditionally; deep_verify_max_gb is accepted but has no effect.
    from types import SimpleNamespace
    from training.prep.core import _resolve_verify_level

    small = SimpleNamespace(no_verify_output=False, verify_level="auto", deep_verify_max_gb=4.0)
    huge = SimpleNamespace(no_verify_output=False, verify_level="auto", deep_verify_max_gb=0.0001)
    assert _resolve_verify_level(small) == "deep"
    assert _resolve_verify_level(huge) == "deep"

    # an explicit level still wins over 'auto'
    explicit = SimpleNamespace(no_verify_output=False, verify_level="probe", deep_verify_max_gb=4.0)
    assert _resolve_verify_level(explicit) == "probe"

    # --no_verify_output still short-circuits to the cheapest level
    off = SimpleNamespace(no_verify_output=True, verify_level="auto", deep_verify_max_gb=4.0)
    assert _resolve_verify_level(off) == "structural"
