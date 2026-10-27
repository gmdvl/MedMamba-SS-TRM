# -*- coding: utf-8 -*-
"""
test_synthetic_mix.py
=======================
`training/synthetic_mix.py`: the synthetic export schema, its loader, the
real + synthetic training set and the minority plan over it; and its
`train.py --synthetic_dir` wiring (`training/train_pipeline.py`).

The properties everything else rests on:
  - with no synthetic rows and no copy mask, `plan_minority_with_synthetic` IS
    `plan_minority_augmentation`, so the default path cannot move;
  - a real row of the mixed set is exactly the real `NpyDatasetV16` item, and
    a synthetic row is normalized with the REAL statistics;
  - a synthetic row is never a mixing partner and never copied;
  - the loader refuses an export it cannot vouch for, each with its own error;
  - `train.py` with an EMPTY export trains bit-identically to `train.py`
    without one, and refuses a bad export before it reads the real build.

Every fixture is a few dozen 5 x 5 x 8 patches in `tmp_path`; the `train.py`
runs train a tiny model on 8 x 8 x 8 patches on the CPU (~8 s each).

Run with: pytest tests/test_synthetic_mix.py -q
"""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Subset

from training.minority_augmentation import MinorityAugmentedDataset, plan_minority_augmentation, root_indices
from training.npy_atomic import (
    MANIFEST_NAME, save_json_atomic, verify_against_manifest, write_dataset_manifest,
)
from training.npy_data import NpyDatasetV16, compute_global_channel_stats
from training.synthetic_mix import (
    EXPORT_ARRAYS, GEN_INDEX_NAME, GROUPS_SYN_NAME, PROVENANCE_NAME, SCHEMA_VERSION, SYNTHETIC_GROUP_IDS,
    X_SYN_NAME, Y_SYN_NAME, SyntheticExportError, attach_synthetic, load_synthetic_export,
    plan_minority_with_synthetic, write_synthetic_export,
)
from training.train_cli import resolve
from training.train_pipeline import _class_weight_labels, _load_copy_mask, build_run_name
from training.train_preflight import _probe_in_channels

REPO_ROOT = Path(__file__).resolve().parent.parent

NAMES =["healthy", "DCIS", "IDC"]
HWC = (5, 5, 8)
REAL_Y = np.repeat([0, 1, 2], [12, 4, 14]).astype(np.int64)
REAL_GROUPS = (np.arange(len(REAL_Y)) % 3 + 10 * REAL_Y).astype(np.int32)   # DCIS: patients 10, 11, 12


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _real_build(root: Path, manifest: bool = True, seed: int = 0) -> Path:
    d = root / "real"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    np.save(d / "X_train.npy", (rng.random((len(REAL_Y), *HWC)) * 0.6 + 0.1).astype(np.float16))
    np.save(d / "y_train.npy", REAL_Y)
    np.save(d / "groups_train.npy", REAL_GROUPS)
    np.save(d / "wavelengths.npy", np.linspace(450, 900, HWC[2]).astype(np.float32))
    (d / "class_names.json").write_text(json.dumps(NAMES))
    if manifest:
        write_dataset_manifest(str(d), ["X_train.npy", "y_train.npy", "groups_train.npy", "wavelengths.npy"],
                               sha256=True)
    return d


def _syn_x(n: int, seed: int = 1) -> np.ndarray:
    return (np.random.default_rng(seed).random((n, *HWC)) * 0.5 + 0.3).astype(np.float32)


def _export(root: Path, real: Path, n: int = 6, source: str = "cvae", label: int = 1, **kw) -> Path:
    out = root / f"export_{source}"
    write_synthetic_export(str(out), _syn_x(n), np.full(n, label), source, str(real), **kw)
    return out


def _load(export: Path, real: Path, **kw):
    return load_synthetic_export(str(export), str(real), len(NAMES), HWC, **kw)


def _reseal(export: Path) -> None:
    """Re-write the manifest and export_id after a test edits a file, so the
    edit is the ONLY thing wrong with the export."""
    arrays = list(EXPORT_ARRAYS) + (["wavelengths.npy"] if (export / "wavelengths.npy").is_file() else [])
    write_dataset_manifest(str(export), arrays, sha256=True)
    prov = json.loads((export / PROVENANCE_NAME).read_text())
    prov["export_id"] = _sha256(export / X_SYN_NAME)[:12]
    save_json_atomic(str(export / PROVENANCE_NAME), prov)


def _edit_provenance(export: Path, **changes) -> None:
    prov = json.loads((export / PROVENANCE_NAME).read_text())
    prov.update(changes)
    save_json_atomic(str(export / PROVENANCE_NAME), prov)


@pytest.fixture
def real(tmp_path) -> Path:
    return _real_build(tmp_path)


# ============================================================================
# writer
# ============================================================================

def test_writer_writes_a_consistent_export(tmp_path, real):
    exp = _export(tmp_path, real, n=6, extra_provenance={"seeds": [3], "generator_ckpt_sha256": "ab"})
    prov = json.loads((exp / PROVENANCE_NAME).read_text())
    assert prov["schema_version"] == SCHEMA_VERSION
    assert prov["export_id"] == _sha256(exp / X_SYN_NAME)[:12]
    assert prov["source_manifest_sha256"] == _sha256(real / MANIFEST_NAME)
    assert (prov["source"], prov["group_id"], prov["n_rows"], prov["hwc"]) == ("cvae", -1, 6, list(HWC))
    assert prov["class_counts"] == {"healthy": 0, "DCIS": 6, "IDC": 0}
    assert prov["seeds"] == [3] and prov["generator_ckpt_sha256"] == "ab"

    assert verify_against_manifest(str(exp)) == []
    listed = json.loads((exp / MANIFEST_NAME).read_text())["files"]
    assert set(listed) == set(EXPORT_ARRAYS) | {"wavelengths.npy"}
    assert listed[X_SYN_NAME]["sha256"][:12] == prov["export_id"]

    X = np.load(exp / X_SYN_NAME)
    assert X.dtype == np.float16 and X.shape == (6, *HWC)
    np.testing.assert_array_equal(X, _syn_x(6).astype(np.float16))
    assert np.load(exp / Y_SYN_NAME).dtype == np.int64
    groups = np.load(exp / GROUPS_SYN_NAME)
    assert groups.dtype == np.int32 and groups.tolist() == [-1] * 6
    assert np.load(exp / GEN_INDEX_NAME).tolist() == list(range(6))
    assert (exp / "wavelengths.npy").read_bytes() == (real / "wavelengths.npy").read_bytes()
    assert json.loads((exp / "class_names.json").read_text()) == NAMES


def test_ldm_rows_carry_group_minus_two_and_unknown_sources_are_refused(tmp_path, real):
    exp = _export(tmp_path, real, source="ldm", gen_index=np.array([4, 9, 10, 11, 20, 21]))
    assert np.load(exp / GROUPS_SYN_NAME).tolist() == [-2] * 6
    assert np.load(exp / GEN_INDEX_NAME).tolist() == [4, 9, 10, 11, 20, 21]
    assert _load(exp, real).source == "ldm"
    assert set(SYNTHETIC_GROUP_IDS.values()) == {-1, -2}
    with pytest.raises(SyntheticExportError, match="unknown source"):
        _export(tmp_path, real, source="gan")


@pytest.mark.parametrize("bad, match", [
    ({"extra_provenance": {"export_id": "x"}}, "may not set"),
    ({"gen_index": np.arange(5)}, "gen_index"),
])
def test_writer_refuses_bad_arguments(tmp_path, real, bad, match):
    with pytest.raises(SyntheticExportError, match=match):
        _export(tmp_path, real, **bad)


def test_writer_refuses_values_that_are_not_finite_after_the_float16_cast(tmp_path, real):
    X = _syn_x(4)
    X[2, 0, 0, 0] = 70000.0                   # finite in float32, +inf in float16
    with pytest.raises(SyntheticExportError, match="1 row"):
        write_synthetic_export(str(tmp_path / "e"), X, np.ones(4), "cvae", str(real))
    X[2, 0, 0, 0], X[3, 1, 1, 1] = np.nan, -np.inf
    with pytest.raises(SyntheticExportError, match=r"2 row\(s\).*first: row 2"):
        write_synthetic_export(str(tmp_path / "e"), X, np.ones(4), "cvae", str(real))


def test_writer_refuses_an_unbound_build_bad_shapes_and_labels(tmp_path, real):
    bare = _real_build(tmp_path / "b", manifest=False)
    with pytest.raises(SyntheticExportError, match="no dataset_manifest.json"):
        write_synthetic_export(str(tmp_path / "e"), _syn_x(2), np.ones(2), "cvae", str(bare))
    with pytest.raises(SyntheticExportError, match="real build's"):
        write_synthetic_export(str(tmp_path / "e"), np.zeros((2, 5, 5, 9)), np.ones(2), "cvae", str(real))
    with pytest.raises(SyntheticExportError, match="outside"):
        write_synthetic_export(str(tmp_path / "e"), _syn_x(2), np.array([1, 3]), "cvae", str(real))
    with pytest.raises(SyntheticExportError, match="integer labels"):
        write_synthetic_export(str(tmp_path / "e"), _syn_x(2), np.array([1.0, 1.0]), "cvae", str(real))
    with pytest.raises(SyntheticExportError, match="own directory"):
        write_synthetic_export(str(real), _syn_x(2), np.ones(2), "cvae", str(real))


# ============================================================================
# loader: accepts / refuses
# ============================================================================

def test_loader_accepts_the_writers_export(tmp_path, real):
    exp = _export(tmp_path, real, n=6)
    e = _load(exp, real)
    assert e.n_syn == 6 and e.hwc == HWC and e.source == "cvae"
    assert e.export_id == json.loads((exp / PROVENANCE_NAME).read_text())["export_id"]
    assert e.y.tolist() == [1] * 6 and e.groups.tolist() == [-1] * 6
    assert Path(e.x_path) == exp / X_SYN_NAME and Path(e.y_path) == exp / Y_SYN_NAME
    assert load_synthetic_export(str(exp), str(real), 3, hwc=None).hwc == HWC   # read from X_train.npy


def test_loader_refuses_an_export_of_another_build(tmp_path, real):
    exp = _export(tmp_path, real)
    # same files, re-manifested as another build: its manifest bytes change
    write_dataset_manifest(str(real), ["X_train.npy", "y_train.npy", "wavelengths.npy"], sha256=True,
                           extra={"build": "v9"})
    with pytest.raises(SyntheticExportError, match="different build"):
        _load(exp, real)
    assert _load(exp, real, build_check="off").n_syn == 6


def test_loader_refuses_a_real_build_without_manifest_unless_the_build_check_is_off(tmp_path, real, capsys):
    exp = _export(tmp_path, real)
    (real / MANIFEST_NAME).unlink()
    with pytest.raises(SyntheticExportError, match="cannot be matched"):
        _load(exp, real)
    assert _load(exp, real, build_check="off").n_syn == 6
    assert "WARNING (build check off)" in capsys.readouterr().out


def test_loader_refuses_a_schema_mismatch(tmp_path, real):
    exp = _export(tmp_path, real)
    _edit_provenance(exp, schema_version=SCHEMA_VERSION + 1)
    with pytest.raises(SyntheticExportError, match="schema_version"):
        _load(exp, real)


def test_loader_refuses_a_non_finite_value(tmp_path, real):
    exp = _export(tmp_path, real)
    X = np.load(exp / X_SYN_NAME)
    X[4, 2, 2, 7] = np.float16(np.inf)
    np.save(exp / X_SYN_NAME, X)
    _reseal(exp)
    with pytest.raises(SyntheticExportError, match="first: row 4"):
        _load(exp, real)


@pytest.mark.parametrize("name, dtype", [(X_SYN_NAME, np.float32), (Y_SYN_NAME, np.int32),
                                         (GROUPS_SYN_NAME, np.int64)])
def test_loader_refuses_a_wrong_dtype(tmp_path, real, name, dtype):
    exp = _export(tmp_path, real)
    np.save(exp / name, np.load(exp / name).astype(dtype))
    _reseal(exp)
    with pytest.raises(SyntheticExportError, match=f"{name}: dtype"):
        _load(exp, real)


def test_loader_refuses_the_wrong_patch_shape(tmp_path, real):
    exp = _export(tmp_path, real)
    with pytest.raises(SyntheticExportError, match="expected"):
        load_synthetic_export(str(exp), str(real), 3, (5, 5, 9))


@pytest.mark.parametrize("label", [3, -1])
def test_loader_refuses_a_label_outside_the_classes(tmp_path, real, label):
    exp = _export(tmp_path, real)
    y = np.load(exp / Y_SYN_NAME)
    y[1] = label
    np.save(exp / Y_SYN_NAME, y)
    _reseal(exp)
    with pytest.raises(SyntheticExportError, match=rf"labels \[{label}\]"):
        _load(exp, real)


def test_loader_refuses_other_wavelengths_or_class_names(tmp_path, real):
    exp = _export(tmp_path, real)
    np.save(exp / "wavelengths.npy", np.linspace(450, 900, HWC[2]).astype(np.float32) + 1.0)
    _reseal(exp)
    with pytest.raises(SyntheticExportError, match="wavelengths.npy"):
        _load(exp, real)

    exp = _export(tmp_path / "b", real)
    (exp / "class_names.json").write_text(json.dumps(["healthy", "IDC", "DCIS"]))
    with pytest.raises(SyntheticExportError, match="class_names.json"):
        _load(exp, real)


def test_loader_refuses_groups_that_are_not_the_sources(tmp_path, real):
    exp = _export(tmp_path, real)
    np.save(exp / GROUPS_SYN_NAME, np.array([-1, -1, 7, -1, -1, -1], dtype=np.int32))
    _reseal(exp)
    with pytest.raises(SyntheticExportError, match="every row must be -1"):
        _load(exp, real)


def test_loader_refuses_incomplete_and_inconsistent_exports(tmp_path, real):
    exp = _export(tmp_path, real)
    np.save(exp / GEN_INDEX_NAME, np.arange(5))
    _reseal(exp)
    with pytest.raises(SyntheticExportError, match=r"gen_index.npy: shape \(5,\)"):
        _load(exp, real)
    (exp / PROVENANCE_NAME).unlink()
    with pytest.raises(SyntheticExportError, match="incomplete"):
        _load(exp, real)
    with pytest.raises(SyntheticExportError, match="no export directory"):
        _load(tmp_path / "nowhere", real)


def test_a_truncated_array_is_refused_before_it_is_memory_mapped(tmp_path, real):
    exp = _export(tmp_path, real)
    raw = (exp / X_SYN_NAME).read_bytes()
    (exp / X_SYN_NAME).write_bytes(raw[:-10])
    with pytest.raises(SyntheticExportError, match="truncated"):
        _load(exp, real, manifest_check="off")


def test_manifest_check_follows_the_train_py_flag(tmp_path, real, capsys):
    exp = _export(tmp_path, real)
    X = np.load(exp / X_SYN_NAME)
    X[0, 0, 0, 0] += np.float16(0.25)            # same shape and dtype, other bytes
    np.save(exp / X_SYN_NAME, X)
    for mode in ("auto", "strict"):
        with pytest.raises(SyntheticExportError, match="sha256 mismatch"):
            _load(exp, real, manifest_check=mode)
    assert _load(exp, real, manifest_check="off").n_syn == 6

    _reseal(exp)
    _edit_provenance(exp, export_id="0" * 12)    # provenance of some other export
    with pytest.raises(SyntheticExportError, match="another export"):
        _load(exp, real)

    _reseal(exp)
    (exp / MANIFEST_NAME).unlink()
    with pytest.raises(SyntheticExportError, match="strict"):
        _load(exp, real, manifest_check="strict")
    assert _load(exp, real, manifest_check="auto").n_syn == 6
    assert "structural validation only" in capsys.readouterr().out
    with pytest.raises(ValueError):
        _load(exp, real, manifest_check="on")


# ============================================================================
# the mixed training set
# ============================================================================

def _builder(real: Path, normalization="global_zscore", return_norm_stats=True, storage_mode="mmap"):
    mean, std = compute_global_channel_stats(str(real / "X_train.npy"))   # REAL X_train only

    def build(xp, yp):                                 # train_pipeline._build_dataset
        return NpyDatasetV16(xp, yp, storage_mode=storage_mode, normalization=normalization, global_mean=mean,
                             global_std=std, return_norm_stats=return_norm_stats)
    return build, mean, std


def _mixed(tmp_path, real, n=6, groups=REAL_GROUPS, **kw):
    build, mean, std = _builder(real, **kw)
    real_ds = build(str(real / "X_train.npy"), str(real / "y_train.npy"))
    mix = attach_synthetic(real_ds, REAL_Y, groups, _load(_export(tmp_path, real, n=n), real), build)
    return mix, real_ds, mean, std


def test_real_rows_are_the_real_items_and_synthetic_rows_use_the_real_stats(tmp_path, real):
    mix, real_ds, mean, std = _mixed(tmp_path, real)
    assert isinstance(mix.dataset, ConcatDataset)
    assert (mix.n_real, mix.n_syn, len(mix.dataset)) == (30, 6, 36)
    for i in range(mix.n_real):
        x, y, stats = mix.dataset[i]
        rx, ry, rstats = real_ds[i]
        assert torch.equal(x, rx) and y == ry and torch.equal(stats, rstats)

    X_syn = np.load(mix.export.x_path)
    for j in range(mix.n_syn):
        item = mix.dataset[mix.n_real + j]
        assert len(item) == 3                                       # same tuple as a real row
        want = (X_syn[j].astype(np.float32) - mean) / std
        torch.testing.assert_close(item[0], torch.from_numpy(want).permute(2, 0, 1), rtol=0, atol=0)
        assert item[1] == 1
        torch.testing.assert_close(item[2], torch.from_numpy(np.stack([mean, std])), rtol=0, atol=0)


def test_a_synthetic_dataset_with_its_own_statistics_is_refused(tmp_path, real):
    build, _mean, _std = _builder(real)
    real_ds = build(str(real / "X_train.npy"), str(real / "y_train.npy"))
    export = _load(_export(tmp_path, real), real)

    def refit(xp, yp):                                 # stats fitted on the synthetic rows: wrong
        m, s = compute_global_channel_stats(xp)
        return NpyDatasetV16(xp, yp, normalization="global_zscore", global_mean=m, global_std=s,
                             return_norm_stats=True)
    with pytest.raises(ValueError, match="global_mean"):
        attach_synthetic(real_ds, REAL_Y, REAL_GROUPS, export, refit)
    other, _m, _s = _builder(real, return_norm_stats=False)
    with pytest.raises(ValueError, match="return_norm_stats"):
        attach_synthetic(real_ds, REAL_Y, REAL_GROUPS, export, other)


def test_labels_groups_and_partner_pool_labels(tmp_path, real):
    mix, real_ds, _m, _s = _mixed(tmp_path, real)
    assert mix.y.tolist() == REAL_Y.tolist() + [1] * 6
    assert mix.groups.dtype == np.int32 and mix.groups.tolist() == REAL_GROUPS.tolist() + [-1] * 6
    assert mix.pool_labels.tolist() == REAL_Y.tolist() + [-1] * 6

    build, _m, _s = _builder(real)
    mix_ng = attach_synthetic(real_ds, REAL_Y, None, mix.export, build)
    assert mix_ng.groups is None and mix_ng.pool_labels.tolist() == mix.pool_labels.tolist()

    with pytest.raises(ValueError, match="collide"):
        attach_synthetic(real_ds, REAL_Y, np.where(REAL_Y == 0, -1, REAL_GROUPS), mix.export, build)
    with pytest.raises(ValueError, match="y_real"):
        attach_synthetic(real_ds, REAL_Y[:-1], REAL_GROUPS, mix.export, build)


def test_synthetic_rows_are_never_mixing_partners(tmp_path, real):
    mix, _real_ds, _m, _s = _mixed(tmp_path, real)
    plan = plan_minority_with_synthetic(REAL_Y, mix.n_syn, 3, NAMES, "auto", 2.0, seed=0, y_syn=mix.export.y)
    base = Subset(mix.dataset, plan.indices)
    root, _idx = root_indices(base)
    assert root is mix.dataset                         # the ConcatDataset is the root the labels align with
    assert _probe_in_channels(base) == HWC[2]

    ds = MinorityAugmentedDataset(base, lambda x, r, g: x, lambda x, r, g: x, plan.classes,
                                  labels=mix.pool_labels, groups=mix.groups, mix_mode="pixel",
                                  mix_prob=1.0, seed=0)
    pool_rows = ds._pool[1][0]
    assert sorted(pool_rows.tolist()) == np.flatnonzero(REAL_Y == 1).tolist()     # real DCIS only
    syn_positions = np.flatnonzero(plan.indices >= mix.n_real)
    assert len(syn_positions) == mix.n_syn
    for p in syn_positions:
        for s in range(20):
            assert ds._partner_row(1, int(p), random.Random(s)) < mix.n_real
        x, y, stats = ds[int(p)]                       # a synthetic anchor mixes with a real partner
        assert y == 1 and x.shape == (HWC[2], *HWC[:2]) and stats.shape == (2, HWC[2])


def test_describe_is_the_config_block(tmp_path, real):
    mix, _r, _m, _s = _mixed(tmp_path, real)
    d = json.loads(json.dumps(mix.describe(NAMES)))
    assert d["export_id"] == mix.export.export_id and d["source"] == "cvae"
    assert (d["n_real"], d["n_syn"], d["n_total"]) == (30, 6, 36)
    assert d["syn_class_counts"] == {"healthy": 0, "DCIS": 6, "IDC": 0}
    assert d["cross_patient_groups"] is True and d["provenance"]["n_rows"] == 6
    assert d["source_manifest_sha256"] == _sha256(real / MANIFEST_NAME)


# ============================================================================
# the minority plan
# ============================================================================

def _assert_same_plan(a, b):
    assert a.classes == b.classes and a.factor == b.factor
    assert a.counts_before == b.counts_before and a.counts_after == b.counts_after
    assert (a.indices is None) == (b.indices is None)
    if a.indices is not None:
        assert a.indices.dtype == b.indices.dtype
        np.testing.assert_array_equal(a.indices, b.indices)
    assert a.effective_labels.dtype == b.effective_labels.dtype
    np.testing.assert_array_equal(a.effective_labels, b.effective_labels)


@pytest.mark.parametrize("factor", [1.0, 1.1, 2.0, 2.5, 3.0, 4.75])
@pytest.mark.parametrize("spec", ["auto", "DCIS", "IDC,DCIS", "0,1,2"])
@pytest.mark.parametrize("seed", [0, 42])
def test_without_synthetic_rows_the_plan_is_plan_minority_augmentation(factor, spec, seed):
    y = np.repeat([0, 1, 2], [40, 7, 60])
    ref = plan_minority_augmentation(y, 3, NAMES, spec, factor, seed=seed)
    _assert_same_plan(plan_minority_with_synthetic(y, 0, 3, NAMES, spec, factor, seed=seed), ref)
    _assert_same_plan(plan_minority_with_synthetic(y, 0, 3, NAMES, spec, factor, seed=seed,
                                                   y_syn=np.zeros(0, np.int64)), ref)
    # an all-True mask copies from the same rows with the same draws
    _assert_same_plan(plan_minority_with_synthetic(y, 0, 3, NAMES, spec, factor, seed=seed,
                                                   copy_mask=np.ones(len(y), bool)), ref)


def test_the_no_minority_error_is_the_same():
    y = np.repeat([0, 1], [10, 10])
    with pytest.raises(ValueError, match="--minority_classes"):
        plan_minority_with_synthetic(y, 0, 2, ["a", "b"], "auto", 2.0)


@pytest.mark.parametrize("factor, allowed", [(2.5, [41, 43]), (3.0, [40, 42, 44]), (2.0, [46])])
def test_copy_mask_keeps_n_total_and_copies_only_allowed_rows(factor, allowed):
    y = np.repeat([0, 1, 2], [40, 7, 60])
    mask = np.zeros(len(y), bool)
    mask[allowed] = True
    mask[:40] = True                                   # majority rows are never copied anyway
    ref = plan_minority_augmentation(y, 3, NAMES, "DCIS", factor)
    plan = plan_minority_with_synthetic(y, 0, 3, NAMES, "DCIS", factor, copy_mask=mask)
    assert len(plan.indices) == len(ref.indices)       # N_total unchanged
    assert plan.counts_after == ref.counts_after
    copies = plan.indices[len(y):]
    assert set(copies.tolist()) == set(allowed)
    assert np.all(plan.indices[:len(y)] == np.arange(len(y)))


def test_synthetic_rows_follow_the_real_rows_once_and_are_never_copied():
    y = np.repeat([0, 1, 2], [12, 4, 14])
    y_syn = np.ones(6, np.int64)
    plan = plan_minority_with_synthetic(y, 6, 3, NAMES, "auto", 2.5, seed=0, y_syn=y_syn)
    assert plan.classes == [1]           # resolved on REAL counts: 4 < 0.5 x 12 (with synthetic: 10, not)
    real_part = plan_minority_augmentation(y, 3, NAMES, "auto", 2.5, seed=0).indices
    np.testing.assert_array_equal(plan.indices, np.concatenate([real_part, np.arange(30, 36)]))
    assert plan.counts_before == [12, 4, 14] and plan.counts_after == [12, 10 + 6, 14]
    assert np.bincount(plan.effective_labels).tolist() == plan.counts_after

    flat = plan_minority_with_synthetic(y, 6, 3, NAMES, "auto", 1.0, y_syn=y_syn)
    assert flat.indices is None and flat.counts_after == [12, 10, 14]
    assert len(flat.effective_labels) == 36


@pytest.mark.parametrize("kwargs, match", [
    ({"n_syn": 6}, "y_syn is needed"),
    ({"n_syn": 6, "y_syn": np.ones(5, np.int64)}, "y_syn has 5"),
    ({"n_syn": 0, "copy_mask": np.ones(29, bool)}, "boolean array"),
    ({"n_syn": 0, "copy_mask": np.ones(30, np.int64)}, "boolean array"),
    ({"n_syn": 0, "copy_mask": np.repeat([True, False, True], [12, 4, 14])}, "allows no row"),
])
def test_planner_refuses_inconsistent_inputs(kwargs, match):
    y = np.repeat([0, 1, 2], [12, 4, 14])
    with pytest.raises(ValueError, match=match):
        plan_minority_with_synthetic(y, num_classes=3, class_names=NAMES, spec="auto", factor=2.0, **kwargs)


# ============================================================================
# an empty export
# ============================================================================

def test_an_empty_export_is_valid_and_changes_nothing(tmp_path, real):
    exp = _export(tmp_path, real, n=0)
    prov = json.loads((exp / PROVENANCE_NAME).read_text())
    assert prov["n_rows"] == 0 and prov["class_counts"] == {"healthy": 0, "DCIS": 0, "IDC": 0}
    assert np.load(exp / X_SYN_NAME).shape == (0, *HWC)

    # "ram" is what --dataset_storage auto picks for a file this small; a 0-row read differs from mmap's
    mix, real_ds, _m, _s = _mixed(tmp_path / "m", real, n=0, storage_mode="ram")
    assert (mix.n_syn, len(mix.dataset)) == (0, len(real_ds))
    for i in range(len(real_ds)):
        a, b = mix.dataset[i], real_ds[i]
        assert torch.equal(a[0], b[0]) and a[1] == b[1] and torch.equal(a[2], b[2])
    assert mix.pool_labels.tolist() == REAL_Y.tolist() and mix.groups.tolist() == REAL_GROUPS.tolist()
    _assert_same_plan(plan_minority_with_synthetic(REAL_Y, mix.n_syn, 3, NAMES, "auto", 2.0, y_syn=mix.export.y),
                      plan_minority_augmentation(REAL_Y, 3, NAMES, "auto", 2.0))


# ============================================================================
# train.py --synthetic_dir: flags, run name, class weights, copy mask
# ============================================================================

HSI = "data/hsi_v8-80_10_10_importance-new/hsi"


def test_the_flags_are_off_by_default_and_nothing_is_checked_at_parse_time():
    args = resolve(["--data_dir", HSI])[0].args
    assert (args.synthetic_dir, args.synthetic_build_check, args.minority_copy_mask) == (None, "on", None)
    # sweeps parse their argv on machines that do not hold the export (tests/test_sweeps.py)
    args = resolve(["--data_dir", HSI, "--synthetic_dir", "no/such/export", "--minority_copy_mask",
                    "no/such/mask.npy", "--synthetic_build_check", "off"])[0].args
    assert (args.synthetic_dir, args.minority_copy_mask) == ("no/such/export", "no/such/mask.npy")
    assert args.synthetic_build_check == "off"


def test_the_run_name_gains_syn_only_with_synthetic_dir():
    ts = "20261027_000000"
    off = resolve(["--data_dir", HSI])[0].args
    on = resolve(["--data_dir", HSI, "--synthetic_dir", "exports/cvae"])[0].args
    assert build_run_name(on, ts, modality="x") == build_run_name(off, ts, modality="x") + "_syn"
    both = resolve(["--data_dir", HSI, "--minority_augment", "on", "--synthetic_dir", "exports/cvae",
                    "--run_tag", "t1"])[0].args
    assert build_run_name(both, ts, modality="x").endswith("_minaug_syn_t1")


_Y = np.array([0, 0, 0, 1, 2, 2])
_PLAN = SimpleNamespace(effective_labels=np.array([0, 0, 0, 1, 2, 2, 1, 1]))
_MIX = SimpleNamespace(y=np.array([0, 0, 0, 1, 2, 2, 1]))


@pytest.mark.parametrize("minority, mix, mode, want, name", [
    (None, None, "effective", _Y, "y_train"),                          # the default path
    (None, _MIX, "effective", _MIX.y, "y_train_and_synthetic"),
    (None, _MIX, "original", _MIX.y, "y_train_and_synthetic"),         # ignored with --minority_augment off
    (_PLAN, None, "effective", _PLAN.effective_labels, "minority_effective_labels"),
    (_PLAN, _MIX, "effective", _PLAN.effective_labels, "minority_effective_labels"),
    (_PLAN, None, "original", _Y, "y_train"),
    (_PLAN, _MIX, "original", _Y, "y_train"),                          # y_train.npy: real rows only
])
def test_class_weight_labels(minority, mix, mode, want, name):
    labels, got = _class_weight_labels(_Y, minority, mix, mode)
    assert labels is want and got == name


def test_a_copy_mask_is_one_bool_per_real_train_row(tmp_path):
    ok = np.zeros(5, bool)
    ok[1] = True
    np.save(tmp_path / "ok.npy", ok)
    assert _load_copy_mask(str(tmp_path / "ok.npy"), 5).tolist() == ok.tolist()
    for name, arr in (("short", np.ones(4, bool)), ("int", np.ones(5, np.int64)), ("2d", np.ones((5, 1), bool))):
        np.save(tmp_path / f"{name}.npy", arr)
        with pytest.raises(SystemExit, match="boolean array over the 5 real train rows"):
            _load_copy_mask(str(tmp_path / f"{name}.npy"), 5)
    (tmp_path / "junk.npy").write_text("not an array")
    for name in ("junk.npy", "missing.npy"):
        with pytest.raises(SystemExit, match="cannot read"):
            _load_copy_mask(str(tmp_path / name), 5)


# ============================================================================
# train.py --synthetic_dir end to end
# ============================================================================

TINY = ["--epochs", "2", "--batch_size", "16", "--trm_dim", "16", "--trm_n_latent", "1",
        "--trm_n_improve", "1", "--trm_deep_supervision_steps", "1", "--compile", "off",
        "--loader_mode", "safe", "--drift_check_patches", "64", "--min_logit_std", "1e-4"]
TIMING = {"GPU_memory_MB", "artifact_time_s", "data_wait_s", "epoch_time", "s_per_step",
          "train_time_s", "val_time_s"}
MINAUG = ("--minority_augment", "on", "--minority_oversample_factor", "3")
E2E_HWC = (8, 8, 8)
E2E_COUNTS = {"train": [40, 8, 56], "val": [16, 16, 16], "test": [16, 16, 16]}
N_REAL = sum(E2E_COUNTS["train"])


def _e2e_syn_x(n: int) -> np.ndarray:
    return (np.random.default_rng(7).random((n, *E2E_HWC)) * 0.5 + 0.15).astype(np.float32)   # DCIS-like


@pytest.fixture(scope="module")
def tiny_bound_hsi(tmp_path_factory) -> SimpleNamespace:
    """`tests/test_minority_augmentation.py`'s tiny imbalanced split, with the
    sha256 manifest that binds an export to it; a 6-row DCIS export; an empty one."""
    root = tmp_path_factory.mktemp("tiny_synthetic")
    data = root / "hsi"
    data.mkdir()
    rng = np.random.default_rng(0)
    files = ["wavelengths.npy"]
    for split, per in E2E_COUNTS.items():
        y = np.repeat(np.arange(3), per).astype(np.int64)
        x = rng.random((len(y), *E2E_HWC), dtype=np.float32) * 0.5 + (y[:, None, None, None] * 0.15)
        np.save(data / f"X_{split}.npy", x.astype(np.float32))
        np.save(data / f"y_{split}.npy", y)
        np.save(data / f"groups_{split}.npy", (np.arange(len(y)) % 4 + 10 * y).astype(np.int32))
        files += [f"X_{split}.npy", f"y_{split}.npy", f"groups_{split}.npy"]
    np.save(data / "wavelengths.npy", np.linspace(450, 900, E2E_HWC[2]).astype(np.float32))
    (data / "class_names.json").write_text(json.dumps(NAMES))
    write_dataset_manifest(str(data), files, sha256=True)
    syn, empty = root / "export_dcis", root / "export_empty"
    write_synthetic_export(str(syn), _e2e_syn_x(6), np.ones(6, np.int64), "cvae", str(data))
    write_synthetic_export(str(empty), _e2e_syn_x(0), np.ones(0, np.int64), "cvae", str(data))
    return SimpleNamespace(data=data, syn=syn, empty=empty)


def _train_py(argv, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "train.py")] + list(argv), cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def train_runs(tiny_bound_hsi, tmp_path_factory):
    """`train_runs(*flags)`: the run directory of `train.py --profile paper_recipe`
    + TINY + flags on the tiny split, trained once per module per set of flags."""
    done = {}

    def run(*flags):
        if flags not in done:
            cwd = tmp_path_factory.mktemp("run")
            proc = _train_py(["--data_dir", str(tiny_bound_hsi.data), "--profile", "paper_recipe"] + TINY
                             + list(flags), cwd)
            assert proc.returncode == 0, f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
            (done[flags],) = list((cwd / "experiments").iterdir())
        return done[flags]
    return run


def _json(run_dir: Path, name: str):
    return json.loads((run_dir / name).read_text())


def _history(run_dir: Path):
    return [{k: v for k, v in row.items() if k not in TIMING} for row in _json(run_dir, "history.json")]


def _gates(run_dir: Path) -> dict:
    """gates.json minus G5's checkpoint path, which contains the run directory."""
    gates = _json(run_dir, "gates.json")
    if isinstance(gates.get("G5"), dict):
        gates["G5"].pop("checkpoint", None)
    return gates


@pytest.mark.parametrize("arm", [(), MINAUG], ids=["plain", "minaug"])
def test_an_empty_export_trains_bit_identically(arm, tiny_bound_hsi, train_runs):
    base = train_runs(*arm)
    empty = train_runs(*arm, "--synthetic_dir", str(tiny_bound_hsi.empty))
    assert empty.name[16:] == base.name[16:] + "_syn"          # past the timestamp

    h_base, h_empty = _history(base), _history(empty)
    assert len(h_base) == len(h_empty) == 2
    for epoch, (a, b) in enumerate(zip(h_base, h_empty), 1):
        diff = {k: (a.get(k), b.get(k)) for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        assert not diff, f"epoch {epoch}: {diff}"
    for name in ("test_predictions.npz", "predictions/predictions_epoch01.npz"):
        p_base, p_empty = np.load(base / name), np.load(empty / name)
        assert sorted(p_base.files) == sorted(p_empty.files)
        for key in p_base.files:
            np.testing.assert_array_equal(p_base[key], p_empty[key], err_msg=f"{name}: {key}")
    assert _gates(base) == _gates(empty)
    assert _json(base, "split_drift_report.json") == _json(empty, "split_drift_report.json")

    c_base, c_empty = _json(base, "config.json"), _json(empty, "config.json")
    assert (c_empty["synthetic"]["n_real"], c_empty["synthetic"]["n_syn"]) == (N_REAL, 0)
    assert c_base.get("minority_augmentation") == c_empty.get("minority_augmentation")
    assert c_base["train_subsample"] == c_empty["train_subsample"]
    assert "synthetic" not in c_base and not (base / "synthetic_mix_report.json").exists()


def test_train_py_trains_on_real_and_synthetic_rows(tiny_bound_hsi, train_runs, tmp_path_factory):
    """Acceptance 2, 3, 4 and 7, on one run with 6 synthetic DCIS rows and a copy mask."""
    mask = np.ones(N_REAL, bool)
    mask[np.flatnonzero(np.repeat(np.arange(3), E2E_COUNTS["train"]) == 1)[:4]] = False   # 4 of 8 DCIS rows
    mask_path = tmp_path_factory.mktemp("mask") / "copy_mask.npy"
    np.save(mask_path, mask)
    base = train_runs(*MINAUG)
    run = train_runs(*MINAUG, "--synthetic_dir", str(tiny_bound_hsi.syn), "--minority_copy_mask", str(mask_path))
    assert run.name.endswith("_minaug_syn")
    cfg = _json(run, "config.json")

    # 7: config.json["synthetic"], and the report next to it
    syn = cfg["synthetic"]
    prov = json.loads((tiny_bound_hsi.syn / PROVENANCE_NAME).read_text())
    assert syn["export_id"] == prov["export_id"] and syn["source"] == "cvae" and syn["provenance"] == prov
    assert (syn["n_real"], syn["n_syn"], syn["n_total"]) == (N_REAL, 6, N_REAL + 6)
    assert syn["syn_class_counts"] == {"healthy": 0, "DCIS": 6, "IDC": 0}
    assert syn["source_manifest_sha256"] == _sha256(tiny_bound_hsi.data / MANIFEST_NAME)
    assert syn["in_reconstruction_loss"] is True                 # paper_recipe: recon_mode latent
    assert syn["class_weight_labels"] == "minority_effective_labels"
    assert syn["class_counts"] == {"real": {"healthy": 40, "DCIS": 8, "IDC": 56},
                                   "synthetic": {"healthy": 0, "DCIS": 6, "IDC": 0},
                                   "train_set": {"healthy": 40, "DCIS": 3 * 8 + 6, "IDC": 56}}
    assert _json(run, "synthetic_mix_report.json") == syn

    # 3: only real rows are copied (a copied synthetic row would give DCIS round(3 x 14) = 42), the
    # mask keeps the number of copies, and the partners are the 8 real DCIS rows of 4 real patients
    m = cfg["minority_augmentation"]
    assert m["counts_before"] == {"healthy": 40, "DCIS": 8, "IDC": 56}
    assert m["counts_after"] == {"healthy": 40, "DCIS": 30, "IDC": 56}
    assert m["partner_pool_rows"] == {"1": 8} and m["partner_pool_patients"] == {"1": 4}
    assert m["copy_mask"] == {"path": str(mask_path), "copyable_rows": N_REAL - 4}
    assert cfg["train_subsample"]["total"] == 40 + 30 + 56

    # 2: the statistics are still fitted on the real X_train alone; gate G8 normalizes every split
    # with them, so its report is the one of the run without synthetic rows, to the bit
    assert (cfg["cli_args"]["normalization"], cfg["cli_args"]["global_stats"]) == ("global_zscore", "train_fit")
    assert _json(run, "split_drift_report.json") == _json(base, "split_drift_report.json")

    # 4: validation and test are the real splits: the same rows, in the same order
    assert cfg["val_subsample"] == _json(base, "config.json")["val_subsample"]
    for a, b in zip(_history(run), _history(base)):
        assert a["per_class_support"] == b["per_class_support"] == E2E_COUNTS["val"]
    for name in ("test_predictions.npz", "predictions/predictions_epoch01.npz"):
        np.testing.assert_array_equal(np.load(run / name)["true_label"], np.load(base / name)["true_label"])


@pytest.mark.parametrize("case, match", [
    ("build_mismatch", r"--synthetic_dir: the export was made from a different build.*"
                       r"--synthetic_build_check off overrides"),
    ("non_finite", r"--synthetic_dir: X_syn.npy: 1 row\(s\) hold a NaN/inf.*first: row 2"),
    ("sampler", "--sampler balanced cannot be combined with --synthetic_dir"),
    ("mask_without_minority", "--minority_copy_mask .* pass --minority_augment on"),
    ("mask_wrong_shape", rf"--minority_copy_mask: .* boolean array over the {N_REAL} real train rows"),
])
def test_train_py_refuses_before_it_reads_the_real_build(case, match, tiny_bound_hsi, tmp_path):
    """Acceptance 5 and 6, and the flag combinations: a non-zero exit with the
    reason, before the sha256 pass over the real build and before any training."""
    argv = ["--data_dir", str(tiny_bound_hsi.data)] + TINY
    if case in ("build_mismatch", "non_finite"):
        exp = tmp_path / "export"
        write_synthetic_export(str(exp), _e2e_syn_x(4), np.ones(4, np.int64), "cvae", str(tiny_bound_hsi.data))
        if case == "build_mismatch":                       # bound to some other build's manifest
            _edit_provenance(exp, source_manifest_sha256="0" * 64)
        else:
            X = np.load(exp / X_SYN_NAME)
            X[2, 0, 0, 0] = np.float16(np.nan)
            np.save(exp / X_SYN_NAME, X)
            _reseal(exp)
        argv += ["--synthetic_dir", str(exp)]
    elif case == "sampler":
        argv += ["--synthetic_dir", str(tiny_bound_hsi.syn), "--sampler", "balanced"]
    else:
        np.save(tmp_path / "mask.npy", np.ones(N_REAL - (case == "mask_wrong_shape"), bool))
        argv += ["--minority_copy_mask", str(tmp_path / "mask.npy")]
        argv += list(MINAUG) if case == "mask_wrong_shape" else []
    proc = _train_py(argv, tmp_path)
    assert proc.returncode != 0
    assert re.search(match, proc.stderr), proc.stderr[-2000:]
    assert "[dataset-manifest]" not in proc.stdout
    assert not list(tmp_path.glob("experiments/*/history.json"))
