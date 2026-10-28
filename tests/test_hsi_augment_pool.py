# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_pool.py
==============================
The `pool` stage (#16, Plan §Phase 4 items 3-4): the generator pool and the
real-vs-real references, on the tiny fixture.

  * quotas and patient balance: every tissue DCIS row; `rows_per_class` rows
    of healthy and of IDC, shared over patients by water-filling
    (`balanced_allocation`, also checked against a one-row-at-a-time oracle
    on random counts); tissue rows only, in train-row order;
  * the log transform round-trips: the codec's statistics are float64 numpy
    over EVERY train tissue row, clip_hi is 1.1 x the exact p99.99, X_pool is
    the codec's float16 encoding of the train rows, and decoding it gives them
    back within the float16 bound;
  * the files follow pool_io's contract (`load_pool`, `check_against_train`),
    X_pool.npy written last;
  * the references: the PCA equals an SVD of its fit rows, the nearest
    neighbours equal a brute-force search per fold and class (cross-patient,
    cross-capture), the fixture's planted duplicate is flagged and left out of
    the quantiles, the feature summaries equal numpy;
  * a LOPO fold is a mask over the one pool;
  * determinism, the size cap, the tau sign-off, upstream checks, the split
    access log, the registration and the CLI;
  * the real build's pool fits in 3.5 GB (from its known counts), and an
    optional real-data smoke test (HSI_AUGMENT_REAL_DATA).

CPU only, a few seconds in total.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hsi_augment import cli, splits, tissue
from hsi_augment import pool as P
from hsi_augment import provenance as pv
from hsi_augment.codec import FLOAT16_MAX, LOG_EPS
from hsi_augment.io import read_access_log
from hsi_augment.pool_io import CODEC_NAME, POOL_FILES, check_against_train, load_pool
from hsi_augment.registry import load_builtin_stages
from hsi_augment.stage import DONE_NAME, RunOptions, StageError, StageNotReady, run_stage
from hsi_augment.stages.leakage import LeakageConfig
from hsi_augment.stages.pool import PoolConfig
from hsi_augment.stages.tissue_mask import TissueMaskConfig
from tests.hsi_augment_fixtures import make_tiny_build

STAGE_DIR = Path("pool")
TISSUE_DIR = Path("prep") / "tissue"
LEAKAGE_DIR = Path("prep") / "leakage"
QUOTA = 30                 # healthy and IDC have ~80 tissue rows each on the fixture: the quota binds
DCIS = 1
POOL_OUTPUTS = {*POOL_FILES.values(), CODEC_NAME, P.REFERENCE_NAME, P.POOL_META_NAME}


def _run(build, out, cfg=None, **opts):
    reg = load_builtin_stages()
    options = RunOptions(data_dir=str(build.data_dir), out=str(out), **opts)
    run_stage(reg.get("tissue-mask"), TissueMaskConfig(), options, reg)
    run_stage(reg.get("leakage"), LeakageConfig(), options, reg)
    return run_stage(reg.get("pool"), cfg or PoolConfig(rows_per_class=QUOTA), options, reg)


def _pool_only(build, out, cfg, upstream_out):
    """Run only the pool stage, on copies of an earlier run's upstream directories."""
    for d in (TISSUE_DIR, LEAKAGE_DIR):
        shutil.copytree(upstream_out / d, out / d)
    reg = load_builtin_stages()
    return run_stage(reg.get("pool"), cfg, RunOptions(data_dir=str(build.data_dir), out=str(out)), reg)


def _train(build):
    d = build.data_dir
    return SimpleNamespace(X=np.load(d / "X_train.npy"), y=np.load(d / "y_train.npy").astype(np.int64),
                           groups=np.load(d / "groups_train.npy").astype(np.int64),
                           captures=np.load(d / "captures_train.npy").astype(np.int64))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def ran(tiny, tmp_path_factory):
    """tissue-mask, leakage, then pool with rows_per_class=QUOTA, once, shared by the read-only tests."""
    out = tmp_path_factory.mktemp("aug")
    res = _run(tiny, out)
    d = out / STAGE_DIR
    return SimpleNamespace(out=out, dir=d, result=res, pool=load_pool(d, mmap=False), ref=P.load_reference(d),
                           meta=json.loads((d / P.POOL_META_NAME).read_text()),
                           mask=np.load(out / TISSUE_DIR / tissue.MASK_NAME),
                           folds=splits.load_folds(out / LEAKAGE_DIR), train=_train(tiny))


def _greedy(counts, quota):
    """Oracle: hand out rows one at a time to the patient with the fewest
    taken among those with rows left (ties: most rows, then lower id)."""
    taken = {p: 0 for p in counts}
    for _ in range(min(quota, sum(counts.values()))):
        p = min((p for p in counts if taken[p] < counts[p]), key=lambda p: (taken[p], -counts[p], p))
        taken[p] += 1
    return taken


# ============================================================================
# quotas and patient balance
# ============================================================================

def test_balanced_allocation_water_fills():
    assert P.balanced_allocation({1: 5, 2: 100, 3: 100}, 45) == ({1: 5, 2: 20, 3: 20}, 20)
    assert P.balanced_allocation({1: 5, 2: 100, 3: 100}, 46) == ({1: 5, 2: 21, 3: 20}, 20)   # tie: lower id
    assert P.balanced_allocation({1: 3, 2: 9, 3: 4}, 7) == ({1: 2, 2: 3, 3: 2}, 2)            # extra: most rows
    assert P.balanced_allocation({1: 0, 2: 10}, 4) == ({1: 0, 2: 4}, 4)
    for quota in (None, 12, 100):                                    # the quota covers every row
        assert P.balanced_allocation({1: 5, 2: 7}, quota) == ({1: 5, 2: 7}, None)
    with pytest.raises(ValueError, match=">= 0"):
        P.balanced_allocation({1: -1}, 3)

    rng = np.random.default_rng(0)
    for _ in range(300):
        counts = {int(p): int(rng.integers(0, 40)) for p in rng.choice(1000, size=rng.integers(1, 9), replace=False)}
        total = sum(counts.values())
        quota = int(rng.integers(0, total + 5))
        alloc, level = P.balanced_allocation(counts, quota)
        assert alloc == _greedy(counts, quota)
        assert sum(alloc.values()) == min(quota, total)
        assert all(0 <= alloc[p] <= counts[p] for p in counts)
        if level is None:
            assert alloc == counts
            continue
        capped = [p for p in counts if alloc[p] < counts[p]]
        assert capped and all(alloc[p] in (level, level + 1) for p in capped)
        assert all(counts[p] <= level + 1 for p in counts if alloc[p] == counts[p])
        assert max(alloc.values()) - min(alloc[p] for p in capped) <= 1          # max-min fair


def test_select_rows_draws_from_one_stream_per_class_and_patient():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=600)
    groups = rng.integers(1, 7, size=600)
    mask = rng.random(600) < 0.8
    quotas = {0: 40, 1: None, 2: 25}
    seeds = {(c, p): 100 * c + p for c in range(3) for p in range(1, 7)}

    def rng_for(c, p):
        return np.random.default_rng(seeds[(c, p)])

    rows, table = P.select_rows(y, groups, mask, quotas, rng_for)
    assert rows.dtype == np.int64 and np.all(np.diff(rows) > 0) and mask[rows].all()
    np.testing.assert_array_equal(rows[y[rows] == 1], np.flatnonzero(mask & (y == 1)))   # quota None: all
    for c in (0, 2):
        taken = Counter(groups[rows][y[rows] == c].tolist())
        avail = Counter(groups[mask & (y == c)].tolist())
        assert dict(taken) == {p: v for p, v in _greedy(dict(avail), quotas[c]).items() if v}
        assert table[c]["pool_rows"] == quotas[c] and table[c]["tissue_rows"] == sum(avail.values())
        assert {p: v["pool_rows"] for p, v in table[c]["patients"].items()} == _greedy(dict(avail), quotas[c])
    assert table[1]["level"] is None and table[0]["level"] is not None
    again, _ = P.select_rows(y, groups, mask, quotas, rng_for)
    np.testing.assert_array_equal(again, rows)
    seeds[(0, 3)] += 1                               # another stream for ONE (class, patient)
    other, _ = P.select_rows(y, groups, mask, quotas, rng_for)
    moved = np.setxor1d(rows, other)
    assert moved.size and set(groups[moved].tolist()) == {3} and set(y[moved].tolist()) == {0}


def test_pool_holds_every_tissue_dcis_row_and_patient_balanced_quotas(ran):
    pool, tr, mask = ran.pool, ran.train, ran.mask
    assert np.all(np.diff(pool.row) > 0) and mask[pool.row].all()            # tissue only, train-row order
    np.testing.assert_array_equal(pool.row[pool.y == DCIS], np.flatnonzero(mask & (tr.y == DCIS)))
    for c, name in ((0, "healthy"), (2, "IDC")):
        avail = Counter(tr.groups[mask & (tr.y == c)].tolist())
        taken = Counter(pool.patient[pool.y == c].tolist())
        assert sum(avail.values()) > QUOTA and sum(taken.values()) == QUOTA
        assert dict(taken) == {p: v for p, v in _greedy(dict(avail), QUOTA).items() if v}
        capped = [p for p in avail if taken[p] < avail[p]]
        assert capped and max(taken.values()) - min(taken[p] for p in capped) <= 1
        cls = ran.meta["classes"][name]
        assert (cls["quota"], cls["pool_rows"], cls["tissue_rows"], cls["quota_binds"]) == \
            (QUOTA, QUOTA, sum(avail.values()), True)
        assert {int(p): v["pool_rows"] for p, v in cls["patients"].items()} == dict(taken)
    d = ran.meta["classes"]["DCIS"]
    assert d["quota"] is None and d["quota_binds"] is False and d["pool_rows"] == d["tissue_rows"]
    assert ran.result.done["summary"]["rows_per_class"] == {"healthy": QUOTA, "DCIS": d["pool_rows"], "IDC": QUOTA}


def test_pool_follows_the_pool_io_contract(ran, tiny):
    pool, tr = ran.pool, ran.train
    assert check_against_train(pool, tr.y, tr.groups, tr.captures) == []
    assert pool.X.dtype == np.float16 and pool.X.shape[1:] == (tiny.patch, tiny.patch, tiny.n_bands)
    assert pool.codec.n_bands == tiny.n_bands
    np.testing.assert_array_equal(pool.capture, tr.captures[pool.row])
    done = json.loads((ran.dir / DONE_NAME).read_text())
    assert set(done["outputs"]) == POOL_OUTPUTS
    assert ran.meta["x_bytes"] == pool.X.nbytes and ran.meta["n_rows"] == pool.n
    assert ran.meta["x_shape"] == list(pool.X.shape) and ran.meta["x_layout"] == "NHWC"
    mtimes = {n: (ran.dir / n).stat().st_mtime_ns for n in POOL_OUTPUTS}
    assert mtimes[POOL_FILES["X"]] == max(mtimes.values())                 # X_pool.npy last


# ============================================================================
# the log transform
# ============================================================================

def test_log_space_statistics_are_float64_over_every_train_tissue_row(ran, tiny):
    tr, mask, codec = ran.train, ran.mask, ran.pool.codec
    x = tr.X[mask].reshape(-1, tiny.n_bands)
    lg = np.log(np.maximum(x.astype(np.float64), 0.0) + LOG_EPS)
    np.testing.assert_allclose(codec.mu, lg.mean(axis=0), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(codec.sigma, lg.std(axis=0), rtol=1e-10)
    assert codec.eps == LOG_EPS
    # every tissue row, not only the pool's: the pool-only statistics differ
    pool_lg = np.log(tr.X[ran.pool.row].reshape(-1, tiny.n_bands).astype(np.float64) + LOG_EPS)
    assert not np.allclose(codec.mu, pool_lg.mean(axis=0), rtol=1e-6)
    p = np.quantile(x, 0.9999, axis=0, method="inverted_cdf").astype(np.float64)
    np.testing.assert_array_equal(codec.clip_hi, np.minimum(1.1 * p, np.nextafter(FLOAT16_MAX, 0.0)))
    meta = ran.meta
    assert meta["log_space"]["n_rows"] == meta["tissue"]["n_tissue_rows"] == int(mask.sum())
    assert meta["clip"]["p99_99"] == p.tolist() and meta["clip"]["clip_hi"] == codec.clip_hi.tolist()
    assert json.loads((ran.dir / CODEC_NAME).read_text()) == codec.to_dict()


def test_log_transform_round_trips(ran):
    pool, codec = ran.pool, ran.pool.codec
    raw = ran.train.X[pool.row]
    # X_pool IS the codec's float16 encoding of the train rows
    np.testing.assert_array_equal(pool.X.view(np.uint16), codec.encode_np(raw).astype(np.float16).view(np.uint16))
    # the algebra: decode(encode(x)) = x in float64, and in the codec's float32 without float16 storage
    x = raw.astype(np.float64)
    mu, sigma = codec.mu, codec.sigma
    y64 = (np.log(x + LOG_EPS) - mu) / sigma
    np.testing.assert_allclose(np.exp(y64 * sigma + mu) - LOG_EPS, x, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(codec.decode_np(codec.encode_np(raw), dtype=np.float32), x, rtol=2e-5, atol=1e-6)
    # through the stored float16 log-z: within the float16 bound, value by value
    y = pool.X.astype(np.float64)
    dec = codec.decode_np(pool.X).astype(np.float64)
    rel = np.abs(dec - x) / (x + LOG_EPS)
    bound = (np.abs(y) * sigma + 1.0) * 2.0 ** -11 * 1.01 + 1e-5
    assert (rel <= bound).all() and rel.max() < 2e-3
    rt = ran.meta["round_trip"]
    assert rt["max_rel_err"] == pytest.approx(rel.max(), rel=1e-4)
    assert rt["n_values"] == raw.size and rt["n_above_clip_hi"] == 0 and rt["n_negative"] == 0
    assert np.isfinite(pool.X).all()


# ============================================================================
# the references
# ============================================================================

def _proj(ran):
    ref = ran.ref
    flat = ran.pool.X.reshape(ran.pool.n, -1).astype(np.float64)
    return (flat - ref["pca_mean"]) @ ref["pca_components"].T


def test_reference_schema(ran, tiny, tmp_path):
    ref = ran.ref
    assert int(ref["schema_version"]) == P.REFERENCE_SCHEMA_VERSION
    assert ref["class_names"].tolist() == list(tiny.class_names)
    assert ref["quantile_levels"].tolist() == list(PoolConfig().quantiles)
    assert ref["fold_dcis_patient"].tolist() == list(tiny.dcis_patients["train"]) == ran.folds["dcis_patients"]
    F, K, Q, C2 = len(ran.folds["folds"]), 3, len(PoolConfig().quantiles), 2 * tiny.n_bands
    D = tiny.patch * tiny.patch * tiny.n_bands
    shapes = {"pca_mean": (D,), "pca_components": (64, D), "feats": (ran.pool.n, C2), "nn_quantiles": (F, K, Q),
              "nn_quantiles_pooled": (K, Q), "feat_n": (F, K, 2), "feat_mean": (F, K, 2, C2),
              "feat_quantiles": (F, K, 2, Q, C2), "nn_n_query": (F, K)}
    assert {k: ref[k].shape for k in shapes} == shapes
    assert ref["nn_duplicate"].dtype == np.bool_ and ref["nn_dist"].dtype == np.float64
    bad = tmp_path / "bad_reference.npz"
    np.savez(bad, schema_version=np.array(99))
    with pytest.raises(ValueError, match="schema_version"):
        P.load_reference(bad)


def test_pca_equals_an_svd_of_its_fit_rows(ran):
    ref = ran.ref
    fit = ref["pca_fit_index"]
    assert fit.tolist() == list(range(ran.pool.n))                    # fewer rows than pca_fit_rows: all
    flat = ran.pool.X[fit].reshape(fit.size, -1).astype(np.float64)
    np.testing.assert_allclose(ref["pca_mean"], flat.mean(axis=0), rtol=1e-12, atol=1e-13)
    _, s, vt = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    ev = s ** 2 / (fit.size - 1)
    comps = ref["pca_components"]
    k = comps.shape[0]
    np.testing.assert_allclose(ref["pca_explained_variance"], ev[:k], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(comps @ comps.T, np.eye(k), atol=1e-10)
    for i in range(8):                                                # well-separated leading components
        if (ev[i] - ev[i + 1]) > 1e-3 * ev[i]:
            assert abs(float(comps[i] @ vt[i])) == pytest.approx(1.0, abs=1e-8)
    assert np.all(np.diff(ref["pca_explained_variance"]) <= 1e-12)
    assert ref["pca_explained_variance_ratio"].sum() == pytest.approx(ev[:k].sum() / ev.sum(), rel=1e-9)
    big = np.argmax(np.abs(comps), axis=1)
    assert (comps[np.arange(k), big] > 0).all()                       # deterministic signs


def test_nearest_neighbours_equal_a_brute_force_search(ran):
    from scipy.spatial.distance import cdist
    pool, ref, proj = ran.pool, ran.ref, _proj(ran)
    levels = ref["quantile_levels"]
    for fi, fold in enumerate(ran.folds["folds"]):
        train_m, held_m = splits.fold_masks(pool.patient, fold)
        for c in range(3):
            held = np.flatnonzero(held_m & (pool.y == c))
            refs = np.flatnonzero(train_m & (pool.y == c))
            sel = (ref["nn_fold"] == fi) & (ref["nn_class"] == c)
            q, match, dist, dup = (ref[k][sel] for k in ("nn_query", "nn_match", "nn_dist", "nn_duplicate"))
            np.testing.assert_array_equal(q, held)                    # no subsample below nn_max_queries
            assert (ref["nn_n_heldout"][fi, c], ref["nn_n_query"][fi, c], ref["nn_n_reference"][fi, c]) == \
                (held.size, held.size, refs.size)
            assert held.size and refs.size
            np.testing.assert_allclose(dist, cdist(proj[q], proj[refs]).min(axis=1), rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(dist, np.linalg.norm(proj[q] - proj[match], axis=1), rtol=1e-9, atol=1e-9)
            assert np.isin(match, refs).all() and (pool.y[match] == c).all()
            assert not np.isin(pool.patient[match], fold["heldout_patients"]).any()   # cross-patient
            assert (pool.capture[match] != pool.capture[q]).all()                       # cross-capture
            np.testing.assert_allclose(ref["nn_quantiles"][fi, c], np.quantile(dist[~dup], levels), rtol=1e-12)
    for c in range(3):
        sel = (ref["nn_class"] == c) & ~ref["nn_duplicate"]
        np.testing.assert_allclose(ref["nn_quantiles_pooled"][c], np.quantile(ref["nn_dist"][sel], levels))


def test_planted_duplicate_is_flagged_and_left_out(ran, tiny):
    pool, ref = ran.pool, ran.ref
    assert {tiny.duplicate_row, tiny.duplicate_of} <= set(pool.row.tolist())       # both DCIS tissue rows
    dup = ref["nn_duplicate"]
    pairs = {(int(pool.row[q]), int(pool.row[m])) for q, m in zip(ref["nn_query"][dup], ref["nn_match"][dup])}
    assert pairs == {(tiny.duplicate_of, tiny.duplicate_row), (tiny.duplicate_row, tiny.duplicate_of)}
    assert (ref["nn_dist"][dup] == 0).all() and (ref["nn_dist"][~dup] > 0).all()
    assert (ref["nn_quantiles"][:, DCIS, 0] > 0).all() and ref["nn_quantiles_pooled"][DCIS, 0] > 0
    nn = ran.meta["reference"]["nn"]
    assert nn["n_duplicates"] == 2 and {(d["query_row"], d["match_row"]) for d in nn["duplicates"]} == pairs
    assert ran.result.done["summary"]["nn_duplicates"] == 2


def test_features_and_their_fold_summaries_equal_numpy(ran):
    pool, ref = ran.pool, ran.ref
    feats = ref["feats"]
    np.testing.assert_array_equal(feats, pv.patch_feats(ran.train.X[pool.row]))       # = index_train patch_feats
    levels = ref["quantile_levels"]
    for fi, fold in enumerate(ran.folds["folds"]):
        masks = splits.fold_masks(pool.patient, fold)[::-1]             # side 0 = held out, 1 = train side
        for c in range(3):
            for side, m in enumerate(masks):
                f = feats[m & (pool.y == c)].astype(np.float64)
                assert ref["feat_n"][fi, c, side] == f.shape[0] > 0
                np.testing.assert_allclose(ref["feat_mean"][fi, c, side], f.mean(axis=0), rtol=1e-12)
                np.testing.assert_allclose(ref["feat_std"][fi, c, side], f.std(axis=0), rtol=1e-10, atol=1e-15)
                np.testing.assert_allclose(ref["feat_quantiles"][fi, c, side], np.quantile(f, levels, axis=0))


def test_a_lopo_fold_is_a_mask_over_the_one_pool(ran):
    pool = ran.pool
    names = ran.meta["class_names"]
    assert ran.meta["folds"]["n_folds"] == len(ran.folds["folds"]) == 2
    for f, rec in zip(ran.folds["folds"], ran.meta["folds"]["folds"]):
        train_m, held_m = splits.fold_masks(pool.patient, f)
        assert (train_m ^ held_m).all() and held_m.any() and train_m.any()
        assert set(pool.patient[held_m].tolist()) <= set(f["heldout_patients"])
        assert rec["heldout_patients"] == f["heldout_patients"] and rec["dcis_patient"] == f["dcis_patient"]
        assert rec["heldout_pool_rows"] == {n: int(((pool.y == k) & held_m).sum()) for k, n in enumerate(names)}
        assert rec["train_pool_rows"] == {n: int(((pool.y == k) & train_m).sum()) for k, n in enumerate(names)}


# ============================================================================
# split access and registration
# ============================================================================

def test_access_log_shows_train_reads_only(ran):
    log = read_access_log(ran.out / "split_access_log.jsonl")
    mine = [r for r in log if r["stage"] == "pool"]
    assert mine and {(r["split"], r["purpose"], r["allowed"]) for r in mine} == {("train", "pool", True)}
    assert set(mine[0]["files"]) == {f"{k}_train.npy" for k in ("X", "y", "groups", "captures")}


def test_stage_registration_declares_its_purposes_and_upstreams():
    reg = load_builtin_stages()
    spec = reg.get("pool")
    assert (spec.dirname, spec.requires, spec.purposes, spec.group) == \
        ("pool", ("tissue-mask", "leakage"), ("pool",), None)
    assert spec.in_all and not spec.reads_held_out()
    assert "pool" not in reg.planned_only()
    assert splits.registry_policy_violations(reg) == []
    names = [s.name for s in reg.resolve(["all"])]
    assert names.index("pool") > max(names.index("tissue-mask"), names.index("leakage"))


# ============================================================================
# determinism, configuration and failures
# ============================================================================

def test_same_seed_and_config_give_the_same_pool(ran, tiny, tmp_path):
    res = _pool_only(tiny, tmp_path, PoolConfig(rows_per_class=QUOTA), ran.out)
    assert res.status == "ran" and res.fingerprint == ran.result.fingerprint
    done = json.loads((ran.dir / DONE_NAME).read_text())["outputs"]
    for name, meta in res.done["outputs"].items():
        if name != P.REFERENCE_NAME:                    # np.savez stamps zip members with the time
            assert meta["sha256"] == done[name]["sha256"], name
    again = P.load_reference(tmp_path / STAGE_DIR)
    assert set(again) == set(ran.ref)
    for k, v in ran.ref.items():
        np.testing.assert_array_equal(again[k], v, err_msg=k)


def test_subsampled_pca_and_nn_queries(ran, tiny, tmp_path):
    cfg = PoolConfig(rows_per_class=QUOTA, pca_components=8, pca_fit_rows=40, nn_max_queries=3,
                     quantiles=(0.01, 0.5))
    _pool_only(tiny, tmp_path, cfg, ran.out)
    ref = P.load_reference(tmp_path / STAGE_DIR)
    fit = ref["pca_fit_index"]
    assert fit.size == 40 and np.all(np.diff(fit) > 0) and fit.max() < ran.pool.n
    assert ref["pca_components"].shape == (8, ran.ref["pca_mean"].size)
    np.testing.assert_array_equal(ref["nn_n_query"], np.minimum(ref["nn_n_heldout"], 3))
    assert ref["nn_quantiles"].shape[-1] == 2 and ref["quantile_levels"].tolist() == [0.01, 0.5]
    np.testing.assert_array_equal(load_pool(tmp_path / STAGE_DIR).row, ran.pool.row)   # same pool
    meta = json.loads((tmp_path / STAGE_DIR / P.POOL_META_NAME).read_text())
    assert meta["reference"]["pca"]["n_fit"] == 40 and meta["reference"]["pca"]["seed_key"] == P.PCA_RNG_KEY


def test_size_cap_refuses_before_writing(ran, tiny, tmp_path):
    with pytest.raises(StageError, match="max_pool_gb"):
        _pool_only(tiny, tmp_path, PoolConfig(rows_per_class=QUOTA, max_pool_gb=1e-6), ran.out)
    d = tmp_path / STAGE_DIR
    assert not (d / DONE_NAME).exists() and not (d / POOL_FILES["X"]).exists()
    assert not (d / P.REFERENCE_NAME).exists() and (d / "failed.json").is_file()


def test_tau_sign_off(ran, tiny, tmp_path):
    tau = ran.meta["tissue"]["tau_deg"]
    assert ran.meta["tissue"]["tau_signed_off"] is False and ran.result.done["summary"]["tau_signed_off"] is False
    stale = tmp_path / "a" / STAGE_DIR / POOL_FILES["X"]           # a previous run's pool
    stale.parent.mkdir(parents=True)
    shutil.copy(ran.dir / POOL_FILES["X"], stale)
    with pytest.raises(StageError, match="signed-off tau"):
        _pool_only(tiny, tmp_path / "a", PoolConfig(rows_per_class=QUOTA, expect_tau_deg=tau + 0.01), ran.out)
    assert not stale.exists()                                       # never left looking complete
    res = _pool_only(tiny, tmp_path / "b", PoolConfig(rows_per_class=QUOTA, expect_tau_deg=round(tau, 3)), ran.out)
    meta = json.loads((tmp_path / "b" / STAGE_DIR / P.POOL_META_NAME).read_text())
    assert res.done["summary"]["tau_signed_off"] is True and meta["tissue"]["expect_tau_deg"] == round(tau, 3)


def test_upstreams_are_required_and_checked(ran, tiny, tmp_path):
    reg = load_builtin_stages()
    with pytest.raises(StageNotReady, match="tissue-mask|leakage"):
        run_stage(reg.get("pool"), PoolConfig(), RunOptions(data_dir=str(tiny.data_dir), out=str(tmp_path / "x")),
                  reg)
    out = tmp_path / "aug"
    for d in (TISSUE_DIR, LEAKAGE_DIR):
        shutil.copytree(ran.out / d, out / d)
    opts = RunOptions(data_dir=str(tiny.data_dir), out=str(out))
    mask_path = out / TISSUE_DIR / tissue.MASK_NAME
    mask = np.load(mask_path)
    np.save(mask_path, ~mask)                                         # not the rule applied to the score
    with pytest.raises(StageError, match="different runs"):
        run_stage(reg.get("pool"), PoolConfig(), opts, reg)
    np.save(mask_path, mask[:-1])                                     # wrong length
    with pytest.raises(StageError, match="do not describe this train split"):
        run_stage(reg.get("pool"), PoolConfig(), opts, reg)
    np.save(mask_path, mask)
    folds_path = out / LEAKAGE_DIR / splits.FOLDS_NAME
    doc = json.loads(folds_path.read_text())
    doc["n_train_rows"] += 1                                          # folds from another build
    folds_path.write_text(json.dumps(doc))
    with pytest.raises(StageError, match="folds.json does not fit"):
        run_stage(reg.get("pool"), PoolConfig(), opts, reg)
    assert not (out / STAGE_DIR / DONE_NAME).exists()


def test_config_validation():
    from hsi_augment.config import ConfigError
    for bad in ({"rows_per_class": 0}, {"minority_rows": 0}, {"max_pool_gb": 0.0}, {"expect_tau_deg": 0.0},
                {"pca_components": 0}, {"pca_fit_rows": 1}, {"nn_max_queries": 0}, {"quantiles": ()},
                {"quantiles": (0.5, 1.5)}, {"quantiles": (0.5, 0.5)}, {"chunk_rows": 0}):
        with pytest.raises(ConfigError):
            PoolConfig(**bad)
    cfg = PoolConfig()
    assert (cfg.rows_per_class, cfg.minority_rows, cfg.max_pool_gb, cfg.pca_components) == (120_000, None, 3.5, 64)


def test_nearest_neighbours_blocks_and_empty_inputs():
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(3)
    q, r = rng.normal(size=(23, 5)), rng.normal(size=(41, 5))
    d, j = P.nearest_neighbours(q, r, query_block=4, ref_block=7)
    full = cdist(q, r)
    np.testing.assert_allclose(d, full.min(axis=1), rtol=1e-12)
    np.testing.assert_array_equal(j, full.argmin(axis=1))
    d, j = P.nearest_neighbours(q, r[:0])
    assert np.isnan(d).all() and (j == -1).all()
    assert P.nearest_neighbours(q[:0], r)[0].shape == (0,)


# ============================================================================
# CLI, defaults and the real build's size
# ============================================================================

def test_cli_runs_skips_and_takes_every_tissue_row_by_default(tiny, tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["pool", "--help"])
    out_help = capsys.readouterr().out
    assert "--rows_per_class" in out_help and "--expect_tau_deg" in out_help and "requires tissue-mask" in out_help
    out = tmp_path / "aug"
    base = ["--data_dir", str(tiny.data_dir), "--out", str(out)]
    assert cli.main(["pool", *base]) == 1                                  # upstreams not done
    assert cli.main(["tissue-mask", *base]) == 0 and cli.main(["leakage", *base]) == 0
    assert cli.main(["pool", *base]) == 0
    assert "pool: ran" in capsys.readouterr().out
    pool = load_pool(out / STAGE_DIR)
    mask = np.load(out / TISSUE_DIR / tissue.MASK_NAME)
    tr = _train(tiny)
    np.testing.assert_array_equal(pool.row, np.flatnonzero(mask))       # 120k > the fixture: every tissue row
    assert check_against_train(pool, tr.y, tr.groups, tr.captures) == []
    assert cli.main(["pool", *base]) == 0 and "pool: skipped" in capsys.readouterr().out
    assert cli.main(["pool", *base, "--quantiles", "1.5"]) == 2 and "quantiles" in capsys.readouterr().err


def test_real_build_pool_fits_the_size_cap():
    """Acceptance "on real data, the pool is at most 3.5 GB", from the trainsel
    build's known counts: at most every DCIS train row (122,850, an upper bound
    on its tissue rows) plus 2 x 120,000, of 11 x 11 x 32 float16."""
    cfg = PoolConfig()
    worst = P.pool_nbytes(122_850 + 2 * cfg.rows_per_class, (11, 11, 32))
    assert worst == 362_850 * 7_744 == 2_809_910_400
    assert worst <= cfg.max_pool_gb * 1e9


REAL_DATA = os.environ.get("HSI_AUGMENT_REAL_DATA")


@pytest.mark.skipif(not REAL_DATA, reason="set HSI_AUGMENT_REAL_DATA to the trainsel hsi dir to run")
def test_real_data_smoke(tmp_path):
    """tissue-mask on 3 captures (so only their ~10k rows are tissue and read),
    leakage (int sidecars only), then a small pool. Never the whole X_train."""
    from hsi_augment.adapters import get_adapter
    real = get_adapter("histologyhsi_bc", REAL_DATA)
    assert P.pool_nbytes(122_850 + 2 * PoolConfig().rows_per_class, real.x_shape[1:]) <= 3.5e9
    reg = load_builtin_stages()
    opts = RunOptions(data_dir=REAL_DATA, out=str(tmp_path))
    run_stage(reg.get("tissue-mask"), TissueMaskConfig(max_captures=3, sample_rows=5_000, mosaics_per_class=1,
                                                       grid_strict=False), opts, reg)
    run_stage(reg.get("leakage"), LeakageConfig(), opts, reg)
    res = run_stage(reg.get("pool"), PoolConfig(rows_per_class=500, minority_rows=500, pca_components=16,
                                                pca_fit_rows=1_000, nn_max_queries=200), opts, reg)
    assert res.done["summary"]["tissue_complete"] is False
    pool = load_pool(tmp_path / STAGE_DIR)
    assert pool.X.shape[1:] == (11, 11, 32) and pool.n <= 1_500
    tr = SimpleNamespace(**{k: np.load(Path(REAL_DATA) / f"{k}_train.npy", mmap_mode="r")
                            for k in ("y", "groups", "captures")})
    assert check_against_train(pool, tr.y, tr.groups, tr.captures) == []
    assert res.done["summary"]["round_trip_max_rel_err"] < 1e-2
    files = {f for r in read_access_log(tmp_path / "split_access_log.jsonl") if r["stage"] == "pool"
             for f in r["files"]}
    assert "X_train.npy" in files and not any(f.endswith(("_val.npy", "_test.npy")) for f in files)
