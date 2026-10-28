# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_cvae.py
==============================
Phase 6 (#23): the conditional VAE (`hsi_augment/models/cvae.py`), its stages
`cvae-lopo` / `cvae-final` (group `fit-cvae`; `hsi_augment/fit_cvae.py`) and
the pool loader they read (`hsi_augment/pool_io.py`).

CPU acceptance of the issue: the model overfits one batch; resume is bitwise
(a stage killed mid-fold and continued with `--resume auto` ends with the same
checkpoints, diagnostics and summary as an uninterrupted run); the sampling
sha256 is deterministic and independent of how the rows are requested. Also:
shapes, a finite loss, the SAM calibration, weights_only checkpoints, fold
directories that do not overwrite each other, the final stage refusing to
run without LOPO results, and a split-access log with train only.

Everything runs on `tests/hsi_augment_fixtures.make_tiny_build` (5x5x8
patches, 2 DCIS train patients -> 2 LOPO folds) with a ~24k-parameter model;
the pool is written here with `pool_io.write_pool` (`make_tiny_pool`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch

from hsi_augment import fit_cvae
from hsi_augment.codec import LogSpaceCodec, make_decoded_sam_loss
from hsi_augment.io import read_access_log
from hsi_augment.models.blocks import assert_no_batchnorm
from hsi_augment.models.cvae import (CVAE, CVAELoss, GEN_BLOCK_ROWS, LOG_2PI, beta_at, calibrate_sam_weight,
                                     count_parameters, load_cvae, sample)
from hsi_augment.pool_io import (POOL_FILES, PoolError, check_against_train, load_pool, pool_fingerprint,
                                 write_pool)
from hsi_augment.registry import load_builtin_stages
from hsi_augment.splits import registry_policy_violations
from hsi_augment.stage import DONE_NAME, RunOptions, StageError, StageNotReady, read_done, run_stage
from hsi_augment.stages.fit_cvae import CVAEFinalConfig, CVAELopoConfig
from hsi_augment.stages.leakage import LeakageConfig
from hsi_augment.train_generative import (BEST_NAME, HISTORY_NAME, LAST_NAME, TIMING_COLUMNS, StepInfo,
                                          TrainConfig, Trainer, load_checkpoint)
from tests.hsi_augment_fixtures import make_tiny_build

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY = dict(widths=(8, 16, 16), cond_dim=8)
RUN_FILES = (BEST_NAME, LAST_NAME, HISTORY_NAME, "model.json", "calibration.json", "diagnostics.json")


# ============================================================================
# fixtures and helpers
# ============================================================================

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def registry():
    return load_builtin_stages()


def tiny_codec(X: np.ndarray) -> LogSpaceCodec:
    stats = LogSpaceCodec.fit_stats([X])
    p9999 = np.percentile(X.reshape(-1, X.shape[-1]).astype(np.float64), 99.99, axis=0)
    return LogSpaceCodec.from_stats(stats, LogSpaceCodec.clip_hi_from_quantile(p9999))


def make_tiny_pool(tiny, pool_dir: Path) -> Path:
    """Every train row of the tiny build, encoded, in the pool contract's layout."""
    d = tiny.data_dir
    X = np.load(d / "X_train.npy")
    codec = tiny_codec(X)
    return write_pool(pool_dir, codec.encode_np(X).astype(np.float16), np.load(d / "y_train.npy"),
                      np.load(d / "groups_train.npy"), np.arange(X.shape[0]), np.load(d / "captures_train.npy"),
                      codec)


def opts(tiny, out: Path, **kw) -> RunOptions:
    return RunOptions(data_dir=str(tiny.data_dir), out=str(out), **kw)


def prepare(tiny, registry, out: Path, *, pool: bool = True) -> Path:
    """`out` with the leakage stage done and (optionally) a pool."""
    run_stage(registry.get("leakage"), LeakageConfig(), opts(tiny, out), registry)
    if pool:
        make_tiny_pool(tiny, out / "pool")
    return out


def lopo_cfg(**kw) -> CVAELopoConfig:
    base = dict(latent_dims=(4,), sam_ratios=(0.1,), max_steps=6, eval_every=3, log_every=2, ckpt_every=2,
                batch_size=12, eval_batch_size=16, early_stop_patience=None, ema_decay=0.9,
                diag_train_rows=40, prior_samples=16, device="cpu", **TINY)
    base.update(kw)
    return CVAELopoConfig(**base)


def final_cfg(**kw) -> CVAEFinalConfig:
    base = dict(enforce_gates=False, device="cpu", prior_samples=16)
    base.update(kw)
    return CVAEFinalConfig(**base)


def tiny_model(seed: int = 0, latent_dim: int = 4) -> CVAE:
    torch.manual_seed(seed)
    return CVAE(in_bands=8, patch=5, n_classes=3, latent_dim=latent_dim, **TINY)


def tiny_batch(tiny, n: int = 12):
    X = np.load(tiny.data_dir / "X_train.npy")
    y = np.load(tiny.data_dir / "y_train.npy")
    codec = tiny_codec(X)
    rows = np.concatenate([np.flatnonzero(y == c)[: n // 3] for c in range(3)])
    x = torch.from_numpy(codec.encode_np(X[rows])).permute(0, 3, 1, 2).contiguous()
    return {"x": x, "y": torch.from_numpy(y[rows].astype(np.int64))}, codec


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def assert_same(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {set(a) ^ set(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b)):
            assert_same(u, v, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def no_timing(rows):
    return [{k: v for k, v in r.items() if k not in TIMING_COLUMNS} for r in rows]


def info(step: int = 0, max_steps: int = 100, training: bool = True) -> StepInfo:
    return StepInfo(step=step, epoch=0, step_in_epoch=step, max_steps=max_steps, training=training)


# ============================================================================
# the model and the loss
# ============================================================================

@pytest.mark.parametrize("patch", [5, 7, 9, 11])
def test_shapes_for_any_patch_and_no_batchnorm(patch):
    torch.manual_seed(0)
    m = CVAE(in_bands=8, patch=patch, n_classes=3, latent_dim=4, **TINY)
    x, y = torch.randn(3, 8, patch, patch), torch.tensor([0, 1, 2])
    x_mu, mu, logvar = m(x, y)
    assert x_mu.shape == x.shape and mu.shape == logvar.shape == (3, 4)
    assert m.decode(torch.zeros(2, 4), torch.tensor([1, 1])).shape == (2, 8, patch, patch)
    assert_no_batchnorm(m)
    with pytest.raises(ValueError, match="NCHW"):
        m.encode(x.permute(0, 2, 3, 1), y)


def test_full_size_model_matches_the_plan():
    """~3.5M parameters in the plan; 3.9M at d = 64, 3.7M at d = 32 here."""
    for d, lo, hi in ((64, 3.8e6, 4.0e6), (32, 3.6e6, 3.8e6)):
        m = CVAE(latent_dim=d)
        assert lo < count_parameters(m) < hi
        assert_no_batchnorm(m)
    m = CVAE(latent_dim=64)
    x_mu, mu, _ = m(torch.randn(2, 32, 11, 11), torch.tensor([0, 1]))
    assert x_mu.shape == (2, 32, 11, 11) and mu.shape == (2, 64) and m.bottleneck == 3


def test_loss_is_finite_and_its_terms_add_up(tiny):
    batch, codec = tiny_batch(tiny)
    m = tiny_model()
    loss_fn = CVAELoss(codec, sam_weight=0.5, free_bits=0.05, beta_warmup_frac=0.1)
    torch.manual_seed(1)
    loss, met = loss_fn(m, batch, info(step=5, max_steps=100))
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    n_el = batch["x"][0].numel()
    assert met["beta"] == pytest.approx(0.5)
    assert float(met["nll"]) == pytest.approx(float(met["rec_nll"]) + float(met["kl"]) / n_el, rel=1e-6)
    assert float(met["kl_fb"]) >= 0.05 * m.latent_dim - 1e-6 and float(met["kl_fb"]) >= float(met["kl"]) - 1e-6
    expected = float(met["rec_nll"]) + 0.5 * float(met["sam"]) + 0.5 * float(met["kl_fb"]) / n_el
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-5)
    # at init sigma = 1, so rec_nll = 0.5 mse + 0.5 log 2 pi
    assert float(met["rec_nll"]) == pytest.approx(0.5 * float(met["mse"]) + 0.5 * LOG_2PI, rel=1e-5)


def test_beta_warmup_and_free_bits():
    assert [beta_at(s, 1000, 0.1) for s in (0, 50, 100, 500)] == [0.0, 0.5, 1.0, 1.0]
    assert beta_at(0, 10, 0.0) == 1.0
    loss = CVAELoss(LogSpaceCodec([0.0] * 2, [1.0] * 2, 10.0), free_bits=0.05)
    m = CVAE(in_bands=2, patch=5, n_classes=3, latent_dim=3, **TINY)
    with torch.no_grad():                         # q = prior: true KL 0, every dim floored
        m.enc_head.weight.zero_()
        m.enc_head.bias.zero_()
    _, met = loss(m, {"x": torch.randn(4, 2, 5, 5), "y": torch.tensor([0, 1, 2, 0])}, info())
    assert float(met["kl"]) == pytest.approx(0.0, abs=1e-7) and float(met["kl_fb"]) == pytest.approx(0.15)


def test_sam_weight_calibration(tiny):
    batch, codec = tiny_batch(tiny)
    m = tiny_model()
    probe = CVAELoss(codec)
    rng_before = torch.get_rng_state()
    cal = calibrate_sam_weight(m, probe, [batch, batch], 0.1, seed=3)
    assert torch.equal(rng_before, torch.get_rng_state())          # global RNG untouched
    assert cal["lambda"] * cal["sam_init_rad"] == pytest.approx(0.1 * cal["rec_init"], rel=1e-9)
    assert cal["rec_init"] == pytest.approx(cal["rec_init_with_constant"] - 0.5 * LOG_2PI)
    assert cal["rows"] == 2 * len(batch["y"]) and m.training
    assert calibrate_sam_weight(m, probe, [batch, batch], 0.1, seed=3) == cal       # deterministic
    assert calibrate_sam_weight(m, probe, [batch], 0.0, seed=3)["lambda"] == 0.0


def test_the_model_overfits_one_batch(tiny, tmp_path):
    batch, codec = tiny_batch(tiny)

    class OneBatch:
        def num_batches(self, epoch):
            return 1

        def get_batch(self, epoch, i):
            return batch

    torch.manual_seed(0)
    m = tiny_model(latent_dim=8)
    loss = CVAELoss(codec, sam_weight=0.0, beta_warmup_frac=0.2)
    with torch.no_grad():
        mse0 = float(((m(batch["x"], batch["y"], sample=False)[0] - batch["x"]) ** 2).mean())
    cfg = TrainConfig(max_steps=300, lr=3e-3, min_lr=3e-4, ema_decay=None, log_every=50, ckpt_every=1000)
    Trainer(m, loss, OneBatch(), cfg, run_dir=tmp_path, seed=0).fit()
    with torch.no_grad():
        mse = float(((m(batch["x"], batch["y"], sample=False)[0] - batch["x"]) ** 2).mean())
    assert mse < 0.05 * mse0, (mse0, mse)


def test_spectral_angle_is_unclamped_and_exact():
    a = torch.tensor([[1.0, 2.0, 3.0]]).view(1, 3, 1, 1)
    assert float(fit_cvae.spectral_angle_deg(a, a)) == 0.0
    assert float(fit_cvae.spectral_angle_deg(a, 7.0 * a)) == pytest.approx(0.0, abs=1e-12)
    u = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
    v = torch.tensor([1.0, 1.0]).view(1, 2, 1, 1)
    assert float(fit_cvae.spectral_angle_deg(u, v)) == pytest.approx(45.0, abs=1e-12)
    eps = 1e-4                                                     # 0.0057 deg: below the 2.56 deg floor
    w = torch.tensor([1.0, eps], dtype=torch.float64).view(1, 2, 1, 1)
    assert float(fit_cvae.spectral_angle_deg(u.double(), w)) == pytest.approx(math.degrees(math.atan(eps)),
                                                                                rel=1e-9)
    assert float(fit_cvae.spectral_angle_deg(u, torch.zeros_like(u))) == pytest.approx(90.0)
    from training.spectral_recon_metrics_v2 import sam_deg
    assert sam_deg(a, a) == pytest.approx(2.5626, abs=1e-3)       # the clamped metric's floor


# ============================================================================
# sampling
# ============================================================================

def test_sample_is_deterministic_and_independent_of_batching(tiny):
    _, codec = tiny_batch(tiny)
    m = tiny_model(seed=5)
    a = sample(m, codec, 1, 10, 1.0, seed=7)
    assert a.dtype == np.float16 and a.shape == (10, 5, 5, 8)
    assert np.all(np.isfinite(a)) and np.all(a >= 0) and np.all(a <= codec.clip_hi.astype(np.float16))
    assert sha(a) == sha(sample(m, codec, 1, 10, 1.0, seed=7))
    pieces = np.concatenate([sample(m, codec, 1, 3, 1.0, seed=7), sample(m, codec, 1, 7, 1.0, seed=7, start=3)])
    assert sha(pieces) == sha(a)
    # across a decode-block boundary, and a fresh model with the same weights
    big = sample(m, codec, 1, GEN_BLOCK_ROWS + 6, 1.0, seed=7)
    m2 = tiny_model(seed=5)
    across = sample(m2, codec, 1, 5, 1.0, seed=7, start=GEN_BLOCK_ROWS - 2)
    assert sha(across) == sha(big[GEN_BLOCK_ROWS - 2:GEN_BLOCK_ROWS + 3]) and sha(big[:10]) == sha(a)
    # class, tau and seed each change the stream; the model's train mode is restored
    assert sha(sample(m, codec, 2, 10, 1.0, seed=7)) != sha(a)
    assert sha(sample(m, codec, 1, 10, 0.8, seed=7)) != sha(a)
    assert sha(sample(m, codec, 1, 10, 1.0, seed=8)) != sha(a)
    assert m.training and sample(m, codec, 1, 0, 1.0, seed=7).shape == (0, 5, 5, 8)
    with pytest.raises(ValueError):
        sample(m, codec, 3, 1, 1.0, seed=7)


# ============================================================================
# pool_io
# ============================================================================

def test_pool_io_round_trip_and_validation(tiny, tmp_path):
    d = make_tiny_pool(tiny, tmp_path / "pool")
    pool = load_pool(d)
    assert isinstance(pool.X, np.memmap) and pool.X.dtype == np.float16
    assert (pool.n, pool.patch, pool.n_bands) == (tiny.n_rows["train"], tiny.patch, tiny.n_bands)
    assert pool.y.dtype == np.int64 and pool.patient.dtype == np.int32 and pool.capture.dtype == np.int32
    assert pool.codec.to_dict() == json.loads((d / "codec.json").read_text())
    y = np.load(tiny.data_dir / "y_train.npy")
    g = np.load(tiny.data_dir / "groups_train.npy")
    assert check_against_train(pool, y, g) == []
    assert check_against_train(pool, y, g[::-1].copy()) != []
    assert "another build" in check_against_train(pool, y[:10], g[:10])[0]
    fp = pool_fingerprint(d)
    assert fp == pool.fingerprint() and len(fp) == 64

    np.save(d / POOL_FILES["patient"], pool.patient.astype(np.int64))
    with pytest.raises(PoolError, match="patient_pool.npy is int64"):
        load_pool(d)
    with pytest.raises(PoolError, match="bands"):
        write_pool(tmp_path / "bad", np.zeros((2, 5, 5, 3), np.float16), [0, 1], [1, 2], [0, 1], [0, 0],
                   pool.codec)
    with pytest.raises(PoolError, match="duplicate"):
        write_pool(tmp_path / "bad", np.zeros((2, 5, 5, 8), np.float16), [0, 1], [1, 2], [0, 0], [0, 0],
                   pool.codec)
    (d / POOL_FILES["y"]).unlink()
    with pytest.raises(FileNotFoundError, match="y_pool.npy"):
        load_pool(d)
    with pytest.raises(FileNotFoundError, match="no pool directory"):
        load_pool(tmp_path / "nowhere")


# ============================================================================
# the stages
# ============================================================================

def test_stages_are_registered_train_only(registry):
    lopo, final = registry.get("cvae-lopo"), registry.get("cvae-final")
    assert (lopo.group, lopo.dirname, lopo.requires, lopo.purposes) == \
        ("fit-cvae", "gen/cvae/lopo", ("leakage",), ("generator",))
    assert (final.group, final.dirname, final.requires, final.purposes) == \
        ("fit-cvae", "gen/cvae/final", ("cvae-lopo",), ("generator",))
    assert [s.name for s in registry.resolve(["fit-cvae"])] == ["cvae-lopo", "cvae-final"]
    assert not lopo.reads_held_out() and not final.reads_held_out()
    assert registry_policy_violations(registry) == []


@pytest.fixture(scope="module")
def ran(tiny, registry, tmp_path_factory):
    """A 2-cell grid x 2 folds, then the final refit: shared by the read-only tests."""
    out = prepare(tiny, registry, tmp_path_factory.mktemp("cvae"))
    cfg = lopo_cfg(latent_dims=(2, 4))
    lopo = run_stage(registry.get("cvae-lopo"), cfg, opts(tiny, out), registry)
    final = run_stage(registry.get("cvae-final"), final_cfg(), opts(tiny, out), registry)
    return SimpleNamespace(out=out, lopo=lopo, final=final, cfg=cfg, lopo_dir=out / "gen/cvae/lopo",
                           final_dir=out / "gen/cvae/final")


def test_fold_runs_have_their_own_directories(ran):
    cells, folds = ("d2-r0.1", "d4-r0.1"), (0, 1)
    shas, heldout = set(), {}
    for cell in cells:
        for k in folds:
            run = fit_cvae.lopo_run_dir(ran.out, cell, k)
            assert run == ran.lopo_dir / cell / f"fold-{k}"
            for name in RUN_FILES:
                assert (run / name).is_file(), run / name
                assert f"{cell}/fold-{k}/{name}" in ran.lopo.done["outputs"]
            meta = json.loads((run / "model.json").read_text())
            assert (meta["cell"], meta["fold"], meta["model"]["latent_dim"]) == (cell, k, int(cell[1]))
            heldout.setdefault(k, set()).add(tuple(meta["heldout_patients"]))
            shas.add(ran.lopo.done["outputs"][f"{cell}/fold-{k}/{BEST_NAME}"]["sha256"])
    assert len(shas) == 4                                          # four different fits
    assert all(len(v) == 1 for v in heldout.values()) and heldout[0] != heldout[1]
    summary = json.loads((ran.lopo_dir / fit_cvae.LOPO_SUMMARY_NAME).read_text())
    assert set(summary["cells"]) == set(cells) and summary["n_folds"] == 2
    for c in summary["cells"].values():
        assert [f["fold"] for f in c["folds"]] == [0, 1]
        agg = c["aggregate"]
        assert agg["median_best_step"] == fit_cvae.median_step(agg["best_steps"])
        assert agg["sam_ratio"] == pytest.approx(agg["sam_deg"] / agg["sam_train_deg"])


def test_diagnostics_are_on_held_out_patients(ran, tiny):
    folds = json.loads((ran.out / "prep/leakage/folds.json").read_text())["folds"]
    for k, f in enumerate(folds):
        diag = json.loads((ran.lopo_dir / "d4-r0.1" / f"fold-{k}" / "diagnostics.json").read_text())
        groups = np.load(tiny.data_dir / "groups_train.npy")
        assert diag["heldout"]["rows"] == int(np.isin(groups, f["heldout_patients"]).sum())
        assert diag["heldout_patients"] == f["heldout_patients"]
        for c in ("0", "1", "2"):
            pc = diag["heldout"]["per_class"][c]
            assert pc["rows"] > 0 and pc["sam_deg"] > 0 and pc["mse"] > 0 and -1 <= pc["ssim2d"] <= 1
        lat = diag["heldout"]["latent"]
        assert len(lat["kl_per_dim"]) == 4 and 0 <= lat["active_units"] <= 4
        assert set(diag["prior_samples"]["per_tau"]) == {"0.8", "1"}
        assert diag["prior_samples"]["class"] == tiny.minority_class
        assert diag["best_step"] in (3, 6) and diag["sam_weight"] > 0


def test_final_refit_uses_the_lopo_choice_and_ema(ran):
    sel = json.loads((ran.final_dir / fit_cvae.SELECTION_NAME).read_text())
    summary = json.loads((ran.lopo_dir / fit_cvae.LOPO_SUMMARY_NAME).read_text())
    chosen = summary["cells"][sel["chosen"]]
    assert sel["budget_steps"] == chosen["aggregate"]["median_best_step"]
    assert sel["fold_best_steps"] == chosen["aggregate"]["best_steps"]
    nll = {r["cell"]: r["value"] for r in sel["table"]}
    assert sel["chosen"] == min(nll, key=lambda c: (nll[c], int(c[1])))
    ckpt = load_checkpoint(ran.final_dir / BEST_NAME)
    assert ckpt["step"] == sel["budget_steps"] and ckpt["finished"] and ckpt["ema"] is not None
    assert ckpt["config"]["latent_dim"] == sel["latent_dim"] and ckpt["config"]["early_stop_patience"] is None
    model, codec, meta = load_cvae(ran.final_dir)
    assert meta["tau_s_candidates"] == [0.8, 1.0] and meta["tau_s"] is None and meta["weights"] == "ema"
    ema = ckpt["ema"]["module"]
    assert all(torch.equal(v, ema[k]) for k, v in model.state_dict().items())
    x = sample(model, codec, 1, 4, 0.8, seed=0)
    assert x.shape == (4, 5, 5, 8) and x.dtype == np.float16
    assert ran.final.done["summary"]["chosen"] == sel["chosen"]


def test_checkpoints_load_with_weights_only(ran):
    paths = sorted(ran.lopo_dir.glob("*/fold-*/*.pt")) + sorted(ran.final_dir.glob("*.pt"))
    assert len(paths) == 10
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=True)
        assert ck["format"] == "hsi_augment.train_generative/1" and "log_sigma" in ck["model"]


def test_the_access_log_shows_train_only(ran):
    rec = [r for r in read_access_log(ran.out / "split_access_log.jsonl")
           if r["stage"] in ("cvae-lopo", "cvae-final")]
    assert {r["stage"] for r in rec} == {"cvae-lopo", "cvae-final"}
    assert all(r["split"] == "train" and r["allowed"] and r["purpose"] == "generator" for r in rec)
    assert all(not f.startswith("X_") for r in rec for f in r["files"])     # sidecars only; X is the pool's


def test_final_refuses_to_run_without_lopo_results(tiny, registry, tmp_path):
    out = prepare(tiny, registry, tmp_path / "aug")
    with pytest.raises(StageNotReady, match="cvae-lopo"):
        run_stage(registry.get("cvae-final"), final_cfg(), opts(tiny, out), registry)
    assert not (out / "gen/cvae/final" / DONE_NAME).exists()


def test_a_missing_pool_is_a_stage_error(tiny, registry, tmp_path):
    out = prepare(tiny, registry, tmp_path / "aug", pool=False)
    with pytest.raises(StageError, match="no pool directory"):
        run_stage(registry.get("cvae-lopo"), lopo_cfg(), opts(tiny, out), registry)
    assert not (out / "gen/cvae/lopo" / DONE_NAME).exists()


def test_gates_refuse_a_collapsed_grid(ran, registry, tiny):
    with pytest.raises(StageError, match="no LOPO cell passes the gates"):
        fit_cvae.select_cell(json.loads((ran.lopo_dir / fit_cvae.LOPO_SUMMARY_NAME).read_text()),
                             final_cfg(enforce_gates=True, min_active_frac=1.0, max_sam_ratio=1e-9),
                             free_bits=0.05)


# ============================================================================
# bitwise resume, inside the stages
# ============================================================================

def _killer(monkeypatch, crash: Dict[str, Any]):
    """Wrap every fit's step_fn; the `crash["run"]`-th fit created raises at
    training step `crash["step"]` while `crash["on"]`."""
    real = fit_cvae.make_step_fn
    crash["made"] = 0

    def make(loss):
        crash["made"] += 1
        me = crash["made"]
        inner = real(loss)

        def step(model, batch, info):
            if crash["on"] and me == crash["run"] and info.training and info.step == crash["step"]:
                raise RuntimeError("killed mid-fit")
            return inner(model, batch, info)
        return step

    monkeypatch.setattr(fit_cvae, "make_step_fn", make)


def _compare_runs(a: Path, b: Path) -> None:
    for key in ("model", "ema", "optimizer", "scheduler", "rng", "step", "early_stopping", "best"):
        assert_same(load_checkpoint(a / LAST_NAME)[key], load_checkpoint(b / LAST_NAME)[key], f"{b} {key}")
    assert_same(load_checkpoint(a / BEST_NAME)["ema"], load_checkpoint(b / BEST_NAME)["ema"], f"{b} best")
    assert no_timing(load_checkpoint(a / LAST_NAME)["history"]) == \
        no_timing(load_checkpoint(b / LAST_NAME)["history"])
    for name in ("diagnostics.json", "calibration.json"):
        assert json.loads((a / name).read_text()) == json.loads((b / name).read_text()), name


def test_resume_auto_is_bitwise_inside_both_stages(tiny, registry, tmp_path, monkeypatch):
    cfg, fcfg = lopo_cfg(), final_cfg()
    ref = prepare(tiny, registry, tmp_path / "ref")
    run_stage(registry.get("cvae-lopo"), cfg, opts(tiny, ref), registry)
    run_stage(registry.get("cvae-final"), fcfg, opts(tiny, ref), registry)

    out = prepare(tiny, registry, tmp_path / "run")
    crash = {"on": True, "run": 2, "step": 4}                    # fold 1, after its step-4 checkpoint
    _killer(monkeypatch, crash)
    with pytest.raises(RuntimeError, match="killed"):
        run_stage(registry.get("cvae-lopo"), cfg, opts(tiny, out), registry)
    lopo = out / "gen/cvae/lopo"
    assert not (lopo / DONE_NAME).exists() and (lopo / "d4-r0.1/fold-0/diagnostics.json").is_file()
    assert load_checkpoint(lopo / "d4-r0.1/fold-1" / LAST_NAME)["step"] == 4
    crash["on"] = False
    r = run_stage(registry.get("cvae-lopo"), cfg, opts(tiny, out, resume="auto"), registry)
    assert r.done["resumed"] and r.done["attempt"] == 2
    for k in (0, 1):
        _compare_runs(ref / "gen/cvae/lopo/d4-r0.1" / f"fold-{k}", lopo / "d4-r0.1" / f"fold-{k}")
    sa = json.loads((ref / "gen/cvae/lopo" / fit_cvae.LOPO_SUMMARY_NAME).read_text())
    sb = json.loads((lopo / fit_cvae.LOPO_SUMMARY_NAME).read_text())
    assert sa["cells"] == sb["cells"] and sa["pool"]["sha256"] == sb["pool"]["sha256"]
    assert set(r.done["outputs"]) == set(read_done(ref / "gen/cvae/lopo")["outputs"])

    crash.update(on=True, run=crash["made"] + 1, step=1)       # the final refit's first fit
    with pytest.raises(RuntimeError, match="killed"):
        run_stage(registry.get("cvae-final"), fcfg, opts(tiny, out), registry)
    crash["on"] = False
    r = run_stage(registry.get("cvae-final"), fcfg, opts(tiny, out, resume="auto"), registry)
    assert r.done["resumed"] and r.done["summary"]["steps"] == read_done(ref / "gen/cvae/final")["summary"]["steps"]
    _compare_runs(ref / "gen/cvae/final", out / "gen/cvae/final")


def test_a_changed_pool_is_refused_by_the_final_stage(tiny, registry, tmp_path):
    out = prepare(tiny, registry, tmp_path / "aug")
    run_stage(registry.get("cvae-lopo"), lopo_cfg(), opts(tiny, out), registry)
    pool = load_pool(out / "pool")
    X = np.array(pool.X)
    X[0, 0, 0, 0] = np.float16(0.5) + X[0, 0, 0, 0]
    write_pool(out / "pool", X, pool.y, pool.patient, pool.row, pool.capture, pool.codec)
    with pytest.raises(StageError, match="changed since cvae-lopo"):
        run_stage(registry.get("cvae-final"), final_cfg(), opts(tiny, out), registry)


# ============================================================================
# command line
# ============================================================================

def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600)


def test_cli_fit_cvae_group_runs_both_stages(tiny, registry, tmp_path):
    r = _cli("cvae-lopo", "--help")
    assert r.returncode == 0 and "--latent_dims" in r.stdout and "--sam_ratios" in r.stdout
    out = prepare(tiny, registry, tmp_path / "aug")
    cfg = {"stages": {"cvae-lopo": lopo_cfg(max_steps=3, eval_every=3).to_dict(),
                      "cvae-final": final_cfg().to_dict()}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    r = _cli("fit-cvae", "--data_dir", tiny.data_dir, "--out", out, "--config", tmp_path / "cfg.json")
    assert r.returncode == 0, r.stdout + r.stderr
    for stage in ("lopo", "final"):
        assert read_done(out / "gen/cvae" / stage)["status"] == "ok"
    r = _cli("fit-cvae", "--data_dir", tiny.data_dir, "--out", out, "--config", tmp_path / "cfg.json")
    assert r.returncode == 0 and r.stdout.count("already done") == 2
