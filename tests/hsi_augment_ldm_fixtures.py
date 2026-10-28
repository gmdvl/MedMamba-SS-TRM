# -*- coding: utf-8 -*-
"""
tests/hsi_augment_ldm_fixtures.py
=================================
Shared by `test_hsi_augment_autoencoder.py` and `test_hsi_augment_diffusion.py`
(Phase 7, #24, arm A3): the generator pool of the tiny build of
`tests/hsi_augment_fixtures.py`, tiny stage configs, and helpers that run the
`fit-ae` / `fit-ldm` stages through the stage API and compare two runs bit
for bit.

    out = prepare(tiny, registry, tmp_path / "aug")     # tissue-mask, leakage and pool done
    run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())  # StageResult

The pool is the real `pool` stage's, run through the stage API with its
default config: tissue rows only, and the healthy / IDC quotas
(`rows_per_class`, 120k) exceed the fixture, so it holds every train tissue
row (184 of the 240 train rows), in train-row order. The tiny build has 6
train patients, 2 of them with DCIS, so `leakage` makes 2 LOPO folds; the
models are 5x5x8 patches -> a 2x3x3 latent, ~17k (autoencoder) and ~98k
(denoiser) parameters, 6 optimizer steps per fit: every stage runs in about
a second on a CPU.

`StepSpy` patches a loss class's `__call__` (AELoss, DiffusionLoss) to record
the pool rows of every training and held-out batch (fold isolation) and,
when armed, to kill a fit at a given step (bitwise resume).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hsi_augment.codec import LogSpaceCodec
from hsi_augment.stage import RunOptions, run_stage
from hsi_augment.stages.fit_ae import AEFinalConfig, AELopoConfig
from hsi_augment.stages.fit_ldm import LDMFinalConfig, LDMLopoConfig
from hsi_augment.stages.leakage import LeakageConfig
from hsi_augment.stages.pool import PoolConfig
from hsi_augment.stages.tissue_mask import TissueMaskConfig
from hsi_augment.train_generative import BEST_NAME, LAST_NAME, TIMING_COLUMNS, load_checkpoint

TINY_AE = dict(widths=(8, 16), blocks_per_level=1, latent_channels=2)
TINY_LDM = dict(channels=(16, 32), blocks_per_level=1, mid_blocks=1, emb_dim=16, freq_dim=8, timesteps=50,
                sample_steps=5)
TINY_LOOP = dict(max_steps=6, warmup_steps=1, eval_every=3, log_every=2, ckpt_every=2, ema_decay=0.9,
                 batch_size=12, eval_batch_size=32, device="cpu")


# ============================================================================
# the pool and the stages
# ============================================================================

def tiny_codec(X: np.ndarray) -> LogSpaceCodec:
    stats = LogSpaceCodec.fit_stats([X])
    p9999 = np.percentile(X.reshape(-1, X.shape[-1]).astype(np.float64), 99.99, axis=0)
    return LogSpaceCodec.from_stats(stats, LogSpaceCodec.clip_hi_from_quantile(p9999))


def opts(tiny, out: Path, **kw) -> RunOptions:
    return RunOptions(data_dir=str(tiny.data_dir), out=str(out), **kw)


def prepare(tiny, registry, out: Path, *, pool: bool = True) -> Path:
    """`out` with tissue-mask and leakage done and, unless `pool` is False,
    the real pool stage (default config) too, at <out>/pool."""
    stages = [("tissue-mask", TissueMaskConfig()), ("leakage", LeakageConfig())]
    if pool:
        stages.append(("pool", PoolConfig()))
    for name, cfg in stages:
        run_stage(registry.get(name), cfg, opts(tiny, out), registry)
    return out


def run(registry, tiny, out: Path, name: str, cfg, **kw):
    return run_stage(registry.get(name), cfg, opts(tiny, out, **kw), registry)


def ae_lopo_cfg(**kw) -> AELopoConfig:
    return AELopoConfig(**{**TINY_LOOP, **TINY_AE, "metric_rows": 20, **kw})


def ae_final_cfg(**kw) -> AEFinalConfig:
    return AEFinalConfig(**{"device": "cpu", "metric_rows": 20, **kw})


def ldm_lopo_cfg(**kw) -> LDMLopoConfig:
    return LDMLopoConfig(**{**TINY_LOOP, **TINY_LDM, "guidance": (1.0, 2.0), "encode_batch_size": 64,
                            "sample_batch_size": 16, "mmd_rows": 16, "precision": "fp32", **kw})


def ldm_final_cfg(**kw) -> LDMFinalConfig:
    return LDMFinalConfig(**{"device": "cpu", **kw})


# ============================================================================
# recording / killing fits
# ============================================================================

class StepSpy:
    """Replaces `loss_cls.__call__` (through `mp`, a pytest MonkeyPatch) with a
    wrapper that records every call as {fit, training, step, rows}: `fit`
    counts the fits seen (a training call at step 0 starts a new one), `rows`
    are the batch's POOL rows. Arm it with `crash = (fit, step)` to raise
    RuntimeError("killed mid-fit") at that training step of that fit."""

    def __init__(self, mp, loss_cls):
        self.calls: List[Dict[str, Any]] = []
        self.fits = 0
        self.crash: Optional[tuple] = None
        real = loss_cls.__call__
        spy = self

        def call(loss, model, batch, info):
            if info.training and info.step == 0:
                spy.fits += 1
            if spy.crash is not None and info.training and (spy.fits, info.step) == tuple(spy.crash):
                raise RuntimeError("killed mid-fit")
            spy.calls.append({"fit": spy.fits, "training": bool(info.training), "step": int(info.step),
                              "rows": batch["row"].cpu().numpy().copy()})
            return real(loss, model, batch, info)

        mp.setattr(loss_cls, "__call__", call)

    def rows(self, fit: int, training: bool) -> np.ndarray:
        got = [c["rows"] for c in self.calls if c["fit"] == fit and c["training"] == training]
        return np.concatenate(got) if got else np.empty(0, dtype=np.int64)

    def steps(self, fit: int) -> List[int]:
        return [c["step"] for c in self.calls if c["fit"] == fit and c["training"]]


# ============================================================================
# comparing runs
# ============================================================================

def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def assert_same(a: Any, b: Any, path: str = "") -> None:
    """Exact equality of nested checkpoint values (tensors bit for bit)."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {set(a) ^ set(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    else:
        assert a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b)), f"{path}: {a!r} != {b!r}"


def no_timing(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in TIMING_COLUMNS} for r in history]


def assert_same_training(a: Path, b: Path) -> None:
    """Two Trainer run directories hold the same weights, optimizer, RNG and
    history (wall-clock columns aside)."""
    la, lb = load_checkpoint(a / LAST_NAME), load_checkpoint(b / LAST_NAME)
    for key in ("model", "ema", "optimizer", "scheduler", "rng", "step", "epoch", "early_stopping", "best"):
        assert_same(la[key], lb[key], f"{b} {key}")
    assert no_timing(la["history"]) == no_timing(lb["history"])
    assert_same(load_checkpoint(a / BEST_NAME)["ema"], load_checkpoint(b / BEST_NAME)["ema"], f"{b} best")


FOLD_SUMMARY_KEYS = ("step", "epoch", "finished", "best_step", "best_metric", "last_train", "last_eval")
