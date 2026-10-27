# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_generative_infra.py
==========================================
The generator training infrastructure (#22): `hsi_augment.train_generative`
(exact resume, weights_only checkpoints, atomic writes, the balanced sampler,
determinism, EMA, early stopping, CPU-generator noise), `hsi_augment.codec`
(log-space round trip, clipping, float16, SAM on decoded NCHW values) and
`hsi_augment.models.blocks`. CPU only, fp32, toy models of a few hundred
parameters; the stage test runs on `tests/hsi_augment_fixtures.make_tiny_build`.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

from hsi_augment import train_generative as tg
from hsi_augment.codec import (FLOAT16_MAX, LogSpaceCodec, make_decoded_sam_loss, sam_deg_decoded)
from hsi_augment.config import ConfigError, param
from hsi_augment.io import RunningBandStats, iter_chunks
from hsi_augment.models.blocks import (Downsample, FiLM, ResBlock, TimestepEmbedder, Upsample,
                                       assert_no_batchnorm, crop_to, group_norm, timestep_embedding)
from hsi_augment.registry import Registry
from hsi_augment.stage import RunOptions, read_json, run_stage
from hsi_augment.train_generative import (BEST_NAME, HISTORY_NAME, LAST_NAME, TIMING_COLUMNS,
                                          ArrayBatchSource, BalancedBatchSampler, EMA, ResumeError,
                                          SequentialBatches, StepInfo, TrainConfig, Trainer, WarmupCosine,
                                          find_weights_only_violations, load_checkpoint, load_weights,
                                          read_history)
from tests.hsi_augment_fixtures import make_tiny_build

BANDS, PATCH, N_CLASSES = 4, 5, 3


# ============================================================================
# toy data, model and step function
# ============================================================================

def toy_arrays(seed: int = 0):
    """96 rows, imbalanced classes (48/12/36) and patients (class 1's patient
    5 has only 2 rows), NHWC float32."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], [48, 12, 36])
    patient = np.concatenate([np.repeat([0, 1, 2, 3], 12), np.repeat([4, 5], [10, 2]),
                              np.repeat([6, 7, 8], [20, 10, 6])])
    X = (rng.normal(size=(96, PATCH, PATCH, BANDS)) + y[:, None, None, None]).astype(np.float32)
    return X, y, patient


class ToyNet(nn.Module):
    """Conv -> FiLM'd ResBlock (class embedding) -> dropout -> conv, plus a
    learned per-band log sigma (a 1-D parameter, as the sigma-VAE's)."""

    def __init__(self, width: int = 8):
        super().__init__()
        self.inp = nn.Conv2d(BANDS, width, 3, padding=1)
        self.emb = nn.Embedding(N_CLASSES, 6)
        self.block = ResBlock(width, width, cond_dim=6, dropout=0.2, zero_init=False)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Conv2d(width, BANDS, 1)
        self.log_sigma = nn.Parameter(torch.zeros(BANDS))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.block(self.inp(x), self.emb(y))))


def toy_step(model: ToyNet, batch: Dict[str, torch.Tensor], info: StepInfo):
    """Random augmentation from all three global RNGs (torch, numpy, Python),
    plus the model's dropout: everything resume has to reproduce."""
    x, y = batch["x"], batch["y"]
    x_in = x
    if info.training:
        flip = torch.rand(x.shape[0], 1, 1, 1) < 0.5
        x_in = torch.where(flip, x.flip(-1), x) + 0.05 * torch.randn_like(x)
        x_in = x_in * float(1.0 + 0.01 * np.random.rand()) + 0.001 * random.random()
    pred = model(x_in, y)
    ls = model.log_sigma.view(1, -1, 1, 1)
    nll = (((pred - x) ** 2) / (2 * torch.exp(2 * ls)) + ls).mean()
    return nll, {"mse": ((pred - x) ** 2).mean().detach(), "nll": nll.detach()}


def toy_sources(X, y, patient, *, batch: int = 9, steps_per_epoch: int = 3, sampler_seed: int = 5):
    held = np.arange(0, 96, 8)                                    # one row in 8, all classes
    train_rows = np.setdiff1d(np.arange(96), held)
    sampler = BalancedBatchSampler(y, patient, batch, seed=sampler_seed, rows=train_rows,
                                   steps_per_epoch=steps_per_epoch)
    train = ArrayBatchSource(X, sampler, extras={"y": y, "patient": patient})
    evals = ArrayBatchSource(X, SequentialBatches(held, 5), extras={"y": y})
    return train, evals


def toy_cfg(**kw) -> TrainConfig:
    base = dict(max_steps=4, lr=3e-3, min_lr=1e-4, warmup_steps=1, weight_decay=1e-2, grad_clip=0.5,
                ema_decay=0.9, eval_every=2, eval_metric="nll", log_every=3, ckpt_every=100)
    base.update(kw)
    return TrainConfig(**base)


def make_trainer(run_dir, cfg: TrainConfig, *, init_seed: int = 0, train_seed: int = 11, resume=False,
                 step_fn=toy_step, with_eval: bool = True, **kw) -> Trainer:
    X, y, patient = toy_arrays()
    train, evals = toy_sources(X, y, patient)
    tg.setup_determinism(init_seed)
    model = ToyNet()
    return Trainer(model, step_fn, train, cfg, eval_data=evals if with_eval else None, run_dir=run_dir,
                   seed=train_seed, resume=resume, **kw)


def assert_same(a: Any, b: Any, path: str = "") -> None:
    """Bitwise equality of nested state (tensors with torch.equal)."""
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


@pytest.fixture
def restore_torch_flags():
    """`deterministic=True` flips process-global torch flags; put them back."""
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(det, warn_only=warn)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


# ============================================================================
# exact resume
# ============================================================================

@pytest.mark.parametrize("interruption", ["pause", "crash"])
def test_resume_two_plus_two_steps_is_bitwise_four_steps(tmp_path, interruption):
    """2 steps + resume + 2 steps == 4 steps, bitwise: weights, EMA, optimizer,
    scheduler, RNG states, early-stopping state and history. The resume point
    (step 2) is inside a logging window (log_every 3) and before an epoch
    boundary (3 batches per epoch), and the resumed model starts from a
    DIFFERENT init, so everything has to come from the checkpoint."""
    cfg = toy_cfg(ckpt_every=2)
    ref = make_trainer(tmp_path / "ref", cfg)
    r_ref = ref.fit()
    assert r_ref.finished and ref.step == 4 and ref.epoch == 1 and ref.step_in_epoch == 1

    run = tmp_path / "run"
    if interruption == "pause":
        first = make_trainer(run, cfg)
        r1 = first.fit(until_step=2)
        assert not r1.finished and r1.step == 2
    else:
        def dies_at_step_3(model, batch, info):
            if info.training and info.step == 2:
                raise RuntimeError("simulated crash in the middle of step 3")
            return toy_step(model, batch, info)
        with pytest.raises(RuntimeError, match="simulated crash"):
            make_trainer(run, cfg, step_fn=dies_at_step_3).fit()
    assert load_checkpoint(run / LAST_NAME)["step"] == 2

    resumed = make_trainer(run, cfg, init_seed=999, resume=True)
    r2 = resumed.fit()
    assert r2.finished and r2.resumed_from == 2 and (resumed.step, resumed.epoch) == (4, 1)

    assert_same(ref.model.state_dict(), resumed.model.state_dict(), "model")
    assert_same(ref.ema.module.state_dict(), resumed.ema.module.state_dict(), "ema")
    assert_same(ref.optimizer.state_dict(), resumed.optimizer.state_dict(), "optimizer")
    assert_same(ref.scheduler.state_dict(), resumed.scheduler.state_dict(), "scheduler")
    a, b = load_checkpoint(tmp_path / "ref" / LAST_NAME), load_checkpoint(run / LAST_NAME)
    for key in ("model", "ema", "optimizer", "scheduler", "rng", "step", "epoch", "step_in_epoch",
                "early_stopping", "best", "window", "finished"):
        assert_same(a[key], b[key], key)
    assert no_timing(a["history"]) == no_timing(b["history"])
    assert no_timing(read_history(tmp_path / "ref" / HISTORY_NAME)) == no_timing(read_history(run / HISTORY_NAME))
    # the history has the window that spanned the resume, and both evaluations
    steps = [r["step"] for r in a["history"]]
    assert steps == [2, 3, 4] and "train_loss" in a["history"][1] and "eval_nll" in a["history"][0]
    # the event log is append-only and tags the attempt
    events = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events if e["event"] in ("start", "resume", "end")] == ["start", "resume", "end"]
    assert {e["attempt"] for e in events if e["event"] == "end"} == {2}


def test_resume_refuses_another_config_or_seed(tmp_path):
    make_trainer(tmp_path, toy_cfg()).fit(until_step=2)
    with pytest.raises(ResumeError, match="lr"):
        make_trainer(tmp_path, toy_cfg(lr=1e-3), resume=True).fit()
    with pytest.raises(ResumeError, match="seed"):
        make_trainer(tmp_path, toy_cfg(), train_seed=12, resume=True).fit()
    with pytest.raises(ResumeError, match="eval data"):
        make_trainer(tmp_path, toy_cfg(), with_eval=False, resume=True).fit()
    # log/ckpt cadence may change; the refused attempts wrote nothing
    assert load_checkpoint(tmp_path / LAST_NAME)["step"] == 2
    r = make_trainer(tmp_path, toy_cfg(log_every=1, ckpt_every=1), resume=True).fit()
    assert r.finished and r.resumed_from == 2
    # resume=False starts over, removing the old run's files
    fresh = make_trainer(tmp_path, toy_cfg(), resume=False)
    fresh.fit(until_step=1)
    assert load_checkpoint(tmp_path / LAST_NAME)["step"] == 1 and not (tmp_path / BEST_NAME).exists()


def test_resuming_a_finished_run_returns_without_training(tmp_path):
    t = make_trainer(tmp_path, toy_cfg())
    r1 = t.fit()
    again = make_trainer(tmp_path, toy_cfg(), resume=True)
    r2 = again.fit()
    assert r2.finished and r2.step == r1.step == 4 and r2.best_step == r1.best_step
    assert_same(t.model.state_dict(), again.model.state_dict())


# ============================================================================
# checkpoints
# ============================================================================

def _walk_types(obj, seen):
    seen.add(type(obj))
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_types(k, seen)
            _walk_types(v, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_types(v, seen)


def test_checkpoints_load_weights_only_with_no_numpy_scalars(tmp_path):
    t = make_trainer(tmp_path, toy_cfg())
    # The root cause of the known --resume crash: a numpy LR in the optimizer
    # and scheduler state. Inject it, as an np.cos-based schedule would.
    t.optimizer.param_groups[0]["lr"] = np.float64(t.optimizer.param_groups[0]["lr"])
    t.scheduler._last_lr = [np.float64(v) for v in t.scheduler._last_lr]
    raw = io.BytesIO()
    torch.save(t.optimizer.state_dict(), raw)
    raw.seek(0)
    with pytest.raises(pickle.UnpicklingError):           # what the old trainer hit
        torch.load(raw, weights_only=True)
    t.fit()
    for name in (BEST_NAME, LAST_NAME):
        ckpt = torch.load(tmp_path / name, map_location="cpu", weights_only=True)
        assert find_weights_only_violations(ckpt) == []
        types: set = set()
        _walk_types(ckpt, types)
        assert not [ty for ty in types if ty.__module__.startswith("numpy")], types
        for g in ckpt["optimizer"]["param_groups"]:
            assert type(g["lr"]) is float
        assert all(type(v) is float for v in ckpt["scheduler"]["_last_lr"])
        assert {"model", "ema", "optimizer", "scheduler", "step", "epoch", "step_in_epoch", "rng"} <= set(ckpt)
        assert set(ckpt["rng"]) == {"python", "numpy", "torch", "cuda"}
    assert load_checkpoint(tmp_path / BEST_NAME)["kind"] == "best"
    # load_weights puts the EMA weights into a fresh model
    m = ToyNet()
    meta = load_weights(tmp_path / LAST_NAME, m)
    assert_same(m.state_dict(), t.ema.module.state_dict())
    assert meta["step"] == 4
    # a value that cannot go into a weights_only checkpoint is caught before writing
    t.history.append({"step": 99, "bad": {1, 2}})
    with pytest.raises(TypeError):
        t.save(LAST_NAME)
    assert load_checkpoint(tmp_path / LAST_NAME)["step"] == 4


def test_atomic_save_crash_leaves_the_previous_checkpoint(tmp_path, monkeypatch):
    t = make_trainer(tmp_path, toy_cfg())
    t.fit(until_step=2)
    before = (tmp_path / LAST_NAME).read_bytes()
    real_save = torch.save

    def dies_mid_write(obj, f, *a, **k):
        f.write(b"\x80\x02 half a pickle")
        raise OSError("disk full / killed mid-write")

    monkeypatch.setattr(torch, "save", dies_mid_write)
    t._train_step()
    with pytest.raises(OSError, match="mid-write"):
        t.save(LAST_NAME)
    monkeypatch.setattr(torch, "save", real_save)
    assert (tmp_path / LAST_NAME).read_bytes() == before
    assert load_checkpoint(tmp_path / LAST_NAME)["step"] == 2
    assert not list(tmp_path.glob(".*.tmp-*"))
    # a temp file left by a SIGKILLed writer is ignored by the loader and removed on start
    (tmp_path / f".{LAST_NAME}.tmp-424242").write_bytes(b"torn")
    r = make_trainer(tmp_path, toy_cfg(), resume=True).fit()
    assert r.resumed_from == 2 and r.finished and not list(tmp_path.glob(".*.tmp-*"))


# ============================================================================
# determinism, EMA, early stopping, overfitting, bf16
# ============================================================================

def test_two_identical_deterministic_runs_give_identical_weights(tmp_path, restore_torch_flags):
    cfg = toy_cfg(max_steps=6, deterministic=True)
    a = make_trainer(tmp_path / "a", cfg)
    a.fit()
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert a.determinism["use_deterministic_algorithms"] == "strict"
    import os
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8")
    b = make_trainer(tmp_path / "b", cfg)
    b.fit()
    assert_same(a.model.state_dict(), b.model.state_dict())
    assert_same(a.ema.module.state_dict(), b.ema.module.state_dict())
    c = make_trainer(tmp_path / "c", cfg, train_seed=12)
    c.fit()
    assert not torch.equal(a.model.out.weight, c.model.out.weight)


def test_ema_update_rule_and_trainer_ema():
    torch.manual_seed(0)
    m = nn.Linear(3, 2)
    ema = EMA(m, 0.9, warmup=False)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    assert torch.allclose(ema.module.weight, 0.9 * w0 + 0.1 * (w0 + 1.0), atol=1e-6)
    assert not ema.module.weight.requires_grad and not ema.module.training
    warm = EMA(m, 0.999, warmup=True)
    assert warm.current_decay() == pytest.approx(0.1)        # (1 + 0) / (10 + 0)
    warm.update(m)
    assert warm.current_decay() == pytest.approx(2 / 11)


def test_trainer_ema_moves_and_lags(tmp_path):
    t = make_trainer(tmp_path, toy_cfg(max_steps=5, ema_warmup=False))
    init = {k: v.clone() for k, v in t.model.state_dict().items()}
    t.fit()
    w, e = t.model.out.weight, t.ema.module.out.weight
    assert not torch.equal(e, init["out.weight"]) and not torch.equal(e, w)
    assert t.ema.num_updates == 5


def test_early_stopping_hook_triggers(tmp_path):
    def worsening_eval(model, batch, info):
        # held-out loss rises every evaluation after the first
        return torch.tensor(float(info.step)), {"nll": torch.tensor(float(info.step))}

    cfg = toy_cfg(max_steps=50, eval_every=2, early_stop_patience=2, ckpt_every=4)
    t = make_trainer(tmp_path, cfg, eval_fn=worsening_eval)
    r = t.fit()
    assert r.stopped_early and r.finished and r.step == 6 and r.best_step == 2 and r.best_metric == 2.0
    assert load_checkpoint(tmp_path / BEST_NAME)["step"] == 2
    last = load_checkpoint(tmp_path / LAST_NAME)
    assert last["step"] == 6 and last["finished"] and last["early_stopping"]["stopped"]
    rows = read_history(tmp_path / HISTORY_NAME)
    assert [(r["step"], r["eval_best"]) for r in rows if r.get("eval_nll") is not None] == \
        [(2, 1), (4, 0), (6, 0)]
    # a held-out metric that does not exist is an error, not a silent "never improves"
    with pytest.raises(KeyError, match="nope"):
        make_trainer(tmp_path / "x", toy_cfg(eval_metric="nope")).fit()


def test_evaluation_does_not_perturb_training_and_is_repeatable(tmp_path):
    with_eval = make_trainer(tmp_path / "a", toy_cfg(max_steps=4, eval_every=1))
    with_eval.fit()
    without = make_trainer(tmp_path / "b", toy_cfg(max_steps=4), with_eval=False)
    without.fit()
    assert_same(with_eval.model.state_dict(), without.model.state_dict())
    assert with_eval.evaluate() == with_eval.evaluate()
    assert (tmp_path / "b" / BEST_NAME).is_file()              # no eval data: best = final


def test_toy_model_overfits_one_batch(tmp_path):
    X, y, patient = toy_arrays()

    class OneBatch:
        def num_batches(self, epoch):
            return 1

        def get_batch(self, epoch, index):
            return {"x": torch.from_numpy(X[:16]).permute(0, 3, 1, 2).contiguous(),
                    "y": torch.from_numpy(y[:16].astype(np.int64))}

    def recon(model, batch):                        # the 2-argument form of step_fn
        loss = ((model(batch["x"], batch["y"]) - batch["x"]) ** 2).mean()
        return loss, {}

    tg.setup_determinism(0)
    model = ToyNet()
    model.drop.p = 0.0
    model.block.dropout = nn.Identity()
    t = Trainer(model, recon, OneBatch(), TrainConfig(max_steps=300, lr=1e-2, min_lr=1e-3, log_every=10,
                                                      ema_decay=None, weight_decay=0.0), run_dir=tmp_path)
    t.fit()
    losses = [r["train_loss"] for r in t.history if "train_loss" in r]
    assert losses[-1] < 0.05 * losses[0], (losses[0], losses[-1])


def test_bf16_autocast_runs_on_cpu(tmp_path):
    r = make_trainer(tmp_path, toy_cfg(max_steps=2, precision="bf16")).fit()
    assert r.finished and math.isfinite(r.last_train["loss"])


def test_train_config_validation_and_schedule():
    with pytest.raises(ConfigError):
        TrainConfig(min_lr=1.0, lr=1e-3)
    with pytest.raises(ConfigError):
        TrainConfig(max_steps=10, warmup_steps=10)
    with pytest.raises(ConfigError):
        TrainConfig(precision="fp16")
    sched = WarmupCosine(max_steps=10, warmup_steps=2, min_ratio=0.01)
    vals = [sched(s) for s in range(12)]
    assert all(type(v) is float for v in vals)
    assert vals[0] == 0.5 and vals[1] == 1.0 and vals[2] == 1.0
    assert all(a >= b for a, b in zip(vals[2:], vals[3:])) and vals[10] == pytest.approx(0.01)
    # a stage config can extend TrainConfig and is still accepted by the Trainer
    @dataclass(frozen=True)
    class FitToyConfig(TrainConfig):
        latent_dim: int = param(64, help="toy")
    assert isinstance(FitToyConfig(max_steps=3), TrainConfig)


# ============================================================================
# sampler, sources, noise
# ============================================================================

def test_balanced_sampler_is_class_then_patient_balanced_and_deterministic():
    X, y, patient = toy_arrays()
    excluded = np.array([0, 50, 95])
    rows = np.setdiff1d(np.arange(96), excluded)
    s = BalancedBatchSampler(y, patient, 10, seed=3, rows=rows, steps_per_epoch=12)
    e0 = s.epoch_rows(0)
    assert e0.shape == (12, 10) and not np.isin(e0, excluded).any()
    per_batch = np.stack([np.bincount(y[b], minlength=3) for b in e0])
    assert set(per_batch.ravel().tolist()) <= {3, 4}          # 10 = 3 + 3 + 3, +1 rotating
    assert (per_batch.sum(0) == 40).all()                     # 12 batches: exact over the epoch
    for c in range(3):
        drawn = patient[e0.ravel()][y[e0.ravel()] == c]
        counts = np.array([np.sum(drawn == p) for p in np.unique(patient[rows][y[rows] == c])])
        assert counts.max() - counts.min() <= 1, (c, counts)
    # patient 5 has 2 rows but gets the same share as patient 4 (10 rows): oversampled
    assert np.sum(patient[e0.ravel()] == 5) >= 19
    # deterministic per (seed, epoch); a new epoch or seed gives a new order
    again = BalancedBatchSampler(y, patient, 10, seed=3, rows=rows, steps_per_epoch=12)
    assert np.array_equal(again.epoch_rows(0), e0) and np.array_equal(again.batch_rows(0, 5), e0[5])
    assert not np.array_equal(s.epoch_rows(1), e0)
    assert not np.array_equal(BalancedBatchSampler(y, patient, 10, seed=4, rows=rows,
                                                   steps_per_epoch=12).epoch_rows(0), e0)
    mask = np.ones(96, dtype=bool)
    mask[excluded] = False
    assert np.array_equal(BalancedBatchSampler(y, patient, 10, seed=3, rows=mask,
                                               steps_per_epoch=12).epoch_rows(0), e0)
    assert s.describe() == again.describe()
    with pytest.raises(ValueError, match="duplicates"):
        BalancedBatchSampler(y, patient, 10, seed=3, rows=np.array([1, 1, 2]))
    with pytest.raises(ValueError, match="batch_size"):
        BalancedBatchSampler(y, patient, 2, seed=3)


def test_array_batch_source_layout_and_sequential_batches():
    X, y, patient = toy_arrays()
    seq = SequentialBatches(np.array([9, 3, 5, 7, 1]), 2)
    assert seq.num_batches() == 3 and seq.batch_rows(0, 2).tolist() == [9]
    src = ArrayBatchSource(X.astype(np.float16), seq, extras={"y": y})
    b = src.get_batch(0, 0)
    assert b["x"].shape == (2, BANDS, PATCH, PATCH) and b["x"].dtype == torch.float32
    assert b["y"].dtype == torch.int64 and b["row"].tolist() == [1, 3]
    assert torch.equal(b["x"][1], torch.from_numpy(X[3].astype(np.float16).astype(np.float32)).permute(2, 0, 1))
    with pytest.raises(TypeError):
        ArrayBatchSource(X, seq, extras={"p": patient.astype(str)})


def test_cpu_generator_noise_is_reproducible_and_batch_size_independent():
    a = tg.noise_rows(7, 0, 10, (3,), block_rows=4)
    b = torch.cat([tg.noise_rows(7, 0, 3, (3,), block_rows=4), tg.noise_rows(7, 3, 7, (3,), block_rows=4)])
    assert torch.equal(a, b) and a.shape == (10, 3)
    assert not torch.equal(a, tg.noise_rows(8, 0, 10, (3,), block_rows=4))
    g1, g2 = tg.cpu_generator(5, "sampling"), tg.cpu_generator(5, "sampling")
    assert torch.equal(tg.randn_cpu((4, 2), g1), tg.randn_cpu((4, 2), g2))
    assert tg.derive_seed(1, "a") != tg.derive_seed(1, "b") and tg.derive_seed(1, "a") < 2 ** 63


# ============================================================================
# codec
# ============================================================================

def _codec_data(seed=0, n=40):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(mean=0.0, sigma=0.5, size=(n, 3, 3, 8)) * np.linspace(0.5, 2.0, 8)
    x[0, 0, 0, :] = 0.0                                      # dark pixels: log stays finite via eps
    return x.astype(np.float16)


def test_codec_fit_stats_matches_float64_numpy_and_round_trips():
    x = _codec_data()
    stats = LogSpaceCodec.fit_stats(iter_chunks(x, chunk_rows=7))
    ref = np.log(x.astype(np.float64).reshape(-1, 8) + 1e-3)
    assert isinstance(stats, RunningBandStats)
    np.testing.assert_allclose(stats.mean, ref.mean(0), rtol=1e-12)
    np.testing.assert_allclose(stats.std(), ref.std(0), rtol=1e-10)
    clip_hi = LogSpaceCodec.clip_hi_from_quantile(np.quantile(x.astype(np.float64).reshape(-1, 8), 0.9999, axis=0))
    codec = LogSpaceCodec.from_stats(stats, clip_hi)
    y = codec.encode_np(x)
    assert y.dtype == np.float32 and abs(float(y.mean())) < 1e-4 and abs(float(y.std()) - 1) < 1e-3
    back = codec.decode_np(y)
    assert back.dtype == np.float16
    np.testing.assert_allclose(back.astype(np.float32), x.astype(np.float32), rtol=2e-3, atol=2e-3)
    # JSON round trip of the codec, and of the stats the pool stage will save
    c2 = LogSpaceCodec.from_dict(json.loads(json.dumps(codec.to_dict())))
    assert np.array_equal(c2.encode_np(x), y)
    c3 = LogSpaceCodec.from_stats(json.loads(json.dumps(stats.to_dict())), clip_hi)
    assert np.array_equal(c3.mu, codec.mu) and np.array_equal(c3.sigma, codec.sigma)


def test_codec_decode_is_clipped_non_negative_and_float16_finite():
    codec = LogSpaceCodec(mu=np.zeros(8), sigma=np.full(8, 2.0), clip_hi=np.linspace(1.0, 8.0, 8))
    y = np.array([-1e4, -50.0, -np.inf, 0.0, 3.0, 50.0, 1e4, np.inf], dtype=np.float32)
    y = np.broadcast_to(y[:, None], (8, 8)).reshape(8, 1, 1, 8)
    x = codec.decode_np(y)
    assert x.dtype == np.float16 and np.isfinite(x).all() and (x >= 0).all()
    assert (x.astype(np.float32) <= codec.clip_hi.astype(np.float16).astype(np.float32)).all()
    assert (x[0] == 0).all() and np.allclose(x[-1].ravel(), codec.clip_hi, rtol=1e-3)
    with pytest.raises(ValueError, match="NaN"):
        codec.decode_np(np.full((1, 8), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="float16"):
        LogSpaceCodec(mu=np.zeros(2), sigma=np.ones(2), clip_hi=[1.0, 2 * FLOAT16_MAX])
    with pytest.raises(ValueError, match="sigma"):
        LogSpaceCodec(mu=np.zeros(2), sigma=[1.0, 0.0], clip_hi=1.0)
    # negative linear inputs are treated as 0
    assert np.array_equal(codec.encode_np(np.full((1, 8), -0.5)), codec.encode_np(np.zeros((1, 8))))


def test_codec_torch_paths_match_numpy_in_either_layout():
    x = _codec_data(1)
    codec = LogSpaceCodec.from_stats(LogSpaceCodec.fit_stats([x]), 10.0)
    y_nhwc = codec.encode_np(x)
    y_nchw = torch.from_numpy(y_nhwc).permute(0, 3, 1, 2).contiguous()
    np.testing.assert_allclose(codec.encode_torch(torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2)),
                               y_nchw.numpy(), rtol=1e-5, atol=1e-5)
    dec_t = codec.decode_torch(y_nchw)
    assert dec_t.dtype == torch.float32 and (dec_t >= 0).all()
    np.testing.assert_allclose(dec_t.permute(0, 2, 3, 1).numpy().astype(np.float16), codec.decode_np(y_nhwc))
    assert codec.decode_torch(y_nchw, dtype=torch.float16).dtype == torch.float16
    np.testing.assert_allclose(codec.decode_torch(torch.from_numpy(y_nhwc), channel_dim=-1).numpy(),
                               dec_t.permute(0, 2, 3, 1).numpy(), rtol=1e-6)
    assert np.array_equal(codec.encode_np(np.moveaxis(x, -1, 1), band_axis=1), np.moveaxis(y_nhwc, -1, 1))
    with pytest.raises(ValueError, match="band"):
        codec.decode_torch(torch.from_numpy(y_nhwc))           # NHWC passed as NCHW: caught (C=3 != 8)


def test_sam_on_decoded_nchw_values():
    """sam_deg and the frozen SAMLoss both take NCHW and compare the spectra
    along dim 1 of the DECODED values."""
    x = _codec_data(2, n=6)
    codec = LogSpaceCodec.from_stats(LogSpaceCodec.fit_stats([x]), 10.0)
    y_true = torch.from_numpy(codec.encode_np(x)).permute(0, 3, 1, 2).contiguous()
    g = torch.Generator().manual_seed(0)
    y_pred = (y_true + 0.3 * torch.randn(y_true.shape, generator=g)).requires_grad_(True)
    a = codec.decode_torch(y_true).permute(0, 2, 3, 1).reshape(-1, 8).double().numpy()
    b = codec.decode_torch(y_pred.detach()).permute(0, 2, 3, 1).reshape(-1, 8).double().numpy()
    # per-vector norm floor 1e-6, as training.gan.safe_cosine_similarity (x[0, 0, 0] is all zeros)
    cos = (a * b).sum(1) / (np.maximum(np.linalg.norm(a, axis=1), 1e-6) *
                            np.maximum(np.linalg.norm(b, axis=1), 1e-6))
    manual = np.degrees(np.arccos(np.clip(cos, -1 + 1e-3, 1 - 1e-3))).mean()
    assert sam_deg_decoded(codec, y_true, y_pred) == pytest.approx(manual, rel=1e-4)
    loss = make_decoded_sam_loss(codec)(y_true, y_pred)
    assert float(loss.detach()) == pytest.approx(math.radians(manual), rel=1e-4)
    loss.backward()
    assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all() and y_pred.grad.abs().sum() > 0
    # Property of the IMPORTED metrics, pinned here because #23/#25 depend on it: both clamp
    # cos at 1 - 1e-3 (SAMLoss's default angle_eps), so identical spectra score acos(0.999)
    # = 2.56 deg, not 0, and SAMLoss passes no gradient to pixels already within 2.56 deg.
    floor = math.degrees(math.acos(1 - 1e-3))
    assert sam_deg_decoded(codec, y_true[1:], y_true[1:]) == pytest.approx(floor, rel=1e-3)
    near = y_true.clone().requires_grad_(True)
    make_decoded_sam_loss(codec)(y_true[1:], near[1:]).backward()
    assert near.grad.abs().sum() == 0
    tight = y_true.clone().requires_grad_(True)
    make_decoded_sam_loss(codec, angle_eps=1e-7)(y_true[1:], (tight * 1.01)[1:]).backward()
    assert tight.grad.abs().sum() > 0                 # a smaller angle_eps restores the gradient


# ============================================================================
# blocks
# ============================================================================

def test_blocks_shapes_film_identity_and_no_batchnorm():
    torch.manual_seed(0)
    assert group_norm(96).num_groups == 32 and group_norm(4).num_groups == 4 and group_norm(6).num_groups == 6
    for size in (11, 6, 3):
        blk = ResBlock(8, 16, cond_dim=5)
        out = blk(torch.randn(2, 8, size, size), torch.randn(2, 5))
        assert out.shape == (2, 16, size, size)
    x = torch.randn(2, 8, 11, 11)
    h = Downsample(8)(x)
    assert h.shape[-1] == 6 and Downsample(8)(h).shape[-1] == 3
    up = Upsample(8)(Upsample(8)(torch.randn(2, 8, 3, 3)))
    assert up.shape[-1] == 12 and crop_to(up, 11).shape[-2:] == (11, 11)
    film = FiLM(5, 8)
    assert torch.equal(film(x, torch.randn(2, 5)), x)               # zero-init = identity
    blk = ResBlock(8, 8)
    assert torch.equal(blk(x), x)                                   # zero-init conv2 = identity
    with pytest.raises(ValueError):
        ResBlock(8, cond_dim=5)(x)
    emb = timestep_embedding(torch.tensor([0, 10, 999]), 16)
    assert emb.shape == (3, 16) and torch.equal(emb[0, :8], torch.ones(8)) and torch.equal(emb[0, 8:], torch.zeros(8))
    assert timestep_embedding(torch.tensor([1.5]), 7).shape == (1, 7)
    assert TimestepEmbedder(32)(torch.tensor([1, 2])).shape == (2, 32)
    assert_no_batchnorm(nn.Sequential(ResBlock(8, 16, cond_dim=4), Downsample(16), ToyNet()))
    with pytest.raises(TypeError, match="BatchNorm"):
        assert_no_batchnorm(nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)))


# ============================================================================
# inside a stage: ctx.stage_dir, ctx.log_metrics, --resume auto
# ============================================================================

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@dataclass(frozen=True)
class ToyFitConfig(TrainConfig):
    width: int = param(8, help="toy model width")


def _toy_fit_registry(crash: Dict[str, bool]) -> Registry:
    reg = Registry()

    @reg.stage("toy-fit", config=ToyFitConfig, dirname="gen/toy", purposes=("generator",))
    def toy_fit(ctx, cfg):
        seed = ctx.seed_int("train")
        X, y, patient = toy_arrays()
        train, evals = toy_sources(X, y, patient, sampler_seed=ctx.seed_int("sampler"))
        tg.setup_determinism(seed)
        model = ToyNet(cfg.width)

        def step(model, batch, info):
            if crash["on"] and not ctx.resume and info.training and info.step == 2:
                raise RuntimeError("first attempt killed after its step-2 checkpoint")
            return toy_step(model, batch, info)

        trainer = Trainer(model, step, train, cfg, eval_data=evals, seed=seed, ctx=ctx)
        return trainer.fit().summary()

    return reg


def test_trainer_inside_a_stage_resumes_with_resume_auto(tiny, tmp_path):
    cfg = ToyFitConfig(max_steps=4, eval_every=2, eval_metric="nll", log_every=3, ckpt_every=2,
                       warmup_steps=1, ema_decay=0.9)
    crash = {"on": False}
    reg = _toy_fit_registry(crash)
    opts = lambda out, **kw: RunOptions(data_dir=str(tiny.data_dir), out=str(out), **kw)

    ref = run_stage(reg.get("toy-fit"), cfg, opts(tmp_path / "ref"), reg)
    assert ref.status == "ran" and ref.done["summary"]["finished"]

    crash["on"] = True
    with pytest.raises(RuntimeError, match="killed"):
        run_stage(reg.get("toy-fit"), cfg, opts(tmp_path / "run"), reg)
    stage_dir = tmp_path / "run" / "gen" / "toy"
    assert not (stage_dir / "done.json").exists() and load_checkpoint(stage_dir / LAST_NAME)["step"] == 2

    r = run_stage(reg.get("toy-fit"), cfg, opts(tmp_path / "run", resume="auto"), reg)
    assert r.done["resumed"] and r.done["attempt"] == 2 and r.done["summary"]["resumed_from"] == 2
    assert {BEST_NAME, LAST_NAME, HISTORY_NAME} <= set(r.done["outputs"])
    a = load_checkpoint(tmp_path / "ref" / "gen" / "toy" / LAST_NAME)
    b = load_checkpoint(stage_dir / LAST_NAME)
    for key in ("model", "ema", "optimizer", "rng", "history"):
        assert_same(no_timing(a[key]) if key == "history" else a[key],
                    no_timing(b[key]) if key == "history" else b[key], key)
    events = [json.loads(l) for l in (stage_dir / "metrics.jsonl").read_text().splitlines()]
    assert [(e["event"], e["attempt"]) for e in events if e["event"] in ("start", "resume", "end")] == \
        [("start", 1), ("resume", 2), ("end", 2)]
    assert all("elapsed_s" in e for e in events)                     # written via ctx.log_metrics
    done = read_json(stage_dir / "done.json")
    assert done["outputs"][LAST_NAME]["size_bytes"] == (stage_dir / LAST_NAME).stat().st_size
