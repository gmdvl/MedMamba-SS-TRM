# -*- coding: utf-8 -*-
"""
hsi_augment/train_generative.py
===============================
The one training loop behind every generator (Plan §Phase 6 "Training" and
"Checkpoints", §Phase 7): the CVAE (#23), the autoencoder and the latent
diffusion denoiser (#24) differ only in their model and their `step_fn`.

    trainer = Trainer(model, step_fn, train_source, TrainConfig(max_steps=40_000),
                      eval_data=held_out_source, seed=seed, ctx=ctx)
    result = trainer.fit()          # resumes from last.pt when ctx.resume is True

Contracts
---------
* `step_fn(model, batch[, info]) -> (loss, metrics)`. `loss` is a scalar
  tensor with grad; `metrics` maps names to scalar tensors or numbers (logged,
  never backpropagated). If the function takes a third positional argument it
  receives a `StepInfo` (step, epoch, max_steps, training, progress): the
  CVAE's beta warm-up needs the step, a diffusion loss does not. The same
  function (or `eval_fn`) is called with `info.training == False`, under
  `torch.no_grad()` and `model.eval()`, for the held-out loss.
  Everything trainable lives in `model` (wrap model + loss parameters such as
  the sigma-VAE's per-band log sigma in one `nn.Module`). Randomness inside
  `step_fn` (dropout, noise, timesteps, label dropout, augmentation) must come
  from torch's, numpy's or Python's GLOBAL generators: those are checkpointed.
* A data source has `num_batches(epoch) -> int` and `get_batch(epoch, i)`,
  and the batch must be a pure function of (epoch, i). That is what makes
  resume exact without replaying anything: a resumed run asks for batch
  (epoch, i) directly. An optional `describe() -> dict` is stored in the
  checkpoint and compared on resume. `ArrayBatchSource` over a
  `BalancedBatchSampler` (training) or `SequentialBatches` (held-out) covers
  the pool; anything with the two methods works.

Why each piece exists
---------------------
* Class-balanced, then patient-balanced batches (`BalancedBatchSampler`).
  The train split has 30 / 5 / 34 patients per class, and per-class row counts
  13x apart. Uniform rows would teach the generators IDC and the largest
  patients; each batch instead holds B / K rows per class (the remainder
  rotates over the classes), and within a class every patient gets the same
  share, drawing its rows without replacement until they run out. The order is
  `np.random.default_rng([seed, epoch])`, so any epoch can be rebuilt.
* AdamW + warm-up + cosine (`WarmupCosine`, a `LambdaLR` multiplier) and
  gradient clipping, as the plan specifies. The multiplier uses `math.cos`
  and returns a Python float: `training/train_pipeline.py`'s `np.cos`
  version put an np.float64 into `param_groups[*]['lr']` and the scheduler's
  `_last_lr`, and `torch.load(weights_only=True)` then refused the checkpoint
  (the known --resume crash). The optimizer and scheduler state are ALSO run
  through `to_plain`, and every checkpoint is checked by
  `find_weights_only_violations` before it is written.
* EMA of the weights (`EMA`), optionally with the (1 + n) / (10 + n) decay
  cap, so a short final refit is not dominated by its random init. The
  held-out loss is evaluated on the EMA weights by default, because those are
  the weights that are exported.
* bf16 autocast is opt-in (`precision="bf16"`, the LDM). No GradScaler:
  bf16 has float32's exponent range.
* Early stopping on a held-out metric every `eval_every` steps
  (`EarlyStopping`); the best evaluation is `best.pt`. The fold-wise best
  steps set the final refit's step budget (Plan §Phase 6 "Final model").
* Exact resume. `last.pt` and `best.pt` each hold the model, EMA,
  optimizer, scheduler, step, epoch, step within the epoch, every RNG state
  (Python, numpy, torch CPU, CUDA), the early-stopping state, the partial
  logging window and the history so far. A resumed run restores all of it and
  continues at batch (epoch, step_in_epoch), so 2 steps + resume + 2 steps is
  bitwise identical to 4 steps. Evaluation runs inside `torch.random.fork_rng`
  with its own fixed seed: it never perturbs the training stream, and every
  evaluation sees the same noise (comparable held-out losses for a diffusion
  model, whose loss draws t and noise).
* Atomic checkpoints (`atomic_torch_save`): temp file in the same directory,
  fsync, `os.replace`. A crash mid-write leaves the previous file intact.
* `history.csv` (one row per logged step; train and eval columns) is
  rewritten together with every `last.pt`, so it always matches the
  checkpoint a resume would start from; the history inside the checkpoint is
  the source of truth. `metrics.jsonl` is the append-only event log
  (train / eval / start / resume / end): through `ctx.log_metrics` into the
  stage's `metrics.jsonl` when a `StageContext` is given, else
  `<run_dir>/metrics.jsonl`. Records carry `attempt`, so rows a dead attempt
  wrote past its last checkpoint can be told apart.
* `--deterministic` (`TrainConfig.deterministic`): the frozen
  `training.dataloader_config.seed_everything` (imported, never copied) sets
  cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8 and warn-only
  deterministic algorithms; `setup_determinism` then makes it strict
  (`torch.use_deterministic_algorithms(True)`) unless
  `deterministic_warn_only`. Both are process-global.
* Generation noise from a CPU `torch.Generator` (`cpu_generator`,
  `noise_rows`): the same seed gives the same samples on any device, and
  `noise_rows` gives row r the same noise whatever the export's batch size.

Inside a stage
--------------
`ctx` supplies the run directory (`ctx.stage_dir` by default; a `run_dir`
such as `ctx.stage_dir / "fold0"` must stay inside it so its files can be
registered as outputs), the resume decision (`ctx.resume`:
True only under `--resume auto` after an unfinished attempt with the same
fingerprint), the event log and, once training finished, the outputs
(`best.pt`, `last.pt`, `history.csv` are registered with `ctx.add_output`).
Without a ctx (unit tests, scripts) pass `run_dir` and `resume`.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import csv
import dataclasses
import hashlib
import inspect
import json
import logging
import math
import os
import random
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple,
                    Union, runtime_checkable)

import numpy as np
import torch
import torch.nn as nn

from hsi_augment.config import ConfigError, StageConfig, param, to_jsonable

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.stage import StageContext

PathLike = Union[str, os.PathLike]

CHECKPOINT_FORMAT = "hsi_augment.train_generative/1"
BEST_NAME = "best.pt"
LAST_NAME = "last.pt"
HISTORY_NAME = "history.csv"
METRICS_NAME = "metrics.jsonl"
# history.csv columns that measure wall time, so differ between otherwise identical runs
TIMING_COLUMNS = ("wall_s", "steps_per_s")
# Config fields a resume may change: they alter what is logged/saved when, not the weights.
RESUME_FREE_FIELDS = ("log_every", "ckpt_every")
NOISE_BLOCK_ROWS = 1024

logger = logging.getLogger("hsi_augment.train_generative")


class ResumeError(RuntimeError):
    """The checkpoint does not belong to this run (config, seed or data differ)."""


class NonFiniteLossError(FloatingPointError):
    """A training or held-out quantity became NaN/Inf; nothing was checkpointed from it."""


# ============================================================================
# configuration
# ============================================================================

@dataclass(frozen=True)
class TrainConfig(StageConfig):
    """The loop's settings. A fit stage can subclass this (its own fields
    added), get every field as a flag, and pass its config straight to
    `Trainer`; resume then also refuses a change to the stage's own fields.
    Not registered as a stage itself. Data settings (batch size, sampler
    seed) belong to the data source, the model's to the model."""
    max_steps: int = param(1000, help="optimizer steps; the cosine schedule spans exactly these")
    lr: float = param(1e-3, help="AdamW peak learning rate")
    min_lr: float = param(1e-5, help="cosine floor (reached at max_steps)")
    warmup_steps: int = param(0, help="linear warm-up steps before the cosine")
    weight_decay: float = param(1e-4, help="AdamW weight decay on weight matrices/kernels "
                                           "(biases, norm scales and other 1-D params get none)")
    betas: Tuple[float, float] = param((0.9, 0.999), help="AdamW betas")
    adam_eps: float = param(1e-8, help="AdamW eps")
    grad_clip: Optional[float] = param(1.0, help="max global gradient norm (none = no clipping)")
    ema_decay: Optional[float] = param(0.999, help="EMA decay of the weights (none = no EMA)")
    ema_warmup: bool = param(True, help="cap the EMA decay at (1+n)/(10+n) for the first updates")
    precision: str = param("fp32", choices=("fp32", "bf16"), help="bf16 = autocast (opt-in)")
    eval_every: int = param(500, help="held-out evaluation every N steps (and at the end)")
    eval_metric: str = param("loss", help="held-out metric that picks best.pt and drives early "
                                          "stopping (a key of step_fn's metrics, or 'loss')")
    eval_with_ema: bool = param(True, help="evaluate the EMA weights when there is an EMA")
    early_stop_patience: Optional[int] = param(None, help="stop after N evaluations without "
                                                          "improvement (none = never)")
    early_stop_min_delta: float = param(0.0, help="improvement that counts as one")
    log_every: int = param(50, help="steps per history.csv row / metrics.jsonl train record")
    ckpt_every: int = param(500, help="last.pt every N steps (also at every evaluation and the end)")
    deterministic: bool = param(False, help="torch.use_deterministic_algorithms(True) and "
                                            "CUBLAS_WORKSPACE_CONFIG=:4096:8 (process-global)")
    deterministic_warn_only: bool = param(False, help="with deterministic: warn instead of raise "
                                                      "on an op without a deterministic kernel")

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ConfigError("max_steps must be > 0")
        if not self.lr > 0 or not 0 <= self.min_lr <= self.lr:
            raise ConfigError("need lr > 0 and 0 <= min_lr <= lr")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ConfigError("warmup_steps must be in [0, max_steps)")
        if self.weight_decay < 0 or self.adam_eps <= 0:
            raise ConfigError("weight_decay must be >= 0 and adam_eps > 0")
        if not all(0 <= b < 1 for b in self.betas):
            raise ConfigError("betas must be in [0, 1)")
        if self.grad_clip is not None and not self.grad_clip > 0:
            raise ConfigError("grad_clip must be > 0 (or none)")
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ConfigError("ema_decay must be in [0, 1) (or none)")
        for name in ("eval_every", "log_every", "ckpt_every"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be > 0")
        if self.early_stop_patience is not None and self.early_stop_patience < 1:
            raise ConfigError("early_stop_patience must be >= 1 (or none)")
        if self.early_stop_min_delta < 0:
            raise ConfigError("early_stop_min_delta must be >= 0")
        if not self.eval_metric:
            raise ConfigError("eval_metric must be a metric name")


# ============================================================================
# seeds, determinism and generation noise
# ============================================================================

def derive_seed(seed: int, *keys: Union[int, str]) -> int:
    """A 63-bit int from `SeedSequence([seed, *keys])` (string keys are
    crc32-hashed, as in `stage.stage_seed_sequence`). Valid for
    `torch.manual_seed`, `random.seed` and `seed_everything`."""
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    ent = [int(seed)] + [zlib.crc32(k.encode("utf-8")) if isinstance(k, str) else int(k) for k in keys]
    return int(np.random.SeedSequence(ent).generate_state(1, np.uint64)[0]) >> 1


def setup_determinism(seed: int, *, deterministic: bool = False, warn_only: bool = False) -> Dict[str, Any]:
    """Seed Python, numpy, torch and CUDA with the frozen
    `training.dataloader_config.seed_everything`; with `deterministic`, also
    make `torch.use_deterministic_algorithms` strict (unless `warn_only`).
    Call it BEFORE building the model, so the init is reproducible too; the
    flags and CUBLAS_WORKSPACE_CONFIG must be set before the first cuBLAS call.
    Returns the state for a config/summary record."""
    from training.dataloader_config import seed_everything  # frozen: imported, not copied
    state = seed_everything(int(seed), deterministic=bool(deterministic))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # seed_everything does too
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        state["use_deterministic_algorithms"] = "warn_only" if warn_only else "strict"
        state["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return state


def cpu_generator(seed: int, *keys: Union[int, str]) -> torch.Generator:
    """A CPU `torch.Generator` seeded with `derive_seed(seed, *keys)`. Draw
    generation noise from it on the CPU and move the result to the device:
    CUDA generators give different numbers, CPU ones are the same everywhere."""
    return torch.Generator(device="cpu").manual_seed(derive_seed(seed, *keys))


def randn_cpu(shape: Sequence[int], generator: torch.Generator, *, device: Any = None,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """`torch.randn(shape)` drawn on the CPU from `generator`, then moved."""
    x = torch.randn(tuple(int(s) for s in shape), generator=generator, dtype=torch.float32)
    return x.to(device=device or "cpu", dtype=dtype)


def noise_rows(seed: int, start: int, n: int, row_shape: Sequence[int], *,
               block_rows: int = NOISE_BLOCK_ROWS, device: Any = None,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Rows [start, start + n) of an endless, reproducible N(0, 1) stream:
    row r comes from block r // block_rows, drawn with
    `cpu_generator(seed, "noise", block)`. An export therefore gets the same
    noise for row r whatever batch size it generates with."""
    if n <= 0:
        return torch.empty((0, *row_shape), dtype=dtype, device=device or "cpu")
    if start < 0 or block_rows <= 0:
        raise ValueError("start must be >= 0 and block_rows > 0")
    parts = []
    for b in range(start // block_rows, (start + n - 1) // block_rows + 1):
        block = randn_cpu((block_rows, *row_shape), cpu_generator(seed, "noise", b))
        lo = max(start, b * block_rows) - b * block_rows
        hi = min(start + n, (b + 1) * block_rows) - b * block_rows
        parts.append(block[lo:hi])
    return torch.cat(parts).to(device=device or "cpu", dtype=dtype)


# ============================================================================
# sampling and data sources
# ============================================================================

def _rows_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()


class BalancedBatchSampler:
    """Class-balanced, then patient-balanced batches of pool rows.

        sampler = BalancedBatchSampler(pool_y, pool_patient, 512, seed=s, rows=fold_train_rows)
        sampler.batch_rows(epoch, i)       # (512,) int64 pool rows of batch i
        sampler.epoch_rows(epoch)          # (steps_per_epoch, 512)

    Per batch: B // K rows of each of the K classes; the B % K remaining
    slots rotate over the classes, (s * extra + j) % K, so over any K
    consecutive batches every class gets exactly the same number. Within a
    class, the patients are drawn in shuffled rounds (each round visits every
    patient once), so over an epoch any two patients of a class differ by at
    most one row; a patient's rows are drawn without replacement and
    reshuffled when they run out (small patients are oversampled, large ones
    subsampled). Rows are shuffled within each batch. All of it comes from
    `np.random.default_rng([seed, epoch])` with a fixed call order, so an epoch
    is rebuilt identically on resume.

    `rows`: the eligible pool rows (a LOPO fold's train mask, as indices or a
    boolean mask); default all. Returned values are POOL rows, not positions.
    `steps_per_epoch`: default len(rows) // batch_size (at least 1)."""

    def __init__(self, y: np.ndarray, patient: np.ndarray, batch_size: int, *, seed: int,
                 steps_per_epoch: Optional[int] = None, rows: Optional[np.ndarray] = None,
                 classes: Optional[Sequence[Any]] = None):
        y = np.asarray(y).reshape(-1)
        patient = np.asarray(patient).reshape(-1)
        if y.shape != patient.shape:
            raise ValueError(f"y has {y.shape[0]} rows, patient {patient.shape[0]}")
        if rows is None:
            rows = np.arange(y.shape[0], dtype=np.int64)
        else:
            rows = np.asarray(rows)
            if rows.dtype == bool:
                if rows.shape != y.shape:
                    raise ValueError("boolean rows mask must have one entry per pool row")
                rows = np.flatnonzero(rows)
            rows = np.sort(rows.astype(np.int64, copy=False))
            if rows.size and (rows[0] < 0 or rows[-1] >= y.shape[0]):
                raise IndexError("rows out of range")
            if rows.size != np.unique(rows).size:
                raise ValueError("rows contain duplicates (that would double their weight)")
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("seed must be a non-negative int")
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.rows = rows
        yr, pr = y[rows], patient[rows]
        present = np.unique(yr)
        self.classes = list(present.tolist() if classes is None else classes)
        missing = [c for c in self.classes if not np.any(yr == c)]
        if missing:
            raise ValueError(f"no eligible rows for class(es) {missing}")
        if self.batch_size < len(self.classes):
            raise ValueError(f"batch_size {self.batch_size} < {len(self.classes)} classes")
        # per class: [(patient id, its pool rows)], patients sorted -> fixed rng call order
        self._groups: List[List[Tuple[Any, np.ndarray]]] = []
        for c in self.classes:
            in_c = yr == c
            pids = np.unique(pr[in_c])
            self._groups.append([(p, rows[in_c & (pr == p)]) for p in pids.tolist()])
        n_eligible = int(sum(len(r) for g in self._groups for _, r in g))
        self.steps_per_epoch = int(steps_per_epoch or max(1, n_eligible // self.batch_size))
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        self._cache: Tuple[Optional[int], Optional[np.ndarray]] = (None, None)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def num_batches(self, epoch: int = 0) -> int:
        return self.steps_per_epoch

    def patients_per_class(self) -> Dict[Any, int]:
        return {c: len(g) for c, g in zip(self.classes, self._groups)}

    def class_quotas(self) -> np.ndarray:
        """(steps_per_epoch, K) rows per class in each batch."""
        S, K = self.steps_per_epoch, len(self.classes)
        base, extra = divmod(self.batch_size, K)
        quota = np.full((S, K), base, dtype=np.int64)
        if extra:
            s = np.repeat(np.arange(S), extra)
            cls = (s * extra + np.tile(np.arange(extra), S)) % K
            np.add.at(quota, (s, cls), 1)
        return quota

    def epoch_rows(self, epoch: int) -> np.ndarray:
        if self._cache[0] == epoch:
            return self._cache[1]
        rng = np.random.default_rng([self.seed, int(epoch)])
        S, B = self.steps_per_epoch, self.batch_size
        quota = self.class_quotas()
        out = np.empty((S, B), dtype=np.int64)
        cursor = np.zeros(S, dtype=np.int64)
        for k, group in enumerate(self._groups):
            need = int(quota[:, k].sum())
            P = len(group)
            rounds = -(-need // P)
            pat_seq = rng.permuted(np.tile(np.arange(P), (rounds, 1)), axis=1).reshape(-1)[:need]
            seq = np.empty(need, dtype=np.int64)
            for p, (_, prow) in enumerate(group):
                where = np.flatnonzero(pat_seq == p)
                if where.size == 0:
                    continue
                reps = -(-where.size // prow.size)
                seq[where] = rng.permuted(np.tile(prow, (reps, 1)), axis=1).reshape(-1)[:where.size]
            ends = np.cumsum(quota[:, k])
            for s in range(S):
                q = int(quota[s, k])
                out[s, cursor[s]:cursor[s] + q] = seq[ends[s] - q:ends[s]]
                cursor[s] += q
        out = rng.permuted(out, axis=1)
        self._cache = (int(epoch), out)
        return out

    def batch_rows(self, epoch: int, index: int) -> np.ndarray:
        if not 0 <= index < self.steps_per_epoch:
            raise IndexError(f"batch {index} of an epoch with {self.steps_per_epoch}")
        return self.epoch_rows(epoch)[index]

    def describe(self) -> Dict[str, Any]:
        return {"kind": "balanced", "seed": self.seed, "batch_size": self.batch_size,
                "steps_per_epoch": self.steps_per_epoch, "classes": to_jsonable(self.classes),
                "patients_per_class": [len(g) for g in self._groups], "n_rows": int(self.rows.size),
                "rows_sha256": _rows_sha256(self.rows)}


class SequentialBatches:
    """Fixed-order batches over `rows` (sorted), the last one partial; the
    same for every epoch. For held-out evaluation and encoding passes."""

    def __init__(self, rows: np.ndarray, batch_size: int):
        rows = np.asarray(rows)
        self.rows = np.flatnonzero(rows) if rows.dtype == bool else np.sort(rows.astype(np.int64))
        self.batch_size = int(batch_size)
        if self.batch_size <= 0 or self.rows.size == 0:
            raise ValueError("need batch_size > 0 and at least one row")

    def num_batches(self, epoch: int = 0) -> int:
        return -(-int(self.rows.size) // self.batch_size)

    def batch_rows(self, epoch: int, index: int) -> np.ndarray:
        if not 0 <= index < self.num_batches():
            raise IndexError(f"batch {index} of {self.num_batches()}")
        return self.rows[index * self.batch_size:(index + 1) * self.batch_size]

    def describe(self) -> Dict[str, Any]:
        return {"kind": "sequential", "batch_size": self.batch_size, "n_rows": int(self.rows.size),
                "rows_sha256": _rows_sha256(self.rows)}


@runtime_checkable
class BatchSource(Protocol):
    """What `Trainer` reads batches from. `get_batch(epoch, i)` must depend on
    (epoch, i) only. Optional: `describe() -> dict` (compared on resume)."""

    def num_batches(self, epoch: int) -> int: ...

    def get_batch(self, epoch: int, index: int) -> Any: ...


LAYOUTS = ("NHWC", "NCHW", "none")


class ArrayBatchSource:
    """Batches gathered from in-RAM arrays (the pool, or AE latents) by a row
    plan (`BalancedBatchSampler`, `SequentialBatches`, or anything with
    `num_batches(epoch)` and `batch_rows(epoch, i)`).

        source = ArrayBatchSource(X_pool, sampler, extras={"y": pool_y})
        source.get_batch(epoch, i) -> {"x": float32 (B, C, H, W), "row": int64 (B,),
                                       "y": int64 (B,)}

    `layout="NHWC"` (the pool, channel-last like the build) is permuted to
    NCHW; "NCHW" (latents) and "none" (flat vectors) are passed as they are.
    Integer extras become int64 tensors, float extras float32. `transform`
    (batch dict -> batch dict) runs last; if it is random it must use the
    global torch/numpy/Python generators, which the trainer checkpoints."""

    def __init__(self, X: np.ndarray, plan: Any, *, extras: Optional[Mapping[str, np.ndarray]] = None,
                 layout: str = "NHWC", dtype: torch.dtype = torch.float32,
                 transform: Optional[Callable[[Dict[str, torch.Tensor]], Any]] = None):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if layout != "none" and np.ndim(X) != 4:
            raise ValueError(f"layout {layout} needs a 4-D X, got shape {np.shape(X)}")
        self.X, self.plan, self.layout, self.dtype, self.transform = X, plan, layout, dtype, transform
        self.extras: Dict[str, np.ndarray] = {}
        for k, a in (extras or {}).items():
            a = np.asarray(a) if not isinstance(a, np.ndarray) else a
            if k in ("x", "row"):
                raise ValueError(f"extra name {k!r} is reserved")
            if a.shape[0] != X.shape[0]:
                raise ValueError(f"extra {k!r} has {a.shape[0]} rows, X has {X.shape[0]}")
            if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
                raise TypeError(f"extra {k!r} must be numeric (got {a.dtype}); map ids to ints first")
            self.extras[k] = a

    def num_batches(self, epoch: int) -> int:
        return int(self.plan.num_batches(epoch))

    def get_batch(self, epoch: int, index: int) -> Dict[str, torch.Tensor]:
        rows = np.asarray(self.plan.batch_rows(epoch, index), dtype=np.int64)
        x = torch.from_numpy(np.ascontiguousarray(self.X[rows]))
        if self.layout == "NHWC":
            x = x.permute(0, 3, 1, 2)
        batch: Dict[str, torch.Tensor] = {"x": x.to(self.dtype).contiguous(), "row": torch.from_numpy(rows)}
        for k, a in self.extras.items():
            v = np.asarray(a[rows])
            if v.dtype == bool:
                batch[k] = torch.from_numpy(v.copy())
            elif np.issubdtype(v.dtype, np.integer):
                batch[k] = torch.from_numpy(v.astype(np.int64))
            else:
                batch[k] = torch.from_numpy(v.astype(np.float32))
        return self.transform(batch) if self.transform is not None else batch

    def describe(self) -> Dict[str, Any]:
        plan = self.plan.describe() if hasattr(self.plan, "describe") else type(self.plan).__name__
        return {"kind": "array", "x_shape": [int(s) for s in np.shape(self.X)],
                "x_dtype": str(np.asarray(self.X[:0]).dtype), "layout": self.layout,
                "dtype": str(self.dtype), "extras": sorted(self.extras), "plan": plan}


# ============================================================================
# checkpoint-safe values, RNG state, atomic save/load
# ============================================================================

_TORCH_LEAVES = (torch.Tensor, torch.dtype, torch.device, torch.Size)


def to_plain(obj: Any, where: str = "state") -> Any:
    """`obj` with numpy scalars -> Python scalars, numpy arrays -> tensors,
    tuples/lists/dicts rebuilt, tensors kept; anything else is a TypeError.
    np.float64 IS a subclass of float, so numpy is checked first: an
    `isinstance(x, float)` test alone lets the --resume crash through."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, _TORCH_LEAVES):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(obj)).clone()
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, Mapping):
        items = [(to_plain(k, f"{where} key"), to_plain(v, f"{where}[{k!r}]")) for k, v in obj.items()]
        return collections.OrderedDict(items) if isinstance(obj, collections.OrderedDict) else dict(items)
    if isinstance(obj, tuple):
        return tuple(to_plain(v, f"{where}[{i}]") for i, v in enumerate(obj))
    if isinstance(obj, list):
        return [to_plain(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    raise TypeError(f"{where}: a {type(obj).__module__}.{type(obj).__name__} cannot go into a "
                    f"weights_only checkpoint")


def find_weights_only_violations(obj: Any, where: str = "checkpoint") -> List[str]:
    """Paths of every value `torch.load(weights_only=True)` would refuse or
    that is a numpy type (empty list = safe). Tensors are leaves."""
    out: List[str] = []

    def walk(o: Any, path: str) -> None:
        if isinstance(o, np.generic) or isinstance(o, np.ndarray):
            out.append(f"{path}: numpy {type(o).__name__}")
        elif isinstance(o, _TORCH_LEAVES) or o is None or type(o) in (bool, int, float, str):
            return
        elif type(o) in (dict, collections.OrderedDict):
            for k, v in o.items():
                walk(k, f"{path} key {k!r}")
                walk(v, f"{path}[{k!r}]")
        elif type(o) in (list, tuple):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        else:
            out.append(f"{path}: {type(o).__module__}.{type(o).__name__}")

    walk(obj, where)
    return out


def capture_rng_state() -> Dict[str, Any]:
    """Python, numpy (legacy global), torch CPU and every CUDA device's RNG
    state, as Python values and uint8 tensors only."""
    name, key, pos, has_gauss, cached = np.random.get_state()
    return {"python": to_plain(random.getstate()),
            "numpy": [str(name), [int(v) for v in key], int(pos), int(has_gauss), float(cached)],
            "torch": torch.get_rng_state(),
            "cuda": list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None}


def restore_rng_state(state: Mapping[str, Any]) -> None:
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    name, key, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(key, dtype=np.uint32), int(pos), int(has_gauss), float(cached)))
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
        else:
            logger.warning(f"checkpoint has {len(cuda)} CUDA RNG states, this host has "
                           f"{torch.cuda.device_count()} devices: CUDA RNG not restored")


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tmp_name(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def atomic_torch_save(obj: Any, path: PathLike) -> Path:
    """`torch.save` to a temp file in the same directory, fsync, then
    `os.replace` onto `path`: a crash at any point leaves either the old file
    or the new one under `path`, never a torn one. The temp file is removed
    on any exception."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(p)
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    _fsync_dir(p.parent)
    return p


def load_checkpoint(path: PathLike, map_location: Any = "cpu") -> Dict[str, Any]:
    """`torch.load(weights_only=True)` of a checkpoint this module wrote."""
    ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ResumeError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    return ckpt


def load_weights(path: PathLike, model: nn.Module, *, ema: bool = True, strict: bool = True) -> Dict[str, Any]:
    """Load a checkpoint's EMA weights (or the raw weights if `ema=False` or
    it has no EMA) into `model`, for sampling/export. Returns the checkpoint
    (step, history, config, ...)."""
    ckpt = load_checkpoint(path)
    state = ckpt["ema"]["module"] if (ema and ckpt.get("ema") is not None) else ckpt["model"]
    model.load_state_dict(state, strict=strict)
    return ckpt


def read_history(path: PathLike) -> List[Dict[str, Optional[float]]]:
    """history.csv as a list of rows; empty cells are None, step/epoch ints."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row: Dict[str, Optional[float]] = {}
            for k, v in r.items():
                if v in ("", None):
                    row[k] = None
                elif k in ("step", "epoch"):
                    row[k] = int(v)
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


# ============================================================================
# EMA, schedule, early stopping
# ============================================================================

class EMA:
    """Exponential moving average of a model's weights, in a deep copy
    (`ema.module`, eval mode, no grad). update: e <- d e + (1 - d) w, with
    d = min(decay, (1 + n) / (10 + n)) when `warmup` (n = updates so far).
    Buffers are copied, not averaged."""

    def __init__(self, model: nn.Module, decay: float, *, warmup: bool = True):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        n = self.num_updates
        return min(self.decay, (1.0 + n) / (10.0 + n)) if self.warmup else self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.current_decay()
        for e, p in zip(self.module.parameters(), model.parameters(), strict=True):
            if e.dtype.is_floating_point:
                e.lerp_(p.detach(), 1.0 - d)
            else:
                e.copy_(p)
        for e, b in zip(self.module.buffers(), model.buffers(), strict=True):
            e.copy_(b)
        self.num_updates += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "warmup": self.warmup, "num_updates": int(self.num_updates),
                "module": self.module.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state["module"])
        self.num_updates = int(state["num_updates"])


class WarmupCosine:
    """LR multiplier for optimizer step s (0-based), for `LambdaLR`: linear
    warm-up (s + 1) / warmup over the first `warmup_steps`, then a cosine from
    1 down to `min_ratio` over the rest. A callable object (not a lambda) so
    `LambdaLR.state_dict` stores its settings; Python floats only (math.cos,
    never np.cos: see the module docstring)."""

    def __init__(self, max_steps: int, warmup_steps: int = 0, min_ratio: float = 0.0):
        self.max_steps = int(max_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_ratio = float(min_ratio)

    def __call__(self, s: int) -> float:
        s = int(s)
        if self.warmup_steps > 0 and s < self.warmup_steps:
            return float((s + 1) / self.warmup_steps)
        span = max(1, self.max_steps - self.warmup_steps)
        p = min(1.0, max(0.0, (s - self.warmup_steps) / span))
        return float(self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p)))


class EarlyStopping:
    """Lower is better. `update(value, step)` -> True when `value` beats the
    best by more than `min_delta` (that evaluation becomes best.pt); after
    `patience` evaluations in a row without that, `stopped` is True.
    `patience=None` only tracks the best."""

    def __init__(self, patience: Optional[int] = None, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.best_step: Optional[int] = None
        self.bad_evals = 0
        self.stopped = False
        self.stop_step: Optional[int] = None

    def update(self, value: float, step: int) -> bool:
        value = float(value)
        if not math.isfinite(value):
            raise NonFiniteLossError(f"held-out metric is {value} at step {step}")
        if self.best is None or value < self.best - self.min_delta:
            self.best, self.best_step, self.bad_evals = value, int(step), 0
            return True
        self.bad_evals += 1
        if self.patience is not None and self.bad_evals >= self.patience:
            self.stopped, self.stop_step = True, int(step)
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {"patience": self.patience, "min_delta": self.min_delta, "best": self.best,
                "best_step": self.best_step, "bad_evals": int(self.bad_evals), "stopped": bool(self.stopped),
                "stop_step": self.stop_step}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.best = None if state["best"] is None else float(state["best"])
        self.best_step = None if state["best_step"] is None else int(state["best_step"])
        self.bad_evals = int(state["bad_evals"])
        self.stopped = bool(state["stopped"])
        self.stop_step = None if state["stop_step"] is None else int(state["stop_step"])


# ============================================================================
# the loop
# ============================================================================

@dataclass(frozen=True)
class StepInfo:
    """Passed to a `step_fn` that takes a third argument."""
    step: int            # optimizer steps completed before this one (0-based index)
    epoch: int
    step_in_epoch: int
    max_steps: int
    training: bool       # False for the held-out evaluation

    @property
    def progress(self) -> float:
        return self.step / max(1, self.max_steps)


StepOutput = Union[torch.Tensor, Tuple[torch.Tensor, Mapping[str, Any]]]
StepFn = Callable[..., StepOutput]


@dataclass
class TrainResult:
    step: int
    epoch: int
    finished: bool             # reached max_steps or stopped early (not just paused by until_step)
    stopped_early: bool
    best_step: Optional[int]
    best_metric: Optional[float]
    eval_metric: str
    resumed_from: Optional[int]
    run_dir: str
    last_train: Dict[str, float]
    last_eval: Dict[str, float]
    wall_s: float

    def summary(self) -> Dict[str, Any]:
        """JSON-able, for a stage's done.json summary."""
        return to_jsonable(dataclasses.asdict(self))


def _accepts_info(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    n = 0
    for p in params:
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            n += 1
    return n >= 3


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device, non_blocking=True)
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(_to_device(v, device) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _batch_size(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.ndim else 1
    if isinstance(batch, Mapping):
        if "x" in batch:
            return _batch_size(batch["x"])
        batch = list(batch.values())
    if isinstance(batch, (list, tuple)):
        for v in batch:
            if isinstance(v, (torch.Tensor, Mapping, list, tuple)):
                return _batch_size(v)
    return 1


class _MeanWindow:
    """Running (weighted) sums of scalar metrics as float64 tensors on the
    device: no host sync per step, and exact to restore from Python floats."""

    def __init__(self) -> None:
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, float] = {}
        self.steps = 0

    def add(self, values: Mapping[str, Any], device: torch.device, weight: float = 1.0) -> None:
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    raise ValueError(f"metric {k!r} must be a scalar, got shape {tuple(v.shape)}")
                t = v.detach().reshape(()).to(device=device, dtype=torch.float64)
            else:
                t = torch.tensor(float(v), dtype=torch.float64, device=device)
            if weight != 1.0:
                t = t * weight
            self.sums[k] = t if k not in self.sums else self.sums[k] + t
            self.counts[k] = self.counts.get(k, 0.0) + float(weight)
        self.steps += 1

    def totals(self) -> Dict[str, float]:
        if not self.sums:
            return {}
        keys = list(self.sums)
        vals = torch.stack([self.sums[k] for k in keys]).tolist()     # one sync
        return dict(zip(keys, (float(v) for v in vals)))

    def means(self) -> Dict[str, float]:
        return {k: v / self.counts[k] for k, v in self.totals().items()}

    def state_dict(self) -> Dict[str, Any]:
        return {"sums": self.totals(), "counts": dict(self.counts), "steps": int(self.steps)}

    def load_state_dict(self, state: Mapping[str, Any], device: torch.device) -> None:
        self.sums = {k: torch.tensor(float(v), dtype=torch.float64, device=device)
                     for k, v in state["sums"].items()}
        self.counts = {k: float(v) for k, v in state["counts"].items()}
        self.steps = int(state["steps"])


def _nonfinite(values: Mapping[str, float]) -> List[str]:
    return [k for k, v in values.items() if not math.isfinite(v)]


class Trainer:
    """Trains `model` with `step_fn` on `train_data` (see the module docstring
    for both contracts). Construct it identically for a resumed run (same
    model class and arguments, data sources, config and seed): the checkpoint's
    config, seed and data `describe()` are compared, and a mismatch raises
    `ResumeError` before anything is written.

    `seed`: re-seeds the global RNGs at a fresh start (via `derive_seed(seed,
    "train-loop")`) and seeds every evaluation (`derive_seed(seed, "eval")`).
    Seed the model's init yourself, `setup_determinism(seed)` before building it.
    `run_dir`: where best.pt / last.pt / history.csv go (default
    `ctx.stage_dir`). `resume`: continue from `run_dir/last.pt` if it exists
    (default `ctx.resume`, else False). A fresh start removes this trainer's
    old files from `run_dir`."""

    def __init__(self, model: nn.Module, step_fn: StepFn, train_data: BatchSource, config: TrainConfig, *,
                 eval_data: Optional[BatchSource] = None, eval_fn: Optional[StepFn] = None,
                 run_dir: Optional[PathLike] = None, seed: int = 0, device: Any = None,
                 ctx: Optional["StageContext"] = None, resume: Optional[bool] = None):
        if not isinstance(config, TrainConfig):
            raise TypeError(f"config must be a TrainConfig (or a subclass), got {type(config).__name__}")
        for name, src in (("train_data", train_data), ("eval_data", eval_data)):
            if src is not None and not isinstance(src, BatchSource):
                raise TypeError(f"{name} needs num_batches(epoch) and get_batch(epoch, index)")
        if run_dir is None:
            if ctx is None:
                raise ValueError("pass run_dir (or a StageContext)")
            run_dir = ctx.stage_dir
        self.cfg = config
        self.ctx = ctx
        self.run_dir = Path(run_dir)
        self.seed = int(seed)
        derive_seed(self.seed)  # validates
        self.resume_requested = bool(ctx.resume if (resume is None and ctx is not None) else resume)
        if device is None:
            first = next(model.parameters(), None)
            device = first.device if first is not None else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.step_fn, self.eval_fn = step_fn, (eval_fn or step_fn)
        self._step_info = _accepts_info(self.step_fn)
        self._eval_info = _accepts_info(self.eval_fn)
        self.train_data, self.eval_data = train_data, eval_data

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("the model has no trainable parameters")
        decay = [p for p in self.params if p.ndim >= 2]
        no_decay = [p for p in self.params if p.ndim < 2]
        groups = [g for g in ({"params": decay, "weight_decay": config.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}) if g["params"]]
        self.optimizer = torch.optim.AdamW(groups, lr=config.lr, betas=tuple(config.betas),
                                           eps=config.adam_eps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, WarmupCosine(config.max_steps, config.warmup_steps, config.min_lr / config.lr))
        self.ema = (EMA(self.model, config.ema_decay, warmup=config.ema_warmup)
                    if config.ema_decay is not None else None)
        self.early_stopping = EarlyStopping(config.early_stop_patience, config.early_stop_min_delta)

        self.step = 0
        self.epoch = 0
        self.step_in_epoch = 0
        self.finished = False
        self.history: List[Dict[str, Any]] = []
        self.resumed_from: Optional[int] = None
        self.resumes = 0
        self._window = _MeanWindow()
        self._last_lr = float(self.scheduler.get_last_lr()[0])
        self._wall_before = 0.0
        self._t_session = time.perf_counter()
        self._t_window = self._t_session
        self._saved_step: Optional[int] = None
        self._started = False

    # -- small helpers ------------------------------------------------------------
    @property
    def _last_epoch(self) -> int:
        """Epoch of the batch the last step used."""
        return self.epoch - 1 if (self.step_in_epoch == 0 and self.step > 0) else self.epoch

    @property
    def attempt(self) -> int:
        return int(self.ctx.attempt) if self.ctx is not None else self.resumes + 1

    def _log(self, msg: str) -> None:
        if self.ctx is not None:
            self.ctx.log(msg)
        else:
            logger.info(msg)

    def _event(self, event: str, **fields: Any) -> None:
        rec = {"event": event, "step": self.step, "epoch": self._last_epoch, "attempt": self.attempt,
               **fields}
        if self.ctx is not None:
            if Path(os.path.abspath(self.run_dir)) != Path(os.path.abspath(self.ctx.stage_dir)):
                rec["run_dir"] = os.path.relpath(self.run_dir, self.ctx.stage_dir)
            self.ctx.log_metrics(rec)
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.run_dir / METRICS_NAME, "a") as f:
                f.write(json.dumps(to_jsonable(rec), sort_keys=True, allow_nan=True) + "\n")

    def _autocast(self) -> Any:
        if self.cfg.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _wall(self) -> float:
        return self._wall_before + (time.perf_counter() - self._t_session)

    def _row(self, step: int) -> Dict[str, Any]:
        if self.history and self.history[-1]["step"] == step:
            return self.history[-1]
        row: Dict[str, Any] = {"step": int(step), "epoch": int(self._last_epoch)}
        self.history.append(row)
        return row

    @staticmethod
    def _call(fn: StepFn, with_info: bool, model: nn.Module, batch: Any,
              info: StepInfo) -> Tuple[torch.Tensor, Dict[str, Any]]:
        out = fn(model, batch, info) if with_info else fn(model, batch)
        if isinstance(out, torch.Tensor):
            loss, metrics = out, {}
        elif isinstance(out, (tuple, list)) and len(out) == 2:
            loss, metrics = out[0], dict(out[1] or {})
        else:
            raise TypeError("step_fn must return loss or (loss, metrics)")
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TypeError("step_fn's loss must be a scalar tensor")
        return loss.reshape(()), metrics

    # -- start / resume ---------------------------------------------------------------
    def _owned_files(self) -> List[Path]:
        names = [BEST_NAME, LAST_NAME, HISTORY_NAME] + ([METRICS_NAME] if self.ctx is None else [])
        return [self.run_dir / n for n in names]

    def _start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in (BEST_NAME, LAST_NAME, HISTORY_NAME):     # temp files of a killed writer
            for p in self.run_dir.glob(f".{name}.tmp-*"):
                with contextlib.suppress(OSError):
                    p.unlink()
        self.determinism = setup_determinism(derive_seed(self.seed, "train-loop"),
                                             deterministic=self.cfg.deterministic,
                                             warn_only=self.cfg.deterministic_warn_only)
        last = self.run_dir / LAST_NAME
        if self.resume_requested and last.is_file():
            ckpt = load_checkpoint(last)              # read and checked BEFORE anything is written
            self._check_resumable(ckpt, last)
            self._load_state(ckpt)
            self.resumed_from = self.step
            self.resumes += 1
            self._saved_step = self.step
            self._log(f"resumed from {last} at step {self.step} (epoch {self.epoch}, batch "
                      f"{self.step_in_epoch}){' - already finished' if self.finished else ''}")
            self._write_history()
            self._event("resume", from_step=self.step, finished=self.finished)
        else:
            if self.resume_requested:
                self._log(f"no {LAST_NAME} in {self.run_dir}: starting fresh")
            for p in self._owned_files():
                with contextlib.suppress(FileNotFoundError):
                    p.unlink()
            self._event("start", max_steps=self.cfg.max_steps, seed=self.seed, device=str(self.device),
                        deterministic=self.determinism.get("use_deterministic_algorithms", False))
        self._t_session = self._t_window = time.perf_counter()
        self._started = True

    def _describe(self, src: Optional[BatchSource]) -> Optional[Dict[str, Any]]:
        if src is None or not hasattr(src, "describe"):
            return None
        return to_plain(to_jsonable(src.describe()), "describe()")

    def _comparable_config(self, d: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if k not in RESUME_FREE_FIELDS}

    def _check_resumable(self, ckpt: Mapping[str, Any], path: Path) -> None:
        saved = self._comparable_config(ckpt["config"])
        now = self._comparable_config(to_plain(self.cfg.to_dict()))
        if saved != now:
            diff = sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))
            raise ResumeError(f"{path}: config differs in {diff}; start fresh (resume=False) instead")
        if int(ckpt["seed"]) != self.seed:
            raise ResumeError(f"{path}: written with seed {ckpt['seed']}, this run uses {self.seed}")
        if bool(ckpt["data"]["has_eval"]) != (self.eval_data is not None):
            raise ResumeError(f"{path}: eval data {'added' if self.eval_data is not None else 'removed'} "
                              f"on resume")
        for name, src in (("train", self.train_data), ("eval", self.eval_data)):
            have, want = ckpt["data"].get(name), self._describe(src)
            if have is not None and want is not None and have != want:
                raise ResumeError(f"{path}: the {name} data source differs from the checkpoint's")

    def _load_state(self, ckpt: Mapping[str, Any]) -> None:
        self.model.load_state_dict(ckpt["model"])
        if self.ema is not None:
            self.ema.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step, self.epoch = int(ckpt["step"]), int(ckpt["epoch"])
        self.step_in_epoch = int(ckpt["step_in_epoch"])
        self.finished = bool(ckpt["finished"])
        self.early_stopping.load_state_dict(ckpt["early_stopping"])
        self._window.load_state_dict(ckpt["window"], self.device)
        self.history = [dict(r) for r in ckpt["history"]]
        self._last_lr = float(ckpt["last_lr"])
        self._wall_before = float(ckpt["wall_s"])
        self.resumes = int(ckpt["resumes"])
        restore_rng_state(ckpt["rng"])               # last: nothing above may draw after it

    # -- checkpoints -------------------------------------------------------------------
    def state(self, kind: str) -> Dict[str, Any]:
        """The checkpoint dict (Python values and tensors only)."""
        window = self._window.state_dict()
        bad = _nonfinite(window["sums"])
        if bad:
            raise NonFiniteLossError(f"non-finite training metric(s) {bad} before step {self.step}; "
                                     f"not checkpointing")
        with torch.no_grad():
            if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
                raise NonFiniteLossError(f"non-finite weights at step {self.step}; not checkpointing")
        return {
            "format": CHECKPOINT_FORMAT,
            "kind": kind,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "optimizer": to_plain(self.optimizer.state_dict(), "optimizer"),
            "scheduler": to_plain(self.scheduler.state_dict(), "scheduler"),
            "step": int(self.step),
            "epoch": int(self.epoch),
            "step_in_epoch": int(self.step_in_epoch),
            "finished": bool(self.finished),
            "rng": capture_rng_state(),
            "early_stopping": to_plain(self.early_stopping.state_dict()),
            "best": {"step": self.early_stopping.best_step, "metric": self.early_stopping.best,
                     "eval_metric": self.cfg.eval_metric},
            "window": to_plain(window),
            "history": to_plain(self.history, "history"),
            "last_lr": float(self._last_lr),
            "wall_s": float(self._wall()),
            "resumes": int(self.resumes),
            "seed": int(self.seed),
            "config": to_plain(self.cfg.to_dict(), "config"),
            "config_class": f"{type(self.cfg).__module__}.{type(self.cfg).__qualname__}",
            "data": {"train": self._describe(self.train_data), "eval": self._describe(self.eval_data),
                     "has_eval": self.eval_data is not None},
            "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
        }

    def save(self, name: str = LAST_NAME) -> Path:
        state = self.state("best" if name == BEST_NAME else "last")
        problems = find_weights_only_violations(state)
        if problems:
            raise TypeError(f"checkpoint would not load with weights_only=True: {problems[:5]}")
        path = atomic_torch_save(state, self.run_dir / name)
        if name == LAST_NAME:
            self._saved_step = self.step
            self._write_history()
        return path

    def _write_history(self) -> None:
        path = self.run_dir / HISTORY_NAME
        if not self.history:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            return
        header: List[str] = ["step", "epoch"]
        for r in self.history:
            header.extend(k for k in r if k not in header)
        tmp = _tmp_name(path)
        try:
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header, restval="")
                w.writeheader()
                for r in self.history:
                    w.writerow({k: (int(v) if isinstance(v, bool) else ("" if v is None else v))
                                for k, v in r.items()})
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

    # -- training ----------------------------------------------------------------------
    def _train_step(self) -> None:
        n_batches = int(self.train_data.num_batches(self.epoch))
        if n_batches <= 0:
            raise ValueError(f"train data has no batches in epoch {self.epoch}")
        batch = _to_device(self.train_data.get_batch(self.epoch, self.step_in_epoch), self.device)
        info = StepInfo(step=self.step, epoch=self.epoch, step_in_epoch=self.step_in_epoch,
                        max_steps=self.cfg.max_steps, training=True)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            loss, metrics = self._call(self.step_fn, self._step_info, self.model, batch, info)
        if not loss.requires_grad:
            raise ValueError("step_fn's loss does not require grad (is the model's output detached?)")
        loss.backward()
        self._last_lr = float(self.scheduler.get_last_lr()[0])       # the LR this step uses
        values = {**metrics, "loss": loss.detach()}
        if self.cfg.grad_clip is not None:
            values["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self._window.add(values, self.device)
        self.step += 1
        self.step_in_epoch += 1
        if self.step_in_epoch >= n_batches:
            self.epoch += 1
            self.step_in_epoch = 0

    def _flush_train(self) -> None:
        if self._window.steps == 0:
            return
        means = self._window.means()
        steps = self._window.steps
        self._window = _MeanWindow()
        bad = _nonfinite(means)
        if bad:
            raise NonFiniteLossError(f"non-finite training metric(s) {bad} in the {steps} steps ending "
                                     f"at step {self.step}")
        now = time.perf_counter()
        rate = steps / max(now - self._t_window, 1e-9)
        self._t_window = now
        row = self._row(self.step)
        row.update({f"train_{k}": v for k, v in means.items()})
        row.update({"lr": self._last_lr, "wall_s": round(self._wall(), 3), "steps_per_s": round(rate, 3)})
        self._event("train", lr=self._last_lr, **{f"train_{k}": v for k, v in means.items()})
        self._log(f"step {self.step}/{self.cfg.max_steps} epoch {self._last_epoch} loss "
                  f"{means.get('loss', float('nan')):.5g} lr {self._last_lr:.3g} ({rate:.1f} steps/s)")

    @torch.no_grad()
    def evaluate(self, data: Optional[BatchSource] = None, *, use_ema: Optional[bool] = None) -> Dict[str, float]:
        """Batch-size-weighted means of the loss and metrics over every batch
        of `data` (default the eval data), keys prefixed `eval_`. Runs in
        eval mode, in a forked RNG seeded with derive_seed(seed, "eval"): the
        training RNG streams are untouched and repeated calls see the same noise."""
        data = data if data is not None else self.eval_data
        if data is None:
            raise ValueError("no eval data")
        use_ema = self.cfg.eval_with_ema if use_ema is None else use_ema
        model = self.ema.module if (use_ema and self.ema is not None) else self.model
        was_training = model.training
        model.eval()
        py_state, np_state = random.getstate(), np.random.get_state()
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        acc = _MeanWindow()
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                eval_seed = derive_seed(self.seed, "eval")
                torch.manual_seed(eval_seed)
                random.seed(eval_seed)
                np.random.seed(eval_seed % (2 ** 32))
                for i in range(int(data.num_batches(0))):
                    batch = _to_device(data.get_batch(0, i), self.device)
                    info = StepInfo(step=self.step, epoch=self._last_epoch, step_in_epoch=i,
                                    max_steps=self.cfg.max_steps, training=False)
                    with self._autocast():
                        loss, metrics = self._call(self.eval_fn, self._eval_info, model, batch, info)
                    acc.add({**metrics, "loss": loss}, self.device, weight=float(_batch_size(batch)))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            model.train(was_training)
        return {f"eval_{k}": v for k, v in acc.means().items()}

    def _evaluate_and_select(self) -> None:
        metrics = self.evaluate()
        key = f"eval_{self.cfg.eval_metric}"
        if key not in metrics:
            raise KeyError(f"eval_metric {self.cfg.eval_metric!r} is not among the held-out metrics "
                           f"{sorted(k[5:] for k in metrics)}")
        bad = _nonfinite(metrics)
        if bad:
            raise NonFiniteLossError(f"non-finite held-out metric(s) {bad} at step {self.step}")
        improved = self.early_stopping.update(metrics[key], self.step)
        row = self._row(self.step)
        row.update(metrics)
        row["eval_best"] = int(improved)
        self._event("eval", best=improved, **metrics)
        self._log(f"eval step {self.step}: {key} {metrics[key]:.5g}"
                  f"{' (best)' if improved else ''}")
        if improved:
            self.save(BEST_NAME)
        if self.early_stopping.stopped:
            self._log(f"early stop at step {self.step}: no improvement in "
                      f"{self.early_stopping.bad_evals} evaluations; best step {self.early_stopping.best_step}")

    def _finish(self, evaluated: bool) -> None:
        self._flush_train()
        if self.eval_data is not None and not evaluated:
            self._evaluate_and_select()
        self.finished = True
        if self.eval_data is None:
            self.save(BEST_NAME)            # no held-out data: the final model is the best one
        self.save(LAST_NAME)
        self._event("end", stopped_early=self.early_stopping.stopped,
                    best_step=self.early_stopping.best_step if self.eval_data is not None else self.step,
                    best_metric=self.early_stopping.best, wall_s=round(self._wall(), 3))

    def fit(self, *, until_step: Optional[int] = None) -> TrainResult:
        """Train to `max_steps` (or an early stop). `until_step` pauses after
        that many TOTAL steps, checkpointing last.pt without finishing (the
        schedule still spans max_steps); a later `fit()` continues."""
        if not self._started:
            self._start()
        cfg = self.cfg
        target = cfg.max_steps if until_step is None else min(int(until_step), cfg.max_steps)
        while not self.finished and self.step < target:
            self._train_step()
            s = self.step
            if s % cfg.log_every == 0:
                self._flush_train()
            evaluated = False
            if self.eval_data is not None and s % cfg.eval_every == 0:
                self._evaluate_and_select()
                evaluated = True
            if self.early_stopping.stopped or s >= cfg.max_steps:
                self._finish(evaluated)
                break
            if evaluated or s % cfg.ckpt_every == 0:
                self.save(LAST_NAME)
        if not self.finished and self._saved_step != self.step:
            self.save(LAST_NAME)
        if self.finished and self.ctx is not None:
            self._register_outputs()
        return self.result()

    def _register_outputs(self) -> None:
        for name in (BEST_NAME, LAST_NAME, HISTORY_NAME):
            p = self.run_dir / name
            if p.is_file():
                try:
                    self.ctx.add_output(p)
                except ValueError as e:      # run_dir outside the stage dir
                    self.ctx.warn(f"{name} not recorded as a stage output: {e}")

    def result(self) -> TrainResult:
        last_train = next(({k[6:]: v for k, v in r.items() if k.startswith("train_")}
                           for r in reversed(self.history) if "train_loss" in r), {})
        last_eval = next(({k[5:]: v for k, v in r.items() if k.startswith("eval_") and k != "eval_best"}
                          for r in reversed(self.history) if "eval_loss" in r), {})
        has_eval = self.eval_data is not None
        return TrainResult(
            step=self.step, epoch=self.epoch, finished=self.finished,
            stopped_early=self.early_stopping.stopped,
            best_step=self.early_stopping.best_step if has_eval else (self.step if self.finished else None),
            best_metric=self.early_stopping.best if has_eval else None,
            eval_metric=self.cfg.eval_metric, resumed_from=self.resumed_from, run_dir=str(self.run_dir),
            last_train=last_train, last_eval=last_eval, wall_s=round(self._wall(), 3))
