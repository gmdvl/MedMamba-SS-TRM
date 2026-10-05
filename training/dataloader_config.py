# -*- coding: utf-8 -*-
"""
training/dataloader_config.py
==============================
MedMamba-SS-TRM v12/v13 stability plan, Phases 1/2/7/8/9.

Centralizes DataLoader resource policy instead of embedding
`num_workers=8, prefetch_factor=4, persistent_workers=(...), pin_memory=...`
directly in every training entry point. The previous defaults (8 workers x
prefetch_factor 4 x persistent_workers=True, simultaneously on BOTH the
train and validation loaders) can hold on the order of dozens of prefetched
HSI batches in `/dev/shm` at once - exactly the configuration this plan's
introduction identifies as "completely consistent with" the observed
worker `SIGBUS` / "insufficient shared memory (shm)" failures.

Three named modes (Phase 1.1):

    safe        - num_workers=0, no prefetching, no persistence, no pin_memory.
                  Removes /dev/shm from the critical path entirely. This is
                  the default.
    balanced    - 1-2 workers, prefetch_factor=1, not persistent, pin_memory
                  optional.
    performance - user-controlled worker count, modest prefetch (1-2),
                  persistence/pin_memory optional. Opt-in only - never the
                  default.

`--loader_mode` selects one of these; any of `--num_workers`,
`--prefetch_factor`, `--persistent_workers`, `--pin_memory` passed
explicitly on the CLI overrides that one field of the selected mode.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch

LOADER_MODES = {
    "safe":        dict(num_workers=0, prefetch_factor=None, persistent_workers=False, pin_memory=False),
    "balanced":    dict(num_workers=2, prefetch_factor=1,    persistent_workers=False, pin_memory=False),
    "performance": dict(num_workers=4, prefetch_factor=2,    persistent_workers=True,  pin_memory=True),
}


@dataclass
class DataLoaderPolicy:
    loader_mode: str = "safe"
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory: bool = False

    def as_dict(self):
        return {
            "loader_mode": self.loader_mode,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
        }

    def dataloader_kwargs(self, device_type: str = "cpu", seed: int = 0) -> dict:
        """Returns kwargs suitable for `torch.utils.data.DataLoader(**kwargs)`.
        `prefetch_factor` must be omitted entirely (not just None) when
        `num_workers == 0`, or PyTorch raises.

        v15 R6.4 - `seed` is threaded into `make_worker_init_fn`. This used to
        be called with NO arguments, so worker seeds were `0 + worker_id`
        regardless of `--seed`: two runs differing only in `--seed` got
        identical worker RNG streams, and the run was not reproducible from the
        flag that claims to make it so.
        """
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": bool(self.pin_memory and device_type == "cuda"),
            "persistent_workers": bool(self.persistent_workers and self.num_workers > 0),
        }
        if self.num_workers > 0 and self.prefetch_factor:
            kwargs["prefetch_factor"] = self.prefetch_factor
        if self.num_workers > 0:
            kwargs["worker_init_fn"] = make_worker_init_fn(base_seed=seed)
        return kwargs


def resolve_loader_policy(loader_mode: str = "safe", num_workers: Optional[int] = None,
                           prefetch_factor: Optional[int] = None,
                           persistent_workers: Optional[bool] = None,
                           pin_memory: Optional[bool] = None) -> DataLoaderPolicy:
    """Starts from the named mode's defaults (Phase 1.1's table), then lets
    any explicitly-passed (non-None) CLI value override a single field -
    e.g. `--loader_mode safe --pin_memory` still gets 0 workers/no
    prefetch/no persistence, just with pin_memory forced on."""
    if loader_mode not in LOADER_MODES:
        raise ValueError(f"Unknown --loader_mode {loader_mode!r}. Choices: {sorted(LOADER_MODES)}")
    base = LOADER_MODES[loader_mode]
    policy = DataLoaderPolicy(
        loader_mode=loader_mode,
        num_workers=base["num_workers"] if num_workers is None else num_workers,
        prefetch_factor=base["prefetch_factor"] if prefetch_factor is None else prefetch_factor,
        persistent_workers=base["persistent_workers"] if persistent_workers is None else persistent_workers,
        pin_memory=base["pin_memory"] if pin_memory is None else pin_memory,
    )
    if policy.num_workers == 0:
        # Phase 1.1 - N/A fields must not leak through when there are no
        # worker processes to persist or prefetch for.
        policy = replace(policy, prefetch_factor=None, persistent_workers=False)
    return policy


def train_val_policies(loader_mode: str, num_workers: Optional[int], prefetch_factor: Optional[int],
                        persistent_workers: Optional[bool], pin_memory: Optional[bool]):
    """Phase 2.4 - train and validation do not need identical worker
    policies; validation never benefits from aggressive prefetching. If the
    resolved train policy uses >0 workers, validation gets at most
    (workers - 1), floor 0, and never persistent (a deterministic,
    single-pass-per-epoch loader has little to gain from persistence)."""
    train_policy = resolve_loader_policy(loader_mode, num_workers, prefetch_factor,
                                          persistent_workers, pin_memory)
    val_workers = max(0, min(train_policy.num_workers, max(0, train_policy.num_workers - 1)))
    val_policy = replace(train_policy, num_workers=val_workers,
                          persistent_workers=False if val_workers == 0 else train_policy.persistent_workers)
    if val_workers == 0:
        val_policy = replace(val_policy, prefetch_factor=None)
    return train_policy, val_policy


# ----------------------------------------------------------------------
# Phase 8 - deterministic worker initialization
# ----------------------------------------------------------------------

def make_worker_init_fn(base_seed: int = 0, cpu_threads_per_worker: Optional[int] = 1):
    """Returns a `worker_init_fn` that: seeds numpy/random/torch
    independently per worker (derived from `base_seed` + worker id, so
    workers don't share RNG state - see `training/augmentation.py`'s own
    documented caveat about this), and optionally caps intra-worker thread
    counts (Phase 9) to avoid CPU oversubscription from
    NumPy/OpenMP/MKL/PyTorch all spawning their own thread pools inside
    every one of several worker PROCESSES simultaneously."""

    def worker_init_fn(worker_id: int) -> None:
        # v15 R6.4 - mixed rather than `base_seed + worker_id`, which makes
        # worker 1 of a seed-42 run draw exactly what worker 0 of a seed-43 run
        # draws. numpy's legacy seeder wants a value below 2**32.
        seed = ((int(base_seed) * 0x9E3779B1) ^ ((worker_id + 1) * 0x85EBCA6B)) % (2 ** 31)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if cpu_threads_per_worker:
            try:
                torch.set_num_threads(cpu_threads_per_worker)
            except Exception:
                pass
            os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads_per_worker))
            os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads_per_worker))

    return worker_init_fn


def apply_main_process_thread_limits(cpu_threads: Optional[int]) -> None:
    """Phase 9 - optional `--cpu_threads` control for the MAIN process
    (workers are handled by `make_worker_init_fn` above). Left as a no-op
    if `cpu_threads` is None (don't override user/system defaults)."""
    if not cpu_threads:
        return
    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)


def seed_everything(seed: int, deterministic: bool = False) -> dict:
    """v15 R6.4 - seed ALL FOUR RNGs, or stop claiming the run is seeded.

    `train_example_v6..v14._main` set `torch.manual_seed` and `np.random.seed`
    and neither `torch.cuda.manual_seed_all` nor `random.seed`, while
    `cudnn.benchmark = True` was set unconditionally with no
    `use_deterministic_algorithms`. Returns the state it established, for
    `config.json`.

    `deterministic=True` sets `cudnn.deterministic`, clears `cudnn.benchmark`
    (which picks algorithms by timing, so it is nondeterministic by design)
    and calls `torch.use_deterministic_algorithms(True, warn_only=True)`.
    `warn_only` because several ops used here have no deterministic CUDA
    implementation; the warnings say which, and a run that needs bit-exact
    reproducibility should read them rather than assume.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    state = {"seed": seed, "python_random": True, "numpy": True, "torch": True,
             "torch_cuda_all": bool(torch.cuda.is_available()), "deterministic": bool(deterministic)}
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            state["use_deterministic_algorithms"] = "warn_only"
        except Exception as e:                       # older torch, or an unsupported backend
            state["use_deterministic_algorithms"] = f"unavailable: {e}"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        state["cudnn_benchmark"] = bool(getattr(torch.backends.cudnn, "benchmark", False))
    return state
