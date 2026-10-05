# -*- coding: utf-8 -*-
"""
training/utils.py
==================
Cross-cutting helpers: deterministic seeding, device selection, and
collecting the hardware/software metadata that goes into config.yaml
(Phase 3 of the plan) so every experiment records exactly what it ran on.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Optional

import numpy as np
import torch


# ============================================================================
# Reproducibility (Phase 22 / "Deterministic reproducibility")
# ============================================================================

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seeds python/numpy/torch (CPU+CUDA). If `deterministic` is True, also
    asks cuDNN for deterministic algorithms (slower, but bit-for-bit
    reproducible runs on the same hardware/driver)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def get_rng_state() -> dict:
    """Snapshot of all RNG states, for checkpointing (Phase 5)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass


# ============================================================================
# Device selection / perf toggles
# ============================================================================

def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def enable_fast_math(tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    """Phase 20 performance toggles that are safe defaults for fixed input
    sizes on Ampere+ GPUs. No-ops on CPU-only machines."""
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = cudnn_benchmark


# ============================================================================
# Reproducible metadata (Phase 3)
# ============================================================================

def get_git_commit(cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        commit = out.decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=cwd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "name": props.name,
        "count": torch.cuda.device_count(),
        "total_memory_mb": round(props.total_memory / 1e6, 1),
        "capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def get_cpu_ram_info() -> dict:
    info = {"cpu_count": os.cpu_count()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
    except Exception:
        # fall back to /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_total_gb"] = round(kb / 1e6, 2)
                        break
        except Exception:
            pass
    return info


def collect_system_metadata(cwd: Optional[str] = None) -> dict:
    """Everything Phase 3 asks for: hardware, software, git, host."""
    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "git_commit": get_git_commit(cwd),
        "gpu": get_gpu_info(),
        **get_cpu_ram_info(),
    }


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return {"total": int(total), "trainable": int(trainable), "param_memory_mb": round(mem_mb, 3)}


class AverageMeter:
    """Running mean tracker, used for loss/acc/timing averages within an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
