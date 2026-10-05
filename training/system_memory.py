# -*- coding: utf-8 -*-
"""
training/system_memory.py
==========================
MedMamba-SS-TRM v12/v13 "Definitive Training Stability & SIGBUS Remediation"
plan, Phase 3 - `/dev/shm` / RAM / GPU monitoring.

This module turns an otherwise-mysterious `SIGBUS` into an observable
resource condition: log `/dev/shm` and system RAM availability BEFORE any
DataLoader worker is spawned, and again periodically during training, so a
worker crash can be correlated against a concrete "available shared memory
dropped to X MB" data point instead of guessed at after the fact.

Every function here is defensive - on a platform without `/dev/shm` (e.g.
non-Linux) or without `psutil` installed, it degrades to `None`/best-effort
fields rather than raising, since this module must never itself be the
reason a training run fails.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, Optional

import torch

SHM_PATH = "/dev/shm"


def get_shm_info() -> Dict:
    """Total/used/available bytes for `/dev/shm` (POSIX shared memory,
    which is what PyTorch DataLoader workers use for tensor IPC). Returns
    `{"available": False, ...}` on platforms without it."""
    if not os.path.isdir(SHM_PATH):
        return {"available": False, "path": SHM_PATH, "reason": "path does not exist on this platform"}
    try:
        usage = shutil.disk_usage(SHM_PATH)
        return {
            "available": True,
            "path": SHM_PATH,
            "total_mb": round(usage.total / 1e6, 2),
            "used_mb": round(usage.used / 1e6, 2),
            "free_mb": round(usage.free / 1e6, 2),
        }
    except OSError as e:
        return {"available": False, "path": SHM_PATH, "reason": str(e)}


def get_ram_info() -> Dict:
    """System RAM total/available/used, plus this process's own RSS.
    Prefers `psutil`; falls back to `/proc/meminfo` (Linux) for the
    system-wide numbers, and `resource` for RSS, if `psutil` isn't
    installed."""
    info: Dict = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_mb"] = round(vm.total / 1e6, 2)
        info["available_mb"] = round(vm.available / 1e6, 2)
        info["used_mb"] = round(vm.used / 1e6, 2)
        info["percent_used"] = vm.percent
        info["process_rss_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 2)
        info["source"] = "psutil"
        return info
    except Exception:
        pass

    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].strip().split()[0]) * 1024  # kB -> bytes
        if meminfo:
            info["total_mb"] = round(meminfo.get("MemTotal", 0) / 1e6, 2)
            info["available_mb"] = round(meminfo.get("MemAvailable", 0) / 1e6, 2)
            info["used_mb"] = round((meminfo.get("MemTotal", 0) - meminfo.get("MemAvailable", 0)) / 1e6, 2)
            info["source"] = "/proc/meminfo"
    except Exception as e:
        info["error"] = str(e)

    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
        info["process_rss_mb"] = round(rss_kb / 1e3, 2)
    except Exception:
        pass

    return info


def get_gpu_memory_info() -> Dict:
    if not torch.cuda.is_available():
        return {"available": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available": True,
        "device_name": props.name,
        "total_mb": round(props.total_memory / 1e6, 2),
        "allocated_mb": round(torch.cuda.memory_allocated(idx) / 1e6, 2),
        "reserved_mb": round(torch.cuda.memory_reserved(idx) / 1e6, 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(idx) / 1e6, 2),
    }


def collect_system_memory_snapshot() -> Dict:
    return {"ram": get_ram_info(), "shm": get_shm_info(), "gpu": get_gpu_memory_info()}


def format_system_memory_snapshot(snapshot: Optional[Dict] = None) -> str:
    snapshot = snapshot or collect_system_memory_snapshot()
    ram, shm, gpu = snapshot["ram"], snapshot["shm"], snapshot["gpu"]
    lines = ["[system-memory]"]
    if ram:
        lines.append(f"  RAM: available={ram.get('available_mb', 'n/a')}MB "
                      f"total={ram.get('total_mb', 'n/a')}MB "
                      f"process_rss={ram.get('process_rss_mb', 'n/a')}MB")
    if shm.get("available"):
        lines.append(f"  SHM: free={shm.get('free_mb')}MB total={shm.get('total_mb')}MB "
                      f"({shm.get('path')})")
    else:
        lines.append(f"  SHM: unavailable ({shm.get('reason', 'n/a')})")
    if gpu.get("available"):
        lines.append(f"  GPU: {gpu.get('device_name')} allocated={gpu.get('allocated_mb')}MB "
                      f"reserved={gpu.get('reserved_mb')}MB total={gpu.get('total_mb')}MB")
    else:
        lines.append("  GPU: not available")
    return "\n".join(lines)


def log_system_memory(logger=None, label: str = "") -> Dict:
    """Prints/logs one snapshot; returns it so callers can persist it (e.g.
    into `system_memory.json` / a preflight report)."""
    snapshot = collect_system_memory_snapshot()
    text = format_system_memory_snapshot(snapshot)
    if label:
        text = f"[system-memory:{label}]\n" + text.split("\n", 1)[1] if "\n" in text else text
    if logger is not None:
        logger.info(text)
    else:
        print(text, flush=True)
    return snapshot


def check_shm_sufficient(min_free_mb: float = 256.0) -> bool:
    """Best-effort pre-flight check: is there enough `/dev/shm` headroom to
    even attempt multi-worker DataLoaders? Returns True (optimistic) if
    `/dev/shm` info can't be determined at all (e.g. non-Linux), since in
    that case workers use a different IPC mechanism and this check doesn't
    apply."""
    shm = get_shm_info()
    if not shm.get("available"):
        return True
    return shm.get("free_mb", 0.0) >= min_free_mb


def write_system_memory_report(out_path: str, label: str = "") -> Dict:
    snapshot = collect_system_memory_snapshot()
    snapshot["label"] = label
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot
