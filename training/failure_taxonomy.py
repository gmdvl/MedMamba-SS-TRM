# -*- coding: utf-8 -*-
"""
training/failure_taxonomy.py
=============================
MedMamba-SS-TRM v13 stability plan, Workstream 5 - one shared vocabulary for
"why did this run stop", mapped from the several typed exceptions and
`stop_reason` strings the pipeline already produces.

This is purely additive. The closed `StopReason` enum in
`training/trainerg_v4.py` is untouched; `FailureClass` is a cross-cutting
label written into failure artifacts (`dataset_failure/failure.json`,
`numerical_failure/failure.json`, `<exp_dir>/failure_class.json`) so a
post-mortem / experiment-aggregation script has a single field to group on.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class FailureClass(str, Enum):
    # dataset / storage
    DATASET_TRUNCATED = "DATASET_TRUNCATED"
    DATASET_HEADER_INVALID = "DATASET_HEADER_INVALID"
    DATASET_SIZE_MISMATCH = "DATASET_SIZE_MISMATCH"
    DATASET_READ_ERROR = "DATASET_READ_ERROR"
    DATASET_IO_ERROR = "DATASET_IO_ERROR"
    DATASET_MMAP_ERROR = "DATASET_MMAP_ERROR"
    DATASET_XY_MISMATCH = "DATASET_XY_MISMATCH"
    DATASET_MISSING = "DATASET_MISSING"
    DATASET_EMPTY = "DATASET_EMPTY"
    DATASET_MANIFEST_MISMATCH = "DATASET_MANIFEST_MISMATCH"
    DATASET_LEAKAGE = "DATASET_LEAKAGE"
    CLASS_COVERAGE_INCOMPLETE = "CLASS_COVERAGE_INCOMPLETE"

    # dataloader / system
    DATALOADER_SHM = "DATALOADER_SHM"
    DATALOADER_WORKER_CRASH = "DATALOADER_WORKER_CRASH"
    CUDA_OOM = "CUDA_OOM"
    SYSTEM_MEMORY_PRESSURE = "SYSTEM_MEMORY_PRESSURE"

    # numerical
    NONFINITE_FORWARD = "NONFINITE_FORWARD"
    NONFINITE_LOSS = "NONFINITE_LOSS"
    NONFINITE_GRADIENT = "NONFINITE_GRADIENT"
    PARAMETER_CORRUPTION = "PARAMETER_CORRUPTION"
    AMP_NUMERICAL_INSTABILITY = "AMP_NUMERICAL_INSTABILITY"
    NUMERICAL_SMOKE_TEST_FAILED = "NUMERICAL_SMOKE_TEST_FAILED"

    # representation (v15 R0.3) - the model is finite and training "works",
    # but it is a constant function of its input: every validation sample gets
    # the same predicted class, epoch after epoch. Distinct from the NONFINITE_*
    # family, which is about arithmetic, not about the model learning nothing.
    REPRESENTATION_COLLAPSE = "REPRESENTATION_COLLAPSE"

    # other
    UNKNOWN_SIGBUS = "UNKNOWN_SIGBUS"
    USER_INTERRUPTED = "USER_INTERRUPTED"
    COMPLETED = "TRAINING_COMPLETED"
    UNKNOWN = "UNKNOWN"


# npy_integrity status string -> FailureClass
_NPY_STATUS_MAP = {
    "TRUNCATED": FailureClass.DATASET_TRUNCATED,
    "HEADER_INVALID": FailureClass.DATASET_HEADER_INVALID,
    "SIZE_MISMATCH": FailureClass.DATASET_SIZE_MISMATCH,
    "UNREADABLE": FailureClass.DATASET_READ_ERROR,
    "IO_ERROR": FailureClass.DATASET_IO_ERROR,
    "MISSING": FailureClass.DATASET_MISSING,
    "EMPTY": FailureClass.DATASET_EMPTY,
    "VALID": FailureClass.COMPLETED,
    "NOT_CHECKED": FailureClass.UNKNOWN,
}


def classify_npy_status(status: Optional[str]) -> FailureClass:
    return _NPY_STATUS_MAP.get(str(status), FailureClass.DATASET_READ_ERROR)


def classify_exception(exc: BaseException) -> FailureClass:
    """Best-effort mapping of an exception raised anywhere in the pipeline."""
    name = type(exc).__name__
    msg = str(exc).lower()

    # A physically-unreadable file (OSError(5)/EIO) must win over the misleading
    # numpy "not fully written?" / "failed to read all data" text and over the
    # word "truncated" that appears (negated) in the IO_ERROR guidance message.
    io_error = ("physically unreadable" in msg or "oserror(5)" in msg or "errno 5" in msg
                or "[errno 5]" in msg or "input/output error" in msg or "eio" in msg)

    if name == "DatasetLeakageError":
        return FailureClass.DATASET_LEAKAGE
    if name == "FileNotFoundError" or "missing required file" in msg:
        return FailureClass.DATASET_MISSING
    if "class coverage check failed" in msg or "zero samples for a required class" in msg:
        return FailureClass.CLASS_COVERAGE_INCOMPLETE
    if name == "DatasetStorageError":
        if io_error:
            return FailureClass.DATASET_IO_ERROR
        if "truncated" in msg:
            return FailureClass.DATASET_TRUNCATED
        if "has" in msg and "samples but" in msg:
            return FailureClass.DATASET_XY_MISMATCH
        if "header" in msg:
            return FailureClass.DATASET_HEADER_INVALID
        if "manifest" in msg:
            return FailureClass.DATASET_MANIFEST_MISMATCH
        return FailureClass.DATASET_READ_ERROR
    if name in ("OutOfMemoryError", "CudaError") or "cuda out of memory" in msg or "cublas" in msg and "alloc" in msg:
        return FailureClass.CUDA_OOM
    if io_error:
        return FailureClass.DATASET_IO_ERROR
    if "mmap length is greater than file size" in msg:
        return FailureClass.DATASET_MMAP_ERROR
    if "bus error" in msg or "sigbus" in msg or "signal 7" in msg:
        # a worker bus error is the DataLoader/shm path; a bare one is unknown
        return FailureClass.DATALOADER_WORKER_CRASH if "worker" in msg else FailureClass.UNKNOWN_SIGBUS
    if "insufficient shared memory" in msg or "/dev/shm" in msg:
        return FailureClass.DATALOADER_SHM
    if "failed to read all data" in msg:
        return FailureClass.DATASET_TRUNCATED
    if name == "KeyboardInterrupt":
        return FailureClass.USER_INTERRUPTED
    if name == "MemoryError":
        return FailureClass.SYSTEM_MEMORY_PRESSURE
    if name == "SystemExit":
        if "numerical_smoke_test_failed" in msg:
            return FailureClass.NUMERICAL_SMOKE_TEST_FAILED
        if "numerical_instability" in msg:
            return FailureClass.NONFINITE_GRADIENT
        if "representation_collapse" in msg:
            return FailureClass.REPRESENTATION_COLLAPSE
    return FailureClass.UNKNOWN


def classify_stop_reason(stop_reason: Optional[str]) -> FailureClass:
    """Map a trainer `stop_reason` (bare string or `StopReason` enum value)."""
    if stop_reason is None:
        return FailureClass.UNKNOWN
    s = str(getattr(stop_reason, "value", stop_reason)).upper()
    table = {
        "TRAINING_COMPLETED": FailureClass.COMPLETED,
        "TRAINING_ABORTED_REPRESENTATION_COLLAPSE": FailureClass.REPRESENTATION_COLLAPSE,
        "USER_INTERRUPTED": FailureClass.USER_INTERRUPTED,
        "TRAINING_ABORTED_NUMERICAL_INSTABILITY": FailureClass.NONFINITE_GRADIENT,
        "LOSS_NAN": FailureClass.NONFINITE_LOSS,
        "LOSS_INF": FailureClass.NONFINITE_LOSS,
        "GRADIENT_EXPLOSION": FailureClass.NONFINITE_GRADIENT,
    }
    if s in table:
        return table[s]
    if "REPRESENTATION_COLLAPSE" in s:
        return FailureClass.REPRESENTATION_COLLAPSE
    if s.startswith("EARLY_STOPPED_PATIENCE"):
        return FailureClass.COMPLETED
    if "NUMERIC" in s or "NONFINITE" in s or "NAN" in s or "INF" in s:
        return FailureClass.NONFINITE_GRADIENT
    if "PARAM" in s and "CORRUPT" in s:
        return FailureClass.PARAMETER_CORRUPTION
    return FailureClass.UNKNOWN
