# -*- coding: utf-8 -*-
"""tests/test_failure_taxonomy.py - MedMamba-SS-TRM v13 plan, Workstream 5.

Run with: pytest test_failure_taxonomy.py -v
"""
from training.failure_taxonomy import (
    FailureClass, classify_npy_status, classify_exception, classify_stop_reason,
)


class _DatasetStorageError(RuntimeError):
    pass
_DatasetStorageError.__name__ = "DatasetStorageError"


class _DatasetLeakageError(RuntimeError):
    pass
_DatasetLeakageError.__name__ = "DatasetLeakageError"


def test_classify_npy_status():
    assert classify_npy_status("TRUNCATED") is FailureClass.DATASET_TRUNCATED
    assert classify_npy_status("HEADER_INVALID") is FailureClass.DATASET_HEADER_INVALID
    assert classify_npy_status("MISSING") is FailureClass.DATASET_MISSING
    assert classify_npy_status("EMPTY") is FailureClass.DATASET_EMPTY
    assert classify_npy_status("UNREADABLE") is FailureClass.DATASET_READ_ERROR
    assert classify_npy_status("VALID") is FailureClass.COMPLETED
    assert classify_npy_status("something-else") is FailureClass.DATASET_READ_ERROR


def test_classify_exception_dataset():
    assert classify_exception(_DatasetStorageError("TRUNCATED: 10 bytes ...")) is FailureClass.DATASET_TRUNCATED
    assert classify_exception(_DatasetStorageError("X_train has 40 samples but y_train has 5")) \
        is FailureClass.DATASET_XY_MISMATCH
    assert classify_exception(_DatasetStorageError("does not match its manifest")) \
        is FailureClass.DATASET_MANIFEST_MISMATCH
    assert classify_exception(_DatasetLeakageError("leak")) is FailureClass.DATASET_LEAKAGE


def test_classify_exception_runtime():
    assert classify_exception(ValueError("Failed to read all data for array")) is FailureClass.DATASET_TRUNCATED
    assert classify_exception(ValueError("mmap length is greater than file size")) is FailureClass.DATASET_MMAP_ERROR
    assert classify_exception(RuntimeError("DataLoader worker (pid 1) is killed by signal: Bus error")) \
        is FailureClass.DATALOADER_WORKER_CRASH
    assert classify_exception(RuntimeError("CUDA out of memory")) is FailureClass.CUDA_OOM
    assert classify_exception(MemoryError()) is FailureClass.SYSTEM_MEMORY_PRESSURE
    assert classify_exception(KeyboardInterrupt()) is FailureClass.USER_INTERRUPTED
    assert classify_exception(SystemExit("NUMERICAL_SMOKE_TEST_FAILED: stage=loss")) \
        is FailureClass.NUMERICAL_SMOKE_TEST_FAILED


def test_classify_exception_io_error_and_gaps():
    # v13.1 remediation: physically-unreadable file (OSError(5)/EIO)
    assert classify_npy_status("IO_ERROR") is FailureClass.DATASET_IO_ERROR
    assert classify_exception(_DatasetStorageError(
        "DATASET_STORAGE_ERROR: X_train (...): PHYSICALLY UNREADABLE - "
        "OSError(5)/EIO ... not a truncated write")) is FailureClass.DATASET_IO_ERROR
    assert classify_exception(OSError(5, "Input/output error")) is FailureClass.DATASET_IO_ERROR
    # missing dataset file
    assert classify_exception(FileNotFoundError(
        "Missing required file: data/rgb/X_train.npy")) is FailureClass.DATASET_MISSING
    # class coverage hard gate
    assert classify_exception(ValueError(
        "Class coverage check failed (Main Development Plan, Phase 3) - at least one "
        "split has ZERO samples for a required class")) is FailureClass.CLASS_COVERAGE_INCOMPLETE
    # the misleading numpy text still maps to TRUNCATED when no EIO marker present
    assert classify_exception(ValueError("Failed to read all data for array")) \
        is FailureClass.DATASET_TRUNCATED


def test_classify_stop_reason():
    assert classify_stop_reason("TRAINING_COMPLETED") is FailureClass.COMPLETED
    assert classify_stop_reason("TRAINING_ABORTED_NUMERICAL_INSTABILITY") is FailureClass.NONFINITE_GRADIENT
    assert classify_stop_reason("EARLY_STOPPED_PATIENCE_F1_MACRO") is FailureClass.COMPLETED
    assert classify_stop_reason("GRADIENT_EXPLOSION") is FailureClass.NONFINITE_GRADIENT
    assert classify_stop_reason(None) is FailureClass.UNKNOWN


def test_failureclass_is_str_enum():
    assert FailureClass.DATASET_TRUNCATED.value == "DATASET_TRUNCATED"
    assert FailureClass.DATASET_TRUNCATED == "DATASET_TRUNCATED"
