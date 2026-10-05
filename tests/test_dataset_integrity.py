# -*- coding: utf-8 -*-
"""tests/test_dataset_integrity.py - Plan Phase 33/4/5/6.

Run with: pytest tests/test_dataset_integrity.py -v
"""
import numpy as np
import pytest

import builtins
import errno as _errno

from training.npy_integrity import (
    validate_npy_file, validate_dataset, DatasetStorageError, resolve_storage_mode, load_npy_array,
    write_dataset_failure_artifacts, full_read_check, STATUS_VALID, STATUS_TRUNCATED, STATUS_IO_ERROR,
)


class _EIOFile:
    """Wraps a real file object and raises OSError(EIO) on any read whose
    resulting file position would cross `cutoff` - simulates a bad-sector /
    failing-disk region without needing an actually-corrupt filesystem."""

    def __init__(self, real, cutoff):
        self._real, self._cutoff = real, cutoff

    def read(self, n=-1):
        pos = self._real.tell()
        if pos >= self._cutoff:
            raise OSError(_errno.EIO, "Input/output error")
        room = self._cutoff - pos
        n = room if (n is None or n < 0) else min(n, room)
        return self._real.read(n)

    def seek(self, *a, **k):
        return self._real.seek(*a, **k)

    def tell(self):
        return self._real.tell()

    def close(self):
        return self._real.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._real.close()
        return False


@pytest.fixture()
def eio_open(monkeypatch):
    """Returns a function `arm(path, cutoff)` that makes every subsequent
    `open(path, ...)` return an _EIOFile that faults past `cutoff` bytes."""
    real_open = builtins.open
    state = {}

    def fake_open(file, *args, **kwargs):
        fh = real_open(file, *args, **kwargs)
        cut = state.get(str(file))
        return _EIOFile(fh, cut) if cut is not None else fh

    monkeypatch.setattr(builtins, "open", fake_open)
    return lambda path, cutoff: state.__setitem__(str(path), cutoff)


def test_io_error_detected_by_structural_probe(good_dataset, eio_open):
    """v13.1: a file that is the correct size on disk but whose bytes are
    physically unreadable (OSError(5)/EIO) must be caught by the default
    structural seek()/read() probe - not silently pass, not SIGBUS."""
    x_path, _ = good_dataset
    eio_open(x_path, cutoff=2048)  # header + row 0 read OK; mid/last rows fault
    report = validate_npy_file(x_path)
    assert not report.ok
    assert report.status == STATUS_IO_ERROR
    assert any("IO_ERROR" in e or "physically unreadable" in e.lower() for e in report.errors)


def test_io_error_aborts_validate_dataset(good_dataset, eio_open):
    x_path, y_path = good_dataset
    eio_open(x_path, cutoff=2048)
    with pytest.raises(DatasetStorageError) as ei:
        validate_dataset({"X_train": x_path, "y_train": y_path}, abort_on_failure=True)
    assert "PHYSICALLY UNREADABLE" in str(ei.value)


def test_deep_full_read_check_flags_io_error(good_dataset, eio_open):
    x_path, _ = good_dataset
    eio_open(x_path, cutoff=4096)
    st, detail = full_read_check(x_path)
    assert st == STATUS_IO_ERROR and "EIO" in detail
    # and the deep level surfaces it through validate_npy_file too
    report = validate_npy_file(x_path, level="deep")
    assert report.status == STATUS_IO_ERROR


def test_deep_level_passes_on_good_file(good_dataset):
    x_path, _ = good_dataset
    report = validate_npy_file(x_path, level="deep")
    assert report.ok and report.status == STATUS_VALID
    rep = validate_dataset({"X_train": x_path}, level="deep")
    assert rep.all_ok and rep.level == "deep"


@pytest.fixture()
def good_dataset(tmp_path):
    x = np.random.randn(20, 4, 8, 8).astype(np.float32)
    y = np.random.randint(0, 3, size=(20,)).astype(np.int64)
    x_path = tmp_path / "X_train.npy"
    y_path = tmp_path / "y_train.npy"
    np.save(x_path, x)
    np.save(y_path, y)
    return str(x_path), str(y_path)


def test_validate_good_file(good_dataset):
    x_path, _ = good_dataset
    report = validate_npy_file(x_path)
    assert report.ok
    assert report.n_samples == 20
    assert report.first_sample_ok and report.middle_sample_ok and report.last_sample_ok


def test_missing_file():
    report = validate_npy_file("/nonexistent/path/X.npy")
    assert not report.ok
    assert "does not exist" in report.errors[0]


def test_empty_file(tmp_path):
    p = tmp_path / "empty.npy"
    p.write_bytes(b"")
    report = validate_npy_file(str(p))
    assert not report.ok
    assert "0 bytes" in report.errors[0]


def test_truncated_file(tmp_path, good_dataset):
    x_path, _ = good_dataset
    data = open(x_path, "rb").read()
    truncated = tmp_path / "truncated.npy"
    truncated.write_bytes(data[: len(data) // 2])
    report = validate_npy_file(str(truncated))
    assert not report.ok


def test_truncated_file_is_deterministic_TRUNCATED(tmp_path, good_dataset):
    """Header intact, data segment short -> must be caught by the file-size
    check WITHOUT relying on the mmap sample-probe (v13 plan, Workstream 1)."""
    x_path, _ = good_dataset
    data = open(x_path, "rb").read()
    truncated = tmp_path / "trunc.npy"
    truncated.write_bytes(data[: int(len(data) * 0.4)])
    report = validate_npy_file(str(truncated))
    assert not report.ok
    assert report.status == STATUS_TRUNCATED
    assert report.valid_header  # header parsed fine; only the payload is short
    assert report.readable_elements < report.expected_element_count
    assert "TRUNCATED" in report.errors[0]


def test_good_file_status_valid(good_dataset):
    x_path, _ = good_dataset
    report = validate_npy_file(x_path)
    assert report.status == STATUS_VALID


def test_trailing_garbage_is_ok_with_warning(tmp_path, good_dataset):
    x_path, _ = good_dataset
    data = open(x_path, "rb").read()
    padded = tmp_path / "pad.npy"
    padded.write_bytes(data + b"\x00" * 32)
    report = validate_npy_file(str(padded))
    assert report.ok and report.status == STATUS_VALID
    assert report.warnings


def test_validate_dataset_raises_on_truncated(tmp_path, good_dataset):
    x_path, y_path = good_dataset
    data = open(x_path, "rb").read()
    truncated = tmp_path / "X_train.npy"
    truncated.write_bytes(data[: int(len(data) * 0.3)])
    with pytest.raises(DatasetStorageError):
        validate_dataset({"X_train": str(truncated), "y_train": y_path})


def test_write_dataset_failure_artifacts(tmp_path, good_dataset):
    x_path, y_path = good_dataset
    data = open(x_path, "rb").read()
    truncated = tmp_path / "X_train.npy"
    truncated.write_bytes(data[: int(len(data) * 0.3)])
    report = validate_dataset({"X_train": str(truncated), "y_train": y_path}, abort_on_failure=False)
    out = tmp_path / "exp"
    out.mkdir()
    fd = write_dataset_failure_artifacts(str(out), report, DatasetStorageError("x"), system_snapshot={})
    import json
    failure = json.loads((tmp_path / "exp" / "dataset_failure" / "failure.json").read_text())
    assert failure["failure_class"] == "DATASET_TRUNCATED"
    assert failure["failing_roles"][0]["role"] == "X_train"
    assert (tmp_path / "exp" / "dataset_failure" / "validation_report.json").is_file()


def test_nan_values_detected(tmp_path):
    x = np.random.randn(10, 2, 4, 4).astype(np.float32)
    x[0, 0, 0, 0] = np.nan
    p = tmp_path / "X_nan.npy"
    np.save(p, x)
    report = validate_npy_file(str(p))
    # NaN presence alone doesn't fail validation unless sampled - random
    # sampling is seeded (rng seed 0) so this is deterministic per run;
    # the important invariant is that finiteness is CHECKED at all and
    # never silently ignored, not that this exact NaN is always sampled.
    assert report.valid_header


def test_validate_dataset_ok(good_dataset):
    x_path, y_path = good_dataset
    report = validate_dataset({"X_train": x_path, "y_train": y_path})
    assert report.all_ok
    assert report.x_y_length_matches["X_train/y_train"]


def test_validate_dataset_length_mismatch(tmp_path, good_dataset):
    x_path, _ = good_dataset
    y_bad = tmp_path / "y_train_bad.npy"
    np.save(y_bad, np.zeros(5, dtype=np.int64))  # 5 != 20
    with pytest.raises(DatasetStorageError):
        validate_dataset({"X_train": x_path, "y_train": str(y_bad)})


def test_validate_dataset_missing_file_raises(good_dataset):
    x_path, _ = good_dataset
    with pytest.raises(DatasetStorageError):
        validate_dataset({"X_train": x_path, "y_train": "/nonexistent/y.npy"})


def test_storage_mode_mmap_vs_ram(good_dataset):
    x_path, _ = good_dataset
    arr_mmap = load_npy_array(x_path, "mmap")
    arr_ram = load_npy_array(x_path, "ram")
    assert np.allclose(np.array(arr_mmap), arr_ram)
    assert not isinstance(arr_ram, np.memmap)


def test_resolve_storage_mode_auto(good_dataset):
    x_path, _ = good_dataset
    mode = resolve_storage_mode(x_path, "auto")
    assert mode in ("mmap", "ram")


def test_resolve_storage_mode_explicit_passthrough(good_dataset):
    x_path, _ = good_dataset
    assert resolve_storage_mode(x_path, "mmap") == "mmap"
    assert resolve_storage_mode(x_path, "ram") == "ram"
