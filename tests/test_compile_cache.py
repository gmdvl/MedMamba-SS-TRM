# -*- coding: utf-8 -*-
"""
tests/test_compile_cache_v17.py
==================================
MedMamba-SS-TRM v17, Stage 12b - `training/compile_cache.py`.

The first version of that module redirected `TMPDIR` into `<repo>/.cache/compile/tmp` and
broke every DataLoader worker with `OSError: AF_UNIX path too long`, because
`multiprocessing` binds a Unix socket under `TMPDIR` and `sun_path` is 108 bytes. The
compiler was fine; the data pipeline died. `test_tmpdir_is_not_redirected_by_default` and
`test_af_unix_budget_rejects_the_path_that_broke_the_run` are that regression, pinned.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from training import compile_cache as cc


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
        monkeypatch.delenv(v, raising=False)


def test_tmpdir_is_not_redirected_by_default(tmp_path, monkeypatch):
    """THE regression. TMPDIR is process-wide and feeds multiprocessing's socket path."""
    # `tmp_path` is under /tmp, which on this machine is tmpfs, so the real `_usable`
    # correctly refuses it - see test_usable_rejects_a_memory_backed_target.
    monkeypatch.setattr(cc, "_usable", lambda p: True)
    monkeypatch.setattr(cc, "describe_compile_scratch",
                        lambda: {"path": "/tmp", "fstype": "tmpfs", "free_gb": 2.6,
                                 "memory_backed": True, "already_set": []})
    cc.ensure_compile_scratch_on_disk(preferred=str(tmp_path / "c"), verbose=False)
    assert "TMPDIR" not in os.environ
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(str(tmp_path))
    assert os.environ["TRITON_CACHE_DIR"].startswith(str(tmp_path))


def test_af_unix_budget_rejects_the_path_that_broke_the_run():
    """The real repository path, which produced a 111-byte socket address against a 107
    limit. Every candidate this returns must leave room for `/pymp-*/listener-*`."""
    root = Path("/data/dante_data/documents/Masters/courses/Thesis/g-medmamba")
    cands = cc._short_tmpdir_candidates(root)
    assert root / ".medmamba-ss-trm-tmp" not in cands, "the over-long repo-local path came back"
    assert cands, "walking up from the repo should find something short enough"
    for c in cands:
        assert len(str(c)) + cc._AF_UNIX_BUDGET <= 107, c


def test_tmpdir_opt_in_still_respects_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "describe_compile_scratch",
                        lambda: {"path": "/tmp", "fstype": "tmpfs", "free_gb": 2.6,
                                 "memory_backed": True, "already_set": []})
    monkeypatch.setattr(cc, "_short_tmpdir_candidates", lambda root: [])   # nothing short enough
    cc.ensure_compile_scratch_on_disk(preferred=str(tmp_path / "c"), tmpdir=True, verbose=False)
    assert "TMPDIR" not in os.environ, "opted in, but no safe path - must decline, not break IPC"


def test_each_variable_is_independent(tmp_path, monkeypatch):
    """A caller who set one variable should not suppress the others."""
    monkeypatch.setenv("TRITON_CACHE_DIR", "/mine")
    monkeypatch.setattr(cc, "_usable", lambda p: True)
    monkeypatch.setattr(cc, "describe_compile_scratch",
                        lambda: {"path": "/tmp", "fstype": "tmpfs", "free_gb": 2.6,
                                 "memory_backed": True, "already_set": ["TRITON_CACHE_DIR"]})
    cc.ensure_compile_scratch_on_disk(preferred=str(tmp_path / "c"), verbose=False)
    assert os.environ["TRITON_CACHE_DIR"] == "/mine"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(str(tmp_path))


def test_noop_when_scratch_is_already_on_real_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "describe_compile_scratch",
                        lambda: {"path": "/var/tmp", "fstype": "btrfs", "free_gb": 500.0,
                                 "memory_backed": False, "already_set": []})
    assert cc.ensure_compile_scratch_on_disk(preferred=str(tmp_path / "c"), verbose=False) is None
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ


def test_never_raises_when_the_target_is_unusable(monkeypatch):
    """An optimisation must never be what kills a run."""
    monkeypatch.setattr(cc, "describe_compile_scratch",
                        lambda: {"path": "/tmp", "fstype": "tmpfs", "free_gb": 2.6,
                                 "memory_backed": True, "already_set": []})
    assert cc.ensure_compile_scratch_on_disk(preferred="/proc/nope/cannot", verbose=False) is None
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ


def test_memory_backed_filesystems_are_detected():
    assert cc._fstype("/tmp") == "tmpfs"          # this machine; the failure's root cause
    assert cc._fstype("/tmp") in cc._MEMORY_BACKED


def test_usable_rejects_a_memory_backed_target(tmp_path):
    """Redirecting one tmpfs onto another buys nothing, and `df` would not reveal it.
    pytest's own `tmp_path` is under /tmp here, which makes it the perfect fixture."""
    if cc._fstype(str(tmp_path)) not in cc._MEMORY_BACKED:
        pytest.skip("this machine's pytest tmp_path is not on a memory-backed filesystem")
    assert cc._usable(tmp_path / "x") is False


def test_usable_accepts_the_repository_disk():
    """The default target must actually pass the probe on this machine, or the module is
    decorative."""
    assert cc._usable(Path(__file__).resolve().parent.parent / ".cache" / "probe-test") is True


def test_warns_when_an_inherited_tmpdir_would_break_workers(monkeypatch, capsys):
    """The mirror of the shipped bug: a long TMPDIR arriving from the caller's shell.
    Same symptom, same misdirecting traceback - so say so before the workers start."""
    monkeypatch.setenv("TMPDIR", "/" + "x" * 90)
    assert cc.warn_if_tmpdir_breaks_worker_ipc() is not None
    assert "AF_UNIX path too long" in capsys.readouterr().out


def test_no_warning_for_a_sane_tmpdir(monkeypatch, capsys):
    monkeypatch.setenv("TMPDIR", "/data/dante_data/.cache/tmp")
    assert cc.warn_if_tmpdir_breaks_worker_ipc() is None
    assert capsys.readouterr().out == ""


def test_no_warning_when_tmpdir_is_unset(monkeypatch):
    monkeypatch.delenv("TMPDIR", raising=False)
    assert cc.warn_if_tmpdir_breaks_worker_ipc() is None
