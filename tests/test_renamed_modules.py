# -*- coding: utf-8 -*-
"""
test_renamed_modules.py
=========================
On 2026-10-01 ten `training/*_vNN.py` modules lost their version suffix and
`npy_dataset_v16.py` was merged into `npy_data.py`. The frozen archived entry points
still import the old names, so `training/__init__.py` maps each old name onto the
current module. This pins that mapping: an old name must give back
the very same module object (so a monkeypatch through either name reaches both), and
no file may sit at an old name, where it would silently win over the mapping.

Run with: pytest tests/test_renamed_modules.py -q
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import training

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES = sorted(training.RENAMED_MODULES.items())


@pytest.mark.parametrize("old,new", CASES, ids=[old for old, _ in CASES])
def test_old_name_is_the_current_module(old, new):
    current = importlib.import_module(f"training.{new}")
    assert importlib.import_module(f"training.{old}") is current
    assert current.__spec__.name == f"training.{new}"
    assert Path(current.__file__).name == f"{new}.py"


@pytest.mark.parametrize("old,new", CASES, ids=[old for old, _ in CASES])
def test_only_the_current_name_exists_on_disk(old, new):
    assert (REPO_ROOT / "training" / f"{new}.py").is_file()
    assert not (REPO_ROOT / "training" / f"{old}.py").exists()


def test_old_names_resolve_in_a_fresh_interpreter():
    """The way an archived script meets them: first import, nothing preloaded."""
    code = ("import importlib, training\n"
            "for old, new in training.RENAMED_MODULES.items():\n"
            "    assert importlib.import_module('training.' + old) is "
            "importlib.import_module('training.' + new), old\n")
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
