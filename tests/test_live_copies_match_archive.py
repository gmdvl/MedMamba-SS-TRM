# -*- coding: utf-8 -*-
"""
test_live_copies_match_archive.py
===================================
`train_example_v6.py`, `_v7.py` and `_v15.py` moved to `archive/` on 2026-09-29. The
pieces `train.py` and the scripts still use were copied into `training/` instead of
being imported from `archive/`, which nothing live imports from. This test pins each
copy to its frozen original, comparing the source text exactly.

It matters most for `NpyDataset`. Both sides of `test_train_e2e_parity.py` build their
datasets through `NpyDatasetV16`, which now subclasses the copy, so that test cannot
tell a changed copy from the original. This one can.

If you change a copy on purpose, take it out of `COPIES` in the same commit.

The one sanctioned difference is the 2026-10-02 rename (GMedMamba -> MedMamba-SS /
MedMamba-SS-TRM): the copies carry the new names, the archived originals keep the old
ones. `RENAMES` is applied to the original before comparing, so anything else that
differs still fails.

Run with: pytest tests/test_live_copies_match_archive.py -q
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# archived original -> {top-level name in the original: the live file holding its copy}
COPIES = {
    "archive/train_example_v6.py": {
        "compute_global_channel_stats": "training/npy_data.py",
        "NpyDataset": "training/npy_data.py",
        "discover_data": "training/npy_data.py",
        "setup_experiment_dir": "training/npy_data.py",
    },
    "archive/train_example_v7.py": {
        "GMedMambaRawReconWrapper": "training/reconstruction_head_v2.py",
    },
    "archive/train_example_v15.py": {
        "EFFICIENT_FUSION_CHOICES": "training/train_cli.py",
        "build_arg_parser": "training/train_cli.py",
        "_add_v15_arguments": "training/train_cli.py",
        "apply_safe_mode": "training/train_cli.py",
        "print_preflight": "training/train_preflight.py",
        "run_loader_test": "training/train_preflight.py",
        "_probe_in_channels": "training/train_preflight.py",
        "_run_leakage_check": "training/train_preflight.py",
        "_check_manifest": "training/train_preflight.py",
    },
}

# old -> new, applied in this order to the archived source (the 2026-10-02 rename)
RENAMES = [
    ("GMedMambaRawReconWrapper", "MedMambaSSTRMRawReconWrapper"),
    ("GMedMambaConfig", "MedMambaSSTRMConfig"),
    ("gmedmamba_run", "medmamba_ss_trm_run"),
    ("gmedmamba.", "medmamba_ss_trm."),
    ("GMedMamba v15", "MedMamba-SS-TRM v15"),
    ("GMedMamba Stability Preflight", "MedMamba-SS-TRM Stability Preflight"),
]


def _renamed(text: str) -> str:
    for old, new in RENAMES:
        text = text.replace(old, new)
    return text


CASES = [(orig, name, live) for orig, names in COPIES.items() for name, live in names.items()]


def _top_level_sources(rel_path: str) -> dict:
    src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    out = {}
    for node in ast.parse(src).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name is not None:
            out[name] = ast.get_source_segment(src, node)
    return out


@pytest.mark.parametrize("original,name,live", CASES, ids=[name for _, name, _ in CASES])
def test_copy_matches_archived_original(original, name, live):
    want = _top_level_sources(original).get(name)
    got = _top_level_sources(live).get(_renamed(name))
    assert want is not None, f"{name} is no longer defined in {original}"
    assert got is not None, f"{name} is no longer defined in {live}"
    assert got == _renamed(want), f"{live}:{name} differs from the frozen {original}:{name}"


def test_train_py_imports_nothing_from_archive():
    """In a fresh interpreter with only the repository root on the path, a
    leftover `import train_example_v15` in the train.py chain raises here instead
    of resolving silently through a PYTHONPATH that happens to include archive/."""
    code = ("import sys, training.train_cli, training.train_pipeline; "
            "bad = sorted(m for m in sys.modules if m.startswith('train_example')); "
            "assert not bad, bad")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
