# -*- coding: utf-8 -*-
"""
test_frozen_files_untouched.py
=================================
MedMamba-SS-TRM v16 plan - THE INVARIANT. Green tests elsewhere show v15/v11 still
WORK; this is the only thing that shows they are UNCHANGED, which is what
keeps the two audited runs (`20260903_180812`, `20260904_010453`)
bit-reproducible.

`git diff --name-only <base>` intersected with the FROZEN list must be
empty. Default base is `HEAD` - override with the `MEDMAMBA_SS_TRM_FREEZE_BASE`
env var if a later commit needs to be the reference point (e.g. once
`training/losses.py`'s pre-existing `class_weight_power` work, which
predates this plan, is itself committed and HEAD moves past it).
"""

import os
import subprocess
from pathlib import Path

import pytest

# v17 S12 - `.parent.parent` since this file moved from the repository root into `tests/`.
# This is THE invariant test, so the path it resolves has to be the repository, not `tests/`:
# with the wrong root, `_git` would run outside the work tree and every FROZEN_GLOBS entry
# would silently stop resolving.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Verbatim from the v16 plan's Verification section.
FROZEN_GLOBS = [
    # v6/v7/v15 moved to archive/ on 2026-09-29; the pieces train.py still needs are verbatim
    # copies in training/, pinned by tests/test_live_copies_match_archive.py.
    "archive/train_example_v6.py", "archive/train_example_v7.py", "archive/train_example_v14.py",
    "archive/train_example_v15.py",
    "medmamba_ss_trm.py", "medmamba_ss_trm_ema.py", "medmamba_ss_fullchannel.py", "medmamba_ss_efficient.py",
    "archive/trainerg_v3.py", "training/trainerg_v4.py", "training/trainerg_v5.py",
    "training/trainerg_v6.py", "training/trainerg_v7.py", "training/trainerg_v8.py",
    "training/trainerg_v9.py", "training/trainerg_v10.py", "training/trainerg_v11.py",
    "training/gan.py", "training/eval_utils.py", "training/losses.py",
    "training/numerical_stability.py", "training/reconstruction_head.py",
    "training/spectral_recon_metrics.py", "training/plots.py", "training/augmentation.py",
    "training/dataloader_config.py", "training/run_naming.py",
    "scripts/shallow_baseline.py",
    # Unified into prepare_histologyhsi_bc.py / prepare_pad_ufes_20.py; the originals
    # now live, byte-for-byte unchanged, under archive/ and stay frozen there.
    "archive/prepare_histologyhsi_bc_v3.py", "archive/prepare_histologyhsi_bc_v4.py",
    "archive/prepare_histologyhsi_bc_v5.py", "archive/prepare_histologyhsi_bc_v6.py",
    "archive/prepare_histologyhsi_bc_v7.py", "archive/prepare_histologyhsi_bc_v8.py",
    "archive/prepare_pad_ufes_20_v4.py", "archive/prepare_pad_ufes_20_v5.py",
    "archive/prepare_pad_ufes_20_v6.py",
    # Replaced by train.py --profile (2026-09-28). The reference the parity tests
    # (tests/test_train_parity.py, tests/test_train_e2e_parity.py) compare against,
    # and train_example_v18.py produced every number in the paper.
    "archive/train_example_v16.py", "archive/train_example_v16_recon.py",
    "archive/train_example_v16_original.py", "archive/train_example_v16_optimal.py",
    "archive/train_example_v16_optimal_recon.py", "archive/train_example_v18.py",
]

FROZEN_DIR_PREFIXES = ["training/prep/"]


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout


def _is_git_repo() -> bool:
    try:
        _git("rev-parse", "--is-inside-work-tree")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@pytest.mark.skipif(not _is_git_repo(), reason="not a git repository")
def test_no_frozen_file_is_modified_relative_to_base():
    base = os.environ.get("MEDMAMBA_SS_TRM_FREEZE_BASE", "HEAD")
    changed = set(_git("diff", "--name-only", base).splitlines())
    changed |= set(_git("diff", "--name-only", "--cached", base).splitlines())

    hit_globs = [f for f in FROZEN_GLOBS if f in changed]
    hit_prefixes = [c for c in changed if any(c.startswith(p) for p in FROZEN_DIR_PREFIXES)]

    assert not hit_globs and not hit_prefixes, (
        f"FREEZE VIOLATION: the following files on the FROZEN list differ from {base!r}: "
        f"{hit_globs + hit_prefixes}. No file that train_example_v15.py or TrainerG_v11 imports "
        f"may be edited - see the v16 plan's freezing rule.")


@pytest.mark.skipif(not _is_git_repo(), reason="not a git repository")
def test_frozen_list_files_actually_exist():
    """A stale/typo'd path in FROZEN_GLOBS would make the invariant test
    above vacuously pass for that file - catch that here."""
    missing = [f for f in FROZEN_GLOBS if not (REPO_ROOT / f).is_file()]
    assert not missing, f"FROZEN_GLOBS lists file(s) that do not exist: {missing}"


if __name__ == "__main__":
    test_no_frozen_file_is_modified_relative_to_base()
    test_frozen_list_files_actually_exist()
    print("OK")
