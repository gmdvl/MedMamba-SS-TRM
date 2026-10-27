# -*- coding: utf-8 -*-
"""
test_minority_sweep.py
=======================
`sweeps/minority_augmentation.py` on the trainsel build.

Its references are the paper sweep's `v20-abl-base-s42` / `v20-head-s*`, and
the augmentation plan (Phase 10) reuses `minaug-nomix` / `minaug-default` as
its copy arms A0os / A1. The runner counts a job done once a run with its tag
exists on its data, so a shared tag must be one command, byte for byte.

Run with: pytest tests/test_minority_sweep.py -q
"""

from pathlib import Path

import numpy as np
import pytest

import run_experiments as rx
from training import train_profiles
from training.minority_augmentation import resolve_minority_classes
from training.train_cli import resolve

ROOT = Path(__file__).resolve().parent.parent
SWEEP = rx.load_sweep(str(ROOT / "sweeps/minority_augmentation.py"))
PAPER = rx.load_sweep(str(ROOT / "sweeps/paper_evidence_upgrade.py"))
N_TRAIN, N_DCIS, HSI = SWEEP["N_TRAIN"], SWEEP["N_DCIS"], SWEEP["HSI"]


def _argv(sweep, stage, name):
    return next(j.resolved_argv() for j in sweep["STAGES"][stage] if j.name == name)


def _args(name):
    return resolve(_argv(SWEEP, "screen", name)[2:])[0].args


def test_counts_match_the_trainsel_labels():
    """N_TRAIN / N_DCIS are hard-coded so the sweep loads without data; every
    frac is computed from them. 19.6 MB of int64, memory-mapped."""
    path = ROOT / HSI / "y_train.npy"
    if not path.is_file():
        pytest.skip(f"no {HSI}/y_train.npy on this machine")
    y = np.load(path, mmap_mode="r")
    assert y.shape == (N_TRAIN,)
    assert np.bincount(y, minlength=3)[1] == N_DCIS
    assert resolve_minority_classes("auto", y, 3) == [1]      # why no `--minority_classes DCIS`


def test_references_are_the_paper_sweeps_runs():
    paper = {j.resolved_argv()[-1]: j.resolved_argv() for jobs in PAPER["STAGES"].values() for j in jobs
             if j.kind == "train" and not callable(j.argv) and j.resolved_argv()[3] == HSI}
    refs = [("screen", "v20-abl-base-s42")] + [("seeds", f"v20-head-s{s}") for s in (1, 7, 13, 23, 42)]
    for stage, tag in refs:
        assert _argv(SWEEP, stage, tag) == paper[tag], tag


def test_copy_arms_match_the_plan():
    """A0os = `minaug-nomix`, A1 = `minaug-default`: factor 2, and a frac that
    keeps A0's 250,113 rows per epoch over the 2,574,936 oversampled rows."""
    a0 = _args("v20-abl-base-s42")
    assert round(N_TRAIN * a0.train_subsample_frac) == 250113
    for tag, mix_prob in (("minaug-nomix", 0.0), ("minaug-default", 0.5)):
        a = _args(tag)
        assert (a.minority_augment, a.minority_oversample_factor) == ("on", 2.0), tag
        assert a.train_subsample_frac == 0.0971337, tag
        assert round((N_TRAIN + N_DCIS) * a.train_subsample_frac) == 250113, tag
        assert a.minority_mix_prob == mix_prob, tag
        assert (a.minority_class_weights, a.minority_classes) == ("effective", "auto"), tag


def test_no_job_pins_what_the_shared_tags_leave_to_the_profile(monkeypatch):
    """No `--trm_ema_rate` / `--minority_classes`: the shared tags need identical
    flags, and the parse-time EMA rate (from y_train's length, not the
    oversampled one: 931 vs 978 steps per epoch) clamps to 0.9995 in every arm."""
    for job in SWEEP["STAGES"]["screen"] + SWEEP["STAGES"]["seeds"]:
        argv = job.resolved_argv()
        assert "--trm_ema_rate" not in argv and "--minority_classes" not in argv, job.name
    monkeypatch.setattr(train_profiles, "split_size",
                        lambda d, s: {"train": N_TRAIN, "val": 334516}.get(s) if d else None)
    rates = {j.name: resolve(j.resolved_argv()[2:])[0].args.trm_ema_rate for j in SWEEP["STAGES"]["screen"]}
    assert set(rates.values()) == {0.9995}, rates
