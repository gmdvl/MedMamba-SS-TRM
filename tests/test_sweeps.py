# -*- coding: utf-8 -*-
"""
test_sweeps.py
===============
The sweep files and `run_experiments.py`.

The strongest check is `test_sweep_reproduces_the_recorded_runs`: every
train job in `sweeps/paper_runs.py` is resolved exactly as `train.py` would resolve it
and compared, argument by argument, with the `cli_args` recorded in the
`config.json` of the run that produced the paper's number. That ties the new
entry point AND the transcribed commands to the evidence, not to the old code.
It skips on a machine without `experiments/`.

Run with: pytest tests/test_sweeps.py -q
"""

import json
from pathlib import Path

import pytest

import run_experiments as rx
from training.train_cli import resolve

ROOT = Path(__file__).resolve().parent.parent
SWEEPS = sorted((ROOT / "sweeps").glob("*.py"))


@pytest.fixture(scope="module", params=SWEEPS, ids=[p.stem for p in SWEEPS])
def sweep(request):
    return rx.load_sweep(str(request.param))


def _train_py_jobs(sweep):
    for jobs in sweep["STAGES"].values():
        for job in jobs:
            argv = job.resolved_argv() if not callable(job.argv) else None
            if job.kind == "train" and argv and Path(argv[1]).name == "train.py":
                yield job, argv


def test_job_names_are_unique(sweep):
    names = [j.name for jobs in sweep["STAGES"].values() for j in jobs]
    assert len(names) == len(set(names))


def test_groups_name_real_stages(sweep):
    for group, members in sweep.get("GROUPS", {}).items():
        assert set(members) <= set(sweep["STAGES"]), group


def test_every_train_job_parses(sweep):
    for job, argv in _train_py_jobs(sweep):
        parsed, _ = resolve(argv[2:])
        assert parsed.args.run_tag, job.name


def test_flags_to_argv():
    assert rx.flags_to_argv({"lr": 1e-4, "no_use_wavelengths": True, "x": None, "y": False,
                             "keep": [32, 16]}) == \
        ["--lr", "0.0001", "--no_use_wavelengths", "--keep", "32", "16"]


def _recorded_runs(tag, data):
    for c in sorted((ROOT / "experiments").glob("*/config.json")):
        try:
            cfg = json.loads(c.read_text())
        except Exception:
            continue
        a = cfg.get("cli_args", {})
        if a.get("run_tag") == tag and a.get("data_dir", "").rstrip("/") == data.rstrip("/"):
            yield c.parent.name, a


@pytest.mark.parametrize("name,minimum", [("paper_runs", 20), ("paper_evidence_upgrade", 1)])
def test_sweep_reproduces_the_recorded_runs(name, minimum):
    sweep = rx.load_sweep(str(ROOT / f"sweeps/{name}.py"))
    compared = 0
    for job, argv in _train_py_jobs(sweep):
        parsed, _ = resolve(argv[2:])
        new = vars(parsed.args)
        for run, recorded in _recorded_runs(parsed.args.run_tag, parsed.args.data_dir):
            diff = {k: (v, new.get(k)) for k, v in recorded.items() if new.get(k) != v}
            assert not diff, f"{job.name} vs {run}: {diff}"
            compared += 1
    if compared == 0:
        pytest.skip(f"no recorded {name} runs on this machine")
    assert compared >= minimum


def test_status_and_dry_run_run_nothing(capsys):
    assert rx.main([str(ROOT / "sweeps/paper_runs.py"), "depth", "--dry_run"]) == 0
    assert rx.main([str(ROOT / "sweeps/paper_runs.py"), "--status"]) == 0


def test_find_prints_a_run_or_exits_1(capsys):
    assert rx.main(["--find", "no-such-tag-anywhere"]) == 1
    assert capsys.readouterr().out == ""


def test_dry_run_fails_on_an_invalid_job(tmp_path, capsys):
    """`--dry_run` is the check to run before a multi-hour sweep: a job that would
    die in argparse must fail it, not print usage and stop."""
    spec = tmp_path / "bad.py"
    spec.write_text("from run_experiments import train\n"
                    "STAGES = {'s': [train('bad_tag', 'data/x', profile='paper_recipe'),\n"
                    "                train('good-tag', 'data/x', profile='paper_recipe')]}\n")
    assert rx.main([str(spec), "s", "--dry_run", "--log_dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "INVALID" in out and "would run good-tag" in out
