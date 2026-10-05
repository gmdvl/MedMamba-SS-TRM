# -*- coding: utf-8 -*-
"""
run_experiments.py
===================
Many MedMamba-SS-TRM runs from one sweep file.

    python run_experiments.py sweeps/paper_evidence_upgrade.py --list              # stages, groups, jobs
    python run_experiments.py sweeps/paper_evidence_upgrade.py p1                  # a group of stages, in order
    python run_experiments.py sweeps/paper_evidence_upgrade.py headline depth      # individual stages
    python run_experiments.py sweeps/paper_evidence_upgrade.py headline --dry_run  # resolve every train job, run nothing
    python run_experiments.py sweeps/paper_evidence_upgrade.py --status            # done / waiting / missing
    python run_experiments.py sweeps/paper_evidence_upgrade.py depth --jobs 2      # two runs at a time
    python run_experiments.py --find v20-head-s42 --data data/hsi_v9-trainsel/hsi   # print a run dir

A sweep file is Python. It defines `STAGES = {name: [job, ...]}` and optionally
`GROUPS = {name: [stage, ...]}`, building jobs with `train(...)` and `cmd(...)`
from this module. Python rather than YAML because a sweep is seed loops, a
shared reference recipe and a few conditions, and because a test can import it.

Resumable: a train job is done when an `experiments/*/config.json` records its
`run_tag` (and data dir) and the run finished (`test_predictions.npz`); a cmd
job is done when its `done_if` file exists. After an interruption, re-run the
same command. A failed job is reported and the sweep moves on.

Every job is its own process. Some state is still per-process (seeding, compile
caches, CUDA memory), and an out-of-memory crash must not end the sweep.
`--jobs N` runs up to N jobs of the SAME stage at once - stages stay in order,
because later stages read earlier stages' runs. Concurrent runs share the GPU:
their wall clock and ms/step are not comparable to a solo run's.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

ROOT = Path(__file__).resolve().parent
PY = os.environ.get("PY", sys.executable)

Argv = Union[List[str], Callable[[], List[str]]]


@dataclass
class Job:
    name: str
    argv: Argv                                  # a callable is resolved when the job runs
    done: Callable[[], bool]
    kind: str = "cmd"                           # "train" | "cmd"
    cwd: Optional[str] = None
    requires: Callable[[], Optional[str]] = field(default=lambda: None)   # reason it cannot run yet

    def resolved_argv(self) -> List[str]:
        return list(self.argv() if callable(self.argv) else self.argv)


# ============================================================================
# finding runs
# ============================================================================

def find_runs(tag: str, data: Optional[str] = None, arch: Optional[str] = None,
              need: Optional[str] = "test_predictions.npz") -> List[Path]:
    """Run directories whose config.json records `run_tag == tag` (and a data_dir
    containing `data`, and architecture `arch`), holding `need` - the last file a
    finished run writes. Oldest first."""
    hits = []
    for c in sorted(glob.glob(str(ROOT / "experiments/*/config.json"))):
        try:
            cfg = json.load(open(c))
        except Exception:
            continue
        a = cfg.get("cli_args", {})
        if a.get("run_tag") != tag:
            continue
        if data and data.rstrip("/") not in str(a.get("data_dir", "")):
            continue
        if arch and cfg.get("architecture", a.get("arch")) != arch:
            continue
        d = Path(c).parent
        if need is None or (d / need).exists():
            hits.append(d)
    return hits


def run_dir(tag: str, data: Optional[str] = None, arch: Optional[str] = None) -> Optional[Path]:
    """The newest FINISHED run for `tag`, or None."""
    hits = find_runs(tag, data, arch)
    return hits[-1] if hits else None


# ============================================================================
# job builders (used by sweep files)
# ============================================================================

def flags_to_argv(flags: Dict) -> List[str]:
    """{lr: 1e-4, no_use_wavelengths: True, keep_bands: [32, 16]} ->
    ['--lr', '0.0001', '--no_use_wavelengths', '--keep_bands', '32', '16'].
    False / None drop the flag."""
    out = []
    for k, v in flags.items():
        if v is None or v is False:
            continue
        if v is True:
            out.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            out += [f"--{k}", *map(str, v)]
        else:
            out += [f"--{k}", str(v)]
    return out


def train(tag: str, data: str, name: Optional[str] = None, script: str = "train.py",
          match_arch: Optional[str] = None, **flags) -> Job:
    """One training run, tagged `tag` on `data`. Keyword arguments are flags."""
    argv = [PY, script, "--data_dir", data, *flags_to_argv(flags), "--run_tag", tag]
    return Job(name or tag, argv, done=lambda: run_dir(tag, data, match_arch) is not None,
               kind="train")


def cmd(name: str, argv: Argv, done_if: Union[None, str, Path, Callable[[], bool]] = None,
        cwd: Optional[str] = None, requires: Optional[Callable[[], Optional[str]]] = None) -> Job:
    """Any other command. `done_if`: a path that exists once it has run, a
    callable, or None (always runs). `requires` returns a reason it cannot run
    yet (e.g. the run it evaluates does not exist), or None."""
    if done_if is None:
        done = lambda: False                                    # noqa: E731
    elif callable(done_if):
        done = done_if
    else:
        done = lambda: Path(done_if).exists()                   # noqa: E731
    return Job(name, argv, done=done, cwd=cwd, requires=requires or (lambda: None))


def needs_run(tag: str, data: Optional[str] = None) -> Callable[[], Optional[str]]:
    """A `requires` for jobs that read a finished run."""
    return lambda: None if run_dir(tag, data) else f"run {tag!r} on {data} missing"


# ============================================================================
# the runner
# ============================================================================

def _say(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M}] {msg}", flush=True)


def load_sweep(path: str) -> dict:
    sys.modules.setdefault("run_experiments", sys.modules[__name__])   # one Job class, not two
    sweep = runpy.run_path(path)
    if "STAGES" not in sweep:
        raise SystemExit(f"{path} defines no STAGES")
    return sweep


def expand(names: List[str], stages: dict, groups: dict) -> List[str]:
    out = []
    for n in names:
        for s in groups.get(n, [n]):
            if s not in stages:
                raise SystemExit(f"unknown stage {s!r}; stages: {', '.join(stages)}; "
                                 f"groups: {', '.join(groups) or '-'}")
            out.append(s)
    return out


def dry_run(job: Job) -> str:
    """For a train.py job: the resolved configuration, without running anything."""
    argv = job.resolved_argv()
    if job.kind != "train" or Path(argv[1]).name != "train.py":
        return "  " + " ".join(argv)
    import contextlib
    import io
    from training.train_banner import deviations
    from training.train_cli import resolve
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            parsed, _notes = resolve(argv[2:])
    except SystemExit:
        lines = err.getvalue().strip().splitlines()
        return f"  INVALID: {lines[-1] if lines else 'argument error'}"
    a = parsed.args
    missing = "" if Path(a.data_dir).is_dir() else "   DATA DIR MISSING"
    devs = "; ".join(d.split("  (")[0] for d in deviations(parsed)) or "no deviations"
    return (f"  --profile {parsed.profile}: {devs}\n"
            f"    -> epochs {a.epochs}, batch {a.batch_size}, lr {a.lr}, seed {a.seed}, "
            f"early_stopping {a.early_stopping}{missing}")


def _execute(job: Job, log_dir: Path) -> bool:
    log = log_dir / f"{job.name}.log"
    _say(f"run  {job.name}  ->  {log}")
    with open(log, "w") as f:
        rc = subprocess.run(job.resolved_argv(), cwd=job.cwd or ROOT, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        _say(f"FAIL {job.name} (exit {rc}, see {log})")
    return rc == 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", nargs="?", help="sweep file, e.g. sweeps/paper_evidence_upgrade.py")
    p.add_argument("stages", nargs="*", help="stages or groups to run, in order")
    p.add_argument("--find", metavar="TAG", help="print the newest finished run with this run_tag "
                   "(exit 1 if none) and stop")
    p.add_argument("--data", help="with --find: the run's data dir (substring match)")
    p.add_argument("--arch", help="with --find: the run's architecture")
    p.add_argument("--list", action="store_true", help="list stages, groups and jobs")
    p.add_argument("--status", action="store_true", help="done / waiting / to-run, runs nothing")
    p.add_argument("--dry_run", action="store_true", help="resolve every job, run nothing")
    p.add_argument("--jobs", type=int, default=1, help="jobs of one stage run at once (default 1)")
    p.add_argument("--log_dir", default=None, help="default logs/<sweep name>/")
    a = p.parse_args(argv)
    if a.find:
        d = run_dir(a.find, a.data, a.arch)
        if d is None:
            return 1
        print(d.relative_to(ROOT) if d.is_relative_to(ROOT) else d)
        return 0
    if not a.sweep:
        p.error("a sweep file is required (or --find TAG)")

    sweep = load_sweep(a.sweep)
    stages, groups = sweep["STAGES"], sweep.get("GROUPS", {})
    if a.list:
        for g, members in groups.items():
            print(f"group {g}: {' '.join(members)}")
        for s, jobs in stages.items():
            print(f"stage {s}: {' '.join(j.name for j in jobs) or '-'}")
        return 0
    selected = expand(a.stages or ([] if not a.status else list(stages)), stages, groups)
    if not selected:
        p.error("name at least one stage or group (see --list)")

    log_dir = Path(a.log_dir or ROOT / "logs" / Path(a.sweep).stem)
    if not (a.status or a.dry_run):
        log_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for stage in selected:
        todo = []
        for job in stages[stage]:
            if job.done():
                _say(f"{'done' if (a.status or a.dry_run) else 'skip'} {job.name}")
                continue
            reason = job.requires()
            if reason:
                _say(f"wait {job.name} ({reason})")
                continue
            todo.append(job)
        if a.status:
            for job in todo:
                _say(f"todo {job.name}")
            continue
        if a.dry_run:
            for job in todo:
                _say(f"would run {job.name}")
                report = dry_run(job)
                print(report, flush=True)
                if "INVALID:" in report:
                    failed.append(job.name)
            continue
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
            for job, ok in zip(todo, pool.map(lambda j: _execute(j, log_dir), todo)):
                if not ok:
                    failed.append(job.name)
    if failed:
        _say(f"FAILED jobs: {' '.join(failed)}")
        return 1
    if not (a.status or a.dry_run):
        _say("sweep finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
