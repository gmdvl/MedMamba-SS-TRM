# -*- coding: utf-8 -*-
"""
scripts/run_top3.py
====================
Re-run the three best MedMamba-SS-TRM experiments (test balanced accuracy, runs
with reconstruction images) on any dataset. The ranking and the recorded
parameters are in documentations/18_top_runs.md.

    python scripts/run_top3.py --dry_run                                   # resolve, run nothing
    python scripts/run_top3.py                                             # all three, paper build
    python scripts/run_top3.py --data_dir data/hsi_v9-trainsel/hsi         # another dataset
    python scripts/run_top3.py --only ref20 depth-n2 --data_dir data/lusc_v20/hsi
    python scripts/run_top3.py --status                                    # done / to run

    rank  experiment     original run        what differs from the reference
    1     depth-n2       20260915_150351     --trm_n_improve 2 --epochs 12
    2     ref20          20260915_031356     - (the reference, 20 epochs)
    3     ref20-v16      20260914_181738     ref20's recipe, run through
                                             train_example_v16_optimal_recon.py; only the
                                             loader / dataset-check flags differ

Each run is `train.py --profile paper_recipe` with those flags. On the paper build
(data/hsi_v8-80_10_10_importance-new/hsi) the commands resolve to exactly the
`cli_args` the original runs recorded.

--subsample: `paper` types the fractions the paper runs typed (0.102 / 0.0986,
i.e. ~250k train and ~33k validation patches per epoch on the paper build).
`auto` (the default) leaves them out, so the profile sizes the epoch for the
dataset given - ~250k / ~33k patches, or the whole split when it is smaller. On
the paper build `auto` resolves to the same 0.102 / 0.0986.

Run tags are `top3-<experiment>` (change with --tag_prefix), so these runs never
count as the paper's own `ref20` / `depth-n2` runs in sweeps/paper_runs.py.
Resumable: a run already finished with the same tag on the same data dir is
skipped. Runs go one at a time (host RAM fits two at most); logs go to
logs/top3/<data dir name>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_experiments import ROOT, _execute, _say, dry_run, train  # noqa: E402

PAPER_DATA = "data/hsi_v8-80_10_10_importance-new/hsi"
PAPER_FRACS = dict(train_subsample_frac=0.102, val_subsample_frac=0.0986)
REF = dict(profile="paper_recipe", batch_size=256, lambda_sam=0.1, seed=42)

# rank order; flags on top of REF
EXPERIMENTS = {
    "depth-n2": dict(epochs=12, trm_n_improve=2),
    "ref20": dict(epochs=20),
    "ref20-v16": dict(epochs=20, num_workers=2, no_pin_memory=True,
                      no_persistent_workers=True, dataset_validation_level="structural"),
}


def build_jobs(data_dir: str, only, subsample: str, tag_prefix: str):
    fracs = PAPER_FRACS if subsample == "paper" else {}
    return [train(f"{tag_prefix}{name}", data_dir, **REF, **fracs, **EXPERIMENTS[name])
            for name in EXPERIMENTS if not only or name in only]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=PAPER_DATA,
                   help=f"dataset build to train on (default: the paper build, {PAPER_DATA})")
    p.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=None,
                   help="run only these experiments (default: all three, in rank order)")
    p.add_argument("--subsample", choices=["auto", "paper"], default="auto",
                   help="per-epoch subsample fractions: 'auto' sizes them for --data_dir, "
                        "'paper' types 0.102 / 0.0986 (default: auto)")
    p.add_argument("--tag_prefix", default="top3-",
                   help="prefix of each run's --run_tag (default: top3-)")
    p.add_argument("--dry_run", action="store_true",
                   help="print each command and its resolved configuration, run nothing")
    p.add_argument("--status", action="store_true", help="done / to run, runs nothing")
    p.add_argument("--log_dir", default=None, help="default logs/top3/<data dir name>/")
    a = p.parse_args(argv)

    data_dir = a.data_dir.rstrip("/")
    if not (ROOT / data_dir).is_dir() and not Path(data_dir).is_dir():
        p.error(f"--data_dir {data_dir} does not exist")
    jobs = build_jobs(data_dir, a.only, a.subsample, a.tag_prefix)

    slug = data_dir.removeprefix("data/").replace("/", "_")
    log_dir = Path(a.log_dir or ROOT / "logs" / "top3" / slug)
    failed = []
    for job in jobs:
        if job.done():
            _say(f"{'done' if (a.status or a.dry_run) else 'skip'} {job.name} on {data_dir}")
            continue
        if a.status:
            _say(f"todo {job.name} on {data_dir}")
            continue
        if a.dry_run:
            print(f"{job.name}:\n  python " + " ".join(job.resolved_argv()[1:]))
            print(dry_run(job))
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        if not _execute(job, log_dir):
            failed.append(job.name)
    if failed:
        _say(f"{len(failed)} failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
