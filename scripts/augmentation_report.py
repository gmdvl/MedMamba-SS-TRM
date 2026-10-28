#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/augmentation_report.py
==============================
The HSI augmentation study's report (Plan §Phase 10, issue #34), from
artifacts only: training runs found by `run_tag`, the build's y/groups
sidecars, the pipeline's stage `done.json` files and `selection.json`. The
logic is `hsi_augment/report.py`; the statistics `hsi_augment/stats.py`.

Usage (repo root; CPU, minutes)
-------------------------------
    # screen + validation; test sealed
    python scripts/augmentation_report.py

    # after `scripts/augmentation_select.py` wrote selection.json and the seeds stage ran:
    python augment_pipeline.py report-tissue --data_dir data/hsi_v9-trainsel/hsi --out data/hsi_v9-trainsel-aug
    python scripts/augmentation_report.py --reveal_test

    -> reports/augmentation/<date>/{report.md, report.json, figures/*.png}

Test columns render only when selection.json exists AND --reveal_test is
passed. The report regenerates byte-identically from the same artifacts: the
date is the directory name only (`--date` to rebuild an earlier one).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsi_augment import report as R  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", default=R.DEFAULT_EXPERIMENTS, help="training run directories")
    p.add_argument("--data_dir", default=R.DEFAULT_DATA_DIR,
                   help="the build's array dir (y_/groups_ sidecars of val and test)")
    p.add_argument("--run_data", default=R.DEFAULT_RUN_DATA,
                   help="a run counts when its cli_args.data_dir contains this")
    p.add_argument("--aug_out", default=R.DEFAULT_AUG_OUT, help="augment_pipeline.py --out (stage done.json files)")
    p.add_argument("--selection", default=R.DEFAULT_SELECTION, help="selection.json of augmentation_select.py")
    p.add_argument("--arms_file", default=None,
                   help="JSON arm table (list of {key, family, screen_tag, seed_tag, bench_tag, generator, "
                        "description}) replacing the plan's; selection.json's 'arms' wins over both")
    p.add_argument("--out_root", default=R.DEFAULT_OUT_ROOT, help="the report goes to <out_root>/<date>/")
    p.add_argument("--date", default=None, help="YYYY-MM-DD directory name (default: today, UTC)")
    p.add_argument("--reveal_test", action="store_true",
                   help="show test columns; honoured only when selection.json exists")
    p.add_argument("--seeds", type=int, nargs="+", default=list(R.SEEDS))
    p.add_argument("--screen_epochs", type=int, nargs="+", default=list(R.SCREEN_EPOCHS))
    p.add_argument("--kept_expected", type=int, default=R.KEPT_EXPECTED)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = R.ReportConfig(experiments=a.experiments, data_dir=a.data_dir, run_data=a.run_data, aug_out=a.aug_out,
                         selection=a.selection, out_root=a.out_root, date=a.date, reveal_test=a.reveal_test,
                         arms_file=a.arms_file, seeds=tuple(a.seeds), screen_epochs=tuple(a.screen_epochs),
                         kept_expected=a.kept_expected, n_boot=a.n_boot, boot_seed=a.boot_seed)
    try:
        out = R.generate_report(cfg)
    except (R.ReportError, R.TestSealError) as e:
        print(f"[augmentation_report] {e}", file=sys.stderr)
        return 1
    print(f"[augmentation_report] wrote {out / R.REPORT_MD} and {out / R.REPORT_JSON}")
    if a.reveal_test and not os.path.isfile(a.selection):
        print(f"[augmentation_report] --reveal_test ignored: no {a.selection}; test stays sealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
