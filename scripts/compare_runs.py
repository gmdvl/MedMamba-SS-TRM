# -*- coding: utf-8 -*-
"""
scripts/compare_runs.py
=============================
Tabulate finished runs side by side, so an ablation is read as a table instead
of by opening N `test_report.json` files by hand.

Every analysis in `documentations/12_commands_datasets.md` beyond the first
three is a COMPARISON - band count 32/16/8/3, `--rgb_source` synthetic vs
camera, reconstruction on vs off, architecture recursive vs split vs
efficient, `--target_token_grid` 28 vs 56. None of them mean anything as a
single number, and nothing in the repository lined them up.

Read-only. It opens `config.json`, `test_report.json`, `history.json` and
(with `--per_patient`) `per_patient_metrics.json`, and writes nothing unless
`--csv` is given.

Usage
-----
    # a band-count sweep
    python scripts/compare_runs.py 'experiments/*bands*'

    # everything on one dataset, newest first
    python scripts/compare_runs.py --all --filter hsi_v8

    # the honest unit of analysis on the histology corpus: 45 patients,
    # not 348,894 patches
    python scripts/compare_runs.py --all --filter hsi --per_patient

    # machine-readable
    python scripts/compare_runs.py --all --csv results.csv

Columns
-------
`acc`/`bal`/`f1M` are TEST-split metrics from `test_report.json`
(`sklearn_metrics`), i.e. the held-out number, computed on the checkpoint the
run actually selected - not the best validation epoch. A run with no
`test_report.json` never reached gate G6 and is listed with `-`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys

_METRIC_KEYS = ("accuracy", "balanced_accuracy", "f1_macro")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _short(name: str, width: int) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def collect(run_dir: str) -> dict:
    cfg = _load(os.path.join(run_dir, "config.json")) or {}
    a = cfg.get("cli_args") or {}
    tr = _load(os.path.join(run_dir, "test_report.json")) or {}
    m = tr.get("sklearn_metrics") or {}
    hist = _load(os.path.join(run_dir, "history.json")) or []
    last = hist[-1] if hist else {}

    data_dir = (a.get("data_dir") or "").rstrip("/")
    row = {
        "run": os.path.basename(run_dir.rstrip("/")),
        "dataset": os.path.basename(os.path.dirname(data_dir)) + "/" + os.path.basename(data_dir)
                    if data_dir else "?",
        "modality": cfg.get("modality") or "?",
        "arch": a.get("architecture") or "?",
        "bands": cfg.get("in_channels"),
        "batch": a.get("batch_size"),
        "lr": a.get("lr"),
        "epochs_run": len(hist),
        "patch": a.get("patch_size"),
        "recon": a.get("recon_mode"),
        "lmse": a.get("lambda_mse"),
        "loss": a.get("loss"),
        "cwp": a.get("class_weight_power"),
        "subsample": a.get("train_subsample_frac"),
        "ckpt_epoch": tr.get("checkpoint_epoch"),
        "n_test": tr.get("n_test_samples"),
        "gpu_mb": last.get("GPU_memory_MB"),
        "s_per_ep": last.get("epoch_time"),
    }
    for k in _METRIC_KEYS:
        row[k] = m.get(k)

    pp = _load(os.path.join(run_dir, "per_patient_metrics.json"))
    if pp and pp.get("per_patient"):
        recalls = [v["recall_macro"] for v in pp["per_patient"].values()
                   if v.get("recall_macro") is not None]
        if recalls:
            row["pp_n"] = len(recalls)
            row["pp_mean"] = statistics.mean(recalls)
            row["pp_std"] = statistics.pstdev(recalls) if len(recalls) > 1 else 0.0
            row["pp_min"] = min(recalls)
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tabulate finished MedMamba-SS-TRM runs side by side.")
    p.add_argument("patterns", nargs="*", help="run directories or globs")
    p.add_argument("--all", action="store_true", help="every experiments/2*/ directory")
    p.add_argument("--filter", default=None, help="substring the run directory name must contain")
    p.add_argument("--sort", default="f1_macro",
                   choices=["f1_macro", "balanced_accuracy", "accuracy", "run"],
                   help="sort key, descending (except 'run', which is chronological)")
    p.add_argument("--per_patient", action="store_true",
                   help="add per-patient recall mean/std/min from per_patient_metrics.json. "
                        "Only the histology corpus has the groups_*.npy sidecars this needs; "
                        "PAD carries images_*.npy instead (image-level, see "
                        "--eval_group_aggregation).")
    p.add_argument("--csv", default=None, help="also write every collected field to this CSV")
    p.add_argument("--incomplete", action="store_true",
                   help="include runs that produced no test_report.json")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    dirs = []
    for pat in args.patterns:
        dirs.extend(sorted(glob.glob(pat)))
    if args.all or not args.patterns:
        dirs.extend(sorted(glob.glob("experiments/2*/")))
    dirs = [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]
    if args.filter:
        dirs = [d for d in dirs if args.filter in os.path.basename(d.rstrip("/"))]
    if not dirs:
        print("no run directories matched", file=sys.stderr)
        return 1

    rows = [collect(d) for d in dirs]
    if not args.incomplete:
        rows = [r for r in rows if r.get("f1_macro") is not None]
    if not rows:
        print("no runs with a test_report.json matched (use --incomplete to list them anyway)",
              file=sys.stderr)
        return 1

    if args.sort == "run":
        rows.sort(key=lambda r: r["run"])
    else:
        rows.sort(key=lambda r: (r.get(args.sort) is None, -(r.get(args.sort) or 0)))

    cols = [("run", 52), ("modality", 4), ("arch", 9), ("bands", 5), ("batch", 5),
            ("lr", 7), ("patch", 5), ("recon", 6), ("accuracy", 7), ("balanced_accuracy", 7),
            ("f1_macro", 7)]
    if args.per_patient:
        cols += [("pp_n", 5), ("pp_mean", 7), ("pp_std", 7), ("pp_min", 7)]
    head = {"balanced_accuracy": "bal", "accuracy": "acc", "f1_macro": "f1M",
            "pp_mean": "pp_avg", "pp_std": "pp_sd", "pp_min": "pp_min"}

    line = "  ".join(f"{head.get(c, c):>{w}}" if c not in ("run",) else f"{head.get(c, c):<{w}}"
                     for c, w in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = []
        for c, w in cols:
            v = r.get(c)
            if v is None:
                s = "-"
            elif isinstance(v, float):
                s = f"{v:.4f}" if c in _METRIC_KEYS or c.startswith("pp_") else f"{v:g}"
            else:
                s = str(v)
            cells.append(f"{_short(s, w):<{w}}" if c == "run" else f"{_short(s, w):>{w}}")
        print("  ".join(cells))

    if args.per_patient:
        print("\npp_n/pp_avg/pp_sd/pp_min = per-PATIENT macro recall on the validation split. "
              "On the histology corpus there are 45 patients against ~3.1M patches, and patches "
              "from one slide are not independent samples - quote pp_sd, not the patch-level "
              "metric, when you need a spread.")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows x {len(keys)} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
