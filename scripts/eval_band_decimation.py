#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/eval_band_decimation.py
=====================================
v18 E8 - the behavioural half of RQ1, from a checkpoint that already exists.

The question
------------
Manuscript section 7.1 proves band-count agnosticism *structurally*: the same
model at 32 and at 3 bands instantiates to identically 446,409 parameters. The
section says so itself - "it is arithmetic rather than statistics" - and then
concedes that "behavioural agnosticism is expensive and was not attempted".

It is not expensive. A model whose parameter set does not know how many bands
it is reading can be handed fewer bands AT TEST TIME with no retraining, no
new dataset and no architectural change. This script does that with the
trained 32-band checkpoint: minutes, not hours.

What it measures, precisely
---------------------------
ZERO-SHOT transfer to a narrower sensor - not "a model trained on C bands".
Those are different claims and only the first is nearly free; E10's retrained
arms answer the second, over the SAME band subsets so the two read as one
curve with two conditions.

Slicing happens AFTER normalization, and that is deliberate. Under
`global_zscore` the per-channel mean/std are fitted on the full 32-channel
training array, so slicing afterwards hands the model byte-identical values
for the bands it keeps. Normalizing a 16-band cube from its own statistics
would change the values of the retained bands too, and would confound "fewer
bands" with "different preprocessing".

The C = 32 case is the identity and is the script's own regression test: it
must reproduce the run's `test_report.json` exactly. Run it first.

Where it writes
---------------
`<run_dir>/band_decimation/C{n}/test_report.json` - NEVER `<run_dir>` itself.
`TrainerG_v11.evaluate_test_split` writes `test_report.json` and
`test_predictions.npz` into `self.exp_dir` (l. 722, 724) and `TrainerG_v12`
rewrites the report at l. 875, so pointing `exp_dir` at the run directory
would overwrite the artefact sections 7.1-7.3 cite. A summary lands at
`<run_dir>/band_decimation/summary.{json,csv}`.

Usage
-----
    python scripts/eval_band_decimation.py --run_dir experiments/<32-band run>
    python scripts/eval_band_decimation.py --run_dir <run> --keep_bands 32 --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.npy_data import NpyDatasetV16
from training.dataloader_config_v16 import train_val_policies_v16
from training.trainerg_v12 import TrainerG_v12
from training.numerical_stability import StabilityConfig

from scripts.eval_test_split import _rebuild_model  # noqa: E402


class BandSlicedDataset(Dataset):
    """Wraps a dataset yielding `(patch[C,H,W], label[, norm_stats])` and keeps
    `band_indices` along the channel axis.

    Post-normalization by construction: the wrapped dataset has already
    normalized with the full-band training statistics before this sees the
    tensor.
    """

    def __init__(self, base: Dataset, band_indices):
        self.base = base
        self.band_indices = torch.as_tensor(list(band_indices), dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        patch = item[0].index_select(0, self.band_indices)
        return (patch, *item[1:])


def decimate_indices(n_source: int, keep: int) -> np.ndarray:
    """`keep` band positions spread evenly over `n_source`, endpoints included.

    Uniform in INDEX, not in nanometres. On this sensor those differ sharply -
    the 32 selected centres run 400.5-938.2 nm with eighteen at or below
    633.3 nm, then a 219.0 nm gap, then fourteen from 852.3 nm - so the kept
    set is recorded in nm alongside, and the choice is stated rather than
    implied.
    """
    if keep >= n_source:
        return np.arange(n_source)
    return np.unique(np.linspace(0, n_source - 1, keep).round().astype(int))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Zero-shot band decimation for a finished HSI run.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--keep_bands", default="32,16,8,4,2",
                   help="comma-separated band counts; 32 (the full set) is the identity check")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--checkpoint", default=None, help="default: <run_dir>/best_model.pt")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--dry_run", action="store_true",
                   help="resolve everything, build nothing heavy, print the band sets, stop")
    p.add_argument("--max_test_samples", type=int, default=None,
                   help="SMOKE TEST ONLY: evaluate this many test patches instead of all of them. "
                        "The numbers are then not comparable to anything and the identity check is "
                        "skipped; it exists to prove the forward runs at C != 32 without a GPU.")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cfg = json.load(open(run_dir / "config.json"))
    a = cfg["cli_args"]
    data_dir = Path(args.data_dir or a["data_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or a["batch_size"]
    keep_list = [int(v) for v in args.keep_bands.split(",") if v.strip()]

    x_test, y_test = data_dir / "X_test.npy", data_dir / "y_test.npy"
    if not (x_test.is_file() and y_test.is_file()):
        print(f"ERROR: no X_test.npy/y_test.npy under {data_dir}")
        return 2

    n_source = int(cfg["in_channels"])
    wl_path = data_dir / "wavelengths.npy"
    wavelengths_full = np.load(wl_path) if wl_path.is_file() else None

    print(f"[run]    {run_dir.name}")
    print(f"[data]   {data_dir}   C={n_source}   norm={a['normalization']}")
    print(f"[device] {device}   batch_size={batch_size}")
    for keep in keep_list:
        idx = decimate_indices(n_source, keep)
        if wavelengths_full is not None:
            nm = np.asarray(wavelengths_full)[idx]
            print(f"  C={keep:>2d} -> {len(idx)} bands, indices {idx.tolist()}")
            print(f"         nm {[round(float(v), 1) for v in nm]}")
        else:
            print(f"  C={keep:>2d} -> {len(idx)} bands, indices {idx.tolist()}")
    if args.dry_run:
        print("[dry-run] stopping before model construction.")
        return 0

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "best_model.pt")
    if not ckpt_path.is_file():
        print(f"ERROR: {ckpt_path} does not exist.")
        return 2

    global_mean = global_std = None
    if a["normalization"] == "global_zscore":
        from training.npy_data import compute_global_channel_stats
        global_mean, global_std = compute_global_channel_stats(
            str(data_dir / "X_train.npy"), sample_cap=a["norm_stats_sample_cap"], seed=a["seed"])

    base_test = NpyDatasetV16(str(x_test), str(y_test), storage_mode=a["dataset_storage"],
                              normalization=a["normalization"],
                              global_mean=global_mean, global_std=global_std,
                              return_norm_stats=False)
    if args.max_test_samples:
        rng = np.random.default_rng(a["seed"])
        n = min(args.max_test_samples, len(base_test))
        base_test = Subset(base_test, sorted(rng.choice(len(base_test), size=n, replace=False).tolist()))
        print(f"[SMOKE TEST] {n} of {len(base_test.dataset)} test patches - numbers below are NOT a result")
    _, val_policy = train_val_policies_v16(a["loader_mode"], a["num_workers"], a["prefetch_factor"],
                                           a["persistent_workers"], a["pin_memory"])

    out_root = run_dir / "band_decimation"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for keep in keep_list:
        idx = decimate_indices(n_source, keep)
        n_bands = len(idx)
        exp_dir = out_root / f"C{n_bands}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(BandSlicedDataset(base_test, idx), batch_size=batch_size, shuffle=False,
                            **val_policy.dataloader_kwargs(device, seed=a["seed"]))

        model = _rebuild_model(cfg)
        wavelengths = None if wavelengths_full is None else np.asarray(wavelengths_full)[idx]

        trainer = TrainerG_v12(
            model=model, train_loader=None, val_loader=loader, optimizer=None, scheduler=None,
            device=device, exp_dir=exp_dir, class_names=cfg["class_names"],
            config=cfg, wavelengths=wavelengths, checkpoint_metric=a["checkpoint_metric"],
            amp_mode=a["amp"], stability_cfg=StabilityConfig(),
            sensor_range=(tuple(a["sensor_range"]) if a.get("sensor_range") else None),
            use_wavelengths=a["use_wavelengths"],
            ema_rate=(a["trm_ema_rate"] if cfg["architecture"] == "recursive" else None),
            scheduler_interval=a.get("scheduler_interval", "step"))

        report = trainer.evaluate_test_split(loader, checkpoint_path=str(ckpt_path),
                                             num_classes=cfg["classes"])
        if report is None:
            print(f"  C={n_bands}: evaluate_test_split returned None - skipped")
            continue

        m = report["sklearn_metrics"]
        row = {"n_bands": n_bands,
               "band_indices": idx.tolist(),
               "wavelengths_nm": ([round(float(v), 2) for v in wavelengths]
                                  if wavelengths is not None else None),
               "accuracy": m["accuracy"],
               "balanced_accuracy": m["balanced_accuracy"],
               "f1_macro": m["f1_macro"],
               "per_class_f1": m["per_class_f1"],
               "report_path": str(exp_dir / "test_report.json")}
        rows.append(row)
        print(f"  C={n_bands:>2d}  acc {m['accuracy']:.4f}  bal {m['balanced_accuracy']:.4f}  "
              f"macroF1 {m['f1_macro']:.4f}  per-class F1 "
              f"{[round(v, 3) for v in m['per_class_f1']]}")

    summary = {
        "run": run_dir.name,
        "checkpoint": str(ckpt_path),
        "source_bands": n_source,
        "condition": "zero-shot: the 32-band checkpoint evaluated on decimated input, no retraining",
        "smoke_test_n": args.max_test_samples,
        "slicing": "after normalization, with the full-band training statistics",
        "band_selection": "uniform in index over the source band set (NOT uniform in nm)",
        "rows": rows,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if rows:
        with open(out_root / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_bands", "accuracy", "balanced_accuracy", "f1_macro"])
            for r in rows:
                w.writerow([r["n_bands"], f"{r['accuracy']:.6f}",
                            f"{r['balanced_accuracy']:.6f}", f"{r['f1_macro']:.6f}"])
    print(f"[written] {out_root/'summary.json'}  and  summary.csv")

    # The identity check, stated rather than left to the reader.
    full = next((r for r in rows if r["n_bands"] == n_source), None)
    if args.max_test_samples:
        print("[identity] skipped - --max_test_samples was used, so this is a smoke test")
    elif full is not None and (run_dir / "test_report.json").is_file():
        ref = json.load(open(run_dir / "test_report.json"))["sklearn_metrics"]
        delta = abs(full["f1_macro"] - ref["f1_macro"])
        # 1e-6 was too tight and reported a spurious DIFFERS. The identity case
        # feeds byte-identical data to identical weights, but the evaluation runs
        # under bf16 autocast in a fresh process, so kernel selection alone moves
        # macro-F1 by ~1e-5 over 348,894 patches. Measured on ref20: delta
        # 2.48e-05, with accuracy and balanced accuracy agreeing to four decimals.
        # A real slicing or wavelength bug is orders of magnitude larger than this
        # - the collapsed C=16 arm is 0.72 away - so 1e-4 separates them cleanly.
        verdict = ("MATCHES (within bf16 evaluation tolerance)" if delta < 1e-4
                   else f"DIFFERS by {delta:.2e}  <-- investigate")
        print(f"[identity] C={n_source} vs the run's own test_report.json: macro-F1 {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
