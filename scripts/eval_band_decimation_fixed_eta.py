#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/eval_band_decimation_fixed_eta.py
=============================================
Zero-shot band decimation with the spectral position encoding held at its
TRAINING values - the control the manuscript names as missing (Sec. VI-C).

In the reported configuration the encoding of band c is

    lambda~_c = (lambda_c - min lambda) / (max lambda - min lambda)     (eq. 5)
    pi_c      = sin/cos(eta * lambda~_c * omega_i),  eta = C              (eq. 6)

so removing bands changes BOTH the normalization range (when an end band is
dropped) and eta (always). Every retained band then gets a code the 32-band
model never saw. `scripts/eval_band_decimation.py` measures that
condition. This script measures the other one:

    eta               fixed at the training band count  (C_train = 32)
    normalization     fixed at the training range, via the model's own
                      `sensor_range` argument (normalize_wavelengths clamps
                      against (lo, hi) = the training min/max)

At C = C_train both conditions are the identity, and this script checks that
it reproduces the run's own test_report.json exactly as the v18 script does.

Nothing is retrained and nothing frozen is edited: eta is set on the loaded
model's tokenizer attribute (`wavelength_encoding_scale`, None => C), and the
range is passed through the trainer's existing `sensor_range` argument.

Writes <run_dir>/band_decimation_fixed_eta/{C<n>/test_report.json, summary.json, summary.csv}.

Usage (host, GPU, ~0.5 h for five band counts):
    python scripts/eval_band_decimation_fixed_eta.py --run_dir experiments/<run> \
        --keep_bands 32,16,8,4,2
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_band_decimation import BandSlicedDataset, decimate_indices  # noqa: E402

OUT_NAME = "band_decimation_fixed_eta"


def fix_encoding(model: torch.nn.Module, eta: float) -> int:
    """Pin eta on every spectral tokenizer of `model`. Returns how many were
    changed. A tokenizer trained with an explicit numeric scale keeps it
    (then eta was never C-dependent and there is nothing to fix)."""
    n = 0
    for m in model.modules():
        if hasattr(m, "wavelength_encoding_scale") and hasattr(m, "d_token"):
            if m.wavelength_encoding_scale is None:
                m.wavelength_encoding_scale = float(eta)
                n += 1
    return n


def training_range(wavelengths_full) -> tuple:
    wl = np.asarray(wavelengths_full, dtype=np.float64)
    return float(wl.min()), float(wl.max())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--keep_bands", default="32,16,8,4,2")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--checkpoint", default=None, help="default: <run_dir>/best_model.pt")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None,
                   help="SMOKE TEST ONLY - numbers are not a result")
    args = p.parse_args(argv)

    # Heavy imports only past argument parsing (keeps --help instant).
    from training.npy_data import NpyDatasetV16
    from training.dataloader_config_v16 import train_val_policies_v16
    from training.trainerg_v12 import TrainerG_v12
    from training.numerical_stability import StabilityConfig
    from scripts.eval_test_split import _rebuild_model

    run_dir = Path(args.run_dir).resolve()
    cfg = json.load(open(run_dir / "config.json"))
    a = cfg["cli_args"]
    if not a.get("use_wavelengths", True):
        print("ERROR: this run uses the index encoding; eta/range fixing does not apply.")
        return 2
    if a.get("sensor_range"):
        print("ERROR: this run already normalizes against a fixed sensor_range; use the v18 script.")
        return 2
    data_dir = Path(args.data_dir or a["data_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or a["batch_size"]
    keep_list = [int(v) for v in args.keep_bands.split(",") if v.strip()]
    n_source = int(cfg["in_channels"])
    wavelengths_full = np.load(data_dir / "wavelengths.npy")
    fixed_range = training_range(wavelengths_full)
    eta = float(n_source) if a.get("wavelength_scale") is None else float(a["wavelength_scale"])
    print(f"[run] {run_dir.name}\n[fixed] eta = {eta:g}, normalization range = "
          f"{fixed_range[0]:.1f}-{fixed_range[1]:.1f} nm")

    global_mean = global_std = None
    if a["normalization"] == "global_zscore":
        from training.npy_data import compute_global_channel_stats
        global_mean, global_std = compute_global_channel_stats(
            str(data_dir / "X_train.npy"), sample_cap=a["norm_stats_sample_cap"], seed=a["seed"])
    base_test = NpyDatasetV16(str(data_dir / "X_test.npy"), str(data_dir / "y_test.npy"),
                              storage_mode=a["dataset_storage"], normalization=a["normalization"],
                              global_mean=global_mean, global_std=global_std, return_norm_stats=False)
    if args.max_test_samples:
        rng = np.random.default_rng(a["seed"])
        n = min(args.max_test_samples, len(base_test))
        base_test = Subset(base_test, sorted(rng.choice(len(base_test), size=n, replace=False).tolist()))
        print(f"[SMOKE TEST] {n} patches - NOT a result")
    _, val_policy = train_val_policies_v16(a["loader_mode"], a["num_workers"], a["prefetch_factor"],
                                           a["persistent_workers"], a["pin_memory"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "best_model.pt")

    out_root = run_dir / OUT_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for keep in keep_list:
        idx = decimate_indices(n_source, keep)
        exp_dir = out_root / f"C{len(idx)}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(BandSlicedDataset(base_test, idx), batch_size=batch_size, shuffle=False,
                            **val_policy.dataloader_kwargs(device, seed=a["seed"]))
        model = _rebuild_model(cfg)
        n_fixed = fix_encoding(model, eta)
        wavelengths = np.asarray(wavelengths_full)[idx]
        trainer = TrainerG_v12(
            model=model, train_loader=None, val_loader=loader, optimizer=None, scheduler=None,
            device=device, exp_dir=exp_dir, class_names=cfg["class_names"], config=cfg,
            wavelengths=wavelengths, checkpoint_metric=a["checkpoint_metric"], amp_mode=a["amp"],
            stability_cfg=StabilityConfig(), sensor_range=fixed_range,
            use_wavelengths=a["use_wavelengths"],
            ema_rate=(a["trm_ema_rate"] if cfg["architecture"] == "recursive" else None),
            scheduler_interval=a.get("scheduler_interval", "step"))
        # evaluate_test_split reloads weights only; eta is a plain attribute
        # (not a parameter or buffer), so the reload leaves it pinned.
        report = trainer.evaluate_test_split(loader, checkpoint_path=str(ckpt_path),
                                             num_classes=cfg["classes"])
        if report is None:
            print(f"  C={len(idx)}: no report - skipped")
            continue
        m = report["sklearn_metrics"]
        rows.append({"n_bands": len(idx), "band_indices": idx.tolist(),
                     "wavelengths_nm": [round(float(v), 2) for v in wavelengths],
                     "tokenizers_pinned": n_fixed, "accuracy": m["accuracy"],
                     "balanced_accuracy": m["balanced_accuracy"], "f1_macro": m["f1_macro"],
                     "per_class_f1": m["per_class_f1"]})
        print(f"  C={len(idx):>2d}  acc {m['accuracy']:.4f}  bal {m['balanced_accuracy']:.4f}  "
              f"macroF1 {m['f1_macro']:.4f}")

    summary = {"run": run_dir.name, "checkpoint": str(ckpt_path), "source_bands": n_source,
               "condition": "zero-shot, eta and normalization range fixed at training values",
               "eta": eta, "normalization_range_nm": fixed_range,
               "smoke_test_n": args.max_test_samples, "rows": rows}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_bands", "accuracy", "balanced_accuracy", "f1_macro"])
        for r in rows:
            w.writerow([r["n_bands"], r["accuracy"], r["balanced_accuracy"], r["f1_macro"]])
    full = next((r for r in rows if r["n_bands"] == n_source), None)
    if full is not None and not args.max_test_samples and (run_dir / "test_report.json").is_file():
        ref = json.load(open(run_dir / "test_report.json"))["sklearn_metrics"]["f1_macro"]
        verdict = "OK" if abs(full["f1_macro"] - ref) < 1e-3 else "DIFFERS"
        print(f"[identity] C={n_source}: macro-F1 {full['f1_macro']:.5f} vs run {ref:.5f} -> {verdict}")
    print(f"[written] {out_root/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
