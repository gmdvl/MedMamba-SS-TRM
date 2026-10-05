# -*- coding: utf-8 -*-
"""
training/train_banner.py
=========================
What a run prints before it touches data: the configuration in force, every
typed flag that moved a profile default, and the bars and budgets that make a
bad run recognisable in epoch 1 instead of after it.

One banner for every profile. The former entry points each had their own, with
three hand-written "deviations" functions that re-derived what the parser had
resolved; here a deviation is simply a flag the user typed whose value differs
from the profile's default, read off the same parser.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from training.train_profiles import (
    PROFILES, input_side, split_size, steps_per_epoch,
)

# Activation cost per token-row at bf16, fitted to the 1,432 MB measured at 28x28
# x batch 32 on the RTX 5060 Ti. An estimate for the preflight only.
_MB_PER_TOKEN_ROW_BF16 = (1432.0 - 200.0) / (784 * 32)
_VRAM_FLOOR_MB = 200.0
_IMBALANCE_WARN_RATIO = 5.0
_TRM_ONLY_FLAGS = ("trm_dim", "trm_mixer", "trm_halting", "trm_dropout", "trm_drop_path",
                   "trm_core_layers", "trm_n_latent", "trm_n_improve",
                   "trm_deep_supervision_steps", "trm_ema_rate")


def deviations(parsed) -> list:
    """Typed flags whose value differs from the profile's default."""
    out = []
    for dest in sorted(parsed.explicit - {"profile", "data_dir", "run_tag"}):
        got, want = getattr(parsed.args, dest, None), parsed.parser.get_default(dest)
        if got != want:
            out.append(f"{dest}={got!r}  ({parsed.profile}: {want!r})")
    return out


def print_banner(parsed) -> None:
    args, profile = parsed.args, parsed.profile
    print("=" * 78, flush=True)
    print(f"MedMamba-SS-TRM - profile {profile!r}: {PROFILES[profile].summary}"
          + (f"  [stage: {args.stage}]" if args.stage != "full" else ""), flush=True)
    print(f"  architecture       : {args.architecture}  (stem stride {args.patch_size}, "
          f"trm_dim={args.trm_dim}, mixer={args.trm_mixer}, halting={args.trm_halting})", flush=True)
    print(f"  objective          : loss={args.loss} gamma={args.focal_gamma} "
          f"class_weight_power={args.class_weight_power} sampler={args.sampler}", flush=True)
    print(f"  optimiser          : AdamW lr={args.lr} wd={args.weight_decay} "
          f"wd_groups={args.weight_decay_groups}  clip={args.max_gradient_norm}", flush=True)
    print(f"  schedule           : {args.lr_schedule} (interval={args.scheduler_interval})", flush=True)
    print(f"  batch/epochs       : {args.batch_size} / {args.epochs}   amp={args.amp}   "
          f"compile={args.compile_model}   loader={args.loader_mode}", flush=True)
    print(f"  normalization      : {args.normalization}"
          + (f" / stats {args.global_stats}" if args.normalization == "global_zscore" else ""),
          flush=True)
    print(f"  regularisation     : classifier_dropout={args.classifier_dropout} "
          f"trm_dropout={args.trm_dropout} trm_drop_path={args.trm_drop_path} "
          f"drop_path={args.drop_path_rate} ema={args.trm_ema_rate}", flush=True)
    print(f"  augmentation       : {args.augment_preset}", flush=True)
    print(f"  subsample          : train {args.train_subsample_frac} ({args.train_subsample_mode}), "
          f"val {args.val_subsample_frac}", flush=True)
    print(f"  reconstruction     : {args.recon_mode}  lambda_mse={args.lambda_mse} "
          f"lambda_sam={args.lambda_sam}", flush=True)
    print(f"  selection/stopping : checkpoint on {args.checkpoint_metric}; early stopping "
          f"{_early_stop_text(args)}; metric-stall rule {args.stop_on_metric_stall}", flush=True)
    devs = deviations(parsed)
    if devs:
        print(f"  DEVIATIONS from profile {profile!r} (typed flags):", flush=True)
        for d in devs:
            print(f"    - {d}", flush=True)
    else:
        print(f"  no deviations: every {profile!r} default is in force.", flush=True)
    print(f"  run_tag            : {args.run_tag!r}"
          + ("" if args.run_tag else "  <-- a run with the same flags differs only by timestamp"),
          flush=True)
    print("=" * 78, flush=True)

    print_trivial_baselines(args)
    print_step_budget(args)
    print_trm_cost(args)
    print_ema_horizon(args)
    _warnings(args)


def _early_stop_text(args) -> str:
    if args.early_stopping == "on" and args.early_stop_patience:
        return f"ON after {args.early_stop_patience} epochs without a new best"
    return (f"OFF (--early_stopping {args.early_stopping}, --early_stop_patience "
            f"{args.early_stop_patience}: recorded, not enforced)")


def _warnings(args) -> None:
    if args.augment_preset in ("light", "medium"):
        try:
            channels = int(np.load(Path(args.data_dir) / "X_train.npy", mmap_mode="r").shape[-1])
        except Exception:
            channels = None
        if channels is not None and channels <= 4:
            print(f"[augment] WARNING: presets 'light'/'medium' were written for 32-band HSI. On "
                  f"{channels}-channel input band_dropout zeroes a WHOLE colour channel; prefer "
                  f"--augment_preset custom.", flush=True)
    if args.architecture != "recursive":
        named = [f"{f}={getattr(args, f)}" for f in _TRM_ONLY_FLAGS if hasattr(args, f)]
        print(f"[trm] --architecture {args.architecture!r} is NOT the recursive core, so these are "
              f"inert: {', '.join(named)}.", flush=True)
    if args.recon_mode != "none" and not args.lambda_mse and not args.lambda_sam:
        print("[recon] WARNING: --lambda_mse and --lambda_sam are both 0, so the decoder gets no "
              "gradient and reconstructs noise. Its metrics will be written and be meaningless.",
              flush=True)
    side = input_side(args.data_dir)
    if side is not None and side <= 32 and not (Path(args.data_dir) / "wavelengths.npy").exists():
        print(f"[input] WARNING: {side}x{side} RGB samples. On PAD-UFES-20's 11x11 grid a linear "
              f"probe scored macro-F1 0.185 because the crop mostly misses the lesion; re-prep "
              f"with prepare_pad_ufes_20_optimal.py --tiling whole.", flush=True)


def print_trivial_baselines(args) -> None:
    """The constant predictor's scores and the best shallow probe next to the
    dataset: the bars every result has to be read against."""
    try:
        y_train = np.load(Path(args.data_dir) / "y_train.npy", mmap_mode="r")
        counts = np.bincount(np.asarray(y_train).astype(int))
        priors = counts[counts > 0] / counts.sum()
        floor = float(-(priors * np.log(priors)).sum())
    except Exception:
        return
    present = counts[counts > 0]
    if (present.size >= 2 and present.max() / present.min() >= _IMBALANCE_WARN_RATIO
            and args.loss == "ce" and args.sampler == "none"):
        print(f"[imbalance] WARNING: train imbalance {present.max() / present.min():.1f}x under "
              f"--loss ce --sampler none: a weak learner's argmax is the majority class for every "
              f"input. Remedy: --loss weighted_ce (or focal_weighted).", flush=True)
    print(f"[baseline] train prior-entropy CE floor = {floor:.4f} (ln(k) = "
          f"{np.log(len(priors)):.4f} for a uniform predictor).", flush=True)
    for split, fname in (("val", "y_val.npy"), ("test", "y_test.npy")):
        try:
            c = np.bincount(np.asarray(np.load(Path(args.data_dir) / fname, mmap_mode="r")).astype(int))
            k = int((c > 0).sum()) or 1
            print(f"[baseline] {split}: majority-class accuracy {c.max() / c.sum():.4f}, "
                  f"balanced accuracy {1.0 / k:.4f} - the constant predictor's score.", flush=True)
        except Exception:
            continue
    try:
        with open(Path(args.data_dir) / "shallow_baseline.json") as f:
            report = json.load(f)
        results = (report[0] if isinstance(report, list) else report)["results"]
    except Exception:
        return
    best = max((r for r in results.values() if r.get("f1_macro") is not None),
               key=lambda r: r["f1_macro"], default=None)
    if best is not None:
        name = next(k for k, v in results.items() if v is best)
        print(f"[baseline] best shallow probe ({name}, {best['dim']} features): accuracy "
              f"{best['accuracy']:.4f}, balanced {best['balanced_accuracy']:.4f}, macro-F1 "
              f"{best['f1_macro']:.4f}. A network below it has an optimisation problem.", flush=True)


def print_step_budget(args) -> None:
    """Optimizer steps this run will take - over what the subsample leaves, not
    the full split (the cosine schedule decays over exactly this horizon)."""
    spe = steps_per_epoch(args.data_dir, args.batch_size, args.train_subsample_frac)
    if spe is None:
        return
    extra = ("; --sampler moderate_oversample enlarges the train set, so the real count is higher"
             if args.sampler == "moderate_oversample" else "")
    print(f"[budget] {split_size(args.data_dir, 'train'):,} train samples x {args.train_subsample_frac} "
          f"/ batch {args.batch_size} = {spe:,} steps/epoch x {args.epochs} epochs = "
          f"{spe * args.epochs:,} optimizer steps{extra}. Only compare runs with the same --epochs: "
          f"the schedule decays over this horizon.", flush=True)


def print_trm_cost(args) -> None:
    """The recursive core holds every token through all its applications, so the
    working set is tokens x batch: print it, and the card it has to fit on."""
    side = input_side(args.data_dir)
    if side is None or args.architecture != "recursive":
        return
    stride = args.patch_size or 1
    grid = max(1, side // stride)
    rows = grid * grid * args.batch_size
    applications = args.trm_deep_supervision_steps * args.trm_n_improve * (args.trm_n_latent + 1)
    est = _VRAM_FLOOR_MB + _MB_PER_TOKEN_ROW_BF16 * rows * (2.0 if args.amp == "off" else 1.0)
    line = (f"[trm-cost] input {side}x{side}, stem stride {stride} -> {grid}x{grid} = {grid * grid} "
            f"tokens x batch {args.batch_size}; core applied ~{applications}x per step; "
            f"~{est:,.0f} MB estimated at --amp {args.amp}")
    if torch.cuda.is_available():
        total_mb = torch.cuda.mem_get_info()[1] / 1024 ** 2
        line += f" ({100.0 * est / total_mb:.1f}% of {torch.cuda.get_device_name(0)})"
        if est > 0.85 * total_mb:
            line += ". WARNING: close to the card - raise --patch_size before lowering --batch_size"
    print(line, flush=True)


def print_ema_horizon(args) -> None:
    """How much of the random initialisation is still inside the model that
    validation runs on (the EMA shadow starts there, with no bias correction)."""
    if args.architecture != "recursive" or not args.trm_ema_rate:
        print("[ema] off - validation runs on the live weights.", flush=True)
        return
    spe = steps_per_epoch(args.data_dir, args.batch_size, args.train_subsample_frac)
    if spe is None:
        return
    mu = args.trm_ema_rate
    horizon = 1.0 / max(1e-12, (1.0 - mu) * spe)
    marks = [e for e in (1, 3, 5, 10, 25, 50) if e <= args.epochs]
    below5 = np.log(0.05) / np.log(mu) / spe if mu < 1 else float("inf")
    print(f"[ema] rate {mu:.5f} at {spe} steps/epoch = {horizon:.1f}-epoch horizon; init weight "
          f"remaining: " + "  ".join(f"ep{e}={mu ** (e * spe):.1%}" for e in marks)
          + f"; >95% trained model from epoch ~{below5:.0f}"
          + ("" if below5 <= 12 else " - TOO SLOW: raise --batch_size or lower --trm_ema_rate"),
          flush=True)
