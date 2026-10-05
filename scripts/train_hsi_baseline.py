#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_hsi_baseline.py
=================================
Train HybridSN or SpectralFormer (training/hsi_baselines.py) on the
paired hyperspectral build under the SAME recipe as the reported
MedMamba-SS-TRM runs, assembled from the pipeline's own components so that
nothing but the architecture differs where it can:

    normalization    training.npy_data.compute_global_channel_stats (cap 5000, seed)
    dataset          training.npy_data.NpyDatasetV16, global z-score
    augmentation     training.augmentation.AugmentationConfig with the reported
                     run's values, filtered by augmentation_v16.build_active
    train subsample  RandomSampler without replacement, 10.2 % per epoch, re-drawn
                     every epoch (the reported runs' `per_epoch` mode)
    val subsample    training.train_pipeline._subsample_indices_v16, stratified_group,
                     9.86 %, seed + 1  -> the same 32,985 patches
    loss             training.losses.build_criterion('focal_weighted', gamma 1.5,
                     balanced weights ** 0.75)
    optimizer        AdamW 3e-4, weight decay 0.05 via training.optim_groups
    schedule         training.train_pipeline.build_warmup_cosine_scheduler, per step
    precision        bf16 autocast, gradient clipping at 1.0
    selection        best validation macro-F1

Two deliberate differences, both matching how the pipeline treats every
non-recursive architecture: no EMA, and no reconstruction decoder.

Writes into experiments/<timestamp>_<arch>_baseline_<tag>/: config.json,
history.json, best_model.pt, test_predictions.npz (full test split),
val_predictions.npz (full validation split), test_report.json.

Usage (host, GPU):
    python scripts/train_hsi_baseline.py --arch hybridsn \
        --data_dir data/hsi_v8-80_10_10_importance-new/hsi --epochs 20 --seed 42
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.npy_data import compute_global_channel_stats  # noqa: E402
from training.train_pipeline import _subsample_indices_v16, build_warmup_cosine_scheduler  # noqa: E402
from training.augmentation import AugmentationConfig  # noqa: E402
from training.augmentation_v16 import AugmentedPatchDatasetV16, build_active  # noqa: E402
from training.heldout_metrics import classification_metrics  # noqa: E402
from training.hsi_baselines import build_baseline  # noqa: E402
from training.losses import build_criterion  # noqa: E402
from training.npy_data import NpyDatasetV16  # noqa: E402
from training.optim_groups import build_param_groups  # noqa: E402

# The reported MedMamba-SS-TRM runs' augmentation values (config.json cli_args).
REPORTED_AUG = dict(spectral_noise_std=0.02, spectral_scale_range=0.15, spectral_offset_std=0.05,
                    band_dropout_prob=0.0, band_dropout_max_frac=0.1, spectral_mask_prob=0.0,
                    spectral_mask_max_width_frac=0.1, flip_h_prob=0.5, flip_v_prob=0.5,
                    rotate90_prob=0.5, crop_scale_max_frac=0.3)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels).astype(np.int64)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--arch", choices=["hybridsn", "spectralformer"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_subsample_frac", type=float, default=0.102)
    p.add_argument("--val_subsample_frac", type=float, default=0.0986)
    p.add_argument("--focal_gamma", type=float, default=1.5)
    p.add_argument("--class_weight_power", type=float, default=0.75)
    p.add_argument("--max_gradient_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--run_tag", default="s42")
    p.add_argument("--out_root", default="experiments")
    args = p.parse_args(argv)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data_dir)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_root) / f"{stamp}_{data.parent.name}-{data.name}_{args.arch}_baseline_{args.run_tag}"
    out.mkdir(parents=True, exist_ok=False)
    print(f"[out] {out}", flush=True)

    mean, std = compute_global_channel_stats(str(data / "X_train.npy"), sample_cap=5000, seed=args.seed)

    def ds(split):
        return NpyDatasetV16(str(data / f"X_{split}.npy"), str(data / f"y_{split}.npy"),
                             storage_mode="auto", normalization="global_zscore",
                             global_mean=mean, global_std=std, return_norm_stats=False)

    train_base, val_full, test_full = ds("train"), ds("val"), ds("test")
    y_train = np.load(data / "y_train.npy")
    y_val = np.load(data / "y_val.npy")
    groups_val = np.load(data / "groups_val.npy")
    num_classes = int(max(y_train.max(), y_val.max())) + 1
    in_channels = int(np.load(data / "X_train.npy", mmap_mode="r").shape[-1])

    aug = build_active(AugmentationConfig(**REPORTED_AUG, seed=args.seed))
    train_ds = AugmentedPatchDatasetV16(train_base, aug, seed=args.seed)
    kept = int(round(len(train_ds) * args.train_subsample_frac))
    sampler = RandomSampler(train_ds, replacement=False, num_samples=kept,
                            generator=torch.Generator().manual_seed(args.seed))
    val_idx = _subsample_indices_v16(y_val, args.val_subsample_frac, "stratified_group", num_classes,
                                     args.seed + 1, groups=groups_val)
    val_sub = Subset(val_full, val_idx.tolist())
    print(f"[data] C={in_channels} train {kept}/{len(train_ds)} per epoch, val subset {len(val_idx)}, "
          f"test {len(test_full)}", flush=True)

    lk = dict(num_workers=args.num_workers, pin_memory=device == "cuda",
              persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=False,
                              **lk, prefetch_factor=4 if args.num_workers else None)
    eval_kw = dict(batch_size=1024, shuffle=False, num_workers=args.num_workers,
                   pin_memory=device == "cuda")
    val_loader = DataLoader(val_sub, **eval_kw)

    model = build_baseline(args.arch, in_channels, num_classes).to(device)
    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    print(f"[model] {args.arch}: {n_params:,} trainable parameters", flush=True)

    criterion = build_criterion("focal_weighted", y_train=y_train, num_classes=num_classes,
                                focal_gamma=args.focal_gamma, weight_method="balanced", device=device,
                                class_weight_power=args.class_weight_power)
    opt = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = len(train_loader)
    sched = build_warmup_cosine_scheduler(opt, total_steps=args.epochs * steps_per_epoch)

    config = {"entry_point": "scripts/train_hsi_baseline.py", "cli_args": vars(args),
              "architecture": args.arch, "in_channels": in_channels, "classes": num_classes,
              "backbone_num_params": n_params, "steps_per_epoch": steps_per_epoch,
              "scheduler": {"total_steps": sched.total_steps, "warmup_steps": sched.warmup_steps},
              "val_subsample_kept": int(len(val_idx)), "train_subsample_kept": kept,
              "augmentation": REPORTED_AUG, "ema": None, "recon": None,
              "torch_version": torch.__version__}
    json.dump(config, open(out / "config.json", "w"), indent=2, default=str)

    history, best = [], (-1.0, 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n_ok, n_seen = time.time(), 0.0, 0, 0
        for batch in train_loader:
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
            loss = criterion(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
            opt.step()
            sched.step()
            tot += float(loss) * len(y)
            n_ok += int((logits.argmax(1) == y).sum())
            n_seen += len(y)
        probs, labels = predict(model, val_loader, device)
        m = classification_metrics(labels, probs, num_classes=num_classes)
        row = {"epoch": epoch, "train_loss": tot / n_seen, "train_accuracy": n_ok / n_seen,
               "grad_norm_last": float(gn), "lr": sched.get_last_lr()[0],
               "val_balanced_accuracy": m["balanced_accuracy"], "val_f1_macro": m["f1_macro"],
               "val_accuracy": m["accuracy"], "epoch_time_s": time.time() - t0}
        history.append(row)
        json.dump(history, open(out / "history.json", "w"), indent=2)
        flag = ""
        if m["f1_macro"] > best[0]:
            best = (m["f1_macro"], epoch)
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, out / "best_model.pt")
            flag = "  *best*"
        print(f"[ep {epoch:02d}] loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} | "
              f"val bal {m['balanced_accuracy']:.4f} F1 {m['f1_macro']:.4f} | "
              f"{row['epoch_time_s']:.0f}s{flag}", flush=True)

    state = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    report = {"best_epoch": best[1], "best_val_f1_macro": best[0], "backbone_num_params": n_params}
    for split, dset in (("test", test_full), ("val", val_full)):
        probs, labels = predict(model, DataLoader(dset, **eval_kw), device)
        np.savez_compressed(out / f"{split}_predictions.npz", probs=probs.astype(np.float32),
                            true_label=labels, predicted_label=probs.argmax(1))
        report[split] = classification_metrics(labels, probs, num_classes=num_classes)
        print(f"[{split}] bal {report[split]['balanced_accuracy']:.4f} "
              f"F1 {report[split]['f1_macro']:.4f} acc {report[split]['accuracy']:.4f}", flush=True)
    json.dump(report, open(out / "test_report.json", "w"), indent=2)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
