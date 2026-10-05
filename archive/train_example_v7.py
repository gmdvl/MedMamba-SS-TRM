# -*- coding: utf-8 -*-
"""
train_example_v7.py
====================
Stage A ("Correctness") entry point, per GMedMamba_Improvement_Plan_Current_Scope.md.

Reuses train_example_v6.py's data discovery / dataset-integrity-check
machinery unchanged, and wires in:

  Phase 3     - class coverage is verified (fail-fast by default) before
                training starts.
  Phase 13    - best-checkpoint selection uses macro-F1 by default, not
                accuracy (training/trainerg_v5.py).
  Phase 15    - per-class-recall class-collapse monitoring.
  Phase 16/17 - gradient-health counters; an epoch with zero valid
                optimizer updates is marked INVALID and can never become
                "best".
  Phase 19-21 - --activation leakyrelu (default): post-hoc nn.ReLU ->
                nn.LeakyReLU swap; gmedmamba.py itself stays untouched.
  Phase 34/35 - --recon_mode latent (default): reconstruct from the
                backbone's learned feature_map instead of the raw input.
                --recon_mode raw_input keeps the OLD (architecturally
                wrong) path only so it can be A/B-ablated against the fix.
  Phase 39    - --spectral_dropout for the latent-reconstruction wrapper.

Everything else (GAN, spectral metrics, dataset-integrity/leakage check,
band-selection metadata copy-through) is unchanged from train_example_v6.py.

Usage:
    python train_example_v7.py --data_dir ./data/hsi --epochs 30

    # A/B the Stage-A fixes against the old behavior:
    python train_example_v7.py --data_dir ./data/hsi --epochs 30 \\
        --activation relu --recon_mode raw_input --checkpoint_metric accuracy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gmedmamba import gmedmamba_tiny, gmedmamba_hsi_small
from train_example_v6 import NpyDataset, discover_data, setup_experiment_dir
from training.trainerg_v5 import TrainerG_v5
from training.gan import SpectralDiscriminator
from training.class_coverage import verify_class_coverage
from training.activation_patch import replace_relu_with_leakyrelu
from training.reconstruction_head import GMedMambaLatentReconWrapper


class GMedMambaRawReconWrapper(nn.Module):
    """The OLD (Phase-34-violating) reconstruction path: the decoder sees
    only the raw input, never the backbone's learned representation. Kept
    only so `--recon_mode raw_input` can be A/B-ablated against the fix."""

    def __init__(self, base_model, in_channels):
        super().__init__()
        self.base_model = base_model
        hidden_channels = max(16, in_channels // 2)
        self.recon_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels), nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x, **_ignored):
        logits = self.base_model(x)
        x_recon = self.recon_head(x)
        return logits, x_recon


def main():
    p = argparse.ArgumentParser(description="GMedMamba Stage-A ('Correctness') training entry point")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--disc_lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_sam", type=float, default=0.1)
    p.add_argument("--lambda_gan", type=float, default=0.0)
    p.add_argument("--use_gan", action="store_true")

    p.add_argument("--check_leakage", action="store_true", default=True)
    p.add_argument("--no_check_leakage", dest="check_leakage", action="store_false")
    p.add_argument("--abort_on_leakage", action="store_true", default=True)
    p.add_argument("--allow_leakage", dest="abort_on_leakage", action="store_false")

    # Phase 3
    p.add_argument("--check_class_coverage", action="store_true", default=True)
    p.add_argument("--no_check_class_coverage", dest="check_class_coverage", action="store_false")
    p.add_argument("--allow_missing_classes", action="store_true",
                    help="warn instead of raising when a split is missing a required class")

    # Phase 13
    p.add_argument("--checkpoint_metric", default="f1_macro",
                    help="key in the validation classification dict used to pick the best "
                         "epoch/checkpoint (Phase 13). Default is macro-F1, NOT accuracy. "
                         "Other useful values: balanced_accuracy, f1_weighted, accuracy.")
    p.add_argument("--class_collapse_streak", type=int, default=3)

    # Phase 19-21
    p.add_argument("--activation", choices=["relu", "leakyrelu"], default="leakyrelu",
                    help="Phase 19: swap the conv/spectral branches' nn.ReLU for "
                         "nn.LeakyReLU(--leaky_slope) post-hoc. SiLU (SSM path) / GELU are left "
                         "untouched per Phase 20's 'don't replace SiLU blindly'.")
    p.add_argument("--leaky_slope", type=float, default=0.01)

    # Phase 34/35/39
    p.add_argument("--recon_mode", choices=["latent", "raw_input", "none"], default="latent",
                    help="'latent' (default, Phase 34 fix): reconstruct from the backbone's "
                         "learned feature_map. 'raw_input': the old, architecturally-wrong path, "
                         "kept only for an A/B ablation. 'none': classification only.")
    p.add_argument("--spectral_dropout", type=float, default=0.0,
                    help="Phase 39: probability per training step of masking a random subset of "
                         "spectral bands on the input (latent recon mode only). 0 disables it.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    exp_dir = setup_experiment_dir(exp_name="gmedmamba_stageA_run", resume_path=args.resume)

    if args.check_leakage:
        try:
            from training.dataset_integrity import run_full_integrity_check, DatasetLeakageError
            manifest_path = None
            for candidate in (Path(args.data_dir) / "_progress.json",
                               Path(args.data_dir).parent / "_progress.json"):
                if candidate.is_file():
                    manifest_path = str(candidate)
                    break
            run_full_integrity_check(out_dir=str(exp_dir), manifest_path=manifest_path,
                                      npy_data_dir=args.data_dir, abort_on_leakage=args.abort_on_leakage)
        except DatasetLeakageError:
            raise
        except Exception as e:
            print(f"WARNING: dataset integrity check could not run ({e}); continuing.", flush=True)

    print(f"Discovering dataset files in {args.data_dir}...", flush=True)
    x_tr, y_tr, x_te, y_te, wavelengths, num_classes, class_names, n_train, n_val = discover_data(args.data_dir)
    print(f"Dataset Loaded: {n_train} Train, {n_val} Val | Classes: {num_classes} | Device: {device}", flush=True)

    if args.check_class_coverage:
        y_train_arr = np.load(y_tr, mmap_mode="r")
        y_val_arr = np.load(y_te, mmap_mode="r")
        report = verify_class_coverage(
            {"train": y_train_arr, "validation": y_val_arr}, num_classes=num_classes,
            class_names=class_names, fail_hard=not args.allow_missing_classes)
        with open(exp_dir / "class_coverage_report.json", "w") as f:
            json.dump(report, f, indent=2)
        if not report["coverage_ok"]:
            print(f"WARNING: class coverage check found missing classes: {report['missing']} "
                  f"(continuing because --allow_missing_classes was passed)", flush=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump({"cli_args": vars(args), "classes": num_classes, "device": device}, f, indent=4)

    train_ds = NpyDataset(x_tr, y_tr)
    val_ds = NpyDataset(x_te, y_te)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device == "cuda"), persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device == "cuda"), persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None)

    if wavelengths is not None:
        base_model = gmedmamba_hsi_small(num_classes=num_classes)
    else:
        base_model = gmedmamba_tiny(num_classes=num_classes)

    in_channels = train_ds[0][0].shape[0]

    if args.activation == "leakyrelu":
        replace_relu_with_leakyrelu(base_model, negative_slope=args.leaky_slope)

    if args.recon_mode == "latent":
        model = GMedMambaLatentReconWrapper(base_model, in_channels=in_channels,
                                             spectral_dropout=args.spectral_dropout)
    elif args.recon_mode == "raw_input":
        model = GMedMambaRawReconWrapper(base_model, in_channels=in_channels)
    else:
        model = base_model

    use_gan = (args.use_gan or args.lambda_gan > 0) and args.recon_mode != "none"
    discriminator = disc_optimizer = None
    if use_gan:
        discriminator = SpectralDiscriminator(in_channels=in_channels)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr, weight_decay=0.0)
        if args.lambda_gan == 0.0:
            args.lambda_gan = 0.05

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {"cli_args": vars(args), "classes": num_classes, "device": device,
              "class_names": class_names, "in_channels": in_channels}

    trainer = TrainerG_v5(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device, exp_dir=exp_dir,
        class_names=class_names,
        lambda_mse=(args.lambda_mse if args.recon_mode != "none" else 0.0),
        lambda_sam=(args.lambda_sam if args.recon_mode != "none" else 0.0),
        lambda_gan=args.lambda_gan, discriminator=discriminator, disc_optimizer=disc_optimizer,
        config=config, wavelengths=wavelengths,
        checkpoint_metric=args.checkpoint_metric, class_collapse_streak=args.class_collapse_streak,
    )

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume) + 1
        else:
            print(f"Warning: Specified resume path not found: {args.resume}. Starting fresh.", flush=True)

    trainer.fit(max_epochs=args.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
