# -*- coding: utf-8 -*-
"""
train_example_v6.py
====================
Pipeline entry point for improve-prompt_v5.plan.md, Phases 2-6:

    HSI -> Band Selection (upstream, prepare_histologyhsi_bc_v3.py --band_selection importance)
        -> Patch Embedding -> MedMamba Encoder -> Classification Head
                                                -> Spectral Generator -> Reconstructed Spectrum
                                                                       -> Discriminator (optional)

Total loss:  L = CE + lambda_mse*MSE + lambda_sam*SAM + lambda_gan*GAN  (all lambdas configurable)

This is a new file (not an edit of train_example-gemini-v5.py) so the v5
pipeline keeps working unchanged for anyone already depending on it.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gmedmamba import gmedmamba_tiny, gmedmamba_hsi_small
from training.trainerg_v4 import TrainerG_v4
from training.gan import SpectralDiscriminator
from training.npy_integrity import resolve_storage_mode, load_npy_array


class GMedMambaGANWrapper(nn.Module):
    """Classification + spectral-reconstruction ("generator") head. The
    discriminator itself is a separate module (SpectralDiscriminator),
    trained by TrainerG_v4 - it isn't part of the forward pass here so a
    plain `model(x) -> (logits, x_recon)` contract still holds, matching
    TrainerG_v3/TrainerG_v4's expectations."""

    def __init__(self, base_model, in_channels):
        super().__init__()
        self.base_model = base_model
        hidden_channels = max(16, in_channels // 2)
        self.recon_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.base_model(x)
        x_recon = self.recon_head(x)
        return logits, x_recon


def compute_global_channel_stats(x_path: str, sample_cap: int = 5000, seed: int = 0):
    """Phase 0.6 ablation axis: 'global_zscore' normalization needs
    per-channel mean/std fit ONCE on a bounded sample of the TRAINING set
    (never on val/test - that would leak test-set statistics into
    normalization) and then applied identically to every split."""
    X = np.load(x_path, mmap_mode="r")
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_cap, n), replace=False)
    idx.sort()
    sample = np.asarray(X[idx], dtype=np.float32)
    sample = np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0)
    mean = sample.mean(axis=(0, 1, 2))
    std = sample.std(axis=(0, 1, 2))
    std = np.where(std > 1e-6, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


class NpyDataset(Dataset):
    """Memory-mapped dataset loader.

    `normalization`:
      - "per_sample_minmax" (default, original behaviour): each patch is
        independently min-max scaled to [0,1] using ITS OWN min/max.
      - "global_zscore": each patch is standardized with per-channel
        mean/std fit once on a training-set sample (`global_mean`/
        `global_std`, see `compute_global_channel_stats`) and applied
        identically to every split - a Phase 0.6 ablation axis for the
        validation-loss investigation (per-sample min-max normalization
        means every patch is rescaled by ITS OWN range, which can distort
        relative intensities across patches in a way a fixed, dataset-level
        normalization does not).

    `augment` (train split only - NEVER set True for a validation/test
    Dataset; Phase 0.3's pipeline-consistency audit checks that the
    train/val Dataset *configuration* matches, so this flag being the one
    deliberate difference between them is by design, not an oversight):
      - "none" (default): no augmentation.
      - "flip": random horizontal/vertical flip, each independently with
        p=0.5. Flips are a safe augmentation for HSI/RGB tissue patches
        (no canonical up/down orientation), unlike e.g. color jitter, which
        would need per-band-physically-meaningful handling.
    """

    def __init__(self, x_path: str, y_path: str, normalization: str = "per_sample_minmax",
                 global_mean=None, global_std=None, augment: str = "none", seed: int = 0,
                 storage_mode: str = "mmap"):
        """`storage_mode` (v12/v13 SIGBUS-remediation plan, Phase 5):
        "mmap" (default, unchanged legacy behaviour) | "ram" | "auto".
        Callers that have already run `training.npy_integrity.validate_dataset`
        upstream (e.g. `train_example_v13.py`) can safely pass "ram"/"auto";
        this constructor itself does NOT re-validate - see
        `training/npy_integrity.py` for why validation must happen before
        any DataLoader worker exists, not lazily on first access.
        """
        self.storage_mode = resolve_storage_mode(x_path, storage_mode) if storage_mode == "auto" else storage_mode
        self.X = load_npy_array(x_path, self.storage_mode)
        self.y = load_npy_array(y_path, self.storage_mode)
        assert len(self.X) == len(self.y), f"X/y length mismatch: {len(self.X)} vs {len(self.y)}"
        assert normalization in ("per_sample_minmax", "global_zscore")
        self.normalization = normalization
        self.global_mean = None if global_mean is None else np.asarray(global_mean, dtype=np.float32)
        self.global_std = None if global_std is None else np.asarray(global_std, dtype=np.float32)
        if normalization == "global_zscore":
            assert self.global_mean is not None and self.global_std is not None, \
                "global_zscore normalization requires global_mean/global_std"
        assert augment in ("none", "flip")
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        patch = np.array(self.X[idx], dtype=np.float32, copy=True)
        patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

        if self.normalization == "per_sample_minmax":
            p_min, p_max = patch.min(), patch.max()
            if p_max - p_min > 1e-6:
                patch = (patch - p_min) / (p_max - p_min)
            else:
                patch = np.zeros_like(patch)
        else:  # global_zscore
            patch = (patch - self.global_mean) / self.global_std

        if self.augment == "flip":
            if self._rng.random() < 0.5:
                patch = np.ascontiguousarray(patch[::-1, :, :])
            if self._rng.random() < 0.5:
                patch = np.ascontiguousarray(patch[:, ::-1, :])

        patch = torch.from_numpy(patch).permute(2, 0, 1).contiguous()
        label = int(self.y[idx])
        return patch, label


def discover_data(data_dir: str):
    base = Path(data_dir)
    x_train, y_train = base / "X_train.npy", base / "y_train.npy"
    x_test, y_test = base / "X_test.npy", base / "y_test.npy"
    wl_path = base / "wavelengths.npy"
    for p in [x_train, y_train, x_test, y_test]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    x_val, y_val = base / "X_val.npy", base / "y_val.npy"
    if x_val.exists() and y_val.exists():
        val_x_path, val_y_path = x_val, y_val
    else:
        val_x_path, val_y_path = x_test, y_test

    y_tr = np.load(y_train, mmap_mode="r")
    y_va = np.load(val_y_path, mmap_mode="r")
    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    wavelengths = np.load(wl_path).astype("float32") if wl_path.exists() else None

    class_names = None
    class_names_path = base / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path) as f:
            class_names = json.load(f)

    return (str(x_train), str(y_train), str(val_x_path), str(val_y_path),
            wavelengths, num_classes, class_names, len(y_tr), len(y_va))


def setup_experiment_dir(base_dir="experiments", exp_name="gmedmamba_run", resume_path=None,
                          append_timestamp=True):
    """`append_timestamp=True` (default, unchanged behaviour for every caller
    that predates this argument): the directory is `{exp_name}_{YYYYmmdd_HHMMSS}`.

    `append_timestamp=False`: `exp_name` is used verbatim. This exists for
    `train_example_v14.py`, whose naming scheme puts the timestamp FIRST so
    `ls experiments/` still sorts chronologically - see
    `training/run_naming.build_run_name`, which has already embedded it.
    """
    if resume_path:
        return Path(resume_path).resolve().parent.parent
    if append_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{exp_name}_{timestamp}"
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="GMedMamba Pipeline: Classification + Reconstruction + GAN")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    # Phase 2 - configurable loss weights: L = CE + lambda_mse*MSE + lambda_sam*SAM + lambda_gan*GAN
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_sam", type=float, default=0.1,
                         help="weight for the SAM (spectral-shape) loss term")
    parser.add_argument("--lambda_gan", type=float, default=0.0,
                         help="weight for the adversarial loss term; 0 disables the discriminator")
    parser.add_argument("--use_gan", action="store_true",
                         help="attach a SpectralDiscriminator and train it jointly "
                              "(also auto-enabled if --lambda_gan > 0)")
    # Main Development Plan, Phase 7 - dataset integrity / leakage check
    parser.add_argument("--check_leakage", action="store_true", default=True,
                         help="run the Phase 7 dataset-integrity check (patient/slide-level, from "
                              "the prep script's manifest, plus patch-level content-hash duplicate "
                              "detection) before training starts. On by default.")
    parser.add_argument("--no_check_leakage", dest="check_leakage", action="store_false",
                         help="skip the Phase 7 dataset-integrity check.")
    parser.add_argument("--abort_on_leakage", action="store_true", default=True,
                         help="abort training if the Phase 7 check finds leakage. On by default.")
    parser.add_argument("--allow_leakage", dest="abort_on_leakage", action="store_false",
                         help="log leakage (if found) but continue training anyway.")
    # improve-promp_v8.plan.md, Phase 0.6 - controlled ablation axes for the
    # validation-loss investigation. Defaults reproduce the exact prior
    # behaviour (per-sample min-max normalization, no augmentation, no label
    # smoothing) so leaving all three unset changes nothing.
    parser.add_argument("--normalization", choices=["per_sample_minmax", "global_zscore"],
                         default="per_sample_minmax",
                         help="Phase 0.6 ablation axis: per-patch min-max (old default) vs. a "
                              "fixed per-channel mean/std fit once on a training-set sample and "
                              "applied identically to every split.")
    parser.add_argument("--augmentation", choices=["none", "flip"], default="none",
                         help="Phase 0.6 ablation axis: random horizontal/vertical flips on the "
                              "TRAIN split only (never applied to validation/test).")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                         help="Phase 0.6 ablation axis: label smoothing epsilon for the "
                              "classification cross-entropy term (0.0 = off, old default).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    exp_dir = setup_experiment_dir(resume_path=args.resume)

    # Main Development Plan, Phase 7 - dataset integrity / leakage check, run
    # BEFORE any data is loaded for training so a leaky dataset is caught
    # immediately rather than after an expensive training run.
    if args.check_leakage:
        try:
            from training.dataset_integrity import run_full_integrity_check, DatasetLeakageError
            manifest_path = None
            for candidate in (Path(args.data_dir) / "_progress.json",
                               Path(args.data_dir).parent / "_progress.json"):
                if candidate.is_file():
                    manifest_path = str(candidate)
                    break
            run_full_integrity_check(
                out_dir=str(exp_dir), manifest_path=manifest_path, npy_data_dir=args.data_dir,
                abort_on_leakage=args.abort_on_leakage)
        except DatasetLeakageError:
            raise
        except Exception as e:
            print(f"WARNING: dataset integrity check could not run ({e}); continuing.", flush=True)

    print(f"Discovering dataset files in {args.data_dir}...", flush=True)
    x_tr, y_tr, x_te, y_te, wavelengths, num_classes, class_names, n_train, n_val = discover_data(args.data_dir)
    print(f"Dataset Loaded: {n_train} Train, {n_val} Val | Classes: {num_classes} | Device: {device}", flush=True)

    global_mean = global_std = None
    if args.normalization == "global_zscore":
        global_mean, global_std = compute_global_channel_stats(x_tr)
        print(f"  [normalization] global_zscore stats fit on X_train sample: "
              f"mean~{float(global_mean.mean()):.4f} std~{float(global_std.mean()):.4f}", flush=True)

    train_ds = NpyDataset(x_tr, y_tr, normalization=args.normalization,
                           global_mean=global_mean, global_std=global_std,
                           augment=args.augmentation)
    # Phase 0.3 - validation must NEVER receive augmentation, only ever
    # `augment="none"` regardless of what --augmentation was set to for train.
    val_ds = NpyDataset(x_te, y_te, normalization=args.normalization,
                         global_mean=global_mean, global_std=global_std,
                         augment="none")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0), prefetch_factor=4 if args.num_workers > 0 else None)
    # Phase 4 item 2 - validation must be deterministic: shuffle=False, no augmentation.
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0), prefetch_factor=4 if args.num_workers > 0 else None)

    if wavelengths is not None:
        print("Initializing HSI model variant...", flush=True)
        base_model = gmedmamba_hsi_small(num_classes=num_classes)
    else:
        print("Initializing RGB/Grayscale model variant...", flush=True)
        base_model = gmedmamba_tiny(num_classes=num_classes)

    in_channels = train_ds[0][0].shape[0]
    print(f"Attaching Spectral Generator (reconstruction head) for {in_channels} channels...", flush=True)
    model = GMedMambaGANWrapper(base_model, in_channels=in_channels)

    use_gan = args.use_gan or args.lambda_gan > 0
    discriminator = disc_optimizer = None
    if use_gan:
        print("Attaching SpectralDiscriminator for adversarial reconstruction training...", flush=True)
        discriminator = SpectralDiscriminator(in_channels=in_channels)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr, weight_decay=0.0)
        if args.lambda_gan == 0.0:
            args.lambda_gan = 0.05  # --use_gan without an explicit weight still needs a nonzero one

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Phase 1 metadata (if the dataset was produced with --band_selection importance) is copied
    # into the experiment dir so every run stays reproducible end-to-end (Phase 6).
    for fname in ("selection_report.json", "band_selection_report.json", "selected_wavelengths.npy",
                  "band_importance.npy", "band_ranking.npy", "selected_band_indices.npy",
                  "dataset_statistics.json", "balancing_report.json",
                  "class_distribution_before.png", "class_distribution_after.png",
                  "dataset_split_report.json"):
        src = Path(args.data_dir) / fname
        if src.exists():
            (exp_dir / fname).write_bytes(src.read_bytes())

    config = {"cli_args": vars(args), "classes": num_classes, "device": device,
              "class_names": class_names, "in_channels": in_channels}

    trainer = TrainerG_v4(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device, exp_dir=exp_dir,
        class_names=class_names,
        lambda_mse=args.lambda_mse, lambda_sam=args.lambda_sam, lambda_gan=args.lambda_gan,
        discriminator=discriminator, disc_optimizer=disc_optimizer,
        config=config, wavelengths=wavelengths, label_smoothing=args.label_smoothing,
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
