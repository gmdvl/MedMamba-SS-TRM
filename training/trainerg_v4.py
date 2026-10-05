# -*- coding: utf-8 -*-
"""
training/trainerg_v4.py
========================
Implements Phases 2, 3, 4, 5 and 6 of improve-prompt_v5.plan.md on top of
TrainerG_v3's engine:

  Phase 2 - total loss is configurable  L = CE + lambda_mse*MSE + lambda_sam*SAM
            + lambda_gan*GAN, with an optional adversarial discriminator
            (training.gan.SpectralDiscriminator) trained jointly.
  Phase 3 - validation reports the full spectral-reconstruction metric set
            (RMSE, MAE, SAM, SID, Pearson, Cosine, PSNR) via
            training.spectral_recon_metrics.
  Phase 4 - the evaluation pipeline: model.eval()/no_grad (always), sklearn
            metrics (training.eval_utils), per-class classification report,
            confusion matrix (.png + .npy), prediction-distribution dump,
            and automatic sanity-check warnings every epoch.
  Phase 5 - per-epoch logging: train_loss, classification_loss, MSE_loss,
            SAM_loss, GAN_loss, learning_rate, gradient_norm, GPU_memory,
            epoch_time.
  Phase 6 - experiment/ layout: config.json, history.csv, history.json,
            classification_report_epochXX.txt, confusion_matrix_epochXX.png,
            spectral_metrics.csv, best_model.pt, plots/.

Backward compatible in spirit with TrainerG_v3 (same model contract: a
forward pass returning either `logits` or `(logits, x_recon)`), but this is
a NEW class/file (matches the repo's own versioning convention, e.g.
prepare_histologyhsi_bc_v3.py) so nothing that already depends on
TrainerG_v3 is touched.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import RandomSampler

from training.eval_utils import (
    compute_classification_metrics, write_classification_report,
    save_confusion_matrix, save_prediction_distribution, sanity_check_metrics,
)
from training.gan import SAMLoss, SpectralDiscriminator, discriminator_loss, generator_adversarial_loss
from training.spectral_recon_metrics import compute_all_spectral_metrics, compute_per_sample_spectral_metrics
from training.metric_validation import MetricValidator
# improve-promp_v8.plan.md, Phase 0 - validation-loss investigation & stabilization
from training.per_class_metrics import compute_per_class_extended_metrics, identify_dominant_class
from training import pipeline_consistency


class StopReason(str, Enum):
    COMPLETED = "TRAINING_COMPLETED"
    VAL_DIVERGENCE = "VAL_DIVERGENCE"
    TRAIN_DIVERGENCE = "TRAIN_DIVERGENCE"
    LOSS_NAN = "LOSS_NAN"
    LOSS_INF = "LOSS_INF"
    GRAD_EXPLOSION = "GRADIENT_EXPLOSION"
    VAL_ACC_STALLED = "VAL_ACCURACY_STALLED"
    VAL_LOSS_STALLED = "VAL_LOSS_STALLED"
    USER_INTERRUPTED = "USER_INTERRUPTED"


class TrainerG_v4:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, exp_dir,
                 class_names=None,
                 lambda_mse: float = 1.0, lambda_sam: float = 0.0, lambda_gan: float = 0.0,
                 discriminator: Optional[nn.Module] = None, disc_optimizer=None,
                 grad_clip_norm: Optional[float] = 5.0, config=None,
                 wavelengths=None, recon_sample_cap: int = 2000, label_smoothing: float = 0.0):
        self.device = torch.device(device)
        self.device_type = "cuda" if "cuda" in str(self.device) else "cpu"

        # Phase 2 - configurable loss weights: L = CE + lambda_mse*MSE + lambda_sam*SAM + lambda_gan*GAN
        self.lambda_mse = lambda_mse
        self.lambda_sam = lambda_sam
        self.lambda_gan = lambda_gan
        # improve-promp_v8.plan.md, Phase 0.6 ablation axis - applied identically
        # to the classification loss in BOTH training and validation (Phase 0.3:
        # identical loss computation), same as every other loss component.
        self.label_smoothing = label_smoothing
        self.sam_loss_fn = SAMLoss()
        self.grad_clip_norm = grad_clip_norm

        self.model = model.to(self.device)
        if self.device_type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        self.use_gan = lambda_gan > 0 and discriminator is not None
        self.discriminator = discriminator.to(self.device) if discriminator is not None else None
        self.disc_optimizer = disc_optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp_dir = Path(exp_dir)
        self.class_names = class_names
        self.config = config or {}
        # Main Development Plan, Phase 4/6/8 diagnostics
        self.wavelengths = wavelengths
        self.recon_sample_cap = recon_sample_cap
        self.metric_validator = MetricValidator()
        self._last_val_metrics = None

        # Phase 4 item 2 - the validation loader must be deterministic; this can't fully verify
        # "no augmentation", but a shuffling sampler on val is a strong, checkable red flag.
        if isinstance(getattr(val_loader, "sampler", None), RandomSampler):
            self._warn_once = getattr(self, "_warn_once", [])
            print("WARNING: val_loader appears to use a shuffling sampler (shuffle=True). "
                  "Validation should use shuffle=False and deterministic transforms "
                  "(Phase 4 item 2) for reproducible metrics.", flush=True)

        self.logger = logging.getLogger("TrainerG_v4")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(stdout_handler)
        self.logger.propagate = False

        self.scaler = GradScaler(device=self.device_type)
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_val_acc = -1.0
        self.history = []
        self.stall_counters = {}
        self.stop_reason = None
        self.is_stopped = False

        # Phase 6 - experiment/ layout
        self.dirs = {
            "checkpoints": self.exp_dir / "checkpoints",
            "plots": self.exp_dir / "plots",
            "predictions": self.exp_dir / "predictions",
            "confidence_analysis": self.exp_dir / "confidence_analysis",
            "latent_space": self.exp_dir / "latent_space",
            "reconstruction": self.exp_dir / "reconstruction",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        if self.config:
            with open(self.exp_dir / "config.json", "w") as f:
                json.dump(self.config, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self, max_epochs, start_epoch=1):
        self.current_epoch = start_epoch - 1
        self.logger.info(f"Starting execution loop from epoch {start_epoch} to {max_epochs}...")

        # Phase 0.3 - Training/Validation Consistency Audit. Runs once, before any
        # epochs, so a broken pipeline (mismatched loss formula, shuffled val
        # loader, mismatched Dataset config, missing eval()/no_grad) is caught
        # immediately rather than discovered after a long run. Never aborts
        # training itself (this is diagnostics, not a hard gate) - failures are
        # logged loudly and recorded in pipeline_consistency_report.json.
        try:
            consistency_report = pipeline_consistency.write_report(
                self, str(self.exp_dir / "pipeline_consistency_report.json"))
            if not consistency_report["all_passed"]:
                for c in consistency_report["checks"]:
                    if not c["passed"]:
                        self.logger.warning(f"[pipeline consistency] {c['check']}: {c['message']}")
        except Exception as e:
            self.logger.warning(f"[pipeline consistency] audit failed to run: {e}")

        try:
            for epoch in range(start_epoch, max_epochs + 1):
                if self.is_stopped:
                    break
                self.current_epoch = epoch
                epoch_start_time = time.time()
                if self.device_type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                train_metrics = self._train_one_epoch()
                if self.is_stopped:
                    break
                val_metrics = self._validate_one_epoch()

                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()

                gpu_mem_mb = (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                              if self.device_type == "cuda" else 0.0)

                metrics = {
                    "epoch": epoch,
                    # Phase 5 - per-epoch training logging
                    "train_loss": train_metrics["loss"],
                    "classification_loss": train_metrics["cls_loss"],
                    "MSE_loss": train_metrics["mse_loss"],
                    "SAM_loss": train_metrics["sam_loss"],
                    "GAN_loss": train_metrics["gan_loss"],
                    "train_accuracy": train_metrics["acc"],
                    "learning_rate": current_lr,
                    "gradient_norm": train_metrics["grad_norm"],
                    # Phase 0.4 - Training Dynamics Diagnostics
                    "parameter_norm": train_metrics["param_norm"],
                    "update_norm": train_metrics["update_norm"],
                    "GPU_memory_MB": gpu_mem_mb,
                    "epoch_time": epoch_time,
                    # Phase 4 - validation-side classification metrics (sklearn)
                    "val_loss": val_metrics["loss"],
                    "val_cls_loss": val_metrics["cls_loss"],
                    # Phase 0.2 - validation now mirrors every training loss component
                    "val_mse_loss": val_metrics["mse_loss"],
                    "val_sam_loss": val_metrics["sam_loss"],
                    "val_gan_loss": val_metrics["gan_loss"],
                    **val_metrics["classification"],
                    # Phase 3 - full spectral reconstruction metric set
                    **{f"spectral_{k}": v for k, v in val_metrics["spectral"].items()},
                }
                metrics["val_accuracy"] = metrics.pop("accuracy")
                # Phase 0.4 - Generalization Gap (val_loss - train_loss; also
                # tracked in accuracy terms since that's often the more
                # intuitive read for a classifier).
                metrics["generalization_gap_loss"] = metrics["val_loss"] - metrics["train_loss"]
                metrics["generalization_gap_accuracy"] = metrics["train_accuracy"] - metrics["val_accuracy"]

                # Phase 4 item 7 - sanity checks (warnings only, never fatal)
                warnings = sanity_check_metrics(val_metrics["classification"], history=self.history)
                for w in warnings:
                    self.logger.warning(f"[metric sanity check] {w}")
                metrics["sanity_warnings"] = warnings

                # Main Development Plan, Phase 8 - metric validation (NaN/Inf,
                # constant streaks, impossible F1, balanced-accuracy
                # inconsistencies, sudden jumps, missing classes, empty preds)
                num_classes = val_metrics["probs"].shape[1] if val_metrics["probs"].size else None
                validation_issues = self.metric_validator.validate_epoch(
                    epoch, val_metrics["classification"], preds=val_metrics["preds"],
                    labels=val_metrics["labels"], num_classes=num_classes)
                for issue in validation_issues:
                    self.logger.warning(f"[metric validation] {issue['message']}")
                metrics["validation_issues"] = [i["type"] for i in validation_issues]

                # Phase 0.5 - dominant-class flag surfaced in the main history too,
                # not just the per-epoch per_class_metrics_epochXX.json file.
                dominant = val_metrics.get("dominant_class")
                if dominant:
                    self.logger.warning(f"[per-class analysis] {dominant['message']}")
                metrics["dominant_validation_loss_class"] = dominant["class_name"] if dominant else None

                self._last_val_metrics = val_metrics
                self.history.append(metrics)
                self._check_stopping_rules()

                is_best = metrics["val_accuracy"] > self.best_val_acc
                if is_best:
                    self.best_val_acc = metrics["val_accuracy"]
                    self.best_epoch = epoch

                # Phase 4 items 4-6 - per-class report / confusion matrix / prediction distribution
                self._save_eval_artifacts(epoch, val_metrics)
                # Phase 0.5 - per-class extended metrics (Loss/Specificity/Sensitivity/MCC/Kappa)
                self._save_per_class_metrics(epoch, val_metrics)
                # Phase 0.2 - per-epoch loss-component breakdown (train + val, every component)
                self._update_loss_components_file(metrics)

                self._save_checkpoint(is_best)
                self._update_history_files()
                self._log_epoch(metrics, is_best)

                if epoch == max_epochs and not self.is_stopped:
                    self.stop_reason = StopReason.COMPLETED
        except KeyboardInterrupt:
            self.stop_reason = StopReason.USER_INTERRUPTED
            self.is_stopped = True
        finally:
            self._generate_rich_diagnostics()
            self._generate_final_report()
        return self.stop_reason

    def _train_one_epoch(self):
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()
        total_loss = total_cls = total_mse = total_sam = total_gan = 0.0
        total_correct = total_n = 0
        total_grad_norm = 0.0
        n_batches = 0

        # Phase 0.4 - Training Dynamics Diagnostics: parameter norm and update
        # norm. Computed once per epoch (not per batch, to stay cheap): snapshot
        # the flattened parameter vector before any optimizer step this epoch,
        # then diff it against the post-epoch vector.
                
        with torch.no_grad():
            params_before = torch.nn.utils.parameters_to_vector(
                [p.detach().reshape(-1) for p in self.model.parameters() if p.requires_grad]).detach().clone()

        use_bf16 = self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        # float16 autocast on CPU is unreliable for custom ops (e.g. the selective-scan
        # kernels) and can silently produce NaN gradients - only autocast on CUDA.
        amp_enabled = self.device_type == "cuda"

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, x_recon = out
                else:
                    logits, x_recon = out, None

                cls_loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
                if x_recon is not None:
                    mse_loss = F.mse_loss(x_recon, x)
                    sam_loss = self.sam_loss_fn(x, x_recon)
                else:
                    mse_loss = torch.tensor(0.0, device=self.device)
                    sam_loss = torch.tensor(0.0, device=self.device)

                gan_loss = torch.tensor(0.0, device=self.device)
                if self.use_gan and x_recon is not None:
                    gan_loss = generator_adversarial_loss(self.discriminator, x_recon)

                loss = (cls_loss + self.lambda_mse * mse_loss + self.lambda_sam * sam_loss
                        + self.lambda_gan * gan_loss)

            if torch.isnan(loss) or torch.isinf(loss):
                self.stop_reason = StopReason.LOSS_NAN if torch.isnan(loss) else StopReason.LOSS_INF
                self.is_stopped = True
                break

            self.scaler.scale(loss).backward()

            # Phase 5 - gradient norm, measured before (optional) clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm if self.grad_clip_norm else float("inf"))
            grad_norm = float(grad_norm)

            # Phase 5 - a NaN/Inf gradient norm means this batch's step would corrupt the
            # weights; skip the optimizer step for it (and don't let it poison the logged
            # average) instead of silently applying it.
            if not np.isfinite(grad_norm):
                self.logger.warning("Non-finite gradient norm encountered - skipping this "
                                     "batch's optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue
            total_grad_norm += grad_norm

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Discriminator step (Phase 2's GAN loss): trained separately from the generator,
            # on a detached copy of the reconstruction so gradients don't leak into the encoder.
            if self.use_gan and x_recon is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                    d_loss = discriminator_loss(self.discriminator, x, x_recon.detach())
                if not (torch.isnan(d_loss) or torch.isinf(d_loss)):
                    d_loss.backward()
                    self.disc_optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_cls += cls_loss.item() * x.size(0)
            total_mse += mse_loss.item() * x.size(0)
            total_sam += sam_loss.item() * x.size(0)
            total_gan += gan_loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += x.size(0)
            n_batches += 1  # only reached for batches whose optimizer step wasn't skipped above

        with torch.no_grad():
            params_after = torch.nn.utils.parameters_to_vector(
                [p.detach().reshape(-1) for p in self.model.parameters() if p.requires_grad]).detach()
            param_norm = float(params_after.norm())
            update_norm = float((params_after - params_before).norm())

        denom = max(1, total_n)
        return {
            "loss": total_loss / denom, "cls_loss": total_cls / denom,
            "mse_loss": total_mse / denom, "sam_loss": total_sam / denom,
            "gan_loss": total_gan / denom, "acc": total_correct / denom,
            "grad_norm": total_grad_norm / max(1, n_batches),
            # Phase 0.4 - Training Dynamics Diagnostics
            "param_norm": param_norm, "update_norm": update_norm,
        }

    # ------------------------------------------------------------------
    # Validation - Phase 4: always eval()/no_grad, deterministic loader
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        # Phase 0.2/0.3 - loss-component audit: validation must compute the
        # SAME weighted total-loss formula as training (CE + lambda_mse*MSE +
        # lambda_sam*SAM + lambda_gan*GAN), not a subset of it, or val_loss and
        # train_loss aren't measuring the same objective and their curves are
        # not directly comparable - a real, previously-present root-cause
        # candidate for the validation-loss behavior this phase investigates.
        total_loss = total_cls = total_mse = total_sam = total_gan = 0.0
        total_n = 0
        spectral_accum = {}
        spectral_batches = 0
        all_preds, all_labels, all_probs = [], [], []
        all_per_sample_ce = []
        # Main Development Plan, Phase 4 - per-sample spectral metrics + mean
        # spectra, capped at self.recon_sample_cap, kept only for the most
        # recent validation pass (used for reconstruction visualizations /
        # histograms and rebuilt every epoch - cheap since it is vectorized
        # and bounded by the cap, not by dataset size).
        per_sample_metrics_accum = {}
        recon_true_means, recon_pred_means = [], []
        recon_collected = 0

        use_bf16 = self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        amp_enabled = self.device_type == "cuda"

        if self.discriminator is not None:
            self.discriminator.eval()

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, x_recon = out
                else:
                    logits, x_recon = out, None
                cls_loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
                per_sample_cls_loss = F.cross_entropy(logits, y, reduction="none",
                                                       label_smoothing=self.label_smoothing)
                if x_recon is not None:
                    mse_loss = F.mse_loss(x_recon, x)
                    sam_loss = self.sam_loss_fn(x, x_recon)
                else:
                    mse_loss = torch.tensor(0.0, device=self.device)
                    sam_loss = torch.tensor(0.0, device=self.device)

                gan_loss = torch.tensor(0.0, device=self.device)
                if self.use_gan and x_recon is not None:
                    gan_loss = generator_adversarial_loss(self.discriminator, x_recon)

                # Same weighted-sum formula as _train_one_epoch - see the
                # Phase 0.2/0.3 note above the function signature.
                loss = (cls_loss + self.lambda_mse * mse_loss + self.lambda_sam * sam_loss
                        + self.lambda_gan * gan_loss)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * x.size(0)
            total_cls += cls_loss.item() * x.size(0)
            total_mse += mse_loss.item() * x.size(0)
            total_sam += sam_loss.item() * x.size(0)
            total_gan += gan_loss.item() * x.size(0)
            total_n += x.size(0)

            probs = F.softmax(logits.float(), dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_per_sample_ce.append(per_sample_cls_loss.float().cpu().numpy())

            if x_recon is not None:
                wl_tensor = torch.as_tensor(self.wavelengths, device=x.device) if self.wavelengths is not None else None
                batch_spectral = compute_all_spectral_metrics(x, x_recon, wavelengths=wl_tensor)
                for k, v in batch_spectral.items():
                    spectral_accum[k] = spectral_accum.get(k, 0.0) + v * x.size(0)
                spectral_batches += x.size(0)

                if recon_collected < self.recon_sample_cap:
                    per_sample = compute_per_sample_spectral_metrics(x, x_recon, wavelengths=wl_tensor)
                    for k, v in per_sample.items():
                        per_sample_metrics_accum.setdefault(k, []).extend(v)
                    x_mean = x.float().mean(dim=(2, 3)).cpu().numpy()
                    xr_mean = x_recon.float().mean(dim=(2, 3)).cpu().numpy()
                    remaining = self.recon_sample_cap - recon_collected
                    recon_true_means.append(x_mean[:remaining])
                    recon_pred_means.append(xr_mean[:remaining])
                    recon_collected += min(x.size(0), remaining)

        preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
        labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
        probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 0))
        per_sample_ce = np.concatenate(all_per_sample_ce) if all_per_sample_ce else np.array([], dtype=np.float32)

        classification = (compute_classification_metrics(labels, preds) if len(labels) > 0
                           else {"accuracy": 0.0, "balanced_accuracy": 0.0, "precision_macro": 0.0,
                                 "recall_macro": 0.0, "f1_macro": 0.0, "precision_weighted": 0.0,
                                 "recall_weighted": 0.0, "f1_weighted": 0.0})

        spectral = ({k: v / max(1, spectral_batches) for k, v in spectral_accum.items()}
                    if spectral_batches > 0 else {})

        # Phase 0.5 - Per-Class Validation Analysis: Loss/Accuracy/Precision/
        # Recall/Specificity/Sensitivity/F1/MCC/Cohen's Kappa per class, plus
        # a heuristic flag for whether one class is disproportionately
        # responsible for the overall validation loss.
        per_class_extended = None
        dominant_class = None
        if len(labels) > 0:
            num_classes = probs.shape[1] if probs.size else int(max(labels.max(), preds.max())) + 1
            per_class_extended = compute_per_class_extended_metrics(
                labels, preds, num_classes, per_sample_loss=per_sample_ce,
                class_names=self.class_names)
            dominant_class = identify_dominant_class(per_class_extended)

        denom = max(1, total_n)
        return {
            "loss": total_loss / denom, "cls_loss": total_cls / denom,
            "mse_loss": total_mse / denom, "sam_loss": total_sam / denom, "gan_loss": total_gan / denom,
            "classification": classification, "spectral": spectral,
            "preds": preds, "labels": labels, "probs": probs,
            "per_sample_ce_loss": per_sample_ce,
            "per_class_extended": per_class_extended, "dominant_class": dominant_class,
            "recon_per_sample_metrics": per_sample_metrics_accum,
            "recon_true_means": np.concatenate(recon_true_means, axis=0) if recon_true_means else None,
            "recon_pred_means": np.concatenate(recon_pred_means, axis=0) if recon_pred_means else None,
        }

    # ------------------------------------------------------------------
    # Phase 4 - per-epoch evaluation artifacts
    # ------------------------------------------------------------------
    def _save_eval_artifacts(self, epoch, val_metrics):
        preds, labels, probs = val_metrics["preds"], val_metrics["labels"], val_metrics["probs"]
        if len(labels) == 0:
            return
        write_classification_report(
            labels, preds, self.exp_dir / f"classification_report_epoch{epoch:02d}.txt",
            class_names=self.class_names)
        save_confusion_matrix(labels, preds, str(self.exp_dir), epoch, class_names=self.class_names)
        save_prediction_distribution(
            labels, preds, probs, self.dirs["predictions"] / f"predictions_epoch{epoch:02d}.npz")

    # ------------------------------------------------------------------
    # Phase 0.5 - Per-Class Validation Analysis
    # ------------------------------------------------------------------
    def _save_per_class_metrics(self, epoch, val_metrics):
        """Writes per_class_metrics_epochXX.json (this epoch's full extended
        per-class breakdown) and appends a flattened row to the running
        per_class_metrics.csv (one row per class per epoch, easy to pivot/plot
        to see whether one class's loss/metrics diverge over training)."""
        per_class_extended = val_metrics.get("per_class_extended")
        if not per_class_extended:
            return

        epoch_path = self.exp_dir / f"per_class_metrics_epoch{epoch:02d}.json"
        payload = dict(per_class_extended)
        if val_metrics.get("dominant_class"):
            payload["dominant_class"] = val_metrics["dominant_class"]
        with open(epoch_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        rows = []
        for class_id, entry in per_class_extended["per_class"].items():
            rows.append({"epoch": epoch, "class_id": class_id, **entry})
        if not rows:
            return
        csv_path = self.exp_dir / "per_class_metrics.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Phase 0.2 - Loss Function Audit: every component, every epoch, train
    # AND validation, in its own dedicated file (in addition to living inside
    # history.csv/history.json) so this is trivially diffable/plottable on
    # its own without pulling in every other metric in history.
    # ------------------------------------------------------------------
    def _update_loss_components_file(self, metrics):
        row = {
            "epoch": metrics["epoch"],
            "train_total_loss": metrics["train_loss"],
            "train_classification_loss": metrics["classification_loss"],
            "train_mse_loss": metrics["MSE_loss"],
            "train_sam_loss": metrics["SAM_loss"],
            "train_gan_loss": metrics["GAN_loss"],
            "val_total_loss": metrics["val_loss"],
            "val_classification_loss": metrics["val_cls_loss"],
            "val_mse_loss": metrics["val_mse_loss"],
            "val_sam_loss": metrics["val_sam_loss"],
            "val_gan_loss": metrics["val_gan_loss"],
            "generalization_gap_loss": metrics["generalization_gap_loss"],
        }
        self._loss_component_rows = getattr(self, "_loss_component_rows", [])
        self._loss_component_rows.append(row)

        csv_path = self.exp_dir / "loss_components.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(self._loss_component_rows)

        json_path = self.exp_dir / "loss_components.json"
        with open(json_path, "w") as f:
            json.dump(self._loss_component_rows, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Stopping rules
    # ------------------------------------------------------------------
    def _check_stopping_rules(self):
        if len(self.history) < 2:
            return
        curr, prev = self.history[-1], self.history[-2]
        if curr["val_loss"] > prev["val_loss"] and curr["val_accuracy"] < prev["val_accuracy"]:
            self._increment_stall("val_diverge")
        else:
            self.stall_counters["val_diverge"] = 0
        if self.stall_counters.get("val_diverge", 0) >= 3:
            self._trigger_stop(StopReason.VAL_DIVERGENCE)
        if curr["val_accuracy"] == prev["val_accuracy"]:
            self._increment_stall("val_acc_stall")
        else:
            self.stall_counters["val_acc_stall"] = 0
        if self.stall_counters.get("val_acc_stall", 0) >= 5:
            self._trigger_stop(StopReason.VAL_ACC_STALLED)

    def _increment_stall(self, key):
        self.stall_counters[key] = self.stall_counters.get(key, 0) + 1

    def _trigger_stop(self, reason):
        self.stop_reason = reason
        self.is_stopped = True

    # ------------------------------------------------------------------
    # Phase 6 - reports / checkpoints
    # ------------------------------------------------------------------
    def _save_checkpoint(self, is_best):
        state = {
            "epoch": self.current_epoch, "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc, "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(), "history": self.history,
            "stall_counters": self.stall_counters,
        }
        if self.discriminator is not None:
            state["discriminator_state"] = self.discriminator.state_dict()
            state["disc_optimizer_state"] = self.disc_optimizer.state_dict()
        torch.save(state, self.dirs["checkpoints"] / f"epoch_{self.current_epoch:04d}.pt")
        torch.save(state, self.dirs["checkpoints"] / "latest.pt")
        if is_best:
            torch.save(state, self.exp_dir / "best_model.pt")

    def _update_history_files(self):
        # history.csv
        fieldnames = list(self.history[-1].keys())
        with open(self.exp_dir / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow({k: row.get(k) for k in fieldnames})
        # history.json
        with open(self.exp_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        # spectral_metrics.csv (Phase 3)
        spectral_rows = [
            {"epoch": r["epoch"], **{k[len("spectral_"):]: v for k, v in r.items()
                                      if k.startswith("spectral_")}}
            for r in self.history
        ]
        if spectral_rows and len(spectral_rows[0]) > 1:
            with open(self.exp_dir / "spectral_metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(spectral_rows[0].keys()))
                writer.writeheader()
                writer.writerows(spectral_rows)
        # plots/
        try:
            from training.plots import generate_all_training_plots
            plot_rows = [{"epoch": r["epoch"], "train_loss": r["train_loss"], "val_loss": r["val_loss"],
                          "train_acc": r["train_accuracy"], "val_acc": r["val_accuracy"],
                          "lr": r["learning_rate"], "peak_gpu_memory_mb": r["GPU_memory_MB"],
                          "epoch_time_s": r["epoch_time"],
                          # Phase 0.2 - loss components
                          "classification_loss": r.get("classification_loss"),
                          "val_cls_loss": r.get("val_cls_loss"),
                          "MSE_loss": r.get("MSE_loss"), "val_mse_loss": r.get("val_mse_loss"),
                          "SAM_loss": r.get("SAM_loss"), "val_sam_loss": r.get("val_sam_loss"),
                          # Phase 0.4 - training dynamics diagnostics
                          "gradient_norm": r.get("gradient_norm"),
                          "parameter_norm": r.get("parameter_norm"),
                          "update_norm": r.get("update_norm"),
                          "generalization_gap_loss": r.get("generalization_gap_loss"),
                          "generalization_gap_accuracy": r.get("generalization_gap_accuracy"),
                          } for r in self.history]
            generate_all_training_plots(plot_rows, str(self.dirs["plots"]))
        except Exception as e:
            self.logger.warning(f"plot generation failed: {e}")

    def _log_epoch(self, m, is_best):
        msg = (
            f"\n--- Epoch {m['epoch']:04d} ---\n"
            f"Train -> Loss: {m['train_loss']:.4f} (CE: {m['classification_loss']:.4f}, "
            f"MSE: {m['MSE_loss']:.4f}, SAM: {m['SAM_loss']:.4f}, GAN: {m['GAN_loss']:.4f}) | "
            f"Acc: {m['train_accuracy']:.4f} | GradNorm: {m['gradient_norm']:.3f}\n"
            f"Val   -> Loss: {m['val_loss']:.4f} (CE: {m['val_cls_loss']:.4f}, "
            f"MSE: {m['val_mse_loss']:.4f}, SAM: {m['val_sam_loss']:.4f}, GAN: {m['val_gan_loss']:.4f}) | "
            f"Acc: {m['val_accuracy']:.4f} | "
            f"BalAcc: {m['balanced_accuracy']:.4f} | "
            f"P(macro): {m['precision_macro']:.4f} | R(macro): {m['recall_macro']:.4f} | "
            f"F1(macro): {m['f1_macro']:.4f}\n"
            # Phase 0.4 - Training Dynamics Diagnostics
            f"Dynamics -> ParamNorm: {m['parameter_norm']:.3f} | UpdateNorm: {m['update_norm']:.5f} | "
            f"GenGap(loss): {m['generalization_gap_loss']:+.4f} | "
            f"GenGap(acc): {m['generalization_gap_accuracy']:+.4f}\n"
        )
        if m.get("spectral_rmse") is not None:
            msg += (f"Spectral -> RMSE: {m['spectral_rmse']:.4f} | MAE: {m.get('spectral_mae', 0):.4f} | "
                    f"SAM: {m.get('spectral_sam_deg', 0):.2f}\u00b0 | SID: {m.get('spectral_sid', 0):.4f} | "
                    f"Pearson: {m.get('spectral_pearson_correlation', 0):.4f} | "
                    f"Cosine: {m.get('spectral_cosine_similarity', 0):.4f} | "
                    f"PSNR: {m.get('spectral_psnr', 0):.2f}\n")
        msg += (f"Stats -> Time: {m['epoch_time']:.1f}s | LR: {m['learning_rate']:.2e} | "
                f"GPU: {m['GPU_memory_MB']:.0f}MB | "
                f"Saved: epoch_{m['epoch']:04d}.pt ({'BEST' if is_best else 'LATEST'})")
        if m.get("dominant_validation_loss_class"):
            msg += f"\n  [PER-CLASS] dominant validation-loss class: {m['dominant_validation_loss_class']}"
        if m.get("sanity_warnings"):
            for w in m["sanity_warnings"]:
                msg += f"\n  [WARN] {w}"
        print(msg, flush=True)

    # ------------------------------------------------------------------
    # Main Development Plan, Phase 6/4/8 - end-of-training diagnostics
    # ------------------------------------------------------------------
    def _generate_rich_diagnostics(self):
        """Writes validation_report.json (Phase 8) and, using the most
        recent validation pass (self._last_val_metrics), the confidence
        analysis plots, latent-space projection, and reconstruction
        visualizations/histograms (Phase 6/4). Every piece is wrapped so a
        failure (missing optional dependency, model without an extractable
        penultimate layer, no reconstruction head, etc.) only logs a
        warning instead of losing the training run's other artifacts.
        """
        try:
            report_path = self.exp_dir / "validation_report.json"
            self.metric_validator.write_report(str(report_path))
        except Exception as e:
            self.logger.warning(f"[diagnostics] failed to write validation_report.json: {e}")

        vm = self._last_val_metrics
        if vm is None or len(vm.get("labels", [])) == 0:
            return

        try:
            from training.plots import generate_confidence_analysis
            generate_confidence_analysis(vm["probs"], vm["preds"], vm["labels"],
                                          str(self.dirs["confidence_analysis"]))
        except Exception as e:
            self.logger.warning(f"[diagnostics] confidence analysis failed: {e}")

        try:
            from training.latent_space import extract_penultimate_features
            from training.plots import plot_latent_space
            extracted = extract_penultimate_features(self.model, self.val_loader, str(self.device))
            if extracted is not None:
                features, feat_labels = extracted
                plot_latent_space(features, feat_labels, self.class_names, str(self.dirs["latent_space"]))
            else:
                self.logger.info("[diagnostics] latent-space extraction not supported for this "
                                  "model - skipping t-SNE/UMAP plots.")
        except Exception as e:
            self.logger.warning(f"[diagnostics] latent-space visualization failed: {e}")

        per_sample = vm.get("recon_per_sample_metrics") or {}
        true_means, pred_means = vm.get("recon_true_means"), vm.get("recon_pred_means")
        if per_sample.get("sam_deg") and true_means is not None and pred_means is not None:
            try:
                from training.plots import plot_reconstruction_examples, plot_reconstruction_metric_histograms
                wl = np.asarray(self.wavelengths) if self.wavelengths is not None else None
                plot_reconstruction_examples(true_means, pred_means, wl, per_sample["sam_deg"],
                                              str(self.dirs["reconstruction"]))
                plot_reconstruction_metric_histograms(per_sample, str(self.dirs["reconstruction"]))
            except Exception as e:
                self.logger.warning(f"[diagnostics] reconstruction visualizations failed: {e}")

    def _generate_final_report(self):
        report = {
            "completion_status": {
                "stop_reason": self.stop_reason.value if self.stop_reason else "UNKNOWN",
                "total_epochs_run": self.current_epoch, "best_epoch": self.best_epoch,
                "best_val_accuracy": self.best_val_acc,
            },
            "history": self.history,
        }
        with open(self.exp_dir / "experiment_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("\n====================================", flush=True)
        print("TRAINING PROCESS ENDED", flush=True)
        print("====================================", flush=True)
        print(f"Reason: {self.stop_reason.value if self.stop_reason else 'UNKNOWN'}", flush=True)
        print(f"Total Epochs Completed: {self.current_epoch}", flush=True)
        print(f"Best Epoch: {self.best_epoch} (Val Acc: {self.best_val_acc:.4f})", flush=True)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        if self.discriminator is not None and "discriminator_state" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state"])
            self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state"])
        self.current_epoch = checkpoint["epoch"]
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_val_acc = checkpoint.get("best_val_acc", -1.0)
        self.history = checkpoint.get("history", [])
        self.stall_counters = checkpoint.get("stall_counters", {})
        return self.current_epoch
