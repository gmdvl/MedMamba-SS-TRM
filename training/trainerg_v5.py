# -*- coding: utf-8 -*-
"""
training/trainerg_v5.py
========================
Stage A ("Correctness") fixes on top of `training/trainerg_v4.py`
(TrainerG_v4), per GMedMamba_Improvement_Plan_Current_Scope.md:

  Phase 13 - Change Model Selection Metric. Best-checkpoint/best-epoch
    selection now uses `checkpoint_metric` (default "f1_macro", i.e.
    macro-F1) instead of raw val_accuracy - TrainerG_v4's `is_best =
    val_accuracy > best_val_acc` meant a model that gets high OA purely by
    predicting majority classes would be selected as "best" on an
    imbalanced set. balanced_accuracy/MCC/weighted_f1/OA are all still
    logged every epoch as secondary metrics (already true of TrainerG_v4).

  Phase 15 - Add Class-Collapse Monitoring. Flags any class whose recall
    has been exactly 0 for `class_collapse_streak` consecutive epochs.

  Phase 16/17 - Fix Gradient/Training Instability + Invalid Epoch
    Protection. Tracks total_batches/valid_updates/skipped_updates/
    nonfinite_loss/nonfinite_gradients per epoch via
    `training.gradient_health.GradientHealthTracker`. Unlike TrainerG_v4
    (which aborts the ENTIRE run on the first NaN/Inf training loss), a
    single bad batch is just skipped; only if EVERY batch in an epoch fails
    is that epoch marked INVALID. An invalid epoch can never become the
    best epoch, never updates best_model.pt, and is logged with a clear
    warning - but training continues into the next epoch rather than
    stopping outright (a single bad epoch is much more common under mixed
    precision than a truly unrecoverable divergence).

Everything else (GAN loss, full spectral-metric validation, diagnostics,
checkpoint/report file layout) is inherited unchanged from TrainerG_v4.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from training.trainerg_v4 import TrainerG_v4, StopReason
from training.eval_utils import sanity_check_metrics
from training.gan import discriminator_loss, generator_adversarial_loss
from training.gradient_health import GradientHealthTracker


class TrainerG_v5(TrainerG_v4):
    def __init__(self, *args, checkpoint_metric: str = "f1_macro",
                 class_collapse_streak: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_metric = checkpoint_metric
        self.class_collapse_streak = class_collapse_streak
        self.best_metric_value = float("-inf")
        self.grad_health = GradientHealthTracker()
        self._per_class_recall_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Phase 16/17 - gradient-health-aware training loop
    # ------------------------------------------------------------------
    def _train_one_epoch(self):
        self.grad_health.start_epoch()
        self.model.train()
        total_loss = total_cls = total_mse = total_sam = total_gan = 0.0
        total_correct = total_n = 0
        total_grad_norm = 0.0
        n_batches = 0

        use_bf16 = self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
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

                cls_loss = F.cross_entropy(logits, y)
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

            loss_finite = bool(torch.isfinite(loss))
            if not loss_finite:
                self.grad_health.record_batch(loss_is_finite=False, grad_is_finite=False, optimizer_stepped=False)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm if self.grad_clip_norm else float("inf"))
            grad_norm = float(grad_norm)
            grad_finite = bool(np.isfinite(grad_norm))

            if not grad_finite:
                self.grad_health.record_batch(loss_is_finite=True, grad_is_finite=False, optimizer_stepped=False)
                self.logger.warning("Non-finite gradient norm - skipping this batch's optimizer step.")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                continue

            total_grad_norm += grad_norm
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.grad_health.record_batch(loss_is_finite=True, grad_is_finite=True, optimizer_stepped=True)

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
            n_batches += 1

        epoch_health = self.grad_health.finish_epoch()
        denom = max(1, total_n)
        return {
            "loss": total_loss / denom, "cls_loss": total_cls / denom,
            "mse_loss": total_mse / denom, "sam_loss": total_sam / denom,
            "gan_loss": total_gan / denom, "acc": total_correct / denom,
            "grad_norm": total_grad_norm / max(1, n_batches),
            "gradient_health": epoch_health,
        }

    # ------------------------------------------------------------------
    # Phase 15 - class-collapse monitoring
    # ------------------------------------------------------------------
    def _check_class_collapse(self, classification_metrics) -> List[str]:
        recall = classification_metrics.get("per_class_recall")
        if recall is None:
            return []
        recall = np.asarray(recall)
        self._per_class_recall_history.append(recall)
        window = self._per_class_recall_history[-self.class_collapse_streak:]
        if len(window) < self.class_collapse_streak:
            return []
        stacked = np.stack(window, axis=0)
        n_classes = stacked.shape[1]
        collapsed = [c for c in range(n_classes) if np.all(stacked[:, c] == 0.0)]
        if not collapsed:
            return []
        names = self.class_names if self.class_names and len(self.class_names) == n_classes else \
            [str(c) for c in range(n_classes)]
        return [f"class '{names[c]}' has had 0 recall for {self.class_collapse_streak} epochs in a row "
                f"(class collapse)" for c in collapsed]

    # ------------------------------------------------------------------
    # Phase 13/17 - training loop with metric-based checkpoint selection
    # and invalid-epoch protection
    # ------------------------------------------------------------------
    def fit(self, max_epochs, start_epoch=1):
        self.current_epoch = start_epoch - 1
        self.logger.info(f"[TrainerG_v5] epochs {start_epoch}->{max_epochs}, "
                          f"checkpoint_metric={self.checkpoint_metric!r}")
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                if self.is_stopped:
                    break
                self.current_epoch = epoch
                epoch_start_time = time.time()
                if self.device_type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                # Phase 0.4 training-dynamics diagnostics: TrainerG_v4 computed
                # these inside its own `_train_one_epoch`; TrainerG_v5/v9 replaced
                # that method and dropped them, but the inherited
                # `TrainerG_v4._log_epoch` still hard-indexes `parameter_norm`/
                # `update_norm`. Snapshot the flat parameter vector before any
                # optimizer step this epoch and diff it post-epoch (once per
                # epoch, not per batch - cheap).
                with torch.no_grad():
                    params_before = torch.nn.utils.parameters_to_vector(
                        [p.detach().reshape(-1) for p in self.model.parameters()
                         if p.requires_grad]).detach().clone()

                train_metrics = self._train_one_epoch()
                val_metrics = self._validate_one_epoch()

                with torch.no_grad():
                    params_after = torch.nn.utils.parameters_to_vector(
                        [p.detach().reshape(-1) for p in self.model.parameters()
                         if p.requires_grad]).detach()
                    parameter_norm = float(params_after.norm())
                    update_norm = float((params_after - params_before).norm())

                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()

                gpu_mem_mb = (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                              if self.device_type == "cuda" else 0.0)

                metrics = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "classification_loss": train_metrics["cls_loss"],
                    "MSE_loss": train_metrics["mse_loss"],
                    "SAM_loss": train_metrics["sam_loss"],
                    "GAN_loss": train_metrics["gan_loss"],
                    "train_accuracy": train_metrics["acc"],
                    "learning_rate": current_lr,
                    "gradient_norm": train_metrics["grad_norm"],
                    "GPU_memory_MB": gpu_mem_mb,
                    "epoch_time": epoch_time,
                    "val_loss": val_metrics["loss"],
                    "val_cls_loss": val_metrics["cls_loss"],
                    # NOTE: pre-existing bug fix, unrelated to the v12/v13 SIGBUS/stability
                    # remediation plan. `TrainerG_v4._log_epoch`/`_generate_final_report`
                    # (inherited unchanged) hard-index `metrics["val_mse_loss"/"val_sam_loss"/
                    # "val_gan_loss"]`, but `TrainerG_v6._validate_one_epoch` only returns
                    # "loss"/"cls_loss" (not the per-component reconstruction/GAN losses) -
                    # this trainer's own `fit()` never backfilled them, so ANY epoch that
                    # actually reached `_log_epoch` raised `KeyError: 'val_mse_loss'`.
                    # Discovered while verifying this plan's Stage 6-9 acceptance criteria
                    # (a full epoch must complete and be reported) - fixed here rather than
                    # left in place, since without it no run using TrainerG_v5+ can ever log
                    # a completed epoch. Validation doesn't compute per-component SAM/GAN
                    # losses (only reconstruction MSE, when `x_recon is not None`), so SAM/GAN
                    # are reported as 0.0 for validation (they always were train-only entries
                    # in the original per-epoch report; this restores that, it doesn't add
                    # anything new).
                    "val_mse_loss": val_metrics.get("mse_loss", 0.0),
                    "val_sam_loss": val_metrics.get("sam_loss", 0.0),
                    "val_gan_loss": val_metrics.get("gan_loss", 0.0),
                    **val_metrics["classification"],
                    **{f"spectral_{k}": v for k, v in val_metrics["spectral"].items()},
                    "gradient_health": train_metrics["gradient_health"],
                    "is_valid_epoch": train_metrics["gradient_health"]["is_valid_epoch"],
                    # Phase 0.4 - training dynamics (see params_before/after above)
                    "parameter_norm": parameter_norm,
                    "update_norm": update_norm,
                }
                metrics["val_accuracy"] = metrics.pop("accuracy")
                # Phase 0.4 - generalization gap (also hard-indexed by
                # TrainerG_v4._log_epoch); must follow the val_accuracy rename.
                metrics["generalization_gap_loss"] = metrics["val_loss"] - metrics["train_loss"]
                metrics["generalization_gap_accuracy"] = metrics["train_accuracy"] - metrics["val_accuracy"]

                warnings = sanity_check_metrics(val_metrics["classification"], history=self.history)
                for w in warnings:
                    self.logger.warning(f"[metric sanity check] {w}")
                metrics["sanity_warnings"] = warnings

                num_classes = val_metrics["probs"].shape[1] if val_metrics["probs"].size else None
                validation_issues = self.metric_validator.validate_epoch(
                    epoch, val_metrics["classification"], preds=val_metrics["preds"],
                    labels=val_metrics["labels"], num_classes=num_classes)
                for issue in validation_issues:
                    self.logger.warning(f"[metric validation] {issue['message']}")
                metrics["validation_issues"] = [i["type"] for i in validation_issues]

                collapse_warnings = self._check_class_collapse(val_metrics["classification"])
                metrics["class_collapse_warnings"] = collapse_warnings
                for w in collapse_warnings:
                    self.logger.warning(f"[class collapse] {w}")

                self._last_val_metrics = val_metrics
                self.history.append(metrics)
                self._check_stopping_rules()

                is_best = False
                if metrics["is_valid_epoch"]:
                    candidate = metrics.get(self.checkpoint_metric)
                    if candidate is not None and candidate > self.best_metric_value:
                        self.best_metric_value = candidate
                        self.best_val_acc = metrics["val_accuracy"]
                        self.best_epoch = epoch
                        is_best = True
                else:
                    self.logger.warning(
                        f"Epoch {epoch} marked INVALID (valid_updates=0 - every training batch "
                        f"this epoch had a non-finite loss/gradient). Excluded from best-epoch/"
                        f"best_model.pt selection (Phase 17).")

                self._save_eval_artifacts(epoch, val_metrics)
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
