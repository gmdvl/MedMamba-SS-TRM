# -*- coding: utf-8 -*-
"""
training/trainerg_v9.py
========================
MedMamba-SS-TRM v12/v13 "Definitive Training Stability & SIGBUS Remediation"
plan, Stage G ("Stability") on top of `training/trainerg_v8.py`
(TrainerG_v8, which adds Cohen's Kappa/MCC to validation).

This is the trainer half of Phases 10-23 and 36-37:

  Phase 10 - AMP is now an explicit `amp_mode` constructor argument
             (`"off" | "bf16" | "fp16" | "auto"`), instead of trainerg_v5's
             hardcoded "bf16-if-supported-else-fp16" `autocast` selection.
             `"auto"` reproduces the old behaviour exactly.
  Phase 12 - every non-finite-gradient batch is now inspected at the
             PARAMETER level (`training.numerical_stability.inspect_gradients`)
             so `gradient_health` records which parameter went bad first,
             not just that the aggregate norm was NaN.
  Phase 14 - classification/MSE/SAM/GAN/total loss are checked for
             finiteness INDIVIDUALLY every batch, before backward.
  Phase 18 - parameters are checked for NaN/Inf immediately after every
             `optimizer.step()`; if any are found, the run aborts rather
             than continuing to train on now-corrupted weights.
  Phase 21/22/23 - an isolated bad batch is still just skipped (unchanged
             from TrainerG_v5's behaviour: reset grads, continue), but
             `training.numerical_stability.NumericalStabilityController`
             now classifies each epoch's skip ratio (healthy/warning/
             degraded/critical/invalid/fatal) and aborts the RUN - not just
             marks one epoch invalid - once `max_consecutive_bad_batches`,
             `max_gradient_skip_ratio`, or `max_consecutive_unhealthy_epochs`
             is crossed. `self.stop_reason` becomes the string
             `"TRAINING_ABORTED_NUMERICAL_INSTABILITY"` in that case (read
             via `.value`, same convention `trainerg_v7._StopReasonStr`
             already established for early stopping).
  Phase 37 - on abort, `numerical_failure/failure.json` (+ any `extra`
             files supplied) is written into the experiment directory
             before the run ends.

`gradient_health`'s per-epoch dict (still populated, for backward
compatibility with anything reading `history[i]["gradient_health"]`) now
also carries `stability_status`/`skip_ratio` from the richer controller.

Everything else - validation, Cohen's Kappa/MCC, early stopping,
class-collapse monitoring, macro-F1 checkpoint selection - is inherited
UNCHANGED from TrainerG_v5/v6/v7/v8.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from training.trainerg_v8 import TrainerG_v8
from training.trainerg_v7 import _StopReasonStr
from training.gan import discriminator_loss, generator_adversarial_loss
from training.numerical_stability import (
    NumericalStabilityController, StabilityConfig, inspect_gradients, check_parameters_finite,
    check_loss_components, format_loss_component_status,
)


class TrainerG_v9(TrainerG_v8):
    def __init__(self, *args, stability_cfg: Optional[StabilityConfig] = None,
                 amp_mode: str = "auto", debug_numerics: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stability_cfg = stability_cfg or StabilityConfig()
        self.stability_cfg.debug_numerics = debug_numerics
        self.stability = NumericalStabilityController(self.stability_cfg, exp_dir=str(self.exp_dir))
        assert amp_mode in ("off", "bf16", "fp16", "auto"), f"Unknown amp_mode {amp_mode!r}"
        self.amp_mode = amp_mode

    # ------------------------------------------------------------------
    def _resolve_amp(self):
        """Phase 10 - explicit AMP control. Returns (enabled, dtype)."""
        if self.amp_mode == "off":
            return False, torch.float16
        if self.amp_mode == "bf16":
            return (self.device_type == "cuda"), torch.bfloat16
        if self.amp_mode == "fp16":
            return (self.device_type == "cuda"), torch.float16
        # "auto" - reproduces TrainerG_v5/v6's original implicit behaviour.
        use_bf16 = self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        return (self.device_type == "cuda"), (torch.bfloat16 if use_bf16 else torch.float16)

    # ------------------------------------------------------------------
    def _train_one_epoch(self):
        self.stability.start_epoch()
        self.grad_health.start_epoch()
        self.model.train()
        total_loss = total_cls = total_mse = total_sam = total_gan = 0.0
        total_correct = total_n = 0
        total_grad_norm = 0.0
        n_batches = 0
        aborted_mid_epoch = False

        amp_enabled, amp_dtype = self._resolve_amp()

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

                cls_loss = self.criterion(logits, y)
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

            # Phase 14 - per-component finiteness, before backward.
            components = {
                "classification": cls_loss,
                "MSE": mse_loss if x_recon is not None else None,
                "SAM": sam_loss if x_recon is not None else None,
                "GAN": gan_loss if (self.use_gan and x_recon is not None) else None,
                "total": loss,
            }
            comp_status = check_loss_components(components)
            loss_finite = comp_status["total"] == "finite"

            if not loss_finite:
                if self.stability_cfg.debug_numerics:
                    self.logger.warning(f"[epoch {self.current_epoch}] {format_loss_component_status(comp_status)}")
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_health.record_batch(loss_is_finite=False, grad_is_finite=False, optimizer_stepped=False)
                keep_going = self.stability.record_batch(
                    loss_finite=False, grad_inspection=None, grad_norm=None, optimizer_stepped=False)
                if not keep_going:
                    aborted_mid_epoch = True
                    break
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Phase 12 - parameter-level gradient inspection, before clipping.
            grad_inspect = inspect_gradients(self.model)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm if self.grad_clip_norm else float("inf"))
            grad_norm = float(grad_norm)
            grad_ok = grad_inspect.all_finite and np.isfinite(grad_norm)

            if not grad_ok:
                if self.stability_cfg.debug_numerics:
                    self.logger.warning(
                        f"[epoch {self.current_epoch}] NONFINITE_GRADIENT "
                        f"first_bad_parameter={grad_inspect.first_bad_parameter!r} "
                        f"nan_elements={grad_inspect.nan_count} inf_elements={grad_inspect.inf_count}")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.grad_health.record_batch(
                    loss_is_finite=True, grad_is_finite=False, optimizer_stepped=False,
                    grad_norm=grad_norm if np.isfinite(grad_norm) else None,
                    first_bad_parameter=grad_inspect.first_bad_parameter,
                    gradient_nan_elements=grad_inspect.nan_count, gradient_inf_elements=grad_inspect.inf_count)
                keep_going = self.stability.record_batch(
                    loss_finite=True, grad_inspection=grad_inspect,
                    grad_norm=grad_norm if np.isfinite(grad_norm) else None, optimizer_stepped=False)
                if not keep_going:
                    aborted_mid_epoch = True
                    break
                continue

            total_grad_norm += grad_norm
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Phase 18 - post-step parameter health check.
            params_ok, bad_params = check_parameters_finite(self.model)
            self.grad_health.record_batch(loss_is_finite=True, grad_is_finite=True, optimizer_stepped=True,
                                           grad_norm=grad_norm)
            self.stability.record_batch(loss_finite=True, grad_inspection=grad_inspect,
                                         grad_norm=grad_norm, optimizer_stepped=True)
            if not params_ok:
                self.stability.record_external_abort(
                    f"NUMERICAL_INSTABILITY: model parameters non-finite immediately after "
                    f"optimizer.step() (Phase 18): {bad_params[:5]}"
                    + (f" (+{len(bad_params) - 5} more)" if len(bad_params) > 5 else ""))
                aborted_mid_epoch = True
                break

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
        stability_summary = self.stability.finish_epoch()
        epoch_health = {**epoch_health, "stability_status": stability_summary.status,
                         "stability_skip_ratio": stability_summary.skip_ratio}
        # Phase 22 - a "critical"/"invalid"/"fatal" skip-ratio classification
        # invalidates the epoch even when valid_updates > 0 (GradientHealthTracker
        # alone only checks the latter).
        if stability_summary.status in ("critical", "invalid", "fatal"):
            epoch_health["is_valid_epoch"] = False

        abort_reason = self.stability.should_abort_run()
        if abort_reason or aborted_mid_epoch:
            reason = abort_reason or "NUMERICAL_INSTABILITY: aborted mid-epoch (see epoch_health)."
            self.stop_reason = _StopReasonStr("TRAINING_ABORTED_NUMERICAL_INSTABILITY")
            self.is_stopped = True
            self.logger.error(f"[numerical-stability] ABORTING RUN: {reason}")
            try:
                failure_dir = self.stability.write_failure_artifacts(
                    str(self.exp_dir),
                    extra={
                        "dataloader_config": getattr(self, "dataloader_config", None),
                        "amp_mode": self.amp_mode,
                    })
                self.logger.error(f"[numerical-stability] failure artifacts written to {failure_dir}")
            except Exception as e:
                self.logger.warning(f"[numerical-stability] failed to write failure artifacts: {e}")

        denom = max(1, total_n)
        return {
            "loss": total_loss / denom, "cls_loss": total_cls / denom,
            "mse_loss": total_mse / denom, "sam_loss": total_sam / denom,
            "gan_loss": total_gan / denom, "acc": total_correct / denom,
            "grad_norm": total_grad_norm / max(1, n_batches),
            "gradient_health": epoch_health,
        }

    # Note: `_validate_one_epoch` is inherited unchanged from TrainerG_v8
    # (-> v7 -> v6). It still picks AMP dtype via the original
    # "bf16-if-supported" logic rather than `self.amp_mode` - acceptable
    # because validation is read-only (`@torch.no_grad()`, no optimizer
    # step), so it cannot itself be a source of the non-finite-GRADIENT
    # instability this trainer guards against. `--amp off` during
    # remediation still fully controls the TRAINING path above, which is
    # what Phase 10 is actually about.
