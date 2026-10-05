# -*- coding: utf-8 -*-
"""
training/trainerg_v11.py
==========================
MedMamba-SS-TRM v15 plan - the reporting-integrity and stop-condition trainer, on
top of `training/trainerg_v10.py` (`TrainerG_v10`). Per contract C-2,
`TrainerG_v10` is UNTOUCHED: `train_example_v14.py` keeps the trainer it was
run with, so the two evidence runs stay reproducible.

Everything TrainerG_v10 and its ancestors do - deep supervision, EMA, AMP,
per-component numerical-stability gating, gradient health, class-collapse
monitoring, checkpointing, early stopping - still happens. This subclass adds:

  R0.3  Class collapse is a STOP CONDITION, not a warning string.
        `TrainerG_v5._check_class_collapse` produced warnings and nothing
        else, and `--class_collapse_streak 3` read like a control while only
        setting the warning threshold. The HSI run spent 2.2 GPU-hours
        re-confirming a state that `predictions_epoch01.npz` had already
        settled. v11 tracks a streak of epochs whose predicted-label histogram
        has exactly ONE non-zero bin and, at `--on_class_collapse abort`,
        stops with `TRAINING_ABORTED_REPRESENTATION_COLLAPSE` and writes
        `<exp_dir>/collapse_failure/failure.json`.

  R2.3  Wavelengths actually reach the model. `train_example_v14.py` loads
        `wavelengths.npy` and hands it to the trainer, and no trainer in the
        v4->v10 chain ever forwarded it into a model call - so
        `use_wavelength_metadata`, `continuous_wavelength_encoding` and
        `sensor_reference_range` were all inert and `medmamba_ss_trm.py`'s
        sensor-agnostic claim was untested. Depends on R1.3 (the encoding's
        angular span); do not enable this without it.

  R5.1  `halt_loss` is logged. `_train_one_epoch_deep_supervision` returned it
        and `TrainerG_v5.fit`'s `metrics` dict never read it, which is why the
        RGB run's `Loss: 1.6955 (CE: 1.3647, MSE: 0.0000, SAM: 0.0000,
        GAN: 0.0000)` had 0.33 unaccounted for.

  R5.2  The generalization gap is computed from LAST-SEGMENT CE on both sides.
        Train `cls_loss` is the mean over all deep-supervision segments plus
        `0.5*halt_loss`, on augmented data, from the live weights; validation
        is last-segment CE on clean data from the EMA weights. Hence the RGB
        run's `GenGap(loss): -0.2247` on every single epoch.

  R5.3  The checkpoint matches the report (gate G5). `TrainerG_v10` validates
        the EMA copy but saved the LIVE weights to `best_model.pt`, so
        reloading it did not reproduce the logged best metric. In v11, when
        EMA drives validation `best_model.pt` IS the EMA model as a full
        loadable state, the live weights go to `best_model_live.pt`, and
        `config.json`/the checkpoint record which.

  R5.4  `--amp` is honoured in validation. `TrainerG_v6._validate_one_epoch`
        hardcoded bf16-if-supported, so `--amp off` only affected training and
        `amp_mode` in `config.json` did not describe the numerics that
        produced the reported metrics.

  R5.5  Test-split evaluation (gate G6) - `evaluate_test_split()`, called once
        after `fit` by `train_example_v15.py`.

  R5.6  Artifact retention. `--keep_last_n` epoch checkpoints (plus always the
        best), `--eval_artifact_stride` for the per-epoch confusion-matrix PNG
        / classification report / prediction `.npz` (the HSI run wrote six
        byte-identical PNGs and six 334k-row `.npz` files), and the full
        history is no longer embedded in every checkpoint - it lives in
        `history.json`, which is where anyone actually reads it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from training.trainerg_v10 import TrainerG_v10
from training.trainerg_v7 import _StopReasonStr
from training.trainerg_v4 import StopReason
from training.eval_utils import compute_classification_metrics, sanity_check_metrics
from training.failure_taxonomy import FailureClass
from training.gan import discriminator_loss, generator_adversarial_loss
from training.numerical_stability import (
    inspect_gradients, check_parameters_finite, check_loss_components, format_loss_component_status,
)
from training.spectral_recon_metrics import (
    compute_all_spectral_metrics, compute_per_sample_spectral_metrics,
)

COLLAPSE_STOP_REASON = "TRAINING_ABORTED_REPRESENTATION_COLLAPSE"


class TrainerG_v11(TrainerG_v10):
    def __init__(self, *args,
                  on_class_collapse: str = "abort",
                  use_wavelengths: bool = True,
                  sensor_range=None,
                  keep_last_n: int = 3,
                  eval_artifact_stride: int = 1,
                  embed_history_in_checkpoint: bool = False,
                  **kwargs):
        super().__init__(*args, **kwargs)
        if on_class_collapse not in ("warn", "abort"):
            raise ValueError(f"on_class_collapse must be 'warn' | 'abort', got {on_class_collapse!r}")
        self.on_class_collapse = on_class_collapse
        self.keep_last_n = int(keep_last_n)
        self.eval_artifact_stride = max(1, int(eval_artifact_stride))
        self.embed_history_in_checkpoint = bool(embed_history_in_checkpoint)
        self.sensor_range = sensor_range

        self._single_class_streak = 0
        self._current_val_preds: Optional[np.ndarray] = None
        self._collapse_failure_written = False
        # R6.3 - set by `train_example_v15.py` when the train split is wrapped
        # in an `AugmentedPatchDataset`, so the per-item augmentation seed
        # advances with the epoch instead of replaying epoch 0 forever.
        self.augmented_train_dataset = None

        # R2.3 - the wavelength vector, on the training device, or None.
        self._wl = None
        if use_wavelengths and self.wavelengths is not None:
            self._wl = torch.as_tensor(np.asarray(self.wavelengths, dtype=np.float32),
                                        device=self.device)
            self.logger.info(f"[wavelengths] forwarding {tuple(self._wl.shape)} band centers "
                              f"({float(self._wl.min()):.1f}-{float(self._wl.max()):.1f} nm) into the "
                              f"model (v15 R2.3; this was inert in every trainer up to v10)")

    # ==================================================================
    # R2.3 - wavelength plumbing
    # ==================================================================

    def _model_forward(self, model, x):
        """`model(x)`, with `wavelengths=`/`sensor_range=` when the model
        accepts them and a wavelength vector is available. Every model in this
        repo whose forward takes `wavelengths` also takes `sensor_range`."""
        if self._wl is None or not self._accepts_wavelengths(model):
            return model(x)
        return model(x, wavelengths=self._wl, sensor_range=self.sensor_range)

    @staticmethod
    def _accepts_wavelengths(fn_owner) -> bool:
        import inspect
        target = getattr(fn_owner, "forward", fn_owner)
        try:
            return "wavelengths" in inspect.signature(target).parameters
        except (TypeError, ValueError):
            return False

    def _deep_supervision_forward(self, base, x):
        if self._wl is None or not self._accepts_wavelengths_ds(base):
            return base.forward_deep_supervision(x)
        return base.forward_deep_supervision(x, wavelengths=self._wl, sensor_range=self.sensor_range)

    @staticmethod
    def _accepts_wavelengths_ds(base) -> bool:
        import inspect
        try:
            return "wavelengths" in inspect.signature(base.forward_deep_supervision).parameters
        except (TypeError, ValueError):
            return False

    # ==================================================================
    # R5.4 - AMP in validation
    # ==================================================================

    def _resolve_val_amp(self):
        """R5.4 - validation uses the SAME `--amp` resolution as training.
        `TrainerG_v6._validate_one_epoch` hardcoded bf16-if-supported inline,
        so `--amp off` produced fp32 training and bf16 validation and
        `amp_mode` in `config.json` described neither."""
        return self._resolve_amp()

    # ==================================================================
    # Training epoch (R2.3 + R5.1 + R5.2)
    # ==================================================================

    def _train_one_epoch_deep_supervision(self, base):
        """`TrainerG_v10._train_one_epoch_deep_supervision`, plus:
          * wavelengths threaded into `forward_deep_supervision` (R2.3);
          * the LAST-SEGMENT classification loss accumulated separately, so
            the generalization gap can compare like with like (R5.2).
        Every stability/gradient-health mechanism is byte-for-byte the same."""
        self.stability.start_epoch()
        self.grad_health.start_epoch()
        self.model.train()
        total_loss = total_cls = total_halt = total_mse = total_sam = total_gan = 0.0
        total_cls_last = 0.0
        total_correct = total_n = 0
        total_grad_norm = 0.0
        n_batches = 0
        aborted_mid_epoch = False

        amp_enabled, amp_dtype = self._resolve_amp()
        decoder = getattr(self.model, "decoder", None)
        halting = bool(getattr(base.cfg, "trm_act_halting", False))

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                logits_list, q_list = self._deep_supervision_forward(base, x)
                logits = logits_list[-1]
                per_segment = [self.criterion(l, y) for l in logits_list]
                cls_loss = sum(per_segment) / len(per_segment)
                # R5.2 - the only quantity comparable with validation's CE.
                cls_loss_last = per_segment[-1]

                if halting:
                    targets = [(l.detach().argmax(dim=-1) == y).to(q_list[0].dtype) for l in logits_list]
                    halt_loss = sum(F.binary_cross_entropy_with_logits(q, t)
                                    for q, t in zip(q_list, targets)) / len(q_list)
                else:
                    halt_loss = torch.tensor(0.0, device=self.device)

                x_recon = decoder(base.last_feature_map, target_hw=x.shape[-2:]) if decoder is not None else None
                if x_recon is not None:
                    mse_loss = F.mse_loss(x_recon, x)
                    sam_loss = self.sam_loss_fn(x, x_recon)
                else:
                    mse_loss = torch.tensor(0.0, device=self.device)
                    sam_loss = torch.tensor(0.0, device=self.device)

                gan_loss = torch.tensor(0.0, device=self.device)
                if self.use_gan and x_recon is not None:
                    gan_loss = generator_adversarial_loss(self.discriminator, x_recon)

                loss = (cls_loss + 0.5 * halt_loss + self.lambda_mse * mse_loss
                        + self.lambda_sam * sam_loss + self.lambda_gan * gan_loss)

            components = {
                "classification": cls_loss, "halt": halt_loss if halting else None,
                "MSE": mse_loss if x_recon is not None else None,
                "SAM": sam_loss if x_recon is not None else None,
                "GAN": gan_loss if (self.use_gan and x_recon is not None) else None,
                "total": loss,
            }
            comp_status = check_loss_components(components)
            if comp_status["total"] != "finite":
                if self.stability_cfg.debug_numerics:
                    self.logger.warning(f"[epoch {self.current_epoch}] {format_loss_component_status(comp_status)}")
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_health.record_batch(loss_is_finite=False, grad_is_finite=False, optimizer_stepped=False)
                if not self.stability.record_batch(loss_finite=False, grad_inspection=None,
                                                    grad_norm=None, optimizer_stepped=False):
                    aborted_mid_epoch = True
                    break
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            grad_inspect = inspect_gradients(self.model)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm if self.grad_clip_norm else float("inf"))
            grad_norm = float(grad_norm)
            grad_ok = grad_inspect.all_finite and bool(np.isfinite(grad_norm))

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
                if not self.stability.record_batch(
                        loss_finite=True, grad_inspection=grad_inspect,
                        grad_norm=grad_norm if np.isfinite(grad_norm) else None, optimizer_stepped=False):
                    aborted_mid_epoch = True
                    break
                continue

            total_grad_norm += grad_norm
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema_helper is not None:
                self.ema_helper.update(self.model)

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

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_cls += cls_loss.item() * bs
            total_cls_last += cls_loss_last.item() * bs
            total_halt += halt_loss.item() * bs
            total_mse += mse_loss.item() * bs
            total_sam += sam_loss.item() * bs
            total_gan += gan_loss.item() * bs
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += bs
            n_batches += 1

        epoch_health = self.grad_health.finish_epoch()
        stability_summary = self.stability.finish_epoch()
        epoch_health = {**epoch_health, "stability_status": stability_summary.status,
                         "stability_skip_ratio": stability_summary.skip_ratio}
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
                    extra={"dataloader_config": getattr(self, "dataloader_config", None),
                           "amp_mode": self.amp_mode})
                self.logger.error(f"[numerical-stability] failure artifacts written to {failure_dir}")
            except Exception as e:
                self.logger.warning(f"[numerical-stability] failed to write failure artifacts: {e}")

        denom = max(1, total_n)
        return {
            "loss": total_loss / denom, "cls_loss": total_cls / denom,
            "cls_loss_last_segment": total_cls_last / denom,
            "halt_loss": total_halt / denom,
            "mse_loss": total_mse / denom, "sam_loss": total_sam / denom,
            "gan_loss": total_gan / denom, "acc": total_correct / denom,
            "grad_norm": total_grad_norm / max(1, n_batches),
            "gradient_health": epoch_health,
            "n_optimizer_steps": n_batches,
        }

    # ==================================================================
    # Validation (R5.4 + R2.3 + R0.4f + R0.3 bookkeeping)
    # ==================================================================

    @torch.no_grad()
    def _validate_one_epoch_impl(self):
        """`TrainerG_v6._validate_one_epoch`, with `--amp` honoured (R5.4),
        wavelengths forwarded (R2.3), the full label set passed to the metric
        computation (R0.4f), and the raw predictions stashed for the collapse
        stop condition (R0.3)."""
        self.model.eval()
        total_loss = total_cls = 0.0
        total_n = 0
        spectral_accum: Dict[str, float] = {}
        spectral_batches = 0
        all_preds, all_labels, all_probs = [], [], []
        per_sample_metrics_accum: Dict[str, list] = {}
        recon_true_means, recon_pred_means = [], []
        recon_collected = 0

        amp_enabled, amp_dtype = self._resolve_val_amp()

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                out = self._model_forward(self.model, x)
                logits, x_recon = out if isinstance(out, tuple) else (out, None)
                cls_loss = self.criterion(logits, y)
                mse_loss = (F.mse_loss(x_recon, x) if x_recon is not None
                            else torch.tensor(0.0, device=self.device))
                loss = cls_loss + self.lambda_mse * mse_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * x.size(0)
            total_cls += cls_loss.item() * x.size(0)
            total_n += x.size(0)

            probs = F.softmax(logits.float(), dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

            if x_recon is not None:
                wl_tensor = self._wl if self._wl is not None else (
                    torch.as_tensor(self.wavelengths, device=x.device) if self.wavelengths is not None else None)
                batch_spectral = compute_all_spectral_metrics(x, x_recon, wavelengths=wl_tensor)
                for k, v in batch_spectral.items():
                    spectral_accum[k] = spectral_accum.get(k, 0.0) + v * x.size(0)
                spectral_batches += x.size(0)

                if recon_collected < self.recon_sample_cap:
                    per_sample = compute_per_sample_spectral_metrics(x, x_recon, wavelengths=wl_tensor)
                    for k, v in per_sample.items():
                        per_sample_metrics_accum.setdefault(k, []).extend(v)
                    remaining = self.recon_sample_cap - recon_collected
                    recon_true_means.append(x.float().mean(dim=(2, 3)).cpu().numpy()[:remaining])
                    recon_pred_means.append(x_recon.float().mean(dim=(2, 3)).cpu().numpy()[:remaining])
                    recon_collected += min(x.size(0), remaining)

        preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
        labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
        probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 0))

        label_set = list(range(probs.shape[1])) if probs.size else None
        classification = (compute_classification_metrics(labels, preds, labels=label_set)
                           if len(labels) > 0
                           else {"accuracy": 0.0, "balanced_accuracy": 0.0, "precision_macro": 0.0,
                                 "recall_macro": 0.0, "f1_macro": 0.0, "precision_weighted": 0.0,
                                 "recall_weighted": 0.0, "f1_weighted": 0.0})

        spectral = ({k: v / max(1, spectral_batches) for k, v in spectral_accum.items()}
                    if spectral_batches > 0 else {})

        denom = max(1, total_n)
        return {
            "loss": total_loss / denom, "cls_loss": total_cls / denom,
            "classification": classification, "spectral": spectral,
            "preds": preds, "labels": labels, "probs": probs,
            "recon_per_sample_metrics": per_sample_metrics_accum,
            "recon_true_means": np.concatenate(recon_true_means, axis=0) if recon_true_means else None,
            "recon_pred_means": np.concatenate(recon_pred_means, axis=0) if recon_pred_means else None,
        }

    def _validate_one_epoch(self):
        """Same EMA swap as `TrainerG_v10`, around v11's own validation loop."""
        if self.ema_helper is None:
            out = self._validate_one_epoch_impl()
        else:
            live_model = self.model
            self.model = self.ema_helper.ema_copy(live_model)
            try:
                out = self._validate_one_epoch_impl()
            finally:
                self.model = live_model
        self._current_val_preds = out.get("preds")
        return out

    # ==================================================================
    # R0.3 - class collapse is a stop condition
    # ==================================================================

    def _check_class_collapse(self, classification_metrics) -> List[str]:
        """`TrainerG_v5`'s zero-recall warnings, plus the streak that ACTS.

        The two are different questions. v5 asks "has some class had zero
        recall for N epochs" - true whenever one class is merely hard. This
        asks "did the model emit exactly ONE distinct label across the entire
        validation split", which is representation collapse and is not
        recoverable by waiting.
        """
        messages = list(super()._check_class_collapse(classification_metrics))

        preds = self._current_val_preds
        n_distinct = int(len(np.unique(preds))) if preds is not None and len(preds) else 0
        if n_distinct == 1:
            self._single_class_streak += 1
            messages.append(
                f"every one of the {len(preds)} validation samples was assigned the same class "
                f"(streak: {self._single_class_streak}/{self.class_collapse_streak}) - the model is "
                f"a constant function of its input (REPRESENTATION_COLLAPSE)")
        else:
            self._single_class_streak = 0

        if (self.on_class_collapse == "abort"
                and self._single_class_streak >= self.class_collapse_streak):
            self._abort_on_representation_collapse(n_distinct, preds)
        return messages

    def _abort_on_representation_collapse(self, n_distinct, preds) -> None:
        self.stop_reason = _StopReasonStr(COLLAPSE_STOP_REASON)
        self.is_stopped = True
        self.logger.error(
            f"[representation-collapse] ABORTING RUN: the validation split has been assigned a "
            f"single class for {self._single_class_streak} consecutive epochs. Re-run "
            f"`pytest test_representation_sensitivity.py` and check gates G1/G2 before spending "
            f"more GPU time (v15 plan, R0.3/R1.1).")
        if self._collapse_failure_written:
            return
        try:
            failure_dir = Path(self.exp_dir) / "collapse_failure"
            failure_dir.mkdir(parents=True, exist_ok=True)
            hist = (np.bincount(preds).tolist() if preds is not None and len(preds) else [])
            payload = {
                "failure_class": FailureClass.REPRESENTATION_COLLAPSE.value,
                "stop_reason": COLLAPSE_STOP_REASON,
                "epoch": self.current_epoch,
                "streak": self._single_class_streak,
                "class_collapse_streak_threshold": self.class_collapse_streak,
                "distinct_predicted_classes": n_distinct,
                "predicted_label_histogram": hist,
                "n_validation_samples": int(len(preds)) if preds is not None else 0,
                "guidance": (
                    "The model produced one distinct label for the entire validation split. This is "
                    "representation collapse, not slow convergence. Check gate G1 (stem sensitivity "
                    ">= 0.05) and G2 (logit std >= 1e-2) with "
                    "`pytest test_representation_sensitivity.py`, and verify "
                    "`spectral_token_fusion` is not left at the legacy 'add'."),
            }
            with open(failure_dir / "failure.json", "w") as f:
                json.dump(payload, f, indent=2, default=str)
            self._collapse_failure_written = True
            self.logger.error(f"[representation-collapse] failure artifacts written to {failure_dir}")
        except Exception as e:
            self.logger.warning(f"[representation-collapse] failed to write failure artifacts: {e}")

    # ==================================================================
    # R5.6 - artifact retention
    # ==================================================================

    def _save_eval_artifacts(self, epoch, val_metrics):
        """R5.6 - write the per-epoch confusion matrix PNG, classification
        report and prediction `.npz` only every `eval_artifact_stride` epochs
        (and always on the last one, and always when the epoch is the best so
        far). The HSI run wrote six byte-identical PNGs and six 334,516-row
        `.npz` files for a model that predicted one class throughout."""
        if not self._should_write_eval_artifacts(epoch):
            return
        return super()._save_eval_artifacts(epoch, val_metrics)

    def _should_write_eval_artifacts(self, epoch) -> bool:
        if self.eval_artifact_stride <= 1:
            return True
        return (epoch == 1 or epoch % self.eval_artifact_stride == 0
                or epoch == self.best_epoch or self.is_stopped)

    def _save_checkpoint(self, is_best):
        """R5.3 + R5.6.

        R5.3: when EMA drives validation, `best_model.pt` holds the EMA
        weights - the weights whose metric was reported - and the live weights
        go to `best_model_live.pt`. `TrainerG_v10` saved the live weights as
        `best_model.pt` and only a bare shadow dict as `best_model_ema.pt`, so
        `best_model.pt` reproduced nothing and `best_model_ema.pt` was not
        loadable as a model. Gate G5.

        R5.6: `--keep_last_n` epoch checkpoints, and the training history is
        no longer embedded in every checkpoint (5,399,147 -> 5,407,403 bytes
        over six epochs, growing every epoch, duplicating `history.json`).
        """
        state = {
            "epoch": self.current_epoch, "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "best_metric_value": self.best_metric_value,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "stall_counters": self.stall_counters,
            # C-4 - a checkpoint is tied to the tokenizer it was trained with.
            "spectral_token_fusion": self._spectral_token_fusion(),
            "validated_weights": "ema" if self.ema_helper is not None else "live",
        }
        if self.embed_history_in_checkpoint:
            state["history"] = self.history
        if self.discriminator is not None:
            state["discriminator_state"] = self.discriminator.state_dict()
            state["disc_optimizer_state"] = self.disc_optimizer.state_dict()

        torch.save(state, self.dirs["checkpoints"] / f"epoch_{self.current_epoch:04d}.pt")
        torch.save(state, self.dirs["checkpoints"] / "latest.pt")

        if is_best:
            if self.ema_helper is None:
                torch.save(state, self.exp_dir / "best_model.pt")
            else:
                ema_model = self.ema_helper.ema_copy(self.model)
                ema_state = dict(state)
                ema_state["model_state"] = ema_model.state_dict()
                ema_state["weights"] = "ema"
                torch.save(ema_state, self.exp_dir / "best_model.pt")
                live_state = dict(state)
                live_state["weights"] = "live"
                torch.save(live_state, self.exp_dir / "best_model_live.pt")
                del ema_model
            # `latest_ema.pt` / `best_model_ema.pt` (the bare shadow dicts) are
            # still written by TrainerG_v10 for backward compatibility.
        if self.ema_helper is not None:
            ema_shadow = {"ema_shadow": self.ema_helper.state_dict(), "epoch": self.current_epoch}
            torch.save(ema_shadow, self.dirs["checkpoints"] / "latest_ema.pt")
            if is_best:
                torch.save(ema_shadow, self.exp_dir / "best_model_ema.pt")

        self._prune_old_checkpoints()

    def _spectral_token_fusion(self) -> Optional[str]:
        base = getattr(self.model, "base_model", self.model)
        cfg = getattr(base, "cfg", None)
        return getattr(cfg, "spectral_token_fusion", None) if cfg is not None else None

    def _prune_old_checkpoints(self) -> None:
        """R5.6 - keep the newest `keep_last_n` per-epoch checkpoints. `latest.pt`,
        `best_model*.pt` and `latest_ema.pt` live outside this glob and are never
        touched. `keep_last_n <= 0` disables pruning."""
        if self.keep_last_n <= 0:
            return
        ckpts = sorted(self.dirs["checkpoints"].glob("epoch_*.pt"))
        for stale in ckpts[:-self.keep_last_n]:
            try:
                stale.unlink()
            except OSError as e:
                self.logger.warning(f"could not remove old checkpoint {stale}: {e}")

    def load_checkpoint(self, path):
        """C-4 - a checkpoint trained under one tokenizer mode is never
        silently loaded into another. The tokenizer mode changes what the
        parameters MEAN (and, for `concat_mlp`, which parameters exist at
        all), so a mismatch raises instead of producing a plausible-looking
        run from mismatched weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        recorded = checkpoint.get("spectral_token_fusion")
        current = self._spectral_token_fusion()
        if recorded is not None and current is not None and recorded != current:
            raise ValueError(
                f"checkpoint {path} was trained with spectral_token_fusion={recorded!r} but this "
                f"model is built with {current!r}. These are different tokenizers - "
                f"'concat_mlp' has no `value_embed` at all - so the weights do not transfer. "
                f"Rebuild the model with spectral_token_fusion={recorded!r}, or start from scratch.")
        if recorded is None and current is not None and current != "add":
            self.logger.warning(
                f"checkpoint {path} predates `spectral_token_fusion` (so it is 'add') but this "
                f"model is built with {current!r}; loading with strict=False. Expect the "
                f"tokenizer's weights to be freshly initialised.")
        super().load_checkpoint(path)
        # `TrainerG_v10.load_checkpoint` calls super() and discards its return
        # value, so it returns None - and `train_example_v14.py:1031` does
        # `start_epoch = trainer.load_checkpoint(args.resume) + 1`, which raises
        # TypeError. `--resume` has therefore never worked under v14. v10 is
        # frozen by contract C-2 and v14 by C-1, so the fix lives here: return
        # the epoch the way `TrainerG_v4.load_checkpoint` does.
        return self.current_epoch

    # ==================================================================
    # R5.5 - test-split evaluation (gate G6)
    # ==================================================================

    def evaluate_test_split(self, test_loader, checkpoint_path=None, num_classes=None) -> Optional[Dict]:
        """Gate G6 - ONE pass over the held-out test split with the best
        checkpoint, written to `<exp_dir>/test_report.json`.

        Both datasets ship a patient-disjoint test split (the HSI one holds out
        five patients) and neither has ever been evaluated: `--eval_test` did
        not exist, and `training/evaluator.py`, `training/per_class_metrics.py`,
        `training/metrics.py`'s calibration block and `training/flops_counter.py`
        were all present and never invoked.

        Returns the report dict, or None if the test split is unavailable.
        """
        from training.evaluator import evaluate_model
        from training.metrics import calibration_metrics
        from training.per_class_metrics import compute_per_class_extended_metrics

        if test_loader is None:
            self.logger.warning("[test-eval] no test loader - skipping (gate G6 NOT satisfied)")
            return None

        ckpt = Path(checkpoint_path) if checkpoint_path else (self.exp_dir / "best_model.pt")
        if not ckpt.is_file():
            self.logger.warning(f"[test-eval] {ckpt} does not exist - skipping (gate G6 NOT satisfied)")
            return None

        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.logger.info(f"[test-eval] loaded {ckpt.name} "
                          f"(epoch {state.get('epoch')}, weights={state.get('weights', 'live')})")

        result = evaluate_model(self.model, test_loader, str(self.device),
                                 wavelengths=self._wl, sensor_range=self.sensor_range,
                                 class_names=self.class_names)
        preds, labels, probs = result["preds"], result["labels"], result["probs"]
        n_classes = int(num_classes or probs.shape[1])

        report = {
            "checkpoint": str(ckpt),
            "checkpoint_epoch": state.get("epoch"),
            "checkpoint_weights": state.get("weights", "live"),
            "best_epoch": self.best_epoch,
            "n_test_samples": int(len(labels)),
            "num_classes": n_classes,
            "class_names": self.class_names,
            "amp_mode": self.amp_mode,
            "sklearn_metrics": compute_classification_metrics(labels, preds,
                                                               labels=list(range(n_classes))),
            "evaluator_metrics": result["accuracy"],
            "per_class_extended": compute_per_class_extended_metrics(
                labels, preds, n_classes, class_names=self.class_names),
            "calibration": calibration_metrics(probs, labels),
            "efficiency": result["efficiency"],
            "spectral": result["spectral"],
            "predicted_label_histogram": np.bincount(preds, minlength=n_classes).tolist(),
            "true_label_histogram": np.bincount(labels, minlength=n_classes).tolist(),
        }
        try:
            from training.flops_counter import count_flops
            sample_x, _ = next(iter(test_loader))
            report["flops"] = count_flops(self.model, tuple(sample_x.shape[1:]),
                                           device=str(self.device))
        except Exception as e:
            report["flops"] = {"error": f"{type(e).__name__}: {e}"}

        with open(self.exp_dir / "test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        np.savez(self.exp_dir / "test_predictions.npz",
                  true_label=labels, predicted_label=preds, probs=probs)
        m = report["sklearn_metrics"]
        self.logger.info(
            f"[test-eval] n={len(labels)} accuracy={m['accuracy']:.4f} "
            f"balanced={m['balanced_accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
            f"-> {self.exp_dir / 'test_report.json'}")
        return report

    def verify_best_checkpoint_reproduces(self, tolerance: float = 1e-6) -> Dict:
        """Gate G5 - reload `best_model.pt` and re-run the validation split;
        the checkpoint metric must come back to within `tolerance` of the
        logged best. Cheap, and the only way to know the number in the report
        belongs to the weights on disk."""
        ckpt = self.exp_dir / "best_model.pt"
        out = {"gate": "G5", "checkpoint": str(ckpt), "tolerance": tolerance, "passed": False}
        if not ckpt.is_file() or not self.history:
            out["reason"] = "no best_model.pt or no history"
            return out

        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        live_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            self.model.load_state_dict(state["model_state"])
            # Bypass the EMA swap: `best_model.pt` already IS the reported model.
            metrics = self._validate_one_epoch_impl()
        finally:
            self.model.load_state_dict(live_state)

        reproduced = metrics["classification"].get(self.checkpoint_metric)
        logged = self.best_metric_value
        out.update(metric=self.checkpoint_metric, logged=logged, reproduced=reproduced,
                    delta=(None if reproduced is None else abs(reproduced - logged)))
        out["passed"] = reproduced is not None and abs(reproduced - logged) <= tolerance
        with open(self.exp_dir / "checkpoint_reproducibility.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        level = self.logger.info if out["passed"] else self.logger.warning
        level(f"[gate G5] {self.checkpoint_metric}: logged {logged!r} vs reloaded {reproduced!r} "
              f"-> {'PASS' if out['passed'] else 'FAIL'}")
        return out

    # ==================================================================
    # fit - R5.1/R5.2 in the metrics dict, R0.3 as a stop condition
    # ==================================================================

    def fit(self, max_epochs, start_epoch=1):
        """`TrainerG_v5.fit` with the v15 reporting items folded in. Copied
        rather than hooked because the metrics dict is assembled inline there
        and every v15 reporting fix is a change to that dict."""
        self.current_epoch = start_epoch - 1
        self.logger.info(f"[TrainerG_v11] epochs {start_epoch}->{max_epochs}, "
                          f"checkpoint_metric={self.checkpoint_metric!r}, "
                          f"on_class_collapse={self.on_class_collapse!r}, "
                          f"amp={self.amp_mode!r}, keep_last_n={self.keep_last_n}, "
                          f"eval_artifact_stride={self.eval_artifact_stride}")
        total_steps = 0
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                if self.is_stopped:
                    break
                self.current_epoch = epoch
                epoch_start_time = time.time()
                if self.device_type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                # R6.3 - a different augmentation draw per epoch.
                if self.augmented_train_dataset is not None:
                    self.augmented_train_dataset.set_epoch(epoch)

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
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                total_steps += int(train_metrics.get("n_optimizer_steps", 0))

                gpu_mem_mb = (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                              if self.device_type == "cuda" else 0.0)

                metrics = {
                    "epoch": epoch,
                    # R7.2 - report steps alongside epochs everywhere. "6 epochs
                    # vs 33 epochs" compared 2,868 optimizer steps with 105,336.
                    "optimizer_steps_this_epoch": int(train_metrics.get("n_optimizer_steps", 0)),
                    "optimizer_steps_total": total_steps,
                    "train_loss": train_metrics["loss"],
                    "classification_loss": train_metrics["cls_loss"],
                    # R5.2 - the segment-matched CE, comparable with val_cls_loss.
                    "train_cls_loss_last_segment": train_metrics.get("cls_loss_last_segment"),
                    # R5.1 - previously computed, returned, and then dropped.
                    "halt_loss": train_metrics.get("halt_loss", 0.0),
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
                    "val_mse_loss": val_metrics.get("mse_loss", 0.0),
                    "val_sam_loss": val_metrics.get("sam_loss", 0.0),
                    "val_gan_loss": val_metrics.get("gan_loss", 0.0),
                    **val_metrics["classification"],
                    **{f"spectral_{k}": v for k, v in val_metrics["spectral"].items()},
                    "gradient_health": train_metrics["gradient_health"],
                    "is_valid_epoch": train_metrics["gradient_health"]["is_valid_epoch"],
                    "parameter_norm": parameter_norm,
                    "update_norm": update_norm,
                    "validated_weights": "ema" if self.ema_helper is not None else "live",
                }
                metrics["val_accuracy"] = metrics.pop("accuracy")

                # R5.2 - the generalization gap, from LAST-SEGMENT CE on both
                # sides. `train_loss` is the mean over all deep-supervision
                # segments plus 0.5*halt_loss, on augmented data, from the live
                # weights; `val_loss` is last-segment CE on clean data from the
                # EMA weights. Differencing them produced the RGB run's
                # identical `GenGap(loss): -0.2247` on all 33 epochs. The raw
                # difference is kept under its own name so nothing that reads
                # it breaks, but the headline number is the comparable one.
                train_last = metrics.get("train_cls_loss_last_segment")
                metrics["generalization_gap_loss_raw"] = metrics["val_loss"] - metrics["train_loss"]
                metrics["generalization_gap_loss"] = (
                    metrics["val_cls_loss"] - train_last if train_last is not None
                    else metrics["generalization_gap_loss_raw"])
                metrics["generalization_gap_loss_is_comparable"] = train_last is not None
                # `train_accuracy` is the LIVE model's, `val_accuracy` the EMA
                # model's, so this compares two different networks whenever EMA
                # is on. Flagged rather than silently reported (R5.3).
                metrics["generalization_gap_accuracy"] = metrics["train_accuracy"] - metrics["val_accuracy"]
                metrics["generalization_gap_accuracy_is_comparable"] = self.ema_helper is None

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
                metrics["single_class_streak"] = self._single_class_streak
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

    def _log_epoch(self, m, is_best):
        """R5.1/R5.2 - the printed breakdown now accounts for the whole loss,
        and says which generalization gap it is showing."""
        super()._log_epoch(m, is_best)
        extra = [f"Halt: {m.get('halt_loss', 0.0):.4f}"]
        if m.get("train_cls_loss_last_segment") is not None:
            extra.append(f"CE(last segment): {m['train_cls_loss_last_segment']:.4f}")
            extra.append(f"GenGap(CE, matched): {m['generalization_gap_loss']:+.4f}")
        extra.append(f"steps: {m.get('optimizer_steps_this_epoch', 0)} "
                     f"(total {m.get('optimizer_steps_total', 0)})")
        if not m.get("generalization_gap_accuracy_is_comparable", True):
            extra.append("GenGap(acc) compares LIVE train vs EMA val - not comparable")
        print("  [v15] " + " | ".join(extra), flush=True)
