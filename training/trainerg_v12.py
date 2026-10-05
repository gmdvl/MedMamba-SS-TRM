# -*- coding: utf-8 -*-
"""
training/trainerg_v12.py
==========================
MedMamba-SS-TRM v16 plan - `TrainerG_v12(TrainerG_v11)`. `TrainerG_v11` (and
everything it inherits from) is FROZEN - `train_example_v15.py` keeps the
trainer it was audited with. This subclass overrides only what the v16 audit
found broken; everything else (AMP, per-component numerical-stability gating,
gradient health, class-collapse monitoring, checkpointing, EMA, R5.3-R5.6)
is inherited unchanged.

  R-1/R-2/R-5  `_forward_with_recon` - the ONE decode path for both training
               and validation. For a recursive base + decoder, it decodes the
               LIVE final feature map
               (`training.recursive_features.forward_deep_supervision_with_features`,
               R-2) instead of the detached `base.last_feature_map`
               (`trainerg_v10.py:125`) or a second, independent segment
               recursion through `backbone.forward_features` (R-5's "two
               different segment recursions").
  R-3/R-7      Reconstruction metrics are computed in REFLECTANCE units
               (`training.normalization.denormalize` + `training.
               spectral_recon_metrics_v2`), not the network's normalized
               input/output space.
  R-4          Validation MSE/SAM/GAN are real accumulated quantities,
               returned in the validation dict and indexed DIRECTLY in
               `_build_epoch_metrics` (no `.get(..., 0.0)` - a missing key
               fails loudly, per the plan: "That defaulting is what hid this
               for four trainer versions, and deleting it is half the fix").
  R-6          `_update_loss_components_file` / `_save_per_class_metrics`
               (`TrainerG_v4`, inherited, correct) are actually called again;
               `TrainerG_v5.fit`/`TrainerG_v11.fit` stopped calling them.
               `plots_v16.generate_all_training_plots` is used for the
               non-zero-reconstruction-loss plot gate.
  E-1          `_check_stopping_rules` watches `self.checkpoint_metric`
               (macro-F1/balanced accuracy), not raw `val_accuracy`, and its
               patience is `--val_divergence_patience` (default 3, 0 disables).
  E-2          the LR schedule can be stepped on the STEP axis
               (`scheduler_interval="step"`), so `--train_subsample_frac`
               and early stopping no longer desynchronize it from wall-clock
               "epochs".
  A-4          per-patient / per-capture validation breakdown
               (`per_patient_metrics.json`, `per_capture_metrics.csv`), and a
               `PATIENT_OUTLIER` warning in `history.json` when a single
               patient's recall drops while the macro metric rises.
  Stage 5      per-epoch confusion-matrix/classification-report files move
               into `confusion_matrix/` and `classification_reports/`
               instead of littering the run root.
  C-1          `evaluate_test_split` gains `test_groups`/`eval_group_aggregation`
               so gate G6 can additionally report IMAGE-level metrics for
               patch-MIL datasets (PAD-UFES-20's 11x11 patches inherit a
               whole-clinical-image diagnosis) - reported as patch-MIL, not
               a like-for-like modality comparison with the HSI arm.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from training.trainerg_v11 import TrainerG_v11
from training.trainerg_v7 import _StopReasonStr
from training.trainerg_v4 import StopReason
from training.eval_utils import (
    compute_classification_metrics, sanity_check_metrics,
    write_classification_report, save_confusion_matrix, save_prediction_distribution,
)
from training.gan import discriminator_loss, generator_adversarial_loss
from training.numerical_stability import (
    inspect_gradients, check_parameters_finite, check_loss_components, format_loss_component_status,
)
# v17 S5 - fused, one-sync equivalents of the first two. See fused_stability_checks's docstring:
# the frozen versions do 4 device syncs PER PARAMETER TENSOR, i.e. 292 per step on this
# model. The frozen names stay imported: the failure path delegates to them.
from training.fused_stability_checks import inspect_gradients_fused, check_parameters_finite_fused
from training.progress import ProgressReporter, format_duration
from training.recursive_features import forward_deep_supervision_with_features
from training.normalization import denormalize
from training.spectral_recon_metrics_v2 import (
    compute_all_spectral_metrics as compute_all_spectral_metrics_v2,
    compute_per_sample_spectral_metrics as compute_per_sample_spectral_metrics_v2,
    NonReflectanceDataError,
)


def aggregate_group_predictions(probs: np.ndarray, labels: np.ndarray, group_ids: np.ndarray,
                                 method: str = "mean_prob") -> Dict:
    """C-1 - collapse per-patch predictions to one prediction per group
    (clinical image, or patient/capture) via `"mean_prob"` (average the
    softmax posteriors, then argmax) or `"majority"` (majority vote over
    per-patch argmax predictions). The group's true label is the majority
    label among its patches (by construction every patch in a group carries
    the same whole-item label, so this is just a robust way to read it off)."""
    if method not in ("mean_prob", "majority"):
        raise ValueError(f"eval_group_aggregation must be 'mean_prob' | 'majority', got {method!r}")
    group_ids = np.asarray(group_ids)
    unique_groups = sorted(set(group_ids.tolist()))
    agg_preds, agg_labels = [], []
    for g in unique_groups:
        mask = group_ids == g
        g_labels = labels[mask]
        true_label = int(np.bincount(g_labels).argmax())
        if method == "mean_prob":
            pred = int(probs[mask].mean(axis=0).argmax())
        else:
            patch_preds = probs[mask].argmax(axis=1)
            pred = int(np.bincount(patch_preds).argmax())
        agg_preds.append(pred)
        agg_labels.append(true_label)

    agg_preds_arr = np.asarray(agg_preds)
    agg_labels_arr = np.asarray(agg_labels)
    metrics = compute_classification_metrics(agg_labels_arr, agg_preds_arr,
                                              labels=list(range(probs.shape[1])))
    return {"n_groups": len(unique_groups), "method": method, **metrics}


def _timed_iter(loader, sink: Dict[str, float]):
    """v17 S1 - yield from `loader`, accumulating into `sink["s"]` the time spent BLOCKED
    waiting for a batch.

    A wrapper rather than two `time.time()` calls around the loop body, because
    `_train_one_epoch_deep_supervision` has `continue` and `break` paths: a timestamp reset
    at the bottom of the body would be skipped by a `continue` and the next batch's wait
    would silently absorb the previous batch's compute. Timing `next()` itself cannot get
    that wrong.

    What the number means: with a prefetching DataLoader `next()` returns immediately when
    a batch is ready, so a non-trivial `data_wait_s` is genuine input starvation - too few
    workers, too small a prefetch, or per-sample augmentation that costs more than a step.
    """
    it = iter(loader)
    while True:
        t0 = time.time()
        try:
            batch = next(it)
        except StopIteration:
            sink["s"] += time.time() - t0
            return
        sink["s"] += time.time() - t0
        yield batch


class _LogitsOnlyView(torch.nn.Module):
    """Presents a reconstruction-WRAPPED model to the frozen evaluation path
    as if it returned logits alone.

    `MedMambaSSTRMLatentReconWrapperV2.forward` (and `MedMambaSSTRMRawReconWrapper`'s)
    returns `(logits, x_recon)` unconditionally - there is no eval-mode
    special case. `training/evaluator.py:33` does `F.softmax(out, dim=-1)`
    on whatever `model(x)` returned, so gate G6 dies with
    `AttributeError: 'tuple' object has no attribute 'softmax'` for any run
    with `--recon_mode != none`. `training/evaluator.py` is not on the FROZEN
    list, but `training/trainerg_v11.evaluate_test_split` imports it, so
    editing it would change behaviour for the two audited v15 runs; this
    subclass is the correct place to adapt instead.

    Two further latent failures on the same path are fixed by the same
    object, both of the same shape - the frozen evaluator reaches for an
    attribute a reconstruction wrapper does not define, only its
    `base_model` does: `evaluate_model` reads `.cfg` to decide whether
    band-selection metrics apply, and `spectral_metrics.collect_band_weights`
    reads `.backbone` to run the backbone directly. `dynamic_band_selection`
    is True for the HSI preset, so both are reached in practice.

    `state_dict`/`load_state_dict` delegate to the inner module so the
    checkpoint's keys still match: registering `inner` as a submodule would
    otherwise prefix every key with `inner.`.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def __getattr__(self, name):
        """Everything except `forward` is delegated, so the frozen evaluation
        path can reach through the wrapper for whatever it needs. It needs at
        least two things a reconstruction wrapper does not define: `cfg`
        (`evaluator.evaluate_model`, to decide whether band-selection metrics
        apply) and `backbone` (`spectral_metrics.collect_band_weights`, which
        runs the backbone directly to reach `band_weights`). Both live on the
        wrapped `base_model`, so the lookup walks the `base_model` chain."""
        try:
            return super().__getattr__(name)                 # params/buffers/submodules
        except AttributeError:
            module = super().__getattr__("inner")
            while module is not None:
                if hasattr(module, name):
                    return getattr(module, name)
                nxt = getattr(module, "base_model", None)
                module = None if nxt is module else nxt
            raise

    def state_dict(self, *a, **kw):
        return self.inner.state_dict(*a, **kw)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.inner.load_state_dict(state_dict, strict=strict)

    def forward(self, *a, **kw):
        out = self.inner(*a, **kw)
        return out[0] if isinstance(out, tuple) else out


class TrainerG_v12(TrainerG_v11):
    def __init__(self, *args,
                 val_groups: Optional[np.ndarray] = None,
                 val_captures: Optional[np.ndarray] = None,
                 capture_index: Optional[Dict] = None,
                 min_patient_recall: Optional[float] = None,
                 val_divergence_patience: int = 3,
                 scheduler_interval: str = "epoch",
                 **kwargs):
        super().__init__(*args, **kwargs)
        if scheduler_interval not in ("epoch", "step"):
            raise ValueError(f"scheduler_interval must be 'epoch' | 'step', got {scheduler_interval!r}")
        self.scheduler_interval = scheduler_interval
        self.val_divergence_patience = int(val_divergence_patience)

        # A-4 - aligned to the (possibly subsampled) val split's iteration
        # order: `val_loader` is built with shuffle=False, so batch order ==
        # dataset/Subset order == the order these arrays were built in.
        self.val_groups = np.asarray(val_groups) if val_groups is not None else None
        self.val_captures = np.asarray(val_captures) if val_captures is not None else None
        self.capture_index = capture_index or {}
        self.min_patient_recall = min_patient_recall

        # Stage 5 - per-epoch artifacts get their own folders instead of the
        # run root (trainerg_v4.py:143-152 sets up self.dirs; extending it
        # here is a pure v12 addition, no shared-module change needed since
        # training/eval_utils.py already takes an out_dir/out_path argument).
        self.dirs["confusion_matrix"] = self.exp_dir / "confusion_matrix"
        self.dirs["classification_reports"] = self.exp_dir / "classification_reports"
        for d in ("confusion_matrix", "classification_reports"):
            self.dirs[d].mkdir(parents=True, exist_ok=True)

        # v17 S5 - the GradScaler is only for fp16.
        #
        # `training/trainerg_v4.py:133` builds `GradScaler(device=self.device_type)` with
        # no `enabled=`, i.e. ALWAYS ON, and nothing in the v9..v11 chain overrides it. So
        # every `--amp bf16` run has been scaling the loss by 65536, unscaling all 73
        # gradient tensors, running a `found_inf` reduction and syncing on it inside
        # `scaler.step` - on a dtype with fp32's exponent range, which is exactly why bf16
        # exists and why PyTorch's own AMP guidance says no scaler is needed for it. Under
        # `--amp off` it did the same to fp32.
        #
        # Scaling by a power of two is exact, so this removes work, not precision: results
        # are unchanged. `trainerg_v4.py` is frozen, so the object is replaced here rather
        # than constructed differently there. `scaler_state` stays in the checkpoint
        # (`trainerg_v11.py:569`) - a disabled scaler still serialises, so checkpoints
        # written before v17 keep loading.
        _amp_enabled, _amp_dtype = self._resolve_amp()
        _need_scaler = bool(_amp_enabled and _amp_dtype is torch.float16)
        if self.scaler.is_enabled() != _need_scaler:
            self.scaler = GradScaler(device=self.device_type, enabled=_need_scaler)
            self.logger.info(f"[amp] GradScaler {'enabled' if _need_scaler else 'DISABLED'} "
                             f"(amp_mode={self.amp_mode!r}): loss scaling is an fp16-only "
                             f"requirement.")

    def load_checkpoint(self, path):
        """`TrainerG_v4.load_checkpoint`, tolerant of a checkpoint written with the v17
        S5 scaler disabled.

        `trainerg_v4.py:826` calls `self.scaler.load_state_dict(checkpoint["scaler_state"])`
        unguarded, and a DISABLED `GradScaler` serialises to `{}`. Feeding `{}` to an
        ENABLED scaler raises `RuntimeError: The source state dict is empty`. That is
        exactly the cross-precision resume `--amp bf16` (trains, saves `{}`) then
        `--resume ... --amp fp16` (wants a scaler), which worked before v17 and would now
        abort at load. The reverse - an old enabled state into a disabled scaler - is
        already a silent no-op in PyTorch, so a pre-v17 checkpoint resumes unchanged.

        Only the empty-into-enabled case is skipped; a genuine fp16 -> fp16 resume still
        restores its scale, which is why this shadows the method for the duration of the
        call rather than disabling the scaler around it.
        """
        real_load = self.scaler.load_state_dict

        def _tolerant(state):
            if self.scaler.is_enabled() and not state:
                self.logger.warning(
                    "[amp] checkpoint carries no GradScaler state (it was written under "
                    "bf16/fp32, where v17 disables the scaler). Resuming under fp16 with a "
                    "fresh scale; the first few steps may be skipped while it calibrates.")
                return
            real_load(state)

        self.scaler.load_state_dict = _tolerant
        try:
            return super().load_checkpoint(path)
        finally:
            del self.scaler.load_state_dict

    # ==================================================================
    # R-1/R-2/R-5 - the ONE decode path for training AND validation
    # ==================================================================

    def _forward_with_recon(self, model, x):
        """`-> (logits_list, q_list, x_recon)`.

        - recursive base + decoder -> `forward_deep_supervision_with_features`
          (R-2's LIVE feature map) -> `decoder(feat, target_hw=x.shape[-2:])`.
          Used by BOTH `_train_one_epoch_deep_supervision` and
          `_validate_one_epoch_impl` (R-5): one segment recursion, so train
          and val MSE/SAM measure the same thing.
        - recursive base, no decoder (`recon_mode=none`) -> deep supervision
          without reconstruction, same as `TrainerG_v11`.
        - anything else (non-recursive, or `recon_mode=raw_input`'s
          `MedMambaSSTRMRawReconWrapper`, which decodes from `x` directly and has
          no per-segment logits) -> the existing single-forward path
          (`_model_forward`, R2.3's wavelength plumbing), wrapped in a
          1-element logits list so callers have one contract either way.
        """
        base = model if hasattr(model, "forward_deep_supervision") else getattr(model, "base_model", None)
        decoder = getattr(model, "decoder", None)
        recursive = base is not None and hasattr(base, "forward_deep_supervision")

        if recursive and decoder is not None:
            if self._wl is not None and self._accepts_wavelengths_ds(base):
                logits_list, q_list, feat = forward_deep_supervision_with_features(
                    base, x, wavelengths=self._wl, sensor_range=self.sensor_range)
            else:
                logits_list, q_list, feat = forward_deep_supervision_with_features(base, x)
            x_recon = decoder(feat, target_hw=x.shape[-2:])
            return logits_list, q_list, x_recon

        if recursive:
            logits_list, q_list = self._deep_supervision_forward(base, x)
            return logits_list, q_list, None

        out = self._model_forward(model, x)
        logits, x_recon = out if isinstance(out, tuple) else (out, None)
        return [logits], None, x_recon

    # ==================================================================
    # Training epoch - R-5 single decode path (everything else identical
    # to TrainerG_v11._train_one_epoch_deep_supervision)
    # ==================================================================

    @staticmethod
    def _loader_len(loader) -> Optional[int]:
        """Batch count, or None when the loader cannot say (an IterableDataset has no
        `__len__`). `ProgressReporter` degrades to a count without a percentage or an ETA
        rather than crashing a multi-hour run over a progress line."""
        try:
            return len(loader)
        except TypeError:
            return None

    def _train_one_epoch_deep_supervision(self, base):
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
        halting = bool(getattr(base.cfg, "trm_act_halting", False))

        data_wait = {"s": 0.0}                                   # v17 S1, see `_timed_iter`
        # v17 S4 - an HSI epoch is 958 steps and eleven minutes, and printed nothing
        # between one epoch header and the next.
        n_steps_total = self._loader_len(self.train_loader)
        progress = ProgressReporter(f"[train] epoch {self.current_epoch}", min_interval=2.0)
        for batch in _timed_iter(self.train_loader, data_wait):
            x, y = batch[0], batch[1]
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                logits_list, q_list, x_recon = self._forward_with_recon(self.model, x)
                logits = logits_list[-1]
                per_segment = [self.criterion(l, y) for l in logits_list]
                cls_loss = sum(per_segment) / len(per_segment)
                cls_loss_last = per_segment[-1]

                if halting and q_list is not None:
                    targets = [(l.detach().argmax(dim=-1) == y).to(q_list[0].dtype) for l in logits_list]
                    halt_loss = sum(F.binary_cross_entropy_with_logits(q, t)
                                    for q, t in zip(q_list, targets)) / len(q_list)
                else:
                    halt_loss = torch.tensor(0.0, device=self.device)

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

            grad_inspect = inspect_gradients_fused(self.model)     # v17 S5
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
            # E-2 - step-axis LR schedule: advance on every successful
            # optimizer step, not once per (possibly subsampled) "epoch".
            if self.scheduler_interval == "step":
                self.scheduler.step()

            if self.ema_helper is not None:
                self.ema_helper.update(self.model)

            params_ok, bad_params = check_parameters_finite_fused(self.model)   # v17 S5
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
            progress.emit(n_batches, n_steps_total,
                          f"loss {total_loss / max(1, total_n):.4f} "
                          f"acc {total_correct / max(1, total_n):.4f}")

        progress.close()                                         # v17 S4b, see ProgressReporter
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
            "data_wait_s": data_wait["s"],                        # v17 S1
        }

    # ==================================================================
    # Validation - R-3/R-4/R-5/R-7
    # ==================================================================

    @torch.no_grad()
    def _validate_one_epoch_impl(self):
        self.model.eval()
        total_loss = total_cls = total_mse = total_sam = total_gan = 0.0
        total_n = 0
        spectral_accum: Dict[str, float] = {}
        spectral_batches = 0
        all_preds, all_labels, all_probs = [], [], []
        per_sample_metrics_accum: Dict[str, list] = {}
        recon_true_means, recon_pred_means = [], []
        recon_collected = 0

        amp_enabled, amp_dtype = self._resolve_val_amp()
        val_total = self._loader_len(self.val_loader)                      # v17 S4
        val_progress = ProgressReporter(f"[val]   epoch {self.current_epoch}", min_interval=2.0)
        n_val_batches = 0

        for batch in self.val_loader:
            if len(batch) == 3:
                x, y, norm_stats = batch
                norm_stats = norm_stats.to(self.device, non_blocking=True)
            else:
                x, y = batch[0], batch[1]
                norm_stats = None
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                logits_list, _q_list, x_recon = self._forward_with_recon(self.model, x)
                logits = logits_list[-1]
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

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_cls += cls_loss.item() * bs
            total_mse += mse_loss.item() * bs
            total_sam += sam_loss.item() * bs
            total_gan += gan_loss.item() * bs
            total_n += bs

            n_val_batches += 1
            val_progress.emit(n_val_batches, val_total)

            probs = F.softmax(logits.float(), dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

            if x_recon is not None:
                # R-3/R-7 - metrics in REFLECTANCE units. The loss above stays
                # in normalized space (scale-uniform, what an auxiliary loss
                # wants); the REPORTED metrics do not.
                if norm_stats is not None:
                    x_r = denormalize(x, norm_stats)
                    recon_r = denormalize(x_recon, norm_stats)
                    require_nonneg = True
                else:
                    x_r, recon_r = x, x_recon
                    require_nonneg = False
                wl_tensor = self._wl if self._wl is not None else (
                    torch.as_tensor(self.wavelengths, device=x.device) if self.wavelengths is not None else None)
                try:
                    batch_spectral = compute_all_spectral_metrics_v2(
                        x_r, recon_r, wavelengths=wl_tensor, require_nonnegative=require_nonneg)
                    for k, v in batch_spectral.items():
                        spectral_accum[k] = spectral_accum.get(k, 0.0) + v * bs
                    spectral_batches += bs

                    if recon_collected < self.recon_sample_cap:
                        remaining = self.recon_sample_cap - recon_collected
                        # R-6 bug fix (pre-existing in the frozen TrainerG_v4
                        # ..v11 lineage too, but never exercised there since
                        # R-1 force-disabled reconstruction on every real
                        # recursive run): `compute_per_sample_spectral_metrics`
                        # flattens over H*W, so calling it on the full
                        # [B,C,H,W] patch returns B*H*W PIXEL-level values,
                        # not one per SAMPLE - `plot_reconstruction_examples`
                        # expects `len(sam_per_sample) == len(x_true)`, where
                        # `x_true`/`x_pred` are the per-sample MEAN spectra
                        # (its own docstring), so indexing with an order
                        # derived from B*H*W pixel values ran off the end of
                        # the B-row mean arrays. Mean-pool spatially first
                        # (H=W=1 after pooling -> exactly one "pixel" = one
                        # sample) so the function's existing per-pixel
                        # semantics degenerate to per-sample, matching
                        # recon_true_means/recon_pred_means's granularity.
                        x_r_mean = x_r.mean(dim=(2, 3), keepdim=True)
                        recon_r_mean = recon_r.mean(dim=(2, 3), keepdim=True)
                        per_sample = compute_per_sample_spectral_metrics_v2(
                            x_r_mean, recon_r_mean, wavelengths=wl_tensor, require_nonnegative=require_nonneg)
                        for k, v in per_sample.items():
                            per_sample_metrics_accum.setdefault(k, []).extend(v[:remaining])
                        recon_true_means.append(x_r.float().mean(dim=(2, 3)).cpu().numpy()[:remaining])
                        recon_pred_means.append(recon_r.float().mean(dim=(2, 3)).cpu().numpy()[:remaining])
                        recon_collected += min(bs, remaining)
                except NonReflectanceDataError as e:
                    self.logger.warning(f"[spectral metrics] skipping batch: {e}")

        val_progress.close()                                     # v17 S4b
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
            # R-4 - real accumulated quantities now, not structural zeros.
            "mse_loss": total_mse / denom, "sam_loss": total_sam / denom, "gan_loss": total_gan / denom,
            "classification": classification, "spectral": spectral,
            "preds": preds, "labels": labels, "probs": probs,
            "recon_per_sample_metrics": per_sample_metrics_accum,
            "recon_true_means": np.concatenate(recon_true_means, axis=0) if recon_true_means else None,
            "recon_pred_means": np.concatenate(recon_pred_means, axis=0) if recon_pred_means else None,
        }

    # ==================================================================
    # E-1 - stopping rules watch the checkpoint metric, not raw accuracy
    # ==================================================================

    def _check_stopping_rules(self):
        if len(self.history) < 2:
            return
        curr, prev = self.history[-1], self.history[-2]
        metric = self.checkpoint_metric

        if self.val_divergence_patience > 0:
            if curr["val_loss"] > prev["val_loss"] and curr.get(metric) is not None \
                    and prev.get(metric) is not None and curr[metric] < prev[metric]:
                self._increment_stall("val_diverge")
            else:
                self.stall_counters["val_diverge"] = 0
            if self.stall_counters.get("val_diverge", 0) >= self.val_divergence_patience:
                self._trigger_stop(StopReason.VAL_DIVERGENCE)

        if curr.get(metric) == prev.get(metric):
            self._increment_stall("val_acc_stall")
        else:
            self.stall_counters["val_acc_stall"] = 0
        if self.stall_counters.get("val_acc_stall", 0) >= 5:
            self._trigger_stop(StopReason.VAL_ACC_STALLED)

    # ==================================================================
    # A-4 - per-patient / per-capture breakdown
    # ==================================================================

    def _per_group_recall(self, preds, labels, group_ids):
        """`{group_id: {"n": int, "recall_macro": float,
        "per_class": {class_idx: {"recall","precision","f1","n"}}}}`."""
        out = {}
        classes = sorted(set(labels.tolist()) | set(preds.tolist()))
        for gid in sorted(set(group_ids.tolist())):
            mask = group_ids == gid
            g_labels, g_preds = labels[mask], preds[mask]
            per_class = {}
            recalls = []
            for c in classes:
                c_mask = g_labels == c
                n_c = int(c_mask.sum())
                if n_c == 0:
                    continue
                tp = int(((g_preds == c) & c_mask).sum())
                fp = int(((g_preds == c) & ~c_mask).sum())
                recall = tp / n_c
                precision = tp / max(1, tp + fp)
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                per_class[str(c)] = {"recall": recall, "precision": precision, "f1": f1, "n": n_c}
                recalls.append(recall)
            out[str(gid)] = {
                "n": int(mask.sum()),
                "recall_macro": float(np.mean(recalls)) if recalls else None,
                "per_class": per_class,
            }
        return out

    def _save_per_patient_capture_metrics(self, epoch, val_metrics):
        preds, labels = val_metrics["preds"], val_metrics["labels"]
        if len(labels) == 0:
            return

        if self.val_groups is not None and len(self.val_groups) == len(labels):
            per_patient = self._per_group_recall(preds, labels, self.val_groups)
            class_names = self.class_names or [str(i) for i in range(int(labels.max()) + 1)]
            payload = {"epoch": epoch, "class_names": class_names, "per_patient": per_patient}
            with open(self.exp_dir / "per_patient_metrics.json", "w") as f:
                json.dump(payload, f, indent=2, default=str)
        else:
            per_patient = None

        if self.val_captures is not None and len(self.val_captures) == len(labels):
            per_capture = self._per_group_recall(preds, labels, self.val_captures)
            rows = []
            for cap_idx, entry in per_capture.items():
                meta = self.capture_index.get(cap_idx, self.capture_index.get(int(cap_idx), {}))
                rows.append({
                    "epoch": epoch, "capture_index": cap_idx,
                    "key": meta.get("key"), "patient": meta.get("patient"),
                    "tissue": meta.get("tissue"), "n": entry["n"],
                    "recall_macro": entry["recall_macro"],
                    "mean_raw_intensity": meta.get("median_intensity"),
                })
            if rows:
                csv_path = self.exp_dir / "per_capture_metrics.csv"
                write_header = not csv_path.exists()
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerows(rows)

        return per_patient

    def _check_patient_outliers(self, epoch_macro_metric, per_patient) -> List[str]:
        """PATIENT_OUTLIER - a single patient's recall drops below
        `--min_patient_recall` WHILE the macro metric rose (the shape of the
        v15 audit's Run A: patient 68's dark-slide IDC recall collapsed while
        the run kept reporting improving macro numbers)."""
        if self.min_patient_recall is None or per_patient is None:
            return []
        warnings = []
        prev_macro = self.history[-2].get(self.checkpoint_metric) if len(self.history) >= 2 else None
        macro_rose = prev_macro is not None and epoch_macro_metric is not None and epoch_macro_metric > prev_macro
        for pid, entry in per_patient.items():
            recall = entry.get("recall_macro")
            if recall is not None and recall < self.min_patient_recall:
                warnings.append(
                    f"PATIENT_OUTLIER: patient {pid} recall_macro={recall:.4f} < "
                    f"--min_patient_recall={self.min_patient_recall}"
                    + (" while the macro metric ROSE" if macro_rose else ""))
        return warnings

    # ==================================================================
    # Stage 5 - run-directory layout
    # ==================================================================

    def _save_eval_artifacts(self, epoch, val_metrics):
        if not self._should_write_eval_artifacts(epoch):
            return
        preds, labels, probs = val_metrics["preds"], val_metrics["labels"], val_metrics["probs"]
        if len(labels) == 0:
            return
        write_classification_report(
            labels, preds, self.dirs["classification_reports"] / f"classification_report_epoch{epoch:02d}.txt",
            class_names=self.class_names)
        save_confusion_matrix(labels, preds, str(self.dirs["confusion_matrix"]), epoch,
                               class_names=self.class_names)
        save_prediction_distribution(
            labels, preds, probs, self.dirs["predictions"] / f"predictions_epoch{epoch:02d}.npz")
        self._save_per_patient_capture_metrics(epoch, val_metrics)

    # ==================================================================
    # C-1 - image-level (patch-MIL) aggregation on the test split
    # ==================================================================

    def evaluate_test_split(self, test_loader, checkpoint_path=None, num_classes=None,
                             test_groups: Optional[np.ndarray] = None,
                             eval_group_aggregation: str = "none") -> Optional[Dict]:
        """`TrainerG_v11.evaluate_test_split` (gate G6), plus an
        IMAGE-level aggregation of the same per-patch predictions when
        `test_groups` (clinical image id per test patch, aligned to the test
        loader's iteration order - `scripts/derive_patch_groups.py`'s
        `images_{split}.npy` for `data/pad_v6`) and
        `eval_group_aggregation != "none"` are given.

        Reported under `report["image_level"]`. This is a patch-MIL result,
        not a like-for-like modality comparison with the HSI arm (C-1) - the
        11x11 patches were never meant to see enough of a clinical
        photograph to separate melanoma from nevus on their own; aggregating
        posteriors per clinical image is what makes the number mean anything
        at the unit the diagnosis was actually made at.
        """
        # A reconstruction-wrapped model returns (logits, x_recon), which the
        # frozen evaluation path cannot consume. Swap in a logits-only view for
        # the duration; a model without a `base_model` attribute is not wrapped
        # (the same test `_forward_with_recon` uses), so the non-reconstruction
        # path - and every number already reported from it - is untouched.
        _real_model = self.model
        _wrapped = getattr(self.model, "base_model", None) is not None
        if _wrapped:
            self.model = _LogitsOnlyView(_real_model).to(self.device)
        try:
            report = super().evaluate_test_split(test_loader, checkpoint_path=checkpoint_path,
                                                  num_classes=num_classes)
        finally:
            self.model = _real_model
        if report is None or eval_group_aggregation == "none" or test_groups is None:
            return report

        npz_path = self.exp_dir / "test_predictions.npz"
        if not npz_path.is_file():
            self.logger.warning(f"[C-1] {npz_path} missing - skipping image-level aggregation")
            return report
        npz = np.load(npz_path)
        labels, probs = npz["true_label"], npz["probs"]
        test_groups = np.asarray(test_groups)
        if len(test_groups) != len(labels):
            self.logger.warning(
                f"[C-1] test_groups length {len(test_groups)} != n_test {len(labels)} - "
                f"skipping image-level aggregation (a misaligned group vector is worse than none)")
            return report

        image_level = aggregate_group_predictions(probs, labels, test_groups,
                                                    method=eval_group_aggregation)
        report["image_level"] = image_level
        with open(self.exp_dir / "test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(
            f"[C-1] image-level ({eval_group_aggregation}): n_images={image_level['n_groups']} "
            f"balanced_accuracy={image_level['balanced_accuracy']:.4f} "
            f"f1_macro={image_level['f1_macro']:.4f}  (patch-MIL result - NOT a like-for-like "
            f"comparison with the HSI arm)")
        return report

    # ==================================================================
    # fit - R-4 direct indexing, R-6 artifact writers, E-2 step-axis
    # ==================================================================

    def _build_epoch_metrics(self, epoch, train_metrics, val_metrics, epoch_time, current_lr,
                              gpu_mem_mb, parameter_norm, update_norm, total_steps) -> Dict:
        """v15's inline dict assembly (`TrainerG_v11.fit:815-847`), extracted
        so v13 can override this instead of copying `fit` a fourth time -
        `TrainerG_v11` copied `fit` from `TrainerG_v5`, which copied it from
        `TrainerG_v4`, for exactly this reason."""
        train_time_s = getattr(self, "_train_time_s", None)
        n_steps = int(train_metrics.get("n_optimizer_steps", 0))
        s_per_step = (train_time_s / n_steps) if (train_time_s and n_steps) else None

        metrics = {
            "epoch": epoch,
            "optimizer_steps_this_epoch": int(train_metrics.get("n_optimizer_steps", 0)),
            "optimizer_steps_total": total_steps,
            "train_loss": train_metrics["loss"],
            "classification_loss": train_metrics["cls_loss"],
            "train_cls_loss_last_segment": train_metrics.get("cls_loss_last_segment"),
            "halt_loss": train_metrics.get("halt_loss", 0.0),
            # R-4 - direct indexing. A missing key is a bug, not a silent 0.0.
            "MSE_loss": train_metrics["mse_loss"],
            "SAM_loss": train_metrics["sam_loss"],
            "GAN_loss": train_metrics["gan_loss"],
            "train_accuracy": train_metrics["acc"],
            "learning_rate": current_lr,
            "gradient_norm": train_metrics["grad_norm"],
            "GPU_memory_MB": gpu_mem_mb,
            "epoch_time": epoch_time,
            # v17 S1 - where the epoch went. `epoch_time` is unchanged and still means
            # params_before + train + val + params_after; `artifact_time_s` is the block
            # AFTER it and is filled in by `fit` once that block has run, so it is absent
            # from the row only in the file written during its own epoch.
            "train_time_s": train_time_s,
            "val_time_s": getattr(self, "_val_time_s", None),
            "artifact_time_s": None,
            # None, not 0.0, on the non-recursive path: `TrainerG_v9._train_one_epoch` is
            # in a FROZEN file and cannot report this. A zero there would read as "the
            # loader never blocked", which is a measurement, and this is its absence.
            "data_wait_s": train_metrics.get("data_wait_s"),
            "s_per_step": s_per_step,
            "val_loss": val_metrics["loss"],
            "val_cls_loss": val_metrics["cls_loss"],
            "val_mse_loss": val_metrics["mse_loss"],
            "val_sam_loss": val_metrics["sam_loss"],
            "val_gan_loss": val_metrics["gan_loss"],
            **val_metrics["classification"],
            **{f"spectral_{k}": v for k, v in val_metrics["spectral"].items()},
            "gradient_health": train_metrics["gradient_health"],
            "is_valid_epoch": train_metrics["gradient_health"]["is_valid_epoch"],
            "parameter_norm": parameter_norm,
            "update_norm": update_norm,
            "validated_weights": "ema" if self.ema_helper is not None else "live",
        }
        metrics["val_accuracy"] = metrics.pop("accuracy")

        train_last = metrics.get("train_cls_loss_last_segment")
        metrics["generalization_gap_loss_raw"] = metrics["val_loss"] - metrics["train_loss"]
        metrics["generalization_gap_loss"] = (
            metrics["val_cls_loss"] - train_last if train_last is not None
            else metrics["generalization_gap_loss_raw"])
        metrics["generalization_gap_loss_is_comparable"] = train_last is not None
        metrics["generalization_gap_accuracy"] = metrics["train_accuracy"] - metrics["val_accuracy"]
        metrics["generalization_gap_accuracy_is_comparable"] = self.ema_helper is None
        return metrics

    def _log_epoch(self, m, is_best):
        """`TrainerG_v11._log_epoch`, plus the v17 S1 line that says where the epoch went.

        Before this, `epoch_time` was the ONLY timing a run recorded, and it does not even
        cover the whole epoch: `fit` takes its delta before the artifact block. So a 43.4 s
        PAD epoch reported one number and nothing could say how much of it was the
        optimiser and how much was waiting on the DataLoader - which is the first question
        anyone asks when a 0.45 M-parameter model on an idle 16 GB card takes 0.85 s/step
        against an arithmetic floor nearer 0.06 s.

        `unacc` is the closure check: `epoch_time - train - val`, i.e. the two
        `parameters_to_vector` calls. If it is ever more than a few hundredths of a second,
        something is running inside `epoch_time` that this line does not name.
        """
        super()._log_epoch(m, is_best)
        tr, va = m.get("train_time_s"), m.get("val_time_s")
        if tr is None or va is None:
            return
        parts = [f"train {tr:.2f}s"]
        if m.get("s_per_step"):
            parts[0] += f" ({m['s_per_step']:.3f} s/step x {m.get('optimizer_steps_this_epoch', 0)})"
        parts.append(f"val {va:.2f}s")
        if m.get("data_wait_s") is not None:
            parts.append(f"data-wait {m['data_wait_s']:.2f}s")
        if m.get("artifact_time_s") is not None:
            parts.append(f"artifacts {m['artifact_time_s']:.2f}s")
        parts.append(f"unacc {m['epoch_time'] - tr - va:+.2f}s")
        print("  [time] " + " | ".join(parts), flush=True)

        # v17 S4 - the run-level line. "how long is left" and "is this about to early-stop"
        # were both unanswerable from a running log: --early_stop_patience 40 against a
        # 200-epoch budget meant counting epochs by hand to know which limit would bite.
        done = len(self.history)
        max_epochs = getattr(self, "_max_epochs", None)   # absent outside fit (tests)
        if done and max_epochs:
            mean_epoch = sum(r["epoch_time"] for r in self.history) / done
            remaining = max(0, max_epochs - m["epoch"])
            bits = [f"epoch {m['epoch']}/{max_epochs}"]
            if self.best_metric_value is not None and self.best_epoch:
                bits.append(f"best {self.checkpoint_metric} {self.best_metric_value:.4f} "
                            f"@ ep {self.best_epoch}")
            if self.early_stop_patience:
                stale = getattr(self, "_epochs_without_improvement", 0)
                bits.append(f"early-stop {stale}/{self.early_stop_patience}")
                # Whichever limit lands first is the one worth an ETA.
                remaining = min(remaining, self.early_stop_patience - stale)
            bits.append(f"elapsed {format_duration(sum(r['epoch_time'] for r in self.history))}")
            bits.append(f"ETA {format_duration(remaining * mean_epoch)}")
            print("  [progress] " + " | ".join(bits), flush=True)

    def fit(self, max_epochs, start_epoch=1):
        self._max_epochs = max_epochs            # v17 S4 - the denominator of the run ETA
        self.current_epoch = start_epoch - 1
        self.logger.info(f"[TrainerG_v12] epochs {start_epoch}->{max_epochs}, "
                          f"checkpoint_metric={self.checkpoint_metric!r}, "
                          f"scheduler_interval={self.scheduler_interval!r}, "
                          f"val_divergence_patience={self.val_divergence_patience}")
        total_steps = 0
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                if self.is_stopped:
                    break
                self.current_epoch = epoch
                epoch_start_time = time.time()
                if self.device_type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                if self.augmented_train_dataset is not None:
                    self.augmented_train_dataset.set_epoch(epoch)

                with torch.no_grad():
                    params_before = torch.nn.utils.parameters_to_vector(
                        [p.detach().reshape(-1) for p in self.model.parameters()
                         if p.requires_grad]).detach().clone()

                # v17 S1 - the four numbers that say where an epoch went. `epoch_time`
                # below keeps its EXACT pre-v17 meaning (params_before + train + val +
                # params_after, artifacts excluded) so every run already in experiments/
                # stays comparable; these are additions, not a redefinition.
                _t = time.time()
                train_metrics = self._train_one_epoch()
                self._train_time_s = time.time() - _t

                _t = time.time()
                val_metrics = self._validate_one_epoch()
                self._val_time_s = time.time() - _t

                with torch.no_grad():
                    params_after = torch.nn.utils.parameters_to_vector(
                        [p.detach().reshape(-1) for p in self.model.parameters()
                         if p.requires_grad]).detach()
                    parameter_norm = float(params_after.norm())
                    update_norm = float((params_after - params_before).norm())

                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.scheduler_interval == "epoch":
                    self.scheduler.step()
                total_steps += int(train_metrics.get("n_optimizer_steps", 0))

                gpu_mem_mb = (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                              if self.device_type == "cuda" else 0.0)

                metrics = self._build_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time,
                                                     current_lr, gpu_mem_mb, parameter_norm, update_norm,
                                                     total_steps)

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

                # A-4 - computed every epoch (cheap); WRITTEN only on the
                # eval-artifact stride (`_save_eval_artifacts`, below).
                per_patient = None
                if self.val_groups is not None and len(self.val_groups) == len(val_metrics["labels"]):
                    per_patient = self._per_group_recall(val_metrics["preds"], val_metrics["labels"],
                                                          self.val_groups)
                patient_warnings = self._check_patient_outliers(metrics.get(self.checkpoint_metric), per_patient)
                metrics["patient_outlier_warnings"] = patient_warnings
                for w in patient_warnings:
                    self.logger.warning(f"[per-patient] {w}")

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
                        f"Epoch {epoch} marked INVALID (valid_updates=0). Excluded from "
                        f"best-epoch/best_model.pt selection.")

                # v17 S1 - everything below is OUTSIDE `epoch_time` (see above), and until
                # now nothing measured it. Measured on the PAD run from file mtimes it is
                # 0.78 s/epoch, 1.7% of wall clock - small, but it was unknown, and
                # `--eval_artifact_stride` exists to control exactly this.
                _t = time.time()
                self._save_eval_artifacts(epoch, val_metrics)
                # R-6 - actually called again (v5/v11 stopped calling these).
                self._save_per_class_metrics(epoch, val_metrics)
                self._update_loss_components_file(metrics)
                self._save_checkpoint(is_best)
                self._update_history_files()
                # `metrics` is already in `self.history` by reference, so this assignment
                # completes the row in memory. The row written to disk by the
                # `_update_history_files()` call just above therefore lacks this ONE field
                # for the current epoch and gains it on the next write; the unconditional
                # flush in `finally` below closes that gap for the final epoch.
                metrics["artifact_time_s"] = time.time() - _t
                self._log_epoch(metrics, is_best)

                if epoch == max_epochs and not self.is_stopped:
                    self.stop_reason = StopReason.COMPLETED
        except KeyboardInterrupt:
            self.stop_reason = StopReason.USER_INTERRUPTED
            self.is_stopped = True
        finally:
            # v17 S1 - flush the in-memory history one last time so the FINAL epoch's
            # `artifact_time_s` (assigned after that epoch's own write) reaches disk. Also
            # the only thing that persists a history at all when a run is interrupted
            # between the append and the write. Never let it lose the diagnostics below.
            try:
                # `_final_flush` forces the plots past the S2 stride gate: a run that
                # simply reaches `max_epochs` never sets `is_stopped`, so
                # `_should_write_eval_artifacts` would otherwise leave the final figures
                # showing whatever the last stride epoch held.
                self._final_flush = True
                self._update_history_files()
            except Exception as e:                                   # pragma: no cover
                self.logger.warning(f"[history] final flush failed: {type(e).__name__}: {e}")
            finally:
                self._final_flush = False
            self._generate_rich_diagnostics()
            self._generate_final_report()
        return self.stop_reason

    def _generate_rich_diagnostics(self):
        """`TrainerG_v4._generate_rich_diagnostics`, but reconstruction
        visualizations go through `training.plots_v16` (R-6's non-zero gate)
        instead of the frozen `training.plots`."""
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

    def _update_history_files(self):
        """`TrainerG_v4._update_history_files`, routed through
        `training.plots_v16.generate_all_training_plots` (R-6's non-zero
        reconstruction-loss gate) instead of the frozen `training.plots`."""
        import csv as _csv
        fieldnames = list(self.history[-1].keys())
        with open(self.exp_dir / "history.csv", "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow({k: row.get(k) for k in fieldnames})
        with open(self.exp_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=str)

        spectral_rows = [
            {"epoch": r["epoch"], **{k[len("spectral_"):]: v for k, v in r.items()
                                      if k.startswith("spectral_")}}
            for r in self.history
        ]
        if spectral_rows and len(spectral_rows[0]) > 1:
            with open(self.exp_dir / "spectral_metrics.csv", "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=list(spectral_rows[0].keys()))
                writer.writeheader()
                writer.writerows(spectral_rows)

        # v17 S2 - regenerate the plots on the eval-artifact stride, not every epoch.
        # All 10 curves were rebuilt from scratch on every single epoch and thrown away
        # on the next one: 630 figure renders for a 63-epoch run, of which 10 survive.
        # Measured, that block is ~0.56 s of the 0.78 s the whole artifact phase costs.
        # The CSV/JSON rewrites above are milliseconds and stay unconditional, so the
        # numbers a run reports are never stride-delayed - only the pictures are.
        # `fit`'s `finally` calls this method once more, and `_should_write_eval_artifacts`
        # returns True when `self.is_stopped`, so the final plots are always current.
        if not (getattr(self, "_final_flush", False)
                or self._should_write_eval_artifacts(self.current_epoch)):
            return
        try:
            from training.plots_v16 import generate_all_training_plots
            plot_rows = [{"epoch": r["epoch"], "train_loss": r["train_loss"], "val_loss": r["val_loss"],
                          "train_acc": r["train_accuracy"], "val_acc": r["val_accuracy"],
                          "lr": r["learning_rate"], "peak_gpu_memory_mb": r["GPU_memory_MB"],
                          "epoch_time_s": r["epoch_time"],
                          "classification_loss": r.get("classification_loss"),
                          "val_cls_loss": r.get("val_cls_loss"),
                          "MSE_loss": r.get("MSE_loss"), "val_mse_loss": r.get("val_mse_loss"),
                          "SAM_loss": r.get("SAM_loss"), "val_sam_loss": r.get("val_sam_loss"),
                          "gradient_norm": r.get("gradient_norm"),
                          "parameter_norm": r.get("parameter_norm"),
                          "update_norm": r.get("update_norm"),
                          "generalization_gap_loss": r.get("generalization_gap_loss"),
                          "generalization_gap_accuracy": r.get("generalization_gap_accuracy"),
                          } for r in self.history]
            generate_all_training_plots(plot_rows, str(self.dirs["plots"]))
        except Exception as e:
            self.logger.warning(f"plot generation failed: {e}")
