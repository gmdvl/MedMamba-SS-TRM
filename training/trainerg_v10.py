# -*- coding: utf-8 -*-
"""
training/trainerg_v10.py
==========================
Adds TRM-style (Tiny Recursive Model, arXiv:2510.04871) in-loop deep
supervision + EMA on top of `training/trainerg_v9.py` (`TrainerG_v9`), for
`medmamba_ss_trm.MedMambaSSTRM` (`--architecture recursive`). Every other
concern - AMP, per-component/per-parameter numerical-stability checks,
gradient-health tracking, class-collapse monitoring, Cohen's Kappa/MCC,
checkpointing, early stopping - is inherited UNCHANGED from
TrainerG_v9/v8/v7/v6/v5/v4.

Non-recursive models (`split`/`fullchannel`/`efficient`) are completely
unaffected: `_train_one_epoch` detects whether the model exposes
`forward_deep_supervision` and, if not, delegates straight to
`TrainerG_v9._train_one_epoch`.

Deep supervision (`MedMambaSSTRM.forward_deep_supervision`, see
`medmamba_ss_trm.py`) runs *in-loop*: all `cfg.trm_deep_supervision_steps` segments
inside one batch, one `loss.backward()` - not TRM's own `pretrain.py`, which
streams one optimizer step per segment with the `(y, z)` carry persisting
across training steps. In-loop keeps every existing per-batch stability/
gradient-health mechanism in this repo directly applicable.

Reconstruction (`--recon_mode latent`, `MedMambaSSTRMLatentReconWrapper`) is
supported: the decoder reads `base_model.last_feature_map`, the *detached*
final-segment feature the model caches for exactly this purpose (detached
so `EMAHelper.ema_copy`'s `copy.deepcopy` never has to deep-copy a live
autograd graph - see the comment on `MedMambaSSTRM.forward_deep_supervision`
in `medmamba_ss_trm.py`). Practical effect: the reconstruction loss still trains
the decoder, but no longer backpropagates into the recursive core itself -
the classification loss at every deep-supervision segment already supplies
that signal, TRM detaches between segments for the same reason, and this
keeps recon safe to combine with EMA. `--recon_mode raw_input`
(`MedMambaSSTRMRawReconWrapper`, which calls `base_model(x)` directly rather than
threading through the backbone) does not expose the per-segment logits list
through its `forward`, so that combination falls back to the ordinary
single-forward `TrainerG_v9` path (no deep supervision, still exercises EMA).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.amp import autocast

from training.trainerg_v9 import TrainerG_v9
from training.trainerg_v7 import _StopReasonStr
from training.gan import discriminator_loss, generator_adversarial_loss
from training.numerical_stability import (
    inspect_gradients, check_parameters_finite, check_loss_components, format_loss_component_status,
)
from medmamba_ss_trm_ema import EMAHelper


class TrainerG_v10(TrainerG_v9):
    def __init__(self, *args, ema_rate: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        base = self._deep_supervision_base()
        # Default EMA rate comes from the model's own MedMambaSSTRMConfig
        # (`cfg.trm_ema_rate`) unless the caller overrides it explicitly.
        if ema_rate is None:
            ema_rate = getattr(getattr(base, "cfg", None), "trm_ema_rate", 0.0) if base is not None else 0.0
        self.ema_helper = None
        if base is not None and ema_rate and ema_rate > 0:
            self.ema_helper = EMAHelper(mu=ema_rate)
            self.ema_helper.register(self.model)

    # ------------------------------------------------------------------
    def _deep_supervision_base(self):
        """Returns the `MedMambaSSTRM` instance if `self.model` (possibly
        wrapped by a reconstruction wrapper) exposes `forward_deep_supervision`
        AND that wrapper can supply the reconstruction target from it
        (`MedMambaSSTRMLatentReconWrapper` or no wrapper); `None` otherwise, in
        which case training falls back to `TrainerG_v9._train_one_epoch`
        unchanged."""
        m = self.model
        if hasattr(m, "forward_deep_supervision"):
            return m
        base = getattr(m, "base_model", None)
        if base is not None and hasattr(base, "forward_deep_supervision") and hasattr(m, "decoder"):
            return base
        return None

    # ------------------------------------------------------------------
    def _train_one_epoch(self):
        base = self._deep_supervision_base()
        if base is None:
            return super()._train_one_epoch()
        return self._train_one_epoch_deep_supervision(base)

    def _train_one_epoch_deep_supervision(self, base):
        self.stability.start_epoch()
        self.grad_health.start_epoch()
        self.model.train()
        total_loss = total_cls = total_halt = total_mse = total_sam = total_gan = 0.0
        total_correct = total_n = 0
        total_grad_norm = 0.0
        n_batches = 0
        aborted_mid_epoch = False

        amp_enabled, amp_dtype = self._resolve_amp()
        decoder = getattr(self.model, "decoder", None)   # set only when wrapped by MedMambaSSTRMLatentReconWrapper
        halting = bool(getattr(base.cfg, "trm_act_halting", False))

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            if x.ndim == 4 and self.device_type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                logits_list, q_list = base.forward_deep_supervision(x)
                logits = logits_list[-1]
                cls_loss = sum(self.criterion(l, y) for l in logits_list) / len(logits_list)

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

            # Same per-component finiteness gate as TrainerG_v9, extended with "halt".
            components = {
                "classification": cls_loss, "halt": halt_loss if halting else None,
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

            grad_inspect = inspect_gradients(self.model)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm if self.grad_clip_norm else float("inf"))
            grad_norm = float(grad_norm)
            grad_ok = grad_inspect.all_finite and torch.isfinite(torch.tensor(grad_norm))

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
                    grad_norm=grad_norm if torch.isfinite(torch.tensor(grad_norm)) else None,
                    first_bad_parameter=grad_inspect.first_bad_parameter,
                    gradient_nan_elements=grad_inspect.nan_count, gradient_inf_elements=grad_inspect.inf_count)
                keep_going = self.stability.record_batch(
                    loss_finite=True, grad_inspection=grad_inspect,
                    grad_norm=grad_norm if torch.isfinite(torch.tensor(grad_norm)) else None, optimizer_stepped=False)
                if not keep_going:
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

            total_loss += loss.item() * x.size(0)
            total_cls += cls_loss.item() * x.size(0)
            total_halt += halt_loss.item() * x.size(0)
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
            "loss": total_loss / denom, "cls_loss": total_cls / denom, "halt_loss": total_halt / denom,
            "mse_loss": total_mse / denom, "sam_loss": total_sam / denom,
            "gan_loss": total_gan / denom, "acc": total_correct / denom,
            "grad_norm": total_grad_norm / max(1, n_batches),
            "gradient_health": epoch_health,
        }

    # ------------------------------------------------------------------
    def _validate_one_epoch(self):
        """Inherited validation/metrics logic (TrainerG_v6-v9), run against the
        EMA-averaged weights when EMA is enabled - a temporary swap of
        `self.model`, restored in `finally` so checkpointing below still saves
        the live (non-EMA) model plus the EMA shadow (see `_save_checkpoint`)."""
        if self.ema_helper is None:
            return super()._validate_one_epoch()
        live_model = self.model
        self.model = self.ema_helper.ema_copy(live_model)
        try:
            return super()._validate_one_epoch()
        finally:
            self.model = live_model

    # ------------------------------------------------------------------
    def _save_checkpoint(self, is_best):
        super()._save_checkpoint(is_best)
        if self.ema_helper is None:
            return
        ema_state = {"ema_shadow": self.ema_helper.state_dict(), "epoch": self.current_epoch}
        torch.save(ema_state, self.dirs["checkpoints"] / "latest_ema.pt")
        if is_best:
            torch.save(ema_state, self.exp_dir / "best_model_ema.pt")

    def load_checkpoint(self, path):
        super().load_checkpoint(path)
        if self.ema_helper is None:
            return
        ema_path = self.dirs["checkpoints"] / "latest_ema.pt"
        if ema_path.exists():
            ema_state = torch.load(ema_path, map_location=self.device)
            self.ema_helper.load_state_dict(ema_state["ema_shadow"])
