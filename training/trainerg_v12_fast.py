# -*- coding: utf-8 -*-
"""
training/trainerg_v12_fast.py
================================
MedMamba-SS-TRM v17, Stage 12. `TrainerG_v12Fast(TrainerG_v12)` - the deep-supervision training
epoch with the per-step device-to-host synchronisations removed. `training/trainerg_v12.py`
is not on the FROZEN list, but it produced every v16 run in `experiments/`, so it is treated
the same way: subclassed, never edited (the rule `training/trainerg_v13.py:7-10` states).

The problem
-----------
After v17 S5 removed 292 syncs per step, `_train_one_epoch_deep_supervision`
(`trainerg_v12.py:356-540`) still drains the pipeline ~17 times per step:

    :418  check_loss_components(components)          ~4   (frozen, numerical_stability.py:141)
    :433  inspect_gradients_fused(self.model)         3   (isfinite + max + min)
    :436  float(grad_norm)                            1
    :470  check_parameters_finite_fused               1
    :492  seven .item() calls + (argmax == y).sum()   8

Only the first ten decide anything. The last eight are epoch means nobody reads until the
loop is over. Measured context (`plan/v17_baseline.md`): the PAD step is 0.852 s against
0.437 s for forward+backward+`opt.step()` alone - roughly 380 ms per step inside the loop
and outside the optimiser.

The fix
-------
ONE sync per step: a single `torch.stack(...).tolist()` placed after `backward()` and
BEFORE the clip, carrying everything the CPU needs to decide and to record - the loss value,
the pre-clip total gradient norm, this batch's correct-count, and the previous step's
parameter-health norm.

Nothing here is an approximation:

  * `isfinite(total_norm)` is EXACTLY v12's `grad_inspect.all_finite and isfinite(grad_norm)`.
    A NaN or Inf element propagates through the L2 reduction, so the first term can never be
    False while the second is True; the conjunction collapses to the second. So
    `inspect_gradients_fused` is not called at all on the healthy path.
  * `torch.nn.utils.get_total_norm` + `clip_grads_with_norm_` ARE `clip_grad_norm_` - its
    own docstring says "This function is equivalent to", and its body is those two calls.
    Splitting them is what lets the branch happen while the gradients are still unclipped,
    and that MATTERS: `torch.clamp(nan, max=1.0)` is `nan`, so clipping against a non-finite
    norm multiplies EVERY gradient by NaN. A post-clip diagnostic walk would then name
    whatever parameter comes first in iteration order as `first_bad_parameter`, destroying
    the one diagnostic the failure path exists to produce.
  * `check_loss_components` and `inspect_gradients_fused` are still called, verbatim, ON THE
    FAILURE BRANCH, where their syncs buy the diagnostic they were written for. Same policy
    as `training/fused_stability_checks.py`.
  * the six component losses are summed ON THE DEVICE in FLOAT64. `x.item() * bs` into a
    Python float is a float64 accumulation; this reproduces it operation for operation
    (fp32->fp64 is exact, `* bs` is a correctly-rounded fp64 product, same order, same
    values), so the returned dict is bit-identical rather than merely close. fp32
    accumulation over 958 steps would drift into the 5th decimal.

The one deferral: Phase 18's post-`optimizer.step()` parameter check is read one step late,
inside the next step's probe, and drained once after the loop so the final step is never
unchecked. When it fires the run ABORTS - there is no skip-and-continue to preserve, and the
parameters were already destroyed at the instant of the step - so the only cost is that
`bad_params` is read one step later and may name more parameters than were bad at the time.

Loss and gradient finiteness are NOT deferred, on purpose. The existing policy is not "stop
on a NaN", it is SKIP THE BATCH AND KEEP TRAINING (`NumericalStabilityController.record_batch`
returns True for an isolated bad batch; up to `max_gradient_skip_ratio` of an epoch may be
skipped and the epoch is still valid). That policy only exists because the decision is made
before `optimizer.step()`. Deferring it means stepping AdamW on a NaN gradient, after which
every parameter is NaN and there is nothing left to skip - turning a recoverable event into
a dead run. Masking the gradient by an on-device flag is not equivalent either: decoupled
weight decay and the moment updates still move the parameters.

Under fp16 this delegates to `TrainerG_v12`: an enabled `GradScaler` syncs on `found_inf`
inside `scaler.step` anyway, so there is nothing to win, and deferring the loss check past
`unscale_` would break the scaler's unscale/update pairing on a skipped batch. Every run in
`experiments/` is `--amp bf16`, where v17 S5 disabled the scaler.

`torch.nn.utils.get_total_norm` / `clip_grads_with_norm_` arrived in torch 2.7 and
`requirements.txt` pins only `torch>=2.1`, so they are probed at import; without them the
epoch delegates to `TrainerG_v12` and says so once. An optimisation must never be what kills
a run.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.amp import autocast

from training.trainerg_v12 import TrainerG_v12, _timed_iter
from training.numerical_stability import (          # frozen - failure branch only
    GradientInspection, check_loss_components, format_loss_component_status,
)
from training.fused_stability_checks import inspect_gradients_fused, check_parameters_finite_fused
from training.gan import discriminator_loss, generator_adversarial_loss
from training.progress import ProgressReporter
from training.ema import upgrade_ema_helper

_FAST_OK = all(hasattr(torch.nn.utils, n) for n in ("get_total_norm", "clip_grads_with_norm_"))
_WARNED = False


def _degrade(detail: str) -> None:
    """Print once, then never again. `fused_stability_checks.py:75-82`'s message shape."""
    global _WARNED
    if not _WARNED:
        _WARNED = True
        print(f"[v17 S12] de-synced training loop unavailable ({detail}); falling back to "
              f"TrainerG_v12's loop for the rest of this run. Correct, just slower.", flush=True)


class TrainerG_v12Fast(TrainerG_v12):
    """`TrainerG_v12` with one device sync per training step instead of ~17.

    Overrides exactly one method plus `__init__`, so it composes by being placed FIRST -
    see `make_fast`, which is how every entry point reaches it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The frozen TrainerG_v10 built an EMAHelper by name with no injection hook, so the
        # object is replaced here - as trainerg_v12.py:245-266 replaces the GradScaler that
        # frozen trainerg_v4.py:133 built, and for the same reason.
        self.ema_helper = upgrade_ema_helper(self.ema_helper, self.model,
                                              getattr(self, "logger", None))

    # ==================================================================
    # Phase 18, one step later
    # ==================================================================

    def _abort_on_nonfinite_parameters(self) -> bool:
        """Phase 18's abort, deferred from `trainerg_v12.py:470-481`. True = stop the run.

        The naming walk gets the last word rather than the fused norm: a global L2 norm can
        overflow to Inf over finite parameters, and v12 could not tell that apart from a
        genuine NaN. Confirming is free on a branch that is already aborting - so this is
        strictly safer than the check it replaces, not merely cheaper.
        """
        params_ok, bad_params = check_parameters_finite_fused(self.model)
        if params_ok:
            return False
        self.logger.error(
            "[numerical-stability] non-finite parameters detected at the step AFTER the one "
            "that produced them (v17 S12 defers this read by one step); the list below may "
            "name parameters that went bad in between.")
        self.stability.record_external_abort(          # v12's message, byte for byte
            f"NUMERICAL_INSTABILITY: model parameters non-finite immediately after "
            f"optimizer.step() (Phase 18): {bad_params[:5]}"
            + (f" (+{len(bad_params) - 5} more)" if len(bad_params) > 5 else ""))
        return True

    # ==================================================================
    # the epoch
    # ==================================================================

    def _train_one_epoch_deep_supervision(self, base):
        if not _FAST_OK:
            _degrade("torch.nn.utils.get_total_norm/clip_grads_with_norm_ missing (needs torch>=2.7)")
            return super()._train_one_epoch_deep_supervision(base)
        if self.scaler.is_enabled():
            _degrade("--amp fp16: GradScaler already syncs on found_inf every step")
            return super()._train_one_epoch_deep_supervision(base)

        self.stability.start_epoch()
        self.grad_health.start_epoch()
        self.model.train()
        total_loss = total_grad_norm = 0.0
        total_correct = total_n = n_batches = 0
        aborted_mid_epoch = False
        # cls, cls_last, halt, mse, sam, gan - summed on the DEVICE in float64, read once
        # at epoch end. See the module docstring for why float64 and not float32.
        totals = torch.zeros(6, dtype=torch.float64, device=self.device)
        pending_param_norm = None                    # Phase 18, read in the NEXT step's probe

        amp_enabled, amp_dtype = self._resolve_amp()
        halting = bool(getattr(base.cfg, "trm_act_halting", False))

        data_wait = {"s": 0.0}
        n_steps_total = self._loader_len(self.train_loader)
        progress = ProgressReporter(f"[train] epoch {self.current_epoch}", min_interval=2.0)
        for batch in _timed_iter(self.train_loader, data_wait):
            # ---- verbatim TrainerG_v12:376-411 ----------------------------------------
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
            # ---- end verbatim ---------------------------------------------------------

            # v12 checked loss finiteness HERE, before backward. Doing it after costs one
            # wasted backward on a batch that is about to be skipped (rare by construction)
            # and buys the gradient norm in the same transfer - one sync instead of two.
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            # `clip_grad_norm_`'s own first half. With no grads it returns a CPU tensor,
            # which would make the stack below straddle two devices.
            total_norm = (torch.nn.utils.get_total_norm(grads) if grads
                          else torch.zeros((), device=self.device))
            correct = (logits.detach().argmax(dim=-1) == y).sum()

            probe = [loss.detach().double(), total_norm.double(), correct.double()]
            if pending_param_norm is not None:
                probe.append(pending_param_norm.double())
            probe = torch.stack(probe).tolist()                  # <<< THE sync of the step
            loss_val, norm_val, correct_val = probe[0], probe[1], int(probe[2])

            # Phase 18 first: if the PREVIOUS step destroyed the parameters then this step's
            # loss is NaN too, and the loss branch below would report the wrong cause.
            if pending_param_norm is not None:
                pending_param_norm = None
                if not math.isfinite(probe[3]) and self._abort_on_nonfinite_parameters():
                    aborted_mid_epoch = True
                    break

            if not math.isfinite(loss_val):
                if self.stability_cfg.debug_numerics:
                    components = {
                        "classification": cls_loss, "halt": halt_loss if halting else None,
                        "MSE": mse_loss if x_recon is not None else None,
                        "SAM": sam_loss if x_recon is not None else None,
                        "GAN": gan_loss if (self.use_gan and x_recon is not None) else None,
                        "total": loss,
                    }
                    self.logger.warning(f"[epoch {self.current_epoch}] "
                                        f"{format_loss_component_status(check_loss_components(components))}")
                self.optimizer.zero_grad(set_to_none=True)
                self.grad_health.record_batch(loss_is_finite=False, grad_is_finite=False,
                                               optimizer_stepped=False)
                if not self.stability.record_batch(loss_finite=False, grad_inspection=None,
                                                    grad_norm=None, optimizer_stepped=False):
                    aborted_mid_epoch = True
                    break
                continue

            if not math.isfinite(norm_val):
                # Gradients are still UNCLIPPED here, so this is the same GradientInspection
                # v12 built at :433 - `first_bad_parameter` and the element counts survive.
                grad_inspect = inspect_gradients_fused(self.model)
                # ---- verbatim TrainerG_v12:439-457, `grad_norm` -> `norm_val` -----------
                if self.stability_cfg.debug_numerics:
                    self.logger.warning(
                        f"[epoch {self.current_epoch}] NONFINITE_GRADIENT "
                        f"first_bad_parameter={grad_inspect.first_bad_parameter!r} "
                        f"nan_elements={grad_inspect.nan_count} inf_elements={grad_inspect.inf_count}")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.grad_health.record_batch(
                    loss_is_finite=True, grad_is_finite=False, optimizer_stepped=False,
                    grad_norm=None,
                    first_bad_parameter=grad_inspect.first_bad_parameter,
                    gradient_nan_elements=grad_inspect.nan_count,
                    gradient_inf_elements=grad_inspect.inf_count)
                if not self.stability.record_batch(
                        loss_finite=True, grad_inspection=grad_inspect,
                        grad_norm=None, optimizer_stepped=False):
                    aborted_mid_epoch = True
                    break
                continue

            # `clip_grad_norm_`'s own second half, against the norm already computed.
            torch.nn.utils.clip_grads_with_norm_(
                self.model.parameters(),
                self.grad_clip_norm if self.grad_clip_norm else float("inf"), total_norm)

            total_grad_norm += norm_val
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # E-2 - step-axis LR schedule: advance on every successful optimizer step, not
            # once per (possibly subsampled) "epoch".
            if self.scheduler_interval == "step":
                self.scheduler.step()

            if self.ema_helper is not None:
                self.ema_helper.update(self.model)

            # Phase 18, read in the next step's probe. One pass over the parameters, which
            # is what check_parameters_finite_fused costs today - only the READ is deferred.
            pending_param_norm = torch.nn.utils.get_total_norm(
                [p.detach() for p in self.model.parameters()])

            self.grad_health.record_batch(loss_is_finite=True, grad_is_finite=True,
                                           optimizer_stepped=True, grad_norm=norm_val)
            # The controller reads only `.all_finite` and `.first_bad_parameter`; v12 also
            # filled max/min_abs_gradient here, which nothing in this repo consumes.
            self.stability.record_batch(loss_finite=True,
                                         grad_inspection=GradientInspection(all_finite=True),
                                         grad_norm=norm_val, optimizer_stepped=True)

            # ---- verbatim TrainerG_v12:483-489 (the GAN block keeps its own 1-2 syncs) --
            if self.use_gan and x_recon is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=amp_enabled):
                    d_loss = discriminator_loss(self.discriminator, x, x_recon.detach())
                if not (torch.isnan(d_loss) or torch.isinf(d_loss)):
                    d_loss.backward()
                    self.disc_optimizer.step()

            bs = x.size(0)
            # `.double() * bs` then `add_`, NOT `add_(..., alpha=bs)`: the alpha form may
            # contract to an FMA and round differently from the Python `float * int`.
            totals.add_(torch.stack((cls_loss.detach(), cls_loss_last.detach(),
                                      halt_loss.detach(), mse_loss.detach(),
                                      sam_loss.detach(), gan_loss.detach())).double() * bs)
            total_loss += loss_val * bs
            total_correct += correct_val
            total_n += bs
            n_batches += 1
            progress.emit(n_batches, n_steps_total,
                          f"loss {total_loss / max(1, total_n):.4f} "
                          f"acc {total_correct / max(1, total_n):.4f}")

        progress.close()
        # Drain the last step's Phase-18 flag - one sync per EPOCH, not per step. Without
        # this the final optimizer step of every epoch would go unchecked.
        if not aborted_mid_epoch and pending_param_norm is not None \
                and not math.isfinite(float(pending_param_norm)) \
                and self._abort_on_nonfinite_parameters():
            aborted_mid_epoch = True

        total_cls, total_cls_last, total_halt, total_mse, total_sam, total_gan = totals.tolist()

        # ---- verbatim TrainerG_v12:508-540 --------------------------------------------
        epoch_health = self.grad_health.finish_epoch()
        stability_summary = self.stability.finish_epoch()
        epoch_health = {**epoch_health, "stability_status": stability_summary.status,
                         "stability_skip_ratio": stability_summary.skip_ratio}
        if stability_summary.status in ("critical", "invalid", "fatal"):
            epoch_health["is_valid_epoch"] = False

        abort_reason = self.stability.should_abort_run()
        if abort_reason or aborted_mid_epoch:
            reason = abort_reason or "NUMERICAL_INSTABILITY: aborted mid-epoch (see epoch_health)."
            from training.trainerg_v7 import _StopReasonStr
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
            "data_wait_s": data_wait["s"],
        }


def make_fast(trainer_cls):
    """`trainer_cls` with `TrainerG_v12Fast`'s epoch in front of it.

    One factory rather than a named `...Fast` class per entry point, because there are five
    of them (`v16`, `_recon`, `_optimal`, `_original`, `_optimal_recon`) and the bare
    `train_example_v16.py` has no `build_overrides` at all to hang a class off. Five
    declarations would be five places to keep in sync - the drift this repo's composition
    docstrings exist to complain about. `train_example_v16._main` calls this once.

    Placing `TrainerG_v12Fast` FIRST is what makes it safe: it defines exactly
    `_train_one_epoch_deep_supervision`, `_abort_on_nonfinite_parameters` and `__init__`, so
    it cannot shadow anything the composed base contributes. For the deepest entry point:

        TrainerG_v13OptimalFast
          -> TrainerG_v12Fast        _train_one_epoch_deep_supervision, FastEMAHelper swap
          -> TrainerG_v13Optimal     (empty)
          -> TrainerG_v12Optimal     _train_one_epoch (3-tuple batches), _trigger_stop
          -> TrainerG_v12Original    _check_stopping_rules, gate-G5 metric alias
          -> TrainerG_v13            __init__ (--recon_* flags), export_reconstruction_samples
          -> TrainerG_v12            everything else

    C3 linearises that because `TrainerG_v12Fast`'s only base, `TrainerG_v12`, appears in
    the other branch's tail. `TrainerG_v12Optimal._train_one_epoch` still wins (Fast does not
    define it) and still dispatches into the fast epoch for the recursive path; Fast's
    `super()._train_one_epoch_deep_supervision` resolves down to `TrainerG_v12`'s, so the
    fp16 / missing-capability delegation still works.
    """
    if issubclass(trainer_cls, TrainerG_v12Fast):
        return trainer_cls
    if trainer_cls is TrainerG_v12:
        return TrainerG_v12Fast
    return type(f"{trainer_cls.__name__}Fast", (TrainerG_v12Fast, trainer_cls),
                {"__doc__": f"v17 S12: {trainer_cls.__name__} with the de-synced training "
                            f"epoch. Generated by training.trainerg_v12_fast.make_fast."})
