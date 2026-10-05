# -*- coding: utf-8 -*-
"""
training/numerical_stability.py
=================================
MedMamba-SS-TRM v12/v13 stability plan, Phases 11-23 and 36-37 - the central
numerical-health mechanism.

`training/gradient_health.py` (Stage A, Phase 16/17) already tracks
per-epoch counters and marks a zero-valid-update epoch as INVALID. This
module extends that in the direction the plan explicitly calls for:

  * skip-ratio CLASSIFICATION, not just a valid/invalid boolean (Phase 22):
    healthy / warning / degraded / critical / invalid / fatal.
  * PARAMETER-LEVEL gradient inspection (Phase 12) - which parameter's
    gradient went non-finite first, not just "grad_norm is NaN".
  * PER-LOSS-COMPONENT finiteness checks (Phase 14) - classification / MSE
    / SAM / GAN / total, checked independently.
  * a persistent-failure -> ABORT policy (Phase 21/23): an isolated bad
    batch is skipped and training continues; N consecutive bad batches, or
    an epoch's skip ratio crossing a hard threshold, or N consecutive
    unhealthy epochs, terminates the RUN with a clear reason instead of
    limping on indefinitely.
  * failure-artifact capture (Phase 37) so an aborted run leaves behind
    exactly what was happening when it broke, not just a warning line in
    a scrollback buffer.

This module is intentionally standalone (no dependency on any specific
`TrainerG_vN`) so it can be composed into a trainer via delegation rather
than requiring another subclass rewrite of the whole training loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ----------------------------------------------------------------------
# Phase 22 - skip-ratio classification
# ----------------------------------------------------------------------

SKIP_RATIO_THRESHOLDS = (
    # (upper_bound_exclusive, status) - evaluated in order, first match wins.
    # upper_bound_exclusive == None means "no upper bound" (last bucket).
    (0.0, "healthy"),     # exactly 0%
    (0.01, "warning"),    # (0%, 1%)
    (0.05, "degraded"),   # [1%, 5%)
    (0.10, "critical"),   # [5%, 10%)
    (1.0, "invalid"),     # [10%, 100%)
    (None, "fatal"),      # 100%
)


def classify_skip_ratio(skip_ratio: float) -> str:
    if skip_ratio <= 0.0:
        return "healthy"
    if skip_ratio >= 1.0:
        return "fatal"
    if skip_ratio < 0.01:
        return "warning"
    if skip_ratio < 0.05:
        return "degraded"
    if skip_ratio < 0.10:
        return "critical"
    return "invalid"


# ----------------------------------------------------------------------
# Phase 12 - parameter-level gradient inspection
# ----------------------------------------------------------------------

@dataclass
class GradientInspection:
    all_finite: bool
    first_bad_parameter: Optional[str] = None
    n_bad_parameters: int = 0
    nan_count: int = 0
    inf_count: int = 0
    max_abs_gradient: Optional[float] = None
    min_abs_gradient: Optional[float] = None


def inspect_gradients(model: torch.nn.Module) -> GradientInspection:
    """Per-parameter `torch.isfinite` check (Phase 12), rather than only
    trusting the aggregate `grad_norm`. Cheap relative to a training step
    (one boolean reduction per parameter tensor)."""
    first_bad = None
    n_bad = 0
    nan_count = 0
    inf_count = 0
    max_abs = None
    min_abs = None

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite_mask = torch.isfinite(g)
        if not bool(finite_mask.all()):
            n_bad += 1
            if first_bad is None:
                first_bad = name
            nan_count += int(torch.isnan(g).sum().item())
            inf_count += int(torch.isinf(g).sum().item())
            continue
        g_abs = g.abs()
        if g_abs.numel() == 0:
            continue
        g_max = float(g_abs.max().item())
        g_min = float(g_abs.min().item())
        max_abs = g_max if max_abs is None else max(max_abs, g_max)
        min_abs = g_min if min_abs is None else min(min_abs, g_min)

    return GradientInspection(
        all_finite=(n_bad == 0), first_bad_parameter=first_bad, n_bad_parameters=n_bad,
        nan_count=nan_count, inf_count=inf_count, max_abs_gradient=max_abs, min_abs_gradient=min_abs,
    )


def check_parameters_finite(model: torch.nn.Module) -> Tuple[bool, List[str]]:
    """Phase 18 - post-optimizer-step parameter health check. Returns
    (all_finite, bad_parameter_names)."""
    bad = []
    for name, p in model.named_parameters():
        if p is None:
            continue
        if not bool(torch.isfinite(p.detach()).all()):
            bad.append(name)
    return (len(bad) == 0), bad


# ----------------------------------------------------------------------
# Phase 14 - per-loss-component finiteness
# ----------------------------------------------------------------------

def check_loss_components(components: Dict[str, Optional[torch.Tensor]]) -> Dict[str, str]:
    """`components`: e.g. {"classification": cls_loss, "MSE": mse_loss,
    "SAM": sam_loss, "GAN": gan_loss, "total": loss}. A value of None means
    "not used this run" (e.g. GAN disabled) and is reported as such rather
    than finite/non-finite."""
    status = {}
    for name, value in components.items():
        if value is None:
            status[name] = "not_used"
        elif bool(torch.isfinite(value)):
            status[name] = "finite"
        else:
            status[name] = "NONFINITE"
    return status


def format_loss_component_status(status: Dict[str, str]) -> str:
    parts = [f"{k}={v}" for k, v in status.items()]
    return "NONFINITE_LOSS: " + ", ".join(parts)


# ----------------------------------------------------------------------
# Epoch-level accumulator + persistent-failure policy
# ----------------------------------------------------------------------

@dataclass
class StabilityConfig:
    max_gradient_norm: float = 1.0
    max_gradient_skip_ratio: float = 0.10
    max_consecutive_bad_batches: int = 3
    max_consecutive_unhealthy_epochs: int = 2
    # "unhealthy" epoch = skip-ratio status in this set
    unhealthy_statuses: Tuple[str, ...] = ("critical", "invalid", "fatal")
    debug_numerics: bool = False

    def as_dict(self):
        return asdict(self)


@dataclass
class EpochStabilitySummary:
    total_batches: int = 0
    valid_updates: int = 0
    skipped_updates: int = 0
    nonfinite_loss: int = 0
    nonfinite_gradients: int = 0
    skip_ratio: float = 0.0
    status: str = "healthy"
    first_bad_parameter: Optional[str] = None
    max_gradient_norm_seen: Optional[float] = None
    is_valid_epoch: bool = True

    def as_dict(self):
        return asdict(self)


class NumericalStabilityController:
    """Composed into a trainer (not a base class) - the trainer calls
    `start_epoch()`/`record_batch()`/`finish_epoch()` around its existing
    loop, and consults `should_abort_run()` after each epoch."""

    def __init__(self, cfg: Optional[StabilityConfig] = None, exp_dir: Optional[str] = None):
        self.cfg = cfg or StabilityConfig()
        self.exp_dir = exp_dir
        self.epoch_history: List[EpochStabilitySummary] = []
        self._consecutive_bad_batches = 0
        self._consecutive_unhealthy_epochs = 0
        self._abort_reason: Optional[str] = None
        self._cur = EpochStabilitySummary()
        self._cur_first_bad_param: Optional[str] = None
        self._cur_max_grad_norm: Optional[float] = None

    # ------------------------------------------------------------------
    def start_epoch(self) -> None:
        self._cur = EpochStabilitySummary()
        self._cur_first_bad_param = None
        self._cur_max_grad_norm = None

    def record_batch(self, loss_finite: bool, grad_inspection: Optional[GradientInspection],
                      grad_norm: Optional[float], optimizer_stepped: bool) -> bool:
        """Returns True if this was an ISOLATED bad batch that should just
        be skipped, False if persistent-failure thresholds were crossed and
        the caller should abort immediately (checked mid-epoch, not only at
        epoch end - Phase 21's "persistent bad batches -> abort" applies
        within an epoch too)."""
        self._cur.total_batches += 1
        if not loss_finite:
            self._cur.nonfinite_loss += 1
        if grad_inspection is not None and not grad_inspection.all_finite:
            self._cur.nonfinite_gradients += 1
            if self._cur_first_bad_param is None:
                self._cur_first_bad_param = grad_inspection.first_bad_parameter
        if grad_norm is not None and np.isfinite(grad_norm):
            self._cur_max_grad_norm = grad_norm if self._cur_max_grad_norm is None \
                else max(self._cur_max_grad_norm, grad_norm)

        if optimizer_stepped:
            self._cur.valid_updates += 1
            self._consecutive_bad_batches = 0
            return True
        else:
            self._cur.skipped_updates += 1
            self._consecutive_bad_batches += 1
            if self._consecutive_bad_batches >= self.cfg.max_consecutive_bad_batches:
                self._abort_reason = (
                    f"NUMERICAL_INSTABILITY: {self._consecutive_bad_batches} consecutive bad "
                    f"batches (>= max_consecutive_bad_batches={self.cfg.max_consecutive_bad_batches})."
                )
                return False
            return True

    def finish_epoch(self) -> EpochStabilitySummary:
        s = self._cur
        s.skip_ratio = s.skipped_updates / s.total_batches if s.total_batches else 0.0
        s.status = classify_skip_ratio(s.skip_ratio)
        s.first_bad_parameter = self._cur_first_bad_param
        s.max_gradient_norm_seen = self._cur_max_grad_norm
        s.is_valid_epoch = s.valid_updates > 0 and s.status not in ("invalid", "fatal") \
            and s.skip_ratio <= self.cfg.max_gradient_skip_ratio

        if s.status in self.cfg.unhealthy_statuses:
            self._consecutive_unhealthy_epochs += 1
        else:
            self._consecutive_unhealthy_epochs = 0

        if s.skip_ratio > self.cfg.max_gradient_skip_ratio and self._abort_reason is None:
            self._abort_reason = (
                f"NUMERICAL_INSTABILITY: epoch skip_ratio={s.skip_ratio:.4f} exceeds "
                f"max_gradient_skip_ratio={self.cfg.max_gradient_skip_ratio}."
            )
        if self._consecutive_unhealthy_epochs >= self.cfg.max_consecutive_unhealthy_epochs \
                and self._abort_reason is None:
            self._abort_reason = (
                f"NUMERICAL_INSTABILITY: {self._consecutive_unhealthy_epochs} consecutive unhealthy "
                f"epochs (status in {self.cfg.unhealthy_statuses})."
            )

        self.epoch_history.append(s)
        return s

    # ------------------------------------------------------------------
    def should_abort_run(self) -> Optional[str]:
        """Returns the abort reason string if the run should terminate now,
        else None. Phase 23: parameter/forward-output non-finiteness is
        reported by the caller via `record_external_abort` below."""
        return self._abort_reason

    def record_external_abort(self, reason: str) -> None:
        """For conditions this controller doesn't itself observe (e.g. a
        post-optimizer-step parameter NaN check, or a non-finite forward
        activation caught by a debug hook) - Phase 18/23."""
        if self._abort_reason is None:
            self._abort_reason = reason

    # ------------------------------------------------------------------
    def write_failure_artifacts(self, out_dir: str, extra: Optional[Dict] = None) -> str:
        """Phase 37 - writes `numerical_failure/failure.json` (and, if
        provided via `extra`, the sibling files the plan lists: batch_info,
        loss_components, gradient_summary, parameter_summary,
        model_config, dataloader_config, system_memory, gpu_memory).
        Returns the directory written to."""
        failure_dir = os.path.join(out_dir, "numerical_failure")
        os.makedirs(failure_dir, exist_ok=True)

        payload = {
            "abort_reason": self._abort_reason,
            "stability_config": self.cfg.as_dict(),
            "epoch_history": [e.as_dict() for e in self.epoch_history],
            "consecutive_bad_batches_at_failure": self._consecutive_bad_batches,
            "consecutive_unhealthy_epochs_at_failure": self._consecutive_unhealthy_epochs,
        }
        with open(os.path.join(failure_dir, "failure.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

        if extra:
            for name, content in extra.items():
                try:
                    with open(os.path.join(failure_dir, f"{name}.json"), "w") as f:
                        json.dump(content, f, indent=2, default=str)
                except (TypeError, OSError):
                    pass

        return failure_dir


# ----------------------------------------------------------------------
# Phase 17 - one-batch numerical smoke test
# ----------------------------------------------------------------------

def run_smoke_test(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                    forward_and_loss_fn, optimizer: torch.optim.Optimizer,
                    max_gradient_norm: float = 1.0) -> Dict:
    """`forward_and_loss_fn(model, x, y) -> (total_loss, loss_components_dict)`.

    Runs exactly one forward/backward/optimizer-step cycle and checks every
    stage the plan's Phase 17 lists. Returns a report dict with `"passed"`
    and, on failure, the reason. Does not mutate the caller's dataset/
    dataloader state - `x`/`y` should be a single already-collated batch.
    """
    report: Dict = {"passed": False, "stage": None, "detail": None}

    if not torch.isfinite(x).all():
        report.update(stage="input", detail="input batch `x` contains NaN/Inf")
        return report

    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        total_loss, components = forward_and_loss_fn(model, x, y)
    except Exception as e:
        report.update(stage="forward", detail=f"forward pass raised: {e}")
        return report

    comp_status = check_loss_components(components)
    if any(v == "NONFINITE" for v in comp_status.values()):
        report.update(stage="loss", detail=comp_status)
        return report

    try:
        total_loss.backward()
    except Exception as e:
        report.update(stage="backward", detail=f"backward pass raised: {e}")
        return report

    grad_inspect = inspect_gradients(model)
    if not grad_inspect.all_finite:
        report.update(stage="gradient", detail=grad_inspect.__dict__)
        return report

    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm))
    if not np.isfinite(grad_norm):
        report.update(stage="grad_norm", detail=f"grad_norm={grad_norm}")
        return report

    optimizer.step()

    params_ok, bad_params = check_parameters_finite(model)
    if not params_ok:
        report.update(stage="parameters", detail=f"non-finite parameters after step: {bad_params[:5]}")
        return report

    report.update(passed=True, stage=None, detail=None, loss_components=comp_status,
                   grad_norm=grad_norm)
    return report


# ----------------------------------------------------------------------
# v15 R0.1/R0.2 - representation-sensitivity (input-dependence) check
# ----------------------------------------------------------------------

#: Gate G1 - relative input-dependence of the stem embedding.
G1_MIN_STEM_SENSITIVITY = 0.05
#: Gate G2 - per-class standard deviation of the logits across a batch.
G2_MIN_LOGIT_STD = 1e-2


def stem_sensitivity(embedding: torch.Tensor) -> float:
    """Gate G1: `S = mean_j(std_i(e_ij)) / mean_ij(|e_ij|)` over a stem
    embedding `e` of shape `[B, ...]` (i indexes the batch, j every other
    element). The fraction of the embedding's magnitude that actually VARIES
    with the input; 1.0 would mean the embedding is pure signal, 0.0 that the
    model is a constant function of its input.

    Measured pre-v15 on 64 real patches through a freshly built
    `--architecture recursive` model: 0.0025 (HSI) / 0.0032 (RGB) - i.e. the
    embedding entering the recursive core varied with the input by a quarter
    of one percent, because `SpectralTokenizer` added a ~0.006-magnitude value
    embedding to a ~0.55-magnitude constant positional encoding. See the v15
    plan, section 0.
    """
    e = embedding.detach().float().reshape(embedding.shape[0], -1)
    if e.shape[0] < 2:
        raise ValueError("stem_sensitivity needs a batch of at least 2 samples")
    denom = e.abs().mean()
    if float(denom) <= 0.0:
        return 0.0
    return float(e.std(dim=0).mean() / denom)


def extract_stem_embedding(model: torch.nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    """The stem embedding G1 is defined on: `stem_norm(stem(ctx_map))`, i.e.
    the spectral context after the projection into the backbone's working
    width and before any spatial/positional encoding is added.

    Returns `None` for a model that has no such stem (a wrapper this module
    does not recognise), so callers can degrade to the logit check alone
    rather than crashing a training run over a diagnostic.
    """
    base = getattr(model, "base_model", model)          # unwrap recon wrappers
    backbone = getattr(base, "backbone", None)
    pathway = getattr(backbone, "spectral_pathway", None)
    stem = getattr(backbone, "stem", None)
    stem_norm = getattr(backbone, "stem_norm", None)
    if pathway is None or stem is None or stem_norm is None:
        return None
    ctx_map, _ = pathway(x)
    normalize = getattr(backbone, "_normalize_ctx", None)
    if normalize is not None:
        ctx_map = normalize(ctx_map)
    return stem_norm(stem(ctx_map))


def run_sensitivity_check(model: torch.nn.Module, x: torch.Tensor,
                           min_logit_std: float = G2_MIN_LOGIT_STD,
                           min_stem_sensitivity: float = G1_MIN_STEM_SENSITIVITY,
                           check_disjoint_batches: bool = True) -> Dict:
    """v15 R0.2 - one forward pass that answers "does this model's output
    depend on its input at all?".

    A NEW function, deliberately not folded into `run_smoke_test`: that
    function's contract (one forward/backward/optimizer-step finiteness cycle)
    is depended on by `train_example_v13/v14.py` and is left untouched.

    Returns a report dict with `"passed"`, the measured `logit_std` and
    `stem_sensitivity` (gates G2 and G1), and on failure a `reason`. The check
    is diagnostic-only: it never mutates the model, never steps an optimizer,
    and runs under `no_grad` in eval mode, restoring the original training
    flag afterwards.

    `check_disjoint_batches` additionally splits `x` in half and asserts the
    two halves produce different MEAN logits - a model can have non-zero
    per-sample logit spread from dropout-like noise alone while still being
    blind to the actual content of a batch.
    """
    was_training = model.training
    model.eval()
    report: Dict = {"passed": False, "check": "representation_sensitivity"}
    try:
        with torch.no_grad():
            embedding = extract_stem_embedding(model, x)
            if embedding is not None:
                report["stem_sensitivity"] = stem_sensitivity(embedding)
                report["min_stem_sensitivity"] = float(min_stem_sensitivity)

            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits.detach().float()
            report["logit_std"] = float(logits.std(dim=0).mean())
            report["min_logit_std"] = float(min_logit_std)
            report["logit_shape"] = tuple(logits.shape)

            if check_disjoint_batches and logits.shape[0] >= 4:
                half = logits.shape[0] // 2
                delta = (logits[:half].mean(dim=0) - logits[half:].mean(dim=0)).abs().max()
                report["disjoint_batch_mean_logit_delta"] = float(delta)
    finally:
        model.train(was_training)

    failures = []
    if report["logit_std"] < min_logit_std:
        failures.append(
            f"logit std {report['logit_std']:.3e} < {min_logit_std:.3e} (gate G2): the model's "
            f"output barely varies across the batch - it is close to a constant function")
    stem = report.get("stem_sensitivity")
    if stem is not None and stem < min_stem_sensitivity:
        failures.append(
            f"stem sensitivity {stem:.3e} < {min_stem_sensitivity:.3e} (gate G1): the embedding "
            f"entering the backbone is dominated by input-independent constants (see "
            f"medmamba_ss_trm.SpectralTokenizer and MedMambaSSTRMConfig.spectral_token_fusion)")

    report["passed"] = not failures
    if failures:
        report["reason"] = "; ".join(failures)
    return report
