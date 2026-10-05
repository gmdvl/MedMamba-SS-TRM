# -*- coding: utf-8 -*-
"""
training/gates.py
=======================
MedMamba-SS-TRM v16 plan - new preflight/training gates. `training/
numerical_stability.py` is imported at `train_example_v15.py:103` and by
v9/v10/v11, so it is FROZEN; every gate here wraps it (or is entirely new)
rather than editing it.

  G7  Reconstruction gradient reaches the encoder (R-1, R-2). One preflight
      batch: zero the classification term, backprop
      `lambda_mse*mse + lambda_sam*sam` alone, and assert parameters INSIDE
      the recursive core (not just the decoder) received finite, non-zero
      gradient. `RECONSTRUCTION_DETACHED` otherwise. This is the structural
      guarantee that R-2's fix (`training/recursive_features.py`) stays fixed
      - if someone re-introduces a detach between the core and the decoder,
      this gate fails loudly instead of the loss quietly floor-ing again.

  G8  Post-normalization split drift (A-3). The check that would have caught
      patient 68 in seconds: sample patches per split, apply the RESOLVED
      normalization, and fail when a split's post-norm mean/std has drifted
      too far from the train split's own post-norm mean/std (in train-group
      sigmas) - exactly what a fixed, train-fit affine (`global_zscore`) does
      to an under-exposed acquisition it was never fit on.

  C-2 A margin wrapper around `numerical_stability.run_sensitivity_check`
      (G1/G2): keeps its verdict, adds `{"value", "threshold", "margin"}` per
      gate (warning below 3x), and additionally gates on
      `disjoint_batch_mean_logit_delta`, which the frozen function measures
      but never puts into `failures`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from training.numerical_stability import (  # frozen, imported not edited
    run_sensitivity_check, G1_MIN_STEM_SENSITIVITY, G2_MIN_LOGIT_STD,
)
from training.normalization import normalize_patch

MARGIN_WARN_BELOW = 3.0
RECONSTRUCTION_DETACHED = "RECONSTRUCTION_DETACHED"


# ============================================================================
# C-2 - margin reporting around G1/G2 (+ the disjoint-batch check)
# ============================================================================

def run_sensitivity_check_with_margin(
    model: torch.nn.Module, x: torch.Tensor,
    min_logit_std: float = G2_MIN_LOGIT_STD,
    min_stem_sensitivity: float = G1_MIN_STEM_SENSITIVITY,
    min_disjoint_delta: float = 1e-4,
) -> Dict:
    """Wraps the frozen `run_sensitivity_check` - same forward pass, same
    verdict on G1/G2 - and adds:

      - `"margins"`: `{gate: {"value", "threshold", "margin": value/threshold,
        "warn": margin < 3.0}}` for every gate that has a threshold. RGB's
        G1/G2 both clear their thresholds but by <2.2x (C-2's finding); this
        is what makes that visible instead of a bare PASS.
      - a THIRD gate on `disjoint_batch_mean_logit_delta`, which the frozen
        function computes (`numerical_stability.py:482-485`) and reports but
        never fails on: two disjoint halves of the same batch producing
        (near-)identical mean logits means the model is not distinguishing
        the CONTENT of a batch, even if G1/G2 both pass.
    """
    report = run_sensitivity_check(model, x, min_logit_std=min_logit_std,
                                    min_stem_sensitivity=min_stem_sensitivity,
                                    check_disjoint_batches=True)

    margins: Dict[str, Dict] = {}
    logit_std = report.get("logit_std")
    if logit_std is not None:
        margins["G2_logit_std"] = {"value": logit_std, "threshold": min_logit_std,
                                    "margin": (logit_std / min_logit_std) if min_logit_std else float("inf")}
    stem = report.get("stem_sensitivity")
    if stem is not None:
        margins["G1_stem_sensitivity"] = {"value": stem, "threshold": min_stem_sensitivity,
                                           "margin": (stem / min_stem_sensitivity)
                                           if min_stem_sensitivity else float("inf")}
    for m in margins.values():
        m["warn_weak_margin"] = m["margin"] < MARGIN_WARN_BELOW

    failures = list(report.get("reason", "").split("; ")) if report.get("reason") else []
    failures = [f for f in failures if f]

    delta = report.get("disjoint_batch_mean_logit_delta")
    if delta is not None:
        margins["G9_disjoint_batch_delta"] = {
            "value": delta, "threshold": min_disjoint_delta,
            "margin": (delta / min_disjoint_delta) if min_disjoint_delta else float("inf"),
        }
        margins["G9_disjoint_batch_delta"]["warn_weak_margin"] = margins["G9_disjoint_batch_delta"]["margin"] < MARGIN_WARN_BELOW
        if delta < min_disjoint_delta:
            failures.append(
                f"disjoint-batch mean logit delta {delta:.3e} < {min_disjoint_delta:.3e} (gate G9): "
                f"two disjoint halves of the same batch produced ~identical mean logits - the model "
                f"is not sensitive to batch CONTENT even though G1/G2 may pass")
            report["passed"] = False

    report["margins"] = margins
    report["reason"] = "; ".join(failures) if failures else None
    return report


# ============================================================================
# G7 - reconstruction gradient reaches the encoder
# ============================================================================

def default_encoder_params(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    """Parameters INSIDE the recursive core - not the decoder, not the head.
    Unwraps `MedMambaSSTRMLatentReconWrapper(V2)` -> `base_model` ->
    `backbone.core`, which is exactly the module R-2 was about (the shared
    weight core `RecursiveCore.f`)."""
    base = getattr(model, "base_model", model)
    backbone = getattr(base, "backbone", None)
    core = getattr(backbone, "core", None)
    if core is None:
        return {}
    return dict(core.named_parameters())


def check_reconstruction_gradient(
    model: torch.nn.Module,
    forward_with_recon_fn: Callable,
    x: torch.Tensor,
    sam_loss_fn: Callable,
    lambda_mse: float,
    lambda_sam: float,
    encoder_params_fn: Callable = default_encoder_params,
) -> Dict:
    """Gate G7. `forward_with_recon_fn(model, x) -> (logits_list_or_logits,
    q_list_or_None, x_recon)` - the same helper both the training and
    validation loops use (`TrainerG_v12._forward_with_recon`, R-5), so this
    gate exercises the ACTUAL path a real epoch runs, not a special case.

    Zeroes every parameter's `.grad`, backprops ONLY
    `lambda_mse*mse + lambda_sam*sam` (the classification term is never
    computed here), and asserts every encoder parameter with
    `requires_grad=True` received a finite, non-zero gradient. Restores the
    model's `.training` flag and clears gradients again before returning, so
    this is safe to call as a pure preflight step.
    """
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    report: Dict = {"gate": "G7", "passed": False}
    try:
        _logits, _q, x_recon = forward_with_recon_fn(model, x)
        if x_recon is None:
            report["reason"] = "forward_with_recon_fn returned no reconstruction (recon_mode=none?)"
            return report

        mse = F.mse_loss(x_recon, x)
        sam = sam_loss_fn(x, x_recon)
        loss = lambda_mse * mse + lambda_sam * sam
        report["mse"] = float(mse.detach())
        report["sam"] = float(sam.detach())
        loss.backward()

        encoder_params = encoder_params_fn(model)
        if not encoder_params:
            report["reason"] = "no encoder parameters found (encoder_params_fn returned empty - " \
                                "is this a recursive architecture?)"
            return report

        dead, nonfinite = [], []
        for name, p in encoder_params.items():
            if not p.requires_grad:
                continue
            if p.grad is None:
                dead.append(name)
                continue
            if not torch.isfinite(p.grad).all():
                nonfinite.append(name)
                continue
            if float(p.grad.abs().sum()) == 0.0:
                dead.append(name)

        report["n_encoder_params_checked"] = sum(1 for p in encoder_params.values() if p.requires_grad)
        report["dead_params"] = dead[:10]
        report["n_dead"] = len(dead)
        report["nonfinite_params"] = nonfinite[:10]
        report["n_nonfinite"] = len(nonfinite)

        if dead or nonfinite:
            report["reason"] = (
                f"{RECONSTRUCTION_DETACHED}: {len(dead)} encoder parameter(s) received zero/no "
                f"gradient and {len(nonfinite)} received a non-finite gradient from the "
                f"reconstruction loss alone. The decoder is not backpropagating into the recursive "
                f"core (R-2) - check that forward_with_recon_fn decodes a LIVE tensor, not a "
                f"detached one.")
        else:
            report["passed"] = True
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    return report


# ============================================================================
# G8 - post-normalization split drift
# ============================================================================

def _sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    idx.sort()
    return idx


def _split_stats(x_path: str, mode: str, indices: np.ndarray,
                  global_mean=None, global_std=None):
    X = np.load(x_path, mmap_mode="r")
    raw_vals, norm_vals = [], []
    for i in indices:
        patch = np.array(X[i], dtype=np.float32)
        patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
        raw_vals.append(float(patch.mean()))
        normalized, _off, _scale = normalize_patch(patch, mode, global_mean=global_mean, global_std=global_std)
        norm_vals.append(normalized.ravel())
    norm_arr = np.concatenate(norm_vals) if norm_vals else np.zeros((0,), dtype=np.float32)
    return {
        "raw_mean": float(np.mean(raw_vals)) if raw_vals else float("nan"),
        "post_norm_mean": float(norm_arr.mean()) if norm_arr.size else float("nan"),
        "post_norm_std": float(norm_arr.std()) if norm_arr.size else float("nan"),
        "_norm_values": norm_arr,
    }


def check_split_drift(
    x_paths: Dict[str, str],
    mode: str,
    *,
    global_mean=None, global_std=None,
    groups: Optional[Dict[str, np.ndarray]] = None,
    n_patches: int = 8000, seed: int = 0,
    fail_mean_sigma: float = 0.25, fail_std_range=(0.75, 1.25),
    exp_dir: Optional[str] = None,
) -> Dict:
    """Gate G8. `x_paths`: `{"train": "...X_train.npy", "validation": "...",
    "test": "..."}` (missing keys are skipped). `groups[split]`: optional
    int array of patient ids, ALIGNED to the split's full `X_*.npy` (i.e.
    `groups[split][i]` is patient id of row `i`) - when present, per-patient
    outliers are named explicitly.
    """
    if "train" not in x_paths:
        raise ValueError("check_split_drift requires a 'train' split to compare against")

    report: Dict = {"gate": "G8", "mode": mode, "n_patches_requested": n_patches, "splits": {}}
    stats_by_split = {}
    indices_by_split = {}
    for split, path in x_paths.items():
        n = len(np.load(path, mmap_mode="r"))
        idx = _sample_indices(n, n_patches, seed=seed + (0 if split == "train" else 1))
        indices_by_split[split] = idx
        stats_by_split[split] = _split_stats(path, mode, idx, global_mean=global_mean, global_std=global_std)

    train_stats = stats_by_split["train"]
    train_p1 = float(np.percentile(train_stats["_norm_values"], 1)) if train_stats["_norm_values"].size else float("nan")
    train_sigma = train_stats["post_norm_std"] if train_stats["post_norm_std"] > 1e-9 else 1.0

    overall_pass = True
    for split, stats in stats_by_split.items():
        norm_vals = stats["_norm_values"]
        frac_below_p1 = float((norm_vals < train_p1).mean()) if norm_vals.size else float("nan")
        mean_sigma = (stats["post_norm_mean"] - train_stats["post_norm_mean"]) / train_sigma
        std_ratio = stats["post_norm_std"] / train_sigma

        split_report = {
            "n_sampled": int(len(indices_by_split[split])),
            "raw_mean": stats["raw_mean"],
            "post_norm_mean": stats["post_norm_mean"],
            "post_norm_std": stats["post_norm_std"],
            "frac_below_train_p1": frac_below_p1,
            "mean_sigma_vs_train": mean_sigma,
            "std_ratio_vs_train": std_ratio,
        }

        is_reference = split == "train"
        failed = (not is_reference) and (
            abs(mean_sigma) > fail_mean_sigma or not (fail_std_range[0] <= std_ratio <= fail_std_range[1]))
        split_report["passed"] = not failed
        if failed:
            overall_pass = False
            split_report["reason"] = (
                f"post-norm mean {mean_sigma:+.3f} train-sigma (limit +-{fail_mean_sigma}) or "
                f"post-norm std ratio {std_ratio:.3f} outside {list(fail_std_range)}")

        if groups is not None and split in groups:
            g = np.asarray(groups[split])
            idx = indices_by_split[split]
            per_patient: Dict[str, Dict] = {}
            patients_here = g[idx] if len(g) == len(np.load(x_paths[split], mmap_mode="r")) else None
            if patients_here is not None:
                for pid in sorted(set(patients_here.tolist())):
                    mask = patients_here == pid
                    if not mask.any():
                        continue
                    p_norm = norm_vals.reshape(len(idx), -1)[mask] if norm_vals.size else np.zeros((0,))
                    p_mean = float(p_norm.mean()) if p_norm.size else float("nan")
                    p_mean_sigma = (p_mean - train_stats["post_norm_mean"]) / train_sigma
                    per_patient[str(pid)] = {"n_sampled": int(mask.sum()), "post_norm_mean": p_mean,
                                              "mean_sigma_vs_train": p_mean_sigma,
                                              "outlier": abs(p_mean_sigma) > fail_mean_sigma}
                split_report["per_patient"] = per_patient
                outliers = sorted((pid for pid, v in per_patient.items() if v["outlier"]),
                                   key=lambda pid: -abs(per_patient[pid]["mean_sigma_vs_train"]))
                split_report["patient_outliers"] = outliers

        report["splits"][split] = split_report

    report["passed"] = overall_pass
    if exp_dir is not None:
        # `_norm_values` is per-pixel data purely for the sigma math above -
        # strip it before serializing.
        serializable = json.loads(json.dumps(report, default=lambda o: None))
        for split in serializable.get("splits", {}):
            serializable["splits"][split].pop("_norm_values", None)
        with open(Path(exp_dir) / "split_drift_report.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)
    return report


def enforce_split_drift(report: Dict, on_split_drift: str, logger=None) -> None:
    """`--on_split_drift {abort,warn,off}` (default `abort`)."""
    if on_split_drift == "off" or report.get("passed", True):
        return
    failed = [s for s, r in report["splits"].items() if not r.get("passed", True)]
    msg = f"[gate G8] SPLIT_DRIFT: post-normalization drift on split(s) {failed}: " \
          + "; ".join(f"{s}: {report['splits'][s].get('reason')}" for s in failed)
    if on_split_drift == "abort":
        raise SystemExit(f"SPLIT_DRIFT_ABORT: {msg}")
    (logger.warning if logger is not None else print)(msg)
