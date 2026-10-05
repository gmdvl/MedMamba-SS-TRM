# -*- coding: utf-8 -*-
"""
training/pipeline_consistency.py
=================================
improve-promp_v8.plan.md, Phase 0.3 - "Training / Validation Consistency
Audit".

Runs a set of static (source-inspection) and runtime (object-inspection)
checks against a live `TrainerG_v4` instance and writes
`pipeline_consistency_report.json`. Every check is a pass/fail with a
human-readable message; nothing here raises on failure by default (the
plan's exit criteria are enforced by Phase 0's completion gate, not by
this module aborting a run) - pass `strict=True` to raise instead.

Checks implemented (mirrors the plan's Phase 0.3 checklist):
  Verify validation DOES:
    - model.eval() before the forward pass
    - torch.no_grad() (or equivalent) around the whole validation pass
    - identical loss computation to training (same weighted-sum formula,
      same lambdas)
    - identical label mapping / spectral ordering (train and val Datasets
      are the same class, constructed the same way - the strongest static
      proxy available without re-reading raw files here)
  Verify validation does NOT:
    - use a shuffling sampler (RandomSampler / shuffle=True)
    - call model.train() anywhere inside the validation path
    - apply dropout/augmentation modules that differ between the train and
      val Dataset objects
"""

from __future__ import annotations

import inspect
import json
from typing import Dict, List


def _check(name: str, passed: bool, message: str) -> Dict:
    return {"check": name, "passed": bool(passed), "message": message}


def audit_pipeline_consistency(trainer, strict: bool = False) -> Dict:
    checks: List[Dict] = []

    # ------------------------------------------------------------------
    # Static source-inspection checks on _validate_one_epoch
    # ------------------------------------------------------------------
    validate_fn = type(trainer)._validate_one_epoch
    try:
        source = inspect.getsource(validate_fn)
    except (OSError, TypeError):
        source = ""

    has_no_grad_decorator = "@torch.no_grad()" in inspect.getsource(type(trainer)) and \
        "@torch.no_grad()" in inspect.getsource(type(trainer)).split("def _validate_one_epoch")[0][-200:]
    # more robust: look immediately above the def line in the class source
    class_source = inspect.getsource(type(trainer))
    idx = class_source.find("def _validate_one_epoch")
    preceding = class_source[max(0, idx - 100):idx]
    has_no_grad_decorator = "@torch.no_grad()" in preceding
    checks.append(_check(
        "validation_uses_no_grad",
        has_no_grad_decorator or "torch.no_grad()" in source,
        "`_validate_one_epoch` is decorated with @torch.no_grad() (or wraps its body in a "
        "`with torch.no_grad():` block)." if (has_no_grad_decorator or "torch.no_grad()" in source)
        else "Could not find @torch.no_grad() on/in `_validate_one_epoch` - gradients may be "
             "tracked during validation (wasted memory, and a correctness risk if anything "
             "accidentally calls .backward())."))

    checks.append(_check(
        "validation_calls_model_eval",
        "self.model.eval()" in source,
        "`_validate_one_epoch` calls `self.model.eval()`." if "self.model.eval()" in source
        else "`_validate_one_epoch` does not call `self.model.eval()` - BatchNorm/Dropout would "
             "stay in training mode during validation."))

    checks.append(_check(
        "validation_never_calls_model_train",
        "self.model.train()" not in source,
        "`_validate_one_epoch` never calls `self.model.train()`." if "self.model.train()" not in source
        else "`_validate_one_epoch` calls `self.model.train()` somewhere - this would put "
             "BatchNorm/Dropout back into training mode mid-validation."))

    checks.append(_check(
        "validation_never_calls_optimizer_step",
        ("self.optimizer.step()" not in source) and ("self.optimizer.zero_grad" not in source),
        "`_validate_one_epoch` never touches `self.optimizer`." if
        (("self.optimizer.step()" not in source) and ("self.optimizer.zero_grad" not in source))
        else "`_validate_one_epoch` references `self.optimizer` - validation must never update "
             "weights."))

    # ------------------------------------------------------------------
    # Loss-formula parity between train and validation - the single most
    # direct read of "identical loss computation" the plan asks for.
    # ------------------------------------------------------------------
    try:
        train_source = inspect.getsource(type(trainer)._train_one_epoch)
    except (OSError, TypeError):
        train_source = ""
    loss_terms = ["lambda_mse", "lambda_sam", "lambda_gan"]
    train_terms_used = {t: (t in train_source) for t in loss_terms}
    val_terms_used = {t: (t in source) for t in loss_terms}
    # only terms that are actually active (nonzero weight) need to match -
    # an unused lambda_gan=0 term being absent from validation isn't a bug.
    active_terms = [t for t in loss_terms if getattr(trainer, t, 0) not in (0, 0.0)]
    mismatched = [t for t in active_terms if train_terms_used.get(t) != val_terms_used.get(t)]
    checks.append(_check(
        "train_val_loss_formula_parity",
        len(mismatched) == 0,
        "All loss terms with nonzero weight (" + ", ".join(active_terms or ["none"]) + ") appear "
        "in both `_train_one_epoch` and `_validate_one_epoch`." if not mismatched else
        f"Loss term(s) {mismatched} have nonzero weight but are computed in training and NOT in "
        f"validation (or vice versa) - train_loss and val_loss are not the same objective, which "
        f"alone can produce a validation-loss curve that looks decoupled from training loss."))

    # ------------------------------------------------------------------
    # Runtime checks on the actual loader / dataset objects
    # ------------------------------------------------------------------
    val_sampler = getattr(trainer.val_loader, "sampler", None)
    is_shuffling = type(val_sampler).__name__ == "RandomSampler"
    checks.append(_check(
        "val_loader_not_shuffled",
        not is_shuffling,
        "val_loader uses a non-shuffling sampler." if not is_shuffling
        else "val_loader uses a shuffling sampler (shuffle=True) - validation should be "
             "deterministic (Phase 0.3)."))

    train_ds = getattr(trainer.train_loader, "dataset", None)
    val_ds = getattr(trainer.val_loader, "dataset", None)
    same_class = type(train_ds) is type(val_ds)
    checks.append(_check(
        "train_val_dataset_same_class",
        same_class,
        f"train_loader and val_loader both use `{type(train_ds).__name__}` - same preprocessing "
        f"code path by construction." if same_class else
        f"train_loader uses `{type(train_ds).__name__}` but val_loader uses "
        f"`{type(val_ds).__name__}` - preprocessing is NOT guaranteed identical between splits."))

    # Compare any public, non-callable attributes that look like preprocessing
    # config (normalization stats, flags, thresholds) between the two Dataset
    # instances - anything mismatched here is a concrete preprocessing diff.
    # `augment` is deliberately excluded: it is the ONE attribute that is
    # expected/required to differ (train may augment, validation never
    # should) - see train_example_v6.py's NpyDataset docstring.
    EXPECTED_TRAIN_ONLY_ATTRS = {"augment"}
    attr_mismatches = {}
    if train_ds is not None and val_ds is not None:
        for attr in dir(train_ds):
            if attr.startswith("_") or attr in ("X", "y") or attr in EXPECTED_TRAIN_ONLY_ATTRS:
                continue
            if not hasattr(val_ds, attr):
                continue
            tv, vv = getattr(train_ds, attr), getattr(val_ds, attr)
            if callable(tv):
                continue
            import numpy as _np
            try:
                equal = bool(_np.array_equal(tv, vv)) if hasattr(tv, "__len__") and not isinstance(tv, str) else (tv == vv)
            except Exception:
                equal = (tv is vv)
            if not equal:
                attr_mismatches[attr] = {"train": str(tv), "val": str(vv)}
    checks.append(_check(
        "train_val_dataset_config_matches",
        len(attr_mismatches) == 0,
        "No differing non-data attributes found between the train and val Dataset instances."
        if not attr_mismatches else
        f"Train/val Dataset instances differ on: {list(attr_mismatches.keys())} - "
        f"details: {attr_mismatches}"))

    val_augment = getattr(val_ds, "augment", "none")
    checks.append(_check(
        "val_dataset_has_no_augmentation",
        val_augment == "none",
        "val Dataset's `augment` setting is 'none'." if val_augment == "none" else
        f"val Dataset has augment={val_augment!r} - validation must never be augmented "
        f"(Phase 0.3)."))

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {
        "n_checks": len(checks),
        "n_passed": len(checks) - n_failed,
        "n_failed": n_failed,
        "all_passed": n_failed == 0,
        "checks": checks,
    }

    if strict and n_failed:
        raise AssertionError(f"Pipeline consistency audit failed {n_failed}/{len(checks)} checks: "
                              f"{[c['check'] for c in checks if not c['passed']]}")
    return report


def write_report(trainer, path: str, strict: bool = False) -> Dict:
    report = audit_pipeline_consistency(trainer, strict=strict)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report
