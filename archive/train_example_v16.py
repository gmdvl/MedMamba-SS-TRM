# -*- coding: utf-8 -*-
"""
train_example_v16.py
======================
GMedMamba v16 plan - "Acquisition, Reconstruction & Reporting Remediation".
New entry point, built ON TOP OF `train_example_v15.py` (imported as a
module, never edited - the freezing rule: v15 stays bit-reproducible for the
two audited runs). Every dataset-discovery / integrity / leakage / DDP-free
preflight helper that v15 already gets right is reused verbatim via
`train_example_v15.<name>`; this file only replaces what the v16 audit found
broken:

  A-1/A-2   `--normalization` gains `per_patch_zscore` (the audit's fix for
            patient 68's under-exposed acquisition); `auto` now resolves HSI
            to it instead of `global_zscore`.
  A-3       gate G8 (`training.gates_v16.check_split_drift`) runs before
            `trainer.fit`.
  A-4       per-patient/per-capture group sidecars (`scripts/
            derive_patch_groups.py`), when present next to the dataset, are
            loaded and threaded into `TrainerG_v12`.
  A-5       validation subsampling can stratify BY PATIENT, not just by class.
  R-1       `--recon_mode` is no longer force-set to "none" for
            `--architecture recursive`; gate G7 replaces the silent override.
  R-2..R-7  `training.reconstruction_head_v2.GMedMambaLatentReconWrapperV2` +
            `training.trainerg_v12.TrainerG_v12` (see their docstrings).
  E-1..E-4  `TrainerG_v12`'s stopping rules/step-axis scheduler, this file's
            step-axis warmup+cosine schedule, `AugmentedPatchDatasetV16`.
  C-1/C-2   `--eval_group_aggregation`, margin-reporting sensitivity gate.

Nothing in this file is imported by `train_example_v15.py` or any frozen
`trainerg_v3..v11.py` - importing v15 as a module is one-directional.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import train_example_v15 as v15
from train_example_v6 import setup_experiment_dir, compute_global_channel_stats
from train_example_v7 import GMedMambaRawReconWrapper
from training.trainerg_v12 import TrainerG_v12
from training.numerical_stability import StabilityConfig
from training.gates_v16 import (
    run_sensitivity_check_with_margin, check_split_drift, enforce_split_drift,
    check_reconstruction_gradient,
)
from training.gan import SpectralDiscriminator
from training.class_coverage import verify_class_coverage
from training.class_imbalance import write_imbalance_report
from training.activation_patch import replace_relu_with_leakyrelu
from training.reconstruction_head_v2 import (
    GMedMambaLatentReconWrapperV2, resolve_out_activation, validate_out_activation_for_normalization,
)
from training.losses import build_criterion
from training.samplers import build_balanced_sampler, build_moderate_oversample_indices
from training.augmentation import AugmentationConfig, PRESETS
from training.augmentation_v16 import AugmentedPatchDatasetV16, build_active
from training.config_presets import build_model, v15_config_overrides
from training.optim_groups import build_param_groups, summarize_param_groups
from training.grad_checkpoint import enable_gradient_checkpointing
from training.spectral_checkpoint import enable_spectral_gradient_checkpointing
from training.torch_compile import enable_torch_compile
from training.dataloader_config_v16 import train_val_policies_v16
from training.fd_limit_v16 import raise_open_file_limit, is_fd_exhaustion
from training.dataloader_config import apply_main_process_thread_limits, seed_everything
from training.system_memory import (
    collect_system_memory_snapshot, log_system_memory, write_system_memory_report,
)
from training.npy_integrity import (
    validate_dataset, write_integrity_report, write_dataset_failure_artifacts, DatasetStorageError,
)
from training.failure_taxonomy import FailureClass, classify_exception
from training.npy_dataset_v16 import NpyDatasetV16
from training.run_naming_v16 import build_run_name, dataset_slug, infer_modality, modality_warnings

_LAST_EXP_DIR = {"path": None}


# ============================================================================
# CLI - v15's parser, patched
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = v15.build_arg_parser()
    p.description = "GMedMamba v16 ('Acquisition, Reconstruction & Reporting Remediation') training entry point"

    # A-2 - a third normalization mode. argparse has no public API to widen
    # an existing action's `choices`, so find it and mutate the attribute -
    # the only way to add a choice without re-declaring the whole argument
    # (which would need `conflict_handler='resolve'` and lose its help text).
    for action in p._actions:
        if action.dest == "normalization":
            action.choices = ["per_sample_minmax", "global_zscore", "per_patch_zscore", "auto"]

    g = p.add_argument_group("v16")
    g.add_argument("--drift_check_patches", type=int, default=8000,
                    help="gate G8: patches sampled per split for the post-normalization drift check")
    g.add_argument("--on_split_drift", choices=["abort", "warn", "off"], default="abort")
    g.add_argument("--min_patient_recall", type=float, default=None,
                    help="A-4: log a PATIENT_OUTLIER warning when any validation patient's "
                         "recall_macro drops below this while the checkpoint metric rises. "
                         "None (default) = off.")
    g.add_argument("--val_divergence_patience", type=int, default=3,
                    help="E-1: epochs of (worsening checkpoint metric + rising val_loss) before "
                         "VAL_DIVERGENCE. 0 disables.")
    g.add_argument("--scheduler_interval", choices=["epoch", "step"], default="step",
                    help="E-2: step the LR schedule per optimizer step (default) or per epoch "
                         "(v11's behaviour, desynchronized from --train_subsample_frac/early stopping).")
    g.add_argument("--warmup_steps", type=int, default=None,
                    help="E-2: linear warmup length in optimizer steps. None = 3%% of total steps, "
                         "floor 200.")
    g.add_argument("--recon_out_activation", choices=["linear", "sigmoid", "auto"], default="auto",
                    help="R-3: decoder output activation. 'auto' resolves from --normalization "
                         "(linear for any z-score mode, sigmoid for per_sample_minmax) and refuses "
                         "to start on an explicit mismatch.")
    g.add_argument("--eval_group_aggregation", choices=["none", "mean_prob", "majority"], default="none",
                    help="C-1: additionally report IMAGE-level test metrics (patch-MIL), when "
                         "images_{split}.npy (scripts/derive_patch_groups.py) is present.")
    g.add_argument("--group_sidecar_dir", default=None,
                    help="directory holding groups_*.npy/captures_*.npy/images_*.npy "
                         "(default: next to X_val.npy)")
    g.add_argument("--patch_size", type=int, default=None,
                    help="override the architecture preset's patch-embedding stride. The "
                         "hierarchical presets use 4, which was chosen for 224x224 inputs and "
                         "reduces an 11x11 patch to a 2x2 token grid before the backbone starts. "
                         "Pass 1 to keep full spatial resolution on small patches, as the "
                         "recursive preset and the MedMamba-HSI baseline both do. None = preset.")
    g.add_argument("--g7_check", choices=["auto", "on", "off"], default="auto",
                    help="gate G7 (reconstruction gradient reaches the encoder): 'auto' runs it "
                         "whenever architecture=recursive and recon_mode=latent.")
    g.add_argument("--fast_loop", choices=["on", "off"], default="off",
                    help="v17 S12: run the training epoch through "
                         "`training.trainerg_v12_fast.TrainerG_v12Fast`, which performs ONE "
                         "device-to-host sync per step instead of ~17 and updates the EMA shadow "
                         "with two fused kernels instead of ~4 per parameter tensor. The training "
                         "metrics are bit-identical (asserted in test_fast_loop_v17.py); the EMA "
                         "shadow is not, because the fused update reassociates the same "
                         "expression (~1e-8 over 300 steps). 'off' here so this bare entry point "
                         "stays byte-comparable to every run already in experiments/; "
                         "OPTIMAL_DEFAULTS turns it on, exactly as it does for --compile.")
    return p


def apply_v16_defaults(args: argparse.Namespace, explicit: set) -> list:
    """v15's `apply_v15_defaults`, minus R4.4's `recon_mode -> none` override
    (R-1: that override is exactly what this plan removes), plus A-2's
    `per_patch_zscore` resolution for `auto`."""
    notes = []

    def _set(name, value, why):
        if name in explicit:
            return
        if getattr(args, name, None) != value:
            notes.append(f"{name}={value}  ({why})")
        setattr(args, name, value)

    # v17 S7 - GMedMamba-R (TRM) is the default architecture.
    #
    # It already was on all three real entry points (OPTIMAL_DEFAULTS:423 and
    # MEDMAMBA_PROTOCOL_DEFAULTS:233 both set "recursive"); only this bare one still
    # inherited "split" from the FROZEN train_example_v15.py:209. Both documented
    # invocations of this file (documentations/README.md:120,127) pass --architecture
    # recursive explicitly, so no command anyone runs changes behaviour - it just stops
    # being something to remember.
    #
    # `_set` skips anything the user passed explicitly, so --architecture split |
    # fullchannel | efficient still selects the hierarchical backbones unchanged.
    _set("architecture", "recursive",
         "v17 - GMedMamba-R (TRM) is the default; pass --architecture "
         "split|fullchannel|efficient for the hierarchical backbones")

    if infer_modality(args.data_dir) == "hsi":
        _set("dataset_validation_level", "deep",
             "R6.5 (v15) - carried forward: this dataset family had a physically-unreadable "
             "region during prep")

    # A-2 - per_patch_zscore is HSI's default now, not global_zscore: it does
    # not need a fixed train-fit affine to reproduce a mis-exposed capture's
    # OWN scale, so the split-drift gate G8 cannot fail on it by construction.
    if args.normalization == "auto":
        resolved = "per_patch_zscore" if infer_modality(args.data_dir) == "hsi" else "per_sample_minmax"
        args.normalization = resolved
        notes.append(f"normalization={resolved}  (A-2 - auto, by modality, v16)")

    if args.recon_out_activation == "auto":
        args.recon_out_activation = resolve_out_activation(args.normalization)
        notes.append(f"recon_out_activation={args.recon_out_activation}  (R-3 - auto, by normalization)")
    else:
        validate_out_activation_for_normalization(args.recon_out_activation, args.normalization)

    return notes


# ============================================================================
# E-2 - step-axis warmup + cosine schedule
# ============================================================================

def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: Optional[int] = None):
    """Linear warmup (`warmup_steps`, default 3% of `total_steps` floored at
    200) into cosine decay to 0, on the STEP axis. `TrainerG_v12` calls
    `.step()` once per optimizer step when `scheduler_interval="step"`
    (the default here) - so `--train_subsample_frac`/early stopping changing
    how many steps an "epoch" is no longer desynchronizes the schedule from
    wall-clock epochs (E-2)."""
    total_steps = max(1, int(total_steps))
    if warmup_steps is None:
        warmup_steps = max(200, int(round(0.03 * total_steps)))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler.warmup_steps = warmup_steps      # stashed for logging/tests
    scheduler.total_steps = total_steps
    return scheduler


# ============================================================================
# A-4/A-5/C-1 - group sidecars
# ============================================================================

def _load_group_sidecars(x_val_path: str, sidecar_dir: Optional[str]):
    """Best-effort load of `scripts/derive_patch_groups.py`'s sidecars. Never
    raises - a missing/misaligned sidecar just means A-4/A-5/C-1 are
    unavailable for this run, not that training should fail."""
    base = Path(sidecar_dir) if sidecar_dir else Path(x_val_path).parent
    out = {"groups": {}, "captures": {}, "images": {}, "capture_index": {}}
    for split, fname in (("train", "groups_train.npy"), ("validation", "groups_val.npy"), ("test", "groups_test.npy")):
        p = base / fname
        if p.is_file():
            out["groups"][split] = np.load(p)
    for split, fname in (("train", "captures_train.npy"), ("validation", "captures_val.npy"), ("test", "captures_test.npy")):
        p = base / fname
        if p.is_file():
            out["captures"][split] = np.load(p)
    for split, fname in (("train", "images_train.npy"), ("validation", "images_val.npy"), ("test", "images_test.npy")):
        p = base / fname
        if p.is_file():
            out["images"][split] = np.load(p)
    ci_path = base / "capture_index.json"
    if ci_path.is_file():
        with open(ci_path) as f:
            out["capture_index"] = json.load(f)
    return out


def _subsample_indices_v16(labels: np.ndarray, frac: float, mode: str, num_classes: int,
                            seed: int, groups: Optional[np.ndarray] = None) -> np.ndarray:
    """`train_example_v15._subsample_indices`, ported, plus A-5's
    `stratified_group` mode: draw the per-class fraction WITHIN each patient,
    so a small subset still represents every patient in proportion (the
    validation split is 5 patients - a plain per-class draw can still starve
    one of them)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if mode == "random":
        keep = max(1, int(round(n * frac)))
        return np.sort(rng.choice(n, size=keep, replace=False))

    if mode == "stratified_group" and groups is not None:
        picked = []
        for g in sorted(set(groups.tolist())):
            g_idx = np.flatnonzero(groups == g)
            for c in range(num_classes):
                idx = g_idx[labels[g_idx] == c]
                if idx.size == 0:
                    continue
                keep = max(1, int(round(idx.size * frac)))
                picked.append(rng.choice(idx, size=min(keep, idx.size), replace=False))
        return np.sort(np.concatenate(picked)) if picked else np.arange(n)

    picked = []
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        keep = max(1, int(round(idx.size * frac)))
        picked.append(rng.choice(idx, size=min(keep, idx.size), replace=False))
    return np.sort(np.concatenate(picked)) if picked else np.arange(n)


# ============================================================================
# gate G6 - a test loader that knows about `per_patch_zscore`
# ============================================================================

def _build_test_loader_v16(args, data_dir, storage_mode, normalization, global_mean, global_std,
                            policy, device, val_x_path):
    """`train_example_v15._build_test_loader`, but building an
    `NpyDatasetV16` instead of the frozen `train_example_v6.NpyDataset`.

    v15's version is unusable for v16's own HSI default: `NpyDataset.__init__`
    asserts `normalization in ("per_sample_minmax", "global_zscore")`
    (`train_example_v6.py:42`), which predates `per_patch_zscore` - the mode
    A-2 made the default for HSI. Calling it with that mode raises a bare
    `AssertionError`, which the caller's `except Exception` turns into
    `WARNING: test-split evaluation failed: AssertionError:` and
    `gates.json: {"G6": false}`. Gate G6 was therefore unreachable for every
    v16 HSI run, silently, while the run itself completed normally
    (`20260904_230812`). Both frozen files stay untouched; only this
    entry point, which is not frozen, changes.

    `return_norm_stats=False` is deliberate and not merely a default:
    `training.evaluator.evaluate_model` does `sample_x, _ = next(iter(loader))`,
    which cannot unpack the 3-tuple a `return_norm_stats=True` dataset yields.
    Reconstruction metrics are not part of the G6 report anyway.
    """
    base = Path(data_dir)
    x_test, y_test = base / "X_test.npy", base / "y_test.npy"
    if not (x_test.is_file() and y_test.is_file()):
        print(f"WARNING: no X_test.npy/y_test.npy in {data_dir} - gate G6 cannot be satisfied.",
              flush=True)
        return None
    if Path(val_x_path).resolve() == x_test.resolve():
        print(f"WARNING: {data_dir} has no X_val.npy, so discover_data used X_test.npy as the "
              f"VALIDATION split. Skipping the test evaluation rather than reporting the "
              f"model-selection split as held-out. Gate G6 cannot be satisfied for this dataset "
              f"until it is re-prepped with a three-way split.", flush=True)
        return None

    test_ds = NpyDatasetV16(str(x_test), str(y_test), storage_mode=storage_mode,
                             normalization=normalization,
                             global_mean=global_mean, global_std=global_std,
                             return_norm_stats=False)
    print(f"[test-eval] {len(test_ds)} held-out test patches from {x_test}", flush=True)
    return DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       **policy.dataloader_kwargs(device, seed=args.seed))


# ============================================================================
# main
# ============================================================================

# v17 S9 - `Overrides` replaces the module-global rebinding chain.
#
# Composition used to work by assigning into THIS module's namespace from the derived
# entry points - `v16.build_arg_parser = ...`, `v16.build_run_name = ...`,
# `v16.build_model = ...`, `v16.TrainerG_v12 = ...` - in up to three layers, each
# overwriting the last, plus a save/restore of `train_example_v16_recon._v16_build_arg_parser`
# and a substitution of `optimal.config_deviations` performed from inside `print_banner`.
# It worked, but it was order-dependent, invisible at the call site, and most of
# `train_example_v16_optimal_recon.py`'s docstring existed to explain it.
#
# The class composition it enabled is good and is untouched:
# `TrainerG_v13Optimal(TrainerG_v12Optimal, TrainerG_v13)` with an empty body, where the
# MRO is the design. Only the rebinding goes: `_main` now TAKES what it should use.
# Every field defaults to this module's own, so `main()` is unchanged and any caller that
# still rebinds the module globals keeps working.
class Overrides:
    """What a derived entry point contributes to `_main`. All optional."""

    __slots__ = ("parser_fn", "run_name_fn", "model_fn", "trainer_cls")

    def __init__(self, parser_fn=None, run_name_fn=None, model_fn=None, trainer_cls=None):
        self.parser_fn = parser_fn
        self.run_name_fn = run_name_fn
        self.model_fn = model_fn
        self.trainer_cls = trainer_cls

    def resolved(self):
        """Fill every unset slot from this module's globals, which is also what makes a
        legacy `v16.<name> = ...` rebinding still take effect."""
        return Overrides(
            parser_fn=self.parser_fn or build_arg_parser,
            run_name_fn=self.run_name_fn or build_run_name,
            model_fn=self.model_fn or build_model,
            trainer_cls=self.trainer_cls or TrainerG_v12,
        )


def explicitly_passed(argv=None, parser_fn=None) -> set:
    """`train_example_v15.explicitly_passed`, but built from THIS module's
    `build_arg_parser()` - v15's own version builds v15's OWN (unpatched)
    parser, whose `--normalization` choices do not include `per_patch_zscore`
    and which has none of the v16-only flags, so it would reject a valid v16
    command line while merely trying to detect which flags were explicit."""
    p = (parser_fn or build_arg_parser)()
    for action in p._actions:
        action.default = argparse.SUPPRESS
    known, _ = p.parse_known_args(argv)
    return set(vars(known))


def _main(overrides: Optional["Overrides"] = None):
    ov = (overrides or Overrides()).resolved()
    p = ov.parser_fn()
    args = p.parse_args()
    explicit = explicitly_passed(parser_fn=ov.parser_fn)

    # v17 S12 - `--fast_loop on` puts the de-synced training epoch in front of whatever
    # trainer this entry point composed. Done HERE rather than in each `build_overrides`
    # because there are five entry points and this one has no `build_overrides` at all;
    # see `training.trainerg_v12_fast.make_fast` for the resulting MRO.
    if getattr(args, "fast_loop", "off") == "on":
        from training.trainerg_v12_fast import make_fast
        ov.trainer_cls = make_fast(ov.trainer_cls)
    v15.apply_safe_mode(args, explicit)
    v16_notes = apply_v16_defaults(args, explicit)

    seed_state = seed_everything(args.seed, deterministic=args.deterministic)
    apply_main_process_thread_limits(args.cpu_threads)
    # v17 - headroom for DataLoader worker descriptors. See training/fd_limit_v16.py: the
    # full test split is the largest loader in a run and is built last, and a 2-hour HSI
    # run lost gate G6 to "Too many open files" after training had completed normally.
    raise_open_file_limit()
    # v17 S12b - before ANY Triton compilation, which is lazy and first fires inside gate
    # G1/G2 or G7, long after this point. A run died at G7 with "[Errno 28] No space left
    # on device" writing a cubin into /tmp, which is a 3.1 GB tmpfs on this machine: its
    # free-space column is a size cap, not free memory. See the module docstring.
    from training.compile_cache_v17 import (
        ensure_compile_scratch_on_disk, warn_if_tmpdir_breaks_worker_ipc)
    # Unconditional: DataLoader workers exist whether or not the model is compiled, and an
    # over-long inherited TMPDIR kills them with a traceback that names multiprocessing.
    warn_if_tmpdir_breaks_worker_ipc()
    if args.compile_model == "on":
        ensure_compile_scratch_on_disk()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            print("WARNING: --amp bf16 requested but this GPU does not support bfloat16 - "
                  "falling back to --amp off (fp32).", flush=True)
            args.amp = "off"

    for note in v16_notes:
        print(f"[v16 default] {note}", flush=True)
    for w in modality_warnings(args.data_dir):
        print(f"WARNING: {w}", flush=True)

    startup_snapshot = log_system_memory(label="startup")

    run_modality = infer_modality(args.data_dir)
    run_name = ov.run_name_fn(args, modality=run_modality)
    exp_dir = setup_experiment_dir(exp_name=run_name, resume_path=args.resume, append_timestamp=False)
    _LAST_EXP_DIR["path"] = exp_dir
    write_system_memory_report(str(exp_dir / "system_memory_startup.json"), label="startup")

    print(f"Discovering dataset files in {args.data_dir}...", flush=True)
    x_tr, y_tr, x_te, y_te, wavelengths, num_classes, class_names, n_train, n_val = \
        v15.discover_data(args.data_dir)
    print(f"Dataset Loaded: {n_train} Train, {n_val} Val | Classes: {num_classes} | Device: {device}",
          flush=True)

    _real_modality = "hsi" if wavelengths is not None else "rgb"
    if _real_modality != run_modality:
        raise SystemExit(f"INTERNAL: modality mismatch - {run_modality!r} vs {_real_modality!r}.")

    dataset_paths = {"X_train": x_tr, "y_train": y_tr, "X_val": x_te, "y_val": y_te}
    integrity_report = None
    try:
        v15._check_manifest(args, exp_dir)
        if not args.skip_dataset_validation:
            integrity_report = validate_dataset(dataset_paths, abort_on_failure=True,
                                                  level=args.dataset_validation_level)
            write_integrity_report(integrity_report, str(exp_dir / "dataset_integrity_report.json"))
    except DatasetStorageError as e:
        print(f"DATASET_STORAGE_ERROR: {e}", flush=True)
        write_dataset_failure_artifacts(str(exp_dir), integrity_report, e,
                                         system_snapshot=collect_system_memory_snapshot())
        raise

    v15._run_leakage_check(args, exp_dir)

    y_train_arr = np.asarray(np.load(y_tr, mmap_mode="r"))
    y_val_arr = np.asarray(np.load(y_te, mmap_mode="r"))

    if args.check_class_coverage:
        verify_class_coverage({"train": y_train_arr, "validation": y_val_arr}, num_classes=num_classes,
                               class_names=class_names, fail_hard=not args.allow_missing_classes)
    write_imbalance_report(str(exp_dir / "class_imbalance_report.json"),
                            {"train": y_train_arr, "validation": y_val_arr},
                            num_classes=num_classes, class_names=class_names)

    # ------------------------------------------------------------------
    # A-1/A-2 - normalization
    # ------------------------------------------------------------------
    global_mean = global_std = None
    if args.normalization == "global_zscore":
        global_mean, global_std = compute_global_channel_stats(
            x_tr, sample_cap=args.norm_stats_sample_cap, seed=args.seed)

    return_norm_stats = args.recon_mode != "none"

    def _build_dataset(xp, yp):
        return NpyDatasetV16(xp, yp, storage_mode=args.dataset_storage, normalization=args.normalization,
                              global_mean=global_mean, global_std=global_std,
                              return_norm_stats=return_norm_stats)

    train_ds = _build_dataset(x_tr, y_tr)
    val_ds = _build_dataset(x_te, y_te)

    # ------------------------------------------------------------------
    # A-3/A-4/A-5 - group sidecars + gate G8
    # ------------------------------------------------------------------
    sidecars = _load_group_sidecars(x_te, args.group_sidecar_dir)
    groups_val_full = sidecars["groups"].get("validation")
    captures_val_full = sidecars["captures"].get("validation")
    groups_test_full = sidecars["groups"].get("test")
    images_test_full = sidecars["images"].get("test")
    capture_index = sidecars["capture_index"]

    x_paths_g8 = {"train": x_tr, "validation": x_te}
    x_test_path = Path(args.data_dir) / "X_test.npy"
    if x_test_path.is_file() and x_test_path.resolve() != Path(x_te).resolve():
        x_paths_g8["test"] = str(x_test_path)
    groups_g8 = {k: v for k, v in {"validation": groups_val_full, "test": groups_test_full}.items()
                 if v is not None}
    drift_report = check_split_drift(
        x_paths_g8, args.normalization, global_mean=global_mean, global_std=global_std,
        groups=groups_g8 or None, n_patches=args.drift_check_patches, seed=args.seed,
        exp_dir=str(exp_dir))
    enforce_split_drift(drift_report, args.on_split_drift)
    print(f"[gate G8] split drift: {'PASS' if drift_report['passed'] else 'FAIL'} "
          f"-> {exp_dir}/split_drift_report.json", flush=True)

    # ------------------------------------------------------------------
    # val subsample (A-5: patient-stratified when a group vector is available)
    # ------------------------------------------------------------------
    val_subsample_info = {"frac": args.val_subsample_frac, "mode": args.val_subsample_mode,
                          "kept": len(val_ds), "total": len(val_ds)}
    val_keep_idx = None
    if not 0.0 < args.val_subsample_frac <= 1.0:
        raise SystemExit(f"--val_subsample_frac must be in (0, 1], got {args.val_subsample_frac}")
    if args.val_subsample_frac < 1.0:
        mode = ("stratified_group" if (args.val_subsample_mode == "stratified" and groups_val_full is not None)
                else args.val_subsample_mode)
        val_keep_idx = _subsample_indices_v16(y_val_arr, args.val_subsample_frac, mode, num_classes,
                                               args.seed + 1, groups=groups_val_full)
        val_ds = Subset(val_ds, val_keep_idx.tolist())
        val_subsample_info.update(kept=len(val_keep_idx), mode=mode)
        print(f"[val-subsample] frac={args.val_subsample_frac} mode={mode} -> {len(val_keep_idx)} of "
              f"{val_subsample_info['total']} validation patches (seed={args.seed + 1})", flush=True)

    val_groups = groups_val_full[val_keep_idx] if (groups_val_full is not None and val_keep_idx is not None) \
        else groups_val_full
    val_captures = captures_val_full[val_keep_idx] if (captures_val_full is not None and val_keep_idx is not None) \
        else captures_val_full

    # ------------------------------------------------------------------
    # train sampler / subsample (v15's logic, ported verbatim)
    # ------------------------------------------------------------------
    train_sampler = None
    shuffle = True
    if args.sampler == "balanced":
        train_sampler = build_balanced_sampler(y_train_arr, num_classes)
        shuffle = False
    elif args.sampler == "moderate_oversample":
        idx = build_moderate_oversample_indices(y_train_arr, num_classes, seed=args.seed,
                                                 target_percentile=args.oversample_target_percentile)
        train_ds = Subset(train_ds, idx.tolist())

    if not 0.0 < args.train_subsample_frac <= 1.0:
        raise SystemExit(f"--train_subsample_frac must be in (0, 1], got {args.train_subsample_frac}")
    train_subsample_info = {"frac": args.train_subsample_frac, "mode": args.train_subsample_mode,
                            "kept": len(train_ds), "total": len(train_ds)}
    if args.train_subsample_frac < 1.0:
        if train_sampler is not None:
            full_n = train_sampler.num_samples
            train_sampler.num_samples = max(1, int(round(full_n * args.train_subsample_frac)))
            kept, total, resamples = train_sampler.num_samples, full_n, True
        elif args.train_subsample_mode == "per_epoch":
            total = len(train_ds)
            kept = max(1, int(round(total * args.train_subsample_frac)))
            gen = torch.Generator().manual_seed(args.seed)
            train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=False, num_samples=kept,
                                                             generator=gen)
            shuffle, resamples = False, True
        else:
            total = len(train_ds)
            kept = max(1, int(round(total * args.train_subsample_frac)))
            pick = np.random.default_rng(args.seed).permutation(total)[:kept]
            train_ds = Subset(train_ds, pick.tolist())
            resamples = False
        train_subsample_info.update(kept=kept, total=total, resamples_each_epoch=resamples)
        print(f"[train-subsample] frac={args.train_subsample_frac} mode={args.train_subsample_mode} -> "
              f"{kept} of {total} training samples per epoch (seed={args.seed})", flush=True)

    aug_cfg = (PRESETS[args.augment_preset] if args.augment_preset != "custom" else AugmentationConfig(
        spectral_noise_std=args.spectral_noise_std, spectral_scale_range=args.spectral_scale_range,
        spectral_offset_std=args.spectral_offset_std, band_dropout_prob=args.band_dropout_prob,
        band_dropout_max_frac=args.band_dropout_max_frac, spectral_mask_prob=args.spectral_mask_prob,
        spectral_mask_max_width_frac=args.spectral_mask_max_width_frac, flip_h_prob=args.flip_h_prob,
        flip_v_prob=args.flip_v_prob, rotate90_prob=args.rotate90_prob,
        crop_scale_max_frac=args.crop_scale_max_frac, seed=args.seed))
    augmented_train_ds = None
    if args.augment_preset != "none":
        # v17 S8 - `build_active` is `aug_cfg.build()` with the switched-off steps removed.
        # Bit-identical: every guard in the frozen module short-circuits before touching
        # the RNG, so a disabled step consumes no draws. See augmentation_v16.build_active.
        train_ds = AugmentedPatchDatasetV16(train_ds, build_active(aug_cfg), seed=args.seed)  # E-4
        augmented_train_ds = train_ds

    train_policy, val_policy = train_val_policies_v16(args.loader_mode, args.num_workers,
                                                        args.prefetch_factor, args.persistent_workers,
                                                        args.pin_memory)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=train_sampler,
                               **train_policy.dataloader_kwargs(device, seed=args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             **val_policy.dataloader_kwargs(device, seed=args.seed))

    modality = "hsi" if wavelengths is not None else "rgb"
    in_channels = int(v15._probe_in_channels(train_ds))
    system_snapshot = collect_system_memory_snapshot()
    v15.print_preflight(args, train_policy, val_policy, dataset_paths, system_snapshot,
                         in_channels=in_channels, modality=modality)
    print(f"  normalization: {args.normalization}  recon_mode: {args.recon_mode}  "
          f"recon_out_activation: {args.recon_out_activation}", flush=True)

    if args.loader_test:
        v15.run_loader_test(args, train_loader, train_policy)
        return

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    cfg_overrides = dict(v15_config_overrides(args.architecture))
    cfg_overrides.update({
        "spectral_token_fusion": args.spectral_token_fusion, "spectral_pe_gain": args.spectral_pe_gain,
        "spectral_value_init_std": args.spectral_value_init_std, "spectral_ctx_norm": args.spectral_ctx_norm,
        "classifier_init": args.classifier_init, "wavelength_encoding_scale": args.wavelength_scale,
        "spectral_chunk_size": args.spectral_chunk_size,
    })
    if args.patch_size is not None:
        cfg_overrides["patch_size"] = args.patch_size
    trm_kwargs = None
    if args.architecture == "recursive":
        trm_kwargs = dict(
            trm_dim=args.trm_dim, trm_core_layers=args.trm_core_layers, trm_n_latent=args.trm_n_latent,
            trm_n_improve=args.trm_n_improve, trm_deep_supervision_steps=args.trm_deep_supervision_steps,
            trm_mixer=args.trm_mixer, trm_ema_rate=args.trm_ema_rate, trm_act_halting=args.trm_halting,
            trm_halt_threshold=args.trm_halt_threshold, trm_halt_exploration_prob=args.trm_halt_exploration_prob,
            trm_spatial_pe_gain=args.spatial_pe_gain, trm_checkpoint_core=args.trm_checkpoint_core,
            trm_mixer_channel_mlp=args.trm_mixer_channel_mlp,
            trm_drop_path=(args.trm_drop_path if args.trm_drop_path > 0 else (args.drop_path_rate or 0.0)),
            trm_dropout=args.trm_dropout,
        )
    base_model = ov.model_fn(args.architecture, modality, num_classes, drop_path_rate=args.drop_path_rate,
                              fusion_type=(args.fusion_type if args.architecture == "efficient" else None),
                              classifier_dropout=args.classifier_dropout, trm_kwargs=trm_kwargs,
                              cfg_overrides=cfg_overrides)
    n_params = sum(p_.numel() for p_ in base_model.parameters())
    print(f"[architecture] {args.architecture!r} ({modality}): {n_params/1e6:.3f}M parameters", flush=True)

    if args.activation == "leakyrelu":
        replace_relu_with_leakyrelu(base_model, negative_slope=args.leaky_slope)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(base_model, strict=True)
    spectral_ckpt_on = (args.spectral_checkpointing == "on" or (
        args.spectral_checkpointing == "auto" and in_channels >= args.spectral_checkpoint_min_channels))
    if spectral_ckpt_on:
        enable_spectral_gradient_checkpointing(base_model, strict=True)
    if args.compile_model == "on":
        # v17 S3 - print the summary. `enable_torch_compile` swallows every failure and
        # returns "FAILED (...) - continuing uncompiled" or "unavailable (...)", and this
        # call discarded it, so a run that silently fell back to eager looked exactly like
        # a run where compilation did nothing. With compile now ON by default that is the
        # difference between a 266 ms step and a 437 ms one, unannounced.
        _compile_summary = enable_torch_compile(base_model, mode=args.compile_mode)
        print(f"[compile] {_compile_summary}", flush=True)
        if "FAILED" in _compile_summary or "unavailable" in _compile_summary:
            print("[compile] WARNING: running UNCOMPILED - expect ~1.64x the step time "
                  "measured for this configuration.", flush=True)

    # R-1/R-2/R-3 - reconstruction wiring, no forced override to "none".
    if args.recon_mode == "latent":
        model = GMedMambaLatentReconWrapperV2(base_model, in_channels=in_channels,
                                               out_activation=args.recon_out_activation,
                                               spectral_dropout=args.spectral_dropout)
    elif args.recon_mode == "raw_input":
        model = GMedMambaRawReconWrapper(base_model, in_channels=in_channels)
    else:
        model = base_model

    use_gan = (args.use_gan or args.lambda_gan > 0) and args.recon_mode != "none"
    discriminator = disc_optimizer = None
    if use_gan:
        discriminator = SpectralDiscriminator(in_channels=in_channels)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr, weight_decay=0.0)
        if args.lambda_gan == 0.0:
            args.lambda_gan = 0.05

    criterion = build_criterion(args.loss, y_train=y_train_arr, num_classes=num_classes,
                                 focal_gamma=args.focal_gamma, weight_method=args.weight_method,
                                 device=device, class_weight_power=args.class_weight_power)

    if args.weight_decay_groups:
        optimizer = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # E-2 - step-axis schedule.
    steps_per_epoch = max(1, -(-train_subsample_info["kept"] // args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=args.warmup_steps)
    print(f"[scheduler] step-axis warmup+cosine: total_steps={total_steps} "
          f"warmup_steps={scheduler.warmup_steps} interval={args.scheduler_interval}", flush=True)

    # ------------------------------------------------------------------
    # gates G1/G2/G9 (C-2) and G7
    # ------------------------------------------------------------------
    sensitivity_result = None
    if args.sensitivity_check == "on":
        x0, _y0 = next(iter(train_loader))[:2]
        sensitivity_result = run_sensitivity_check_with_margin(
            model.to(device), x0.to(device), min_logit_std=args.min_logit_std,
            min_stem_sensitivity=args.min_stem_sensitivity)
        if not sensitivity_result.get("passed"):
            fail_dir = exp_dir / "numerical_failure"
            fail_dir.mkdir(parents=True, exist_ok=True)
            with open(fail_dir / "sensitivity_failure.json", "w") as f:
                json.dump(sensitivity_result, f, indent=2, default=str)
            raise SystemExit(f"REPRESENTATION_COLLAPSE: {sensitivity_result.get('reason')}")
        print(f"[gates G1/G2/G9] margins: " +
              ", ".join(f"{k}={v['margin']:.2f}x" for k, v in sensitivity_result.get("margins", {}).items()),
              flush=True)

    g7_result = None
    run_g7 = (args.g7_check == "on" or (args.g7_check == "auto" and args.architecture == "recursive"
                                          and args.recon_mode == "latent"))
    if run_g7:
        from training.gan import SAMLoss
        x0, _y0 = next(iter(train_loader))[:2]
        g7_shim = _G7Shim()
        g7_result = check_reconstruction_gradient(
            # v17 S9: `ov.trainer_cls`, not the module global. Before S9 the derived entry
            # points REBOUND v16.TrainerG_v12, so this gate exercised the trainer the run
            # would actually use. No subclass overrides `_forward_with_recon` today, so
            # this resolves to the same function - but reading it off the module global
            # would silently stop tracking the run's trainer the moment one did.
            model.to(device), lambda m, x: ov.trainer_cls._forward_with_recon(g7_shim, m, x),
            x0.to(device), SAMLoss(), lambda_mse=args.lambda_mse, lambda_sam=args.lambda_sam)
        with open(exp_dir / "gate_g7_report.json", "w") as f:
            json.dump(g7_result, f, indent=2, default=str)
        if not g7_result.get("passed"):
            raise SystemExit(f"RECONSTRUCTION_DETACHED (gate G7): {g7_result.get('reason')}")
        print(f"[gate G7] PASS - reconstruction gradient reaches {g7_result['n_encoder_params_checked']} "
              f"recursive-core parameter(s)", flush=True)

    config = {
        "cli_args": vars(args), "classes": num_classes, "device": device, "class_names": class_names,
        "in_channels": in_channels, "architecture": args.architecture, "modality": modality,
        "run_name": exp_dir.name, "dataset_slug": dataset_slug(args.data_dir),
        "backbone_num_params": int(n_params),
        "dataloader_config": {"train": train_policy.as_dict(), "validation": val_policy.as_dict()},
        "amp_mode": args.amp, "torch_version": torch.__version__,
        # v17 S12 - this is the "later stage" the Stage 9 note below asked for.
        #
        # Stage 9 deliberately left both as literals, because its own gate was "config.json
        # is byte-identical to a pre-Stage-9 run" and that is only a safety net if the
        # refactor changes no output. It flagged the cost: "trainer" had always been the
        # literal "TrainerG_v12" even when the actual class was TrainerG_v13Optimal, so
        # every `optimal`/`recon` run in experiments/ under-reports the trainer that
        # produced it. S12 makes that worse if left alone - `--fast_loop on` changes the
        # class and nothing in the run would say so - so it is fixed here, where the
        # behaviour change is the point rather than a side effect.
        #
        # `entry_point` stays a literal: it is corrected by `trainerg_v13.ENTRY_POINT_OVERRIDE`
        # (see training/trainerg_v13.py:56) on the entry points that need it.
        "entry_point": "train_example_v16.py", "trainer": ov.trainer_cls.__name__,
        "train_subsample": train_subsample_info, "val_subsample": val_subsample_info,
        "seeding": seed_state, "sensitivity_check": sensitivity_result, "gate_g7": g7_result,
        "split_drift": {"passed": drift_report["passed"]},
        "scheduler": {"interval": args.scheduler_interval, "total_steps": total_steps,
                      "warmup_steps": int(scheduler.warmup_steps)},
        "system_snapshot_startup": startup_snapshot,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4, default=str)

    stability_cfg = StabilityConfig(
        max_gradient_norm=args.max_gradient_norm, max_gradient_skip_ratio=args.max_gradient_skip_ratio,
        max_consecutive_bad_batches=args.max_consecutive_bad_batches,
        max_consecutive_unhealthy_epochs=args.max_consecutive_unhealthy_epochs,
        debug_numerics=args.debug_numerics)

    trainer = ov.trainer_cls(
        model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
        scheduler=scheduler, device=device, exp_dir=exp_dir, class_names=class_names,
        lambda_mse=(args.lambda_mse if args.recon_mode != "none" else 0.0),
        lambda_sam=(args.lambda_sam if args.recon_mode != "none" else 0.0),
        lambda_gan=args.lambda_gan, discriminator=discriminator, disc_optimizer=disc_optimizer,
        config=config, wavelengths=wavelengths, checkpoint_metric=args.checkpoint_metric,
        class_collapse_streak=args.class_collapse_streak, criterion=criterion,
        early_stop_patience=args.early_stop_patience, grad_clip_norm=args.max_gradient_norm,
        stability_cfg=stability_cfg, amp_mode=args.amp, debug_numerics=args.debug_numerics,
        ema_rate=(args.trm_ema_rate if args.architecture == "recursive" else None),
        on_class_collapse=args.on_class_collapse, use_wavelengths=args.use_wavelengths,
        sensor_range=(tuple(args.sensor_range) if args.sensor_range else None),
        keep_last_n=args.keep_last_n, eval_artifact_stride=args.eval_artifact_stride,
        val_groups=val_groups, val_captures=val_captures, capture_index=capture_index,
        min_patient_recall=args.min_patient_recall, val_divergence_patience=args.val_divergence_patience,
        scheduler_interval=args.scheduler_interval,
    )
    trainer.dataloader_config = config["dataloader_config"]
    if augmented_train_ds is not None:
        trainer.augmented_train_dataset = augmented_train_ds

    log_system_memory(trainer.logger, label="pre-training")

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        start_epoch = trainer.load_checkpoint(args.resume) + 1

    stop_reason = trainer.fit(max_epochs=args.epochs, start_epoch=start_epoch)
    stop_str = str(getattr(stop_reason, "value", stop_reason)) if stop_reason is not None else ""

    gate_results = {"G7": (g7_result or {}).get("passed"), "G8": drift_report["passed"],
                    "G1_G2_G9": (sensitivity_result or {}).get("passed")}
    if args.verify_best_checkpoint:
        try:
            gate_results["G5"] = trainer.verify_best_checkpoint_reproduces()
        except Exception as e:
            print(f"WARNING: gate G5 check failed to run: {type(e).__name__}: {e}", flush=True)

    if args.eval_test != "off":

        def _run_test_eval(policy) -> bool:
            """Build the test loader with `policy` and evaluate. Factored out so the
            descriptor-exhaustion retry below runs the SAME evaluation, differing only in
            how the data is loaded - not a second, subtly different code path."""
            loader = _build_test_loader_v16(
                args, data_dir=args.data_dir, storage_mode=args.dataset_storage,
                normalization=args.normalization, global_mean=global_mean,
                global_std=global_std, policy=policy, device=device, val_x_path=x_te)
            groups = images_test_full if modality == "rgb" else groups_test_full
            return bool(trainer.evaluate_test_split(
                loader, num_classes=num_classes, test_groups=groups,
                eval_group_aggregation=args.eval_group_aggregation) is not None)

        try:
            gate_results["G6"] = _run_test_eval(val_policy)
        except Exception as e:
            import traceback
            # v17 - one retry, single-process, when the failure is descriptor exhaustion.
            # This runs at the END of a multi-hour job: a run that trained and validated
            # correctly should not forfeit its held-out number to an FD ceiling. A
            # 0-worker loader is slower and cannot hit the limit at all. Same weights,
            # same data, same evaluation - only the loading differs, so the resulting G6
            # is a real number and not a workaround artifact.
            if is_fd_exhaustion(e):
                print(f"WARNING: test-split evaluation ran out of file descriptors with "
                      f"{val_policy.num_workers} worker(s). Retrying single-process.", flush=True)
                try:
                    from dataclasses import replace as _replace
                    gate_results["G6"] = _run_test_eval(_replace(
                        val_policy, num_workers=0, persistent_workers=False,
                        prefetch_factor=None))
                    print("[test-eval] single-process retry succeeded.", flush=True)
                    e = None
                except Exception as e2:
                    e = e2
            if e is not None:                    # None == the retry above recovered it
                print(f"WARNING: test-split evaluation failed: {type(e).__name__}: {e}",
                      flush=True)
                # A bare AssertionError prints as an empty message, which is how the
                # frozen NpyDataset's normalization assert hid this for a whole run.
                traceback.print_exc()
                with open(exp_dir / "gate_g6_failure.json", "w") as f:
                    json.dump({"exception": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()}, f, indent=2, default=str)
                gate_results["G6"] = False

    with open(exp_dir / "gates.json", "w") as f:
        json.dump(gate_results, f, indent=2, default=str)

    if stop_str == "TRAINING_ABORTED_NUMERICAL_INSTABILITY":
        raise SystemExit(f"TRAINING_ABORTED_NUMERICAL_INSTABILITY: see {exp_dir}/numerical_failure/")
    if stop_str == "TRAINING_ABORTED_REPRESENTATION_COLLAPSE":
        raise SystemExit(f"TRAINING_ABORTED_REPRESENTATION_COLLAPSE: see {exp_dir}/collapse_failure/")


class _G7Shim:
    """`TrainerG_v12._forward_with_recon` is an instance method that reads
    `self._wl`/`self.sensor_range`; G7 runs before a trainer exists, so this
    supplies a minimal stand-in with those two attributes set to None
    (no wavelength plumbing needed for the preflight gate itself - it is
    exercising the SHAPE of the decode path, not the sensor-metadata path)."""
    _wl = None
    sensor_range = None

    def _accepts_wavelengths_ds(self, base):
        return False

    def _deep_supervision_forward(self, base, x):
        return base.forward_deep_supervision(x)

    def _model_forward(self, model, x):
        return model(x)


def main(overrides: Optional["Overrides"] = None):
    try:
        _main(overrides)
    except SystemExit:
        raise
    except BaseException as e:
        fc = classify_exception(e)
        try:
            exp_dir = _LAST_EXP_DIR.get("path")
            if exp_dir is not None:
                import datetime as _dt
                with open(Path(exp_dir) / "failure_class.json", "w") as f:
                    json.dump({"failure_class": fc.value, "exception": f"{type(e).__name__}: {e}",
                               "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat()},
                              f, indent=2, default=str)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
