# -*- coding: utf-8 -*-
"""
train_example_v14.py
======================
Thin copy of `train_example_v13.py` that adds `--architecture recursive`:
`gmedmamba.GMedMambaRecursive`, the TRM-style (Tiny Recursive Model,
arXiv:2510.04871) weight-shared recursive variant of G-MedMamba (~1-4M
params vs ~27M for `--architecture split`, see `gmedmamba.py`). Everything
else - dataset validation, leakage/coverage checks, augmentation, GAN/
reconstruction wiring, preflight, numerical-stability knobs - is byte-for-
byte identical to v13; `--architecture {split,fullchannel,efficient}` still
behave exactly as before (this file swaps in `training.trainerg_v10.TrainerG_v10`,
which is a no-op passthrough to `TrainerG_v9` for those three).

DEFAULTS CHANGED IN THIS REVISION (see documentations/07_parameters_v14.md
section 6). A bare-defaults `--architecture recursive` run over 740,800
11x11x3 patches went from ~92 h/epoch to ~26 min/epoch:

  --trm_mixer                   ss2d -> mlp       (39x; `ss2d` is a Python
                                                   loop over 121 timesteps
                                                   called 168x per step)
  --amp                         off  -> bf16      (3.8x; auto-downgrades on
                                                   a GPU without bf16)
  --batch_size                  128  -> 256
  --trm_deep_supervision_steps  4    -> 3         (1.27x)
  --loader_mode                 safe -> balanced  (`safe` means 0 workers)

Unchanged on purpose: --recon_mode latent, --loss ce, --sampler none,
--augment_preset none, --lr 1e-4, --dataset_storage auto. Those are
dataset-specific scientific choices, not throughput knobs - the right value
for one dataset is the wrong value for another - so they live in
documentations/08_commands.md as per-dataset commands instead.
`--safe_mode` still forces the old conservative bundle, and now genuinely
honours an explicit override (`--safe_mode --amp bf16` used to silently
give you fp32).

Also new: `--train_subsample_frac` (train on a random fraction of the split
each epoch; deterministic given --seed, recorded in config.json and in the
run directory name), and run directories are now named
`<ts>_<dataset>_<arch>_<modality>_<key>_bs<N>_<amp>` via
`training/run_naming.py` - timestamp first so `ls` still sorts
chronologically, but carrying the dataset and the settings that changed the
result. The old `gmedmamba_stageG_<arch>_run_<ts>` recorded neither, and v13
passed the identical exp_name, so v13 and v14 runs were indistinguishable.

New for `--architecture recursive`:
  --trm_dim / --trm_core_layers / --trm_n_latent / --trm_n_improve /
  --trm_deep_supervision_steps / --trm_mixer / --trm_ema_rate /
  --no_trm_halting   thread into `GMedMambaConfig`'s `trm_*` fields via
                      `training.config_presets.build_model(..., trm_kwargs=...)`.
  TrainerG_v10        runs in-loop deep supervision (all
                      `trm_deep_supervision_steps` segments per batch, one
                      backward) + an optional halting BCE loss + EMA
                      (`gmedmamba_ema.EMAHelper`, rate = `--trm_ema_rate`,
                      0 disables) - see `training/trainerg_v10.py`.

------------------------------------------------------------------------
Everything below this line is train_example_v13.py's original docstring,
describing the Stage G ("Stability") remediation this file still carries
forward unchanged for every architecture.
------------------------------------------------------------------------

GMedMamba v12/v13 "Definitive Training Stability & SIGBUS Remediation"
plan - Stage G ("Stability") entry point, on top of train_example_v12.py
(which carries everything from Stage A through F forward unchanged).

This file is the "recommended file changes / train_example_v12.py" item
from the plan's Phase 39, implemented as a NEW versioned entry point
(matching this repository's own convention: v7 added Stage A on v6, v9
added Stage C-adjacent choices on v8, etc.) rather than editing
train_example_v12.py in place, so anyone already depending on v12's exact
behaviour keeps it unmodified.

Adds:

  Phase 0/1/2 - explicit DataLoader memory policy via
                `training/dataloader_config.py`:
                  --loader_mode {safe,balanced,performance}  (v13 default:
                  safe; v14 default: balanced - see the note above)
                  --num_workers / --prefetch_factor / --persistent_workers /
                  --pin_memory  (each, if passed, overrides just that one
                  field of the selected --loader_mode)
                Train and validation get INDEPENDENT worker policies
                (Phase 2.4) - validation never gets more workers than train.
  Phase 3     - `training/system_memory.py`: RAM/`/dev/shm`/GPU snapshot
                logged once at startup (before any DataLoader is built) and
                once more right before training starts.
  Phase 4/5/6 - `training/npy_integrity.py`: every `.npy` file is validated
                BEFORE any Dataset/DataLoader/mmap is created. A validation
                failure raises `DatasetStorageError` (exits with
                `DATASET_STORAGE_ERROR`) - it is never caught and silently
                downgraded to "try mmap anyway".
                --dataset_storage {auto,mmap,ram}  (default: auto)
  Phase 7     - --loader_test: builds the dataset + DataLoader, reads a few
                batches, reports shape/dtype/timing/RAM/SHM, and exits
                WITHOUT constructing the GPU model - for isolating the
                first worker count that becomes unstable.
  Phase 8/9   - deterministic per-worker seeding (`make_worker_init_fn`)
                and an optional `--cpu_threads` cap to avoid CPU
                oversubscription across the DataLoader-worker fan-out
                introduced by --loader_mode performance.
  Phase 10    - --amp {off,bf16,fp16,auto}  (v13 default: off, per the
                plan's recommendation "while debugging persistent NaN/Inf
                gradients". v14 default: bf16 - that debugging is done, and
                --safe_mode still forces off.)
  Phase 11-23 - numerical-stability enforcement lives in
                `training/trainerg_v9.py` (TrainerG_v9). This file exposes
                its knobs on the CLI:
                  --max_gradient_norm / --max_gradient_skip_ratio /
                  --max_consecutive_bad_batches /
                  --max_consecutive_unhealthy_epochs / --debug_numerics
  Phase 29    - a preflight block prints dataset/DataLoader/system/GPU/
                numerical configuration before training starts.
  Phase 30    - --safe_mode: one flag that forces the conservative
                combination the plan calls the "primary user-facing
                solution" (loader_mode=safe, amp=off, gradient
                checkpointing on, dataset validation on, debug_numerics on).
  Phase 38    - every one of the above is recorded in config.json.

v13.1 remediation (the "GMedMamba v13 - Plan Review" plan) - after a run
where a byte-truncated `X_train.npy` slipped past both integrity checks and
only crashed later inside `np.load`:

  W1  - `validate_npy_file` now does a DETERMINISTIC header-offset + file-size
        check (catches a truncated-but-header-intact file with no mmap / no
        full load) and `NpyFileReport.ok` finally reflects the sample probes.
        Each file gets an explicit `status` (VALID/TRUNCATED/HEADER_INVALID/
        UNREADABLE/MISSING/EMPTY).
  W2  - dataset validation is the FIRST gate: discover -> validate_dataset ->
        leakage check (no longer the other way round). The leakage check's
        `except Exception ... "continuing."` no longer swallows a storage/read
        error - only a genuinely-missing optional prerequisite is a warning.
  W3  - on `DatasetStorageError`, `<exp_dir>/dataset_failure/{failure.json,
        validation_report.json,system_memory.json}` is written.
  W4  - `--numerical_smoke_test {auto,on,off}` runs one forward/backward/step
        cycle on a throwaway model copy before training; a non-finite stage
        aborts with `NUMERICAL_SMOKE_TEST_FAILED`.
  W5  - `training/failure_taxonomy.py` maps every terminal error to one
        `FailureClass`; `<exp_dir>/failure_class.json` records it.
  W6  - `training/npy_atomic.py` (atomic `.npy` writes + `dataset_manifest.json`);
        `--manifest_check {auto,off,strict}` cross-checks it before training.

Everything else (architecture choices, augmentation, class-imbalance
handling, dataset integrity/leakage checks, early stopping, ...) is
unchanged from train_example_v12.py.

Usage (first post-fix qualification run, per Phase 42):

    python train_example_v13.py --data_dir ./data/hsi --epochs 3 \\
        --batch_size 32 --safe_mode --recon_mode latent --lambda_gan 0 \\
        --gradient_checkpointing

    # isolate the first unstable worker count:
    python train_example_v13.py --data_dir ./data/hsi --loader_test --num_workers 0
    python train_example_v13.py --data_dir ./data/hsi --loader_test --num_workers 1
    python train_example_v13.py --data_dir ./data/hsi --loader_test --num_workers 2

    python train_example_v13.py --data_dir ./data_pad_v4-v4/ --loader_test --num_workers 0
        
    ./data_pad_v4-v4/

    # once stable, progressively reintroduce performance features:
    python train_example_v13.py --data_dir ./data/hsi --epochs 40 \\
        --loader_mode balanced --amp bf16 --architecture efficient

    # TRM recursive variant (this file), ~0.44M params instead of ~27M.
    # mlp / bf16 / bs256 / sup3 are now DEFAULTS, so the short form IS the
    # fast form (~26 min/epoch on data/pad_v6; it was ~92 h before):
    python train_example_v14.py --data_dir ./data/pad_v6/ --epochs 50 \\
        --architecture recursive

    # ...and the form tuned for that dataset's 16.3:1 class imbalance
    # (documentations/08_commands.md section A):
    python train_example_v14.py --data_dir ./data/pad_v6/ --epochs 50 \\
        --architecture recursive --recon_mode none \\
        --loss ce --sampler moderate_oversample --checkpoint_metric f1_macro \\
        --augment_preset custom --flip_h_prob 0.5 --flip_v_prob 0.5 --rotate90_prob 0.5 \\
        --loader_mode performance --dataset_storage ram --early_stop_patience 8

    # HSI: 2.6M patches / 46 GB, so subsample and NEVER --dataset_storage ram
    # (section B, which also documents that this dataset has no val split):
    python train_example_v14.py --data_dir ./data/hsi_v7/hsi/ --epochs 50 \\
        --architecture recursive --recon_mode none --train_subsample_frac 0.05 \\
        --loss weighted_ce --loader_mode performance --dataset_storage auto

Full flag reference:  documentations/07_parameters_v14.md
Verified commands:    documentations/08_commands.md
"""

import argparse
import errno
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_example_v6 import NpyDataset, discover_data, setup_experiment_dir
from train_example_v7 import GMedMambaRawReconWrapper
from training.trainerg_v10 import TrainerG_v10
from training.numerical_stability import StabilityConfig
from training.gan import SpectralDiscriminator
from training.class_coverage import verify_class_coverage
from training.class_imbalance import write_imbalance_report
from training.activation_patch import replace_relu_with_leakyrelu
from training.reconstruction_head import GMedMambaLatentReconWrapper
from training.losses import build_criterion
from training.samplers import build_balanced_sampler, build_moderate_oversample_indices
from training.augmentation import AugmentationConfig, PRESETS, AugmentedPatchDataset
from training.config_presets import build_model
from training.grad_checkpoint import enable_gradient_checkpointing
from training.spectral_checkpoint import enable_spectral_gradient_checkpointing
from training.dataloader_config import train_val_policies, apply_main_process_thread_limits
from training.system_memory import (
    collect_system_memory_snapshot, format_system_memory_snapshot, log_system_memory,
    write_system_memory_report,
)
from training.npy_integrity import (
    validate_dataset, write_integrity_report, write_dataset_failure_artifacts, DatasetStorageError,
)
from training.npy_atomic import verify_against_manifest, MANIFEST_NAME
from training.failure_taxonomy import FailureClass, classify_exception
from training.dataset_integrity import run_full_integrity_check, DatasetLeakageError
from training.run_naming import build_run_name, dataset_slug, infer_modality

EFFICIENT_FUSION_CHOICES = ["film", "gated", "cross_attention", "multiplicative", "residual",
                            "se_gate", "eca", "none"]


def build_arg_parser():
    p = argparse.ArgumentParser(description="GMedMamba Stage-G ('Stability') training entry point")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256,
                    help="256 is the measured throughput sweet spot on 11x11 patches: 128-150 "
                         "leaves the GPU launch-bound, while 512 stops paying off because "
                         "SpectralPathway's fixed chunk_size=1024 loop grows linearly with the "
                         "batch (see documentations/PARAMETERS.md).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--disc_lr", type=float, default=1e-4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_sam", type=float, default=0.1)
    p.add_argument("--lambda_gan", type=float, default=0.0)
    p.add_argument("--use_gan", action="store_true")

    p.add_argument("--check_leakage", action="store_true", default=True)
    p.add_argument("--no_check_leakage", dest="check_leakage", action="store_false")
    p.add_argument("--abort_on_leakage", action="store_true", default=True)
    p.add_argument("--allow_leakage", dest="abort_on_leakage", action="store_false")

    p.add_argument("--check_class_coverage", action="store_true", default=True)
    p.add_argument("--no_check_class_coverage", dest="check_class_coverage", action="store_false")
    p.add_argument("--allow_missing_classes", action="store_true")

    p.add_argument("--checkpoint_metric", default="f1_macro")
    p.add_argument("--class_collapse_streak", type=int, default=3)

    p.add_argument("--activation", choices=["relu", "leakyrelu"], default="leakyrelu")
    p.add_argument("--leaky_slope", type=float, default=0.01)

    p.add_argument("--recon_mode", choices=["latent", "raw_input", "none"], default="latent")
    p.add_argument("--spectral_dropout", type=float, default=0.0)

    p.add_argument("--gradient_checkpointing", action="store_true",
                    help="see train_example_v12.py's docstring - addresses GPU activation memory, "
                         "NOT DataLoader /dev/shm (Phase 25 of the stability plan: these are "
                         "independent fixes).")

    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal", "focal_weighted"], default="ce")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--weight_method", choices=["balanced", "inverse"], default="balanced")
    p.add_argument("--sampler", choices=["none", "balanced", "moderate_oversample"], default="none")
    p.add_argument("--oversample_target_percentile", type=float, default=50.0)
    p.add_argument("--train_subsample_frac", type=float, default=1.0,
                    help="train on a random FRACTION of the training split each run (1.0 = "
                         "default, the whole split). Applied after --sampler, so it composes "
                         "with moderate_oversample/balanced. Deterministic given --seed. "
                         "This exists because an 'epoch' over these datasets is far more "
                         "redundant than it looks: data/pad_v6 is 740,800 patches tiled 674-per-"
                         "image from only 1,099 images, and data/hsi_v7/hsi is 2,594,592 patches "
                         "(10,135 batches at bs=256). A fractional epoch is a legitimate "
                         "experimental choice here, not a shortcut - but report it: it is "
                         "recorded in config.json and in the run directory name.")

    p.add_argument("--architecture", choices=["split", "fullchannel", "efficient", "recursive"],
                    default="split")
    p.add_argument("--fusion_type", choices=EFFICIENT_FUSION_CHOICES, default="se_gate")

    # TRM recursive variant (--architecture recursive only; see gmedmamba.GMedMambaConfig's
    # trm_* fields and training/trainerg_v10.py).
    p.add_argument("--trm_dim", type=int, default=128)
    p.add_argument("--trm_core_layers", type=int, default=2)
    p.add_argument("--trm_n_latent", type=int, default=6)
    p.add_argument("--trm_n_improve", type=int, default=3)
    p.add_argument("--trm_deep_supervision_steps", type=int, default=3,
                    help="ACT segments per batch. Each segment costs a full (trm_n_latent+1) x "
                         "trm_n_improve pass over the shared core, so this multiplies the whole "
                         "step cost: measured 502 ms/step at 3 vs 635 ms at 4 (batch 256, mlp, "
                         "bf16). 3 keeps three deep-supervision signals.")
    p.add_argument("--trm_mixer", choices=["ss2d", "mlp", "attention"], default="mlp",
                    help="token mixer inside the shared recursive core. 'mlp' (default) is a "
                         "depthwise-conv + GEGLU mixer. 'ss2d' routes through "
                         "_selective_scan_pure_pytorch, a Python loop over all Hp*Wp timesteps "
                         "that the core calls 168x per training step - MEASURED 39x SLOWER "
                         "end-to-end (66.9 s/step vs 1.7 s/step at batch 150). Only choose "
                         "'ss2d' for a deliberate accuracy ablation, or after registering a real "
                         "CUDA/Triton kernel via gmedmamba.register_scan_backend().")
    p.add_argument("--trm_ema_rate", type=float, default=0.999)
    p.add_argument("--trm_halting", action="store_true", default=True)
    p.add_argument("--no_trm_halting", dest="trm_halting", action="store_false")

    p.add_argument("--augment_preset", choices=["none", "light", "medium", "custom"], default="none")
    p.add_argument("--spectral_noise_std", type=float, default=0.0)
    p.add_argument("--spectral_scale_range", type=float, default=0.0)
    p.add_argument("--spectral_offset_std", type=float, default=0.0)
    p.add_argument("--band_dropout_prob", type=float, default=0.0)
    p.add_argument("--band_dropout_max_frac", type=float, default=0.1)
    p.add_argument("--spectral_mask_prob", type=float, default=0.0)
    p.add_argument("--spectral_mask_max_width_frac", type=float, default=0.1)
    p.add_argument("--flip_h_prob", type=float, default=0.0)
    p.add_argument("--flip_v_prob", type=float, default=0.0)
    p.add_argument("--rotate90_prob", type=float, default=0.0)
    p.add_argument("--crop_scale_max_frac", type=float, default=0.0)

    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--drop_path_rate", type=float, default=None)
    p.add_argument("--classifier_dropout", type=float, default=None)

    p.add_argument("--early_stop_patience", type=int, default=None)

    # ------------------------------------------------------------------
    # Phase 0/1/2 - explicit DataLoader memory policy
    # ------------------------------------------------------------------
    p.add_argument("--loader_mode", choices=["safe", "balanced", "performance"], default="balanced",
                    help="Phase 1.1. 'balanced' (default): 2 workers, prefetch=1, no persistence/"
                         "pin_memory. 'safe': num_workers=0 - removes /dev/shm from the critical "
                         "path entirely; use it to reproduce a suspected worker SIGBUS. "
                         "'performance': 4 workers, prefetch=2, persistent + pinned - opt-in "
                         "only, never the default (Phase 2). NOTE any of --num_workers/"
                         "--prefetch_factor/--persistent_workers/--pin_memory overrides just "
                         "THAT ONE field of the selected mode; the rest keep the mode's values.")
    p.add_argument("--num_workers", type=int, default=None,
                    help="overrides --loader_mode's worker count.")
    p.add_argument("--prefetch_factor", type=int, default=None,
                    help="overrides --loader_mode's prefetch factor; ignored if num_workers==0.")
    p.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=None,
                    help="overrides --loader_mode's persistence (on).")
    p.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false",
                    help="overrides --loader_mode's persistence (off).")
    p.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=None,
                    help="overrides --loader_mode's pin_memory (on).")
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false",
                    help="overrides --loader_mode's pin_memory (off).")
    p.add_argument("--cpu_threads", type=int, default=None,
                    help="Phase 9 - caps OMP/MKL/torch thread counts (main process; workers get "
                         "their own cap via the worker_init_fn) to avoid CPU oversubscription "
                         "when multiple DataLoader workers are enabled.")

    # ------------------------------------------------------------------
    # Phase 4/5/6 - dataset integrity + storage mode
    # ------------------------------------------------------------------
    p.add_argument("--dataset_storage", choices=["auto", "mmap", "ram"], default="auto",
                    help="Phase 5. 'auto' (default): validated mmap, or fully-materialized RAM "
                         "loading if the file comfortably fits in available RAM. 'ram' is the "
                         "recommended debugging mode for a suspected mmap-related main-process "
                         "SIGBUS (Phase 6).")
    p.add_argument("--skip_dataset_validation", action="store_true",
                    help="skip the Phase 4 .npy integrity validation pass. NOT recommended - "
                         "this is exactly the check that turns a guaranteed main-process SIGBUS "
                         "into a clean DATASET_STORAGE_ERROR before training starts. A run with "
                         "this flag is recorded as scientifically_qualified=false in config.json.")
    p.add_argument("--dataset_validation_level", choices=["structural", "deep"], default="structural",
                    help="v13.1 remediation. 'structural' (default): header + on-disk size + a "
                         "SIGBUS-free seek()/read() probe of ~19 rows per file. 'deep': also read "
                         "every X_*.npy end-to-end - catches a physically-unreadable region "
                         "(OSError(5)/EIO: bad sectors / failing disk / fs corruption) that the "
                         "row probes happen to miss, at the cost of a full sequential read per "
                         "file at startup.")
    p.add_argument("--manifest_check", choices=["auto", "off", "strict"], default="auto",
                    help="Workstream 6. 'auto' (default): if <data_dir>/dataset_manifest.json "
                         "exists, cross-check every file's shape/dtype/size against it and abort "
                         "on mismatch; skip silently if absent. 'strict': also require the "
                         "manifest to exist. 'off': never check.")

    # ------------------------------------------------------------------
    # Phase 7 - loader isolation test
    # ------------------------------------------------------------------
    p.add_argument("--loader_test", action="store_true",
                    help="build the dataset+DataLoader, read a few batches, report, and exit "
                         "WITHOUT building the GPU model (Phase 7).")
    p.add_argument("--loader_test_batches", type=int, default=5)

    # ------------------------------------------------------------------
    # Phase 10 - explicit AMP
    # ------------------------------------------------------------------
    p.add_argument("--amp", choices=["off", "bf16", "fp16", "auto"], default="bf16",
                    help="Phase 10. 'bf16' (default) is ~3.8x faster than 'off' (measured 363 vs "
                         "1385 ms/step) and, unlike fp16, has fp32 exponent range so it does not "
                         "need loss scaling to stay finite; it is silently downgraded to 'off' on "
                         "a GPU without bf16 support. Use 'off' when debugging non-finite "
                         "losses/gradients (this was the v13 remediation default, and --safe_mode "
                         "still forces it). 'auto' reproduces the old bf16-if-supported-else-fp16 "
                         "implicit behaviour. NOTE the validation pass currently ignores this and "
                         "always uses bf16-if-supported (training/trainerg_v9.py:255-263).")

    # ------------------------------------------------------------------
    # Phase 11-23 - numerical stability controller knobs
    # ------------------------------------------------------------------
    p.add_argument("--max_gradient_norm", type=float, default=1.0)
    p.add_argument("--max_gradient_skip_ratio", type=float, default=0.10)
    p.add_argument("--max_consecutive_bad_batches", type=int, default=3)
    p.add_argument("--max_consecutive_unhealthy_epochs", type=int, default=2)
    p.add_argument("--debug_numerics", action="store_true",
                    help="Phase 13 (partial)/14 - verbose per-batch NONFINITE_LOSS/"
                         "NONFINITE_GRADIENT diagnostics.")
    p.add_argument("--numerical_smoke_test", choices=["auto", "on", "off"], default="auto",
                    help="Workstream 4. Run one forward/backward/optimizer-step cycle on a "
                         "throwaway copy of the model before training starts and abort with "
                         "NUMERICAL_SMOKE_TEST_FAILED if any stage is non-finite. 'auto' (default) "
                         "= on when --safe_mode or --debug_numerics is set, else off.")

    # ------------------------------------------------------------------
    # Phase 30 - one-flag safe mode
    # ------------------------------------------------------------------
    p.add_argument("--safe_mode", action="store_true",
                    help="Phase 30. Forces loader_mode=safe, amp=off, gradient_checkpointing=on, "
                         "dataset validation on, debug_numerics on - the conservative combination "
                         "recommended for the first post-fix qualification run (Phase 42).")

    return p


def explicitly_passed(argv=None) -> set:
    """The set of `dest` names actually present on the command line.

    Uses the standard `argparse.SUPPRESS` two-pass idiom: a throwaway copy of
    the parser with every action's default suppressed produces a namespace
    containing ONLY what the user typed. Needed because `apply_safe_mode`
    below must distinguish "left at its default" from "explicitly set to the
    same value as the default".
    """
    p = build_arg_parser()
    for action in p._actions:
        action.default = argparse.SUPPRESS
    known, _ = p.parse_known_args(argv)
    return set(vars(known))


def apply_safe_mode(args: argparse.Namespace, explicit: set = None) -> None:
    """Phase 30 - mutates `args` in place. Explicit CLI overrides for any of
    these fields still win: `explicit` is the set of dest names the user
    actually typed (see `explicitly_passed`), and every field below is only
    touched when it is NOT in that set.

    This used to be a lie. The docstring made exactly this claim while the
    code unconditionally clobbered all ten fields, so `--safe_mode --amp bf16`
    silently trained in fp32. That matters more now that five parser defaults
    (amp, loader_mode, batch_size, trm_mixer, trm_deep_supervision_steps) are
    tuned for throughput and `--safe_mode` deliberately reverts two of them:
    overriding one back has to actually work.
    """
    if not args.safe_mode:
        return
    explicit = set() if explicit is None else explicit

    def _set(name, value):
        if name not in explicit:
            setattr(args, name, value)

    _set("loader_mode", "safe")
    _set("amp", "off")
    _set("gradient_checkpointing", True)
    # safe mode NEVER continues past a bad dataset - this one is not overridable,
    # because "--safe_mode --skip_dataset_validation" is a contradiction and the
    # whole point of the flag is that a corrupt file fails cleanly.
    args.skip_dataset_validation = False
    # NOTE: safe_mode does NOT force --dataset_validation_level deep - the deep
    # full-file read stays opt-in (it adds a large sequential read at startup).
    _set("debug_numerics", True)
    if args.numerical_smoke_test == "auto":
        args.numerical_smoke_test = "on"
    # Worker policy: `safe` means 0 workers / no /dev/shm on the critical path.
    # Still overridable one field at a time, e.g. `--safe_mode --num_workers 2`
    # to test whether 2 workers is where a SIGBUS starts.
    _set("num_workers", 0)
    _set("prefetch_factor", None)
    _set("persistent_workers", False)
    _set("pin_memory", False)
    if args.max_gradient_norm is None:
        args.max_gradient_norm = 1.0


# ============================================================================
# Phase 29 - preflight
# ============================================================================

def print_preflight(args, train_policy, val_policy, dataset_paths, system_snapshot, in_channels=None,
                     modality=None):
    lines = ["===== GMedMamba Stability Preflight =====", "Dataset:"]
    for k, v in dataset_paths.items():
        lines.append(f"  {k}: {v}")
    if in_channels is not None:
        lines.append(f"  channels: {in_channels}  modality: {modality}")
    lines.append(f"  storage_mode: {args.dataset_storage}")
    lines.append("")
    lines.append("DataLoader (train):")
    for k, v in train_policy.as_dict().items():
        lines.append(f"  {k}: {v}")
    lines.append("DataLoader (validation):")
    for k, v in val_policy.as_dict().items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("System:")
    ram, shm, gpu = system_snapshot["ram"], system_snapshot["shm"], system_snapshot["gpu"]
    lines.append(f"  RAM available: {ram.get('available_mb', 'n/a')} MB")
    lines.append(f"  /dev/shm available: {shm.get('free_mb', 'n/a') if shm.get('available') else 'n/a'} MB")
    if gpu.get("available"):
        lines.append(f"  GPU: {gpu.get('device_name')}  VRAM total: {gpu.get('total_mb')} MB  "
                      f"reserved: {gpu.get('reserved_mb')} MB")
    else:
        lines.append("  GPU: not available (CPU run)")
    lines.append("")
    lines.append("Numerical:")
    lines.append(f"  AMP: {args.amp}")
    lines.append(f"  gradient clipping: {args.max_gradient_norm}")
    lines.append(f"  max_gradient_skip_ratio: {args.max_gradient_skip_ratio}")
    lines.append(f"  max_consecutive_bad_batches: {args.max_consecutive_bad_batches}")
    lines.append(f"  SAM: {'ON' if args.lambda_sam > 0 and args.recon_mode != 'none' else 'off'}")
    lines.append(f"  GAN: {'ON' if (args.use_gan or args.lambda_gan > 0) and args.recon_mode != 'none' else 'off'}")
    lines.append(f"  gradient_checkpointing: {args.gradient_checkpointing}")
    lines.append("==========================================")
    print("\n".join(lines), flush=True)


# ============================================================================
# Phase 7 - loader isolation test
# ============================================================================

def run_loader_test(args, train_loader, train_policy) -> None:
    print(f"[loader-test] policy={train_policy.as_dict()}", flush=True)
    t0 = time.time()
    n_read = 0
    for i, (x, y) in enumerate(train_loader):
        if i >= args.loader_test_batches:
            break
        print(f"  batch {i}: x.shape={tuple(x.shape)} x.dtype={x.dtype} y.shape={tuple(y.shape)}", flush=True)
        n_read += 1
    elapsed = time.time() - t0
    snapshot = collect_system_memory_snapshot()
    print(f"[loader-test] read {n_read} batch(es) in {elapsed:.2f}s", flush=True)
    print(format_system_memory_snapshot(snapshot), flush=True)
    print("[loader-test] PASSED (no worker crash) - exiting without building the GPU model.", flush=True)


# Set by _main() right after the experiment dir is created, so the top-level
# main() wrapper can drop a failure_class.json next to the other artifacts.
_LAST_EXP_DIR = {"path": None}


def run_numerical_smoke_test(args, model, criterion, discriminator, use_gan, train_loader,
                              device, exp_dir):
    """Workstream 4 - one forward/backward/optimizer-step cycle on a THROWAWAY
    deep copy of the model, mirroring TrainerG_v9._train_one_epoch's loss
    assembly. Raises SystemExit("NUMERICAL_SMOKE_TEST_FAILED: ...") on any
    non-finite stage. Returns the run_smoke_test report dict."""
    import copy
    import torch.nn.functional as F
    from training.numerical_stability import run_smoke_test
    from training.gan import SAMLoss, generator_adversarial_loss

    try:
        x, y = next(iter(train_loader))
    except StopIteration:
        print("[smoke-test] train_loader yielded no batch - skipping.", flush=True)
        return {"passed": True, "skipped": True}

    x = x.to(device)
    y = y.to(device)
    sam_loss_fn = SAMLoss()
    lambda_mse = args.lambda_mse if args.recon_mode != "none" else 0.0
    lambda_sam = args.lambda_sam if args.recon_mode != "none" else 0.0
    lambda_gan = args.lambda_gan

    smoke_model = copy.deepcopy(model).to(device)
    smoke_opt = torch.optim.AdamW(smoke_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward_and_loss_fn(m, xb, yb):
        out = m(xb)
        logits, x_recon = out if isinstance(out, tuple) else (out, None)
        cls_loss = criterion(logits, yb)
        if x_recon is not None:
            mse_loss = F.mse_loss(x_recon, xb)
            sam_loss = sam_loss_fn(xb, x_recon)
        else:
            mse_loss = torch.zeros((), device=xb.device)
            sam_loss = torch.zeros((), device=xb.device)
        gan_loss = torch.zeros((), device=xb.device)
        if use_gan and x_recon is not None:
            gan_loss = generator_adversarial_loss(discriminator.to(device), x_recon)
        total = cls_loss + lambda_mse * mse_loss + lambda_sam * sam_loss + lambda_gan * gan_loss
        return total, {
            "classification": cls_loss,
            "MSE": mse_loss if x_recon is not None else None,
            "SAM": sam_loss if x_recon is not None else None,
            "GAN": gan_loss if (use_gan and x_recon is not None) else None,
            "total": total,
        }

    res = run_smoke_test(smoke_model, x, y, forward_and_loss_fn, smoke_opt,
                          max_gradient_norm=(args.max_gradient_norm or 1.0))
    del smoke_model, smoke_opt
    if device == "cuda":
        torch.cuda.empty_cache()

    if not res.get("passed"):
        fail_dir = exp_dir / "numerical_failure"
        fail_dir.mkdir(parents=True, exist_ok=True)
        with open(fail_dir / "smoke_failure.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        raise SystemExit(f"NUMERICAL_SMOKE_TEST_FAILED: stage={res.get('stage')} "
                          f"detail={res.get('detail')} (see {fail_dir}/smoke_failure.json)")
    gn = res.get("grad_norm")
    print(f"[smoke-test] PASSED (grad_norm={gn:.4g})" if isinstance(gn, float)
          else "[smoke-test] PASSED", flush=True)
    return res


def _run_leakage_check(args, exp_dir):
    """Phase 7 leakage check. Runs AFTER Phase-4 structural validation, so a
    truncated/corrupt file is already a fatal DATASET_STORAGE_ERROR by now.
    Only a genuinely-missing *optional prerequisite* (no manifest, module
    import failure) is downgraded to a warning - a read/storage error is
    re-raised, never swallowed as 'continuing'."""
    if not args.check_leakage:
        return
    manifest_path = None
    for candidate in (Path(args.data_dir) / "_progress.json",
                       Path(args.data_dir).parent / "_progress.json"):
        if candidate.is_file():
            manifest_path = str(candidate)
            break
    try:
        run_full_integrity_check(out_dir=str(exp_dir), manifest_path=manifest_path,
                                  npy_data_dir=args.data_dir, abort_on_leakage=args.abort_on_leakage)
    except DatasetLeakageError:
        raise
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARNING: leakage check skipped (optional prerequisite missing: {e!r})", flush=True)
    except OSError as e:
        # A physically-unreadable .npy (OSError(5)/EIO) surfacing here instead
        # of in the structural gate: intermittent bad-sector reads. Convert to
        # the same typed, classified failure the dataset gate produces.
        eio = getattr(e, "errno", None) == errno.EIO or "input/output error" in str(e).lower()
        msg = (f"PHYSICALLY UNREADABLE (OSError(5)/EIO) during leakage check: {e}" if eio
               else f"read error during leakage check: {e}")
        exc = DatasetStorageError(f"DATASET_STORAGE_ERROR: {msg}")
        write_dataset_failure_artifacts(str(exp_dir), None, exc,
                                         system_snapshot=collect_system_memory_snapshot())
        raise exc from e
    # ValueError / DatasetStorageError / MemoryError still PROPAGATE -
    # a truncated or unreadable .npy is fatal, never "continuing".


def _check_manifest(args, exp_dir):
    """Workstream 6 - cross-check <data_dir>/dataset_manifest.json if present."""
    if args.manifest_check == "off":
        return
    manifest_exists = (Path(args.data_dir) / MANIFEST_NAME).is_file()
    if not manifest_exists:
        if args.manifest_check == "strict":
            raise DatasetStorageError(
                f"DATASET_STORAGE_ERROR: --manifest_check strict but no {MANIFEST_NAME} in {args.data_dir}")
        print(f"[dataset-manifest] no {MANIFEST_NAME} (dataset predates manifest support) - "
              f"structural validation only.", flush=True)
        return
    problems = verify_against_manifest(args.data_dir)
    if problems:
        raise DatasetStorageError(
            "DATASET_STORAGE_ERROR: dataset does not match its manifest:\n  " + "\n  ".join(problems))
    print(f"[dataset-manifest] all files match {MANIFEST_NAME}.", flush=True)


def _main():
    p = build_arg_parser()
    args = p.parse_args()
    apply_safe_mode(args, explicitly_passed())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    apply_main_process_thread_limits(args.cpu_threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # --amp bf16 is now the parser default, so a card without bf16 support
        # must degrade rather than fail: _resolve_amp (training/trainerg_v9.py:81)
        # honours an explicit "bf16" without checking. Pre-Ampere users get a
        # working fp32 run and a warning instead of a crash mid-epoch.
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            print("WARNING: --amp bf16 requested but this GPU does not support bfloat16 - "
                  "falling back to --amp off (fp32). Pass --amp fp16 to use half precision "
                  "instead; note fp16 needs the GradScaler and can produce non-finite "
                  "gradients this bf16 default was chosen to avoid.", flush=True)
            args.amp = "off"

    # Phase 3 - system memory snapshot, BEFORE any DataLoader worker exists.
    startup_snapshot = log_system_memory(label="startup")

    # Run directory name (training/run_naming.py). Timestamp-first, so a plain
    # `ls experiments/` still sorts chronologically, followed by the dataset and
    # the settings that actually change the result - the old
    # `gmedmamba_stageG_{arch}_run_{ts}` recorded neither, and v13 passed the
    # very same exp_name, so v13 and v14 runs were indistinguishable.
    #
    # This runs BEFORE discover_data() on purpose: exp_dir has to exist early
    # enough for write_dataset_failure_artifacts() and failure_class.json to
    # have somewhere to land when the dataset gate itself fails. So the modality
    # is inferred here from the same one-line rule discover_data uses, and
    # asserted against the real thing once discover_data has run (below).
    run_modality = infer_modality(args.data_dir)
    run_name = build_run_name(args, modality=run_modality)
    exp_dir = setup_experiment_dir(exp_name=run_name, resume_path=args.resume,
                                    append_timestamp=False)
    _LAST_EXP_DIR["path"] = exp_dir
    write_system_memory_report(str(exp_dir / "system_memory_startup.json"), label="startup")

    # ------------------------------------------------------------------
    # STRICT DATASET VALIDATION IS THE FIRST GATE (v13 plan, Workstream 2a).
    # discover_data returns paths and only touches the (small) label
    # arrays; nothing loads X until it has passed validate_dataset.
    # ------------------------------------------------------------------
    print(f"Discovering dataset files in {args.data_dir}...", flush=True)
    try:
        x_tr, y_tr, x_te, y_te, wavelengths, num_classes, class_names, n_train, n_val = \
            discover_data(args.data_dir)
    except FileNotFoundError as e:
        dd = Path(args.data_dir)
        shards = [p for p in ([dd / "_batches"] + list(dd.glob("*_batches"))) if p.is_dir()]
        if shards:
            raise FileNotFoundError(
                f"{e}. Found a per-batch shard dir ({shards[0]}) but no merged X_train.npy/"
                f"y_train.npy - preprocessing wrote shards and never ran the finalize/merge "
                f"step. Re-run the dataset merge before training.") from e
        raise
    print(f"Dataset Loaded: {n_train} Train, {n_val} Val | Classes: {num_classes} | Device: {device}", flush=True)

    # The run directory name was built before this point from run_naming.infer_modality's
    # copy of discover_data's rule. If the two ever diverge the directory name is a lie,
    # so fail loudly rather than mislabel every artifact in it.
    _real_modality = "hsi" if wavelengths is not None else "rgb"
    if _real_modality != run_modality:
        raise SystemExit(
            f"INTERNAL: modality mismatch - run directory was named {run_modality!r} but "
            f"discover_data resolved {_real_modality!r}. training.run_naming.infer_modality "
            f"has drifted from train_example_v6.discover_data; fix run_naming.py.")

    dataset_paths = {"X_train": x_tr, "y_train": y_tr, "X_val": x_te, "y_val": y_te}
    integrity_report = None
    try:
        _check_manifest(args, exp_dir)   # runs even when --skip_dataset_validation
        if not args.skip_dataset_validation:
            integrity_report = validate_dataset(dataset_paths, abort_on_failure=True,
                                                 level=args.dataset_validation_level)
            write_integrity_report(integrity_report, str(exp_dir / "dataset_integrity_report.json"))
            print(f"[dataset-integrity] all .npy files passed Phase-4 validation "
                  f"(level={args.dataset_validation_level}).", flush=True)
    except DatasetStorageError as e:
        print(f"DATASET_STORAGE_ERROR: {e}", flush=True)
        write_dataset_failure_artifacts(str(exp_dir), integrity_report, e,
                                         system_snapshot=collect_system_memory_snapshot())
        raise
    if args.skip_dataset_validation:
        print("WARNING:\n"
              "DATASET VALIDATION DISABLED (--skip_dataset_validation).\n"
              "THIS RUN CANNOT BE MARKED SCIENTIFICALLY QUALIFIED. A corrupt/truncated file "
              "can now cause a genuine main-process SIGBUS instead of a clean "
              "DATASET_STORAGE_ERROR.", flush=True)

    # Phase 7 leakage check - now on known-good files.
    _run_leakage_check(args, exp_dir)

    y_train_arr = np.asarray(np.load(y_tr, mmap_mode="r"))
    y_val_arr = np.asarray(np.load(y_te, mmap_mode="r"))

    if args.check_class_coverage:
        try:
            report = verify_class_coverage(
                {"train": y_train_arr, "validation": y_val_arr}, num_classes=num_classes,
                class_names=class_names, fail_hard=not args.allow_missing_classes)
        except ValueError as e:
            # Not a bug - a split genuinely has zero samples of some class.
            # Give it the same typed/artifact treatment the dataset gate has
            # so failure_class.json is CLASS_COVERAGE_INCOMPLETE, not UNKNOWN.
            probe = verify_class_coverage(
                {"train": y_train_arr, "validation": y_val_arr}, num_classes=num_classes,
                class_names=class_names, fail_hard=False)
            with open(exp_dir / "class_coverage_report.json", "w") as f:
                json.dump({**probe, "coverage_ok": False, "error": str(e)}, f, indent=2)
            print("[class-coverage] per-split class counts:", flush=True)
            for split, counts in probe["per_split_counts"].items():
                print(f"  {split}: {counts}", flush=True)
            print(f"CLASS_COVERAGE_INCOMPLETE: {e}\n"
                  f"Pass --allow_missing_classes to proceed with this as a documented dataset "
                  f"limitation, or re-run the patient-level split with a different --split/--seed.",
                  flush=True)
            raise
        with open(exp_dir / "class_coverage_report.json", "w") as f:
            json.dump(report, f, indent=2)
        if not report["coverage_ok"]:
            print(f"WARNING: class coverage check found missing classes: {report['missing']} "
                  f"(continuing because --allow_missing_classes was passed)", flush=True)

    write_imbalance_report(str(exp_dir / "class_imbalance_report.json"),
                            {"train": y_train_arr, "validation": y_val_arr},
                            num_classes=num_classes, class_names=class_names)

    # Belt-and-suspenders: with the Phase-4 gate above this should be
    # unreachable, but if any residual raw np.load failure slips through it is
    # converted to the typed, classified DATASET_STORAGE_ERROR here rather
    # than a bare ValueError deep in the Dataset constructor.
    def _build_npy_dataset(role, xp, yp):
        try:
            return NpyDataset(xp, yp, storage_mode=args.dataset_storage)
        except (ValueError, OSError) as e:
            eio = (isinstance(e, OSError) and getattr(e, "errno", None) == errno.EIO) \
                or "input/output error" in str(e).lower()
            prefix = "PHYSICALLY UNREADABLE (OSError(5)/EIO) - " if eio else ""
            raise DatasetStorageError(
                f"DATASET_LOAD_ERROR role={role} storage_mode={args.dataset_storage} "
                f"path={xp}: {prefix}{e}") from e

    train_ds = _build_npy_dataset("X_train", x_tr, y_tr)
    val_ds = _build_npy_dataset("X_val", x_te, y_te)

    train_sampler = None
    shuffle = True
    if args.sampler == "balanced":
        train_sampler = build_balanced_sampler(y_train_arr, num_classes)
        shuffle = False
    elif args.sampler == "moderate_oversample":
        idx = build_moderate_oversample_indices(y_train_arr, num_classes, seed=args.seed,
                                                  target_percentile=args.oversample_target_percentile)
        train_ds = Subset(train_ds, idx.tolist())

    # --train_subsample_frac - applied AFTER --sampler so the two compose. Each
    # of the three sampler modes needs a different mechanism:
    #   balanced           -> a WeightedRandomSampler with its own num_samples,
    #                          so shrink num_samples (the Dataset is untouched).
    #   moderate_oversample -> already a Subset, so shrink that Subset.
    #   none                -> wrap the Dataset in a fresh Subset.
    # Deterministic given --seed, and independent of the sampler's own RNG.
    if not 0.0 < args.train_subsample_frac <= 1.0:
        raise SystemExit(f"--train_subsample_frac must be in (0, 1], got {args.train_subsample_frac}")
    if args.train_subsample_frac < 1.0:
        sub_rng = np.random.default_rng(args.seed)
        if train_sampler is not None:
            full_n = train_sampler.num_samples
            train_sampler.num_samples = max(1, int(round(full_n * args.train_subsample_frac)))
            kept, total = train_sampler.num_samples, full_n
        else:
            total = len(train_ds)
            keep = max(1, int(round(total * args.train_subsample_frac)))
            pick = sub_rng.permutation(total)[:keep]
            train_ds = Subset(train_ds, pick.tolist())
            kept = keep
        print(f"[train-subsample] frac={args.train_subsample_frac} -> {kept} of {total} training "
              f"samples per epoch (seed={args.seed}). Report this alongside any metric from this "
              f"run: an 'epoch' here is not a full pass over the split.", flush=True)

    if args.augment_preset == "custom":
        aug_cfg = AugmentationConfig(
            spectral_noise_std=args.spectral_noise_std, spectral_scale_range=args.spectral_scale_range,
            spectral_offset_std=args.spectral_offset_std, band_dropout_prob=args.band_dropout_prob,
            band_dropout_max_frac=args.band_dropout_max_frac, spectral_mask_prob=args.spectral_mask_prob,
            spectral_mask_max_width_frac=args.spectral_mask_max_width_frac, flip_h_prob=args.flip_h_prob,
            flip_v_prob=args.flip_v_prob, rotate90_prob=args.rotate90_prob,
            crop_scale_max_frac=args.crop_scale_max_frac, seed=args.seed,
        )
    else:
        aug_cfg = PRESETS[args.augment_preset]
    if args.augment_preset != "none":
        train_ds = AugmentedPatchDataset(train_ds, aug_cfg.build())
        print(f"[augmentation] train split wrapped with preset={args.augment_preset!r}", flush=True)

    # Phase 1/2 - independent, explicit train/val DataLoader policies.
    train_policy, val_policy = train_val_policies(
        args.loader_mode, args.num_workers, args.prefetch_factor, args.persistent_workers, args.pin_memory)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=train_sampler,
        **train_policy.dataloader_kwargs(device))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        **val_policy.dataloader_kwargs(device))

    system_snapshot = collect_system_memory_snapshot()
    print_preflight(args, train_policy, val_policy, dataset_paths, system_snapshot)

    if args.loader_test:
        run_loader_test(args, train_loader, train_policy)
        return

    modality = "hsi" if wavelengths is not None else "rgb"
    trm_kwargs = None
    if args.architecture == "recursive":
        trm_kwargs = dict(
            trm_dim=args.trm_dim, trm_core_layers=args.trm_core_layers,
            trm_n_latent=args.trm_n_latent, trm_n_improve=args.trm_n_improve,
            trm_deep_supervision_steps=args.trm_deep_supervision_steps,
            trm_mixer=args.trm_mixer, trm_ema_rate=args.trm_ema_rate,
            trm_act_halting=args.trm_halting,
        )
    base_model = build_model(args.architecture, modality, num_classes,
                              drop_path_rate=args.drop_path_rate,
                              fusion_type=(args.fusion_type if args.architecture == "efficient" else None),
                              classifier_dropout=args.classifier_dropout, trm_kwargs=trm_kwargs)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"[architecture] {args.architecture!r} ({modality}): {n_params/1e6:.3f}M parameters "
          f"(drop_path_rate={args.drop_path_rate})", flush=True)

    in_channels = train_ds[0][0].shape[0]

    if args.activation == "leakyrelu":
        replace_relu_with_leakyrelu(base_model, negative_slope=args.leaky_slope)

    if args.gradient_checkpointing:
        n_stages = enable_gradient_checkpointing(base_model)
        n_spectral = enable_spectral_gradient_checkpointing(base_model)
        print(f"[gradient-checkpointing] wrapped {n_stages} backbone stage(s) and "
              f"{n_spectral} spectral pathway(s)", flush=True)

    if args.recon_mode == "latent":
        model = GMedMambaLatentReconWrapper(base_model, in_channels=in_channels,
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
                                 focal_gamma=args.focal_gamma, weight_method=args.weight_method, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Workstream 4 - numerical smoke test BEFORE the real training loop.
    smoke_on = args.numerical_smoke_test == "on" or (
        args.numerical_smoke_test == "auto" and (args.safe_mode or args.debug_numerics))
    smoke_result = None
    if smoke_on:
        smoke_result = run_numerical_smoke_test(
            args, model, criterion, discriminator, use_gan, train_loader, device, exp_dir)

    # Phase 38 - reproducibility metadata.
    config = {
        "cli_args": vars(args), "classes": num_classes, "device": device,
        "class_names": class_names, "in_channels": in_channels,
        "architecture": args.architecture, "fusion_type": args.fusion_type,
        "run_name": exp_dir.name, "dataset_slug": dataset_slug(args.data_dir),
        "modality": modality, "train_subsample_frac": args.train_subsample_frac,
        "backbone_num_params": int(n_params),
        "dataloader_config": {"train": train_policy.as_dict(), "validation": val_policy.as_dict()},
        "dataset_storage_mode": args.dataset_storage,
        "dataset_validation_level": args.dataset_validation_level,
        "amp_mode": args.amp,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_clip_norm": args.max_gradient_norm,
        "sam_loss_config": {"angle_eps": 1e-3, "norm_eps": 1e-6},
        "scientifically_qualified": not args.skip_dataset_validation,
        "dataset_validation": ({"all_ok": integrity_report.all_ok,
                                "statuses": {k: v.get("status")
                                             for k, v in integrity_report.files.items()}}
                               if integrity_report is not None else "skipped"),
        "numerical_smoke_test": smoke_result,
        "system_snapshot_startup": startup_snapshot,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4, default=str)

    stability_cfg = StabilityConfig(
        max_gradient_norm=args.max_gradient_norm,
        max_gradient_skip_ratio=args.max_gradient_skip_ratio,
        max_consecutive_bad_batches=args.max_consecutive_bad_batches,
        max_consecutive_unhealthy_epochs=args.max_consecutive_unhealthy_epochs,
        debug_numerics=args.debug_numerics,
    )

    trainer = TrainerG_v10(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device, exp_dir=exp_dir,
        class_names=class_names,
        lambda_mse=(args.lambda_mse if args.recon_mode != "none" else 0.0),
        lambda_sam=(args.lambda_sam if args.recon_mode != "none" else 0.0),
        lambda_gan=args.lambda_gan, discriminator=discriminator, disc_optimizer=disc_optimizer,
        config=config, wavelengths=wavelengths,
        checkpoint_metric=args.checkpoint_metric, class_collapse_streak=args.class_collapse_streak,
        criterion=criterion, early_stop_patience=args.early_stop_patience,
        grad_clip_norm=args.max_gradient_norm,
        stability_cfg=stability_cfg, amp_mode=args.amp, debug_numerics=args.debug_numerics,
        # TrainerG_v10-only: in-loop deep supervision + EMA for --architecture recursive;
        # a no-op passthrough to TrainerG_v9 for split/fullchannel/efficient.
        ema_rate=(args.trm_ema_rate if args.architecture == "recursive" else None),
    )
    trainer.dataloader_config = config["dataloader_config"]

    # Phase 3 - one more snapshot right before training starts.
    log_system_memory(trainer.logger, label="pre-training")

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume) + 1
        else:
            print(f"Warning: Specified resume path not found: {args.resume}. Starting fresh.", flush=True)

    stop_reason = trainer.fit(max_epochs=args.epochs, start_epoch=start_epoch)
    if stop_reason is not None and str(getattr(stop_reason, "value", stop_reason)) == \
            "TRAINING_ABORTED_NUMERICAL_INSTABILITY":
        raise SystemExit(
            "TRAINING_ABORTED_NUMERICAL_INSTABILITY: see "
            f"{exp_dir}/numerical_failure/failure.json for diagnostics.")


def main():
    """Workstream 5 - stamp every terminal failure with a single
    FailureClass and drop `<exp_dir>/failure_class.json`, then re-raise so
    the process still exits non-zero."""
    try:
        _main()
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            fc = FailureClass.USER_INTERRUPTED
        else:
            fc = classify_exception(e)
        try:
            print(f"FAILURE_CLASS={fc.value}  ({type(e).__name__}: {e})", flush=True)
            exp_dir = _LAST_EXP_DIR.get("path")
            if exp_dir is not None:
                import datetime as _dt
                with open(Path(exp_dir) / "failure_class.json", "w") as f:
                    json.dump({"failure_class": fc.value,
                               "exception": f"{type(e).__name__}: {e}",
                               "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat()},
                              f, indent=2, default=str)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
