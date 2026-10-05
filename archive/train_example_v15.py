# -*- coding: utf-8 -*-
"""
train_example_v15.py
======================
GMedMamba v15 "Representation Collapse Remediation" entry point.

`train_example_v14.py` IS NOT MODIFIED (contract C-1). It remains a valid
entry point for reproducing the two runs this plan was written from -
`experiments/20260902_111050_hsi_v7-80_10_10-hsi_recursive_hsi_mlp_sup3_bs256_bf16`
and `experiments/20260901_195001_pad_v6_recursive_rgb_mlp_sup3_bs256_bf16`.
Do not delete it and do not "fix" it. v15 flips defaults that change results,
which is exactly why it is a new file.

------------------------------------------------------------------------
THE DEFECT (v15 plan, section 0)
------------------------------------------------------------------------
`gmedmamba.SpectralTokenizer` built every band token as
`value_embed(v) + positional_encoding`. `value_embed` is a shared
`nn.Linear(1, d_token)` initialised at `trunc_normal_(std=0.02)` with a zeroed
bias, so on inputs in [0,1] its output has magnitude ~0.006. The positional
encoding - identical for every sample in the dataset - has magnitude ~0.55.
The input signal was therefore 86-93x SMALLER than the constant it was added
to, and after `tokens.mean(dim=1)`, the GELU compressor and `stem_norm`, the
embedding entering the recursive core varied with the input by 0.25%.

Both completed runs consequently began life as constant functions. The HSI
run's `predictions_epoch01.npz` shows max-softmax confidence spanning 0.00022
across all 334,516 validation patches, every sample predicted `DCIS`, and 2.2
GPU-hours spent re-confirming it.

------------------------------------------------------------------------
DEFAULTS CHANGED IN v15 (each is a `GMedMambaConfig` field or CLI flag whose
default in the shared modules still reproduces v14 - contract C-3)
------------------------------------------------------------------------
  --spectral_token_fusion   add -> concat_mlp   R1.1  scale AND rank
  --spectral_pe_gain        1.0 -> 0.1          R1.1
  --spectral_value_init_std 0.02 -> 0.5         R1.1
  --spectral_ctx_norm       off -> on           R1.1b so stem_norm works
  --spatial_pe_gain         1.0 -> 0.1          R1.1c the 2-D PE, same defect
  --classifier_init         shared -> fan_in    R1.1d
  --wavelength_scale        1000 -> band count  R1.3  encoding aliasing
  --use_wavelengths         (new) on            R2.3  first real sensor test
  --normalization           (new) zscore/minmax R2.1  per modality
  --weight_decay_groups     (new) on            R1.2  never decay value_embed
  --numerical_smoke_test    auto -> on          R0.2
  --on_class_collapse       (new) abort         R0.3
  --spectral_chunk_size     1024 -> 0           R3.2
  --train_subsample_mode    (new) per_epoch     R6.1
  --trm_halting             on -> off           R4.3
  --recon_mode              latent -> none      R4.4  (recursive only)
  --eval_test               (new) best          R5.5  gate G6
  --seed_everything         (new) on            R6.4
  --dataset_validation_level structural -> deep for HSI               R6.5

Gates (v15 plan section 2.2), checked by
`pytest test_representation_sensitivity.py` and by this file's preflight:

  G1 stem sensitivity >= 0.05     was 0.0025 (HSI) / 0.0032 (RGB)
  G2 logit std        >= 1e-2     was ~5e-07
  G3 >= 2 distinct predicted classes after epoch 1
  G4 beats the shallow ceiling in `scripts/shallow_baseline.py`
  G5 best_model.pt reproduces the logged best metric
  G6 test_report.json exists

Usage:

    # Q1 - the gates, seconds, no GPU
    pytest test_representation_sensitivity.py -q

    # Q2 - HSI qualification, ~20 min
    python train_example_v15.py --data_dir data/hsi_v7-80_10_10/hsi --epochs 3 \
        --architecture recursive --train_subsample_frac 0.05 --val_subsample_frac 0.1

    # Q3 - RGB qualification, ~30 min
    python train_example_v15.py --data_dir data/pad_v6 --epochs 3 --architecture recursive

    # the legacy tokenizer, as an ablation (the headline R7 result)
    python train_example_v15.py --data_dir data/pad_v6 --epochs 3 \
        --architecture recursive --spectral_token_fusion add

Everything below is inherited unchanged from train_example_v14.py's own
docstring, which describes the Stage-G stability remediation this file still
carries forward.
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

from train_example_v6 import (
    NpyDataset, discover_data, setup_experiment_dir, compute_global_channel_stats,
)
from train_example_v7 import GMedMambaRawReconWrapper
from training.trainerg_v11 import TrainerG_v11
from training.numerical_stability import StabilityConfig, run_sensitivity_check
from training.gan import SpectralDiscriminator
from training.class_coverage import verify_class_coverage
from training.class_imbalance import write_imbalance_report
from training.activation_patch import replace_relu_with_leakyrelu
from training.reconstruction_head import GMedMambaLatentReconWrapper
from training.losses import build_criterion
from training.samplers import build_balanced_sampler, build_moderate_oversample_indices
from training.augmentation import AugmentationConfig, PRESETS, AugmentedPatchDataset
from training.config_presets import build_model, v15_config_overrides
from training.optim_groups import build_param_groups, summarize_param_groups
from training.grad_checkpoint import enable_gradient_checkpointing
from training.spectral_checkpoint import enable_spectral_gradient_checkpointing
from training.torch_compile import enable_torch_compile
from training.dataloader_config import (
    train_val_policies, apply_main_process_thread_limits, seed_everything,
)
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
from training.run_naming import build_run_name, dataset_slug, infer_modality, modality_warnings

EFFICIENT_FUSION_CHOICES = ["film", "gated", "cross_attention", "multiplicative", "residual",
                            "se_gate", "eca", "none"]


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="GMedMamba v15 ('Representation Collapse Remediation') training entry point")
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
    p.add_argument("--weight_method", choices=["balanced", "inverse"], default="balanced",
                    help="NOTE both options are proportional to 1/class_count and are re-normalized "
                         "to mean 1, so they produce IDENTICAL weights. Use --class_weight_power to "
                         "change the STRENGTH of the correction.")
    p.add_argument("--class_weight_power", type=float, default=1.0,
                    help="[R2.4] exponent on the inverse-frequency class weights: 0.0 = uniform "
                         "(equivalent to plain --loss ce), 1.0 = full inverse-frequency (the "
                         "pre-v15.1 behaviour and still the default), 0.5 = sqrt-inverse-frequency. "
                         "Inert unless --loss is weighted_ce or focal_weighted. On this repo's HSI "
                         "split, 1.0 OVERSHOOTS: it gives DCIS (5.0%% of train) 13.1x the weight of "
                         "IDC, and the resulting model calls 32.4%% of genuinely healthy test "
                         "patches DCIS (DCIS recall 0.946 at precision 0.456), capping test "
                         "macro-F1 at 0.804. Re-deciding that run's saved probabilities at an "
                         "effective 0.75 gives macro-F1 0.847; at 0.5, 0.839; at 0.0, 0.666. "
                         "Start at 0.75 for this dataset. See training/losses.py.")
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

    _add_v15_arguments(p)
    return p


def _add_v15_arguments(p) -> None:
    """Everything the v15 plan adds. Grouped in one function so the diff
    against `train_example_v14.py`'s parser is one call."""

    # ------------------------------------------------------------------
    # R1.1/R1.3 - the representation fix
    # ------------------------------------------------------------------
    g = p.add_argument_group("v15 R1 - representation (the collapse fix)")
    g.add_argument("--spectral_token_fusion", choices=["add", "scaled", "concat_mlp"],
                    default="concat_mlp",
                    help="how SpectralTokenizer combines a band value with its positional "
                         "encoding. 'add' is the PRE-v15 behaviour and the defect: a ~0.006 value "
                         "embedding added to a ~0.55 constant. 'scaled' fixes the magnitude. "
                         "'concat_mlp' (default) fixes magnitude AND rank - with a shared "
                         "Linear(1,d) and a zeroed bias every band token is a scalar multiple of "
                         "ONE fixed direction, so all C bands are collinear and the band-mean "
                         "reduces a whole spectrum to a single scalar. Keep 'add' only to "
                         "reproduce a pre-v15 run or as the headline R7 ablation.")
    g.add_argument("--spectral_pe_gain", type=float, default=0.1,
                    help="initial value of the learnable gain on the spectral positional "
                         "encoding. Ignored for --spectral_token_fusion add.")
    g.add_argument("--spectral_value_init_std", type=float, default=0.5,
                    help="init std for the tokenizer's value branch, applied AFTER the backbone's "
                         "global trunc_normal(0.02). At 0.5 against --spectral_pe_gain 0.1 the "
                         "scale ratio is ~4.5:1 in favour of the signal; it was 93:1 against.")
    g.add_argument("--spectral_ctx_norm", dest="spectral_ctx_norm", action="store_true", default=True,
                    help="[R1.1b, default ON] RMS-normalize the spectral context before the stem. "
                         "Without it the stem's input arrives with a per-position variance of "
                         "~1.1e-5, BELOW nn.LayerNorm's eps of 1e-5, so stem_norm degenerates "
                         "into a constant rescale and the embedding stays ~500x smaller than the "
                         "2-D positional encoding it is added to.")
    g.add_argument("--no_spectral_ctx_norm", dest="spectral_ctx_norm", action="store_false")
    g.add_argument("--spatial_pe_gain", type=float, default=0.1,
                    help="[R1.1c, recursive only] initial value of the learnable gain on the "
                         "recursive backbone's 2-D sinusoidal positional encoding - the same "
                         "per-dataset constant one layer below the tokenizer's. 1.0 = pre-v15.")
    g.add_argument("--classifier_init", choices=["shared", "fan_in"], default="fan_in",
                    help="[R1.1d] 'shared' applies the backbone's global trunc_normal(0.02) to "
                         "the classifier's final projection, ~4.4x below its own fan-in scale, "
                         "costing a matching factor of logit sensitivity (gate G2).")
    g.add_argument("--wavelength_scale", type=float, default=None,
                    help="[R1.3] angular span for continuous_wavelength_encoding. Unset (default) "
                         "= the band count, which makes it reduce EXACTLY to the index encoding "
                         "for a uniformly-spaced sensor. The pre-v15 hardcoded 1000.0 makes "
                         "adjacent bands differ by ~32 radians at 32 bands - more than five full "
                         "cycles - so the encoding carries no locality at all.")
    g.add_argument("--weight_decay_groups", dest="weight_decay_groups", action="store_true", default=True,
                    help="[R1.2, default ON] decay only parameters with ndim >= 2 that are not "
                         "norm weights. AdamW(model.parameters(), weight_decay=0.05) decays every "
                         "norm gain and bias and - critically - the one weight that has to GROW "
                         "for the model to become input-sensitive.")
    g.add_argument("--no_weight_decay_groups", dest="weight_decay_groups", action="store_false")

    # ------------------------------------------------------------------
    # R0 - guardrails
    # ------------------------------------------------------------------
    g = p.add_argument_group("v15 R0 - guardrails")
    g.add_argument("--sensitivity_check", choices=["on", "off"], default="on",
                    help="[R0.2] before training, assert gates G1/G2 on one real batch and abort "
                         "with REPRESENTATION_COLLAPSE if the model is a constant function of its "
                         "input. Costs one forward pass.")
    g.add_argument("--min_logit_std", type=float, default=1e-2, help="[R0.2] gate G2 threshold")
    g.add_argument("--min_stem_sensitivity", type=float, default=0.05,
                    help="[R0.2] gate G1 threshold")
    g.add_argument("--on_class_collapse", choices=["warn", "abort"], default="abort",
                    help="[R0.3] what to do when the validation split has been assigned a single "
                         "class for --class_collapse_streak consecutive epochs. v14 warned and "
                         "kept going for 2.2 GPU-hours.")

    # ------------------------------------------------------------------
    # R2 - data pipeline
    # ------------------------------------------------------------------
    g = p.add_argument_group("v15 R2 - data pipeline")
    g.add_argument("--normalization", choices=["per_sample_minmax", "global_zscore", "auto"],
                    default="auto",
                    help="[R2.1] 'auto' (default) = global_zscore for HSI, per_sample_minmax for "
                         "RGB. NpyDataset defaults to per_sample_minmax and v14 never passed the "
                         "argument, so every HSI patch was rescaled by its own scalar min and max "
                         "over all 11x11xC values, both driven by outlier pixels (measured: min "
                         "2005 +/- 1855, max 14820 +/- 3862). Cost on the linear probe: balanced "
                         "accuracy 0.4942 -> 0.4499. RGB is already in [0,1] with meaningful "
                         "absolute colour, so re-stretching each patch there is the more "
                         "questionable operation and stays an explicit ablation.")
    g.add_argument("--norm_stats_sample_cap", type=int, default=5000,
                    help="[R2.1] patches sampled from the TRAIN split (never val/test) to fit "
                         "global_mean/global_std.")
    g.add_argument("--use_wavelengths", dest="use_wavelengths", action="store_true", default=True,
                    help="[R2.3, default ON] forward wavelengths.npy into the model. v14 loaded "
                         "the file and handed it to the trainer, and no trainer in the v4->v10 "
                         "chain ever passed it to a model call - so the sensor-agnostic claim in "
                         "gmedmamba.py's header was untested. Depends on R1.3.")
    g.add_argument("--no_use_wavelengths", dest="use_wavelengths", action="store_false")
    g.add_argument("--sensor_range", type=float, nargs=2, default=None, metavar=("LO_NM", "HI_NM"),
                    help="[R2.3] fixed physical range for cross-sensor-consistent wavelength "
                         "normalization, e.g. --sensor_range 400 1000. Default: per-sample "
                         "min/max.")

    # ------------------------------------------------------------------
    # R3 - compute budget and throughput
    # ------------------------------------------------------------------
    g = p.add_argument_group("v15 R3 - compute budget")
    g.add_argument("--val_subsample_frac", type=float, default=1.0,
                    help="[R3.1] validate on a fixed fraction of the validation split. The HSI "
                         "run validated on 334,516 patches per epoch while TRAINING on 122,604 - "
                         "a 2.7:1 ratio, plus a full deepcopy of the model for the EMA pass, every "
                         "epoch. A stratified 20-40k subset has ample statistical power at these "
                         "sizes. The subset is drawn once and fixed for the whole run, so "
                         "epoch-to-epoch metrics stay comparable.")
    g.add_argument("--val_subsample_mode", choices=["random", "stratified"], default="stratified")
    g.add_argument("--spectral_chunk_size", type=int, default=1024,
                    help="[R3.2] patches per SpectralPathway chunk; 0 = one pass over all N. This "
                         "used to be HARDCODED at 1024 regardless of batch size, which is the "
                         "defect: it was not a choice anyone could make. MEASURED (RTX 5060 Ti, "
                         "bs=256, bf16, 11x11 patches - see findings/finds_20260902_v15_r3.md), "
                         "1024 is also the right VALUE, and the v15 plan's expectation that "
                         "removing the chunk loop would speed things up is wrong on this "
                         "hardware: HSI runs 1175 ms/step at 1024 against 1592 ms/step at 0, a "
                         "35%% REGRESSION at identical peak memory, and the ordering holds at "
                         "bs=128. The chunk loop is also what gives --spectral_checkpointing "
                         "something to discard: with one chunk it saves nothing, with 1024 it "
                         "cuts HSI peak memory 10,243 MB -> 1,007 MB.")
    g.add_argument("--spectral_checkpointing", choices=["auto", "on", "off"], default="auto",
                    help="[R3.2] gradient-checkpoint the spectral pathway INDEPENDENTLY of "
                         "--gradient_checkpointing. 'auto' = on when C >= "
                         "--spectral_checkpoint_min_channels. Peak GPU was 9,695 MB (HSI) vs "
                         "1,714 MB (RGB) for the same 0.44M-parameter model and neither completed "
                         "run reached training/spectral_checkpoint.py. MEASURED at bs=256 with "
                         "--spectral_chunk_size 1024: HSI 10,243 -> 1,007 MB for +15%% step time, "
                         "RGB 1,886 -> 1,004 MB for +10%%. Requires a non-zero chunk size to do "
                         "anything - see --spectral_chunk_size.")
    g.add_argument("--spectral_checkpoint_min_channels", type=int, default=16,
                    help="[R3.2] the C threshold --spectral_checkpointing auto uses")
    g.add_argument("--compile", dest="compile_model", choices=["off", "on"], default="off",
                    help="[R3.4] torch.compile the two hot loops of the recursive architecture: "
                         "gmedmamba._selective_scan_pure_pytorch (a PYTHON loop over the band "
                         "dimension, run spectral_depth x ceil(N/spectral_chunk_size) times per "
                         "forward, and again during backward under --spectral_checkpointing) and "
                         "RecursiveCore._f_forward. Measured on an RTX 5060 Ti at --batch_size 256 "
                         "on 11x11x32 HSI: 2873 -> 1412 ms/step, i.e. 89.5 -> 181.3 samples/s "
                         "(2.03x), with the spectral half alone going 1733 -> 843 ms. Max relative "
                         "error vs the eager scan is 1.7e-07 in fp32, the precision the scan "
                         "actually runs at. Costs ~1-2 min of Inductor compilation on the first "
                         "step; falls back to eager with a warning if compilation fails. "
                         "'off' (default) keeps every pre-v15.1 run bit-comparable.")
    g.add_argument("--compile_mode", type=str, default=None,
                    help="[R3.4] torch.compile `mode` (e.g. max-autotune). Unset = Inductor's "
                         "default, which is what the numbers above were measured with; "
                         "max-autotune is not worth its compile time on this GPU (too few SMs "
                         "for max_autotune_gemm).")
    g.add_argument("--trm_checkpoint_core", dest="trm_checkpoint_core", action="store_true", default=True,
                    help="[R3.3, recursive only] gradient-checkpoint every RecursiveCore.f call. "
                         "This used to be unconditional, so --gradient_checkpointing did not "
                         "control it and the recompute was always paid.")
    g.add_argument("--no_trm_checkpoint_core", dest="trm_checkpoint_core", action="store_false")

    # ------------------------------------------------------------------
    # R4 - capacity and dead configuration
    # ------------------------------------------------------------------
    g = p.add_argument_group("v15 R4 - capacity")
    g.add_argument("--trm_mixer_channel_mlp", dest="trm_mixer_channel_mlp",
                    action="store_true", default=True,
                    help="[R4.1] keep _TokenMLPMixer's own GatedMLP. With it, each 'token mixer + "
                         "channel MLP' block is really one depthwise conv (1,280 params) and TWO "
                         "channel MLPs (98,944 each) - 90%% of the model on channel mixing and "
                         "4,608 params in total on spatial mixing. --no_trm_mixer_channel_mlp "
                         "makes the mixer a pure spatial operator; spend the freed budget with "
                         "--trm_mixer attention.")
    g.add_argument("--no_trm_mixer_channel_mlp", dest="trm_mixer_channel_mlp", action="store_false")
    g.add_argument("--trm_drop_path", type=float, default=0.0,
                    help="[R4.2] stochastic depth inside RecursiveMambaBlock. The recursive block "
                         "has no DropPath, no LayerScale and no dropout while the hierarchical "
                         "GBlock has all three - and R1.2 removes weight decay from norms and "
                         "biases, so without this the recursive model has essentially no "
                         "regularizer. --drop_path_rate is used when this is left at 0.")
    g.add_argument("--trm_dropout", type=float, default=0.0,
                    help="[R4.2] dropout inside the recursive block's GatedMLPs")
    g.add_argument("--trm_halt_threshold", type=float, default=None,
                    help="[R4.3] turn the halt head into REAL ACT: stop segments once "
                         "sigmoid(q) > this for every sample in the batch, with "
                         "--trm_halt_exploration_prob as the training-time probability of running "
                         "the full budget anyway. Unset (default) = the pre-v15 state, where `q` "
                         "was trained by a BCE term that nothing ever read: measured on the RGB "
                         "run the halt BCE sat at ~0.66 against ln 2 = 0.693 for 33 epochs - "
                         "chance, flat, and ~20%% of the training objective. That is why v15 "
                         "defaults to --no_trm_halting; pass --trm_halting --trm_halt_threshold "
                         "0.5 for the full-ACT arm of the R7 ablation.")
    g.add_argument("--trm_halt_exploration_prob", type=float, default=0.1,
                    help="[R4.3] see --trm_halt_threshold. Inert without it.")

    # ------------------------------------------------------------------
    # R5 - reporting integrity
    # ------------------------------------------------------------------
    g = p.add_argument_group("v15 R5 - reporting integrity")
    g.add_argument("--eval_test", choices=["off", "best"], default="best",
                    help="[R5.5, gate G6] after fit, reload the best checkpoint and evaluate the "
                         "held-out test split ONCE. Both datasets ship a patient-disjoint test "
                         "split (the HSI one holds out five patients) that has never been "
                         "evaluated.")
    g.add_argument("--verify_best_checkpoint", dest="verify_best_checkpoint",
                    action="store_true", default=True,
                    help="[R5.3, gate G5] after fit, reload best_model.pt and confirm it "
                         "reproduces the logged best metric.")
    g.add_argument("--no_verify_best_checkpoint", dest="verify_best_checkpoint", action="store_false")
    g.add_argument("--keep_last_n", type=int, default=3,
                    help="[R5.6] per-epoch checkpoints to keep (the best is always kept). 0 = keep "
                         "everything.")
    g.add_argument("--eval_artifact_stride", type=int, default=1,
                    help="[R5.6] write the per-epoch confusion-matrix PNG, classification report "
                         "and prediction .npz every N epochs. The HSI run wrote six byte-identical "
                         "PNGs and six 334k-row .npz files.")

    # ------------------------------------------------------------------
    # R6 - reproducibility and provenance
    # ------------------------------------------------------------------
    g = p.add_argument_group("v15 R6 - reproducibility")
    g.add_argument("--train_subsample_mode", choices=["fixed", "per_epoch"], default="per_epoch",
                    help="[R6.1] 'per_epoch' (default) draws a FRESH subset every epoch, so 20 "
                         "epochs at --train_subsample_frac 0.05 see the whole split rather than "
                         "the same 5%% twenty times. 'fixed' is v14's actual behaviour (one subset "
                         "chosen before the loop) and is what reproduces the HSI evidence run - "
                         "v14's module docstring said 'each epoch', its argument help said 'each "
                         "run', its console message said 'per epoch', and the code picked one "
                         "fixed subset.")
    g.add_argument("--deterministic", action="store_true",
                    help="[R6.4] cudnn.deterministic on, cudnn.benchmark off, "
                         "use_deterministic_algorithms(warn_only=True). Slower; use it when a run "
                         "has to be bit-reproducible.")

    # ------------------------------------------------------------------
    # Defaults inherited from v14 that v15 overrides
    # ------------------------------------------------------------------
    p.set_defaults(numerical_smoke_test="on",   # R0.2 - it costs one forward pass
                    trm_halting=False)           # R4.3 - see _apply_v15_defaults


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
    # R3.4 - safe mode is the "reproduce it exactly, in eager, with every guard
    # on" combination; a JIT in the middle of that defeats the purpose.
    _set("compile_model", "off")
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


def apply_v15_defaults(args: argparse.Namespace, explicit: set = None) -> list:
    """v15 defaults that depend on the architecture or the modality, applied
    after parsing. Returns the list of notes to print, so the run log says
    exactly which defaults were resolved and why. An explicit CLI value always
    wins - `explicit` is the set of dests the user actually typed.
    """
    explicit = set() if explicit is None else explicit
    notes = []

    def _set(name, value, why):
        if name in explicit:
            return
        if getattr(args, name, None) != value:
            notes.append(f"{name}={value}  ({why})")
        setattr(args, name, value)

    # R4.4 - --recon_mode. v14's parser default is `latent`; combined with
    # `recursive`, TrainerG_v10 feeds the decoder `base.last_feature_map`,
    # which is DETACHED - so MSE + 0.1*SAM add ~1.0 to the reported loss while
    # training only the decoder, and nothing reaches the recursive core.
    if args.architecture == "recursive":
        _set("recon_mode", "none",
             "R4.4 - the latent decoder reads a DETACHED feature map, so it adds ~1.0 to the "
             "loss and trains only itself")
        if args.recon_mode == "latent":
            notes.append(
                "WARNING: --recon_mode latent with --architecture recursive trains only the "
                "decoder (gmedmamba.GMedMambaRecursive caches a detached last_feature_map so EMA "
                "can deepcopy the module). The MSE/SAM terms will move and mean nothing.")

    # R6.5 - --dataset_validation_level. `data/hsi_v7-80_10_10/hsi`'s
    # prep_log.txt records `PHYSICALLY UNREADABLE - deep read failed:
    # OSError(5)/EIO at byte 12946243456` for this exact X_train.npy on
    # 2026-09-02 10:14. The array was re-finalized and deep-verified
    # afterwards, but the training run then used `structural`, which probes
    # ~19 rows per file.
    if infer_modality(args.data_dir) == "hsi":
        _set("dataset_validation_level", "deep",
             "R6.5 - this dataset had a physically-unreadable region during prep; `structural` "
             "probes ~19 rows per file")

    # R2.1 - --normalization auto.
    if args.normalization == "auto":
        resolved = "global_zscore" if infer_modality(args.data_dir) == "hsi" else "per_sample_minmax"
        args.normalization = resolved
        notes.append(f"normalization={resolved}  (R2.1 - auto, by modality)")

    return notes


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
    lines.append(f"  torch.compile: {args.compile_model}"
                  + (f" (mode={args.compile_mode})" if args.compile_mode else "")
                  + ("   <-- ~2.0x slower than --compile on, on the recursive architecture"
                     if args.compile_model == "off" and args.architecture == "recursive" else ""))
    lines.append("")
    lines.append("Representation (v15 R1):")
    lines.append(f"  spectral_token_fusion: {args.spectral_token_fusion}"
                  + ("   <-- PRE-v15 DEFECT: the input signal is ~90x smaller than the "
                     "constant it is added to" if args.spectral_token_fusion == "add" else ""))
    lines.append(f"  spectral_pe_gain: {args.spectral_pe_gain}   "
                  f"spectral_value_init_std: {args.spectral_value_init_std}")
    lines.append(f"  spectral_ctx_norm: {args.spectral_ctx_norm}   "
                  f"classifier_init: {args.classifier_init}")
    if args.architecture == "recursive":
        lines.append(f"  spatial_pe_gain: {args.spatial_pe_gain}")
    lines.append(f"  wavelength_encoding_scale: "
                  f"{args.wavelength_scale if args.wavelength_scale is not None else 'band count (R1.3)'}")
    lines.append(f"  use_wavelengths: {args.use_wavelengths}   sensor_range: {args.sensor_range}")
    lines.append(f"  weight_decay_groups: {args.weight_decay_groups}  "
                  f"(weight_decay={args.weight_decay})")
    lines.append("")
    lines.append("Data / budget (v15 R2-R3, R6):")
    lines.append(f"  normalization: {args.normalization}")
    lines.append(f"  train_subsample: {args.train_subsample_frac} ({args.train_subsample_mode})   "
                  f"val_subsample: {args.val_subsample_frac} ({args.val_subsample_mode})")
    lines.append(f"  spectral_chunk_size: {args.spectral_chunk_size}   "
                  f"spectral_checkpointing: {args.spectral_checkpointing}")
    lines.append(f"  seed: {args.seed}   deterministic: {args.deterministic}")
    lines.append("")
    lines.append("Gates:")
    lines.append(f"  G1/G2 sensitivity check: {args.sensitivity_check} "
                  f"(stem >= {args.min_stem_sensitivity}, logit std >= {args.min_logit_std})")
    lines.append(f"  G3 on_class_collapse: {args.on_class_collapse} "
                  f"(streak {args.class_collapse_streak})")
    lines.append(f"  G5 verify_best_checkpoint: {args.verify_best_checkpoint}")
    lines.append(f"  G6 eval_test: {args.eval_test}")
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


def _probe_in_channels(dataset) -> int:
    """v15 R6.6 - the channel count, read WITHOUT consuming an augmentation
    RNG draw.

    v14 did `train_ds[0][0].shape[0]` after wrapping the dataset in
    `AugmentedPatchDataset`, so sample 0 was augmented once, off-loop, purely
    to learn its channel count - which (before R6.3) advanced the single shared
    augmentation RNG that every worker then inherited. Unwrap to the underlying
    dataset first; the channel count is the same and nothing is disturbed.
    """
    seen = set()
    while id(dataset) not in seen:
        seen.add(id(dataset))
        inner = getattr(dataset, "base_dataset", None) or getattr(dataset, "dataset", None)
        if inner is None:
            break
        dataset = inner
    X = getattr(dataset, "X", None)
    if X is not None and getattr(X, "ndim", 0) == 4:
        return int(X.shape[-1])          # NpyDataset stores [N,H,W,C]
    return int(dataset[0][0].shape[0])   # already [C,H,W]


def _build_test_loader(args, data_dir, num_classes, storage_mode, normalization,
                        global_mean, global_std, policy, device, val_x_path):
    """v15 R5.5 (gate G6) - a loader over the HELD-OUT test split.

    Returns None, with a warning, when `discover_data` already used
    `X_test.npy` as the validation split (a dataset with no `X_val.npy`):
    "evaluating the test set" on the split that selected the checkpoint is
    worse than not evaluating it, because it produces a number that looks
    held-out and is not.
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

    test_ds = NpyDataset(str(x_test), str(y_test), storage_mode=storage_mode,
                          normalization=normalization,
                          global_mean=global_mean, global_std=global_std)
    print(f"[test-eval] {len(test_ds)} held-out test patches from {x_test}", flush=True)
    return DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       **policy.dataloader_kwargs(device, seed=args.seed))


def _subsample_indices(labels: np.ndarray, frac: float, mode: str, num_classes: int,
                        seed: int) -> np.ndarray:
    """v15 R3.1 - indices for a `frac` subsample of `labels`, deterministic
    given `seed`.

    `stratified` (the default) keeps each class's share, so a 10% subset of an
    imbalanced validation split still contains the rare classes - a plain
    random 10% of the HSI split can plausibly drop a minority class entirely
    and make `balanced_accuracy` undefined for it.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    if mode == "random":
        keep = max(1, int(round(n * frac)))
        return np.sort(rng.choice(n, size=keep, replace=False))

    picked = []
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        keep = max(1, int(round(idx.size * frac)))
        picked.append(rng.choice(idx, size=min(keep, idx.size), replace=False))
    return np.sort(np.concatenate(picked)) if picked else np.arange(n)


def run_representation_sensitivity_check(args, model, train_loader, device, exp_dir):
    """v15 R0.2 - the whole audit, as a precondition.

    One forward pass on one real batch. If the model is a constant function of
    its input, this says so in under a minute instead of after 19.7 GPU-hours.
    Aborts with `SystemExit("REPRESENTATION_COLLAPSE: ...")` and writes
    `<exp_dir>/numerical_failure/sensitivity_failure.json`.
    """
    try:
        x, _ = next(iter(train_loader))
    except StopIteration:
        print("[sensitivity] train_loader yielded no batch - skipping.", flush=True)
        return {"passed": True, "skipped": True}

    report = run_sensitivity_check(
        model.to(device), x.to(device),
        min_logit_std=args.min_logit_std,
        min_stem_sensitivity=args.min_stem_sensitivity)

    stem = report.get("stem_sensitivity")
    line = (f"[sensitivity] G1 stem={stem:.5f} (>= {args.min_stem_sensitivity})  "
            if stem is not None else "[sensitivity] G1 stem=n/a  ")
    line += f"G2 logit_std={report['logit_std']:.3e} (>= {args.min_logit_std:.0e})"
    print(line, flush=True)

    if not report.get("passed"):
        fail_dir = exp_dir / "numerical_failure"
        fail_dir.mkdir(parents=True, exist_ok=True)
        with open(fail_dir / "sensitivity_failure.json", "w") as f:
            json.dump({**report, "cli_args": vars(args)}, f, indent=2, default=str)
        raise SystemExit(
            f"REPRESENTATION_COLLAPSE: {report.get('reason')}\n"
            f"The model is (close to) a constant function of its input, so training it would "
            f"measure nothing. See {fail_dir}/sensitivity_failure.json, run "
            f"`pytest test_representation_sensitivity.py -q`, and check "
            f"--spectral_token_fusion is not left at the legacy 'add'.")
    print("[sensitivity] PASSED (gates G1 and G2)", flush=True)
    return report


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
    explicit = explicitly_passed()
    apply_safe_mode(args, explicit)
    v15_notes = apply_v15_defaults(args, explicit)

    # R6.4 - seed EVERYTHING, or stop claiming the run is seeded. v6..v14 set
    # torch.manual_seed and np.random.seed and neither torch.cuda.manual_seed_all
    # nor random.seed, while cudnn.benchmark was set unconditionally.
    seed_state = seed_everything(args.seed, deterministic=args.deterministic)
    apply_main_process_thread_limits(args.cpu_threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not args.deterministic:
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

    for note in v15_notes:
        print(f"[v15 default] {note}", flush=True)
    # R6.6 - a spectral dataset must not train silently as RGB.
    for w in modality_warnings(args.data_dir):
        print(f"WARNING: {w}", flush=True)

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
    # R2.1 - the normalization the model actually sees, fitted on TRAIN ONLY
    # and recorded. `NpyDataset` defaults to per_sample_minmax and v14 never
    # passed the argument, so neither completed run states how its inputs were
    # scaled.
    global_mean = global_std = None
    if args.normalization == "global_zscore":
        t0 = time.time()
        global_mean, global_std = compute_global_channel_stats(
            x_tr, sample_cap=args.norm_stats_sample_cap, seed=args.seed)
        print(f"[normalization] global_zscore: per-channel mean/std fit on "
              f"{min(args.norm_stats_sample_cap, n_train)} TRAIN patches in {time.time() - t0:.1f}s "
              f"(mean {global_mean.min():.4g}..{global_mean.max():.4g}, "
              f"std {global_std.min():.4g}..{global_std.max():.4g})", flush=True)

    def _build_npy_dataset(role, xp, yp):
        try:
            return NpyDataset(xp, yp, storage_mode=args.dataset_storage,
                               normalization=args.normalization,
                               global_mean=global_mean, global_std=global_std)
        except (ValueError, OSError) as e:
            eio = (isinstance(e, OSError) and getattr(e, "errno", None) == errno.EIO) \
                or "input/output error" in str(e).lower()
            prefix = "PHYSICALLY UNREADABLE (OSError(5)/EIO) - " if eio else ""
            raise DatasetStorageError(
                f"DATASET_LOAD_ERROR role={role} storage_mode={args.dataset_storage} "
                f"path={xp}: {prefix}{e}") from e

    train_ds = _build_npy_dataset("X_train", x_tr, y_tr)
    val_ds = _build_npy_dataset("X_val", x_te, y_te)

    # R3.1 - subsample the VALIDATION split. The HSI run validated on 334,516
    # patches per epoch while training on 122,604 - a 2.7:1 ratio, plus a full
    # copy.deepcopy of the model for the EMA pass, every epoch. The subset is
    # drawn ONCE and fixed for the whole run so epoch-to-epoch metrics stay
    # comparable, and it is recorded in config.json and in the run name.
    val_subsample_info = {"frac": args.val_subsample_frac, "mode": args.val_subsample_mode,
                          "kept": len(val_ds), "total": len(val_ds)}
    if not 0.0 < args.val_subsample_frac <= 1.0:
        raise SystemExit(f"--val_subsample_frac must be in (0, 1], got {args.val_subsample_frac}")
    if args.val_subsample_frac < 1.0:
        keep_idx = _subsample_indices(y_val_arr, args.val_subsample_frac, args.val_subsample_mode,
                                       num_classes, args.seed + 1)
        val_ds = Subset(val_ds, keep_idx.tolist())
        val_subsample_info.update(kept=len(keep_idx))
        print(f"[val-subsample] frac={args.val_subsample_frac} mode={args.val_subsample_mode} -> "
              f"{len(keep_idx)} of {val_subsample_info['total']} validation patches, fixed for the "
              f"whole run (seed={args.seed + 1}). Class counts: "
              f"{np.bincount(y_val_arr[keep_idx], minlength=num_classes).tolist()}", flush=True)

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
    # R6.1 - `--train_subsample_frac` finally means one specific thing, and all
    # three strings agree with the code. v14's module docstring said "each
    # epoch", its argument help said "each run", its console message said "per
    # epoch", and the code picked ONE fixed subset before the loop. At 5% that
    # is the difference between seeing all 2.45M HSI patches over 20 epochs and
    # seeing the same 122,604 forever.
    if not 0.0 < args.train_subsample_frac <= 1.0:
        raise SystemExit(f"--train_subsample_frac must be in (0, 1], got {args.train_subsample_frac}")
    train_subsample_info = {"frac": args.train_subsample_frac, "mode": args.train_subsample_mode,
                            "kept": len(train_ds), "total": len(train_ds)}
    if args.train_subsample_frac < 1.0:
        if train_sampler is not None:
            # A WeightedRandomSampler already draws `num_samples` per epoch WITH
            # replacement, so it resamples every epoch by construction: shrink
            # num_samples and both modes coincide.
            full_n = train_sampler.num_samples
            train_sampler.num_samples = max(1, int(round(full_n * args.train_subsample_frac)))
            kept, total = train_sampler.num_samples, full_n
            resamples = True
        elif args.train_subsample_mode == "per_epoch":
            # A fresh random subset every epoch, via RandomSampler's own
            # num_samples. `shuffle` must be off when a sampler is given.
            total = len(train_ds)
            kept = max(1, int(round(total * args.train_subsample_frac)))
            gen = torch.Generator().manual_seed(args.seed)
            train_sampler = torch.utils.data.RandomSampler(
                train_ds, replacement=False, num_samples=kept, generator=gen)
            shuffle = False
            resamples = True
        else:   # "fixed" - v14's actual behaviour, kept so the HSI run reproduces
            total = len(train_ds)
            kept = max(1, int(round(total * args.train_subsample_frac)))
            pick = np.random.default_rng(args.seed).permutation(total)[:kept]
            train_ds = Subset(train_ds, pick.tolist())
            resamples = False
        train_subsample_info.update(kept=kept, total=total, resamples_each_epoch=resamples)
        print(f"[train-subsample] frac={args.train_subsample_frac} mode={args.train_subsample_mode} "
              f"-> {kept} of {total} training samples per epoch (seed={args.seed}); "
              f"{'a FRESH subset every epoch' if resamples else 'the SAME subset every epoch'}. "
              f"Report this alongside any metric from this run: an 'epoch' here is not a full pass "
              f"over the split - report optimizer STEPS too.", flush=True)

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
    augmented_train_ds = None
    if args.augment_preset != "none":
        # R6.3 - the augmentation RNG is per (seed, epoch, index, worker) now,
        # so the four DataLoader workers no longer draw the identical
        # flip/rot90 sequence. See training/augmentation.py.
        train_ds = AugmentedPatchDataset(train_ds, aug_cfg.build(), seed=args.seed)
        augmented_train_ds = train_ds
        print(f"[augmentation] train split wrapped with preset={args.augment_preset!r} "
              f"(per-item RNG, seed={args.seed})", flush=True)

    # Phase 1/2 - independent, explicit train/val DataLoader policies.
    train_policy, val_policy = train_val_policies(
        args.loader_mode, args.num_workers, args.prefetch_factor, args.persistent_workers, args.pin_memory)

    # R6.4 - the worker seeds finally derive from --seed. They were `0 + worker_id`
    # regardless of it.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=train_sampler,
        **train_policy.dataloader_kwargs(device, seed=args.seed))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        **val_policy.dataloader_kwargs(device, seed=args.seed))

    modality = "hsi" if wavelengths is not None else "rgb"
    # R6.6 - `print_preflight` accepts in_channels/modality and v14 called it
    # without them, so the line never printed. Read the channel count from the
    # UNDERLYING dataset, before the augmentation wrapper, so this does not
    # consume an augmentation RNG draw for sample 0.
    in_channels = int(_probe_in_channels(train_ds))

    system_snapshot = collect_system_memory_snapshot()
    print_preflight(args, train_policy, val_policy, dataset_paths, system_snapshot,
                     in_channels=in_channels, modality=modality)

    if args.loader_test:
        run_loader_test(args, train_loader, train_policy)
        return

    # R1.1/R1.1b-d/R1.3/R3.2 - the representation fix, as config, from the
    # single source of truth in training/config_presets.py.
    cfg_overrides = dict(v15_config_overrides(args.architecture))
    cfg_overrides.update({
        "spectral_token_fusion": args.spectral_token_fusion,
        "spectral_pe_gain": args.spectral_pe_gain,
        "spectral_value_init_std": args.spectral_value_init_std,
        "spectral_ctx_norm": args.spectral_ctx_norm,
        "classifier_init": args.classifier_init,
        "wavelength_encoding_scale": args.wavelength_scale,   # None => the band count (R1.3)
        "spectral_chunk_size": args.spectral_chunk_size,      # R3.2
    })
    trm_kwargs = None
    if args.architecture == "recursive":
        trm_kwargs = dict(
            trm_dim=args.trm_dim, trm_core_layers=args.trm_core_layers,
            trm_n_latent=args.trm_n_latent, trm_n_improve=args.trm_n_improve,
            trm_deep_supervision_steps=args.trm_deep_supervision_steps,
            trm_mixer=args.trm_mixer, trm_ema_rate=args.trm_ema_rate,
            trm_act_halting=args.trm_halting,                       # R4.3: off by default
            trm_halt_threshold=args.trm_halt_threshold,             # R4.3: None => no ACT
            trm_halt_exploration_prob=args.trm_halt_exploration_prob,
            trm_spatial_pe_gain=args.spatial_pe_gain,               # R1.1c
            trm_checkpoint_core=args.trm_checkpoint_core,           # R3.3
            trm_mixer_channel_mlp=args.trm_mixer_channel_mlp,       # R4.1
            trm_drop_path=(args.trm_drop_path if args.trm_drop_path > 0
                           else (args.drop_path_rate or 0.0)),      # R4.2
            trm_dropout=args.trm_dropout,                           # R4.2
        )
    base_model = build_model(args.architecture, modality, num_classes,
                              drop_path_rate=args.drop_path_rate,
                              fusion_type=(args.fusion_type if args.architecture == "efficient" else None),
                              classifier_dropout=args.classifier_dropout, trm_kwargs=trm_kwargs,
                              cfg_overrides=cfg_overrides)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"[architecture] {args.architecture!r} ({modality}): {n_params/1e6:.3f}M parameters "
          f"(spectral_token_fusion={args.spectral_token_fusion}, "
          f"drop_path_rate={args.drop_path_rate})", flush=True)

    if args.activation == "leakyrelu":
        replace_relu_with_leakyrelu(base_model, negative_slope=args.leaky_slope)

    if args.gradient_checkpointing:
        # R3.3 - strict=True: a silent no-op on a flag the caller explicitly
        # passed is how the recursive backbone went unchecked for 19.7 GPU-hours
        # while the preflight reported `gradient_checkpointing: True`.
        n_stages = enable_gradient_checkpointing(base_model, strict=True)
        print(f"[gradient-checkpointing] wrapped {n_stages} backbone module(s)", flush=True)

    # R3.2 - the spectral pathway is checkpointed INDEPENDENTLY of
    # --gradient_checkpointing. Peak GPU was 9,695 MB (HSI) vs 1,714 MB (RGB)
    # for the same 0.44M-parameter model, and neither run reached this.
    spectral_ckpt_on = (args.spectral_checkpointing == "on"
                        or (args.spectral_checkpointing == "auto"
                            and in_channels >= args.spectral_checkpoint_min_channels))
    if spectral_ckpt_on:
        n_spectral = enable_spectral_gradient_checkpointing(base_model, strict=True)
        print(f"[spectral-checkpointing] {args.spectral_checkpointing} -> wrapped {n_spectral} "
              f"spectral pathway(s) (C={in_channels} >= "
              f"{args.spectral_checkpoint_min_channels})", flush=True)
    else:
        print(f"[spectral-checkpointing] off (C={in_channels})", flush=True)

    # R3.4 - after the checkpoint wrappers, so `RecursiveCore._f_forward` is
    # compiled INSIDE `torch.utils.checkpoint` (the recompute and the backward
    # then run on the fused kernels too), and before any EMA/recon wrapper
    # deep-copies the module.
    if args.compile_model == "on":
        summary = enable_torch_compile(base_model, mode=args.compile_mode)
        print(f"[torch-compile] {summary}", flush=True)
    else:
        print("[torch-compile] off (--compile on is ~2.0x faster on the "
              "recursive architecture; see --compile)", flush=True)

    if args.recon_mode == "latent":
        model = GMedMambaLatentReconWrapper(base_model, in_channels=in_channels,
                                             spectral_dropout=args.spectral_dropout)
    elif args.recon_mode == "raw_input":
        model = GMedMambaRawReconWrapper(base_model, in_channels=in_channels)
    else:
        model = base_model

    use_gan = (args.use_gan or args.lambda_gan > 0) and args.recon_mode != "none"
    discriminator = disc_optimizer = None
    lambda_gan_promoted = False
    if use_gan:
        discriminator = SpectralDiscriminator(in_channels=in_channels)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr, weight_decay=0.0)
        if args.lambda_gan == 0.0:
            args.lambda_gan = 0.05
            lambda_gan_promoted = True   # R6.6 - recorded in config.json below

    criterion = build_criterion(args.loss, y_train=y_train_arr, num_classes=num_classes,
                                 focal_gamma=args.focal_gamma, weight_method=args.weight_method,
                                 device=device, class_weight_power=args.class_weight_power)

    # R1.2 - decay only ndim>=2 non-norm weights. `AdamW(model.parameters(),
    # weight_decay=0.05)` decays every norm gain and bias and - critically -
    # the tokenizer's value embedding, the one weight that has to GROW for the
    # model to become input-sensitive.
    param_group_summary = None
    if args.weight_decay_groups:
        param_group_summary = summarize_param_groups(model, args.weight_decay)
        optimizer = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)
        print(f"[optimizer] weight decay {args.weight_decay} on "
              f"{param_group_summary['n_decay_tensors']} tensors "
              f"({param_group_summary['n_decay_elements']:,} params); 0.0 on "
              f"{param_group_summary['n_no_decay_tensors']} "
              f"({param_group_summary['n_no_decay_elements']:,})", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Workstream 4 - numerical smoke test BEFORE the real training loop.
    smoke_on = args.numerical_smoke_test == "on" or (
        args.numerical_smoke_test == "auto" and (args.safe_mode or args.debug_numerics))
    smoke_result = None
    if smoke_on:
        smoke_result = run_numerical_smoke_test(
            args, model, criterion, discriminator, use_gan, train_loader, device, exp_dir)

    # R0.2 - the representation gate, right after the finiteness gate. This is
    # the one that would have stopped both evidence runs in under a minute.
    sensitivity_result = None
    if args.sensitivity_check == "on":
        sensitivity_result = run_representation_sensitivity_check(
            args, model, train_loader, device, exp_dir)

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
        "dataset_validation": ({"all_ok": integrity_report.all_ok,
                                "statuses": {k: v.get("status")
                                             for k, v in integrity_report.files.items()}}
                               if integrity_report is not None else "skipped"),
        "numerical_smoke_test": smoke_result,
        "system_snapshot_startup": startup_snapshot,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),

        # ---- v15 ----
        "entry_point": "train_example_v15.py",
        "trainer": "TrainerG_v11",
        # R1 - every field that decides whether the model can see its input.
        "representation": {
            "spectral_token_fusion": args.spectral_token_fusion,
            "spectral_pe_gain": args.spectral_pe_gain,
            "spectral_value_init_std": args.spectral_value_init_std,
            "spectral_ctx_norm": args.spectral_ctx_norm,
            "spatial_pe_gain": (args.spatial_pe_gain if args.architecture == "recursive" else None),
            "classifier_init": args.classifier_init,
            "wavelength_encoding_scale": args.wavelength_scale,
            "cfg_overrides": cfg_overrides,
        },
        # R0.2 - gates G1/G2, as measured on this run's own first batch.
        "sensitivity_check": sensitivity_result,
        # R1.2 - which parameters were decayed.
        "weight_decay_groups": param_group_summary,
        # R2.1 - the normalization, and the statistics it was fitted with.
        # NEITHER was recorded anywhere before: the two completed runs do not
        # state how their inputs were scaled.
        "normalization": {
            "mode": args.normalization,
            "stats_sample_cap": args.norm_stats_sample_cap,
            "global_mean": (global_mean.tolist() if global_mean is not None else None),
            "global_std": (global_std.tolist() if global_std is not None else None),
            "fitted_on": "train split only",
        },
        # R2.3 - whether the sensor metadata reached the model at all.
        "wavelengths": {
            "available": wavelengths is not None,
            "forwarded_to_model": bool(args.use_wavelengths and wavelengths is not None),
            "n_bands": (int(len(wavelengths)) if wavelengths is not None else None),
            "min_nm": (float(np.min(wavelengths)) if wavelengths is not None else None),
            "max_nm": (float(np.max(wavelengths)) if wavelengths is not None else None),
            "sensor_range": args.sensor_range,
        },
        # R3.1/R6.1 - the budget, unambiguously.
        "train_subsample": train_subsample_info,
        "val_subsample": val_subsample_info,
        "spectral_chunk_size": args.spectral_chunk_size,
        "spectral_checkpointing": {"mode": args.spectral_checkpointing, "enabled": spectral_ckpt_on},
        # R6.4 - what "seeded" actually means for this run.
        "seeding": seed_state,
        # R6.4 - `scientifically_qualified` was a single boolean meaning only
        # "dataset validation was not skipped". It is a dict now, and every
        # entry is a fact about THIS run.
        "scientifically_qualified": {
            "dataset_validated": not args.skip_dataset_validation,
            "dataset_validation_level": args.dataset_validation_level,
            "seeded": True,
            "deterministic": bool(args.deterministic),
            "representation_gates_checked": args.sensitivity_check == "on",
            "representation_gates_passed": (bool(sensitivity_result.get("passed"))
                                            if sensitivity_result else None),
            "test_evaluated": args.eval_test != "off",
        },
    }
    # R6.6 - record the auto-promotion instead of leaving config.json claiming
    # a lambda_gan the run did not use.
    if lambda_gan_promoted:
        config["lambda_gan_auto_promoted"] = {"from": 0.0, "to": args.lambda_gan,
                                              "reason": "--use_gan with --lambda_gan 0.0"}
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4, default=str)

    stability_cfg = StabilityConfig(
        max_gradient_norm=args.max_gradient_norm,
        max_gradient_skip_ratio=args.max_gradient_skip_ratio,
        max_consecutive_bad_batches=args.max_consecutive_bad_batches,
        max_consecutive_unhealthy_epochs=args.max_consecutive_unhealthy_epochs,
        debug_numerics=args.debug_numerics,
    )

    trainer = TrainerG_v11(
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
        ema_rate=(args.trm_ema_rate if args.architecture == "recursive" else None),
        # ---- v15 (TrainerG_v11) ----
        on_class_collapse=args.on_class_collapse,                       # R0.3
        use_wavelengths=args.use_wavelengths,                           # R2.3
        sensor_range=(tuple(args.sensor_range) if args.sensor_range else None),
        keep_last_n=args.keep_last_n,                                   # R5.6
        eval_artifact_stride=args.eval_artifact_stride,                 # R5.6
    )
    trainer.dataloader_config = config["dataloader_config"]
    # R6.3 - so the per-item augmentation seed advances with the epoch. With
    # persistent_workers this cannot reach the forked workers; see
    # training/augmentation.AugmentedPatchDataset.
    if augmented_train_ds is not None:
        trainer.augmented_train_dataset = augmented_train_ds

    # Phase 3 - one more snapshot right before training starts.
    log_system_memory(trainer.logger, label="pre-training")

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume) + 1
        else:
            print(f"Warning: Specified resume path not found: {args.resume}. Starting fresh.", flush=True)

    stop_reason = trainer.fit(max_epochs=args.epochs, start_epoch=start_epoch)
    stop_str = str(getattr(stop_reason, "value", stop_reason)) if stop_reason is not None else ""

    # ------------------------------------------------------------------
    # Post-fit gates. These run even after an abort, because a run that
    # collapsed still owes the record a test_report.json saying so.
    # ------------------------------------------------------------------
    gate_results = {}
    if args.verify_best_checkpoint:                                      # R5.3, gate G5
        try:
            gate_results["G5"] = trainer.verify_best_checkpoint_reproduces()
        except Exception as e:
            print(f"WARNING: gate G5 check failed to run: {type(e).__name__}: {e}", flush=True)

    if args.eval_test != "off":                                          # R5.5, gate G6
        try:
            test_loader = _build_test_loader(
                args, data_dir=args.data_dir, num_classes=num_classes,
                storage_mode=args.dataset_storage, normalization=args.normalization,
                global_mean=global_mean, global_std=global_std,
                policy=val_policy, device=device, val_x_path=x_te)
            gate_results["G6"] = bool(trainer.evaluate_test_split(
                test_loader, num_classes=num_classes) is not None)
        except Exception as e:
            print(f"WARNING: test-split evaluation failed: {type(e).__name__}: {e}", flush=True)
            gate_results["G6"] = False

    if gate_results:
        with open(exp_dir / "gates.json", "w") as f:
            json.dump(gate_results, f, indent=2, default=str)

    if stop_str == "TRAINING_ABORTED_NUMERICAL_INSTABILITY":
        raise SystemExit(
            "TRAINING_ABORTED_NUMERICAL_INSTABILITY: see "
            f"{exp_dir}/numerical_failure/failure.json for diagnostics.")
    if stop_str == "TRAINING_ABORTED_REPRESENTATION_COLLAPSE":
        raise SystemExit(
            "TRAINING_ABORTED_REPRESENTATION_COLLAPSE: the validation split was assigned a single "
            f"class for {args.class_collapse_streak} consecutive epochs. See "
            f"{exp_dir}/collapse_failure/failure.json, then run "
            f"`pytest test_representation_sensitivity.py -q`.")


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
