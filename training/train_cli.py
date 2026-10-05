# -*- coding: utf-8 -*-
"""
training/train_cli.py
======================
The ONE command line of `train.py`: every flag any training entry point ever
had, plus `--profile`, which picks the defaults (`training/train_profiles.py`).

    parse(argv) -> Parsed(args, explicit, profile, parser)

`explicit` is the set of dests the user actually typed. The pipeline needs it
because `apply_safe_mode` / `apply_run_defaults` may only change a value the
user did not choose; it used to be computed by re-building each entry point's
parser a second and third time.

The flags are v15's parser (copied below from the frozen
`archive/train_example_v15.py`) plus the groups the v16-line entry points
added. Flags that meant the same thing under two names are one flag with both
spellings:

    --stop_on_metric_stall  == --stop_on_val_acc_stall   (optimal / original)
    --early_stopping        == --early_stopping_enabled  (original / optimal_recon)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Optional

from training.reconstruction_head_v2 import (
    resolve_out_activation, validate_out_activation_for_normalization,
)
from training.run_naming_v16 import infer_modality
from training.train_profiles import (
    DEFAULT_PROFILE, PROFILE_ALIASES, PROFILES, STAGES, TARGET_TOKEN_GRID, apply_profile,
    canonical_profile, peek,
)

_RUN_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")


def _valid_run_tag(value: str) -> str:
    if not _RUN_TAG_RE.match(value or ""):
        raise argparse.ArgumentTypeError(
            f"--run_tag must be alphanumeric with dashes and start with a letter or digit, got "
            f"{value!r}. It becomes part of a directory name, where '_' separates fields.")
    return value


# ============================================================================
# v15's parser and --safe_mode, verbatim from archive/train_example_v15.py
# (lines 132-133, 136-355, 358-590, 609-653; see tests/test_live_copies_match_archive.py)
# ============================================================================

EFFICIENT_FUSION_CHOICES = ["film", "gated", "cross_attention", "multiplicative", "residual",
                            "se_gate", "eca", "none"]


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="MedMamba-SS-TRM v15 ('Representation Collapse Remediation') training entry point")
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

    # TRM recursive variant (--architecture recursive only; see medmamba_ss_trm.MedMambaSSTRMConfig's
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
                         "CUDA/Triton kernel via medmamba_ss_trm.register_scan_backend().")
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
                         "medmamba_ss_trm.py's header was untested. Depends on R1.3.")
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
                         "medmamba_ss_trm._selective_scan_pure_pytorch (a PYTHON loop over the band "
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
                    trm_halting=False)


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


def _add_flags(p: argparse.ArgumentParser) -> None:
    """Everything the v16-line entry points added to v15's parser."""
    # A-2 - a third normalization mode. argparse has no public API to widen an
    # existing action's `choices`.
    for action in p._actions:
        if action.dest == "normalization":
            action.choices = ["per_sample_minmax", "global_zscore", "per_patch_zscore", "auto"]

    g = p.add_argument_group("run")
    g.add_argument("--profile", type=canonical_profile, choices=sorted(PROFILES),
                   default=DEFAULT_PROFILE,
                   help="which configuration supplies the defaults - see training/train_profiles.py. "
                        "A name ending in _norecon trains without reconstruction. "
                        + "; ".join(f"{k}: {v.summary}" for k, v in PROFILES.items())
                        + ". Old names still accepted: "
                        + ", ".join(f"{old} = {new}" for old, new in PROFILE_ALIASES.items()))
    g.add_argument("--run_tag", type=_valid_run_tag, default=None,
                   help="appended to the run directory name. Needed for any ablation that varies a "
                        "flag the name does not encode (--trm_n_improve, --recon_mode, "
                        "--no_use_wavelengths), or the arms differ only by timestamp.")
    g.add_argument("--stage", choices=sorted(STAGES), default="full",
                   help="debugging ladder of the OPTIMAL_DEFAULTS profiles (pad_ufes_best_norecon, "
                        "paper_recipe): 'fit' turns off every regulariser, the "
                        "class weighting and EMA; 'balance' adds the weighting back; 'full' is the "
                        "profile unchanged. Explicit flags override the stage.")
    g.add_argument("--target_token_grid", type=int, default=TARGET_TOKEN_GRID,
                   help=f"tokens per side the stem aims for when the profile resolves --patch_size "
                        f"from the input size (default {TARGET_TOKEN_GRID}). Cost is quadratic in it.")

    g = p.add_argument_group("stopping")
    g.add_argument("--early_stopping", "--early_stopping_enabled", dest="early_stopping",
                   choices=["on", "off"], default="off",
                   help="stop after --early_stop_patience epochs without a new best "
                        "--checkpoint_metric. Off, --early_stop_patience is recorded and ignored "
                        "(TrainerG_v12's rules have no patience rule). A patience of 0 or None "
                        "leaves it off either way.")
    g.add_argument("--stop_on_metric_stall", "--stop_on_val_acc_stall", dest="stop_on_metric_stall",
                   choices=["on", "off"], default="on",
                   help="stop after 5 epochs whose checkpoint metric EXACTLY equals the previous "
                        "epoch's. A one-class predictor does that before the class weighting takes "
                        "hold, and val_accuracy on a small split repeats all the time.")

    g = p.add_argument_group("schedule / normalization")
    g.add_argument("--lr_schedule", choices=["warmup_cosine", "constant"], default="warmup_cosine",
                   help="'warmup_cosine': step-axis linear warmup into cosine decay over "
                        "epochs x steps_per_epoch. 'constant': the MedMamba reference protocol.")
    g.add_argument("--global_stats", choices=["train_fit", "medmamba_fixed", "identity"],
                   default="train_fit",
                   help="--normalization global_zscore only. 'train_fit' fits per-channel mean/std "
                        "on the train split; 'medmamba_fixed' pins 0.5/0.5 (== torchvision "
                        "Normalize(.5,.5)); 'identity' pins 0/1.")

    g = p.add_argument_group("v16")
    g.add_argument("--drift_check_patches", type=int, default=8000,
                   help="gate G8: patches sampled per split for the post-normalization drift check")
    g.add_argument("--on_split_drift", choices=["abort", "warn", "off"], default="abort")
    g.add_argument("--min_patient_recall", type=float, default=None,
                   help="log a PATIENT_OUTLIER warning when any validation patient's recall_macro "
                        "drops below this while the checkpoint metric rises. None = off.")
    g.add_argument("--val_divergence_patience", type=int, default=3,
                   help="epochs of (worsening checkpoint metric + rising val_loss) before "
                        "VAL_DIVERGENCE. 0 disables.")
    g.add_argument("--scheduler_interval", choices=["epoch", "step"], default="step")
    g.add_argument("--warmup_steps", type=int, default=None,
                   help="linear warmup in optimizer steps. None = 3%% of total steps, floor 200.")
    g.add_argument("--recon_out_activation", choices=["linear", "sigmoid", "auto"], default="auto",
                   help="decoder output activation; 'auto' resolves from --normalization.")
    g.add_argument("--eval_group_aggregation", choices=["none", "mean_prob", "majority"],
                   default="none",
                   help="additionally report IMAGE-level test metrics when images_{split}.npy exists.")
    g.add_argument("--group_sidecar_dir", default=None,
                   help="directory holding groups_*.npy/captures_*.npy/images_*.npy "
                        "(default: next to X_val.npy)")
    g.add_argument("--patch_size", type=int, default=None,
                   help="stem stride. None = architecture preset. Pass 1 to keep full spatial "
                        "resolution on small patches.")
    g.add_argument("--g7_check", choices=["auto", "on", "off"], default="auto",
                   help="gate G7 (reconstruction gradient reaches the encoder); 'auto' runs it for "
                        "architecture=recursive with recon_mode=latent.")
    g.add_argument("--fast_loop", choices=["on", "off"], default="off",
                   help="one device sync per step and a fused EMA update "
                        "(training/trainerg_v12_fast.py). Training metrics bit-identical.")

    g = p.add_argument_group("reconstruction artifacts")
    g.add_argument("--recon_save_samples", choices=["on", "off"], default="on",
                   help="write reconstructed cubes and figures at the end of training.")
    g.add_argument("--recon_sample_per_class", type=int, default=3,
                   help="samples per class at evenly spaced SAM quantiles (3 = best/median/worst).")
    g.add_argument("--recon_rgb_bands", default=None,
                   help="three wavelengths in nm ('640,550,460') or band indices for the composite.")
    g.add_argument("--recon_render_png", choices=["on", "off"], default="on")
    g.add_argument("--recon_max_cubes", type=int, default=64,
                   help="hard cap on retained cubes.")


def build_parser(argv=None, profile: Optional[str] = None) -> argparse.ArgumentParser:
    """The full parser with `profile`'s defaults applied (peeked from `argv` when
    not given)."""
    p = build_arg_parser()
    p.description = "MedMamba-SS-TRM training - one run. See training/train_profiles.py for --profile."
    _add_flags(p)
    profile = canonical_profile(profile or peek("--profile", DEFAULT_PROFILE, argv=argv))
    if profile not in PROFILES:
        p.error(f"argument --profile: invalid choice: {profile!r} (choose from {sorted(PROFILES)})")
    apply_profile(p, profile, argv)
    return p


def explicitly_passed(p: argparse.ArgumentParser, argv=None) -> set:
    """Dests the user typed, with the same parser (a different parser would
    reject valid flags or miss aliases).

    `p._defaults` is cleared too. `set_defaults` writes there as well as onto
    the actions, and argparse copies it into the namespace after parsing, so
    the former entry points' version of this function reported every profile
    default as typed. The only consumer that noticed was `--safe_mode`, which
    then left a profile's amp / compile / loader settings alone."""
    for action in p._actions:
        action.default = argparse.SUPPRESS
    p._defaults.clear()
    known, _ = p.parse_known_args(argv)
    return set(vars(known))


@dataclass
class Parsed:
    args: argparse.Namespace
    explicit: set
    profile: str
    parser: argparse.ArgumentParser


def parse(argv=None) -> Parsed:
    p = build_parser(argv)
    args = p.parse_args(argv)
    if args.stage != "full" and not PROFILES[args.profile].stages:
        p.error(f"--stage applies to the profiles built on OPTIMAL_DEFAULTS "
                f"({sorted(k for k, v in PROFILES.items() if v.stages)}), not {args.profile!r}")
    explicit = explicitly_passed(build_parser(argv, args.profile), argv)
    return Parsed(args, explicit, args.profile, p)


# ============================================================================
# defaults resolved AFTER parsing (they need the modality)
# ============================================================================

def apply_run_defaults(args: argparse.Namespace, explicit: set) -> list:
    """`train_example_v16.apply_v16_defaults`: the recursive architecture by
    default, deep validation on HSI, `normalization auto` and
    `recon_out_activation auto` resolved. Returns the notes to print."""
    notes = []

    def _set(name, value, why):
        if name in explicit:
            return
        if getattr(args, name, None) != value:
            notes.append(f"{name}={value}  ({why})")
        setattr(args, name, value)

    _set("architecture", "recursive",
         "MedMamba-SS-TRM (TRM) is the default; pass --architecture split|fullchannel|efficient "
         "for the hierarchical backbones")
    if infer_modality(args.data_dir) == "hsi":
        _set("dataset_validation_level", "deep",
             "R6.5 (v15) - this dataset family had a physically-unreadable region during prep")
    if args.normalization == "auto":
        resolved = "per_patch_zscore" if infer_modality(args.data_dir) == "hsi" else "per_sample_minmax"
        args.normalization = resolved
        notes.append(f"normalization={resolved}  (A-2 - auto, by modality)")
    if args.recon_out_activation == "auto":
        args.recon_out_activation = resolve_out_activation(args.normalization)
        notes.append(f"recon_out_activation={args.recon_out_activation}  (R-3 - auto, by normalization)")
    else:
        validate_out_activation_for_normalization(args.recon_out_activation, args.normalization)
    return notes


def resolve(argv=None):
    """`parse` + every post-parse default the pipeline applies before it touches
    data - i.e. the `cli_args` a run would record. Used by the pipeline, by
    `run_experiments.py --dry_run`, and by the parity tests."""
    parsed = parse(argv)
    apply_safe_mode(parsed.args, parsed.explicit)
    notes = apply_run_defaults(parsed.args, parsed.explicit)
    typed = peek("--profile", argv=argv)
    if typed in PROFILE_ALIASES:
        notes.insert(0, f"profile={parsed.profile}  (--profile {typed} is its name before 2026-10-01)")
    return parsed, notes
