# -*- coding: utf-8 -*-
"""
train_example_v16_optimal.py
=============================
GMedMamba-R (`gmedmamba.GMedMambaRecursive`, the TRM variant) on
PAD-UFES-20, configured for the BEST ACHIEVABLE metrics rather than for
protocol fidelity.

Companion to `prepare_pad_ufes_20_optimal.py`. Sibling of
`train_example_v16_original.py`, which answers the opposite question - that
file makes GMedMamba-R differ from a published MedMamba run in as few places
as possible, and pays whatever it costs in accuracy to do so. This file
spends every one of those degrees of freedom on the metric.

This is a new ENTRY POINT that imports `train_example_v16.py` and passes it a
`v16.Overrides` value (parser, run name, model builder, trainer class) - see
`build_overrides` below. Before v17 S9 it REBOUND those names in v16's module namespace
instead; `install_overrides` remains as a deprecated shim, and
`train_example_v16_original.py` still uses that mechanism for the reason given there. `v16._main` is one long function; copying it would let
the two drift silently, which is the exact failure the v16 plan's freezing
rule exists to prevent. Nothing here is imported by v16, v15 or any frozen
`trainerg_*`.

Where the numbers below come from
---------------------------------
Every PAD-UFES-20 run in `experiments/` was read before choosing these. The
two regimes and their ceilings:

  data/pad_v6 (11x11 patches, 400/image, ~100k samples)
    best ever: balanced accuracy 0.2710, macro-F1 0.2345, test accuracy
    0.3072  (20260905_133922). Chance is 0.1667.
  data/pad_original (whole 224x224 images, 1,384 train)
    every run collapsed to the constant predictor: val accuracy exactly
    0.4000 = the majority-class rate, balanced accuracy exactly 0.1667,
    train loss 1.5036 against a prior-entropy floor of 1.4973
    (20260905_235026, 20260906_000618).

And the linear probes on the same two datasets
(`scripts/shallow_baseline_v16.py`, on disk next to each):

    input            probe  accuracy  balanced  macro-F1
    11x11 patches    flat     0.2128    0.2316    0.1846
    whole 224x224    flat     0.4735    0.3365    0.3252

Read together those say two different things, and this file responds to
both:

  * On 11x11 crops, GMedMamba-R at 0.2345 macro-F1 already BEATS the best
    linear probe at 0.1846. There is no optimisation problem there, only a
    data one - the crop is 0.24% of the frame and mostly does not contain
    the lesion. No configuration recovers that; the input has to change.
    Hence `prepare_pad_ufes_20_optimal.py --tiling whole`.
  * On whole images, GMedMamba-R at 0.1667 balanced accuracy LOSES to a
    logistic regression on raw pixels at 0.3365. That IS an optimisation
    problem, and it is what the defaults below fix.

So the target this file is built to clear is the whole-image linear probe:
**macro-F1 > 0.325, balanced accuracy > 0.337**, on a patient-grouped split.
The banner prints those bars at startup (via
`train_example_v16_original.print_trivial_baselines`, which reads whatever
`shallow_baseline.json` is next to the dataset) so epoch 1 already says
whether the run is on the right side of them.

The changes, and why each one
-----------------------------
Grouped by the failure each addresses. `v16 default` is what
`train_example_v16.py` would have used.

1. THE COLLAPSE (constant predictor on whole images)
   PAD-UFES-20 is 2,298 images over 6 classes with BCC at 845 and MEL at
   **52** - 15.7x imbalance inside the 70/15/15 train split, which holds 613
   BCC and 39 MEL. Under plain CE a weak learner's posterior is near the
   class prior, and the argmax of a
   near-prior posterior is the majority class for every input - which scores
   exactly the numbers quoted above.

     loss                focal_weighted   (v16: ce)
     focal_gamma         1.5              (v16: 2.0)
     class_weight_power  0.75             (v16: 1.0)
     sampler             none             (v16: none)

   `class_weight_power` is the important one. It was 0.5 here until the first
   real run, on the argument that full inverse-frequency weighting (1.0)
   over-corrects: `20260903_010639` reached balanced accuracy 0.2767 - the
   best of any pad_v6 run - while accuracy fell to 0.1815, and
   `training.losses.compute_class_weights` documents the same trade on the
   HSI dataset (power 1.00 -> macro-F1 0.8038, 0.75 -> 0.8474, 0.50 ->
   0.8391, 0.00 -> 0.6657). Both of those measurements come from datasets
   with 10^5 samples; neither transfers to 1,626 whole images, where the
   failure mode is not over-correction but no correction at all. What 0.5
   actually buys on THIS split:

       power  weights (ACK BCC MEL NEV SCC SEK)         BCC:MEL mass
       0.50   0.55 0.50 1.98 0.94 1.06 0.98                3.96x
       0.75   0.37 0.32 2.56 0.84 1.00 0.90                1.99x
       1.00   0.24 0.20 3.16 0.72 0.91 0.78                1.00x

   At 0.5, BCC still carries four times MEL's gradient mass - too little to
   pull a collapsing model off the majority class. 0.75 halves that while
   keeping the HSI evidence's best-measured value. `--class_weight_power 1.0`
   is the escalation if one class is still never predicted after ~30 epochs,
   and `scripts/tune_class_weight_power.py` re-fits it from a finished run's
   saved probabilities.

   `--sampler none` is deliberate and not an oversight. Stacking
   `moderate_oversample` ON TOP of a weighted loss double-counts the
   correction: `20260904_141142` did exactly that and landed at accuracy
   0.1645 / macro-F1 0.1768, worse on both axes than the weighted-loss-only
   run. Correct the objective or the sampling, not both.

2. THE SELECTION METRIC
     checkpoint_metric   f1_macro         (v16: f1_macro - unchanged)

   Kept, and called out because `train_example_v16_original.py` must use
   `val_accuracy` and that is what let a collapsed model be checkpointed as
   "best": on a split whose majority class is 40% of it, val_accuracy 0.4000
   IS the constant predictor. macro-F1 cannot be gamed that way.

3. THE NORMALIZATION
     normalization       global_zscore    (v16 auto for RGB: per_sample_minmax)

   `per_sample_minmax` (`training/normalization.py:72`) rescales each image
   by ITS OWN global min and max. On hyperspectral patches that is the right
   call - it is what makes a mis-exposed capture comparable to a normal one.
   On clinical dermatology photographs it deletes a diagnostic cue: absolute
   lesion darkness and erythema relative to surrounding skin are exactly what
   separates MEL from NEV, and a per-image contrast stretch normalises that
   difference away. `global_zscore` fits per-channel mean/std ONCE on the
   train split and applies that single affine everywhere - the fixed
   normalisation every ImageNet-lineage vision pipeline uses, for this
   reason. Every PAD run to date used `per_sample_minmax`.

     on_split_drift      warn             (v16: abort)

   Gate G8 compares post-normalization channel statistics across splits, and
   a fixed train-fitted affine is precisely the mode it can legitimately flag
   (it did so for the HSI dataset's patient 68). The report is still written
   to `split_drift_report.json` and still printed; it just does not kill a
   run before epoch 1. READ IT - a genuine drift finding means the split, not
   the gate, is wrong.

4. THE STEP BUDGET AND THE LEARNING RATE
     batch_size          32               (v16: 256)
     lr                  3e-4             (v16: 1e-4)
     epochs              200              (v16: 10)

   Whole images are ~270x fewer samples than the 11x11 grid, so an epoch is
   ~51 optimizer steps instead of 394. At the old batch 64 the protocol run
   got 3,300 steps total against the manuscript run's 15,760, and it visibly
   underfit - train accuracy 0.3432 after 16 epochs, barely above the 0.3689
   a constant predictor scores. Batch 32 x 200 epochs is ~10,200 steps, and
   1e-4 was never re-tuned after the batch size fell by 8x. 3e-4 with the
   step-axis warmup+cosine schedule v16 already defaults to (warmup = 3% of
   total steps) is the standard AdamW setting for a 0.45 M-parameter model
   at this batch size.

5. OVERFITTING 1,626 IMAGES
     augment_preset      custom (below)   (v16: none)
     classifier_dropout  0.1              (v16: preset)
     trm_dropout         0.0              (v16: 0.0 - unchanged)
     trm_ema_rate        auto, ~0.993     (v16: 0.999 - see below)
     weight_decay        0.05 + groups    (v16: 0.05 + groups - unchanged)

   Both dropout rates read lower than a first pass would set them, for the
   same reason: `RecursiveCore` is WEIGHT-SHARED and re-entered ~63 times per
   forward, so a rate inside it compounds. `trm_dropout` is 0.0 - the comment
   above `_STAGES` derives that in full. `classifier_dropout` is 0.1, the same
   rate `gmedmamba.ClassificationHead` hardcodes for the hierarchical
   architectures, so 'recursive' is not regularised on a different scale from
   its siblings for no stated reason. NEITHER RATE HAS BEEN SWEPT; 0.3 on the
   head is the obvious thing to try if the fit overshoots. Reach for
   augmentation and weight decay first - both apply once per forward rather
   than ~63 times.

   `trm_ema_rate` is resolved from steps/epoch by `resolve_ema_rate`, and the
   reason is the single most expensive defect this configuration has had.
   `EMAHelper` seeds its shadow with the RANDOM INITIALISATION and never
   bias-corrects it, and `TrainerG_v10._validate_one_epoch` validates the EMA
   copy - so every validation number is computed on a model that is `mu ** N`
   initialisation. At 51 steps/epoch a fixed 0.999 leaves 44% of the init in
   the validation model at epoch 16 and does not fall below 5% until epoch 59,
   which `--early_stop_patience 40` can cut short. The first real run showed
   exactly that: live train loss 0.8869, BELOW this objective's 0.9509
   prior-predictor floor - the model was learning - while the EMA copy scored
   balanced accuracy 0.1667 and predicted one class. A stale EMA and a
   collapsed model are indistinguishable in the epoch log, so
   `print_ema_horizon` now prints the contamination schedule at startup.

   The augmentation preset is the part that needed thought.
   `training.augmentation.PRESETS` has `light` and `medium`, and BOTH are
   actively harmful on 3-channel input: `medium` sets
   `band_dropout_prob=0.2` with `band_dropout_max_frac=0.08`, and
   `band_dropout` computes `n_drop = max(1, round(C * max_frac * rand))`
   (`training/augmentation.py:87`), which on C=3 is always 1. So `medium`
   zeroes an entire colour channel on 20% of samples, plus a contiguous
   channel run on another 10% via `spectral_mask_prob`. Those presets were
   written for 32-band HSI, where dropping one band of 32 is a mild
   perturbation. Destroying a third of the colour information is not a mild
   perturbation on a task whose signal IS colour.

   So this file uses `--augment_preset custom` with the spectral-destructive
   steps off and the geometry/photometry ones on:

     flip_h / flip_v / rotate90   0.5 each   dermatology has no canonical
                                             orientation; free 8x diversity
     crop_scale_max_frac          0.3        scale invariance; note
                                             `spatial_crop_scale` resizes
                                             with nearest-neighbour, so keep
                                             this moderate
     spectral_scale_range         0.15       per-channel gain = contrast and
                                             white-balance jitter
     spectral_offset_std          0.05       per-channel shift = colour cast
     spectral_noise_std           0.02       sensor noise
     band_dropout_prob            0.0        OFF - see above
     spectral_mask_prob           0.0        OFF - see above

   These act on the ALREADY-NORMALIZED tensor (`AugmentedPatchDatasetV16`
   wraps `NpyDatasetV16`), so in z-score space `spectral_scale` reads as a
   per-channel contrast jitter and `spectral_offset` as a colour cast. Both
   are the intended semantics; the magnitudes are chosen for z-units, not
   for [0,1] units.

6. STOPPING RULES THAT FIRE ON A COLLAPSED FIRST EPOCH
     val_divergence_patience  0             (v16: 3)
     on_class_collapse        warn          (v16: abort)
     class_collapse_streak    10            (v16: 3)
     early_stop_patience      40            (v16: None)
     VAL_ACC_STALLED          suppressed    (v16: active)

   A class-weighted objective on a 328-image validation split makes
   validation LOSS rise while macro-F1 is still improving - that is what the
   weighting does, not a divergence - and E-1's rule stopped
   `20260905_022614` at epoch 12 of 40 for it. The class-collapse abort fires
   on the first few epochs, which legitimately predict one class before the
   weighting takes hold. And `TrainerG_v12._check_stopping_rules` stops after
   5 epochs whose checkpoint metric is EXACTLY equal to the previous epoch's:
   a model that predicts one class for its first five epochs produces
   bit-identical macro-F1 each time, so that rule would end this run at epoch
   5 every time. It is suppressed through the same
   `suppressed_stop_reasons` funnel `train_example_v16_original.py` added,
   and `--stop_on_metric_stall on` puts it back.

   `early_stop_patience 40` is long on purpose: macro-F1 on 328 validation
   images whose MEL cell holds SIX of them is noisy enough that a patience of
   10-15 would routinely stop on that noise. For the same reason, read
   balanced accuracy and the per-class support next to any macro-F1 this run
   reports - the MEL column is six images wide.

7. COST
     patch_size          auto (see below) (v16: preset 1)
     amp                 bf16             (v16: bf16 - unchanged)
     recon_mode          none             (v16: latent)

   The recursive core never downsamples, so the stem stride sets the token
   count that is then carried through all ~63 core applications. At 224 with
   stride 1 that is 50,176 tokens per sample, which does not fit anywhere.
   `--patch_size auto` targets a `--target_token_grid` token grid (default
   28): stride 8 on a 224 frame (784 tokens), stride 4 on the 112px tiles
   `--tiling mil9` produces, and stride 1 on an 11x11 grid, which is what
   every published GMedMamba-R run used.

   WHAT THIS ACTUALLY COSTS ON THE CARD IN THIS MACHINE. The estimate in
   `train_example_v16_original.print_trm_cost` is a linear extrapolation from
   one fp32 OOM; the runs in `experiments/` have since measured the real
   thing, and `history.json`'s `GPU_memory_MB` is the column to read:

       dataset / grid            batch  amp    GPU_memory_MB   of 16,651
       11x11  (121 tok)  HSI      256   bf16          874        5.2%
       11x11  (121 tok)  RGB       32   bf16          201        1.2%
       28x28  (784 tok)  PAD       32   bf16        1,432        8.6%
       28x28  (784 tok)  PAD       64   bf16        3,676       22.1%

   So on an RTX 5060 Ti (16 GB) THIS CONFIGURATION USES 8.6% OF THE CARD and
   leaves ~15 GB idle. Net of a ~200 MB floor (context + weights + optimizer;
   the 0.45 M-parameter model is negligible), activations cost ~51 KB per
   token-row, where a token-row is one token of one sample. That constant
   predicts, at batch 32:

       --target_token_grid 28  ->  stride 8,  784 tok  ->  ~1.4 GB   (current)
       --target_token_grid 56  ->  stride 4, 3136 tok  ->  ~5.3 GB   fits
       --target_token_grid 112 ->  stride 2,12544 tok  ->  ~20.5 GB  OOM

   i.e. 28 is NOT the ceiling this card imposes - it is the ceiling a batch
   of 64 imposes, and this file uses 32. The honest reading is that the
   default trades spatial resolution it does not have to trade: a stride of 8
   discards 98% of the pixels before the core sees anything, on a task whose
   open problem is that the input does not carry the lesion.

   It is left at 28 because the cost is TIME, not memory, and the run has to
   finish: tokens scale the step cost roughly linearly, so `--target_token_grid
   56` turns the measured ~0.9 s/step into ~3.5 s/step and a 200-epoch PAD run
   from ~2.5 h into ~10 h. Raise it when there is a night to spend - it is the
   single most promising unexplored axis on this dataset - and read
   `[gpu-budget]` in the preflight, which prints the measured free VRAM and
   the estimate above against it before epoch 1.

   `recon_mode none` turns off v16's auxiliary latent reconstruction decoder.
   It is not a judgement that the decoder is useless - the whole R-1..R-7
   half of the v16 plan exists because it was silently broken and is now
   fixed - but no PAD run has ever measured it helping CLASSIFICATION, and it
   spends capacity and gradient budget that 1,626 images cannot spare.
   `--recon_mode latent --lambda_mse 0.1` is the single most interesting
   ablation to run second, because auxiliary reconstruction is exactly the
   kind of regulariser a dataset this small should benefit from.

What this file does NOT claim
-----------------------------
These defaults are derived from the run history above and from the
mechanisms in the code. Only ONE full run has informed them - a
`data/pad_optimal` run stopped at epoch 16, which is what corrected
`trm_ema_rate` and `class_weight_power` above and is the source of every
epoch-16 number quoted in this docstring. Nothing here is a swept optimum.
The knobs most likely to repay an actual sweep, in order:

    --class_weight_power   0.75 / 1.0        (escalate if a class is never
                                              predicted past ~epoch 30)
    --lr                   1e-4 / 3e-4 / 1e-3
    --patch_size           4 (at --batch_size 16) / 8 / 16
    --recon_mode latent --lambda_mse 0.1     (auxiliary regulariser)

Read `[ema]`, `[budget]` and `[baseline]` in the preflight before concluding
anything from an epoch log: balanced accuracy pinned at 0.1667 means a
one-class predictor, and that is a stale EMA at least as often as it is a
real collapse.

Usage
-----
    python prepare_pad_ufes_20_optimal.py \\
        --root ../Dataset/PAD-UFES-20 --out_dir ./data/pad_optimal --num_workers 8
    python scripts/shallow_baseline_v16.py --data_dir ./data/pad_optimal
    python train_example_v16_optimal.py --data_dir ./data/pad_optimal

    # patch-MIL variant, image-level test metrics
    python train_example_v16_optimal.py --data_dir ./data/pad_optimal_mil9 \\
        --eval_group_aggregation mean_prob

Any v16/v15 flag still works and overrides these defaults; the run's
`config.json` records what was actually used, and the banner prints every
default the command line moved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import train_example_v16 as v16
# Importing v16_original is side-effect free (it only CAPTURES v16's
# module-level names at import time; nothing is rebound until its own
# `install_overrides` runs, which this file never calls). Three of its
# preflight printers and its stopping-rule/gate-G5 trainer subclass are
# reused verbatim rather than reimplemented.
from train_example_v16_original import (
    TrainerG_v12Original, print_trivial_baselines, print_step_budget, print_trm_cost,
)
from training.trainerg_v4 import StopReason

_V16_BUILD_ARG_PARSER = v16.build_arg_parser
_V16_BUILD_RUN_NAME = v16.build_run_name
_V16_BUILD_MODEL = v16.build_model


# ============================================================================
# the configuration, as data
# ============================================================================

# `dest -> value`, applied as argparse DEFAULTS so each stays overridable on
# the command line and `v16.explicitly_passed()` still reports the user's own
# flags as explicit. Every entry is justified in the module docstring under
# the numbered heading named in its comment.
OPTIMAL_DEFAULTS = {
    # --- (1) the collapse: objective ------------------------------------
    "loss": "focal_weighted",
    "focal_gamma": 1.5,
    "class_weight_power": 0.75,
    "sampler": "none",

    # --- (2) model selection --------------------------------------------
    "checkpoint_metric": "f1_macro",

    # --- (3) normalization ----------------------------------------------
    "normalization": "global_zscore",
    "on_split_drift": "warn",

    # --- (4) step budget / LR -------------------------------------------
    "batch_size": 32,
    "lr": 3e-4,
    "epochs": 200,
    "weight_decay": 0.05,
    "weight_decay_groups": True,
    "scheduler_interval": "step",
    "warmup_steps": None,               # -> 3% of total steps, floor 200
    "max_gradient_norm": 1.0,

    # --- (5) regularisation ---------------------------------------------
    "augment_preset": "custom",
    "flip_h_prob": 0.5,
    "flip_v_prob": 0.5,
    "rotate90_prob": 0.5,
    "crop_scale_max_frac": 0.3,
    "spectral_scale_range": 0.15,
    "spectral_offset_std": 0.05,
    "spectral_noise_std": 0.02,
    "band_dropout_prob": 0.0,           # OFF on RGB: zeroes a whole channel
    "spectral_mask_prob": 0.0,          # OFF on RGB: same
    "classifier_dropout": 0.1,
    "trm_dropout": 0.0,       # NOT 0.1: it fires ~63x per forward, see `_STAGES`
    "drop_path_rate": 0.0,
    "trm_drop_path": 0.0,
    # trm_ema_rate is NOT here: it is resolved from steps/epoch in
    # `resolve_ema_rate` and applied by `build_arg_parser`, for the reason
    # written there. A fixed 0.999 is wrong by ~8x at this batch size.

    # --- (6) stopping rules ---------------------------------------------
    "early_stop_patience": 40,
    "val_divergence_patience": 0,
    "on_class_collapse": "warn",
    "class_collapse_streak": 10,

    # --- (7) cost / architecture ----------------------------------------
    "architecture": "recursive",
    "trm_dim": 128,
    "trm_mixer": "mlp",
    "trm_halting": False,
    "amp": "bf16",
    "recon_mode": "none",
    "lambda_mse": 0.0,
    "lambda_sam": 0.0,
    "lambda_gan": 0.0,

    # --- (8) throughput (v17 S2/S3; measured, see plan/v17_baseline.md) ---
    # None of these three was here before v17, so the PAD runs went through THIS
    # "best-metrics" entry point and still ran eager, on the slow loader, writing
    # all 10 matplotlib curves every epoch - while the HSI runs got the good
    # settings because someone typed them on a bare train_example_v16.py command
    # line. The settings were never the problem; their absence from the defaults was.
    "loader_mode": "performance",       # balanced = 2 workers, prefetch 1, pin_memory
                                        # OFF (which makes non_blocking=True a no-op)
                                        # and workers re-forked every 51-step epoch
    "eval_artifact_stride": 5,          # 1 wrote 64 confusion matrices for 63 epochs
    "fast_loop": "on",                  # v17 S12 - one device sync per step instead of ~17
                                        # and a fused EMA update. Training metrics are
                                        # bit-identical (test_fast_loop_v17.py); the EMA
                                        # shadow reassociates (~1e-8/300 steps), so pass
                                        # --fast_loop off to byte-compare against an
                                        # existing run. Same on/off split as --compile.
    "compile_model": "on",              # 437.0 -> 266.7 ms/step on PAD (1.64x) and
                                        # 1,428 -> 1,110 MB: Inductor fuses the six-op
                                        # _rms_norm_lastdim chain that runs twice per
                                        # block, 126 blocks per forward. ~1.7e-7 numeric
                                        # drift vs eager; --compile off restores it.

    # --- reporting ------------------------------------------------------
    "eval_test": "best",
    "verify_best_checkpoint": True,
    "train_subsample_frac": 1.0,
    "val_subsample_frac": 1.0,
    "seed": 42,
}

# --------------------------------------------------------------------------
# `--stage` - the debugging ladder
# --------------------------------------------------------------------------
# Regularising a model that cannot yet FIT is backwards, and that is what the
# `full` defaults were doing: the first runs on `data/pad_optimal` reached
# train accuracy 0.12-0.17 - at or BELOW the 1/6 chance rate, and far below
# the 0.31 a majority-class predictor scores - while a logistic regression on
# the network's own 28x28 input scored balanced accuracy 0.3844. A model that
# loses to a linear probe on its own input has an optimisation problem, and
# no amount of class weighting or augmentation addresses one.
#
# So the ladder: prove the model can fit, THEN correct the imbalance, THEN
# regularise. Each stage overrides `OPTIMAL_DEFAULTS`, and any explicit flag
# still overrides the stage.
_STAGES = {
    # Can the architecture memorise this data at all? Every regulariser and
    # every imbalance correction off, so train accuracy is a clean readout.
    # It should reach well above 0.31 within ~30 epochs; if it does not, the
    # problem is the model or the learning rate, and nothing further down the
    # ladder will help.
    "fit": {
        "loss": "ce", "class_weight_power": 0.0, "sampler": "none",
        "augment_preset": "none", "classifier_dropout": 0.0, "trm_dropout": 0.0,
        "drop_path_rate": 0.0, "trm_drop_path": 0.0,
        "trm_ema_rate": 0.0,          # validate the LIVE weights, no EMA lag
        "weight_decay": 0.0, "recon_mode": "none",
        "lambda_mse": 0.0, "lambda_sam": 0.0,
        "epochs": 40, "early_stop_patience": 0,
    },
    # Fitting is established; now make it predict more than one class.
    # Still no augmentation and no dropout, so a change here is attributable
    # to the class weighting alone.
    "balance": {
        "augment_preset": "none", "classifier_dropout": 0.0, "trm_dropout": 0.0,
        "trm_ema_rate": 0.0, "epochs": 60,
    },
    # `OPTIMAL_DEFAULTS` unchanged - regularisation and EMA back on.
    "full": {},
}

# Why `trm_dropout` is 0.0 in every stage including `full`.
# `cfg.trm_dropout` is dropout inside the GatedMLPs of `RecursiveCore.layers`
# (`gmedmamba.py:286`, applied at `gmedmamba.py:614`). Those layers are the
# WEIGHT-SHARED core, re-entered `trm_deep_supervision_steps * trm_n_improve *
# (trm_n_latent + 1)` = ~63 times per forward pass. A rate of 0.1 there is not
# 10% dropout, it is 10% applied 63 times in series. This is exactly the trap
# `train_example_v16_original.py` documents for `trm_drop_path` - MedMamba
# spreads 0.1 over ~10 blocks applied ONCE - and every published GMedMamba-R
# run used 0.0 for both. Regularise with `--classifier_dropout`, weight decay
# and augmentation, all of which apply once per forward.


# `--patch_size auto` aims the stem at this many tokens per side.
#
# 28 is a THROUGHPUT choice, not a memory ceiling - see heading (7). Measured
# on the RTX 5060 Ti this repository runs on, a 28x28 grid at batch 32 peaks at
# 1,432 MB of 16,651 (8.6%), and 56x56 is projected at ~5.3 GB, which fits.
# What 56 does not fit is the clock: token count scales step time roughly
# linearly, so it turns a ~2.5 h PAD run into ~10 h. `--target_token_grid`
# makes that a decision the command line takes rather than a constant.
_TARGET_TOKEN_GRID = 28

# Activation cost per token-row (one token of one sample), in MB, at bf16.
# Solved from the measured PAD point: (1432 MB - 200 MB floor) / (784 * 32).
# Used only to print an estimate in the preflight; nothing depends on it being
# exact, and it is deliberately fitted to the LARGEST measured grid, since
# that is the regime the warning is for.
_MB_PER_TOKEN_ROW_BF16 = (1432.0 - 200.0) / (784 * 32)
_VRAM_FLOOR_MB = 200.0


def _input_side(data_dir: str) -> Optional[int]:
    """H of `X_train.npy`, from the .npy header alone. None, never an
    exception: this feeds a default, and a missing file has to reach
    `v16._main`'s own error handling rather than dying in the parser."""
    try:
        shape = np.load(Path(data_dir) / "X_train.npy", mmap_mode="r").shape
        return int(shape[1]) if len(shape) == 4 else None
    except Exception:
        return None


def _peek_data_dir(argv=None) -> Optional[str]:
    """`--data_dir` off the command line before the real parser exists. A
    throwaway parser rather than a string scan, so `--data_dir=x`,
    abbreviations and `--` behave as argparse says they should."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--data_dir", default=None)
    known, _rest = peek.parse_known_args(argv)
    return known.data_dir


def _peek_target_grid(argv=None) -> int:
    """`--target_token_grid` before the real parser exists, for the same
    reason `_peek_data_dir` does: it sets `patch_size`, which is a DEFAULT."""
    return _peek_int("--target_token_grid", _TARGET_TOKEN_GRID, argv)


def resolve_patch_size(data_dir: Optional[str], target_grid: int = _TARGET_TOKEN_GRID
                        ) -> Optional[int]:
    """Stem stride giving a token grid no finer than `target_grid` per side.

    The recursive core does not downsample, so whatever grid the stem emits
    is held at full resolution through all ~63 of its applications and the
    working set is (tokens x batch). That makes the stride a cost knob with
    quadratic leverage, and the right value a function of the INPUT SIZE,
    which differs per prep:

        224 (whole clinical image)  -> 8   ->  28x28 =  784 tokens
        112 (--tiling mil9/mil4)    -> 4   ->  28x28 =  784 tokens
         11 (data/pad_v6 patches)   -> 1   ->  11x11 =  121 tokens

    The 11x11 case reproducing stride 1 matters: that is the preset every
    published GMedMamba-R run used, so pointing this entry point at an old
    dataset does not silently change the architecture as well as the recipe.
    Returns None (= keep the architecture preset) when the side is unknown.
    """
    side = _input_side(data_dir) if data_dir else None
    if side is None:
        return None
    return max(1, side // target_grid)


# EMA horizon, in EPOCHS. `gmedmamba_ema.EMAHelper` seeds its shadow with the
# RANDOM INITIALISATION and has no bias correction (`gmedmamba_ema.py:38`), so
# after N optimizer steps the init still carries `mu ** N` of the validation
# model's weight. Three epochs puts that below 5% by epoch ~10 while still
# averaging over enough steps to be worth having.
_EMA_HORIZON_EPOCHS = 3.0
_EMA_RATE_BOUNDS = (0.90, 0.9995)


def _peek_int(flag: str, default: int, argv=None) -> int:
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument(flag, type=int, default=default)
    known, _rest = peek.parse_known_args(argv)
    return getattr(known, flag.lstrip("-"))


def _peek_float(flag: str, default: float, argv=None) -> float:
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument(flag, type=float, default=default)
    known, _rest = peek.parse_known_args(argv)
    return getattr(known, flag.lstrip("-"))


def _split_size(data_dir: Optional[str], split: str) -> Optional[int]:
    """len(y_<split>.npy) from the header alone. None, never an exception:
    this feeds a default, and a missing file has to reach `v16._main`'s own
    error handling rather than dying in the parser."""
    if not data_dir:
        return None
    try:
        return int(len(np.load(Path(data_dir) / f"y_{split}.npy", mmap_mode="r")))
    except Exception:
        return None


def _train_size(data_dir: Optional[str]) -> Optional[int]:
    """len(y_train.npy) from the header alone. None, never an exception."""
    return _split_size(data_dir, "train")


def steps_per_epoch(data_dir: Optional[str], batch_size: int,
                     subsample_frac: float = 1.0) -> Optional[int]:
    """Optimizer steps in ONE epoch - over what `--train_subsample_frac` leaves
    to iterate, not over the full split.

    `subsample_frac` is not cosmetic. `resolve_ema_rate` divides by this number
    to hit a fixed averaging horizon in EPOCHS, so feeding it the full-split
    count for a subsampled run overstates steps/epoch by `1/frac` and leaves
    `mu` that much too slow - the exact failure `resolve_ema_rate`'s own
    docstring documents, re-armed by a different route.
    `train_example_v16.py:679` derives the real number the same way, from
    `train_subsample_info["kept"]`.
    """
    n = _train_size(data_dir) if data_dir else None
    if n is None:
        return None
    kept = max(1, int(round(n * max(0.0, min(1.0, subsample_frac)))))
    return max(1, -(-kept // max(1, batch_size)))


# Epoch SIZE, in patches. The HSI corpus is 2,452,086 training patches drawn from
# 499 captures of 35 patients (`captures_train.npy`, `groups_train.npy`) - about
# 4,914 patches per cube - so a full-split epoch shows the model ~500 distinct
# scenes several thousand times each, and costs 1h49m against the 11m an epoch
# at 0.1 costs. Every HSI run worth keeping was launched with
# `--train_subsample_frac 0.1` typed on the command line; that is what
# `sub01_vsub01` in the run directory names records. `OPTIMAL_DEFAULTS` was
# tuned on PAD-UFES-20, whose splits are smaller than both targets below, so
# moving the HSI work to this entry point silently reinstated 1.0 and turned a
# 4-hour run into a 34-hour one.
#
# A target COUNT rather than a fraction, for the reason `resolve_ema_rate`
# exists: a fraction is only right for the corpus it was measured on, and
# re-prepping at a different size would quietly change the epoch budget again.
_TARGET_TRAIN_PATCHES = 250_000     # rank-1 HSI ran 245,209 (0.1 of 2,452,086)
_TARGET_VAL_PATCHES = 33_000        # v15 R3.1: "a stratified 20-40k subset has
                                    # ample statistical power at these sizes"


def resolve_subsample_frac(n: Optional[int], target: int) -> float:
    """Fraction of a split that makes one epoch ~`target` samples.

    1.0 whenever the split is already at or below `target`, so every PAD
    configuration keeps its current full-split behaviour and only a corpus
    that is actually oversized gets cut. `--train_subsample_mode per_epoch`
    (v15's default, and still in force here) redraws that fraction every
    epoch, so nothing is permanently discarded.
    """
    if not n or n <= target:
        return 1.0
    return round(target / n, 4)


def resolve_ema_rate(data_dir: Optional[str], batch_size: int,
                      horizon_epochs: float = _EMA_HORIZON_EPOCHS,
                      subsample_frac: float = 1.0) -> float:
    """EMA decay giving an averaging horizon of `horizon_epochs`.

    This is the defect the first real run on `data/pad_optimal` exposed, and
    it is worth stating exactly, because it looks like a model failure and is
    not one. `EMAHelper.update` is
    `shadow = mu*shadow + (1-mu)*param` with `shadow` REGISTERED FROM THE
    INITIAL WEIGHTS and no bias correction, and `TrainerG_v10._validate_one_epoch`
    validates the EMA copy, not the live model
    (`training/trainerg_v10.py:270-273`). So every validation number is
    computed on a model that is `mu ** N` random initialisation.

    At 1,626 training images and batch 32 that is 51 steps/epoch, and the
    fixed 0.999 this file used to pass - inherited from runs that had 394
    steps/epoch - leaves:

        epoch  1: 95.0% init      epoch 25: 27.9% init
        epoch 16: 44.2% init      epoch 50:  7.8% init

    i.e. validation is not a measurement of the trained model until roughly
    epoch 59, and `--early_stop_patience 40` can end the run before that ever
    happens. Observed directly: at epoch 16 the live model's train loss was
    0.8869, BELOW the 0.9509 prior-predictor floor of this objective - it was
    learning - while the EMA copy scored balanced accuracy exactly 0.1667 and
    predicted one class.

    Tying `mu` to steps/epoch rather than hardcoding it makes the horizon a
    fixed number of EPOCHS at any batch size or dataset length, so changing
    `--batch_size` cannot silently re-break this.

    `subsample_frac` closes the other route to the same break. An epoch is
    `--train_subsample_frac` of the split, so on the HSI corpus at 0.1 the
    real steps/epoch is 977 and not 9,579; passing the full-split count would
    make the horizon ten times too long. Both callers pass the frac they
    resolved - see `build_arg_parser` and `config_deviations`.
    """
    spe = steps_per_epoch(data_dir, batch_size, subsample_frac)
    if spe is None:
        return 0.99
    lo, hi = _EMA_RATE_BOUNDS
    return float(min(hi, max(lo, 1.0 - 1.0 / max(1.0, horizon_epochs * spe))))


# ============================================================================
# override 1 - the parser
# ============================================================================

def build_arg_parser(argv=None) -> argparse.ArgumentParser:
    """v16's parser with `OPTIMAL_DEFAULTS` applied and one flag added.

    Rebound onto `v16.build_arg_parser`, which is what both `v16._main` and
    `v16.explicitly_passed` call. Neither passes `argv`, so in normal use the
    `--data_dir` peek reads `sys.argv` - the same command line `parse_args`
    is about to read. `argv` exists so the resolution is testable without
    mutating `sys.argv`.
    """
    p = _V16_BUILD_ARG_PARSER()
    p.description = ("GMedMamba-R (TRM) on PAD-UFES-20, configured for best achievable metrics "
                     "- see the module docstring for the evidence behind each default")

    unknown = set(OPTIMAL_DEFAULTS) - {a.dest for a in p._actions}
    if unknown:
        # A silently-ignored key would produce a run that claims this
        # configuration while not using it - fail at import instead.
        raise RuntimeError(f"OPTIMAL_DEFAULTS names flag(s) v16's parser does not have: "
                            f"{sorted(unknown)}")
    p.set_defaults(**OPTIMAL_DEFAULTS)
    data_dir = _peek_data_dir(argv)
    p.set_defaults(patch_size=resolve_patch_size(data_dir, _peek_target_grid(argv)))

    # Epoch BUDGET first, then the EMA horizon that depends on it - in that
    # order, because `resolve_ema_rate` needs the frac the user will actually
    # run with. `_split_size(..., "val")` falls back to the test split for the
    # same reason `train_example_v6.discover_data:166-170` does: a dataset
    # prepped without X_val.npy validates on X_test.npy.
    train_frac = resolve_subsample_frac(_split_size(data_dir, "train"), _TARGET_TRAIN_PATCHES)
    val_frac = resolve_subsample_frac(
        _split_size(data_dir, "val") or _split_size(data_dir, "test"), _TARGET_VAL_PATCHES)
    p.set_defaults(train_subsample_frac=train_frac, val_subsample_frac=val_frac)
    p.set_defaults(trm_ema_rate=resolve_ema_rate(
        data_dir, _peek_int("--batch_size", OPTIMAL_DEFAULTS["batch_size"], argv),
        subsample_frac=_peek_float("--train_subsample_frac", train_frac, argv)))

    # `--stage` is declared below but has to be READ here, because it changes
    # defaults the parser is still being built with.
    stage_peek = argparse.ArgumentParser(add_help=False)
    stage_peek.add_argument("--stage", default="full")
    p.set_defaults(**_STAGES.get(stage_peek.parse_known_args(argv)[0].stage, {}))

    g = p.add_argument_group("optimal")
    g.add_argument("--stage", choices=sorted(_STAGES), default="full",
                    help="debugging ladder. 'fit' turns off every regulariser, the class "
                         "weighting and EMA, so train accuracy is a clean readout of whether the "
                         "model can learn this data at all - run it FIRST, and expect train "
                         "accuracy well above the 0.31 majority rate within ~30 epochs. "
                         "'balance' adds the class weighting back. 'full' (default) is the "
                         "complete configuration. Explicit flags override the stage.")
    g.add_argument("--target_token_grid", type=int, default=_TARGET_TOKEN_GRID,
                    help=f"tokens per side the stem aims for when --patch_size is left at auto "
                         f"(default {_TARGET_TOKEN_GRID}). The recursive core never downsamples, so "
                         f"this sets the working set for all ~63 of its applications: cost is "
                         f"quadratic in this number and so is the spatial detail the model can "
                         f"see. Measured on a 16 GB card at --batch_size 32, 28 costs ~1.4 GB and "
                         f"56 is projected at ~5.3 GB - the limit at 28 is RUN TIME (~4x), not "
                         f"memory. See the module docstring, heading 7.")
    g.add_argument("--stop_on_metric_stall", choices=["on", "off"], default="off",
                    help="TrainerG_v12._check_stopping_rules stops after 5 epochs whose "
                         "checkpoint metric EXACTLY equals the previous epoch's. A model that "
                         "predicts one class for its first five epochs - which a 15.7x imbalanced "
                         "6-class split does before the class weighting takes hold - produces "
                         "bit-identical macro-F1 each time and would end the run at epoch 5. "
                         "'off' suppresses that one rule; --early_stop_patience still applies.")
    return p


# ============================================================================
# override 2 - the trainer
# ============================================================================

class _TwoTupleLoader:
    """A DataLoader view that yields `(x, y)` from `(x, y, norm_stats)` batches.

    `train_example_v16.py:411` sets `return_norm_stats = args.recon_mode !=
    "none"`, so with any reconstruction mode the datasets emit 3-tuples. Only
    ONE of the two training loops can consume those:

      * `TrainerG_v12._train_one_epoch_deep_supervision` (v12:454) reads
        `len(batch) == 3` and denormalizes with the stats - but
        `TrainerG_v10._train_one_epoch` dispatches to it only when
        `_deep_supervision_base()` is not None, i.e. architecture='recursive'.
      * every other architecture falls back to `TrainerG_v9._train_one_epoch`,
        whose `for x, y in self.train_loader` (v9:102) raises
        `ValueError: too many values to unpack (expected 2)` on the first batch.

    So `--architecture split --recon_mode latent` died at epoch 1 even though
    the reconstruction itself is fully supported there:
    `ReconWrapperV2.forward` has an explicit non-recursive branch
    (`training/reconstruction_head_v2.py:116-122`) and `TrainerG_v9`'s loop
    already handles a `(logits, x_recon)` tuple and computes the MSE/SAM terms.
    Only the batch arity is wrong.

    Rather than copy v9's 100-line loop to add three lines of unpacking - the
    drift this file's freezing rule exists to prevent - the loader is wrapped
    for the duration of the training epoch only. VALIDATION keeps the 3-tuples:
    `TrainerG_v12._validate_one_epoch_impl` handles them and uses the stats to
    report reconstruction error in raw units.
    """

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            yield (batch[0], batch[1]) if len(batch) > 2 else batch

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        # .dataset, .sampler, .batch_size, ... - anything the trainer reads off
        # the loader still resolves to the real one.
        return getattr(self._loader, name)


class TrainerG_v12Optimal(TrainerG_v12Original):
    """`TrainerG_v12Original` for its two mechanisms, with no protocol
    semantics attached:

      * `suppressed_stop_reasons`, which filters `_trigger_stop` - the single
        two-line funnel every stopping rule goes through
        (`training/trainerg_v4.py:647`) - so one rule can be neutralised
        without copying `_check_stopping_rules` and without touching the
        rules this run does want (`--early_stop_patience`, the numerical
        instability aborts).
      * the gate-G5 checkpoint-metric alias, which is inert here:
        `_CHECKPOINT_METRIC_ALIASES` only rewrites `val_accuracy`, and this
        file checkpoints on `f1_macro`, whose name is the same in
        `metrics["classification"]` and in the epoch metrics dict.

    Set as a class attribute by `install_overrides` from
    `--stop_on_metric_stall`.
    """

    suppressed_stop_reasons = frozenset({StopReason.VAL_ACC_STALLED})

    _warned_norm_stats = False

    def _train_one_epoch(self):
        """`TrainerG_v10._train_one_epoch`, with the norm-stats element removed
        from the batches before the non-recursive fallback loop sees them.
        See `_TwoTupleLoader` for why this is needed and why it is done here."""
        if self._deep_supervision_base() is not None:
            return super()._train_one_epoch()          # recursive: v12 unpacks 3-tuples itself
        real_loader = self.train_loader
        if not TrainerG_v12Optimal._warned_norm_stats:
            TrainerG_v12Optimal._warned_norm_stats = True
            self.logger.info(
                "[recon] architecture is not 'recursive', so training runs TrainerG_v9's loop, "
                "which takes (x, y) batches only. The norm_stats element the reconstruction modes "
                "add is dropped for the TRAINING epoch: train MSE/SAM are measured in NORMALIZED "
                "units. Validation is unaffected - it keeps the stats and reports raw-unit "
                "reconstruction error.")
        self.train_loader = _TwoTupleLoader(real_loader)
        try:
            return super()._train_one_epoch()
        finally:
            self.train_loader = real_loader

    def _trigger_stop(self, reason):
        """`TrainerG_v12Original._trigger_stop` logs its suppression as "it has
        no analogue in the reference protocol (see --stop_on_val_acc_stall)".
        Neither clause is true here - this run follows no reference protocol
        and its flag is `--stop_on_metric_stall` - so say the right thing
        rather than inherit a message that sends the reader to a flag this
        entry point does not have."""
        if reason in self.suppressed_stop_reasons:
            self.logger.info(
                f"[stopping rule] {getattr(reason, 'value', reason)} suppressed: the checkpoint "
                f"metric repeating EXACTLY for 5 epochs is what a one-class predictor does before "
                f"the class weighting takes hold, not evidence of convergence. Re-enable with "
                f"--stop_on_metric_stall on; --early_stop_patience still applies. Training continues.")
            return
        return super(TrainerG_v12Original, self)._trigger_stop(reason)


# ============================================================================
# override 2b - the model builder
# ============================================================================

# Which architectures accept a head dropout at all. `training/config_presets.py`
# raises rather than silently ignoring an inapplicable one, because
# `gmedmamba.ClassificationHead` - used by 'split' and 'fullchannel' - hardcodes
# 0.1 and cannot be overridden without editing frozen `gmedmamba.py`.
_CLASSIFIER_DROPOUT_ARCHS = ("efficient", "recursive")


def build_model_optimal(architecture, *a, classifier_dropout=None, **kw):
    """`v16.build_model` with an inapplicable `--classifier_dropout` dropped
    instead of fatal.

    `OPTIMAL_DEFAULTS` sets `classifier_dropout=0.1` because this file's own
    architecture is 'recursive', where `gmedmamba.RecursiveHead` accepts it.
    That default followed the flag into every architecture, so
    `--architecture split` on this entry point raised

        ValueError: --classifier_dropout only applies to --architecture
        efficient or recursive

    before epoch 1 - for a value the user never typed. On split/fullchannel the
    head dropout is 0.1 in `gmedmamba.ClassificationHead` either way, so the
    honest resolution is to drop the argument and SAY SO, not to fail and not to
    pretend the requested rate is in force.
    """
    if classifier_dropout is not None and architecture not in _CLASSIFIER_DROPOUT_ARCHS:
        print(f"[classifier_dropout] {classifier_dropout} IGNORED on --architecture "
              f"{architecture!r}: it uses gmedmamba.ClassificationHead, whose dropout is "
              f"hardcoded at 0.1 (gmedmamba.py) and is not overridable without editing a frozen "
              f"file. The head runs at 0.1. Use --architecture recursive (or efficient) if the "
              f"rate has to change.", flush=True)
        classifier_dropout = None
    return _V16_BUILD_MODEL(architecture, *a, classifier_dropout=classifier_dropout, **kw)


# ============================================================================
# override 3 - the run name
# ============================================================================

def build_run_name_optimal(args, *a, **kw) -> str:
    """`training.run_naming_v16.build_run_name` builds its tokens from the
    v15/v16 flag set and cannot tell this configuration from an ordinary run
    on the same data and architecture - the two would land in directory names
    differing only by timestamp. Suffix it, as v16_original does."""
    return f"{_V16_BUILD_RUN_NAME(args, *a, **kw)}_optimal"


# ============================================================================
# banner
# ============================================================================

def config_deviations(args, argv=None) -> list:
    """Every default the command line overrode, so a run that is no longer
    this configuration says so in its own log."""
    out = []
    expected = dict(OPTIMAL_DEFAULTS)
    expected.update(_STAGES.get(getattr(args, "stage", "full"), {}))
    data_dir = _peek_data_dir(argv)
    # Against the grid the user actually asked for: --target_token_grid moving
    # patch_size is that flag working, not a deviation from this configuration.
    expected["patch_size"] = resolve_patch_size(
        data_dir, getattr(args, "target_token_grid", _TARGET_TOKEN_GRID))
    # Same treatment for the epoch budget: a frac this file RESOLVED from the
    # corpus size is this configuration, not a deviation from it. Only a frac
    # the user typed differs from `expected` and gets listed.
    expected["train_subsample_frac"] = resolve_subsample_frac(
        _split_size(data_dir, "train"), _TARGET_TRAIN_PATCHES)
    expected["val_subsample_frac"] = resolve_subsample_frac(
        _split_size(data_dir, "val") or _split_size(data_dir, "test"), _TARGET_VAL_PATCHES)
    if "trm_ema_rate" not in _STAGES.get(getattr(args, "stage", "full"), {}):
        expected["trm_ema_rate"] = resolve_ema_rate(
            data_dir, _peek_int("--batch_size", OPTIMAL_DEFAULTS["batch_size"], argv),
            subsample_frac=getattr(args, "train_subsample_frac", 1.0))
    for dest, want in sorted(expected.items()):
        got = getattr(args, dest, None)
        if got != want:
            out.append(f"{dest}={got!r}  (optimal: {want!r})")
    if args.stop_on_metric_stall != "off":
        out.append(f"stop_on_metric_stall={args.stop_on_metric_stall!r}  (optimal: 'off')")
    return out


def print_augmentation(args) -> None:
    """`--augment_preset custom` is assembled from eleven separate flags, so
    the one line that says what augmentation is actually in force has to be
    built here. The two zeros are the point of it: see heading (5)."""
    if args.augment_preset != "custom":
        print(f"  augmentation       : preset {args.augment_preset!r}", flush=True)
        if args.augment_preset in ("light", "medium"):
            print(f"  WARNING: presets 'light'/'medium' were written for 32-band HSI. On 3-channel "
                  f"RGB, band_dropout zeroes a WHOLE colour channel (n_drop = max(1, ...) at C=3, "
                  f"training/augmentation.py:87) and spectral_masking zeroes a contiguous run of "
                  f"them. Colour is this task's main cue; prefer --augment_preset custom.",
                  flush=True)
        return
    print(f"  augmentation       : custom - flip_h={args.flip_h_prob} flip_v={args.flip_v_prob} "
          f"rot90={args.rotate90_prob} crop_scale={args.crop_scale_max_frac}", flush=True)
    print(f"                       photometric: scale={args.spectral_scale_range} "
          f"offset={args.spectral_offset_std} noise={args.spectral_noise_std}", flush=True)
    print(f"                       band_dropout={args.band_dropout_prob} "
          f"spectral_mask={args.spectral_mask_prob}"
          + ("   (both OFF - they destroy colour on 3-channel input)"
             if args.band_dropout_prob == 0.0 and args.spectral_mask_prob == 0.0 else
             "   WARNING: non-zero on RGB destroys a whole colour channel"), flush=True)


def print_banner(args) -> None:
    print("=" * 78, flush=True)
    print(f"GMedMamba-R (TRM) on PAD-UFES-20 - BEST-METRICS configuration  [stage: {args.stage}]",
          flush=True)
    if args.stage != "full":
        print(f"  {args.stage!r} is a DIAGNOSTIC stage, not a result: "
              + {"fit": "every regulariser, the class weighting and EMA are off, so train "
                        "accuracy reads out whether the model can learn this data at all.",
                 "balance": "augmentation, dropout and EMA are off; only the class weighting "
                            "is active."}.get(args.stage, ""), flush=True)
    print(f"  architecture       : {args.architecture}  (stem stride {args.patch_size}, "
          f"trm_dim={args.trm_dim}, mixer={args.trm_mixer}, halting={args.trm_halting})", flush=True)
    print(f"  objective          : loss={args.loss} gamma={args.focal_gamma} "
          f"class_weight_power={args.class_weight_power} sampler={args.sampler}", flush=True)
    print(f"  optimiser          : AdamW lr={args.lr} wd={args.weight_decay} "
          f"wd_groups={args.weight_decay_groups}  clip={args.max_gradient_norm}", flush=True)
    print(f"  schedule           : step-axis warmup+cosine  (interval={args.scheduler_interval})",
          flush=True)
    print(f"  batch/epochs       : {args.batch_size} / {args.epochs}   amp={args.amp}", flush=True)
    # v17 S3 - say it loudly, and say the caveat once. A compiled run is NOT
    # bit-comparable to the runs the manuscript reports; it is comparable to ~1.7e-7
    # (training/torch_compile.py, validated against the eager scan in fp32, which is the
    # path that matters since _SelectiveScanParams.run upcasts before calling the backend).
    if args.compile_model == "on":
        print(f"  throughput         : torch.compile ON (v17 default), loader={args.loader_mode}, "
              f"artifact stride={args.eval_artifact_stride}", flush=True)
        print(f"                       measured 437.0 -> 266.7 ms/step on PAD and 1,428 -> "
              f"1,110 MB peak. Numerics differ from eager by ~1.7e-7; pass --compile off to "
              f"reproduce a pre-v17 run exactly.", flush=True)
    else:
        print(f"  throughput         : torch.compile OFF - this is the pre-v17 eager path and "
              f"is ~1.64x slower on PAD. Deliberate only if you are reproducing an old run.",
              flush=True)
    print(f"  normalization      : {args.normalization}"
          + ("   (train-fitted fixed affine - preserves absolute colour across images)"
             if args.normalization == "global_zscore" else
             "   NOTE: per-image normalization removes absolute lesion darkness, a "
             "diagnostic cue on this dataset"), flush=True)
    print(f"  regularisation     : classifier_dropout={args.classifier_dropout} "
          f"trm_dropout={args.trm_dropout} drop_path={args.drop_path_rate} "
          f"ema={args.trm_ema_rate}", flush=True)
    print_augmentation(args)
    print(f"  selection          : checkpoint on {args.checkpoint_metric}, early stop after "
          f"{args.early_stop_patience} epochs without improvement", flush=True)
    print(f"  stopping rules off : val_divergence={args.val_divergence_patience} "
          f"class_collapse={args.on_class_collapse}/{args.class_collapse_streak} "
          f"metric_stall={args.stop_on_metric_stall}", flush=True)
    print(f"  reconstruction     : {args.recon_mode}"
          + ("   (--recon_mode latent --lambda_mse 0.1 is the first ablation worth running)"
             if args.recon_mode == "none" else ""), flush=True)

    deviations = config_deviations(args)
    if deviations:
        print("  DEVIATIONS from this configuration (explicit flags):", flush=True)
        for d in deviations:
            print(f"    - {d}", flush=True)
    else:
        print("  no deviations: every default in OPTIMAL_DEFAULTS is in force.", flush=True)
    print("=" * 78, flush=True)

    # The bars this run has to clear, read off whatever is next to the
    # dataset: class priors, the constant-predictor scores, and the linear
    # probe from scripts/shallow_baseline_v16.py. Reused from v16_original
    # rather than reimplemented - same numbers, same formatting, one source.
    print_trivial_baselines(args)
    print_step_budget(args)
    print_trm_cost(args)
    print_gpu_budget(args)
    print_ema_horizon(args)
    _warn_on_trm_flags(args)
    _warn_on_patch_grid(args)


def print_gpu_budget(args) -> None:
    """What this configuration will ask of the GPU, against what the GPU has.

    `print_trm_cost` (reused from v16_original) already prints a token count
    and an activation estimate, but it estimates from a single fp32 OOM
    extrapolation and compares against nothing - so a run that uses 8% of the
    card reads exactly like a run that uses 80%. This prints the DENOMINATOR,
    from `torch.cuda`, and says which direction the headroom points.

    The point is not to change anything automatically. It is that
    `--target_token_grid` and `--batch_size` are the two knobs with real
    headroom behind them on this machine, and nothing in the preflight said so.
    """
    if not torch.cuda.is_available():
        print("[gpu-budget] no CUDA device - running on CPU.", flush=True)
        return
    free_b, total_b = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    free_mb, total_mb = free_b / 1024 ** 2, total_b / 1024 ** 2

    side = _input_side(args.data_dir)
    stride = args.patch_size or 1
    grid = max(1, side // stride) if side else None
    print(f"[gpu-budget] {name}: {total_mb:,.0f} MB total, {free_mb:,.0f} MB free now", flush=True)
    if grid is None:
        return

    rows = grid * grid * args.batch_size
    scale = 1.0 if args.amp == "off" else 0.5
    est = _VRAM_FLOOR_MB + _MB_PER_TOKEN_ROW_BF16 * rows * (2.0 * scale)
    print(f"[gpu-budget] {grid}x{grid} tokens x batch {args.batch_size} = {rows:,} token-rows "
          f"-> ~{est:,.0f} MB estimated ({100.0 * est / total_mb:.1f}% of the card) at "
          f"--amp {args.amp}", flush=True)
    print(f"[gpu-budget]   (~{_MB_PER_TOKEN_ROW_BF16:.3f} MB/token-row at bf16, fitted to the "
          f"1,432 MB MEASURED at 28x28 x 32. Where this disagrees with [trm-cost] above, prefer "
          f"this one: that estimate extrapolates from a single fp32 OOM and reads ~2x low at this "
          f"grid. Both are guesses - GPU_memory_MB in history.json is the truth.)", flush=True)

    if est > 0.85 * total_mb:
        print(f"[gpu-budget] WARNING: that is close to the card. Lower --target_token_grid "
              f"(quadratic) before --batch_size (linear).", flush=True)
    elif est < 0.25 * total_mb:
        nxt = grid * 2
        print(f"[gpu-budget] {100.0 * (total_mb - est) / total_mb:.0f}% of this card will sit "
              f"IDLE. The memory is there for --target_token_grid {nxt} "
              f"(~{_VRAM_FLOOR_MB + _MB_PER_TOKEN_ROW_BF16 * rows * 4 * (2.0 * scale):,.0f} MB, "
              f"4x the spatial detail) - but also ~4x the wall clock, so spend it deliberately. "
              f"Nothing here changes on its own.", flush=True)


def print_ema_horizon(args) -> None:
    """How much of the RANDOM INITIALISATION is still inside the model that
    validation is computed on. Printed because a stale EMA is indistinguishable
    from a collapsed model in the epoch log - both show balanced accuracy
    0.1667 - and the first real run on this config lost 16 epochs to that."""
    if args.architecture != "recursive":
        # train_example_v16.py:685 passes ema_rate only when architecture is
        # 'recursive'; every other architecture falls back to cfg.trm_ema_rate
        # (0.0 in the presets), so no EMA is built and validation runs on the
        # live weights. Printing an averaging horizon here would be a fiction.
        print(f"[ema] --trm_ema_rate {args.trm_ema_rate} has NO EFFECT on --architecture "
              f"{args.architecture!r} (TrainerG_v10 receives an EMA rate only for 'recursive'): "
              f"validation runs on the live weights.", flush=True)
        return
    if not args.trm_ema_rate:
        print("[ema] disabled - validation runs on the live weights.", flush=True)
        return
    spe = steps_per_epoch(args.data_dir, args.batch_size, args.train_subsample_frac)
    if spe is None:
        return
    mu = args.trm_ema_rate
    horizon = 1.0 / max(1e-12, (1.0 - mu) * spe)
    print(f"[ema] rate {mu:.5f} at {spe} steps/epoch = a {horizon:.1f}-epoch averaging horizon. "
          f"TrainerG_v10 validates the EMA copy, whose shadow starts at the random init and is "
          f"never bias-corrected, so early validation numbers are partly a measurement of noise:",
          flush=True)
    marks = [e for e in (1, 3, 5, 10, 25, 50) if e <= args.epochs]
    print("[ema]   init weight remaining: "
          + "  ".join(f"ep{e}={mu ** (e * spe):.1%}" for e in marks), flush=True)
    below5 = np.log(0.05) / np.log(mu) / spe if mu < 1 else float("inf")
    verdict = ("fine" if below5 <= 12 else
               f"TOO SLOW - raise --batch_size or lower --trm_ema_rate; "
               f"--early_stop_patience {args.early_stop_patience} may end the run first")
    print(f"[ema]   validation is >95% trained model from epoch ~{below5:.0f} onward: {verdict}",
          flush=True)


# The trm_* flags reach the model only through `trm_kwargs`, which
# `train_example_v16.py:552` builds ONLY for architecture='recursive'. On any
# other architecture they parse, print in the banner, land in config.json - and
# do nothing. This file defaults several of them, so say which ones are inert.
_TRM_ONLY_FLAGS = ("trm_dim", "trm_mixer", "trm_halting", "trm_dropout", "trm_drop_path",
                   "trm_core_layers", "trm_n_latent", "trm_n_improve",
                   "trm_deep_supervision_steps", "trm_ema_rate")


def _warn_on_trm_flags(args) -> None:
    if args.architecture == "recursive":
        return
    named = [f"{f}={getattr(args, f)}" for f in _TRM_ONLY_FLAGS if hasattr(args, f)]
    print(f"[trm] --architecture {args.architecture!r} is NOT the recursive core: "
          f"train_example_v16.py builds trm_kwargs only for 'recursive', so these are inert "
          f"here - {', '.join(named)}. The banner's architecture line prints them because it "
          f"reads the parsed flags, not the model.", flush=True)


def _warn_on_patch_grid(args) -> None:
    """The one dataset-shape mistake that no other preflight line catches: a
    dataset of tiny crops, where the label mostly does not describe the
    sample and no configuration can help."""
    side = _input_side(args.data_dir)
    if side is None or side > 32:
        return
    print(f"[input] WARNING: {side}x{side} samples. On this dataset's 11x11 grid a linear probe "
          f"over all 363 pixels scored macro-F1 0.185 - BELOW the same probe on 3 mean channel "
          f"values - because the crop is 0.24% of the frame and mostly does not contain the "
          f"lesion. GMedMamba-R already beats that probe there (0.235). Re-prep with "
          f"prepare_pad_ufes_20_optimal.py --tiling whole; nothing in this file recovers a "
          f"signal the input does not carry.", flush=True)


# ============================================================================
# install + main
# ============================================================================

def build_overrides(args) -> "v16.Overrides":
    """What this entry point contributes to `v16._main` (v17 S9).

    This used to be `install_overrides`, which ASSIGNED into v16's module namespace -
    `v16.build_arg_parser = ...`, `v16.build_run_name = ...`, `v16.build_model = ...`,
    `v16.TrainerG_v12 = ...`. That worked, but the effect was invisible at the call site,
    order-dependent when a third layer (`train_example_v16_optimal_recon.py`) rebound the
    same names again, and left the module permanently mutated for anything else in the
    process. Returning a value instead makes the composition explicit and local.

    The one thing that is still a mutation is `suppressed_stop_reasons`: it is a CLASS
    attribute read by the inherited `_trigger_stop`, which is the single two-line funnel
    every stopping rule goes through (`training/trainerg_v4.py:647`), and setting it is
    what lets one rule be neutralised without copying `_check_stopping_rules`.
    """
    TrainerG_v12Optimal.suppressed_stop_reasons = (
        frozenset() if args.stop_on_metric_stall == "on" else frozenset({StopReason.VAL_ACC_STALLED}))
    return v16.Overrides(parser_fn=build_arg_parser,
                         run_name_fn=build_run_name_optimal,
                         model_fn=build_model_optimal,
                         trainer_cls=TrainerG_v12Optimal)


def install_overrides(args) -> None:
    """Deprecated shim (v17 S9). Kept because `train_example_v16_optimal_recon.py` and any
    external script may still call it; it now applies `build_overrides` the old way."""
    ov = build_overrides(args)
    v16.build_arg_parser = ov.parser_fn
    v16.build_run_name = ov.run_name_fn
    v16.build_model = ov.model_fn
    v16.TrainerG_v12 = ov.trainer_cls


def main() -> None:
    # v16._main parses again from the same (patched) parser; argparse is pure,
    # so this preview parse has no side effects and is only here because the
    # overrides have to be chosen before _main starts.
    args = build_arg_parser().parse_args()
    print_banner(args)
    try:
        v16.main(build_overrides(args))
    except torch.OutOfMemoryError:
        side = _input_side(args.data_dir)
        stride = args.patch_size or 1
        grid = (side // stride) if side else None
        print("\n" + "=" * 78, flush=True)
        print("CUDA OUT OF MEMORY - the recursive core's working set is (tokens x batch), and "
              "every one of its ~63 applications stores a checkpointed activation.", flush=True)
        if grid:
            print(f"  this run: {grid}x{grid} = {grid * grid} tokens x batch {args.batch_size}",
                  flush=True)
        print(f"  --patch_size {stride * 2}   halves the grid, quartering activations - the "
              f"cheapest fix, and it costs spatial resolution rather than optimisation.", flush=True)
        print(f"  --batch_size {max(1, args.batch_size // 2)}   halves activations too, but also "
              f"halves the gradient-noise scale this configuration was chosen around.", flush=True)
        print("=" * 78, flush=True)
        raise


if __name__ == "__main__":
    main()
