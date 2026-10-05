# -*- coding: utf-8 -*-
"""
train_example_v16_original.py
==============================
GMedMamba (`gmedmamba.py`, `--architecture recursive` = the TRM variant)
trained under the ORIGINAL MedMamba TRAINING protocol, so the PAD-UFES-20
head-to-head in the manuscript's Section VI-C can be tightened from "loss,
class weighting, data, objective, normalization, weight averaging and
augmentation matched" to "essentially only the architecture differs".

Companion to `prepare_pad_ufes_20_original.py`, which produces the matching
dataset (one 224x224 clinical image per sample, 60/10/30). This file is a
new ENTRY POINT built on top of `train_example_v16.py`, imported as a module
and never edited - the same one-directional relationship v16 has to v15.
Nothing here is imported by v16, v15 or any frozen `trainerg_*`, so the two
audited runs stay bit-reproducible.

The gap this closes
-------------------
`paper/GMedMamba_manuscript_v6.md:564` lists what still differed between
GMedMamba-R and the MedMamba baseline on PAD after the Section VI-C
re-run:

    "... but gradient clipping, learning-rate schedule axis, early-stopping
     patience and classifier dropout still differ, and every cell is a
     single seed."

All four are matched below, along with the input protocol itself (whole
224x224 images instead of 11x11 patches), which is the larger confound and
the one Section VI-C blames for both models barely clearing a six-feature
colour baseline.

The protocol, and where each line comes from
--------------------------------------------
Reference: `../medmamba-original/MedMamba/train.py`, whose defaults follow
MedMamba (arXiv:2403.03849) for the non-MedMNIST datasets, of which
PAD-UFES-20 is one (reported there with MedMamba-T).

  setting              this file      v16 default    reference
  -------------------- -------------- -------------- ------------------------
  epochs               150            10             train.py --epochs
  batch_size           64             256            train.py --batch_size
  lr                   1e-4           1e-4           train.py --lr
  weight_decay         1e-4           0.05           train.py --weight_decay
  wd param groups      off            on             AdamW(model.parameters())
  betas                (0.9, 0.999)   (0.9, 0.999)   train.py optim.AdamW
  LR schedule          constant       warmup+cosine  train.py --scheduler none
  gradient clipping    off            1.0            (train.py never clips)
  AMP                  off (fp32)     bf16           train.py --amp default off
  augmentation         none           none           train.py --augment default off
  loss                 plain CE       plain CE       nn.CrossEntropyLoss()
  class weights        none           none           train.py --class_weights off
  sampler              none           none           (ImageFolder, shuffle)
  reconstruction       none           latent         (no such head upstream)
  weight averaging     none           EMA 0.999      (no EMA upstream)
  normalization        (x-0.5)/0.5    per-sample     transforms.Normalize(.5,.5)
                                      min-max        (--global_stats identity
                                                      instead, for input parity
                                                      with train_hsi.py)
  checkpoint metric    val_accuracy   f1_macro       "best validation accuracy"
  early-stop patience  off (see below) none           train.py --early_stopping_patience
  classifier dropout   0.0            preset         VSSM head is a bare Linear
  drop-path rate       0.0            preset         VSSM drop_path_rate=0.1
                       (see below - this one deliberately does NOT match either)
  patch/stem stride    4 hierarchical 1 (recursive)  PatchEmbed2D(patch_size=4)
                       8 recursive                    (recursive deliberately
                                                       does NOT match; see below)
  seed                 42             42             train.py --seed
  test evaluation      best ckpt      best ckpt      "metrics on the test split"

Two v16-only stopping rules are also disabled, because the reference has no
analogue and leaving them on would end the run before the protocol says it
should - which would be a difference in the *measurement*, not in the model:

  --val_divergence_patience 0   E-1 stops when the checkpoint metric worsens
                                while val_loss rises. On 1.4k training images
                                with plain CE that is guaranteed within a few
                                epochs; the reference instead runs to 150
                                epochs or 20 epochs without a val-accuracy
                                improvement.
  --on_class_collapse warn      a plain-CE run on a 6-class split this
                                imbalanced legitimately predicts one class for
                                the first few epochs. The reference has no
                                such guard, so aborting on it would make the
                                comparison impossible rather than fair.
  --early_stopping off          the reference train.py stops after 20 epochs
                                without a val-accuracy improvement; v16's
                                --early_stop_patience does not implement that
                                (v12 shadows the rule - see
                                `TrainerG_v12Original._check_stopping_rules`).
                                `--early_stopping on` reinstates it. It is off
                                by default because train_hsi.py - the baseline
                                that reads this same .npy split - has no early
                                stopping either, so off is the matched choice
                                for that pairing, and because the model needs
                                ~55 epochs here to leave the class prior.
  --stop_on_val_acc_stall off   `TrainerG_v12._check_stopping_rules` stops
                                after 5 epochs whose checkpoint metric is
                                EXACTLY equal to the previous epoch's, at a
                                threshold hardcoded to 5. The reference
                                protocol's only stopping rule is 20 epochs
                                without an improvement. That difference is
                                harmless when the metric is macro-F1 (a
                                continuous quantity that rarely repeats to the
                                bit) and severe when it is validation accuracy
                                on a 245-sample split, which takes one of 246
                                values: the first run under this protocol died
                                at epoch 16 of 150 on six consecutive 0.4000s.

The one protocol line this file deliberately does NOT match: stem stride
---------------------------------------------------------------------
MedMamba's `PatchEmbed2D` uses stride 4, so a 224x224 input enters its
backbone as a 56x56 grid - and is then halved by every stage, so the
backbone's four stages see 3136, 784, 196 and 49 tokens. GMedMamba-R has no
downsampling at all: whatever grid the stem produces is carried at full
resolution through all ~63 applications of the shared core, each of which
stores a checkpointed activation. Copying the number 4 therefore does not
copy the workload; it makes the recursive model hold 3136 tokens 63 times
where MedMamba holds that many once.

Measured on a 16 GB RTX 5060 Ti at the protocol's batch size of 64, stride 4
needs about 11.9 GB of activations and dies with `torch.OutOfMemoryError`
partway through the first epoch. Stride 8 (28x28 = 784 tokens) needs about
3.0 GB and is the default here, because the alternative - keeping stride 4
and dropping to `--batch_size 16`, which costs the same memory - would break
a line of the protocol that genuinely affects optimisation to preserve one
that was never a match to begin with. `--patch_size 4 --batch_size 16` is
still available for anyone who prefers that trade; the `[trm-cost]` preflight
line prints the working-set size either way.

The second line it does not match: stochastic depth
---------------------------------------------------
MedMamba's `VSSM` takes `drop_path_rate=0.1` and spreads it over its blocks
with `torch.linspace(0, 0.1, sum(depths))`, so most blocks get far less than
0.1 and only the last gets the full rate - a depth-wise schedule, not a flat
rate. `GMedMambaConfig.trm_drop_path` also linspaces, but over
`trm_core_layers` (2), and that core is then replayed ~63 times per forward
pass, so the rate is applied ~63x per sample per step instead of once.
Copying the number 0.1 therefore buys far more regularisation than MedMamba
applies, on a 0.447 M-parameter model that is already the smaller of the two.
Every GMedMamba-R run in the manuscript used 0.0, so 0.0 also keeps this run
comparable to them. `--drop_path_rate 0.1` restores the literal value for
anyone who wants to measure its effect.

Everything else in the v16 pipeline is untouched and deliberately so: the
same `TrainerG_v12`, the same metric definitions, the same gates (G1/G2/G5,
G6, G8, G9), the same reports. Comparing numbers across runs only means
something if they were computed by the same code.

How the overrides are applied
-----------------------------
`train_example_v16._main` is one long function, so instead of copying it
(which would let the two drift silently - the exact failure the v16 plan's
freezing rule exists to prevent) this file rebinds five names in v16's
module namespace before calling it. Each is a named function/class below
with the reason it exists; there is no other behavioural change.

Usage
-----
    python prepare_pad_ufes_20_original.py \
        --root /path/to/PAD-UFES-20 --out_dir ./data/pad_original --num_workers 8

    python train_example_v16_original.py --data_dir ./data/pad_original/

    # the hierarchical GMedMamba under the identical protocol
    python train_example_v16_original.py --data_dir ./data/pad_original/ --architecture split

Any v16/v15 flag still works and overrides the protocol default; the run's
`config.json` records what was actually used, and the banner below prints
every deviation it detected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import train_example_v16 as v16
from training.trainerg_v12 import TrainerG_v12
from training.trainerg_v4 import StopReason
from training.trainerg_v7 import _StopReasonStr

# Captured BEFORE any rebinding, so the overrides below can delegate to the
# real implementations instead of recursing into themselves.
_V16_BUILD_ARG_PARSER = v16.build_arg_parser
_V16_BUILD_RUN_NAME = v16.build_run_name
_V16_GLOBAL_STATS = v16.compute_global_channel_stats
_V16_SCHEDULER = v16.build_warmup_cosine_scheduler
_V16_TRAINER = v16.TrainerG_v12


# ============================================================================
# the protocol, as data
# ============================================================================

# `dest -> value`. Applied as argparse DEFAULTS, so every one of them is
# still overridable on the command line, and `explicitly_passed()` still
# reports the user's flags as explicit.
MEDMAMBA_PROTOCOL_DEFAULTS = {
    # --- optimisation (train.py's argparse defaults) --------------------
    "epochs": 150,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "weight_decay_groups": False,      # reference: AdamW(model.parameters(), weight_decay=..)
    "max_gradient_norm": 0.0,          # falsy -> TrainerG clips at inf, i.e. not at all
    "amp": "off",                      # "so that results match the paper's full-precision training"
    "early_stop_patience": 20,
    "checkpoint_metric": "val_accuracy",

    # --- objective ------------------------------------------------------
    "loss": "ce",
    "sampler": "none",
    "class_weight_power": 1.0,
    "recon_mode": "none",              # no auxiliary decoder exists upstream
    "lambda_mse": 0.0,
    "lambda_sam": 0.0,
    "lambda_gan": 0.0,

    # --- data -----------------------------------------------------------
    "normalization": "global_zscore",  # with the fixed 0.5/0.5 stats below
    "augment_preset": "none",
    "train_subsample_frac": 1.0,
    "val_subsample_frac": 1.0,
    "eval_group_aggregation": "none",  # one sample per clinical image already

    # --- model ----------------------------------------------------------
    "architecture": "recursive",
    "patch_size": 8,                   # recursive only; see `_PATCH_SIZE_BY_ARCHITECTURE`
    "classifier_dropout": 0.0,         # VSSM's head is `nn.Linear`, no dropout
    "drop_path_rate": 0.0,             # NOT VSSM's 0.1 - see `stochastic depth` below
    "trm_ema_rate": 0.0,               # 0 -> TrainerG_v10 builds no EMAHelper
    "trm_halting": False,              # ACT has no analogue upstream

    # --- v16-only stopping rules with no upstream analogue --------------
    "val_divergence_patience": 0,
    "on_class_collapse": "warn",

    # --- reporting ------------------------------------------------------
    "eval_test": "best",
    "seed": 42,
}


# `--patch_size` is the one protocol default that cannot be a constant. The
# hierarchical backbones halve their token grid at every stage, exactly as
# MedMamba does, so its stride of 4 is both the matching value and an
# affordable one. The recursive core never downsamples - see the docstring -
# so 4 there means 3136 tokens held through ~63 core applications, which does
# not fit on a 16 GB card at the protocol's batch size.
_PATCH_SIZE_BY_ARCHITECTURE = {"recursive": 8, "split": 4, "fullchannel": 4, "efficient": 4}


def _peek_architecture(argv=None) -> str:
    """`--architecture` off the command line, before the real parser exists.
    A throwaway parser rather than a string scan of `sys.argv`, so
    `--architecture=split`, abbreviations and `--` all behave as argparse
    says they should."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--architecture", default=MEDMAMBA_PROTOCOL_DEFAULTS["architecture"])
    known, _rest = peek.parse_known_args(argv)
    return known.architecture


def protocol_defaults(argv=None) -> dict:
    """`MEDMAMBA_PROTOCOL_DEFAULTS` with `patch_size` resolved for the
    architecture actually being run."""
    defaults = dict(MEDMAMBA_PROTOCOL_DEFAULTS)
    defaults["patch_size"] = _PATCH_SIZE_BY_ARCHITECTURE.get(
        _peek_architecture(argv), MEDMAMBA_PROTOCOL_DEFAULTS["patch_size"])
    return defaults


# ============================================================================
# override 1 - the parser
# ============================================================================

def build_arg_parser(argv=None) -> argparse.ArgumentParser:
    """v16's parser with the protocol defaults applied and two flags added.
    Rebound onto `v16.build_arg_parser`, which is what both `v16._main` and
    `v16.explicitly_passed` call - neither passes `argv`, so in normal use the
    architecture peek reads `sys.argv`, which is the same command line
    `parse_args` is about to read. `argv` exists so the resolution is
    testable without mutating `sys.argv`."""
    p = _V16_BUILD_ARG_PARSER()
    p.description = ("GMedMamba under the ORIGINAL MedMamba training protocol "
                     "(arXiv:2403.03849, non-MedMNIST) - matched-conditions baseline comparison")

    unknown = set(MEDMAMBA_PROTOCOL_DEFAULTS) - {a.dest for a in p._actions}
    if unknown:
        # A silently-ignored key would produce a run that claims to follow the
        # protocol while not following it - fail at import instead.
        raise RuntimeError(f"MEDMAMBA_PROTOCOL_DEFAULTS names flag(s) v16's parser does not "
                            f"have: {sorted(unknown)}")
    p.set_defaults(**protocol_defaults(argv))

    g = p.add_argument_group("original-protocol")
    g.add_argument("--lr_schedule", choices=["constant", "warmup_cosine"], default="constant",
                    help="'constant' is the reference protocol ('constant learning rate', "
                         "train.py --scheduler none). 'warmup_cosine' restores v16's step-axis "
                         "warmup+cosine schedule.")
    g.add_argument("--early_stopping", choices=["off", "on"], default="off",
                    help="v16's --early_stop_patience is INERT: TrainerG_v12._check_stopping_rules "
                         "overrides TrainerG_v7's without calling super(), so the patience rule at "
                         "trainerg_v7.py:74 never runs. 'on' reinstates it (the reference "
                         "train.py's rule, 20 epochs without a val-accuracy improvement). 'off' is "
                         "the default because the baseline you can actually run on this same split, "
                         "train_hsi.py, has no early stopping at all - so 'off' is the MATCHED "
                         "choice for that pairing. Beware: on PAD-original the model sits at the "
                         "class prior for ~55 epochs before it separates classes, so 'on' stops it "
                         "around epoch 43, before it has learned anything.")
    g.add_argument("--stop_on_val_acc_stall", choices=["on", "off"], default="off",
                    help="v16 stops after 5 epochs whose checkpoint metric exactly equals the "
                         "previous epoch's (threshold hardcoded in "
                         "TrainerG_v12._check_stopping_rules). 'off' is the reference protocol, "
                         "whose only stopping rule is --early_stop_patience. Leave it off "
                         "whenever --checkpoint_metric is val_accuracy: on a few-hundred-sample "
                         "split that metric is discrete and repeats exactly all the time.")
    g.add_argument("--global_stats", choices=["medmamba_fixed", "identity", "train_fit"],
                    default="medmamba_fixed",
                    help="only used with --normalization global_zscore. 'medmamba_fixed' pins the "
                         "per-channel mean/std to 0.5/0.5, which makes the transform exactly "
                         "torchvision's Normalize((.5,.5,.5),(.5,.5,.5)) on [0,1] inputs - the "
                         "reference's normalization, expressed in v16's existing affine mode so "
                         "no new normalization code (and no new bug) enters the comparison. "
                         "'identity' pins them to 0/1, i.e. feeds the stored [0,1] array through "
                         "unchanged - use it to be bit-identical to a baseline run of "
                         "medmamba-original/MedMamba/train_hsi.py --normalize none, which has no "
                         "fixed-affine mode; the two models then see the same tensor, at the cost "
                         "of the published recipe's centering. 'train_fit' restores v16's "
                         "train-split-fitted statistics.")
    return p


# ============================================================================
# override 2 - the normalization statistics
# ============================================================================

def _constant_channel_stats(x_path: str, mean: float, std: float) -> Tuple[np.ndarray, np.ndarray]:
    """`(mean, std)` broadcast to the array's channel count, as float32 - the
    shape `global_zscore` expects. Reads only the header via `mmap_mode`."""
    n_channels = int(np.load(x_path, mmap_mode="r").shape[-1])
    return (np.full((n_channels,), mean, dtype=np.float32),
            np.full((n_channels,), std, dtype=np.float32))


def identity_channel_stats(x_path: str, sample_cap: int = 5000, seed: int = 0
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """`mean=0, std=1`, i.e. `global_zscore` becomes the identity and the
    stored `[0, 1]` array reaches the model untouched - what
    `medmamba-original/MedMamba/train_hsi.py --normalize none` feeds the
    baseline. Use when input parity between the two models matters more than
    fidelity to the reference's `[-1, 1]` centering, which `train_hsi.py`
    cannot reproduce (it offers only none / per_patch_zscore /
    per_sample_minmax)."""
    return _constant_channel_stats(x_path, 0.0, 1.0)


def medmamba_fixed_channel_stats(x_path: str, sample_cap: int = 5000, seed: int = 0
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Drop-in for `train_example_v6.compute_global_channel_stats` that
    returns the constants 0.5 / 0.5 instead of fitting the train split.

    `global_zscore` computes `(x - mean_c) / std_c`; with `mean_c = std_c =
    0.5` that IS `transforms.Normalize((0.5,)*3, (0.5,)*3)` applied to the
    `[0, 1]` arrays `prepare_pad_ufes_20_original.py` stores, i.e. the
    reference pipeline's `ToTensor() -> Normalize(.5, .5)`. Expressing it
    this way rather than as a new normalization mode means every downstream
    consumer - `NpyDatasetV16`, gate G8's `check_split_drift`, the G6 test
    loader, `denormalize` - keeps working unmodified and stays covered by
    `test_normalization.py`.

    `sample_cap`/`seed` are accepted and ignored: there is nothing to sample.
    The channel count is read from the array so this is not RGB-specific.
    """
    return _constant_channel_stats(x_path, 0.5, 0.5)


# ============================================================================
# override 3 - the learning-rate schedule
# ============================================================================

def build_constant_scheduler(optimizer, total_steps: int, warmup_steps: Optional[int] = None):
    """Drop-in for `train_example_v16.build_warmup_cosine_scheduler` holding
    the LR at its initial value for the whole run - the reference's
    `--scheduler none`, "which states only a constant initial learning rate".

    A `LambdaLR` returning 1.0 rather than `scheduler=None`, because
    `v16._main` reads `scheduler.warmup_steps` / `.total_steps` for its log
    line and `config.json`, and `TrainerG_v12` calls `.step()` on whatever it
    is given. Stepping a constant lambda is a no-op, so this is also
    indifferent to `--scheduler_interval`, which is what makes the
    "learning-rate schedule axis" confound disappear rather than merely move.
    """
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scheduler.warmup_steps = 0
    scheduler.total_steps = max(1, int(total_steps))
    # `v16._main` logs a fixed "[scheduler] step-axis warmup+cosine: ..." line
    # for whatever this returns, which would now be wrong. Say the true thing
    # first, on the line above it, rather than leave the log self-contradicting.
    print(f"[scheduler] CONSTANT lr for all {scheduler.total_steps} steps (original MedMamba "
          f"protocol; train.py --scheduler none). The 'warmup+cosine' line below is v16's fixed "
          f"format string - warmup_steps=0 and the multiplier is 1.0 at every step.", flush=True)
    return scheduler


# ============================================================================
# override 4 - gate G5 under a `val_accuracy` checkpoint metric
# ============================================================================

class TrainerG_v12Original(TrainerG_v12):
    """`TrainerG_v11.verify_best_checkpoint_reproduces` (gate G5) looks
    `self.checkpoint_metric` up in `metrics["classification"]`, but the epoch
    metrics dict renames that dict's `"accuracy"` key to `"val_accuracy"`
    (`training/trainerg_v12.py:810`). So a run selecting checkpoints on
    validation accuracy - which is what the reference protocol does, "the
    checkpoint with the best validation accuracy" - gets
    `reproduced=None -> FAIL` from G5 every time, for a naming reason and not
    a reproducibility one. Observed on the first smoke run of this file.

    Both frozen trainers stay untouched; the lookup name is translated for the
    duration of the check only. The two names denote the same quantity
    computed on the same loader, so the comparison G5 makes is unchanged.
    """

    _CHECKPOINT_METRIC_ALIASES = {"val_accuracy": "accuracy"}

    # Stop reasons this run declines to act on. Set from
    # `--stop_on_val_acc_stall`; `_trigger_stop` is the single two-line
    # funnel every rule goes through (`TrainerG_v4:647`), so filtering here
    # neutralises one rule without copying `_check_stopping_rules` and
    # without touching a rule the protocol does want.
    suppressed_stop_reasons: frozenset = frozenset()

    # `--early_stopping on`. Off by default; see the flag's help.
    early_stopping_enabled: bool = False

    def _check_stopping_rules(self):
        """v12's rules, plus `TrainerG_v7`'s patience rule when asked for.

        `TrainerG_v12._check_stopping_rules` does not call `super()`, and it
        sits ahead of `TrainerG_v7` in the MRO, so v7's
        `--early_stop_patience` block is unreachable for every v12 run - the
        flag is accepted, recorded in config.json, and does nothing. That is a
        pre-existing v16 defect, not something this protocol introduced, but
        this file's docstring claimed the patience was matched, so it has to
        either work or be described accurately. It now does both.

        The counter logic below mirrors trainerg_v7.py:52-81 rather than
        calling it, because `TrainerG_v7._check_stopping_rules` also invokes
        the v4/v6 heuristics that v12 deliberately replaced (E-1); running
        those again would reintroduce rules the v16 plan removed.
        """
        super()._check_stopping_rules()
        if not self.early_stopping_enabled or self.early_stop_patience is None or self.is_stopped:
            return
        if not self.history:
            return
        last = self.history[-1]
        if not last.get("is_valid_epoch", True):
            return                      # an invalid epoch is not a failed attempt (v7, Phase 17)
        candidate = last.get(self.checkpoint_metric)
        if candidate is None:
            return
        # `fit` calls this BEFORE updating best_metric_value, so the comparison
        # asks "did THIS epoch beat the prior best" - v7's semantics exactly.
        if candidate > self.best_metric_value + 1e-9:
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        if self._epochs_without_improvement >= self.early_stop_patience:
            self.stop_reason = _StopReasonStr(
                f"EARLY_STOPPED_PATIENCE_{self.checkpoint_metric.upper()}")
            self.is_stopped = True
            self.logger.info(
                f"Early stopping: no improvement in {self.checkpoint_metric!r} for "
                f"{self.early_stop_patience} epochs (best={self.best_metric_value:.4f}).")

    def _trigger_stop(self, reason):
        if reason in self.suppressed_stop_reasons:
            self.logger.info(f"[stopping rule] {getattr(reason, 'value', reason)} suppressed - it "
                              f"has no analogue in the reference protocol (see "
                              f"--stop_on_val_acc_stall). Training continues.")
            return
        return super()._trigger_stop(reason)

    def verify_best_checkpoint_reproduces(self, *a, **kw):
        alias = self._CHECKPOINT_METRIC_ALIASES.get(self.checkpoint_metric)
        if alias is None:
            return super().verify_best_checkpoint_reproduces(*a, **kw)

        reported = self.checkpoint_metric
        self.checkpoint_metric = alias
        try:
            out = super().verify_best_checkpoint_reproduces(*a, **kw)
        finally:
            self.checkpoint_metric = reported

        out["metric"] = reported
        out["metric_looked_up_as"] = alias
        # super() already wrote the artefact under the aliased name; rewrite it
        # so the file and the return value agree.
        try:
            with open(self.exp_dir / "checkpoint_reproducibility.json", "w") as f:
                json.dump(out, f, indent=2, default=str)
        except OSError:
            pass
        return out


# ============================================================================
# override 5 - the run name
# ============================================================================

def build_run_name_original(args, *a, **kw) -> str:
    """`training.run_naming_v16.build_run_name` builds its tokens from the
    v15/v16 flag set and so cannot tell a protocol run from an ordinary one -
    two runs on the same data with the same architecture would land in
    directory names that differ only by timestamp. Suffix it."""
    return f"{_V16_BUILD_RUN_NAME(args, *a, **kw)}_medmamba_protocol"


# ============================================================================
# banner + install
# ============================================================================

def protocol_deviations(args, argv=None) -> list:
    """Every protocol default the command line overrode, so a run that is no
    longer a matched-conditions run says so in its own log."""
    out = []
    for dest, expected in sorted(protocol_defaults(argv).items()):
        actual = getattr(args, dest, None)
        if actual != expected:
            out.append(f"{dest}={actual!r}  (protocol: {expected!r})")
    if args.lr_schedule != "constant":
        out.append(f"lr_schedule={args.lr_schedule!r}  (protocol: 'constant')")
    if args.global_stats != "medmamba_fixed":
        out.append(f"global_stats={args.global_stats!r}  (protocol: 'medmamba_fixed')")
    if args.early_stopping != "off":
        out.append(f"early_stopping={args.early_stopping!r}  (protocol: 'off')")
    if args.stop_on_val_acc_stall != "off":
        out.append(f"stop_on_val_acc_stall={args.stop_on_val_acc_stall!r}  (protocol: 'off')")
    return out


# The manuscript's GMedMamba-R runs, as the reference point the estimates
# below are anchored on: PAD-UFES-20 as an 11x11 patch grid, stem stride 1,
# batch 256, bf16, 394 optimizer steps/epoch x 40 epochs
# (experiments/20260905_133922_.../history.json).
_PUBLISHED_TOKENS, _PUBLISHED_BATCH = 121, 256
_PUBLISHED_STEPS = 394 * 40

# Activation bytes per token-row, backed out of a measured OOM: 3136 tokens x
# batch 64 in fp32 reached ~11.9 GB of activations on a 16 GB RTX 5060 Ti.
# A crude single-point fit, so the print below rounds hard and is labelled as
# an estimate - its job is to make the trade-off visible before a run dies an
# hour in, not to be accurate to the megabyte.
_GB_PER_TOKEN_ROW = 11.9 / (3136 * 64)


def _input_side(data_dir: str) -> Optional[int]:
    """H of `X_train.npy`, read from the .npy header alone. Returns None
    rather than raising: this is a log line, not a gate."""
    try:
        shape = np.load(Path(data_dir) / "X_train.npy", mmap_mode="r").shape
        return int(shape[1]) if len(shape) == 4 else None
    except Exception:
        return None


# Above this train max/min class ratio, plain CE on a small split reliably
# converges to the majority class. Chosen well below PAD-UFES-20's 15.6x
# rather than tuned - the point is to fire before a run is wasted, and a
# false positive costs one printed paragraph.
_IMBALANCE_WARN_RATIO = 5.0


def _warn_on_imbalance(args, counts: np.ndarray) -> None:
    """The single most expensive thing to discover after a full run rather
    than before it.

    Every GMedMamba-R PAD run in the manuscript trained on
    `data/pad_v6-undersample`, which is class-BALANCED by construction
    (16,800 patches per class, ratio 1.0) and 73x larger. The whole-image prep
    this file is built for is neither: 1,384 images at ratio 15.6, with 33
    MEL. Under plain cross-entropy a model that extracts only a weak signal
    then has near-prior posteriors, and the argmax of a near-prior posterior
    is the majority class for every input - which scores exactly the
    constant-predictor row printed above, and predicts one class per epoch.
    That is what the first two runs under this protocol did.

    Not fixed by changing a default: plain CE on the natural distribution IS
    the reference protocol. But it is fixable symmetrically, which keeps the
    comparison controlled - see the message.
    """
    present = counts[counts > 0]
    if present.size < 2:
        return
    ratio = float(present.max() / present.min())
    if ratio < _IMBALANCE_WARN_RATIO or args.loss != "ce" or args.sampler != "none":
        return
    try:
        with open(Path(args.data_dir) / "class_names.json") as f:
            names = json.load(f)
        rarest = names[int(np.argmin(np.where(counts > 0, counts, counts.max() + 1)))]
    except Exception:
        rarest = f"class {int(np.argmin(np.where(counts > 0, counts, counts.max() + 1)))}"
    print(f"[imbalance] WARNING: train imbalance {ratio:.1f}x (rarest: {rarest}, "
          f"{int(present.min())} of {int(counts.sum()):,} samples) under --loss ce "
          f"--sampler none.", flush=True)
    print(f"[imbalance] A weak learner's argmax is then the majority class for every input, "
          f"scoring exactly the constant-predictor row above. The manuscript's PAD runs used a "
          f"class-BALANCED dataset (pad_v6-undersample: equal counts, 73x more samples), so "
          f"they never met this.", flush=True)
    print(f"[imbalance] Symmetric remedy, if the run collapses: --loss weighted_ce here and "
          f"--class_weights on the MedMamba side. Both compute n/(k*count), so the comparison "
          f"stays matched - it deviates from the published recipe on BOTH arms, not one.",
          flush=True)


def print_trivial_baselines(args) -> None:
    """The two numbers every result on this dataset has to be read against.

    A model that learns the class marginal and nothing about its input
    converges to the training prior's entropy and predicts the majority class
    everywhere. Printing both up front makes that outcome recognisable in the
    first epoch instead of after a full run: the first run under this protocol
    sat at train_loss 1.50 against a floor of 1.4973, val_accuracy exactly
    0.4000 against a majority-class rate of 0.4000, and balanced accuracy
    exactly 1/6 - i.e. it had learned the priors and stopped."""
    try:
        y_train = np.load(Path(args.data_dir) / "y_train.npy")
        counts = np.bincount(y_train.astype(int))
        priors = counts[counts > 0] / counts.sum()
        floor = float(-(priors * np.log(priors)).sum())
    except Exception:
        return
    _warn_on_imbalance(args, counts)
    print(f"[baseline] train prior-entropy CE floor = {floor:.4f} (ln(k) = "
          f"{np.log(len(priors)):.4f} for a uniform predictor). A run whose train loss settles "
          f"here has learned the class marginal and nothing else.", flush=True)
    for split, fname in (("val", "y_val.npy"), ("test", "y_test.npy")):
        try:
            y = np.load(Path(args.data_dir) / fname)
            c = np.bincount(y.astype(int))
            k = int((c > 0).sum()) or 1
            print(f"[baseline] {split}: majority-class accuracy {c.max() / c.sum():.4f}, "
                  f"balanced accuracy {1.0 / k:.4f} - the constant predictor's score.", flush=True)
        except Exception:
            continue

    # scripts/shallow_baseline_v16.py writes this next to the dataset. The
    # strongest probe in it is the bar a 0.447 M-parameter network has to
    # clear to have earned its complexity, and - unlike the majority-class
    # row - it also says whether the task is learnable at all from these
    # inputs, which is the first thing to know when a run sits at the floor.
    try:
        with open(Path(args.data_dir) / "shallow_baseline.json") as f:
            report = json.load(f)
        results = (report[0] if isinstance(report, list) else report)["results"]
    except Exception:
        return
    best = max((r for r in results.values() if r.get("f1_macro") is not None),
               key=lambda r: r["f1_macro"], default=None)
    if best is None:
        return
    name = next(k for k, v in results.items() if v is best)
    print(f"[baseline] best shallow probe ({name}, {best['dim']} features): "
          f"accuracy {best['accuracy']:.4f}, balanced {best['balanced_accuracy']:.4f}, "
          f"macro-F1 {best['f1_macro']:.4f}. A linear model reaching this means the signal is "
          f"present in these inputs; a network below it has an optimisation problem, not a data "
          f"problem.", flush=True)


def print_step_budget(args) -> None:
    """Whole clinical images instead of an 11x11 patch grid is ~270x fewer
    training samples, and the reference protocol's batch size is 4x smaller
    than the manuscript's runs used. Both cut the same way, so the protocol's
    150 epochs buys far fewer gradient steps than any published GMedMamba-R
    run - which is worth knowing BEFORE concluding anything from a model that
    sits at the prior."""
    try:
        n_train = int(len(np.load(Path(args.data_dir) / "y_train.npy", mmap_mode="r")))
    except Exception:
        return
    steps_per_epoch = max(1, -(-n_train // args.batch_size))
    total = steps_per_epoch * args.epochs
    print(f"[budget] {n_train:,} train samples / batch {args.batch_size} = {steps_per_epoch} "
          f"steps/epoch x {args.epochs} epochs = {total:,} optimizer steps", flush=True)
    print(f"[budget] the manuscript's PAD run took {_PUBLISHED_STEPS:,} steps "
          f"({_PUBLISHED_STEPS / max(1, total):.1f}x this) on the 11x11 patch grid. Whole-image "
          f"prep trades ~270x fewer samples for samples that actually contain the lesion; if the "
          f"model settles at the prior-entropy floor above, suspect the step budget before the "
          f"architecture.", flush=True)


def print_trm_cost(args) -> None:
    """The recursive core keeps every token at full resolution through every
    one of its ~63 applications, so the working set is (tokens x batch), and
    `--patch_size` moves it quadratically while `--batch_size` moves it
    linearly. Print both before training rather than discovering the product
    through a CUDA OOM in epoch 1."""
    side = _input_side(args.data_dir)
    if side is None or args.architecture != "recursive":
        return
    stride = args.patch_size or 1
    grid = max(1, side // stride)
    tokens = grid * grid
    rows = tokens * args.batch_size
    applications = args.trm_deep_supervision_steps * args.trm_n_improve * (args.trm_n_latent + 1)
    published_rows = _PUBLISHED_TOKENS * _PUBLISHED_BATCH
    gb = _GB_PER_TOKEN_ROW * rows * (1.0 if args.amp == "off" else 0.5)

    print(f"[trm-cost] input {side}x{side}, stem stride {stride} -> {grid}x{grid} = {tokens} "
          f"tokens/sample; core applied ~{applications}x per step", flush=True)
    print(f"[trm-cost] {tokens} tokens x batch {args.batch_size} = {rows:,} token-rows "
          f"({rows / published_rows:.1f}x the manuscript's 11x11 runs at "
          f"{_PUBLISHED_TOKENS}x{_PUBLISHED_BATCH}); rough activation estimate "
          f"~{gb:.1f} GB at --amp {args.amp}", flush=True)
    if gb > 8.0:
        print(f"[trm-cost] WARNING: that is likely to exhaust a 16 GB card. Raise --patch_size "
              f"(each doubling cuts tokens 4x) before lowering --batch_size, which is a protocol "
              f"setting.", flush=True)


def print_banner(args) -> None:
    print("=" * 78, flush=True)
    print("ORIGINAL MedMamba training protocol (arXiv:2403.03849, non-MedMNIST)", flush=True)
    print(f"  architecture       : {args.architecture}  (patch/stem stride {args.patch_size})", flush=True)
    print(f"  optimiser          : AdamW lr={args.lr} wd={args.weight_decay} "
          f"betas=(0.9,0.999) wd_groups={args.weight_decay_groups}", flush=True)
    print(f"  schedule           : {args.lr_schedule}", flush=True)
    print(f"  batch/epochs       : {args.batch_size} / {args.epochs}  "
          f"(early stop after {args.early_stop_patience} epochs without a "
          f"{args.checkpoint_metric} improvement)", flush=True)
    print(f"  precision          : amp={args.amp}   gradient clipping: "
          f"{'off' if not args.max_gradient_norm else args.max_gradient_norm}", flush=True)
    print(f"  objective          : loss={args.loss} sampler={args.sampler} "
          f"recon={args.recon_mode} ema={args.trm_ema_rate}", flush=True)
    print(f"  normalization      : {args.normalization} / {args.global_stats}"
          + ("  == transforms.Normalize((.5,.5,.5),(.5,.5,.5))"
             if args.normalization == "global_zscore" and args.global_stats == "medmamba_fixed"
             else ""), flush=True)
    print(f"  augmentation       : {args.augment_preset}", flush=True)

    deviations = protocol_deviations(args)
    if deviations:
        print("  DEVIATIONS from the protocol (explicit flags):", flush=True)
        for d in deviations:
            print(f"    - {d}", flush=True)
    else:
        print("  no deviations: every protocol default is in force.", flush=True)

    if args.normalization != "global_zscore" and args.global_stats == "medmamba_fixed":
        print(f"  NOTE: --global_stats medmamba_fixed is inert under "
              f"--normalization {args.normalization} (it only parameterises global_zscore).",
              flush=True)
    print("=" * 78, flush=True)
    print_trivial_baselines(args)
    print_step_budget(args)
    print_trm_cost(args)


def install_overrides(args) -> None:
    """Rebind v16's module-level names. Called once, before `v16._main()`.

    v17 S9 note: the other three entry points now pass a `v16.Overrides` value instead of
    assigning into `v16`'s namespace. This one deliberately does NOT, because it also
    rebinds `v16.compute_global_channel_stats` - a normalization helper, not one of the
    four composition points `Overrides` covers - and widening `Overrides` for a single
    caller would be a worse trade than leaving this file on the mechanism it was audited
    with. It keeps working because `Overrides.resolved()` fills every unset slot from
    `v16`'s module globals AT CALL TIME, so a legacy rebinding still wins.

    This file reproduces a published protocol; it is the path that should change least.
    """
    v16.build_arg_parser = build_arg_parser          # also reached by v16.explicitly_passed
    v16.build_run_name = build_run_name_original
    TrainerG_v12Original.early_stopping_enabled = (args.early_stopping == "on")
    TrainerG_v12Original.suppressed_stop_reasons = (
        frozenset() if args.stop_on_val_acc_stall == "on" else frozenset({StopReason.VAL_ACC_STALLED}))
    v16.TrainerG_v12 = TrainerG_v12Original

    v16.compute_global_channel_stats = {
        "medmamba_fixed": medmamba_fixed_channel_stats,
        "identity": identity_channel_stats,
        "train_fit": _V16_GLOBAL_STATS,
    }[args.global_stats]

    if args.lr_schedule == "constant":
        v16.build_warmup_cosine_scheduler = build_constant_scheduler
    else:
        v16.build_warmup_cosine_scheduler = _V16_SCHEDULER


def main() -> None:
    # v16._main parses again from the same (patched) parser; argparse is pure,
    # so this preview parse is free of side effects and is only here because
    # the overrides have to be chosen before _main starts.
    args = build_arg_parser().parse_args()
    print_banner(args)
    install_overrides(args)
    try:
        v16.main()
    except torch.OutOfMemoryError:
        # v16.main() writes failure_class.json and re-raises; the traceback
        # alone does not say which of three knobs to reach for, and two of
        # them cost protocol fidelity while one does not.
        side = _input_side(args.data_dir)
        grid = (side // (args.patch_size or 1)) if side else None
        print("\n" + "=" * 78, flush=True)
        print("CUDA OUT OF MEMORY - the recursive core's working set is (tokens x batch), and "
              "every one of its ~63 applications stores a checkpointed activation.", flush=True)
        if grid:
            print(f"  this run: {grid}x{grid} = {grid * grid} tokens x batch {args.batch_size}", flush=True)
        print(f"  --patch_size {(args.patch_size or 1) * 2}   halves the grid, quartering activations. "
              f"Costs spatial resolution, NOT a protocol setting (see the docstring).", flush=True)
        print(f"  --batch_size {max(1, args.batch_size // 4)}   same saving, but batch size IS a "
              f"protocol setting and changes optimisation.", flush=True)
        print(f"  --amp bf16       halves activations, but the reference protocol is fp32.", flush=True)
        print("=" * 78, flush=True)
        raise


if __name__ == "__main__":
    main()
