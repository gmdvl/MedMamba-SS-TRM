# -*- coding: utf-8 -*-
"""
training/train_profiles.py
===========================
Every training configuration `train.py` can run, as DATA.

Before 2026-09-28 each configuration was its own entry point stacked on the one
below it (`train_example_v16.py` -> `_original` / `_optimal` -> `_optimal_recon`
-> `train_example_v18.py`), and a configuration was the ORDER in which those files
called `parser.set_defaults`. That order is kept exactly - it is what the parity
tests pin - but it is now written down in one place, as a list of layers per
profile:

    profile                     layers, applied in order (later wins; explicit flags win over all)
    pipeline_defaults           -                                                       (train_example_v16.py)
    medmamba_protocol_norecon   MEDMAMBA_PROTOCOL_DEFAULTS, patch by arch               (train_example_v16_original.py)
    pad_ufes_best_norecon       OPTIMAL_DEFAULTS, resolved, stage                       (train_example_v16_optimal.py)
    paper_recipe                pad_ufes_best_norecon + RECON_DEFAULTS                  (train_example_v18.py)

A profile is named for what it is FOR, and a name ending in `_norecon` says the
profile trains with `recon_mode none` (no reconstruction head, no MSE/SAM
term); every other profile reconstructs. `tests/test_profile_names.py` holds
the names to that. `paper_recipe` (the default since 2026-10-02) is the
configuration behind every number in the paper. The names before 2026-10-01
(`base`, `original`, `optimal`, `v18`) still work: see `PROFILE_ALIASES`.

The evidence behind each `OPTIMAL_DEFAULTS` / `MEDMAMBA_PROTOCOL_DEFAULTS` value
is the module docstring of the archived entry point that introduced it
(`archive/train_example_v16_optimal.py`, `archive/train_example_v16_original.py`).

A "resolved" layer computes defaults from the dataset (`--data_dir`, read from
the .npy headers only) or from other flags on the command line, which is why
layers take `argv`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np

# ============================================================================
# the configurations
# ============================================================================

# `train_example_v16_original.py` - MedMamba-SS-TRM under the original MedMamba
# training protocol (arXiv:2403.03849, non-MedMNIST).
MEDMAMBA_PROTOCOL_DEFAULTS = {
    "epochs": 150, "batch_size": 64, "lr": 1e-4, "weight_decay": 1e-4,
    "weight_decay_groups": False,      # reference: AdamW(model.parameters(), weight_decay=..)
    "max_gradient_norm": 0.0,          # falsy -> not clipped
    "amp": "off",                      # the reference trains in full precision
    "early_stop_patience": 20, "checkpoint_metric": "val_accuracy",
    "loss": "ce", "sampler": "none", "class_weight_power": 1.0,
    "recon_mode": "none", "lambda_mse": 0.0, "lambda_sam": 0.0, "lambda_gan": 0.0,
    "normalization": "global_zscore",  # with --global_stats medmamba_fixed == Normalize(.5, .5)
    "augment_preset": "none", "train_subsample_frac": 1.0, "val_subsample_frac": 1.0,
    "eval_group_aggregation": "none",
    "architecture": "recursive",
    "patch_size": 8,                   # replaced per architecture, see PATCH_SIZE_BY_ARCHITECTURE
    "classifier_dropout": 0.0, "drop_path_rate": 0.0,
    "trm_ema_rate": 0.0, "trm_halting": False,
    "val_divergence_patience": 0, "on_class_collapse": "warn",
    "eval_test": "best", "seed": 42,
    # the four knobs the protocol entry point added as its own flags
    "lr_schedule": "constant", "global_stats": "medmamba_fixed",
    "early_stopping": "off", "stop_on_metric_stall": "off",
}

# MedMamba's stride 4 matches on the hierarchical backbones, which downsample per
# stage; the recursive core never does, and 4 there OOMs a 16 GB card at batch 64.
PATCH_SIZE_BY_ARCHITECTURE = {"recursive": 8, "split": 4, "fullchannel": 4, "efficient": 4}

# `train_example_v16_optimal.py` - best achievable metrics on PAD-UFES-20.
OPTIMAL_DEFAULTS = {
    # (1) the collapse: objective
    "loss": "focal_weighted", "focal_gamma": 1.5, "class_weight_power": 0.75, "sampler": "none",
    # (2) model selection
    "checkpoint_metric": "f1_macro",
    # (3) normalization
    "normalization": "global_zscore", "on_split_drift": "warn",
    # (4) step budget / LR
    "batch_size": 32, "lr": 3e-4, "epochs": 200, "weight_decay": 0.05,
    "weight_decay_groups": True, "scheduler_interval": "step",
    "warmup_steps": None,              # -> 3% of total steps, floor 200
    "max_gradient_norm": 1.0,
    # (5) regularisation - band_dropout / spectral_mask OFF: on RGB they zero a colour channel
    "augment_preset": "custom", "flip_h_prob": 0.5, "flip_v_prob": 0.5, "rotate90_prob": 0.5,
    "crop_scale_max_frac": 0.3, "spectral_scale_range": 0.15, "spectral_offset_std": 0.05,
    "spectral_noise_std": 0.02, "band_dropout_prob": 0.0, "spectral_mask_prob": 0.0,
    "classifier_dropout": 0.1,
    "trm_dropout": 0.0,                # NOT 0.1: the core is re-entered ~63x per forward
    "drop_path_rate": 0.0, "trm_drop_path": 0.0,
    # trm_ema_rate is resolved from steps/epoch - see resolve_ema_rate
    # (6) stopping rules
    "early_stop_patience": 40, "val_divergence_patience": 0,
    "on_class_collapse": "warn", "class_collapse_streak": 10,
    "stop_on_metric_stall": "off",
    # (7) cost / architecture
    "architecture": "recursive", "trm_dim": 128, "trm_mixer": "mlp", "trm_halting": False,
    "amp": "bf16", "recon_mode": "none", "lambda_mse": 0.0, "lambda_sam": 0.0, "lambda_gan": 0.0,
    # (8) throughput (v17 S2/S3/S12, measured)
    "loader_mode": "performance", "eval_artifact_stride": 5, "fast_loop": "on",
    "compile_model": "on",
    # reporting
    "eval_test": "best", "verify_best_checkpoint": True,
    "train_subsample_frac": 1.0, "val_subsample_frac": 1.0, "seed": 42,
}

# `--stage` - the debugging ladder of the optimal family: prove the model can fit,
# THEN correct the imbalance, THEN regularise. Applied over the profile's defaults.
STAGES = {
    "fit": {
        "loss": "ce", "class_weight_power": 0.0, "sampler": "none",
        "augment_preset": "none", "classifier_dropout": 0.0, "trm_dropout": 0.0,
        "drop_path_rate": 0.0, "trm_drop_path": 0.0,
        "trm_ema_rate": 0.0,          # validate the LIVE weights, no EMA lag
        "weight_decay": 0.0, "recon_mode": "none",
        "lambda_mse": 0.0, "lambda_sam": 0.0,
        "epochs": 40, "early_stop_patience": 0,
    },
    "balance": {
        "augment_preset": "none", "classifier_dropout": 0.0, "trm_dropout": 0.0,
        "trm_ema_rate": 0.0, "epochs": 60,
    },
    "full": {},
}

# `train_example_v16_optimal_recon.py` before 2026-09-28: a reconstruction run
# that reconstructs nothing is not worth the wall clock. `lambda_sam` is NOT here -
# every v18 HSI command passes `--lambda_sam 0.1` explicitly.
RECON_DEFAULTS = {"recon_mode": "latent", "lambda_mse": 0.1}


# ============================================================================
# defaults resolved from the dataset
# ============================================================================

TARGET_TOKEN_GRID = 28              # a throughput choice: 28x28 at batch 32 is 1.4 GB
EMA_HORIZON_EPOCHS = 3.0
EMA_RATE_BOUNDS = (0.90, 0.9995)
TARGET_TRAIN_PATCHES = 250_000      # rank-1 HSI ran 245,209 (0.1 of 2,452,086)
TARGET_VAL_PATCHES = 33_000


def input_side(data_dir: Optional[str]) -> Optional[int]:
    """H of `X_train.npy`, from the .npy header alone. None, never an exception:
    this feeds a default, and a missing file has to reach the pipeline's own
    error handling rather than dying in the parser."""
    try:
        shape = np.load(Path(data_dir) / "X_train.npy", mmap_mode="r").shape
        return int(shape[1]) if len(shape) == 4 else None
    except Exception:
        return None


def split_size(data_dir: Optional[str], split: str) -> Optional[int]:
    """len(y_<split>.npy) from the header alone. None, never an exception."""
    if not data_dir:
        return None
    try:
        return int(len(np.load(Path(data_dir) / f"y_{split}.npy", mmap_mode="r")))
    except Exception:
        return None


def steps_per_epoch(data_dir: Optional[str], batch_size: int,
                    subsample_frac: float = 1.0) -> Optional[int]:
    """Optimizer steps in ONE epoch, over what `--train_subsample_frac` leaves."""
    n = split_size(data_dir, "train") if data_dir else None
    if n is None:
        return None
    kept = max(1, int(round(n * max(0.0, min(1.0, subsample_frac)))))
    return max(1, -(-kept // max(1, batch_size)))


def resolve_patch_size(data_dir: Optional[str], target_grid: int = TARGET_TOKEN_GRID) -> Optional[int]:
    """Stem stride giving a token grid no finer than `target_grid` per side
    (224 -> 8, 112 -> 4, 11 -> 1). None keeps the architecture preset."""
    side = input_side(data_dir) if data_dir else None
    if side is None:
        return None
    return max(1, side // target_grid)


def resolve_subsample_frac(n: Optional[int], target: int) -> float:
    """Fraction of a split that makes one epoch ~`target` samples; 1.0 when the
    split is already at or below it."""
    if not n or n <= target:
        return 1.0
    return round(target / n, 4)


def resolve_ema_rate(data_dir: Optional[str], batch_size: int,
                     horizon_epochs: float = EMA_HORIZON_EPOCHS, subsample_frac: float = 1.0) -> float:
    """EMA decay giving an averaging horizon of `horizon_epochs`. The shadow starts
    at the random init with no bias correction and validation runs on it, so a
    rate that ignores steps/epoch leaves validation measuring noise for dozens of
    epochs (archive/train_example_v16_optimal.py, `resolve_ema_rate`)."""
    spe = steps_per_epoch(data_dir, batch_size, subsample_frac)
    if spe is None:
        return 0.99
    lo, hi = EMA_RATE_BOUNDS
    return float(min(hi, max(lo, 1.0 - 1.0 / max(1.0, horizon_epochs * spe))))


# ============================================================================
# the layers
# ============================================================================

def peek(flag: str, default=None, type=str, argv=None):
    """One flag off the command line before the real parser exists. A throwaway
    parser rather than a string scan, so `--flag=x`, abbreviations and `--`
    behave as argparse says they should."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(flag, type=type, default=default)
    return getattr(p.parse_known_args(argv)[0], flag.lstrip("-"))


Layer = Union[Dict, Callable[[argparse.ArgumentParser, Optional[list]], Dict]]


def _protocol_patch_size(p, argv) -> Dict:
    arch = peek("--architecture", MEDMAMBA_PROTOCOL_DEFAULTS["architecture"], argv=argv)
    return {"patch_size": PATCH_SIZE_BY_ARCHITECTURE.get(arch, MEDMAMBA_PROTOCOL_DEFAULTS["patch_size"])}


def _optimal_resolved(p, argv) -> Dict:
    """Stem stride, then the epoch budget, then the EMA horizon that depends on
    it - in that order, because `resolve_ema_rate` needs the frac in force."""
    data_dir = peek("--data_dir", argv=argv)
    out = {"patch_size": resolve_patch_size(
        data_dir, peek("--target_token_grid", TARGET_TOKEN_GRID, int, argv))}
    train_frac = resolve_subsample_frac(split_size(data_dir, "train"), TARGET_TRAIN_PATCHES)
    out["train_subsample_frac"] = train_frac
    out["val_subsample_frac"] = resolve_subsample_frac(
        split_size(data_dir, "val") or split_size(data_dir, "test"), TARGET_VAL_PATCHES)
    out["trm_ema_rate"] = resolve_ema_rate(
        data_dir, peek("--batch_size", p.get_default("batch_size"), int, argv),
        subsample_frac=peek("--train_subsample_frac", train_frac, float, argv))
    return out


def _stage(p, argv) -> Dict:
    return STAGES.get(peek("--stage", "full", argv=argv), {})


@dataclass(frozen=True)
class Profile:
    layers: List[Layer]
    # Appended to the run directory name. It is the archived entry point's suffix
    # and does NOT follow the profile renames: the parity tests pin run names
    # byte for byte, and existing run paths (sweeps/paper_runs.py:HEAD42) contain it.
    name_suffix: str
    summary: str
    stages: bool = False    # accepts --stage


_OPTIMAL_LAYERS: List[Layer] = [OPTIMAL_DEFAULTS, _optimal_resolved, _stage]

PROFILES: Dict[str, Profile] = {
    "pipeline_defaults": Profile(
        [], "",
        "the bare pipeline, parser defaults only: CE, 10 epochs, latent reconstruction at "
        "lambda_mse 1.0 (train_example_v16.py)"),
    "medmamba_protocol_norecon": Profile(
        [MEDMAMBA_PROTOCOL_DEFAULTS, _protocol_patch_size], "_medmamba_protocol",
        "MedMamba-SS-TRM under the original MedMamba training protocol, to compare against the "
        "MedMamba baseline; NO reconstruction (train_example_v16_original.py)"),
    "pad_ufes_best_norecon": Profile(
        _OPTIMAL_LAYERS, "_optimal",
        "best metrics on PAD-UFES-20 (RGB); NO reconstruction (train_example_v16_optimal.py)",
        stages=True),
    "paper_recipe": Profile(
        _OPTIMAL_LAYERS + [RECON_DEFAULTS], "_optimal_recon",
        "the configuration behind every number in the paper, with latent reconstruction "
        "(lambda_mse 0.1); HSI runs also pass --lambda_sam 0.1 (train_example_v18.py)",
        stages=True),
}
DEFAULT_PROFILE = "paper_recipe"

# The names before 2026-10-01. `--profile` still accepts them and resolves them
# to the profile on the right; nothing records the old name.
PROFILE_ALIASES: Dict[str, str] = {
    "base": "pipeline_defaults",
    "original": "medmamba_protocol_norecon",
    "optimal": "pad_ufes_best_norecon",
    "v18": "paper_recipe",
}


def canonical_profile(name: str) -> str:
    """A profile's current name. An old name maps to its replacement; anything
    else comes back unchanged, so argparse's `choices` still rejects it."""
    return PROFILE_ALIASES.get(name, name)


def apply_profile(p: argparse.ArgumentParser, profile: str, argv=None) -> None:
    """Apply `profile`'s layers to `p` as argparse DEFAULTS, in order, so every
    value stays overridable on the command line."""
    known = {a.dest for a in p._actions}
    for layer in PROFILES[profile].layers:
        values = layer(p, argv) if callable(layer) else layer
        unknown = set(values) - known
        if unknown:
            # A silently-ignored key would produce a run that claims this
            # configuration while not using it - fail loudly instead.
            raise RuntimeError(f"profile {profile!r} names flag(s) the parser does not have: "
                               f"{sorted(unknown)}")
        p.set_defaults(**values)
