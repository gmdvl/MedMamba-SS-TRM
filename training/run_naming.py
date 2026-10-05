# -*- coding: utf-8 -*-
"""
training/run_naming.py
========================
Builds the experiment directory name for `train_example_v14.py`.

Replaces the old `gmedmamba_stageG_{architecture}_run_{timestamp}` scheme,
which recorded neither the dataset nor any setting that changed the result -
so `experiments/` filled with directories whose names were indistinguishable
from each other and from `train_example_v13.py`'s (v13 and v14 passed the
*same* `exp_name`). Finding a specific run meant opening `config.json` in
each one.

The new format is timestamp-first, so a plain `ls experiments/` still sorts
chronologically, followed by the few fields that actually change the outcome:

    {timestamp}_{dataset}_{architecture}_{modality}_{key}_bs{N}_{amp}

    20260901_140126_pad_v6_recursive_rgb_mlp_sup3_bs256_bf16
    20260901_151203_hsi_v7-hsi_split_hsi_bs256_bf16
    20260901_163344_pad_v6_efficient_rgb_segate_bs256_fp32

`{key}` is architecture-dependent, because the flag that matters most differs
per architecture and there is no point spending name length on a flag the
architecture ignores (see `documentations/PARAMETERS.md` section 3):

    recursive           -> `{trm_mixer}_sup{trm_deep_supervision_steps}`
    efficient           -> `{fusion_type}`
    split/fullchannel   -> omitted entirely

This module deliberately holds no import of `train_example_v14`, `torch`, or
any model code: it is pure path/string manipulation over an argparse
namespace, so `test_run_naming.py` can exercise it without a GPU or a
dataset.

NOTE for anyone adding a caller: `train_example_v14.py` needs this name
*before* `discover_data()` runs, because the experiment directory must exist
early enough for `write_dataset_failure_artifacts()` and `failure_class.json`
to have somewhere to land when the dataset gate itself fails. That is why
`infer_modality()` below re-implements `discover_data`'s one-line rule rather
than taking the modality as an argument - see its docstring.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Path components that carry no information on their own: every prepared
# dataset has an `hsi/` and/or `rgb/` subdirectory, and `--data_dir
# ./data/hsi_v7/hsi` would otherwise slug to the useless "hsi".
_UNINFORMATIVE_LEAF = {"hsi", "rgb", "data", "dataset", "npy", "out", "output"}

# Keep directory names comfortably under a terminal width once the timestamp,
# architecture, modality, batch size and AMP mode are appended.
_MAX_SLUG_LEN = 24

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def dataset_slug(data_dir: str, max_len: int = _MAX_SLUG_LEN) -> str:
    """Short, filesystem-safe identifier for a `--data_dir`.

    Uses the last path component, unless that component is uninformative on
    its own (`hsi`, `rgb`, ...), in which case the parent is prepended:

        ./data/pad_v6/                        -> "pad_v6"
        ./data/hsi_v7/hsi/                    -> "hsi_v7-hsi"
        ./data_hsi_v6-v4-nb32_mb_40_v2.1/hsi  -> "v6-v4-nb32_mb_40_v2.1-hsi"

    Any character outside `[A-Za-z0-9._-]` collapses to a single `-`, and an
    over-long result is truncated from the LEFT, keeping the tail: dataset
    directories in this repo are versioned by suffix (`_v6`, `_v2.1`,
    `-nb32_mb_40`), so the distinguishing part is at the end.
    """
    parts = [p for p in Path(data_dir).parts if p not in ("", "/", ".")]
    if not parts:
        return "dataset"

    leaf = parts[-1]
    if leaf.lower() in _UNINFORMATIVE_LEAF and len(parts) >= 2:
        slug = f"{parts[-2]}-{leaf}"
    else:
        slug = leaf

    slug = _SANITIZE_RE.sub("-", slug).strip("-._")
    if not slug:
        return "dataset"
    if len(slug) > max_len:
        slug = slug[-max_len:].lstrip("-._") or slug[-max_len:]
    return slug


def infer_modality(data_dir: str) -> str:
    """"hsi" | "rgb", using the SAME rule as
    `train_example_v6.discover_data` (train_example_v6.py:161,175): the
    presence of a file named exactly `wavelengths.npy` at the top level of
    `data_dir`.

    Duplicated here on purpose. `discover_data` also mmaps the label arrays
    and validates the four required `.npy` files, and it runs *after* the
    experiment directory has to exist - so calling it early just to learn the
    modality would move the dataset gate ahead of the directory that its own
    failure artifacts are written into. `train_example_v14._main()` asserts
    the two agree once `discover_data` has actually run.

    Note this is a filename check, not a content check: `selected_wavelengths.npy`
    and `wavelengths_full.npy` do NOT count.

    v15 R6.6 - the RETURN VALUE deliberately still mirrors `discover_data`
    exactly, because `train_example_v14/v15._main` cross-check the two and a
    "helpful" guess here would turn a dataset problem into a spurious
    `INTERNAL: modality mismatch`. The content check lives in
    `modality_warnings()` instead, which the entry points print LOUDLY - so a
    spectral dataset that is about to train as `rgb` says so, in the log, in
    the words of the thing that is actually wrong.
    """
    return "hsi" if (Path(data_dir) / "wavelengths.npy").is_file() else "rgb"


def modality_evidence(data_dir: str) -> dict:
    """What `infer_modality`'s decision rests on, plus the things that would
    make it WRONG. Pure filesystem inspection; no arrays are read."""
    base = Path(data_dir)
    return {
        "canonical_wavelengths": (base / "wavelengths.npy").is_file(),
        "other_wavelength_files": sorted(p.name for p in base.glob("*wavelength*.npy")
                                          if p.name != "wavelengths.npy"),
        "shard_dirs": sorted(p.name for p in list(base.glob("*_batches")) + [base / "_batches"]
                              if p.is_dir()),
        "merged_x_train": (base / "X_train.npy").is_file(),
        "inferred_modality": infer_modality(data_dir),
    }


def modality_warnings(data_dir: str) -> list:
    """Warning strings for a dataset whose modality is not unambiguous
    (v15 R6.6). Empty list = nothing to say.

    Both cases here are ways a 32-band cube ends up training silently against
    an RGB preset: a wavelength file under a non-canonical name, and a
    preprocessing run whose finalize/merge step never produced the merged
    arrays."""
    ev = modality_evidence(data_dir)
    out = []
    if ev["inferred_modality"] == "rgb" and ev["other_wavelength_files"]:
        out.append(
            f"MODALITY: {data_dir} has no `wavelengths.npy` but does contain "
            f"{ev['other_wavelength_files']}. `train_example_v6.discover_data` looks ONLY for the "
            f"canonical name, so this dataset will train as RGB with index positional encoding, "
            f"and every sensor-aware code path stays inert. If it is spectral, copy or rename the "
            f"file to `wavelengths.npy` and re-run.")
    if ev["shard_dirs"] and not ev["merged_x_train"]:
        out.append(
            f"DATASET: {data_dir} contains per-batch shard directories {ev['shard_dirs']} but no "
            f"merged `X_train.npy` - the preprocessing finalize/merge step never ran. Re-run the "
            f"dataset merge before training.")
    elif ev["shard_dirs"]:
        out.append(
            f"DATASET: {data_dir} still contains shard directories {ev['shard_dirs']} alongside "
            f"the merged arrays. They are ignored here, but check they are not a stale "
            f"half-merge from an interrupted prep.")
    return out


def _architecture_key(args) -> str:
    """The one or two fields that most change the result for this
    architecture, or "" when the architecture has none worth naming."""
    arch = getattr(args, "architecture", None)
    if arch == "recursive":
        mixer = getattr(args, "trm_mixer", "mlp")
        sup = getattr(args, "trm_deep_supervision_steps", None)
        mixer = {"attention": "attn"}.get(mixer, mixer)
        return f"{mixer}_sup{sup}" if sup is not None else str(mixer)
    if arch == "efficient":
        # "se_gate" -> "segate": underscores already separate name fields.
        return str(getattr(args, "fusion_type", "")).replace("_", "")
    return ""


# v15 R6.2 - the settings that changed the RESULT and were invisible in the
# old name. `(dest, default, formatter)`; a value equal to `default` is
# omitted, so a run at the defaults keeps the short v14-style name. The two
# evidence runs differed in `--loss`, `--sampler`, `--train_subsample_frac` and
# `--dataset_storage`, and their directory names said none of it - while v14's
# own docstring claimed the subsample fraction was "recorded in config.json
# and in the run directory name".
_NAME_ALLOWLIST = (
    # v15's headline ablation axis: three runs differing only in the tokenizer
    # would otherwise get identical directory names.
    ("spectral_token_fusion", "concat_mlp",
     lambda v: {"concat_mlp": "cmlp", "scaled": "scaled", "add": "ADD-legacy"}.get(str(v), str(v))),
    ("loss", "ce", lambda v: {"weighted_ce": "wce", "focal_weighted": "focalw"}.get(str(v), str(v))),
    ("sampler", "none", lambda v: {"moderate_oversample": "modover",
                                    "balanced": "balsamp"}.get(str(v), str(v))),
    ("normalization", None, lambda v: {"global_zscore": "zscore",
                                        "per_sample_minmax": "minmax"}.get(str(v), str(v))),
    ("train_subsample_frac", 1.0, lambda v: f"sub{_frac_token(v)}"),
    ("train_subsample_mode", "per_epoch", lambda v: str(v)),
    ("val_subsample_frac", 1.0, lambda v: f"vsub{_frac_token(v)}"),
    ("seed", 42, lambda v: f"seed{int(v)}"),
)


def _frac_token(value: float) -> str:
    """0.05 -> "05", 0.5 -> "5", 0.125 -> "125" - a filesystem-safe fraction
    with no decimal point, which would otherwise read as a file extension."""
    return f"{float(value):g}".replace("0.", "").replace(".", "")


def _settings_suffix(args) -> list:
    """The non-default allowlisted settings, in a fixed order so two runs with
    the same settings always get the same name."""
    out = []
    for dest, default, fmt in _NAME_ALLOWLIST:
        if not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        if value is None or value == default:
            continue
        out.append(fmt(value))
    return out


def build_run_name(args, timestamp: Optional[str] = None,
                   modality: Optional[str] = None) -> str:
    """`{ts}_{dataset}_{arch}_{modality}_{key}_bs{N}_{amp}[_{settings}]`.

    `timestamp` defaults to local wall-clock `%Y%m%d_%H%M%S`, matching
    `train_example_v6.setup_experiment_dir`'s own format so v13 and v14 runs
    remain sortable against each other. `modality` defaults to
    `infer_modality(args.data_dir)`.

    v15 R6.2 - non-default values of the `_NAME_ALLOWLIST` settings are
    appended, e.g. `..._bs256_bf16_wce_sub05_seed7`. An argparse namespace
    that lacks a field (v14's, which has no `--normalization`) simply
    contributes nothing, so this stays a drop-in for older callers.

    Pass the result to `setup_experiment_dir(exp_name=..., append_timestamp=False)`
    - this function has already placed the timestamp at the front.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    mod = modality or infer_modality(args.data_dir)
    amp = getattr(args, "amp", "off")
    amp_token = "fp32" if amp == "off" else str(amp)

    fields = [
        ts,
        dataset_slug(args.data_dir),
        str(getattr(args, "architecture", "model")),
        mod,
        _architecture_key(args),
        f"bs{getattr(args, 'batch_size', '?')}",
        amp_token,
        *_settings_suffix(args),
    ]
    return "_".join(f for f in fields if f)
