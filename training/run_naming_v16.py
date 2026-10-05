# -*- coding: utf-8 -*-
"""
training/run_naming_v16.py
=============================
MedMamba-SS-TRM v16 plan, Stage 1.1. `training/run_naming.py` is imported at
`train_example_v15.py:130` and is therefore FROZEN.

Imports `dataset_slug`, `infer_modality` and `modality_warnings` from the
frozen module unchanged, and redefines only `_NAME_ALLOWLIST` +
`_settings_suffix` + `build_run_name`, because `_settings_suffix` in the
frozen module reads the module-level `_NAME_ALLOWLIST` tuple directly
(`run_naming.py:213`) and cannot be parameterised from outside.

Two changes vs the frozen version:

  A-2  `"per_patch_zscore": "ppz"` added to the normalization token map, so
       the v16 audit's fix (`training/normalization.py`) is visible in the
       run directory name instead of falling through to `str(v)`.

  `_frac_token` bug fix (v16 copy only, per the plan): the frozen
       `_frac_token` renders `0.1` as `"1"` (`f"{0.1:g}".replace("0.", "").
       replace(".", "")` -> `"1"`), which is indistinguishable from `1.0`, so
       a directory named `sub1_vsub1` from the frozen naming module actually
       means *0.1*. Prepending `"0"` for any value < 1 makes that `"01"`
       instead. Fixing it in the frozen module would rename future v15 runs,
       which the freeze forbids - so this is a v16-only fix; existing v15/v11
       run directories keep their (ambiguous) names.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from training.run_naming import dataset_slug, infer_modality, modality_warnings  # frozen

__all__ = ["dataset_slug", "infer_modality", "modality_warnings", "build_run_name",
           "_NAME_ALLOWLIST", "_frac_token"]


def _frac_token(value: float) -> str:
    """0.05 -> "05", 0.5 -> "5", 0.125 -> "0125", 0.1 -> "01" (v16 fix: the
    frozen `run_naming._frac_token` renders 0.1 and 1.0 identically as "1")."""
    token = f"{float(value):g}".replace("0.", "").replace(".", "")
    if float(value) < 1.0 and not token.startswith("0"):
        token = f"0{token}"
    return token


def _architecture_key(args) -> str:
    """Byte-identical to `training.run_naming._architecture_key` (not
    exported by the frozen module, so ported rather than imported)."""
    arch = getattr(args, "architecture", None)
    if arch == "recursive":
        mixer = getattr(args, "trm_mixer", "mlp")
        sup = getattr(args, "trm_deep_supervision_steps", None)
        mixer = {"attention": "attn"}.get(mixer, mixer)
        return f"{mixer}_sup{sup}" if sup is not None else str(mixer)
    if arch == "efficient":
        return str(getattr(args, "fusion_type", "")).replace("_", "")
    return ""


_NAME_ALLOWLIST = (
    ("spectral_token_fusion", "concat_mlp",
     lambda v: {"concat_mlp": "cmlp", "scaled": "scaled", "add": "ADD-legacy"}.get(str(v), str(v))),
    ("loss", "ce", lambda v: {"weighted_ce": "wce", "focal_weighted": "focalw"}.get(str(v), str(v))),
    ("sampler", "none", lambda v: {"moderate_oversample": "modover",
                                    "balanced": "balsamp"}.get(str(v), str(v))),
    # A-2 - "per_patch_zscore" -> "ppz", so the v16 audit's normalization fix
    # shows up in the run name instead of falling through to str(v).
    ("normalization", None, lambda v: {"global_zscore": "zscore",
                                        "per_sample_minmax": "minmax",
                                        "per_patch_zscore": "ppz"}.get(str(v), str(v))),
    ("train_subsample_frac", 1.0, lambda v: f"sub{_frac_token(v)}"),
    ("train_subsample_mode", "per_epoch", lambda v: str(v)),
    ("val_subsample_frac", 1.0, lambda v: f"vsub{_frac_token(v)}"),
    ("seed", 42, lambda v: f"seed{int(v)}"),
)


def _settings_suffix(args) -> list:
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
    """Same contract as `training.run_naming.build_run_name`, using this
    module's `_NAME_ALLOWLIST` (with `per_patch_zscore` and the `_frac_token`
    fix) instead of the frozen one's."""
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
