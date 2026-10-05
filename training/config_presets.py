# -*- coding: utf-8 -*-
"""
training/config_presets.py
============================
Centralizes the (dims, depths, ...) presets that `medmamba_ss_trm.py`'s /
`medmamba_ss_fullchannel.py`'s / `medmamba_ss_efficient.py`'s convenience
constructors (`medmamba_ss_hsi_small`, `medmamba_ss_fullchannel_hsi_small`,
`medmamba_ss_efficient_hsi_small`, ...) build internally, so Stage E's
`train_example_v11.py` can override `drop_path_rate` (Phase 32) before
construction.

This is needed because those convenience constructors build a
`MedMambaSSTRMConfig` internally and only forward `**kwargs` to the model
class's `task`/`num_classes`/head-kwargs - there's no way to reach
`drop_path_rate` (or any other `MedMambaSSTRMConfig` field) through them. This
module builds the same config, but as data you can override first.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSS, MedMambaSSTRM
from medmamba_ss_fullchannel import MedMambaSSFullChannel
from medmamba_ss_efficient import MedMambaSSEfficient

_PRESETS = {
    ("split", "hsi"): dict(dims=(64, 128, 256), depths=(2, 2, 2), d_state=8, d_ctx=64,
                            patch_size=1, spectral_depth=3),
    ("split", "rgb"): dict(dims=(96, 192, 384, 768), depths=(2, 2, 4, 2), patch_size=4),
    ("fullchannel", "hsi"): dict(dims=(64, 128, 256), depths=(2, 2, 2), d_state=8, d_ctx=64,
                                  patch_size=1, spectral_depth=3),
    ("fullchannel", "rgb"): dict(dims=(96, 192, 384, 768), depths=(2, 2, 4, 2), patch_size=4),
    ("efficient", "hsi"): dict(dims=(64, 128, 256), depths=(2, 2, 2), d_state=8, d_ctx=64,
                                patch_size=1, spectral_depth=3),
    ("efficient", "rgb"): dict(dims=(96, 192, 384, 768), depths=(2, 2, 4, 2), patch_size=4),
    # TRM-style recursive variant (medmamba_ss_trm.MedMambaSSTRM): a single
    # weight-shared recursive core instead of a hierarchical backbone.
    # Channel-agnostic + patch_size=1, so hsi/rgb share the same shape - both
    # real datasets are 11x11 patches (HSI: C=32, PAD-UFES RGB: C=3).
    ("recursive", "hsi"): dict(recursive=True, dims=(128,), depths=(1,), d_state=8, d_ctx=64,
                                patch_size=1, spectral_depth=3, trm_dim=128),
    ("recursive", "rgb"): dict(recursive=True, dims=(128,), depths=(1,), d_state=8, d_ctx=64,
                                patch_size=1, spectral_depth=3, trm_dim=128),
}
_MODEL_CLASSES = {
    "split": MedMambaSS, "fullchannel": MedMambaSSFullChannel, "efficient": MedMambaSSEfficient,
    "recursive": MedMambaSSTRM,
}


# v15 - the settings `train_example_v15.py` flips relative to the legacy
# defaults, in one place so `test_representation_sensitivity.py` and the
# training entry point cannot drift apart. Every key is a `MedMambaSSTRMConfig`
# field whose default reproduces pre-v15 behaviour; see the v15 plan, stage R1.
V15_REPRESENTATION_OVERRIDES = {
    # R1.1  - a tokenizer that can see its input (scale AND rank).
    "spectral_token_fusion": "concat_mlp",
    "spectral_pe_gain": 0.1,
    "spectral_value_init_std": 0.5,
    # R1.1b - so `stem_norm` operates above its own eps instead of degenerating
    #         into a constant rescale.
    "spectral_ctx_norm": True,
    # R1.1d - the classifier's final projection at its own fan-in scale.
    "classifier_init": "fan_in",
    # R1.3  - the wavelength encoding's angular span = the band count, which
    #         makes it reduce to the index encoding for a uniform sensor.
    "wavelength_encoding_scale": None,
}

# Recursive-only additions (rejected by the hierarchical architectures).
V15_RECURSIVE_OVERRIDES = {
    # R1.1c - the 2-D spatial positional encoding is the same per-dataset
    #         constant one layer below the tokenizer's.
    "trm_spatial_pe_gain": 0.1,
    # R3.3  - `--gradient_checkpointing` actually controls the core.
    "trm_checkpoint_core": True,
}


def v15_config_overrides(architecture: str) -> dict:
    """The `cfg_overrides` dict `train_example_v15.py` passes to `build_model`
    for `architecture`. Recursive-only fields are omitted for the hierarchical
    architectures, where they are inert."""
    out = dict(V15_REPRESENTATION_OVERRIDES)
    if architecture == "recursive":
        out.update(V15_RECURSIVE_OVERRIDES)
    return out


def build_model(architecture: str, modality: str, num_classes: int,
                 drop_path_rate: Optional[float] = None, fusion_type: Optional[str] = None,
                 classifier_dropout: Optional[float] = None, trm_kwargs: Optional[dict] = None,
                 cfg_overrides: Optional[dict] = None):
    """`architecture`: 'split' | 'fullchannel' | 'efficient' | 'recursive'.
    `modality`: 'hsi' | 'rgb'. `fusion_type`/`classifier_dropout` only apply to
    architecture='efficient' (they raise ValueError otherwise, rather than
    silently doing nothing, since 'split'/'fullchannel' use
    `medmamba_ss_trm.ClassificationHead`, whose dropout rate is hardcoded and not
    overridable without editing `medmamba_ss_trm.py`). `trm_kwargs` (a dict of
    `MedMambaSSTRMConfig` `trm_*`/`recursive` field overrides, e.g.
    `{"trm_dim": 96, "trm_deep_supervision_steps": 8}`) only applies to
    architecture='recursive'.

    `cfg_overrides` (v15) is a dict of ANY `MedMambaSSTRMConfig` fields, applied
    before the architecture-specific arguments above - for settings that are
    not per-architecture, e.g. `v15_config_overrides(architecture)`. An
    unknown field name raises instead of being silently ignored: a typo'd
    override would otherwise produce a run that quietly reproduces the exact
    defect it was meant to fix."""
    key = (architecture, modality)
    if key not in _PRESETS:
        raise ValueError(f"No preset for architecture={architecture!r}, modality={modality!r}")

    cfg_kwargs = dict(_PRESETS[key])
    if cfg_overrides:
        known = {f.name for f in dataclasses.fields(MedMambaSSTRMConfig)}
        unknown = sorted(set(cfg_overrides) - known)
        if unknown:
            raise ValueError(f"cfg_overrides contains unknown MedMambaSSTRMConfig field(s): {unknown}")
        cfg_kwargs.update(cfg_overrides)
    if drop_path_rate is not None:
        cfg_kwargs["drop_path_rate"] = drop_path_rate
    if fusion_type is not None:
        if architecture != "efficient":
            raise ValueError("--fusion_type only applies to --architecture efficient (the extended "
                              "fusion types 'se_gate'/'eca'/'none' are only registered there)")
        cfg_kwargs["fusion_type"] = fusion_type
    if trm_kwargs:
        if architecture != "recursive":
            raise ValueError("trm_kwargs only applies to --architecture recursive")
        cfg_kwargs.update(trm_kwargs)

    head_kwargs = {}
    if classifier_dropout is not None:
        # v15 R4.2 - `recursive` accepts it now too (`medmamba_ss_trm.RecursiveHead`
        # gained a dropout before its final projection). split/fullchannel
        # still cannot: `medmamba_ss_trm.ClassificationHead`'s 0.1 is hardcoded.
        if architecture not in ("efficient", "recursive"):
            raise ValueError("--classifier_dropout only applies to --architecture efficient or "
                              "recursive (medmamba_ss_trm.ClassificationHead's dropout is hardcoded at "
                              "0.1 and isn't overridable without editing medmamba_ss_trm.py)")
        head_kwargs["dropout"] = classifier_dropout

    cfg = MedMambaSSTRMConfig(**cfg_kwargs)
    model_cls = _MODEL_CLASSES[architecture]
    return model_cls(cfg, task="classification", num_classes=num_classes, **head_kwargs)
