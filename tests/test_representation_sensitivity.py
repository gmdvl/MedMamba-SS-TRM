# -*- coding: utf-8 -*-
"""
test_representation_sensitivity.py
====================================
MedMamba-SS-TRM v15 plan, R0.1 - the whole audit reduced to two assertions.

Gate **G1** (`stem sensitivity`)
    `S = mean_j(std_i(e_ij)) / mean_ij(|e_ij|)` over the stem embedding
    `e = stem_norm(stem(ctx_map))`. The fraction of that embedding's
    magnitude that varies with the input. Target **>= 0.05**.

Gate **G2** (`logit sensitivity`)
    `logits.std(dim=0).mean()` on the same batch. Target **>= 1e-2**.

MEASURED ON THE PRE-v15 TREE (64 real patches, freshly built model, the
numbers this test existed to fail on):

    architecture   modality   G1        G2
    recursive      hsi        0.0025    5e-07
    recursive      rgb        0.0032    5e-07
    split          hsi        0.0035    1.8e-04
    split          rgb        0.0041    2.0e-04
    fullchannel    hsi        0.0029    9e-05
    fullchannel    rgb        0.0029    8e-05
    efficient      hsi        0.0029    9e-05
    efficient      rgb        0.0029    8e-05

(The v15 plan quotes 0.0018 HSI / 0.0029 RGB for G1 from the two completed
runs; the small difference is the batch and the seed.) Those models were
constant functions of their input: `SpectralTokenizer` added a value
embedding of magnitude ~0.006 to a per-dataset-CONSTANT positional encoding
of magnitude ~0.55.

AND ON THIS TREE with `v15_config_overrides(...)`:

    recursive      hsi        0.058     0.024
    recursive      rgb        0.058     0.021
    split          hsi        0.064     0.005
    split          rgb        0.072     0.009
    fullchannel    hsi        0.062     0.010
    fullchannel    rgb        0.094     0.016

`recursive` - the architecture v15 trains - clears both gates on both
modalities. The three hierarchical architectures clear G1 everywhere but sit
between 5e-3 and 1.6e-2 on G2, a 25-100x improvement over their pre-v15
values that does not uniformly reach 1e-2. Their remaining attenuation is in
the 3/4-stage backbone (`layerscale_init=1e-4` makes every residual branch
near-identity at init), not in the tokenizer, and the v15 plan defers them
explicitly: section 12, "Do not retrain --architecture split/fullchannel/
efficient yet ... They come back in R7.3". They are therefore held to G1 in
full and to a 1e-3 G2 floor here, and the strict G2 gate is asserted for
`recursive`.

Data: real patches from `data/` when a prepared dataset is present,
otherwise a seeded synthetic fallback. The fallback is NOT i.i.d. uniform
noise: an 11x11 patch of independent pixels is spatially uncorrelated, so the
head's `mean(dim=(1,2))` over 121 positions attenuates the batch variance by
~sqrt(121) and G2 comes out ~10x too low for ANY model, correct or not.
`synthetic_patches` therefore draws a smooth per-sample spectrum plus a smooth
spatial field, which is what a real tissue patch looks like; it reproduces the
real-data numbers above to within ~15%.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from training.config_presets import build_model, v15_config_overrides
from training.numerical_stability import (
    G1_MIN_STEM_SENSITIVITY, G2_MIN_LOGIT_STD, run_sensitivity_check, stem_sensitivity,
    extract_stem_embedding,
)

BATCH = 64
PATCH_HW = 11
# (architecture, modality, in_channels, num_classes)
PRESETS = [
    ("recursive", "hsi", 32, 3),
    ("recursive", "rgb", 3, 6),
    ("split", "hsi", 32, 3),
    ("split", "rgb", 3, 6),
    ("fullchannel", "hsi", 32, 3),
    ("fullchannel", "rgb", 3, 6),
    ("efficient", "hsi", 32, 3),
    ("efficient", "rgb", 3, 6),
]
# See the module docstring: the hierarchical architectures are held to G1 in
# full and to this G2 floor until R7.3 retrains them.
DEFERRED_G2_ARCHITECTURES = {"split", "fullchannel", "efficient"}
DEFERRED_G2_MIN = 1e-3

_REAL_DATA = {
    "hsi": ["data/hsi_v7-80_10_10/hsi/X_val.npy", "data/hsi_v7/hsi/X_val.npy"],
    "rgb": ["data/pad_v6/X_val.npy"],
}


# ----------------------------------------------------------------------
# batches
# ----------------------------------------------------------------------

def synthetic_patches(n: int, channels: int, hw: int = PATCH_HW, seed: int = 0) -> torch.Tensor:
    """Seeded stand-in for real patches: a smooth per-sample spectrum (a
    normalized random walk over the band axis) plus a smooth per-sample
    spatial field plus a little per-band texture, all in [0,1]. See the module
    docstring for why i.i.d. uniform noise is not usable here."""
    g = torch.Generator().manual_seed(seed)
    spec = torch.cumsum(torch.randn(n, channels, generator=g), dim=1)
    spec = spec - spec.min(dim=1, keepdim=True).values
    spec = spec / spec.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    spec = spec.view(n, channels, 1, 1)
    field = F.interpolate(torch.rand(n, 1, 3, 3, generator=g), size=(hw, hw),
                          mode="bilinear", align_corners=False)
    texture = torch.rand(n, channels, hw, hw, generator=g)
    return (0.6 * spec + 0.3 * field + 0.1 * texture).clamp(0.0, 1.0)


def real_patches(modality: str, n: int = BATCH, seed: int = 0, offset: int = 0):
    """`n` per-sample-min-max-normalized patches from a prepared dataset (the
    same normalization `train_example_v6.NpyDataset` applies by default), or
    `None` if no dataset for `modality` is present on this machine."""
    # v17 S12 - `.parent.parent` since this file moved into `tests/`; `_REAL_DATA` holds
    # paths relative to the REPOSITORY root (`data/...`). Getting this wrong is silent:
    # every path simply misses and the test skips instead of failing.
    root = Path(__file__).resolve().parent.parent
    for rel in _REAL_DATA.get(modality, []):
        path = root / rel
        if not path.is_file():
            continue
        X = np.load(path, mmap_mode="r")
        if len(X) < n * (offset + 1):
            continue
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), size=n * (offset + 1), replace=False))[offset * n:]
        a = np.nan_to_num(np.asarray(X[idx], dtype=np.float32))
        flat = a.reshape(len(a), -1)
        lo = flat.min(axis=1)[:, None, None, None]
        hi = flat.max(axis=1)[:, None, None, None]
        a = (a - lo) / np.maximum(hi - lo, 1e-6)
        return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()
    return None


def batch_for(modality: str, channels: int, seed: int = 0, offset: int = 0) -> torch.Tensor:
    real = real_patches(modality, seed=seed, offset=offset)
    if real is not None and real.shape[1] == channels:
        return real
    return synthetic_patches(BATCH, channels, seed=seed + 1000 * offset)


def v15_model(architecture: str, modality: str, num_classes: int):
    torch.manual_seed(0)
    return build_model(architecture, modality, num_classes,
                        cfg_overrides=v15_config_overrides(architecture),
                        trm_kwargs=({"trm_mixer": "mlp"} if architecture == "recursive" else None))


def legacy_model(architecture: str, modality: str, num_classes: int):
    torch.manual_seed(0)
    return build_model(architecture, modality, num_classes,
                        trm_kwargs=({"trm_mixer": "mlp"} if architecture == "recursive" else None))


# ----------------------------------------------------------------------
# G1 / G2
# ----------------------------------------------------------------------

@pytest.mark.parametrize("architecture,modality,channels,num_classes", PRESETS)
def test_g1_stem_sensitivity(architecture, modality, channels, num_classes):
    """Gate G1: the stem embedding must vary with the input by at least 5%
    of its own magnitude."""
    model = v15_model(architecture, modality, num_classes)
    x = batch_for(modality, channels)
    report = run_sensitivity_check(model, x)
    s = report["stem_sensitivity"]
    assert s >= G1_MIN_STEM_SENSITIVITY, (
        f"G1 FAILED for {architecture}/{modality}: stem sensitivity {s:.5f} < "
        f"{G1_MIN_STEM_SENSITIVITY}. The embedding entering the backbone is dominated by "
        f"input-independent constants; see medmamba_ss_trm.SpectralTokenizer.")


@pytest.mark.parametrize("architecture,modality,channels,num_classes", PRESETS)
def test_g2_logit_sensitivity(architecture, modality, channels, num_classes):
    """Gate G2: the logits must actually vary across a batch."""
    model = v15_model(architecture, modality, num_classes)
    x = batch_for(modality, channels)
    report = run_sensitivity_check(model, x)
    threshold = (DEFERRED_G2_MIN if architecture in DEFERRED_G2_ARCHITECTURES
                 else G2_MIN_LOGIT_STD)
    std = report["logit_std"]
    assert std >= threshold, (
        f"G2 FAILED for {architecture}/{modality}: logit std {std:.3e} < {threshold:.3e}. "
        f"The model is close to a constant function of its input.")


@pytest.mark.parametrize("architecture,modality,channels,num_classes", PRESETS)
def test_disjoint_batches_produce_different_mean_logits(architecture, modality, channels, num_classes):
    """R0.1's second test: two DISJOINT batches must produce different mean
    logits. Per-sample spread alone can come from noise; this is the check
    that the model sees batch CONTENT."""
    model = v15_model(architecture, modality, num_classes).eval()
    xa = batch_for(modality, channels, seed=0, offset=0)
    xb = batch_for(modality, channels, seed=0, offset=1)
    assert not torch.allclose(xa, xb), "the two probe batches are identical - test is vacuous"
    with torch.no_grad():
        la, lb = model(xa), model(xb)
    delta = float((la.mean(dim=0) - lb.mean(dim=0)).abs().max())
    assert delta > 1e-4, (
        f"{architecture}/{modality}: two disjoint batches produced mean logits differing by "
        f"only {delta:.3e} - the model is not reading its input.")


# ----------------------------------------------------------------------
# The legacy path must still exhibit the defect (contract C-3)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("architecture,modality,channels,num_classes", PRESETS)
def test_legacy_defaults_still_reproduce_the_defect(architecture, modality, channels, num_classes):
    """Contract C-3: `train_example_v14.py` must keep building the model it
    built before, defect included, so the two completed runs stay reproducible
    and citable. If this test ever starts FAILING, a v15 default has leaked
    into the legacy path."""
    model = legacy_model(architecture, modality, num_classes)
    x = batch_for(modality, channels)
    report = run_sensitivity_check(model, x)
    assert not report["passed"]
    assert report["stem_sensitivity"] < 0.01, (
        f"{architecture}/{modality}: legacy stem sensitivity is {report['stem_sensitivity']:.5f}, "
        f"but the pre-v15 tree measured ~0.003. A v15 default has leaked into the legacy path.")


def test_v15_is_a_large_improvement_over_legacy():
    """The headline number, asserted rather than asserted-about: the v15
    settings raise the recursive model's stem sensitivity by more than an
    order of magnitude on both modalities."""
    for modality, channels, num_classes in (("hsi", 32, 3), ("rgb", 3, 6)):
        x = batch_for(modality, channels)
        before = run_sensitivity_check(legacy_model("recursive", modality, num_classes), x)
        after = run_sensitivity_check(v15_model("recursive", modality, num_classes), x)
        ratio = after["stem_sensitivity"] / max(before["stem_sensitivity"], 1e-12)
        assert ratio > 10.0, (
            f"{modality}: v15 raised stem sensitivity only {ratio:.1f}x "
            f"({before['stem_sensitivity']:.5f} -> {after['stem_sensitivity']:.5f})")


# ----------------------------------------------------------------------
# The measurement itself
# ----------------------------------------------------------------------

def test_stem_sensitivity_of_a_constant_embedding_is_zero():
    e = torch.ones(16, 4, 4, 8) * 3.0
    assert stem_sensitivity(e) == pytest.approx(0.0)


def test_stem_sensitivity_of_pure_signal_is_order_one():
    g = torch.Generator().manual_seed(0)
    e = torch.randn(256, 64, generator=g)
    assert stem_sensitivity(e) == pytest.approx(1.0 / 0.7979, rel=0.05)


def test_stem_sensitivity_needs_a_batch():
    with pytest.raises(ValueError):
        stem_sensitivity(torch.randn(1, 8))


def test_run_sensitivity_check_flags_a_constant_module():
    """R0.2's unit test: a deliberately constant `nn.Module` must fail."""

    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 3)

    report = run_sensitivity_check(Constant(), torch.rand(8, 3, 11, 11))
    assert report["passed"] is False
    assert "logit std" in report["reason"]
    # No stem to measure on a bare module: the check degrades, it does not crash.
    assert "stem_sensitivity" not in report


def test_run_sensitivity_check_does_not_change_training_mode():
    model = v15_model("recursive", "hsi", 3)
    model.train()
    run_sensitivity_check(model, batch_for("hsi", 32))
    assert model.training is True
    model.eval()
    run_sensitivity_check(model, batch_for("hsi", 32))
    assert model.training is False


def test_extract_stem_embedding_returns_none_for_an_unknown_model():
    assert extract_stem_embedding(torch.nn.Linear(4, 4), torch.rand(2, 4)) is None


def test_extract_stem_embedding_unwraps_a_recon_wrapper():
    from training.reconstruction_head import MedMambaSSTRMLatentReconWrapper

    model = v15_model("recursive", "hsi", 3)
    wrapped = MedMambaSSTRMLatentReconWrapper(model, in_channels=32)
    e = extract_stem_embedding(wrapped, batch_for("hsi", 32))
    assert e is not None and e.shape[0] == BATCH


@pytest.mark.skipif(not os.path.isdir("data"), reason="no prepared dataset on this machine")
def test_real_patches_are_used_when_available():
    """The gates above are only as meaningful as the batch they run on; make
    it visible when this machine is falling back to synthetic patches."""
    found = [m for m in ("hsi", "rgb") if real_patches(m) is not None]
    assert found, ("no prepared dataset found under data/ - the G1/G2 gates ran on synthetic "
                    "patches. That is supported, but re-run this on a machine with data/ before "
                    "citing the numbers.")
