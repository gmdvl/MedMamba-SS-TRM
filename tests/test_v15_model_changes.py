# -*- coding: utf-8 -*-
"""
test_v15_model_changes.py
===========================
MedMamba-SS-TRM v15 plan - the "Verify" clause of every model-side item, plus the
compatibility contract those items are allowed to change nothing outside of.

  C-3   every behavioural change sits behind a `MedMambaSSTRMConfig` field whose
        DEFAULT reproduces today's behaviour exactly
  R1.1  the tokenizer can see its input
  R1.2  optimizer parameter groups
  R1.3  wavelength encoding scale
  R3.2  spectral chunk size
  R3.3  gradient checkpointing that is not a silent no-op
  R4.1  the duplicate channel MLP
  R4.2  regularization in the recursive core
  R4.3  the halting head is either wired or off, never half-wired
  R4.4  API and hygiene
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import medmamba_ss_trm as G
from medmamba_ss_trm import (
    MedMambaSSTRMConfig, MedMambaSSTRM, SpectralTokenizer, RecursiveHead,
    continuous_wavelength_encoding, index_positional_encoding, normalize_wavelengths,
)
from training.config_presets import build_model, v15_config_overrides
from training.grad_checkpoint import enable_gradient_checkpointing
from training.spectral_checkpoint import enable_spectral_gradient_checkpointing
from training.optim_groups import build_param_groups, classify_parameter, summarize_param_groups

ARCHITECTURES = ("recursive", "split", "fullchannel", "efficient")


def recursive_cfg(**over):
    base = dict(recursive=True, dims=(128,), depths=(1,), d_state=8, d_ctx=64,
                patch_size=1, spectral_depth=3, trm_dim=128, trm_mixer="mlp")
    base.update(over)
    return MedMambaSSTRMConfig(**base)


def make(**over):
    torch.manual_seed(0)
    return MedMambaSSTRM(recursive_cfg(**over), num_classes=3)


# ======================================================================
# C-3 - the legacy defaults are untouched
# ======================================================================

def test_every_new_config_field_defaults_to_legacy_behaviour():
    cfg = MedMambaSSTRMConfig()
    assert cfg.spectral_token_fusion == "add"
    assert cfg.spectral_pe_gain == 1.0
    assert cfg.spectral_value_init_std == 0.02
    assert cfg.spectral_ctx_norm is False
    assert cfg.classifier_init == "shared"
    assert cfg.wavelength_encoding_scale == 1000.0
    assert cfg.spectral_chunk_size == 1024
    assert cfg.trm_checkpoint_core is True
    assert cfg.trm_mixer_channel_mlp is True
    assert cfg.trm_drop_path == 0.0
    assert cfg.trm_dropout == 0.0
    assert cfg.trm_spatial_pe_gain == 1.0
    assert cfg.trm_halt_threshold is None


def test_legacy_tokenizer_is_exactly_value_embed_plus_pe():
    """R1.1's C-3 guarantee, stated as arithmetic rather than as a hash: in
    `add` mode the tokenizer computes precisely the pre-v15 expression."""
    torch.manual_seed(0)
    tok = SpectralTokenizer(32)
    v = torch.rand(16, 8)
    expected = tok.value_embed(v.unsqueeze(-1)) + index_positional_encoding(
        8, 32, device=v.device, dtype=torch.float32).unsqueeze(0)
    assert torch.equal(tok(v), expected)


def test_legacy_tokenizer_has_no_extra_parameters():
    """An extra parameter would break `load_state_dict(strict=True)` on an
    existing checkpoint, which nothing in v15 is allowed to do for the legacy
    configuration."""
    assert {n for n, _ in SpectralTokenizer(32).named_parameters()} == {
        "value_embed.weight", "value_embed.bias"}


def test_legacy_recursive_backbone_has_no_spatial_pe_gain_parameter():
    names = {n for n, _ in make().named_parameters()}
    assert not any("spatial_pe_gain" in n for n in names)
    assert any("spatial_pe_gain" in n for n, _ in
               make(trm_spatial_pe_gain=0.1).named_parameters())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_legacy_build_is_deterministic_from_the_seed(architecture):
    """Two builds from the same seed must be identical - the property the
    bit-for-bit comparison against the pre-v15 tree relies on."""
    trm = {"trm_mixer": "mlp"} if architecture == "recursive" else None
    torch.manual_seed(1234)
    a = build_model(architecture, "hsi", 3, trm_kwargs=trm)
    torch.manual_seed(1234)
    b = build_model(architecture, "hsi", 3, trm_kwargs=trm)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb), na


# ======================================================================
# R1.1 - scale and rank
# ======================================================================

@pytest.mark.parametrize("fusion", ["add", "scaled", "concat_mlp"])
def test_tokenizer_output_shape(fusion):
    tok = SpectralTokenizer(32, fusion=fusion, pe_gain=0.1, value_init_std=0.5)
    assert tok(torch.rand(4, 11)).shape == (4, 11, 32)


def test_unknown_fusion_raises():
    with pytest.raises(ValueError):
        SpectralTokenizer(32, fusion="nonsense")
    with pytest.raises(AssertionError):
        MedMambaSSTRMConfig(spectral_token_fusion="nonsense").validate()


def test_pe_gain_is_learnable_only_when_it_is_used():
    assert not hasattr(SpectralTokenizer(32, fusion="add"), "pe_gain")
    for fusion in ("scaled", "concat_mlp"):
        gain = SpectralTokenizer(32, fusion=fusion, pe_gain=0.1).pe_gain
        assert isinstance(gain, nn.Parameter) and gain.requires_grad
        assert float(gain.detach()) == pytest.approx(0.1)


def test_concat_mlp_has_no_dead_value_embed():
    """The concat MLP consumes the raw value directly, so `value_embed` would
    be 64 parameters that receive a gradient of exactly zero forever."""
    tok = SpectralTokenizer(32, fusion="concat_mlp")
    assert not hasattr(tok, "value_embed")
    assert {n for n, _ in tok.named_parameters()} == {
        "pe_gain", "fuse_in.weight", "fuse_in.bias", "fuse_out.weight", "fuse_out.bias"}


def _relative_rank_energy(matrix: torch.Tensor) -> float:
    """Second singular value over the first. ~0 means every row is a scalar
    multiple of one fixed direction; a scale-free measure, so it is not
    confounded by the fact that the legacy value branch is 100x smaller than
    everything else in the model."""
    sv = torch.linalg.svdvals(matrix.detach().double())
    return float(sv[1] / sv[0].clamp_min(1e-30))


def test_add_mode_tokens_are_rank_one_in_the_value():
    """The rank half of the defect: with a shared `Linear(1,d)` and a ZEROED
    BIAS, every band token is `v_c * W` - a scalar multiple of one fixed
    direction - so all C bands are collinear and `tokens.mean(dim=1)` reduces
    a whole spectrum to a single scalar times W.

    The zeroed bias is not incidental, it is what makes the rank exactly 1:
    `MedMambaSSBackbone._init_weights` zeroes every `nn.Linear` bias in the
    model, so this reproduces that rather than using `nn.Linear`'s own default
    (which leaves a bias and would make the token span rank 2).
    """
    torch.manual_seed(0)
    tok = SpectralTokenizer(32, fusion="add")
    tok.apply(G.MedMambaSSBackbone._init_weights)
    assert float(tok.value_embed.bias.detach().abs().max()) == 0.0
    v = torch.rand(1, 16)
    value_part = tok(v) - index_positional_encoding(16, 32, dtype=torch.float32).unsqueeze(0)
    assert _relative_rank_energy(value_part[0]) < 1e-6


def test_the_real_model_tokenizer_has_the_zeroed_bias():
    """...and the model really is built that way."""
    tok = make().backbone.spectral_pathway.tokenizer
    assert float(tok.value_embed.bias.detach().abs().max()) == 0.0


def test_concat_mlp_tokens_are_not_rank_one():
    """The concat MLP's nonlinearity makes each band's response DIRECTION
    depend on its own positional encoding, which is the property a spectral
    tokenizer needs and the only one genuinely absent before v15."""
    torch.manual_seed(0)
    tok = SpectralTokenizer(32, fusion="concat_mlp", pe_gain=0.1, value_init_std=0.5)
    tok.reinit_value_branch()
    v = torch.rand(1, 16)
    pe = index_positional_encoding(16, 32, dtype=torch.float32).unsqueeze(0)
    assert _relative_rank_energy((tok(v) - tok.pe_gain * pe)[0]) > 0.01


def test_value_init_std_survives_the_backbones_global_init():
    """`_init_weights` re-initialises every nn.Linear at std=0.02, including
    the tokenizer's, so `spectral_value_init_std` only means anything if the
    backbone re-runs the tokenizer's own init afterwards."""
    legacy = make().backbone.spectral_pathway.tokenizer.value_embed.weight.detach().std()
    scaled = (make(spectral_token_fusion="scaled", spectral_value_init_std=0.5, spectral_pe_gain=0.1)
              .backbone.spectral_pathway.tokenizer.value_embed.weight.detach().std())
    assert float(legacy) == pytest.approx(0.02, rel=0.6)
    assert float(scaled) > 5 * float(legacy)


def test_layernorm_on_the_value_branch_would_be_a_constant():
    """Recorded in the plan because it is the obvious-looking fix and it is
    strictly worse: `LayerNorm(v*W)` for v > 0 is `LayerNorm(W)`."""
    W = torch.randn(1, 32)
    ln = nn.LayerNorm(32)
    a = ln(2.0 * W)
    b = ln(37.0 * W)
    assert torch.allclose(a, b, atol=1e-5)


# ======================================================================
# R1.1b - the context norm
# ======================================================================

def test_spectral_ctx_norm_is_identity_when_off():
    m = make()
    ctx = torch.randn(2, 3, 3, 64)
    assert torch.equal(m.backbone._normalize_ctx(ctx), ctx)


@pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-5, 1e-9])
def test_spectral_ctx_norm_makes_the_stem_input_unit_scale_at_any_scale(scale):
    """R1.1b must be SCALE-INVARIANT. An absolute eps is what broke the two
    normalizations already in the path: `nn.LayerNorm`'s 1e-5 and
    `_rms_norm_lastdim`'s 1e-6 are both added to a mean-square, and the
    context map arrives with a mean-square of ~2.5e-9."""
    m = make(spectral_ctx_norm=True)
    out = m.backbone._normalize_ctx(torch.randn(2, 3, 3, 64) * scale)
    assert float(out.pow(2).mean().sqrt()) == pytest.approx(1.0, rel=0.05)


def test_an_absolute_eps_norm_would_not_have_worked():
    """Why `_rms_norm_scale_invariant` exists rather than a reuse of
    `_rms_norm_lastdim`."""
    x = torch.randn(4, 64) * 1e-5
    assert float(G._rms_norm_lastdim(x).pow(2).mean().sqrt()) < 0.05
    assert float(G._rms_norm_scale_invariant(x).pow(2).mean().sqrt()) == pytest.approx(1.0, rel=0.05)


def test_ctx_norm_preserves_direction():
    """It must rescale, not rotate: normalizing the context cannot be allowed
    to destroy the input-dependence R1.1 just restored."""
    x = torch.randn(2, 3, 3, 64) * 1e-5
    out = G._rms_norm_scale_invariant(x)
    cos = torch.nn.functional.cosine_similarity(x.reshape(-1, 64), out.reshape(-1, 64), dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_stem_norm_is_eps_dominated_without_it():
    """The measurement behind R1.1b: the stem's input arrives with a
    per-position variance below `nn.LayerNorm`'s eps, so the LayerNorm cannot
    normalize and the embedding stays orders of magnitude below the constant
    2-D positional encoding it is then added to."""
    ln = nn.LayerNorm(128)
    tiny = torch.randn(4, 128) * 1e-5
    assert float(tiny.std(-1).mean()) < 1e-4
    assert float(ln(tiny).detach().abs().mean()) < 0.05          # nowhere near unit scale
    healthy = torch.randn(4, 128)
    assert float(ln(healthy).detach().abs().mean()) == pytest.approx(0.8, rel=0.15)


# ======================================================================
# R1.2 - optimizer parameter groups
# ======================================================================

def test_param_groups_cover_every_trainable_parameter():
    m = build_model("recursive", "hsi", 3, cfg_overrides=v15_config_overrides("recursive"),
                     trm_kwargs={"trm_mixer": "mlp"})
    groups = build_param_groups(m, 0.05)
    assert sum(len(g["params"]) for g in groups) == sum(1 for p in m.parameters() if p.requires_grad)
    assert groups[0]["weight_decay"] == 0.05 and groups[1]["weight_decay"] == 0.0


@pytest.mark.parametrize("name", [
    "backbone.spectral_pathway.tokenizer.value_embed.weight",
    "backbone.spectral_pathway.tokenizer.fuse_in.weight",
    "backbone.spectral_pathway.tokenizer.pe_gain",
    "backbone.spatial_pe_gain",
    "backbone.stem_norm.weight",
    "head.fc.bias",
])
def test_these_are_never_decayed(name):
    param = torch.nn.Parameter(torch.randn(4, 4))
    assert classify_parameter(name, param) == "no_decay"


@pytest.mark.parametrize("name", ["head.fc.weight", "backbone.stem.weight",
                                   "backbone.core.layers.0.ffn.proj_in.weight"])
def test_these_are_decayed(name):
    assert classify_parameter(name, torch.nn.Parameter(torch.randn(4, 4))) == "decay"


def test_one_dimensional_parameters_are_never_decayed():
    assert classify_parameter("anything.at.all", torch.nn.Parameter(torch.randn(8))) == "no_decay"


def test_the_value_embedding_does_not_decay_over_a_short_run():
    """R1.2's verify clause: the one weight that has to GROW must not shrink
    under weight decay. Run both optimizers on ZERO gradients, so the only
    force acting on the parameter is the decay itself."""
    m = make(spectral_token_fusion="scaled", spectral_pe_gain=0.1, spectral_value_init_std=0.5)
    grouped = copy.deepcopy(m)
    plain = copy.deepcopy(m)

    def run(model, optimizer):
        w = model.backbone.spectral_pathway.tokenizer.value_embed.weight
        before = float(w.detach().norm())
        for _ in range(200):
            optimizer.zero_grad(set_to_none=False)
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            optimizer.step()
        return before, float(w.detach().norm())

    b1, a1 = run(grouped, torch.optim.AdamW(build_param_groups(grouped, 0.05), lr=1e-3))
    b2, a2 = run(plain, torch.optim.AdamW(plain.parameters(), lr=1e-3, weight_decay=0.05))

    # Decoupled AdamW decay is `w -= lr * wd * w` per step, so 200 steps at
    # lr=1e-3, wd=0.05 shrink the norm by (1 - 5e-5)^200 = 0.990.
    expected = b2 * (1.0 - 1e-3 * 0.05) ** 200
    assert a1 == pytest.approx(b1, rel=1e-9), "grouped AdamW must not decay value_embed at all"
    assert a2 < b2, "the plain optimizer really does shrink it (this is the defect)"
    assert a2 == pytest.approx(expected, rel=1e-3)


def test_summary_is_json_serializable_and_names_the_no_decay_parameters():
    import json
    m = make(spectral_token_fusion="scaled")
    s = summarize_param_groups(m, 0.05)
    json.dumps(s)
    assert any("value_embed" in n for n in s["no_decay_parameter_names"])
    assert s["n_decay_elements"] + s["n_no_decay_elements"] == sum(
        p.numel() for p in m.parameters() if p.requires_grad)


# ======================================================================
# R1.3 - wavelength encoding scale
# ======================================================================

def test_continuous_encoding_reduces_to_the_index_encoding_for_a_uniform_sensor():
    """R1.3's verify clause. With `span = C` the two encodings are the same
    function of band position, which is the property that makes wavelength
    metadata an interpolation of the fallback rather than a different signal."""
    C, dim = 32, 32
    wl = torch.linspace(400.0, 1000.0, C)
    wl_norm = normalize_wavelengths(wl)
    continuous = continuous_wavelength_encoding(wl_norm, dim, span=float(C))
    index = index_positional_encoding(C, dim, dtype=torch.float32)
    assert torch.allclose(continuous, index, atol=1e-5)


def test_the_legacy_scale_is_pseudo_random_noise():
    """At 1000.0 with 32 bands, adjacent bands differ by ~32 radians in the
    lowest-frequency component - more than five full cycles - so the encoding
    provides no locality, which is the one thing a positional encoding is for."""
    C, dim = 32, 32
    wl_norm = normalize_wavelengths(torch.linspace(400.0, 1000.0, C))
    legacy = continuous_wavelength_encoding(wl_norm, dim, span=1000.0)
    fixed = continuous_wavelength_encoding(wl_norm, dim, span=float(C))

    def neighbour_similarity(pe):
        a = torch.nn.functional.normalize(pe[:-1], dim=-1)
        b = torch.nn.functional.normalize(pe[1:], dim=-1)
        return float((a * b).sum(-1).mean())

    assert neighbour_similarity(fixed) > 0.95
    assert neighbour_similarity(legacy) < neighbour_similarity(fixed)


def test_irregular_spacing_is_represented():
    dim = 32
    uniform = normalize_wavelengths(torch.linspace(400.0, 1000.0, 16))
    irregular = normalize_wavelengths(
        torch.tensor([400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0,
                      600.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0]))
    a = continuous_wavelength_encoding(uniform, dim, span=16.0)
    b = continuous_wavelength_encoding(irregular, dim, span=16.0)
    assert not torch.allclose(a, b, atol=1e-3)


def test_config_none_means_the_band_count():
    m = make(wavelength_encoding_scale=None)
    tok = m.backbone.spectral_pathway.tokenizer
    assert tok.wavelength_encoding_scale is None
    wl = torch.linspace(400.0, 1000.0, 24)
    got = tok._positional_encoding(torch.rand(2, 24), wl, None, torch.float32)
    want = index_positional_encoding(24, tok.d_token, dtype=torch.float32).unsqueeze(0)
    assert torch.allclose(got, want.expand_as(got), atol=1e-5)


def test_wavelengths_change_the_forward_for_an_irregular_sensor():
    """R2.3's verify clause. Note the qualifier: under R1.3 a UNIFORMLY-spaced
    sensor is *designed* to give the identical result, because `span = C` makes
    the continuous encoding reduce exactly to the index encoding. Passing
    wavelengths can only change the forward when the spacing is irregular -
    which is the entire point of passing them."""
    m = build_model("recursive", "hsi", 3, cfg_overrides=v15_config_overrides("recursive"),
                     trm_kwargs={"trm_mixer": "mlp"}).eval()
    x = torch.rand(4, 32, 11, 11)
    irregular = torch.cat([torch.linspace(400.0, 450.0, 24), torch.linspace(600.0, 1000.0, 8)])
    with torch.no_grad():
        assert not torch.allclose(m(x), m(x, wavelengths=irregular), atol=1e-6)


def test_a_uniform_sensor_is_designed_to_match_the_index_fallback():
    """The other half of R1.3, asserted so nobody 'fixes' it later: on a
    uniformly-spaced sensor, wavelength metadata and the index fallback are the
    same signal. `data/hsi_v7-80_10_10/hsi` is 32 consecutive bands of one
    spectrometer, so this is the case that dataset actually hits."""
    m = build_model("recursive", "hsi", 3, cfg_overrides=v15_config_overrides("recursive"),
                     trm_kwargs={"trm_mixer": "mlp"}).eval()
    x = torch.rand(4, 32, 11, 11)
    uniform = torch.linspace(400.5, 938.2, 32)
    with torch.no_grad():
        assert torch.allclose(m(x), m(x, wavelengths=uniform), atol=1e-5)


def test_the_legacy_scale_does_change_a_uniform_sensors_forward():
    """...and under the PRE-v15 scale of 1000.0 it did not match, because the
    encoding was aliased into noise. This is what R1.3 fixes."""
    m = build_model("recursive", "hsi", 3,
                     cfg_overrides={**v15_config_overrides("recursive"),
                                    "wavelength_encoding_scale": 1000.0},
                     trm_kwargs={"trm_mixer": "mlp"}).eval()
    x = torch.rand(4, 32, 11, 11)
    with torch.no_grad():
        assert not torch.allclose(m(x), m(x, wavelengths=torch.linspace(400.5, 938.2, 32)),
                                   atol=1e-5)


# ======================================================================
# R3.2 - spectral chunk size
# ======================================================================

@pytest.mark.parametrize("chunk", [0, 1024, 7])
def test_chunk_size_does_not_change_the_result(chunk):
    torch.manual_seed(0)
    ref = MedMambaSSTRM(recursive_cfg(spectral_chunk_size=1024), num_classes=3).eval()
    torch.manual_seed(0)
    got = MedMambaSSTRM(recursive_cfg(spectral_chunk_size=chunk), num_classes=3).eval()
    x = torch.rand(3, 32, 11, 11)
    with torch.no_grad():
        assert torch.allclose(ref(x), got(x), atol=1e-5)


def test_chunk_size_zero_makes_one_pass():
    """At bs=256, patch_size=1, 11x11 patches, N = 30,976 - so the hardcoded
    1024 meant 31 SEQUENTIAL Python iterations of the whole spectral encoder
    per forward, each containing its own loop over C timesteps."""
    calls = []
    m = make(spectral_chunk_size=0).eval()
    pathway = m.backbone.spectral_pathway
    orig = pathway._process_patch_chunk
    pathway._process_patch_chunk = lambda *a, **k: (calls.append(a[0].shape[0]), orig(*a, **k))[1]
    with torch.no_grad():
        m(torch.rand(8, 32, 11, 11))
    assert len(calls) == 1 and calls[0] == 8 * 11 * 11


def test_negative_chunk_size_is_rejected():
    with pytest.raises(AssertionError):
        recursive_cfg(spectral_chunk_size=-1).validate()


# ======================================================================
# R3.3 - checkpointing that is not a silent no-op
# ======================================================================

@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_both_checkpoint_helpers_wrap_something_for_every_architecture(architecture):
    trm = {"trm_mixer": "mlp"} if architecture == "recursive" else None
    m = build_model(architecture, "hsi", 3, trm_kwargs=trm)
    assert enable_gradient_checkpointing(m, strict=True) > 0
    assert enable_spectral_gradient_checkpointing(m, strict=True) > 0


def test_strict_raises_rather_than_silently_doing_nothing():
    for fn in (enable_gradient_checkpointing, enable_spectral_gradient_checkpointing):
        with pytest.raises(ValueError):
            fn(nn.Linear(4, 4), strict=True)
        assert fn(nn.Linear(4, 4), strict=False) == 0


def test_trm_checkpoint_core_actually_controls_the_core():
    assert make().backbone.core.checkpoint_core is True
    off = make(trm_checkpoint_core=False)
    assert off.backbone.core.checkpoint_core is False
    enable_gradient_checkpointing(off, strict=True)
    assert off.backbone.core.checkpoint_core is True


def test_checkpointing_does_not_change_the_gradient():
    x = torch.rand(2, 32, 11, 11)
    y = torch.tensor([0, 1])
    grads = []
    for flag in (True, False):
        m = make(trm_checkpoint_core=flag).train()
        loss = nn.functional.cross_entropy(m(x), y)
        loss.backward()
        grads.append(m.head.fc.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


# ======================================================================
# R4.1 / R4.2 - capacity and regularization
# ======================================================================

def test_dropping_the_mixer_channel_mlp_frees_the_expected_budget():
    """`_TokenMLPMixer` is dwconv -> GatedMLP, and `RecursiveMambaBlock` then
    applies a SECOND GatedMLP as `ffn`: at trm_dim=128 / trm_ffn_mult=2.0 that
    is 98,944 params of channel mixing per layer against 1,280 for the conv."""
    with_mlp = sum(p.numel() for p in make().parameters())
    without = sum(p.numel() for p in make(trm_mixer_channel_mlp=False).parameters())
    per_layer = with_mlp - without
    assert per_layer == 2 * 98944       # trm_core_layers = 2
    assert without < with_mlp * 0.6


def test_pure_spatial_mixer_still_runs_and_mixes_spatially():
    m = make(trm_mixer_channel_mlp=False).eval()
    assert m.backbone.core.layers[0].mixer.pw is None
    with torch.no_grad():
        assert m(torch.rand(2, 32, 11, 11)).shape == (2, 3)


def test_drop_path_and_dropout_are_off_by_default_and_wire_up_when_set():
    block = make().backbone.core.layers[0]
    assert isinstance(block.drop_path, nn.Identity)
    assert isinstance(block.ffn.drop, nn.Identity)

    reg = make(trm_drop_path=0.2, trm_dropout=0.1).backbone.core.layers
    assert isinstance(reg[-1].drop_path, G.DropPath) and reg[-1].drop_path.drop_prob > 0
    assert isinstance(reg[0].ffn.drop, nn.Dropout)


def test_drop_path_actually_perturbs_training_and_not_eval():
    m = make(trm_drop_path=0.5)
    x = torch.rand(8, 32, 11, 11)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))
    m.train()
    torch.manual_seed(0)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert not torch.allclose(a, b)


def test_classifier_dropout_is_accepted_for_recursive():
    m = build_model("recursive", "hsi", 3, classifier_dropout=0.3,
                     trm_kwargs={"trm_mixer": "mlp"})
    assert isinstance(m.head.dropout, nn.Dropout) and m.head.dropout.p == 0.3
    with pytest.raises(ValueError):
        build_model("split", "hsi", 3, classifier_dropout=0.3)


# ======================================================================
# R4.3 - the halting head
# ======================================================================

def test_halting_runs_every_segment_when_no_threshold_is_set():
    m = make(trm_act_halting=True, trm_deep_supervision_steps=4).eval()
    with torch.no_grad():
        logits, q = m.forward_deep_supervision(torch.rand(4, 32, 11, 11))
    assert len(logits) == len(q) == 4


def test_act_stops_early_once_every_sample_has_halted():
    m = make(trm_act_halting=True, trm_halt_threshold=0.5, trm_deep_supervision_steps=4).eval()
    with torch.no_grad():
        m.head.q_head.bias.fill_(5.0)              # sigmoid(q) ~ 1 for everything
        logits, _ = m.forward_deep_supervision(torch.rand(4, 32, 11, 11))
    assert len(logits) == 1


def test_act_does_not_stop_while_any_sample_is_unhalted():
    """Conservative by construction: the batch stays rectangular and no sample
    is cut short before it has halted."""
    m = make(trm_act_halting=True, trm_halt_threshold=0.5, trm_deep_supervision_steps=3).eval()
    with torch.no_grad():
        m.head.q_head.bias.fill_(-5.0)             # nothing halts
        logits, _ = m.forward_deep_supervision(torch.rand(4, 32, 11, 11))
    assert len(logits) == 3


def test_exploration_prob_forces_the_full_budget_during_training():
    m = make(trm_act_halting=True, trm_halt_threshold=0.5, trm_deep_supervision_steps=3,
             trm_halt_exploration_prob=1.0).train()
    with torch.no_grad():
        m.head.q_head.bias.fill_(5.0)
    logits, _ = m.forward_deep_supervision(torch.rand(4, 32, 11, 11))
    assert len(logits) == 3


def test_halt_threshold_is_validated():
    with pytest.raises(AssertionError):
        recursive_cfg(trm_halt_threshold=1.5).validate()
    with pytest.raises(AssertionError):
        recursive_cfg(trm_halt_exploration_prob=2.0).validate()


# ======================================================================
# R4.4 - API and hygiene
# ======================================================================

def test_head_has_two_methods_with_two_return_types():
    head = RecursiveHead(recursive_cfg(), num_classes=3)
    y = torch.randn(2, 5, 5, 128)
    logits, q = head(y)
    assert logits.shape == (2, 3) and q.shape == (2,)
    assert head.forward_features({"feature_map": y}).shape == (2, 3)
    assert torch.allclose(head.forward_features({"feature_map": y}), logits)


def test_the_dict_path_still_works_through_call():
    head = RecursiveHead(recursive_cfg(), num_classes=3)
    out = head({"feature_map": torch.randn(2, 5, 5, 128)})
    assert isinstance(out, torch.Tensor) and out.shape == (2, 3)


def test_constructing_the_model_does_not_mutate_the_callers_config():
    cfg = MedMambaSSTRMConfig(recursive=True, dims=(96, 192), depths=(2, 2), trm_dim=64,
                          trm_mixer="mlp")
    before = (cfg.recursive, tuple(cfg.dims), tuple(cfg.depths))
    MedMambaSSTRM(cfg, num_classes=3)
    assert (cfg.recursive, tuple(cfg.dims), tuple(cfg.depths)) == before


def test_forward_features_uses_the_same_recursion_as_deep_supervision():
    m = make().eval()
    x = torch.rand(2, 32, 11, 11)
    with torch.no_grad():
        from_features = m.head.forward_features(m.backbone.forward_features(x))
        from_ds = m.forward_deep_supervision(x)[0][-1]
    assert torch.allclose(from_features, from_ds, atol=1e-5)


def test_last_feature_map_is_only_written_while_training():
    m = make()
    x = torch.rand(2, 32, 11, 11)
    m.eval()
    with torch.no_grad():
        m(x)
    assert m.last_feature_map is None
    m.train()
    m(x)
    assert m.last_feature_map is not None and not m.last_feature_map.requires_grad


def test_recon_wrapper_still_works_with_the_split_head_api():
    from training.reconstruction_head import MedMambaSSTRMLatentReconWrapper
    m = make()
    wrapped = MedMambaSSTRMLatentReconWrapper(m, in_channels=32).eval()
    with torch.no_grad():
        logits, recon = wrapped(torch.rand(2, 32, 11, 11))
    assert logits.shape == (2, 3) and recon.shape == (2, 32, 11, 11)


def test_recon_wrapper_still_works_for_the_hierarchical_head():
    from training.reconstruction_head import MedMambaSSTRMLatentReconWrapper
    m = build_model("split", "hsi", 3)
    wrapped = MedMambaSSTRMLatentReconWrapper(m, in_channels=32).eval()
    with torch.no_grad():
        logits, recon = wrapped(torch.rand(2, 32, 11, 11))
    assert logits.shape == (2, 3) and recon.shape == (2, 32, 11, 11)
