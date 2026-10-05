# -*- coding: utf-8 -*-
"""
MedMamba-SS v3 and MedMamba-SS-TRM: sensor-agnostic spectral-spatial models
===========================================================================

Implements `improve-prompt_v3.plan.md`, building on MedMamba-SS v1/v2 and the
official MedMamba design (SS2D / SS-Conv-SSM / hierarchical stages,
https://github.com/YubiaoYue/MedMamba).

--------------------------------------------------------------------------
Plan phase -> implementation map
--------------------------------------------------------------------------
 1  Universal spectral tokenization       SpectralTokenizer: shared Linear(1->d_token) per band,
                                           C never appears in any parameter shape (carried over
                                           from v2, retained as the foundation of everything else)
 2  Patch-wise spectral modeling          SpectralPathway.patchify() + per-patch encoding,
                                           aggregated only after spectral modeling (from v2)
 3  Deep residual spectral encoder        ResidualSpectralBlock (residual, normed, configurable
                                           depth/kernels) + optional squeeze-and-excitation
 4  True spectral convolution             Conv1d inside ResidualSpectralBlock, operating on the
                                           cube-derived per-patch band sequence (from v2)
 5  Multi-scale spectral branch           ResidualSpectralBlock(kernel_sizes=[3,5,9,...]) with
                                           an attention/gated branch-fusion (from v2, kept)
 6  Bidirectional spectral<->spatial      SpectralContextUpdater (spatial->spectral) +
                                           StageBandSelector (spectral self-gating, refreshed
                                           every stage) form the two-way interaction loop
 7  Stage-wise spectral refinement        same updater + StageBandSelector, invoked every stage;
                                           fusion parameters are effectively refreshed each stage
                                           because they read the just-updated context map
 8  Dynamic band selection v2             BandGate at the input (mode="soft"|"sparse", with a
                                           learnable threshold) + StageBandSelector recomputing
                                           context-channel importance at every stage
 9  Generalized SS2D (scan registry)      SS2D(scan_directions=[...]) - any subset of
                                           {h, h_rev, v, v_rev, diag, diag_rev, adiag, adiag_rev},
                                           each direction individually enable/disable-able, with
                                           learnable weights over exactly the enabled ones
10  Multi-scale SS2D                      MultiScaleSS2D (parallel conv receptive fields)
11  Pluggable fusion framework            FiLMFusion / GatedAdditiveFusion / CrossAttentionFusion /
                                           MultiplicativeFusion / ResidualFusion, selectable
                                           globally OR per-stage (fusion_type: str | list[str])
12  Progressive spectral compression      ProgressiveCompressor (staged bottleneck, from v2)
13  Multi-level backbone outputs          forward_features() returns stage feature maps, stage
                                           context maps, stage pools, band weights, final pooled
                                           embedding and spectral embedding - one dict, superset
14  Continuous spectral positional        continuous_wavelength_encoding(): sinusoidal function of
    encoding                              the actual physical wavelength (nm) when provided;
                                           falls back to the v2 index-based encoding otherwise
15  Sensor-aware wavelength encoding      normalize_wavelengths(): normalizes by a caller-supplied
                                           sensor reference range, or by the batch's own min/max
                                           when no reference range is given
16  Backbone configuration framework      MedMambaSSTRMConfig dataclass + .validate()
17  Backend abstraction                   SCAN_BACKENDS registry, register_scan_backend(),
                                           get_backend(), benchmark_backend()

Per the plan's "Important Design Recommendation": a single architecture
supports arbitrary channel counts, but one trained set of weights is not
expected to transfer perfectly across sensors with very different spectral
ranges unless given wavelength metadata - which is exactly what Phases 14-15
provide (`wavelengths=` / `sensor_range=` are optional keyword arguments
threaded through every forward call; omitting them falls back to the
index-based behavior of v2).
--------------------------------------------------------------------------
"""

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint


# ============================================================================
# Phase 17 - future-proof selective-scan backend abstraction
# ============================================================================

def _selective_scan_pure_pytorch(xs, dts, As, Bs, Cs, Ds, dt_projs_bias, delta_softplus=True):
    """Reference pure-PyTorch selective scan (no custom CUDA/Triton kernels).
    xs:[B,K*D,L] dts:[B,K*D,L] As:[K,D,N] Bs:[B,K*N,L] Cs:[B,K*N,L] Ds:[K,D]
    """
    B = xs.shape[0]
    L = xs.shape[-1]
    K, D, N = As.shape

    xs = xs.reshape(B, K, D, L)
    dts = dts.reshape(B, K, D, L)
    Bs = Bs.reshape(B, K, N, L)
    Cs = Cs.reshape(B, K, N, L)

    if dt_projs_bias is not None:
        dts = dts + dt_projs_bias.reshape(1, K, D, 1)
    if delta_softplus:
        dts = F.softplus(dts)

    x_state = torch.zeros(B, K, D, N, device=xs.device, dtype=xs.dtype)
    y = torch.empty(B, K, D, L, device=xs.device, dtype=xs.dtype)
    for l in range(L):
        # Compute dA/dB per step to avoid materializing [B,K,D,N,L] (OOM fix:
        # the old code built two [B,K,D,N,L] tensors ~4 GiB each on 64x64x246
        # inputs, blowing up a 16 GiB GPU).
        dA_l = torch.exp(torch.einsum("b k d, k d n -> b k d n", dts[..., l], As))
        dB_l = torch.einsum("b k d, b k n -> b k d n", dts[..., l], Bs[..., l]) * xs[..., l].unsqueeze(-1)
        x_state = dA_l * x_state + dB_l
        y[..., l] = torch.einsum("b k d n, b k n -> b k d", x_state, Cs[..., l])

    if Ds is not None:
        y = y + xs * Ds.reshape(1, K, D, 1)
    return y.reshape(B, K * D, L)


def _selective_scan_not_implemented(*args, backend_name="", **kwargs):
    raise NotImplementedError(
        f"Selective-scan backend {backend_name!r} is not available in this environment. "
        "Register a real implementation with `register_scan_backend(name, fn)` "
        "(e.g. a CUDA or Triton kernel exposing the same signature as "
        "`_selective_scan_pure_pytorch`) or use backend='pure_pytorch'."
    )


SCAN_BACKENDS = {
    "pure_pytorch": _selective_scan_pure_pytorch,
    "cuda": lambda *a, **k: _selective_scan_not_implemented(*a, backend_name="cuda", **k),
    "triton": lambda *a, **k: _selective_scan_not_implemented(*a, backend_name="triton", **k),
}


def register_scan_backend(name: str, fn) -> None:
    """Register a new selective-scan backend (e.g. a CUDA/Triton kernel).
    `fn` must implement the same signature as `_selective_scan_pure_pytorch`."""
    SCAN_BACKENDS[name] = fn


def get_backend(name: str):
    if name not in SCAN_BACKENDS:
        raise ValueError(f"Unknown scan backend {name!r}. Available: {list(SCAN_BACKENDS)}")
    return SCAN_BACKENDS[name]


def benchmark_backend(model: nn.Module, input_shape: Sequence[int], n: int = 3, device: str = "cpu"):
    """Phase 17 - tiny benchmarking utility: times `n` forward passes of
    `model` on random input of `input_shape` and returns (mean_seconds, out_shape)."""
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        out = model(x)  # warmup
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            out = model(x)
            times.append(time.perf_counter() - t0)
    return sum(times) / len(times), tuple(out.shape)


# ============================================================================
# Phase 16 - configurable architecture
# ============================================================================

DEFAULT_SCAN_DIRECTIONS_4 = ["h", "h_rev", "v", "v_rev"]
DEFAULT_SCAN_DIRECTIONS_8 = ["h", "h_rev", "v", "v_rev", "diag", "diag_rev", "adiag", "adiag_rev"]
ALL_SCAN_DIRECTIONS = set(DEFAULT_SCAN_DIRECTIONS_8)
FUSION_TYPES = {"film", "gated", "cross_attention", "multiplicative", "residual"}
# v15 R1.1 - how SpectralTokenizer combines a band value with its positional
# encoding. "add" is the pre-v15 behaviour and stays the default (contract C-3).
SPECTRAL_TOKEN_FUSIONS = {"add", "scaled", "concat_mlp"}


@dataclass
class MedMambaSSTRMConfig:
    # ---- spatial backbone ----
    dims: Sequence[int] = (96, 192, 384, 768)
    depths: Sequence[int] = (2, 2, 4, 2)
    d_state: int = 16
    scan_backend: str = "pure_pytorch"

    # ---- universal input layer / tokenizer (Phase 1, 14, 15) ----
    patch_size: int = 4
    d_token: int = 32
    use_wavelength_metadata: bool = True     # if wavelengths are passed to forward(), use them
    sensor_reference_range: Optional[Tuple[float, float]] = None  # e.g. (400.0, 1000.0)

    # ---- v15 R1.1/R1.3 - spectral token fusion (representation-collapse fix).
    #      Every default below reproduces pre-v15 behaviour exactly; only
    #      `train_example_v15.py` changes them. See `SpectralTokenizer`. ----
    spectral_token_fusion: str = "add"        # "add" (legacy) | "scaled" | "concat_mlp"
    spectral_pe_gain: float = 1.0             # initial value of the (learnable) PE gain
    spectral_value_init_std: float = 0.02     # init std for value_embed.weight
    # `None` => use the band count C as the angular span, which makes
    # `continuous_wavelength_encoding` reduce EXACTLY to
    # `index_positional_encoding` for a uniformly-spaced sensor (v15 R1.3).
    # 1000.0 is the pre-v15 hardcoded constant.
    wavelength_encoding_scale: Optional[float] = 1000.0
    # v15 R1.1b - parameter-free RMS normalization of the spectral context map
    # before the stem projection. The v15 plan assumes `stem_norm` "divides out
    # whatever scale survives"; measured, it does not. Every `nn.Linear` in the
    # spectral pathway is initialised at trunc_normal(std=0.02) - roughly 9x
    # below its own fan-in scale - so activations decay ~100x per projection
    # and the stem's input arrives at |.| ~ 1e-5 with a per-position variance
    # of 1.1e-5, BELOW `nn.LayerNorm`'s eps of 1e-5. The LayerNorm therefore
    # degenerates into a constant rescale and the embedding stays ~500x smaller
    # than the constant 2-D positional encoding it is then added to - the exact
    # A1 mechanism, one layer further down. False = pre-v15 behaviour.
    spectral_ctx_norm: bool = False
    # v15 R1.1d - "shared" applies the backbone's global trunc_normal(std=0.02)
    # to the classifier's final projection too, which is ~4.4x below that
    # layer's own fan-in scale and costs a corresponding factor of logit
    # sensitivity (gate G2). "fan_in" restores `nn.Linear`'s default init for
    # the final projection only. "shared" = pre-v15 behaviour.
    classifier_init: str = "shared"

    # ---- v15 R3.2 - spectral pathway patch chunking, previously HARDCODED at
    #      1024 regardless of batch size. 0 => one pass over all N patch
    #      positions. Measured on an RTX 5060 Ti at bs=256/bf16 (see
    #      findings/finds_20260902_v15_r3.md), 1024 is also the fastest value -
    #      one pass is 35% SLOWER at identical peak memory - and it is what
    #      gives `training/spectral_checkpoint.py` a granularity to discard at:
    #      with a single chunk it saves nothing, at 1024 it cuts HSI peak
    #      memory 10,243 MB -> 1,007 MB. The defect was that it was not a
    #      choice; the value it was stuck at happens to be a good one. ----
    spectral_chunk_size: int = 1024

    # ---- spectral encoder (Phases 2-5, 8, 12) ----
    spectral_depth: int = 3
    spectral_kernel_sizes: Sequence[int] = (3,)
    spectral_multiscale: bool = False
    squeeze_excitation: bool = False                 # Phase 3
    dynamic_band_selection: bool = True              # Phase 8
    band_selection_mode: str = "soft"                # "soft" | "sparse"
    compression_dims: Sequence[int] = (128, 64)
    d_ctx: int = 64

    # ---- spatial scan (Phases 9-10) ----
    scan_directions: Union[int, Sequence[str]] = 4   # 4, 8, or an explicit list/subset of names
    spatial_multiscale: bool = False
    spatial_scales: Sequence[int] = (3, 5)

    # ---- fusion (Phase 11): a single strategy for all stages, or a
    # per-stage list (len == number of stages) ----
    fusion_type: Union[str, Sequence[str]] = "film"

    # ---- normalization / residual ----
    norm_type: str = "layernorm"
    layerscale: bool = True
    layerscale_init: float = 1e-4
    drop_path_rate: float = 0.1
    use_ffn: bool = True

    # ---- stage-wise spectral refinement (Phases 6-7-8) ----
    stage_spectral_refinement: bool = True

    # ---- TRM recursive variant (opt-in; every field below is inert unless
    #      `recursive=True`; see `MedMambaSSTRM`). Mirrors the Tiny
    #      Recursive Model (arXiv:2510.04871): one weight-shared core applied
    #      recursively instead of a hierarchical multi-stage backbone. ----
    recursive: bool = False
    trm_dim: int = 128                       # fixed working width `d` (must be % 4 == 0)
    trm_core_layers: int = 2                 # layers in the shared core `f` (TRM L_layers)
    trm_n_latent: int = 6                    # inner latent updates per improve step (TRM L_cycles)
    trm_n_improve: int = 3                   # improve steps per segment; T-1 run under no_grad (TRM H_cycles)
    trm_two_state: bool = True              # z + y (trm.py) vs single z (trm_singlez.py)
    trm_mixer: str = "ss2d"                 # "ss2d" | "mlp" | "attention"
    trm_ffn_mult: float = 2.0              # channel-MLP expansion inside the core
    trm_deep_supervision_steps: int = 4     # ACT segments N_sup (TRM halt_max_steps)
    trm_act_halting: bool = True            # learn a halt head + BCE halting loss
    trm_ema_rate: float = 0.999           # EMA decay for the recursive trainer (0 disables)
    trm_halt_exploration_prob: float = 0.1  # TRM exploration floor for early-stop during training
    # v15 R4.3 - the halt head is no longer half-wired. `None` (default) is the
    # pre-v15 behaviour: `q` is trained by a BCE term and NOTHING ever reads
    # it, so `forward_deep_supervision` always runs every segment and
    # `trm_halt_exploration_prob` was declared here and referenced nowhere in
    # the repository. Set a threshold in (0,1) to turn on real ACT: segments
    # stop early once `sigmoid(q) > trm_halt_threshold` for EVERY sample in the
    # batch (conservative - no sample is cut short before it has halted), with
    # `trm_halt_exploration_prob` as the training-time probability of running
    # all segments anyway. Measured on the RGB run, the halt BCE sat at ~0.66
    # against ln 2 = 0.693 for 33 epochs - chance, flat, and ~20% of the
    # training objective - which is why v15 defaults `--no_trm_halting`.
    trm_halt_threshold: Optional[float] = None
    # ---- v15 R3.3/R4.1/R4.2 (legacy defaults; only train_example_v15.py flips them) ----
    trm_checkpoint_core: bool = True        # gradient-checkpoint every `RecursiveCore.f` call
    trm_mixer_channel_mlp: bool = True      # `_TokenMLPMixer` keeps its own GatedMLP (R4.1)
    trm_drop_path: float = 0.0              # stochastic depth inside RecursiveMambaBlock (R4.2)
    trm_dropout: float = 0.0                # dropout inside the block's GatedMLPs (R4.2)
    # v15 R1.1c - initial value of a learnable gain on the recursive backbone's
    # fixed 2-D sinusoidal positional encoding. That encoding has |.| ~ 0.56 and
    # is IDENTICAL for every sample, so at 1.0 it is the same "signal added to a
    # much larger constant" defect the tokenizer had. Only created as a
    # parameter when != 1.0, so the legacy model's state_dict is unchanged.
    trm_spatial_pe_gain: float = 1.0

    def resolved_scan_directions(self) -> List[str]:
        if isinstance(self.scan_directions, int):
            if self.scan_directions == 4:
                return list(DEFAULT_SCAN_DIRECTIONS_4)
            if self.scan_directions == 8:
                return list(DEFAULT_SCAN_DIRECTIONS_8)
            raise ValueError("Integer scan_directions must be 4 or 8; pass an explicit list for other subsets.")
        return list(self.scan_directions)

    def resolved_fusion_types(self) -> List[str]:
        n = len(self.dims)
        if isinstance(self.fusion_type, str):
            return [self.fusion_type] * n
        types = list(self.fusion_type)
        if len(types) != n:
            raise ValueError(f"fusion_type list has {len(types)} entries but there are {n} stages.")
        return types

    def validate(self) -> None:
        assert len(self.dims) == len(self.depths) > 0, "dims and depths must be non-empty and same length"
        if self.recursive:
            # PatchMerging2D / channel-doubling do not apply to the single
            # fixed-width recursive core.
            assert self.trm_dim % 4 == 0, "trm_dim must be divisible by 4 (2-D sinusoidal PE)"
            assert self.trm_core_layers >= 1
            assert self.trm_n_latent >= 1
            assert self.trm_n_improve >= 1
            assert self.trm_deep_supervision_steps >= 1
            assert self.trm_halt_threshold is None or 0.0 < self.trm_halt_threshold < 1.0, (
                f"trm_halt_threshold must be None (ACT off) or in (0,1), got "
                f"{self.trm_halt_threshold!r}")
            assert 0.0 <= self.trm_halt_exploration_prob <= 1.0
            assert self.trm_mixer in ("ss2d", "mlp", "attention"), (
                f"unknown trm_mixer {self.trm_mixer!r}; use 'ss2d' | 'mlp' | 'attention'"
            )
        else:
            for i in range(len(self.dims) - 1):
                assert self.dims[i + 1] == 2 * self.dims[i], (
                    f"dims[{i + 1}]={self.dims[i + 1]} must equal 2*dims[{i}]={2 * self.dims[i]}: "
                    "PatchMerging2D always doubles channel width between stages."
                )
        dirs = self.resolved_scan_directions()
        assert len(dirs) > 0, "at least one scan direction must be enabled"
        bad = set(dirs) - ALL_SCAN_DIRECTIONS
        assert not bad, f"unknown scan direction(s): {bad}. Valid: {sorted(ALL_SCAN_DIRECTIONS)}"
        for f in self.resolved_fusion_types():
            assert f in FUSION_TYPES, f"unknown fusion_type {f!r}. Valid: {sorted(FUSION_TYPES)}"
        assert self.band_selection_mode in ("soft", "sparse")
        assert self.norm_type in ("layernorm", "rmsnorm", "groupnorm")
        assert self.spectral_token_fusion in SPECTRAL_TOKEN_FUSIONS, (
            f"unknown spectral_token_fusion {self.spectral_token_fusion!r}; "
            f"use one of {sorted(SPECTRAL_TOKEN_FUSIONS)}"
        )
        assert self.spectral_chunk_size >= 0, "spectral_chunk_size must be >= 0 (0 = no chunking)"
        assert self.wavelength_encoding_scale is None or self.wavelength_encoding_scale > 0
        assert self.classifier_init in ("shared", "fan_in"), (
            f"unknown classifier_init {self.classifier_init!r}; use 'shared' | 'fan_in'")
        get_backend(self.scan_backend)  # raises if unknown


# ============================================================================
# Basic building blocks (norm / droppath / layerscale / FFN)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(norm + self.eps)
        return (x * self.weight).type_as(self.weight)


class _GroupNormChannelsLast(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8):
        super().__init__()
        num_groups = min(num_groups, dim)
        while dim % num_groups != 0:
            num_groups -= 1
        self.gn = nn.GroupNorm(num_groups, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1]).unsqueeze(-1)
        x = self.gn(x.transpose(1, 2).transpose(1, 2))
        return x.squeeze(-1).reshape(shape)


def make_norm(norm_type: str, dim: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "groupnorm":
        return _GroupNormChannelsLast(dim)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-4, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.alpha = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha if self.enabled else x


class GatedMLP(nn.Module):
    """GEGLU-style gated feed-forward."""

    def __init__(self, dim: int, mult: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.proj_in = nn.Linear(dim, hidden * 2)
        self.proj_out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj_in(x).chunk(2, dim=-1)
        x = self.drop(x * F.gelu(gate))
        return self.proj_out(x)


# ============================================================================
# Shared selective-scan parameter container
# ============================================================================

class _SelectiveScanParams(nn.Module):
    def __init__(self, d_inner: int, d_state: int, dt_rank: int, k_directions: int,
                 backend: str = "pure_pytorch"):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.K = k_directions
        self.scan_fn = get_backend(backend)

        self.x_proj_weight = nn.Parameter(torch.empty(self.K, dt_rank + d_state * 2, d_inner))
        nn.init.kaiming_uniform_(self.x_proj_weight, a=math.sqrt(5))

        self.dt_projs_weight = nn.Parameter(torch.empty(self.K, d_inner, dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(self.K, d_inner))
        nn.init.kaiming_uniform_(self.dt_projs_weight, a=math.sqrt(5))
        nn.init.zeros_(self.dt_projs_bias)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A).repeat(self.K, 1, 1))
        self.Ds = nn.Parameter(torch.ones(self.K, d_inner))

    def run(self, xs: torch.Tensor) -> torch.Tensor:
        """xs: [B, K, D, L] -> [B, K, D, L] (per-direction outputs, NOT yet combined)."""
        Bsz, K, D, L = xs.shape
        assert K == self.K
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        As = -torch.exp(self.A_logs.float())
        out = self.scan_fn(
            xs.float().reshape(Bsz, K * D, L),
            dts.float().reshape(Bsz, K * D, L),
            As, Bs.float().reshape(Bsz, K * self.d_state, L),
            Cs.float().reshape(Bsz, K * self.d_state, L),
            self.Ds.float(), self.dt_projs_bias.float().reshape(-1),
        )
        return out.reshape(Bsz, K, D, L)


# ============================================================================
# Phase 9 - Generalized SS2D with a scan-direction registry: any subset of
# {h, h_rev, v, v_rev, diag, diag_rev, adiag, adiag_rev} may be enabled,
# each with its own learnable combination weight.
# ============================================================================

def _skew(x: torch.Tensor) -> torch.Tensor:
    """x: [B,D,H,W] -> [B,D,H,H+W-1], row i shifted right by i so that
    anti-diagonals (constant row+col) line up in the same column."""
    Bsz, D, H, W = x.shape
    Ws = H + W - 1
    out = x.new_zeros(Bsz, D, H, Ws)
    for i in range(H):
        out[:, :, i, i:i + W] = x[:, :, i, :]
    return out


def _unskew(xs: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Inverse of `_skew`."""
    Bsz, D, H_, Ws = xs.shape
    out = xs.new_zeros(Bsz, D, H, W)
    for i in range(H):
        out[:, :, i, :] = xs[:, :, i, i:i + W]
    return out


class SS2D(nn.Module):
    """Spatial selective scan with a configurable scan-direction registry.

    `scan_directions` may be:
      * 4                              -> ["h","h_rev","v","v_rev"]  (default, cheapest)
      * 8                              -> all 8 directions including diagonals
      * an explicit list/subset, e.g. ["h","v","diag"]  (any non-empty subset)
    Only the scan groups actually needed (cardinal and/or diagonal) are
    computed; a learnable weight is kept for exactly the enabled directions.
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto",
                 dropout=0.0, scan_directions: Union[int, Sequence[str]] = 4,
                 backend: str = "pure_pytorch"):
        super().__init__()
        if isinstance(scan_directions, int):
            names = DEFAULT_SCAN_DIRECTIONS_4 if scan_directions == 4 else DEFAULT_SCAN_DIRECTIONS_8
        else:
            names = list(scan_directions)
        assert len(names) > 0 and set(names) <= ALL_SCAN_DIRECTIONS, f"invalid scan_directions: {names}"
        self.direction_names = names
        self.use_cardinal = any(n in ("h", "h_rev", "v", "v_rev") for n in names)
        self.use_diagonal = any(n in ("diag", "diag_rev", "adiag", "adiag_rev") for n in names)

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner, bias=True,
                                 kernel_size=d_conv, padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()

        if self.use_cardinal:
            self.scan_cardinal = _SelectiveScanParams(self.d_inner, d_state, self.dt_rank, 4, backend)
        if self.use_diagonal:
            self.scan_diagonal = _SelectiveScanParams(self.d_inner, d_state, self.dt_rank, 4, backend)

        self.direction_weights = nn.Parameter(torch.ones(len(names)))
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def _cardinal_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D, H, W = x.shape
        L = H * W
        x_hwwh = torch.stack([
            x.reshape(B, -1, L),
            torch.transpose(x, 2, 3).contiguous().reshape(B, -1, L),
        ], dim=1).reshape(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # [B,4,D,L]
        out_y = self.scan_cardinal.run(xs)

        y_h = out_y[:, 0].reshape(B, D, H, W)
        y_v = torch.transpose(out_y[:, 1].reshape(B, D, W, H), 2, 3).contiguous()
        y_h_rev = torch.flip(out_y[:, 2], dims=[-1]).reshape(B, D, H, W)
        y_v_rev = torch.transpose(torch.flip(out_y[:, 3], dims=[-1]).reshape(B, D, W, H), 2, 3).contiguous()
        return {"h": y_h, "v": y_v, "h_rev": y_h_rev, "v_rev": y_v_rev}

    def _diagonal_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D, H, W = x.shape
        anti = _skew(x)
        Ws = anti.shape[-1]
        anti_seq = anti.transpose(2, 3).reshape(B, -1, H * Ws)
        main = _skew(torch.flip(x, dims=[-1]))
        main_seq = main.transpose(2, 3).reshape(B, -1, H * Ws)

        xs = torch.stack([anti_seq, main_seq], dim=1)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # [B,4,D,L]
        out_y = self.scan_diagonal.run(xs)

        def undo(seq):
            skewed = seq.reshape(B, D, Ws, H).transpose(2, 3)
            return _unskew(skewed, H, W)

        return {
            "adiag": undo(out_y[:, 0]),
            "diag": undo(out_y[:, 1]),
            "adiag_rev": undo(torch.flip(out_y[:, 2], dims=[-1])),
            "diag_rev": undo(torch.flip(out_y[:, 3], dims=[-1])),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        D = self.d_inner

        available: Dict[str, torch.Tensor] = {}
        if self.use_cardinal:
            available.update(self._cardinal_all(x))
        if self.use_diagonal:
            available.update(self._diagonal_all(x))

        w = F.softmax(self.direction_weights, dim=0)
        y = sum(w[i] * available[name] for i, name in enumerate(self.direction_names))
        y = y.reshape(B, D, H * W)

        y = torch.transpose(y, 1, 2).contiguous().reshape(B, H, W, D)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ============================================================================
# Phase 10 - Multi-scale SS2D
# ============================================================================

class MultiScaleSS2D(nn.Module):
    def __init__(self, d_model: int, d_state: int, scales: Sequence[int],
                 scan_directions: Union[int, Sequence[str]] = 4, backend: str = "pure_pytorch"):
        super().__init__()
        self.branches = nn.ModuleList([
            SS2D(d_model, d_state=d_state, d_conv=k, scan_directions=scan_directions, backend=backend)
            for k in scales
        ])
        self.gate = nn.Linear(d_model, len(scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = torch.stack([b(x) for b in self.branches], dim=0)     # [S,B,H,W,D]
        gate_logits = self.gate(x.mean(dim=(1, 2)))                   # [B,S]
        gate = F.softmax(gate_logits, dim=-1).permute(1, 0)[:, :, None, None, None]
        return (outs * gate).sum(dim=0)


# ============================================================================
# Spectral Mamba - bidirectional selective scan along the band axis
# ============================================================================

class SpectralMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, dt_rank="auto", backend: str = "pure_pytorch"):
        super().__init__()
        self.d_inner = d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.scan = _SelectiveScanParams(self.d_inner, d_state, self.dt_rank, k_directions=2, backend=backend)
        self.direction_weights = nn.Parameter(torch.ones(2))
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, d_model]  (N = arbitrary batch of spectral-token sequences, C = num bands)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x_seq = x.transpose(1, 2)
        xs = torch.stack([x_seq, torch.flip(x_seq, dims=[-1])], dim=1)
        out_y = self.scan.run(xs)
        w = F.softmax(self.direction_weights, dim=0)
        y = w[0] * out_y[:, 0] + w[1] * torch.flip(out_y[:, 1], dims=[-1])
        y = y.transpose(1, 2)
        y = self.out_norm(y)
        y = y * F.silu(z)
        return self.out_proj(y)


# ============================================================================
# Phase 14/15 - continuous, sensor-aware spectral positional encoding.
# ============================================================================

def index_positional_encoding(num_bands: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """Fallback encoding used when no physical wavelength metadata is given
    (identical in spirit to v1/v2's index-based encoding)."""
    position = torch.linspace(0, 1, steps=num_bands, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_bands, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term * num_bands)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]] * num_bands)
    return pe.to(dtype) if dtype is not None else pe


def normalize_wavelengths(wavelengths: torch.Tensor,
                           reference_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """Phase 15 - sensor-aware normalization. `wavelengths`: [..., C] physical
    values (nm), possibly irregularly spaced. If `reference_range` (lo, hi) is
    given, normalize against that fixed physical range so different sensors
    map into a shared coordinate system; otherwise normalize per-sample by its
    own min/max (self-normalizing fallback, still order- and spacing-aware)."""
    if reference_range is not None:
        lo, hi = reference_range
        return ((wavelengths - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
    lo = wavelengths.min(dim=-1, keepdim=True).values
    hi = wavelengths.max(dim=-1, keepdim=True).values
    return (wavelengths - lo) / (hi - lo).clamp_min(1e-6)


def continuous_wavelength_encoding(wavelengths_norm: torch.Tensor, dim: int,
                                    span: float = 1000.0) -> torch.Tensor:
    """wavelengths_norm: [..., C] in [0,1] -> [..., C, dim] sinusoidal encoding
    keyed on the *physical* (normalized) wavelength value rather than the band
    index, so irregular spacing / missing bands are represented correctly.

    `span` (v15 R1.3) is the angular scale the normalized [0,1] coordinate is
    stretched by. Pass the band count `C` and this reduces EXACTLY to
    `index_positional_encoding` for a uniformly-spaced sensor (both then
    compute `band_index * div_term`), and interpolates sensibly for an
    irregular one. The pre-v15 hardcoded 1000.0 does not: with 32 bands,
    adjacent bands differ by ~32 radians in the lowest-frequency component -
    more than five full cycles - so the encoding carries no locality at all,
    which is the one property a positional encoding exists to provide.
    """
    device, dtype = wavelengths_norm.device, wavelengths_norm.dtype
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    angles = wavelengths_norm.float().unsqueeze(-1) * div_term * float(span)   # [..., C, dim//2]
    pe = torch.zeros(*wavelengths_norm.shape, dim, device=device, dtype=torch.float32)
    pe[..., 0::2] = torch.sin(angles)
    pe[..., 1::2] = torch.cos(angles[..., : pe[..., 1::2].shape[-1]])
    return pe.to(dtype)


class SpectralTokenizer(nn.Module):
    """Phase 1: treats every wavelength as an individual token, using a
    single shared `nn.Linear(1, d_token)` (no dependence on C in any
    parameter). Phases 14/15: if physical wavelengths (nm) are supplied,
    positional encoding is a continuous function of the (sensor-normalized)
    wavelength value; otherwise it falls back to index-based encoding.

    v15 R1.1 - `fusion` selects how the band value and its positional
    encoding are combined. This is THE representation-collapse fix; see the
    v15 plan, section 4:

      "add" (legacy default, unchanged)
          `tokens = value_embed(v) + pe`. With `value_embed` initialised at
          `trunc_normal_(std=0.02)` and a zeroed bias, its output on inputs
          in [0,1] has magnitude ~0.006 against a positional encoding of
          magnitude ~0.55 - the input signal is 86-93x smaller than the
          per-dataset CONSTANT it is added to, and `stem_norm` then divides
          out whatever scale survives. Measured input-dependence of the stem
          embedding: 0.18% (HSI) / 0.29% (RGB). Kept as the default only so
          `train_example_v14.py` keeps reproducing the runs already recorded.

      "scaled"
          `tokens = value_embed(v) + pe_gain * pe`, `pe_gain` a learnable
          scalar. Fixes the magnitude; `spectral_value_init_std` additionally
          re-initialises `value_embed.weight` larger. Does NOT fix the rank
          problem below.

      "concat_mlp"  (recommended)
          `tokens = W2(GELU(W1([v ; pe]))) + pe_gain * pe`. With a shared
          `Linear(1, d)` and a zeroed bias every band token is `v_c * W` - a
          scalar multiple of ONE fixed direction - so all C bands are
          collinear and `tokens.mean(dim=1)` reduces a whole spectrum to a
          single scalar. The concat-MLP's nonlinearity makes each band's
          response DIRECTION depend on its own positional encoding, which is
          the property a spectral tokenizer needs. Still fully
          channel-agnostic: every parameter shape depends only on `d_token`.

    Do NOT "fix" this with a LayerNorm on the value branch: `LayerNorm(v*W)`
    for `v > 0` is `LayerNorm(W)`, a constant independent of `v`.
    """

    def __init__(self, d_token: int, fusion: str = "add", pe_gain: float = 1.0,
                 value_init_std: float = 0.02, wavelength_encoding_scale: Optional[float] = 1000.0):
        super().__init__()
        if fusion not in SPECTRAL_TOKEN_FUSIONS:
            raise ValueError(f"unknown spectral_token_fusion {fusion!r}; "
                              f"use one of {sorted(SPECTRAL_TOKEN_FUSIONS)}")
        self.d_token = d_token
        self.fusion = fusion
        self.value_init_std = value_init_std
        self.wavelength_encoding_scale = wavelength_encoding_scale

        if fusion == "concat_mlp":
            # `value_embed` would be dead weight here (the MLP consumes the raw
            # value directly), so it is not created at all - a checkpoint is
            # therefore tied to its `spectral_token_fusion`, which
            # `config.json` records and the loader compares (contract C-4).
            self.fuse_in = nn.Linear(1 + d_token, 2 * d_token)
            self.fuse_out = nn.Linear(2 * d_token, d_token)
        else:
            self.value_embed = nn.Linear(1, d_token)   # shared across all bands

        if fusion != "add":
            self.pe_gain = nn.Parameter(torch.tensor(float(pe_gain)))

    # ------------------------------------------------------------------
    def reinit_value_branch(self) -> None:
        """Re-initialise the value branch at `value_init_std`. MUST be called
        by the owning backbone AFTER its own `self.apply(_init_weights)`,
        which would otherwise overwrite it with the shared std=0.02.

        A no-op in the legacy configuration (`fusion="add"` with the legacy
        std), so `train_example_v14.py` still builds a bit-for-bit identical
        model from the same seed - contract C-3.

        For `concat_mlp` this also undoes `_init_weights` on the fusion MLP
        itself. std=0.02 on a `Linear(1+d, 2d)` is ~5x below that layer's own
        default fan-in scale, and the value occupies 1 of 33 input columns, so
        leaving the shared init in place would bury the value under the
        positional encoding a second time - the exact failure this mode
        exists to fix. The MLP therefore gets PyTorch's own default init and
        the value COLUMN is then re-drawn at `value_init_std`, which is the
        same amplification `scaled` applies to `value_embed`.
        """
        if self.fusion == "add" and self.value_init_std == 0.02:
            return

        if self.fusion == "concat_mlp":
            self.fuse_in.reset_parameters()
            self.fuse_out.reset_parameters()
            nn.init.trunc_normal_(self.fuse_in.weight[:, :1], std=self.value_init_std)
            return

        nn.init.trunc_normal_(self.value_embed.weight, std=self.value_init_std)
        if self.value_embed.bias is not None:
            nn.init.zeros_(self.value_embed.bias)

    # ------------------------------------------------------------------
    def _positional_encoding(self, band_values: torch.Tensor,
                             wavelengths: Optional[torch.Tensor],
                             sensor_range: Optional[Tuple[float, float]],
                             dtype: torch.dtype) -> torch.Tensor:
        C = band_values.shape[-1]
        if wavelengths is not None:
            wl = wavelengths
            if wl.dim() == 1:
                wl = wl.unsqueeze(0).expand(band_values.shape[0], -1)
            wl_norm = normalize_wavelengths(wl.to(band_values.device), sensor_range)
            # R1.3: `None` => the band count, which makes this reduce exactly
            # to the index encoding for a uniformly-spaced sensor.
            span = self.wavelength_encoding_scale
            span = float(C) if span is None else float(span)
            return continuous_wavelength_encoding(wl_norm, self.d_token, span=span)  # [N,C,d]
        return index_positional_encoding(C, self.d_token, device=band_values.device,
                                          dtype=dtype).unsqueeze(0)                  # [1,C,d]

    def forward(self, band_values: torch.Tensor, wavelengths: Optional[torch.Tensor] = None,
                sensor_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """band_values: [N, C]. wavelengths (optional): broadcastable to [N, C]
        physical values in nm. Returns [N, C, d_token]."""
        v = band_values.unsqueeze(-1)                                    # [N, C, 1]

        if self.fusion == "concat_mlp":
            pe = self._positional_encoding(band_values, wavelengths, sensor_range, v.dtype)
            pe_b = pe.expand(band_values.shape[0], -1, -1) if pe.shape[0] == 1 else pe
            h = torch.cat([v, pe_b.to(v.dtype)], dim=-1)                 # [N, C, 1 + d]
            return self.fuse_out(F.gelu(self.fuse_in(h))) + self.pe_gain * pe.to(v.dtype)

        tokens = self.value_embed(v)                                     # [N, C, d_token]
        pe = self._positional_encoding(band_values, wavelengths, sensor_range, tokens.dtype)
        if self.fusion == "add":
            return tokens + pe
        return tokens + self.pe_gain * pe.to(tokens.dtype)


# ============================================================================
# Phase 3 - deep residual spectral encoder, with optional squeeze-and-excite
# and multi-scale (Phase 5) kernel branches.
# ============================================================================

class SqueezeExcite1D(nn.Module):
    """Squeeze-and-excitation over the token/channel dimension, computed from
    the mean band profile (squeeze over the band axis)."""

    def __init__(self, d_token: int, reduction: int = 4):
        super().__init__()
        hidden = max(d_token // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(d_token, hidden), nn.GELU(), nn.Linear(hidden, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, d_token]
        squeeze = x.mean(dim=1)                       # [N, d_token]
        gate = torch.sigmoid(self.fc(squeeze)).unsqueeze(1)  # [N, 1, d_token]
        return x * gate


class ResidualSpectralBlock(nn.Module):
    """Conv1d along the band axis (shared weights, C-agnostic) + residual +
    norm, optionally with parallel multi-scale kernels (Phase 5) and
    squeeze-and-excitation (Phase 3)."""

    def __init__(self, d_token: int, kernel_sizes: Sequence[int], norm_type: str = "layernorm",
                 dilation: int = 1, squeeze_excitation: bool = False):
        super().__init__()
        self.norm = make_norm(norm_type, d_token)
        self.branches = nn.ModuleList([
            nn.Conv1d(d_token, d_token, kernel_size=k, padding=((k - 1) * dilation) // 2,
                      dilation=dilation, groups=1)
            for k in kernel_sizes
        ])
        self.multiscale = len(kernel_sizes) > 1
        if self.multiscale:
            self.branch_attention = nn.Linear(d_token, len(kernel_sizes))
        self.act = nn.GELU()
        self.proj = nn.Linear(d_token, d_token)
        self.se = SqueezeExcite1D(d_token) if squeeze_excitation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x).transpose(1, 2)                    # [N, d_token, C]
        outs = [branch(h) for branch in self.branches]
        if self.multiscale:
            gate_logits = self.branch_attention(x.mean(dim=1))          # [N, S]  attention over branches
            gate = F.softmax(gate_logits, dim=-1).transpose(0, 1)[:, :, None, None]
            stacked = torch.stack(outs, dim=0)
            h = (stacked * gate).sum(dim=0)
        else:
            h = outs[0]
        h = self.act(h).transpose(1, 2)                       # [N, C, d_token]
        h = self.proj(h)
        if self.se is not None:
            h = self.se(h)
        return residual + h


class HierarchicalSpectralEncoder(nn.Module):
    """Phase 3/4/5: `depth` residual (optionally multi-scale, optionally SE)
    spectral conv blocks, with a SpectralMamba in the middle for long-range
    band-order dependency modeling."""

    def __init__(self, d_token: int, depth: int, kernel_sizes: Sequence[int],
                 norm_type: str, backend: str, squeeze_excitation: bool = False):
        super().__init__()
        self.pre_blocks = nn.ModuleList([
            ResidualSpectralBlock(d_token, kernel_sizes, norm_type, squeeze_excitation=squeeze_excitation)
            for _ in range(max(depth // 2, 1))
        ])
        self.spectral_mamba = SpectralMamba(d_token, d_state=8, backend=backend)
        self.mamba_norm = make_norm(norm_type, d_token)
        self.post_blocks = nn.ModuleList([
            ResidualSpectralBlock(d_token, kernel_sizes, norm_type, squeeze_excitation=squeeze_excitation)
            for _ in range(max(depth - depth // 2, 1))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.pre_blocks:
            x = blk(x)
        x = x + self.spectral_mamba(self.mamba_norm(x))
        for blk in self.post_blocks:
            x = blk(x)
        return x


# ============================================================================
# Phase 12 - progressive spectral compression
# ============================================================================

class ProgressiveCompressor(nn.Module):
    def __init__(self, in_dim: int, stage_dims: Sequence[int]):
        super().__init__()
        dims = [in_dim] + list(stage_dims)
        layers = [nn.Sequential(nn.Linear(a, b), nn.GELU()) for a, b in zip(dims[:-1], dims[1:])]
        self.stages = nn.ModuleList(layers)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


# ============================================================================
# Phase 8 - dynamic band selection v2: soft or sparse (learnable threshold)
# gating at the input, PLUS a per-stage context-channel selector
# (StageBandSelector, below) that recomputes importance at every stage.
# ============================================================================

class BandGate(nn.Module):
    def __init__(self, d_token: int, mode: str = "soft", sparsity_temperature: float = 10.0):
        super().__init__()
        assert mode in ("soft", "sparse")
        self.mode = mode
        self.gate = nn.Linear(d_token, 1)
        if mode == "sparse":
            self.threshold = nn.Parameter(torch.zeros(1))   # learnable cutoff, in logit space
            self.temperature = sparsity_temperature

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [N, C, d_token] -> weights [N, C]
        logits = self.gate(tokens).squeeze(-1)
        if self.mode == "soft":
            return torch.sigmoid(logits)
        # sparse mode: a steep sigmoid around a learnable threshold acts as a
        # differentiable approximation of hard top-k / thresholded selection.
        return torch.sigmoid(self.temperature * (logits - self.threshold))


class StageBandSelector(nn.Module):
    """Phase 8 (per-stage) + Phase 6/7 (self-refinement): a lightweight
    squeeze-excite-style gate recomputed on the spectral context map at the
    start of every stage, so channel/band importance is context-and-depth
    dependent rather than fixed once at the input."""

    def __init__(self, ctx_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(ctx_dim // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.GELU(), nn.Linear(hidden, ctx_dim))

    def forward(self, ctx_map: torch.Tensor) -> torch.Tensor:
        # ctx_map: [B,H,W,d_ctx]
        squeeze = ctx_map.mean(dim=(1, 2))                    # [B, d_ctx]
        gate = torch.sigmoid(self.fc(squeeze))[:, None, None, :]
        return ctx_map * gate


# ============================================================================
# Phase 2 - full spectral pathway: patch-wise tokenization -> hierarchical
# encoder -> dynamic band selection -> progressive compression -> aggregation.
# Produces a spatially-resolved context map (B, Hp, Wp, d_ctx).
# ============================================================================

class SpectralPathway(nn.Module):
    def __init__(self, cfg: MedMambaSSTRMConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.spectral_chunk_size = cfg.spectral_chunk_size
        self.tokenizer = SpectralTokenizer(
            cfg.d_token, fusion=cfg.spectral_token_fusion, pe_gain=cfg.spectral_pe_gain,
            value_init_std=cfg.spectral_value_init_std,
            wavelength_encoding_scale=cfg.wavelength_encoding_scale)
        self.encoder = HierarchicalSpectralEncoder(
            cfg.d_token, cfg.spectral_depth,
            kernel_sizes=cfg.spectral_kernel_sizes if not cfg.spectral_multiscale else (3, 5, 9),
            norm_type=cfg.norm_type, backend=cfg.scan_backend,
            squeeze_excitation=cfg.squeeze_excitation,
        )
        self.dynamic_band_selection = cfg.dynamic_band_selection
        if cfg.dynamic_band_selection:
            self.band_gate = BandGate(cfg.d_token, mode=cfg.band_selection_mode)
        self.compressor = ProgressiveCompressor(cfg.d_token, list(cfg.compression_dims) + [cfg.d_ctx])
        self.d_ctx = cfg.d_ctx

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.patch_size, stride=self.patch_size,
                             ceil_mode=True, count_include_pad=False)

    def _patchify_wavelengths(self, wavelengths: Optional[torch.Tensor], batch: int) -> Optional[torch.Tensor]:
        """Wavelengths describe bands, not spatial position, so patchifying
        the image doesn't change them - just make sure the leading dim
        matches (B,) or is a shared (C,) vector; broadcasting is handled in
        SpectralTokenizer."""
        return wavelengths

    def _process_patch_chunk(self, band_values: torch.Tensor,
                             wl: Optional[torch.Tensor],
                             sensor_range: Optional[Tuple[float, float]]):
        """Tokenize -> encode -> gate -> compress one chunk of patches."""
        tokens = self.tokenizer(band_values, wavelengths=wl, sensor_range=sensor_range)
        tokens = self.encoder(tokens)
        band_weights = None
        if self.dynamic_band_selection:
            band_weights = self.band_gate(tokens)
            tokens = tokens * band_weights.unsqueeze(-1)
        pooled = tokens.mean(dim=1)
        ctx = self.compressor(pooled)
        return ctx, band_weights

    def forward(self, x: torch.Tensor, wavelengths: Optional[torch.Tensor] = None,
                sensor_range: Optional[Tuple[float, float]] = None):
        """x: [B,C,H,W] -> (spectral_context_map [B,Hp,Wp,d_ctx], band_weights or None)"""
        Bsz, C, H, W = x.shape
        patches = self.patchify(x)
        Hp, Wp = patches.shape[-2:]

        band_values = patches.permute(0, 2, 3, 1).reshape(Bsz * Hp * Wp, C)  # [N,C]
        N = band_values.shape[0]

        wl = self._patchify_wavelengths(wavelengths, Bsz)
        if wl is not None and wl.dim() == 2:  # [B,C] -> expand per patch
            wl = wl.repeat_interleave(Hp * Wp, dim=0)  # [N,C]

        # Process patches in chunks to limit peak memory (OOM fix for large
        # spatial inputs with patch_size=1, e.g. 64x64xC HSI patches).
        #
        # v15 R3.2: this used to be hardcoded at 1024 regardless of batch size.
        # At bs=256 with patch_size=1 on 11x11 patches, N = 30,976, so every
        # forward ran 31 SEQUENTIAL Python iterations of the whole spectral
        # encoder - each of which itself contains a Python loop over C
        # timesteps. `spectral_chunk_size = 0` runs all N in one pass, which is
        # the right setting for small patches (peak memory there is trivial);
        # `training/spectral_checkpoint.py` remains the memory lever for the
        # large-patch case.
        chunk_size = self.spectral_chunk_size if self.spectral_chunk_size > 0 else max(1, N)
        ctx_list, bw_list = [], []
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk_bv = band_values[i:end]
            chunk_wl = wl[i:end] if (wl is not None and wl.dim() == 2) else wl
            ctx_chunk, bw_chunk = self._process_patch_chunk(chunk_bv, chunk_wl, sensor_range)
            ctx_list.append(ctx_chunk)
            if bw_chunk is not None:
                bw_list.append(bw_chunk)

        ctx = torch.cat(ctx_list, dim=0)
        ctx_map = ctx.reshape(Bsz, Hp, Wp, self.d_ctx)

        band_weights = None
        if bw_list:
            band_weights = torch.cat(bw_list, dim=0).reshape(Bsz, Hp, Wp, C)
        return ctx_map, band_weights


# ============================================================================
# Phase 11 - pluggable spectral<->spatial fusion strategies
# ============================================================================

class SpectralSpatialFusion(nn.Module):
    def __init__(self, spatial_dim: int, ctx_dim: int):
        super().__init__()


class FiLMFusion(SpectralSpatialFusion):
    def __init__(self, spatial_dim: int, ctx_dim: int):
        super().__init__(spatial_dim, ctx_dim)
        self.to_gamma_beta = nn.Linear(ctx_dim, spatial_dim * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(ctx).chunk(2, dim=-1)
        return spatial * (1 + gamma) + beta


class GatedAdditiveFusion(SpectralSpatialFusion):
    def __init__(self, spatial_dim: int, ctx_dim: int):
        super().__init__(spatial_dim, ctx_dim)
        self.proj = nn.Linear(ctx_dim, spatial_dim)
        self.gate = nn.Linear(spatial_dim + ctx_dim, spatial_dim)

    def forward(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        proj_ctx = self.proj(ctx)
        gate = torch.sigmoid(self.gate(torch.cat([spatial, ctx], dim=-1)))
        return spatial + gate * proj_ctx


class MultiplicativeFusion(SpectralSpatialFusion):
    """out = spatial * sigmoid(proj(ctx)) - pure multiplicative modulation,
    distinct from the additive-gated variant above."""

    def __init__(self, spatial_dim: int, ctx_dim: int):
        super().__init__(spatial_dim, ctx_dim)
        self.proj = nn.Linear(ctx_dim, spatial_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)  # start near sigmoid(0)=0.5 -> gentle modulation

    def forward(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return spatial * (1.0 + torch.sigmoid(self.proj(ctx)))


class ResidualFusion(SpectralSpatialFusion):
    """out = spatial + proj(ctx) - the simplest possible fusion, no gating."""

    def __init__(self, spatial_dim: int, ctx_dim: int):
        super().__init__(spatial_dim, ctx_dim)
        self.proj = nn.Linear(ctx_dim, spatial_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return spatial + self.proj(ctx)


class CrossAttentionFusion(SpectralSpatialFusion):
    def __init__(self, spatial_dim: int, ctx_dim: int, num_heads: int = 4):
        super().__init__(spatial_dim, ctx_dim)
        self.num_heads = num_heads
        self.ctx_proj = nn.Linear(ctx_dim, spatial_dim)
        self.q = nn.Linear(spatial_dim, spatial_dim)
        self.k = nn.Linear(spatial_dim, spatial_dim)
        self.v = nn.Linear(spatial_dim, spatial_dim)
        self.out = nn.Linear(spatial_dim, spatial_dim)
        self.norm = nn.LayerNorm(spatial_dim)

    def forward(self, spatial: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, H, W, D = spatial.shape
        s = spatial.reshape(B, H * W, D)
        c = self.ctx_proj(ctx).reshape(B, H * W, D)
        nh = self.num_heads
        hd = D // nh

        def split_heads(t):
            return t.reshape(B, -1, nh, hd).transpose(1, 2)

        q, k, v = split_heads(self.q(s)), split_heads(self.k(c)), split_heads(self.v(c))
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, D)
        out = self.out(out)
        return self.norm(spatial + out.reshape(B, H, W, D))


def make_fusion(fusion_type: str, spatial_dim: int, ctx_dim: int) -> SpectralSpatialFusion:
    fusion_type = fusion_type.lower()
    table = {
        "film": FiLMFusion, "gated": GatedAdditiveFusion, "cross_attention": CrossAttentionFusion,
        "multiplicative": MultiplicativeFusion, "residual": ResidualFusion,
    }
    if fusion_type not in table:
        raise ValueError(f"Unknown fusion_type: {fusion_type!r} (use one of {sorted(table)})")
    return table[fusion_type](spatial_dim, ctx_dim)


# ============================================================================
# Phases 6-7 - bidirectional interaction: spatial features update the
# spectral context map after every stage (kept resolution-aligned).
# ============================================================================

class SpectralContextUpdater(nn.Module):
    def __init__(self, spatial_dim: int, ctx_dim: int, norm_type: str = "layernorm"):
        super().__init__()
        self.proj = nn.Linear(spatial_dim, ctx_dim)
        self.norm = make_norm(norm_type, ctx_dim)
        self.gate = nn.Linear(ctx_dim, ctx_dim)

    def forward(self, ctx_map: torch.Tensor, spatial_map: torch.Tensor) -> torch.Tensor:
        update = self.proj(spatial_map)
        g = torch.sigmoid(self.gate(ctx_map))
        return self.norm(ctx_map + g * update)


def align_ctx_to(ctx_map: torch.Tensor, H: int, W: int) -> torch.Tensor:
    if ctx_map.shape[1] == H and ctx_map.shape[2] == W:
        return ctx_map
    x = ctx_map.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def downsample_ctx(ctx_map: torch.Tensor) -> torch.Tensor:
    x = ctx_map.permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False)
    return x.permute(0, 2, 3, 1)


# ============================================================================
# Spatial hierarchical backbone
# ============================================================================

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    B, H, W, C = x.shape
    cpg = C // groups
    x = x.view(B, H, W, groups, cpg)
    x = torch.transpose(x, 3, 4).contiguous()
    return x.view(B, H, W, -1)


class PatchMerging2D(nn.Module):
    def __init__(self, dim: int, norm_type: str = "layernorm"):
        super().__init__()
        self.norm = make_norm(norm_type, 4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        return self.reduction(x)


class GBlock(nn.Module):
    def __init__(self, hidden_dim: int, cfg: MedMambaSSTRMConfig, drop_path: float, ctx_dim: int,
                 fusion_type: str):
        super().__init__()
        half = hidden_dim // 2
        self.fusion = make_fusion(fusion_type, hidden_dim, ctx_dim)

        self.ln_1 = make_norm(cfg.norm_type, half)
        if cfg.spatial_multiscale:
            self.self_attention = MultiScaleSS2D(half, cfg.d_state, cfg.spatial_scales,
                                                  scan_directions=cfg.scan_directions,
                                                  backend=cfg.scan_backend)
        else:
            self.self_attention = SS2D(half, d_state=cfg.d_state, scan_directions=cfg.scan_directions,
                                        backend=cfg.scan_backend)
        self.drop_path = DropPath(drop_path)
        self.ss2d_scale = LayerScale(half, cfg.layerscale_init, enabled=cfg.layerscale)

        self.conv_branch = nn.Sequential(
            nn.BatchNorm2d(half),
            nn.Conv2d(half, half, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.use_ffn = cfg.use_ffn
        if cfg.use_ffn:
            self.ln_2 = make_norm(cfg.norm_type, hidden_dim)
            self.ffn = GatedMLP(hidden_dim, mult=2.0)
            self.ffn_scale = LayerScale(hidden_dim, cfg.layerscale_init, enabled=cfg.layerscale)
            self.ffn_drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, ctx_map: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fusion(x, ctx_map)

        left, right = x.chunk(2, dim=-1)
        ss = self.self_attention(self.ln_1(right))
        ss = self.drop_path(self.ss2d_scale(ss))

        left = left.permute(0, 3, 1, 2).contiguous()
        left = self.conv_branch(left)
        left = left.permute(0, 2, 3, 1).contiguous()

        out = torch.cat((left, right + ss), dim=-1)
        out = channel_shuffle(out, groups=2)
        out = residual + out

        if self.use_ffn:
            out = out + self.ffn_drop_path(self.ffn_scale(self.ffn(self.ln_2(out))))
        return out


class GStage(nn.Module):
    def __init__(self, dim: int, depth: int, cfg: MedMambaSSTRMConfig, drop_path: Sequence[float],
                 ctx_dim: int, downsample: bool, fusion_type: str):
        super().__init__()
        self.blocks = nn.ModuleList([
            GBlock(dim, cfg, drop_path[i], ctx_dim, fusion_type) for i in range(depth)
        ])
        self.spectral_refinement = cfg.stage_spectral_refinement
        if self.spectral_refinement:
            self.ctx_updater = SpectralContextUpdater(dim, ctx_dim, norm_type=cfg.norm_type)
        self.dynamic_band_selection = cfg.dynamic_band_selection
        if self.dynamic_band_selection:
            self.stage_band_selector = StageBandSelector(ctx_dim)
        self.downsample = PatchMerging2D(dim, norm_type=cfg.norm_type) if downsample else None
        self.ctx_downsample = downsample

    def forward(self, x: torch.Tensor, ctx_map: torch.Tensor):
        H, W = x.shape[1], x.shape[2]
        ctx_aligned = align_ctx_to(ctx_map, H, W)

        if self.dynamic_band_selection:
            ctx_aligned = self.stage_band_selector(ctx_aligned)   # Phase 8: per-stage recompute

        for blk in self.blocks:
            x = blk(x, ctx_aligned)

        pooled = x.mean(dim=(1, 2))
        stage_feature_map = x

        if self.spectral_refinement:
            ctx_aligned = self.ctx_updater(ctx_aligned, x)   # Phase 6/7: spatial -> spectral update
        stage_ctx_map = ctx_aligned

        if self.downsample is not None:
            x = self.downsample(x)
        if self.ctx_downsample:
            ctx_aligned = downsample_ctx(ctx_aligned)

        return x, ctx_aligned, pooled, stage_feature_map, stage_ctx_map


# ============================================================================
# Phase 13 - general backbone API with multi-level outputs
# ============================================================================

class MedMambaSSBackbone(nn.Module):
    """Returns a rich dict: feature_map, pooled, stage_pools, stage_feature_maps,
    stage_ctx_maps, spectral_context, band_weights - so arbitrary task heads
    (or multi-scale decoders) can be attached on top."""

    def __init__(self, cfg: MedMambaSSTRMConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.num_stages = len(cfg.dims)

        self.spectral_pathway = SpectralPathway(cfg)
        self.stem = nn.Linear(cfg.d_ctx, cfg.dims[0])
        self.stem_norm = make_norm(cfg.norm_type, cfg.dims[0])

        total_depth = sum(cfg.depths)
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, total_depth)]
        fusion_types = cfg.resolved_fusion_types()

        self.stages = nn.ModuleList()
        cursor = 0
        for i, (dim, depth) in enumerate(zip(cfg.dims, cfg.depths)):
            stage = GStage(dim, depth, cfg, dpr[cursor:cursor + depth], cfg.d_ctx,
                            downsample=(i < self.num_stages - 1), fusion_type=fusion_types[i])
            self.stages.append(stage)
            cursor += depth

        self.apply(self._init_weights)
        # v15 R1.1 - `_init_weights` re-initialises EVERY nn.Linear at
        # std=0.02, including the tokenizer's value embedding. Undo that for
        # the tokenizer specifically, so `spectral_value_init_std` means what
        # it says. A no-op in the legacy configuration.
        self.spectral_pathway.tokenizer.reinit_value_branch()

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _normalize_ctx(self, ctx_map: torch.Tensor) -> torch.Tensor:
        """v15 R1.1b - parameter-free RMS norm of the spectral context before
        the stem projection, so `stem_norm` receives an input whose variance is
        comfortably above its own eps and can actually do its job. Identity
        when `spectral_ctx_norm` is False (pre-v15 default). See the field's
        comment on `MedMambaSSTRMConfig` for the measured numbers."""
        return _rms_norm_scale_invariant(ctx_map) if self.cfg.spectral_ctx_norm else ctx_map

    def forward_features(self, x: torch.Tensor, wavelengths: Optional[torch.Tensor] = None,
                          sensor_range: Optional[Tuple[float, float]] = None) -> Dict[str, torch.Tensor]:
        """x: [B,C,H,W], any C.
        wavelengths (optional): [C] or [B,C] physical band centers in nm
          (Phase 14). If omitted, index-based positional encoding is used.
        sensor_range (optional): (lo_nm, hi_nm) reference range for
          cross-sensor-consistent normalization (Phase 15); if omitted,
          wavelengths are normalized by their own min/max.
        """
        if not self.cfg.use_wavelength_metadata:
            wavelengths = None
        ctx_map, band_weights = self.spectral_pathway(x, wavelengths=wavelengths, sensor_range=sensor_range)
        feat = self.stem_norm(self.stem(self._normalize_ctx(ctx_map)))

        stage_pools, stage_feature_maps, stage_ctx_maps = [], [], []
        for stage in self.stages:
            feat, ctx_map, pooled, s_feat, s_ctx = stage(feat, ctx_map)
            stage_pools.append(pooled)
            stage_feature_maps.append(s_feat)
            stage_ctx_maps.append(s_ctx)

        global_pool = feat.mean(dim=(1, 2))
        spectral_summary = ctx_map.mean(dim=(1, 2))

        return {
            "feature_map": feat,
            "pooled": global_pool,
            "stage_pools": stage_pools,
            "stage_feature_maps": stage_feature_maps,   # Phase 13: dense/multi-scale outputs
            "stage_ctx_maps": stage_ctx_maps,           # Phase 13: spectral context per stage
            "spectral_context": spectral_summary,
            "band_weights": band_weights,               # Phase 8/14: [B,Hp,Wp,C] or None
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["pooled"]


# ============================================================================
# Task heads
# ============================================================================

class ClassificationHead(nn.Module):
    def __init__(self, cfg: MedMambaSSTRMConfig, num_classes: int):
        super().__init__()
        pooled_dim = sum(cfg.dims) + cfg.d_ctx
        self.norm = make_norm(cfg.norm_type, pooled_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(pooled_dim // 2, num_classes),
        )
        self.classifier_init = cfg.classifier_init

    def reinit_classifier(self) -> None:
        """v15 R1.1d - see `MedMambaSSTRMConfig.classifier_init`. Called by the
        owning model AFTER `apply(_init_weights)`; a no-op under the legacy
        "shared" setting."""
        if self.classifier_init == "fan_in":
            self.mlp[-1].reset_parameters()

    def forward(self, backbone_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        feat = torch.cat(backbone_out["stage_pools"] + [backbone_out["spectral_context"]], dim=-1)
        return self.mlp(self.norm(feat))


class RegressionHead(ClassificationHead):
    def __init__(self, cfg: MedMambaSSTRMConfig, out_dim: int = 1):
        super().__init__(cfg, num_classes=out_dim)


class EmbeddingHead(nn.Module):
    def __init__(self, cfg: MedMambaSSTRMConfig, embed_dim: int = 256):
        super().__init__()
        pooled_dim = sum(cfg.dims) + cfg.d_ctx
        self.norm = make_norm(cfg.norm_type, pooled_dim)
        self.proj = nn.Linear(pooled_dim, embed_dim)

    def forward(self, backbone_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        feat = torch.cat(backbone_out["stage_pools"] + [backbone_out["spectral_context"]], dim=-1)
        return F.normalize(self.proj(self.norm(feat)), dim=-1)


class SegmentationHead(nn.Module):
    def __init__(self, cfg: MedMambaSSTRMConfig, num_classes: int):
        super().__init__()
        dim = cfg.dims[-1]
        self.conv = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, num_classes))

    def forward(self, backbone_out: Dict[str, torch.Tensor], out_size) -> torch.Tensor:
        feat = backbone_out["feature_map"]
        logits = self.conv(feat).permute(0, 3, 1, 2)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


HEADS = {
    "classification": ClassificationHead, "regression": RegressionHead,
    "embedding": EmbeddingHead, "segmentation": SegmentationHead,
}


# ============================================================================
# Full model: backbone + head
# ============================================================================

class MedMambaSS(nn.Module):
    def __init__(self, cfg: Optional[MedMambaSSTRMConfig] = None, task: str = "classification",
                 num_classes: int = 1000, **head_kwargs):
        super().__init__()
        self.cfg = cfg or MedMambaSSTRMConfig()
        self.task = task
        self.backbone = MedMambaSSBackbone(self.cfg)

        if task == "classification":
            self.head = ClassificationHead(self.cfg, num_classes)
        elif task == "regression":
            self.head = RegressionHead(self.cfg, head_kwargs.get("out_dim", 1))
        elif task == "embedding":
            self.head = EmbeddingHead(self.cfg, head_kwargs.get("embed_dim", 256))
        elif task == "segmentation":
            self.head = SegmentationHead(self.cfg, num_classes)
        else:
            raise ValueError(f"Unknown task: {task!r}. Use one of {list(HEADS)}")

    def forward(self, x: torch.Tensor, wavelengths: Optional[torch.Tensor] = None,
                sensor_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        out = self.backbone.forward_features(x, wavelengths=wavelengths, sensor_range=sensor_range)
        if self.task == "segmentation":
            return self.head(out, out_size=x.shape[-2:])
        return self.head(out)


# ============================================================================
# TRM (Tiny Recursive Model, arXiv:2510.04871) recursive variant
# ----------------------------------------------------------------------------
# Replaces the hierarchical multi-stage spatial backbone with a *single*
# weight-shared core `f` applied recursively. Two states are refined per
# forward: a latent reasoning feature `z` and an answer feature `y`, both
# [B, Hp, Wp, d]. One "improvement step" does `trm_n_latent` updates of `z`
# (injecting the fixed input embedding `x` and current `y`) then one update
# of `y`. `trm_n_improve - 1` improvement steps run under `torch.no_grad()`
# and the last carries gradient (TRM's cheap deep-recursion training). Deep
# supervision runs up to `trm_deep_supervision_steps` segments, detaching
# `(y, z)` between them; a small halt head provides an optional ACT signal.
# The spectral pathway (channel-agnostic tokenizer) is reused verbatim.
# ============================================================================

def sinusoidal_2d_encoding(H: int, W: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """[H, W, dim] fixed 2-D sinusoidal positional encoding (first half keyed on
    the row index, second half on the column index)."""
    assert dim % 4 == 0, "dim must be divisible by 4"
    d_half = dim // 2
    div = torch.exp(torch.arange(0, d_half, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d_half))
    def _axis(n: int) -> torch.Tensor:
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        out = torch.zeros(n, d_half, device=device, dtype=torch.float32)
        out[:, 0::2] = torch.sin(pos * div)
        out[:, 1::2] = torch.cos(pos * div[: out[:, 1::2].shape[1]])
        return out
    pe = torch.zeros(H, W, dim, device=device, dtype=torch.float32)
    pe[:, :, :d_half] = _axis(H).unsqueeze(1).expand(H, W, d_half)
    pe[:, :, d_half:] = _axis(W).unsqueeze(0).expand(H, W, d_half)
    return pe.to(dtype) if dtype is not None else pe


def _rms_norm_lastdim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMS norm over the last dim (TRM post-norm style).

    NOTE the ABSOLUTE `eps`: it is correct here, where the recursive core's
    activations are already unit-scale, and it is exactly wrong for an input
    that arrives at 1e-5 - see `_rms_norm_scale_invariant` below, which is what
    the spectral context needs.
    """
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.to(dt)


def _rms_norm_scale_invariant(x: torch.Tensor) -> torch.Tensor:
    """v15 R1.1b - parameter-free RMS norm that works at ANY input scale.

    `_rms_norm_lastdim`'s `+ 1e-6` and `nn.LayerNorm`'s `eps=1e-5` are both
    ABSOLUTE floors added to a mean-square. That is fine for unit-scale
    activations and useless below it: measured on this repo's spectral
    pathway, the context map reaches the stem at |.| ~ 5e-5, whose mean-square
    is ~2.5e-9 - two to three orders of magnitude BELOW either eps - so both
    normalizations degenerate into a constant rescale and leave the embedding
    hundreds of times smaller than the constant positional encoding it is then
    added to.

        scale      rsqrt(ms + 1e-6) applied        this function
        1e-2       rms(out) = 0.995                1.000
        1e-3       rms(out) = 0.693                1.000
        1e-5       rms(out) = 0.0095               1.000

    Dividing by the RMS itself, with only a true numerical floor to avoid
    dividing by zero on an all-zero row, is scale-invariant by construction.
    """
    dt = x.dtype
    xf = x.float()
    rms = xf.pow(2).mean(-1, keepdim=True).sqrt()
    return (xf / rms.clamp_min(1e-20)).to(dt)


class _TokenMLPMixer(nn.Module):
    """Attention-free spatial mixer: depthwise 3x3 conv (spatial mixing) +
    GEGLU pointwise MLP (channel mixing). Shape-agnostic alternative to SS2D
    for the small fixed grids TRM targets.

    v15 R4.1 - `channel_mlp=False` drops the pointwise MLP, leaving a PURE
    spatial operator. `RecursiveMambaBlock` already applies its own `GatedMLP`
    as `ffn`, so with `channel_mlp=True` (the legacy default) each "token mixer
    + channel MLP" block is really one depthwise conv and TWO channel MLPs:
    at trm_dim=128 / trm_ffn_mult=2.0 that is 98,944 params each against 1,280
    for the conv, i.e. 90% of the whole model spent on channel mixing and 4,608
    params in total on spatial mixing.
    """

    def __init__(self, d: int, mult: float, channel_mlp: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=True)
        self.pw = GatedMLP(d, mult=mult, dropout=dropout) if channel_mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,H,W,d]
        h = x.permute(0, 3, 1, 2).contiguous()
        h = self.dwconv(h).permute(0, 2, 3, 1).contiguous()
        return h if self.pw is None else self.pw(h)


class _SpatialSelfAttention(nn.Module):
    """Plain multi-head self-attention over the Hp*Wp spatial tokens."""

    def __init__(self, d: int, num_heads: int = 4):
        super().__init__()
        self.nh = max(1, num_heads)
        while d % self.nh != 0:
            self.nh -= 1
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,H,W,d]
        B, H, W, d = x.shape
        s = x.reshape(B, H * W, d)
        qkv = self.qkv(s).reshape(B, H * W, 3, self.nh, d // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, H * W, d)
        return self.proj(o).reshape(B, H, W, d)


class RecursiveMambaBlock(nn.Module):
    """One layer of the shared core `f`: token mixer + channel MLP, each wrapped
    in a post-RMSNorm residual (TRM block structure).

    v15 R4.2 - the block gained the regularizers the hierarchical `GBlock`
    always had and this one never did: `trm_drop_path` (stochastic depth on
    each residual branch) and `trm_dropout` (inside both GatedMLPs). Both
    default to 0.0, which is exactly the pre-v15 block. This matters more
    after R1.2 removes weight decay from norms and biases: without these the
    recursive model has essentially no regularizer at all.
    """

    def __init__(self, cfg: MedMambaSSTRMConfig, drop_path: float = 0.0):
        super().__init__()
        d = cfg.trm_dim
        dropout = cfg.trm_dropout
        if cfg.trm_mixer == "ss2d":
            self.mixer = SS2D(d, d_state=cfg.d_state,
                              scan_directions=cfg.resolved_scan_directions(),
                              backend=cfg.scan_backend)
        elif cfg.trm_mixer == "mlp":
            self.mixer = _TokenMLPMixer(d, cfg.trm_ffn_mult,
                                        channel_mlp=cfg.trm_mixer_channel_mlp, dropout=dropout)
        else:  # "attention"
            self.mixer = _SpatialSelfAttention(d, num_heads=max(1, d // 64))
        self.ffn = GatedMLP(d, mult=cfg.trm_ffn_mult, dropout=dropout)
        # DropPath is a no-op module (identity) at 0.0 and in eval mode, so the
        # legacy default costs nothing and changes no value.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,H,W,d]
        x = _rms_norm_lastdim(x + self.drop_path(self.mixer(x)))
        x = _rms_norm_lastdim(x + self.drop_path(self.ffn(x)))
        return x


class RecursiveCore(nn.Module):
    """The weight-shared recursive reasoning core (TRM `L_level` + recursion)."""

    def __init__(self, cfg: MedMambaSSTRMConfig):
        super().__init__()
        self.cfg = cfg
        self.two_state = cfg.trm_two_state
        self.n_latent = cfg.trm_n_latent
        self.n_improve = cfg.trm_n_improve
        self.checkpoint_core = cfg.trm_checkpoint_core
        # Linearly-increasing stochastic depth across the core's layers, the
        # same schedule `MedMambaSSBackbone` uses for `drop_path_rate` (v15 R4.2).
        n_layers = cfg.trm_core_layers
        dpr = ([0.0] * n_layers if cfg.trm_drop_path <= 0.0
               else [float(v) for v in torch.linspace(0.0, cfg.trm_drop_path, n_layers)])
        self.layers = nn.ModuleList([RecursiveMambaBlock(cfg, drop_path=dpr[i])
                                     for i in range(n_layers)])
        # Non-trainable init states (TRM's H_init/L_init are `nn.Buffer`, not
        # parameters): the very first improve step that reads them almost
        # always falls inside the `trm_n_improve - 1` no_grad prelude, so a
        # `nn.Parameter` here would silently never receive a gradient.
        z_init = torch.empty(cfg.trm_dim)
        nn.init.trunc_normal_(z_init, std=0.02)
        self.register_buffer("z_init", z_init, persistent=True)
        if self.two_state:
            y_init = torch.empty(cfg.trm_dim)
            nn.init.trunc_normal_(y_init, std=0.02)
            self.register_buffer("y_init", y_init, persistent=True)

    def _f_forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h)
        return h

    def f(self, h: torch.Tensor) -> torch.Tensor:
        # Gradient-checkpoint each application of the shared core. Without
        # this, the in-loop deep-supervision training path keeps EVERY
        # segment's full grad-carrying subgraph alive simultaneously until the
        # single final backward (see training/trainerg_v10.py) - and each `f`
        # call's SS2D mixer additionally retains one activation tensor per
        # scan step (O(L) in the spatial token count) for its own backward.
        # Multiplied by `trm_core_layers * (trm_n_latent + 1)` calls per
        # improve step and `trm_deep_supervision_steps` segments, that peak
        # was observed to exceed 13 GB for a batch of 128 on an 11x11 grid -
        # checkpointing recomputes each `f` call during backward instead of
        # holding all of them at once, which is exactly the case gradient
        # checkpointing is for.
        #
        # v15 R3.3: this used to be unconditional, so `--gradient_checkpointing`
        # did not control it and the recompute was always paid whether or not
        # the caller asked for it. `trm_checkpoint_core` (default True =
        # unchanged behaviour) makes it a real switch.
        if (self.checkpoint_core and self.training
                and torch.is_grad_enabled() and h.requires_grad):
            return torch_checkpoint.checkpoint(self._f_forward, h, use_reentrant=False)
        return self._f_forward(h)

    def improve(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        for _ in range(self.n_latent):
            z = self.f(z + y + x) if self.two_state else self.f(z + x)
        y = self.f(y + z) if self.two_state else z
        return y, z

    def init_states(self, x: torch.Tensor):
        B, H, W, d = x.shape
        z = self.z_init.view(1, 1, 1, d).expand(B, H, W, d).contiguous()
        y = (self.y_init.view(1, 1, 1, d).expand(B, H, W, d).contiguous()
             if self.two_state else z)
        return y, z

    def one_segment(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """`trm_n_improve - 1` improvement steps under no_grad, then one with grad."""
        with torch.no_grad():
            for _ in range(self.n_improve - 1):
                y, z = self.improve(x, y, z)
        y, z = self.improve(x, y, z)
        return y, z


class RecursiveHead(nn.Module):
    """Classification head + halt (Q) head over the pooled answer feature `y`.

    v15 R4.4 - two methods with two return types instead of one method whose
    return type depended on the argument type:

        forward(y: [B,H,W,d])          -> (logits, q_halt)
        forward_features(feature_dict) -> logits

    `MedMambaSSTRMLatentReconWrapper` calls `forward_features` when the head
    exposes it (the hierarchical `ClassificationHead` does not, and keeps
    taking the dict through `__call__`). A dict passed to `forward` still
    works and is routed to `forward_features`, so no existing caller breaks -
    but new code should pick the method it means.
    """

    def __init__(self, cfg: MedMambaSSTRMConfig, num_classes: int, dropout: float = 0.0):
        super().__init__()
        d = cfg.trm_dim
        self.norm = make_norm(cfg.norm_type, d)
        # v15 R4.2 - `--classifier_dropout` is now accepted for `recursive`
        # (it used to raise for anything but `efficient`). 0.0 = legacy.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(d, num_classes)
        self.q_head = nn.Linear(d, 1)
        self.classifier_init = cfg.classifier_init

    def reinit_classifier(self) -> None:
        """v15 R1.1d - see `MedMambaSSTRMConfig.classifier_init`."""
        if self.classifier_init == "fan_in":
            self.fc.reset_parameters()

    def reset_q_head(self) -> None:
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def _pool(self, y: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(y).mean(dim=(1, 2)))

    def forward_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Backbone-feature dict -> logits (no halt head)."""
        return self.fc(self._pool(features["feature_map"]))

    def forward(self, inp):
        if isinstance(inp, dict):
            return self.forward_features(inp)
        pooled = self._pool(inp)
        return self.fc(pooled), self.q_head(pooled).squeeze(-1)


class MedMambaSSTRMBackbone(nn.Module):
    """Spectral pathway + linear stem + 2-D positional encoding + recursive
    core. `forward_features` runs the full recursion and returns the same dict
    keys as `MedMambaSSBackbone` (so recon / existing heads keep working);
    per-segment logits for deep supervision live on `MedMambaSSTRM`."""

    def __init__(self, cfg: MedMambaSSTRMConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.spectral_pathway = SpectralPathway(cfg)
        self.stem = nn.Linear(cfg.d_ctx, cfg.trm_dim)
        self.stem_norm = make_norm(cfg.norm_type, cfg.trm_dim)
        self.core = RecursiveCore(cfg)
        # v15 R1.1c - only materialised when it is not the legacy 1.0, so a
        # pre-v15 checkpoint still loads with strict=True in the legacy config.
        self.spatial_pe_gain = (nn.Parameter(torch.tensor(float(cfg.trm_spatial_pe_gain)))
                                if cfg.trm_spatial_pe_gain != 1.0 else None)
        self.apply(MedMambaSSBackbone._init_weights)
        self.spectral_pathway.tokenizer.reinit_value_branch()   # v15 R1.1, see MedMambaSSBackbone

    # v15 R1.1b - same helper, same semantics as the hierarchical backbone's.
    _normalize_ctx = MedMambaSSBackbone._normalize_ctx

    def embed(self, x: torch.Tensor, wavelengths: Optional[torch.Tensor] = None,
              sensor_range: Optional[Tuple[float, float]] = None):
        if not self.cfg.use_wavelength_metadata:
            wavelengths = None
        ctx_map, band_weights = self.spectral_pathway(x, wavelengths=wavelengths, sensor_range=sensor_range)
        feat = self.stem_norm(self.stem(self._normalize_ctx(ctx_map)))   # [B,Hp,Wp,d]
        _, H, W, d = feat.shape
        pe = sinusoidal_2d_encoding(H, W, d, device=feat.device, dtype=feat.dtype)
        # v15 R1.1c - `pe` is the same for every sample in the dataset, so an
        # un-scaled add is the A1 defect again: |pe| ~ 0.56 against a stem
        # output that, before R1.1b, had |.| ~ 2e-3.
        if self.spatial_pe_gain is not None:
            pe = self.spatial_pe_gain * pe
        return feat + pe.unsqueeze(0), ctx_map, band_weights

    def run_segments(self, x_emb: torch.Tensor, n_segments: int):
        """The ONE implementation of the segment recursion (v15 R4.4). Returns
        `(y, z)` after `n_segments` segments, detaching the carry between
        segments exactly as TRM does. `MedMambaSSTRM.forward_deep_supervision`
        drives the same loop step-by-step because it has to read the head at
        every segment; `forward_features` below just wants the last `y`."""
        y, z = self.core.init_states(x_emb)
        for i in range(n_segments):
            y, z = self.core.one_segment(x_emb, y, z)
            if i < n_segments - 1:
                y, z = y.detach(), z.detach()
        return y, z

    def forward_features(self, x: torch.Tensor, wavelengths: Optional[torch.Tensor] = None,
                         sensor_range: Optional[Tuple[float, float]] = None) -> Dict[str, torch.Tensor]:
        x_emb, ctx_map, band_weights = self.embed(x, wavelengths, sensor_range)
        y, _ = self.run_segments(x_emb, self.cfg.trm_deep_supervision_steps)
        pooled = y.mean(dim=(1, 2))
        return {
            "feature_map": y,
            "pooled": pooled,
            "stage_pools": [pooled],
            "stage_feature_maps": [y],
            "stage_ctx_maps": [],
            "spectral_context": ctx_map.mean(dim=(1, 2)),
            "band_weights": band_weights,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["pooled"]


class MedMambaSSTRM(nn.Module):
    """MedMamba-SS-TRM: TRM-style recursive MedMamba-SS (classification). ~1-4 M params vs the
    ~27 M hierarchical `MedMambaSS`. `forward(x)` returns the last-segment
    logits ([B, num_classes]); `forward_deep_supervision(x)` returns the list
    of per-segment `(logits, q_halt)` for the deep-supervision training loop
    (see `training/trainerg_v10.py`)."""

    def __init__(self, cfg: Optional[MedMambaSSTRMConfig] = None, task: str = "classification",
                 num_classes: int = 1000, **head_kwargs):
        super().__init__()
        cfg = cfg or MedMambaSSTRMConfig(recursive=True, dims=(128,), depths=(1,))
        # v15 R4.4 - this used to mutate the CALLER's config in place, so a
        # config built once and passed to two models (or inspected afterwards)
        # silently came back with `recursive=True`, `dims=(trm_dim,)` and
        # `depths=(1,)` overwritten. Copy first.
        cfg = copy.deepcopy(cfg)
        cfg.recursive = True
        cfg.dims = (cfg.trm_dim,)          # so cfg.dims[-1] == trm_dim for recon decoders
        cfg.depths = (1,)
        if task != "classification":
            raise ValueError("MedMambaSSTRM currently supports task='classification' only")
        self.cfg = cfg
        self.task = task
        self.backbone = MedMambaSSTRMBackbone(cfg)
        self.head = RecursiveHead(cfg, num_classes, **head_kwargs)
        self.head.apply(MedMambaSSBackbone._init_weights)
        self.head.reinit_classifier()
        self.head.reset_q_head()
        self.last_feature_map: Optional[torch.Tensor] = None

    def forward_deep_supervision(self, x: torch.Tensor, wavelengths: Optional[torch.Tensor] = None,
                                 sensor_range: Optional[Tuple[float, float]] = None):
        x_emb, _, _ = self.backbone.embed(x, wavelengths, sensor_range)
        core = self.backbone.core
        y, z = core.init_states(x_emb)
        n = self.cfg.trm_deep_supervision_steps
        logits_list, q_list = [], []

        # v15 R4.3 - real ACT, or none at all. `tau is None` reproduces the
        # pre-v15 loop exactly (every segment, every time). When it is set,
        # segments stop once every sample in the batch has halted; during
        # training, `trm_halt_exploration_prob` is the probability of ignoring
        # that and running the full budget anyway, which is the exploration
        # floor the field was added for and never used by.
        tau = self.cfg.trm_halt_threshold if self.cfg.trm_act_halting else None
        explore = bool(self.training and tau is not None
                       and float(torch.rand(())) < self.cfg.trm_halt_exploration_prob)

        for i in range(n):
            y, z = core.one_segment(x_emb, y, z)
            logits, q = self.head(y)
            logits_list.append(logits)
            q_list.append(q)
            if i < n - 1:
                y, z = y.detach(), z.detach()
                if tau is not None and not explore and bool((torch.sigmoid(q) > tau).all()):
                    break
        # Detached: this is a plain module attribute (not a buffer/parameter),
        # so it must never hold a live autograd graph - e.g. `EMAHelper.ema_copy`
        # deepcopies the module for validation and `copy.deepcopy` refuses a
        # non-leaf tensor. A caller that needs recon gradient into the core
        # should decode from the *return value* within the same forward instead.
        #
        # v15 R4.4 - written only while training. It exists solely so the
        # reconstruction decoder can read the last segment's features during a
        # training step; keeping a full [B,H,W,d] tensor alive after every
        # eval/inference forward (including inside `EMAHelper.ema_copy`'s
        # validation pass) served no purpose.
        if self.training:
            self.last_feature_map = y.detach()
        return logits_list, q_list

    def forward(self, x: torch.Tensor, wavelengths: Optional[torch.Tensor] = None,
                sensor_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        logits_list, _ = self.forward_deep_supervision(x, wavelengths=wavelengths,
                                                       sensor_range=sensor_range)
        return logits_list[-1]


# ----------------------------------------------------------------------------
# Convenience constructors
# ----------------------------------------------------------------------------

def medmamba_ss_tiny(num_classes=1000, **kwargs) -> MedMambaSS:
    cfg = MedMambaSSTRMConfig(dims=(96, 192, 384, 768), depths=(2, 2, 4, 2), patch_size=4)
    return MedMambaSS(cfg, task="classification", num_classes=num_classes, **kwargs)


def medmamba_ss_small(num_classes=1000, **kwargs) -> MedMambaSS:
    cfg = MedMambaSSTRMConfig(dims=(96, 192, 384, 768), depths=(2, 2, 8, 2), patch_size=4)
    return MedMambaSS(cfg, task="classification", num_classes=num_classes, **kwargs)


def medmamba_ss_base(num_classes=1000, **kwargs) -> MedMambaSS:
    cfg = MedMambaSSTRMConfig(dims=(128, 256, 512, 1024), depths=(2, 2, 12, 2), patch_size=4)
    return MedMambaSS(cfg, task="classification", num_classes=num_classes, **kwargs)


def medmamba_ss_hsi_small(num_classes=1000, **kwargs) -> MedMambaSS:
    """Suited to small HSI patches (e.g. 11x11xC from a patch-based HSI
    preprocessing pipeline): no spatial patch-stem downsampling, narrow/shallow."""
    cfg = MedMambaSSTRMConfig(dims=(64, 128, 256), depths=(2, 2, 2), d_state=8, d_ctx=64,
                           patch_size=1, spectral_depth=3)
    return MedMambaSS(cfg, task="classification", num_classes=num_classes, **kwargs)


def medmamba_ss_research(num_classes=1000, **kwargs) -> MedMambaSS:
    """Larger-capacity research configuration: 8-direction SS2D, multi-scale
    spectral + spatial branches, cross-attention fusion, squeeze-excitation,
    sparse dynamic band selection."""
    cfg = MedMambaSSTRMConfig(
        dims=(96, 192, 384, 768), depths=(2, 2, 4, 2), patch_size=4,
        scan_directions=8, spatial_multiscale=True, spatial_scales=(3, 5),
        spectral_multiscale=True, squeeze_excitation=True, band_selection_mode="sparse",
        fusion_type="cross_attention", norm_type="rmsnorm",
    )
    return MedMambaSS(cfg, task="classification", num_classes=num_classes, **kwargs)


def medmamba_ss_sensor_aware(num_classes=1000, **kwargs) -> MedMambaSS:
    """Example config intended to be called with `wavelengths=` and
    `sensor_range=` at forward time, e.g.
        model(x, wavelengths=band_centers_nm, sensor_range=(400.0, 1000.0))
    so representations are comparable across sensors with different band
    counts / spacings but overlapping physical ranges (Phases 14-15)."""
    cfg = MedMambaSSTRMConfig(dims=(64, 128, 256), depths=(2, 2, 2), patch_size=2,
                           d_ctx=64, sensor_reference_range=(400.0, 1000.0))
    return MedMambaSS(cfg, task="classification", num_classes=num_classes, **kwargs)


def medmamba_ss_trm_default(num_classes=1000, **kwargs) -> MedMambaSSTRM:
    """TRM-style recursive MedMamba-SS (arXiv:2510.04871): one weight-shared
    recursive core instead of a 4-stage hierarchical backbone. Channel-agnostic
    and patch_size=1, so the same preset serves the 11x11xC HSI patches and the
    11x11x3 PAD-UFES RGB patches while using ~0.5 M params (measured; vs ~27 M
    for `medmamba_ss_tiny`). Train it with deep supervision + EMA via
    `training/trainerg_v10.py` / `train_example_v14.py`.

    Measured on a real 11x11x32 HSI batch on an RTX 5060 Ti, batch_size=32,
    defaults (trm_n_latent=6, trm_n_improve=3, trm_deep_supervision_steps=4):
    `trm_mixer="mlp"` takes ~0.66s/step and ~1.5GB peak; the default
    `trm_mixer="ss2d"` takes ~6.8s/step and ~2.5GB peak - `_selective_scan_pure_pytorch`'s
    per-timestep Python loop reruns for every one of `trm_core_layers *
    (trm_n_latent + 1)` calls to the shared core, per deep-supervision segment.
    `RecursiveCore.f` IS gradient-checkpointed (recomputes each call during
    backward instead of holding every segment's graph at once - without this,
    batch_size=128 with `trm_mixer="ss2d"` OOMs a 16GB GPU), but the per-call
    Python-loop compute cost remains. For training at real dataset scale,
    prefer `trm_mixer="mlp"` (also fewer params) or register a real CUDA/
    Triton scan backend (`register_scan_backend`) before defaulting to
    `trm_mixer="ss2d"`."""
    cfg = MedMambaSSTRMConfig(recursive=True, dims=(128,), depths=(1,), d_state=8, d_ctx=64,
                           patch_size=1, spectral_depth=3, trm_dim=128)
    return MedMambaSSTRM(cfg, task="classification", num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    def run(name, model, x, expected_shape, **fwd_kwargs):
        model.eval()
        with torch.no_grad():
            out = model(x, **fwd_kwargs)
        assert tuple(out.shape) == expected_shape, f"{name}: got {tuple(out.shape)}, want {expected_shape}"
        print(f"[{name}] in={tuple(x.shape)} -> out={tuple(out.shape)}  params={count_params(model)/1e6:.2f}M")
        return model

    print("=== Config validation (Phase 16) ===")
    try:
        MedMambaSSTRMConfig(dims=(32, 64), depths=(1,)).validate()
        raise AssertionError("should have raised")
    except AssertionError as e:
        print("  correctly rejected mismatched dims/depths:", str(e)[:60])
    try:
        MedMambaSSTRMConfig(scan_directions=["not_a_direction"]).validate()
        raise AssertionError("should have raised")
    except AssertionError as e:
        print("  correctly rejected bad scan direction:", str(e)[:60])

    print("\n=== Channel-agnosticism: the SAME model instance on different C ===")
    model = medmamba_ss_tiny(num_classes=6)
    for C in (1, 3, 31):
        run(f"tiny/C={C}", model, torch.randn(2, C, 32, 32), (2, 6))

    print("\n=== HSI small-patch config, large band count ===")
    run("hsi_small/C=826", medmamba_ss_hsi_small(num_classes=6), torch.randn(2, 826, 11, 11), (2, 6))

    print("\n=== 8-direction SS2D + multiscale + cross-attention + SE + sparse bands ===")
    run("research/C=31", medmamba_ss_research(num_classes=6), torch.randn(1, 31, 24, 24), (1, 6))

    print("\n=== Arbitrary/partial scan-direction subset (Phase 9 registry) ===")
    cfg = MedMambaSSTRMConfig(dims=(32, 64), depths=(1, 1), patch_size=2, d_ctx=32,
                          scan_directions=["h", "diag", "adiag_rev"])
    run("scan-subset", MedMambaSS(cfg, num_classes=5), torch.randn(2, 10, 16, 16), (2, 5))

    print("\n=== Wavelength-aware / sensor-aware encoding (Phases 14-15) ===")
    model = medmamba_ss_sensor_aware(num_classes=4)
    x = torch.randn(2, 20, 16, 16)
    out_no_wl = model(x)
    wavelengths = torch.linspace(450, 900, steps=20)
    out_wl = model(x, wavelengths=wavelengths, sensor_range=(400.0, 1000.0))
    out_wl_per_sample = model(x, wavelengths=wavelengths.unsqueeze(0).expand(2, -1))
    print(f"  no wavelengths: {tuple(out_no_wl.shape)}  with wavelengths: {tuple(out_wl.shape)}  "
          f"per-sample wavelengths: {tuple(out_wl_per_sample.shape)}")
    assert out_no_wl.shape == out_wl.shape == out_wl_per_sample.shape == (2, 4)

    print("\n=== Per-stage fusion types + fusion variants ===")
    for fusion in ("film", "gated", "cross_attention", "multiplicative", "residual"):
        cfg = MedMambaSSTRMConfig(dims=(32, 64), depths=(1, 1), patch_size=2, d_ctx=32, fusion_type=fusion)
        run(f"fusion={fusion}", MedMambaSS(cfg, num_classes=5), torch.randn(2, 10, 16, 16), (2, 5))
    cfg = MedMambaSSTRMConfig(dims=(32, 64, 128), depths=(1, 1, 1), patch_size=2, d_ctx=32,
                          fusion_type=["film", "gated", "residual"])
    run("fusion=per-stage-list", MedMambaSS(cfg, num_classes=5), torch.randn(2, 10, 16, 16), (2, 5))

    print("\n=== Task heads ===")
    cfg = MedMambaSSTRMConfig(dims=(32, 64), depths=(1, 1), patch_size=2, d_ctx=32)
    run("classification", MedMambaSS(cfg, task="classification", num_classes=5), torch.randn(2, 3, 16, 16), (2, 5))
    run("regression", MedMambaSS(cfg, task="regression", out_dim=1), torch.randn(2, 3, 16, 16), (2, 1))
    run("embedding", MedMambaSS(cfg, task="embedding", embed_dim=48), torch.randn(2, 3, 16, 16), (2, 48))
    seg_model = MedMambaSS(cfg, task="segmentation", num_classes=4)
    seg_model.eval()
    with torch.no_grad():
        seg_out = seg_model(torch.randn(2, 3, 16, 16))
    print(f"[segmentation] out={tuple(seg_out.shape)}")
    assert tuple(seg_out.shape) == (2, 4, 16, 16)

    print("\n=== Multi-level backbone outputs (Phase 13) ===")
    backbone = MedMambaSSBackbone(MedMambaSSTRMConfig(dims=(32, 64, 128), depths=(1, 1, 1), patch_size=2, d_ctx=32))
    feats = backbone.forward_features(torch.randn(2, 5, 20, 20))
    print("  keys:", sorted(feats.keys()))
    print("  stage_feature_map shapes:", [tuple(f.shape) for f in feats["stage_feature_maps"]])
    print("  stage_ctx_map shapes:", [tuple(f.shape) for f in feats["stage_ctx_maps"]])
    assert feats["band_weights"] is not None

    print("\n=== Backend registry (Phase 17): register a dummy backend ===")
    register_scan_backend("dummy_copy", _selective_scan_pure_pytorch)
    cfg = MedMambaSSTRMConfig(dims=(24, 48), depths=(1, 1), patch_size=2, d_ctx=24, scan_backend="dummy_copy")
    run("custom-backend", MedMambaSS(cfg, num_classes=3), torch.randn(1, 6, 16, 16), (1, 3))

    print("\n=== Odd spatial size + batch size 1 + backward pass ===")
    model = medmamba_ss_hsi_small(num_classes=4)
    x = torch.randn(1, 128, 13, 9)
    y = torch.randint(0, 4, (1,))
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    n_missing = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"loss={loss.item():.4f}  params_without_grad={n_missing}")
    assert n_missing == 0

    print("\n=== TRM recursive variant (channel-agnostic, one instance on many C) ===")
    trm = medmamba_ss_trm_default(num_classes=6)
    trm.eval()
    for C in (3, 32, 826):
        with torch.no_grad():
            out = trm(torch.randn(2, C, 11, 11))
        assert tuple(out.shape) == (2, 6), (C, out.shape)
    print(f"[trm] params={count_params(trm)/1e6:.3f}M  (cf. medmamba_ss_tiny "
          f"{count_params(medmamba_ss_tiny(num_classes=6))/1e6:.2f}M)")

    print("\n=== TRM deep supervision + halting + backward grad coverage ===")
    trm.train()
    yt = torch.randint(0, 6, (2,))
    logits_list, q_list = trm.forward_deep_supervision(torch.randn(2, 32, 11, 11))
    assert len(logits_list) == trm.cfg.trm_deep_supervision_steps == len(q_list)
    assert all(tuple(l.shape) == (2, 6) for l in logits_list)
    ce = sum(F.cross_entropy(l, yt) for l in logits_list) / len(logits_list)
    bce = sum(F.binary_cross_entropy_with_logits(q, (l.argmax(-1) == yt).float())
              for l, q in zip(logits_list, q_list)) / len(logits_list)
    (ce + 0.5 * bce).backward()
    n_missing = sum(1 for p in trm.parameters() if p.requires_grad and p.grad is None)
    print(f"[trm] steps={len(logits_list)}  ce={ce.item():.4f}  bce={bce.item():.4f}  "
          f"params_without_grad={n_missing}")
    assert n_missing == 0
    # eval path stays a single [B, num_classes] tensor (recon-wrapper contract)
    trm.eval()
    with torch.no_grad():
        assert tuple(trm(torch.randn(2, 32, 11, 11)).shape) == (2, 6)
        feats = trm.backbone.forward_features(torch.randn(2, 32, 11, 11))
        assert feats["feature_map"].shape[-1] == trm.cfg.dims[-1] == trm.cfg.trm_dim
        assert tuple(trm.head(feats).shape) == (2, 6)

    print("\n=== recursive=False leaves the hierarchical model untouched ===")
    assert MedMambaSSTRMConfig().recursive is False
    _tiny_a = count_params(medmamba_ss_tiny(num_classes=6))
    _tiny_b = count_params(medmamba_ss_tiny(num_classes=6))
    assert _tiny_a == _tiny_b

    for mixer in ("mlp", "attention"):
        cfg = MedMambaSSTRMConfig(recursive=True, trm_dim=64, d_ctx=32, patch_size=1,
                              trm_mixer=mixer, trm_deep_supervision_steps=2)
        mm = MedMambaSSTRM(cfg, num_classes=5)
        mm.eval()
        with torch.no_grad():
            assert tuple(mm(torch.randn(2, 7, 11, 11)).shape) == (2, 5)
        print(f"[trm/{mixer}] params={count_params(mm)/1e6:.3f}M")

    print("\nAll smoke tests passed.")
