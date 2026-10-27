# -*- coding: utf-8 -*-
"""
hsi_augment/codec.py
====================
The log-space representation every generator trains in (Plan §Phase 6,
"Representation"; P2.3b):

    encode:  y = (log(x + eps) - mu_b) / sigma_b                  eps = 1e-3
    decode:  x = clip(exp(y * sigma_b + mu_b) - eps, 0, clip_hi_b) -> float16

Why log space
-------------
Transmitted intensity follows Beer-Lambert, I = G exp(-d A(lambda)): a capture's
gain G and the stain density d are MULTIPLICATIVE in linear space and ADDITIVE
in log space. A generator that works on log values models illumination and
stain variation as shifts it can learn, and its decoded output goes through
exp(.), so it is positive by construction. `eps` keeps log finite on the
darkest pixels (values are on the build's value scale, median ~1, so 1e-3 is
0.1% of a typical value). `mu_b`, `sigma_b` standardise each band, so a
Gaussian likelihood or a diffusion schedule sees unit-scale inputs.

Why the decode clip
-------------------
`exp` turns a generator's occasional large output into an enormous intensity,
which would overflow float16 (max 65504) or dominate any statistic computed on
the export. Decoded values are clipped per band at `clip_hi` (1.1 x the train
p99.99 of that band, fitted elsewhere and passed in) and at 0 (`exp(.) - eps`
can dip just below 0 for the darkest outputs), then cast to float16, the
build's storage dtype. `clip_hi` must be below the float16 maximum, so the
cast is always finite.

Fitting the statistics
----------------------
`LogSpaceCodec.fit_stats(chunks)` streams linear-space chunks (e.g.
`io.iter_chunks` over the train TISSUE rows; choosing the rows is the
caller's job) through log(x + eps) into `io.RunningBandStats`: float64,
Chan et al. pairwise merge, so the result does not depend on the chunking.
The pool stage (#16) calls it, saves `stats.to_dict()`, and builds the codec
with `LogSpaceCodec.from_stats(stats, clip_hi)`.

Layout
------
numpy arrays follow the build: channel-last (N, H, W, C), `band_axis=-1`.
torch tensors follow the models: NCHW (B, C, H, W), `channel_dim=1`. Both are
parameters, so either layout works with either backend.

SAM on decoded values
---------------------
The spectral angle is only meaningful on non-negative intensities, so it is
computed on DECODED values, never on z-scored log values. Both metrics here
are imported, not reimplemented: `training.spectral_recon_metrics_v2.sam_deg`
(a float, degrees; refuses meaningfully negative input) and the frozen
`training.gan.SAMLoss` (differentiable, radians). Both expect NCHW:
they flatten with `permute(0, 2, 3, 1).reshape(-1, C)`, i.e. one spectrum per
(b, h, w) pixel along dim 1. An NHWC tensor would silently compute angles
along the width axis instead; permute first. `sam_deg_decoded` and
`make_decoded_sam_loss` below decode and then call them.

Both clamp cos(angle) at 1 - 1e-3 (`SAMLoss(angle_eps=1e-3)` by default), so
identical spectra score acos(0.999) = 2.56 deg, not 0, and `SAMLoss` passes
NO gradient to a pixel whose angle is already below 2.56 deg. A SAM loss term
meant to act on well-reconstructed spectra needs a smaller angle_eps
(`make_decoded_sam_loss(codec, angle_eps=1e-6)`; the frozen module explains
why 1e-8 is unsafe), and SAM degrees reported from `sam_deg` carry the 2.56
deg floor (it compresses ratios such as held-out / train SAM).

torch is imported lazily, so the pool stage can use the numpy half of this
module without paying for a torch import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from hsi_augment.io import STATS_BLOCK_VALUES, RunningBandStats

if TYPE_CHECKING:  # pragma: no cover
    import torch

LOG_EPS = 1e-3
FLOAT16_MAX = float(np.finfo(np.float16).max)    # 65504
CLIP_HI_FACTOR = 1.1                             # clip_hi = 1.1 x train p99.99 per band


def _band_shape(ndim: int, axis: int) -> tuple:
    """Broadcast shape that puts a (C,) vector on `axis` of an ndim array."""
    axis = axis % ndim
    return tuple(-1 if i == axis else 1 for i in range(ndim))


def _as_vector(v: Any, name: str, n: Optional[int] = None) -> np.ndarray:
    a = np.array(v, dtype=np.float64).reshape(-1)          # a writable copy, never a view
    if n is not None and a.shape[0] != n:
        raise ValueError(f"{name} has {a.shape[0]} bands, expected {n}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


class LogSpaceCodec:
    """Per-band log-space standardisation and its clipped inverse.

        codec = LogSpaceCodec.from_stats(LogSpaceCodec.fit_stats(chunks), clip_hi)
        y = codec.encode_np(x_nhwc)                  # float32 log-z
        x = codec.decode_np(y)                       # float16, in [0, clip_hi]
        x_t = codec.decode_torch(y_nchw)             # float32 tensor, differentiable

    `mu`, `sigma`: mean and std (ddof 0) of log(x + eps) per band, float64.
    `clip_hi`: per-band upper clip in LINEAR space (a scalar is broadcast)."""

    def __init__(self, mu: Sequence[float], sigma: Sequence[float],
                 clip_hi: Union[float, Sequence[float]], *, eps: float = LOG_EPS):
        self.mu = _as_vector(mu, "mu")
        self.n_bands = int(self.mu.shape[0])
        self.sigma = _as_vector(sigma, "sigma", self.n_bands)
        if np.any(self.sigma <= 0):
            raise ValueError("sigma must be > 0 in every band (a constant band cannot be standardised)")
        hi = np.asarray(clip_hi, dtype=np.float64).reshape(-1)
        self.clip_hi = _as_vector(np.broadcast_to(hi, (self.n_bands,)) if hi.size == 1 else hi,
                                  "clip_hi", self.n_bands)
        if np.any(self.clip_hi <= 0) or np.any(self.clip_hi >= FLOAT16_MAX):
            raise ValueError(f"clip_hi must be in (0, {FLOAT16_MAX:g}) so the float16 cast stays finite")
        self.eps = float(eps)
        if not self.eps > 0:
            raise ValueError("eps must be > 0")
        self._torch_cache: Dict[Any, tuple] = {}

    # -- fitting -----------------------------------------------------------------
    @staticmethod
    def fit_stats(chunks: Iterable[Any], *, n_bands: Optional[int] = None, eps: float = LOG_EPS,
                  band_axis: int = -1) -> RunningBandStats:
        """Float64 per-band stats of log(max(x, 0) + eps) over an iterable of
        linear-space chunks (arrays, or `io.Chunk`s, whose `.data` is used).
        Each chunk is widened to float64 in blocks of `io.STATS_BLOCK_VALUES`
        values, so a 20k-row float16 chunk never materialises as float64."""
        stats: Optional[RunningBandStats] = None if n_bands is None else RunningBandStats(n_bands)
        for ch in chunks:
            data = np.asarray(getattr(ch, "data", ch))
            if band_axis not in (-1, data.ndim - 1):
                data = np.moveaxis(data, band_axis, -1)
            if stats is None:
                stats = RunningBandStats(int(data.shape[-1]))
            flat = data.reshape(-1, stats.n_bands)
            step = max(1, STATS_BLOCK_VALUES // stats.n_bands)
            for a in range(0, flat.shape[0], step):
                b = flat[a:a + step].astype(np.float64)
                # Non-finite values stay non-finite (log(NaN) = NaN), so
                # RunningBandStats counts them instead of hiding them.
                with np.errstate(invalid="ignore"):
                    stats.update(np.log(np.maximum(b, 0.0) + eps))
        if stats is None:
            raise ValueError("fit_stats: no chunks (and no n_bands given)")
        return stats

    @classmethod
    def from_stats(cls, stats: Union[RunningBandStats, Mapping[str, Any]],
                   clip_hi: Union[float, Sequence[float]], *, eps: float = LOG_EPS) -> "LogSpaceCodec":
        """From the log-space stats `fit_stats` returned (or their `to_dict()`)."""
        if isinstance(stats, Mapping):
            stats = RunningBandStats.from_dict(stats)
        if np.any(stats.count <= 0):
            raise ValueError("stats have a band with no finite values")
        return cls(stats.mean, stats.std(), clip_hi, eps=eps)

    @staticmethod
    def clip_hi_from_quantile(p9999: Union[float, Sequence[float]],
                              factor: float = CLIP_HI_FACTOR) -> np.ndarray:
        """`factor` x the train p99.99 per band, kept below the float16 max."""
        hi = float(factor) * np.asarray(p9999, dtype=np.float64)
        return np.minimum(hi, np.nextafter(FLOAT16_MAX, 0.0))

    # -- (de)serialisation --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-able; floats round-trip exactly through `json`."""
        return {"kind": "log_space_codec", "eps": self.eps, "n_bands": self.n_bands,
                "mu": self.mu.tolist(), "sigma": self.sigma.tolist(), "clip_hi": self.clip_hi.tolist()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LogSpaceCodec":
        return cls(d["mu"], d["sigma"], d["clip_hi"], eps=float(d.get("eps", LOG_EPS)))

    def __repr__(self) -> str:
        return f"LogSpaceCodec(n_bands={self.n_bands}, eps={self.eps:g})"

    # -- numpy ----------------------------------------------------------------------
    def _check_bands(self, shape: tuple, axis: int) -> None:
        if not shape or shape[axis] != self.n_bands:
            raise ValueError(f"axis {axis} of shape {shape} is not the {self.n_bands}-band axis")

    def encode_np(self, x: np.ndarray, *, band_axis: int = -1, dtype: Any = np.float32) -> np.ndarray:
        """Linear -> log-z. Negative inputs are treated as 0 (the codec's domain
        is [0, clip_hi]). Computed in float32: its rounding (~1e-7) is far
        below the float16 the pool stores; peak RAM is ~3 float32 copies of x."""
        x = np.asarray(x)
        self._check_bands(x.shape, band_axis)
        shape = _band_shape(x.ndim, band_axis)
        mu = self.mu.astype(np.float32).reshape(shape)
        sigma = self.sigma.astype(np.float32).reshape(shape)
        y = np.maximum(x.astype(np.float32, copy=False), np.float32(0.0))
        y = np.log(y + np.float32(self.eps))
        y -= mu
        y /= sigma
        return y.astype(dtype, copy=False)

    def decode_np(self, y: np.ndarray, *, band_axis: int = -1, dtype: Any = np.float16) -> np.ndarray:
        """Log-z -> linear, clipped to [0, clip_hi], cast to `dtype` (float16).
        NaN input raises: a generator that emits NaN is broken, and a NaN must
        not reach an export. +-inf is fine (clipped)."""
        y = np.asarray(y)
        self._check_bands(y.shape, band_axis)
        if np.isnan(y).any():
            raise ValueError("decode_np: input contains NaN")
        shape = _band_shape(y.ndim, band_axis)
        x = y.astype(np.float32, copy=True)
        x *= self.sigma.astype(np.float32).reshape(shape)
        x += self.mu.astype(np.float32).reshape(shape)
        with np.errstate(over="ignore"):
            np.exp(x, out=x)                          # overflow -> inf -> clipped below
        x -= np.float32(self.eps)
        np.clip(x, np.float32(0.0), self.clip_hi.astype(np.float32).reshape(shape), out=x)
        return x.astype(dtype, copy=False)

    # -- torch ------------------------------------------------------------------------
    def _torch_params(self, device: "torch.device", ndim: int, channel_dim: int) -> tuple:
        import torch
        key = (str(device), ndim, channel_dim % ndim)
        if key not in self._torch_cache:
            shape = _band_shape(ndim, channel_dim)
            as_t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device).reshape(shape)
            self._torch_cache[key] = (as_t(self.mu), as_t(self.sigma), as_t(self.clip_hi))
        return self._torch_cache[key]

    def encode_torch(self, x: "torch.Tensor", *, channel_dim: int = 1) -> "torch.Tensor":
        """Linear -> log-z on a tensor (float32), e.g. for real batches on the GPU."""
        import torch
        self._check_bands(tuple(x.shape), channel_dim)
        mu, sigma, _ = self._torch_params(x.device, x.ndim, channel_dim)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return (torch.log(x.float().clamp_min(0.0) + self.eps) - mu) / sigma

    def decode_torch(self, y: "torch.Tensor", *, channel_dim: int = 1, clip: bool = True,
                     dtype: Optional["torch.dtype"] = None) -> "torch.Tensor":
        """Log-z -> linear on a tensor; differentiable (the clip passes no
        gradient where it binds). Always computed in float32 with autocast off,
        so a bf16 training step cannot run exp(.) in bf16. `dtype=torch.float16`
        for exports; the default float32 is what a loss wants."""
        import torch
        self._check_bands(tuple(y.shape), channel_dim)
        mu, sigma, hi = self._torch_params(y.device, y.ndim, channel_dim)
        with torch.autocast(device_type=y.device.type, enabled=False):
            x = torch.exp(y.float() * sigma + mu) - self.eps
            if clip:
                x = torch.minimum(x.clamp_min(0.0), hi)
        return x if dtype is None else x.to(dtype)


# ============================================================================
# SAM on decoded values (imported metrics; NCHW)
# ============================================================================

def sam_deg_decoded(codec: LogSpaceCodec, y_true: "torch.Tensor", y_pred: "torch.Tensor") -> float:
    """Mean spectral angle in degrees between two NCHW log-z batches, on their
    decoded (non-negative) values, via `spectral_recon_metrics_v2.sam_deg`."""
    import torch
    from training.spectral_recon_metrics_v2 import sam_deg
    with torch.no_grad():
        return sam_deg(codec.decode_torch(y_true), codec.decode_torch(y_pred))


def make_decoded_sam_loss(codec: LogSpaceCodec, **sam_kwargs: Any) -> "torch.nn.Module":
    """A differentiable SAM loss (radians) on decoded values:
    `loss(y_true, y_pred)` with NCHW log-z tensors. Wraps the frozen
    `training.gan.SAMLoss` (imported, never edited)."""
    import torch
    from training.gan import SAMLoss

    class DecodedSAMLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sam = SAMLoss(**sam_kwargs)

        def forward(self, y_true: "torch.Tensor", y_pred: "torch.Tensor") -> "torch.Tensor":
            return self.sam(codec.decode_torch(y_true), codec.decode_torch(y_pred))

    return DecodedSAMLoss()
