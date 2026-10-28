# -*- coding: utf-8 -*-
"""
hsi_augment/models/diffusion.py
===============================
Class-conditional latent diffusion, arm A3 (Plan §Phase 7, #24): the
denoiser, its training loss and the DDIM / classifier-free-guidance sampler,
in plain torch (`diffusers` is not installed).

    unet = LatentUNet(latent_channels=4, latent_size=6, num_classes=3)
    loss = DiffusionLoss(unet.schedule, min_snr_gamma=5.0, label_dropout=0.1)   # Trainer step_fn
    x = sample(ae, unet, codec, class_id=1, n=10_000, w=1.5, seed=s)            # (n, 11, 11, 32) float16

Parameterisation
----------------
* Cosine schedule (Nichol & Dhariwal 2021), T = 1000, s = 0.008, built from
  betas clipped at 0.999 in float64: alpha_bar_t strictly decreasing in
  (0, 1). Index t = 0..T-1 is diffusion step t + 1. alpha_t = sqrt(alpha_bar_t),
  sigma_t = sqrt(1 - alpha_bar_t), so alpha^2 + sigma^2 = 1.
* v-prediction (Salimans & Ho 2022): z_t = alpha z_0 + sigma eps,
  v = alpha eps - sigma z_0, and back, z_0 = alpha z_t - sigma v,
  eps = sigma z_t + alpha v (`v_target`, `x0_from_v`, `eps_from_v`).
  v stays unit-scale at both ends of the schedule, where an eps- or
  x0-target blows up; with a cosine schedule that reaches SNR ~ 0 this matters.
* Min-SNR-gamma weighting (Hang et al. 2023), gamma = 5: the per-sample v-MSE
  is weighted min(SNR_t, gamma) / (SNR_t + 1), the v-space form of
  min(SNR, gamma) on the x0 error. Low-noise steps, which are easy and would
  otherwise dominate the gradient, are capped.
* Classifier-free guidance (Ho & Salimans 2022): one network, conditioned on
  a class embedding with K + 1 entries, the last a NULL token. Training
  replaces the label by the null token with probability 0.1 (label dropout),
  so the same network is also the unconditional model. Sampling uses
  v = v_null + w (v_class - v_null): w = 0 is the unconditional model,
  w = 1 the plain conditional one, w > 1 guidance. The candidates are
  w in {1, 1.5, 2, 3}; w > 3 pulls samples toward the training modes
  (Plan §Phase 7), so stage configs refuse it (`MAX_GUIDANCE`).

Why a small UNet and not an MLP-ResNet over the 144 values
----------------------------------------------------------
The issue allows either. The latent is the 4 x 6 x 6 output of a
convolutional encoder, so each latent cell describes one 2 x 2 block of
pixels and the 36 cells share their statistics. A conv UNet uses that: a 3x3
kernel is learned once and applied at every cell, so every training latent
is 36 examples of the same local statistics for it, which matters with five
DCIS patients to learn from. It is also equivariant to the flips and 90 deg
rotations that leave a patch's class unchanged (the classifier's own
augmentation), where an MLP would have to learn each position and
orientation separately from data. The 6x6 -> 3x3 level gives every output
cell a receptive field over the whole latent, so nothing global is lost. It
is built from #22's blocks (GroupNorm + SiLU ResBlocks with FiLM, stride-2
conv down, deterministic ConvTranspose up), so it has no BatchNorm and its
CUDA backward is deterministic. An MLP-ResNet is the fallback for a
non-spatial latent, which this autoencoder does not have.

    in conv3x3 4->128 (6x6) | ResBlock(128) | Downsample 128->256 (3x3) | ResBlock(256)
    | mid ResBlock(256) | ResBlock(256+256 -> 256) | Upsample 256->128 (6x6, cropped to L)
    | ResBlock(128+128 -> 128) | GN, SiLU, conv3x3 128->4 (zero-init)

Every ResBlock is FiLM-conditioned on SiLU(time MLP(sinusoidal(t)) + class
embedding(y)). With the defaults (channels 128/256, one block per level, one
mid block, 512-d embedding) it has ~7.3M parameters, inside the issue's 6-8M.

Sampling: DDIM, deterministic
-----------------------------
DDIM (Song et al. 2021) with 50 "trailing" steps, t = 999, 979, ..., 19
(the noisiest step is always visited; Lin et al. 2024), eta = 0 by default.
The initial noise of row r comes from `train_generative.noise_rows(seed, r,
...)`, a CPU generator stream addressed by row, so the same (seed, start, n)
gets the same noise on any device and with any batch size, and an export can
be generated in pieces. With eta = 0 the update has no other randomness;
eta > 0 draws the per-step noise the same way, keyed by step. The same
arguments, batch size and device give the same bytes. A different batch
size or device gives the same rows only up to float rounding (convolution
kernels may sum in another order: ~1e-6 per network evaluation on CPU,
accumulated over the DDIM steps), enough to flip some float16 values. So a
sampling sha256 is recorded with its batch size and device.
The same seed gives the same initial noise for every class and every w, on
purpose: comparing guidance scales on common random numbers isolates the
effect of w. Callers that want independent streams per class pass distinct
seeds (e.g. `derive_seed(seed, class_id)`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hsi_augment.models.blocks import (Downsample, ResBlock, TimestepEmbedder, Upsample, crop_to,
                                       group_norm)
from hsi_augment.train_generative import derive_seed, noise_rows

EXPORT_FORMAT = "hsi_augment.diffusion/1"
DEFAULT_TIMESTEPS = 1000
COSINE_S = 0.008
MAX_BETA = 0.999
DEFAULT_GUIDANCE = (1.0, 1.5, 2.0, 3.0)
MAX_GUIDANCE = 3.0
DEFAULT_SAMPLE_STEPS = 50


# ============================================================================
# schedule and the v-parameterisation
# ============================================================================

class CosineSchedule:
    """alpha_bar_t of the cosine schedule for t = 0..T-1 (float64 numpy),
    plus float32 torch lookups. Index -1 means "no noise" (alpha_bar = 1),
    the target of DDIM's last step."""

    def __init__(self, timesteps: int = DEFAULT_TIMESTEPS, s: float = COSINE_S, max_beta: float = MAX_BETA):
        if int(timesteps) < 2:
            raise ValueError("timesteps must be >= 2")
        self.timesteps, self.s, self.max_beta = int(timesteps), float(s), float(max_beta)
        f = lambda u: math.cos((u + self.s) / (1.0 + self.s) * math.pi / 2.0) ** 2
        betas = np.array([min(1.0 - f((i + 1) / self.timesteps) / f(i / self.timesteps), self.max_beta)
                          for i in range(self.timesteps)], dtype=np.float64)
        self.betas = betas
        self.alpha_bar = np.cumprod(1.0 - betas)
        self._torch: Dict[str, torch.Tensor] = {}

    def describe(self) -> Dict[str, Any]:
        return {"kind": "cosine", "timesteps": self.timesteps, "s": self.s, "max_beta": self.max_beta}

    def _table(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._torch:
            ab = np.concatenate([self.alpha_bar, [1.0]])       # [-1] -> alpha_bar 1 (no noise)
            self._torch[key] = torch.as_tensor(ab, dtype=torch.float64, device=device)
        return self._torch[key]

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(alpha_t, sigma_t), float32, the shape of `t` (int indices; -1 = no noise)."""
        ab = self._table(t.device)[t.long()]
        return ab.sqrt().float(), (1.0 - ab).clamp_min(0.0).sqrt().float()

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        ab = self._table(t.device)[t.long()]
        return (ab / (1.0 - ab)).float()

    def ddim_timesteps(self, steps: int) -> np.ndarray:
        """`steps` "trailing" timesteps, descending from T - 1: 999, 979, ..., 19."""
        steps = int(steps)
        if not 1 <= steps <= self.timesteps:
            raise ValueError(f"steps must be in [1, {self.timesteps}]")
        ts = np.round(np.arange(self.timesteps, 0, -self.timesteps / steps)).astype(np.int64) - 1
        if ts.size != steps or np.any(np.diff(ts) >= 0) or ts[-1] < 0:
            raise ValueError(f"{steps} DDIM steps do not give distinct timesteps of {self.timesteps}")
        return ts


def _bcast(a: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return a.reshape(a.shape + (1,) * (like.ndim - a.ndim))


def v_target(x0: torch.Tensor, eps: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """v = alpha eps - sigma x0 (alpha, sigma per sample, broadcast)."""
    return _bcast(alpha, x0) * eps - _bcast(sigma, x0) * x0


def x0_from_v(z_t: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """x0 = alpha z_t - sigma v (exact when alpha^2 + sigma^2 = 1)."""
    return _bcast(alpha, z_t) * z_t - _bcast(sigma, z_t) * v


def eps_from_v(z_t: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """eps = sigma z_t + alpha v."""
    return _bcast(sigma, z_t) * z_t + _bcast(alpha, z_t) * v


def min_snr_weight(snr: torch.Tensor, gamma: Optional[float]) -> torch.Tensor:
    """Min-SNR-gamma weight of the v-MSE: min(SNR, gamma) / (SNR + 1).
    gamma None -> 1 (the plain v-MSE)."""
    if gamma is None:
        return torch.ones_like(snr)
    return torch.clamp(snr, max=float(gamma)) / (snr + 1.0)


# ============================================================================
# the denoiser
# ============================================================================

class LatentUNet(nn.Module):
    """v-prediction UNet over (latent_channels, L, L) latents, conditioned on
    the timestep index and a class id in 0..K (K = `num_classes` is the null
    token). `forward(z_t, t, y) -> v_hat`. See the module docstring."""

    def __init__(self, latent_channels: int = 4, latent_size: int = 6, channels: Sequence[int] = (128, 256),
                 blocks_per_level: int = 1, mid_blocks: int = 1, num_classes: int = 3, emb_dim: int = 512,
                 freq_dim: int = 256, dropout: float = 0.0, timesteps: int = DEFAULT_TIMESTEPS,
                 cosine_s: float = COSINE_S):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if len(channels) < 1 or min(channels) <= 0:
            raise ValueError("channels must be one or more positive widths (one per resolution level)")
        if blocks_per_level < 1 or mid_blocks < 0 or num_classes < 1 or latent_size < 1:
            raise ValueError("need blocks_per_level >= 1, mid_blocks >= 0, num_classes >= 1, latent_size >= 1")
        self.kwargs: Dict[str, Any] = {
            "latent_channels": int(latent_channels), "latent_size": int(latent_size), "channels": list(channels),
            "blocks_per_level": int(blocks_per_level), "mid_blocks": int(mid_blocks),
            "num_classes": int(num_classes), "emb_dim": int(emb_dim), "freq_dim": int(freq_dim),
            "dropout": float(dropout), "timesteps": int(timesteps), "cosine_s": float(cosine_s)}
        self.latent_channels, self.latent_size = int(latent_channels), int(latent_size)
        self.num_classes = int(num_classes)
        self.null_class = self.num_classes
        self.schedule = CosineSchedule(timesteps, cosine_s)

        self.time_embed = TimestepEmbedder(emb_dim, freq_dim=freq_dim)
        self.class_embed = nn.Embedding(self.num_classes + 1, emb_dim)
        self.inp = nn.Conv2d(latent_channels, channels[0], kernel_size=3, padding=1)

        block = lambda i, o: ResBlock(i, o, cond_dim=emb_dim, dropout=dropout)
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skips = []
        for lvl, ch in enumerate(channels):
            prev = channels[lvl - 1] if lvl else channels[0]
            if lvl:
                self.downsample.append(Downsample(prev, ch))
            self.down.append(nn.ModuleList([block(ch, ch) for _ in range(blocks_per_level)]))
            skips += [ch] * blocks_per_level
        self.mid = nn.ModuleList([block(channels[-1], channels[-1]) for _ in range(mid_blocks)])
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for lvl in reversed(range(len(channels))):
            ch = channels[lvl]
            self.up.append(nn.ModuleList([block(ch + skips.pop(), ch) for _ in range(blocks_per_level)]))
            if lvl:
                self.upsample.append(Upsample(ch, channels[lvl - 1]))
        self.out_norm = group_norm(channels[0])
        self.out = nn.Conv2d(channels[0], latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)       # v_hat = 0 at init: the loss starts at E[v^2]
        nn.init.zeros_(self.out.bias)

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self.latent_channels, self.latent_size, self.latent_size)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z_t.ndim != 4 or tuple(z_t.shape[1:]) != self.latent_shape:
            raise ValueError(f"expected latents (B, {self.latent_shape}), got {tuple(z_t.shape)}")
        # Only the dtype is checked here: a value check would force a host sync
        # every forward. Ids are range-checked where they enter (pool loader,
        # sample()), and nn.Embedding refuses an out-of-range id anyway.
        if y.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"class ids must be an int tensor in [0, {self.null_class}] "
                             f"({self.null_class} = null), got {y.dtype}")
        cond = F.silu(self.time_embed(t.float()) + self.class_embed(y.long()))
        h = self.inp(z_t)
        sizes, skips = [], []
        for lvl, blocks in enumerate(self.down):
            if lvl:
                sizes.append(h.shape[-1])
                h = self.downsample[lvl - 1](h)
            for blk in blocks:
                h = blk(h, cond)
                skips.append(h)
        for blk in self.mid:
            h = blk(h, cond)
        for i, blocks in enumerate(self.up):
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], dim=1), cond)
            if i < len(self.upsample):
                h = crop_to(self.upsample[i](h), sizes.pop())    # 3 -> 6 (or 2 -> 4 -> 3 on odd sizes)
        return self.out(F.silu(self.out_norm(h)))

    def export_state(self, **meta: Any) -> Dict[str, Any]:
        """A `torch.load(weights_only=True)`-safe dict: kwargs, CPU weights, schedule, meta."""
        return {"format": EXPORT_FORMAT, "kwargs": dict(self.kwargs),
                "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                "schedule": self.schedule.describe(), "null_class": int(self.null_class), "meta": dict(meta)}

    @classmethod
    def from_export(cls, state: Mapping[str, Any], *, device: Any = None) -> "LatentUNet":
        if state.get("format") != EXPORT_FORMAT:
            raise ValueError(f"not a {EXPORT_FORMAT} export (format {state.get('format')!r})")
        model = cls(**state["kwargs"])
        model.load_state_dict(state["state_dict"])
        return model.to(device or "cpu").eval()


# ============================================================================
# training loss
# ============================================================================

class DiffusionLoss:
    """`step_fn` for `train_generative.Trainer`: Min-SNR-gamma weighted
    v-MSE at a uniformly drawn timestep, with label dropout to the null
    token while training (`info.training`; never in evaluation).

    t, the noise and the dropout mask are drawn on the CPU from torch's GLOBAL
    generator (checkpointed by the trainer) and then moved, so a seed gives the
    same draws on any device. The loss is computed in float32 even under bf16
    autocast. Batch: {"x": scaled latents (B, C, L, L), "y": class ids}."""

    def __init__(self, schedule: CosineSchedule, *, min_snr_gamma: Optional[float] = 5.0,
                 label_dropout: float = 0.1):
        if not 0.0 <= label_dropout <= 1.0:
            raise ValueError("label_dropout must be in [0, 1]")
        self.schedule, self.gamma, self.label_dropout = schedule, min_snr_gamma, float(label_dropout)

    def __call__(self, model: LatentUNet, batch: Mapping[str, torch.Tensor], info: Any
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z0, y = batch["x"].float(), batch["y"].long()
        dev, B = z0.device, z0.shape[0]
        t = torch.randint(0, self.schedule.timesteps, (B,)).to(dev)
        eps = torch.randn(tuple(z0.shape)).to(dev)
        if bool(getattr(info, "training", True)) and self.label_dropout > 0:
            drop = (torch.rand(B) < self.label_dropout).to(dev)
            y = torch.where(drop, torch.full_like(y, model.null_class), y)
        alpha, sigma = self.schedule.alpha_sigma(t)
        z_t = _bcast(alpha, z0) * z0 + _bcast(sigma, z0) * eps
        v = v_target(z0, eps, alpha, sigma)
        v_hat = model(z_t, t, y).float()
        per_sample = (v_hat - v).pow(2).flatten(1).mean(dim=1)
        weight = min_snr_weight(self.schedule.snr(t), self.gamma)
        loss = (weight * per_sample).mean()
        with torch.no_grad():
            x0_err = (x0_from_v(z_t, v_hat, alpha, sigma) - z0).pow(2).mean()
        return loss, {"v_mse": per_sample.mean().detach(), "x0_mse": x0_err}


# ============================================================================
# DDIM sampling with classifier-free guidance
# ============================================================================

def guided_v(model: LatentUNet, z_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, w: float) -> torch.Tensor:
    """v_null + w (v_y - v_null). w = 1 is the conditional pass alone and
    w = 0 the null pass alone (one forward instead of two, same value)."""
    w = float(w)
    if w == 1.0:
        return model(z_t, t, y).float()
    v_null = model(z_t, t, torch.full_like(y, model.null_class)).float()
    if w == 0.0:
        return v_null
    return v_null + w * (model(z_t, t, y).float() - v_null)


@torch.no_grad()
def ddim_sample(model: LatentUNet, z_T: torch.Tensor, y: torch.Tensor, w: float, *,
                steps: int = DEFAULT_SAMPLE_STEPS, eta: float = 0.0,
                step_noise: Optional[Any] = None) -> torch.Tensor:
    """DDIM from the noise `z_T` (B, C, L, L) to z_0, guided with scale `w`,
    `y` the (B,) class ids (the null token for an unconditional chain).
    `step_noise(i) -> tensor like z_T` supplies the noise of step i when
    eta > 0 (unused at eta = 0, which is deterministic)."""
    if eta < 0:
        raise ValueError("eta must be >= 0")
    if eta > 0 and step_noise is None:
        raise ValueError("eta > 0 needs step_noise")
    sched = model.schedule
    ts = sched.ddim_timesteps(steps)
    prevs = np.append(ts[1:], -1)
    z = z_T.float()
    dev = z.device
    for i, (t_i, p_i) in enumerate(zip(ts.tolist(), prevs.tolist())):
        t = torch.full((z.shape[0],), t_i, dtype=torch.long, device=dev)
        a_t, s_t = sched.alpha_sigma(t)
        a_p, s_p = sched.alpha_sigma(torch.full_like(t, p_i))
        v = guided_v(model, z, t, y, w)
        x0 = x0_from_v(z, v, a_t, s_t)
        eps = eps_from_v(z, v, a_t, s_t)
        if eta > 0 and p_i >= 0:
            ab_t, ab_p = a_t.double() ** 2, a_p.double() ** 2
            sig = (eta * torch.sqrt((1 - ab_p) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_p)).float()
            dir_coef = torch.sqrt((s_p.double() ** 2 - sig.double() ** 2).clamp_min(0.0)).float()
            z = _bcast(a_p, x0) * x0 + _bcast(dir_coef, eps) * eps + _bcast(sig, z) * step_noise(i).to(dev)
        else:
            z = _bcast(a_p, x0) * x0 + _bcast(s_p, eps) * eps
    return z


def _check_request(model: LatentUNet, class_id: int, n: int, w: float, start: int) -> None:
    if not 0 <= int(class_id) < model.num_classes:
        raise ValueError(f"class_id must be in [0, {model.num_classes})")
    if int(n) < 0 or int(start) < 0:
        raise ValueError("n and start must be >= 0")
    if not (math.isfinite(float(w)) and float(w) >= 0):
        raise ValueError("guidance scale w must be a finite number >= 0")


def _resolve_device(model: nn.Module, device: Any) -> torch.device:
    if device is not None:
        return torch.device(device)
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


@torch.no_grad()
def sample_latents(denoiser: LatentUNet, class_id: int, n: int, w: float, seed: int, start: int = 0,
                   steps: int = DEFAULT_SAMPLE_STEPS, *, eta: float = 0.0, batch_size: int = 1024,
                   device: Any = None) -> torch.Tensor:
    """Rows [start, start + n) of class `class_id` as SCALED latents,
    (n, C, L, L) float32 on the CPU. Deterministic for the same arguments
    (batch_size included) on the same device; the initial noise of each row
    does not depend on `batch_size` or `start` (module docstring)."""
    _check_request(denoiser, class_id, n, w, start)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    dev = _resolve_device(denoiser, device)
    was_training = denoiser.training
    denoiser.eval()
    shape = denoiser.latent_shape
    out = torch.empty((int(n), *shape), dtype=torch.float32)
    try:
        for a in range(0, int(n), int(batch_size)):
            b = min(int(n), a + int(batch_size))
            z_T = noise_rows(seed, start + a, b - a, shape, device=dev)
            y = torch.full((b - a,), int(class_id), dtype=torch.long, device=dev)
            step_noise = (lambda i, a=a, b=b: noise_rows(derive_seed(seed, "ddim-eta", i), start + a, b - a,
                                                         shape)) if eta > 0 else None
            out[a:b] = ddim_sample(denoiser, z_T, y, w, steps=steps, eta=eta, step_noise=step_noise).cpu()
    finally:
        denoiser.train(was_training)
    return out


@torch.no_grad()
def sample(ae: nn.Module, denoiser: LatentUNet, codec: Any, class_id: int, n: int, w: float, seed: int,
           start: int = 0, steps: int = DEFAULT_SAMPLE_STEPS, *, eta: float = 0.0, batch_size: int = 1024,
           device: Any = None) -> np.ndarray:
    """Generate rows [start, start + n) of class `class_id` with guidance
    scale `w`: DDIM in the latent space, `ae.decode_scaled`, then
    `codec.decode_np` -> VALUE-space float16 NHWC (n, patch, patch, bands),
    clipped to [0, clip_hi] (the build's storage format, ready to export).

    Deterministic: the same arguments (batch_size included) on the same
    device give the same bytes. The noise of row r is `noise_rows(seed, r)`
    whatever the chunking, so other batch sizes give the same rows up to
    float rounding (module docstring). `ae` and `denoiser` are used in eval
    mode (pass the EMA weights, e.g. from `load_generator`)."""
    _check_request(denoiser, class_id, n, w, start)
    if tuple(getattr(ae, "latent_shape", ())) != denoiser.latent_shape:
        raise ValueError(f"autoencoder latent {getattr(ae, 'latent_shape', None)} != denoiser latent "
                         f"{denoiser.latent_shape}")
    dev = _resolve_device(denoiser, device)
    was_training = ae.training
    ae.eval()
    out = np.empty((int(n), ae.patch, ae.patch, ae.in_channels), dtype=np.float16)
    try:
        for a in range(0, int(n), int(batch_size)):
            b = min(int(n), a + int(batch_size))
            z = sample_latents(denoiser, class_id, b - a, w, seed, start + a, steps, eta=eta,
                               batch_size=b - a, device=dev)
            y = ae.decode_scaled(z.to(dev)).float()
            out[a:b] = codec.decode_np(y.permute(0, 2, 3, 1).cpu().numpy())
    finally:
        ae.train(was_training)
    return out
