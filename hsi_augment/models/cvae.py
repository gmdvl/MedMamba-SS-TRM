# -*- coding: utf-8 -*-
"""
hsi_augment/models/cvae.py
==========================
The conditional VAE of arm A2 (Plan §Phase 6, #23): the model, its loss (the
`step_fn` that `train_generative.Trainer` calls), the SAM-weight calibration
and `sample`, the generation entry point #29 uses.

Everything runs in the pool's representation: y = (log(x + 1e-3) - mu_b) /
sigma_b (`codec.LogSpaceCodec`), NCHW float32 inside the model, NHWC float16
value space out of `sample`.

Architecture (~3.9M parameters at d = 64, ~3.7M at d = 32; plan "~3.5M")
-----------------------------------------------------------------------
    encoder  conv3x3 32->96 (11x11) -> ResBlock(96) -> Downsample 96->192 (6x6)
             -> ResBlock(192) -> Downsample 192->256 (3x3) -> GN, SiLU -> Linear -> (mu, log var)
    decoder  Linear d -> 256x3x3, SiLU -> Upsample 256->192 (6x6) -> ResBlock(192)
             -> Upsample 192->96 (12x12) -> ResBlock(96) -> GN, SiLU -> conv3x3 96->32 -> crop 11
    class    nn.Embedding(3, cond_dim), FiLM in every ResBlock of both halves
    sigma    one learned log sigma per band (the sigma-VAE's), floored at `min_log_sigma`

* Both halves see the class: q(z | x, y) and p(x | z, y), the standard CVAE
  (Sohn et al. 2015). The latent then carries what the class does not
  (patient, stain, texture), which is what sampling varies.
* GroupNorm only (`assert_no_batchnorm` is in the tests): BatchNorm couples
  the rows of a class-balanced batch and differs between train and eval mode.
* ConvTranspose upsampling, never bilinear: deterministic backward on CUDA.
* Any square patch works (the test fixture's is 5x5): P -> ceil(P/2) ->
  ceil(ceil(P/2)/2) =: s, then 2s -> 4s >= P, centre-cropped to P.

Loss (`CVAELoss`, per element of a patch: loss = rec + lambda SAM + beta KL_fb / (C H W))
-----------------------------------------------------------------------------------------
* rec: Gaussian negative log-likelihood with a learned per-band sigma
  (sigma-VAE, Rybkin et al. 2021), 0.5 ((x - mu) / sigma)^2 + log sigma +
  0.5 log 2 pi, averaged over the batch's elements. Learning sigma removes the
  hand-tuned reconstruction weight that decides how blurry a Gaussian VAE is.
* SAM: the frozen `training.gan.SAMLoss` on DECODED (linear, non-negative)
  values through `codec.make_decoded_sam_loss(codec, angle_eps=1e-6)`; the
  default angle_eps of 1e-3 would give a 2.56 deg floor with no gradient
  under it (#22's finding). lambda = r x rec_init / SAM_init, r in {0, 0.1}:
  "SAM about 10% of the reconstruction term at init" (`calibrate_sam_weight`;
  rec_init is the data-dependent part, i.e. without the 0.5 log 2 pi constant,
  which carries no gradient).
* KL: KL(q(z|x,y) || N(0, I)) per latent dim, averaged over the batch, each
  dim floored at `free_bits` = 0.05 nat (free bits, Kingma et al. 2016: a dim
  under the floor is not pushed further towards the prior, which keeps units
  from collapsing early); beta rises linearly 0 -> 1 over the first 10% of
  the run's steps (`beta_at`).
* Dividing the per-patch KL by C H W keeps the per-element scale, so the loss
  is O(1) and the gradient clip of 1.0 is meaningful, without changing the
  ratio of reconstruction to KL (the ELBO's).

Metrics the step returns (train and held-out): `nll`, the negative ELBO with
beta = 1 and no free bits, an upper bound on -log p(x) in nats per element
(the held-out metric that early-stops each LOPO fold); `rec_nll`; `kl` and
`kl_fb` (nats per patch); `sam` (radians); `mse` (log-z); `beta`; `sigma`.
The reconstruction term alone would be the wrong early-stopping metric: it is
lowest while beta is still small, i.e. for a model whose prior samples are poor.

Sampling (`sample`)
-------------------
z = tau_s * eps, eps ~ N(0, I) from `train_generative.noise_rows` (a CPU
generator keyed by (seed, class) and the row index), decoded, then mapped to
value space by `codec.decode_np` (exp, clipped to [0, clip_hi], float16; NaN
raises). The decoder always runs on whole 1024-row blocks aligned to absolute
row numbers, so row r's sample depends only on (weights, seed, class, tau_s,
r): neither on `n` / `start` nor on how a caller batches its requests, bitwise.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hsi_augment.models.blocks import Downsample, ResBlock, Upsample, crop_to, group_norm
from hsi_augment.train_generative import NOISE_BLOCK_ROWS, cpu_generator, derive_seed, load_weights, noise_rows

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.codec import LogSpaceCodec
    from hsi_augment.train_generative import StepInfo

PathLike = Union[str, os.PathLike]

MODEL_FORMAT = "hsi_augment.cvae/1"
MODEL_JSON = "model.json"
LOG_2PI = math.log(2.0 * math.pi)
LOGVAR_RANGE = (-30.0, 20.0)                 # keeps exp(log var) finite in fp32
GEN_BLOCK_ROWS = NOISE_BLOCK_ROWS            # `sample` decodes whole aligned blocks of this many rows
DEFAULT_WIDTHS = (96, 192, 256)


# ============================================================================
# the model
# ============================================================================

def _down(n: int) -> int:
    """Spatial size after a 3x3 stride-2 padding-1 conv: ceil(n / 2)."""
    return (int(n) + 1) // 2


class CVAE(nn.Module):
    """Conditional VAE on (B, C, P, P) log-z patches; see the module docstring.

        model = CVAE(in_bands=32, patch=11, n_classes=3, latent_dim=64)
        x_mu, mu, logvar = model(x, y)          # z sampled from q with the global torch RNG
        x_mu = model.decode(z, y)               # (B, C, P, P), the decoder mean
    """

    def __init__(self, in_bands: int = 32, patch: int = 11, n_classes: int = 3, latent_dim: int = 64,
                 widths: Sequence[int] = DEFAULT_WIDTHS, cond_dim: int = 64, dropout: float = 0.0,
                 min_log_sigma: float = -6.0):
        super().__init__()
        if len(widths) != 3 or min(widths) <= 0:
            raise ValueError(f"widths must be three positive channel counts, got {widths}")
        if min(in_bands, patch, n_classes, latent_dim, cond_dim) <= 0:
            raise ValueError("in_bands, patch, n_classes, latent_dim and cond_dim must be > 0")
        w0, w1, w2 = (int(w) for w in widths)
        self.in_bands, self.patch, self.n_classes = int(in_bands), int(patch), int(n_classes)
        self.latent_dim, self.widths, self.cond_dim = int(latent_dim), (w0, w1, w2), int(cond_dim)
        self.dropout, self.min_log_sigma = float(dropout), float(min_log_sigma)
        self.bottleneck = _down(_down(self.patch))           # 11 -> 6 -> 3
        flat = w2 * self.bottleneck ** 2

        self.cond = nn.Embedding(self.n_classes, self.cond_dim)
        # encoder q(z | x, y)
        self.enc_in = nn.Conv2d(self.in_bands, w0, kernel_size=3, padding=1)
        self.enc_block0 = ResBlock(w0, w0, cond_dim=self.cond_dim, dropout=self.dropout)
        self.enc_down1 = Downsample(w0, w1)
        self.enc_block1 = ResBlock(w1, w1, cond_dim=self.cond_dim, dropout=self.dropout)
        self.enc_down2 = Downsample(w1, w2)
        self.enc_norm = group_norm(w2)
        self.enc_head = nn.Linear(flat, 2 * self.latent_dim)
        # decoder p(x | z, y)
        self.dec_in = nn.Linear(self.latent_dim, flat)
        self.dec_up1 = Upsample(w2, w1)
        self.dec_block1 = ResBlock(w1, w1, cond_dim=self.cond_dim, dropout=self.dropout)
        self.dec_up0 = Upsample(w1, w0)
        self.dec_block0 = ResBlock(w0, w0, cond_dim=self.cond_dim, dropout=self.dropout)
        self.dec_norm = group_norm(w0)
        self.dec_out = nn.Conv2d(w0, self.in_bands, kernel_size=3, padding=1)
        # sigma-VAE: one learned log sigma per band (1-D, so the Trainer gives it no weight decay)
        self.log_sigma = nn.Parameter(torch.zeros(self.in_bands))

    def spec(self) -> Dict[str, Any]:
        """The constructor arguments (what `model.json` stores to rebuild it)."""
        return {"in_bands": self.in_bands, "patch": self.patch, "n_classes": self.n_classes,
                "latent_dim": self.latent_dim, "widths": list(self.widths), "cond_dim": self.cond_dim,
                "dropout": self.dropout, "min_log_sigma": self.min_log_sigma}

    def log_sigma_eff(self) -> torch.Tensor:
        """(C,) log sigma, floored at `min_log_sigma` (e^-6 ~ 0.0025 log-z
        units, about the float16 quantisation of the pool)."""
        return self.log_sigma.clamp_min(self.min_log_sigma)

    def _check(self, x: Optional[torch.Tensor], y: torch.Tensor) -> None:
        if x is not None and tuple(x.shape[1:]) != (self.in_bands, self.patch, self.patch):
            raise ValueError(f"expected (B, {self.in_bands}, {self.patch}, {self.patch}) NCHW, got "
                             f"{tuple(x.shape)}")
        if y.ndim != 1:
            raise ValueError(f"y must be (B,) class indices, got shape {tuple(y.shape)}")

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mu, log var), each (B, d)."""
        self._check(x, y)
        c = self.cond(y)
        h = self.enc_block0(self.enc_in(x), c)
        h = self.enc_block1(self.enc_down1(h), c)
        h = F.silu(self.enc_norm(self.enc_down2(h)))
        mu, logvar = self.enc_head(h.flatten(1)).chunk(2, dim=-1)
        return mu, logvar.clamp(*LOGVAR_RANGE)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The decoder mean, (B, C, P, P) log-z."""
        self._check(None, y)
        c = self.cond(y)
        h = F.silu(self.dec_in(z)).view(-1, self.widths[2], self.bottleneck, self.bottleneck)
        h = self.dec_block1(self.dec_up1(h), c)
        h = self.dec_block0(self.dec_up0(h), c)
        h = self.dec_out(F.silu(self.dec_norm(h)))
        return crop_to(h, self.patch)

    def forward(self, x: torch.Tensor, y: torch.Tensor, *, sample: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(x_mu, mu, log var). `sample=True` draws z = mu + sigma eps with the
        GLOBAL torch RNG (checkpointed by the Trainer); False uses z = mu."""
        mu, logvar = self.encode(x, y)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode(z, y), mu, logvar


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ============================================================================
# loss
# ============================================================================

def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Element-wise -log N(x; mean, sigma^2); `log_sigma` broadcasts (e.g. (1, C, 1, 1))."""
    return 0.5 * ((x - mean) * torch.exp(-log_sigma)) ** 2 + log_sigma + 0.5 * LOG_2PI


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, 1)) per row and latent dim, (B, d), nats."""
    return 0.5 * (mu ** 2 + torch.exp(logvar) - 1.0 - logvar)


def beta_at(step: int, max_steps: int, warmup_frac: float) -> float:
    """The KL weight before optimizer step `step` (0-based): linear 0 -> 1
    over the first `warmup_frac` of `max_steps`, then 1."""
    if warmup_frac <= 0:
        return 1.0
    return float(min(1.0, step / max(1e-12, warmup_frac * max_steps)))


class CVAELoss:
    """The CVAE's `step_fn(model, batch, info) -> (loss, metrics)` for
    `train_generative.Trainer` (also its held-out `eval_fn`). Batches are
    `ArrayBatchSource` dicts: "x" (B, C, P, P) log-z float32, "y" (B,) int64.
    The only randomness is the reparameterisation noise, from the global
    torch RNG (the Trainer checkpoints it and forks it for evaluation)."""

    def __init__(self, codec: "LogSpaceCodec", *, sam_weight: float = 0.0, free_bits: float = 0.05,
                 beta_warmup_frac: float = 0.1, sam_angle_eps: float = 1e-6):
        from hsi_augment.codec import make_decoded_sam_loss
        if sam_weight < 0 or free_bits < 0 or not 0 <= beta_warmup_frac <= 1:
            raise ValueError("need sam_weight >= 0, free_bits >= 0 and beta_warmup_frac in [0, 1]")
        self.codec = codec
        self.sam_weight = float(sam_weight)
        self.free_bits = float(free_bits)
        self.beta_warmup_frac = float(beta_warmup_frac)
        self.sam_loss = make_decoded_sam_loss(codec, angle_eps=float(sam_angle_eps))

    def __call__(self, model: CVAE, batch: Mapping[str, torch.Tensor],
                 info: "StepInfo") -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, y = batch["x"], batch["y"]
        x_mu, mu, logvar = model(x, y)
        ls = model.log_sigma_eff().view(1, -1, 1, 1)
        rec = gaussian_nll(x, x_mu, ls).mean()
        kl_dim = kl_per_dim(mu, logvar).mean(0)                       # (d,) batch mean, nats
        kl = kl_dim.sum()
        kl_fb = kl_dim.clamp_min(self.free_bits).sum()
        n_el = float(x[0].numel())
        beta = beta_at(info.step, info.max_steps, self.beta_warmup_frac)
        if self.sam_weight > 0:
            sam = self.sam_loss(x, x_mu)
        else:
            with torch.no_grad():                                     # logged, not trained on
                sam = self.sam_loss(x, x_mu)
        loss = rec + self.sam_weight * sam + beta * kl_fb / n_el
        metrics = {"nll": rec.detach() + kl.detach() / n_el, "rec_nll": rec.detach(), "kl": kl.detach(),
                   "kl_fb": kl_fb.detach(), "sam": sam.detach(),
                   "mse": ((x_mu.detach() - x) ** 2).mean(), "beta": beta,
                   "sigma": torch.exp(ls.detach()).mean()}
        return loss, metrics


@torch.no_grad()
def calibrate_sam_weight(model: CVAE, loss: CVAELoss, batches: Iterable[Mapping[str, torch.Tensor]],
                         ratio: float, *, seed: int) -> Dict[str, Any]:
    """lambda such that lambda x SAM_init = ratio x rec_init on `batches`, at
    the model's CURRENT (initial) weights; eval mode, z sampled from q with a
    CPU generator (`cpu_generator(seed, "sam-calibration")`), so the global
    RNGs are not touched and the result is a pure function of (init, data,
    seed). rec_init is the per-element reconstruction NLL without its constant
    0.5 log 2 pi (that part has no gradient; at init, sigma = 1, it is half the
    log-z MSE). Returns the record `calibration.json` stores; `lambda` is 0
    for ratio 0."""
    if ratio < 0:
        raise ValueError("ratio must be >= 0")
    device = next(model.parameters()).device
    gen = cpu_generator(seed, "sam-calibration")
    was = model.training
    model.eval()
    rec_sum = sam_sum = 0.0
    rows = n_batches = 0
    try:
        for b in batches:
            x, y = b["x"].to(device), b["y"].to(device)
            mu, logvar = model.encode(x, y)
            eps = torch.randn(tuple(mu.shape), generator=gen, dtype=torch.float32).to(device)
            x_mu = model.decode(mu + torch.exp(0.5 * logvar) * eps, y)
            ls = model.log_sigma_eff().view(1, -1, 1, 1)
            bsz = int(x.shape[0])
            rec_sum += float(gaussian_nll(x, x_mu, ls).mean()) * bsz
            sam_sum += float(loss.sam_loss(x, x_mu)) * bsz
            rows += bsz
            n_batches += 1
    finally:
        model.train(was)
    if rows == 0:
        raise ValueError("calibrate_sam_weight: no batches")
    rec_init = rec_sum / rows
    rec_data = rec_init - 0.5 * LOG_2PI
    sam_init = sam_sum / rows
    if ratio > 0 and not (rec_data > 0 and sam_init > 0):
        raise ValueError(f"cannot calibrate lambda: rec_init {rec_data:.4g}, SAM_init {sam_init:.4g} rad")
    lam = float(ratio * rec_data / sam_init) if ratio > 0 else 0.0
    return {"rule": "lambda * SAM_init = ratio * rec_init (rec_init without 0.5 log 2pi), at init",
            "ratio": float(ratio), "lambda": lam, "rec_init": rec_data, "rec_init_with_constant": rec_init,
            "sam_init_rad": sam_init, "sam_init_deg": math.degrees(sam_init),
            "sam_term_init": lam * sam_init, "batches": n_batches, "rows": rows, "seed": int(seed)}


# ============================================================================
# sampling, saving and loading
# ============================================================================

@torch.no_grad()
def sample(model: CVAE, codec: "LogSpaceCodec", class_id: int, n: int, tau_s: float, seed: int,
           start: int = 0) -> np.ndarray:
    """Rows [start, start + n) of class `class_id`'s sample stream, as
    value-space float16 NHWC (n, P, P, C): z = tau_s * eps, eps from
    `noise_rows(derive_seed(seed, "cvae-sample", class_id), ...)`, decoded by
    the model and then by `codec.decode_np` (clipped to [0, clip_hi]; a NaN
    raises). Deterministic and independent of how the rows are requested (see
    the module docstring). The same seed gives every class its own stream and
    every tau_s the same eps (common random numbers across tau_s)."""
    if n < 0 or start < 0:
        raise ValueError("n and start must be >= 0")
    if not 0 <= int(class_id) < model.n_classes:
        raise ValueError(f"class_id {class_id} is not in [0, {model.n_classes})")
    if not tau_s > 0:
        raise ValueError("tau_s must be > 0")
    P, C, K = model.patch, model.in_bands, GEN_BLOCK_ROWS
    out = np.empty((n, P, P, C), dtype=np.float16)
    if n == 0:
        return out
    device = next(model.parameters()).device
    noise_seed = derive_seed(int(seed), "cvae-sample", int(class_id))
    was = model.training
    model.eval()
    try:
        for b in range(start // K, (start + n - 1) // K + 1):
            eps = noise_rows(noise_seed, b * K, K, (model.latent_dim,))
            z = (eps * float(tau_s)).to(device)
            y = torch.full((K,), int(class_id), dtype=torch.long, device=device)
            x = model.decode(z, y).float().cpu()
            lo, hi = max(start, b * K) - b * K, min(start + n, (b + 1) * K) - b * K
            dst = b * K + lo - start
            out[dst:dst + hi - lo] = codec.decode_np(x[lo:hi].permute(0, 2, 3, 1).numpy(), band_axis=-1)
    finally:
        model.train(was)
    return out


def model_record(model: CVAE, codec: "LogSpaceCodec", **extra: Any) -> Dict[str, Any]:
    """What `model.json` holds: enough to rebuild the model and decode its output."""
    return {"format": MODEL_FORMAT, "model": model.spec(), "codec": codec.to_dict(), **extra}


def load_cvae(run_dir: PathLike, *, checkpoint: str = "best.pt", ema: bool = True,
              device: Any = "cpu") -> Tuple[CVAE, "LogSpaceCodec", Dict[str, Any]]:
    """(model in eval mode, codec, model.json) from a fit's run directory
    (`gen/cvae/final` or `gen/cvae/lopo/<cell>/fold-<k>`): the EMA weights of
    `checkpoint` by default. The checkpoint is read with weights_only=True."""
    from hsi_augment.codec import LogSpaceCodec
    d = Path(run_dir)
    with open(d / MODEL_JSON) as f:
        meta = json.load(f)
    if meta.get("format") != MODEL_FORMAT:
        raise ValueError(f"{d / MODEL_JSON} is not a {MODEL_FORMAT} record")
    model = CVAE(**meta["model"])
    load_weights(d / checkpoint, model, ema=ema)
    model.to(device).eval()
    return model, LogSpaceCodec.from_dict(meta["codec"]), meta
