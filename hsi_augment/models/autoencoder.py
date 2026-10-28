# -*- coding: utf-8 -*-
"""
hsi_augment/models/autoencoder.py
=================================
The unconditional, low-KL spatial autoencoder of arm A3 (Plan §Phase 7, #24).
The latent diffusion denoiser (`diffusion.py`) is trained in its latent space.

Why a separate autoencoder, and not the CVAE's latent
-----------------------------------------------------
The CVAE (#23, arm A2) decodes with the class as an input (FiLM), so its
encoder is rewarded for leaving the class OUT of z: the decoder gets it for
free. Its latent is class-stripped. A class-conditional diffusion model in
that latent would have nothing for classifier-free guidance to act on (the
class-conditional and unconditional latent distributions nearly coincide),
and A3 would end up about the same generator as A2. This autoencoder never
sees the class, so its latent has to carry everything, class signal
included, and the diffusion model is what carries the class information.
The two arms are then genuinely different generators.

Architecture (~3.9M parameters at the defaults; 11x11x32 patches)
-----------------------------------------------------------------
The CVAE's backbone (conv 32 -> 96 at 11x11, stride-2 to 192 at 6x6, GroupNorm
+ SiLU residual blocks, no BatchNorm; deterministic ConvTranspose up, crop
12 -> 11), stopped at the 6x6 level instead of going on to 3x3 and a vector:

    encoder  conv3x3 32->96 | ResBlock(96) x b | Downsample 96->192 (11->6)
             | ResBlock(192) x b | GN, SiLU, conv3x3 192->2*4        (mean, log-variance)
    decoder  conv3x3 4->192 | ResBlock(192) x b | Upsample 192->96 (6->12) | crop 12->11
             | ResBlock(96) x b | GN, SiLU, conv3x3 96->32           (log-z reconstruction)

The latent is SPATIAL, 4 x 6 x 6 = 144 values (11*11*32 / 144 = 27x
compression), so neighbouring latent cells describe neighbouring 2x2 pixel
blocks and the denoiser can be a small convolutional UNet.

Loss (`AELoss`)
---------------
    MSE(x_hat, x) in log-z  +  0.1 * SAM(decode(x), decode(x_hat))  +  1e-6 * KL

* MSE in the codec's log-z space, the space every generator trains in.
* SAM in radians on DECODED (linear, non-negative) values, through
  `codec.make_decoded_sam_loss(codec, angle_eps=1e-6)` (`make_sam_loss`).
  The frozen SAMLoss default (angle_eps 1e-3) clamps at 2.56 deg and passes
  no gradient below it, i.e. exactly where a good reconstruction lives
  (#22's finding). 1e-6 is fixed (`SAM_ANGLE_EPS`), not a flag: a SAM term
  with the 2.56 deg floor is not an option.
* KL of the diagonal-Gaussian posterior to N(0, I), in nats per patch (summed
  over the 144 latent elements, averaged over the batch). With weight 1e-6
  it does not make this a VAE; it only keeps the latent from drifting to an
  arbitrary scale or a degenerate zero-variance posterior (the "KL-reg"
  autoencoder of latent diffusion, Rombach et al. 2022).
* Training decodes a reparameterised sample z = mu + sigma * eps (eps from
  torch's global RNG, which the trainer checkpoints); evaluation, encoding
  and export use the posterior mean.

Latent scale
------------
After training, the posterior means of the fold's training rows are encoded
and `latent_scale = 1 / std` (over every latent element) is recorded, so the
diffusion model sees unit-variance latents (as in latent diffusion). It is a
plain attribute, not a parameter or buffer: it is not in the trainer's
checkpoints (which hold the weights during training, scale unknown), only in
the exported `ae.pt` (`export_state` / `from_export`), next to the weights it
belongs to. `encode_scaled` / `decode_scaled` apply it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hsi_augment.models.blocks import Downsample, ResBlock, Upsample, crop_to, group_norm

EXPORT_FORMAT = "hsi_augment.autoencoder/1"
LOGVAR_RANGE = (-30.0, 20.0)      # the latent diffusion autoencoder's clamp: exp() stays finite
SAM_ANGLE_EPS = 1e-6              # SAMLoss angle_eps; its 1e-3 default floors SAM at 2.56 deg


def latent_size(patch: int, levels: int = 1) -> int:
    """Spatial size after `levels` stride-2 convolutions: 11 -> 6 (-> 3)."""
    n = int(patch)
    for _ in range(int(levels)):
        n = -(-n // 2)
    return n


class SpatialAutoencoder(nn.Module):
    """Unconditional autoencoder with a (latent_channels, L, L) Gaussian
    posterior, L = ceil(patch / 2). Inputs and outputs are NCHW log-z
    tensors (B, in_channels, patch, patch). See the module docstring."""

    def __init__(self, in_channels: int = 32, patch: int = 11, widths: Sequence[int] = (96, 192),
                 blocks_per_level: int = 2, latent_channels: int = 4, dropout: float = 0.0):
        super().__init__()
        widths = tuple(int(w) for w in widths)
        if len(widths) != 2 or min(widths) <= 0:
            raise ValueError("widths must be two positive channel counts (full-res level, 1/2-res level)")
        if blocks_per_level < 1 or latent_channels < 1 or patch < 2 or in_channels < 1:
            raise ValueError("need blocks_per_level >= 1, latent_channels >= 1, patch >= 2, in_channels >= 1")
        self.kwargs: Dict[str, Any] = {"in_channels": int(in_channels), "patch": int(patch), "widths": list(widths),
                                       "blocks_per_level": int(blocks_per_level),
                                       "latent_channels": int(latent_channels), "dropout": float(dropout)}
        self.in_channels, self.patch, self.latent_channels = int(in_channels), int(patch), int(latent_channels)
        self.latent_hw = latent_size(self.patch, 1)
        w0, w1 = widths
        b = int(blocks_per_level)

        self.enc_in = nn.Conv2d(in_channels, w0, kernel_size=3, padding=1)
        self.enc_hi = nn.ModuleList([ResBlock(w0, dropout=dropout) for _ in range(b)])
        self.enc_down = Downsample(w0, w1)
        self.enc_lo = nn.ModuleList([ResBlock(w1, dropout=dropout) for _ in range(b)])
        self.enc_norm = group_norm(w1)
        self.enc_out = nn.Conv2d(w1, 2 * latent_channels, kernel_size=3, padding=1)

        self.dec_in = nn.Conv2d(latent_channels, w1, kernel_size=3, padding=1)
        self.dec_lo = nn.ModuleList([ResBlock(w1, dropout=dropout) for _ in range(b)])
        self.dec_up = Upsample(w1, w0)
        self.dec_hi = nn.ModuleList([ResBlock(w0, dropout=dropout) for _ in range(b)])
        self.dec_norm = group_norm(w0)
        self.dec_out = nn.Conv2d(w0, in_channels, kernel_size=3, padding=1)

        self.latent_scale: float = 1.0      # set after training; see the module docstring

    # -- shapes ---------------------------------------------------------------------
    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self.latent_channels, self.latent_hw, self.latent_hw)

    @property
    def latent_dim(self) -> int:
        c, h, w = self.latent_shape
        return c * h * w

    # -- encoder / decoder ---------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mean, log-variance) of the posterior, each (B, latent_channels, L, L)."""
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.in_channels, self.patch, self.patch):
            raise ValueError(f"expected (B, {self.in_channels}, {self.patch}, {self.patch}) log-z NCHW, "
                             f"got {tuple(x.shape)}")
        h = self.enc_in(x)
        for blk in self.enc_hi:
            h = blk(h)
        h = self.enc_down(h)
        for blk in self.enc_lo:
            h = blk(h)
        mean, logvar = self.enc_out(F.silu(self.enc_norm(h))).chunk(2, dim=1)
        return mean, logvar.clamp(*LOGVAR_RANGE)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_channels, L, L) unscaled latent -> (B, in_channels, patch, patch) log-z."""
        if z.ndim != 4 or tuple(z.shape[1:]) != self.latent_shape:
            raise ValueError(f"expected latents of shape (B, {self.latent_shape}), got {tuple(z.shape)}")
        h = self.dec_in(z)
        for blk in self.dec_lo:
            h = blk(h)
        h = crop_to(self.dec_up(h), self.patch)
        for blk in self.dec_hi:
            h = blk(h)
        return self.dec_out(F.silu(self.dec_norm(h)))

    def forward(self, x: torch.Tensor, *, sample: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(x_hat, mean, logvar, z). `sample` draws z = mean + sigma * eps from
        torch's global RNG (training); otherwise z = mean."""
        mean, logvar = self.encode(x)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean
        return self.decode(z), mean, logvar, z

    # -- the scaled latent the diffusion model works in ---------------------------------------
    def encode_scaled(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior mean x latent_scale (unit variance over the training rows)."""
        return self.encode(x)[0] * self.latent_scale

    def decode_scaled(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z / self.latent_scale)

    # -- export ---------------------------------------------------------------------------
    def export_state(self, **meta: Any) -> Dict[str, Any]:
        """A `torch.load(weights_only=True)`-safe dict: constructor kwargs,
        weights (cloned, CPU), the latent scale and any JSON-able `meta`."""
        return {"format": EXPORT_FORMAT, "kwargs": dict(self.kwargs),
                "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                "latent_scale": float(self.latent_scale), "latent_shape": list(self.latent_shape),
                "meta": dict(meta)}

    @classmethod
    def from_export(cls, state: Mapping[str, Any], *, device: Any = None) -> "SpatialAutoencoder":
        if state.get("format") != EXPORT_FORMAT:
            raise ValueError(f"not a {EXPORT_FORMAT} export (format {state.get('format')!r})")
        model = cls(**state["kwargs"])
        model.load_state_dict(state["state_dict"])
        model.latent_scale = float(state["latent_scale"])
        return model.to(device or "cpu").eval()


def kl_to_standard_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mean, exp(logvar)) || N(0, 1)) per sample, summed over every
    latent element: (B, ...) -> (B,), in nats. Computed in float32."""
    mean, logvar = mean.float(), logvar.float()
    return 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).flatten(1).sum(dim=1)


def make_sam_loss(codec: Any) -> nn.Module:
    """The SAM term: `codec.make_decoded_sam_loss(codec, angle_eps=1e-6)`,
    radians on decoded values, NCHW log-z inputs."""
    from hsi_augment.codec import make_decoded_sam_loss
    return make_decoded_sam_loss(codec, angle_eps=SAM_ANGLE_EPS)


class AELoss:
    """The autoencoder's `step_fn` for `train_generative.Trainer`:
    MSE (log-z) + sam_weight * SAM (decoded, radians) + kl_weight * KL.

    `sam_loss(y_true, y_pred)` is `make_sam_loss(codec)`, or None for
    sam_weight 0. Training decodes a sampled z,
    evaluation (`info.training` False) the posterior mean. Metrics: mse,
    sam_deg (the decoded SAM term in degrees), kl (nats per patch),
    kl_per_dim."""

    def __init__(self, sam_loss: Optional[nn.Module], *, sam_weight: float = 0.1, kl_weight: float = 1e-6):
        if sam_weight > 0 and sam_loss is None:
            raise ValueError("sam_weight > 0 needs a sam_loss")
        self.sam_loss, self.sam_weight, self.kl_weight = sam_loss, float(sam_weight), float(kl_weight)

    def __call__(self, model: SpatialAutoencoder, batch: Mapping[str, torch.Tensor], info: Any
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = batch["x"]
        x_hat, mean, logvar, _ = model(x, sample=bool(getattr(info, "training", True)))
        x_hat = x_hat.float()
        mse = F.mse_loss(x_hat, x.float())
        kl = kl_to_standard_normal(mean, logvar).mean()
        loss = mse + self.kl_weight * kl
        metrics: Dict[str, torch.Tensor] = {"mse": mse.detach(), "kl": kl.detach(),
                                            "kl_per_dim": kl.detach() / float(mean[0].numel())}
        if self.sam_loss is not None:
            sam = self.sam_loss(x.float(), x_hat)
            loss = loss + self.sam_weight * sam
            metrics["sam_deg"] = sam.detach() * (180.0 / math.pi)
        return loss, metrics


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
