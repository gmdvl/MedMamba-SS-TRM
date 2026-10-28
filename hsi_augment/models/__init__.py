# -*- coding: utf-8 -*-
"""
hsi_augment/models/
===================
The generators of Phases 6-7 and the blocks they share:

  blocks.py        GroupNorm + SiLU residual blocks, FiLM, stride-2 down/up
                   sampling for 11x11 -> 6x6 -> 3x3, sinusoidal timestep
                   embedding, `assert_no_batchnorm` (#22)
  cvae.py          conditional VAE, arm A2 (#23)
  autoencoder.py   unconditional low-KL autoencoder (#24)
  diffusion.py     class-conditional latent diffusion denoiser, arm A3 (#24)

All of them are trained by `hsi_augment.train_generative.Trainer` in the
log-space representation of `hsi_augment.codec.LogSpaceCodec`. This package
imports torch; `hsi_augment` itself does not, so import it inside the stage
functions that need it.
"""

from hsi_augment.models.blocks import (Downsample, FiLM, ResBlock, SinusoidalTimeEmbedding,
                                       TimestepEmbedder, Upsample, assert_no_batchnorm, crop_to,
                                       group_norm, timestep_embedding)

__all__ = ["Downsample", "FiLM", "ResBlock", "SinusoidalTimeEmbedding", "TimestepEmbedder", "Upsample",
           "assert_no_batchnorm", "crop_to", "group_norm", "timestep_embedding"]
