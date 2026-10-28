# -*- coding: utf-8 -*-
"""
hsi_augment/stages/fit_ldm.py
=============================
`augment_pipeline.py fit-ldm` (Plan §Phase 7, #24): the class-conditional
latent diffusion denoiser of arm A3, trained on the autoencoder's scaled
latents. Two stages, one group:

    ldm-lopo   gen/ldm/lopo/fold-<k>/   one fit per LOPO fold, in fold k's autoencoder
                                        (gen/ae/lopo/fold-<k>/ae.pt); latent MMD of
                                        generated vs held-out per class and guidance scale
    ldm-final  gen/ldm/final/           refit on every pool row, in ae-final's latent space,
                                        with ldm-lopo's settings; step budget = median of
                                        the folds' best steps

Each run directory ends with `generator.pt`, the autoencoder, denoiser and
codec in one `weights_only` file, for `latent_diffusion.load_generator` and
`models.diffusion.sample` (#29). The guidance scale is NOT chosen here: the
candidates (w in {1, 1.5, 2, 3}) are recorded, and the LOPO QC (#28) on the
host (#26) picks one. w > 3 is refused.

Both stages declare only the `generator` purpose (train rows only). The
logic is in `hsi_augment/latent_diffusion.py`. Like `fit-ae`, not in `all`
until the pool stage (#16) is wired in.

    python augment_pipeline.py fit-ldm --data_dir data/hsi_v9-trainsel/hsi \\
        --out data/hsi_v9-trainsel-aug
    python augment_pipeline.py ldm-lopo --data_dir ... --out ... --max_steps 150000
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.models.diffusion import MAX_GUIDANCE
from hsi_augment.registry import register_stage
from hsi_augment.stages.fit_ae import FINAL_STEP_RULES, check_device, check_folds
from hsi_augment.train_generative import TrainConfig   # imports torch: TrainConfig lives with the loop


@dataclass(frozen=True)
class LDMLopoConfig(TrainConfig):
    """`TrainConfig` with the denoiser's defaults (AdamW 2e-4, 1k warm-up,
    150k steps, EMA 0.9995, bf16 autocast), plus its data, model, diffusion,
    sampling and validation settings."""
    max_steps: int = param(150_000, help="optimizer steps per fit (the plan's 150k)")
    lr: float = param(2e-4, help="AdamW peak learning rate")
    warmup_steps: int = param(1000, help="linear warm-up steps before the cosine")
    ema_decay: Optional[float] = param(0.9995, help="EMA decay of the weights (none = no EMA)")
    precision: str = param("bf16", choices=("fp32", "bf16"), help="bf16 = autocast (the plan's); the loss, "
                                                                   "schedule and sampler stay float32")
    eval_every: int = param(2500, help="held-out diffusion loss every N steps (and at the end)")
    log_every: int = param(100, help="steps per history.csv row / metrics.jsonl train record")
    ckpt_every: int = param(2500, help="last.pt every N steps (also at every evaluation and the end)")
    folds: Tuple[int, ...] = param((0, 1), help="LOPO folds to fit; each needs ae-lopo's fold-<k>/ae.pt")
    batch_size: int = param(1024, help="class- then patient-balanced training batch of latents")
    eval_batch_size: int = param(4096, help="latents per held-out batch")
    encode_batch_size: int = param(4096, help="pool rows per batch when the latents are precomputed")
    channels: Tuple[int, ...] = param((128, 256), help="UNet widths per resolution level (6x6, 3x3)")
    blocks_per_level: int = param(1, help="residual blocks per level, down and up")
    mid_blocks: int = param(1, help="residual blocks at the lowest resolution")
    emb_dim: int = param(512, help="time + class embedding width (FiLM conditioning)")
    freq_dim: int = param(256, help="sinusoidal timestep features before the time MLP")
    dropout: float = param(0.0, help="dropout inside the residual blocks")
    timesteps: int = param(1000, help="diffusion steps T of the cosine schedule")
    min_snr_gamma: Optional[float] = param(5.0, help="Min-SNR-gamma loss weighting (none = plain v-MSE)")
    label_dropout: float = param(0.1, help="probability of training on the null label (classifier-free "
                                           "guidance)")
    sample_steps: int = param(50, help="DDIM steps (eta = 0)")
    guidance: Tuple[float, ...] = param((1.0, 1.5, 2.0, 3.0), help="guidance scales w to validate and "
                                                                   "record; the host QC (#26/#28) picks one")
    sample_batch_size: int = param(1024, help="rows per sampling batch (the noise does not depend on it; the "
                                              "sampling sha256 is recorded with it)")
    mmd_rows: int = param(2000, help="rows per class for the latent MMD (held-out, train reference and "
                                     "generated sets each get this many, or the held-out count if smaller)")
    mmd_bandwidths: Tuple[float, ...] = param((0.5, 1.0, 2.0), help="RBF bandwidths as multiples of the "
                                                                    "median pairwise distance")
    device: str = param("auto", help="auto (cuda if available) | cpu | cuda[:n]")

    def validate(self) -> None:
        super().validate()
        check_folds(self.folds)
        check_device(self.device)
        for name in ("batch_size", "eval_batch_size", "encode_batch_size", "sample_batch_size", "mmd_rows"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be > 0")
        if self.batch_size < 3:
            raise ConfigError("batch_size must be >= 3 (one row per class)")
        if not self.channels or min(self.channels) <= 0 or self.blocks_per_level < 1 or self.mid_blocks < 0:
            raise ConfigError("channels must be > 0, blocks_per_level >= 1 and mid_blocks >= 0")
        if self.emb_dim <= 0 or self.freq_dim < 2 or not 0.0 <= self.dropout < 1.0:
            raise ConfigError("emb_dim must be > 0, freq_dim >= 2 and dropout in [0, 1)")
        if self.timesteps < 2 or not 1 <= self.sample_steps <= self.timesteps:
            raise ConfigError("timesteps must be >= 2 and sample_steps in [1, timesteps]")
        if self.min_snr_gamma is not None and not self.min_snr_gamma > 0:
            raise ConfigError("min_snr_gamma must be > 0 (or none)")
        if not 0.0 <= self.label_dropout < 1.0:
            raise ConfigError("label_dropout must be in [0, 1): 1 would never train the conditional model")
        if self.label_dropout == 0.0 and any(w != 1.0 for w in self.guidance):
            raise ConfigError("guidance w != 1 needs the null token trained: label_dropout must be > 0")
        if not self.guidance or any(not 0.0 <= w <= MAX_GUIDANCE for w in self.guidance) \
                or len(set(self.guidance)) != len(self.guidance):
            raise ConfigError(f"guidance must be distinct scales in [0, {MAX_GUIDANCE:g}]: w > {MAX_GUIDANCE:g} "
                              f"pulls samples toward the training modes (Plan §Phase 7)")
        if not self.mmd_bandwidths or any(not b > 0 for b in self.mmd_bandwidths):
            raise ConfigError("mmd_bandwidths must be positive multipliers")


@dataclass(frozen=True)
class LDMFinalConfig(StageConfig):
    """Only what the final refit may choose itself: the model, diffusion and
    training settings are ldm-lopo's (read from its done.json)."""
    steps: str = param("median_best", choices=FINAL_STEP_RULES,
                       help="step budget: median of ldm-lopo's best steps, or ldm-lopo's max_steps")
    device: str = param("auto", help="auto (cuda if available) | cpu | cuda[:n]")

    def validate(self) -> None:
        check_device(self.device)


@register_stage("ldm-lopo", config=LDMLopoConfig, group="fit-ldm", dirname="gen/ldm/lopo", order=80,
                requires=("ae-lopo",), purposes=("generator",), in_all=False,
                help="P7: class-conditional latent diffusion (v-pred, CFG), one fit per LOPO fold")
def run_lopo(ctx, cfg: LDMLopoConfig) -> Dict[str, Any]:
    from hsi_augment import latent_diffusion
    return latent_diffusion.run_ldm_lopo(ctx, cfg)


@register_stage("ldm-final", config=LDMFinalConfig, group="fit-ldm", dirname="gen/ldm/final", order=81,
                requires=("ae-final", "ldm-lopo"), purposes=("generator",), in_all=False,
                help="P7: latent diffusion refit on all pool rows with the LOPO settings")
def run_final(ctx, cfg: LDMFinalConfig) -> Dict[str, Any]:
    from hsi_augment import latent_diffusion
    return latent_diffusion.run_ldm_final(ctx, cfg)
