# -*- coding: utf-8 -*-
"""
hsi_augment/stages/fit_ae.py
============================
`augment_pipeline.py fit-ae` (Plan §Phase 7, #24): the unconditional spatial
autoencoder whose latent space the class-conditional diffusion model of arm
A3 is trained in. Two stages, one group:

    ae-lopo   gen/ae/lopo/fold-<k>/   one fit per LOPO fold (default folds 0 and 1:
                                      the plan selects on 2 folds), held-out
                                      patients for best.pt and the diagnostics
    ae-final  gen/ae/final/           refit on every pool row with ae-lopo's
                                      settings; step budget = median of the
                                      folds' best steps (as Phase 6's final refit)

Why a separate, unconditional autoencoder (and not the CVAE's latent): see
`hsi_augment/models/autoencoder.py`. The logic, the output files and the
pool contract are in `hsi_augment/latent_diffusion.py`.

Both stages require the `pool` stage and read its directory (contract in
`hsi_augment/pool_io.py`; train rows only, checked against the train split's
sidecars through the SplitGuard): without a finished pool they stop with
StageNotReady, and re-running the pool with another config makes them stale.
They declare only the `generator` purpose, so they never read val or test.
`ae-final` takes every model, loss and optimiser setting from `ae-lopo`'s
done.json, so the final model is the LOPO-chosen one by construction; its own
flags are only the step-budget rule, the device and the diagnostics size.

Not in `all` (`in_all=False`): the defaults are 30k-step fits for the host
GPU (#26), run as their own command (`fit-ae`); `all` is the CPU prep chain.

    python augment_pipeline.py fit-ae --data_dir data/hsi_v9-trainsel/hsi \\
        --out data/hsi_v9-trainsel-aug
    python augment_pipeline.py ae-lopo --data_dir ... --out ... --folds 0 1 --max_steps 30000
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.registry import register_stage
from hsi_augment.train_generative import TrainConfig   # imports torch: TrainConfig lives with the loop

FINAL_STEP_RULES = ("median_best", "lopo_max_steps")


def check_folds(folds: Tuple[int, ...]) -> None:
    if not folds or any(k < 0 for k in folds) or len(set(folds)) != len(folds):
        raise ConfigError("folds must be one or more distinct non-negative fold numbers")


def check_device(device: str) -> None:
    if not (device == "auto" or device == "cpu" or device.startswith("cuda")):
        raise ConfigError("device must be auto, cpu, cuda or cuda:<n>")


@dataclass(frozen=True)
class AELopoConfig(TrainConfig):
    """`TrainConfig` (optimizer, schedule, EMA, evaluation, checkpoints) with
    the autoencoder's defaults, plus its data, model, loss and diagnostics."""
    max_steps: int = param(30_000, help="optimizer steps per fit (the plan's 30k)")
    lr: float = param(1e-3, help="AdamW peak learning rate (the CVAE's: same backbone)")
    warmup_steps: int = param(500, help="linear warm-up steps before the cosine")
    eval_every: int = param(1000, help="held-out evaluation every N steps (and at the end)")
    log_every: int = param(100, help="steps per history.csv row / metrics.jsonl train record")
    ckpt_every: int = param(1000, help="last.pt every N steps (also at every evaluation and the end)")
    folds: Tuple[int, ...] = param((0, 1), help="LOPO folds of prep/leakage/folds.json to fit, one fold-<k>/ "
                                                "each (the plan selects on 2 folds)")
    batch_size: int = param(512, help="class- then patient-balanced training batch")
    eval_batch_size: int = param(2048, help="rows per held-out / encoding batch")
    widths: Tuple[int, int] = param((96, 192), help="channels at full and half resolution (11x11, 6x6)")
    blocks_per_level: int = param(2, help="residual blocks per resolution level, encoder and decoder each")
    latent_channels: int = param(4, help="latent channels at half resolution (4 x 6 x 6 = 144 values)")
    dropout: float = param(0.0, help="dropout inside the residual blocks")
    sam_weight: float = param(0.1, help="weight of the decoded SAM term (radians; angle_eps 1e-6, no "
                                        "2.56 deg floor)")
    kl_weight: float = param(1e-6, help="weight of the KL term (nats per patch)")
    metric_rows: int = param(20_000, help="rows per class (seeded draw) for the reconstruction diagnostics, "
                                          "on each side (train / held-out)")
    device: str = param("auto", help="auto (cuda if available) | cpu | cuda[:n]")

    def validate(self) -> None:
        super().validate()
        check_folds(self.folds)
        check_device(self.device)
        if self.batch_size < 3 or self.eval_batch_size <= 0:
            raise ConfigError("batch_size must be >= 3 (one row per class) and eval_batch_size > 0")
        if min(self.widths) <= 0 or self.blocks_per_level < 1 or self.latent_channels < 1:
            raise ConfigError("widths must be > 0, blocks_per_level >= 1 and latent_channels >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("dropout must be in [0, 1)")
        if self.sam_weight < 0 or self.kl_weight < 0:
            raise ConfigError("sam_weight and kl_weight must be >= 0")
        if self.metric_rows <= 0:
            raise ConfigError("metric_rows must be > 0")


@dataclass(frozen=True)
class AEFinalConfig(StageConfig):
    """Only what the final refit may choose itself: the model and training
    settings are ae-lopo's (read from its done.json)."""
    steps: str = param("median_best", choices=FINAL_STEP_RULES,
                       help="step budget: median of ae-lopo's best steps, or ae-lopo's max_steps")
    metric_rows: int = param(20_000, help="rows per class (seeded draw) for the train reconstruction diagnostics")
    device: str = param("auto", help="auto (cuda if available) | cpu | cuda[:n]")

    def validate(self) -> None:
        check_device(self.device)
        if self.metric_rows <= 0:
            raise ConfigError("metric_rows must be > 0")


@register_stage("ae-lopo", config=AELopoConfig, group="fit-ae", dirname="gen/ae/lopo", order=70,
                requires=("leakage", "pool"), purposes=("generator",), in_all=False,
                help="P7: unconditional spatial autoencoder (4x6x6 latent), one fit per LOPO fold")
def run_lopo(ctx, cfg: AELopoConfig) -> Dict[str, Any]:
    from hsi_augment import latent_diffusion       # torch inside, so --help stays fast
    return latent_diffusion.run_ae_lopo(ctx, cfg)


@register_stage("ae-final", config=AEFinalConfig, group="fit-ae", dirname="gen/ae/final", order=71,
                requires=("ae-lopo", "pool"), purposes=("generator",), in_all=False,
                help="P7: autoencoder refit on all pool rows with the LOPO settings")
def run_final(ctx, cfg: AEFinalConfig) -> Dict[str, Any]:
    from hsi_augment import latent_diffusion
    return latent_diffusion.run_ae_final(ctx, cfg)
