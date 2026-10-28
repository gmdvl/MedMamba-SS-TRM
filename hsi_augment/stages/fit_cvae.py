# -*- coding: utf-8 -*-
"""
hsi_augment/stages/fit_cvae.py
==============================
`augment_pipeline.py fit-cvae` (Phase 6, #23): the conditional VAE of arm A2,
as two stages of the group `fit-cvae`, run in this order:

    cvae-lopo   -> <out>/gen/cvae/lopo/    every grid cell x every LOPO fold, early-stopped
                                           on the fold's held-out patients; diagnostics
    cvae-final  -> <out>/gen/cvae/final/   picks the cell on the LOPO results, refits it on
                                           every train patient for the median of the folds'
                                           best steps; EMA weights; no further selection

    python augment_pipeline.py fit-cvae --data_dir data/hsi_v9-trainsel/hsi \\
                                        --out data/hsi_v9-trainsel-aug [--resume auto]
    python augment_pipeline.py cvae-lopo ... --latent_dims 64 --sam_ratios 0.1   # one cell

Why two stages and not one stage per fold: a stage has one directory and one
done.json, and the final refit needs every fold's best step. The LOPO stage
trains each (cell, fold) in its own `Trainer` run directory, so folds never
overwrite each other and `--resume auto` continues the interrupted fit and
re-uses the finished ones. The cell choice lives in the FINAL stage's config:
changing the rule re-runs the cheap refit, not twenty LOPO fits.

Inputs. The pool (`<out>/pool/`, contract in `hsi_augment/pool_io.py`, written
by #16) is read from `--pool_dir` (default `<out>/pool`); the pool stage is
not a registered upstream yet, so a missing pool is a StageError, and the
pool's content sha256 is recorded and compared by the final stage. The folds
come from the `leakage` stage (`requires`). Both stages declare the single
purpose "generator": their SplitGuard refuses val and test, and the only
access in the log is train's y / groups / captures, read to check that the
pool belongs to this build.

Importing this module imports torch (`TrainConfig` lives in
`hsi_augment.train_generative`, and the configs subclass it); the fitting
code itself is `hsi_augment/fit_cvae.py`, imported inside the stage functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.registry import register_stage
from hsi_augment.train_generative import TrainConfig

DEVICES = ("auto", "cpu", "cuda")
EVAL_METRICS = ("nll", "rec_nll", "loss", "mse", "sam")
SELECT_METRICS = ("nll", "sam_deg")


@dataclass(frozen=True)
class CVAETrainConfig(TrainConfig):
    """One CVAE fit's settings shared by every cell (Plan §Phase 6,
    "Training", "Loss"). Not a stage config by itself."""
    # TrainConfig fields whose defaults the plan sets differently
    max_steps: int = param(40_000, help="optimizer steps at most; the cosine spans exactly these")
    eval_metric: str = param("nll", choices=EVAL_METRICS,
                             help="held-out metric for best.pt and early stopping; nll = negative ELBO "
                                  "(beta 1, no free bits), nats per element")
    early_stop_patience: Optional[int] = param(10, help="stop after N evaluations without improvement "
                                                        "(none = never)")
    # data
    batch_size: int = param(512, help="rows per class-, then patient-balanced training batch")
    eval_batch_size: int = param(1024, help="rows per held-out / diagnostics batch")
    # model
    widths: Tuple[int, int, int] = param((96, 192, 256), help="channels at 11x11, 6x6, 3x3")
    cond_dim: int = param(64, help="class embedding size (FiLM input)")
    dropout: float = param(0.0, help="dropout inside the ResBlocks")
    min_log_sigma: float = param(-6.0, help="floor of the learned per-band log sigma")
    # loss
    free_bits: float = param(0.05, help="KL floor per latent dim, nats (batch mean)")
    beta_warmup_frac: float = param(0.1, help="beta rises 0 -> 1 over this fraction of max_steps")
    sam_angle_eps: float = param(1e-6, help="SAMLoss angle_eps (1e-3 would floor SAM at 2.56 deg)")
    sam_calib_batches: int = param(4, help="training batches the SAM weight is calibrated on at init")

    def validate(self) -> None:
        super().validate()
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ConfigError("batch_size and eval_batch_size must be > 0")
        if min(self.widths) < 1 or self.cond_dim < 1:
            raise ConfigError("widths and cond_dim must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("dropout must be in [0, 1)")
        if self.free_bits < 0 or not 0.0 <= self.beta_warmup_frac <= 1.0:
            raise ConfigError("need free_bits >= 0 and beta_warmup_frac in [0, 1]")
        if not 0.0 < self.sam_angle_eps < 0.1:
            raise ConfigError("sam_angle_eps must be in (0, 0.1)")
        if self.sam_calib_batches < 1:
            raise ConfigError("sam_calib_batches must be >= 1")
        if self.ema_decay is None:
            raise ConfigError("the CVAE exports EMA weights (Plan §Phase 6 'Final model'): ema_decay "
                              "cannot be none")


@dataclass(frozen=True)
class CVAEFitConfig(CVAETrainConfig):
    """What `Trainer` gets for ONE fit (a LOPO cell x fold, or the final
    refit): the shared settings plus the cell. Resume compares all of it."""
    latent_dim: int = param(64, help="latent dimension d")
    sam_ratio: float = param(0.1, help="lambda x SAM_init = sam_ratio x rec_init")

    def validate(self) -> None:
        super().validate()
        if self.latent_dim < 1 or self.sam_ratio < 0:
            raise ConfigError("need latent_dim >= 1 and sam_ratio >= 0")


@dataclass(frozen=True)
class CVAELopoConfig(CVAETrainConfig):
    latent_dims: Tuple[int, ...] = param((32, 64), help="grid: latent dimension d")
    sam_ratios: Tuple[float, ...] = param((0.0, 0.1), help="grid: SAM term / reconstruction term at init "
                                                           "(lambda is calibrated per fit)")
    pool_dir: Optional[str] = param(None, help="generator pool directory (none = <out>/pool)")
    pool_in_ram: bool = param(True, help="copy X_pool into RAM (about 3 GB) instead of reading the memmap")
    device: str = param("auto", choices=DEVICES, help="auto = cuda if available")
    tau_s: Tuple[float, ...] = param((0.8, 1.0), help="prior temperature candidates; the choice is LOPO "
                                                      "QC's (#28), these are recorded and diagnosed")
    diag_train_rows: int = param(20_000, help="seeded fold-train rows for the train-side SAM reference "
                                              "(held-out SAM <= 1.25 x train SAM)")
    active_unit_threshold: float = param(0.01, help="a latent dim is active if Cov_x E[z|x] exceeds this")
    prior_samples: int = param(2048, help="prior samples per tau_s for the clip / mean-spectrum check")

    def validate(self) -> None:
        super().validate()
        for name in ("latent_dims", "sam_ratios", "tau_s"):
            v = getattr(self, name)
            if not v or len(set(v)) != len(v):
                raise ConfigError(f"{name} must be one or more distinct values")
        if min(self.latent_dims) < 1 or min(self.sam_ratios) < 0 or min(self.tau_s) <= 0:
            raise ConfigError("need latent_dims >= 1, sam_ratios >= 0 and tau_s > 0")
        if self.diag_train_rows < 0 or self.prior_samples < 0:
            raise ConfigError("diag_train_rows and prior_samples must be >= 0")
        if not self.active_unit_threshold > 0:
            raise ConfigError("active_unit_threshold must be > 0")


@dataclass(frozen=True)
class CVAEFinalConfig(StageConfig):
    select_metric: str = param("nll", choices=SELECT_METRICS,
                               help="cell choice: lowest across-fold mean held-out nll (negative ELBO) "
                                    "or SAM deg, among the cells passing the gates")
    min_active_frac: float = param(0.5, help="gate: mean fraction of active latent units")
    max_sam_ratio: float = param(1.25, help="gate: mean held-out SAM / mean train SAM")
    enforce_gates: bool = param(True, help="off: if no cell passes, take the best one anyway (warning)")
    device: str = param("auto", choices=DEVICES, help="auto = cuda if available")
    prior_samples: int = param(2048, help="prior samples per tau_s for the clip / mean-spectrum check")

    def validate(self) -> None:
        if not 0.0 <= self.min_active_frac <= 1.0:
            raise ConfigError("min_active_frac must be in [0, 1]")
        if not self.max_sam_ratio > 0:
            raise ConfigError("max_sam_ratio must be > 0")
        if self.prior_samples < 0:
            raise ConfigError("prior_samples must be >= 0")


# in_all=False until the pool stage (#16) is registered: `all` cannot produce the pool, so with these
# stages in it `all` would always fail. Whoever adds requires=("pool",) (after #16) also drops in_all.
@register_stage("cvae-lopo", config=CVAELopoConfig, group="fit-cvae", dirname="gen/cvae/lopo", order=60,
                requires=("leakage",), purposes=("generator",), in_all=False,
                help="P6: CVAE grid x LOPO folds, early-stopped on held-out patients; diagnostics")
def run_lopo(ctx, cfg: CVAELopoConfig) -> Dict[str, Any]:
    from hsi_augment import fit_cvae
    return fit_cvae.run_lopo(ctx, cfg)


@register_stage("cvae-final", config=CVAEFinalConfig, group="fit-cvae", dirname="gen/cvae/final", order=61,
                requires=("cvae-lopo",), purposes=("generator",), in_all=False,
                help="P6: CVAE refit on all train patients, LOPO-chosen cell and step budget, EMA")
def run_final(ctx, cfg: CVAEFinalConfig) -> Dict[str, Any]:
    from hsi_augment import fit_cvae
    return fit_cvae.run_final(ctx, cfg)
