# -*- coding: utf-8 -*-
"""
hsi_augment/stages/spectral_audit.py
====================================
`augment_pipeline.py spectral-audit`: Phase 2a (#12, Plan §Phase 2 items
1-3), one of the stages of `augment_pipeline.py preprocess`. It audits the
build and fits the train-only linear statistics; it changes no value (there is
no white reference to re-calibrate with, and the classifier's data stays
fixed). The logic is in `hsi_augment/preprocess.py`.

Outputs in `<out>/prep/spectral/`:

    calibration_report.json  gain and median intensity per split, class and patient;
                             clipped and suspect captures (metadata only)
    band_report.json         per-band validity (variance, saturation, zeros, negatives,
                             implied raw ceiling), the band-band correlation matrix of a
                             seeded 200k-row sample, the noise estimate, `problems`
    norm_stats.json          exact float64 per-band mean/std over ALL train rows, and
                             train.py's 5,000-row estimate against it in SE units
    noise_sigma.npy          float32 [C]: per-pixel noise std sigma_b in value units,
                             for `sensor_noise` (#19, via --phys_artifacts)

It reads X_train once, in memmap chunks, through the SplitGuard with purpose
`fit`; val and test arrays are never opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.io import DEFAULT_CHUNK_ROWS
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class SpectralAuditConfig(StageConfig):
    chunk_rows: int = param(DEFAULT_CHUNK_ROWS, help="rows per memmap chunk")
    max_rows: Optional[int] = param(None, help="scan only the first N train rows, for smoke tests "
                                               "(none = all); the reports then say the scan is partial")
    corr_sample_rows: int = param(200_000, help="seeded train rows for the band-band correlation matrix")
    noise_sample_rows: int = param(20_000, help="seeded train rows for the per-band noise sigma_b")
    noise_flat_fraction: float = param(0.1, help="flattest fraction of the noise sample for the "
                                                 "secondary flat-patch sigma_b")
    min_std: float = param(1e-6, help="a band with train std at or below this has zero variance "
                                      "(train.py floors such a std to 1.0)")
    near_zero_eps: float = param(1e-3, help="|x| at or below this is near zero (value units; 1e-3 is "
                                            "about 8.6 raw counts at gain 1)")
    raw_ceiling: float = param(65535.0, help="sensor ceiling in raw counts (16-bit)")
    raw_ceiling_frac: float = param(0.98, help="implied raw count at or above this fraction of "
                                               "raw_ceiling is reported as near saturation")
    corr_flag: float = param(0.98, help="report band pairs whose |r| is above this")
    compare_sample_cap: int = param(5000, help="rows of the compute_global_channel_stats estimate "
                                               "(train.py --norm_stats_sample_cap)")
    compare_seeds: Tuple[int, ...] = param((1, 7, 13, 23, 42), help="seeds of that estimate (train.py "
                                                                    "--seed; the A0 seeds)")
    z_tol: float = param(4.0, help="|estimate - exact| / standard error above this is a disagreement")
    gain_outlier_z: float = param(3.5, help="robust z of log(gain), against train captures, that "
                                            "flags a capture")

    def validate(self) -> None:
        for name in ("chunk_rows", "corr_sample_rows", "noise_sample_rows", "compare_sample_cap"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be > 0")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ConfigError("max_rows must be > 0 (or none)")
        if not 0.0 < self.noise_flat_fraction <= 1.0:
            raise ConfigError("noise_flat_fraction must be in (0, 1]")
        if self.min_std < 0 or self.near_zero_eps < 0:
            raise ConfigError("min_std and near_zero_eps must be >= 0")
        if self.raw_ceiling <= 0 or not 0.0 < self.raw_ceiling_frac <= 1.0:
            raise ConfigError("raw_ceiling must be > 0 and raw_ceiling_frac in (0, 1]")
        if not 0.0 < self.corr_flag <= 1.0:
            raise ConfigError("corr_flag must be in (0, 1]")
        if not self.compare_seeds or any(s < 0 for s in self.compare_seeds):
            raise ConfigError("compare_seeds must be one or more non-negative ints")
        if self.z_tol <= 0 or self.gain_outlier_z <= 0:
            raise ConfigError("z_tol and gain_outlier_z must be > 0")


@register_stage("spectral-audit", config=SpectralAuditConfig, group="preprocess", dirname="prep/spectral",
                order=20, purposes=("fit",),
                help="P2a: calibration audit, band validity, noise sigma_b, train-only linear stats")
def run(ctx, cfg: SpectralAuditConfig) -> Dict[str, Any]:
    from hsi_augment import preprocess      # heavy work inside, so --help stays instant
    return preprocess.run_spectral_audit(ctx, cfg)
