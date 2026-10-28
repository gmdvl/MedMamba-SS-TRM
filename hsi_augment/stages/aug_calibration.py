# -*- coding: utf-8 -*-
"""
hsi_augment/stages/aug_calibration.py
=====================================
`augment_pipeline.py aug-calibration` (also run by `preprocess`), issue #20:
the largest `spectral_tilt` magnitude whose change of the band-band
correlation is within the paper recipe's own augmentation's, and the same
check for `sensor_noise` at k = 1 with the real sigma_b of `spectral-audit`.
`train.py --phys_tilt_max auto --phys_artifacts <out>` reads the result. The
criterion, the measurement and the output schema are in `hsi_augment/augcal.py`.

It reads a seeded sample of X_train, through the SplitGuard with purpose
`fit`; val and test are never opened.

    python augment_pipeline.py aug-calibration --data_dir data/hsi_v9-trainsel/hsi \\
                                               --out data/hsi_v9-trainsel-aug
    # -> data/hsi_v9-trainsel-aug/prep/augcal/{phys_calibration.json, augmentation_report.json,
    #    augmentation_montage.png}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.io import DEFAULT_CHUNK_ROWS
from hsi_augment.registry import register_stage

MAX_TILT = 0.5     # hsi_augment.transforms.MAX_TILT; not imported, so --help does not load torch


@dataclass(frozen=True)
class AugCalibrationConfig(StageConfig):
    sample_rows: int = param(20_000, help="seeded train rows the correlation changes are measured on")
    tilt_grid: Tuple[float, ...] = param(tuple(round(0.001 * i, 4) for i in range(1, 31)),
                                         help="tilt_max values scanned; the largest within the base "
                                              "augmentation's change is chosen")
    noise_k: float = param(1.0, help="sensor_noise multiplier checked against the criterion (0 = skip)")
    norm_sample_cap: int = param(5000, help="rows of train.py's normalization estimate "
                                            "(--norm_stats_sample_cap)")
    norm_seed: int = param(42, help="seed of train.py's normalization estimate (train.py --seed)")
    chunk_rows: int = param(DEFAULT_CHUNK_ROWS, help="rows per memmap read of the sample")
    montage: bool = param(True, help="write augmentation_montage.png of the chosen ops")
    montage_n: int = param(4, help="samples in the montage")

    def validate(self) -> None:
        if self.sample_rows <= 0 or self.norm_sample_cap <= 0 or self.chunk_rows <= 0 or self.montage_n <= 0:
            raise ConfigError("sample_rows, norm_sample_cap, chunk_rows and montage_n must be > 0")
        if not self.tilt_grid or any(not 0.0 < t < MAX_TILT for t in self.tilt_grid):
            raise ConfigError(f"tilt_grid must be one or more values in (0, {MAX_TILT})")
        if len(set(self.tilt_grid)) != len(self.tilt_grid):
            raise ConfigError("tilt_grid has duplicate values")
        if self.noise_k < 0:
            raise ConfigError("noise_k must be >= 0")
        if self.norm_seed < 0:
            raise ConfigError("norm_seed must be >= 0")


@register_stage("aug-calibration", config=AugCalibrationConfig, group="preprocess", dirname="prep/augcal",
                order=40, requires=("spectral-audit",), purposes=("fit",),
                help="#20: tilt_max bound and sensor_noise check against the paper recipe's augmentation")
def run(ctx, cfg: AugCalibrationConfig) -> Dict[str, Any]:
    from hsi_augment import augcal      # torch inside, so --help stays instant
    return augcal.run_aug_calibration(ctx, cfg)
