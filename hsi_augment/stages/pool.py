# -*- coding: utf-8 -*-
"""
hsi_augment/stages/pool.py
==========================
`augment_pipeline.py pool`: Phase 4b, issue #16 (Plan §Phase 4 items 3-4). The
generator pool every generator trains on, and the real-vs-real references:

  * the log-space statistics of log(x + 1e-3) over EVERY train tissue row
    (float64) and the per-band decode clip (1.1 x the exact train-tissue
    p99.99), saved as the pool's `codec.json`;
  * `X_pool.npy` (float16, NHWC, already log-z encoded): every tissue row of
    the minority class (DCIS) plus `rows_per_class` patient-balanced tissue
    rows of each other class, read in ONE pass over the train memmap, with the
    `y` / `patient` / `row` / `capture` sidecars (file contract:
    `hsi_augment/pool_io.py`). A LOPO fold is `splits.fold_masks` over
    `patient_pool.npy`, so there is one pool, not one per fold;
  * `reference.npz`: per LOPO fold, the held-out patients' pool rows against
    the remaining patients' pool rows of the same class: cross-capture
    nearest-neighbour distances in PCA-64 (PCA fitted on the pool) and the
    mean/std feature distributions.

The logic and the schemas are in `hsi_augment/pool.py`.

Split access, declared here in one line: the purpose `pool` (train only), so
the SplitGuard refuses val and test. Upstreams: `tissue-mask` (which rows are
tissue, and tau) and `leakage` (the LOPO folds).

tau must be signed off by the user before the real pool is built (Plan
§Phase 2 item 6). Pass the signed-off value as `--expect_tau_deg`; the stage
then refuses a tissue mask fitted with another tau.

    python augment_pipeline.py pool --data_dir data/hsi_v9-trainsel/hsi \\
                                    --out data/hsi_v9-trainsel-aug --expect_tau_deg <tau>
    # -> data/hsi_v9-trainsel-aug/pool/{X_pool.npy, y_pool.npy, patient_pool.npy, row_pool.npy,
    #    capture_pool.npy, codec.json, reference.npz, pool.json}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.io import DEFAULT_CHUNK_ROWS
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class PoolConfig(StageConfig):
    rows_per_class: int = param(120_000, help="tissue rows of each non-minority class (healthy, IDC), shared "
                                              "as evenly as possible across that class's train patients")
    minority_rows: Optional[int] = param(None, help="cap on the minority class's (DCIS) tissue rows, "
                                                    "patient-balanced like the others (none = every one)")
    max_pool_gb: float = param(3.5, help="refuse an X_pool.npy larger than this many GB (1e9 bytes); "
                                         "checked before any X is read")
    expect_tau_deg: Optional[float] = param(None, help="the tissue threshold tau (degrees) the user signed "
                                                       "off: a tissue-mask tau further than 0.0005 from it is "
                                                       "refused (none = not checked; a warning)")
    pca_components: int = param(64, help="dimensions of the reference PCA (fitted on the encoded pool)")
    pca_fit_rows: Optional[int] = param(100_000, help="seeded pool rows the PCA is fitted on "
                                                      "(none = every pool row)")
    nn_max_queries: Optional[int] = param(20_000, help="per LOPO fold and class: seeded held-out pool rows "
                                                       "whose nearest neighbour is searched (none = all)")
    quantiles: Tuple[float, ...] = param((0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
                                         help="quantile levels of the reference summaries, each in [0, 1]")
    chunk_rows: int = param(DEFAULT_CHUNK_ROWS, help="rows per memmap read of X_train")

    def validate(self) -> None:
        if self.rows_per_class < 1:
            raise ConfigError("rows_per_class must be >= 1")
        if self.minority_rows is not None and self.minority_rows < 1:
            raise ConfigError("minority_rows must be >= 1 (or none)")
        if not self.max_pool_gb > 0:
            raise ConfigError("max_pool_gb must be > 0")
        if self.expect_tau_deg is not None and not 0.0 < self.expect_tau_deg < 180.0:
            raise ConfigError("expect_tau_deg must be an angle in (0, 180) degrees (or none)")
        if self.pca_components < 1:
            raise ConfigError("pca_components must be >= 1")
        if self.pca_fit_rows is not None and self.pca_fit_rows < 2:
            raise ConfigError("pca_fit_rows must be >= 2 (or none)")
        if self.nn_max_queries is not None and self.nn_max_queries < 1:
            raise ConfigError("nn_max_queries must be >= 1 (or none)")
        if not self.quantiles or any(not 0.0 <= q <= 1.0 for q in self.quantiles):
            raise ConfigError("quantiles must be one or more levels in [0, 1]")
        if len(set(self.quantiles)) != len(self.quantiles):
            raise ConfigError("quantiles must be distinct")
        if self.chunk_rows <= 0:
            raise ConfigError("chunk_rows must be > 0")


@register_stage("pool", config=PoolConfig, dirname="pool", order=50, requires=("tissue-mask", "leakage"),
                purposes=("pool",),
                help="P4b: generator pool (log space, patient-balanced) and real-vs-real references")
def run(ctx, cfg: PoolConfig) -> Dict[str, Any]:
    from hsi_augment import pool               # numpy only
    return pool.run_pool(ctx, cfg)
