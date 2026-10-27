# -*- coding: utf-8 -*-
"""
hsi_augment/stages/index.py
===========================
`augment_pipeline.py index` (also run by `preprocess`): Phase 4a, issue #15.
The provenance index of the train split (one entry per train row: label,
patient, capture, grid position, tissue score and mask, per-band mean/std
features) and the imbalance report (class and patient tables per split, the
same on train tissue rows, the effective number of samples). The logic and the
output schemas are in `hsi_augment/provenance.py`; the figures in
`hsi_augment/provenance_figures.py`.

Split access, declared here in one line: `describe` (train only) for X_train
and its sidecars, and `structural` for the y / groups / captures sidecars of
val and test, which the per-split count tables need. Val and test X are never
opened, so there is no val provenance index (see `provenance.py`).

    python augment_pipeline.py index --data_dir data/hsi_v9-trainsel/hsi \\
                                     --out data/hsi_v9-trainsel-aug
    # -> data/hsi_v9-trainsel-aug/prep/index/{index_train.npz, index_train.json,
    #    imbalance_report.json, figures/}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.io import DEFAULT_CHUNK_ROWS
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class IndexConfig(StageConfig):
    chunk_rows: int = param(DEFAULT_CHUNK_ROWS, help="rows per memmap read of X_train")
    betas: Tuple[float, ...] = param((0.999, 0.9999), help="beta values of the effective number of samples "
                                                           "(Cui et al. 2019), each in [0, 1)")
    max_rows: Optional[int] = param(None, help="smoke runs only: index the first N train rows (none = all); "
                                               "index_train.json then says complete=false. The imbalance "
                                               "report reads no X and always covers every row")

    def validate(self) -> None:
        if self.chunk_rows <= 0:
            raise ConfigError("chunk_rows must be > 0")
        if not self.betas or any(not 0.0 <= b < 1.0 for b in self.betas):
            raise ConfigError("betas must be one or more values in [0, 1)")
        if len(set(self.betas)) != len(self.betas):
            raise ConfigError("betas must be distinct")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ConfigError("max_rows must be > 0 (or none)")


@register_stage("index", config=IndexConfig, group="preprocess", dirname="prep/index", order=40,
                requires=("tissue-mask",), purposes=("describe", "structural"),
                help="P4a: provenance index of train rows (patch mean/std features) and imbalance report")
def run(ctx, cfg: IndexConfig) -> Dict[str, Any]:
    from hsi_augment import provenance       # numpy only; matplotlib loads inside, for the figures
    return provenance.run_index(ctx, cfg)
