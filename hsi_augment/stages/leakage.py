# -*- coding: utf-8 -*-
"""
hsi_augment/stages/leakage.py
=============================
`augment_pipeline.py leakage` (Phase 3, #14; in the `preprocess` group):
verify the fixed split, draw the LOPO folds inside train, write the leakage
ledger. Outputs in `<out>/prep/leakage/`: split_report.json, folds.json,
leakage_ledger.json. The logic and the folds.json / ledger schemas are in
`hsi_augment/splits.py`.

It reads y, groups and captures of every split, never X, so it declares only
the `structural` purpose (counts and ids of any split). It does not declare
`pool` even though it draws the folds: a stage that pools, generates, runs QC
or LOPO may not also see val or test (splits.py, "The split-access policy").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class LeakageConfig(StageConfig):
    companions_per_class: int = param(1, help="train patients held out per non-DCIS class (healthy, IDC) "
                                              "in each LOPO fold, besides the fold's DCIS patient")

    def validate(self) -> None:
        if self.companions_per_class < 1:
            raise ConfigError("companions_per_class must be >= 1")


@register_stage("leakage", config=LeakageConfig, group="preprocess", dirname="prep/leakage", order=30,
                purposes=("structural",),
                help="P3: verify split disjointness, LOPO folds inside train, leakage ledger")
def run(ctx, cfg: LeakageConfig) -> Dict[str, Any]:
    from hsi_augment import splits
    return splits.run(ctx, cfg)
