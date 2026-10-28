# -*- coding: utf-8 -*-
"""
hsi_augment/stages/export.py
============================
`augment_pipeline.py export` (Phase 9, issue #29): the training export of
each source, read by `train.py --synthetic_dir` through
`training.synthetic_mix.load_synthetic_export`:

    export-cvae  [group export]  -> <out>/export/cvae/  requires generate-cvae, qc-cvae, qc-reference
    export-ldm   [group export]  -> <out>/export/ldm/   requires generate-ldm, qc-ldm, qc-reference

    python augment_pipeline.py export-cvae --data_dir data/hsi_v9-trainsel/hsi --out data/hsi_v9-trainsel-aug
    python train.py ... --synthetic_dir data/hsi_v9-trainsel-aug/export/cvae

The export holds exactly the selection of qc-<src> (per target class, as
many rows as the train split has real ones: 122,850 DCIS on trainsel), in the
schema of `training/synthetic_mix.py`, written by its `write_synthetic_export`;
plus qc_scores.npz (every generated sample), qc_report.json and the visual QC
figures. Layout and provenance keys: `hsi_augment/export.py`.

`hsi_augment/report.py` reads `export/<src>/provenance.json` (`export_id`) for
its cost table and to check that a run trained on the current export.

Split access: the figures read real train patches with the train-only
purpose "qc". The stages stay out of `all`, like the generators they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class ExportConfig(StageConfig):
    figures: bool = param(True, help="write the visual QC figures into <export>/figures/")
    figure_samples: int = param(12, help="synthetic (and real reference) patches per contact sheet and gallery")

    def validate(self) -> None:
        if self.figure_samples < 1:
            raise ConfigError("figure_samples must be >= 1")


@register_stage("export-cvae", config=ExportConfig, group="export", dirname="export/cvae", order=92,
                requires=("generate-cvae", "qc-cvae", "qc-reference"), purposes=("qc",), in_all=False,
                help="P9: the CVAE training export (synthetic_mix schema), QC report and figures")
def run_export_cvae(ctx, cfg: ExportConfig) -> Dict[str, Any]:
    from hsi_augment import export
    return export.run_export(ctx, cfg, "cvae")


@register_stage("export-ldm", config=ExportConfig, group="export", dirname="export/ldm", order=92,
                requires=("generate-ldm", "qc-ldm", "qc-reference"), purposes=("qc",), in_all=False,
                help="P9: the LDM training export (synthetic_mix schema), QC report and figures")
def run_export_ldm(ctx, cfg: ExportConfig) -> Dict[str, Any]:
    from hsi_augment import export
    return export.run_export(ctx, cfg, "ldm")
