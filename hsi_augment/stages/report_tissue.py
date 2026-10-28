# -*- coding: utf-8 -*-
"""
hsi_augment/stages/report_tissue.py
===================================
`augment_pipeline.py report-tissue` (group `report`): Phase 10, issue #34. The
per-row tissue / glass flags of val and test for the report's tissue-vs-glass
breakdown (Plan §Phase 10 "Metrics", analysis only).

Each row is scored with the TRAIN-fitted reference of `tissue-mask`
(`tissue.load_reference(prep/tissue)`: s_glass and tau), without labels: the
stage opens X only. Reading val/test X is the `final_eval` purpose, so this
stage is listed in `splits.FINAL_EVAL_STAGES` (the split-access policy). It is
not part of `all`, and it refuses to read test until `--selection` exists
(the test seal: test is read for the final report only, after selection.json
is frozen); `--splits val` runs before that.

    python augment_pipeline.py report-tissue --data_dir data/hsi_v9-trainsel/hsi \\
                                             --out data/hsi_v9-trainsel-aug
    # -> data/hsi_v9-trainsel-aug/report/tissue/{is_tissue_val.npy, is_tissue_test.npy,
    #    tissue_score_val.npy, tissue_score_test.npy, report_tissue.json}

Row i of each array is row i of `X_<split>.npy`; `scripts/augmentation_report.py`
joins them with the run predictions by row index (`hsi_augment/report.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.io import DEFAULT_CHUNK_ROWS
from hsi_augment.registry import register_stage

REPORT_SPLITS = ("val", "test")


@dataclass(frozen=True)
class ReportTissueConfig(StageConfig):
    splits: Tuple[str, ...] = param(REPORT_SPLITS, help="held-out splits to score (val, test)")
    selection: str = param("reports/augmentation/selection.json",
                           help="selection.json of scripts/augmentation_select.py; test is read only once it "
                                "exists")
    chunk_rows: int = param(DEFAULT_CHUNK_ROWS, help="rows per memmap read")

    def validate(self) -> None:
        if not self.splits or len(set(self.splits)) != len(self.splits) or \
                not set(self.splits) <= set(REPORT_SPLITS):
            raise ConfigError(f"splits must be distinct values of {REPORT_SPLITS}")
        if self.chunk_rows <= 0:
            raise ConfigError("chunk_rows must be > 0")


@register_stage("report-tissue", config=ReportTissueConfig, group="report", dirname="report/tissue", order=100,
                requires=("tissue-mask",), purposes=("final_eval",), in_all=False,
                help="P10: per-row tissue/glass flags of val/test (train-fitted reference, no labels) for "
                     "the report")
def run(ctx, cfg: ReportTissueConfig) -> Dict[str, Any]:
    from hsi_augment import report        # numpy only at import; sklearn/matplotlib load inside
    return report.run_report_tissue(ctx, cfg)
