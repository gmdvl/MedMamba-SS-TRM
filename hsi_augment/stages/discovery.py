# -*- coding: utf-8 -*-
"""
hsi_augment/stages/discovery.py
===============================
`augment_pipeline.py discovery` (or the group `validate`): Phase 1, dataset
discovery and validation (#11). The logic is in `hsi_augment/discovery.py`;
this module is the configuration and the registration.

It is the one stage besides the final evaluation that reads val and test, and
it reads them for structure only: its single declared purpose is
"structural", so the SplitGuard refuses anything else and logs every access.

Outputs in `<out>/prep/validate/`: `discovery_report.json` (inventory) and
`integrity_report.json` (named checks, pass / warn / fail / skip). Any "fail"
raises StageError after both reports are written: the CLI exits 1 and no
done.json appears, so nothing downstream runs on a build that failed P1.

A quick look at a build without reading X (headers, JSON and the y / groups /
captures sidecars only):

    python augment_pipeline.py discovery --data_dir data/hsi_v9-trainsel/hsi --out /tmp/p1 \\
        --verify_sha256 off --content_scan off --leakage_rows 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.io import DEFAULT_CHUNK_ROWS
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class DiscoveryConfig(StageConfig):
    verify_sha256: bool = param(True, help="re-hash every manifest file against its recorded sha256 (a streamed "
                                           "read of the whole build: minutes on the 19 GB X_train)")
    content_scan: bool = param(True, help="read every X row: NaN/Inf/float16-saturation/all-zero counts and "
                                          "per-band min/max/mean/std/percentiles per split and class")
    scan_max_rows: Optional[int] = param(None, help="scan only the first N rows of each split (none = all); a "
                                                    "partial scan makes content.nonfinite a warning, not a pass")
    chunk_rows: int = param(DEFAULT_CHUNK_ROWS, help="rows per memmap chunk")
    percentiles: Tuple[float, ...] = param((0.05, 99.95), help="per-band percentiles, in percent (exact on "
                                                               "float16 builds)")
    leakage_rows: Optional[int] = param(None, help="rows per split hashed for the duplicate-patch check: none = "
                                                   "every row (hashed during the content scan, no extra read), "
                                                   "N = a seeded random sample, 0 = off")
    expected_bands: Optional[int] = param(None, help="required number of bands (none = the build's own: X's "
                                                     "band axis and the manifest's num_bands)")

    def validate(self) -> None:
        if self.chunk_rows <= 0:
            raise ConfigError("chunk_rows must be > 0")
        if self.scan_max_rows is not None and self.scan_max_rows <= 0:
            raise ConfigError("scan_max_rows must be > 0 (or none; content_scan off skips the scan)")
        if self.leakage_rows is not None and self.leakage_rows < 0:
            raise ConfigError("leakage_rows must be >= 0 (or none)")
        if self.expected_bands is not None and self.expected_bands <= 0:
            raise ConfigError("expected_bands must be > 0 (or none)")
        if not self.percentiles or any(not 0.0 <= p <= 100.0 for p in self.percentiles):
            raise ConfigError("percentiles must be one or more values in [0, 100]")


@register_stage("discovery", config=DiscoveryConfig, group="validate", dirname="prep/validate", order=10,
                purposes=("structural",),
                help="P1: inventory, manifest sha256, headers, provenance, content scan, duplicate patches")
def run(ctx, cfg: DiscoveryConfig) -> Dict[str, Any]:
    from hsi_augment import discovery
    return discovery.run(ctx, cfg)
