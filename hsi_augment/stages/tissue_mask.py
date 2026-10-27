# -*- coding: utf-8 -*-
"""
hsi_augment/stages/tissue_mask.py
=================================
`augment_pipeline.py tissue-mask` (also run by `preprocess`): Phase 2b,
issue #13. The scale-invariant tissue/glass score t = SAM(patch, s_glass), its
threshold tau (GMM posterior crossing on log t, Otsu fallback), the
pre-registered glass rule, the capture mosaics for the user's sign-off of
tau, and the grid edge-continuity check of the derived coordinates.

Everything it fits is fitted on train rows only: the stage declares the one
purpose "fit", so its SplitGuard refuses val and test outright. The logic and
the output schemas are in `hsi_augment/tissue.py`; the figures in
`hsi_augment/tissue_figures.py`.

    python augment_pipeline.py tissue-mask --data_dir data/hsi_v9-trainsel/hsi \\
                                           --out data/hsi_v9-trainsel-aug
    # -> data/hsi_v9-trainsel-aug/prep/tissue/{s_glass.npy, tau.json, glass_rule.json,
    #    dcis_copy_mask.npy, tissue_{score,mask}_train.npy, grid_check.json, figures/, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.io import DEFAULT_CHUNK_ROWS
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class TissueMaskConfig(StageConfig):
    top_fraction: float = param(0.01, help="brightest fraction of each train capture's patches (by band-mean) "
                                           "whose mean spectrum is that capture's glass candidate")
    method: str = param("gmm", choices=("gmm", "otsu"),
                        help="threshold on log t: 2-component GMM posterior crossing (Otsu if it fails), "
                             "or Otsu only")
    sample_rows: int = param(500_000, help="class x capture stratified train sample for the threshold fit")
    gmm_n_init: int = param(3, help="GMM restarts (sklearn n_init)")
    log_floor_deg: float = param(1e-3, help="t is floored at this before the log (degrees)")
    otsu_bins: int = param(256, help="histogram bins for Otsu's threshold")
    glass_rule_threshold: float = param(0.10, help="pre-registered: a pooled minority glass fraction above "
                                                   "this makes the copy arms copy tissue rows only")
    grid_min_delta: float = param(0.2, help="grid check, per capture and axis: median edge correlation of "
                                            "true neighbours minus a shuffled grid's must reach this")
    grid_min_pass_fraction: float = param(0.95, help="grid check: fraction of captures that must pass")
    grid_baseline_pairs: int = param(2000, help="grid check: minimum shuffled-grid pairs per capture and "
                                                "axis (the capture is re-permuted until there are this many)")
    grid_vertical: bool = param(True, help="grid check: also test (r, c)/(r + 1, c) pairs, the only ones "
                                           "that show the grid width is right")
    grid_strict: bool = param(True, help="fail the stage (no done.json) when the grid check fails")
    mosaics_per_class: int = param(3, help="captures per class in the sign-off mosaics (a minority class "
                                           "also gets one per patient)")
    rgb_nm: Tuple[float, float, float] = param((640.0, 550.0, 460.0),
                                               help="wavelengths of the mosaics' RGB composite (nearest bands)")
    chunk_rows: int = param(DEFAULT_CHUNK_ROWS, help="rows per memmap read (a capture is read in pieces "
                                                     "of at most this)")
    max_captures: Optional[int] = param(None, help="smoke runs only: process N train captures, round-robin "
                                                   "over classes; other rows get t = NaN, mask False, and "
                                                   "the outputs say complete=false (none = all)")

    def validate(self) -> None:
        if not 0.0 < self.top_fraction <= 0.5:
            raise ConfigError("top_fraction must be in (0, 0.5]")
        if self.sample_rows <= 0:
            raise ConfigError("sample_rows must be > 0")
        if self.gmm_n_init < 1:
            raise ConfigError("gmm_n_init must be >= 1")
        if not self.log_floor_deg > 0.0:
            raise ConfigError("log_floor_deg must be > 0")
        if self.otsu_bins < 8:
            raise ConfigError("otsu_bins must be >= 8")
        if not 0.0 <= self.glass_rule_threshold <= 1.0:
            raise ConfigError("glass_rule_threshold must be in [0, 1]")
        if not 0.0 < self.grid_min_delta < 2.0:
            raise ConfigError("grid_min_delta must be in (0, 2)")
        if not 0.0 < self.grid_min_pass_fraction <= 1.0:
            raise ConfigError("grid_min_pass_fraction must be in (0, 1]")
        if self.grid_baseline_pairs < 1:
            raise ConfigError("grid_baseline_pairs must be >= 1")
        if self.mosaics_per_class < 1:
            raise ConfigError("mosaics_per_class must be >= 1")
        if any(not v > 0 for v in self.rgb_nm):
            raise ConfigError("rgb_nm must be three positive wavelengths")
        if self.chunk_rows <= 0:
            raise ConfigError("chunk_rows must be > 0")
        if self.max_captures is not None and self.max_captures <= 0:
            raise ConfigError("max_captures must be > 0 (or none)")


@register_stage("tissue-mask", config=TissueMaskConfig, group="preprocess", dirname="prep/tissue", order=30,
                purposes=("fit",),
                help="P2b: tissue/glass score, tau, glass rule, capture mosaics, grid check")
def run(ctx, cfg: TissueMaskConfig) -> Dict[str, Any]:
    from hsi_augment import tissue         # numpy only; sklearn, scipy, matplotlib, torch load inside
    return tissue.run_tissue_mask(ctx, cfg)
