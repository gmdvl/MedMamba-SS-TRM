# -*- coding: utf-8 -*-
"""
hsi_augment/stages/qc_reference.py
==================================
`augment_pipeline.py qc-reference` (group `qc`): Phase 8b, issue #28. Fits the
QC feature spaces (PCA-64 of the band-z-scored patch, standardised mean/std),
the cross-fitted shallow QC probe, the per-fold class references and the
memorization banks on train TISSUE rows, scores the real LOPO held-out
patches exactly as generated samples will be scored, and calibrates the
thresholds of rules R1-R6 so each accepts >= 99% and all together >= 95% of
those real patches (or fails, with no done.json).

It declares the one purpose "qc", so its SplitGuard refuses val and test; it
reads the train split, the tissue mask (`prep/tissue`) and the LOPO folds
(`prep/leakage`). The logic and the output schemas are in
`hsi_augment/qc/reference.py`; #29 and #30 load the result with
`hsi_augment.qc.QCReference.load`.

    python augment_pipeline.py qc-reference --data_dir data/hsi_v9-trainsel/hsi \\
                                            --out data/hsi_v9-trainsel-aug
    # -> data/hsi_v9-trainsel-aug/qc/reference/{thresholds.json, qc_reference.json, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class QCReferenceConfig(StageConfig):
    target_classes: Tuple[str, ...] = param((), help="classes to build QC references for (none = the "
                                                     "adapter's minority class, DCIS)")
    target_rows: Optional[int] = param(None, help="cap on a target class's tissue rows in the reference "
                                                  "sample (none = all; they are also the memorization bank)")
    rows_per_class: int = param(50_000, help="seeded tissue rows of every other class in the reference sample "
                                             "(kNN bank and probe training)")
    pca_components: int = param(64, help="PCA components of the flattened band-z-scored patch")
    pca_fit_rows: int = param(30_000, help="reference-sample rows the PCA covariance is accumulated over")
    lr_rows_per_class: int = param(20_000, help="rows per class the mean/std QC probe is fitted on")
    lr_max_iter: int = param(1500, help="LogisticRegression max_iter (shallow_probe_heldout.py's default)")
    knn_k: int = param(10, help="R5: neighbours of the kNN label agreement")
    nn_candidates: int = param(10, help="R6: PCA candidates re-ranked with the exact distance")
    hash_all_rows: bool = param(True, help="R6: hash every train row (one extra pass over X_train); off = "
                                           "the reference sample only (smoke runs)")
    r1_band_tail: float = param(0.0005, help="R1: per-band range = [tail, 1 - tail] quantiles of train "
                                             "target tissue values (p0.05, p99.95)")
    r1_max_out_fraction: float = param(0.01, help="R1: largest allowed fraction of values outside the band "
                                                  "range (calibration may only raise it)")
    r1_level: float = param(0.01, help="R1: nominal tail level of the real out-of-range fraction")
    r2_level: float = param(0.005, help="R2: nominal tail level (Mahalanobis q99.5; curvature both sides)")
    r3_level: float = param(0.005, help="R3: nominal tail level on each side ([q0.5, q99.5])")
    r5_level: float = param(0.05, help="R5: nominal tail level (p(class) and kNN agreement >= q5)")
    r6_level: float = param(0.01, help="R6: nominal tail level (NN distance >= q1 of cross-capture NN)")
    hf_cutoff: float = param(0.25, help="R3: radial DCT-II frequency (cycles/pixel) above which power is 'high'")
    gauss_ridge: float = param(1e-4, help="R2: covariance ridge, as a fraction of the mean variance")
    rule_accept: float = param(0.99, help="calibration: each rule must accept this fraction of real held-out")
    joint_accept: float = param(0.95, help="calibration: all rules together must accept this fraction")
    max_halvings: int = param(10, help="calibration: a level is halved at most this often before 0")
    prdc_k: int = param(5, help="set metrics: k of precision/recall/density/coverage")
    prdc_rows: int = param(5000, help="set metrics: rows per side for PRDC (seeded subsample)")
    mmd_rows: int = param(2000, help="set metrics: rows per side for MMD (seeded subsample)")
    bootstrap: int = param(200, help="set metrics: bootstrap resamples for the CIs")
    attribution_flag: float = param(0.4, help="set metrics: flag when one patient takes more than this "
                                              "share of nearest-neighbour attributions")
    device: str = param("auto", choices=("auto", "cpu", "cuda"), help="torch device of the NN searches")
    chunk_rows: int = param(4096, help="train rows per memmap read")
    score_chunk_rows: int = param(1024, help="samples scored at a time")
    query_chunk: int = param(512, help="NN queries per exact re-rank block")
    block_elems: int = param(1 << 24, help="largest distance block (elements) of a cdist call")

    def validate(self) -> None:
        if self.target_rows is not None and self.target_rows < 10:
            raise ConfigError("target_rows must be >= 10 (or none)")
        for name in ("rows_per_class", "pca_components", "pca_fit_rows", "lr_rows_per_class", "lr_max_iter",
                     "knn_k", "nn_candidates", "prdc_k", "prdc_rows", "mmd_rows", "chunk_rows",
                     "score_chunk_rows", "query_chunk", "block_elems"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be > 0")
        if self.bootstrap < 0 or self.max_halvings < 0:
            raise ConfigError("bootstrap and max_halvings must be >= 0")
        if not 0.0 <= self.r1_band_tail < 0.5:
            raise ConfigError("r1_band_tail must be in [0, 0.5)")
        for name in ("r1_max_out_fraction", "r1_level", "r2_level", "r3_level", "r5_level", "r6_level",
                     "attribution_flag"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ConfigError(f"{name} must be in [0, 1)")
        if not 0.0 < self.hf_cutoff < 0.5 * 2 ** 0.5:
            raise ConfigError("hf_cutoff must be in (0, 0.707) cycles/pixel")
        if self.gauss_ridge < 0:
            raise ConfigError("gauss_ridge must be >= 0")
        if not (0.0 < self.rule_accept <= 1.0 and 0.0 < self.joint_accept <= 1.0):
            raise ConfigError("rule_accept and joint_accept must be in (0, 1]")


@register_stage("qc-reference", config=QCReferenceConfig, group="qc", dirname="qc/reference", order=80,
                purposes=("qc",), requires=("tissue-mask", "leakage"),
                help="P8b: QC feature spaces, cross-fitted QC probe, real LOPO references, calibrated thresholds")
def run(ctx, cfg: QCReferenceConfig) -> Dict[str, Any]:
    from hsi_augment.qc import reference     # numpy at import; torch, scipy, sklearn load inside
    return reference.run_qc_reference(ctx, cfg)
