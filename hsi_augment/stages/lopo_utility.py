# -*- coding: utf-8 -*-
"""
hsi_augment/stages/lopo_utility.py
==================================
`augment_pipeline.py lopo-utility` (Phase 8c, issue #30). In each LOPO fold, a
mean/std logistic regression is trained on four sets: real only, real +
count-matched copies, real + CVAE, and real + LDM (the fold's own generators,
QC'd against the fold's references). Each is scored on the held-out DCIS
patient (recall, F1) and on the fold (macro-F1). The generator with the better
mean LOPO DCIS F1 becomes A4's source:

    lopo-utility  -> <out>/lopo_utility/decision.json   {"a4_source": "cvae" | "ldm", ...}
                     requires leakage, pool, qc-reference, cvae-lopo, cvae-final, ldm-lopo,
                              generate-cvae, generate-ldm

    python augment_pipeline.py lopo-utility --data_dir data/hsi_v9-trainsel/hsi --out data/hsi_v9-trainsel-aug

The real rows are the pool's (prep/leakage folds over `patient_pool.npy`),
read from X_train in value space. The synthetic rows are sampled at the
tau_s / w in `synth/<src>/choice.json`. The count matching, the classifier,
the metrics, the decision rule with its tie-break, and the decision.json schema
are described in `hsi_augment/lopo_utility.py`. The sweep
(`sweeps/augmentation_pipeline.py`) and the report (`hsi_augment/report.py`)
read `a4_source`.

Split access, declared here in one line: the purpose "lopo" only (train), so
the SplitGuard refuses val and test. The stage stays out of `all`
(`in_all=False`), like the generator fits and the generate stages it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.registry import register_stage

DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class LopoUtilityConfig(StageConfig):
    target_classes: str = param("auto", help="target classes: 'auto' (resolve_minority_classes on the train "
                                             "counts: DCIS) or a comma list of names/indices")
    syn_ratio: float = param(1.0, help="rows added per fold and target class, synthetic and copied alike, as a "
                                       "multiple of the fold's train-side pool rows of that class (1.0 = 1:1, "
                                       "the ratio of arms A2/A3)")
    max_added_rows: Optional[int] = param(None, help="cap on those rows per (fold, class) (none = no cap; set "
                                                     "only to shorten a run)")
    overgenerate: float = param(2.0, help="at most this many samples generated per row to add "
                                          "(generate-<src>'s 2x)")
    min_acceptance: float = param(0.4, help="fail when a fold generator's QC acceptance is below this "
                                            "fraction (the plan's 40 percent)")
    max_iter: int = param(1500, help="LogisticRegression max_iter (shallow_probe_heldout.py's default)")
    tie_tolerance: float = param(0.0, help="mean DCIS F1 within this of each other is a tie, which goes to cvae "
                                           "(the cheaper generator)")
    chunk_rows: int = param(4096, help="train rows read and featurised at a time")
    qc_chunk_rows: int = param(4096, help="samples scored per QC call")
    device: str = param("auto", choices=DEVICES, help="generator and QC device; auto = cuda if available")

    def validate(self) -> None:
        if not self.syn_ratio > 0:
            raise ConfigError("syn_ratio must be > 0")
        if self.max_added_rows is not None and self.max_added_rows < 1:
            raise ConfigError("max_added_rows must be >= 1 (or none)")
        if not self.overgenerate >= 1.0:
            raise ConfigError("overgenerate must be >= 1")
        if not 0.0 <= self.min_acceptance <= 1.0:
            raise ConfigError("min_acceptance must be in [0, 1]")
        if not self.tie_tolerance >= 0.0:
            raise ConfigError("tie_tolerance must be >= 0")
        for name in ("max_iter", "chunk_rows", "qc_chunk_rows"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")
        if not str(self.target_classes).strip():
            raise ConfigError("target_classes must be 'auto' or a comma list of classes")


@register_stage("lopo-utility", config=LopoUtilityConfig, dirname="lopo_utility", order=93,
                requires=("leakage", "pool", "qc-reference", "cvae-lopo", "cvae-final", "ldm-lopo",
                          "generate-cvae", "generate-ldm"),
                purposes=("lopo",), in_all=False,
                help="P8c: real vs copies vs CVAE vs LDM on unseen DCIS patients (LOPO, train only); "
                     "decides A4's generator")
def run(ctx, cfg: LopoUtilityConfig) -> Dict[str, Any]:
    from hsi_augment import lopo_utility     # torch, sklearn and the QC library load inside
    return lopo_utility.run_lopo_utility(ctx, cfg)
