# -*- coding: utf-8 -*-
"""
hsi_augment/stages/generate.py
==============================
`augment_pipeline.py generate` and the generated-sample half of `qc`
(Phase 8a and 8b, issue #29): per source (cvae, ldm), one stage that
generates and one that quality-controls and selects:

    generate-cvae  [group generate]  -> <out>/synth/cvae/  requires cvae-lopo, cvae-final, qc-reference
    generate-ldm   [group generate]  -> <out>/synth/ldm/   requires ldm-lopo, ldm-final, qc-reference
    qc-cvae        [group qc]        -> <out>/qc/cvae/     requires generate-cvae, qc-reference
    qc-ldm         [group qc]        -> <out>/qc/ldm/      requires generate-ldm, qc-reference

    python augment_pipeline.py generate-cvae qc-cvae export-cvae --data_dir data/hsi_v9-trainsel/hsi \\
                                                                  --out data/hsi_v9-trainsel-aug
    python augment_pipeline.py generate-cvae ... --tau_s 0.8        # skip the LOPO QC choice

The target (exactly as many accepted rows per target class as the train split
has real ones: 122,850 DCIS on trainsel), the 2x over-generation, the LOPO QC
choice of tau_s / w, the selection (the first n_target accepted, in
generation order) and the 40% acceptance floor are described in
`hsi_augment/generate.py`, which holds the logic.

Split access, declared here: generate-* reads the train labels (purpose
"generator") and the train patches a QC reference re-ranks against
("qc"); qc-* only the latter. Both purposes are train-only.

All four stages stay out of `all` (`in_all=False`), like the generator fits
they depend on: `all` cannot produce a generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.registry import register_stage

DEVICES = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class GenerateConfig(StageConfig):
    """Settings both generators share."""
    target_classes: str = param("auto", help="classes to generate: 'auto' (resolve_minority_classes on the train "
                                             "counts: DCIS) or a comma list of names/indices")
    n_per_class: Optional[int] = param(None, help="accepted rows to export per target class (none = the train "
                                                  "split's count of that class, 1:1; set only for smoke runs)")
    overgenerate: float = param(2.0, help="rows generated per target row (2x: 245,700 for 122,850)")
    min_acceptance: float = param(0.4, help="LOPO QC choice: a tau_s / w candidate needs at least this mean "
                                            "acceptance")
    choice_rows: int = param(2048, help="LOPO QC choice: samples per (fold, class, candidate)")
    batch_size: int = param(1024, help="rows per generator call (the LDM's samples depend on it; the CVAE's do not)")
    qc_chunk_rows: int = param(4096, help="samples scored per QC call during the LOPO choice")
    device: str = param("auto", choices=DEVICES, help="generator and QC device; auto = cuda if available")

    def validate(self) -> None:
        if self.n_per_class is not None and self.n_per_class < 1:
            raise ConfigError("n_per_class must be >= 1 (or none)")
        if not self.overgenerate >= 1.0:
            raise ConfigError("overgenerate must be >= 1")
        if not 0.0 <= self.min_acceptance <= 1.0:
            raise ConfigError("min_acceptance must be in [0, 1]")
        for name in ("choice_rows", "batch_size", "qc_chunk_rows"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")
        if not str(self.target_classes).strip():
            raise ConfigError("target_classes must be 'auto' or a comma list of classes")


@dataclass(frozen=True)
class GenerateCVAEConfig(GenerateConfig):
    tau_s: Optional[float] = param(None, help="prior temperature (z = tau_s * eps); none = model.json's tau_s if "
                                              "set, else chosen on LOPO QC among its candidates")

    def validate(self) -> None:
        super().validate()
        if self.tau_s is not None and not self.tau_s > 0:
            raise ConfigError("tau_s must be > 0 (or none)")


@dataclass(frozen=True)
class GenerateLDMConfig(GenerateConfig):
    guidance: Optional[float] = param(None, help="classifier-free guidance scale w in [0, 3]; none = chosen on "
                                                 "LOPO QC among generator.pt's candidates")
    sample_steps: Optional[int] = param(None, help="DDIM steps, eta = 0 (none = the generator's)")

    def validate(self) -> None:
        super().validate()
        if self.guidance is not None and not 0.0 <= self.guidance <= 3.0:
            raise ConfigError("guidance must be in [0, 3] (w > 3 pulls samples to the training modes)")
        if self.sample_steps is not None and self.sample_steps < 1:
            raise ConfigError("sample_steps must be >= 1 (or none)")


@dataclass(frozen=True)
class QCApplyConfig(StageConfig):
    min_acceptance: float = param(0.4, help="fail when a class's QC acceptance is below this (the plan's 40%)")
    chunk_rows: int = param(4096, help="generated samples read and scored at a time")
    bootstrap: Optional[int] = param(None, help="set metrics: bootstrap resamples (none = qc-reference's)")
    device: str = param("auto", choices=DEVICES, help="device of the nearest-neighbour searches")

    def validate(self) -> None:
        if not 0.0 <= self.min_acceptance <= 1.0:
            raise ConfigError("min_acceptance must be in [0, 1]")
        if self.chunk_rows < 1:
            raise ConfigError("chunk_rows must be >= 1")
        if self.bootstrap is not None and self.bootstrap < 0:
            raise ConfigError("bootstrap must be >= 0 (or none)")


@register_stage("generate-cvae", config=GenerateCVAEConfig, group="generate", dirname="synth/cvae", order=90,
                requires=("cvae-lopo", "cvae-final", "qc-reference"), purposes=("generator", "qc"), in_all=False,
                help="P8a: CVAE samples, 2x the target, tau_s chosen on LOPO QC")
def run_generate_cvae(ctx, cfg: GenerateCVAEConfig) -> Dict[str, Any]:
    from hsi_augment import generate
    return generate.run_generate(ctx, cfg, "cvae")


@register_stage("generate-ldm", config=GenerateLDMConfig, group="generate", dirname="synth/ldm", order=90,
                requires=("ldm-lopo", "ldm-final", "qc-reference"), purposes=("generator", "qc"), in_all=False,
                help="P8a: latent diffusion samples, 2x the target, w chosen on LOPO QC")
def run_generate_ldm(ctx, cfg: GenerateLDMConfig) -> Dict[str, Any]:
    from hsi_augment import generate
    return generate.run_generate(ctx, cfg, "ldm")


@register_stage("qc-cvae", config=QCApplyConfig, group="qc", dirname="qc/cvae", order=91,
                requires=("generate-cvae", "qc-reference"), purposes=("qc",), in_all=False,
                help="P8b: R1-R6 on every CVAE sample, first n_target accepted, set metrics; fails below 40%")
def run_qc_cvae(ctx, cfg: QCApplyConfig) -> Dict[str, Any]:
    from hsi_augment import generate
    return generate.run_qc(ctx, cfg, "cvae")


@register_stage("qc-ldm", config=QCApplyConfig, group="qc", dirname="qc/ldm", order=91,
                requires=("generate-ldm", "qc-reference"), purposes=("qc",), in_all=False,
                help="P8b: R1-R6 on every LDM sample, first n_target accepted, set metrics; fails below 40%")
def run_qc_ldm(ctx, cfg: QCApplyConfig) -> Dict[str, Any]:
    from hsi_augment import generate
    return generate.run_qc(ctx, cfg, "ldm")
