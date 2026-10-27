# -*- coding: utf-8 -*-
"""
hsi_augment/
============
The HSI augmentation pipeline (plan/HSI_augmentation_pipeline.plan.md, epic #1):
data validation, train-only preprocessing, the generator pool, conventional and
generative augmentation, QC and export. The CLI is `augment_pipeline.py` in the
repository root.

The package imports frozen code and never edits it (Plan §Non-negotiable
engineering rules, 1). Its core, which every phase builds on:

  config.py     one frozen dataclass per stage: JSON round trip, `config_hash()`,
                flags generated from the fields, `--config file.json` defaults
  stage.py      the stage-directory contract: `done.json` written atomically and
                last, skip-if-done, `--force`, `--resume auto`, `stage.log`,
                `metrics.jsonl`, per-stage seeds
  registry.py   `@register_stage(...)`: how a stage becomes a CLI subcommand
  io.py         memmap chunk iterators, float64 running band stats, `SplitGuard`,
                atomic writes, disk and RAM preflight
  adapters/     the build layout (paths, class names, minority class, sidecars,
                capture grid coordinates) behind a thin `BuildAdapter`
  stages/       one small file per stage; every module here is imported by the CLI

Importing this package is cheap (numpy only). Heavy imports (torch, sklearn)
belong inside the stage functions that need them, so `--help` stays instant.
"""

SCHEMA_VERSION = 1
