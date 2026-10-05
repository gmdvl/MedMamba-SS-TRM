# -*- coding: utf-8 -*-
"""
training.prep - generic, dataset-agnostic preparation pipeline.

A concrete `prepare_*.py` subclasses `DatasetAdapter` (format-specific:
discovery + one item -> sample arrays) and calls `run_prep(adapter)`; the
core owns splitting, class-coverage gating, parallel extraction, spanning
shards, crash-safe unify/finalize, and provenance.
"""

from training.prep.adapter import DatasetAdapter, ModalitySpec, PrepItem
from training.prep.core import build_parser, run_prep

__all__ = ["DatasetAdapter", "ModalitySpec", "PrepItem", "run_prep", "build_parser"]
