# -*- coding: utf-8 -*-
"""
training/prep/adapter.py
========================
The dataset-agnostic contract between a raw dataset and the generic prep
core (`training.prep.core.run_prep`). A concrete `prepare_*.py` script
subclasses `DatasetAdapter` and implements only what is *format-specific*
(how to find items, how to turn one item into sample arrays); the core
owns everything else - splitting, coverage checks, parallel extraction,
sharding, crash-safe unify/finalize, provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PrepItem:
    """One labelled unit of raw data (a PAD image, an HSI capture)."""
    key: str                                  # globally-unique, stable: resume + deterministic-order key
    group_id: str                             # split grouping key (patient); never empty
    label: int                                # class id in range(len(adapter.class_names))
    payload: Mapping[str, Any] = field(default_factory=dict)   # small + picklable (paths, not arrays)
    strat_keys: Tuple = ()                    # optional extra stratification keys (hashable)


@dataclass(frozen=True)
class ModalitySpec:
    """How the core should store one modality's samples."""
    store_dtype: str = "float32"             # "float32" | "float16" | "uint8"
    value_clip: Optional[Tuple[float, float]] = None   # applied (np.clip) before the dtype cast
    sample_shape: Optional[Tuple[int, ...]] = None     # (H, W, C) if fixed - needed for --writer direct
    extra_outputs: Tuple[str, ...] = ()      # basenames the adapter writes into the modality dir


class DatasetAdapter(ABC):
    """Implement the abstract bits; override the hooks you need."""

    name: str = "dataset"
    #: True  -> the (single) modality's arrays go straight into `out_dir`
    #: False -> each modality gets its own `out_dir/<modality>/` subdir
    flat_output: bool = False

    # ---- static description ------------------------------------------------
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        ...

    def modalities(self) -> List[str]:
        return ["image"]

    def compat_keys(self, args) -> Dict[str, Any]:
        """Settings that MUST match to resume an existing `_progress.json`
        (they change what bytes go into a shard or which item goes where)."""
        return {}

    def add_cli_args(self, parser) -> None:
        """Register adapter-specific flags on the shared arg parser."""
        return None

    def preprocess_args(self, args) -> None:
        """Called once, right after arg parsing - normalise / cross-populate
        adapter flags (e.g. map a deprecated --test_size onto --split)."""
        return None

    # ---- discovery (parent process; cheap, no pixel data) ---------------
    @abstractmethod
    def discover(self, args) -> Iterable[PrepItem]:
        ...

    # ---- optional global pre-pass (e.g. spectral band selection) --------
    def pass1(self, items: Sequence[PrepItem], args) -> Optional[dict]:
        return None

    def write_pass1_outputs(self, state: Optional[dict], out_dir: str) -> List[str]:
        """Write any global Pass-1 artefacts (e.g. selected_band_indices.npy)
        into `out_dir`. Return the basenames written."""
        return []

    def finalize_modality(self, modality: str, modality_dir: str,
                           state: Optional[dict], manifest: dict) -> List[str]:
        """After a modality's X/y are unified, write per-modality extras
        (e.g. wavelengths.npy) into `modality_dir`. Return basenames written
        (they are added to that modality's dataset_manifest.json)."""
        return []

    # ---- per-item extraction (runs INSIDE a worker) --------------------
    def modality_spec(self, modality: str, args, state: Optional[dict]) -> ModalitySpec:
        return ModalitySpec()

    @abstractmethod
    def load_item(self, item: PrepItem, modality: str, args, state: Optional[dict]
                  ) -> Iterator[Tuple[np.ndarray, int]]:
        """Yield `(sample_HWC_float32, label)` one or more times. ALL raw IO /
        decode / calibrate / patch-grid logic lives here. Bounded memory:
        yield, never accumulate."""
        ...

    def load_item_multi(self, item: PrepItem, modalities: List[str], args, state: Optional[dict]
                        ) -> Optional[Dict[str, Iterator[Tuple[np.ndarray, int]]]]:
        """Optional power hook: compute shared per-item state once (e.g. HSI
        patch coordinates) and return a `{modality: iterator}` mapping. Return
        None to let the core call `load_item` per modality."""
        return None

    # ---- optional -----------------------------------------------------
    def expected_sample_count(self, item: PrepItem, modality: str, args, state: Optional[dict]
                               ) -> Optional[int]:
        """Number of samples `load_item` will yield, if known WITHOUT reading
        pixels. Non-None for every item + a fixed `ModalitySpec.sample_shape`
        enables `--writer direct`."""
        return None

    def provenance(self, args, state: Optional[dict]) -> Dict[str, Any]:
        return {}
