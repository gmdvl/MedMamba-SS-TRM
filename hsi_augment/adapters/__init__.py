# -*- coding: utf-8 -*-
"""
hsi_augment/adapters/
=====================
Build adapters by name. The pipeline is HistologyHSI-BC only (Plan, "Dataset
scope"); another dataset would add one subclass of `BuildAdapter` and one entry
in `ADAPTERS`.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from hsi_augment.adapters.base import (BuildAdapter, CaptureInfo, CaptureRun, GridCoordinates,
                                       NonContiguousCaptureError, resolve_build_dirs, spectral_gaps,
                                       spectral_segments)
from hsi_augment.adapters.histologyhsi_bc import HistologyHSIBCAdapter

ADAPTERS: Dict[str, Type[BuildAdapter]] = {HistologyHSIBCAdapter.name: HistologyHSIBCAdapter}
DEFAULT_ADAPTER = HistologyHSIBCAdapter.name


def get_adapter(name: str, data_dir: Any, **kwargs: Any) -> BuildAdapter:
    try:
        cls = ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown adapter {name!r}; known: {sorted(ADAPTERS)}") from None
    return cls(data_dir, **kwargs)


__all__ = ["ADAPTERS", "DEFAULT_ADAPTER", "BuildAdapter", "CaptureInfo", "CaptureRun", "GridCoordinates",
           "HistologyHSIBCAdapter", "NonContiguousCaptureError", "get_adapter", "resolve_build_dirs",
           "spectral_gaps", "spectral_segments"]
