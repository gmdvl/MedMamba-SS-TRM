# -*- coding: utf-8 -*-
"""
hsi_augment/adapters/histologyhsi_bc.py
=======================================
HistologyHSI-BC builds from `prepare_histologyhsi_bc*.py` (e.g.
`data/hsi_v9-trainsel`). The dataset facts, verified on trainsel from the JSON
files and `.npy` headers (Plan, "Ground truth the plan relies on"):

  * X float16 NHWC (N, 11, 11, 32): patch 11, stride 11.
  * Classes healthy / DCIS / IDC; DCIS is the minority class (5.0% of train
    patches, 5 train patients).
  * Captures are cut row-major into a 91-column grid. All 499 train captures
    (and every test capture) hold 4,914 = 54 x 91 patches; 10 val captures of
    patient 65 hold 2,002 = 22 x 91. Coordinates: `(r, c) = divmod(ordinal, 91)`.
  * capture_index.json and split_assignment.json spell the validation split
    "validation"; the array files say "_val". `canonical_split` maps both to "val".
  * Build-root sidecars: split_assignment.json (patient -> split),
    pass1_scope_report.json (Pass 1 fitted on train captures only),
    selected_wavelengths.npy (equals hsi/wavelengths.npy), capture_gain_report.json
    (per-capture median-ratio gain; patient 68's captures are flagged as clipped).
  * Values are raw counts x gain / corpus median (8640.0), not reflectance.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, Tuple

import numpy as np

from hsi_augment.adapters.base import BuildAdapter
from hsi_augment.io import SPLITS, canonical_split


class HistologyHSIBCAdapter(BuildAdapter):
    name = "histologyhsi_bc"
    GRID_COLS = 91
    STRIDE = 11
    FULL_GRID = (54, 91)
    MINORITY_CLASS_NAMES = ("DCIS",)
    BUILD_SIDECARS = {
        "split_assignment": "split_assignment.json",
        "pass1_scope_report": "pass1_scope_report.json",
        "selected_wavelengths": "selected_wavelengths.npy",
        "capture_gain_report": "capture_gain_report.json",
    }

    @cached_property
    def split_assignment(self) -> Dict[str, Any]:
        return self.sidecar("split_assignment")

    def split_patients(self) -> Dict[str, Tuple[int, ...]]:
        """{train/val/test: sorted patient ids} from split_assignment.json."""
        out: Dict[str, list] = {s: [] for s in SPLITS}
        for patient, split in (self.split_assignment.get("groups") or {}).items():
            out[canonical_split(split)].append(int(patient))
        return {s: tuple(sorted(v)) for s, v in out.items()}

    @cached_property
    def pass1_scope(self) -> Dict[str, Any]:
        return self.sidecar("pass1_scope_report")

    def pass1_train_patients(self) -> Tuple[int, ...]:
        """Patients Pass 1 (band selection, gain, value scale) was fitted on."""
        return tuple(sorted(int(p) for p in self.pass1_scope.get("train_patients", [])))

    @cached_property
    def selected_wavelengths(self) -> np.ndarray:
        return self.sidecar("selected_wavelengths")

    @cached_property
    def capture_gain_report(self) -> Dict[str, Any]:
        return self.sidecar("capture_gain_report")
