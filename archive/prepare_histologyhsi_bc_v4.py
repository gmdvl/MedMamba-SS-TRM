# -*- coding: utf-8 -*-
"""
prepare_histologyhsi_bc_v4.py
==============================
improve-promp_v8.plan.md, Phase 0.1 names this file as the dataset-prep
script to audit. The repository's actual, up-to-date prep script is
`prepare_histologyhsi_bc_v3.py` (there is no separate "v4" implementation -
band selection, class balancing, dataset-integrity checks, and now the
Phase 0.1 `dataset_diagnostics.json` audit all already live there).

This module is a thin compatibility shim: it re-exports everything from
`prepare_histologyhsi_bc_v3` so anything invoking this file by name (as the
plan does) transparently runs the real, current implementation instead of
silently doing nothing / erroring on "file not found". Prefer importing
`prepare_histologyhsi_bc_v3` directly in new code - this file exists only
for plan/tooling name compatibility.
"""

from prepare_histologyhsi_bc_v3 import *  # noqa: F401,F403
from prepare_histologyhsi_bc_v3 import main  # noqa: F401

if __name__ == "__main__":
    main()
