# -*- coding: utf-8 -*-
"""
archive/gmedmamba.py
====================
`gmedmamba.py` was renamed to `medmamba_ss_trm.py` on 2026-10-02, and its names
with it: the hierarchical model is MedMamba-SS (`MedMambaSS`), the recursive one
is MedMamba-SS-TRM (`MedMambaSSTRM`).

The frozen scripts in this directory still `from gmedmamba import ...`. They run
with `archive/` on the path (`PYTHONPATH=.:archive`), so this module answers that
import: every name of `medmamba_ss_trm`, plus the old names in `OLD_NAMES`, each
bound to the very same object. Nothing live imports it.
"""

import medmamba_ss_trm as _new
from medmamba_ss_trm import *  # noqa: F401,F403

# old name -> name in medmamba_ss_trm
OLD_NAMES = {
    "GMedMambaConfig": "MedMambaSSTRMConfig",
    "GMedMamba": "MedMambaSS",
    "GMedMambaBackbone": "MedMambaSSBackbone",
    "GMedMambaRecursive": "MedMambaSSTRM",
    "GMedMambaRecursiveBackbone": "MedMambaSSTRMBackbone",
    "gmedmamba_tiny": "medmamba_ss_tiny",
    "gmedmamba_small": "medmamba_ss_small",
    "gmedmamba_base": "medmamba_ss_base",
    "gmedmamba_hsi_small": "medmamba_ss_hsi_small",
    "gmedmamba_research": "medmamba_ss_research",
    "gmedmamba_sensor_aware": "medmamba_ss_sensor_aware",
    "gmedmamba_trm": "medmamba_ss_trm_default",
}

globals().update({old: getattr(_new, new) for old, new in OLD_NAMES.items()})
