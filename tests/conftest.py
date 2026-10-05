# -*- coding: utf-8 -*-
"""
tests/conftest.py
====================
MedMamba-SS-TRM v17, Stage 12. Puts the repository root on `sys.path` so the suite imports the
same way it did when every `test_*.py` lived in the root.

Why this file is what makes the move safe
-----------------------------------------
The tests import the code under test as top-level modules - `from training.x import ...`,
`import train_example_v16 as v16`, `import prepare_histologyhsi_bc as v6`. From the root
that resolved for free: pytest's default `prepend` import mode inserts the rootdir at
`sys.path[0]`. From `tests/` it inserts `tests/` instead, and every one of those imports
fails. Three lines here restore it.

It also fixes something that was already broken: `tests/test_group_sidecars.py` does
`from scripts.derive_patch_groups import ...` and `scripts/` has no `__init__.py`, so that
import only ever worked as an implicit namespace package with the root on the path - it
would break under `--import-mode=importlib` or from any other working directory.

`insert(0, ...)` rather than `append`, so a stale same-named package installed in
site-packages cannot shadow the repository being tested.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
