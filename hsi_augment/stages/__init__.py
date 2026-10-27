# -*- coding: utf-8 -*-
"""
hsi_augment/stages/
===================
One small module per pipeline stage. `registry.load_builtin_stages()` imports
every module here (sorted by name; names starting with "_" are skipped), and
each module's `@register_stage(...)` makes it an `augment_pipeline.py`
subcommand. Adding a stage = adding a file; no shared list to edit.

Keep these modules light: a config dataclass, the registration, and a run
function that imports its heavy logic (torch, sklearn, the phase module)
inside the function, so `augment_pipeline.py --help` stays fast.
"""
