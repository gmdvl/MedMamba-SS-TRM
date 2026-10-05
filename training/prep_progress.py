# -*- coding: utf-8 -*-
"""
training/prep_progress.py
================================
MedMamba-SS-TRM v17, Stage 6. Gives dataset prep a rate and an ETA.

`training/prep/core.py:395-402` prints

    ...  412/644 done (410 ok, 2 skipped)

on the manifest-flush cadence, and immediately above that line it computes

    rate = state_box["n"] / max(1e-6, now - state_box.get("t0", now))

which it then never uses. So the one number that would tell you whether a prep has five
minutes or two hours left is calculated and thrown away, every time.

`training/prep/` is FROZEN in its entirety (`test_frozen_files_untouched.py:
FROZEN_DIR_PREFIXES`), so this cannot be fixed where it lives. `install()` rebinds
`training.prep.core.run_extraction` to a wrapper that decorates the caller's `on_result`
with a `ProgressReporter` - the same module-attribute rebinding
`prepare_histologyhsi_bc_v8.py:346` already uses for `_v6._find_rgb_image`, and for the
same reason: the behaviour has to change without the frozen file changing.

Idempotent, and a no-op if the frozen signature ever moves - a progress line must never be
what breaks a prep run.
"""

from __future__ import annotations

from training.progress import ProgressReporter

_INSTALLED = False


def install(min_interval: float = 5.0) -> bool:
    """Wrap `training.prep.core.run_extraction` so per-item completions report a rate and
    an ETA. Returns True if the wrapper is in place. Safe to call more than once."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from training.prep import core as _core
    except Exception:
        return False

    original = getattr(_core, "run_extraction", None)
    if original is None:
        return False

    def run_extraction_with_progress(*, adapter, args, state, tasks, num_workers, on_result):
        reporter = ProgressReporter("  [extract]", min_interval=min_interval, stream_prefix="")
        total = len(tasks)
        seen = {"n": 0}

        def wrapped(res):
            # The frozen callback owns the manifest, the skip registry and the FatalItemError
            # SystemExit. Call it FIRST and unchanged, then only count.
            on_result(res)
            seen["n"] += 1
            reporter.emit(seen["n"], total)

        return original(adapter=adapter, args=args, state=state, tasks=tasks,
                        num_workers=num_workers, on_result=wrapped)

    _core.run_extraction = run_extraction_with_progress
    _INSTALLED = True
    return True
