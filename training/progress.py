# -*- coding: utf-8 -*-
"""
training/progress.py
===========================
MedMamba-SS-TRM v17, Stage 4. One rate-limited progress line with an ETA, used by the
training loop, the validation loop and the dataset-prep wrappers.

Why this exists
---------------
An HSI epoch is 958 optimizer steps and eleven minutes, and until v17 it printed
NOTHING between "Epoch 0012" and "Epoch 0013". A PAD run is 200 epochs with
`--early_stop_patience 40`, and nothing said how many epochs were left or how close the
run was to stopping. Dataset prep printed `412/644 done` with no rate and no ETA -
`training/prep/core.py:400` even computes a rate and then discards it.

Interactive vs redirected
-------------------------
A terminal wants ONE line that updates in place; a log file wants periodic whole lines,
because a carriage-return redraw written to a file produces one unreadable mega-line. This
picks by `stream.isatty()`:

    TTY        `\r`-redraw, refreshed every `tty_interval` (0.25 s), terminated by a real
               newline on the final item so the next output starts clean.
    redirected whole lines every `min_interval` (2 s+), exactly as before.

Every run in this repo is launched with `2>&1 | tee run.log`, so both halves matter. That
is also why this is ~90 lines instead of a `tqdm` dependency: tqdm is installed here but is
not in `requirements.txt`, and getting the redirected half right still means bypassing its
bar.
"""

from __future__ import annotations

import sys
import time
from typing import Optional


def format_duration(seconds: Optional[float]) -> str:
    """`3725.0 -> '1h02m'`, `125.0 -> '2m05s'`, `9.4 -> '9s'`. `None`/negative -> '?'."""
    if seconds is None or seconds < 0 or seconds != seconds:      # NaN-safe
        return "?"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


class ProgressReporter:
    """Prints at most one line per `min_interval` seconds, and always on the final item.

    `emit(done, total, suffix)` is the whole API. The ETA is a flat extrapolation from the
    mean rate since construction, which is the right model here: every step of an epoch
    does identical work on identical shapes, so there is no warmup curve to fit after the
    first step or two.
    """

    def __init__(self, label: str, min_interval: float = 2.0, stream_prefix: str = "  ",
                 tty_interval: float = 0.25, stream=None):
        self.label = label
        self.min_interval = float(min_interval)
        self.prefix = stream_prefix
        self.stream = stream if stream is not None else sys.stdout
        try:
            self.is_tty = bool(self.stream.isatty())
        except Exception:                                    # a stream without isatty()
            self.is_tty = False
        self.interval = float(tty_interval) if self.is_tty else float(min_interval)
        self.t0 = time.time()
        self._last = 0.0
        self._width = 0                                      # longest line drawn, for erasing

    def reset(self, label: Optional[str] = None) -> None:
        if label is not None:
            self.label = label
        self.t0 = time.time()
        self._last = 0.0
        self._width = 0

    def close(self) -> None:
        """Release an in-place line that never reached its final item.

        The training loop can `break` (a numerical-stability abort) or `continue` past the
        last counted batch, so `emit` may never see `done >= total`. Without this the next
        `print` - the epoch summary - would be written onto the same terminal line.
        No-op when redirected, and idempotent.
        """
        if self.is_tty and self._width:
            self.stream.write("\n")
            self.stream.flush()
            self._width = 0

    def emit(self, done: int, total: Optional[int], suffix: str = "") -> None:
        now = time.time()
        final = total is not None and done >= total
        if not final and (now - self._last) < self.interval:
            return
        self._last = now
        elapsed = now - self.t0
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0

        if total:
            pct = f"{100.0 * done / total:3.0f}%"
            eta = format_duration((total - done) / rate) if rate > 0 else "?"
            head = f"{self.label} {done}/{total} {pct}"
        else:
            eta = "?"
            head = f"{self.label} {done}"

        parts = [head]
        if suffix:
            parts.append(suffix)
        # Report the reciprocal for slow items: a training step is naturally read as
        # "0.437 s/step", not "2.29/s", and s/step is the unit every measurement in
        # plan/v17_baseline.md and scripts/gpu_tune.py already uses.
        if rate <= 0:
            parts.append("-")
        elif rate < 4.0:
            parts.append(f"{1.0 / rate:.3f} s/it")
        else:
            parts.append(f"{rate:.0f}/s")
        parts.append(f"ETA {eta}")
        line = self.prefix + "  ".join(parts)

        if not self.is_tty:
            print(line, file=self.stream, flush=True)
            return

        # In a terminal: redraw one line in place. Pad to the widest line drawn so far so
        # a shortening line (99% -> 100%) cannot leave characters from the previous one
        # stranded on screen.
        self._width = max(self._width, len(line))
        self.stream.write("\r" + line.ljust(self._width))
        if final:
            self.stream.write("\n")                          # release the line
            self._width = 0
        self.stream.flush()
