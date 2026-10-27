# -*- coding: utf-8 -*-
"""
hsi_augment/stages/demo.py
==========================
`augment_pipeline.py demo`: a smoke test of the stage machinery, not part of
`all`. It opens the train split through the stage's SplitGuard, streams it in
memmap chunks into float64 per-band statistics, derives the capture grid
coordinates, and writes `band_stats.json`. Exactly the path every real stage
takes, on whatever build `--data_dir` points at (the test fixture, or the real
build, where it reads all 19 GB of X_train in about the time P1 does).

It also serves as the worked example of a stage module: config dataclass next
to the stage, registration, light imports, a summary dict for done.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from hsi_augment.config import ConfigError, StageConfig, param
from hsi_augment.io import DEFAULT_CHUNK_ROWS, RunningBandStats, iter_chunks
from hsi_augment.registry import register_stage


@dataclass(frozen=True)
class DemoConfig(StageConfig):
    chunk_rows: int = param(DEFAULT_CHUNK_ROWS, help="rows per memmap chunk")
    max_rows: Optional[int] = param(None, help="read only the first N train rows (none = all)")
    note: str = param("", help="free text; part of the config hash, so changing it forces a re-run")

    def validate(self) -> None:
        if self.chunk_rows <= 0:
            raise ConfigError("chunk_rows must be > 0")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ConfigError("max_rows must be > 0 (or none)")


@register_stage("demo", config=DemoConfig, dirname="demo", order=0, in_all=False,
                purposes=("describe",),
                help="smoke test: chunked float64 band stats of the train split")
def run(ctx, cfg: DemoConfig) -> Dict[str, Any]:
    train = ctx.open_split("train", purpose="describe")
    n = train.n if cfg.max_rows is None else min(cfg.max_rows, train.n)
    index = None if n == train.n else np.arange(n)

    row_bytes = int(np.prod(train.X.shape[1:]))
    # Peak: one float16 chunk plus RunningBandStats' float64 blocks (~3 x 32 MB).
    ctx.ram_guard(min(cfg.chunk_rows, n) * row_bytes * train.X.dtype.itemsize + 100_000_000,
                  "one chunk + float64 blocks")

    stats = RunningBandStats(ctx.adapter.n_bands)
    with ctx.timer("scan"):
        for i, ch in enumerate(iter_chunks(train.X, index, chunk_rows=cfg.chunk_rows)):
            stats.update(ch.data)
            ctx.log_metrics(chunk=i, rows_done=int(ch.rows[-1]) + 1)
    coords = train.coordinates()
    with_grid = int(coords.has_grid[:n].sum())
    ctx.log(f"{n} train rows, {ctx.adapter.n_bands} bands, {len(coords.runs)} captures, "
            f"{with_grid} rows with grid coordinates")
    ctx.save_json("band_stats.json", {"rows": n, "stats": stats.to_dict(), "note": cfg.note})
    return {"rows": n, "captures": len(coords.runs), "rows_with_grid": with_grid,
            "nonfinite": int(stats.n_nonfinite.sum())}
