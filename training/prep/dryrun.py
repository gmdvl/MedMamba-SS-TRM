# -*- coding: utf-8 -*-
"""training/prep/dryrun.py - `--dry_run` summary (no extraction)."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from training.prep.adapter import PrepItem


def dry_run_report(*, adapter, args, items: Sequence[PrepItem], train, val, test,
                    split_meta: dict, warnings: List[str]) -> str:
    by_split = {"train": train, "validation": val, "test": test}

    def _assign(g):
        return "test" if g in test else "validation" if g in val else "train"

    per_split_cls: Dict[str, Counter] = {s: Counter() for s in by_split}
    per_split_items = Counter()
    for it in items:
        s = _assign(it.group_id)
        per_split_items[s] += 1
        per_split_cls[s][it.label] += 1

    lines = []
    lines.append(f"DRY RUN  adapter={adapter.name}  items={len(items)}  "
                 f"groups={len({it.group_id for it in items})}  classes={len(adapter.class_names)}")
    lines.append(f"  split: spec={split_meta.get('spec')} strategy={split_meta.get('strategy')}")
    for s, groups in by_split.items():
        if not groups and s == "validation":
            continue
        cls = "  ".join(f"{adapter.class_names[c]}={n}" for c, n in sorted(per_split_cls[s].items()))
        lines.append(f"  {s:<11} groups={len(groups):<5} items={per_split_items[s]:<6} [{cls}]")
    for w in warnings:
        lines.append(f"  [split-warning] {w}")
    for m in adapter.modalities():
        spec = adapter.modality_spec(m, args, None)
        lines.append(f"  modality {m!r}: store_dtype={spec.store_dtype} "
                     f"clip={spec.value_clip} sample_shape={spec.sample_shape}")
    return "\n".join(lines)
