# -*- coding: utf-8 -*-
"""
prepare_histologyhsi_bc_trainsel.py
===================================
HistologyHSI-BC-Recurrence preparation with **every label- or corpus-dependent
Pass-1 statistic restricted to TRAINING patients**.

Why this file exists
--------------------
`prepare_histologyhsi_bc.py` (the v8 behaviour) hands Pass 1 *all* 644
captures: `training/prep/core.py` calls `adapter.pass1(items, args)` before
any split filtering. Two statistics are affected:

1. **Band selection.** `compute_band_statistics` samples 10 % of all captures
   (seed 42 -> 64 captures: 48 train, 8 validation, 8 test) and the mutual
   information half of the importance score uses their tissue labels. The
   32 input bands were therefore chosen with held-out labels.
2. **The capture-gain reference.** Every capture is scaled towards the median
   of all captures' medians, held-out captures included (label-free, but not
   training-only).

This adapter keeps the v8 pipeline byte-for-byte except for the item list the
two statistics see:

* `pass1` reads the split that `core.run_prep` has already written to
  `<out_dir>/split_assignment.json` (it is written *before* Pass 1), runs the
  whole v8 Pass 1 on the **training** captures only - band statistics,
  importance ranking, capture medians, corpus median, value scale and
  calibration probe - and then gain-corrects every held-out capture against
  that **training** corpus median, with the same clip and suspect logging.
* Nothing else changes: extraction, patching, RGB rendering, dtype selection
  and unification are inherited from `HistologyHsiAdapterV8`.

`prepare_histologyhsi_bc.py` and `training/prep/*` are FROZEN (they produced
the v8 build every reported number comes from); this file only subclasses.

Usage (host; ~2 h with 8 workers). Reuse the v8 split so ONLY the leak changes:

    python prepare_histologyhsi_bc_trainsel.py \\
        --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \\
        --out_dir ./data/hsi_v9-trainsel --modality both --label_source tissue \\
        --patch_size 11 --stride 11 --roi_min_frac 0.8 \\
        --band_selection importance --num_bands 32 --band_min_gap 8 \\
        --band_max_corr 0.95 --band_min_coverage 0.30 \\
        --split 80_10_10 --split_strategy stratified \\
        --split_file data/hsi_v8-80_10_10_importance-new/split_assignment.json \\
        --capture_gain median_ratio --hsi_value_scale auto --rgb_source synthetic \\
        --seed 42 --num_workers 8 --verify_level deep

Writes, next to the usual outputs, `pass1_scope_report.json`: which captures
(and patients) each statistic saw, so the manuscript can state it exactly.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

import prepare_histologyhsi_bc as _v8
from prepare_histologyhsi_bc import HistologyHsiAdapterV8
from training.dataset_prep_common import load_split_assignment
from training.prep import run_prep
from training.prep_progress import install as install_prep_progress

SCOPE_REPORT = "pass1_scope_report.json"


def training_groups(out_dir: str) -> set:
    """Training patient ids from the split `core.run_prep` wrote before Pass 1."""
    path = os.path.join(out_dir, "split_assignment.json")
    if not os.path.isfile(path):
        raise RuntimeError(
            f"{path} not found: core.run_prep writes it before Pass 1, so its absence means "
            "the call order changed - refusing to guess the training set.")
    train, _val, _test, _meta = load_split_assignment(path)
    if not train:
        raise RuntimeError(f"{path} lists no training groups")
    return set(train)


def partition_items(items, train_groups: set):
    train = [it for it in items if it.group_id in train_groups]
    held = [it for it in items if it.group_id not in train_groups]
    return train, held


def gain_for(median: float, corpus_median: float, lo: float, hi: float):
    """The v8 rule (prepare_histologyhsi_bc.py, HistologyHsiAdapterV8.pass1)."""
    raw = corpus_median / median
    return float(np.clip(raw, lo, hi)), raw


class HistologyHsiAdapterTrainSel(HistologyHsiAdapterV8):
    # A new name: a resumed v8 manifest must never be mistaken for this build.
    name = "histology_hsi_bc_v8_trainsel"

    def pass1(self, items, args) -> Optional[dict]:
        train_groups = training_groups(args.out_dir)
        train_items, held_items = partition_items(items, train_groups)
        if not train_items:
            raise RuntimeError("no training captures to compute Pass-1 statistics from")
        print(f"  [trainsel] Pass 1 on {len(train_items)} training captures "
              f"({len({it.group_id for it in train_items})} patients); "
              f"{len(held_items)} held-out captures excluded from every statistic", flush=True)

        # The complete v8 Pass 1, on training captures only.
        state = super().pass1(train_items, args) or {}

        # Held-out captures still need a gain; use the TRAINING corpus median.
        if args.capture_gain != "none" and held_items:
            sel = state.get("selected_band_indices")
            sel_arr = None if sel is None else np.asarray(sel, dtype=np.int64)
            medians = _v8._capture_medians(held_items, args, sel_arr)
            corpus = float(state.get("capture_gain_corpus_median", 1.0))
            gains: Dict[str, dict] = dict(state.get("capture_gain", {}))
            suspects: List[str] = list(state.get("capture_gain_suspects", []))
            for key, m in sorted(medians.items()):
                g, raw = gain_for(m, corpus, args.gain_clip_lo, args.gain_clip_hi)
                gains[key] = {"gain": g, "median_intensity": m, "n_sampled": args.gain_sample_pixels,
                              "raw_gain_before_clip": raw, "reference": "training_corpus_median"}
                if not (args.gain_clip_lo <= raw <= args.gain_clip_hi):
                    suspects.append(key)
            state["capture_gain"] = gains
            state["capture_gain_suspects"] = sorted(set(suspects))

        state["pass1_scope"] = {
            "rule": "band selection, band statistics, capture-gain corpus median, value scale and "
                    "calibration probe computed on TRAINING captures only",
            "train_patients": sorted(train_groups),
            "n_train_captures": len(train_items),
            "n_heldout_captures": len(held_items),
            "heldout_patients": sorted({it.group_id for it in held_items}),
            "band_selection_sample_fraction": args.band_selection_sample_fraction,
            "seed": args.seed,
        }
        return state

    def write_pass1_outputs(self, state, out_dir) -> List[str]:
        scope = (state or {}).pop("pass1_scope", None)
        written = list(super().write_pass1_outputs(state, out_dir))
        if scope is not None:
            with open(os.path.join(out_dir, SCOPE_REPORT), "w") as f:
                json.dump(scope, f, indent=2)
        return written


def main(argv=None) -> int:
    install_prep_progress()
    return run_prep(HistologyHsiAdapterTrainSel(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
