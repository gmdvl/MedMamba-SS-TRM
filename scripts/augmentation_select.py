#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/augmentation_select.py
==============================
The pre-registered arm selection of the HSI augmentation study (Plan §Phase 10,
stage `select`; issue #33). After the screen (every arm, seed 42, 12 epochs)
it names the arms that go on to the seeds stage and writes `selection.json`:
`sweeps/augmentation_pipeline.py` builds its seeds stage from it, and
`hsi_augment/report.py` opens the test columns only once it exists.

Usage (repo root; CPU, seconds)
-------------------------------
    python scripts/augmentation_select.py                 # the arms of sweeps/augmentation_pipeline.py
    python scripts/augmentation_select.py --dry_run       # print the decision, write nothing
    python scripts/augmentation_select.py --arms_file arms.json --out other/selection.json
    -> reports/augmentation/selection.json  (refuses to replace one without --force)

The rule (pre-registered: constants here, not flags)
----------------------------------------------------
The best arm of each family goes on: generative (A2 / A3 / A4) and
non-generative (A0os / A1 / A1p). The control A0 is never selected; the seeds
stage always has it.
  1. score = mean val `f1_macro` over epochs 10, 11, 12 of history.json;
     arms less than 0.005 below the family's best score tie;
  2. among them, mean val DCIS F1 (`per_class_f1[DCIS]`, same epochs), with
     the same 0.005 tie gap;
  3. then the cheaper method: fewer generator GPU-hours (none < CVAE < AE +
     LDM, Plan §Phase 6/7), then fewer on-the-fly operations (DCIS copies,
     mixing, physics ops), both read from the run's resolved flags;
  4. then the arm table's order (a deterministic last resort).
Scores come from `hsi_augment.report.screen_score`, the report's own function,
so the report's check of `scores` against history.json is exact.

Before anything is chosen, every candidate's screen run must exist and follow
the protocol (seed 42, 12 epochs, 250,113 rows per epoch), and a generative
arm's run must have trained on the CURRENT export:
`config.json["synthetic"]["export_id"]` equal to
`export/<source>/provenance.json`'s, A4's source being the one
`lopo_utility/decision.json` names. Otherwise nothing is written, and every
problem is listed.

What it reads (the test seal, Plan rule 3)
------------------------------------------
Per run, config.json (found by `cli_args.run_tag`, as `run_experiments.py`
and the report find runs; plus its protocol) and history.json (the val
subsample's per-epoch metrics). Never test_predictions.npz or
test_report.json: `train.py` writes both for every run, so the seal is where
files are read. Every read goes through `hsi_augment.report.Inputs` with the
seal closed, which refuses a test file; selection.json records the sha256 of
each file read (export provenance and decision.json included).

selection.json
--------------
    {"schema_version": 1, "rule": "...", "protocol": {...},
     "selected": ["A3", "A1"],                      # generative, then non-generative
     "per_family": {"generative": "A3", "non-generative": "A1"},
     "decisions": {family: {"winner", "decided_by", "steps": [...]}},
     "scores": {arm: {"score", "dcis_f1", "per_epoch_f1_macro", "cost", "tag", "run", ...}},
     "a4_source": "ldm" | null,
     "arms": [{key, family, screen_tag, seed_tag, bench_tag, generator, description}, ...],
     "inputs_sha256": {...}, "test_files_read": [], "git_sha": "...", "created_utc": "..."}
`selected`, `scores` and `arms` are what `hsi_augment.report._load_selection`
reads.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hsi_augment import report as R  # noqa: E402

SCHEMA_VERSION = 1
SCREEN_RUN_EPOCHS = 12
TIE_GAP = R.SCREEN_TIE_GAP                  # 0.005
FAMILIES = ("generative", "non-generative")
GENERATOR_COST = {None: 0, "cvae": 1, "ldm": 2}  # GPU-h: none < CVAE (~2-2.5) < AE + LDM (~2.5-3)
DEFAULT_SWEEP = "sweeps/augmentation_pipeline.py"
RULE = (f"per family (generative, non-generative), the best screen arm: mean val f1_macro over epochs "
        f"{list(R.SCREEN_EPOCHS)} of history.json; within {TIE_GAP} of the best -> mean val DCIS F1 over the "
        f"same epochs (within {TIE_GAP} of the best) -> the cheaper method (generator GPU-h: none < cvae < ldm; "
        f"then fewer of DCIS copies / mixing / physics ops) -> arm table order. Test is never read.")


class SelectError(RuntimeError):
    """The screen cannot be decided on (missing or non-conforming runs)."""


# ============================================================================
# the arms
# ============================================================================

def load_arms(sweep: Optional[str], arms_file: Optional[str], aug_out: Optional[str] = None
              ) -> Tuple[R.Arm, ...]:
    """The arm table: `--arms_file` (the report's format), else the sweep's
    `ARM_TABLE` (it follows the glass rule and AUG_PHYS; its AUG_OUT defaults
    to `aug_out`), else the plan's."""
    if arms_file:
        with open(arms_file) as f:
            return R.arms_from_doc(json.load(f), Path(arms_file).name)
    if not sweep:
        return R.DEFAULT_ARMS
    import run_experiments as rx
    path = Path(sweep) if Path(sweep).is_file() else REPO_ROOT / sweep
    unset = "AUG_OUT" not in os.environ
    if unset and aug_out:
        os.environ["AUG_OUT"] = aug_out
    try:
        table = rx.load_sweep(str(path)).get("ARM_TABLE")
    finally:
        if unset:
            os.environ.pop("AUG_OUT", None)
    if table is None:
        raise SelectError(f"{sweep} defines no ARM_TABLE")
    return R.arms_from_doc(table, path.name)


def read_decision(inp: R.Inputs, aug_out: Path) -> Optional[str]:
    """A4's generator, as the report reads it (first string of DECISION_KEYS)."""
    path = aug_out / R.DECISION_JSON
    if not path.is_file():
        return None
    doc = inp.json(path)
    return next((doc[k] for k in R.DECISION_KEYS if isinstance(doc, Mapping) and isinstance(doc.get(k), str)),
                None)


def method_cost(config: Mapping[str, Any]) -> Tuple[int, int]:
    """(generator GPU-h rank, on-the-fly operations) of a run's resolved flags."""
    a = config.get("cli_args") or {}
    syn = config.get("synthetic") or {}
    gen = GENERATOR_COST.get(syn.get("source"), len(GENERATOR_COST)) if syn else 0
    minority = a.get("minority_augment") == "on"
    ops = (int(minority and float(a.get("minority_oversample_factor") or 1.0) > 1.0)
           + int(minority and float(a.get("minority_mix_prob") or 0.0) > 0.0)
           + int((a.get("phys_aug") or "off") != "off"))
    return gen, ops


# ============================================================================
# one screen run
# ============================================================================

def screen_row(inp: R.Inputs, index, arm: R.Arm, order: int, run_data: str, aug_out: Path,
               decision: Optional[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """(row, problems) of an arm's screen run. Reads its config.json and
    history.json, and the export provenance of a generative arm."""
    rd = R.find_run(index, arm.screen_tag, run_data, need="history.json")
    if rd is None:
        return None, [f"{arm.key} ({arm.screen_tag}): no screen run with a history.json on {run_data}"]
    who = f"{arm.key} ({arm.screen_tag}, {rd.name})"
    cfg = inp.json(rd / "config.json")
    a = cfg.get("cli_args") or {}
    bad = []
    if a.get("seed") != R.SCREEN_SEED or a.get("epochs") != SCREEN_RUN_EPOCHS:
        bad.append(f"{who}: seed {a.get('seed')}, {a.get('epochs')} epochs; the screen is seed "
                   f"{R.SCREEN_SEED}, {SCREEN_RUN_EPOCHS} epochs")
    kept = (cfg.get("train_subsample") or {}).get("kept")
    if kept != R.KEPT_EXPECTED:
        bad.append(f"{who}: {kept} training rows per epoch, every arm keeps {R.KEPT_EXPECTED}")
    syn = cfg.get("synthetic")
    src = decision if arm.generator == "decision" else arm.generator
    if arm.generator:
        prov = aug_out / "export" / str(src) / "provenance.json"
        if src is None:
            bad.append(f"{who}: {R.DECISION_JSON} names no generator")
        elif not syn or syn.get("source") != src:
            bad.append(f"{who}: trained on synthetic source {(syn or {}).get('source')!r}, expected {src!r}")
        elif not prov.is_file():
            bad.append(f"{who}: no export/{src}/provenance.json to check its export_id against")
        else:
            current = inp.json(prov).get("export_id")
            if syn.get("export_id") != current:
                bad.append(f"{who}: trained on export {syn.get('export_id')}, the current export/{src} is "
                           f"{current}: run it again")
    elif syn:
        bad.append(f"{who}: has synthetic rows ({syn.get('source')}) but the arm is not generative")
    names = list(cfg.get("class_names") or R.DEFAULT_CLASS_NAMES)
    if R.DCIS not in names:
        return None, bad + [f"{who}: class names {names} lack {R.DCIS!r}"]
    try:
        sc = R.screen_score(inp.json(rd / "history.json"), R.SCREEN_EPOCHS, names.index(R.DCIS))
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return None, bad + [f"{who}: history.json lacks f1_macro / per_class_f1 ({type(e).__name__}: {e})"]
    if sc["score"] is None:
        bad.append(f"{who}: {sc['reason']}")
    if bad:
        return None, bad
    return {"arm": arm.key, "family": arm.family, "tag": arm.screen_tag, "run": rd.name, "order": order,
            "score": sc["score"], "dcis_f1": sc["dcis_f1"], "per_epoch_f1_macro": sc["per_epoch_f1_macro"],
            "cost": list(method_cost(cfg)), "generator": src,
            "export_id": (syn or {}).get("export_id")}, []


# ============================================================================
# the rule
# ============================================================================

def choose(rows: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """The family winner of `rows` and how it was decided."""
    pool = list(rows)
    if not pool:
        raise SelectError("no arm to choose from")
    steps = []
    for by, key in (("f1_macro", "score"), ("dcis_f1", "dcis_f1")):
        best = max(r[key] for r in pool)
        pool = [r for r in pool if best - r[key] < TIE_GAP]
        steps.append({"by": by, "best": best, "tied": [r["arm"] for r in pool]})
        if len(pool) == 1:
            return pool[0]["arm"], {"winner": pool[0]["arm"], "decided_by": by, "steps": steps}
    cheapest = min(tuple(r["cost"]) for r in pool)
    pool = [r for r in pool if tuple(r["cost"]) == cheapest]
    steps.append({"by": "cost", "best": list(cheapest), "tied": [r["arm"] for r in pool]})
    if len(pool) == 1:
        return pool[0]["arm"], {"winner": pool[0]["arm"], "decided_by": "cost", "steps": steps}
    winner = min(pool, key=lambda r: r["order"])["arm"]
    steps.append({"by": "arm_order", "tied": [winner]})
    return winner, {"winner": winner, "decided_by": "arm_order", "steps": steps}


def select(arms: Sequence[R.Arm], experiments: str, run_data: str, aug_out: str) -> Dict[str, Any]:
    """The selection document. Raises SelectError, listing every problem."""
    roots = {"experiments": experiments, "aug_out": aug_out}
    inp = R.Inputs(roots, reveal_test=False)
    index = R.index_runs(experiments)
    decision = read_decision(inp, Path(aug_out))
    rows, problems = [], []
    for order, arm in enumerate(arms):
        if arm.family not in FAMILIES:
            continue
        row, bad = screen_row(inp, index, arm, order, run_data, Path(aug_out), decision)
        problems += bad
        if row is not None:
            rows.append(row)
    if problems:
        raise SelectError("cannot select:\n  " + "\n  ".join(problems))
    decisions = {fam: choose([r for r in rows if r["family"] == fam])[1]
                 for fam in FAMILIES if any(r["family"] == fam for r in rows)}
    if not decisions:
        raise SelectError("the arm table has no generative or non-generative arm")
    return {
        "schema_version": SCHEMA_VERSION,
        "rule": RULE,
        "protocol": {"seed": R.SCREEN_SEED, "epochs": SCREEN_RUN_EPOCHS, "score_epochs": list(R.SCREEN_EPOCHS),
                     "tie_gap": TIE_GAP, "kept": R.KEPT_EXPECTED, "run_data": run_data},
        "selected": [d["winner"] for d in decisions.values()],
        "per_family": {fam: d["winner"] for fam, d in decisions.items()},
        "decisions": decisions,
        "scores": {r["arm"]: {k: v for k, v in r.items() if k not in ("arm", "order")} for r in rows},
        "a4_source": decision,
        "arms": [asdict(a) for a in arms],
        "inputs_sha256": dict(sorted(inp.sha256.items())),
        "test_files_read": [],
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", default=R.DEFAULT_EXPERIMENTS, help="training run directories")
    p.add_argument("--run_data", default=R.DEFAULT_RUN_DATA,
                   help="a run counts when its cli_args.data_dir contains this")
    p.add_argument("--aug_out", default=R.DEFAULT_AUG_OUT,
                   help="augment_pipeline.py --out (export provenance, lopo_utility/decision.json)")
    p.add_argument("--sweep", default=DEFAULT_SWEEP,
                   help="the sweep whose ARM_TABLE is screened (it reads the glass rule and AUG_PHYS; "
                        "AUG_OUT defaults to --aug_out)")
    p.add_argument("--arms_file", default=None,
                   help="JSON arm table (the report's format) instead of --sweep's")
    p.add_argument("--out", default=R.DEFAULT_SELECTION, help="selection.json to write")
    p.add_argument("--force", action="store_true", help="replace an existing selection.json")
    p.add_argument("--dry_run", action="store_true", help="print the selection, write nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.out)
    if out.exists() and not (a.force or a.dry_run):
        print(f"[augmentation_select] {out} exists (the arms are already chosen, and the report may have "
              f"opened test with it); --force replaces it", file=sys.stderr)
        return 1
    try:
        arms = load_arms(a.sweep, a.arms_file, a.aug_out)
        doc = select(arms, a.experiments, a.run_data, a.aug_out)
    except (SelectError, R.ReportError, R.TestSealError, OSError, ValueError) as e:
        print(f"[augmentation_select] {e}", file=sys.stderr)
        return 1
    if a.dry_run:
        print(json.dumps(doc, indent=2, sort_keys=True))
        return 0
    from hsi_augment.io import save_json
    from hsi_augment.stage import git_sha
    doc.update(git_sha=git_sha(), created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    save_json(out, doc)
    for fam, d in doc["decisions"].items():
        s = doc["scores"][d["winner"]]
        print(f"[augmentation_select] {fam}: {d['winner']} ({s['tag']}) score {s['score']:.4f}, "
              f"DCIS F1 {s['dcis_f1']:.4f}, decided by {d['decided_by']}")
    print(f"[augmentation_select] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
