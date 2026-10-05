#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print manuscript-v18 table rows and key numbers from
paper/source/v18_short_analysis.json, so no number is transcribed by hand."""
import json
from pathlib import Path

A = json.load(open(Path(__file__).resolve().parents[1] / "paper/source/v18_short_analysis.json"))


def ms(v, pct=True, d=2):
    if v is None:
        return "—"
    m, s = v
    return f"{100 * m:.{d}f} ± {100 * s:.{d}f}" if pct else f"{m:.3f} ± {s:.3f}"


for name, key in (("HybridSN [@hybridsn]", "hybridsn"), ("SpectralFormer [@spectralformer]", "spectralformer")):
    s = A[key]["summary"]
    t, v, p = s["test"], s["val_full"], s["pooled"]
    if not (t and v and p):
        print(f"| {name}: fewer than 2 seeds with full predictions - no mean row |")
        continue
    n = t["n"]
    print(f"| {name}, {n} seeds (mean ± s.d.) | — | {ms(t['accuracy'])} | {ms(t['balanced_accuracy'])} | "
          f"{ms(t['f1_macro'], False)} | {ms(t['cohen_kappa'], False)} | {ms(v['balanced_accuracy'])} | "
          f"{ms(v['f1_macro'], False)} | {ms(p['balanced_accuracy'])} | {ms(p['f1_macro'], False)} |")
    for split, w in A[key]["paired_vs_ss_trm"].items():
        print(f"    paired {split}: SS-TRM - {key} = {100 * w['trm_minus_baseline_mean']:+.2f} "
              f"(s.d. {100 * (w['trm_minus_baseline_sd'] or 0):.2f}); SS-TRM ahead at {w['trm_ahead_at']}/{w['n_seeds']}; "
              f"per seed {({k: round(100 * x, 2) for k, x in w['per_seed'].items()})}")
    for sd, r in A[key]["per_seed"].items():
        print(f"    seed {sd}: test {100 * r['test']['balanced_accuracy']:.2f}  val {100 * r['val_full']['balanced_accuracy']:.2f}"
              f"  pooled {100 * r['pooled']['balanced_accuracy']:.2f}  ECE {r['test']['ece']:.3f}")
print("SS-TRM summary:", {k: ms(v["balanced_accuracy"]) for k, v in A["mmmt_ss_trm_32band"]["summary"].items() if v})
print("fixed eta:", json.dumps(A["zero_shot"].get("fixed_eta"), indent=0)[:600])
print("variable eta:", {k: round(v["balanced_accuracy"], 4) for k, v in A["zero_shot"].get("variable_eta", {}).items()})
print("missing:", A["missing"])
