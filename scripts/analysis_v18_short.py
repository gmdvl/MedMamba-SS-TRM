#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analysis_v18_short.py
=============================
Aggregate the <= 2 h experiments of manuscript v18 (the short-run queue, now sweeps/paper_runs.py)
on the EXISTING v8 build, with the metric code of scripts/analysis_v19.py, so
the new numbers are directly comparable with the published tables.

  baselines   HybridSN / SpectralFormer at seeds 1, 7, 13, 23 (+ the existing 42):
              test, full validation and pooled metrics per seed, mean +- s.d.,
              and seed-paired differences to MedMamba-SS-TRM (32-band arm).
  k2_control  K = 2 core applications vs the 12-epoch baseline (seed 42).
  fixed_eta   zero-shot band removal with eta/range fixed vs the variable-eta run.
  eval_mode   the hierarchical checkpoints' eval-mode diagnostic.
  band_repeat training-only band re-selection vs the reported band set.

Read-only on every run. Writes paper/source/v18_short_analysis.json.
Missing inputs are reported, never filled in.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.heldout_metrics import classification_metrics  # noqa: E402
from scripts.analysis_v19 import SEED_RUNS, load_probs, run_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DATA = ROOT / "data/hsi_v8-80_10_10_importance-new"
OUT = ROOT / "paper/source/v18_short_analysis.json"
SEEDS = (1, 7, 13, 23, 42)
KEYS = ("balanced_accuracy", "f1_macro", "accuracy", "cohen_kappa", "ece", "roc_auc_macro")
MISSING = []


def metrics_for(test_npz, val_npz):
    pt, lt = load_probs(test_npz)
    e = {"test": classification_metrics(lt, pt)}
    if val_npz is not None and Path(val_npz).is_file():
        pv, lv = load_probs(val_npz)
        e["val_full"] = classification_metrics(lv, pv)
        e["pooled"] = classification_metrics(np.concatenate([lv, lt]), np.concatenate([pv, pt]))
    return {k: {m: v[m] for m in KEYS if m in v} for k, v in e.items()}


def mean_sd(rows, split):
    vals = [r[split] for r in rows.values() if split in r]
    if len(vals) < 2:
        return None
    return {m: [float(np.mean([v[m] for v in vals])), float(np.std([v[m] for v in vals], ddof=1))]
            for m in KEYS if all(m in v for v in vals)} | {"n": len(vals)}


def trm_rows():
    rows = {}
    for s in SEEDS:
        rd = run_dir(SEED_RUNS[s][0])
        rows[str(s)] = metrics_for(rd / "test_predictions.npz",
                                   rd / "heldout_v19/validation/test_predictions.npz")
    return rows


def baseline_rows(arch):
    rows = {}
    for s in SEEDS:
        hits = sorted(glob.glob(str(EXP / f"*importance-new-hsi_{arch}_baseline_s{s}")))
        hits = [h for h in hits if (Path(h) / "test_predictions.npz").is_file()]
        if not hits:
            MISSING.append(f"{arch} seed {s}")
            continue
        rd = Path(hits[-1])
        rows[str(s)] = metrics_for(rd / "test_predictions.npz", rd / "val_predictions.npz")
        rows[str(s)]["run"] = rd.name
        rows[str(s)]["params"] = json.load(open(rd / "config.json"))["backbone_num_params"]
    return rows


def paired(trm, base):
    out = {}
    for split in ("test", "val_full", "pooled"):
        d = [trm[s][split]["balanced_accuracy"] - base[s][split]["balanced_accuracy"]
             for s in base if s in trm and split in trm[s] and split in base[s]]
        if d:
            out[split] = {"trm_minus_baseline_mean": float(np.mean(d)),
                          "trm_minus_baseline_sd": float(np.std(d, ddof=1)) if len(d) > 1 else None,
                          "trm_ahead_at": int(np.sum(np.array(d) > 0)), "n_seeds": len(d),
                          "per_seed": dict(zip([s for s in base if s in trm], d))}
    return out


def test_report(rd: Path):
    t = json.load(open(rd / "test_report.json"))
    m = t["sklearn_metrics"]
    return {"run": rd.name, "balanced_accuracy": m["balanced_accuracy"], "f1_macro": m["f1_macro"],
            "accuracy": m["accuracy"], "per_class_f1": m["per_class_f1"],
            "gflops": (t.get("flops") or {}).get("total_gflops_approx"),
            "params": json.load(open(rd / "config.json")).get("backbone_num_params"),
            "best_epoch": t.get("best_epoch")}


def k2_control():
    base = run_dir("20260915_031555")
    hits = []
    for c in glob.glob(str(EXP / "*/config.json")):
        a = json.load(open(c)).get("cli_args", {})
        if a.get("run_tag") == "k2-control" and (Path(c).parent / "test_report.json").is_file():
            hits.append(Path(c).parent)
    if not hits:
        MISSING.append("k2-control run")
        return {"baseline_63": test_report(base)}
    k2 = sorted(hits)[-1]
    h = json.load(open(k2 / "history.json"))
    return {"baseline_63": test_report(base), "k2": test_report(k2),
            "k2_wallclock_h": float(sum(float(e.get("epoch_time", 0) or 0) for e in h) / 3600),
            "k2_trm_settings": {k: json.load(open(k2 / "config.json"))["cli_args"][k]
                                for k in ("trm_n_latent", "trm_n_improve", "trm_deep_supervision_steps", "epochs")}}


def zero_shot():
    rd = run_dir(SEED_RUNS[42][0])
    out = {}
    for key, sub in (("variable_eta", "band_decimation"), ("fixed_eta", "band_decimation_fixed_eta")):
        p = rd / sub / "summary.json"
        if p.is_file():
            out[key] = {str(r["n_bands"]): {"balanced_accuracy": r["balanced_accuracy"], "f1_macro": r["f1_macro"],
                                            "per_class_f1": r["per_class_f1"]} for r in json.load(open(p))["rows"]}
        else:
            MISSING.append(f"zero-shot {key}")
    return out


def eval_mode():
    out = {}
    for h in sorted(glob.glob(str(EXP / "*/eval_mode_check_v20.json"))):
        r = json.load(open(h))
        out[Path(h).parent.name[:15]] = {"verdict": r["verdict"], "n_patches": r["n_patches"],
                                         "modes": {m: {k: v[k] for k in ("balanced_accuracy", "prediction_counts",
                                                                          "collapsed")}
                                                   for m, v in r["modes"].items()},
                                         "bn_max_mean_shift": max(b["mean_shift_max"] for b in r["batchnorm_layers"]),
                                         "bn_max_var_ratio": max(b["var_ratio_max"] for b in r["batchnorm_layers"])}
    return out


def band_repeat():
    a = np.sort(np.load(ROOT / "data/hsi_v9-trainsel/selected_wavelengths.npy").astype(float))
    b = np.sort(np.load(DATA / "selected_wavelengths.npy").astype(float))
    d = np.abs(a[:, None] - b[None, :]).min(1)
    ga, gb = np.diff(a), np.diff(b)
    g8 = json.load(open(DATA / "capture_gain_report.json"))["corpus_median"]
    g9 = json.load(open(ROOT / "data/hsi_v9-trainsel/capture_gain_report.json"))["corpus_median"]
    return {"reselected_nm": np.round(a, 1).tolist(), "reported_nm": np.round(b, 1).tolist(),
            "nearest_shift_nm": {"max": float(d.max()), "median": float(np.median(d)), "identical": int((d < 0.05).sum())},
            "visible_le_640": [int((a <= 640).sum()), int((b <= 640).sum())],
            "largest_gap_nm": [float(ga.max()), float(gb.max())],
            "gap_edges_nm": [[float(a[ga.argmax()]), float(a[ga.argmax() + 1])],
                             [float(b[gb.argmax()]), float(b[gb.argmax() + 1])]],
            "gain_reference": {"training_only": g9, "all_captures": g8, "rel_diff": float(g9 / g8 - 1)},
            "scope": json.load(open(ROOT / "data/hsi_v9-trainsel/pass1_scope_report.json"))}


def main():
    trm = trm_rows()
    out = {"build": str(DATA.relative_to(ROOT)), "mmmt_ss_trm_32band": {"per_seed": trm,
           "summary": {s: mean_sd(trm, s) for s in ("test", "val_full", "pooled")}}}
    for arch in ("hybridsn", "spectralformer"):
        rows = baseline_rows(arch)
        out[arch] = {"per_seed": rows, "summary": {s: mean_sd(rows, s) for s in ("test", "val_full", "pooled")},
                     "paired_vs_ss_trm": paired(trm, rows)}
    out["k2_control"] = k2_control()
    out["zero_shot"] = zero_shot()
    out["eval_mode"] = eval_mode()
    out["band_repeat"] = band_repeat()
    out["missing"] = MISSING
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"[written] {OUT}\n[missing] {MISSING or 'none'}")
    for arch in ("hybridsn", "spectralformer"):
        for split, v in out[arch]["paired_vs_ss_trm"].items():
            print(f"  SS-TRM - {arch:14s} {split:8s} {100 * v['trm_minus_baseline_mean']:+.2f} pts, "
                  f"SS-TRM ahead at {v['trm_ahead_at']}/{v['n_seeds']} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
