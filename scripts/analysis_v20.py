#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analysis_v20.py
=======================
Aggregate every number manuscript v18 reports, from the v20 runs on the
training-only band-selection build (data/hsi_v9-trainsel) - one metric
implementation (training/heldout_metrics.py), read-only on every run.

Unlike analysis_v19.py nothing is hard-coded by timestamp: runs are found by
the `--run_tag` recorded in their config.json (tag scheme in
documentations/16_v20_commands.md), so this can be re-run at any point of S4
and reports what exists and what is still missing:

  headline      v20-head-s{S}            (hsi + rgb, 20 epochs, 5 seeds)
  baselines     v20-s{S}                 (hybridsn, spectralformer; scripts/train_hsi_baseline.py)
  medmamba      ../medmamba-original/MedMamba/runs_hsi/v20_s{S}   (train_hsi_v20.py)
  controls      v20-abl-base-s{S}, v20-k2-s{S}, v20-recon-off-s{S}, v20-enc-index-s{S}   (12 epochs)
  depth         v20-depth-n{T}           (12 epochs, seed 42; T = 3 is v20-abl-base-s42)
  bands         v20-bands{N}             (12 epochs, data/hsi_v9-bands{N})
  hierarchical  v20-arch-split, v20-arch-fullchannel (12 epochs)
  cv            v20-cv-f{i}              (hsi + rgb, data/hsi_v9-cv5-f{i})
  lusc          v20-lusc-f{i}            (HMI-LUSC 5-fold patient CV: recursive, hybridsn, spectralformer)

Writes paper/source/v20_analysis.json. Usage (host; CPU is enough, ~10 min):
    python scripts/analysis_v20.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.heldout_metrics import MCE_MIN_FRAC, classification_metrics, per_patient  # noqa: E402
from scripts.analysis_v19 import load_probs, pooled_bootstrap, temperature_crossfit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
BUILD = ROOT / "data/hsi_v9-trainsel"
MEDMAMBA_RUNS = ROOT.parent / "medmamba-original/MedMamba/runs_hsi"
OUT = ROOT / "paper/source/v20_analysis.json"
SEEDS = (1, 7, 13, 23, 42)
CONTROL_SEEDS = (1, 7, 42)
KEYS = ("balanced_accuracy", "f1_macro", "accuracy", "cohen_kappa", "ece", "brier_score", "nll",
        "roc_auc_macro")

_INDEX = None
MISSING: list = []


# --------------------------------------------------------------------------- run discovery
def index_runs():
    global _INDEX
    if _INDEX is None:
        _INDEX = []
        for c in sorted(glob.glob(str(EXP / "*/config.json"))):
            try:
                cfg = json.load(open(c))
            except Exception:
                continue
            a = cfg.get("cli_args", {})
            _INDEX.append({"dir": Path(c).parent, "tag": a.get("run_tag"),
                           "data_dir": str(a.get("data_dir", "")),
                           "arch": cfg.get("architecture", a.get("arch"))})
    return _INDEX


def find_run(tag, data_contains=None, arch=None, need="test_predictions.npz"):
    hits = [r for r in index_runs() if r["tag"] == tag
            and (data_contains is None or data_contains in r["data_dir"])
            and (arch is None or r["arch"] == arch)
            and (r["dir"] / need).is_file()]
    if not hits:
        MISSING.append({"tag": tag, "data": data_contains, "arch": arch})
        return None
    return hits[-1]["dir"]          # latest complete run with that tag


def val_preds_path(rd: Path):
    for p in (rd / "heldout_v19/validation/test_predictions.npz", rd / "val_predictions.npz",
              rd / "predictions_val_full.npz"):
        if p.is_file():
            return p
    return None


def test_preds_path(rd: Path):
    for p in (rd / "test_predictions.npz", rd / "predictions_test.npz"):
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------- one run
def score_run(rd: Path, g_test, g_val, num_classes=3):
    pt, lt = load_probs(test_preds_path(rd))
    e = {"run": rd.name, "test": classification_metrics(lt, pt, num_classes=num_classes),
         "test_per_patient": per_patient(lt, pt.argmax(1), g_test)}
    vp = val_preds_path(rd)
    if vp is not None and g_val is not None:
        pv, lv = load_probs(vp)
        e["val_full"] = classification_metrics(lv, pv, num_classes=num_classes)
        e["val_per_patient"] = per_patient(lv, pv.argmax(1), g_val)
        e["pooled"] = classification_metrics(np.concatenate([lv, lt]), np.concatenate([pv, pt]),
                                             num_classes=num_classes)
        e["temperature"] = temperature_crossfit(pt, lt, pv, lv)
        e["_preds"] = {"test": pt.argmax(1), "val": pv.argmax(1)}
    else:
        e["_preds"] = {"test": pt.argmax(1)}
    cfg = rd / "config.json"
    if cfg.is_file():
        c = json.load(open(cfg))
        e["params"] = c.get("backbone_num_params")
    rep = rd / "test_report.json"
    if rep.is_file():
        e["best_epoch"] = json.load(open(rep)).get("best_epoch")
    return e


def mean_sd(rows, split):
    vals = [r[split] for r in rows if split in r]
    if len(vals) < 2:
        return None
    return {k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals], ddof=1))]
            for k in KEYS if all(k in v for v in vals)} | {"n": len(vals)}


def strip(e):
    return {k: v for k, v in e.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- sections
def headline(y_test, y_val, g_test, g_val):
    per_seed, pairs = {}, {"test": [], "val": [], "pooled": []}
    for s in SEEDS:
        row = {}
        for mod in ("hsi", "rgb"):
            rd = find_run(f"v20-head-s{s}", f"hsi_v9-trainsel/{mod}")
            if rd is not None:
                row[mod] = score_run(rd, g_test, g_val)
        if row:
            per_seed[str(s)] = row
        if "hsi" in row and "rgb" in row:
            h, r = row["hsi"]["_preds"], row["rgb"]["_preds"]
            pairs["test"].append((y_test, h["test"], r["test"]))
            if "val" in h and "val" in r:
                pairs["val"].append((y_val, h["val"], r["val"]))
                pairs["pooled"].append((np.concatenate([y_val, y_test]),
                                        np.concatenate([h["val"], h["test"]]),
                                        np.concatenate([r["val"], r["test"]])))
    summ = {}
    for mod in ("hsi", "rgb"):
        rows = [per_seed[s][mod] for s in per_seed if mod in per_seed[s]]
        for split in ("test", "val_full", "pooled"):
            ms = mean_sd(rows, split)
            if ms:
                summ[f"{split}_{mod}"] = ms
    for split in ("test", "val_full", "pooled"):
        d = [per_seed[s]["hsi"][split]["balanced_accuracy"] - per_seed[s]["rgb"][split]["balanced_accuracy"]
             for s in per_seed if "hsi" in per_seed[s] and "rgb" in per_seed[s]
             and split in per_seed[s]["hsi"] and split in per_seed[s]["rgb"]]
        if len(d) >= 2:
            summ[f"{split}_delta"] = {"bal_mean": float(np.mean(d)), "bal_sd": float(np.std(d, ddof=1)),
                                      "bal_pos": int(np.sum(np.array(d) > 0)), "n": len(d), "per_seed": d}
    boot = {}
    if len(pairs["test"]) >= 2:
        boot["test"] = pooled_bootstrap(pairs["test"], g_test)
    if len(pairs["val"]) >= 2:
        boot["val"] = pooled_bootstrap(pairs["val"], g_val)
        boot["pooled"] = pooled_bootstrap(pairs["pooled"], np.concatenate([g_val, g_test]))
    pp = {}
    for split, key in (("test", "test_per_patient"), ("val", "val_per_patient")):
        for mod in ("hsi", "rgb"):
            recs = [per_seed[s][mod][key] for s in per_seed if mod in per_seed[s] and key in per_seed[s][mod]]
            for r in recs:
                for pid, v in r.items():
                    pp.setdefault(pid, {"split": split, "n": v["n"]}).setdefault(mod, []).append(v["macro_recall"])
    return per_seed, summ, boot, pp


def baselines(g_test, g_val, trm_per_seed):
    out = {}
    for arch in ("hybridsn", "spectralformer"):
        rows = {}
        for s in SEEDS:
            rd = find_run(f"v20-s{s}", "hsi_v9-trainsel/hsi", arch=arch)
            if rd is not None:
                rows[str(s)] = score_run(rd, g_test, g_val)
        out[arch] = rows
    rows = {}
    for s in SEEDS:
        rd = MEDMAMBA_RUNS / f"v20_s{s}"
        if (rd / "predictions_test.npz").is_file():
            rows[str(s)] = score_run(rd, g_test, g_val)
            rows[str(s)]["params"] = json.load(open(rd / "experiment_report.json"))["backbone_num_params"]
        else:
            MISSING.append({"medmamba": str(rd)})
    out["medmamba"] = rows
    summary = {}
    for name, rows in out.items():
        summary[name] = {split: mean_sd(list(rows.values()), split) for split in ("test", "val_full", "pooled")}
        # paired by seed: does MedMamba-SS-TRM (hsi) beat this baseline?
        wins = {}
        for split in ("test", "val_full", "pooled"):
            d = [trm_per_seed[s]["hsi"][split]["balanced_accuracy"] - rows[s][split]["balanced_accuracy"]
                 for s in rows if s in trm_per_seed and "hsi" in trm_per_seed[s]
                 and split in trm_per_seed[s]["hsi"] and split in rows[s]]
            if d:
                wins[split] = {"trm_minus_baseline_mean": float(np.mean(d)), "trm_wins": int(np.sum(np.array(d) > 0)),
                               "n": len(d), "per_seed": d}
        summary[name]["paired_vs_trm"] = wins
    return {k: {s: strip(v) for s, v in rows.items()} for k, rows in out.items()}, summary


def probes():
    out = {}
    for mod in ("hsi", "rgb"):
        pj = BUILD / mod / "shallow_probe_heldout_v19.json"
        if not pj.is_file():
            MISSING.append({"probes": str(pj)})
            continue
        for kind, r in json.load(open(pj))["results"].items():
            tp = BUILD / mod / f"shallow_probe_heldout_v19_{kind}_test.npz"
            vp = BUILD / mod / f"shallow_probe_heldout_v19_{kind}_val.npz"
            e = {"dim": r["dim"]}
            if tp.is_file() and vp.is_file():
                pt, lt = load_probs(tp)
                pv, lv = load_probs(vp)
                e.update(test=classification_metrics(lt, pt), val_full=classification_metrics(lv, pv),
                         pooled=classification_metrics(np.concatenate([lv, lt]), np.concatenate([pv, pt])),
                         temperature=temperature_crossfit(pt, lt, pv, lv))
            out[f"{mod}_{kind}"] = e
    return out


def test_only(tag, data="hsi_v9-trainsel/hsi", arch=None):
    rd = find_run(tag, data, arch=arch, need="test_report.json")
    if rd is None:
        return None
    t = json.load(open(rd / "test_report.json"))
    m = t["sklearn_metrics"]
    return {"run": rd.name, "bal": m["balanced_accuracy"], "f1": m["f1_macro"], "acc": m["accuracy"],
            "per_class_f1": m["per_class_f1"], "gflops": (t.get("flops") or {}).get("total_gflops_approx"),
            "params": json.load(open(rd / "config.json")).get("backbone_num_params")}


def controls():
    arms = {"baseline": "v20-abl-base-s{}", "k2_no_recursion": "v20-k2-s{}",
            "recon_off": "v20-recon-off-s{}", "index_encoding": "v20-enc-index-s{}"}
    out = {}
    for arm, pat in arms.items():
        rows = {str(s): test_only(pat.format(s)) for s in CONTROL_SEEDS}
        rows = {s: r for s, r in rows.items() if r}
        out[arm] = {"per_seed": rows}
        if len(rows) >= 2:
            out[arm]["mean_sd"] = {k: [float(np.mean([r[k] for r in rows.values()])),
                                       float(np.std([r[k] for r in rows.values()], ddof=1))]
                                   for k in ("bal", "f1", "acc")}
    base = out["baseline"]["per_seed"]
    for arm in arms:
        if arm == "baseline":
            continue
        d = [out[arm]["per_seed"][s]["bal"] - base[s]["bal"] for s in out[arm]["per_seed"] if s in base]
        if d:
            out[arm]["delta_vs_baseline"] = {"bal_mean": float(np.mean(d)), "n": len(d),
                                             "worse_at": int(np.sum(np.array(d) < 0)), "per_seed": d}
    return out


def cross_validation():
    folds, pooled = {}, {"hsi": ([], [], []), "rgb": ([], [], [])}
    for f in range(5):
        root = ROOT / f"data/hsi_v9-cv5-f{f}"
        if not (root / "hsi/groups_test.npy").is_file():
            MISSING.append({"cv_fold_data": str(root)})
            continue
        g_test = np.load(root / "hsi/groups_test.npy", allow_pickle=True)
        row = {}
        for mod in ("hsi", "rgb"):
            rd = find_run(f"v20-cv-f{f}", f"hsi_v9-cv5-f{f}/{mod}")
            if rd is None:
                continue
            pt, lt = load_probs(test_preds_path(rd))
            row[mod] = {"run": rd.name, "test": classification_metrics(lt, pt),
                        "test_per_patient": per_patient(lt, pt.argmax(1), g_test)}
            pooled[mod][0].append(lt); pooled[mod][1].append(pt); pooled[mod][2].append(g_test)
        folds[str(f)] = row
    out = {"folds": folds}
    for mod in ("hsi", "rgb"):
        if pooled[mod][0]:
            y = np.concatenate(pooled[mod][0]); p = np.concatenate(pooled[mod][1])
            g = np.concatenate(pooled[mod][2])
            out[f"pooled_{mod}"] = classification_metrics(y, p)
            out[f"pooled_{mod}_per_patient"] = per_patient(y, p.argmax(1), g)
            out[f"pooled_{mod}_n_patients"] = int(len(set(g.tolist())))
    d = [folds[f]["hsi"]["test"]["balanced_accuracy"] - folds[f]["rgb"]["test"]["balanced_accuracy"]
         for f in folds if "hsi" in folds[f] and "rgb" in folds[f]]
    if d:
        out["fold_delta_bal"] = {"mean": float(np.mean(d)), "pos": int(np.sum(np.array(d) > 0)), "n": len(d),
                                 "per_fold": d}
    # Patients that were TRAINING patients throughout recipe development (v8 fixed split):
    # their CV scores cannot have influenced any design decision.
    v8 = json.load(open(ROOT / "data/hsi_v8-80_10_10_importance-new/split_assignment.json"))["groups"]
    never_heldout = {int(g) for g, sp in v8.items() if sp == "train"}
    for mod in ("hsi", "rgb"):
        if pooled[mod][0]:
            y = np.concatenate(pooled[mod][0]); p = np.concatenate(pooled[mod][1])
            g = np.concatenate(pooled[mod][2]).astype(int)
            m = np.isin(g, list(never_heldout))
            out[f"pooled_{mod}_never_heldout"] = classification_metrics(y[m], p[m])
            out[f"pooled_{mod}_never_heldout_n_patients"] = int(len(set(g[m].tolist())))
    same = [f for f in folds if "hsi" in folds[f] and "rgb" in folds[f]]
    if len(same) == len(folds) and same:
        yh = np.concatenate(pooled["hsi"][0]); yr = np.concatenate(pooled["rgb"][0])
        if len(yh) == len(yr) and (yh == yr).all():
            ph = np.concatenate(pooled["hsi"][1]).argmax(1); pr = np.concatenate(pooled["rgb"][1]).argmax(1)
            g = np.concatenate(pooled["hsi"][2])
            out["bootstrap_hsi_minus_rgb"] = pooled_bootstrap([(yh, ph, pr)], g)
    return out


def zero_shot(head_s42):
    out = {}
    if head_s42 is None:
        return out
    rd = EXP / head_s42
    for key, sub in (("variable_eta", "band_decimation"), ("fixed_eta", "band_decimation_fixed_eta")):
        p = rd / sub / "summary.json"
        if p.is_file():
            out[key] = {r["n_bands"]: {"bal": r["balanced_accuracy"], "f1": r["f1_macro"],
                                       "per_class_f1": r["per_class_f1"]} for r in json.load(open(p))["rows"]}
        else:
            MISSING.append({"zero_shot": str(p)})
    return out


def lusc_cv():
    """Second dataset: HMI-LUSC, patient-level 5-fold CV, binary tumour vs non-tumour."""
    out = {}
    for arch in ("recursive", "hybridsn", "spectralformer"):
        ys, ps, gs, folds = [], [], [], {}
        for f in range(5):
            data = f"lusc_v20-f{f}/hsi"
            rd = find_run(f"v20-lusc-f{f}", data, arch=arch)
            if rd is None:
                continue
            pt, lt = load_probs(test_preds_path(rd))
            g = np.load(ROOT / "data" / data / "groups_test.npy")
            folds[str(f)] = {"run": rd.name, "test": classification_metrics(lt, pt, num_classes=2),
                             "params": json.load(open(rd / "config.json")).get("backbone_num_params")}
            ys.append(lt); ps.append(pt); gs.append(g)
        e = {"folds": folds}
        if ys:
            y, pr, g = np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)
            e["pooled"] = classification_metrics(y, pr, num_classes=2)
            e["per_patient"] = per_patient(y, pr.argmax(1), g, num_classes=2)
            e["n_patients"] = int(len(set(g.tolist())))
            if len(folds) >= 2:
                e["fold_mean_sd_bal"] = [float(np.mean([v["test"]["balanced_accuracy"] for v in folds.values()])),
                                         float(np.std([v["test"]["balanced_accuracy"] for v in folds.values()],
                                                      ddof=1))]
        out[arch] = e
    return out


def eval_mode():
    hits = sorted(glob.glob(str(EXP / "*/eval_mode_check_v20.json")))
    return {Path(h).parent.name: json.load(open(h)) for h in hits}


def band_scope():
    p = BUILD / "pass1_scope_report.json"
    sel = BUILD / "selected_wavelengths.npy"
    out = json.load(open(p)) if p.is_file() else None
    if sel.is_file():
        wl = np.load(sel).astype(float)
        out = (out or {}) | {"selected_nm": [round(v, 1) for v in wl.tolist()],
                             "largest_gap_nm": float(np.max(np.diff(np.sort(wl))))}
    return out


def main():
    hsi = BUILD / "hsi"
    if not (hsi / "y_test.npy").is_file():
        print(f"[stop] {hsi} not prepared yet (plan stage G1).")
        return 1
    y_test, y_val = np.load(hsi / "y_test.npy"), np.load(hsi / "y_val.npy")
    g_test = np.load(hsi / "groups_test.npy", allow_pickle=True)
    g_val = np.load(hsi / "groups_val.npy", allow_pickle=True)

    per_seed, summ, boot, pp = headline(y_test, y_val, g_test, g_val)
    base_rows, base_summ = baselines(g_test, g_val, per_seed)
    out = {
        "build": str(BUILD.relative_to(ROOT)), "band_selection_scope": band_scope(),
        "mce_min_frac": MCE_MIN_FRAC,
        "headline": {"per_seed": {s: {m: strip(v) for m, v in r.items()} for s, r in per_seed.items()},
                     "summary": summ, "bootstrap_bal_delta": boot, "per_patient_over_seeds": pp},
        "baselines": {"per_seed": base_rows, "summary": base_summ},
        "probes": probes(),
        "controls_12ep": controls(),
        "depth_12ep": {T: test_only(f"v20-depth-n{T}") for T in (1, 2, 4)},
        "bands_12ep": {N: test_only(f"v20-bands{N}", f"hsi_v9-bands{N}") for N in (16, 8)},
        "hierarchical_12ep": {a: test_only(f"v20-arch-{a}") for a in ("split", "fullchannel")},
        "zero_shot": zero_shot(per_seed.get("42", {}).get("hsi", {}).get("run")),
        "cross_validation": cross_validation(),
        "lusc_cv": lusc_cv(),
        "eval_mode_check": eval_mode(),
    }
    have = sum(len(r) for r in per_seed.values())
    out["status"] = {"headline_runs_found": have, "headline_runs_expected": 2 * len(SEEDS),
                     "missing": MISSING}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"[written] {OUT}")
    print(f"[status] headline {have}/{2 * len(SEEDS)} runs; {len(MISSING)} missing item(s)")
    for k in ("test", "val_full", "pooled"):
        if f"{k}_delta" in summ:
            d = summ[f"{k}_delta"]
            print(f"  HSI-RGB {k:8s} {100 * d['bal_mean']:+.2f} pts (positive at {d['bal_pos']}/{d['n']})")
    for name, s in base_summ.items():
        w = s.get("paired_vs_trm", {}).get("pooled")
        if w:
            print(f"  TRM vs {name:15s} pooled: {100 * w['trm_minus_baseline_mean']:+.2f} pts, "
                  f"TRM ahead at {w['trm_wins']}/{w['n']} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
