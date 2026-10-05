#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/analysis_v19.py
=======================
Aggregate every number the v11 manuscript reports from saved predictions and
run records, with one metric implementation (training/heldout_metrics.py).
Read-only on every run directory. Writes paper/source/v19_analysis.json.

Sections of the output:
  modality     per seed x {hsi, rgb}: test / full-validation / pooled metrics,
               per-patient macro recall, calibration; seed-level summaries;
               patient-bootstrap intervals for the HSI - RGB difference.
  comparison   the primary-dataset comparison table (probes, MedMamba,
               HybridSN, SpectralFormer, MedMamba-SS, MedMamba-SS-TRM).
  depth, bands, ablations, recon   read from their run records.
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
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/hsi_v8-80_10_10_importance-new"
EXP = ROOT / "experiments"
MEDMAMBA = ROOT.parent / "medmamba-original/MedMamba/runs_hsi/matched_natural_v8new"

SEED_RUNS = {  # seed: (hsi prefix, rgb prefix)
    42: ("20260915_031356", "20260915_031947"), 1: ("20260916_084156", "20260916_131540"),
    7: ("20260916_155129", "20260916_192436"), 13: ("20260919_150920", "20260919_184739"),
    23: ("20260919_205302", "20260920_003948"),
}
DEPTH = {21: "20260915_123136", 42: "20260915_150351", 63: "20260915_031555", 84: "20260915_183538"}
ABL = {"baseline": "20260915_031555", "recon_off": "20260915_160442", "index_enc_a": "20260915_202942",
       "index_enc_b": "20260916_013913", "bands16": "20260919_015237", "bands8": "20260919_020050",
       "split": "20260915_083810", "fullchannel": "20260915_152009"}


def run_dir(prefix):
    hits = sorted(glob.glob(str(EXP / f"{prefix}_*")))
    return Path(hits[0]) if hits else None


def load_probs(path):
    d = np.load(path, allow_pickle=True)
    if "probs" in d.files:
        return d["probs"], d["true_label"]
    return d["probabilities"], d["labels"]


def fit_temperature(probs_val, y_val):
    """Temperature T minimizing validation NLL of softmax(log p / T)."""
    from scipy.optimize import minimize_scalar
    lp = np.log(np.clip(probs_val.astype(np.float64), 1e-12, 1.0))

    def nll(t):
        z = lp / t
        z -= z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(y_val)), y_val].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature(probs, t):
    z = np.log(np.clip(probs.astype(np.float64), 1e-12, 1.0)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def temperature_crossfit(probs_t, lab_t, probs_v, lab_v):
    """Fit T on one held-out set and score the other, in both directions."""
    t = fit_temperature(probs_v, lab_v)
    ts = classification_metrics(lab_t, apply_temperature(probs_t, t))
    t2 = fit_temperature(probs_t, lab_t)
    vs = classification_metrics(lab_v, apply_temperature(probs_v, t2))
    return {"T": t, "test_ece": ts["ece"], "test_nll": ts["nll"], "test_brier": ts["brier_score"],
            "T_test": t2, "val_ece": vs["ece"], "val_nll": vs["nll"], "val_brier": vs["brier_score"]}


def pooled_bootstrap(pairs, groups_all, n_boot=2000, seed=0):
    """pairs: list over seeds of (labels, preds_hsi, preds_rgb) on the SAME
    patches. Resample PATIENTS with replacement; statistic = mean over seeds of
    bal_acc(hsi) - bal_acc(rgb). Balanced accuracy of a resample is computed
    exactly from per-patient per-class counts (correct, total)."""
    uniq = np.array(sorted(set(groups_all.tolist())))
    gidx = {g: i for i, g in enumerate(uniq)}
    gi = np.array([gidx[g] for g in groups_all.tolist()])
    ncls = 3
    tot = np.zeros((len(uniq), ncls))
    np.add.at(tot, (gi, pairs[0][0]), 1)
    corr = []
    for y, a, b in pairs:
        ca, cb = np.zeros((len(uniq), ncls)), np.zeros((len(uniq), ncls))
        np.add.at(ca, (gi, y), (a == y))
        np.add.at(cb, (gi, y), (b == y))
        corr.append((ca, cb))

    def stat(w):
        t = (w[:, None] * tot).sum(0)
        if (t == 0).any():
            return None
        return float(np.mean([((w[:, None] * ca).sum(0) / t).mean() - ((w[:, None] * cb).sum(0) / t).mean()
                              for ca, cb in corr]))
    obs = stat(np.ones(len(uniq)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        v = stat(w)
        if v is not None:
            draws.append(v)
    draws = np.asarray(draws)
    return {"observed": obs, "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "frac_positive": float((draws > 0).mean()), "n_boot": int(len(draws)),
            "n_patients": int(len(uniq))}


def main():
    y_test = np.load(DATA / "hsi/y_test.npy")
    y_val = np.load(DATA / "hsi/y_val.npy")
    g_test = np.load(DATA / "hsi/groups_test.npy")
    g_val = np.load(DATA / "hsi/groups_val.npy")
    out = {"modality": {"per_seed": {}}, "comparison": {}, "mce_min_frac": MCE_MIN_FRAC}

    # ---------------- modality ----------------
    pairs = {"test": [], "val": [], "pooled": []}
    for seed, (ph, pr) in SEED_RUNS.items():
        row = {}
        preds = {}
        for mod, prefix in (("hsi", ph), ("rgb", pr)):
            rd = run_dir(prefix)
            probs_t, lab_t = load_probs(rd / "test_predictions.npz")
            rec = {"run": rd.name, "test": classification_metrics(lab_t, probs_t),
                   "test_per_patient": per_patient(lab_t, probs_t.argmax(1), g_test)}
            preds[(mod, "test")] = probs_t.argmax(1)
            vp = rd / "heldout_v19/validation/test_predictions.npz"
            if vp.is_file():
                probs_v, lab_v = load_probs(vp)
                assert (lab_v == y_val).all()
                rec["val_full"] = classification_metrics(lab_v, probs_v)
                rec["val_per_patient"] = per_patient(lab_v, probs_v.argmax(1), g_val)
                preds[(mod, "val")] = probs_v.argmax(1)
                lab_p = np.concatenate([lab_v, lab_t])
                prob_p = np.concatenate([probs_v, probs_t])
                rec["pooled"] = classification_metrics(lab_p, prob_p)
                rec["temperature"] = temperature_crossfit(probs_t, lab_t, probs_v, lab_v)
            row[mod] = rec
        out["modality"]["per_seed"][str(seed)] = row
        pairs["test"].append((y_test, preds[("hsi", "test")], preds[("rgb", "test")]))
        if ("hsi", "val") in preds and ("rgb", "val") in preds:
            pairs["val"].append((y_val, preds[("hsi", "val")], preds[("rgb", "val")]))
            pairs["pooled"].append((np.concatenate([y_val, y_test]),
                                    np.concatenate([preds[("hsi", "val")], preds[("hsi", "test")]]),
                                    np.concatenate([preds[("rgb", "val")], preds[("rgb", "test")]])))

    summ = {}
    for split in ("test", "val_full", "pooled"):
        for mod in ("hsi", "rgb"):
            vals = [out["modality"]["per_seed"][s][mod].get(split) for s in out["modality"]["per_seed"]]
            if any(v is None for v in vals):
                continue
            summ[f"{split}_{mod}"] = {k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals], ddof=1))]
                                      for k in ("balanced_accuracy", "f1_macro", "accuracy", "ece", "mce",
                                                "mce_floor", "brier_score", "nll", "roc_auc_macro")}
            summ[f"{split}_{mod}"]["per_class_f1"] = np.mean([v["per_class_f1"] for v in vals], axis=0).tolist()
            summ[f"{split}_{mod}"]["mce_bin_n_min"] = int(min(v["mce_bin_n"] for v in vals))
        if f"{split}_hsi" in summ and f"{split}_rgb" in summ:
            d_bal = [out["modality"]["per_seed"][s]["hsi"][split]["balanced_accuracy"]
                     - out["modality"]["per_seed"][s]["rgb"][split]["balanced_accuracy"] for s in out["modality"]["per_seed"]]
            d_f1 = [out["modality"]["per_seed"][s]["hsi"][split]["f1_macro"]
                    - out["modality"]["per_seed"][s]["rgb"][split]["f1_macro"] for s in out["modality"]["per_seed"]]
            summ[f"{split}_delta"] = {"bal_mean": float(np.mean(d_bal)), "bal_sd": float(np.std(d_bal, ddof=1)),
                                      "bal_pos": int(np.sum(np.array(d_bal) > 0)), "f1_mean": float(np.mean(d_f1)),
                                      "f1_sd": float(np.std(d_f1, ddof=1)), "f1_pos": int(np.sum(np.array(d_f1) > 0)),
                                      "bal_per_seed": d_bal, "f1_per_seed": d_f1}
    out["modality"]["summary"] = summ
    boot = {}
    if pairs["test"]:
        boot["test"] = pooled_bootstrap(pairs["test"], g_test)
    if len(pairs["val"]) == 5:
        boot["val"] = pooled_bootstrap(pairs["val"], g_val)
        boot["pooled"] = pooled_bootstrap(pairs["pooled"], np.concatenate([g_val, g_test]))
    out["modality"]["bootstrap_bal_delta"] = boot

    # per-patient mean over seeds, all held-out patients
    pp = {}
    for split, key in (("test", "test_per_patient"), ("val", "val_per_patient")):
        for mod in ("hsi", "rgb"):
            recs = [out["modality"]["per_seed"][s][mod].get(key) for s in out["modality"]["per_seed"]]
            if any(r is None for r in recs):
                continue
            for pid in recs[0]:
                pp.setdefault(pid, {"split": split})[mod] = [r[pid]["macro_recall"] for r in recs]
                pp[pid]["n"] = recs[0][pid]["n"]
    out["modality"]["per_patient_over_seeds"] = pp

    # ---------------- comparison (primary dataset) ----------------
    comp = {}
    ref = out["modality"]["per_seed"]["42"]["hsi"]
    comp["MedMamba-SS-TRM (seed 42)"] = {"params": 446409, "test": ref["test"], "val_full": ref.get("val_full")}
    comp["MedMamba-SS-TRM (5-seed mean)"] = {"params": 446409, "test_mean_sd": summ.get("test_hsi"),
                                             "val_mean_sd": summ.get("val_full_hsi")}
    probs, lab = load_probs(MEDMAMBA / "predictions_test.npz")
    comp["MedMamba"] = {"params": 3648995, "test": classification_metrics(lab, probs),
                        "test_per_patient": per_patient(lab, probs.argmax(1), g_test)}
    for arch in ("hybridsn", "spectralformer"):
        hits = sorted(glob.glob(str(EXP / f"*_{arch}_baseline_s42")))
        if hits:
            rd = Path(hits[-1])
            rep = json.load(open(rd / "test_report.json")) if (rd / "test_report.json").is_file() else None
            if rep:
                pt, lt = load_probs(rd / "test_predictions.npz")
                pv, lv = load_probs(rd / "val_predictions.npz")
                comp[arch] = {"params": rep["backbone_num_params"], "test": classification_metrics(lt, pt),
                              "val_full": classification_metrics(lv, pv),
                              "pooled": classification_metrics(np.concatenate([lv, lt]), np.concatenate([pv, pt])),
                              "test_per_patient": per_patient(lt, pt.argmax(1), g_test),
                              "val_per_patient": per_patient(lv, pv.argmax(1), g_val),
                              "temperature": temperature_crossfit(pt, lt, pv, lv),
                              "best_epoch": rep["best_epoch"], "run": rd.name}
    # MedMamba: full-validation predictions were not kept, but its epoch-1 validation record has
    # per-class precision, recall and support, which fix TP, FN and FP per class exactly.
    h = json.load(open(MEDMAMBA / "training_history.json"))[0]
    tm = comp["MedMamba"]["test"]
    tp = fn = fp = 0
    tp_c, fn_c, fp_c = [], [], []
    for c in range(3):
        n_v, r_v, p_v = h["per_class_support"][c], h["per_class_recall"][c], h["per_class_precision"][c]
        n_t, r_t, p_t = tm["per_class_support"][c], tm["per_class_recall"][c], tm["per_class_precision"][c]
        tpv, tpt = r_v * n_v, r_t * n_t
        tp_c.append(tpv + tpt); fn_c.append(n_v - tpv + n_t - tpt)
        fp_c.append(tpv / p_v - tpv + tpt / p_t - tpt)
    rec_ = [tp_c[c] / (tp_c[c] + fn_c[c]) for c in range(3)]
    pre_ = [tp_c[c] / (tp_c[c] + fp_c[c]) for c in range(3)]
    f1_ = [2 * pre_[c] * rec_[c] / (pre_[c] + rec_[c]) for c in range(3)]
    comp["MedMamba"]["val_full"] = {"balanced_accuracy": h["val_balanced_accuracy"], "f1_macro": h["val_f1_macro"],
                                    "accuracy": h["val_accuracy"]}
    comp["MedMamba"]["pooled"] = {"balanced_accuracy": float(np.mean(rec_)), "f1_macro": float(np.mean(f1_)),
                                  "derived_from": "per-class P/R/support of validation epoch 1 and test"}
    for mod in ("hsi", "rgb"):
        pj = DATA / mod / "shallow_probe_heldout_v19.json"
        if pj.is_file():
            pr = json.load(open(pj))["results"]
            for kind, r in pr.items():
                tp = DATA / mod / f"shallow_probe_heldout_v19_{kind}_test.npz"
                entry = {"dim": r["dim"], "test_simple": r["test"], "val_simple": r["val"]}
                vp = DATA / mod / f"shallow_probe_heldout_v19_{kind}_val.npz"
                if tp.is_file():
                    probs, lab = load_probs(tp)
                    entry["test"] = classification_metrics(lab, probs)
                if tp.is_file() and vp.is_file():
                    pv, lv = load_probs(vp)
                    entry["val_full"] = classification_metrics(lv, pv)
                    entry["pooled"] = classification_metrics(np.concatenate([lv, lab]), np.concatenate([pv, probs]))
                    entry["temperature"] = temperature_crossfit(probs, lab, pv, lv)
                comp[f"probe_{mod}_{kind}"] = entry
    maj = int(np.bincount(np.load(DATA / "hsi/y_train.npy")).argmax())
    y_pool = np.concatenate([y_val, y_test])
    comp["Majority"] = {"params": 0, "class": maj,
                        "test": classification_metrics(y_test, preds=np.full_like(y_test, maj)),
                        "val_full": classification_metrics(y_val, preds=np.full_like(y_val, maj)),
                        "pooled": classification_metrics(y_pool, preds=np.full_like(y_pool, maj))}
    for name in ("split", "fullchannel"):
        rd = run_dir(ABL[name])
        if rd:
            pt, lt = load_probs(rd / "test_predictions.npz")
            e = {"params": json.load(open(rd / "config.json"))["backbone_num_params"], "run": rd.name,
                 "best_epoch": json.load(open(rd / "test_report.json")).get("best_epoch"),
                 "test": classification_metrics(lt, pt), "test_per_patient": per_patient(lt, pt.argmax(1), g_test)}
            vp_ = rd / "heldout_v19/validation/test_predictions.npz"
            if vp_.is_file():
                pv, lv = load_probs(vp_)
                e["val_full"] = classification_metrics(lv, pv)
                e["pooled"] = classification_metrics(np.concatenate([lv, lt]), np.concatenate([pv, pt]))
                e["val_per_patient"] = per_patient(lv, pv.argmax(1), g_val)
            comp[f"MedMamba-SS ({name})"] = e
    out["comparison"] = comp

    # ---------------- depth / ablations / bands ----------------
    dep = {}
    for k, prefix in DEPTH.items():
        rd = run_dir(prefix)
        t = json.load(open(rd / "test_report.json"))
        dep[k] = {"bal": t["sklearn_metrics"]["balanced_accuracy"], "f1": t["sklearn_metrics"]["f1_macro"],
                  "dcis_f1": t["sklearn_metrics"]["per_class_f1"][1],
                  "gflops": (t.get("flops") or {}).get("total_gflops_approx"), "run": rd.name}
    out["depth"] = dep
    abl = {}
    for k, prefix in ABL.items():
        rd = run_dir(prefix)
        if rd and (rd / "test_report.json").is_file():
            t = json.load(open(rd / "test_report.json"))["sklearn_metrics"]
            abl[k] = {"bal": t["balanced_accuracy"], "f1": t["f1_macro"], "per_class_f1": t["per_class_f1"],
                      "run": rd.name}
    out["ablations"] = abl
    bd = json.load(open(run_dir(SEED_RUNS[42][0]) / "band_decimation/summary.json"))
    out["band_zero_shot"] = {r["n_bands"]: {"bal": r["balanced_accuracy"], "f1": r["f1_macro"],
                                            "per_class_f1": r["per_class_f1"], "nm": r["wavelengths_nm"]}
                             for r in bd["rows"]}

    # ---------------- whole-image PAD-UFES-20 ----------------
    def macro_specificity(y, p, k):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, p, labels=list(range(k)))
        n = cm.sum()
        return float(np.mean([(n - cm[c].sum() - cm[:, c].sum() + cm[c, c]) / (n - cm[c].sum()) for c in range(k)]))

    pad_src = {
        "MedMamba-SS-TRM": (run_dir("20260906_183513") / "test_predictions.npz", 446796, "recursive, lr 1e-3"),
        "MedMamba-SS": (run_dir("20260916_032044") / "test_predictions.npz", 27425314, "hierarchical, run 1"),
        "MedMamba-SS (repeat)": (run_dir("20260916_084319") / "test_predictions.npz", 27425314, "hierarchical, run 2"),
    }
    pad = {}
    for lab_, (path, params, note) in pad_src.items():
        pr, y = load_probs(path)
        m = classification_metrics(y, pr, num_classes=6)
        m["specificity_macro"] = macro_specificity(y, pr.argmax(1), 6)
        pad[lab_] = {"params": params, "note": note, "test": m, "source": str(path.relative_to(ROOT))}
    mt = ROOT.parent / "medmamba-original/MedMamba/runs/PAD-UFES-20-paper/results"
    pr, y = np.load(mt / "test_probabilities.npy"), np.load(mt / "test_labels.npy").astype(np.int64)
    m = classification_metrics(y, pr, num_classes=6)
    m["specificity_macro"] = macro_specificity(y, pr.argmax(1), 6)
    pad["MedMamba-T"] = {"params": 14.47e6, "note": "image-level 60/10/30 split", "test": m,
                         "reported": json.load(open(mt / "test_metrics.json")),
                         "source": "medmamba-original/MedMamba/runs/PAD-UFES-20-paper/results"}
    out["pad_whole_image"] = pad

    # ---------------- selective prediction (risk-coverage) ----------------
    def aurc_acc90(pr, y):
        conf, pred = pr.max(1), pr.argmax(1)
        o = np.argsort(-conf)
        err = (pred[o] != y[o]).astype(float)
        n90 = int(round(0.9 * len(y)))
        return float((np.cumsum(err) / np.arange(1, len(y) + 1)).mean()), float((pred[o[:n90]] == y[o[:n90]]).mean())
    sel = {}
    srcs = {"MedMamba-SS-TRM (HSI)": (run_dir(SEED_RUNS[42][0]) / "test_predictions.npz",
                                      run_dir(SEED_RUNS[42][0]) / "heldout_v19/validation/test_predictions.npz"),
            "MedMamba-SS-TRM (RGB)": (run_dir(SEED_RUNS[42][1]) / "test_predictions.npz",
                                      run_dir(SEED_RUNS[42][1]) / "heldout_v19/validation/test_predictions.npz"),
            "MedMamba": (MEDMAMBA / "predictions_test.npz", None)}
    for arch, lab in (("hybridsn", "HybridSN"), ("spectralformer", "SpectralFormer")):
        hits = sorted(glob.glob(str(EXP / f"*_{arch}_baseline_s42")))
        if hits:
            srcs[lab] = (Path(hits[-1]) / "test_predictions.npz", Path(hits[-1]) / "val_predictions.npz")
    for lab, (tp_, vp_) in srcs.items():
        pr, y = load_probs(tp_)
        e = dict(zip(("test_aurc", "test_acc90"), aurc_acc90(pr, y)))
        if vp_ is not None and Path(vp_).is_file():
            pr, y = load_probs(vp_)
            e.update(zip(("val_aurc", "val_acc90"), aurc_acc90(pr, y)))
        sel[lab] = e
    out["selective_prediction"] = sel

    dst = ROOT / "paper/source/v19_analysis.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1, default=float)
    print(f"[written] {dst}")
    for k, v in summ.items():
        if k.endswith("delta"):
            print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if "per_seed" not in kk})
    print("bootstrap", json.dumps(boot, indent=0)[:800])
    for name, c in comp.items():
        t = c.get("test") or c.get("test_sklearn") or {}
        if t:
            print(f"{name:34s} bal {t.get('balanced_accuracy', float('nan')):.4f} F1 {t.get('f1_macro', float('nan')):.4f}"
                  f" ece {t.get('ece', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
