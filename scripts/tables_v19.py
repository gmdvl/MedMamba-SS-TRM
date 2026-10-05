#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print the v11 manuscript's result tables as Markdown from
paper/source/v19_analysis.json, so no number is transcribed by hand."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = json.load(open(ROOT / "paper/source/v19_analysis.json"))


def f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}"


def comparison():
    c = A["comparison"]
    ps = A["modality"]["per_seed"]["42"]
    rows = []

    def cells(r):
        t, v, p = r.get("test") or {}, r.get("val_full") or {}, r.get("pooled") or {}
        return (f"{pct(t.get('accuracy'))} | {pct(t.get('balanced_accuracy'))} | {f(t.get('f1_macro'), 3)} | "
                f"{f(t.get('cohen_kappa'), 3)} | {pct(v.get('balanced_accuracy'))} | {f(v.get('f1_macro'), 3)} | "
                f"{pct(p.get('balanced_accuracy'))} | {f(p.get('f1_macro'), 3)}")
    spec = [("Majority class (always IDC)", "Majority", "—"),
            ("Logistic regression, band means", "probe_hsi_mean", "32 feat."),
            ("Logistic regression, band means + s.d.", "probe_hsi_mean_std", "64 feat."),
            ("Logistic regression, raw patch", "probe_hsi_flat", "3,872 feat."),
            ("Logistic regression, RGB means + s.d. (RGB build)", "probe_rgb_mean_std", "6 feat."),
            ("MedMamba [@medmamba]", "MedMamba", "3.649 M"),
            ("HybridSN [@hybridsn]", "hybridsn", "0.570 M"),
            ("SpectralFormer [@spectralformer]", "spectralformer", "0.121 M"),
            ("MedMamba-SS, 12-epoch budget", "MedMamba-SS (split)", "2.773 M"),
            ("MedMamba-SS, full-channel variant, 12-epoch budget", "MedMamba-SS (fullchannel)", "3.513 M")]
    for name, key, params in spec:
        if key in c:
            rows.append(f"| {name} | {params} | {cells(c[key])} |")
    rows.append(f"| **MedMamba-SS-TRM** | **0.446 M** | {cells(ps['hsi'])} |")
    print("| Model | Params | Acc. | BA | F1 | κ | BA | F1 | BA | F1 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print("\n".join(rows))


def ablations():
    ab, dep = A["ablations"], A["depth"]
    base = ab["baseline"]

    def row(name, bal, f1v, dcis, delta, bold=False):
        b = (lambda x: f"**{x}**") if bold else (lambda x: x)
        return f"| {name} | {b(bal)} | {b(f1v)} | {b(dcis)} | {delta} |"
    out = [row("MedMamba-SS-TRM, full", f"{base['bal']:.4f}", f"{base['f1']:.4f}",
               f"{base['per_class_f1'][1]:.3f}", "—", bold=True)]

    def d(x):
        return f"{x - base['f1']:+.3f}".replace("-", "\u2212")
    r = ab["recon_off"]
    out.append(row("− reconstruction objective ($\\lambda = 0$)", f"{r['bal']:.4f}", f"{r['f1']:.4f}",
                   f"{r['per_class_f1'][1]:.3f}", d(r["f1"])))
    a_, b_ = ab["index_enc_a"], ab["index_enc_b"]
    out.append(row("− wavelength encoding (index encoding; two runs)", f"{a_['bal']:.4f} / {b_['bal']:.4f}",
                   f"{a_['f1']:.4f} / {b_['f1']:.4f}", f"{a_['per_class_f1'][1]:.3f} / {b_['per_class_f1'][1]:.3f}",
                   d((a_["f1"] + b_["f1"]) / 2)))
    for key, name in (("bands16", "16 bands (retrained)"), ("bands8", "8 bands (retrained)")):
        r = ab[key]
        out.append(row(name, f"{r['bal']:.4f}", f"{r['f1']:.4f}", f"{r['per_class_f1'][1]:.3f}", d(r["f1"])))
    r = dep["21"]
    out.append(row("21 core applications instead of 63", f"{r['bal']:.4f}", f"{r['f1']:.4f}", f"{r['dcis_f1']:.3f}",
                   d(r["f1"])))
    for key, name in (("split", "hierarchical backbone (MedMamba-SS)"),
                      ("fullchannel", "hierarchical backbone, full-channel variant")):
        r = ab[key]
        out.append(row(name, f"{r['bal']:.4f}", f"{r['f1']:.4f}", f"{r['per_class_f1'][1]:.3f}", d(r["f1"])))
    print("| Variant | Balanced accuracy | Macro-F1 | DCIS F1 | Δ macro-F1 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    print("\n".join(out))


def pad():
    P = A["pad_whole_image"]
    cols = ["MedMamba-SS-TRM", "MedMamba-SS", "MedMamba-T"]
    static = [("Parameters", ["**0.45 M**", "27.4 M", "14.5 M"]),
              ("Split", ["patient-disjoint", "patient-disjoint", "image-level"]),
              ("Training images", ["234 (class-undersampled)", "1,626", "1,378"]),
              ("Loss", ["focal, weights $\\propto n_k^{-1}$", "focal, weights $\\propto n_k^{-0.75}$",
                        "cross-entropy"])]
    metrics = [("Accuracy", "accuracy", True), ("Balanced accuracy (macro sensitivity)", "balanced_accuracy", True),
               ("Macro precision", "precision_macro", True), ("Macro specificity", "specificity_macro", True),
               ("Macro-F1", "f1_macro", False), ("Cohen's κ", "cohen_kappa", False),
               ("Macro ROC-AUC", "roc_auc_macro", False)]
    print("| | " + " | ".join(cols) + " |")
    print("| --- | ---: | ---: | ---: |")
    for name, vals in static:
        print(f"| {name} | " + " | ".join(vals) + " |")
    for name, key, as_pct in metrics:
        vals = [P[c]["test"][key] for c in cols]
        best = max(vals)
        cells = [(f"{100 * v:.1f} %" if as_pct else f"{v:.3f}") for v in vals]
        cells = [f"**{x}**" if v == best else x for x, v in zip(cells, vals)]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("repeat run of MedMamba-SS:", {k: round(P['MedMamba-SS (repeat)']['test'][k], 4) for _, k, _ in metrics})
    print("MedMamba-T reported:", P["MedMamba-T"]["reported"])


def modality():
    ps = A["modality"]["per_seed"]
    print("| Seed | Test HSI | Test RGB | Δ test | Val. HSI | Val. RGB | Δ val. | Pooled HSI | Pooled RGB | Δ pooled |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for seed in ("1", "7", "13", "23", "42"):
        h, r = ps[seed]["hsi"], ps[seed]["rgb"]
        cells = []
        for split in ("test", "val_full", "pooled"):
            if split in h and split in r:
                a, b = h[split]["balanced_accuracy"], r[split]["balanced_accuracy"]
                cells += [f"{a:.4f}", f"{b:.4f}", f"{100 * (a - b):+.2f}"]
            else:
                cells += ["—"] * 3
        print(f"| {seed} | " + " | ".join(cells) + " |")
    s = A["modality"]["summary"]
    for k in ("test_delta", "val_full_delta", "pooled_delta"):
        if k in s:
            d = s[k]
            print(k, f"bal {100*d['bal_mean']:+.2f} ± {100*d['bal_sd']:.2f} ({d['bal_pos']}/5)  "
                     f"F1 {d['f1_mean']:+.4f} ± {d['f1_sd']:.4f} ({d['f1_pos']}/5)")
    print("bootstrap", json.dumps(A["modality"]["bootstrap_bal_delta"]))


CALIB_MODELS = [("MedMamba [@medmamba]", "MedMamba"), ("HybridSN [@hybridsn]", "hybridsn"),
                ("SpectralFormer [@spectralformer]", "spectralformer"),
                ("Logistic regression, band means + s.d. (64 feat.)", "probe_hsi_mean_std"),
                ("Logistic regression, RGB means + s.d. (6 feat.)", "probe_rgb_mean_std")]


def calibration():
    """MedMamba-SS-TRM as mean ± s.d. over five seeds, every other model at seed 42.
    Last column: ECE after a temperature fitted on the other held-out set."""
    s, ps, c = A["modality"]["summary"], A["modality"]["per_seed"], A["comparison"]
    print("| Held-out set | Model | ECE (29) | MCE | Brier | NLL | Macro ROC-AUC | ECE, temperature fitted on the other set |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for split, lab, tkey in (("test", "Test", "test_ece"), ("val_full", "Validation", "val_ece")):
        for mod, name in (("hsi", "**MedMamba-SS-TRM, 32 bands**"), ("rgb", "MedMamba-SS-TRM, 3 bands")):
            v = s[f"{split}_{mod}"]
            ts = [ps[k][mod]["temperature"][tkey] for k in ps]
            ms = lambda k, d=3: f"{v[k][0]:.{d}f} ± {v[k][1]:.{d}f}"  # noqa: E731
            print(f"| {lab} | {name} | {ms('ece')} | {ms('mce_floor')} | {ms('brier_score')} | {ms('nll')} | "
                  f"{ms('roc_auc_macro', 4)} | {np.mean(ts):.3f} ± {np.std(ts, ddof=1):.3f} |")
        for name, key in CALIB_MODELS:
            v = c[key].get(split)
            if not v or "ece" not in v:
                continue
            t = (c[key].get("temperature") or {}).get(tkey)
            print(f"| {lab} | {name} | {v['ece']:.3f} | {v['mce_floor']:.3f} | {v['brier_score']:.3f} | {v['nll']:.3f} | "
                  f"{v['roc_auc_macro']:.4f} | {f(t, 3)} |")


def probes_by_modality():
    c = A["comparison"]
    for mod in ("hsi", "rgb"):
        for kind in ("mean", "mean_std", "flat"):
            k = f"probe_{mod}_{kind}"
            if k in c:
                print(k, "test", {m: round(c[k]["test_simple"][m], 4) for m in ("balanced_accuracy", "f1_macro")},
                      "val", {m: round(c[k]["val_simple"][m], 4) for m in ("balanced_accuracy", "f1_macro")})


if __name__ == "__main__":
    for fn in (comparison, ablations, pad, modality, calibration, probes_by_modality):
        print(f"\n## {fn.__name__}")
        try:
            fn()
        except Exception as e:
            print(f"[incomplete] {type(e).__name__}: {e}")
