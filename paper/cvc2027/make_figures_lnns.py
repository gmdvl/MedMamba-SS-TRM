#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-render the v18 result figures at Springer LNNS text width (122 mm = 4.8 in) for the
CVC 2027 submission. The figures were drawn for IEEE double-column width (7.16 in) with
7-8 pt text; placed at LNNS width they would shrink to about 4.7 pt. Plotting code, data
and style are those of scripts/make_figures_v19.py and scripts/make_fig_bands_v18.py;
only the widths change, and the cost-model legend no longer cites an equation number
(the LNNS paper renumbers the equations). Writes paper/cvc2027/figures/*.{png,pdf};
paper/figures/results/ is left untouched.

    python paper/cvc2027/make_figures_lnns.py [dataset error modality bands depth xai tsne recon comparison learning]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.make_figures_v19 as mf  # noqa: E402

LNNS_W = 4.8
mf.W2 = LNNS_W
mf.W1 = LNNS_W / 1.55          # the two W1-based figures use W1 * 1.55 and W1 * 1.45
mf.OUT = ROOT / "paper/cvc2027/figures"
plt = mf.plt
# Dense multi-panel figures are drawn wider and scaled down to the text width by LaTeX
# (0.8x at 6.0 in: 7-8 pt text lands at 5.6-6.4 pt); at 4.8 in their panels collide.
WIDTH = {"dataset": 6.0, "error": 6.0, "learning": 6.0, "modality": 6.0, "depth": 6.0,
         "xai": 6.0, "tsne": 6.0, "recon": 6.0, "comparison": LNNS_W, "bands": LNNS_W}


_save = mf.save


def _save_lnns(fig, name):
    """Label-only fixes where panels collide at the narrower width; no data change."""
    axes = fig.get_axes()
    if name == "fig_modality":
        axes[0].set_xticks(range(5), ["1", "7", "13", "23", "42"])
        axes[0].set_xlabel("Seed")
        fig.tight_layout(w_pad=1.2)
    elif name == "fig_error_analysis":     # "test, seed 42" moves to the caption
        axes[0].set_title("(a) 32-band HSI", loc="left")
        axes[1].set_title("(b) 3-band RGB", loc="left")
        axes[2].set_title("(c) Reliability", loc="left")
        fig.tight_layout(w_pad=1.2)
    _save(fig, name)


mf.save = _save_lnns


def fig_depth_cost(an):
    """make_figures_v19.fig_depth_cost(decoder_free=True), legend label 'cost model'."""
    dep = an["depth"]
    ks = sorted(int(k) for k in dep)
    fig, axes = plt.subplots(1, 3, figsize=(mf.W2, 1.95))
    ax = axes[0]
    ax.plot(ks, [dep[str(k)]["bal"] for k in ks], color=mf.HSI, marker="o", label="balanced accuracy")
    ax.plot(ks, [dep[str(k)]["f1"] for k in ks], color=mf.HSI, marker="s", ls="--", label="macro-F1")
    ax.set_ylim(0.85, 0.96)
    ax.set_xticks(ks); ax.set_xlabel("Core applications K")
    ax.set_title("(a) Quality vs depth", loc="left")
    ax.legend(loc="lower right")
    ax = axes[1]
    dec = mf.DECODER_GFLOPS
    g = [dep[str(k)]["gflops"] - dec for k in ks]
    kk = np.linspace(0, 90, 50)
    ax.plot(kk, 0.204 - dec + kk * 2 * 121 * 2 * (12 * 128 ** 2 + 9 * 128) / 1e9, color=mf.GRAY, lw=1,
            label="cost model")
    ax.plot(ks, g, "o", color=mf.HSI, label="measured")
    ax.set_xticks(ks); ax.set_xlabel("Core applications K"); ax.set_ylabel("GFLOPs per patch")
    ax.set_title("(b) Arithmetic vs depth", loc="left")
    ax.legend(loc="upper left", fontsize=6.3)
    ax = axes[2]
    pts = [("MedMamba-SS", 2.773007, 0.278, (-7, -3), "right"),
           ("full-channel", 3.513451, 0.305, (0, 7), "center"),
           ("MedMamba-SS-TRM", 0.446409, 6.203, (5, -3), "left")]
    for name, p_, f_, off, ha in pts:
        ours = "TRM" in name
        ax.plot(p_, f_, "o", color=mf.HSI if ours else mf.GRAY, ms=6)
        ax.annotate(name, (p_, f_), xytext=off, textcoords="offset points", ha=ha,
                    fontsize=6.3, color=mf.INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.2, 12); ax.set_ylim(0.08, 20)
    ax.set_xlabel("Parameters (M)"); ax.set_ylabel("GFLOPs per patch")
    ax.set_title("(c) Storage vs arithmetic", loc="left")
    fig.tight_layout(w_pad=0.6)
    mf.save(fig, "fig_recursion_cost")


def fig_bands():
    """scripts/make_fig_bands_v18.py at LNNS width."""
    an = json.load(open(ROOT / "paper/source/v19_analysis.json"))
    sh = json.load(open(ROOT / "paper/source/v18_short_analysis.json"))
    zs, ab, fx = an["band_zero_shot"], an["ablations"], sh["zero_shot"]["fixed_eta"]
    xs = [32, 16, 8, 4, 2]
    xf = sorted((int(k) for k in fx), reverse=True)
    re_ = {32: ab["baseline"], 16: ab["bands16"], 8: ab["bands8"]}
    fig, axes = plt.subplots(1, 2, figsize=(LNNS_W, 1.9))
    for ax, key, fkey, lab in ((axes[0], "bal", "balanced_accuracy", "Balanced accuracy"),
                               (axes[1], "f1", "f1_macro", "Macro-F1")):
        ax.plot(xs, [zs[str(x)][key] for x in xs], color=mf.GRAY, ls="--", marker="s", label="zero-shot, $\\eta = C$")
        ax.plot(xf, [fx[str(x)][fkey] for x in xf], color="#222222", ls="-.", marker="^", label="zero-shot, $\\eta$ fixed at 32")
        ax.plot([32, 16, 8], [re_[x][key] for x in (32, 16, 8)], color=mf.HSI, marker="o", label="retrained at C")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel("Bands C")
        ax.set_ylim(0, 1)
        ax.set_title(f"({'ab'[ax is axes[1]]}) {lab}", loc="left")
    h, l = axes[0].get_legend_handles_labels()   # one legend below both panels, clear of the curves
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, -0.01))
    mf.save(fig, "fig_bands")


def main(which):
    an = json.load(open(mf.AN))
    figs = {
        "dataset": mf.fig_dataset, "error": mf.fig_error_analysis, "learning": mf.fig_learning_curves,
        "modality": lambda: mf.fig_modality(an), "bands": fig_bands, "depth": lambda: fig_depth_cost(an),
        "xai": mf.fig_xai_bands, "tsne": mf.fig_tsne, "recon": mf.fig_reconstruction,
        "comparison": lambda: mf.fig_comparison(an),
    }
    for name, fn in figs.items():
        if which and name not in which:
            continue
        mf.W2 = WIDTH.get(name, LNNS_W)
        try:
            fn()
        except Exception as e:  # keep going; report
            print(f"[error] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
