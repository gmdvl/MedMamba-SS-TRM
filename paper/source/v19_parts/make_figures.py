#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Result figures of manuscript v19 (the short, hyperspectral-only rewrite of v18).

    python paper/source/v19_parts/make_figures.py [dataset learning bands depth error]

v19 drops every RGB result, so the v18 figures that showed an RGB arm are re-drawn with
the hyperspectral model only. Plotting style, colours and data sources are those of
scripts/make_figures_v19.py and scripts/make_fig_bands_v18.py (imported, not edited);
only panels change. Writes paper/v19/figures/*.{png,pdf}. The band-gate and
reconstruction figures carry no RGB content and are used unchanged from
paper/figures/results/.

Memory: the class-mean spectra are accumulated over the memmapped training patches in
chunks of 1,000 (about 15 MB each), not loaded at once.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import scripts.make_figures_v19 as mf  # noqa: E402

mf.OUT = ROOT / "paper/v19/figures"
plt = mf.plt
W1, W2 = mf.W1, mf.W2          # IEEE single / double column, inches


def fig_dataset():
    """(a) class-mean spectra of 30,000 training patches (same draw as v18), (b) one test patch per class."""
    X = np.load(mf.DATA / "hsi/X_train.npy", mmap_mode="r")
    y = np.load(mf.DATA / "hsi/y_train.npy")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(y), 30000, replace=False))
    spec = np.empty((len(idx), X.shape[-1]), dtype=np.float64)
    for s in range(0, len(idx), 1000):
        spec[s:s + 1000] = np.asarray(X[idx[s:s + 1000]], dtype=np.float64).mean(axis=(1, 2))
    lab = y[idx]
    fig = plt.figure(figsize=(W1, 3.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.0], hspace=0.85, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    for c in range(3):
        s = spec[lab == c]
        m, sd = mf._gap_nan(s.mean(0)), mf._gap_nan(s.std(0))
        ax.fill_between(mf.WLG, m - sd, m + sd, color=mf.CLS[c], alpha=0.14, lw=0)
        ax.plot(mf.WLG, m, color=mf.CLS[c], marker="o", ms=2.0,
                label=f"{mf.CLS_NAMES[c]} (n={int((lab == c).sum()):,})")
    ax.set_xlabel("Band centre (nm)")
    ax.set_ylabel("Relative reflectance")
    ax.set_title("(a) Class-mean spectra (±1 s.d.)", loc="left")
    mf.shade_gap(ax)
    ax.legend(loc="lower right", fontsize=6.3)
    Xt = np.load(mf.DATA / "hsi/X_test.npy", mmap_mode="r")
    yt = np.load(mf.DATA / "hsi/y_test.npy")
    picks = [int(np.flatnonzero(yt == c)[len(np.flatnonzero(yt == c)) // 3]) for c in range(3)]
    cubes = [np.asarray(Xt[i], dtype=np.float32).transpose(2, 0, 1) for i in picks]
    allc = np.concatenate([np.stack([cb[b] for b in (17, 15, 10)]).ravel() for cb in cubes])
    lo, hi = np.percentile(allc, 2), np.percentile(allc, 98)
    for c in range(3):
        a = fig.add_subplot(gs[1, c])
        a.imshow(mf.composite(cubes[c], lo=lo, hi=hi)[0], interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(mf.CLS[c]); sp.set_linewidth(1.5)
        a.set_xlabel(mf.CLS_NAMES[c], color=mf.INK, labelpad=2)
        if c == 0:
            a.set_title("(b) One 11×11 test patch per class (633/562/463 nm)", loc="left", x=0.0)
    mf.save(fig, "fig_dataset_hsi")


def fig_learning():
    """Seed-42 hyperspectral run: (a) classification loss, (b) validation macro-F1."""
    h = json.load(open(mf.REF_HSI / "history.json"))
    best = json.load(open(mf.REF_HSI / "test_report.json"))["best_epoch"]
    ep = [e["epoch"] for e in h]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.55))
    ax = axes[0]
    ax.plot(ep, [e["classification_loss"] for e in h], color=mf.HSI, label="training")
    ax.plot(ep, [e["val_cls_loss"] for e in h], color=mf.HSI, ls="--", label="validation")
    ax.set_yscale("log")
    ax.set_title("(a) Classification loss", loc="left")
    ax.annotate("validation", (ep[-1], h[-1]["val_cls_loss"]), xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.3, color=mf.INK)
    ax.annotate("training", (ep[-1], h[-1]["classification_loss"]), xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.3, color=mf.INK)
    ax = axes[1]
    ax.plot(ep, [e["f1_macro"] for e in h], color=mf.HSI)
    ax.plot([best], [h[best - 1]["f1_macro"]], marker="o", color=mf.HSI, mec="white", mew=1.0, ms=6)
    ax.annotate(f"selected (epoch {best})", (best, h[best - 1]["f1_macro"]), xytext=(4, -12),
                textcoords="offset points", fontsize=6.3, color=mf.INK)
    ax.set_title("(b) Validation macro-F1", loc="left")
    for a in axes:
        a.set_xlabel("Epoch")
        a.set_xticks([1, 5, 10, 15, 20])
    fig.tight_layout(w_pad=0.8)
    mf.save(fig, "fig_learning_hsi")


def fig_bands():
    """make_fig_bands_v18 at single-column width, legend below both panels."""
    an = json.load(open(ROOT / "paper/source/v19_analysis.json"))
    sh = json.load(open(ROOT / "paper/source/v18_short_analysis.json"))
    zs, ab, fx = an["band_zero_shot"], an["ablations"], sh["zero_shot"]["fixed_eta"]
    xs = [32, 16, 8, 4, 2]
    xf = sorted((int(k) for k in fx), reverse=True)
    re_ = {32: ab["baseline"], 16: ab["bands16"], 8: ab["bands8"]}
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.95))
    for ax, key, fkey, lab in ((axes[0], "bal", "balanced_accuracy", "Balanced accuracy"),
                               (axes[1], "f1", "f1_macro", "Macro-F1")):
        ax.plot(xs, [zs[str(x)][key] for x in xs], color=mf.GRAY, ls="--", marker="s",
                label="zero-shot, $\\eta = C$")
        ax.plot(xf, [fx[str(x)][fkey] for x in xf], color="#222222", ls="-.", marker="^",
                label="zero-shot, $\\eta$ fixed at 32")
        ax.plot([32, 16, 8], [re_[x][key] for x in (32, 16, 8)], color=mf.HSI, marker="o",
                label="retrained at $C$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel("Bands $C$")
        ax.set_ylim(0, 1)
        ax.set_title(f"({'ab'[ax is axes[1]]}) {lab}", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0.09, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.0, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.2, columnspacing=0.8)
    mf.save(fig, "fig_bands_hsi")


def fig_depth():
    """Recursion depth and cost, decoder-free; the cost-model legend cites no equation number."""
    an = json.load(open(mf.AN))
    dep = an["depth"]
    ks = sorted(int(k) for k in dep)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 1.85))
    ax = axes[0]
    ax.plot(ks, [dep[str(k)]["bal"] for k in ks], color=mf.HSI, marker="o", label="balanced accuracy")
    ax.plot(ks, [dep[str(k)]["f1"] for k in ks], color=mf.HSI, marker="s", ls="--", label="macro-F1")
    ax.set_ylim(0.85, 0.96)
    ax.set_xticks(ks); ax.set_xlabel("Core applications $K$")
    ax.set_title("(a) Quality vs depth", loc="left")
    ax.legend(loc="lower right")
    ax = axes[1]
    dec = mf.DECODER_GFLOPS
    g = [dep[str(k)]["gflops"] - dec for k in ks]
    kk = np.linspace(0, 90, 50)
    ax.plot(kk, 0.204 - dec + kk * 2 * 121 * 2 * (12 * 128 ** 2 + 9 * 128) / 1e9, color=mf.GRAY, lw=1,
            label="cost model")
    ax.plot(ks, g, "o", color=mf.HSI, label="measured")
    ax.set_xticks(ks); ax.set_xlabel("Core applications $K$"); ax.set_ylabel("GFLOPs per patch")
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
    fig.tight_layout(w_pad=1.0)
    mf.save(fig, "fig_recursion_cost_hsi")


def fig_error():
    """Seed 42: (a) test confusion matrix, (b) reliability on the test and the validation patients."""
    from sklearn.metrics import confusion_matrix
    test = np.load(mf.REF_HSI / "test_predictions.npz")
    val = np.load(mf.REF_HSI / "heldout_v19/validation/test_predictions.npz")
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.85), gridspec_kw={"width_ratios": [1, 1.05]})
    cm = confusion_matrix(test["true_label"], test["predicted_label"], labels=[0, 1, 2])
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax = axes[0]
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pct[i, j]:.1f}%\n{cm[i, j]:,}", ha="center", va="center", fontsize=5.4,
                    color="white" if pct[i, j] > 60 else mf.INK)
    ax.set_xticks(range(3), mf.CLS_NAMES, fontsize=6.3)
    ax.set_yticks(range(3), mf.CLS_NAMES, fontsize=6.3)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("(a) Test confusion", loc="left")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=mf.GRAY, lw=1, ls="--")
    for d, ls, name in ((test, "-", "test"), (val, "--", "validation")):
        mf.reliability(ax, d["probs"], d["true_label"], mf.HSI, name)
        ax.get_lines()[-1].set_linestyle(ls)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("(b) Reliability", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.03, -0.03), fontsize=6.0, handlelength=2.4, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    mf.save(fig, "fig_error_hsi")


FIGS = {"dataset": fig_dataset, "learning": fig_learning, "bands": fig_bands,
        "depth": fig_depth, "error": fig_error}


def main(which):
    for name, fn in FIGS.items():
        if which and name not in which:
            continue
        try:
            fn()
        except Exception as e:  # keep going; report
            print(f"[error] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
