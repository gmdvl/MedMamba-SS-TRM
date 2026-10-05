#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/make_figures_v19.py
===========================
Every result figure of the v11 manuscript, from run artefacts and
paper/source/v19_analysis.json (scripts/analysis_v19.py). Writes PNG (300 dpi)
and PDF to paper/figures/results/. Each figure is skipped, with a message, if its
inputs are not there yet.

Colour roles (validated with the dataviz palette validator, light mode):
    modality   HSI #2a78d6 (blue), RGB #eb6834 (orange)       all-pairs PASS
    class      healthy #1baf7a, DCIS #4a3aa7, IDC #eda100    all-pairs PASS,
               two below 3:1 contrast -> always legend + direct labels
No dual axes; two measures of different scale get two panels.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/results"
DATA = ROOT / "data/hsi_v8-80_10_10_importance-new"
EXP = ROOT / "experiments"
AN = ROOT / "paper/source/v19_analysis.json"

HSI, RGB = "#2a78d6", "#eb6834"
CLS = ["#1baf7a", "#4a3aa7", "#eda100"]
CLS_NAMES = ["healthy", "DCIS", "IDC"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GRAY = "#8a8985"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
W1, W2 = 3.5, 7.16      # IEEE single / double column, inches


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"[fig] {name}")


def run(prefix):
    hits = sorted(glob.glob(str(EXP / f"{prefix}_*")))
    return Path(hits[0]) if hits else None


REF_HSI, REF_RGB = run("20260915_031356"), run("20260915_031947")
WL = np.load(DATA / "hsi/wavelengths.npy")


def composite(cube_chw, bands=(17, 15, 10), lo=None, hi=None):
    img = np.stack([cube_chw[b] for b in bands], axis=-1).astype(np.float32)
    lo = np.percentile(img, 2) if lo is None else lo
    hi = np.percentile(img, 98) if hi is None else hi
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), lo, hi


def shade_gap(ax):
    ax.axvspan(WL[17], WL[18], color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text((WL[17] + WL[18]) / 2, ax.get_ylim()[1], "219 nm\ngap", ha="center", va="top",
            fontsize=6.5, color=INK2)


# ---------------------------------------------------------------------------
def _gap_nan(v):
    """Insert NaN between band 18 and 19 so no line is drawn across the 219 nm gap."""
    return np.insert(np.asarray(v, dtype=float), 18, np.nan, axis=-1)


WLG = _gap_nan(WL)


def fig_dataset():
    X = np.load(DATA / "hsi/X_train.npy", mmap_mode="r")
    y = np.load(DATA / "hsi/y_train.npy")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(y), 30000, replace=False))
    spec = np.asarray(X[idx], dtype=np.float32).mean(axis=(1, 2))        # [N, 32]
    lab = y[idx]
    fig = plt.figure(figsize=(W2, 2.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[3.3, 0.35, 1, 1, 1], wspace=0.1, hspace=0.1)
    ax = fig.add_subplot(gs[:, 0])
    for c in range(3):
        s = spec[lab == c]
        m, sd = _gap_nan(s.mean(0)), _gap_nan(s.std(0))
        ax.fill_between(WLG, m - sd, m + sd, color=CLS[c], alpha=0.14, lw=0)
        ax.plot(WLG, m, color=CLS[c], marker="o", ms=2.5, label=f"{CLS_NAMES[c]} (n={int((lab == c).sum()):,})")
    ax.set_xlabel("Band centre (nm)")
    ax.set_ylabel("Relative reflectance")
    ax.set_title("(a) Class-mean spectra (±1 s.d.)", loc="left")
    shade_gap(ax)
    ax.legend(loc="lower right")
    Xt = np.load(DATA / "hsi/X_test.npy", mmap_mode="r")
    Rt = np.load(DATA / "rgb/X_test.npy", mmap_mode="r")
    yt = np.load(DATA / "hsi/y_test.npy")
    picks = [int(np.flatnonzero(yt == c)[len(np.flatnonzero(yt == c)) // 3]) for c in range(3)]
    cubes = [np.asarray(Xt[i], dtype=np.float32).transpose(2, 0, 1) for i in picks]
    allc = np.concatenate([np.stack([cb[b] for b in (17, 15, 10)]).ravel() for cb in cubes])
    lo, hi = np.percentile(allc, 2), np.percentile(allc, 98)
    for c in range(3):
        a1 = fig.add_subplot(gs[0, 2 + c])
        a1.imshow(composite(cubes[c], lo=lo, hi=hi)[0], interpolation="nearest")
        a2 = fig.add_subplot(gs[1, 2 + c])
        a2.imshow(np.clip(np.asarray(Rt[picks[c]], dtype=np.float32) / 255.0, 0, 1), interpolation="nearest")
        for a in (a1, a2):
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color(CLS[c]); sp.set_linewidth(1.5)
        a1.set_title(CLS_NAMES[c], color=INK)
        if c == 0:
            a1.set_ylabel("HSI composite\n(633/562/463 nm)", fontsize=6.5)
            a2.set_ylabel("paired\nsynthetic RGB", fontsize=6.5)
    fig.text(0.575, 0.99, "(b) One 11×11 test patch per class, both builds", fontsize=8.5, ha="left",
             va="bottom")
    save(fig, "fig_dataset")


def fig_learning_curves():
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0))
    for rd, col, name in ((REF_HSI, HSI, "HSI"), (REF_RGB, RGB, "RGB")):
        h = json.load(open(rd / "history.json"))
        ep = [e["epoch"] for e in h]
        best = json.load(open(rd / "test_report.json"))["best_epoch"]
        axes[0].plot(ep, [e["classification_loss"] for e in h], color=col, label=f"{name} train")
        axes[0].plot(ep, [e["val_cls_loss"] for e in h], color=col, ls="--", label=f"{name} validation")
        axes[1].plot(ep, [e["f1_macro"] for e in h], color=col, label=name)
        axes[1].plot([best], [h[best - 1]["f1_macro"]], marker="o", color=col, mec="white", mew=1.0, ms=6)
        axes[2].plot(ep, [e["SAM_loss"] for e in h], color=col, label=f"{name} train")
        axes[2].plot(ep, [e["val_sam_loss"] for e in h], color=col, ls="--", label=f"{name} validation")
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].set_title("(a) Classification loss", loc="left")
    axes[1].set_title("(b) Validation macro-F1", loc="left")
    axes[2].set_title("(c) Spectral-angle loss", loc="left")
    for a in axes:
        a.set_xlabel("Epoch")
        a.set_xticks([1, 5, 10, 15, 20])
    axes[1].legend(loc="lower right")
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[Line2D([], [], color=INK2, label="train"),
                            Line2D([], [], color=INK2, ls="--", label="validation")],
                   loc="upper right", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_learning_curves")


def reliability(ax, probs, labels, color, name, n_bins=15):
    conf = probs.max(1)
    corr = (probs.argmax(1) == labels)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 200:
            xs.append(conf[m].mean()); ys.append(corr[m].mean())
    ax.plot(xs, ys, marker="o", color=color, label=name)


def fig_error_analysis():
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for k, (rd, name) in enumerate(((REF_HSI, "32-band HSI"), (REF_RGB, "3-band RGB"))):
        d = np.load(rd / "test_predictions.npz")
        cm = confusion_matrix(d["true_label"], d["predicted_label"], labels=[0, 1, 2])
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax = axes[k]
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{pct[i, j]:.1f}%\n{cm[i, j]:,}", ha="center", va="center", fontsize=6.3,
                        color="white" if pct[i, j] > 60 else INK)
        ax.set_xticks(range(3), CLS_NAMES)
        ax.set_yticks(range(3), CLS_NAMES)
        ax.set_xlabel("Predicted")
        if k == 0:
            ax.set_ylabel("True")
        ax.set_title(f"({'ab'[k]}) {name}, test, seed 42", loc="left")
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
    for rd, col, name in ((REF_HSI, HSI, "HSI"), (REF_RGB, RGB, "RGB")):
        d = np.load(rd / "test_predictions.npz")
        ece = json.load(open(rd / "test_report.json"))["calibration"]["ece"]
        reliability(ax, d["probs"], d["true_label"], col, f"{name} (ECE {ece:.3f})")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("(c) Reliability, test, seed 42", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_error_analysis")


def fig_modality(an):
    s = an["modality"]["summary"]
    if "val_full_delta" not in s:
        print("[skip] fig_modality: full-validation predictions not available yet"); return
    seeds = [1, 7, 13, 23, 42]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    ps = an["modality"]["per_seed"]
    for k, (split, lab, mk) in enumerate((("test", "test (5 patients)", "o"),
                                          ("val_full", "validation (5 patients)", "s"),
                                          ("pooled", "pooled (10 patients)", "D"))):
        d = [100 * (ps[str(sd)]["hsi"][split]["balanced_accuracy"] - ps[str(sd)]["rgb"][split]["balanced_accuracy"])
             for sd in seeds]
        ax.plot(np.arange(5) + (k - 1) * 0.2, d, ls="none", marker=mk, ms=5 if mk != "D" else 4.5,
                color=(INK, GRAY, INK)[k], mfc=(INK, GRAY, "white")[k], mew=1.2, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(5), [f"seed {x}" for x in seeds])
    ax.set_ylabel("HSI − RGB, balanced acc. (pts)")
    ax.set_title("(a) Modality difference per seed", loc="left")
    ax.legend(loc="lower left", fontsize=6.3)
    ax = axes[1]
    pp = an["modality"]["per_patient_over_seeds"]
    order = sorted(pp, key=lambda p: (pp[p]["split"] != "test", np.mean(pp[p]["hsi"])))
    for i, pid in enumerate(order):
        h, r = np.mean(pp[pid]["hsi"]), np.mean(pp[pid]["rgb"])
        ax.plot([r, h], [i, i], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        ax.plot(h, i, "o", color=HSI, zorder=2, label="HSI" if i == 0 else None)
        ax.plot(r, i, "o", color=RGB, zorder=2, label="RGB" if i == 0 else None)
    ax.set_yticks(range(len(order)), [f"{p} ({pp[p]['split']}, {pp[p]['n']:,})" for p in order], fontsize=6.3)
    ax.axhline(4.5, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("Macro recall, mean over five seeds")
    ax.set_title("(b) Every held-out patient", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_modality")


def fig_bands(an):
    zs = an["band_zero_shot"]
    ab = an["ablations"]
    xs = [32, 16, 8, 4, 2]
    re = {32: ab["baseline"], 16: ab["bands16"], 8: ab["bands8"]}
    fig, axes = plt.subplots(1, 2, figsize=(W1 * 1.55, 1.9))
    for ax, key, lab in ((axes[0], "bal", "Balanced accuracy"), (axes[1], "f1", "Macro-F1")):
        ax.plot(xs, [zs[str(x)][key] for x in xs], color=GRAY, ls="--", marker="s", label="zero-shot (trained at 32)")
        ax.plot([32, 16, 8], [re[x][key] for x in (32, 16, 8)], color=HSI, marker="o", label="retrained at C")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel("Bands C")
        ax.set_ylim(0, 1)
        ax.set_title(f"({'ab'[ax is axes[1]]}) {lab}", loc="left")
    axes[0].legend(loc="center left", fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_bands")


# The depth-sweep runs were counted with the reconstruction decoder attached
# (6.234543 GF at K=63, experiments/*_ref20/flops_report.json); the preset count
# without it is 6.203319 GF (findings/flops_architectures_hsi.json). The decoder
# reads the final answer state only, so its cost is the same at every K.
DECODER_GFLOPS = 6.234543 - 6.203319


def fig_depth_cost(an, decoder_free=False, out_name="fig_depth_cost"):
    dep = an["depth"]
    ks = sorted(int(k) for k in dep)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 1.95))
    ax = axes[0]
    ax.plot(ks, [dep[str(k)]["bal"] for k in ks], color=HSI, marker="o", label="balanced accuracy")
    ax.plot(ks, [dep[str(k)]["f1"] for k in ks], color=HSI, marker="s", ls="--", label="macro-F1")
    ax.set_ylim(0.85, 0.96)
    ax.set_xticks(ks); ax.set_xlabel("Core applications K")
    ax.set_title("(a) Quality vs depth", loc="left")
    ax.legend(loc="lower right")
    ax = axes[1]
    dec = DECODER_GFLOPS if decoder_free else 0.0
    g = [dep[str(k)]["gflops"] - dec for k in ks]
    kk = np.linspace(0, 90, 50)
    ax.plot(kk, 0.204 - dec + kk * 2 * 121 * 2 * (12 * 128 ** 2 + 9 * 128) / 1e9, color=GRAY, lw=1,
            label="eq. (23)")
    ax.plot(ks, g, "o", color=HSI, label="measured")
    ax.set_xticks(ks); ax.set_xlabel("Core applications K"); ax.set_ylabel("GFLOPs per patch")
    ax.set_title("(b) Arithmetic vs depth", loc="left")
    ax.legend(loc="upper left", fontsize=6.3)
    ax = axes[2]
    pts = [("MedMamba-SS", 2.773007, 0.278, (-7, -3), "right"),
           ("full-channel", 3.513451, 0.305, (0, 7), "center"),
           ("MedMamba-SS-TRM", 0.446409, 6.203, (5, -3), "left")]
    for name, p_, f_, off, ha in pts:
        ours = "TRM" in name
        ax.plot(p_, f_, "o", color=HSI if ours else GRAY, ms=6)
        ax.annotate(name, (p_, f_), xytext=off, textcoords="offset points", ha=ha,
                    fontsize=6.3, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.2, 12); ax.set_ylim(0.08, 20)
    ax.set_xlabel("Parameters (M)"); ax.set_ylabel("GFLOPs per patch")
    ax.set_title("(c) Storage vs arithmetic", loc="left")
    fig.tight_layout(w_pad=1.0)
    save(fig, out_name)


def fig_xai_bands():
    p = REF_HSI / "heldout_v19/xai_test.npz"
    if not p.is_file() or np.load(p)["band_weights"].shape[0] < 1000:
        print("[skip] fig_xai_bands: xai export not available yet"); return
    d = np.load(p)
    bw, lab = d["band_weights"], d["labels"]
    X = np.load(DATA / "hsi/X_test.npy", mmap_mode="r")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1))
    ax = axes[0]
    for c in range(3):
        m, sd = _gap_nan(bw[lab == c].mean(0)), _gap_nan(bw[lab == c].std(0))
        ax.fill_between(WLG, m - sd, m + sd, color=CLS[c], alpha=0.15, lw=0)
        ax.plot(WLG, m, color=CLS[c], marker="o", ms=2.5, label=CLS_NAMES[c])
    ax.set_xlabel("Band centre (nm)"); ax.set_ylabel("Band-gate weight $a_c$")
    ax.set_title("(a) Learned band gate (±1 s.d.)", loc="left")
    shade_gap(ax)
    ax.legend(loc="upper right")
    ax = axes[1]
    spec = np.asarray(X[np.sort(d["test_index"])], dtype=np.float32).mean(axis=(1, 2))
    lab_s = np.load(DATA / "hsi/y_test.npy")[np.sort(d["test_index"])]
    grand = spec.mean(0)
    for c in range(3):
        ax.plot(WLG, _gap_nan(spec[lab_s == c].mean(0) - grand), color=CLS[c], marker="o", ms=2.5,
                label=CLS_NAMES[c])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Band centre (nm)"); ax.set_ylabel("Reflectance difference")
    ax.set_title("(b) Class mean minus overall mean", loc="left")
    shade_gap(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_xai_bands")


def fig_tsne():
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9))
    for ax, rd, name in ((axes[0], REF_HSI, "(a) 32-band HSI"), (axes[1], REF_RGB, "(b) 3-band RGB")):
        p = rd / "heldout_v19/xai_test.npz"
        if not p.is_file():
            print("[skip] fig_tsne: xai export missing"); plt.close(fig); return
        d = np.load(p)
        z = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(d["pooled"])
        for c in range(3):
            m = d["labels"] == c
            ax.scatter(z[m, 0], z[m, 1], s=1.2, color=CLS[c], alpha=0.55, lw=0, label=CLS_NAMES[c], rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{name}", loc="left")
    leg = axes[0].legend(loc="lower left", markerscale=6)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_tsne")


def fig_reconstruction():
    d = np.load(REF_HSI / "reconstruction/samples.npz", allow_pickle=True)
    met = {m["tag"]: m for m in json.load(open(REF_HSI / "reconstruction/samples_metrics.json"))}
    tags = list(d["tag"])
    rows = [(c, q) for c in (0, 1, 2) for q in ("q0.50", "q1.00")]
    allc = np.concatenate([np.stack([x[b] for b in (17, 15, 10)]).ravel() for x in d["x_true"]])
    lo, hi = np.percentile(allc, 2), np.percentile(allc, 98)
    fig = plt.figure(figsize=(W2, 6.0))
    gs = fig.add_gridspec(len(rows) + 1, 5, width_ratios=[1, 1, 1, 0.35, 2.6],
                          height_ratios=[1] * len(rows) + [0.08], wspace=0.12, hspace=0.35)
    sam_im = None
    for i, (c, q) in enumerate(rows):
        tag = f"class{c}_{q}"
        k = tags.index(tag)
        xt, xr = d["x_true"][k], d["x_recon"][k]
        num = (xt * xr).sum(0)
        den = np.linalg.norm(xt, axis=0) * np.linalg.norm(xr, axis=0) + 1e-8
        sam = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
        panels = ((composite(xt, lo=lo, hi=hi)[0], None, "input"),
                  (composite(xr, lo=lo, hi=hi)[0], None, "reconstruction"),
                  (sam, "Blues", "per-pixel angle"))
        for j, (img, cmap, ttl) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(img, cmap=cmap, vmin=(0 if cmap else None), vmax=(20 if cmap else None),
                           interpolation="nearest")
            if cmap:
                sam_im = im
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(ttl, fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{CLS_NAMES[c]}, {'median' if q == 'q0.50' else 'worst'}", fontsize=7, color=INK)
        ax = fig.add_subplot(gs[i, 4])
        mt, mr = xt.mean((1, 2)), xr.mean((1, 2))
        ax.plot(WLG, _gap_nan(mt), color=INK, lw=1.3, label="input")
        ax.plot(WLG, _gap_nan(mr), color=HSI, lw=1.3, ls="--", label="reconstruction")
        top = max(np.nanmax(mt), np.nanmax(mr))
        bot = min(np.nanmin(mt), np.nanmin(mr))
        ax.set_ylim(bot - 0.05 * (top - bot), top + 0.45 * (top - bot))
        m = met[tag]
        ax.text(0.01, 0.97, f"SAM {m['sam_deg']:.1f}°   PSNR {m['psnr']:.1f} dB   2-D SSIM {m['ssim2d']:.2f}",
                transform=ax.transAxes, fontsize=6.3, va="top", color=INK)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.set_title("patch-mean spectrum (reflectance)", fontsize=7)
            ax.legend(loc="lower right", fontsize=6)
        if i == len(rows) - 1:
            ax.set_xlabel("Band centre (nm)", fontsize=7)
        else:
            ax.set_xticklabels([])
    cax = fig.add_subplot(gs[len(rows), 2])
    cb = fig.colorbar(sam_im, cax=cax, orientation="horizontal")
    cb.set_label("spectral angle (°)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig_reconstruction")


def fig_comparison(an):
    comp = an["comparison"]
    ps = an["modality"]["per_seed"]["42"]
    rows = [
        ("LR, RGB band means + s.d. (6 feat.)", comp.get("probe_rgb_mean_std"), False),
        ("LR, HSI band means (32 feat.)", comp.get("probe_hsi_mean"), False),
        ("LR, HSI band means + s.d. (64 feat.)", comp.get("probe_hsi_mean_std"), False),
        ("MedMamba, 3.65 M", comp.get("MedMamba"), False),
        ("SpectralFormer, 0.12 M", comp.get("spectralformer"), False),
        ("HybridSN, 0.57 M", comp.get("hybridsn"), False),
        ("MedMamba-SS-TRM, RGB input, 0.45 M", ps.get("rgb"), True),
        ("MedMamba-SS-TRM, HSI input, 0.45 M", ps.get("hsi"), True),
    ]
    rows = [r for r in rows if r[1] and r[1].get("test")]
    fig, ax = plt.subplots(figsize=(W1 * 1.45, 0.3 * len(rows) + 0.8))
    for i, (lab, r, ours) in enumerate(rows):
        col = HSI if ours else INK2
        t = r["test"]["balanced_accuracy"]
        v = (r.get("val_full") or {}).get("balanced_accuracy")
        pz = (r.get("pooled") or {}).get("balanced_accuracy")
        xs = [x for x in (t, v, pz) if x is not None]
        ax.plot([min(xs), max(xs)], [i, i], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        ax.plot(t, i, "o", color=col, ms=5, zorder=3, label="test (5 patients)" if i == 0 else None)
        if v is not None:
            ax.plot(v, i, "o", mfc="white", mec=col, mew=1.3, ms=5, zorder=3,
                    label="validation (5 patients)" if i == 0 else None)
        if pz is not None:
            ax.plot(pz, i, "D", color=col, ms=4, zorder=4, mec="white", mew=0.6,
                    label="pooled (10 patients)" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.5)
    ax.set_xlabel("Balanced accuracy (seed 42)")
    ax.set_xlim(0.55, 0.97)
    ax.legend(loc="lower left", fontsize=6.3, bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "fig_comparison")


def main(which=None):
    an = json.load(open(AN)) if AN.is_file() else None
    figs = {
        "dataset": fig_dataset, "learning": fig_learning_curves, "error": fig_error_analysis,
        "modality": lambda: fig_modality(an), "bands": lambda: fig_bands(an),
        "depth": lambda: fig_depth_cost(an),
        "depth14": lambda: fig_depth_cost(an, decoder_free=True, out_name="fig_depth_cost_v14"), "xai": fig_xai_bands, "tsne": fig_tsne,
        "recon": fig_reconstruction, "comparison": lambda: fig_comparison(an),
    }
    for name, fn in figs.items():
        if which and name not in which:
            continue
        try:
            fn()
        except Exception as e:  # keep going; report
            print(f"[error] {name}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
