#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig. 9 of manuscript v18: band-count behaviour with THREE conditions - zero-shot with
the encoding as defined (eta = C), zero-shot with eta fixed at the training value
(scripts/eval_band_decimation_fixed_eta.py), and retrained. Style from
make_figures_v19.fig_bands; data from v19_analysis.json and v18_short_analysis.json.
Writes paper/figures/results/fig_bands_v18.{png,pdf}; the v19 figure is left untouched."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.make_figures_v19 as mf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
an = json.load(open(ROOT / "paper/source/v19_analysis.json"))
sh = json.load(open(ROOT / "paper/source/v18_short_analysis.json"))
zs, ab, fx = an["band_zero_shot"], an["ablations"], sh["zero_shot"]["fixed_eta"]
xs = [32, 16, 8, 4, 2]
xf = sorted((int(k) for k in fx), reverse=True)
re_ = {32: ab["baseline"], 16: ab["bands16"], 8: ab["bands8"]}
plt = mf.plt
fig, axes = plt.subplots(1, 2, figsize=(mf.W1 * 1.55, 1.9))
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
axes[0].legend(loc="lower left", fontsize=6)
fig.tight_layout(w_pad=1.0)
mf.save(fig, "fig_bands_v18")
print("written fig_bands_v18")
