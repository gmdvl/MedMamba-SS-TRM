# -*- coding: utf-8 -*-
"""
hsi_augment/provenance_figures.py
=================================
The figures of the `index` stage (issue #15, Plan §Phase 4 item 2), drawn from
`imbalance_report.json` alone. Kept apart from `provenance.py` so the numeric
library (which #16, #28 and #30 import for `load_index` and `patch_feats`)
never imports matplotlib.

    class_balance.png          left: each class's share of the rows, per split
                               (train, train tissue rows, val, test), with the
                               row count on each bar; right: patients per class
                               per split. Shares, not counts, so splits of very
                               different sizes are comparable on one axis.
    patient_contributions.png  one panel per class: every train patient with
                               rows of that class, by rows descending; the pale
                               bar is all rows, the solid bar the tissue rows.
                               Five DCIS bars against ~30 per other class is the
                               bottleneck the report is about.

The style (categorical colours in class order, text tokens, recessive grid) is
the one `tissue_figures.py` defines; it is imported, not copied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Tuple

import numpy as np
from matplotlib.ticker import MaxNLocator

from hsi_augment.tissue_figures import (MUTED, SERIES, TEXT, TEXT_2, _save,  # noqa: F401 (sets Agg)
                                        _style, plt)

ALL_ALPHA = 0.35
MIN_PANEL_BARS = 8


def _compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 10_000:
        return f"{n / 1e3:.0f}k"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(int(n))


def _groups(report: Mapping[str, Any]) -> List[Tuple[str, Mapping[str, Any]]]:
    """(label, table) per bar group: train, train tissue rows, then val, test."""
    splits = report["splits"]
    out: List[Tuple[str, Mapping[str, Any]]] = [("train", splits["train"])]
    tissue = report.get("tissue") or {}
    if tissue.get("train") is not None:
        out.append(("train\ntissue rows", tissue["train"]))
    out += [(s, splits[s]) for s in ("val", "test") if s in splits]
    return out


def plot_class_balance(path: Path, report: Mapping[str, Any]) -> Path:
    names = list(report["class_names"])
    groups = _groups(report)
    C = len(names)
    width = 0.8 / C
    x = np.arange(len(groups))
    fig, (a_share, a_pat) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.4, 1.0]})
    for k, name in enumerate(names):
        color = SERIES[k % len(SERIES)]
        shares = [100.0 * (t["share_per_class"][name] or 0.0) for _, t in groups]
        counts = [int(t["rows_per_class"][name]) for _, t in groups]
        pats = [int(t["patients_per_class"][name]) for _, t in groups]
        off = (k - (C - 1) / 2) * width
        a_share.bar(x + off, shares, width=width * 0.92, color=color, label=name)
        a_pat.bar(x + off, pats, width=width * 0.92, color=color, label=name)
        for xi, s, n in zip(x + off, shares, counts):
            a_share.text(xi, s, _compact(n), ha="center", va="bottom", fontsize=6.5, color=TEXT_2)
        for xi, p in zip(x + off, pats):
            a_pat.text(xi, p, str(p), ha="center", va="bottom", fontsize=7, color=TEXT_2)
    for ax, ylab, title in ((a_share, "share of rows (%)", "Rows per class (bar label: row count)"),
                            (a_pat, "patients", "Patients per class")):
        _style(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([g for g, _ in groups], fontsize=8, color=TEXT)
        ax.set_ylabel(ylab, color=TEXT, fontsize=9)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.margins(y=0.12)
    a_pat.yaxis.set_major_locator(MaxNLocator(integer=True))
    a_share.legend(fontsize=8, frameon=False, labelcolor=TEXT_2, loc="upper left")
    tr = report["splits"]["train"]["imbalance"]
    if tr.get("max_over_min"):
        fig.suptitle(f"Class balance - train {tr['label']} = {tr['max_over_min']:.1f} in rows; "
                     f"the minority's patients are the bottleneck", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_patient_contributions(path: Path, report: Mapping[str, Any]) -> Path:
    names = list(report["class_names"])
    train = report["splits"]["train"]
    tissue_train = (report.get("tissue") or {}).get("train")
    tissue_rows = ({e["patient"]: e["rows_per_class"] for e in tissue_train["per_patient"]}
                   if tissue_train is not None else {})
    rows = {e["patient"]: e["rows_per_class"] for e in train["per_patient"]}
    # panel width ~ its number of bars, with a floor so a 5-patient class keeps a readable title
    ratios = [max(MIN_PANEL_BARS, len(train["patient_concentration"][n]["patients"])) for n in names]
    fig, axes = plt.subplots(1, len(names), figsize=(max(9.0, 1.5 + 0.24 * sum(ratios)), 4.0), squeeze=False,
                             gridspec_kw={"width_ratios": ratios})
    for k, (name, ax) in enumerate(zip(names, axes[0])):
        color = SERIES[k % len(SERIES)]
        conc = train["patient_concentration"][name]
        order = conc["patients"]
        allv = np.array([rows[p][name] for p in order], dtype=np.float64)
        tisv = np.array([tissue_rows.get(p, {}).get(name, 0) for p in order], dtype=np.float64)
        xs = np.arange(len(order))
        _style(ax)
        ax.bar(xs, allv, width=0.8, color=color, alpha=ALL_ALPHA, label="all rows")
        if tissue_train is not None:
            ax.bar(xs, tisv, width=0.5, color=color, label="tissue rows")
        ax.set_xticks(xs)
        ax.set_xlim(-0.6, ratios[k] - 0.4)             # one bar slot per unit: equal bar widths in every panel
        ax.set_xticklabels([str(p) for p in order], rotation=90, fontsize=6.5, color=TEXT_2)
        ax.set_xlabel("train patient", color=TEXT, fontsize=8)
        eff = conc["effective_patients"]
        ax.set_title(f"{name}: {conc['n_patients']} patients, {train['rows_per_class'][name]:,} rows"
                     + (f"\neffective patients {eff:.1f}" if eff else ""), fontsize=9, color=TEXT, loc="left")
        if k == 0:
            ax.set_ylabel("rows (patches)", color=TEXT, fontsize=9)
            ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_2)
        if not order:
            ax.text(0.5, 0.5, "no rows", transform=ax.transAxes, ha="center", color=MUTED)
    fig.tight_layout()
    return _save(fig, path)
