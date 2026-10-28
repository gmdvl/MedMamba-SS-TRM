# -*- coding: utf-8 -*-
"""
hsi_augment/stats.py
====================
The pre-registered statistics of Phase 10 (#34; Plan §Phase 10, "Statistics").
Pure functions of numbers and label arrays: they never open a file, so the
report (`report.py`) decides what is read and the test seal stays in one place.

Seed level (the unit the arms are compared on)
----------------------------------------------
* `paired_deltas(arm, ref)`: per-seed arm - A0 on the SAME seed, then the
  mean, sd (ddof 1), a 95% t interval (n - 1 df) and the sign count. Seeds that
  only one side has are listed and left out, never imputed.
* `minimum_effect(...)`: the pre-registered minimum effect, Delta macro-F1 >=
  0.01 with at least 4 of the 5 seeds positive. With fewer than 5 paired
  seeds the verdict is None ("incomplete"), not False.
* `wilcoxon_one_sided(...)`: reported ONLY as p = 0.03125, and only when all 5
  seeds improve. With n = 5 the exact one-sided signed-rank test reaches
  1/32 = 0.03125 at 5/5 and 2/32 = 0.0625 at best otherwise, and a two-sided
  test can never go below 0.0625, so no other p value would mean anything.

Patient level
-------------
`patient_bootstrap(y, preds_a, preds_b, groups, labels=[0, 1, 2])` resamples
PATIENTS with replacement (the unit of independence) and recomputes
metric(a) - metric(b) from per-patient confusion matrices, exactly, pooled over
seeds when given one prediction array per seed (statistic = mean over seeds).

Why not `training.heldout_metrics.patient_bootstrap_delta` (`:99-124`): it
lets sklearn infer the class set from each resample, so a resample without
DCIS is scored as a 2-class macro-F1 next to 3-class ones; and with one DCIS
patient among five test patients, (4/5)^5 = 33% of its resamples have no DCIS
row at all. Here the class set is fixed (`labels`), every resample that lacks
a class is COUNTED (`n_resamples_missing_a_class`, `missing_by_class`), and by
default (`missing="skip"`) it is left out of the interval, so the interval is
conditional on every class being present and says so.

No patient-level CI is claimed for a class held by fewer than
`MIN_PATIENTS_FOR_CLASS_CI` patients (`class_ci_policy`): each held-out split
has ONE DCIS patient, so a "DCIS interval over patients" would describe one
patient's rows, not a population. DCIS metrics are reported per patient.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_LABELS: Tuple[int, ...] = (0, 1, 2)
N_SEEDS = 5
MIN_EFFECT_DELTA = 0.01
MIN_EFFECT_POSITIVE = 4
WILCOXON_ALL_POSITIVE_P = 0.03125           # 2 ** -5: exact one-sided signed-rank, n = 5, 5/5 positive
BOOTSTRAP_METRICS = ("f1_macro", "balanced_accuracy")
MISSING_POLICIES = ("skip", "keep")
MIN_PATIENTS_FOR_CLASS_CI = 2

MIN_EFFECT_RULE = (f"pre-registered minimum effect: mean Delta macro-F1 >= {MIN_EFFECT_DELTA} with at least "
                   f"{MIN_EFFECT_POSITIVE} of {N_SEEDS} seeds positive (paired against A0 on the same seed)")
WILCOXON_RULE = (f"one-sided exact Wilcoxon signed-rank, reported only as p = {WILCOXON_ALL_POSITIVE_P} when "
                 f"all {N_SEEDS} seeds improve (n = {N_SEEDS}: any other sign pattern gives p >= 0.0625; a "
                 f"two-sided test can never go below 0.0625)")

Number = Union[int, float]


# ============================================================================
# seed level
# ============================================================================

def mean_sd(values: Sequence[Number]) -> Dict[str, Any]:
    """{n, mean, sd}; sd uses ddof 1 and is None below two values."""
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None}


def t_ci95(values: Sequence[Number]) -> Optional[List[float]]:
    """Two-sided 95% t interval of the mean (n - 1 df); None below two values."""
    from scipy import stats as st
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    if v.size < 2:
        return None
    half = float(st.t.ppf(0.975, v.size - 1)) * float(v.std(ddof=1)) / math.sqrt(v.size)
    m = float(v.mean())
    return [m - half, m + half]


def paired_deltas(arm: Mapping[int, Optional[Number]], ref: Mapping[int, Optional[Number]], *,
                  higher_is_better: bool = True) -> Dict[str, Any]:
    """Per-seed `arm[s] - ref[s]` over the seeds both have (None = missing).

    `n_positive` / `n_negative` count the sign of the difference;
    `n_improved` counts improvements in the metric's own direction (a lower
    ECE is an improvement), and `sign_count` is "n_improved/n"."""
    a = {int(s): float(v) for s, v in arm.items() if v is not None and np.isfinite(v)}
    r = {int(s): float(v) for s, v in ref.items() if v is not None and np.isfinite(v)}
    seeds = sorted(set(a) & set(r))
    d = [a[s] - r[s] for s in seeds]
    n_pos = sum(x > 0 for x in d)
    n_neg = sum(x < 0 for x in d)
    improved = n_pos if higher_is_better else n_neg
    ms = mean_sd(d)
    return {
        "seeds": seeds,
        "per_seed": {str(s): x for s, x in zip(seeds, d)},
        "n": len(d), "mean": ms["mean"], "sd": ms["sd"], "ci95": t_ci95(d),
        "n_positive": int(n_pos), "n_negative": int(n_neg), "n_zero": int(len(d) - n_pos - n_neg),
        "n_improved": int(improved), "sign_count": f"{improved}/{len(d)}",
        "higher_is_better": bool(higher_is_better),
        "missing_seeds": {"arm": sorted(set(r) - set(a)), "ref": sorted(set(a) - set(r))},
    }


def minimum_effect(deltas: Mapping[str, Any], *, delta: float = MIN_EFFECT_DELTA,
                   min_positive: int = MIN_EFFECT_POSITIVE, n_seeds: int = N_SEEDS) -> Dict[str, Any]:
    """The pre-registered rule on a `paired_deltas` result for macro-F1.
    `met` is True / False, or None when fewer than `n_seeds` seeds are paired."""
    n, mean, pos = int(deltas.get("n", 0)), deltas.get("mean"), int(deltas.get("n_positive", 0))
    out = {"rule": MIN_EFFECT_RULE, "delta_threshold": delta, "min_positive": min_positive,
           "n_seeds_required": n_seeds, "n": n, "mean_delta": mean, "n_positive": pos}
    if n < n_seeds or mean is None:
        return {**out, "met": None, "reason": f"incomplete: {n} of {n_seeds} seeds paired"}
    ok_mean, ok_sign = mean >= delta, pos >= min_positive
    if ok_mean and ok_sign:
        reason = f"mean Delta {mean:+.4f} >= {delta} and {pos}/{n} seeds positive"
    else:
        why = []
        if not ok_mean:
            why.append(f"mean Delta {mean:+.4f} < {delta}")
        if not ok_sign:
            why.append(f"only {pos}/{n} seeds positive (< {min_positive})")
        reason = "; ".join(why)
    return {**out, "met": bool(ok_mean and ok_sign), "reason": reason}


def wilcoxon_one_sided(deltas: Union[Mapping[str, Any], Sequence[Number]], *,
                       n_seeds: int = N_SEEDS) -> Dict[str, Any]:
    """p = 0.03125 when exactly `n_seeds` (5) paired seeds all improve, else
    not reported (p None) with the reason. `deltas` is a `paired_deltas`
    result (its direction is honoured) or a plain list of arm - ref values."""
    if isinstance(deltas, Mapping):
        n, improved = int(deltas.get("n", 0)), int(deltas.get("n_improved", 0))
    else:
        v = [float(x) for x in deltas]
        n, improved = len(v), sum(x > 0 for x in v)
    out = {"rule": WILCOXON_RULE, "n": n, "n_improved": improved}
    if n == n_seeds and improved == n_seeds:
        return {**out, "reported": True, "p": WILCOXON_ALL_POSITIVE_P,
                "note": f"all {n_seeds} seeds improve: exact one-sided p = 1/2^{n_seeds}"}
    return {**out, "reported": False, "p": None,
            "note": f"not reported: {improved}/{n} seeds improve (pre-registered: only at {n_seeds}/{n_seeds})"}


# ============================================================================
# confusion matrices
# ============================================================================

def _label_index(values: np.ndarray, labels: Sequence[int], what: str) -> np.ndarray:
    values = np.asarray(values).astype(np.int64, copy=False).reshape(-1)
    lab = np.asarray(labels, dtype=np.int64)
    lut_off = int(min(lab.min(), values.min() if values.size else 0))
    span = int(max(lab.max(), values.max() if values.size else 0)) - lut_off + 1
    lut = np.full(span, -1, dtype=np.int64)
    lut[lab - lut_off] = np.arange(len(lab))
    idx = lut[values - lut_off]
    if (idx < 0).any():
        bad = sorted(set(values[idx < 0].tolist()))[:5]
        raise ValueError(f"{what} holds values {bad} outside labels {list(labels)}")
    return idx


def confusion(y: np.ndarray, pred: np.ndarray, labels: Sequence[int] = DEFAULT_LABELS) -> np.ndarray:
    """(K, K) int64 counts, rows = true label, columns = predicted, in `labels` order."""
    k = len(labels)
    yi, pi = _label_index(y, labels, "y"), _label_index(pred, labels, "pred")
    if yi.shape != pi.shape:
        raise ValueError(f"y has {yi.size} rows, pred {pi.size}")
    return np.bincount(yi * k + pi, minlength=k * k).reshape(k, k).astype(np.int64)


def patient_confusions(y: np.ndarray, pred: np.ndarray, groups: np.ndarray,
                       labels: Sequence[int] = DEFAULT_LABELS) -> Tuple[np.ndarray, np.ndarray]:
    """(patients (P,) sorted, confusions (P, K, K) int64): one matrix per patient."""
    k = len(labels)
    groups = np.asarray(groups).reshape(-1)
    yi, pi = _label_index(y, labels, "y"), _label_index(pred, labels, "pred")
    if not (len(groups) == len(yi) == len(pi)):
        raise ValueError(f"groups/y/pred lengths differ: {len(groups)}/{len(yi)}/{len(pi)}")
    patients, gi = np.unique(groups, return_inverse=True)
    flat = np.bincount(gi.astype(np.int64) * k * k + yi * k + pi, minlength=len(patients) * k * k)
    return patients, flat.reshape(len(patients), k, k).astype(np.int64)


def _f1_macro(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 over ALL K labels of (..., K, K) confusions; an undefined
    per-class F1 is 0 (sklearn's `labels=..., zero_division=0`)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(np.float64)
    denom = cm.sum(-1) + cm.sum(-2)                     # support + predicted = 2tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(-1)


def _balanced_accuracy(cm: np.ndarray) -> np.ndarray:
    """Mean recall over the labels PRESENT in y (sklearn's balanced_accuracy_score)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(np.float64)
    support = cm.sum(-1).astype(np.float64)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    present = (support > 0).sum(-1)
    return np.divide(rec.sum(-1), present, out=np.full(rec.shape[:-1], np.nan), where=present > 0)


_METRIC_FNS = {"f1_macro": _f1_macro, "balanced_accuracy": _balanced_accuracy}


def metric_from_confusion(cm: np.ndarray, metric: str) -> Union[float, np.ndarray]:
    if metric not in _METRIC_FNS:
        raise ValueError(f"metric {metric!r}; expected one of {BOOTSTRAP_METRICS}")
    v = _METRIC_FNS[metric](np.asarray(cm))
    return float(v) if np.ndim(v) == 0 else v


def per_class_from_confusion(cm: np.ndarray) -> Dict[str, List[float]]:
    """Per-class precision / recall / F1 / support (zero_division 0), in label order."""
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    support, predicted = cm.sum(1).astype(np.float64), cm.sum(0).astype(np.float64)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2.0 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    return {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
            "support": support.astype(np.int64).tolist()}


# ============================================================================
# patient level
# ============================================================================

def patients_with_class(y: np.ndarray, groups: np.ndarray,
                        labels: Sequence[int] = DEFAULT_LABELS) -> Dict[str, List[int]]:
    """{str(label): sorted patient ids holding at least one row of it}."""
    y, groups = np.asarray(y).reshape(-1), np.asarray(groups).reshape(-1)
    return {str(c): sorted(int(g) for g in np.unique(groups[y == c])) for c in labels}


def class_ci_policy(y: np.ndarray, groups: np.ndarray, label: int, *,
                    min_patients: int = MIN_PATIENTS_FOR_CLASS_CI) -> Dict[str, Any]:
    """Whether a patient-level CI may be claimed for one class's metrics."""
    pts = patients_with_class(y, groups, (label,))[str(label)]
    ok = len(pts) >= min_patients
    return {"label": int(label), "patients": pts, "n_patients": len(pts), "claimable": bool(ok),
            "reason": (f"{len(pts)} patient(s) hold this class" if ok else
                       f"no patient-level CI: only {len(pts)} patient(s) {pts} hold this class, so a "
                       f"bootstrap over patients would describe one patient's rows; reported per patient")}


def _as_list(preds: Union[np.ndarray, Sequence[np.ndarray]]) -> List[np.ndarray]:
    if isinstance(preds, np.ndarray) and preds.ndim == 1:
        return [preds]
    return [np.asarray(p) for p in preds]


def patient_bootstrap(y: np.ndarray, preds_a: Union[np.ndarray, Sequence[np.ndarray]],
                      preds_b: Union[np.ndarray, Sequence[np.ndarray]], groups: np.ndarray, *,
                      labels: Sequence[int] = DEFAULT_LABELS, metric: str = "f1_macro",
                      n_boot: int = 2000, seed: int = 0, missing: str = "skip") -> Dict[str, Any]:
    """Patient bootstrap of metric(a) - metric(b) on the same rows.

    `preds_a` / `preds_b`: one prediction array each, or one per seed, paired
    by position (seed s of the arm against seed s of A0); the statistic is the
    mean over seeds of the per-seed difference. Every resample draws P patients
    with replacement and pools their rows (as per-patient confusion matrices,
    so the metric is exact). `labels` fixes the class set of every resample.

    missing  "skip" (default): a resample in which some label has no true row
             is counted and left out of the interval; "keep": it is scored
             anyway (macro-F1 then counts the absent class as F1 0, balanced
             accuracy averages the present classes). Either way the counts are
             returned, overall and per label."""
    if metric not in BOOTSTRAP_METRICS:
        raise ValueError(f"metric {metric!r}; expected one of {BOOTSTRAP_METRICS}")
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing {missing!r}; expected one of {MISSING_POLICIES}")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    labels = tuple(int(c) for c in labels)
    a_list, b_list = _as_list(preds_a), _as_list(preds_b)
    if len(a_list) != len(b_list) or not a_list:
        raise ValueError(f"{len(a_list)} prediction array(s) for a, {len(b_list)} for b: pair them by seed")
    patients, cm_a = None, []
    cm_b = []
    for pa, pb in zip(a_list, b_list):
        p, ca = patient_confusions(y, pa, groups, labels)
        _, cb = patient_confusions(y, pb, groups, labels)
        patients = p
        cm_a.append(ca)
        cm_b.append(cb)
    A = np.stack(cm_a).astype(np.float64)      # (S, P, K, K)
    B = np.stack(cm_b).astype(np.float64)
    fn = _METRIC_FNS[metric]
    support = A[0].sum(-1)                      # (P, K): true rows per patient and label (same for every seed)
    n_pat = len(patients)

    def stat(w: np.ndarray) -> float:
        ca = np.einsum("p,spij->sij", w, A)
        cb = np.einsum("p,spij->sij", w, B)
        return float(np.mean(fn(ca) - fn(cb)))

    observed = stat(np.ones(n_pat))
    rng = np.random.default_rng(seed)
    draws: List[float] = []
    missing_by_class = np.zeros(len(labels), dtype=np.int64)
    n_missing = 0
    for _ in range(int(n_boot)):
        w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat).astype(np.float64)
        absent = (w @ support) == 0
        if absent.any():
            n_missing += 1
            missing_by_class += absent
            if missing == "skip":
                continue
        draws.append(stat(w))
    d = np.asarray(draws, dtype=np.float64)
    used = int(d.size)
    return {
        "metric": metric, "labels": list(labels), "missing_policy": missing,
        "observed": observed,
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if used else None,
        "frac_positive": float((d > 0).mean()) if used else None,
        "n_boot_requested": int(n_boot), "n_boot_used": used,
        "n_resamples_missing_a_class": int(n_missing),
        "frac_resamples_missing_a_class": float(n_missing / n_boot),
        "missing_by_class": {str(c): int(m) for c, m in zip(labels, missing_by_class)},
        "n_patients": int(n_pat), "patients_per_class": {
            str(c): int((support[:, i] > 0).sum()) for i, c in enumerate(labels)},
        "n_seeds": len(a_list), "seed": int(seed),
        "statistic": "mean over seeds of metric(arm, seed s) - metric(A0, seed s), patients resampled "
                     "with replacement, rows pooled",
    }
