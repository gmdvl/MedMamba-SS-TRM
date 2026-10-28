# -*- coding: utf-8 -*-
"""
hsi_augment/qc/rules.py
=======================
The per-sample rules R1-R6, their rejection bitmask, and the calibration of
their thresholds on real LOPO held-out patches (Plan §Phase 8b, issue #28).

The bitmask
-----------
One bit per rule, `RULE_BITS`; 0 means accepted. uint8, one entry per sample,
stored by #29 for EVERY generated sample (accepted or not):

    bit 0  R1 validity     non-finite after the float16 cast; or too many values
                           outside the per-band [p0.05, p99.95] of train target tissue
    bit 1  R2 spectral     Mahalanobis distance of the log mean spectrum to the
                           class Gaussian; per-pixel curvature energy (within segments)
    bit 2  R3 spatial      lag-1 autocorrelation and HF power fraction inside the real range
    bit 3  R4 tissue       the tissue-mask stage's t >= tau
    bit 4  R5 consistency  cross-fitted mean/std probe p(class); kNN label agreement
    bit 5  R6 memorization exact byte match with a train patch; NN distance too small

A rule is a set of CHECKS (`CHECKS`): a flag (fails when its score is not 0)
or a threshold on one score, on the "lower" or "upper" side. A NaN score
fails every check, so a sample whose features are undefined (a flat patch has
no spatial autocorrelation) is rejected rather than waved through.

Calibration
-----------
Each rule must accept at least `rule_accept` (99%) of real LOPO held-out
target-class tissue patches, and all rules together at least `joint_accept`
(95%). The real patches are scored exactly as a generated sample is
(`QCReference.score`), by the models of the fold that held their patient out.

A thresholded check starts at its nominal tail level (the plan's: q99.5 for
R2, [q0.5, q99.5] for R3, q5 for R5, q1 for R6, the issue's 1% for R1) and
its threshold is that quantile of its reference distribution: the real
held-out scores themselves, except R6's distance, whose reference is the real
CROSS-CAPTURE nearest-neighbour distances. Quantiles are taken on the
conservative side (`method="lower"` for a lower bound, `"higher"` for an
upper bound), so a threshold is always an observed real value. If a rule
then accepts less than 99%, its checks are loosened one step at a time (the
level halves, down to 0 = the most extreme real value), always the step
that gains the most acceptance, until the rule meets the target; then the
same for the joint target over all checks. A check with a `bound` is never
made STRICTER than the bound (R1's out-of-range fraction stays at least 1%).
If a target cannot be met even with every check at level 0, calibration
fails loudly (`CalibrationError`, no done.json): the rules would reject real
unseen-patient data, and no threshold can fix that.

Thresholds are pooled over all folds. The models (Gaussian, probe, banks)
are per fold, so `fold=k` scores a sample from fold k's generators against
references that never saw fold k's held-out patients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hsi_augment.stage import StageError

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.qc.reference import QCReference

RULES: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6")
RULE_BITS: Dict[str, int] = {r: 1 << i for i, r in enumerate(RULES)}
RULE_NAMES: Dict[str, str] = {"R1": "validity", "R2": "spectral", "R3": "spatial", "R4": "tissue",
                              "R5": "class consistency", "R6": "memorization"}
BITMASK_DTYPE = np.uint8
SIDES = ("flag", "lower", "upper")
THRESHOLDS_SCHEMA_VERSION = 1


class CalibrationError(StageError):
    """The real held-out patches cannot meet the acceptance targets."""


@dataclass(frozen=True)
class Check:
    name: str                       # "R2.mahalanobis"
    rule: str                       # "R2"
    score: str                      # key in the scores dict
    side: str                       # "flag" | "lower" | "upper"
    level_param: Optional[str]      # config field holding the nominal tail level
    bound_param: Optional[str] = None   # config field of a bound the threshold never tightens past
    reference: str = "lopo_heldout"     # or "cross_capture" (R6 distance)
    doc: str = ""


CHECKS: Tuple[Check, ...] = (
    Check("R1.nonfinite", "R1", "nonfinite_frac", "flag", None,
          doc="any value is NaN or +-inf after the float16 cast"),
    Check("R1.out_of_range", "R1", "out_of_range_frac", "upper", "r1_level", bound_param="r1_max_out_fraction",
          doc="fraction of values outside the per-band [p0.05, p99.95] of train target tissue"),
    Check("R2.mahalanobis", "R2", "mahalanobis", "upper", "r2_level",
          doc="Mahalanobis distance of log(mean spectrum + eps) to the class Gaussian"),
    Check("R2.curvature_low", "R2", "curvature", "lower", "r2_level",
          doc="per-pixel spectral curvature energy within segments: too smooth"),
    Check("R2.curvature_high", "R2", "curvature", "upper", "r2_level",
          doc="per-pixel spectral curvature energy within segments: too jagged"),
    Check("R3.autocorr_low", "R3", "lag1_autocorr", "lower", "r3_level", doc="lag-1 autocorrelation too low (noise)"),
    Check("R3.autocorr_high", "R3", "lag1_autocorr", "upper", "r3_level", doc="lag-1 autocorrelation too high (blur)"),
    Check("R3.hf_low", "R3", "hf_power_frac", "lower", "r3_level", doc="high-frequency power fraction too low (blur)"),
    Check("R3.hf_high", "R3", "hf_power_frac", "upper", "r3_level", doc="high-frequency power fraction too high"),
    Check("R4.tissue", "R4", "not_tissue", "flag", None, doc="tissue score t below the tissue-mask tau"),
    Check("R5.p_class", "R5", "p_class", "lower", "r5_level", doc="cross-fitted mean/std probe p(class)"),
    Check("R5.knn_agreement", "R5", "knn_agreement", "lower", "r5_level",
          doc="fraction of the k nearest real reference patches with the class label"),
    Check("R6.exact_match", "R6", "exact_match", "flag", None, doc="byte-identical to a train patch"),
    Check("R6.nn_dist", "R6", "nn_dist", "lower", "r6_level", reference="cross_capture",
          doc="exact distance to the nearest train patch of the class, vs real cross-capture NN distances"),
)
CHECKS_BY_NAME: Dict[str, Check] = {c.name: c for c in CHECKS}


# ============================================================================
# thresholds and failures
# ============================================================================

def threshold_at(values: np.ndarray, side: str, level: float, bound: Optional[float] = None) -> float:
    """The `level` tail quantile of the finite `values` on `side`, on the
    conservative side (an observed value), never stricter than `bound`."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    level = float(min(max(level, 0.0), 1.0))
    if side == "lower":
        t = float(np.quantile(v, level, method="lower"))
        return t if bound is None else min(t, float(bound))
    if side == "upper":
        t = float(np.quantile(v, 1.0 - level, method="higher"))
        return t if bound is None else max(t, float(bound))
    raise ValueError(f"no threshold for side {side!r}")


def check_fails(score: np.ndarray, side: str, threshold: Optional[float]) -> np.ndarray:
    """Bool per sample: True = the check rejects it. NaN always fails."""
    s = np.asarray(score)
    if side == "flag":
        return ~(s == 0)
    s = s.astype(np.float64)
    if threshold is None or not np.isfinite(threshold):
        return np.ones(s.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        return ~(s >= threshold) if side == "lower" else ~(s <= threshold)


def evaluate(class_thresholds: Mapping[str, Any], scores: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """{check name: bool fail per sample} for one class's thresholds."""
    out = {}
    for name, spec in class_thresholds["checks"].items():
        out[name] = check_fails(scores[spec["score"]], spec["side"], spec.get("threshold"))
    return out


def bitmask_from_failures(fails: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=BITMASK_DTYPE)
    for name, f in fails.items():
        mask |= np.where(np.asarray(f, dtype=bool), RULE_BITS[CHECKS_BY_NAME[name].rule], 0).astype(BITMASK_DTYPE)
    return mask


def decode_bitmask(value: int) -> List[str]:
    """The rules a bitmask value rejects, e.g. 0b100001 -> ['R1', 'R6']."""
    return [r for r in RULES if int(value) & RULE_BITS[r]]


def summarize_bitmask(mask: np.ndarray) -> Dict[str, Any]:
    """Acceptance and per-rule rejection rates of a bitmask array."""
    m = np.asarray(mask, dtype=np.int64)
    n = int(m.size)
    return {"n": n, "accepted": int((m == 0).sum()), "acceptance": float((m == 0).mean()) if n else None,
            "rejected_by": {r: int(((m & RULE_BITS[r]) != 0).sum()) for r in RULES}}


def apply_rules(ref: "QCReference", x: np.ndarray, class_id: int, *, fold: Optional[int] = None,
                device: Optional[str] = None, chunk_rows: Optional[int] = None
                ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Score `x` (any float NHWC array; cast to the float16 export first) as
    samples of `class_id` and apply the calibrated thresholds.

    Returns (bitmask uint8 [n], scores): scores holds every per-sample score
    (`QCReference.score`) plus `fail.<check>` bool arrays. `fold=None` uses the
    references fitted on all of train (samples from generators trained on all
    of train); `fold=k` uses fold k's (samples from fold k's generators, #30)."""
    scores = ref.score(x, class_id, fold=fold, device=device, chunk_rows=chunk_rows)
    fails = evaluate(ref.class_thresholds(class_id), scores)
    n = len(next(iter(scores.values()))) if scores else 0
    scores.update({f"fail.{k}": v for k, v in fails.items()})
    return bitmask_from_failures(fails, n), scores


# ============================================================================
# calibration
# ============================================================================

def level_schedule(nominal: float, max_halvings: int) -> List[float]:
    """nominal, nominal/2, ..., nominal/2^max_halvings, then 0."""
    levels = [float(nominal) * 0.5 ** j for j in range(int(max_halvings) + 1)] if nominal > 0 else []
    return levels + [0.0]


def calibrate(cal_scores: Mapping[str, np.ndarray], references: Mapping[str, np.ndarray],
              nominal_levels: Mapping[str, float], bounds: Mapping[str, Optional[float]], *,
              rule_accept: float = 0.99, joint_accept: float = 0.95, max_halvings: int = 10,
              checks: Sequence[Check] = CHECKS) -> Dict[str, Any]:
    """Thresholds meeting the acceptance targets on the calibration scores.

    `cal_scores[score]`: per calibration patch (all the same length).
    `references[check name]`: the distribution a thresholded check takes its
    quantile from. Returns the per-class thresholds document (see
    `thresholds.json` in reference.py); `ok` False and `problems` when a target
    cannot be met (the caller saves the document, then raises)."""
    n = len(next(iter(cal_scores.values()))) if cal_scores else 0
    schedules = {c.name: level_schedule(nominal_levels[c.name], max_halvings) for c in checks if c.side != "flag"}
    state = {name: 0 for name in schedules}
    cache: Dict[Tuple[str, int], Tuple[float, np.ndarray]] = {}

    def at(c: Check, i: int) -> Tuple[Optional[float], np.ndarray]:
        if c.side == "flag":
            return None, check_fails(cal_scores[c.score], "flag", None)
        key = (c.name, i)
        if key not in cache:
            t = threshold_at(references[c.name], c.side, schedules[c.name][i], bounds.get(c.name))
            cache[key] = (t, check_fails(cal_scores[c.score], c.side, t))
        return cache[key]

    def fails_of(cs: Sequence[Check], override: Optional[Tuple[str, int]] = None) -> np.ndarray:
        f = np.zeros(n, dtype=bool)
        for c in cs:
            i = state.get(c.name, 0)
            if override is not None and override[0] == c.name:
                i = override[1]
            f |= at(c, i)[1]
        return f

    def acceptance(f: np.ndarray) -> float:
        return float(1.0 - f.mean()) if n else 0.0

    problems: List[str] = []
    steps: List[Dict[str, Any]] = []
    if n == 0:
        problems.append("no calibration patches")

    def loosen(cs: Sequence[Check], target: float, what: str) -> None:
        while n and acceptance(fails_of(cs)) < target:
            movable = [c for c in cs if c.side != "flag" and state[c.name] < len(schedules[c.name]) - 1]
            if not movable:
                problems.append(f"{what}: acceptance {acceptance(fails_of(cs)):.4f} < {target} with every "
                                f"check at its loosest level; failing checks: "
                                + ", ".join(f"{c.name} ({int(at(c, state.get(c.name, 0))[1].sum())})"
                                            for c in cs if at(c, state.get(c.name, 0))[1].any()))
                return
            best = max(movable, key=lambda c: (acceptance(fails_of(cs, (c.name, state[c.name] + 1))),
                                               int(at(c, state[c.name])[1].sum()), -checks.index(c)))
            state[best.name] += 1
            steps.append({"stage": what, "check": best.name, "level": schedules[best.name][state[best.name]],
                          "acceptance_after": acceptance(fails_of(cs))})

    for r in RULES:
        loosen([c for c in checks if c.rule == r], rule_accept, f"rule {r}")
    loosen(list(checks), joint_accept, "joint")

    out_checks: Dict[str, Any] = {}
    for c in checks:
        t, f = at(c, state.get(c.name, 0))
        ref = references.get(c.name) if c.side != "flag" else None
        out_checks[c.name] = {
            "rule": c.rule, "score": c.score, "side": c.side, "doc": c.doc,
            "nominal_level": None if c.side == "flag" else float(nominal_levels[c.name]),
            "level": None if c.side == "flag" else schedules[c.name][state[c.name]],
            "loosened_steps": 0 if c.side == "flag" else int(state[c.name]),
            "threshold": t, "bound": bounds.get(c.name),
            "reference": None if c.side == "flag" else c.reference,
            "n_reference": None if ref is None else int(np.isfinite(np.asarray(ref, dtype=np.float64)).sum()),
            "calibration_rejects": int(f.sum())}
    rule_acc = {r: acceptance(fails_of([c for c in checks if c.rule == r])) for r in RULES}
    joint = acceptance(fails_of(list(checks)))
    return {"ok": not problems, "problems": problems, "n_calibration": int(n),
            "targets": {"per_rule_accept": float(rule_accept), "joint_accept": float(joint_accept)},
            "rule_accept": rule_acc, "joint_accept": joint,
            "loosened": [name for name, i in state.items() if i > 0], "steps": steps, "checks": out_checks}
