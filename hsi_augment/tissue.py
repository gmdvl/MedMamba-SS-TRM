# -*- coding: utf-8 -*-
"""
hsi_augment/tissue.py
=====================
Phase 2b (issue #13, Plan §Phase 2 items 4-7): the tissue / background (glass)
score, its threshold tau, the pre-registered glass rule, and the grid
edge-continuity check. `stages/tissue_mask.py` registers the stage; the
figures are in `tissue_figures.py`.

Why the score is an angle and not a brightness
----------------------------------------------
The build stores raw sensor counts x a per-capture gain / the corpus median
(8640.0 on trainsel), NOT reflectance: the gain is a median ratio, so how
bright glass ends up depends on how much of the capture is glass. A brightness
threshold (like the 0.85 reflectance cut of `prepare_hmi_lusc_v20.py`) does
not transfer. The SHAPE of the spectrum does: glass passes the illuminant
nearly unchanged, while stained tissue absorbs in a class-shaped way. So

    s_glass   for every train capture, the mean spectrum of its brightest
              `top_fraction` (1%) of patches by band-mean, unit-normalised;
              the band-wise median over captures, renormalised.
    t         SAM(patch mean spectrum, s_glass) in degrees. A scalar gain
              (per capture, per patch) leaves it unchanged by construction.

    tissue  <=>  t >= tau     (NaN, i.e. a zero or non-finite patch: not tissue)

tau is fitted on train rows only (`SplitGuard` purpose "fit"): a 2-component
Gaussian mixture on log t over a class x capture stratified sample, tau where
the two posteriors cross. When the mixture does not converge or its posteriors
do not cross between the two means, Otsu's threshold on the same log t
(numpy, no scikit-image) is used, and `tau.json` says so. Otsu is computed and
recorded either way, so the user signing off tau sees both.

Within-patch spatial variation is a secondary feature: `spatial_cv` = the
band-averaged coefficient of variation over the patch's pixels (std / mean per
band, so a gain cancels). It is recorded per row, and `tau.json` reports how
well it agrees with the mask (AUC), but it is not thresholded. The mask is the
one pre-registered decision; a second threshold would need its own fit and
sign-off, and CV is not scale-free in the way that matters: real glass carries
illumination falloff, dust and the sensor's stripe noise, so a smooth tissue
patch and a textured glass patch can have the same CV, while their spectral
shapes still differ.

The glass rule (Plan §Decisions, pre-registered)
------------------------------------------------
Measure the fraction of minority (DCIS) train rows the mask calls glass, per
patient and pooled. If the POOLED fraction is above 10%, the copy arms copy
only tissue DCIS rows (`train.py --minority_copy_mask dcis_copy_mask.npy`);
at or below 10% they copy every row and the confound is reported.
`dcis_copy_mask.npy` is written either way: bool, one entry per real train
row, True where a row may be copied (= the tissue mask; `training.
synthetic_mix.plan_minority_with_synthetic` reads it only at minority rows).

The grid edge-continuity check
------------------------------
Coordinates are derived, not stored (`BuildAdapter.coordinates`). If they are
right, the facing edges of grid neighbours are two adjacent pixel lines of one
image and correlate; under a wrong derivation they are unrelated. Per capture:

    pair corr   the patches' facing edges, (P pixels x B bands) each, centred
                per band along the edge (removes the spectral shape, which any
                two patches share) and correlated as one vector
    statistic   median over the capture's (r, c)/(r, c + 1) pairs, minus the
                same median over the pairs of random permutations of its
                patches (a "shuffled grid"), re-shuffled until there are at
                least `baseline_pairs` (2000) of them: one permutation of a
                54 x 91 capture, ~130 of the fixture's 4 x 5 one
    pass        that difference >= `min_delta` (0.2)

(r, c)/(r + 1, c) pairs are tested too by default: horizontal pairs are
consecutive rows of the capture's run whatever the grid width is, so only the
vertical pairs show that the width (91) is right. The stage passes when at
least `min_pass_fraction` (95%) of the tested captures pass.

The correlation is signed on purpose. A pair that straddles a glass/tissue
boundary can correlate NEGATIVELY although it is a true pair (one structure
brightens the illumination on glass and darkens absorbing tissue), so the
median, not the mean, is used; on a 54 x 91 grid such pairs are a small
minority. |corr| was tried and rejected: random short edges already have a
large |corr|, which shrinks the true-vs-shuffled gap (fixture, 10 seeds:
median delta 0.16 vs 0.74 signed).

Memory (Plan rule 5)
--------------------
`X_train` is 2.45M x 11 x 11 x 32 float16 = 19 GB and is never loaded. Two
passes stream it one capture at a time (`iter_chunks`, at most `chunk_rows`
rows per read; a capture is 4,914 rows = 38 MB), widened to float64 in blocks
of `FEATURE_BLOCK_ROWS` rows:

    pass 1  per patch: mean spectrum, band-mean brightness, spatial CV, the 4
            edges -> per capture: its glass candidate (top 1%) and its grid check
    pass 2  per patch: t against s_glass, and the 3 RGB bands of its mean
            spectrum (for the mosaics)

What is kept for every train row is a few bytes (t float32, CV and RGB
float16, the label/patient/capture sidecars): about 100 MB on the real build.

Public outputs (in the stage directory, `prep/tissue/`)
-------------------------------------------------------
    s_glass.npy              float64 [B], unit L2 norm
    tissue_score_train.npy   float16 [N_train], t in degrees, NaN where undefined
    tissue_mask_train.npy    bool    [N_train], t >= tau (`apply_threshold`)
    spatial_cv_train.npy     float16 [N_train], the secondary feature
    dcis_copy_mask.npy       bool    [N_train], = the tissue mask
    tau.json, glass_rule.json, grid_check.json, capture_tissue.json, figures/*.png

Val/test (the Plan's tissue-vs-glass breakdown, analysis only) and synthetic
samples (QC rule R4) are scored with the TRAIN reference, without labels:

    ref = load_reference(out / "prep/tissue")
    t = ref.score(x_block)           # float32 degrees, any (n, H, W, C) block
    keep = ref.is_tissue(x_block)
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hsi_augment import SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.stage import StageContext
    from hsi_augment.stages.tissue_mask import TissueMaskConfig

PathLike = Union[str, os.PathLike]

# float16 X rows widened to float64 at a time: 512 x 11 x 11 x 32 x 8 B = 16 MB.
FEATURE_BLOCK_ROWS = 512
SCORE_DTYPE = np.float16
DEFAULT_TOP_FRACTION = 0.01
DEFAULT_LOG_FLOOR_DEG = 1e-3        # log t is taken of max(t, floor): glass can be ~0 deg
GLASS_RULE_THRESHOLD = 0.10         # pre-registered (Plan §Decisions)
# Edge-correlation histogram for the pooled medians and the figure.
CORR_BINS = np.linspace(-1.0, 1.0, 401)

# Edge order in `patch_edges`.
EDGE_RIGHT, EDGE_LEFT, EDGE_BOTTOM, EDGE_TOP = range(4)

S_GLASS_NAME = "s_glass.npy"
TAU_NAME = "tau.json"
SCORE_NAME = "tissue_score_train.npy"
MASK_NAME = "tissue_mask_train.npy"
CV_NAME = "spatial_cv_train.npy"
COPY_MASK_NAME = "dcis_copy_mask.npy"
GLASS_RULE_NAME = "glass_rule.json"
GRID_CHECK_NAME = "grid_check.json"
CAPTURE_TISSUE_NAME = "capture_tissue.json"
FIGURES_DIR = "figures"

MASK_RULE = "tissue = float32(tissue_score) >= float32(tau_deg); a NaN score is not tissue"
SCORE_DEFINITION = ("t = spectral angle (degrees) between a patch's mean spectrum (mean over its "
                    "H x W pixels) and s_glass")
CV_DEFINITION = "mean over bands of (std / mean) over the patch's H x W pixels"


# ============================================================================
# per-patch features
# ============================================================================

def _blocks(n: int, size: int = FEATURE_BLOCK_ROWS):
    for a in range(0, n, size):
        yield a, min(a + size, n)


def patch_mean_spectra(x: np.ndarray) -> np.ndarray:
    """(n, H, W, B) patches -> (n, B) float64 mean spectra, widened in blocks."""
    x = np.asarray(x)
    out = np.empty((x.shape[0], x.shape[-1]), dtype=np.float64)
    for a, b in _blocks(x.shape[0]):
        out[a:b] = x[a:b].astype(np.float64).mean(axis=(1, 2))
    return out


def patch_features(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean spectra (n, B) float64, brightness = band-mean (n,), spatial CV (n,)).
    The CV of a band whose patch mean is not > 0 is undefined (NaN) and left
    out of that patch's band average."""
    x = np.asarray(x)
    n, bands = x.shape[0], x.shape[-1]
    mean = np.empty((n, bands), dtype=np.float64)
    cv = np.empty(n, dtype=np.float64)
    for a, b in _blocks(n):
        blk = x[a:b].astype(np.float64)
        m = blk.mean(axis=(1, 2))
        s = blk.std(axis=(1, 2))
        ratio = np.divide(s, m, out=np.full_like(s, np.nan), where=m > 0)
        with warnings.catch_warnings():
            # an all-NaN row warns; the NaN np.nanmean returns for it is what we want
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cv[a:b] = np.nanmean(ratio, axis=1)
        mean[a:b] = m
    return mean, mean.mean(axis=1), cv


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalised along `axis` (a zero vector stays zero)."""
    v = np.asarray(v, dtype=np.float64)
    nrm = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, nrm, out=np.zeros_like(v), where=nrm > 0)


def sam_deg(spectra: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Spectral angle in degrees between each row of `spectra` (n, B) and `ref`
    (B,), as atan2(|component orthogonal to ref|, projection on ref): unlike
    arccos(cos) it stays accurate for the sub-degree angles of glass. A row
    that is all zero or not finite gets NaN."""
    s = np.asarray(spectra, dtype=np.float64)
    r = unit(np.asarray(ref, dtype=np.float64).reshape(-1))
    if s.ndim == 1:
        s = s[None]
    if s.shape[-1] != r.shape[0]:
        raise ValueError(f"spectra have {s.shape[-1]} bands, the reference {r.shape[0]}")
    proj = s @ r
    resid = s - proj[:, None] * r[None, :]
    ang = np.degrees(np.arctan2(np.linalg.norm(resid, axis=1), proj))
    bad = ~np.isfinite(s).all(axis=1) | ~(np.linalg.norm(s, axis=1) > 0)
    ang[bad] = np.nan
    return ang


def tissue_score(x: np.ndarray, s_glass: np.ndarray) -> np.ndarray:
    """t in degrees (float32) for a block of patches (n, H, W, B). Pure function
    of the patches and the train-fitted `s_glass`: the same call scores train,
    val/test (analysis only) and synthetic samples (QC)."""
    return sam_deg(patch_mean_spectra(x), s_glass).astype(np.float32)


def apply_threshold(score: np.ndarray, tau_deg: float) -> np.ndarray:
    """The mask rule (`MASK_RULE`): bool, True = tissue. Compared in float32 so
    the saved float16 scores, re-read by a consumer, give the saved mask."""
    with np.errstate(invalid="ignore"):
        return np.asarray(score, dtype=np.float32) >= np.float32(tau_deg)


# ============================================================================
# the glass reference
# ============================================================================

def top_k(n: int, top_fraction: float) -> int:
    """Patches in a capture's "brightest top_fraction": at least one."""
    return max(1, int(math.ceil(float(top_fraction) * int(n))))


def capture_glass_candidate(mean_spectra: np.ndarray, brightness: np.ndarray,
                            top_fraction: float = DEFAULT_TOP_FRACTION) -> Tuple[np.ndarray, int, float]:
    """One capture's glass spectrum: the unit-normalised mean spectrum of its
    brightest `top_fraction` patches by band-mean (rows with an undefined
    brightness are skipped). Returns (unit spectrum, k, mean brightness of
    those k)."""
    b = np.asarray(brightness, dtype=np.float64)
    ok = np.flatnonzero(np.isfinite(b))
    if len(ok) == 0:
        raise ValueError("no patch with a finite brightness")
    k = top_k(len(ok), top_fraction)
    # stable sort on -brightness: ties keep row order, so the choice is deterministic
    top = ok[np.argsort(-b[ok], kind="stable")[:k]]
    return unit(np.asarray(mean_spectra, dtype=np.float64)[top].mean(axis=0)), int(k), float(b[top].mean())


def glass_reference(candidates: np.ndarray) -> np.ndarray:
    """s_glass: the band-wise median of the captures' unit glass candidates
    (n_captures, B), renormalised. A median, so the few captures with no
    glass (whose brightest patches are pale tissue) do not move it."""
    c = np.asarray(candidates, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] == 0:
        raise ValueError(f"need an (n_captures, B) array of candidates, got shape {c.shape}")
    return unit(np.median(c, axis=0))


# ============================================================================
# the threshold
# ============================================================================

def stratified_sample(strata: np.ndarray, valid: np.ndarray, n_target: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Sorted row indices: `n_target` of the `valid` rows, allocated to the
    strata in proportion to their size (largest remainder), drawn without
    replacement inside each stratum. All valid rows if there are no more than
    `n_target`. Proportional, so the sample has the train mixture's glass/tissue
    weights (which the GMM posterior crossing depends on), with every stratum
    represented."""
    strata = np.asarray(strata)
    idx = np.flatnonzero(np.asarray(valid, dtype=bool))
    if n_target <= 0:
        raise ValueError("n_target must be > 0")
    if len(idx) <= n_target:
        return idx.astype(np.int64)
    keys, inverse, counts = np.unique(strata[idx], return_inverse=True, return_counts=True)
    quota = n_target * counts / counts.sum()
    alloc = np.floor(quota).astype(np.int64)
    short = int(n_target - alloc.sum())
    if short:
        # largest fractional part first; ties go to the earlier stratum
        order = np.lexsort((np.arange(len(keys)), -(quota - alloc)))
        alloc[order[:short]] += 1
    members = np.argsort(inverse, kind="stable")
    bounds = np.r_[0, np.cumsum(counts)]
    picked = []
    for s in range(len(keys)):
        if alloc[s]:
            pool = idx[members[bounds[s]:bounds[s + 1]]]
            picked.append(rng.choice(pool, size=int(alloc[s]), replace=False))
    return np.sort(np.concatenate(picked)).astype(np.int64)


def gmm_crossing(weights: Sequence[float], means: Sequence[float],
                 variances: Sequence[float]) -> Optional[float]:
    """Where the posteriors of a 1-D two-component mixture are equal, between
    the two means: w1 N(x; m1, v1) = w2 N(x; m2, v2) is the quadratic
    a x^2 + b x + c = 0 with
        a = 1/(2 v2) - 1/(2 v1),  b = m1/v1 - m2/v2,
        c = m2^2/(2 v2) - m1^2/(2 v1) + ln(w1/sqrt(v1)) - ln(w2/sqrt(v2)).
    Of the roots strictly between the means, the one where the upper
    component starts to dominate (nearest the midpoint if two do). None if
    there is no such root (one component dominates the whole interval)."""
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(means, dtype=np.float64)
    v = np.asarray(variances, dtype=np.float64)
    if not (w.shape == m.shape == v.shape == (2,)) or (v <= 0).any() or (w <= 0).any():
        return None
    lo, hi = np.argsort(m)
    w1, m1, v1, w2, m2, v2 = w[lo], m[lo], v[lo], w[hi], m[hi], v[hi]
    if not m2 > m1:
        return None
    a = 1.0 / (2 * v2) - 1.0 / (2 * v1)
    b = m1 / v1 - m2 / v2
    c = m2 ** 2 / (2 * v2) - m1 ** 2 / (2 * v1) + math.log(w1 / math.sqrt(v1)) - math.log(w2 / math.sqrt(v2))
    if abs(a) < 1e-12 * max(abs(b), 1.0):
        roots = [-c / b] if b != 0 else []
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]

    def upper_minus_lower(x: float) -> float:   # log-posterior difference, > 0 where the upper wins
        return (math.log(w2) - 0.5 * math.log(v2) - (x - m2) ** 2 / (2 * v2)
                - math.log(w1) + 0.5 * math.log(v1) + (x - m1) ** 2 / (2 * v1))

    eps = 1e-9 * (m2 - m1)
    inside = [r for r in roots if m1 < r < m2
              and upper_minus_lower(r - eps) < 0 < upper_minus_lower(r + eps)]
    if not inside:
        return None
    mid = 0.5 * (m1 + m2)
    return float(min(inside, key=lambda r: abs(r - mid)))


def otsu_threshold(values: np.ndarray, bins: int = 256) -> Tuple[float, Tuple[float, float]]:
    """Otsu's threshold of 1-D `values` (numpy only): the histogram cut that
    maximises the between-class variance. Returns (threshold, histogram range);
    the threshold is a bin edge, the middle one of a run of equally good cuts."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("Otsu needs at least 2 finite values")
    lo, hi = float(v.min()), float(v.max())
    if not hi > lo:
        return lo, (lo, hi)
    hist, edges = np.histogram(v, bins=int(bins), range=(lo, hi))
    p = hist.astype(np.float64) / hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = (mu[-1] * w0 - mu) ** 2 / (w0 * (1.0 - w0))
    sb[(w0 <= 0) | (w0 >= 1)] = np.nan
    if not np.isfinite(sb).any():
        return float(np.median(v)), (lo, hi)
    # Two well-separated modes leave an empty gap in which every cut ties; the
    # middle of the tied run, not its first bin (the edge of the lower mode).
    best = np.nanmax(sb)
    tied = np.flatnonzero(sb >= best * (1.0 - 1e-9))
    run_end = tied[0]
    while run_end + 1 < len(sb) and sb[run_end + 1] >= best * (1.0 - 1e-9):
        run_end += 1
    return float(0.5 * (edges[tied[0] + 1] + edges[run_end + 1])), (lo, hi)


@dataclass(frozen=True)
class ThresholdFit:
    tau_deg: float
    log_tau: float
    method_requested: str
    method_used: str                       # "gmm" | "otsu"
    fallback_reason: Optional[str]
    gmm: Optional[Dict[str, Any]]          # None when method_requested == "otsu"
    otsu: Dict[str, Any]
    n_fit: int
    log_floor_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {"tau_deg": self.tau_deg, "log_tau": self.log_tau,
                "method_requested": self.method_requested, "method_used": self.method_used,
                "fallback_reason": self.fallback_reason, "gmm": self.gmm, "otsu": self.otsu,
                "n_fit": self.n_fit, "log_floor_deg": self.log_floor_deg}


def log_score(score_deg: np.ndarray, log_floor_deg: float = DEFAULT_LOG_FLOOR_DEG) -> np.ndarray:
    """ln(max(t, floor)) of the finite scores (the GMM / Otsu input)."""
    t = np.asarray(score_deg, dtype=np.float64)
    t = t[np.isfinite(t)]
    return np.log(np.maximum(t, float(log_floor_deg)))


def fit_threshold(score_deg: np.ndarray, *, method: str = "gmm", seed: int = 0,
                  log_floor_deg: float = DEFAULT_LOG_FLOOR_DEG, n_init: int = 3,
                  otsu_bins: int = 256) -> ThresholdFit:
    """tau from a sample of train scores (degrees). See the module docstring."""
    if method not in ("gmm", "otsu"):
        raise ValueError(f"method must be 'gmm' or 'otsu', got {method!r}")
    v = log_score(score_deg, log_floor_deg)
    if v.size < 4:
        raise ValueError(f"need at least 4 finite scores to fit a threshold, got {v.size}")
    otsu_log, otsu_range = otsu_threshold(v, otsu_bins)
    otsu = {"tau_deg": float(math.exp(otsu_log)), "log_tau": otsu_log, "bins": int(otsu_bins),
            "range_log": list(otsu_range)}

    gmm_info: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    tau_log: Optional[float] = None
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=2, covariance_type="full", n_init=int(n_init), max_iter=500,
                             tol=1e-5, random_state=int(seed))
        gm.fit(v.reshape(-1, 1))
        order = np.argsort(gm.means_.ravel())
        w = gm.weights_[order]
        m = gm.means_.ravel()[order]
        var = gm.covariances_.reshape(-1)[order]
        crossing = gmm_crossing(w, m, var)
        ashman = float(abs(m[1] - m[0]) * math.sqrt(2.0) / math.sqrt(var[0] + var[1]))
        gmm_info = {"weights": w.tolist(), "means_log": m.tolist(), "variances_log": var.tolist(),
                    "means_deg": np.exp(m).tolist(), "converged": bool(gm.converged_),
                    "n_iter": int(gm.n_iter_), "lower_bound": float(gm.lower_bound_),
                    "n_init": int(n_init), "random_state": int(seed), "crossing_log": crossing,
                    "tau_deg": None if crossing is None else float(math.exp(crossing)),
                    "ashman_d": ashman}
        if not gm.converged_:
            reason = "the GMM did not converge"
        elif crossing is None:
            reason = "the GMM posteriors do not cross between the component means"
        else:
            tau_log = crossing
        if reason is None:
            # the crossing must agree with sklearn's own posteriors
            p = gm.predict_proba(np.array([[tau_log - 1e-6], [tau_log + 1e-6]]))[:, order[1]]
            if not (p[0] <= 0.5 + 1e-3 and p[1] >= 0.5 - 1e-3):
                reason = f"the analytic crossing disagrees with predict_proba ({p.tolist()})"
                tau_log = None
    if tau_log is None:
        tau_log = otsu_log
        used = "otsu"
    else:
        used = "gmm"
    return ThresholdFit(tau_deg=float(math.exp(tau_log)), log_tau=float(tau_log), method_requested=method,
                        method_used=used, fallback_reason=reason, gmm=gmm_info, otsu=otsu, n_fit=int(v.size),
                        log_floor_deg=float(log_floor_deg))


# ============================================================================
# the glass rule
# ============================================================================

def glass_rule(is_tissue: np.ndarray, y: np.ndarray, groups: np.ndarray,
               minority_classes: Sequence[int], class_names: Sequence[str],
               threshold: float = GLASS_RULE_THRESHOLD) -> Dict[str, Any]:
    """The pre-registered decision (`glass_rule.json`): pooled glass fraction of
    the minority rows > `threshold` -> `use_copy_mask` True. Per-patient
    fractions are reported, not decided on."""
    is_tissue = np.asarray(is_tissue, dtype=bool)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if not (is_tissue.shape == y.shape == groups.shape):
        raise ValueError(f"shapes differ: mask {is_tissue.shape}, y {y.shape}, groups {groups.shape}")
    sel = np.isin(y, np.asarray(list(minority_classes), dtype=y.dtype))
    n = int(sel.sum())
    n_glass = int((~is_tissue[sel]).sum())
    frac = (n_glass / n) if n else None
    per_patient: Dict[str, Dict[str, Any]] = {}
    for p in np.unique(groups[sel]):
        m = sel & (groups == p)
        k, g = int(m.sum()), int((~is_tissue[m]).sum())
        per_patient[str(int(p))] = {"n_rows": k, "n_glass": g, "glass_fraction": g / k}
    use = bool(frac is not None and frac > float(threshold))
    return {
        "schema_version": SCHEMA_VERSION,
        "rule": ("pre-registered (Plan §Decisions): if the pooled glass fraction of the minority-class "
                 "train rows is above the threshold, the copy arms copy only tissue minority rows "
                 "(train.py --minority_copy_mask dcis_copy_mask.npy); otherwise they copy all rows "
                 "and the confound is reported"),
        "threshold": float(threshold),
        "comparison": "overall.glass_fraction > threshold",
        "minority_classes": [int(c) for c in minority_classes],
        "minority_class_names": [str(class_names[c]) for c in minority_classes],
        "overall": {"n_rows": n, "n_glass": n_glass, "glass_fraction": frac},
        "per_patient": per_patient,
        "max_patient_glass_fraction": max((v["glass_fraction"] for v in per_patient.values()), default=None),
        "use_copy_mask": use,
        "copy_mask_file": COPY_MASK_NAME,
        "copy_mask_semantics": "bool [N_train]; True = the row may be copied (= tissue mask); "
                               "only the minority rows are read",
        "n_copyable_minority_rows": n - n_glass,
        "note": None if n else "no minority-class rows were scored",
    }


# ============================================================================
# the grid edge-continuity check
# ============================================================================

def patch_edges(x: np.ndarray) -> np.ndarray:
    """(n, P, P, B) square patches -> (n, 4, P, B) float32 edges in the order
    EDGE_RIGHT (last column), EDGE_LEFT (first column), EDGE_BOTTOM (last
    row), EDGE_TOP (first row); each runs along the edge's P pixels."""
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[1] != x.shape[2]:
        raise ValueError(f"expected square NHWC patches, got shape {x.shape}")
    return np.stack([x[:, :, -1, :], x[:, :, 0, :], x[:, -1, :, :], x[:, 0, :, :]], axis=1).astype(np.float32)


def edge_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per pair, the Pearson correlation of two facing edges (m, P, B), each
    centred per band along its P pixels and then correlated as one P x B
    vector. NaN where an edge has no variation."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=(1, 2))
    den = np.sqrt((a * a).sum(axis=(1, 2)) * (b * b).sum(axis=(1, 2)))
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def grid_pair_correlations(edges_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(R, C, 4, P, B) edges on the grid -> (horizontal (R*(C-1),), vertical ((R-1)*C,))
    correlations of (r, c)/(r, c + 1) and (r, c)/(r + 1, c) pairs."""
    g = edges_grid
    rows, cols, _, p, b = g.shape
    h = edge_correlation(g[:, :-1, EDGE_RIGHT].reshape(-1, p, b), g[:, 1:, EDGE_LEFT].reshape(-1, p, b)) \
        if cols > 1 else np.empty(0)
    v = edge_correlation(g[:-1, :, EDGE_BOTTOM].reshape(-1, p, b), g[1:, :, EDGE_TOP].reshape(-1, p, b)) \
        if rows > 1 else np.empty(0)
    return h, v


def _nanmedian(a: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def capture_grid_check(edges: np.ndarray, r: np.ndarray, c: np.ndarray, grid_shape: Tuple[int, int],
                       rng: np.random.Generator, *, baseline_pairs: int = 2000, min_delta: float = 0.2,
                       vertical: bool = True) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """The check for one capture. `edges` (n, 4, P, B) in row order; `r`, `c`
    the DERIVED coordinates of those rows (the thing under test), which must
    cover the (rows, cols) grid exactly once. The shuffled-grid baseline uses
    as many permutations as it takes to reach `baseline_pairs` pairs on the
    sparser tested axis (at least one). Returns (result dict, histogram counts
    over `CORR_BINS` for the pooled statistics: h, v, h_shuffled, v_shuffled)."""
    rows, cols = grid_shape
    n = edges.shape[0]
    pos = np.asarray(r, dtype=np.int64) * cols + np.asarray(c, dtype=np.int64)
    if n != rows * cols or pos.min() < 0 or pos.max() >= n or np.unique(pos).size != n:
        raise ValueError(f"coordinates do not cover the {rows}x{cols} grid exactly once")
    grid = np.empty_like(edges)
    grid[pos] = edges
    shape = (rows, cols) + edges.shape[1:]
    h, v = grid_pair_correlations(grid.reshape(shape))
    counts = [a.size for a in ((h, v) if vertical else (h,)) if a.size]
    n_shuffles = max(1, math.ceil(int(baseline_pairs) / min(counts))) if counts else 0
    hs, vs = [], []
    for _ in range(n_shuffles):
        a, b = grid_pair_correlations(grid[rng.permutation(n)].reshape(shape))
        hs.append(a)
        vs.append(b)
    hs = np.concatenate(hs) if hs else np.empty(0)
    vs = np.concatenate(vs) if vs else np.empty(0)

    res: Dict[str, Any] = {"grid": [int(rows), int(cols)], "n_shuffles": int(n_shuffles)}
    axes = [("h", h, hs)] + ([("v", v, vs)] if vertical else [])
    ok = True
    for name, true, shuf in axes:
        med, smed = _nanmedian(true), _nanmedian(shuf)
        delta = None if med is None or smed is None else med - smed
        res.update({f"n_pairs_{name}": int(true.size), f"n_nan_{name}": int((~np.isfinite(true)).sum()),
                    f"median_{name}": med, f"shuffled_median_{name}": smed, f"delta_{name}": delta})
        if true.size == 0:
            continue                      # a 1-row or 1-column grid has no pairs on that axis
        ok = ok and delta is not None and delta >= float(min_delta)
    res["passed"] = bool(ok)

    def hist(a: np.ndarray) -> np.ndarray:
        a = a[np.isfinite(a)]
        return np.histogram(np.clip(a, -1.0, 1.0), bins=CORR_BINS)[0].astype(np.int64)

    return res, {"h": hist(h), "v": hist(v), "h_shuffled": hist(hs), "v_shuffled": hist(vs)}


def hist_median(counts: np.ndarray, edges: np.ndarray = CORR_BINS) -> Optional[float]:
    """Median of histogrammed values, linear inside the bin (resolution: one bin)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return None
    cum = np.cumsum(counts)
    k = int(np.searchsorted(cum, total / 2.0))
    before = cum[k - 1] if k else 0.0
    frac = (total / 2.0 - before) / counts[k] if counts[k] else 0.0
    return float(edges[k] + frac * (edges[k + 1] - edges[k]))


# ============================================================================
# mosaic selection
# ============================================================================

def select_mosaic_captures(captures: Sequence[Mapping[str, Any]], per_class: int,
                           minority_classes: Sequence[int]) -> Dict[int, List[int]]:
    """{label: [capture, ...]} for the sign-off mosaics. `captures`: dicts with
    capture, patient, label, glass_fraction (only captures that have a grid).
    A minority class gets one capture per patient (its most-glass capture, the
    one the glass rule is most sensitive to); every class is then filled up to
    `per_class` with the remaining captures at evenly spaced glass-fraction
    quantiles (least glass, median, most glass for 3)."""
    by_label: Dict[int, List[Mapping[str, Any]]] = {}
    for c in captures:
        by_label.setdefault(int(c["label"]), []).append(c)
    out: Dict[int, List[int]] = {}
    for label in sorted(by_label):
        caps = by_label[label]
        chosen: List[int] = []
        if label in set(int(m) for m in minority_classes):
            for p in sorted({int(c["patient"]) for c in caps}):
                mine = [c for c in caps if int(c["patient"]) == p]
                best = max(mine, key=lambda c: (float(c["glass_fraction"]), -int(c["capture"])))
                chosen.append(int(best["capture"]))
        rest = sorted((c for c in caps if int(c["capture"]) not in chosen),
                      key=lambda c: (float(c["glass_fraction"]), int(c["capture"])))
        need = int(per_class) - len(chosen)
        if need > 0 and rest:
            picks = np.unique(np.round(np.linspace(0, len(rest) - 1, min(need, len(rest)))).astype(int))
            chosen += [int(rest[i]["capture"]) for i in picks]
        out[label] = chosen
    return out


# ============================================================================
# reading the outputs back (#15, #16, #28, #33)
# ============================================================================

@dataclass(frozen=True)
class TissueReference:
    """The train-fitted score and threshold, for scoring any patches."""
    s_glass: np.ndarray
    tau_deg: float
    stage_dir: Path

    def score(self, x: np.ndarray) -> np.ndarray:
        return tissue_score(x, self.s_glass)

    def is_tissue(self, x: np.ndarray) -> np.ndarray:
        return apply_threshold(self.score(x).astype(SCORE_DTYPE), self.tau_deg)


def load_reference(stage_dir: PathLike) -> TissueReference:
    d = Path(stage_dir)
    with open(d / TAU_NAME) as f:
        tau = json.load(f)
    return TissueReference(s_glass=np.load(d / S_GLASS_NAME), tau_deg=float(tau["tau_deg"]), stage_dir=d)


# ============================================================================
# the stage
# ============================================================================

def _pick_subset(runs, labels: np.ndarray, max_captures: Optional[int]) -> List[int]:
    """Indices into `runs` to process: all, or the first `max_captures`
    round-robin over the classes (in run order), so a smoke run sees every class."""
    if max_captures is None or max_captures >= len(runs):
        return list(range(len(runs)))
    queues: Dict[int, List[int]] = {}
    for i, run in enumerate(runs):
        queues.setdefault(int(labels[run.start]), []).append(i)
    picked: List[int] = []
    while len(picked) < max_captures and any(queues.values()):
        for lab in sorted(queues):
            if queues[lab] and len(picked) < max_captures:
                picked.append(queues[lab].pop(0))
    return sorted(picked)


def run_tissue_mask(ctx: "StageContext", cfg: "TissueMaskConfig") -> Dict[str, Any]:
    """The `tissue-mask` stage: two passes over the train split, the threshold,
    the glass rule, the grid check and the figures. See the module docstring."""
    from hsi_augment.adapters.base import NonContiguousCaptureError
    from hsi_augment.io import iter_chunks
    from hsi_augment.stage import StageError

    adapter = ctx.adapter
    class_names = adapter.class_names
    minority = adapter.minority_classes
    wavelengths = np.asarray(adapter.wavelengths, dtype=np.float64)
    train = ctx.open_split("train", purpose="fit")
    n_rows = train.n
    n_bands = int(train.X.shape[-1])
    if len(wavelengths) != n_bands:
        raise StageError(f"wavelengths.npy has {len(wavelengths)} bands, X_train has {n_bands}")
    row_values = int(np.prod(train.X.shape[1:]))

    y = np.asarray(train.y)
    groups = np.asarray(train.groups) if train.groups is not None else None
    if groups is None or train.captures is None:
        raise StageError("the train split needs groups_train.npy and captures_train.npy")
    try:
        coords = train.coordinates()
    except NonContiguousCaptureError as e:
        raise StageError(f"grid coordinates undefined: {e}") from e
    runs = coords.runs
    grid_shape = coords.grid_shape
    # keep only what is used per row; int32 is plenty for a grid position
    r_all = coords.r.astype(np.int32)
    c_all = coords.c.astype(np.int32)
    del coords

    todo = _pick_subset(runs, y, cfg.max_captures)
    complete = len(todo) == len(runs)
    if not complete:
        ctx.warn(f"max_captures={cfg.max_captures}: processing {len(todo)} of {len(runs)} train captures; "
                 f"the outputs cover only their rows and say complete=false")

    max_run = max((runs[i].n for i in todo), default=0)
    chunk = min(cfg.chunk_rows, max_run) if max_run else 0
    p = int(train.X.shape[1])
    per_row = 4 + 2 + 6 + 3 + 8 + 4 + 4 + 8          # score, cv, rgb, masks, y, groups, captures, r/c
    edges_bytes = max_run * 4 * p * n_bands * 4
    projected = (n_rows * per_row
                 + chunk * row_values * train.X.dtype.itemsize
                 + 3 * FEATURE_BLOCK_ROWS * row_values * 8
                 + 6 * edges_bytes)
    ctx.ram_guard(projected, "per-row features + one capture's chunk, float64 blocks and edges")

    rgb_bands, rgb_note = _rgb_bands(n_bands, wavelengths, cfg.rgb_nm)
    rgb_label = "R/G/B = " + "/".join(f"{wavelengths[i]:.0f}" for i in rgb_bands) + " nm"
    ctx.log(f"{n_rows} train rows, {len(runs)} captures ({len(todo)} processed), {n_bands} bands, "
            f"patch {p}; spectral gaps after bands {adapter.spectral_gaps()}; {rgb_note}")

    # ---------------------------------------------------------------- pass 1
    processed = np.zeros(n_rows, dtype=bool)
    cv_rows = np.full(n_rows, np.nan, dtype=np.float16)
    candidates = np.zeros((len(todo), n_bands), dtype=np.float64)
    cap_info: List[Dict[str, Any]] = []
    grid_results: List[Dict[str, Any]] = []
    hists = {k: np.zeros(len(CORR_BINS) - 1, dtype=np.int64) for k in ("h", "v", "h_shuffled", "v_shuffled")}
    with ctx.timer("pass1"):
        for j, i in enumerate(todo):
            run = runs[i]
            info = adapter.capture_index.get(run.capture)
            label = int(y[run.start])
            gs = grid_shape.get(run.capture)
            means, bright, edges = [], [], []
            for ch in iter_chunks(train.X, np.arange(run.start, run.stop), chunk_rows=cfg.chunk_rows,
                                  assume_sorted=True):
                m, b, cv = patch_features(ch.data)
                means.append(m)
                bright.append(b)
                cv_rows[ch.rows] = cv.astype(np.float16)
                if gs is not None:
                    edges.append(patch_edges(ch.data))
            m = np.concatenate(means)
            b = np.concatenate(bright)
            processed[run.start:run.stop] = True
            try:
                cand, k, top_b = capture_glass_candidate(m, b, cfg.top_fraction)
            except ValueError as e:
                raise StageError(f"capture {run.capture}: {e}") from e
            candidates[j] = cand
            entry = {"capture": int(run.capture), "key": info.key if info else "",
                     "patient": int(groups[run.start]), "label": label, "class": class_names[label],
                     "rows": [int(run.start), int(run.stop)], "n": int(run.n), "top_k": k,
                     "top_brightness": top_b, "median_brightness": float(np.nanmedian(b)),
                     "has_grid": gs is not None}
            cap_info.append(entry)
            if gs is not None:
                try:
                    res, h = capture_grid_check(
                        np.concatenate(edges), r_all[run.start:run.stop], c_all[run.start:run.stop], gs,
                        ctx.rng("grid-shuffle", int(run.capture)), baseline_pairs=cfg.grid_baseline_pairs,
                        min_delta=cfg.grid_min_delta, vertical=cfg.grid_vertical)
                except ValueError as e:
                    raise StageError(f"capture {run.capture}: {e}") from e
                grid_results.append({"capture": entry["capture"], "key": entry["key"],
                                     "patient": entry["patient"], "class": entry["class"], **res})
                for kk in hists:
                    hists[kk] += h[kk]
            del means, bright, edges, m, b
            if (j + 1) % 50 == 0 or j + 1 == len(todo):
                ctx.log(f"pass 1: {j + 1}/{len(todo)} captures")
            ctx.log_metrics({"pass": 1, "capture": int(run.capture), "captures_done": j + 1})

    grid = _grid_summary(grid_results, hists, cfg)
    ctx.save_json(GRID_CHECK_NAME, grid)
    from hsi_augment import tissue_figures
    with ctx.timer("figures"):
        ctx.add_output(tissue_figures.plot_grid_check(ctx.path(FIGURES_DIR, "grid_check.png"), grid, hists,
                                                      CORR_BINS))
    if grid["passed"] is False:
        msg = (f"grid edge-continuity check FAILED: {grid['n_captures_passed']}/{grid['n_captures_tested']} "
               f"captures pass ({grid['pass_fraction']:.1%} < {cfg.grid_min_pass_fraction:.0%}); the derived "
               f"(r, c) coordinates are not trustworthy. See {ctx.path(GRID_CHECK_NAME)}")
        if cfg.grid_strict:
            raise StageError(msg)
        ctx.warn(msg + " (grid_strict off: continuing)")
    elif grid["passed"] is None:
        ctx.warn("grid check: no capture has grid coordinates; nothing was tested")
    else:
        ctx.log(f"grid check passed: {grid['n_captures_passed']}/{grid['n_captures_tested']} captures "
                f"(pooled median h {grid['pooled']['median_h']:.3f} vs shuffled "
                f"{grid['pooled']['shuffled_median_h']:.3f})")

    s_glass = glass_reference(candidates)
    cand_sam = sam_deg(candidates, s_glass)
    for entry, a in zip(cap_info, cand_sam):
        entry["candidate_sam_deg"] = float(a)
    ctx.log(f"s_glass from {len(todo)} captures; their glass candidates are within median "
            f"{np.median(cand_sam):.2f} deg (max {np.max(cand_sam):.2f}) of it")

    # ---------------------------------------------------------------- pass 2
    score = np.full(n_rows, np.nan, dtype=np.float32)
    rgb_rows = np.full((n_rows, 3), np.nan, dtype=np.float16)
    class_sum = np.zeros((len(class_names), n_bands), dtype=np.float64)
    class_n = np.zeros(len(class_names), dtype=np.int64)
    with ctx.timer("pass2"):
        for j, i in enumerate(todo):
            run = runs[i]
            for ch in iter_chunks(train.X, np.arange(run.start, run.stop), chunk_rows=cfg.chunk_rows,
                                  assume_sorted=True):
                m = patch_mean_spectra(ch.data)
                score[ch.rows] = sam_deg(m, s_glass).astype(np.float32)
                rgb_rows[ch.rows] = m[:, list(rgb_bands)].astype(np.float16)
                u = unit(m)
                ok = np.isfinite(u).all(axis=1)
                lab = y[ch.rows]
                for k in np.unique(lab[ok]):
                    sel = ok & (lab == k)
                    class_sum[k] += u[sel].sum(axis=0)
                    class_n[k] += int(sel.sum())
            if (j + 1) % 50 == 0 or j + 1 == len(todo):
                ctx.log(f"pass 2: {j + 1}/{len(todo)} captures")
            ctx.log_metrics({"pass": 2, "capture": int(run.capture), "captures_done": j + 1})

    score16 = score.astype(SCORE_DTYPE)
    del score
    finite = processed & np.isfinite(score16)
    n_undefined = int((processed & ~np.isfinite(score16)).sum())
    if n_undefined:
        ctx.warn(f"{n_undefined} processed rows have an undefined score (zero or non-finite patch): not tissue")

    # ------------------------------------------------------------- threshold
    with ctx.timer("fit"):
        strata = train.captures.astype(np.int64) * len(class_names) + y
        sample = stratified_sample(strata, finite, cfg.sample_rows, ctx.rng("gmm-sample"))
        try:
            fit = fit_threshold(score16[sample].astype(np.float64), method=cfg.method,
                                seed=ctx.seed_int("gmm"), log_floor_deg=cfg.log_floor_deg,
                                n_init=cfg.gmm_n_init, otsu_bins=cfg.otsu_bins)
        except ValueError as e:
            raise StageError(f"threshold fit: {e}") from e
    if fit.fallback_reason:
        ctx.warn(f"GMM fallback to Otsu: {fit.fallback_reason}")
    ctx.log(f"tau = {fit.tau_deg:.3f} deg ({fit.method_used}; Otsu {fit.otsu['tau_deg']:.3f}"
            + (f", GMM {fit.gmm['tau_deg']:.3f}" if fit.gmm and fit.gmm.get("tau_deg") else "")
            + f") from {len(sample)} sampled rows")

    mask = apply_threshold(score16, fit.tau_deg) & processed
    rule = glass_rule(mask[processed], y[processed], groups[processed], minority, class_names,
                      cfg.glass_rule_threshold)
    rule["complete"] = complete
    of = rule["overall"]["glass_fraction"]
    ctx.log(f"glass rule: minority glass fraction "
            f"{'n/a' if of is None else f'{of:.1%}'} (threshold {cfg.glass_rule_threshold:.0%}) -> "
            f"use_copy_mask = {rule['use_copy_mask']}")
    if rule["use_copy_mask"] and rule["n_copyable_minority_rows"] == 0:
        ctx.warn("the copy mask allows no minority row: train.py --minority_copy_mask would refuse it")

    # per capture
    for entry in cap_info:
        a, b = entry["rows"]
        s = score16[a:b].astype(np.float64)
        entry["glass_fraction"] = float(1.0 - mask[a:b].mean())
        entry["median_score_deg"] = _nanmedian(s)
    per_class = {}
    for lab, name in enumerate(class_names):
        sel = processed & (y == lab)
        per_class[name] = float(mask[sel].mean()) if sel.any() else None

    # secondary feature vs the mask, on the threshold sample
    cv_s = cv_rows[sample].astype(np.float64)
    m_s = mask[sample]
    cv_summary = {"definition": CV_DEFINITION, "thresholded": False,
                  "median_tissue": _nanmedian(cv_s[m_s]), "median_glass": _nanmedian(cv_s[~m_s]),
                  "auc_tissue_vs_glass": _auc(cv_s[m_s], cv_s[~m_s]), "n": int(len(sample))}

    sample_per_class = {class_names[k]: int(v) for k, v in
                        enumerate(np.bincount(y[sample], minlength=len(class_names)))}
    tau_doc = {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        **fit.to_dict(),
        "mask_rule": MASK_RULE,
        "score": {"definition": SCORE_DEFINITION, "units": "degrees", "dtype": "float16",
                  "file": SCORE_NAME, "s_glass_file": S_GLASS_NAME,
                  "log_input": "ln(max(t, log_floor_deg))"},
        "sample": {"requested": int(cfg.sample_rows), "n_rows": int(len(sample)),
                   "strata": "capture x class", "n_strata": int(np.unique(strata[sample]).size),
                   "allocation": "proportional to stratum size (largest remainder), without replacement",
                   "per_class": sample_per_class},
        "s_glass": {"top_fraction": float(cfg.top_fraction), "n_captures": len(todo),
                    "top_k_min": min(e["top_k"] for e in cap_info), "top_k_max": max(e["top_k"] for e in cap_info),
                    "candidate_sam_deg": {"median": float(np.median(cand_sam)),
                                          "p95": float(np.percentile(cand_sam, 95)), "max": float(np.max(cand_sam))},
                    "vector": s_glass.tolist()},
        "wavelengths_nm": wavelengths.tolist(),
        "spectral_gaps_after_band": adapter.spectral_gaps(),
        "n_train_rows": int(n_rows), "n_scored_rows": int(processed.sum()),
        "n_undefined_scores": n_undefined,
        "tissue_fraction": {"overall": float(mask[processed].mean()) if processed.any() else None,
                            "per_class": per_class},
        "spatial_cv": cv_summary,
        "grid_check_passed": grid["passed"],
    }

    with ctx.timer("save"):
        ctx.save_npy(S_GLASS_NAME, s_glass)
        ctx.save_npy(SCORE_NAME, score16)
        ctx.save_npy(MASK_NAME, mask)
        ctx.save_npy(CV_NAME, cv_rows)
        ctx.save_npy(COPY_MASK_NAME, mask.copy())
        ctx.save_json(TAU_NAME, tau_doc)
        ctx.save_json(GLASS_RULE_NAME, rule)
        ctx.save_json(CAPTURE_TISSUE_NAME, {"schema_version": SCHEMA_VERSION, "complete": complete,
                                            "tau_deg": fit.tau_deg, "rgb_bands": list(rgb_bands),
                                            "rgb_note": rgb_note, "captures": cap_info})

    # ---------------------------------------------------------------- figures
    with ctx.timer("figures"):
        figs = _write_figures(ctx, cfg, tissue_figures, cap_info, grid_shape, r_all, c_all, score16, mask,
                              rgb_rows, rgb_label, fit, score16[sample], s_glass, candidates, class_sum,
                              class_n, wavelengths, rule, minority)
    ctx.log(f"{len(figs) + 1} figure(s) in {ctx.path(FIGURES_DIR)}")

    return {"complete": complete, "n_train_rows": int(n_rows), "n_scored_rows": int(processed.sum()),
            "tau_deg": fit.tau_deg, "method_used": fit.method_used,
            "tissue_fraction": tau_doc["tissue_fraction"]["overall"],
            "minority_glass_fraction": rule["overall"]["glass_fraction"],
            "use_copy_mask": rule["use_copy_mask"], "grid_pass_fraction": grid["pass_fraction"],
            "grid_passed": grid["passed"], "figures": len(figs) + 1}


def _rgb_bands(n_bands: int, wavelengths: np.ndarray,
               rgb_nm: Sequence[float]) -> Tuple[Tuple[int, int, int], str]:
    from training.recon_artifacts import resolve_rgb_bands
    spec = ",".join(f"{float(v):g}" for v in rgb_nm)
    idx, note = resolve_rgb_bands(n_bands, wavelengths, spec)
    return tuple(int(i) for i in idx), note  # type: ignore[return-value]


def _auc(pos: np.ndarray, neg: np.ndarray) -> Optional[float]:
    """P(pos > neg) + 0.5 P(tie) (the Mann-Whitney AUC), finite values only."""
    from scipy.stats import rankdata
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not (pos.size and neg.size):
        return None
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _grid_summary(results: List[Dict[str, Any]], hists: Mapping[str, np.ndarray],
                  cfg: "TissueMaskConfig") -> Dict[str, Any]:
    tested = len(results)
    passed_n = sum(1 for r in results if r["passed"])
    frac = passed_n / tested if tested else None
    return {
        "schema_version": SCHEMA_VERSION,
        "test": ("facing edges of grid neighbours (r, c)/(r, c + 1)"
                 + (" and (r, c)/(r + 1, c)" if cfg.grid_vertical else "")
                 + " correlate if the derived coordinates are right; compared per capture against "
                   "random permutations of the same capture's patches (a shuffled grid)"),
        "pair_statistic": "Pearson correlation of the two P x B edges, each centred per band along the edge",
        "capture_statistic": "median over the capture's pairs minus the median over its shuffled-grid pairs",
        "min_delta": float(cfg.grid_min_delta), "baseline_pairs": int(cfg.grid_baseline_pairs),
        "vertical": bool(cfg.grid_vertical), "min_pass_fraction": float(cfg.grid_min_pass_fraction),
        "n_captures_tested": tested, "n_captures_passed": passed_n, "pass_fraction": frac,
        "passed": None if frac is None else bool(frac >= cfg.grid_min_pass_fraction),
        "pooled": {"method": f"histogram median, bin width {CORR_BINS[1] - CORR_BINS[0]:g}",
                   **{f"{'shuffled_' if k.endswith('_shuffled') else ''}median_{k[0]}": hist_median(v)
                      for k, v in hists.items()}},
        "failed_captures": [r["capture"] for r in results if not r["passed"]],
        "per_capture": results,
    }


def _write_figures(ctx, cfg, figmod, cap_info, grid_shape, r_all, c_all, score16, mask, rgb_rows, rgb_label,
                   fit, sample_scores, s_glass, candidates, class_sum, class_n, wavelengths, rule,
                   minority) -> List[Path]:
    from training.recon_artifacts import composite_range, rgb_composite
    names = ctx.adapter.class_names
    out: List[Path] = []

    eligible = [e for e in cap_info if e["has_grid"]]
    chosen = select_mosaic_captures(eligible, cfg.mosaics_per_class, minority)
    by_cap = {e["capture"]: e for e in eligible}
    finite = score16[np.isfinite(score16)].astype(np.float64)
    vmax = float(max(2.0 * fit.tau_deg, np.percentile(finite, 99))) if finite.size else 2.0 * fit.tau_deg
    for label, caps in chosen.items():
        panels = []
        for cap in caps:
            e = by_cap[cap]
            a, b = e["rows"]
            rows, cols = grid_shape[cap]
            pos = r_all[a:b].astype(np.int64) * cols + c_all[a:b]
            sc = np.full(rows * cols, np.nan)
            mk = np.zeros(rows * cols, dtype=bool)
            cube = np.full((rows * cols, 3), np.nan)
            sc[pos] = score16[a:b]
            mk[pos] = mask[a:b]
            cube[pos] = rgb_rows[a:b]
            cube_chw = cube.T.reshape(3, rows, cols)
            lo, hi = composite_range(cube_chw, (0, 1, 2))
            panels.append(figmod.MosaicPanel(
                title=f"{e['key'] or 'capture ' + str(cap)}\npatient {e['patient']} - glass {e['glass_fraction']:.1%}",
                rgb=rgb_composite(cube_chw, (0, 1, 2), lo, hi), score=sc.reshape(rows, cols),
                mask=mk.reshape(rows, cols)))
        if panels:
            p = ctx.path(FIGURES_DIR, f"tissue_maps_{_slug(names[label])}.png")
            out.append(ctx.add_output(figmod.plot_capture_mosaics(p, names[label], panels, fit.tau_deg, vmax,
                                                                  rgb_label)))
    out.append(ctx.add_output(figmod.plot_score_histogram(ctx.path(FIGURES_DIR, "tissue_score_hist.png"),
                                                          sample_scores.astype(np.float64), fit)))
    means = {names[k]: class_sum[k] / class_n[k] for k in range(len(names)) if class_n[k]}
    out.append(ctx.add_output(figmod.plot_glass_spectrum(
        ctx.path(FIGURES_DIR, "s_glass_spectrum.png"), wavelengths, ctx.adapter.spectral_segments(), s_glass,
        candidates, means)))
    out.append(ctx.add_output(figmod.plot_glass_rule(ctx.path(FIGURES_DIR, "glass_rule.png"), rule)))
    return out


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower() or "class"
