# -*- coding: utf-8 -*-
"""
hsi_augment/preprocess.py
=========================
Phase 2a (#12, Plan §Phase 2 items 1-3): the spectral audit and the train-only
linear statistics, run by `augment_pipeline.py spectral-audit` (group
`preprocess`; stage module `hsi_augment/stages/spectral_audit.py`).

This phase AUDITS; it does not re-calibrate
-------------------------------------------
Pass 1 of the build already gain-corrected and band-selected the cubes, fitted
on train captures only. There is no white or dark reference to re-calibrate
with, and the classifier's data must stay fixed, so nothing here changes a
value. Problems are measured and REPORTED (`problems` in each report, one
warning per problem in `stage.log`); nothing is fixed, clipped or dropped.

Units
-----
Values are raw sensor counts x capture gain / value scale. With the
median-ratio gain (corpus median / the capture's median intensity) and the
value scale equal to the corpus median (8640.0 on trainsel), a value is the
raw count divided by its capture's median intensity: relative brightness, NOT
reflectance. Every report carries this note under `units`.

One streaming pass
------------------
`X_train` is 2.45M x 11x11x32 float16, 19 GB, so it is never held. Everything
that needs rows comes from ONE sequential pass over 20k-row memmap chunks
(`SpectralScan`). Each chunk is widened to float64 in small row blocks
(`io.STATS_BLOCK_VALUES`), and each block feeds:

  * the exact per-band mean/std over every train pixel (`io.RunningBandStats`);
  * per-row band moments, for the standard errors below (`RowMoments`);
  * saturation, zero, near-zero and negative counts, and the implied raw count
    (value x value scale / capture gain) against the sensor ceiling;
  * the band-band correlation of a seeded 200k-row sample (`RunningCovariance`);
  * the Laplacian residuals of a seeded noise sample;
  * the rows `compute_global_channel_stats` would sample, gathered as they
    stream past.

Every sample is drawn up front as sorted row numbers and picked out of the
chunk that holds it, so the pass reads each byte once and never seeks.

Per-band noise sigma_b
----------------------
At the 9x9 interior pixels of each 11x11 patch (no padding, so no border
artefact):

    r = x(i,j) - [x(i-1,j) + x(i+1,j) + x(i,j-1) + x(i,j+1)] / 4

A locally linear signal leaves r = 0. Independent noise of std s leaves
Var(r) = s^2 (1 + 4/16) = 1.25 s^2. So

    sigma_b = 1.4826 x MAD_b(r) / sqrt(1.25)

is the per-pixel noise std in value units. That is what `sensor_noise` (#19)
adds as N(0, k sigma_b). The un-normalised robust std of r is reported too,
as `residual_robust_std`. Texture finer than a pixel also leaves a residual,
so sigma_b is an upper bound on the sensor noise. `noise_sigma_flat` repeats
the estimate on the flattest patches of the sample (mostly glass) to show by
how much.

Comparison with train.py's sample estimate
------------------------------------------
`train.py` never persists its normalisation. It calls
`training.npy_data.compute_global_channel_stats(X_train, sample_cap=5000,
seed=--seed)`, a float32 mean/std over 5,000 random rows. `norm_stats.json`
replicates that estimate bit for bit, on the same rows, and gives its
difference from the exact stats in units of its standard error. The sample is
5,000 ROWS of 121 correlated pixels each, drawn without replacement, so the
standard error is a cluster one:

    SE(mean_b) = sqrt(S_u^2 / n x (1 - n/N))      u_i = row i's band mean
    SE(var_b)  = sqrt(S_w^2 / n x (1 - n/N))      w_i = row i's mean of (x - mean_b)^2
    SE(std_b)  = SE(var_b) / (2 std_b)            (delta method)

The S^2 are over all N train rows. With v_i = row i's mean of x^2,
S_w^2 = S_v^2 - 4 mean_b S_uv + 4 mean_b^2 S_u^2, so one pass is enough. The
estimate agrees when every |z| <= z_tol.

The log-space generator statistics are not computed here. They are fitted on
tissue rows in #16, after the tissue mask (#13).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hsi_augment import SCHEMA_VERSION
from hsi_augment import io as hio
from hsi_augment.config import to_jsonable
from hsi_augment.io import SPLITS, RunningBandStats, iter_chunks

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.adapters.base import BuildAdapter

FLOAT16_MAX = 65504.0          # the largest finite float16: where an overflowing value is clipped to
TRAIN_PY_STD_FLOOR = 1e-6      # compute_global_channel_stats replaces std <= 1e-6 by 1.0
TRAIN_PY_SAMPLE_CAP = 5000     # train.py --norm_stats_sample_cap default
MAD_TO_SIGMA = 1.4826          # MAD -> std for a Gaussian
LAPLACIAN_NOISE_VAR = 1.25     # Var(r) / s^2 for the 4-neighbour residual of i.i.d. noise
EXHAUSTIVE_RTOL = 1e-5         # float32-vs-float64 tolerance when the "sample" is every row

NORM_STATS_NAME = "norm_stats.json"
BAND_REPORT_NAME = "band_report.json"
CALIBRATION_REPORT_NAME = "calibration_report.json"
NOISE_SIGMA_NAME = "noise_sigma.npy"

SEVERITIES = ("error", "warning", "info")


# ============================================================================
# small helpers
# ============================================================================

def json_safe(obj: Any) -> Any:
    """`to_jsonable`, with NaN/Inf floats written as null. `json.dump` would
    write them as the bare tokens NaN / Infinity, which are not JSON."""
    def walk(o: Any) -> Any:
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        return o
    return walk(to_jsonable(obj))


def problem(severity: str, kind: str, detail: str, **extra: Any) -> Dict[str, Any]:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    return {"severity": severity, "kind": kind, "detail": detail, **extra}


def count_by_severity(problems: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    c = Counter(p["severity"] for p in problems)
    return {s: int(c.get(s, 0)) for s in SEVERITIES}


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    """n, mean, std and quantiles of the finite values (None and NaN skipped)."""
    v = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))],
                   dtype=np.float64)
    out: Dict[str, Any] = {"n": int(v.size)}
    if v.size == 0:
        return out
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    out.update({"min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]), "median": float(q[3]),
                "p75": float(q[4]), "p95": float(q[5]), "max": float(q[6]),
                "mean": float(v.mean()), "std": float(v.std())})
    return out


def sample_rows(rng: np.random.Generator, n_rows: int, k: int) -> np.ndarray:
    """`k` distinct rows of `range(n_rows)` (all of them if k >= n_rows),
    sorted, int64."""
    if k >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    return np.sort(rng.choice(n_rows, size=int(k), replace=False)).astype(np.int64)


# ============================================================================
# units and calibration metadata
# ============================================================================

def _gain_report(adapter: "BuildAdapter") -> Dict[str, Any]:
    """`capture_gain_report.json`, or {} if the adapter/build has none."""
    try:
        report = getattr(adapter, "capture_gain_report")
    except (AttributeError, FileNotFoundError, KeyError):
        return {}
    return report if isinstance(report, dict) else {}


def _provenance(adapter: "BuildAdapter") -> Dict[str, Any]:
    try:
        extra = adapter.manifest.get("extra") or {}
    except FileNotFoundError:
        return {}
    return {"provenance": extra.get("provenance") or {}, "cli_args": extra.get("cli_args") or {}}


def _float_or_none(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def value_scale(adapter: "BuildAdapter") -> Optional[float]:
    """The divisor of the stored values: `hsi_value_scale_resolved` from the
    gain report or the manifest provenance, else the corpus median."""
    gr = _gain_report(adapter)
    prov = _provenance(adapter).get("provenance", {})
    for v in (gr.get("hsi_value_scale_resolved"), prov.get("hsi_value_scale_resolved"),
              gr.get("corpus_median")):
        f = _float_or_none(v)
        if f is not None and f > 0:
            return f
    return None


def gain_clip(adapter: "BuildAdapter") -> Optional[Tuple[float, float]]:
    """(lo, hi) of the gain clip recorded in the manifest, or None."""
    p = _provenance(adapter)
    clip = p.get("provenance", {}).get("gain_clip")
    if isinstance(clip, (list, tuple)) and len(clip) == 2:
        lo, hi = _float_or_none(clip[0]), _float_or_none(clip[1])
        if lo is not None and hi is not None:
            return lo, hi
    cli = p.get("cli_args", {})
    lo, hi = _float_or_none(cli.get("gain_clip_lo")), _float_or_none(cli.get("gain_clip_hi"))
    return (lo, hi) if lo is not None and hi is not None else None


def units_note(adapter: "BuildAdapter") -> Dict[str, Any]:
    gr = _gain_report(adapter)
    scale = value_scale(adapter)
    cm = _float_or_none(gr.get("corpus_median"))
    method = gr.get("capture_gain_method") or _provenance(adapter).get("provenance", {}).get(
        "capture_gain_method")
    return {
        "value_space": "raw sensor counts x capture gain / value scale",
        "capture_gain_method": method,
        "corpus_median": cm,
        "value_scale": scale,
        "note": (f"Values are raw counts / the capture's median intensity (median-ratio gain = "
                 f"corpus median / capture median; value scale = corpus median = {cm} on this build; "
                 f"a clipped gain breaks the identity for its capture). They are NOT reflectance: "
                 f"there is no white or dark reference."),
    }


def gain_lookup(adapter: "BuildAdapter") -> np.ndarray:
    """float64 array indexed by capture number: the capture's gain, NaN if unknown."""
    ci = adapter.capture_index
    if not ci:
        return np.zeros(0, dtype=np.float64)
    lut = np.full(max(ci) + 1, np.nan, dtype=np.float64)
    for idx, c in ci.items():
        if c.gain is not None and c.gain > 0:
            lut[idx] = c.gain
    return lut


def calibration_audit(adapter: "BuildAdapter", *, outlier_z: float = 3.5,
                      atol: float = 1e-9) -> Dict[str, Any]:
    """Gain and median-intensity distributions per split, class and patient,
    and the list of clipped or otherwise suspect captures, from
    `capture_index.json` and `capture_gain_report.json` (metadata only: no
    array is read, so no split is opened).

    A capture is suspect for any of these reasons:
      listed_suspect      in the gain report's `suspect_captures`
      gain_clipped        `raw_gain_before_clip` differs from the applied gain
      at_clip_bound       the applied gain equals a bound of the manifest's gain clip
      gain_outlier        |robust z| of log(gain) above `outlier_z`; the median
                          and MAD are fitted on TRAIN captures only
      gain_mismatch       capture_index.json and the gain report disagree (an error)
      no_gain             no gain recorded
      not_in_gain_report  the gain report has no entry for it
    """
    caps = list(adapter.capture_index.values())
    gr = _gain_report(adapter)
    per_capture: Mapping[str, Any] = gr.get("per_capture") or {}
    listed = [str(k) for k in (gr.get("suspect_captures") or [])]
    clip = gain_clip(adapter)
    names = adapter.class_names
    problems: List[Dict[str, Any]] = []
    if not gr:
        problems.append(problem("warning", "no_gain_report",
                                "capture_gain_report.json is missing: clipping cannot be checked "
                                "against the raw gains"))

    def cls(c) -> str:
        return names[c.label] if c.label is not None else str(c.tissue)

    train_log = np.array([math.log(c.gain) for c in caps
                          if c.split == "train" and c.gain is not None and c.gain > 0])
    ref: Dict[str, Any] = {"fitted_on": "train", "n_captures": int(train_log.size), "z": outlier_z}
    scale = 0.0
    if train_log.size:
        med = float(np.median(train_log))
        mad = float(np.median(np.abs(train_log - med)))
        scale = MAD_TO_SIGMA * mad
        ref.update({"median_log_gain": med, "mad_log_gain": mad, "robust_sigma_log_gain": scale})

    suspects: List[Dict[str, Any]] = []
    for c in caps:
        reasons: List[str] = []
        pc = per_capture.get(c.key) or {}
        raw = _float_or_none(pc.get("raw_gain_before_clip"))
        z: Optional[float] = None
        if c.key in listed:
            reasons.append("listed_suspect")
        if c.gain is None:
            reasons.append("no_gain")
        else:
            if raw is not None and abs(raw - c.gain) > atol * max(1.0, abs(raw)):
                reasons.append("gain_clipped")
            if clip is not None and min(abs(c.gain - clip[0]), abs(c.gain - clip[1])) <= atol:
                reasons.append("at_clip_bound")
            rep_gain = _float_or_none(pc.get("gain"))
            if rep_gain is not None and abs(rep_gain - c.gain) > atol * max(1.0, abs(c.gain)):
                reasons.append("gain_mismatch")
            if scale > 0 and c.gain > 0:
                z = (math.log(c.gain) - ref["median_log_gain"]) / scale
                if abs(z) > outlier_z:
                    reasons.append("gain_outlier")
        if per_capture and c.key not in per_capture:
            reasons.append("not_in_gain_report")
        if reasons:
            suspects.append({"key": c.key, "index": c.index, "patient": c.patient, "split": c.split,
                             "tissue": c.tissue, "class": cls(c), "n_patches": c.n, "gain": c.gain,
                             "raw_gain_before_clip": raw, "median_intensity": c.median_intensity,
                             "robust_z_log_gain": z, "reasons": reasons})

    known = {c.key for c in caps}
    listed_missing = sorted(set(listed) - known)
    if listed_missing:
        problems.append(problem("error", "listed_suspect_not_in_index",
                                f"{len(listed_missing)} capture(s) listed as suspect are not in "
                                f"capture_index.json", keys=listed_missing))
    mismatched = [s["key"] for s in suspects if "gain_mismatch" in s["reasons"]]
    if mismatched:
        problems.append(problem("error", "gain_mismatch",
                                "capture_index.json and capture_gain_report.json disagree on the gain",
                                keys=mismatched))
    for split in SPLITS:
        here = [s for s in suspects if s["split"] == split]
        if here:
            sev = "warning"
            where = ("they are part of every train statistic in this stage" if split == "train"
                     else "held out: they do not enter any train statistic")
            problems.append(problem(sev, "suspect_captures",
                                    f"{len(here)} suspect {split} capture(s) of patient(s) "
                                    f"{sorted({s['patient'] for s in here})}; {where}",
                                    split=split, keys=[s["key"] for s in here]))

    present = [s for s in SPLITS if any(c.split == s for c in caps)]
    classes = list(names) + sorted({cls(c) for c in caps} - set(names))
    gain = {"by_split": {s: summarize(c.gain for c in caps if c.split == s) for s in present},
            "by_split_class": {s: {k: summarize(c.gain for c in caps if c.split == s and cls(c) == k)
                                   for k in classes} for s in present}}
    intensity = {"by_split": {s: summarize(c.median_intensity for c in caps if c.split == s)
                              for s in present},
                 "by_split_class": {s: {k: summarize(c.median_intensity for c in caps
                                                     if c.split == s and cls(c) == k)
                                        for k in classes} for s in present}}
    by_patient: Dict[str, Any] = {}
    for p in sorted({c.patient for c in caps}):
        mine = [c for c in caps if c.patient == p]
        by_patient[str(p)] = {
            "split": sorted({c.split for c in mine}),
            "classes": sorted({cls(c) for c in mine}),
            "n_captures": len(mine),
            "n_suspect": sum(1 for s in suspects if s["patient"] == p),
            "gain": summarize(c.gain for c in mine),
            "median_intensity": summarize(c.median_intensity for c in mine),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "sources": ["capture_index.json", "capture_gain_report.json", "dataset_manifest.json"],
        "reads_arrays": False,
        "units": units_note(adapter),
        "capture_gain_method": gr.get("capture_gain_method"),
        "corpus_median": _float_or_none(gr.get("corpus_median")),
        "value_scale": value_scale(adapter),
        "gain_clip": list(clip) if clip else None,
        "n_captures": {s: sum(1 for c in caps if c.split == s) for s in present},
        "gain": gain,
        "median_intensity": intensity,
        "by_patient": by_patient,
        "gain_outlier_reference": ref,
        "suspect_captures": suspects,
        "suspect_summary": {
            "n": len(suspects),
            "by_split": {s: sum(1 for x in suspects if x["split"] == s) for s in present},
            "by_reason": dict(Counter(r for x in suspects for r in x["reasons"])),
            "patients": sorted({x["patient"] for x in suspects}),
            "listed_in_gain_report": listed,
        },
        "problems": problems,
    }


# ============================================================================
# per-band noise sigma_b
# ============================================================================

def laplacian_residual(x: np.ndarray) -> np.ndarray:
    """(n, H, W, C) -> (n, H-2, W-2, C) float64: each interior pixel minus the
    mean of its 4 neighbours, inside each patch (never across patches)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 4 or x.shape[1] < 3 or x.shape[2] < 3:
        raise ValueError(f"expected (n, H, W, C) patches with H, W >= 3, got shape {x.shape}")
    nb = x[:, :-2, 1:-1] + x[:, 2:, 1:-1] + x[:, 1:-1, :-2] + x[:, 1:-1, 2:]
    return x[:, 1:-1, 1:-1] - 0.25 * nb


def noise_sigma_from_residuals(residuals: np.ndarray) -> Dict[str, np.ndarray]:
    """Per band (rows of `residuals`, shape (C, m)): `residual_robust_std` =
    1.4826 x MAD of the finite residuals, `sigma` = that / sqrt(1.25) (the
    per-pixel noise std), `n` = residuals used. NaN where a band has none."""
    res = np.asarray(residuals)
    C = res.shape[0]
    robust = np.full(C, np.nan)
    n = np.zeros(C, dtype=np.int64)
    for b in range(C):
        r = res[b].ravel()
        r = r[np.isfinite(r)].astype(np.float64)
        n[b] = r.size
        if r.size:
            med = np.median(r)
            robust[b] = MAD_TO_SIGMA * float(np.median(np.abs(r - med)))
    return {"residual_robust_std": robust, "sigma": robust / math.sqrt(LAPLACIAN_NOISE_VAR), "n": n}


def laplacian_noise_sigma(patches: np.ndarray) -> np.ndarray:
    """sigma_b (float64, [C]) of a stack of (n, H, W, C) patches. The stage
    uses the same arithmetic on its sample; this is the in-memory form."""
    r = laplacian_residual(patches)
    C = r.shape[-1]
    return noise_sigma_from_residuals(r.reshape(-1, C).T)["sigma"]


def patch_flatness(x: np.ndarray) -> np.ndarray:
    """Per patch: the band-averaged coefficient of variation over its pixels
    (inf where a band mean is 0). Low = flat, like glass."""
    x = np.asarray(x, dtype=np.float64)
    m = np.abs(x.mean(axis=(1, 2)))
    s = x.std(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(m > 0, s / np.where(m > 0, m, 1.0), np.inf)
    return np.nan_to_num(cv.mean(axis=1), nan=np.inf)


# ============================================================================
# running accumulators
# ============================================================================

class RowMoments:
    """Per band: the mean and co-moments, over ROWS, of u = the row's band
    mean and v = the row's band mean of x^2. Float64, merged across blocks with
    the Chan et al. update (as `io.RunningBandStats`). `standard_errors`
    turns them into the cluster SEs of a row sample (module docstring)."""

    def __init__(self, n_bands: int):
        C = int(n_bands)
        self.n_bands = C
        self.count = np.zeros(C)
        self.mean_u = np.zeros(C)
        self.mean_v = np.zeros(C)
        self.m_uu = np.zeros(C)
        self.m_vv = np.zeros(C)
        self.m_uv = np.zeros(C)

    def update(self, u: np.ndarray, v: np.ndarray, valid: Optional[np.ndarray] = None) -> "RowMoments":
        """u, v: (rows, C). `valid` (rows, C) excludes a row from a band (a
        non-finite pixel), as RunningBandStats excludes non-finite values."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            if valid is None:
                n_b = np.full(self.n_bands, float(u.shape[0]))
                mu_b, mv_b = u.mean(axis=0), v.mean(axis=0)
                du, dv = u - mu_b, v - mv_b
            else:
                n_b = valid.sum(axis=0).astype(np.float64)
                den = np.maximum(n_b, 1.0)
                mu_b = np.where(valid, u, 0.0).sum(axis=0) / den
                mv_b = np.where(valid, v, 0.0).sum(axis=0) / den
                du = np.where(valid, u - mu_b, 0.0)
                dv = np.where(valid, v - mv_b, 0.0)
            muu_b, mvv_b, muv_b = (du * du).sum(axis=0), (dv * dv).sum(axis=0), (du * dv).sum(axis=0)
            n_a = self.count
            n = n_a + n_b
            f = np.where(n > 0, n_b / np.where(n > 0, n, 1.0), 0.0)
            d_u, d_v = mu_b - self.mean_u, mv_b - self.mean_v
            self.mean_u = self.mean_u + d_u * f
            self.mean_v = self.mean_v + d_v * f
            self.m_uu = self.m_uu + muu_b + d_u * d_u * n_a * f
            self.m_vv = self.m_vv + mvv_b + d_v * d_v * n_a * f
            self.m_uv = self.m_uv + muv_b + d_u * d_v * n_a * f
        self.count = n
        return self

    def standard_errors(self, mean: np.ndarray, std: np.ndarray, n_sample: int,
                        n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
        """(SE of the mean, SE of the std) of `n_sample` rows drawn without
        replacement from `n_rows`, per band. 0 when the sample is every row,
        NaN when undefined (fewer than 2 rows, or std = 0 for the std)."""
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        n = float(n_sample)
        fpc = max(0.0, 1.0 - n / float(n_rows)) if n_rows > 0 else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            dof = np.where(self.count > 1, self.count - 1.0, np.nan)
            s2u, s2v, suv = self.m_uu / dof, self.m_vv / dof, self.m_uv / dof
            s2w = np.maximum(s2v - 4.0 * mean * suv + 4.0 * mean * mean * s2u, 0.0)
            se_mean = np.sqrt(s2u / n * fpc) if n > 0 else np.full(self.n_bands, np.nan)
            se_var = np.sqrt(s2w / n * fpc) if n > 0 else np.full(self.n_bands, np.nan)
            se_std = np.where(std > 0, se_var / (2.0 * np.where(std > 0, std, 1.0)), np.nan)
        return se_mean, se_std


class RunningCovariance:
    """Mean and co-moment matrix of feature vectors (pixel spectra), float64,
    Chan et al. merge. A vector with any non-finite entry is dropped and
    counted (`n_dropped`)."""

    def __init__(self, n_features: int):
        d = int(n_features)
        self.n_features = d
        self.count = 0.0
        self.mean = np.zeros(d)
        self.m2 = np.zeros((d, d))
        self.n_dropped = 0

    def update(self, z: np.ndarray) -> "RunningCovariance":
        z = np.asarray(z, dtype=np.float64).reshape(-1, self.n_features)
        ok = np.isfinite(z).all(axis=1)
        if not ok.all():
            self.n_dropped += int((~ok).sum())
            z = z[ok]
        n_b = float(z.shape[0])
        if n_b == 0:
            return self
        mean_b = z.mean(axis=0)
        d = z - mean_b
        n = self.count + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n)
        self.m2 = self.m2 + d.T @ d + np.outer(delta, delta) * (self.count * n_b / n)
        self.count = n
        return self

    def cov(self, ddof: int = 0) -> np.ndarray:
        if self.count <= ddof:
            return np.full((self.n_features, self.n_features), np.nan)
        return self.m2 / (self.count - ddof)

    def corr(self) -> np.ndarray:
        """Pearson correlation; NaN in the row/column of a zero-variance feature."""
        sd = np.sqrt(np.clip(np.diag(self.m2), 0.0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self.m2 / np.outer(sd, sd)
        r[~np.isfinite(r)] = np.nan
        r = np.clip(r, -1.0, 1.0)
        live = sd > 0
        r[np.diag_indices_from(r)] = np.where(live, 1.0, np.nan)
        return r


# ============================================================================
# train.py's sample estimate, replicated
# ============================================================================

def train_py_sample_indices(n_rows: int, sample_cap: int = TRAIN_PY_SAMPLE_CAP, seed: int = 0) -> np.ndarray:
    """The rows `training.npy_data.compute_global_channel_stats(x, sample_cap,
    seed)` reads from an array of `n_rows` rows: same generator, same draw."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=min(sample_cap, n_rows), replace=False)
    idx.sort()
    return idx


def train_py_stats_from_rows(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, std, unfloored std) exactly as `compute_global_channel_stats`
    computes them from its sampled rows: float32, nan_to_num, ddof 0, std
    floored to 1.0 at or below 1e-6. Bit-identical for the same rows."""
    sample = np.asarray(rows, dtype=np.float32)
    sample = np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0)
    mean = sample.mean(axis=(0, 1, 2))
    std = sample.std(axis=(0, 1, 2))
    floored = np.where(std > TRAIN_PY_STD_FLOOR, std, 1.0)
    return mean.astype(np.float32), floored.astype(np.float32), std.astype(np.float32)


def train_py_sample_stats(X: np.ndarray, sample_cap: int = TRAIN_PY_SAMPLE_CAP,
                          seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """`compute_global_channel_stats` on an already-open array (memmap)."""
    idx = train_py_sample_indices(int(X.shape[0]), sample_cap, seed)
    mean, std, _ = train_py_stats_from_rows(X[idx])
    return mean, std


def compare_sample_estimates(exact_mean: np.ndarray, exact_std: np.ndarray, se_mean: np.ndarray,
                             se_std: np.ndarray, estimates: Mapping[int, Tuple[np.ndarray, np.ndarray]],
                             *, n_sample: int, n_rows: int, z_tol: float,
                             skip_std_bands: Sequence[int] = ()) -> Dict[str, Any]:
    """Each seed's (mean, std) estimate against the exact stats, in SE units.
    `skip_std_bands` (zero-variance bands, whose estimate train.py floors to
    1.0) are left out of the std comparison; they are reported on their own."""
    exhaustive = n_sample >= n_rows
    skip = np.zeros(len(exact_mean), dtype=bool)
    skip[list(skip_std_bands)] = True
    per_seed: Dict[str, Any] = {}
    zs: List[np.ndarray] = []
    worst_exhaustive = 0.0
    for seed, (m_s, s_s) in estimates.items():
        dm = np.asarray(m_s, dtype=np.float64) - exact_mean
        ds = np.asarray(s_s, dtype=np.float64) - exact_std
        ds = np.where(skip, np.nan, ds)
        with np.errstate(invalid="ignore", divide="ignore"):
            zm = np.where(se_mean > 0, dm / np.where(se_mean > 0, se_mean, 1.0), np.nan)
            zsd = np.where(se_std > 0, ds / np.where(se_std > 0, se_std, 1.0), np.nan)
        entry = {"mean": m_s, "std": s_s, "diff_mean": dm, "diff_std": ds, "z_mean": zm, "z_std": zsd}
        if exhaustive:
            tol = EXHAUSTIVE_RTOL * (np.abs(exact_mean) + exact_std + 1e-12)
            rel = np.nanmax(np.concatenate([np.abs(dm) / tol, np.abs(ds) / tol]))
            worst_exhaustive = max(worst_exhaustive, float(rel))
            entry["within_float32_tolerance"] = bool(rel <= 1.0)
        z = np.concatenate([zm, zsd])
        z = z[np.isfinite(z)]
        zs.append(z)
        entry["max_abs_z"] = float(np.abs(z).max()) if z.size else None
        per_seed[str(int(seed))] = entry
    allz = np.concatenate(zs) if zs else np.zeros(0)
    if exhaustive:
        agree = worst_exhaustive <= 1.0
    else:
        agree = bool(allz.size) and bool(np.all(np.abs(allz) <= z_tol))
    return {
        "method": ("training.npy_data.compute_global_channel_stats, replicated bit for bit on the "
                   "same rows: np.random.default_rng(seed).choice(N, min(cap, N), replace=False), "
                   "sorted; float32; nan_to_num; ddof 0; std <= 1e-6 floored to 1.0"),
        "standard_error": ("cluster (row) sampling without replacement: SE(mean_b) = "
                           "sqrt(S_u^2/n (1-n/N)), u = per-row band mean; SE(std_b) = SE(var_b)/(2 std_b), "
                           "SE(var_b) = sqrt(S_w^2/n (1-n/N)), w = per-row mean of (x - mean_b)^2"),
        "sample_rows": int(min(n_sample, n_rows)),
        "population_rows": int(n_rows),
        "exhaustive": bool(exhaustive),
        "se_mean": se_mean, "se_std": se_std,
        "z_tol": float(z_tol),
        "seeds": per_seed,
        "n_z": int(allz.size),
        "max_abs_z": float(np.abs(allz).max()) if allz.size else None,
        "rms_z": float(np.sqrt(np.mean(allz ** 2))) if allz.size else None,
        "frac_within_2se": float(np.mean(np.abs(allz) <= 2.0)) if allz.size else None,
        "std_bands_skipped": [int(b) for b in np.flatnonzero(skip)],
        "agree": bool(agree),
    }


# ============================================================================
# the streaming pass
# ============================================================================

class _Pick:
    """Sorted unique row numbers, picked out of the contiguous chunks they fall in."""

    def __init__(self, rows: Any):
        self.rows = np.unique(np.asarray(rows, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.rows.size)

    def span(self, start: int, stop: int) -> Tuple[int, int]:
        """Positions [lo, hi) of the picked rows inside [start, stop)."""
        lo, hi = np.searchsorted(self.rows, [start, stop], side="left")
        return int(lo), int(hi)


COUNT_KINDS = ("saturated", "zero", "near_zero", "negative", "raw_near_ceiling")


class SpectralScan:
    """Everything the audit needs from the rows, accumulated over ONE
    sequential pass of contiguous chunks (`update(start, chunk, captures)`).
    See the module docstring for what each accumulator is for."""

    def __init__(self, *, n_bands: int, patch_hw: Tuple[int, int], x_dtype: Any,
                 corr_rows: Any = (), noise_rows: Any = (), replica_rows: Optional[Mapping[int, Any]] = None,
                 near_zero_eps: float = 1e-3, raw_threshold: Optional[float] = None,
                 value_scale: Optional[float] = None, gain_lut: Optional[np.ndarray] = None):
        C = int(n_bands)
        H, W = (int(s) for s in patch_hw)
        if H < 3 or W < 3:
            raise ValueError(f"patches must be at least 3x3 for the Laplacian residual, got {H}x{W}")
        self.n_bands, self.H, self.W = C, H, W
        self.row_values = H * W * C
        self.near_zero_eps = float(near_zero_eps)
        self.stats = RunningBandStats(C)
        self.moments = RowMoments(C)
        self.cov = RunningCovariance(C)
        self.counts = {k: np.zeros(C, dtype=np.int64) for k in COUNT_KINDS}
        self.raw_enabled = raw_threshold is not None and value_scale is not None and gain_lut is not None
        self.raw_threshold = raw_threshold
        self.value_scale = value_scale
        self.gain_lut = gain_lut
        self.raw_max = np.full(C, -np.inf)
        self.rows_with_gain = 0
        self.rows_without_gain = 0
        self.rows_seen = 0
        self._next_row: Optional[int] = None
        self._corr = _Pick(corr_rows)
        self._noise = _Pick(noise_rows)
        self.residuals = np.zeros((C, len(self._noise), (H - 2) * (W - 2)), dtype=np.float32)
        self.flatness = np.full(len(self._noise), np.inf)
        self._replica = {int(s): _Pick(r) for s, r in (replica_rows or {}).items()}
        self.replica_rows = {s: np.zeros((len(p), H, W, C), dtype=x_dtype) for s, p in self._replica.items()}

    @property
    def corr_rows(self) -> np.ndarray:
        return self._corr.rows

    @property
    def noise_rows(self) -> np.ndarray:
        return self._noise.rows

    @staticmethod
    def projected_bytes(*, chunk_rows: int, row_values: int, itemsize: int, n_noise: int,
                        residual_values: int, n_bands: int, replica_sizes: Sequence[int]) -> int:
        """Peak RAM of a pass: one chunk, the float64 block temporaries, the
        noise residuals, and the gathered sample-estimate rows plus their
        float32 working copies at the end."""
        block = 6 * hio.STATS_BLOCK_VALUES * 8
        noise = n_noise * residual_values * n_bands * 4 * 2
        replica = sum(replica_sizes) * row_values * itemsize + 3 * max(replica_sizes or [0]) * row_values * 4
        return chunk_rows * row_values * itemsize + block + noise + replica + 50_000_000

    def update(self, start: int, data: np.ndarray, captures: Optional[np.ndarray] = None) -> None:
        """Rows [start, start + len(data)) of the split, in order."""
        m = int(data.shape[0])
        if data.shape[1:] != (self.H, self.W, self.n_bands):
            raise ValueError(f"chunk rows have shape {data.shape[1:]}, expected {(self.H, self.W, self.n_bands)}")
        if self._next_row is not None and start != self._next_row:
            raise ValueError(f"chunks must be contiguous and in order: expected row {self._next_row}, got {start}")
        self._next_row = start + m
        stop = start + m
        self.stats.update(data)
        for seed, pick in self._replica.items():
            lo, hi = pick.span(start, stop)
            if hi > lo:
                self.replica_rows[seed][lo:hi] = data[pick.rows[lo:hi] - start]
        block = max(1, hio.STATS_BLOCK_VALUES // self.row_values)
        for a in range(0, m, block):
            b = np.asarray(data[a:a + block], dtype=np.float64)
            cap = None if captures is None else np.asarray(captures[a:a + block])
            self._block(start + a, b, cap)
        self.rows_seen += m

    def _block(self, g0: int, b: np.ndarray, cap: Optional[np.ndarray]) -> None:
        k = b.shape[0]
        axes = (0, 1, 2)
        finite = np.isfinite(b)
        all_finite = bool(finite.all())
        with np.errstate(invalid="ignore", over="ignore"):
            u = b.mean(axis=(1, 2))
            v = np.square(b).mean(axis=(1, 2))
        self.moments.update(u, v, None if all_finite else finite.all(axis=(1, 2)))

        absb = np.abs(b)
        sat = absb >= FLOAT16_MAX
        if not all_finite:
            sat &= finite
        self.counts["saturated"] += sat.sum(axis=axes)
        self.counts["zero"] += (b == 0).sum(axis=axes)
        self.counts["near_zero"] += (absb <= self.near_zero_eps).sum(axis=axes)
        self.counts["negative"] += (b < 0).sum(axis=axes)

        if self.raw_enabled and cap is not None:
            lut = self.gain_lut
            cap = cap.astype(np.int64)
            inside = (cap >= 0) & (cap < lut.size)
            gains = np.full(k, np.nan)
            gains[inside] = lut[cap[inside]]
            known = np.isfinite(gains) & (gains > 0)
            self.rows_with_gain += int(known.sum())
            self.rows_without_gain += int((~known).sum())
            if known.any():
                with np.errstate(invalid="ignore", over="ignore"):
                    raw = b[known] * (self.value_scale / gains[known])[:, None, None, None]
                rfin = np.isfinite(raw)
                self.raw_max = np.maximum(self.raw_max, np.where(rfin, raw, -np.inf).max(axis=axes))
                self.counts["raw_near_ceiling"] += ((raw >= self.raw_threshold) & rfin).sum(axis=axes)

        lo, hi = self._corr.span(g0, g0 + k)
        if hi > lo:
            self.cov.update(b[self._corr.rows[lo:hi] - g0].reshape(-1, self.n_bands))
        lo, hi = self._noise.span(g0, g0 + k)
        if hi > lo:
            sel = b[self._noise.rows[lo:hi] - g0]
            r = laplacian_residual(sel)
            self.residuals[:, lo:hi, :] = r.reshape(hi - lo, -1, self.n_bands).transpose(2, 0, 1)
            self.flatness[lo:hi] = patch_flatness(sel)

    # -- results ------------------------------------------------------------------
    def noise(self, flat_fraction: float) -> Dict[str, Any]:
        """sigma_b over the whole noise sample, and over its flattest
        `flat_fraction` of patches."""
        C = self.n_bands
        full = noise_sigma_from_residuals(self.residuals.reshape(C, -1))
        n_rows = len(self._noise)
        k = max(1, int(round(flat_fraction * n_rows))) if n_rows else 0
        flat_rows = np.argsort(self.flatness, kind="stable")[:k]
        flat = noise_sigma_from_residuals(self.residuals[:, flat_rows, :].reshape(C, -1)) if k else \
            {"sigma": np.full(C, np.nan), "residual_robust_std": np.full(C, np.nan), "n": np.zeros(C, int)}
        return {"sigma": full["sigma"], "residual_robust_std": full["residual_robust_std"],
                "n_residuals": full["n"], "sigma_flat": flat["sigma"], "n_flat_rows": int(k),
                "flatness_cut": float(self.flatness[flat_rows].max()) if k else None}

    def replica_estimates(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        return {s: train_py_stats_from_rows(rows)[:2] for s, rows in self.replica_rows.items()}


# ============================================================================
# reports
# ============================================================================

def band_problems(per_band: Mapping[str, Any], *, min_std: float,
                  corr_pairs: Sequence[Mapping[str, Any]], corr_flag: float) -> List[Dict[str, Any]]:
    """Every band-level problem, grouped by kind (each lists its bands)."""
    out: List[Dict[str, Any]] = []

    def bands_where(mask: np.ndarray) -> List[int]:
        return [int(b) for b in np.flatnonzero(mask)]

    nonfinite = np.asarray(per_band["n_nonfinite"])
    if nonfinite.any():
        bands = bands_where(nonfinite > 0)
        out.append(problem("error", "nonfinite",
                           f"{int(nonfinite.sum())} non-finite value(s) in band(s) {bands}; excluded "
                           f"from the exact stats, replaced by 0/1 in train.py's sample estimate",
                           bands=bands, counts=[int(nonfinite[b]) for b in bands]))
    std = np.asarray(per_band["std"], dtype=np.float64)
    flat = ~(std > min_std)
    if flat.any():
        bands = bands_where(flat)
        out.append(problem("error", "zero_variance",
                           f"band(s) {bands} have std <= {min_std:g} over the train pixels; "
                           f"compute_global_channel_stats replaces their std by 1.0",
                           bands=bands))
    labels = {
        "saturated": ("warning", "float16_saturated", f"|x| == {FLOAT16_MAX:g} (the float16 maximum)"),
        "raw_near_ceiling": ("warning", "raw_near_ceiling",
                             "implied raw count (value x value scale / capture gain) at or above the "
                             "near-saturation threshold"),
        "zero": ("warning", "exact_zero", "x == 0"),
        "near_zero": ("warning", "near_zero", "|x| at or below near_zero_eps"),
        "negative": ("warning", "negative", "x < 0 (counts cannot be negative)"),
    }
    for key, (sev, kind, what) in labels.items():
        counts = np.asarray(per_band[f"{key}_count"])
        if counts.any():
            bands = bands_where(counts > 0)
            out.append(problem(sev, kind, f"{what}: {int(counts.sum())} value(s) in band(s) {bands}",
                               bands=bands, counts=[int(counts[b]) for b in bands],
                               fractions=[float(per_band[f'{key}_fraction'][b]) for b in bands]))
    sigma = np.asarray(per_band["noise_sigma"], dtype=np.float64)
    undefined = ~np.isfinite(sigma)
    if undefined.any():
        bands = bands_where(undefined)
        out.append(problem("warning", "noise_sigma_undefined",
                           f"no finite Laplacian residual in band(s) {bands}; noise_sigma.npy holds 0 "
                           f"there (no noise is added)", bands=bands))
    zero_sigma = np.isfinite(sigma) & (sigma == 0) & ~flat
    if zero_sigma.any():
        bands = bands_where(zero_sigma)
        out.append(problem("warning", "noise_sigma_zero",
                           f"sigma_b = 0 in varying band(s) {bands}: over half the residuals are "
                           f"identical (coarse quantisation?)", bands=bands))
    if corr_pairs:
        out.append(problem("info", "high_correlation",
                           f"{len(corr_pairs)} band pair(s) with |r| > {corr_flag:g} on the pixel sample",
                           pairs=[[p["a"], p["b"]] for p in corr_pairs]))
    return out


def correlation_report(cov: RunningCovariance, wavelengths: Optional[np.ndarray], *, n_rows: int,
                       corr_flag: float,
                       gaps: Sequence[int]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    r = cov.corr()
    C = r.shape[0]
    off = r[~np.eye(C, dtype=bool)]
    off = off[np.isfinite(off)]
    pairs = []
    for a in range(C):
        for b in range(a + 1, C):
            if np.isfinite(r[a, b]) and abs(r[a, b]) > corr_flag:
                pairs.append({"a": a, "b": b, "r": float(r[a, b]),
                              "wavelength_nm": None if wavelengths is None else
                              [float(wavelengths[a]), float(wavelengths[b])]})
    adjacent = [float(r[b, b + 1]) for b in range(C - 1)]
    rep = {
        "level": "pixel: every pixel spectrum of the sampled rows",
        "sample_rows": int(n_rows),
        "pixels": int(cov.count),
        "pixels_dropped_nonfinite": int(cov.n_dropped),
        "seed_key": "correlation",
        "matrix": r,
        "mean": cov.mean,
        "std": np.sqrt(np.clip(np.diag(cov.cov()), 0.0, None)) if cov.count else np.full(C, np.nan),
        "max_offdiag_abs": float(np.abs(off).max()) if off.size else None,
        "min_offdiag": float(off.min()) if off.size else None,
        "adjacent": adjacent,
        "across_gaps": [{"after_band": int(g), "r": adjacent[g]} for g in gaps if 0 <= g < C - 1],
        "flag_threshold": float(corr_flag),
        "pairs_above": pairs,
    }
    return rep, pairs


def run_spectral_audit(ctx: Any, cfg: Any) -> Dict[str, Any]:
    """The `spectral-audit` stage body (see `stages/spectral_audit.py` for
    the config). Writes the four outputs and returns the done.json summary."""
    from hsi_augment.stage import StageError

    adapter = ctx.adapter

    # 1. calibration audit: metadata only, no split is opened ------------------
    with ctx.timer("calibration"):
        calib = calibration_audit(adapter, outlier_z=cfg.gain_outlier_z)
    s = calib["suspect_summary"]
    ctx.log(f"calibration: {sum(calib['n_captures'].values())} captures, gain median by split "
            f"{ {k: round(v.get('median', float('nan')), 4) for k, v in calib['gain']['by_split'].items()} }, "
            f"{s['n']} suspect capture(s) {s['by_split']} of patient(s) {s['patients']}")

    # 2. the train split, through the stage's SplitGuard -------------------------
    train = ctx.open_split("train", purpose="fit", kinds=("X", "captures"))
    X = train.X
    if X is None or X.ndim != 4:
        raise StageError(f"X_train must be (N, H, W, C); got {None if X is None else X.shape}")
    N = int(X.shape[0])
    if N == 0:
        raise StageError("X_train has no rows")
    H, W, C = (int(v) for v in X.shape[1:])
    if H < 3 or W < 3:
        raise StageError(f"patches are {H}x{W}; the Laplacian noise estimate needs at least 3x3")
    M = N if cfg.max_rows is None else min(int(cfg.max_rows), N)
    complete = M == N
    if not complete:
        ctx.warn(f"max_rows={cfg.max_rows}: scanning the first {M} of {N} train rows; the stats are NOT "
                 f"over all train rows (smoke test only)")

    wl: Optional[np.ndarray]
    try:
        wl = np.asarray(adapter.wavelengths, dtype=np.float64)
    except FileNotFoundError:
        wl = None
    meta_problems: List[Dict[str, Any]] = []
    if wl is None:
        meta_problems.append(problem("warning", "no_wavelengths", "wavelengths.npy is missing"))
    elif wl.shape != (C,):
        meta_problems.append(problem("error", "wavelength_count",
                                     f"wavelengths.npy has shape {wl.shape}, X_train has {C} bands"))
        wl = None
    gaps = adapter.spectral_gaps() if wl is not None else []

    corr_rows = sample_rows(ctx.rng("correlation"), M, cfg.corr_sample_rows)
    noise_rows = sample_rows(ctx.rng("noise"), M, cfg.noise_sample_rows)
    replica_rows = {int(seed): train_py_sample_indices(M, cfg.compare_sample_cap, int(seed))
                    for seed in cfg.compare_seeds}

    lut = gain_lookup(adapter)
    scale = value_scale(adapter)
    raw_threshold = cfg.raw_ceiling * cfg.raw_ceiling_frac
    if train.captures is None or scale is None or not lut.size:
        meta_problems.append(problem("warning", "raw_check_unavailable",
                                     "no captures sidecar, value scale or capture gains: the implied "
                                     "raw-count ceiling check is skipped"))

    ctx.ram_guard(SpectralScan.projected_bytes(
        chunk_rows=min(cfg.chunk_rows, M), row_values=H * W * C, itemsize=X.dtype.itemsize,
        n_noise=len(noise_rows), residual_values=(H - 2) * (W - 2), n_bands=C,
        replica_sizes=[len(r) for r in replica_rows.values()]),
        "one chunk + float64 blocks + noise residuals + sample-estimate rows")

    scan = SpectralScan(n_bands=C, patch_hw=(H, W), x_dtype=X.dtype, corr_rows=corr_rows,
                        noise_rows=noise_rows, replica_rows=replica_rows,
                        near_zero_eps=cfg.near_zero_eps, raw_threshold=raw_threshold,
                        value_scale=scale, gain_lut=lut if lut.size else None)
    index = None if complete else np.arange(M, dtype=np.int64)
    ctx.log(f"scan: {M} train rows of {H}x{W}x{C} {X.dtype} in chunks of {cfg.chunk_rows}; samples: "
            f"correlation {len(corr_rows)} rows, noise {len(noise_rows)} rows, train.py estimate "
            f"{cfg.compare_sample_cap} rows x seeds {list(cfg.compare_seeds)}")
    with ctx.timer("scan"):
        for i, ch in enumerate(iter_chunks(X, index, chunk_rows=cfg.chunk_rows)):
            a, b = int(ch.rows[0]), int(ch.rows[-1]) + 1
            cap = None if train.captures is None else np.asarray(train.captures[a:b])
            scan.update(a, ch.data, cap)
            ctx.log_metrics(chunk=i, rows_done=b, rows_total=M)

    with ctx.timer("finalize"):
        stats = scan.stats
        mean, std, var = stats.mean, stats.std(), stats.var()
        n_vals = M * H * W
        noise = scan.noise(cfg.noise_flat_fraction)
        sigma = noise["sigma"]
        step = np.array([float(np.spacing(X.dtype.type(abs(m)))) if np.isfinite(m) else np.nan
                         for m in mean])
        raw_den = scan.rows_with_gain * H * W
        with np.errstate(invalid="ignore", divide="ignore"):
            per_band: Dict[str, Any] = {
                "band": list(range(C)),
                "wavelength_nm": None if wl is None else wl,
                "mean": mean, "std": std, "var": var, "min": stats.min, "max": stats.max,
                "n_values": stats.count.astype(np.int64), "n_nonfinite": stats.n_nonfinite,
                "variance_positive": std > cfg.min_std,
            }
            for key in COUNT_KINDS:
                den = raw_den if key == "raw_near_ceiling" else n_vals
                per_band[f"{key}_count"] = scan.counts[key]
                per_band[f"{key}_fraction"] = scan.counts[key] / den if den else np.full(C, np.nan)
            per_band.update({
                "raw_max_counts": np.where(np.isfinite(scan.raw_max), scan.raw_max, np.nan),
                "noise_sigma": sigma,
                "residual_robust_std": noise["residual_robust_std"],
                "noise_sigma_flat": noise["sigma_flat"],
                "noise_to_std": sigma / std,
                "storage_step_at_mean": step,
                "storage_quantization_sigma": step / math.sqrt(12.0),
            })
        corr, pairs = correlation_report(scan.cov, wl, n_rows=len(corr_rows), corr_flag=cfg.corr_flag,
                                         gaps=gaps)
        problems = meta_problems + band_problems(per_band, min_std=cfg.min_std,
                                                 corr_pairs=pairs, corr_flag=cfg.corr_flag)

        zero_var = [int(b) for b in np.flatnonzero(~(std > cfg.min_std))]
        se_mean, se_std = scan.moments.standard_errors(mean, std, min(cfg.compare_sample_cap, M), M)
        comparison = compare_sample_estimates(mean, std, se_mean, se_std, scan.replica_estimates(),
                                              n_sample=min(cfg.compare_sample_cap, M), n_rows=M,
                                              z_tol=cfg.z_tol, skip_std_bands=zero_var)
        comparison["sample_cap"] = int(cfg.compare_sample_cap)
        comparison["replicates_train_py"] = bool(complete)
        if not comparison["agree"]:
            problems.append(problem("error", "sample_estimate_disagrees",
                                    f"train.py's sample estimate is off by max |z| = "
                                    f"{comparison['max_abs_z']} (> {cfg.z_tol:g}) standard errors"))
        if not complete:
            problems.append(problem("warning", "partial_scan",
                                    f"only the first {M} of {N} train rows were scanned (max_rows); "
                                    f"nothing here is a whole-train statistic"))

    units = units_note(adapter)
    common = {"schema_version": SCHEMA_VERSION, "split": "train", "purpose": "fit",
              "rows_total": N, "rows_scanned": M, "complete": complete,
              "patch_shape": [H, W], "pixels_per_row": H * W, "n_bands": C,
              "storage_dtype": str(X.dtype), "units": units}

    sigma_out = np.where(np.isfinite(sigma), sigma, 0.0).astype(np.float32)
    ctx.save_npy(NOISE_SIGMA_NAME, sigma_out)
    ctx.save_json(NORM_STATS_NAME, json_safe({
        **common,
        "kind": "train_linear_band_stats",
        "statistic": "exact per-band mean/std over every pixel of every scanned train row; float64; ddof 0",
        "wavelength_nm": None if wl is None else wl,
        "mean": mean, "std": std, "var": var, "min": stats.min, "max": stats.max,
        "count": stats.count, "n_nonfinite": stats.n_nonfinite,
        "running_stats": stats.to_dict(),
        "sample_estimate": comparison,
        "notes": ["log-space generator statistics are fitted on train tissue rows in #16, not here",
                  "train.py does not read this file; it re-estimates from 5,000 rows "
                  "(training.npy_data.compute_global_channel_stats); `sample_estimate` measures that gap"],
    }))
    ctx.save_json(BAND_REPORT_NAME, json_safe({
        **common,
        "wavelength_nm": None if wl is None else wl,
        "spectral_gaps_after_band": gaps,
        "spectral_segments": [list(sg) for sg in adapter.spectral_segments()] if wl is not None else None,
        "thresholds": {"min_std": cfg.min_std, "float16_max": FLOAT16_MAX, "near_zero_eps": cfg.near_zero_eps,
                       "raw_ceiling": cfg.raw_ceiling, "raw_ceiling_frac": cfg.raw_ceiling_frac,
                       "raw_threshold_counts": raw_threshold, "corr_flag": cfg.corr_flag},
        "raw_check": {"enabled": scan.raw_enabled and train.captures is not None,
                      "value_scale": scale, "rows_with_gain": scan.rows_with_gain,
                      "rows_without_gain": scan.rows_without_gain},
        "per_band": per_band,
        "correlation": corr,
        "noise": {
            "method": "1.4826 x MAD of the 4-neighbour Laplacian residual at the interior pixels of each "
                      "patch, / sqrt(1.25) = per-pixel noise std (value units); written to noise_sigma.npy",
            "sample_rows": int(len(noise_rows)), "seed_key": "noise",
            "residuals_per_band": noise["n_residuals"],
            "flat_fraction": cfg.noise_flat_fraction, "flat_rows": noise["n_flat_rows"],
            "flatness_cut": noise["flatness_cut"],
            "caveat": "sub-pixel texture also leaves a residual: sigma_b is an upper bound on sensor noise; "
                      "noise_sigma_flat is the same estimate on the flattest patches",
        },
        "problems": problems,
        "problem_counts": count_by_severity(problems),
    }))
    ctx.save_json(CALIBRATION_REPORT_NAME, json_safe(calib))

    for p in calib["problems"] + problems:
        (ctx.warn if p["severity"] in ("error", "warning") else ctx.log)(
            f"{p['severity']}: {p['kind']}: {p['detail']}")
    ctx.log(f"norm stats over {M} rows: mean {np.round(mean, 4).tolist()}")
    ctx.log(f"noise sigma_b median {float(np.nanmedian(sigma)):.5g} (flat patches "
            f"{float(np.nanmedian(noise['sigma_flat'])):.5g}); train.py estimate vs exact: max |z| "
            f"{comparison['max_abs_z']}, agree={comparison['agree']}")
    return {
        "rows_scanned": M, "rows_total": N, "complete": complete, "n_bands": C,
        "band_problems": count_by_severity(problems),
        "calibration_problems": count_by_severity(calib["problems"]),
        "suspect_captures": calib["suspect_summary"]["n"],
        "zero_variance_bands": zero_var,
        "sample_estimate_agrees": comparison["agree"],
        "sample_estimate_max_abs_z": comparison["max_abs_z"],
        "noise_sigma_median": float(np.nanmedian(sigma)) if np.isfinite(sigma).any() else None,
    }
