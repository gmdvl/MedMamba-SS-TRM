# -*- coding: utf-8 -*-
"""
hsi_augment/qc/setmetrics.py
============================
Set-level QC metrics (Plan §Phase 8b, issue #28): how a SET of generated
samples of class c compares with the real train patches of class c.

Per-sample rules (rules.py) cannot see a generator that produces only
plausible samples of one patient, or only the centre of the distribution.
These metrics can. Each is reported with a bootstrap CI and, next to it, its
REAL CROSS-PATIENT RANGE: the same metric between one real target patient's
patches and the other target patients' patches, for every target patient
(computed once by the qc-reference stage, `real_reference`). A synthetic
value inside that range means "as different from the real data as one real
patient is from the others": the scale a bare number lacks.

    wasserstein1_band_mean   per-band 1-D Wasserstein-1 between the patch mean
                             values, divided by the real band's std, averaged over
                             bands (the per-band raw values are reported too)
    band_corr_frobenius      || corr(syn) - corr(real) ||_F of the band-band
                             correlation of patch mean spectra
    class_mean_sam_deg       unclamped SAM between the two sets' mean spectra
    frechet_<space>          Frechet distance of Gaussians fitted to the features
    mmd_<space>              unbiased RBF-MMD^2, bandwidth = median real pairwise
                             distance (fixed by the stage, so values compare)
    precision/recall/density/coverage_<space>   Naeem et al. 2020, k = 5
    nn_attribution           histogram of the real patient of each sample's nearest
                             train patch (R6's search); flagged when one patient
                             takes more than 40%
    a0                       optional, REPORTING ONLY: Frechet distance in the A0
                             classifier's penultimate features and its predicted
                             class mix (features from training/latent_space_v18.
                             extract_penultimate_features, passed in by the caller).
                             QC never gates on the classifier being evaluated.

<space> is `mean_std` (standardised mean/std features) or `pca` (PCA-64).

Bootstrap: Frechet, MMD, Wasserstein, correlation and SAM resample the
synthetic set; precision and density are means of per-SYNTHETIC-sample
indicators and resample those; recall and coverage are means of per-REAL-
sample indicators and resample those. 95% percentile intervals. RAM is
bounded by subsampling (`prdc_rows`, `mmd_rows`, seeded) and by computing
distances in blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hsi_augment.qc.features import sam_deg
from hsi_augment.qc.memorization import DEFAULT_BLOCK_ELEMS, knn_search, resolve_device

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.qc.reference import QCReference

SPACES = ("mean_std", "pca")
PRDC_NAMES = ("precision", "recall", "density", "coverage")
CI_LEVEL = 0.95
SET_REFERENCE_SCHEMA_VERSION = 1


# ============================================================================
# point metrics
# ============================================================================

def wasserstein1_per_band(syn: np.ndarray, real: np.ndarray) -> np.ndarray:
    from scipy.stats import wasserstein_distance
    s, r = np.asarray(syn, dtype=np.float64), np.asarray(real, dtype=np.float64)
    return np.array([wasserstein_distance(s[:, j], r[:, j]) for j in range(s.shape[1])])


def band_corr(x: np.ndarray) -> np.ndarray:
    """Band-band Pearson correlation of rows; a constant band correlates 0."""
    x = np.asarray(x, dtype=np.float64)
    d = x - x.mean(axis=0)
    s = np.sqrt((d * d).sum(axis=0))
    s = np.where(s > 0, s, np.inf)
    return (d.T @ d) / np.outer(s, s)


def corr_frobenius(syn: np.ndarray, real: np.ndarray) -> float:
    return float(np.linalg.norm(band_corr(syn) - band_corr(real)))


def _psd_sqrt(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (a + a.T))
    return (v * np.sqrt(np.maximum(w, 0.0))) @ v.T


def frechet_from_moments(m1: np.ndarray, c1: np.ndarray, m2: np.ndarray, c2: np.ndarray) -> float:
    """||m1 - m2||^2 + tr(C1 + C2 - 2 (C1 C2)^1/2), via the symmetric form
    sqrt(C1) C2 sqrt(C1), which stays real and PSD (scipy's sqrtm of C1 C2 can
    return complex noise)."""
    s1 = _psd_sqrt(c1)
    mid = s1 @ c2 @ s1
    tr_sqrt = float(np.sqrt(np.maximum(np.linalg.eigvalsh(0.5 * (mid + mid.T)), 0.0)).sum())
    d = np.asarray(m1) - np.asarray(m2)
    return float(max(d @ d + np.trace(c1) + np.trace(c2) - 2.0 * tr_sqrt, 0.0))


def moments(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=0), (np.cov(x, rowvar=False) if len(x) > 1 else np.zeros((x.shape[1], x.shape[1])))


def frechet(syn: np.ndarray, real: np.ndarray) -> float:
    return frechet_from_moments(*moments(syn), *moments(real))


def _sqdist_blocks(a: np.ndarray, b: np.ndarray, *, device: str,
                   block_elems: int = DEFAULT_BLOCK_ELEMS) -> Iterator[Tuple[int, int, np.ndarray]]:
    """(row start, row stop, squared distances float64 [rows, len(b)]) in blocks."""
    import torch
    at = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(device)
    bt = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float64)).to(device)
    step = max(1, block_elems // max(len(b), 1))
    for s in range(0, len(a), step):
        e = min(s + step, len(a))
        yield s, e, (torch.cdist(at[s:e], bt, compute_mode="donot_use_mm_for_euclid_dist") ** 2).cpu().numpy()


def rbf_kernel(a: np.ndarray, b: np.ndarray, sigma: float, *, device: str) -> np.ndarray:
    out = np.empty((len(a), len(b)))
    for s, e, d2 in _sqdist_blocks(a, b, device=device):
        out[s:e] = np.exp(-d2 / (2.0 * sigma * sigma))
    return out


def median_bandwidth(x: np.ndarray, rng: np.random.Generator, max_rows: int = 2000, *, device: str = "cpu") -> float:
    """The median pairwise distance of (a sample of) `x`: the RBF bandwidth."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) > max_rows:
        x = x[np.sort(rng.choice(len(x), max_rows, replace=False))]
    vals = []
    for s, e, d2 in _sqdist_blocks(x, x, device=device):
        iu = np.arange(s, e)[:, None] < np.arange(len(x))[None, :]
        vals.append(np.sqrt(d2[iu]))
    v = np.concatenate(vals) if vals else np.zeros(0)
    med = float(np.median(v)) if v.size else 1.0
    return med if med > 0 else 1.0


def mmd2_from_kernels(kss: np.ndarray, ksr: np.ndarray, krr: np.ndarray) -> float:
    """Unbiased MMD^2 from the three kernel blocks (diagonals excluded within a set)."""
    m, n = kss.shape[0], krr.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    xx = (kss.sum() - np.trace(kss)) / (m * (m - 1))
    yy = (krr.sum() - np.trace(krr)) / (n * (n - 1))
    return float(xx + yy - 2.0 * ksr.mean())


def mmd2(syn: np.ndarray, real: np.ndarray, sigma: float, *, device: str = "cpu") -> float:
    return mmd2_from_kernels(rbf_kernel(syn, syn, sigma, device=device), rbf_kernel(syn, real, sigma, device=device),
                             rbf_kernel(real, real, sigma, device=device))


def kth_nn_radius(x: np.ndarray, k: int, *, device: str, block_elems: int = DEFAULT_BLOCK_ELEMS) -> np.ndarray:
    """Distance from each row to its k-th nearest OTHER row."""
    ids = np.arange(len(x))
    d, _ = knn_search(x, x, k, device=device, block_elems=block_elems, query_groups=ids, bank_groups=ids)
    return d[:, k - 1].astype(np.float64)


def prdc_indicators(real: np.ndarray, syn: np.ndarray, k: int = 5, *, device: str = "cpu",
                    block_elems: int = DEFAULT_BLOCK_ELEMS) -> Dict[str, np.ndarray]:
    """Per-sample indicators of Naeem et al. (2020), whose means are the metrics:
    precision_j / density_j per synthetic sample, recall_i / coverage_i per real
    sample. Real (synthetic) k-NN balls exclude the centre itself."""
    real = np.asarray(real, dtype=np.float64)
    syn = np.asarray(syn, dtype=np.float64)
    r_rad = kth_nn_radius(real, k, device=device, block_elems=block_elems)
    s_rad = kth_nn_radius(syn, k, device=device, block_elems=block_elems)
    precision = np.zeros(len(syn), dtype=bool)
    density = np.zeros(len(syn))
    coverage = np.zeros(len(real), dtype=bool)
    recall = np.zeros(len(real), dtype=bool)
    for s, e, d2 in _sqdist_blocks(syn, real, device=device, block_elems=block_elems):
        d = np.sqrt(d2)                                          # [syn rows, real]
        inside_real = d <= r_rad[None, :]
        precision[s:e] = inside_real.any(axis=1)
        density[s:e] = inside_real.sum(axis=1) / k
        coverage |= inside_real.any(axis=0)
        recall |= (d <= s_rad[s:e, None]).any(axis=0)
    return {"precision": precision.astype(np.float64), "density": density,
            "recall": recall.astype(np.float64), "coverage": coverage.astype(np.float64)}


# ============================================================================
# feature bundles
# ============================================================================

def _subsample(n: int, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if cap is None or n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, int(cap), replace=False))


def point_metrics(syn: Mapping[str, np.ndarray], real: Mapping[str, np.ndarray], *, sigma: Mapping[str, float],
                  k: int, prdc_rows: int, mmd_rows: int, rng: np.random.Generator,
                  device: str) -> Dict[str, Any]:
    """Every set metric (no CI) between two feature bundles {mean, mean_std, pca}."""
    out: Dict[str, Any] = {}
    real_sd = np.asarray(real["mean"]).std(axis=0)
    w = wasserstein1_per_band(syn["mean"], real["mean"])
    out["wasserstein1_band_mean"] = float(np.mean(w / np.where(real_sd > 0, real_sd, 1.0)))
    out["wasserstein1_per_band"] = w.tolist()
    out["band_corr_frobenius"] = corr_frobenius(syn["mean"], real["mean"])
    out["class_mean_sam_deg"] = float(sam_deg(np.mean(syn["mean"], axis=0), np.mean(real["mean"], axis=0)))
    for sp in SPACES:
        out[f"frechet_{sp}"] = frechet(syn[sp], real[sp])
        si, ri = _subsample(len(syn[sp]), mmd_rows, rng), _subsample(len(real[sp]), mmd_rows, rng)
        out[f"mmd_{sp}"] = mmd2(syn[sp][si], real[sp][ri], sigma[sp], device=device)
        if min(len(syn[sp]), len(real[sp])) > k:
            si, ri = _subsample(len(syn[sp]), prdc_rows, rng), _subsample(len(real[sp]), prdc_rows, rng)
            ind = prdc_indicators(real[sp][ri], syn[sp][si], k, device=device)
            out.update({f"{m}_{sp}": float(ind[m].mean()) for m in PRDC_NAMES})
        else:
            out.update({f"{m}_{sp}": float("nan") for m in PRDC_NAMES})
    return out


def _ci(values: Sequence[float]) -> List[Optional[float]]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return [None, None]
    a = (1.0 - CI_LEVEL) / 2.0
    return [float(np.quantile(v, a)), float(np.quantile(v, 1.0 - a))]


def bootstrap_cis(syn: Mapping[str, np.ndarray], real: Mapping[str, np.ndarray], *, sigma: Mapping[str, float],
                  k: int, prdc_rows: int, mmd_rows: int, n_boot: int, rng: np.random.Generator,
                  device: str) -> Dict[str, List[Optional[float]]]:
    """95% percentile intervals of every scalar metric of `point_metrics`."""
    n_syn = len(syn["mean"])
    draws: Dict[str, List[float]] = {}

    def add(name: str, v: float) -> None:
        draws.setdefault(name, []).append(float(v))

    real_mean = np.asarray(real["mean"], dtype=np.float64)
    real_sd = real_mean.std(axis=0)
    real_sd = np.where(real_sd > 0, real_sd, 1.0)
    real_mom = {sp: moments(real[sp]) for sp in SPACES}
    real_mu = real_mean.mean(axis=0)
    for _ in range(n_boot):
        b = rng.integers(0, n_syn, n_syn)
        sm = np.asarray(syn["mean"])[b]
        add("wasserstein1_band_mean", float(np.mean(wasserstein1_per_band(sm, real_mean) / real_sd)))
        add("band_corr_frobenius", corr_frobenius(sm, real_mean))
        add("class_mean_sam_deg", float(sam_deg(sm.mean(axis=0), real_mu)))
        for sp in SPACES:
            add(f"frechet_{sp}", frechet_from_moments(*moments(np.asarray(syn[sp])[b]), *real_mom[sp]))
    for sp in SPACES:
        si, ri = _subsample(len(syn[sp]), mmd_rows, rng), _subsample(len(real[sp]), mmd_rows, rng)
        s_x, r_x = np.asarray(syn[sp])[si], np.asarray(real[sp])[ri]
        kss = rbf_kernel(s_x, s_x, sigma[sp], device=device)
        ksr = rbf_kernel(s_x, r_x, sigma[sp], device=device)
        krr = rbf_kernel(r_x, r_x, sigma[sp], device=device)
        for _ in range(n_boot):
            b = rng.integers(0, len(si), len(si))
            kb = kss[np.ix_(b, b)]
            same = b[:, None] == b[None, :]          # a resampled duplicate is not a distinct pair
            m = len(b)
            xx = (kb.sum() - kb[same].sum()) / max(m * m - int(same.sum()), 1)
            yy = (krr.sum() - np.trace(krr)) / max(len(ri) * (len(ri) - 1), 1)
            add(f"mmd_{sp}", float(xx + yy - 2.0 * ksr[b].mean()))
        if min(len(syn[sp]), len(real[sp])) > k:
            si, ri = _subsample(len(syn[sp]), prdc_rows, rng), _subsample(len(real[sp]), prdc_rows, rng)
            ind = prdc_indicators(np.asarray(real[sp])[ri], np.asarray(syn[sp])[si], k, device=device)
            for name in PRDC_NAMES:
                v = ind[name]
                for _ in range(n_boot):
                    add(f"{name}_{sp}", float(v[rng.integers(0, len(v), len(v))].mean()))
    return {name: _ci(v) for name, v in draws.items()}


# ============================================================================
# the real cross-patient reference (stage time)
# ============================================================================

def patient_bundles(features: Mapping[str, np.ndarray], patients: np.ndarray) -> Dict[int, np.ndarray]:
    """{patient: row indices} of a feature bundle."""
    p = np.asarray(patients, dtype=np.int64)
    return {int(v): np.flatnonzero(p == v) for v in np.unique(p)}


def take(bundle: Mapping[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v)[idx] for k, v in bundle.items()}


def real_reference(real: Mapping[str, np.ndarray], patients: np.ndarray, *, k: int, prdc_rows: int, mmd_rows: int,
                   rng: np.random.Generator, device: str = "cpu",
                   heldout_attribution: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """For every target patient p: each metric between p's patches and the
    other target patients' patches. Also fixes the MMD bandwidths."""
    sigma = {sp: median_bandwidth(real[sp], rng, device=device) for sp in SPACES}
    per_patient: Dict[str, Dict[str, Any]] = {}
    groups = patient_bundles(real, patients)
    for p, idx in groups.items():
        rest = np.setdiff1d(np.arange(len(patients)), idx)
        if len(rest) < 2 or len(idx) < 2:
            continue
        vals = point_metrics(take(real, idx), take(real, rest), sigma=sigma, k=k, prdc_rows=prdc_rows,
                             mmd_rows=mmd_rows, rng=rng, device=device)
        vals.pop("wasserstein1_per_band", None)
        per_patient[str(p)] = vals
    names = sorted({m for v in per_patient.values() for m in v})
    ranges = {}
    for m in names:
        v = {p: d[m] for p, d in per_patient.items() if d.get(m) is not None and np.isfinite(d[m])}
        arr = np.asarray(list(v.values()), dtype=np.float64)
        ranges[m] = {"min": float(arr.min()) if arr.size else None,
                     "median": float(np.median(arr)) if arr.size else None,
                     "max": float(arr.max()) if arr.size else None,
                     "n_patients": int(arr.size), "per_patient": v}
    counts = {str(p): int(len(idx)) for p, idx in groups.items()}
    total = max(sum(counts.values()), 1)
    return {"schema_version": SET_REFERENCE_SCHEMA_VERSION, "sigma": sigma, "k": int(k),
            "prdc_rows": int(prdc_rows), "mmd_rows": int(mmd_rows), "ranges": ranges,
            "bank_patient_share": {p: c / total for p, c in counts.items()},
            "heldout_attribution": dict(heldout_attribution or {})}


def attribution(nn_patient: np.ndarray, flag_share: float) -> Dict[str, Any]:
    p = np.asarray(nn_patient, dtype=np.int64)
    p = p[p >= 0]
    if p.size == 0:
        return {"n": 0, "hist": {}, "max_share": None, "max_patient": None, "flagged": False}
    vals, cnt = np.unique(p, return_counts=True)
    share = cnt / cnt.sum()
    i = int(np.argmax(share))
    return {"n": int(p.size), "hist": {str(int(v)): float(s) for v, s in zip(vals, share)},
            "max_share": float(share[i]), "max_patient": int(vals[i]), "flagged": bool(share[i] > flag_share),
            "flag_share": float(flag_share)}


# ============================================================================
# the public entry point
# ============================================================================

def set_metrics(ref: "QCReference", x: np.ndarray, class_id: int, *, scores: Optional[Mapping[str, np.ndarray]] = None,
                fold: Optional[int] = None, n_boot: Optional[int] = None, seed: int = 0,
                device: Optional[str] = None, a0: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, Any]:
    """Set-level metrics of the samples `x` (NHWC, cast to the float16 export)
    as class `class_id` against the real reference patches of that class.

    `scores`: the dict `apply_rules`/`QCReference.score` returned for these
    same samples; its `nn_patient` is reused for the attribution histogram
    (otherwise R6's search is run again). `fold=k` compares with fold k's
    train side only. `a0`: optional {"real_features", "synthetic_features"
    [, "synthetic_pred"]} for the reporting-only A0 block."""
    dev = resolve_device(device or ref.device)
    cfg = ref.config
    rng = np.random.default_rng(seed)
    syn = ref.feature_bundle(x)
    keep = syn.pop("finite")
    if scores is not None and "nn_patient" in scores and len(scores["nn_patient"]) != len(keep):
        raise ValueError(f"scores hold {len(scores['nn_patient'])} samples, x has {len(keep)}: pass the scores "
                         f"of these same samples (e.g. scores['nn_patient'][bitmask == 0])")
    n_dropped = int((~keep).sum())
    syn = take(syn, np.flatnonzero(keep))
    real = ref.real_bundle(class_id, fold=fold)
    sref = ref.set_reference(class_id)
    sigma = sref["sigma"]
    k = int(cfg.get("prdc_k", 5))
    prdc_rows, mmd_rows = int(cfg.get("prdc_rows", 5000)), int(cfg.get("mmd_rows", 2000))
    out: Dict[str, Any] = {"class": ref.class_name(class_id), "class_id": int(class_id), "fold": fold,
                           "n_synthetic": int(len(syn["mean"])), "n_dropped_nonfinite": n_dropped,
                           "n_real": int(len(real["mean"])), "ci_level": CI_LEVEL, "metrics": {}, "flags": []}
    if len(syn["mean"]) < 2:
        out["flags"].append("fewer than 2 finite synthetic samples: no set metrics")
        return out
    point = point_metrics(syn, real, sigma=sigma, k=k, prdc_rows=prdc_rows, mmd_rows=mmd_rows, rng=rng, device=dev)
    nb = int(cfg.get("bootstrap", 200) if n_boot is None else n_boot)
    cis = bootstrap_cis(syn, real, sigma=sigma, k=k, prdc_rows=prdc_rows, mmd_rows=mmd_rows, n_boot=nb, rng=rng,
                        device=dev) if nb > 0 else {}
    per_band = point.pop("wasserstein1_per_band")
    for name, v in point.items():
        rr = sref["ranges"].get(name, {})
        lo, hi = rr.get("min"), rr.get("max")
        within = None if lo is None or not np.isfinite(v) else bool(lo <= v <= hi)
        out["metrics"][name] = {"value": v, "ci": cis.get(name, [None, None]),
                                "real_range": {"min": lo, "median": rr.get("median"), "max": hi,
                                               "n_patients": rr.get("n_patients", 0)},
                                "within_real_range": within}
    out["metrics"]["wasserstein1_band_mean"]["per_band"] = per_band
    if scores is not None and "nn_patient" in scores:
        nn_patient = np.asarray(scores["nn_patient"])
    else:
        nn_patient = ref.score(x, class_id, fold=fold, device=dev)["nn_patient"]
    att = attribution(nn_patient, float(cfg.get("attribution_flag", 0.4)))
    att["real_bank_share"] = sref.get("bank_patient_share", {})
    att["real_heldout"] = sref.get("heldout_attribution", {})
    out["nn_attribution"] = att
    if att["flagged"]:
        out["flags"].append(f"nearest-neighbour attribution: patient {att['max_patient']} takes "
                            f"{att['max_share']:.1%} (> {att['flag_share']:.0%})")
    if a0 is not None:
        out["a0"] = a0_metrics(a0)
    return out


def a0_metrics(a0: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """REPORTING ONLY (Plan rule 7): Frechet distance between the A0
    classifier's penultimate features of real and synthetic patches, and the
    class mix A0 predicts for the synthetic ones. The features come from
    `training.latent_space_v18.extract_penultimate_features` on the caller's
    side; nothing here gates."""
    out: Dict[str, Any] = {"reporting_only": True,
                           "frechet": frechet(a0["synthetic_features"], a0["real_features"])}
    if a0.get("synthetic_pred") is not None:
        pred = np.asarray(a0["synthetic_pred"], dtype=np.int64)
        vals, cnt = np.unique(pred, return_counts=True)
        out["predicted_class_mix"] = {str(int(v)): float(c / cnt.sum()) for v, c in zip(vals, cnt)}
    return out
