# -*- coding: utf-8 -*-
"""
hsi_augment/pool.py
===================
Phase 4b (issue #16, Plan §Phase 4 items 3-4): the generator pool and the
real-vs-real reference statistics, run by `augment_pipeline.py pool`
(`stages/pool.py`). The pool's file contract, its loader and its writer are
`pool_io.py`; this module decides which rows go in, encodes them, and measures
the references.

Outputs in `<out>/pool/`
------------------------
    reference.npz          the real-vs-real references (schema below)
    pool.json              what went in and why: quotas, per-patient takes, the
                           log-space statistics, the clip, the round-trip error,
                           the fold sizes
    codec.json, y_pool.npy, patient_pool.npy, row_pool.npy, capture_pool.npy,
    X_pool.npy             `pool_io.write_pool`, LAST; X_pool.npy is the very
                           last file, so a directory holding it is complete

Which rows: the quotas
----------------------
Only TISSUE rows (`prep/tissue/tissue_mask_train.npy`, t >= tau): generators
are asked for tissue, never glass.

* The minority class (DCIS): every tissue row (`minority_rows` none). It has
  five train patients; leaving any of their rows out only loses data.
* Every other class (healthy, IDC): `rows_per_class` (120k) rows, balanced over
  PATIENTS. Rows of one patient are not independent, and drawn in proportion
  to the rows, the patients with the most captures would dominate. The quota
  is shared by water-filling (`balanced_allocation`): every patient gets the
  same share; a patient with fewer tissue rows than the share gives all of
  them, and what it leaves is re-shared among the others. So every patient
  takes min(n_p, L) or L + 1 rows for one level L, and the takes of the
  patients that were not exhausted differ by at most one. The leftover units
  go to the patients with the most rows (ties: lower id).
* Within a patient the rows are drawn uniformly without replacement from its
  own seeded stream, `ctx.rng("pool-rows", class, patient)`, so a change to
  one patient's rows does not reshuffle any other patient's draw.

The pool is stored in train-row order (`row_pool` ascending), so reading it
from X_train is a forward sequential pass.

One pass over the tissue rows
-----------------------------
The log-space statistics cover EVERY train tissue row, not only the pool's
(P2.3b: exact, float64). The stage therefore reads each train tissue row once,
in sorted memmap chunks of `chunk_rows`, and every chunk feeds:

  * `LogSpaceCodec.fit_stats`: per-band float64 mean/std of log(max(x, 0) +
    1e-3), merged with the Chan et al. update (exact whatever the chunking);
  * an exact per-band float16 value histogram (`discovery.Float16Histogram`)
    -> the exact p99.99 of each band (numpy's "inverted_cdf" quantile) -> the
    decode clip, clip_hi = 1.1 x p99.99 (`LogSpaceCodec.clip_hi_from_quantile`);
  * the pool rows it holds, copied into the in-RAM pool as raw float16, and
    their mean/std features.

Then the pool is encoded in place, block by block:
y = (log(max(x, 0) + eps) - mu_b) / sigma_b, stored as float16 (the contract).
Every block is decoded again to measure the round trip
(`pool.json["round_trip"]`). Relative to x + eps, the error is the float16
storage of y (about 2^-11 |y| sigma_b) plus the float16 rounding of the decoded
value (2^-11); values above clip_hi, or below 0, are clipped by design and
counted instead.

RAM (Plan rule 5): the pool is held in RAM, as the plan says (N x H x W x C
float16: 2.81 GB for the at most 122,850 + 2 x 120,000 rows of 11x11x32 on
trainsel), plus its features, its PCA projections and the PCA covariance
(~0.5 GB). `max_pool_gb` (3.5) is checked before any X is read.

reference.npz (`REFERENCE_SCHEMA_VERSION` 1, uncompressed `np.savez`)
---------------------------------------------------------------------
The LOPO folds are prep/leakage/folds.json; fold k's held-out and train sides
are `splits.fold_masks(patient_pool, fold)` over the pool. K classes, F folds,
Q quantile levels, N pool rows, C bands, D = H*W*C, k = PCA dimensions, M =
nearest-neighbour queries over every fold and class.

    schema_version               ()               int64
    class_names                  (K,)             str
    quantile_levels              (Q,)             float64
    fold_dcis_patient            (F,)             int64
  PCA, fitted on the ENCODED pool rows, each flattened (H, W, C) row-major:
    pca_mean                     (D,)             float64
    pca_components               (k, D)           float64  orthonormal rows, by variance
                                                           descending; largest |entry| > 0
    pca_explained_variance       (k,)             float64  (ddof 1)
    pca_explained_variance_ratio (k,)             float64  of the total variance
    pca_fit_index                (n_fit,)         int64    the pool rows it was fitted on
  Cross-capture nearest neighbours. Per fold k and class c, every held-out
  pool row of class c (a seeded `nn_max_queries` subsample) against EVERY pool
  row of class c on the fold's train side, Euclidean in the PCA space; the
  distance of the chosen pair is recomputed exactly:
    nn_query                     (M,)             int64    pool index of the held-out row
    nn_fold, nn_class            (M,)             int64
    nn_match                     (M,)             int64    pool index of its nearest neighbour
    nn_dist                      (M,)             float64
    nn_duplicate                 (M,)             bool     raw patches byte-identical
    nn_n_heldout, nn_n_query, nn_n_reference (F, K) int64
    nn_quantiles                 (F, K, Q)        float64  of nn_dist, duplicates excluded;
                                                           NaN where there is none
    nn_quantiles_pooled          (K, Q)           float64  the same over every fold
  Mean/std features: `provenance.patch_feats` of the RAW patch ([:C] per-band
  mean, [C:] per-band std, ddof 0), the feature space of index_train.npz and
  of QC (#28); `feats[i]` equals index_train.npz `patch_feats[row_pool[i]]`:
    feats                        (N, 2C)          float32  every pool row
    feat_n                       (F, K, 2)        int64    side 0 = held out, 1 = train side
    feat_mean, feat_std          (F, K, 2, 2C)    float64
    feat_quantiles               (F, K, 2, Q, 2C) float64

Quantiles are numpy's default ("linear"). Read it with `load_reference`:

    ref = load_reference(out / "pool")
    q01 = ref["nn_quantiles_pooled"][1, list(ref["quantile_levels"]).index(0.01)]

The set-level metric ranges between patients (Frechet distance, MMD) are the
QC library's (#28), computed from these features with its own code.

Why held-out patients against the rest
--------------------------------------
A generator fitted on a fold's train side should produce samples about as far
from its training rows as rows of a real patient it never saw are. The nearest
train-side neighbour of a held-out patient's row is that natural distance, and
every such pair is cross-patient, hence cross-capture: it is the reference of
the memorization rule (R6: a sample's NN distance >= q1 of these). A pair whose
raw patches are byte-identical is a DATA duplicate (P1 reports them), not
natural variation; it is flagged and left out of the quantiles. The PCA is
fitted once on the pool (the plan's "PCA fitted on the train pool"): one fixed
feature space for every fold, which a generated sample is projected into the
same way.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hsi_augment import SCHEMA_VERSION
from hsi_augment.codec import CLIP_HI_FACTOR, LOG_EPS, LogSpaceCodec
from hsi_augment.io import RunningBandStats, iter_chunks
from hsi_augment.splits import fold_heldout_patients, fold_masks

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.discovery import Float16Histogram
    from hsi_augment.stage import StageContext
    from hsi_augment.stages.pool import PoolConfig

PathLike = Union[str, os.PathLike]

POOL_META_NAME = "pool.json"
REFERENCE_NAME = "reference.npz"
POOL_META_SCHEMA_VERSION = 1
REFERENCE_SCHEMA_VERSION = 1
CLIP_QUANTILE = 0.9999                 # clip_hi = CLIP_HI_FACTOR x the train-tissue p99.99, per band
TAU_SIGNOFF_ATOL_DEG = 5e-4            # the stage logs tau with 3 decimals
ROWS_RNG_KEY = "pool-rows"             # ctx.rng(ROWS_RNG_KEY, class, patient)
PCA_RNG_KEY = "pca-rows"               # ctx.rng(PCA_RNG_KEY)
NN_RNG_KEY = "nn-queries"              # ctx.rng(NN_RNG_KEY, fold, class)

# Rows widened to float32/float64 at a time: 512 x 11 x 11 x 32 x 8 B = 16 MB.
BLOCK_ROWS = 512
PCA_BLOCK_ROWS = 2048                  # 2048 x 3872 x 8 B = 63 MB per centred block
NN_QUERY_BLOCK = 1024
NN_REF_BLOCK = 16384                   # 1024 x 16384 x 8 B = 134 MB of squared distances

SIDES = ("heldout", "train")
POOL_RULE = ("Tissue rows only (prep/tissue mask). The minority class: every tissue row, unless minority_rows caps "
             "it. Every other class: rows_per_class tissue rows, shared across its train patients by "
             "water-filling (each patient takes min(n_p, L) or L + 1 rows for one level L; leftover units to the "
             "patients with the most rows, ties to the lower id), drawn uniformly without replacement within each "
             "patient from the seeded stream (pool-rows, class, patient). A capped minority class is shared the "
             "same way. Stored in train-row order.")
NN_RULE = ("Per LOPO fold and class: every held-out pool row of the class (a seeded nn_max_queries subsample) "
           "against every pool row of the same class on the fold's train side, Euclidean distance in the PCA "
           "space, exact for the chosen pair. Held-out and train-side patients differ, so every pair is "
           "cross-patient and cross-capture. Byte-identical raw patches are data duplicates: flagged in "
           "nn_duplicate and excluded from the quantiles.")


# ============================================================================
# the quotas
# ============================================================================

def balanced_allocation(counts: Mapping[int, int], quota: Optional[int]) -> Tuple[Dict[int, int], Optional[int]]:
    """Share `quota` rows across patients with `counts[p]` rows each, as evenly
    as their rows allow (water-filling). Returns ({patient: rows taken}, L):
    every patient takes min(n_p, L) rows, or L + 1 for the leftover units,
    which go to the patients with the most rows (ties: lower id). L is None
    when the quota covers every row (`quota` None or >= the total): all taken."""
    n = {int(p): int(c) for p, c in counts.items()}
    if any(c < 0 for c in n.values()):
        raise ValueError("row counts must be >= 0")
    total = sum(n.values())
    if quota is None or int(quota) >= total:
        return dict(n), None
    if int(quota) < 0:
        raise ValueError("quota must be >= 0")
    alloc = {p: 0 for p in n}
    pending = sorted((p for p in n if n[p] > 0), key=lambda p: (n[p], p))
    remaining = int(quota)
    i = 0
    while i < len(pending):                 # patients below the current even share give everything
        share = remaining // (len(pending) - i)
        p = pending[i]
        if n[p] > share:
            break
        alloc[p] = n[p]
        remaining -= n[p]
        i += 1
    rest = pending[i:]                      # non-empty: quota < total
    level, extra = divmod(remaining, len(rest))
    for j, p in enumerate(sorted(rest, key=lambda p: (-n[p], p))):
        alloc[p] = level + (1 if j < extra else 0)    # n[p] > level, so L + 1 always fits
    return alloc, level


def class_quotas(num_classes: int, minority_classes: Sequence[int], rows_per_class: int,
                 minority_rows: Optional[int]) -> Dict[int, Optional[int]]:
    """{class: quota}; None = every tissue row of the class."""
    minority = {int(c) for c in minority_classes}
    return {c: (minority_rows if c in minority else int(rows_per_class)) for c in range(int(num_classes))}


def select_rows(y: np.ndarray, groups: np.ndarray, is_tissue: np.ndarray, quotas: Mapping[int, Optional[int]],
                rng_for: Callable[[int, int], np.random.Generator]) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """The pool's train rows (int64, ascending) and the allocation table
    {class: {quota, tissue_rows, pool_rows, level, patients: {p: {tissue_rows,
    pool_rows}}}}. `rng_for(class, patient)` gives that patient's stream; it is
    only drawn from when the patient gives fewer rows than it has."""
    yy = np.asarray(y, dtype=np.int64)
    g = np.asarray(groups, dtype=np.int64)
    m = np.asarray(is_tissue, dtype=bool)
    if not (yy.shape == g.shape == m.shape):
        raise ValueError(f"y, groups and the tissue mask have different shapes {yy.shape, g.shape, m.shape}")
    picked: List[np.ndarray] = []
    table: Dict[int, Dict[str, Any]] = {}
    for c in sorted(int(k) for k in quotas):
        rows_c = np.flatnonzero(m & (yy == c))
        order = np.argsort(g[rows_c], kind="stable")          # rows_c grouped by patient, ascending rows
        pats, starts, n_p = np.unique(g[rows_c][order], return_index=True, return_counts=True)
        counts = {int(p): int(k) for p, k in zip(pats, n_p)}
        alloc, level = balanced_allocation(counts, quotas[c])
        for p, a0, k in zip(pats, starts, n_p):
            rows_p = rows_c[order[a0:a0 + k]]
            take = alloc[int(p)]
            if take < k:
                rows_p = np.sort(rng_for(c, int(p)).choice(rows_p, size=take, replace=False))
            picked.append(rows_p)
        table[c] = {"quota": quotas[c], "tissue_rows": int(rows_c.size), "pool_rows": int(sum(alloc.values())),
                    "level": level,
                    "patients": {int(p): {"tissue_rows": counts[int(p)], "pool_rows": alloc[int(p)]} for p in pats}}
    rows = np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)
    return rows.astype(np.int64, copy=False), table


def pool_nbytes(n_rows: int, row_shape: Sequence[int]) -> int:
    """Bytes of X_pool.npy's data (float16) for `n_rows` rows of `row_shape`."""
    return int(n_rows) * int(np.prod(tuple(row_shape))) * np.dtype(np.float16).itemsize


# ============================================================================
# the pass over the tissue rows, and the encoding
# ============================================================================

def scan_tissue_rows(X: np.ndarray, tissue_rows: np.ndarray, pool_rows: np.ndarray, *,
                     chunk_rows: int, eps: float = LOG_EPS,
                     on_chunk: Optional[Callable[[int], None]] = None
                     ) -> Tuple[RunningBandStats, "Float16Histogram", np.ndarray, np.ndarray]:
    """One sorted pass over `X[tissue_rows]`: the log-space stats, the exact
    float16 value histogram, the raw pool rows (`pool_rows` must be a sorted
    subset of `tissue_rows`) and their mean/std features. Returns
    (stats, hist, X_pool_raw float16 (n, H, W, C), feats float32 (n, 2C))."""
    from hsi_augment.discovery import Float16Histogram
    from hsi_augment.provenance import patch_feats

    tissue_rows = np.asarray(tissue_rows, dtype=np.int64)
    pool_rows = np.asarray(pool_rows, dtype=np.int64)
    n_pool = int(pool_rows.size)
    if n_pool and (np.any(np.diff(pool_rows) <= 0) or not np.isin(pool_rows, tissue_rows).all()):
        raise ValueError("pool_rows must be unique, ascending, and a subset of tissue_rows")
    n_bands = int(X.shape[-1])
    X_pool = np.empty((n_pool, *X.shape[1:]), dtype=np.float16)
    feats = np.empty((n_pool, 2 * n_bands), dtype=np.float32)
    hist = Float16Histogram(n_bands)
    filled = [0]

    def chunks():
        for ch in iter_chunks(X, tissue_rows, chunk_rows=chunk_rows, assume_sorted=True):
            hist.update(ch.data)
            if n_pool:
                pos = np.searchsorted(pool_rows, ch.rows)
                hit = pos < n_pool
                hit[hit] = pool_rows[pos[hit]] == ch.rows[hit]
                if hit.any():
                    block = np.asarray(ch.data[hit], dtype=np.float16)
                    X_pool[pos[hit]] = block
                    feats[pos[hit]] = patch_feats(block)
                    filled[0] += int(hit.sum())
            if on_chunk is not None:
                on_chunk(int(ch.rows.size))
            yield ch.data

    stats = LogSpaceCodec.fit_stats(chunks(), n_bands=n_bands, eps=eps)
    if filled[0] != n_pool:
        raise RuntimeError(f"the pass filled {filled[0]} of {n_pool} pool rows")
    return stats, hist, X_pool, feats


def encode_in_place(X_pool: np.ndarray, codec: LogSpaceCodec, *,
                    block_rows: int = BLOCK_ROWS) -> Tuple[Dict[str, Any], RunningBandStats]:
    """Overwrite the raw float16 pool with float16 log-z values, block by
    block, decoding each block again to measure the round trip. Returns the
    round-trip report and float64 band stats of the stored log-z values.
    Raises ValueError on a non-finite raw value (log of NaN stays NaN)."""
    n = int(X_pool.shape[0])
    eps32 = np.float32(codec.eps)
    hi32 = codec.clip_hi.astype(np.float32)
    enc_stats = RunningBandStats(codec.n_bands)
    max_rel, sum_rel, n_vals = 0.0, 0.0, 0
    n_clipped_hi = n_negative = 0
    worst_row = None
    for a in range(0, n, block_rows):
        b = min(a + block_rows, n)
        raw = X_pool[a:b].astype(np.float32)
        if not np.isfinite(raw).all():
            bad = a + int(np.flatnonzero(~np.isfinite(raw).reshape(b - a, -1).all(axis=1))[0])
            raise ValueError(f"pool row {bad} holds a non-finite value")
        enc = codec.encode_np(raw).astype(np.float16)
        if not np.isfinite(enc).all():
            raise ValueError(f"pool rows {a}..{b - 1}: log-z values overflow float16")
        dec = codec.decode_np(enc).astype(np.float32)
        ok = (raw >= 0) & (raw <= hi32)
        n_negative += int((raw < 0).sum())
        n_clipped_hi += int((raw > hi32).sum())
        rel = np.abs(dec - raw) / (raw + eps32)
        rel = np.where(ok, rel, np.float32(0.0))
        blk_max = float(rel.max()) if rel.size else 0.0
        if blk_max > max_rel:
            max_rel = blk_max
            worst_row = a + int(np.argmax(rel.reshape(b - a, -1).max(axis=1)))
        sum_rel += float(rel.sum(dtype=np.float64))
        n_vals += int(ok.sum())
        X_pool[a:b] = enc
        enc_stats.update(enc)
    report = {
        "metric": "|decode(stored y) - x| / (x + eps) per value, over 0 <= x <= clip_hi",
        "n_values": n_vals, "max_rel_err": max_rel, "mean_rel_err": (sum_rel / n_vals) if n_vals else None,
        "worst_pool_row": worst_row, "n_above_clip_hi": n_clipped_hi, "n_negative": n_negative,
        "expected_bound": "float16 storage of y (2^-11 |y| sigma_b) + float16 output rounding (2^-11)",
    }
    return report, enc_stats


# ============================================================================
# PCA and nearest neighbours
# ============================================================================

def fit_pca(X: np.ndarray, fit_index: np.ndarray, n_components: int, *,
            block_rows: int = PCA_BLOCK_ROWS) -> Dict[str, np.ndarray]:
    """PCA of the rows `X[fit_index]`, each flattened: float64 mean, then the
    centred covariance (ddof 1) accumulated in blocks, then `eigh`. Returns
    {mean (D,), components (k, D), explained_variance (k,),
    explained_variance_ratio (k,)}, k = min(n_components, D). Components are
    orthonormal, by variance descending, signed so their largest |entry| is
    positive (deterministic)."""
    idx = np.asarray(fit_index, dtype=np.int64)
    n = int(idx.size)
    if n < 2:
        raise ValueError("the PCA needs at least 2 rows")
    D = int(np.prod(X.shape[1:]))
    k = min(int(n_components), D)
    total = np.zeros(D, dtype=np.float64)
    for a in range(0, n, block_rows):
        total += np.asarray(X[idx[a:a + block_rows]]).reshape(-1, D).astype(np.float64).sum(axis=0)
    mean = total / n
    cov = np.zeros((D, D), dtype=np.float64)
    for a in range(0, n, block_rows):
        blk = np.asarray(X[idx[a:a + block_rows]]).reshape(-1, D).astype(np.float64)
        blk -= mean
        cov += blk.T @ blk
    cov /= n - 1
    trace = float(np.trace(cov))
    w, V = np.linalg.eigh(cov)                       # ascending
    del cov
    order = np.arange(D - 1, D - 1 - k, -1)
    comps = np.ascontiguousarray(V[:, order].T)
    del V
    j = np.argmax(np.abs(comps), axis=1)
    sign = np.sign(comps[np.arange(k), j])
    sign[sign == 0] = 1.0
    comps *= sign[:, None]
    ev = np.maximum(w[order], 0.0)
    ratio = ev / trace if trace > 0 else np.zeros_like(ev)
    return {"mean": mean, "components": comps, "explained_variance": ev, "explained_variance_ratio": ratio}


def project(X: np.ndarray, pca: Mapping[str, np.ndarray], *, block_rows: int = PCA_BLOCK_ROWS) -> np.ndarray:
    """(n, k) float64 PCA coordinates of every row of X (flattened, centred)."""
    comps = np.asarray(pca["components"], dtype=np.float64)
    mean = np.asarray(pca["mean"], dtype=np.float64)
    D = int(mean.size)
    n = int(X.shape[0])
    out = np.empty((n, comps.shape[0]), dtype=np.float64)
    for a in range(0, n, block_rows):
        blk = np.asarray(X[a:a + block_rows]).reshape(-1, D).astype(np.float64)
        blk -= mean
        out[a:a + blk.shape[0]] = blk @ comps.T
    return out


def nearest_neighbours(queries: np.ndarray, reference: np.ndarray, *, query_block: int = NN_QUERY_BLOCK,
                       ref_block: int = NN_REF_BLOCK) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Euclidean nearest neighbour of every query row among the
    reference rows: (dist float64 (m,), index int64 (m,) into `reference`).
    Candidates are ranked by the float64 expansion |q|^2 + |r|^2 - 2 q.r in
    blocks (no m x n matrix), and the chosen pair's distance is recomputed
    directly. NaN / -1 when there is no reference row."""
    q = np.asarray(queries, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    m, n = int(q.shape[0]), int(r.shape[0])
    dist = np.full(m, np.nan)
    index = np.full(m, -1, dtype=np.int64)
    if m == 0 or n == 0:
        return dist, index
    rn = np.einsum("ij,ij->i", r, r)
    for a in range(0, m, query_block):
        qb = q[a:a + query_block]
        qn = np.einsum("ij,ij->i", qb, qb)
        best = np.full(qb.shape[0], np.inf)
        arg = np.zeros(qb.shape[0], dtype=np.int64)
        for c0 in range(0, n, ref_block):
            rb = r[c0:c0 + ref_block]
            d2 = qn[:, None] + rn[None, c0:c0 + rb.shape[0]] - 2.0 * (qb @ rb.T)
            j = np.argmin(d2, axis=1)
            v = d2[np.arange(qb.shape[0]), j]
            better = v < best
            best[better] = v[better]
            arg[better] = c0 + j[better]
        index[a:a + qb.shape[0]] = arg
        dist[a:a + qb.shape[0]] = np.sqrt(((qb - r[arg]) ** 2).sum(axis=1))
    return dist, index


# ============================================================================
# the real-vs-real references
# ============================================================================

def _quantiles(values: np.ndarray, levels: Sequence[float], axis: int = 0) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=np.float64), np.asarray(levels, dtype=np.float64), axis=axis)


def fold_references(proj: np.ndarray, feats: np.ndarray, y: np.ndarray, patient: np.ndarray,
                    folds: Sequence[Mapping[str, Any]], num_classes: int, *, quantiles: Sequence[float],
                    max_queries: Optional[int], rng_for: Callable[[int, int], np.random.Generator],
                    is_duplicate: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Dict[str, np.ndarray]:
    """The `nn_*` and `feat_*` members of reference.npz (module docstring),
    over pool rows. `proj` (N, k) PCA coordinates, `feats` (N, 2C), `y` and
    `patient` per pool row. `rng_for(fold, class)` draws the query subsample;
    `is_duplicate(a, b)` says, for pool index pairs at distance 0, whether
    their raw patches are byte-identical."""
    yy = np.asarray(y, dtype=np.int64)
    F, K, Q = len(folds), int(num_classes), len(quantiles)
    C2 = int(feats.shape[1])
    out: Dict[str, Any] = {
        "nn_n_heldout": np.zeros((F, K), dtype=np.int64), "nn_n_query": np.zeros((F, K), dtype=np.int64),
        "nn_n_reference": np.zeros((F, K), dtype=np.int64), "nn_quantiles": np.full((F, K, Q), np.nan),
        "feat_n": np.zeros((F, K, 2), dtype=np.int64), "feat_mean": np.full((F, K, 2, C2), np.nan),
        "feat_std": np.full((F, K, 2, C2), np.nan), "feat_quantiles": np.full((F, K, 2, Q, C2), np.nan),
    }
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in ("query", "fold", "class", "match", "dist", "dup")}
    for fi, fold in enumerate(folds):
        train_m, held_m = fold_masks(patient, fold)
        for c in range(K):
            held = np.flatnonzero(held_m & (yy == c))
            ref = np.flatnonzero(train_m & (yy == c))
            for side, rows in enumerate((held, ref)):
                out["feat_n"][fi, c, side] = rows.size
                if rows.size:
                    f = feats[rows].astype(np.float64)
                    out["feat_mean"][fi, c, side] = f.mean(axis=0)
                    out["feat_std"][fi, c, side] = f.std(axis=0)
                    out["feat_quantiles"][fi, c, side] = _quantiles(f, quantiles)
            out["nn_n_heldout"][fi, c] = held.size
            out["nn_n_reference"][fi, c] = ref.size
            query = held
            if max_queries is not None and held.size > max_queries:
                query = np.sort(rng_for(fi, c).choice(held, size=int(max_queries), replace=False))
            out["nn_n_query"][fi, c] = query.size
            if query.size == 0 or ref.size == 0:
                continue
            dist, j = nearest_neighbours(proj[query], proj[ref])
            match = ref[j]
            dup = np.zeros(query.size, dtype=bool)
            zero = dist == 0.0
            if zero.any():
                dup[zero] = np.asarray(is_duplicate(query[zero], match[zero]), dtype=bool)
            if (~dup).any():
                out["nn_quantiles"][fi, c] = _quantiles(dist[~dup], quantiles)
            for key, v in (("query", query), ("fold", np.full(query.size, fi)), ("class", np.full(query.size, c)),
                           ("match", match), ("dist", dist), ("dup", dup)):
                parts[key].append(v)

    def cat(key: str, dtype: Any) -> np.ndarray:
        return np.concatenate(parts[key]).astype(dtype) if parts[key] else np.zeros(0, dtype=dtype)

    out.update({"nn_query": cat("query", np.int64), "nn_fold": cat("fold", np.int64),
                "nn_class": cat("class", np.int64), "nn_match": cat("match", np.int64),
                "nn_dist": cat("dist", np.float64), "nn_duplicate": cat("dup", bool)})
    pooled = np.full((K, Q), np.nan)
    for c in range(K):
        sel = (out["nn_class"] == c) & ~out["nn_duplicate"]
        if sel.any():
            pooled[c] = _quantiles(out["nn_dist"][sel], quantiles)
    out["nn_quantiles_pooled"] = pooled
    return out


def load_reference(path: PathLike) -> Dict[str, np.ndarray]:
    """reference.npz, from its path or the pool directory, schema-checked."""
    p = Path(path)
    if p.is_dir():
        p = p / REFERENCE_NAME
    with np.load(p, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files}
    if int(ref.get("schema_version", -1)) != REFERENCE_SCHEMA_VERSION:
        raise ValueError(f"{p}: not a reference.npz of schema_version {REFERENCE_SCHEMA_VERSION}")
    return ref


# ============================================================================
# the stage
# ============================================================================

def _json_safe(obj: Any) -> Any:
    from hsi_augment.preprocess import json_safe
    return json_safe(obj)


def _per_class(values: Sequence[Any], class_names: Sequence[str]) -> Dict[str, Any]:
    return {name: values[i] for i, name in enumerate(class_names)}


def _gb(n: float) -> str:
    return f"{n / 1e9:.3f} GB"


def run_pool(ctx: "StageContext", cfg: "PoolConfig") -> Dict[str, Any]:
    """The `pool` stage: quotas, one pass over the train tissue rows, the
    encoding, the references, then the pool files. See the module docstring."""
    from hsi_augment import splits, tissue
    from hsi_augment.pool_io import CODEC_NAME, POOL_FILES, write_pool
    from hsi_augment.provenance import save_npz_atomic
    from hsi_augment.stage import StageError

    adapter = ctx.adapter
    class_names = list(adapter.class_names)
    K = len(class_names)
    minority = sorted(int(c) for c in adapter.minority_classes)
    if not minority:
        raise StageError("the adapter declares no minority class")
    levels = [float(q) for q in cfg.quantiles]
    # A previous run's X_pool.npy would make a half-rewritten directory look
    # complete (pool_io: a pool with X present is complete). The runner has
    # already removed that run's done.json, so the old pool is invalid anyway.
    (ctx.stage_dir / POOL_FILES["X"]).unlink(missing_ok=True)

    # 1. The tissue-mask outputs, and tau against the signed-off value.
    tdir = ctx.upstream_dir("tissue-mask")
    try:
        with open(tdir / tissue.TAU_NAME) as f:
            tau_doc = json.load(f)
        score = np.load(tdir / tissue.SCORE_NAME, mmap_mode="r")
        mask = np.load(tdir / tissue.MASK_NAME, mmap_mode="r")
    except (OSError, ValueError) as e:
        raise StageError(f"tissue-mask outputs unreadable in {tdir}: {e}") from e
    tau = float(tau_doc["tau_deg"])
    tissue_complete = bool(tau_doc.get("complete", True))
    if cfg.expect_tau_deg is None:
        ctx.warn(f"expect_tau_deg is not set: tau = {tau:.3f} deg is used without a recorded sign-off "
                 f"(Plan Phase 2 item 6: the user approves tau before the pool is built)")
    elif abs(tau - float(cfg.expect_tau_deg)) > TAU_SIGNOFF_ATOL_DEG:
        raise StageError(f"the tissue mask was fitted with tau = {tau:.6f} deg, the signed-off tau is "
                         f"{cfg.expect_tau_deg} deg (tolerance {TAU_SIGNOFF_ATOL_DEG}): re-run tissue-mask "
                         f"or sign off the new tau")
    if not tissue_complete:
        ctx.warn("the tissue-mask stage ran on a subset of captures (complete=false): only their rows can be "
                 "tissue, so the pool and its statistics cover only those captures")

    # 2. Train, with X, under `pool` (train only).
    train = ctx.open_split("train", purpose="pool")
    if train.groups is None or train.captures is None:
        raise StageError("the train split needs groups_train.npy and captures_train.npy")
    N = train.n
    if score.shape != (N,) or mask.shape != (N,) or mask.dtype != np.bool_:
        raise StageError(f"tissue-mask outputs do not describe this train split: score {score.shape}, mask "
                         f"{mask.shape} {mask.dtype}, train has {N} rows; re-run tissue-mask")
    mask = np.asarray(mask, dtype=bool)
    if not np.array_equal(mask, tissue.apply_threshold(np.asarray(score), tau)):
        raise StageError(f"{tissue.MASK_NAME} is not {tissue.MASK_RULE} applied to {tissue.SCORE_NAME} with "
                         f"tau {tau}: tissue outputs from different runs; re-run tissue-mask")
    del score
    y = np.asarray(train.y, dtype=np.int64)
    groups = np.asarray(train.groups, dtype=np.int64)
    captures = np.asarray(train.captures, dtype=np.int64)

    # 3. The LOPO folds, checked against this train split.
    folds_doc = splits.load_folds(ctx.upstream_dir("leakage"))
    problems = splits.check_folds(folds_doc, groups, y, class_names=class_names, minority_classes=minority,
                                  captures=captures)
    if problems:
        raise StageError("folds.json does not fit this train split: " + "; ".join(problems))
    folds = list(folds_doc["folds"])

    # 4. Quotas and rows; the size cap before any X is read.
    quotas = class_quotas(K, minority, cfg.rows_per_class, cfg.minority_rows)
    rows, table = select_rows(y, groups, mask, quotas, lambda c, p: ctx.rng(ROWS_RNG_KEY, c, p))
    n_pool = int(rows.size)
    row_shape = tuple(int(s) for s in train.X.shape[1:])
    n_bands = row_shape[-1]
    x_bytes = pool_nbytes(n_pool, row_shape)
    limit = int(round(float(cfg.max_pool_gb) * 1e9))
    for c in range(K):
        t = table[c]
        quota = "all" if t["quota"] is None else f"{t['quota']:,}"
        level = "" if t["level"] is None else f", level {t['level']}"
        ctx.log(f"{class_names[c]}: {t['pool_rows']:,} of {t['tissue_rows']:,} tissue rows from "
                f"{len(t['patients'])} patients (quota {quota}{level})")
    empty = [class_names[c] for c in range(K) if table[c]["pool_rows"] == 0]
    if empty:
        raise StageError(f"no tissue rows of class(es) {empty} in train: a class-conditional generator needs "
                         f"every class")
    if x_bytes > limit:
        raise StageError(f"the pool would be {n_pool:,} rows x {row_shape} float16 = {_gb(x_bytes)}, above "
                         f"max_pool_gb = {cfg.max_pool_gb} ({_gb(limit)}); lower rows_per_class / minority_rows")
    D = int(np.prod(row_shape))
    k = min(int(cfg.pca_components), D)
    n_fit = n_pool if cfg.pca_fit_rows is None else min(int(cfg.pca_fit_rows), n_pool)
    if n_fit < 2:
        raise StageError(f"the pool has {n_pool} row(s); the reference PCA needs at least 2")
    feat_bytes = n_pool * 2 * n_bands * 4
    projected = (x_bytes + feat_bytes + n_pool * k * 8              # pool, features, projections
                 + 3 * D * D * 8                                    # covariance, eigenvectors, workspace
                 + 2 * PCA_BLOCK_ROWS * D * 8 + 2 * NN_QUERY_BLOCK * NN_REF_BLOCK * 8
                 + N * (3 * 8 + 1)                                  # y, groups, captures, mask
                 + int(cfg.chunk_rows) * D * 2 * 2                  # a memmap chunk and its fancy-index copy
                 + 2 * n_bands * 65536 * 8)                         # the float16 histogram and its bincount
    ctx.ram_guard(projected, "pool in RAM + features + PCA + nearest-neighbour blocks")
    ref_bytes = feat_bytes + 2 * k * D * 8 + 64 * (n_pool + 1)
    ctx.disk_preflight(ctx.path(POOL_FILES["X"]), x_bytes + ref_bytes)
    ctx.log(f"pool: {n_pool:,} rows x {row_shape} = {_gb(x_bytes)} (cap {_gb(limit)}); "
            f"tau {tau:.3f} deg; {len(folds)} LOPO folds")

    # 5. One pass over every train tissue row.
    tissue_rows = np.flatnonzero(mask)
    progress = {"done": 0, "next": max(1, tissue_rows.size // 10)}

    def on_chunk(n: int) -> None:
        progress["done"] += n
        if progress["done"] >= progress["next"] or progress["done"] == tissue_rows.size:
            ctx.log(f"scan: {progress['done']:,}/{tissue_rows.size:,} tissue rows")
            progress["next"] += max(1, tissue_rows.size // 10)
        ctx.log_metrics({"tissue_rows_done": progress["done"]})

    with ctx.timer("scan"):
        stats, hist, X_pool, feats = scan_tissue_rows(train.X, tissue_rows, rows, chunk_rows=cfg.chunk_rows,
                                                      on_chunk=on_chunk)
    if stats.n_nonfinite.any():
        ctx.warn(f"{int(stats.n_nonfinite.sum())} non-finite values among the train tissue rows: excluded from "
                 f"the log-space statistics")
    p_hi = hist.quantiles([CLIP_QUANTILE])[0]
    if not (np.all(np.isfinite(p_hi)) and np.all(p_hi > 0)):
        raise StageError(f"the train-tissue p{100 * CLIP_QUANTILE:g} is not positive and finite in every band: "
                         f"{p_hi.tolist()}")
    clip_hi = LogSpaceCodec.clip_hi_from_quantile(p_hi)
    try:
        codec = LogSpaceCodec.from_stats(stats, clip_hi, eps=LOG_EPS)
    except ValueError as e:
        raise StageError(f"log-space statistics unusable: {e}") from e
    ctx.log(f"log space over {tissue_rows.size:,} tissue rows: mu {np.round(codec.mu, 3).tolist()}, "
            f"sigma {np.round(codec.sigma, 3).tolist()}")

    # 6. Encode in place, measuring the round trip.
    with ctx.timer("encode"):
        try:
            round_trip, enc_stats = encode_in_place(X_pool, codec)
        except ValueError as e:
            raise StageError(f"cannot encode the pool: {e}") from e
    ctx.log(f"round trip: max |decode(y) - x| / (x + eps) = {round_trip['max_rel_err']:.2e} over "
            f"{round_trip['n_values']:,} values ({round_trip['n_above_clip_hi']} above clip_hi, "
            f"{round_trip['n_negative']} negative)")

    # 7. The references: PCA on the encoded pool, then per fold and class.
    pool_y, pool_patient, pool_capture = y[rows], groups[rows], captures[rows]
    with ctx.timer("pca"):
        if n_fit == n_pool:
            fit_index = np.arange(n_pool, dtype=np.int64)
        else:
            fit_index = np.sort(ctx.rng(PCA_RNG_KEY).choice(n_pool, size=n_fit, replace=False)).astype(np.int64)
        pca = fit_pca(X_pool, fit_index, k)
        proj = project(X_pool, pca)
    ctx.log(f"PCA-{k} on {n_fit:,} pool rows: {pca['explained_variance_ratio'].sum():.1%} of the variance")

    def is_duplicate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array([np.asarray(train.X[rows[i]]).tobytes() == np.asarray(train.X[rows[j]]).tobytes()
                         for i, j in zip(a.tolist(), b.tolist())], dtype=bool)

    with ctx.timer("references"):
        ref = fold_references(proj, feats, pool_y, pool_patient, folds, K, quantiles=levels,
                              max_queries=cfg.nn_max_queries, rng_for=lambda f, c: ctx.rng(NN_RNG_KEY, f, c),
                              is_duplicate=is_duplicate)
    del proj
    arrays = {
        "schema_version": np.array(REFERENCE_SCHEMA_VERSION, dtype=np.int64),
        "class_names": np.array(class_names, dtype=str),
        "quantile_levels": np.array(levels, dtype=np.float64),
        "fold_dcis_patient": np.array([int(f["dcis_patient"]) for f in folds], dtype=np.int64),
        "pca_mean": pca["mean"], "pca_components": pca["components"],
        "pca_explained_variance": pca["explained_variance"],
        "pca_explained_variance_ratio": pca["explained_variance_ratio"], "pca_fit_index": fit_index,
        "feats": feats, **ref,
    }
    with ctx.timer("save"):
        ctx.add_output(save_npz_atomic(ctx.path(REFERENCE_NAME), arrays))
    dups = np.flatnonzero(ref["nn_duplicate"])
    for c in range(K):
        q = ref["nn_quantiles_pooled"][c]
        ctx.log(f"{class_names[c]}: held-out -> train-side NN distance (PCA-{k}) "
                + ", ".join(f"q{100 * lv:g} {v:.3f}" for lv, v in zip(levels, q)))
    if dups.size:
        ctx.warn(f"{dups.size} held-out pool row(s) have a byte-identical patch on their fold's train side "
                 f"(data duplicates; excluded from the NN quantiles)")
    empty_refs = [(int(f), class_names[c]) for f, c in zip(*np.nonzero((ref["nn_n_query"] == 0)
                                                                       | (ref["nn_n_reference"] == 0)))]
    if empty_refs:
        ctx.warn(f"(fold, class) pairs without held-out or train-side pool rows, no NN reference: {empty_refs}")

    # 8. pool.json.
    fold_rows = []
    for f in folds:
        tm, hm = fold_masks(pool_patient, f)
        fold_rows.append({"fold": int(f["fold"]), "dcis_patient": int(f["dcis_patient"]),
                          "heldout_patients": list(fold_heldout_patients(f)),
                          "heldout_pool_rows": _per_class(np.bincount(pool_y[hm], minlength=K).tolist(), class_names),
                          "train_pool_rows": _per_class(np.bincount(pool_y[tm], minlength=K).tolist(), class_names)})
    meta = {
        "schema_version": SCHEMA_VERSION,
        "pool_schema_version": POOL_META_SCHEMA_VERSION,
        "contract": "hsi_augment/pool_io.py (load_pool / write_pool)",
        "files": {**POOL_FILES, "codec": CODEC_NAME, "reference": REFERENCE_NAME},
        "rule": POOL_RULE,
        "n_rows": n_pool, "x_shape": [n_pool, *row_shape], "x_dtype": "float16", "x_layout": "NHWC",
        "x_bytes": x_bytes, "max_pool_bytes": limit, "n_train_rows": N,
        "class_names": class_names, "minority_classes": [class_names[c] for c in minority],
        "classes": {class_names[c]: {
            "quota": table[c]["quota"], "quota_binds": table[c]["level"] is not None,
            "tissue_rows": table[c]["tissue_rows"], "pool_rows": table[c]["pool_rows"], "level": table[c]["level"],
            "patients": {str(p): v for p, v in table[c]["patients"].items()}} for c in range(K)},
        "tissue": {"stage_dir": "prep/tissue", "tau_deg": tau, "method_used": tau_doc.get("method_used"),
                   "mask_rule": tissue.MASK_RULE, "complete": tissue_complete,
                   "expect_tau_deg": cfg.expect_tau_deg, "tau_signed_off": cfg.expect_tau_deg is not None,
                   "n_tissue_rows": int(tissue_rows.size)},
        "log_space": {"definition": "y = (log(max(x, 0) + eps) - mu_b) / sigma_b, stored float16",
                      "eps": LOG_EPS, "fitted_on": "every train tissue row (float64, Chan et al. merge)",
                      "n_rows": int(tissue_rows.size), "stats": stats.to_dict(),
                      "mu": codec.mu.tolist(), "sigma": codec.sigma.tolist()},
        "clip": {"quantile": CLIP_QUANTILE, "fitted_on": "every train tissue row",
                 "method": "exact per-band float16 histogram, numpy 'inverted_cdf' quantile",
                 "p99_99": p_hi.tolist(), "factor": CLIP_HI_FACTOR, "clip_hi": codec.clip_hi.tolist()},
        "encoded": {"mean": enc_stats.mean.tolist(), "std": enc_stats.std().tolist(),
                    "min": enc_stats.min.tolist(), "max": enc_stats.max.tolist()},
        "round_trip": round_trip,
        "folds": {"stage_dir": "prep/leakage", "file": splits.FOLDS_NAME, "seed": folds_doc.get("seed"),
                  "n_folds": len(folds), "membership": "splits.fold_masks(patient_pool, fold)", "folds": fold_rows},
        "reference": {
            "file": REFERENCE_NAME, "schema_version": REFERENCE_SCHEMA_VERSION, "quantile_levels": levels,
            "pca": {"fitted_on": "the encoded pool rows (log-z), flattened (H, W, C)", "dim": D,
                    "n_components": k, "n_fit": n_fit, "seed_key": PCA_RNG_KEY if n_fit < n_pool else None,
                    "explained_variance_ratio_sum": float(pca["explained_variance_ratio"].sum())},
            "nn": {"rule": NN_RULE, "max_queries": cfg.nn_max_queries, "n_queries": int(ref["nn_query"].size),
                   "n_duplicates": int(dups.size),
                   "duplicates": [{"fold": int(ref["nn_fold"][i]), "query_row": int(rows[ref["nn_query"][i]]),
                                   "match_row": int(rows[ref["nn_match"][i]])} for i in dups[:50]],
                   "pooled_quantiles": {class_names[c]: dict(zip((f"q{100 * lv:g}" for lv in levels),
                                                                 ref["nn_quantiles_pooled"][c].tolist()))
                                        for c in range(K)}},
            "features": {"function": "hsi_augment.provenance.patch_feats on the raw patch (as index_train.npz)",
                         "layout": f"[:, :{n_bands}] per-band mean, [:, {n_bands}:] per-band std (ddof 0)",
                         "sides": list(SIDES)},
            "set_level_metrics": "computed by the QC library (#28), not here",
        },
    }
    ctx.save_json(POOL_META_NAME, _json_safe(meta))

    # 9. The pool itself, X_pool.npy last.
    with ctx.timer("save"):
        write_pool(ctx.stage_dir, X_pool, pool_y, pool_patient, rows, pool_capture, codec)
        for name in (CODEC_NAME, *(POOL_FILES[k_] for k_ in ("y", "patient", "row", "capture", "X"))):
            ctx.add_output(ctx.stage_dir / name)
    ctx.log(f"pool written: {n_pool:,} rows, X_pool.npy {_gb(x_bytes)}")

    return {
        "n_rows": n_pool,
        "rows_per_class": _per_class([table[c]["pool_rows"] for c in range(K)], class_names),
        "patients_per_class": _per_class([sum(1 for v in table[c]["patients"].values() if v["pool_rows"])
                                          for c in range(K)], class_names),
        "x_bytes": x_bytes,
        "tau_deg": tau,
        "tau_signed_off": cfg.expect_tau_deg is not None,
        "tissue_complete": tissue_complete,
        "n_folds": len(folds),
        "round_trip_max_rel_err": round_trip["max_rel_err"],
        "pca_explained_variance_ratio": float(pca["explained_variance_ratio"].sum()),
        "nn_duplicates": int(dups.size),
        "nn_q1_pooled": _json_safe(_per_class(
            [float(ref["nn_quantiles_pooled"][c][levels.index(0.01)]) if 0.01 in levels else None
             for c in range(K)], class_names)),
    }
