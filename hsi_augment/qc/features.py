# -*- coding: utf-8 -*-
"""
hsi_augment/qc/features.py
==========================
Per-patch QC features and the two feature spaces (Plan §Phase 8b, issue #28).

Every check runs on the export
------------------------------
Generated samples are exported as float16 NHWC in value space (#7's schema),
and that file, not the generator's float32 output, is what the classifier
trains on. `as_export` therefore casts whatever it is given to float16 FIRST
and every feature is computed from that cast, widened to float32 (float64
where a sum needs it). A float32 value above 65504 becomes inf in the cast;
`training/npy_data.py` would later turn such an overflow into 1.0 silently,
so R1 must see the inf, which it only does on the cast.

The feature spaces
------------------
    mean/std   `scripts.shallow_baseline.featurize(x, "mean_std")`: per band, the
               patch's mean and std over its H x W pixels (2C values), then each of
               the 2C features z-scored with the reference sample's statistics. It
               is the space of the independent QC classifier (R5), of the kNN label
               agreement, and of one set of the set-level metrics.
    PCA-64     every value z-scored with its band's train mean/std, the patch
               flattened (H*W*C = 3,872 values on trainsel) and projected on the
               top 64 principal components of the reference sample. It is the
               coarse search space of the memorization check (R6) and the other
               space of the set-level metrics. The exact distance (`zflat`
               distance, root mean square over the flattened values) re-ranks the
               PCA candidates.

PCA is fitted from a covariance accumulated chunk by chunk in RAW value space
(`CovarianceAccumulator`, float64, shifted by the first block's mean so the
second moment does not cancel); the z-scoring is applied to the finished
covariance (cov_z = cov_x / s s^T), so the band statistics and the PCA come
out of ONE pass over the rows and no sample matrix is ever held in RAM.

Spectral and spatial scores
---------------------------
* curvature (R2): second differences of log(x + eps) along the band axis,
  per pixel, within each contiguous band segment (`adapter.spectral_segments()`),
  never across the 629 -> 851 nm gap; the mean of their squares. A scalar gain
  adds a constant to log x and a tilt that is linear in the band index adds a
  line, so neither changes it: it measures jaggedness (too high) and
  over-smoothing (too low) of the per-pixel spectra.
* lag-1 autocorrelation and high-frequency power fraction (R3): per band, on
  the patch with its band mean removed, each normalised by the band's own
  variance (so a gain cancels), then averaged over the bands whose variance is
  not zero. A blurred patch has a higher autocorrelation and less power above
  `hf_cutoff` cycles/pixel; a flat patch has no variance at all and scores
  NaN, which every thresholded check rejects.
  The radial spectrum is the DCT-II one (orthonormal; coefficient k has
  frequency k / 2N cycles/pixel), not the FFT's. The FFT treats an 11 x 11
  patch as periodic, so any gradient across it becomes a jump at the wrap-around
  and spreads power over every frequency; a smooth, blurred patch with a ramp
  then shows "high-frequency" power and passes. The DCT's implicit mirror
  extension has no such jump. With R3 calibrated to [q0.5, q99.5] on 300 sharp
  textured 11 x 11 patches, the FFT version let 0.5-4.5% of 200 other sharp
  patches' Gaussian (sigma 1, 1.5) or 3 x 3 box-blurred copies through, the
  DCT version none (the in-test patches of tests/test_hsi_augment_qc.py).
* R1's per-band range uses an exact float16 histogram (`code_histogram`): a
  float16 has 65,536 bit patterns, so per-band counts of the codes give EXACT
  quantiles of any number of rows in 2 MB per band set, with no value sample.

SAM here is unclamped (`sam_deg`): `training.spectral_recon_metrics_v2.sam_deg`
and the frozen `training.gan.SAMLoss` clamp cos at 1 - 1e-3, so identical
spectra would score 2.56 degrees instead of 0.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOG_EPS = 1e-3                    # as hsi_augment.codec.LOG_EPS: log(x + eps) of value-space x
N_CODES = 1 << 16                 # float16 bit patterns
BLOCK_ROWS = 256                  # rows widened to float32/float64 at a time
HASH_BYTES = 16                   # blake2b digest: two uint64 words
MIN_STD = 1e-6                    # a feature/band std at or below this is replaced by 1

# float16 code -> its value, and the finite codes in ascending value order
# (-0.0 and +0.0 compare equal; the stable sort keeps them adjacent).
_CODE_VALUES = np.arange(N_CODES, dtype=np.uint16).view(np.float16).astype(np.float64)
_FINITE_CODES = np.flatnonzero(np.isfinite(_CODE_VALUES))
_CODE_ORDER = _FINITE_CODES[np.argsort(_CODE_VALUES[_FINITE_CODES], kind="stable")]
_ORDERED_VALUES = _CODE_VALUES[_CODE_ORDER]


# ============================================================================
# the export cast
# ============================================================================

def as_export(x: Any) -> np.ndarray:
    """`x` as the export stores it: float16, C-contiguous, NHWC (n, H, W, C).
    Overflow becomes +-inf here, on purpose (R1 must see it)."""
    a = np.asarray(x)
    if a.ndim != 4:
        raise ValueError(f"expected NHWC patches (n, H, W, C), got shape {a.shape}")
    if not (np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.integer)):
        raise TypeError(f"expected a numeric array, got {a.dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.ascontiguousarray(a, dtype=np.float16)


def blocks(n: int, size: int = BLOCK_ROWS) -> Iterable[Tuple[int, int]]:
    for a in range(0, n, size):
        yield a, min(a + size, n)


# ============================================================================
# per-patch scalar features
# ============================================================================

def nonfinite_fraction(x: np.ndarray) -> np.ndarray:
    """Fraction of a patch's values that are NaN or +-inf (float64 [n])."""
    x = np.asarray(x)
    return (~np.isfinite(x)).reshape(len(x), -1).mean(axis=1)


def out_of_range_fraction(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Fraction of a patch's values outside its band's [lo, hi] (a non-finite
    value counts as outside). float64 [n]."""
    x = np.asarray(x, dtype=np.float32)
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    out = np.empty(len(x), dtype=np.float64)
    for a, b in blocks(len(x)):
        with np.errstate(invalid="ignore"):
            inside = (x[a:b] >= lo) & (x[a:b] <= hi)
        out[a:b] = 1.0 - inside.reshape(b - a, -1).mean(axis=1)
    return out


def mean_std_features(x: np.ndarray) -> np.ndarray:
    """`scripts.shallow_baseline.featurize(x, "mean_std")` (frozen; imported,
    never copied): [per-band mean, per-band std] over the H x W pixels,
    float32 [n, 2C]."""
    from scripts import shallow_baseline as sb
    x = np.asarray(x)
    out = np.empty((len(x), 2 * x.shape[-1]), dtype=np.float32)
    for a, b in blocks(len(x)):
        out[a:b] = sb.featurize(np.asarray(x[a:b], dtype=np.float32), "mean_std")
    return out


def log_spectrum(mean_spectra: np.ndarray, eps: float = LOG_EPS) -> np.ndarray:
    """log(max(m, 0) + eps), float64: the patch mean spectrum in log space."""
    m = np.asarray(mean_spectra, dtype=np.float64)
    return np.log(np.maximum(m, 0.0) + eps)


def curvature_energy(x: np.ndarray, segments: Sequence[Tuple[int, int]], eps: float = LOG_EPS) -> np.ndarray:
    """Mean squared second difference of log(x + eps) along the bands, per
    pixel, WITHIN each segment [a, b) of `segments` (a segment of fewer than 3
    bands has none); never across a gap. float64 [n]; NaN if no segment has 3
    bands or the patch is not finite."""
    x = np.asarray(x)
    segs = [(int(a), int(b)) for a, b in segments if int(b) - int(a) >= 3]
    out = np.full(len(x), np.nan)
    if not segs:
        return out
    for a, b in blocks(len(x)):
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.log(np.maximum(np.asarray(x[a:b], dtype=np.float64), 0.0) + eps)
        tot = np.zeros(b - a)
        cnt = 0
        for s0, s1 in segs:
            s = z[..., s0:s1]
            d2 = s[..., 2:] - 2.0 * s[..., 1:-1] + s[..., :-2]
            tot += (d2 * d2).reshape(b - a, -1).sum(axis=1)
            cnt += int(np.prod(d2.shape[1:]))
        out[a:b] = tot / cnt
    return out


def _band_variance_valid(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """A band has spatial variance when it is clearly above float64 rounding of
    its mean (a constant band computes to ~1e-32, not exactly 0)."""
    return v > (1e-9 * np.abs(m)) ** 2 + 1e-20


def spatial_stats(x: np.ndarray, hf_cutoff: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """(lag-1 autocorrelation, high-frequency power fraction), float64 [n] each.

    Per band, on the patch minus its band mean: the autocorrelation is the mean
    of the horizontal and vertical lag-1 products over the variance; the HF
    fraction is the 2-D DCT-II power above `hf_cutoff` cycles/pixel (radial
    frequency, k / 2N per axis) over all non-DC power. Both are averaged over
    the bands with spatial variance; NaN when no band has any (a flat patch) or
    the patch is not finite."""
    from scipy.fft import dctn
    x = np.asarray(x)
    n, h, w = x.shape[0], x.shape[1], x.shape[2]
    fy = (np.arange(h) / (2.0 * h))[:, None]
    fx = (np.arange(w) / (2.0 * w))[None, :]
    rho = np.sqrt(fy * fy + fx * fx)
    hi_mask = rho > hf_cutoff
    ac_mask = rho > 0
    lag1 = np.full(n, np.nan)
    hf = np.full(n, np.nan)
    for a, b in blocks(n):
        blk = np.asarray(x[a:b], dtype=np.float64)
        finite = np.isfinite(blk).reshape(b - a, -1).all(axis=1)
        blk = np.where(np.isfinite(blk), blk, 0.0)
        m = blk.mean(axis=(1, 2), keepdims=True)
        d = blk - m
        v = (d * d).mean(axis=(1, 2))                                   # (k, C)
        ch = (d[:, :, 1:, :] * d[:, :, :-1, :]).mean(axis=(1, 2))
        cv = (d[:, 1:, :, :] * d[:, :-1, :, :]).mean(axis=(1, 2))
        valid = _band_variance_valid(v, m[:, 0, 0, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(valid, 0.5 * (ch + cv) / np.where(valid, v, 1.0), np.nan)
            p = dctn(d, type=2, axes=(1, 2), norm="ortho") ** 2         # (k, H, W, C)
            num = p[:, hi_mask, :].sum(axis=1)
            den = p[:, ac_mask, :].sum(axis=1)
            frac = np.where(valid & (den > 0), num / np.where(den > 0, den, 1.0), np.nan)
            any_valid = valid.any(axis=1)
            r_mean = np.where(any_valid, np.nansum(r, axis=1) / np.maximum(valid.sum(axis=1), 1), np.nan)
            f_ok = np.isfinite(frac)
            f_mean = np.where(f_ok.any(axis=1), np.nansum(frac, axis=1) / np.maximum(f_ok.sum(axis=1), 1),
                              np.nan)
        lag1[a:b] = np.where(finite, r_mean, np.nan)
        hf[a:b] = np.where(finite, f_mean, np.nan)
    return lag1, hf


def sam_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spectral angle (degrees) between matching rows of `a` and `b` (..., C),
    UNCLAMPED: atan2(|a minus its projection on b|, a . b_hat), so identical
    spectra are 0 and sub-degree angles stay accurate. NaN for a zero or
    non-finite spectrum."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        bh = b / nb
        proj = (a * bh).sum(axis=-1)
        resid = a - proj[..., None] * bh
        ang = np.degrees(np.arctan2(np.linalg.norm(resid, axis=-1), proj))
    bad = ~(np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)) | \
        ~(np.linalg.norm(a, axis=-1) > 0) | ~(nb[..., 0] > 0)
    return np.where(bad, np.nan, ang)


# ============================================================================
# exact byte hashes
# ============================================================================

def patch_hashes(x16: np.ndarray) -> np.ndarray:
    """blake2b-128 of each patch's float16 bytes, as (n, 2) uint64. The input
    is cast to the export dtype first, so a float32 patch hashes like its
    float16 export. Two patches match iff their stored bytes are identical
    (collision odds ~1e-26 at the real build's size)."""
    x16 = np.ascontiguousarray(as_export(x16) if np.asarray(x16).dtype != np.float16 else x16)
    out = np.empty((len(x16), 2), dtype=np.uint64)
    for i in range(len(x16)):
        out[i] = np.frombuffer(hashlib.blake2b(x16[i], digest_size=HASH_BYTES).digest(), dtype="<u8")
    return out


# ============================================================================
# exact float16 quantiles (R1)
# ============================================================================

def code_histogram(x16: np.ndarray) -> np.ndarray:
    """Per-band counts of the 65,536 float16 codes: int64 [C, 65536]."""
    codes = np.ascontiguousarray(x16, dtype=np.float16).view(np.uint16)
    c = codes.shape[-1]
    flat = codes.reshape(-1, c)
    return np.stack([np.bincount(flat[:, j], minlength=N_CODES) for j in range(c)]).astype(np.int64)


def histogram_quantiles(counts: np.ndarray, tail: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-band [lo, hi] = the `tail` and 1 - `tail` quantiles of the finite
    values counted in `counts` [C, 65536] (inverted CDF: actual float16
    values; tail 0 gives min and max). float64 [C] each; NaN for an empty band."""
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts < 0):
        raise ValueError("negative histogram counts (held-out counts larger than the total?)")
    c = counts.shape[0]
    lo = np.full(c, np.nan)
    hi = np.full(c, np.nan)
    for j in range(c):
        cum = np.cumsum(counts[j, _CODE_ORDER])
        n = int(cum[-1]) if len(cum) else 0
        if n == 0:
            continue
        r_lo = max(1, int(np.ceil(tail * n)))
        r_hi = max(1, int(np.ceil((1.0 - tail) * n)))
        lo[j] = _ORDERED_VALUES[np.searchsorted(cum, r_lo, side="left")]
        hi[j] = _ORDERED_VALUES[np.searchsorted(cum, r_hi, side="left")]
    return lo, hi


# ============================================================================
# covariance, PCA and the feature spaces
# ============================================================================

class CovarianceAccumulator:
    """Mean and covariance of row vectors, accumulated block by block in
    float64 around a fixed shift (the first block's mean), so E[xx^T] - E[x]^2
    does not cancel catastrophically."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.shift: Optional[np.ndarray] = None
        self.s1 = np.zeros(self.dim)
        self.s2 = np.zeros((self.dim, self.dim))

    def update(self, rows: np.ndarray) -> "CovarianceAccumulator":
        r = np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)
        if r.shape[1] != self.dim:
            raise ValueError(f"rows have {r.shape[1]} values, expected {self.dim}")
        if len(r) == 0:
            return self
        if self.shift is None:
            self.shift = r.mean(axis=0)
        d = r - self.shift
        self.n += len(r)
        self.s1 += d.sum(axis=0)
        self.s2 += d.T @ d
        return self

    def mean(self) -> np.ndarray:
        return self.shift + self.s1 / self.n

    def cov(self) -> np.ndarray:
        if self.n < 2:
            raise ValueError("a covariance needs at least 2 rows")
        c = (self.s2 - np.outer(self.s1, self.s1) / self.n) / (self.n - 1)
        return 0.5 * (c + c.T)


def top_components(cov: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """(components [k, D] float64, eigenvalues [k]) of a covariance, largest
    first. Signs are fixed (largest-magnitude loading positive) so a refit on
    the same data gives the same basis."""
    from scipy.linalg import eigh
    d = cov.shape[0]
    k = int(min(k, d))
    vals, vecs = eigh(cov, subset_by_index=[d - k, d - 1])
    order = np.argsort(vals)[::-1]
    vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order].T
    flip = np.sign(vecs[np.arange(k), np.argmax(np.abs(vecs), axis=1)])
    return vecs * np.where(flip == 0, 1.0, flip)[:, None], vals


@dataclass
class FeatureSpaces:
    """The fitted spaces. Arrays only, so it round-trips through one .npz."""
    patch_shape: Tuple[int, int, int]           # (H, W, C)
    band_mean: np.ndarray                       # [C] float64, over the reference sample's pixels
    band_std: np.ndarray                        # [C] float64 (floored at MIN_STD -> 1)
    pca_mean: np.ndarray                        # [D] float64, mean of the z-scored flattened patch
    pca_components: np.ndarray                  # [k, D] float32
    pca_explained_variance: np.ndarray          # [k] float64 (z units squared)
    pca_total_variance: float                   # trace of the z covariance
    msf_mean: np.ndarray                        # [2C] float64, mean/std feature means
    msf_std: np.ndarray                         # [2C] float64

    @property
    def dim(self) -> int:
        h, w, c = self.patch_shape
        return h * w * c

    @property
    def n_components(self) -> int:
        return int(self.pca_components.shape[0])

    def check_shape(self, x: np.ndarray) -> None:
        if tuple(np.asarray(x).shape[1:]) != tuple(self.patch_shape):
            raise ValueError(f"patches have shape {tuple(np.asarray(x).shape[1:])}, the QC reference was "
                             f"built for {tuple(self.patch_shape)} (NHWC)")

    def zflat(self, x: np.ndarray) -> np.ndarray:
        """Values z-scored per band with the train stats, flattened: float32 [n, D]."""
        x = np.asarray(x, dtype=np.float32)
        z = (x - self.band_mean.astype(np.float32)) / self.band_std.astype(np.float32)
        return z.reshape(len(x), -1)

    def pca(self, x: np.ndarray) -> np.ndarray:
        """PCA-k coordinates, float32 [n, k]."""
        out = np.empty((len(x), self.n_components), dtype=np.float32)
        comps = self.pca_components
        mean = self.pca_mean.astype(np.float32)
        for a, b in blocks(len(x)):
            out[a:b] = (self.zflat(x[a:b]) - mean) @ comps.T
        return out

    def msf(self, raw: np.ndarray) -> np.ndarray:
        """Standardised mean/std features from raw `mean_std_features` output."""
        return ((np.asarray(raw, dtype=np.float64) - self.msf_mean) / self.msf_std).astype(np.float32)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {"patch_shape": np.asarray(self.patch_shape, dtype=np.int64), "band_mean": self.band_mean,
                "band_std": self.band_std, "pca_mean": self.pca_mean, "pca_components": self.pca_components,
                "pca_explained_variance": self.pca_explained_variance,
                "pca_total_variance": np.asarray(self.pca_total_variance), "msf_mean": self.msf_mean,
                "msf_std": self.msf_std}

    @classmethod
    def from_arrays(cls, d: Mapping[str, np.ndarray]) -> "FeatureSpaces":
        return cls(patch_shape=tuple(int(v) for v in d["patch_shape"]),
                   band_mean=np.asarray(d["band_mean"], dtype=np.float64),
                   band_std=np.asarray(d["band_std"], dtype=np.float64),
                   pca_mean=np.asarray(d["pca_mean"], dtype=np.float64),
                   pca_components=np.asarray(d["pca_components"], dtype=np.float32),
                   pca_explained_variance=np.asarray(d["pca_explained_variance"], dtype=np.float64),
                   pca_total_variance=float(np.asarray(d["pca_total_variance"])),
                   msf_mean=np.asarray(d["msf_mean"], dtype=np.float64),
                   msf_std=np.asarray(d["msf_std"], dtype=np.float64))

    def describe(self) -> Dict[str, Any]:
        ev = self.pca_explained_variance
        return {"patch_shape": list(self.patch_shape), "dim": self.dim, "pca_components": self.n_components,
                "pca_explained_variance_ratio": float(ev.sum() / self.pca_total_variance)
                if self.pca_total_variance > 0 else None,
                "mean_std_dim": int(self.msf_mean.shape[0])}


def fit_feature_spaces(patch_shape: Sequence[int], band_mean: np.ndarray, band_std: np.ndarray,
                       raw_cov: CovarianceAccumulator, msf_raw: np.ndarray, n_components: int) -> FeatureSpaces:
    """The spaces from one pass's accumulators: per-band stats, the RAW
    flattened covariance, and the raw mean/std features of the sample."""
    h, w, c = (int(v) for v in patch_shape)
    std = np.where(np.asarray(band_std) > MIN_STD, band_std, 1.0)
    s_vec = np.tile(std, h * w)                     # flattened NHWC: band is the fastest axis
    m_vec = np.tile(np.asarray(band_mean, dtype=np.float64), h * w)
    cov_z = raw_cov.cov() / np.outer(s_vec, s_vec)
    k = int(min(n_components, cov_z.shape[0], max(raw_cov.n - 1, 1)))
    comps, ev = top_components(cov_z, k)
    msf = np.asarray(msf_raw, dtype=np.float64)
    msf_std = msf.std(axis=0)
    return FeatureSpaces(patch_shape=(h, w, c), band_mean=np.asarray(band_mean, dtype=np.float64),
                         band_std=std.astype(np.float64), pca_mean=(raw_cov.mean() - m_vec) / s_vec,
                         pca_components=comps.astype(np.float32), pca_explained_variance=ev,
                         pca_total_variance=float(np.trace(cov_z)), msf_mean=msf.mean(axis=0),
                         msf_std=np.where(msf_std > MIN_STD, msf_std, 1.0))


def segments_list(segments: Iterable[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(a), int(b)) for a, b in segments]
