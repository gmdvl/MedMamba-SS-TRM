# -*- coding: utf-8 -*-
"""
hsi_augment/transforms.py
=========================
Physics-motivated spectral augmentation (Plan §Phase 5, issue #19): three
label-preserving ops on a normalized `[C, H, W]` patch, the helper that runs
them after the existing augmentation, and a report that measures how far each
op moves the data.

This is a library, not a stage. `train.py` imports it (`--phys_aug`, #20), so
the module-level imports are numpy and torch only. sklearn (the NMF fit),
matplotlib (the montage), `training.augmentation` (the report's per-item seeds)
and `hsi_augment.adapters` (gap detection for plots) are imported inside the
functions that use them.

Value space
-----------
Under `--normalization global_zscore` the classifier sees x_n = (x - mu) / sigma,
per band. The physics lives in x, the build's value space (raw counts / corpus
median, NOT reflectance). Each op therefore runs as

    x = x_n * sigma + mu  ->  op(x)  ->  (x' - mu) / sigma

with `mu`/`sigma` given at construction. They must be the very arrays the
dataset normalized with (`global_mean, global_std` in `train_pipeline.py`, the
`GLOBAL_STATS` sample estimate), not the exact `prep/norm_stats.json`: the
round trip is only a round trip with the same affine map.

The ops
-------
spectral_tilt     x * (1 + a P1(l) + b P2(l)); P1, P2 are the Legendre
                  polynomials of the REAL wavelengths rescaled to l in [-1, 1],
                  and a, b ~ U(-tilt_max, tilt_max) once per sample. A smooth
                  illumination or stain-tint change. The bands are not uniform
                  (a 629 -> 851 nm gap on trainsel); a curve that is smooth in
                  wavelength is physically fine across the gap, whereas one
                  that is smooth in band INDEX would put a kink there. Since
                  |P1|, |P2| <= 1 the factor is >= 1 - 2 tilt_max, so
                  tilt_max < 0.5 keeps it positive and the sign of x.
sensor_noise      x + N(0, (k sigma_b)^2), sigma_b the per-band noise of
                  Plan P2.2 (#12: robust std of the 4-neighbour Laplacian
                  residual, value units). Homoscedastic: shot noise would scale
                  with sqrt(x), but sigma_b is the one estimate the build gives.
                  The result is clamped at 0 (a sensor count cannot be
                  negative). The clamp is the only clip in this module; it also
                  lifts any value the base augmentation already pushed below 0.
od_stain_jitter   EXPERIMENTAL, off in every arm. Optical density
                  OD = -log(clip(x / s_glass, eps)) against the glass reference
                  (Beer-Lambert in transmission: the light through bare glass is
                  I0, so OD ~ sum_k c_k A_k(l) over the stains). OD is projected
                  onto K NMF components A_k (`fit_od_components`, fitted on
                  train tissue OD), each coefficient is scaled by U(1 +- delta)
                  once per sample, and the result is mapped back. Assumptions:
                  (1) transmission Beer-Lambert holds, i.e. scattering is small
                  or itself absorbance-like; (2) `s_glass` is the right I0 for
                  this patch, in shape AND level - a wrong level adds a flat
                  log-offset to OD, which NMF can absorb into a "grey"
                  component, and jittering that is a global gain change (dropped
                  below on purpose); (3) K components span the stains; the
                  part of OD they do not explain is kept unchanged, never
                  jittered.

How od_stain_jitter maps back. Least squares gives the coefficients,
w = OD pinv(A), clamped at 0 (a stain amount is non-negative; a negative fit is
noise and would flip the jitter's sign). The perturbed density is
OD' = OD + ((u - 1) * w) A, and since x = s_glass exp(-OD),

    x' = s_glass exp(-OD') = x exp(-((u - 1) * w) A).

The op uses the right-hand form: it never re-introduces the `eps` clip into x,
it keeps the sign of x, and at delta = 0 the factor is exp(0) = 1 exactly, so
the value-space round trip is the identity bit for bit. `optical_density` /
`od_to_value` are the literal transforms, for fitting and for tests.

Randomness
----------
Every op has the frozen op signature, `(patch, rng: random.Random,
gen: torch.Generator | None)` (`training/augmentation.py`): scalars (a, b, the
K jitter factors) come from `rng`, noise tensors from `gen`. Both are the
per-item pair `AugmentedPatchDatasetV16.__getitem__` derives from
`item_augmentation_seed(seed, epoch, index, worker)`, so every draw is a
deterministic function of that tuple.

`compose_with_physics(base_fn, ops)` calls `base_fn` FIRST and lets the ops
draw from the same pair AFTER it. The base augmentation therefore sees exactly
the stream it sees without physics, and with no ops `compose_with_physics`
returns `base_fn` itself: output and stream are bit-identical by construction.
A switched-off op (magnitude <= 0) is never built, so it consumes no draws
either. The composed function receives the patch only; the `norm_stats`
element `AugmentedPatchDatasetV16` passes through is never seen, let alone
touched. Under `global_zscore` those stats are the same global (mu, sigma) for
every patch, so they still de-normalize the augmented patch correctly.

Order: od_stain_jitter -> spectral_tilt -> sensor_noise. The sample's stains,
then the light through it, then the sensor reading it.

Dropped on purpose
------------------
* Global gain. The build's median-ratio correction (`gain = corpus_median /
  capture_median`) already removes per-capture brightness, and per-band scale
  is in the paper recipe (`spectral_scale_range`).
* Band dropout and spectral masking. Zeroed bands are not a physical state of
  this sensor, and they break the band-band relationships the spectral model
  reads.
* Wavelength shift. It needs interpolation between bands, which is invalid
  across the 629 -> 851 nm gap and smears the non-uniform grid elsewhere.
* Arbitrary rotations and elastic warps. They interpolate between pixels, i.e.
  they mix spectra; flips and rot90 (already in the recipe) do not.
* Inter-class mixup. A convex mix of two classes' spectra has no single label;
  intra-class mixing lives in `training/minority_augmentation.py`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]
AugmentFn = Callable[..., torch.Tensor]

DEFAULT_TILT_MAX = 0.03
DEFAULT_NOISE_K = 1.0
DEFAULT_OD_DELTA = 0.1          # the op's own default; the builder's is 0.0 (off in every arm)
DEFAULT_OD_K = 3
OD_EPS = 1e-3                   # x / s_glass floor before the log: OD <= -ln(1e-3) ~ 6.9
MAX_TILT = 0.5                  # 1 + a P1 + b P2 >= 1 - 2 tilt_max > 0 below this
GLASS_PERCENTILE = 99.0         # `estimate_glass_level`: the brightest spectra are bare glass

# The order ops are built and applied in (see the module docstring).
OP_ORDER = ("od_stain_jitter", "spectral_tilt", "sensor_noise")

# `load_phys_artifacts`: key (= `build_physics_ops` keyword) -> file name.
# noise_sigma.npy is #12's output; the two OD files are written by whoever fits
# the experimental OD model with `fit_od_components` / `glass_reference`.
PHYS_ARTIFACT_FILES = {
    "noise_sigma": "noise_sigma.npy",      # float32 [C], value units
    "od_components": "od_components.npy",  # float32 [K, C], unit-norm rows
    "s_glass": "od_s_glass.npy",           # float32 [C], glass reference in VALUE units
}
# Searched in this order under `--phys_artifacts DIR`; the first hit wins, so
# DIR may be the stage output itself, its parent, or the pipeline's --out.
PHYS_ARTIFACT_SUBDIRS = ("", "spectral", "prep", "prep/spectral")

# Report defaults.
REPORT_PERCENTILES = (5.0, 50.0, 95.0, 99.0)
MONTAGE_RGB_NM = (640.0, 550.0, 460.0)


# ============================================================================
# Small helpers
# ============================================================================

def _band_vector(x: ArrayLike, name: str) -> np.ndarray:
    """1-D finite float64 copy of a per-band array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} has non-finite values")
    return v


def _column(v: np.ndarray) -> torch.Tensor:
    """[C] float64 numpy -> [C, 1, 1] float32 torch, broadcasting over H, W."""
    return torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).view(-1, 1, 1)


def _magnitude(value: Optional[float], name: str) -> float:
    """A finite float; None counts as 0 (off)."""
    v = 0.0 if value is None else float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def legendre_coordinate(wavelengths: ArrayLike) -> np.ndarray:
    """The wavelengths (nm) mapped affinely onto [-1, 1]: min -> -1, max -> +1."""
    wl = _band_vector(wavelengths, "wavelengths")
    lo, hi = float(wl.min()), float(wl.max())
    if not hi > lo:
        raise ValueError(f"wavelengths must span a range, got min = max = {lo}")
    return 2.0 * (wl - lo) / (hi - lo) - 1.0


def legendre_basis(wavelengths: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """(P1, P2) evaluated at `legendre_coordinate(wavelengths)`, float64 [C]."""
    t = legendre_coordinate(wavelengths)
    return t, 0.5 * (3.0 * t * t - 1.0)


def sam_degrees(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Spectral angle between `a` and `b` along `axis`, in degrees, float64.

    Not `training.spectral_recon_metrics_v2.sam_deg`: that clamps cos to
    1 - 1e-3, a 2.56 degree floor, which would make a 2 degree bound untestable.
    A zero spectrum against anything is reported as 0.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    cos = np.divide(dot, den, out=np.ones_like(dot), where=den > 0)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# ============================================================================
# Value space
# ============================================================================

class ValueSpace:
    """The per-band `global_zscore` map, x = x_n * std + mean, and its inverse."""

    def __init__(self, global_mean: ArrayLike, global_std: ArrayLike):
        mean = _band_vector(global_mean, "global_mean")
        std = _band_vector(global_std, "global_std")
        if mean.shape != std.shape:
            raise ValueError(f"global_mean has {mean.size} bands, global_std {std.size}")
        if not np.all(std > 0):
            raise ValueError("global_std must be > 0 in every band")
        self.n_bands = int(mean.size)
        self.mean = _column(mean)
        self.std = _column(std)

    def _stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # `.to` is a no-op (same object) for the usual float32 CPU patch
        return (self.mean.to(device=x.device, dtype=x.dtype),
                self.std.to(device=x.device, dtype=x.dtype))

    def to_value(self, x_n: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(x_n)
        return x_n * std + mean

    def to_normalized(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(x)
        return (x - mean) / std


# ============================================================================
# The ops
# ============================================================================

class PhysicsOp:
    """One op: `op(patch, rng, gen)` on a normalized [C, H, W] tensor, defined
    by `apply_value` on the value-space tensor. Subclasses set `name`."""

    name = "physics_op"

    def __init__(self, global_mean: ArrayLike, global_std: ArrayLike):
        self.space = ValueSpace(global_mean, global_std)

    @property
    def n_bands(self) -> int:
        return self.space.n_bands

    def _check_bands(self, n: int, what: str) -> None:
        if n != self.n_bands:
            raise ValueError(f"{self.name}: {what} has {n} bands, global_mean/global_std {self.n_bands}")

    def apply_value(self, x: torch.Tensor, rng: random.Random,
                    gen: Optional[torch.Generator] = None) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, patch: torch.Tensor, rng: random.Random,
                 gen: Optional[torch.Generator] = None) -> torch.Tensor:
        if patch.ndim != 3 or patch.shape[0] != self.n_bands:
            raise ValueError(f"{self.name}: expected a [{self.n_bands}, H, W] patch, "
                             f"got {tuple(patch.shape)}")
        # float16/bfloat16 patches are computed in float32 and cast back
        work = patch if patch.dtype in (torch.float32, torch.float64) else patch.float()
        x = self.apply_value(self.space.to_value(work), rng, gen)
        return self.space.to_normalized(x).to(dtype=patch.dtype)

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.describe().items() if k != "op")
        return f"{type(self).__name__}({params})"


class SpectralTilt(PhysicsOp):
    """x * (1 + a P1(l) + b P2(l)), a, b ~ U(-tilt_max, tilt_max) per sample."""

    name = "spectral_tilt"

    def __init__(self, wavelengths: ArrayLike, global_mean: ArrayLike, global_std: ArrayLike,
                 tilt_max: float = DEFAULT_TILT_MAX):
        super().__init__(global_mean, global_std)
        tilt_max = _magnitude(tilt_max, "tilt_max")
        if not 0.0 <= tilt_max < MAX_TILT:
            raise ValueError(f"tilt_max must be in [0, {MAX_TILT}) so the factor stays positive, "
                             f"got {tilt_max}")
        wl = _band_vector(wavelengths, "wavelengths")
        self._check_bands(wl.size, "wavelengths")
        p1, p2 = legendre_basis(wl)
        self.tilt_max = tilt_max
        self.wavelength_range_nm = (float(wl.min()), float(wl.max()))
        self._p1, self._p2 = _column(p1), _column(p2)

    def draw(self, rng: random.Random) -> Tuple[float, float]:
        t = self.tilt_max
        return rng.uniform(-t, t), rng.uniform(-t, t)

    def factor(self, a: float, b: float) -> torch.Tensor:
        """The [C, 1, 1] multiplier for coefficients (a, b)."""
        return 1.0 + a * self._p1 + b * self._p2

    def apply_value(self, x, rng, gen=None):
        a, b = self.draw(rng)
        return x * self.factor(a, b).to(device=x.device, dtype=x.dtype)

    def describe(self):
        return {"op": self.name, "tilt_max": self.tilt_max, "basis": "legendre P1, P2 of wavelength",
                "wavelength_range_nm": list(self.wavelength_range_nm)}


class SensorNoise(PhysicsOp):
    """x + N(0, (k sigma_b)^2) per element, clamped at 0."""

    name = "sensor_noise"

    def __init__(self, noise_sigma: ArrayLike, global_mean: ArrayLike, global_std: ArrayLike,
                 k: float = DEFAULT_NOISE_K):
        super().__init__(global_mean, global_std)
        k = _magnitude(k, "noise_k")
        if k < 0:
            raise ValueError(f"noise_k must be >= 0, got {k}")
        sigma = _band_vector(noise_sigma, "noise_sigma")
        self._check_bands(sigma.size, "noise_sigma")
        if np.any(sigma < 0):
            raise ValueError("noise_sigma must be >= 0 in every band")
        self.k = k
        self.sigma_b = sigma.astype(np.float32)
        self._scale = _column(k * sigma)

    def apply_value(self, x, rng, gen=None):
        # drawn in float32 on the CPU whatever x is, so the stream (and the
        # values, up to the cast) do not depend on the patch's dtype or device
        eps = (torch.randn(x.shape, generator=gen, dtype=torch.float32) if gen is not None
               else torch.randn(x.shape, dtype=torch.float32))
        eps = eps.to(device=x.device, dtype=x.dtype)
        return (x + eps * self._scale.to(device=x.device, dtype=x.dtype)).clamp_min(0.0)

    def describe(self):
        return {"op": self.name, "k": self.k, "sigma_b_min": float(self.sigma_b.min()),
                "sigma_b_median": float(np.median(self.sigma_b)), "sigma_b_max": float(self.sigma_b.max())}


class ODStainJitter(PhysicsOp):
    """EXPERIMENTAL. Stain coefficients in optical density scaled by U(1 +- delta).
    See the module docstring for the model, its assumptions and the back-transform."""

    name = "od_stain_jitter"

    def __init__(self, od_components: ArrayLike, s_glass: ArrayLike, global_mean: ArrayLike,
                 global_std: ArrayLike, delta: float = DEFAULT_OD_DELTA, eps: float = OD_EPS):
        super().__init__(global_mean, global_std)
        delta = _magnitude(delta, "od_delta")
        if not 0.0 <= delta < 1.0:
            raise ValueError(f"od_delta must be in [0, 1) so every factor stays positive, got {delta}")
        if not 0.0 < eps < 1.0:
            raise ValueError(f"eps must be in (0, 1), got {eps}")
        comps = (od_components.detach().cpu().numpy() if isinstance(od_components, torch.Tensor)
                 else np.asarray(od_components))
        comps = comps.astype(np.float64)
        if comps.ndim != 2:
            raise ValueError(f"od_components must be [K, C], got shape {comps.shape}")
        self._check_bands(comps.shape[1], "od_components")
        if not np.all(np.isfinite(comps)):
            raise ValueError("od_components has non-finite values")
        if np.linalg.matrix_rank(comps) < comps.shape[0]:
            raise ValueError("od_components are linearly dependent; the projection is undefined")
        ref = _band_vector(s_glass, "s_glass")
        self._check_bands(ref.size, "s_glass")
        if not np.all(ref > 0):
            raise ValueError("s_glass must be > 0 in every band")
        self.delta, self.eps, self.n_components = delta, float(eps), int(comps.shape[0])
        self._comps = torch.from_numpy(comps.astype(np.float32))                      # [K, C]
        self._pinv = torch.from_numpy(np.linalg.pinv(comps).astype(np.float32))      # [C, K]
        self._ref = torch.from_numpy(ref.astype(np.float32))                          # [C]

    def draw(self, rng: random.Random) -> List[float]:
        d = self.delta
        return [rng.uniform(1.0 - d, 1.0 + d) for _ in range(self.n_components)]

    def coefficients(self, od: torch.Tensor) -> torch.Tensor:
        """[P, C] optical density -> [P, K] non-negative stain coefficients."""
        return (od @ self._pinv.to(device=od.device, dtype=od.dtype)).clamp_min(0.0)

    def apply_value(self, x, rng, gen=None):
        factors = self.draw(rng)
        c, h, w = x.shape
        spectra = x.reshape(c, h * w).transpose(0, 1)                               # [P, C]
        ref = self._ref.to(device=x.device, dtype=x.dtype)
        od = -torch.log(torch.clamp(spectra / ref, min=self.eps))
        jitter = torch.tensor(factors, dtype=x.dtype, device=x.device) - 1.0        # [K]
        d_od = (self.coefficients(od) * jitter) @ self._comps.to(device=x.device, dtype=x.dtype)
        out = spectra * torch.exp(-d_od)
        return out.transpose(0, 1).reshape(c, h, w)

    def describe(self):
        return {"op": self.name, "delta": self.delta, "n_components": self.n_components, "eps": self.eps,
                "experimental": True}


# ============================================================================
# Optical density helpers (fitting the experimental op)
# ============================================================================

def optical_density(spectra: np.ndarray, s_glass: ArrayLike, eps: float = OD_EPS) -> np.ndarray:
    """OD = -log(clip(x / s_glass, eps)), bands on the LAST axis, float64."""
    ref = _band_vector(s_glass, "s_glass")
    x = np.asarray(spectra, dtype=np.float64)
    return -np.log(np.clip(x / ref, eps, None))


def od_to_value(od: np.ndarray, s_glass: ArrayLike) -> np.ndarray:
    """The inverse of `optical_density` above its clip: s_glass * exp(-OD)."""
    return _band_vector(s_glass, "s_glass") * np.exp(-np.asarray(od, dtype=np.float64))


def estimate_glass_level(spectra: np.ndarray, percentile: float = GLASS_PERCENTILE) -> float:
    """The `percentile`-th percentile of the per-spectrum band mean ([N, C], value
    units). In a train sample the brightest spectra are bare glass, so this is
    the glass level I0 after the build's median-ratio correction."""
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError(f"spectra must be a non-empty [N, C] array, got shape {x.shape}")
    return float(np.percentile(x.mean(axis=1), percentile))


def glass_reference(s_glass_unit: ArrayLike, level: float) -> np.ndarray:
    """Plan P2.4's unit-normalised glass SHAPE scaled to band mean `level`: the
    `s_glass` (value units) the op and `optical_density` expect."""
    s = _band_vector(s_glass_unit, "s_glass_unit")
    if not np.all(s > 0):
        raise ValueError("s_glass_unit must be > 0 in every band")
    if not level > 0:
        raise ValueError(f"level must be > 0, got {level}")
    return (s * (float(level) / s.mean())).astype(np.float32)


def fit_od_components(od_samples: np.ndarray, n_components: int = DEFAULT_OD_K, seed: int = 0,
                      max_iter: int = 1000) -> np.ndarray:
    """NMF stain components of train TISSUE optical density, [N, C] -> [K, C].

    Negative OD (a pixel brighter than the glass reference: noise, or a level
    slightly too low) is clipped to 0, since NMF needs non-negative input. The
    rows are scaled to unit L2 norm (NMF's W/H split is scale-free) and sorted
    by the band of their peak, so the result is stable to read and compare.
    Deterministic for a given `seed` (nndsvda init).
    """
    from sklearn.decomposition import NMF  # heavy; only the offline fit needs it

    od = np.asarray(od_samples, dtype=np.float64)
    if od.ndim != 2 or od.shape[0] < n_components:
        raise ValueError(f"od_samples must be [N >= {n_components}, C], got shape {od.shape}")
    if not np.all(np.isfinite(od)):
        raise ValueError("od_samples has non-finite values")
    model = NMF(n_components=int(n_components), init="nndsvda", random_state=int(seed),
                max_iter=int(max_iter))
    model.fit(np.clip(od, 0.0, None))
    comps = model.components_
    norms = np.linalg.norm(comps, axis=1, keepdims=True)
    comps = comps / np.where(norms > 0, norms, 1.0)
    return comps[np.argsort(np.argmax(comps, axis=1), kind="stable")].astype(np.float32)


# ============================================================================
# Builder + composition
# ============================================================================

def build_physics_ops(wavelengths: Optional[ArrayLike], global_mean: ArrayLike, global_std: ArrayLike,
                      noise_sigma: Optional[ArrayLike] = None, tilt_max: float = DEFAULT_TILT_MAX,
                      noise_k: float = DEFAULT_NOISE_K, od_delta: float = 0.0,
                      od_components: Optional[ArrayLike] = None,
                      s_glass: Optional[ArrayLike] = None) -> List[PhysicsOp]:
    """The active ops, in `OP_ORDER`. A magnitude <= 0 (or None) means the op is
    off: it is not built, so it consumes no random draws. An op that is ON but
    lacks its inputs raises instead of silently switching itself off.

    `s_glass` is the glass reference in VALUE units (`glass_reference`).
    """
    tilt_max = _magnitude(tilt_max, "tilt_max")
    noise_k = _magnitude(noise_k, "noise_k")
    od_delta = _magnitude(od_delta, "od_delta")
    ops: List[PhysicsOp] = []
    if od_delta > 0:
        if od_components is None or s_glass is None:
            raise ValueError("od_stain_jitter (od_delta > 0) needs od_components and s_glass "
                             f"({PHYS_ARTIFACT_FILES['od_components']}, {PHYS_ARTIFACT_FILES['s_glass']}); "
                             "pass od_delta=0 to turn it off")
        ops.append(ODStainJitter(od_components, s_glass, global_mean, global_std, delta=od_delta))
    if tilt_max > 0:
        if wavelengths is None:
            raise ValueError("spectral_tilt (tilt_max > 0) needs the band wavelengths (an HSI build's "
                             "wavelengths.npy); pass tilt_max=0 to turn it off")
        ops.append(SpectralTilt(wavelengths, global_mean, global_std, tilt_max=tilt_max))
    if noise_k > 0:
        if noise_sigma is None:
            raise ValueError(f"sensor_noise (noise_k > 0) needs noise_sigma ({PHYS_ARTIFACT_FILES['noise_sigma']}, "
                             "#12); pass noise_k=0 to turn it off")
        ops.append(SensorNoise(noise_sigma, global_mean, global_std, k=noise_k))
    return ops


class PhysicsAugment:
    """`base_fn(patch, rng, gen)`, then each op on the same (rng, gen)."""

    def __init__(self, base_fn: AugmentFn, phys_fns: Sequence[AugmentFn]):
        self.base_fn = base_fn
        self.phys_fns = list(phys_fns)
        # `AugmentedPatchDataset(seed=None)` inherits `augment_fn.seed`; keep the base's
        self.seed = int(getattr(base_fn, "seed", 0))
        self._fallback_rng = random.Random(self.seed)
        self._fallback_gen = torch.Generator().manual_seed(self.seed)

    def __call__(self, patch: torch.Tensor, rng: Optional[random.Random] = None,
                 gen: Optional[torch.Generator] = None) -> torch.Tensor:
        # the datasets always pass the per-item pair; the fallback mirrors `Compose`
        rng = self._fallback_rng if rng is None else rng
        gen = self._fallback_gen if gen is None else gen
        patch = self.base_fn(patch, rng, gen)
        for op in self.phys_fns:
            patch = op(patch, rng, gen)
        return patch

    def describe(self) -> List[Dict[str, Any]]:
        return [op.describe() if hasattr(op, "describe") else {"op": repr(op)} for op in self.phys_fns]


def compose_with_physics(base_fn: AugmentFn, phys_fns: Optional[Sequence[AugmentFn]]) -> AugmentFn:
    """`base_fn` followed by `phys_fns`, keeping the `augment_fn(patch, rng, gen)`
    signature. With no ops this returns `base_fn` itself, so `--phys_aug off`
    is bit-identical by construction, not by a test tolerance."""
    phys = list(phys_fns or ())
    if not phys:
        return base_fn
    return PhysicsAugment(base_fn, phys)


# ============================================================================
# Artifacts (`--phys_artifacts DIR`)
# ============================================================================

@dataclass(frozen=True)
class PhysArtifacts:
    noise_sigma: Optional[np.ndarray] = None
    od_components: Optional[np.ndarray] = None
    s_glass: Optional[np.ndarray] = None
    paths: Dict[str, str] = field(default_factory=dict)    # key -> file actually read

    def builder_kwargs(self) -> Dict[str, Optional[np.ndarray]]:
        """`build_physics_ops(..., **artifacts.builder_kwargs())`."""
        return {"noise_sigma": self.noise_sigma, "od_components": self.od_components, "s_glass": self.s_glass}


def load_phys_artifacts(directory: Union[str, Path]) -> PhysArtifacts:
    """Read the `PHYS_ARTIFACT_FILES` found under `directory` (each searched in
    `PHYS_ARTIFACT_SUBDIRS` order). A missing file is None; whether that is an
    error depends on which ops are on, so `build_physics_ops` decides."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    found: Dict[str, np.ndarray] = {}
    paths: Dict[str, str] = {}
    for key, name in PHYS_ARTIFACT_FILES.items():
        for sub in PHYS_ARTIFACT_SUBDIRS:
            p = root / sub / name
            if p.is_file():
                found[key] = np.load(p).astype(np.float32)
                paths[key] = str(p)
                break
    return PhysArtifacts(paths=paths, **found)


# ============================================================================
# Augmentation report
# ============================================================================

def _as_patch_batch(samples: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    t = samples if isinstance(samples, torch.Tensor) else torch.from_numpy(np.asarray(samples))
    if t.ndim != 4:
        raise ValueError(f"samples must be normalized [N, C, H, W], got shape {tuple(t.shape)}")
    return t.detach().cpu()


class _CorrAccumulator:
    """Band-band correlation from streamed [P, C] spectra (float64 moments,
    centred on a fixed shift so the subtraction does not cancel)."""

    def __init__(self, shift: np.ndarray):
        c = shift.size
        self.shift, self.n = shift, 0
        self.s, self.ss = np.zeros(c), np.zeros((c, c))

    def add(self, spectra: np.ndarray) -> None:
        d = spectra - self.shift
        self.n += d.shape[0]
        self.s += d.sum(axis=0)
        self.ss += d.T @ d

    def corr(self) -> np.ndarray:
        m = self.s / self.n
        cov = self.ss / self.n - np.outer(m, m)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        den = np.outer(sd, sd)
        return np.divide(cov, den, out=np.zeros_like(cov), where=den > 0)


def _spectra(x_value_chw: torch.Tensor) -> np.ndarray:
    """[C, H, W] -> [H*W, C] float64."""
    c = x_value_chw.shape[0]
    return x_value_chw.detach().to(torch.float64).reshape(c, -1).T.cpu().numpy()


def augmentation_report(samples: Union[np.ndarray, torch.Tensor], ops: Sequence[PhysicsOp],
                        wavelengths: Optional[ArrayLike] = None, *, seed: int = 0,
                        percentiles: Sequence[float] = REPORT_PERCENTILES, include_all: bool = True,
                        montage_path: Optional[Union[str, Path]] = None, montage_n: int = 4) -> Dict[str, Any]:
    """How far each op moves the data, measured in value space.

    `samples` are NORMALIZED patches [N, C, H, W] (what the train dataset
    yields); sample i is augmented with the pair derived from
    `item_augmentation_seed(seed, 0, i)`, as `AugmentedPatchDatasetV16` would at
    epoch 0. Per op (and, with `include_all`, for every op in order - what
    `--phys_aug` adds after the base augmentation):

      sam_deg            mean / percentiles / max of the per-pixel SAM(x, x')
      corr_frobenius     ||R(x') - R(x)||_F, R the band-band correlation over all
                         pixels of all samples (the plan's metric; its bound is
                         relative since #20: <= the paper recipe's own
                         augmentation's, see hsi_augment/augcal.py)
      corr_frobenius_rel the same divided by ||R(x)||_F
      corr_max_abs       its largest single entry
      min_value          min of x' (min_value_original: min of x)

    The pooled correlation reacts strongly to any per-SAMPLE change made in
    value space when a band varies little across the data: a +-3% tilt adds
    variance comparable to that of a NIR band with a ~10% coefficient of
    variation, so `spectral_tilt` at its default moves `corr_frobenius` far more
    than its SAM suggests (tests/test_hsi_augment_transforms.py measures it).
    Read the number together with the band CVs of the data it came from.

    `montage_path` also writes `save_augmentation_montage` there.
    """
    from training.augmentation import item_augmentation_seed  # frozen; light, but keep the module numpy+torch

    batch = _as_patch_batch(samples)
    ops = list(ops)
    if not ops:
        raise ValueError("no ops to report on")
    space = ops[0].space
    for op in ops[1:]:
        if not (torch.equal(op.space.mean, space.mean) and torch.equal(op.space.std, space.std)):
            raise ValueError("the ops were built with different global_mean/global_std")
    if batch.shape[1] != space.n_bands:
        raise ValueError(f"samples have {batch.shape[1]} bands, the ops {space.n_bands}")

    runs: List[Tuple[str, AugmentFn]] = [(op.name, op) for op in ops]
    if include_all and len(ops) > 1:
        runs.append(("all", compose_with_physics(lambda p, rng, gen: p, ops)))

    shift = space.mean.view(-1).double().numpy()
    orig_corr = _CorrAccumulator(shift)
    acc = {name: {"corr": _CorrAccumulator(shift), "sam": [], "min": math.inf} for name, _ in runs}
    orig_min = math.inf
    for i in range(batch.shape[0]):
        patch = batch[i]
        x = _spectra(space.to_value(patch.to(torch.float32)))
        orig_corr.add(x)
        orig_min = min(orig_min, float(x.min()))
        s = item_augmentation_seed(seed, 0, i)
        for name, fn in runs:
            out = fn(patch, random.Random(s), torch.Generator().manual_seed(s))
            y = _spectra(space.to_value(out.to(torch.float32)))
            a = acc[name]
            a["corr"].add(y)
            a["sam"].append(sam_degrees(x, y, axis=1))
            a["min"] = min(a["min"], float(y.min()))

    r0 = orig_corr.corr()
    report: Dict[str, Any] = {
        "n_samples": int(batch.shape[0]), "patch_shape": list(batch.shape[1:]), "seed": int(seed),
        "value_space": "x = x_n * global_std + global_mean",
        "wavelengths_nm": None if wavelengths is None else [float(w) for w in _band_vector(wavelengths, "wavelengths")],
        "min_value_original": orig_min, "ops": {},
    }
    by_name = {op.name: op for op in ops}
    for name, _fn in runs:
        a = acc[name]
        sam = np.concatenate(a["sam"])
        diff = a["corr"].corr() - r0
        sam_stats = {"mean": float(sam.mean()), "max": float(sam.max())}
        sam_stats.update({f"p{p:g}": float(np.percentile(sam, p)) for p in percentiles})
        report["ops"][name] = {
            "params": by_name[name].describe() if name in by_name else [op.describe() for op in ops],
            "sam_deg": sam_stats, "corr_frobenius": float(np.linalg.norm(diff)),
            "corr_frobenius_rel": float(np.linalg.norm(diff) / max(float(np.linalg.norm(r0)), 1e-12)),
            "corr_max_abs": float(np.abs(diff).max()), "min_value": a["min"],
        }
    report["montage"] = None
    if montage_path is not None:
        report["montage"] = str(save_augmentation_montage(batch, ops, wavelengths, montage_path,
                                                          n=montage_n, seed=seed))
    return report


def _rgb_bands(wavelengths: Optional[np.ndarray], n_bands: int) -> List[int]:
    if wavelengths is None:
        return [n_bands - 1, n_bands // 2, 0]
    return [int(np.argmin(np.abs(wavelengths - nm))) for nm in MONTAGE_RGB_NM]


def save_augmentation_montage(samples: Union[np.ndarray, torch.Tensor], ops: Sequence[PhysicsOp],
                              wavelengths: Optional[ArrayLike], path: Union[str, Path], *, n: int = 4,
                              seed: int = 0) -> Path:
    """One row per sample: a false-colour image (bands nearest 640/550/460 nm)
    of the original and of each op's output, all on the ORIGINAL's colour
    scale so a tint or a brightness change shows; then the patch-mean spectra
    against wavelength, with the line broken at spectral gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from training.augmentation import item_augmentation_seed  # frozen

    batch = _as_patch_batch(samples)
    ops = list(ops)
    space = ops[0].space
    n = max(1, min(int(n), batch.shape[0]))
    wl = None if wavelengths is None else _band_vector(wavelengths, "wavelengths")
    xs = wl if wl is not None else np.arange(space.n_bands, dtype=np.float64)
    breaks: List[int] = []
    if wl is not None:
        from hsi_augment.adapters.base import spectral_gaps  # pure function; the adapter rule
        breaks = spectral_gaps(wl)
    rgb = _rgb_bands(wl, space.n_bands)

    def broken(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # a NaN after each gap's first band stops the line crossing the gap
        return np.insert(xs, [g + 1 for g in breaks], np.nan), np.insert(v, [g + 1 for g in breaks], np.nan)

    cols = len(ops) + 2
    fig, axes = plt.subplots(n, cols, figsize=(2.2 * cols + 1.5, 2.2 * n), squeeze=False)
    for i in range(n):
        patch = batch[i]
        x = space.to_value(patch.to(torch.float32))
        s = item_augmentation_seed(seed, 0, i)
        views = [("original", x)] + [
            (op.name, space.to_value(op(patch, random.Random(s), torch.Generator().manual_seed(s)).to(torch.float32)))
            for op in ops]
        img0 = x[rgb].permute(1, 2, 0).numpy()
        lo = np.percentile(img0, 1, axis=(0, 1))
        hi = np.percentile(img0, 99, axis=(0, 1))
        span = np.where(hi > lo, hi - lo, 1.0)
        spec_ax = axes[i, -1]
        for j, (name, v) in enumerate(views):
            img = (v[rgb].permute(1, 2, 0).numpy() - lo) / span
            axes[i, j].imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(name, fontsize=8)
            px, py = broken(v.mean(dim=(1, 2)).double().numpy())
            spec_ax.plot(px, py, lw=1.0 if j else 1.6, color="black" if j == 0 else None, label=name)
        spec_ax.set_xlabel("wavelength (nm)" if wl is not None else "band", fontsize=7)
        spec_ax.tick_params(labelsize=6)
        if i == 0:
            spec_ax.set_title("patch-mean spectrum (value units)", fontsize=8)
            spec_ax.legend(fontsize=6)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
