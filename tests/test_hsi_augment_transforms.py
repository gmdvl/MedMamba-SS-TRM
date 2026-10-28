# -*- coding: utf-8 -*-
"""
test_hsi_augment_transforms.py
==============================
`hsi_augment/transforms.py` (issue #19, Plan §Phase 5): property tests on
random tiny patches. `hypothesis` is not installed, so the "random" part is a
seeded loop.

The data model. Value-space spectra follow transmission Beer-Lambert like the
shared fixture (`tests/hsi_augment_fixtures.py`): bare glass at its level
(1.45) times a smooth lamp shape, and tissue I = glass * exp(-c A(lambda)) with
three stain-like absorbers, a +-10% per-patch level, and the fixture's sensor
noise (0.004). The bands are trainsel's 32 selected wavelengths, gap included.
The two numbers the bounds depend on are stated where they are used.

Run with: pytest tests/test_hsi_augment_transforms.py -q
"""

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from hsi_augment import transforms as T
from hsi_augment.transforms import (
    ODStainJitter, PhysicsAugment, SensorNoise, SpectralTilt, ValueSpace, augmentation_report,
    build_physics_ops, compose_with_physics, estimate_glass_level, fit_od_components, glass_reference,
    legendre_basis, legendre_coordinate, load_phys_artifacts, od_to_value, optical_density, sam_degrees,
)
from tests.hsi_augment_fixtures import GLASS_LEVEL, NOISE_STD
from training.augmentation import AugmentationConfig, item_augmentation_seed
from training.augmentation_v16 import AugmentedPatchDatasetV16, build_active
from training.minority_augmentation import MinorityAugmentedDataset, scaled_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# data/hsi_v9-trainsel/selected_wavelengths.npy (nm): 628.9 -> 850.9 is the gap
WAVELENGTHS = np.array([
    400.482, 407.758, 413.579, 419.399, 425.947, 431.768, 437.589, 443.409, 450.685, 456.506,
    464.509, 471.785, 478.333, 490.702, 531.446, 559.094, 575.1, 628.941, 850.851, 857.399,
    863.22, 870.496, 877.044, 884.319, 893.05, 899.598, 905.419, 911.967, 917.788, 926.519,
    932.339, 938.16], dtype=np.float32)
C = WAVELENGTHS.size
GAP_AFTER = 17

# The paper recipe's base augmentation (training/train_profiles.py, paper_recipe).
PAPER_RECIPE = AugmentationConfig(flip_h_prob=0.5, flip_v_prob=0.5, rotate90_prob=0.5,
                                  crop_scale_max_frac=0.3, spectral_scale_range=0.15,
                                  spectral_offset_std=0.05, spectral_noise_std=0.02, seed=42)


# ============================================================================
# Synthetic value-space data
# ============================================================================

def _absorbance(wl: np.ndarray) -> np.ndarray:
    """Three stain-like absorbers at unit density: a broad ~590 nm band
    (haematoxylin-like), a narrow ~525 nm band (eosin-like), and a weak
    lambda^-2 term (scattering) that is all the NIR bands see."""
    wl = wl.astype(np.float64)
    return np.stack([0.55 * np.exp(-((wl - 590.0) / 70.0) ** 2) + 0.04,
                     0.75 * np.exp(-((wl - 525.0) / 32.0) ** 2) + 0.02,
                     0.12 * (550.0 / wl) ** 2])


ABSORBANCE = _absorbance(WAVELENGTHS)
LAMP_SHAPE = 1.0 + 0.08 * legendre_basis(WAVELENGTHS)[0]      # glass spectrum's shape
GLASS = GLASS_LEVEL * LAMP_SHAPE


def value_patches(n: int, size: int, seed: int, noise: float = NOISE_STD) -> np.ndarray:
    """[n, C, size, size] float32 value-space patches, all >= 0."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(0.9, 1.1, size=(n, 1, 1, 1))
    is_glass = rng.random((n, size, size)) < rng.uniform(0.0, 0.4, size=(n, 1, 1))
    conc = rng.gamma(4.0, 0.25, size=(n, size, size, 3)) * np.array([1.0, 1.0, 0.6])
    conc[is_glass] = 0.0
    x = level * GLASS * np.exp(-(conc @ ABSORBANCE))
    x = x + rng.normal(scale=noise, size=x.shape)
    return np.clip(x, 0.0, None).transpose(0, 3, 1, 2).astype(np.float32)


class Data:
    """Normalized patches + the global_zscore stats + fitted OD artifacts."""

    def __init__(self, n: int, size: int, seed: int):
        self.value = value_patches(n, size, seed)
        self.mean = self.value.mean(axis=(0, 2, 3))
        self.std = self.value.std(axis=(0, 2, 3))
        self.norm = torch.from_numpy((self.value - self.mean[:, None, None]) / self.std[:, None, None])


@pytest.fixture(scope="module")
def data() -> Data:
    return Data(2000, 5, seed=0)


@pytest.fixture(scope="module")
def od_artifacts():
    """The OD model as the offline fit would build it: glass level and stain
    components from a TRAIN sample (a different seed from the test data)."""
    train = value_patches(800, 5, seed=99).transpose(0, 2, 3, 1).reshape(-1, C)
    s_glass = glass_reference(LAMP_SHAPE / np.linalg.norm(LAMP_SHAPE), estimate_glass_level(train))
    tissue = train.mean(axis=1) < 0.8 * s_glass.mean()
    comps = fit_od_components(optical_density(train[tissue], s_glass), 3, seed=0)
    return comps, s_glass


def default_ops(data: Data, od_artifacts, noise=NOISE_STD):
    """Every op at its default magnitude, od_stain_jitter included (delta 0.1)."""
    comps, s_glass = od_artifacts
    return build_physics_ops(WAVELENGTHS, data.mean, data.std, noise_sigma=np.full(C, noise, np.float32),
                             od_delta=T.DEFAULT_OD_DELTA, od_components=comps, s_glass=s_glass)


def _pair(seed: int):
    return random.Random(seed), torch.Generator().manual_seed(seed)


# ============================================================================
# Module shape
# ============================================================================

def test_module_imports_only_numpy_and_torch():
    """train.py imports it: sklearn, matplotlib, scipy and the heavy package
    modules must stay out until a function needs them."""
    code = ("import sys, hsi_augment.transforms\n"
            "heavy = ('sklearn', 'matplotlib', 'scipy', 'hsi_augment.stage', 'hsi_augment.io',\n"
            "         'hsi_augment.adapters', 'hsi_augment.config', 'training')\n"
            "print(sorted(m for m in heavy if m in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "[]"


def test_dropped_ops_are_documented():
    doc = T.__doc__.lower()
    for word in ("global gain", "band dropout", "wavelength shift", "elastic", "mixup"):
        assert word in doc


# ============================================================================
# Building blocks
# ============================================================================

def test_legendre_basis_is_over_real_wavelengths_not_band_index():
    t = legendre_coordinate(WAVELENGTHS)
    assert t[0] == -1.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0)
    # the 222 nm gap is 41% of the 538 nm range: one band step jumps that far
    assert t[GAP_AFTER + 1] - t[GAP_AFTER] == pytest.approx(2 * (850.851 - 628.941) / (938.16 - 400.482), rel=1e-5)
    p1, p2 = legendre_basis(WAVELENGTHS)
    np.testing.assert_allclose(p1, np.polynomial.legendre.legval(t, [0, 1]))
    np.testing.assert_allclose(p2, np.polynomial.legendre.legval(t, [0, 0, 1]))


def test_value_space_round_trip(data):
    space = ValueSpace(data.mean, data.std)
    x = space.to_value(data.norm[0])
    np.testing.assert_allclose(x.numpy(), data.value[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(space.to_normalized(x).numpy(), data.norm[0].numpy(), atol=1e-5)


def test_sam_degrees():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    b = np.array([[0.0, 2.0], [3.0, 3.0], [1.0, 0.0]])
    np.testing.assert_allclose(sam_degrees(a, b), [90.0, 0.0, 0.0], atol=1e-6)


def test_tilt_factor_is_bounded_and_smooth_in_wavelength(data):
    op = SpectralTilt(WAVELENGTHS, data.mean, data.std, tilt_max=0.03)
    p1, p2 = legendre_basis(WAVELENGTHS)
    for a, b in ((0.03, 0.03), (-0.03, 0.03), (0.03, -0.03), (-0.03, -0.03), (0.01, -0.02)):
        f = op.factor(a, b).view(-1).double().numpy()
        np.testing.assert_allclose(f, 1 + a * p1 + b * p2, rtol=1e-6)
        assert f.min() >= 1 - 2 * 0.03 - 1e-6 and f.max() <= 1 + 2 * 0.03 + 1e-6
    rng = random.Random(0)
    draws = [op.draw(rng) for _ in range(2000)]
    assert max(abs(v) for d in draws for v in d) <= 0.03
    assert min(d[0] for d in draws) < -0.029 and max(d[0] for d in draws) > 0.029


def test_tilt_multiplies_every_pixel_by_the_same_factor(data):
    op = SpectralTilt(WAVELENGTHS, data.mean, data.std)
    rng, gen = _pair(5)
    out = op.space.to_value(op(data.norm[3], rng, gen)).double().numpy()
    ratio = out / data.value[3]
    a, b = op.draw(random.Random(5))
    np.testing.assert_allclose(ratio, op.factor(a, b).double().numpy() * np.ones_like(ratio), rtol=2e-4)


def test_sensor_noise_has_std_k_sigma_per_band(data):
    sigma = np.linspace(0.002, 0.008, C).astype(np.float32)
    x_n = torch.from_numpy((np.full((C, 64, 64), 1.0, np.float32) - data.mean[:, None, None])
                           / data.std[:, None, None])
    for k in (0.5, 1.0, 2.0):
        op = SensorNoise(sigma, data.mean, data.std, k=k)
        d = op.space.to_value(op(x_n, *_pair(1))).double().numpy() - 1.0
        np.testing.assert_allclose(d.std(axis=(1, 2)), k * sigma, rtol=0.08)
        assert abs(d.mean()) < 0.1 * k * sigma.mean()


# ============================================================================
# Acceptance: label, shape, dtype, device; norm_stats untouched
# ============================================================================

class _TupleDataset(Dataset):
    """(patch, label, norm_stats), as NpyDatasetV16(return_norm_stats=True) returns."""

    def __init__(self, patches: torch.Tensor, labels, norm_stats: np.ndarray):
        self.patches, self.labels, self.norm_stats = patches, list(labels), norm_stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.patches[i], self.labels[i], self.norm_stats


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_each_op_preserves_shape_dtype_and_device(data, od_artifacts, dtype):
    x = data.norm[0].to(dtype)
    for op in default_ops(data, od_artifacts):
        y = op(x, *_pair(3))
        assert y.shape == x.shape and y.dtype == dtype and y.device == x.device
        assert torch.isfinite(y).all()


# float32/float64 only: the frozen base Compose itself promotes float16 to
# float32 (`spectral_scale` multiplies by a float32 factor); the datasets yield float32
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ops_preserve_label_shape_dtype_and_norm_stats(data, od_artifacts, dtype):
    ops = default_ops(data, od_artifacts)
    assert [op.name for op in ops] == list(T.OP_ORDER)
    fn = compose_with_physics(build_active(PAPER_RECIPE), ops)
    patches = data.norm[:12].to(dtype)
    stats = np.stack([data.mean, data.std]).astype(np.float32)
    ds = AugmentedPatchDatasetV16(_TupleDataset(patches, [0, 1, 2] * 4, stats), fn, seed=7)
    for i in range(len(ds)):
        out, label, ns = ds[i]
        assert label == [0, 1, 2][i % 3]
        assert ns is stats                        # passed through, never copied or touched
        assert out.shape == patches[i].shape and out.dtype == dtype and out.device == patches[i].device
        assert torch.isfinite(out).all()


def test_minority_dataset_routes_physics_to_minority_rows_only(data, od_artifacts):
    """#20's `minority` mode: majority rows stay on the plain base path."""
    base = build_active(PAPER_RECIPE)
    phys = compose_with_physics(build_active(scaled_config(PAPER_RECIPE, 1.0)), default_ops(data, od_artifacts))
    stats = np.stack([data.mean, data.std]).astype(np.float32)
    labels = [0, 1, 2, 1, 0, 2]
    inner = _TupleDataset(data.norm[:6], labels, stats)
    mixed = MinorityAugmentedDataset(inner, base, phys, [1], seed=7)
    plain = AugmentedPatchDatasetV16(inner, base, seed=7)
    for i, y in enumerate(labels):
        out, label, ns = mixed[i]
        assert label == y and ns is stats
        assert torch.equal(out, plain[i][0]) == (y != 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_ops_keep_a_cuda_patch_on_cuda(data, od_artifacts):  # pragma: no cover - CPU sandbox
    for op in default_ops(data, od_artifacts):
        y = op(data.norm[0].cuda(), *_pair(1))
        assert y.is_cuda and y.shape == data.norm[0].shape


# ============================================================================
# Acceptance: deterministic for (seed, epoch, index, worker)
# ============================================================================

def test_deterministic_for_seed_epoch_index_worker(data, od_artifacts):
    fn = compose_with_physics(build_active(PAPER_RECIPE), default_ops(data, od_artifacts))
    x = data.norm[0]

    def run(seed, epoch, index, worker):
        return fn(x.clone(), *_pair(item_augmentation_seed(seed, epoch, index, worker)))

    ref = run(42, 3, 17, 1)
    assert torch.equal(ref, run(42, 3, 17, 1))
    for other in ((43, 3, 17, 1), (42, 4, 17, 1), (42, 3, 18, 1), (42, 3, 17, 2)):
        assert not torch.equal(ref, run(*other))

    ds = AugmentedPatchDatasetV16(_TupleDataset(data.norm[:4], [0, 1, 2, 0], np.zeros(1)), fn, seed=42)
    ds.set_epoch(2)
    first = [ds[i][0] for i in range(4)]
    assert all(torch.equal(a, ds[i][0]) for i, a in enumerate(first))
    ds.set_epoch(3)
    assert not any(torch.equal(a, ds[i][0]) for i, a in enumerate(first))


# ============================================================================
# Acceptance: physics off = bit-identical output and RNG stream
# ============================================================================

def test_physics_off_is_the_base_function_itself(data):
    base = build_active(PAPER_RECIPE)
    assert compose_with_physics(base, []) is base
    assert compose_with_physics(base, None) is base
    off = build_physics_ops(WAVELENGTHS, data.mean, data.std, noise_sigma=None, tilt_max=0.0, noise_k=0.0,
                            od_delta=0.0)
    assert off == []
    assert build_physics_ops(None, data.mean, data.std, tilt_max=-1.0, noise_k=None, od_delta=-0.1) == []
    fn = compose_with_physics(base, off)
    for i in range(20):
        s = item_augmentation_seed(42, 0, i)
        (r1, g1), (r2, g2) = _pair(s), _pair(s)
        assert torch.equal(fn(data.norm[i], r1, g1), base(data.norm[i], r2, g2))
        assert r1.getstate() == r2.getstate()
        assert torch.equal(g1.get_state(), g2.get_state())


def test_physics_draws_after_the_base_augmentation(data, od_artifacts):
    """The base sees exactly the stream it sees without physics; the ops then
    continue the SAME pair."""
    base = build_active(PAPER_RECIPE)
    ops = default_ops(data, od_artifacts)
    seen = []

    def spy(patch, rng, gen):
        seen.append((patch.clone(), rng.getstate(), gen.get_state().clone()))
        return patch

    fn = compose_with_physics(base, [spy, *ops])
    s = item_augmentation_seed(42, 1, 5)
    out = fn(data.norm[5], *_pair(s))

    r_ref, g_ref = _pair(s)
    base_out = base(data.norm[5], r_ref, g_ref)
    assert torch.equal(seen[0][0], base_out)
    assert seen[0][1] == r_ref.getstate() and torch.equal(seen[0][2], g_ref.get_state())
    manual = base_out
    for op in ops:
        manual = op(manual, r_ref, g_ref)
    assert torch.equal(out, manual)
    assert not torch.equal(out, base_out)


def test_composed_function_keeps_the_seed_and_fallback_rngs(data):
    base = build_active(PAPER_RECIPE)
    fn = compose_with_physics(base, [SpectralTilt(WAVELENGTHS, data.mean, data.std)])
    assert isinstance(fn, PhysicsAugment) and fn.seed == 42
    assert AugmentedPatchDatasetV16(_TupleDataset(data.norm[:1], [0], np.zeros(1)), fn).seed == 42
    assert fn(data.norm[0]).shape == data.norm[0].shape    # no rng/gen: the Compose-style fallback
    assert fn.describe()[0]["op"] == "spectral_tilt"


# ============================================================================
# Acceptance: mean SAM <= 2 degrees per op, correlation change, values >= 0
# ============================================================================

def test_mean_sam_at_most_2_degrees_per_op_at_default_magnitudes(data, od_artifacts):
    report = augmentation_report(data.norm, default_ops(data, od_artifacts), WAVELENGTHS)
    for name in T.OP_ORDER:
        r = report["ops"][name]
        assert 0.0 < r["sam_deg"]["mean"] <= 2.0, (name, r["sam_deg"])
    # measured: od_stain_jitter ~0.3, spectral_tilt ~0.8, sensor_noise ~0.2, all three ~0.9
    assert report["ops"]["all"]["sam_deg"]["mean"] <= 2.0


@pytest.fixture(scope="module")
def report_10k(od_artifacts):
    """10k samples of 3 x 3 pixels (90k spectra), every op at its default."""
    big = Data(10_000, 3, seed=1)
    return augmentation_report(big.norm, default_ops(big, od_artifacts), WAVELENGTHS, include_all=False)


@pytest.mark.parametrize("name", ["sensor_noise", "od_stain_jitter"])
def test_correlation_change_below_0_05_over_10k_samples(report_10k, name):
    # measured: sensor_noise ~0.032, od_stain_jitter ~0.022. sensor_noise uses
    # sigma_b = 0.004 (the fixture's sensor noise, 0.28% of the glass level); the
    # change grows with sigma_b^2 (0.006 gives ~0.07, 0.01 ~0.20 on this model),
    # so on the real build the bound holds only if #12's sigma_b is that small
    # next to the bands' spread. augmentation_report measures it there.
    r = report_10k["ops"][name]
    assert report_10k["n_samples"] == 10_000
    assert r["corr_frobenius"] < 0.05, (name, r)


def test_spectral_tilt_meets_the_recipe_relative_bound_at_its_calibrated_tilt_max():
    """#20's decision (2026-10-27) replaced the plan's absolute bound (< 0.05,
    missed at tilt_max 0.03 by ~1.1 here: a per-sample +-3% tilt adds variance
    comparable to a NIR band's with a ~7% coefficient of variation). A physics op
    may change the correlation no more than the paper recipe's own augmentation
    does on the same data. `augcal.calibrate` picks the largest grid tilt_max that
    meets it; here, and the plan's default does not."""
    from hsi_augment.augcal import BASE_OP_NAME, calibrate, paper_recipe_aug_config

    assert paper_recipe_aug_config(seed=42) == PAPER_RECIPE      # this file's copy is train_profiles' recipe
    big = Data(10_000, 3, seed=1)
    grid = (0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.03)
    decision, report = calibrate(big.norm, WAVELENGTHS, big.mean, big.std, np.full(C, NOISE_STD, np.float32), grid)
    base, t = decision["base_corr_change"], decision["tilt_max"]
    assert 0.2 < base < 0.4                                      # measured ~0.30
    assert decision["tilt_bound"] == "interior" and 0.01 <= t <= 0.0175     # measured 0.015
    tilt = report["chosen"]["ops"]["spectral_tilt"]["corr_frobenius"]
    assert tilt == decision["tilt_corr_change"] <= base == report["scan"]["ops"][BASE_OP_NAME]["corr_frobenius"]
    assert all(r["corr_frobenius"] > base for r in decision["grid"] if r["tilt_max"] > t)
    assert report["scan"]["ops"]["spectral_tilt@0.03"]["corr_frobenius"] > 3 * base      # the plan's default
    # sensor_noise at k = 1 with this model's sigma_b: the same criterion, checked (measured ~0.03)
    assert decision["noise_check"]["within_base"] and decision["noise_check"]["corr_frobenius"] < 0.2 * base


def test_spectral_tilt_correlation_change_is_quadratic_and_meets_the_bound_when_small(od_artifacts):
    big = Data(10_000, 3, seed=1)
    res = {}
    for t in (0.005, 0.01):
        rep = augmentation_report(big.norm, [SpectralTilt(WAVELENGTHS, big.mean, big.std, tilt_max=t)],
                                  include_all=False)
        res[t] = rep["ops"]["spectral_tilt"]["corr_frobenius"]
    assert res[0.005] < 0.05                            # measured ~0.036
    assert 3.0 < res[0.01] / res[0.005] < 5.0          # ~4x: second order in tilt_max


def test_values_stay_nonnegative_in_value_space(data, od_artifacts):
    """Positive input at the largest allowed magnitudes, plus dark spectra
    near 0 where only sensor_noise's clamp keeps the result >= 0."""
    comps, s_glass = od_artifacts
    dark = value_patches(200, 5, seed=3) * 0.01                # ~0.003-0.017: noise reaches 0
    norm_dark = torch.from_numpy((dark - data.mean[:, None, None]) / data.std[:, None, None])
    ops = [SpectralTilt(WAVELENGTHS, data.mean, data.std, tilt_max=T.MAX_TILT - 1e-3),
           SensorNoise(np.full(C, 0.01, np.float32), data.mean, data.std, k=3.0),
           ODStainJitter(comps, s_glass, data.mean, data.std, delta=0.9)]
    for batch in (data.norm[:300], norm_dark):
        for op in ops:
            for i in range(len(batch)):
                y = op.space.to_value(op(batch[i], *_pair(i)))
                # the normalized round trip costs ~1e-7 x mean/std in float32
                assert float(y.min()) >= -1e-5, op.name
    clipped = ops[1].space.to_value(ops[1](norm_dark[0], *_pair(0)))
    assert (clipped.abs() < 1e-5).any()                 # the clamp did engage on the dark patch


def test_default_report_min_value(data, od_artifacts):
    report = augmentation_report(data.norm[:300], default_ops(data, od_artifacts), WAVELENGTHS)
    assert report["min_value_original"] >= -1e-5
    assert all(r["min_value"] >= -1e-5 for r in report["ops"].values())


# ============================================================================
# od_stain_jitter
# ============================================================================

def test_od_round_trip_is_the_identity_at_zero_delta(data, od_artifacts):
    comps, s_glass = od_artifacts
    op = ODStainJitter(comps, s_glass, data.mean, data.std, delta=0.0)
    for i in range(20):
        x = op.space.to_value(data.norm[i])
        assert torch.equal(op.apply_value(x, *_pair(i)), x)   # exp(0) = 1: exact in value space
        np.testing.assert_allclose(op(data.norm[i], *_pair(i)).numpy(), data.norm[i].numpy(), atol=1e-5)
    # the literal transforms invert each other above the clip
    spectra = data.value[:50].transpose(0, 2, 3, 1).reshape(-1, C).astype(np.float64)
    keep = np.all(spectra > T.OD_EPS * s_glass, axis=1)
    np.testing.assert_allclose(od_to_value(optical_density(spectra[keep], s_glass), s_glass), spectra[keep],
                               rtol=1e-10)


def test_od_jitter_equals_the_literal_back_transform(data, od_artifacts):
    """x * exp(-dOD) is s_glass * exp(-(OD + dOD)) wherever x is above the clip."""
    comps, s_glass = od_artifacts
    op = ODStainJitter(comps, s_glass, data.mean, data.std, delta=0.1)
    x = op.space.to_value(data.norm[4].double())
    out = op.apply_value(x, *_pair(4)).numpy().reshape(C, -1).T
    spectra = x.numpy().reshape(C, -1).T
    od = optical_density(spectra, s_glass)
    w = np.clip(od @ np.linalg.pinv(comps.astype(np.float64)), 0.0, None)
    factors = np.array(op.draw(random.Random(4)))
    literal = od_to_value(od + (w * (factors - 1.0)) @ comps.astype(np.float64), s_glass)
    np.testing.assert_allclose(out, literal, rtol=1e-5)


def test_fit_od_components_recovers_the_stains_and_is_deterministic():
    rng = np.random.default_rng(0)
    truth = ABSORBANCE / np.linalg.norm(ABSORBANCE, axis=1, keepdims=True)
    od = rng.gamma(2.0, 0.5, size=(4000, 3)) @ truth + rng.normal(scale=0.002, size=(4000, C))
    comps = fit_od_components(od, 3, seed=0)
    assert comps.shape == (3, C) and comps.dtype == np.float32 and np.all(comps >= 0)
    np.testing.assert_allclose(np.linalg.norm(comps, axis=1), 1.0, rtol=1e-5)
    cos = comps.astype(np.float64) @ truth.T
    assert np.all(cos.max(axis=1) > 0.98), np.round(cos, 3)
    assert np.array_equal(comps, fit_od_components(od, 3, seed=0))


def test_glass_level_and_reference():
    spectra = value_patches(300, 5, seed=2).transpose(0, 2, 3, 1).reshape(-1, C)
    level = estimate_glass_level(spectra)
    assert abs(level - GLASS.mean()) < 0.15 * GLASS.mean()      # the brightest spectra are glass
    ref = glass_reference(LAMP_SHAPE / np.linalg.norm(LAMP_SHAPE), level)
    assert ref.mean() == pytest.approx(level, rel=1e-6)
    np.testing.assert_allclose(ref / ref.mean(), LAMP_SHAPE / LAMP_SHAPE.mean(), rtol=1e-6)


# ============================================================================
# Validation, artifacts, report
# ============================================================================

def test_on_ops_without_their_inputs_raise(data):
    with pytest.raises(ValueError, match="noise_sigma"):
        build_physics_ops(WAVELENGTHS, data.mean, data.std, noise_sigma=None)
    with pytest.raises(ValueError, match="wavelengths"):
        build_physics_ops(None, data.mean, data.std, noise_k=0.0)
    with pytest.raises(ValueError, match="od_components"):
        build_physics_ops(WAVELENGTHS, data.mean, data.std, noise_k=0.0, od_delta=0.1)
    with pytest.raises(ValueError, match="finite"):
        build_physics_ops(WAVELENGTHS, data.mean, data.std, tilt_max=float("nan"), noise_k=0.0)


def test_bad_arguments_raise(data, od_artifacts):
    comps, s_glass = od_artifacts
    with pytest.raises(ValueError, match="tilt_max"):
        SpectralTilt(WAVELENGTHS, data.mean, data.std, tilt_max=0.5)
    with pytest.raises(ValueError, match="bands"):
        SpectralTilt(WAVELENGTHS[:-1], data.mean, data.std)
    with pytest.raises(ValueError, match=">= 0"):
        SensorNoise(-np.ones(C), data.mean, data.std)
    with pytest.raises(ValueError, match="global_std"):
        ValueSpace(data.mean, np.zeros(C))
    with pytest.raises(ValueError, match="od_delta"):
        ODStainJitter(comps, s_glass, data.mean, data.std, delta=1.0)
    with pytest.raises(ValueError, match="linearly dependent"):
        ODStainJitter(np.stack([comps[0], comps[0]]), s_glass, data.mean, data.std)
    op = SpectralTilt(WAVELENGTHS, data.mean, data.std)
    with pytest.raises(ValueError, match="patch"):
        op(torch.zeros(C - 1, 3, 3), *_pair(0))


def test_load_phys_artifacts(tmp_path, data, od_artifacts):
    comps, s_glass = od_artifacts
    (tmp_path / "prep" / "spectral").mkdir(parents=True)
    np.save(tmp_path / "prep" / "spectral" / "noise_sigma.npy", np.full(C, 0.004, np.float32))
    np.save(tmp_path / "od_components.npy", comps)
    arts = load_phys_artifacts(tmp_path)
    assert arts.s_glass is None and arts.noise_sigma.shape == (C,)
    assert set(arts.paths) == {"noise_sigma", "od_components"}
    ops = build_physics_ops(WAVELENGTHS, data.mean, data.std, **arts.builder_kwargs())
    assert [op.name for op in ops] == ["spectral_tilt", "sensor_noise"]
    with pytest.raises(ValueError, match="s_glass"):
        build_physics_ops(WAVELENGTHS, data.mean, data.std, od_delta=0.1, **arts.builder_kwargs())
    np.save(tmp_path / "od_s_glass.npy", s_glass)
    ops = build_physics_ops(WAVELENGTHS, data.mean, data.std, od_delta=0.1,
                            **load_phys_artifacts(tmp_path).builder_kwargs())
    assert [op.name for op in ops] == list(T.OP_ORDER)
    with pytest.raises(FileNotFoundError):
        load_phys_artifacts(tmp_path / "missing")


def test_report_fields_and_montage(tmp_path, data, od_artifacts):
    ops = default_ops(data, od_artifacts)
    png = tmp_path / "figs" / "montage.png"
    report = augmentation_report(data.norm[:40], ops, WAVELENGTHS, montage_path=png, montage_n=3)
    assert set(report["ops"]) == {*T.OP_ORDER, "all"}
    for r in report["ops"].values():
        assert {"mean", "max", "p5", "p50", "p95", "p99"} <= set(r["sam_deg"])
        assert {"corr_frobenius", "corr_frobenius_rel", "corr_max_abs", "min_value", "params"} <= set(r)
    assert report["n_samples"] == 40 and report["patch_shape"] == [C, 5, 5]
    assert report["montage"] == str(png) and png.stat().st_size > 10_000
    # deterministic: the same seed reproduces every number
    again = augmentation_report(data.norm[:40], ops, WAVELENGTHS)
    assert again["ops"] == report["ops"]
