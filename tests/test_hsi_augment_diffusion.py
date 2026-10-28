# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_diffusion.py
===================================
Phase 7 (#24), arm A3, second half: the class-conditional latent diffusion
denoiser and its DDIM / classifier-free-guidance sampler
(`hsi_augment/models/diffusion.py`), and the stages `ldm-lopo` / `ldm-final`
(group `fit-ldm`, `hsi_augment/latent_diffusion.py`).

The issue's CPU acceptance, each a test below: the v-target algebra round
trips; DDIM is deterministic at eta = 0 (and, from an oracle denoiser, lands
exactly on the data); the CFG null token works (label dropout, w = 0 / 1 /
w > 1); the model overfits one batch; resume is bitwise (inside ldm-lopo and
ldm-final); the sampling sha256 is deterministic (recorded by the stage,
reproduced from the exported generator.pt). Also: shapes and conditioning,
the cosine schedule, Min-SNR weights, bf16 autocast, the latent cache, the
latent MMD, fold isolation, the final refit, refusing changed inputs, and the
`fit-ae` / `fit-ldm` command lines.

Shared helpers: `tests/hsi_augment_ldm_fixtures.py`.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hsi_augment import latent_diffusion as ld
from hsi_augment.codec import LogSpaceCodec
from hsi_augment.config import ConfigError
from hsi_augment.io import read_access_log, sha256_file
from hsi_augment.models.autoencoder import SpatialAutoencoder, count_parameters
from hsi_augment.models.blocks import assert_no_batchnorm
from hsi_augment.models.diffusion import (CosineSchedule, DiffusionLoss, LatentUNet, ddim_sample, eps_from_v,
                                          guided_v, min_snr_weight, sample, sample_latents, v_target, x0_from_v)
from hsi_augment.pool_io import load_pool, pool_fingerprint, write_pool
from hsi_augment.registry import load_builtin_stages
from hsi_augment.splits import fold_masks
from hsi_augment.stage import DONE_NAME, StageError, StageNotReady, read_done
from hsi_augment.stages.fit_ldm import LDMLopoConfig
from hsi_augment.train_generative import BEST_NAME, HISTORY_NAME, LAST_NAME, load_checkpoint
from tests.hsi_augment_fixtures import make_tiny_build
from tests.hsi_augment_ldm_fixtures import (FOLD_SUMMARY_KEYS, TINY_AE, StepSpy, ae_final_cfg, ae_lopo_cfg,
                                            assert_same, assert_same_training, ldm_final_cfg, ldm_lopo_cfg, prepare,
                                            run, sha)

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLD_FILES = (ld.LATENTS_NAME, ld.LATENTS_META_NAME, BEST_NAME, LAST_NAME, HISTORY_NAME, ld.GENERATOR_NAME,
              ld.MMD_NAME, ld.FOLD_DONE_NAME)
TRAIN = SimpleNamespace(training=True)
EVAL = SimpleNamespace(training=False)


# ============================================================================
# fixtures
# ============================================================================

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def registry():
    return load_builtin_stages()


@pytest.fixture(scope="module")
def ran(tiny, registry, tmp_path_factory):
    """leakage -> pool -> ae-lopo -> ae-final -> ldm-lopo (loss calls recorded) -> ldm-final."""
    out = prepare(tiny, registry, tmp_path_factory.mktemp("ldm"))
    ae_lopo = run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())
    ae_final = run(registry, tiny, out, "ae-final", ae_final_cfg())
    with pytest.MonkeyPatch.context() as mp:
        spy = StepSpy(mp, DiffusionLoss)
        lopo = run(registry, tiny, out, "ldm-lopo", ldm_lopo_cfg())
    final = run(registry, tiny, out, "ldm-final", ldm_final_cfg())
    folds = json.loads((out / "prep/leakage/folds.json").read_text())["folds"]
    return SimpleNamespace(out=out, ae_lopo=ae_lopo, ae_final=ae_final, lopo=lopo, final=final, spy=spy,
                           folds=folds, pool=load_pool(out / "pool"), lopo_dir=out / "gen/ldm/lopo",
                           final_dir=out / "gen/ldm/final")


def tiny_unet(seed: int = 0, *, live: bool = False, **kw) -> LatentUNet:
    """A tiny denoiser. An untrained one ignores its conditioning and outputs 0
    (FiLM, every block's second conv and the output conv are zero-initialised);
    `live` re-draws every all-zero parameter, so the output depends on z, t
    and the class from the start."""
    torch.manual_seed(seed)
    m = LatentUNet(**{"latent_channels": 2, "latent_size": 3, "channels": (16, 32), "num_classes": 3,
                      "emb_dim": 16, "freq_dim": 8, "timesteps": 50, **kw})
    if live:
        with torch.no_grad():
            for p in m.parameters():
                if not p.any():
                    torch.nn.init.normal_(p, std=0.1)
    return m.eval()


def tiny_ae_and_codec(seed: int = 0):
    torch.manual_seed(seed)
    ae = SpatialAutoencoder(in_channels=8, patch=5, **TINY_AE).eval()
    ae.latent_scale = 0.5
    X = np.random.default_rng(seed).uniform(0.2, 2.0, size=(64, 5, 5, 8)).astype(np.float32)
    return ae, LogSpaceCodec.from_stats(LogSpaceCodec.fit_stats([X]), 3.0)


class Recorder(torch.nn.Module):
    """Wraps a denoiser and records the class ids of every forward pass."""

    def __init__(self, inner: LatentUNet):
        super().__init__()
        self.inner, self.seen = inner, []
        self.schedule, self.null_class, self.num_classes = inner.schedule, inner.null_class, inner.num_classes

    def forward(self, z_t, t, y):
        self.seen.append(y.clone())
        return self.inner(z_t, t, y)


# ============================================================================
# schedule, v-parameterisation, Min-SNR
# ============================================================================

def test_cosine_schedule():
    s = CosineSchedule(1000)
    ab = s.alpha_bar
    assert ab.shape == (1000,) and np.all(np.diff(ab) < 0) and 0 < ab[-1] < ab[0] < 1
    t = torch.tensor([0, 10, 500, 999, -1])
    a, sig = s.alpha_sigma(t)
    assert torch.allclose(a ** 2 + sig ** 2, torch.ones(5), atol=1e-6)
    assert a[-1] == 1 and sig[-1] == 0                       # index -1 = no noise (DDIM's last target)
    assert s.ddim_timesteps(50).tolist() == list(range(999, 0, -20))   # trailing: 999, 979, ..., 19
    assert s.ddim_timesteps(1).tolist() == [999]
    for bad in (0, 1001):
        with pytest.raises(ValueError):
            s.ddim_timesteps(bad)
    assert torch.all(s.snr(torch.tensor([0, 999])) > 0)


def test_v_target_algebra_round_trips():
    torch.manual_seed(0)
    s = CosineSchedule(1000)
    x0, eps = torch.randn(64, 2, 3, 3), torch.randn(64, 2, 3, 3)
    t = torch.randint(0, 1000, (64,))
    a, sig = s.alpha_sigma(t)
    z_t = a.view(-1, 1, 1, 1) * x0 + sig.view(-1, 1, 1, 1) * eps
    v = v_target(x0, eps, a, sig)
    assert torch.allclose(v, a.view(-1, 1, 1, 1) * eps - sig.view(-1, 1, 1, 1) * x0)
    assert torch.allclose(x0_from_v(z_t, v, a, sig), x0, atol=1e-5)
    assert torch.allclose(eps_from_v(z_t, v, a, sig), eps, atol=1e-5)


def test_min_snr_weight():
    snr = torch.tensor([0.01, 1.0, 5.0, 100.0])
    assert torch.equal(min_snr_weight(snr, None), torch.ones(4))
    w = min_snr_weight(snr, 5.0)
    assert torch.allclose(w, torch.minimum(snr, torch.tensor(5.0)) / (snr + 1))
    assert torch.all(w <= 1) and w[-1] < w[1]               # easy, low-noise steps are capped


# ============================================================================
# the denoiser
# ============================================================================

def test_denoiser_shapes_and_conditioning():
    m = tiny_unet()
    assert_no_batchnorm(m)
    assert m.null_class == 3 and m.class_embed.num_embeddings == 4       # 3 classes + the null token
    z, t = torch.randn(4, 2, 3, 3), torch.tensor([0, 10, 25, 49])
    assert torch.equal(m(z, t, torch.tensor([0, 1, 2, 3])), torch.zeros(4, 2, 3, 3))   # zero-init: v_hat = 0
    live = tiny_unet(live=True)
    v = {c: live(z, t, torch.full((4,), c)) for c in range(4)}
    assert v[0].shape == (4, 2, 3, 3)
    assert all(not torch.equal(v[a], v[b]) for a in range(4) for b in range(a + 1, 4))   # each id conditions
    assert not torch.equal(live(z, t, torch.zeros(4, dtype=torch.long)),
                           live(z, t.flip(0), torch.zeros(4, dtype=torch.long)))       # and the timestep
    with pytest.raises(ValueError, match="int tensor"):
        live(z, t, torch.zeros(4))
    with pytest.raises(IndexError):
        live(z, t, torch.full((4,), 4))                    # beyond the null token
    with pytest.raises(ValueError, match="latents"):
        live(torch.randn(4, 2, 6, 6), t, torch.zeros(4, dtype=torch.long))


def test_default_size_follows_the_plan():
    m = LatentUNet()                                       # 4x6x6 latents, 3 classes
    assert m.latent_shape == (4, 6, 6) and 6e6 <= count_parameters(m) <= 8e6
    assert_no_batchnorm(m)
    z = torch.randn(2, 4, 6, 6)
    assert m(z, torch.tensor([0, 999]), torch.tensor([1, 3])).shape == z.shape


def test_export_round_trip_is_weights_only(tmp_path):
    m = tiny_unet(3, live=True)
    torch.save(m.export_state(source="test"), tmp_path / "d.pt")
    back = LatentUNet.from_export(torch.load(tmp_path / "d.pt", map_location="cpu", weights_only=True))
    z, t, y = torch.randn(3, 2, 3, 3), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 3])
    with torch.no_grad():
        assert torch.equal(back(z, t, y), m(z, t, y))
    assert back.schedule.timesteps == 50 and not back.training


def test_cfg_null_token():
    rec = Recorder(tiny_unet(live=True))
    z, t, y = torch.randn(5, 2, 3, 3), torch.full((5,), 20), torch.tensor([0, 1, 2, 1, 0])
    with torch.no_grad():
        cond, null = rec.inner(z, t, y), rec.inner(z, t, torch.full_like(y, 3))
        rec.seen.clear()
        assert torch.equal(guided_v(rec, z, t, y, 1.0), cond) and [s.tolist() for s in rec.seen] == [y.tolist()]
        rec.seen.clear()
        assert torch.equal(guided_v(rec, z, t, y, 0.0), null) and [s.tolist() for s in rec.seen] == [[3] * 5]
        rec.seen.clear()
        assert torch.allclose(guided_v(rec, z, t, y, 1.5), null + 1.5 * (cond - null), atol=1e-6)
        assert len(rec.seen) == 2
    # label dropout: training replaces labels by the null token, evaluation never does
    batch = {"x": torch.randn(400, 2, 3, 3), "y": torch.arange(400) % 3}
    for p, info, lo, hi in ((1.0, TRAIN, 400, 400), (0.0, TRAIN, 0, 0), (1.0, EVAL, 0, 0), (0.25, TRAIN, 60, 140)):
        rec.seen.clear()
        torch.manual_seed(0)
        with torch.no_grad():
            DiffusionLoss(rec.schedule, label_dropout=p)(rec, batch, info)
        nulls = int((rec.seen[0] == 3).sum())
        assert lo <= nulls <= hi, (p, info, nulls)
        kept = rec.seen[0] != 3
        assert torch.equal(rec.seen[0][kept], batch["y"][kept])


def test_ddim_from_an_oracle_lands_on_the_data():
    """A denoiser that knows the data is one point x* predicts the exact v at
    every step: DDIM at eta = 0 must then return x* from any noise, for any
    guidance scale (the oracle ignores the class)."""
    target = torch.randn(1, 2, 3, 3)

    class Oracle(LatentUNet):
        def forward(self, z_t, t, y):
            a, s = self.schedule.alpha_sigma(t)
            a, s = a.view(-1, 1, 1, 1), s.view(-1, 1, 1, 1)
            eps = (z_t - a * target) / s
            return a * eps - s * target

    oracle = Oracle(latent_channels=2, latent_size=3, channels=(4,), num_classes=3, emb_dim=4, freq_dim=4)
    z_T = torch.randn(6, 2, 3, 3)
    for w in (1.0, 2.0):
        out = ddim_sample(oracle, z_T, torch.zeros(6, dtype=torch.long), w, steps=10)
        assert torch.allclose(out, target.expand(6, -1, -1, -1), atol=1e-4)


def test_ddim_is_deterministic_at_eta_0():
    m = tiny_unet(live=True)
    z_T, y = torch.randn(8, 2, 3, 3), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    a = ddim_sample(m, z_T, y, 1.5, steps=5)
    assert torch.equal(a, ddim_sample(m, z_T, y, 1.5, steps=5))
    assert not torch.equal(a, ddim_sample(m, z_T, y, 2.0, steps=5))
    s1 = sample_latents(m, 1, 10, 1.5, seed=7, steps=5)
    assert s1.shape == (10, 2, 3, 3) and s1.dtype == torch.float32
    assert torch.equal(s1, sample_latents(m, 1, 10, 1.5, seed=7, steps=5))
    assert not torch.equal(s1, sample_latents(m, 1, 10, 1.5, seed=8, steps=5))
    # the noise of row r does not depend on the chunking: other batch sizes / offsets agree up to rounding
    # (~1e-6 per forward pass on CPU, accumulated over the DDIM steps)
    parts = torch.cat([sample_latents(m, 1, 4, 1.5, seed=7, steps=5),
                       sample_latents(m, 1, 6, 1.5, seed=7, start=4, steps=5)])
    assert torch.allclose(parts, s1, atol=5e-4)
    assert torch.allclose(sample_latents(m, 1, 10, 1.5, seed=7, steps=5, batch_size=3), s1, atol=5e-4)
    # eta > 0 draws per-step noise, also from the seed
    e1 = sample_latents(m, 1, 10, 1.5, seed=7, steps=5, eta=1.0)
    assert torch.equal(e1, sample_latents(m, 1, 10, 1.5, seed=7, steps=5, eta=1.0)) and not torch.equal(e1, s1)
    with pytest.raises(ValueError, match="eta"):
        ddim_sample(m, z_T, y, 1.0, eta=-1.0)


def test_sample_decodes_to_the_export_format_and_its_sha_is_deterministic():
    ae, codec = tiny_ae_and_codec()
    m = tiny_unet(live=True)
    x = sample(ae, m, codec, 2, 10, 1.5, seed=3, steps=5, batch_size=4)
    assert x.shape == (10, 5, 5, 8) and x.dtype == np.float16
    assert np.isfinite(x.astype(np.float32)).all() and x.min() >= 0
    assert np.all(x.astype(np.float32) <= np.float16(codec.clip_hi.max()))
    assert sha(x) == sha(sample(ae, m, codec, 2, 10, 1.5, seed=3, steps=5, batch_size=4))
    other = sample(ae, m, codec, 2, 10, 1.5, seed=3, steps=5, batch_size=10)
    assert np.allclose(other.astype(np.float32), x.astype(np.float32), rtol=1e-2, atol=1e-2)
    assert sha(x) != sha(sample(ae, m, codec, 1, 10, 1.5, seed=3, steps=5, batch_size=4))
    for bad in (dict(class_id=3), dict(w=-1.0), dict(w=float("nan")), dict(n=-1)):
        args = {"class_id": 0, "n": 2, "w": 1.0, **bad}
        with pytest.raises(ValueError):
            sample(ae, m, codec, args["class_id"], args["n"], args["w"], seed=0, steps=2)
    with pytest.raises(ValueError, match="latent"):
        sample(ae, tiny_unet(latent_channels=3), codec, 0, 2, 1.0, seed=0, steps=2)


def test_the_denoiser_overfits_one_batch():
    m = tiny_unet().train()
    loss = DiffusionLoss(m.schedule, min_snr_gamma=5.0, label_dropout=0.1)
    batch = {"x": torch.randn(12, 2, 3, 3), "y": torch.arange(12) % 3}

    def held_loss() -> float:                       # fixed timesteps and noise
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(123)
            return float(np.mean([loss(m, batch, EVAL)[0].item() for _ in range(8)]))

    start = held_loss()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    torch.manual_seed(0)
    for _ in range(300):
        opt.zero_grad()
        loss(m, batch, TRAIN)[0].backward()
        opt.step()
    assert held_loss() < 0.5 * start


def test_bf16_autocast_keeps_the_loss_in_float32():
    m = tiny_unet(live=True).train()
    loss = DiffusionLoss(m.schedule)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        value, metrics = loss(m, {"x": torch.randn(6, 2, 3, 3), "y": torch.arange(6) % 3}, TRAIN)
    assert value.dtype == torch.float32 and math.isfinite(value.item())
    value.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_mmd_rbf():
    torch.manual_seed(0)
    x = torch.randn(32, 2, 3, 3)
    assert ld.mmd_rbf(x, x, [1.0, 2.0]) == pytest.approx(0.0, abs=1e-12)
    near, far = ld.mmd_rbf(x, x + 0.1, [3.0]), ld.mmd_rbf(x, x + 3.0, [3.0])
    assert 0 < near < far and ld.mmd_rbf(x + 3.0, x, [3.0]) == pytest.approx(far)


def test_config_validation():
    cfg = LDMLopoConfig()
    assert (cfg.max_steps, cfg.lr, cfg.warmup_steps, cfg.batch_size, cfg.ema_decay, cfg.precision) == \
        (150_000, 2e-4, 1000, 1024, 0.9995, "bf16")
    assert (cfg.timesteps, cfg.min_snr_gamma, cfg.label_dropout, cfg.sample_steps, cfg.guidance) == \
        (1000, 5.0, 0.1, 50, (1.0, 1.5, 2.0, 3.0))
    with pytest.raises(ConfigError, match="pulls samples toward the training modes"):
        LDMLopoConfig(guidance=(1.0, 3.5))
    with pytest.raises(ConfigError, match="null token"):
        LDMLopoConfig(label_dropout=0.0)
    assert LDMLopoConfig(label_dropout=0.0, guidance=(1.0,)).label_dropout == 0.0
    with pytest.raises(ConfigError, match="label_dropout"):
        LDMLopoConfig(label_dropout=1.0)
    with pytest.raises(ConfigError, match="sample_steps"):
        LDMLopoConfig(timesteps=50, sample_steps=60)


# ============================================================================
# the stages, on the tiny build's pool
# ============================================================================

def test_registration(registry):
    for name in ("ldm-lopo", "ldm-final"):
        spec = registry.get(name)
        assert spec.group == "fit-ldm" and spec.purposes == ("generator",) and not spec.in_all
    assert registry.get("ldm-lopo").requires == ("ae-lopo",)
    assert registry.get("ldm-final").requires == ("ae-final", "ldm-lopo")
    assert [s.name for s in registry.resolve(["fit-ae", "fit-ldm"])] == ["ae-lopo", "ae-final", "ldm-lopo",
                                                                         "ldm-final"]


def test_lopo_writes_one_directory_per_fold(ran):
    done = ran.lopo.done
    assert done["status"] == "ok" and done["upstream"] == {"ae-lopo": ran.ae_lopo.done["fingerprint"]}
    s = done["summary"]
    assert s["pool"]["pool_sha256"] == pool_fingerprint(ran.out / "pool")
    assert s["guidance_candidates"] == [1.0, 2.0] and "picks w" in s["guidance_choice"]
    for k in (0, 1):
        d = ran.lopo_dir / f"fold-{k}"
        for name in FOLD_FILES:
            assert (d / name).is_file() and done["outputs"][f"fold-{k}/{name}"]["sha256"] == sha256_file(d / name)
        f = s["folds"][k]
        assert f["fold"]["heldout_patients"] == [int(p) for p in ran.folds[k]["heldout_patients"]]
        assert f["finished"] and f["best_step"] in (3, 6) and f["latent_shape"] == [2, 3, 3]
        assert f["ae_sha256"] == sha256_file(ran.out / "gen/ae/lopo" / f"fold-{k}" / ld.AE_EXPORT_NAME)


def test_latents_are_the_folds_autoencoder_encoding_cached(ran):
    X = torch.from_numpy(np.asarray(ran.pool.X, dtype=np.float32)).permute(0, 3, 1, 2)
    for k in (0, 1):
        d = ran.lopo_dir / f"fold-{k}"
        ae_path = ran.out / "gen/ae/lopo" / f"fold-{k}" / ld.AE_EXPORT_NAME
        ae = SpatialAutoencoder.from_export(torch.load(ae_path, map_location="cpu", weights_only=True))
        lat = np.load(d / ld.LATENTS_NAME)
        meta = json.loads((d / ld.LATENTS_META_NAME).read_text())
        assert meta["ae_sha256"] == sha256_file(ae_path) and meta["pool_sha256"] == pool_fingerprint(ran.out / "pool")
        assert lat.dtype == np.float32 and list(lat.shape) == meta["shape"] == [ran.pool.n, 2, 3, 3]
        with torch.no_grad():
            assert np.allclose(lat, ae.encode_scaled(X).numpy(), atol=1e-5)
        train_mask, _ = fold_masks(ran.pool.patient, ran.folds[k])
        assert lat[train_mask].std() == pytest.approx(1.0, rel=1e-4)   # unit variance on the fold's train rows


def test_held_out_rows_never_reach_a_training_batch(ran):
    assert ran.spy.fits == 2
    for k in (0, 1):
        train_mask, held_mask = fold_masks(ran.pool.patient, ran.folds[k])
        trained = ran.spy.rows(k + 1, training=True)
        assert ran.spy.steps(k + 1) == list(range(6)) and trained.size == 6 * 12
        assert not held_mask[trained].any()
        evaluated = ran.spy.rows(k + 1, training=False)
        assert set(evaluated.tolist()) == set(np.flatnonzero(held_mask).tolist())


def test_the_sampling_sha_is_reproduced_from_generator_pt(ran):
    for d, summary in [(ran.lopo_dir / f"fold-{k}", ran.lopo.done["summary"]["folds"][k]) for k in (0, 1)] + \
                      [(ran.final_dir, ran.final.done["summary"])]:
        check = summary["sample_check"]
        assert torch.load(d / ld.GENERATOR_NAME, map_location="cpu", weights_only=True)["format"] == \
            ld.GENERATOR_FORMAT
        gen = ld.load_generator(d)
        assert check["class_id"] == 1 and check["finite"] and check["device"] == "cpu"   # DCIS, the minority
        x = gen.sample(check["class_id"], check["n"], check["w"], check["seed"], steps=check["steps"],
                       batch_size=check["batch_size"])
        assert sha(x) == check["sha256"] and list(x.shape) == check["shape"] == [64, 5, 5, 8]
        assert sha(sample(gen.ae, gen.denoiser, gen.codec, check["class_id"], check["n"], check["w"], check["seed"],
                          steps=check["steps"], batch_size=check["batch_size"])) == check["sha256"]
        ckpt = load_checkpoint(d / BEST_NAME)                # generator.pt = EMA weights of best.pt
        assert all(torch.equal(v, ckpt["ema"]["module"][n]) for n, v in gen.denoiser.state_dict().items())
        assert gen.meta["class_names"] == ["healthy", "DCIS", "IDC"] and gen.meta["guidance_candidates"] == [1.0, 2.0]


def test_latent_mmd_per_class_and_guidance_scale(ran):
    for k in (0, 1):
        mmd = json.loads((ran.lopo_dir / f"fold-{k}" / ld.MMD_NAME).read_text())
        assert mmd["pass_ratio"] == 1.5 and set(mmd["classes"]) == {"healthy", "DCIS", "IDC"}
        for c, rec in mmd["classes"].items():
            assert rec["n"] == 16 and rec["mmd_train_heldout"] > 0 and len(rec["bandwidths"]) == 3
            assert set(rec["by_w"]) == {"1", "2"}
            for w in rec["by_w"].values():
                assert w["mmd"] >= 0 and math.isfinite(w["ratio"]) and isinstance(w["pass"], bool)
                assert w["pass"] == (w["ratio"] <= 1.5)
        ratios = ran.lopo.done["summary"]["folds"][k]["latent_mmd_ratio"]
        assert ratios == {c: {w: r["ratio"] for w, r in v["by_w"].items()} for c, v in mmd["classes"].items()}


def test_the_final_refit(ran):
    s, lopo = ran.final.done["summary"], ran.lopo.done
    best = [f["best_step"] for f in lopo["summary"]["folds"]]
    budget = int(math.floor(float(np.median(best)) + 0.5))
    assert s["steps"] == {"rule": "median_best", "steps": budget, "lopo_best_steps": best}
    assert load_checkpoint(ran.final_dir / BEST_NAME)["step"] == budget
    assert ran.final.done["upstream"] == {"ae-final": ran.ae_final.done["fingerprint"],
                                          "ldm-lopo": lopo["fingerprint"]}
    assert s["ae_sha256"] == sha256_file(ran.out / "gen/ae/final" / ld.AE_EXPORT_NAME)
    assert not (ran.final_dir / ld.MMD_NAME).exists() and "latent_mmd_ratio" not in s
    assert s["train_config"]["precision"] == "fp32" and s["train_config"]["guidance"] == [1.0, 2.0]
    for name in (ld.LATENTS_NAME, ld.LATENTS_META_NAME, BEST_NAME, LAST_NAME, HISTORY_NAME, ld.GENERATOR_NAME):
        assert name in ran.final.done["outputs"]


def test_the_access_log_shows_train_only(ran):
    stages = ("ae-lopo", "ae-final", "ldm-lopo", "ldm-final")
    rec = [r for r in read_access_log(ran.out / "split_access_log.jsonl") if r["stage"] in stages]
    assert {r["stage"] for r in rec} == set(stages)
    assert all(r["split"] == "train" and r["allowed"] and r["purpose"] == "generator" for r in rec)


def test_refusals(ran, tiny, registry, tmp_path):
    out = prepare(tiny, registry, tmp_path / "aug")
    with pytest.raises(StageNotReady, match="ae-lopo"):
        run(registry, tiny, out, "ldm-lopo", ldm_lopo_cfg())
    run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())
    with pytest.raises(StageError, match=r"fold\(s\) \[2\]"):
        run(registry, tiny, out, "ldm-lopo", ldm_lopo_cfg(folds=(0, 2)))
    ae_pt = out / "gen/ae/lopo/fold-1" / ld.AE_EXPORT_NAME
    ae_pt.write_bytes(ae_pt.read_bytes() + b"\0")
    with pytest.raises(StageError, match="changed after ae-lopo finished"):
        run(registry, tiny, out, "ldm-lopo", ldm_lopo_cfg(folds=(1,)))
    assert not (out / "gen/ldm/lopo" / DONE_NAME).exists()
    with pytest.raises(StageNotReady, match="ae-final"):
        run(registry, tiny, out, "ldm-final", ldm_final_cfg())


def test_resume_auto_is_bitwise_inside_both_stages(ran, tiny, registry, tmp_path, monkeypatch):
    out = prepare(tiny, registry, tmp_path / "run")
    run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())
    run(registry, tiny, out, "ae-final", ae_final_cfg())
    lopo = out / "gen/ldm/lopo"
    spy = StepSpy(monkeypatch, DiffusionLoss)
    spy.crash = (2, 4)                                           # fold 1, after its step-4 checkpoint
    with pytest.raises(RuntimeError, match="killed"):
        run(registry, tiny, out, "ldm-lopo", ldm_lopo_cfg())
    assert not (lopo / DONE_NAME).exists() and (lopo / "fold-0" / ld.FOLD_DONE_NAME).is_file()
    assert load_checkpoint(lopo / "fold-1" / LAST_NAME)["step"] == 4
    spy.crash = None
    r = run(registry, tiny, out, "ldm-lopo", ldm_lopo_cfg(), resume="auto")
    assert r.done["resumed"] and r.done["attempt"] == 2
    log = (lopo / "stage.log").read_text()
    assert "fold-0: finished in an earlier attempt; kept" in log and "fold-1: cached latents" in log
    for k in (0, 1):
        a, b = ran.lopo_dir / f"fold-{k}", lopo / f"fold-{k}"
        assert_same_training(a, b)
        assert np.array_equal(np.load(a / ld.LATENTS_NAME), np.load(b / ld.LATENTS_NAME))
        assert (a / ld.MMD_NAME).read_text() == (b / ld.MMD_NAME).read_text()
        ga = torch.load(a / ld.GENERATOR_NAME, weights_only=True)
        gb = torch.load(b / ld.GENERATOR_NAME, weights_only=True)
        assert_same(ga["denoiser"]["state_dict"], gb["denoiser"]["state_dict"])
        fa, fb = ran.lopo.done["summary"]["folds"][k], r.done["summary"]["folds"][k]
        assert {x: fa[x] for x in FOLD_SUMMARY_KEYS} == {x: fb[x] for x in FOLD_SUMMARY_KEYS}
        assert fa["sample_check"]["sha256"] == fb["sample_check"]["sha256"]

    spy.crash = (spy.fits + 1, 2)                                # the final refit, after its step-2 checkpoint
    with pytest.raises(RuntimeError, match="killed"):
        run(registry, tiny, out, "ldm-final", ldm_final_cfg())
    spy.crash = None
    r = run(registry, tiny, out, "ldm-final", ldm_final_cfg(), resume="auto")
    assert r.done["resumed"]
    assert_same_training(ran.final_dir, out / "gen/ldm/final")
    assert r.done["summary"]["sample_check"]["sha256"] == ran.final.done["summary"]["sample_check"]["sha256"]


def test_a_changed_pool_is_refused_by_ldm_final(tiny, registry, tmp_path):
    out = prepare(tiny, registry, tmp_path / "aug")
    for name, cfg in (("ae-lopo", ae_lopo_cfg()), ("ae-final", ae_final_cfg()), ("ldm-lopo", ldm_lopo_cfg())):
        run(registry, tiny, out, name, cfg)
    pool = load_pool(out / "pool")
    X = np.array(pool.X)
    X[1, 1, 1, 1] += np.float16(0.5)
    write_pool(out / "pool", X, pool.y, pool.patient, pool.row, pool.capture, pool.codec)
    with pytest.raises(StageError, match="not the one ae-final was trained on"):
        run(registry, tiny, out, "ldm-final", ldm_final_cfg())


# ============================================================================
# command line
# ============================================================================

def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600)


def test_cli_fit_ae_and_fit_ldm_groups(tiny, registry, tmp_path):
    r = _cli("ldm-lopo", "--help")
    assert r.returncode == 0 and "--guidance" in r.stdout and "--label_dropout" in r.stdout
    r = _cli("ae-lopo", "--help")
    assert r.returncode == 0 and "--latent_channels" in r.stdout and "--sam_angle_eps" not in r.stdout
    out = prepare(tiny, registry, tmp_path / "aug")
    cfg = {"stages": {"ae-lopo": ae_lopo_cfg(max_steps=3, eval_every=3).to_dict(),
                      "ae-final": ae_final_cfg().to_dict(),
                      "ldm-lopo": ldm_lopo_cfg(max_steps=3, eval_every=3).to_dict(),
                      "ldm-final": ldm_final_cfg().to_dict()}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    base = ("--data_dir", tiny.data_dir, "--out", out, "--config", tmp_path / "cfg.json")
    for group in ("fit-ae", "fit-ldm"):
        r = _cli(group, *base)
        assert r.returncode == 0, r.stdout + r.stderr
    for stage in ("ae/lopo", "ae/final", "ldm/lopo", "ldm/final"):
        assert read_done(out / "gen" / stage)["status"] == "ok"
    r = _cli("fit-ae", "fit-ldm", *base)
    assert r.returncode == 0 and r.stdout.count("already done") == 4
