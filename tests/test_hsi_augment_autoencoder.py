# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_autoencoder.py
=====================================
Phase 7 (#24), arm A3, first half: the unconditional spatial autoencoder
(`hsi_augment/models/autoencoder.py`) and its stages `ae-lopo` / `ae-final`
(group `fit-ae`, `hsi_augment/latent_diffusion.py`), which read the generator
pool through `hsi_augment/pool_io.py`.

Covered: shapes and the Gaussian posterior; the weights_only export and the
latent scale; the loss (its SAM term has no 2.56 deg floor); overfitting one
batch; and, through the stage API on the tiny build's pool, the fold layout,
fold isolation (a held-out patient's rows never reach a training batch or the
latent scale), the final refit's step budget, done.json / fingerprint /
skip-if-done behaviour, a changed or foreign pool being refused, the access
log (train only), and bitwise resume inside both stages.

The denoiser, the sampler and the `fit-ldm` stages are in
`test_hsi_augment_diffusion.py`; the shared helpers in
`tests/hsi_augment_ldm_fixtures.py`.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hsi_augment import latent_diffusion as ld
from hsi_augment.codec import make_decoded_sam_loss
from hsi_augment.config import ConfigError
from hsi_augment.io import read_access_log, sha256_file
from hsi_augment.models.autoencoder import (SAM_ANGLE_EPS, AELoss, SpatialAutoencoder, count_parameters,
                                            kl_to_standard_normal, latent_size, make_sam_loss)
from hsi_augment.models.blocks import assert_no_batchnorm
from hsi_augment.pool_io import load_pool, pool_fingerprint, write_pool
from hsi_augment.registry import load_builtin_stages
from hsi_augment.splits import fold_masks, registry_policy_violations
from hsi_augment.stage import DONE_NAME, StageError, StageNotReady, stage_status
from hsi_augment.stages.fit_ae import AELopoConfig
from hsi_augment.train_generative import BEST_NAME, HISTORY_NAME, LAST_NAME, load_checkpoint
from tests.hsi_augment_fixtures import make_tiny_build
from tests.hsi_augment_ldm_fixtures import (FOLD_SUMMARY_KEYS, TINY_AE, StepSpy, ae_final_cfg, ae_lopo_cfg,
                                            assert_same, assert_same_training, opts, prepare, run, tiny_codec)

FOLD_FILES = (BEST_NAME, LAST_NAME, HISTORY_NAME, ld.AE_EXPORT_NAME, ld.RECON_NAME, ld.FOLD_DONE_NAME)


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
    """leakage -> pool -> ae-lopo (every loss call recorded) -> ae-final."""
    out = prepare(tiny, registry, tmp_path_factory.mktemp("ae"))
    with pytest.MonkeyPatch.context() as mp:
        spy = StepSpy(mp, AELoss)
        lopo = run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())
    final = run(registry, tiny, out, "ae-final", ae_final_cfg())
    folds = json.loads((out / "prep/leakage/folds.json").read_text())["folds"]
    return SimpleNamespace(out=out, lopo=lopo, final=final, spy=spy, folds=folds, pool=load_pool(out / "pool"),
                           lopo_dir=out / "gen/ae/lopo", final_dir=out / "gen/ae/final")


def tiny_ae(seed: int = 0) -> SpatialAutoencoder:
    torch.manual_seed(seed)
    return SpatialAutoencoder(in_channels=8, patch=5, **TINY_AE)


def tiny_batch(tiny, n: int = 12):
    X = np.load(tiny.data_dir / "X_train.npy")
    y = np.load(tiny.data_dir / "y_train.npy")
    codec = tiny_codec(X)
    rows = np.concatenate([np.flatnonzero(y == c)[: n // 3] for c in range(3)])
    x = torch.from_numpy(codec.encode_np(X[rows])).permute(0, 3, 1, 2).contiguous()
    return {"x": x, "y": torch.from_numpy(y[rows].astype(np.int64))}, codec


TRAIN = SimpleNamespace(training=True)
EVAL = SimpleNamespace(training=False)


# ============================================================================
# the model
# ============================================================================

def test_shapes_posterior_and_no_batchnorm():
    assert (latent_size(11), latent_size(5), latent_size(11, 2)) == (6, 3, 3)
    ae = tiny_ae()
    assert_no_batchnorm(ae)
    assert ae.latent_shape == (2, 3, 3) and ae.latent_dim == 18
    x = torch.randn(4, 8, 5, 5)
    mean, logvar = ae.encode(x)
    assert mean.shape == logvar.shape == (4, 2, 3, 3)
    x_hat, m, lv, z = ae(x, sample=False)
    assert x_hat.shape == x.shape and torch.equal(z, m)
    torch.manual_seed(1)
    _, _, _, z1 = ae(x)                          # sampled: mean + sigma * eps
    assert not torch.equal(z1, m)
    with pytest.raises(ValueError, match="log-z NCHW"):
        ae.encode(torch.randn(4, 5, 5, 8))       # NHWC by mistake
    with pytest.raises(ValueError, match="latents"):
        ae.decode(torch.randn(4, 3, 3, 3))


def test_default_sizes_follow_the_plan():
    ae = SpatialAutoencoder()                    # 11x11x32 patches
    assert ae.latent_shape == (4, 6, 6) and ae.latent_dim == 144     # 27x compression
    assert 3.0e6 <= count_parameters(ae) <= 4.5e6
    assert_no_batchnorm(ae)


def test_export_round_trip_is_weights_only_and_keeps_the_latent_scale(tmp_path):
    ae = tiny_ae(3)
    ae.latent_scale = 0.37
    torch.save(ae.export_state(source="test"), tmp_path / "ae.pt")
    state = torch.load(tmp_path / "ae.pt", map_location="cpu", weights_only=True)
    back = SpatialAutoencoder.from_export(state)
    assert back.latent_scale == 0.37 and back.kwargs == ae.kwargs and not back.training
    x = torch.randn(3, 8, 5, 5)
    ae.eval()
    with torch.no_grad():
        assert torch.equal(back.encode_scaled(x), ae.encode(x)[0] * 0.37)
        z = back.encode_scaled(x)
        assert torch.allclose(back.decode_scaled(z), ae.decode(ae.encode(x)[0]), atol=1e-6)
    with pytest.raises(ValueError, match="export"):
        SpatialAutoencoder.from_export({**state, "format": "other"})


def test_kl_and_the_loss_terms(tiny):
    mean, logvar = torch.zeros(2, 2, 3, 3), torch.zeros(2, 2, 3, 3)
    assert torch.equal(kl_to_standard_normal(mean, logvar), torch.zeros(2))
    kl = kl_to_standard_normal(torch.ones(2, 2, 3, 3), logvar)
    assert torch.allclose(kl, torch.full((2,), 0.5 * 18))          # 0.5 * mean^2 per element, summed
    batch, codec = tiny_batch(tiny)
    loss, metrics = AELoss(make_sam_loss(codec))(tiny_ae(), batch, EVAL)
    assert loss.ndim == 0 and math.isfinite(loss.item())
    assert set(metrics) == {"mse", "kl", "kl_per_dim", "sam_deg"}
    expected = metrics["mse"] + 1e-6 * metrics["kl"] + 0.1 * metrics["sam_deg"] * math.pi / 180
    assert torch.allclose(loss, expected, rtol=1e-5)
    loss0, m0 = AELoss(None, sam_weight=0.0)(tiny_ae(), batch, EVAL)
    assert "sam_deg" not in m0
    with pytest.raises(ValueError, match="sam_loss"):
        AELoss(None, sam_weight=0.1)


def test_the_sam_term_has_no_2_56_degree_floor(tiny):
    """The user decision: SAM loss terms use angle_eps 1e-6. Identical spectra
    score ~0.08 deg (the frozen default would say 2.56) and a small error
    still has a gradient."""
    batch, codec = tiny_batch(tiny)
    x = batch["x"]
    sam = make_sam_loss(codec)
    assert SAM_ANGLE_EPS == 1e-6 and sam.sam.angle_eps == SAM_ANGLE_EPS
    assert math.degrees(sam(x, x).item()) < 0.1
    assert math.degrees(make_decoded_sam_loss(codec)(x, x).item()) == pytest.approx(2.5626, abs=1e-3)
    torch.manual_seed(0)
    noise = 0.01 * torch.randn_like(x)                            # angles well under 2.56 deg
    pred = (x + noise).requires_grad_(True)
    sam(x, pred).backward()
    assert pred.grad.abs().sum() > 0
    floored = (x + noise).requires_grad_(True)
    make_decoded_sam_loss(codec)(x, floored).backward()
    assert floored.grad.abs().sum() == 0                          # what the 2.56 deg floor would do
    assert "sam_angle_eps" not in AELopoConfig().to_dict()      # not a flag: fixed at 1e-6


def test_the_autoencoder_overfits_one_batch(tiny):
    batch, codec = tiny_batch(tiny)
    ae = tiny_ae()
    loss = AELoss(make_sam_loss(codec))
    opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
    with torch.no_grad():
        start = loss(ae, batch, EVAL)[0].item()
    for _ in range(200):
        opt.zero_grad()
        loss(ae, batch, TRAIN)[0].backward()
        opt.step()
    with torch.no_grad():
        end, metrics = loss(ae, batch, EVAL)
    assert end.item() < 0.2 * start and metrics["mse"].item() < 0.1


def test_config_validation():
    with pytest.raises(ConfigError, match="folds"):
        AELopoConfig(folds=(0, 0))
    with pytest.raises(ConfigError, match="device"):
        AELopoConfig(device="tpu")
    with pytest.raises(ConfigError, match="warmup"):
        AELopoConfig(max_steps=10)                 # the default warm-up (500) must stay below it
    cfg = AELopoConfig()
    assert (cfg.max_steps, cfg.sam_weight, cfg.kl_weight, cfg.latent_channels, cfg.folds) == \
        (30_000, 0.1, 1e-6, 4, (0, 1))


# ============================================================================
# the stages, on the tiny build's pool
# ============================================================================

def test_registration_reads_train_only_and_stays_out_of_all(registry):
    for name in ("ae-lopo", "ae-final"):
        spec = registry.get(name)
        assert spec.group == "fit-ae" and spec.purposes == ("generator",) and not spec.reads_held_out()
        assert not spec.in_all                   # until the pool stage (#16) is wired in
        assert "pool" not in spec.requires
    assert registry.get("ae-lopo").requires == ("leakage",) and registry.get("ae-final").requires == ("ae-lopo",)
    assert registry_policy_violations(registry) == []
    assert [s.name for s in registry.resolve(["fit-ae"])] == ["ae-lopo", "ae-final"]


def test_lopo_writes_one_directory_per_fold(ran):
    done = ran.lopo.done
    assert done["status"] == "ok" and not done["resumed"]
    summary = done["summary"]
    assert [f["fold"]["fold"] for f in summary["folds"]] == [0, 1]
    for k in (0, 1):
        d = ran.lopo_dir / f"fold-{k}"
        for name in FOLD_FILES:
            assert (d / name).is_file(), name
            assert done["outputs"][f"fold-{k}/{name}"]["sha256"] == sha256_file(d / name)
        marker = json.loads((d / ld.FOLD_DONE_NAME).read_text())
        assert marker["fingerprint"] == done["fingerprint"]
        assert marker["ident"]["pool_sha256"] == summary["pool"]["pool_sha256"]
        assert marker["summary"]["best_step"] == summary["folds"][k]["best_step"]
        f = summary["folds"][k]
        assert f["fold"]["heldout_patients"] == [int(p) for p in ran.folds[k]["heldout_patients"]]
        assert f["finished"] and f["best_step"] in (3, 6) and f["latent_shape"] == [2, 3, 3]
        recon = json.loads((d / ld.RECON_NAME).read_text())
        for side in ("train", "heldout"):
            for c in ("healthy", "DCIS", "IDC", "all"):
                m = recon[side][c]
                assert m["n"] > 0 and m["sam_deg"] >= 0 and m["mse_logz"] > 0 and -1 <= m["ssim2d"] <= 1
        assert set(recon["heldout_over_train_sam"]) == {"healthy", "DCIS", "IDC", "all"}


def test_the_pool_fingerprint_is_recorded(ran):
    pool = ran.lopo.done["summary"]["pool"]
    assert pool["pool_sha256"] == pool_fingerprint(ran.out / "pool") == ran.pool.fingerprint()
    assert pool["n"] == ran.pool.n and pool["pool_dir_setting"] is None
    assert ran.final.done["summary"]["pool"]["pool_sha256"] == pool["pool_sha256"]
    state = torch.load(ran.lopo_dir / "fold-0" / ld.AE_EXPORT_NAME, map_location="cpu", weights_only=True)
    assert state["meta"]["pool_sha256"] == pool["pool_sha256"]


def test_held_out_rows_never_reach_a_training_batch(ran):
    """Fold isolation: fit k's training batches hold only fold k's train-side
    rows; its held-out evaluation holds exactly the held-out patients' rows."""
    assert ran.spy.fits == 2
    for k in (0, 1):
        train_mask, held_mask = fold_masks(ran.pool.patient, ran.folds[k])
        fit = k + 1
        trained = ran.spy.rows(fit, training=True)
        assert ran.spy.steps(fit) == list(range(6)) and trained.size == 6 * 12
        assert not held_mask[trained].any()
        assert not np.isin(ran.pool.patient[trained], ran.folds[k]["heldout_patients"]).any()
        evaluated = ran.spy.rows(fit, training=False)
        assert set(evaluated.tolist()) == set(np.flatnonzero(held_mask).tolist())


def test_the_latent_scale_comes_from_the_train_rows_only(ran):
    X = torch.from_numpy(np.asarray(ran.pool.X, dtype=np.float32)).permute(0, 3, 1, 2)
    for k in (0, 1):
        state = torch.load(ran.lopo_dir / f"fold-{k}" / ld.AE_EXPORT_NAME, map_location="cpu", weights_only=True)
        ae = SpatialAutoencoder.from_export(state)
        train_mask, _ = fold_masks(ran.pool.patient, ran.folds[k])
        with torch.no_grad():
            mean = ae.encode(X)[0].double()
        assert ae.latent_scale == pytest.approx(1.0 / mean[torch.from_numpy(train_mask)].std(unbiased=False).item(),
                                                rel=1e-5)
        assert ae.latent_scale != pytest.approx(1.0 / mean.std(unbiased=False).item(), rel=1e-5)
        ckpt = load_checkpoint(ran.lopo_dir / f"fold-{k}" / BEST_NAME)    # ae.pt = EMA weights of best.pt
        assert all(torch.equal(v, ckpt["ema"]["module"][n]) for n, v in state["state_dict"].items())
        assert state["meta"]["step"] == ckpt["step"]


def test_the_final_refit_uses_lopo_settings_and_the_median_best_step(ran):
    lopo, final = ran.lopo.done, ran.final.done
    best = [f["best_step"] for f in lopo["summary"]["folds"]]
    budget = int(math.floor(float(np.median(best)) + 0.5))
    s = final["summary"]
    assert s["steps"] == {"rule": "median_best", "steps": budget, "lopo_best_steps": best}
    ckpt = load_checkpoint(ran.final_dir / BEST_NAME)
    assert ckpt["step"] == budget and ckpt["finished"] and ckpt["ema"] is not None
    expect = AELopoConfig.from_dict(lopo["config"]).replace(max_steps=budget, warmup_steps=min(1, budget - 1),
                                                            early_stop_patience=None)
    assert s["train_config"] == expect.to_dict() and ckpt["config"] == expect.to_dict()
    assert final["upstream"] == {"ae-lopo": lopo["fingerprint"]}
    for name in (BEST_NAME, LAST_NAME, HISTORY_NAME, ld.AE_EXPORT_NAME, ld.RECON_NAME):
        assert (ran.final_dir / name).is_file() and name in final["outputs"]
    recon = json.loads((ran.final_dir / ld.RECON_NAME).read_text())
    assert set(recon) == {"train"}                                  # no held-out data in a final refit
    ae = SpatialAutoencoder.from_export(torch.load(ran.final_dir / ld.AE_EXPORT_NAME, map_location="cpu",
                                                   weights_only=True))
    assert ae.latent_scale == pytest.approx(s["latent_scale"])


def test_every_checkpoint_loads_weights_only(ran):
    paths = sorted(ran.lopo_dir.glob("fold-*/*.pt")) + sorted(ran.final_dir.glob("*.pt"))
    assert len(paths) == 9                                        # 3 per fold, 3 final
    for p in paths:
        assert isinstance(torch.load(p, map_location="cpu", weights_only=True), dict)


def test_the_access_log_shows_train_only(ran):
    rec = [r for r in read_access_log(ran.out / "split_access_log.jsonl") if r["stage"] in ("ae-lopo", "ae-final")]
    assert {r["stage"] for r in rec} == {"ae-lopo", "ae-final"}
    assert all(r["split"] == "train" and r["allowed"] and r["purpose"] == "generator" for r in rec)
    assert all(not f.startswith("X_") for r in rec for f in r["files"])      # sidecars only; X is the pool's


def test_skip_if_done_and_the_fingerprint_chain(ran, registry, tiny):
    r = run(registry, tiny, ran.out, "ae-lopo", ae_lopo_cfg())
    assert r.status == "skipped" and r.fingerprint == ran.lopo.done["fingerprint"]
    assert stage_status(registry.get("ae-final"), ae_final_cfg(), opts(tiny, ran.out), registry)[0] == "done"
    status, why = stage_status(registry.get("ae-lopo"), ae_lopo_cfg(kl_weight=2e-6), opts(tiny, ran.out), registry)
    assert status == "stale" and "fingerprint" in why
    assert stage_status(registry.get("ae-final"), ae_final_cfg(steps="lopo_max_steps"), opts(tiny, ran.out),
                        registry)[0] == "stale"


def test_force_refits_and_reproduces_the_same_bytes(ran, registry, tiny, tmp_path):
    """Same seed, config and pool in another --out: the same weights."""
    out = prepare(tiny, registry, tmp_path / "again")
    r = run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())
    assert r.fingerprint == ran.lopo.done["fingerprint"]          # the pool is not in it (no pool stage yet) ...
    for k in (0, 1):
        assert_same_training(ran.lopo_dir / f"fold-{k}", out / "gen/ae/lopo" / f"fold-{k}")
    r2 = run(registry, tiny, out, "ae-lopo", ae_lopo_cfg(), force=True)
    assert r2.status == "ran" and r2.done["attempt"] == 1
    for k in (0, 1):
        assert_same_training(ran.lopo_dir / f"fold-{k}", out / "gen/ae/lopo" / f"fold-{k}")


def test_a_changed_pool_is_refused(tiny, registry, tmp_path, monkeypatch):
    """... so the pool's sha256 guards it: ae-final refuses another pool, and
    an interrupted ae-lopo will not resume a fold on one."""
    out = prepare(tiny, registry, tmp_path / "aug")
    spy = StepSpy(monkeypatch, AELoss)
    spy.crash = (2, 4)                                            # fold 1, after its step-4 checkpoint
    with pytest.raises(RuntimeError, match="killed"):
        run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())
    pool = load_pool(out / "pool")
    X = np.array(pool.X)
    X[0, 0, 0, 0] += np.float16(0.5)
    write_pool(out / "pool", X, pool.y, pool.patient, pool.row, pool.capture, pool.codec)
    spy.crash = None
    with pytest.raises(StageError, match="fold-0: cannot resume"):
        run(registry, tiny, out, "ae-lopo", ae_lopo_cfg(), resume="auto")
    run(registry, tiny, out, "ae-lopo", ae_lopo_cfg(), force=True)  # a fresh run on the new pool is fine
    X[0, 0, 0, 0] -= np.float16(0.25)
    write_pool(out / "pool", X, pool.y, pool.patient, pool.row, pool.capture, pool.codec)
    with pytest.raises(StageError, match="not the one ae-lopo was trained on"):
        run(registry, tiny, out, "ae-final", ae_final_cfg())
    assert not (out / "gen/ae/final" / DONE_NAME).exists()


def test_a_missing_or_foreign_pool_is_a_stage_error(tiny, registry, tmp_path):
    out = prepare(tiny, registry, tmp_path / "aug", pool=False)
    with pytest.raises(StageError, match="no pool directory"):
        run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())
    assert not (out / "gen/ae/lopo" / DONE_NAME).exists()
    d = tiny.data_dir
    X = np.load(d / "X_train.npy")
    codec = tiny_codec(X)
    groups = np.load(d / "groups_train.npy")
    write_pool(out / "pool", codec.encode_np(X).astype(np.float16), np.load(d / "y_train.npy"), groups[::-1],
               np.arange(X.shape[0]), np.load(d / "captures_train.npy"), codec)
    with pytest.raises(StageError, match="does not belong to the train split"):
        run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())


def test_final_refuses_to_run_without_lopo(tiny, registry, tmp_path):
    out = prepare(tiny, registry, tmp_path / "aug")
    with pytest.raises(StageNotReady, match="ae-lopo"):
        run(registry, tiny, out, "ae-final", ae_final_cfg())


def test_resume_auto_is_bitwise_inside_both_stages(ran, tiny, registry, tmp_path, monkeypatch):
    out = prepare(tiny, registry, tmp_path / "run")
    spy = StepSpy(monkeypatch, AELoss)
    spy.crash = (2, 4)                                            # fold 1, after its step-4 checkpoint
    with pytest.raises(RuntimeError, match="killed"):
        run(registry, tiny, out, "ae-lopo", ae_lopo_cfg())
    lopo = out / "gen/ae/lopo"
    assert not (lopo / DONE_NAME).exists() and (lopo / "fold-0" / ld.FOLD_DONE_NAME).is_file()
    assert load_checkpoint(lopo / "fold-1" / LAST_NAME)["step"] == 4
    fold0 = {n: sha256_file(lopo / "fold-0" / n) for n in FOLD_FILES}
    spy.crash = None
    r = run(registry, tiny, out, "ae-lopo", ae_lopo_cfg(), resume="auto")
    assert r.done["resumed"] and r.done["attempt"] == 2
    assert {n: sha256_file(lopo / "fold-0" / n) for n in FOLD_FILES} == fold0        # kept, not refitted
    assert "fold-0: finished in an earlier attempt; kept" in (lopo / "stage.log").read_text()
    for k in (0, 1):
        assert_same_training(ran.lopo_dir / f"fold-{k}", lopo / f"fold-{k}")
        a = torch.load(ran.lopo_dir / f"fold-{k}" / ld.AE_EXPORT_NAME, weights_only=True)
        b = torch.load(lopo / f"fold-{k}" / ld.AE_EXPORT_NAME, weights_only=True)
        assert_same(a["state_dict"], b["state_dict"]) and a["latent_scale"] == b["latent_scale"]
        recon = ld.RECON_NAME
        assert (ran.lopo_dir / f"fold-{k}" / recon).read_text() == (lopo / f"fold-{k}" / recon).read_text()
        fa, fb = ran.lopo.done["summary"]["folds"][k], r.done["summary"]["folds"][k]
        assert {x: fa[x] for x in FOLD_SUMMARY_KEYS} == {x: fb[x] for x in FOLD_SUMMARY_KEYS}
    assert set(r.done["outputs"]) == set(ran.lopo.done["outputs"])

    spy.crash = (spy.fits + 1, 2)                                 # the final refit, after its step-2 checkpoint
    with pytest.raises(RuntimeError, match="killed"):
        run(registry, tiny, out, "ae-final", ae_final_cfg())
    assert load_checkpoint(out / "gen/ae/final" / LAST_NAME)["step"] == 2
    spy.crash = None
    r = run(registry, tiny, out, "ae-final", ae_final_cfg(), resume="auto")
    assert r.done["resumed"]
    assert_same_training(ran.final_dir, out / "gen/ae/final")
    assert (ran.final_dir / ld.RECON_NAME).read_text() == (out / "gen/ae/final" / ld.RECON_NAME).read_text()
