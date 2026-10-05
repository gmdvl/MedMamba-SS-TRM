# -*- coding: utf-8 -*-
"""
test_fast_loop_v17.py
========================
MedMamba-SS-TRM v17, Stage 12 - `training/trainerg_v12_fast.py` and `training/ema.py`.

The whole point of S12 is that it changes nothing observable, so the centrepiece is a
BIT-IDENTICAL parity test against `TrainerG_v12` rather than an `allclose` one. The rest
covers the failure paths, which are the only places the two loops can legitimately differ
and are exactly the places a silent regression would hide:

  * a non-finite LOSS is still skipped, with the same counters (the skip-and-continue
    policy, not an abort - see the module docstring);
  * a non-finite GRADIENT still names `first_bad_parameter`. This is the subtle one: the
    fast loop branches BEFORE the clip because `torch.clamp(nan, max=1.0)` is `nan`, so
    clipping against a non-finite norm poisons every gradient and the diagnostic walk would
    name whatever comes first in iteration order;
  * Phase 18's deferred parameter check still fires, INCLUDING on the last step of an epoch,
    which is the one step whose flag is never read by the next iteration;
  * fp16 delegates to `TrainerG_v12`.

Plus the contract-2 gap: `test_scheduler_warmup.py:66` asserts the step-axis scheduler
survives via `inspect.getsource(TrainerG_v12._train_one_epoch_deep_supervision)` - it names
the BASE class, so it stays green while a subclass that actually runs steps nothing.
`test_every_override_keeps_the_step_axis_scheduler` closes that, and the behavioural test
next to it is worth more than either source check.

CPU only, tiny synthetic model, seconds to run.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from medmamba_ss_trm_ema import EMAHelper
from training.ema import FastEMAHelper
from training.trainerg_v12 import TrainerG_v12
from training.trainerg_v12_fast import TrainerG_v12Fast

C, HW, NUM_CLASSES = 4, 5, 3


def tiny_loader(n=24, batch=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.rand(n, C, HW, HW, generator=g),
                                     torch.arange(n, dtype=torch.long) % NUM_CLASSES),
                       batch_size=batch)


def tiny_model():
    torch.manual_seed(0)
    return MedMambaSSTRM(MedMambaSSTRMConfig(
        recursive=True, dims=(16,), depths=(1,), d_state=4, d_ctx=8, patch_size=1,
        spectral_depth=1, d_token=8, compression_dims=(8,), trm_dim=16, trm_mixer="mlp",
        trm_core_layers=1, trm_n_latent=1, trm_n_improve=1, trm_deep_supervision_steps=2,
        spectral_token_fusion="concat_mlp", spectral_pe_gain=0.1, spectral_value_init_std=0.5,
        spectral_ctx_norm=True, classifier_init="fan_in", wavelength_encoding_scale=None,
        spectral_chunk_size=0), num_classes=NUM_CLASSES)


def make(cls, tmp_path, **kw):
    """A trainer of `cls`, seeded so two of them are step-for-step comparable."""
    torch.manual_seed(0)
    model = kw.pop("model", None) or tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    defaults = dict(
        model=model, train_loader=tiny_loader(seed=0), val_loader=tiny_loader(seed=1),
        optimizer=opt, scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5),
        device="cpu", exp_dir=Path(tmp_path), class_names=[f"c{i}" for i in range(NUM_CLASSES)],
        lambda_mse=0.0, lambda_sam=0.0, lambda_gan=0.0, criterion=nn.CrossEntropyLoss(),
        amp_mode="off", ema_rate=0.0, scheduler_interval="step",
    )
    defaults.update(kw)
    t = cls(**defaults)
    t.current_epoch = 1
    return t


def run_epoch(trainer):
    return trainer._train_one_epoch_deep_supervision(trainer._deep_supervision_base())


# ======================================================================
# the centrepiece - bit-identical, not approximately equal
# ======================================================================

_METRIC_KEYS = ("loss", "cls_loss", "cls_loss_last_segment", "halt_loss", "mse_loss",
                "sam_loss", "gan_loss", "acc", "n_optimizer_steps")


def test_fast_loop_matches_the_base_loop(tmp_path):
    """Same seed, same batches, same order -> the same dict, exactly.

    Device-side float64 accumulation reproduces `x.item() * bs` into a Python float
    operation for operation, so this is `==` and not `pytest.approx`. If it ever has to be
    loosened, the accumulator dtype is the first thing to check.
    """
    base = run_epoch(make(TrainerG_v12, tmp_path / "base"))
    fast = run_epoch(make(TrainerG_v12Fast, tmp_path / "fast"))

    assert set(fast) == set(base), "contract 1: every key of the returned dict survives"
    for k in _METRIC_KEYS:
        assert fast[k] == base[k], f"{k}: {fast[k]!r} != {base[k]!r}"
    assert fast["grad_norm"] == pytest.approx(base["grad_norm"], rel=1e-12)
    assert fast["gradient_health"] == base["gradient_health"]


def test_fast_loop_leaves_the_same_weights(tmp_path):
    """The dict could match while the clip was perturbed - compare the model itself.

    This is what proves `get_total_norm` + `clip_grads_with_norm_` really is
    `clip_grad_norm_` split in half.
    """
    tb, tf = make(TrainerG_v12, tmp_path / "b"), make(TrainerG_v12Fast, tmp_path / "f")
    run_epoch(tb)
    run_epoch(tf)
    for (nb, pb), (nf, pf) in zip(tb.model.named_parameters(), tf.model.named_parameters()):
        assert nb == nf
        assert torch.equal(pb, pf), f"{nb} diverged after one epoch"


# ======================================================================
# contract 2 - the step-axis scheduler (test_scheduler_warmup.py:66's blind spot)
# ======================================================================

def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def test_every_override_keeps_the_step_axis_scheduler():
    """Generalises `test_scheduler_warmup.py:66` from the base class to every subclass that
    REPLACES the loop - the reason the original passes vacuously for `TrainerG_v12Fast`."""
    import training.trainerg_v12_fast          # noqa: F401 - registers the subclass
    seen = 0
    for cls in {TrainerG_v12, *_subclasses(TrainerG_v12)}:
        if "_train_one_epoch_deep_supervision" not in cls.__dict__:
            continue
        seen += 1
        src = inspect.getsource(cls.__dict__["_train_one_epoch_deep_supervision"])
        assert 'self.scheduler_interval == "step"' in src, cls.__name__
        assert "self.scheduler.step()" in src, cls.__name__
    assert seen >= 2, "expected at least TrainerG_v12 and TrainerG_v12Fast"


class _CountingScheduler:
    """Minimal scheduler stand-in: the trainer only ever calls `.step()` on it here."""

    def __init__(self):
        self.n_steps = 0

    def step(self):
        self.n_steps += 1

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


def test_fast_loop_steps_the_scheduler_once_per_optimizer_step(tmp_path):
    """Behavioural, which the source assertion never was."""
    sched = _CountingScheduler()
    out = run_epoch(make(TrainerG_v12Fast, tmp_path, scheduler=sched, scheduler_interval="step"))
    assert sched.n_steps == out["n_optimizer_steps"] == 3


def test_scheduler_interval_epoch_does_not_step_per_batch(tmp_path):
    sched = _CountingScheduler()
    run_epoch(make(TrainerG_v12Fast, tmp_path, scheduler=sched, scheduler_interval="epoch"))
    assert sched.n_steps == 0


# ======================================================================
# failure paths
# ======================================================================

class _NanLossOnBatch(nn.Module):
    """CrossEntropyLoss that returns NaN on the `when`-th call (1-based).

    Deep supervision calls the criterion once per segment, so `when` counts criterion
    calls, not batches; with `trm_deep_supervision_steps=2` batch 2 is calls 3 and 4.
    """

    def __init__(self, when):
        super().__init__()
        self.inner, self.when, self.n = nn.CrossEntropyLoss(), when, 0

    def forward(self, logits, y):
        self.n += 1
        out = self.inner(logits, y)
        return out * float("nan") if self.n == self.when else out


def test_nonfinite_loss_is_skipped_identically(tmp_path):
    """A NaN loss is SKIPPED, not fatal, and the counters match the base loop exactly.

    This is the policy the fast loop must not quietly trade away: up to
    `max_gradient_skip_ratio` of an epoch may be skipped and the epoch is still valid.
    """
    base = run_epoch(make(TrainerG_v12, tmp_path / "b", criterion=_NanLossOnBatch(3)))
    fast = run_epoch(make(TrainerG_v12Fast, tmp_path / "f", criterion=_NanLossOnBatch(3)))

    assert base["n_optimizer_steps"] == fast["n_optimizer_steps"] == 2, "the bad batch is skipped"
    assert fast["gradient_health"] == base["gradient_health"]
    assert base["gradient_health"]["nonfinite_loss"] == 1


def test_nonfinite_gradient_still_names_the_parameter(tmp_path):
    """The clip-poisoning risk, in one test.

    A finite loss with a NaN GRADIENT must still report the parameter that carries it. The
    fast loop only gets this right because it branches while the gradients are unclipped:
    `torch.clamp(nan, max=1.0)` is `nan`, so a post-clip walk would find every gradient
    non-finite and name whichever comes first.
    """
    t = make(TrainerG_v12Fast, tmp_path, debug_numerics=True)
    target = "trm.core.z_proj.weight"
    named = dict(t.model.named_parameters())
    if target not in named:                       # keep the test honest if the model changes
        target = next(n for n, p in named.items() if p.requires_grad and p.dim() > 1)
    named[target].register_hook(lambda g: g * float("nan"))

    out = run_epoch(t)
    assert out["n_optimizer_steps"] == 0, "every batch has a NaN gradient"
    gh = out["gradient_health"]
    assert gh["nonfinite_gradients"] == 3 and gh["nonfinite_loss"] == 0
    assert gh["first_bad_parameter"] == target, \
        f"expected {target!r}, got {gh['first_bad_parameter']!r} - the clip ran before the walk"


class _CorruptOnStep(torch.optim.AdamW):
    """AdamW that NaNs the first parameter on its `when`-th step - Phase 18's trigger."""

    def __init__(self, *a, when=1, **kw):
        super().__init__(*a, **kw)
        self.when, self.n = when, 0

    @torch.no_grad()
    def step(self, *a, **kw):
        out = super().step(*a, **kw)
        self.n += 1
        if self.n == self.when:
            self.param_groups[0]["params"][0].mul_(float("nan"))
        return out


def _abort_run(tmp_path, when):
    model = tiny_model()
    opt = _CorruptOnStep(model.parameters(), lr=1e-3, when=when)
    t = make(TrainerG_v12Fast, tmp_path, model=model, optimizer=opt)
    out = run_epoch(t)
    return t, out


def test_nonfinite_parameters_still_abort_the_run(tmp_path):
    t, _ = _abort_run(tmp_path, when=1)
    assert t.is_stopped
    assert str(t.stop_reason) == "TRAINING_ABORTED_NUMERICAL_INSTABILITY"
    assert (Path(tmp_path) / "numerical_failure").is_dir()


def test_phase18_flag_is_drained_at_epoch_end(tmp_path):
    """The deferred read is one step behind, so the LAST step of an epoch has nobody to read
    its flag. Corrupt the parameters on the final step and the run must still abort."""
    t, out = _abort_run(tmp_path, when=3)                 # 3 batches -> the last one
    assert out["n_optimizer_steps"] == 3, "all three steps ran"
    assert t.is_stopped, "the final step's Phase-18 flag was never drained"
    assert str(t.stop_reason) == "TRAINING_ABORTED_NUMERICAL_INSTABILITY"


def test_fp16_delegates_to_the_base_loop(tmp_path, monkeypatch):
    """An enabled GradScaler syncs on `found_inf` anyway, and deferring the loss check past
    `unscale_` would break the unscale/update pairing. So fp16 must take v12's loop."""
    t = make(TrainerG_v12Fast, tmp_path)
    monkeypatch.setattr(t.scaler, "is_enabled", lambda: True)
    called = {}
    real = TrainerG_v12._train_one_epoch_deep_supervision

    def spy(self, base):
        called["base_loop"] = True
        monkeypatch.setattr(self.scaler, "is_enabled", lambda: False)   # let it actually run
        return real(self, base)

    monkeypatch.setattr(TrainerG_v12, "_train_one_epoch_deep_supervision", spy)
    out = run_epoch(t)
    assert called.get("base_loop"), "fp16 did not delegate"
    assert out["n_optimizer_steps"] == 3


def test_capability_probe_falls_back(tmp_path, monkeypatch):
    """No torch.nn.utils.get_total_norm (torch < 2.7) -> v12's loop, not a dead run."""
    import training.trainerg_v12_fast as fastmod
    monkeypatch.setattr(fastmod, "_FAST_OK", False)
    monkeypatch.setattr(fastmod, "_WARNED", False)
    out = run_epoch(make(TrainerG_v12Fast, tmp_path))
    assert out["n_optimizer_steps"] == 3


# ======================================================================
# the EMA swap
# ======================================================================

def test_fast_trainer_swaps_in_the_fused_ema(tmp_path):
    t = make(TrainerG_v12Fast, tmp_path, ema_rate=0.9)
    assert isinstance(t.ema_helper, FastEMAHelper)
    assert t.ema_helper.mu == 0.9
    base = make(TrainerG_v12, tmp_path / "b", ema_rate=0.9)
    assert isinstance(base.ema_helper, EMAHelper) and not isinstance(base.ema_helper, FastEMAHelper)


def test_no_ema_is_left_alone(tmp_path):
    assert make(TrainerG_v12Fast, tmp_path, ema_rate=0.0).ema_helper is None


def test_fast_ema_tracks_the_frozen_helper(tmp_path):
    """`allclose`, not equal - `_foreach_mul_`/`_foreach_add_` reassociate. Measured
    ~1e-8 over 300 updates at mu=0.999; the tolerance here is deliberately looser."""
    def drive(helper):
        torch.manual_seed(1)
        m = tiny_model()
        helper.register(m)
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        torch.manual_seed(2)
        for _ in range(60):
            x = torch.rand(4, C, HW, HW)
            y = torch.arange(4) % NUM_CLASSES
            opt.zero_grad()
            nn.functional.cross_entropy(m(x), y).backward()
            opt.step()
            helper.update(m)
        return helper

    slow, fast = drive(EMAHelper(mu=0.99)), drive(FastEMAHelper(mu=0.99))
    assert list(slow.shadow) == list(fast.shadow)
    for k in slow.shadow:
        assert torch.allclose(slow.shadow[k], fast.shadow[k], rtol=1e-5, atol=1e-7), k


def test_fast_ema_survives_a_state_dict_round_trip():
    """The silent one: the frozen `load_state_dict` REBINDS `self.shadow`, stranding the
    cached alias lists. A resumed run would then average into orphaned tensors and
    checkpoint the loaded shadow unchanged forever - and a smoke test would not notice."""
    m = tiny_model()
    h = FastEMAHelper(mu=0.5)
    h.register(m)
    sd = {k: torch.zeros_like(v) for k, v in h.state_dict().items()}
    h.load_state_dict(sd)
    before = {k: v.clone() for k, v in h.shadow.items()}
    h.update(m)
    assert any(not torch.equal(before[k], h.shadow[k]) for k in before), \
        "update() wrote into orphaned tensors"
    assert all(torch.equal(h.state_dict()[k], h.shadow[k]) for k in before)


def test_fast_ema_falls_back_exactly_when_foreach_raises(monkeypatch):
    """The fallback must be EXACT, not approximate - it is the frozen code path."""
    import training.ema as ema_mod
    m = tiny_model()
    slow, fast = EMAHelper(mu=0.9), FastEMAHelper(mu=0.9)
    slow.register(m)
    fast.register(m)
    monkeypatch.setattr(ema_mod, "_FUSED_OK", False)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    slow.update(m)
    fast.update(m)
    for k in slow.shadow:
        assert torch.equal(slow.shadow[k], fast.shadow[k]), k
