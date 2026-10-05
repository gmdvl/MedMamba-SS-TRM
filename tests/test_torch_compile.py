# -*- coding: utf-8 -*-
"""
test_torch_compile.py
=======================
`training/torch_compile.py` (v15 R3.4) - the `--compile` fast path.

The load-bearing test here is `test_deepcopy_rebinds_the_compiled_core`.
`EMAHelper` deep-copies the model every epoch to validate the EMA weights, so
a compiled core that does not survive `copy.deepcopy` with its binding intact
does not crash - it silently validates the LIVE weights instead, and every
metric, checkpoint-selection decision and early-stopping decision in the run
is computed from the wrong model. The first implementation of this module
compiled the BOUND method and had exactly that defect.

Runs on CPU: `torch.compile`'s Inductor CPU backend is enough to exercise the
binding and equivalence contracts, which are what this module can get wrong.
"""

from __future__ import annotations

import copy

import pytest
import torch

import medmamba_ss_trm as G
from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from training.torch_compile import (
    enable_compiled_recursive_core,
    enable_compiled_scan,
    enable_torch_compile,
)


def _model():
    cfg = MedMambaSSTRMConfig(recursive=True, dims=(32,), depths=(1,), trm_dim=32,
                          d_ctx=32, d_state=4, patch_size=1, spectral_depth=1,
                          trm_core_layers=1, trm_n_latent=2, trm_n_improve=2,
                          trm_deep_supervision_steps=2)
    torch.manual_seed(0)
    return MedMambaSSTRM(cfg, num_classes=3).eval()


def _x(b=2, c=8):
    torch.manual_seed(1)
    return torch.randn(b, c, 5, 5)


def test_enable_reports_what_it_touched():
    m = _model()
    assert enable_compiled_scan(m) > 0
    assert enable_compiled_recursive_core(m) == 1


def test_scan_is_only_rebound_once_and_not_over_a_custom_backend():
    """A caller who registered a real CUDA/Triton kernel has already beaten
    Inductor; `enable_compiled_scan` must leave that backend alone."""
    cfg = MedMambaSSTRMConfig(recursive=True, dims=(32,), depths=(1,), trm_dim=32,
                          d_ctx=32, d_state=4, patch_size=1, spectral_depth=1,
                          scan_backend="mytriton")
    sentinel = lambda *a, **k: G._selective_scan_pure_pytorch(*a, **k)
    G.register_scan_backend("mytriton", sentinel)
    try:
        m = MedMambaSSTRM(cfg, num_classes=3)
        assert enable_compiled_scan(m) == 0
        assert all(p.scan_fn is sentinel for p in m.modules()
                   if isinstance(p, G._SelectiveScanParams))
    finally:
        G.SCAN_BACKENDS.pop("mytriton", None)


def test_compiling_does_not_change_the_prediction():
    m = _model()
    x = _x()
    with torch.no_grad():
        before = m(x).clone()
    enable_torch_compile(m)
    with torch.no_grad():
        after = m(x)
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-4)


def test_deepcopy_rebinds_the_compiled_core():
    """The EMA-validation contract: a deep copy must run the COPY's weights.

    Compiling the bound method leaves `__self__` pointing at the original
    core, `copy.deepcopy` cannot see through the resulting object, and the
    copy silently evaluates the live model. Regression test for that.
    """
    m = _model()
    enable_torch_compile(m)
    x = _x()
    with torch.no_grad():
        base = m(x).clone()
        copied = copy.deepcopy(m).eval()
        for p in copied.backbone.core.parameters():
            p.zero_()
        moved = (copied(x) - base).abs().max()
        original_still = (m(x) - base).abs().max()

    assert copied.backbone.core._f_forward.__self__ is copied.backbone.core
    assert moved > 1e-4, "zeroing the COPY's core changed nothing - it is still calling the original"
    assert original_still == 0, "perturbing the copy moved the ORIGINAL model's output"


def test_enable_is_idempotent_and_shares_one_compiled_artifact():
    """Two models (or the same model twice) must not each build their own
    Inductor artifact - that is what made the EMA deepcopy cost 28s/epoch."""
    a, b = _model(), _model()
    enable_torch_compile(a)
    enable_torch_compile(b)
    assert a.backbone.core._f_forward.__func__ is b.backbone.core._f_forward.__func__


def test_failure_is_a_warning_not_a_dead_run(monkeypatch):
    """A torch.compile failure must never cost a multi-hour run."""
    import training.torch_compile as TC
    monkeypatch.setattr(TC, "_COMPILED_SCAN", None)
    monkeypatch.setattr(TC, "_COMPILED_F_FORWARD", None)
    monkeypatch.setattr(TC, "_compile", lambda fn, mode=None: (_ for _ in ()).throw(RuntimeError("boom")))
    m = _model()
    summary = TC.enable_torch_compile(m)
    assert "FAILED" in summary and "boom" in summary
    with torch.no_grad():
        m(_x())          # still runnable
