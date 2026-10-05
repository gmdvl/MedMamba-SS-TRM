# -*- coding: utf-8 -*-
"""
test_flops_report_v18.py
==========================
v18 F2. Locks down the corrected FLOP accounting and, just as importantly,
pins the defect it replaces so nobody "simplifies" the caller back to it.

Run with: pytest tests/test_flops_report_v18.py -q
"""

import pytest
import torch

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from training.flops_counter import count_flops
from training.flops_report import (
    build_flops_report, core_applications, count_scan_flops, unwrap_model,
)

# Small enough to instantiate and forward in well under a second, while keeping
# every structural property under test: a spectral pathway (so there IS a
# selective scan to count) and a recursive core (so there ARE core applications).
_SMALL = dict(recursive=True, dims=(32,), depths=(1,), d_state=8, d_ctx=32,
              patch_size=1, spectral_depth=1, trm_dim=32, trm_core_layers=1,
              trm_n_latent=2, trm_n_improve=2, trm_deep_supervision_steps=2,
              trm_mixer="mlp", trm_act_halting=False)


def _small_model(in_channels_unused=8, num_classes=3):
    return MedMambaSSTRM(MedMambaSSTRMConfig(**_SMALL), task="classification",
                              num_classes=num_classes)


# ----------------------------------------------------------------------------
# The defect, pinned
# ----------------------------------------------------------------------------

def test_three_dim_shape_is_what_broke_every_run():
    """`trainerg_v11.py:717` passes `sample_x.shape[1:]`.

    This is the exact call every archived `test_report.json` recorded as
    `{"error": "ValueError: not enough values to unpack (expected 4, got 3)"}`.
    If this test ever stops raising, the frozen caller has been changed and the
    v18 override is redundant - which is a thing to notice, not to ignore.
    """
    model = _small_model()
    with pytest.raises(Exception):
        count_flops(model, (8, 5, 5), device="cpu")     # 3-D: C, H, W


def test_four_dim_shape_counts_something():
    """The same call with the batch axis restored."""
    model = _small_model()
    out = count_flops(model, (1, 8, 5, 5), device="cpu")
    assert out["conv_linear_macs"] > 0
    assert out["macs_by_layer_type"]["Linear"] > 0


# ----------------------------------------------------------------------------
# The selective-scan term
# ----------------------------------------------------------------------------

def test_scan_flops_are_measured_not_assumed():
    """`_SelectiveScanParams.run` flattens [B,K,D,L] to [B,K*D,L] before
    calling `scan_fn` (medmamba_ss_trm.py:473), so K and D must come from the module
    and not from the tensor. A wrapper that read four axes off the argument
    would raise - that was the first version of this code."""
    model = _small_model()
    scan = count_scan_flops(model, (1, 8, 5, 5), device="cpu")
    assert scan["n_scan_modules"] >= 1, "the spectral pathway has a SpectralMamba"
    assert scan["n_scan_calls"] >= 1, "it must actually be called during the forward"
    assert scan["selective_scan_flops"] > 0
    shape = scan["scan_call_shapes_sample"][0]
    assert len(shape["arrived_as"]) == 3, "the tensor really does arrive 3-D"
    assert shape["K"] * shape["d_inner"] == shape["arrived_as"][1]


def test_scan_wrapper_is_restored_after_counting():
    """The wrapper must not outlive the count, or every later forward pays it."""
    from medmamba_ss_trm import _SelectiveScanParams
    model = _small_model()
    before = {id(m): m.scan_fn for m in model.modules() if isinstance(m, _SelectiveScanParams)}
    count_scan_flops(model, (1, 8, 5, 5), device="cpu")
    after = {id(m): m.scan_fn for m in model.modules() if isinstance(m, _SelectiveScanParams)}
    assert before and before == after


# ----------------------------------------------------------------------------
# core_applications - the number section 7.6 says must accompany a param count
# ----------------------------------------------------------------------------

def test_core_applications_at_the_reported_configuration():
    """(6+1) * 3 * 3 = 63, the value the manuscript reports throughout."""
    assert core_applications({"architecture": "recursive", "trm_n_latent": 6,
                              "trm_n_improve": 3, "trm_deep_supervision_steps": 3}) == 63


@pytest.mark.parametrize("n_improve,expected", [(1, 21), (2, 42), (3, 63), (4, 84)])
def test_core_applications_ladder_matches_the_depth_sweep(n_improve, expected):
    """E2 varies `--trm_n_improve` at fixed deep supervision; these are the four
    rungs section 7.6 already prices in FLOPs."""
    assert core_applications({"architecture": "recursive", "trm_n_latent": 6,
                              "trm_n_improve": n_improve,
                              "trm_deep_supervision_steps": 3}) == expected


def test_core_applications_is_none_for_a_hierarchical_model():
    assert core_applications({"architecture": "split"}) is None


# ----------------------------------------------------------------------------
# The assembled report
# ----------------------------------------------------------------------------

def test_build_flops_report_shape_and_totals():
    model = _small_model()
    report = build_flops_report(
        model, in_channels=8, patch_hw=(5, 5),
        cli_args={"architecture": "recursive", "trm_n_latent": 2, "trm_n_improve": 2,
                  "trm_deep_supervision_steps": 2},
        device="cpu", backbone_num_params=sum(p.numel() for p in model.parameters()))
    assert report["input_shape"] == [1, 8, 5, 5], "the batch axis is the whole fix"
    assert report["total_flops_approx"] == (report["conv_linear_flops_approx"]
                                            + report["selective_scan_flops"])
    assert report["core_applications"] == (2 + 1) * 2 * 2
    assert report["wrapped_num_params"] > 0
    assert report["param_memory_mb"] > 0


def test_unwrap_peels_compile_but_keeps_the_recon_decoder():
    """`_orig_mod` is a wrapper around the same arithmetic; `base_model` is not -
    with `--recon_mode latent` the decoder runs on every forward and its cost is
    part of the model's cost, so it must NOT be peeled for counting."""
    model = _small_model()

    class FakeCompiled(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    assert unwrap_model(FakeCompiled(model)) is model

    class FakeReconWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.base_model = inner

    wrapped = FakeReconWrapper(model)
    assert unwrap_model(wrapped) is wrapped


# ----------------------------------------------------------------------------
# --compile on must not silently gut the count (v18, found 2026-09-19)
# ----------------------------------------------------------------------------

def test_eager_recursive_core_removes_and_restores_the_compiled_binding():
    """`training/torch_compile.py:143` rebinds `core._f_forward` per instance.

    The count has to go through the EAGER class method or Inductor fuses the
    core's Linears and their forward hooks never fire.
    """
    import types
    from medmamba_ss_trm import RecursiveCore
    from training.flops_report import eager_recursive_core

    model = _small_model()
    cores = [m for m in model.modules() if isinstance(m, RecursiveCore)]
    assert cores, "the small config must still have a recursive core"

    sentinel = types.MethodType(lambda self, *a, **k: None, cores[0])
    cores[0]._f_forward = sentinel
    with eager_recursive_core(model) as n:
        assert n == 1
        assert "_f_forward" not in cores[0].__dict__, "the compiled binding must be off during the count"
    assert cores[0].__dict__.get("_f_forward") is sentinel, "it must be put back afterwards"


def test_counting_bypasses_a_compiled_core_binding():
    """Functional proof, without paying for a real Inductor compile.

    A stub binding that raises stands in for the compiled one. If the count
    still succeeds, it went through the eager class method - which is exactly
    what makes the Linear MACs correct. Before this fix the ref20 run recorded
    0.204 GF against a true 6.235 GF, a 30x undercount, because the stub's
    real-world equivalent silently swallowed 126 mixer calls.
    """
    import types
    from medmamba_ss_trm import RecursiveCore

    model = _small_model()
    core = next(m for m in model.modules() if isinstance(m, RecursiveCore))

    def _boom(self, *a, **k):
        raise AssertionError("the compiled binding was used for counting")
    core._f_forward = types.MethodType(_boom, core)

    report = build_flops_report(model, in_channels=8, patch_hw=(5, 5),
                                cli_args={"architecture": "recursive", "trm_n_latent": 2,
                                          "trm_n_improve": 2, "trm_deep_supervision_steps": 2},
                                device="cpu")
    assert report["decompiled_cores_for_counting"] == 1
    assert report["macs_by_layer_type"]["Linear"] > 0
    assert isinstance(core.__dict__.get("_f_forward"), types.MethodType), "binding not restored"
