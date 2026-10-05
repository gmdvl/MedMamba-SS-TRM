# -*- coding: utf-8 -*-
"""
test_g6_test_loader.py
=========================
Regression test for the gate-G6 defect found after the `20260904_230812`
HSI run: `train_example_v16` delegated its test-split loader to the FROZEN
`train_example_v15._build_test_loader`, which builds the FROZEN
`train_example_v6.NpyDataset`, whose `__init__` asserts

    normalization in ("per_sample_minmax", "global_zscore")

That predates `per_patch_zscore` - the mode v16's A-2 made the DEFAULT for
HSI - so the assert fired on every v16 HSI run, was swallowed by the caller's
`except Exception`, and was recorded only as `gates.json: {"G6": false}`.
The run itself completed; only the held-out evaluation was lost.

These tests pin the two halves of the fix:
  1. the frozen dataset really does reject the mode (so the test fails loudly
     if someone "fixes" it by editing the frozen file instead);
  2. `_build_test_loader_v16` accepts all three modes and yields 2-tuples,
     which is what `training.evaluator.evaluate_model` can unpack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training import train_pipeline as v16
from training.dataloader_config_v16 import train_val_policies_v16
from training.normalization import NORMALIZATION_MODES


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for split, n in (("train", 24), ("val", 16), ("test", 20)):
        np.save(tmp_path / f"X_{split}.npy",
                rng.random((n, 5, 5, 4), dtype=np.float32) + 0.1)
        np.save(tmp_path / f"y_{split}.npy", rng.integers(0, 3, size=n).astype(np.int64))
    (tmp_path / "class_names.json").write_text(json.dumps(["a", "b", "c"]))
    return tmp_path


class _Args:
    batch_size = 8
    seed = 0


def _policy():
    return train_val_policies_v16("safe", 0, 2, False, False)[1]


def test_frozen_npydataset_still_rejects_per_patch_zscore(tiny_dataset):
    """The defect's root cause, pinned. If this ever fails, someone edited
    `training/npy_data.py`'s verbatim copy of the frozen class, and
    `test_live_copies_match_archive.py` should be failing too."""
    from training.npy_data import NpyDataset
    with pytest.raises(AssertionError):
        NpyDataset(str(tiny_dataset / "X_test.npy"), str(tiny_dataset / "y_test.npy"),
                   normalization="per_patch_zscore")


@pytest.mark.parametrize("mode", ["per_sample_minmax", "per_patch_zscore", "global_zscore"])
def test_v16_test_loader_accepts_every_normalization_mode(tiny_dataset, mode):
    """The regression itself: G6 must be reachable under every mode v16
    exposes, `per_patch_zscore` (the HSI default) above all."""
    gm = gs = None
    if mode == "global_zscore":
        gm, gs = np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)

    loader = v16._build_test_loader_v16(
        _Args(), data_dir=str(tiny_dataset), storage_mode="mmap", normalization=mode,
        global_mean=gm, global_std=gs, policy=_policy(), device="cpu",
        val_x_path=str(tiny_dataset / "X_val.npy"))

    assert loader is not None, f"G6 loader unavailable for normalization={mode!r}"
    batch = next(iter(loader))
    # `training.evaluator.evaluate_model` does `sample_x, _ = next(iter(loader))`.
    assert len(batch) == 2, f"expected a 2-tuple for evaluate_model, got {len(batch)}"
    x, y = batch
    assert x.shape[1:] == (4, 5, 5) and torch.isfinite(x).all()
    assert y.dtype == torch.int64


def test_every_declared_normalization_mode_is_covered():
    """If a fourth mode is added to `training/normalization.py`, this test
    fails until the parametrization above covers it too."""
    covered = {"per_sample_minmax", "per_patch_zscore", "global_zscore"}
    assert set(NORMALIZATION_MODES) == covered, (
        f"NORMALIZATION_MODES changed to {NORMALIZATION_MODES}; extend the G6 parametrization.")


def test_test_loader_refuses_when_val_is_test(tiny_dataset):
    """v15's guard, preserved: never report the model-selection split as held-out."""
    loader = v16._build_test_loader_v16(
        _Args(), data_dir=str(tiny_dataset), storage_mode="mmap",
        normalization="per_patch_zscore", global_mean=None, global_std=None,
        policy=_policy(), device="cpu", val_x_path=str(tiny_dataset / "X_test.npy"))
    assert loader is None


def test_test_loader_returns_none_without_a_test_split(tmp_path):
    loader = v16._build_test_loader_v16(
        _Args(), data_dir=str(tmp_path), storage_mode="mmap",
        normalization="per_patch_zscore", global_mean=None, global_std=None,
        policy=_policy(), device="cpu", val_x_path=str(tmp_path / "X_val.npy"))
    assert loader is None


# ============================================================================
# The second and third G6 defects, found when the fixed loader let the
# evaluation actually run: a reconstruction-wrapped model returns
# (logits, x_recon), and the frozen evaluation path both softmaxes the result
# directly and reaches for `.cfg` / `.backbone`, which only the inner
# base_model has.
# ============================================================================

def _tiny_recon_model(in_channels=8, num_classes=3):
    from training.config_presets import build_model, v15_config_overrides
    from training.reconstruction_head_v2 import MedMambaSSTRMLatentReconWrapperV2
    ov = dict(v15_config_overrides("recursive"))
    ov["spectral_chunk_size"] = 1024
    tk = dict(trm_dim=32, trm_core_layers=1, trm_n_latent=1, trm_n_improve=1,
              trm_deep_supervision_steps=1, trm_mixer="mlp", trm_ema_rate=0.999,
              trm_act_halting=False, trm_checkpoint_core=False, trm_mixer_channel_mlp=True)
    base = build_model("recursive", "hsi", num_classes, trm_kwargs=tk, cfg_overrides=ov)
    return base, MedMambaSSTRMLatentReconWrapperV2(base, in_channels=in_channels,
                                               out_activation="linear")


def test_recon_wrapper_returns_a_tuple_that_frozen_evaluator_cannot_softmax():
    """Root cause of defect 2, pinned. `training/evaluator.py:33` does
    `F.softmax(out, dim=-1)` on whatever `model(x)` returns."""
    import torch.nn.functional as F
    _base, model = _tiny_recon_model()
    out = model(torch.randn(2, 8, 5, 5))
    assert isinstance(out, tuple) and len(out) == 2
    with pytest.raises(AttributeError):
        F.softmax(out, dim=-1)


def test_logits_only_view_makes_the_wrapper_evaluable():
    from training.trainerg_v12 import _LogitsOnlyView
    _base, model = _tiny_recon_model()
    view = _LogitsOnlyView(model)
    out = view(torch.randn(2, 8, 5, 5))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 3)


def test_logits_only_view_preserves_checkpoint_keys():
    """Registering `inner` as a submodule would prefix every key with
    `inner.`, breaking `load_state_dict` against a saved checkpoint."""
    from training.trainerg_v12 import _LogitsOnlyView
    _base, model = _tiny_recon_model()
    view = _LogitsOnlyView(model)
    assert set(view.state_dict()) == set(model.state_dict())
    view.load_state_dict(model.state_dict())          # must not raise


def test_logits_only_view_delegates_cfg_and_backbone():
    """Defect 3: `evaluate_model` reads `.cfg`, `collect_band_weights` reads
    `.backbone`; a reconstruction wrapper has neither."""
    from training.trainerg_v12 import _LogitsOnlyView
    base, model = _tiny_recon_model()
    assert not hasattr(model, "backbone"), "wrapper unexpectedly grew a .backbone"
    view = _LogitsOnlyView(model)
    assert view.cfg is base.cfg
    assert view.backbone is base.backbone
    with pytest.raises(AttributeError):
        view.definitely_not_an_attribute


def test_band_selection_is_on_for_the_hsi_preset():
    """If this ever goes False, the `.backbone` delegation above stops being
    exercised by a real run and the test below becomes vacuous."""
    base, _model = _tiny_recon_model()
    assert getattr(base.cfg, "dynamic_band_selection", False) is True


def test_non_reconstruction_model_is_not_wrapped():
    """The PAD run (`--recon_mode none`) already passed G6 and its numbers are
    published; the fix must not touch that path. `TrainerG_v12` applies the
    view only when the model has a `base_model` - the same test
    `_forward_with_recon` uses to detect a wrapper."""
    base, model = _tiny_recon_model()
    assert getattr(base, "base_model", None) is None, "plain model must not look wrapped"
    assert getattr(model, "base_model", None) is base, "wrapper must look wrapped"
