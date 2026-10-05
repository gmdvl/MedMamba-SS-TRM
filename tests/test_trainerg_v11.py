# -*- coding: utf-8 -*-
"""
test_trainerg_v11.py
======================
MedMamba-SS-TRM v15 plan - `training/trainerg_v11.py`'s items, on a tiny synthetic
dataset so the whole file runs in seconds on CPU.

  R0.3  class collapse is a stop condition, not a warning string
  R2.3  wavelengths reach the model
  R5.1  halt_loss is logged
  R5.2  the generalization gap compares like with like
  R5.3  the checkpoint matches the report (gate G5)
  R5.4  --amp is honoured in validation
  R5.5  test-split evaluation (gate G6)
  R5.6  artifact retention
  R6.3  the augmentation RNG is per (seed, epoch, index, worker)
  R6.4  seeding

Also asserts contract C-2: `TrainerG_v10` is untouched, so
`train_example_v14.py` keeps the trainer its runs were produced with.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from training.failure_taxonomy import FailureClass, classify_stop_reason, classify_exception
from training.trainerg_v10 import TrainerG_v10
from training.trainerg_v11 import TrainerG_v11, COLLAPSE_STOP_REASON

C, HW, NUM_CLASSES = 4, 5, 3


def tiny_loader(n=24, batch=8, seed=0, single_class=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, C, HW, HW, generator=g)
    y = (torch.zeros(n, dtype=torch.long) if single_class
         else torch.arange(n, dtype=torch.long) % NUM_CLASSES)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def tiny_model(**over):
    cfg_kwargs = dict(recursive=True, dims=(16,), depths=(1,), d_state=4, d_ctx=8,
                      patch_size=1, spectral_depth=1, d_token=8, compression_dims=(8,),
                      trm_dim=16, trm_mixer="mlp", trm_core_layers=1, trm_n_latent=1,
                      trm_n_improve=1, trm_deep_supervision_steps=2,
                      spectral_token_fusion="concat_mlp", spectral_pe_gain=0.1,
                      spectral_value_init_std=0.5, spectral_ctx_norm=True,
                      classifier_init="fan_in", wavelength_encoding_scale=None,
                      spectral_chunk_size=0)
    cfg_kwargs.update(over)
    torch.manual_seed(0)
    return MedMambaSSTRM(MedMambaSSTRMConfig(**cfg_kwargs), num_classes=NUM_CLASSES)


def make_trainer(tmp_path, model=None, train=None, val=None, **kwargs):
    model = model if model is not None else tiny_model()
    train = train if train is not None else tiny_loader(seed=0)
    val = val if val is not None else tiny_loader(seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    defaults = dict(model=model, train_loader=train, val_loader=val, optimizer=opt,
                     scheduler=sched, device="cpu", exp_dir=tmp_path,
                     class_names=[f"c{i}" for i in range(NUM_CLASSES)],
                     lambda_mse=0.0, lambda_sam=0.0, lambda_gan=0.0,
                     criterion=nn.CrossEntropyLoss(), amp_mode="off", ema_rate=0.0)
    defaults.update(kwargs)
    return TrainerG_v11(**defaults)


# ======================================================================
# C-2 - TrainerG_v10 is untouched
# ======================================================================

def test_v11_is_a_subclass_and_v10_keeps_its_own_fit():
    assert issubclass(TrainerG_v11, TrainerG_v10)
    assert TrainerG_v11.fit is not TrainerG_v10.fit
    assert TrainerG_v10.fit.__qualname__.startswith("TrainerG_v5")


def test_v10_has_none_of_the_v15_knobs():
    import inspect
    params = inspect.signature(TrainerG_v10.__init__).parameters
    for knob in ("on_class_collapse", "use_wavelengths", "keep_last_n", "eval_artifact_stride"):
        assert knob not in params


# ======================================================================
# R0.3 - class collapse is a stop condition
# ======================================================================

def test_a_single_class_prediction_history_aborts(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="abort", class_collapse_streak=3)
    metrics = {"per_class_recall": [1.0, 0.0, 0.0]}
    for expected_streak in (1, 2):
        t._current_val_preds = np.zeros(50, dtype=np.int64)
        t._check_class_collapse(metrics)
        assert t._single_class_streak == expected_streak
        assert not t.is_stopped
    t._current_val_preds = np.zeros(50, dtype=np.int64)
    t._check_class_collapse(metrics)
    assert t.is_stopped
    assert str(t.stop_reason) == COLLAPSE_STOP_REASON


def test_warn_mode_never_stops(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="warn", class_collapse_streak=2)
    for _ in range(6):
        t._current_val_preds = np.zeros(50, dtype=np.int64)
        msgs = t._check_class_collapse({"per_class_recall": [1.0, 0.0, 0.0]})
    assert not t.is_stopped
    assert any("REPRESENTATION_COLLAPSE" in m for m in msgs)


def test_the_streak_resets_when_a_second_class_appears(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="abort", class_collapse_streak=3)
    m = {"per_class_recall": [0.5, 0.5, 0.0]}
    for preds, expected in ((np.zeros(20, np.int64), 1),
                             (np.zeros(20, np.int64), 2),
                             (np.arange(20) % 3, 0),
                             (np.zeros(20, np.int64), 1)):
        t._current_val_preds = preds
        t._check_class_collapse(m)
        assert t._single_class_streak == expected
    assert not t.is_stopped


def test_the_failure_artifact_is_written_once_and_is_diagnosable(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="abort", class_collapse_streak=1)
    t.current_epoch = 4
    t._current_val_preds = np.full(100, 2, dtype=np.int64)
    t._check_class_collapse({"per_class_recall": [0.0, 0.0, 1.0]})
    payload = json.loads((tmp_path / "collapse_failure" / "failure.json").read_text())
    assert payload["failure_class"] == FailureClass.REPRESENTATION_COLLAPSE.value
    assert payload["epoch"] == 4
    assert payload["distinct_predicted_classes"] == 1
    assert payload["predicted_label_histogram"] == [0, 0, 100]
    assert "test_representation_sensitivity" in payload["guidance"]


def test_the_stop_reason_maps_to_the_new_failure_class():
    assert classify_stop_reason(COLLAPSE_STOP_REASON) is FailureClass.REPRESENTATION_COLLAPSE
    assert classify_exception(SystemExit("REPRESENTATION_COLLAPSE: constant function")) \
        is FailureClass.REPRESENTATION_COLLAPSE


def test_an_invalid_on_class_collapse_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_trainer(tmp_path, on_class_collapse="maybe")


# ======================================================================
# R2.3 - wavelengths reach the model
# ======================================================================

def test_wavelengths_are_forwarded_when_available(tmp_path):
    wl = np.linspace(400.0, 900.0, C).astype(np.float32)
    t = make_trainer(tmp_path, wavelengths=wl, use_wavelengths=True)
    assert t._wl is not None and t._wl.shape == (C,)

    seen = {}
    base = t.model
    orig = base.forward_deep_supervision

    def spy(x, wavelengths=None, sensor_range=None):
        seen["wavelengths"] = wavelengths
        return orig(x, wavelengths=wavelengths, sensor_range=sensor_range)

    base.forward_deep_supervision = spy
    t._deep_supervision_forward(base, torch.rand(2, C, HW, HW))
    assert seen["wavelengths"] is not None


def test_wavelengths_are_not_forwarded_when_disabled(tmp_path):
    wl = np.linspace(400.0, 900.0, C).astype(np.float32)
    assert make_trainer(tmp_path, wavelengths=wl, use_wavelengths=False)._wl is None
    assert make_trainer(tmp_path, wavelengths=None, use_wavelengths=True)._wl is None


def test_a_model_that_does_not_accept_wavelengths_is_called_plainly(tmp_path):
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(C * HW * HW, NUM_CLASSES)

        def forward(self, x):
            return self.fc(x.flatten(1))

    t = make_trainer(tmp_path, model=Plain(), wavelengths=np.arange(C, dtype=np.float32))
    assert t._model_forward(t.model, torch.rand(2, C, HW, HW)).shape == (2, NUM_CLASSES)


# ======================================================================
# R5.4 - AMP in validation
# ======================================================================

@pytest.mark.parametrize("mode", ["off", "bf16", "fp16", "auto"])
def test_validation_amp_matches_training_amp(tmp_path, mode):
    t = make_trainer(tmp_path, amp_mode=mode)
    assert t._resolve_val_amp() == t._resolve_amp()


def test_amp_off_really_means_off_in_validation(tmp_path):
    enabled, _ = make_trainer(tmp_path, amp_mode="off")._resolve_val_amp()
    assert enabled is False


# ======================================================================
# The full loop: R5.1, R5.2, R5.3, R5.6
# ======================================================================

@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    t = make_trainer(d, keep_last_n=2, eval_artifact_stride=1, on_class_collapse="warn")
    t.fit(max_epochs=3)
    return t, d


def test_halt_loss_is_in_the_history(completed_run):
    t, d = completed_run
    assert all("halt_loss" in row for row in t.history)
    assert "halt_loss" in json.loads((d / "history.json").read_text())[0]
    assert "halt_loss" in (d / "history.csv").read_text().splitlines()[0]


def test_optimizer_steps_are_reported_alongside_epochs(completed_run):
    t, _ = completed_run
    assert [r["optimizer_steps_this_epoch"] for r in t.history] == [3, 3, 3]
    assert [r["optimizer_steps_total"] for r in t.history] == [3, 6, 9]


def test_the_generalization_gap_uses_the_matched_losses(completed_run):
    t, _ = completed_run
    for row in t.history:
        assert row["generalization_gap_loss_is_comparable"] is True
        assert row["generalization_gap_loss"] == pytest.approx(
            row["val_cls_loss"] - row["train_cls_loss_last_segment"])
        # the old, uninterpretable number is kept but renamed
        assert row["generalization_gap_loss_raw"] == pytest.approx(
            row["val_loss"] - row["train_loss"])


def test_last_segment_ce_differs_from_the_segment_mean(completed_run):
    t, _ = completed_run
    assert any(abs(r["classification_loss"] - r["train_cls_loss_last_segment"]) > 1e-9
               for r in t.history)


def test_history_is_not_embedded_in_the_checkpoint(completed_run):
    t, d = completed_run
    state = torch.load(d / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert "history" not in state
    assert state["spectral_token_fusion"] == "concat_mlp"


def test_only_keep_last_n_epoch_checkpoints_survive(completed_run):
    t, d = completed_run
    epochs = sorted(p.name for p in (d / "checkpoints").glob("epoch_*.pt"))
    assert epochs == ["epoch_0002.pt", "epoch_0003.pt"]
    assert (d / "checkpoints" / "latest.pt").is_file()
    assert (d / "best_model.pt").is_file()


def test_gate_g5_the_checkpoint_reproduces_the_report(completed_run):
    t, d = completed_run
    result = t.verify_best_checkpoint_reproduces()
    assert result["passed"] is True, result
    assert result["delta"] == pytest.approx(0.0, abs=1e-6)
    assert json.loads((d / "checkpoint_reproducibility.json").read_text())["passed"] is True


def test_ema_best_model_is_the_ema_weights(tmp_path):
    """R5.3's core claim. With EMA on, `TrainerG_v10` validated the EMA copy
    and saved the LIVE weights to best_model.pt, so the file did not reproduce
    the number it was named for."""
    t = make_trainer(tmp_path, ema_rate=0.9, on_class_collapse="warn")
    t.fit(max_epochs=2)
    assert t.ema_helper is not None
    best = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=False)
    live = torch.load(tmp_path / "best_model_live.pt", map_location="cpu", weights_only=False)
    assert best["weights"] == "ema" and live["weights"] == "live"
    key = "head.fc.weight"
    assert not torch.allclose(best["model_state"][key], live["model_state"][key])
    # `best_model.pt` holds the EMA weights AS OF THE BEST EPOCH, which is what
    # `best_model_ema.pt`'s shadow records - not the shadow at the end of the run.
    shadow = torch.load(tmp_path / "best_model_ema.pt", map_location="cpu",
                        weights_only=False)["ema_shadow"]
    assert torch.allclose(best["model_state"][key], shadow[key])
    assert t.verify_best_checkpoint_reproduces()["passed"] is True


def test_load_checkpoint_returns_the_epoch(tmp_path):
    """`TrainerG_v10.load_checkpoint` discards `TrainerG_v4`'s return value, so
    it returns None and `start_epoch = trainer.load_checkpoint(...) + 1` raises
    TypeError - `--resume` has never worked under train_example_v14.py."""
    t = make_trainer(tmp_path, on_class_collapse="warn")
    t.fit(max_epochs=2)
    fresh = make_trainer(tmp_path / "fresh", on_class_collapse="warn")
    assert fresh.load_checkpoint(tmp_path / "checkpoints" / "latest.pt") == 2
    assert TrainerG_v10.load_checkpoint(fresh, tmp_path / "checkpoints" / "latest.pt") is None


def test_eval_artifact_stride_skips_epochs(tmp_path):
    t = make_trainer(tmp_path, eval_artifact_stride=3, on_class_collapse="warn")
    t.fit(max_epochs=4)
    written = sorted(p.name for p in tmp_path.glob("classification_report_epoch*.txt"))
    # epoch 1 always, then every third, plus any best epoch
    assert "classification_report_epoch01.txt" in written
    assert len(written) < 4


def test_stride_one_writes_every_epoch(completed_run):
    t, d = completed_run
    assert len(sorted(d.glob("classification_report_epoch*.txt"))) == 3


# ======================================================================
# R5.5 - the test split (gate G6)
# ======================================================================

def test_gate_g6_writes_a_test_report(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="warn")
    t.fit(max_epochs=1)
    report = t.evaluate_test_split(tiny_loader(seed=2), num_classes=NUM_CLASSES)
    assert report is not None
    on_disk = json.loads((tmp_path / "test_report.json").read_text())
    for key in ("sklearn_metrics", "per_class_extended", "calibration", "efficiency",
                 "predicted_label_histogram", "checkpoint_weights", "flops"):
        assert key in on_disk
    assert on_disk["n_test_samples"] == 24
    assert sum(on_disk["predicted_label_histogram"]) == 24
    assert (tmp_path / "test_predictions.npz").is_file()


def test_test_evaluation_degrades_without_a_loader(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="warn")
    t.fit(max_epochs=1)
    assert t.evaluate_test_split(None) is None


def test_test_evaluation_degrades_without_a_checkpoint(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="warn")
    assert t.evaluate_test_split(tiny_loader(seed=2), checkpoint_path=tmp_path / "nope.pt") is None


# ======================================================================
# C-4 - a checkpoint is tied to its tokenizer
# ======================================================================

def test_loading_a_checkpoint_from_a_different_tokenizer_raises(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="warn")
    t.fit(max_epochs=1)
    other = make_trainer(tmp_path / "other",
                          model=tiny_model(spectral_token_fusion="add", spectral_pe_gain=1.0,
                                           spectral_value_init_std=0.02),
                          on_class_collapse="warn")
    with pytest.raises(ValueError, match="spectral_token_fusion"):
        other.load_checkpoint(tmp_path / "checkpoints" / "latest.pt")


def test_loading_a_matching_checkpoint_works(tmp_path):
    t = make_trainer(tmp_path, on_class_collapse="warn")
    t.fit(max_epochs=1)
    again = make_trainer(tmp_path / "again", on_class_collapse="warn")
    assert again.load_checkpoint(tmp_path / "checkpoints" / "latest.pt") == 1


# ======================================================================
# R6.3 / R6.4 - RNG
# ======================================================================

def test_augmentation_draws_differ_per_worker_and_per_epoch():
    from training.augmentation import AugmentationConfig, AugmentedPatchDataset, item_augmentation_seed

    seeds = {item_augmentation_seed(42, 0, i, w) for i in range(4) for w in range(4)}
    assert len(seeds) == 16, "index/worker collisions would reproduce the pre-v15 duplication"
    assert item_augmentation_seed(42, 0, 5, 0) != item_augmentation_seed(42, 1, 5, 0)
    assert item_augmentation_seed(42, 0, 5, 0) != item_augmentation_seed(43, 0, 5, 0)
    # the old scheme collided: worker 1 of seed 42 == worker 0 of seed 43
    assert (42 + 1) == (43 + 0)

    cfg = AugmentationConfig(flip_h_prob=0.5, flip_v_prob=0.5, rotate90_prob=0.5, seed=42)
    base = TensorDataset(torch.arange(8 * C * HW * HW, dtype=torch.float32)
                          .reshape(8, C, HW, HW), torch.zeros(8, dtype=torch.long))
    ds = AugmentedPatchDataset(base, cfg.build(), seed=42)
    first = ds[0][0].clone()
    assert not torch.equal(first, ds[1][0])
    ds.set_epoch(1)
    assert not torch.equal(first, ds[0][0])
    ds.set_epoch(0)
    assert torch.equal(first, ds[0][0])


def test_seed_everything_seeds_all_four_rngs():
    import random
    from training.dataloader_config import seed_everything

    state = seed_everything(1234)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(1234)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b
    assert state["python_random"] and state["numpy"] and state["torch"]


def test_worker_seeds_depend_on_the_run_seed():
    from training.dataloader_config import resolve_loader_policy
    policy = resolve_loader_policy("performance")
    a = policy.dataloader_kwargs("cpu", seed=42)["worker_init_fn"]
    b = policy.dataloader_kwargs("cpu", seed=43)["worker_init_fn"]

    def draw(fn, worker_id):
        fn(worker_id)
        return float(torch.rand(1))

    assert draw(a, 0) != draw(b, 0), "--seed must change the worker RNG stream"
    assert draw(a, 0) != draw(a, 1), "workers must not share a stream"
    # the pre-v15 collision: base_seed + worker_id made these identical
    assert draw(a, 1) != draw(b, 0)
