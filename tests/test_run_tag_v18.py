# -*- coding: utf-8 -*-
"""
test_run_tag_v18.py
=====================
`--run_tag`, and the reason it has to exist - now on `train.py --profile paper_recipe`
(the configuration of `train_example_v18.py`, archived).

`training/run_naming_v16.py:_NAME_ALLOWLIST` puts `loss`, `sampler`,
`normalization`, the two subsample fractions and `seed` into a run's directory
name. It does NOT include `trm_n_improve`, `recon_mode` or `use_wavelengths` -
exactly the three flags the v18 depth (E2), reconstruction (E4) and encoding
(E6) ablations vary. Without a tag those arms differ only by timestamp.

Run with: pytest tests/test_run_tag_v18.py -q
"""

import argparse

import pytest

from training.train_cli import _valid_run_tag, resolve
from training.train_pipeline import build_run_name, build_model_guarded
from training.trainerg import TrainerG

_BASE = ["--profile", "paper_recipe", "--data_dir", "data/hsi_v8-80_10_10_importance-new/hsi",
         "--batch_size", "256", "--epochs", "12"]
_TS = "20260915_000000"      # fixed, so names differ by content and not by clock


def _args(extra):
    return resolve(_BASE + extra)[0].args


@pytest.mark.parametrize("flags", [
    ["--trm_n_improve", "1"],          # E2, the depth sweep
    ["--recon_mode", "none"],          # E4, the reconstruction ablation
    ["--no_use_wavelengths"],          # E6, the encoding ablation
])
def test_untagged_ablation_arms_collide_with_the_baseline(flags):
    """Pinned as a fact about the naming module, not as desired behaviour."""
    assert build_run_name(_args(flags), _TS) == build_run_name(_args([]), _TS)


@pytest.mark.parametrize("flags,tag", [
    (["--trm_n_improve", "1"], "depth-n1"),
    (["--recon_mode", "none"], "recon-off"),
    (["--no_use_wavelengths"], "enc-index"),
])
def test_a_tag_separates_them(flags, tag):
    baseline = build_run_name(_args([]), _TS)
    arm = build_run_name(_args(flags + ["--run_tag", tag]), _TS)
    assert arm != baseline
    assert arm.endswith(f"_{tag}")


def test_two_tags_give_two_names():
    assert build_run_name(_args(["--run_tag", "depth-n1"]), _TS) != \
        build_run_name(_args(["--run_tag", "depth-n2"]), _TS)


def test_untagged_v18_name_ends_like_the_old_entry_point():
    assert build_run_name(_args([]), _TS).endswith("_optimal_recon")


def test_v18_inherits_the_optimal_and_recon_defaults():
    args = _args([])
    assert args.recon_mode == "latent"
    assert args.lambda_mse == 0.1
    assert args.loss == "focal_weighted"
    assert args.architecture == "recursive"
    assert args.checkpoint_metric == "f1_macro"


def test_lambda_sam_is_zero_by_default_and_must_be_passed_for_hsi():
    """Every v18 HSI command passes --lambda_sam 0.1, or the spectral-angle term
    silently disappears from the objective."""
    assert _args([]).lambda_sam == 0.0
    assert _args(["--lambda_sam", "0.1"]).lambda_sam == 0.1


@pytest.mark.parametrize("bad", ["", "has space", "under_score", "-leading", "semi;colon"])
def test_invalid_tags_are_rejected(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _valid_run_tag(bad)


@pytest.mark.parametrize("good", ["abl-base", "depth-n4", "rq0-rgb", "bands16", "s7"])
def test_valid_tags_are_accepted(good):
    assert _valid_run_tag(good) == good


def test_the_trainer_carries_every_former_layer():
    """FLOPs recount + latent fallback (v18), 3-tuple batches + stop filter
    (optimal), patience + G5 alias (original), artifacts (v13)."""
    for method in ("evaluate_test_split", "_generate_rich_diagnostics", "_train_one_epoch",
                   "_trigger_stop", "_check_stopping_rules", "verify_best_checkpoint_reproduces"):
        assert method in TrainerG.__dict__, method
    assert "TrainerG_v13" in [c.__name__ for c in TrainerG.__mro__]


def test_model_builder_drops_an_inapplicable_classifier_dropout(capsys):
    """What lets `--architecture split` run under an optimal-family profile at all."""
    model = build_model_guarded("split", "hsi", 3, classifier_dropout=0.1)
    assert model is not None
    assert "IGNORED" in capsys.readouterr().out
