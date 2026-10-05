# -*- coding: utf-8 -*-
"""tests/test_check_eval_mode.py - helpers of scripts/check_eval_mode.py."""
import numpy as np
import torch

from scripts.check_eval_mode import set_bn_batchstats, stratified_indices, summarize


def test_stratified_indices_balanced_and_sorted():
    y = np.array([0] * 50 + [1] * 5 + [2] * 30)
    idx = stratified_indices(y, 10, seed=0)
    assert np.all(np.diff(idx) > 0)
    assert np.bincount(y[idx]).tolist() == [10, 5, 10]


def test_summarize_flags_collapse():
    y = np.array([0, 1, 2] * 100)
    s = summarize(np.zeros_like(y), y, 3)
    assert s["collapsed"] and abs(s["balanced_accuracy"] - 1 / 3) < 1e-9
    assert not summarize(y.copy(), y, 3)["collapsed"]


def test_bn_batchstats_leaves_running_stats_untouched():
    m = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1), torch.nn.BatchNorm2d(4), torch.nn.Dropout(0.5))
    m.eval()
    set_bn_batchstats(m)
    before = m[1].running_mean.clone()
    m(torch.randn(8, 2, 3, 3) + 5.0)
    assert m[1].training and not m[2].training
    assert torch.equal(before, m[1].running_mean)


def test_split_config_rebuilds_without_classifier_dropout():
    """Regression: the hierarchical run's config carries classifier_dropout 0.1,
    which build_model rejects for split/fullchannel (seen on the host 2026-09-26)."""
    import glob
    import json
    import pytest
    hits = glob.glob("experiments/20260915_083810_*arch-split/config.json")
    if not hits:
        pytest.skip("hierarchical run not present")
    cfg = json.load(open(hits[0]))
    from scripts.eval_test_split import _rebuild_model
    with pytest.raises(ValueError):
        _rebuild_model(cfg)
    cfg["cli_args"]["classifier_dropout"] = None
    m = _rebuild_model(cfg)
    assert sum(p.numel() for p in m.parameters()) > 2_000_000
