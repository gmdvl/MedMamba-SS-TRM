# -*- coding: utf-8 -*-
"""tests/test_analysis_v20.py - tag-based run discovery and the headline
aggregation of scripts/analysis_v20.py on a synthetic experiments/ tree."""
import json

import numpy as np

import scripts.analysis_v20 as an


def _fake_run(exp, name, tag, data_dir, y, g_ok=True, arch="recursive", val=None):
    rd = exp / name
    rd.mkdir(parents=True)
    (rd / "config.json").write_text(json.dumps({"architecture": arch, "backbone_num_params": 446409,
                                                "cli_args": {"run_tag": tag, "data_dir": data_dir}}))
    rng = np.random.default_rng(len(name))
    p = rng.dirichlet(np.ones(3), size=len(y))
    if g_ok:
        p[np.arange(len(y)), y] += 1.0
        p /= p.sum(1, keepdims=True)
    np.savez(rd / "test_predictions.npz", probs=p, true_label=y)
    if val is not None:
        (rd / "heldout_v19/validation").mkdir(parents=True)
        pv = rng.dirichlet(np.ones(3), size=len(val))
        np.savez(rd / "heldout_v19/validation/test_predictions.npz", probs=pv, true_label=val)
    return rd


def test_headline_pairs_and_deltas(tmp_path, monkeypatch):
    exp = tmp_path / "experiments"
    y = np.array([0, 1, 2] * 20)
    yv = np.array([0, 1, 2] * 10)
    g = np.array(["a", "b"] * 30)
    gv = np.array(["c"] * 30)
    for i, s in enumerate(an.SEEDS[:3]):
        _fake_run(exp, f"2026_{i}h", f"v20-head-s{s}", "data/hsi_v9-trainsel/hsi", y, True, val=yv)
        _fake_run(exp, f"2026_{i}r", f"v20-head-s{s}", "data/hsi_v9-trainsel/rgb", y, False, val=yv)
    monkeypatch.setattr(an, "EXP", exp)
    monkeypatch.setattr(an, "_INDEX", None)
    monkeypatch.setattr(an, "MISSING", [])
    per_seed, summ, boot, pp = an.headline(y, yv, g, gv)
    assert set(per_seed) == {"1", "7", "13"}
    assert summ["test_delta"]["bal_pos"] == 3 and summ["test_delta"]["n"] == 3
    assert "test_hsi" in summ and summ["test_hsi"]["n"] == 3
    assert set(pp) == {"a", "b", "c"}
    # seeds 23 and 42 are reported missing, per modality
    assert sum(1 for m in an.MISSING if m.get("tag") in ("v20-head-s23", "v20-head-s42")) == 4


def test_find_run_filters_by_arch(tmp_path, monkeypatch):
    exp = tmp_path / "experiments"
    y = np.array([0, 1, 2])
    _fake_run(exp, "r1", "v20-s42", "data/hsi_v9-trainsel/hsi", y, arch="hybridsn")
    _fake_run(exp, "r2", "v20-s42", "data/hsi_v9-trainsel/hsi", y, arch="spectralformer")
    monkeypatch.setattr(an, "EXP", exp)
    monkeypatch.setattr(an, "_INDEX", None)
    assert an.find_run("v20-s42", "hsi_v9-trainsel/hsi", arch="spectralformer").name == "r2"
