# -*- coding: utf-8 -*-
"""tests/test_trainsel_prep_v20.py - prepare_histologyhsi_bc_trainsel: Pass-1
statistics must see training captures only, and held-out captures must be
gain-corrected against the TRAINING corpus median. Runs without `spectral`:
the v8 Pass 1 and the capture-median reader are replaced by recorders."""
import json
import types

import numpy as np
import pytest

import prepare_histologyhsi_bc as v8
import prepare_histologyhsi_bc_trainsel as ts


def _item(key, group):
    return types.SimpleNamespace(key=key, group_id=group, label=0, payload={})


def _write_split(out_dir, groups):
    (out_dir / "split_assignment.json").write_text(json.dumps({"groups": groups}))


def _args(out_dir):
    return types.SimpleNamespace(out_dir=str(out_dir), capture_gain="median_ratio",
                                 gain_clip_lo=0.5, gain_clip_hi=2.0, gain_sample_pixels=400,
                                 band_selection_sample_fraction=0.1, seed=42)


def test_training_groups_requires_split(tmp_path):
    with pytest.raises(RuntimeError):
        ts.training_groups(str(tmp_path))
    _write_split(tmp_path, {"p1": "train", "p2": "validation", "p3": "test"})
    assert ts.training_groups(str(tmp_path)) == {"p1"}


def test_gain_rule_matches_v8():
    g, raw = ts.gain_for(50.0, 100.0, 0.5, 2.0)
    assert raw == 2.0 and g == 2.0
    g, raw = ts.gain_for(10.0, 100.0, 0.5, 2.0)
    assert raw == 10.0 and g == 2.0          # clipped


def test_pass1_sees_training_items_only(tmp_path, monkeypatch):
    _write_split(tmp_path, {"p1": "train", "p2": "train", "p3": "validation", "p4": "test"})
    items = [_item("a", "p1"), _item("b", "p2"), _item("c", "p3"), _item("d", "p4")]
    seen = {}

    def fake_v8_pass1(self, its, args):
        seen["pass1"] = [it.key for it in its]
        return {"selected_band_indices": [0, 1], "capture_gain_corpus_median": 100.0,
                "capture_gain": {"a": {"gain": 1.0, "median_intensity": 100.0},
                                 "b": {"gain": 1.0, "median_intensity": 100.0}},
                "capture_gain_suspects": []}

    def fake_medians(its, args, sel_arr):
        seen["medians"] = [it.key for it in its]
        return {"c": 50.0, "d": 400.0}

    monkeypatch.setattr(v8.HistologyHsiAdapterV8, "pass1", fake_v8_pass1)
    monkeypatch.setattr(v8, "_capture_medians", fake_medians)

    state = ts.HistologyHsiAdapterTrainSel().pass1(items, _args(tmp_path))
    assert seen["pass1"] == ["a", "b"]                   # statistics: training only
    assert seen["medians"] == ["c", "d"]                 # held-out: gain only
    assert state["capture_gain"]["c"]["gain"] == 2.0     # 100 / 50 against the TRAIN median
    assert state["capture_gain"]["d"]["gain"] == 0.5     # 100 / 400 = 0.25 -> clipped
    assert state["capture_gain"]["d"]["reference"] == "training_corpus_median"
    assert "d" in state["capture_gain_suspects"]
    assert state["pass1_scope"]["heldout_patients"] == ["p3", "p4"]


def test_scope_report_written_and_popped(tmp_path, monkeypatch):
    monkeypatch.setattr(v8.HistologyHsiAdapterV8, "write_pass1_outputs",
                        lambda self, state, out_dir: ["x"])
    state = {"pass1_scope": {"n_train_captures": 2}}
    out = ts.HistologyHsiAdapterTrainSel().write_pass1_outputs(state, str(tmp_path))
    assert out == ["x"] and "pass1_scope" not in state
    assert json.loads((tmp_path / ts.SCOPE_REPORT).read_text())["n_train_captures"] == 2
