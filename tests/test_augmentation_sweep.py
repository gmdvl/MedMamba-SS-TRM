# -*- coding: utf-8 -*-
"""
tests/test_augmentation_sweep.py
================================
Issue #33 (Plan §Phase 10): `sweeps/augmentation_pipeline.py` and
`scripts/augmentation_select.py`.

* The selection rule reads, per run, config.json and history.json only: open /
  np.load are patched to fail on any `test_*` file, and every run in the
  fixture HAS test_predictions.npz / test_report.json (train.py always writes
  them). The tie-breaks (DCIS F1, then the cheaper method, then table order),
  the protocol and export checks, and the report reading selection.json
  unchanged.
* The sweep: `seeds == []` without selection.json, and built from it with
  the report's tags; every arm resolves through `train_cli.resolve` to
  250,113 +- 1 rows per epoch and one `trm_ema_rate`, without pinning it;
  the shared tags are the paper and minority sweeps' commands byte for byte;
  the export-aware `done`; the glass rule switch from the recorded
  measurement; A4 from decision.json; one-job bench stages.

The sweep is loaded with `run_experiments.ROOT` pointed at a temporary
directory, so the files it reads at load (glass_rule.json, decision.json,
selection.json) and the runs it looks for are the fixture's. CPU, seconds.

Run with: pytest tests/test_augmentation_sweep.py -q
"""

from __future__ import annotations

import builtins
import io
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import run_experiments as rx
from hsi_augment import report as R
from scripts import augmentation_select as S
from training import train_profiles
from training.train_cli import resolve

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "sweeps/augmentation_pipeline.py"
PAPER = rx.load_sweep(str(ROOT / "sweeps/paper_evidence_upgrade.py"))
MINAUG = rx.load_sweep(str(ROOT / "sweeps/minority_augmentation.py"))
HSI, OUT, SEL = "data/hsi_v9-trainsel/hsi", "data/hsi_v9-trainsel-aug", "reports/augmentation/selection.json"
N_TRAIN, N_DCIS = 2452086, 122850
ENV = ("AUG_OUT", "AUG_SELECTION", "AUG_TAU", "AUG_PHYS")


# ============================================================================
# fixtures: a repo root holding the pipeline's decisions and training runs
# ============================================================================

def _wj(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def load(root: Path, monkeypatch, phys: bool = False):
    """The sweep as `run_experiments.py` would load it in `root`."""
    monkeypatch.setattr(rx, "ROOT", root)
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    if phys:
        monkeypatch.setenv("AUG_PHYS", "1")
    return rx.load_sweep(str(SWEEP))


def write_glass(root: Path, use, complete: bool = True) -> None:
    d = root / OUT / "prep/tissue"
    _wj(d / "glass_rule.json", {"use_copy_mask": use, "complete": complete, "threshold": 0.1,
                                "overall": {"glass_fraction": 0.2 if use else 0.05}})
    _wj(d / "done.json", {"status": "ok"})
    np.save(d / "dcis_copy_mask.npy", np.ones(4, dtype=bool))


def write_export(root: Path, src: str, eid: str) -> None:
    _wj(root / OUT / "export" / src / "provenance.json", {"export_id": eid, "source": src})


def write_decision(root: Path, src: str, key: str = "a4_source") -> None:
    _wj(root / OUT / "lopo_utility/decision.json", {key: src})


def write_phys(root: Path) -> None:
    _wj(root / OUT / "prep/augcal/phys_calibration.json", {"tilt_max": 0.01})
    (root / OUT / "prep/spectral").mkdir(parents=True, exist_ok=True)
    np.save(root / OUT / "prep/spectral/noise_sigma.npy", np.ones(4, dtype=np.float32))


def _history(score: float, dcis: float, epochs: int = 12):
    return [{"epoch": e, "f1_macro": score if e >= 10 else 0.5 + 0.01 * e, "balanced_accuracy": 0.8,
             "per_class_f1": [0.9, dcis if e >= 10 else 0.2, 0.99], "epoch_time": 300.0}
            for e in range(1, epochs + 1)]


def write_run(root: Path, name: str, tag: str, *, score: float = 0.8, dcis: float = 0.5, seed: int = 42,
              epochs: int = 12, kept: int = 250113, synthetic=None, cli=None, finished: bool = True) -> Path:
    """A run directory in the trainer's formats, test files included."""
    d = root / "experiments" / name
    cfg = {"cli_args": {"run_tag": tag, "seed": seed, "epochs": epochs, "data_dir": HSI, **(cli or {})},
           "class_names": ["healthy", "DCIS", "IDC"],
           "train_subsample": {"frac": 0.0971337, "kept": kept, "total": 2574936}}
    if synthetic:
        cfg["synthetic"] = synthetic
    _wj(d / "config.json", cfg)
    _wj(d / "history.json", _history(score, dcis, epochs))
    if finished:
        np.savez(d / "test_predictions.npz", true_label=np.zeros(3), predicted_label=np.ones(3))
        _wj(d / "test_report.json", {"f1_macro": 1.0})
    return d


# screen scores: A3 best generative; A1 best non-generative
SCREEN = {"A0": (0.80, 0.40), "A0os": (0.81, 0.45), "A1": (0.83, 0.50), "A2": (0.82, 0.46),
          "A3": (0.85, 0.52), "A4": (0.835, 0.55)}
CLI = {"A0os": {"minority_augment": "on", "minority_oversample_factor": 2.0, "minority_mix_prob": 0.0},
       "A1": {"minority_augment": "on", "minority_oversample_factor": 2.0, "minority_mix_prob": 0.5},
       "A4": {"minority_augment": "on", "minority_oversample_factor": 1.0, "minority_mix_prob": 0.5}}
SOURCE = {"A2": "cvae", "A3": "ldm", "A4": "ldm"}
EXPORT = {"cvae": "cvae00000001", "ldm": "ldm000000002"}


def make_screen(root: Path, arms=R.DEFAULT_ARMS, scores=None) -> None:
    """A finished screen of `arms` (plan tags), the exports and decision.json."""
    for src, eid in EXPORT.items():
        write_export(root, src, eid)
    write_decision(root, "ldm")
    for i, a in enumerate(arms):
        if a.key not in SCREEN:
            continue
        score, dcis = (scores or SCREEN)[a.key]
        syn = {"source": SOURCE[a.key], "export_id": EXPORT[SOURCE[a.key]]} if a.key in SOURCE else None
        write_run(root, f"20261101_0000{i:02d}_{a.screen_tag}", a.screen_tag, score=score, dcis=dcis,
                  synthetic=syn, cli=CLI.get(a.key))


def select_main(root: Path, *extra) -> int:
    arms = _wj(root / "arms.json", [asdict(a) for a in R.DEFAULT_ARMS if a.key != "A1p"])
    return S.main(["--experiments", str(root / "experiments"), "--run_data", HSI, "--aug_out", str(root / OUT),
                   "--arms_file", str(arms), "--out", str(root / SEL), *extra])


def _guard_test_reads(monkeypatch, root: Path, opened: list) -> None:
    """open / io.open / np.load fail on any test file under `root`."""
    real_open, real_load = builtins.open, np.load

    def under(f) -> bool:
        return isinstance(f, (str, os.PathLike)) and str(os.path.abspath(f)).startswith(str(root))

    def g_open(file, *a, **k):
        if under(file):
            if R._is_test_path(Path(file)):
                raise AssertionError(f"test file opened: {file}")
            opened.append(Path(os.path.abspath(file)))
        return real_open(file, *a, **k)

    def g_load(file, *a, **k):
        if under(file) and R._is_test_path(Path(file)):
            raise AssertionError(f"test file loaded: {file}")
        return real_load(file, *a, **k)

    monkeypatch.setattr(builtins, "open", g_open)
    monkeypatch.setattr(io, "open", g_open)
    monkeypatch.setattr(np, "load", g_load)


def _train_jobs(sweep):
    return [j for jobs in sweep["STAGES"].values() for j in jobs if j.kind == "train"]


def _by_tag(sweep):
    return {j.resolved_argv()[-1]: j for j in _train_jobs(sweep)}


# ============================================================================
# scripts/augmentation_select.py
# ============================================================================

def test_selection_reads_only_config_and_history(tmp_path, monkeypatch, capsys):
    make_screen(tmp_path)
    opened: list = []
    _guard_test_reads(monkeypatch, tmp_path, opened)
    assert select_main(tmp_path) == 0, capsys.readouterr().err
    monkeypatch.undo()

    doc = json.loads((tmp_path / SEL).read_text())
    assert doc["selected"] == ["A3", "A1"]
    assert doc["per_family"] == {"generative": "A3", "non-generative": "A1"}
    assert doc["decisions"]["generative"]["decided_by"] == "f1_macro"
    assert doc["test_files_read"] == [] and doc["a4_source"] == "ldm"
    run_files = {p.name for p in opened if "experiments" in p.parts}
    assert run_files == {"config.json", "history.json"}, run_files
    assert not any(R._is_test_path(p) for p in opened)
    assert all(k.endswith(("/config.json", "/history.json", "/provenance.json", "/decision.json"))
               for k in doc["inputs_sha256"]), sorted(doc["inputs_sha256"])
    assert "A0" not in doc["scores"]                                     # the control is never a candidate
    # the scores are the report's own: mean f1_macro over epochs 10-12, per_class_f1[1]
    assert doc["scores"]["A3"]["score"] == pytest.approx(0.85, abs=1e-12)
    assert doc["scores"]["A3"]["per_epoch_f1_macro"] == [0.85, 0.85, 0.85]
    assert doc["scores"]["A4"]["dcis_f1"] == pytest.approx(0.55)


def test_report_reads_the_selection_unchanged(tmp_path):
    make_screen(tmp_path)
    assert select_main(tmp_path) == 0
    inp = R.Inputs({"sel": tmp_path / "reports"}, reveal_test=False)
    selection, arms = R._load_selection(inp, tmp_path / SEL, R.DEFAULT_ARMS)
    assert selection["selected"] == ["A3", "A1"]
    assert [a.key for a in arms] == ["A0", "A0os", "A1", "A2", "A3", "A4"]     # the table it was made with
    for k in selection["selected"]:
        assert selection["document"]["scores"][k]["score"] == pytest.approx(SCREEN[k][0], abs=1e-12)


def _row(arm, score, dcis, cost=(0, 0), order=0):
    return {"arm": arm, "score": score, "dcis_f1": dcis, "cost": list(cost), "order": order}


def test_rule_tie_breaks():
    # a clear winner
    assert S.choose([_row("A2", 0.82, 0.9), _row("A3", 0.85, 0.1)])[0] == "A3"
    # within 0.005 -> DCIS F1 decides
    w, trace = S.choose([_row("A2", 0.851, 0.40), _row("A3", 0.854, 0.30)])
    assert (w, trace["decided_by"]) == ("A2", "dcis_f1")
    assert trace["steps"][0]["tied"] == ["A2", "A3"]
    # a gap of 0.006 is not a tie
    assert S.choose([_row("A2", 0.848, 0.90), _row("A3", 0.854, 0.30)])[0] == "A3"
    # tied on both -> the cheaper method (CVAE before AE + LDM; fewer on-the-fly ops)
    w, trace = S.choose([_row("A3", 0.850, 0.500, (2, 0), 4), _row("A2", 0.849, 0.498, (1, 0), 3)])
    assert (w, trace["decided_by"]) == ("A2", "cost")
    assert S.choose([_row("A1", 0.83, 0.5, (0, 2), 2), _row("A0os", 0.83, 0.5, (0, 1), 1)])[0] == "A0os"
    # tied on everything -> the arm table's order
    w, trace = S.choose([_row("A1", 0.83, 0.5, (0, 1), 2), _row("A0os", 0.83, 0.5, (0, 1), 1)])
    assert (w, trace["decided_by"]) == ("A0os", "arm_order")


def test_method_cost_from_resolved_flags():
    cost = lambda cli, syn=None: S.method_cost({"cli_args": cli, "synthetic": syn})   # noqa: E731
    assert cost({}) == (0, 0)
    assert cost(CLI["A0os"]) == (0, 1) and cost(CLI["A1"]) == (0, 2)
    assert cost({**CLI["A1"], "phys_aug": "minority"}) == (0, 3)
    assert cost({}, {"source": "cvae"}) == (1, 0) and cost({}, {"source": "ldm"}) == (2, 0)
    assert cost(CLI["A4"], {"source": "cvae"}) == (1, 1)                 # factor 1.0: mixing, no copies


def test_selection_refuses_a_nonconforming_screen(tmp_path, capsys):
    make_screen(tmp_path)
    write_export(tmp_path, "ldm", "regenerated0")                         # A3 / A4 trained on an old export
    write_run(tmp_path, "20261102_000000_minaug-nomix", "minaug-nomix", epochs=20, cli=CLI["A0os"])
    for d in (tmp_path / "experiments").glob("*_aug-cvae"):
        (d / "history.json").unlink()                                   # A2 has no screen run any more
    assert select_main(tmp_path) == 1
    err = capsys.readouterr().err
    assert "A3 (aug-ldm" in err and "A4 (aug-combo" in err and "regenerated0" in err
    assert "A0os (minaug-nomix" in err and "20 epochs" in err
    assert "A2 (aug-cvae): no screen run" in err
    assert not (tmp_path / SEL).exists()


def test_selection_is_not_replaced_without_force(tmp_path, capsys):
    make_screen(tmp_path)
    assert select_main(tmp_path) == 0
    first = (tmp_path / SEL).read_text()
    assert select_main(tmp_path) == 1 and "--force" in capsys.readouterr().err
    assert (tmp_path / SEL).read_text() == first
    assert select_main(tmp_path, "--dry_run") == 0 and '"selected"' in capsys.readouterr().out
    assert select_main(tmp_path, "--force") == 0


# ============================================================================
# sweeps/augmentation_pipeline.py
# ============================================================================

def test_seeds_is_empty_without_selection(tmp_path, monkeypatch):
    sweep = load(tmp_path, monkeypatch)
    assert sweep["STAGES"]["seeds"] == []
    assert not (tmp_path / SEL).exists()


def test_seeds_from_selection_use_the_reports_tags(tmp_path, monkeypatch):
    _wj(tmp_path / SEL, {"selected": ["aug-ldm", {"arm": "A1"}]})          # a tag and a dict, as the report allows
    sweep = load(tmp_path, monkeypatch)
    seeds = sweep["STAGES"]["seeds"]
    tags = [j.resolved_argv()[-1] for j in seeds]
    assert tags == ([f"v20-head-s{s}" for s in (1, 7, 13, 23, 42)] + [f"aug-ldm-s{s}" for s in (1, 7, 13, 23, 42)]
                    + [f"minaug-default-s{s}" for s in (1, 7, 13, 23, 42)])
    report_tags = {a.key: a for a in R.DEFAULT_ARMS}
    for k in ("A0", "A3", "A1"):
        assert report_tags[k].tag_for_seed(7) in tags
    assert all("--epochs 20" in " ".join(j.resolved_argv()) for j in seeds)
    # the shared seed runs are the other sweeps' commands, byte for byte
    paper = {j.resolved_argv()[-1]: j.resolved_argv() for jobs in PAPER["STAGES"].values() for j in jobs
             if j.kind == "train" and j.resolved_argv()[3] == HSI}
    minaug = {j.resolved_argv()[-1]: j.resolved_argv() for j in MINAUG["STAGES"]["seeds"]}
    for j in seeds:
        tag = j.resolved_argv()[-1]
        if tag.startswith("v20-head"):
            assert j.resolved_argv() == paper[tag]
        if tag.startswith("minaug-default"):
            assert j.resolved_argv() == minaug[tag]


def test_a_selection_from_another_arm_table_is_refused(tmp_path, monkeypatch):
    write_glass(tmp_path, True)                                          # A1 is now aug-copymix
    _wj(tmp_path / SEL, {"selected": ["A1"], "arms": [{"key": "A1", "screen_tag": "minaug-default"}]})
    with pytest.raises(SystemExit, match="aug-copymix"):
        load(tmp_path, monkeypatch)
    _wj(tmp_path / SEL, {"selected": ["A1p"]})
    with pytest.raises(SystemExit, match="AUG_PHYS"):
        load(tmp_path, monkeypatch)


def _variants(tmp_path, monkeypatch):
    """The sweep before any decision, and with every decision taken: tissue-only
    copies, A4 on the LDM, the optional arm, a selection of A4 and A1p."""
    yield "nothing measured", load(tmp_path / "a", monkeypatch)
    b = tmp_path / "b"
    write_glass(b, True)
    write_decision(b, "ldm")
    _wj(b / SEL, {"selected": ["A4", "A1p"]})
    yield "every decision", load(b, monkeypatch, phys=True)


def test_every_arm_keeps_250113_rows_and_one_ema_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(train_profiles, "split_size",
                        lambda d, s: {"train": N_TRAIN, "val": 334516}.get(s) if d else None)
    for name, sweep in _variants(tmp_path, monkeypatch):
        jobs = _train_jobs(sweep)
        assert len(jobs) == (6 + 6 if name == "nothing measured" else 7 + 7 + 15), name
        emas = set()
        for j in jobs:
            argv = j.resolved_argv()
            assert "--trm_ema_rate" not in argv and "--minority_classes" not in argv, j.name
            a = resolve(argv[2:])[0].args
            n_total = N_TRAIN
            if a.minority_augment == "on":
                n_total += round(N_DCIS * (a.minority_oversample_factor - 1))
            if a.synthetic_dir:
                n_total += N_DCIS
            kept = round(n_total * a.train_subsample_frac)
            assert abs(kept - 250113) <= 1, (name, j.name, n_total, a.train_subsample_frac)
            assert n_total in (N_TRAIN, N_TRAIN + N_DCIS), j.name
            assert (n_total == N_TRAIN) == (a.run_tag in ("v20-abl-base-s42", "aug-bench-a0")
                                            or a.run_tag.startswith("v20-head-s")), j.name
            assert a.minority_classes == "auto" and a.minority_class_weights == "effective"
            emas.add(a.trm_ema_rate)
        assert emas == {0.9995}, (name, emas)


def test_arms_match_the_plan(tmp_path, monkeypatch):
    jobs = _by_tag(load(tmp_path, monkeypatch))
    args = {t: resolve(j.resolved_argv()[2:])[0].args for t, j in jobs.items()}
    assert (args["minaug-nomix"].minority_oversample_factor, args["minaug-nomix"].minority_mix_prob) == (2.0, 0.0)
    assert (args["minaug-default"].minority_oversample_factor, args["minaug-default"].minority_mix_prob) == (2.0, 0.5)
    assert args["aug-cvae"].synthetic_dir == f"{OUT}/export/cvae" and args["aug-cvae"].minority_augment == "off"
    assert args["aug-ldm"].synthetic_dir == f"{OUT}/export/ldm" and args["aug-ldm"].minority_augment == "off"
    a4 = args["aug-combo"]
    assert (a4.minority_augment, a4.minority_oversample_factor, a4.minority_mix_prob) == ("on", 1.0, 0.5)
    for t, a in args.items():
        assert (a.profile, a.batch_size, a.lambda_sam, a.val_subsample_frac) == ("paper_recipe", 256, 0.1, 0.0986)
        assert a.data_dir == HSI and a.phys_aug == "off" and a.minority_copy_mask is None, t
        assert (a.seed, a.epochs) == (42, 1 if t.startswith("aug-bench-") else 12), t


def test_shared_screen_tags_are_the_other_sweeps_commands(tmp_path, monkeypatch):
    for use in (None, False):                                            # before the rule, and copy-all
        root = tmp_path / str(use)
        if use is not None:
            write_glass(root, use)
        jobs = _by_tag(load(root, monkeypatch))
        minaug = {j.resolved_argv()[-1]: j.resolved_argv() for j in MINAUG["STAGES"]["screen"]}
        paper = {j.resolved_argv()[-1]: j.resolved_argv() for j in PAPER["STAGES"]["controls"]}
        for tag in ("minaug-nomix", "minaug-default"):
            assert jobs[tag].resolved_argv() == minaug[tag], tag
        assert jobs["v20-abl-base-s42"].resolved_argv() == paper["v20-abl-base-s42"] == minaug["v20-abl-base-s42"]


def test_the_minority_sweep_targets_trainsel():
    assert MINAUG["HSI"] == HSI
    assert (MINAUG["N_TRAIN"], MINAUG["N_DCIS"]) == (N_TRAIN, N_DCIS)


def test_glass_rule_switches_the_copy_arms(tmp_path, monkeypatch):
    # not measured: the shared tags, waiting for the rule
    jobs = _by_tag(load(tmp_path / "none", monkeypatch))
    assert "no glass rule yet" in jobs["minaug-nomix"].requires()
    assert jobs["v20-abl-base-s42"].requires() is None

    # measured, copy every row (<= 10%): the shared tags, no mask
    write_glass(tmp_path / "all", False)
    jobs = _by_tag(load(tmp_path / "all", monkeypatch))
    assert jobs["minaug-default"].requires() is None and "--minority_copy_mask" not in jobs["minaug-default"].argv

    # measured, tissue-only copies (> 10%): new tags with the stage's copy mask
    root = tmp_path / "tissue"
    write_glass(root, True)
    sweep = load(root, monkeypatch)
    jobs = _by_tag(sweep)
    assert {"minaug-nomix", "minaug-default"}.isdisjoint(jobs)
    for tag, mix in (("aug-copy", 0.0), ("aug-copymix", 0.5)):
        a = resolve(jobs[tag].resolved_argv()[2:])[0].args                 # the flag exists in train.py
        assert a.minority_copy_mask == f"{OUT}/prep/tissue/dcis_copy_mask.npy" and a.minority_mix_prob == mix
        assert jobs[tag].requires() is None
    table = {a["key"]: a for a in sweep["ARM_TABLE"]}
    assert (table["A0os"]["screen_tag"], table["A1"]["seed_tag"]) == ("aug-copy", "aug-copymix-s{seed}")
    assert all("tissue rows only" in table[k]["description"] for k in ("A0os", "A1"))
    assert "--minority_copy_mask" in jobs["aug-bench-a1"].argv and "--minority_copy_mask" not in jobs["aug-combo"].argv

    # the rule changed after the sweep loaded: wait for a reload, never run the stale command
    write_glass(root, False)
    assert "changed since the sweep loaded" in jobs["aug-copy"].requires()

    # a smoke run over part of the captures is not the measurement
    write_glass(tmp_path / "part", True, complete=False)
    jobs = _by_tag(load(tmp_path / "part", monkeypatch))
    assert "part of the train captures" in jobs["minaug-nomix"].requires()


def test_a4_trains_on_the_generator_decision_json_names(tmp_path, monkeypatch):
    jobs = _by_tag(load(tmp_path / "none", monkeypatch))
    assert "no generator chosen" in jobs["aug-combo"].requires()
    for key in R.DECISION_KEYS:
        root = tmp_path / key
        write_decision(root, "cvae", key=key)
        write_export(root, "cvae", "cvae00000001")
        jobs = _by_tag(load(root, monkeypatch))
        a = resolve(jobs["aug-combo"].resolved_argv()[2:])[0].args
        assert a.synthetic_dir == f"{OUT}/export/cvae", key
        assert jobs["aug-combo"].requires() is None
    write_decision(root, "ldm")
    assert "changed since the sweep loaded" in jobs["aug-combo"].requires()


def test_synthetic_arms_are_done_only_on_the_current_export(tmp_path, monkeypatch):
    jobs = _by_tag(load(tmp_path, monkeypatch))
    job = jobs["aug-cvae"]
    assert not job.done() and "export/cvae/provenance.json" in job.requires()       # no export yet

    write_export(tmp_path, "cvae", "cvae00000001")
    write_run(tmp_path, "20261101_000000_aug-cvae", "aug-cvae",
              synthetic={"source": "cvae", "export_id": "cvae00000001"})
    assert job.done() and job.requires() is None

    write_export(tmp_path, "cvae", "regenerated0")                       # the export was regenerated
    assert not job.done()
    write_run(tmp_path, "20261102_000000_aug-cvae", "aug-cvae",
              synthetic={"source": "cvae", "export_id": "regenerated0"}, finished=False)
    assert not job.done()                                                # unfinished runs do not count
    write_run(tmp_path, "20261103_000000_aug-cvae", "aug-cvae",
              synthetic={"source": "cvae", "export_id": "regenerated0"})
    assert job.done()

    (tmp_path / OUT / "export/cvae/provenance.json").unlink()            # a missing export: not done, waits
    assert not job.done() and "run the export stage" in job.requires()
    assert jobs["aug-bench-a2"].done() is False                          # bench and seeds runs: the same check

    # a real-only arm keeps the runner's done: a finished run with the tag
    write_run(tmp_path, "20261101_000001_v20-abl-base-s42", "v20-abl-base-s42", kept=250113)
    assert jobs["v20-abl-base-s42"].done()


def test_select_waits_for_the_screen_and_ignores_a0(tmp_path, monkeypatch):
    write_glass(tmp_path, False)
    write_decision(tmp_path, "ldm")
    sweep = load(tmp_path, monkeypatch)
    select = sweep["STAGES"]["select"][0]
    assert not select.done() and "minaug-nomix" in select.requires()
    make_screen(tmp_path)
    for d in (tmp_path / "experiments").glob("*_v20-abl-base-s42"):
        (d / "test_predictions.npz").unlink()                            # A0 unfinished: select does not need it
    assert select.requires() is None
    argv = select.resolved_argv()
    assert argv[1] == "scripts/augmentation_select.py" and argv[argv.index("--out") + 1] == SEL


def test_select_with_the_sweeps_arm_table_feeds_the_seeds_stage(tmp_path, monkeypatch):
    """Tissue-only copies: select reads the sweep's ARM_TABLE (aug-copy /
    aug-copymix), writes it into selection.json, and the reloaded sweep
    builds the seeds stage with those tags."""
    write_glass(tmp_path, True)
    sweep = load(tmp_path, monkeypatch)
    arms = R.arms_from_doc(sweep["ARM_TABLE"], "sweep")
    make_screen(tmp_path, arms=arms, scores={**SCREEN, "A0os": (0.84, 0.5)})
    assert S.main(["--experiments", str(tmp_path / "experiments"), "--run_data", HSI,
                   "--aug_out", str(tmp_path / OUT), "--sweep", str(SWEEP), "--out", str(tmp_path / SEL)]) == 0
    doc = json.loads((tmp_path / SEL).read_text())
    assert doc["selected"] == ["A3", "A0os"]
    assert {a["key"]: a["screen_tag"] for a in doc["arms"]}["A0os"] == "aug-copy"
    seeds = [j.resolved_argv()[-1] for j in load(tmp_path, monkeypatch)["STAGES"]["seeds"]]
    assert seeds[5:] == [f"aug-ldm-s{s}" for s in (1, 7, 13, 23, 42)] + [f"aug-copy-s{s}" for s in (1, 7, 13, 23, 42)]


def test_optional_arm_a1p(tmp_path, monkeypatch):
    assert "A1p" not in load(tmp_path / "off", monkeypatch)["ARMS"]
    root = tmp_path / "on"
    write_glass(root, False)
    sweep = load(root, monkeypatch, phys=True)
    job = _by_tag(sweep)["aug-conv-phys"]
    a = resolve(job.resolved_argv()[2:])[0].args
    assert (a.phys_aug, a.phys_tilt_max, a.phys_artifacts) == ("minority", "auto", OUT)
    assert (a.minority_augment, a.minority_oversample_factor, a.minority_mix_prob) == ("on", 2.0, 0.5)
    assert "phys_calibration.json" in job.requires()
    write_phys(root)
    assert job.requires() is None
    assert "bench-a1p" in sweep["GROUPS"]["bench"]


def test_stage_order_and_solo_bench(tmp_path, monkeypatch):
    sweep = load(tmp_path, monkeypatch, phys=True)
    order = list(sweep["STAGES"])
    bench = sweep["GROUPS"]["bench"]
    assert order == ["prep", "gen-cvae", "gen-ldm", "export", "lopo-utility", *bench, "screen", "select",
                     "seeds", "report"]
    assert sweep["GROUPS"]["all"] == order
    tags = {a["key"]: a["bench_tag"] for a in sweep["ARM_TABLE"]}
    assert tags == {a.key: a.bench_tag for a in R.DEFAULT_ARMS}           # the report's bench tags
    for stage in bench:
        (job,) = sweep["STAGES"][stage]                                  # alone in its stage: solo under --jobs N
        argv = job.resolved_argv()
        assert argv[argv.index("--epochs") + 1] == "1" and argv[-1] in tags.values()
    assert sweep["ARM_TABLE"] == [{k: v for k, v in asdict(a).items()} for a in R.DEFAULT_ARMS]


def test_pipeline_jobs_follow_the_stage_registry(tmp_path, monkeypatch):
    sweep = load(tmp_path, monkeypatch)
    validate = sweep["STAGES"]["prep"][0]
    assert validate.resolved_argv()[1:] == ["augment_pipeline.py", "validate", "--data_dir", HSI, "--out", OUT]
    assert not validate.done() and validate.requires() is None
    _wj(tmp_path / OUT / "prep/validate/done.json", {"status": "ok"})
    assert validate.done()
    planned = sweep["pipeline"]("x", "no-such-command")
    assert not planned.done() and "no-such-command" in planned.requires()
    pool = sweep["STAGES"]["prep"][2]
    assert "AUG_TAU" in pool.requires() and "--expect_tau_deg" not in pool.argv
    monkeypatch.setenv("AUG_TAU", "4.25")
    monkeypatch.setattr(rx, "ROOT", tmp_path)
    pool = rx.load_sweep(str(SWEEP))["STAGES"]["prep"][2]
    assert pool.argv[-2:] == ["--expect_tau_deg", "4.25"] and pool.requires() is None
