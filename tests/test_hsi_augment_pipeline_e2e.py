# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_pipeline_e2e.py
======================================
The pipeline end to end (#39, Phase 8.1): `augment_pipeline.py` on the tiny
fixture, driven through `hsi_augment.cli.main` exactly as the command line
does, with every stage's tiny settings in ONE `--config` file.

What "`augment_pipeline.py all` on the fixture" means here
----------------------------------------------------------
`all` is the CPU prep chain only: discovery, spectral-audit, leakage,
tissue-mask, index, aug-calibration, pool and qc-reference. The generator fits,
generation, QC, the exports and the LOPO utility are kept out of `all` on
purpose (`in_all=False`: on the real build they are multi-hour GPU runs). So
the end-to-end run is `all` followed by the explicit commands, in dependency
order:

    all -> fit-cvae -> fit-ae -> fit-ldm -> generate -> qc -> export -> lopo-utility

each as `<command> --config pipeline.json`. Together they run every
registered stage except `demo` (a smoke test) and `report-tissue` (Phase 10's
report, which reads val/test); a stage added later fails
`test_the_commands_run_every_stage_once_then_skip` until it is added here.

Checked, on ONE run of the chain shared by every test (a module fixture):
  * each command runs its stages once, and a second identical invocation
    skips every one of them ("already done") without touching a done.json;
  * a fit killed mid-way (ldm-lopo, fold 1, after its step-4 checkpoint;
    `StepSpy` from tests/hsi_augment_ldm_fixtures.py) leaves no done.json, and
    `fit-ldm --resume auto` continues it to the same weights, optimizer, RNG,
    history, latents and samples as an uninterrupted run (`ldm-lopo --force`
    in a copy of <out>, both folds fitted from step 0);
  * `--force` re-runs a done stage; with the same config its fingerprint does
    not change, so nothing downstream is stale. A changed config (the pool's
    `rows_per_class`) changes the fingerprint: the stages that require the
    pool are stale, the ones that do not stay done, and re-running the next
    command (`fit-cvae`) carries the staleness one step further down;
  * both exports (export/cvae, export/ldm) pass
    `training.synthetic_mix.load_synthetic_export(..., manifest_check="strict")`;
  * lopo_utility/decision.json is valid and names `a4_source`;
  * a one-epoch `train.py --synthetic_dir export/<a4_source>` run (tiny model,
    CPU) succeeds, and its config.json["synthetic"]["export_id"] is the
    export's provenance.json export_id;
  * the split-access log: only the stages that declare a val/test purpose
    opened val or test.

Why the QC bitmask is overridden (as in tests/test_hsi_augment_export.py):
generators trained for 6 steps make flat patches that the calibrated rules
reject, so generate / qc / export / lopo-utility run under that module's
`oracle_qc`: the REAL `qc.apply_rules` computes every score, and only its
accept bitmask is replaced, by a deterministic function of each patch's bytes.
That needs the pipeline in this process (a monkeypatch), so the commands go
through `cli.main([...])`, not a subprocess; `train.py` does run as one.

The fixture's validation patient has no healthy capture, so the train.py run
passes `--allow_missing_classes`.

CPU only; every model and array is tiny. The chain itself runs in about
6 s, the whole module (with the train.py run) in about 20 s.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pytest
import torch

from hsi_augment import cli
from hsi_augment import latent_diffusion as ld
from hsi_augment import lopo_utility as L
from hsi_augment.io import ACCESS_LOG_NAME, read_access_log
from hsi_augment.models.diffusion import DiffusionLoss
from hsi_augment.pool_io import pool_fingerprint
from hsi_augment.registry import load_builtin_stages
from hsi_augment.splits import held_out_readers
from hsi_augment.stage import DONE_NAME, STARTED_NAME, StageResult, read_done
from hsi_augment.train_generative import LAST_NAME, load_checkpoint
from tests.hsi_augment_fixtures import make_tiny_build
from tests.hsi_augment_ldm_fixtures import FOLD_SUMMARY_KEYS, StepSpy, assert_same, assert_same_training
from tests.test_hsi_augment_export import ACCEPT, AE_LOPO, CVAE_LOPO, GEN, LDM_LOPO, LOOP, oracle_qc
from tests.test_synthetic_mix import TINY, _train_py
from training.synthetic_mix import PROVENANCE_NAME, SYNTHETIC_GROUP_IDS, load_synthetic_export

DCIS = 1
SOURCES = ("cvae", "ldm")
PREP = {"discovery", "spectral-audit", "leakage", "tissue-mask", "index", "aug-calibration", "pool", "qc-reference"}
COMMANDS = ("all", "fit-cvae", "fit-ae", "fit-ldm", "generate", "qc", "export", "lopo-utility")
ORACLE = {"generate", "qc", "export", "lopo-utility"}           # run under oracle_qc(ACCEPT)
NOT_RUN = {"demo", "report-tissue"}
KILL_AT = (2, 4)                     # ldm-lopo: the 2nd fit (fold 1), at training step 4 (after its checkpoint)

# The one --config file: every stage's tiny settings (the per-stage tests' values).
STAGE_SETTINGS = {
    "pool": {"rows_per_class": 30},
    "qc-reference": {"device": "cpu"},
    "cvae-lopo": {**LOOP, **CVAE_LOPO},
    "cvae-final": {"enforce_gates": False, "device": "cpu", "prior_samples": 16},
    "ae-lopo": {**LOOP, **AE_LOPO},
    "ae-final": {"device": "cpu", "metric_rows": 20},
    "ldm-lopo": {**LOOP, **LDM_LOPO},
    "ldm-final": {"device": "cpu"},
    **{f"generate-{s}": dict(GEN) for s in SOURCES},
    **{f"qc-{s}": {"device": "cpu", "chunk_rows": 32, "bootstrap": 5} for s in SOURCES},
    **{f"export-{s}": {"figure_samples": 6} for s in SOURCES},
    "lopo-utility": {"device": "cpu", "qc_chunk_rows": 32, "chunk_rows": 64},
}
STATUS_RE = re.compile(r"\s(done|stale|failed|incomplete|missing|blocked)(?: - (.*))?$")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_config(path: Path, data_dir: Path, out: Path, **pool) -> Path:
    stages = {k: dict(v) for k, v in STAGE_SETTINGS.items()}
    stages["pool"].update(pool)
    path.write_text(json.dumps({"data_dir": str(data_dir), "out": str(out), "stages": stages}, indent=2))
    return path


def invoke(argv: List[str]) -> Dict[str, StageResult]:
    """`cli.main(argv)`, which must exit 0; the StageResult of every stage it ran or skipped."""
    seen: Dict[str, StageResult] = {}
    real = cli.run_stage

    def spy(spec, config, options, registry, **kw):
        r = real(spec, config, options, registry, **kw)
        seen[r.name] = r
        return r

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "run_stage", spy)
        assert cli.main(argv) == 0, argv
    return seen


def done_files(out: Path) -> Dict[str, bytes]:
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob(DONE_NAME))}


def listing(cfg: Path, out: Optional[Path] = None) -> Dict[str, Tuple[str, str]]:
    """`augment_pipeline.py --list --config cfg [--out out]` -> {stage: (status, reason)}."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["--list", "--config", str(cfg), *(["--out", str(out)] if out else [])]) == 0
    rows = {}
    for line in buf.getvalue().splitlines()[1:]:
        m = STATUS_RE.search(line)
        if m and not line.startswith("planned"):
            rows[line.split()[0]] = (m.group(1), m.group(2) or "")
    return rows


def descendants(registry, name: str) -> Set[str]:
    below: Set[str] = set()
    frontier = {name}
    while frontier:
        frontier = {s.name for s in registry.specs() if set(s.requires) & frontier} - below
        below |= frontier
    return below


# ============================================================================
# the run, once per module
# ============================================================================

@dataclass
class Pipeline:
    tiny: object
    out: Path
    cfg: Path
    # command -> one {stage: StageResult} per completed invocation, in order
    runs: Dict[str, List[Dict[str, StageResult]]] = field(default_factory=dict)
    # command -> (done.json files before, after) the repeated invocation
    repeat: Dict[str, Tuple[Dict[str, bytes], Dict[str, bytes]]] = field(default_factory=dict)
    killed: Dict[str, object] = field(default_factory=dict)
    decision: Dict[str, object] = field(default_factory=dict)

    def run(self, command: str, *flags: str) -> Dict[str, StageResult]:
        r = invoke([command, "--config", str(self.cfg), *flags])
        self.runs.setdefault(command, []).append(r)
        return r


@pytest.fixture(scope="module")
def registry():
    return load_builtin_stages()


@pytest.fixture(scope="module")
def pipe(tmp_path_factory) -> Pipeline:
    tiny = make_tiny_build(tmp_path_factory.mktemp("tiny"))
    root = tmp_path_factory.mktemp("pipeline")
    out = root / "aug"
    p = Pipeline(tiny=tiny, out=out, cfg=write_config(root / "pipeline.json", tiny.data_dir, out))
    for command in COMMANDS:
        with oracle_qc(ACCEPT) if command in ORACLE else contextlib.nullcontext():
            if command == "fit-ldm":
                _kill_then_resume(p)
            else:
                p.run(command)
            before = done_files(out)
            p.run(command)                                   # the identical second invocation
            p.repeat[command] = (before, done_files(out))
    p.decision = json.loads((out / "lopo_utility" / L.DECISION_NAME).read_text())
    return p


def _kill_then_resume(p: Pipeline) -> None:
    """fit-ldm killed inside ldm-lopo's second fold, then `fit-ldm --resume auto`."""
    lopo, final = p.out / "gen/ldm/lopo", p.out / "gen/ldm/final"
    with pytest.MonkeyPatch.context() as mp:
        spy = StepSpy(mp, DiffusionLoss)
        spy.crash = KILL_AT
        with pytest.raises(RuntimeError, match="killed mid-fit") as e:
            cli.main(["fit-ldm", "--config", str(p.cfg)])
    p.killed = {"error": str(e.value), "lopo_done": (lopo / DONE_NAME).exists(),
                "lopo_started": (lopo / STARTED_NAME).is_file(),
                "fold0_done": (lopo / "fold-0" / ld.FOLD_DONE_NAME).is_file(),
                "fold1_step": load_checkpoint(lopo / "fold-1" / LAST_NAME)["step"],
                "final_started": final.exists()}
    p.run("fit-ldm", "--resume", "auto")


@pytest.fixture
def copy(pipe, tmp_path) -> Path:
    """A private copy of the pipeline's <out>, for runs that change it."""
    dst = tmp_path / "aug"
    shutil.copytree(pipe.out, dst)
    return dst


# ============================================================================
# the chain
# ============================================================================

def test_all_is_the_cpu_prep_chain(registry):
    assert {s.name for s in registry.resolve(["all"])} == PREP
    out_of_all = {s.name for s in registry.specs() if not s.in_all}
    stages = {s.name for c in COMMANDS[1:] for s in registry.resolve([c])}
    assert stages - PREP == out_of_all - NOT_RUN


def test_the_commands_run_every_stage_once_then_skip(pipe, registry):
    ran: Dict[str, StageResult] = {}
    for command in COMMANDS:
        assert len(pipe.runs[command]) == 2, command
        first, second = pipe.runs[command]
        assert set(first) == set(second) == {s.name for s in registry.resolve([command])}, command
        skipped = {n for n, r in first.items() if r.status == "skipped"}
        assert skipped == ({"qc-reference"} if command == "qc" else set()), command    # done by `all`
        for n, r in first.items():
            if r.status == "ran":
                assert n not in ran, n
                ran[n] = r
        # the identical second invocation: every stage "already done", no done.json rewritten
        assert all(r.status == "skipped" and r.reason == "done" for r in second.values()), command
        before, after = pipe.repeat[command]
        assert before == after, command
    # every registered stage but demo and report-tissue ran, each exactly once
    assert set(ran) == {s.name for s in registry.specs()} - NOT_RUN
    # each done.json: the config file's settings, and its upstreams' fingerprints chained in
    for name, r in ran.items():
        done = read_done(r.stage_dir)
        assert done["fingerprint"] == r.fingerprint and done["seed"] == 42, name
        for k, v in STAGE_SETTINGS.get(name, {}).items():
            assert done["config"][k] == (list(v) if isinstance(v, tuple) else v), (name, k)
        assert done["upstream"] == {u: ran[u].fingerprint for u in registry.get(name).requires}, name


def test_the_listing_shows_every_stage_done(pipe, registry):
    rows = listing(pipe.cfg)
    assert set(rows) == {s.name for s in registry.specs()}
    assert {n for n, (st, _) in rows.items() if st == "done"} == set(rows) - NOT_RUN
    assert {rows[n][0] for n in NOT_RUN} == {"missing"}


def test_a_killed_fit_leaves_no_done_json_and_resume_auto_finishes_it(pipe):
    k = pipe.killed
    assert "killed mid-fit" in k["error"]
    assert not k["lopo_done"] and k["lopo_started"] and k["fold0_done"] and k["fold1_step"] == KILL_AT[1]
    assert not k["final_started"]                       # the group stopped at the killed stage
    resumed = pipe.runs["fit-ldm"][0]
    lopo, final = resumed["ldm-lopo"], resumed["ldm-final"]
    assert lopo.status == "ran" and lopo.done["resumed"] and lopo.done["attempt"] == 2
    assert final.status == "ran" and not final.done["resumed"] and final.done["attempt"] == 1
    log = (lopo.stage_dir / "stage.log").read_text()
    assert "fold-0: finished in an earlier attempt; kept" in log and "fold-1: cached latents" in log
    assert not (lopo.stage_dir / STARTED_NAME).exists()


def test_resume_auto_gives_the_outputs_of_an_uninterrupted_run(pipe, copy):
    """`ldm-lopo --force` in a copy, with its cached latents deleted: both
    folds fitted from step 0, uninterrupted. Same bits as the resumed run;
    same fingerprint, so nothing downstream of it is stale."""
    resumed = pipe.runs["fit-ldm"][0]["ldm-lopo"]
    a, b = pipe.out / "gen/ldm/lopo", copy / "gen/ldm/lopo"
    for k in (0, 1):
        (b / f"fold-{k}" / ld.LATENTS_NAME).unlink()
    with pytest.MonkeyPatch.context() as mp:
        spy = StepSpy(mp, DiffusionLoss)
        ref = invoke(["ldm-lopo", "--config", str(pipe.cfg), "--out", str(copy), "--force"])["ldm-lopo"]
    assert spy.fits == 2 and spy.steps(1) == spy.steps(2) == list(range(LOOP["max_steps"]))
    assert ref.status == "ran" and not ref.done["resumed"] and ref.done["attempt"] == 1
    assert ref.fingerprint == resumed.fingerprint
    forced = (b / "stage.log").read_text().rsplit("--force", 1)[-1]          # this attempt's log lines
    assert "kept" not in forced and "cached" not in forced and forced.count("-> latents") == 2
    for k in (0, 1):
        fa, fb = a / f"fold-{k}", b / f"fold-{k}"
        assert_same_training(fb, fa)
        assert np.array_equal(np.load(fa / ld.LATENTS_NAME), np.load(fb / ld.LATENTS_NAME))
        assert (fa / ld.MMD_NAME).read_text() == (fb / ld.MMD_NAME).read_text()
        ga = torch.load(fa / ld.GENERATOR_NAME, weights_only=True)
        gb = torch.load(fb / ld.GENERATOR_NAME, weights_only=True)
        assert_same(ga["denoiser"]["state_dict"], gb["denoiser"]["state_dict"])
        sa, sb = resumed.done["summary"]["folds"][k], ref.done["summary"]["folds"][k]
        assert {x: sa[x] for x in FOLD_SUMMARY_KEYS} == {x: sb[x] for x in FOLD_SUMMARY_KEYS}
        assert sa["sample_check"]["sha256"] == sb["sample_check"]["sha256"]
    rows = listing(pipe.cfg, copy)
    assert {n for n, (st, _) in rows.items() if st == "done"} == set(rows) - NOT_RUN


def test_force_reruns_and_a_changed_fingerprint_makes_downstream_stale(pipe, registry, copy, tmp_path):
    before = read_done(copy / "pool")
    pool_sha = pool_fingerprint(copy / "pool")
    # --force, same config: re-run, same fingerprint and bytes; nothing is stale
    r = invoke(["pool", "--config", str(pipe.cfg), "--out", str(copy), "--force"])["pool"]
    assert r.status == "ran" and r.fingerprint == before["fingerprint"] and pool_fingerprint(copy / "pool") == pool_sha
    rows = listing(pipe.cfg, copy)
    assert {n for n, (st, _) in rows.items() if st == "done"} == set(rows) - NOT_RUN
    again = invoke(["fit-cvae", "--config", str(pipe.cfg), "--out", str(copy)])
    assert {n: x.status for n, x in again.items()} == {"cvae-lopo": "skipped", "cvae-final": "skipped"}

    # a changed config: a new fingerprint, and the stages that require the pool are stale
    cfg2 = write_config(tmp_path / "pipeline2.json", pipe.tiny.data_dir, copy, rows_per_class=20)
    r2 = invoke(["pool", "--config", str(cfg2), "--force"])["pool"]
    assert r2.status == "ran" and r2.fingerprint != before["fingerprint"]
    assert pool_fingerprint(copy / "pool") != pool_sha                          # fewer rows per class
    rows = listing(cfg2)
    stale = {n for n, (st, _) in rows.items() if st == "stale"}
    direct = {s.name for s in registry.specs() if "pool" in s.requires}
    assert direct and direct <= stale
    assert all("fingerprint changed" in rows[n][1] for n in stale)
    below = descendants(registry, "pool")
    assert {"cvae-lopo", "generate-cvae", "export-cvae", "lopo-utility"} <= below
    assert all(rows[n][0] == "done" for n in set(rows) - below - NOT_RUN)
    assert rows["pool"][0] == "done" and rows["qc-reference"][0] == "done"

    # re-running the next command re-runs its stages and carries the change one step down
    fit = invoke(["fit-cvae", "--config", str(cfg2)])
    assert {n: x.status for n, x in fit.items()} == {"cvae-lopo": "ran", "cvae-final": "ran"}
    assert fit["cvae-lopo"].done["upstream"]["pool"] == r2.fingerprint
    rows = listing(cfg2)
    assert rows["generate-cvae"][0] == "stale" and rows["cvae-final"][0] == "done"


# ============================================================================
# what the pipeline hands on
# ============================================================================

@pytest.mark.parametrize("source", SOURCES)
def test_the_exports_load_with_the_strict_manifest_check(pipe, source):
    d = pipe.out / "export" / source
    exp = load_synthetic_export(str(d), str(pipe.tiny.data_dir), num_classes=3, manifest_check="strict")
    n_dcis = int((np.load(pipe.tiny.data_dir / "y_train.npy") == DCIS).sum())
    assert exp.n_syn == n_dcis == 40 and np.all(exp.y == DCIS)
    assert np.all(exp.groups == SYNTHETIC_GROUP_IDS[source])
    assert exp.hwc == (pipe.tiny.patch, pipe.tiny.patch, pipe.tiny.n_bands)
    prov = json.loads((d / PROVENANCE_NAME).read_text())
    assert exp.export_id == prov["export_id"] == read_done(d)["summary"]["export_id"] and prov["source"] == source


def test_the_lopo_utility_decision_names_the_a4_source(pipe):
    doc = pipe.decision
    assert L.check_decision(doc) == [] and doc["a4_source"] in SOURCES
    assert pipe.runs["lopo-utility"][0]["lopo-utility"].done["summary"]["a4_source"] == doc["a4_source"]
    for src in SOURCES:                  # the decision is about THIS chain's generators
        assert doc["generators"][src]["choice_sha256"] == sha(pipe.out / "synth" / src / "choice.json")


@pytest.fixture(scope="module")
def trained(pipe, tmp_path_factory):
    """One epoch of `train.py --profile paper_recipe` + tiny model flags on the
    fixture, with the export lopo-utility chose for A4."""
    source = pipe.decision["a4_source"]
    export = pipe.out / "export" / source
    flags = list(TINY)
    flags[flags.index("--epochs") + 1] = "1"
    cwd = tmp_path_factory.mktemp("train")
    proc = _train_py(["--data_dir", str(pipe.tiny.data_dir), "--profile", "paper_recipe", *flags,
                      "--allow_missing_classes", "--synthetic_dir", str(export)], cwd)
    assert proc.returncode == 0, f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    (run_dir,) = list((cwd / "experiments").iterdir())
    return {"source": source, "export": export, "run": run_dir}


def test_train_py_trains_one_epoch_on_the_export(pipe, trained):
    run = trained["run"]
    assert run.name.endswith("_syn")
    cfg = json.loads((run / "config.json").read_text())
    prov = json.loads((trained["export"] / PROVENANCE_NAME).read_text())
    syn = cfg["synthetic"]
    assert syn["export_id"] == prov["export_id"] and syn["source"] == trained["source"]
    n_real = pipe.tiny.n_rows["train"]
    assert (syn["n_real"], syn["n_syn"], syn["n_total"]) == (n_real, 40, n_real + 40)
    assert syn["syn_class_counts"] == {"healthy": 0, "DCIS": 40, "IDC": 0}
    assert syn["source_manifest_sha256"] == sha(pipe.tiny.data_dir / "dataset_manifest.json")
    assert json.loads((run / "synthetic_mix_report.json").read_text()) == syn
    assert len(json.loads((run / "history.json").read_text())) == 1


def test_only_the_declared_stages_opened_val_or_test(pipe, registry):
    rec = read_access_log(pipe.out / ACCESS_LOG_NAME)
    assert rec and all(r["allowed"] for r in rec)
    readers = set(held_out_readers(registry))
    assert {r["stage"] for r in rec if r["split"] != "train"} <= readers
    assert not readers & {s.name for c in COMMANDS[1:] for s in registry.resolve([c])}   # generator side: train only
