# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_core.py
==============================
The `hsi_augment` core (#9): config round trip and precedence, the stage
contract (skip-if-done, --force, config/input/upstream changes re-run, no
done.json after an interruption or a hard kill, --resume auto, per-stage
seeds), chunked float64 stats against numpy, SplitGuard, the resource guards,
atomic writes, the adapter's grid coordinates, the fixture's ground truth, and
the CLI. CPU only, on `tests/hsi_augment_fixtures.make_tiny_build`.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import subprocess
import sys
import textwrap
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

import hsi_augment.io as hio
import hsi_augment.stage as stage_mod
from hsi_augment import cli
from hsi_augment.adapters import (HistologyHSIBCAdapter, NonContiguousCaptureError, get_adapter,
                                  spectral_gaps, spectral_segments)
from hsi_augment.config import (ConfigError, StageConfig, add_config_flags, build_config, explicit_values,
                                load_config_file, param)
from hsi_augment.io import (RunningBandStats, SplitAccessError, SplitGuard, band_stats, disk_preflight,
                            iter_chunks, npy_writer, ram_guard, read_access_log, save_json, sha256_file)
from hsi_augment.registry import (PLANNED_COMMANDS, Registry, RegistryError, load_builtin_stages)
from hsi_augment.stage import (DONE_NAME, FAILED_NAME, STARTED_NAME, RunOptions, StageNotReady,
                               run_stage, stage_seed_sequence, stage_status)
from tests.hsi_augment_fixtures import make_tiny_build

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@dataclass(frozen=True)
class ToyConfig(StageConfig):
    value: int = param(1, help="an int")
    scale: float = 1.0
    bands: Tuple[int, ...] = (0, 1)
    mode: Optional[str] = param(None, help="what the toy stage does", choices=("raise", "interrupt", "die-once"))
    on: bool = False


def _opts(tiny, out, **kw) -> RunOptions:
    return RunOptions(data_dir=str(tiny.data_dir), out=str(out), **kw)


def _toy_registry(calls: list) -> Registry:
    reg = Registry()

    @reg.stage("alpha", config=ToyConfig, dirname="grp/alpha", purposes=("fit",))
    def alpha(ctx, cfg):
        calls.append(("alpha", ctx.resume))
        ctx.save_json("draws.json", {"draws": ctx.rng().integers(0, 1 << 30, size=4), "value": cfg.value})
        ctx.log_metrics(step=1, value=cfg.value)
        ctx.log("alpha ran")
        if cfg.mode == "raise":
            raise RuntimeError("boom")
        if cfg.mode == "interrupt":
            raise KeyboardInterrupt
        if cfg.mode == "die-once" and not ctx.resume:
            raise RuntimeError("first attempt dies after its checkpoint")
        return {"value": cfg.value}

    @reg.stage("beta", config=ToyConfig, requires=("alpha",), dirname="grp/beta")
    def beta(ctx, cfg):
        calls.append(("beta", ctx.resume))
        up = json.loads((ctx.upstream_dir("alpha") / "draws.json").read_text())
        ctx.save_json("sum.json", {"sum": sum(up["draws"])})

    return reg


def _run(reg, name, tiny, out, cfg=None, **kw):
    return run_stage(reg.get(name), cfg or ToyConfig(), _opts(tiny, out, **kw), reg)


# ============================================================================
# config
# ============================================================================

def test_config_round_trip_hash_and_coercion():
    c = ToyConfig(value=3, scale=2, bands=[4, 5])
    assert c.scale == 2.0 and isinstance(c.scale, float) and c.bands == (4, 5)
    assert ToyConfig.from_json(c.to_json()) == c
    assert ToyConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    # 2 vs 2.0, list vs tuple: same canonical JSON, same hash
    assert c.config_hash() == ToyConfig(value=3, scale=2.0, bands=(4, 5)).config_hash()
    assert c.canonical_json() == '{"bands":[4,5],"mode":null,"on":false,"scale":2.0,"value":3}'
    assert len(c.config_hash()) == 64
    assert c.replace(value=4).config_hash() != c.config_hash()
    with pytest.raises(ConfigError, match="unknown key"):
        ToyConfig.from_dict({"valeu": 1})
    with pytest.raises(ConfigError):
        ToyConfig(value=1.5)
    with pytest.raises(ConfigError):
        ToyConfig(on=1)
    with pytest.raises(ConfigError, match="not one of"):
        ToyConfig(mode="explode")
    with pytest.raises(ConfigError):
        ToyConfig(scale=float("nan")).config_hash()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.value = 9  # type: ignore[misc]


def test_config_file_supplies_defaults_and_typed_flags_win(tmp_path):
    import argparse
    p = argparse.ArgumentParser()
    add_config_flags(p, ToyConfig)
    ns = p.parse_args(["--value", "9", "--bands", "7", "8", "--on", "on", "--mode", "none"])
    explicit = explicit_values(ns, ToyConfig)
    assert explicit == {"value": 9, "bands": [7, 8], "on": True, "mode": None}
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(json.dumps({"seed": 3, "stages": {"toy": {"value": 2, "scale": 5}}}))
    section = load_config_file(str(cfg_file))["stages"]["toy"]
    cfg = build_config(ToyConfig, section, explicit)
    assert (cfg.value, cfg.scale, cfg.bands, cfg.on) == (9, 5.0, (7, 8), True)   # typed > file > default
    assert build_config(ToyConfig, section, {}).value == 2
    assert build_config(ToyConfig).value == 1
    cfg_file.write_text(json.dumps({"stagez": {}}))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config_file(str(cfg_file))


def test_config_class_checked_at_registration():
    @dataclass(frozen=True)
    class Shadows(StageConfig):
        seed: int = 1

    @dataclass(frozen=True)
    class NotAStageConfig:          # (a non-frozen StageConfig subclass cannot even be declared)
        a: int = 1

    reg = Registry()
    for bad in (Shadows, NotAStageConfig):
        with pytest.raises(RegistryError):
            reg.stage("x", config=bad)(lambda ctx, cfg: None)


# ============================================================================
# stage contract
# ============================================================================

def test_skip_if_done_then_force(tiny, tmp_path):
    calls = []
    reg = _toy_registry(calls)
    r1 = _run(reg, "alpha", tiny, tmp_path)
    assert r1.status == "ran" and calls == [("alpha", False)]
    d = r1.stage_dir
    assert d == tmp_path / "grp" / "alpha"
    done = json.loads((d / DONE_NAME).read_text())
    assert done["status"] == "ok" and done["stage"] == "alpha"
    assert done["config_hash"] == ToyConfig().config_hash()
    assert done["input_manifest_sha256"] == sha256_file(tiny.data_dir / "dataset_manifest.json")
    assert done["outputs"]["draws.json"]["sha256"] == sha256_file(d / "draws.json")
    assert done["seed"] == 42 and done["stage_seed_entropy"] == [42, zlib.crc32(b"alpha")]
    assert done["summary"] == {"value": 1} and done["git_sha"] and done["timings"]["wall_s"] >= 0
    assert not (d / STARTED_NAME).exists()
    assert (d / "config.json").is_file() and (d / "stage.log").is_file()

    r2 = _run(reg, "alpha", tiny, tmp_path)
    assert r2.status == "skipped" and len(calls) == 1

    r3 = _run(reg, "alpha", tiny, tmp_path, force=True)
    assert r3.status == "ran" and len(calls) == 2
    assert stage_status(reg.get("alpha"), ToyConfig(), _opts(tiny, tmp_path), reg)[0] == "done"


def test_config_seed_input_and_upstream_changes_rerun(tiny, tmp_path):
    calls = []
    reg = _toy_registry(calls)
    with pytest.raises(StageNotReady, match="needs 'alpha' done first"):
        _run(reg, "beta", tiny, tmp_path)
    assert not (tmp_path / "grp" / "beta" / DONE_NAME).exists()

    _run(reg, "alpha", tiny, tmp_path)
    assert _run(reg, "beta", tiny, tmp_path).status == "ran"
    assert _run(reg, "beta", tiny, tmp_path).status == "skipped"

    # a config change re-runs alpha, and beta follows through the fingerprint chain
    assert _run(reg, "alpha", tiny, tmp_path, ToyConfig(value=2)).status == "ran"
    assert stage_status(reg.get("beta"), ToyConfig(), _opts(tiny, tmp_path), reg)[0] == "stale"
    assert _run(reg, "beta", tiny, tmp_path).status == "ran"
    assert _run(reg, "beta", tiny, tmp_path).status == "skipped"

    # a missing output re-runs
    (tmp_path / "grp" / "beta" / "sum.json").unlink()
    assert _run(reg, "beta", tiny, tmp_path).status == "ran"

    # another seed: upstream must be re-run first
    with pytest.raises(StageNotReady, match="seed"):
        _run(reg, "beta", tiny, tmp_path, seed=7)

    # another input build (its manifest sha differs): alpha re-runs, beta refuses the stale upstream
    other = make_tiny_build(tmp_path / "other_build")
    opts = RunOptions(data_dir=str(other.data_dir), out=str(tmp_path))
    with pytest.raises(StageNotReady, match="different input build"):
        run_stage(reg.get("beta"), ToyConfig(), opts, reg)
    assert run_stage(reg.get("alpha"), ToyConfig(value=2), opts, reg).status == "ran"
    assert run_stage(reg.get("beta"), ToyConfig(), opts, reg).status == "ran"


@pytest.mark.parametrize("mode, status", [("raise", "failed"), ("interrupt", "interrupted")])
def test_interrupted_stage_leaves_no_done_json(tiny, tmp_path, mode, status):
    reg = _toy_registry([])
    _run(reg, "alpha", tiny, tmp_path)
    d = tmp_path / "grp" / "alpha"
    assert (d / DONE_NAME).is_file()
    exc = RuntimeError if mode == "raise" else KeyboardInterrupt
    with pytest.raises(exc):
        _run(reg, "alpha", tiny, tmp_path, ToyConfig(mode=mode))
    assert not (d / DONE_NAME).exists(), "the old done.json must not survive a re-run that died"
    failed = json.loads((d / FAILED_NAME).read_text())
    assert failed["status"] == status and (d / STARTED_NAME).is_file()
    assert stage_status(reg.get("alpha"), ToyConfig(mode=mode), _opts(tiny, tmp_path), reg)[0] == "failed"
    # downstream refuses it
    with pytest.raises(StageNotReady):
        _run(reg, "beta", tiny, tmp_path)


def test_error_while_writing_done_json_leaves_none(tiny, tmp_path, monkeypatch):
    reg = _toy_registry([])
    real = stage_mod.save_json

    def flaky(path, obj):
        if Path(path).name == DONE_NAME:
            raise OSError("disk full")
        return real(path, obj)

    monkeypatch.setattr(stage_mod, "save_json", flaky)
    with pytest.raises(OSError):
        _run(reg, "alpha", tiny, tmp_path)
    assert not (tmp_path / "grp" / "alpha" / DONE_NAME).exists()


def test_hard_kill_leaves_no_done_json_and_resume_auto_picks_it_up(tiny, tmp_path):
    script = tmp_path / "victim.py"
    script.write_text(textwrap.dedent("""
        import os, sys
        from dataclasses import dataclass
        from hsi_augment.config import StageConfig
        from hsi_augment.registry import Registry
        from hsi_augment.stage import RunOptions, run_stage

        @dataclass(frozen=True)
        class C(StageConfig):
            v: int = 0

        reg = Registry()

        @reg.stage("victim", config=C)
        def victim(ctx, cfg):
            ctx.save_json("partial.json", {"v": cfg.v})
            if cfg.v == 1:
                os._exit(9)          # SIGKILL-like: no finally, no atexit

        run_stage(reg.get("victim"), C(v=int(sys.argv[3])),
                  RunOptions(data_dir=sys.argv[1], out=sys.argv[2]), reg)
    """))
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    out = tmp_path / "out"

    def run(v):
        return subprocess.run([sys.executable, str(script), str(tiny.data_dir), str(out), str(v)],
                              cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120)

    assert run(0).returncode == 0 and (out / "victim" / DONE_NAME).is_file()
    killed = run(1)
    assert killed.returncode == 9, killed.stderr
    assert not (out / "victim" / DONE_NAME).exists()
    assert (out / "victim" / STARTED_NAME).is_file() and (out / "victim" / "partial.json").is_file()

    # the same stage and config, in this process: --resume auto sees the unfinished attempt
    @dataclass(frozen=True)
    class C(StageConfig):
        v: int = 0

    seen = []
    reg = Registry()

    @reg.stage("victim", config=C)
    def victim(ctx, cfg):
        seen.append((ctx.resume, ctx.attempt))
        ctx.save_json("partial.json", {"v": cfg.v})

    r = run_stage(reg.get("victim"), C(v=1), _opts(tiny, out, resume="auto"), reg)
    assert seen == [(True, 2)] and r.done["resumed"] is True and r.done["attempt"] == 2


def test_resume_auto_only_for_the_same_fingerprint(tiny, tmp_path):
    calls = []
    reg = _toy_registry(calls)
    with pytest.raises(RuntimeError):
        _run(reg, "alpha", tiny, tmp_path, ToyConfig(mode="die-once"))
    # changed config: a fresh start even under --resume auto
    with pytest.raises(RuntimeError):
        _run(reg, "alpha", tiny, tmp_path, ToyConfig(mode="die-once", value=5), resume="auto")
    # resume off: fresh start
    with pytest.raises(RuntimeError):
        _run(reg, "alpha", tiny, tmp_path, ToyConfig(mode="die-once", value=5), resume="off")
    # same fingerprint + auto: resumes and finishes
    r = _run(reg, "alpha", tiny, tmp_path, ToyConfig(mode="die-once", value=5), resume="auto")
    assert [c[1] for c in calls] == [False, False, False, True]
    assert r.done["resumed"] and r.done["attempt"] == 3
    # --force never resumes, even under --resume auto
    out2 = tmp_path / "forced"
    with pytest.raises(RuntimeError):
        _run(reg, "alpha", tiny, out2, ToyConfig(mode="die-once"))
    with pytest.raises(RuntimeError):
        _run(reg, "alpha", tiny, out2, ToyConfig(mode="die-once"), resume="auto", force=True)
    assert calls[-1] == ("alpha", False)


def test_stage_seeds_do_not_shift_when_a_stage_is_added(tiny, tmp_path):
    def registry(names):
        reg = Registry()
        for i, n in enumerate(names):
            def fn(ctx, cfg):
                ctx.save_json("draws.json", {"a": ctx.rng().integers(0, 1 << 62, size=3),
                                             "fold3": ctx.rng("fold", 3).integers(0, 1 << 62, size=3)})
            reg.stage(n, config=ToyConfig, order=i)(fn)
        return reg

    def draws(reg, out):
        res = {}
        for s in reg.specs():
            run_stage(s, ToyConfig(), _opts(tiny, out), reg)
            res[s.name] = json.loads((out / s.dirname / "draws.json").read_text())
        return res

    before = draws(registry(["pre", "post"]), tmp_path / "a")
    after = draws(registry(["new-first", "pre", "added", "post"]), tmp_path / "b")
    assert before["pre"] == after["pre"] and before["post"] == after["post"]
    assert before["pre"]["a"] != before["post"]["a"] and before["pre"]["a"] != before["pre"]["fold3"]
    ref = np.random.default_rng(np.random.SeedSequence([42, zlib.crc32(b"pre")])).integers(0, 1 << 62, size=3)
    assert before["pre"]["a"] == ref.tolist()
    # string keys are crc32-hashed, int keys used as is; same keys -> same stream
    ss = stage_seed_sequence(42, "pre", "fold", 3)
    assert list(ss.entropy) == [42, zlib.crc32(b"pre"), zlib.crc32(b"fold"), 3]
    with pytest.raises(ValueError):
        stage_seed_sequence(42, "pre", -1)


def test_stage_log_metrics_and_console_mirror(tiny, tmp_path, capsys):
    reg = _toy_registry([])
    _run(reg, "alpha", tiny, tmp_path)
    d = tmp_path / "grp" / "alpha"
    assert "alpha ran" in (d / "stage.log").read_text()
    rows = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["step"] == 1 and rows[0]["value"] == 1 and "elapsed_s" in rows[0]
    out = capsys.readouterr().out
    assert "[alpha] alpha ran" in out and "[alpha] done in" in out
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["stages"]["alpha"] == ToyConfig().to_dict() and cfg["seed"] == 42
    # it is a valid --config file
    assert load_config_file(str(d / "config.json"))["stages"]["alpha"]["value"] == 1


def test_outputs_must_stay_inside_the_stage_dir(tiny, tmp_path):
    reg = Registry()

    @reg.stage("sneaky", config=ToyConfig)
    def sneaky(ctx, cfg):
        ctx.save_json("../elsewhere.json", {})

    with pytest.raises(ValueError, match="outside the stage directory"):
        run_stage(reg.get("sneaky"), ToyConfig(), _opts(tiny, tmp_path), reg)

    reg2 = Registry()

    @reg2.stage("reserved", config=ToyConfig)
    def reserved(ctx, cfg):
        ctx.save_json(DONE_NAME, {"status": "ok"})

    with pytest.raises(ValueError, match="reserved"):
        run_stage(reg2.get("reserved"), ToyConfig(), _opts(tiny, tmp_path), reg2)
    assert not (tmp_path / "reserved" / DONE_NAME).exists()


# ============================================================================
# registry
# ============================================================================

def test_registry_validation_ordering_and_groups():
    reg = Registry()
    noop = lambda ctx, cfg: None  # noqa: E731
    reg.stage("b-stage", config=ToyConfig, requires=("a-stage",), group="prep", dirname="prep/b", order=1)(noop)
    reg.stage("a-stage", config=ToyConfig, group="prep", dirname="prep/a", order=5)(noop)
    reg.stage("c-stage", config=ToyConfig, dirname="c", order=0, in_all=False)(noop)
    assert [s.name for s in reg.resolve(["prep"])] == ["a-stage", "b-stage"]   # requires beats order
    assert [s.name for s in reg.resolve(["all"])] == ["a-stage", "b-stage"]
    assert [s.name for s in reg.resolve(["b-stage", "c-stage", "a-stage"])] == ["c-stage", "a-stage", "b-stage"]
    with pytest.raises(RegistryError, match="planned"):
        reg.resolve(["fit-cvae"])
    with pytest.raises(RegistryError, match="did you mean"):
        reg.resolve(["a-stag"])
    for kwargs in ({"dirname": "prep/a/sub"}, {"dirname": "prep"}, {"dirname": "../x"},
                   {"purposes": ("peek",)}, {"group": "a-stage"}, {"group": "all"}):
        with pytest.raises(RegistryError):
            reg.stage("d-stage", config=ToyConfig, **kwargs)(noop)
    for name in ("all", "Bad_Name", "prep"):
        with pytest.raises(RegistryError):
            reg.stage(name, config=ToyConfig, dirname="zz")(noop)
    cyc = Registry()
    cyc.stage("x", config=ToyConfig, requires=("y",))(noop)
    cyc.stage("y", config=ToyConfig, requires=("x",))(noop)
    with pytest.raises(RegistryError, match="cycle"):
        cyc.resolve(["x", "y"])


def test_builtin_registry_has_the_demo_and_every_planned_command():
    reg = load_builtin_stages()
    assert "demo" in reg and not reg.get("demo").in_all
    assert set(PLANNED_COMMANDS) == {"validate", "preprocess", "pool", "fit-cvae", "fit-ae", "fit-ldm",
                                     "generate", "qc", "export", "lopo-utility"}
    # The leakage policy is visible at registration: the default purposes are
    # train-only, so a stage reads val/test only if its registration says so.
    from hsi_augment.registry import StageSpec
    default = StageSpec(name="x", fn=None, config_cls=ToyConfig)
    assert not default.reads_held_out() and set(default.purposes) == set(hio.TRAIN_ONLY_PURPOSES)
    assert set(hio.HELD_OUT_PURPOSES) == {"structural", "final_eval"}
    assert not reg.get("demo").reads_held_out()


# ============================================================================
# io: chunks, stats, SplitGuard, guards, writes
# ============================================================================

def test_chunked_float64_stats_equal_numpy(tiny, monkeypatch):
    X = np.load(tiny.data_dir / "X_train.npy", mmap_mode="r")
    y = np.load(tiny.data_dir / "y_train.npy")
    ref = np.asarray(X, dtype=np.float64).reshape(-1, tiny.n_bands)

    def same(s, r):
        np.testing.assert_allclose(s.mean, r.mean(0), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(s.var(), r.var(0), rtol=1e-10)
        np.testing.assert_allclose(s.std(ddof=1), r.std(0, ddof=1), rtol=1e-10)
        np.testing.assert_array_equal(s.min, r.min(0))
        np.testing.assert_array_equal(s.max, r.max(0))
        assert np.all(s.count == r.shape[0])

    for chunk_rows in (7, 64, 100_000):
        same(band_stats(X, chunk_rows=chunk_rows), ref)
    # several float64 blocks inside each chunk
    monkeypatch.setattr(hio, "STATS_BLOCK_VALUES", 50)
    same(band_stats(X, chunk_rows=13), ref)
    monkeypatch.undo()

    # an unsorted fancy index with repeats
    idx = np.random.default_rng(1).choice(len(X), size=150, replace=True)
    same(band_stats(X, idx, chunk_rows=11), np.asarray(X[np.sort(idx)], np.float64).reshape(-1, tiny.n_bands))

    # per-class accumulators merge into the whole
    parts = [band_stats(X, np.flatnonzero(y == c), chunk_rows=9) for c in range(3)]
    merged = parts[0].merge(parts[1]).merge(parts[2])
    same(merged, ref)
    # JSON round trip is exact
    again = RunningBandStats.from_dict(json.loads(json.dumps(merged.to_dict())))
    np.testing.assert_array_equal(again.mean, merged.mean)
    np.testing.assert_array_equal(again.var(), merged.var())

    # non-finite values are counted and excluded
    bad = np.array(X[:4], dtype=np.float32)
    bad[0, 0, 0, 2] = np.nan
    bad[1, 1, 1, 2] = np.inf
    s = RunningBandStats(tiny.n_bands).update(bad)
    finite = bad[..., 2][np.isfinite(bad[..., 2])].astype(np.float64)
    assert s.n_nonfinite.tolist() == [0, 0, 2, 0, 0, 0, 0, 0]
    assert s.count[2] == finite.size and np.isclose(s.mean[2], finite.mean()) and np.isclose(s.max[2], finite.max())


def test_iter_chunks_slices_fancy_and_masks(tiny):
    X = np.load(tiny.data_dir / "X_train.npy", mmap_mode="r")
    got = np.concatenate([c.data for c in iter_chunks(X, chunk_rows=50)])
    np.testing.assert_array_equal(got, X)
    for index in ([5, 3, 3, 9], [3, 3, 5], [10, 11, 12], np.arange(240) % 7 == 0):
        chunks = list(iter_chunks(X, np.asarray(index), chunk_rows=2))
        rows = np.concatenate([c.rows for c in chunks])
        idx = np.asarray(index)
        expect = np.flatnonzero(idx) if idx.dtype == bool else np.sort(idx)
        np.testing.assert_array_equal(rows, expect)
        np.testing.assert_array_equal(np.concatenate([c.data for c in chunks]), X[expect])
    assert next(iter_chunks(X, [0, 1], dtype=np.float64)).data.dtype == np.float64
    with pytest.raises(IndexError):
        next(iter_chunks(X, [240]))


def test_split_guard_refuses_held_out_splits_and_logs_every_access(tiny, tmp_path):
    adapter = get_adapter("histologyhsi_bc", tiny.data_dir)
    log = tmp_path / "split_access_log.jsonl"
    guard = SplitGuard(adapter, log, stage="gen")
    refused = 0
    for purpose in ("generator", "qc", "pool", "fit", "describe", "lopo"):
        for split in ("val", "validation", "test"):
            with pytest.raises(SplitAccessError, match="reserved"):
                guard.open(split, purpose)
            refused += 1
    train = guard.open("train", "generator")
    assert isinstance(train.X, np.memmap) and train.X.shape == (tiny.n_rows["train"], 5, 5, 8)
    assert guard.open("val", "structural").n == tiny.n_rows["val"]
    assert guard.open("test", "final_eval").n == tiny.n_rows["test"]
    with pytest.raises(SplitAccessError, match="unknown purpose"):
        guard.open("train", "peek")

    # a stage can use only the purposes it declared
    pool_guard = SplitGuard(adapter, log, stage="pool", allowed_purposes=("pool",))
    with pytest.raises(SplitAccessError, match="did not declare"):
        pool_guard.open("val", "structural")
    pool_guard.open("train", "pool")

    rec = read_access_log(log)
    assert len(rec) == refused + 3 + 1 + 2
    assert sum(not r["allowed"] for r in rec) == refused + 2
    assert all(r["split"] in ("train", "val", "test") and r["purpose"] and "utc" in r for r in rec)
    allowed_held_out = {(r["split"], r["purpose"]) for r in rec if r["allowed"] and r["split"] != "train"}
    assert allowed_held_out == {("val", "structural"), ("test", "final_eval")}
    assert "X_train.npy" in rec[refused]["files"]


def test_a_generator_stage_reading_val_fails_without_done_json(tiny, tmp_path):
    reg = Registry()

    @reg.stage("fit-gen", config=ToyConfig, purposes=("generator",))
    def fit_gen(ctx, cfg):
        ctx.open_split("train", "generator")
        ctx.open_split("val", "generator")

    with pytest.raises(SplitAccessError):
        run_stage(reg.get("fit-gen"), ToyConfig(), _opts(tiny, tmp_path), reg)
    assert not (tmp_path / "fit-gen" / DONE_NAME).exists()
    rec = read_access_log(tmp_path / "split_access_log.jsonl")
    assert [(r["stage"], r["split"], r["allowed"]) for r in rec] == [("fit-gen", "train", True),
                                                                    ("fit-gen", "val", False)]


def test_disk_preflight_and_ram_guard():
    Usage = collections.namedtuple("Usage", "total used free")
    assert disk_preflight("/nowhere/x.npy", 10**9) is None                  # <= 1 GB: unchecked
    ok = disk_preflight("/nowhere/x.npy", 3 * 10**9, usage_fn=lambda p: Usage(0, 0, 5 * 10**9))
    assert ok["need"] == 4 * 10**9
    with pytest.raises(hio.ResourceError, match="disk preflight"):
        disk_preflight("/nowhere/x.npy", 3 * 10**9, usage_fn=lambda p: Usage(0, 0, 3 * 10**9))

    assert ram_guard(400 * 10**6, ram_info={"available_mb": 1000.0})["checked"]
    with pytest.raises(hio.ResourceError, match="RAM guard"):
        ram_guard(600 * 10**6, what="pool", ram_info={"available_mb": 1000.0})
    assert ram_guard(10**15, ram_info={})["checked"] is False               # unmeasurable: allowed


def test_atomic_writes(tmp_path):
    p = save_json(tmp_path / "a" / "b.json", {"f": np.float32(1.5), "i": np.int64(3), "arr": np.arange(2)})
    assert json.loads(p.read_text()) == {"f": 1.5, "i": 3, "arr": [0, 1]}

    data = np.arange(7 * 3 * 2, dtype=np.float16).reshape(7, 3, 2)
    with npy_writer(tmp_path / "x.npy", np.float16, data.shape) as w:
        for a in range(0, 7, 3):
            w.write(data[a:a + 3])
    np.testing.assert_array_equal(np.load(tmp_path / "x.npy"), data)
    assert w.sha256 == sha256_file(tmp_path / "x.npy")
    with pytest.raises(Exception):
        with npy_writer(tmp_path / "short.npy", np.float16, data.shape) as w:
            w.write(data[:3])              # 3 of 7 rows: refused at commit
    assert not (tmp_path / "short.npy").exists() and not (tmp_path / "short.npy.tmp").exists()


# ============================================================================
# adapter
# ============================================================================

def test_adapter_metadata_on_the_fixture(tiny):
    a = get_adapter("histologyhsi_bc", tiny.root)          # the build root works as well as hsi/
    assert a.data_dir == tiny.data_dir and isinstance(a, HistologyHSIBCAdapter)
    assert a.class_names == tiny.class_names and a.minority_classes == (tiny.minority_class,)
    assert (a.n_bands, a.patch_size, a.stride, a.grid_cols) == (8, 5, 5, 5)
    assert a.spectral_gaps() == [tiny.gap_after] == spectral_gaps(tiny.wavelengths)
    assert spectral_segments(tiny.wavelengths) == [(0, tiny.gap_after + 1), (tiny.gap_after + 1, 8)]
    assert a.array_header("X", "validation") == ((tiny.n_rows["val"], 5, 5, 8), np.dtype("float16"))
    assert {c.split for c in a.capture_index.values()} == {"train", "val", "test"}
    assert a.split_patients() == tiny.patients
    assert a.pass1_train_patients() == tiny.patients["train"]
    np.testing.assert_array_equal(a.selected_wavelengths, a.wavelengths)
    assert a.capture_gain_report["suspect_captures"] == list(tiny.suspect_captures)
    for split in ("train", "val", "test"):
        y = np.load(tiny.data_dir / f"y_{split}.npy")
        caps = np.load(tiny.data_dir / f"captures_{split}.npy")
        groups = np.load(tiny.data_dir / f"groups_{split}.npy")
        assert all(a.capture_index[int(c)].label == int(l) for c, l in zip(caps, y))
        assert all(a.capture_index[int(c)].patient == int(g) for c, g in zip(caps, groups))
    assert a.describe()["n_rows"] == tiny.n_rows


def test_adapter_coordinates_match_the_fixture_grid(tiny, tmp_path):
    a = get_adapter("histologyhsi_bc", tiny.data_dir)
    guard = SplitGuard(a, tmp_path / "log.jsonl", stage="test")
    for split in ("train", "val", "test"):
        co = guard.open(split, "structural", kinds=("captures",)).coordinates()
        np.testing.assert_array_equal(np.stack([co.r, co.c], 1), tiny.grid_rc[split])
        assert co.has_grid.all()
    co = guard.open("val", "structural").coordinates()
    assert co.grid_shape[tiny.short_captures[0]] == (3, 5)


def test_adapter_coordinates_real_sized_grid_and_edge_cases(tiny):
    a = HistologyHSIBCAdapter(tiny.data_dir, grid_cols=91)   # what a real build resolves to
    caps = np.repeat([4, 7, 9], [4914, 2002, 100])
    co = a.coordinates(caps)
    assert co.grid_shape == {4: (54, 91), 7: (22, 91), 9: None}
    k = 4914 + 1000
    assert (co.ordinal[k], co.r[k], co.c[k]) == (1000, *divmod(1000, 91))
    assert (co.r[-1], co.c[-1], co.has_grid[-1]) == (-1, -1, False)
    assert co.has_grid[:4914 + 2002].all()
    with pytest.raises(NonContiguousCaptureError):
        a.coordinates(np.array([1, 1, 2, 2, 1]))
    assert a.non_contiguous_captures(np.array([1, 1, 2, 2, 1])) == [1]
    assert HistologyHSIBCAdapter.GRID_COLS == 91


REAL_DATA = Path(os.environ.get("HSI_AUGMENT_REAL_DATA", REPO_ROOT / "data" / "hsi_v9-trainsel" / "hsi"))


@pytest.mark.skipif(not (REAL_DATA / "dataset_manifest.json").is_file(), reason="real trainsel build not present")
def test_adapter_and_fixture_schemas_on_the_real_build(tiny):
    """JSON files and .npy headers only: no X rows are read."""
    real = get_adapter("histologyhsi_bc", REAL_DATA)
    shape, dtype = real.array_header("X", "train")
    assert shape[1:] == (11, 11, 32) and dtype == np.float16
    assert real.grid_cols == 91 and real.stride == 11 and real.minority_classes == (1,)
    grids = collections.Counter((c.split, real.grid_shape(c.n)) for c in real.capture_index.values())
    assert grids[("train", (54, 91))] == 499 and grids[("val", (22, 91))] == 10
    assert real.spectral_gaps() == [17]
    assert set(real.pass1_train_patients()) == set(real.split_patients()["train"])
    np.testing.assert_array_equal(real.selected_wavelengths, real.wavelengths)

    fx = get_adapter("histologyhsi_bc", tiny.data_dir)
    def keys(obj):
        return set(obj) if isinstance(obj, dict) else set()
    rd = json.loads(real.capture_index_path.read_text())
    fd = json.loads(fx.capture_index_path.read_text())
    assert keys(next(iter(rd.values()))) == keys(next(iter(fd.values())))
    assert keys(real.split_assignment) == keys(fx.split_assignment)
    assert keys(real.pass1_scope) == keys(fx.pass1_scope)
    assert keys(real.capture_gain_report) == keys(fx.capture_gain_report)
    assert keys(next(iter(real.capture_gain_report["per_capture"].values()))) == \
        keys(next(iter(fx.capture_gain_report["per_capture"].values())))
    assert keys(real.manifest) == keys(fx.manifest)
    assert set(real.manifest["files"]) == set(fx.manifest["files"])
    assert keys(next(iter(real.manifest["files"].values()))) == keys(next(iter(fx.manifest["files"].values())))


# ============================================================================
# the fixture's ground truth
# ============================================================================

def test_fixture_ground_truth(tiny):
    from training.npy_atomic import verify_against_manifest
    assert verify_against_manifest(str(tiny.data_dir)) == []
    X = np.load(tiny.data_dir / "X_train.npy")
    assert X.dtype == np.float16 and X.shape == (tiny.n_rows["train"], tiny.patch, tiny.patch, tiny.n_bands)
    assert np.isfinite(X).all() and (X > 0).all()

    # glass is spectrally flat, tissue absorbs
    m = X.astype(np.float64).mean(axis=(1, 2))
    cos = m.sum(1) / (np.linalg.norm(m, axis=1) * np.sqrt(tiny.n_bands))
    sam = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    glass = tiny.is_glass["train"]
    assert 0.1 < glass.mean() < 0.5 and sam[glass].max() < 1.5 < 3.0 < sam[~glass].min()

    # the duplicate is byte-exact, between two DCIS patients' tissue rows
    assert X[tiny.duplicate_row].tobytes() == X[tiny.duplicate_of].tobytes()
    y = np.load(tiny.data_dir / "y_train.npy")
    g = np.load(tiny.data_dir / "groups_train.npy")
    assert y[tiny.duplicate_row] == y[tiny.duplicate_of] == 1 and g[tiny.duplicate_row] != g[tiny.duplicate_of]
    assert not glass[tiny.duplicate_row]
    assert sum(X[i].tobytes() == X[tiny.duplicate_of].tobytes() for i in range(len(X))) == 2

    # grid neighbours' facing edges correlate; a shuffled grid's do not
    caps = np.load(tiny.data_dir / "captures_train.npy")
    img = X.astype(np.float64).mean(-1)                       # (n, P, P) band-mean image
    rng = np.random.default_rng(0)

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    true_c, shuf_c = [], []
    for cap in np.unique(caps):
        rows = np.flatnonzero(caps == cap)
        for grid, out in ((img[rows], true_c), (img[rows][rng.permutation(len(rows))], shuf_c)):
            grid = grid.reshape(-1, tiny.grid_cols, tiny.patch, tiny.patch)
            for r in range(grid.shape[0]):
                for c in range(tiny.grid_cols - 1):
                    out.append(corr(grid[r, c][:, -1], grid[r, c + 1][:, 0]))
    assert np.median(true_c) > 0.8 and np.median(true_c) > np.median(shuf_c) + 0.4

    # one DCIS patient per held-out split, two in train
    assert tiny.dcis_patients == {"train": (101, 102), "val": (201,), "test": (301,)}


def test_fixture_is_deterministic(tmp_path):
    a = make_tiny_build(tmp_path / "a", seed=3)
    b = make_tiny_build(tmp_path / "b", seed=3)
    for name in ("X_train.npy", "y_val.npy", "captures_test.npy", "capture_index.json"):
        assert (a.data_dir / name).read_bytes() == (b.data_dir / name).read_bytes()
    assert a.duplicate_row == b.duplicate_row


# ============================================================================
# CLI
# ============================================================================

def _cli(*args, cwd=REPO_ROOT):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300)


def test_cli_help_lists_every_planned_command():
    r = _cli("--help")
    assert r.returncode == 0, r.stderr
    for name in [*PLANNED_COMMANDS, "all", "demo", "--data_dir", "--out", "--seed", "--config", "--force",
                 "--resume"]:
        assert name in r.stdout
    r = _cli("demo", "--help")
    assert r.returncode == 0 and "--chunk_rows" in r.stdout and "--max_rows" in r.stdout
    # a command that is still only planned, while one is left (fit-cvae was the example until #23)
    planned = sorted(load_builtin_stages().planned_only().items())
    if planned:
        name, where = planned[0]
        r = _cli(name, "--data_dir", "x", "--out", "y")
        assert r.returncode == 2 and where.split()[0] in r.stderr


def test_every_command_help_formats():
    # argparse %-formats help text, so a bare "%" in a param's help (e.g. "40%)") crashes `<stage> --help`
    reg = load_builtin_stages()
    for name in reg.command_names():
        assert cli.build_stage_parser(reg, [name]).format_help(), name


def test_cli_demo_stage_runs_skips_and_forces(tiny, tmp_path):
    out = tmp_path / "aug"
    base = ["demo", "--data_dir", tiny.data_dir, "--out", out, "--chunk_rows", "7"]
    r = _cli(*base)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[demo] start" in r.stdout and "demo: ran" in r.stdout
    done = json.loads((out / "demo" / DONE_NAME).read_text())
    assert done["config"]["chunk_rows"] == 7 and done["summary"]["rows"] == tiny.n_rows["train"]
    stats = json.loads((out / "demo" / "band_stats.json").read_text())["stats"]
    X = np.load(tiny.data_dir / "X_train.npy").astype(np.float64).reshape(-1, tiny.n_bands)
    np.testing.assert_allclose(stats["mean"], X.mean(0), rtol=1e-12)
    np.testing.assert_allclose(stats["std"], X.std(0), rtol=1e-10)

    r = _cli(*base)
    assert r.returncode == 0 and "already done" in r.stdout and "demo: skipped" in r.stdout
    r = _cli(*base, "--force")
    assert r.returncode == 0 and "demo: ran" in r.stdout
    r = _cli(*base, "--note", "changed")
    assert r.returncode == 0 and "demo: ran" in r.stdout

    rec = read_access_log(out / "split_access_log.jsonl")
    assert rec and {(x["stage"], x["split"], x["purpose"], x["allowed"]) for x in rec} == \
        {("demo", "train", "describe", True)}


def test_cli_config_file_precedence_list_and_errors(tiny, tmp_path, capsys):
    out = tmp_path / "aug"
    cfg = tmp_path / "pipeline.json"
    cfg.write_text(json.dumps({"data_dir": str(tiny.data_dir), "out": str(out), "seed": 7,
                               "stages": {"demo": {"chunk_rows": 11, "note": "from-file"}}}))
    assert cli.main(["demo", "--config", str(cfg), "--note", "typed"]) == 0
    done = json.loads((out / "demo" / DONE_NAME).read_text())
    assert (done["config"]["chunk_rows"], done["config"]["note"], done["seed"]) == (11, "typed", 7)
    # the stage's own config.json reproduces it exactly
    capsys.readouterr()
    assert cli.main(["demo", "--config", str(out / "demo" / "config.json")]) == 0
    assert "already done" in capsys.readouterr().out

    assert cli.main(["--list", "--data_dir", str(tiny.data_dir), "--out", str(out), "--seed", "7"]) == 0
    listing = capsys.readouterr().out
    assert "stale" in listing                                                  # defaults != the done config
    # the planned line is printed only while a planned command is left (lopo-utility, #30, was the last)
    assert ("planned (not implemented yet)" in listing) == bool(load_builtin_stages().planned_only())

    # 'all' is the CPU prep chain (through the pool); the generator fits are their own commands
    assert cli.main(["all", "--data_dir", str(tiny.data_dir), "--out", str(out)]) == 0
    assert cli.main(["nope", "--data_dir", str(tiny.data_dir), "--out", str(out)]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"stages": {"demo": {"chunk_rowz": 1}}}))
    assert cli.main(["demo", "--config", str(bad), "--data_dir", str(tiny.data_dir), "--out", str(out)]) == 2
    with pytest.raises(SystemExit):
        cli.main(["demo", "--data_dir", str(tiny.data_dir)])                       # --out missing
    with pytest.raises(SystemExit):
        cli.main(["demo", "--data_dir", str(tiny.data_dir), "--out", str(out), "--chunk_rows", "x"])
