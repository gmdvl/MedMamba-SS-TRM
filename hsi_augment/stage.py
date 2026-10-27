# -*- coding: utf-8 -*-
"""
hsi_augment/stage.py
====================
The stage-directory contract, and the runner that enforces it.

Every stage writes into its own directory, `<--out>/<StageSpec.dirname>/`:

    done.json      written atomically and LAST, only after the stage succeeded:
                   {schema_version, stage, status: "ok", fingerprint, config_hash,
                    config, seed, stage_seed_entropy, input_data_dir,
                    input_manifest_sha256, upstream: {stage: fingerprint},
                    outputs: {relpath: {sha256, size_bytes}}, git_sha, timings,
                    resumed, attempt, summary, versions}
    config.json    the resolved configuration, in `--config` file format, so
                   `--config <stage dir>/config.json` reproduces the stage
    stage.log      Python `logging`, mirrored to the console as `[stage] message`
    metrics.jsonl  one JSON object per `ctx.log_metrics(...)` call
    started.json   {fingerprint, attempt}: present while a run is in progress or
                   after it died; what `--resume auto` checks
    failed.json    the error, when the stage raised (never a done.json)

Why these rules
---------------
* Skip-if-done. A stage is done when `done.json` says status "ok", its
  `fingerprint` equals the one this invocation computes, and every recorded
  output still exists with its recorded size. The fingerprint is the sha256 of
  {stage, config_hash, master seed, input manifest sha256, upstream
  fingerprints}: change any of them and the stage re-runs, and because the
  upstream fingerprints are chained, so does everything downstream.
* The input build is identified by the sha256 of its `dataset_manifest.json`
  FILE. That manifest already records a sha256 per array, so hashing it is
  enough; re-hashing the 19 GB `X_train.npy` on every invocation is not needed.
* No `done.json` after an interruption. The runner deletes any old `done.json`
  BEFORE the stage starts, and writes the new one with a temp file + fsync +
  `os.replace` only after the stage function returned. A crash, Ctrl-C or
  SIGKILL anywhere in between leaves no `done.json`; sweeps
  (`run_experiments.py` `done_if`) and downstream stages therefore never mistake
  a half-finished stage for a finished one.
* `--force` re-runs a done stage. `--resume auto` sets `ctx.resume = True` when
  the previous attempt with the SAME fingerprint did not finish (its
  `started.json` is still there); a stage with checkpoints (#22) then continues
  from them. A changed config never resumes.
* Per-stage seeds: `np.random.SeedSequence([master_seed, zlib.crc32(name)])`.
  A stage's stream depends only on the master seed and its own name, so adding,
  removing or reordering stages never shifts another stage's seed.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hsi_augment import SCHEMA_VERSION
from hsi_augment.config import StageConfig, canonical_json, sha256_hex, to_jsonable
from hsi_augment.io import (ACCESS_LOG_NAME, ARRAY_KINDS, SplitArrays, SplitGuard, disk_preflight,
                            npy_writer, ram_guard, save_json, save_npy, sha256_file)

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.adapters.base import BuildAdapter
    from hsi_augment.registry import Registry, StageSpec

DONE_NAME = "done.json"
STARTED_NAME = "started.json"
FAILED_NAME = "failed.json"
CONFIG_NAME = "config.json"
LOG_NAME = "stage.log"
METRICS_NAME = "metrics.jsonl"
BOOKKEEPING_NAMES = frozenset({DONE_NAME, STARTED_NAME, FAILED_NAME, CONFIG_NAME, LOG_NAME,
                               METRICS_NAME, DONE_NAME + ".tmp"})

RESUME_MODES = ("off", "auto")
DEFAULT_SEED = 42
REPO_ROOT = Path(__file__).resolve().parent.parent

Key = Union[int, str]


class StageError(RuntimeError):
    """A hard failure the stage detected itself (e.g. P1 found a NaN). The CLI
    prints the message and exits 1, without a traceback."""


class StageNotReady(StageError):
    """An upstream stage is missing, not done, or built from other inputs."""


# ============================================================================
# options, seeds, fingerprints
# ============================================================================

@dataclass(frozen=True)
class RunOptions:
    """The invocation-level settings shared by every stage of one CLI call."""
    data_dir: str                 # the build's array dir (…/hsi) or its root
    out: str                      # output root, e.g. data/hsi_v9-trainsel-aug
    seed: int = DEFAULT_SEED      # master seed; each stage derives its own
    force: bool = False
    resume: str = "off"           # "off" | "auto"
    adapter: str = "histologyhsi_bc"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {self.seed!r}")
        if self.resume not in RESUME_MODES:
            raise ValueError(f"resume must be one of {RESUME_MODES}, got {self.resume!r}")


def _key_entropy(key: Key) -> int:
    if isinstance(key, bool):
        raise TypeError("seed keys must be ints or strings, not bools")
    if isinstance(key, (int, np.integer)):
        if key < 0:
            raise ValueError(f"integer seed keys must be >= 0, got {key}")
        return int(key)
    if isinstance(key, str):
        return zlib.crc32(key.encode("utf-8"))
    raise TypeError(f"seed keys must be ints or strings, got {type(key).__name__}")


def stage_seed_sequence(master_seed: int, stage_name: str, *keys: Key) -> np.random.SeedSequence:
    """`SeedSequence([master_seed, crc32(stage_name), *keys])`. Extra keys (a
    LOPO fold, an epoch, "sampling") give independent sub-streams; a string key
    is crc32-hashed. The same arguments always give the same stream."""
    return np.random.SeedSequence([int(master_seed), _key_entropy(stage_name),
                                   *(_key_entropy(k) for k in keys)])


def input_manifest_sha256(manifest_path: Union[str, os.PathLike]) -> str:
    p = Path(manifest_path)
    if not p.is_file():
        raise StageNotReady(f"input build has no {p.name} at {p}: the input cannot be fingerprinted")
    return sha256_file(p)


@lru_cache(maxsize=None)
def git_sha() -> str:
    """HEAD of this repository, '+dirty' if tracked files differ from it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True).returncode != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def compute_fingerprint(stage: str, config_hash: str, seed: int, input_sha256: str,
                        upstream: Mapping[str, str]) -> str:
    return sha256_hex(canonical_json({"stage": stage, "config_hash": config_hash, "seed": int(seed),
                                      "input_manifest_sha256": input_sha256,
                                      "upstream": dict(upstream)}))


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def read_json(path: Union[str, os.PathLike]) -> Optional[Dict[str, Any]]:
    """The parsed object, or None if the file is missing or unreadable."""
    try:
        with open(path) as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def read_done(stage_dir: Union[str, os.PathLike]) -> Optional[Dict[str, Any]]:
    return read_json(Path(stage_dir) / DONE_NAME)


def done_mismatch(done: Optional[Mapping[str, Any]], fingerprint: str,
                  stage_dir: Union[str, os.PathLike]) -> Optional[str]:
    """None if `done` proves the stage is complete for `fingerprint`; else why not."""
    if done is None:
        return "no done.json"
    if done.get("status") != "ok":
        return f"done.json status is {done.get('status')!r}"
    if done.get("fingerprint") != fingerprint:
        return "fingerprint changed (config, seed, input build or an upstream stage)"
    for rel, meta in (done.get("outputs") or {}).items():
        p = Path(stage_dir) / rel
        if not p.is_file():
            return f"output {rel} is missing"
        if meta.get("size_bytes") is not None and p.stat().st_size != meta["size_bytes"]:
            return f"output {rel} changed size"
    return None


# ============================================================================
# logging
# ============================================================================

class _ConsoleHandler(logging.StreamHandler):
    """Writes to whatever `sys.stdout` is when the record is emitted (pytest's
    capsys and the CLI's redirections replace it)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


class _ConsoleFormatter(logging.Formatter):
    """`[stage] message` - the repo's plain print style."""

    def __init__(self, stage: str):
        super().__init__()
        self.stage = stage

    def format(self, record: logging.LogRecord) -> str:
        tag = "ERROR: " if record.levelno >= logging.ERROR else \
              "WARNING: " if record.levelno >= logging.WARNING else ""
        return f"[{self.stage}] {tag}{record.getMessage()}"


def _open_logger(stage: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"hsi_augment.stage.{stage}")
    _close_logger(logger)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    ch = _ConsoleHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ConsoleFormatter(stage))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


# ============================================================================
# what a stage function receives
# ============================================================================

class StageContext:
    """Everything a stage function needs, so it never builds paths, seeds or
    loggers itself. Signature of a stage: `fn(ctx: StageContext, cfg) -> dict | None`
    (the dict becomes `done.json["summary"]`).

    Paths and outputs
        ctx.stage_dir, ctx.out_root, ctx.data_dir, ctx.path("figs", "a.png")
        ctx.upstream_dir("validate")        only for declared `requires`
        ctx.save_json("report.json", obj)   atomic; recorded with its sha256
        ctx.save_npy("mask.npy", arr)       atomic; disk preflight over 1 GB
        with ctx.npy_writer("X_pool.npy", np.float16, shape) as w: w.write(block)
        ctx.add_output(path)                anything written another way
    Data
        ctx.adapter                         BuildAdapter for --data_dir
        ctx.open_split("train", purpose="fit")   through the stage's SplitGuard
    Seeds
        ctx.rng(), ctx.rng("fold", 3)       np.random.Generator from the stage seed
        ctx.seed_int("torch")               32-bit int, e.g. for torch.manual_seed
    Logging and resources
        ctx.log(msg), ctx.warn(msg), ctx.log_metrics(step=1, loss=0.3)
        with ctx.timer("scan"): ...         seconds land in done.json["timings"]
        ctx.ram_guard(nbytes, "what"), ctx.disk_preflight(path, nbytes)
    Resume
        ctx.resume                          True only under --resume auto after an
                                            unfinished attempt with this fingerprint
    """

    def __init__(self, *, spec: "StageSpec", config: StageConfig, options: RunOptions,
                 stage_dir: Path, out_root: Path, adapter: "BuildAdapter", fingerprint: str,
                 resume: bool, attempt: int, upstream_dirs: Mapping[str, Path], logger: logging.Logger):
        self.spec = spec
        self.name = spec.name
        self.config = config
        self.options = options
        self.stage_dir = stage_dir
        self.out_root = out_root
        self.adapter = adapter
        self.data_dir = adapter.data_dir
        self.fingerprint = fingerprint
        self.resume = resume
        self.attempt = attempt
        self.seed = options.seed
        self.seed_sequence = stage_seed_sequence(options.seed, spec.name)
        self.logger = logger
        self._upstream_dirs = dict(upstream_dirs)
        self._outputs: Dict[str, Optional[str]] = {}
        self._timings: Dict[str, float] = {}
        self._t0 = time.time()
        self._guard: Optional[SplitGuard] = None

    # -- paths ------------------------------------------------------------------
    def path(self, *parts: Union[str, os.PathLike]) -> Path:
        """A path inside the stage directory; parent dirs are created."""
        p = self.stage_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def upstream_dir(self, name: str) -> Path:
        if name not in self._upstream_dirs:
            raise KeyError(f"stage {self.name!r} did not declare requires={name!r}; "
                           f"declared: {sorted(self._upstream_dirs)}")
        return self._upstream_dirs[name]

    # -- seeds -------------------------------------------------------------------
    def rng(self, *keys: Key) -> np.random.Generator:
        """A fresh Generator for (stage, *keys): the same keys give the same
        stream however many times, and in whatever order, it is asked for."""
        return np.random.default_rng(stage_seed_sequence(self.seed, self.name, *keys))

    def seed_int(self, *keys: Key) -> int:
        return int(stage_seed_sequence(self.seed, self.name, *keys).generate_state(1, np.uint32)[0])

    # -- logging -----------------------------------------------------------------
    def log(self, msg: str, level: int = logging.INFO) -> None:
        self.logger.log(level, msg)

    def warn(self, msg: str) -> None:
        self.logger.warning(msg)

    def log_metrics(self, record: Optional[Mapping[str, Any]] = None, **kw: Any) -> None:
        row = {"elapsed_s": round(time.time() - self._t0, 3), **(record or {}), **kw}
        with open(self.stage_dir / METRICS_NAME, "a") as f:
            f.write(json.dumps(to_jsonable(row), sort_keys=True, allow_nan=True) + "\n")

    @contextlib.contextmanager
    def timer(self, name: str) -> Iterator[None]:
        t = time.time()
        try:
            yield
        finally:
            self._timings[name] = self._timings.get(name, 0.0) + (time.time() - t)

    # -- outputs -------------------------------------------------------------------
    def _resolve_output(self, path: Union[str, os.PathLike]) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.stage_dir / p
        p = Path(os.path.abspath(p))
        root = Path(os.path.abspath(self.stage_dir))
        if root not in p.parents:
            raise ValueError(f"output {p} is outside the stage directory {root}")
        if p.parent == root and p.name in BOOKKEEPING_NAMES:
            raise ValueError(f"{p.name} is reserved for the stage runner")
        return p

    def add_output(self, path: Union[str, os.PathLike], *, sha256: Optional[str] = None) -> Path:
        """Record an output file; its sha256 (computed now unless given) and size
        go into done.json, and skip-if-done checks it still exists."""
        p = self._resolve_output(path)
        if not p.is_file():
            raise FileNotFoundError(f"output {p} does not exist")
        rel = p.relative_to(Path(os.path.abspath(self.stage_dir))).as_posix()
        self._outputs[rel] = sha256 or sha256_file(p)
        return p

    def save_json(self, rel: Union[str, os.PathLike], obj: Any) -> Path:
        p = self._resolve_output(rel)
        save_json(p, obj)
        return self.add_output(p)

    def save_npy(self, rel: Union[str, os.PathLike], array: np.ndarray) -> Path:
        p = self._resolve_output(rel)
        save_npy(p, array)
        return self.add_output(p)

    @contextlib.contextmanager
    def npy_writer(self, rel: Union[str, os.PathLike], dtype: Any, shape: Sequence[int]) -> Iterator[Any]:
        p = self._resolve_output(rel)
        with npy_writer(p, dtype, shape, sha256=True) as w:
            yield w
        self.add_output(p, sha256=w.sha256)

    # -- data and resources -----------------------------------------------------------
    @property
    def split_guard(self) -> SplitGuard:
        if self._guard is None:
            self._guard = SplitGuard(self.adapter, self.out_root / ACCESS_LOG_NAME, stage=self.name,
                                     allowed_purposes=self.spec.purposes)
        return self._guard

    def open_split(self, split: str, purpose: str, *, kinds: Sequence[str] = ARRAY_KINDS) -> SplitArrays:
        return self.split_guard.open(split, purpose, kinds=kinds)

    def ram_guard(self, projected_bytes: int, what: str = "work", **kw: Any) -> Dict[str, Any]:
        report = ram_guard(projected_bytes, what=f"{self.name}: {what}", **kw)
        self.logger.debug(f"ram guard: {report}")
        return report

    def disk_preflight(self, path: Union[str, os.PathLike], nbytes: int) -> Optional[Dict[str, Any]]:
        return disk_preflight(path, nbytes)


# ============================================================================
# the runner
# ============================================================================

@dataclass
class StagePlan:
    """What the runner works out before deciding to run or skip."""
    spec: "StageSpec"
    config: StageConfig
    options: RunOptions
    adapter: "BuildAdapter"
    stage_dir: Path
    input_sha256: str
    upstream: Dict[str, str]
    upstream_dirs: Dict[str, Path]
    fingerprint: str
    done: Optional[Dict[str, Any]]
    not_done_reason: Optional[str]


@dataclass
class StageResult:
    name: str
    status: str                      # "ran" | "skipped"
    stage_dir: Path
    fingerprint: str
    done: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def _make_adapter(options: RunOptions) -> "BuildAdapter":
    from hsi_augment.adapters import get_adapter
    try:
        return get_adapter(options.adapter, options.data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise StageNotReady(f"--data_dir {options.data_dir}: {e}") from e


def plan_stage(spec: "StageSpec", config: StageConfig, options: RunOptions, registry: "Registry",
               *, adapter: Optional["BuildAdapter"] = None) -> StagePlan:
    """Fingerprint the stage and check its upstreams; touches no file."""
    if not isinstance(config, spec.config_cls):
        raise TypeError(f"stage {spec.name!r} expects {spec.config_cls.__name__}, "
                        f"got {type(config).__name__}")
    adapter = adapter or _make_adapter(options)
    out_root = Path(options.out)
    input_sha = input_manifest_sha256(adapter.manifest_path)
    upstream: Dict[str, str] = {}
    upstream_dirs: Dict[str, Path] = {}
    for r in spec.requires:
        if r not in registry:
            raise StageNotReady(f"{spec.name} requires stage {r!r}, which is not registered "
                                f"(not implemented yet?)")
        up_dir = out_root / registry.get(r).dirname
        up = read_done(up_dir)
        if up is None or up.get("status") != "ok":
            raise StageNotReady(f"{spec.name} needs {r!r} done first: "
                                f"python augment_pipeline.py {r} --data_dir ... --out {options.out}")
        if up.get("input_manifest_sha256") != input_sha:
            raise StageNotReady(f"{spec.name}: upstream {r!r} was built from a different input build "
                                f"(manifest sha256 differs); re-run {r} first")
        if up.get("seed") != options.seed:
            raise StageNotReady(f"{spec.name}: upstream {r!r} ran with seed {up.get('seed')}, this run "
                                f"uses {options.seed}; re-run {r} with --seed {options.seed}")
        upstream[r] = str(up.get("fingerprint"))
        upstream_dirs[r] = up_dir
    fp = compute_fingerprint(spec.name, config.config_hash(), options.seed, input_sha, upstream)
    stage_dir = out_root / spec.dirname
    done = read_done(stage_dir)
    return StagePlan(spec=spec, config=config, options=options, adapter=adapter, stage_dir=stage_dir,
                     input_sha256=input_sha, upstream=upstream, upstream_dirs=upstream_dirs,
                     fingerprint=fp, done=done, not_done_reason=done_mismatch(done, fp, stage_dir))


def stage_status(spec: "StageSpec", config: StageConfig, options: RunOptions,
                 registry: "Registry") -> Tuple[str, str]:
    """('done' | 'stale' | 'failed' | 'incomplete' | 'missing' | 'blocked', reason)."""
    try:
        plan = plan_stage(spec, config, options, registry)
    except StageNotReady as e:
        return "blocked", str(e)
    if plan.not_done_reason is None:
        return "done", ""
    if plan.done is not None:
        return "stale", plan.not_done_reason
    if (plan.stage_dir / FAILED_NAME).is_file():
        failed = read_json(plan.stage_dir / FAILED_NAME) or {}
        return "failed", str(failed.get("error", ""))
    if (plan.stage_dir / STARTED_NAME).is_file():
        return "incomplete", "started.json present: interrupted or still running"
    return "missing", ""


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def run_stage(spec: "StageSpec", config: StageConfig, options: RunOptions, registry: "Registry",
              *, adapter: Optional["BuildAdapter"] = None) -> StageResult:
    """Skip if done; otherwise run `spec.fn(ctx, config)` under the contract
    in the module docstring. Exceptions from the stage propagate (after
    `failed.json` is written); `done.json` is never written for them."""
    plan = plan_stage(spec, config, options, registry, adapter=adapter)
    stage_dir, fp = plan.stage_dir, plan.fingerprint
    if not options.force and plan.not_done_reason is None:
        print(f"[{spec.name}] already done (fingerprint {fp[:12]}), skipping; --force to re-run",
              flush=True)
        return StageResult(spec.name, "skipped", stage_dir, fp, plan.done or {}, "done")

    stage_dir.mkdir(parents=True, exist_ok=True)
    # FIRST: no done.json may survive into a run that could be interrupted.
    for name in (DONE_NAME, DONE_NAME + ".tmp", FAILED_NAME):
        _remove(stage_dir / name)
    started = read_json(stage_dir / STARTED_NAME)
    same_attempt = bool(started) and started.get("fingerprint") == fp
    resume = options.resume == "auto" and not options.force and same_attempt
    attempt = int(started.get("attempt", 0)) + 1 if same_attempt else 1
    t_start = time.time()
    started_utc = _utc_now()
    save_json(stage_dir / STARTED_NAME, {"stage": spec.name, "fingerprint": fp, "attempt": attempt,
                                         "started_utc": started_utc, "pid": os.getpid()})
    save_json(stage_dir / CONFIG_NAME, {
        "schema_version": SCHEMA_VERSION, "data_dir": str(options.data_dir), "out": str(options.out),
        "seed": options.seed, "stages": {spec.name: config.to_dict()},
        "config_hashes": {spec.name: config.config_hash()}})

    logger = _open_logger(spec.name, stage_dir / LOG_NAME)
    ctx = StageContext(spec=spec, config=config, options=options, stage_dir=stage_dir,
                       out_root=Path(options.out), adapter=plan.adapter, fingerprint=fp, resume=resume,
                       attempt=attempt, upstream_dirs=plan.upstream_dirs, logger=logger)
    why = "--force" if options.force else (plan.not_done_reason or "")
    ctx.log(f"start: fingerprint {fp[:12]} config {config.config_hash()[:12]} seed {options.seed} "
            f"attempt {attempt}{' (resuming)' if resume else ''} - {why}")
    logger.debug(f"config: {config.to_json(indent=None)}")
    try:
        summary = spec.fn(ctx, config)
        if summary is not None and not isinstance(summary, Mapping):
            raise TypeError(f"stage {spec.name!r} returned {type(summary).__name__}; "
                            f"expected a dict or None")
    except BaseException as e:
        status = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
        try:
            ctx.log(f"{status}: {type(e).__name__}: {e}", logging.ERROR)
            save_json(stage_dir / FAILED_NAME, {
                "stage": spec.name, "status": status, "fingerprint": fp, "attempt": attempt,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-8000:], "utc": _utc_now()})
        except Exception:  # bookkeeping must never mask the stage's own error
            pass
        finally:
            _close_logger(logger)
        raise

    t_end = time.time()
    done = {
        "schema_version": SCHEMA_VERSION,
        "stage": spec.name,
        "status": "ok",
        "fingerprint": fp,
        "config_hash": config.config_hash(),
        "config": config.to_dict(),
        "seed": options.seed,
        "stage_seed_entropy": [int(v) for v in ctx.seed_sequence.entropy],
        "input_data_dir": str(options.data_dir),
        "input_manifest_sha256": plan.input_sha256,
        "upstream": plan.upstream,
        "outputs": {rel: {"sha256": sha, "size_bytes": (stage_dir / rel).stat().st_size}
                    for rel, sha in sorted(ctx._outputs.items())},
        "git_sha": git_sha(),
        "timings": {"started_utc": started_utc, "finished_utc": _utc_now(),
                    "wall_s": round(t_end - t_start, 3),
                    "sections_s": {k: round(v, 3) for k, v in ctx._timings.items()}},
        "resumed": resume,
        "attempt": attempt,
        "summary": to_jsonable(summary or {}),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }
    save_json(stage_dir / DONE_NAME, done)     # LAST: atomic, after every output exists
    _remove(stage_dir / STARTED_NAME)
    ctx.log(f"done in {t_end - t_start:.1f}s: {len(done['outputs'])} output(s) in {stage_dir}")
    _close_logger(logger)
    return StageResult(spec.name, "ran", stage_dir, fp, done, why)
