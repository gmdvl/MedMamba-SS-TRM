# -*- coding: utf-8 -*-
"""
test_phys_aug.py
================
Issue #20: `train.py --phys_aug` (training/train_cli.py, training/train_pipeline.py)
and the `aug-calibration` stage behind `--phys_tilt_max auto`
(hsi_augment/augcal.py, hsi_augment/stages/aug_calibration.py).

  * the flags are off by default, parse without any file, and `_phys` joins
    the run name only when on;
  * the calibration stage reads train only, and the tilt_max it picks meets
    the decision's criterion (change <= the paper recipe's own augmentation's)
    while the next grid value does not;
  * every SystemExit of the wiring: the issue's two (minority without
    --minority_augment on, a normalization other than global_zscore) and the
    three artifact refusals in a real `train.py` process, before the sha256
    pass over the build; the rest in process;
  * `train.py --synthetic_dir ... --minority_augment on --phys_aug minority`
    trains end to end and records `config["phys_aug"]`; so does `--phys_aug all`.

`--phys_aug off` staying bit-identical is the parity tests' job
(tests/test_train_e2e_parity.py, tests/test_synthetic_mix.py's empty-export
runs): the default path goes through `compose_with_physics(fn, None)`, which is
`fn` itself.

Run with: pytest tests/test_phys_aug.py -q
"""

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import hsi_augment.io as hio
from hsi_augment import augcal
from hsi_augment.augcal import (
    BASE_OP_NAME, CALIBRATION_KIND, PHYS_CALIBRATION_NAME, base_augment_op, choose_tilt, find_phys_calibration,
    paper_recipe_aug_config, read_phys_calibration, resolve_tilt_max,
)
from hsi_augment.io import read_access_log
from hsi_augment.registry import load_builtin_stages
from hsi_augment.stage import RunOptions, StageNotReady, run_stage, stage_seed_sequence
from hsi_augment.stages.aug_calibration import AugCalibrationConfig
from hsi_augment.stages.spectral_audit import SpectralAuditConfig
from hsi_augment.transforms import SpectralTilt, augmentation_report, load_phys_artifacts
from tests.hsi_augment_fixtures import make_tiny_build
from tests.test_synthetic_mix import MINAUG, TINY, _train_py, tiny_bound_hsi  # noqa: F401 (fixture)
from training.augmentation import AugmentationConfig
from training.npy_atomic import MANIFEST_NAME
from training.train_cli import resolve
from training.train_pipeline import _check_phys_aug, build_run_name

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE = "aug-calibration"
HSI = "data/hsi_v8-80_10_10_importance-new/hsi"
TS = "20261027_000000"


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ============================================================================
# flags and run name
# ============================================================================

def test_the_flags_are_off_by_default_and_nothing_is_checked_at_parse_time():
    args = resolve(["--data_dir", HSI])[0].args
    assert (args.phys_aug, args.phys_tilt_max, args.phys_noise_k, args.phys_od_delta, args.phys_artifacts) == \
        ("off", "auto", 1.0, 0.0, None)
    # sweeps parse their argv on machines without the artifacts (tests/test_sweeps.py)
    args = resolve(["--data_dir", HSI, "--phys_aug", "minority", "--phys_artifacts", "no/such/dir",
                    "--phys_tilt_max", "0.012", "--phys_noise_k", "0.5"])[0].args
    assert (args.phys_aug, args.phys_tilt_max, args.phys_noise_k) == ("minority", 0.012, 0.5)
    with pytest.raises(SystemExit):
        resolve(["--data_dir", HSI, "--phys_tilt_max", "big"])


def test_the_run_name_gains_phys_only_when_on_after_minaug_and_syn():
    off = resolve(["--data_dir", HSI])[0].args
    on = resolve(["--data_dir", HSI, "--phys_aug", "all"])[0].args
    assert build_run_name(on, TS, modality="x") == build_run_name(off, TS, modality="x") + "_phys"
    every = resolve(["--data_dir", HSI, "--minority_augment", "on", "--synthetic_dir", "e", "--phys_aug", "minority",
                     "--run_tag", "t1"])[0].args
    assert build_run_name(every, TS, modality="x").endswith("_optimal_recon_minaug_syn_phys_t1")


def test_the_base_augmentation_is_what_train_py_builds_for_paper_recipe():
    """`paper_recipe_aug_config` reads OPTIMAL_DEFAULTS; train.py builds its
    AugmentationConfig from the resolved flags of the same names."""
    args = resolve(["--data_dir", HSI, "--profile", "paper_recipe"])[0].args
    assert args.augment_preset == "custom"
    cfg = paper_recipe_aug_config(seed=args.seed)
    for f in dataclasses.fields(AugmentationConfig):
        assert getattr(cfg, f.name) == getattr(args, f.name), f.name
    assert cfg.spectral_scale_range > 0 and cfg.crop_scale_max_frac > 0     # the recipe, not the empty preset


# ============================================================================
# the decision rule and reading it back
# ============================================================================

def test_choose_tilt_picks_the_largest_value_within_the_base():
    c = choose_tilt(0.25, {0.01: 0.1, 0.02: 0.24, 0.03: 0.5})
    assert (c["tilt_max"], c["bound"], c["monotone"]) == (0.02, "interior", True)
    assert [r["within_base"] for r in c["grid"]] == [True, True, False]
    assert choose_tilt(0.25, {0.01: 0.1, 0.02: 0.2})["bound"] == "grid_max"
    none = choose_tilt(0.05, {0.01: 0.1, 0.02: 0.2})
    assert (none["tilt_max"], none["bound"]) == (0.0, "below_grid")
    assert choose_tilt(0.25, {0.01: 0.3, 0.02: 0.2})["monotone"] is False
    with pytest.raises(ValueError):
        augcal.check_grid((0.01, 0.6))


def _write_calibration(path: Path, tilt_max: float, manifest_sha: str = None, noise_within: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "kind": CALIBRATION_KIND, "schema_version": 1, "tilt_max": tilt_max, "tilt_bound": "interior",
        "base_corr_change": 0.3, "tilt_corr_change": 0.29,
        "noise_check": {"k": 1.0, "checked": True, "corr_frobenius": 0.03 if noise_within else 0.4,
                        "within_base": noise_within},
        "input_manifest_sha256": manifest_sha, "seed": 42}))
    return path


def test_resolve_tilt_max(tmp_path):
    assert resolve_tilt_max(0.01, None) == (0.01, {"source": "--phys_tilt_max", "calibration": None})
    with pytest.raises(FileNotFoundError, match="--phys_artifacts, which is not set"):
        resolve_tilt_max("auto", None)
    with pytest.raises(FileNotFoundError, match="there is none under"):
        resolve_tilt_max("auto", tmp_path)
    p = _write_calibration(tmp_path / "prep" / "augcal" / PHYS_CALIBRATION_NAME, 0.0137)
    assert find_phys_calibration(tmp_path) == p
    t, src = resolve_tilt_max("auto", tmp_path)
    assert t == 0.0137 and src["source"] == "calibration"
    assert src["calibration"]["path"] == str(p) and src["calibration"]["sha256"] == _sha256(p)
    p.write_text(json.dumps({"kind": "other", "tilt_max": 0.01}))
    with pytest.raises(ValueError, match="not a phys_aug_calibration"):
        read_phys_calibration(p)
    for bad in (0.5, -0.01, "0.01", None, float("inf")):
        p.write_text(json.dumps({"kind": CALIBRATION_KIND, "tilt_max": bad}))
        with pytest.raises(ValueError, match="tilt_max must be"):
            read_phys_calibration(p)


# ============================================================================
# the aug-calibration stage
# ============================================================================

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


def _run(reg, name, cfg, data_dir, out):
    return run_stage(reg.get(name), cfg, RunOptions(data_dir=str(data_dir), out=str(out)), reg)


@pytest.fixture(scope="module")
def calibrated(tiny, tmp_path_factory):
    """spectral-audit, then aug-calibration with the default grid, on the tiny build;
    np.load refuses every val/test array meanwhile."""
    out = tmp_path_factory.mktemp("augout")
    reg = load_builtin_stages()
    real_load = np.load

    def guarded_load(path, *a, **kw):
        name = os.path.basename(str(path))
        if name.endswith(("_val.npy", "_test.npy")):
            raise AssertionError(f"opened held-out array {name}")
        return real_load(path, *a, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(np, "load", guarded_load)
    try:
        _run(reg, "spectral-audit", SpectralAuditConfig(compare_sample_cap=60), tiny.data_dir, out)
        result = _run(reg, STAGE, AugCalibrationConfig(), tiny.data_dir, out)
    finally:
        mp.undo()
    cal = json.loads((out / "prep" / "augcal" / PHYS_CALIBRATION_NAME).read_text())
    return out, result, cal


def test_the_stage_is_a_train_only_preprocess_stage_after_spectral_audit():
    spec = load_builtin_stages().get(STAGE)
    assert spec.purposes == ("fit",) and not spec.reads_held_out()
    assert (spec.group, spec.dirname, spec.requires) == ("preprocess", "prep/augcal", ("spectral-audit",))
    # the stage module keeps torch out of --help, so it carries its own copy of the cap
    from hsi_augment import transforms
    from hsi_augment.stages import aug_calibration
    assert aug_calibration.MAX_TILT == transforms.MAX_TILT


def test_only_the_train_split_is_read(calibrated):
    out, _result, _cal = calibrated
    rec = [r for r in read_access_log(out / hio.ACCESS_LOG_NAME) if r["stage"] == STAGE]
    assert rec and {(r["split"], r["purpose"], r["allowed"]) for r in rec} == {("train", "fit", True)}
    assert {f for r in rec for f in r["files"]} == {"X_train.npy"}


def test_the_chosen_tilt_max_meets_the_criterion_and_the_next_grid_value_does_not(tiny, calibrated):
    out, result, cal = calibrated
    base, t = cal["base_corr_change"], cal["tilt_max"]
    assert cal["kind"] == CALIBRATION_KIND and cal["tilt_bound"] == "interior" and cal["grid_monotone"]
    assert base > 0 and 0 < t < max(AugCalibrationConfig().tilt_grid)
    grid = {r["tilt_max"]: r for r in cal["grid"]}
    assert set(grid) == set(AugCalibrationConfig().tilt_grid)
    assert grid[t]["within_base"] and grid[t]["corr_frobenius"] == cal["tilt_corr_change"] <= base
    assert all(r["within_base"] == (r["corr_frobenius"] <= base) for r in cal["grid"])
    assert not any(r["within_base"] for r in cal["grid"] if r["tilt_max"] > t)
    assert result.done["summary"]["tilt_max"] == t

    # re-measured from scratch: the stage's sample (its seed key), train.py's normalization, the report seed
    from hsi_augment import preprocess as pp
    X = np.load(tiny.data_dir / "X_train.npy", mmap_mode="r")
    rows = pp.sample_rows(np.random.default_rng(stage_seed_sequence(42, STAGE, "sample")), len(X), 20_000)
    mean, std = pp.train_py_sample_stats(X, 5000, 42)
    np.testing.assert_array_equal(cal["normalization"]["mean"], mean)
    x = np.nan_to_num(np.asarray(X[rows], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    # contiguous like NpyDatasetV16's patches: the recipe's spectral_noise fills `empty_like(patch)` in memory
    # order, so a strided view would put the same draws on other elements
    samples = np.ascontiguousarray(((x - mean) / std).transpose(0, 3, 1, 2))
    seed = int(stage_seed_sequence(42, STAGE, "augment").generate_state(1, np.uint32)[0])
    assert cal["report_seed"] == seed and cal["sample"]["rows"] == len(rows) == tiny.n_rows["train"]
    rep = augmentation_report(samples, [base_augment_op(mean, std), SpectralTilt(tiny.wavelengths, mean, std, t)],
                              seed=seed, include_all=False)
    assert rep["ops"][BASE_OP_NAME]["corr_frobenius"] == pytest.approx(base, rel=1e-9)
    assert rep["ops"]["spectral_tilt"]["corr_frobenius"] <= rep["ops"][BASE_OP_NAME]["corr_frobenius"]


def test_sensor_noise_is_checked_with_the_real_sigma_b(calibrated):
    out, _result, cal = calibrated
    nc = cal["noise_check"]
    sigma = np.load(out / "prep" / "spectral" / "noise_sigma.npy")
    assert nc["checked"] and nc["k"] == 1.0 and nc["within_base"] == (nc["corr_frobenius"] <= cal["base_corr_change"])
    assert nc["sigma_b_median"] == pytest.approx(float(np.median(sigma)))
    assert cal["noise_sigma"]["sha256"] == _sha256(out / "prep" / "spectral" / "noise_sigma.npy")


def test_outputs_provenance_and_what_train_py_reads_back(tiny, calibrated):
    out, result, cal = calibrated
    assert set(result.done["outputs"]) == {PHYS_CALIBRATION_NAME, augcal.AUG_REPORT_NAME, augcal.MONTAGE_NAME}
    assert cal["input_manifest_sha256"] == _sha256(tiny.data_dir / MANIFEST_NAME)
    assert cal["base_augmentation"]["config"] == {k: v for k, v in dataclasses.asdict(paper_recipe_aug_config()).items()
                                                  if k != "seed"}
    report = json.loads((out / "prep" / "augcal" / augcal.AUG_REPORT_NAME).read_text())
    assert set(report["scan"]["ops"]) >= {BASE_OP_NAME, "sensor_noise", f"spectral_tilt@{cal['tilt_max']:g}"}
    assert set(report["chosen"]["ops"]) == {"spectral_tilt", "sensor_noise", "all"}
    assert report["chosen"]["ops"]["spectral_tilt"]["corr_frobenius"] == pytest.approx(cal["tilt_corr_change"])
    # --phys_artifacts <out> finds both files
    assert resolve_tilt_max("auto", out)[0] == cal["tilt_max"]
    assert load_phys_artifacts(out).paths["noise_sigma"] == str(out / "prep" / "spectral" / "noise_sigma.npy")


def test_the_stage_needs_spectral_audit_first(tiny, tmp_path):
    reg = load_builtin_stages()
    with pytest.raises(StageNotReady, match="spectral-audit"):
        _run(reg, STAGE, AugCalibrationConfig(), tiny.data_dir, tmp_path)


def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)


def test_cli_runs_and_skips_when_done(tiny, tmp_path):
    r = _cli("preprocess", "--help")
    assert r.returncode == 0 and STAGE in r.stdout and "prep/augcal" in r.stdout
    out = tmp_path / "aug"
    r = _cli("spectral-audit", "--data_dir", tiny.data_dir, "--out", out, "--compare_sample_cap", "60")
    assert r.returncode == 0, r.stdout + r.stderr
    base = [STAGE, "--data_dir", tiny.data_dir, "--out", out, "--tilt_grid", "0.005", "0.01", "0.02", "0.04",
            "--montage", "off"]
    r = _cli(*base)
    assert r.returncode == 0, r.stdout + r.stderr
    cal = json.loads((out / "prep" / "augcal" / PHYS_CALIBRATION_NAME).read_text())
    assert [g["tilt_max"] for g in cal["grid"]] == [0.005, 0.01, 0.02, 0.04] and cal["montage"] is None
    assert cal["tilt_max"] in (0.005, 0.01, 0.02)
    assert "already done" in _cli(*base).stdout
    assert _cli(STAGE, "--data_dir", tiny.data_dir, "--out", out, "--tilt_grid", "0.6").returncode == 2


# ============================================================================
# train.py: every SystemExit
# ============================================================================

@pytest.fixture(scope="module")
def phys_dirs(tiny_bound_hsi, tmp_path_factory):  # noqa: F811
    """--phys_artifacts directories for the tiny bound split (8 bands), laid out
    like the augment pipeline's --out: both files, noise only, calibration only."""
    root = tmp_path_factory.mktemp("phys")
    sha = _sha256(tiny_bound_hsi.data / MANIFEST_NAME)
    dirs = {name: root / name for name in ("full", "noise_only", "calibration_only")}
    for name in ("full", "noise_only"):
        (dirs[name] / "prep" / "spectral").mkdir(parents=True)
        np.save(dirs[name] / "prep" / "spectral" / "noise_sigma.npy", np.full(8, 0.01, np.float32))
    for name in ("full", "calibration_only"):
        _write_calibration(dirs[name] / "prep" / "augcal" / PHYS_CALIBRATION_NAME, 0.02, sha)
    return dirs


@pytest.mark.parametrize("case, match", [
    ("minority_without_minaug", "--phys_aug minority applies the ops to the --minority_augment classes; pass "
                                "--minority_augment on"),
    ("not_global_zscore", "--phys_aug all needs --normalization global_zscore"),
    ("auto_without_calibration", r"--phys_tilt_max auto: .*phys_calibration.json.*there is none under"),
    ("auto_without_artifacts", r"--phys_tilt_max auto: .*under --phys_artifacts, which is not set"),
    ("noise_without_sigma", r"--phys_aug all: sensor_noise \(noise_k > 0\) needs noise_sigma"),
])
def test_train_py_refuses_before_it_reads_the_real_build(case, match, tiny_bound_hsi,  # noqa: F811
                                                         phys_dirs, tmp_path):
    argv = ["--data_dir", str(tiny_bound_hsi.data)] + TINY
    argv += {
        "minority_without_minaug": ["--phys_aug", "minority", "--phys_artifacts", str(phys_dirs["full"])],
        "not_global_zscore": ["--phys_aug", "all", "--normalization", "per_patch_zscore",
                              "--phys_artifacts", str(phys_dirs["full"])],
        "auto_without_calibration": ["--phys_aug", "all", "--phys_artifacts", str(phys_dirs["noise_only"])],
        "auto_without_artifacts": ["--phys_aug", "all"],
        "noise_without_sigma": ["--phys_aug", "all", "--phys_artifacts", str(phys_dirs["calibration_only"])],
    }[case]
    proc = _train_py(argv, tmp_path)
    assert proc.returncode != 0
    assert re.search(match, proc.stderr), proc.stderr[-2000:]
    assert "[dataset-manifest]" not in proc.stdout
    assert not list(tmp_path.glob("experiments/*/history.json"))


@pytest.mark.parametrize("phys", [[], ["--phys_aug", "minority", "--phys_tilt_max", "0.01", "--phys_noise_k", "0"]],
                         ids=["phys_off", "phys_minority"])
def test_a_negative_minority_strength_is_still_the_minority_augment_systemexit(phys, tiny_bound_hsi,  # noqa: F811
                                                                                tmp_path):
    """The minority branch builds both augment functions inside the try it always
    had, so `scaled_config`'s ValueError stays `--minority_augment: ...`, with the
    ops off and on."""
    argv = (["--data_dir", str(tiny_bound_hsi.data)] + TINY + list(MINAUG)
            + ["--minority_aug_strength", "-1"] + phys)
    proc = _train_py(argv, tmp_path)
    assert proc.returncode != 0
    assert "--minority_augment: strength must be >= 0" in proc.stderr, proc.stderr[-2000:]
    assert "Traceback" not in proc.stderr
    assert not list(tmp_path.glob("experiments/*/history.json"))


def _args(data, *flags):
    return resolve(["--data_dir", str(data)] + list(flags))[0].args


@pytest.mark.parametrize("flags, wavelengths, match", [
    (["--phys_tilt_max", "0", "--phys_noise_k", "0"], True, "with every op off"),
    (["--phys_noise_k", "-1"], True, r"--phys_noise_k must be a number >= 0"),
    (["--phys_od_delta", "nan"], True, r"--phys_od_delta must be a number >= 0"),
    (["--phys_tilt_max", "-0.01"], True, r"--phys_tilt_max must be a number >= 0"),
    (["--phys_tilt_max", "0.5"], True, r"tilt_max must be in \[0, 0.5\)"),
    (["--phys_od_delta", "0.1"], True, "od_stain_jitter .* needs od_components and s_glass"),
    ([], False, "spectral_tilt .* needs the band wavelengths"),
])
def test_every_other_refusal(flags, wavelengths, match, tiny_bound_hsi, phys_dirs):  # noqa: F811
    args = _args(tiny_bound_hsi.data, "--phys_aug", "all", "--phys_artifacts", str(phys_dirs["full"]), *flags)
    wl = np.load(tiny_bound_hsi.data / "wavelengths.npy") if wavelengths else None
    with pytest.raises(SystemExit, match=match):
        _check_phys_aug(args, wl, str(tiny_bound_hsi.data / "X_train.npy"))


def test_refusals_of_the_artifacts_themselves(tiny_bound_hsi, phys_dirs, tmp_path):  # noqa: F811
    wl, x_train = np.load(tiny_bound_hsi.data / "wavelengths.npy"), str(tiny_bound_hsi.data / "X_train.npy")
    with pytest.raises(SystemExit, match="--phys_artifacts: .* is not a directory"):
        _check_phys_aug(_args(tiny_bound_hsi.data, "--phys_aug", "all", "--phys_artifacts", str(tmp_path / "no")),
                        wl, x_train)
    np.save(tmp_path / "X_9_bands.npy", np.zeros((2, 3, 3, 9), np.float32))
    with pytest.raises(SystemExit, match="noise_sigma has 8 bands, global_mean/global_std 9"):
        _check_phys_aug(_args(tiny_bound_hsi.data, "--phys_aug", "all", "--phys_tilt_max", "0",
                              "--phys_artifacts", str(phys_dirs["full"])), wl, str(tmp_path / "X_9_bands.npy"))
    bad = tmp_path / "bad"
    _write_calibration(bad / PHYS_CALIBRATION_NAME, 0.02).write_text(json.dumps({"kind": "x"}))
    with pytest.raises(SystemExit, match="--phys_tilt_max auto: .*is not a phys_aug_calibration file"):
        _check_phys_aug(_args(tiny_bound_hsi.data, "--phys_aug", "all", "--phys_noise_k", "0",
                              "--phys_artifacts", str(bad)), wl, x_train)


def test_a_calibration_of_another_build_or_a_failed_noise_check_warns(tiny_bound_hsi, tmp_path, capsys):  # noqa: F811
    wl = np.load(tiny_bound_hsi.data / "wavelengths.npy")
    _write_calibration(tmp_path / PHYS_CALIBRATION_NAME, 0.02, "0" * 64, noise_within=False)
    np.save(tmp_path / "noise_sigma.npy", np.full(8, 0.01, np.float32))
    phys = _check_phys_aug(_args(tiny_bound_hsi.data, "--phys_aug", "all", "--phys_artifacts", str(tmp_path)), wl,
                           str(tiny_bound_hsi.data / "X_train.npy"))
    out = capsys.readouterr().out
    assert "calibrated on another build" in out and "more than the paper recipe's augmentation" in out
    assert phys.tilt_from["calibration"]["build_match"] is False and phys.tilt_max == 0.02


# ============================================================================
# train.py end to end
# ============================================================================

def test_train_py_with_synthetic_rows_minority_augment_and_phys_minority(tiny_bound_hsi,  # noqa: F811
                                                                         phys_dirs, tmp_path):
    full = phys_dirs["full"]
    argv = ["--data_dir", str(tiny_bound_hsi.data), "--profile", "paper_recipe"] + TINY + list(MINAUG) + [
        "--synthetic_dir", str(tiny_bound_hsi.syn), "--phys_aug", "minority", "--phys_artifacts", str(full)]
    proc = _train_py(argv, tmp_path)
    assert proc.returncode == 0, f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert "[phys-aug] minority: spectral_tilt tilt_max=0.02, sensor_noise k=1" in proc.stdout

    (run_dir,) = list((tmp_path / "experiments").iterdir())
    assert run_dir.name.endswith("_minaug_syn_phys")
    cfg = json.loads((run_dir / "config.json").read_text())
    phys = cfg["phys_aug"]
    noise = full / "prep" / "spectral" / "noise_sigma.npy"
    calib = full / "prep" / "augcal" / PHYS_CALIBRATION_NAME
    assert phys["mode"] == "minority" and phys["applied_to"]["rows"] == "minority"
    assert phys["applied_to"]["classes"] == ["DCIS"] and "scaled_config" in phys["applied_to"]["after"]
    assert (phys["tilt_max"], phys["tilt_max_from"]) == (0.02, "calibration")
    assert (phys["noise_k"], phys["od_delta"]) == (1.0, 0.0)
    assert phys["calibration"]["path"] == str(calib) and phys["calibration"]["sha256"] == _sha256(calib)
    assert phys["calibration"]["build_match"] is True and phys["calibration"]["base_corr_change"] == 0.3
    assert phys["artifacts_dir"] == str(full)
    assert phys["artifacts"] == {"noise_sigma": {"path": str(noise), "sha256": _sha256(noise)}}
    assert [op["op"] for op in phys["ops"]] == ["spectral_tilt", "sensor_noise"]
    assert phys["ops"][0]["tilt_max"] == 0.02 and phys["ops"][1]["k"] == 1.0
    assert phys["base_augment_preset"] == "custom"
    assert cfg["minority_augmentation"]["counts_after"]["DCIS"] == 3 * 8 + 6 and cfg["synthetic"]["n_syn"] == 6
    assert cfg["cli_args"]["phys_aug"] == "minority" and cfg["cli_args"]["phys_tilt_max"] == "auto"
    assert len(json.loads((run_dir / "history.json").read_text())) == 2


def test_train_py_with_phys_all_and_a_number_for_tilt_max(tiny_bound_hsi, phys_dirs, tmp_path):  # noqa: F811
    argv = ["--data_dir", str(tiny_bound_hsi.data), "--profile", "paper_recipe"] + TINY + [
        "--phys_aug", "all", "--phys_tilt_max", "0.01", "--phys_noise_k", "0.5",
        "--phys_artifacts", str(phys_dirs["noise_only"])]
    proc = _train_py(argv, tmp_path)
    assert proc.returncode == 0, f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    (run_dir,) = list((tmp_path / "experiments").iterdir())
    assert run_dir.name.endswith("_optimal_recon_phys")
    phys = json.loads((run_dir / "config.json").read_text())["phys_aug"]
    assert phys["mode"] == "all" and phys["applied_to"] == {"rows": "all", "after": "build_active(aug_cfg)"}
    assert (phys["tilt_max"], phys["tilt_max_from"], phys["calibration"]) == (0.01, "--phys_tilt_max", None)
    assert [(op["op"], op.get("tilt_max", op.get("k"))) for op in phys["ops"]] == [("spectral_tilt", 0.01),
                                                                                   ("sensor_noise", 0.5)]
