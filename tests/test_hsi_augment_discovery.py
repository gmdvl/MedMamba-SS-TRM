# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_discovery.py
===================================
Phase 1, the `discovery` stage (#11): the tiny fixture passes with only its
planted within-train duplicate as a warning; the content statistics equal
numpy's; the SplitGuard log shows structural reads only; and corrupted copies
of the fixture (NaN / +-Inf, a label or patient that contradicts its capture,
a non-contiguous capture, a wrong count, a Pass-1 scope leak, a cross-split
duplicate, a sha256 mismatch, unequal row counts) each fail exactly the check
meant to catch them, with no done.json. CPU only.

The corrupted builds are copies made here, on top of
`tests/hsi_augment_fixtures.make_tiny_build`, which this file does not edit.
The real-build smoke test runs only when HSI_AUGMENT_REAL_DATA points at a
build's `hsi` dir; it reads headers, JSON and the y / groups / captures
sidecars, never X.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

from hsi_augment import discovery as disc
from hsi_augment.config import ConfigError
from hsi_augment.io import read_access_log
from hsi_augment.registry import load_builtin_stages
from hsi_augment.stage import DONE_NAME, FAILED_NAME, RunOptions, StageError, StageResult, run_stage
from hsi_augment.stages.discovery import DiscoveryConfig
from tests.hsi_augment_fixtures import make_tiny_build
from training.npy_atomic import MANIFEST_NAME, write_dataset_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_DIR = Path("prep") / "validate"
SPLITS = ("train", "val", "test")
CONTENT_CHECKS = ("content.nonfinite", "content.float16_saturation", "content.all_zero_patches")
LEAKAGE_CHECKS = ("leakage.cross_split", "leakage.within_split")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


def _run(data_dir: Path, out: Path, **cfg: Any) -> StageResult:
    reg = load_builtin_stages()
    return run_stage(reg.get("discovery"), DiscoveryConfig(**cfg),
                     RunOptions(data_dir=str(data_dir), out=str(out)), reg)


def _reports(out: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    d = out / STAGE_DIR
    return (json.loads((d / disc.DISCOVERY_REPORT).read_text()),
            json.loads((d / disc.INTEGRITY_REPORT).read_text()))


@pytest.fixture(scope="module")
def clean(tiny, tmp_path_factory):
    """One default run on the untouched fixture, shared by the read-only tests."""
    out = tmp_path_factory.mktemp("clean_out")
    result = _run(tiny.data_dir, out)
    return result, out, *_reports(out)


# ---- corrupted copies ---------------------------------------------------------------

def _copy(tiny, dest: Path) -> Path:
    shutil.copytree(tiny.root, dest)
    return dest / "hsi"


def _remanifest(data_dir: Path) -> None:
    """Rewrite the manifest for the edited arrays, so only the check under test fails."""
    m = json.loads((data_dir / MANIFEST_NAME).read_text())
    write_dataset_manifest(str(data_dir), list(m["files"]), sha256=True, extra=m.get("extra"))


def _edit_npy(path: Path, fn: Callable[[np.ndarray], Any]) -> None:
    a = np.load(path)
    b = fn(a)
    np.save(path, a if b is None else b)


def _edit_json(path: Path, fn: Callable[[Any], None]) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj, indent=2))


def _run_failing(data_dir: Path, out: Path, **cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with pytest.raises(StageError) as err:
        _run(data_dir, out, **cfg)
    stage = out / STAGE_DIR
    assert not (stage / DONE_NAME).exists() and (stage / FAILED_NAME).is_file()
    report, integrity = _reports(out)          # both reports are written before the stage raises
    assert integrity["status"] == "fail" and integrity["failed"]
    assert all(name in str(err.value) for name in integrity["failed"])
    return report, integrity


# ============================================================================
# registration and config
# ============================================================================

def test_registration_and_config():
    spec = load_builtin_stages().get("discovery")
    assert (spec.group, spec.dirname, spec.purposes) == ("validate", "prep/validate", ("structural",))
    assert spec.reads_held_out() and spec.in_all and spec.requires == ()
    cfg = DiscoveryConfig()
    assert cfg.verify_sha256 and cfg.content_scan and cfg.leakage_rows is None and cfg.percentiles == (0.05, 99.95)
    for bad in ({"chunk_rows": 0}, {"scan_max_rows": 0}, {"leakage_rows": -1}, {"expected_bands": 0},
                {"percentiles": (101.0,)}, {"percentiles": ()}):
        with pytest.raises(ConfigError):
            DiscoveryConfig(**bad)


# ============================================================================
# the clean fixture
# ============================================================================

def test_fixture_passes_cleanly(tiny, clean):
    result, out, report, integrity = clean
    assert result.status == "ran" and (out / STAGE_DIR / DONE_NAME).is_file()
    # the only finding is the fixture's planted within-train duplicate (for the memorization test)
    assert integrity["status"] == "warn" and integrity["failed"] == [] and integrity["skipped"] == []
    assert integrity["warnings"] == ["leakage.within_split"]
    within = integrity["checks"]["leakage.within_split"]["details"]["splits"]
    assert within["train"]["n_groups"] == 1 and within["val"]["n_groups"] == within["test"]["n_groups"] == 0
    assert within["train"]["examples"][0]["rows"] == sorted([tiny.duplicate_of, tiny.duplicate_row])
    assert integrity["counts"] == {"pass": len(integrity["checks"]) - 1, "warn": 1, "fail": 0, "skip": 0}

    splits = report["splits"]
    for s in SPLITS:
        assert splits[s]["n_rows"] == tiny.n_rows[s]
        assert splits[s]["patients"] == list(tiny.patients[s])
        assert splits[s]["n_captures"] == len(tiny.captures[s]) and splits[s]["non_contiguous_captures"] == []
    assert splits["train"]["capture_sizes"] == {"20": 12} and splits["train"]["grid_shapes"] == {"4x5": 12}
    assert splits["val"]["grid_shapes"] == {"3x5": 1, "4x5": 1}          # the short capture, like patient 65's

    wl = report["wavelengths"]
    assert wl["values_nm"] == tiny.wavelengths.tolist() and wl["selected_wavelengths_match"] is True
    assert [g["after_band"] for g in wl["gaps"]] == [tiny.gap_after] and wl["gaps"][0]["width_nm"] == 225.0
    assert report["class_names"] == list(tiny.class_names) and report["minority_classes"] == [tiny.minority_class]

    npy = report["files"]["npy"]
    assert npy["hsi/X_train.npy"]["shape"] == [tiny.n_rows["train"], tiny.patch, tiny.patch, tiny.n_bands]
    assert npy["hsi/X_train.npy"]["in_manifest"] and npy["hsi/X_train.npy"]["size_matches_header"]
    assert "selected_band_indices.npy" in npy and not npy["selected_band_indices.npy"]["in_manifest"]
    assert report["pass1_scope"]["n_train_captures"] == 12
    assert report["capture_gain_report"]["suspect_captures"] == list(tiny.suspect_captures)
    assert report["leakage"]["mode"] == ["all"] and report["leakage"]["rows_hashed"] == tiny.n_rows

    summary = result.done["summary"]
    assert summary["scope_verified"] is True and summary["nonfinite"] == 0
    assert summary["rows_scanned"] == tiny.n_rows and summary["train_capture_sizes"] == {"20": 12}
    assert summary["peak_rss_mb"] > 0


def test_content_stats_equal_numpy(tiny, clean):
    """Per split and class: count/min/max exact, mean/std to rounding, and the
    percentiles exactly numpy's inverted_cdf over the float16 values."""
    report = clean[2]
    content = report["content"]
    assert content["quantile_method"] == "float16_exact" and content["percentiles"] == [0.05, 99.95]
    qs = [0.0005, 0.9995]
    for s in SPLITS:
        X = np.load(tiny.data_dir / f"X_{s}.npy").astype(np.float64).reshape(tiny.n_rows[s], -1, tiny.n_bands)
        y = np.load(tiny.data_dir / f"y_{s}.npy")
        got = content["splits"][s]
        assert got["complete"] and got["rows_scanned"] == tiny.n_rows[s] and got["fit_allowed"] == (s == "train")
        groups = {"all": np.ones(len(y), dtype=bool)}
        groups.update({tiny.class_names[k]: y == k for k in np.unique(y)})
        assert set(got["classes"]) == set(groups) - {"all"}
        for name, sel in groups.items():
            vals = X[sel].reshape(-1, tiny.n_bands)
            d = got["all"] if name == "all" else got["classes"][name]
            pb = d["per_band"]
            assert d["rows"] == int(sel.sum()) and pb["count"] == [len(vals)] * tiny.n_bands
            np.testing.assert_array_equal(pb["min"], vals.min(0))
            np.testing.assert_array_equal(pb["max"], vals.max(0))
            np.testing.assert_allclose(pb["mean"], vals.mean(0), rtol=1e-12)
            np.testing.assert_allclose(pb["std"], vals.std(0), rtol=1e-10)
            q = np.quantile(vals, qs, axis=0, method="inverted_cdf")
            np.testing.assert_array_equal(pb["p0.05"], q[0])
            np.testing.assert_array_equal(pb["p99.95"], q[1])
            assert d["nan"] == d["posinf"] == d["neginf"] == d["all_zero_patches"] == d["float16_saturated"] == 0


def test_split_guard_log_shows_only_structural_reads(tiny, clean):
    out = clean[1]
    rec = read_access_log(out / "split_access_log.jsonl")
    assert rec and {(r["stage"], r["purpose"], r["allowed"]) for r in rec} == {("discovery", "structural", True)}
    assert {r["split"] for r in rec} == set(SPLITS)
    files = {f for r in rec for f in r["files"]}
    # the sha256 pass and the scan both announce the held-out X files
    assert {"X_val.npy", "X_test.npy", "captures_test.npy", "y_test.npy"} <= files
    assert sum("X_test.npy" in r["files"] for r in rec) == 2


# ============================================================================
# injected corruption: each is caught by its own check, and nothing else fails
# ============================================================================

@pytest.mark.parametrize("split,row,value,field", [("val", 7, np.nan, "nan"), ("train", 101, np.inf, "posinf"),
                                                   ("test", 33, -np.inf, "neginf")])
def test_injected_nonfinite_is_caught(tiny, tmp_path, split, row, value, field):
    data = _copy(tiny, tmp_path / "bad")
    _edit_npy(data / f"X_{split}.npy", lambda a: a.__setitem__((row, 2, 3, 5), value))
    _remanifest(data)
    report, integrity = _run_failing(data, tmp_path / "out")
    assert integrity["failed"] == ["content.nonfinite"]
    d = integrity["checks"]["content.nonfinite"]["details"]["splits"][split]
    assert d[field] == 1 and d["rows_with_nonfinite"] == 1 and d["first_nonfinite_rows"] == [row]
    # the bad value is counted, and kept out of the band's statistics
    pb = report["content"]["splits"][split]["all"]["per_band"]
    assert pb[field][5] == 1 and pb["count"][5] == tiny.n_rows[split] * tiny.patch ** 2 - 1
    assert all(v is not None for v in pb["mean"])


@pytest.mark.parametrize("kind,check", [("y", "provenance.labels"), ("groups", "provenance.patients")])
def test_label_or_patient_mismatch_is_caught(tiny, tmp_path, kind, check):
    data = _copy(tiny, tmp_path / "bad")
    row = 30
    if kind == "y":
        _edit_npy(data / "y_train.npy", lambda a: a.__setitem__(row, (a[row] + 1) % 3))
    else:   # another TRAIN patient, so only the row-level check can see it
        other = next(p for p in tiny.patients["train"] if p != int(np.load(data / "groups_train.npy")[row]))
        _edit_npy(data / "groups_train.npy", lambda a: a.__setitem__(row, other))
    _remanifest(data)
    _, integrity = _run_failing(data, tmp_path / "out")
    assert integrity["failed"] == [check]
    d = integrity["checks"][check]["details"]["splits"]
    assert d["train"]["n_bad"] == 1 and d["train"]["examples"][0]["row"] == row
    assert d["val"]["n_bad"] == d["test"]["n_bad"] == 0


def test_non_contiguous_capture_is_caught(tiny, tmp_path):
    """Swap one row of two train captures in all four arrays: labels, patients
    and counts stay consistent, only contiguity breaks."""
    data = _copy(tiny, tmp_path / "bad")
    i, j = 5, 45
    caps = np.load(data / "captures_train.npy")
    a_cap, b_cap = int(caps[i]), int(caps[j])
    assert a_cap != b_cap
    for kind in ("X", "y", "groups", "captures"):
        _edit_npy(data / f"{kind}_train.npy", lambda a: a.__setitem__([i, j], a[[j, i]]))
    _remanifest(data)
    report, integrity = _run_failing(data, tmp_path / "out")
    assert integrity["failed"] == ["provenance.contiguous"]
    d = integrity["checks"]["provenance.contiguous"]["details"]["splits"]["train"]
    assert d["n_bad"] == 2 and {e["capture"] for e in d["examples"]} == {a_cap, b_cap}
    assert sorted(report["splits"]["train"]["non_contiguous_captures"]) == sorted([a_cap, b_cap])


def _wrong_count(tiny, data):
    _edit_json(data / "capture_index.json", lambda ci: ci[str(tiny.captures["train"][3])].__setitem__("n", 25))


def _scope_leak(tiny, data):
    _edit_json(data.parent / "pass1_scope_report.json",
               lambda r: r["train_patients"].append(str(tiny.patients["val"][0])))


def _selected_wavelengths(tiny, data):
    _edit_npy(data.parent / "selected_wavelengths.npy", lambda a: a + np.float32(1.0))


def _cross_split_duplicate(tiny, data):
    src = np.load(data / "X_train.npy")[50]
    _edit_npy(data / "X_test.npy", lambda a: a.__setitem__(3, src))
    _remanifest(data)


def _sha256_mismatch(tiny, data):     # same size, different bytes; the manifest is NOT rewritten
    _edit_npy(data / "X_val.npy", lambda a: a.__setitem__((0, 0, 0, 0), a[0, 0, 0, 0] + np.float16(0.5)))


def _row_counts(tiny, data):
    _edit_npy(data / "y_val.npy", lambda a: a[:-1])
    _remanifest(data)


@pytest.mark.parametrize("corrupt,expected", [
    (_wrong_count, ["provenance.counts"]),
    (_scope_leak, ["scope.pass1_train_patients"]),
    (_selected_wavelengths, ["wavelengths.selected_match"]),
    (_cross_split_duplicate, ["leakage.cross_split"]),
    (_sha256_mismatch, ["manifest.sha256"]),
    (_row_counts, ["headers.row_counts"]),
], ids=lambda v: v.__name__.strip("_") if callable(v) else None)
def test_other_hard_failures(tiny, tmp_path, corrupt, expected):
    data = _copy(tiny, tmp_path / "bad")
    corrupt(tiny, data)
    report, integrity = _run_failing(data, tmp_path / "out")
    assert integrity["failed"] == expected
    checks = integrity["checks"]
    if corrupt is _scope_leak:
        assert "held-out patients [201]" in checks["scope.pass1_train_patients"]["message"]
    if corrupt is _cross_split_duplicate:
        pair = checks["leakage.cross_split"]["details"]["pairs"]["train-test"]
        assert pair["n_shared_patches"] == 1
        assert pair["examples"][0]["train_rows"] == [50] and pair["examples"][0]["test_rows"] == [3]
    if corrupt is _row_counts:
        # row-level work on inconsistent arrays is skipped, never attempted
        assert {f"provenance.{n}" for n in disc.PROVENANCE_CHECKS} | set(CONTENT_CHECKS) | set(LEAKAGE_CHECKS) \
            <= set(integrity["skipped"])
        assert "content" not in report and "splits" not in report


def test_all_zero_patch_is_not_leakage(tiny, tmp_path):
    data = _copy(tiny, tmp_path / "zeros")
    for s in ("train", "test"):
        _edit_npy(data / f"X_{s}.npy", lambda a: a.__setitem__(0, 0))
    _remanifest(data)
    _run(data, tmp_path / "out")
    _, integrity = _reports(tmp_path / "out")
    assert integrity["failed"] == [] and "content.all_zero_patches" in integrity["warnings"]
    zero = integrity["checks"]["content.all_zero_patches"]["details"]["splits"]
    assert zero["train"] == {"patches": 1, "first_rows": [0]} and zero["test"] == {"patches": 1, "first_rows": [0]}
    pair = integrity["checks"]["leakage.cross_split"]["details"]["pairs"]["train-test"]
    assert integrity["checks"]["leakage.cross_split"]["status"] == "pass"
    assert pair["all_zero_patch_shared"] is True and pair["n_shared_patches"] == 0


# ============================================================================
# cheaper modes
# ============================================================================

def test_headers_and_json_only_mode_never_opens_x(tiny, tmp_path):
    result = _run(tiny.data_dir, tmp_path, verify_sha256=False, content_scan=False, leakage_rows=0)
    report, integrity = _reports(tmp_path)
    assert result.status == "ran" and integrity["status"] == "pass" and integrity["failed"] == []
    assert integrity["skipped"] == ["manifest.sha256", *CONTENT_CHECKS, *LEAKAGE_CHECKS]
    assert "content" not in report and "leakage" not in report
    assert integrity["checks"]["provenance.contiguous"]["status"] == "pass"
    rec = read_access_log(tmp_path / "split_access_log.jsonl")
    assert len(rec) == 3 and not any(f.startswith("X_") for r in rec for f in r["files"])


def test_partial_scan_and_sampled_leakage(tiny, tmp_path):
    _run(tiny.data_dir, tmp_path, scan_max_rows=7, leakage_rows=10)
    report, integrity = _reports(tmp_path)
    assert integrity["failed"] == [] and "content.nonfinite" in integrity["warnings"]   # partial: not a pass
    assert "7 of 240 rows" in integrity["checks"]["content.nonfinite"]["message"]
    content = report["content"]["splits"]
    assert all(content[s]["rows_scanned"] == 7 and not content[s]["complete"] for s in SPLITS)
    assert content["train"]["all"]["rows"] == 7
    assert report["leakage"]["mode"] == ["sample"] and report["leakage"]["rows_hashed"] == {s: 10 for s in SPLITS}


def test_scan_split_hashes_rows_inside_and_beyond_the_scan(tiny):
    X = np.load(tiny.data_dir / "X_train.npy", mmap_mode="r")
    y = np.load(tiny.data_dir / "y_train.npy")
    want = np.array([1, 3, 100, 200], dtype=np.int64)
    sc = disc.scan_split("train", X, y, scan_rows=7, chunk_rows=3, hash_rows=want)
    assert sc.rows_scanned == 7 and sum(b.rows for b in sc.buckets.values()) == 7
    np.testing.assert_array_equal(sc.digests.rows, want)
    assert sc.digests.digests.tobytes() == disc.digest_rows(np.asarray(X)[want]).tobytes()
    every = disc.scan_split("train", X, y, scan_rows=0, chunk_rows=50, hash_rows=None)
    assert every.buckets == {} and every.digests.mode == "all"
    np.testing.assert_array_equal(every.digests.rows, np.arange(len(X)))
    none = disc.scan_split("train", X, y, scan_rows=len(X), chunk_rows=50, hash_rows=np.empty(0, np.int64))
    assert none.digests is None
    a = disc.leakage_hash_rows(240, 10, np.random.default_rng(5))
    b = disc.leakage_hash_rows(240, 10, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    assert len(a) == 10 and np.all(np.diff(a) > 0)
    assert disc.leakage_hash_rows(240, None, np.random.default_rng(0)) is None
    assert len(disc.leakage_hash_rows(240, 0, np.random.default_rng(0))) == 0


# ============================================================================
# building blocks
# ============================================================================

def test_float16_histogram_quantiles_are_exact():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((400, 3, 3, 4)) * 300).astype(np.float16)
    x[0, 0, 0, :] = [0.0, -0.0, 6e-8, -6e-8]                   # zeros and subnormals
    x[1, 1, 1, 0], x[2, 2, 2, 1], x[3, 0, 1, 2] = np.nan, np.inf, -np.inf
    flat = x.astype(np.float64).reshape(-1, 4)
    qs = [0.0, 0.0005, 0.01, 0.25, 0.5, 0.9, 0.9995, 1.0]
    h = disc.Float16Histogram(4).update(x)
    got = h.quantiles(qs)
    for b in range(4):
        col = flat[:, b][np.isfinite(flat[:, b])]
        np.testing.assert_array_equal(got[:, b], np.quantile(col, qs, method="inverted_cdf"))
    np.testing.assert_array_equal(h.n_finite(), np.isfinite(flat).sum(0))
    # chunks and classes merge exactly
    merged = disc.Float16Histogram(4).update(x[:150]).merge(disc.Float16Histogram(4).update(x[150:]))
    np.testing.assert_array_equal(merged.counts, h.counts)
    assert not h.rounded
    # another float dtype is rounded to float16, and says so
    r = disc.Float16Histogram(4).update(x.astype(np.float32))
    assert r.rounded and np.array_equal(r.counts, h.counts)
    assert np.isnan(disc.Float16Histogram(2).quantiles([0.5])).all()
    for n in (1, 7, 20_000):
        for q in (0.0, 0.0005, 0.5, 0.9995, 1.0):
            v = np.arange(n, dtype=np.float64)
            assert v[disc.inverted_cdf_rank(n, q)] == np.quantile(v, q, method="inverted_cdf")


def test_capture_runs_and_release_pages(tmp_path):
    starts, run_caps, uniq, n_runs = disc.capture_runs_summary(np.array([4, 4, 7, 7, 7, 4, 9]))
    assert starts.tolist() == [0, 2, 5, 6] and uniq[n_runs > 1].tolist() == [4]
    p = tmp_path / "a.npy"
    np.save(p, np.arange(300_000, dtype=np.float32).reshape(1000, 300))
    m = np.load(p, mmap_mode="r")
    total = float(m[:500].sum())
    assert disc.release_pages(m, 0, 500) is True                    # Linux: madvise(MADV_DONTNEED)
    assert float(m[:500].sum()) == total                            # pages re-fault from the file
    assert disc.release_pages(np.zeros((4, 4)), 0, 2) is False      # not a memmap
    assert disc.release_pages(m, 5, 5) is False


# ============================================================================
# CLI
# ============================================================================

def _cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, str(REPO_ROOT / "augment_pipeline.py"), *map(str, args)],
                          cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)


def test_cli_discovery_validate_group_and_failure_exit(tiny, tmp_path):
    r = _cli("discovery", "--help")
    assert r.returncode == 0
    for flag in ("--verify_sha256", "--content_scan", "--scan_max_rows", "--leakage_rows", "--percentiles"):
        assert flag in r.stdout

    out = tmp_path / "aug"
    r = _cli("discovery", "--data_dir", tiny.data_dir, "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS provenance.contiguous" in r.stdout and "discovery: ran" in r.stdout
    assert (out / STAGE_DIR / disc.INTEGRITY_REPORT).is_file()
    r = _cli("validate", "--data_dir", tiny.data_dir, "--out", out)          # the group: same stage, done
    assert r.returncode == 0 and "already done" in r.stdout

    data = _copy(tiny, tmp_path / "bad")
    _edit_npy(data / "y_test.npy", lambda a: a.__setitem__(0, (a[0] + 1) % 3))
    _remanifest(data)
    r = _cli("validate", "--data_dir", data, "--out", tmp_path / "bad_out")
    assert r.returncode == 1
    assert "FAILED" in r.stderr and "provenance.labels" in r.stderr
    assert not (tmp_path / "bad_out" / STAGE_DIR / DONE_NAME).exists()


# ============================================================================
# the real build (opt-in)
# ============================================================================

REAL_DATA = os.environ.get("HSI_AUGMENT_REAL_DATA", "")


@pytest.mark.skipif(not REAL_DATA or not (Path(REAL_DATA) / MANIFEST_NAME).is_file(),
                    reason="set HSI_AUGMENT_REAL_DATA to a real build's hsi dir to run")
def test_real_build_headers_json_and_sidecars(tmp_path):
    """No sha256 pass, no content scan, no hashing: JSON, headers and the
    y / groups / captures sidecars (~40 MB for train). X is never opened."""
    _run(Path(REAL_DATA), tmp_path, verify_sha256=False, content_scan=False, leakage_rows=0)
    report, integrity = _reports(tmp_path)
    assert integrity["failed"] == [], integrity["failed"]
    for name in ("provenance.contiguous", "provenance.counts", "provenance.labels", "provenance.patients",
                 "scope.pass1_train_patients", "wavelengths.selected_match", "splits.patients_disjoint"):
        assert integrity["checks"][name]["status"] == "pass", name
    train = report["splits"]["train"]
    assert train["n_captures"] == 499 and train["capture_sizes"] == {"4914": 499}
    assert train["grid_shapes"] == {"54x91": 499} and train["non_contiguous_captures"] == []
    wl = report["wavelengths"]
    assert wl["n"] == 32 and [g["after_band"] for g in wl["gaps"]] == [17]
    assert report["files"]["npy"]["hsi/X_train.npy"]["shape"][1:] == [11, 11, 32]
    rec = read_access_log(tmp_path / "split_access_log.jsonl")
    assert rec and all(r["purpose"] == "structural" and r["allowed"] for r in rec)
    assert not any(f.startswith("X_") for r in rec for f in r["files"])
