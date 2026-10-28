# -*- coding: utf-8 -*-
"""
tests/test_hsi_augment_export.py
================================
Generation, QC selection and the training export (#29; Plan §Phase 8a, the
selection of §8b, §Phase 9): the stages `generate-<src>`, `qc-<src>` and
`export-<src>` for both sources, on the tiny fixture, through the stage API.

The chain is real: tissue-mask -> leakage -> pool, the CVAE (cvae-lopo,
cvae-final) and the AE/LDM (ae-lopo, ae-final, ldm-lopo, ldm-final) fitted
with tiny models for 6 steps, then qc-reference. Checked:
  * exactly n_target rows per target class, where n_target is the TRAIN
    split's count of that class (40 DCIS rows on the fixture; the same rule
    gives 122,850 on trainsel, asserted by arithmetic on its class counts);
  * `training.synthetic_mix.load_synthetic_export` accepts each export
    (manifest_check strict); the manifest round-trips; tampering is refused;
  * the group ids are negative (-1 cvae, -2 ldm), the labels the target class,
    the values float16 VALUE space (the generator's decoded output, bit for bit);
  * the selection is the first n_target accepted samples in generation order,
    qc_scores.npz holds every generated sample;
  * a re-run reproduces every array's sha256 and the export_id;
  * tau_s / w: chosen on LOPO QC (every fold x candidate scored), or taken
    from the config without any LOPO sampling;
  * acceptance below 40% fails loudly (no done.json), in the LOPO choice and
    in the QC stage; 40-50% at 2x fails with the --overgenerate to use;
  * the stages read the train split only, and stay out of `all`.

Why the QC bitmask is overridden in most tests: generators trained for 6
steps on 184 tissue rows make flat patches, and the calibrated rules reject
all of them (R3; `test_real_qc_rejects_the_tiny_generators` shows it). To
test selection and export, `oracle_qc` wraps the REAL `qc.apply_rules` (every
score is still computed and stored) and replaces only its bitmask by a
deterministic function of each patch's bytes: a fixed fraction is accepted,
the rest keep (at least) one rule bit.

CPU only; every model and array is tiny.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from hsi_augment import generate as G
from hsi_augment.io import ACCESS_LOG_NAME, TRAIN_ONLY_PURPOSES, read_access_log
from hsi_augment.qc import RULE_BITS
from hsi_augment.qc import features as QF
from hsi_augment.registry import load_builtin_stages
from hsi_augment.splits import held_out_readers, registry_policy_violations
from hsi_augment.stage import DONE_NAME, FAILED_NAME, RunOptions, StageError, StageNotReady, run_stage
from hsi_augment.stages.export import ExportConfig
from hsi_augment.stages.fit_ae import AEFinalConfig, AELopoConfig
from hsi_augment.stages.fit_cvae import CVAEFinalConfig, CVAELopoConfig
from hsi_augment.stages.fit_ldm import LDMFinalConfig, LDMLopoConfig
from hsi_augment.stages.generate import GenerateCVAEConfig, GenerateLDMConfig, QCApplyConfig
from hsi_augment.stages.leakage import LeakageConfig
from hsi_augment.stages.pool import PoolConfig
from hsi_augment.stages.qc_reference import QCReferenceConfig
from hsi_augment.stages.tissue_mask import TissueMaskConfig
from tests.hsi_augment_fixtures import make_tiny_build
from training.npy_atomic import MANIFEST_NAME, verify_against_manifest
from training.synthetic_mix import (EXPORT_ARRAYS, GEN_INDEX_NAME, GROUPS_SYN_NAME, PROVENANCE_KEYS,
                                    PROVENANCE_NAME, QC_REPORT_NAME, QC_SCORES_NAME, SYNTHETIC_GROUP_IDS,
                                    SyntheticExportError, X_SYN_NAME, Y_SYN_NAME, load_synthetic_export)

DCIS = 1
SOURCES = ("cvae", "ldm")
REAL_TRAIN_COUNTS = (722_358, 122_850, 1_606_878)      # trainsel train: healthy, DCIS, IDC
ACCEPT = 0.75                                           # the oracle's acceptance in the happy path

LOOP = dict(max_steps=6, warmup_steps=1, eval_every=3, log_every=2, ckpt_every=2, ema_decay=0.9, batch_size=12,
            eval_batch_size=16, device="cpu")
CVAE_LOPO = dict(latent_dims=(4,), sam_ratios=(0.1,), early_stop_patience=None, diag_train_rows=40, prior_samples=16,
                 widths=(8, 16, 16), cond_dim=8)
AE_LOPO = dict(widths=(8, 16), blocks_per_level=1, latent_channels=2, metric_rows=20)
LDM_LOPO = dict(channels=(16, 32), blocks_per_level=1, mid_blocks=1, emb_dim=16, freq_dim=8, timesteps=50,
                sample_steps=5, guidance=(1.0, 2.0), encode_batch_size=64, sample_batch_size=16, mmd_rows=16,
                precision="fp32")
GEN = dict(device="cpu", choice_rows=24, batch_size=16, qc_chunk_rows=32)
GEN_CONFIGS = {"cvae": GenerateCVAEConfig, "ldm": GenerateLDMConfig}
MY_STAGES = tuple(f"{k}-{s}" for k in ("generate", "qc", "export") for s in SOURCES)


# ============================================================================
# helpers
# ============================================================================

def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def opts(tiny, out: Path, **kw) -> RunOptions:
    return RunOptions(data_dir=str(tiny.data_dir), out=str(out), **kw)


def run(registry, tiny, out: Path, name: str, cfg, **kw):
    return run_stage(registry.get(name), cfg, opts(tiny, out, **kw), registry)


def oracle_accepts(x: np.ndarray, frac: float) -> np.ndarray:
    """A deterministic ~frac of the patches, by their float16 bytes."""
    h = QF.patch_hashes(x)[:, 0]
    return (h % np.uint64(1000)) < np.uint64(int(round(1000 * frac)))


@contextlib.contextmanager
def oracle_qc(frac: float):
    """`qc.apply_rules` with its scores intact and its bitmask replaced (module docstring)."""
    import hsi_augment.qc as Q
    real = Q.apply_rules

    def apply(ref, x, class_id, *, fold=None, device=None, chunk_rows=None):
        mask, scores = real(ref, x, class_id, fold=fold, device=device, chunk_rows=chunk_rows)
        keep = oracle_accepts(x, frac)
        return np.where(keep, 0, np.where(mask == 0, RULE_BITS["R2"], mask)).astype(np.uint8), scores

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Q, "apply_rules", apply)
        yield


def run_source(registry, tiny, out: Path, source: str, *, gen_kw=None, **kw) -> Dict[str, object]:
    return {"generate": run(registry, tiny, out, f"generate-{source}", GEN_CONFIGS[source](**{**GEN, **(gen_kw or {})}),
                            **kw),
            "qc": run(registry, tiny, out, f"qc-{source}", QCApplyConfig(device="cpu", chunk_rows=32, bootstrap=5),
                      **kw),
            "export": run(registry, tiny, out, f"export-{source}", ExportConfig(figure_samples=6), **kw)}


# ============================================================================
# fixtures
# ============================================================================

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def registry():
    return load_builtin_stages()


@pytest.fixture(scope="module")
def chain(tiny, registry, tmp_path_factory) -> Path:
    """<out> with every upstream of the generate stages done."""
    out = tmp_path_factory.mktemp("aug")
    for name, cfg in (("tissue-mask", TissueMaskConfig()), ("leakage", LeakageConfig()),
                      ("pool", PoolConfig(rows_per_class=30)),
                      ("cvae-lopo", CVAELopoConfig(**LOOP, **CVAE_LOPO)),
                      ("cvae-final", CVAEFinalConfig(enforce_gates=False, device="cpu", prior_samples=16)),
                      ("ae-lopo", AELopoConfig(**LOOP, **AE_LOPO)),
                      ("ae-final", AEFinalConfig(device="cpu", metric_rows=20)),
                      ("ldm-lopo", LDMLopoConfig(**LOOP, **LDM_LOPO)), ("ldm-final", LDMFinalConfig(device="cpu")),
                      ("qc-reference", QCReferenceConfig(device="cpu"))):
        assert run(registry, tiny, out, name, cfg).status == "ran", name
    return out


@dataclass
class Ran:
    out: Path
    results: Dict[str, Dict[str, object]]

    def synth(self, src: str) -> Path:
        return self.out / "synth" / src

    def qc(self, src: str) -> Path:
        return self.out / "qc" / src

    def export(self, src: str) -> Path:
        return self.out / "export" / src


@pytest.fixture(scope="module")
def ran(tiny, registry, chain) -> Ran:
    """generate -> qc -> export for both sources, tau_s / w chosen on LOPO QC."""
    with oracle_qc(ACCEPT):
        results = {src: run_source(registry, tiny, chain, src) for src in SOURCES}
    return Ran(out=chain, results=results)


@pytest.fixture
def chain_copy(chain, ran, tmp_path) -> Path:
    """A private copy of the module's <out>, for runs that change or break it."""
    dst = tmp_path / "aug"
    shutil.copytree(chain, dst)
    return dst


def train_dcis(tiny) -> int:
    return int((np.load(tiny.data_dir / "y_train.npy") == DCIS).sum())


# ============================================================================
# registration, the target rule, the selection rule, the choice rule
# ============================================================================

def test_stages_are_registered_train_only_and_out_of_all(registry):
    from hsi_augment import cli
    want = {"generate-cvae": ("generate", "synth/cvae", {"cvae-lopo", "cvae-final", "qc-reference"}),
            "generate-ldm": ("generate", "synth/ldm", {"ldm-lopo", "ldm-final", "qc-reference"}),
            "qc-cvae": ("qc", "qc/cvae", {"generate-cvae", "qc-reference"}),
            "qc-ldm": ("qc", "qc/ldm", {"generate-ldm", "qc-reference"}),
            "export-cvae": ("export", "export/cvae", {"generate-cvae", "qc-cvae", "qc-reference"}),
            "export-ldm": ("export", "export/ldm", {"generate-ldm", "qc-ldm", "qc-reference"})}
    for name, (group, dirname, requires) in want.items():
        s = registry.get(name)
        assert (s.group, s.dirname, set(s.requires)) == (group, dirname, requires)
        assert set(s.purposes) <= set(TRAIN_ONLY_PURPOSES) and not s.reads_held_out() and not s.in_all
    assert {s.name for s in registry.resolve(["generate"])} == {"generate-cvae", "generate-ldm"}
    assert {s.name for s in registry.resolve(["export"])} == {"export-cvae", "export-ldm"}
    assert {"qc-reference", "qc-cvae", "qc-ldm"} <= {s.name for s in registry.resolve(["qc"])}
    assert not set(MY_STAGES) & {s.name for s in registry.resolve(["all"])}
    assert not set(MY_STAGES) & set(held_out_readers(registry)) and registry_policy_violations(registry) == []
    assert {"generate", "qc", "export"} <= set(registry.command_names())
    names = [s.name for s in registry.resolve(["export-cvae", "qc-cvae", "generate-cvae"])]
    assert names == ["generate-cvae", "qc-cvae", "export-cvae"]
    helptext = cli.build_stage_parser(registry, ["generate-cvae"]).format_help()
    assert "--tau_s" in helptext and "--overgenerate" in helptext and "--n_per_class" in helptext
    assert "--guidance" in cli.build_stage_parser(registry, ["generate-ldm"]).format_help()


def test_the_target_is_the_train_count_of_each_minority_class(tiny):
    y = np.load(tiny.data_dir / "y_train.npy")
    names = list(tiny.class_names)
    assert G.target_counts(y, 3, names) == {DCIS: train_dcis(tiny)} == {DCIS: 40}
    assert G.generate_counts({DCIS: 40}, 2.0) == {DCIS: 80}
    # the same rule on the real trainsel train counts: 122,850 DCIS, 245,700 generated at 2x
    y_real = np.repeat(np.arange(3), REAL_TRAIN_COUNTS)
    assert G.target_counts(y_real, 3, names) == {DCIS: 122_850}
    assert G.generate_counts(G.target_counts(y_real, 3, names), 2.0) == {DCIS: 245_700}
    assert G.target_counts(y_real, 3, names, "DCIS,IDC", n_per_class=7) == {1: 7, 2: 7}
    with pytest.raises(StageError, match="unknown class"):
        G.target_counts(y, 3, names, "nope")


REAL_DATA = os.environ.get("HSI_AUGMENT_REAL_DATA")


@pytest.mark.skipif(not REAL_DATA, reason="set HSI_AUGMENT_REAL_DATA to the trainsel hsi dir to run")
def test_real_build_target_is_122850_dcis_rows():
    """The rule on the real build's train labels (the int64 sidecar only, never X)."""
    from hsi_augment.adapters import get_adapter
    real = get_adapter("histologyhsi_bc", REAL_DATA)
    y = np.load(Path(REAL_DATA) / "y_train.npy", mmap_mode="r")
    targets = G.target_counts(np.asarray(y), real.num_classes, real.class_names)
    assert targets == {DCIS: 122_850} and G.generate_counts(targets, 2.0) == {DCIS: 245_700}


def test_selection_is_the_first_accepted_in_generation_order():
    rng = np.random.default_rng(0)
    mask = np.where(rng.random(200) < 0.7, 0, RULE_BITS["R3"]).astype(np.uint8)
    sel, rec = G.select_accepted(mask, 50, min_acceptance=0.4, overgenerate=2.0)
    assert np.array_equal(sel, np.flatnonzero(mask == 0)[:50])
    assert rec["n_selected"] == 50 and rec["n_generated"] == 200 and rec["last_selected_position"] == sel[-1]
    # below the 40% floor: loud, whatever the target
    low = np.where(np.arange(100) % 10 < 3, 0, RULE_BITS["R1"]).astype(np.uint8)        # 30%
    with pytest.raises(G.AcceptanceError, match="below the 40% floor"):
        G.select_accepted(low, 10, min_acceptance=0.4, overgenerate=2.0)
    # 45% at 2x cannot fill the target: loud too, naming the overgeneration that would
    mid = np.where(np.arange(100) % 20 < 9, 0, RULE_BITS["R2"]).astype(np.uint8)        # 45%
    with pytest.raises(G.AcceptanceError, match="--overgenerate 2.34"):
        G.select_accepted(mid, 50, min_acceptance=0.4, overgenerate=2.0)
    assert len(G.select_accepted(mid, 45, min_acceptance=0.4, overgenerate=2.0)[0]) == 45


def test_the_choice_rule():
    def rec(v, acc, cov):
        return {"value": v, "acceptance": acc, "coverage_pca": cov, "rejected_by": {"R6": 0.0}}
    # 0.8 covers most but is below the floor; 1.0 wins on coverage among the eligible
    recs = [rec(0.8, 0.3, 0.9), rec(0.8, 0.35, 0.9), rec(1.0, 0.6, 0.5), rec(1.0, 0.5, 0.7), rec(1.2, 0.9, 0.4),
            rec(1.2, 0.9, None)]
    table, chosen = G.rank_candidates(recs, (0.8, 1.0, 1.2), 0.4)
    assert chosen == 1.0 and [t["eligible"] for t in table] == [False, True, True]
    assert table[2]["score"] == pytest.approx(0.2) and table[2]["coverage_pca_missing"] == 1
    # equal coverage: the higher acceptance; then the smaller value
    assert G.rank_candidates([rec(1.0, 0.5, 0.6), rec(2.0, 0.7, 0.6)], (1.0, 2.0), 0.4)[1] == 2.0
    assert G.rank_candidates([rec(2.0, 0.7, 0.6), rec(1.5, 0.7, 0.6)], (1.5, 2.0), 0.4)[1] == 1.5
    assert G.rank_candidates([rec(1.0, 0.1, 0.9)], (1.0,), 0.4)[1] is None


# ============================================================================
# the exports
# ============================================================================

@pytest.mark.parametrize("source", SOURCES)
def test_export_has_exactly_the_target_rows_and_loads(ran, tiny, source):
    d = ran.export(source)
    assert all(r.status == "ran" for r in ran.results[source].values())
    exp = load_synthetic_export(str(d), str(tiny.data_dir), num_classes=3, manifest_check="strict")
    n = train_dcis(tiny)
    assert exp.n_syn == n == 40 and exp.hwc == (tiny.patch, tiny.patch, tiny.n_bands)
    assert np.all(exp.y == DCIS)
    assert exp.groups.dtype == np.int32 and np.all(exp.groups == SYNTHETIC_GROUP_IDS[source]) and np.all(exp.groups < 0)
    prov = json.loads((d / PROVENANCE_NAME).read_text())
    assert exp.export_id == prov["export_id"] == sha(d / X_SYN_NAME)[:12]
    assert prov["class_counts"] == {"healthy": 0, "DCIS": n, "IDC": 0}
    for key in (*PROVENANCE_KEYS, "generator_ckpt_sha256", "config_hash", "git_sha", "seeds", "n_generated",
                "n_accepted", "thresholds", "stats_sha256", G.PARAM_NAMES[source]):
        assert key in prov, key
    assert prov["n_generated"] == 2 * n and prov["n_target"] == {"DCIS": n}

    # the selection: the first n accepted samples in generation order; scores for every generated sample
    qc = np.load(d / QC_SCORES_NAME)
    assert len(qc["bitmask"]) == 2 * n and np.array_equal(qc["gen_index"], np.arange(2 * n))
    gen_index = np.load(d / GEN_INDEX_NAME)
    assert np.array_equal(gen_index, np.flatnonzero(qc["bitmask"] == 0)[:n])
    assert np.array_equal(np.flatnonzero(qc["selected"]), gen_index)
    assert prov["n_accepted"] == int((qc["bitmask"] == 0).sum())
    assert {"nn_dist", "mahalanobis", "p_class", "fail.R6.nn_dist"} <= set(qc.files)

    # VALUE space: the generator's decoded float16 output, row for row
    X = np.load(d / X_SYN_NAME)
    X_gen = np.load(ran.synth(source) / G.X_GEN_NAME)
    assert X.dtype == np.float16 and np.isfinite(X).all() and X.min() >= 0
    assert np.array_equal(X.view(np.uint16), X_gen[gen_index].view(np.uint16))


def test_the_export_is_the_generators_decoded_output(ran, chain):
    """X_syn row i is CVAE sample stream row gen_index[i] at the chosen tau_s,
    decoded by the codec (value space, clipped to [0, clip_hi])."""
    from hsi_augment.models.cvae import load_cvae, sample
    model, codec, _ = load_cvae(chain / "gen" / "cvae" / "final", device="cpu")
    gen = json.loads((ran.synth("cvae") / G.GENERATION_NAME).read_text())
    X = np.load(ran.export("cvae") / X_SYN_NAME)
    idx = np.load(ran.export("cvae") / GEN_INDEX_NAME)
    seed = gen["targets"][str(DCIS)]["seed"]
    again = np.stack([sample(model, codec, DCIS, 1, gen["value"], seed, int(r))[0] for r in idx[:5]])
    assert np.array_equal(again.view(np.uint16), X[:5].view(np.uint16))
    assert np.all(X <= codec.clip_hi.astype(np.float32).astype(np.float16)[None, None, None])


def test_manifest_round_trip_and_tampering(ran, tiny, tmp_path):
    d = ran.export("cvae")
    assert verify_against_manifest(str(d)) == []
    files = json.loads((d / MANIFEST_NAME).read_text())["files"]
    assert set(EXPORT_ARRAYS) | {"wavelengths.npy"} <= set(files)
    for name in EXPORT_ARRAYS:
        assert files[name]["sha256"] == sha(d / name)
    assert np.array_equal(np.load(d / "wavelengths.npy"), np.load(tiny.data_dir / "wavelengths.npy"))
    assert (d / "class_names.json").read_bytes() == (tiny.data_dir / "class_names.json").read_bytes()
    # the stage's done.json records the same sha256s
    done = json.loads((d / DONE_NAME).read_text())
    assert done["outputs"][X_SYN_NAME]["sha256"] == files[X_SYN_NAME]["sha256"]
    assert done["summary"]["export_id"] == files[X_SYN_NAME]["sha256"][:12]

    bad = tmp_path / "bad"
    shutil.copytree(d, bad)
    y = np.load(bad / Y_SYN_NAME)
    y[0] = 2
    np.save(bad / Y_SYN_NAME, y)
    with pytest.raises(SyntheticExportError, match="manifest"):
        load_synthetic_export(str(bad), str(tiny.data_dir), num_classes=3)
    other = tmp_path / "other_build"
    shutil.copytree(tiny.data_dir, other)
    (other / MANIFEST_NAME).write_text((tiny.data_dir / MANIFEST_NAME).read_text().replace("fixture", "fixtur3"))
    with pytest.raises(SyntheticExportError, match="different build"):
        load_synthetic_export(str(d), str(other), num_classes=3)


def test_qc_report_and_figures_live_in_the_export(ran):
    for src in SOURCES:
        d = ran.export(src)
        rep = json.loads((d / QC_REPORT_NAME).read_text())
        dc = rep["classes"]["DCIS"]
        assert dc["n_generated"] == 80 and dc["n_selected"] == 40 and dc["acceptance"] >= 0.4
        assert set(dc["rejected_by"]) == set(RULE_BITS) and "R6.nn_dist" in dc["thresholds"]
        assert dc["set_metrics"]["n_synthetic"] == 40 and "nn_attribution" in dc["set_metrics"]
        assert rep["choice"]["source"] == "lopo_qc" and rep[G.PARAM_NAMES[src]] == rep["choice"]["value"]
        assert (d / QC_SCORES_NAME).read_bytes() == (ran.qc(src) / QC_SCORES_NAME).read_bytes()
        for fig in ("spectra_dcis.png", "contact_sheet_dcis.png", "nn_gallery_dcis.png"):
            assert (d / "figures" / fig).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert not list(ran.synth(src).rglob("*.png")) and not list(ran.qc(src).rglob("*.png"))


def test_report_finds_the_export_id(ran):
    """hsi_augment/report.py reads export/<src>/provenance.json's export_id."""
    for src in SOURCES:
        prov = json.loads((ran.out / "export" / src / "provenance.json").read_text())
        done = json.loads((ran.export(src) / DONE_NAME).read_text())
        assert prov["export_id"] == done["summary"]["export_id"] and done["summary"]["n_accepted"] > 0


# ============================================================================
# tau_s / w
# ============================================================================

@pytest.mark.parametrize("source", SOURCES)
def test_tau_s_and_w_are_chosen_on_lopo_qc(ran, chain, source):
    choice = json.loads((ran.synth(source) / G.CHOICE_NAME).read_text())
    cands = (0.8, 1.0) if source == "cvae" else (1.0, 2.0)
    assert choice["source"] == "lopo_qc" and tuple(choice["candidates"]) == cands and choice["value"] in cands
    folds = json.loads((chain / "prep" / "leakage" / "folds.json").read_text())["folds"]
    assert len(folds) == 2
    assert {(r["fold"], r["value"]) for r in choice["records"]} == {(f["fold"], v) for f in folds for v in cands}
    for r in choice["records"]:
        assert r["n"] == GEN["choice_rows"] and set(r["rejected_by"]) == set(RULE_BITS)
        assert r["coverage_pca"] is not None and 0 <= r["coverage_pca"] <= 1 and r["n_heldout_real"] > 5
    assert G.rank_candidates(choice["records"], cands, 0.4)[1] == choice["value"]
    gen = json.loads((ran.synth(source) / G.GENERATION_NAME).read_text())
    assert gen["value"] == choice["value"] and gen["param"] == G.PARAM_NAMES[source]
    meta = np.load(ran.synth(source) / G.GEN_META_NAME)
    ckpt = chain / gen["generator"]["checkpoint"]
    assert str(meta["checkpoint_sha256"]) == sha(ckpt) == gen["generator"]["checkpoint_sha256"]
    assert str(meta["source"]) == source and str(meta["param_name"]) == G.PARAM_NAMES[source]
    assert np.all(meta["param"] == choice["value"]) and np.all(meta["seed"] == gen["targets"][str(DCIS)]["seed"])
    assert np.array_equal(meta["gen_index"], np.arange(80)) and np.array_equal(meta["stream_row"], np.arange(80))


def test_a_configured_value_skips_the_lopo_choice(registry, tiny, chain_copy):
    metrics = chain_copy / "synth" / "cvae" / "metrics.jsonl"          # appended to by every run

    def choice_events():
        return sum(json.loads(line).get("event") == "lopo_choice" for line in metrics.read_text().splitlines())

    seen = choice_events()
    assert seen == 2 * 2                                                 # the module run: 2 folds x 2 tau_s
    with oracle_qc(ACCEPT):
        r = run(registry, tiny, chain_copy, "generate-cvae", GenerateCVAEConfig(**GEN, tau_s=1.0))
    assert r.status == "ran" and choice_events() == seen
    choice = json.loads((chain_copy / "synth" / "cvae" / G.CHOICE_NAME).read_text())
    assert choice == {"param": "tau_s", "candidates": [0.8, 1.0], "source": "config", "value": 1.0}
    assert r.done["summary"]["tau_s"] == 1.0 and r.done["summary"]["choice"] == "config"


# ============================================================================
# failures
# ============================================================================

def test_acceptance_below_40_percent_fails_loudly(registry, tiny, chain_copy):
    out = chain_copy
    with oracle_qc(0.25):
        with pytest.raises(StageError, match="no tau_s candidate reaches a mean acceptance of 40%"):
            run(registry, tiny, out, "generate-cvae", GenerateCVAEConfig(**GEN), force=True)
        assert not (out / "synth" / "cvae" / DONE_NAME).exists()
        assert run(registry, tiny, out, "generate-cvae", GenerateCVAEConfig(**GEN, tau_s=1.0)).status == "ran"
        with pytest.raises(G.AcceptanceError, match="below the 40% floor"):
            run(registry, tiny, out, "qc-cvae", QCApplyConfig(device="cpu", chunk_rows=32, bootstrap=5))
    assert not (out / "qc" / "cvae" / DONE_NAME).exists() and (out / "qc" / "cvae" / FAILED_NAME).exists()
    failed = np.load(out / "qc" / "cvae" / QC_SCORES_NAME)          # kept for the diagnosis
    assert len(failed["bitmask"]) == 80 and 0 < (failed["bitmask"] == 0).mean() < 0.4 and not failed["selected"].any()
    with pytest.raises(StageNotReady, match="qc-cvae"):
        run(registry, tiny, out, "export-cvae", ExportConfig())
    # above the floor but short of the target (1x over-generation): loud, with the factor to use
    with oracle_qc(ACCEPT):
        assert run(registry, tiny, out, "generate-cvae",
                   GenerateCVAEConfig(**GEN, tau_s=1.0, overgenerate=1.0)).status == "ran"
        with pytest.raises(G.AcceptanceError, match="Re-run the generate stage with --overgenerate"):
            run(registry, tiny, out, "qc-cvae", QCApplyConfig(device="cpu", chunk_rows=32, bootstrap=5))


def test_real_qc_rejects_the_tiny_generators(registry, tiny, chain_copy):
    """No oracle: the calibrated rules reject the 6-step generator's flat
    patches, so the stage fails instead of exporting them."""
    assert run(registry, tiny, chain_copy, "generate-cvae", GenerateCVAEConfig(**GEN, tau_s=1.0)).status == "ran"
    with pytest.raises(G.AcceptanceError, match="below the 40% floor"):
        run(registry, tiny, chain_copy, "qc-cvae", QCApplyConfig(device="cpu", chunk_rows=32, bootstrap=5))


# ============================================================================
# split access and determinism
# ============================================================================

def test_generation_qc_and_export_read_train_only(ran):
    rec = [r for r in read_access_log(ran.out / ACCESS_LOG_NAME) if r["stage"] in (*MY_STAGES, "qc-apply")]
    assert {r["stage"] for r in rec} == set(MY_STAGES)
    assert all(r["allowed"] and r["split"] == "train" for r in rec)
    assert {(r["stage"].split("-")[0], r["purpose"]) for r in rec} == \
        {("generate", "generator"), ("generate", "qc"), ("qc", "qc"), ("export", "qc")}


def test_a_rerun_reproduces_every_sha256(ran, registry, tiny):
    files = {src: [ran.synth(src) / G.X_GEN_NAME, ran.qc(src) / G.SELECTED_NAME,
                   *(ran.export(src) / n for n in (X_SYN_NAME, Y_SYN_NAME, GROUPS_SYN_NAME, GEN_INDEX_NAME))]
             for src in SOURCES}
    before = {p: sha(p) for ps in files.values() for p in ps}
    bitmask = {src: np.load(ran.qc(src) / QC_SCORES_NAME)["bitmask"] for src in SOURCES}
    ids = {src: json.loads((ran.export(src) / PROVENANCE_NAME).read_text())["export_id"] for src in SOURCES}
    with oracle_qc(ACCEPT):
        for src in SOURCES:
            assert all(r.status == "ran" for r in run_source(registry, tiny, ran.out, src, force=True).values())
    assert {p: sha(p) for p in before} == before
    for src in SOURCES:
        assert np.array_equal(np.load(ran.qc(src) / QC_SCORES_NAME)["bitmask"], bitmask[src])
        assert json.loads((ran.export(src) / PROVENANCE_NAME).read_text())["export_id"] == ids[src]
    # and an unchanged chain is skipped
    with oracle_qc(ACCEPT):
        assert all(r.status == "skipped" for r in run_source(registry, tiny, ran.out, "cvae").values())
