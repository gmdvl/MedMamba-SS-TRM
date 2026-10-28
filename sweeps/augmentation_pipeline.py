# -*- coding: utf-8 -*-
"""
sweeps/augmentation_pipeline.py - the HSI augmentation study end to end
(plan/HSI_augmentation_pipeline.plan.md, Phase 10; issue #33): the pipeline's
stages (`augment_pipeline.py`), then the classifier arms on the trainsel build.

    python run_experiments.py sweeps/augmentation_pipeline.py --status
    python run_experiments.py sweeps/augmentation_pipeline.py prep            # P1-P4; pool waits for AUG_TAU
    AUG_TAU=<deg> python run_experiments.py sweeps/augmentation_pipeline.py prep
    python run_experiments.py sweeps/augmentation_pipeline.py pipeline        # gen-cvae ... lopo-utility
    python run_experiments.py sweeps/augmentation_pipeline.py bench screen select
    python run_experiments.py sweeps/augmentation_pipeline.py seeds report    # a NEW invocation after select
    AUG_PHYS=1 python run_experiments.py sweeps/augmentation_pipeline.py ...  # + the optional arm A1p

Stages, in order: prep -> gen-cvae -> gen-ldm -> export -> lopo-utility ->
bench -> screen -> select -> seeds -> report. A job that cannot run yet is
skipped with the reason (`--status` lists it as "wait").

Arms (Plan §Phase 10). REF = the paper's reference recipe: paper_recipe,
batch 256, --lambda_sam 0.1, --val_subsample_frac 0.0986, effective class
weights.
    A0    v20-abl-base-s42 / v20-head-s*   real rows only, frac 0.102       shared: paper sweep
    A0os  minaug-nomix                      DCIS copy x2, no mixing          shared: minority sweep
    A1    minaug-default                    DCIS copy x2, stats mixing 0.5   shared: minority sweep
    A2    aug-cvae                          +122,850 CVAE DCIS rows
    A3    aug-ldm                           +122,850 LDM DCIS rows
    A4    aug-combo                         +122,850 rows of the LOPO-chosen generator, factor 1.0,
                                            stats mixing 0.5
    A1p   aug-conv-phys (AUG_PHYS=1)        A1 + --phys_aug minority (optional, ~2 GPU-h)
Seeds-stage tags are `<screen tag>-s<seed>` (A0: the paper's `v20-head-s<seed>`),
bench tags `aug-bench-<arm>`: what hsi_augment/report.py looks for.

Every arm but A0 trains on N_TRAIN + 122,850 = 2,574,936 rows at frac
0.0971337: 250,113 rows per epoch, as A0 at 0.102 - the same optimizer steps.
No arm pins `--trm_ema_rate` or `--minority_classes`: the shared tags must be
the other sweeps' commands byte for byte (a job is done once a run with its
tag exists), `auto` resolves to DCIS, and the parse-time EMA rate (931 vs 978
steps per epoch x 3 horizon epochs) clamps to 0.9995 in every arm
(tests/test_augmentation_sweep.py checks both).

Decisions read from files when the sweep LOADS, so every argv is static (re-run
the command after one of them changes; a job built before it waits):
    <AUG_OUT>/prep/tissue/glass_rule.json   the pre-registered glass rule (tissue-mask stage): if
                                            it says tissue-only copies, A0os/A1 become aug-copy /
                                            aug-copymix with --minority_copy_mask dcis_copy_mask.npy
                                            (and stop sharing tags with the minority sweep)
    <AUG_OUT>/lopo_utility/decision.json    A4's generator ("a4_source" | "source" | "generator" |
                                            "chosen", as hsi_augment/report.py reads it)
    reports/augmentation/selection.json     scripts/augmentation_select.py; `seeds` is [] until it
                                            exists
A synthetic arm (A2/A3/A4) is done only when its newest finished run trained on
the CURRENT export: `config.json["synthetic"]["export_id"]` equals
`export/<source>/provenance.json`'s, so regenerating an export re-runs its arms.

`bench` is a group of one-job stages, one per arm: each one-epoch run is alone
on the GPU even under --jobs N (its s/epoch is the cost table's only classifier
timing). `select` never reads test: scripts/augmentation_select.py reads each
screen run's config.json and history.json only.

Environment: PY, AUG_OUT (default data/hsi_v9-trainsel-aug), AUG_SELECTION
(default reports/augmentation/selection.json), AUG_TAU (the tau in degrees the
user signed off, Plan §Phase 2 item 6; the pool waits for it), AUG_PHYS=1.
"""

import json
import os
from pathlib import Path

from run_experiments import PY, ROOT, cmd, run_dir, train

HSI = "data/hsi_v9-trainsel/hsi"
OUT = os.environ.get("AUG_OUT", "data/hsi_v9-trainsel-aug")
SELECTION = os.environ.get("AUG_SELECTION", "reports/augmentation/selection.json")
TAU = os.environ.get("AUG_TAU")
PHYS = os.environ.get("AUG_PHYS") == "1"

REF = dict(profile="paper_recipe", batch_size=256, lambda_sam=0.1,
           train_subsample_frac=0.102, val_subsample_frac=0.0986)
SEEDS = (1, 7, 13, 23, 42)
SCREEN_SEED, SCREEN_EPOCHS, SEED_EPOCHS, BENCH_EPOCHS = 42, 12, 20, 1

N_TRAIN, N_DCIS = 2452086, 122850                    # y_train.npy of the trainsel build
N_SYN = N_DCIS                                       # Phase 8a: exactly 122,850 accepted rows per source
KEPT = round(N_TRAIN * REF["train_subsample_frac"])  # 250,113 rows per epoch

GENERATORS = ("cvae", "ldm")
GLASS_RULE = f"{OUT}/prep/tissue/glass_rule.json"
COPY_MASK = f"{OUT}/prep/tissue/dcis_copy_mask.npy"
DECISION = f"{OUT}/lopo_utility/decision.json"
DECISION_KEYS = ("a4_source", "source", "generator", "chosen")
PHYS_FILES = (f"{OUT}/prep/augcal/phys_calibration.json", f"{OUT}/prep/spectral/noise_sigma.npy")


# ============================================================================
# recorded decisions (read at load; re-read by `requires` to catch a change)
# ============================================================================

def _here(path) -> Path:
    """`path` as the jobs see it: they run in the repo root."""
    return ROOT / path


def _json(path):
    try:
        with open(_here(path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def glass_rule() -> dict:
    """The tissue-mask stage's recorded glass rule. `use_copy_mask` None: not
    measured, or not usable (unfinished stage, a run over part of the captures,
    a malformed file); `why` says which."""
    done, rule = _json(f"{OUT}/prep/tissue/done.json"), _json(GLASS_RULE)
    if not isinstance(done, dict) or done.get("status") != "ok" or not isinstance(rule, dict):
        return {"use_copy_mask": None, "why": f"no glass rule yet ({GLASS_RULE}): run the prep stage"}
    if rule.get("complete") is False:
        return {"use_copy_mask": None, "why": f"{GLASS_RULE} covers only part of the train captures"}
    if not isinstance(rule.get("use_copy_mask"), bool):
        return {"use_copy_mask": None, "why": f"{GLASS_RULE} has no boolean use_copy_mask"}
    frac = (rule.get("overall") or {}).get("glass_fraction")
    return {"use_copy_mask": rule["use_copy_mask"],
            "why": f"DCIS glass fraction {frac} vs threshold {rule.get('threshold')}"}


def a4_source():
    """The generator lopo_utility/decision.json names, or None."""
    doc = _json(DECISION)
    src = next((doc[k] for k in DECISION_KEYS if isinstance(doc.get(k), str)), None) \
        if isinstance(doc, dict) else None
    return src if src in GENERATORS else None


def export_id(source):
    """`export_id` of export/<source>/provenance.json (#29), or None."""
    prov = _json(f"{OUT}/export/{source}/provenance.json")
    return prov.get("export_id") if isinstance(prov, dict) else None


GLASS = glass_rule()
MASKED = GLASS["use_copy_mask"] is True
A4_SOURCE = a4_source()


# ============================================================================
# arms
# ============================================================================

def matched(n_extra: int, **flags) -> dict:
    """REF + `flags`, `--train_subsample_frac` keeping KEPT rows per epoch over
    N_TRAIN + n_extra rows (sweeps/minority_augmentation.py's `matched`,
    counted in extra rows rather than by factor)."""
    frac = REF["train_subsample_frac"] if n_extra == 0 else round(KEPT / (N_TRAIN + n_extra), 7)
    return {**REF, "train_subsample_frac": frac, **flags}


def arm(key, family, tag, flags, description, seed_tag=None, generator=None, copies=False, phys=False):
    """One arm: hsi_augment/report.py's arm-table fields + what the jobs need."""
    return {"key": key, "family": family, "screen_tag": tag, "seed_tag": seed_tag or f"{tag}-s{{seed}}",
            "bench_tag": f"aug-bench-{key.lower()}", "generator": generator, "description": description,
            "flags": flags, "copies": copies, "phys": phys}


COPY = dict(minority_augment="on", minority_oversample_factor=2.0)
MASK = {"minority_copy_mask": COPY_MASK} if MASKED else {}
ONLY = " (tissue rows only)" if MASKED else ""
A1_FLAGS = matched(N_DCIS, **COPY, **MASK)

ARMS = {a["key"]: a for a in (
    arm("A0", "control", "v20-abl-base-s42", dict(REF), "real only (REF, frac 0.102)",
        seed_tag="v20-head-s{seed}"),
    arm("A0os", "non-generative", "aug-copy" if MASKED else "minaug-nomix",
        matched(N_DCIS, **COPY, minority_mix_prob=0.0, **MASK), f"DCIS copy x2{ONLY}, no mixing", copies=True),
    arm("A1", "non-generative", "aug-copymix" if MASKED else "minaug-default", A1_FLAGS,
        f"DCIS copy x2{ONLY}, stats mixing 0.5", copies=True),
    arm("A2", "generative", "aug-cvae", matched(N_SYN, synthetic_dir=f"{OUT}/export/cvae"),
        "+122,850 CVAE DCIS rows", generator="cvae"),
    arm("A3", "generative", "aug-ldm", matched(N_SYN, synthetic_dir=f"{OUT}/export/ldm"),
        "+122,850 LDM DCIS rows", generator="ldm"),
    arm("A4", "generative", "aug-combo",
        matched(N_SYN, minority_augment="on", minority_oversample_factor=1.0,
                synthetic_dir=f"{OUT}/export/{A4_SOURCE or 'undecided'}"),
        "+122,850 rows from the LOPO-chosen generator, factor 1.0, stats mixing 0.5", generator="decision"),
    *([arm("A1p", "non-generative", "aug-conv-phys",
           {**A1_FLAGS, "phys_aug": "minority", "phys_artifacts": OUT},
           "A1 + --phys_aug minority (optional)", copies=True, phys=True)] if PHYS else []),
)}
# The arm table in hsi_augment/report.py's format: scripts/augmentation_select.py
# reads it from here and writes it into selection.json for the report.
ARM_TABLE = [{k: a[k] for k in ("key", "family", "screen_tag", "seed_tag", "bench_tag", "generator",
                                "description")} for a in ARMS.values()]


def source(a):
    """The export an arm trains on: cvae / ldm / A4's decision (None: undecided)."""
    return A4_SOURCE if a["generator"] == "decision" else a["generator"]


def _waits(a):
    """Why the arm's jobs cannot run yet, or None."""
    if a["copies"]:
        if GLASS["use_copy_mask"] is None:
            return GLASS["why"]
        if glass_rule()["use_copy_mask"] != GLASS["use_copy_mask"]:
            return f"{GLASS_RULE} changed since the sweep loaded: run it again"
        if MASKED and not _here(COPY_MASK).is_file():
            return f"no {COPY_MASK}"
    if a["generator"] == "decision":
        if A4_SOURCE is None:
            return f"no generator chosen when the sweep loaded ({DECISION}): run lopo-utility, then the sweep again"
        if a4_source() != A4_SOURCE:
            return f"{DECISION} changed since the sweep loaded: run it again"
    if a["generator"] and export_id(source(a)) is None:
        return f"no {OUT}/export/{source(a)}/provenance.json: run the export stage"
    if a["phys"]:
        missing = [p for p in PHYS_FILES if not _here(p).is_file()]
        if missing:
            return f"no {', '.join(missing)}: run the prep stage (spectral-audit, aug-calibration)"
    return None


def trained_on_current_export(tag: str, src) -> bool:
    """The custom `done` of a synthetic arm: its newest finished run recorded
    the current export's id (sweeps/paper_evidence_upgrade.py overrides
    `requires` the same way)."""
    current = export_id(src) if src else None
    d = run_dir(tag, HSI)
    if current is None or d is None:
        return False
    try:
        with open(d / "config.json") as f:
            recorded = (json.load(f).get("synthetic") or {}).get("export_id")
    except (OSError, ValueError):
        return False
    return recorded == current


def job(a, tag: str, epochs: int, seed: int):
    run = train(tag, HSI, **a["flags"], epochs=epochs, seed=seed)
    if a["generator"]:
        run.done = lambda tag=tag, src=source(a): trained_on_current_export(tag, src)
    run.requires = lambda a=a: _waits(a)
    return run


def _pending(jobs):
    """`requires` of a job that reads `jobs`' runs."""
    for j in jobs:
        if not j.done():
            return f"{j.name}: {j.requires() or 'not run yet'}"
    return None


# ============================================================================
# the seeds stage: only once selection.json exists
# ============================================================================

def selected() -> list:
    """Arm keys selection.json names (keys, screen tags or {"arm": key} dicts,
    under "selected" or "per_family"), [] without the file."""
    if not _here(SELECTION).is_file():
        return []
    doc = _json(SELECTION)
    if not isinstance(doc, dict):
        raise SystemExit(f"{SELECTION}: not a JSON object")
    raw = doc.get("selected")
    if raw is None:
        raw = list((doc.get("per_family") or {}).values())
    by_tag = {a["screen_tag"]: k for k, a in ARMS.items()}
    recorded = {d.get("key"): d.get("screen_tag") for d in doc.get("arms") or [] if isinstance(d, dict)}
    keys = []
    for s in raw:
        k = s.get("arm", s.get("key")) if isinstance(s, dict) else s
        k = by_tag.get(k, k)
        if k not in ARMS or ARMS[k]["family"] == "control":
            raise SystemExit(f"{SELECTION}: {k!r} is not a selectable arm of this sweep {sorted(ARMS)}"
                             + ("" if PHYS else " (AUG_PHYS=1 adds A1p)"))
        if recorded.get(k, ARMS[k]["screen_tag"]) != ARMS[k]["screen_tag"]:
            raise SystemExit(f"{SELECTION} selected {k} as {recorded[k]!r}, but this sweep's {k} is "
                             f"{ARMS[k]['screen_tag']!r} (did the glass rule change?)")
        if k not in keys:
            keys.append(k)
    if not keys:
        raise SystemExit(f"{SELECTION}: no selected arm")
    return keys


def _seeds():
    keys = selected()
    if not keys:
        return []
    return [job(ARMS[k], ARMS[k]["seed_tag"].format(seed=s), SEED_EPOCHS, s)
            for k in ("A0", *keys) for s in SEEDS]


# ============================================================================
# the pipeline's commands
# ============================================================================

def _stage_dirs(commands):
    """(output dirs of the stages `commands` run, None) or (None, why not)."""
    from hsi_augment.registry import RegistryError, load_builtin_stages     # imports torch: only when asked
    try:
        return [s.dirname for s in load_builtin_stages().resolve(list(commands))], None
    except RegistryError as e:
        return None, f"augment_pipeline.py: {e}"


def pipeline(name, *commands, extra=(), also=None, requires=None):
    """`augment_pipeline.py <commands>` on HSI -> OUT. Done when every stage
    the commands run has an ok done.json (and `also()`); waits while a command
    is only planned (hsi_augment/registry.py)."""
    def done():
        dirs, _ = _stage_dirs(commands)
        return dirs is not None and all((_json(f"{OUT}/{d}/done.json") or {}).get("status") == "ok"
                                        for d in dirs) and (also is None or also())

    def waits():
        return _stage_dirs(commands)[1] or (requires() if requires else None)

    return cmd(name, [PY, "augment_pipeline.py", *commands, "--data_dir", HSI, "--out", OUT, *extra],
               done_if=done, requires=waits)


BENCH = {f"bench-{k.lower()}": [job(a, a["bench_tag"], BENCH_EPOCHS, SCREEN_SEED)] for k, a in ARMS.items()}
SCREEN = [job(a, a["screen_tag"], SCREEN_EPOCHS, SCREEN_SEED) for a in ARMS.values()]

STAGES = {
    "prep": [
        pipeline("validate", "validate"),
        pipeline("preprocess", "preprocess"),
        pipeline("pool", "pool", extra=("--expect_tau_deg", TAU) if TAU else (),
                 requires=lambda: None if TAU else
                 f"tau is not signed off: check {OUT}/prep/tissue (tau.json, figures/), then AUG_TAU=<deg>"),
    ],
    "gen-cvae": [pipeline("fit-cvae", "fit-cvae")],
    "gen-ldm": [pipeline("fit-ae", "fit-ae"), pipeline("fit-ldm", "fit-ldm")],
    "export": [pipeline("export", "generate", "qc", "export",
                        also=lambda: all(export_id(s) for s in GENERATORS))],
    "lopo-utility": [pipeline("lopo-utility", "lopo-utility", also=lambda: a4_source() is not None)],
    **BENCH,
    "screen": SCREEN,
    "select": [cmd("select", [PY, "scripts/augmentation_select.py", "--sweep", "sweeps/augmentation_pipeline.py",
                              "--run_data", HSI, "--aug_out", OUT, "--out", SELECTION],
                   done_if=lambda: _here(SELECTION).is_file(),
                   requires=lambda: _pending([j for j, a in zip(SCREEN, ARMS.values()) if a["family"] != "control"]))],
    "seeds": _seeds(),
    "report": [
        pipeline("report-tissue", "report-tissue", extra=("--selection", SELECTION),
                 requires=lambda: None if _here(SELECTION).is_file() else f"no {SELECTION} yet (test is sealed)"),
        cmd("report", [PY, "scripts/augmentation_report.py", "--run_data", HSI, "--aug_out", OUT,
                       "--selection", SELECTION]),
    ],
}

GROUPS = {
    "pipeline": ["prep", "gen-cvae", "gen-ldm", "export", "lopo-utility"],
    "bench": list(BENCH),
    "classifier": [*BENCH, "screen", "select", "seeds", "report"],
    "all": list(STAGES),
}
