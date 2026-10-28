# -*- coding: utf-8 -*-
"""
hsi_augment/report.py
=====================
Phase 10 (#34; Plan §Phase 10 "Metrics", "Statistics", "Cost table", "Report"):
the augmentation study's report, built from ARTIFACTS ONLY. Nothing is
recomputed on a GPU; nothing is trained. The CLI is
`scripts/augmentation_report.py`; the statistics are in `stats.py`.

    python scripts/augmentation_report.py                  # screen + val (+ seeds val once selected)
    python scripts/augmentation_report.py --reveal_test    # + test columns, only if selection.json exists
    # -> reports/augmentation/<date>/{report.md, report.json, figures/*.png}

What it reads
-------------
* Training runs, `experiments/*/`, found by the `run_tag` in `config.json`
  (`cli_args.run_tag`) and a data dir containing `--run_data` - the
  `run_experiments.find_runs` / `scripts/analysis_v20.find_run` rule: the
  newest run with that tag that holds the file a finished run writes last.
    config.json        resolved flags, `train_subsample.kept`, `trm_ema_rate`,
                       `synthetic.export_id`, `minority_augmentation`
    history.json       per-epoch val metrics on the stratified val subsample
                       (the screen score; the selected epoch's val metrics)
    heldout_v19/validation/test_predictions.npz
                       ALL val rows, written by `scripts/heldout_eval.py` (the
                       evaluator's file name; the split is VALIDATION). Needed
                       for val per-patient, calibration and tissue/glass tables
    per_patient_metrics.json
                       val subsample per patient at the last eval-artifact epoch;
                       only a fallback for the DCIS patient's val recall
    test_predictions.npz, test_report.json
                       test, read ONLY when the seal is open (below)
* The build's int sidecars `y_{val,test}.npy`, `groups_{val,test}.npy` (a few MB).
* Stage outputs under the pipeline's `--out` (`--aug_out`): every `done.json`
  (cost and QC tables), `lopo_utility/decision.json` (A4's generator),
  `export/<source>/provenance.json` (export ids), and the `report-tissue`
  stage's `report/tissue/is_tissue_{val,test}.npy`.
* `selection.json` from `scripts/augmentation_select.py` (#33).

The test seal (Plan rule 3)
---------------------------
Test columns render only when `selection.json` exists AND `--reveal_test` was
passed. Every file the report uses goes through `Inputs`, which refuses a
test file while the seal is closed (`TestSealError`), so the rule is enforced
where files are read, not by the order of the code: `train.py` always writes
`test_predictions.npz` / `test_report.json` (`--eval_test best`), and a code
path that points at one of them by mistake fails instead of reading it. (Run
discovery, `index_runs`, opens only `config.json` for the run tag; a run's
`test_predictions.npz` is a "finished" marker whose existence is checked,
never its content.) The screen section reads `config.json` and `history.json`
only; the seeds section reads validation artifacts unless the seal is open.
In the pipeline, `report-tissue` is the only stage that may declare
`final_eval` (`splits.FINAL_EVAL_STAGES`), and it refuses test until
`selection.json` exists.

Row order
---------
`test_predictions.npz` rows are `X_test.npy` rows in file order: the trainer's
test loader is `DataLoader(NpyDatasetV16(X_test, y_test), shuffle=False)` over
the whole file (`training/train_pipeline.py:_build_test_loader_v16`), and the
evaluator concatenates batches in loader order
(`training/trainerg_v11.py:evaluate_test_split`); `scripts/heldout_eval.py`
does the same for val. The report CHECKS it for every file it reads: the row
count equals `len(y_<split>.npy)` and `true_label` equals `y_<split>.npy`
element-wise. A file that fails is excluded from every row-level analysis and
listed under `checks`. The `report-tissue` arrays are written in the same
`X_<split>.npy` row order and joined by row index.

Determinism
-----------
`report.md`, `report.json` and the figures are byte-identical when rebuilt
from the same artifacts with the same code: no timestamps in the content (the
date is the directory name only), sorted keys, sorted runs, a seeded bootstrap,
PNGs without a software tag. `report.json` records the report's own resolved
config (minus the output location), the git sha, every input file's sha256
and every run's resolved training config.

Cost-table contract for the generator stages (#23, #24, #29, #30)
-----------------------------------------------------------------
Every `done.json` under `--aug_out` gives the stage's wall time
(`timings.wall_s`) and output sizes. Peak VRAM and throughput come from the
stage's `summary`, under the first key found of `PEAK_VRAM_KEYS` (MB, i.e.
`torch.cuda.max_memory_allocated() / 2**20`) and `THROUGHPUT_KEYS` (samples/s);
a stage without them shows "-". GPU-minutes = wall minutes of the generator
stages (`fit-*`, `generate*`/`sample*`): the GPU is held for the whole stage.
The source of a stage (cvae / ldm) is read from its name and directory.
"""

from __future__ import annotations

import datetime as _dt
import io as _io
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hsi_augment import stats

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.stage import StageContext
    from hsi_augment.stages.report_tissue import ReportTissueConfig

PathLike = Union[str, os.PathLike]

REPORT_SCHEMA_VERSION = 1
DEFAULT_EXPERIMENTS = "experiments"
DEFAULT_DATA_DIR = "data/hsi_v9-trainsel/hsi"
DEFAULT_RUN_DATA = "hsi_v9-trainsel/hsi"
DEFAULT_AUG_OUT = "data/hsi_v9-trainsel-aug"
DEFAULT_SELECTION = "reports/augmentation/selection.json"
DEFAULT_OUT_ROOT = "reports/augmentation"
SEEDS: Tuple[int, ...] = (1, 7, 13, 23, 42)
SCREEN_SEED = 42
SCREEN_EPOCHS: Tuple[int, ...] = (10, 11, 12)
SCREEN_TIE_GAP = 0.005
KEPT_EXPECTED = 250_113
DEFAULT_CLASS_NAMES: Tuple[str, ...] = ("healthy", "DCIS", "IDC")
HEALTHY, DCIS = "healthy", "DCIS"

TEST_PRED = "test_predictions.npz"
TEST_REPORT = "test_report.json"
VAL_FULL_PRED = "heldout_v19/validation/test_predictions.npz"
# the only directories whose `test_*` files are val (see `_is_test_path`); fixed at import
VAL_PRED_DIR_PARTS: Tuple[str, ...] = Path(VAL_FULL_PRED).parent.parts
PER_PATIENT_JSON = "per_patient_metrics.json"
TISSUE_STAGE_DIR = "report/tissue"
TISSUE_JSON = "report_tissue.json"
DECISION_JSON = "lopo_utility/decision.json"
DECISION_KEYS = ("a4_source", "source", "generator", "chosen")
PEAK_VRAM_KEYS = ("peak_vram_mb", "max_memory_allocated_mb", "peak_gpu_memory_mb", "peak_vram_MB")
THROUGHPUT_KEYS = ("samples_per_s", "throughput_samples_per_s", "sampling_samples_per_s")
QC_KEY_RE = re.compile(r"accept|reject|n_generated|n_accepted|export_id|pass_rate")
REPORT_MD, REPORT_JSON, FIGURES_DIR = "report.md", "report.json", "figures"
BOOT_METRICS = ("f1_macro", "balanced_accuracy")

# metric -> (label, higher is better); the seed tables and the delta tables use this order
METRICS: Tuple[Tuple[str, str, bool], ...] = (
    ("f1_macro", "macro-F1", True),
    ("balanced_accuracy", "bal. acc.", True),
    ("dcis_f1", "DCIS F1", True),
    ("dcis_recall", "DCIS recall", True),
    ("dcis_precision", "DCIS precision", True),
    ("healthy_to_dcis_fpr", "healthy->DCIS FPR", False),
    ("roc_auc_macro", "AUROC", True),
    ("pr_auc_macro", "PR-AUC", True),
    ("ece", "ECE", False),
    ("nll", "NLL", False),
)
HISTORY_METRICS = ("f1_macro", "balanced_accuracy", "dcis_f1", "dcis_recall", "dcis_precision")
TISSUE_METRICS = ("f1_macro", "balanced_accuracy", "dcis_f1", "dcis_recall", "dcis_precision",
                  "healthy_to_dcis_fpr")

SEED42_NOTE = (f"Seed {SCREEN_SEED} appears in both the screen and the seeds stage, so the seed-{SCREEN_SEED} "
               f"runs are not independent of the arm selection.")
VAL_NOTE = ("Validation selected every run's checkpoint and the arms (one DCIS patient), so validation "
            "numbers are optimistic; test is the final evaluation.")
DCIS_NOTE = ("Each held-out split has ONE DCIS patient: DCIS metrics are one patient's rows, reported per "
             "patient, and no patient-level CI is claimed for DCIS.")
TISSUE_NOTE = ("Tissue / glass: the train-fitted tissue score and threshold (prep/tissue, P2b) applied to "
               "val/test rows without labels (stage report-tissue). Analysis only. 'glass' = not tissue, "
               "including rows whose score is undefined.")


class ReportError(RuntimeError):
    """An input the report cannot use (malformed selection.json, unknown arm, ...)."""


class TestSealError(PermissionError):
    """A test file was about to be read while the test seal is closed."""
    __test__ = False    # not a pytest test class


# ============================================================================
# arms
# ============================================================================

FAMILIES = ("control", "non-generative", "generative")


@dataclass(frozen=True)
class Arm:
    """One arm of Plan §Phase 10. `seed_tag` is a template with `{seed}`.
    `generator`: None (real rows only), "cvae", "ldm", or "decision" (A4: the
    source named by lopo_utility/decision.json)."""
    key: str
    family: str
    screen_tag: str
    seed_tag: str
    bench_tag: str
    generator: Optional[str] = None
    description: str = ""

    def tag_for_seed(self, seed: int) -> str:
        return self.seed_tag.format(seed=seed)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Arm":
        unknown = set(d) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ReportError(f"arm {d.get('key')!r}: unknown fields {sorted(unknown)}")
        arm = cls(**{k: d[k] for k in d})
        if arm.family not in FAMILIES:
            raise ReportError(f"arm {arm.key!r}: family {arm.family!r} not in {FAMILIES}")
        if "{seed}" not in arm.seed_tag:
            raise ReportError(f"arm {arm.key!r}: seed_tag {arm.seed_tag!r} has no {{seed}}")
        if arm.generator not in (None, "cvae", "ldm", "decision"):
            raise ReportError(f"arm {arm.key!r}: generator {arm.generator!r}")
        return arm


# Plan §Phase 10 "Arms". Seeds-stage tags follow sweeps/minority_augmentation.py
# (`<tag>-s<seed>`); A0's are the paper's shared runs. If glass_rule.json says
# tissue-only copies, A0os/A1 become aug-copy/aug-copymix: pass --arms_file or
# put the arm table in selection.json ("arms").
DEFAULT_ARMS: Tuple[Arm, ...] = (
    Arm("A0", "control", "v20-abl-base-s42", "v20-head-s{seed}", "aug-bench-a0", None,
        "real only (REF, frac 0.102)"),
    Arm("A0os", "non-generative", "minaug-nomix", "minaug-nomix-s{seed}", "aug-bench-a0os", None,
        "DCIS copy x2, no mixing"),
    Arm("A1", "non-generative", "minaug-default", "minaug-default-s{seed}", "aug-bench-a1", None,
        "DCIS copy x2, stats mixing 0.5"),
    Arm("A2", "generative", "aug-cvae", "aug-cvae-s{seed}", "aug-bench-a2", "cvae",
        "+122,850 CVAE DCIS rows"),
    Arm("A3", "generative", "aug-ldm", "aug-ldm-s{seed}", "aug-bench-a3", "ldm",
        "+122,850 LDM DCIS rows"),
    Arm("A4", "generative", "aug-combo", "aug-combo-s{seed}", "aug-bench-a4", "decision",
        "+122,850 rows from the LOPO-chosen generator, factor 1.0, stats mixing 0.5"),
    Arm("A1p", "non-generative", "aug-conv-phys", "aug-conv-phys-s{seed}", "aug-bench-a1p", None,
        "A1 + --phys_aug minority (optional)"),
)


def arms_from_doc(doc: Any, where: str) -> Tuple[Arm, ...]:
    """A list of arm dicts, or {"arms": [...]}; exactly one control arm."""
    items = doc.get("arms") if isinstance(doc, Mapping) else doc
    if not isinstance(items, list) or not items:
        raise ReportError(f"{where}: expected a non-empty list of arms")
    arms = tuple(Arm.from_dict(a) for a in items)
    keys = [a.key for a in arms]
    if len(set(keys)) != len(keys):
        raise ReportError(f"{where}: duplicate arm keys {keys}")
    if sum(a.family == "control" for a in arms) != 1:
        raise ReportError(f"{where}: exactly one arm must have family 'control'")
    return arms


# ============================================================================
# configuration
# ============================================================================

@dataclass(frozen=True)
class ReportConfig:
    experiments: str = DEFAULT_EXPERIMENTS
    data_dir: str = DEFAULT_DATA_DIR          # the build's array dir: y_/groups_ sidecars
    run_data: str = DEFAULT_RUN_DATA          # a run counts if its cli_args.data_dir contains this
    aug_out: str = DEFAULT_AUG_OUT            # augment_pipeline.py --out
    selection: str = DEFAULT_SELECTION
    out_root: str = DEFAULT_OUT_ROOT
    date: Optional[str] = None                # directory name only; default: today (UTC)
    reveal_test: bool = False
    arms_file: Optional[str] = None
    seeds: Tuple[int, ...] = SEEDS
    screen_seed: int = SCREEN_SEED
    screen_epochs: Tuple[int, ...] = SCREEN_EPOCHS
    kept_expected: int = KEPT_EXPECTED
    n_boot: int = 2000
    boot_seed: int = 0

    def content_dict(self) -> Dict[str, Any]:
        """What the report records about itself: everything but where it is written."""
        d = asdict(self)
        for k in ("date", "out_root"):
            d.pop(k)
        return d

    def resolved_date(self) -> str:
        d = self.date or _dt.datetime.now(_dt.timezone.utc).date().isoformat()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            raise ReportError(f"date {d!r} is not YYYY-MM-DD")
        return d


# ============================================================================
# every read goes through Inputs: sha256 recorded, test sealed
# ============================================================================

def _is_test_path(path: Path) -> bool:
    """A file of the TEST split: `test_*` (or `*_test.npy`), unless the
    directories IMMEDIATELY above it are `VAL_PRED_DIR_PARTS`
    (`heldout_v19/validation/`): `scripts/heldout_eval.py`'s val files reuse
    the evaluator's `test_*` names. Only the file's own parent and grandparent
    count, so a `validation` directory higher up (an experiments root nested
    under one) never exempts a run's real test files."""
    path = Path(path)
    if path.parent.parts[-len(VAL_PRED_DIR_PARTS):] == VAL_PRED_DIR_PARTS:
        return False
    name = path.name
    return name.startswith("test_") or bool(re.search(r"_test\.npy$", name))


class Inputs:
    """The report's file reader. Labels every file relative to a named root
    (no machine paths in the content), records its sha256, and refuses a test
    file while the seal is closed.

    A file is a test file when its name says so (`_is_test_path`) OR the caller
    passes `test=True` (for a file whose name does not reveal its split, such
    as a sidecar read for the test split). `test` can only ADD the seal: no
    call site can declare a `test_*` file to be something else, so a code path
    that points at a test file by mistake is refused, not read."""

    def __init__(self, roots: Mapping[str, PathLike], *, reveal_test: bool):
        self.roots = {k: Path(os.path.abspath(v)) for k, v in roots.items()}
        self.reveal_test = bool(reveal_test)
        self.sha256: Dict[str, str] = {}

    def label(self, path: PathLike) -> str:
        p = Path(os.path.abspath(path))
        best = None
        for name, root in self.roots.items():
            if p == root or root in p.parents:
                rel = p.relative_to(root).as_posix()
                if best is None or len(str(root)) > best[0]:
                    best = (len(str(root)), f"{name}/{rel}")
        if best is None:
            raise ReportError(f"{p} is outside every report input root {sorted(self.roots)}")
        return best[1]

    def _before_read(self, path: Path, test: bool) -> str:
        is_test = bool(test) or _is_test_path(path)
        if is_test and not self.reveal_test:
            raise TestSealError(f"test is sealed: refusing to read {self.label(path)} (needs selection.json "
                                f"and --reveal_test)")
        from hsi_augment.io import sha256_file
        lab = self.label(path)
        self.sha256[lab] = sha256_file(path)
        return lab

    def json(self, path: PathLike, *, test: bool = False) -> Any:
        p = Path(path)
        self._before_read(p, test)
        with open(p) as f:
            return json.load(f)

    def npy(self, path: PathLike, *, test: bool = False) -> np.ndarray:
        p = Path(path)
        self._before_read(p, test)
        return np.load(p)

    def npz(self, path: PathLike, *, test: bool = False) -> Dict[str, np.ndarray]:
        p = Path(path)
        self._before_read(p, test)
        with np.load(p) as z:
            return {k: z[k] for k in z.files}


# ============================================================================
# finding runs (run_experiments.find_runs / analysis_v20.find_run)
# ============================================================================

def index_runs(experiments: PathLike) -> List[Dict[str, Any]]:
    """[{dir, tag, data_dir}] for every experiments/*/config.json, sorted by dir name."""
    out = []
    for c in sorted(Path(experiments).glob("*/config.json")):
        try:
            with open(c) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        a = cfg.get("cli_args") or {}
        out.append({"dir": c.parent, "tag": a.get("run_tag"), "data_dir": str(a.get("data_dir", ""))})
    return out


def find_run(index: Sequence[Mapping[str, Any]], tag: str, data_contains: Optional[str],
             need: str) -> Optional[Path]:
    """The newest run with `tag` (and a data dir containing `data_contains`)
    that holds `need`. Only `need`'s existence is checked, never its content."""
    hits = [r["dir"] for r in index if r["tag"] == tag
            and (not data_contains or data_contains.rstrip("/") in r["data_dir"])
            and (Path(r["dir"]) / need).is_file()]
    return Path(hits[-1]) if hits else None


def _class_names(config: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(config.get("class_names") or DEFAULT_CLASS_NAMES)


def _class_idx(names: Sequence[str]) -> Tuple[int, int]:
    try:
        return list(names).index(HEALTHY), list(names).index(DCIS)
    except ValueError as e:
        raise ReportError(f"class names {list(names)} lack {HEALTHY!r}/{DCIS!r}") from e


def run_record(config: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    """The resolved training config of a run, as the report records it."""
    a = dict(config.get("cli_args") or {})
    syn = config.get("synthetic")
    return {
        "run": run_dir.name, "tag": a.get("run_tag"), "seed": a.get("seed"), "epochs": a.get("epochs"),
        "profile": config.get("profile"), "trainer": config.get("trainer"),
        "entry_point": config.get("entry_point"), "class_names": list(_class_names(config)),
        "train_subsample": config.get("train_subsample"), "val_subsample": config.get("val_subsample"),
        "trm_ema_rate": a.get("trm_ema_rate"),
        "minority_augmentation": config.get("minority_augmentation"),
        "synthetic": syn,
        "cli_args": a,
    }


# ============================================================================
# metrics
# ============================================================================

def _nan_none(x: Any) -> Any:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def confusion_summary(cm: np.ndarray, healthy: int, dcis: int) -> Dict[str, Any]:
    """Every metric a confusion matrix gives (rows true, columns predicted)."""
    cm = np.asarray(cm, dtype=np.int64)
    pc = stats.per_class_from_confusion(cm)
    total = int(cm.sum())
    h_rows = int(cm[healthy].sum())
    return {
        "n": total,
        "f1_macro": stats.metric_from_confusion(cm, "f1_macro"),
        "balanced_accuracy": _nan_none(stats.metric_from_confusion(cm, "balanced_accuracy")),
        "accuracy": float(np.trace(cm) / total) if total else None,
        "per_class_precision": pc["precision"], "per_class_recall": pc["recall"],
        "per_class_f1": pc["f1"], "per_class_support": pc["support"],
        "dcis_f1": pc["f1"][dcis], "dcis_recall": pc["recall"][dcis] if pc["support"][dcis] else None,
        "dcis_precision": pc["precision"][dcis],
        "healthy_to_dcis_fpr": float(cm[healthy, dcis] / h_rows) if h_rows else None,
        "confusion_matrix": cm.tolist(),
    }


def prediction_metrics(y: np.ndarray, pred: np.ndarray, probs: Optional[np.ndarray],
                       class_names: Sequence[str]) -> Dict[str, Any]:
    """Confusion-derived metrics plus AUROC / PR-AUC (one-vs-rest, per class
    and macro) and calibration (`training.metrics.calibration_metrics`, 15
    bins: the same code as each run's test_report.json)."""
    k = len(class_names)
    healthy, dcis = _class_idx(class_names)
    labels = tuple(range(k))
    out = confusion_summary(stats.confusion(y, pred, labels), healthy, dcis)
    if probs is None:
        return out
    from sklearn.metrics import average_precision_score, roc_auc_score
    from training.metrics import calibration_metrics
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(y).astype(np.int64)
    roc, pr = [], []
    for c in labels:
        pos = y == c
        both = bool(pos.any() and (~pos).any())
        roc.append(float(roc_auc_score(pos, p[:, c])) if both else None)
        pr.append(float(average_precision_score(pos, p[:, c])) if pos.any() else None)
    out["per_class_roc_auc"], out["per_class_pr_auc"] = roc, pr
    out["roc_auc_macro"] = float(np.mean([v for v in roc if v is not None])) if any(v is not None for v in roc) \
        else None
    out["pr_auc_macro"] = float(np.mean([v for v in pr if v is not None])) if any(v is not None for v in pr) \
        else None
    out.update({k_: float(v) for k_, v in calibration_metrics(p, y).items()})
    return out


def history_metrics(entry: Mapping[str, Any], dcis: int) -> Dict[str, Any]:
    """The val-subsample metrics history.json holds for one epoch."""
    def pc(key: str) -> Optional[float]:
        v = entry.get(key)
        return float(v[dcis]) if isinstance(v, list) and len(v) > dcis else None
    return {"epoch": entry.get("epoch"), "f1_macro": _nan_none(entry.get("f1_macro")),
            "balanced_accuracy": _nan_none(entry.get("balanced_accuracy")),
            "dcis_f1": pc("per_class_f1"), "dcis_recall": pc("per_class_recall"),
            "dcis_precision": pc("per_class_precision"),
            "per_class_f1": entry.get("per_class_f1"), "per_class_precision": entry.get("per_class_precision"),
            "per_class_recall": entry.get("per_class_recall")}


def best_epoch(history: Sequence[Mapping[str, Any]], metric: str = "f1_macro") -> Optional[Mapping[str, Any]]:
    """The first epoch with the highest val `metric` (the trainer keeps a
    checkpoint only on a strict improvement)."""
    best = None
    for e in history:
        v = e.get(metric)
        if v is not None and math.isfinite(float(v)) and (best is None or float(v) > float(best[metric])):
            best = e
    return best


def screen_score(history: Sequence[Mapping[str, Any]], epochs: Sequence[int], dcis: int) -> Dict[str, Any]:
    """Pre-registered: mean val f1_macro over `epochs` (10-12); tie-break input:
    mean val DCIS F1 (per_class_f1[DCIS]) over the same epochs."""
    by_epoch = {int(e.get("epoch")): e for e in history if e.get("epoch") is not None}
    missing = [ep for ep in epochs if ep not in by_epoch or by_epoch[ep].get("f1_macro") is None]
    if missing:
        return {"score": None, "dcis_f1": None, "epochs": list(epochs), "reason": f"epochs {missing} missing"}
    f1 = [float(by_epoch[ep]["f1_macro"]) for ep in epochs]
    d = [float(by_epoch[ep]["per_class_f1"][dcis]) for ep in epochs]
    return {"score": float(np.mean(f1)), "dcis_f1": float(np.mean(d)), "epochs": list(epochs),
            "per_epoch_f1_macro": f1}


def tissue_breakdown(y: np.ndarray, pred: np.ndarray, is_tissue: np.ndarray,
                     class_names: Sequence[str]) -> Dict[str, Any]:
    healthy, dcis = _class_idx(class_names)
    labels = tuple(range(len(class_names)))
    out = {}
    for region, mask in (("tissue", is_tissue), ("glass", ~is_tissue)):
        s = confusion_summary(stats.confusion(y[mask], pred[mask], labels), healthy, dcis)
        out[region] = {k: s[k] for k in ("n", "per_class_support", "confusion_matrix", *TISSUE_METRICS)}
    return out


def patient_table(patients: np.ndarray, pcm: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Per patient: rows, macro recall over the classes it has
    (heldout_metrics.per_patient), per-class recall."""
    out = {}
    for pid, cm in zip(patients.tolist(), pcm):
        support = cm.sum(1)
        present = [c for c in range(cm.shape[0]) if support[c] > 0]
        rec = {str(c): float(cm[c, c] / support[c]) for c in present}
        out[str(pid)] = {"n": int(cm.sum()), "classes_present": present,
                         "macro_recall": float(np.mean(list(rec.values()))) if rec else None,
                         "per_class_recall": rec}
    return out


# ============================================================================
# the report-tissue stage (registered in stages/report_tissue.py)
# ============================================================================

def run_report_tissue(ctx: "StageContext", cfg: "ReportTissueConfig") -> Dict[str, Any]:
    """`augment_pipeline.py report-tissue`: per-row tissue flags of val/test
    from the TRAIN-fitted reference (`tissue.load_reference(prep/tissue)`),
    without labels: only X is opened, under the `final_eval` purpose. Rows are
    scored and written in X_<split>.npy order, `cfg.chunk_rows` at a time.
    Test is read only once `cfg.selection` exists."""
    from hsi_augment import tissue
    from hsi_augment.io import iter_chunks, sha256_file
    from hsi_augment.stage import StageError, read_done

    if "test" in cfg.splits and not Path(cfg.selection).is_file():
        raise StageError(f"test is sealed until selection.json exists ({cfg.selection}); run "
                         f"`--splits val` before the selection")
    up = ctx.upstream_dir("tissue-mask")
    ref = tissue.load_reference(up)
    summary: Dict[str, Any] = {}
    for split in cfg.splits:
        arr = ctx.open_split(split, purpose="final_eval", kinds=("X",))
        n = arr.n
        n_tissue = n_undef = 0
        with ctx.npy_writer(f"is_tissue_{split}.npy", np.bool_, (n,)) as wm, \
                ctx.npy_writer(f"tissue_score_{split}.npy", tissue.SCORE_DTYPE, (n,)) as ws:
            for ch in iter_chunks(arr.X, chunk_rows=cfg.chunk_rows):
                score = ref.score(ch.data).astype(tissue.SCORE_DTYPE)
                mask = tissue.apply_threshold(score, ref.tau_deg)     # == ref.is_tissue(ch.data)
                ws.write(score)
                wm.write(mask)
                n_tissue += int(mask.sum())
                n_undef += int(np.isnan(score).sum())
        summary[split] = {"n": n, "n_tissue": n_tissue, "n_glass": n - n_tissue, "n_undefined_score": n_undef,
                          "frac_tissue": n_tissue / n if n else None}
        ctx.log(f"{split}: {n_tissue:,}/{n:,} rows tissue ({n_tissue / max(n, 1):.1%}); "
                f"{n_undef} undefined scores")
    up_done = read_done(up) or {}
    doc = {"schema_version": REPORT_SCHEMA_VERSION, "rule": tissue.MASK_RULE, "tau_deg": ref.tau_deg,
           "s_glass_sha256": sha256_file(up / tissue.S_GLASS_NAME),
           "tissue_mask_fingerprint": up_done.get("fingerprint"),
           "row_order": "row i of is_tissue_<split>.npy / tissue_score_<split>.npy is row i of X_<split>.npy",
           "labels_read": False, "splits": summary}
    ctx.save_json(TISSUE_JSON, doc)
    return {s: {"n": v["n"], "frac_tissue": v["frac_tissue"]} for s, v in summary.items()}


# ============================================================================
# building the report
# ============================================================================

@dataclass
class _Run:
    """One seeds-stage run's row-level results on one split (kept small)."""
    metrics: Dict[str, Any]
    pred: np.ndarray                      # int8, X_<split> row order
    patients: np.ndarray
    pcm: np.ndarray                       # (P, K, K)
    tissue: Optional[Dict[str, Any]] = None


@dataclass
class _Ctx:
    cfg: ReportConfig
    inp: Inputs
    arms: Tuple[Arm, ...]
    index: List[Dict[str, Any]]
    sealed_reason: Optional[str]
    checks: List[Dict[str, Any]] = field(default_factory=list)
    runs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    class_names: Tuple[str, ...] = DEFAULT_CLASS_NAMES

    @property
    def reveal(self) -> bool:
        return self.sealed_reason is None

    def check(self, name: str, ok: Optional[bool], detail: str) -> None:
        self.checks.append({"check": name, "ok": ok, "detail": detail})

    def read_run_config(self, rd: Path, arm: Arm, role: str) -> Dict[str, Any]:
        cfg = self.inp.json(rd / "config.json")
        prev = self.runs.get(rd.name)
        if prev is not None:         # one run in two roles (e.g. screen and bench): keep both
            prev["role"] = "+".join(sorted(set(prev["role"].split("+")) | {role}))
            if arm.key not in prev["arm"].split("+"):
                prev["arm"] = "+".join([*prev["arm"].split("+"), arm.key])
        else:
            self.runs[rd.name] = {**run_record(cfg, rd), "arm": arm.key, "role": role}
        self.class_names = _class_names(cfg)
        return cfg


def _load_selection(inp: Inputs, path: Path, arms: Tuple[Arm, ...]) -> Tuple[Optional[Dict[str, Any]],
                                                                               Tuple[Arm, ...]]:
    """(selection, arms): the selection document (None if absent) and the arm
    table it carries ("arms"), else `arms`.

    Expected from #33 (`scripts/augmentation_select.py`):
        {"schema_version": 1, "rule": "...",
         "selected": ["A3", "A1"] | [{"arm": "A3", ...}, ...],   # or "per_family": {family: arm}
         "scores": {"A3": 0.83 | {"score": 0.83, ...}, ...},    # optional, cross-checked
         "arms": [{key, family, screen_tag, seed_tag, bench_tag, generator, description}, ...]}  # optional
    """
    if not path.is_file():
        return None, arms
    doc = inp.json(path)
    if not isinstance(doc, Mapping):
        raise ReportError(f"{path.name}: expected a JSON object")
    if doc.get("arms") is not None:
        arms = arms_from_doc(doc["arms"], path.name)
    raw = doc.get("selected")
    if raw is None:
        raw = list((doc.get("per_family") or {}).values())
    keys = []
    by_key = {a.key: a for a in arms}
    by_tag = {a.screen_tag: a for a in arms}
    for s in raw or []:
        k = s.get("arm", s.get("key")) if isinstance(s, Mapping) else s
        if k in by_tag and k not in by_key:
            k = by_tag[k].key
        if k not in by_key:
            raise ReportError(f"{path.name}: selected arm {k!r} is not one of {sorted(by_key)}")
        if by_key[k].family == "control":
            raise ReportError(f"{path.name}: the control arm {k!r} cannot be selected")
        if k not in keys:
            keys.append(k)
    if not keys:
        raise ReportError(f"{path.name}: no selected arm ('selected' or 'per_family')")
    return {"selected": keys, "document": doc}, arms


def _screen(c: _Ctx, selection: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Screen (seed 42, 12 epochs): config.json and history.json only."""
    rows = []
    for arm in c.arms:
        rd = find_run(c.index, arm.screen_tag, c.cfg.run_data, need="history.json")
        row: Dict[str, Any] = {"arm": arm.key, "family": arm.family, "tag": arm.screen_tag,
                               "description": arm.description}
        if rd is None:
            rows.append({**row, "status": "missing"})
            continue
        cfg = c.read_run_config(rd, arm, "screen")
        hist = c.inp.json(rd / "history.json")
        _, dcis = _class_idx(_class_names(cfg))
        sc = screen_score(hist, c.cfg.screen_epochs, dcis)
        be = best_epoch(hist)
        row.update({"status": "ok" if sc["score"] is not None else "incomplete", "run": rd.name,
                    "epochs_run": len(hist), "seed": (cfg.get("cli_args") or {}).get("seed"),
                    **{k: sc[k] for k in ("score", "dcis_f1")}, "score_detail": sc,
                    "best_epoch": history_metrics(be, dcis) if be else None})
        rows.append(row)
    ref = next((r for r in rows if r["family"] == "control"), None)
    ref_score = ref.get("score") if ref else None
    for r in rows:
        r["delta_score_vs_A0"] = (r["score"] - ref_score) if (r.get("score") is not None
                                                              and ref_score is not None) else None
    for fam in ("non-generative", "generative"):
        scored = sorted((r for r in rows if r["family"] == fam and r.get("score") is not None),
                        key=lambda r: (-r["score"], -(r["dcis_f1"] or 0.0), r["arm"]))
        for i, r in enumerate(scored):
            r["family_rank"] = i + 1
            r["gap_to_family_best"] = scored[0]["score"] - r["score"]
    out: Dict[str, Any] = {"seed": c.cfg.screen_seed, "score_epochs": list(c.cfg.screen_epochs),
                           "rule": f"best arm per family by mean val f1_macro over epochs "
                                   f"{list(c.cfg.screen_epochs)}; ties (gap < {SCREEN_TIE_GAP}) -> val DCIS F1 "
                                   f"-> the cheaper method (scripts/augmentation_select.py, #33)",
                           "rows": rows}
    if selection is not None:
        doc_scores = selection["document"].get("scores") or {}
        sel_rows = []
        for k in selection["selected"]:
            r = next(r for r in rows if r["arm"] == k)
            gap = r.get("gap_to_family_best")
            given = doc_scores.get(k)
            given = given.get("score") if isinstance(given, Mapping) else given
            sel_rows.append({"arm": k, "family": r["family"], "score": r.get("score"),
                             "gap_to_family_best": gap,
                             "within_tie_gap": None if gap is None else bool(gap < SCREEN_TIE_GAP),
                             "selection_json_score": given,
                             "score_matches": None if (given is None or r.get("score") is None)
                             else bool(abs(float(given) - r["score"]) < 1e-9)})
            if gap is not None:
                c.check(f"selection_{k}_permitted_by_rule", bool(gap < SCREEN_TIE_GAP),
                        f"{k} is {gap:.4f} below its family's best screen score (tie gap {SCREEN_TIE_GAP})")
        out["selection"] = sel_rows
    return out


def _split_truth(c: _Ctx, split: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    d = Path(c.cfg.data_dir)
    test = split == "test"
    y = c.inp.npy(d / f"y_{split}.npy", test=test).astype(np.int64)
    g = c.inp.npy(d / f"groups_{split}.npy", test=test).astype(np.int64)
    tissue = None
    tdir = Path(c.cfg.aug_out) / TISSUE_STAGE_DIR
    tp = tdir / f"is_tissue_{split}.npy"
    if tp.is_file() and (tdir / "done.json").is_file():
        done = c.inp.json(tdir / "done.json")
        tissue = c.inp.npy(tp, test=test).astype(bool)
        ok = done.get("status") == "ok" and len(tissue) == len(y)
        c.check(f"tissue_{split}_rows", ok, f"is_tissue_{split}.npy: {len(tissue)} rows, y_{split}.npy: {len(y)}; "
                                            f"report-tissue done.json status {done.get('status')!r}")
        if not ok:
            tissue = None
    return y, g, tissue


def _score_file(c: _Ctx, path: Path, split: str, y: np.ndarray, g: np.ndarray,
                tissue: Optional[np.ndarray]) -> Optional[_Run]:
    z = c.inp.npz(path, test=(split == "test"))
    true = np.asarray(z["true_label"]).astype(np.int64)
    pred = np.asarray(z["predicted_label"] if "predicted_label" in z else z["probs"].argmax(1)).astype(np.int64)
    probs = z.get("probs")
    ok = len(true) == len(y) and bool(np.array_equal(true, y)) and len(pred) == len(y)
    c.check(f"row_order:{c.inp.label(path)}", ok,
            f"{len(true)} rows; true_label {'equals' if ok else 'DIFFERS FROM'} y_{split}.npy row by row")
    if not ok:
        return None
    m = prediction_metrics(y, pred, probs, c.class_names)
    patients, pcm = stats.patient_confusions(y, pred, g, tuple(range(len(c.class_names))))
    t = tissue_breakdown(y, pred, tissue, c.class_names) if tissue is not None else None
    return _Run(metrics=m, pred=pred.astype(np.int8), patients=patients, pcm=pcm, tissue=t)


def _mean_sd_map(values: Mapping[Any, Optional[float]]) -> Dict[str, Any]:
    v = [x for x in values.values() if x is not None]
    return stats.mean_sd(v)


def _seeds(c: _Ctx, selection: Mapping[str, Any]) -> Dict[str, Any]:
    """A0 + the selected arms x seeds. Val always; test only when revealed."""
    arms = [a for a in c.arms if a.family == "control"] + [a for k in selection["selected"]
                                                         for a in c.arms if a.key == k]
    ref = arms[0]
    runs: Dict[str, Dict[int, Path]] = {}
    status = []
    for arm in arms:
        runs[arm.key] = {}
        for s in c.cfg.seeds:
            tag = arm.tag_for_seed(s)
            rd = find_run(c.index, tag, c.cfg.run_data, need=TEST_PRED)
            status.append({"arm": arm.key, "seed": s, "tag": tag, "run": rd.name if rd else None})
            if rd is not None:
                runs[arm.key][s] = rd
                c.read_run_config(rd, arm, "seeds")
    out: Dict[str, Any] = {"arms": [a.key for a in arms], "seeds": list(c.cfg.seeds), "runs": status,
                           "splits": {}}
    healthy, dcis = _class_idx(c.class_names)

    # history (val subsample, selected epoch) for every run: the fallback val source
    hist_val: Dict[str, Dict[int, Dict[str, Any]]] = {}
    pp_val: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for arm in arms:
        hist_val[arm.key], pp_val[arm.key] = {}, {}
        for s, rd in runs[arm.key].items():
            be = best_epoch(c.inp.json(rd / "history.json"))
            hist_val[arm.key][s] = history_metrics(be, dcis) if be else {}
            if (rd / PER_PATIENT_JSON).is_file():
                pp_val[arm.key][s] = c.inp.json(rd / PER_PATIENT_JSON)

    for split in ["val"] + (["test"] if c.reveal else []):
        y, g, tissue = _split_truth(c, split)
        dcis_patients = stats.patients_with_class(y, g, (dcis,))[str(dcis)]
        scored: Dict[str, Dict[int, _Run]] = {}
        for arm in arms:
            scored[arm.key] = {}
            for s, rd in runs[arm.key].items():
                path = rd / (VAL_FULL_PRED if split == "val" else TEST_PRED)
                if path.is_file():
                    r = _score_file(c, path, split, y, g, tissue)
                    if r is not None:
                        scored[arm.key][s] = r
                if split == "test" and (rd / TEST_REPORT).is_file() and s in scored[arm.key]:
                    rep = c.inp.json(rd / TEST_REPORT, test=True)
                    sk = rep.get("sklearn_metrics") or {}
                    mine = scored[arm.key][s].metrics
                    same = all(abs(float(sk[k]) - float(mine[k])) < 1e-6 for k in ("f1_macro", "balanced_accuracy")
                               if k in sk)
                    c.check(f"test_report_matches:{rd.name}", same,
                            f"test_report.json f1_macro/balanced_accuracy vs recomputed from {TEST_PRED}")
                    c.runs[rd.name]["test_checkpoint_epoch"] = rep.get("checkpoint_epoch")
                    c.runs[rd.name]["test_checkpoint_weights"] = rep.get("checkpoint_weights")
        all_full = all(len(scored[a.key]) == len(runs[a.key]) for a in arms) and \
            any(len(runs[a.key]) for a in arms)
        source = "full" if (split == "test" or all_full) else "history"
        values: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for arm in arms:
            if source == "full":
                values[arm.key] = {s: r.metrics for s, r in scored[arm.key].items()}
            else:
                values[arm.key] = dict(hist_val[arm.key])
        metric_keys = [m for m in METRICS if source == "full" or m[0] in HISTORY_METRICS]
        per_arm = {}
        for arm in arms:
            per_arm[arm.key] = {
                "per_seed": {str(s): {k: v.get(k) for k, _, _ in metric_keys}
                             for s, v in sorted(values[arm.key].items())},
                "summary": {k: _mean_sd_map({s: v.get(k) for s, v in values[arm.key].items()})
                            for k, _, _ in metric_keys}}
        deltas, decisions, boots = {}, {}, {}
        for arm in arms[1:]:
            deltas[arm.key] = {k: stats.paired_deltas({s: v.get(k) for s, v in values[arm.key].items()},
                                                      {s: v.get(k) for s, v in values[ref.key].items()},
                                                      higher_is_better=hib)
                               for k, _, hib in metric_keys}
            f1d = deltas[arm.key]["f1_macro"]
            decisions[arm.key] = {"minimum_effect": stats.minimum_effect(f1d),
                                  "wilcoxon": stats.wilcoxon_one_sided(f1d)}
            if source == "full":
                paired = sorted(set(scored[arm.key]) & set(scored[ref.key]))
                if paired:
                    boots[arm.key] = {m: stats.patient_bootstrap(
                        y, [scored[arm.key][s].pred for s in paired], [scored[ref.key][s].pred for s in paired], g,
                        labels=tuple(range(len(c.class_names))), metric=m, n_boot=c.cfg.n_boot,
                        seed=c.cfg.boot_seed) | {"seeds": paired} for m in BOOT_METRICS}
        sp: Dict[str, Any] = {
            "source": {"full": f"all {split} rows ({'test_predictions.npz' if split == 'test' else VAL_FULL_PRED})",
                       "history": "history.json at the selected epoch (val subsample; "
                                  f"{VAL_FULL_PRED} missing for some run)"}[source],
            "source_kind": source, "n_rows": int(len(y)),
            "dcis_patients": dcis_patients,
            "dcis_ci": stats.class_ci_policy(y, g, dcis),
            "per_arm": per_arm, "deltas_vs_A0": deltas, "decisions": decisions, "patient_bootstrap": boots,
        }
        # per patient, confusion and tissue/glass from the full-row runs available
        pts, conf, tis, dcis_seed = {}, {}, {}, {}
        for arm in arms:
            rs = scored[arm.key]
            if rs:
                tables = {s: patient_table(r.patients, r.pcm) for s, r in rs.items()}
                pids = sorted({p for t in tables.values() for p in t}, key=int)
                pts[arm.key] = {p: {
                    "n": next(t[p]["n"] for t in tables.values() if p in t),
                    "macro_recall": stats.mean_sd([t[p]["macro_recall"] for t in tables.values() if p in t]),
                    "dcis_recall": stats.mean_sd([t[p]["per_class_recall"][str(dcis)] for t in tables.values()
                                                  if p in t and str(dcis) in t[p]["per_class_recall"]]),
                } for p in pids}
                conf[arm.key] = {"n_seeds": len(rs),
                                 "summed_over_seeds": np.sum([r.metrics["confusion_matrix"] for r in rs.values()],
                                                             axis=0).tolist()}
                dcis_seed[arm.key] = {str(p): {str(s): tables[s].get(str(p), {}).get("per_class_recall", {}).get(
                    str(dcis)) for s in sorted(tables)} for p in dcis_patients}
                if all(r.tissue is not None for r in rs.values()):
                    tis[arm.key] = {region: {
                        "n": rs[min(rs)].tissue[region]["n"],
                        "per_class_support": rs[min(rs)].tissue[region]["per_class_support"],
                        **{k: stats.mean_sd([r.tissue[region][k] for r in rs.values() if r.tissue[region][k]
                                             is not None]) for k in TISSUE_METRICS}}
                        for region in ("tissue", "glass")}
            elif split == "val" and pp_val[arm.key]:
                # fallback: the val-subsample per-patient file (last eval-artifact epoch)
                dcis_seed[arm.key] = {str(p): {str(s): (((doc.get("per_patient") or {}).get(str(p)) or {})
                                                        .get("per_class", {}).get(str(dcis)) or {}).get("recall")
                                               for s, doc in sorted(pp_val[arm.key].items())}
                                      for p in dcis_patients}
                dcis_seed[arm.key]["_source"] = (f"{PER_PATIENT_JSON}: val subsample at the last eval-artifact "
                                                 f"epoch (not necessarily the selected checkpoint)")
        sp.update({"per_patient": pts, "dcis_patient_recall_per_seed": dcis_seed, "confusion": conf,
                   "tissue_glass": tis if tis else None,
                   # for the per-class bars only (dropped from report.json: it is in per_arm already)
                   "_per_class_f1": {a.key: [scored[a.key][s].metrics["per_class_f1"] for s in sorted(scored[a.key])]
                                     for a in arms if source == "full" and scored[a.key]}})
        out["splits"][split] = sp
    return out


# -- cost ------------------------------------------------------------------------

def _find_number(obj: Any, keys: Sequence[str]) -> Optional[float]:
    """The first of `keys` found in a (nested) summary, as a float."""
    if isinstance(obj, Mapping):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                return float(v)
        for k in sorted(obj):
            v = _find_number(obj[k], keys)
            if v is not None:
                return v
    return None


def _stage_kind(name: str) -> str:
    if name.startswith("fit-") or name.startswith("generate") or name.startswith("sample"):
        return "generator"
    for k in ("qc", "export", "lopo", "report"):
        if k in name:
            return k
    return "data"


def _stage_source(name: str, rel_dir: str) -> Optional[str]:
    toks = set(re.split(r"[/_\-.]", f"{name}/{rel_dir}".lower()))
    if "cvae" in toks:
        return "cvae"
    if "ldm" in toks or "ae" in toks:
        return "ldm"
    return None


def _qc_scalars(summary: Any) -> Dict[str, Any]:
    out = {}
    if isinstance(summary, Mapping):
        for k in sorted(summary):
            v = summary[k]
            if QC_KEY_RE.search(str(k)) and isinstance(v, (int, float, str, bool)):
                out[str(k)] = v
    return out


def _cost(c: _Ctx, screen: Mapping[str, Any], seeds: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    aug = Path(c.cfg.aug_out)
    stages = []
    for p in sorted(aug.rglob("done.json")) if aug.is_dir() else []:
        d = c.inp.json(p)
        if not isinstance(d, Mapping) or d.get("status") != "ok":
            continue
        rel = p.parent.relative_to(aug).as_posix()
        name = str(d.get("stage", rel))
        kind = _stage_kind(name)
        wall = _nan_none((d.get("timings") or {}).get("wall_s"))
        outputs = d.get("outputs") or {}
        stages.append({
            "stage": name, "dir": rel, "kind": kind, "source": _stage_source(name, rel),
            "wall_min": wall / 60.0 if wall is not None else None,
            "gpu_min": wall / 60.0 if (wall is not None and kind == "generator") else None,
            "peak_vram_mb": _find_number(d.get("summary"), PEAK_VRAM_KEYS),
            "samples_per_s": _find_number(d.get("summary"), THROUGHPUT_KEYS),
            "output_bytes": int(sum(int((m or {}).get("size_bytes") or 0) for m in outputs.values())),
            "qc": _qc_scalars(d.get("summary")) if kind in ("qc", "export", "generator") else {},
        })
    decision = None
    dp = aug / DECISION_JSON
    if dp.is_file():
        doc = c.inp.json(dp)
        decision = next((doc[k] for k in DECISION_KEYS if isinstance(doc, Mapping)
                         and isinstance(doc.get(k), str)), None)
    per_source = {}
    for src in ("cvae", "ldm"):
        rows = [s for s in stages if s["source"] == src]
        if not rows:
            continue
        gm = [s["gpu_min"] for s in rows if s["gpu_min"] is not None]
        per_source[src] = {
            "gpu_h": sum(gm) / 60.0 if gm else None,
            "qc_wall_min": sum(s["wall_min"] or 0 for s in rows if s["kind"] == "qc") or None,
            "export_bytes": sum(s["output_bytes"] for s in rows if s["kind"] == "export") or None,
            "peak_vram_mb": max((s["peak_vram_mb"] for s in rows if s["peak_vram_mb"] is not None), default=None),
            "samples_per_s": next((s["samples_per_s"] for s in rows if s["samples_per_s"] is not None), None),
        }
    lopo = [s for s in stages if s["kind"] == "lopo"]

    def gen_of(arm: Arm) -> Optional[str]:
        return decision if arm.generator == "decision" else arm.generator

    screened = {r["arm"] for r in screen["rows"] if r.get("status") != "missing"}
    users = {src: sorted(a.key for a in c.arms if gen_of(a) == src and a.key in screened) for src in per_source}
    seeds_arms = set(seeds["arms"]) if seeds else set()
    arms_out = []
    for arm in c.arms:
        rd = find_run(c.index, arm.bench_tag, c.cfg.run_data, need="history.json")
        row: Dict[str, Any] = {"arm": arm.key, "bench_tag": arm.bench_tag, "bench_run": rd.name if rd else None}
        s_epoch = peak = None
        if rd is not None:
            c.read_run_config(rd, arm, "bench")
            hist = c.inp.json(rd / "history.json")
            times = [float(e["epoch_time"]) for e in hist if e.get("epoch_time") is not None]
            s_epoch = float(np.mean(times)) if times else None
            mem = [float(e["GPU_memory_MB"]) for e in hist if e.get("GPU_memory_MB") is not None]
            peak = max(mem) if mem else None
        screen_row = next((r for r in screen["rows"] if r["arm"] == arm.key), {})
        screen_epochs = (c.runs.get(screen_row.get("run"), {}) or {}).get("epochs") or 12
        n_seed_runs = len(c.cfg.seeds) if arm.key in seeds_arms else 0
        seed_epochs = 20
        for run in c.runs.values():
            if arm.key in run["arm"].split("+") and "seeds" in run["role"].split("+") and run.get("epochs"):
                seed_epochs = int(run["epochs"])
                break
        src = gen_of(arm)
        gen_h = per_source.get(src, {}).get("gpu_h") if src else None
        amort = gen_h / max(len(users.get(src, [])), 1) if gen_h is not None else None
        cls_screen = s_epoch * screen_epochs / 3600.0 if s_epoch is not None else None
        cls_seeds = s_epoch * seed_epochs * n_seed_runs / 3600.0 if s_epoch is not None else None
        total = None
        if cls_screen is not None:
            total = cls_screen + (cls_seeds or 0.0) + (amort or 0.0)
        row.update({"s_per_epoch": s_epoch, "peak_gpu_memory_mb": peak, "generator": src,
                    "generator_shared_by": users.get(src, []) if src else [],
                    "generator_gpu_h_amortised": amort, "classifier_gpu_h_screen": cls_screen,
                    "classifier_gpu_h_seeds": cls_seeds, "gpu_h_total": total,
                    "screen_epochs": screen_epochs, "seed_runs": n_seed_runs, "seed_epochs": seed_epochs})
        arms_out.append(row)
    return {"stages": stages, "per_source": per_source, "a4_generator": decision,
            "lopo_wall_min": sum(s["wall_min"] or 0 for s in lopo) if lopo else None,
            "arms": arms_out,
            "notes": ["GPU-minutes are the wall minutes of the generator stages (fit-*, generate*/sample*).",
                      "Classifier s/epoch comes only from the one-epoch solo `bench` runs (--jobs 1); "
                      "concurrent runs' wall clocks are not comparable.",
                      "Generator cost is amortised over the screened arms that use the generator."]}


def _consistency(c: _Ctx) -> None:
    trained = {n: r for n, r in sorted(c.runs.items()) if r["role"] != "bench"}
    kept = {n: (r.get("train_subsample") or {}).get("kept") for n, r in trained.items()}
    if kept:
        bad = {n: k for n, k in kept.items() if k != c.cfg.kept_expected}
        c.check("train_subsample_kept", not bad,
                f"every screen/seeds run keeps {c.cfg.kept_expected} rows/epoch" if not bad else f"differs: {bad}")
        emas = sorted({str(r.get("trm_ema_rate")) for r in trained.values()})
        c.check("trm_ema_rate_equal", len(emas) == 1, f"trm_ema_rate values: {emas}")
    for n, r in sorted(c.runs.items()):
        syn = r.get("synthetic")
        if not syn:
            continue
        prov = Path(c.cfg.aug_out) / "export" / str(syn.get("source")) / "provenance.json"
        if prov.is_file():
            cur = c.inp.json(prov).get("export_id")
            c.check(f"export_id:{n}", cur == syn.get("export_id"),
                    f"run export_id {syn.get('export_id')} vs current export {cur}")
        else:
            c.check(f"export_id:{n}", None, f"no export/{syn.get('source')}/provenance.json to compare against")


# ============================================================================
# figures (matplotlib's object API: no pyplot state, PNG without a software tag)
# ============================================================================

def _png(fig) -> bytes:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, metadata={"Software": None})
    return buf.getvalue()


def _forest(rows: Sequence[Tuple[str, Mapping[str, Any]]], title: str, threshold: float) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6.4, 1.4 + 0.7 * len(rows)))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    n = len(rows)
    for i, (label, d) in enumerate(rows):
        yv = n - 1 - i
        pts = [float(v) for _, v in sorted(d["per_seed"].items(), key=lambda kv: int(kv[0]))]
        ax.scatter(pts, [yv] * len(pts), s=22, color="0.45", zorder=2, label="seed" if i == 0 else None)
        if d.get("ci95"):
            ax.hlines(yv, d["ci95"][0], d["ci95"][1], color="black", lw=2, zorder=3)
        if d.get("mean") is not None:
            ax.plot([d["mean"]], [yv], marker="D", color="black", ms=7, zorder=4,
                    label="mean, 95% t-CI" if i == 0 else None)
    ax.axvline(0.0, color="0.6", lw=1)
    ax.axvline(threshold, color="tab:red", ls="--", lw=1, label=f"minimum effect +{threshold}")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Delta macro-F1 vs A0 (paired by seed)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return _png(fig)


def _class_bars(series: Sequence[Tuple[str, Sequence[float], Sequence[float]]], class_names: Sequence[str],
                title: str) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6.4, 3.6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    k, m = len(class_names), max(len(series), 1)
    width = 0.8 / m
    for j, (label, means, sds) in enumerate(series):
        xs = [i - 0.4 + width * (j + 0.5) for i in range(k)]
        ax.bar(xs, [v if v is not None else 0.0 for v in means], width,
               yerr=[v if v is not None else 0.0 for v in sds], capsize=3, label=label)
    ax.set_xticks(range(k))
    ax.set_xticklabels(list(class_names))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=min(m, 4), loc="lower right")
    fig.tight_layout()
    return _png(fig)


def _figures(c: _Ctx, screen: Mapping[str, Any], seeds: Optional[Mapping[str, Any]]) -> Dict[str, bytes]:
    figs: Dict[str, bytes] = {}
    names = c.class_names
    ser = []
    for r in screen["rows"]:
        be = r.get("best_epoch") or {}
        if be.get("per_class_f1"):
            ser.append((r["arm"], be["per_class_f1"], [0.0] * len(names)))
    if ser:
        figs["per_class_f1_screen.png"] = _class_bars(
            ser, names, f"Screen (seed {c.cfg.screen_seed}): val-subsample per-class F1 at the selected epoch")
    if seeds:
        for split, sp in seeds["splits"].items():
            rows = [(k, d["f1_macro"]) for k, d in sp["deltas_vs_A0"].items() if d["f1_macro"]["n"]]
            if rows:
                figs[f"forest_delta_f1_macro_{split}.png"] = _forest(
                    rows, f"{split}: per-seed Delta macro-F1 vs A0", stats.MIN_EFFECT_DELTA)
            ser = []
            for arm, vals in sp.get("_per_class_f1", {}).items():
                a = np.asarray(vals, dtype=np.float64)
                sd = a.std(axis=0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1])
                ser.append((arm, a.mean(axis=0).tolist(), sd.tolist()))
            if ser:
                figs[f"per_class_f1_{split}.png"] = _class_bars(
                    ser, names, f"{split}: per-class F1, mean +- sd over seeds (all rows)")
    return figs


# ============================================================================
# markdown
# ============================================================================

def _f(x: Any, nd: int = 4, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _pm(ms: Mapping[str, Any], nd: int = 4) -> str:
    if not ms or ms.get("mean") is None:
        return "-"
    return f"{ms['mean']:.{nd}f} +- {ms['sd']:.{nd}f}" if ms.get("sd") is not None else f"{ms['mean']:.{nd}f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _markdown(doc: Mapping[str, Any]) -> str:
    L: List[str] = ["# HSI augmentation study: report", ""]
    L += [f"Built from artifacts only by `scripts/augmentation_report.py` (git `{doc['git_sha']}`); it "
          f"regenerates byte-identically from the same inputs. Schema {doc['schema_version']}.", ""]
    seal = doc["test_seal"]
    L += [f"**Test columns: {'shown' if seal['revealed'] else 'sealed'}.** {seal['detail']}", ""]
    L += ["## Notes", ""] + [f"- {n}" for n in doc["notes"]] + [""]

    sc = doc["screen"]
    L += [f"## Screen (seed {sc['seed']}, val only)", "", f"Rule: {sc['rule']}.", ""]
    L += _table(["arm", "family", "tag", "run", "score (val F1, ep " + "-".join(map(str, sc["score_epochs"])) + ")",
                 "Delta vs A0", "DCIS F1", "rank", "best ep", "val F1 (best ep)", "bal. acc. (best ep)",
                 "DCIS P/R (best ep)"],
                [[r["arm"], r["family"], f"`{r['tag']}`", r.get("run") or r["status"], _f(r.get("score")),
                  _f(r.get("delta_score_vs_A0"), signed=True), _f(r.get("dcis_f1")), r.get("family_rank", "-"),
                  (r.get("best_epoch") or {}).get("epoch", "-"), _f((r.get("best_epoch") or {}).get("f1_macro")),
                  _f((r.get("best_epoch") or {}).get("balanced_accuracy")),
                  f"{_f((r.get('best_epoch') or {}).get('dcis_precision'), 3)}/"
                  f"{_f((r.get('best_epoch') or {}).get('dcis_recall'), 3)}"] for r in sc["rows"]])
    L += ["", "Screen metrics are on the stratified val subsample (history.json); a single seed, no statistics."]
    if sc.get("selection"):
        L += ["", "Selection (selection.json):", ""]
        L += _table(["arm", "family", "score", "gap to family best", "within tie gap", "selection.json score"],
                    [[s["arm"], s["family"], _f(s["score"]), _f(s["gap_to_family_best"]), s["within_tie_gap"],
                      _f(s["selection_json_score"])] for s in sc["selection"]])
    L.append("")

    sd = doc["seeds"]
    if sd is None:
        L += ["## Seeds stage", "", "Not available: no selection.json yet (the seeds stage runs only after the "
              "selection).", ""]
    else:
        L += ["## Seeds stage", "", f"Arms: {', '.join(sd['arms'])}; seeds {sd['seeds']}.", ""]
        missing = [f"{r['arm']} s{r['seed']} (`{r['tag']}`)" for r in sd["runs"] if r["run"] is None]
        if missing:
            L += ["Missing runs: " + ", ".join(missing), ""]
        for split, sp in sd["splits"].items():
            L += [f"### {split}", "", f"Source: {sp['source']}. {sp['n_rows']:,} rows. DCIS patient(s) in "
                  f"{split}: {sp['dcis_patients']}.", ""]
            keys = [m for m in METRICS if m[0] in next(iter(sp["per_arm"].values()))["summary"]]
            L += ["Mean +- sd over seeds:", ""]
            L += _table(["arm", "n"] + [lab for _, lab, _ in keys],
                        [[a, pa["summary"][keys[0][0]]["n"]] + [_pm(pa["summary"][k]) for k, _, _ in keys]
                         for a, pa in sp["per_arm"].items()])
            L += ["", "Paired per-seed deltas vs A0 (mean, sd, 95% t-CI, seeds improved):", ""]
            rows = []
            for a, dd in sp["deltas_vs_A0"].items():
                for k, lab, _ in keys:
                    d = dd[k]
                    ci = d["ci95"]
                    rows.append([a, lab, _f(d["mean"], signed=True), _f(d["sd"]),
                                 f"[{_f(ci[0], signed=True)}, {_f(ci[1], signed=True)}]" if ci else "-",
                                 d["sign_count"], " ".join(f"s{s}:{_f(v, signed=True)}"
                                                           for s, v in d["per_seed"].items())])
            L += _table(["arm", "metric", "mean Delta", "sd", "95% t-CI", "improved", "per seed"], rows)
            L += [""]
            for a, dec in sp["decisions"].items():
                me, wx = dec["minimum_effect"], dec["wilcoxon"]
                verdict = {True: "MET", False: "NOT MET", None: "INCOMPLETE"}[me["met"]]
                L += [f"- **{a}**: minimum effect {verdict} ({me['reason']}). Wilcoxon: "
                      + (f"p = {wx['p']}" if wx["reported"] else wx["note"]) + "."]
            L += ["", f"Rule: {stats.MIN_EFFECT_RULE}. {stats.WILCOXON_RULE}.", ""]
            if sp["patient_bootstrap"]:
                L += ["Patient bootstrap of Delta (pooled over paired seeds, labels fixed to every class):", ""]
                rows = []
                for a, bm in sp["patient_bootstrap"].items():
                    for m, b in bm.items():
                        ci = b["ci95"]
                        rows.append([a, m, _f(b["observed"], signed=True),
                                     f"[{_f(ci[0], signed=True)}, {_f(ci[1], signed=True)}]" if ci else "-",
                                     _f(b["frac_positive"], 3), f"{b['n_boot_used']}/{b['n_boot_requested']}",
                                     f"{b['n_resamples_missing_a_class']} ({b['frac_resamples_missing_a_class']:.1%})",
                                     ", ".join(f"{k}:{v}" for k, v in b["missing_by_class"].items())])
                L += _table(["arm", "metric", "observed", "95% CI", "P(Delta>0)", "resamples used",
                             "resamples missing a class (skipped)", "missing by class"], rows)
                L += ["", f"DCIS: {sp['dcis_ci']['reason']}.", ""]
            if sp["dcis_patient_recall_per_seed"]:
                L += ["DCIS recall of the DCIS patient(s), per seed:", ""]
                rows = []
                for a, per in sp["dcis_patient_recall_per_seed"].items():
                    note = per.get("_source")
                    for p, ss in per.items():
                        if p == "_source":
                            continue
                        rows.append([a, p, " ".join(f"s{s}:{_f(v, 3)}" for s, v in ss.items()),
                                     _pm(stats.mean_sd([v for v in ss.values() if v is not None]), 3),
                                     note or "all rows"])
                L += _table(["arm", "patient", "per seed", "mean +- sd", "source"], rows) + [""]
            if sp["per_patient"]:
                L += ["Per patient (mean +- sd over seeds):", ""]
                rows = []
                for a, pp in sp["per_patient"].items():
                    for p, v in pp.items():
                        rows.append([a, p, v["n"], _pm(v["macro_recall"], 3), _pm(v["dcis_recall"], 3)])
                L += _table(["arm", "patient", "rows", "macro recall", "DCIS recall"], rows) + [""]
            if sp["confusion"]:
                L += ["Confusion matrices, summed over seeds (rows true, columns predicted; "
                      f"{', '.join(doc['class_names'])}):", ""]
                for a, cmd in sp["confusion"].items():
                    L += [f"- {a} ({cmd['n_seeds']} seeds): {cmd['summed_over_seeds']}"]
                L += [""]
            if sp["tissue_glass"]:
                L += ["Tissue vs glass (mean +- sd over seeds):", ""]
                rows = []
                for a, regs in sp["tissue_glass"].items():
                    for reg, v in regs.items():
                        rows.append([a, reg, v["n"], v["per_class_support"]]
                                    + [_pm(v[k], 3) for k in TISSUE_METRICS])
                L += _table(["arm", "rows", "n", "per class"] + list(TISSUE_METRICS), rows) + [""]
            elif split in ("val", "test"):
                L += ["Tissue vs glass: not available (run `augment_pipeline.py report-tissue` and keep full "
                      "predictions).", ""]

    cost = doc["cost"]
    L += ["## Cost", ""] + [f"- {n}" for n in cost["notes"]] + [""]
    if cost["stages"]:
        L += _table(["stage", "dir", "kind", "source", "wall min", "GPU min", "peak VRAM MB", "samples/s",
                     "outputs MB"],
                    [[s["stage"], s["dir"], s["kind"], s["source"] or "-", _f(s["wall_min"], 1),
                      _f(s["gpu_min"], 1), _f(s["peak_vram_mb"], 0), _f(s["samples_per_s"], 1),
                      _f(s["output_bytes"] / 2 ** 20, 1)] for s in cost["stages"]]) + [""]
    else:
        L += ["No stage done.json found under the pipeline output.", ""]
    L += [f"A4 generator (lopo_utility/decision.json): {cost['a4_generator'] or '-'}. "
          f"LOPO wall: {_f(cost['lopo_wall_min'], 1)} min.", ""]
    L += _table(["arm", "bench run", "s/epoch", "peak GPU MB", "generator", "gen GPU-h (amortised)",
                 "classifier GPU-h screen", "classifier GPU-h seeds", "GPU-h total"],
                [[a["arm"], a["bench_run"] or "-", _f(a["s_per_epoch"], 1), _f(a["peak_gpu_memory_mb"], 0),
                  a["generator"] or "-", _f(a["generator_gpu_h_amortised"], 2), _f(a["classifier_gpu_h_screen"], 2),
                  _f(a["classifier_gpu_h_seeds"], 2), _f(a["gpu_h_total"], 2)] for a in cost["arms"]]) + [""]
    qc = [s for s in cost["stages"] if s["qc"]]
    if qc:
        L += ["## QC summary", ""] + [f"- {s['stage']} ({s['dir']}): "
                                      + ", ".join(f"{k}={v}" for k, v in s["qc"].items()) for s in qc] + [""]

    L += ["## Checks", ""]
    L += _table(["check", "ok", "detail"], [[k["check"], k["ok"], k["detail"]] for k in doc["checks"]]) \
        if doc["checks"] else ["(none)"]
    L += ["", "## Inputs", "", f"{len(doc['inputs_sha256'])} input files; sha256 of each in report.json "
          f"(`inputs_sha256`). Runs used:", ""]
    L += _table(["run", "arm", "role", "tag", "seed", "epochs", "kept", "trm_ema_rate", "export_id"],
                [[n, r["arm"], r["role"], f"`{r['tag']}`", r["seed"], r["epochs"],
                  (r.get("train_subsample") or {}).get("kept"), r.get("trm_ema_rate"),
                  (r.get("synthetic") or {}).get("export_id", "-")] for n, r in sorted(doc["runs"].items())])
    L += ["", "Figures: " + (", ".join(f"`{FIGURES_DIR}/{n}`" for n in doc["figures"]) or "none"), ""]
    return "\n".join(L)


# ============================================================================
# entry points
# ============================================================================

def _clean(obj: Any) -> Any:
    """JSON-ready: numpy -> Python, NaN/inf -> None, tuples -> lists."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def build_report(cfg: ReportConfig) -> Tuple[Dict[str, Any], str, Dict[str, bytes]]:
    """(report.json document, report.md text, {figure name: PNG bytes}).
    Reads artifacts only and writes nothing."""
    from hsi_augment.stage import git_sha
    sel_path = Path(cfg.selection)
    roots = {"experiments": cfg.experiments, "data": cfg.data_dir, "aug": cfg.aug_out,
             "selection": sel_path.parent}
    if cfg.arms_file:
        roots["arms"] = Path(cfg.arms_file).parent
    sealed_reason = None
    if not sel_path.is_file():
        sealed_reason = f"no selection.json at {cfg.selection} yet"
    elif not cfg.reveal_test:
        sealed_reason = "--reveal_test was not passed"
    inp = Inputs(roots, reveal_test=sealed_reason is None)
    arms = DEFAULT_ARMS
    if cfg.arms_file:
        arms = arms_from_doc(inp.json(cfg.arms_file), Path(cfg.arms_file).name)
    selection, arms = _load_selection(inp, sel_path, arms)
    c = _Ctx(cfg=cfg, inp=inp, arms=arms, index=index_runs(cfg.experiments), sealed_reason=sealed_reason)

    screen = _screen(c, selection)
    seeds = _seeds(c, selection) if selection is not None else None
    cost = _cost(c, screen, seeds)
    _consistency(c)
    figs = _figures(c, screen, seeds)

    doc = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "git_sha": git_sha(),
        "report_config": cfg.content_dict(),
        "test_seal": {"revealed": sealed_reason is None,
                      "detail": ("selection.json exists and --reveal_test was passed: test columns are shown."
                                 if sealed_reason is None else f"Test is sealed: {sealed_reason}. Test columns "
                                 "render only when selection.json exists AND --reveal_test is passed.")},
        "notes": [SEED42_NOTE, VAL_NOTE, DCIS_NOTE, TISSUE_NOTE],
        "class_names": list(c.class_names),
        "arms": [asdict(a) for a in arms],
        "selection": selection["selected"] if selection else None,
        "screen": screen, "seeds": _strip_private(seeds), "cost": cost,
        "checks": c.checks, "runs": c.runs,
        "figures": sorted(figs),
        "inputs_sha256": dict(sorted(inp.sha256.items())),
    }
    doc = _clean(doc)
    return doc, _markdown(doc), figs


def _strip_private(seeds: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if seeds is None:
        return None
    out = dict(seeds)
    out["splits"] = {s: {k: v for k, v in sp.items() if not k.startswith("_")} for s, sp in seeds["splits"].items()}
    return out


def write_report(doc: Mapping[str, Any], markdown: str, figures: Mapping[str, bytes], out_dir: PathLike) -> Path:
    """Write report.md, report.json and figures/ atomically; drop stale PNGs."""
    out = Path(out_dir)
    (out / FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    def atomic(path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    for name, data in sorted(figures.items()):
        atomic(out / FIGURES_DIR / name, data)
    for stale in sorted((out / FIGURES_DIR).glob("*.png")):
        if stale.name not in figures:
            stale.unlink()
    atomic(out / REPORT_JSON, (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    atomic(out / REPORT_MD, (markdown.rstrip("\n") + "\n").encode())
    return out


def generate_report(cfg: ReportConfig) -> Path:
    """Build and write `<out_root>/<date>/`; returns that directory."""
    doc, md, figs = build_report(cfg)
    return write_report(doc, md, figs, Path(cfg.out_root) / cfg.resolved_date())
