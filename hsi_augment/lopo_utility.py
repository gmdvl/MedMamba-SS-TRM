# -*- coding: utf-8 -*-
"""
hsi_augment/lopo_utility.py
===========================
Phase 8c (#30): the LOPO utility test, `augment_pipeline.py lopo-utility`
(registered in `hsi_augment/stages/lopo_utility.py`). On train data only, and
before any GPU arm of the screen runs, it asks the plan's honest question: does
generative interpolation beat copying on an UNSEEN DCIS patient? It also
decides A4's generator:

    <out>/lopo_utility/decision.json    {"a4_source": "cvae" | "ldm", ...}   (schema below)

`sweeps/augmentation_pipeline.py` (#33) and `hsi_augment/report.py` (#34) read
A4's generator from it: the first string among the keys `a4_source`, `source`,
`generator` and `chosen`. This file writes `a4_source` and never writes the
other three at the top level.

The test
--------
In every LOPO fold k of `prep/leakage/folds.json` (fold k holds out DCIS train
patient k plus one healthy and one IDC companion, with all their rows), a
mean/std logistic regression is fitted on four training sets. Each is scored on
the fold's held-out patients:

    real     the fold's train-side real rows
    copies   real + count-matched copies of the fold's train-side target-class rows
    cvae     real + QC-accepted samples of fold k's CVAE (cvae-lopo)
    ldm      real + QC-accepted samples of fold k's LDM (ldm-lopo)

Real rows. The generator pool (`pool/`): every train tissue row of the target
class, plus a patient-balanced sample of the other classes. These are the rows
the fold generators were trained on. A fold's train and held-out sides are
`splits.fold_masks` over `patient_pool.npy`. The pool decides WHICH rows are
used, and the train split supplies their VALUES: X_train is read at
`row_pool.npy` (purpose "lopo"). The pool stores the same patches log-z encoded
in float16. Reading the build's own float16 value-space patches avoids a
float16 log-z round trip and gives exactly what the classifier would see.
`pool_io.check_against_train` first checks that the pool belongs to this
build's train split.

Synthetic rows. They come from fold k's generator, the LOPO fit that never saw
fold k's held-out patients, and are sampled at the tau_s / w that its source's
generate stage chose (`synth/<src>/choice.json`; generation.json must agree).
The sampler is `generate.GeneratorHandle`, the same one `generate-<src>` uses:
the noise streams of `train_generative.noise_rows`, plus the LDM's DDIM steps
and batch size from that stage's generation.json. The generator's own codec
decodes the samples into the build's value space, as float16 (the export's
dtype), so real and synthetic rows are featurised in one space. QC is R1-R6
against fold k's references (`qc.apply_rules(fold=k)` via
`generate.score_in_chunks`). The set is the first n_add accepted samples in
generation order (`generate.select_accepted`), out of at most
`overgenerate` x n_add generated. Sampling stops as soon as n_add samples are
accepted, which gives the same selection with less work. When too few samples
are accepted, or acceptance is below `min_acceptance`, the stage fails loudly
(AcceptanceError, no decision.json). Each fold generator must have been trained
on THIS pool (its record's pool_sha256) and must hold out exactly the fold's
patients.

Count-matched copies (`COUNT_MATCHING_RULE`)
--------------------------------------------
For fold k and target class c:

    n_train(k, c) = the fold's train-side pool rows of class c
    n_add(k, c)   = max(1, round(syn_ratio x n_train(k, c))), capped at max_added_rows

The default is 1:1, the ratio of arms A2/A3, which add as many synthetic DCIS
rows as train has real ones. Copies: with q, r = divmod(n_add, n_train), every
train-side class-c row is copied q times, and a seeded draw of r distinct rows
(`ctx.rng("copies", k, c)`) is copied once more. That is exactly n_add rows,
and each real row is copied q or q + 1 times. Both generator sets add exactly
n_add accepted samples (otherwise the stage fails). So the three augmented sets
have the same size and class mix, and the comparison isolates generation from
oversampling.

The classifier
--------------
`qc.consistency.fit_probe`, i.e. `scripts.shallow_probe_heldout.fit`: a
class_weight="balanced" multinomial LogisticRegression, max_iter 1500. Its
input is `provenance.patch_feats`, which is the frozen
`scripts.shallow_baseline.featurize(x, "mean_std")`: per band, the mean and std
over the H x W pixels. Real and synthetic patches are featurised alike. One
standardisation per fold, the mean and std of the fold's real train-side
features, is applied to all four sets of that fold. It is fitted on train-side
rows only, and the four sets differ only in their added rows. (The probe script
also z-scores before it featurises.) Under class-balanced weights, copies change
little beyond the effective regularisation. A generator that beats them adds
information, not weight: that is what the control measures.

Metrics on the fold's held-out pool rows (`score_heldout`): per class, the
precision, recall and F1 (F1 = 2TP / (2TP + FP + FN); None for a class with no
held-out rows). For each target class, its recall and F1: the positives are the
held-out DCIS patient's rows, and the companions' rows are the negatives. Also
the macro-F1 over the classes present in the held-out rows (the fold
macro-F1), the balanced accuracy and the confusion matrix.

The decision (`DECISION_RULE`, `TIE_BREAK`)
-------------------------------------------
The decision folds are the folds on which all four sets were scored, i.e. where
both generators have a LOPO fit. ldm-lopo fits folds 0 and 1 by default;
cvae-lopo fits every fold. score(g) = the mean, over the decision folds, of set
g's target-class F1 (averaged over the target classes when there are several).
With DCIS as the only target, this is the mean LOPO DCIS F1. A4's source is
the argmax. A tie, |score(cvae) - score(ldm)| <= `tie_tolerance` (default 0:
exact equality), goes to cvae, the cheaper generator: one network and one
decoder pass per sample, against an autoencoder plus DDIM steps. This is the
plan's rule, and it does not depend on beating copies; `vs_copies` and
`vs_real` report that separately.

decision.json (`DECISION_SCHEMA_VERSION` 1; `check_decision` validates it)
--------------------------------------------------------------------------
    {"schema_version": 1, "a4_source": "cvae" | "ldm",
     "rule": DECISION_RULE, "tie_break": TIE_BREAK, "tie_tolerance": 0.0,
     "decision": {"metric": "target_f1", "scores": {"cvae": .., "ldm": ..}, "ldm_minus_cvae": ..,
                  "tie": bool, "contenders": [..], "a4_source": ..},
     "decision_folds": [0, 1, ...], "n_folds": 5, "target_classes": ["DCIS"],
     "training_sets": ["real", "copies", "cvae", "ldm"], "set_docs": {set: text},
     "per_fold": [{"fold", "dcis_patient", "heldout_patients", "n_heldout", "heldout_rows_per_class",
                   "n_train_real", "train_rows_per_class", "n_added": {class: n},
                   "copies": {class: {"n_source_rows", "n_added", "repeats", "extra_draw"}},
                   "synthetic": {src: {class: {"n_generated", "n_accepted", "acceptance", "n_selected",
                                               "rejected_by", "seed", "qc_fold", ...}}},
                   "sets": {set: {"n_train", "converged", "target_f1", "target_recall", "macro_f1",
                                  "macro_f1_classes", "balanced_accuracy", "accuracy",
                                  "targets": {class: {"n", "precision", "recall", "f1"}},
                                  "per_class": {...}, "confusion": [[..]]}}}, ...],
     "mean":     {set: {"folds", "n_folds", "target_f1", "target_recall", "macro_f1",
                        "balanced_accuracy", "per_target": {class: {"f1", "recall"}},
                        "per_fold_target_f1": {fold: ..}}},      # over the decision folds
     "mean_all": {set: {...}},                                   # every fold the set was scored on
     "vs_copies": {src: {"mean_delta_target_f1", "per_fold", "better", "equal", "worse", "beats"}},
     "vs_real":   {src: {...}},
     "count_matching": COUNT_MATCHING_RULE, "syn_ratio": 1.0, "max_added_rows": null,
     "classifier": {"model", "features", "standardisation", "max_iter", "seed"},
     "qc": {"rule", "overgenerate", "min_acceptance"},
     "generators": {src: {"param", "value", "choice_source", "choice_sha256", "generation_sha256",
                          "final_checkpoint_sha256", "lopo_stage", "lopo_fingerprint",
                          "generate_fingerprint", "sample_steps", "batch_size",
                          "folds": {fold: {"checkpoint", "checkpoint_sha256", "pool_sha256",
                                           "codec_sha256", "heldout_patients"}}}},
     "provenance": {"folds_sha256", "folds_seed", "pool_fingerprint", "pool_stage_fingerprint",
                    "qc_reference_fingerprint", "thresholds_sha256", "input_manifest_sha256",
                    "upstream": {stage: fingerprint}, "seed"},
     "hybridsn": {"run": false, "note": ..}}

decision.json holds no wall-clock time (done.json has the timings), so a
re-run on the same inputs writes the same bytes. Any old decision.json is
deleted when the stage starts, so a failed re-run leaves none for the sweep to
read.

HybridSN
--------
The plan's optional repeat with HybridSN (`hsi_baselines.build_baseline`,
about 1 GPU-h) is NOT implemented. It would need a patch-level training loop
per (fold, set): 4 x n_folds GPU fits, with their own data loading. That is a
separate piece of work, beyond this train-only CPU test. decision.json records
`hybridsn.run = false`.

Split access and memory
-----------------------
The stage uses the purpose "lopo" only, so only train is read: its y / groups
/ captures (to check the pool), and its X, for the real patches and for the
QC reference's R6 re-rank (passed in as `train_X`, so the reference opens
nothing itself). The SplitGuard refuses val and test, and the access log shows
train only. The real features are n_pool x 2C float32 (~93 MB for trainsel's
~363k pool rows). X is read `chunk_rows` rows at a time. Synthetic patches are
generated in batches, then QC'd and featurised, and only their features are
kept. Neither X_train nor a generated set is ever materialised.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hsi_augment import generate as G
from hsi_augment.io import iter_chunks, sha256_file
from hsi_augment.stage import StageError, read_json

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.pool_io import Pool
    from hsi_augment.qc.reference import QCReference
    from hsi_augment.stage import StageContext

PURPOSE = "lopo"
DECISION_NAME = "decision.json"
DECISION_SCHEMA_VERSION = 1

SOURCES: Tuple[str, ...] = G.SOURCES                       # ("cvae", "ldm")
TRAINING_SETS: Tuple[str, ...] = ("real", "copies", *SOURCES)
TIE_ORDER: Tuple[str, ...] = ("cvae", "ldm")               # cheaper generator first: ties go to it
DECISION_METRIC = "target_f1"
# The readers' keys (sweeps/augmentation_pipeline.py, hsi_augment/report.py):
# only `a4_source` is written; the others must never hold a string at the top level.
READER_KEYS: Tuple[str, ...] = ("a4_source", "source", "generator", "chosen")
MIN_STD = 1e-6                                             # as qc.features.MIN_STD
STANDARDISE_BLOCK = 65_536                                 # rows standardised at a time

SET_DOCS: Dict[str, str] = {
    "real": "the fold's train-side real pool rows",
    "copies": "real + n_add copies of the fold's train-side target-class rows (count-matched to the generators)",
    "cvae": "real + the first n_add QC-accepted samples of fold k's CVAE (cvae-lopo) at generate-cvae's tau_s",
    "ldm": "real + the first n_add QC-accepted samples of fold k's LDM (ldm-lopo) at generate-ldm's w",
}
DECISION_RULE = ("A4's source is the generator whose set (real + its fold generator's QC-accepted samples) has "
                 "the higher mean target-class F1 on the held-out patients over the decision folds (the LOPO "
                 "folds on which all four training sets were scored); with DCIS as the only target class this "
                 "is the mean LOPO DCIS F1")
TIE_BREAK = ("a tie, |mean F1 cvae - mean F1 ldm| <= tie_tolerance (default 0: exact equality), goes to cvae, "
             "the cheaper generator (one network and one decoder pass per sample; the LDM needs an autoencoder "
             "and DDIM steps)")
COUNT_MATCHING_RULE = ("per fold k and target class c: n_add = max(1, round(syn_ratio x n_train(k, c))), capped "
                       "at max_added_rows, where n_train(k, c) is the fold's train-side pool rows of class c. "
                       "copies: every train-side class-c row floor(n_add / n_train) times plus a seeded draw "
                       "(ctx.rng('copies', k, c)) of (n_add mod n_train) distinct rows once more, exactly n_add "
                       "rows; cvae / ldm: exactly n_add QC-accepted samples (the first n_add in generation "
                       "order), so the three augmented sets have the same size and class mix")
QC_RULE = ("R1-R6 against fold k's QC references (qc.apply_rules(fold=k)); the first n_add accepted samples in "
           "generation order, from at most overgenerate x n_add generated; fewer accepted, or an acceptance "
           "below min_acceptance, fails the stage")
CLASSIFIER_DOC = {
    "model": "scripts.shallow_probe_heldout.fit via hsi_augment.qc.consistency.fit_probe: LogisticRegression, "
             "class_weight='balanced' (multinomial, lbfgs)",
    "features": "hsi_augment.provenance.patch_feats = scripts.shallow_baseline.featurize(x, 'mean_std') on the "
                "float16 value-space patches (real: X_train at row_pool; synthetic: the generator's decoded output)",
    "standardisation": "per fold, mean and std of the fold's real train-side features, applied to all four sets",
}
HYBRIDSN_NOTE = ("the plan's optional HybridSN repeat (hsi_baselines.build_baseline, about 1 GPU-h) is not "
                 "implemented: it needs a patch-level GPU training loop per (fold, set)")


# ============================================================================
# count matching
# ============================================================================

def added_rows(n_train: int, ratio: float, cap: Optional[int] = None) -> int:
    """n_add for a fold's target class with `n_train` train-side rows
    (COUNT_MATCHING_RULE). 0 only when there is no train-side row."""
    n = int(n_train)
    if n <= 0:
        return 0
    k = max(1, int(math.floor(float(ratio) * n + 0.5)))
    return min(k, int(cap)) if cap is not None else k


def copy_positions(n_source: int, n_add: int, rng: np.random.Generator) -> np.ndarray:
    """Positions (into the `n_source` rows to copy) of exactly `n_add` copies:
    every position floor(n_add / n_source) times, plus a seeded draw of
    n_add mod n_source distinct positions once more. Sorted, int64."""
    n_source, n_add = int(n_source), int(n_add)
    if n_add < 0:
        raise ValueError(f"n_add must be >= 0, got {n_add}")
    if n_add == 0:
        return np.zeros(0, dtype=np.int64)
    if n_source <= 0:
        raise ValueError(f"cannot copy {n_add} rows from none")
    q, r = divmod(n_add, n_source)
    extra = np.sort(rng.choice(n_source, size=r, replace=False)) if r else np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate([np.tile(np.arange(n_source, dtype=np.int64), q),
                                   extra.astype(np.int64)]), kind="stable")


# ============================================================================
# metrics
# ============================================================================

def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(v)) if v else None


def score_heldout(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str],
                  targets: Sequence[int]) -> Dict[str, Any]:
    """Per-class precision / recall / F1, the target classes' recall and F1,
    the macro-F1 over the classes present in `y_true`, balanced accuracy and
    the confusion matrix (rows: truth, columns: prediction)."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    k = len(class_names)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} differ")
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    per: Dict[str, Dict[str, Any]] = {}
    for c, name in enumerate(class_names):
        tp = int(cm[c, c])
        n = int(cm[c].sum())
        pred = int(cm[:, c].sum())
        fn, fp = n - tp, pred - tp
        per[str(name)] = {"n": n, "predicted": pred,
                          "precision": tp / pred if pred else None,
                          "recall": tp / n if n else None,
                          "f1": 2 * tp / (2 * tp + fp + fn) if n else None}
    present = [str(class_names[c]) for c in range(k) if cm[c].sum() > 0]
    tnames = [str(class_names[int(c)]) for c in targets]
    return {"n": int(yt.size),
            "target_f1": _mean([per[t]["f1"] for t in tnames]),
            "target_recall": _mean([per[t]["recall"] for t in tnames]),
            "macro_f1": _mean([per[n]["f1"] for n in present]),
            "macro_f1_classes": present,
            "balanced_accuracy": _mean([per[n]["recall"] for n in present]),
            "accuracy": float((yt == yp).mean()) if yt.size else None,
            "targets": {t: {m: per[t][m] for m in ("n", "precision", "recall", "f1")} for t in tnames},
            "per_class": per, "confusion": cm.tolist()}


def _aggregate(per_fold: Sequence[Mapping[str, Any]], set_name: str, folds: Sequence[int],
               target_names: Sequence[str]) -> Dict[str, Any]:
    recs = {int(f["fold"]): f["sets"][set_name] for f in per_fold
            if int(f["fold"]) in set(folds) and set_name in f["sets"]}
    rs = [recs[k] for k in sorted(recs)]
    return {"folds": sorted(recs), "n_folds": len(recs),
            "target_f1": _mean([r["target_f1"] for r in rs]),
            "target_recall": _mean([r["target_recall"] for r in rs]),
            "macro_f1": _mean([r["macro_f1"] for r in rs]),
            "balanced_accuracy": _mean([r["balanced_accuracy"] for r in rs]),
            "per_target": {t: {"f1": _mean([r["targets"][t]["f1"] for r in rs]),
                               "recall": _mean([r["targets"][t]["recall"] for r in rs])} for t in target_names},
            "per_fold_target_f1": {str(k): recs[k]["target_f1"] for k in sorted(recs)}}


def summarize(per_fold: Sequence[Mapping[str, Any]], target_names: Sequence[str],
              sets: Sequence[str] = TRAINING_SETS) -> Dict[str, Any]:
    """The decision folds (every set scored), the per-set means over them
    (`mean`) and over every fold each set was scored on (`mean_all`)."""
    decision_folds = sorted(int(f["fold"]) for f in per_fold if all(s in f["sets"] for s in sets))
    scored = {s: sorted(int(f["fold"]) for f in per_fold if s in f["sets"]) for s in sets}
    return {"decision_folds": decision_folds,
            "mean": {s: _aggregate(per_fold, s, decision_folds, target_names) for s in sets},
            "mean_all": {s: _aggregate(per_fold, s, scored[s], target_names) for s in sets}}


def compare(per_fold: Sequence[Mapping[str, Any]], a: str, b: str, folds: Sequence[int]) -> Dict[str, Any]:
    """Set `a` against set `b` on `folds`: per-fold and mean target-F1 deltas (a - b)."""
    deltas: Dict[str, Optional[float]] = {}
    for f in per_fold:
        k = int(f["fold"])
        if k in set(folds) and a in f["sets"] and b in f["sets"]:
            fa, fb = f["sets"][a]["target_f1"], f["sets"][b]["target_f1"]
            deltas[str(k)] = None if fa is None or fb is None else float(fa) - float(fb)
    d = [v for v in deltas.values() if v is not None]
    mean = _mean(d)
    return {"mean_delta_target_f1": mean, "per_fold": deltas, "better": int(sum(v > 0 for v in d)),
            "equal": int(sum(v == 0 for v in d)), "worse": int(sum(v < 0 for v in d)),
            "beats": bool(mean is not None and mean > 0)}


def decide(scores: Mapping[str, Optional[float]], tie_tolerance: float = 0.0) -> Dict[str, Any]:
    """A4's source from the generators' mean target F1 (DECISION_RULE,
    TIE_BREAK). ValueError when a generator has no finite score."""
    if not float(tie_tolerance) >= 0.0:
        raise ValueError(f"tie_tolerance must be >= 0, got {tie_tolerance!r}")
    vals: Dict[str, float] = {}
    for s in TIE_ORDER:
        v = scores.get(s)
        if v is None or not math.isfinite(float(v)):
            raise ValueError(f"no mean {DECISION_METRIC} for {s!r} ({v!r})")
        vals[s] = float(v)
    best = max(vals.values())
    contenders = [s for s in TIE_ORDER if vals[s] >= best - float(tie_tolerance)]
    return {"metric": DECISION_METRIC, "scores": vals, "ldm_minus_cvae": vals["ldm"] - vals["cvae"],
            "tie": len(contenders) > 1, "contenders": contenders, "tie_tolerance": float(tie_tolerance),
            "a4_source": contenders[0]}


# ============================================================================
# the decision.json schema
# ============================================================================

TOP_KEYS: Tuple[str, ...] = (
    "schema_version", "a4_source", "rule", "tie_break", "tie_tolerance", "decision", "decision_folds", "n_folds",
    "target_classes", "training_sets", "set_docs", "per_fold", "mean", "mean_all", "vs_copies", "vs_real",
    "count_matching", "syn_ratio", "max_added_rows", "classifier", "qc", "generators", "provenance", "hybridsn")
FOLD_KEYS: Tuple[str, ...] = ("fold", "heldout_patients", "n_heldout", "heldout_rows_per_class", "n_train_real",
                              "train_rows_per_class", "n_added", "copies", "synthetic", "sets")
SET_METRIC_KEYS: Tuple[str, ...] = ("n_train", "target_f1", "target_recall", "macro_f1", "balanced_accuracy",
                                    "targets", "per_class", "confusion", "converged")
MEAN_KEYS: Tuple[str, ...] = ("folds", "n_folds", "target_f1", "target_recall", "macro_f1", "per_target")
GENERATOR_KEYS: Tuple[str, ...] = ("param", "value", "choice_source", "choice_sha256", "generation_sha256",
                                   "final_checkpoint_sha256", "lopo_fingerprint", "generate_fingerprint", "folds")
PROVENANCE_KEYS: Tuple[str, ...] = ("folds_sha256", "pool_fingerprint", "qc_reference_fingerprint",
                                    "thresholds_sha256", "upstream", "seed")


def check_decision(doc: Any) -> List[str]:
    """Problems with a decision.json document (empty = valid, schema above)."""
    if not isinstance(doc, Mapping):
        return [f"not a JSON object: {type(doc).__name__}"]
    out = [f"missing key {k!r}" for k in TOP_KEYS if k not in doc]
    if out:
        return out
    if doc["schema_version"] != DECISION_SCHEMA_VERSION:
        out.append(f"schema_version {doc['schema_version']!r}, expected {DECISION_SCHEMA_VERSION}")
    if doc["a4_source"] not in SOURCES:
        out.append(f"a4_source {doc['a4_source']!r} is not one of {list(SOURCES)}")
    for k in READER_KEYS[1:]:
        if isinstance(doc.get(k), str):
            out.append(f"top-level {k!r} is a string: the readers would take it if a4_source were absent")
    dec = doc["decision"]
    if not isinstance(dec, Mapping) or dec.get("a4_source") != doc["a4_source"]:
        out.append("decision.a4_source differs from a4_source")
    elif set(dec.get("scores") or {}) != set(SOURCES) or not isinstance(dec.get("tie"), bool):
        out.append("decision needs scores for every generator and a boolean tie")
    if list(doc["training_sets"]) != list(TRAINING_SETS):
        out.append(f"training_sets {doc['training_sets']!r}, expected {list(TRAINING_SETS)}")
    folds = [int(f.get("fold", -1)) for f in doc["per_fold"]] if isinstance(doc["per_fold"], list) else []
    if not doc["decision_folds"] or not set(doc["decision_folds"]) <= set(folds):
        out.append(f"decision_folds {doc['decision_folds']!r} must be a non-empty subset of the folds {folds}")
    for f in doc["per_fold"] if isinstance(doc["per_fold"], list) else []:
        miss = [k for k in FOLD_KEYS if k not in f]
        if miss:
            out.append(f"fold {f.get('fold')}: missing {miss}")
            continue
        for s, rec in f["sets"].items():
            if s not in TRAINING_SETS:
                out.append(f"fold {f['fold']}: unknown set {s!r}")
            elif [k for k in SET_METRIC_KEYS if k not in rec]:
                out.append(f"fold {f['fold']} set {s}: missing {[k for k in SET_METRIC_KEYS if k not in rec]}")
        if int(f["fold"]) in set(doc["decision_folds"]) and set(f["sets"]) != set(TRAINING_SETS):
            out.append(f"decision fold {f['fold']} lacks sets {sorted(set(TRAINING_SETS) - set(f['sets']))}")
    for block in ("mean", "mean_all"):
        for s in TRAINING_SETS:
            rec = (doc[block] or {}).get(s)
            if not isinstance(rec, Mapping) or [k for k in MEAN_KEYS if k not in rec]:
                out.append(f"{block}.{s} is missing or incomplete")
    for block in ("vs_copies", "vs_real"):
        if set(doc[block] or {}) != set(SOURCES):
            out.append(f"{block} must have one entry per generator")
    for s in SOURCES:
        g = (doc["generators"] or {}).get(s)
        if not isinstance(g, Mapping) or [k for k in GENERATOR_KEYS if k not in g]:
            out.append(f"generators.{s} is missing or incomplete")
    prov = doc["provenance"]
    if not isinstance(prov, Mapping) or [k for k in PROVENANCE_KEYS if k not in prov]:
        out.append(f"provenance must hold {list(PROVENANCE_KEYS)}")
    return out


# ============================================================================
# features
# ============================================================================

def real_features(X: np.ndarray, rows: np.ndarray, *, chunk_rows: int) -> np.ndarray:
    """`provenance.patch_feats` of X[rows] (float32 [n, 2C], in `rows`
    order), read in sorted row order `chunk_rows` rows at a time."""
    from hsi_augment.provenance import patch_feats
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    out = np.empty((len(rows), 2 * int(X.shape[-1])), dtype=np.float32)
    pos = 0
    for ch in iter_chunks(X, rows[order], chunk_rows=int(chunk_rows), assume_sorted=True):
        out[order[pos:pos + len(ch.rows)]] = patch_feats(ch.data)
        pos += len(ch.rows)
    return out


def standardiser(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(mean, std) per feature, float64; a std <= MIN_STD becomes 1."""
    f = np.asarray(f)
    mu = f.mean(axis=0, dtype=np.float64)
    sd = f.std(axis=0, dtype=np.float64)
    return mu, np.where(sd > MIN_STD, sd, 1.0)


def standardise(f: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    out = np.empty(np.asarray(f).shape, dtype=np.float32)
    for a in range(0, len(f), STANDARDISE_BLOCK):
        out[a:a + STANDARDISE_BLOCK] = (np.asarray(f[a:a + STANDARDISE_BLOCK], dtype=np.float64) - mu) / sd
    return out


def synthetic_features(handle: "G.GeneratorHandle", ref: "QCReference", class_id: int, n_add: int, value: float,
                       seed: int, qc_fold: int, *, batch_size: int, overgenerate: float, min_acceptance: float,
                       qc_chunk_rows: int, qc_device: str, what: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """`patch_feats` of the first `n_add` samples of `handle`'s class stream
    that pass R1-R6 against QC fold `qc_fold`, and a record. Batches of
    `batch_size` (rows [a, a + batch) of the stream, as generate-<src>
    samples them) until `n_add` are accepted or `overgenerate` x n_add are
    generated. AcceptanceError below `min_acceptance` or short of `n_add`."""
    from hsi_augment.provenance import patch_feats
    from hsi_augment.qc import summarize_bitmask
    budget = G.generate_counts({int(class_id): int(n_add)}, overgenerate)[int(class_id)]
    masks: List[np.ndarray] = []
    feats: List[np.ndarray] = []
    n_acc = 0
    for a in range(0, budget, int(batch_size)):
        x = handle.sample(class_id, min(int(batch_size), budget - a), value, seed, a)
        mask, _ = G.score_in_chunks(ref, x, class_id, fold=qc_fold, device=qc_device, chunk_rows=qc_chunk_rows)
        masks.append(mask)
        ok = np.flatnonzero(mask == 0)[:max(int(n_add) - n_acc, 0)]
        if ok.size:
            feats.append(patch_feats(x[ok]))
            n_acc += int(ok.size)
        if n_acc >= int(n_add):
            break
    bitmask = np.concatenate(masks) if masks else np.zeros(0, dtype=np.uint8)
    _, rec = G.select_accepted(bitmask, int(n_add), min_acceptance=min_acceptance, overgenerate=overgenerate,
                               what=what)
    f = np.concatenate(feats)[:int(n_add)]
    if not np.isfinite(f).all():
        raise StageError(f"{what}: QC-accepted samples have non-finite mean/std features")
    rec.update(rejected_by=summarize_bitmask(bitmask)["rejected_by"], budget=int(budget),
               stopped_early=bool(bitmask.size < budget), seed=int(seed), qc_fold=int(qc_fold))
    return f, rec


# ============================================================================
# inputs
# ============================================================================

def open_pool(ctx: "StageContext", train: Any) -> "Pool":
    """The pool stage's pool, checked against this build's train split."""
    from hsi_augment.pool_io import PoolError, check_against_train, load_pool
    d = ctx.upstream_dir("pool")
    try:
        pool = load_pool(d)
    except (FileNotFoundError, PoolError) as e:
        raise StageError(f"cannot load the pool: {e}") from e
    problems = check_against_train(pool, train.y, train.groups, train.captures)
    if pool.n_bands != ctx.adapter.n_bands or pool.patch != ctx.adapter.patch_size:
        problems.append(f"pool patches are {pool.patch}x{pool.patch}x{pool.n_bands}, the build's "
                        f"{ctx.adapter.patch_size}x{ctx.adapter.patch_size}x{ctx.adapter.n_bands}")
    if problems:
        raise StageError(f"the pool does not belong to the train split of {ctx.data_dir}: {problems}")
    return pool


def load_reference(ctx: "StageContext", train_X: np.ndarray) -> "QCReference":
    """The qc-reference stage's references, with the train patches this
    stage opened (purpose "lopo"), so the reference opens nothing itself."""
    from hsi_augment.qc import QCReference
    try:
        return QCReference.load(ctx.upstream_dir(G.QC_REFERENCE_STAGE), train_X=train_X)
    except (OSError, ValueError, KeyError) as e:
        raise StageError(f"cannot load the QC reference: {e}") from e


def source_inputs(ctx: "StageContext", source: str, targets: Sequence[int], folds: Mapping[int, Any],
                  device: Any) -> Dict[str, Any]:
    """Everything about one generator: the chosen value (choice.json), the
    generate stage's sampling settings, and its LOPO fold generators (loaded
    lazily), checked against the folds."""
    gen_stage = G.stage_name("generate", source)
    gen_dir = ctx.upstream_dir(gen_stage)
    choice = read_json(gen_dir / G.CHOICE_NAME)
    gen_doc = read_json(gen_dir / G.GENERATION_NAME)
    if choice is None or gen_doc is None:
        raise StageError(f"{gen_stage}: {G.CHOICE_NAME} or {G.GENERATION_NAME} is missing; re-run {gen_stage}")
    value = choice.get("value")
    if value is None or gen_doc.get("value") is None or float(value) != float(gen_doc["value"]):
        raise StageError(f"{gen_stage}: choice.json value {value!r} and generation.json value "
                         f"{gen_doc.get('value')!r} disagree; re-run {gen_stage}")
    G.check_value(source, float(value))
    missing = [int(c) for c in targets if str(int(c)) not in (gen_doc.get("targets") or {})]
    if missing:
        raise StageError(f"{gen_stage} generated no samples of target class(es) {missing}")
    sampling = SimpleNamespace(sample_steps=(gen_doc.get("generator") or {}).get("sample_steps"),
                               batch_size=int(gen_doc.get("batch_size") or 1024))
    lopo_stage = G.GENERATOR_STAGES[source][0]
    fold_gens = {}
    for fg in G.lopo_generators(ctx, source, sampling, device):
        want = folds.get(int(fg.fold))
        if want is None:
            raise StageError(f"{lopo_stage} has a fold {fg.fold} that folds.json does not")
        held = tuple(sorted(int(p) for p in want["heldout_patients"]))
        if tuple(fg.heldout_patients) != held:
            raise StageError(f"{lopo_stage} fold {fg.fold} held out {list(fg.heldout_patients)}, folds.json "
                             f"{list(held)}: the generator was fitted on other folds; re-run {lopo_stage}")
        fold_gens[int(fg.fold)] = fg
    return {"source": source, "param": G.PARAM_NAMES[source], "value": float(value), "choice": choice,
            "generation": gen_doc, "sampling": sampling, "folds": fold_gens,
            "record": {"param": G.PARAM_NAMES[source], "value": float(value),
                       "choice_source": choice.get("source"), "choice_sha256": sha256_file(gen_dir / G.CHOICE_NAME),
                       "generation_sha256": sha256_file(gen_dir / G.GENERATION_NAME),
                       "final_checkpoint_sha256": (gen_doc.get("generator") or {}).get("checkpoint_sha256"),
                       "lopo_stage": lopo_stage,
                       "lopo_fingerprint": G.upstream_done(ctx, lopo_stage).get("fingerprint"),
                       "generate_fingerprint": G.upstream_done(ctx, gen_stage).get("fingerprint"),
                       "sample_steps": sampling.sample_steps, "batch_size": sampling.batch_size,
                       "lopo_folds": sorted(fold_gens), "folds": {}}}


# ============================================================================
# the stage
# ============================================================================

def _fit_and_score(f_train: np.ndarray, y_train: np.ndarray, f_held: np.ndarray, y_held: np.ndarray, *,
                   names: Sequence[str], targets: Sequence[int], max_iter: int, seed: int) -> Dict[str, Any]:
    from hsi_augment.qc.consistency import fit_probe
    probe = fit_probe(f_train, y_train, max_iter=max_iter, seed=seed)
    pred = probe.predict_proba(f_held, len(names)).argmax(axis=1)
    return {"n_train": int(len(y_train)),
            "train_rows_per_class": {names[c]: int(n) for c, n in
                                     enumerate(np.bincount(np.asarray(y_train, dtype=np.int64),
                                                           minlength=len(names)))},
            "converged": bool(probe.converged), **score_heldout(y_held, pred, names, targets)}


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3f}"


def run_lopo_utility(ctx: "StageContext", cfg: Any) -> Dict[str, Any]:
    from hsi_augment.splits import FOLDS_NAME, fold_masks, load_folds

    adapter = ctx.adapter
    names = [str(n) for n in adapter.class_names]
    n_classes = int(adapter.num_classes)
    ctx.path(DECISION_NAME).unlink(missing_ok=True)     # a failed re-run must leave no decision behind
    device = G.torch_device(cfg.device)
    qc_device = "cuda" if device.type == "cuda" else "cpu"

    train = ctx.open_split("train", purpose=PURPOSE, kinds=("X", "y", "groups", "captures"))
    targets = sorted(G.target_counts(train.y, n_classes, names, cfg.target_classes))
    target_names = [names[c] for c in targets]
    leak_dir = ctx.upstream_dir("leakage")
    folds_doc = load_folds(leak_dir)
    folds = {int(f["fold"]): f for f in folds_doc["folds"]}
    pool = open_pool(ctx, train)
    with ctx.timer("pool sha256"):
        pool_sha = pool.fingerprint()
    ref = load_reference(ctx, train.X)
    G.check_targets_have_reference(ref, targets, names)
    sources = {src: source_inputs(ctx, src, targets, folds, device) for src in SOURCES}
    for src, s in sources.items():
        ctx.log(f"{src}: {s['param']} = {s['value']:g} ({s['record']['choice_source']}), LOPO fold generators "
                f"{sorted(s['folds'])}")

    feat_dim = 2 * int(pool.n_bands)
    ctx.ram_guard(pool.n * feat_dim * 4 * 4, f"mean/std features of {pool.n} pool rows")
    with ctx.timer("real features"):
        feats = real_features(train.X, pool.row, chunk_rows=cfg.chunk_rows)
    bad = int((~np.isfinite(feats).all(axis=1)).sum())
    if bad:
        raise StageError(f"{bad} pool rows have non-finite mean/std features in X_train")
    ctx.log(f"real: mean/std features of {pool.n} pool rows ({feat_dim} each)")
    probe_seed = ctx.seed_int("probe")

    per_fold: List[Dict[str, Any]] = []
    for k in sorted(folds):
        fold = folds[k]
        tr, ho = fold_masks(pool.patient, fold)
        y_tr, y_ho = pool.y[tr].astype(np.int64), pool.y[ho].astype(np.int64)
        missing = [names[c] for c in targets if not (y_tr == c).any()]
        if missing or not ho.any():
            raise StageError(f"fold {k}: no train-side pool rows of {missing} or no held-out pool rows")
        mu, sd = standardiser(feats[tr])
        f_tr, f_ho = standardise(feats[tr], mu, sd), standardise(feats[ho], mu, sd)
        n_train_c = {c: int((y_tr == c).sum()) for c in targets}
        n_add = {c: added_rows(n_train_c[c], cfg.syn_ratio, cfg.max_added_rows) for c in targets}
        rec: Dict[str, Any] = {
            "fold": k, "dcis_patient": fold.get("dcis_patient"),
            "heldout_patients": [int(p) for p in sorted(fold["heldout_patients"])],
            "n_heldout": int(ho.sum()),
            "heldout_rows_per_class": {names[c]: int(n) for c, n in enumerate(np.bincount(y_ho, minlength=n_classes))},
            "n_train_real": int(tr.sum()),
            "train_rows_per_class": {names[c]: int(n) for c, n in enumerate(np.bincount(y_tr, minlength=n_classes))},
            "n_added": {names[c]: n_add[c] for c in targets}, "copies": {}, "synthetic": {}, "sets": {}}
        ctx.log(f"fold {k}: held out {rec['heldout_patients']} ({rec['n_heldout']} pool rows), "
                f"{rec['n_train_real']} train-side rows; adding " +
                ", ".join(f"{names[c]} {n_add[c]}" for c in targets))

        added: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "real": (np.zeros((0, feat_dim), dtype=np.float32), np.zeros(0, dtype=np.int64))}
        cp_parts = []
        for c in targets:
            pos_c = np.flatnonzero(y_tr == c)
            sel = copy_positions(len(pos_c), n_add[c], ctx.rng("copies", k, c))
            cp_parts.append(pos_c[sel])
            q, r = divmod(n_add[c], len(pos_c))
            rec["copies"][names[c]] = {"n_source_rows": int(len(pos_c)), "n_added": int(len(sel)),
                                       "repeats": int(q), "extra_draw": int(r)}
        cp = np.concatenate(cp_parts)
        added["copies"] = (f_tr[cp], y_tr[cp])

        for src in SOURCES:
            s = sources[src]
            fg = s["folds"].get(k)
            if fg is None:
                ctx.warn(f"fold {k}: {G.GENERATOR_STAGES[src][0]} has no generator for this fold; set {src} is "
                         f"not scored here")
                continue
            with ctx.timer(f"load {src}"):
                handle = fg.load()
            have = handle.record.get("pool_sha256")
            if have is None:
                ctx.warn(f"fold {k}: the {src} generator records no pool sha256; cannot check its pool")
            elif have != pool_sha:
                raise StageError(f"fold {k}: the {src} generator was trained on another pool (sha256 "
                                 f"{str(have)[:12]} != {pool_sha[:12]}); re-run {G.GENERATOR_STAGES[src][0]}")
            qk = G.qc_fold_index(ref, fg.heldout_patients)
            s["record"]["folds"][str(k)] = {
                "checkpoint": G.relative_to_out(ctx, handle.ckpt_path), "checkpoint_sha256": handle.ckpt_sha256,
                "pool_sha256": have, "codec_sha256": handle.record.get("codec_sha256"),
                "heldout_patients": list(fg.heldout_patients)}
            parts_f, parts_y, syn = [], [], {}
            for c in targets:
                seed = ctx.seed_int("sample", src, k, c)
                with ctx.timer(f"sample+qc {src}"):
                    f_c, srec = synthetic_features(
                        handle, ref, c, n_add[c], s["value"], seed, qk, batch_size=s["sampling"].batch_size,
                        overgenerate=cfg.overgenerate, min_acceptance=cfg.min_acceptance,
                        qc_chunk_rows=cfg.qc_chunk_rows, qc_device=qc_device, what=f"{src} fold {k} {names[c]}")
                parts_f.append(standardise(f_c, mu, sd))
                parts_y.append(np.full(len(f_c), c, dtype=np.int64))
                syn[names[c]] = srec
                ctx.log(f"fold {k} {src} {names[c]}: {srec['n_accepted']} of {srec['n_generated']} accepted "
                        f"({srec['acceptance']:.1%}), {srec['n_selected']} added")
            del handle
            rec["synthetic"][src] = syn
            added[src] = (np.concatenate(parts_f), np.concatenate(parts_y))

        for name in TRAINING_SETS:
            if name not in added:
                continue
            fa, ya = added[name]
            with ctx.timer("fit"):
                sc = _fit_and_score(np.concatenate([f_tr, fa]), np.concatenate([y_tr, ya]), f_ho, y_ho,
                                    names=names, targets=targets, max_iter=cfg.max_iter, seed=probe_seed)
            rec["sets"][name] = sc
            if not sc["converged"]:
                ctx.warn(f"fold {k} {name}: the logistic regression did not converge in {cfg.max_iter} iterations")
            ctx.log(f"fold {k} {name:<6}: n_train {sc['n_train']}, " +
                    ", ".join(f"{t} recall {_fmt(sc['targets'][t]['recall'])} F1 {_fmt(sc['targets'][t]['f1'])}"
                              for t in target_names) + f", macro-F1 {_fmt(sc['macro_f1'])}")
            ctx.log_metrics({"event": "lopo_set", "fold": k, "set": name, "n_train": sc["n_train"],
                             "target_f1": sc["target_f1"], "target_recall": sc["target_recall"],
                             "macro_f1": sc["macro_f1"]})
        per_fold.append(rec)

    summary = summarize(per_fold, target_names)
    dfolds = summary["decision_folds"]
    if not dfolds:
        raise StageError("no LOPO fold has both a CVAE and an LDM fold generator (cvae-lopo folds "
                         f"{sorted(sources['cvae']['folds'])}, ldm-lopo folds {sorted(sources['ldm']['folds'])}): "
                         "nothing to decide on")
    if len(dfolds) < len(folds):
        ctx.warn(f"the decision uses {len(dfolds)} of {len(folds)} LOPO folds ({dfolds}): the folds both "
                 f"generators were fitted on")
    try:
        decision = decide({s: summary["mean"][s][DECISION_METRIC] for s in SOURCES}, cfg.tie_tolerance)
    except ValueError as e:
        raise StageError(f"cannot decide A4's generator: {e}") from e
    vs_copies = {s: compare(per_fold, s, "copies", dfolds) for s in SOURCES}
    vs_real = {s: compare(per_fold, s, "real", dfolds) for s in SOURCES}
    a4 = decision["a4_source"]

    def done_fp(name: str) -> Optional[str]:
        return G.upstream_done(ctx, name).get("fingerprint")

    from hsi_augment.qc.reference import THRESHOLDS_NAME
    ref_dir = ctx.upstream_dir(G.QC_REFERENCE_STAGE)
    doc = {
        "schema_version": DECISION_SCHEMA_VERSION, "a4_source": a4,
        "rule": DECISION_RULE, "tie_break": TIE_BREAK, "tie_tolerance": float(cfg.tie_tolerance),
        "decision": decision, "decision_folds": dfolds, "n_folds": len(folds), "target_classes": target_names,
        "training_sets": list(TRAINING_SETS), "set_docs": dict(SET_DOCS), "per_fold": per_fold,
        "mean": summary["mean"], "mean_all": summary["mean_all"], "vs_copies": vs_copies, "vs_real": vs_real,
        "count_matching": COUNT_MATCHING_RULE, "syn_ratio": float(cfg.syn_ratio),
        "max_added_rows": cfg.max_added_rows,
        "classifier": {**CLASSIFIER_DOC, "max_iter": int(cfg.max_iter), "seed": int(probe_seed)},
        "qc": {"rule": QC_RULE, "overgenerate": float(cfg.overgenerate), "min_acceptance": float(cfg.min_acceptance)},
        "generators": {s: sources[s]["record"] for s in SOURCES},
        "provenance": {
            "folds_sha256": sha256_file(Path(leak_dir) / FOLDS_NAME), "folds_seed": folds_doc.get("seed"),
            "pool_fingerprint": pool_sha, "pool_stage_fingerprint": done_fp("pool"),
            "qc_reference_fingerprint": done_fp(G.QC_REFERENCE_STAGE),
            "thresholds_sha256": sha256_file(ref_dir / THRESHOLDS_NAME),
            "input_manifest_sha256": G.upstream_done(ctx, "leakage").get("input_manifest_sha256"),
            "upstream": {name: done_fp(name) for name in ctx.spec.requires}, "seed": int(ctx.seed)},
        "hybridsn": {"run": False, "note": HYBRIDSN_NOTE},
    }
    problems = check_decision(doc)
    if problems:
        raise StageError(f"decision.json would not follow its schema: {problems}")
    ctx.save_json(DECISION_NAME, doc)
    means = summary["mean"]
    ctx.log("mean over folds " + str(dfolds) + ": " +
            ", ".join(f"{s} F1 {_fmt(means[s]['target_f1'])}" for s in TRAINING_SETS) +
            f" -> A4's generator: {a4}" + (" (tie)" if decision["tie"] else ""))
    for s in SOURCES:
        ctx.log(f"{s} vs copies: mean delta F1 {_fmt(vs_copies[s]['mean_delta_target_f1'])} "
                f"(better in {vs_copies[s]['better']} of {len(dfolds)} folds)")
    return {"a4_source": a4, "scores": decision["scores"], "tie": decision["tie"], "decision_folds": dfolds,
            "n_folds": len(folds), "target_classes": target_names,
            "mean_target_f1": {s: means[s]["target_f1"] for s in TRAINING_SETS},
            "mean_macro_f1": {s: means[s]["macro_f1"] for s in TRAINING_SETS},
            "beats_copies": {s: vs_copies[s]["beats"] for s in SOURCES}}
