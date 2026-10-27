# -*- coding: utf-8 -*-
"""
hsi_augment/provenance.py
=========================
Phase 4a (issue #15, Plan §Phase 4 items 1, 2 and 5): the provenance index of
the train split and the imbalance report. `stages/index.py` registers the
stage (`augment_pipeline.py index`, group `preprocess`); the figures are in
`provenance_figures.py`, so this numeric library never imports matplotlib.

Outputs in `<out>/prep/index/`:

    index_train.npz        one entry per train row (schema below); #16 pool,
                           #28 QC and #30 LOPO utility read it with `load_index`
    index_train.json       its schema, column definitions, provenance, `complete`
    imbalance_report.json  class/patient tables per split, the same tables on
                           train tissue rows, the effective number of samples
    figures/class_balance.png, figures/patient_contributions.png

Why an index at all
-------------------
The build stores no coordinates and no per-row provenance beyond three int
sidecars, and X_train is 19 GB. Every later phase needs to ask "which train
rows are tissue DCIS rows of patient 45, where on their capture are they, and
what do they look like on average" without touching X again. The index answers
that in one ~690 MB file (below), read once by each consumer.

index_train.npz (`INDEX_SCHEMA_VERSION` 1, uncompressed `np.savez`)
-------------------------------------------------------------------
Row k of every column describes train row `row[k]` of X_train. The rows are
0 .. n - 1 in order; n = N_train unless the stage ran with `max_rows` (a smoke
run: index_train.json then says `complete: false`).

    row           int32   [n]       row number in X_train / y_train (N < 2**31)
    y             int8    [n]       class index (= y_train)
    patient       int32   [n]       patient id (= groups_train)
    capture       int32   [n]       capture index (= captures_train; capture_index.json key)
    ordinal       int32   [n]       position within the capture's row run
    r, c          int16   [n]       grid row and column, (r, c) = divmod(ordinal, grid_cols);
                                    -1 for a capture that is not a whole number of grid rows
    tissue_score  float16 [n]       t in degrees, copied from prep/tissue/tissue_score_train.npy
    is_tissue     bool    [n]       copied from prep/tissue/tissue_mask_train.npy
    patch_feats   float32 [n, 2C]   [:, :C] per-band mean, [:, C:] per-band std (ddof 0)
                                    over the patch's H x W pixels, raw build values

Sizes on trainsel (N = 2,452,086, C = 32): the nine scalar columns are 24 bytes
a row, 59 MB, which is the plan's "about 60 MB"; `patch_feats` adds 8C bytes a
row, 628 MB, so the file is ~690 MB. It is written uncompressed: `np.load`
returns a lazy NpzFile, so a consumer that needs only `y` and `patient` reads
only those members, and `patch_feats` costs RAM only when it is asked for.

patch_feats and the shallow probe
---------------------------------
`patch_feats(x)` IS `scripts.shallow_baseline.featurize(x, "mean_std")`
(frozen; imported, not copied), applied to the raw patches widened to float64
in blocks of `FEATURE_BLOCK_ROWS`, then stored as float32. QC (#28) scores
synthetic patches with the same function, so real and synthetic features
cannot drift apart. Two differences from what the probe script feeds it:

  * value space. `shallow_baseline.evaluate_dataset` z-scores the patches with
    train per-band mean mu_b and std sigma_b first. Here they are raw. The two
    feature sets are a per-band affine map of each other (mean_z = (mean - mu_b)
    / sigma_b, std_z = std / sigma_b), so any standardised linear model
    (the 0.903 balanced-accuracy logistic probe) gives the same predictions on
    either; consumers that need z-scored features standardise these.
  * precision and NaN. The probe computes in float32 after `np.nan_to_num`;
    here float64 accumulation, and a non-finite pixel gives a non-finite
    feature row (counted in index_train.json; P1 found none in the build).

No val provenance index (coordinator decision for #15)
------------------------------------------------------
The plan's "val version without coordinates, for analysis only" would need
val X for `patch_feats` and the tissue score, and no SplitGuard purpose allows
reading val X here: `describe` is train-only, and `structural` is for counts,
ids and hashes. So none is written. The report stage (#34) scores val/test
tissue at report time with `tissue.load_reference(prep/tissue)` under
`final_eval`. This stage reads val and test through `structural` only, and
only their y / groups / captures sidecars, for the per-split count tables.

The imbalance report
--------------------
Per split (train, val, test; all rows) and for train tissue rows:

  * rows and shares per class; the max/min ratio and the ratio of every class
    to the declared minority class (IDC:DCIS = 13.08 on trainsel train);
  * the effective number of samples of Cui et al. (CVPR 2019),
    E_n = (1 - beta^n) / (1 - beta), for beta in `betas` (0.999, 0.9999), with
    each class's saturation 1 - beta^n and the class-balanced weights 1 / E_n
    normalised to sum to the number of classes present. With 10^5 - 10^6 rows a
    class, every class saturates at 1 / (1 - beta) and the weights are ~uniform:
    counted in patches, the imbalance looks mild;
  * patients and captures per class (30 / 5 / 34 patients on train), and per
    class the effective number of patients (n_eff = (sum_i n_i)^2 / sum_i n_i^2
    over that class's per-patient row counts, Kish's effective sample size).
    Patches of one patient are not independent observations; five DCIS patients
    are the real bottleneck, and no patch-count reweighting changes that;
  * per-patient contributions: rows and captures per class and the patient's
    share of each class's rows.

Limitation (Plan §Phase 4 item 5): at ~0.74 um/px an 11 x 11 patch spans
~8 um, about one cell, so DCIS-vs-IDC architecture (ducts, invasion through
the basement membrane) may be invisible to any patch-level model. Context
tiles would need a new raw build (FUTURE_EXPERIMENTS F13, added by #40). The
report carries this note under `limitations`.

Memory (Plan rule 5): X_train is streamed in `chunk_rows` memmap chunks and
widened to float64 `FEATURE_BLOCK_ROWS` rows at a time (16 MB); what is held
is the index itself (~690 MB on trainsel) plus the int sidecars.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hsi_augment import SCHEMA_VERSION
from hsi_augment.splits import patient_class_rows

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.stage import StageContext
    from hsi_augment.stages.index import IndexConfig

PathLike = Union[str, os.PathLike]

INDEX_SCHEMA_VERSION = 1
INDEX_NAME = "index_train.npz"
INDEX_META_NAME = "index_train.json"
REPORT_NAME = "imbalance_report.json"
FIGURES_DIR = "figures"

# float16 X rows widened to float64 at a time: 512 x 11 x 11 x 32 x 8 B = 16 MB.
FEATURE_BLOCK_ROWS = 512
DEFAULT_BETAS: Tuple[float, ...] = (0.999, 0.9999)
# HistologyHSI-BC at x10 (Plan §Phase 4 item 5); used only for the limitation note.
PIXEL_PITCH_UM = 0.74

# Column order and dtypes of index_train.npz; `patch_feats` is [n, 2C].
INDEX_DTYPES: Dict[str, np.dtype] = {
    "row": np.dtype(np.int32),
    "y": np.dtype(np.int8),
    "patient": np.dtype(np.int32),
    "capture": np.dtype(np.int32),
    "ordinal": np.dtype(np.int32),
    "r": np.dtype(np.int16),
    "c": np.dtype(np.int16),
    "tissue_score": np.dtype(np.float16),
    "is_tissue": np.dtype(np.bool_),
    "patch_feats": np.dtype(np.float32),
}
INDEX_COLUMNS: Tuple[str, ...] = tuple(INDEX_DTYPES)
COLUMN_DEFINITIONS: Dict[str, str] = {
    "row": "row number in X_train / y_train / groups_train / captures_train",
    "y": "class index (y_train)",
    "patient": "patient id (groups_train)",
    "capture": "capture index (captures_train; the capture_index.json key)",
    "ordinal": "position of the row within its capture's contiguous run of rows",
    "r": "grid row = ordinal // grid_cols; -1 if the capture is not a whole number of grid rows",
    "c": "grid column = ordinal % grid_cols; -1 if the capture is not a whole number of grid rows",
    "tissue_score": "t = spectral angle (degrees) to s_glass; copy of prep/tissue/tissue_score_train.npy "
                    "(NaN where the tissue stage did not score the row)",
    "is_tissue": "copy of prep/tissue/tissue_mask_train.npy (t >= tau)",
    "patch_feats": "[:, :C] per-band mean and [:, C:] per-band std (ddof 0) over the H x W pixels, raw build "
                   "values: scripts.shallow_baseline.featurize(x, 'mean_std') on float64 patches, stored float32",
}
PATCH_FEATS_DIFFERENCES = (
    "value space: raw build values; shallow_baseline.evaluate_dataset z-scores the patches with train "
    "per-band stats first, which maps these features per band by an affine map (mean_z = (mean - mu_b) / "
    "sigma_b, std_z = std / sigma_b) and leaves standardised linear models unchanged",
    "precision: float64 accumulation stored as float32; the probe computes in float32",
    "NaN: kept (a non-finite pixel gives a non-finite feature row); the probe applies np.nan_to_num first",
)
VAL_INDEX_NOTE = ("No val/test provenance index: it would need val X, and no SplitGuard purpose allows that "
                  "outside 'structural' / 'final_eval'. The report stage (#34) scores val/test tissue at report "
                  "time with tissue.load_reference(prep/tissue) under 'final_eval'. Val/test here: the y, groups "
                  "and captures sidecars only, purpose 'structural'.")


# ============================================================================
# per-patch features
# ============================================================================

def patch_feats(x: np.ndarray, *, block_rows: int = FEATURE_BLOCK_ROWS) -> np.ndarray:
    """(n, H, W, C) patches -> (n, 2C) float32: per-band mean then per-band std
    (ddof 0) over the H x W pixels, in the patches' own value space. Exactly
    `scripts.shallow_baseline.featurize(x, "mean_std")` on float64 blocks of
    `block_rows` rows; QC (#28) uses it on synthetic patches."""
    from scripts.shallow_baseline import featurize      # frozen: import only
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"expected (n, H, W, C) patches, got shape {x.shape}")
    out = np.empty((x.shape[0], 2 * x.shape[-1]), dtype=np.float32)
    for a in range(0, x.shape[0], block_rows):
        b = min(a + block_rows, x.shape[0])
        out[a:b] = featurize(x[a:b].astype(np.float64), "mean_std")
    return out


# ============================================================================
# imbalance statistics
# ============================================================================

def effective_number(n: Any, beta: float) -> np.ndarray:
    """Cui et al. (2019): E_n = (1 - beta^n) / (1 - beta) = sum_{k<n} beta^k,
    float64, elementwise over `n`. Computed as -expm1(n log1p(beta - 1)) / (1 - beta),
    i.e. beta^n = exp(n ln beta) with ln beta = log1p(beta - 1) (beta - 1 is exact
    in floating point for beta near 1), so 1 - beta^n keeps full precision when
    beta^n is close to 1. beta = 0 gives 1 for every n >= 1 (one effective
    sample) and 0 for n = 0."""
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    n = np.asarray(n, dtype=np.float64)
    if np.any(n < 0):
        raise ValueError("counts must be >= 0")
    if beta == 0.0:
        return (n > 0).astype(np.float64)
    return -np.expm1(n * np.log1p(beta - 1.0)) / (1.0 - beta)


def class_balanced_weights(counts: Sequence[int], beta: float) -> List[Optional[float]]:
    """Cui et al.'s class-balanced weights: 1 / E_n per class, normalised to sum
    to the number of classes PRESENT; None for an absent class."""
    e = effective_number(counts, beta)
    present = e > 0
    if not present.any():
        return [None] * len(e)
    inv = np.where(present, 1.0 / np.where(present, e, 1.0), 0.0)
    w = inv * present.sum() / inv.sum()
    return [float(v) if p else None for v, p in zip(w, present)]


def kish_effective_n(counts: Sequence[int]) -> Optional[float]:
    """(sum n_i)^2 / sum n_i^2: how many equally sized groups the counts are
    worth. k equal patients give k; one dominant patient gives ~1."""
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    if c.size == 0:
        return None
    return float(c.sum() ** 2 / (c ** 2).sum())


def _per_class(values: Sequence[Any], class_names: Sequence[str]) -> Dict[str, Any]:
    return {name: values[i] for i, name in enumerate(class_names)}


def _ratio(a: float, b: float) -> Optional[float]:
    return float(a) / float(b) if b else None


def _beta_key(beta: float) -> str:
    return f"{float(beta):g}"


def split_table(y: np.ndarray, groups: np.ndarray, captures: np.ndarray, class_names: Sequence[str], *,
                minority_classes: Sequence[int] = (), betas: Sequence[float] = DEFAULT_BETAS) -> Dict[str, Any]:
    """The imbalance tables of one set of rows (a split, or its tissue rows).
    `y`, `groups` and `captures` are per-row and aligned. Pure; no I/O."""
    C = len(class_names)
    yy = np.asarray(y, dtype=np.int64).reshape(-1)
    g = np.asarray(groups, dtype=np.int64).reshape(-1)
    cap = np.asarray(captures, dtype=np.int64).reshape(-1)
    if not (yy.shape == g.shape == cap.shape):
        raise ValueError(f"y, groups and captures have different lengths {yy.shape, g.shape, cap.shape}")
    if yy.size and (yy.min() < 0 or yy.max() >= C):
        raise ValueError(f"labels outside 0..{C - 1}")
    rows = np.bincount(yy, minlength=C)
    n = int(rows.sum())

    pcr = patient_class_rows(g, yy, C)                      # {patient: rows per class}
    patients = sorted(pcr)
    pat_rows = np.array([pcr[p] for p in patients], dtype=np.int64).reshape(len(patients), C)
    # captures per (patient, class): unique (patient, class, capture) triples
    pat_idx = np.searchsorted(np.asarray(patients, dtype=np.int64), g)
    cap_ids, cap_idx = np.unique(cap, return_inverse=True)
    trip = np.unique((pat_idx * C + yy) * max(len(cap_ids), 1) + cap_idx)
    pc = trip // max(len(cap_ids), 1)                       # patient * C + class, one per capture
    pat_caps = np.bincount(pc, minlength=len(patients) * C).reshape(len(patients), C)
    caps_per_class = np.bincount(np.unique(cap_idx * C + yy) % C, minlength=C) if n else np.zeros(C, np.int64)
    patients_per_class = (pat_rows > 0).sum(axis=0)

    present = rows > 0
    shares = [float(v) / n if n else None for v in rows]
    imbalance: Dict[str, Any] = {"max_over_min": None, "largest_class": None, "smallest_class": None}
    if present.any():
        lo = int(np.flatnonzero(present)[np.argmin(rows[present])])
        hi = int(np.argmax(rows))
        imbalance = {"max_over_min": _ratio(rows[hi], rows[lo]), "largest_class": class_names[hi],
                     "smallest_class": class_names[lo], "label": f"{class_names[hi]}:{class_names[lo]}"}
    imbalance["ratio_to_minority"] = {
        class_names[m]: _per_class([_ratio(rows[c], rows[m]) for c in range(C)], class_names)
        for m in minority_classes}
    pp_present = patients_per_class > 0
    imbalance["patients_max_over_min"] = (
        _ratio(patients_per_class.max(), patients_per_class[pp_present].min()) if pp_present.any() else None)

    eff: Dict[str, Any] = {}
    for beta in betas:
        e = effective_number(rows, beta)
        eff[_beta_key(beta)] = {
            "per_class": _per_class([float(v) for v in e], class_names),
            "saturation": _per_class([float(v * (1.0 - beta)) for v in e], class_names),
            "upper_bound": (1.0 / (1.0 - beta)),
            "class_balanced_weights": _per_class(class_balanced_weights(rows, beta), class_names),
        }

    concentration: Dict[str, Any] = {}
    for k, name in enumerate(class_names):
        col = pat_rows[:, k]
        have = np.flatnonzero(col > 0)
        if have.size == 0:
            concentration[name] = {"n_patients": 0, "effective_patients": None, "rows_per_patient": None,
                                   "largest_patient": None, "largest_share": None, "patients": []}
            continue
        order = have[np.argsort(-col[have], kind="stable")]
        concentration[name] = {
            "n_patients": int(have.size),
            "effective_patients": kish_effective_n(col[have]),
            "rows_per_patient": {"min": int(col[have].min()), "median": float(np.median(col[have])),
                                 "max": int(col[have].max())},
            "largest_patient": int(patients[order[0]]),
            "largest_share": float(col[order[0]]) / float(rows[k]),
            "patients": [int(patients[i]) for i in order],        # by rows, descending
        }

    per_patient = []
    for i, p in enumerate(patients):
        per_patient.append({
            "patient": int(p),
            "rows": int(pat_rows[i].sum()),
            "share_of_rows": float(pat_rows[i].sum()) / n if n else None,
            "rows_per_class": _per_class([int(v) for v in pat_rows[i]], class_names),
            "captures_per_class": _per_class([int(v) for v in pat_caps[i]], class_names),
            "share_of_class_rows": _per_class([_ratio(pat_rows[i, k], rows[k]) for k in range(C)], class_names),
        })

    return {
        "rows": n,
        "rows_per_class": _per_class([int(v) for v in rows], class_names),
        "share_per_class": _per_class(shares, class_names),
        "imbalance": imbalance,
        "effective_number": eff,
        "n_patients": len(patients),
        "patients_per_class": _per_class([int(v) for v in patients_per_class], class_names),
        "n_captures": int(len(cap_ids)),
        "captures_per_class": _per_class([int(v) for v in caps_per_class], class_names),
        "patient_concentration": concentration,
        "per_patient": per_patient,
    }


def tissue_fractions(all_table: Mapping[str, Any], tissue_table: Mapping[str, Any],
                     class_names: Sequence[str]) -> Dict[str, Any]:
    """Tissue rows / all rows, overall, per class and per (patient, class)."""
    def frac(t: Any, a: Any) -> Optional[float]:
        return float(t) / float(a) if a else None

    tissue_pp = {e["patient"]: e["rows_per_class"] for e in tissue_table["per_patient"]}
    per_patient = {}
    for e in all_table["per_patient"]:
        t = tissue_pp.get(e["patient"], {})
        per_patient[str(e["patient"])] = {name: frac(t.get(name, 0), e["rows_per_class"][name])
                                          for name in class_names}
    return {"overall": frac(tissue_table["rows"], all_table["rows"]),
            "per_class": {name: frac(tissue_table["rows_per_class"][name], all_table["rows_per_class"][name])
                          for name in class_names},
            "per_patient": per_patient}


def limitation_note(patch_size: int, pixel_pitch_um: float = PIXEL_PITCH_UM) -> str:
    span = patch_size * pixel_pitch_um
    return (f"Each patch ({patch_size} x {patch_size} px at ~{pixel_pitch_um} um/px) spans ~{span:.0f} um, about "
            f"one cell, so DCIS-vs-IDC architecture (duct structure, invasion through the basement membrane) may "
            f"be invisible to any patch-level model, real or synthetic. Context tiles would need a new raw "
            f"build: FUTURE_EXPERIMENTS F13 (added by #40).")


# ============================================================================
# reading the index back (#16, #28, #30)
# ============================================================================

def load_index(path: PathLike, columns: Optional[Sequence[str]] = None, *,
               require_complete: bool = False) -> Dict[str, np.ndarray]:
    """Columns of index_train.npz, from its path or the stage directory. Only
    the requested members are read (`patch_feats` is ~630 MB on trainsel).
    `require_complete` refuses a `max_rows` smoke index."""
    p = Path(path)
    if p.is_dir():
        p = p / INDEX_NAME
    meta_path = p.with_name(INDEX_META_NAME)
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError(f"{meta_path}: index_schema_version {meta.get('index_schema_version')}, "
                             f"expected {INDEX_SCHEMA_VERSION}")
        if require_complete and not meta.get("complete"):
            raise ValueError(f"{p} is a partial index ({meta.get('n_rows')} of {meta.get('n_train_rows')} "
                             f"train rows, max_rows smoke run)")
    elif require_complete:
        raise ValueError(f"{meta_path} is missing: completeness cannot be checked")
    with np.load(p, allow_pickle=False) as z:
        names = list(columns) if columns is not None else list(z.files)
        missing = [c for c in names if c not in z.files]
        if missing:
            raise KeyError(f"{p} has no column(s) {missing}; columns: {z.files}")
        return {c: z[c] for c in names}


def save_npz_atomic(path: PathLike, arrays: Mapping[str, np.ndarray]) -> Path:
    """Uncompressed `np.savez` to `<path>.tmp`, fsync, `os.replace`: an
    interrupted write leaves nothing under `path`. np.savez streams each member
    in 16 MiB pieces, so no member is copied whole."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return p


# ============================================================================
# the stage
# ============================================================================

def _fmt_counts(counts: Mapping[str, int]) -> str:
    return " / ".join(f"{k} {v:,}" for k, v in counts.items())


def run_index(ctx: "StageContext", cfg: "IndexConfig") -> Dict[str, Any]:
    """The `index` stage: provenance index, imbalance report, figures. See the
    module docstring."""
    from hsi_augment import tissue
    from hsi_augment.adapters.base import NonContiguousCaptureError
    from hsi_augment.io import SPLITS, iter_chunks
    from hsi_augment.splits import open_split_labels
    from hsi_augment.stage import StageError

    adapter = ctx.adapter
    class_names = list(adapter.class_names)
    C = len(class_names)
    minority = list(adapter.minority_classes)
    betas = tuple(float(b) for b in cfg.betas)

    # 1. The tissue-mask outputs (train-fitted score and mask, full length).
    tdir = ctx.upstream_dir("tissue-mask")
    try:
        with open(tdir / tissue.TAU_NAME) as f:
            tau_doc = json.load(f)
        score = np.load(tdir / tissue.SCORE_NAME, mmap_mode="r")
        mask = np.load(tdir / tissue.MASK_NAME, mmap_mode="r")
    except (OSError, ValueError) as e:
        raise StageError(f"tissue-mask outputs unreadable in {tdir}: {e}") from e
    tissue_complete = bool(tau_doc.get("complete", True))

    # 2. Train, with X, under `describe` (train-only).
    train = ctx.open_split("train", purpose="describe")
    if train.groups is None or train.captures is None:
        raise StageError("the train split needs groups_train.npy and captures_train.npy")
    N = train.n
    if N >= np.iinfo(np.int32).max:
        raise StageError(f"{N} train rows do not fit the int32 `row` column")
    if score.shape != (N,) or mask.shape != (N,) or mask.dtype != np.bool_:
        raise StageError(f"tissue-mask outputs do not describe this train split: score {score.shape}, mask "
                         f"{mask.shape} {mask.dtype}, train has {N} rows; re-run tissue-mask")
    if not np.array_equal(np.asarray(mask), tissue.apply_threshold(np.asarray(score), tau_doc["tau_deg"])):
        raise StageError(f"{tissue.MASK_NAME} is not {tissue.MASK_RULE} applied to {tissue.SCORE_NAME} with "
                         f"tau {tau_doc['tau_deg']}: tissue outputs from different runs; re-run tissue-mask")
    if not tissue_complete:
        ctx.warn("the tissue-mask stage ran on a subset of captures (complete=false): unscored rows have "
                 "tissue_score NaN and is_tissue False, and the tissue tables cover only scored rows")

    y = np.asarray(train.y, dtype=np.int64)
    groups = np.asarray(train.groups)
    captures = np.asarray(train.captures)
    n_index = N if cfg.max_rows is None else min(int(cfg.max_rows), N)
    complete = n_index == N
    if not complete:
        ctx.warn(f"max_rows={cfg.max_rows}: indexing only train rows 0..{n_index - 1} of {N}; "
                 f"index_train.json says complete=false (the imbalance report still covers every row)")
    try:
        coords = train.coordinates()
    except NonContiguousCaptureError as e:
        raise StageError(f"grid coordinates undefined: {e}") from e
    n_bands = int(train.X.shape[-1])
    row_values = int(np.prod(train.X.shape[1:]))
    if int(coords.ordinal.max(initial=0)) > np.iinfo(np.int32).max or \
            max(int(coords.r.max(initial=0)), int(coords.c.max(initial=0))) > np.iinfo(np.int16).max:
        raise StageError("grid positions do not fit the index's int32 ordinal / int16 r, c columns")

    feat_bytes = n_index * 2 * n_bands * 4
    scalar_bytes = n_index * sum(INDEX_DTYPES[c].itemsize for c in INDEX_COLUMNS if c != "patch_feats")
    projected = (feat_bytes + scalar_bytes
                 + N * (8 + 8 + 8 + 1)                 # coordinates (ordinal, r, c int64, has_grid)
                 + N * (8 + 8 + 8)                     # y, groups, captures as int64 in the tables
                 + 4 * FEATURE_BLOCK_ROWS * row_values * 8)
    ctx.ram_guard(projected, "index columns + patch_feats + coordinates + float64 blocks")
    ctx.log(f"{N:,} train rows ({n_index:,} indexed), {len(coords.runs)} captures, {n_bands} bands, patch "
            f"{int(train.X.shape[1])}; tissue tau {tau_doc['tau_deg']:.3f} deg (complete={tissue_complete})")

    # 3. patch_feats: one sequential pass over X_train[:n_index].
    feats = np.empty((n_index, 2 * n_bands), dtype=np.float32)
    X = train.X if complete else train.X[:n_index]
    with ctx.timer("patch_feats"):
        done_rows, next_log = 0, max(1, n_index // 10)
        for ch in iter_chunks(X, chunk_rows=cfg.chunk_rows):
            feats[ch.rows] = patch_feats(ch.data)
            done_rows += len(ch.rows)
            if done_rows >= next_log or done_rows == n_index:
                ctx.log(f"patch_feats: {done_rows:,}/{n_index:,} rows")
                next_log += max(1, n_index // 10)
            ctx.log_metrics({"rows_done": done_rows})
    nonfinite = int((~np.isfinite(feats).all(axis=1)).sum())
    if nonfinite:
        ctx.warn(f"{nonfinite} indexed rows have a non-finite patch_feats row (a non-finite pixel)")

    # 4. index_train.npz and its schema.
    cols: Dict[str, np.ndarray] = {
        "row": np.arange(n_index, dtype=INDEX_DTYPES["row"]),
        "y": y[:n_index].astype(INDEX_DTYPES["y"]),
        "patient": groups[:n_index].astype(INDEX_DTYPES["patient"]),
        "capture": captures[:n_index].astype(INDEX_DTYPES["capture"]),
        "ordinal": coords.ordinal[:n_index].astype(INDEX_DTYPES["ordinal"]),
        "r": coords.r[:n_index].astype(INDEX_DTYPES["r"]),
        "c": coords.c[:n_index].astype(INDEX_DTYPES["c"]),
        "tissue_score": np.asarray(score[:n_index]).astype(INDEX_DTYPES["tissue_score"]),
        "is_tissue": np.asarray(mask[:n_index]).astype(INDEX_DTYPES["is_tissue"]),
        "patch_feats": feats,
    }
    rows_without_grid = int((~coords.has_grid[:n_index]).sum())
    grid_shapes = sorted({tuple(gs) for gs in coords.grid_shape.values() if gs is not None})
    del coords
    with ctx.timer("save_index"):
        ctx.disk_preflight(ctx.path(INDEX_NAME), sum(a.nbytes for a in cols.values()) + (1 << 20))
        ctx.add_output(save_npz_atomic(ctx.path(INDEX_NAME), cols))
    wl = np.asarray(adapter.wavelengths, dtype=np.float64)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "file": INDEX_NAME,
        "complete": complete,
        "n_rows": n_index,
        "n_train_rows": N,
        "max_rows": cfg.max_rows,
        "rows": "row k of every column describes train row row[k] = k (0 .. n_rows - 1)",
        "columns": {c: {"dtype": str(a.dtype), "shape": list(a.shape), "definition": COLUMN_DEFINITIONS[c]}
                    for c, a in cols.items()},
        "patch_feats": {"layout": f"[:, :{n_bands}] per-band mean, [:, {n_bands}:] per-band std (ddof 0)",
                        "function": "hsi_augment.provenance.patch_feats = scripts.shallow_baseline.featurize("
                                    "x, 'mean_std') on float64 blocks",
                        "value_space": "raw build values (no normalisation)",
                        "differences_from_shallow_probe": list(PATCH_FEATS_DIFFERENCES),
                        "n_nonfinite_rows": nonfinite, "wavelengths_nm": wl.tolist(),
                        "spectral_gaps_after_band": adapter.spectral_gaps()},
        "grid": {"grid_cols": adapter.grid_cols, "grid_shapes": [list(s) for s in grid_shapes],
                 "rows_without_grid": rows_without_grid},
        "tissue": {"stage_dir": "prep/tissue", "tau_deg": float(tau_doc["tau_deg"]),
                   "method_used": tau_doc.get("method_used"), "mask_rule": tissue.MASK_RULE,
                   "complete": tissue_complete, "n_unscored_rows": int((~np.isfinite(cols["tissue_score"])).sum())},
        "val_test": VAL_INDEX_NOTE,
        "sizes_bytes": {c: int(a.nbytes) for c, a in cols.items()},
    }
    ctx.save_json(INDEX_META_NAME, meta)
    ctx.log(f"index: {n_index:,} rows, {sum(a.nbytes for a in cols.values()) / 1e6:.1f} MB "
            f"(patch_feats {feats.nbytes / 1e6:.1f} MB) -> {INDEX_NAME}")
    del cols, feats

    # 5. The imbalance report. Val/test: int sidecars only, purpose `structural`.
    with ctx.timer("report"):
        held = {s: open_split_labels(ctx, s, "structural") for s in SPLITS if s != "train"}
        tables = {"train": split_table(y, groups, captures, class_names, minority_classes=minority, betas=betas)}
        for s, lab in held.items():
            tables[s] = split_table(lab.y, lab.groups, lab.captures, class_names, minority_classes=minority,
                                    betas=betas)
        m = np.asarray(mask, dtype=bool)
        tissue_train = split_table(y[m], groups[m], captures[m], class_names, minority_classes=minority,
                                   betas=betas)
    names_min = [class_names[k] for k in minority]
    report = {
        "schema_version": SCHEMA_VERSION,
        "class_names": class_names,
        "minority_classes": names_min,
        "betas": list(betas),
        "definitions": {
            "rows": "patches (rows of X_*); patches of one capture or patient are not independent observations",
            "imbalance.max_over_min": "rows of the largest class / rows of the smallest present class",
            "imbalance.ratio_to_minority": "rows of each class / rows of the declared minority class",
            "effective_number": "E_n = (1 - beta^n) / (1 - beta) per class (Cui et al., CVPR 2019); saturation "
                                "= E_n (1 - beta) = 1 - beta^n; class_balanced_weights = 1 / E_n normalised to "
                                "sum to the number of classes present",
            "patients_per_class": "patients with at least one row of the class; a patient with several tissue "
                                  "types counts once per class, so the classes sum to more than n_patients",
            "patient_concentration.effective_patients": "(sum_i n_i)^2 / sum_i n_i^2 over the class's "
                                                        "per-patient row counts (Kish)",
        },
        "splits": tables,
        "tissue": {
            "rule": tissue.MASK_RULE, "tau_deg": float(tau_doc["tau_deg"]), "stage_dir": "prep/tissue",
            "complete": tissue_complete, "n_unscored_rows": int((~np.isfinite(np.asarray(score))).sum()),
            "train": tissue_train,
            "tissue_fraction": tissue_fractions(tables["train"], tissue_train, class_names),
            "val_test": VAL_INDEX_NOTE,
        },
        "limitations": [limitation_note(adapter.patch_size)],
        "split_access": {"train": "X, y, groups, captures (purpose 'describe')",
                         "val": "y, groups, captures only (purpose 'structural')",
                         "test": "y, groups, captures only (purpose 'structural')"},
    }
    ctx.save_json(REPORT_NAME, report)

    tr = tables["train"]
    imb = tr["imbalance"]
    ctx.log(f"train rows per class {_fmt_counts(tr['rows_per_class'])}; {imb.get('label')} = "
            f"{imb['max_over_min']:.2f}" if imb["max_over_min"] else "train has no rows")
    ctx.log("patients per class (train/val/test): " + ", ".join(
        f"{n} " + "/".join(str(tables[s]["patients_per_class"][n]) for s in SPLITS) for n in class_names))
    for n in names_min:
        conc = tr["patient_concentration"][n]
        ctx.log(f"{n}: {conc['n_patients']} train patients, effective {conc['effective_patients'] or 0:.2f}; "
                f"largest patient {conc['largest_patient']} holds {100 * (conc['largest_share'] or 0):.1f}% "
                f"of its train rows")
    ctx.log(f"train tissue rows per class {_fmt_counts(tissue_train['rows_per_class'])}")
    ctx.log(f"limitation: {report['limitations'][0]}")

    # 6. Figures.
    from hsi_augment import provenance_figures as figs
    with ctx.timer("figures"):
        ctx.add_output(figs.plot_class_balance(ctx.path(FIGURES_DIR, "class_balance.png"), report))
        ctx.add_output(figs.plot_patient_contributions(ctx.path(FIGURES_DIR, "patient_contributions.png"),
                                                       report))

    return {
        "complete": complete,
        "n_train_rows": N,
        "n_index_rows": n_index,
        "tissue_complete": tissue_complete,
        "rows_per_class": {s: tables[s]["rows_per_class"] for s in tables},
        "patients_per_class": {s: tables[s]["patients_per_class"] for s in tables},
        "train_imbalance_max_over_min": imb["max_over_min"],
        "train_tissue_rows_per_class": tissue_train["rows_per_class"],
        "patch_feats_nonfinite_rows": nonfinite,
    }
