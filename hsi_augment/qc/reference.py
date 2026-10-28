# -*- coding: utf-8 -*-
"""
hsi_augment/qc/reference.py
===========================
The `qc-reference` stage (issue #28, Plan §Phase 8b) and `QCReference`, the
object #29 (generation/export) and #30 (LOPO utility) load to QC generated
samples:

    from hsi_augment.qc import QCReference, apply_rules, set_metrics
    ref = QCReference.load(out / "qc/reference",
                           train_X=ctx.open_split("train", purpose="qc").X)
    bitmask, scores = apply_rules(ref, x_f16_nhwc, class_id)          # fold=k for fold k's generators
    report = set_metrics(ref, x_f16_nhwc[bitmask == 0], class_id, scores=...)

What the stage fits, on train rows only
---------------------------------------
It reads the train split through its SplitGuard with the one purpose "qc"
(val and test are refused), the tissue mask and the LOPO folds; nothing else.

1. A reference sample: the TISSUE train rows (tissue-mask stage) of every
   class, all of a target class's (`target_rows` caps it) and a seeded
   `rows_per_class` of each other class. Glass is not what a generator is
   asked for, so it is not what QC compares with.
2. One pass over the sample: float64 band statistics, the raw flattened
   covariance (-> PCA-64 of the band-z-scored patch), the mean/std features,
   each row's byte hash, and exact float16 histograms of the target class's
   values (-> R1's per-band [p0.05, p99.95], per fold). A second pass projects
   the sample on the PCA.
3. A hash of EVERY train row (`hash_all_rows`, one more pass over X_train):
   R6's exact-copy check covers the whole split, glass and other classes too.
4. Per LOPO fold k, from the rows of patients NOT held out in fold k: the
   mean/std probe (R5), the class Gaussian of log mean spectra (R2), R1's band
   range, and the fold's view of the kNN and memorization banks (masks). The
   same once more on all rows: the "full" references for samples from
   generators trained on all of train.
5. Real references: every target-class sample patch of a fold's held-out
   patients is scored EXACTLY as a generated sample is (`QCReference.score`,
   with `fold=k`), and each target patch's exact cross-capture nearest
   neighbour distance is found (R6's reference, and Alaa authenticity).
6. Calibration (`rules.calibrate`) -> thresholds.json, then the real
   cross-patient ranges of the set metrics -> set_reference.json.

Real held-out patches that are byte-identical to a patch on their fold's
train side are DATA duplicates (P1 reports them; the fixture plants one), not
samples of natural variation: they are excluded from the calibration set and
from the cross-capture distance reference, and listed in thresholds.json.
Every other held-out patch counts.

Outputs (`<out>/qc/reference/`)
-------------------------------
    qc_reference.json   what was fitted on what: classes, folds, sample sizes,
                        feature spaces, tissue tau, the build's manifest sha256,
                        the config, the file list
    thresholds.json     the calibrated thresholds, schema below
    features.npz        band_mean/std, pca_mean, pca_components [64, D], msf_mean/std,
                        s_glass (R4's reference, copied so the directory is self-contained)
    sample.npz          the reference sample: row, y, patient, capture, msf_raw [n, 2C],
                        pca [n, 64], hash [n, 2]
    hashes.npz          sorted blake2b-128 of the hashed train rows: h1, h2, row, patient, capture
    probes.npz          fold_<k>_* and full_*: coef, intercept, classes, converged
    class_<name>.npz    per target class: gauss_mean [K+1, C], gauss_prec [K+1, C, C],
                        r1_lo/r1_hi [K+1, C] (index K = full), bank (sample indices),
                        bank_nn_dist, bank_nn_idx, bank_duplicate_row
    calibration_<name>.npz  every real held-out score, row, fold, patient, excluded
    set_reference.json  per class: MMD bandwidths, real cross-patient metric ranges,
                        bank patient shares, held-out NN attribution

thresholds.json (`THRESHOLDS_SCHEMA_VERSION` 1)
-----------------------------------------------
    {"schema_version": 1, "rule_bits": {"R1": 1, ..., "R6": 32}, "rule_names": {...},
     "bitmask_dtype": "uint8", "targets": {"per_rule_accept": 0.99, "joint_accept": 0.95},
     "fixed": {"R1.band_tail": 0.0005, "R1.max_out_fraction": 0.01, "R3.hf_cutoff": 0.25,
               "R4.tau_deg": ..., "R5.knn_k": 10, "R6.nn_candidates": 10, "log_eps": 0.001},
     "classes": {"DCIS": {
         "class_id": 1, "ok": true, "problems": [], "n_calibration": n,
         "n_calibration_patients": 5, "excluded_duplicates": [{row, fold, patient, matches_row}],
         "rule_accept": {"R1": .., ...}, "joint_accept": .., "per_fold": {"0": {...}},
         "loosened": [check, ...], "steps": [...],
         "checks": {"R2.mahalanobis": {"rule", "score", "side", "doc", "nominal_level", "level",
                                       "loosened_steps", "threshold", "bound", "reference",
                                       "n_reference", "calibration_rejects"}, ...}}}}

A check rejects a sample when: side "flag" -> score != 0; "lower" -> not
(score >= threshold); "upper" -> not (score <= threshold). NaN always rejects.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hsi_augment.qc import features as F
from hsi_augment.qc.consistency import LinearProbe, fit_probe, knn_label_agreement
from hsi_augment.qc.memorization import DEFAULT_BLOCK_ELEMS, HashIndex, nearest_neighbours, resolve_device
from hsi_augment.qc.rules import (CHECKS, RULE_BITS, RULE_NAMES, RULES, THRESHOLDS_SCHEMA_VERSION,
                                  CalibrationError, bitmask_from_failures, calibrate, evaluate)

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.stage import StageContext
    from hsi_augment.stages.qc_reference import QCReferenceConfig
    from hsi_augment.tissue import TissueReference

PathLike = Union[str, os.PathLike]

REFERENCE_SCHEMA_VERSION = 1
REFERENCE_NAME = "qc_reference.json"
THRESHOLDS_NAME = "thresholds.json"
FEATURES_NAME = "features.npz"
SAMPLE_NAME = "sample.npz"
HASHES_NAME = "hashes.npz"
PROBES_NAME = "probes.npz"
SET_REFERENCE_NAME = "set_reference.json"
APPLY_STAGE = "qc-apply"          # the SplitGuard log's stage name for reads made by a loaded reference
SAMPLE_RNG_KEY = "reference-sample"

# Per-sample scores `QCReference.score` returns (all arrays of length n).
SCORE_KEYS: Tuple[str, ...] = (
    "nonfinite_frac", "out_of_range_frac", "mahalanobis", "curvature", "lag1_autocorr", "hf_power_frac",
    "tissue_score", "not_tissue", "p_class", "knn_agreement", "exact_match", "exact_match_row",
    "nn_dist", "nn_pca_dist", "nn_row", "nn_patient", "nn_capture", "authentic")


def class_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "class"


def class_file(name: str) -> str:
    return f"class_{class_slug(name)}.npz"


def calibration_file(name: str) -> str:
    return f"calibration_{class_slug(name)}.npz"


def resolve_target_classes(names: Sequence[str], adapter: Any) -> List[int]:
    """Class ids from names (case-insensitive); none = the adapter's minority class(es)."""
    if not names:
        ids = list(adapter.minority_classes)
        if not ids:
            raise ValueError("no target classes given and the adapter declares no minority class")
        return sorted(ids)
    return sorted({adapter.class_index(n) for n in names})


def fit_gaussian(v: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray]:
    """(mean, precision) of the rows of `v`, the covariance ridged by
    `ridge` x its mean variance so a few rows still give an invertible matrix."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v).all(axis=1)]
    if len(v) < 2:
        raise ValueError(f"a Gaussian needs at least 2 rows, got {len(v)}")
    mu = v.mean(axis=0)
    cov = np.atleast_2d(np.cov(v, rowvar=False))
    cov = cov + float(ridge) * (np.trace(cov) / cov.shape[0] + 1e-12) * np.eye(cov.shape[0])
    return mu, np.linalg.inv(cov)


def mahalanobis(v: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    d = np.asarray(v, dtype=np.float64) - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0.0))


def _save_npz(ctx: "StageContext", rel: str, arrays: Mapping[str, np.ndarray]) -> Path:
    """Atomic .npz write recorded as a stage output (no pickles: plain arrays)."""
    path = ctx.path(rel)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **{k: np.asarray(v) for k, v in arrays.items()})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return ctx.add_output(path)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ============================================================================
# the loaded reference
# ============================================================================

@dataclass
class ClassModel:
    """One target class's fitted references; leading axis K + 1 = folds, then full."""
    class_id: int
    name: str
    gauss_mean: np.ndarray          # [K+1, C] log mean spectrum
    gauss_prec: np.ndarray          # [K+1, C, C]
    r1_lo: np.ndarray               # [K+1, C]
    r1_hi: np.ndarray               # [K+1, C]
    bank: np.ndarray                # int64 sample indices of the class's rows, ascending
    bank_nn_dist: np.ndarray        # float64 exact cross-capture NN distance per bank row (inf: none)
    bank_nn_idx: np.ndarray         # int64 bank index of that neighbour (-1: none)
    bank_duplicate_row: np.ndarray  # int64 train row byte-identical to it in another capture (-1: none)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {"class_id": np.asarray(self.class_id), "gauss_mean": self.gauss_mean, "gauss_prec": self.gauss_prec,
                "r1_lo": self.r1_lo, "r1_hi": self.r1_hi, "bank": self.bank, "bank_nn_dist": self.bank_nn_dist,
                "bank_nn_idx": self.bank_nn_idx, "bank_duplicate_row": self.bank_duplicate_row}

    @classmethod
    def from_arrays(cls, name: str, d: Mapping[str, np.ndarray]) -> "ClassModel":
        return cls(class_id=int(d["class_id"]), name=name, gauss_mean=d["gauss_mean"], gauss_prec=d["gauss_prec"],
                   r1_lo=d["r1_lo"], r1_hi=d["r1_hi"], bank=d["bank"].astype(np.int64),
                   bank_nn_dist=d["bank_nn_dist"], bank_nn_idx=d["bank_nn_idx"].astype(np.int64),
                   bank_duplicate_row=d["bank_duplicate_row"].astype(np.int64))


class QCReference:
    """Everything a QC check needs, loaded from the stage directory (or built
    in memory by the stage). Train patches for R6's exact re-rank are read
    from `train_X` (a memmap; pass `ctx.open_split("train", purpose="qc").X`),
    or, when it is not given, opened on first use through a SplitGuard with
    purpose "qc", logged as stage `qc-apply`."""

    def __init__(self, *, stage_dir: PathLike, meta: Mapping[str, Any], spaces: F.FeatureSpaces,
                 sample: Mapping[str, np.ndarray], hashes: HashIndex, probes: Sequence[LinearProbe],
                 classes: Mapping[int, ClassModel], tissue: "TissueReference", heldout: Sequence[np.ndarray],
                 thresholds: Optional[Mapping[str, Any]] = None,
                 set_references: Optional[Mapping[str, Any]] = None, train_X: Optional[np.ndarray] = None):
        self.stage_dir = Path(stage_dir)
        self.meta = dict(meta)
        self.spaces = spaces
        self.sample = {k: np.asarray(v) for k, v in sample.items()}
        self.sample_msf = spaces.msf(self.sample["msf_raw"])
        self.hashes = hashes
        self.probes = list(probes)
        self.classes = dict(classes)
        self.tissue = tissue
        self.heldout = [np.asarray(h, dtype=np.int64) for h in heldout]
        self.thresholds = dict(thresholds) if thresholds is not None else None
        self.set_references = dict(set_references or {})
        self._train_X = train_X
        if train_X is not None:
            self._check_train_X(train_X)

    # -- loading -------------------------------------------------------------------
    @classmethod
    def load(cls, stage_dir: PathLike, *, train_X: Optional[np.ndarray] = None) -> "QCReference":
        from hsi_augment.tissue import TissueReference
        d = Path(stage_dir)
        meta = json.loads((d / REFERENCE_NAME).read_text())
        if meta.get("schema_version") != REFERENCE_SCHEMA_VERSION:
            raise ValueError(f"{d / REFERENCE_NAME}: schema_version {meta.get('schema_version')}, "
                             f"expected {REFERENCE_SCHEMA_VERSION}")
        feats = _load_npz(d / FEATURES_NAME)
        spaces = F.FeatureSpaces.from_arrays(feats)
        probes_npz = _load_npz(d / PROBES_NAME)
        k = int(meta["folds"]["n_folds"])
        probes = [LinearProbe.from_arrays(probes_npz, f"fold_{i}_") for i in range(k)]
        probes.append(LinearProbe.from_arrays(probes_npz, "full_"))
        classes = {}
        for t in meta["target_classes"]:
            classes[int(t["id"])] = ClassModel.from_arrays(t["name"], _load_npz(d / t["file"]))
        tissue = TissueReference(s_glass=feats["s_glass"], tau_deg=float(meta["tissue"]["tau_deg"]),
                                 stage_dir=d)
        return cls(stage_dir=d, meta=meta, spaces=spaces, sample=_load_npz(d / SAMPLE_NAME),
                   hashes=HashIndex.from_arrays(_load_npz(d / HASHES_NAME)), probes=probes, classes=classes,
                   tissue=tissue, heldout=meta["folds"]["heldout_patients"],
                   thresholds=json.loads((d / THRESHOLDS_NAME).read_text()),
                   set_references=json.loads((d / SET_REFERENCE_NAME).read_text()), train_X=train_X)

    # -- small accessors -------------------------------------------------------------
    @property
    def config(self) -> Dict[str, Any]:
        return dict(self.meta.get("config") or {})

    @property
    def device(self) -> str:
        return resolve_device(self.config.get("device", "auto"))

    @property
    def n_folds(self) -> int:
        return len(self.heldout)

    @property
    def class_ids(self) -> List[int]:
        return sorted(self.classes)

    @property
    def out_root(self) -> Path:
        root = self.stage_dir
        for _ in Path(self.meta.get("dirname", "qc/reference")).parts:
            root = root.parent
        return root

    def class_model(self, class_id: int) -> ClassModel:
        try:
            return self.classes[int(class_id)]
        except KeyError:
            raise KeyError(f"class {class_id} has no QC reference; target classes: {self.class_ids} "
                           f"(re-run qc-reference with --target_classes)") from None

    def class_name(self, class_id: int) -> str:
        return self.class_model(class_id).name

    def class_thresholds(self, class_id: int) -> Dict[str, Any]:
        if self.thresholds is None:
            raise ValueError("this reference has no calibrated thresholds yet")
        return self.thresholds["classes"][self.class_name(class_id)]

    def set_reference(self, class_id: int) -> Dict[str, Any]:
        return self.set_references[self.class_name(class_id)]

    def fold_index(self, fold: Optional[int]) -> int:
        if fold is None:
            return self.n_folds
        if not 0 <= int(fold) < self.n_folds:
            raise ValueError(f"fold {fold} out of range: {self.n_folds} LOPO folds")
        return int(fold)

    def heldout_patients(self, fold: Optional[int]) -> Optional[np.ndarray]:
        return None if fold is None else self.heldout[self.fold_index(fold)]

    def _fold_valid(self, patients: np.ndarray, fold: Optional[int]) -> Optional[np.ndarray]:
        held = self.heldout_patients(fold)
        return None if held is None else ~np.isin(patients, held)

    # -- train patches (R6 exact re-rank) ----------------------------------------------
    def _check_train_X(self, X: np.ndarray) -> None:
        want = (int(self.meta["n_train_rows"]), *self.spaces.patch_shape)
        if tuple(X.shape) != want:
            raise ValueError(f"train_X has shape {tuple(X.shape)}; the QC reference was built on {want}")

    def train_X(self) -> np.ndarray:
        if self._train_X is None:
            from hsi_augment.adapters import get_adapter
            from hsi_augment.io import ACCESS_LOG_NAME, SplitGuard, sha256_file
            adapter = get_adapter(self.meta["adapter"], self.meta["data_dir"])
            sha = self.meta.get("input_manifest_sha256")
            if sha and sha256_file(adapter.manifest_path) != sha:
                raise ValueError(f"{adapter.manifest_path} is not the build this QC reference was fitted on")
            guard = SplitGuard(adapter, self.out_root / ACCESS_LOG_NAME, stage=APPLY_STAGE,
                               allowed_purposes=("qc",))
            X = guard.open("train", "qc", kinds=("X",)).X
            self._check_train_X(X)
            self._train_X = X
        return self._train_X

    def sample_z(self, sample_idx: np.ndarray) -> np.ndarray:
        """z-flat train patches of reference-sample rows, read in row order."""
        idx = np.asarray(sample_idx, dtype=np.int64)
        rows = self.sample["row"][idx]
        order = np.argsort(rows, kind="stable")
        data = np.asarray(self.train_X()[rows[order]], dtype=np.float32)
        z = self.spaces.zflat(data)
        out = np.empty_like(z)
        out[order] = z
        return out

    # -- features --------------------------------------------------------------------
    def feature_bundle(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """{mean [n, C] raw patch means, mean_std [n, 2C] standardised, pca [n, k],
        finite [n]} of the float16 export of `x` (non-finite rows hold zeros)."""
        x16 = F.as_export(x)
        self.spaces.check_shape(x16)
        n, c = len(x16), self.spaces.patch_shape[-1]
        out = {"mean": np.zeros((n, c), np.float32), "mean_std": np.zeros((n, 2 * c), np.float32),
               "pca": np.zeros((n, self.spaces.n_components), np.float32), "finite": np.zeros(n, bool)}
        for a, b in F.blocks(n, int(self.config.get("score_chunk_rows", 1024))):
            xb = x16[a:b].astype(np.float32)
            ok = F.nonfinite_fraction(xb) == 0
            if not ok.any():
                continue
            pos = a + np.flatnonzero(ok)
            raw = F.mean_std_features(xb[ok])
            out["mean"][pos] = raw[:, :c]
            out["mean_std"][pos] = self.spaces.msf(raw)
            out["pca"][pos] = self.spaces.pca(xb[ok])
            out["finite"][pos] = True
        return out

    def real_bundle(self, class_id: int, fold: Optional[int] = None) -> Dict[str, np.ndarray]:
        """The real target-class reference patches' features (fold k: its train side)."""
        cm = self.class_model(class_id)
        idx = cm.bank
        valid = self._fold_valid(self.sample["patient"][idx], fold)
        if valid is not None:
            idx = idx[valid]
        c = self.spaces.patch_shape[-1]
        return {"mean": self.sample["msf_raw"][idx, :c], "mean_std": self.sample_msf[idx],
                "pca": self.sample["pca"][idx]}

    # -- scoring -----------------------------------------------------------------------
    def score(self, x: np.ndarray, class_id: int, *, fold: Optional[int] = None, device: Optional[str] = None,
              chunk_rows: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Every per-sample QC score of `x` as samples of `class_id` (keys:
        `SCORE_KEYS`). `x` is cast to the float16 export first. `fold=None`
        scores against the full references, `fold=k` against fold k's (its
        held-out patients' rows removed from every bank and fit)."""
        cfg = self.config
        cm = self.class_model(class_id)
        fi = self.fold_index(fold)
        dev = resolve_device(device or cfg.get("device", "auto"))
        x16 = F.as_export(x)
        self.spaces.check_shape(x16)
        n, c = len(x16), self.spaces.patch_shape[-1]
        segments = F.segments_list(self.meta["spectral_segments"])
        eps = float(self.meta.get("log_eps", F.LOG_EPS))
        out: Dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in
                                      ("nonfinite_frac", "out_of_range_frac", "mahalanobis", "curvature",
                                       "lag1_autocorr", "hf_power_frac", "p_class", "knn_agreement", "nn_dist")}
        out["tissue_score"] = np.full(n, np.nan, dtype=np.float32)
        out["nn_pca_dist"] = np.full(n, np.inf, dtype=np.float32)
        for k in ("not_tissue", "exact_match", "authentic"):
            out[k] = np.zeros(n, dtype=bool)
        for k in ("exact_match_row", "nn_row", "nn_patient", "nn_capture"):
            out[k] = np.full(n, -1, dtype=np.int64)
        held = self.heldout_patients(fold)
        knn_valid = self._fold_valid(self.sample["patient"], fold)
        bank = cm.bank
        bank_valid = self._fold_valid(self.sample["patient"][bank], fold)
        bank_pca = self.sample["pca"][bank]
        probe = self.probes[fi]
        n_classes = len(self.meta["class_names"])
        knn_k = int(cfg.get("knn_k", 10))
        block = int(cfg.get("block_elems", DEFAULT_BLOCK_ELEMS))
        for a, b in F.blocks(n, int(chunk_rows or cfg.get("score_chunk_rows", 1024))):
            xb16 = x16[a:b]
            xb = xb16.astype(np.float32)
            nf = F.nonfinite_fraction(xb)
            out["nonfinite_frac"][a:b] = nf
            out["out_of_range_frac"][a:b] = F.out_of_range_fraction(xb, cm.r1_lo[fi], cm.r1_hi[fi])
            with np.errstate(invalid="ignore", over="ignore"):      # a non-finite patch scores NaN
                t = self.tissue.score(xb16)
                out["tissue_score"][a:b] = t
                out["not_tissue"][a:b] = ~self.tissue.is_tissue(xb16)
            m = self.hashes.lookup(F.patch_hashes(xb16), exclude_patients=held)
            out["exact_match_row"][a:b] = m
            out["exact_match"][a:b] = m >= 0
            ok = nf == 0
            if not ok.any():
                continue
            pos = a + np.flatnonzero(ok)
            xo = xb[ok]
            raw = F.mean_std_features(xo)
            f = self.spaces.msf(raw)
            out["mahalanobis"][pos] = mahalanobis(F.log_spectrum(raw[:, :c], eps), cm.gauss_mean[fi],
                                                  cm.gauss_prec[fi])
            out["curvature"][pos] = F.curvature_energy(xo, segments, eps)
            lag1, hf = F.spatial_stats(xo, float(self.meta.get("hf_cutoff", 0.25)))
            out["lag1_autocorr"][pos], out["hf_power_frac"][pos] = lag1, hf
            out["p_class"][pos] = probe.predict_proba(f, n_classes)[:, int(class_id)]
            agree, _ = knn_label_agreement(f, self.sample_msf, self.sample["y"], int(class_id), knn_k,
                                           bank_valid=knn_valid, device=dev, block_elems=block)
            out["knn_agreement"][pos] = agree
            z = self.spaces.zflat(xo)
            nn = nearest_neighbours(self.spaces.pca(xo), lambda qi: z[qi], bank_pca,
                                    lambda bi: self.sample_z(bank[bi]),
                                    candidates=int(cfg.get("nn_candidates", 10)),
                                    query_chunk=int(cfg.get("query_chunk", 512)), device=dev, block_elems=block,
                                    bank_valid=bank_valid)
            hit = nn.idx >= 0
            srow = np.where(hit, bank[np.where(hit, nn.idx, 0)], -1)
            out["nn_dist"][pos] = np.where(hit, nn.dist, np.nan)
            out["nn_pca_dist"][pos] = nn.approx_dist
            out["nn_row"][pos] = np.where(hit, self.sample["row"][np.where(hit, srow, 0)], -1)
            out["nn_patient"][pos] = np.where(hit, self.sample["patient"][np.where(hit, srow, 0)], -1)
            out["nn_capture"][pos] = np.where(hit, self.sample["capture"][np.where(hit, srow, 0)], -1)
            real_nn = np.where(hit, cm.bank_nn_dist[np.where(hit, nn.idx, 0)], np.inf)
            out["authentic"][pos] = hit & (nn.dist > real_nn)
        return out


# ============================================================================
# the stage
# ============================================================================

def _class_rows(y: np.ndarray, mask: np.ndarray, c: int, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    cand = np.flatnonzero((y == c) & mask)
    if cap is not None and len(cand) > cap:
        cand = np.sort(rng.choice(cand, int(cap), replace=False))
    return cand


def run_qc_reference(ctx: "StageContext", cfg: "QCReferenceConfig") -> Dict[str, Any]:
    """The `qc-reference` stage. See the module docstring."""
    from hsi_augment import tissue as tissue_mod
    from hsi_augment.io import RunningBandStats, iter_chunks
    from hsi_augment.qc import setmetrics
    from hsi_augment.splits import check_folds, fold_heldout_patients, load_folds
    from hsi_augment.stage import StageError, input_manifest_sha256

    adapter = ctx.adapter
    class_names = list(adapter.class_names)
    try:
        targets = resolve_target_classes(cfg.target_classes, adapter)
    except (KeyError, ValueError) as e:
        raise StageError(f"target classes: {e}") from e
    device = resolve_device(cfg.device)
    train = ctx.open_split("train", purpose="qc")
    if train.groups is None or train.captures is None:
        raise StageError("groups_train.npy and captures_train.npy are needed (patients, captures)")
    X = train.X
    n_train = int(train.n)
    y = np.asarray(train.y, dtype=np.int64)
    groups = np.asarray(train.groups, dtype=np.int64)
    captures = np.asarray(train.captures, dtype=np.int64)
    patch_shape = tuple(int(v) for v in X.shape[1:])
    n_bands = patch_shape[-1]
    wavelengths = np.asarray(adapter.wavelengths, dtype=np.float64)
    if len(wavelengths) != n_bands:
        raise StageError(f"wavelengths.npy has {len(wavelengths)} bands, X_train has {n_bands}")
    segments = adapter.spectral_segments()

    # -- upstreams: the tissue mask and reference, the LOPO folds -------------------------
    tdir = ctx.upstream_dir("tissue-mask")
    tissue_mask = np.load(tdir / tissue_mod.MASK_NAME)
    if tissue_mask.shape != (n_train,):
        raise StageError(f"{tissue_mod.MASK_NAME} has shape {tissue_mask.shape}, train has {n_train} rows")
    tau_doc = json.loads((tdir / tissue_mod.TAU_NAME).read_text())
    if tau_doc.get("complete") is False:
        ctx.warn("the tissue-mask stage scored only part of train (max_captures): unscored rows count as glass")
    tref = tissue_mod.load_reference(tdir)
    folds_path = ctx.upstream_dir("leakage")
    folds_doc = load_folds(folds_path)
    problems = check_folds(folds_doc, groups, y, class_names=class_names,
                           minority_classes=adapter.minority_classes, captures=captures)
    if problems:
        raise StageError("LOPO folds do not match this train split: " + "; ".join(problems))
    heldout = [np.asarray(fold_heldout_patients(f), dtype=np.int64) for f in folds_doc["folds"]]
    n_folds = len(heldout)

    # -- 1. the reference sample ------------------------------------------------------------
    rng = ctx.rng(SAMPLE_RNG_KEY)
    parts, per_class = [], {}
    for c in range(len(class_names)):
        cap = cfg.target_rows if c in targets else cfg.rows_per_class
        rows = _class_rows(y, tissue_mask, c, cap, rng)
        parts.append(rows)
        per_class[class_names[c]] = {"tissue_rows": int(((y == c) & tissue_mask).sum()), "sampled": int(len(rows)),
                                     "target": c in targets}
    S = np.sort(np.concatenate(parts)).astype(np.int64)
    S_y, S_p, S_c = y[S], groups[S], captures[S]
    for c in targets:
        name = class_names[c]
        for k, h in enumerate(heldout):
            on_train = int(((S_y == c) & ~np.isin(S_p, h)).sum())
            if on_train < 2:
                raise StageError(f"fold {k}: only {on_train} {name} tissue rows on the train side of the "
                                 f"reference sample; a class reference needs at least 2")
    dim = int(np.prod(patch_shape))
    ctx.ram_guard(len(S) * (4 * (2 * n_bands + cfg.pca_components) + 16) + 2 * dim * dim * 8
                  + (n_train if cfg.hash_all_rows else len(S)) * 40
                  + len(targets) * (n_folds + 1) * n_bands * F.N_CODES * 8
                  + cfg.chunk_rows * dim * 8, "QC reference sample, covariance, hashes and histograms")
    ctx.log(f"reference sample: {len(S)} tissue rows "
            + ", ".join(f"{k} {v['sampled']}/{v['tissue_rows']}" for k, v in per_class.items())
            + f"; target class(es) {[class_names[c] for c in targets]}; {n_folds} LOPO folds")

    # -- 2. pass 1: band stats, raw covariance, mean/std features, hashes, histograms -------
    pca_fit = np.zeros(len(S), dtype=bool)
    pca_fit[ctx.rng("pca-fit").choice(len(S), min(int(cfg.pca_fit_rows), len(S)), replace=False)] = True
    stats = RunningBandStats(n_bands)
    cov = F.CovarianceAccumulator(dim)
    msf_raw = np.empty((len(S), 2 * n_bands), dtype=np.float32)
    s_hash = np.empty((len(S), 2), dtype=np.uint64)
    hist_total = {c: np.zeros((n_bands, F.N_CODES), dtype=np.int64) for c in targets}
    hist_held = {c: np.zeros((n_folds, n_bands, F.N_CODES), dtype=np.int64) for c in targets}
    with ctx.timer("pass1"):
        for ch in iter_chunks(X, S, chunk_rows=cfg.chunk_rows, assume_sorted=True):
            pos = np.searchsorted(S, ch.rows)
            x16 = np.ascontiguousarray(ch.data, dtype=np.float16)
            stats.update(x16)
            msf_raw[pos] = F.mean_std_features(x16)
            s_hash[pos] = F.patch_hashes(x16)
            fit_rows = x16[pca_fit[pos]]
            for a, b in F.blocks(len(fit_rows)):
                cov.update(fit_rows[a:b].reshape(b - a, -1))
            for c in targets:
                mc = S_y[pos] == c
                if not mc.any():
                    continue
                hist_total[c] += F.code_histogram(x16[mc])
                for k in range(n_folds):
                    mk = mc & np.isin(S_p[pos], heldout[k])
                    if mk.any():
                        hist_held[c][k] += F.code_histogram(x16[mk])
    if int(stats.n_nonfinite.sum()):
        raise StageError(f"{int(stats.n_nonfinite.sum())} non-finite values in the train reference sample")
    spaces = F.fit_feature_spaces(patch_shape, stats.mean, stats.std(), cov, msf_raw, cfg.pca_components)
    del cov
    ctx.log(f"feature spaces: PCA-{spaces.n_components} of {dim} z-scored values "
            f"({spaces.describe()['pca_explained_variance_ratio']:.1%} of the variance, fitted on "
            f"{int(pca_fit.sum())} rows); mean/std {2 * n_bands}-d")

    # -- 3. pass 2: PCA coordinates -----------------------------------------------------------
    pca = np.empty((len(S), spaces.n_components), dtype=np.float32)
    with ctx.timer("pass2"):
        for ch in iter_chunks(X, S, chunk_rows=cfg.chunk_rows, assume_sorted=True):
            pca[np.searchsorted(S, ch.rows)] = spaces.pca(ch.data)

    # -- 4. exact hashes of every train row ---------------------------------------------------
    with ctx.timer("hash"):
        if cfg.hash_all_rows:
            h_rows = np.arange(n_train, dtype=np.int64)
            h_all = np.empty((n_train, 2), dtype=np.uint64)
            for ch in iter_chunks(X, None, chunk_rows=cfg.chunk_rows):
                h_all[ch.rows] = F.patch_hashes(ch.data)
        else:
            h_rows, h_all = S, s_hash
        hidx = HashIndex.build(h_all, h_rows, groups[h_rows], captures[h_rows])
        del h_all
    ctx.log(f"hashed {len(hidx)} train rows ({'all' if cfg.hash_all_rows else 'reference sample only'})")

    # -- 5. the cross-fitted probes -----------------------------------------------------------
    msf = spaces.msf(msf_raw)
    prng = ctx.rng("probe-rows")
    lr_sel = np.zeros(len(S), dtype=bool)
    for c in range(len(class_names)):
        idx = np.flatnonzero(S_y == c)
        if len(idx) > cfg.lr_rows_per_class:
            idx = prng.choice(idx, cfg.lr_rows_per_class, replace=False)
        lr_sel[idx] = True
    probes: List[LinearProbe] = []
    with ctx.timer("probes"):
        for k in range(n_folds):
            m = lr_sel & ~np.isin(S_p, heldout[k])
            probes.append(fit_probe(msf[m], S_y[m], max_iter=cfg.lr_max_iter, seed=ctx.seed_int("probe", k)))
        probes.append(fit_probe(msf[lr_sel], S_y[lr_sel], max_iter=cfg.lr_max_iter, seed=ctx.seed_int("probe", "full")))
    if not all(p.converged for p in probes):
        ctx.warn(f"QC probe did not converge in {cfg.lr_max_iter} iterations for "
                 f"{[i for i, p in enumerate(probes) if not p.converged]} (index {n_folds} = full)")

    # -- 6. per target class: Gaussians and R1 ranges, per fold and full ----------------------
    classes: Dict[int, ClassModel] = {}
    for c in targets:
        bank = np.flatnonzero(S_y == c).astype(np.int64)
        logm = F.log_spectrum(msf_raw[bank, :n_bands], F.LOG_EPS)
        gm, gp, lo, hi = [], [], [], []
        for k in range(n_folds + 1):
            keep = np.ones(len(bank), bool) if k == n_folds else ~np.isin(S_p[bank], heldout[k])
            mu, prec = fit_gaussian(logm[keep], cfg.gauss_ridge)
            counts = hist_total[c] if k == n_folds else hist_total[c] - hist_held[c][k]
            l, h = F.histogram_quantiles(counts, cfg.r1_band_tail)
            gm.append(mu), gp.append(prec), lo.append(l), hi.append(h)
        classes[c] = ClassModel(class_id=c, name=class_names[c], gauss_mean=np.stack(gm), gauss_prec=np.stack(gp),
                                r1_lo=np.stack(lo), r1_hi=np.stack(hi), bank=bank,
                                bank_nn_dist=np.full(len(bank), np.inf), bank_nn_idx=np.full(len(bank), -1),
                                bank_duplicate_row=np.full(len(bank), -1, dtype=np.int64))
    del hist_total, hist_held

    meta: Dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA_VERSION, "stage": ctx.name, "dirname": ctx.spec.dirname,
        "adapter": adapter.name, "data_dir": str(Path(adapter.data_dir).resolve()),
        "input_manifest_sha256": input_manifest_sha256(adapter.manifest_path),
        "n_train_rows": n_train, "class_names": class_names, "patch_shape": list(patch_shape),
        "wavelengths": wavelengths.tolist(), "spectral_segments": [list(s) for s in segments],
        "log_eps": F.LOG_EPS, "hf_cutoff": cfg.hf_cutoff, "config": cfg.to_dict(),
        "target_classes": [{"id": c, "name": class_names[c], "file": class_file(class_names[c]),
                            "calibration_file": calibration_file(class_names[c]),
                            "n_bank": int(len(classes[c].bank))} for c in targets],
        "folds": {"n_folds": n_folds, "heldout_patients": [h.tolist() for h in heldout],
                  "dcis_patients": folds_doc.get("dcis_patients"), "source": "prep/leakage/folds.json",
                  "seed": folds_doc.get("seed")},
        "tissue": {"tau_deg": float(tref.tau_deg), "complete": tau_doc.get("complete", True),
                   "source": "prep/tissue/{s_glass.npy, tau.json}"},
        "sample": {"n_rows": int(len(S)), "per_class": per_class,
                   "rule": "train TISSUE rows (tissue-mask stage); every target-class row up to target_rows, a "
                           "seeded rows_per_class of every other class"},
        "feature_spaces": spaces.describe(),
        "hashes": {"scope": "all train rows" if cfg.hash_all_rows else "reference sample", "n_rows": len(hidx),
                   "digest": f"blake2b-{8 * F.HASH_BYTES} of the float16 NHWC bytes"},
        "probes": {"converged": [bool(p.converged) for p in probes], "fit_rows_full": int(lr_sel.sum()),
                   "fit": "scripts.shallow_probe_heldout.fit on standardised "
                          "scripts.shallow_baseline.featurize(x, 'mean_std')"},
        "rule_bits": RULE_BITS, "rule_names": RULE_NAMES,
        "checks": {c.name: {"rule": c.rule, "score": c.score, "side": c.side, "doc": c.doc} for c in CHECKS},
    }
    sample = {"row": S, "y": S_y, "patient": S_p, "capture": S_c, "msf_raw": msf_raw, "pca": pca, "hash": s_hash}
    ref = QCReference(stage_dir=ctx.stage_dir, meta=meta, spaces=spaces, sample=sample, hashes=hidx,
                      probes=probes, classes=classes, tissue=tref, heldout=heldout, train_X=X)

    # -- 7. cross-capture nearest neighbours and data duplicates of the class banks ------------
    with ctx.timer("cross_capture_nn"):
        for c in targets:
            cm = classes[c]
            bank = cm.bank
            nn = nearest_neighbours(pca[bank], lambda qi, b=bank: ref.sample_z(b[qi]), pca[bank],
                                    lambda bi, b=bank: ref.sample_z(b[bi]), candidates=cfg.nn_candidates,
                                    query_chunk=cfg.query_chunk, device=device, block_elems=cfg.block_elems,
                                    query_groups=S_c[bank], bank_groups=S_c[bank])
            cm.bank_nn_dist, cm.bank_nn_idx = nn.dist, nn.idx
            cm.bank_duplicate_row = hidx.lookup(s_hash[bank], exclude_capture=S_c[bank])

    # -- 8. real LOPO held-out scores and the calibration -------------------------------------
    thresholds: Dict[str, Any] = {
        "schema_version": THRESHOLDS_SCHEMA_VERSION, "rule_bits": RULE_BITS, "rule_names": RULE_NAMES,
        "bitmask_dtype": "uint8",
        "targets": {"per_rule_accept": cfg.rule_accept, "joint_accept": cfg.joint_accept},
        "fixed": {"R1.band_tail": cfg.r1_band_tail, "R1.max_out_fraction": cfg.r1_max_out_fraction,
                  "R3.hf_cutoff": cfg.hf_cutoff, "R4.tau_deg": float(tref.tau_deg), "R5.knn_k": cfg.knn_k,
                  "R6.nn_candidates": cfg.nn_candidates, "log_eps": F.LOG_EPS},
        "classes": {}}
    calib: Dict[int, Dict[str, np.ndarray]] = {}
    with ctx.timer("calibration"):
        for c in targets:
            cm = classes[c]
            parts_c: List[Dict[str, np.ndarray]] = []
            for k in range(n_folds):
                sel = cm.bank[np.isin(S_p[cm.bank], heldout[k])]
                for ch in iter_chunks(X, S[sel], chunk_rows=cfg.chunk_rows, assume_sorted=True):
                    sc = ref.score(ch.data, c, fold=k, device=device)
                    sc.update(row=ch.rows, fold=np.full(len(ch.rows), k, dtype=np.int64),
                              patient=groups[ch.rows])
                    parts_c.append(sc)
            if not parts_c:
                raise StageError(f"no {class_names[c]} tissue rows among the LOPO held-out patients: nothing "
                                 f"to calibrate on")
            cal = {key: np.concatenate([p[key] for p in parts_c]) for key in parts_c[0]}
            excluded = cal["exact_match"].copy()
            cal["excluded"] = excluded
            calib[c] = cal
            used = {key: v[~excluded] for key, v in cal.items()}
            dup_bank = cm.bank_duplicate_row >= 0
            cross = cm.bank_nn_dist[~dup_bank]
            references = {chk.name: (cross if chk.reference == "cross_capture" else used[chk.score])
                          for chk in CHECKS if chk.side != "flag"}
            nominal = {chk.name: float(getattr(cfg, chk.level_param)) for chk in CHECKS if chk.side != "flag"}
            bounds = {chk.name: float(getattr(cfg, chk.bound_param)) for chk in CHECKS if chk.bound_param}
            doc = calibrate(used, references, nominal, bounds, rule_accept=cfg.rule_accept,
                            joint_accept=cfg.joint_accept, max_halvings=cfg.max_halvings)
            fails = evaluate(doc, used)
            mask = bitmask_from_failures(fails, int((~excluded).sum()))
            per_fold = {}
            for k in range(n_folds):
                mk = used["fold"] == k
                if mk.any():
                    per_fold[str(k)] = {"n": int(mk.sum()),
                                        "rule_accept": {r: float(((mask[mk] & RULE_BITS[r]) == 0).mean())
                                                        for r in RULES},
                                        "joint_accept": float((mask[mk] == 0).mean())}
            doc = {"class_id": int(c), **doc,
                   "n_calibration_patients": int(len(np.unique(used["patient"]))),
                   "excluded_duplicates": [{"row": int(r), "fold": int(f), "patient": int(p), "matches_row": int(m)}
                                           for r, f, p, m in zip(cal["row"][excluded], cal["fold"][excluded],
                                                                 cal["patient"][excluded],
                                                                 cal["exact_match_row"][excluded])],
                   "cross_capture_duplicates": int(dup_bank.sum()),
                   "per_fold": per_fold}
            thresholds["classes"][class_names[c]] = doc
            if excluded.any():
                ctx.warn(f"{class_names[c]}: {int(excluded.sum())} real held-out patch(es) are byte-identical to "
                         f"a patch on their fold's train side (data duplicates), excluded from calibration: rows "
                         f"{cal['row'][excluded][:10].tolist()}")
            ctx.log(f"{class_names[c]}: calibrated on {doc['n_calibration']} real held-out patches of "
                    f"{doc['n_calibration_patients']} patients: per rule "
                    + ", ".join(f"{r} {v:.3f}" for r, v in doc["rule_accept"].items())
                    + f", joint {doc['joint_accept']:.3f}; loosened {doc['loosened'] or 'none'}")
    ref.thresholds = thresholds
    ctx.save_json(THRESHOLDS_NAME, thresholds)
    for c in targets:
        _save_npz(ctx, calibration_file(class_names[c]), calib[c])
    bad = {name: d["problems"] for name, d in thresholds["classes"].items() if not d["ok"]}
    if bad:
        raise CalibrationError("QC calibration cannot meet the targets (each rule >= "
                               f"{cfg.rule_accept:.0%}, all rules >= {cfg.joint_accept:.0%} of real held-out "
                               f"patches): " + "; ".join(f"{n}: {' | '.join(p)}" for n, p in bad.items()))

    # -- 9. the set metrics' real cross-patient ranges -----------------------------------------
    set_refs: Dict[str, Any] = {}
    with ctx.timer("set_reference"):
        for c in targets:
            cm = classes[c]
            used = {k: v[~calib[c]["excluded"]] for k, v in calib[c].items()}
            held_att = {str(k): setmetrics.attribution(used["nn_patient"][used["fold"] == k], cfg.attribution_flag)
                        for k in range(n_folds) if (used["fold"] == k).any()}
            set_refs[class_names[c]] = setmetrics.real_reference(
                ref.real_bundle(c), S_p[cm.bank], k=cfg.prdc_k, prdc_rows=cfg.prdc_rows, mmd_rows=cfg.mmd_rows,
                rng=ctx.rng("set-reference", c), device=device, heldout_attribution=held_att)
    ref.set_references = set_refs
    ctx.save_json(SET_REFERENCE_NAME, set_refs)

    # -- 10. the arrays, then the description ---------------------------------------------------
    _save_npz(ctx, FEATURES_NAME, {**spaces.to_arrays(), "s_glass": np.asarray(tref.s_glass, dtype=np.float64)})
    _save_npz(ctx, SAMPLE_NAME, sample)
    _save_npz(ctx, HASHES_NAME, hidx.to_arrays())
    probe_arrays: Dict[str, np.ndarray] = {}
    for k, p in enumerate(probes[:-1]):
        probe_arrays.update(p.to_arrays(f"fold_{k}_"))
    probe_arrays.update(probes[-1].to_arrays("full_"))
    _save_npz(ctx, PROBES_NAME, probe_arrays)
    for c in targets:
        _save_npz(ctx, class_file(class_names[c]), classes[c].to_arrays())
    meta["files"] = sorted([THRESHOLDS_NAME, SET_REFERENCE_NAME, FEATURES_NAME, SAMPLE_NAME, HASHES_NAME,
                            PROBES_NAME] + [t["file"] for t in meta["target_classes"]]
                           + [t["calibration_file"] for t in meta["target_classes"]])
    ctx.save_json(REFERENCE_NAME, meta)

    return {
        "target_classes": [class_names[c] for c in targets],
        "n_reference_rows": int(len(S)),
        "n_hashed_rows": len(hidx),
        "pca_components": spaces.n_components,
        "n_folds": n_folds,
        "probes_converged": all(p.converged for p in probes),
        "classes": {class_names[c]: {
            "n_bank": int(len(classes[c].bank)),
            "n_calibration": thresholds["classes"][class_names[c]]["n_calibration"],
            "n_excluded_duplicates": len(thresholds["classes"][class_names[c]]["excluded_duplicates"]),
            "rule_accept": thresholds["classes"][class_names[c]]["rule_accept"],
            "joint_accept": thresholds["classes"][class_names[c]]["joint_accept"],
            "loosened": thresholds["classes"][class_names[c]]["loosened"]} for c in targets},
    }
