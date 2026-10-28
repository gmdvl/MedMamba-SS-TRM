# -*- coding: utf-8 -*-
"""
hsi_augment/qc/consistency.py
=============================
R5, class consistency (Plan §Phase 8b, issue #28): does a sample generated for
class c look like class c to something that is NOT the classifier being
evaluated?

Why an independent, cross-fitted probe
--------------------------------------
QC never depends on the classifier under evaluation (Plan rule 7): filtering
synthetic DCIS by what MedMamba already calls DCIS would select samples that
cannot teach it anything new, and would make the A0-vs-augmented comparison
circular. The QC classifier is the repository's shallow probe instead:
`scripts.shallow_probe_heldout.fit` (a class-balanced multinomial
LogisticRegression) on `scripts.shallow_baseline.featurize(x, "mean_std")`
features. Both are imported; `shallow_baseline.py` is frozen.

`fit` featurises the patches it is given. The mean/std features are computed
once, chunked, while the reference sample is streamed (`features.
mean_std_features` calls the same `featurize`), so no patch array is held in
RAM; they are handed to `fit` as 1 x 1 "patches" with `kind="mean"`, whose
featurisation is the identity. The fitted coefficients are kept as plain
arrays (`LinearProbe`), so the stage directory holds no pickle and
`predict_proba` is reproduced in numpy (checked against sklearn in the tests).

Cross-fitting: one probe per LOPO fold, fitted on the fold's train side
(every row of the fold's held-out patients removed), scores that fold's
held-out patients. Those out-of-patient scores are the real reference R5's
thresholds are calibrated on. A "full" probe fitted on every reference row
scores samples from generators trained on all of train (`fold=None`); a fold
probe scores samples from that fold's generators (#30), so a held-out patient
never shapes what QC keeps for its own fold.

The second half of R5 is a k = 10 nearest-neighbour label agreement in the
same standardised mean/std space: the fraction of a sample's 10 nearest real
reference patches (all classes, class-balanced sample) that carry class c.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from hsi_augment.qc.memorization import DEFAULT_BLOCK_ELEMS, knn_search

DEFAULT_MAX_ITER = 1500          # shallow_probe_heldout.py's --max_iter


@dataclass
class LinearProbe:
    """A fitted LogisticRegression as arrays: coef [K, F] (K = 1 for two
    classes), intercept [K], classes [K'] (the label values it saw)."""
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    converged: bool = True

    def decision(self, f: np.ndarray) -> np.ndarray:
        return np.asarray(f, dtype=np.float64) @ self.coef.T.astype(np.float64) + self.intercept

    def predict_proba(self, f: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        """Class probabilities as sklearn computes them (softmax for more than
        two classes, logistic for two), with one column per class id
        0..num_classes-1 (0 for a class the probe never saw)."""
        z = self.decision(f)
        if self.coef.shape[0] == 1:
            p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
            p = np.stack([1.0 - p1, p1], axis=1)
        else:
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
        k = int(num_classes if num_classes is not None else int(self.classes.max()) + 1)
        out = np.zeros((len(p), k))
        out[:, self.classes.astype(np.int64)] = p
        return out

    def to_arrays(self, prefix: str = "") -> Dict[str, np.ndarray]:
        return {f"{prefix}coef": self.coef, f"{prefix}intercept": self.intercept,
                f"{prefix}classes": self.classes, f"{prefix}converged": np.asarray(self.converged)}

    @classmethod
    def from_arrays(cls, d: Mapping[str, Any], prefix: str = "") -> "LinearProbe":
        return cls(coef=np.asarray(d[f"{prefix}coef"], dtype=np.float64),
                   intercept=np.asarray(d[f"{prefix}intercept"], dtype=np.float64),
                   classes=np.asarray(d[f"{prefix}classes"], dtype=np.int64),
                   converged=bool(np.asarray(d[f"{prefix}converged"])))


def fit_probe(features: np.ndarray, y: np.ndarray, *, max_iter: int = DEFAULT_MAX_ITER,
              seed: int = 0) -> LinearProbe:
    """`shallow_probe_heldout.fit` on precomputed (standardised) mean/std
    features. They go in as [n, 1, 1, F] with kind "mean", which
    `shallow_baseline.featurize` reduces back to [n, F] unchanged."""
    from sklearn.exceptions import ConvergenceWarning

    from scripts import shallow_probe_heldout as sp
    f = np.asarray(features, dtype=np.float32)
    yy = np.asarray(y, dtype=np.int64)
    if len(np.unique(yy)) < 2:
        raise ValueError("the QC probe needs rows of at least two classes")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        # sklearn >= 1.8 warns that the probe's n_jobs=-1 has no effect; harmless.
        warnings.filterwarnings("ignore", message=".*n_jobs.*", category=FutureWarning)
        clf = sp.fit("mean", f[:, None, None, :], yy, int(max_iter), int(seed))
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return LinearProbe(coef=np.asarray(clf.coef_, dtype=np.float64),
                       intercept=np.asarray(clf.intercept_, dtype=np.float64),
                       classes=np.asarray(clf.classes_, dtype=np.int64), converged=converged)


def knn_label_agreement(query: np.ndarray, bank: np.ndarray, bank_y: np.ndarray, class_id: int, k: int, *,
                        bank_valid: Optional[np.ndarray] = None, device: Optional[str] = "auto",
                        block_elems: int = DEFAULT_BLOCK_ELEMS) -> Tuple[np.ndarray, np.ndarray]:
    """(agreement, neighbour bank indices): the fraction of each query's `k`
    nearest eligible bank rows whose label is `class_id` (float64 [n]; NaN
    when there is no eligible neighbour)."""
    _, idx = knn_search(query, bank, k, device=device, block_elems=block_elems, bank_valid=bank_valid)
    labels = np.asarray(bank_y, dtype=np.int64)
    ok = idx >= 0
    hits = np.where(ok, labels[np.where(ok, idx, 0)] == int(class_id), False)
    n_ok = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = np.where(n_ok > 0, hits.sum(axis=1) / np.maximum(n_ok, 1), np.nan)
    return agree, idx
