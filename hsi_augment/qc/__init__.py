# -*- coding: utf-8 -*-
"""
hsi_augment/qc/
===============
Quality control of generated samples (Plan §Phase 8b, issue #28). The
`qc-reference` stage (`stages/qc_reference.py`) fits everything on train rows
and calibrates the thresholds on real LOPO held-out patches; #29 (generation
and export) and #30 (LOPO utility) apply it:

    from hsi_augment.qc import QCReference, apply_rules, set_metrics
    ref = QCReference.load(out / "qc/reference", train_X=ctx.open_split("train", purpose="qc").X)
    bitmask, scores = apply_rules(ref, x_f16_nhwc, class_id)         # uint8 per sample, 0 = accepted
    bitmask, scores = apply_rules(ref, x_f16_nhwc, class_id, fold=k) # samples of fold k's generators
    report = set_metrics(ref, x_f16_nhwc[bitmask == 0], class_id)

Modules
-------
    features.py       the export cast, per-patch features, PCA-64 and mean/std spaces, hashes
    consistency.py    R5: the cross-fitted shallow probe, kNN label agreement
    memorization.py   R6: hash index, chunked torch.cdist search, exact re-rank
    rules.py          R1-R6 checks, the bitmask, calibration, `apply_rules`
    setmetrics.py     set-level metrics with bootstrap CIs and real cross-patient ranges
    reference.py      `QCReference` and the stage logic

Importing this package is cheap (numpy); torch, scipy and sklearn load inside
the functions that need them.
"""

from hsi_augment.qc.reference import (REFERENCE_NAME, SCORE_KEYS, SET_REFERENCE_NAME, THRESHOLDS_NAME,
                                      QCReference)
from hsi_augment.qc.rules import (CHECKS, RULE_BITS, RULE_NAMES, RULES, CalibrationError, apply_rules,
                                  decode_bitmask, summarize_bitmask)
from hsi_augment.qc.setmetrics import set_metrics

__all__ = ["CHECKS", "REFERENCE_NAME", "RULE_BITS", "RULE_NAMES", "RULES", "SCORE_KEYS", "SET_REFERENCE_NAME",
           "THRESHOLDS_NAME", "CalibrationError", "QCReference", "apply_rules", "decode_bitmask",
           "set_metrics", "summarize_bitmask"]
