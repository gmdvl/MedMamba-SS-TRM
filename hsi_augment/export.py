# -*- coding: utf-8 -*-
"""
hsi_augment/export.py
=====================
Phase 9 (#29): the training export of one source, `export-<src>` (registered
in `hsi_augment/stages/export.py`), from the selection of `qc-<src>` and the
samples of `generate-<src>` (`hsi_augment/generate.py`).

The schema is NOT defined here: it is `training/synthetic_mix.py`'s (#7), and
the arrays, the manifest and provenance.json are written by its one writer,
`write_synthetic_export`, so what `train.py --synthetic_dir` loads
(`load_synthetic_export`) and what this stage writes cannot drift apart. Real
data is referenced (by the sha256 of the build's dataset_manifest.json), never
copied; there is no merged copy.

    <out>/export/<src>/
        X_syn.npy              float16 (N, H, W, C) NHWC, VALUE space (raw counts / corpus
                               median, the build's scale: the generators decode log-z to it)
        y_syn.npy              int64 (N,)
        groups_syn.npy         int32 (N,): -1 cvae, -2 ldm, never a patient id
        gen_index.npy          int64 (N,): each row's gen_index in synth/<src>/X_gen.npy
        wavelengths.npy, class_names.json   byte copies of the real build's
        dataset_manifest.json  write_dataset_manifest(sha256=True)
        provenance.json        written LAST (by the writer); keys below
        qc_scores.npz          every GENERATED sample (accepted or not): its bitmask and scores
        qc_report.json         acceptance, rejections per rule, the thresholds, set metrics, the
                               tau_s / w choice
        figures/               spectra_<class>.png, contact_sheet_<class>.png, nn_gallery_<class>.png

N is the selection's size: per target class, exactly the train split's count
of that class (122,850 DCIS rows on trainsel). Rows are in generation order.

provenance.json: the writer's keys (schema_version, export_id = sha256(X_syn)[:12],
source, group_id, source_manifest_sha256, n_rows, hwc, class_counts,
created_utc) plus generator_ckpt_sha256, generator (the record, tau_s or w and
how it was chosen), config_hash (over the three stages' config hashes) and
config_hashes, stage_fingerprints, git_sha, seeds, n_generated, n_accepted,
n_target, acceptance, thresholds (thresholds.json sha256 and every check's
threshold), stats_sha256 (the generator's log-space codec, the pool, the QC
feature spaces), target_rule, selection_rule and real_data_dir.

Determinism: X_syn, y_syn, groups_syn and gen_index are functions of the
upstream outputs only, so a re-run reproduces their sha256 and the export_id
(`created_utc` in provenance.json and the manifest are wall-clock times).

Split access: the figures show real train patches (the QC reference sample,
and each sample's nearest train patch), read through the stage's SplitGuard
with the train-only purpose "qc".
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

import numpy as np

from hsi_augment.config import canonical_json, sha256_hex
from hsi_augment.generate import (CHOICE_NAME, GENERATION_NAME, GEN_META_NAME, PARAM_NAMES, QC_REFERENCE_STAGE,
                                  QC_SUMMARY_NAME, SELECTED_NAME, SELECTION_RULE, SET_METRICS_NAME, TARGET_RULE,
                                  X_GEN_NAME, load_npz, load_reference, stage_name, upstream_done)
from hsi_augment.io import sha256_file
from hsi_augment.stage import StageError, git_sha, read_json

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.qc.reference import QCReference
    from hsi_augment.stage import StageContext

EXPORT_SCHEMA_NOTE = "training/synthetic_mix.py SCHEMA_VERSION (write_synthetic_export)"
QC_REPORT_SCHEMA_VERSION = 1


def _copy_file(src: Path, dst: Path) -> Path:
    """Byte copy, tmp + os.replace."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def patch_mean_spectra(x: np.ndarray, block: int = 4096) -> np.ndarray:
    """(n, C) per-patch mean spectra of NHWC patches, float64 accumulation."""
    out = np.empty((len(x), x.shape[-1]), dtype=np.float64)
    for a in range(0, len(x), block):
        out[a:a + block] = np.asarray(x[a:a + block], dtype=np.float64).mean(axis=(1, 2))
    return out


def class_thresholds(thresholds: Mapping[str, Any], class_name: str) -> Dict[str, Any]:
    d = thresholds["classes"][class_name]
    return {name: {"rule": c.get("rule"), "side": c.get("side"), "threshold": c.get("threshold")}
            for name, c in d["checks"].items()}


def _read_rows(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """X[rows] read in sorted row order (memmap friendly), returned in `rows` order."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    out = np.empty((len(rows), *X.shape[1:]), dtype=X.dtype)
    out[order] = np.asarray(X[rows[order]])
    return out


def write_figures(ctx: "StageContext", ref: "QCReference", X_syn: np.ndarray, y_syn: np.ndarray,
                  scores: Mapping[str, np.ndarray], selected: np.ndarray, classes: Sequence[int], source: str,
                  n_show: int) -> List[Path]:
    """The three figures per class (hsi_augment/export_figures.py) into figures/."""
    from hsi_augment import export_figures as EF
    from training.recon_artifacts import resolve_rgb_bands
    from training.synthetic_mix import FIGURES_DIR
    adapter = ctx.adapter
    names = list(adapter.class_names)
    wl = np.asarray(adapter.wavelengths, dtype=np.float64)
    segments = adapter.spectral_segments()
    bands, rgb_note = resolve_rgb_bands(adapter.n_bands, wl)
    train_X = ref.train_X()
    out: List[Path] = []
    for c in classes:
        slug = "".join(ch if ch.isalnum() else "_" for ch in names[c].lower())
        rows = np.flatnonzero(y_syn == c)
        cm = ref.class_model(c)
        real_spec = ref.sample["msf_raw"][cm.bank, :adapter.n_bands].astype(np.float64)
        out.append(EF.plot_class_spectra(ctx.path(FIGURES_DIR, f"spectra_{slug}.png"), wl, segments, real_spec,
                                         patch_mean_spectra(X_syn[rows]), names[c], source))
        pick = ctx.rng("figures", c).choice(len(cm.bank), size=min(int(n_show), len(cm.bank)), replace=False)
        real = _read_rows(train_X, ref.sample["row"][cm.bank[np.sort(pick)]])
        stretch = EF.rgb_stretch(real, bands)
        show = rows[:int(n_show)]
        out.append(EF.plot_contact_sheet(ctx.path(FIGURES_DIR, f"contact_sheet_{slug}.png"), real, X_syn[show],
                                         bands, stretch, rgb_note,
                                         f"{names[c]}: real train tissue (top) vs {source} synthetic (first "
                                         f"{len(show)} exported rows)"))
        gen_rows = selected[show]
        nn_row = np.asarray(scores["nn_row"])[gen_rows]
        has = nn_row >= 0
        nearest = np.zeros_like(X_syn[show], dtype=np.float32)
        if has.any():
            nearest[has] = _read_rows(train_X, nn_row[has]).astype(np.float32)
        dist = [float(d) if h else None for d, h in zip(np.asarray(scores["nn_dist"])[gen_rows], has)]
        out.append(EF.plot_nn_gallery(ctx.path(FIGURES_DIR, f"nn_gallery_{slug}.png"), X_syn[show], nearest, dist,
                                      bands, stretch, rgb_note,
                                      f"{names[c]}: {source} synthetic vs its nearest real train patch"))
    return out


def run_export(ctx: "StageContext", cfg: Any, source: str) -> Dict[str, Any]:
    from training.npy_atomic import MANIFEST_NAME
    from training.synthetic_mix import (CLASS_NAMES_NAME, EXPORT_ARRAYS, PROVENANCE_NAME, QC_REPORT_NAME,
                                        QC_SCORES_NAME, SYNTHETIC_GROUP_IDS, WAVELENGTHS_NAME,
                                        SyntheticExportError, write_synthetic_export)
    from hsi_augment.qc.reference import FEATURES_NAME, THRESHOLDS_NAME

    gen_name, qc_name = stage_name("generate", source), stage_name("qc", source)
    gen_dir, qc_dir = ctx.upstream_dir(gen_name), ctx.upstream_dir(qc_name)
    gen_done, qc_done = upstream_done(ctx, gen_name), upstream_done(ctx, qc_name)
    gen_doc = read_json(gen_dir / GENERATION_NAME)
    choice = read_json(gen_dir / CHOICE_NAME)
    qc_doc = read_json(qc_dir / QC_SUMMARY_NAME)
    set_doc = read_json(qc_dir / SET_METRICS_NAME)
    if not gen_doc or not choice or not qc_doc or set_doc is None:
        raise StageError(f"{gen_name} / {qc_name} outputs are missing; re-run them")
    meta = load_npz(gen_dir / GEN_META_NAME)
    X_gen = np.load(gen_dir / X_GEN_NAME, mmap_mode="r")
    selected = np.load(qc_dir / SELECTED_NAME)
    names = list(ctx.adapter.class_names)
    classes = sorted(int(c) for c in gen_doc["targets"])
    n_target = {c: int(gen_doc["targets"][str(c)]["n_target"]) for c in classes}

    n_gen = int(gen_doc["n_generated"])
    if X_gen.shape[0] != n_gen or len(meta["gen_index"]) != n_gen:
        raise StageError(f"{gen_dir}: X_gen.npy has {X_gen.shape[0]} rows, generation.json {n_gen}")
    if selected.ndim != 1 or (selected.size and (np.any(np.diff(selected) <= 0) or selected[0] < 0
                                                  or selected[-1] >= n_gen)):
        raise StageError(f"{qc_dir / SELECTED_NAME} is not an ascending set of generated rows")
    y_syn = meta["class_id"][selected].astype(np.int64)
    counts = {c: int((y_syn == c).sum()) for c in classes}
    if counts != n_target or selected.size != sum(n_target.values()):
        raise StageError(f"the selection holds {counts} rows per class, the target is {n_target}")

    ctx.ram_guard(int(selected.size) * int(np.prod(X_gen.shape[1:])) * 2 * 3, "the export's X_syn")
    with ctx.timer("gather"):
        X_syn = np.asarray(X_gen[selected])
    gen_index = meta["gen_index"][selected].astype(np.int64)

    # extras first: provenance.json, written last by the writer, marks a complete export
    _copy_file(qc_dir / QC_SCORES_NAME, ctx.path(QC_SCORES_NAME))
    ctx.add_output(ctx.path(QC_SCORES_NAME))
    ref = load_reference(ctx)
    scores = load_npz(qc_dir / QC_SCORES_NAME)
    ref_dir = ctx.upstream_dir(QC_REFERENCE_STAGE)
    thresholds = read_json(ref_dir / THRESHOLDS_NAME) or {}
    if cfg.figures:
        with ctx.timer("figures"):
            for p in write_figures(ctx, ref, X_syn, y_syn, scores, selected, classes, source, cfg.figure_samples):
                ctx.add_output(p)

    param = PARAM_NAMES[source]
    value = float(gen_doc["value"])
    per_class = {}
    for c in classes:
        q = qc_doc["classes"][names[c]]
        per_class[names[c]] = {"class_id": c, "n_target": n_target[c], "n_generated": q["n_generated"],
                               "n_accepted": q["n_accepted"], "acceptance": q["acceptance"],
                               "rejected_by": q.get("rejected_by"), "n_selected": q["n_selected"],
                               "last_selected_gen_index": q.get("last_selected_gen_index"),
                               "thresholds": class_thresholds(thresholds, names[c]),
                               "set_metrics": set_doc.get(names[c])}
    n_generated = int(sum(v["n_generated"] for v in per_class.values()))
    n_accepted = int(sum(v["n_accepted"] for v in per_class.values()))
    ctx.save_json(QC_REPORT_NAME, {
        "schema_version": QC_REPORT_SCHEMA_VERSION, "source": source, param: value,
        "selection_rule": SELECTION_RULE, "min_acceptance": qc_doc["min_acceptance"],
        "rule_bits": qc_doc["rule_bits"], "classes": per_class, "set_metric_flags": qc_doc["set_metric_flags"],
        "choice": {k: v for k, v in choice.items() if k != "records"},
        "choice_records": choice.get("records", []), "thresholds_sha256": qc_doc["thresholds_sha256"],
        "qc_reference_fingerprint": qc_doc["qc_reference_fingerprint"]})

    config_hashes = {gen_name: gen_done["config_hash"], qc_name: qc_done["config_hash"],
                     ctx.name: ctx.config.config_hash()}
    gen_rec = dict(gen_doc["generator"])
    extra = {
        "pipeline_schema": EXPORT_SCHEMA_NOTE,
        "generator_ckpt_sha256": gen_rec["checkpoint_sha256"],
        "generator": {**gen_rec, "param": param, "value": value, "choice_source": choice["source"],
                      "choice_rule": choice.get("rule"), "candidates": choice.get("candidates")},
        param: value,
        "config_hash": sha256_hex(canonical_json(config_hashes)),
        "config_hashes": config_hashes,
        "stage_fingerprints": {gen_name: gen_done["fingerprint"], qc_name: qc_done["fingerprint"],
                               ctx.name: ctx.fingerprint},
        "git_sha": git_sha(),
        "seeds": {"master": int(ctx.seed),
                  "generate": {names[int(c)]: int(v["seed"]) for c, v in gen_doc["targets"].items()},
                  "lopo_choice": choice.get("seeds")},
        "n_generated": n_generated, "n_accepted": n_accepted,
        "n_target": {names[c]: n_target[c] for c in classes},
        "acceptance": n_accepted / n_generated if n_generated else None,
        "thresholds": {"thresholds_sha256": qc_doc["thresholds_sha256"],
                       "min_acceptance": qc_doc["min_acceptance"],
                       "per_class": {names[c]: per_class[names[c]]["thresholds"] for c in classes}},
        "stats_sha256": {"log_space_codec": gen_rec.get("codec_sha256"), "pool": gen_rec.get("pool_sha256"),
                         "qc_features": sha256_file(ref_dir / FEATURES_NAME),
                         "norm_stats": "global_zscore of the real X_train, fitted at train time "
                                       "(bound by source_manifest_sha256)"},
        "qc_reference_fingerprint": qc_doc["qc_reference_fingerprint"],
        "target_rule": TARGET_RULE, "n_per_class_override": gen_doc.get("n_per_class_override"),
        "selection_rule": SELECTION_RULE, "overgenerate": gen_doc["overgenerate"],
        "real_data_dir": str(Path(ctx.data_dir).resolve()),
    }
    with ctx.timer("write"):
        try:
            prov = write_synthetic_export(str(ctx.stage_dir), X_syn, y_syn, source, str(ctx.data_dir),
                                          gen_index=gen_index, extra_provenance=extra)
        except SyntheticExportError as e:
            raise StageError(f"the export was refused by its writer: {e}") from e
    manifest = read_json(ctx.stage_dir / MANIFEST_NAME) or {}
    files = manifest.get("files") or {}
    for name in (*EXPORT_ARRAYS, WAVELENGTHS_NAME):
        if (ctx.stage_dir / name).is_file():
            ctx.add_output(ctx.stage_dir / name, sha256=(files.get(name) or {}).get("sha256"))
    for name in (CLASS_NAMES_NAME, MANIFEST_NAME, PROVENANCE_NAME):
        if (ctx.stage_dir / name).is_file():
            ctx.add_output(ctx.stage_dir / name)
    if int(prov["group_id"]) != SYNTHETIC_GROUP_IDS[source] or int(prov["group_id"]) >= 0:
        raise StageError(f"group id {prov['group_id']} is not the source's negative id")
    ctx.log(f"export {prov['export_id']}: {prov['n_rows']} rows {prov['class_counts']}, group id "
            f"{prov['group_id']}, {param} {value:g}, acceptance {extra['acceptance']:.1%}")
    return {"source": source, "export_id": prov["export_id"], "n_rows": int(prov["n_rows"]),
            "group_id": int(prov["group_id"]), param: value, "n_generated": n_generated, "n_accepted": n_accepted,
            "acceptance": extra["acceptance"], "class_counts": prov["class_counts"]}
