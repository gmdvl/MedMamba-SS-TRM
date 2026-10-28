# -*- coding: utf-8 -*-
"""
hsi_augment/generate.py
=======================
Phase 8a (#29): generating one source's synthetic minority patches, and
applying the QC of Phase 8b (#28) to them. Two stages per source, registered
in `hsi_augment/stages/generate.py`; the export (Phase 9) is `export.py`:

    generate-<src>  -> <out>/synth/<src>/   the chosen tau_s / w, then 2x the target, in generation order
    qc-<src>        -> <out>/qc/<src>/      R1-R6 on every generated sample, the selection, set metrics
    export-<src>    -> <out>/export/<src>/  the training export (export.py)

<src> is `cvae` (arm A2, `models/cvae.py`) or `ldm` (arm A3,
`latent_diffusion.load_generator`). Both are read through `GeneratorHandle`,
so everything below is source-agnostic.

The target: 1:1 with the real minority class
--------------------------------------------
Target classes come from `training.minority_augmentation.resolve_minority_classes`
(`--target_classes auto`, the default: DCIS) on the TRAIN split's labels, and
each target class gets exactly as many accepted synthetic rows as the train
split has real rows of it (`target_counts`). On trainsel that is 122,850 DCIS
rows per source, so every arm trains on the same rows per epoch, steps and EMA
rate. On the test fixture it is the fixture's own DCIS count: the rule, not
the number, is the contract. `--n_per_class` overrides it for smoke runs only
(provenance then records the override).

Generation (`generate-<src>`)
-----------------------------
`overgenerate` x the target (2x: 245,700 rows on trainsel) per class, at one
tau_s (CVAE) or guidance scale w (LDM), from the FINAL generator. Noise comes
from CPU generators (`train_generative.noise_rows`): row r of class c is a
function of (weights, seed_c, value, r) only, so a re-run reproduces
`X_gen.npy` byte for byte, on any batch size for the CVAE and on the same
`batch_size` for the LDM (`models/diffusion.sample`). seed_c =
`ctx.seed_int("generate", c)`.

    X_gen.npy        float16 (n, P, P, C) NHWC, VALUE space (raw counts / corpus median, the
                     build's own scale; the codec's decode, clipped to [0, clip_hi]); classes in
                     ascending order, each class's rows in stream order
    gen_meta.npz     per generated sample: gen_index (its row in X_gen), class_id, stream_row,
                     seed, param (tau_s or w); and source, param_name, checkpoint_sha256
    generation.json  targets, seeds, the generator (checkpoint sha256, record), the value, timings
    choice.json      how the value was chosen (below)

Choosing tau_s and w on LOPO QC
-------------------------------
The fits leave the choice open: the CVAE's model.json records `tau_s: null`
and its candidates (0.8, 1.0), the LDM's generator.pt its guidance candidates
(1, 1.5, 2, 3). In order of precedence the value is:
  1. `--tau_s` / `--guidance` (choice.json source "config");
  2. a non-null value in the generator's own record (source "generator record");
  3. the LOPO QC choice (source "lopo_qc"), on train-only LOPO folds: for
     every candidate and every LOPO fold whose generator was fitted (CVAE: the
     chosen cell's folds; LDM: ldm-lopo's folds), `choice_rows` samples of each
     target class from that fold's generator, scored by R1-R6 against fold
     k's QC references (`apply_rules(fold=k)`, which never saw fold k's
     held-out patients). The accepted samples are compared with the held-out
     patients' real tissue patches of the class (the QC reference sample):
     precision, recall, density and coverage (Naeem et al. 2020, k =
     qc-reference's prdc_k) in the PCA-64 and mean/std spaces. Memorization is
     R6's rejection rate. Among the candidates whose mean acceptance is at
     least `min_acceptance` (40%), the highest mean PCA coverage of the
     held-out patients wins; ties go to the higher acceptance, then to the
     smaller value (smaller tau_s, weaker guidance: w > 3 pulls samples to the
     training modes and is refused). No eligible candidate is a StageError.
     The same noise is used for every candidate of a (fold, class): common
     random numbers, so the comparison is the values', not the draws'.

QC and selection (`qc-<src>`)
-----------------------------
Every generated sample is scored by R1-R6 against the FULL references
(`apply_rules(fold=None)`; the final generator saw every train patient), in
chunks of `chunk_rows`. Per target class, the selection is the first
n_target accepted samples in generation order (deterministic). The stage
FAILS LOUDLY, with no done.json, when a class's acceptance is below
`min_acceptance` (40%): the generator or its value is not usable. It also
fails when acceptance is above that floor but the accepted rows still fall
short of the target (40-50% at 2x); the message names the `--overgenerate`
that would suffice.

    qc_scores.npz    EVERY generated sample: gen_index, class_id, bitmask (uint8, 0 = accepted,
                     bit i = rule R(i+1)), accepted, selected, every score of
                     `QCReference.score` and every `fail.<check>`
    selected.npy     int64 gen_index of the selected rows, ascending (= generation order)
    set_metrics.json per class: `qc.set_metrics` of the selected set against the real train
                     patches, each with its real cross-patient range; the attribution flag
    qc_summary.json  per class: generated, accepted, acceptance, rejections per rule, selected,
                     the thresholds' sha256; the QC reference's fingerprint

Split access
------------
`generate-<src>` declares "generator" (the train labels, to count the
target) and "qc" (the train patches a QC reference re-ranks against);
`qc-<src>` declares "qc". Both purposes are train-only, so the SplitGuard
refuses val and test, and the access log shows train only.

Heavy imports (torch, the models, the QC library) happen inside the
functions; `stages/generate.py` imports none of them.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hsi_augment.config import canonical_json, sha256_hex
from hsi_augment.io import sha256_file
from hsi_augment.stage import StageError, read_done, read_json

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.qc.reference import QCReference
    from hsi_augment.stage import StageContext

PathLike = Union[str, os.PathLike]

SOURCES: Tuple[str, ...] = ("cvae", "ldm")
PARAM_NAMES: Dict[str, str] = {"cvae": "tau_s", "ldm": "guidance"}
# (LOPO stage, final stage) of each source's generator
GENERATOR_STAGES: Dict[str, Tuple[str, str]] = {"cvae": ("cvae-lopo", "cvae-final"),
                                                "ldm": ("ldm-lopo", "ldm-final")}
QC_REFERENCE_STAGE = "qc-reference"

X_GEN_NAME = "X_gen.npy"
GEN_META_NAME = "gen_meta.npz"
GENERATION_NAME = "generation.json"
CHOICE_NAME = "choice.json"
SELECTED_NAME = "selected.npy"
QC_SUMMARY_NAME = "qc_summary.json"
SET_METRICS_NAME = "set_metrics.json"
GENERATION_SCHEMA_VERSION = 1

MAX_GUIDANCE = 3.0          # models.diffusion.MAX_GUIDANCE; w > 3 pulls samples to the training modes
TARGET_RULE = ("each target class (resolve_minority_classes on the train labels) gets exactly as many accepted "
               "synthetic rows as the train split has real rows of it (1:1)")
SELECTION_RULE = "the first n_target QC-accepted samples of each class, in generation order"
CHOICE_RULE = ("LOPO QC: per candidate, the mean over (LOPO fold, target class) of R1-R6 acceptance (fold k's "
               "references, fold k's generator) and of the PRDC coverage (PCA space) of the fold's held-out "
               "patients' real tissue patches by the accepted samples; among the candidates with mean "
               "acceptance >= min_acceptance, the highest mean coverage; ties: higher acceptance, then the "
               "smaller value")


def stage_name(kind: str, source: str) -> str:
    """`generate-cvae`, `qc-ldm`, `export-cvae`, ..."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    return f"{kind}-{source}"


# ============================================================================
# small helpers
# ============================================================================

def _save_npz(ctx: "StageContext", rel: str, arrays: Mapping[str, np.ndarray]) -> Path:
    """Atomic `.npz` (plain arrays, no pickles) recorded as a stage output."""
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


def load_npz(path: PathLike) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _finite_or_none(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(v)) if v else None


def torch_device(name: str) -> Any:
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise StageError(f"--device {name}, but CUDA is not available")
    return torch.device(name)


def upstream_done(ctx: "StageContext", name: str) -> Dict[str, Any]:
    done = read_done(ctx.upstream_dir(name))
    if not done or done.get("status") != "ok":
        raise StageError(f"upstream {name!r} has no valid done.json")
    return done


def relative_to_out(ctx: "StageContext", path: PathLike) -> str:
    """`path` relative to the output root when it is inside it (provenance
    then carries no machine-specific prefix), else as given."""
    p, root = Path(path).resolve(), Path(ctx.out_root).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


def codec_sha256(codec: Any) -> str:
    """sha256 of the canonical JSON of a `LogSpaceCodec` (the log-space stats and clip)."""
    return sha256_hex(canonical_json(codec.to_dict()))


# ============================================================================
# the target
# ============================================================================

def target_counts(y_train: np.ndarray, num_classes: int, class_names: Sequence[str], spec: str = "auto",
                  n_per_class: Optional[int] = None) -> Dict[int, int]:
    """{class id: accepted rows to export}: the classes `resolve_minority_classes(spec)`
    picks on the train labels, each with its train count (1:1), or `n_per_class`."""
    from training.minority_augmentation import resolve_minority_classes
    y = np.asarray(y_train, dtype=np.int64)
    try:
        classes = resolve_minority_classes(spec, y, int(num_classes), list(class_names))
    except ValueError as e:
        raise StageError(f"target classes {spec!r}: {e}") from e
    if not classes:
        raise StageError(f"target classes {spec!r} resolve to no class on the train counts "
                         f"{np.bincount(y, minlength=num_classes).tolist()}")
    counts = np.bincount(y, minlength=int(num_classes))
    return {int(c): int(n_per_class if n_per_class is not None else counts[c]) for c in classes}


def generate_counts(targets: Mapping[int, int], overgenerate: float) -> Dict[int, int]:
    """Rows to generate per class: ceil(overgenerate x target)."""
    return {int(c): int(math.ceil(float(overgenerate) * int(n) - 1e-9)) for c, n in targets.items()}


# ============================================================================
# generators
# ============================================================================

@dataclass
class GeneratorHandle:
    """One trained generator, behind a source-agnostic `sample`. `sample(c, n,
    value, seed, start)` returns rows [start, start + n) of class c's stream
    as float16 NHWC in value space."""
    source: str
    param_name: str
    candidates: Tuple[float, ...]
    preset: Optional[float]              # a value the generator's own record fixes, else None
    ckpt_path: Path
    ckpt_sha256: str
    codec: Any                           # LogSpaceCodec
    n_classes: int
    patch: int
    n_bands: int
    record: Dict[str, Any]
    sample_fn: Callable[[int, int, float, int, int], np.ndarray] = field(repr=False)

    def sample(self, class_id: int, n: int, value: float, seed: int, start: int = 0) -> np.ndarray:
        x = self.sample_fn(int(class_id), int(n), float(value), int(seed), int(start))
        want = (int(n), self.patch, self.patch, self.n_bands)
        if x.dtype != np.float16 or tuple(x.shape) != want:
            raise StageError(f"{self.source} generator returned {x.dtype} {tuple(x.shape)}, expected float16 {want}")
        return x


def load_cvae_handle(run_dir: PathLike, device: Any) -> GeneratorHandle:
    """A CVAE fit's run directory (`gen/cvae/final` or a LOPO fold's), EMA weights."""
    from hsi_augment.models.cvae import MODEL_JSON, load_cvae, sample
    d = Path(run_dir)
    meta = read_json(d / MODEL_JSON)
    if meta is None:
        raise StageError(f"{d / MODEL_JSON} is missing or unreadable")
    ckpt = d / str(meta.get("checkpoint") or "best.pt")
    model, codec, meta = load_cvae(d, checkpoint=ckpt.name, ema=True, device=device)

    def fn(c: int, n: int, v: float, seed: int, start: int) -> np.ndarray:
        return sample(model, codec, c, n, v, seed, start)

    tau = meta.get("tau_s")
    return GeneratorHandle(
        source="cvae", param_name="tau_s", candidates=tuple(float(t) for t in meta.get("tau_s_candidates") or ()),
        preset=None if tau is None else float(tau), ckpt_path=ckpt, ckpt_sha256=sha256_file(ckpt), codec=codec,
        n_classes=int(model.n_classes), patch=int(model.patch), n_bands=int(model.in_bands),
        record={"format": meta.get("format"), "model_json_sha256": sha256_file(d / MODEL_JSON),
                "cell": meta.get("cell"), "model": meta.get("model"), "weights": meta.get("weights", "ema"),
                "pool_sha256": meta.get("pool_sha256"), "codec_sha256": codec_sha256(codec)},
        sample_fn=fn)


def load_ldm_handle(path: PathLike, device: Any, *, steps: Optional[int], batch_size: int) -> GeneratorHandle:
    """An LDM `generator.pt` (or its run directory): autoencoder + EMA denoiser + codec."""
    from hsi_augment.latent_diffusion import GENERATOR_NAME, load_generator
    p = Path(path)
    if p.is_dir():
        p = p / GENERATOR_NAME
    gen = load_generator(p, device=device)
    n_steps = int(steps or gen.meta.get("sample_steps") or 50)

    def fn(c: int, n: int, w: float, seed: int, start: int) -> np.ndarray:
        return gen.sample(c, n, w, seed, start, n_steps, batch_size=int(batch_size))

    return GeneratorHandle(
        source="ldm", param_name="guidance",
        candidates=tuple(float(w) for w in gen.meta.get("guidance_candidates") or ()), preset=None,
        ckpt_path=p, ckpt_sha256=sha256_file(p), codec=gen.codec, n_classes=int(gen.denoiser.num_classes),
        patch=int(gen.ae.patch), n_bands=int(gen.ae.in_channels),
        record={"format": gen.meta.get("format"), "sample_steps": n_steps, "eta": 0.0, "batch_size": int(batch_size),
                "ae_sha256": gen.meta.get("ae_sha256"), "pool_sha256": gen.meta.get("pool_sha256"),
                "codec_sha256": codec_sha256(gen.codec)},
        sample_fn=fn)


def final_handle(ctx: "StageContext", source: str, cfg: Any, device: Any) -> GeneratorHandle:
    final_dir = ctx.upstream_dir(GENERATOR_STAGES[source][1])
    if source == "cvae":
        return load_cvae_handle(final_dir, device)
    return load_ldm_handle(final_dir, device, steps=cfg.sample_steps, batch_size=cfg.batch_size)


@dataclass(frozen=True)
class FoldGenerator:
    """A LOPO fold's generator, loaded on demand (one fold in memory at a time)."""
    fold: int
    heldout_patients: Tuple[int, ...]
    load: Callable[[], GeneratorHandle] = field(repr=False, compare=False)


def lopo_generators(ctx: "StageContext", source: str, cfg: Any, device: Any) -> List[FoldGenerator]:
    """The LOPO fold generators of the fit the final model was refitted from."""
    lopo_dir = ctx.upstream_dir(GENERATOR_STAGES[source][0])
    out: List[FoldGenerator] = []
    if source == "cvae":
        from hsi_augment.fit_cvae import LOPO_SUMMARY_NAME, SELECTION_NAME
        selection = read_json(ctx.upstream_dir("cvae-final") / SELECTION_NAME)
        summary = read_json(lopo_dir / LOPO_SUMMARY_NAME)
        if not selection or not summary or selection.get("chosen") not in summary.get("cells", {}):
            raise StageError("cannot find cvae-final's chosen cell in cvae-lopo's lopo_summary.json")
        for f in summary["cells"][selection["chosen"]]["folds"]:
            run_dir = lopo_dir / f["run_dir"]
            out.append(FoldGenerator(int(f["fold"]), tuple(sorted(int(p) for p in f["heldout_patients"])),
                                     lambda d=run_dir: load_cvae_handle(d, device)))
    else:
        from hsi_augment.latent_diffusion import GENERATOR_NAME
        done = upstream_done(ctx, GENERATOR_STAGES[source][0])
        for f in (done.get("summary") or {}).get("folds") or []:
            rec = f["fold"]
            path = lopo_dir / f"fold-{int(rec['fold'])}" / GENERATOR_NAME
            out.append(FoldGenerator(int(rec["fold"]), tuple(sorted(int(p) for p in rec["heldout_patients"])),
                                     lambda p=path: load_ldm_handle(p, device, steps=cfg.sample_steps,
                                                                    batch_size=cfg.batch_size)))
    if not out:
        raise StageError(f"{GENERATOR_STAGES[source][0]} has no LOPO fold generator to choose "
                         f"{PARAM_NAMES[source]} on; pass --{PARAM_NAMES[source]}")
    return out


def check_value(source: str, value: float) -> None:
    v = float(value)
    if not math.isfinite(v):
        raise StageError(f"{PARAM_NAMES[source]} must be finite, got {value!r}")
    if source == "cvae" and not v > 0:
        raise StageError(f"tau_s must be > 0, got {v}")
    if source == "ldm" and not 0.0 <= v <= MAX_GUIDANCE:
        raise StageError(f"guidance w must be in [0, {MAX_GUIDANCE:g}] (w > {MAX_GUIDANCE:g} pulls samples to the "
                         f"training modes), got {v}")


# ============================================================================
# QC helpers
# ============================================================================

def load_reference(ctx: "StageContext") -> "QCReference":
    """The qc-reference stage's references, with the train patches opened
    through THIS stage's SplitGuard (purpose "qc")."""
    from hsi_augment.qc import QCReference
    train_X = ctx.open_split("train", purpose="qc", kinds=("X",)).X
    try:
        return QCReference.load(ctx.upstream_dir(QC_REFERENCE_STAGE), train_X=train_X)
    except (OSError, ValueError, KeyError) as e:
        raise StageError(f"cannot load the QC reference: {e}") from e


def check_targets_have_reference(ref: "QCReference", class_ids: Sequence[int], class_names: Sequence[str]) -> None:
    missing = [class_names[c] for c in class_ids if c not in ref.class_ids]
    if missing:
        raise StageError(f"target class(es) {missing} have no QC reference (its classes: "
                         f"{[class_names[c] for c in ref.class_ids]}); re-run qc-reference with --target_classes")


def score_in_chunks(ref: "QCReference", X: np.ndarray, class_id: int, *, start: int = 0, stop: Optional[int] = None,
                    fold: Optional[int] = None, device: Optional[str] = None, chunk_rows: int = 4096
                    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """`qc.apply_rules` over rows [start, stop) of `X` (a memmap is read one
    chunk at a time). Returns (bitmask uint8, scores), both over those rows."""
    from hsi_augment import qc as qc_lib
    stop = len(X) if stop is None else int(stop)
    masks: List[np.ndarray] = []
    parts: List[Dict[str, np.ndarray]] = []
    for a in range(int(start), stop, int(chunk_rows)):
        b = min(stop, a + int(chunk_rows))
        m, s = qc_lib.apply_rules(ref, np.asarray(X[a:b]), int(class_id), fold=fold, device=device)
        masks.append(np.asarray(m, dtype=np.uint8))
        parts.append(s)
    if not masks:
        return np.zeros(0, dtype=np.uint8), {}
    return np.concatenate(masks), {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def qc_fold_index(ref: "QCReference", heldout_patients: Sequence[int]) -> int:
    """The QC reference's fold with exactly these held-out patients."""
    want = {int(p) for p in heldout_patients}
    for i, h in enumerate(ref.heldout):
        if {int(p) for p in h} == want:
            return i
    raise StageError(f"no QC reference fold holds out patients {sorted(want)} (its folds: "
                     f"{[sorted(int(p) for p in h) for h in ref.heldout]}): the generator and qc-reference were "
                     f"built on different LOPO folds")


def heldout_bundle(ref: "QCReference", class_id: int, fold: int) -> Dict[str, np.ndarray]:
    """Features of the real tissue patches of `class_id` of fold `fold`'s
    held-out patients (the QC reference sample's class bank)."""
    cm = ref.class_model(class_id)
    idx = cm.bank[np.isin(ref.sample["patient"][cm.bank], ref.heldout[fold])]
    c = ref.spaces.patch_shape[-1]
    return {"mean": ref.sample["msf_raw"][idx, :c], "mean_std": ref.sample_msf[idx], "pca": ref.sample["pca"][idx]}


def heldout_prdc(ref: "QCReference", x_accepted: np.ndarray, real: Mapping[str, np.ndarray], *, seed: int,
                 device: str) -> Dict[str, Optional[float]]:
    """Precision/recall/density/coverage of `x_accepted` against `real`, per
    feature space; None where either side has <= k rows."""
    from hsi_augment.qc import setmetrics as SM
    k = int(ref.config.get("prdc_k", 5))
    cap = int(ref.config.get("prdc_rows", 5000))
    out: Dict[str, Optional[float]] = {f"{m}_{sp}": None for sp in SM.SPACES for m in SM.PRDC_NAMES}
    if len(x_accepted) <= k or len(real["pca"]) <= k:
        return out
    syn = ref.feature_bundle(x_accepted)
    keep = syn.pop("finite")
    syn = SM.take(syn, np.flatnonzero(keep))
    if len(syn["pca"]) <= k:
        return out
    rng = np.random.default_rng(seed)

    def sub(n: int) -> np.ndarray:
        return np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))

    for sp in SM.SPACES:
        si, ri = sub(len(syn[sp])), sub(len(real[sp]))
        ind = SM.prdc_indicators(np.asarray(real[sp])[ri], np.asarray(syn[sp])[si], k, device=device)
        out.update({f"{m}_{sp}": _finite_or_none(ind[m].mean()) for m in SM.PRDC_NAMES})
    return out


def _rule_rejections(bitmask: np.ndarray) -> Dict[str, float]:
    from hsi_augment.qc import RULE_BITS, RULES
    m = np.asarray(bitmask, dtype=np.int64)
    return {r: (float(((m & RULE_BITS[r]) != 0).mean()) if m.size else 0.0) for r in RULES}


# ============================================================================
# choosing tau_s / w on LOPO QC
# ============================================================================

def rank_candidates(records: Sequence[Mapping[str, Any]], candidates: Sequence[float],
                    min_acceptance: float) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """(per-candidate aggregates, the chosen value or None) from the per
    (fold, class, candidate) records; see CHOICE_RULE."""
    table = []
    for v in candidates:
        rs = [r for r in records if float(r["value"]) == float(v)]
        acc = _mean([r["acceptance"] for r in rs])
        cov = [r.get("coverage_pca") for r in rs]
        table.append({
            "value": float(v), "records": len(rs), "acceptance": acc,
            "coverage_pca": _mean(cov), "coverage_pca_missing": sum(c is None for c in cov),
            "recall_pca": _mean([r.get("recall_pca") for r in rs]),
            "precision_pca": _mean([r.get("precision_pca") for r in rs]),
            "density_pca": _mean([r.get("density_pca") for r in rs]),
            "coverage_mean_std": _mean([r.get("coverage_mean_std") for r in rs]),
            "r6_reject": _mean([r["rejected_by"]["R6"] for r in rs]),
            # a (fold, class) whose accepted samples are too few for PRDC covers nothing
            "score": float(np.mean([c if c is not None else 0.0 for c in cov])) if cov else 0.0,
            "eligible": acc is not None and acc >= float(min_acceptance)})
    eligible = [t for t in table if t["eligible"]]
    if not eligible:
        return table, None
    best = min(eligible, key=lambda t: (-t["score"], -(t["acceptance"] or 0.0), t["value"]))
    return table, best["value"]


def choose_value(ctx: "StageContext", cfg: Any, source: str, final: GeneratorHandle, ref: "QCReference",
                 class_ids: Sequence[int], device: Any, qc_device: str) -> Dict[str, Any]:
    """The value of tau_s (CVAE) or w (LDM) and the record of how it was
    chosen (module docstring, 'Choosing tau_s and w on LOPO QC')."""
    param = PARAM_NAMES[source]
    base = {"param": param, "candidates": list(final.candidates)}
    configured = getattr(cfg, param)
    if configured is not None:
        if final.candidates and float(configured) not in final.candidates:
            ctx.warn(f"--{param} {configured:g} is not one of the fit's candidates {list(final.candidates)}")
        return {**base, "source": "config", "value": float(configured)}
    if final.preset is not None:
        return {**base, "source": "generator record", "value": float(final.preset)}
    if not final.candidates:
        raise StageError(f"the {source} generator records no {param} candidates; pass --{param}")
    names = list(ctx.adapter.class_names)
    records: List[Dict[str, Any]] = []
    seeds: Dict[str, Dict[str, int]] = {}
    for fg in lopo_generators(ctx, source, cfg, device):
        qk = qc_fold_index(ref, fg.heldout_patients)
        gen = fg.load()
        for c in class_ids:
            seed = ctx.seed_int("choice", fg.fold, c)
            seeds.setdefault(str(fg.fold), {})[names[c]] = seed
            real = heldout_bundle(ref, c, qk)
            for v in final.candidates:
                with ctx.timer("choice"):
                    x = gen.sample(c, cfg.choice_rows, v, seed, 0)
                    mask, scores = score_in_chunks(ref, x, c, fold=qk, device=qc_device, chunk_rows=cfg.qc_chunk_rows)
                    ok = mask == 0
                    rec = {"fold": fg.fold, "qc_fold": qk, "heldout_patients": list(fg.heldout_patients),
                           "class_id": int(c), "class": names[c], "value": float(v), "n": int(len(x)),
                           "n_accepted": int(ok.sum()), "acceptance": float(ok.mean()) if len(x) else 0.0,
                           "rejected_by": _rule_rejections(mask),
                           "authentic_frac": _finite_or_none(np.mean(scores["authentic"][ok])) if ok.any() else None,
                           "n_heldout_real": int(len(real["pca"])),
                           **heldout_prdc(ref, x[ok], real, seed=seed, device=qc_device)}
                records.append(rec)
                ctx.log_metrics({"event": "lopo_choice", **{k: rec[k] for k in ("fold", "class", "value",
                                                                                 "acceptance", "coverage_pca")}})
                ctx.log(f"LOPO QC fold {fg.fold} {names[c]} {param}={v:g}: acceptance {rec['acceptance']:.1%}, "
                        f"held-out coverage (PCA) {rec['coverage_pca']}, R6 {rec['rejected_by']['R6']:.1%}")
        del gen
    table, chosen = rank_candidates(records, final.candidates, cfg.min_acceptance)
    doc = {**base, "source": "lopo_qc", "rule": CHOICE_RULE, "min_acceptance": float(cfg.min_acceptance),
           "choice_rows": int(cfg.choice_rows), "seeds": seeds, "table": table, "records": records, "value": chosen}
    if chosen is None:
        ctx.save_json(CHOICE_NAME, doc)
        raise StageError(f"LOPO QC: no {param} candidate reaches a mean acceptance of {cfg.min_acceptance:.0%} "
                         f"(" + ", ".join(f"{t['value']:g}: {t['acceptance']:.1%}" for t in table
                                          if t["acceptance"] is not None) + "); the generator is not usable")
    return doc


# ============================================================================
# selection
# ============================================================================

class AcceptanceError(StageError):
    """QC accepted too few generated samples."""


def select_accepted(bitmask: np.ndarray, n_target: int, *, min_acceptance: float, overgenerate: float,
                    what: str = "") -> Tuple[np.ndarray, Dict[str, Any]]:
    """Positions (in generation order) of the first `n_target` accepted
    samples, and a record. AcceptanceError below `min_acceptance`, or when the
    accepted samples cannot fill the target."""
    m = np.asarray(bitmask)
    n = int(m.size)
    accepted = np.flatnonzero(m == 0)
    acceptance = accepted.size / n if n else 0.0
    rec = {"n_generated": n, "n_accepted": int(accepted.size), "acceptance": float(acceptance),
           "n_target": int(n_target), "min_acceptance": float(min_acceptance)}
    tag = f"{what}: " if what else ""
    if n == 0 or acceptance < float(min_acceptance):
        raise AcceptanceError(f"{tag}QC accepted {accepted.size} of {n} generated samples ({acceptance:.1%}), below "
                              f"the {min_acceptance:.0%} floor: the generator (or its tau_s / w) is not usable")
    if accepted.size < int(n_target):
        need = math.ceil(105.0 * int(n_target) / max(acceptance, 1e-12) / max(int(n_target), 1)) / 100
        raise AcceptanceError(f"{tag}QC accepted {accepted.size} of {n} generated samples ({acceptance:.1%}): fewer "
                              f"than the {n_target} needed. Re-run the generate stage with --overgenerate {need:g} "
                              f"(now {overgenerate:g})")
    sel = accepted[:int(n_target)]
    rec.update(n_selected=int(sel.size), last_selected_position=int(sel[-1]) if sel.size else None)
    return sel, rec


# ============================================================================
# stage generate-<src>
# ============================================================================

def run_generate(ctx: "StageContext", cfg: Any, source: str) -> Dict[str, Any]:
    import torch
    adapter = ctx.adapter
    names = list(adapter.class_names)
    device = torch_device(cfg.device)
    qc_device = "cuda" if device.type == "cuda" else "cpu"

    train = ctx.open_split("train", purpose="generator", kinds=("y",))
    targets = target_counts(train.y, adapter.num_classes, names, cfg.target_classes, cfg.n_per_class)
    n_gen = generate_counts(targets, cfg.overgenerate)
    classes = sorted(targets)
    ctx.log(f"targets ({cfg.target_classes}): " + ", ".join(f"{names[c]} {targets[c]} accepted <- {n_gen[c]} "
                                                          f"generated" for c in classes)
            + (f" (--n_per_class {cfg.n_per_class})" if cfg.n_per_class is not None else " (1:1 with train)"))

    final = final_handle(ctx, source, cfg, device)
    if (final.patch, final.n_bands) != (adapter.patch_size, adapter.n_bands) or final.n_classes != adapter.num_classes:
        raise StageError(f"the {source} generator makes {final.patch}x{final.patch}x{final.n_bands} patches of "
                         f"{final.n_classes} classes; the build has {adapter.patch_size}x{adapter.patch_size}x"
                         f"{adapter.n_bands} and {adapter.num_classes}")
    ref = load_reference(ctx)
    check_targets_have_reference(ref, classes, names)

    with ctx.timer("choice"):
        choice = choose_value(ctx, cfg, source, final, ref, classes, device, qc_device)
    value = float(choice["value"])
    check_value(source, value)
    ctx.save_json(CHOICE_NAME, choice)
    ctx.log(f"{final.param_name} = {value:g} ({choice['source']})")

    seeds = {c: ctx.seed_int("generate", c) for c in classes}
    P, C = adapter.patch_size, adapter.n_bands
    total = int(sum(n_gen.values()))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    with ctx.timer("generate"), ctx.npy_writer(X_GEN_NAME, np.float16, (total, P, P, C)) as w:
        for c in classes:
            for a in range(0, n_gen[c], int(cfg.batch_size)):
                m = min(int(cfg.batch_size), n_gen[c] - a)
                w.write(final.sample(c, m, value, seeds[c], a))
            ctx.log(f"{names[c]}: {n_gen[c]} samples")
    wall = time.time() - t0
    peak = (torch.cuda.max_memory_allocated(device) / 2 ** 20) if device.type == "cuda" else None

    class_id = np.concatenate([np.full(n_gen[c], c, dtype=np.int64) for c in classes])
    stream_row = np.concatenate([np.arange(n_gen[c], dtype=np.int64) for c in classes])
    _save_npz(ctx, GEN_META_NAME, {
        "gen_index": np.arange(total, dtype=np.int64), "class_id": class_id, "stream_row": stream_row,
        "seed": np.array([seeds[int(c)] for c in class_id], dtype=np.int64),
        "param": np.full(total, value, dtype=np.float64),
        "source": np.array(source), "param_name": np.array(final.param_name),
        "checkpoint_sha256": np.array(final.ckpt_sha256)})
    rel_ckpt = relative_to_out(ctx, final.ckpt_path)
    doc = {
        "schema_version": GENERATION_SCHEMA_VERSION, "source": source, "param": final.param_name, "value": value,
        "choice_source": choice["source"], "target_rule": TARGET_RULE, "target_classes_spec": cfg.target_classes,
        "n_per_class_override": cfg.n_per_class, "overgenerate": float(cfg.overgenerate),
        "train_counts": {names[c]: int(v) for c, v in
                         enumerate(np.bincount(np.asarray(train.y, dtype=np.int64), minlength=adapter.num_classes))},
        "targets": {str(c): {"class": names[c], "n_target": targets[c], "n_generated": n_gen[c], "seed": seeds[c],
                             "rows": [int(sum(n_gen[k] for k in classes if k < c)),
                                      int(sum(n_gen[k] for k in classes if k <= c))]} for c in classes},
        "n_generated": total, "generator": {"checkpoint": rel_ckpt, "checkpoint_sha256": final.ckpt_sha256,
                                            "candidates": list(final.candidates), **final.record},
        "device": str(device), "batch_size": int(cfg.batch_size), "noise": "CPU torch.Generator streams "
        "(train_generative.noise_rows), row r of class c from (seed_c, r)",
        "wall_s": round(wall, 3), "samples_per_s": total / wall if wall > 0 else None, "peak_vram_mb": peak}
    ctx.save_json(GENERATION_NAME, doc)
    return {"source": source, final.param_name: value, "choice": choice["source"], "n_generated": total,
            "n_target": int(sum(targets.values())), "samples_per_s": doc["samples_per_s"], "peak_vram_mb": peak,
            "checkpoint_sha256": final.ckpt_sha256, "device": str(device)}


# ============================================================================
# stage qc-<src>
# ============================================================================

def read_generation(gen_dir: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], np.ndarray]:
    """(generation.json, gen_meta.npz, X_gen memmap) of a generate stage directory."""
    doc = read_json(gen_dir / GENERATION_NAME)
    if doc is None or doc.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise StageError(f"{gen_dir / GENERATION_NAME} is missing or not schema_version {GENERATION_SCHEMA_VERSION}")
    meta = load_npz(gen_dir / GEN_META_NAME)
    X = np.load(gen_dir / X_GEN_NAME, mmap_mode="r")
    n = int(doc["n_generated"])
    if X.shape[0] != n or X.dtype != np.float16 or any(len(v) != n for k, v in meta.items() if v.ndim == 1):
        raise StageError(f"{gen_dir}: X_gen.npy {X.dtype} {X.shape} / gen_meta.npz disagree with "
                         f"generation.json's {n} rows")
    return doc, meta, X


def run_qc(ctx: "StageContext", cfg: Any, source: str) -> Dict[str, Any]:
    from hsi_augment.qc import RULE_BITS, set_metrics, summarize_bitmask
    from hsi_augment.qc.memorization import resolve_device
    from hsi_augment.qc.reference import THRESHOLDS_NAME
    from training.synthetic_mix import QC_SCORES_NAME

    gen_name = stage_name("generate", source)
    gen_dir = ctx.upstream_dir(gen_name)
    gen_done = upstream_done(ctx, gen_name)
    rec = (gen_done.get("outputs") or {}).get(X_GEN_NAME) or {}
    if rec.get("size_bytes") is not None and (gen_dir / X_GEN_NAME).stat().st_size != rec["size_bytes"]:
        raise StageError(f"{gen_dir / X_GEN_NAME} changed since {gen_name} wrote it; re-run {gen_name}")
    doc, meta, X = read_generation(gen_dir)
    names = list(ctx.adapter.class_names)
    device = resolve_device(cfg.device)
    ref = load_reference(ctx)
    classes = sorted(int(c) for c in doc["targets"])
    check_targets_have_reference(ref, classes, names)

    n = int(doc["n_generated"])
    bitmask = np.zeros(n, dtype=np.uint8)
    score_parts: List[Dict[str, np.ndarray]] = []
    with ctx.timer("score"):
        for c in classes:
            a, b = doc["targets"][str(c)]["rows"]
            if not np.all(meta["class_id"][a:b] == c):
                raise StageError(f"gen_meta.npz rows {a}:{b} are not all class {c}")
            m, s = score_in_chunks(ref, X, c, start=a, stop=b, fold=None, device=device, chunk_rows=cfg.chunk_rows)
            bitmask[a:b] = m
            score_parts.append(s)
            ctx.log(f"{names[c]}: scored {b - a} samples, accepted {int((m == 0).sum())}")
    scores = {k: np.concatenate([p[k] for p in score_parts]) for k in score_parts[0]}

    selected_parts, per_class = [], {}
    failures = []
    for c in classes:
        a, b = doc["targets"][str(c)]["rows"]
        try:
            pos, sel_rec = select_accepted(bitmask[a:b], doc["targets"][str(c)]["n_target"],
                                           min_acceptance=cfg.min_acceptance, overgenerate=doc["overgenerate"],
                                           what=f"{source} {names[c]}")
        except StageError as e:
            rej = summarize_bitmask(bitmask[a:b])["rejected_by"]
            failures.append(f"{e} (rejections per rule: {rej})")
            continue
        selected_parts.append(a + pos)
        per_class[names[c]] = {"class_id": c, **sel_rec, **summarize_bitmask(bitmask[a:b]),
                               "last_selected_gen_index": int(a + pos[-1]) if pos.size else None}
    selected = (np.concatenate(selected_parts) if selected_parts else np.zeros(0)).astype(np.int64)
    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[selected] = True

    # stored for every generated sample, also when the stage is about to fail (for the diagnosis)
    _save_npz(ctx, QC_SCORES_NAME, {"gen_index": meta["gen_index"], "class_id": meta["class_id"],
                                    "bitmask": bitmask, "accepted": bitmask == 0, "selected": sel_mask,
                                    **scores})
    if failures:
        raise AcceptanceError("; ".join(failures))
    ctx.save_npy(SELECTED_NAME, selected)

    set_docs, flags = {}, []
    with ctx.timer("set_metrics"):
        for c in classes:
            rows = selected[meta["class_id"][selected] == c]
            ctx.ram_guard(int(rows.size) * int(np.prod(X.shape[1:])) * 8, f"set metrics of {rows.size} {names[c]}")
            sm = set_metrics(ref, np.asarray(X[rows]), c, scores={"nn_patient": scores["nn_patient"][rows]},
                             n_boot=cfg.bootstrap, seed=ctx.seed_int("set-metrics", c), device=device)
            set_docs[names[c]] = sm
            flags += [f"{names[c]}: {f}" for f in sm["flags"]]
            per_class[names[c]]["set_metric_flags"] = list(sm["flags"])
            per_class[names[c]]["nn_attribution_max_share"] = (sm.get("nn_attribution") or {}).get("max_share")
    for f in flags:
        ctx.warn(f"set metrics: {f}")
    ctx.save_json(SET_METRICS_NAME, set_docs)

    ref_dir = ctx.upstream_dir(QC_REFERENCE_STAGE)
    summary_doc = {
        "schema_version": GENERATION_SCHEMA_VERSION, "source": source, "selection_rule": SELECTION_RULE,
        "min_acceptance": float(cfg.min_acceptance), "rule_bits": dict(RULE_BITS),
        "n_generated": n, "n_selected": int(selected.size), "classes": per_class,
        "thresholds_sha256": sha256_file(ref_dir / THRESHOLDS_NAME),
        "qc_reference_fingerprint": upstream_done(ctx, QC_REFERENCE_STAGE).get("fingerprint"),
        "generate_fingerprint": gen_done.get("fingerprint"), "set_metric_flags": flags}
    ctx.save_json(QC_SUMMARY_NAME, summary_doc)
    n_acc = int((bitmask == 0).sum())
    return {"source": source, "n_generated": n, "n_accepted": n_acc, "acceptance": n_acc / n if n else None,
            "n_selected": int(selected.size), "set_metric_flags": len(flags),
            "acceptance_per_class": {k: v["acceptance"] for k, v in per_class.items()}}
