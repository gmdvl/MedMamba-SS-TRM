# -*- coding: utf-8 -*-
"""
hsi_augment/fit_cvae.py
=======================
Phase 6 (#23): fitting, diagnosing and selecting the conditional VAE. The
stages are registered in `hsi_augment/stages/fit_cvae.py` (configs there);
the model, loss and `sample` are in `hsi_augment/models/cvae.py`; the loop is
`train_generative.Trainer` (#22).

Layout
------
    gen/cvae/lopo/                                  stage `cvae-lopo`
        lopo_summary.json                           per cell: every fold's record + the aggregate
        <cell>/fold-<k>/                            one Trainer run per (cell, fold), cell = "d64-r0.1"
            best.pt  last.pt  history.csv           the Trainer's (best = lowest held-out nll, EMA)
            model.json  calibration.json  diagnostics.json
    gen/cvae/final/                                 stage `cvae-final`
        selection.json                              the cell table, gates, choice, step budget
        best.pt  last.pt  history.csv               best.pt == last.pt: no held-out data, no selection
        model.json  calibration.json  diagnostics.json

`load_cvae(run_dir)` (models/cvae.py) rebuilds any of these runs from its
model.json and the EMA weights of best.pt; `sample(model, codec, class_id, n,
tau_s, seed, start)` generates. `lopo_run_dir` / `final_run_dir` name them.

The pool
--------
Both stages require the `pool` stage and read its directory,
`ctx.upstream_dir("pool")` (`open_pool`), checked against this build's train
split. The pool stage's fingerprint is chained into theirs; the pool's bytes
are guarded by its content sha256 (`Pool.fingerprint()`), recorded in the
LOPO summary, every model.json and every batch source.

One fit (`train_run`)
---------------------
1. `setup_determinism(init seed)`, build the CVAE (GroupNorm only, checked).
2. Calibrate lambda at init on the first `sam_calib_batches` training
   batches (`calibrate_sam_weight`), unless resuming a run whose
   calibration.json exists (then it is re-read, so a resumed loss is the same
   loss bit for bit).
3. `Trainer(..., ctx=ctx, run_dir=...)`. Training batches: `BalancedBatchSampler`
   over the fold's train rows (class-, then patient-balanced, seeded per
   epoch). Held-out batches: every pool row of the fold's held-out patients
   (`splits.fold_masks` on the pool's patient sidecar). The data sources'
   `describe()` carries the pool's sha256, so a resume on another pool is refused.

Seeds. Per fold k: init / train / sampler / calibration / prior streams are
`ctx.seed_int(<name>, k)`, the SAME for every cell: the cells of a fold see
the same batches, init stream and noise (common random numbers), so their
differences are the settings'. The final stage has its own streams.

Diagnostics (LOPO held-out patients only, at best.pt's EMA weights)
-------------------------------------------------------------------
Per class and overall: MSE and PSNR on decoded (linear) values, log-z MSE,
SAM in degrees, unclamped (`spectral_angle_deg`, float64, 2 atan2 form, so
identical spectra give 0 and small angles are exact), and
`training.recon_artifacts.ssim2d`; the reconstruction uses z = E[z | x, y].
PSNR and SSIM use one data range, max(codec.clip_hi), the largest value a
decoded patch can take. Latent: active units (Burda et al. 2016: dim j is
active if Var_x E[z_j | x] > 0.01) and the mean KL per dim. Train-side
reference: the same SAM on `diag_train_rows` seeded fold-train rows, for the
GPU acceptance "held-out SAM <= 1.25 x train SAM". Prior samples: for each
tau_s candidate, `prior_samples` minority-class samples: the fraction of
values at 0 or at clip_hi, and the angle between their mean spectrum and the
held-out minority patient's. That is a sanity check, not QC acceptance (#28);
and a patient probe on the latent would not measure memorization (patient
signatures are expected to be decodable; #28's nearest-neighbour distance does).

Selection (`select_cell`, in the FINAL stage)
---------------------------------------------
Gates (the plan's GPU acceptance, on across-fold means): active fraction >=
`min_active_frac`, mean held-out SAM / mean train SAM <= `max_sam_ratio`,
mean KL per dim > free_bits. Among the passing cells, the lowest mean of
`select_metric` (held-out nll by default); ties go to the smaller d, then
the smaller SAM ratio. Step budget: the median of the chosen cell's fold
best steps (rounded half up). tau_s stays open: model.json records the
candidates and `tau_s: null` for QC to fill in.
"""

from __future__ import annotations

import dataclasses
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from hsi_augment.models.blocks import assert_no_batchnorm
from hsi_augment.models.cvae import (CVAE, MODEL_JSON, CVAELoss, calibrate_sam_weight, kl_per_dim,
                                     model_record, sample)
from hsi_augment.pool_io import Pool, PoolError, check_against_train, load_pool
from hsi_augment.splits import fold_heldout_patients, fold_masks, load_folds
from hsi_augment.stage import StageError, read_done, read_json
from hsi_augment.stages.fit_cvae import (CVAEFinalConfig, CVAEFitConfig, CVAELopoConfig,
                                         CVAETrainConfig)
from hsi_augment.train_generative import (BEST_NAME, LAST_NAME, ArrayBatchSource, BalancedBatchSampler,
                                          SequentialBatches, Trainer, TrainResult, load_weights,
                                          setup_determinism)

if TYPE_CHECKING:  # pragma: no cover
    from hsi_augment.codec import LogSpaceCodec
    from hsi_augment.stage import StageContext

LOPO_SUMMARY_NAME = "lopo_summary.json"
SELECTION_NAME = "selection.json"
DIAGNOSTICS_NAME = "diagnostics.json"
CALIBRATION_NAME = "calibration.json"
CELL_RE = re.compile(r"^d\d+-r[0-9.e+-]+$")


# ============================================================================
# names and small helpers
# ============================================================================

@dataclass(frozen=True)
class Cell:
    """One grid point: latent dimension and SAM ratio."""
    latent_dim: int
    sam_ratio: float

    @property
    def name(self) -> str:
        return f"d{int(self.latent_dim)}-r{float(self.sam_ratio):g}"


def grid(cfg: CVAELopoConfig) -> List[Cell]:
    return [Cell(int(d), float(r)) for d in cfg.latent_dims for r in cfg.sam_ratios]


def fold_dirname(fold: int) -> str:
    return f"fold-{int(fold)}"


def lopo_run_dir(out_root: Any, cell: str, fold: int) -> Path:
    """`<out>/gen/cvae/lopo/<cell>/fold-<k>`."""
    return Path(out_root) / "gen" / "cvae" / "lopo" / cell / fold_dirname(fold)


def final_run_dir(out_root: Any) -> Path:
    """`<out>/gen/cvae/final`."""
    return Path(out_root) / "gen" / "cvae" / "final"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise StageError("--device cuda, but CUDA is not available")
    return torch.device(name)


def fit_config(cfg: CVAETrainConfig, cell: Cell, **changes: Any) -> CVAEFitConfig:
    """The Trainer config of one fit: `cfg`'s shared fields plus the cell."""
    base = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(CVAETrainConfig)}
    base.update(latent_dim=cell.latent_dim, sam_ratio=cell.sam_ratio, **changes)
    return CVAEFitConfig(**base)


def make_step_fn(loss: CVAELoss) -> Any:
    """The Trainer's step_fn for `loss`. A module-level hook, so a test can
    wrap it (e.g. to kill a fit mid-run)."""
    return loss


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(v)) if len(v) == len(values) and v else None


def median_step(steps: Sequence[int]) -> int:
    """Median of the folds' best steps, rounded half up, at least 1."""
    return max(1, int(math.floor(float(np.median(np.asarray(steps, dtype=np.float64))) + 0.5)))


def spectral_angle_deg(a: torch.Tensor, b: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Per-pixel spectral angle in degrees along `dim` (NCHW -> (N, H, W)),
    WITHOUT the 1 - 1e-3 cos clamp of `sam_deg` / `SAMLoss` (#22's 2.56 deg
    floor): float64, angle = 2 atan2(|a^ - b^|, |a^ + b^|) on unit vectors,
    exact for small angles where acos(cos) is not. Two zero spectra give 0,
    one zero spectrum 90 deg."""
    a64, b64 = a.double(), b.double()
    ua = a64 / a64.norm(dim=dim, keepdim=True).clamp_min(1e-300)
    ub = b64 / b64.norm(dim=dim, keepdim=True).clamp_min(1e-300)
    return torch.rad2deg(2.0 * torch.atan2((ua - ub).norm(dim=dim), (ua + ub).norm(dim=dim)))


class PoolBatchSource(ArrayBatchSource):
    """`ArrayBatchSource` over the pool whose `describe()` also carries the
    pool's sha256: a resume on a different pool is refused by the Trainer."""

    def __init__(self, X: np.ndarray, plan: Any, *, y: np.ndarray, pool_sha256: str):
        super().__init__(X, plan, extras={"y": y}, layout="NHWC")
        self.pool_sha256 = str(pool_sha256)

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "pool_sha256": self.pool_sha256}


# ============================================================================
# the pool, read through the stage's SplitGuard
# ============================================================================

def open_pool(ctx: "StageContext") -> Pool:
    """Load and validate the pool stage's output (a declared upstream, so the
    runner has checked that it finished), and check it against THIS build's
    train sidecars (opened with purpose "generator": the access log's only
    record for the CVAE stages, and train only)."""
    d = ctx.upstream_dir("pool")
    try:
        pool = load_pool(d)
    except FileNotFoundError as e:
        raise StageError(f"{e}: the pool stage finished, but its files are gone; re-run the pool stage") from e
    except PoolError as e:
        raise StageError(str(e)) from e
    train = ctx.open_split("train", purpose="generator", kinds=("y", "groups", "captures"))
    problems = check_against_train(pool, train.y, train.groups, train.captures)
    n_classes = ctx.adapter.num_classes
    if int(pool.y.max()) >= n_classes:
        problems.append(f"y_pool has class {int(pool.y.max())}; the build has {n_classes} classes")
    if pool.n_bands != ctx.adapter.n_bands or pool.patch != ctx.adapter.patch_size:
        problems.append(f"pool patches are {pool.patch}x{pool.patch}x{pool.n_bands}, the build's "
                        f"{ctx.adapter.patch_size}x{ctx.adapter.patch_size}x{ctx.adapter.n_bands}")
    if problems:
        raise StageError(f"pool {d} does not belong to the train split of {ctx.data_dir}: {problems}")
    ctx.log(f"pool {d}: {pool.n} rows {pool.patch}x{pool.patch}x{pool.n_bands}, "
            f"rows per class {pool.describe()['rows_per_class']}")
    return pool


def pool_array(ctx: "StageContext", pool: Pool, in_ram: bool) -> np.ndarray:
    """X_pool in RAM (Plan §Phase 4: "held in RAM during fitting"), behind
    the RAM guard; or the memmap itself."""
    if not in_ram:
        return pool.X
    ctx.ram_guard(int(pool.X.nbytes), "X_pool in RAM")
    with ctx.timer("load pool"):
        return np.array(pool.X)


# ============================================================================
# one fit
# ============================================================================

@dataclass
class RunOutcome:
    trainer: Trainer
    result: TrainResult
    calibration: Dict[str, Any]
    model_spec: Dict[str, Any]

    def best_eval(self) -> Dict[str, float]:
        """The held-out metrics of the best evaluation ({} without held-out data)."""
        step = self.result.best_step
        for r in self.trainer.history:
            if r.get("step") == step and "eval_nll" in r:
                return {k[5:]: v for k, v in r.items() if k.startswith("eval_") and k != "eval_best"}
        return {}


def _rel(ctx: "StageContext", path: Path) -> str:
    return Path(path).resolve().relative_to(Path(ctx.stage_dir).resolve()).as_posix()


def train_run(ctx: "StageContext", *, pool: Pool, X: np.ndarray, cfg: CVAEFitConfig, train_rows: np.ndarray,
              eval_rows: Optional[np.ndarray], run_dir: Path, seeds: Mapping[str, int], device: torch.device,
              pool_sha256: str, record: Mapping[str, Any]) -> RunOutcome:
    """Build, calibrate and train one CVAE in `run_dir` (resuming it under
    `ctx.resume`); see the module docstring. Writes model.json and
    calibration.json; the Trainer writes best.pt / last.pt / history.csv."""
    n_classes = ctx.adapter.num_classes
    try:
        sampler = BalancedBatchSampler(pool.y, pool.patient, cfg.batch_size, seed=seeds["sampler"],
                                       rows=train_rows, classes=list(range(n_classes)))
    except ValueError as e:
        raise StageError(f"{_rel(ctx, run_dir)}: cannot build balanced batches: {e}") from e
    train_src = PoolBatchSource(X, sampler, y=pool.y, pool_sha256=pool_sha256)
    eval_src = None
    if eval_rows is not None:
        eval_src = PoolBatchSource(X, SequentialBatches(eval_rows, cfg.eval_batch_size), y=pool.y,
                                   pool_sha256=pool_sha256)

    setup_determinism(seeds["init"], deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
    model = CVAE(in_bands=pool.n_bands, patch=pool.patch, n_classes=n_classes, latent_dim=cfg.latent_dim,
                 widths=cfg.widths, cond_dim=cfg.cond_dim, dropout=cfg.dropout,
                 min_log_sigma=cfg.min_log_sigma).to(device)
    assert_no_batchnorm(model)

    calib_path = run_dir / CALIBRATION_NAME
    calib = read_json(calib_path) if (ctx.resume and (run_dir / LAST_NAME).is_file()) else None
    if calib is None or float(calib.get("ratio", -1.0)) != float(cfg.sam_ratio):
        probe = CVAELoss(pool.codec, sam_weight=0.0, free_bits=cfg.free_bits,
                         beta_warmup_frac=cfg.beta_warmup_frac, sam_angle_eps=cfg.sam_angle_eps)
        n_cal = min(cfg.sam_calib_batches, train_src.num_batches(0))
        calib = calibrate_sam_weight(model, probe, (train_src.get_batch(0, i) for i in range(n_cal)),
                                     cfg.sam_ratio, seed=seeds["calibration"])
    ctx.save_json(_rel(ctx, calib_path), calib)
    loss = CVAELoss(pool.codec, sam_weight=float(calib["lambda"]), free_bits=cfg.free_bits,
                    beta_warmup_frac=cfg.beta_warmup_frac, sam_angle_eps=cfg.sam_angle_eps)
    ctx.save_json(_rel(ctx, run_dir / MODEL_JSON),
                  model_record(model, pool.codec, checkpoint=BEST_NAME, weights="ema",
                               sam_weight=float(calib["lambda"]), class_names=list(ctx.adapter.class_names),
                               minority_classes=list(ctx.adapter.minority_classes), seeds=dict(seeds),
                               fit_config=cfg.to_dict(), pool_sha256=pool_sha256, **record))
    trainer = Trainer(model, make_step_fn(loss), train_src, cfg, eval_data=eval_src, run_dir=run_dir,
                      seed=seeds["train"], device=device, ctx=ctx)
    with ctx.timer("train"):
        result = trainer.fit()
    return RunOutcome(trainer=trainer, result=result, calibration=calib, model_spec=model.spec())


def load_best(outcome: RunOutcome, run_dir: Path, device: torch.device) -> CVAE:
    """A fresh CVAE with best.pt's EMA weights (what is exported), eval mode."""
    model = CVAE(**outcome.model_spec)
    load_weights(run_dir / BEST_NAME, model, ema=True)
    return model.to(device).eval()


# ============================================================================
# diagnostics
# ============================================================================

@torch.no_grad()
def reconstruction_metrics(model: CVAE, codec: "LogSpaceCodec", X: np.ndarray, y: np.ndarray,
                           rows: np.ndarray, *, n_classes: int, batch_size: int, device: torch.device,
                           data_range: float, active_threshold: float = 0.01) -> Dict[str, Any]:
    """Reconstruction (z = E[z|x,y]) and latent statistics over `rows` (sorted
    pool rows); see the module docstring. Per class and "all" (row-weighted):
    rows, mse (decoded), mse_logz, psnr, sam_deg, ssim2d, mean_spectrum
    (decoded, per band). Latent: active units, KL per dim."""
    from training.recon_artifacts import ssim2d
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    d = model.latent_dim
    keys = ("mse", "mse_logz", "sam_deg", "ssim2d")
    acc = {k: np.zeros(n_classes) for k in keys}
    count = np.zeros(n_classes, dtype=np.int64)
    spec = np.zeros((n_classes, model.in_bands))
    s1, s2, kl = (torch.zeros(d, dtype=torch.float64, device=device) for _ in range(3))
    for a in range(0, rows.size, batch_size):
        idx = rows[a:a + batch_size]
        x = torch.from_numpy(np.ascontiguousarray(X[idx])).permute(0, 3, 1, 2).float().to(device)
        yb = torch.from_numpy(np.asarray(y[idx], dtype=np.int64)).to(device)
        mu, logvar = model.encode(x, yb)
        x_hat = model.decode(mu, yb)
        t, p = codec.decode_torch(x), codec.decode_torch(x_hat)
        per_row = {"mse": ((p - t) ** 2).flatten(1).mean(1).double(),
                   "mse_logz": ((x_hat - x) ** 2).flatten(1).mean(1).double(),
                   "sam_deg": spectral_angle_deg(t, p).flatten(1).mean(1),
                   "ssim2d": ssim2d(t, p, data_range=data_range, reduce=False).double()}
        band_sum = t.double().sum(dim=(2, 3))                                  # (B, C)
        yn = yb.cpu().numpy()
        for c in np.unique(yn):
            m = torch.from_numpy(yn == c).to(device)
            count[c] += int(m.sum())
            for k in keys:
                acc[k][c] += float(per_row[k][m].sum())
            spec[c] += band_sum[m].sum(0).cpu().numpy()
        mu64 = mu.double()
        s1 += mu64.sum(0)
        s2 += (mu64 ** 2).sum(0)
        kl += kl_per_dim(mu, logvar).double().sum(0)

    def summarise(n: int, sums: Mapping[str, float], band: np.ndarray) -> Dict[str, Any]:
        if n == 0:
            return {"rows": 0}
        out = {"rows": int(n), **{k: float(sums[k] / n) for k in keys}}
        out["psnr"] = float(10.0 * math.log10(data_range ** 2 / out["mse"])) if out["mse"] > 0 else None
        out["mean_spectrum"] = (band / (n * model.patch ** 2)).tolist()
        return out

    per_class = {int(c): summarise(int(count[c]), {k: acc[k][c] for k in keys}, spec[c])
                 for c in range(n_classes)}
    n_all = int(count.sum())
    overall = summarise(n_all, {k: acc[k].sum() for k in keys}, spec.sum(0))
    out: Dict[str, Any] = {"rows": n_all, "data_range": float(data_range), "per_class": per_class,
                           "all": overall}
    if n_all:
        var = (s2 / n_all - (s1 / n_all) ** 2).clamp_min(0).cpu().numpy()
        kl_dim = (kl / n_all).cpu().numpy()
        active = var > active_threshold
        out["latent"] = {"active_units": int(active.sum()), "active_frac": float(active.mean()),
                         "active_threshold": float(active_threshold), "var_mean_z": var.tolist(),
                         "kl_per_dim": kl_dim.tolist(), "kl_per_dim_mean": float(kl_dim.mean()),
                         "kl_total": float(kl_dim.sum())}
    return out


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    return float(spectral_angle_deg(torch.from_numpy(np.asarray(u, np.float64))[None],
                                    torch.from_numpy(np.asarray(v, np.float64))[None], dim=1)[0])


def prior_sample_stats(model: CVAE, codec: "LogSpaceCodec", class_id: int, n: int, taus: Sequence[float],
                       seed: int, ref_spectrum: Optional[Sequence[float]]) -> Dict[str, Any]:
    """Per tau_s: n samples of `class_id` through `sample()`, and the fraction
    of values at 0 / at clip_hi, plus their mean spectrum's angle and mean
    relative error against `ref_spectrum` (a real mean spectrum)."""
    out: Dict[str, Any] = {}
    if n <= 0:
        return out
    hi = codec.clip_hi.astype(np.float16)
    for tau in taus:
        v = sample(model, codec, class_id, n, float(tau), seed)
        rec: Dict[str, Any] = {"n": int(n), "zero_frac": float(np.mean(v == 0)),
                               "clip_hi_frac": float(np.mean(v >= hi))}
        mean = v.reshape(-1, v.shape[-1]).astype(np.float64).mean(0)
        rec["mean_spectrum"] = mean.tolist()
        if ref_spectrum is not None:
            ref = np.asarray(ref_spectrum, dtype=np.float64)
            rec["mean_spectrum_sam_deg"] = _angle_deg(mean, ref)
            rec["mean_spectrum_rel_err"] = float(np.mean(np.abs(mean - ref) / np.maximum(ref, 1e-12)))
        out[f"{float(tau):g}"] = rec
    return out


def _sample_rows(rng: np.random.Generator, mask: np.ndarray, n: int) -> np.ndarray:
    rows = np.flatnonzero(mask)
    if n <= 0 or rows.size == 0:
        return rows[:0]
    if rows.size > n:
        rows = np.sort(rng.choice(rows, size=n, replace=False))
    return rows


# ============================================================================
# stage cvae-lopo
# ============================================================================

def _clear_stale(stage_dir: Path) -> None:
    """A fresh (non-resumed) LOPO run starts from an empty tree: cell
    directories of another grid and an old summary would be mistaken for
    this run's."""
    for p in Path(stage_dir).iterdir():
        if p.is_dir() and CELL_RE.match(p.name):
            shutil.rmtree(p)
    (Path(stage_dir) / LOPO_SUMMARY_NAME).unlink(missing_ok=True)


def _aggregate(folds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    steps = [int(f["best_step"]) for f in folds]
    sam = _mean([f["heldout"]["sam_deg"] for f in folds])
    sam_train = _mean([f["train_ref"]["sam_deg"] for f in folds])
    return {"folds": len(folds), "nll": _mean([f["best_eval"].get("nll") for f in folds]),
            "sam_deg": sam, "sam_train_deg": sam_train,
            "sam_ratio": (sam / sam_train) if (sam is not None and sam_train) else None,
            "mse": _mean([f["heldout"]["mse"] for f in folds]),
            "psnr": _mean([f["heldout"]["psnr"] for f in folds]),
            "ssim2d": _mean([f["heldout"]["ssim2d"] for f in folds]),
            "active_frac": _mean([f["latent"]["active_frac"] for f in folds]),
            "kl_per_dim": _mean([f["latent"]["kl_per_dim_mean"] for f in folds]),
            "best_steps": steps, "median_best_step": median_step(steps)}


def run_lopo(ctx: "StageContext", cfg: CVAELopoConfig) -> Dict[str, Any]:
    device = resolve_device(cfg.device)
    pool = open_pool(ctx)
    with ctx.timer("pool sha256"):
        pool_sha = pool.fingerprint()
    folds_doc = load_folds(ctx.upstream_dir("leakage"))
    folds = list(folds_doc["folds"])
    n_classes = ctx.adapter.num_classes
    class_names = list(ctx.adapter.class_names)
    minority = list(ctx.adapter.minority_classes)
    masks = []
    for f in folds:
        train_mask, held_mask = fold_masks(pool.patient, f)
        if not held_mask.any():
            raise StageError(f"fold {f['fold']}: none of its held-out patients "
                             f"{list(fold_heldout_patients(f))} has a pool row")
        masks.append((train_mask, held_mask))
    X = pool_array(ctx, pool, cfg.pool_in_ram)
    data_range = float(pool.codec.clip_hi.max())
    if not ctx.resume:
        _clear_stale(ctx.stage_dir)
    cells = grid(cfg)
    ctx.log(f"{len(cells)} cell(s) {[c.name for c in cells]} x {len(folds)} fold(s) on {device}")

    doc_cells: Dict[str, Any] = {}
    for cell in cells:
        records = []
        for f, (train_mask, held_mask) in zip(folds, masks):
            k = int(f["fold"])
            run_dir = ctx.stage_dir / cell.name / fold_dirname(k)
            seeds = {"init": ctx.seed_int("init", k), "train": ctx.seed_int("train", k),
                     "sampler": ctx.seed_int("sampler", k), "calibration": ctx.seed_int("sam-calibration", k),
                     "prior": ctx.seed_int("prior", k)}
            ctx.log(f"{cell.name} fold {k}: held out {list(fold_heldout_patients(f))}, "
                    f"{int(train_mask.sum())} train / {int(held_mask.sum())} held-out pool rows")
            outcome = train_run(ctx, pool=pool, X=X, cfg=fit_config(cfg, cell), train_rows=train_mask,
                                eval_rows=held_mask, run_dir=run_dir, seeds=seeds, device=device,
                                pool_sha256=pool_sha,
                                record={"stage": "cvae-lopo", "cell": cell.name, "fold": k,
                                        "heldout_patients": list(fold_heldout_patients(f)),
                                        "tau_s_candidates": list(cfg.tau_s), "tau_s": None})
            model = load_best(outcome, run_dir, device)
            with ctx.timer("diagnostics"):
                held = reconstruction_metrics(model, pool.codec, X, pool.y, np.flatnonzero(held_mask),
                                              n_classes=n_classes, batch_size=cfg.eval_batch_size,
                                              device=device, data_range=data_range,
                                              active_threshold=cfg.active_unit_threshold)
                ref_rows = _sample_rows(ctx.rng("diag-train", k), train_mask, cfg.diag_train_rows)
                train_ref = reconstruction_metrics(model, pool.codec, X, pool.y, ref_rows, n_classes=n_classes,
                                                   batch_size=cfg.eval_batch_size, device=device,
                                                   data_range=data_range,
                                                   active_threshold=cfg.active_unit_threshold)
                ref_spec = held["per_class"].get(minority[0], {}).get("mean_spectrum") if minority else None
                prior = (prior_sample_stats(model, pool.codec, minority[0], cfg.prior_samples, cfg.tau_s,
                                            seeds["prior"], ref_spec) if minority else {})
            res = outcome.result
            diag = {"cell": cell.name, "fold": k, "dcis_patient": int(f["dcis_patient"]),
                    "heldout_patients": list(fold_heldout_patients(f)), "best_step": res.best_step,
                    "stopped_early": res.stopped_early, "steps": res.step, "best_eval": outcome.best_eval(),
                    "sam_weight": float(outcome.calibration["lambda"]), "class_names": class_names,
                    "heldout": held, "train_ref": {k2: v for k2, v in train_ref.items() if k2 != "latent"},
                    "prior_samples": {"class": minority[0] if minority else None,
                                      "reference": "held-out minority mean spectrum", "per_tau": prior,
                                      "note": "sanity check only; QC acceptance and the tau_s choice are #28's"}}
            ctx.save_json(_rel(ctx, run_dir / DIAGNOSTICS_NAME), diag)
            h_all, t_all = held["all"], train_ref["all"]
            rec = {"fold": k, "run_dir": _rel(ctx, run_dir), "dcis_patient": int(f["dcis_patient"]),
                   "heldout_patients": list(fold_heldout_patients(f)), "best_step": int(res.best_step),
                   "stopped_early": bool(res.stopped_early), "steps": int(res.step),
                   "best_eval": outcome.best_eval(), "sam_weight": float(outcome.calibration["lambda"]),
                   "heldout": {m: h_all.get(m) for m in ("mse", "mse_logz", "psnr", "sam_deg", "ssim2d")},
                   "heldout_per_class": {class_names[c]: {m: v.get(m) for m in ("rows", "mse", "psnr",
                                                                                   "sam_deg", "ssim2d")}
                                         for c, v in held["per_class"].items()},
                   "train_ref": {"rows": train_ref["rows"], "sam_deg": t_all.get("sam_deg")},
                   "latent": {m: held["latent"][m] for m in ("active_units", "active_frac", "kl_per_dim_mean")}}
            ctx.log_metrics({"event": "fold_diagnostics", "cell": cell.name, "fold": k,
                             "best_step": rec["best_step"], "heldout_nll": rec["best_eval"].get("nll"),
                             "heldout_sam_deg": rec["heldout"]["sam_deg"],
                             "train_sam_deg": rec["train_ref"]["sam_deg"],
                             "active_frac": rec["latent"]["active_frac"]})
            ctx.log(f"{cell.name} fold {k}: best step {rec['best_step']}, held-out nll "
                    f"{rec['best_eval'].get('nll')}, SAM {rec['heldout']['sam_deg']} deg (train "
                    f"{rec['train_ref']['sam_deg']}), active units {rec['latent']['active_units']}/{cell.latent_dim}")
            records.append(rec)
            del model
        doc_cells[cell.name] = {"latent_dim": cell.latent_dim, "sam_ratio": cell.sam_ratio, "folds": records,
                                "aggregate": _aggregate(records)}

    doc = {"schema_version": 1, "stage": "cvae-lopo", "pool": {"dir": str(pool.pool_dir), "sha256": pool_sha,
                                                               **pool.describe()},
           "folds_from": "prep/leakage/folds.json", "n_folds": len(folds), "class_names": class_names,
           "minority_classes": minority, "free_bits": cfg.free_bits, "tau_s_candidates": list(cfg.tau_s),
           "eval_metric": cfg.eval_metric, "cells": doc_cells}
    ctx.save_json(LOPO_SUMMARY_NAME, doc)
    return {"cells": {n: {k: c["aggregate"][k] for k in ("nll", "sam_deg", "sam_ratio", "active_frac",
                                                          "median_best_step")}
                      for n, c in doc_cells.items()},
            "folds": len(folds), "pool_sha256": pool_sha, "device": str(device)}


# ============================================================================
# stage cvae-final
# ============================================================================

def select_cell(summary: Mapping[str, Any], cfg: CVAEFinalConfig, *, free_bits: float) -> Dict[str, Any]:
    """The selection record (module docstring): every cell's aggregate, its
    gates, the choice and the step budget. StageError if no cell passes and
    `enforce_gates`."""
    table = []
    for name, c in summary["cells"].items():
        agg = c["aggregate"]
        gates = {"active_frac": agg["active_frac"] is not None and agg["active_frac"] >= cfg.min_active_frac,
                 "sam_ratio": agg["sam_ratio"] is not None and agg["sam_ratio"] <= cfg.max_sam_ratio,
                 "kl_per_dim": agg["kl_per_dim"] is not None and agg["kl_per_dim"] > free_bits}
        table.append({"cell": name, "latent_dim": int(c["latent_dim"]), "sam_ratio": float(c["sam_ratio"]),
                      "value": agg[cfg.select_metric], "gates": gates, "passes": all(gates.values()),
                      "aggregate": agg})
    if not table:
        raise StageError("lopo_summary.json has no cells")
    candidates = [r for r in table if r["passes"] and r["value"] is not None]
    warning = None
    if not candidates:
        msg = "no LOPO cell passes the gates: " + "; ".join(
            f"{r['cell']} {[g for g, ok in r['gates'].items() if not ok]}" for r in table)
        if cfg.enforce_gates:
            raise StageError(msg + " (--enforce_gates off takes the best cell anyway)")
        warning = msg
        candidates = [r for r in table if r["value"] is not None]
        if not candidates:
            raise StageError(f"no LOPO cell has a finite {cfg.select_metric}")
    chosen = min(candidates, key=lambda r: (r["value"], r["latent_dim"], r["sam_ratio"]))
    return {"rule": {"select_metric": cfg.select_metric, "min_active_frac": cfg.min_active_frac,
                     "max_sam_ratio": cfg.max_sam_ratio, "kl_per_dim_floor": float(free_bits),
                     "enforce_gates": cfg.enforce_gates,
                     "order": f"lowest across-fold mean held-out {cfg.select_metric} among passing cells; "
                              f"ties: smaller latent_dim, then smaller sam_ratio"},
            "table": table, "chosen": chosen["cell"], "latent_dim": chosen["latent_dim"],
            "sam_ratio": chosen["sam_ratio"], "fold_best_steps": chosen["aggregate"]["best_steps"],
            "budget_steps": int(chosen["aggregate"]["median_best_step"]),
            "lopo_fold_runs": [f["run_dir"] for f in summary["cells"][chosen["cell"]]["folds"]],
            "warning": warning}


def run_final(ctx: "StageContext", cfg: CVAEFinalConfig) -> Dict[str, Any]:
    lopo_dir = ctx.upstream_dir("cvae-lopo")
    done = read_done(lopo_dir) or {}
    summary = read_json(lopo_dir / LOPO_SUMMARY_NAME)
    if summary is None or "cells" not in summary:
        raise StageError(f"{lopo_dir / LOPO_SUMMARY_NAME} is missing or unreadable: re-run cvae-lopo")
    lopo_cfg = CVAELopoConfig.from_dict(done.get("config") or {})
    selection = select_cell(summary, cfg, free_bits=lopo_cfg.free_bits)
    if selection["warning"]:
        ctx.warn(selection["warning"])
    device = resolve_device(cfg.device)
    pool = open_pool(ctx)
    with ctx.timer("pool sha256"):
        pool_sha = pool.fingerprint()
    if pool_sha != summary["pool"]["sha256"]:
        raise StageError(f"the pool at {pool.pool_dir} changed since cvae-lopo ran on it (sha256 "
                         f"{summary['pool']['sha256'][:12]} -> {pool_sha[:12]}); re-run cvae-lopo")
    X = pool_array(ctx, pool, lopo_cfg.pool_in_ram)
    cell = Cell(selection["latent_dim"], selection["sam_ratio"])
    budget = selection["budget_steps"]
    run_cfg = fit_config(lopo_cfg, cell, max_steps=budget, early_stop_patience=None,
                         warmup_steps=min(lopo_cfg.warmup_steps, budget - 1))
    ctx.save_json(SELECTION_NAME, selection)
    value = next(r["value"] for r in selection["table"] if r["cell"] == selection["chosen"])
    ctx.log(f"chosen {cell.name} (mean held-out {cfg.select_metric} {value:.5g}); budget {budget} steps = "
            f"median of {selection['fold_best_steps']}; on {device}")
    all_rows = np.ones(pool.n, dtype=bool)
    seeds = {"init": ctx.seed_int("init"), "train": ctx.seed_int("train"), "sampler": ctx.seed_int("sampler"),
             "calibration": ctx.seed_int("sam-calibration"), "prior": ctx.seed_int("prior")}
    minority = list(ctx.adapter.minority_classes)
    outcome = train_run(ctx, pool=pool, X=X, cfg=run_cfg, train_rows=all_rows, eval_rows=None,
                        run_dir=Path(ctx.stage_dir), seeds=seeds, device=device, pool_sha256=pool_sha,
                        record={"stage": "cvae-final", "cell": cell.name, "budget_steps": budget,
                                "tau_s_candidates": list(lopo_cfg.tau_s), "tau_s": None,
                                "tau_s_note": "chosen on LOPO QC (#28) by the host run (#25)"})
    model = load_best(outcome, Path(ctx.stage_dir), device)
    with ctx.timer("diagnostics"):
        ref_rows = _sample_rows(ctx.rng("diag-train"), all_rows, lopo_cfg.diag_train_rows)
        fit = reconstruction_metrics(model, pool.codec, X, pool.y, ref_rows, n_classes=ctx.adapter.num_classes,
                                     batch_size=lopo_cfg.eval_batch_size, device=device,
                                     data_range=float(pool.codec.clip_hi.max()),
                                     active_threshold=lopo_cfg.active_unit_threshold)
        ref_spec = fit["per_class"].get(minority[0], {}).get("mean_spectrum") if minority else None
        prior = (prior_sample_stats(model, pool.codec, minority[0], cfg.prior_samples, lopo_cfg.tau_s,
                                    seeds["prior"], ref_spec) if minority else {})
    ctx.save_json(DIAGNOSTICS_NAME, {
        "cell": cell.name, "steps": outcome.result.step, "budget_steps": budget,
        "heldout": None, "note": "no held-out data: the refit uses every train patient; held-out "
                                 "diagnostics are the LOPO folds' (gen/cvae/lopo)",
        "train_fit": fit, "prior_samples": {"class": minority[0] if minority else None,
                                            "reference": "train minority mean spectrum", "per_tau": prior}})
    return {"chosen": cell.name, "budget_steps": budget, "steps": outcome.result.step,
            "sam_weight": float(outcome.calibration["lambda"]), "select_metric": cfg.select_metric,
            "selection_warning": selection["warning"], "pool_sha256": pool_sha, "device": str(device)}
