# -*- coding: utf-8 -*-
"""
hsi_augment/latent_diffusion.py
===============================
Phase 7 (#24), arm A3: the stages `ae-lopo`, `ae-final` (group `fit-ae`) and
`ldm-lopo`, `ldm-final` (group `fit-ldm`). The models are in
`models/autoencoder.py` and `models/diffusion.py`; the loop is
`train_generative.Trainer`; this module wires them to the pool, the LOPO
folds and the stage directories, and writes the exports and diagnostics.

Output layout (under --out)
---------------------------
    gen/ae/lopo/fold-<k>/   best.pt last.pt history.csv   Trainer checkpoints (weights_only)
                            ae.pt                         EMA weights of best.pt + latent_scale
                                                          (SpatialAutoencoder.from_export)
                            recon_metrics.json            per class, train and held-out patients:
                                                          MSE (log-z), RMSE / PSNR / SAM deg
                                                          (unclamped) / ssim2d on decoded values
                            fold_done.json                the fold's outputs + summary (resume marker)
    gen/ae/final/           best.pt last.pt history.csv ae.pt recon_metrics.json (train only)
    gen/ldm/lopo/fold-<k>/  latents.npy latents.json      the whole pool encoded once by fold k's
                                                          ae.pt (posterior mean x latent_scale),
                                                          float32 NCHW, keyed by the ae.pt sha256
                            best.pt last.pt history.csv   denoiser checkpoints
                            generator.pt                  ae + denoiser (EMA of best.pt) + codec:
                                                          `load_generator(...)`, then `sample(...)`
                            latent_mmd.json               per class and guidance scale w
                            fold_done.json
    gen/ldm/final/          latents.npy latents.json best.pt last.pt history.csv generator.pt

Each stage's done.json summary lists the folds (held-out patients, best step,
latent scale, the diagnostics' headline numbers, the sampling sha).

The pool
--------
The `pool` stage's directory (`<out>/pool/`, found with
`ctx.upstream_dir("pool")`), in the contract of `hsi_augment/pool_io.py`:
`X_pool.npy` float16 NHWC (N, 11, 11, 32), ALREADY in the codec's log-z
space, the y / patient / row / capture sidecars and `codec.json`.
`pool_io.load_pool` opens and validates it (X memory-mapped); `_open_pool`
adds the checks that need this build:
* `pool_io.check_against_train`: every pool row's class, patient and capture
  must equal the train split's at `row_pool`, so a pool of another build is
  refused. The sidecars are opened through the stage's SplitGuard with
  purpose `generator`: the access log records that these stages read train,
  and only train (generators never read val or test);
* the classes, patch size and band count must be the build's.

Every stage here declares `requires` on the pool stage, so without a
finished pool it stops with StageNotReady, and the pool stage's fingerprint
is chained into its own: re-running the pool with another config makes
ae-lopo, ldm-lopo and the final refits stale. The fingerprint does not see
the pool's bytes (a pool rewritten under the same config), so
`Pool.fingerprint()` (sha256 over the six contract files, X included) guards
them: it is recorded in every summary, fold marker, latent cache, batch
source and export; ae-final and the LDM stages refuse a pool that is not the
one their upstream was trained on, and a fold is only kept or resumed on an
unchanged pool.

The stages are not in `all` (`in_all=False`): their defaults are 30k- and
150k-step fits (bf16 for ldm-lopo) for the host GPU (#26), run as their own
commands (`fit-ae`, `fit-ldm`); `all` is the CPU prep chain.

LOPO folds and resume
---------------------
`ae-lopo` reads `prep/leakage/folds.json` (it requires the `leakage` stage)
and fits the configured folds (default 0 and 1; the plan selects on two).
Fold k's train side is every pool row of the other patients, its held-out
side every pool row of the fold's held-out patients (`splits.fold_masks`:
by patient, all tissues). Only train-side rows are ever in a training batch
and in the latent scale; the held-out rows pick best.pt (the held-out loss,
on the EMA weights) and feed the diagnostics. `ldm-lopo` uses the same fold
records, from ae-lopo's done.json, and fold k's autoencoder.

One stage, one directory, so each fold trains in its own `fold-<k>/`, and a
finished fold writes `fold_done.json` (the stage fingerprint, the pool
sha256, its outputs with sha256 and the fold summary). Under `--resume auto`
a fold whose marker matches is kept as it is, the interrupted fold continues
from its `last.pt` (bitwise, `Trainer`), and later folds start fresh. The
batch sources' `describe()` carries the pool sha256 (and the autoencoder's,
for latents), so a fold is never continued on other data: that is a
StageError naming the fold. Without resume every fold is refitted.

Final refits
------------
`ae-final` / `ldm-final` refit on every pool row with the LOPO stage's
settings (read from its done.json, so they cannot drift apart) and a step
budget of the median of the folds' best steps (`--steps median_best`, the
Phase 6 rule) or the LOPO max_steps. They have no held-out data, so best.pt
is the final model and there is no selection.

Seeds
-----
Per stage and run, from the stage seed: ("fold", k, "init") for the weights,
("fold", k, "train") for the Trainer, ("fold", k, "batches") for the batch
order; ("final", ...) for the final refits; the diagnostics' row draws and
sampling noise likewise. Folds never share a stream, and a fold's result
does not depend on which other folds ran.

GPU commands (#26)
------------------
    python augment_pipeline.py fit-ae  --data_dir data/hsi_v9-trainsel/hsi --out data/hsi_v9-trainsel-aug
    python augment_pipeline.py fit-ldm --data_dir data/hsi_v9-trainsel/hsi --out data/hsi_v9-trainsel-aug
Add `--resume auto` to continue an interrupted stage. ldm-lopo trains in bf16
autocast by default; the final stages take that (and every other setting)
from the LOPO stage.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple,
                    Union)

import numpy as np

from hsi_augment.codec import LogSpaceCodec
from hsi_augment.config import to_jsonable
from hsi_augment.io import save_json, sha256_file
from hsi_augment.pool_io import Pool, PoolError, check_against_train, load_pool
from hsi_augment.stage import StageError, read_done, read_json
from hsi_augment.splits import fold_masks, load_folds

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from hsi_augment.models.autoencoder import SpatialAutoencoder
    from hsi_augment.models.diffusion import LatentUNet
    from hsi_augment.stage import StageContext

PathLike = Union[str, Path]

FOLD_DONE_NAME = "fold_done.json"
AE_EXPORT_NAME = "ae.pt"
GENERATOR_NAME = "generator.pt"
GENERATOR_FORMAT = "hsi_augment.latent_diffusion/1"
LATENTS_NAME = "latents.npy"
LATENTS_META_NAME = "latents.json"
RECON_NAME = "recon_metrics.json"
MMD_NAME = "latent_mmd.json"
MMD_PASS_RATIO = 1.5            # Plan §Phase 7 acceptance: MMD(gen, held-out) <= 1.5 x MMD(train, held-out)
SAMPLE_CHECK_ROWS = 64
MEDIAN_SAMPLE_ROWS = 1000       # rows behind the RBF median heuristic


# ============================================================================
# the pool (contract: hsi_augment/pool_io.py)
# ============================================================================

def _open_pool(ctx: "StageContext", *, expect_sha256: Optional[str] = None,
               upstream: str = "") -> Tuple[Pool, Dict[str, Any]]:
    """Load the pool stage's output (a declared upstream, so the runner has
    checked that it finished), check it is THIS build's train rows (module
    docstring) and fingerprint it. `expect_sha256`: the fingerprint `upstream`
    recorded; any other pool is refused. Returns (pool, its record for the
    summaries)."""
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
    with ctx.timer("pool sha256"):
        sha = pool.fingerprint()
    if expect_sha256 is not None and sha != expect_sha256:
        raise StageError(f"the pool at {d} is not the one {upstream} was trained on (sha256 {sha[:12]} != "
                         f"{expect_sha256[:12]}); re-run {upstream} with --force")
    record = {"pool_sha256": sha, "train_rows": int(train.n), **pool.describe()}
    ctx.log(f"pool {d}: {pool.n} rows {pool.patch}x{pool.patch}x{pool.n_bands}, rows per class "
            f"{record['rows_per_class']}, sha256 {sha[:12]}")
    return pool, record


# ============================================================================
# small helpers
# ============================================================================

def _device(name: str) -> "torch.device":
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise StageError(f"device {name!r} requested, but CUDA is not available")
    return dev


def _class_counts(y: np.ndarray, mask: np.ndarray, k: int) -> List[int]:
    return np.bincount(y[mask], minlength=k)[:k].astype(int).tolist()


def _fold_split(pool: Pool, fold: Mapping[str, Any], k: int, class_names: Sequence[str]
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fold k's (train, held-out) pool masks; both sides must hold every class."""
    train, held = fold_masks(pool.patient, fold)
    n_cls = len(class_names)
    tr, he = _class_counts(pool.y, train, n_cls), _class_counts(pool.y, held, n_cls)
    if min(tr) == 0 or min(he) == 0:
        raise StageError(f"fold {k}: the pool has rows per class {tr} on the train side and {he} held out "
                         f"(patients {list(fold['heldout_patients'])}); every class needs rows on both sides")
    record = {"fold": int(k), "heldout_patients": [int(p) for p in fold["heldout_patients"]],
              "dcis_patient": fold.get("dcis_patient"), "train_rows": int(train.sum()),
              "heldout_rows": int(held.sum()), "train_rows_per_class": dict(zip(class_names, tr)),
              "heldout_rows_per_class": dict(zip(class_names, he))}
    return train, held, record


def _cap_rows(rows: np.ndarray, y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """At most `per_class` rows of each class (a seeded draw), sorted."""
    keep = []
    for c in np.unique(y[rows]):
        rc = rows[y[rows] == c]
        keep.append(rc if rc.size <= per_class else rng.choice(rc, size=per_class, replace=False))
    return np.sort(np.concatenate(keep)) if keep else rows[:0]


class _TaggedSource:
    """A batch source whose `describe()` also names the data behind it (the
    pool's sha256; for latents, the autoencoder's too). The Trainer compares
    `describe()` on resume, so a fold is never continued on other data."""

    def __init__(self, src: Any, identity: Mapping[str, Any]):
        self.src, self.identity = src, dict(identity)

    def num_batches(self, epoch: int) -> int:
        return self.src.num_batches(epoch)

    def get_batch(self, epoch: int, index: int) -> Any:
        return self.src.get_batch(epoch, index)

    def describe(self) -> Dict[str, Any]:
        return {**self.src.describe(), **self.identity}


def _data_sources(X: np.ndarray, pool: Pool, train_mask: np.ndarray, held_mask: Optional[np.ndarray],
                  tcfg: Any, seed: int, layout: str, identity: Mapping[str, Any]) -> Tuple[Any, Any]:
    """(train, held-out) batch sources over `X` (the pool, or its latents):
    class- then patient-balanced batches of the fold's TRAIN rows only; the
    held-out patients' rows in order, for best.pt (None for a final refit)."""
    from hsi_augment.train_generative import ArrayBatchSource, BalancedBatchSampler, SequentialBatches
    sampler = BalancedBatchSampler(pool.y, pool.patient, tcfg.batch_size, seed=seed, rows=train_mask)
    train = _TaggedSource(ArrayBatchSource(X, sampler, extras={"y": pool.y}, layout=layout), identity)
    held = None if held_mask is None else _TaggedSource(ArrayBatchSource(
        X, SequentialBatches(held_mask, tcfg.eval_batch_size), extras={"y": pool.y}, layout=layout), identity)
    return train, held


def _train(trainer: Any, run_dir: Path) -> Any:
    from hsi_augment.train_generative import ResumeError
    try:
        return trainer.fit()
    except ResumeError as e:
        raise StageError(f"{run_dir.name}: cannot resume ({e}); re-run without --resume auto, or with "
                         f"--force") from e


def _rel(ctx: "StageContext", path: Path) -> str:
    return Path(path).resolve().relative_to(Path(ctx.stage_dir).resolve()).as_posix()


def _register(ctx: "StageContext", files: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """add_output each file; {relpath: {sha256, size_bytes}}."""
    out = {}
    for p in files:
        sha = sha256_file(p)
        ctx.add_output(p, sha256=sha)
        out[_rel(ctx, p)] = {"sha256": sha, "size_bytes": int(Path(p).stat().st_size)}
    return out


def _run_fold(ctx: "StageContext", run_dir: Path, ident: Mapping[str, Any],
              fit: Callable[[], Tuple[Dict[str, Any], List[Path]]]) -> Dict[str, Any]:
    """Run one fold in its own directory, or, under --resume auto, keep a fold
    an earlier attempt of this same stage (fingerprint) finished on the same
    inputs (`ident`) whose outputs are intact (module docstring)."""
    marker = run_dir / FOLD_DONE_NAME
    ident = to_jsonable(dict(ident))
    if ctx.resume:
        m = read_json(marker)
        if m and m.get("fingerprint") == ctx.fingerprint and m.get("ident") == ident and all(
                (ctx.stage_dir / r).is_file() and (ctx.stage_dir / r).stat().st_size == meta["size_bytes"]
                for r, meta in (m.get("outputs") or {}).items()):
            for r, meta in m["outputs"].items():
                ctx.add_output(ctx.stage_dir / r, sha256=meta["sha256"])
            ctx.add_output(marker)
            ctx.log(f"{run_dir.name}: finished in an earlier attempt; kept")
            return m["summary"]
    with contextlib.suppress(FileNotFoundError):
        marker.unlink()
    summary, files = fit()
    outputs = _register(ctx, files)
    save_json(marker, {"fingerprint": ctx.fingerprint, "ident": ident, "outputs": outputs, "summary": summary})
    ctx.add_output(marker)
    return summary


def _final_steps(rule: str, lopo_summary: Mapping[str, Any], lopo_max_steps: int, upstream: str) -> Dict[str, Any]:
    folds = list(lopo_summary.get("folds") or [])
    best = [f.get("best_step") for f in folds]
    if rule == "lopo_max_steps":
        return {"rule": rule, "steps": int(lopo_max_steps), "lopo_best_steps": best}
    if not folds or any(b is None for b in best):
        raise StageError(f"{upstream} has no best step for every fold ({best}); it must finish its folds first")
    # rounded half up (Phase 6's rule; round() would take 4.5 to 4)
    return {"rule": rule, "steps": max(1, int(math.floor(float(np.median(best)) + 0.5))), "lopo_best_steps": best}


def _final_train_config(lopo_cfg: Any, steps: int) -> Any:
    """The LOPO config with the final step budget (warm-up kept below it)."""
    return lopo_cfg.replace(max_steps=int(steps), warmup_steps=min(int(lopo_cfg.warmup_steps), int(steps) - 1),
                            early_stop_patience=None)


def _upstream_done(ctx: "StageContext", name: str) -> Dict[str, Any]:
    done = read_done(ctx.upstream_dir(name))
    if not done or done.get("status") != "ok" or not isinstance(done.get("summary"), dict):
        raise StageError(f"{name} has no finished done.json in {ctx.upstream_dir(name)}")
    return done


def _check_upstream_file(ctx: "StageContext", upstream: str, done: Mapping[str, Any], rel: str) -> Tuple[Path, str]:
    """An upstream output, verified against the sha256 its done.json recorded."""
    path = ctx.upstream_dir(upstream) / rel
    rec = (done.get("outputs") or {}).get(rel)
    if rec is None or not path.is_file():
        raise StageError(f"{upstream} has no {rel} (re-run {upstream})")
    sha = sha256_file(path)
    if sha != rec.get("sha256"):
        raise StageError(f"{upstream}/{rel} changed after {upstream} finished (sha256 differs); re-run it")
    return path, sha


# ============================================================================
# autoencoder: fit, latent scale, reconstruction diagnostics
# ============================================================================

def _iter_rows(rows: np.ndarray, batch: int):
    for a in range(0, int(rows.size), int(batch)):
        yield rows[a:a + batch]


def _x_batch(pool: Pool, rows: np.ndarray, device: Any) -> "torch.Tensor":
    import torch
    x = torch.from_numpy(np.ascontiguousarray(pool.X[rows])).permute(0, 3, 1, 2)
    return x.to(device=device, dtype=torch.float32)


def _encode_means(ae: "SpatialAutoencoder", pool: Pool, rows: np.ndarray, batch: int, device: Any
                  ) -> np.ndarray:
    """Posterior means (unscaled) of `rows`, float32 (n, C, L, L)."""
    import torch
    out = np.empty((rows.size, *ae.latent_shape), dtype=np.float32)
    ae.eval()
    with torch.no_grad():
        pos = 0
        for r in _iter_rows(rows, batch):
            out[pos:pos + r.size] = ae.encode(_x_batch(pool, r, device))[0].float().cpu().numpy()
            pos += r.size
    return out


def _latent_stats(ae: "SpatialAutoencoder", pool: Pool, rows: np.ndarray, batch: int, device: Any
                  ) -> Dict[str, Any]:
    """Float64 mean / std over every latent element of `rows` (and per channel)."""
    import torch
    C = ae.latent_channels
    s = np.zeros(C)
    s2 = np.zeros(C)
    n = 0
    ae.eval()
    with torch.no_grad():
        for r in _iter_rows(rows, batch):
            m = ae.encode(_x_batch(pool, r, device))[0].double().transpose(0, 1).reshape(C, -1)
            s += m.sum(dim=1).cpu().numpy()
            s2 += m.pow(2).sum(dim=1).cpu().numpy()
            n += int(m.shape[1])
    mean_c, var_c = s / n, np.maximum(s2 / n - (s / n) ** 2, 0.0)
    mean = float(s.sum() / (n * C))
    std = float(math.sqrt(max(s2.sum() / (n * C) - mean ** 2, 0.0)))
    if not std > 0 or not math.isfinite(std):
        raise StageError(f"latent std over the training rows is {std}: the autoencoder collapsed")
    return {"rows": int(rows.size), "mean": mean, "std": std, "latent_scale": 1.0 / std,
            "channel_mean": mean_c.tolist(), "channel_std": np.sqrt(var_c).tolist()}


def sam_deg_per_sample(v_true: "torch.Tensor", v_pred: "torch.Tensor") -> "torch.Tensor":
    """Mean spectral angle per sample, degrees, (B,) float64, of NCHW
    non-negative (decoded) values; UNCLAMPED and in float64. The shared
    `sam_deg` / `SAMLoss` clamp cos at 1 - 1e-3 (a 2.56 deg floor) and
    compute in float32, which would compress the train / held-out ratios
    these diagnostics exist for (#22's finding)."""
    import torch
    a = v_true.double().permute(0, 2, 3, 1)
    b = v_pred.double().permute(0, 2, 3, 1)
    cos = (a * b).sum(-1) / (a.norm(dim=-1).clamp_min(1e-12) * b.norm(dim=-1).clamp_min(1e-12))
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))).flatten(1).mean(dim=1)


def _recon_metrics(ae: "SpatialAutoencoder", pool: Pool, rows: np.ndarray, class_names: Sequence[str],
                   batch: int, device: Any) -> Dict[str, Any]:
    """Reconstruction through the posterior mean, per class and over all rows:
    MSE in log-z; on decoded values RMSE, PSNR and ssim2d (data range = the
    codec's largest clip_hi, the export's ceiling, so folds and classes share
    one scale) and the unclamped SAM in degrees."""
    import torch
    from training.recon_artifacts import ssim2d          # imported, not reimplemented
    codec = pool.codec
    peak = float(np.max(codec.clip_hi))
    K = len(class_names)
    acc = {k: np.zeros(K) for k in ("n", "se_logz", "se_val", "sam", "ssim")}
    elems = pool.patch * pool.patch * pool.n_bands
    ae.eval()
    with torch.no_grad():
        for r in _iter_rows(rows, batch):
            x = _x_batch(pool, r, device)
            x_hat = ae.decode(ae.encode(x)[0]).float()
            v, v_hat = codec.decode_torch(x), codec.decode_torch(x_hat)
            per = {"se_logz": (x_hat - x).pow(2).flatten(1).sum(1).double(),
                   "se_val": (v_hat - v).pow(2).flatten(1).sum(1).double(),
                   "sam": sam_deg_per_sample(v, v_hat),
                   "ssim": ssim2d(v, v_hat, data_range=peak, reduce=False).double()}
            yb = pool.y[r]
            for key, t in per.items():
                np.add.at(acc[key], yb, t.cpu().numpy())
            np.add.at(acc["n"], yb, 1.0)

    def summarise(sel: np.ndarray) -> Optional[Dict[str, Any]]:
        n = float(acc["n"][sel].sum())
        if n == 0:
            return None
        mse_val = float(acc["se_val"][sel].sum() / (n * elems))
        return {"n": int(n), "mse_logz": float(acc["se_logz"][sel].sum() / (n * elems)),
                "rmse": math.sqrt(mse_val), "psnr": (10.0 * math.log10(peak ** 2 / mse_val) if mse_val > 0 else None),
                "sam_deg": float(acc["sam"][sel].sum() / n), "ssim2d": float(acc["ssim"][sel].sum() / n)}

    out = {name: summarise(np.arange(K) == c) for c, name in enumerate(class_names)}
    out["all"] = summarise(np.ones(K, dtype=bool))
    out["data_range"] = peak
    return out


def _fit_ae(ctx: "StageContext", tcfg: Any, pool: Pool, pool_ident: Mapping[str, Any], train_mask: np.ndarray,
            held_mask: Optional[np.ndarray], run_dir: Path, key: Tuple[Any, ...], device: Any,
            metric_rows: int) -> Tuple[Dict[str, Any], List[Path]]:
    """One autoencoder fit in `run_dir` (a fold, or the final refit), then
    its export (ae.pt) and reconstruction diagnostics."""
    from hsi_augment.models.autoencoder import AELoss, SpatialAutoencoder, count_parameters, make_sam_loss
    from hsi_augment.models.blocks import assert_no_batchnorm
    from hsi_augment.train_generative import (BEST_NAME, HISTORY_NAME, LAST_NAME, Trainer, atomic_torch_save,
                                              load_weights, setup_determinism)
    class_names = list(ctx.adapter.class_names)
    train_src, eval_src = _data_sources(pool.X, pool, train_mask, held_mask, tcfg, ctx.seed_int(*key, "batches"),
                                        "NHWC", {"pool_sha256": pool_ident["pool_sha256"]})

    def build() -> SpatialAutoencoder:
        return SpatialAutoencoder(in_channels=pool.n_bands, patch=pool.patch, widths=tcfg.widths,
                                  blocks_per_level=tcfg.blocks_per_level, latent_channels=tcfg.latent_channels,
                                  dropout=tcfg.dropout)

    setup_determinism(ctx.seed_int(*key, "init"))
    model = build()
    assert_no_batchnorm(model)
    sam = make_sam_loss(pool.codec) if tcfg.sam_weight > 0 else None
    loss = AELoss(sam, sam_weight=tcfg.sam_weight, kl_weight=tcfg.kl_weight)
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx.log(f"{run_dir.name}: autoencoder {count_parameters(model):,} params, latent {model.latent_shape}, "
            f"{int(train_mask.sum())} train rows, {0 if held_mask is None else int(held_mask.sum())} held out, "
            f"{tcfg.max_steps} steps on {device}")
    trainer = Trainer(model, loss, train_src, tcfg, eval_data=eval_src, run_dir=run_dir,
                      seed=ctx.seed_int(*key, "train"), device=device, ctx=ctx, resume=ctx.resume)
    result = _train(trainer, run_dir)

    ema = build()
    ckpt = load_weights(run_dir / BEST_NAME, ema, ema=True)
    ema.to(device).eval()
    train_rows = np.flatnonzero(train_mask)
    stats = _latent_stats(ema, pool, train_rows, tcfg.eval_batch_size, device)
    ema.latent_scale = stats["latent_scale"]
    export = run_dir / AE_EXPORT_NAME
    atomic_torch_save(ema.export_state(source=BEST_NAME, step=int(ckpt["step"]), latent_stats=stats,
                                       codec=pool.codec.to_dict(), pool_sha256=pool_ident["pool_sha256"],
                                       class_names=class_names), export)

    rng = ctx.rng(*key, "metric-rows")
    recon: Dict[str, Any] = {"train": _recon_metrics(ema, pool, _cap_rows(train_rows, pool.y, metric_rows, rng),
                                                     class_names, tcfg.eval_batch_size, device)}
    if held_mask is not None:
        recon["heldout"] = _recon_metrics(ema, pool, _cap_rows(np.flatnonzero(held_mask), pool.y, metric_rows, rng),
                                          class_names, tcfg.eval_batch_size, device)
        recon["heldout_over_train_sam"] = {
            c: (recon["heldout"][c]["sam_deg"] / recon["train"][c]["sam_deg"]
                if recon["heldout"].get(c) and recon["train"].get(c) and recon["train"][c]["sam_deg"] > 0 else None)
            for c in class_names + ["all"]}
    recon_path = run_dir / RECON_NAME
    save_json(recon_path, recon)

    headline = {side: {c: {k: (recon[side][c] or {}).get(k) for k in ("sam_deg", "psnr", "mse_logz")}
                       for c in class_names + ["all"]} for side in ("train", "heldout") if side in recon}
    summary = {**result.summary(), "params": count_parameters(model), "latent_shape": list(model.latent_shape),
               "latent_scale": stats["latent_scale"], "latent_std": stats["std"], "latent_mean": stats["mean"],
               "export_step": int(ckpt["step"]), "recon": headline,
               "heldout_over_train_sam": recon.get("heldout_over_train_sam")}
    files = [run_dir / n for n in (BEST_NAME, LAST_NAME, HISTORY_NAME)] + [export, recon_path]
    return summary, [p for p in files if p.is_file()]


def _load_ae_export(path: Path, device: Any) -> Tuple["SpatialAutoencoder", Dict[str, Any]]:
    import torch
    from hsi_augment.models.autoencoder import SpatialAutoencoder
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return SpatialAutoencoder.from_export(state, device=device), state


# ============================================================================
# latents, denoiser fit, latent MMD
# ============================================================================

def _pool_latents(ctx: "StageContext", run_dir: Path, ae: "SpatialAutoencoder", ae_sha: str,
                  pool: Pool, pool_sha256: str, batch: int, device: Any) -> Tuple[np.ndarray, List[Path]]:
    """The whole pool encoded by `ae` (posterior mean x latent_scale), float32
    NCHW, computed once and cached in the run directory, keyed by the ae.pt
    sha256 and the pool id."""
    path, meta_path = run_dir / LATENTS_NAME, run_dir / LATENTS_META_NAME
    want = {"ae_sha256": ae_sha, "pool_sha256": pool_sha256, "shape": [pool.n, *ae.latent_shape], "dtype": "float32",
            "latent_scale": float(ae.latent_scale), "kind": "posterior mean x latent_scale"}
    if path.is_file() and read_json(meta_path) == want:
        lat = np.load(path)
        if list(lat.shape) == want["shape"] and lat.dtype == np.float32:
            ctx.log(f"{run_dir.name}: cached latents {lat.shape} reused")
            return lat, [path, meta_path]
    lat = _encode_means(ae, pool, np.arange(pool.n), batch, device) * np.float32(ae.latent_scale)
    if not np.isfinite(lat).all():
        raise StageError("the autoencoder produced non-finite latents")
    run_dir.mkdir(parents=True, exist_ok=True)
    from hsi_augment.io import save_npy
    save_npy(path, lat)
    save_json(meta_path, want)
    ctx.log(f"{run_dir.name}: encoded {pool.n} pool rows -> latents {lat.shape}")
    return lat, [path, meta_path]


def mmd_rbf(x: "torch.Tensor", y: "torch.Tensor", bandwidths: Sequence[float]) -> float:
    """Biased (V-statistic) squared MMD with a sum of RBF kernels,
    k(a, b) = sum_h exp(-|a - b|^2 / (2 h^2)), in float64. Always >= 0, so
    the ratio of two of them is defined; with equal set sizes the two
    comparisons it is used for carry the same bias."""
    import torch
    x, y = x.double().flatten(1), y.double().flatten(1)

    def k(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2.0 * h * h)) for h in bandwidths)

    return max(float((k(x, x).mean() + k(y, y).mean() - 2.0 * k(x, y).mean()).item()), 0.0)


def _median_distance(z: "torch.Tensor", rng: np.random.Generator) -> float:
    import torch
    if z.shape[0] > MEDIAN_SAMPLE_ROWS:
        z = z[torch.from_numpy(np.sort(rng.choice(z.shape[0], MEDIAN_SAMPLE_ROWS, replace=False)))]
    d = torch.cdist(z.double().flatten(1), z.double().flatten(1))
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
    med = float(d[iu[0], iu[1]].median().item()) if iu.shape[1] else 1.0
    return med if med > 0 else 1.0


def _latent_mmd(ctx: "StageContext", tcfg: Any, denoiser: "LatentUNet", latents: np.ndarray, pool: Pool,
                train_mask: np.ndarray, held_mask: np.ndarray, key: Tuple[Any, ...], device: Any) -> Dict[str, Any]:
    """Per class c: MMD(generated_w, held-out) for each guidance scale w,
    against MMD(train, held-out), all on the scaled latents of fold k's
    autoencoder and with one kernel per class (median heuristic on held-out
    and train reference pooled). ratio <= MMD_PASS_RATIO passes (Plan §Phase 7)."""
    import torch
    from hsi_augment.models.diffusion import sample_latents
    from hsi_augment.train_generative import derive_seed
    rng = ctx.rng(*key, "mmd-rows")
    seed = ctx.seed_int(*key, "mmd-noise")
    out: Dict[str, Any] = {"pass_ratio": MMD_PASS_RATIO, "steps": tcfg.sample_steps, "classes": {}}
    for c, name in enumerate(ctx.adapter.class_names):
        held = np.flatnonzero(held_mask & (pool.y == c))
        ref = np.flatnonzero(train_mask & (pool.y == c))
        n = int(min(tcfg.mmd_rows, held.size, ref.size))
        if n < 2:
            out["classes"][name] = None
            continue
        H = torch.from_numpy(latents[np.sort(rng.choice(held, n, replace=False))])
        R = torch.from_numpy(latents[np.sort(rng.choice(ref, n, replace=False))])
        med = _median_distance(torch.cat([H, R]), rng)
        bw = [m * med for m in tcfg.mmd_bandwidths]
        base = math.sqrt(mmd_rbf(R, H, bw))
        by_w = {}
        for w in tcfg.guidance:
            G = sample_latents(denoiser, c, n, w, derive_seed(seed, c), steps=tcfg.sample_steps,
                               batch_size=tcfg.sample_batch_size, device=device)
            m = math.sqrt(mmd_rbf(G, H, bw))
            ratio = m / base if base > 0 else None
            by_w[f"{w:g}"] = {"mmd": m, "ratio": ratio, "pass": bool(ratio is not None and ratio <= MMD_PASS_RATIO)}
        out["classes"][name] = {"n": n, "median_distance": med, "bandwidths": bw, "mmd_train_heldout": base,
                                "by_w": by_w}
    return out


def _sample_check(gen: "LatentDiffusionGenerator", class_id: int, w: float, seed: int, steps: int,
                  batch_size: int) -> Dict[str, Any]:
    """A small deterministic sample from the EXPORTED generator: its sha256
    lets the host check that a re-run, or #29's export code, reproduces it
    (with the same batch size and device, which are recorded with it)."""
    x = gen.sample(class_id, SAMPLE_CHECK_ROWS, w, seed, steps=steps, batch_size=batch_size)
    xf = x.astype(np.float32)
    return {"class_id": int(class_id), "n": SAMPLE_CHECK_ROWS, "w": float(w), "seed": int(seed), "steps": int(steps),
            "batch_size": int(batch_size), "device": str(gen.device), "shape": list(x.shape), "dtype": str(x.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            "finite": bool(np.isfinite(xf).all()), "min": float(xf.min()), "max": float(xf.max())}


def _fit_ldm(ctx: "StageContext", tcfg: Any, pool: Pool, pool_ident: Mapping[str, Any], ae_state: Mapping[str, Any],
             ae: "SpatialAutoencoder", ae_sha: str, train_mask: np.ndarray, held_mask: Optional[np.ndarray],
             run_dir: Path, key: Tuple[Any, ...], device: Any) -> Tuple[Dict[str, Any], List[Path]]:
    """Latents, one denoiser fit in `run_dir`, the generator.pt export, the
    sampling check and (with held-out rows) the latent MMD."""
    from hsi_augment.models.autoencoder import count_parameters
    from hsi_augment.models.blocks import assert_no_batchnorm
    from hsi_augment.models.diffusion import DiffusionLoss, LatentUNet
    from hsi_augment.train_generative import (BEST_NAME, HISTORY_NAME, LAST_NAME, Trainer, atomic_torch_save,
                                              load_weights, setup_determinism)
    class_names = list(ctx.adapter.class_names)
    run_dir.mkdir(parents=True, exist_ok=True)
    latents, lat_files = _pool_latents(ctx, run_dir, ae, ae_sha, pool, pool_ident["pool_sha256"],
                                       tcfg.encode_batch_size, device)
    train_src, eval_src = _data_sources(latents, pool, train_mask, held_mask, tcfg, ctx.seed_int(*key, "batches"),
                                        "NCHW", {"pool_sha256": pool_ident["pool_sha256"], "ae_sha256": ae_sha})

    def build() -> LatentUNet:
        return LatentUNet(latent_channels=ae.latent_channels, latent_size=ae.latent_hw, channels=tcfg.channels,
                          blocks_per_level=tcfg.blocks_per_level, mid_blocks=tcfg.mid_blocks,
                          num_classes=len(class_names), emb_dim=tcfg.emb_dim, freq_dim=tcfg.freq_dim,
                          dropout=tcfg.dropout, timesteps=tcfg.timesteps)

    setup_determinism(ctx.seed_int(*key, "init"))
    model = build()
    assert_no_batchnorm(model)
    loss = DiffusionLoss(model.schedule, min_snr_gamma=tcfg.min_snr_gamma, label_dropout=tcfg.label_dropout)
    ctx.log(f"{run_dir.name}: denoiser {count_parameters(model):,} params on latents {model.latent_shape}, "
            f"{int(train_mask.sum())} train rows, {0 if held_mask is None else int(held_mask.sum())} held out, "
            f"{tcfg.max_steps} steps on {device} ({tcfg.precision})")
    trainer = Trainer(model, loss, train_src, tcfg, eval_data=eval_src, run_dir=run_dir,
                      seed=ctx.seed_int(*key, "train"), device=device, ctx=ctx, resume=ctx.resume)
    result = _train(trainer, run_dir)

    ema = build()
    ckpt = load_weights(run_dir / BEST_NAME, ema, ema=True)
    gen_path = run_dir / GENERATOR_NAME
    atomic_torch_save({
        "format": GENERATOR_FORMAT, "ae": dict(ae_state), "ae_sha256": ae_sha,
        "denoiser": ema.export_state(source=BEST_NAME, step=int(ckpt["step"])),
        "codec": pool.codec.to_dict(), "class_names": class_names,
        "minority_classes": [int(c) for c in ctx.adapter.minority_classes],
        "guidance_candidates": [float(w) for w in tcfg.guidance], "sample_steps": int(tcfg.sample_steps),
        "pool_sha256": pool_ident["pool_sha256"]}, gen_path)

    gen = load_generator(gen_path, device=device)          # the round trip #29 will make
    minority = int(ctx.adapter.minority_classes[0]) if ctx.adapter.minority_classes else 0
    check = _sample_check(gen, minority, float(tcfg.guidance[0]), ctx.seed_int(*key, "sample-check"),
                          tcfg.sample_steps, tcfg.sample_batch_size)
    if not check["finite"]:
        raise StageError(f"{run_dir.name}: the exported generator produced non-finite samples")
    files = lat_files + [run_dir / n for n in (BEST_NAME, LAST_NAME, HISTORY_NAME)] + [gen_path]
    summary = {**result.summary(), "params": count_parameters(model), "latent_shape": list(model.latent_shape),
               "export_step": int(ckpt["step"]), "ae_sha256": ae_sha,
               "guidance_candidates": [float(w) for w in tcfg.guidance], "sample_check": check}
    if held_mask is not None:
        mmd = _latent_mmd(ctx, tcfg, gen.denoiser, latents, pool, train_mask, held_mask, key, device)
        save_json(run_dir / MMD_NAME, mmd)
        files.append(run_dir / MMD_NAME)
        summary["latent_mmd_ratio"] = {c: (None if v is None else {w: r["ratio"] for w, r in v["by_w"].items()})
                                       for c, v in mmd["classes"].items()}
    return summary, [p for p in files if p.is_file()]


# ============================================================================
# the exported generator (#29's entry point)
# ============================================================================

@dataclass
class LatentDiffusionGenerator:
    """What `generator.pt` holds, rebuilt: the autoencoder and the EMA
    denoiser (eval mode, on `device`), the codec, and the run's metadata.
    `sample(...)` is `models.diffusion.sample` bound to them."""
    ae: "SpatialAutoencoder"
    denoiser: "LatentUNet"
    codec: LogSpaceCodec
    meta: Dict[str, Any]
    device: Any

    def sample(self, class_id: int, n: int, w: float, seed: int, start: int = 0, steps: Optional[int] = None, *,
               eta: float = 0.0, batch_size: int = 1024) -> np.ndarray:
        from hsi_augment.models.diffusion import sample
        return sample(self.ae, self.denoiser, self.codec, class_id, n, w, seed, start,
                      int(steps or self.meta.get("sample_steps") or 50), eta=eta, batch_size=batch_size,
                      device=self.device)


def load_generator(path: PathLike, *, device: Any = None) -> LatentDiffusionGenerator:
    """Load `generator.pt` (or the run directory holding it), with
    `torch.load(weights_only=True)`."""
    import torch
    from hsi_augment.models.autoencoder import SpatialAutoencoder
    from hsi_augment.models.diffusion import LatentUNet
    p = Path(path)
    if p.is_dir():
        p = p / GENERATOR_NAME
    state = torch.load(str(p), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != GENERATOR_FORMAT:
        raise ValueError(f"{p} is not a {GENERATOR_FORMAT} file")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    meta = {k: v for k, v in state.items() if k not in ("ae", "denoiser", "codec")}
    meta["path"] = str(p)
    return LatentDiffusionGenerator(ae=SpatialAutoencoder.from_export(state["ae"], device=dev),
                                    denoiser=LatentUNet.from_export(state["denoiser"], device=dev),
                                    codec=LogSpaceCodec.from_dict(state["codec"]), meta=meta, device=dev)


# ============================================================================
# the stages
# ============================================================================

def _select_folds(ctx: "StageContext", folds: Sequence[int]) -> List[Mapping[str, Any]]:
    try:
        doc = load_folds(ctx.upstream_dir("leakage"))
    except (OSError, ValueError) as e:
        raise StageError(f"cannot read the LOPO folds: {e}") from e
    by_k = {int(f["fold"]): f for f in doc["folds"]}
    missing = [k for k in folds if k not in by_k]
    if missing:
        raise StageError(f"fold(s) {missing} are not in folds.json (folds {sorted(by_k)})")
    return [by_k[k] for k in folds]


def run_ae_lopo(ctx: "StageContext", cfg: Any) -> Dict[str, Any]:
    pool, ident = _open_pool(ctx)
    device = _device(cfg.device)
    class_names = list(ctx.adapter.class_names)
    results = []
    for fold in _select_folds(ctx, cfg.folds):
        k = int(fold["fold"])
        train, held, record = _fold_split(pool, fold, k, class_names)
        run_dir = ctx.stage_dir / f"fold-{k}"
        with ctx.timer(f"fold-{k}"):
            summary = _run_fold(ctx, run_dir, {"pool_sha256": ident["pool_sha256"], "fold": record},
                                lambda: _fit_ae(ctx, cfg, pool, ident, train, held, run_dir, ("fold", k), device,
                                                cfg.metric_rows))
        results.append({"fold": record, **summary})
    return {"pool": ident, "device": str(device), "folds": results}


def run_ae_final(ctx: "StageContext", cfg: Any) -> Dict[str, Any]:
    from hsi_augment.stages.fit_ae import AELopoConfig
    lopo = _upstream_done(ctx, "ae-lopo")
    lopo_cfg = AELopoConfig.from_dict(lopo["config"])
    pool, ident = _open_pool(ctx, expect_sha256=lopo["summary"]["pool"]["pool_sha256"], upstream="ae-lopo")
    steps = _final_steps(cfg.steps, lopo["summary"], lopo_cfg.max_steps, "ae-lopo")
    tcfg = _final_train_config(lopo_cfg, steps["steps"])
    device = _device(cfg.device)
    ctx.log(f"final refit with ae-lopo's settings: {steps['steps']} steps ({steps['rule']}, LOPO best steps "
            f"{steps['lopo_best_steps']})")
    with ctx.timer("fit"):
        summary, files = _fit_ae(ctx, tcfg, pool, ident, np.ones(pool.n, dtype=bool), None, ctx.stage_dir,
                                 ("final",), device, cfg.metric_rows)
    _register(ctx, files)
    return {"pool": ident, "device": str(device), "steps": steps, "train_config": tcfg.to_dict(), **summary}


def run_ldm_lopo(ctx: "StageContext", cfg: Any) -> Dict[str, Any]:
    ae_done = _upstream_done(ctx, "ae-lopo")
    pool, ident = _open_pool(ctx, expect_sha256=ae_done["summary"]["pool"]["pool_sha256"], upstream="ae-lopo")
    device = _device(cfg.device)
    class_names = list(ctx.adapter.class_names)
    ae_folds = {int(f["fold"]["fold"]): f["fold"] for f in ae_done["summary"]["folds"]}
    missing = [k for k in cfg.folds if k not in ae_folds]
    if missing:
        raise StageError(f"fold(s) {missing} have no autoencoder in ae-lopo (its folds: {sorted(ae_folds)}); "
                         f"add them to ae-lopo's --folds")
    results = []
    for k in cfg.folds:
        record = ae_folds[k]
        train, held, rec = _fold_split(pool, record, k, class_names)
        ae_path, ae_sha = _check_upstream_file(ctx, "ae-lopo", ae_done, f"fold-{k}/{AE_EXPORT_NAME}")
        ae, ae_state = _load_ae_export(ae_path, device)
        run_dir = ctx.stage_dir / f"fold-{k}"
        with ctx.timer(f"fold-{k}"):
            summary = _run_fold(ctx, run_dir,
                                {"pool_sha256": ident["pool_sha256"], "fold": rec, "ae_sha256": ae_sha},
                                lambda: _fit_ldm(ctx, cfg, pool, ident, ae_state, ae, ae_sha, train, held, run_dir,
                                                 ("fold", k), device))
        results.append({"fold": rec, **summary})
    return {"pool": ident, "device": str(device), "guidance_candidates": [float(w) for w in cfg.guidance],
            "guidance_choice": "not made here: the host LOPO QC (#26 / #28) picks w", "folds": results}


def run_ldm_final(ctx: "StageContext", cfg: Any) -> Dict[str, Any]:
    from hsi_augment.stages.fit_ldm import LDMLopoConfig
    ae_done = _upstream_done(ctx, "ae-final")
    lopo = _upstream_done(ctx, "ldm-lopo")
    lopo_cfg = LDMLopoConfig.from_dict(lopo["config"])
    pool_shas = {ae_done["summary"]["pool"]["pool_sha256"], lopo["summary"]["pool"]["pool_sha256"]}
    if len(pool_shas) != 1:
        raise StageError("ae-final and ldm-lopo were trained on different pools; re-run them on one pool")
    pool, ident = _open_pool(ctx, expect_sha256=pool_shas.pop(), upstream="ae-final")
    steps = _final_steps(cfg.steps, lopo["summary"], lopo_cfg.max_steps, "ldm-lopo")
    tcfg = _final_train_config(lopo_cfg, steps["steps"])
    device = _device(cfg.device)
    ae_path, ae_sha = _check_upstream_file(ctx, "ae-final", ae_done, AE_EXPORT_NAME)
    ae, ae_state = _load_ae_export(ae_path, device)
    ctx.log(f"final refit with ldm-lopo's settings: {steps['steps']} steps ({steps['rule']}, LOPO best steps "
            f"{steps['lopo_best_steps']})")
    with ctx.timer("fit"):
        summary, files = _fit_ldm(ctx, tcfg, pool, ident, ae_state, ae, ae_sha, np.ones(pool.n, dtype=bool), None,
                                  ctx.stage_dir, ("final",), device)
    _register(ctx, files)
    return {"pool": ident, "device": str(device), "steps": steps, "train_config": tcfg.to_dict(),
            "guidance_choice": "not made here: the host LOPO QC (#26 / #28) picks w", **summary}
