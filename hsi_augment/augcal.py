# -*- coding: utf-8 -*-
"""
hsi_augment/augcal.py
=====================
How large the physics ops (`hsi_augment/transforms.py`, #19) may be, measured
on the build itself (#20), and how `train.py --phys_tilt_max auto` reads the
answer back.

The criterion (decision of 2026-10-27 on #20)
---------------------------------------------
A physics op may not change the band-band correlation matrix more than the
paper recipe's OWN augmentation already does, on the same data:

    corr_frobenius(op)  <=  corr_frobenius(paper_recipe base augmentation)

`corr_frobenius` is `augmentation_report`'s ||R(x') - R(x)||_F, with R the
correlation over every pixel of a seeded train sample. This replaces the plan's
absolute bound (< 0.05), which `spectral_tilt` missed by a factor of 20 at the
plan's default `tilt_max` 0.03; the base augmentation itself moves R by about
0.3 on #19's synthetic model, so an absolute 0.05 asked the physics ops for
something the recipe never met. The change of `spectral_tilt` grows as
tilt_max^2, so the bound is a tilt_max: the largest value of a grid whose
change is within the base's. `sensor_noise` has no free magnitude once sigma_b
is measured (#12), so it is CHECKED at k = 1 against the same number, not
calibrated.

The base augmentation is `AugmentationConfig` built from
`training.train_profiles.OPTIMAL_DEFAULTS` (`paper_recipe_aug_config`), the
values `train.py --profile paper_recipe` passes, read rather than copied. It
runs on the NORMALIZED patch, as in training, so the sample is normalized with
the statistics train.py fits (`compute_global_channel_stats`, replicated by
`preprocess.train_py_sample_stats`). The physics ops run in value space, so
their change does not depend on those statistics; the base's does, a little.
Spatial steps (flips, rot90, crop) move pixels: the pooled correlation is blind
to a permutation of pixels, the crop's nearest-neighbour resize duplicates some
and drops others. The base's `sam_deg` compares pixel i before and after those
moves, so it is not a spectral angle and is not used.

Outputs of the `aug-calibration` stage (`hsi_augment/stages/aug_calibration.py`),
in `<out>/prep/augcal/`:

    phys_calibration.json     the decision record; `tilt_max` is what --phys_tilt_max auto reads
    augmentation_report.json  {"scan": base + grid + noise, "chosen": the chosen ops, all in order}
    augmentation_montage.png  `save_augmentation_montage` of the chosen ops (optional)

train.py imports this module (`resolve_tilt_max`), so its module-level imports
are numpy, torch (through `transforms`) and the standard library.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from hsi_augment.transforms import (
    MAX_TILT, PHYS_ARTIFACT_FILES, PHYS_ARTIFACT_SUBDIRS, ArrayLike, AugmentFn, PhysicsOp, SensorNoise,
    SpectralTilt, augmentation_report,
)

CALIBRATION_KIND = "phys_aug_calibration"
CALIBRATION_SCHEMA_VERSION = 1
PHYS_CALIBRATION_NAME = "phys_calibration.json"
AUG_REPORT_NAME = "augmentation_report.json"
MONTAGE_NAME = "augmentation_montage.png"
# `--phys_artifacts DIR` is searched for phys_calibration.json where
# `load_phys_artifacts` looks for noise_sigma.npy, then in the stage's own
# directory, so DIR may be the pipeline's --out, its prep/, or prep/augcal.
CALIBRATION_SUBDIRS = tuple(PHYS_ARTIFACT_SUBDIRS) + ("augcal", "prep/augcal")

BASE_OP_NAME = "paper_recipe_base"
TILT_OP_PREFIX = "spectral_tilt@"
CRITERION = ("corr_frobenius(op) <= corr_frobenius(paper_recipe base augmentation) on the same seeded, "
             "normalized train sample (decision of 2026-10-27 on #20; replaces the plan's absolute < 0.05)")
METRIC = ("||R(x') - R(x)||_F, R the band-band correlation over every pixel of the sample, in value space "
          "(hsi_augment.transforms.augmentation_report)")


# ============================================================================
# The base augmentation, measured like an op
# ============================================================================

def paper_recipe_aug_config(seed: int = 0):
    """The `AugmentationConfig` `train.py --profile paper_recipe` builds, read from
    `OPTIMAL_DEFAULTS` (fields it does not set keep the parser's defaults, which
    are `AugmentationConfig`'s own)."""
    from training.augmentation import AugmentationConfig  # frozen
    from training.train_profiles import OPTIMAL_DEFAULTS

    if OPTIMAL_DEFAULTS.get("augment_preset") != "custom":
        raise ValueError(f"OPTIMAL_DEFAULTS sets augment_preset {OPTIMAL_DEFAULTS.get('augment_preset')!r}; "
                         f"the calibration expects the 'custom' recipe built from its flags")
    names = {f.name for f in dataclasses.fields(AugmentationConfig)} - {"seed"}
    return AugmentationConfig(**{k: OPTIMAL_DEFAULTS[k] for k in sorted(names & set(OPTIMAL_DEFAULTS))},
                              seed=int(seed))


class NormalizedAugment(PhysicsOp):
    """An augmentation that works on the NORMALIZED patch (the recipe's
    `Compose`), wrapped so `augmentation_report` measures it next to the
    physics ops: same value space, same per-item (rng, gen)."""

    def __init__(self, fn: AugmentFn, global_mean: ArrayLike, global_std: ArrayLike, *,
                 name: str = BASE_OP_NAME, params: Optional[Mapping[str, Any]] = None):
        super().__init__(global_mean, global_std)
        self.fn = fn
        self.name = name
        self.params = dict(params or {})

    def __call__(self, patch, rng, gen=None):
        return self.fn(patch, rng, gen)

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "space": "normalized", **self.params}


def base_augment_op(global_mean: ArrayLike, global_std: ArrayLike, cfg=None) -> NormalizedAugment:
    """`build_active(cfg)` (default: `paper_recipe_aug_config()`) as a measurable op."""
    from training.augmentation_v16 import build_active

    cfg = paper_recipe_aug_config() if cfg is None else cfg
    params = {k: v for k, v in dataclasses.asdict(cfg).items() if k != "seed"}
    return NormalizedAugment(build_active(cfg), global_mean, global_std,
                             params={"source": "training.train_profiles.OPTIMAL_DEFAULTS (paper_recipe)",
                                     "config": params})


def tilt_grid_ops(wavelengths: ArrayLike, global_mean: ArrayLike, global_std: ArrayLike,
                  grid: Sequence[float]) -> List[SpectralTilt]:
    """One `SpectralTilt` per grid value, named `spectral_tilt@<t>` so one
    `augmentation_report` pass measures them all against one original."""
    ops = []
    for t in grid:
        op = SpectralTilt(wavelengths, global_mean, global_std, tilt_max=float(t))
        op.name = f"{TILT_OP_PREFIX}{float(t):g}"
        ops.append(op)
    return ops


# ============================================================================
# The decision
# ============================================================================

def check_grid(grid: Sequence[float]) -> Tuple[float, ...]:
    g = tuple(float(t) for t in grid)
    if not g:
        raise ValueError("the tilt_max grid is empty")
    if any(not (0.0 < t < MAX_TILT) for t in g):
        raise ValueError(f"every grid value must be in (0, {MAX_TILT}), got {g}")
    if len(set(g)) != len(g):
        raise ValueError(f"the grid has duplicate values: {g}")
    return tuple(sorted(g))


def choose_tilt(base_change: float, grid_changes: Mapping[float, float]) -> Dict[str, Any]:
    """The largest grid value whose change is within `base_change`, or 0 (off)
    if none is. `bound` says whether the answer is a real bound ("interior"),
    capped by the grid ("grid_max": a larger value might pass too) or below it
    ("below_grid")."""
    rows = [{"tilt_max": float(t), "corr_frobenius": float(c), "within_base": bool(c <= base_change)}
            for t, c in sorted(grid_changes.items())]
    ok = [r["tilt_max"] for r in rows if r["within_base"]]
    changes = [r["corr_frobenius"] for r in rows]
    chosen = max(ok) if ok else 0.0
    bound = "below_grid" if not ok else ("grid_max" if chosen == rows[-1]["tilt_max"] else "interior")
    return {"tilt_max": chosen, "bound": bound, "grid": rows,
            "monotone": bool(all(a <= b for a, b in zip(changes, changes[1:])))}


def calibrate(samples: Union[np.ndarray, torch.Tensor], wavelengths: ArrayLike, global_mean: ArrayLike,
              global_std: ArrayLike, noise_sigma: Optional[ArrayLike], grid: Sequence[float], *,
              noise_k: float = 1.0, base_cfg=None, seed: int = 0,
              montage_path: Optional[Union[str, Path]] = None, montage_n: int = 4
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The calibration on normalized `samples` [N, C, H, W]. Returns
    (decision, report): `decision` is `phys_calibration.json` minus the stage's
    provenance; `report` is {"scan", "chosen"} from `augmentation_report`."""
    grid = check_grid(grid)
    # contiguous, like the dataset's patches: the recipe's spectral_noise fills `empty_like(patch)` in
    # memory order, so a strided view would put the same draws on other elements
    samples = (samples if isinstance(samples, torch.Tensor) else torch.from_numpy(np.asarray(samples))).contiguous()
    base = base_augment_op(global_mean, global_std, base_cfg)
    tilts = tilt_grid_ops(wavelengths, global_mean, global_std, grid)
    noise = None
    if noise_sigma is not None and noise_k > 0:
        noise = SensorNoise(noise_sigma, global_mean, global_std, k=noise_k)
    scan = augmentation_report(samples, [base, *tilts] + ([noise] if noise is not None else []), wavelengths,
                               seed=seed, include_all=False)
    base_change = scan["ops"][BASE_OP_NAME]["corr_frobenius"]
    choice = choose_tilt(base_change, {op.tilt_max: scan["ops"][op.name]["corr_frobenius"] for op in tilts})

    noise_check: Dict[str, Any] = {"k": float(noise_k), "checked": noise is not None}
    if noise is not None:
        c = scan["ops"][noise.name]["corr_frobenius"]
        noise_check.update(corr_frobenius=c, within_base=bool(c <= base_change), **noise.describe())

    chosen_ops: List[PhysicsOp] = []
    if choice["tilt_max"] > 0:
        chosen_ops.append(SpectralTilt(wavelengths, global_mean, global_std, tilt_max=choice["tilt_max"]))
    if noise is not None:
        chosen_ops.append(noise)
    chosen = (augmentation_report(samples, chosen_ops, wavelengths, seed=seed, montage_path=montage_path,
                                  montage_n=montage_n) if chosen_ops else None)
    combined = None
    if chosen is not None:
        key = "all" if "all" in chosen["ops"] else chosen_ops[0].name
        combined = {"ops": [op.name for op in chosen_ops],
                    "corr_frobenius": chosen["ops"][key]["corr_frobenius"],
                    "within_base": bool(chosen["ops"][key]["corr_frobenius"] <= base_change),
                    "note": "informational: the criterion is per op"}

    decision = {
        "criterion": CRITERION, "metric": METRIC,
        "base_augmentation": base.describe(),
        "base_corr_change": base_change,
        "base_sam_note": "the base's spatial steps move pixels, so its sam_deg is not a spectral angle; unused",
        "grid": choice["grid"],
        "grid_monotone": choice["monotone"],
        "tilt_max": choice["tilt_max"],
        "tilt_bound": choice["bound"],
        "tilt_corr_change": next((r["corr_frobenius"] for r in choice["grid"]
                                  if r["tilt_max"] == choice["tilt_max"]), None),
        "noise_check": noise_check,
        "combined": combined,
        "report_seed": int(seed),
        "n_samples": int(scan["n_samples"]), "patch_shape": scan["patch_shape"],
        "wavelengths_nm": scan["wavelengths_nm"],
        "montage": None if chosen is None else chosen["montage"],
    }
    return decision, {"scan": scan, "chosen": chosen}


# ============================================================================
# Reading it back (`train.py --phys_tilt_max auto`)
# ============================================================================

def find_phys_calibration(directory: Union[str, Path]) -> Optional[Path]:
    """The first `phys_calibration.json` under `directory`, in `CALIBRATION_SUBDIRS` order."""
    root = Path(directory)
    for sub in CALIBRATION_SUBDIRS:
        p = root / sub / PHYS_CALIBRATION_NAME
        if p.is_file():
            return p
    return None


def read_phys_calibration(path: Union[str, Path]) -> Dict[str, Any]:
    """The parsed file, with `tilt_max` checked; ValueError if it is not one."""
    try:
        with open(path) as f:
            cal = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {path}: {e}")
    kind = cal.get("kind") if isinstance(cal, dict) else None
    if kind != CALIBRATION_KIND:
        raise ValueError(f"{path} is not a {CALIBRATION_KIND} file (kind {kind!r})")
    t = cal.get("tilt_max")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0.0 <= t < MAX_TILT:
        raise ValueError(f"{path}: tilt_max must be a number in [0, {MAX_TILT}), got {t!r}")
    return cal


def resolve_tilt_max(value: Union[str, float], directory: Optional[Union[str, Path]]
                     ) -> Tuple[float, Dict[str, Any]]:
    """`--phys_tilt_max`: a number is used as given; 'auto' reads the
    calibration under `--phys_artifacts`. Returns (tilt_max, where it came
    from); FileNotFoundError / ValueError when 'auto' cannot be resolved."""
    if value != "auto":
        return float(value), {"source": "--phys_tilt_max", "calibration": None}
    how = (f"--phys_tilt_max auto reads tilt_max from {PHYS_CALIBRATION_NAME} (augment_pipeline.py "
           f"aug-calibration --data_dir <build> --out <out>)")
    if directory is None:
        raise FileNotFoundError(f"{how} under --phys_artifacts, which is not set; pass --phys_artifacts <out> "
                                f"or a number")
    path = find_phys_calibration(directory)
    if path is None:
        raise FileNotFoundError(f"{how}; there is none under {directory} (searched "
                                f"{', '.join(s or '.' for s in CALIBRATION_SUBDIRS)}); run the stage or pass "
                                f"a number")
    cal = read_phys_calibration(path)
    from hsi_augment.io import sha256_file
    return float(cal["tilt_max"]), {
        "source": "calibration", "calibration": {
            "path": str(path), "sha256": sha256_file(path),
            **{k: cal.get(k) for k in ("tilt_bound", "base_corr_change", "tilt_corr_change", "noise_check",
                                       "input_manifest_sha256", "sample", "seed", "report_seed")}}}


# ============================================================================
# The stage body
# ============================================================================

def run_aug_calibration(ctx: Any, cfg: Any) -> Dict[str, Any]:
    """`aug-calibration` (see `stages/aug_calibration.py` for the config)."""
    from hsi_augment import preprocess as pp
    from hsi_augment.io import iter_chunks, sha256_file
    from hsi_augment.stage import StageError

    adapter = ctx.adapter
    noise_path = ctx.upstream_dir("spectral-audit") / PHYS_ARTIFACT_FILES["noise_sigma"]
    if not noise_path.is_file():
        raise StageError(f"{noise_path} is missing; re-run spectral-audit")
    noise_sigma = np.load(noise_path).astype(np.float32)
    try:
        wl = np.asarray(adapter.wavelengths, dtype=np.float64)
    except FileNotFoundError as e:
        raise StageError(f"spectral_tilt needs the band wavelengths: {e}")

    train = ctx.open_split("train", purpose="fit", kinds=("X",))
    X = train.X
    if X is None or X.ndim != 4:
        raise StageError(f"X_train must be (N, H, W, C); got {None if X is None else X.shape}")
    N, H, W, C = (int(v) for v in X.shape)
    if wl.shape != (C,) or noise_sigma.shape != (C,):
        raise StageError(f"X_train has {C} bands, wavelengths.npy {wl.shape}, noise_sigma.npy {noise_sigma.shape}")

    # train.py's normalization: compute_global_channel_stats(X_train, sample_cap, seed), bit for bit
    with ctx.timer("norm_stats"):
        mean, std = pp.train_py_sample_stats(X, cfg.norm_sample_cap, cfg.norm_seed)
    rows = pp.sample_rows(ctx.rng("sample"), N, cfg.sample_rows)
    n = int(len(rows))
    ctx.ram_guard(2 * n * H * W * C * 4, "normalized float32 sample [n, C, H, W] + one chunk")
    ctx.log(f"sample: {n} of {N} train rows ({n * H * W} pixels of {C} bands), normalized with train.py's "
            f"estimate (sample_cap {cfg.norm_sample_cap}, seed {cfg.norm_seed})")
    samples = np.empty((n, C, H, W), dtype=np.float32)
    with ctx.timer("read"):
        at = 0
        for ch in iter_chunks(X, rows, chunk_rows=cfg.chunk_rows):
            x = np.nan_to_num(np.asarray(ch.data, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            samples[at:at + len(x)] = ((x - mean) / std).transpose(0, 3, 1, 2)
            at += len(x)
    if at != n:
        raise StageError(f"read {at} sample rows, expected {n}")

    seed = ctx.seed_int("augment")
    montage = ctx.path(MONTAGE_NAME) if cfg.montage else None
    with ctx.timer("measure"):
        decision, report = calibrate(torch.from_numpy(samples), wl, mean, std, noise_sigma, cfg.tilt_grid,
                                     noise_k=cfg.noise_k, seed=seed, montage_path=montage,
                                     montage_n=cfg.montage_n)
    del samples

    decision.update({
        "schema_version": CALIBRATION_SCHEMA_VERSION, "kind": CALIBRATION_KIND,
        "sample": {"split": "train", "purpose": "fit", "rows": n, "rows_total": N, "pixels": n * H * W,
                   "patch_shape": [H, W], "n_bands": C, "seed_key": "sample"},
        "normalization": {"method": "training.npy_data.compute_global_channel_stats (train.py global_zscore)",
                          "sample_cap": int(cfg.norm_sample_cap), "seed": int(cfg.norm_seed),
                          "mean": mean, "std": std},
        "seed": int(ctx.seed),
        "noise_sigma": {"path": str(noise_path), "sha256": sha256_file(noise_path)},
        "input_manifest_sha256": sha256_file(adapter.manifest_path),
        "augmentation_report": AUG_REPORT_NAME,
    })
    if decision["montage"] is not None:
        ctx.add_output(decision["montage"])
        decision["montage"] = MONTAGE_NAME
    ctx.save_json(AUG_REPORT_NAME, report)
    ctx.save_json(PHYS_CALIBRATION_NAME, decision)

    b, t = decision["base_corr_change"], decision["tilt_max"]
    ctx.log(f"base augmentation (paper_recipe) moves the correlation by {b:.4g}; spectral_tilt: tilt_max {t:g} "
            f"({decision['tilt_bound']}, change {decision['tilt_corr_change']})")
    if decision["tilt_bound"] == "below_grid":
        ctx.warn("no grid value of tilt_max is within the base's change; tilt_max = 0, so "
                 "--phys_tilt_max auto turns spectral_tilt off")
    elif decision["tilt_bound"] == "grid_max":
        ctx.warn(f"the largest grid value ({t:g}) is within the base's change; the bound may be higher")
    if not decision["grid_monotone"]:
        ctx.warn("the change is not monotone over the grid; read the grid in phys_calibration.json")
    nc = decision["noise_check"]
    if nc["checked"]:
        (ctx.log if nc["within_base"] else ctx.warn)(
            f"sensor_noise at k = {nc['k']:g}: change {nc['corr_frobenius']:.4g}, "
            f"{'within' if nc['within_base'] else 'OVER'} the base's {b:.4g}")
    return {"tilt_max": t, "tilt_bound": decision["tilt_bound"], "base_corr_change": b,
            "tilt_corr_change": decision["tilt_corr_change"],
            "noise_corr_change": nc.get("corr_frobenius"), "noise_within_base": nc.get("within_base"),
            "sample_rows": n}
