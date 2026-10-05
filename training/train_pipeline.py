# -*- coding: utf-8 -*-
"""
training/train_pipeline.py
===========================
One training run, start to finish: `run(parsed)`.

This is `train_example_v16._main` (the only training pipeline the v16-line entry
points ever had - every other entry point wrapped it) with its five injection
points turned into flags. They used to be supplied by a derived entry point
assigning into v16's module namespace or passing a `v16.Overrides`:

    injection point           was                                   now
    normalization statistics  original rebound compute_global_...   --global_stats
    LR schedule               original rebound build_warmup_...     --lr_schedule
    model builder             optimal's build_model_optimal         always (it only drops a
                                                                    value that would raise)
    trainer class             one class per entry point             training.trainerg.TrainerG
    run name                  one suffix function per entry point   the profile's name_suffix

Every step in between - gates G1/G2/G9, G5, G6, G7, G8, the group sidecars, the
subsampling, the step-axis schedule - is v16's code, in v16's order.
"""

from __future__ import annotations

import datetime
import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from training.activation_patch import replace_relu_with_leakyrelu
from training.augmentation import PRESETS, AugmentationConfig
from training.augmentation_v16 import AugmentedPatchDatasetV16, build_active
from training.class_coverage import verify_class_coverage
from training.class_imbalance import write_imbalance_report
from training.config_presets import build_model, v15_config_overrides
from training.dataloader_config import apply_main_process_thread_limits, seed_everything
from training.dataloader_config_v16 import train_val_policies_v16
from training.failure_taxonomy import classify_exception
from training.fd_limit import is_fd_exhaustion, raise_open_file_limit
from training.gan import SpectralDiscriminator
from training.gates import (
    check_reconstruction_gradient, check_split_drift, enforce_split_drift,
    run_sensitivity_check_with_margin,
)
from training.grad_checkpoint import enable_gradient_checkpointing
from training.losses import build_criterion
from training.npy_data import (NpyDatasetV16, compute_global_channel_stats, discover_data,
                                setup_experiment_dir)
from training.npy_integrity import (
    DatasetStorageError, validate_dataset, write_dataset_failure_artifacts, write_integrity_report,
)
from training.numerical_stability import StabilityConfig
from training.optim_groups import build_param_groups
from training.reconstruction_head_v2 import MedMambaSSTRMLatentReconWrapperV2, MedMambaSSTRMRawReconWrapper
from training.run_naming_v16 import build_run_name as _base_run_name
from training.run_naming_v16 import dataset_slug, infer_modality, modality_warnings
from training.samplers import build_balanced_sampler, build_moderate_oversample_indices
from training.spectral_checkpoint import enable_spectral_gradient_checkpointing
from training.system_memory import (
    collect_system_memory_snapshot, log_system_memory, write_system_memory_report,
)
from training.torch_compile import enable_torch_compile
from training.train_preflight import (
    _check_manifest, _probe_in_channels, _run_leakage_check, print_preflight, run_loader_test,
)
from training.train_profiles import PROFILES
from training.trainerg import TrainerG, stop_knobs

_LAST_EXP_DIR = {"path": None}


# ============================================================================
# the former injection points
# ============================================================================

def build_run_name(args, timestamp: Optional[str] = None, modality: Optional[str] = None) -> str:
    """`run_naming_v16.build_run_name` + the profile's suffix + `_<run_tag>` -
    byte-identical to what each former entry point produced."""
    name = _base_run_name(args, timestamp, modality=modality) + PROFILES[args.profile].name_suffix
    return f"{name}_{args.run_tag}" if getattr(args, "run_tag", None) else name


def _constant_channel_stats(x_path: str, mean: float, std: float):
    n_channels = int(np.load(x_path, mmap_mode="r").shape[-1])
    return (np.full((n_channels,), mean, dtype=np.float32),
            np.full((n_channels,), std, dtype=np.float32))


# --global_stats. 'medmamba_fixed' makes global_zscore exactly torchvision's
# Normalize((.5,)*C, (.5,)*C) on [0, 1] inputs; 'identity' feeds the array unchanged.
GLOBAL_STATS = {
    "train_fit": compute_global_channel_stats,
    "medmamba_fixed": lambda x_path, sample_cap=5000, seed=0: _constant_channel_stats(x_path, 0.5, 0.5),
    "identity": lambda x_path, sample_cap=5000, seed=0: _constant_channel_stats(x_path, 0.0, 1.0),
}


def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: Optional[int] = None):
    """Linear warmup (default 3% of `total_steps`, floor 200) into cosine decay
    to 0, on the STEP axis (E-2)."""
    total_steps = max(1, int(total_steps))
    if warmup_steps is None:
        warmup_steps = max(200, int(round(0.03 * total_steps)))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler.warmup_steps = warmup_steps
    scheduler.total_steps = total_steps
    return scheduler


def build_constant_scheduler(optimizer, total_steps: int, warmup_steps: Optional[int] = None):
    """The LR held at its initial value - the MedMamba reference's
    `--scheduler none`. A LambdaLR returning 1.0 rather than None, because the
    pipeline logs `.warmup_steps` / `.total_steps` and the trainer steps it."""
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scheduler.warmup_steps = 0
    scheduler.total_steps = max(1, int(total_steps))
    return scheduler


SCHEDULERS = {"warmup_cosine": build_warmup_cosine_scheduler, "constant": build_constant_scheduler}

# Which architectures accept a head dropout at all: 'split'/'fullchannel' use
# medmamba_ss_trm.ClassificationHead, whose dropout is hardcoded at 0.1 in a frozen file.
_CLASSIFIER_DROPOUT_ARCHS = ("efficient", "recursive")


def build_model_guarded(architecture, *a, classifier_dropout=None, **kw):
    """`config_presets.build_model` with an inapplicable `--classifier_dropout`
    dropped - and said so - instead of raising before epoch 1 for a value the
    profile, not the user, supplied."""
    if classifier_dropout is not None and architecture not in _CLASSIFIER_DROPOUT_ARCHS:
        print(f"[classifier_dropout] {classifier_dropout} IGNORED on --architecture "
              f"{architecture!r}: its ClassificationHead hardcodes 0.1 (medmamba_ss_trm.py, frozen). "
              f"The head runs at 0.1.", flush=True)
        classifier_dropout = None
    return build_model(architecture, *a, classifier_dropout=classifier_dropout, **kw)


# ============================================================================
# A-4/A-5/C-1 - group sidecars
# ============================================================================

def _load_group_sidecars(x_val_path: str, sidecar_dir: Optional[str]):
    """Best-effort load of `scripts/derive_patch_groups.py`'s sidecars. Never
    raises - a missing sidecar just means A-4/A-5/C-1 are unavailable."""
    base = Path(sidecar_dir) if sidecar_dir else Path(x_val_path).parent
    out = {"groups": {}, "captures": {}, "images": {}, "capture_index": {}}
    for kind in ("groups", "captures", "images"):
        for split, suffix in (("train", "train"), ("validation", "val"), ("test", "test")):
            p = base / f"{kind}_{suffix}.npy"
            if p.is_file():
                out[kind][split] = np.load(p)
    ci_path = base / "capture_index.json"
    if ci_path.is_file():
        with open(ci_path) as f:
            out["capture_index"] = json.load(f)
    return out


def _subsample_indices_v16(labels: np.ndarray, frac: float, mode: str, num_classes: int,
                           seed: int, groups: Optional[np.ndarray] = None) -> np.ndarray:
    """`random`, per-class `stratified`, or A-5's `stratified_group`: the
    per-class fraction drawn WITHIN each patient, so a small subset still
    represents every patient in proportion."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if mode == "random":
        keep = max(1, int(round(n * frac)))
        return np.sort(rng.choice(n, size=keep, replace=False))

    if mode == "stratified_group" and groups is not None:
        picked = []
        for g in sorted(set(groups.tolist())):
            g_idx = np.flatnonzero(groups == g)
            for c in range(num_classes):
                idx = g_idx[labels[g_idx] == c]
                if idx.size == 0:
                    continue
                keep = max(1, int(round(idx.size * frac)))
                picked.append(rng.choice(idx, size=min(keep, idx.size), replace=False))
        return np.sort(np.concatenate(picked)) if picked else np.arange(n)

    picked = []
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        keep = max(1, int(round(idx.size * frac)))
        picked.append(rng.choice(idx, size=min(keep, idx.size), replace=False))
    return np.sort(np.concatenate(picked)) if picked else np.arange(n)


def _build_test_loader_v16(args, data_dir, storage_mode, normalization, global_mean, global_std,
                           policy, device, val_x_path):
    """Gate G6's loader, on `NpyDatasetV16` (the frozen `NpyDataset` asserts it
    only knows two normalization modes, which silently cost every v16 HSI run
    its G6). `return_norm_stats=False` because `evaluate_model` unpacks 2-tuples."""
    base = Path(data_dir)
    x_test, y_test = base / "X_test.npy", base / "y_test.npy"
    if not (x_test.is_file() and y_test.is_file()):
        print(f"WARNING: no X_test.npy/y_test.npy in {data_dir} - gate G6 cannot be satisfied.",
              flush=True)
        return None
    if Path(val_x_path).resolve() == x_test.resolve():
        print(f"WARNING: {data_dir} has no X_val.npy, so discover_data used X_test.npy as the "
              f"VALIDATION split. Skipping the test evaluation rather than reporting the "
              f"model-selection split as held-out.", flush=True)
        return None
    test_ds = NpyDatasetV16(str(x_test), str(y_test), storage_mode=storage_mode,
                            normalization=normalization, global_mean=global_mean,
                            global_std=global_std, return_norm_stats=False)
    print(f"[test-eval] {len(test_ds)} held-out test patches from {x_test}", flush=True)
    return DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                      **policy.dataloader_kwargs(device, seed=args.seed))


class _G7Shim:
    """`TrainerG_v12._forward_with_recon` is an instance method reading
    `self._wl` / `self.sensor_range`; G7 runs before a trainer exists."""
    _wl = None
    sensor_range = None

    def _accepts_wavelengths_ds(self, base):
        return False

    def _deep_supervision_forward(self, base, x):
        return base.forward_deep_supervision(x)

    def _model_forward(self, model, x):
        return model(x)


# ============================================================================
# the run
# ============================================================================

def _run(parsed, notes):
    args, profile = parsed.args, parsed.profile
    trainer_cls = TrainerG
    if args.fast_loop == "on":
        from training.trainerg_v12_fast import make_fast
        trainer_cls = make_fast(trainer_cls)

    seed_state = seed_everything(args.seed, deterministic=args.deterministic)
    apply_main_process_thread_limits(args.cpu_threads)
    # headroom for DataLoader worker descriptors: the test split is the largest
    # loader and is built last (training/fd_limit.py).
    raise_open_file_limit()
    # before ANY Triton compilation: /tmp is a 3.1 GB tmpfs on the GPU host.
    from training.compile_cache import (
        ensure_compile_scratch_on_disk, warn_if_tmpdir_breaks_worker_ipc)
    warn_if_tmpdir_breaks_worker_ipc()
    if args.compile_model == "on":
        ensure_compile_scratch_on_disk()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            print("WARNING: --amp bf16 requested but this GPU does not support bfloat16 - "
                  "falling back to --amp off (fp32).", flush=True)
            args.amp = "off"

    for note in notes:
        print(f"[default] {note}", flush=True)
    for w in modality_warnings(args.data_dir):
        print(f"WARNING: {w}", flush=True)

    startup_snapshot = log_system_memory(label="startup")

    run_modality = infer_modality(args.data_dir)
    exp_dir = setup_experiment_dir(exp_name=build_run_name(args, modality=run_modality),
                                   resume_path=args.resume, append_timestamp=False)
    _LAST_EXP_DIR["path"] = exp_dir
    write_system_memory_report(str(exp_dir / "system_memory_startup.json"), label="startup")

    print(f"Discovering dataset files in {args.data_dir}...", flush=True)
    x_tr, y_tr, x_te, y_te, wavelengths, num_classes, class_names, n_train, n_val = \
        discover_data(args.data_dir)
    print(f"Dataset Loaded: {n_train} Train, {n_val} Val | Classes: {num_classes} | Device: {device}",
          flush=True)
    if ("hsi" if wavelengths is not None else "rgb") != run_modality:
        raise SystemExit(f"INTERNAL: modality mismatch for {args.data_dir}.")

    dataset_paths = {"X_train": x_tr, "y_train": y_tr, "X_val": x_te, "y_val": y_te}
    integrity_report = None
    try:
        _check_manifest(args, exp_dir)
        if not args.skip_dataset_validation:
            integrity_report = validate_dataset(dataset_paths, abort_on_failure=True,
                                                level=args.dataset_validation_level)
            write_integrity_report(integrity_report, str(exp_dir / "dataset_integrity_report.json"))
    except DatasetStorageError as e:
        print(f"DATASET_STORAGE_ERROR: {e}", flush=True)
        write_dataset_failure_artifacts(str(exp_dir), integrity_report, e,
                                        system_snapshot=collect_system_memory_snapshot())
        raise

    _run_leakage_check(args, exp_dir)

    y_train_arr = np.asarray(np.load(y_tr, mmap_mode="r"))
    y_val_arr = np.asarray(np.load(y_te, mmap_mode="r"))
    if args.check_class_coverage:
        verify_class_coverage({"train": y_train_arr, "validation": y_val_arr}, num_classes=num_classes,
                              class_names=class_names, fail_hard=not args.allow_missing_classes)
    write_imbalance_report(str(exp_dir / "class_imbalance_report.json"),
                           {"train": y_train_arr, "validation": y_val_arr},
                           num_classes=num_classes, class_names=class_names)

    # ---- A-1/A-2 normalization -------------------------------------------
    global_mean = global_std = None
    if args.normalization == "global_zscore":
        if args.global_stats != "train_fit":
            print(f"[normalization] global_zscore with FIXED statistics ({args.global_stats}), "
                  f"not fitted to the train split.", flush=True)
        global_mean, global_std = GLOBAL_STATS[args.global_stats](
            x_tr, sample_cap=args.norm_stats_sample_cap, seed=args.seed)

    return_norm_stats = args.recon_mode != "none"

    def _build_dataset(xp, yp):
        return NpyDatasetV16(xp, yp, storage_mode=args.dataset_storage, normalization=args.normalization,
                             global_mean=global_mean, global_std=global_std,
                             return_norm_stats=return_norm_stats)

    train_ds = _build_dataset(x_tr, y_tr)
    val_ds = _build_dataset(x_te, y_te)

    # ---- A-3/A-4/A-5 group sidecars + gate G8 ------------------------------
    sidecars = _load_group_sidecars(x_te, args.group_sidecar_dir)
    groups_val_full = sidecars["groups"].get("validation")
    captures_val_full = sidecars["captures"].get("validation")
    groups_test_full = sidecars["groups"].get("test")
    images_test_full = sidecars["images"].get("test")
    capture_index = sidecars["capture_index"]

    x_paths_g8 = {"train": x_tr, "validation": x_te}
    x_test_path = Path(args.data_dir) / "X_test.npy"
    if x_test_path.is_file() and x_test_path.resolve() != Path(x_te).resolve():
        x_paths_g8["test"] = str(x_test_path)
    groups_g8 = {k: v for k, v in {"validation": groups_val_full, "test": groups_test_full}.items()
                 if v is not None}
    drift_report = check_split_drift(
        x_paths_g8, args.normalization, global_mean=global_mean, global_std=global_std,
        groups=groups_g8 or None, n_patches=args.drift_check_patches, seed=args.seed,
        exp_dir=str(exp_dir))
    enforce_split_drift(drift_report, args.on_split_drift)
    print(f"[gate G8] split drift: {'PASS' if drift_report['passed'] else 'FAIL'} "
          f"-> {exp_dir}/split_drift_report.json", flush=True)

    # ---- validation subsample (A-5: patient-stratified when groups exist) --
    val_subsample_info = {"frac": args.val_subsample_frac, "mode": args.val_subsample_mode,
                          "kept": len(val_ds), "total": len(val_ds)}
    val_keep_idx = None
    if not 0.0 < args.val_subsample_frac <= 1.0:
        raise SystemExit(f"--val_subsample_frac must be in (0, 1], got {args.val_subsample_frac}")
    if args.val_subsample_frac < 1.0:
        mode = ("stratified_group" if (args.val_subsample_mode == "stratified" and groups_val_full is not None)
                else args.val_subsample_mode)
        val_keep_idx = _subsample_indices_v16(y_val_arr, args.val_subsample_frac, mode, num_classes,
                                              args.seed + 1, groups=groups_val_full)
        val_ds = Subset(val_ds, val_keep_idx.tolist())
        val_subsample_info.update(kept=len(val_keep_idx), mode=mode)
        print(f"[val-subsample] frac={args.val_subsample_frac} mode={mode} -> {len(val_keep_idx)} of "
              f"{val_subsample_info['total']} validation patches (seed={args.seed + 1})", flush=True)

    def _subset(arr):
        return arr[val_keep_idx] if (arr is not None and val_keep_idx is not None) else arr

    val_groups, val_captures = _subset(groups_val_full), _subset(captures_val_full)

    # ---- train sampler / subsample -----------------------------------------
    train_sampler = None
    shuffle = True
    if args.sampler == "balanced":
        train_sampler = build_balanced_sampler(y_train_arr, num_classes)
        shuffle = False
    elif args.sampler == "moderate_oversample":
        idx = build_moderate_oversample_indices(y_train_arr, num_classes, seed=args.seed,
                                                target_percentile=args.oversample_target_percentile)
        train_ds = Subset(train_ds, idx.tolist())

    if not 0.0 < args.train_subsample_frac <= 1.0:
        raise SystemExit(f"--train_subsample_frac must be in (0, 1], got {args.train_subsample_frac}")
    train_subsample_info = {"frac": args.train_subsample_frac, "mode": args.train_subsample_mode,
                            "kept": len(train_ds), "total": len(train_ds)}
    if args.train_subsample_frac < 1.0:
        if train_sampler is not None:
            full_n = train_sampler.num_samples
            train_sampler.num_samples = max(1, int(round(full_n * args.train_subsample_frac)))
            kept, total, resamples = train_sampler.num_samples, full_n, True
        elif args.train_subsample_mode == "per_epoch":
            total = len(train_ds)
            kept = max(1, int(round(total * args.train_subsample_frac)))
            gen = torch.Generator().manual_seed(args.seed)
            train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=False, num_samples=kept,
                                                           generator=gen)
            shuffle, resamples = False, True
        else:
            total = len(train_ds)
            kept = max(1, int(round(total * args.train_subsample_frac)))
            pick = np.random.default_rng(args.seed).permutation(total)[:kept]
            train_ds = Subset(train_ds, pick.tolist())
            resamples = False
        train_subsample_info.update(kept=kept, total=total, resamples_each_epoch=resamples)
        print(f"[train-subsample] frac={args.train_subsample_frac} mode={args.train_subsample_mode} -> "
              f"{kept} of {total} training samples per epoch (seed={args.seed})", flush=True)

    aug_cfg = (PRESETS[args.augment_preset] if args.augment_preset != "custom" else AugmentationConfig(
        spectral_noise_std=args.spectral_noise_std, spectral_scale_range=args.spectral_scale_range,
        spectral_offset_std=args.spectral_offset_std, band_dropout_prob=args.band_dropout_prob,
        band_dropout_max_frac=args.band_dropout_max_frac, spectral_mask_prob=args.spectral_mask_prob,
        spectral_mask_max_width_frac=args.spectral_mask_max_width_frac, flip_h_prob=args.flip_h_prob,
        flip_v_prob=args.flip_v_prob, rotate90_prob=args.rotate90_prob,
        crop_scale_max_frac=args.crop_scale_max_frac, seed=args.seed))
    augmented_train_ds = None
    if args.augment_preset != "none":
        # `build_active` drops switched-off steps; bit-identical, since a disabled
        # step consumes no RNG draws (training/augmentation_v16.py).
        train_ds = AugmentedPatchDatasetV16(train_ds, build_active(aug_cfg), seed=args.seed)  # E-4
        augmented_train_ds = train_ds

    train_policy, val_policy = train_val_policies_v16(args.loader_mode, args.num_workers,
                                                      args.prefetch_factor, args.persistent_workers,
                                                      args.pin_memory)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=train_sampler,
                              **train_policy.dataloader_kwargs(device, seed=args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **val_policy.dataloader_kwargs(device, seed=args.seed))

    modality = "hsi" if wavelengths is not None else "rgb"
    in_channels = int(_probe_in_channels(train_ds))
    system_snapshot = collect_system_memory_snapshot()
    print_preflight(args, train_policy, val_policy, dataset_paths, system_snapshot,
                        in_channels=in_channels, modality=modality)
    print(f"  normalization: {args.normalization}  recon_mode: {args.recon_mode}  "
          f"recon_out_activation: {args.recon_out_activation}", flush=True)

    if args.loader_test:
        run_loader_test(args, train_loader, train_policy)
        return

    # ---- model ---------------------------------------------------------------
    cfg_overrides = dict(v15_config_overrides(args.architecture))
    cfg_overrides.update({
        "spectral_token_fusion": args.spectral_token_fusion, "spectral_pe_gain": args.spectral_pe_gain,
        "spectral_value_init_std": args.spectral_value_init_std, "spectral_ctx_norm": args.spectral_ctx_norm,
        "classifier_init": args.classifier_init, "wavelength_encoding_scale": args.wavelength_scale,
        "spectral_chunk_size": args.spectral_chunk_size,
    })
    if args.patch_size is not None:
        cfg_overrides["patch_size"] = args.patch_size
    trm_kwargs = None
    if args.architecture == "recursive":
        trm_kwargs = dict(
            trm_dim=args.trm_dim, trm_core_layers=args.trm_core_layers, trm_n_latent=args.trm_n_latent,
            trm_n_improve=args.trm_n_improve, trm_deep_supervision_steps=args.trm_deep_supervision_steps,
            trm_mixer=args.trm_mixer, trm_ema_rate=args.trm_ema_rate, trm_act_halting=args.trm_halting,
            trm_halt_threshold=args.trm_halt_threshold, trm_halt_exploration_prob=args.trm_halt_exploration_prob,
            trm_spatial_pe_gain=args.spatial_pe_gain, trm_checkpoint_core=args.trm_checkpoint_core,
            trm_mixer_channel_mlp=args.trm_mixer_channel_mlp,
            trm_drop_path=(args.trm_drop_path if args.trm_drop_path > 0 else (args.drop_path_rate or 0.0)),
            trm_dropout=args.trm_dropout,
        )
    base_model = build_model_guarded(args.architecture, modality, num_classes,
                                     drop_path_rate=args.drop_path_rate,
                                     fusion_type=(args.fusion_type if args.architecture == "efficient" else None),
                                     classifier_dropout=args.classifier_dropout, trm_kwargs=trm_kwargs,
                                     cfg_overrides=cfg_overrides)
    n_params = sum(p_.numel() for p_ in base_model.parameters())
    print(f"[architecture] {args.architecture!r} ({modality}): {n_params/1e6:.3f}M parameters", flush=True)

    if args.activation == "leakyrelu":
        replace_relu_with_leakyrelu(base_model, negative_slope=args.leaky_slope)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(base_model, strict=True)
    if (args.spectral_checkpointing == "on" or (
            args.spectral_checkpointing == "auto" and in_channels >= args.spectral_checkpoint_min_channels)):
        enable_spectral_gradient_checkpointing(base_model, strict=True)
    if args.compile_model == "on":
        # `enable_torch_compile` swallows every failure; say which path is running.
        summary = enable_torch_compile(base_model, mode=args.compile_mode)
        print(f"[compile] {summary}", flush=True)
        if "FAILED" in summary or "unavailable" in summary:
            print("[compile] WARNING: running UNCOMPILED - expect ~1.64x the step time "
                  "measured for this configuration.", flush=True)

    if args.recon_mode == "latent":
        model = MedMambaSSTRMLatentReconWrapperV2(base_model, in_channels=in_channels,
                                              out_activation=args.recon_out_activation,
                                              spectral_dropout=args.spectral_dropout)
    elif args.recon_mode == "raw_input":
        model = MedMambaSSTRMRawReconWrapper(base_model, in_channels=in_channels)
    else:
        model = base_model

    use_gan = (args.use_gan or args.lambda_gan > 0) and args.recon_mode != "none"
    discriminator = disc_optimizer = None
    if use_gan:
        discriminator = SpectralDiscriminator(in_channels=in_channels)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr, weight_decay=0.0)
        if args.lambda_gan == 0.0:
            args.lambda_gan = 0.05

    criterion = build_criterion(args.loss, y_train=y_train_arr, num_classes=num_classes,
                                focal_gamma=args.focal_gamma, weight_method=args.weight_method,
                                device=device, class_weight_power=args.class_weight_power)
    if args.weight_decay_groups:
        optimizer = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # E-2 - step-axis schedule over the steps this run will actually take.
    steps_per_epoch = max(1, -(-train_subsample_info["kept"] // args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    scheduler = SCHEDULERS[args.lr_schedule](optimizer, total_steps, warmup_steps=args.warmup_steps)
    print(f"[scheduler] {args.lr_schedule}: total_steps={total_steps} "
          f"warmup_steps={scheduler.warmup_steps} interval={args.scheduler_interval}", flush=True)

    # ---- gates G1/G2/G9 and G7 ----------------------------------------------
    sensitivity_result = None
    if args.sensitivity_check == "on":
        x0, _y0 = next(iter(train_loader))[:2]
        sensitivity_result = run_sensitivity_check_with_margin(
            model.to(device), x0.to(device), min_logit_std=args.min_logit_std,
            min_stem_sensitivity=args.min_stem_sensitivity)
        if not sensitivity_result.get("passed"):
            fail_dir = exp_dir / "numerical_failure"
            fail_dir.mkdir(parents=True, exist_ok=True)
            with open(fail_dir / "sensitivity_failure.json", "w") as f:
                json.dump(sensitivity_result, f, indent=2, default=str)
            raise SystemExit(f"REPRESENTATION_COLLAPSE: {sensitivity_result.get('reason')}")
        print("[gates G1/G2/G9] margins: " +
              ", ".join(f"{k}={v['margin']:.2f}x" for k, v in sensitivity_result.get("margins", {}).items()),
              flush=True)

    g7_result = None
    if (args.g7_check == "on" or (args.g7_check == "auto" and args.architecture == "recursive"
                                  and args.recon_mode == "latent")):
        from training.gan import SAMLoss
        x0, _y0 = next(iter(train_loader))[:2]
        g7_shim = _G7Shim()
        g7_result = check_reconstruction_gradient(
            model.to(device), lambda m, x: trainer_cls._forward_with_recon(g7_shim, m, x),
            x0.to(device), SAMLoss(), lambda_mse=args.lambda_mse, lambda_sam=args.lambda_sam)
        with open(exp_dir / "gate_g7_report.json", "w") as f:
            json.dump(g7_result, f, indent=2, default=str)
        if not g7_result.get("passed"):
            raise SystemExit(f"RECONSTRUCTION_DETACHED (gate G7): {g7_result.get('reason')}")
        print(f"[gate G7] PASS - reconstruction gradient reaches {g7_result['n_encoder_params_checked']} "
              f"recursive-core parameter(s)", flush=True)

    config = {
        "cli_args": vars(args), "classes": num_classes, "device": device, "class_names": class_names,
        "in_channels": in_channels, "architecture": args.architecture, "modality": modality,
        "run_name": exp_dir.name, "dataset_slug": dataset_slug(args.data_dir),
        "backbone_num_params": int(n_params),
        "dataloader_config": {"train": train_policy.as_dict(), "validation": val_policy.as_dict()},
        "amp_mode": args.amp, "torch_version": torch.__version__,
        "entry_point": "train.py", "profile": profile, "trainer": trainer_cls.__name__,
        "train_subsample": train_subsample_info, "val_subsample": val_subsample_info,
        "seeding": seed_state, "sensitivity_check": sensitivity_result, "gate_g7": g7_result,
        "split_drift": {"passed": drift_report["passed"]},
        "scheduler": {"interval": args.scheduler_interval, "total_steps": total_steps,
                      "warmup_steps": int(scheduler.warmup_steps), "schedule": args.lr_schedule},
        "system_snapshot_startup": startup_snapshot,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4, default=str)

    stability_cfg = StabilityConfig(
        max_gradient_norm=args.max_gradient_norm, max_gradient_skip_ratio=args.max_gradient_skip_ratio,
        max_consecutive_bad_batches=args.max_consecutive_bad_batches,
        max_consecutive_unhealthy_epochs=args.max_consecutive_unhealthy_epochs,
        debug_numerics=args.debug_numerics)

    trainer = trainer_cls(
        model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
        scheduler=scheduler, device=device, exp_dir=exp_dir, class_names=class_names,
        lambda_mse=(args.lambda_mse if args.recon_mode != "none" else 0.0),
        lambda_sam=(args.lambda_sam if args.recon_mode != "none" else 0.0),
        lambda_gan=args.lambda_gan, discriminator=discriminator, disc_optimizer=disc_optimizer,
        config=config, wavelengths=wavelengths, checkpoint_metric=args.checkpoint_metric,
        class_collapse_streak=args.class_collapse_streak, criterion=criterion,
        early_stop_patience=args.early_stop_patience, grad_clip_norm=args.max_gradient_norm,
        stability_cfg=stability_cfg, amp_mode=args.amp, debug_numerics=args.debug_numerics,
        ema_rate=(args.trm_ema_rate if args.architecture == "recursive" else None),
        on_class_collapse=args.on_class_collapse, use_wavelengths=args.use_wavelengths,
        sensor_range=(tuple(args.sensor_range) if args.sensor_range else None),
        keep_last_n=args.keep_last_n, eval_artifact_stride=args.eval_artifact_stride,
        val_groups=val_groups, val_captures=val_captures, capture_index=capture_index,
        min_patient_recall=args.min_patient_recall, val_divergence_patience=args.val_divergence_patience,
        scheduler_interval=args.scheduler_interval,
        **stop_knobs(args),
    )
    trainer.dataloader_config = config["dataloader_config"]
    if augmented_train_ds is not None:
        trainer.augmented_train_dataset = augmented_train_ds

    log_system_memory(trainer.logger, label="pre-training")

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        start_epoch = trainer.load_checkpoint(args.resume) + 1

    stop_reason = trainer.fit(max_epochs=args.epochs, start_epoch=start_epoch)
    stop_str = str(getattr(stop_reason, "value", stop_reason)) if stop_reason is not None else ""

    gate_results = {"G7": (g7_result or {}).get("passed"), "G8": drift_report["passed"],
                    "G1_G2_G9": (sensitivity_result or {}).get("passed")}
    if args.verify_best_checkpoint:
        try:
            gate_results["G5"] = trainer.verify_best_checkpoint_reproduces()
        except Exception as e:
            print(f"WARNING: gate G5 check failed to run: {type(e).__name__}: {e}", flush=True)

    if args.eval_test != "off":

        def _run_test_eval(policy) -> bool:
            loader = _build_test_loader_v16(
                args, data_dir=args.data_dir, storage_mode=args.dataset_storage,
                normalization=args.normalization, global_mean=global_mean,
                global_std=global_std, policy=policy, device=device, val_x_path=x_te)
            groups = images_test_full if modality == "rgb" else groups_test_full
            return bool(trainer.evaluate_test_split(
                loader, num_classes=num_classes, test_groups=groups,
                eval_group_aggregation=args.eval_group_aggregation) is not None)

        try:
            gate_results["G6"] = _run_test_eval(val_policy)
        except Exception as e:
            # One single-process retry on descriptor exhaustion: a run that trained
            # for hours should not forfeit its held-out number to an FD ceiling.
            if is_fd_exhaustion(e):
                print(f"WARNING: test-split evaluation ran out of file descriptors with "
                      f"{val_policy.num_workers} worker(s). Retrying single-process.", flush=True)
                try:
                    gate_results["G6"] = _run_test_eval(replace(
                        val_policy, num_workers=0, persistent_workers=False, prefetch_factor=None))
                    print("[test-eval] single-process retry succeeded.", flush=True)
                    e = None
                except Exception as e2:
                    e = e2
            if e is not None:
                print(f"WARNING: test-split evaluation failed: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                with open(exp_dir / "gate_g6_failure.json", "w") as f:
                    json.dump({"exception": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()}, f, indent=2, default=str)
                gate_results["G6"] = False

    with open(exp_dir / "gates.json", "w") as f:
        json.dump(gate_results, f, indent=2, default=str)

    if stop_str == "TRAINING_ABORTED_NUMERICAL_INSTABILITY":
        raise SystemExit(f"TRAINING_ABORTED_NUMERICAL_INSTABILITY: see {exp_dir}/numerical_failure/")
    if stop_str == "TRAINING_ABORTED_REPRESENTATION_COLLAPSE":
        raise SystemExit(f"TRAINING_ABORTED_REPRESENTATION_COLLAPSE: see {exp_dir}/collapse_failure/")
    return exp_dir


def run(parsed, notes=()):
    """`_run`, writing `failure_class.json` into the run directory on any crash
    after the directory exists."""
    try:
        return _run(parsed, notes)
    except SystemExit:
        raise
    except BaseException as e:
        fc = classify_exception(e)
        try:
            exp_dir = _LAST_EXP_DIR.get("path")
            if exp_dir is not None:
                with open(Path(exp_dir) / "failure_class.json", "w") as f:
                    json.dump({"failure_class": fc.value, "exception": f"{type(e).__name__}: {e}",
                               "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                              f, indent=2, default=str)
        except Exception:
            pass
        raise
