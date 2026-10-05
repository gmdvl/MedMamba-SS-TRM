# -*- coding: utf-8 -*-
"""
training/train_preflight.py
============================
What `train.py` runs and prints before it builds a model, copied verbatim from
`archive/train_example_v15.py` when v6, v7 and v15 were archived (2026-09-29):

    print_preflight      the "MedMamba-SS-TRM Stability Preflight" block        (v15 lines 710-781)
    run_loader_test      --loader_test: read a few batches, then exit     (788-801)
    _probe_in_channels   the channel count, without an augmentation draw  (877-897)
    _run_leakage_check   Phase 7 patient/patch leakage check              (997-1030)
    _check_manifest      --manifest_check against dataset_manifest.json   (1033-1049)

The leading underscores are v15's, kept so each copy matches its original exactly
(`tests/test_live_copies_match_archive.py`).
"""

import errno
import json
import time
from pathlib import Path

from training.dataset_integrity import DatasetLeakageError, run_full_integrity_check
from training.npy_atomic import MANIFEST_NAME, verify_against_manifest
from training.npy_integrity import DatasetStorageError, write_dataset_failure_artifacts
from training.system_memory import collect_system_memory_snapshot, format_system_memory_snapshot


def print_preflight(args, train_policy, val_policy, dataset_paths, system_snapshot, in_channels=None,
                     modality=None):
    lines = ["===== MedMamba-SS-TRM Stability Preflight =====", "Dataset:"]
    for k, v in dataset_paths.items():
        lines.append(f"  {k}: {v}")
    if in_channels is not None:
        lines.append(f"  channels: {in_channels}  modality: {modality}")
    lines.append(f"  storage_mode: {args.dataset_storage}")
    lines.append("")
    lines.append("DataLoader (train):")
    for k, v in train_policy.as_dict().items():
        lines.append(f"  {k}: {v}")
    lines.append("DataLoader (validation):")
    for k, v in val_policy.as_dict().items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("System:")
    ram, shm, gpu = system_snapshot["ram"], system_snapshot["shm"], system_snapshot["gpu"]
    lines.append(f"  RAM available: {ram.get('available_mb', 'n/a')} MB")
    lines.append(f"  /dev/shm available: {shm.get('free_mb', 'n/a') if shm.get('available') else 'n/a'} MB")
    if gpu.get("available"):
        lines.append(f"  GPU: {gpu.get('device_name')}  VRAM total: {gpu.get('total_mb')} MB  "
                      f"reserved: {gpu.get('reserved_mb')} MB")
    else:
        lines.append("  GPU: not available (CPU run)")
    lines.append("")
    lines.append("Numerical:")
    lines.append(f"  AMP: {args.amp}")
    lines.append(f"  gradient clipping: {args.max_gradient_norm}")
    lines.append(f"  max_gradient_skip_ratio: {args.max_gradient_skip_ratio}")
    lines.append(f"  max_consecutive_bad_batches: {args.max_consecutive_bad_batches}")
    lines.append(f"  SAM: {'ON' if args.lambda_sam > 0 and args.recon_mode != 'none' else 'off'}")
    lines.append(f"  GAN: {'ON' if (args.use_gan or args.lambda_gan > 0) and args.recon_mode != 'none' else 'off'}")
    lines.append(f"  gradient_checkpointing: {args.gradient_checkpointing}")
    lines.append(f"  torch.compile: {args.compile_model}"
                  + (f" (mode={args.compile_mode})" if args.compile_mode else "")
                  + ("   <-- ~2.0x slower than --compile on, on the recursive architecture"
                     if args.compile_model == "off" and args.architecture == "recursive" else ""))
    lines.append("")
    lines.append("Representation (v15 R1):")
    lines.append(f"  spectral_token_fusion: {args.spectral_token_fusion}"
                  + ("   <-- PRE-v15 DEFECT: the input signal is ~90x smaller than the "
                     "constant it is added to" if args.spectral_token_fusion == "add" else ""))
    lines.append(f"  spectral_pe_gain: {args.spectral_pe_gain}   "
                  f"spectral_value_init_std: {args.spectral_value_init_std}")
    lines.append(f"  spectral_ctx_norm: {args.spectral_ctx_norm}   "
                  f"classifier_init: {args.classifier_init}")
    if args.architecture == "recursive":
        lines.append(f"  spatial_pe_gain: {args.spatial_pe_gain}")
    lines.append(f"  wavelength_encoding_scale: "
                  f"{args.wavelength_scale if args.wavelength_scale is not None else 'band count (R1.3)'}")
    lines.append(f"  use_wavelengths: {args.use_wavelengths}   sensor_range: {args.sensor_range}")
    lines.append(f"  weight_decay_groups: {args.weight_decay_groups}  "
                  f"(weight_decay={args.weight_decay})")
    lines.append("")
    lines.append("Data / budget (v15 R2-R3, R6):")
    lines.append(f"  normalization: {args.normalization}")
    lines.append(f"  train_subsample: {args.train_subsample_frac} ({args.train_subsample_mode})   "
                  f"val_subsample: {args.val_subsample_frac} ({args.val_subsample_mode})")
    lines.append(f"  spectral_chunk_size: {args.spectral_chunk_size}   "
                  f"spectral_checkpointing: {args.spectral_checkpointing}")
    lines.append(f"  seed: {args.seed}   deterministic: {args.deterministic}")
    lines.append("")
    lines.append("Gates:")
    lines.append(f"  G1/G2 sensitivity check: {args.sensitivity_check} "
                  f"(stem >= {args.min_stem_sensitivity}, logit std >= {args.min_logit_std})")
    lines.append(f"  G3 on_class_collapse: {args.on_class_collapse} "
                  f"(streak {args.class_collapse_streak})")
    lines.append(f"  G5 verify_best_checkpoint: {args.verify_best_checkpoint}")
    lines.append(f"  G6 eval_test: {args.eval_test}")
    lines.append("==========================================")
    print("\n".join(lines), flush=True)


def run_loader_test(args, train_loader, train_policy) -> None:
    print(f"[loader-test] policy={train_policy.as_dict()}", flush=True)
    t0 = time.time()
    n_read = 0
    for i, (x, y) in enumerate(train_loader):
        if i >= args.loader_test_batches:
            break
        print(f"  batch {i}: x.shape={tuple(x.shape)} x.dtype={x.dtype} y.shape={tuple(y.shape)}", flush=True)
        n_read += 1
    elapsed = time.time() - t0
    snapshot = collect_system_memory_snapshot()
    print(f"[loader-test] read {n_read} batch(es) in {elapsed:.2f}s", flush=True)
    print(format_system_memory_snapshot(snapshot), flush=True)
    print("[loader-test] PASSED (no worker crash) - exiting without building the GPU model.", flush=True)


def _probe_in_channels(dataset) -> int:
    """v15 R6.6 - the channel count, read WITHOUT consuming an augmentation
    RNG draw.

    v14 did `train_ds[0][0].shape[0]` after wrapping the dataset in
    `AugmentedPatchDataset`, so sample 0 was augmented once, off-loop, purely
    to learn its channel count - which (before R6.3) advanced the single shared
    augmentation RNG that every worker then inherited. Unwrap to the underlying
    dataset first; the channel count is the same and nothing is disturbed.
    """
    seen = set()
    while id(dataset) not in seen:
        seen.add(id(dataset))
        inner = getattr(dataset, "base_dataset", None) or getattr(dataset, "dataset", None)
        if inner is None:
            break
        dataset = inner
    X = getattr(dataset, "X", None)
    if X is not None and getattr(X, "ndim", 0) == 4:
        return int(X.shape[-1])          # NpyDataset stores [N,H,W,C]
    return int(dataset[0][0].shape[0])


def _run_leakage_check(args, exp_dir):
    """Phase 7 leakage check. Runs AFTER Phase-4 structural validation, so a
    truncated/corrupt file is already a fatal DATASET_STORAGE_ERROR by now.
    Only a genuinely-missing *optional prerequisite* (no manifest, module
    import failure) is downgraded to a warning - a read/storage error is
    re-raised, never swallowed as 'continuing'."""
    if not args.check_leakage:
        return
    manifest_path = None
    for candidate in (Path(args.data_dir) / "_progress.json",
                       Path(args.data_dir).parent / "_progress.json"):
        if candidate.is_file():
            manifest_path = str(candidate)
            break
    try:
        run_full_integrity_check(out_dir=str(exp_dir), manifest_path=manifest_path,
                                  npy_data_dir=args.data_dir, abort_on_leakage=args.abort_on_leakage)
    except DatasetLeakageError:
        raise
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARNING: leakage check skipped (optional prerequisite missing: {e!r})", flush=True)
    except OSError as e:
        # A physically-unreadable .npy (OSError(5)/EIO) surfacing here instead
        # of in the structural gate: intermittent bad-sector reads. Convert to
        # the same typed, classified failure the dataset gate produces.
        eio = getattr(e, "errno", None) == errno.EIO or "input/output error" in str(e).lower()
        msg = (f"PHYSICALLY UNREADABLE (OSError(5)/EIO) during leakage check: {e}" if eio
               else f"read error during leakage check: {e}")
        exc = DatasetStorageError(f"DATASET_STORAGE_ERROR: {msg}")
        write_dataset_failure_artifacts(str(exp_dir), None, exc,
                                         system_snapshot=collect_system_memory_snapshot())
        raise exc from e


def _check_manifest(args, exp_dir):
    """Workstream 6 - cross-check <data_dir>/dataset_manifest.json if present."""
    if args.manifest_check == "off":
        return
    manifest_exists = (Path(args.data_dir) / MANIFEST_NAME).is_file()
    if not manifest_exists:
        if args.manifest_check == "strict":
            raise DatasetStorageError(
                f"DATASET_STORAGE_ERROR: --manifest_check strict but no {MANIFEST_NAME} in {args.data_dir}")
        print(f"[dataset-manifest] no {MANIFEST_NAME} (dataset predates manifest support) - "
              f"structural validation only.", flush=True)
        return
    problems = verify_against_manifest(args.data_dir)
    if problems:
        raise DatasetStorageError(
            "DATASET_STORAGE_ERROR: dataset does not match its manifest:\n  " + "\n  ".join(problems))
    print(f"[dataset-manifest] all files match {MANIFEST_NAME}.", flush=True)
