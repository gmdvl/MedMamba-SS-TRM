#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/eval_test_split.py
=================================
Run gate G6 (held-out test-split evaluation) for a finished
`train_example_v16.py` run, from its saved `config.json` + `best_model.pt`.

Why this exists
---------------
`train_example_v16._main` delegates the test evaluation to the FROZEN
`train_example_v15._build_test_loader`, which builds a FROZEN
`train_example_v6.NpyDataset`. That class asserts

    normalization in ("per_sample_minmax", "global_zscore")

(`train_example_v6.py:42`) - it predates `per_patch_zscore`, which v16 made
the DEFAULT for HSI (A-2). So every v16 HSI run raises a bare `AssertionError`
inside the `try:` around the test evaluation, which is swallowed into
`WARNING: test-split evaluation failed: AssertionError:` and recorded as
`gates.json: {"G6": false}`. The training run itself is unaffected - only the
test evaluation is lost, and it is lost silently apart from that one line.

Gate G6 is therefore unreachable for any v16 HSI run under v16's own defaults.
`20260905_034435` (PAD, `per_sample_minmax`) passes G6; `20260904_230812`
(HSI, `per_patch_zscore`) does not, for this reason alone.

This script rebuilds the loader with `training.npy_data.NpyDatasetV16`,
which routes through `training/normalization.py` and handles all three modes,
then calls the same `TrainerG_v12.evaluate_test_split` the entry point would
have, so the report is identical in shape and semantics to the PAD run's.

Nothing frozen is imported for mutation and nothing is edited: this is a
read-only consumer of a finished run directory.

Usage
-----
    python scripts/eval_test_split.py --run_dir experiments/<run>
    python scripts/eval_test_split.py --run_dir experiments/<run> --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.npy_data import NpyDatasetV16
from training.config_presets import build_model, v15_config_overrides
from training.trainerg_v12 import TrainerG_v12
from training.reconstruction_head_v2 import MedMambaSSTRMLatentReconWrapperV2
from training.dataloader_config_v16 import train_val_policies_v16
from training.numerical_stability import StabilityConfig


def _load_cfg(run_dir: Path) -> dict:
    with open(run_dir / "config.json") as f:
        return json.load(f)


def _resolve_data_dir(cfg: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(cfg["cli_args"]["data_dir"])


def _rebuild_model(cfg: dict):
    """Mirror `train_example_v16._main`'s model construction from config.json."""
    a = cfg["cli_args"]
    arch, modality = cfg["architecture"], cfg["modality"]
    num_classes, in_channels = cfg["classes"], cfg["in_channels"]

    cfg_overrides = dict(v15_config_overrides(arch))
    cfg_overrides.update({
        "spectral_token_fusion": a["spectral_token_fusion"], "spectral_pe_gain": a["spectral_pe_gain"],
        "spectral_value_init_std": a["spectral_value_init_std"], "spectral_ctx_norm": a["spectral_ctx_norm"],
        "classifier_init": a["classifier_init"], "wavelength_encoding_scale": a["wavelength_scale"],
        "spectral_chunk_size": a["spectral_chunk_size"],
    })
    trm_kwargs = None
    if arch == "recursive":
        trm_kwargs = dict(
            trm_dim=a["trm_dim"], trm_core_layers=a["trm_core_layers"], trm_n_latent=a["trm_n_latent"],
            trm_n_improve=a["trm_n_improve"], trm_deep_supervision_steps=a["trm_deep_supervision_steps"],
            trm_mixer=a["trm_mixer"], trm_ema_rate=a["trm_ema_rate"], trm_act_halting=a["trm_halting"],
            trm_halt_threshold=a["trm_halt_threshold"], trm_halt_exploration_prob=a["trm_halt_exploration_prob"],
            trm_spatial_pe_gain=a["spatial_pe_gain"], trm_checkpoint_core=a["trm_checkpoint_core"],
            trm_mixer_channel_mlp=a["trm_mixer_channel_mlp"],
            trm_drop_path=(a["trm_drop_path"] if a["trm_drop_path"] > 0 else (a["drop_path_rate"] or 0.0)),
            trm_dropout=a["trm_dropout"],
        )
    base = build_model(arch, modality, num_classes, drop_path_rate=a["drop_path_rate"],
                       fusion_type=(a["fusion_type"] if arch == "efficient" else None),
                       classifier_dropout=a["classifier_dropout"], trm_kwargs=trm_kwargs,
                       cfg_overrides=cfg_overrides)

    if a["recon_mode"] == "latent":
        model = MedMambaSSTRMLatentReconWrapperV2(base, in_channels=in_channels,
                                              out_activation=a["recon_out_activation"],
                                              spectral_dropout=a["spectral_dropout"])
    elif a["recon_mode"] == "raw_input":
        from training.reconstruction_head_v2 import MedMambaSSTRMRawReconWrapper
        model = MedMambaSSTRMRawReconWrapper(base, in_channels=in_channels)
    else:
        model = base
    return model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run gate G6 for a finished train_example_v16.py run.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default=None, help="override config.json's data_dir (e.g. if it moved)")
    p.add_argument("--checkpoint", default=None, help="default: <run_dir>/best_model.pt")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_group_aggregation", default=None,
                   help="default: the run's own --eval_group_aggregation")
    p.add_argument("--dry_run", action="store_true", help="check everything, load nothing heavy, then stop")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cfg = _load_cfg(run_dir)
    a = cfg["cli_args"]
    data_dir = _resolve_data_dir(cfg, args.data_dir)
    batch_size = args.batch_size or a["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_test, y_test = data_dir / "X_test.npy", data_dir / "y_test.npy"
    x_val = data_dir / "X_val.npy"
    print(f"[run]      {run_dir.name}")
    print(f"[data]     {data_dir}")
    print(f"[norm]     {a['normalization']}   recon_mode={a['recon_mode']}   amp={a['amp']}")
    print(f"[device]   {device}   batch_size={batch_size}")

    if not (x_test.is_file() and y_test.is_file()):
        print(f"ERROR: no X_test.npy/y_test.npy under {data_dir} - gate G6 cannot be satisfied.")
        return 2
    if x_val.is_file() and x_val.resolve() == x_test.resolve():
        print("ERROR: X_val.npy IS X_test.npy - refusing to report the model-selection split as held-out.")
        return 2

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "best_model.pt")
    if not ckpt_path.is_file():
        print(f"ERROR: {ckpt_path} does not exist.")
        return 2

    global_mean = global_std = None
    if a["normalization"] == "global_zscore":
        from training.npy_data import compute_global_channel_stats
        gm_path = data_dir / "X_train.npy"
        global_mean, global_std = compute_global_channel_stats(
            str(gm_path), sample_cap=a["norm_stats_sample_cap"], seed=a["seed"])

    # Group sidecars, so C-1's image-level aggregation works when available.
    agg = args.eval_group_aggregation or a.get("eval_group_aggregation", "none")
    test_groups = None
    if agg != "none":
        base = Path(a["group_sidecar_dir"]) if a.get("group_sidecar_dir") else data_dir
        fname = "images_test.npy" if cfg["modality"] == "rgb" else "groups_test.npy"
        if (base / fname).is_file():
            test_groups = np.load(base / fname)
            print(f"[groups]   {fname}: {len(test_groups)} ids, {len(set(test_groups.tolist()))} groups")
        else:
            print(f"[groups]   {base / fname} not found - image-level aggregation unavailable")
            agg = "none"

    n_test = int(np.load(y_test, mmap_mode="r").shape[0])
    print(f"[test]     {n_test} held-out patches   aggregation={agg}")

    if args.dry_run:
        print("[dry-run]  preflight OK - stopping before model construction.")
        return 0

    # `return_norm_stats=False`: `training.evaluator.evaluate_model` does
    # `sample_x, _ = next(iter(loader))`, which cannot unpack a 3-tuple.
    test_ds = NpyDatasetV16(str(x_test), str(y_test), storage_mode=a["dataset_storage"],
                            normalization=a["normalization"],
                            global_mean=global_mean, global_std=global_std,
                            return_norm_stats=False)
    _, val_policy = train_val_policies_v16(a["loader_mode"], a["num_workers"], a["prefetch_factor"],
                                           a["persistent_workers"], a["pin_memory"])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             **val_policy.dataloader_kwargs(device, seed=a["seed"]))

    model = _rebuild_model(cfg)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[model]    rebuilt: {n_params/1e6:.3f}M params "
          f"(config.json recorded {cfg['backbone_num_params']/1e6:.3f}M for the backbone)")

    wavelengths = None
    wl_path = data_dir / "wavelengths.npy"
    if a["use_wavelengths"] and wl_path.is_file():
        wavelengths = np.load(wl_path)
        print(f"[wl]       {len(wavelengths)} band centres, "
              f"{wavelengths.min():.1f}-{wavelengths.max():.1f} nm")

    trainer = TrainerG_v12(
        model=model, train_loader=None, val_loader=test_loader, optimizer=None, scheduler=None,
        device=device, exp_dir=run_dir, class_names=cfg["class_names"],
        config=cfg, wavelengths=wavelengths, checkpoint_metric=a["checkpoint_metric"],
        amp_mode=a["amp"], stability_cfg=StabilityConfig(),
        sensor_range=(tuple(a["sensor_range"]) if a.get("sensor_range") else None),
        use_wavelengths=a["use_wavelengths"],
        ema_rate=(a["trm_ema_rate"] if cfg["architecture"] == "recursive" else None),
        scheduler_interval=a.get("scheduler_interval", "step"),
    )
    trainer.best_epoch = json.load(open(run_dir / "experiment_report.json")) \
        .get("completion_status", {}).get("best_epoch")

    report = trainer.evaluate_test_split(test_loader, checkpoint_path=str(ckpt_path),
                                         num_classes=cfg["classes"],
                                         test_groups=test_groups, eval_group_aggregation=agg)
    if report is None:
        print("RESULT: evaluate_test_split returned None - gate G6 NOT satisfied.")
        return 1

    m = report["sklearn_metrics"]
    print()
    print("=" * 62)
    print(f"  TEST accuracy           {m['accuracy']:.4f}")
    print(f"  TEST balanced accuracy  {m['balanced_accuracy']:.4f}")
    print(f"  TEST macro F1           {m['f1_macro']:.4f}")
    print(f"  per-class F1            {[round(v, 4) for v in m['per_class_f1']]}")
    print(f"  per-class recall        {[round(v, 4) for v in m['per_class_recall']]}")
    print(f"  support                 {m['per_class_support']}")
    if "image_level" in report:
        il = report["image_level"]
        print(f"  IMAGE-level ({il['method']}): n={il['n_groups']} "
              f"bal_acc={il['balanced_accuracy']:.4f} f1={il['f1_macro']:.4f}")
    print("=" * 62)

    gates_path = run_dir / "gates.json"
    gates = json.load(open(gates_path)) if gates_path.is_file() else {}
    gates["G6"] = True
    gates["G6_source"] = "scripts/eval_test_split.py (rerun; see script docstring)"
    with open(gates_path, "w") as f:
        json.dump(gates, f, indent=2, default=str)
    print(f"[written]  {run_dir/'test_report.json'}  and  {gates_path} (G6=true)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
