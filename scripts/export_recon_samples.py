# -*- coding: utf-8 -*-
"""
scripts/export_recon_samples.py
=====================================
Export reconstructed CUBES + paper figures from a FINISHED run, without
retraining it.

Any run trained with `--recon_mode latent` still has its decoder weights in
`best_model.pt`, so the reconstructions are recoverable - they were simply
never written down. This script rebuilds the model exactly as
`scripts/eval_test_split.py` does, loads the checkpoint, and runs
`TrainerG_v13.export_reconstruction_samples` over a chosen split.

Two things it does that the training run could not:

  * `--split test` gives a HELD-OUT reconstruction number. Every
    reconstruction figure and metric in the repository today is validation
    only, and `scripts/eval_test_split.py` builds its test loader with
    `return_norm_stats=False`, so it cannot denormalize and therefore cannot
    report reflectance-unit reconstruction metrics at all. This script
    builds the loader with `return_norm_stats=True` and uses its own
    forward path, so the frozen evaluator's 2-tuple unpack is never reached.
  * It writes to `--out_dir` (default `<run>/reconstruction`) WITHOUT
    touching the run's `config.json`, `gates.json` or any metric file. The
    run's provenance is left exactly as it was.

Runs already in this repository that carry a usable decoder:

    experiments/20260904_230812_portance_undersample-hsi_...   (30 epochs, HSI)
    experiments/20260904_164643_..._wce_modover_ppz_sub01_vsub02 (6 epochs, HSI)
    experiments/20260906_031815_pad_optimal_..._balsamp_zscore_optimal (10 epochs, RGB)
    experiments/20260904_121119_..._ppz_sub002_vsub03_seed99   (2 epochs, HSI)

Usage:

    python scripts/export_recon_samples.py \
        --run_dir experiments/20260904_230812_portance_undersample-hsi_recursive_hsi_mlp_sup3_bs256_bf16_wce_modover_ppz_sub01_vsub02 \
        --split val --recon_sample_per_class 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_test_split import _load_cfg, _resolve_data_dir, _rebuild_model
from training.npy_data import NpyDatasetV16
from training.dataloader_config_v16 import train_val_policies_v16
from training.numerical_stability import StabilityConfig
from training.trainerg_v13 import TrainerG_v13


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Export reconstructed cubes and figures from a finished v16 run.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--split", choices=["val", "test", "train"], default="val")
    p.add_argument("--data_dir", default=None, help="override config.json's data_dir (e.g. if it moved)")
    p.add_argument("--checkpoint", default=None, help="default: <run_dir>/best_model.pt")
    p.add_argument("--out_dir", default=None, help="default: <run_dir>/reconstruction")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--recon_sample_per_class", type=int, default=3)
    p.add_argument("--recon_rgb_bands", default=None)
    p.add_argument("--recon_render_png", choices=["on", "off"], default="on")
    p.add_argument("--recon_max_cubes", type=int, default=64)
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cfg = _load_cfg(run_dir)
    a = cfg["cli_args"]

    if a.get("recon_mode", "none") == "none":
        print(f"ERROR: this run was trained with --recon_mode none. There is no decoder in "
              f"{run_dir/'best_model.pt'} and nothing to reconstruct.\n"
              f"       Re-train with train_example_v16_recon.py --recon_mode latent.")
        return 2

    data_dir = _resolve_data_dir(cfg, args.data_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or a["batch_size"]
    ckpt = Path(args.checkpoint) if args.checkpoint else (run_dir / "best_model.pt")
    if not ckpt.is_file():
        print(f"ERROR: {ckpt} does not exist.")
        return 2

    x_path = data_dir / f"X_{args.split}.npy"
    y_path = data_dir / f"y_{args.split}.npy"
    if not x_path.is_file() or not y_path.is_file():
        print(f"ERROR: {x_path} / {y_path} not found.")
        return 2

    global_mean = global_std = None
    if a["normalization"] == "global_zscore":
        from training.npy_data import compute_global_channel_stats
        global_mean, global_std = compute_global_channel_stats(
            str(data_dir / "X_train.npy"), sample_cap=a["norm_stats_sample_cap"], seed=a["seed"])

    # `return_norm_stats=True` is the whole point: without the per-patch
    # offset/scale the cubes can only be saved in normalized units, which is
    # the unit error R-3 fixed at metric level and must not reappear here.
    ds = NpyDatasetV16(str(x_path), str(y_path), storage_mode=a["dataset_storage"],
                       normalization=a["normalization"],
                       global_mean=global_mean, global_std=global_std,
                       return_norm_stats=True)
    _, val_policy = train_val_policies_v16(a["loader_mode"], a["num_workers"], a["prefetch_factor"],
                                           a["persistent_workers"], a["pin_memory"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        **val_policy.dataloader_kwargs(device, seed=a["seed"]))

    wavelengths = None
    wl_path = data_dir / "wavelengths.npy"
    if a["use_wavelengths"] and wl_path.is_file():
        wavelengths = np.load(wl_path)

    model = _rebuild_model(cfg)
    print(f"[run]      {run_dir.name}")
    print(f"[split]    {args.split}: {len(ds)} patches   norm={a['normalization']}   "
          f"recon_mode={a['recon_mode']}")
    print(f"[model]    {sum(q.numel() for q in model.parameters())/1e6:.3f}M params on {device}")

    trainer = TrainerG_v13(
        model=model, train_loader=None, val_loader=loader, optimizer=None, scheduler=None,
        device=device, exp_dir=run_dir, class_names=cfg["class_names"],
        config=cfg, wavelengths=wavelengths, checkpoint_metric=a["checkpoint_metric"],
        amp_mode=a["amp"], stability_cfg=StabilityConfig(),
        sensor_range=(tuple(a["sensor_range"]) if a.get("sensor_range") else None),
        use_wavelengths=a["use_wavelengths"],
        ema_rate=(a["trm_ema_rate"] if cfg["architecture"] == "recursive" else None),
        scheduler_interval=a.get("scheduler_interval", "step"),
        recon_save_samples=True,
        recon_sample_per_class=args.recon_sample_per_class,
        recon_rgb_bands=args.recon_rgb_bands,
        recon_render_png=(args.recon_render_png == "on"),
        recon_max_cubes=args.recon_max_cubes,
        record_config=False,          # this run's config.json is left alone
    )

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "reconstruction")
    status = trainer.export_reconstruction_samples(
        loader=loader, split=args.split, checkpoint_path=str(ckpt), out_dir=out_dir)

    print()
    print("=" * 62)
    if not status.get("written"):
        print(f"  NOT WRITTEN: {status.get('reason')}")
        print(f"  see {out_dir/'status.json'}")
        print("=" * 62)
        return 1
    print(f"  wrote {status['n_samples']} cubes  ->  {status['npz']}")
    print(f"  figures: {status['figures']}   RGB bands: {status['rgb_bands']}")
    print(f"  {status['rgb_band_note']}")
    print(f"  denormalized (reflectance units): {status['denormalized']}")
    metrics_path = out_dir / "samples_metrics.json"
    if metrics_path.is_file():
        rows = json.load(open(metrics_path))
        for key in ("sam_deg", "rmse", "psnr", "ssim2d"):
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                print(f"  {key:<12} mean {np.mean(vals):.4f}   "
                      f"min {np.min(vals):.4f}   max {np.max(vals):.4f}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
