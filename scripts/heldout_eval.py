#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/heldout_eval.py
===========================
Two additions to a FINISHED run, neither of which retrains or touches the
run's own artefacts:

1. Full-validation predictions. During training, validation is scored on a
   9.86 % patient-stratified subset and its predictions are not kept beyond
   that subset. This scores the selected checkpoint on ALL 334,516
   validation patches with the same evaluation code the test split uses
   (`TrainerG_v12.evaluate_test_split`), so that the five validation patients
   and the five test patients can be analysed together, per patient.
   -> <run_dir>/heldout_v19/validation/{test_report.json,test_predictions.npz}
      (the file names are the evaluator's; the split is VALIDATION)

2. `--xai`: for a stratified sample of TEST patches, the spectral pathway's
   band-gate weights (averaged over positions) and the pooled answer-state
   feature the classifier reads, for the band-importance and latent-space
   figures.
   -> <run_dir>/heldout_v19/xai_test.npz

Usage (host, GPU):
    python scripts/heldout_eval.py --run_dirs experiments/<run> [...] [--xai]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.npy_data import compute_global_channel_stats  # noqa: E402
from training.dataloader_config_v16 import train_val_policies_v16  # noqa: E402
from training.npy_data import NpyDatasetV16  # noqa: E402
from training.numerical_stability import StabilityConfig  # noqa: E402
from training.trainerg_v12 import TrainerG_v12  # noqa: E402
from scripts.eval_test_split import _rebuild_model  # noqa: E402


def stratified_sample(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        out.append(rng.choice(idx, size=min(per_class, idx.size), replace=False))
    return np.sort(np.concatenate(out))


@torch.no_grad()
def export_xai(trainer, test_ds, labels, groups, per_class, batch_size, out_path, device):
    model = trainer.model
    base = getattr(model, "base_model", model)
    base.eval()
    idx = stratified_sample(labels, per_class, seed=0)
    loader = DataLoader(Subset(test_ds, idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=4)
    wl = trainer._wl
    pooled, band_w, logits_all = [], [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            feats = base.backbone.forward_features(x, wavelengths=wl, sensor_range=trainer.sensor_range)
            logits = base.head.forward_features(feats) if hasattr(base.head, "forward_features") else None
        pooled.append(feats["pooled"].float().cpu().numpy())
        bw = feats.get("band_weights")
        if bw is not None:
            band_w.append(bw.float().mean(dim=(1, 2)).cpu().numpy())
        if logits is not None:
            logits_all.append(logits.float().cpu().numpy())
    np.savez_compressed(
        out_path, test_index=idx, labels=labels[idx], groups=groups[idx],
        pooled=np.concatenate(pooled),
        band_weights=(np.concatenate(band_w) if band_w else np.zeros((len(idx), 0), np.float32)),
        logits=(np.concatenate(logits_all) if logits_all else np.zeros((len(idx), 0), np.float32)),
        wavelengths=(np.zeros(0, np.float32) if wl is None else wl.cpu().numpy()))
    print(f"  [xai] {len(idx)} test patches -> {out_path}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", required=True)
    p.add_argument("--xai", action="store_true")
    p.add_argument("--xai_per_class", type=int, default=3000)
    p.add_argument("--skip_validation", action="store_true")
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats_cache = {}

    for rd in args.run_dirs:
        run_dir = Path(rd).resolve()
        cfg = json.load(open(run_dir / "config.json"))
        a = cfg["cli_args"]
        data_dir = Path(a["data_dir"])
        if not data_dir.is_absolute():
            data_dir = Path(__file__).resolve().parents[1] / data_dir
        out_root = run_dir / "heldout_v19"
        out_root.mkdir(exist_ok=True)
        print(f"[run] {run_dir.name}  seed={a['seed']}  C={cfg['in_channels']}", flush=True)

        key = (str(data_dir), a["norm_stats_sample_cap"], a["seed"])
        if key not in stats_cache:
            stats_cache[key] = compute_global_channel_stats(
                str(data_dir / "X_train.npy"), sample_cap=a["norm_stats_sample_cap"], seed=a["seed"])
        gmean, gstd = stats_cache[key]

        def ds(split):
            return NpyDatasetV16(str(data_dir / f"X_{split}.npy"), str(data_dir / f"y_{split}.npy"),
                                 storage_mode="auto", normalization=a["normalization"],
                                 global_mean=gmean, global_std=gstd, return_norm_stats=False)

        wl_path = data_dir / "wavelengths.npy"
        wavelengths = np.load(wl_path) if wl_path.is_file() else None
        _, val_policy = train_val_policies_v16(a["loader_mode"], a["num_workers"], a["prefetch_factor"],
                                               a["persistent_workers"], a["pin_memory"])

        # The frozen `_rebuild_model` forwards --classifier_dropout to `build_model`, which rejects it for
        # the hierarchical architectures. Training went through `build_model_optimal`, which drops it
        # (the hierarchical head's dropout is fixed at 0.1), so do the same here on a copy of the config.
        rebuild_cfg = json.loads(json.dumps(cfg))
        if cfg["architecture"] not in ("recursive", "efficient"):
            rebuild_cfg["cli_args"]["classifier_dropout"] = None

        def make_trainer(loader, exp_dir):
            exp_dir.mkdir(parents=True, exist_ok=True)
            return TrainerG_v12(
                model=_rebuild_model(rebuild_cfg), train_loader=None, val_loader=loader, optimizer=None,
                scheduler=None, device=device, exp_dir=exp_dir, class_names=cfg["class_names"],
                config=cfg, wavelengths=wavelengths, checkpoint_metric=a["checkpoint_metric"],
                amp_mode=a["amp"], stability_cfg=StabilityConfig(),
                sensor_range=(tuple(a["sensor_range"]) if a.get("sensor_range") else None),
                use_wavelengths=a["use_wavelengths"],
                ema_rate=(a["trm_ema_rate"] if cfg["architecture"] == "recursive" else None),
                scheduler_interval=a.get("scheduler_interval", "step"))

        ckpt = str(run_dir / "best_model.pt")
        if not args.skip_validation:
            val_ds = ds("val")
            loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                **val_policy.dataloader_kwargs(device, seed=a["seed"]))
            trainer = make_trainer(loader, out_root / "validation")
            rep = trainer.evaluate_test_split(loader, checkpoint_path=ckpt, num_classes=cfg["classes"])
            m = rep["sklearn_metrics"]
            print(f"  [validation, full {len(val_ds)}] bal {m['balanced_accuracy']:.4f} "
                  f"F1 {m['f1_macro']:.4f} acc {m['accuracy']:.4f}", flush=True)
            del val_ds, loader

        if args.xai:
            test_ds = ds("test")
            loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            trainer = make_trainer(loader, out_root / "xai_tmp")
            state = torch.load(ckpt, map_location=device, weights_only=False)
            trainer.model.load_state_dict(state["model_state"])
            trainer.model.to(device)
            labels = np.load(data_dir / "y_test.npy")
            groups = np.load(data_dir / "groups_test.npy")
            export_xai(trainer, test_ds, labels, groups, args.xai_per_class, args.batch_size,
                       out_root / "xai_test.npz", device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
