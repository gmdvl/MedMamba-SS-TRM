#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/latent_space.py
=============================
v18 F3 - write the latent-space plots that every reconstruction run silently
lost.

Why this exists
---------------
`training/latent_space.py:62` iterates `for x, y in loader:`. With
`--recon_mode latent` the dataset is constructed with `return_norm_stats=True`
and yields THREE-tuples, so the unpack raises `ValueError`.
`TrainerG_v12._generate_rich_diagnostics` catches it at l. 1178, logs
`[diagnostics] latent-space visualization failed: ...`, and the run completes
with an empty `latent_space/`. Measured:

    20260914_181738  recon_mode=latent  -> 0 files
    20260904_230812  recon_mode=latent  -> 0 files
    20260906_222725  recon_mode=none    -> 1 file
    20260907_025027  recon_mode=none    -> 1 file

The corrected extraction is in `training/latent_space_v18.py` (shared with
`TrainerG_v18`), which additionally unwraps the reconstruction wrapper's
`base_model` - the frozen version peels only `_orig_mod` - and forwards
`wavelengths`, which `trainerg_v12.py:1173` does not, so the frozen path
embedded with the index-encoding fallback rather than the encoding the model
was trained with.

The plotting itself is the FROZEN, unmodified `training.plots.plot_latent_space`.

Read-only: writes only into `<run_dir>/latent_space/`.

Usage
-----
    python scripts/latent_space.py --run_dir experiments/<run>
    python scripts/latent_space.py --run_dir experiments/<run> --split test
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

from training.npy_data import NpyDatasetV16
from training.dataloader_config_v16 import train_val_policies_v16
from training.latent_space_v18 import write_latent_space

from scripts.eval_test_split import _rebuild_model  # noqa: E402


def _stratified_indices(y_path: Path, max_samples: int, seed: int) -> list:
    """Up to `max_samples` indices, spread as evenly over the classes as their
    supports allow, sampled without replacement at `seed`.

    Returned in sorted order so a memory-mapped array is read close to
    sequentially rather than jumping across a multi-gigabyte file.
    """
    y = np.load(y_path, mmap_mode="r")
    labels = np.asarray(y)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)

    per_class = max(1, max_samples // max(1, len(classes)))
    chosen = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        take = min(per_class, len(idx))
        chosen.append(rng.choice(idx, size=take, replace=False))
    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)

    # Spend any budget left by small classes on the large ones.
    if len(chosen) < max_samples:
        remaining = np.setdiff1d(np.arange(len(labels)), chosen, assume_unique=False)
        extra = min(max_samples - len(chosen), len(remaining))
        if extra > 0:
            chosen = np.concatenate([chosen, rng.choice(remaining, size=extra, replace=False)])

    counts = {int(c): int((labels[chosen] == c).sum()) for c in classes}
    print(f"[subset] {len(chosen)} patches, per-class {counts} (stratified, seed={seed})")
    return sorted(int(i) for i in chosen)


def _load_checkpoint(model, ckpt_path: Path, device: str) -> str:
    """Load `best_model.pt` into the rebuilt model.

    The key is `model_state` - `TrainerG_v11._save_checkpoint` (l. 566) writes
    it and `TrainerG_v13._load_state_for_export` (l. 135) reads it as
    `state.get("model_state", state)`. This mirrors that exactly rather than
    guessing at a list of candidate names.

    STRICT on purpose. A partial load would produce an embedding of a
    half-random model and a t-SNE plot that looks entirely plausible, which is
    precisely the class of silent defect this script exists to remove.
    `_orig_mod.` is stripped first because a checkpoint saved from a compiled
    model carries that prefix while the rebuilt model does not.
    """
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob.get("model_state", blob) if isinstance(blob, dict) else blob
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)          # strict
    weights = blob.get("weights") if isinstance(blob, dict) else None
    epoch = blob.get("epoch") if isinstance(blob, dict) else None
    return f"{len(state)} tensors, epoch={epoch}, weights={weights!r}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Latent-space plots for a finished run.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation",
                   help="validation matches what the trainer would have plotted")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--checkpoint", default=None, help="default: <run_dir>/best_model.pt")
    p.add_argument("--max_samples", type=int, default=2000,
                   help="t-SNE and UMAP are both superlinear; 2000 is the frozen default")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cfg = json.load(open(run_dir / "config.json"))
    a = cfg["cli_args"]
    data_dir = Path(args.data_dir or a["data_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or a["batch_size"]

    suffix = "val" if args.split == "validation" else "test"
    x_path, y_path = data_dir / f"X_{suffix}.npy", data_dir / f"y_{suffix}.npy"
    if not (x_path.is_file() and y_path.is_file()):
        print(f"ERROR: {x_path} / {y_path} not found.")
        return 2

    print(f"[run]    {run_dir.name}")
    print(f"[data]   {data_dir}  split={args.split}  norm={a['normalization']}")
    print(f"[device] {device}  batch_size={batch_size}  max_samples={args.max_samples}")

    global_mean = global_std = None
    if a["normalization"] == "global_zscore":
        from training.npy_data import compute_global_channel_stats
        global_mean, global_std = compute_global_channel_stats(
            str(data_dir / "X_train.npy"), sample_cap=a["norm_stats_sample_cap"], seed=a["seed"])

    dataset = NpyDatasetV16(str(x_path), str(y_path), storage_mode=a["dataset_storage"],
                            normalization=a["normalization"],
                            global_mean=global_mean, global_std=global_std,
                            return_norm_stats=False)

    # Stratify the subset, do NOT take the first `max_samples` off an
    # unshuffled loader. The `.npy` splits are written capture by capture and
    # are therefore ordered by class: the first 600 validation patches of the
    # histology build are all DCIS, so the naive path plots one class and looks
    # entirely plausible while saying nothing. The frozen extractor has this
    # flaw too - it breaks out of the loop at `max_samples` - so the two runs
    # that did produce a `latent_space/` plot produced a single-class one.
    dataset = Subset(dataset, _stratified_indices(y_path, args.max_samples, args.seed))

    _, val_policy = train_val_policies_v16(a["loader_mode"], a["num_workers"], a["prefetch_factor"],
                                           a["persistent_workers"], a["pin_memory"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        **val_policy.dataloader_kwargs(device, seed=a["seed"]))

    model = _rebuild_model(cfg)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "best_model.pt")
    if ckpt_path.is_file():
        print(f"[ckpt]   {ckpt_path.name}: {_load_checkpoint(model, ckpt_path, device)}")
    else:
        print(f"[ckpt]   {ckpt_path} not found - embedding an UNTRAINED model, which is not "
              f"a result. Pass --checkpoint or stop here.")

    wavelengths = None
    wl_path = data_dir / "wavelengths.npy"
    if a["use_wavelengths"] and wl_path.is_file():
        wavelengths = np.load(wl_path)
        print(f"[wl]     {len(wavelengths)} band centres forwarded "
              f"(the frozen call site does not forward them)")

    out_dir = run_dir / "latent_space"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_latent_space(
        model, loader, device, str(out_dir), class_names=cfg["class_names"],
        max_samples=args.max_samples, wavelengths=wavelengths,
        sensor_range=(tuple(a["sensor_range"]) if a.get("sensor_range") else None),
        seed=args.seed)

    if not written:
        print("RESULT: no extraction strategy matched this model - nothing written.")
        return 1
    for method, path in sorted(written.items()):
        print(f"[written] {method}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
