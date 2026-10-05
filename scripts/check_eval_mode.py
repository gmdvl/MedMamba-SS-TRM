#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/check_eval_mode.py
==============================
Is the hierarchical MedMamba-SS "collapse" on hyperspectral patches a training
failure, or an evaluation-mode defect?

The manuscript (Sec. VI-D) reports that MedMamba-SS's TRAINING accuracy rises
(0.735 -> 0.906) while its validation macro-F1 stays at 0.2583, the value of a
constant prediction. Rising training accuracy with a constant evaluation output
is the signature of a module that behaves differently in eval() - most often
BatchNorm running statistics that do not match the batch statistics (the
SS-Conv-SSM convolutional branch Phi has four BatchNorm2d layers, and the
11 x 11 hyperspectral patches are far smaller than the 224 x 224 images the
block was designed for).

This script loads a finished hierarchical run's checkpoint and scores the same
class-stratified test sample three ways:

  eval           model.eval()                       (how it was evaluated)
  bn_batchstats  eval, but BatchNorm layers in train() mode: batch statistics,
                 running statistics untouched (momentum 0)
  train          model.train() entirely (dropout / drop-path active too)

and, per BatchNorm layer, compares the running statistics with the batch
statistics of the sample (standardized mean shift, variance ratio).

Verdict:
  * eval collapses to one class, bn_batchstats does not  -> EVAL-MODE DEFECT
    (BatchNorm running statistics); MedMamba-SS can be re-evaluated or retrained
    with a fix (plan G12).
  * all three collapse                                   -> genuine training failure.

Read-only on the run directory except for its output file
<run_dir>/eval_mode_check_v20.json. Cheap: ~9,000 patches, one forward each.

Usage (host; GPU optional - it also runs on CPU in a few minutes):
    python scripts/check_eval_mode.py --run_dir experiments/<hierarchical run> [--per_class 3000]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stratified_indices(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        pool = np.flatnonzero(y == c)
        out.append(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
    return np.sort(np.concatenate(out))


def logits_of(out):
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def summarize(pred: np.ndarray, y: np.ndarray, k: int) -> dict:
    rec = [float((pred[y == c] == c).mean()) if (y == c).any() else None for c in range(k)]
    counts = np.bincount(pred, minlength=k)
    return {"balanced_accuracy": float(np.mean([r for r in rec if r is not None])),
            "per_class_recall": rec, "prediction_counts": counts.tolist(),
            "collapsed": bool(counts.max() >= 0.98 * len(pred))}


def set_bn_batchstats(model: torch.nn.Module):
    """BatchNorm layers use batch statistics without updating running ones."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.train()
            m.momentum = 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Eval-mode diagnosis of a hierarchical MedMamba-SS run.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default=None, help="default: <run_dir>/best_model.pt")
    p.add_argument("--per_class", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args(argv)

    from scripts.eval_test_split import _rebuild_model
    from training.npy_data import compute_global_channel_stats

    run_dir = Path(args.run_dir).resolve()
    cfg = json.load(open(run_dir / "config.json"))
    a = cfg["cli_args"]
    if cfg["architecture"] == "recursive":
        print("NOTE: recursive model - this check targets the hierarchical backbones.")
    data_dir = Path(a["data_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = int(cfg["classes"])

    y_all = np.load(data_dir / "y_test.npy", mmap_mode="r")
    idx = stratified_indices(np.asarray(y_all), args.per_class, a["seed"])
    y = np.asarray(y_all[idx]).astype(np.int64)
    x = np.asarray(np.load(data_dir / "X_test.npy", mmap_mode="r")[idx], dtype=np.float32)  # N,H,W,C
    # X_test is written capture by capture, so sorted indices give near single-class
    # batches; batch-statistics BatchNorm would then normalize within one class.
    # Shuffle once (seeded) so every batch mixes classes as a random test batch would.
    perm = np.random.default_rng(a["seed"] + 7).permutation(len(y))
    x, y = x[perm], y[perm]
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()                                    # N,C,H,W
    if a["normalization"] == "global_zscore":
        mean, std = compute_global_channel_stats(str(data_dir / "X_train.npy"),
                                                 sample_cap=a["norm_stats_sample_cap"], seed=a["seed"])
        mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.as_tensor(np.asarray(std), dtype=torch.float32).view(1, -1, 1, 1)
        x = (x - mean) / std.clamp_min(1e-6)
    wl_path = data_dir / "wavelengths.npy"
    wl = (torch.as_tensor(np.load(wl_path), dtype=torch.float32, device=device)
          if wl_path.is_file() and a.get("use_wavelengths", True) else None)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    # train_example_v16_optimal.build_model_optimal drops an inapplicable
    # --classifier_dropout for split/fullchannel (their head's 0.1 is fixed);
    # _rebuild_model does not, so mirror that here on a copy of the config.
    cfg_build = json.loads(json.dumps(cfg))
    if cfg_build["architecture"] not in ("efficient", "recursive"):
        cfg_build["cli_args"]["classifier_dropout"] = None

    def fresh():
        m = _rebuild_model(cfg_build)
        m.load_state_dict(state, strict=True)
        return m.to(device)

    # Per-layer BatchNorm statistics: inputs recorded in eval mode.
    bn_rows = []

    def hook(name):
        def f(mod, inp, _out):
            t = inp[0].detach().float()
            dims = [0] + list(range(2, t.dim()))
            bm, bv = t.mean(dims).cpu(), t.var(dims, unbiased=False).cpu()
            rm, rv = mod.running_mean.float().cpu(), mod.running_var.float().cpu()
            shift = ((bm - rm).abs() / (rv + mod.eps).sqrt())
            ratio = bv / (rv + mod.eps)
            bn_rows.append({"layer": name, "mean_shift_median": float(shift.median()),
                            "mean_shift_max": float(shift.max()),
                            "var_ratio_median": float(ratio.median()),
                            "var_ratio_max": float(ratio.max())})
        return f

    def run(mode: str):
        model = fresh()
        model.eval()
        handles = []
        if mode == "bn_batchstats":
            set_bn_batchstats(model)
        elif mode == "train":
            model.train()
        if mode == "eval":
            for n, mod in model.named_modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    handles.append(mod.register_forward_hook(hook(n)))
        preds, spread = [], []
        with torch.no_grad():
            for i in range(0, len(x), args.batch_size):
                xb = x[i:i + args.batch_size].to(device)
                out = model(xb, wavelengths=wl) if wl is not None else model(xb)
                lg = logits_of(out).float()
                preds.append(lg.argmax(1).cpu())
                spread.append(lg.std(0).mean().item())
                if mode == "eval" and handles:          # one batch of BN statistics is enough
                    for h in handles:
                        h.remove()
                    handles = []
        pred = torch.cat(preds).numpy()
        res = summarize(pred, y, k)
        res["mean_logit_std_across_samples"] = float(np.mean(spread))
        return res

    results = {m: run(m) for m in ("eval", "bn_batchstats", "train")}
    ev, bs = results["eval"], results["bn_batchstats"]
    if ev["collapsed"] and not bs["collapsed"] and bs["balanced_accuracy"] > ev["balanced_accuracy"] + 0.1:
        verdict = "EVAL-MODE DEFECT: BatchNorm running statistics"
    elif all(r["collapsed"] for r in results.values()):
        verdict = "GENUINE TRAINING FAILURE: collapsed in every mode"
    elif not ev["collapsed"]:
        verdict = "NOT COLLAPSED on this sample in eval mode - re-check the run's own test report"
    else:
        verdict = "INCONCLUSIVE - see per-mode results"

    report = {"run": run_dir.name, "checkpoint": str(ckpt_path), "n_patches": int(len(y)),
              "per_class": args.per_class, "verdict": verdict, "modes": results,
              "batchnorm_layers": bn_rows}
    out = run_dir / "eval_mode_check_v20.json"
    out.write_text(json.dumps(report, indent=2))
    for m, r in results.items():
        print(f"  {m:14s} bal-acc {r['balanced_accuracy']:.4f}  preds {r['prediction_counts']}  "
              f"collapsed={r['collapsed']}")
    worst = sorted(bn_rows, key=lambda r: -r["var_ratio_max"])[:5]
    for r in worst:
        print(f"  BN {r['layer']}: var ratio max {r['var_ratio_max']:.3g}, "
              f"mean shift max {r['mean_shift_max']:.3g}")
    print(f"[verdict] {verdict}\n[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
