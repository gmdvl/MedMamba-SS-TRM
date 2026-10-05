# -*- coding: utf-8 -*-
"""
training/evaluator.py
======================
Standalone-evaluation logic shared by `evaluate.py` and the automatic
end-of-training test pass in `train_example.py`. Wraps:
  - inference -> predictions + softmax probabilities
  - training.metrics.compute_classification_metrics (accuracy family)
  - efficiency profiling (latency/throughput/params/GPU memory)
  - training.spectral_metrics (band-selection summaries, when applicable)
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as metrics_mod
from . import spectral_metrics as spectral_mod


@torch.no_grad()
def run_inference(model, loader, device: str, wavelengths=None, sensor_range=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x, wavelengths=wavelengths, sensor_range=sensor_range) if wavelengths is not None else model(x)
        probs = F.softmax(out, dim=-1)
        all_preds.append(out.argmax(dim=-1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs))


@torch.no_grad()
def measure_efficiency(model, sample_x: torch.Tensor, device: str, wavelengths=None, sensor_range=None,
                        n_warmup: int = 5, n_runs: int = 30, throughput_batch_size: int = 16) -> Dict:
    model = model.eval().to(device)
    from . import utils as utils_mod
    results = utils_mod.count_parameters(model)
    results["num_params"] = results.pop("total")
    results["num_trainable_params"] = results.pop("trainable")
    results["device"] = device

    def timed_forward(x, n):
        x = x.to(device)
        wl = wavelengths.to(device) if wavelengths is not None else None
        for _ in range(n_warmup):
            model(x, wavelengths=wl, sensor_range=sensor_range) if wl is not None else model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x, wavelengths=wl, sensor_range=sensor_range) if wl is not None else model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return np.array(times)

    x1 = sample_x[:1]
    t1 = timed_forward(x1, n_runs)
    results["latency_bs1_ms_mean"] = round(float(t1.mean() * 1000), 4)
    results["latency_bs1_ms_std"] = round(float(t1.std() * 1000), 4)

    reps = -(-throughput_batch_size // sample_x.shape[0])
    xb = sample_x.repeat(reps, 1, 1, 1)[:throughput_batch_size]
    tb = timed_forward(xb, n_runs)
    results[f"latency_bs{throughput_batch_size}_ms_mean"] = round(float(tb.mean() * 1000), 4)
    results["throughput_images_per_sec"] = round(float(throughput_batch_size / tb.mean()), 2)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = timed_forward(xb, 1)
        results["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 2)

    return results


def evaluate_model(model, loader, device: str, wavelengths=None, sensor_range=None,
                    class_names=None, top_k_band_frac: float = 0.25,
                    band_wavelengths: Optional[np.ndarray] = None) -> Dict:
    """One-stop evaluation: accuracy metrics, efficiency, and (if the model
    config has dynamic band selection enabled) band-selection metrics."""
    preds, labels, probs = run_inference(model, loader, device, wavelengths, sensor_range)
    num_classes = probs.shape[1]
    acc_metrics = metrics_mod.compute_classification_metrics(preds, labels, num_classes, probs=probs)

    sample_x, _ = next(iter(loader))
    eff_metrics = measure_efficiency(model, sample_x, device, wavelengths=wavelengths, sensor_range=sensor_range)

    spec_metrics = None
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if getattr(base_model.cfg, "dynamic_band_selection", False):
        band_weights = spectral_mod.collect_band_weights(
            base_model, loader, device, wavelengths=wavelengths, sensor_range=sensor_range, max_batches=50)
        if band_weights is not None:
            spec_metrics = spectral_mod.band_selection_metrics(
                band_weights, wavelengths=band_wavelengths, top_k_frac=top_k_band_frac)
    if spec_metrics is None:
        spec_metrics = spectral_mod.spectral_reconstruction_metrics()

    return {
        "preds": preds, "labels": labels, "probs": probs,
        "accuracy": acc_metrics, "efficiency": eff_metrics, "spectral": spec_metrics,
        "class_names": class_names,
    }
