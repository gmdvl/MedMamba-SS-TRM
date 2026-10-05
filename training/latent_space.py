# -*- coding: utf-8 -*-
"""
training/latent_space.py
==========================
Main Development Plan, Phase 6 - Latent Space visualization (t-SNE, UMAP).

`extract_penultimate_features(model, loader, device, ...)` gets a [N, D]
embedding per validation/test sample without needing to know the exact
architecture: it tries, in order,
  1. `model.forward_features(x)` (common convention, and what
     `medmamba_ss_trm.py`'s backbone exposes for band-selection/spectral work),
  2. a forward hook on the first attribute name found among
     ("head", "classifier", "fc", "cls_head") that captures that module's
     *input* (i.e. the penultimate representation feeding the classifier),
  3. otherwise gives up and returns None (caller should skip the plot
     rather than crash training).

The actual plotting (t-SNE/UMAP) lives in `training.plots.plot_latent_space`
- this module is just the model-agnostic feature extraction half.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


@torch.no_grad()
def extract_penultimate_features(model, loader, device: str, max_samples: int = 2000,
                                  wavelengths=None, sensor_range=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Returns (features [N,D], labels [N]) or None if no extraction
    strategy worked for this model. Capped at `max_samples` patches (t-SNE/
    UMAP are both superlinear, so this keeps the plot fast on large val/test
    sets while still being representative)."""
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    base_model = base_model.eval().to(device)

    strategy = None
    hook_handle = None
    captured = {}

    if hasattr(base_model, "forward_features"):
        strategy = "forward_features"
    else:
        for attr in ("head", "classifier", "fc", "cls_head", "classification_head"):
            module = getattr(base_model, attr, None)
            if module is not None:
                def _hook(mod, inputs, output, _captured=captured):
                    _captured["feat"] = inputs[0].detach()
                hook_handle = module.register_forward_hook(_hook)
                strategy = f"hook:{attr}"
                break

    if strategy is None:
        return None

    feats, labels = [], []
    n_collected = 0
    try:
        for x, y in loader:
            if n_collected >= max_samples:
                break
            x = x.to(device, non_blocking=True)
            try:
                if strategy == "forward_features":
                    f = base_model.forward_features(x, wavelengths=wavelengths, sensor_range=sensor_range) \
                        if wavelengths is not None else base_model.forward_features(x)
                else:
                    out = base_model(x, wavelengths=wavelengths, sensor_range=sensor_range) \
                        if wavelengths is not None else base_model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    f = captured.get("feat")
                    if f is None:
                        return None
            except Exception:
                return None

            if f.ndim > 2:
                f = f.flatten(start_dim=1)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
            n_collected += f.shape[0]
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if not feats:
        return None
    features = np.concatenate(feats, axis=0)[:max_samples]
    all_labels = np.concatenate(labels, axis=0)[:max_samples]
    return features, all_labels
