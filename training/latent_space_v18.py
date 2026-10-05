# -*- coding: utf-8 -*-
"""
training/latent_space_v18.py
==============================
v18 F3 - penultimate-feature extraction that survives a reconstruction run.

The defect
----------
`training/latent_space.py:62` iterates

    for x, y in loader:

With `--recon_mode latent` the dataset is built with `return_norm_stats=True`
and the loader yields THREE-tuples `(patch, label, norm_stats)`, so the unpack
raises `ValueError`. `TrainerG_v12._generate_rich_diagnostics` catches it at
l. 1178 and logs

    [diagnostics] latent-space visualization failed: ...

so the run completes, nothing is flagged, and `latent_space/` is silently
empty. Measured across the archived runs:

    20260914_181738  recon_mode=latent  -> 0 files
    20260904_230812  recon_mode=latent  -> 0 files
    20260906_222725  recon_mode=none    -> 1 file
    20260907_025027  recon_mode=none    -> 1 file

Every reconstruction run in the repository has lost its embedding plot, which
is the natural qualitative figure for "what do 32 bands buy over 3" and for
"does the recursive core separate the classes the way the hierarchical one
does".

Why not a three-line fix in the frozen file
-------------------------------------------
`training/latent_space.py` is NOT on `FROZEN_GLOBS`, so patching it would pass
the invariant test. It is still the wrong fix, for two reasons:

1. The tuple is only half the problem. With `--recon_mode latent` the model is
   a `MedMambaSSTRMLatentReconWrapperV2`, whose `forward` returns
   `(logits, x_recon)` and which has no `forward_features` of its own - the
   backbone is one level down, at `.base_model`. The frozen extractor peels
   `_orig_mod` (torch.compile) but not `base_model`, so even with the tuple
   fixed it would fall through to the hook strategy and capture the wrapper's
   classifier input, or return None.
2. It would not retro-fit the four runs above.

So the corrected logic lives here, is imported by BOTH
`scripts/latent_space.py` (retro-fit) and `TrainerG_v18` (new runs), and
the plotting stays in the frozen, unmodified `training.plots.plot_latent_space`.

One more thing the frozen call site gets wrong: `trainerg_v12.py:1173` calls
`extract_penultimate_features(self.model, self.val_loader, str(self.device))`
with no `wavelengths`, so the spectral pathway takes its index-encoding
fallback (`medmamba_ss_trm.py:836`) and the embedding is not the one the model was
trained with. This module forwards them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

_HEAD_ATTRS = ("head", "classifier", "fc", "cls_head", "classification_head")


def unwrap_for_features(model):
    """Peel to the module that owns `forward_features`.

    `_orig_mod`   - torch.compile's OptimizedModule wrapper.
    `base_model`  - the reconstruction wrapper (`MedMambaSSTRMLatentReconWrapper`,
                    `...V2`, `MedMambaSSTRMRawReconWrapper`), whose forward returns
                    a 2-tuple and which holds the real model underneath.

    Applied repeatedly, so a compiled reconstruction-wrapped model unwraps in
    either nesting order.
    """
    seen = 0
    while seen < 4:
        inner = getattr(model, "_orig_mod", None) or getattr(model, "base_model", None)
        if inner is None:
            break
        model = inner
        seen += 1
    return model


@torch.no_grad()
def extract_penultimate_features(model, loader, device: str, max_samples: int = 2000,
                                  wavelengths=None, sensor_range=None
                                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """`(features [N, D], labels [N])`, or None if no strategy worked.

    Same contract and the same two strategies as
    `training.latent_space.extract_penultimate_features` - `forward_features`
    first, then a hook on the classifier's input - with three differences, each
    of which is a bug in the frozen version rather than a preference:
    the batch unpack tolerates 2- and 3-tuples, the model is unwrapped through
    `base_model` as well as `_orig_mod`, and wavelengths are forwarded.
    """
    base_model = unwrap_for_features(model).eval().to(device)

    strategy = None
    hook_handle = None
    captured = {}
    feature_source = base_model

    # Order matters, and differs from the frozen version's for a reason.
    #
    # Neither top-level model in this repo defines `forward_features` -
    # `MedMambaSS` (medmamba_ss_trm.py:1528) and `MedMambaSSTRM` (l. 1912) both
    # delegate to a `.backbone` that does (l. 1424, l. 1893), and both return
    # the dict this module knows how to reduce. Try the backbone FIRST, so the
    # embedding is the backbone's pooled representation rather than whatever a
    # hook happens to catch.
    #
    # The hook fallback is kept but is genuinely second-best here: `head(...)`
    # for these models is called with the backbone's feature DICT, not a
    # tensor, so `inputs[0]` would be a dict and every downstream tensor
    # operation on it fails. That is why it is a fallback and why
    # `_as_feature_tensor` is tolerant.
    for owner in (getattr(base_model, "backbone", None), base_model):
        if owner is not None and hasattr(owner, "forward_features"):
            strategy, feature_source = "forward_features", owner
            break

    if strategy is None:
        for attr in _HEAD_ATTRS:
            module = getattr(base_model, attr, None)
            if module is not None:
                def _hook(mod, inputs, output, _captured=captured):
                    _captured["feat"] = inputs[0]
                hook_handle = module.register_forward_hook(_hook)
                strategy = f"hook:{attr}"
                break

    if strategy is None:
        return None

    wl = None
    if wavelengths is not None:
        wl = torch.as_tensor(np.asarray(wavelengths, dtype=np.float32), device=device)

    feats, labels = [], []
    n_collected = 0
    try:
        for batch in loader:
            if n_collected >= max_samples:
                break
            # 2-tuple (patch, label) or 3-tuple (patch, label, norm_stats) - the
            # latter is what `--recon_mode latent` produces and what the frozen
            # version cannot unpack.
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            try:
                f = _features_for_batch(base_model, feature_source, x, strategy,
                                        captured, wl, sensor_range)
            except Exception:
                return None
            if f is None:
                return None
            if f.ndim > 2:
                f = f.flatten(start_dim=1)
            feats.append(f.float().cpu().numpy())
            labels.append(np.asarray(y))
            n_collected += f.shape[0]
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if not feats:
        return None
    return (np.concatenate(feats, axis=0)[:max_samples],
            np.concatenate(labels, axis=0)[:max_samples])


def _features_for_batch(base_model, feature_source, x, strategy, captured, wl, sensor_range):
    """One batch through whichever strategy was selected.

    `feature_source` is the module that owns `forward_features` (usually
    `base_model.backbone`); `base_model` is what the hook fallback calls.
    `forward_features` returns a dict for the backbone-level models in this
    repo (`{"feature_map": ..., "pooled": ...}`) and a tensor for others, so
    both are handled rather than assumed.
    """
    kw = {} if wl is None else {"wavelengths": wl, "sensor_range": sensor_range}
    if strategy == "forward_features":
        try:
            out = feature_source.forward_features(x, **kw)
        except TypeError:
            out = feature_source.forward_features(x)
        return _as_feature_tensor(out)

    try:
        base_model(x, **kw)
    except TypeError:
        base_model(x)
    return _as_feature_tensor(captured.get("feat"))


def _as_feature_tensor(out):
    """A `[N, D]`-ish tensor from whatever `forward_features` returned."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for key in ("features", "pooled", "feature_vector", "feature_map"):
            if key in out and torch.is_tensor(out[key]):
                value = out[key]
                # `feature_map` is [B, H, W, D] channels-last; mean-pool the
                # spatial axes so the embedding is per-sample, not per-token.
                if key == "feature_map" and value.ndim == 4:
                    return value.mean(dim=(1, 2))
                return value
        for value in out.values():
            if torch.is_tensor(value):
                return value
        return None
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    return None


def write_latent_space(model, loader, device: str, out_dir: str, class_names=None,
                        max_samples: int = 2000, wavelengths=None, sensor_range=None,
                        seed: int = 0) -> dict:
    """Extract, then plot with the FROZEN `training.plots.plot_latent_space`.

    Returns `{method: path}`; `{}` when extraction found no strategy, so a
    caller can report the skip instead of leaving an empty directory to be
    discovered later.
    """
    from training.plots import plot_latent_space  # frozen, unmodified

    extracted = extract_penultimate_features(
        model, loader, device, max_samples=max_samples,
        wavelengths=wavelengths, sensor_range=sensor_range)
    if extracted is None:
        return {}
    features, labels = extracted
    return plot_latent_space(features, labels, class_names, out_dir, seed=seed)
