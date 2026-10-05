# -*- coding: utf-8 -*-
"""
test_trm_integration.py
=========================
Verification for the TRM (Tiny Recursive Model) recursive variant added to
`medmamba_ss_trm.py` (`MedMambaSSTRM`, `medmamba_ss_trm_default`), `medmamba_ss_trm_ema.py`
(`EMAHelper`), and `training/trainerg_v10.py` (`TrainerG_v10`, when present).

Run with: pytest test_trm_integration.py -q
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from medmamba_ss_trm import (
    MedMambaSSTRMConfig, MedMambaSSTRM, medmamba_ss_trm_default, medmamba_ss_tiny,
)
from medmamba_ss_trm_ema import EMAHelper


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------
# Parameter budget
# ----------------------------------------------------------------------------

def test_trm_param_count_well_under_target():
    trm = medmamba_ss_trm_default(num_classes=6)
    n = count_params(trm)
    assert n < 5_000_000, f"medmamba_ss_trm_default has {n/1e6:.3f}M params, expected < 5M"


def test_recursive_false_leaves_hierarchical_model_unchanged():
    assert MedMambaSSTRMConfig().recursive is False
    a = count_params(medmamba_ss_tiny(num_classes=6))
    b = count_params(medmamba_ss_tiny(num_classes=6))
    assert a == b == 27_421_121, (
        "medmamba_ss_tiny's param count drifted; the recursive addition must be inert "
        "for recursive=False (default)."
    )


# ----------------------------------------------------------------------------
# Forward/backward on both real-data shapes
# ----------------------------------------------------------------------------

def test_forward_hsi_and_rgb_shapes():
    model = medmamba_ss_trm_default(num_classes=6)
    model.eval()
    for C in (3, 32, 826):  # PAD-UFES RGB, current HSI preset, legacy full-spectrum
        with torch.no_grad():
            out = model(torch.randn(2, C, 11, 11))
        assert tuple(out.shape) == (2, 6)


def test_deep_supervision_list_length_and_eval_single_tensor():
    model = medmamba_ss_trm_default(num_classes=6)

    model.train()
    logits_list, q_list = model.forward_deep_supervision(torch.randn(2, 32, 11, 11))
    assert len(logits_list) == model.cfg.trm_deep_supervision_steps
    assert len(q_list) == len(logits_list)
    for l, q in zip(logits_list, q_list):
        assert tuple(l.shape) == (2, 6)
        assert tuple(q.shape) == (2,)

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 32, 11, 11))
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (2, 6), (
        "eval() must return a single [B, num_classes] tensor so MedMambaSSTRMLatentReconWrapper "
        "and every existing trainer's `isinstance(out, tuple)` unpacking keep working."
    )


def test_gradient_coverage_deep_supervision_plus_halting():
    model = medmamba_ss_trm_default(num_classes=6)
    model.train()
    y = torch.randint(0, 6, (2,))
    logits_list, q_list = model.forward_deep_supervision(torch.randn(2, 32, 11, 11))
    ce = sum(F.cross_entropy(l, y) for l in logits_list) / len(logits_list)
    bce = sum(F.binary_cross_entropy_with_logits(q, (l.argmax(-1) == y).float())
              for l, q in zip(logits_list, q_list)) / len(logits_list)
    (ce + 0.5 * bce).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == [], f"parameters with no gradient: {missing}"


def test_backbone_forward_features_compat_with_recon_wrapper():
    """`MedMambaSSTRMLatentReconWrapper` calls `backbone.forward_features(x)` then
    `head(backbone_out)` and reads `backbone_out['feature_map']` /
    `base_model.cfg.dims[-1]` - verify that contract directly."""
    model = medmamba_ss_trm_default(num_classes=6)
    model.eval()
    with torch.no_grad():
        feats = model.backbone.forward_features(torch.randn(2, 32, 11, 11))
        assert feats["feature_map"].shape[-1] == model.cfg.dims[-1] == model.cfg.trm_dim
        logits = model.head(feats)
    assert tuple(logits.shape) == (2, 6)


def test_mixer_variants():
    for mixer in ("ss2d", "mlp", "attention"):
        cfg = MedMambaSSTRMConfig(recursive=True, trm_dim=64, d_ctx=32, patch_size=1,
                               trm_mixer=mixer, trm_deep_supervision_steps=2)
        model = MedMambaSSTRM(cfg, num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 7, 11, 11))
        assert tuple(out.shape) == (2, 5)


def test_single_state_variant():
    cfg = MedMambaSSTRMConfig(recursive=True, trm_dim=64, d_ctx=32, patch_size=1,
                           trm_two_state=False, trm_deep_supervision_steps=2)
    model = MedMambaSSTRM(cfg, num_classes=5)
    model.train()
    y = torch.randint(0, 5, (2,))
    logits_list, q_list = model.forward_deep_supervision(torch.randn(2, 7, 11, 11))
    ce = sum(F.cross_entropy(l, y) for l in logits_list) / len(logits_list)
    bce = sum(F.binary_cross_entropy_with_logits(q, (l.argmax(-1) == y).float())
              for l, q in zip(logits_list, q_list)) / len(logits_list)
    (ce + 0.5 * bce).backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []


# ----------------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------------

def test_ema_update_changes_shadow_weights():
    model = medmamba_ss_trm_default(num_classes=6)
    ema = EMAHelper(mu=0.9)
    ema.register(model)
    before = {k: v.clone() for k, v in ema.shadow.items()}

    # One optimizer step so live weights diverge from the shadow.
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    y = torch.randint(0, 6, (2,))
    logits_list, _ = model.forward_deep_supervision(torch.randn(2, 32, 11, 11))
    loss = sum(F.cross_entropy(l, y) for l in logits_list) / len(logits_list)
    loss.backward()
    opt.step()

    ema.update(model)
    changed = any(not torch.allclose(before[k], ema.shadow[k]) for k in before)
    assert changed, "EMA shadow weights did not change after update()"

    live_before_copy = {k: v.data.clone() for k, v in model.named_parameters()}
    ema_model = ema.ema_copy(model)
    ema_params = dict(ema_model.named_parameters())
    for k in ema.shadow:
        assert torch.allclose(ema_params[k].data, ema.shadow[k])
    # ema_copy must not mutate the live model.
    for k, v in model.named_parameters():
        assert torch.equal(v.data, live_before_copy[k])


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
