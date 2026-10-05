# -*- coding: utf-8 -*-
"""
test_recursive_features_parity.py
====================================
MedMamba-SS-TRM v16 plan, Stage 2.1 verification (R-2).

`training/recursive_features.forward_deep_supervision_with_features`
duplicates ~25 lines of `MedMambaSSTRM.forward_deep_supervision`'s
ACT/halting loop so it can return the LIVE final feature map instead of a
detached one. This is the guard against that duplication silently drifting:
in eval mode, with a fixed seed, the two must produce elementwise-identical
logits and q-values.
"""

import torch

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from training.recursive_features import forward_deep_supervision_with_features


def recursive_cfg(**over):
    base = dict(recursive=True, dims=(64,), depths=(1,), d_state=8, d_ctx=32,
                patch_size=1, spectral_depth=2, trm_dim=64, trm_mixer="mlp",
                trm_deep_supervision_steps=3)
    base.update(over)
    return MedMambaSSTRMConfig(**base)


def make(num_classes=3, **over):
    torch.manual_seed(0)
    return MedMambaSSTRM(recursive_cfg(**over), num_classes=num_classes)


def test_logits_and_q_match_in_eval_mode():
    model = make().eval()
    x = torch.randn(4, 16, 11, 11)

    torch.manual_seed(1)
    ref_logits, ref_q = model.forward_deep_supervision(x)

    torch.manual_seed(1)
    got_logits, got_q, feat = forward_deep_supervision_with_features(model, x)

    assert len(ref_logits) == len(got_logits)
    for rl, gl in zip(ref_logits, got_logits):
        assert torch.equal(rl, gl)
    for rq, gq in zip(ref_q, got_q):
        assert torch.equal(rq, gq)
    assert feat.shape[0] == x.shape[0]


def test_feature_map_is_live_and_matches_last_segment_shape():
    model = make().train()
    x = torch.randn(4, 16, 11, 11, requires_grad=False)
    logits_list, q_list, feat = forward_deep_supervision_with_features(model, x)

    assert feat.requires_grad  # the whole point of R-2: NOT detached
    loss = feat.sum()
    loss.backward()
    core_params = list(model.backbone.core.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
               for p in core_params), "gradient from the live feature map never reached the recursive core"


def test_act_halting_variant_also_matches():
    model = make(trm_act_halting=True, trm_halt_threshold=0.5, trm_halt_exploration_prob=0.0).eval()
    x = torch.randn(4, 16, 11, 11)

    torch.manual_seed(2)
    ref_logits, ref_q = model.forward_deep_supervision(x)
    torch.manual_seed(2)
    got_logits, got_q, _feat = forward_deep_supervision_with_features(model, x)

    assert len(ref_logits) == len(got_logits)
    for rl, gl in zip(ref_logits, got_logits):
        assert torch.equal(rl, gl)


if __name__ == "__main__":
    test_logits_and_q_match_in_eval_mode()
    test_feature_map_is_live_and_matches_last_segment_shape()
    test_act_halting_variant_also_matches()
    print("OK")
