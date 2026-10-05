# -*- coding: utf-8 -*-
"""
test_recon_gradient.py
========================
MedMamba-SS-TRM v16 plan, Stage 2 verification - THE LOAD-BEARING TEST.

With `recon_mode=latent` on a recursive architecture:
  1. backprop of the reconstruction term ALONE (lambda_mse*MSE + lambda_sam*SAM,
     classification term zeroed) puts finite, non-zero gradient on parameters
     INSIDE the recursive core (`RecursiveCore`, not just the decoder) - R-2.
  2. `copy.deepcopy(model)` still succeeds after a forward pass - the exact
     EMA constraint (`EMAHelper.ema_copy`) that made `medmamba_ss_trm.py` detach
     `last_feature_map` in the first place, so R-2's fix must not break it.
  3. gate G7 (`training/gates.check_reconstruction_gradient`) reports
     PASS on this same setup, and reports RECONSTRUCTION_DETACHED when
     handed a decoder that reads a deliberately-detached feature map (i.e.
     it actually distinguishes the two).
"""

import copy

import torch

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from training.gan import SAMLoss
from training.reconstruction_head_v2 import MedMambaSSTRMLatentReconWrapperV2
from training.gates import check_reconstruction_gradient, default_encoder_params, RECONSTRUCTION_DETACHED


def recursive_cfg(**over):
    base = dict(recursive=True, dims=(64,), depths=(1,), d_state=8, d_ctx=32,
                patch_size=1, spectral_depth=2, trm_dim=64, trm_mixer="mlp",
                trm_deep_supervision_steps=3)
    base.update(over)
    return MedMambaSSTRMConfig(**base)


def make_wrapper(in_channels=16, out_activation="linear"):
    torch.manual_seed(0)
    base = MedMambaSSTRM(recursive_cfg(), num_classes=3)
    return MedMambaSSTRMLatentReconWrapperV2(base, in_channels=in_channels, out_activation=out_activation)


def _forward_with_recon(model, x):
    logits, x_recon = model(x)
    return [logits], None, x_recon


def test_reconstruction_gradient_reaches_recursive_core():
    model = make_wrapper()
    x = torch.randn(4, 16, 11, 11)
    sam_loss_fn = SAMLoss()

    report = check_reconstruction_gradient(model, _forward_with_recon, x, sam_loss_fn,
                                            lambda_mse=1.0, lambda_sam=0.1)
    assert report["passed"], report.get("reason")
    assert report["n_dead"] == 0
    assert report["n_nonfinite"] == 0
    assert report["n_encoder_params_checked"] > 0


def test_deepcopy_still_works_after_forward():
    """The exact constraint that made medmamba_ss_trm.py detach last_feature_map:
    EMAHelper.ema_copy calls copy.deepcopy(module) every epoch, and deepcopy
    refuses a module attribute holding a live autograd graph. Nothing in
    MedMambaSSTRMLatentReconWrapperV2/forward_deep_supervision_with_features
    stores a live tensor as a module attribute, so this must still succeed."""
    model = make_wrapper()
    x = torch.randn(4, 16, 11, 11)
    logits, x_recon = model(x)
    loss = logits.sum() + x_recon.sum()
    loss.backward()

    copied = copy.deepcopy(model)  # must not raise
    assert copied is not model


def test_gate_g7_detects_a_genuinely_detached_decoder():
    """A decoder that reads model.base_model.last_feature_map (the OLD,
    detached pathway) must make G7 fail with RECONSTRUCTION_DETACHED."""
    torch.manual_seed(0)
    base = MedMambaSSTRM(recursive_cfg(), num_classes=3)
    from training.reconstruction_head_v2 import LatentReconstructionDecoderV2
    decoder = LatentReconstructionDecoderV2(base.cfg.dims[-1], 16, out_activation="linear")

    def _detached_forward_with_recon(model, x):
        logits_list, _q = model.forward_deep_supervision(x)
        feat = model.last_feature_map  # DETACHED - the R-2 bug, reproduced on purpose
        x_recon = decoder(feat, target_hw=x.shape[-2:])
        return logits_list, None, x_recon

    x = torch.randn(4, 16, 11, 11)
    sam_loss_fn = SAMLoss()
    report = check_reconstruction_gradient(base, _detached_forward_with_recon, x, sam_loss_fn,
                                            lambda_mse=1.0, lambda_sam=0.1,
                                            encoder_params_fn=default_encoder_params)
    assert report["passed"] is False
    assert RECONSTRUCTION_DETACHED in report["reason"]
    assert report["n_dead"] > 0


if __name__ == "__main__":
    test_reconstruction_gradient_reaches_recursive_core()
    test_deepcopy_still_works_after_forward()
    test_gate_g7_detects_a_genuinely_detached_decoder()
    print("OK")
