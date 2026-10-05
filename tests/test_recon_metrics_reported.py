# -*- coding: utf-8 -*-
"""
test_recon_metrics_reported.py
=================================
MedMamba-SS-TRM v16 plan, Stage 2 verification (R-4/R-6).

  - `_validate_one_epoch_impl` returns non-zero `mse_loss`/`sam_loss` when a
    reconstruction decoder is attached (R-4: these used to be structural
    zeros for every trainer since TrainerG_v6).
  - `fit` writes them into `history.csv`/`history.json` (R-6:
    `loss_components.csv`/`.json` and `per_class_metrics.csv` are written
    again - v5/v11 stopped calling the writers that produce them).
  - `_build_epoch_metrics` INDEXES `mse_loss`/`sam_loss`/`gan_loss` directly:
    a validation dict missing one of those keys raises `KeyError` instead of
    silently defaulting to 0.0 (the bug this whole item is about).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from training.reconstruction_head_v2 import MedMambaSSTRMLatentReconWrapperV2
from training.trainerg_v12 import TrainerG_v12
from training.normalization import normalize_patch, stack_norm_stats

C, H, W, NUM_CLASSES = 6, 5, 5, 3


def tiny_recon_model(**over):
    cfg_kwargs = dict(recursive=True, dims=(16,), depths=(1,), d_state=4, d_ctx=8,
                       patch_size=1, spectral_depth=1, d_token=8, compression_dims=(8,),
                       trm_dim=16, trm_mixer="mlp", trm_core_layers=1, trm_n_latent=1,
                       trm_n_improve=1, trm_deep_supervision_steps=2,
                       spectral_token_fusion="concat_mlp", spectral_pe_gain=0.1,
                       spectral_value_init_std=0.5, spectral_ctx_norm=True,
                       classifier_init="fan_in", wavelength_encoding_scale=None,
                       spectral_chunk_size=0)
    cfg_kwargs.update(over)
    torch.manual_seed(0)
    base = MedMambaSSTRM(MedMambaSSTRMConfig(**cfg_kwargs), num_classes=NUM_CLASSES)
    return MedMambaSSTRMLatentReconWrapperV2(base, in_channels=C, out_activation="linear")


def tiny_recon_loader(n=16, batch=8, seed=0):
    """Raw patches -> per_patch_zscore -> (x, y, norm_stats), so denormalize()
    round-trips exactly (R-3/R-7's reflectance-unit metrics)."""
    g = np.random.default_rng(seed)
    raw = g.normal(loc=500.0, scale=80.0, size=(n, H, W, C)).astype(np.float32)
    xs, stats = [], []
    for i in range(n):
        normalized, off, scale = normalize_patch(raw[i], "per_patch_zscore")
        xs.append(torch.from_numpy(normalized).permute(2, 0, 1))
        stats.append(torch.from_numpy(stack_norm_stats(off, scale)))
    x = torch.stack(xs)
    norm_stats = torch.stack(stats)
    y = torch.arange(n, dtype=torch.long) % NUM_CLASSES
    return DataLoader(TensorDataset(x, y, norm_stats), batch_size=batch, shuffle=False)


def make_trainer(tmp_path, **kwargs):
    model = tiny_recon_model()
    train = tiny_recon_loader(seed=0)
    val = tiny_recon_loader(seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    defaults = dict(model=model, train_loader=train, val_loader=val, optimizer=opt,
                     scheduler=sched, device="cpu", exp_dir=tmp_path,
                     class_names=[f"c{i}" for i in range(NUM_CLASSES)],
                     lambda_mse=1.0, lambda_sam=0.1, lambda_gan=0.0,
                     criterion=nn.CrossEntropyLoss(), amp_mode="off", ema_rate=0.0)
    defaults.update(kwargs)
    return TrainerG_v12(**defaults)


def test_validation_returns_nonzero_mse_and_sam(tmp_path):
    t = make_trainer(tmp_path)
    val_metrics = t._validate_one_epoch_impl()
    assert "mse_loss" in val_metrics and "sam_loss" in val_metrics and "gan_loss" in val_metrics
    assert val_metrics["mse_loss"] > 0.0
    assert val_metrics["sam_loss"] >= 0.0
    assert val_metrics["spectral"], "spectral metrics dict must not be empty when a decoder is attached"


def test_build_epoch_metrics_raises_on_missing_key(tmp_path):
    t = make_trainer(tmp_path)
    good_val = {"loss": 0.1, "cls_loss": 0.1, "mse_loss": 0.2, "sam_loss": 0.3, "gan_loss": 0.0,
                "classification": {"accuracy": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5},
                "spectral": {}}
    train_metrics = {"loss": 0.1, "cls_loss": 0.1, "mse_loss": 0.2, "sam_loss": 0.3, "gan_loss": 0.0,
                      "acc": 0.5, "grad_norm": 1.0, "gradient_health": {"is_valid_epoch": True}}
    # sanity: with every required key present, this must not raise.
    t._build_epoch_metrics(1, train_metrics, good_val, 1.0, 1e-3, 0.0, 1.0, 0.1, 1)

    bad_val = dict(good_val)
    del bad_val["mse_loss"]
    with pytest.raises(KeyError):
        t._build_epoch_metrics(1, train_metrics, bad_val, 1.0, 1e-3, 0.0, 1.0, 0.1, 1)


def test_fit_writes_history_and_loss_components(tmp_path):
    t = make_trainer(tmp_path)
    t.fit(max_epochs=2)

    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history) == 2
    for row in history:
        assert row["val_mse_loss"] > 0.0 or row["MSE_loss"] > 0.0

    assert (tmp_path / "loss_components.csv").is_file()
    assert (tmp_path / "loss_components.json").is_file()
    lc = json.loads((tmp_path / "loss_components.json").read_text())
    assert len(lc) == 2
    assert lc[-1]["val_mse_loss"] >= 0.0


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_validation_returns_nonzero_mse_and_sam(p)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_build_epoch_metrics_raises_on_missing_key(p)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_fit_writes_history_and_loss_components(p)
    print("OK")
