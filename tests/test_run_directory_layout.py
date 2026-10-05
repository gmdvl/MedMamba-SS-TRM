# -*- coding: utf-8 -*-
"""
test_run_directory_layout.py
===============================
MedMamba-SS-TRM v16 plan, Stage 5 verification.

After two epochs: no `confusion_matrix_epoch_*.png` / `classification_report_
epoch*.txt` / `confusion_matrix.npy` at the run root; `confusion_matrix/` and
`classification_reports/` exist and are populated instead.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from medmamba_ss_trm import MedMambaSSTRMConfig, MedMambaSSTRM
from training.trainerg_v12 import TrainerG_v12

C, HW, NUM_CLASSES = 4, 5, 3


def tiny_model(**over):
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
    return MedMambaSSTRM(MedMambaSSTRMConfig(**cfg_kwargs), num_classes=NUM_CLASSES)


def tiny_loader(n=24, batch=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, C, HW, HW, generator=g)
    y = torch.arange(n, dtype=torch.long) % NUM_CLASSES
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def make_trainer(tmp_path, **kwargs):
    model = tiny_model()
    train, val = tiny_loader(seed=0), tiny_loader(seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    defaults = dict(model=model, train_loader=train, val_loader=val, optimizer=opt,
                     scheduler=sched, device="cpu", exp_dir=tmp_path,
                     class_names=[f"c{i}" for i in range(NUM_CLASSES)],
                     lambda_mse=0.0, lambda_sam=0.0, lambda_gan=0.0,
                     criterion=nn.CrossEntropyLoss(), amp_mode="off", ema_rate=0.0)
    defaults.update(kwargs)
    return TrainerG_v12(**defaults)


def test_no_per_epoch_files_at_run_root_after_two_epochs(tmp_path):
    t = make_trainer(tmp_path)
    t.fit(max_epochs=2)

    root_files = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert not any(f.startswith("confusion_matrix_epoch") for f in root_files)
    assert not any(f.startswith("classification_report_epoch") for f in root_files)
    assert "confusion_matrix.npy" not in root_files

    cm_dir = tmp_path / "confusion_matrix"
    cr_dir = tmp_path / "classification_reports"
    assert cm_dir.is_dir() and cr_dir.is_dir()
    assert any(p.name.startswith("confusion_matrix_epoch") for p in cm_dir.iterdir())
    assert (cm_dir / "confusion_matrix.npy").is_file()
    assert any(p.name.startswith("classification_report_epoch") for p in cr_dir.iterdir())


def test_dirs_created_at_init(tmp_path):
    t = make_trainer(tmp_path)
    assert t.dirs["confusion_matrix"] == tmp_path / "confusion_matrix"
    assert t.dirs["classification_reports"] == tmp_path / "classification_reports"
    assert t.dirs["confusion_matrix"].is_dir()
    assert t.dirs["classification_reports"].is_dir()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_no_per_epoch_files_at_run_root_after_two_epochs(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_dirs_created_at_init(Path(d))
    print("OK")
