# -*- coding: utf-8 -*-
"""
training/plots_v16.py
========================
MedMamba-SS-TRM v16 plan, Stage 2.6 (R-6). `training/plots.py` is imported at
`train_example_v4.py` (and therefore every trainer since) and is FROZEN.
Redefines only `generate_all_training_plots`, importing every individual
`plot_*` primitive from the frozen module unchanged.

The one change: the reconstruction-loss gate at `plots.py:231`,
`any(r.get("MSE_loss") is not None for r in history_rows)`, is always true -
`MSE_loss` is always a float (`0.0` when `decoder is None`), never `None` -
so every run, reconstruction-enabled or not, got a meaningless flat-zero
`reconstruction_loss.png` beside an empty `reconstruction/` directory. This
uses the non-zero test the frozen module already applies to SAM
(`plots.py:75`, `any(v not in (None, 0) for v in series)`) instead.
"""

from __future__ import annotations

from typing import Dict, List

from training.plots import (  # frozen, imported not edited
    plot_loss_curve, plot_accuracy_curve, plot_lr_curve, plot_gpu_memory, plot_epoch_time,
    plot_classification_loss_curve, plot_reconstruction_loss_curve,
    plot_gradient_norm, plot_parameter_norm, plot_update_norm, plot_generalization_gap,
)

__all__ = ["generate_all_training_plots"]


def generate_all_training_plots(history_rows: List[Dict], out_dir: str) -> None:
    if not history_rows:
        return
    epochs = [r["epoch"] for r in history_rows]
    plot_loss_curve(epochs, [r.get("train_loss") for r in history_rows],
                     [r.get("val_loss") for r in history_rows], out_dir)
    plot_accuracy_curve(epochs, [r.get("train_acc") for r in history_rows],
                         [r.get("val_acc") for r in history_rows], out_dir)
    if any("lr" in r for r in history_rows):
        plot_lr_curve(epochs, [r.get("lr") for r in history_rows], out_dir)
    if any(r.get("peak_gpu_memory_mb") for r in history_rows):
        plot_gpu_memory(epochs, [r.get("peak_gpu_memory_mb", 0) for r in history_rows], out_dir)
    if any("epoch_time_s" in r for r in history_rows):
        plot_epoch_time(epochs, [r.get("epoch_time_s") for r in history_rows], out_dir)

    if any(r.get("classification_loss") is not None for r in history_rows):
        plot_classification_loss_curve(epochs, [r.get("classification_loss") for r in history_rows],
                                        [r.get("val_cls_loss") for r in history_rows], out_dir)

    # R-6 - the non-zero gate. Reconstruction was force-disabled on every
    # audited run (R-1), so MSE_loss/val_mse_loss were 0.0 for every epoch -
    # a flat-zero plot is not a reconstruction curve, it is a screenshot of
    # the bug. Only plot it once there is something to show.
    mse_series = [r.get("MSE_loss") for r in history_rows]
    val_mse_series = [r.get("val_mse_loss") for r in history_rows]
    if any(v not in (None, 0) for v in mse_series) or any(v not in (None, 0) for v in val_mse_series):
        plot_reconstruction_loss_curve(
            epochs, mse_series, val_mse_series,
            train_sam=[r.get("SAM_loss") for r in history_rows],
            val_sam=[r.get("val_sam_loss") for r in history_rows],
            out_dir=out_dir)

    if any(r.get("gradient_norm") is not None for r in history_rows):
        plot_gradient_norm(epochs, [r.get("gradient_norm") for r in history_rows], out_dir)
    if any(r.get("parameter_norm") is not None for r in history_rows):
        plot_parameter_norm(epochs, [r.get("parameter_norm") for r in history_rows], out_dir)
    if any(r.get("update_norm") is not None for r in history_rows):
        plot_update_norm(epochs, [r.get("update_norm") for r in history_rows], out_dir)
    if any(r.get("generalization_gap_loss") is not None for r in history_rows):
        plot_generalization_gap(epochs, [r.get("generalization_gap_loss") for r in history_rows],
                                 [r.get("generalization_gap_accuracy") for r in history_rows], out_dir)
