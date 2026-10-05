# -*- coding: utf-8 -*-
"""
training/plots.py
==================
Phase 8 (loss/accuracy plots) of the .npy plan. Uses matplotlib with the
non-interactive "Agg" backend so this works headless on training servers.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _line_plot(x, series: Dict[str, Sequence[float]], title: str, xlabel: str, ylabel: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, y in series.items():
        ax.plot(x, y, label=label, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curve(epochs: List[int], train_loss: List[float], val_loss: List[float], out_dir: str):
    _line_plot(epochs, {"train": train_loss, "val": val_loss}, "Loss", "epoch", "loss",
               os.path.join(out_dir, "loss.png"))


def plot_accuracy_curve(epochs: List[int], train_acc: List[float], val_acc: List[float], out_dir: str):
    _line_plot(epochs, {"train": train_acc, "val": val_acc}, "Accuracy", "epoch", "accuracy",
               os.path.join(out_dir, "accuracy.png"))


def plot_lr_curve(epochs: List[int], lr: List[float], out_dir: str):
    _line_plot(epochs, {"lr": lr}, "Learning rate", "epoch", "lr", os.path.join(out_dir, "lr.png"))


def plot_gpu_memory(epochs: List[int], gpu_mb: List[float], out_dir: str):
    if not any(v for v in gpu_mb):
        return
    _line_plot(epochs, {"peak GPU mem (MB)": gpu_mb}, "Peak GPU memory", "epoch", "MB",
               os.path.join(out_dir, "gpu_memory.png"))


def plot_epoch_time(epochs: List[int], epoch_time_s: List[float], out_dir: str):
    _line_plot(epochs, {"epoch time (s)": epoch_time_s}, "Epoch time", "epoch", "seconds",
               os.path.join(out_dir, "epoch_time.png"))


# ============================================================================
# improve-promp_v8.plan.md, Phase 0.4 - Training Dynamics Diagnostics
# ============================================================================

def plot_classification_loss_curve(epochs: List[int], train_cls: List[float], val_cls: List[float],
                                    out_dir: str):
    _line_plot(epochs, {"train": train_cls, "val": val_cls}, "Classification loss (CE)",
               "epoch", "loss", os.path.join(out_dir, "classification_loss.png"))


def plot_reconstruction_loss_curve(epochs: List[int], train_mse: List[float], val_mse: List[float],
                                    train_sam: Optional[List[float]] = None,
                                    val_sam: Optional[List[float]] = None, out_dir: str = "."):
    series = {"train MSE": train_mse, "val MSE": val_mse}
    if train_sam is not None and any(v not in (None, 0) for v in train_sam):
        series["train SAM"] = train_sam
        series["val SAM"] = val_sam
    _line_plot(epochs, series, "Reconstruction loss", "epoch", "loss",
               os.path.join(out_dir, "reconstruction_loss.png"))


def plot_gradient_norm(epochs: List[int], grad_norm: List[float], out_dir: str):
    if not any(v for v in grad_norm):
        return
    _line_plot(epochs, {"gradient norm": grad_norm}, "Gradient norm", "epoch", "L2 norm",
               os.path.join(out_dir, "gradient_norm.png"))


def plot_parameter_norm(epochs: List[int], param_norm: List[float], out_dir: str):
    if not any(v for v in param_norm):
        return
    _line_plot(epochs, {"parameter norm": param_norm}, "Parameter norm (post-epoch)", "epoch",
               "L2 norm", os.path.join(out_dir, "parameter_norm.png"))


def plot_update_norm(epochs: List[int], update_norm: List[float], out_dir: str):
    if not any(v for v in update_norm):
        return
    _line_plot(epochs, {"update norm": update_norm}, "Parameter update norm (per epoch)", "epoch",
               "L2 norm", os.path.join(out_dir, "update_norm.png"))


def plot_generalization_gap(epochs: List[int], gap_loss: List[float], gap_acc: List[float], out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, gap_loss, marker="o", markersize=3, color="crimson")
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Generalization gap (val_loss - train_loss)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss gap")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, gap_acc, marker="o", markersize=3, color="steelblue")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Generalization gap (train_acc - val_acc)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy gap")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "generalization_gap.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]], out_dir: str,
                           normalize: bool = False):
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    n = cm.shape[0]
    names = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]
    fig_size = max(4.5, 0.5 * n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (normalized)" if normalize else ""))
    fmt = "{:.2f}" if normalize else "{:.0f}"
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, fmt.format(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    suffix = "_normalized" if normalize else ""
    fig.savefig(os.path.join(out_dir, f"confusion_matrix{suffix}.png"), dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(probs: np.ndarray, labels: np.ndarray, num_classes: int,
                        class_names: Optional[List[str]], out_dir: str):
    """Macro one-vs-rest ROC and PR curves for all classes on one figure each."""
    names = class_names if class_names and len(class_names) == num_classes else [str(i) for i in range(num_classes)]

    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 6))
    for c in range(num_classes):
        y_bin = (labels == c).astype(np.int64)
        scores = probs[:, c]
        order = np.argsort(-scores)
        y_sorted = y_bin[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        n_pos, n_neg = y_bin.sum(), len(y_bin) - y_bin.sum()
        if n_pos == 0 or n_neg == 0:
            continue
        tpr = tp / n_pos
        fpr = fp / n_neg
        ax_roc.plot(fpr, tpr, label=names[c], linewidth=1.2)

        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / n_pos
        ax_pr.plot(recall, precision, label=names[c], linewidth=1.2)

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curves (one-vs-rest)")
    ax_roc.legend(fontsize=7, loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig_roc)

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curves (one-vs-rest)")
    ax_pr.legend(fontsize=7, loc="lower left")
    fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(out_dir, "precision_recall_curve.png"), dpi=150)
    plt.close(fig_pr)


def plot_band_importance(band_importance: List[float], wavelengths: Optional[List[float]],
                          selected_indices: List[int], out_dir: str):
    n = len(band_importance)
    x = wavelengths if wavelengths is not None and len(wavelengths) == n else list(range(n))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, band_importance, width=(max(x) - min(x)) / n * 0.9 if wavelengths else 0.9, color="steelblue")
    if selected_indices:
        sel_x = [x[i] for i in selected_indices]
        sel_y = [band_importance[i] for i in selected_indices]
        ax.scatter(sel_x, sel_y, color="crimson", zorder=3, s=15, label="top-k selected")
        ax.legend()
    ax.set_xlabel("wavelength (nm)" if wavelengths else "band index")
    ax.set_ylabel("mean band weight")
    ax.set_title("Band importance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "band_importance.png"), dpi=150)
    plt.close(fig)


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

    # Phase 0.4 - Training Dynamics Diagnostics
    if any(r.get("classification_loss") is not None for r in history_rows):
        plot_classification_loss_curve(epochs, [r.get("classification_loss") for r in history_rows],
                                        [r.get("val_cls_loss") for r in history_rows], out_dir)
    if any(r.get("MSE_loss") is not None for r in history_rows):
        plot_reconstruction_loss_curve(
            epochs, [r.get("MSE_loss") for r in history_rows], [r.get("val_mse_loss") for r in history_rows],
            train_sam=[r.get("SAM_loss") for r in history_rows], val_sam=[r.get("val_sam_loss") for r in history_rows],
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


# ============================================================================
# Main Development Plan, Phase 4 - Reconstruction Visualizations
# ============================================================================

def plot_reconstruction_examples(x_true: np.ndarray, x_pred: np.ndarray, wavelengths: Optional[np.ndarray],
                                  sam_per_sample: Sequence[float], out_dir: str, n_each: int = 3) -> None:
    """Original vs. reconstructed spectra for the best/worst/median
    reconstructions, ranked by per-sample SAM (lower = better). `x_true` /
    `x_pred` are [N, C] (one averaged spectrum per patch - e.g. the spatial
    mean over H,W of a [C,H,W] patch)."""
    n = len(sam_per_sample)
    if n == 0:
        return
    order = np.argsort(sam_per_sample)
    x_axis = wavelengths if wavelengths is not None and len(wavelengths) == x_true.shape[1] else np.arange(x_true.shape[1])

    groups = {
        "best": order[:n_each],
        "worst": order[::-1][:n_each],
        "median": order[max(0, n // 2 - n_each // 2): max(0, n // 2 - n_each // 2) + n_each],
    }
    for group_name, idxs in groups.items():
        if len(idxs) == 0:
            continue
        fig, axes = plt.subplots(len(idxs), 1, figsize=(7, 2.6 * len(idxs)), squeeze=False)
        for row, i in enumerate(idxs):
            ax = axes[row, 0]
            ax.plot(x_axis, x_true[i], label="original", color="black", linewidth=1.3)
            ax.plot(x_axis, x_pred[i], label="reconstructed", color="crimson", linewidth=1.1, linestyle="--")
            ax.set_title(f"sample {int(i)}  (SAM={sam_per_sample[i]:.2f} deg)", fontsize=9)
            ax.set_xlabel("wavelength (nm)" if wavelengths is not None else "band index")
            ax.set_ylabel("reflectance")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8)
        fig.suptitle(f"{group_name.capitalize()} reconstructions", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"reconstruction_{group_name}.png"), dpi=150)
        plt.close(fig)


def plot_reconstruction_metric_histograms(per_sample_metrics: Dict[str, Sequence[float]], out_dir: str) -> None:
    """Histograms of SAM / RMSE / MAE / SID / Peak Position Error across all
    validation/test samples (Main Development Plan, Phase 4)."""
    labels = {
        "sam_deg": "SAM (deg)", "rmse": "RMSE", "mae": "MAE",
        "sid": "SID", "peak_position_error": "Peak Position Error",
    }
    for key, title in labels.items():
        values = per_sample_metrics.get(key)
        if not values:
            continue
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(float(np.mean(values)), color="crimson", linestyle="--", linewidth=1,
                    label=f"mean={np.mean(values):.3f}")
        ax.axvline(float(np.median(values)), color="darkorange", linestyle="--", linewidth=1,
                    label=f"median={np.median(values):.3f}")
        ax.set_title(f"{title} distribution")
        ax.set_xlabel(title)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"hist_{key}.png"), dpi=150)
        plt.close(fig)


# ============================================================================
# Main Development Plan, Phase 6 - Confidence Analysis
# ============================================================================

def plot_confidence_histogram(confidences: np.ndarray, correct: np.ndarray, out_dir: str, n_bins: int = 20) -> None:
    confidences = np.asarray(confidences)
    correct = np.asarray(correct).astype(bool)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, n_bins + 1)
    ax.hist(confidences[correct], bins=bins, alpha=0.6, label="correct", color="seagreen")
    ax.hist(confidences[~correct], bins=bins, alpha=0.6, label="incorrect", color="crimson")
    ax.set_xlabel("predicted confidence (max softmax prob)")
    ax.set_ylabel("count")
    ax.set_title("Confidence histogram")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confidence_histogram.png"), dpi=150)
    plt.close(fig)


def plot_reliability_diagram(confidences: np.ndarray, correct: np.ndarray, out_dir: str, n_bins: int = 15) -> None:
    """Reliability diagram (accuracy vs. confidence per bin) plus the
    calibration curve overlaid against the perfect-calibration diagonal."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_acc, bin_conf, bin_count = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_acc.append(float(correct[mask].mean()))
        bin_conf.append(float(confidences[mask].mean()))
        bin_count.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.bar(bin_centers, bin_acc, width=1.0 / n_bins * 0.9, color="steelblue", edgecolor="black",
           alpha=0.8, label="accuracy per bin")
    ax.plot(bin_conf, bin_acc, color="crimson", marker="o", markersize=3, linewidth=1,
            label="calibration curve")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram / calibration curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability_diagram.png"), dpi=150)
    plt.close(fig)


def plot_prediction_entropy(probs: np.ndarray, out_dir: str, eps: float = 1e-12) -> None:
    entropy = -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(entropy, bins=40, color="mediumpurple", edgecolor="white")
    ax.axvline(float(entropy.mean()), color="crimson", linestyle="--", linewidth=1,
                label=f"mean={entropy.mean():.3f}")
    ax.set_xlabel("prediction entropy (nats)")
    ax.set_ylabel("count")
    ax.set_title("Prediction entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prediction_entropy.png"), dpi=150)
    plt.close(fig)


def generate_confidence_analysis(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray, out_dir: str) -> None:
    """Convenience wrapper: confidence histogram + reliability diagram +
    calibration curve + prediction entropy in one call (Phase 6)."""
    os.makedirs(out_dir, exist_ok=True)
    confidences = probs.max(axis=1)
    correct = (preds == labels)
    plot_confidence_histogram(confidences, correct, out_dir)
    plot_reliability_diagram(confidences, correct, out_dir)
    plot_prediction_entropy(probs, out_dir)


# ============================================================================
# Main Development Plan, Phase 6 - Latent Space (t-SNE / UMAP)
# ============================================================================

def plot_latent_space(features: np.ndarray, labels: np.ndarray, class_names: Optional[List[str]],
                       out_dir: str, methods: Sequence[str] = ("tsne", "umap"), seed: int = 0) -> Dict[str, str]:
    """2D projections of penultimate-layer features, colored by class.
    `features` is [N, D]. Skips a method gracefully (with a note printed,
    no crash) if its dependency isn't installed. Returns {method: path}."""
    os.makedirs(out_dir, exist_ok=True)
    n = features.shape[0]
    names = class_names if class_names else [str(c) for c in sorted(set(labels.tolist()))]
    written = {}

    def _scatter(coords, title, fname):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        classes = sorted(set(labels.tolist()))
        cmap = plt.get_cmap("tab20" if len(classes) > 10 else "tab10")
        for ci, c in enumerate(classes):
            mask = labels == c
            label_str = names[c] if c < len(names) else str(c)
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.7,
                       color=cmap(ci % cmap.N), label=label_str)
        ax.set_title(title)
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    if "tsne" in methods:
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, n // 4))
            coords = TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                          init="pca", learning_rate="auto").fit_transform(features)
            written["tsne"] = _scatter(coords, "t-SNE of latent features", "latent_tsne.png")
        except ImportError:
            print("  [latent-space] scikit-learn not available - skipping t-SNE plot.")
        except Exception as e:
            print(f"  [latent-space] t-SNE failed ({e}) - skipping.")

    if "umap" in methods:
        try:
            import umap
            coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(features)
            written["umap"] = _scatter(coords, "UMAP of latent features", "latent_umap.png")
        except ImportError:
            print("  [latent-space] umap-learn not installed - skipping UMAP plot "
                  "(pip install umap-learn to enable).")
        except Exception as e:
            print(f"  [latent-space] UMAP failed ({e}) - skipping.")

    return written


# ============================================================================
# Main Development Plan, Phase 2 - Class Balancing distribution plots
# ============================================================================

def plot_class_distribution(class_counts: Sequence[int], class_names: Optional[List[str]],
                             out_path: str, title: str = "Class distribution") -> None:
    n = len(class_counts)
    names = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * n), 4.5))
    bars = ax.bar(range(n), class_counts, color="steelblue")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("number of samples")
    ax.set_title(title)
    for b, c in zip(bars, class_counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(c)),
                ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
