# GMedMamba V7 Stabilization & Evaluation Master Plan

# Objective

The next update is **not** intended to improve accuracy by adding new model features.

Its purpose is to make the entire training and evaluation pipeline **scientifically correct, reproducible, and publication-ready**.

No new architectural experiments should be added until every item below is implemented and verified.

---

# Phase 1 — Fix Validation Pipeline

## Goal

Guarantee that validation metrics are mathematically correct and comparable between experiments.

---

## 1.1 Verify Validation Mode

Before every validation epoch ensure

```python
model.eval()

torch.no_grad()
```

Verify

- Dropout disabled
- BatchNorm frozen
- EMA (if enabled) switched correctly
- No gradients accumulated

---

## 1.2 Verify Validation Dataset

Ensure validation

- contains no augmentations
- uses identical preprocessing
- identical normalization
- identical band ordering
- identical PCA (if enabled)

Only training may perform

- flips
- rotations
- random crops
- spectral jitter
- MixUp
- CutMix

Validation must remain deterministic.

---

## 1.3 Verify Data Leakage

Automatically verify

- no duplicated samples
- no duplicated filenames
- no duplicated patient IDs
- no duplicated image IDs
- no duplicated coordinates
- no duplicated spectra

Generate

```
leakage_report.json
```

---

## 1.4 Verify Class Distribution

Generate

```
dataset_statistics.json
```

containing

- class counts
- percentages
- imbalance ratio
- train distribution
- validation distribution
- test distribution

---

# Phase 2 — Rewrite Metric Pipeline

Current metrics show impossible combinations such as

```
Precision = 1.0
Accuracy = 75%
Macro F1 = 0.28
```

These indicate metric computation bugs.

The entire evaluation module must be rewritten.

---

## 2.1 Classification Metrics

Every validation epoch must compute

### Overall

- Overall Accuracy (OA)
- Top-1 Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted Precision
- Weighted Recall
- Weighted F1
- Micro Precision
- Micro Recall
- Micro F1

---

### Statistical Metrics

- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa

---

### Per-Class Metrics

For every class compute

- Precision
- Recall
- F1
- Support
- Accuracy

Generate

```
classification_report.json
```

and

```
classification_report.csv
```

---

## 2.2 Confusion Matrices

Generate

- Raw confusion matrix
- Normalized confusion matrix

Save as

```
PNG
PDF
CSV
NumPy
```

---

## 2.3 ROC Metrics

Compute

Per class

- ROC
- AUC

Global

- Macro ROC
- Micro ROC

---

## 2.4 Precision Recall Metrics

Generate

Per class

- Precision–Recall Curve
- Average Precision

Global

- Macro PR
- Micro PR

---

## 2.5 Calibration Metrics

Compute

- Expected Calibration Error (ECE)
- Maximum Calibration Error
- Brier Score

Generate

- Reliability diagram
- Calibration curve

---

## 2.6 Confidence Statistics

Generate

- Confidence histogram
- Confidence distribution
- Prediction entropy
- Softmax entropy
- Confidence vs accuracy

---

# Phase 3 — Expand Spectral Reconstruction Metrics

Current metrics

- RMSE
- SAM

are insufficient.

Every validation epoch must additionally compute

---

## Reconstruction Losses

- Reconstruction Loss
- MAE
- RMSE
- MSE

---

## Spectral Metrics

- Spectral Angle Mapper (SAM)
- Spectral Information Divergence (SID)
- Spectral Correlation Coefficient (SCC)
- Cosine Similarity
- Pearson Correlation
- Peak Position Error

---

## Image Quality Metrics

- PSNR
- SSIM

---

Generate

```
reconstruction_metrics.json
```

and

```
reconstruction_metrics.csv
```

---

# Phase 4 — Validation Stability

Current validation loss oscillates excessively.

Implement

---

## Label Smoothing

Configurable

```
0.00
0.02
0.05
0.10
```

---

## Gradient Monitoring

Track

- gradient norm
- exploding gradients
- NaN gradients

Abort training if

```
gradient_norm > threshold
```

---

## Logit Monitoring

Track

- average confidence
- maximum confidence
- logit magnitude

Detect overconfident predictions.

---

## Learning Rate Logging

Every epoch save

```
learning_rate
```

Already partially implemented.

Verify correctness.

---

# Phase 5 — Early Stopping Rewrite

Current divergence detection is overly aggressive.

Replace with

---

## Multi-Criteria Early Stopping

Monitor

- validation loss
- macro F1
- MCC
- OA

Use configurable patience

Example

```
patience = 10
```

---

## Minimum Improvement

```
min_delta
```

should be configurable.

---

## Divergence Detection

Stop only when

- validation loss continuously worsens
- AND accuracy decreases
- AND macro F1 decreases

Never stop because of a single noisy epoch.

---

# Phase 6 — Visualization Suite

Automatically generate every experiment.

---

## Confusion Matrix

- Standard
- Normalized

---

## Training Curves

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Classification Loss
- Reconstruction Loss
- Learning Rate
- Macro F1
- MCC
- OA
- Cohen's Kappa
- Balanced Accuracy
- Precision
- Recall

---

## Spectral Metrics Curves

- RMSE
- MAE
- SAM
- SID
- SCC
- PSNR
- SSIM
- Pearson Correlation
- Cosine Similarity
- Peak Position Error

---

## ROC

Per-class ROC

Macro ROC

Micro ROC

---

## Precision Recall Curves

Per class

Macro

Micro

---

## Confidence Analysis

- Confidence histogram
- Reliability diagram
- Calibration curve
- Prediction entropy
- Confidence vs Accuracy

---

## Latent Space

Generate embeddings using

- t-SNE
- UMAP

Color by

- class
- dataset split
- prediction correctness

---

## Band Importance

Visualize

- wavelength importance
- selected wavelengths
- attention importance
- average spectral response

Highlight retained wavelengths after selection.

---

## Spectral Reconstruction

Generate

- Original spectrum
- Reconstructed spectrum
- Error spectrum
- Residual spectrum

for representative samples.

---

# Phase 7 — Experiment Report Expansion

Every experiment must automatically produce

```
experiment/

├── config.json
├── experiment_report.json
├── history.csv
├── metrics.csv
├── dataset_statistics.json
├── leakage_report.json
├── validation_report.json
├── band_selection_report.json
├── reconstruction_metrics.json
├── reconstruction_metrics.csv
│
├── confusion_matrices/
│   ├── raw.png
│   ├── normalized.png
│   ├── raw.csv
│   └── normalized.csv
│
├── classification_reports/
│   ├── report.json
│   ├── report.csv
│   └── per_class_metrics.csv
│
├── roc_curves/
│
├── pr_curves/
│
├── calibration/
│   ├── reliability.png
│   ├── calibration_curve.png
│   └── confidence_histogram.png
│
├── reconstruction/
│
├── spectra_examples/
│
├── latent_space/
│   ├── tsne.png
│   └── umap.png
│
├── band_selection/
│
├── plots/
│   ├── losses.png
│   ├── accuracies.png
│   ├── learning_rate.png
│   ├── reconstruction_metrics.png
│   ├── classification_metrics.png
│   └── training_summary.png
│
├── best_model.pt
├── final_model.pt
└── logs/
```

---

# Phase 8 — Experiment Report Improvements

`experiment_report.json` must become the complete summary of the experiment.

Include

## Completion

- stop reason
- epochs completed
- best epoch
- training time
- total parameters
- trainable parameters

---

## Best Metrics

- OA
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- MCC
- Cohen's Kappa
- AUROC
- AUPRC

---

## Reconstruction Summary

- RMSE
- MAE
- PSNR
- SSIM
- SAM
- SID
- SCC
- Pearson
- Cosine Similarity
- Peak Position Error

---

## Dataset Summary

- number of classes
- samples
- train
- validation
- test
- imbalance ratio

---

## Model Summary

- parameter count
- FLOPs (optional)
- model size
- inference speed

---

## Hardware Summary

- GPU
- CUDA version
- PyTorch version
- mixed precision enabled
- batch size

---

# Phase 9 — Validation Report

Generate

```
validation_report.json
```

containing

- class distribution
- per-class accuracy
- confusion matrix statistics
- hardest classes
- easiest classes
- calibration statistics
- failure analysis
- confidence statistics
- prediction entropy statistics

---

# Phase 10 — Quality Assurance

Training must fail immediately if

- NaN loss
- Infinite loss
- NaN gradients
- Missing metrics
- Missing plots
- Missing experiment files
- Invalid confusion matrix
- Metric dimensions mismatch
- Duplicate validation samples

Generate explicit error messages.

Silent failures are not acceptable.

---

# Phase 11 — Definition of Done (DoD)

The update is **not considered complete** until every experiment automatically:

✅ Produces mathematically correct classification metrics.

✅ Produces mathematically correct spectral reconstruction metrics.

✅ Generates all required plots without manual intervention.

✅ Saves every report in a structured experiment directory.

✅ Produces reproducible results across repeated runs.

✅ Detects dataset leakage and invalid splits.

✅ Computes all requested metrics:
- OA
- Balanced Accuracy
- Precision (Macro/Weighted/Micro)
- Recall (Macro/Weighted/Micro)
- F1 (Macro/Weighted/Micro)
- MCC
- Cohen's Kappa
- ROC/AUC
- Precision–Recall/AUPRC
- PSNR
- SSIM
- RMSE
- MAE
- SAM
- SID
- SCC
- Pearson Correlation
- Cosine Similarity
- Peak Position Error

✅ Produces a complete publication-quality experiment package suitable for thesis figures, statistical analysis, reproducibility, and future benchmarking.