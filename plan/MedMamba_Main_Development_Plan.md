# MedMamba Main Development Plan
**Version:** 2.0
**Status:** Primary Development Roadmap

---

# Goals

The primary objectives of this project are to:

- Improve classification performance.
- Preserve hyperspectral spectral information.
- Produce scientifically valid reconstruction metrics.
- Build a reproducible preprocessing and evaluation pipeline.
- Provide publication-quality experiment reports.
- Preserve the physical interpretability of hyperspectral wavelengths by using wavelength importance rather than PCA.

---

# Phase 1 — Dataset Preparation

## Objectives

- Reduce spectral redundancy.
- Keep only informative wavelengths.
- Improve preprocessing speed.
- Reduce GPU memory usage.
- Allow reproducible wavelength selection experiments.

---

# 1.1 Spectral Band Selection

Implement wavelength importance ranking.

New CLI options

```bash
--num_bands
--importance_threshold
--max_bands
```

## Parameter Behavior

### --importance_threshold

Minimum normalized wavelength importance.

Example

```bash
--importance_threshold 0.95
```

Only wavelengths above the threshold are retained.

---

### --num_bands

Maximum number of informative bands to keep.

Example

```bash
--num_bands 64
```

If omitted:

Use all informative wavelengths.

---

### --max_bands

Upper limit for evaluation experiments.

Purpose:

Prevent selecting an excessively large number of wavelengths while performing automated experiments.

Example

```bash
--max_bands 128
```

Behavior

Suppose

```
Original bands = 275

Threshold selected = 210

num_bands = None

max_bands = 128
```

Final output

```
128 highest-ranked informative bands
```

If

```
Threshold selected = 61

max_bands = 128
```

Final output

```
61 bands
```

The cap is never exceeded.

---

# Selection Algorithm

1. Compute wavelength importance.
2. Normalize scores.
3. Rank wavelengths.
4. Remove wavelengths below importance threshold.
5. Apply Option A:

If requested number exceeds informative wavelengths:

```
Requested 64

Found 51 informative

Using all 51 informative bands.
```

6. Apply max_bands cap.
7. Save selected wavelengths.

---

# Metadata

Generate

```
selected_wavelengths.npy

band_importance.npy

band_ranking.npy

band_selection_report.json
```

---

# Phase 2 — Class Balancing

New CLI option

```bash
--balance_classes {none,undersample,oversample}
```

Balancing is applied

- ONLY to Training
- AFTER all preprocessing
- AFTER train/validation/test split
- AFTER shard unification

Validation and Test remain untouched.

Generate

```
class_distribution_before.png

class_distribution_after.png

balancing_report.json
```

---

# Phase 3 — Improved MedMamba Architecture

Architecture

Input HSI

↓

Band Selection

↓

Patch Embedding

↓

MedMamba Encoder

├────────► Classification Head

└────────► Spectral Generator

↓

Reconstructed Spectrum

↓

GAN Discriminator (optional)

---

# Loss

```
L =
CE
+ λMSE
+ λSAM
+ λGAN
```

Configurable weights.

---

# Phase 4 — Spectral Reconstruction

Reconstruct

Original spectral vectors

instead of latent features.

---

# Reconstruction Metrics

Compute

- SAM
- RMSE
- MAE
- SID
- Pearson Correlation
- Cosine Similarity
- Peak Position Error

Optional

- SSIM
- PSNR

---

# Reconstruction Visualizations

Generate

- Original vs reconstructed spectra
- Best reconstructions
- Worst reconstructions
- Median reconstructions

Histograms

- SAM
- RMSE
- MAE
- SID
- Peak Position Error

---

# Phase 5 — Classification Evaluation

Overall

- OA
- Balanced Accuracy
- Precision
- Recall
- F1
- Macro F1
- Weighted F1
- Cohen's Kappa
- MCC

Per-class

- Precision
- Recall
- F1
- Support
- Accuracy

---

# Phase 6 — Visualization Suite

Generate

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

---

## ROC

Per-class ROC

Macro ROC

---

## Precision Recall Curves

Per-class

Macro

---

## Confidence Analysis

- Confidence histogram
- Reliability diagram
- Calibration curve
- Prediction entropy

---

## Latent Space

- t-SNE
- UMAP

---

## Band Importance

Plot wavelength importance.

Highlight selected wavelengths.

---

# Phase 7 — Dataset Integrity

Automatically verify

- No duplicated patients
- No duplicated slides
- No duplicated spectra
- No duplicated patches
- No leakage

Generate

```
dataset_split_report.json

leakage_report.json
```

Abort training if leakage exists.

---

# Phase 8 — Metric Validation

Automatically detect

- NaN metrics
- Infinite metrics
- Constant precision
- Impossible F1
- Balanced Accuracy inconsistencies
- Sudden validation jumps
- Missing classes
- Empty predictions

Generate

```
validation_report.json
```

---

# Phase 9 — Training Diagnostics

Log

- GPU memory
- Gradient norm
- Epoch time
- Learning rate
- Classification loss
- Reconstruction loss
- GAN loss
- Batch throughput

---

# Phase 10 — Experiment Reports

Every experiment should generate

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
│
├── confusion_matrices/
├── classification_reports/
├── roc_curves/
├── pr_curves/
├── reconstruction/
├── spectra_examples/
├── latent_space/
├── plots/
│
├── best_model.pt
└── logs/
```

---

# Phase 11 — Ablation Studies

Evaluate

## Band Count

- 16
- 25
- 32
- 64
- 96
- 128
- All informative

## Threshold

- 0.90
- 0.95
- 0.98

## Reconstruction

- CE
- CE+MSE
- CE+MSE+SAM
- CE+MSE+SAM+GAN

Compare

- OA
- Kappa
- MCC
- Macro F1
- SAM
- RMSE
- MAE
- SID
- Training Time
- GPU Memory