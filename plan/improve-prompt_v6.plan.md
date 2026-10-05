# Comprehensive Implementation Plan to Make the MedMamba Evaluation Scientifically Robust

The latest experiment is a significant improvement over the previous one, but it still contains one major concern: **the model suddenly jumps from poor validation performance to nearly perfect validation performance between epochs 6 and 7** while finishing with **99.74% validation accuracy**. The reconstruction metrics, on the other hand, improve smoothly throughout training. 

Before considering any publication or thesis results, the evaluation pipeline should be expanded so that **every reported metric can be verified visually and statistically**.

---

# Phase 1 — Verify Dataset Integrity (Highest Priority)

## Goal

Ensure there is absolutely **no data leakage** between train, validation, and test.

---

## 1.1 Generate Dataset Split Report

Automatically generate

```text
dataset_split_report.json
```

Contents

```text
Dataset Summary
---------------
Total samples
Total patients
Total slides
Total patches

Train
Validation
Test

Samples per class

Patients per split

Slides per split

Class percentages

Band count

Patch size
```

---

## 1.2 Leakage Detection

Automatically verify

✓ No patient appears in multiple splits

✓ No slide appears in multiple splits

✓ No duplicated patch filename

✓ No duplicated patch hash

✓ No duplicated spectra

Generate

```text
leakage_report.json
```

Example

```text
PASS

No duplicated patients

No duplicated slides

No duplicated spectra
```

Otherwise

```text
FAIL

Patient 18 appears in Train and Validation

Slide 034 duplicated
```

Training should stop if leakage is detected.

---

## 1.3 Dataset Visualization

Generate

```
plots/

class_distribution.png

patient_distribution.png

split_distribution.png

band_distribution.png
```

These immediately reveal imbalance problems.

---

# Phase 2 — Validation Pipeline Verification

Currently the validation curve behaves abnormally.

The validation accuracy evolves approximately as

```
0

50

7

50

45

7

99.7

99.5

99.6

99.6
```

which is not a typical learning curve. 

---

## 2.1 Validation Sanity Checks

Before every validation epoch verify

* model in eval mode
* gradients disabled
* deterministic transforms
* no augmentation
* shuffle=False
* labels unchanged
* class mapping unchanged

If any check fails

```text
WARNING

Validation pipeline inconsistent
```

---

## 2.2 Prediction Distribution

Save

```
prediction_distribution_epochXX.csv
```

containing

```
Ground truth

Prediction

Confidence

Entropy

Probability vector
```

This helps detect

* class collapse
* overconfidence
* uncertainty

---

# Phase 3 — Classification Diagnostics

This is the largest missing component.

---

## 3.1 Confusion Matrix

Every validation epoch

Generate

```
confusion_matrix_epoch_01.png

...

confusion_matrix_epoch_10.png
```

Also save

```
confusion_matrix.npy
```

A normalized confusion matrix should also be produced.

```
confusion_matrix_normalized.png
```

---

## 3.2 Classification Report

Generate

```
classification_report_epochXX.txt
```

Containing

```
Class

Precision

Recall

F1

Support

Macro Average

Weighted Average
```

This immediately explains whether one class is much easier than the others.

---

## 3.3 Per-Class Accuracy

Instead of

```
Validation Accuracy
```

only,

plot

```
Healthy

Tumor

Necrosis
```

accuracy separately.

Example

```
Healthy

99.8%

Tumor

97.2%

Necrosis

92.6%
```

This is far more informative.

---

## 3.4 Class Prediction Histogram

Generate

```
predicted_class_distribution.png
```

Compare

```
Ground Truth

vs

Predictions
```

If one class dominates,

it becomes immediately obvious.

---

# Phase 4 — Confidence Analysis

A model can obtain high accuracy while still being poorly calibrated.

---

## 4.1 Confidence Histogram

Plot

```
Prediction Confidence
```

from

```
0

↓

1
```

Ideal

Most predictions

```
0.9+

```

Wrong predictions

Low confidence.

---

## 4.2 Reliability Diagram

Generate

```
reliability_diagram.png
```

Compute

```
Expected Calibration Error

ECE
```

Also

```
Maximum Calibration Error

MCE
```

---

## 4.3 ROC Curves

For each class

```
ROC Curve

AUC
```

Generate

```
roc_curve_class0.png

roc_curve_class1.png

roc_curve_class2.png
```

Also

```
ROC macro average
```

---

## 4.4 Precision Recall Curves

For imbalanced datasets these are often more informative than ROC.

Generate

```
pr_curve_class0.png

...

macro_pr_curve.png
```

---

# Phase 5 — Training Diagnostics

---

## 5.1 Learning Curves

Plot

```
Train Loss

Validation Loss
```

on one graph.

---

## 5.2 Classification Loss

Separate plot

```
Train Classification Loss

Validation Classification Loss
```

---

## 5.3 Reconstruction Loss

Plot

```
Train Reconstruction

Validation Reconstruction
```

---

## 5.4 Accuracy Curve

Plot

```
Training Accuracy

Validation Accuracy
```

This will immediately expose

* overfitting
* underfitting

---

## 5.5 Learning Rate

Plot cosine schedule

```
Learning Rate

vs

Epoch
```

---

## 5.6 Epoch Time

Plot

```
Epoch Duration
```

Useful for optimization.

---

# Phase 6 — Spectral Reconstruction Diagnostics

Currently only three metrics are reported:

* Reconstruction Loss
* SAM
* RMSE 

Expand significantly.

---

## 6.1 Spectral Metrics

Compute

```
SAM

RMSE

MAE

Cosine Similarity

Pearson Correlation

Spectral Information Divergence

PSNR
```

---

## 6.2 Spectral Curves

Randomly choose validation pixels.

Plot

```
True Spectrum

Generated Spectrum
```

Overlay

Both curves.

Produce

```
spectra_examples/

sample001.png

sample002.png
```

This is one of the strongest visualizations in an HSI paper.

---

## 6.3 SAM Distribution

Instead of

```
Average SAM
```

plot

Histogram

```
Number of pixels

vs

SAM angle
```

---

## 6.4 RMSE Distribution

Histogram

```
RMSE

Distribution
```

Much more informative than a mean.

---

## 6.5 Worst Reconstruction

Automatically save

```
Worst 10 spectra

Best 10 spectra
```

This identifies failure cases.

---

# Phase 7 — Latent Space Visualization

Very valuable for a thesis.

---

## 7.1 t-SNE

Project encoder features into 2D.

Color by

Class.

Generate

```
tsne_epoch10.png
```

---

## 7.2 UMAP

Usually better than t-SNE.

Generate

```
umap_epoch10.png
```

This shows

whether latent features cluster naturally.

---

# Phase 8 — Feature Importance

If wavelength selection exists,

generate

```
Band Importance

Plot
```

Example

```
Importance

^

|

|

+---------------------->

Wavelength
```

Also save

```
Selected wavelengths

Removed wavelengths
```

---

# Phase 9 — Automatic Metric Validation

Every experiment should automatically verify that metrics are internally consistent before saving the report.

Checks should include:

### Classification

* Accuracy ∈ [0, 1]
* Precision ∈ [0, 1]
* Recall ∈ [0, 1]
* F1 ∈ [0, 1]
* Macro F1 ∈ [0, 1]
* Balanced Accuracy ∈ [0, 1]

### Consistency

Warn if:

* Precision is exactly 1.0 for many epochs.
* Balanced Accuracy is identical to Accuracy for every epoch.
* Validation Accuracy jumps by more than 30 percentage points in one epoch.
* Validation Loss increases while Accuracy also increases substantially.
* Training Accuracy exceeds Validation Accuracy by more than a configurable threshold (e.g., 10%).

These warnings should not stop training but should be highlighted in the final report.

---

# Phase 10 — Experiment Report Redesign

Replace the current compact report with a structured experiment directory.

```
experiment/
│
├── config.json
├── experiment_report.json
├── history.csv
├── metrics.csv
│
├── leakage_report.json
├── dataset_split_report.json
├── band_selection_report.json
│
├── confusion_matrices/
├── classification_reports/
├── roc_curves/
├── pr_curves/
├── spectra_examples/
├── latent_space/
├── plots/
│
├── best_model.pt
└── logs/
```

---

# Phase 11 — Final Experiment Summary

Automatically generate a human-readable summary (Markdown or PDF) containing:

* **Dataset Summary:** samples, patients, slides, class distribution, selected wavelengths.
* **Training Summary:** best epoch, total epochs, training time, convergence plots.
* **Classification Performance:** overall metrics, per-class metrics, confusion matrix, ROC/PR curves, calibration results.
* **Spectral Reconstruction:** reconstruction loss, SAM, RMSE, MAE, SID, example spectra, metric distributions.
* **Feature Analysis:** wavelength importance, latent-space visualizations (t-SNE/UMAP).
* **Validation Checks:** leakage report, metric consistency checks, warnings encountered during training.
* **Conclusions:** best model checkpoint, reproducibility information (seed, configuration, Git commit if available), and recommended next steps.

This transforms each training run into a **complete, publication-quality experiment package** that not only reports high-level metrics but also provides the evidence needed to verify their correctness, diagnose failures, and support the scientific claims made in your thesis.
