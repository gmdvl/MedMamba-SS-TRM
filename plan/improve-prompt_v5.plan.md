# Comprehensive Implementation Plan for the MedMamba Pipeline

This plan focuses on three major objectives:

1. **Improve hyperspectral preprocessing** through intelligent wavelength selection.
2. **Strengthen the MedMamba architecture** while preserving spectral reconstruction quality.
3. **Fix the entire evaluation pipeline** so all reported metrics are scientifically correct and reproducible.

The implementation should be performed in phases so that each modification can be validated independently.

---

# Phase 1 — Improve Dataset Preparation

## Goal

Instead of always keeping every wavelength, automatically retain only the informative spectral bands while allowing the user to control the behavior.

---

# 1. Add Two New CLI Arguments

Instead of only supporting

```bash
--num_bands
```

support both

```bash
--num_bands
--importance_threshold
```

Example

```bash
python prepare_histologyhsi_bc_new.py \
    --num_bands 64 \
    --importance_threshold 0.95
```

Neither parameter should be mandatory.

---

## Parameter behavior

### --num_bands

Purpose

Maximum number of wavelengths to keep.

Example

```bash
--num_bands 64
```

means

> Never output more than 64 bands.

---

### --importance_threshold

Purpose

Minimum normalized importance score for a wavelength.

Example

```bash
--importance_threshold 0.95
```

means

Keep only wavelengths whose normalized importance ≥ 95%.

---

# 2. Band Selection Logic

The algorithm should always operate in the following order.

## Step 1

Load the full HSI cube.

```
Original bands
↓

Calculate importance score
↓

Normalize importance
↓

Rank wavelengths
```

---

## Step 2

Remove low-information wavelengths

```
importance >= threshold
```

Suppose

```
275 bands
↓

61 informative
```

Only these remain.

---

## Step 3

Apply Option A

If

```
num_bands = 64

informative = 61
```

Return

```
61 bands
```

Print

```
Requested 64 bands.

Only 61 informative bands exceeded the threshold.

Using all informative bands.
```

Never invent additional wavelengths.

---

## Step 4

If

```
num_bands = 32

informative = 61
```

Keep

```
Top 32
```

according to ranking.

---

## Step 5

If

```
No threshold provided
```

Then

```
Keep top N bands
```

---

## Step 6

If

```
No num_bands provided
```

Then

Keep

```
All informative bands
```

This is much more intuitive than forcing a default number.

---

# 3. Remove Hardcoded Default Band Count

Instead of

```
default=64
```

use

```
default=None
```

Reason

The data should determine the number of useful wavelengths unless the researcher explicitly requests a fixed size.

---

# 4. Save Additional Metadata

Each processed dataset should save

```
selected_wavelengths.npy

band_importance.npy

band_ranking.npy

selection_report.json
```

The report should contain

```
Original bands

Bands after threshold

Final bands

Threshold

Selection method

Importance statistics
```

This makes every experiment reproducible.

---

# Phase 2 — Improve the MedMamba Architecture

Current architecture

```
Input

↓

Encoder

├──Classifier

└──Decoder
```

The reconstruction branch is already producing meaningful spectral metrics (e.g., decreasing spectral reconstruction loss, SAM, and RMSE across epochs), so it should be retained and improved rather than removed. 

---

# Recommended Architecture

```
HSI

↓

Band Selection

↓

Patch Embedding

↓

MedMamba Encoder

├───────────────► Classification Head

│

└───────────────► Spectral Generator

↓

Reconstructed Spectrum

↓

Discriminator
```

---

# Three Learning Objectives

## A

Classification

```
Cross Entropy
```

---

## B

Spectral Reconstruction

Instead of

```
Only MSE
```

use

```
MSE

+

SAM Loss
```

Reason

MSE preserves intensity.

SAM preserves spectral shape.

Medical HSI depends more heavily on preserving spectral signatures than only minimizing amplitude differences.

---

## C

GAN Loss

The discriminator learns

```
Real spectrum

vs

Generated spectrum
```

Generator objective

```
Classification

+

Reconstruction

+

Adversarial
```

---

# Total Loss

```
L

=

CE

+

λ1*MSE

+

λ2*SAM

+

λ3*GAN
```

All λ values should be configurable.

---

# Phase 3 — Improve Reconstruction Metrics

Currently the report logs

```
spectral_recon_loss

SAM

RMSE
```

during validation. 

Expand this to include

```
RMSE

MAE

SAM

SID

Pearson Correlation

Cosine Similarity

PSNR
```

---

# Phase 4 — Fix the Evaluation Pipeline

This is the highest-priority software task because several metrics in the current report are internally inconsistent. For example, precision remains exactly 1.0 while validation accuracy varies dramatically, and balanced accuracy matches accuracy every epoch. 

---

# 1. Validation Mode

Always execute

```
model.eval()

torch.no_grad()
```

before evaluation.

---

# 2. Validation Dataset

Ensure

* no random augmentations
* no MixUp
* no CutMix
* no random masking
* deterministic transforms
* shuffle=False

---

# 3. Metric Calculation

Replace every custom metric implementation with sklearn.

Compute

```
Accuracy

Balanced Accuracy

Precision

Recall

F1

Macro F1

Weighted F1
```

using

```
y_true

y_pred
```

---

# 4. Add Per-Class Metrics

Every validation epoch should generate

```
classification_report.txt
```

containing

```
Precision

Recall

F1

Support

for every class
```

---

# 5. Save Confusion Matrix

Each epoch

```
confusion_matrix_epoch_XX.png
```

Also save

```
confusion_matrix.npy
```

---

# 6. Save Prediction Distribution

Log

```
True labels

Predicted labels

Prediction confidence

Softmax entropy
```

This immediately exposes class-collapse or overconfidence.

---

# 7. Verify Metric Consistency

Implement automatic sanity checks after every validation epoch.

Examples:

* Verify that balanced accuracy is computed independently from overall accuracy.
* Check that precision, recall, and F1 are mathematically consistent.
* Warn if a metric is constant across epochs (e.g., precision always equals 1.0), as this often indicates a bug.
* Flag impossible combinations such as near-perfect precision with very low accuracy.

These checks should generate warnings in the logs rather than silently accepting suspicious values.

---

# Phase 5 — Improve Training Logging

Each epoch should save

```
train_loss

classification_loss

MSE_loss

SAM_loss

GAN_loss

learning_rate

gradient_norm

GPU_memory

epoch_time
```

---

# Phase 6 — Improve Experiment Reports

Replace the current report with a richer structure.

```
experiment/

├──config.json
├──history.csv
├──history.json
├──classification_report_epochXX.txt
├──confusion_matrix_epochXX.png
├──band_selection_report.json
├──selected_wavelengths.npy
├──spectral_metrics.csv
├──best_model.pt
└──plots/
```

---

# Phase 7 — Research Experiments

After the pipeline is stable, evaluate the contribution of each component through controlled ablation studies.

### Band Selection

* All wavelengths
* Threshold only
* Top-N only
* Threshold + Top-N

### Reconstruction

* No reconstruction
* MSE only
* SAM only
* MSE + SAM

### Adversarial Learning

* Without GAN
* With GAN

### Loss Weight Sensitivity

Evaluate different values for

```
λMSE

λSAM

λGAN
```

### Encoder

* Classification only
* Classification + Reconstruction
* Classification + Reconstruction + GAN

These experiments will help quantify how much each design choice contributes to classification performance, spectral fidelity, and overall robustness.
