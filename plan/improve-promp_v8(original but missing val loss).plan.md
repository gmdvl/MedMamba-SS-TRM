# GMedMamba Implementation Plan
## Comprehensive Evaluation, Output Organization, and Reconstruction Pipeline Fixes

---

# Objectives

This update focuses on making the entire training and evaluation pipeline production-quality and thesis-ready.

The implementation should accomplish the following:

- Properly organize all experiment outputs.
- Complete the currently unfinished reconstruction evaluation.
- Complete the currently unfinished latent-space evaluation.
- Improve dataset preprocessing so downstream analyses have all required information.
- Ensure every generated folder contains meaningful outputs.
- Remove placeholder functionality.
- Make every experiment completely reproducible.
- Produce publication-quality evaluation artifacts.

---

# Guiding Principles

## 1. No Empty Directories

Every directory that is created must contain outputs.

If a feature is disabled, the directory should not be created.

Example:

```
BAD

run/
    reconstruction/
    latent_space/

(empty)
```

```
GOOD

run/

or

run/
    reconstruction/
        ...
```

---

## 2. Root Directory Contains Only Final Results

The root of every experiment should only contain the final outputs summarizing the entire run.

Everything generated during epochs belongs inside dedicated folders.

---

## 3. Every Evaluation Module Must Be Independent

Each evaluation module should be implemented as an independent pipeline.

Example

```
ClassificationEvaluator

ReconstructionEvaluator

LatentSpaceEvaluator

CalibrationEvaluator

ConfidenceEvaluator
```

Each module

- collects predictions
- computes metrics
- generates figures
- exports reports

without interfering with training.

---

## 4. All Evaluation Runs After Validation

Evaluation should never occur during training batches.

Pipeline

```
Training Epoch

↓

Validation

↓

Evaluation Modules

↓

Save Outputs
```

This keeps the training loop clean.

---

# Phase 1 — Review Dataset Preparation

Files

```
prepare_histologyhsi_bc_v4.py
```

---

## 1.1 Review Saved Metadata

Every generated sample must preserve metadata required later.

Required fields

```
patient_id

slide_id

patch_id

class

coordinates

original_filename

wavelengths

rgb_band_indices

normalization_parameters
```

---

## 1.2 Preserve Wavelength Information

Currently the dataset appears to lose spectral metadata.

Each generated sample should preserve

```
band centers

band spacing

spectral axis
```

This enables

- spectral plots
- reconstruction plots
- SID
- SAM
- wavelength visualization

---

## 1.3 Preserve RGB Mapping

Every sample should know

```
red band

green band

blue band
```

instead of recomputing later.

This enables

```
Original RGB

↓

Reconstructed RGB

↓

Difference Image
```

---

## 1.4 Preserve Normalization

Store

```
mean

std

min

max

normalization type
```

This allows inverse normalization before visualization.

---

## 1.5 Validate PCA / Band Reduction Metadata

If PCA or wavelength reduction is enabled

store

```
selected wavelengths

explained variance

band mapping

removed bands
```

---

## 1.6 Dataset Validation Report

Generate

```
dataset_report.json
```

including

```
number of samples

bands

patch size

class distribution

patients

slides

normalization

wavelength list
```

---

# Phase 2 — Refactor Experiment Directory Structure

Current implementation places many files directly inside the run directory.

Replace with:

```
experiment/

    run_x/

        experiment_report.json
        validation_report.json
        config.json

        history.csv
        history.json

        spectral_metrics.csv

        plots/

        confusion_matrices/

            standard/

            normalized/

        classification_reports/

        reconstruction/

        latent_space/

        confidence_analysis/

        calibration/

        logs/

        checkpoints/

        predictions/
```

---

## Root Directory Rules

Allowed

```
experiment_report.json

validation_report.json

config.json

history.csv

history.json

spectral_metrics.csv
```

Everything else belongs in folders.

---

# Phase 3 — Classification Evaluation Pipeline

Move all classification outputs into

```
classification_reports/

confusion_matrices/
```

---

## Classification Reports

Each epoch

```
epoch_001.txt

epoch_002.txt

...

epoch_100.txt
```

---

## Confusion Matrices

Standard

```
standard/

epoch_001.png
epoch_002.png
```

Normalized

```
normalized/

epoch_001.png
epoch_002.png
```

Also save

```
.npy

.csv
```

versions.

---

## Final Confusion Matrix

Save

```
best_model.png

best_model_normalized.png
```

---

# Phase 4 — Reconstruction Evaluation Pipeline

This is currently incomplete.

It must become a full evaluation module.

---

## Reconstruction Workflow

After validation

```
Validation Dataset

↓

Forward Pass

↓

Reconstruction

↓

Metrics

↓

Visualizations

↓

Save
```

---

## Sample Selection

Configurable

Example

```
10

20

50

100 samples
```

Random or stratified.

---

## Metrics

Compute

```
MSE

RMSE

MAE

PSNR

SSIM

SAM

SID

Pearson Correlation

Cosine Similarity

Peak Position Error
```

Per sample

Per class

Overall

---

## Visualizations

Every sample should generate

```
Original RGB

↓

Reconstructed RGB

↓

Difference RGB

↓

Spectral Curves

↓

Error Heatmap
```

---

## Folder Layout

```
reconstruction/

    epoch_010/

        sample_001/

            original_rgb.png

            reconstructed_rgb.png

            difference.png

            spectrum.png

            metrics.json

        summary.csv

        reconstruction_metrics.json
```

---

## Final Reconstruction Report

```
reconstruction_report.json
```

Include

```
mean metrics

std

best sample

worst sample

per-class averages
```

---

# Phase 5 — Latent Space Evaluation Pipeline

Currently empty.

Implement complete latent analysis.

---

## Feature Extraction

Extract encoder outputs.

Support

```
CLS token

mean pooled features

last hidden state

configurable
```

---

## Save Embeddings

```
embeddings.npy

labels.npy

metadata.csv
```

---

## PCA

Generate

```
pca.png

pca_3d.png
```

---

## t-SNE

Generate

```
tsne.png
```

---

## UMAP

Generate

```
umap.png
```

---

## Interactive Plots

Optional

```
plotly

html
```

---

## Cluster Metrics

Compute

```
Silhouette Score

Davies-Bouldin Index

Calinski-Harabasz Index
```

---

## Folder Structure

```
latent_space/

    epoch_020/

        embeddings.npy

        labels.npy

        metadata.csv

        pca.png

        tsne.png

        umap.png

        cluster_metrics.json
```

---

# Phase 6 — Confidence Evaluation

Move everything into

```
confidence_analysis/
```

Generate

```
confidence_histogram.png

entropy_distribution.png

confidence_vs_accuracy.png

incorrect_predictions.csv
```

Metrics

```
ECE

MCE

Brier Score

Average Confidence
```

---

# Phase 7 — Calibration Module

Create

```
calibration/
```

Generate

```
Reliability Diagram

Temperature Scaling

ECE

MCE

Brier Score

Calibration Curve
```

---

# Phase 8 — Evaluation Scheduling

Allow configurable evaluation frequency.

Examples

```
Every Epoch

Every 5 Epochs

Every 10 Epochs

Best Epoch Only

Final Epoch Only
```

CLI

```
--evaluation_interval

--save_reconstruction_interval

--latent_interval
```

---

# Phase 9 — Configuration Improvements

Expose all evaluation options.

```
--save_reconstruction

--save_latent_space

--save_confusion_matrix

--save_reports

--save_rgb

--save_spectra

--num_visualization_samples

--latent_method

--reconstruction_metrics

--evaluation_interval
```

---

# Phase 10 — Logging Improvements

Every module should log

```
Start

Progress

Elapsed Time

Completion

Errors
```

Example

```
Evaluating Reconstruction...

Saving Sample 3/50...

Computing SSIM...

Saving Metrics...

Completed.
```

---

# Phase 11 — Final Experiment Report

Expand

```
experiment_report.json
```

Include

## Dataset

```
samples

bands

classes

patch size
```

---

## Training

```
loss

accuracy

learning rate

epoch times
```

---

## Classification

```
Accuracy

Balanced Accuracy

Precision

Recall

Specificity

Sensitivity

F1

MCC

Cohen's Kappa

OA
```

---

## Reconstruction

```
MSE

RMSE

MAE

PSNR

SSIM

SAM

SID

Pearson

Cosine Similarity

Peak Position Error
```

---

## Latent Space

```
Silhouette

Davies-Bouldin

Calinski-Harabasz
```

---

## Calibration

```
ECE

MCE

Brier Score
```

---

## Confidence

```
Average Confidence

Entropy

Incorrect Predictions
```

---

# Phase 12 — Code Refactoring

Separate the evaluation code into reusable modules.

Recommended structure

```
evaluation/

    classification.py

    reconstruction.py

    latent_space.py

    calibration.py

    confidence.py

    metrics.py

    visualization.py

    reports.py

    utils.py
```

The training script should only orchestrate these modules, not implement their logic directly.

---

# Phase 13 — Validation & Regression Testing

Before considering the implementation complete, verify the following:

### Dataset
- Every sample includes metadata (patient, slide, patch, wavelengths, RGB indices, normalization parameters).
- Dataset report accurately reflects class distributions and spectral information.

### Experiment Structure
- Run root contains only final summary artifacts.
- All epoch-specific outputs are stored in their designated folders.
- No empty directories are created.

### Classification
- Confusion matrices (standard and normalized) are generated for every configured evaluation epoch.
- Classification reports are saved correctly and match reported metrics.

### Reconstruction
- Reconstruction metrics are computed for every selected sample.
- RGB comparisons, spectral plots, and error heatmaps are generated.
- Reconstruction reports contain per-sample, per-class, and aggregate metrics.

### Latent Space
- Embeddings are saved.
- PCA, t-SNE, and UMAP visualizations are generated.
- Cluster quality metrics are computed and reported.

### Confidence & Calibration
- Reliability diagrams, calibration curves, and confidence analyses are generated.
- ECE, MCE, and Brier Score are included in the final report.

### Reporting
- `experiment_report.json` contains all classification, reconstruction, latent-space, confidence, and calibration metrics.
- All paths referenced in reports exist.
- No placeholder values or missing sections remain.

---

# Expected Final Outcome

The completed implementation should transform the training pipeline into a fully integrated, thesis-quality evaluation framework where:

- Every generated directory contains meaningful outputs.
- No placeholder or partially implemented features remain.
- Reconstruction and latent-space analyses are fully operational and quantitatively evaluated.
- Dataset preprocessing preserves all information required for downstream visualization and analysis.
- Experiment outputs are consistently organized, reproducible, and suitable for publication.
- The training script serves as a lightweight orchestrator, with all evaluation logic encapsulated in dedicated, reusable modules.