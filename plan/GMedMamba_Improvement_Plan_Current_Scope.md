# GMedMamba Improvement Plan — Current Implementation Scope

## 1. Scope

This plan defines the current implementation and experimental roadmap for improving GMedMamba.

### Implement now

1. Fix all data/preprocessing problems.
2. Fix classification and evaluation problems.
3. Fix training instability.
4. Replace ReLU with LeakyReLU.
5. Redesign the spectral/spatial split.
6. Use all channels for spectral-spatial processing.
7. Reduce model parameters.
8. Reduce computational cost.
9. Improve HSI generalization.
10. Improve multiclass classification.
11. Correct the reconstruction pathway.
12. Improve validation/checkpoint selection.
13. Add comprehensive diagnostics.
14. Run controlled ablations.
15. Run fair MedMamba vs. GMedMamba experiments.

### Deferred

**Frozen MedMamba → trainable GMedMamba weight/feature transfer**

This is explicitly reserved for a later experimental phase and must not be included in the current implementation.

---

# Phase 0 — Freeze the Current Baseline

Before changing anything, preserve the current implementation as:

`GMedMamba-Baseline`

Do not modify this copy.

Record:

- architecture;
- parameter count;
- trainable parameter count;
- FLOPs/MACs;
- model size;
- GPU memory;
- training time;
- inference time;
- throughput;
- training loss;
- validation loss;
- training accuracy;
- validation accuracy;
- OA;
- balanced accuracy;
- macro precision;
- macro recall;
- macro F1;
- weighted F1;
- MCC;
- Cohen's kappa;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix.

The manuscript reports approximately 27.42 M parameters for GMedMamba-Tiny versus 13.53 M for MedMamba-Tiny, so parameter reduction must be measured against this baseline.

---

# Phase 1 — Fix the Data Pipeline First

This is the highest-priority stage.

The model should not be optimized against a broken or unreliable dataset split.

The manuscript identifies incomplete validation-class coverage, especially in Histology, as a major issue.

## 1.1 Build a deterministic preprocessing pipeline

Use:

```text
Raw dataset
    ↓
Dataset inspection
    ↓
Data validation
    ↓
Patient/image grouping
    ↓
Train/validation/test split
    ↓
Train-only augmentation/balancing
    ↓
Patch extraction
    ↓
Normalization
    ↓
Final dataset
    ↓
Training
```

The split must occur before any operation that can create correlated samples.

---

# Phase 2 — Fix Patient-Level/Data Leakage Issues

For Histology, every sample must retain:

```text
patient_id
image_id
patch_id
class_id
```

The split should be based on:

```text
patient_id
```

not:

```text
patch_id
```

Desired structure:

```text
Patient A ──→ TRAIN
Patient B ──→ TRAIN
Patient C ──→ VALIDATION
Patient D ──→ TEST
```

Avoid:

```text
Patient A patch 1 → TRAIN
Patient A patch 2 → VALIDATION
Patient A patch 3 → TEST
```

unless a patch-level evaluation is explicitly intended.

---

# Phase 3 — Enforce Class Coverage

The preprocessing script must fail before training if a required class is missing.

For every split, verify that every expected class is represented.

Example:

```text
TRAIN
  class 0 > 0
  class 1 > 0
  class 2 > 0

VALIDATION
  class 0 > 0
  class 1 > 0
  class 2 > 0

TEST
  class 0 > 0
  class 1 > 0
  class 2 > 0
```

For PAD, verify all six expected classes.

If this is impossible because the source dataset genuinely lacks enough independent patients, the preprocessing system must explicitly report the limitation instead of silently producing an incomplete validation set.

---

# Phase 4 — Generate a Preprocessing QC Report

Every preprocessing run should create:

```text
preprocessing_report.json
preprocessing_report.txt
```

The report should contain:

```text
dataset
number_of_patients
number_of_images
number_of_samples
number_of_patches

train_count
validation_count
test_count

class_distribution
patient_distribution
class_distribution_by_split

image_dimensions
spectral_band_count
wavelength_range

nan_count
inf_count
invalid_pixel_count

constant_bands
low_variance_bands
missing_bands
duplicate_bands

normalization_method
augmentation_method
random_seed
```

Every preprocessing transformation must be logged.

---

# Phase 5 — HSI Spectral Quality Control

Inspect every HSI dataset for:

- NaN;
- Inf;
- missing bands;
- duplicate bands;
- constant bands;
- near-zero variance bands;
- corrupted spectra;
- abnormal intensity ranges;
- wavelength ordering;
- wavelength gaps;
- inconsistent dimensions.

Do not silently discard problematic data.

Every modification must be logged.

Example:

```text
Removed bands:
    band 12 → zero variance
    band 27 → >99% invalid
    band 54 → corrupted metadata
```

---

# Phase 6 — Preserve Physical Wavelength Information

The wavelength-aware representation should remain.

When physical wavelength metadata is available:

```text
physical wavelength
        ↓
wavelength encoding
        ↓
spectral embedding
        ↓
GMedMamba
```

Do not reduce HSI to arbitrary feature channels and then claim that those channels represent physical spectral bands.

---

# Phase 7 — Improve Normalization

Compare:

### Method A
Per-band normalization.

### Method B
Per-image normalization.

### Method C
Dataset-level normalization.

### Method D
Robust percentile normalization.

Make normalization configurable:

```text
--normalization {none,band,image,dataset,robust}
```

Normalization statistics used by validation/test must be derived only from training data.

---

# Phase 8 — Train-Only Augmentation

After splitting:

```text
TRAIN
 ↓
augmentation
```

while:

```text
VALIDATION → no augmentation
TEST       → no augmentation
```

No validation/test sample should be altered by training augmentation.

---

# Phase 9 — Add HSI-Specific Augmentation

Test:

### Spectral noise

Small perturbations to spectra.

### Spectral scaling

Small multiplicative changes.

### Spectral offset

Small additive changes.

### Band dropout

Randomly mask a small number of spectral bands.

### Spectral masking

Mask contiguous wavelength regions.

### Spatial transformations

- horizontal flip;
- vertical flip;
- rotations;
- spatial crop;
- limited scaling.

Augmentations must preserve the underlying pathology/class semantics.

---

# Phase 10 — Fix Classification Before Architectural Optimization

The current PAD results show severe class collapse:

- Accuracy: 36.73%;
- Balanced Accuracy: 21.01%;
- Macro F1: 18.37%;
- three of six classes have zero recall.

This must become a primary development target.

---

# Phase 11 — Analyze Class Imbalance

Before training, calculate:

```text
class_count
class_percentage
```

for:

```text
train
validation
test
```

Additionally calculate:

```text
patient_count_per_class
image_count_per_class
patch_count_per_class
```

Thousands of patches from one patient should not automatically be treated as thousands of independent observations.

---

# Phase 12 — Test Classification Strategies

Run controlled experiments with:

### A. Standard Cross Entropy

Baseline.

### B. Weighted Cross Entropy

Use training-set class frequencies.

### C. Focal Loss

For difficult/underrepresented classes.

### D. Balanced sampling

Use a weighted training sampler.

### E. Moderate oversampling

Only on the training set.

Do not combine all strategies immediately. Identify which intervention actually solves class collapse.

---

# Phase 13 — Change Model Selection Metric

Do not select the best checkpoint using accuracy alone.

Primary:

```text
macro-F1
```

Secondary:

```text
balanced accuracy
MCC
weighted F1
OA
```

For imbalanced datasets, a model that obtains high OA by predicting majority classes should not automatically be selected as the best model.

---

# Phase 14 — Fix the Macro-F1 Calculation

Implement macro-F1 correctly.

For each class:

\[
F1_k = \frac{2P_kR_k}{P_k+R_k}
\]

Then:

\[
F1_{macro} = \frac{1}{K}\sum_{k=1}^{K}F1_k.
\]

Do not calculate macro-F1 as:

\[
\frac{2P_{macro}R_{macro}}{P_{macro}+R_{macro}}.
\]

These are not generally equivalent.

---

# Phase 15 — Add Class-Collapse Monitoring

Every epoch should record:

```text
true class distribution
predicted class distribution
per-class recall
per-class precision
per-class F1
```

Automatically flag:

```text
class recall == 0
```

for consecutive epochs.

This makes class collapse visible during training rather than only after final evaluation.

---

# Phase 16 — Fix Gradient/Training Instability

The manuscript identifies a serious problem involving non-finite gradients and skipped optimizer steps.

Implement explicit counters:

```text
total_batches
valid_updates
skipped_updates
nonfinite_loss
nonfinite_gradients
```

Calculate:

```text
skip_ratio
```

per epoch.

---

# Phase 17 — Invalid Epoch Protection

An epoch with:

```text
valid_updates == 0
```

must be marked:

```text
INVALID
```

It must not:

- become the best epoch;
- update the best checkpoint;
- be used as a legitimate validation result;
- update early stopping as if it were normal training;
- produce misleading zero metrics.

---

# Phase 18 — Add Gradient Clipping

Introduce:

```text
--max_grad_norm
```

Test reasonable values rather than assuming one is optimal.

The purpose is to prevent gradient explosion, not to hide an underlying architectural instability.

---

# Phase 19 — Replace ReLU with LeakyReLU

Implement the requested activation change.

Change:

```text
ReLU
```

to:

```text
LeakyReLU
```

with configurable:

```text
negative_slope
```

Start with:

```text
0.01
```

---

# Phase 20 — Do Not Replace SiLU Blindly

The existing SSM path uses SiLU.

Therefore, the first experiment should be:

```text
Conv/spectral branch → LeakyReLU
SSM branch           → SiLU
```

rather than replacing every activation.

Then optionally test:

```text
LeakyReLU everywhere
```

as a separate ablation.

---

# Phase 21 — LeakyReLU Ablation

Run:

| Model | Activation |
|---|---|
| Baseline | ReLU |
| LReLU-01 | LeakyReLU 0.01 |
| LReLU-05 | LeakyReLU 0.05 |
| LReLU-10 | LeakyReLU 0.10 |

Measure:

- OA;
- balanced accuracy;
- macro-F1;
- minority recall;
- validation loss;
- gradient stability;
- generalization gap.

---

# Phase 22 — Major Architecture Change: Remove the Hard Channel Split

This should be the primary architectural experiment.

Do not preserve a design in which channels are permanently divided between spectral and spatial processing.

The new design should not be:

```text
C spectral channels
+
1 spatial channel
```

That would still create an artificial imbalance.

Instead:

```text
C-channel representation
       │
       ├──────────────┐
       │              │
       ▼              ▼
 Spectral            Spatial
 operator            operator
       │              │
       └──────┬───────┘
              ▼
           Fusion
```

Both branches receive the full representation.

---

# Phase 23 — Full-Channel Spectral-Spatial Block

Preferred design:

```text
Input X [B,H,W,C]
        │
        ▼
Shared projection
        │
        ▼
Full feature X
   ┌────┴────┐
   │         │
   ▼         ▼
Spectral   Spatial
branch     branch
   │         │
   └────┬────┘
        ▼
Cross-branch interaction
        ▼
Lightweight fusion
        ▼
Residual
```

Key principle:

> Spectral and spatial processing are separated by operation, not by permanently separating channels.

---

# Phase 24 — Recommended Spectral Branch

Avoid expensive dense 3-D convolution where possible.

Candidate:

```text
Full X
 ↓
spectral depthwise/factorized operation
 ↓
normalization
 ↓
LeakyReLU
 ↓
pointwise projection
```

Potential implementations:

### Option A

Depthwise Conv3D.

### Option B

Factorized Conv3D:

```text
3×1×1
+
1×3×3
```

### Option C — preferred starting point

```text
spectral depthwise
+
spatial depthwise
+
1×1 pointwise
```

The objective is to preserve spectral/spatial locality without paying for dense 3-D convolution everywhere.

---

# Phase 25 — Recommended Spatial Branch

Preserve the effective MedMamba-style spatial processing where it is useful:

```text
Full X
 ↓
SS2D / spatial SSM
 ↓
normalization
 ↓
activation
```

Do not replace the entire spatial pathway without evidence that it is necessary.

---

# Phase 26 — Full-Channel Fusion

Do not simply concatenate:

```text
[spectral | spatial]
```

and create a very large projection.

Instead use lightweight interaction:

```text
Fspectral
Fspatial
   ↓
lightweight interaction
   ↓
gating
   ↓
weighted sum
```

For example:

\[
F = g_sF_s + g_pF_p
\]

with normalized or otherwise constrained gates where appropriate.

---

# Phase 27 — Cross-Branch Interaction

Allow spectral and spatial representations to exchange information.

Conceptually:

```text
Fs = Spectral(X)
Fp = Spatial(X)

Fs' = Fs + A(Fp)
Fp' = Fp + B(Fs)

Fout = Fuse(Fs', Fp')
```

Keep `A` and `B` lightweight.

This is preferable to completely independent branches.

---

# Phase 28 — Reduce Conv3D Parameters

The current spectral branch contains multiple Conv3D operations.

Test:

### Option A
Depthwise Conv3D.

### Option B
Factorized Conv3D.

### Option C
Depthwise spectral + depthwise spatial + pointwise projection.

Measure:

```text
parameters
FLOPs
GPU memory
training time
macro-F1
balanced accuracy
```

---

# Phase 29 — Reduce Spectral Branch Width

Test:

```text
100%
75%
50%
25%
```

of the current spectral hidden width.

Measure:

```text
parameters
FLOPs
validation macro-F1
balanced accuracy
HSI generalization
```

---

# Phase 30 — Reduce Number of Expensive Spectral Blocks

The model does not necessarily need equally expensive spectral processing in every stage.

Candidate:

```text
Stage 1 → strong spectral-spatial
Stage 2 → strong spectral-spatial
Stage 3 → lightweight spectral interaction
Stage 4 → primarily spatial
```

The exact allocation should be determined experimentally.

---

# Phase 31 — Lightweight Spectral Attention

If the current model uses spectral inter-channel attention, compare:

```text
Full attention
vs
Lightweight attention
vs
No attention
```

Potential lightweight alternatives:

```text
SE-style gating
ECA-style channel weighting
simple learned channel gate
```

Keep attention only if it provides measurable benefit relative to its parameter/computational cost.

---

# Phase 32 — Reduce Classifier Parameters

Use a compact classification head:

```text
Final feature
 ↓
Global average pooling
 ↓
LayerNorm
 ↓
Dropout
 ↓
Linear
 ↓
Classes
```

Avoid unnecessarily large fully connected layers.

---

# Phase 33 — Multi-Level Feature Aggregation Without Parameter Explosion

If multi-level aggregation is retained:

```text
Stage1 → small projection ┐
Stage2 → small projection ├→ weighted fusion
Stage3 → small projection ┤
Stage4 → small projection ┘
```

Avoid a very large raw concatenation followed by a huge projection.

---

# Phase 34 — Correct the Reconstruction Pathway

The intended architecture should be:

```text
Input
 ↓
GMedMamba
 ↓
latent representation
 ↓
reconstruction decoder
 ↓
reconstructed HSI
```

The decoder must reconstruct from the learned backbone representation rather than directly from the original input.

---

# Phase 35 — Reconstruction Should Be Auxiliary

Use:

\[
L = L_{classification} + \lambda_{recon}L_{recon}
\]

with configurable:

```text
lambda_recon
```

Test:

```text
λ = 0
λ = small
λ = medium
```

If reconstruction does not improve classification/generalization, it should not automatically remain in the final architecture.

---

# Phase 36 — Directly Address HSI Overfitting

The current HSI results show a large training/validation gap.

Therefore the new architecture should explicitly optimize for generalization rather than training accuracy.

---

# Phase 37 — Regularization Experiments

Test individually:

```text
weight decay
dropout
drop path / stochastic depth
spectral dropout
feature dropout
```

Do not add everything simultaneously.

---

# Phase 38 — Early Stopping

Use:

```text
validation macro-F1
```

or:

```text
balanced accuracy
```

rather than training loss alone.

Example:

```text
best checkpoint =
highest validation macro-F1
```

with patience.

---

# Phase 39 — Spectral Dropout

During training:

```text
HSI
 ↓
randomly mask a small number of spectral bands
 ↓
model
```

This should discourage memorization of exact spectral signatures.

Evaluate carefully because excessive masking can remove diagnostically useful spectral information.

---

# Phase 40 — Capacity-Controlled Architecture Search

Establish explicit parameter targets.

### Current baseline

```text
~27.42 M
```

### Target 1

```text
~20 M
```

### Target 2

```text
~17 M
```

### Target 3

```text
~15 M
```

The ideal outcome is a model around 15–17 M parameters that outperforms the current 27.42 M version.

---

# Phase 41 — Parameter Reduction Priority

Reduce parameters in this order:

1. Dense Conv3D.
2. Spectral branch width.
3. Repeated spectral blocks.
4. Expensive attention.
5. Fusion projections.
6. Classifier.

---

# Phase 42 — Training Configuration Cleanup

Create a single configuration system for:

```text
activation
negative_slope
spectral_operator
spectral_width
spatial_operator
fusion_type
attention_type
dropout
drop_path
weight_decay
classification_loss
class_balancing
spectral_augmentation
reconstruction
reconstruction_weight
normalization
seed
```

This makes experiments reproducible.

---

# Phase 43 — Experiment Naming

Use explicit names:

```text
gmedmamba_baseline

gmedmamba_lrelu

gmedmamba_fullchannel

gmedmamba_fullchannel_lite

gmedmamba_fullchannel_lite_lrelu

gmedmamba_fullchannel_lite_cls

gmedmamba_fullchannel_lite_recon

gmedmamba_final
```

Avoid ambiguous names such as:

```text
v6_final_final2
```

---

# Phase 44 — Required Ablation Study

The final ablation should isolate:

| Component | Test |
|---|---|
| LeakyReLU | On/off |
| Full-channel processing | On/off |
| Lightweight spectral operator | On/off |
| Spectral attention | On/off |
| Spectral gating | On/off |
| Cross-branch fusion | On/off |
| Multi-level aggregation | On/off |
| Reconstruction | On/off |
| Spectral augmentation | On/off |
| Class balancing | On/off |
| Regularization | On/off |

The goal is to establish which components actually contribute to improvement.

---

# Phase 45 — Controlled MedMamba Comparison

Compare:

```text
Original MedMamba
vs
Current GMedMamba
vs
Improved GMedMamba
```

under exactly the same:

- data;
- split;
- preprocessing;
- image resolution;
- augmentation;
- optimizer;
- learning rate;
- batch size;
- epochs;
- random seeds;
- checkpoint strategy;
- evaluation metrics.

The original MedMamba should be referenced as the actual published/original implementation rather than treating a local replication as the source of the architecture.

---

# Phase 46 — Parameter-Matched Comparison

Compare approximately:

```text
MedMamba       ~13.53 M
GMedMamba      ~27.42 M
GMedMamba-lite ~15–20 M
```

Then answer:

> Does the spectral-spatial extension provide useful information, or is the apparent improvement simply due to having substantially more parameters?

This is a critical thesis experiment.

---

# Phase 47 — Multi-Seed Experiments

For final experiments:

```text
minimum: 5 seeds
preferred: 5–10 seeds
```

Report:

\[
mean \pm std
\]

for:

- OA;
- balanced accuracy;
- macro-F1;
- weighted F1;
- MCC;
- Cohen's kappa;
- per-class recall.

---

# Phase 48 — Final Evaluation Metrics

## Classification

Report:

```text
OA
Balanced Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted F1
MCC
Cohen's Kappa
```

## Per class

Report:

```text
Precision
Recall
F1
Support
Sensitivity
Specificity
```

## Probability quality

Where appropriate:

```text
ROC-AUC
PR-AUC
ECE
```

---

# Phase 49 — Required Visualizations

For every final experiment generate:

## Confusion matrix

- raw;
- normalized.

## Training curves

- training loss;
- validation loss;
- training accuracy;
- validation accuracy;
- training macro-F1;
- validation macro-F1.

## Generalization gap

Plot training and validation accuracy together.

## Class recall

Plot recall for every class.

## Parameter/performance plot

Plot:

```text
parameters vs macro-F1
parameters vs balanced accuracy
```

This can become an important thesis figure.

---

# Phase 50 — Final Proposed Architecture

Target architecture:

```text
                   RGB / HSI
                       │
                       ▼
              Patch Embedding
                       │
                       ▼
             Full Feature Tensor
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Spectral Branch     Spatial Branch
              │                 │
       Lightweight          SS2D /
       spectral op          spatial SSM
              │                 │
              └────────┬────────┘
                       │
               Cross Interaction
                       │
                Learned Gating
                       │
              Lightweight Fusion
                       │
                  Residual
                       │
                Next Stage
                       │
                     ...
                       │
                Global Pool
                       │
                 LayerNorm
                       │
                  Dropout
                       │
                 Classifier
```

For HSI reconstruction:

```text
Backbone latent representation
             │
             ▼
      Lightweight decoder
             │
             ▼
       Reconstructed HSI
```

No frozen MedMamba pathway is included.

---

# Phase 51 — Revised Development Stages

## Stage A — Correctness

```text
1. Fix preprocessing
2. Fix patient-level splits
3. Fix class coverage
4. Fix macro-F1
5. Fix invalid epochs
6. Fix gradient handling
7. Fix reconstruction connection
```

Do not move forward until this is stable.

## Stage B — Classification

```text
8. Analyze imbalance
9. Weighted loss
10. Balanced sampler
11. Focal-loss experiment
12. Macro-F1 checkpointing
13. Per-class monitoring
14. Class-collapse detection
```

## Stage C — Architecture

```text
15. ReLU → LeakyReLU
16. Remove hard channel split
17. Full-channel spectral processing
18. Full-channel spatial processing
19. Cross-branch interaction
20. Lightweight fusion
```

## Stage D — Efficiency

```text
21. Replace dense Conv3D
22. Reduce spectral width
23. Reduce spectral blocks
24. Lightweight attention
25. Lightweight classifier
26. Parameter targets: 20M → 17M → 15M
```

## Stage E — Generalization

```text
27. Spectral augmentation
28. Spectral dropout
29. Spatial augmentation
30. Weight decay
31. Dropout
32. Drop path
33. Early stopping
```

## Stage F — Scientific Validation

```text
34. Controlled MedMamba
35. Current GMedMamba
36. Improved GMedMamba
37. Parameter-matched comparison
38. 5+ seeds
39. Full ablations
40. Statistical analysis
41. Efficiency benchmarks
42. Final thesis tables/figures
```

---

# Deferred Stage G — Frozen MedMamba Transfer

This is explicitly outside the current implementation.

Do not modify the current architecture to accommodate it.

Record it as:

> **Future Work: Frozen MedMamba Feature/Weight Transfer**

Potential future design:

```text
Frozen MedMamba
       │
       ▼
Intermediate features
       │
       ▼
Trainable adapter
       │
       ▼
GMedMamba residual
```

Potential future experiments:

```text
frozen backbone
+
feature adapters
+
residual feature injection
+
knowledge distillation
+
partial unfreezing
```

None of these should be included in the current GMedMamba improvement experiment.

This keeps the current study scientifically interpretable because multiple architectural changes are not being confounded by frozen-backbone transfer.

---

# Final Priority List

| Priority | Task | Importance |
|---:|---|---|
| 1 | Fix patient-level splitting | Critical |
| 2 | Enforce class coverage | Critical |
| 3 | Fix invalid epoch/gradient handling | Critical |
| 4 | Fix macro-F1 calculation | Critical |
| 5 | Fix reconstruction connection | Critical |
| 6 | Improve preprocessing/QC | Critical |
| 7 | Diagnose class imbalance | Critical |
| 8 | Fix class collapse | Critical |
| 9 | LeakyReLU | High |
| 10 | Remove hard channel split | Very High |
| 11 | Full-channel spectral-spatial processing | Very High |
| 12 | Lightweight spectral operator | Very High |
| 13 | Reduce spectral width | High |
| 14 | Reduce expensive spectral blocks | High |
| 15 | Lightweight fusion/attention | High |
| 16 | HSI spectral augmentation | High |
| 17 | Regularization | High |
| 18 | Parameter-controlled experiments | High |
| 19 | Controlled MedMamba comparison | Critical for thesis |
| 20 | Multi-seed validation | Critical for thesis |
| 21 | Full ablation study | Critical for thesis |
| Future | Frozen MedMamba transfer | Deferred |

---

# Core Research Hypothesis

The current architectural hypothesis is:

\[
\boxed{
\text{Full-channel spectral-spatial interaction}
+
\text{lightweight operators}
+
\text{LeakyReLU}
}
\]

rather than:

\[
\text{hard channel split}
+
\text{larger spectral branch}.
\]

The engineering/generalization hypothesis is:

\[
\boxed{
\text{better data preparation}
+
\text{correct class handling}
+
\text{stable training}
}
\]

must be established before architectural performance is judged.

The frozen MedMamba transfer mechanism remains completely isolated as future work so that it cannot confound the current GMedMamba improvement study.
