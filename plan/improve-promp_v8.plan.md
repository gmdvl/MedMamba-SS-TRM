# Phase 0 — Validation Loss Investigation & Stabilization (Highest Priority)

## Objective

This is the highest-priority phase of the project.

No additional feature development (reconstruction evaluation, latent-space visualization, expanded metrics, or reporting) should be implemented until the validation loss behavior has been fully investigated, understood, and stabilized.

The goal is to identify the root cause of the validation loss behavior rather than masking it through hyperparameter tuning or early stopping.

---

## Success Criteria

This phase is considered complete only when:

- The validation loss follows a stable and explainable trend.
- Training and validation pipelines are verified to be identical except for data augmentation.
- Every loss component has been independently validated.
- Dataset integrity has been verified.
- Training dynamics have been analyzed.
- The root cause(s) have been identified and corrected.
- The fixes have been validated through controlled ablation experiments.

Only after these criteria are satisfied should implementation continue with the remaining phases.

---

# Phase 0.1 — Complete Dataset Pipeline Audit

**Primary file**

```
prepare_histologyhsi_bc_v4.py
```

Perform a complete audit of the preprocessing pipeline.

Verify:

- identical preprocessing for training and validation
- identical normalization
- identical spectral scaling
- identical wavelength ordering
- identical interpolation
- identical padding
- identical patch extraction
- identical label encoding
- identical PCA/band reduction (when enabled)

Check for:

- patient leakage
- slide leakage
- patch overlap
- duplicate patches
- mirrored duplicates
- rotated duplicates
- corrupted samples
- NaN values
- Inf values
- zero-valued spectra

Generate

```
dataset_diagnostics.json
```

containing:

- dataset statistics
- normalization statistics
- spectral statistics
- class distribution
- patient distribution
- slide distribution
- duplicate detection
- corruption report

---

# Phase 0.2 — Loss Function Audit

Review every component contributing to the total loss.

Examples:

```
Total Loss =
    Classification Loss
  + Reconstruction Loss
  + Contrastive Loss
  + Regularization Loss
```

Log every component independently during both training and validation.

Generate:

```
loss_components.csv
loss_components.json
```

for every epoch.

---

# Phase 0.3 — Training / Validation Consistency Audit

Verify that validation correctly performs:

- model.eval()
- torch.no_grad()
- identical preprocessing
- identical normalization
- identical label mapping
- identical spectral ordering
- identical loss computation

Confirm that validation does **not** perform:

- data augmentation
- Dropout
- BatchNorm updates
- gradient computation
- optimizer updates

---

# Phase 0.4 — Training Dynamics Diagnostics

Log every epoch:

- training loss
- validation loss
- classification loss
- reconstruction loss
- regularization loss
- learning rate
- gradient norm
- parameter norm
- update norm

Generate plots for:

- Training Loss
- Validation Loss
- Classification Loss
- Reconstruction Loss
- Learning Rate
- Gradient Norm
- Generalization Gap

---

# Phase 0.5 — Per-Class Validation Analysis

Generate per-class:

- Loss
- Accuracy
- Precision
- Recall
- Specificity
- Sensitivity
- F1 Score
- MCC
- Cohen's Kappa

Determine whether a single class is responsible for the validation loss behavior.

---

# Phase 0.6 — Controlled Ablation Experiments

Run controlled experiments changing only one variable at a time.

Examples:

- Reconstruction head enabled vs disabled
- Attention module variants
- Class balancing enabled vs disabled
- PCA enabled vs disabled
- Different spectral band counts
- Different normalization strategies
- Different augmentation policies
- Label smoothing enabled vs disabled

Each experiment should modify only one variable to isolate the source of the validation issue.

---

# Phase 0.7 — Root Cause Analysis Report

Produce a dedicated report summarizing:

- Root cause(s) identified
- Evidence supporting each conclusion
- Corrective actions implemented
- Quantitative impact of each fix
- Remaining limitations
- Recommended default configuration

Generate:

```
validation_loss_analysis.md
validation_loss_report.json
```

These reports become part of every experiment and serve as the foundation for the remaining implementation phases.

---

## Mandatory Exit Criteria

Phase 0 is complete only when all of the following are satisfied:

- Stable validation loss behavior has been achieved.
- Training and validation curves exhibit expected convergence.
- No evidence of data leakage exists.
- No preprocessing inconsistencies remain.
- All loss components behave as expected.
- Controlled ablation experiments confirm the identified root cause(s).
- The corrected configuration is adopted as the new project baseline.

Only then should the project proceed to Phase 1 (Dataset Preparation Improvements), followed by the remaining evaluation, reconstruction, latent-space, reporting, and visualization phases.