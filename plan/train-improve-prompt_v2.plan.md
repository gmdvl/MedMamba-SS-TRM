# Phase 1 — Add Spectral Band (Wavelength) Selection

## Goal

Reduce the spectral dimension before patches are written to disk so the generated `.npy` datasets contain only informative wavelengths.

This should happen during preprocessing, not during training.

Advantages:

* smaller datasets
* lower RAM usage
* faster loading
* faster training
* less overfitting
* no changes required to GMedMamba

---

## 1.1 Add new CLI options

Add arguments such as

```
--band_selection
```

choices

```
none
variance
correlation
mutual_information
manual
topk
```

Default

```
none
```

Additional options

```
--num_bands 64
```

```
--band_file selected_bands.npy
```

```
--band_selection_sample_fraction 0.10
```

to avoid scanning the entire dataset.

---

## 1.2 Create a Band Selection module

Create a separate section

```
Band Selection Utilities
```

Functions

```
compute_band_statistics()

select_bands_variance()

select_bands_mutual_information()

select_bands_correlation()

save_selected_bands()

load_selected_bands()
```

Keeping this isolated prevents touching the patch extraction code.

---

## 1.3 Two-pass preprocessing

Current pipeline

```
Open cube

↓

Extract patches

↓

Save shards
```

New pipeline

```
Pass 1

Open subset of cubes

↓

Collect spectral statistics

↓

Determine best wavelengths

↓

Save selected_band_indices.npy

----------------------------

Pass 2

Open cube

↓

Keep only selected bands

↓

Extract patches

↓

Save shards
```

Only one preprocessing run is required from the user's perspective.

---

## 1.4 Band selection algorithms

Implement progressively.

### A. Manual

```
selected_indices = np.load(...)
```

Useful for reproducing papers.

---

### B. Variance

For each wavelength

```
Variance across all sampled pixels
```

Remove bands with almost no variation.

Simple

Fast

Very common.

---

### C. Correlation pruning

Many neighboring wavelengths are nearly identical.

Algorithm

```
Compute correlation matrix

↓

Remove highly correlated bands

↓

Keep representative bands
```

Threshold

```
0.98
```

or

```
0.99
```

---

### D. Mutual Information (optional)

If labels are available

Compute

```
MI(band, class)
```

Rank bands.

Keep top K.

Best supervised approach.

---

### E. Hybrid

Very strong option

```
Variance filter

↓

Correlation pruning

↓

Mutual Information ranking
```

---

## 1.5 Apply selection

Current

```
patch = cube[y:y+p, x:x+p, :]
```

New

```
patch = cube[y:y+p, x:x+p, selected_band_indices]
```

Nothing else changes.

---

## 1.6 Save metadata

Besides

```
wavelengths.npy
```

also save

```
selected_band_indices.npy

selected_wavelengths.npy

band_selection_config.json
```

Example

```json
{
  "method": "variance",
  "original_bands": 246,
  "selected_bands": 64,
  "selected_indices": [...]
}
```

Useful for reproducibility.

---

# Phase 2 — Balance Classes

## Goal

Generate balanced datasets before training.

Current pipeline writes every patch.

New pipeline balances classes before finalizing.

---

## 2.1 Add CLI

```
--balance_classes
```

choices

```
none
undersample
oversample
```

Default

```
none
```

---

## 2.2 Where balancing occurs

Do **not** balance while extracting patches.

Reason:

Streaming would become complicated.

Instead

```
Extract everything

↓

Finalize shards

↓

Balance

↓

Write final arrays
```

---

## 2.3 Why this location?

Current pipeline already performs a final unification step.

Balancing fits perfectly there.

No impact on

* lazy loading
* resumability
* shard writing

---

## 2.4 Undersampling

Current counts

```
Healthy

DCIS

IDC
```

Find

```
minimum class size
```

Randomly sample

```
min_size
```

from every class.

---

## 2.5 Oversampling

Duplicate minority examples until

```
all classes == max_size
```

Optional

```
sampling with replacement
```

---

## 2.6 Deterministic sampling

Always use

```
np.random.default_rng(seed)
```

so datasets are reproducible.

---

## 2.7 Balance only training

Never touch

Validation

Test

Otherwise evaluation becomes unrealistic.

---

## 2.8 Save report

```
dataset_statistics.json
```

Example

```json
Before

Healthy  42000

DCIS     17000

IDC      51000

After

Healthy 17000

DCIS    17000

IDC     17000
```

---

# Phase 3 — Flexible Dataset Splits

Current script appears to store only a train/test split using `test_size` in the manifest. 

Instead support configurable split presets.

---

## 3.1 CLI

Replace

```
--test_size
```

with

```
--split
```

Choices

```
80_20

80_10_10

70_15_15

70_30

60_20_20
```

---

## 3.2 Internal representation

Convert

```
80_20
```

to

```
train = 0.80

val = 0.00

test = 0.20
```

Likewise

```
80_10_10

↓

train = .80

val = .10

test = .10
```

---

## 3.3 Patient-level splitting

Continue splitting by

```
patient
```

never by patch.

This is already one of the strengths of the current preprocessing design and should be preserved. 

---

## 3.4 Folder layout

Current

```
train

test
```

New

```
train

validation

test
```

Validation only exists if requested.

---

## 3.5 Final outputs

For three-way splits

```
X_train.npy

y_train.npy

X_val.npy

y_val.npy

X_test.npy

y_test.npy
```

Two-way split

```
X_train.npy

y_train.npy

X_test.npy

y_test.npy
```

---

## 3.6 Manifest changes

Instead of

```
test_patients
```

store

```
train_patients

validation_patients

test_patients
```

---

## 3.7 Update shard writer

Support

```
train

validation

test
```

instead of

```
train

test
```

Minimal changes.

---

# Phase 4 — Dataset Statistics

Automatically generate

```
dataset_statistics.json
```

Include

```
Original bands

Selected bands

Band selection method

Class counts

Balanced counts

Train size

Validation size

Test size

Patients

Patch size

Stride

Random seed

ROI settings
```

---

# Phase 5 — Update Progress Manifest

Current manifest tracks processing arguments, wavelengths, shard IDs, completed captures, and finalization state. It should be extended so resumable preprocessing also captures the new preprocessing configuration. 

Add

```json
{
    "band_selection": "...",
    "selected_bands": [...],
    "balance_method": "...",
    "split": "80_10_10"
}
```

This prevents resuming a preprocessing run with incompatible settings.

---

# Phase 6 — Backward Compatibility

Maintain compatibility by using sensible defaults:

| Feature         | Default                                  |
| --------------- | ---------------------------------------- |
| Band selection  | Disabled (`none`)                        |
| Class balancing | Disabled (`none`)                        |
| Dataset split   | `80_20`                                  |
| Validation set  | Optional (only created for 3-way splits) |

With these defaults, existing commands should produce the same outputs as the current script.

---

# Recommended Implementation Order

1. **Flexible dataset splitting** – foundational change because it affects manifests, shard writing, and final dataset generation.
2. **Class balancing** – integrates naturally into the final unification step without affecting extraction.
3. **Spectral band selection** – the largest feature, requiring a two-pass preprocessing stage and updates to wavelength metadata.
4. **Dataset statistics and metadata** – add reporting and reproducibility artifacts.
5. **Compatibility testing** – verify resumable preprocessing, patient-level splitting, wavelength metadata, and output compatibility with `train_example-gemini-v4.py` and downstream training scripts.

---

## Expected Benefits

| Feature                               | Benefit                                                                                                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Spectral band selection               | Smaller `.npy` files (potentially 50–80% smaller), faster training, reduced redundancy, lower storage and memory requirements |
| Class balancing                       | Eliminates class imbalance bias and improves classifier robustness                                                            |
| Flexible train/validation/test splits | Supports proper hyperparameter tuning, reproducible experiments, and multiple evaluation protocols                            |
| Extended metadata                     | Improves reproducibility and experiment tracking                                                                              |
| Preserved streaming architecture      | Retains the script's existing memory-bounded, resumable processing while adding these capabilities                            |
