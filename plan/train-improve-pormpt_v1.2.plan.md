# Architecture Goal (Updated)

The entire training framework should be designed around the **preprocessed `.npy` datasets** produced by `prepare_histologyhsi_bc-new.py`.

The training script **must not assume raw images** or `torchvision.datasets.ImageFolder`.

Instead, it should assume preprocessing has already been completed.

```text
Raw HistologyHSI-BC Dataset
            │
            ▼
prepare_histologyhsi_bc-new.py
            │
            ▼
─────────────────────────────────────
RGB
    X_train.npy
    y_train.npy
    X_test.npy
    y_test.npy

HSI
    X_train.npy
    y_train.npy
    X_test.npy
    y_test.npy
    wavelengths.npy
─────────────────────────────────────
            │
            ▼
train_example.py
            │
            ▼
Experiment outputs
```

This should become the official workflow.

---

# Phase 0 – Standardize the Data Pipeline (New Phase)

## Objective

Make `.npy` files the **only supported internal training format**.

The training framework should never need to know where the original images came from.

Everything begins from

```text
X_train.npy
y_train.npy

X_test.npy
y_test.npy
```

Optional

```text
wavelengths.npy
```

for HSI.

---

## Supported input tensor formats

### RGB

```python
X_train.shape

[N,H,W,3]
```

dtype

```python
float32
```

Labels

```python
y_train.shape

[N]
```

dtype

```python
int64
```

---

### HSI

```python
X_train.shape

[N,H,W,C]
```

where

```text
C = arbitrary
```

The model should automatically determine

```python
C = X_train.shape[-1]
```

No hardcoded number of bands.

---

## Validation

Before training begins

Automatically verify

```text
✓ files exist

✓ shapes

✓ dtype

✓ no NaNs

✓ no Infs

✓ label range

✓ class balance

✓ wavelengths length

✓ channels match wavelengths
```

Generate

```text
dataset_summary.json
```

---

# Phase 1 – Replace ImageFolder Completely

The current script still contains

```python
torchvision.datasets.ImageFolder(...)
```

This should be removed from the primary workflow.

Instead

```text
HSIPatchDataset

↓

GenericNPYDataset
```

Both RGB and HSI should use exactly the same Dataset class.

---

## Generic Dataset

Instead of

```text
RGB Dataset

HSI Dataset
```

Use

```python
NPYDataset
```

Supporting

```python
[N,H,W,C]
```

for

```text
RGB

HSI

Multispectral

Future datasets
```

Only

```python
C
```

changes.

---

# Automatic Dataset Discovery

Instead of requiring

```bash
--x_train

--y_train

--x_test

--y_test
```

allow

```bash
python train_example.py \
    --data_dir ./data_rgb/rgb
```

The script automatically finds

```text
X_train.npy

y_train.npy

X_test.npy

y_test.npy
```

If present

```text
wavelengths.npy
```

load automatically.

This makes the CLI much cleaner.

---

# Dataset Metadata

Every experiment should save

```text
dataset.json
```

Example

```json
{
  "dataset_type": "RGB",
  "samples_train": 8670,
  "samples_test": 1785,
  "patch_size": 600,
  "channels": 3,
  "classes": 3,
  "class_names": [
      "Healthy",
      "DCIS",
      "IDC"
  ],
  "dtype": "float32"
}
```

For HSI

```json
{
  "channels": 275,
  "sensor_range": [
      400,
      1000
  ]
}
```

---

# Memory-Mapped Dataset

Currently

```python
np.load(..., mmap_mode="r")
```

This is good.

Improve further.

Keep everything memory mapped.

Never

```python
X = np.load(...)
```

for huge datasets.

Always

```python
np.load(..., mmap_mode="r")
```

unless

```text
cache_dataset=True
```

---

# Smart Dataset Caching

Small datasets

```text
RAM cache
```

Large datasets

```text
Memory mapping
```

Automatically choose.

---

# Automatic Dataset Statistics

Before training

Generate

```text
dataset_report.json
```

Including

Number of samples

Patch size

Channels

Class distribution

Mean

Std

Min

Max

Dynamic range

Memory footprint

Expected GPU memory

---

# Automatic Batch Size Estimation

Since training always starts from

```text
.npy
```

Estimate

```text
patch size

channels

GPU memory
```

Automatically suggest

```text
Recommended batch size:

1

2

4

8
```

instead of crashing.

---

# Resume Training

Checkpoint should also save

```text
dataset checksum

dataset path

dataset statistics
```

When resuming

Verify

```text
same dataset

same labels

same wavelengths

same channels
```

before continuing.

---

# Training Report

Clearly state

```text
Input format:

NumPy tensors (.npy)

Training dataset:

X_train.npy

Validation dataset:

X_test.npy

Patch size:

600×600

Channels:

3

Memory mapping:

Enabled
```

instead of mentioning images.

---

# Performance Optimization for `.npy` Training

Since training is now exclusively based on `.npy` tensors, optimization efforts should focus on this data format rather than image decoding:

* Memory-mapped loading (`np.load(..., mmap_mode="r")`) for large datasets.
* Configurable RAM caching for datasets that fit comfortably in memory.
* Asynchronous prefetching to overlap data loading with GPU computation.
* Optimized `DataLoader` settings (`pin_memory`, `persistent_workers`, `prefetch_factor`, `non_blocking=True`).
* Mixed precision (AMP), `torch.compile`, channels-last memory format, and gradient checkpointing to reduce GPU memory use and improve throughput.
* Automatic batch-size estimation based on input tensor dimensions (`H×W×C`) and available GPU memory to avoid out-of-memory errors.
* Optional conversion of the `.npy` dataset into chunked formats (such as Zarr or HDF5) for extremely large datasets if profiling shows that random access to very large `.npy` files becomes a bottleneck.

---

# Final Architecture Goal

The finished framework should follow a single, consistent workflow:

```text
Raw HistologyHSI-BC Dataset
        │
        ▼
prepare_histologyhsi_bc-new.py
        │
        ▼
Preprocessed NumPy Dataset
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
└── wavelengths.npy (HSI only)
        │
        ▼
train_example.py
        │
        ▼
Automatic Experiment Manager
├── Checkpoints
├── Logs
├── Metrics
├── Plots
├── TensorBoard
├── Reports
└── Resume State
```

This removes the ambiguity between image-based and tensor-based training, makes the preprocessing script the single source of truth for dataset preparation, and ensures that all subsequent training, evaluation, checkpointing, and reporting operate directly on the standardized `.npy` tensors.
