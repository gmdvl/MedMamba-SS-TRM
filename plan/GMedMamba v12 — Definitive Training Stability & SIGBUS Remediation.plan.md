The three errors should be treated as **one stability problem with two distinct failure paths**:

1. **Persistent non-finite gradients** → the training loop keeps skipping optimizer steps instead of fixing/stopping the numerical failure.
2. **DataLoader `/dev/shm` exhaustion** → workers die with `SIGBUS`, and the parent process can consequently report `SIGBUS`.
3. **Potential memory-mapped dataset `SIGBUS`** → `NpyDataset` uses `np.load(..., mmap_mode="r")`, so a damaged/truncated/inaccessible `.npy` backing file can also produce a genuine bus error outside the DataLoader-worker mechanism.

The current `train_example_v12.py` makes the DataLoader problem particularly likely: it defaults to **8 workers**, `prefetch_factor=4`, `persistent_workers=True`, and `pin_memory=True` on CUDA. That means the training and validation loaders can maintain a substantial amount of prefetched/staged data simultaneously. The current dataset also memory-maps the `.npy` files. Meanwhile, the trainer explicitly skips non-finite gradient batches and continues training. The manuscript already identifies non-finite-gradient handling as a critical experimental-validity issue. 

# GMedMamba v12 — Definitive Training Stability & SIGBUS Remediation Plan

## 1. Objectives

The implementation must satisfy all of the following:

### Numerical stability

* No endless stream of:
  `Non-finite gradient norm - skipping this batch's optimizer step.`
* Identify **which loss component or model parameter first becomes non-finite**.
* Prevent NaN/Inf values from entering model parameters.
* Do not allow an epoch with excessive skipped batches to be treated as healthy.
* Do not silently produce misleading training metrics.
* Automatically terminate a genuinely unstable run with a useful diagnostic.
* Make AMP behavior explicitly configurable.
* Stabilize the SAM reconstruction loss.
* Add gradient and activation diagnostics sufficient to identify the actual source.

### DataLoader / shared-memory stability

* Eliminate `/dev/shm` exhaustion under the default configuration.
* Remove unnecessary worker persistence.
* Reduce prefetching.
* Prevent both train and validation workers from independently consuming large amounts of shared memory.
* Provide a safe worker configuration that can be used regardless of GPU size.
* Keep higher-performance worker configurations available as opt-in modes.

### `SIGBUS` stability

The implementation must distinguish:

```text
DataLoader worker SIGBUS
        |
        +--> /dev/shm exhaustion
        |
        +--> worker/process memory pressure
        |
        +--> dataset mmap problem
```

from:

```text
Main process SIGBUS
        |
        +--> invalid/truncated mmap-backed .npy
        |
        +--> filesystem/storage problem
        |
        +--> severe system memory pressure
```

The solution therefore must not simply increase `/dev/shm` and declare the problem fixed.

---

# Phase 0 — Establish a Safe Execution Baseline

Before changing model architecture, make the training entry point capable of running in a **known-safe mode**.

## 0.1 Change the DataLoader defaults

### Current

`train_example_v12.py` currently has:

```text
--num_workers 8
```

and:

```text
pin_memory=True
persistent_workers=True
prefetch_factor=4
```

for CUDA.

### New safe defaults

Use:

```text
num_workers = 0
prefetch_factor = unavailable
persistent_workers = false
pin_memory = false
```

for the initial safe mode.

The important point is:

> **The first successful stability test must run without multiprocessing DataLoader workers.**

This completely removes `/dev/shm` worker IPC from the critical path.

Do not consider the SIGBUS problem fixed until:

```bash
python train_example_v12.py ... --num_workers 0
```

can run through multiple epochs without SIGBUS.

---

# Phase 1 — Build an Explicit DataLoader Memory Policy

Modify:

```text
train_example_v12.py
```

and preferably introduce:

```text
training/dataloader_config.py
```

instead of embedding all DataLoader decisions directly in the training script.

## 1.1 Add explicit controls

Add:

```text
--num_workers
--prefetch_factor
--persistent_workers
--pin_memory
--loader_mode
```

with:

```text
--loader_mode safe
```

as the default.

Recommended modes:

| Mode          |      Workers | Prefetch | Persistent | Pin memory |
| ------------- | -----------: | -------: | ---------- | ---------- |
| `safe`        |            0 |      N/A | No         | No         |
| `balanced`    |          1–2 |        1 | No         | Optional   |
| `performance` | user-defined |      1–2 | Optional   | Yes        |

Do **not** retain the current default of:

```text
8 workers × prefetch 4 × persistent workers
```

for this project.

---

# Phase 2 — Eliminate Excessive `/dev/shm` Pressure

## 2.1 Why the current configuration is dangerous

The current configuration effectively allows each worker to prepare multiple batches ahead of time.

With:

```text
num_workers = 8
prefetch_factor = 4
```

the loader can have approximately:

```text
8 × 4 = 32
```

prefetched batches per loader.

There are also separate train and validation loaders.

With HSI tensors, these batches can be substantially larger than ordinary RGB batches.

Therefore the failure:

```text
DataLoader worker ... killed by signal: Bus error
```

and:

```text
Unexpected bus error encountered in worker.
This might be caused by insufficient shared memory (shm).
```

is completely consistent with the current loader configuration.

---

## 2.2 Set prefetch to 1

Even when workers are enabled:

```text
prefetch_factor = 1
```

should be the maximum default.

Do not use `4` as the default.

---

## 2.3 Disable persistent workers by default

Change:

```python
persistent_workers=(args.num_workers > 0)
```

to an explicit configuration.

For example:

```text
persistent_workers=False
```

unless the user explicitly requests it.

This prevents workers and their associated resources from remaining alive unnecessarily across epochs.

---

## 2.4 Separate train and validation worker policies

Do not assume train and validation need identical worker configurations.

Recommended:

```text
train:
    workers = 0–2

validation:
    workers = 0–1
```

Validation does not require aggressive prefetching.

---

# Phase 3 — Add `/dev/shm` Monitoring

Create:

```text
training/system_memory.py
```

with monitoring for:

* `/dev/shm` total;
* `/dev/shm` used;
* `/dev/shm` available;
* system RAM;
* process RSS;
* CUDA allocated memory;
* CUDA reserved memory;
* CUDA peak memory.

Log once at startup:

```text
[system-memory]
RAM:
SHM:
GPU:
```

and periodically during training.

Before creating DataLoaders, report:

```text
/dev/shm available: XXX MB
```

This turns the current mysterious SIGBUS into an observable resource failure.

---

# Phase 4 — Add Dataset Storage Validation

This is essential because the main-process:

```text
fish: Job ... terminated by signal SIGBUS
```

cannot automatically be attributed to DataLoader `/dev/shm`.

The current `NpyDataset` uses:

```python
np.load(x_path, mmap_mode="r")
```

and:

```python
np.load(y_path, mmap_mode="r")
```

in `train_example_v6.py`.

That means the dataset is backed by memory-mapped files.

## 4.1 Create

```text
training/npy_integrity.py
```

It must validate:

```text
X_train.npy
y_train.npy
X_val.npy
y_val.npy
```

before workers are created.

Validate:

* file exists;
* file size > 0;
* valid `.npy` header;
* dtype;
* shape;
* number of samples;
* expected element count;
* expected byte size;
* X/y length equality;
* finite sampled values;
* random sample reads;
* first sample;
* middle sample;
* final sample.

---

# Phase 5 — Add an Explicit Dataset Loading Mode

Modify `NpyDataset` so that the user can choose between:

```text
mmap
ram
```

and eventually:

```text
safe
```

### `mmap`

Current behavior:

```python
np.load(path, mmap_mode="r")
```

### `ram`

Load normally:

```python
np.load(path)
```

This removes memory-mapped-file access from the training process.

### `safe`

Recommended behavior:

1. validate the `.npy`;
2. determine available RAM;
3. if sufficiently small, load into RAM;
4. otherwise use validated mmap mode.

Add:

```text
--dataset_storage {auto,mmap,ram}
```

with:

```text
auto
```

as the default after validation.

For debugging the SIGBUS problem, provide:

```bash
--dataset_storage ram
```

If the training process becomes completely stable in RAM mode, that is strong evidence that the mmap/storage path was involved.

---

# Phase 6 — Never Use Unsafe mmap Automatically After a Failed Validation

If dataset validation detects:

* invalid header;
* truncated file;
* impossible size;
* failed random access;
* unexpected EOF;

training must terminate with:

```text
DATASET_STORAGE_ERROR
```

rather than proceeding to a guaranteed `SIGBUS`.

Do not catch the resulting SIGBUS and continue.

The correct behavior is:

> detect the condition before memory-mapped training begins.

---

# Phase 7 — Add a DataLoader Worker Isolation Test

Before full training, add:

```text
--loader_test
```

which:

1. creates the dataset;
2. creates the DataLoader;
3. reads N batches;
4. reports:

   * batch shape;
   * dtype;
   * RAM;
   * SHM;
   * worker count;
   * elapsed time;
5. exits without constructing the GPU model.

Example:

```bash
python train_example_v12.py \
    --data_dir ... \
    --loader_test \
    --num_workers 0
```

Then:

```bash
python train_example_v12.py \
    --data_dir ... \
    --loader_test \
    --num_workers 1
```

Then:

```text
2 workers
```

This gives a controlled way to identify the first worker count that becomes unstable.

---

# Phase 8 — Add a Worker Initialization Function

Create a deterministic worker initialization function.

It should:

* seed NumPy;
* seed Python `random`;
* seed PyTorch;
* avoid inherited RNG state;
* optionally set thread counts.

This is particularly relevant because the augmentation implementation currently has special considerations for multi-worker DataLoaders, as documented in the repository.

---

# Phase 9 — Explicitly Control CPU Thread Oversubscription

Multiple DataLoader workers plus:

* NumPy;
* OpenMP;
* MKL;
* PyTorch CPU kernels;

can create excessive process/thread memory pressure.

At worker initialization, configure the worker environment conservatively.

Also expose:

```text
--cpu_threads
```

or document:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
```

as part of the reproducibility configuration.

---

# Phase 10 — Make AMP Explicit

The current trainer automatically selects:

```text
BF16 if supported
otherwise FP16
```

inside:

```text
training/trainerg_v5.py
```

and inherited by `trainerg_v6/v7/v8`.

This should not remain implicit while debugging persistent NaN/Inf gradients.

Add:

```text
--amp {off,bf16,fp16,auto}
```

Recommended default during remediation:

```text
--amp off
```

Once stability is established:

```text
--amp bf16
```

can be tested separately.

Only after BF16 is stable should FP16 be considered.

---

# Phase 11 — Create a Numerical-Stability Controller

Create:

```text
training/numerical_stability.py
```

This becomes the central mechanism for numerical health.

It should track:

```text
total_batches
valid_updates
skipped_updates
nonfinite_losses
nonfinite_gradients
consecutive_bad_batches
maximum_gradient_norm
minimum_gradient_norm
gradient_nan_parameters
gradient_inf_parameters
```

and:

```text
loss_components
```

including:

```text
classification
MSE
SAM
GAN
total
```

---

# Phase 12 — Do Not Treat All Non-Finite Gradients as Equivalent

The current code only checks:

```text
grad_norm
```

That is not sufficient.

A gradient norm of NaN tells us something is wrong, but not **where** it originated.

Add parameter-level inspection:

```text
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        check torch.isfinite(parameter.grad)
```

Record:

```text
first_bad_parameter
number_bad_parameters
nan_count
inf_count
maximum_abs_gradient
```

Example diagnostic:

```text
NONFINITE_GRADIENT
epoch=3
batch=17
parameter=backbone.spectral_pathway.encoder.blocks.2.ssm.dt_proj.weight
nan_elements=...
inf_elements=...
```

That is vastly more useful than:

```text
Non-finite gradient norm
```

---

# Phase 13 — Check Forward Activations

Add optional activation hooks.

Controlled by:

```text
--debug_numerics
```

When enabled, inspect major components:

```text
spectral pathway
spectral tokenizer
spectral encoder
SSM
fusion
spatial stages
classification head
reconstruction decoder
```

The diagnostic should identify the **first tensor** containing:

```text
NaN
Inf
```

rather than discovering the problem only after backward.

---

# Phase 14 — Check Every Loss Component Separately

The current trainer computes:

```text
classification loss
MSE loss
SAM loss
GAN loss
total loss
```

but only the combined loss is checked before backward.

Change the logic so each is independently tested:

```text
isfinite(cls_loss)
isfinite(mse_loss)
isfinite(sam_loss)
isfinite(gan_loss)
isfinite(total_loss)
```

Log something like:

```text
NONFINITE_LOSS:
classification=finite
MSE=finite
SAM=NONFINITE
GAN=not_used
total=NONFINITE
```

This is especially important because the reconstruction/SAM pathway is a plausible numerical source.

---

# Phase 15 — Stabilize SAM Loss

This is a high-priority change.

The current SAM implementation computes:

```text
acos(cosine_similarity)
```

and clamps the cosine extremely close to ±1.

The derivative of `acos` becomes very large near ±1.

That can produce enormous gradients even when the forward loss itself is finite.

## Replace the current formulation with a numerically safer implementation

Requirements:

* calculate the SAM computation in FP32;
* clamp cosine with a safer epsilon;
* protect vector norms;
* handle zero/near-zero spectral vectors;
* never return NaN/Inf;
* test forward and backward explicitly.

Do not simply use:

```text
eps = 1e-8
```

and assume that is sufficient.

The loss should have tests for:

```text
zero vector
identical vector
near-identical vector
orthogonal vector
large magnitude
small magnitude
mixed precision
```

---

# Phase 16 — Separate Classification and Reconstruction Numerical Tests

Before enabling the full objective:

### Test A

```text
classification only
lambda_mse = 0
lambda_sam = 0
lambda_gan = 0
```

### Test B

```text
classification + MSE
```

### Test C

```text
classification + SAM
```

### Test D

```text
classification + MSE + SAM
```

### Test E

```text
full objective
```

This is essential.

If:

```text
A = stable
B = stable
C = unstable
```

the source is immediately identified as SAM.

If:

```text
A = unstable
```

then reconstruction is not the primary cause and the investigation moves into the GMedMamba forward/backward path.

---

# Phase 17 — Add a One-Batch Numerical Smoke Test

Before training starts, run:

```text
1 forward
1 loss calculation
1 backward
1 gradient check
1 optimizer step
```

on one batch.

Require:

```text
finite input
finite logits
finite reconstruction
finite each loss
finite total loss
finite gradients
finite updated parameters
```

If any condition fails:

```text
ABORT BEFORE EPOCH 1
```

This prevents wasting an entire run producing hundreds of:

```text
Non-finite gradient norm
```

warnings.

---

# Phase 18 — Verify Parameters After Every Optimizer Step

After:

```text
optimizer.step()
```

perform an optional/debug parameter health check.

If any parameter becomes:

```text
NaN
Inf
```

the experiment must stop immediately.

Otherwise a single corrupted parameter can poison every future batch.

---

# Phase 19 — Improve Gradient Clipping

The current code performs:

```text
clip_grad_norm_
```

and then checks whether the resulting norm is finite.

Change the sequence to:

1. unscale;
2. inspect gradients;
3. calculate raw norm;
4. reject non-finite gradients;
5. clip finite gradients;
6. verify clipped gradients;
7. optimizer step.

This makes diagnostics distinguish:

```text
gradient already NaN
```

from:

```text
gradient finite but excessively large
```

---

# Phase 20 — Add Gradient Explosion Thresholds

Introduce:

```text
--max_gradient_norm
--max_gradient_skip_ratio
--max_consecutive_bad_batches
```

Recommended initial values:

```text
max_gradient_norm = 1.0
max_gradient_skip_ratio = 0.10
max_consecutive_bad_batches = 3
```

These should be configurable rather than hardcoded.

---

# Phase 21 — Change the Meaning of "Skip Batch"

The current behavior is:

```text
bad gradient
    ↓
skip optimizer
    ↓
continue
    ↓
possibly skip many more
    ↓
eventually continue into next epoch
```

That is too permissive.

Use:

```text
isolated bad batch
    ↓
skip
    ↓
reset gradients
    ↓
continue

persistent bad batches
    ↓
abort epoch/run
    ↓
save diagnostic checkpoint
    ↓
do not select checkpoint
```

---

# Phase 22 — Introduce Epoch Health Thresholds

An epoch should not be considered healthy merely because:

```text
valid_updates > 0
```

The existing implementation correctly prevents a zero-update epoch from becoming the best epoch, but it is still too weak for persistent numerical instability. The manuscript already identifies this area as a critical validity issue. 

Add:

```text
skip_ratio
```

classification:

| Skip ratio | Status   |
| ---------: | -------- |
|         0% | Healthy  |
|       0–1% | Warning  |
|       1–5% | Degraded |
|      5–10% | Critical |
|       >10% | Invalid  |
|       100% | Fatal    |

These thresholds should be configurable.

---

# Phase 23 — Stop the Run on Persistent Numerical Failure

The run should terminate if:

```text
N consecutive bad batches
```

or:

```text
epoch skip ratio > threshold
```

or:

```text
N consecutive unhealthy epochs
```

or:

```text
parameter becomes NaN/Inf
```

or:

```text
forward output becomes NaN/Inf
```

The output should clearly state:

```text
TRAINING_ABORTED_NUMERICAL_INSTABILITY
```

rather than simply returning a partially completed experiment.

---

# Phase 24 — Add Automatic Stability Fallbacks

Implement a controlled fallback sequence.

### Attempt 1

```text
AMP off
SAM off
GAN off
gradient clipping 1.0
```

If stable:

### Attempt 2

```text
SAM enabled
```

If stable:

### Attempt 3

```text
AMP BF16
```

If stable:

### Attempt 4

```text
full configuration
```

Do not silently change the experiment.

Every fallback must be recorded in:

```text
stability_report.json
```

---

# Phase 25 — Investigate the GMedMamba Spectral Pathway Separately

The spectral pathway processes:

```text
N = batch_size × Hp × Wp
```

patch sequences.

The repository already contains `training/spectral_checkpoint.py`, whose purpose is to checkpoint the per-chunk spectral processing because ordinary chunking does not remove the total training autograd memory requirement.

That mechanism should remain enabled for large HSI runs.

However:

> **Gradient checkpointing addresses GPU activation memory. It does not fix DataLoader shared-memory SIGBUS.**

These must remain separate fixes.

---

# Phase 26 — Make Gradient Checkpointing Automatic in Safe Mode

For HSI configurations:

```text
--gradient_checkpointing
```

should be strongly recommended, and optionally automatically enabled under:

```text
--memory_mode safe
```

The current implementation already wraps both the backbone stages and spectral pathway before the reconstruction wrapper, which is the correct placement.

---

# Phase 27 — Reduce Batch Size Dynamically

Add:

```text
--auto_batch_size
```

or a batch-size safety mechanism.

Start with:

```text
requested batch size
```

and progressively reduce:

```text
128 → 96 → 64 → 48 → 32 → 16 → 8 → 4 → 2 → 1
```

if GPU memory allocation fails.

This should be logged explicitly.

Do not dynamically change batch size halfway through an ordinary scientific experiment without recording it.

---

# Phase 28 — Distinguish GPU OOM from System SIGBUS

The training diagnostics must classify failures into:

```text
CUDA_OOM
DATALOADER_SHM
DATASET_MMAP_SIGBUS
SYSTEM_MEMORY_PRESSURE
NUMERICAL_INSTABILITY
UNKNOWN_SIGBUS
```

The final report should never simply say:

```text
training failed
```

---

# Phase 29 — Add System-Level Preflight

Before training:

```text
GPU VRAM
system RAM
/dev/shm
dataset sizes
dataset storage mode
DataLoader configuration
batch size
image size
channels
```

should be printed.

Example:

```text
===== GMedMamba Stability Preflight =====
Dataset:
  X_train: ...
  X_val: ...
  Channels: ...
  Resolution: ...

DataLoader:
  workers: 0
  prefetch: disabled
  persistent: false
  pin_memory: false

System:
  RAM available: ...
  /dev/shm available: ...

GPU:
  VRAM total: ...
  VRAM free: ...

Numerical:
  AMP: OFF
  gradient clipping: 1.0
  SAM: ON
  GAN: OFF
==========================================
```

---

# Phase 30 — Add a Dedicated `--safe_mode`

This should be the primary user-facing solution.

```bash
python train_example_v12.py \
    --data_dir ... \
    --safe_mode
```

Safe mode should automatically select:

```text
num_workers=0
prefetch=none
persistent_workers=false
pin_memory=false
AMP=off
gradient checkpointing=on
gradient clipping=1.0
numerical diagnostics=on
dataset validation=on
```

The goal is:

> **If safe mode cannot complete, the failure must be diagnostic rather than another unexplained SIGBUS/NaN cascade.**

---

# Phase 31 — Add a Performance Mode Separately

After safe mode is proven stable:

```text
--performance_mode
```

can enable:

```text
workers 1–2+
pin_memory
BF16
prefetch 1
```

Performance mode must never silently fall back to the unsafe current configuration.

---

# Phase 32 — Add `/dev/shm` Remediation Documentation

The project should document two levels of remediation.

### Application-level

Preferred:

```text
num_workers=0
prefetch_factor=1
persistent_workers=false
```

### System-level

If multiprocessing is required, the environment may need a larger shared-memory allocation.

For example, containerized execution commonly requires an explicitly increased shared-memory allocation.

However:

> Increasing `/dev/shm` must be considered a secondary optimization, not the primary software fix.

The code must remain safe when `/dev/shm` is small.

---

# Phase 33 — Add Automated Reproduction Tests

Create:

```text
tests/test_dataloader_stability.py
tests/test_dataset_integrity.py
tests/test_numerical_stability.py
tests/test_sam_loss.py
tests/test_gradient_health.py
```

## DataLoader tests

Test:

```text
workers=0
workers=1
workers=2
```

with:

```text
prefetch=1
```

and confirm no worker crashes.

## Dataset tests

Test:

```text
mmap
RAM
auto
```

and random-access reads.

## Numerical tests

Verify:

```text
finite forward
finite losses
finite gradients
finite parameters
```

---

# Phase 34 — Add a 10-Batch Stability Test

Before any expensive experiment, run:

```text
10 training batches
```

and require:

```text
10 valid updates
0 NaN losses
0 Inf losses
0 NaN gradients
0 Inf gradients
0 corrupted parameters
0 DataLoader errors
```

Then run:

```text
100 batches
```

before allowing a multi-epoch experiment.

---

# Phase 35 — Add a Full Stability Qualification Test

A configuration should only be marked:

```text
STABLE
```

after completing:

```text
≥3 epochs
```

with:

```text
0 DataLoader SIGBUS
0 worker crashes
0 dataset access errors
0 parameter NaN/Inf
0 non-finite losses
0 persistent gradient failures
```

A small number of isolated numerical skips should still cause a warning and be recorded.

---

# Phase 36 — Preserve Scientific Validity

This is especially important for the thesis.

Do **not** simply hide the warning by changing:

```text
logger.warning(...)
```

to nothing.

The experiment history must contain:

```json
{
  "total_batches": 100,
  "valid_updates": 98,
  "skipped_updates": 2,
  "skip_ratio": 0.02,
  "nonfinite_loss": 0,
  "nonfinite_gradients": 2,
  "is_valid_epoch": true
}
```

If an epoch is invalid:

```json
{
  "is_valid_epoch": false
}
```

and it must not become the best checkpoint.

The manuscript explicitly identifies non-finite-gradient handling as a critical validity problem because invalid training can otherwise produce misleading training metrics. 

---

# Phase 37 — Add Failure Artifacts

Whenever numerical instability occurs, save:

```text
numerical_failure/
    failure.json
    batch_info.json
    loss_components.json
    gradient_summary.json
    parameter_summary.json
    model_config.json
    dataloader_config.json
    system_memory.json
    gpu_memory.json
```

For the first failure, also optionally save:

```text
failed_batch.pt
```

with:

```text
x
y
```

so the exact failing batch can be reproduced.

---

# Phase 38 — Add Reproducibility Metadata

`config.json` must include:

```text
num_workers
prefetch_factor
persistent_workers
pin_memory
dataset_storage
AMP mode
gradient checkpointing
gradient clipping
SAM epsilon
SAM loss configuration
batch size
image resolution
channels
GPU
CUDA version
PyTorch version
```

This is important because otherwise a future experiment may appear to reproduce the result while using a completely different memory/runtime configuration.

---

# Phase 39 — Recommended File Changes

## Modify

### `train_example_v12.py`

Add:

* safe mode;
* DataLoader configuration;
* dataset storage configuration;
* AMP configuration;
* stability configuration;
* loader test;
* preflight diagnostics.

---

### `training/data.py`

Add:

* safe dataset loading;
* mmap validation;
* RAM mode;
* auto storage mode;
* integrity checks.

---

### `train_example_v6.py`

Because `NpyDataset` is currently defined there and imported by v12, either:

**preferred:**

move the shared dataset implementation into:

```text
training/data.py
```

and make the old implementation a compatibility wrapper,

or modify the existing implementation directly.

---

### `training/trainerg_v5.py`

This is the most important trainer change.

Modify:

* loss checks;
* gradient checks;
* parameter checks;
* skip policy;
* epoch validity;
* failure thresholds;
* numerical diagnostics;
* stopping conditions.

---

### `training/trainerg_v6.py`

Ensure the pluggable criterion does not bypass the numerical checks.

In particular, check:

```text
criterion output
MSE
SAM
GAN
total loss
```

individually.

---

### `training/gradient_health.py`

Extend it from simple counters to:

```text
failure categories
first failure
bad parameter names
maximum gradient
consecutive failures
epoch health classification
```

---

### `training/gan.py`

Stabilize:

```text
SAMLoss
```

and ensure it is numerically safe in FP32.

---

### `training/spectral_checkpoint.py`

Keep the current mechanism, but add:

* configuration reporting;
* verification that wrapping actually occurred;
* optional chunk-size configuration;
* tests.

---

## Add

```text
training/system_memory.py
training/npy_integrity.py
training/dataloader_config.py
training/numerical_stability.py
tests/test_dataloader_stability.py
tests/test_dataset_integrity.py
tests/test_numerical_stability.py
tests/test_sam_loss.py
tests/test_gradient_health.py
```

---

# Phase 40 — Exact Implementation Order

This order is important.

## Stage 1 — Stop SIGBUS

First implement:

```text
num_workers=0
prefetch disabled
persistent_workers=false
pin_memory=false
```

Then verify training.

---

## Stage 2 — Validate datasets

Implement:

```text
npy_integrity.py
```

and verify every `.npy`.

Then test:

```text
--dataset_storage ram
```

If RAM mode eliminates the main-process SIGBUS, investigate the mmap/storage path before continuing.

---

## Stage 3 — Add system monitoring

Add:

```text
RAM
/dev/shm
GPU
```

monitoring.

---

## Stage 4 — Fix numerical instability

Implement:

```text
loss-by-loss checks
parameter-level gradient checks
activation checks
SAM stabilization
AMP control
```

---

## Stage 5 — Enforce failure policy

Replace:

```text
skip forever
```

with:

```text
skip isolated failure
abort persistent failure
```

---

## Stage 6 — Run numerical smoke test

Before epoch 1:

```text
forward
loss
backward
gradient check
optimizer
parameter check
```

---

## Stage 7 — Run 10-batch test

Require:

```text
10/10 valid updates
```

---

## Stage 8 — Run 100-batch test

Require no:

```text
SIGBUS
NaN
Inf
worker death
```

---

## Stage 9 — Run 3-epoch qualification

Only then run a normal experiment.

---

## Stage 10 — Re-enable performance features one at a time

Order:

```text
safe CPU loading
↓
1 worker
↓
2 workers
↓
pin_memory
↓
BF16
↓
persistent workers
```

Never enable all of them simultaneously.

---

# Phase 41 — Acceptance Criteria

The remediation is **not complete** merely because the original warning disappears.

It is complete only when all of these pass.

### DataLoader

* [ ] `num_workers=0` runs successfully.
* [ ] `num_workers=1` runs successfully.
* [ ] `num_workers=2` runs successfully if system SHM permits.
* [ ] No worker SIGBUS.
* [ ] No `/dev/shm` exhaustion.
* [ ] No persistent-worker leaks.

### Dataset

* [ ] All `.npy` files pass integrity validation.
* [ ] mmap mode works.
* [ ] RAM mode works.
* [ ] auto mode selects a safe strategy.
* [ ] No unexplained main-process SIGBUS.

### Numerical stability

* [ ] One-batch smoke test passes.
* [ ] 10-batch test passes.
* [ ] 100-batch test passes.
* [ ] No non-finite loss.
* [ ] No persistent non-finite gradients.
* [ ] No NaN/Inf model parameters.
* [ ] SAM backward test passes.
* [ ] AMP-off test passes.
* [ ] BF16 test passes if supported.

### Training correctness

* [ ] Invalid epochs cannot become best epochs.
* [ ] Excessive skip ratios invalidate an epoch.
* [ ] Persistent instability terminates the run.
* [ ] Training metrics never report fake zero metrics for an all-skipped epoch.
* [ ] Failure artifacts are written.

### Reproducibility

* [ ] Complete DataLoader configuration is saved.
* [ ] Numerical configuration is saved.
* [ ] Dataset storage mode is saved.
* [ ] AMP mode is saved.
* [ ] System/GPU information is saved.

---

# Phase 42 — Recommended Initial Command

For the **first post-fix qualification run**, deliberately use the most conservative configuration:

```bash
python train_example_v12.py \
    --data_dir <DATASET> \
    --epochs 3 \
    --batch_size <SMALL_SAFE_BATCH> \
    --num_workers 0 \
    --recon_mode latent \
    --lambda_gan 0 \
    --amp off \
    --gradient_checkpointing
```

The exact CLI names for `--amp` and the other new options should be implemented as part of this plan rather than assumed to exist beforehand.

Once this passes, progressively reintroduce:

```text
SAM
→ BF16
→ workers
→ pin_memory
→ larger batch
```

---

# Root-Cause Priority

Based on the current v12 implementation, I would prioritize the investigation in this exact order:

| Priority | Problem                         | Why                                                         |
| -------- | ------------------------------- | ----------------------------------------------------------- |
| **P0**   | DataLoader `/dev/shm`           | Directly explains the worker SIGBUS messages                |
| **P0**   | mmap dataset safety             | Possible explanation for main-process SIGBUS                |
| **P0**   | Persistent non-finite gradients | Currently known to occur repeatedly                         |
| **P0**   | SAM numerical stability         | `acos`-based loss can create very large derivatives near ±1 |
| **P1**   | AMP mode                        | Can amplify numerical problems                              |
| **P1**   | Gradient/activation diagnostics | Needed to identify the first bad tensor                     |
| **P1**   | Skip-ratio enforcement          | Prevents invalid training from continuing                   |
| **P1**   | Spectral checkpointing          | Needed for GPU activation memory, but not a `/dev/shm` fix  |
| **P2**   | Worker performance tuning       | Only after safe mode is stable                              |
| **P2**   | Larger batch optimization       | Only after stability qualification                          |

## Most important architectural change

The key principle is:

> **Do not try to fix this by increasing batch size, increasing `/dev/shm`, increasing gradient clipping, or simply suppressing the warnings.**

There are currently **two resource-management layers and one numerical layer** that need to be independently controlled:

```text
                    GMedMamba v12
                         │
            ┌────────────┴────────────┐
            │                         │
       DATA PIPELINE             GPU/TRAINING
            │                         │
     ┌──────┴──────┐          ┌───────┴────────┐
     │             │          │                │
   /dev/shm      mmap       VRAM          numerical
     │             │          │                │
 workers       .npy files  activations      NaN/Inf
 prefetch      storage     checkpointing    gradients
 pin_memory                batch size        SAM/AMP
```

The **DataLoader SIGBUS must be eliminated independently of the GPU memory solution**, and the **non-finite-gradient loop must be made fail-safe independently of both**.

That gives you a much stronger thesis-grade implementation: a run either trains with demonstrably valid numerical and data-pipeline health, or it stops and records exactly why it could not. This directly addresses the training-stability and experimental-validity concerns already identified in the manuscript.  
