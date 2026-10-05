Below is a comprehensive development plan to transform `train_example.py` from a simple demonstration script into a **research-grade training framework** suitable for a Master's thesis or publication. The plan is organized by implementation phases so that each phase is independently testable.

---

# Phase 1 – Project Architecture Refactoring (Highest Priority)

## Objectives

* Convert `train_example.py` from a demo into a modular training framework.
* Separate responsibilities.
* Make future additions easier.

## Proposed structure

```text
train_example.py                 # CLI only

training/
│
├── trainer.py                   # Training loop
├── evaluator.py                 # Validation/testing
├── metrics.py                   # Metrics
├── checkpoint.py                # Saving/resuming
├── logger.py                    # CSV/JSON/TensorBoard logging
├── report.py                    # Final report generation
├── plots.py                     # Graph generation
├── profiler.py                  # Speed/GPU profiling
├── experiment.py                # Experiment manager
└── utils.py
```

This eliminates the current monolithic design.

---

# Phase 2 – Experiment Manager

Automatically create an experiment folder.

Example

```text
experiments/
    2026-08-02_21-41-18/
```

Inside

```text
config.yaml

README.txt

logs/

plots/

checkpoints/

predictions/

reports/

tensorboard/
```

Each experiment becomes fully reproducible.

---

# Phase 3 – Automatic Configuration Saving

Store

* CLI arguments
* Git hash
* CUDA version
* PyTorch version
* NumPy version
* hostname
* GPU name
* CPU model
* RAM
* OS
* preprocessing parameters
* dataset statistics

Save as

```text
config.yaml
```

---

# Phase 4 – Robust Logging

Current

```
print(...)
```

Replace with

```
logging.Logger
```

Outputs

```
console

train.log

history.csv

metrics.json
```

Every epoch automatically recorded.

---

# Phase 5 – Full Checkpoint System

Current

```
best.pt
```

only.

Replace with

```
latest.pt

best.pt

epoch_005.pt

epoch_010.pt

epoch_015.pt
...
```

Checkpoint should include

```
model_state

optimizer_state

scheduler_state

scaler_state

epoch

global_step

best_accuracy

history

random_state

numpy_state

torch_rng

cuda_rng
```

---

# Phase 6 – Resume Interrupted Training

Current

Restart from epoch 1.

New

```
python train_example.py ...

Found interrupted experiment.

Resume?

[Y/n]
```

or

```
--resume
```

Automatically restores

* epoch
* optimizer
* scheduler
* GradScaler
* RNG

Training continues exactly.

---

# Phase 7 – Automatic Recovery

Handle

```
CTRL+C

Out Of Memory

Power outage

SSH disconnect
```

Before exiting

```
save latest checkpoint
```

No progress lost.

---

# Phase 8 – CSV History

history.csv

Example

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | LR | GPU MB | Epoch Time |
| ----- | ---------- | -------- | --------- | ------- | -- | ------ | ---------- |

Updated every epoch.

---

# Phase 9 – JSON Metrics

metrics.json

Contains

```
best epoch

best accuracy

best loss

total training time

average epoch time

GPU

dataset

etc
```

---

# Phase 10 – Classification Metrics

After validation

Compute

Accuracy

Balanced Accuracy

Precision

Recall

F1

Macro F1

Weighted F1

ROC-AUC

PR-AUC

Kappa

Matthews Correlation

Confusion Matrix

Classification Report

---

# Phase 11 – Error Metrics

Store

Cross Entropy

Log Loss

Calibration Error

Brier Score

ECE

MCE

NLL

---

# Phase 12 – Spectral Metrics

## RGB

Skip automatically.

## HSI

If ground-truth spectra exist

Compute

SAM

SID

RMSE

MAE

Relative RMSE

Cosine Similarity

Spectral Correlation

Spectral Energy Error

If reconstruction is unavailable

Generate

```
spectral_metrics.json
```

with

```
status:
Not applicable.

Reason:
Classification model does not reconstruct spectra.
```

No fake values.

---

# Phase 13 – Band Selection Metrics

For your thesis these are actually more useful.

Store

Selected wavelengths

Band importance

Band attention

Selection frequency

Selection stability

Average selected wavelength

Spectral coverage

Channel sparsity

Energy retained

Redundancy score

These directly support the research contribution.

---

# Phase 14 – Performance Metrics

Every epoch

Images/sec

Samples/sec

GPU utilization

CPU utilization

RAM

VRAM

Disk throughput

Data loading time

Forward time

Backward time

Optimizer step time

Validation time

---

# Phase 15 – GPU Profiling

Use

```
torch.cuda.memory_allocated()

torch.cuda.memory_reserved()

torch.cuda.max_memory_allocated()
```

Generate

```
gpu_usage.csv
```

---

# Phase 16 – TensorBoard

Automatically save

Loss

Accuracy

Learning Rate

GPU memory

Weights

Gradients

Histograms

Embeddings

Confusion Matrix

ROC

---

# Phase 17 – Plot Generation

Automatically generate

```
loss.png

accuracy.png

lr.png

gpu_memory.png

epoch_time.png

confusion_matrix.png

roc.png

precision_recall.png
```

---

# Phase 18 – Prediction Export

Export

```
sample

true label

predicted label

probabilities
```

CSV

Useful for analysis.

---

# Phase 19 – PDF Report

Automatically generate

```
experiment_report.pdf
```

Contents

Experiment summary

Hardware

Hyperparameters

Training curves

Metrics

Confusion matrix

GPU usage

Training time

Dataset statistics

---

# Phase 20 – Training Speed Optimization

This is where the largest improvements can be made.

## Mixed Precision (Highest Priority)

Use

```
torch.cuda.amp.autocast()

GradScaler
```

Expected

30–50% less memory

20–60% faster

---

## Torch Compile

PyTorch 2.x

```
model = torch.compile(model)
```

Usually

10–30%

speedup.

---

## Channels Last Memory Format

```
model.to(memory_format=torch.channels_last)

input.to(memory_format=torch.channels_last)
```

Benefits CNN-heavy workloads.

---

## Pinned Memory

```
pin_memory=True
```

in DataLoader.

---

## Persistent Workers

```
persistent_workers=True
```

Avoids worker restart every epoch.

---

## Prefetch Factor

Increase

```
prefetch_factor
```

to overlap loading with GPU.

---

## Non-blocking GPU transfers

Instead of

```
x = x.to(device)
```

Use

```
x = x.to(device, non_blocking=True)
```

---

## Gradient Accumulation

Allows effective larger batches while fitting GPU memory.

---

## Gradient Checkpointing

Very important for Mamba.

Expected

30–60%

memory reduction.

---

## Fused AdamW

If supported

```
torch.optim.AdamW(fused=True)
```

---

## CUDA Benchmark

```
torch.backends.cudnn.benchmark=True
```

For fixed image size.

---

## TF32

RTX GPUs

```
torch.backends.cuda.matmul.allow_tf32=True
```

Can improve throughput with minimal accuracy impact for many models.

---

## Efficient DataLoader

Tune

```
num_workers

prefetch_factor

persistent_workers
```

Automatically based on CPU.

---

## Asynchronous Checkpoint Saving

Currently

training pauses

↓

save

↓

continue

Instead

```
background thread

save checkpoint
```

Training continues.

---

## Early Stopping

Stop after

```
patience=20
```

without improvement.

---

## Automatic Best Model Selection

Keep

Top 3

models.

---

## Learning Rate Finder

Optional

Automatically estimate optimal LR before training.

---

## Cosine Warmup

Replace

simple cosine scheduler

with

```
Warmup

↓

Cosine Annealing
```

Better convergence.

---

# Phase 21 – Multi-GPU Support

Support

```
1 GPU

DDP

DataParallel
```

Automatically.

---

# Phase 22 – Deterministic Reproducibility

Store

```
seed

torch

numpy

python

CUDA RNG
```

Guarantee reproducible experiments.

---

# Phase 23 – Command-Line Improvements

Add options such as:

```
--resume
--experiment-name
--save-every
--early-stop
--mixed-precision
--compile
--channels-last
--tensorboard
--profile
--export-predictions
--report
--seed
```

---

# Recommended implementation order

1. **Experiment manager** (directory structure and configuration)
2. **Checkpointing and resume support**
3. **CSV/JSON logging**
4. **Mixed precision, `torch.compile`, DataLoader optimizations, and gradient checkpointing**
5. **Classification metrics and automatic plots**
6. **TensorBoard integration**
7. **Prediction export and PDF report**
8. **Band-selection metrics and HSI-specific reporting**
9. **Multi-GPU support and advanced profiling**

This sequence prioritizes reliability and reproducibility first, then training speed, and finally advanced reporting. It also addresses the current limitations you've encountered, such as interrupted training, lack of experiment tracking, and high GPU memory usage, while laying the groundwork for evaluating your spectral band-selection research.
