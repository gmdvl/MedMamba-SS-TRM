Based on the uploaded scripts, the project already has a fairly mature training pipeline with experiment management, checkpoint infrastructure, reporting, plotting, evaluation, TensorBoard logging, and resumable training. The missing work is primarily in the training control logic (`training/trainer.py`), checkpoint metadata, reporting, and metric computation rather than the model itself. `train_example.py` is already delegating almost all training behavior to `Trainer.fit()`, so most modifications should remain inside the `training/` package rather than expanding the CLI.  

# Final Implementation Plan

## Goal

Update the training pipeline so that every epoch becomes a complete recoverable experiment checkpoint with fully validated metrics and automatic detection of failed training runs.

The revised pipeline must:

* save a complete checkpoint every epoch
* generate a complete epoch report every epoch
* eliminate NaN/0 metric problems
* automatically stop obviously diverging training
* generate a complete final report regardless of why training stops
* preserve resume functionality

---

# Phase 1 — Redesign the Training State

## Objective

Every epoch should produce a complete training snapshot.

Instead of only thinking about "saving model weights", every epoch should become a fully reproducible experiment state.

Each epoch should contain:

* model weights
* optimizer state
* scheduler state
* scaler state (AMP)
* current epoch
* best epoch
* best metric
* learning rate
* training time
* all epoch metrics
* stopping reason (if applicable)

Checkpoint metadata should include:

```
epoch
train_loss
val_loss
train_accuracy
val_accuracy
precision
recall
f1
balanced_accuracy
macro_f1
learning_rate
epoch_time
total_runtime
gradient_norm
dataset_checksum
seed
random_states
```

---

# Phase 2 — Save Checkpoint Every Epoch

Current code already supports checkpoints.

This needs to become mandatory.

Every epoch should generate

```
checkpoints/

epoch_0001.pt
epoch_0002.pt
epoch_0003.pt
...

latest.pt
best.pt
```

No epoch should ever be skipped.

Even failed epochs should be saved.

---

# Phase 3 — Epoch Report Generation

Every checkpoint should also generate

```
reports/

epoch_0001.json
epoch_0002.json
epoch_0003.json
...
```

Each report should contain

Dataset information

Model information

Optimizer information

Learning rate

Training metrics

Validation metrics

Timing

Memory usage

Stopping status

Warnings

Example

```
Epoch 7

Loss
-----
train_loss
val_loss

Accuracy
---------
train_accuracy
val_accuracy

Classification
--------------
precision
recall
macro_f1
weighted_f1
balanced_accuracy

Timing
------
epoch_time
samples_per_second

System
------
gpu_memory
cpu_memory
learning_rate

Status
------
checkpoint_saved=True
best_checkpoint=False
```

---

# Phase 4 — Final Report

When training finishes,

generate

```
experiment_report.json
experiment_report.pdf
history.csv
history.json
```

This must happen regardless of whether training ends because of

* normal completion

* early stopping

* NaN

* exploding loss

* divergence

* manual stopping condition

No situation should terminate training without a final report.

---

# Phase 5 — Fix Broken Metrics

This is the highest priority.

Current symptoms:

```
val_loss = NaN

val_acc = 0

precision = 0

recall = 0
```

These indicate metric computation is failing rather than the model necessarily failing.

The metric pipeline should be audited.

---

## 5.1 Verify validation loop

Ensure validation uses

```
model.eval()

torch.no_grad()
```

Ensure:

```
loss += batch_loss

correct += predictions

total += labels
```

instead of overwriting values.

---

## 5.2 Verify averaging

Never do

```
loss = loss / len(loader)
```

if

```
len(loader)==0
```

Use

```
max(1,len(loader))
```

---

## 5.3 Detect NaN immediately

After every batch

validate

```
loss

logits

predictions

gradients
```

Example

```
torch.isnan(loss)

torch.isinf(loss)
```

If detected

record

```
status="NaN detected"
```

generate report

save checkpoint

stop training

---

## 5.4 Metric accumulator redesign

Never compute metrics incrementally using partially reset variables.

Instead accumulate

```
predictions

labels

probabilities
```

for the entire epoch.

Then compute

```
accuracy

precision

recall

f1

balanced_accuracy

macro_f1

weighted_f1
```

once.

---

## 5.5 Validation integrity checks

After validation

verify

```
len(predictions)==len(labels)

len(probabilities)==len(labels)
```

If not

raise explicit error.

Never silently continue.

---

## 5.6 Metric sanity validation

Before writing metrics

validate

```
0 <= accuracy <= 1

0 <= precision <= 1

0 <= recall <= 1

0 <= macro_f1 <= 1

loss >= 0
```

Reject

```
NaN

Inf

negative loss
```

---

# Phase 6 — Divergence Detection

Training should stop automatically when it is clearly failing.

---

## Rule 1

Stop if

```
val_loss increases

AND

val_accuracy decreases
```

for

```
2 consecutive epochs
```

Reason

```
Training divergence
```

---

## Rule 2

Stop if

```
train_loss increases

AND

train_accuracy decreases
```

for

```
2 epochs
```

---

## Rule 3

Stop if

```
loss becomes NaN
```

Immediate stop.

---

## Rule 4

Stop if

```
loss becomes Inf
```

Immediate stop.

---

## Rule 5

Stop if

```
gradient norm

>

threshold
```

Example

```
1000
```

---

## Rule 6

Stop if

```
validation accuracy

is identical

for 2 consecutive epochs
```

Example

```
0.8462

0.8462
```

Training stops.

Reason

```
Validation accuracy stalled
```

---

## Rule 7

Stop if

```
validation loss

is identical

for 2 epochs
```

Reason

```
Validation loss stalled
```

---

## Rule 8

Stop if

```
training accuracy

is identical

for 2 epochs
```

---

## Rule 9

Stop if

```
training loss

is identical

for 2 epochs
```

---

# Phase 7 — Stopping Reason Enumeration

Every run should end with an explicit reason.

Example

```
TRAINING_COMPLETED

EARLY_STOPPING

VAL_DIVERGENCE

TRAIN_DIVERGENCE

LOSS_NAN

LOSS_INF

GRADIENT_EXPLOSION

VAL_ACCURACY_STALLED

VAL_LOSS_STALLED

TRAIN_ACCURACY_STALLED

TRAIN_LOSS_STALLED

USER_INTERRUPTED
```

The final report should always contain

```
stop_reason
```

---

# Phase 8 — History Tracking

Every epoch append one row to

```
history.csv
```

Include

```
epoch

train_loss

val_loss

train_accuracy

val_accuracy

precision

recall

f1

balanced_accuracy

macro_f1

learning_rate

epoch_time

checkpoint_path

best_checkpoint

stop_reason
```

Never overwrite history.

---

# Phase 9 — Best Checkpoint Tracking

Continue updating

```
best.pt
```

only when

```
monitor_metric
```

improves.

Always update

```
latest.pt
```

every epoch.

Epoch checkpoints remain immutable.

---

# Phase 10 — Resume Compatibility

When resuming

restore

```
optimizer

scheduler

AMP scaler

best metric

epoch history

learning rate

random states

early stopping counters

divergence counters

stall counters
```

Training should continue exactly where it stopped.

---

# Phase 11 — Logging Improvements

Console output should clearly indicate:

```
Epoch 12/100

Train
------
Loss
Accuracy
Precision
Recall
Macro F1

Validation
----------
Loss
Accuracy
Precision
Recall
Macro F1

Timing
------
Epoch Time
ETA

Checkpoint
----------
Saved:
epoch_0012.pt

Status
------
Best Model: Yes
Stop Check: Passed
```

If a stop condition triggers, print:

```
====================================
TRAINING STOPPED
====================================

Reason:
Validation accuracy stalled

Epoch:
23

Final checkpoint:
epoch_0023.pt

Final report written.
```

---

# Phase 12 — Robustness Testing

Before considering the implementation complete, validate the updated pipeline with the following scenarios:

1. **Normal convergence:** Verify all epoch checkpoints, reports, and the final report are created, and metrics evolve normally.
2. **NaN loss injection:** Confirm training stops immediately, saves the current checkpoint, records `LOSS_NAN`, and generates the final report.
3. **Validation divergence:** Simulate increasing validation loss with decreasing validation accuracy for two consecutive epochs and verify `VAL_DIVERGENCE` is triggered.
4. **Metric stall:** Force identical validation accuracy (and separately identical validation loss) for two consecutive epochs and verify the corresponding stall reason is recorded.
5. **Resume after interruption:** Interrupt training after several epochs, resume from `latest.pt`, and confirm optimizer state, scheduler, AMP scaler, counters, history, and metrics continue correctly.
6. **Metric integrity:** Verify that no reported metric contains `NaN`, `Inf`, or invalid values without immediately triggering an explicit error or stop condition.
7. **Report completeness:** Confirm that `history.csv`, per-epoch JSON reports, the final JSON report, plots, and the PDF (when enabled) are generated for every termination path.

## Expected End State

After these changes, the training system should guarantee:

* Every epoch produces a checkpoint, a structured metrics report, and a history entry.
* Metrics are validated before being recorded, eliminating silent `0` or `NaN` failures.
* Diverging or stalled training is detected automatically with explicit, documented stop reasons.
* A comprehensive final experiment report is always generated, regardless of whether training completes normally or stops early due to instability.
