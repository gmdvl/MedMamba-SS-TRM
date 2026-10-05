Yes. This new error changes the priority of the plan: **the immediate problem is now confirmed to be dataset-file integrity/loading, not `/dev/shm`**.

The important sequence is:

```text
SHM free = 15.5 GB / 16.3 GB
num_workers = 0
        ↓
integrity check initially detects truncated/incomplete array
        ↓
warning says "continuing"
        ↓
dataset discovery/load proceeds
        ↓
Phase-4 validator incorrectly reports all files passed
        ↓
NpyDataset(storage_mode=ram)
        ↓
np.load()
        ↓
ValueError: Failed to read all data
```

The most serious defect is therefore **not just the corrupted/incomplete `.npy` file**. The new implementation has a **validation-control-flow bug**: it detected the bad file and then continued anyway, and a later validation path incorrectly declared the files valid.

I would update the plan accordingly.

I wasn't able to create the downloadable Markdown file in the previous turn because the file-generation tool was unavailable. Below is the **updated plan content**, including the new Phase 0 priority and the changes that should supersede the corresponding parts of the previous plan.

# GMedMamba v13 — Updated Training Stability, Dataset Integrity, DataLoader SIGBUS, and Numerical Stability Remediation Plan

## 1. New Critical Finding

The latest v13 run provides direct evidence of a dataset integrity failure:

```text
Expected:
(743200, 11, 11, 3)
= 269,781,600 elements

Actually readable:
52,887,520 elements
```

Therefore:

```text
52,887,520 / 269,781,600 ≈ 19.60%
```

Only approximately **19.6% of the expected array elements are readable**.

The expected array is therefore severely incomplete/truncated.

The failure occurs while loading:

```text
storage_mode=ram
```

through:

```text
np.load(path)
```

and is independent of DataLoader multiprocessing.

The current run also reports:

```text
SHM free = 15507.06 MB
SHM total = 16321.62 MB
num_workers = 0
```

Therefore the new `ValueError` is **not a `/dev/shm` exhaustion problem**.

---

# 2. Critical Control-Flow Bug

The most important defect revealed by this run is:

```text
WARNING: dataset integrity check could not run (... file seems not fully written?); continuing.
```

This behavior is unacceptable for a training dataset.

A dataset integrity check that discovers:

```text
Expected 269781600 elements
Could only read 52887520
```

must **not** produce:

```text
continuing
```

It must produce:

```text
DATASET_INVALID
```

and terminate before training begins.

The current behavior allowed the program to continue until `np.load()` failed later.

---

# 3. Second Critical Bug — Conflicting Validation Results

The same run subsequently reports:

```text
[dataset-integrity] all .npy files passed Phase-4 validation.
```

This directly conflicts with the earlier failure:

```text
Failed to read all data for array.
Expected ...
could only read ...
```

The implementation must therefore be changed so that **there is exactly one authoritative dataset-validation result**.

No later validation phase may overwrite:

```text
INVALID
```

with:

```text
PASS
```

unless the file has actually been revalidated successfully.

---

# 4. New Required Validation State Machine

Replace the current ambiguous validation behavior with explicit states:

```text
NOT_CHECKED
    |
    v
CHECKING
    |
    +---- success ----> VALID
    |
    +---- failure ----> INVALID
```

Never allow:

```text
INVALID → VALID
```

without an explicit successful revalidation of the exact file.

Also distinguish:

```text
VALID
INVALID
UNREADABLE
TRUNCATED
MISMATCHED
NOT_CHECKED
```

---

# 5. Phase 0 — Dataset Integrity Becomes the First Gate

Before:

* dataset discovery completion;
* class-imbalance analysis;
* DataLoader creation;
* GPU model creation;
* training;
* RAM loading;
* mmap loading;

the dataset must pass validation.

Required sequence:

```text
START
  ↓
SYSTEM PREFLIGHT
  ↓
DISCOVER DATASET
  ↓
VALIDATE EVERY DATASET FILE
  ↓
VALID?
  ├── NO → ABORT
  └── YES
        ↓
CLASS DISTRIBUTION
        ↓
DATALOADER TEST
        ↓
NUMERICAL SMOKE TEST
        ↓
TRAINING
```

The current sequence is incorrect because it permits:

```text
validation failure
    ↓
continue
    ↓
later loading failure
```

---

# 6. Phase 1 — Validate File Header and Shape Without Reading the Entire Array First

For every `.npy` file, inspect:

```text
dtype
shape
fortran_order
header
file size
data offset
```

For:

```text
X_train.npy
```

the validator should determine:

```text
shape = (743200, 11, 11, 3)
dtype = ...
```

and calculate the expected payload size.

The expected number of elements is:

```text
743200 × 11 × 11 × 3
= 269,781,600
```

The validator must compare the expected payload size against the actual file size.

This should detect a truncated file **without attempting to load the complete array**.

---

# 7. Phase 2 — Detect Truncation Deterministically

For each `.npy`:

```text
actual_file_size
header_size
expected_data_bytes
```

must be compared.

Conceptually:

```text
expected_file_size =
    header_offset + expected_data_bytes
```

If:

```text
actual_file_size < expected_file_size
```

the result is:

```text
TRUNCATED
```

and training must stop.

Do not attempt:

```python
np.load(path)
```

after this failure.

---

# 8. Phase 3 — Verify the Specific Failing File

The current traceback identifies:

```text
NpyDataset(x_tr, y_tr)
```

and the expected array:

```text
(743200, 11, 11, 3)
```

The implementation must print the exact filename whenever validation fails.

For example:

```text
DATASET INVALID
role=X_train
path=...
shape=(743200,11,11,3)
expected_elements=269781600
readable_elements=52887520
status=TRUNCATED
```

Do not make the user infer which file failed from a generic exception.

---

# 9. Phase 4 — Validate X/Y Pair Consistency

For each split:

```text
X_train ↔ y_train
X_val   ↔ y_val
```

verify:

```text
X.shape[0] == y.shape[0]
```

The reported dataset currently says:

```text
743200 Train
176000 Val
Classes: 6
```

but those counts must be independently derived from the validated files.

Do not trust metadata or previously generated reports.

---

# 10. Phase 5 — Validate Class Labels

After the files pass structural validation:

Check:

```text
minimum label
maximum label
unique labels
number of classes
NaN labels
Inf labels
```

For six classes:

```text
labels must be valid for the configured six-class problem
```

Also ensure that the class-imbalance report is generated **only after dataset validity is established**.

The current:

```text
worst max/min class ratio = 17.9x
```

should remain a warning/analysis item, but it must not be generated from a dataset that has failed integrity validation.

---

# 11. Phase 6 — Do Not Perform Full `np.load()` as the Primary Integrity Test

The current implementation appears to have a problematic pattern:

```text
integrity check
    ↓
attempt complete read
    ↓
failure
    ↓
catch exception
    ↓
continue
```

This must be replaced.

For large datasets, use:

```text
header validation
+
file-size validation
+
random-access validation
```

before permitting training.

Full loading may be performed later when:

```text
storage_mode=ram
```

but a failure must be fatal.

---

# 12. Phase 7 — RAM Mode Must Fail Fast

Current v13 behavior:

```text
dataset_storage=ram
    ↓
np.load()
    ↓
ValueError
```

is acceptable as the final safety mechanism, but the error should be caught and converted into a useful diagnostic.

Instead of exposing only:

```text
ValueError: Failed to read all data...
```

report:

```text
DATASET_LOAD_ERROR

role: X_train
storage_mode: ram
path: ...
expected_shape: (743200,11,11,3)
expected_elements: 269781600
readable_elements: 52887520
status: TRUNCATED

Training aborted.
```

---

# 13. Phase 8 — `auto` Storage Mode Must Never Override Validation

The storage policy:

```text
auto
```

must operate only after integrity validation.

Correct:

```text
validate
   ↓
VALID
   ↓
select RAM/mmap
```

Incorrect:

```text
select RAM/mmap
   ↓
attempt loading
   ↓
discover corruption
```

---

# 14. Phase 9 — mmap Mode Must Be Validated Separately

For:

```text
storage_mode=mmap
```

perform:

1. header validation;
2. expected-size validation;
3. mmap creation;
4. first element read;
5. middle element read;
6. final element read;
7. random reads.

If any access fails:

```text
DATASET_MMAP_ERROR
```

and terminate.

---

# 15. Phase 10 — RAM Mode Must Be Tested Separately

Run:

```text
dataset_storage=ram
```

only after structural validation.

Because the current dataset is approximately:

```text
743200 × 11 × 11 × 3
```

elements, the validator must calculate the actual memory requirement from the dtype.

Do not assume that 32 GB system RAM automatically makes RAM mode safe.

Report:

```text
dataset_bytes
available_RAM
estimated_total_training_RAM
estimated_validation_RAM
```

before loading.

---

# 16. Phase 11 — Fix the "Could Not Run; Continuing" Policy

The following behavior must be removed:

```text
WARNING: dataset integrity check could not run (...); continuing.
```

Replace with:

```text
ERROR: dataset integrity validation failed.
Training cannot continue.
```

There should be only one exception:

```text
validation explicitly disabled
```

Even then, it should be a clearly recorded unsafe mode:

```text
--skip_dataset_validation
```

and should never be the default.

---

# 17. Phase 12 — Add `--skip_dataset_validation` Only for Debugging

If required for development, add:

```text
--skip_dataset_validation
```

but:

* default = false;
* emit a prominent warning;
* record it in config;
* mark experiment as non-qualified;
* never permit it in thesis production runs.

Example:

```text
WARNING:
DATASET VALIDATION DISABLED.
THIS RUN CANNOT BE MARKED SCIENTIFICALLY QUALIFIED.
```

---

# 18. Phase 13 — Dataset Generation Must Be Investigated

A file this incomplete should not simply be deleted and regenerated without understanding why it was created incorrectly.

The expected:

```text
269,781,600 elements
```

versus readable:

```text
52,887,520 elements
```

suggests the file may have been:

* interrupted during creation;
* copied incompletely;
* truncated by storage limits;
* generated with an incorrect save procedure;
* overwritten concurrently;
* produced by a failed preprocessing process.

The plan must therefore include an audit of the dataset-generation pipeline.

---

# 19. Phase 14 — Atomic Dataset Generation

All dataset creation scripts should write to a temporary file first:

```text
X_train.npy.tmp
```

and only after successful completion:

```text
X_train.npy
```

should be created through an atomic rename.

Conceptually:

```text
generate
   ↓
temporary file
   ↓
flush
   ↓
fsync
   ↓
validate
   ↓
atomic rename
```

This prevents interrupted preprocessing from leaving a file that looks like a valid `.npy` filename but is incomplete.

---

# 20. Phase 15 — Dataset Generation Completion Marker

Add a manifest:

```text
dataset_manifest.json
```

containing:

```json
{
  "file": "X_train.npy",
  "shape": [743200, 11, 11, 3],
  "dtype": "...",
  "size_bytes": ...,
  "sha256": "...",
  "complete": true
}
```

The training pipeline should compare the actual dataset against this manifest.

---

# 21. Phase 16 — Dataset Hash Validation

For production experiments, optionally calculate:

```text
SHA-256
```

for every dataset file.

Record:

```text
filename
size
hash
shape
dtype
```

This prevents an experiment from silently using a partially replaced dataset.

For extremely large files, provide a configurable fast validation mode, but full hashes should be available for final thesis experiments.

---

# 22. Phase 17 — Fix Validation Result Propagation

The validator should return a structured object, not merely log a warning.

For example:

```text
DatasetValidationResult
    status
    path
    role
    shape
    dtype
    expected_bytes
    actual_bytes
    readable
    errors
```

The caller must inspect:

```text
result.status
```

and abort if it is not:

```text
VALID
```

Do not infer success from:

```text
validator returned without raising
```

---

# 23. Phase 18 — Prevent Duplicate/Inconsistent Validators

There should be one canonical function:

```text
validate_dataset_file(...)
```

and one canonical dataset-level function:

```text
validate_dataset(...)
```

Do not maintain separate implementations that can disagree:

```text
Phase-4 validator says VALID
```

while:

```text
loader validator says INVALID
```

All code paths must use the same authoritative result.

---

# 24. Phase 19 — DataLoader SIGBUS Investigation Remains Required

The current run does **not** show a `/dev/shm` shortage:

```text
SHM free = 15.5 GB
SHM total = 16.3 GB
num_workers = 0
```

However, the previously observed errors remain valid concerns:

```text
DataLoader worker ... killed by signal: Bus error
```

and:

```text
Unexpected bus error encountered in worker.
```

Therefore retain the DataLoader remediation from the original plan:

```text
safe default:
workers=0
prefetch disabled
persistent_workers=false
pin_memory=false
```

Only re-enable workers after dataset integrity is proven.

---

# 25. Phase 20 — Correct Diagnostic Ordering

The logs should now follow:

```text
[system-memory:startup]

[dataset-discovery]

[dataset-integrity:start]

[dataset-integrity:X_train]
[dataset-integrity:y_train]
[dataset-integrity:X_val]
[dataset-integrity:y_val]

[dataset-integrity:PASS]

[class-imbalance]

[dataloader-test]

[numerical-smoke-test]

[training]
```

If any dataset file fails:

```text
[dataset-integrity:FAIL]

TRAINING ABORTED
```

Nothing after dataset validation should execute.

---

# 26. Phase 21 — Numerical Stability Work Remains Unchanged

The new dataset error does not invalidate the original numerical-stability plan.

The persistent:

```text
Non-finite gradient norm
```

still requires:

* parameter-level gradient checks;
* loss-component checks;
* activation checks;
* SAM stabilization;
* AMP control;
* gradient clipping;
* skip thresholds;
* epoch validity;
* failure termination.

However, these should now be tested **only after the dataset pipeline passes**.

---

# 27. Phase 22 — SAM Numerical Stability

Continue to stabilize the SAM loss.

Requirements:

* FP32 computation;
* safe normalization;
* zero-vector protection;
* safe cosine calculation;
* safe inverse cosine;
* finite forward result;
* finite backward result.

Test:

```text
zero vector
identical vector
near-identical vector
orthogonal vector
large values
small values
mixed precision
```

---

# 28. Phase 23 — Numerical Smoke Test

Once the dataset passes:

```text
1 batch
→ forward
→ all loss components
→ backward
→ gradient check
→ optimizer step
→ parameter check
```

Require all values to be finite.

---

# 29. Phase 24 — DataLoader Qualification

After dataset integrity passes:

### Test 1

```text
workers=0
```

### Test 2

```text
workers=1
prefetch=1
```

### Test 3

```text
workers=2
prefetch=1
```

No worker SIGBUS is permitted.

---

# 30. Phase 25 — 10-Batch Qualification

Require:

```text
10 valid updates
0 DataLoader failures
0 SIGBUS
0 non-finite losses
0 non-finite gradients
0 parameter corruption
```

---

# 31. Phase 26 — 100-Batch Qualification

Require the same conditions over 100 batches.

Additionally record:

```text
RAM
/dev/shm
GPU VRAM
batch time
DataLoader time
```

---

# 32. Phase 27 — Three-Epoch Qualification

Run at least three epochs.

Only mark:

```text
STABLE
```

when:

* dataset remains readable;
* no DataLoader worker failure occurs;
* no SIGBUS occurs;
* no persistent numerical instability occurs;
* all epochs contain valid optimizer updates.

---

# 33. Phase 28 — Failure Classification

Add:

```text
DATASET_TRUNCATED
DATASET_HEADER_INVALID
DATASET_SIZE_MISMATCH
DATASET_READ_ERROR
DATASET_MMAP_ERROR

DATALOADER_SHM
DATALOADER_WORKER_CRASH

CUDA_OOM
SYSTEM_MEMORY_PRESSURE

NONFINITE_FORWARD
NONFINITE_LOSS
NONFINITE_GRADIENT
PARAMETER_CORRUPTION
AMP_NUMERICAL_INSTABILITY

UNKNOWN_SIGBUS
```

The current failure should specifically be classified as:

```text
DATASET_TRUNCATED
```

assuming the file-size/header inspection confirms the discrepancy.

---

# 34. Phase 29 — Required Failure Artifact

For dataset failure:

```text
dataset_failure/
    failure.json
    dataset_manifest.json
    file_metadata.json
    system_memory.json
    validation_report.json
```

`failure.json` should contain:

```text
role
path
expected_shape
actual_shape
dtype
expected_elements
readable_elements
expected_bytes
actual_bytes
status
exception
timestamp
```

---

# 35. Phase 30 — Dataset Regeneration Procedure

After identifying the bad file:

1. Stop all training processes.
2. Confirm no preprocessing process is still writing the dataset.
3. Inspect file size.
4. Compare with expected size.
5. Inspect dataset-generation logs.
6. Delete/rename the invalid file.
7. Regenerate it.
8. Validate the generated file.
9. Generate/update manifest.
10. Perform random-access testing.
11. Only then launch training.

Do not repeatedly launch training against the same invalid dataset.

---

# 36. Phase 31 — Dataset Writer Requirements

Any preprocessing script that creates `.npy` files must:

* write to a temporary path;
* complete the entire write;
* flush;
* close the file;
* validate shape and byte size;
* optionally calculate checksum;
* atomically rename;
* write manifest;
* mark completion.

A process interruption must not leave an apparently valid final filename.

---

# 37. Phase 32 — Memory Budget for RAM Mode

Before:

```text
np.load(path)
```

calculate:

```text
dataset_size
+
process_memory
+
model_memory
+
DataLoader memory
```

and compare against:

```text
available RAM
```

The reported system currently has:

```text
~15.05 GB available RAM
~32.64 GB total RAM
```

Therefore RAM mode must not assume the full dataset can safely coexist with the training process.

---

# 38. Phase 33 — Storage Strategy

Recommended hierarchy:

```text
Production:
    validate → auto storage

Debugging:
    validate → RAM or mmap explicitly

Corruption investigation:
    validate → mmap/RAM comparison

Final thesis:
    validated immutable dataset + manifest/hash
```

---

# 39. Phase 34 — Revised Safe Mode

`--safe_mode` must now guarantee:

```text
dataset validation = REQUIRED
num_workers = 0
prefetch = disabled
persistent_workers = false
pin_memory = false
AMP = off
gradient checkpointing = on
numerical diagnostics = on
strict dataset loading = on
```

Safe mode must **never** continue after dataset-integrity failure.

---

# 40. Phase 35 — Revised Execution Pipeline

The final architecture should be:

```text
                    GMedMamba v13
                         |
                         v
                  SYSTEM PREFLIGHT
                         |
                         v
                  DATASET DISCOVERY
                         |
                         v
                STRICT DATA VALIDATION
                         |
              +----------+----------+
              |                     |
            FAIL                   PASS
              |                     |
              v                     v
        ABORT IMMEDIATELY      CLASS VALIDATION
                                    |
                                    v
                              LOADER TEST
                                    |
                                    v
                           NUMERICAL SMOKE TEST
                                    |
                                    v
                                TRAINING
                                    |
                 +------------------+------------------+
                 |                  |                  |
              Data error        Numerical error     success
                 |                  |                  |
                 v                  v                  v
               abort             abort             validate
                                                        |
                                                        v
                                                   checkpoint
```

---

# 41. Revised Implementation Priority

The priority is now:

| Priority | Issue                                      | Action                               |
| -------- | ------------------------------------------ | ------------------------------------ |
| **P0**   | Truncated/incomplete `.npy`                | Fix dataset and strict validation    |
| **P0**   | Validator says "continuing" after failure  | Remove continuation behavior         |
| **P0**   | Conflicting validation result              | Create one authoritative validator   |
| **P0**   | Dataset generation can leave partial files | Atomic writes + manifest             |
| **P0**   | RAM loader exposes corruption only late    | Validate before loading              |
| **P1**   | mmap safety                                | Header/size/random-access validation |
| **P1**   | DataLoader `/dev/shm` SIGBUS               | workers=0 safe baseline              |
| **P1**   | Persistent non-finite gradients            | Numerical controller                 |
| **P1**   | SAM instability                            | FP32/safe formulation                |
| **P1**   | AMP instability                            | Explicit AMP modes                   |
| **P2**   | Worker performance                         | Re-enable gradually                  |
| **P2**   | RAM/mmap optimization                      | Optimize only after correctness      |

---

# 42. New Acceptance Criteria

## Dataset integrity

* [ ] Every `.npy` file has a valid header.
* [ ] Every `.npy` file has the expected shape.
* [ ] Every `.npy` file has the expected dtype.
* [ ] Every `.npy` file has the expected byte size.
* [ ] No truncated files are accepted.
* [ ] X/y sample counts match.
* [ ] Random-access reads succeed.
* [ ] RAM loading succeeds when selected.
* [ ] mmap loading succeeds when selected.
* [ ] Invalid files cause immediate termination.
* [ ] Validation cannot report PASS after an earlier failure.
* [ ] Dataset manifest is generated.
* [ ] Dataset generation is atomic.

## DataLoader

* [ ] `workers=0` succeeds.
* [ ] `workers=1` succeeds.
* [ ] `workers=2` succeeds where system resources permit.
* [ ] No worker SIGBUS.
* [ ] No unexplained `/dev/shm` exhaustion.
* [ ] Prefetch defaults to 1 when workers are enabled.
* [ ] Persistent workers are disabled by default.

## Numerical stability

* [ ] One-batch smoke test passes.
* [ ] 10-batch test passes.
* [ ] 100-batch test passes.
* [ ] No non-finite losses.
* [ ] No persistent non-finite gradients.
* [ ] No NaN/Inf parameters.
* [ ] SAM forward/backward tests pass.
* [ ] FP32 is stable.
* [ ] BF16 is tested separately.

## Scientific validity

* [ ] Invalid datasets cannot produce experiment results.
* [ ] Invalid epochs cannot become best epochs.
* [ ] Excessive skipped batches invalidate the epoch.
* [ ] Persistent instability terminates the run.
* [ ] Failure causes are recorded.
* [ ] Dataset identity is recorded.
* [ ] Dataset configuration is reproducible.

---

# 43. Immediate Diagnosis of the Current Run

The current run should be classified as:

```text
FAILURE CLASS:
DATASET_TRUNCATED / DATASET_INTEGRITY_FAILURE
```

not:

```text
DATALOADER_SHM
```

and not:

```text
CUDA_OOM
```

because:

```text
num_workers = 0
```

and:

```text
/dev/shm free ≈ 15.5 GB
```

while the actual fatal exception is:

```text
ValueError:
Failed to read all data for array.
Expected 269781600 elements,
could only read 52887520.
```

The **first implementation fix should therefore be the validator/control-flow bug**, followed by investigation/regeneration of the incomplete `.npy` file.

The most important new rule to add to the original plan is:

> **A dataset-integrity failure is fatal. The training process must never continue after a file is detected as truncated, unreadable, structurally invalid, or inconsistent with its expected shape/size.**
