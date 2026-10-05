# GMedMamba v12/v13 Stability Remediation — Implementation Summary

> **Superseded.** Moved from the repository root by v17 S12. This describes the v12/v13
> stability remediation as a set of files to drop into a repo; they have long since
> landed. Nothing in the repository references it. The current plans are
> `plan/GMedMamba v15/v16/v17*`.

All files below drop into your existing repo (`training/*.py` files go into
your `training/` package; `train_example_v6.py`, `train_example_v13.py` go
at repo root next to `train_example_v12.py`). Everything is additive/new
except two files, both backward-compatible:

- **`training/gan.py`** — `SAMLoss` rewritten (Phase 15), same public
  API (`SAMLoss()(x_true, x_pred) -> scalar radians`).
- **`train_example_v6.py`** — `NpyDataset.__init__` gained an optional
  `storage_mode="mmap"` kwarg (default reproduces old behavior exactly).
- **`training/trainerg_v5.py`** — one small **pre-existing bug fix**,
  unrelated to this plan (see note in the file): `fit()`'s per-epoch
  `metrics` dict was missing `val_mse_loss`/`val_sam_loss`/`val_gan_loss`,
  which `TrainerG_v4._log_epoch`/`_generate_final_report` (inherited by
  every `TrainerG_v5`+ subclass, including the new `TrainerG_v9`) hard-index
  — so **any** completed epoch under `TrainerG_v5`–`v8` currently raises
  `KeyError: 'val_mse_loss'`. I fixed this because it blocked verifying the
  plan's own Stage 6–9 acceptance criteria (an epoch must complete and be
  reported). **Note:** I also hit a second, similar pre-existing missing-key
  bug one line later (`m['parameter_norm']`/`m['update_norm']`, from a
  diagnostics feature `fit()` never populates) that I did **not** fix — it's
  outside this plan's scope and deserves its own look at the whole
  `_log_epoch`/diagnostics contract rather than another one-line patch.

## New files

| File | Plan phase(s) | What it does |
|---|---|---|
| `training/system_memory.py` | 3 | RAM / `/dev/shm` / GPU snapshot + logging, before any DataLoader worker exists. |
| `training/npy_integrity.py` | 4, 5, 6 | Validates every `.npy` (header, size, dtype/shape, first/mid/last/random sample reads, X/y length match) **before** any mmap/DataLoader; raises `DatasetStorageError` rather than proceeding to a guaranteed SIGBUS. Also implements the `mmap`/`ram`/`auto` storage modes. |
| `training/dataloader_config.py` | 1, 2, 7(helper), 8, 9 | `safe`/`balanced`/`performance` loader-mode presets, independent train/val worker policies, deterministic per-worker seeding, CPU-thread capping. |
| `training/numerical_stability.py` | 11–23, 36, 37 | `NumericalStabilityController`: skip-ratio classification (healthy→fatal), parameter-level gradient inspection, per-loss-component finiteness checks, persistent-failure → run-abort policy, failure-artifact writer, one-batch smoke-test helper. |
| `training/gradient_health.py` (extended) | 12 | Existing tracker gained optional parameter-level fields (first bad parameter, nan/inf element counts, max/min grad norm) — fully backward compatible. |
| `training/trainerg_v9.py` | 10, 12, 14, 18, 21–23, 37 | New trainer (`TrainerG_v9`, extends `TrainerG_v8`) wiring all of the above into the actual training loop: explicit `amp_mode`, per-component loss checks, parameter-level gradient inspection, post-step parameter health check, abort-and-write-failure-artifacts on persistent instability. |
| `train_example_v13.py` | 0–9, 10, 29, 30, 38, 42 | New entry point (mirrors `train_example_v12.py`, adds `--loader_mode/--num_workers/.../--dataset_storage/--amp/--safe_mode/--loader_test/--debug_numerics/--max_gradient_*`), dataset validation before DataLoader construction, preflight printout, `--safe_mode` one-flag conservative preset, full config.json reproducibility metadata. |
| `tests/test_sam_loss.py` | 33 (SAM) | Zero/identical/near-identical/orthogonal/large/small-magnitude/mixed-precision + 200-iteration fuzz check — all pass. |
| `tests/test_dataset_integrity.py` | 33 (dataset) | Good file, missing file, empty file, truncated file, NaN detection, length-mismatch abort, mmap-vs-ram equivalence — all pass. |
| `tests/test_numerical_stability.py` | 33/34 (numerical) | Skip-ratio classification, parameter-level gradient inspection, persistent-failure abort, epoch-invalidation, one-batch smoke test — all pass. |

**29/29 new tests pass** (`pytest tests/test_sam_loss.py
tests/test_dataset_integrity.py tests/test_numerical_stability.py`).

I also ran two end-to-end sanity checks against synthetic data:
1. `--loader_test --num_workers 0` — dataset validation, safe DataLoader,
   preflight, and a real batch read all completed cleanly.
2. A hand-built `TrainerG_v9` with a model forced to emit NaN logits —
   confirmed it aborts after `max_consecutive_bad_batches`, sets
   `stop_reason = "TRAINING_ABORTED_NUMERICAL_INSTABILITY"`, and writes
   `numerical_failure/failure.json`.

## What's *not* included (full plan is ~40 phases; this covers all P0/P1
items plus the highest-value P2 items, not literally every checkbox)

- **`training/dataset_storage.py`/`training/data.py` deep integration**:
  `NpyDataset` now *accepts* `storage_mode`; `train_example_v13.py` passes
  `--dataset_storage`. I did not additionally rewrite `training/data.py`
  into "the shared canonical dataset implementation with `train_example_v6`
  as a compat wrapper" (Phase 39's *preferred* option) — that's a larger,
  purely-organizational refactor with no behavioral difference from what's
  delivered.
- **Phase 24 (automatic fallback sequence / `stability_report.json`
  auto-escalation through AMP/SAM/GAN combinations)** and **Phase 27
  (automatic batch-size back-off on CUDA OOM)** are not implemented — both
  are self-contained additional features you can layer on top of
  `TrainerG_v9`/`train_example_v13.py` if you want them; they weren't
  needed to satisfy the core SIGBUS/numerical-stability objectives.
- **Phase 28's full failure-mode taxonomy** (`CUDA_OOM` /
  `DATALOADER_SHM` / `DATASET_MMAP_SIGBUS` / ... as a single enum) isn't
  centralized — `DatasetStorageError` vs `NUMERICAL_INSTABILITY` abort
  reasons vs a plain CUDA `OutOfMemoryError` are each distinguishable today
  (different exception types / stop_reason strings), just not unified under
  one label.
- I did not write `tests/test_dataloader_stability.py` or
  `tests/test_gradient_health.py` as separate files — DataLoader-worker
  crash testing needs a real multi-process run (not meaningfully fakeable
  in a unit test), and `gradient_health.py`'s new fields are exercised
  indirectly by `test_numerical_stability.py`.

## Recommended first command (Phase 42)

```bash
python train_example_v13.py \
    --data_dir <DATASET> \
    --epochs 3 \
    --batch_size <SMALL_SAFE_BATCH> \
    --safe_mode \
    --recon_mode latent \
    --lambda_gan 0
```

Then work through Stage 10's reintroduction order (`--loader_mode balanced`,
`--amp bf16`, more workers, `--pin_memory`, larger batch) one change at a
time, per the plan.
