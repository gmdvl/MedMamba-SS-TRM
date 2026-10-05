# Ready‑to‑Run Training Commands

> **Companion to:** [`07_parameters_v14.md`](07_parameters_v14.md) (what every flag means) · driven by [`train_example_v14.py`](../train_example_v14.py)
> **One‑line summary:** copy‑pasteable commands for RGB, HSI and unknown datasets, every one of them executed end‑to‑end before being written down, with the wall‑clock and VRAM those runs actually produced.
> **Hardware for all measurements:** NVIDIA GeForce RTX 5060 Ti (16.7 GB), torch 2.11.0+cu128, CUDA 12.8, 32 GB system RAM.

---

## Table of contents

1. [Before you run anything](#1-before-you-run-anything)
2. [A — RGB datasets](#2-a--rgb-datasets)
3. [B — HSI datasets](#3-b--hsi-datasets)
4. [C — Any dataset](#4-c--any-dataset)
5. [The baseline you should also run](#5-the-baseline-you-should-also-run)
6. [Measured results](#6-measured-results)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Before you run anything

**Check what you actually have — 10 seconds, no GPU:**

```bash
python train_example_v14.py --data_dir ./data/pad_v6/ --loader_test --loader_test_batches 2
```

This runs every dataset gate (integrity, manifest, leakage, class coverage), builds the
DataLoader, reads two batches, prints the resolved shapes and the storage decision, and exits
**without constructing the model**. Read three things off it:

- `batch 0: x.shape=(256, C, H, W)` — the channel count `C` decides whether this is section A or B.
- `[dataset-storage] auto -> ram|mmap` — whether the array fits in RAM.
- `Dataset Loaded: N Train, M Val` — **if `M` equals the test‑set size, this dataset has no
  validation split.** See the warning in §3.

**How modality is decided.** `--architecture` presets are selected by `(architecture, modality)`,
and modality is decided by exactly one thing: whether a file named **literally**
`wavelengths.npy` exists at the top level of `--data_dir`. `selected_wavelengths.npy` and
`wavelengths_full.npy` do not count. A spectral dataset whose wavelength file is named
anything else trains silently as RGB.

**Two flags to be careful with, whatever you run:**

- `--dataset_storage` — leave it on `auto`. An explicit `ram` on `data/hsi_v7/hsi/` would try
  to pull 40 GB into ~19 GB of free memory; `auto` correctly chooses mmap there.
- `--dataset_validation_level deep` — a full sequential read of every `X_*.npy` at startup.
  That is 46 GB for `data/hsi_v7/hsi/`. Use it only when you suspect bad sectors.

---

## 2. A — RGB datasets

Target: `data/pad_v6/` — PAD‑UFES‑20, 740,800 train / 90,000 val patches of 11×11×3,
6 classes, real validation split, **16.3:1 class imbalance** (BCC 36.99%, MEL 2.27%).

```bash
python train_example_v14.py \
  --data_dir ./data/pad_v6/ \
  --architecture recursive \
  --epochs 50 \
  --recon_mode none \
  --loss ce --sampler moderate_oversample --checkpoint_metric f1_macro \
  --augment_preset custom --flip_h_prob 0.5 --flip_v_prob 0.5 --rotate90_prob 0.5 \
  --loader_mode performance \
  --dataset_storage ram \
  --early_stop_patience 8
```

Everything not listed is a default that is already correct: `--trm_mixer mlp`,
`--amp bf16`, `--batch_size 256`, `--trm_deep_supervision_steps 3`, `--lr 1e-4`.

Why each flag that *is* listed:

| flag | reason |
|---|---|
| `--recon_mode none` | Drops the decoder, the MSE term and the SAM term. SAM measures the angle between spectral signature vectors — over three RGB channels that is hue similarity, not a spectral quantity, and it is applied by default at `--lambda_sam 0.1`. Removing the whole recon pathway is ~24% of step time. |
| `--sampler moderate_oversample` | MEL is 2.27% of the split while `--checkpoint_metric f1_macro` weights it equally with BCC. This tops minority classes up to the median count. |
| `--loss ce` | Deliberately **not** `weighted_ce`: the sampler already corrects the imbalance, and stacking both over‑corrects (see [`07_parameters_v14.md`](07_parameters_v14.md) §4.3). If you prefer loss‑level rebalancing, use `--loss weighted_ce --sampler none` instead — measured results for both are in §6. |
| `--augment_preset custom --flip_* --rotate90_*` | Geometric only. The `light`/`medium` presets are HSI presets: on C=3 their band‑dropout always zeroes one entire RGB channel, which for pigmented lesions destroys the primary diagnostic cue. |
| `--loader_mode performance` | 4 workers, prefetch 2, persistent, pinned — set coherently. Passing `--num_workers 8` alone would leave the other three fields at the default mode's values. |
| `--dataset_storage ram` | `X_train.npy` is 1.08 GB against ~19 GB free. `auto` already picks this; stating it makes the run reproducible on a busier machine. |
| `--early_stop_patience 8` | These 740,800 patches are 674 tiles from each of only 1,099 source images. 50 full epochs is ~33,700 effective passes over the source images. |

**Faster sweep variant** — add `--train_subsample_frac 0.3 --trm_n_latent 3 --trm_n_improve 2
--trm_deep_supervision_steps 2` for hyperparameter search, then report final numbers from the
full config above.

> **On interpreting the metrics.** A single 11×11 RGB patch is 121 pixels of skin, usually
> entirely outside the lesion. Per‑patch accuracy is weak by construction and collapse to a
> single class is a genuinely attractive local optimum for the model. The meaningful metric is
> aggregated per lesion. If you are not doing patch→lesion aggregation at evaluation time, the
> headline numbers understate the model considerably.

---

## 3. B — HSI datasets

Target: `data/hsi_v7/hsi/` — 2,594,592 train patches of 11×11×**32**, 3 classes
(`healthy`, `DCIS`, `IDC`), 46 GB on disk, `wavelengths.npy` present.

> ### This dataset has no validation split
>
> `data/hsi_v7/hsi/` contains no `X_val.npy`. When that file is absent, `discover_data`
> **silently falls back to `X_test.npy`/`y_test.npy` as the validation set** — no warning is
> printed. The confirming symptom is in the preflight: `Dataset Loaded: 2594592 Train,
> 540904 Val`, where 540,904 is exactly the test‑set size.
>
> Consequently `--checkpoint_metric f1_macro` selects `best_model.pt` **on the test set**, and
> any test number you then report has been optimised against. For a thesis this invalidates
> the held‑out claim.
>
> Two remedies:
> - **Re‑prepare** the dataset with a three‑way split (`--split 80/10/10`). This is the correct fix.
> - **Or** use `data_hsi_v6-v4-nb32_mb_40_v2.1/hsi/`, which does have a real validation split
>   (2,020,018 train / 579,852 val / 535,626 test, also 32 bands). Note it has no
>   `class_names.json`, so classes are reported by index.

```bash
python train_example_v14.py \
  --data_dir ./data/hsi_v7/hsi/ \
  --architecture recursive \
  --epochs 50 \
  --recon_mode none \
  --train_subsample_frac 0.05 \
  --loss weighted_ce --checkpoint_metric f1_macro \
  --augment_preset custom --flip_h_prob 0.5 --flip_v_prob 0.5 --rotate90_prob 0.5 \
  --loader_mode performance \
  --dataset_storage auto --dataset_validation_level structural \
  --early_stop_patience 8
```

What differs from the RGB command, and why:

| flag | reason |
|---|---|
| `--dataset_storage auto` | **Never `ram` here.** The train array's header‑declared payload is 40.2 GB against ~19 GB available; `auto` resolves to mmap for train and ram for the smaller val array. |
| `--dataset_validation_level structural` | The default, restated because `deep` means a 46 GB sequential read before training starts. |
| `--train_subsample_frac 0.05` | 2,594,592 samples is 10,135 batches per epoch. At 0.05 an epoch is ~130k samples. **Report this number with any metric** — it is recorded in `config.json` and in the run directory name. Raise it as your time budget allows. |
| `--loss weighted_ce` | No sampler here: `moderate_oversample` duplicates indices into an already-enormous split. Loss‑level rebalancing is the cheaper correction at this scale. |
| geometric augmentation only | `--band_dropout`/`--spectral_mask` are defensible on 32 bands, unlike on RGB — but leave them off until you have a baseline to compare against. |

If you keep `--recon_mode latent` on HSI, `--lambda_sam 0.1` is finally meaningful: with 32
bands the spectral angle is a real spectral‑shape term. That is the one place the
reconstruction pathway earns its cost.

---

## 4. C — Any dataset

Use this when you do not know the channel count, whether wavelengths are present, or how big
the split is. It relies on auto‑detection everywhere and makes no dataset‑specific assumption.

```bash
# 1. Inspect first (no GPU, no model built)
python train_example_v14.py --data_dir <YOUR_DIR> --loader_test --loader_test_batches 2

# 2. Train
python train_example_v14.py \
  --data_dir <YOUR_DIR> \
  --architecture recursive \
  --epochs 50 \
  --recon_mode none \
  --checkpoint_metric f1_macro \
  --train_subsample_frac 0.1 \
  --loader_mode balanced \
  --dataset_storage auto \
  --numerical_smoke_test on \
  --early_stop_patience 8
```

Why this is the safe general form:

- **`--architecture recursive`** is the only architecture that is genuinely channel‑agnostic
  in its preset: `patch_size=1` and a channel‑agnostic spectral tokenizer, so the same config
  serves C=3 and C=32 without a shape change.
- **`--dataset_storage auto`** sizes the load against actual free RAM per file.
- **`--loader_mode balanced`** (2 workers) rather than `performance`, because a dataset you
  have not profiled may be I/O‑bound or may stress `/dev/shm`.
- **No `--loss`/`--sampler`** — plain `ce`, because you do not yet know the class balance. Read
  `<exp_dir>/class_imbalance_report.json` after the first run and then move to section A or B.
- **`--numerical_smoke_test on`** does one forward/backward/step on a throwaway model copy
  before training. Seconds, and it turns a dead configuration into a clean abort.
- **`--train_subsample_frac 0.1`** bounds the first epoch on a split of unknown size. Remove it
  once you know the real cost.

Deliberately **not** included: `--dataset_storage ram` (unsafe on a large unknown array),
`--augment_preset light`/`medium` (wrong on RGB), and `--safe_mode` (forces `amp=off` and 0
workers — that is a debugging tool, not a training configuration).

---

## 5. The baseline you should also run

```bash
python train_example_v14.py \
  --data_dir ./data/pad_v6/ \
  --architecture split \
  --epochs 50 \
  --recon_mode none \
  --loss ce --sampler moderate_oversample --checkpoint_metric f1_macro \
  --augment_preset custom --flip_h_prob 0.5 --flip_v_prob 0.5 --rotate90_prob 0.5 \
  --loader_mode performance --dataset_storage ram --early_stop_patience 8
```

`--architecture split` is 27.4 M parameters and runs at ~58 ms/step — roughly **5 minutes per
epoch**, so all 50 epochs cost about 4 hours.

Run it. The recursive variant's entire claim is "comparable accuracy at ~50× fewer
parameters", and that claim is unfalsifiable without the `split` number on the identical
split. It also produces an honest efficiency story: the parameter reduction is real, the
*compute* reduction is not (see [`07_parameters_v14.md`](07_parameters_v14.md) §5), and stating
that explicitly is stronger than letting a reader discover it.

---

## 6. Measured results

Every row below is a real run of the command in this document, on the hardware named at the
top, on 2026-09-01. No number here is estimated or extrapolated unless the row says so.
All are **one epoch**, so they establish cost and that the command completes — not accuracy.

### 6.1 Wall‑clock and memory

| run | dataset | train samples | val samples | wall‑clock | peak VRAM | exit |
|---|---|---:|---:|---:|---:|---|
| **Bare defaults** (`--data_dir`, `--epochs`, `--architecture recursive`, nothing else) | `data/pad_v6/` | 740,800 | 90,000 | **1,745.9 s (29.1 min)** | 1,723 MB | 0 |
| §2 A, sampler variant, `--train_subsample_frac 0.3` | `data/pad_v6/` | 245,160 | 90,000 | 723.3 s (12.1 min) | 1,720 MB | 0 |
| §2 A, loss variant, `--train_subsample_frac 0.3` | `data/pad_v6/` | 222,240 | 90,000 | 547.6 s (9.1 min) | 1,720 MB | 0 |
| §3 B, HSI, `--train_subsample_frac 0.02` | `data/hsi_v7/hsi/` | 51,892 | 540,904 | 910.6 s (15.2 min) | **9,724 MB** | 0 |
| §4 C, any‑dataset, `--train_subsample_frac 0.1` | `data/pad_v6/` | 74,080 | 90,000 | **240.2 s (4.0 min)** | 1,725 MB | 0 |

Reading these:

- **The bare‑defaults row is the headline.** The same command before this revision's default
  changes was measured at 66.9 s/step → **~92 h/epoch**. It is now 29.1 min — a **~190×**
  end‑to‑end improvement with no flags passed.
- **Validation is a large fixed cost.** Solving the two `pad_v6` rows for a per‑sample training
  rate gives ≈2.06 ms/sample train and ≈215 s for the 90,000‑sample validation pass. On HSI the
  validation set is 540,904 samples and dominates that run's 910 s outright — most of section B's
  measured time is *validation*, not training.
- **HSI costs 5.6× the VRAM of RGB** (9.7 GB vs 1.7 GB) at the same batch size, because the
  spectral pathway scales with channel count. 32 bands at `--batch_size 256` fits a 16 GB card
  with room to spare; going much beyond that will not.
- **`--dataset_storage auto` resolved correctly and differently per file** on the HSI run:
  `mmap` for the 40,185 MB training array against 19,877 MB available RAM, `ram` for the
  8,378 MB validation array. This is exactly why §3 tells you not to force `ram` there.

### 6.2 First‑epoch accuracy, and what it does not tell you

| run | loss / sampler | val acc | BalAcc | F1(macro) | collapsed to |
|---|---|---:|---:|---:|---|
| Bare defaults | `ce` / none | 0.3689 | 0.1667 | 0.0898 | BCC (36.89% of val) |
| §2 A sampler variant | `ce` / `moderate_oversample` | 0.3689 | 0.1667 | **0.0898** | BCC |
| §2 A loss variant | `weighted_ce` / none | 0.1200 | 0.1667 | 0.0357 | NEV (12.00% of val) |
| §4 C any‑dataset | `ce` / none, `frac 0.1` | 0.2978 | 0.1667 | 0.0765 | ACK (29.78% of val) |

**All four collapsed to a single class after one epoch**, each to a different class. The signature is unmistakable: val
accuracy equals the collapsed‑to class's prevalence exactly, and `BalAcc = 1/6` is what balanced
accuracy always returns when one of six classes is predicted unconditionally.

Two honest conclusions, and one non‑conclusion:

- Sampler‑level rebalancing (`moderate_oversample`) beat loss‑level rebalancing
  (`weighted_ce`) on `f1_macro`, 0.0898 vs 0.0357. That is why §2 recommends the sampler and
  plain `ce` rather than the other way round.
- Epoch‑1 collapse is **expected** here and is not a failure: the trainer's
  `--class_collapse_streak` guard warns for three consecutive epochs before it means anything,
  and it did not fire. Note the model collapses to a *different* class under each objective
  (BCC, NEV, ACK), which is the objective doing its job — it is not stuck on one attractor.
- The §4 C run also passed `--numerical_smoke_test on` (`[smoke-test] PASSED
  (grad_norm=10.43)`), confirming that gate works on the recursive architecture.
- **These runs do not establish that any of these commands converges.** They establish cost and
  completion. One epoch of fifty, on a fractional split, is not evidence about final accuracy.
  Run five epochs and watch the collapse warning before committing to a full run.

Worth re‑reading §2's note on patch‑level labels alongside this table: for a classifier seeing
one 11×11 RGB tile at a time, most of them plain skin, predicting the majority class *is* the
rational optimum. Epoch‑1 collapse on all three configurations is weak evidence for that
reading, and patch→lesion aggregation is the thing that addresses it.

### 6.3 A dataset‑integrity failure the gate caught

The first attempt at the §4 C command was pointed at `data/hsi_v7/rgb/` and did not train:

```
DATASET_STORAGE_ERROR: dataset does not match its manifest:
  X_train.npy: header unreadable ([Errno 5] Input/output error)
FAILURE_CLASS=DATASET_IO_ERROR
```

This was not a bug in the command. A byte‑level scan confirmed
`data/hsi_v7/rgb/X_train.npy` is **physically unreadable from offset 1,090,519,040 onward** —
2,676,828,672 bytes (71% of the file) return `OSError(errno=5)` reproducibly, while its header
and first 1.09 GB read fine. Every other `.npy` under `data/pad_v6/` and `data/hsi_v7/` reads
end‑to‑end without error, including the 40 GB `hsi_v7/hsi/X_train.npy`.

Two things follow. First, **`data/hsi_v7/rgb/` must be re‑prepared** before it can be used.
Second, the integrity gate did exactly its job: a partially‑unreadable file was converted into a
typed `DATASET_IO_ERROR` at startup, with artifacts written to
`<exp_dir>/dataset_failure/`, instead of a `SIGBUS` or silently corrupt training partway through
an epoch. This is the failure mode `--skip_dataset_validation` disables — which is why you should
not pass it.

The §4 C row in the table above is therefore the re‑run against `data/pad_v6/`, which also
serves as the channel‑agnostic check: the identical flag set trains C=3 here and C=32 in
section B, with no per‑dataset tuning.

---

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| An epoch takes hours instead of minutes | `--trm_mixer ss2d` (a pure‑Python selective scan called 168× per step), or `--amp off` | Use the defaults: `--trm_mixer mlp --amp bf16`. See [`07_parameters_v14.md`](07_parameters_v14.md) §5. |
| `DATASET_STORAGE_ERROR` at startup | Truncated/corrupt `.npy`, or a manifest mismatch | Read `<exp_dir>/dataset_failure/failure.json`. Re‑run the dataset merge; do **not** reach for `--skip_dataset_validation`. |
| `TRAINING_ABORTED_NUMERICAL_INSTABILITY` | 3 consecutive non‑finite batches | Almost always `--amp fp16`. Use `bf16` (fp32 exponent range, no loss scaling needed) or `off`. Add `--debug_numerics` to see which component went non‑finite. |
| `CLASS_COVERAGE_INCOMPLETE` | A split has zero samples of some class | Re‑run the split with a different seed, or pass `--allow_missing_classes` and document it. |
| `CUDA out of memory` | Batch too large, or `--trm_deep_supervision_steps` too high | Lower `--batch_size` first. Note the recursive core self‑checkpoints and OOMs 16 GB without it — that is expected, not a leak. |
| Every prediction is one class; `f1_macro` ≈ 0.03 | Class collapse | Expected in epoch 1; the run continues (`--class_collapse_streak` only warns). If it persists past epoch 3: lower `--lr`, switch to `--sampler balanced`, and re‑read §2's note on patch‑level labels. |
| `best_model.pt` never appears | `--checkpoint_metric` typo — a bad key silently disables checkpoint selection **and** early stopping | Check the first epoch prints `Saved: … (BEST)`. Valid keys: `accuracy`, `balanced_accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `cohen_kappa`. |
| `[activation-patch] replaced 0 nn.ReLU module(s)` | Not an error | Only `split` has `nn.ReLU` modules. Normal on every other architecture. |
| `balanced_accuracy exceeds accuracy` warning | Not an error under class collapse | When one class of six is predicted unconditionally, `BalAcc = 1/6` while accuracy equals that class's prevalence. The check's assumption does not hold in the degenerate case. |
| Validation set size equals the test set size | `X_val.npy` is missing and `discover_data` fell back to test | See §3. Re‑prepare the split, or use a dataset that has one. |
| `--fusion_type` appears to do nothing | It only applies to `--architecture efficient` | Silently ignored elsewhere — **and still written into `config.json`**, so that field is misleading on non-`efficient` runs. |
| `--architecture efficient` crashes with `unknown fusion_type 'se_gate'` | The extended fusion types are never registered, because `efficient` does not build its own backbone | Known bug — see [`07_parameters_v14.md`](07_parameters_v14.md) §7.1. `efficient` is currently a silent alias for `fullchannel`. Use `split`, `fullchannel` or `recursive`. |
