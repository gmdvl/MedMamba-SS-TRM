# v15 — Ready-to-Run Commands

> **Companion to:** [`09_v15_remediation.md`](09_v15_remediation.md) (what changed and why) · driven by [`train_example_v15.py`](../archive/train_example_v15.py) and [`prepare_histologyhsi_bc_v7.py`](../prepare_histologyhsi_bc_v7.py)
> **Supersedes for v15:** [`08_commands.md`](08_commands.md), which drives `train_example_v14.py` and predates every flag below. v14's commands still work and still reproduce v14's runs — they also still train a constant function.
> **Hardware for all measurements:** NVIDIA GeForce RTX 5060 Ti (16.7 GB), torch 2.11.0+cu128, CUDA 12.8, 32 GB RAM.

---

## v16 — the current recommended commands

> Driven by [`train_example_v16.py`](../train_example_v16.py) /
> [`training/trainerg_v12.py`](../training/trainerg_v12.py) (`TrainerG_v12`), per
> `plan/GMedMamba v16 — Acquisition, Reconstruction & Reporting Remediation.plan.md`.
> Sections 0–6 below describe `train_example_v15.py` and stay accurate for it —
> `train_example_v15.py`/`TrainerG_v11` are FROZEN so those runs stay reproducible. This section
> is what to actually run now. Everything below was exercised against the real datasets and a
> real GPU (this hardware) while building v16, not just written from the plan.
>
> **What changed vs v15, in one line each:** `per_patch_zscore` normalization removes a
> per-patient exposure artifact that `global_zscore` was silently passing into the model (gate G8
> now catches this automatically before training starts); `--recon_mode latent` on
> `--architecture recursive` actually trains now (it used to be silently force-disabled); a
> step-axis LR schedule with warmup replaces the epoch-axis one; per-patient/per-capture metrics
> and image-level (patch-MIL) RGB metrics are new report sections.

### Before training — one-time, read-only, seconds

```bash
# Patient/capture (HSI) and clinical-image (RGB) group sidecars. Needed for gate G8's
# per-patient breakdown, --min_patient_recall, and RGB's --eval_group_aggregation. Safe to
# re-run any time; it only reads _progress.json and the unified y_*.npy files.
python scripts/derive_patch_groups.py --out_dir data/hsi_v7-80_10_10_importance
python scripts/derive_patch_groups.py --out_dir data/pad_v6

# Gates, and the shallow-baseline ceiling (gate G4) under the v16 normalization. `--out` just
# writes a report JSON next to the dataset for YOU to read and compare the real run against —
# it is not an input to anything below; train_example_v16.py never reads it.
pytest test_representation_sensitivity.py test_recon_gradient.py test_split_drift_gate.py -q
python scripts/shallow_baseline_v16.py \
    --data_dir data/hsi_v7-80_10_10_importance/hsi --normalization per_patch_zscore \
    --out data/hsi_v7-80_10_10_importance/hsi/shallow_baseline.json
python scripts/shallow_baseline_v16.py \
    --data_dir data/pad_v6 --normalization per_sample_minmax \
    --out data/pad_v6/shallow_baseline.json
```

The `--data_dir` above is exactly the `--data_dir` the HSI/RGB training commands below use —
`prepare`'s `--out_dir` (+ `/hsi` or `/rgb`) becomes everyone else's `--data_dir`; that's the one
path that carries forward through prepare → shallow_baseline → train. `shallow_baseline`'s `--out`
is a dead end on purpose: a record for you, never an input `train_example_v16.py` looks for.

### HSI — recommended command

```bash
python train_example_v16.py \
    --data_dir data/hsi_v7-80_10_10_importance/hsi \
    --architecture recursive \
    --normalization per_patch_zscore \
    --recon_mode latent --recon_out_activation auto \
    --lambda_mse 1.0 --lambda_sam 0.1 \
    --loss weighted_ce --class_weight_power 1.0 \
    --sampler moderate_oversample \
    --trm_drop_path 0.1 --trm_dropout 0.1 --classifier_dropout 0.2 \
    --augment_preset light \
    --epochs 30 \
    --train_subsample_frac 0.1 --train_subsample_mode per_epoch \
    --val_subsample_frac 0.2 \
    --early_stop_patience 8 --min_patient_recall 0.5 \
    --amp bf16 --compile on \
    --loader_mode performance --dataset_storage mmap \
    --eval_artifact_stride 5 --keep_last_n 2 \
    --seed 42
```

~958 steps/epoch (~22 min at the ~1.4 s/step measured for this model), ~11 h total, and because
`--train_subsample_mode per_epoch` redraws the 10 % subset every epoch, 30 epochs see ~300 % of
the full training split rather than the same tenth thirty times.

**Before trusting `--class_weight_power`:** `1.0` above is the honest, uncontaminated default.
The v15/v16 audit found `0.75` worked better, but that number was derived by re-deciding **test**-
split probabilities (finding E-3) — a test-set-informed choice that must not be reused as-is. Run
`scripts/tune_class_weight_power.py` against this run's saved **validation** probabilities and use
whatever it finds instead.

**If you have hours to spend on prep first:** `prepare_histologyhsi_bc_v8.py` (Stage 1.5) writes a
per-capture-gain-corrected, properly-scaled reflectance dataset — it closes the acquisition fault
at the *source* rather than relying on `per_patch_zscore` to absorb it at train time, and roughly
halves storage via `float16`. It is not required (the command above is already correct and
gate-checked without it) but is the better long-run target once it has finished:

```bash
python prepare_histologyhsi_bc.py \
    --root /path/to/HistologyHSI-BC-Recurrence --out_dir ./data/hsi_v8_gaincorrected \
    --band_selection importance --num_bands 32 --band_min_gap 8 --band_min_coverage 0.30 \
    --capture_gain median_ratio --hsi_value_scale auto \
    --split 80_10_10 --seed 42 --num_workers 8 --verify_level deep
```

Fast dev/smoke variant (~10 min; confirms gates G7/G8 and the pipeline before committing to the
real run — see the plan's own end-to-end smoke test for the full checklist):

```bash
python train_example_v16.py \
    --data_dir data/hsi_v7-80_10_10_importance/hsi \
    --architecture recursive --normalization per_patch_zscore \
    --recon_mode latent --epochs 2 \
    --train_subsample_frac 0.01 --val_subsample_frac 0.02 \
    --debug_numerics --eval_test off
```

### RGB (PAD-UFES-20) — recommended command

```bash
python train_example_v16.py \
    --data_dir data/pad_v6 \
    --architecture recursive \
    --normalization per_sample_minmax \
    --recon_mode none \
    --loss weighted_ce --class_weight_power 1.0 \
    --sampler moderate_oversample \
    --trm_drop_path 0.1 --trm_dropout 0.1 --classifier_dropout 0.2 \
    --augment_preset custom --flip_h_prob 0.5 --flip_v_prob 0.5 --rotate90_prob 0.5 \
    --epochs 40 \
    --early_stop_patience 8 \
    --eval_group_aggregation mean_prob \
    --amp bf16 --compile on \
    --loader_mode performance --dataset_storage ram \
    --eval_artifact_stride 5 --keep_last_n 2 \
    --seed 42
```

`--eval_group_aggregation mean_prob` is the load-bearing flag here (finding C-1): an 11×11 crop of
a clinical photograph is a few percent of the lesion and cannot see enough to separate melanoma
from nevus on its own, so **patch-level accuracy is not the number that matters**. Read
`test_report.json` → `image_level` instead, and report it explicitly as a patch-MIL result, not a
like-for-like comparison with the HSI arm. `--recon_mode none` is deliberate here too — the
reconstruction pathway is built around a wide spectral sequence and is close to degenerate at
RGB's 3 channels (finding C-2); spend the compute on the classification objective instead.

### Parameter glossary

**Representation / data — decides whether the model can see its input correctly**

| Flag | Recommended value | What it does |
|---|---|---|
| `--architecture` | `recursive` | The TRM-style recursive backbone — the only architecture this plan's fixes (reconstruction, deep supervision, EMA) apply to. |
| `--normalization` | `per_patch_zscore` (HSI) / `per_sample_minmax` (RGB) | How each patch is rescaled before the model sees it. `per_patch_zscore` re-centers/re-scales every patch to *its own* mean/std, so a mis-exposed acquisition (patient 68) no longer poisons validation the way a fixed, train-fit `global_zscore` affine does. RGB has no comparable acquisition-exposure problem, so its default (`per_sample_minmax`) is unchanged. |
| `--recon_mode` | `latent` (HSI) / `none` (RGB) | Whether the model also reconstructs its input from the learned features, as an auxiliary loss. Now actually trains the encoder on `recursive` (v16 fixes R-1…R-7); left off for RGB (C-2 — the spectral pathway is close to degenerate at 3 channels). |
| `--recon_out_activation` | `auto` | The decoder's final nonlinearity. `auto` resolves to `linear` under any z-score normalization (the reconstruction target is unbounded and can be negative — a `sigmoid` output there floors the loss near 1.0 regardless of decoder quality) and `sigmoid` under `per_sample_minmax` (target genuinely in `[0,1]`). |
| `--lambda_mse`, `--lambda_sam` | `1.0`, `0.1` | Weight of the reconstruction MSE / spectral-angle terms in the total loss. Defaults; lower them if the auxiliary loss dominates the classification signal (watch `MSE_loss`/`SAM_loss` vs `classification_loss` in `loss_components.csv`). |

**Loss & regularization — accuracy and a healthy training signal under class imbalance**

| Flag | Recommended value | What it does |
|---|---|---|
| `--loss` | `weighted_ce` | Cross-entropy with per-class weights (both datasets are >13:1 imbalanced). |
| `--class_weight_power` | `1.0`, then re-tune | Exponent softening the inverse-frequency weights (`0.0` = plain CE, `1.0` = full inverse-frequency). `1.0` is the honest default; `scripts/tune_class_weight_power.py` on this run's **validation** probabilities will usually find a lower value (e.g. ~0.75) gives a better macro-F1 without the test-set contamination of just reusing that published number. |
| `--sampler` | `moderate_oversample` | Oversamples minority classes at the DataLoader level, on top of (not instead of) `--loss weighted_ce` — the sampler fixes what the model *sees*, the loss fixes what it *pays for*. Drop it if balanced accuracy overshoots while raw accuracy collapses. |
| `--trm_drop_path`, `--trm_dropout`, `--classifier_dropout` | `0.1`, `0.1`, `0.2` | Stochastic depth / dropout inside the recursive core and the classifier head. Not optional at these dataset sizes — without them the matched generalization gap (`generalization_gap_loss` in `history.csv`) grows every epoch while validation CE rises. This is the single biggest lever on "healthy gradient" / "minimum loss" over a full run. |
| `--augment_preset` | `light` (HSI) / `custom` + flip/rotate (RGB) | Train-time augmentation. HSI patches are histology texture at fixed orientation relative to the slide, so `light` (mild spectral noise, no geometric flips) is enough; RGB clinical photographs have no canonical orientation, so flips/90° rotation are safe and effective. |
| `--early_stop_patience` | `8` | Stops training once the checkpoint metric hasn't improved for 8 epochs — the cheapest way to get "minimum loss" without hand-tuning `--epochs`. |

**Schedule & speed — fast processing without sacrificing correctness**

| Flag | Recommended value | What it does |
|---|---|---|
| `--train_subsample_frac` / `--train_subsample_mode` | `0.1` / `per_epoch` (HSI only) | Trains on a fresh random 10 % of the training split each epoch, so the epoch loop stays fast (~22 min/epoch instead of ~3.7 h/epoch on the full 2.45 M patches) while still covering ~300 % of the data over 30 epochs. RGB's split is small enough (740,800 patches) to train on in full — no subsample flag needed there. |
| `--val_subsample_frac` | `0.2` (HSI only) | Validation on a fixed, patient-stratified 20 % of the split — v16 automatically stratifies BY PATIENT (not just by class) when the group sidecars from `derive_patch_groups.py` are present, so even a small validation subset keeps every patient represented. |
| `--scheduler_interval` | `step` (v16 default — no flag needed) | Steps the LR schedule once per optimizer step rather than once per "epoch". With `--train_subsample_frac` in play, an "epoch" is not a fixed amount of work, so a step-axis schedule (with linear warmup into cosine decay) is what actually tracks training progress. |
| `--amp bf16 --compile on` | | Mixed-precision + `torch.compile`. `torch.compile` is ~2× faster on the recursive architecture's hot loops; `bf16` is the safe AMP choice on this hardware (no `GradScaler` needed, unlike `fp16`). |
| `--loader_mode performance` | | 4 train DataLoader workers, persistent + prefetching. v16 fixes the bug where the *validation* loader inherited `persistent_workers=True` from this mode for no benefit (E-4 follow-on) — validation workers are non-persistent automatically now. |
| `--dataset_storage` | `mmap` (HSI) / `ram` (RGB) | Where the `.npy` arrays live during training. **Never `ram` for HSI** — the array (~38 GB, or ~19 GB once re-prepped with `--store_dtype float16`) does not fit in 32 GB of system RAM alongside everything else; RGB's array (~1.1 GB) fits comfortably and removes the loader from the critical path entirely. |
| `--batch_size` | `256` (default — no flag needed) | Measured optimal on this hardware: throughput is flat in batch size on HSI and *falls* on RGB above 256 — the model is compute-bound per sample, not launch-bound, so a bigger batch buys nothing. |

**Safety gates — catch a broken run in seconds, not GPU-hours (all defaults; listed for awareness)**

| Flag | Default | What it does |
|---|---|---|
| `--sensitivity_check on` | on | Gates G1/G2 (stem sensitivity, logit std) + G9 (disjoint-batch logit delta), with **margins** reported (v16 C-2) — not just pass/fail. A run that "passes" by less than 3× its threshold is worth a second look before spending GPU-hours on it. |
| G7 (reconstruction gradient) | always runs when `recon_mode=latent` on `recursive` | Backprops the reconstruction loss alone and asserts the recursive core actually receives gradient from it. This is the gate that would have caught the pre-v16 bug where reconstruction trained only its own decoder. |
| `--on_split_drift` | `abort` | Gate G8. Fails the run before it starts if any split's post-normalization statistics have drifted too far from train's — this is the check that catches patient 68 in seconds instead of after a full run. |
| `--min_patient_recall` | `0.5` (recommended; off by default) | Logs a `PATIENT_OUTLIER` warning in `history.json` when any single validation patient's recall drops below this while the macro metric is still rising — the shape of the exact failure the v15 audit missed. |
| `--max_gradient_norm` | `1.0` (default — no flag needed) | Gradient-norm clipping threshold. Part of "healthy gradient": batches whose gradient would exceed this are clipped, not skipped, so training stays stable without silently dropping data. |

**Reporting**

| Flag | Recommended value | What it does |
|---|---|---|
| `--eval_group_aggregation` | `mean_prob` (RGB only) | Aggregates per-patch posteriors to one prediction per clinical image (mean of the softmax probabilities, then argmax) and reports it under `test_report.json` → `image_level` (finding C-1). Read this number for RGB, not the raw patch-level one. |
| `--checkpoint_metric` | `f1_macro` (default — no flag needed) | Which validation metric selects the best checkpoint. Macro-averaged, so it already weights rare classes correctly under the 13–16:1 imbalance both datasets have. |
| `--eval_artifact_stride 5 --keep_last_n 2` | | Writes the per-epoch confusion matrix / classification report / prediction dump every 5th epoch (always the last, always the best) instead of every epoch, and keeps only the 2 most recent checkpoints — pure I/O speed, no effect on training. |

---

## 0. The two datasets, as they actually are

| | `data/hsi_v7-80_10_10/hsi` | `data/pad_v6` |
|---|---|---|
| train / val / test patches | 2,452,086 / 334,516 / 348,894 | 740,800 / 90,000 / 88,400 |
| patch | 11 × 11 × 32, float32 | 11 × 11 × 3, float32 |
| classes | 3 — healthy, DCIS, IDC | 6 — ACK, BCC, MEL, NEV, SCC, SEK |
| train class counts | 722,358 / 122,850 / 1,606,878 (**13.1:1**) | 234,800 / 274,000 / 16,800 / 79,200 / 62,400 / 73,600 (**16.3:1**) |
| shallow ceiling (gate G4) | **0.6078** balanced / **0.5612** F1-macro | **0.2899** balanced / **0.2217** F1-macro |
| v15 normalization (auto) | `global_zscore` | `per_sample_minmax` |
| measured throughput | ~1.4–1.5 s/step at bs 256, ~1.0 GB peak | ~0.65–1.2 s/step at bs 256, ~1.9 GB peak |

**`--batch_size 256` is optimal and larger is not.** Measured patches/second is *flat* in batch
size on HSI (100 / 98 / 94 / 95 at bs 256 / 512 / 1024 / 2048) and *falls* on RGB (215 → 178 →
177, OOM at 2048). The model is compute-bound per sample, not launch-bound, so the ~14 GB of
headroom that spectral checkpointing frees does not convert into throughput. Spend it on a wider
model or a second concurrent run, not on a bigger batch.

---

## 1. Always run these two first — seconds to minutes, no GPU

```bash
# Gates G1/G2. If this fails, no training run below is worth starting.
pytest test_representation_sensitivity.py -q

# The ceiling every result must be read against (gate G4). ~4 min.
python scripts/shallow_baseline.py --data_dir data/hsi_v7-80_10_10/hsi data/pad_v6
```

---

## 2. HSI — re-prep with `prepare_histologyhsi_bc_v7.py`

**Why re-prep at all.** `data/hsi_v7-80_10_10/selected_band_indices.npy` is exactly `[0..31]` —
the first 32 of 740 bands, **400.5–423.1 nm out of 400.5–938.2 nm, 4.2 % of the sensor**, missing
the entire haemoglobin absorption region (~540–580 nm) and all NIR scatter. `--band_min_coverage`
now **fails the prep** on that selection. The existing dataset is still perfectly usable — it is
the `band set` control arm of the R7 ablation — but it is not a spectral dataset in any meaningful
sense.

Both commands below take ~1 h and write ~45 GB (38 GB HSI + 7 GB RGB). `/data` has 579 GB free.

```bash
RAW=/home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/

# (a) RECOMMENDED — uniform: deterministic, spans the full sensor by construction,
#     needs no Pass-1 sample to decide anything. Coverage = 1.000, verified.
python prepare_histologyhsi_bc_v7.py \
    --root "$RAW" --out_dir ./data/hsi_v8_uniform \
    --band_selection uniform --num_bands 32 \
    --split 80_10_10 --seed 42 \
    --num_workers 8 --verify_level deep

# (b) the supervised alternative — keeps the variance+MI importance ranking but
#     forces the selection to spread. --band_min_gap 8 reaches coverage 1.000 on
#     this sensor's measured importance curve; --band_max_corr 0.95 (the default)
#     additionally drops near-duplicate neighbours using the Pass-1 pixel sample.
python prepare_histologyhsi_bc_v7.py \
    --root "$RAW" --out_dir ./data/hsi_v8_importance \
    --band_selection importance --num_bands 32 \
    --band_min_gap 8 --band_max_corr 0.95 --band_min_coverage 0.30 \
    --split 80_10_10 --seed 42 \
    --num_workers 8 --verify_level deep
```

Then **re-measure the ceiling on the new data before training anything** — that number is the
whole justification for the re-prep:

```bash
python scripts/shallow_baseline.py --data_dir data/hsi_v8_uniform/hsi --out data/hsi_v8_uniform/shallow_baseline.json
```

Useful variants:

* `--band_selection_sample_fraction 0.10` (default) controls how many cubes Pass 1 opens. Raise to
  `0.25` for a more stable importance estimate; it only affects (b).
* `--allow_narrow_bands` downgrades the coverage failure to a warning — for a *deliberate*
  narrow-band ablation, never to get past the guardrail.
* `--modality hsi` skips the RGB companion arrays and saves ~7 GB.
* `--store_dtype float16` halves the HSI array to ~19 GB. `auto` chose float32 for `hsi_v7`
  because the corpus's raw `hsi_value_max` (64,829.0) exceeds `_FLOAT16_SAFE_MAX` (60,000.0),
  not because of the calibration probe — `get_calibration_means` returning `(None, None)` means
  the cube IS already calibrated, and `hsi_all_calibrated` had the opposite sense (v16 audit,
  A-1/Stage 1.5). Fixing that flag alone changes nothing here; `prepare_histologyhsi_bc_v8.py`'s
  `--hsi_value_scale` is what brings the corpus into a range `float16` can represent, unlocking
  the ~19 GB write. Check `dataset_statistics.json` before forcing `float16` on an unscaled corpus.
* Interrupted? Re-run the identical command — the prep is resumable. If only the unify step
  failed, `--out_dir <same> --finalize_only --verify_level deep`.

**PAD-UFES-20 needs no re-prep.** `data/pad_v6` was built by `prepare_pad_ufes_20_v6.py`, which
v15 does not touch (there is no band selection in an RGB pipeline). For the record, it was:

```bash
python prepare_pad_ufes_20.py --root ../Dataset/PAD-UFES-20/ --out_dir ./data/pad_v6 \
    --split 80/10/10 --num_workers 8 --patch_size 11
```

---

## 3. Training — HSI

### 3a. Qualification (~12 min, run this first)

```bash
PYTHONPATH=.:archive python archive/train_example_v15.py \
    --data_dir data/hsi_v7-80_10_10/hsi \
    --architecture recursive \
    --epochs 3 \
    --train_subsample_frac 0.02 \
    --val_subsample_frac 0.05 \
    --loss weighted_ce \
    --loader_mode performance --dataset_storage mmap \
    --dataset_validation_level structural \
    --eval_artifact_stride 3 --keep_last_n 1
```

Everything else is a v15 default and is already right: `concat_mlp` tokenizer, `global_zscore`
normalization, wavelengths forwarded, `--recon_mode none`, halting off, weight-decay groups,
spectral checkpointing on (C = 32 ≥ 16), collapse abort, G5/G6 verification.

What you should see, and what it means:

```
[sensitivity] G1 stem=0.504 (>= 0.05)  G2 logit_std=2.6e-01 (>= 1e-02)   <- the fix, at init
[wavelengths] forwarding (32,) band centers ... into the model            <- inert before v15
```

Measured on this exact command's slightly smaller sibling (1.5 %, 432 steps, 11 min):
balanced accuracy **0.3853 → 0.3965**, F1-macro peaking at **0.3175** — against the pre-v15 run's
0.3333 / 0.0456 with one class predicted, at one seventh of the optimizer steps.
Held-out test: balanced accuracy **0.5728** against the probe's 0.6078.

`--dataset_validation_level structural` is the one deliberate downgrade here: v15 defaults HSI to
`deep`, which reads all 38 GB end-to-end (R6.5, because this exact file had an EIO region during
prep). Keep `deep` for any run whose numbers you intend to publish.

### 3b. The real run (~7 h, 30 epochs)

```bash
PYTHONPATH=.:archive python archive/train_example_v15.py \
    --data_dir data/hsi_v8_uniform/hsi \
    --architecture recursive \
    --epochs 30 \
    --train_subsample_frac 0.05 --train_subsample_mode per_epoch \
    --val_subsample_frac 0.1 \
    --loss weighted_ce --sampler moderate_oversample \
    --trm_drop_path 0.1 --trm_dropout 0.1 --classifier_dropout 0.2 \
    --early_stop_patience 8 \
    --loader_mode performance --dataset_storage mmap \
    --eval_artifact_stride 5 --keep_last_n 2
```

* `--train_subsample_frac 0.05 --train_subsample_mode per_epoch` = 479 steps/epoch (~11 min), and
  because the subset is **redrawn every epoch** (R6.1), 30 epochs see ~150 % of the split. Under
  v14 the same flag showed the model the same 122,604 patches thirty times.
* **The three regularization flags are not optional here.** The qualification run overfits hard —
  the matched generalization gap (R5.2) goes +0.35 → +1.24 → +1.89 over three epochs while train
  CE falls 0.85 → 0.48 and validation CE rises 1.20 → 2.36. Those flags are the R4.2 fix and did
  not exist before v15.
* `--sampler moderate_oversample` on top of `--loss weighted_ce` is deliberate for 13:1: the
  sampler fixes what the model *sees*, the loss fixes what it *pays for*. Drop the sampler if
  balanced accuracy overshoots while accuracy collapses.
* **Never `--dataset_storage ram`** on this dataset — the array is 38 GB against 32 GB of RAM.

---

## 4. Training — RGB (PAD-UFES-20)

### 4a. Qualification (~15 min)

```bash
PYTHONPATH=.:archive python archive/train_example_v15.py \
    --data_dir data/pad_v6 \
    --architecture recursive \
    --epochs 3 \
    --train_subsample_frac 0.25 \
    --loss weighted_ce \
    --loader_mode performance --dataset_storage ram \
    --eval_artifact_stride 3 --keep_last_n 1
```

`--dataset_storage ram` is safe here (the array is 1.1 GB) and removes the loader from the
critical path. The full validation split is only 90,000 patches, so no `--val_subsample_frac`.

### 4b. The real run (~20 h, 40 full epochs)

```bash
PYTHONPATH=.:archive python archive/train_example_v15.py \
    --data_dir data/pad_v6 \
    --architecture recursive \
    --epochs 40 \
    --loss weighted_ce --sampler moderate_oversample \
    --trm_drop_path 0.1 --trm_dropout 0.1 --classifier_dropout 0.2 \
    --early_stop_patience 8 \
    --augment_preset custom --flip_h_prob 0.5 --flip_v_prob 0.5 --rotate90_prob 0.5 \
    --loader_mode performance --dataset_storage ram \
    --eval_artifact_stride 5 --keep_last_n 2
```

* 2,894 steps/epoch × ~0.65 s ≈ 31 min/epoch. The v14 RGB run was 17.5 h for 33 epochs, so this
  is the same order — but every one of those hours now trains a model that can see its input.
* The augmentation flags are worth enabling **now specifically**: under v14 all four DataLoader
  workers drew the identical flip/rot90 sequence (`AugmentationConfig.build()` closed over one
  shared `random.Random`), so `--augment_preset custom` bought roughly a quarter of the diversity
  it claimed. R6.3 fixed that; the draws are now per (seed, epoch, index, worker).
* Class MEL has 16,800 train / 2,000 val patches — the 16.3:1 minority. Watch its per-class recall
  in `per_class_metrics.csv`, not just the macro average.

---

## 5. The ablation that answers the thesis question

How much of the failure was **scale** and how much was **rank**? Three runs, one variable, at a
fixed step budget. RGB because it is the cheaper of the two.

```bash
for F in add scaled concat_mlp; do
  PYTHONPATH=.:archive python archive/train_example_v15.py --data_dir data/pad_v6 --architecture recursive \
      --epochs 10 --train_subsample_frac 0.25 --loss weighted_ce \
      --spectral_token_fusion $F \
      --sensitivity_check off --on_class_collapse warn \
      --loader_mode performance --dataset_storage ram
done
```

`--sensitivity_check off --on_class_collapse warn` are **required for the `add` arm** — the whole
point of that arm is to reproduce the defect, and the v15 guardrails exist to refuse exactly that.
Run directories are distinguishable: `…_bf16_ADD-legacy_wce_minmax_sub25` vs `…_scaled_…` vs the
unmarked v15 default.

To isolate the mechanism further rather than just the mode, add
`--spectral_pe_gain 1.0 --spectral_value_init_std 0.02 --no_spectral_ctx_norm --classifier_init shared`
to the `add` arm — that is the fully pre-v15 model, reachable from v15's entry point.

Other single-variable axes, all one flag each:
`--normalization`, `--use_wavelengths` / `--no_use_wavelengths`, `--trm_mixer {mlp,attention}`,
`--no_trm_mixer_channel_mlp`, `--trm_halting --trm_halt_threshold 0.5`, `--loss` × `--sampler`,
and the band set (`hsi_v7-80_10_10` / `hsi_v8_uniform` / `hsi_v8_importance`).

**Report optimizer steps, not epochs.** `history.json` carries
`optimizer_steps_this_epoch` / `_total` for exactly this reason: the HSI evidence run's "6 epochs"
was 2,868 steps against the RGB run's 105,336, a 37× difference presented as comparable.

---

## 6. Reading the output

| file | what to check |
|---|---|
| `gates.json` | `G5.passed` must be `true`; `G6` must be `true` |
| `test_report.json` | the only number to quote — one pass, best checkpoint, held-out patients |
| `config.json` → `scientifically_qualified` | a dict now: `dataset_validated`, `seeded`, `deterministic`, `representation_gates_passed`, `test_evaluated` |
| `config.json` → `representation`, `normalization` | the settings that decide whether the model can see its input, and how the input was scaled — neither was recorded before v15 |
| `history.json` → `generalization_gap_loss_is_comparable` | `true` means the gap is last-segment CE on both sides and can be read |
| `collapse_failure/failure.json` | present only if the run aborted on class collapse — read it before changing anything else |
| `validation_report.json` | `impossible_f1` should be 0; `class_never_predicted` is the one that matters |

If a run aborts with `REPRESENTATION_COLLAPSE`, do **not** raise the learning rate or the epoch
count. Run `pytest test_representation_sensitivity.py -q` and check `--spectral_token_fusion` was
not left at `add`.
