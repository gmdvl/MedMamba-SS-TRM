# Additional Analyses These Two Datasets Support

> **2026-09-28:** the commands below were rewritten from `train_example_v16*.py` / `train_example_v18.py` / `scripts/v20_queue.sh` to `train.py --profile …` / `run_experiments.py`, with the same flags. The two resolve every argument identically (`tests/test_train_parity.py`; see [`17_train_and_sweeps.md`](17_train_and_sweeps.md)). The originals are in `archive/`. A pre-2026-09-28 `train_example_v16_optimal_recon.py` command is `--profile paper_recipe`. **2026-10-01:** the profiles were renamed for what they are for (`v18` → `paper_recipe`, `optimal` → `pad_ufes_best_norecon`, `original` → `medmamba_protocol_norecon`, `base` → `pipeline_defaults`); `_norecon` marks the two without reconstruction. The old names still work.


> Companion to [`12_commands_datasets.md`](12_commands_datasets.md).
>
> **Status: analyses 1–6 have been PROMOTED.** They are no longer suggestions — they are
> first-class sections of the commands file, with complete runnable commands, and
> `scripts/compare_runs.py` exists to tabulate them:
>
> | here | now lives in doc 12 as |
> |---|---|
> | 1 band-count sweep | [§G](12_commands_datasets.md#g--band-count-sweep) |
> | 2 `--rgb_source` ablation | [§H](12_commands_datasets.md#h--rgb-source-ablation) |
> | 3 reconstruction as a regulariser | [§I](12_commands_datasets.md#i--reconstruction-as-a-regulariser) |
> | 4 architecture ablation | [§J](12_commands_datasets.md#j--architecture-ablation) |
> | 5 PAD spatial-resolution sweep | [§K](12_commands_datasets.md#k--pad-spatial-resolution-sweep) |
> | 6 per-patient reporting | [§L](12_commands_datasets.md#l--per-patient-reporting) |
>
> **This file keeps the *reasoning*** — why each is worth GPU time, what it controls for, what
> could go wrong with it, and what it cannot tell you. Doc 12 has the commands; this has the
> argument. §7 and §8 below remain suggestions: §7 is blocked on data that is not in this
> download, and §8 needs code that does not exist yet.

---

## Summary

| # | analysis | why | cost | status |
|---|---|---|---|---|
| 1 | **Band-count sweep** (32 → 16 → 8 → 3) | turns a binary claim into a *curve* — the strongest result available from this data | 4 preps (~1.8 h each, ~43 GB) + 4 short runs | **promoted → doc 12 §G** |
| 2 | **`--rgb_source` ablation** | separates "spectral information" from "which camera" — closes a confound in the headline result | 2 preps + 2 runs | **promoted → doc 12 §H** |
| 3 | **Auxiliary reconstruction as a regulariser** | does forcing the model to reconstruct improve *classification*? never measured | 2 runs | **promoted → doc 12 §I** |
| 4 | **Architecture ablation** (recursive vs split/efficient) | is the TRM core earning its place? | 3 runs | **promoted → doc 12 §J** |
| 5 | **PAD spatial-resolution sweep** (`--target_token_grid`) | the one axis PAD has never explored; 91 % of the card is idle | 2 runs, ~10 h each | **promoted → doc 12 §K** |
| 6 | **Per-patient / per-capture reporting** | 45 patients — patch-level metrics overstate confidence | free, already computed | **promoted → doc 12 §L** |
| 7 | **Recurrence prediction** (`--label_source recurrence`) | a genuinely different clinical question | — | **blocked**, see §7 |
| 8 | **Cross-dataset transfer** | pretrain on histology, fine-tune on PAD | 2 runs | needs a small script, see §8 |

**§6 has already returned a result, without a single new run** — see doc 12 §L. On the
matched pair the HSI advantage at the patient level is **0.069**, against **0.146** at patch
level, and the per-patient standard deviation (0.13–0.17) is *larger than the gap*, over 5
validation patients. The direction is consistent across every RGB arm and the two matched
seeds agree to within 0.0024, so the finding stands — but the effect size does not survive
being quoted at patch level. Fix how the result is reported before spending GPU hours on
anything else in this list.

---

## 1. Band-count sweep — how many bands do you actually need?

> **Commands: [doc 12 §G](12_commands_datasets.md#g--band-count-sweep).** This section is the reasoning behind them.

**Why.** Right now the claim is binary: 32 bands beat 3 (0.9073 vs 0.7571 balanced accuracy).
That is a good result, but it answers a yes/no question. The interesting question — and the
one a reviewer will ask — is *where the curve turns*: does performance rise smoothly with
band count, or does it saturate at 8, or 12? A saturation point is a **deployable finding**
("a 12-band filter-wheel camera captures most of the benefit"), whereas "32 > 3" is not.

It also controls for a confound the current pair cannot: HSI and RGB differ in *both* channel
count and optics. A sweep that holds the optics fixed (all from the same cube) and varies only
the band count isolates the spectral contribution cleanly.

**How.** `--num_bands` is already a prep flag, and 740 bands are available to select from.

```bash
for N in 3 8 16 32; do
  python prepare_histologyhsi_bc.py \
      --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
      --out_dir ./data/hsi_v8-bands${N} \
      --modality hsi --label_source tissue --patch_size 11 --stride 11 --roi_min_frac 0.8 \
      --band_selection importance --num_bands $N --band_min_gap 8 --band_max_corr 0.95 \
      --split 80_10_10 --capture_gain median_ratio --hsi_value_scale auto \
      --seed 42 --num_workers 8
done

for N in 3 8 16 32; do
  python train.py --profile pipeline_defaults --data_dir ./data/hsi_v8-bands${N}/hsi/ \
      --architecture recursive --normalization global_zscore \
      --loss weighted_ce --class_weight_power 0.75 \
      --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 --checkpoint_metric f1_macro \
      --train_subsample_frac 0.1 --val_subsample_frac 0.1 --eval_test best --seed 42
done
```

**Caveats.** Keep `--seed 42` and `--band_min_gap 8` fixed so the N-band set is a
principled subset, not a different selection algorithm. The N=3 arm is *not* the same as the
RGB arm — it is 3 **narrow** bands chosen by importance, not 3 broad camera channels; that
contrast is itself worth a sentence in the write-up. Budget ~1.8 h of prep per N (dominated by
cube I/O, not band count) — if disk is tight, prep the largest N and subset channels at
training time instead.

---

## 2. `--rgb_source` — is the gap spectral, or optical?

> **Commands: [doc 12 §H](12_commands_datasets.md#h--rgb-source-ablation).** This section is the reasoning behind them.

**Why.** This one closes a real hole in the headline result, and it exists because of a bug
found while validating this dataset (doc 12 §B.1). The RGB arm of
`data/hsi_v8-80_10_10_importance/rgb` was built from a **mixture**: 565 captures from
`RGBImage.png` (an independent camera) and 79 from `SyntheticRGBImage.png` (a rendering of the
cube), with 190 of them NN-resized from a *different field of view* onto the cube's
coordinates. So "RGB" in the current comparison is not one thing.

Two clean versions of the question, and they are different questions:

* `--rgb_source synthetic` — 3 channels **rendered from the same cube**, pixel-registered.
  A strict information subset of the HSI arm. The gap is then purely *spectral*: everything
  else is held constant by construction. **This is the comparison to defend a spectral claim.**
* `--rgb_source camera` — an independent optical sensor, which is what "would an RGB
  microscope do?" actually means. More externally valid, less internally controlled (35 % of
  captures need resizing from a different FOV).

Reporting both is stronger than reporting either, and the difference between them is
interesting on its own.

**How.** Prepare once per source into separate directories, then run §B.3 of doc 12 unchanged:

```bash
for SRC in synthetic camera; do
  python prepare_histologyhsi_bc.py \
      --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
      --out_dir ./data/hsi_v8-rgb-${SRC} --modality both --label_source tissue \
      --patch_size 11 --stride 11 --roi_min_frac 0.8 --band_selection importance --num_bands 32 \
      --split 80_10_10 --rgb_source ${SRC} --seed 42 --num_workers 8
done
```

**Caveat.** `--rgb_source legacy` reproduces the old mixture and is what the existing numbers
came from — use it only to reproduce, never to conclude.

---

## 3. Auxiliary reconstruction as a regulariser

> **Commands: [doc 12 §I](12_commands_datasets.md#i--reconstruction-as-a-regulariser).** This section is the reasoning behind them.

**Why.** `--recon_mode latent` is currently justified as a *deliverable* (you get cubes and
spectral-fidelity metrics). Nobody has measured whether it **helps classification**. On a
small dataset, forcing the latent to retain enough information to reconstruct the input is a
classic regulariser, and PAD — 1,626 training images over 6 classes — is exactly the regime
where it should pay. `train_example_v16_optimal.py`'s own docstring names it "the single most
interesting ablation to run second".

**How.** One pair per dataset, changing exactly one flag:

```bash
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal   # recon_mode none
python train.py --profile paper_recipe          --data_dir ./data/pad_optimal   # latent, lambda_mse 0.1
```

Then sweep the weight if the first pair is promising: `--lambda_mse 0.01 / 0.1 / 1.0`. On HSI
add `--lambda_sam 0.1` — spectral angle is the term that means something there.

**Caveat.** Compare `f1_macro` on the **test** split, not validation, and hold the seed fixed.
The decoder adds parameters and gradient, so a null result is informative too.

---

## 4. Architecture ablation

> **Commands: [doc 12 §J](12_commands_datasets.md#j--architecture-ablation).** This section is the reasoning behind them.

**Why.** Everything reported uses `--architecture recursive` (the TRM core, 0.446 M params).
The repo also ships `split`, `fullchannel` and `efficient`. Without at least one alternative,
"MedMamba-SS-TRM gets 0.9073" is not attributable to the recursive core — it could be the
normalization, the class weighting, or the dataset. A single comparison arm converts a
description into a claim.

**How.**

```bash
for ARCH in recursive split efficient; do
  python train.py --profile pipeline_defaults --data_dir data/hsi_v8-80_10_10_importance/hsi/ \
      --architecture $ARCH --normalization global_zscore \
      --loss weighted_ce --class_weight_power 0.75 \
      --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 --checkpoint_metric f1_macro \
      --train_subsample_frac 0.1 --val_subsample_frac 0.1 --eval_test best --seed 42
done
```

**Caveat.** Parameter counts differ by ~60×, so report params and VRAM alongside accuracy —
otherwise it is not a fair comparison, it is a capacity comparison. Note also that the
`trm_*` flags are **inert** on non-recursive architectures (they parse and land in
`config.json` and do nothing); the banner says so.

---

## 5. PAD spatial-resolution sweep

> **Commands: [doc 12 §K](12_commands_datasets.md#k--pad-spatial-resolution-sweep).** This section is the reasoning behind them.

**Why.** PAD's open problem is that the input does not carry the lesion clearly, and the
default stem stride of 8 discards 98 % of the pixels before the recursive core sees anything.
This is the one axis that has never been varied on this dataset, and the card has the memory
for it: 28×28 tokens costs 1.4 GB of 16.6 GB, 56×56 is projected at ~5.3 GB.

**How.**

```bash
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --target_token_grid 28   # baseline
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --target_token_grid 56
```

**Caveat.** The cost is **time, not memory** — roughly 4× the step cost, turning a ~2.5 h run
into ~10 h. Measure first with `scripts/gpu_tune.py --input_side 224 --batch_size 32
--token_grid 28,56`, and start it when you have a night.

---

## 6. Per-patient reporting (free — already computed)

> **Commands: [doc 12 §L](12_commands_datasets.md#l--per-patient-reporting).** This section is the reasoning behind them.

**Why.** The histology test split holds 348,894 *patches* but only a handful of *patients*.
Patch-level confidence intervals are therefore far too narrow: patches from one slide are not
independent samples. A per-patient breakdown is the honest unit of analysis, and every run
already writes one.

**How.** No new run needed — the sidecars and the metrics already exist:

```bash
python scripts/compare_runs.py --all --filter importance --per_patient
```

**The result, already measured on the existing runs:**

| run | matched? | patch-level `bal` | per-patient mean | per-patient sd | worst patient | patients |
|---|---|---|---|---|---|---|
| HSI `20260906_222725` | bs256 | 0.9073 | **0.8388** | 0.1325 | 0.6375 | 5 |
| RGB `20260909_221520` | bs256 ✓ | 0.7615 | **0.7697** | 0.1664 | 0.5001 | 5 |
| RGB `20260907_025027` | bs256 ✓ | 0.7571 | **0.7674** | 0.1662 | 0.4998 | 5 |
| RGB `20260910_002833` | bs32 ✗ | 0.7689 | 0.7850 | 0.1662 | 0.5000 | 5 |

Only the two `bs256` RGB rows are matched to the HSI arm (same batch, LR, epochs, seed); the
`bs32` row is a different optimiser setting and must not be used for the comparison.

**This changes how the headline result should be stated.** The direction survives — HSI leads
in every arm, and the two matched RGB seeds agree to within **0.0024** — but on the matched
pair the patient-level gap is **0.069**, against **0.146** at patch level, and the per-patient
standard deviation (0.13–0.17) is *larger than the gap*, over **5 validation patients**. Patch-level metrics on
this corpus are measuring 348,894 correlated samples drawn from 45 slides, and they carry far
more apparent precision than the design supports.

Concretely: report HSI > RGB as a **consistent direction across arms**, quote the per-patient
mean with its spread, and do not attach a significance claim to n=5. If one number has to
carry the thesis, this is the argument for leave-one-patient-out cross-validation — which is
affordable here precisely because the patient count is so small.

**PAD note.** PAD carries `images_*.npy` (image-level) rather than `groups_*.npy`, so
`--per_patient` prints `-` there. That is correct rather than a gap: 1,373 patients over 2,298
images is ~1.7 images each, so per-patient cells would mostly be n=1. The image-level
equivalent for PAD tiles is `--eval_group_aggregation mean_prob` (doc 12 §C.3).

---

## 7. Recurrence prediction — **currently blocked**

**Why it is attractive.** `--label_source recurrence` already exists as a flag, and the
dataset is literally named *HistologyHSI-BC-**Recurrence***. Predicting distant recurrence
from histology HSI is a far more clinically interesting question than classifying tissue type,
and it would use the same prepared cubes.

**What blocks it — three separate things, all verified on this download:**

1. **The clinical table is essentially empty.** The workbook has two sheets, `CDE Mapping`
   (a data dictionary) and `Clinical Data`. The `Clinical Data` sheet contains **6 rows ×
   37 columns**, and the 5 data rows do not parse as a patient table — `Case ID` values come
   back as `OS`, `8.0`, `161`, `160`, `57` and `Relapse` as `1`, `0.0`, `3`, `33`, `7.0`.
   There is no usable per-patient outcome for the 45 patients with imaging.
2. **`load_recurrence_labels` reads the wrong sheet.** It uses `wb.active`, which is the
   `CDE Mapping` dictionary, not `Clinical Data`.
3. **It searches for the wrong column name.** It looks for a header containing `"recur"`;
   the outcome column is called **`Relapse`** (with `DFS` and `Vital_status` alongside). It
   would raise `ValueError` even pointed at the right sheet. The function's own docstring
   admits this: *"Inspect your actual sheet once and adjust the column-name matching below."*
4. **The default path is wrong for this download.** The code looks for
   `HistologyHSI-BC-Recurrence-Clinical-Standardized.xlsx`; the file on disk is
   `HistologyHSI-B**RCA**-Recurrence-Clinical-Standardized.xlsx`, so `--clinical_xlsx` must be
   passed explicitly in any case.

**What it would take.** Obtain a populated clinical table (from TCIA, or the authors), then
add a `load_recurrence_labels` override in `prepare_histologyhsi_bc_v8.py` — *not* an edit to
the frozen v6 — selecting the `Clinical Data` sheet by name and mapping `Relapse`. Then:

```bash
python prepare_histologyhsi_bc.py --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
    --out_dir ./data/hsi_v8-recurrence --label_source recurrence \
    --clinical_xlsx "/home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/HistologyHSI-BRCA-Recurrence-Clinical-Standardized.xlsx" \
    --modality both --split 80_10_10 --seed 42 --num_workers 8
```

**Caveat, and it is a serious one.** Even with the table, this is **45 patients** — the label
is per *patient*, so the effective sample size is 45, not 3.1 M patches. With patient-grouped
splits that leaves roughly 4–5 patients in the test split. That is not enough for a
defensible supervised result, and patch-level metrics would be badly misleading (they would
look excellent while measuring almost nothing). If pursued, report it as **leave-one-patient-out
cross-validation** with per-patient predictions, and frame it as exploratory.

---

## 8. Cross-dataset transfer

**Why.** Histology HSI has 3.1 M patches; PAD has 1,626 training images and is visibly
data-starved. Pretraining the shared recursive core on histology and fine-tuning on PAD tests
whether the core learns anything transferable across tissue types and sensors. If it does,
that is a genuine contribution; if it does not, that is an honest negative result about the
architecture's generality.

**How.** No entry point does this today. The pieces exist — `best_model.pt` holds the backbone,
and `training/config_presets.build_model` builds an identical core — so it needs a small
script that loads the histology checkpoint's backbone into a PAD model and skips the
classifier head (3 classes vs 6). Roughly:

```python
sd = torch.load("experiments/<hsi_run>/best_model.pt")["model_state_dict"]
backbone = {k: v for k, v in sd.items() if k.startswith("backbone.")}
model.load_state_dict(backbone, strict=False)   # head stays randomly initialised
```

**Caveats.** Channel counts differ (32 vs 3), so the **stem** cannot transfer — only the core
and anything after it; the stem must be re-initialised. And the two datasets differ in stain,
scale, optics and task, so a null result is likely and would still be worth reporting. Do this
last.

---

## What I would actually do, in order

0. **§6 is already done — act on it.** `scripts/compare_runs.py --all --filter importance
   --per_patient` needed no new run and showed the matched patient-level gap is 0.069
   against the 0.146 quoted at patch level, with a per-patient sd of 0.13–0.17 over 5
   patients. Rewrite
   how the headline result is reported **before** buying more GPU hours.
1. **§2 `--rgb_source synthetic` re-prep and re-run.** Not optional polish: the current
   headline comparison has a known confound (a mixture of two optical sources, 29.5 % of it
   resized from a different field of view) and this removes it. Two preps, two runs.
2. **§1 band-count sweep.** The best genuinely new science available from this data, and the
   only version of the spectral question that yields a curve instead of a yes/no.
3. **§3 reconstruction-as-regulariser** — it also produces the reconstruction figures the
   thesis needs anyway, so it is never wasted GPU time.
4. **§4 architecture ablation.** One alternative arm turns "MedMamba-SS-TRM reaches 0.9073" from
   a description into an attributable claim.
5. **§5 PAD resolution**, if there is a night to spend.

§7 and §8 remain stretch goals. §7 is blocked on data that is not in this download, and no
amount of code fixes 45 patients; §8 needs a small loader script that does not exist yet.
