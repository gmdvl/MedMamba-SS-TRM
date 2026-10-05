# Ready-to-Run Commands — the two real datasets, every analysis

> **2026-09-28:** the commands below were rewritten from `train_example_v16*.py` / `train_example_v18.py` / `scripts/v20_queue.sh` to `train.py --profile …` / `run_experiments.py`, with the same flags. The two resolve every argument identically (`tests/test_train_parity.py`; see [`17_train_and_sweeps.md`](17_train_and_sweeps.md)). The originals are in `archive/`. A pre-2026-09-28 `train_example_v16_optimal_recon.py` command is `--profile paper_recipe`. **2026-10-01:** the profiles were renamed for what they are for (`v18` → `paper_recipe`, `optimal` → `pad_ufes_best_norecon`, `original` → `medmamba_protocol_norecon`, `base` → `pipeline_defaults`); `_norecon` marks the two without reconstruction. The old names still work.


> **Validated against the actual data on this machine**, not written from the docstrings.
> Every count, path and size below was read off disk while writing this file.
>
> | | HistologyHSI-BC-Recurrence | PAD-UFES-20 |
> |---|---|---|
> | root | `/home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/` | `/home/dante/Documents/documents/Masters/courses/Thesis/Dataset/PAD-UFES-20` |
> | raw size | 1.1 TB | 3.4 GB |
> | units | 644 captures, 45 patients | 2,298 images, 1,373 patients |
> | classes | healthy 184 / DCIS 35 / IDC 425 *(captures)* | ACK 730 · BCC 845 · MEL 52 · NEV 244 · SCC 192 · SEK 235 |
> | modalities | **HSI + RGB** | **RGB only** |
> | cube | 600 × 1004 × **740 bands**, 400.5–938.2 nm, uint16 BSQ | — |
>
> **Hardware:** RTX 5060 Ti 16 GB (Blackwell `sm_120`), torch 2.11.0+cu128, CUDA 12.8,
> 32 GB RAM, 16 cores. See [`11_v16_optimal_and_reconstruction.md`](11_v16_optimal_and_reconstruction.md)
> for the GPU budget and the measured results.

There are **three analyses**, and a fourth question the data cannot currently answer —
see [`13_additional_analyses.md`](13_additional_analyses.md).

### Part 1 — the three core analyses

| # | dataset | modality | classes | section |
|---|---|---|---|---|
| A | HistologyHSI-BC | **HSI** (32 of 740 bands) | 3 | [§A](#a--histology-hsi) |
| B | HistologyHSI-BC | **RGB** | 3 | [§B](#b--histology-rgb) |
| C | PAD-UFES-20 | **RGB** | 6 | [§C](#c--pad-ufes-20-rgb) |

Plus [§D](#d--reconstruction-on-any-of-the-three) reconstruction on any of them,
[§E](#e--choosing-batch-size-and-token-grid-on-this-card) GPU tuning, and
[§F](#f--reading-a-run-before-believing-it) how to read a run.

### Part 2 — the six ablations

Each is a **comparison**; none means anything as a single number. Tabulate them with
[§M](#m--tabulating-an-ablation) rather than opening `test_report.json` by hand.

| # | ablation | question it answers | section |
|---|---|---|---|
| G | band count 32 / 16 / 8 / 3 | *how many bands do you actually need?* | [§G](#g--band-count-sweep) |
| H | `--rgb_source synthetic` vs `camera` | is the HSI gap **spectral**, or optical? | [§H](#h--rgb-source-ablation) |
| I | reconstruction on / off | does the aux decoder help **classification**? | [§I](#i--reconstruction-as-a-regulariser) |
| J | recursive vs split vs efficient | is the TRM core earning its place? | [§J](#j--architecture-ablation) |
| K | `--target_token_grid` 28 vs 56 | does PAD need spatial resolution? | [§K](#k--pad-spatial-resolution-sweep) |
| L | per-patient / per-capture | **is the patch-level number honest?** | [§L](#l--per-patient-reporting) |

---

## 0. Before anything

### 0.1 Environment

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
# must print a cu128 build whose arch list contains sm_120.
# A cu124 build (e.g. torch 2.5.1) has NO Blackwell kernels and will fail on this card.

pip install spectral          # ENVI cube IO — required for ANY histology prep (§A/§B)
pip install openpyxl          # only for --label_source recurrence (see doc 13)
```

`spectral` is **not** optional for the histology dataset: `open_envi_lazy` raises `ImportError`
without it, and the prep dies before the first capture.

### 0.1b What changed in v17 (affects every command below)

Four defaults moved. Every command in this document already gets them — they are listed
here so a run's banner is not a surprise:

| default | was | now |
|---|---|---|
| `--compile` | `off` | **`on`** (1.64× on PAD, and lower peak VRAM) |
| `--loader_mode` | `balanced` | **`performance`** |
| `--eval_artifact_stride` | `1` | **`5`** |
| `--architecture`, bare `train_example_v16.py` only | `split` | **`recursive`** |

`--compile on` changes results by ~1.7e-7 against the eager path — not bit-identical.
**To reproduce a run from before v17:** add `--compile off --loader_mode balanced
--eval_artifact_stride 1`. The banner says which path you are on, in both directions.

Two optimisations were measured and **rejected**: `--no_trm_checkpoint_core` (12% faster
for 4.8× the memory, OOMs at batch 64) and larger batches (throughput *falls*: 120 → 106 →
98 samples/s at bs 32/64/128). Do not reach for either.
See [`14_performance_and_progress.md`](14_performance_and_progress.md).

Two things also changed that need no flag: progress lines now **redraw in place in a
terminal** and stay whole lines when redirected through `| tee`; and the run raises its own
open-file limit at startup, with a single-process retry if gate G6's whole-split test loader
still runs out of descriptors (§F).

### 0.2 Two things about this download specifically

1. **The ROI folder has a space in its name** — `01_03_HSI ROI_Annotations`, where the code's
   first guess is `01_03_HSI_ROI_Annotations`. `_find_roi_root`'s fuzzy fallback ("roi" +
   "annot") resolves it, so **no flag is needed**. Verified: it finds all 45 patients'
   GeoJSONs. If you ever see `roi_root: None` in the prep log, ROI masking silently turned
   off and every patch — including background glass — went into the dataset.
2. **`--rgb_source` now matters (new flag).** See §B.1 before running any RGB histology prep.

### 0.3 Disk and time

| output | size | wall clock |
|---|---|---|
| `hsi_v8-…/hsi` (2,452,086 + 334,516 + 348,894 patches, float16) | **23 GB** | ~1.8 h total for both modalities, `--num_workers 8` |
| `hsi_v8-…/rgb` (same counts, float32) | 4.3 GB | (same pass) |
| `pad_optimal` (1,626 + 328 + 344 whole images) | 1.3 GB | minutes |
| `pad_optimal_mil9` (9 tiles/image) | 3.0 GB | minutes |

Budget **~30 GB free** for a histology prep. The prep is resumable per capture — re-run the
identical command after a kill and it continues from `_progress.json`.

Since v17, prep prints a rate and an ETA for both phases, so these figures are a sanity
check rather than the only estimate you get:

```
  [capture-gain] 412/644  64%  388 usable  12.40/s  ETA 18s
  [extract] 412/644  64%  3.10/s  ETA 1m15s
```

Pass 1 (the per-capture gain estimate) now runs over `--num_workers` processes instead of
serially in the parent — it was doing ~400 single-pixel ENVI reads per capture for all 644
captures on one core. The medians are bit-identical either way.

**Training wall clock**, measured on the RTX 5060 Ti:

| run | epochs | pre-v17 | with the v17 defaults |
|---|---|---|---|
| PAD `pad_optimal`, bs 32 | 63 (early-stopped from 200) | 43.4 s/epoch, 47 min | see below |
| HSI `hsi_v8-…/hsi`, bs 256, 10% subsample | 12 | 679 s/epoch, 2.3 h | HSI already used the v17 settings |

The PAD figure is the one that moves: that run went through the *best-metrics* entry point
and still ran eager, on the `balanced` loader, writing all 10 plots every epoch, because
none of that was in `OPTIMAL_DEFAULTS` until v17. Fill in your own number from
`s_per_step` in `history.json`; see
[`14_performance_and_progress.md`](14_performance_and_progress.md).

---

## A — Histology, HSI

The headline result: **test accuracy 0.9436, balanced 0.9073, macro-F1 0.8580**.

### A.1 Prepare

One pass writes **both** modalities into `hsi/` and `rgb/` siblings with *identical sample
counts and identical splits*. That shared split is exactly what makes §B a controlled
comparison against §A, so prepare them together — never in two separate runs.

```bash
python prepare_histologyhsi_bc.py \
    --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
    --out_dir ./data/hsi_v8-80_10_10_importance \
    --modality both --label_source tissue \
    --patch_size 11 --stride 11 --roi_min_frac 0.8 \
    --band_selection importance --num_bands 32 \
    --band_min_gap 8 --band_max_corr 0.95 --band_min_coverage 0.30 \
    --split 80_10_10 --split_strategy stratified \
    --capture_gain median_ratio --hsi_value_scale auto \
    --rgb_source synthetic \
    --seed 42 --num_workers 8 --verify_level deep
```

What each non-obvious flag buys:

| flag | why |
|---|---|
| `--num_bands 32` of 740 | `importance` selection with a minimum spacing (`--band_min_gap 8`) and a decorrelation cap (`--band_max_corr 0.95`), so the 32 are spread over 400–938 nm instead of clustering |
| `--capture_gain median_ratio` | per-capture flat-field residual correction; closes the patient-68 exposure fault (IDC captures at ~50% intensity) **at the source**. Clipped captures are logged as SUSPECT in `capture_gain_report.json` — read it |
| `--hsi_value_scale auto` | lands values in reflectance-like `[0, ~1.5]` so `float16` storage is safe; halves the 23 GB. Resolved to **8618.75** on this corpus |
| `--roi_min_frac 0.8` | a patch must be ≥80% inside the annotated ROI, so background glass is excluded |
| `--split_strategy stratified` + patient grouping | **no patient appears in two splits.** Non-negotiable: this dataset has 45 patients and 644 captures, so an image-level split would leak the same slide across train and test |
| `--verify_level deep` | re-reads every produced array end-to-end before deleting shards |

Afterwards, sanity-check the prep before spending GPU hours:

```bash
cat data/hsi_v8-80_10_10_importance/capture_gain_report.json | head -40   # suspect captures
cat data/hsi_v8-80_10_10_importance/band_selection_report.json           # which 32 bands
cat data/hsi_v8-80_10_10_importance/class_coverage_report.json           # no empty class/split
python scripts/derive_patch_groups.py --out_dir data/hsi_v8-80_10_10_importance  # (v8 writes these already)
```

### A.2 The baseline bar

```bash
python scripts/shallow_baseline_v16.py \
    --data_dir data/hsi_v8-80_10_10_importance/hsi \
    --normalization global_zscore \
    --out data/hsi_v8-80_10_10_importance/hsi/shallow_baseline.json
```

A linear probe on the same inputs. The trainer prints it as `[baseline]`; a network **below**
it has an optimisation problem, not a data problem.

### A.3 Train

```bash
python train.py --profile pipeline_defaults \
    --data_dir data/hsi_v8-80_10_10_importance/hsi/ \
    --architecture recursive --trm_mixer mlp --trm_dim 128 \
    --trm_deep_supervision_steps 3 --no_trm_halting \
    --normalization global_zscore \
    --loss weighted_ce --class_weight_power 0.75 --sampler none \
    --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 \
    --checkpoint_metric f1_macro --eval_test best \
    --train_subsample_frac 0.1 --val_subsample_frac 0.1 \
    --loader_mode performance --num_workers 4 \
    --seed 42
```

* `--train_subsample_frac 0.1` uses 10% of 2.45 M patches — 958 steps/epoch, ~680 s/epoch,
  **874 MB of VRAM**. Drop it (`1.0`) for the full split; nothing else needs to change, and
  the card has room (see doc 11 §4).
* `--class_weight_power 0.75` not 1.0: measured on this dataset, power 1.00 → macro-F1 0.8038,
  **0.75 → 0.8474**, 0.50 → 0.8391, 0.00 → 0.6657. DCIS is 35 captures against IDC's 425.
* `--no_trm_halting`: the halt head sat at BCE ≈ 0.66 against ln 2 = 0.693 for 33 epochs —
  chance, and ~20% of the objective.

### A.4 Full-split variant

```bash
python train.py --profile pipeline_defaults --data_dir data/hsi_v8-80_10_10_importance/hsi/ \
    --architecture recursive --normalization global_zscore \
    --loss weighted_ce --class_weight_power 0.75 \
    --batch_size 256 --lr 1e-4 --epochs 30 --amp bf16 \
    --checkpoint_metric f1_macro --early_stop_patience 8 --eval_test best --seed 42
```

~10× the steps per epoch (≈9,580). Expect ~1.9 h/epoch at the measured 710 ms/step — start it
overnight, and check `[ema]` in the preflight first.

---

## B — Histology, RGB

The controlled comparison against §A: **same tissue, same patients, same split, same
coordinates, 3 channels instead of 32.**

### B.1 `--rgb_source` — read this before preparing

Every capture folder ships **two** renderings, and they are not interchangeable:

| file | resolution | cube-aligned | notes |
|---|---|---|---|
| `SyntheticRGBImage.png` | 1004 × 600 | **644/644 (100 %)** | rendered from the cube, pixel-registered with it, no overlay |
| `RGBImage.png` | mixed — 416 at 1004 × 600, **228 at 5472 × 3648** | 416/644 (65 %) | an independent camera; the wide-field ones are a *different field of view*, and some carry the **green ROI annotation burned into the pixels** |

The frozen `prepare_histologyhsi_bc_v6._find_rgb_image` ranks candidates by
`0 if "rgb" in name or "synt" in name else 1` — and **both** filenames contain "rgb", so both
score 0 and the winner was whatever `os.listdir` returned first. On this machine that is
`RGBImage.png` for 565 captures and `SyntheticRGBImage.png` for 79: the RGB arm was built from
a **mixture of two optical sources**, with **190 captures (29.5 %) NN-resized from a different
field of view** onto the cube's patch coordinates and ROI mask. Nothing is aligned to the HSI
there, and the mixture is not reproducible across filesystems.

`prepare_histologyhsi_bc_v8.py` now makes the choice explicit and deterministic (ties broken by
sorted filename, never by directory order):

```
--rgb_source synthetic   (default)  644/644 cube-aligned, 0 % misaligned  <- use this
--rgb_source camera                 644/644 RGBImage.png, 35.4 % misaligned
--rgb_source legacy                 reproduces the old 565/79 mix, 29.5 % misaligned
```

> **`data/hsi_v8-80_10_10_importance/rgb` as it exists today was built under `legacy`.**
> (Verified: its `dataset_manifest.json` carries no `rgb_source` key at all — it predates the
> flag.) The numbers in doc 11 §1.1 come from it. To reproduce those runs exactly, pass
> `--rgb_source legacy`. For any comparison you intend to *defend*, re-prep with the default
> into a **new** `--out_dir` and rerun both arms.
>
> **Status:** the re-prep exists — `data/hsi_v8-80_10_10_importance-new`, manifest
> `rgb_source: synthetic`, both `hsi/` and `rgb/` written from the same `--modality both`
> pass. Its **HSI** arm has been run (`20260913_121925`, 10 % subsample: test accuracy
> 0.9231, balanced 0.8822, macro-F1 0.8182) and a full-split HSI run is in progress. **The
> matched RGB arm on `-new` has not been run yet**, so the defensible HSI-vs-RGB pair does
> not exist on this dataset; doc 11 §1.1 still quotes the `legacy` pair.

### B.2 Prepare

Already done by §A.1 — one `--modality both` pass writes `hsi/` and `rgb/` together. Only if
you need RGB alone:

```bash
python prepare_histologyhsi_bc.py \
    --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
    --out_dir ./data/hsi_v8-rgbonly --modality rgb --label_source tissue \
    --patch_size 11 --stride 11 --roi_min_frac 0.8 \
    --split 80_10_10 --rgb_source synthetic --seed 42 --num_workers 8
```

**Do not use this for the HSI-vs-RGB comparison.** A separate prep re-derives the split; only
the `--modality both` pass guarantees the two arms share it.

### B.3 Train — identical flags to §A.3, only `--data_dir` changes

```bash
python train.py --profile pipeline_defaults \
    --data_dir data/hsi_v8-80_10_10_importance/rgb/ \
    --architecture recursive --trm_mixer mlp --trm_dim 128 \
    --trm_deep_supervision_steps 3 --no_trm_halting \
    --normalization global_zscore \
    --loss weighted_ce --class_weight_power 0.75 --sampler none \
    --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 \
    --checkpoint_metric f1_macro --eval_test best \
    --train_subsample_frac 0.1 --val_subsample_frac 0.1 \
    --loader_mode performance --num_workers 4 \
    --seed 42
```

**Change nothing else.** The claim is "HSI beats RGB"; every flag that differs between the two
runs is a rival explanation. The trainer auto-detects modality from the presence of
`wavelengths.npy` (`hsi/` has it, `rgb/` does not) — you do not pass `--modality`.

Measured under `legacy`: HSI 0.9073 balanced / 0.8580 macro-F1 vs RGB 0.7571 / 0.7282, with a
**repeat of the same seed** at 0.7615 / 0.7332.

> **Corrected in v18.** That second RGB run (`20260909_221520`) records `seed: 42` in its own
> `config.json` — the same seed as `20260907_025027`. It is a **nondeterminism replicate**
> (`deterministic: false`, `--compile on`), not a second seed, and it is more useful read that
> way: two byte-identical command lines differ by **0.44 points of balanced accuracy and 0.005
> macro-F1**, which is the run-to-run floor a 15-point RQ0 margin has to clear. Seed variance is
> still unmeasured — that is v18's E7. And both arms here are on the **`legacy` RGB source**; the
> corrected pair is E1 in [`15_v18_commands.md`](15_v18_commands.md).

---

## C — PAD-UFES-20, RGB

6 classes, 2,298 clinical photographs, 1,373 patients, **MEL at 52 images (2.3 %)**.

### C.1 Prepare

```bash
python prepare_pad_ufes_20_optimal.py \
    --root /home/dante/Documents/documents/Masters/courses/Thesis/Dataset/PAD-UFES-20 \
    --out_dir ./data/pad_optimal \
    --tiling whole --img_size 224 \
    --split 70/15/15 --split_strategy stratified --balance_classes none \
    --seed 42 --num_workers 8
```

`--tiling whole` (the default) is the single largest measured effect on this dataset: a whole-image
linear probe scores 0.325 macro-F1 against 0.185 on the old 11×11 grid, because an 11×11 window is
0.24 % of the frame and usually misses the lesion. `--split 70/15/15` is a **three-way** spec —
the prep core's `80_20` default writes no `X_val.npy`, and the trainer then falls back to the test
split as validation and refuses to report a test number at all.

Patient grouping is automatic (`group_id` = the `patient_id` column) and not optional: the
MedMamba paper's published PAD numbers use an image-level split that shares patients across
splits and inflates results.

Patch-MIL variant, for image-level metrics out of 112×112 tiles:

```bash
python prepare_pad_ufes_20_optimal.py \
    --root /home/dante/Documents/documents/Masters/courses/Thesis/Dataset/PAD-UFES-20 \
    --out_dir ./data/pad_optimal_mil9 --tiling mil9 --split 70/15/15 --seed 42 --num_workers 8
# writes images_*.npy so --eval_group_aggregation can rebuild image-level metrics
```

### C.2 Baseline

```bash
python scripts/shallow_baseline_v16.py --data_dir ./data/pad_optimal \
    --normalization global_zscore --out ./data/pad_optimal/shallow_baseline.json
```

### C.3 Train — climb the ladder, do not skip to `full`

```bash
# 1. can the model fit at all? every regulariser, the class weighting and EMA OFF,
#    so train accuracy is a clean readout. Expect well above 0.31 within ~30 epochs.
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --stage fit

# 2. add the class weighting back
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --stage balance

# 3. the full configuration
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal
```

`train_example_v16_optimal.py` already defaults everything: `focal_weighted` loss, γ 1.5,
`class_weight_power` 0.75, `global_zscore`, batch 32, lr 3e-4, 200 epochs, the custom
augmentation preset with `band_dropout`/`spectral_mask` **off** (they zero a whole colour
channel at C=3), `patch_size` auto → stride 8, and an EMA rate resolved from steps/epoch.
Run `--help` to see them; the banner prints every one the command line moved.

Best measured so far: **0.4564 accuracy / 0.5020 balanced / 0.4213 macro-F1** (with
`--class_weight_power 1.0 --lr 1e-3` on an undersampled variant). That clears the whole-image
linear probe (0.337 / 0.325) it was built to beat.

MIL variant with image-level test metrics:

```bash
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal_mil9 \
    --eval_group_aggregation mean_prob
```

Spend the idle 91 % of the card on spatial detail instead (~4× the wall clock):

```bash
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --target_token_grid 56
```

> Read balanced accuracy and per-class support beside any macro-F1 from this dataset. The
> test split holds 344 images with **MEL at about 7 of them** — one image flipping moves
> MEL F1 by ~0.1.

---

## D — Reconstruction, on any of the three

Adds the reconstructed cubes, their per-sample metrics, and the paper figures. Everything
else is the §A/§B/§C configuration unchanged.

```bash
# HSI — the modality where spectral angle actually means something
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance/hsi/ \
    --lambda_mse 0.1 --lambda_sam 0.1 --recon_sample_per_class 5 \
    --normalization global_zscore --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16

# Histology RGB
python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance/rgb/ \
    --lambda_mse 0.1 --batch_size 256 --lr 1e-4 --epochs 20

# PAD RGB
python train.py --profile paper_recipe --data_dir ./data/pad_optimal --lambda_mse 0.1
```

Writes `<run>/reconstruction/`: `samples.npz` (the `x_true`/`x_recon` cubes),
`samples_metrics.json` (SAM, RMSE, PSNR, SID, spectral SSIM, **2-D SSIM**, peak-position
error), `provenance.json`, `status.json`, and the figures.

**Recover artifacts from a run that already finished** — the decoder is still in
`best_model.pt`, so no retraining is needed, and this is the only route to a **held-out test**
reconstruction number:

```bash
python scripts/export_recon_samples.py \
    --run_dir experiments/20260904_230812_portance_undersample-hsi_recursive_hsi_mlp_sup3_bs256_bf16_wce_modover_ppz_sub01_vsub02 \
    --split test --recon_sample_per_class 3
```

Runs that still carry a usable decoder: `20260904_230812` (30 ep, HSI — the best),
`20260904_164643` (6 ep, HSI), `20260906_031815` (10 ep, PAD RGB), `20260904_121119` (2 ep, HSI).

---

## E — Choosing batch size and token grid on this card

Nothing above is at the card's limit — every configuration uses 1–9 % of 16 GB. But note
**the memory headroom is not convertible into throughput here**: raising the batch makes
runs *slower* on this card (120 → 106 → 98 samples/s at bs 32/64/128), and turning off
`--trm_checkpoint_core` buys 12 % for 4.8× the memory and OOMs at batch 64. Both were
measured in v17 and rejected; see [`14_performance_and_progress.md`](14_performance_and_progress.md).
The axis with real headroom left is `--target_token_grid` (§K), and its cost is wall clock.

Measure before changing:

```bash
# PAD whole-image: batch and spatial resolution
python scripts/gpu_tune.py --input_side 224 --batch_size 16,32,64 --token_grid 28,56

# Histology HSI: 32 channels, 11x11, 3 classes
python scripts/gpu_tune.py --input_side 11 --in_channels 32 --num_classes 3 \
    --batch_size 256,512,1024 --token_grid 11

# the two questions v17 already answered on PAD - re-run them if the card or torch changes
# (compile: 1.64x and LOWER memory -> now the default. checkpoint_core off: rejected.)
# --spectral_checkpointing matters on HSI: without it the sweep reads ~10 GB against the
# 874 MB a real run uses, because the production path enables it at in_channels >= 16.
python scripts/gpu_tune.py --input_side 11 --in_channels 32 --num_classes 3 \
    --batch_size 256 --token_grid 11 --compile off,on --spectral_checkpointing auto
python scripts/gpu_tune.py --input_side 224 --batch_size 32 --token_grid 28 \
    --checkpoint_core on,off
```

---

## F — Reading a run before believing it

`[baseline]` the linear-probe ceiling · `[budget]` optimizer steps · `[gpu-budget]` VRAM and
how much is idle · `[ema]` **how much of the random initialisation is still inside the model
validation is computed on** — a stale EMA and a collapsed model both show balanced accuracy
pinned at exactly `1/k` (0.3333 for histology, 0.1667 for PAD) and are indistinguishable in
the epoch log.

Per-run artifacts worth opening: `test_report.json` (`sklearn_metrics`), `history.json`
(`GPU_memory_MB`, per-epoch per-class F1, and the v17 timing fields `train_time_s` /
`val_time_s` / `artifact_time_s` / `data_wait_s` / `s_per_step`), `split_drift_report.json`,
`leakage_report.json`, `per_patient_metrics.json`, `gates.json`.

If `gates.json` says `"G6": false`, read `gate_g6_failure.json` next to it — it carries the
exception and traceback. Since v17 S11 a G6 that fails on **file-descriptor exhaustion**
(`Too many open files`, which the whole-split test loader can hit at the end of a long run)
is retried once single-process and usually recovers; the log says so explicitly:

```
WARNING: test-split evaluation ran out of file descriptors with 4 worker(s). Retrying single-process.
[test-eval] single-process retry succeeded.
```

---
---

# Part 2 — the six ablations

Every section below is a **comparison**. Run all arms with the same `--seed`, change exactly
one flag between them, and read the result with [§M](#m--tabulating-an-ablation).

---

## G — Band-count sweep

**The question.** The current claim is binary: 32 bands beat 3. The deployable question is
*where the curve turns* — if performance saturates at 8 or 12 bands, that says a filter-wheel
camera captures most of the benefit, which is a far stronger finding than "more is better".

It also removes a confound §A/§B cannot: HSI and RGB differ in **both** channel count and
optics. A sweep holds the optics fixed (every arm comes from the same cube) and varies only
the band count, isolating the spectral contribution.

**Prepare** — 740 bands are available to select from; only `--num_bands` changes:

```bash
for N in 3 8 16 32; do
  python prepare_histologyhsi_bc.py \
      --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
      --out_dir ./data/hsi_v8-bands${N} \
      --modality hsi --label_source tissue \
      --patch_size 11 --stride 11 --roi_min_frac 0.8 \
      --band_selection importance --num_bands $N \
      --band_min_gap 8 --band_max_corr 0.95 --band_min_coverage 0.30 \
      --split 80_10_10 --split_strategy stratified \
      --capture_gain median_ratio --hsi_value_scale auto \
      --seed 42 --num_workers 8 --verify_level deep
done
```

**Train** — identical flags across arms; the trainer reads the channel count from the data:

```bash
for N in 3 8 16 32; do
  python train.py --profile pipeline_defaults --data_dir ./data/hsi_v8-bands${N}/hsi/ \
      --architecture recursive --trm_mixer mlp --no_trm_halting \
      --normalization global_zscore \
      --loss weighted_ce --class_weight_power 0.75 --sampler none \
      --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 \
      --checkpoint_metric f1_macro --eval_test best \
      --train_subsample_frac 0.1 --val_subsample_frac 0.1 --seed 42
done

python scripts/compare_runs.py 'experiments/*bands*' --per_patient
```

**Cost.** ~1.8 h prep per arm — dominated by cube I/O, *not* by band count, so all four cost
roughly the same. Storage scales with N: the 32-band arm is 23 GB, so 3+8+16+32 ≈ 43 GB total.
If disk is tight, prep only `--num_bands 32` and subset channels at training time instead.

**Caveats.** Keep `--seed 42` and `--band_min_gap 8` fixed, or you are comparing selection
algorithms rather than band counts. The **N=3 arm is not the RGB arm**: it is 3 *narrow*
importance-selected bands, not 3 broad camera channels — the difference between §G's N=3 and
§B is itself worth a sentence, because it separates "3 numbers per pixel" from "3 *broad*
numbers per pixel".

---

## H — RGB-source ablation

**The question.** Is the HSI advantage *spectral*, or is it partly an artefact of which camera
the RGB arm came from? See §B.1 for the defect this exists to close: the RGB arm was built
from a mixture of two optical sources, 29.5 % of it resized from a different field of view.

Two clean versions, and they are **different questions** — report both:

| arm | what it is | what the gap then means |
|---|---|---|
| `--rgb_source synthetic` | 3 channels rendered from the same cube, pixel-registered, a strict information subset | purely **spectral** — everything else held constant by construction |
| `--rgb_source camera` | an independent optical sensor | "would an RGB microscope do?" — externally valid, less internally controlled (35 % need resizing) |

```bash
for SRC in synthetic camera; do
  python prepare_histologyhsi_bc.py \
      --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
      --out_dir ./data/hsi_v8-rgb-${SRC} \
      --modality both --label_source tissue \
      --patch_size 11 --stride 11 --roi_min_frac 0.8 \
      --band_selection importance --num_bands 32 \
      --split 80_10_10 --capture_gain median_ratio --hsi_value_scale auto \
      --rgb_source ${SRC} --seed 42 --num_workers 8
done

for SRC in synthetic camera; do
  python train.py --profile pipeline_defaults --data_dir ./data/hsi_v8-rgb-${SRC}/rgb/ \
      --architecture recursive --normalization global_zscore \
      --loss weighted_ce --class_weight_power 0.75 \
      --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 \
      --checkpoint_metric f1_macro --eval_test best \
      --train_subsample_frac 0.1 --val_subsample_frac 0.1 --seed 42
done
```

Each `--modality both` pass also writes a matching `hsi/` arm, so the HSI comparison is
re-derived on the same split rather than borrowed from another prep.

**Caveat.** `--rgb_source legacy` reproduces the old 565/79 mixture and is what the existing
published numbers came from. Use it to *reproduce*, never to *conclude*.

---

## I — Reconstruction as a regulariser

**The question.** `--recon_mode latent` is currently justified as a deliverable (you get cubes
and spectral-fidelity metrics). Nobody has measured whether it **helps classification**.
Forcing the latent to retain enough information to reconstruct the input is a classic
regulariser, and PAD — 1,626 training images over 6 classes — is exactly the regime where it
should pay.

```bash
# PAD: off vs on, one flag apart
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --seed 42
python train.py --profile paper_recipe          --data_dir ./data/pad_optimal --seed 42

# weight sweep, if the pair is promising
for L in 0.01 0.1 1.0; do
  python train.py --profile paper_recipe --data_dir ./data/pad_optimal \
      --lambda_mse $L --seed 42
done

# HSI: add the spectral-angle term, which is the one that means something there
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance/hsi/ \
    --lambda_mse 0.1 --lambda_sam 0.1 --recon_sample_per_class 5 \
    --normalization global_zscore --loss weighted_ce --class_weight_power 0.75 \
    --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 \
    --train_subsample_frac 0.1 --val_subsample_frac 0.1 --seed 42
```

**Caveats.** Compare `f1_macro` on the **test** split, not validation, and hold the seed fixed.
The decoder adds parameters and gradient, so it costs both memory and step time — a null
result is informative and should be reported as one. This run also produces the reconstruction
artifacts the thesis needs anyway (§D), so it is never wasted.

---

## J — Architecture ablation

**The question.** Everything reported uses `--architecture recursive` (the TRM core, 0.446 M
params). Without an alternative arm, "MedMamba-SS-TRM reaches 0.9073" is not attributable to the
recursive core — it could be the normalization, the class weighting, or the dataset. One
comparison arm turns a description into a claim.

All four architectures have presets for both modalities (`split`, `fullchannel`, `efficient`,
`recursive`), so the same command runs on HSI and RGB:

```bash
for ARCH in recursive split efficient; do
  python train.py --profile pipeline_defaults --data_dir data/hsi_v8-80_10_10_importance/hsi/ \
      --architecture $ARCH --normalization global_zscore \
      --loss weighted_ce --class_weight_power 0.75 --sampler none \
      --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 \
      --checkpoint_metric f1_macro --eval_test best \
      --train_subsample_frac 0.1 --val_subsample_frac 0.1 --seed 42
done

python scripts/compare_runs.py --all --filter importance-hsi --per_patient
```

**Caveats, and they matter.**

* Parameter counts differ by up to ~60×. Report **params and VRAM alongside accuracy**
  (`backbone_num_params` in `config.json`, `GPU_memory_MB` in `history.json`, both columns in
  `compare_runs.py --csv`), or it is a capacity comparison, not an architecture one.
* The `trm_*` flags are **inert** on non-recursive architectures — they parse, land in
  `config.json`, and do nothing. The banner says so; do not read them as active settings.
* `--classifier_dropout` raises on `split`/`fullchannel` under plain v16 (their head hardcodes
  0.1). `train_example_v16_optimal.py` downgrades that to a warning; plain `train_example_v16.py`
  does not, so simply omit the flag, as above.

---

## K — PAD spatial-resolution sweep

**The question.** PAD's open problem is that the input does not clearly carry the lesion, and
the default stem stride of 8 discards **98 % of the pixels** before the recursive core sees
anything. This is the one axis PAD has never varied, and the card has the memory: 28×28 tokens
costs 1.4 GB of 16.6 GB; 56×56 is projected at ~5.3 GB.

```bash
# measure the price first — one command, no training run
python scripts/gpu_tune.py --input_side 224 --batch_size 32 --token_grid 28,56

python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --target_token_grid 28 --seed 42
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --target_token_grid 56 --seed 42
```

**Caveat.** The cost is **time, not memory** — roughly 4× the step cost, turning a ~2.5 h
200-epoch run into ~10 h. Start it overnight. If it OOMs anyway, the preflight's `[gpu-budget]`
block tells you which knob to move and in which direction.

---

## L — Per-patient reporting

**The question — and it is the most important one here.** The histology test split holds
348,894 *patches* but the corpus has only **45 patients**. Patches from one slide are not
independent samples, so patch-level metrics carry far more apparent confidence than the data
supports. The per-patient breakdown is the honest unit of analysis.

**No new run is needed** — every histology run already writes it. `prepare_histologyhsi_bc_v8.py`
emits `groups_*.npy` / `captures_*.npy` / `capture_index.json` (`--emit_group_sidecars`, on by
default), and `TrainerG_v12` turns those into `per_patient_metrics.json` and
`per_capture_metrics.csv` on the validation split.

```bash
python scripts/compare_runs.py --all --filter importance --per_patient

# the raw per-patient and per-capture detail for one run
python -m json.tool experiments/<run>/per_patient_metrics.json | head -60
column -s, -t < experiments/<run>/per_capture_metrics.csv | head -30
```

**What this already shows on the existing runs** — and why it belongs in the thesis:

| run | matched? | patch-level `bal` | per-patient mean | per-patient sd | worst patient | patients |
|---|---|---|---|---|---|---|
| HSI `20260906_222725` | bs256 | 0.9073 | **0.8388** | 0.1325 | 0.6375 | 5 |
| RGB `20260909_221520` | bs256 ✓ | 0.7615 | **0.7697** | 0.1664 | 0.5001 | 5 |
| RGB `20260907_025027` | bs256 ✓ | 0.7571 | **0.7674** | 0.1662 | 0.4998 | 5 |
| RGB `20260910_002833` | bs32 ✗ | 0.7689 | 0.7850 | 0.1662 | 0.5000 | 5 |

Only the two `bs256` RGB rows are matched to the HSI arm (same batch, LR, epochs, seed); the
`bs32` row is a different optimiser setting and must not be used for the comparison.

The direction is consistent — HSI is ahead in every arm, and the two matched RGB seeds agree
to within **0.0024** — but on the matched pair **the patient-level gap is 0.069, against 0.146
at patch level, and the per-patient standard deviation (0.13–0.17) is larger than the gap**,
over 5 validation patients. Quote the per-patient spread, and do not attach a significance
claim to n=5. This is the single strongest argument for leave-one-patient-out
cross-validation.

**PAD note.** PAD carries `images_*.npy` (image-level) rather than `groups_*.npy`, so
`--per_patient` shows `-` there. That is appropriate: PAD has 1,373 patients over 2,298
images — ~1.7 images each — so a per-patient breakdown would mostly be n=1 cells. The
equivalent aggregation for PAD tiles is `--eval_group_aggregation mean_prob` (§C.3).

---

## M — Tabulating an ablation

`scripts/compare_runs.py` lines finished runs up side by side. Read-only; it writes
nothing unless `--csv` is given.

```bash
python scripts/compare_runs.py 'experiments/*bands*'            # one sweep
python scripts/compare_runs.py --all --filter hsi_v8            # one dataset
python scripts/compare_runs.py --all --per_patient              # the honest unit
python scripts/compare_runs.py --all --csv results.csv          # every field, machine-readable
python scripts/compare_runs.py --all --sort balanced_accuracy
python scripts/compare_runs.py --all --incomplete               # include runs that never reached G6
```

`acc` / `bal` / `f1M` are **test-split** metrics from `test_report.json`, computed on the
checkpoint the run actually selected — not the best validation epoch. A run with no
`test_report.json` never reached gate G6 and is hidden unless `--incomplete` is passed.
