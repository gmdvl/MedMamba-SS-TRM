# v20 — GPU/host commands for manuscript v18

> **2026-09-28:** the commands below were rewritten from `train_example_v16*.py` / `train_example_v18.py` / `scripts/v20_queue.sh` to `train.py --profile …` / `run_experiments.py`, with the same flags. The two resolve every argument identically (`tests/test_train_parity.py`; see [`17_train_and_sweeps.md`](17_train_and_sweeps.md)). The originals are in `archive/`. A pre-2026-09-28 `train_example_v16_optimal_recon.py` command is `--profile paper_recipe`. **2026-10-01:** the profiles were renamed for what they are for (`v18` → `paper_recipe`, `optimal` → `pad_ufes_best_norecon`, `original` → `medmamba_protocol_norecon`, `base` → `pipeline_defaults`); `_norecon` marks the two without reconstruction. The old names still work.


Companion to `plan/manuscript_v18_evidence_upgrade.plan.md`, which tracks status. Every command here runs on the **host**, not in the VS Code sandbox: the host has the GPU, `torch cu128` and the `spectral` package.

Run everything from the repository root.

---

## 0. Before you start (5 minutes)

**Environment.** Use the conda env that ran the reported experiments (torch 2.11 + cu128; `spectral` and `openpyxl` installed). Then check the new code:

```bash
python -m pytest -q tests/test_trainsel_prep_v20.py tests/test_fixed_eta_v20.py \
    tests/test_analysis_v20.py tests/test_check_eval_mode_v20.py \
    tests/test_prepare_hmi_lusc_v20.py tests/test_frozen_files_untouched.py
# expect: 14 passed
```

**Disk.** `/data` has about 321 GB free, and `/home` about 50 GB, which is too little to extract HMI-LUSC there.

| Item | Peak disk | Where |
| --- | ---: | --- |
| G1 build `data/hsi_v9-trainsel` (hsi + rgb) | 27 GB | `/data` |
| G10 CV folds, 5 × 27 GB | 135 GB, or ≈ 27 GB with `CV_CLEAN=1` | `/data` |
| G9 band subsets 16 / 8 | 17 GB | `/data` |
| G11 HMI-LUSC extracted cubes | 49 GB | `/data/dante_data/hmi_lusc_raw` |
| G11 HMI-LUSC pool + 5 folds | ≈ 35 GB | `/data` |

Use `CV_CLEAN=1` (§3). It deletes each fold's `X_*.npy` after both of the fold's runs finish, and keeps the labels and groups that the analysis needs.

**Run tags.** Every run gets a `--run_tag`. `run_experiments.py` and `scripts/analysis_v20.py` find runs by tag, so do not change the tags.

| Runs | Tag | Data dir |
| --- | --- | --- |
| Headline, 20 epochs | `v20-head-s{S}` | `data/hsi_v9-trainsel/{hsi,rgb}` |
| HybridSN, SpectralFormer | `v20-s{S}` | `data/hsi_v9-trainsel/hsi` |
| MedMamba | run name `v20_s{S}` | same, in `../medmamba-original/MedMamba/runs_hsi/` |
| Controls, 12 epochs | `v20-abl-base-s{S}`, `v20-k2-s{S}`, `v20-recon-off-s{S}`, `v20-enc-index-s{S}` | `.../hsi` |
| Depth, band count, hierarchy | `v20-depth-n{T}`, `v20-bands{N}`, `v20-arch-{split,fullchannel}` | `.../hsi`, `data/hsi_v9-bands{N}` |
| Breast CV | `v20-cv-f{F}` | `data/hsi_v9-cv5-f{F}/{hsi,rgb}` |
| HMI-LUSC CV | `v20-lusc-f{F}` | `data/lusc_v20-f{F}/hsi` |

---

## 1. The easy way: the resumable queue

```bash
# Priority 1 (required for the target), in dependency order:
CV_CLEAN=1 nohup python run_experiments.py sweeps/paper_evidence_upgrade.py p1 > logs/paper_evidence_upgrade_p1.out 2>&1 &
# Priority 2, afterwards (or in a second terminal once 'prep' has finished):
nohup python run_experiments.py sweeps/paper_evidence_upgrade.py p2 > logs/paper_evidence_upgrade_p2.out 2>&1 &
# Progress at any time:
python run_experiments.py sweeps/paper_evidence_upgrade.py --status
tail -f logs/paper_evidence_upgrade_p1.out
```

The queue skips every job that has already finished, so after a crash, a reboot or Ctrl-C, re-run the same line. Failed jobs are listed at the end. Their logs are in `logs/paper_evidence_upgrade/<job>.log` (the sweep was `sweeps/v20.py` until 2026-10-02; the logs of the runs before that are in `logs/v20/`).

**Running two jobs at once.** One job uses about 5 % of the 16 GB card, so the GPU can take two at a time. Start the two queues in different stages, for example `python run_experiments.py sweeps/paper_evidence_upgrade.py headline` and `python run_experiments.py sweeps/paper_evidence_upgrade.py baselines medmamba controls`. Wall clock per run will rise by about 1.5–2×.

**Estimated time.** These figures come from the timings of the existing runs, taken one job at a time:

| Stage | Jobs | Hours |
| --- | --- | ---: |
| `prep` (G1) | 1 prep | 2 (CPU) |
| `evalmode` (G4) | 2 checks | 0.1 |
| `headline` (G2) | 10 runs | 28–60 |
| `heldout` (G3) | 10 evaluations | 3 |
| `probes` (G3) | 2 | 1 (CPU) |
| `baselines` (G5) | 10 | 3 |
| `medmamba` (G5) | 3 | 19 |
| `controls` (G6) | 6 | 12 |
| `cv` (G10) | 5 preps + 10 runs | 38–60 |
| **P1 total** | | **≈ 105–160** |
| `ablations` (G7) | 6 | 21 |
| `zeroshot` (G8) | 2 evaluations | 1 |
| `bands`, `depth`, `hier` (G9) | 7 | 20 |
| `lusc-prep`, `lusc` (G11) | 1 extract + 15 runs | 25 |
| **P2 total** | | **≈ 65** |

---

## 2. The same jobs as literal commands

Use these if you prefer to run jobs by hand. They are exactly what the queue executes.

```bash
REF="--batch_size 256 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986"
BUILD=data/hsi_v9-trainsel
RAW_BC=/home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/
```

### G1 · Rebuild with training-only band selection (≈ 2 h, CPU)

```bash
python prepare_histologyhsi_bc_trainsel.py --root $RAW_BC --out_dir ./$BUILD \
    --split 80_10_10 --split_strategy stratified \
    --split_file data/hsi_v8-80_10_10_importance-new/split_assignment.json \
    --modality both --label_source tissue --patch_size 11 --stride 11 --roi_min_frac 0.8 \
    --band_selection importance --num_bands 32 --band_min_gap 8 --band_max_corr 0.95 \
    --band_min_coverage 0.30 --capture_gain median_ratio --hsi_value_scale auto \
    --rgb_source synthetic --seed 42 --num_workers 8 --verify_level deep
python scripts/export_val_subset_indices.py --data_dir $BUILD/hsi --seeds 1 7 13 23 42
```

**Check:** `cat $BUILD/pass1_scope_report.json` must list 35 training patients and `n_heldout_captures` = 145. `selected_wavelengths.npy` holds the new 32 bands, which will probably differ from the v8 bands.

**Closes:** the band-selection leak. The patient split is the v8 one (reused through `--split_file`), so only band selection and the gain reference change.

### G4 · Is the MedMamba-SS collapse an eval-mode bug? (5 min)

```bash
python scripts/check_eval_mode.py --run_dir experiments/20260915_083810_*arch-split
python scripts/check_eval_mode.py --run_dir experiments/20260915_152009_*arch-fullchannel
```

This prints a verdict and writes `<run>/eval_mode_check_v20.json`. **If the verdict is `EVAL-MODE DEFECT`, tell me before running `hier`**: the hierarchical model then needs a fix (plan G12), not a re-run.

### G2 · Headline pair, 5 seeds (28–60 h)

```bash
for S in 1 7 13 23 42; do for M in hsi rgb; do
  python train.py --profile paper_recipe --data_dir $BUILD/$M $REF --epochs 20 --seed $S --run_tag v20-head-s$S
done; done
```

### G3 · Held-out evaluation and probes (≈ 4 h)

```bash
for S in 1 7 13 23 42; do for M in hsi rgb; do
  D=$(python run_experiments.py --find v20-head-s$S --data $BUILD/$M)
  X=""; [ $S = 42 ] && X="--xai"
  python scripts/heldout_eval.py --run_dirs $D $X
done; done
python scripts/shallow_probe_heldout.py --data_dir $BUILD/hsi
python scripts/shallow_probe_heldout.py --data_dir $BUILD/rgb
```

### G5 · Baselines at matched seeds (≈ 22 h)

```bash
for A in hybridsn spectralformer; do for S in 1 7 13 23 42; do
  python scripts/train_hsi_baseline.py --arch $A --data_dir $BUILD/hsi --seed $S --run_tag v20-s$S
done; done

cd ../medmamba-original/MedMamba
for S in 1 7 42; do
  python train_hsi_v20.py --data $OLDPWD/$BUILD/hsi --epochs 5 --batch_size 256 --lr 1e-4 --amp bf16 \
      --class_weights --normalize global_zscore --norm_stats_sample_cap 5000 \
      --checkpoint_metric f1_macro --patch_size 1 --dims 64,128,256,512 --depths 1,1,2,1 \
      --seed $S --run_name v20_s$S --val_subset_indices $OLDPWD/$BUILD/hsi/val_subset_s$S.npy
done
cd -
```

`train_hsi_v20.py` selects MedMamba's checkpoint on the same 32,985-patch validation subset the other networks use, and also saves full-validation predictions. This removes the selection asymmetry disclosed in v17.

### G6 · Does recursion matter? Non-recursive control (≈ 12 h)

```bash
for S in 1 7 42; do
  python train.py --profile paper_recipe --data_dir $BUILD/hsi $REF --epochs 12 --seed $S --run_tag v20-abl-base-s$S
  python train.py --profile paper_recipe --data_dir $BUILD/hsi $REF --epochs 12 --seed $S --run_tag v20-k2-s$S \
      --trm_n_latent 1 --trm_n_improve 1 --trm_deep_supervision_steps 1
done
```

**K = 2 core applications instead of 63, with the same 446,409 parameters and the same recipe.** The model asserts n ≥ 1, so K = 2 is the smallest recursion depth the code allows.

### G10 · Patient-level 5-fold cross-validation (38–60 h)

```bash
for F in 0 1 2 3 4; do
  CV=data/hsi_v9-cv5-f$F
  python prepare_histologyhsi_bc_trainsel.py --root $RAW_BC --out_dir ./$CV \
      --kfold 5 --fold $F --cv_val_frac 0.125 \
      --modality both --label_source tissue --patch_size 11 --stride 11 --roi_min_frac 0.8 \
      --band_selection importance --num_bands 32 --band_min_gap 8 --band_max_corr 0.95 \
      --band_min_coverage 0.30 --capture_gain median_ratio --hsi_value_scale auto \
      --rgb_source synthetic --seed 42 --num_workers 8 --verify_level deep
  for M in hsi rgb; do
    python train.py --profile paper_recipe --data_dir $CV/$M $REF --epochs 20 --seed 42 --run_tag v20-cv-f$F
  done
  rm -f $CV/hsi/X_*.npy $CV/rgb/X_*.npy      # optional; keeps y/groups for the analysis
done
```

**Checked in the sandbox (dry run, 2026-09-26):** every one of the 45 patients is tested exactly once, and each fold's test set holds 8–11 patients. All seven DCIS patients are tested: fold 0 tests 107 and 136, fold 1 tests 197 and 45, then 25, 152 and 85. Class coverage passes in every fold. Each fold selects its bands on its own training patients. The recipe is frozen, so no fold's test patients influenced any choice.

### G7 · Ablations at three seeds (≈ 21 h)

```bash
for S in 1 7 42; do
  python train.py --profile paper_recipe --data_dir $BUILD/hsi --batch_size 256 --train_subsample_frac 0.102 \
      --val_subsample_frac 0.0986 --epochs 12 --seed $S --recon_mode none --run_tag v20-recon-off-s$S
  python train.py --profile paper_recipe --data_dir $BUILD/hsi $REF --epochs 12 --seed $S --no_use_wavelengths \
      --run_tag v20-enc-index-s$S
done
```

### G8 · Zero-shot band removal, with η free and with η fixed (≈ 1 h)

```bash
D=$(python run_experiments.py --find v20-head-s42 --data $BUILD/hsi)
python scripts/eval_band_decimation.py            --run_dir $D --keep_bands 32,16,8,4,2
python scripts/eval_band_decimation_fixed_eta.py  --run_dir $D --keep_bands 32,16,8,4,2
```

The second script holds η = 32 and the 32-band wavelength range fixed. Both scripts print an `[identity]` check at C = 32.

### G9 · Band count, depth and hierarchy on the new build (≈ 20 h)

```bash
for N in 16 8; do
  python scripts/make_band_subset_build.py --src $BUILD/hsi --keep $N --out data/hsi_v9-bands$N
  python train.py --profile paper_recipe --data_dir data/hsi_v9-bands$N $REF --epochs 12 --seed 42 --run_tag v20-bands$N
done
for T in 1 2 4; do
  python train.py --profile paper_recipe --data_dir $BUILD/hsi $REF --epochs 12 --seed 42 --trm_n_improve $T --run_tag v20-depth-n$T
done
python train.py --profile paper_recipe --data_dir $BUILD/hsi $REF --epochs 12 --seed 42 --architecture split       --run_tag v20-arch-split
python train.py --profile paper_recipe --data_dir $BUILD/hsi $REF --epochs 12 --seed 42 --architecture fullchannel --run_tag v20-arch-fullchannel
```

### G11 · Second hyperspectral histology dataset: HMI-LUSC (≈ 25 h)

This dataset has 10 patients and 61 bands (450–750 nm, a different sensor), with pixel-level tumour masks. Evaluation is patient-level 5-fold CV: each fold tests 2 patients.

```bash
RAW_LUSC="/home/dante/Downloads/downloads/HMI-LUSC_ A Histological Hyperspectral Imaging Dataset for Lung Squamous Cell Carcinoma - 30188080"
LUSC_ROOT=/data/dante_data/hmi_lusc_raw          # /home has no room for the 49 GB of cubes
mkdir -p $LUSC_ROOT
for z in "$RAW_LUSC"/P*.zip; do p=$(basename "$z" .zip)
  if [ -d "$RAW_LUSC/$p/LUSC_ROI_1" ]; then ln -sfn "$RAW_LUSC/$p" $LUSC_ROOT/$p; else unzip -q "$z" -d $LUSC_ROOT/$p; fi
done
python prepare_hmi_lusc_v20.py extract --root $LUSC_ROOT --pool data/lusc_v20/pool --workers 2
for F in 0 1 2 3 4; do
  python prepare_hmi_lusc_v20.py split --pool data/lusc_v20/pool --out data/lusc_v20-f$F --fold $F
  D=data/lusc_v20-f$F/hsi
  python train.py --profile paper_recipe --data_dir $D --batch_size 256 --lambda_sam 0.1 --train_subsample_frac 1.0 \
      --val_subsample_frac 0.25 --epochs 20 --seed 42 --run_tag v20-lusc-f$F
  for A in hybridsn spectralformer; do
    python scripts/train_hsi_baseline.py --arch $A --data_dir $D --seed 42 --train_subsample_frac 1.0 \
        --val_subsample_frac 0.25 --run_tag v20-lusc-f$F
  done
done
```

**Check after `extract`:** `data/lusc_v20/pool/pool_report.json` lists every ROI. The patch counts should be several thousand per ROI, and `glass_dropped` shows how many empty-slide patches were removed. The model runs **unchanged** on 61 bands: same configuration, and a backbone with the same parameter count as at 32 bands. Only the classifier shrinks, from 3 classes to 2 (446,409 − 129 = 446,280 parameters).

---

## 3. After the runs

```bash
python scripts/analysis_v20.py          # writes paper/source/v20_analysis.json, prints a summary
python run_experiments.py sweeps/paper_evidence_upgrade.py --status        # lists anything still missing
```

Then tell me the stages are done (or paste `python run_experiments.py sweeps/paper_evidence_upgrade.py --status`). I will read `paper/source/v20_analysis.json` and the run directories, write the numbers into manuscript v18 (plan S5) and re-rate it (S6). Nothing needs to be copied by hand.
