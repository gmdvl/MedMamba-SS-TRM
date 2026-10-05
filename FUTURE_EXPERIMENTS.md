# FUTURE_EXPERIMENTS.md — experiments deferred from manuscript v18

**Written:** 2026-09-26, under a 7-hour paper deadline.

**Rule applied:** an experiment was deferred if it could not finish within 2 hours on the available GPU (one RTX 5060 Ti 16 GB).

**Command sources:**
- Arguments marked *verified* were checked against the argument parser, or against the `config.json` of an existing run using the same entry point.
- `documentations/16_v20_commands.md` and `sweeps/paper_evidence_upgrade.py` hold the same commands in runnable form (`python run_experiments.py sweeps/paper_evidence_upgrade.py <stage>`). The runner skips finished jobs.
- Run everything from the repository root on the GPU host.

---

## Measured runtimes used for triage

| Job | Measured | Source |
| --- | --- | --- |
| MedMamba-SS-TRM, 20 epochs, v8 build | 7.38 h (32-band), 4.89 h (3-band) | `history.json` of `20260915_031356`, `20260915_031947` |
| MedMamba-SS-TRM, 12 epochs, K = 21 / 42 / 63 | 2.41 / 3.31 / 5.51 h | `history.json` of `20260915_123136`, `_150351`, `_031555` |
| MedMamba-SS-TRM, new build, epochs 1–4 (run in parallel with an HMI-LUSC job) | 1,123–1,620 s per epoch | `experiments/20260926_125011_*/history.json` |
| MedMamba-SS-TRM on HMI-LUSC fold 0 (run in parallel) | 3.2 s/it × 1,008 it ≈ 54 min per epoch | `logs/v20/lusc-f0-trm.log` |
| MedMamba baseline, 5 epochs | 6.26 h | `../medmamba-original/MedMamba/runs_hsi/matched_natural_v8new/training_history.json` |
| HybridSN / SpectralFormer, 20 epochs | 24 min / 6 min | file timestamps of `20260925_181908_*`, `20260925_184306_*` |
| Zero-shot band removal, 5 band counts | ≈ 50 min | timestamps of `20260915_031356_*/band_decimation/` |
| Preparation of `data/hsi_v9-trainsel` (done) | 12:10–12:49, 39 min | `logs/v20_p1.out` |

**Already done, and therefore not future work:**
- `data/hsi_v9-trainsel`: band selection and gain reference on training patients only (`pass1_scope_report.json`).
- HMI-LUSC patch pool and five folds: `data/lusc_v20/pool/pool_report.json`, 369,007 patches (186,000 tumour, 183,007 non-tumour).
- The MedMamba-SS evaluation-mode diagnostic: `experiments/20260915_083810_*/eval_mode_check_v20.json` and `experiments/20260915_152009_*/eval_mode_check_v20.json`.

---

## F1 · Headline pair on the training-only band-selection build

**Purpose.** Replace every hyperspectral result of v18 with one whose input bands were chosen without held-out labels.

**Why deferred.** It needs 10 runs of 20 epochs. At the measured 4.9–7.4 h per run, that is about 60 h, far beyond the deadline. One seed-1 run was started and reached epoch 4, which is not usable.

**Priority: Critical.** Every hyperspectral number of v18 carries the band-selection caveat (v18 Sec. III-A). A training-only re-selection (done) moved every band by at most 5.1 nm, which bounds the change in the input. Only retraining can show the change in the results.

**Prerequisites.**
- `data/hsi_v9-trainsel/{hsi,rgb}`: exists.
- `train_example_v18.py`.
- GPU with torch cu128.
- About 60 GPU-hours.

**Exact commands** (every argument verified against `experiments/20260915_031555_*/config.json` and the parser):

```bash
for S in 1 7 13 23 42; do for M in hsi rgb; do
  python train.py --profile paper_recipe --data_dir data/hsi_v9-trainsel/$M --batch_size 256 --epochs 20 --lambda_sam 0.1 \
      --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed $S --run_tag v20-head-s$S
done; done
# or: python run_experiments.py sweeps/paper_evidence_upgrade.py headline
```

**Configuration.**

| Setting | Value |
| --- | --- |
| Architecture | recursive, width 128, 2 core blocks |
| Recursion | n = 6, T = 3, N_sup = 3 (K = 63) |
| Mixer | mlp |
| Optimizer | AdamW, lr 3e-4, weight decay 0.05 |
| Schedule | cosine, 587 warmup steps, 19,560 total |
| Precision, batch | bf16, 256 |
| Loss | focal (γ 1.5, class-weight exponent 0.75) + reconstruction (λ_mse = λ_sam = 0.1) |
| EMA | 0.9995 |
| Data | 11 × 11 patches, 32 bands (`hsi`) or 3 (`rgb`) |
| Seeds | 1, 7, 13, 23, 42 |

These are the defaults of `train_example_v18.py`, recorded in the reference run's `config.json`.

**Expected outputs.** `experiments/<timestamp>_hsi_v9-trainsel-{hsi,rgb}_recursive_..._v20-head-s{S}/` containing `config.json`, `history.json`, `best_model.pt`, `test_report.json` and `test_predictions.npz`.

**Metrics to record.**
- Test and full-validation balanced accuracy, macro-F1, accuracy, κ, ECE and macro ROC-AUC.
- Per-patient macro recall.
- The HSI − RGB difference per seed.
- Patient-bootstrap intervals, computed by `scripts/analysis_v20.py`.

**Validation.**
- `config.json` must show `data_dir` = `data/hsi_v9-trainsel/...`, `lambda_sam` = 0.1 and `scheduler.total_steps` = 19560.
- `backbone_num_params` must be 446409.
- All gates in `gates.json` must be true.

**Paper section.** Abstract, VI-A, VI-F, VII, VIII. It also removes the band-selection caveat from III-A and VII-E.

**Scientific question.** Do the test-set ranking and the HSI-vs-RGB pattern survive when no held-out label influences the input bands?

**Reproducibility checklist.**
- Build hash in `data/hsi_v9-trainsel/hsi/dataset_manifest.json`.
- Seed list as above.
- Entry point `train_example_v18.py`.
- Run tags `v20-head-s{S}`.
- Analysis `scripts/analysis_v20.py` → `paper/source/v20_analysis.json`.

---

## F2 · Held-out evaluation and probes for F1

**Purpose.** Full-validation predictions (so all ten held-out patients can be analysed) and linear probes on the new build.

**Why deferred.** It depends on F1.

**Priority: Critical** (for F1's pooled and per-patient numbers).

**Commands** (script arguments verified in the source):

```bash
for S in 1 7 13 23 42; do for M in hsi rgb; do
  D=$(python run_experiments.py --find v20-head-s$S --data data/hsi_v9-trainsel/$M)
  X=""; [ $S = 42 ] && X="--xai"
  python scripts/heldout_eval.py --run_dirs $D $X
done; done
python scripts/shallow_probe_heldout.py --data_dir data/hsi_v9-trainsel/hsi
python scripts/shallow_probe_heldout.py --data_dir data/hsi_v9-trainsel/rgb
```

**Outputs.**
- `<run>/heldout_v19/validation/test_predictions.npz`
- `<run>/heldout_v19/xai_test.npz` (seed 42)
- `data/hsi_v9-trainsel/{hsi,rgb}/shallow_probe_heldout_v19*.{json,npz}`

**Validation.** The validation labels in the npz must equal `y_val.npy`, which `analysis_v20.py` asserts through its metrics.

**Paper section.** VI-A (Table X), VI-F, VI-G, VI-H.

---

## F3 · Patient-level five-fold cross-validation (breast)

**Purpose.** An evaluation whose test patients took no part in recipe development, covering all 45 patients and all 7 DCIS patients.

**Why deferred.**
- It needs 5 preparations (≈ 0.7–2 h each) and 10 runs of 20 epochs (≈ 5–7 h each), about 40–70 GPU-hours.
- It needs about 27 GB of disk per fold.

**Priority: Critical.**
- The fixed split's five test patients were consulted during recipe development (v18 Sec. V-C).
- Each fixed held-out set contains DCIS from one patient.
- The fold assignment was dry-run on 2026-09-26: every patient is tested once, and all 7 DCIS patients are tested (fold 0: 107, 136; fold 1: 197, 45; fold 2: 25; fold 3: 152; fold 4: 85). Class coverage passes in every fold.

**Prerequisites.**
- Raw data at `/home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/`.
- `prepare_histologyhsi_bc_trainsel.py`, which requires the `spectral` package on the host.
- About 135 GB of disk, or about 27 GB with per-fold cleanup.

**Commands** (the preparation arguments are verified against `data/hsi_v8-80_10_10_importance-new/prep_log.txt`; `--kfold`, `--fold` and `--cv_val_frac` against `training/prep/core.py`):

```bash
for F in 0 1 2 3 4; do
  CV=data/hsi_v9-cv5-f$F
  python prepare_histologyhsi_bc_trainsel.py --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
      --out_dir ./$CV --kfold 5 --fold $F --cv_val_frac 0.125 \
      --modality both --label_source tissue --patch_size 11 --stride 11 --roi_min_frac 0.8 \
      --band_selection importance --num_bands 32 --band_min_gap 8 --band_max_corr 0.95 \
      --band_min_coverage 0.30 --capture_gain median_ratio --hsi_value_scale auto \
      --rgb_source synthetic --seed 42 --num_workers 8 --verify_level deep
  for M in hsi rgb; do
    python train.py --profile paper_recipe --data_dir $CV/$M --batch_size 256 --epochs 20 --lambda_sam 0.1 \
        --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --run_tag v20-cv-f$F
  done
done
# or: CV_CLEAN=1 python run_experiments.py sweeps/paper_evidence_upgrade.py cv
```

**Configuration.** As F1, with seed 42. **The recipe must not change once the first fold starts.**

**Metrics.**
- Pooled balanced accuracy and macro-F1 over all 45 patients, and over the 35 patients that were training patients throughout development.
- The HSI − RGB difference per fold.
- Per-patient macro recall.
- A patient bootstrap. All of these are computed by `analysis_v20.py → cross_validation`.

**Validation.** For each fold, `split_assignment.json` has disjoint groups, `pass1_scope_report.json` lists only that fold's training patients, and both runs have `test_predictions.npz`.

**Paper section.** New subsection "Patient-level cross-validation" in Results, the Abstract and the Conclusion. Remove the test-exposure caveat once it is done.

**Scientific question.** How does MedMamba-SS-TRM perform, and how does 32-band compare with 3-band input, when every patient is tested exactly once by a frozen recipe?

---

## F4 · MedMamba baseline with matched checkpoint selection, 3 seeds

**Purpose.** Remove the asymmetry disclosed in v18 Sec. V-D: MedMamba selected on the full validation split it is scored on.

**Why deferred.** It needs 3 runs × 6.26 h (measured).

**Priority: High.** MedMamba is the base architecture and the strongest single-run competitor on test accuracy and κ.

**Commands.** `train_hsi_v20.py` wraps the other repository's `train_hsi.py`, whose arguments are verified in its `create_parser()`:

```bash
python scripts/export_val_subset_indices.py --data_dir data/hsi_v8-80_10_10_importance-new/hsi --seeds 1 7 42
cd ../medmamba-original/MedMamba
for S in 1 7 42; do
  python train_hsi_v20.py --data $OLDPWD/data/hsi_v8-80_10_10_importance-new/hsi --epochs 5 --batch_size 256 \
      --lr 1e-4 --amp bf16 --class_weights --normalize global_zscore --norm_stats_sample_cap 5000 \
      --checkpoint_metric f1_macro --patch_size 1 --dims 64,128,256,512 --depths 1,1,2,1 --seed $S \
      --run_name v18_s$S --val_subset_indices $OLDPWD/data/hsi_v8-80_10_10_importance-new/hsi/val_subset_s$S.npy
done
cd -
```

This is shown for the v8 build, to match v18. For the new build, replace the data path with `data/hsi_v9-trainsel/hsi` and the run name with `v20_s$S`.

**Outputs.** `runs_hsi/v18_s{S}/{predictions_test.npz, predictions_val_full.npz, val_full_metrics.json, experiment_report.json}`.

**Validation.**
- The log prints `[v20] checkpoint selection on 32985 of 334516 validation patches`.
- `backbone_num_params` = 3648995.

**Paper section.** VI-A, Table X and Table XVII; V-D.

---

## F5 · Non-recursive control at more seeds

**Purpose.** Isolate what recursion adds: the same 446,409 parameters with K = 2 instead of 63 core applications.

**Why deferred.** Seed 42 was started in the ≤ 2 h set on 2026-09-26 and stopped after 3 minutes: it measured 1.12 s/it while two other jobs shared the GPU, so 12 epochs need about 3.7 h. Why the step time stays this high with 2 instead of 63 core applications was not measured; a profile of one step would answer it. Seeds 1 and 7 also need their own 12-epoch baselines (5.5 h each, measured at K = 63).

**Priority: High.** If only seed 42 exists, the control is single-seed, and single-seed ablation effects in this study are smaller than the seed spread.

**Commands** (arguments verified against the config of `20260915_031555_*`):

```bash
# seed 42 needs only the K = 2 run (its baseline is experiments/20260915_031555_*abl-base); seeds 1 and 7 need both
for S in 42 1 7; do
  [ $S != 42 ] && python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 12 \
      --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed $S --run_tag abl-base-s$S
  python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 12 \
      --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed $S \
      --trm_n_latent 1 --trm_n_improve 1 --trm_deep_supervision_steps 1 --run_tag k2-control-s$S
done
```

**Validation.**
- `scheduler.total_steps` = 11736.
- `backbone_num_params` = 446409.
- `test_report.json["flops"]` ≈ 0.173 + 2 × 0.0957 GFLOPs, by the cost model (eq. 23).

**Metrics.** Test balanced accuracy, macro-F1 and DCIS F1; the difference to the same-seed baseline.

**Paper section.** VI-D (ablations), VII (recursion discussion).

---

## F6 · Component ablations at three seeds

**Purpose.** Replicate the reconstruction and wavelength-encoding ablations, which are single-seed in v18 and within the seed spread.

**Why deferred.** 6 runs × about 5.5 h.

**Priority: Medium.** It affects secondary claims only.

**Commands** (verified: `--recon_mode none` and `--no_use_wavelengths` exist in the parser, and both were used by the runs `20260915_160442` and `20260915_202942`):

```bash
for S in 1 7 42; do
  python train.py --profile paper_recipe --data_dir data/hsi_v9-trainsel/hsi --batch_size 256 --train_subsample_frac 0.102 \
      --val_subsample_frac 0.0986 --epochs 12 --seed $S --recon_mode none --run_tag v20-recon-off-s$S
  python train.py --profile paper_recipe --data_dir data/hsi_v9-trainsel/hsi --batch_size 256 --lambda_sam 0.1 \
      --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --epochs 12 --seed $S --no_use_wavelengths \
      --run_tag v20-enc-index-s$S
done
```

These runs failed on 2026-09-26 only because they started before `data/hsi_v9-trainsel` existed ("Missing required file ... X_train.npy"). The build exists now.

**Paper section.** VI-D, Table XI.

---

## F7 · Band-count, depth and hierarchy runs on the new build

**Purpose.** Put the remaining ablations on the same build as F1.

**Why deferred.** 7 runs × 2.4–5.8 h.

**Priority: Medium.**

**Command.** `python run_experiments.py sweeps/paper_evidence_upgrade.py bands depth hier`. The literal commands are in `documentations/16_v20_commands.md` §G9, using `scripts/make_band_subset_build.py` and `--trm_n_improve` / `--architecture` (verified).

**Paper section.** VI-C, VI-D, VI-E.

---

## F8 · Second hyperspectral histology dataset: HMI-LUSC

**Purpose.** Test the band-count-agnostic design on a second sensor: 61 bands, 450–750 nm, lung squamous cell carcinoma, 10 patients, pixel tumour masks [Yan et al., *Sci. Data* 13:415, 2026].

**Why deferred.** Fold 0 alone ran at about 54 min per epoch in parallel with another job. 5 folds × 20 epochs plus 10 baseline runs does not fit. The fold-0 run stopped at epoch 4.

**Priority: High.**
- It gives the paper a second hyperspectral dataset.
- The dataset's own published baselines were not patient-disjoint.

**Prerequisites.** All present:
- `data/lusc_v20-f{0..4}/hsi`, prepared by `prepare_hmi_lusc_v20.py`: 90 % mask purity, glass filter at reflectance 0.85, at most 3,000 patches per class per image.

**Commands** (the `train_example_v18.py` arguments are verified as in F1; the `train_hsi_baseline.py` arguments against its parser):

```bash
for F in 0 1 2 3 4; do
  D=data/lusc_v20-f$F/hsi
  python train.py --profile paper_recipe --data_dir $D --batch_size 256 --lambda_sam 0.1 --train_subsample_frac 1.0 \
      --val_subsample_frac 0.25 --epochs 20 --seed 42 --run_tag v20-lusc-f$F
  for A in hybridsn spectralformer; do
    python scripts/train_hsi_baseline.py --arch $A --data_dir $D --seed 42 --train_subsample_frac 1.0 \
        --val_subsample_frac 0.25 --run_tag v20-lusc-f$F
  done
done
```

**Not verified:** whether 20 epochs over the full training fold is a good budget for this dataset. It was chosen to keep about 250 k patches per epoch, as for the breast data. Consider `--train_subsample_frac 0.8` if the time per epoch is prohibitive.

**Metrics.** Pooled balanced accuracy, macro-F1 and ROC-AUC over the 10 patients; per-patient recall; per-fold mean ± s.d. All come from `analysis_v20.py → lusc_cv`.

**Paper section.** Datasets, Results (a new subsection), Discussion.

---

## F9 · MedMamba-SS with batch-independent normalization

**Purpose.** The eval-mode diagnostic (done today) shows that the standard MedMamba-SS collapses to IDC in eval mode because the BatchNorm running statistics of its convolutional branch do not match the test batches (mean shift up to 3.8 s.d., variance ratio up to 4.2). With batch statistics it reaches only 0.490 balanced accuracy on a 9,000-patch stratified sample. The follow-up question is whether a normalization that does not depend on batch statistics trains better.

**Why deferred.** It needs a code change: `GBlock.conv_branch` in `medmamba_ss_trm.py` hard-codes `nn.BatchNorm2d`, and `medmamba_ss_trm.py` is frozen, so it needs a new subclass plus a new entry-point option. It also needs a 12-epoch run of about 5.8 h (measured for `arch-split`).

**Priority: Medium.** It bears on Contribution 1, but MedMamba-SS-TRM's results do not depend on it.

**Command:** COMMAND NOT VERIFIED — the code change is required first (new subclass of `GBlock` replacing `nn.BatchNorm2d` with `nn.GroupNorm`, or with a `LayerNorm` over channels, and an entry-point flag to select it).

**Paper section.** VI-D, VII-B.

---

## F10 · Remaining baseline seeds — **DONE 2026-09-26** (all four seeds of both baselines; see `paper/source/v18_short_analysis.json`). Kept for reproducibility

**Purpose.** HybridSN and SpectralFormer at seeds 1, 7, 13 and 23 on the v8 build, paired with MedMamba-SS-TRM's five seeds.

**Command** (arguments equal the `cli_args` of `20260925_181908_*`):

```bash
for S in 1 7 13 23; do for A in spectralformer hybridsn; do
  python scripts/train_hsi_baseline.py --arch $A --data_dir data/hsi_v8-80_10_10_importance-new/hsi --seed $S --run_tag s$S
done; done
python scripts/analysis_v18_short.py
```

**Priority: High.** It turns the single-run baseline comparison into a seed-paired one.

---

## F11 · Zero-shot band removal with fixed η — **DONE 2026-09-26 17:58** at 32/16/8 bands (in the paper). Remaining: C = 4 and 2, and other checkpoints/seeds

Run it **alone** on the host. It loads the 2.7 GB test split into RAM; next to a training job and the desktop session it pushed swap toward the `systemd-oomd` threshold.

```bash
python scripts/eval_band_decimation_fixed_eta.py \
    --run_dir experiments/20260915_031356_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_ref20 \
    --keep_bands 32,16,8,4,2
```

**Validation.** At C = 32 it must print `[identity] ... OK`.

**Paper section.** VI-C.

---

## F12 · Longer-term items (code not written, so commands are not verified)

| Item | Why | Command |
| --- | --- | --- |
| Band-agnostic baseline (ChannelViT- or DOFA-style embedding in place of the spectral pathway) | Compare against the closest designs, not only by design | COMMAND NOT VERIFIED — model code required |
| Training on random band subsets with the encoding fixed to a reference range | Test a remedy for the zero-shot failure | COMMAND NOT VERIFIED — sampler code required (the existing `band_dropout_prob` zeroes bands and does not remove them, so it is not equivalent) |
| SS2D mixer inside the core with a fused scan kernel | Give the recursive core a spatial state-space mixer | `--trm_mixer ss2d` exists (verified), but it measured 91.8 h per epoch in the benchmark configuration; a fused kernel is required first |
| ROI-masked breast build | Remove capture-level label noise | COMMAND NOT VERIFIED — the ROI lookup expects per-capture files, while the annotations are per patient |
