> **Tracking:** epic [#1](https://github.com/gmdvl/MedMamba-SS-TRM/issues/1) (issues #2–#40). Every work issue names its branch, `aug/<id>-<slug>`, cut from and merged back into the integration branch `feature/hsi-augmentation-pipeline`.
> **Status:** approved 2026-10-26; implementation in progress. The issue bodies are the per-task specification; this file is the overall design.

# HSI Augmentation Preprocessing Pipeline: Implementation Plan

*Written by the panel: AI/ML research (validity, leakage, statistics), ML engineering (data flow, GPU/IO, reproducibility) and Python development (structure, tests, freezing rule). The draft was reviewed by an independent design pass; its corrections are folded in.*

---

## Context

DCIS is the minority class of HistologyHSI-BC. It makes up **5.0% of training patches (122,850 of 2,452,086), and those patches come from only 5 of the 35 training patients.** Healthy has 30 training patients and IDC has 34.

An oversampling and cross-patient-mixing feature is already written, uncommitted, on branch `feature/minority-class-augmentation`. It has been tested on CPU only and has never run on a GPU.

The thesis needs a modular, reproducible pipeline that does the following:

- validates the data;
- preprocesses without leakage;
- provides conventional spectral–spatial augmentation;
- trains a class-conditional VAE and a class-conditional latent diffusion model;
- quality-controls the synthetic data;
- exports synthetic data separately from real data;
- compares real-only training against each technique and their combination, under matched conditions.

**Expected outcome**
- A pipeline package and a CLI.
- Quality-controlled synthetic DCIS exports, plugged into `train.py` by one flag.
- A pre-registered 2×2 comparison: copy vs generate, crossed with mixing off vs on.
- A reproducible report.

### Decisions locked with the user (2026-10-26)

| Topic | Decision |
|---|---|
| Real-data input | `data/hsi_v9-trainsel/hsi`. Pass 1 (band selection, gain, value scale) was fitted on training captures only. It uses the same patient split as the paper build. |
| Background (glass) | The mask is used **only** to pick real patches for generator training and to QC synthetic samples. The classifier's real data is unchanged. |
| Dataset scope | HistologyHSI-BC only, behind a thin adapter. |
| GPU budget | **Screen:** 5 new arms × seed 42 × 12 epochs, about 10 GPU-h. **Seeds:** A0 + 2 arms × 5 seeds × 20 epochs. |
| Which arms go on to the seeds stage | **Best per family:** the best generative arm plus the best non-generative arm. Score = mean val macro-F1 over epochs 10–12. Tie-breaks: val DCIS F1, then the cheaper method. |
| Physics ops | Implemented and tested. Evaluated in an **optional arm A1p** (+~2 GPU-h). |
| Glass rows in copy arms | **Pre-registered rule:** measure the DCIS glass fraction in Phase 2. Above 10%, copy arms duplicate only tissue DCIS rows. At 10% or below, they copy all rows and the confound is reported. |

---

## Ground truth the plan relies on (verified 2026-10-26)

| Fact | Value / location |
|---|---|
| Arrays | `X_*` is float16, **NHWC**. Train / val / test sizes are 2,452,086 / 334,516 / 348,894, each 11×11×32. `X_train` is **19 GB**: always memmap it and read in chunks. |
| Sidecars | `y_*` is int64 (0 healthy, 1 DCIS, 2 IDC). `groups_*` is int32 patient id. `captures_*` is int32. `capture_index.json`. `wavelengths.npy` has 32 non-uniform bands with a gap (trainsel: 628.9–850.9 nm). `dataset_manifest.json` has `sha256: true`. Also `split_assignment.json` and `pass1_scope_report.json`. |
| Class counts (train / val / test) | healthy 722,358 / 98,280 / 83,538 · DCIS 122,850 / 24,570 / 24,570 · IDC 1,606,878 / 211,666 / 240,786 |
| DCIS patients | Train: 25, 45, 85, 107, 152. Val: **197**. Test: **136**. That is one DCIS patient per held-out split. |
| Labels | Capture-level weak labels. ROI polygons are in WSI coordinates and unusable, so **no build has a background filter.** |
| Coordinates | Not stored. All **499 train captures hold 4,914 patches**, which is a full 54×91 row-major grid, so `(r, c) = divmod(ordinal, 91)`. |
| Values | **Raw sensor counts, not reflectance**: per-capture gain × raw ÷ corpus median (**8640.0** on trainsel). The 0.85 glass threshold in `prepare_hmi_lusc_v20.py` does not transfer. |
| Normalization | Online `global_zscore`. Stats come from `compute_global_channel_stats(real X_train path)` (`training/npy_data.py:36-51`, pinned), called at `training/train_pipeline.py:339-340`. They are never persisted. |
| `paper_recipe` | `recon_mode latent`, so items are `(x, y, norm_stats)` 3-tuples. Its standard augmentation is flips .5, rot90 .5, crop .3, scale .15, offset .05, noise .02. Weights are `focal_weighted` with power 0.75. |
| A0 is already defined | `sweeps/paper_evidence_upgrade.py`: `v20-abl-base-s42` (12 epochs) and `v20-head-s{1,7,13,23,42}` (20 epochs) on `data/hsi_v9-trainsel/hsi` with REF (frac 0.102 / 0.0986). Neither has finished yet; the paper needs both anyway. |
| Shallow signal | A mean/std per-band logistic regression reaches test balanced accuracy 0.903 (v8 probe). The class signal is mostly per-band 1st/2nd-order statistics, i.e. patient spectral signatures. |
| Reusable code | `training/gan.SAMLoss` (frozen; import only), `spectral_recon_metrics_v2.sam_deg/sid`, `recon_artifacts.ssim2d / resolve_rgb_bands / rgb_composite / render_contact_sheet`, `hsi_baselines.build_baseline`, `scripts/shallow_probe_heldout.fit`, `scripts/shallow_baseline.featurize` (frozen; import only), `heldout_metrics.per_patient`, `npy_atomic.{save_npy_atomic, save_json_atomic, write_dataset_manifest, verify_against_manifest}`, `dataset_audit`, `dataset_integrity.check_patch_leakage`, `system_memory.get_ram_info`, `dataloader_config.seed_everything`. **There is no VAE, diffusion, FID/MMD or kNN code.** |
| Environment | torch 2.11 cu128, sklearn 1.9, scipy, PyYAML, scikit-image, matplotlib, pandas. **Not installed:** diffusers, umap, omegaconf, faiss, pyarrow, psutil. |
| Hardware | RTX 5060 Ti 16 GB, 32 GB RAM. **`/data` is 98% full, with 88 GB free.** The sandbox has no GPU: only CPU tests run here. |

---

## Non-negotiable engineering rules

1. **Freezing rule.** Never edit:
   - anything in `tests/test_frozen_files_untouched.py:FROZEN_GLOBS`;
   - the pinned definitions in `tests/test_live_copies_match_archive.py`;
   - `prepare_histologyhsi_bc*.py`.

   New behaviour goes in **new files**. New `train.py` flags go only in new functions called at the end of `train_cli._add_flags` (`:564-684`).
2. **The default path stays bit-identical.** `tests/test_train_e2e_parity.py` and `tests/test_run_tag_v18.py` must pass unchanged.
3. **Test is sealed; val is reserved for the classifier.**
   - Generator stages touch **neither** val nor test. Val has one DCIS patient (197), who already drives checkpoint selection and arm selection.
   - All generator tuning uses leave-one-DCIS-patient-out (LOPO) folds *inside train*.
   - Test columns appear only after `selection.json` is frozen.
4. **Only training rows are used for fitting.** This covers:
   - normalization and log-space stats;
   - tissue threshold;
   - feature spaces;
   - QC references and thresholds;
   - QC classifiers;
   - generators.

   P1 may read val/test only for structural checks (counts, NaN, hashes).
5. **Never load a full `X_*`.** Use memmap, 20k-row chunks and float64 accumulators.
6. **Checkpoints use Python floats only**, so they load with `weights_only=True`. This is the root cause of the known `--resume` crash.
7. **QC never depends on the classifier being evaluated.** A0 features may be used for reporting only.

---

## Architecture

```
augment_pipeline.py        # CLI: validate | preprocess | pool | fit-cvae | fit-ae | fit-ldm | generate |
                           #      qc | export | lopo-utility | bench | all   (+ --force, --resume auto)
hsi_augment/               # NEW package; imports frozen code, never edits it
  config.py                dataclass per stage, JSON round-trip, config_hash() over canonical JSON
  stage.py                 stage dir contract: done.json {config_hash, manifest_sha256 of the INPUT
                           build (not a 19 GB re-hash), output sha256s, git sha, timings, status};
                           skip-if-done, --force, stage.log + metrics.jsonl
  io.py                    memmap chunk iterators (sorted fancy index), float64 running stats,
                           SplitGuard (refuses val/test outside P1-structural / final eval;
                           logs every access to split_access_log.jsonl), atomic writes, disk/RAM preflight
  adapters/base.py, adapters/histologyhsi_bc.py   paths, class names, minority class, grid 54x91
  discovery.py             P1
  preprocess.py            P2 (+ provenance index and imbalance report of P4)
  pool.py                  P4 generator pool + LOPO folds
  transforms.py            P5 physics ops + composition helpers
  models/blocks.py, cvae.py, autoencoder.py, diffusion.py      P6/P7
  train_generative.py      shared loop: AMP, EMA, cosine LR, atomic ckpt, exact resume
  generate.py              P8a
  qc/features.py, rules.py, setmetrics.py, memorization.py, consistency.py      P8b
  export.py                P9 (imports the schema from training/synthetic_mix.py)
  lopo_utility.py          P8c train-only utility test
  stats.py                 paired seed deltas, bootstrap with labels=[0,1,2]
  report.py, viz.py        P10 (viz detects the spectral gap from wavelengths; does NOT import
                           scripts/make_figures_v19.py, which loads v8 data at import time)
training/synthetic_mix.py  # NEW, light imports: export schema, load_synthetic_export, attach_synthetic,
                           # plan_minority_with_synthetic
sweeps/augmentation_pipeline.py   # NEW
scripts/augmentation_select.py, scripts/augmentation_report.py   # NEW thin CLIs
tests/hsi_augment_fixtures.py + tests/test_hsi_augment_*.py, tests/test_synthetic_mix.py,
tests/test_augmentation_sweep.py  # NEW
```

**Configuration and seeds**
- Configuration follows the repo convention: argparse flags plus a resolved `config.json`, with optional `--config file.json` defaults (explicit flags win).
- Seeds use `SeedSequence([master, crc32(stage_name)])`, so adding a stage never shifts another stage's seed.
- Torch is seeded through `seed_everything` (import), with `--deterministic` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Sampling noise comes from a CPU `torch.Generator`, so re-running an export reproduces its sha256.

**Output root:** `data/hsi_v9-trainsel-aug/`, which is gitignored. It contains:
- `prep/` and `pool/`;
- `gen/{cvae,ae,ldm}/{lopo-k,final}/`;
- `synth/<source>/`;
- `export/<source>/`;
- `lopo_utility/`.

---

## Phase 0: Branch and hygiene

1. Ask the user to commit the minority-augmentation work on `feature/minority-class-augmentation`. Then branch `feature/hsi-augmentation-pipeline` from it.
2. **Retarget `sweeps/minority_augmentation.py`** so its runs double as this study's copy arms:
   - set `HSI = "data/hsi_v9-trainsel/hsi"`;
   - replace its `abl-base` reference with the shared `v20-abl-base-s42`.

   This file is uncommitted and not frozen.
3. Record the baseline `python -m pytest -q` pass count (~779).
4. **Start A0 on the host now.** It needs no new code: `python run_experiments.py sweeps/paper_evidence_upgrade.py controls headline`. Only the `-hsi` runs are needed here; the paper needs both.

**Acceptance:**
- The suite is green.
- `--dry_run` of the retargeted sweep is valid.

---

## Phase 1: Dataset discovery and validation (`discovery.py`)

1. **Inventory.** Read the class names, wavelengths, manifest, capture index, split assignment and Pass-1 scope report, plus every `.npy` header. Record shapes, dtypes and sizes.
2. **Integrity.**
   - `verify_against_manifest` (streamed sha256).
   - Header shapes must equal the manifest.
   - `len(X)==len(y)==len(groups)==len(captures)` for every split.
   - `check_patch_leakage` across splits, on a sample.
3. **Content scan.** Chunks of 20k rows; float64 accumulators.
   - Counts of NaN, ±Inf, float16 saturation (|x| = 65504) and all-zero patches.
   - Per-band min, max, mean, std and p0.05/p99.95, per split and per class. For val/test this is structural only, never used for fitting.
4. **Provenance consistency.**
   - `y == TISSUE[capture.tissue]`.
   - `groups == capture.patient`.
   - Each row's split equals its capture's split.
   - Per-capture counts equal `n`.
   - Each capture's rows are **contiguous**.
5. **Wavelengths.** Strictly increasing, 32 values, equal to the build's `selected_wavelengths.npy`. Record the gap.
6. **Scope.** The Pass-1 train patients must equal the split's train patients. This is the proof of leak-free band selection.

**Outputs:** `prep/discovery_report.json`, `prep/integrity_report.json`. The stage exits non-zero on a hard failure.

**Acceptance:**
- 0 NaN/Inf.
- Every consistency check passes.
- 499 train captures are contiguous with n = 4,914.
- Scope verified.
- Peak RAM under 2 GB; about 10 min of I/O.

---

## Phase 2: Spectral and spatial preprocessing (`preprocess.py`)

The cubes are already calibrated, gain-corrected and band-selected on train. **This phase audits; it does not re-calibrate.** There is no white reference to re-calibrate with, and the classifier's data must stay fixed.

1. **Calibration audit.**
   - Gain distribution per split, class and patient, from `capture_index.json`.
   - Flag clipped captures; patient 68 is held out.
2. **Band validity.**
   - Per-band variance greater than 0.
   - Saturated and near-zero fractions.
   - Band–band correlation on a 200k train sample.
   - **Per-band noise σ_b** = robust std of the 4-neighbour Laplacian residual. It feeds `sensor_noise` in Phase 5.
   - Problems are reported, never silently dropped.
3. **Train-only statistics**, exact over all train rows (float64), saved to `prep/norm_stats.json`:
   - (a) linear per-band mean/std;
   - (b) **log-space** stats of `log(x + 1e-3)` on tissue rows, used as the generator representation (Phase 6).
4. **Tissue / background score** (generators, QC and analysis only). Absolute brightness depends on each capture's glass content, so the score must be **scale-invariant**.
   - Glass reference shape `s_glass`: for each capture, the brightest 1% of patches by band-mean; their mean spectrum is unit-normalised; take the median across train captures.
   - Score `t = SAM(patch mean spectrum, s_glass)` in degrees. Within-patch std is a secondary feature.
   - Threshold τ: a 2-component GMM on `log t` over a 500k stratified train sample, with Otsu as the fallback.
   - Save τ, `s_glass`, `tissue_score_train.npy` (float16) and `tissue_mask_train.npy`.
5. **Glass-rule measurement (pre-registered).** Report the DCIS glass fraction per patient. If it is above **10%**, the copy arms use tissue-only copies (Phase 10). The decision is written to `prep/glass_rule.json`.
6. **Visual sign-off.**
   - Capture mosaics on the 54×91 grid, coloured by `t` and by the mask.
   - Next to them, an RGB composite using `resolve_rgb_bands((640, 550, 460))`.
   - At least 3 captures per class, including all 5 DCIS patients.
   - **The user approves τ before Phase 4's pool is built.**
7. **Alignment checks.**
   - Label ↔ capture tissue, from P1.
   - Band order ↔ wavelengths.
   - **Edge-continuity test of the derived grid:** the median correlation of adjacent `(r, c)`/`(r, c+1)` patch-edge spectra must be ≫ that of a shuffled grid. This proves the coordinates.

**Acceptance:**
- The stats agree with a `compute_global_channel_stats` sample estimate within sampling error.
- The grid test passes in ≥95% of captures.
- The user signs off on τ.
- `glass_rule.json` is written.

---

## Phase 3: Splitting and leakage prevention (`io.SplitGuard`, `pool.py`)

1. **Keep the fixed split** (seed 42, patient-grouped, 35/5/5). Verify patient and capture disjointness, and report per-class patient counts (DCIS 5/1/1).
2. **Leakage ledger** (`prep/leakage_ledger.json`):

   | Fitted quantity | Fitted on | Evidence |
   |---|---|---|
   | Band selection, gain, value scale | train | `pass1_scope_report` |
   | Linear and log stats, τ, `s_glass`, σ_b | train | P2 |
   | Feature spaces (PCA-64, mean/std) and QC references | train (LOPO for thresholds) | P8 |
   | CVAE / AE / LDM weights | train | P6/P7 |
   | Generator early stopping, β/τ/w choices | train LOPO folds | P6/P7 |
   | QC classifier | train (cross-fitted LOPO) | P8 |
   | A4 generator choice | LOPO utility (train) | P8c |
   | Classifier checkpoint + arm selection | val | P10 |
   | Test | final report only | P10 |

3. **`SplitGuard`.** Every `hsi_augment` reader opens splits through it. Generator, QC and pool code raise on val or test. A unit test asserts that no stage before P10 reads test.
4. **LOPO folds inside train** (`pool/folds.json`). Fold *k* holds out DCIS patient *k* plus one healthy and one IDC train patient, chosen with a seeded draw. This gives 5 folds.
5. **Classifier side.** `train.py` always evaluates test (`--eval_test best`, `train_cli.py:475`), and the sweep's done-detection needs `test_predictions.npz`. So the seal is enforced at the **reading layer**: `augmentation_select.py` and the screen section of the report read only `history.json`. A test monkeypatches `open`/`np.load` to fail on `test_*` paths.

**Acceptance:**
- The disjointness assertions pass.
- The ledger has no unknown entries.
- The SplitGuard and seal tests are green.

---

## Phase 4: Labeled samples, provenance, imbalance, generator pool (`preprocess.py`, `pool.py`)

There is no re-extraction: the build already holds labelled 11×11×32 patches.

1. **Provenance index** (`prep/index_train.npz`, about 60 MB). Columns:
   - `row`, `y`, `patient`, `capture`, `ordinal`, `r`, `c`;
   - `tissue_score`, `is_tissue`;
   - `patch_feats` (per-band mean ⊕ std, 64 values, float32; also used by QC and LOPO utility).

   Val gets a version without coordinates, used for analysis only.
2. **Imbalance report.**
   - Counts and shares per class and split.
   - Imbalance ratio: IDC:DCIS = 13.1 on train.
   - Effective number of samples (Cui 2019; β ∈ {0.999, 0.9999}).
   - **Patients per class: 30 / 5 / 34.** This is the real bottleneck.
   - Per-patient contributions.
   - The same tables restricted to tissue rows.
3. **Generator pool** (`pool/X_pool.npy`, about 3 GB float16, held in RAM during fitting).
   - All tissue DCIS rows, plus about 120k **patient-balanced** tissue rows each of healthy and IDC.
   - Stored in the log-space representation (P2.3b).
   - Sidecars: `y`, `patient`, `row`, `capture`.
   - Read once from the memmap.
   - One pool per LOPO fold is not needed: fold membership is an index mask over the same pool.
4. **Real-vs-real reference statistics**, from the LOPO held-out patients against the remaining patients:
   - **cross-capture** nearest-neighbour distances in PCA-64;
   - mean/std feature distributions;
   - set-level metric ranges between patients.

   Saved to `pool/reference.npz`.
5. **Limitation note.** At ~0.74 µm/px, an 11×11 patch covers about one cell, so DCIS-vs-IDC architecture may be invisible. Context tiles need a new raw build. This goes to `FUTURE_EXPERIMENTS.md` as **F13**.

**Acceptance:**
- Row counts reproduce the known tables.
- Pool ≤ 3.5 GB, with class and patient quotas as configured.
- Reference statistics saved.

---

## Phase 5: Spectral–spatial augmentation (`transforms.py` + `train.py` wiring)

**Principle:** only label-preserving, physically plausible transforms, applied on the fly to train rows only. The val and test datasets are never wrapped (`train_pipeline.py:458-474`).

**Reused unchanged:**
- the D4 spatial group (flips, rot90; no interpolation, so spectra are never mixed);
- `paper_recipe`'s crop, scale, offset and noise;
- minority oversampling and intra-class cross-patient mixing (`stats` = MixStyle, or `pixel`) via `MinorityAugmentedDataset`.

**New ops**, defined in build value space and applied as x = x_n·σ + μ → op → back, using the dataset's global stats:

| Op | Model | Default bound |
|---|---|---|
| `spectral_tilt` | x·(1 + a·P₁(λ̃) + b·P₂(λ̃)), Legendre polynomials over the **real wavelengths**, drawn once per sample (a smooth illumination or stain-tint change) | \|a\|, \|b\| ≤ 0.03 |
| `sensor_noise` | additive N(0, k·σ_b) using the per-band noise from P2.2 | k = 1 |
| `od_stain_jitter` *(experimental, off in every arm)* | optical density is computed against `s_glass`; NMF components (K = 3) are fitted on train tissue OD; their coefficients are jittered by U(1 ± δ) | δ = 0.1 |

**Dropped after review:**
- global gain: median-ratio correction already removes it, and per-band scale is in the recipe;
- band dropout and masking: they break spectral relationships;
- wavelength shift: interpolation is invalid across the gap;
- arbitrary rotations and elastic warps: they interpolate spectra;
- inter-class mixup: it breaks the label.

**Invariants (property tests plus the augmentation report):**
- labels, shape and dtype are preserved;
- deterministic given `(seed, epoch, index, worker)`;
- mean SAM(original, augmented) ≤ 2° per op;
- band–band correlation change (Frobenius) < 0.05 over 10k samples;
- values stay ≥ 0 in value space;
- train-only wrapping.

**`train.py` wiring.** A new `_add_phys_flags(p)` is called at the end of `_add_flags`, with these flags (all off by default):
- `--phys_aug {off,minority,all}`
- `--phys_tilt_max`
- `--phys_noise_k`
- `--phys_od_delta`
- `--phys_artifacts DIR` (P2 outputs)

Rules for the composed function:
- It keeps the `augment_fn(patch, rng, gen)` signature (`training/augmentation_v16.py:79`).
- It calls the base `Compose` first and draws its own randomness **afterwards**, so `--phys_aug off` leaves the stream bit-identical.
- It never touches the `norm_stats` element.
- `minority` requires `--minority_augment on`.
- All modes require `--normalization global_zscore`; otherwise `SystemExit`.
- It adds the run-name suffix `_phys` and writes `config["phys_aug"]`.

**Acceptance:**
- Property tests pass.
- The augmentation report exists (SAM/correlation per op, montages).
- The parity test is unchanged.

---

## Phase 6: Conditional VAE (`models/cvae.py`), arm A2

**Representation (shared by P6 and P7).** y = (log(x + 1e-3) − μ_b)/σ_b. Decoding goes through exp(·), so outputs are always positive. Outputs are clipped at 1.1 × the train p99.99, then cast to float16. Log space makes illumination and stain effects additive (Beer–Lambert), and SAM is computed on the decoded values (`spectral_recon_metrics_v2.sam_deg`; `SAMLoss` imported from `training.gan`).

**Architecture (~3.5M params)**
- Encoder: conv 32→96→192 (6×6)→256 (3×3).
- GroupNorm + SiLU residual blocks. No BatchNorm: MedMamba-SS already showed a BN artefact on this HSI data.
- Latent **d = 64** (grid {32, 64}).
- Decoder: ConvTranspose (deterministic, no bilinear upsampling), FiLM class conditioning, crop 12→11.

**Loss**
- σ-VAE Gaussian NLL with a learned per-band σ.
- \+ λ·SAM. λ is set so SAM is about 10% of reconstruction at init; grid {0, 0.1}.
- \+ β·KL, with β warmed up 0→1 over 10% of steps and free bits 0.05 nat/dim.

**Training**
- Batches are class-balanced, then patient-balanced within each class.
- AdamW 1e-3 → cosine to 1e-5, weight decay 1e-4, gradient clip 1.0, batch 512, fp32.
- Up to 40k steps, with early stopping on the held-out patient's NLL in each LOPO fold.
- Sampling: z ~ N(0, τ_s²), with τ_s ∈ {0.8, 1.0} chosen on LOPO QC.

**Final model**
- Refit on all 5 DCIS patients with the LOPO-chosen settings.
- Step budget = the median of the folds' best steps.
- EMA weights are used.
- No further selection, so no selection data is needed.

**Diagnostics** (on LOPO held-out patients only)
- Per class: MSE, PSNR, SAM°, `ssim2d`.
- Active units, defined as Cov_x E[z] > 0.01.
- KL per dimension.
- Prior-sample QC acceptance.

A patient-identity probe on the latent is **not** used as a memorization measure. Patient signatures are expected to be decodable; memorization is measured by nearest-neighbour distance in P8.

**Checkpoints**
- Atomic writes: temp file, then `os.replace`.
- `best.pt` and `last.pt`, each with model, EMA, optimizer, scheduler (Python floats), step, epoch, step within epoch, and all RNG states.
- Data order is `default_rng([seed, epoch])`; resume skips the batches already consumed.
- Also written: `history.csv`, `metrics.jsonl`, `config.json`, `done.json`.

**GPU estimate:** about 20–30 min per fit. The 5 LOPO folds take about 10 min each. Across the small grid, the total is about **2–2.5 GPU-h**.

**Acceptance**
- CPU: one batch can be overfitted; resume is bitwise equivalent; the sampling sha is deterministic.
- GPU:
  - held-out-patient SAM° ≤ 1.25 × train-patient SAM°;
  - at least 50% of latent units active;
  - KL per dimension above the free-bits floor.

---

## Phase 7: Conditional latent diffusion (`models/autoencoder.py`, `models/diffusion.py`), arm A3

**Design (changed after review).** A **separate, unconditional, low-KL autoencoder** is used instead of the CVAE posterior. The CVAE latent is class-stripped, because the decoder receives the class. With that latent, guidance would have nothing to act on, and A3 would end up nearly the same as A2. With the separate autoencoder, the diffusion model carries the class information, and the two arms are genuinely different generators.

**Autoencoder (~3M params)**
- Same backbone as the CVAE, unconditional.
- **Spatial latent 4×6×6** (144 dims, 27× compression).
- Loss: MSE (log-z) + 0.1·SAM + 1e-6·KL.
- Latent scaled by 1/std.
- 30k steps, about 20 min.

**Denoiser (~6–8M params)**
- Tiny UNet (6×6→3×3, 128/256 channels), or an MLP-ResNet over the 144 dims.
- Sinusoidal time embedding, plus a class embedding with 3 classes and a **null** token.
- v-prediction, cosine schedule with T = 1000, Min-SNR-γ = 5.
- Label dropout 0.1.
- AdamW 2e-4 (1k-step warmup), batch 1024, 150k steps, EMA 0.9995, bf16.

**Sampling**
- DDIM, 50 steps, η = 0.
- Guidance w ∈ {1, 1.5, 2, 3}, chosen on **LOPO QC**: precision, recall and coverage of the held-out patient, plus memorization.
- High guidance (w > 3) pulls samples toward the training modes, so it is avoided.

**LOPO:** 2 folds for selection; the final model is refit on all patients, as in P6.

**GPU estimate:** about **2.5–3 GPU-h** including the autoencoder.

**Acceptance:**
- Held-out-patient coverage at least the CVAE's.
- QC acceptance ≥ 40%.
- Latent MMD between generated and encoded held-out data ≤ 1.5 × the MMD between encoded train and encoded held-out data.

---

## Phase 8: Generation, QC and LOPO utility (`generate.py`, `qc/*`, `lopo_utility.py`)

### 8a. Generation

- **Target:** exactly **122,850 accepted DCIS samples per source**, which is 1:1 with real DCIS.
  - With a fixed count, rows per epoch, total steps and EMA rate are provably identical across arms. This is why `--synthetic_ratio` was dropped.
  - Target classes come from `resolve_minority_classes("auto")`, which selects DCIS. Other targets can be configured for future datasets.
- **Over-generation:** 2× (245,700 per source), in under 10 minutes.
- **Stored with each sample:** seed, source, checkpoint sha, τ_s or w, and its generation index.

### 8b. QC

Every check runs on the **float16 export cast back to value space**. Each rule is **calibrated on real LOPO held-out DCIS**: each rule must accept ≥ 99% of those real patches, and all rules together ≥ 95%. The rejection bitmask is stored for every generated sample.

| Rule | Check |
|---|---|
| R1 validity | Finite **after** the float16 cast; `training/npy_data.py:223` would otherwise silently turn an overflow into 1.0. At most 1% of values outside the per-band [p0.05, p99.95] of train DCIS tissue. |
| R2 spectral | Mahalanobis distance of the log mean spectrum to the DCIS Gaussian. Curvature energy (second differences) computed **within each contiguous band segment**, never across the gap. |
| R3 spatial | Lag-1 autocorrelation and high-frequency power fraction (radial spectrum) within the real [q0.5, q99.5]. This catches blurry or flat samples; `ssim2d` is only for paired comparisons. |
| R4 tissue | t ≥ τ. |
| R5 class consistency | Cross-fitted (LOPO) mean/std logistic regression: p(DCIS) ≥ the real q5, **and** k = 10 nearest-neighbour label agreement ≥ the real q5. The classifier is built with `shallow_probe_heldout.fit` and `shallow_baseline.featurize("mean_std")`; it is independent of MedMamba. |
| R6 memorization | No exact byte-hash match with a train patch. Nearest-neighbour distance (PCA-64 GPU search with chunked `torch.cdist`, exact re-rank of the top 10) ≥ q1 of the real **cross-capture** NN distances. Alaa-style authenticity is reported. |

**Set-level metrics** (report and gate). Each is shown next to its **real cross-patient range**:
- per-band Wasserstein-1;
- Frobenius distance between band-correlation matrices;
- precision, recall, density and coverage (k = 5);
- Fréchet distance and MMD in the mean/std and PCA-64 feature spaces;
- a **nearest-neighbour patient attribution histogram**, flagged if one patient takes more than 40%;
- A0-feature Fréchet distance and the A0-predicted class mix (reporting only).

**Selection:** the first 122,850 accepted samples in generation order, which is deterministic. The stage fails loudly if acceptance is below 40%.

**Visual QC:**
- class-mean ± sd spectra, real vs synthetic, with the gap shaded;
- RGB contact sheets stretched from a real reference;
- a nearest-neighbour gallery (synthetic | nearest real | difference).

### 8c. LOPO utility test (train-only; decides A4 before the screen)

1. In each fold, train a mean/std logistic regression on four training sets:
   - real only;
   - real + copies (count-matched);
   - real + CVAE;
   - real + LDM.
2. Score each on the held-out DCIS patient's recall and F1, and on that fold's macro-F1.
3. Optionally repeat with HybridSN (`hsi_baselines.build_baseline`). That needs a GPU, about 1 GPU-h.

The generator with the better mean LOPO DCIS F1 becomes **A4's source**. That choice is written to `lopo_utility/decision.json` **before** the screen starts. The test runs on the host CPU in about 30–60 minutes.

**Acceptance:**
- Exactly 122,850 rows per source.
- Real held-out acceptance meets the calibration targets.
- Hand-crafted reject cases are caught.
- Re-running gives the same sha256.
- No single patient takes more than 40% of the attribution.
- `decision.json` written.

---

## Phase 9: Integration and export (`export.py`, `training/synthetic_mix.py`)

**Export** (`data/hsi_v9-trainsel-aug/export/<source>/`). Real data is **referenced, never copied**. `--materialize_merged` was dropped: it would be a 19 GB copy and would invite computing statistics on synthetic data. A reader function is provided instead.

```
X_syn.npy          float16 NHWC (122850,11,11,32), build value space
y_syn.npy          int64          groups_syn.npy  int32 (−1 cvae, −2 ldm; never a patient id)
gen_index.npy      int64          qc_scores.npz   ALL generated samples + rejection bitmask
dataset_manifest.json             write_dataset_manifest(sha256=True)
provenance.json    {schema_version, export_id = sha256(X_syn)[:12], source_manifest_sha256,
                    generator_ckpt_sha256, config_hash, git_sha, seeds, n_generated, n_accepted,
                    thresholds, norm/log stats sha}
wavelengths.npy, class_names.json (equal to the real build's), figures/, qc_report.json
```

**`train.py` flags.** A new `_add_synthetic_flags(p)`:
- `--synthetic_dir`: a plain `str`, default `None`, no existence check at parse time (`tests/test_sweeps.py:52-55` parses static argv).
- `--synthetic_build_check {on,off}`, default `on`.

**`training/synthetic_mix.py`** (light imports; owns the schema constants, which `export.py` imports):
- `load_synthetic_export(path, real_data_dir, num_classes, hwc)` checks:
  - the manifest, via `verify_against_manifest`, honouring `--manifest_check` as `train_preflight._check_manifest` does;
  - the schema version;
  - **`source_manifest_sha256` equals the sha256 of `<data_dir>/dataset_manifest.json`**, so a v8 export can never be used on v9;
  - float16 dtype, every value finite, labels in the target classes;
  - wavelengths and class names equal to the real build's.
- `attach_synthetic(real_ds, y_real, groups_real, export, build_ds)` returns `MixedTrainSet{dataset, y, groups, pool_labels, n_real, n_syn, describe()}`:
  - `dataset` is a `ConcatDataset` of the real `NpyDatasetV16` and a synthetic `NpyDatasetV16` built with **the same `global_mean/std` taken from the real `X_train`**, so 3-tuples and normalization are identical;
  - `pool_labels` is `y_real ⧺ −1`.
- `plan_minority_with_synthetic(y_real, n_syn, ..., factor, copy_mask=None) -> MinorityPlan`:
  - indices = `build_minority_oversample_indices` over **real rows only**, followed by `arange(n_real, n_real + n_syn)`;
  - `copy_mask` restricts copies to tissue rows when the glass rule fires. It still adds exactly (factor − 1)·n_class extra rows, so N_total is unchanged;
  - classes are resolved on real counts.
  - It must equal `plan_minority_augmentation` when `n_syn = 0` and `copy_mask = None`; a test pins this.

The glass-rule copy mask is exposed as `--minority_copy_mask PATH`, a boolean `.npy` aligned to the train rows, added to `_add_synthetic_flags`.

**Edits to `training/train_pipeline.py`** (not frozen; all verified against the code):

| Where | Change |
|---|---|
| `:89-91` | Run-name suffix `_syn`, only when `--synthetic_dir` is set. `test_run_tag_v18` stays green. |
| After `discover_data` (`:301`) | `load_synthetic_export`, failing fast **before** the slow sha256 pass over the real manifest. |
| **New block between `:393` and `:395`** | `mix = attach_synthetic(...)`, using the real groups from the sidecars (`:353-358`). |
| `:396-404` | `SystemExit` if `mix` is set and `--sampler` is not `none`. Both samplers read only `y_train_arr`, so they would ignore synthetic rows. |
| `:415` | Use `plan_minority_with_synthetic` when `mix` is set or `--minority_copy_mask` is given. |
| `:459-466` | `labels=mix.pool_labels` and `groups=mix.groups`. Mixing partners therefore come only from **real** rows, because `np.flatnonzero(labels == c)` at `minority_augmentation.py:282-285` skips −1. **`minority_augmentation.py` itself needs no edit.** |
| `:558-559` | Class weights come from `minority.effective_labels`, else `mix.y`, else `y_train_arr`. This keeps `effective` weights in every arm. |
| `:623-626` | `config["synthetic"] = mix.describe()`, holding export_id, counts and provenance, plus `synthetic_mix_report.json`. |

**Unchanged:**
- coverage and imbalance reports (`:326-331`) and gate G8 (`:360-372`) stay real-only;
- global stats remain fitted on the real path only (`:339-340`);
- the default path is untouched.

**Resulting order:**

`NpyDatasetV16(real) ⧺ NpyDatasetV16(syn)` → [minority Subset] → [per-epoch subsample] → augmentation wrapper (outermost) → DataLoader

**How synthetic rows are treated in each arm.** Synthetic rows get **exactly what real rows of their class get**:
- in A2/A3, the base `paper_recipe` augmentation;
- in A4, the minority path: mixing p = 0.5 with **real** partners, then the minority transform.

They also enter the reconstruction MSE/SAM loss of `paper_recipe`. This is documented in `config["synthetic"]`.

**Acceptance (CPU):**
1. An **empty export** gives `history.json` bit-identical to a run without one.
2. Global stats are identical.
3. No synthetic row is a mixing partner, and only real rows are copied.
4. The val and test loaders are unchanged.
5. A build-sha mismatch is refused.
6. A non-finite synthetic value is refused.
7. A tiny end-to-end `train.py` run with `--synthetic_dir --minority_augment on --phys_aug minority` succeeds.
8. The parity, run-tag and sweep tests stay green.

---

## Phase 10: Evaluation and reporting (`sweeps/augmentation_pipeline.py`, `scripts/augmentation_{select,report}.py`)

**REF** = `paper_recipe`, batch 256, `lambda_sam 0.1`, `val_subsample_frac 0.0986`, data `data/hsi_v9-trainsel/hsi`, effective class weights.

Flags must be **identical** to the paper sweep and the minority sweep, or shared tags break. So `--minority_classes` stays `auto` (which resolves to DCIS), and `trm_ema_rate` is **not** pinned. Instead, acceptance checks that every arm's `config.json` has `train_subsample.kept == 250113` and the same `trm_ema_rate`.

Both parse-time EMA values (931 vs 978 steps/epoch × 3 horizon epochs) clamp to **0.9995** (`train_profiles.py:144,197-207`), so they are already identical.

### Arms (2×2 + control)

Every non-A0 arm has N_total = **2,574,936** and frac **0.0971337**, which keeps **250,113 rows/epoch**, the same as A0 at frac 0.102.

| Arm | Tag (screen / seeds) | Extra DCIS rows | Mixing | Key flags |
|---|---|---|---|---|
| A0 | `v20-abl-base-s42` / `v20-head-s*` (shared with the paper) | none | no | REF, frac 0.102 |
| A0os | `minaug-nomix` (shared) † | copy ×2 | no | `--minority_augment on`, factor 2, `--minority_mix_prob 0` |
| A1 | `minaug-default` (shared) † | copy ×2 | stats 0.5 | factor 2 |
| A2 | `aug-cvae` | +122,850 CVAE | no | `--synthetic_dir …/export/cvae` |
| A3 | `aug-ldm` | +122,850 LDM | no | `--synthetic_dir …/export/ldm` |
| A4 | `aug-combo` | +122,850 from the LOPO-chosen generator | stats 0.5 | `--minority_augment on`, factor 1.0, `--synthetic_dir …` |
| A1p *(optional, +2 h)* | `aug-conv-phys` | copy ×2 | stats 0.5 | A1 + `--phys_aug minority` |

† If `glass_rule.json` says **tissue-only copies**, A0os and A1 get new tags (`aug-copy`, `aug-copymix`) with `--minority_copy_mask prep/tissue/dcis_copy_mask.npy`. They then stop sharing tags with the minority sweep.

### Stages

Each stage is separate and ordered. `requires` is checked once per stage, and a job that is still waiting is skipped (`run_experiments.py:250-260`).

1. `prep`: P1 → P4 as `cmd()` jobs. `done_if` points at `done.json`.
2. `gen-cvae`: LOPO folds, then the final fit.
3. `gen-ldm`: AE, LOPO, final.
4. `export`: generate, QC, export per source.
5. `lopo-utility`: writes `decision.json`.
6. `bench`: a one-epoch solo run per arm type with `--jobs 1`. This is the only wall-clock cost source; concurrent runs invalidated the v18 timings.
7. `screen`: A0, A0os, A1, A2, A3, A4 (+ A1p) at 12 epochs, seed 42.
   - Synthetic arms use a custom `job.done`: done only if `config.json["synthetic"]["export_id"]` equals the current export's `provenance.export_id`, following `sweeps/paper_evidence_upgrade.py:153-157`. Regenerating an export therefore re-runs its arms.
8. `select`: `scripts/augmentation_select.py` writes `selection.json`. **Pre-registered rule:**
   - take the best generative arm (A2/A3/A4) plus the best non-generative arm (A0os/A1/A1p);
   - score = **mean val `f1_macro` over epochs 10–12**, read from `history.json`;
   - tie (gap < 0.005) → val DCIS F1 (`per_class_f1[1]`) → the cheaper method;
   - test is never read.
9. `seeds`: the 2 selected arms × seeds {1, 7, 13, 23, 42} × 20 epochs, with A0 = `v20-head-s*`.
   - This stage is built **only if `selection.json` exists when the sweep loads**; otherwise it is `[]`. That keeps argv static and parseable.
10. `report`.

### Metrics

All metrics come from artifacts; nothing is recomputed on the GPU.
- macro-F1 and balanced accuracy.
- Per-class precision, recall and F1, with **DCIS F1/recall, DCIS precision and the healthy→DCIS false-positive rate**. The model already over-predicts DCIS (`training/losses.py:49-60`).
- AUROC and PR-AUC.
- ECE and NLL.
- Confusion matrices.
- **Per-patient tables**, with DCIS recall for 197 (val) and 136 (test).
- **Tissue vs glass breakdown** of val/test metrics, using the train-fitted mask applied without labels. Analysis only.

### Statistics (pre-registered)

- Paired per-seed deltas against A0: mean ± sd, a 95% t-CI and the sign count.
- **Minimum effect:** Δ macro-F1 ≥ 0.01 with at least 4 of 5 seeds positive.
- A one-sided Wilcoxon test is reported only as p = 0.03125 when all 5 seeds improve. A two-sided test with n = 5 can never go below 0.0625.
- A new `stats.patient_bootstrap(labels=[0,1,2])`. The existing `patient_bootstrap_delta` (`heldout_metrics.py:99-124`) mixes 2- and 3-class F1, and 33% of its test resamples contain no DCIS.
- **No patient-level CI is claimed for DCIS**, because each held-out split has one DCIS patient.
- Seed 42 appears in both screen and seeds, so it is not independent of the selection; this is noted.

### Cost table

- Generator GPU-minutes and peak VRAM (`torch.cuda.max_memory_allocated`).
- Sampling throughput.
- QC and LOPO wall time.
- Export disk size.
- Classifier s/epoch from `bench`.
- GPU-h per arm, including the amortised generator cost.

### Report

`reports/augmentation/<date>/report.md` + `.json` + figures. It contains:
- the arm table;
- a seed-delta forest plot;
- per-class bars;
- per-patient tables;
- the QC summary;
- the cost table;
- every resolved config, the git sha and every input sha256.

Test columns render only when `selection.json` exists **and** `--reveal_test` is passed.

### GPU budget

| Item | GPU-h |
|---|---|
| Generators | ~5–6 |
| Screen (5 new arms) | ~10.3 |
| A1p (optional) | +2.1 |
| Seeds (2 arms × 5 × 20 epochs) | ~34 |
| A0 runs (`v20-abl-base-s42` ~2.1, `v20-head-s*` ~17) | needed by the paper anyway |

**Optional M11:** run the best generative arm against `v20-cv-f*` on the F3 patient-level CV folds (~27 GPU-h). That covers all 7 DCIS patients. The pipeline is fold-agnostic through `--data_dir`.

**Acceptance:**
- Every arm finished.
- kept = 250,113, with equal steps and EMA across arms.
- `export_id` matches the current exports.
- `selection.json` was written before any test column.
- The report regenerates byte-identically.

---

## Phase 11: Testing, reproducibility, optimization

### Fixture

`tests/hsi_augment_fixtures.py` builds a tiny trainsel-like set:
- 3 classes, 8 patients, 2 captures each, a 4×5 grid, patch 5, 8 bands with a wavelength gap, float16 NHWC;
- sidecars, `capture_index.json`, manifest, `split_assignment.json`, `pass1_scope_report.json`;
- **flat glass vs absorbing tissue spectra**, so the tissue mask has a known answer;
- one exact duplicate of a train patch, for the memorization test.

It reuses the `TINY` flags and subprocess pattern of `tests/test_minority_augmentation.py:276-313`.

### Test files (CPU; seconds each)

| File | Covers |
|---|---|
| `test_hsi_augment_discovery.py` | integrity passes; injected NaN, label/capture mismatch and non-contiguous capture are each caught |
| `test_hsi_augment_preprocess.py` | float64 chunked stats equal numpy; tissue score separates flat from absorbing spectra; τ is train-only; the grid edge-continuity test passes, and fails on a shuffled grid |
| `test_hsi_augment_splits.py` | disjointness; `SplitGuard` raises; access log written; LOPO folds valid |
| `test_hsi_augment_pool.py` | quotas and patient balance; the log transform round-trips |
| `test_hsi_augment_transforms.py` | label, shape and dtype preserved; determinism; SAM/correlation budgets; ≥ 0; train-only; `--phys_aug off` is bit-identical |
| `test_hsi_augment_cvae.py`, `_autoencoder.py`, `_diffusion.py` | shapes; finite loss; overfits one batch; **resume equivalence** (2+2 steps = 4, bitwise on CPU); `weights_only=True` load; v-target algebra; DDIM deterministic at η = 0; CFG null token; sampling sha reproducible |
| `test_hsi_augment_qc.py` | crafted rejects: duplicate (R6), out of range, or non-finite after the cast (R1), glass (R4), wrong class (R5), blurred or flat (R3); calibration meets the 99%/95% targets on real held-out data |
| `test_hsi_augment_export.py` | manifest round-trip; schema; exact count; negative groups; deterministic export sha |
| `test_synthetic_mix.py` | the **8 acceptance items of Phase 9**, including the empty-export bit-identity and the end-to-end `train.py` run; `plan_minority_with_synthetic` matches `plan_minority_augmentation` |
| `test_augmentation_sweep.py` | selection reads only `history.json` (`open`/`np.load` patched to fail on `test_*`); `seeds` is `[]` without `selection.json`; every arm parses and resolves `kept = 250113` and an equal `trm_ema_rate`; the custom `done` honours `export_id` |
| `test_hsi_augment_pipeline_e2e.py` | `augment_pipeline.py all` on the fixture with tiny step counts; skip-if-done; `--force`; `--resume auto` |

Existing guards must stay green: `test_frozen_files_untouched`, `test_live_copies_match_archive`, `test_train_e2e_parity`, `test_run_tag_v18`, `test_minority_augmentation` and `test_sweeps`.

### Reproducibility

- Per-stage seeds.
- `done.json` records the config hash, the input manifest sha and the output shas.
- Git sha recorded.
- `config.json` written next to every output.
- Checkpoints and resume follow the contract in P6.

### Logging

- `stage.log` (Python `logging`, mirrored to the console in the repo's print style) and `metrics.jsonl`.
- Sweep logs go to `logs/augmentation_pipeline/`.

### Memory and time

- 20k-row memmap chunks with float64 accumulators.
- The pool is materialized once (~3 GB).
- AE latents are precomputed.
- Nearest-neighbour search runs in PCA-64 on the GPU with chunked `cdist`, with a CPU fallback, against same-class reference rows only.
- bf16 for diffusion.
- RAM guard via `training/system_memory.get_ram_info`: refuse if the projected peak exceeds 50% of available RAM.
- Disk preflight before any write larger than 1 GB.

**Sandbox rule:** only CPU tests and tiny fixtures run in this VS Code sandbox. Every real-data stage runs on the host.

---

## Milestones

| M | Work | Where | Acceptance |
|---|---|---|---|
| M0 | Phase 0; retarget the minority sweep; **start A0 runs** | sandbox + host GPU | suite green; dry-run valid |
| M1 | `synthetic_mix` + flags + pipeline edits (Phase 9 integration first: it is the riskiest code) | sandbox CPU | the 8 acceptance items of Phase 9 |
| M2 | Package skeleton (`config`, `stage`, `io`, adapter), fixtures, P1 | sandbox CPU → host | P1 acceptance |
| M3 | P2 + P4 index and imbalance report | sandbox CPU → host CPU (~0.5–1 h) | grid test; **user signs off τ**; `glass_rule.json` |
| M4 | P3 ledger and folds + P4 pool and references | host | pool ≤ 3.5 GB; references saved |
| M5 | P5 ops + `--phys_*` flags | sandbox CPU | property tests; parity unchanged |
| M6 | P6 CVAE (LOPO + final) | sandbox CPU smoke → host GPU ~2.5 h | P6 acceptance |
| M7 | P7 AE + LDM | sandbox CPU smoke → host GPU ~3 h | P7 acceptance |
| M8 | P8 generate, QC and export + LOPO utility | sandbox CPU tiny → host | P8 acceptance; `decision.json` |
| M9 | Screen → select | host GPU ~10–12 h | per-arm checks; `selection.json` from val only |
| M10 | Seeds + report | host GPU ~34 h | P10 acceptance |
| M11 *(optional)* | F3 CV check | host GPU ~27 h | best generative arm vs `v20-cv-f*` over 5 folds |
| M12 | Docs and final suite | sandbox | see below |

**M12 documentation:**
- a `HOW_TO_RUN.md` §7.8 workflow entry and a `HOW_IT_WORKS.md` note;
- `FUTURE_EXPERIMENTS.md` entries **F13** (context tiles), **F14** (pixel-space diffusion), **F15** (generator-seed variance and running on CV folds) and **F16** (OD stain jitter once a blank reference exists);
- the full test suite green.

---

## Files

**New:**
- `augment_pipeline.py`
- `hsi_augment/**`
- `training/synthetic_mix.py`
- `sweeps/augmentation_pipeline.py`
- `scripts/augmentation_select.py` and `scripts/augmentation_report.py`
- `tests/hsi_augment_fixtures.py` and `tests/test_hsi_augment_*.py`
- `tests/test_synthetic_mix.py` and `tests/test_augmentation_sweep.py`

**Modified (none frozen):**
- `training/train_cli.py`: `_add_phys_flags` and `_add_synthetic_flags`, called from `_add_flags`.
- `training/train_pipeline.py`: the edits in the Phase 9 table.
- `sweeps/minority_augmentation.py`: retargeted.
- `HOW_TO_RUN.md`, `HOW_IT_WORKS.md`, `FUTURE_EXPERIMENTS.md`.

**Untouched:**
- every `FROZEN_GLOBS` file;
- the pinned definitions;
- `training/minority_augmentation.py`. The `pool_labels` approach avoids any edit to it.

---

## Risks and limitations (state them in any write-up)

1. **Five DCIS training patients.** The class signal is mostly patient spectral signatures, and generators re-sample 5 patients. The honest hypothesis is: *does generative interpolation beat copying plus mixing on an unseen DCIS patient?* The LOPO utility test asks exactly that, before any GPU arm runs.
2. **One DCIS patient per held-out split.** DCIS metrics are one patient's recall. Use per-patient tables and no DCIS patient-level CI. F3 cross-validation (M11) is the real fix.
3. **An 11×11 patch is about one cell.** The architectural DCIS/IDC cue may be invisible, and no augmentation can add it.
4. **Weak capture-level labels.** Glass and non-lesional tissue in DCIS captures are labelled DCIS. The glass rule and the tissue/glass metric breakdown expose this.
5. **The fixed test split was consulted** during earlier recipe development (F3 note). It is sealed for this study, but it is not pristine.
6. **One trained generator per type.** Seed spread covers classifier variance only (F15).
7. **Disk.** 88 GB free. This plan writes about 3 GB (pool) + about 2 × 2 GB (exports) + small checkpoints.
8. **Physics ops are heuristics.** They are bounded by SAM and correlation budgets and by montages, and evaluated only in the optional arm A1p.

---

## Verification (end to end)

1. **Sandbox (CPU)**
   - `python -m pytest -q` passes: all old tests plus `test_hsi_augment_*`, `test_synthetic_mix` and `test_augmentation_sweep`.
   - `git diff --name-only master -- <frozen list>` is empty.
   - `python run_experiments.py sweeps/augmentation_pipeline.py --dry_run` resolves every static train argv.
2. **Host, data**
   - `python augment_pipeline.py validate preprocess --data_dir data/hsi_v9-trainsel/hsi --out data/hsi_v9-trainsel-aug`
   - Inspect `prep/figures/tissue_maps_*.png`, approve τ, and read `glass_rule.json`.
3. **Host, GPU**
   - `python run_experiments.py sweeps/augmentation_pipeline.py prep gen-cvae gen-ldm export lopo-utility bench screen select`
   - Reload the sweep, then run `... seeds report`.
   - Check `--status` between stages.
4. **Review**
   - Check each phase's acceptance block against its `done.json` and reports.
   - Run `python scripts/augmentation_report.py --reveal_test` only after `selection.json` exists.
