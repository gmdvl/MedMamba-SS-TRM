# MedMamba-SS-TRM — How It Works

**What this is.** The technical guide to this repository: what the project does, how data flows
from raw files to predictions, what the model computes layer by layer, how training and evaluation
work, and how to explain the project to an academic audience. It is written for a technically
capable reader who has never seen the project.

**Companion file:** [HOW_TO_RUN.md](HOW_TO_RUN.md) — installation, dataset setup, configuration,
exact commands, expected outputs, troubleshooting and reproducibility.

**Written:** 2026-10-05, against commit `efc2afd` (`master`, identical to `origin/master`,
`git@github.com:gmdvl/MedMamba-SS-TRM.git`).

**How it was produced.** Every entry point, configuration file, model file, data-preparation
script and trainer class was read, and the main workflows were executed on a CPU (details below).
Nothing here is taken on trust from older documents unless it is labelled as such. The structure
follows the prompt in [`plan/Project_Uderstanding_and_Execution_Guide.md`](plan/Project_Uderstanding_and_Execution_Guide.md).

---

## How to read this guide

Every factual statement carries one evidence label.

| Label | Meaning |
|---|---|
| **[Code]** | Read in the source code. A `file:line` link is given where it helps. |
| **[Config/Doc]** | Stated in a configuration file or in the project's own documentation, and not re-derived from code. |
| **[Run]** | Executed during this audit (2026-10-05). See the list below. |
| **[Artifact]** | Read from outputs recorded on the author's machine (`experiments/`, `data/`, prep logs). These folders are git-ignored, so **a fresh clone does not contain them**. |
| **[Inferred]** | Reasoned from the evidence above, but not observed directly. |
| **[NOT VERIFIED]** | Could not be established. The text says what is missing. |

**What was executed** (Linux, Python 3.13.15, PyTorch 2.14.0+cpu, no GPU available):

1. The whole test suite: `python -m pytest -q` → **757 passed, 2 warnings, 203.5 s**.
2. `python train.py --help`.
3. A 2-epoch `train.py` run of the paper configuration on a **synthetic** 8-band dataset
   (96 / 48 / 48 patches of 11 × 11) → completed, every gate passed, about 25 s.
4. The paper model built through the same function the pipeline uses, plus a 2-sample forward pass,
   to record every tensor shape and the parameter counts.
5. `scripts/eval_test_split.py`, `scripts/flops_report.py`, `scripts/compare_runs.py`,
   `scripts/run_top3.py --dry_run`, and `run_experiments.py sweeps/paper_runs.py --list / --status / --dry_run`.
6. `--resume` of the synthetic run.
7. The effective configuration of five commands, resolved with `training.train_cli.resolve`.

**Not executed:** GPU training, training on the real datasets, data preparation from raw files,
sweeps, and the baselines. For those, the guide relies on the code plus recorded artifacts.

> **Python version.** The project environment is Python 3.11 (`mamba-spec.txt`). This audit ran on
> Python 3.13. The CPU tests pass on 3.13; GPU training on 3.13 is **[NOT VERIFIED]**.

---

## Contents

- [1. Project overview](#1-project-overview)
- [2. Architecture](#2-architecture)
- [3. Directory and file structure](#3-directory-and-file-structure)
- [4. End-to-end execution flow](#4-end-to-end-execution-flow)
- [5. Data pipeline](#5-data-pipeline)
- [6. Model and algorithms](#6-model-and-algorithms)
- [7. Training workflow](#7-training-workflow)
- [8. Evaluation](#8-evaluation)
- [9. Academic / presentation explanation](#9-academic--presentation-explanation)
- [Verified facts](#verified-facts) and [Unknown / not verified](#unknown--not-verified)

---

## 1. Project overview

### In plain words

This repository trains a small neural network to sort tiny pieces of medical images into
diagnostic classes.

Its main data are microscope slides of breast tissue photographed with a **hyperspectral camera**.
An ordinary camera records three colours (red, green, blue). A hyperspectral camera records
hundreds of narrow colour **bands**; here 740 bands between about 400 nm and 938 nm
**[Artifact]**. The images are cut into patches of 11 × 11 pixels. For each patch the network
predicts one of three tissue classes **[Code]**:

- **healthy**,
- **DCIS** (ductal carcinoma *in situ*, a non-invasive breast cancer),
- **IDC** (invasive ductal carcinoma).

The same network can also read ordinary RGB images: a matched RGB version of the same slides, and
the PAD-UFES-20 skin-lesion photographs **[Code]**.

### The problem it addresses

According to the README **[Config/Doc]**:

- **MedMamba**, the published model this work extends, embeds its input with a layer whose shape
  depends on the number of channels, and it does not know which wavelength each channel is.
- **MedMamba-SS** replaces that input layer with a *spectral pathway* whose parameters do not depend
  on the number of bands, and which reads each band's wavelength.
- **MedMamba-SS-TRM** also replaces MedMamba's multi-stage backbone with one small block that is
  applied over and over (the "Tiny Recursive Model" idea), to cut the parameter count.

MedMamba-SS-TRM is the default model of every training profile **[Code]**.

### Technical summary

For every pixel position of a patch, the spectral pathway turns the C band values into C tokens.
Each token mixes the band's value with an encoding of its wavelength. A bidirectional selective
scan (a Mamba-style state-space recurrence) runs along the band axis, and a learned gate weights
the bands. The tokens are then averaged and compressed into a 64-dimensional "spectral context" per
position. A linear stem lifts this to 128 dimensions and adds a 2-D positional encoding.

A two-layer core block is then applied **63 times** per forward pass. It refines a latent state
`z` and an answer state `y` (3 segments × 3 improvement steps × 7 core calls) **[Run]**. A
classification head reads the pooled `y` after every segment; all three segments are supervised
("deep supervision"). An auxiliary decoder reconstructs the input patch from `y`, and the
reconstruction error is added to the loss. Validation and the saved model use an exponential
moving average (EMA) of the weights **[Code]**.

The whole model has **446,409 parameters** at 32 bands and at 3 bands, plus 129,440 for the
reconstruction decoder at 32 bands **[Run]**.

### Inputs and outputs

| | Description | Evidence |
|---|---|---|
| Raw input | HistologyHSI-BC-Recurrence ENVI cubes + PNG renderings; PAD-UFES-20 JPG/PNG images + `metadata.csv` | [Code], [Artifact] |
| Prepared input (what training reads) | `X_{train,val,test}.npy` of shape `[N, H, W, C]`, `y_*.npy` of shape `[N]`, and `wavelengths.npy` for hyperspectral data | [Code], [Artifact] |
| Model output | Class logits `[B, K]` (K = 3 for histology, 6 for PAD-UFES-20) and, during training, a reconstruction `[B, C, H, W]` | [Run] |
| Run output | One directory per run under `experiments/`: checkpoints, `history.json`, `test_report.json`, predictions, plots, reconstruction samples | [Run] |

### Technologies

| Technology | Role | Evidence |
|---|---|---|
| Python 3.11 | language of the recorded environment | [Config/Doc] `mamba-spec.txt` |
| PyTorch (recorded runs: 2.11.0+cu128) | model, training, AMP, `torch.compile` | [Artifact] `config.json` |
| NumPy | `.npy` storage and memory-mapping | [Code] |
| scikit-learn | metrics, mutual information for band selection, shallow probes | [Code] |
| `spectral` | reading ENVI cubes (histology preparation only) | [Code] |
| pandas, Pillow | PAD-UFES-20 metadata and image decoding | [Code] |
| matplotlib | plots | [Code] |
| pytest | CPU test suite | [Code], [Run] |

The selective scan is written in pure PyTorch. `mamba_ssm` and custom CUDA kernels are **not**
used **[Code]** ([`medmamba_ss_trm.py:81`](medmamba_ss_trm.py#L81)).

### Overall workflow

1. **Prepare** each dataset once: raw files → `.npy` arrays with a patient-grouped split.
2. **Train** one run with `train.py`, or many with `run_experiments.py` and a sweep file.
3. **Evaluate**: every run evaluates its best checkpoint on the held-out test split automatically.
   Extra scripts re-evaluate, count FLOPs, export reconstructions and compare runs.
4. **Analyse**: scripts aggregate runs into the paper's tables and figures.

---

## 2. Architecture

```text
Raw data
  HistologyHSI-BC-Recurrence (ENVI cubes + PNGs)     PAD-UFES-20 (images + metadata.csv)
        │                                                      │
        ▼                                                      ▼
prepare_histologyhsi_bc.py                         prepare_pad_ufes_20_optimal.py / _original.py
  (HistologyHsiAdapterV8)                            (PadUfes20OptimalAdapter → PadUfes20Adapter)
        │                                                      │
        └──────────────► training/prep/core.py : run_prep ◄────┘
                         discover → split → Pass 1 → parallel extraction → shards → unify → verify
                                         │
                                         ▼
        Prepared dataset directory:  X_{train,val,test}.npy  y_*.npy  wavelengths.npy (HSI)
                                     class_names.json  groups_/captures_/images_*.npy  manifests
                                         │
                                         ▼
train.py ─► training/train_cli.py : resolve()      argparse + profile layers + run defaults
         ─► training/train_banner.py : print_banner()
         ─► training/train_pipeline.py : run() → _run()
               ├─ seeding, device, run directory  experiments/<run name>/
               ├─ dataset discovery, integrity validation, leakage check, class coverage
               ├─ normalization statistics (train split) → NpyDatasetV16 (train / val)
               ├─ gate G8 (split drift) → subsampling → augmentation → DataLoaders
               ├─ model: config_presets.build_model → MedMambaSSTRM (+ reconstruction wrapper)
               ├─ loss, AdamW (parameter groups), warmup-cosine LR schedule
               ├─ gates G1/G2/G9 (input sensitivity), G7 (reconstruction gradient), config.json
               └─ TrainerG(...).fit()
                     per epoch: deep-supervision training → EMA validation → metrics
                                → stopping rules → best-epoch selection → checkpoints, plots
               ├─ gate G5: reload best_model.pt, re-validate
               ├─ gate G6: evaluate best_model.pt on the test split → test_report.json
               └─ gates.json
                                         │
                                         ▼
Run directory ─► scripts/ (re-evaluation, FLOPs, latent space, reconstruction export,
                 band decimation, comparison, paper analysis)
              ─► run_experiments.py + sweeps/*.py (many runs, resumable)
```

### Components

| # | Component | File | Class / function | Receives | Produces | Called by | Calls next |
|---|---|---|---|---|---|---|---|
| 1 | Dataset adapters | [`prepare_histologyhsi_bc.py`](prepare_histologyhsi_bc.py), [`prepare_pad_ufes_20.py`](prepare_pad_ufes_20.py), [`prepare_pad_ufes_20_optimal.py`](prepare_pad_ufes_20_optimal.py) | `HistologyHsiAdapterV8`, `PadUfes20Adapter`, `PadUfes20OptimalAdapter` | raw files + CLI flags | items, labels, per-item patch streams | `main()` of each script | `run_prep` |
| 2 | Generic prep core | [`training/prep/core.py`](training/prep/core.py) | `run_prep` ([:273](training/prep/core.py#L273)) | an adapter | the prepared dataset directory | prep scripts | `splits`, `parallel`, `dataset_prep_unify` |
| 3 | CLI and profiles | [`training/train_cli.py`](training/train_cli.py), [`training/train_profiles.py`](training/train_profiles.py) | `resolve` ([:741](training/train_cli.py#L741)), `PROFILES` | `argv` | resolved `argparse.Namespace` + notes | `train.py:main` | banner, pipeline |
| 4 | Banner | [`training/train_banner.py`](training/train_banner.py) | `print_banner` | resolved args | console summary + estimates | `train.py:main` | — |
| 5 | Pipeline | [`training/train_pipeline.py`](training/train_pipeline.py) | `run`, `_run` ([:253](training/train_pipeline.py#L253)) | resolved args | a finished run directory | `train.py:main` | everything below |
| 6 | Data layer | [`training/npy_data.py`](training/npy_data.py) | `discover_data`, `NpyDatasetV16`, `compute_global_channel_stats` | data directory | `(x[C,H,W], label[, norm_stats])` items | pipeline | DataLoader |
| 7 | Normalization | [`training/normalization.py`](training/normalization.py) | `normalize_patch`, `denormalize` | one patch `[H,W,C]` | normalized patch + offset/scale | `NpyDatasetV16` | — |
| 8 | Augmentation | [`training/augmentation.py`](training/augmentation.py), [`training/augmentation_v16.py`](training/augmentation_v16.py) | `AugmentationConfig`, `AugmentedPatchDatasetV16` | normalized train patch | augmented patch | train DataLoader | — |
| 9 | Model | [`medmamba_ss_trm.py`](medmamba_ss_trm.py) | `MedMambaSSTRM` ([:1912](medmamba_ss_trm.py#L1912)) | `[B,C,H,W]` + wavelengths | per-segment logits, halting scores | trainer | — |
| 10 | Model factory | [`training/config_presets.py`](training/config_presets.py) | `build_model` ([:92](training/config_presets.py#L92)) | architecture, modality, classes, overrides | a model instance | pipeline, scripts | `MedMambaSSTRMConfig` |
| 11 | Reconstruction wrapper | [`training/reconstruction_head_v2.py`](training/reconstruction_head_v2.py) | `MedMambaSSTRMLatentReconWrapperV2` ([:81](training/reconstruction_head_v2.py#L81)) | base model | `(logits, x_recon)` | pipeline | decoder |
| 12 | Loss | [`training/losses.py`](training/losses.py) | `build_criterion` ([:115](training/losses.py#L115)) | loss type, train labels | `nn.Module` criterion | pipeline | — |
| 13 | Gates | [`training/gates.py`](training/gates.py) | `check_split_drift`, `run_sensitivity_check_with_margin`, `check_reconstruction_gradient` | data / one batch | pass/fail reports | pipeline | — |
| 14 | Trainer | [`training/trainerg.py`](training/trainerg.py) on [`training/trainerg_v12.py`](training/trainerg_v12.py) … [`trainerg_v4.py`](training/trainerg_v4.py) | `TrainerG` ([:65](training/trainerg.py#L65)), `fit` ([v12:1002](training/trainerg_v12.py#L1002)) | model, loaders, optimizer | history, checkpoints, reports | pipeline | evaluator |
| 15 | Test evaluator | [`training/evaluator.py`](training/evaluator.py), [`training/metrics.py`](training/metrics.py) | `evaluate_model`, `calibration_metrics` | best checkpoint, test loader | `test_report.json`, `test_predictions.npz` | `TrainerG.evaluate_test_split` | — |
| 16 | Sweep runner | [`run_experiments.py`](run_experiments.py), [`sweeps/`](sweeps/) | `train`, `cmd`, `main` | a sweep file + stage names | one subprocess per job, logs | user | `train.py`, scripts |
| 17 | Analysis tools | [`scripts/`](scripts/) | see [HOW_TO_RUN §7.7](HOW_TO_RUN.md#77-evaluation-and-analysis-of-finished-runs) | run directories | reports, tables, figures | user, sweeps | — |

---

## 3. Directory and file structure

The repository tracks **1,144 files**; 831 of them are under `paper/` **[Code]** (`git ls-files`).

| Path | Purpose | Important components | Used by |
|---|---|---|---|
| [`train.py`](train.py) | **Entry point: one training run** | `main` → `resolve`, `print_banner`, `run` | users, `run_experiments.py` |
| [`run_experiments.py`](run_experiments.py) | **Entry point: many runs** from a sweep file; resumable | `train()`, `cmd()`, `find_runs()`, `--list/--status/--dry_run` | users |
| [`sweeps/`](sweeps/) | sweep definitions (Python) | `paper_runs.py` (every run behind the paper), `paper_evidence_upgrade.py` (deferred evidence upgrade), `smoke.py` (GPU smoke check) | `run_experiments.py` |
| [`prepare_histologyhsi_bc.py`](prepare_histologyhsi_bc.py) | **Entry point: histology HSI + RGB preparation** | `HistologyHsiAdapterV8`, band selection, capture gain | users |
| [`prepare_histologyhsi_bc_trainsel.py`](prepare_histologyhsi_bc_trainsel.py) | the same preparation with band selection and gain reference computed on **training patients only** | `pass1` override | `sweeps/paper_evidence_upgrade.py` |
| [`prepare_pad_ufes_20_optimal.py`](prepare_pad_ufes_20_optimal.py) | **Entry point: PAD-UFES-20**, tuned protocol (whole 224×224 images, 70/15/15) | `TILINGS`, `PadUfes20OptimalAdapter` | users |
| [`prepare_pad_ufes_20_original.py`](prepare_pad_ufes_20_original.py) | PAD-UFES-20 under MedMamba's input protocol (60/10/30) | `--split_level` | users |
| [`prepare_pad_ufes_20.py`](prepare_pad_ufes_20.py) | base PAD adapter (imported by the two above) | `DIAGNOSTIC_TO_ID`, `PadUfes20Adapter` | the two scripts above |
| [`prepare_hmi_lusc_v20.py`](prepare_hmi_lusc_v20.py) | experimental second HSI dataset (HMI-LUSC); see [§5.1.3](#513-hmi-lusc-experimental) | `extract`, `split` sub-commands | `sweeps/paper_evidence_upgrade.py` |
| [`medmamba_ss_trm.py`](medmamba_ss_trm.py) | **Model file**: both models and the shared config (2,212 lines) | `MedMambaSSTRMConfig`, `SpectralPathway`, `MedMambaSS`, `MedMambaSSTRM` | `training/config_presets.py` |
| [`medmamba_ss_trm_ema.py`](medmamba_ss_trm_ema.py) | weight EMA (ported from the TRM reference code, MIT licence) | `EMAHelper` | `training/trainerg_v10.py` |
| [`medmamba_ss_fullchannel.py`](medmamba_ss_fullchannel.py), [`medmamba_ss_efficient.py`](medmamba_ss_efficient.py) | hierarchical model variants for ablations (`--architecture fullchannel / efficient`) | `MedMambaSSFullChannel`, `MedMambaSSEfficient` | `config_presets.py` |
| [`training/`](training/) | 79 top-level modules + `prep/` (8 files): pipeline, trainer chain, data, losses, metrics, gates, plots, baselines | see the component table in [§2](#2-architecture) | entry points |
| [`training/prep/`](training/prep/) | the dataset-agnostic preparation engine | `core.py`, `splits.py`, `parallel.py`, `manifest.py`, `shard_writer.py` | prep scripts |
| [`scripts/`](scripts/) | 26 evaluation and analysis tools | `eval_test_split.py`, `heldout_eval.py`, `flops_report.py`, `compare_runs.py`, `train_hsi_baseline.py`, `analysis_v19/v20.py`, … | users, sweeps |
| [`tests/`](tests/) | CPU test suite (47 test files + `conftest.py`) | parity tests against `archive/`, model, prep, gates | `pytest` |
| [`archive/`](archive/) | superseded entry points, kept byte-for-byte as the parity reference (22 files) | `train_example_v18.py` etc. | tests only |
| [`documentations/`](documentations/) | 19 detailed reference documents (some predate `train.py`; see its README) | `17_train_and_sweeps.md`, `18_top_runs.md`, `12_commands_datasets.md` | readers |
| [`doc/`](doc/) | presentation decks, speaker scripts, SVG figures and their generators | `build_pptx_v2.py`, `figures/` | presentations |
| [`paper/`](paper/) | manuscript sources, analysis JSON, figures, conference build | `source/`, `figures/`, `cvc2027/`, `draft/` | the paper |
| [`plan/`](plan/) | audit/remediation plans and prompts that drove each code version (28 files) | `*.plan.md` | history only |
| [`README.md`](README.md), [`PROJECT_HISTORY_AND_TECHNICAL_EVOLUTION.md`](PROJECT_HISTORY_AND_TECHNICAL_EVOLUTION.md), [`FUTURE_EXPERIMENTS.md`](FUTURE_EXPERIMENTS.md) | overview, history, deferred experiments | — | readers |
| [`requirements.txt`](requirements.txt), [`mamba-spec.txt`](mamba-spec.txt), [`environment_backup.yml`](environment_backup.yml), [`replicate_from_txt.txt`](replicate_from_txt.txt) | environment files (see [HOW_TO_RUN §3](HOW_TO_RUN.md#3-environment-setup) for their problems) | — | installation |
| [`pytest.ini`](pytest.ini), [`tests/conftest.py`](tests/conftest.py) | test configuration: collect only `tests/`, put the repo root on `sys.path` | — | `pytest` |

### Local folders that a clone does NOT contain

These are git-ignored **[Code]** (`.gitignore`) and exist only on the author's machine **[Artifact]**:

| Path | Contents | Size observed |
|---|---|---|
| `data/` | prepared datasets | 27 GB for the paper's histology build (HSI + RGB); 1.3 GB for `pad_optimal` |
| `experiments/` | one directory per training run | 11 GB for 144 runs; the reference run is 154 MB |
| `logs/` | sweep logs, `logs/<sweep>/<job>.log` | — |
| `.cache/compile/` | `torch.compile` / Triton caches (see [HOW_TO_RUN §9](HOW_TO_RUN.md#9-troubleshooting)) | 28 GB |
| `findings/` | measurement notes | — |

### Where to find each kind of file

| Kind | Location |
|---|---|
| Entry points | `train.py`, `run_experiments.py`, `prepare_*.py`, `scripts/*.py` |
| Configuration | `training/train_profiles.py` (profiles), `training/train_cli.py` (every flag and default), `training/config_presets.py` (model presets), `sweeps/*.py` |
| Model | `medmamba_ss_trm.py` (+ `_fullchannel`, `_efficient`), `training/reconstruction_head*.py`, `training/hsi_baselines.py` |
| Dataset code | `prepare_*.py`, `training/prep/`, `training/npy_data.py`, `training/normalization.py`, `training/augmentation*.py` |
| Training | `training/train_pipeline.py`, `training/trainerg*.py` |
| Evaluation | `training/evaluator.py`, `training/metrics.py`, `training/eval_utils.py`, `scripts/eval_test_split.py`, `scripts/heldout_eval.py` |
| Utilities | `training/npy_integrity.py`, `training/npy_atomic.py`, `training/system_memory.py`, `training/fd_limit.py`, `training/compile_cache.py`, `training/progress.py` |
| Outputs, checkpoints, logs | `experiments/<run>/` (checkpoints inside), `logs/<sweep>/` |
| Documentation | `README.md`, `documentations/`, this file |

---

## 4. End-to-end execution flow

This traces `python train.py --profile paper_recipe --data_dir <dir> [flags]` **[Code]**, confirmed
step by step by the synthetic CPU run **[Run]**.

| Step | What happens | Where |
|---|---|---|
| 1. Command | `train.py:main(argv)` | [`train.py:45`](train.py#L45) |
| 2. Argument parsing | `build_arg_parser()` builds every flag with its parser default; `_add_flags()` adds the run, stopping, schedule and reconstruction-artifact flags | [`train_cli.py:60`](training/train_cli.py#L60), [`:564`](training/train_cli.py#L564) |
| 3. Profile loading | `--profile` is read first, then its layers are applied as argparse defaults in order. `paper_recipe` = `OPTIMAL_DEFAULTS` → values resolved from the dataset (stem stride, subsample fractions, EMA rate; read from `.npy` headers only) → `--stage` → `RECON_DEFAULTS`. Typed flags always win | [`train_profiles.py:264`](training/train_profiles.py#L264), [`:231`](training/train_profiles.py#L231) |
| 4. Post-parse defaults | `--safe_mode` (if typed); architecture `recursive`; deep dataset validation for HSI; `--normalization auto` → `per_patch_zscore` (HSI) / `per_sample_minmax` (RGB); `--recon_out_activation auto` → `linear` for z-score, `sigmoid` for min-max | [`train_cli.py:517`](training/train_cli.py#L517), [`:710`](training/train_cli.py#L710) |
| 5. Banner | configuration, typed deviations from the profile, and estimates (constant-predictor bars, step budget, token cost, EMA horizon) | `print_banner` |
| 6. Trainer class | `TrainerG`, or `make_fast(TrainerG)` → `TrainerGFast` when `--fast_loop on` (paper default) | [`train_pipeline.py:255`](training/train_pipeline.py#L255) |
| 7. Seeding and environment | `seed_everything` (Python, NumPy, torch, CUDA); thread caps; open-file limit; `torch.compile` cache moved off a RAM-backed `/tmp` | [`:260`](training/train_pipeline.py#L260) |
| 8. Device | `cuda` if available, else `cpu`; TF32 on; `cudnn.benchmark` unless `--deterministic`; bf16 falls back to fp32 if unsupported | [`:272`](training/train_pipeline.py#L272) |
| 9. Run directory | `experiments/<run name>` **relative to the current working directory** | [`:291`](training/train_pipeline.py#L291), [`npy_data.py:163`](training/npy_data.py#L163) |
| 10. Dataset discovery | finds `X_/y_{train,val,test}.npy`, `wavelengths.npy`, `class_names.json`; number of classes = max label + 1 | [`npy_data.py:133`](training/npy_data.py#L133) |
| 11. Integrity | manifest cross-check; `.npy` validation (structural, or deep = full read); `DATASET_STORAGE_ERROR` aborts | [`:306`](training/train_pipeline.py#L306) |
| 12. Leakage + coverage | patient/patch leakage check; every class present in train and validation; imbalance report | [`:318`](training/train_pipeline.py#L318) |
| 13. Normalization statistics | `global_zscore` + `train_fit`: per-channel mean/std from ≤ 5,000 training patches (seeded) | [`:330`](training/train_pipeline.py#L330), [`npy_data.py:36`](training/npy_data.py#L36) |
| 14. Datasets | `NpyDatasetV16` for train and validation; a third item `norm_stats` when reconstruction is on | [`:340`](training/train_pipeline.py#L340) |
| 15. Gate G8 | post-normalization drift of validation/test vs train (8,000 patches per split) | [`:362`](training/train_pipeline.py#L362) |
| 16. Subsampling | validation: a fixed subset, stratified by patient and class; training: a fresh random subset every epoch | [`:370`](training/train_pipeline.py#L370), [`:402`](training/train_pipeline.py#L402) |
| 17. Augmentation | training split only | [`:428`](training/train_pipeline.py#L428) |
| 18. DataLoaders | worker policy from `--loader_mode` | [`:442`](training/train_pipeline.py#L442) |
| 19. Preflight | prints dataset, loader, numerical and gate settings; `--loader_test` exits here | [`:451`](training/train_pipeline.py#L451) |
| 20. Model | `build_model_guarded` → `MedMambaSSTRM`; optional gradient checkpointing, spectral checkpointing (auto at C ≥ 16), `torch.compile` | [`:463`](training/train_pipeline.py#L463) |
| 21. Reconstruction wrapper | `recon_mode latent` → `MedMambaSSTRMLatentReconWrapperV2` | [`:507`](training/train_pipeline.py#L507) |
| 22. Loss, optimizer, scheduler | `build_criterion`; AdamW with weight-decay groups; warmup + cosine over `epochs × steps_per_epoch` | [`:524`](training/train_pipeline.py#L524) |
| 23. Gates G1/G2/G9 and G7 | the model must respond to its input; reconstruction gradient must reach the recursive core | [`:540`](training/train_pipeline.py#L540) |
| 24. `config.json` | every resolved argument + run metadata | [`:572`](training/train_pipeline.py#L572) |
| 25. Trainer | `TrainerG(...)`; resume if `--resume` | [`:596`](training/train_pipeline.py#L596) |
| 26. Training loop | `trainer.fit(max_epochs)` | [`trainerg_v12.py:1002`](training/trainerg_v12.py#L1002) |
| 27. Gate G5 | reload `best_model.pt`, re-validate, compare with the logged best | [`trainerg_v11.py:733`](training/trainerg_v11.py#L733) |
| 28. Gate G6 (testing) | evaluate `best_model.pt` on the full test split; one single-process retry on file-descriptor exhaustion | [`:636`](training/train_pipeline.py#L636) |
| 29. Results | `gates.json`; exit with an error for numerical-instability or collapse aborts; `failure_class.json` on any other crash | [`:671`](training/train_pipeline.py#L671), [`:681`](training/train_pipeline.py#L681) |

---

## 5. Data pipeline

### 5.1 Required datasets

**The repository contains no data** **[Config/Doc]** (README). Download each dataset from its
official source and accept its terms (README, "Third-party material and licences").

#### 5.1.1 HistologyHSI-BC-Recurrence (primary; hyperspectral + RGB)

- **Source:** The Cancer Imaging Archive, DOI 10.7937/6KPY-YT49 **[Config/Doc]**. Size about
  1.2 TB (README) **[Config/Doc]**.
- **Expected raw layout** (verified on the author's copy **[Artifact]**; discovery rules **[Code]**):

```text
<RAW_HSI_ROOT>/
├── 02_01_HS_Images/                      # "02_01_HSI_Images" is also accepted
│   ├── DCIS/      HS_VNIR_<patient>_DCIS_x10_C<nn>/
│   ├── Healthy/   HS_VNIR_<patient>_Healthy_x10_C<nn>/
│   └── IDC/       HS_VNIR_<patient>_IDC_x10_C<nn>/
│        each capture folder:
│          calibrated.hdr / calibrated.dat          <- the cube that is read
│          raw.hdr / raw.dat
│          whiteReference.hdr / .dat, darkReference.hdr / .dat
│          SyntheticRGBImage.png                    <- default RGB source
│          RGBImage.png
├── 01_03_HSI ROI_Annotations/            # <patient>.geojson (note the space in the name)
└── HistologyHSI-BRCA-Recurrence-Clinical-Standardized.xlsx
```

- **Counts [Artifact]:** 644 captures from 45 patients (IDC 425, Healthy 184, DCIS 35 capture
  folders). Cubes have 740 bands from 400.48 to 938.16 nm (`wavelengths_full.npy`).
- **Which cube is read [Code]:** the `.hdr` whose name contains `calib` is preferred, and no
  flat-field calibration is applied to it ([`prepare_histologyhsi_bc.py:161`](prepare_histologyhsi_bc.py#L161), [`:240`](prepare_histologyhsi_bc.py#L240)).
- **Labels [Code]:** `--label_source tissue` (default): healthy = 0, DCIS = 1, IDC = 2
  (`class_names.json`).
  `--label_source recurrence` exists, but the automatic workbook lookup expects
  `HistologyHSI-BC-Recurrence-Clinical-Standardized.xlsx` ([`:1065`](prepare_histologyhsi_bc.py#L1065)), while the
  downloaded file is named `HistologyHSI-BRCA-…` **[Artifact]**. Whether recurrence labels work
  at all is **[NOT VERIFIED]**; `documentations/13_additional_analyses.md` describes them as blocked.
- **Split [Code], [Artifact]:** grouped by patient (no patient in two splits), stratified by each
  patient's rarest class, seeded. The paper build used `--split 80_10_10 --seed 42`:
  **35 / 5 / 5 patients → 2,452,086 / 334,516 / 348,894 patches**, identical counts for `hsi/` and `rgb/`.

#### 5.1.2 PAD-UFES-20 (RGB skin-lesion photographs)

- **Source:** Mendeley Data, DOI 10.17632/zr7vgbcyr2.1 **[Config/Doc]**.
- **Expected raw layout [Code]** ([`prepare_pad_ufes_20.py:86`](prepare_pad_ufes_20.py#L86)):
  `metadata.csv` in the root (if absent, the first `.csv` found) with columns `img_id`,
  `patient_id`, `diagnostic`. Image files (`.png/.jpg/.jpeg/.bmp/.tif/.tiff`) may be anywhere
  below the root; they are matched to `img_id` by file stem.
- **Classes [Code]:** ACK = 0, BCC = 1, MEL = 2, NEV = 3, SCC = 4, SEK = 5.
- **Counts [Artifact]:** 2,298 images from 1,373 patients.

| Build | Split | Patients train / val / test | Images train / val / test |
|---|---|---|---|
| `data/pad_optimal` (`prepare_pad_ufes_20_optimal.py`) | 70/15/15 | 961 / 206 / 206 | 1,626 / 328 / 344 |
| `data/pad_original` (`prepare_pad_ufes_20_original.py`) | 60/10/30 | 824 / 137 / 412 | 1,384 / 245 / 669 |

#### 5.1.3 HMI-LUSC (experimental)

`prepare_hmi_lusc_v20.py` prepares a second hyperspectral histology dataset (lung squamous cell
carcinoma, 61 bands, 450–750 nm, two classes: non-tumour / tumour) with patient-level 5-fold
splits **[Code]** (module docstring). `FUTURE_EXPERIMENTS.md` records that its training runs
stopped early without test reports **[Config/Doc]**. The script is tracked but also listed in
`.gitignore`; the intent of that entry is **[NOT VERIFIED]**. It is not part of the paper's results.

#### 5.1.4 The prepared-dataset contract

What `train.py` reads from `--data_dir` **[Code]** ([`npy_data.py:133`](training/npy_data.py#L133),
[`train_pipeline.py:159`](training/train_pipeline.py#L159)):

| File | Shape / type | Required? | Effect |
|---|---|---|---|
| `X_train.npy`, `y_train.npy` | `[N, H, W, C]` (channels last), `[N]` int64 | **required** | training split |
| `X_test.npy`, `y_test.npy` | same | **required** | test split (gate G6) |
| `X_val.npy`, `y_val.npy` | same | optional | if absent, the **test split is used for validation** and the test evaluation is skipped |
| `wavelengths.npy` | `[C]` float32, nm | optional | **its presence alone makes the run "hsi"**; absent → "rgb" and index-based band encoding |
| `class_names.json` | list of names | optional | names in reports |
| `groups_*.npy`, `captures_*.npy`, `capture_index.json` | `[N]` int32 | optional | patient-stratified validation subset, per-patient metrics, patient names in gate G8 |
| `images_*.npy` | `[N]` | optional | image-level test aggregation (`--eval_group_aggregation`) |
| `dataset_manifest.json` | JSON | optional (`--manifest_check strict` requires it) | shape/dtype/size cross-check |
| `_progress.json` (here or in the parent) | JSON | optional | patient-level leakage check |

Recorded paper build **[Artifact]** (`data/hsi_v8-80_10_10_importance-new/`):

| | `hsi/` | `rgb/` |
|---|---|---|
| `X_train.npy` | `[2452086, 11, 11, 32]` float16 (19.0 GB) | `[2452086, 11, 11, 3]` float32 |
| `X_val.npy` | `[334516, 11, 11, 32]` float16 | `[334516, 11, 11, 3]` float32 |
| `X_test.npy` | `[348894, 11, 11, 32]` float16 | `[348894, 11, 11, 3]` float32 |
| `wavelengths.npy` | 32 values, 400.5–938.2 nm, with a gap between 633.3 and 852.3 nm | — (absent) |

`data/pad_optimal/` is flat (no `hsi/`/`rgb/` subfolder): `X_*.npy` of whole 224 × 224 × 3
images in [0, 1] **[Code]**, plus `images_*.npy` **[Artifact]**.

---

### 5.2 Data loading

| Step | Implementation | Detail |
|---|---|---|
| 1. File discovery | `discover_data` ([`npy_data.py:133`](training/npy_data.py#L133)) | returns paths, wavelengths, number of classes, class names, split sizes |
| 2. Storage mode | `NpyDataset.__init__` → `resolve_storage_mode` | `--dataset_storage auto`: load into RAM if the array fits comfortably, else memory-map **[Code]**, **[Run]** (`auto -> ram` on the tiny set) |
| 3. Per item | `NpyDatasetV16.__getitem__` ([`:221`](training/npy_data.py#L221)) | copy as float32 → `nan_to_num` (NaN → 0, +inf → 1, −inf → 0) → normalize → `HWC → CHW` |
| 4. Item returned | | `(x [C,H,W] float32, label int)` or, with reconstruction on, `(x, label, norm_stats [2,C])` |
| 5. Batch | PyTorch `DataLoader` | `x [B,C,H,W]`, `y [B]` int64, `norm_stats [B,2,C]`; on CUDA, `x` is moved to `channels_last` memory format |

Dimensions: **B** = batch size; **C** = channels (spectral bands: 32 for the paper's HSI, 3 for
RGB); **H, W** = patch height and width (11 × 11 for histology, 224 × 224 for whole PAD images).

Verified shapes **[Run]**: a paper HSI batch is `[256, 32, 11, 11]`; the RGB arm `[256, 3, 11, 11]`.

---

### 5.3 Preprocessing

#### Offline (once, during preparation) — histology, in execution order

| # | Step | Implementation | Parameters (paper build) |
|---|---|---|---|
| 1 | Discover captures, assign labels | `discover_captures`, `build_labels`, `HistologyHsiAdapter.discover` ([:1091](prepare_histologyhsi_bc.py#L1091)) | `--label_source tissue` |
| 2 | Patient-grouped stratified split, written to `split_assignment.json` | `training/prep/splits.py` → `stratified_group_split_three` | `--split 80_10_10 --split_strategy stratified --seed 42` |
| 3 | Class-coverage check | `check_prep_class_coverage` | aborts on a missing class unless `--allow_missing_classes` |
| 4 | **Pass 1: band selection** on a 10 % sample of captures. Importance = 0.5 × normalized variance + 0.5 × normalized mutual information with the label; ranking decorrelated (minimum index gap, maximum correlation); top N kept; must span ≥ 30 % of the range | `compute_band_statistics`, `compute_band_importance` ([:427](prepare_histologyhsi_bc.py#L427)), `select_bands_by_importance` ([:521](prepare_histologyhsi_bc.py#L521)) | `--band_selection importance --num_bands 32 --band_min_gap 8 --band_max_corr 0.95 --band_min_coverage 0.30` |
| 5 | **Pass 1: capture gain.** Median intensity of 400 sampled pixels per capture; gain = corpus median / capture median, clipped to [0.5, 2.0]; clipped captures listed as suspect | `HistologyHsiAdapterV8.pass1` ([:1503](prepare_histologyhsi_bc.py#L1503)) | `--capture_gain median_ratio` |
| 6 | **Pass 1: value scale.** `auto` = 99th percentile of the gain-corrected sample; decides float16 vs float32 storage | same | `--hsi_value_scale auto` → resolved to **8618.75** **[Artifact]** |
| 7 | ROI mask from GeoJSON (patch kept only if ≥ `roi_min_frac` inside the ROI) — **see the warning below** | `find_roi_geojson` ([:777](prepare_histologyhsi_bc.py#L777)), `patch_coords` ([:829](prepare_histologyhsi_bc.py#L829)) | `--roi_min_frac 0.8` |
| 8 | Patch extraction on a grid, in parallel workers | `load_item_multi` ([:1217](prepare_histologyhsi_bc.py#L1217)) | `--patch_size 11 --stride 11 --num_workers 8` |
| 9 | Per HSI patch: keep the selected bands → × capture gain → ÷ value scale | `_hsi_stream` + V8 `_corrected` | — |
| 10 | Per RGB patch: `SyntheticRGBImage.png`, nearest-neighbour resized to the cube size if needed, **same patch coordinates** as HSI, raw 0–255 floats | `_rgb_stream`, `make_rgb_finder` ([:1382](prepare_histologyhsi_bc.py#L1382)) | `--rgb_source synthetic` (no `--rgb_normalize`) |
| 11 | Shards → unify into `X_*.npy` / `y_*.npy` (rows sorted by item key), deep read-back verification | `training/prep/core.py`, `dataset_prep_unify.py` | `--verify_level deep` |
| 12 | Finalize: `wavelengths.npy`, `wavelengths_full.npy`, `class_names.json`, group sidecars, manifests, reports | `finalize_modality` | `--emit_group_sidecars` (default on) |

> **Warning — band selection saw held-out labels [Code].** `run_prep` calls `adapter.pass1(items, args)`
> with **all** captures, including validation and test patients ([`training/prep/core.py:346`](training/prep/core.py#L346)).
> The README calls the hyperspectral numbers "provisional" for this reason. The fix exists as a
> separate script, [`prepare_histologyhsi_bc_trainsel.py`](prepare_histologyhsi_bc_trainsel.py) **[Code]**,
> but the paper's numbers come from the leaky build **[Config/Doc]** (README, `FUTURE_EXPERIMENTS.md` F1).

> **Warning — ROI filtering silently does nothing on this download [Code], [Artifact].**
> `find_roi_geojson` looks for a file whose name contains `<patient>_<tissue>_x10_C<capture>`, but the
> annotation folder holds per-patient files named `<patient>.geojson` (e.g. `100.geojson`). No file
> ever matches, so `mask` is `None` and every grid position is kept. Consistent with this, the paper
> build kept 3,135,496 of a maximum 3,164,616 positions (644 × 54 × 91) **[Artifact]**. The prep log
> does not report this. The manuscript audit in `paper/draft/MANUSCRIPT_AUDIT_v18_r2.md` records
> the same finding **[Config/Doc]**.

#### Offline — PAD-UFES-20

Per image **[Code]** ([`prepare_pad_ufes_20.py:149`](prepare_pad_ufes_20.py#L149)): EXIF orientation fix → RGB →
bilinear resize to 224 × 224 → float32 in [0, 1]. `--tiling whole` (default) stores one sample
per image; `mil9` / `mil4` store 112 × 112 tiles plus `images_*.npy`. No class balancing
(`--balance_classes none`).

#### Online (every time a sample is loaded), in execution order

| # | Step | Implementation | Train | Val | Test |
|---|---|---|---|---|---|
| 1 | `nan_to_num` | `NpyDatasetV16.__getitem__` | ✓ | ✓ | ✓ |
| 2 | Normalization. Paper and PAD profiles: `global_zscore`, per-channel mean/std fitted on ≤ 5,000 training patches. Others: `per_patch_zscore` (whole-patch mean/std), `per_sample_minmax`, or fixed 0.5/0.5 (`--global_stats medmamba_fixed`) | [`normalization.py:55`](training/normalization.py#L55) | ✓ | ✓ | ✓ |
| 3 | Augmentation (applied **after** normalization, to the `[C,H,W]` tensor) in this order: Gaussian noise (std 0.02), one scale factor per band (±15 %), one offset per band (std 0.05), band dropout (off), contiguous spectral mask (off), horizontal / vertical flip (p = 0.5 each), 90° rotation (p = 0.5), random crop of up to 30 % per side resized back with nearest neighbour. Deterministic per (seed, epoch, index, worker) | [`augmentation.py:51-153`](training/augmentation.py#L51), `OPTIMAL_DEFAULTS` | ✓ | — | — |
| 4 | Training subsample: a fresh random 10.2 % of the training split every epoch (paper HSI; resolved automatically to about 250,000 patches) | [`train_pipeline.py:411`](training/train_pipeline.py#L411) | ✓ | — | — |
| 5 | Validation subsample: a fixed 9.86 % drawn within each patient and class (seed + 1) → 32,985 patches **[Artifact]** | [`:376`](training/train_pipeline.py#L376) | — | ✓ | — |
| 6 | Spatial pooling to tokens ("patchify", average pooling by the stem stride: 1 for 11 × 11, 8 for 224 × 224) — inside the model | `SpectralPathway.patchify` ([`medmamba_ss_trm.py:1036`](medmamba_ss_trm.py#L1036)) | ✓ | ✓ | ✓ |

The test split is always evaluated in full **[Code]**.

---

## 6. Model and algorithms

### 6.1 Model overview

| Item | Value | Evidence |
|---|---|---|
| Name | **MedMamba-SS-TRM** (`MedMambaSSTRM`); `--architecture recursive` | [Code] |
| Purpose | patch classification with an auxiliary reconstruction of the input | [Code] |
| Input | `x [B, C, H, W]`, any C; optional `wavelengths [C]` (nm) and `sensor_range` | [Code] |
| Output | logits `[B, K]` from the last segment; during training, per-segment logits and halting scores, plus `x_recon [B, C, H, W]` from the wrapper | [Run] |
| Classes | 3 (histology), 6 (PAD-UFES-20); taken from the labels | [Code] |
| Parameters | 446,409 (C = 32 or 3, K = 3); 446,796 (K = 6); reconstruction decoder +129,440 (C = 32) / +112,707 (C = 3) | [Run] |
| Main parts | spectral pathway → stem + 2-D positional encoding → recursive core → recursive head; latent reconstruction decoder | [Code] |

The same file also holds **MedMamba-SS** (`MedMambaSS`, `--architecture split`), the hierarchical
four-stage model, and the repository has two hierarchical variants (`fullchannel`, `efficient`)
and two hyperspectral baselines (HybridSN, SpectralFormer; [`training/hsi_baselines.py`](training/hsi_baselines.py)) **[Code]**.

Parameter breakdown at C = 32, K = 3 **[Run]**:

| Part | Parameters |
|---|---|
| Spectral pathway | 38,724 |
| Stem (linear) + stem norm | 8,320 + 256 |
| Recursive core (2 blocks, reused 63×) | 398,336 |
| Head (classifier + halting) | 772 |
| **Model total** (`backbone_num_params` in `config.json`) | **446,409** |
| Reconstruction decoder (only with `recon_mode latent`) | 129,440 |

### 6.2 Layer-by-layer architecture

Paper configuration, HSI (C = 32, K = 3). N = B × 121 (one row per pixel position). Every shape
below was recorded with forward hooks **[Run]**.

| # | Component | Input shape | Operation | Output shape | Purpose |
|---|---|---|---|---|---|
| 1 | Input | — | normalized patch batch | `[B, 32, 11, 11]` | |
| 2 | `SpectralPathway.patchify` | `[B, 32, 11, 11]` | average pool by the stem stride (1 here = identity) | `[B, 32, 11, 11]` | sets the token grid |
| 3 | Flatten positions | `[B, 32, 11, 11]` | permute + reshape | `[N, 32]` | one band sequence per position |
| 4 | `SpectralTokenizer` (`concat_mlp`, d_token = 32) | `[N, 32]` + wavelengths | per band: concatenate value with a sinusoidal wavelength encoding → Linear(33→64) → GELU → Linear(64→32), plus 0.1 (learnable) × encoding | `[N, 32, 32]` | one token per band; no parameter depends on C |
| 5 | `HierarchicalSpectralEncoder` (depth 3) | `[N, 32, 32]` | 1 residual block (LayerNorm → Conv1d k = 3 along bands → GELU → Linear) → residual `SpectralMamba` (bidirectional selective scan along bands, state size 8) → 2 residual blocks | `[N, 32, 32]` | model dependencies between bands |
| 6 | `BandGate` (soft) | `[N, 32, 32]` | sigmoid(Linear(32→1)) per band; tokens × weights | weights `[N, 32]` | learned band weighting |
| 7 | Band pooling | `[N, 32, 32]` | mean over bands | `[N, 32]` | removes C from the shape |
| 8 | `ProgressiveCompressor` | `[N, 32]` | Linear 32→128 → GELU → 128→64 → GELU → 64→64 → GELU | `[N, 64]` | spectral context |
| 9 | Context map | `[N, 64]` | reshape | `[B, 11, 11, 64]` | |
| 10 | Context normalization | same | parameter-free, scale-invariant RMS normalization | same | keeps the signal above LayerNorm's epsilon |
| 11 | Stem | `[B, 11, 11, 64]` | Linear(64→128) → LayerNorm(128) | `[B, 11, 11, 128]` | working width d = 128 |
| 12 | 2-D positional encoding | `[B, 11, 11, 128]` | + 0.1 (learnable) × fixed sinusoidal encoding | `x_emb [B, 11, 11, 128]` | spatial position |
| 13 | Recursive core (`RecursiveCore`) | `x_emb`; states `y`, `z` initialised from fixed buffers | per segment: 3 improvement steps (the first 2 without gradient); each step updates `z ← f(z + y + x)` six times, then `y ← f(y + z)`. `f` = 2 × `RecursiveMambaBlock`: [depthwise 3 × 3 conv + gated MLP] then gated MLP, each with residual + RMS norm. 3 segments, detached between segments → **63 calls of `f`** | `y [B, 11, 11, 128]` | weight-shared refinement |
| 14 | `RecursiveHead` (after every segment) | `y` | LayerNorm → mean over 11 × 11 → dropout 0.1 → Linear(128→3); halting head Linear(128→1) | logits `[B, 3]`, `q [B]` | classification; `q` is unused while halting is off |
| 15 | Output | 3 × logits | training: all three supervised; inference: last segment | `[B, 3]` | |
| 16 | `LatentReconstructionDecoderV2` (auxiliary) | `y [B, 11, 11, 128]` | permute → bilinear resize to input H × W → Conv3×3(128→64) + BatchNorm + LeakyReLU → Conv3×3(64→64) + BatchNorm + LeakyReLU → Conv3×3(64→32), linear output | `x_recon [B, 32, 11, 11]` | reconstruction loss; reads the live `y`, so its gradient reaches the core (gate G7) |

For the RGB arm (C = 3) only the band axis shrinks: the tokenizer outputs `[N, 3, 32]` and every
shape from step 7 on is identical **[Run]**. For 224 × 224 PAD images the stride resolves to 8,
giving a 28 × 28 token grid **[Code]** ([`train_profiles.py:180`](training/train_profiles.py#L180)).

### 6.3 Important algorithms

#### Selective scan (Mamba-style state-space recurrence) [Code]

[`_selective_scan_pure_pytorch`](medmamba_ss_trm.py#L81), a Python loop over the sequence length L. For each
step l, with input `u_l`, step size `Δ_l = softplus(·)`, `A = −exp(A_log)` and input-dependent
`B_l`, `C_l`:

```text
h_l = exp(Δ_l · A) ⊙ h_{l−1} + Δ_l · B_l · u_l
y_l = C_l · h_l + D · u_l
```

`SpectralMamba` runs it forward and backward along the band axis and mixes the two directions with
learned softmax weights ([`:644`](medmamba_ss_trm.py#L644)). The hierarchical model also runs it spatially (`SS2D`,
4 or 8 scan directions). In MedMamba-SS-TRM the spatial mixer is `mlp` by default; `--trm_mixer ss2d`
is available but its help text records it as 39× slower **[Config/Doc]**.

#### Wavelength-aware band encoding [Code]

([`:685-721`](medmamba_ss_trm.py#L685)) Wavelengths are normalized to [0, 1] by their own min and max (or by
`--sensor_range`). Then `pe = sin/cos(λ̃ · η · ω_i)` with `η = C` (`--wavelength_scale` unset).
With `η = C` this equals the index-based encoding for evenly spaced bands. Without
`wavelengths.npy` (RGB) the index-based encoding is used. `--no_use_wavelengths` forces the index
encoding on HSI.

#### TRM recursion and deep supervision [Code]

([`RecursiveCore`, :1711](medmamba_ss_trm.py#L1711); [`forward_deep_supervision`, :1942](medmamba_ss_trm.py#L1942))
- Defaults (CLI): `--trm_n_latent 6 --trm_n_improve 3 --trm_deep_supervision_steps 3 --trm_core_layers 2 --trm_dim 128`.
- Applications of `f` per forward = (n_latent + 1) × n_improve × segments = 7 × 3 × 3 = 63 **[Run]**.
- Only the last improvement step of each segment carries gradient; the states are detached
  between segments. Each call of `f` is gradient-checkpointed (`--trm_checkpoint_core`, on by default).
- **Adaptive halting** (`--trm_halting`) is **off** in every profile; the halting head then gets
  no loss term and all 3 segments always run.

#### Exponential moving average (EMA) of weights [Code]

`EMAHelper` ([`medmamba_ss_trm_ema.py:51`](medmamba_ss_trm_ema.py#L51)): `shadow = μ · shadow + (1 − μ) · w` after every
optimizer step (`--fast_loop on` uses a fused version, documented as not bit-identical, ~4e-7
**[Run]** log message). The rate is resolved from the data so the average spans about 3 epochs,
clipped to [0.90, 0.9995]: **0.9995** for the paper HSI runs, **0.99346** for `data/pad_optimal`
**[Run]**. **Validation and `best_model.pt` use the EMA weights** **[Code]**.

#### Loss function [Code]

Training loss per batch ([`trainerg_v12.py:384-409`](training/trainerg_v12.py#L384)):

```text
L = mean over the 3 segments of  criterion(logits_s, y)
  + 0.5 · halting BCE          (only with --trm_halting)
  + λ_mse · MSE(x_recon, x)    (normalized units)
  + λ_sam · SAM(x, x_recon)    (spectral angle, radians)
  + λ_gan · GAN loss           (only with --use_gan or λ_gan > 0)
```

Paper values: `focal_weighted`, γ = 1.5, class-weight power 0.75, λ_mse = 0.1, λ_sam = 0.1
(typed explicitly), λ_gan = 0.

- **Focal loss** ([`losses.py:89`](training/losses.py#L89)): `FL = −w_y · (1 − p_y)^γ · log p_y`, averaged over the batch.
- **Class weights** ([`losses.py:30`](training/losses.py#L30)): `w_c ∝ (n / (K · n_c))^p`, then rescaled to
  mean 1. With p = 0.75 the correction is softer than full inverse frequency. `--weight_method`
  has no effect (both options give identical weights, stated in the help text).
- **SAM** ([`gan.py:69`](training/gan.py#L69)): mean angle between the true and reconstructed spectrum of each pixel.

#### Regularization [Code]

AdamW weight decay 0.05 applied only to ≥ 2-D weights that are not normalization parameters
(biases, norms, gains, and the tokenizer's fusion layers are excluded; [`optim_groups.py:83`](training/optim_groups.py#L83));
classifier dropout 0.1; gradient-norm clipping at 1.0; the augmentation above. `trm_dropout`,
`trm_drop_path` and `drop_path_rate` are 0 in the paper profile.

#### Numerical-stability controller [Code]

A batch with a non-finite loss or gradient is skipped instead of applied. The run aborts with
`TRAINING_ABORTED_NUMERICAL_INSTABILITY` if more than 10 % of an epoch's batches are skipped, after
3 consecutive bad batches, after 2 consecutive unhealthy epochs, or if parameters become non-finite
(`--max_gradient_skip_ratio`, `--max_consecutive_bad_batches`, `--max_consecutive_unhealthy_epochs`).

#### Gates (automatic checks) [Code]

| Gate | Checks | When | On failure |
|---|---|---|---|
| G1 / G2 / G9 | the stem and the logits respond to the input; two different batches give different mean logits | before training, one batch | abort `REPRESENTATION_COLLAPSE` |
| G7 | the reconstruction loss sends gradient into the recursive core | before training (recursive + latent reconstruction) | abort `RECONSTRUCTION_DETACHED` |
| G8 | validation/test mean within ±0.25 train-σ and std ratio in [0.75, 1.25] after normalization | before training | `--on_split_drift abort / warn / off` (paper: warn) |
| Class collapse | one predicted label across all validation patches for N epochs | every epoch | `--on_class_collapse abort / warn` (paper: warn, N = 10) |
| G5 | `best_model.pt` reproduces the logged best metric within 1e-6 | after training | recorded in `gates.json` |
| G6 | test-split evaluation of `best_model.pt` succeeded | after training | recorded; `gate_g6_failure.json` |

Recorded margins on the synthetic run **[Run]**: G2 23.4×, G1 11.9×, G9 580×.

---

## 7. Training workflow

Paper configuration on the paper's HSI build unless stated. Recorded values from the reference run
`20260915_031356_…_ref20` are marked **[Artifact]**.

| # | Topic | How it works |
|---|---|---|
| 1 | Dataset initialization | `NpyDatasetV16` for train and validation, with `global_zscore` statistics from ≤ 5,000 training patches (seeded) **[Code]**. Statistics are **not saved** in the run directory; tools recompute them from `X_train.npy` **[Code]**. |
| 2 | DataLoader creation | batch 256 (typed); `--loader_mode performance` = 4 workers, prefetch 2, persistent, pinned memory; validation uses one worker fewer and is never persistent **[Code]**. Training shuffles through a `RandomSampler` that draws 250,113 of 2,452,086 patches per epoch **[Artifact]**. |
| 3 | Model initialization | `build_model("recursive", modality, K, …)` with the v15 representation overrides, then the reconstruction wrapper **[Code]**. `--activation leakyrelu` replaces 0 modules on this architecture (it has no `nn.ReLU`) **[Run]**. |
| 4 | Device | CUDA if available, else CPU **[Code]**. Recorded: `cuda`, NVIDIA GeForce RTX 5060 Ti **[Artifact]**. |
| 5 | Loss | focal loss with class weights (power 0.75), mean over 3 segments, + 0.1 MSE + 0.1 SAM **[Code]** |
| 6 | Optimizer | `torch.optim.AdamW(build_param_groups(model, 0.05), lr=3e-4)`; betas and eps are PyTorch defaults **[Code]** |
| 7 | LR scheduler | linear warmup for max(200, round(3 % × total)) steps (capped at total − 1), then cosine decay to 0, stepped **after every optimizer step**. Total = epochs × ceil(kept / batch). Recorded: total 19,560, warmup 587 for 20 epochs × 978 steps **[Artifact]** |
| 8 | Mixed precision | `--amp bf16` autocast for training **and** validation; the `GradScaler` is enabled only for fp16 **[Code]**. The test evaluation (G6) runs **without autocast**, i.e. in fp32 **[Code]** ([`evaluator.py:27`](training/evaluator.py#L27)) |
| 9 | Forward pass | `forward_deep_supervision_with_features`: per-segment logits plus the **live** final feature map, which the decoder reconstructs from **[Code]** |
| 10 | Loss calculation | as in [§6.3](#loss-function-code); a non-finite total skips the batch **[Code]** |
| 11 | Backpropagation | `loss.backward()`; each core call is recomputed (checkpointing) **[Code]** |
| 12 | Gradient handling | finiteness check; `clip_grad_norm_(…, 1.0)`; the logged `gradient_norm` is the pre-clip norm **[Code]** |
| 13 | Parameter update | `optimizer.step()` → `scheduler.step()` → EMA update → parameter finiteness check **[Code]** |
| 14 | Validation | every epoch, on a deep copy holding the EMA weights, over the 32,985-patch validation subset: loss, accuracy, balanced accuracy, macro/weighted precision/recall/F1, per-class arrays, and reconstruction metrics in reflectance units (RMSE, MAE, SAM, SID, PSNR, SSIM, …) **[Code]**, **[Artifact]** |
| 15 | Checkpointing | every epoch: `checkpoints/epoch_NNNN.pt` (last 3 kept), `latest.pt`, `latest_ema.pt`; on a new best: `best_model.pt` (**EMA** weights), `best_model_live.pt`, `best_model_ema.pt` (bare EMA shadow) **[Code]**, **[Run]** |
| 16 | Logging | console (`TrainerG_v4` logger to stdout + progress lines); `history.json` / `history.csv` every epoch; `loss_components.*`, `spectral_metrics.csv`; plots every 5 epochs and at the end (`--eval_artifact_stride 5`) **[Code]**, **[Run]** |
| 17 | Early stopping | **off** in the paper profile: `--early_stopping off` (the patience of 40 is recorded but not enforced), `--val_divergence_patience 0`, metric-stall rule suppressed. Runs therefore stop only at `--epochs`, on a numerical abort, or on a class-collapse abort **[Code]**. All ten top runs ended `TRAINING_COMPLETED` **[Config/Doc]** (`documentations/18_top_runs.md`) |
| 18 | Final model | `best_model.pt` = EMA weights of the epoch with the highest validation macro-F1 (strictly greater; invalid epochs excluded) **[Code]**. Reference run: epoch 7 of 20 **[Artifact]**. `experiment_report.json` records the stop reason |

### The training loop, step by step [Code]

```text
for epoch in 1..epochs:                                   # TrainerG_v12.fit
    set augmentation epoch; remember parameters
    for (x, y, norm_stats) in train_loader:               # _train_one_epoch_deep_supervision
        zero gradients
        with autocast(bf16):
            logits_1..3, q_1..3, x_recon = model forward (deep supervision + decoder)
            loss = mean_s focal(logits_s, y) + 0.1·MSE(x_recon, x) + 0.1·SAM(x, x_recon)
        if loss not finite: skip batch (stability controller), continue
        backward; check gradients; clip to 1.0
        optimizer.step(); scheduler.step(); ema.update(model)
        if parameters not finite: abort run
    validate EMA copy on the validation subset            # _validate_one_epoch
    build the epoch metrics; sanity and collapse checks; stopping rules
    if valid epoch and val macro-F1 > best: mark best
    write eval artifacts, per-class metrics, checkpoints, history, plots
finally: final history flush, diagnostics (confidence, latent space, reconstruction export),
         experiment_report.json
```

---

## 8. Evaluation

### 8.1 What is evaluated, where

| Stage | Data | Weights | Script / function | Output |
|---|---|---|---|---|
| Every epoch | validation subset (9.86 %, fixed) | EMA | `TrainerG_v12._validate_one_epoch_impl` | `history.json`, `classification_reports/`, `confusion_matrix/`, `predictions/`, `per_patient_metrics.json`, `per_capture_metrics.csv` |
| After training: G5 | validation subset | `best_model.pt` | `verify_best_checkpoint_reproduces` | `checkpoint_reproducibility.json` |
| After training: G6 | **full test split** | `best_model.pt` (EMA) | `TrainerG.evaluate_test_split` → `evaluator.evaluate_model` | `test_report.json`, `test_predictions.npz` |
| On demand | full validation split | `best_model.pt` | `scripts/heldout_eval.py` | `heldout_v19/validation/…` |
| On demand | test split again | a chosen checkpoint | `scripts/eval_test_split.py` | rewrites `test_report.json`, `gates.json` (see [HOW_TO_RUN §9](HOW_TO_RUN.md#9-troubleshooting)) |

**Checkpoint selection:** validation macro-F1 (`--checkpoint_metric f1_macro`) on the EMA weights
**[Code]**. **The test split is never used for selection** in the training code **[Code]**.
(`scripts/tune_class_weight_power.py`'s docstring records that the 0.75 default itself was once
chosen from test-split probabilities **[Config/Doc]**.)

### 8.2 Metrics

Computed by the code **[Code]**, present in recorded reports **[Artifact]**:

| Metric | Where | Formula / note |
|---|---|---|
| Accuracy | validation + test | correct / total |
| **Balanced accuracy** | validation + test | mean over classes of recall |
| Precision / recall / **F1**, macro and weighted, per class | validation + test | scikit-learn, `zero_division=0` |
| Per-class accuracy | validation + test | one-vs-rest (TP + TN) / N |
| Cohen's κ, Matthews correlation | test (`evaluator_metrics`) | from the confusion matrix |
| ROC-AUC, PR-AUC (per class and macro) | test | one-vs-rest on softmax probabilities |
| Calibration: NLL, Brier score, ECE, MCE | test | 15 equal-width confidence bins |
| Per-class sensitivity, specificity, MCC, κ | test (`per_class_extended`) | one-vs-rest |
| Efficiency: parameters, latency at batch 1 and 16, throughput, peak GPU memory | test (`efficiency`) | 5 warm-up + 30 timed runs |
| FLOPs and core applications | test (`flops`), `flops_report.json` | hook-based count for one sample; recursive paper model: **6.2033 GFLOPs** **[Run]** |
| Per-patient macro recall | validation (`per_patient_metrics.json`) | needs `groups_val.npy` |
| Image-level metrics | test (`image_level`) | only with `--eval_group_aggregation mean_prob` (or `majority`) and `images_test.npy` |
| Reconstruction (RMSE, MAE, SAM°, SID, PSNR, SSIM, Pearson, cosine, peak-position error) | validation | in reflectance units via `denormalize` |

`test_report.json` keys **[Artifact]**: `checkpoint`, `checkpoint_epoch`, `checkpoint_weights`,
`best_epoch`, `n_test_samples`, `num_classes`, `class_names`, `amp_mode`, `sklearn_metrics`,
`evaluator_metrics`, `per_class_extended`, `calibration`, `efficiency`, `spectral`,
`predicted_label_histogram`, `true_label_histogram`, `flops`.
`test_predictions.npz` holds `true_label`, `predicted_label` and `probs`.

### 8.3 Reported in documentation (not re-run in this audit)

| Claim | Source | Status |
|---|---|---|
| 446,409 parameters at 32 and 3 bands | README | **verified [Run]** |
| 92.8 ± 1.9 % test balanced accuracy over five seeds (five test patients) | README | **verified [Artifact]**: seeds 1, 7, 13, 23, 42 give 0.9278 ± 0.0187 (HSI); the matched RGB arm gives 0.8867 ± 0.0032 |
| 6.2× fewer parameters but 22.3× more arithmetic than the hierarchical model | README | **verified [Run]**: `split` 2,773,007 params / 0.2781 GFLOPs vs `recursive` 446,409 / 6.2033 GFLOPs (32 bands, 11 × 11, one sample) |
| Reference run: test balanced accuracy 0.9437, macro-F1 0.9036 | `documentations/18_top_runs.md` | **verified [Artifact]** (`scripts/compare_runs.py`) |

### 8.4 Mentioned but not implemented, or inactive

| Item | Finding |
|---|---|
| `scientifically_qualified` | `--skip_dataset_validation`'s help says it is recorded in `config.json`. `train.py` does **not** write it; only the archived v14/v15 entry points did **[Code]**. |
| ROI filtering (`--roi_min_frac`) | requested in the paper's prep command but has no effect on this download ([§5.3](#53-preprocessing)) **[Code]**, **[Artifact]** |
| `--early_stop_patience` | recorded but ignored unless `--early_stopping on` **[Code]** |
| `--weight_method` | both options give identical weights **[Code]** |
| UMAP latent plot | needs `umap-learn`, which no environment file lists; skipped with a message **[Run]** |
| `--amp` for the test split | the G6 evaluation ignores it and runs in fp32 **[Code]** |

---

## 9. Academic / presentation explanation

**1. What is the problem?** Hyperspectral histology images have many bands, but MedMamba, a
state-space image classifier, has an input layer tied to a fixed channel count and ignores which
wavelength each channel is. The project asks whether a band-count-agnostic, wavelength-aware design
with far fewer parameters can classify breast tissue patches, and whether the extra bands help
compared with RGB of the same tissue.

**2. What does the project do?** It prepares a public hyperspectral breast-histology dataset into
patient-separated 11 × 11 patches (32 selected bands, plus an RGB rendering of the same patches),
trains the MedMamba-SS-TRM classifier with a reproducible pipeline, and evaluates it on held-out
patients against RGB, hierarchical, and published hyperspectral baselines.

**3. How does it work?** Each pixel's spectrum becomes a sequence of band tokens that carry their
wavelength. A selective scan reads the sequence, and the bands are pooled into a fixed-size
descriptor. A single small block is then applied 63 times to refine a latent and an answer state,
and every refinement segment is supervised.

**4. What is the model?** MedMamba-SS-TRM: spectral pathway + Tiny-Recursive-Model-style core,
446,409 parameters at any band count, with an auxiliary reconstruction decoder.

**5. What data does it use?** HistologyHSI-BC-Recurrence (644 captures from 45 patients, three
tissue classes) and PAD-UFES-20 (2,298 smartphone skin-lesion images from 1,373 patients, six classes).

**6. What happens during training?** Focal loss with softened class weights over three supervised
segments, plus reconstruction losses; AdamW with warmup-cosine; bf16; EMA weights for model
selection by validation macro-F1; automatic integrity, leakage and sensitivity gates.

**7. How is performance evaluated?** On held-out patients only: balanced accuracy, macro-F1,
per-class metrics, κ, MCC, ROC/PR-AUC, calibration, per-patient recall, FLOPs and latency.

**8. What are the outputs?** A self-describing run directory per experiment: configuration,
history, best checkpoint, test report and predictions, plots, reconstruction samples.

**9. Technical contributions supported by the evidence:**
- a spectral pathway whose parameter count does not depend on the number of bands
  (446,409 parameters at 32 and at 3 bands) **[Run]**;
- a weight-shared recursive backbone: 6.2× fewer parameters than the hierarchical model, at
  22.3× more arithmetic per sample **[Run]**;
- a controlled HSI-vs-RGB comparison on identical patches and splits: 0.928 ± 0.019 vs
  0.887 ± 0.003 test balanced accuracy over five seeds **[Artifact]**;
- an engineering contribution: automatic gates, patient-level leakage checks, and parity tests
  that pin the training code to the runs it produced **[Code]**, **[Run]**.

**10. Limitations and unverified aspects:**
- band selection used labels of held-out patients (README: results "provisional") **[Code]**;
- ROI filtering requested but ineffective **[Code]**, **[Artifact]**;
- five test patients; per-patient spread is large (validation per-patient macro recall
  0.75 ± 0.18 for `ref20`) **[Run]** via `compare_runs.py --per_patient`;
- the project's own documents report that "the recursion-depth sweep is flat: more core
  applications do not help" (`documentations/18_top_runs.md`), and that the hierarchical
  MedMamba-SS collapses to one class in eval mode on this data (`FUTURE_EXPERIMENTS.md` F9)
  **[Config/Doc]**;
- runs are not bit-reproducible (`--deterministic` off) **[Artifact]**;
- the MedMamba baseline lives outside this repository **[Code]**.

---

## Verified facts

1. One training run = `train.py` → `train_cli.resolve` → `train_banner.print_banner` → `train_pipeline.run` → `TrainerG.fit` **[Code]**, **[Run]**.
2. The default profile is `paper_recipe`; the default architecture is `recursive` (`MedMambaSSTRM`) **[Code]**.
3. The paper model has 446,409 parameters at 32 and at 3 bands; 446,796 with 6 classes; the reconstruction decoder adds 129,440 at 32 bands **[Run]**.
4. The recursive core is applied 63 times per forward pass **[Run]**.
5. Recursive preset 446,409 params / 6.2033 GFLOPs vs hierarchical `split` 2,773,007 / 0.2781 GFLOPs (32 × 11 × 11, one sample) **[Run]**.
6. Modality is decided only by the presence of `wavelengths.npy` **[Code]**.
7. Validation and `best_model.pt` use EMA weights; the best epoch maximizes validation macro-F1 **[Code]**.
8. The test split is evaluated once, automatically, in fp32, with `best_model.pt` **[Code]**, **[Run]**.
9. The paper build: 644 captures, 45 patients, 35/5/5 patients, 2,452,086 / 334,516 / 348,894 patches, 32 bands stored as float16 **[Artifact]**.
10. Band selection (Pass 1) sees all captures, including held-out patients **[Code]**.
11. ROI filtering is inactive on this download (per-patient GeoJSON names) **[Code]**, **[Artifact]**.
12. HSI test balanced accuracy over seeds 1, 7, 13, 23, 42: 0.9278 ± 0.0187; RGB arm 0.8867 ± 0.0032 **[Artifact]**.
13. Normalization statistics are not saved with a run **[Code]**.

## Unknown / not verified

1. **GPU training with `train.py`** — not run in this audit; the training description relies on the code, on recorded runs, and on the GPU parity reported in `archive/README.md` and `logs/smoke/` **[Config/Doc]**.
2. **Data preparation from raw files** — not run; the data-pipeline description relies on the code and the recorded prep logs.
3. **`--label_source recurrence`** — the workbook name differs from what the code looks for; usefulness of the labels not established.
4. **Whether the HMI-LUSC preparation is intended to be public** — it is tracked and also listed in `.gitignore`.
5. **The MedMamba baseline code** — outside this repository, with uncommitted local changes; not inspected in this audit.
