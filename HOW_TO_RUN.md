# MedMamba-SS-TRM — How to Run

**What this is.** The user and reproducibility guide to this repository: how to install it,
prepare the datasets, configure and launch training, evaluate finished runs, read the outputs, fix
common problems, and reproduce the paper's experiments. It is written for a technically capable
reader who has never seen the project.

**Companion file:** [HOW_IT_WORKS.md](HOW_IT_WORKS.md) — what the project does, the architecture,
the data pipeline, the model, training and evaluation.

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

- [1. Quick start](#1-quick-start)
- [2. System requirements](#2-system-requirements)
- [3. Environment setup](#3-environment-setup)
- [4. Dependencies](#4-dependencies)
- [5. Dataset setup](#5-dataset-setup)
- [6. Configuration](#6-configuration)
- [7. Commands](#7-commands): [7.1 environment](#71-environment-setup), [7.2 data](#72-dataset-preparation),
  [7.3 training](#73-training), [7.4 validation](#74-validation), [7.5 testing](#75-testing),
  [7.6 inference](#76-inference), [7.7 evaluation and analysis](#77-evaluation-and-analysis-of-finished-runs),
  [7.8 sweeps and other workflows](#78-project-specific-workflows)
- [8. Expected outputs](#8-expected-outputs)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Research reproducibility](#10-research-reproducibility)
- [Verified facts](#verified-facts) and [Unknown / not verified](#unknown--not-verified)

---

## 1. Quick start

All commands from the repository root. Steps 2 and 4 have the caveats of [§3](#3-environment-setup)
and [§7.2](#72-dataset-preparation).

```bash
# 1. Clone / open the project (the remote recorded in the author's clone; public access NOT VERIFIED)
git clone git@github.com:gmdvl/MedMamba-SS-TRM.git && cd MedMamba-SS-TRM

# 2. Create the environment            (NOT VERIFIED here - see section 3)
mamba create --name medmamba-ss-trm python=3.11 && mamba activate medmamba-ss-trm

# 3. Install dependencies              (NOT VERIFIED here - see section 3)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-learn pandas pillow matplotlib spectral openpyxl pytest
python -m pytest -q                    # verified here: 757 passed

# 4. Prepare the dataset               (recorded command of the paper build)
RAW_HSI_ROOT=/absolute/path/to/HistologyHSI-BC-Recurrence          # EDIT
python prepare_histologyhsi_bc.py --root "$RAW_HSI_ROOT" --out_dir ./data/hsi_v8-80_10_10_importance-new \
    --modality both --label_source tissue --patch_size 11 --stride 11 --roi_min_frac 0.8 \
    --band_selection importance --num_bands 32 --band_min_gap 8 --band_max_corr 0.95 \
    --band_min_coverage 0.30 --split 80_10_10 --split_strategy stratified \
    --capture_gain median_ratio --hsi_value_scale auto --rgb_source synthetic \
    --seed 42 --num_workers 8 --verify_level deep

# 5. Configure: nothing to edit - the profile + typed flags are the configuration

# 6. Train the reference run           (recorded command; GPU)
python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 20 --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --run_tag ref20

# 7. Evaluation is automatic (test split, gate G6); compare runs:
python scripts/compare_runs.py --all --filter ref20

# 8. Results
ls experiments/*_ref20/                # test_report.json, history.json, best_model.pt, plots/ ...
```

---

## 2. System requirements

Recorded on the author's machine **[Artifact]** unless stated. **No minimum requirements are stated
anywhere in the project**; the values below are what was used, not minimums.

| Resource | What is known |
|---|---|
| Operating system | Linux (all recorded paths; `linux-64` conda spec) **[Config/Doc]**. Linux-specific probes (`/proc/meminfo`, `/dev/shm`, `resource`) are wrapped in `try` blocks **[Code]**. Windows / macOS: **[NOT VERIFIED]** |
| Python | 3.11 (`mamba-spec.txt` pins 3.11.15; `environment_backup.yml` allows 3.10–3.11) **[Config/Doc]**. Tests also pass on 3.13 **[Run]** |
| PyTorch / CUDA | recorded runs: torch 2.11.0+cu128 **[Artifact]**. The README states that Blackwell GPUs need a cu128 build **[Config/Doc]** |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16,706 MB **[Artifact]** (`system_memory_startup.json`). Training on CPU works for tiny data **[Run]**; full-size CPU training is **[NOT VERIFIED]** and would be very slow **[Inferred]** |
| CPU / RAM | 32,643 MB RAM and 16,321 MB `/dev/shm` recorded **[Artifact]** |
| Storage | raw HSI download ≈ 1.2 TB (README); paper build 27 GB; PAD build 1.3 GB; one paper run ≈ 154 MB; `torch.compile` cache grew to 28 GB **[Artifact]** |
| External software | none beyond Python packages; mamba/conda only for the documented environment route |

## 3. Environment setup

> **The documented route is not reproducible as written [Inferred from the files; NOT VERIFIED by execution].**
> `README.md` and `replicate_from_txt.txt` say:
>
> ```bash
> mamba create --name medmamba-ss-trm --file mamba-spec.txt
> mamba activate medmamba-ss-trm
> pip install -r requirements.txt
> ```
>
> Three problems, all visible in the files:
> 1. `requirements.txt` has 23 lines of the form `package @ file:///home/conda/feedstock_root/…`.
>    Those paths exist only on the machine that produced the file, so `pip install -r requirements.txt`
>    is expected to fail on another machine.
> 2. `requirements.txt` pins `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128`, which are not on
>    PyPI's default index; no `--index-url` / `--extra-index-url` is given.
> 3. `mamba-spec.txt` installs conda `pytorch-2.5.1` built for CUDA 12.1, which per the README has no
>    kernels for the Blackwell GPU the project used. The recorded runs used the pip-installed 2.11.0+cu128.

**A minimal environment** that covers every third-party import of the live code. The package list is
**[Code]** (every `import` in `train.py`, `run_experiments.py`, the model files, `prepare_*.py`,
`training/`, `scripts/`, `sweeps/`); the commands themselves are **NOT VERIFIED — DO NOT EXECUTE WITHOUT CONFIRMATION**
(they were not run in this audit):

```bash
# from the repository root
mamba create --name medmamba-ss-trm python=3.11      # or: python3.11 -m venv .venv && source .venv/bin/activate
mamba activate medmamba-ss-trm

# PyTorch: pick the wheel index that matches your GPU/driver; the recorded runs used cu128
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install numpy scipy scikit-learn pandas pillow matplotlib spectral openpyxl pytest
pip install psutil umap-learn        # optional: better memory reporting; UMAP plot
```

**Check the build against your GPU** (README) **[Config/Doc]**:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
# on an RTX 50-series (Blackwell) card the arch list must contain sm_120
```

**Check the code against your environment (CPU, no data needed)** **[Run]**:

```bash
python -m pytest -q          # 757 passed in this audit (Python 3.13, torch 2.14 CPU), ~3.5 min
```

## 4. Dependencies

| Package | Needed for | Version in `requirements.txt` | Evidence |
|---|---|---|---|
| torch | everything | 2.11.0+cu128 | [Code], [Config/Doc] |
| numpy | everything | 2.4.6 (local wheel) | [Code] |
| scikit-learn | metrics, band selection, probes | 1.9.0 | [Code] |
| scipy | `scripts/analysis_v19.py` | 1.17.1 (local wheel) | [Code] |
| pandas | PAD preparation | 3.0.5 | [Code] |
| Pillow | image decoding in preparation | 12.3.0 | [Code] |
| matplotlib | plots | 3.11.1 | [Code] |
| spectral | ENVI cubes (histology prep; **required** there) | 0.25 | [Code] |
| openpyxl | `--label_source recurrence` only | 3.1.5 | [Code] |
| pytest | tests | 9.1.1 | [Code] |
| psutil | optional (falls back to `/proc/meminfo`) | **not listed** | [Code] |
| umap-learn | optional UMAP plot | **not listed** | [Code], [Run] |

Listed in `requirements.txt` but not imported by the live code: `torchvision`, `seaborn`,
`scikit-image`, `tqdm`, `tifffile`, `ImageIO`, `triton` (a PyTorch dependency), `panda` (sic) **[Code]**.
`torch.compile` (on by default in the paper profile) needs a working Triton/compiler stack on GPU;
if compilation fails the run continues uncompiled with a warning **[Code]**.

**Model-specific:** no `mamba_ssm`, no custom CUDA/Triton kernels **[Code]**.
**External code:** the MedMamba baseline numbers come from the separate MedMamba repository
(`MEDMAMBA_DIR`, default `../medmamba-original/MedMamba`), which is **not** part of this repository
**[Code]** (`sweeps/paper_evidence_upgrade.py`). Its `train_hsi.py` changes are uncommitted in the
author's local copy **[Artifact]**.

## 5. Dataset setup

1. **Obtain** the datasets from the sources in [HOW_IT_WORKS §5.1](HOW_IT_WORKS.md#51-required-datasets) and accept their terms.
2. **Place** them anywhere. The scripts take the raw root as `--root`. The prepared output goes to
   `--out_dir`; the sweep files expect `data/…` relative to the repository root **[Code]**.
3. **Prepare**: see [§7.2](#72-dataset-preparation).
4. **Verify** (read-only; reads `.npy` headers only):

```bash
python - <<'EOF'
import numpy as np, json, pathlib
d = pathlib.Path("data/hsi_v8-80_10_10_importance-new/hsi")
for s in ("train", "val", "test"):
    X = np.load(d / f"X_{s}.npy", mmap_mode="r"); y = np.load(d / f"y_{s}.npy", mmap_mode="r")
    print(s, X.shape, X.dtype, y.shape)
print(np.load(d / "wavelengths.npy").shape, json.load(open(d / "class_names.json")))
EOF
```

Expected for the paper build **[Artifact]**: `(2452086, 11, 11, 32) float16`, `(334516, 11, 11, 32)`,
`(348894, 11, 11, 32)`, `(32,)`, `['healthy', 'DCIS', 'IDC']`.
Then read `capture_gain_report.json`, `band_selection_report.json`, `class_coverage_report.json`
and `prep_log.txt` in the build root (doc 12 §A.1) **[Config/Doc]**.

**Final structure** (paper build) **[Artifact]**:

```text
data/hsi_v8-80_10_10_importance-new/
├── _progress.json  prep_log.txt  split_assignment.json  dataset_statistics.json
├── band_importance.npy  band_ranking.npy  band_selection_report.json  band_coverage_report.json
├── selected_band_indices.npy  selected_wavelengths.npy  selection_report.json
├── capture_gain_report.json  class_coverage_report.json  class_coverage_report.pre.json
├── hsi/   X_{train,val,test}.npy  y_*.npy  wavelengths.npy  wavelengths_full.npy  class_names.json
│          groups_*.npy  captures_*.npy  capture_index.json  dataset_manifest.json
└── rgb/   X_{train,val,test}.npy  y_*.npy  class_names.json  groups_*.npy  captures_*.npy
           capture_index.json  dataset_manifest.json
```

## 6. Configuration

Configuration is **command-line flags only**; there are no YAML/JSON config files for training. A
`--profile` supplies defaults; any typed flag overrides them **[Code]**. `python train.py --help`
lists every flag **[Run]**. Defaults live in [`training/train_cli.py`](training/train_cli.py) (parser) and
[`training/train_profiles.py`](training/train_profiles.py) (profiles).

### Profiles [Code]

| Profile | Use | Reconstruction | Run-name suffix |
|---|---|---|---|
| `paper_recipe` (**default**) | reproduce or extend the paper | latent, λ_mse 0.1 | `_optimal_recon` |
| `pad_ufes_best_norecon` | best metrics on PAD-UFES-20; accepts `--stage fit`, `balance` or `full` | none | `_optimal` |
| `medmamba_protocol_norecon` | compare with MedMamba's own recipe (constant LR, fixed 0.5/0.5 normalization, fp32, no clipping) | none | `_medmamba_protocol` |
| `pipeline_defaults` | bare parser defaults | latent, λ_mse 1.0 | — |

Old names `v18`, `optimal`, `original`, `base` still work **[Code]**.

### Effective values (resolved with `training.train_cli.resolve`) [Run]

| Parameter | Location | Parser default | `paper_recipe` + paper flags (HSI) | `paper_recipe`, bare (HSI) | `pad_ufes_best_norecon` (`data/pad_optimal`) | `medmamba_protocol_norecon` (`data/pad_original`) |
|---|---|---:|---:|---:|---:|---:|
| `--data_dir` | `train_cli.py:63` | **required** | `data/…/hsi` | | | |
| `--architecture` | `:133` | split → `recursive` (post-parse) | recursive | recursive | recursive | recursive |
| `--patch_size` (stem stride) | `:637` | preset | 1 | 1 | 8 | 8 |
| `--batch_size` | `:65` | 256 | **256 (typed)** | 32 | 32 | 64 |
| `--epochs` | `:64` | 10 | **20 or 12 (typed)** | 200 | 200 | 150 |
| `--lr` | `:70` | 1e-4 | 3e-4 | 3e-4 | 3e-4 | 1e-4 |
| `--weight_decay` | `:173` | 0.05 | 0.05 | 0.05 | 0.05 | 1e-4 |
| `--loss` / `--focal_gamma` | `:103` | ce / 2.0 | focal_weighted / 1.5 | focal_weighted / 1.5 | focal_weighted / 1.5 | ce |
| `--class_weight_power` | `:109` | 1.0 | 0.75 | 0.75 | 0.75 | 1.0 |
| `--normalization` | `:355` | auto | global_zscore | global_zscore | global_zscore | global_zscore (0.5/0.5) |
| `--recon_mode` / `--lambda_mse` | `:95` | latent / 1.0 | latent / 0.1 | latent / 0.1 | none / 0 | none / 0 |
| `--lambda_sam` | `:76` | 0.1 | **0.1 (typed)** | **0.0** | 0 | 0 |
| `--trm_ema_rate` | `:156` | 0.999 | 0.9995 (resolved) | 0.9995 | 0.99346 | 0 (off) |
| `--train_subsample_frac` | `:122` | 1.0 | **0.102 (typed)** | 0.102 (resolved) | 1.0 | 1.0 |
| `--val_subsample_frac` | `:383` | 1.0 | **0.0986 (typed)** | 0.0986 (resolved) | 1.0 | 1.0 |
| `--amp` | `:244` | bf16 | bf16 | bf16 | bf16 | off |
| `--compile` | `:414` | off | on | on | on | off |
| `--loader_mode` | `:182` | balanced | performance | performance | performance | balanced |
| `--fast_loop` | `:643` | off | on | on | on | off |
| `--checkpoint_metric` | `:89` | f1_macro | f1_macro | f1_macro | f1_macro | val_accuracy |
| `--lr_schedule` | `:607` | warmup_cosine | warmup_cosine | warmup_cosine | warmup_cosine | constant |
| `--on_split_drift` | `:619` | abort | warn | warn | warn | abort |
| `--max_gradient_norm` | `:257` | 1.0 | 1.0 | 1.0 | 1.0 | 0 (no clipping) |
| `--classifier_dropout` | `:175` | None | 0.1 | 0.1 | 0.1 | 0.0 |
| `--seed` | `:73` | 42 | 42 | 42 | 42 | 42 |
| `--run_tag` | `:580` | None | e.g. `ref20` | | | |

> **The bare `paper_recipe` command is NOT the paper command [Run].** Without the typed flags it trains
> with batch 32, 200 epochs and **no SAM term**. Every paper run typed `--batch_size 256 --epochs 12|20
> --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986` (`sweeps/paper_runs.py`) **[Code]**.

**Parameters that should normally stay unchanged** (the parity tests and recorded runs depend on
them) **[Code]**: the v15 representation flags (`--spectral_token_fusion concat_mlp`,
`--spectral_pe_gain 0.1`, `--spectral_value_init_std 0.5`, `--spectral_ctx_norm`,
`--classifier_init fan_in`, `--spatial_pe_gain 0.1`, `--weight_decay_groups`), `--spectral_chunk_size 1024`,
and the TRM sizes (`--trm_dim 128 --trm_core_layers 2 --trm_n_latent 6 --trm_n_improve 3
--trm_deep_supervision_steps 3 --trm_mixer mlp`).

**Use `--run_tag`** (letters, digits and dashes only) whenever an ablation changes a flag that the run
name does not encode, e.g. `--trm_n_improve`, `--recon_mode`, `--no_use_wavelengths` **[Code]**.

---

## 7. Commands

**Where:** every command runs **from the repository root** (run directories and `data/` paths are
relative to the working directory) **[Code]**. **Environment:** the one from [§3](#3-environment-setup).
Lines that you must edit are marked `# EDIT`.

Evidence per command: **[Run]** = executed here; **[Code]** = arguments checked against the
parser/source but not executed here; **[Artifact]** = the exact command recorded by a real run.

### 7.1 Environment setup

See [§3](#3-environment-setup). The test suite is the install check **[Run]**:

```bash
python -m pytest -q                              # expect: 757 passed
python -m pytest tests/test_train_parity.py -q   # one file
```

### 7.2 Dataset preparation

**Histology HSI + RGB, the paper's build** — this exact argument list is recorded in the build's
`prep_log.txt` **[Artifact]**:

```bash
RAW_HSI_ROOT=/absolute/path/to/HistologyHSI-BC-Recurrence    # EDIT
python prepare_histologyhsi_bc.py --root "$RAW_HSI_ROOT" \
    --out_dir ./data/hsi_v8-80_10_10_importance-new \
    --modality both --label_source tissue \
    --patch_size 11 --stride 11 --roi_min_frac 0.8 \
    --band_selection importance --num_bands 32 \
    --band_min_gap 8 --band_max_corr 0.95 --band_min_coverage 0.30 \
    --split 80_10_10 --split_strategy stratified \
    --capture_gain median_ratio --hsi_value_scale auto \
    --rgb_source synthetic \
    --seed 42 --num_workers 8 --verify_level deep
```

- Writes `hsi/` and `rgb/` with the same samples and split. Resumable: re-run the same command after
  an interruption **[Code]**. Needs `spectral`.
- Recorded wall clock: 52 min with 8 workers (44 min of it in Pass 1) **[Artifact]** (`prep_log.txt`);
  doc 12 reports about 1.8 h **[Config/Doc]**. Budget ~30 GB of free disk (doc 12) **[Config/Doc]**.
- Same split, band selection on training patients only (`prepare_histologyhsi_bc_trainsel.py`; the full
  command is in its docstring and in `sweeps/paper_evidence_upgrade.py`, stage `prep`) **[Code]**.

**PAD-UFES-20, tuned protocol** — recorded command **[Artifact]** (all other values are the script's defaults:
`--tiling whole --img_size 224 --split 70/15/15 --split_strategy stratified --balance_classes none --seed 42`):

```bash
RAW_PAD_ROOT=/absolute/path/to/PAD-UFES-20                   # EDIT
python prepare_pad_ufes_20_optimal.py --root "$RAW_PAD_ROOT" --out_dir ./data/pad_optimal --num_workers 8
```

**PAD-UFES-20, MedMamba protocol** (60/10/30, patient split by default) — recorded command **[Artifact]**:

```bash
python prepare_pad_ufes_20_original.py --root "$RAW_PAD_ROOT" --out_dir ./data/pad_original --num_workers 8
```

**Band-subset builds** (slices an existing build; no raw data needed) **[Code]** (`sweeps/paper_runs.py`):

```bash
python scripts/make_band_subset_build.py --src data/hsi_v8-80_10_10_importance-new/hsi --keep 16 --out data/hsi_v8-bands16
```

Add `--dry_run` to any prep command to print the split without extracting **[Code]**.

### 7.3 Training

**The paper's reference run** (`ref20`) — matches the `cli_args` recorded by the real run, as checked by
`tests/test_sweeps.py` **[Code]**, **[Run]** (test passes):

```bash
python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 20 --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --run_tag ref20
```

- Output: `experiments/<timestamp>_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_ref20/`.
- Recorded duration: 7.38 h for 20 epochs on the RTX 5060 Ti **[Config/Doc]** (`FUTURE_EXPERIMENTS.md`).
  The project history notes that runs of that period "shared the GPU 2–4 at a time", so wall-clock
  is not a clean cost measure **[Config/Doc]** (`PROJECT_HISTORY_AND_TECHNICAL_EVOLUTION.md`).
- The 12-epoch ablation baseline is the same command with `--epochs 12 --run_tag abl-base`. Because
  `--epochs` sets the cosine horizon, compare 12-epoch arms only with each other **[Code]**.
- **Matched RGB arm:** the same flags with `--data_dir data/hsi_v8-80_10_10_importance-new/rgb`
  (`sweeps/paper_runs.py`, stage `rq0`) **[Code]**.

**PAD-UFES-20** (README) **[Config/Doc]**; resolution of the defaults **[Run]**; training not executed:

```bash
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal
# debugging ladder of the same profile:
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --stage fit
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --stage balance
```

**MedMamba protocol on PAD** **[Code]**:

```bash
python train.py --profile medmamba_protocol_norecon --data_dir ./data/pad_original
```

**CPU smoke test on synthetic data** — what this audit ran **[Run]**. Point `--data_dir` at any small
directory that follows [HOW_IT_WORKS §5.1.4](HOW_IT_WORKS.md#514-the-prepared-dataset-contract):

```bash
python train.py --data_dir <tiny_dataset_dir> --batch_size 16 --epochs 2 --lambda_sam 0.1 \
    --compile off --amp off --loader_mode safe --cpu_threads 8 --seed 42 --run_tag cpu-smoke
```

**Hierarchical model or ablations:** add flags to the reference command, e.g.
`--architecture split --epochs 12 --run_tag arch-split`, `--trm_n_improve 2 --epochs 12 --run_tag depth-n2`,
`--recon_mode none` (drop `--lambda_sam`), `--no_use_wavelengths`. `--architecture efficient` also needs
`--fusion_type gated` (doc 15) **[Config/Doc]**. Every paper variant is spelled out in `sweeps/paper_runs.py` **[Code]**.

**Baselines** **[Code]**:

```bash
python scripts/train_hsi_baseline.py --arch hybridsn --data_dir data/hsi_v8-80_10_10_importance-new/hsi --seed 42 --run_tag s42
python scripts/train_hsi_baseline.py --arch spectralformer --data_dir data/hsi_v8-80_10_10_importance-new/hsi --seed 42 --run_tag s42
python scripts/shallow_baseline_v16.py --data_dir data/hsi_v8-80_10_10_importance-new/hsi --normalization global_zscore
```

### 7.4 Validation

Validation runs **automatically every epoch** inside `train.py`; G5 re-validates `best_model.pt` after
training **[Code]**, **[Run]**. To score the **whole** validation split (not the 9.86 % subset)
**[Code]** (docstring; not executed here):

```bash
python scripts/heldout_eval.py --run_dirs experiments/<run_dir>        # EDIT <run_dir>
# -> <run_dir>/heldout_v19/validation/{test_report.json, test_predictions.npz}  (the split is VALIDATION)
```

### 7.5 Testing

Testing runs **automatically** after training (gate G6, `--eval_test best`) **[Code]**, **[Run]**. To
re-run it on a finished run **[Run]**:

```bash
cp experiments/<run_dir>/test_report.json experiments/<run_dir>/test_report.backup.json   # EDIT; see warning
python scripts/eval_test_split.py --run_dir experiments/<run_dir> --dry_run   # checks only
python scripts/eval_test_split.py --run_dir experiments/<run_dir>
python scripts/flops_report.py --run_dir experiments/<run_dir> --write_into_test_report
```

> **Warning [Run].** `eval_test_split.py` overwrites `test_report.json` and `gates.json`, and its `flops`
> block comes back as `{"error": "ValueError: not enough values to unpack (expected 4, got 3)"}`.
> The third command restores the FLOP count. The script rebuilds the model with the architecture
> preset's stem stride, not the run's recorded `--patch_size`. That is exact for every recursive run on
> 11 × 11 patches and for hierarchical runs on HSI, but **not** for 224 × 224 PAD runs (stride 8 vs
> preset 1) or hierarchical runs on RGB (stride 1 vs preset 4); see [§9](#9-troubleshooting).

### 7.6 Inference

**The repository has no inference script for new, unlabelled data [Code].** The snippet below uses only
the repository's own functions. It is **not part of the repository**. It was saved as a file and run on
the synthetic smoke run, where it reproduced that run's `test_predictions.npz` probabilities to within
6e-8 **[Run]**. Run it from the repository root:

```python
# save as predict_new.py;  usage: python predict_new.py <run_dir> <patches.npy>
# patches.npy: [N, H, W, C] raw values, same bands and band order as the training data
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from scripts.eval_test_split import _rebuild_model
from training.npy_data import compute_global_channel_stats
from training.normalization import normalize_patch

run_dir, patches_path = Path(sys.argv[1]), sys.argv[2]
cfg = json.load(open(run_dir / "config.json")); a = cfg["cli_args"]
model = _rebuild_model(cfg)
model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)["model_state"])
model.eval()

gm = gs = None                       # statistics are not saved with the run: recompute as training did
if a["normalization"] == "global_zscore":
    if a.get("global_stats", "train_fit") != "train_fit":
        raise SystemExit("only --global_stats train_fit is handled here")
    gm, gs = compute_global_channel_stats(str(Path(a["data_dir"]) / "X_train.npy"),
                                          sample_cap=a["norm_stats_sample_cap"], seed=a["seed"])
X = np.load(patches_path, mmap_mode="r")
x = np.stack([normalize_patch(np.nan_to_num(np.asarray(p, np.float32), nan=0.0, posinf=1.0, neginf=0.0),
                              a["normalization"], global_mean=gm, global_std=gs)[0] for p in X])
x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
wl_path = Path(a["data_dir"]) / "wavelengths.npy"
wl = torch.from_numpy(np.load(wl_path).astype(np.float32)) if wl_path.is_file() and a.get("use_wavelengths", True) else None
with torch.no_grad():
    out = model(x, wavelengths=wl) if wl is not None else model(x)
    probs = torch.softmax((out[0] if isinstance(out, tuple) else out).float(), dim=-1)
np.save(run_dir / "predictions_new.npy", probs.numpy())
print("class counts:", np.bincount(probs.argmax(-1).numpy(), minlength=probs.shape[1]))
```

Requirements: the run's training `X_train.npy` must still exist at the recorded `data_dir` (for the
normalization statistics). The snippet processes the whole array in one batch; split large arrays into
chunks. Like `eval_test_split.py`, it is exact only when the run's recorded `patch_size` equals the
architecture preset's stem stride (every recursive 11 × 11 run; see the warning in [§7.5](#75-testing)) **[Code]**.

### 7.7 Evaluation and analysis of finished runs

| Command | Does | Evidence |
|---|---|---|
| `python scripts/compare_runs.py --all --filter ref20 [--per_patient] [--csv out.csv]` | table of finished runs (read-only) | [Run] |
| `python scripts/flops_report.py --run_dir experiments/<run>` | `flops_report.json` | [Run] |
| `python scripts/flops_report.py --architectures split,recursive --modality hsi --num_classes 3 --in_channels 32 --patch_hw 11,11` | FLOPs/params of presets, no run needed | [Run] |
| `python scripts/latent_space.py --run_dir experiments/<run> [--split test]` | t-SNE (and UMAP if installed) plots | [Code] |
| `python scripts/export_recon_samples.py --run_dir experiments/<run> --split test` | reconstructed cubes + figures for a split | [Code] |
| `python scripts/eval_band_decimation.py --run_dir experiments/<run> --keep_bands 32,16,8,4,2` | zero-shot band removal on a trained checkpoint | [Code] |
| `python scripts/eval_band_decimation_fixed_eta.py --run_dir experiments/<run> --keep_bands 32,16,8` | the same with the encoding scale fixed at training value | [Code] |
| `python scripts/shallow_probe_heldout.py --data_dir data/…/hsi` | logistic-regression probes on full val/test | [Code] |
| `python scripts/analysis_v19.py`, `python scripts/analysis_v20.py` | paper aggregates → `paper/source/*.json` | [Code] |

> Do **not** `import` `scripts/make_fig_bands_v18.py` or `scripts/tables_v18_short.py`: they have no
> `if __name__ == "__main__":` guard, so importing runs them (the first rewrites
> `paper/figures/results/fig_bands_v18.{png,pdf}`) **[Code]**.

### 7.8 Project-specific workflows

**Sweeps** **[Run]** (`--list`, `--status`, `--dry_run`); running stages **[Code]**:

```bash
python run_experiments.py sweeps/paper_runs.py --list          # stages and jobs
python run_experiments.py sweeps/paper_runs.py --status        # done / todo (here: all done except k2-control)
python run_experiments.py sweeps/paper_runs.py k2 --dry_run    # resolve the job, run nothing
python run_experiments.py sweeps/paper_runs.py k2              # run it
python run_experiments.py sweeps/paper_runs.py depth --jobs 2  # two jobs of one stage at once
python run_experiments.py --find ref20 --data data/hsi_v8-80_10_10_importance-new/hsi   # print a run dir
```

- A train job counts as done when an `experiments/*/config.json` has its `run_tag` and data dir and
  `test_predictions.npz` exists. Logs go to `logs/<sweep file stem>/<job>.log`. A failed job is
  reported and the sweep continues **[Code]**.
- `sweeps/paper_evidence_upgrade.py` reads environment variables `RAW_BC`, `RAW_LUSC`, `LUSC_ROOT`,
  `MEDMAMBA_DIR`, `MM_SEEDS`. Their defaults are paths on the author's machine; set them before use **[Code]**.

**Re-run the three best runs on any dataset** **[Run]** (dry run):

```bash
python scripts/run_top3.py --dry_run
python scripts/run_top3.py --data_dir data/hsi_v8-80_10_10_importance-new/hsi
```

**GPU smoke check** (one short run per profile + archived-vs-current parity) **[Code]**:

```bash
python run_experiments.py sweeps/smoke.py all
```

**Run an archived entry point** (parity reference only) **[Config/Doc]** (`archive/README.md`):

```bash
PYTHONPATH=.:archive python archive/train_example_v18.py --data_dir ...
```

---

## 8. Expected outputs

### 8.1 Console output of `train.py`

Excerpt from the synthetic CPU run **[Run]** (shortened; paths removed):

```text
==============================================================================
MedMamba-SS-TRM - profile 'paper_recipe': the configuration behind every number in the paper, ...
  architecture       : recursive  (stem stride 1, trm_dim=128, mixer=mlp, halting=False)
  objective          : loss=focal_weighted gamma=1.5 class_weight_power=0.75 sampler=none
  optimiser          : AdamW lr=0.0003 wd=0.05 wd_groups=True  clip=1.0
  ...
  DEVIATIONS from profile 'paper_recipe' (typed flags):
    - amp='off'  (paper_recipe: 'bf16')
    - batch_size=16  (paper_recipe: 32)
    ...
[baseline] train prior-entropy CE floor = 1.0986 (ln(k) = 1.0986 for a uniform predictor).
[budget] 96 train samples x 1.0 / batch 16 = 6 steps/epoch x 2 epochs = 12 optimizer steps. ...
[trm-cost] input 11x11, stem stride 1 -> 11x11 = 121 tokens x batch 16; core applied ~63x per step; ...
Dataset Loaded: 96 Train, 48 Val | Classes: 3 | Device: cpu
[gate G8] split drift: PASS -> experiments/<run>/split_drift_report.json
===== MedMamba-SS-TRM Stability Preflight =====
...
[architecture] 'recursive' (hsi): 0.446M parameters
[scheduler] warmup_cosine: total_steps=12 warmup_steps=11 interval=step
[gates G1/G2/G9] margins: G2_logit_std=23.39x, G1_stem_sensitivity=11.90x, G9_disjoint_batch_delta=580.21x
[gate G7] PASS - reconstruction gradient reaches 20 recursive-core parameter(s)
  [train] epoch 1 6/6 100%  loss 0.6933 acc 0.5833  0.498 s/it  ETA 0s

--- Epoch 0001 ---
Train -> Loss: 0.6933 (CE: 0.4379, MSE: 1.0574, SAM: 1.4967, GAN: 0.0000) | Acc: 0.5833 | GradNorm: 419.268
Val   -> Loss: 0.5683 (CE: 0.3045, ...) | Acc: 0.6667 | BalAcc: 0.6667 | ... | F1(macro): 0.5556
Spectral -> RMSE: 0.4929 | MAE: 0.4235 | SAM: 11.24° | SID: 0.0508 | ...
Stats -> Time: 3.6s | LR: 1.91e-04 | GPU: 0MB | Saved: epoch_0001.pt (BEST)
  [time] train 2.99s (0.499 s/step x 6) | val 0.56s | data-wait 0.01s | artifacts 0.64s | unacc +0.00s
  [progress] epoch 1/2 | best f1_macro 0.5556 @ ep 1 | elapsed 3s | ETA 3s
...
TRAINING PROCESS ENDED
Reason: TRAINING_COMPLETED
...
[gate G5] f1_macro: logged 1.0 vs reloaded 1.0 -> PASS
[test-eval] n=48 accuracy=1.0000 balanced=1.0000 f1_macro=1.0000 -> experiments/<run>/test_report.json
[flops-v18] recounted at [1, 8, 11, 11]: 6.105744 GFLOPs, core_applications=63 ...
```

These numbers are from **synthetic** data and say nothing about real performance.

### 8.2 Run directory

Name pattern **[Code]** ([`run_naming_v16.py:94`](training/run_naming_v16.py#L94)):
`<YYYYmmdd_HHMMSS>_<dataset slug>_<architecture>_<modality>_<mixer>_sup<segments>_bs<batch>_<bf16|fp32>_<non-default settings>_<profile suffix>_<run_tag>`.
Real example **[Artifact]**:
`20260915_031356_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_ref20`.

Files written by one `train.py` run **[Run]** (complete list from the synthetic run):

```text
<run>/
├── config.json                   every resolved argument, environment, trainer class, parameter count
├── history.json / history.csv    one row per epoch: losses, all validation metrics, LR, timing, GPU MB
├── experiment_report.json        stop reason, epochs, best epoch, full history
├── best_model.pt                 EMA weights of the best epoch  (load key: "model_state")
├── best_model_live.pt            live (non-EMA) weights of the best epoch
├── best_model_ema.pt             bare EMA shadow dict
├── checkpoints/                  epoch_NNNN.pt (last 3), latest.pt, latest_ema.pt
├── test_report.json              held-out test metrics (HOW_IT_WORKS.md §8.2)
├── test_predictions.npz          true_label, predicted_label, probs
├── gates.json                    G1_G2_G9, G5, G6, G7, G8
├── checkpoint_reproducibility.json, gate_g7_report.json, split_drift_report.json
├── dataset_integrity_report.json, leakage_report.json, class_imbalance_report.json
├── system_memory_startup.json, validation_report.json
├── loss_components.json / .csv, spectral_metrics.csv
├── classification_reports/       classification_report_epochNN.txt
├── confusion_matrix/             confusion_matrix_epoch_NN.png, confusion_matrix.npy
├── predictions/                  predictions_epochNN.npz (validation)
├── plots/                        accuracy, loss, classification_loss, reconstruction_loss, lr,
│                                 gradient_norm, parameter_norm, update_norm, generalization_gap,
│                                 epoch_time, gpu_memory (.png)
├── confidence_analysis/          confidence_histogram, prediction_entropy, reliability_diagram (.png)
├── latent_space/                 latent_tsne.png (+ latent_umap.png if umap-learn is installed)
└── reconstruction/               reconstruction_{samples,best,median,worst}.png, hist_*.png,
                                  metrics_curves.png, samples.npz, samples_metrics.json,
                                  samples/NN_class<k>_q<quantile>.png, provenance.json, status.json
```

Only on real builds (they need sidecars or `_progress.json`): `per_patient_metrics.json`,
`per_capture_metrics.csv`, `dataset_split_report.json` **[Artifact]**. On failure:
`failure_class.json`, `numerical_failure/`, `collapse_failure/`, `gate_g6_failure.json` **[Code]**.
Added later by scripts: `flops_report.json`, `heldout_v19/`, `band_decimation*/` **[Artifact]**.

### 8.3 Other commands

| Command | Output |
|---|---|
| prep scripts | the build directory of [§5](#5-dataset-setup); progress lines with rate and ETA; `prep_log.txt` |
| `pytest -q` | `757 passed, 2 warnings` **[Run]** |
| `compare_runs.py` | a table: `acc 0.9638 bal 0.9437 f1M 0.9036` for `ref20` **[Run]** |
| `run_experiments.py … --status` | `done <job>` / `todo <job>` lines **[Run]** |
| `eval_test_split.py` | `TEST macro F1`, per-class F1/recall, support; rewrites `test_report.json`, `gates.json` (adds `G6_source`) **[Run]** |
| `flops_report.py` | one line per model: params, conv/linear MFLOPs, scan MFLOPs, total GFLOPs, core applications **[Run]** |

---

## 9. Troubleshooting

Every row has evidence in the project; generic problems without project evidence are not listed.

| Problem | Likely cause | Verification | Solution |
|---|---|---|---|
| `pip install -r requirements.txt` fails on `file:///home/conda/…` | the file was frozen from a conda environment and points at local build paths | open `requirements.txt` | use the minimal environment of [§3](#3-environment-setup) |
| PyTorch cannot use an RTX 50-series (Blackwell) GPU | a build without `sm_120` kernels; the README names cu124 as an example | `python -c "import torch; print(torch.cuda.get_arch_list())"` | install a cu128 build (README) |
| `ImportError` when preparing histology | `spectral` not installed (`open_envi_lazy` requires it) | `python -c "import spectral"` | `pip install spectral` |
| `FileNotFoundError: Missing required file: …/X_train.npy` | `--data_dir` points at the build root instead of `…/hsi` or `…/rgb`, or the build is incomplete | `ls <data_dir>` | point at the folder that holds `X_*.npy` |
| HSI data trained as RGB (`modality rgb`, a `MODALITY:` warning) | no file named exactly `wavelengths.npy` in `--data_dir` | read the warning; `ls` | rename the wavelength file to `wavelengths.npy` |
| `no X_test.npy …` / test evaluation skipped | missing test split, or no `X_val.npy` (test then serves as validation) | console warnings at the end | prepare a three-way split (e.g. `--split 80_10_10` or `70/15/15`) |
| ROI filtering has no effect | per-patient GeoJSON names vs per-capture lookup ([HOW_IT_WORKS §5.3](HOW_IT_WORKS.md#53-preprocessing)) | patches per full-size capture = 4,914 | known limitation; the paper build has it too |
| Runs appear in an unexpected `experiments/` folder | the run directory is relative to the working directory | `pwd` | run from the repository root |
| `--resume` crashes: `UnpicklingError: Weights only load failed … numpy._core.multiarray.scalar` | PyTorch ≥ 2.6 defaults `torch.load(weights_only=True)`; `training/trainerg_v4.py:822` does not override it | reproduced **[Run]** on torch 2.14 | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python train.py … --resume <run>/checkpoints/latest.pt` works **[Run]** (only for checkpoints you trust) |
| After a resume, `history.json` lacks the earlier epochs and the first resumed epoch is marked best | checkpoints do not embed the history, and the best-metric value is not restored | seen **[Run]**: history `[3]`, "Saved: epoch_0003.pt (BEST)" | keep a copy of `history.json` and `best_model.pt` before resuming |
| A failed resume changed `config.json` of the original run | the pipeline rewrites `config.json` and preflight reports before loading the checkpoint | seen **[Run]** | back up `config.json` before resuming |
| `--resume` writes into the wrong folder | the run directory is taken as the grandparent of the resume path | [`npy_data.py:174`](training/npy_data.py#L174) | resume from `<run>/checkpoints/latest.pt` (not `best_model.pt`) |
| `test_report.json` lost its FLOP count | `scripts/eval_test_split.py` rewrote it through `TrainerG_v12`, whose FLOP call fails | seen **[Run]** | `python scripts/flops_report.py --run_dir <run> --write_into_test_report` **[Run]** |
| Wrong results or huge memory when re-evaluating a 224 × 224 PAD run with `eval_test_split.py`, `heldout_eval.py`, `export_recon_samples.py`, `latent_space.py` or the band-decimation scripts | their shared `_rebuild_model` ignores the run's recorded `patch_size` and uses the preset stride (1 for `recursive`) | [`scripts/eval_test_split.py:75`](scripts/eval_test_split.py#L75) **[Code]**; effect **[Inferred]** | only `flops_report.py` corrects this; for PAD runs rely on the `test_report.json` written during training |
| CUDA out of memory | recursive core holds tokens × batch through 63 applications | `train.py` prints advice on OOM **[Code]** | `--patch_size` × 2 (quarter of the tokens) or `--batch_size` ÷ 2 |
| `OSError: [Errno 28] No space left on device` during compilation although `df` shows space | `/tmp` is a RAM-backed tmpfs | `findmnt /tmp` | handled automatically by `training/compile_cache.py`; or `--compile off` |
| `OSError: AF_UNIX path too long` in DataLoader workers | `TMPDIR` set to a long path | `echo $TMPDIR` | unset `TMPDIR` or use a short path (`compile_cache.py` docstring) |
| "Too many open files" at test time | many workers on the large test split | console | automatic single-process retry **[Code]** |
| `DATASET_STORAGE_ERROR` | truncated/corrupt `.npy`, unreadable disk region, or mismatch with `dataset_manifest.json` | `dataset_failure/` in the run dir | re-prepare (resumable) or fix the disk |
| `DatasetLeakageError` | a patient or identical patch appears in two splits | `leakage_report.json` | re-prepare with patient grouping; `--allow_leakage` only for diagnosis |
| Class-coverage failure at startup | a class missing from train or validation | console | change split/seed; `--allow_missing_classes` only for diagnosis |
| `SPLIT_DRIFT_ABORT` (gate G8) | normalized validation/test statistics far from train | `split_drift_report.json` | inspect the named patients; `--on_split_drift warn` (paper default) |
| Host RAM exhausted early in a run on a dataset with large patches | gate G8 samples 8,000 patches per split and keeps every normalized value in memory **[Code]**; the effect on large patches is **[Inferred]** | `--drift_check_patches` default 8000 | lower `--drift_check_patches` |
| `REPRESENTATION_COLLAPSE` / `RECONSTRUCTION_DETACHED` | gates G1/G2/G9 or G7 failed | `numerical_failure/sensitivity_failure.json`, `gate_g7_report.json` | usually a changed representation flag; restore the defaults |
| `TRAINING_ABORTED_NUMERICAL_INSTABILITY` | too many non-finite batches | `numerical_failure/` | `--safe_mode` (fp32, eager, 0 workers, checks on) to diagnose |
| `--run_tag must be alphanumeric with dashes` | `_` or other characters in the tag | — | use dashes |
| `--stage applies to the profiles built on OPTIMAL_DEFAULTS` | `--stage` with `pipeline_defaults` or `medmamba_protocol_norecon` | — | use `paper_recipe` or `pad_ufes_best_norecon` |
| `--recon_out_activation … is inconsistent with --normalization` | sigmoid output with a z-score target, or the reverse | — | omit the flag (`auto`) |
| `AssertionError: unknown fusion_type 'se_gate'` with `--architecture efficient` | the parser default `se_gate` is rejected by `build_model` | `documentations/15_v18_commands.md` **[Config/Doc]** | add `--fusion_type gated` |
| Importing a script regenerates paper figures | no main guard in `make_fig_bands_v18.py`, `tables_v18_short.py` | — | run them as scripts only, never import them |
| `git show a94ccc3:…` (from `archive/README.md`) fails | `origin/master` is a single squashed commit; `a94ccc3` exists only on the author's local branches | `git cat-file -t a94ccc3` | the deleted files are not recoverable from the public repository |

Permission errors: none are documented in the project. The run needs write access to
`experiments/` in the working directory and, with `--compile on`, to `<repo>/.cache/compile` **[Code]**.

---

## 10. Research reproducibility

For the paper's reference run `ref20` unless stated.

| Item | Value | Status |
|---|---|---|
| Dataset version | HistologyHSI-BC-Recurrence, TCIA "Version 1", DOI 10.7937/6KPY-YT49 | available [Config/Doc] |
| Prepared build | `data/hsi_v8-80_10_10_importance-new`; command in `prep_log.txt`; sha256 per file in `dataset_manifest.json` | available on the author's machine only [Artifact]; not redistributed |
| Data split | patient-grouped, stratified by rarest class, seed 42: 35/5/5 patients; `split_assignment.json` lists every patient | available [Artifact]; reusable with `--split_file` [Code] |
| Band selection | 32 bands, listed in `selected_band_indices.npy` / `wavelengths.npy` | available [Artifact]; **chose bands with held-out labels** [Code] |
| Random seeds | `--seed 42` (reference); 1, 7, 13, 23 for the other seeds; validation subset uses seed + 1 | available [Artifact] |
| Determinism | `--deterministic` was **off** and `torch.compile` **on** [Artifact]; with `--deterministic` off the pipeline turns `cudnn.benchmark` on ([`train_pipeline.py:276`](training/train_pipeline.py#L276)) [Code] (the `seeding` block of `config.json` is written before that line, so it shows `False`) | not bit-reproducible; doc 18 reports that re-running the same seed moved balanced accuracy by about 0.06 points [Config/Doc] |
| Model configuration | every flag in `config.json` → `cli_args`; 446,409 parameters | available [Artifact], [Run] |
| Hyperparameters | focal loss γ 1.5, class-weight power 0.75, λ_mse 0.1, λ_sam 0.1, dropout 0.1, weight decay 0.05, clipping 1.0, EMA 0.9995 | available [Artifact] |
| Batch size | 256 | available [Artifact] |
| Learning rate | 3e-4, warmup 587 steps, cosine to 0 over 19,560 steps | available [Artifact] |
| Epochs | 20 (best epoch 7) | available [Artifact] |
| Optimizer | AdamW, weight-decay groups, default betas/eps | available [Code] |
| Scheduler | warmup-cosine, stepped per optimizer step | available [Code], [Artifact] |
| Hardware | RTX 5060 Ti 16 GB, 32 GB RAM | available [Artifact] |
| Software | torch 2.11.0+cu128 (in `config.json`); Python 3.11 per `mamba-spec.txt` | partly: the environment files are not portable [Inferred] |
| Checkpoint used | `best_model.pt`, EMA weights, epoch 7 | available [Artifact]; **not in the public repository** |
| Normalization statistics | recomputed from `X_train.npy` (5,000 patches, seed 42) | not stored; recomputable only with the original build [Code] |
| Evaluation procedure | G6: full test split (348,894 patches), fp32, `best_model.pt` | available [Code], [Artifact] |
| Metrics | `test_report.json` (see [HOW_IT_WORKS §8.2](HOW_IT_WORKS.md#82-metrics)) | available [Artifact] |
| Output files | run directory ([§8.2](#82-run-directory)) | available [Artifact] |
| Code version | `efc2afd`; the reference run itself was produced by the archived `train_example_v18.py`; `train.py` resolves the same `cli_args` (`tests/test_sweeps.py`) and trained identically on GPU in a 1-epoch parity check | parity [Code], [Run]; GPU parity [Config/Doc] (`archive/README.md`, `logs/smoke/`) |
| MedMamba baseline | trained by code in a separate local repository | **not reproducible from this repository** [Code], [Artifact] |
| Patient count | five validation and five test patients; DCIS occurs in 5 training patients, 1 validation patient (197) and 1 test patient (136) | small held-out set [Artifact] (`capture_index.json`) |

---

## Verified facts

1. The default profile is `paper_recipe`, but the bare command is **not** the paper command: without the typed flags it trains with batch 32, 200 epochs and no SAM term **[Run]**.
2. Run directories are created under `experiments/` relative to the working directory **[Code]**.
3. The CPU test suite passes: 757 tests **[Run]**.
4. `--resume` fails on PyTorch ≥ 2.6 without `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, and even then loses earlier history and resets the best metric **[Run]**.
5. `scripts/eval_test_split.py` overwrites `test_report.json` with a broken FLOPs block; `flops_report.py --write_into_test_report` repairs it **[Run]**.
6. Normalization statistics are not saved with a run; inference and re-evaluation recompute them from the training `X_train.npy` **[Code]**, **[Run]**.
7. `requirements.txt` contains non-portable local `file://` references and a CUDA-specific torch pin without an index URL **[Code]** (file content).
8. All jobs of `sweeps/paper_runs.py` are finished on the author's machine except `k2-control` **[Run]** (status check).
9. The recorded paper preparation command and its 52-minute duration are in the build's `prep_log.txt` **[Artifact]**.

## Unknown / not verified

1. **Installation from the environment files** — not executed; expected to fail because of the `file://` lines (see [§3](#3-environment-setup)). The minimal environment in §3 was not executed either.
2. **GPU training with `train.py`** — not run in this audit; GPU parity with the archived scripts is reported in `archive/README.md` and `logs/smoke/` **[Config/Doc]**.
3. **Data preparation from raw files** — not run; the commands are the recorded ones.
4. **PAD-UFES-20 training results** with the current profiles — not checked.
5. **Exact reproduction of any reported number** — needs the GPU, the original build and its normalization statistics; runs were not deterministic.
6. **Behaviour on Python 3.13 / PyTorch 2.14 on GPU**, and on Windows or macOS.
7. **Standalone scripts on runs whose stride differs from the preset** (224 × 224 PAD runs, hierarchical RGB runs) — the mismatch is visible in code; its numerical effect was not measured.
8. **Minimum hardware** — the project states none; only the author's machine is known.
9. **Deleted archive files** — `archive/README.md` recovery commands depend on commit `a94ccc3`, which is not on `origin/master`.
10. **Public access to the remote repository.**
11. **The MedMamba baseline code** — outside this repository; not inspected in this audit.
