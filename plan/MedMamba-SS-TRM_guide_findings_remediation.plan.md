# MedMamba-SS-TRM — Remediation Plan for the Project-Guide Findings

**Status:** plan only, nothing implemented. Written 2026-10-05.
**Source of the findings:** the audit behind `HOW_IT_WORKS.md` and `HOW_TO_RUN.md` (commit `f8735d2`,
branch `docs/project-guides`), in particular `HOW_TO_RUN.md` §9 (Troubleshooting) and
`HOW_IT_WORKS.md` §8.4 (Mentioned but not implemented), plus three facts established while writing
this plan (marked **new**).
**Base commit for every check below:** `f8735d2` (`docs/project-guides`). `master` is at `efc2afd`.
**Baseline:** `python -m pytest -q` → 757 passed (CPU, Python 3.13.15, torch 2.14.0+cpu).

Evidence labels follow the guides: **[Code]** read in source, **[Run]** executed during the audit,
**[Artifact]** read from `experiments/` / `data/` on the author's machine, **[Inferred]**, **[NOT VERIFIED]**.

---

## 0. Summary

| ID | Finding | Priority | Recorded results affected? | Fix in one line |
|---|---|---|---|---|
| R1 | `--resume` crashes on torch ≥ 2.6; when forced, it drops history, resets the best metric, and a failed attempt rewrites the original `config.json` | P0 | **No** — no run in `experiments/` was resumed [Artifact] | numpy-free LR lambda + a `TrainerG.load_checkpoint` override + resume validation in the pipeline |
| R2 | `scripts/eval_test_split.py` overwrites the run's `test_report.json` / `gates.json` / `config.json`, and its FLOPs block comes back as an error | P0 | **No** — no run carries `G6_source` [Artifact] | evaluate through `TrainerG(record_config=False)` into a new rerun folder; `--in_place` opt-in |
| R3 | The shared `_rebuild_model` ignores the run's `patch_size` **and** (**new**) its `global_stats`; two scripts each carry a partial workaround | P0 (latent) | **No** — 31 runs have a stride ≠ preset and 6 use fixed stats, none has an artifact from the affected scripts [Artifact] | one rebuild + statistics module used by every script |
| R4 | Environment files are not portable (23 `file://` lines, CUDA torch pin without index URL, conda torch 2.5.1/cu121) | P1 | n/a | portable `requirements.txt` + `environment.yml`; keep the freeze as a record |
| R5 | Normalization statistics are not saved with a run; there is no inference script | P1 | n/a | save `normalization_stats.npz`; add `scripts/predict.py` |
| R6 | `scientifically_qualified` is promised by `--help` but never written | P2 | n/a | write the v15 dict into `config.json` |
| R7 | ROI filtering is silently inert; (**new**) the annotations are capture footprints in another image's coordinate frame, so a lookup fix alone cannot work | P1 (guard) / decision (mask) | the paper build contains no ROI filtering — already disclosed (manuscript audit F40) | a loud guard in the next build's adapter; any real tissue mask is a research decision |
| R8 | Stale documentation: recovery commands point at `a94ccc3` (not public), stale quick start, guides not linked | P1 | n/a | documentation edits |
| R9 | Two scripts run their whole body on `import` (no main guard) | P2 | n/a | add `main()` + guard |
| R10 | `python train.py --data_dir <hsi>` (default profile) is **not** the paper command | P1 | n/a | banner note (and optionally a `paper_hsi` profile) |
| R11 | Band selection saw held-out labels | research | yes — all HSI numbers, already disclosed as provisional | already planned in `FUTURE_EXPERIMENTS.md` F1–F3; this plan only makes the tooling safe first |
| R12 | Small decisions: HMI-LUSC tracked-but-ignored, `.gitignore` ignores itself, recurrence labels, test-time precision | P2 | n/a | decide, then one-line edits |

---

## 1. Ground rules for the implementer

### 1.1 Freeze status of every file this plan touches

`tests/test_frozen_files_untouched.py:FROZEN_GLOBS` is the enforced list. Two more categories are
protected by convention or by tests, and are treated the same way.

| File | Status | Allowed |
|---|---|---|
| `training/trainerg_v4.py` … `trainerg_v11.py`, `training/losses.py`, `training/augmentation.py`, `training/dataloader_config.py`, `training/run_naming.py`, `training/prep/*`, `medmamba_ss_trm*.py`, `archive/*.py` | **FROZEN** (enforced) | import only |
| `training/trainerg_v12.py`, `training/trainerg_v13.py` | frozen by convention (`trainerg_v13.py:7-10`: "subclassed, never edited") | import / subclass only |
| `prepare_histologyhsi_bc.py`, `prepare_pad_ufes_20.py` | frozen by convention (they reproduce the v8 / v6 builds byte-for-byte, `archive/README.md`; `prepare_histologyhsi_bc_trainsel.py` docstring) | subclass only |
| `training/npy_data.py` (the four v6 copies), `training/train_cli.py` (`build_arg_parser`, `_add_v15_arguments`, `apply_safe_mode`, `EFFICIENT_FUSION_CHOICES`), `training/train_preflight.py` (five functions), `training/reconstruction_head_v2.py` (raw wrapper) | verbatim copies pinned by `tests/test_live_copies_match_archive.py` | do not edit those definitions; new code in the same file is fine |
| `training/trainerg.py`, `training/train_pipeline.py`, `training/train_banner.py`, `training/train_profiles.py`, `prepare_histologyhsi_bc_trainsel.py`, `scripts/*.py` (except `shallow_baseline.py`), docs, env files | editable | edit; keep the parity tests green |

**Invariant check after every commit:**

```bash
MEDMAMBA_SS_TRM_FREEZE_BASE=f8735d2 python -m pytest tests/test_frozen_files_untouched.py -q
git diff --name-only f8735d2 -- training/trainerg_v1[23].py prepare_histologyhsi_bc.py prepare_pad_ufes_20.py   # must print nothing
```

### 1.2 What the parity tests allow [Code]

`tests/test_train_parity.py:88-95` and `tests/test_train_e2e_parity.py:112-118` compare **only the
archived side's keys**. Therefore: new CLI flags, new `config.json` keys and new files in the run
directory are parity-safe. Changing an existing default, a history value, a prediction or a gate
result is not. `tests/test_sweeps.py` pins the `cli_args` of 27 real paper runs: no existing flag
may change its default.

### 1.3 Other constraints

- **Do not push `a94ccc3` or any pre-squash branch.** The squash of `origin/master` was deliberate,
  and the old history contains local-only material.
- Do not add local-only dataset names to tracked files.
- One commit per item (or per sub-item where marked). Run `python -m pytest -q` after each and
  record the count (starts at 757).
- Sandbox limits: no GPU and limited RAM. Everything marked **CPU** below was checked to be light;
  items marked **host** need the GPU machine.

---

## 2. Decisions needed from the author

The implementer proceeds with the **recommended** option unless told otherwise, except D3 and D6,
which change data or what is public and must be answered first.

| ID | Question | Options | Recommended |
|---|---|---|---|
| D1 | Where to implement | (a) fast-forward `master` to `docs/project-guides`, then branch `fix/guide-findings`; (b) branch from `docs/project-guides` | (a) |
| D2 | Old environment files (R4) | (a) move them to `env/*.freeze-2026-09-25.*` and write portable ones; (b) leave them and add `requirements-portable.txt` | (a) |
| D3 | Tissue / background mask (R7b) | (a) no mask; builds keep background, only the guard is added; (b) design an explicit mask for the next build and retrain | (a) for now — any mask changes every number. **Must be answered.** |
| D4 | Paper-command guard (R10) | (a) banner note only; (b) note + a `paper_hsi` profile | (a); (b) is optional |
| D5 | `.gitignore` contains the line `.gitignore` (added in `f8735d2`) | remove it / keep it | remove — ignoring a tracked file has no effect and misleads |
| D6 | HMI-LUSC (`prepare_hmi_lusc_v20.py`, tracked but listed in `.gitignore`) | (a) keep it tracked, drop the `.gitignore` line; (b) untrack it with its test and the LUSC sweep stages | **Must be answered.** |
| D7 | `--label_source recurrence` | (a) mark unsupported in the docs; (b) repair the workbook lookup | (a) — even repaired, 45 patients leave ~4–5 in a test split |
| D8 | Test-time precision (G6 runs in fp32, validation in bf16) | (a) keep fp32 and document it; (b) add opt-in `--test_amp match` | (a) |

---

## 3. Items

### R1 — `--resume` (P0)

**Evidence**

1. Crash [Run, torch 2.14.0+cpu]:
   `_pickle.UnpicklingError: Weights only load failed … Unsupported global: GLOBAL numpy._core.multiarray.scalar`,
   raised at `training/trainerg_v4.py:822` (`torch.load(path, map_location=self.device)` with no
   `weights_only=False`). PyTorch 2.6 changed the default to `weights_only=True`.
2. **Root cause of the NumPy scalars** [Run]: in a real checkpoint they sit at
   `optimizer_state.param_groups[*].lr` and `scheduler_state._last_lr[*]` (both `np.float64`).
   `build_warmup_cosine_scheduler.lr_lambda` returns
   `0.5 * (1.0 + np.cos(np.pi * progress))` (`training/train_pipeline.py:118`), a NumPy float, and
   `LambdaLR` multiplies it into every learning rate. The existing resume tests
   (`tests/test_trainerg_v11.py:301-374`) do not use this scheduler, which is why they pass.
3. `training/trainerg_v10.py:295` also calls `torch.load(ema_path, …)` without `weights_only`.
4. With `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, resume runs, but [Run]:
   - `history.json` keeps only the resumed epochs (`[3]` after resuming a 2-epoch run). Cause:
     `TrainerG_v11` saves without history (`embed_history_in_checkpoint=False`, `trainerg_v11.py:101`),
     so `TrainerG_v4.load_checkpoint` sets `self.history = []` (`trainerg_v4.py:833`).
   - The first resumed epoch is marked BEST. Cause: `best_metric_value` is saved
     (`trainerg_v11.py:565`) but never restored; it stays at `-inf` from `trainerg_v5.py:57`.
   - The EMA shadow always comes from `checkpoints/latest_ema.pt` (`trainerg_v10.py:293-296`),
     whatever checkpoint was passed.
5. A failed resume still rewrites the original run's `config.json` (`epochs` became 3) and writes
   `failure_class.json` [Run]. Cause: the pipeline writes `config.json` (`train_pipeline.py:587`)
   and every preflight report before `trainer.load_checkpoint` (`:623`).
6. `setup_experiment_dir` takes the run directory as the resume path's grandparent
   (`training/npy_data.py:174`): `--resume <run>/best_model.pt` resolves to `experiments/` [Code].

**Fix**

- **R1a** — `training/train_pipeline.py:118`: return `float(0.5 * (1.0 + np.cos(np.pi * progress)))`.
  Keeping `np.cos` keeps every value bit-identical; only the type changes. New checkpoints then load
  with `weights_only=True`.
- **R1b** — `training/trainerg.py`, new `TrainerG.load_checkpoint(path)`:
  1. Shadow `torch.load` for the duration of `super().load_checkpoint(path)` so that
     `weights_only` defaults to `False`. This is the same scoped-shadow pattern as
     `trainerg_v12.py:283-298`, and it covers both v4's and v10's calls. Justification for the
     docstring: these checkpoints are written by this trainer on this machine. Old checkpoints
     (which contain NumPy scalars) must keep loading.
  2. Keep the checkpoint dict the shadow loaded, and restore `best_metric_value` (and
     `best_val_acc`) from it.
  3. If `self.history` is empty and `<exp_dir>/history.json` exists, load it and keep the rows with
     `epoch <= checkpoint["epoch"]`. Restore `_single_class_streak` from the last row's
     `single_class_streak`; recompute `_epochs_without_improvement` from the rows when early stopping is on.
  4. Read `epoch` from `checkpoints/latest_ema.pt`. If it differs from the checkpoint's epoch, log a
     WARNING and re-register the EMA from the loaded weights (`self.ema_helper.register(self.model)`).
  5. Return the epoch (the contract `tests/test_trainerg_v11.py:301` pins).
- **R1c** — `training/train_pipeline.py`, before `setup_experiment_dir` (`:291`), when `--resume`:
  - the path must exist and its parent must be named `checkpoints`, else
    `SystemExit("--resume must point to <run>/checkpoints/<file>.pt")`;
  - load it once with `map_location="cpu", weights_only=False` so that a bad file fails **before**
    anything is written.
  Before writing `config.json` (`:587`): if `config.json` already exists, copy it to
  `config.before_resume_<YYYYmmdd_HHMMSS>.json`, and add
  `"resumed_from": {"checkpoint": <path>, "epoch": <n>}` to the new config.

**Tests** — new `tests/test_resume.py` (CPU, tiny synthetic NHWC set like the e2e fixture):

| Test | Asserts |
|---|---|
| `test_checkpoint_loads_weights_only` | after a 1-epoch run, `torch.load(latest.pt, weights_only=True)` succeeds (R1a) |
| `test_resume_without_env_override` | 2 epochs, then `--resume …/checkpoints/latest.pt --epochs 3`, completes without `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` |
| `test_resume_keeps_history` | `history.json` epochs == `[1, 2, 3]` |
| `test_resume_restores_best_metric` | a `TrainerG` loading a checkpoint saved with `best_metric_value=0.9` reports 0.9 |
| `test_old_checkpoint_with_numpy_scalars_loads` | a checkpoint whose scheduler state holds `np.float64` still loads (R1b shadow) |
| `test_resume_path_must_be_in_checkpoints` | `--resume <run>/best_model.pt` → `SystemExit` naming the rule |
| `test_failed_resume_writes_nothing` | corrupt checkpoint → exit before writes; `config.json` bytes unchanged |
| `test_resume_backs_up_config` | `config.before_resume_*.json` equals the original |
| `test_ema_epoch_mismatch_warns` | resuming from `epoch_0001.pt` while `latest_ema.pt` is epoch 2 → WARNING + EMA re-registered |

**Verification:** CPU — the new tests, then the full suite (parity tests must stay green: R1a changes
the type only). Repeat the audit's manual repro without the environment variable. Host (optional): a
2-epoch real HSI run at `--train_subsample_frac 0.005`, resumed to 3 epochs.

**Docs:** `HOW_TO_RUN.md` §9 (four resume rows → current behaviour), Verified fact 4.

---

### R2 — `scripts/eval_test_split.py` overwrites reports and loses FLOPs (P0)

**Evidence**

- The script builds `TrainerG_v12` with `exp_dir=run_dir` (`scripts/eval_test_split.py:204-213`).
  `TrainerG_v11.evaluate_test_split` then writes `<run>/test_report.json` and
  `test_predictions.npz`, and calls `count_flops` with a 3-D shape (`trainerg_v11.py:717`), so
  `flops` becomes `{"error": "ValueError: not enough values to unpack (expected 4, got 3)"}` [Run].
  The script also rewrites `gates.json` (adds `G6_source`) [Run], and `TrainerG_v4.__init__`
  re-dumps `config.json` into `exp_dir` (`trainerg_v4.py:154`) [Code].
- `TrainerG.evaluate_test_split` already recounts FLOPs (`training/trainerg.py:88-108`), but the
  script does not use `TrainerG`.
- The same construction (and the same FLOPs error in their reports) is in `scripts/heldout_eval.py:135`,
  `scripts/eval_band_decimation.py:198` and `scripts/eval_band_decimation_fixed_eta.py:140`.
- `scripts/flops_report.py --run_dir X --write_into_test_report` repairs the block [Run].

**Fix**

1. In all four scripts, construct `TrainerG(..., record_config=False)` instead of `TrainerG_v12`.
   `record_config=False` stops `TrainerG_v13.__init__` from relabelling the run (`trainerg_v13.py:96`).
2. `scripts/eval_test_split.py`: new `--out_dir` (default `<run>/eval_rerun_<YYYYmmdd_HHMMSS>/`)
   used as the trainer's `exp_dir`. Pass the checkpoint explicitly (`checkpoint_path=`). Write
   `gates.json` there. New `--in_place` restores the old behaviour, now with recounted FLOPs.
   Remove the empty folders that `TrainerG_v4.__init__` creates in the rerun folder.
3. Update the module docstring: what it writes, and where.

**Tests** — new `tests/test_eval_test_split.py` (CPU; a 1-epoch synthetic run via `train.py`):
original `test_report.json`, `gates.json` and `config.json` are byte-identical after a default rerun;
the rerun's `test_report.json` has `flops.total_gflops_approx > 0` and no `error`; its
`sklearn_metrics` equal the original's; `--in_place` replaces the original. Existing tests that import
these scripts (`test_fixed_eta.py`, `test_check_eval_mode.py`, `test_latent_space_tuple_v18.py`) stay green.

**Docs:** `HOW_TO_RUN.md` §7.5 (remove the backup step and the warning), §8.3, §9 row; Verified fact 5.

---

### R3 — One model-rebuild and statistics path for every script (P0, latent)

**Evidence**

- `scripts/eval_test_split.py:75-114` (`_rebuild_model`) builds `cfg_overrides` from the
  representation flags only and never sets `patch_size`, so the preset stride is used: 1 for
  `recursive` (`training/config_presets.py:42-45`), 4 for hierarchical RGB (`:31,34,37`). Training
  applies the recorded value (`training/train_pipeline.py:470-471`).
- **new** — no script reads `--global_stats`: every one recomputes train-fitted statistics
  (`eval_test_split.py:154-159`, `heldout_eval.py:112`, `export_recon_samples.py:103-104`,
  `latent_space.py:148-149`, `eval_band_decimation.py:166-167`, `eval_band_decimation_fixed_eta.py:113-114`,
  `check_eval_mode.py:117`). Training uses fixed 0.5/0.5 for `--global_stats medmamba_fixed` and
  0/1 for `identity` (`train_pipeline.py:98-102`).
- Importers of `_rebuild_model`: `export_recon_samples.py:54`, `eval_band_decimation.py:72`,
  `eval_band_decimation_fixed_eta.py:89`, `heldout_eval.py:47`, `latent_space.py:58`,
  `check_eval_mode.py:94`, `flops_report.py:67`.
- Two partial workarounds already exist: `scripts/flops_report.py:98-137` (patches `_PRESETS` for the
  stride and drops `classifier_dropout`) and `scripts/heldout_eval.py:126-129` (drops `classifier_dropout`).
- **Audit [Artifact, 2026-10-05]:** 31 runs record a stride different from their preset (28
  recursive PAD/MIL runs: 8 or 4 vs 1; three `split` runs: 8 vs 4 on PAD-224, 1 vs 4 on histology RGB).
  6 runs use `global_stats medmamba_fixed`, all MedMamba-protocol PAD runs. None has `G6_source`,
  `heldout_v19/` or `band_decimation*/`; the one reconstruction export found
  (`20260913_021957_…`) was written in-run by `TrainerG_v13`. **No recorded number needs re-running.**
  Latent-space plots written by `scripts/latent_space.py` cannot be attributed from files [NOT VERIFIED]; they are plots only.

**Fix**

1. `training/train_pipeline.py`: factor the model construction of `_run` (`:463-514`: cfg overrides,
   `trm_kwargs`, `build_model_guarded`, reconstruction wrapper) into
   `build_model_from_args(args, modality, num_classes, in_channels) -> (base_model, model)`, and
   call it from `_run`. This is a behaviour-preserving move; the e2e parity tests prove it.
2. New `training/run_rebuild.py`:
   - `rebuild_model(cfg)` → `argparse.Namespace(**cfg["cli_args"])` →
     `build_model_from_args(ns, cfg["modality"], cfg["classes"], cfg["in_channels"])`. One
     construction path for training and for every tool, including the `patch_size` and the
     `classifier_dropout` handling.
   - `load_global_stats(cfg, run_dir)` → `normalization_stats.npz` if present (R5), else the same
     dispatch as `train_pipeline.GLOBAL_STATS` on `cli_args["global_stats"]` (train-fit recompute,
     0.5/0.5, or 0/1).
3. `scripts/eval_test_split.py`: replace the body of `_rebuild_model` with `return rebuild_model(cfg)`,
   keeping the name so the seven importers keep working. Switch every script's statistics block to
   `load_global_stats`. Delete the two workarounds (`flops_report.py:98-137` becomes a call to
   `rebuild_model`; `heldout_eval.py:126-129` goes).

**Tests** — new `tests/test_run_rebuild.py` (CPU):
- parametrized `(architecture, modality, recorded patch_size)` ∈ {(recursive, rgb, 8), (recursive, hsi, 1), (split, rgb, 1), (split, rgb, 8)} → rebuilt base model's `cfg.patch_size` equals the recorded value;
- a synthetic run trained by `train.py`: `rebuild_model(cfg).load_state_dict(best_model.pt["model_state"], strict=True)`, and the forward on `X_test` reproduces `test_predictions.npz["probs"]` (atol 1e-6);
- `global_stats medmamba_fixed` → statistics are exactly 0.5 / 0.5;
- presets unchanged: `scripts/flops_report.py --architectures split,recursive …` still prints 2,773,007 / 0.2781 GF and 446,409 / 6.2033 GF [Run baseline].

**Verification:** CPU — the tests above, the existing script tests, and the full suite. **CPU-feasible
on real data (ask before running in the sandbox; minutes):** re-evaluate
`experiments/20260914_015200_pad_optimal-224_recursive_*` with the fixed script into a rerun folder;
the metrics must equal its in-run `test_report.json` exactly (same checkpoint, same data). The old
code would rebuild at stride 1, i.e. 50,176 tokens per image.

**Docs:** `HOW_TO_RUN.md` §7.5 note, §7.6, §9 row, Unknown 7.

---

### R4 — Portable environment (P1)

**Evidence [Code]:** `requirements.txt` has 23 `package @ file:///home/conda/feedstock_root/…`
lines and pins `torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` with no index URL.
`mamba-spec.txt` installs conda `pytorch-2.5.1 … cuda12.1` (README: Blackwell needs cu128).
`environment_backup.yml` mixes `pytorch-cuda=12.1` with the pip cu128 wheels. Optional imports not
listed anywhere: `psutil` (`training/system_memory.py:58`, `training/utils.py:133`) and `umap`
(`training/plots.py:447`). Listed but not imported by live code: `torchvision`, `seaborn`,
`scikit-image`, `tqdm`, `tifffile`, `ImageIO`, `panda`.

**Fix (D2 = a)**

1. `git mv` the three files to `env/requirements.freeze-2026-09-25.txt`,
   `env/mamba-spec.freeze-2026-09-25.txt` and `env/environment.freeze-2026-09-25.yml`, plus a short
   `env/README.md`: these record the author's environment and are not installable elsewhere. The date
   is the files' mtime.
2. New `requirements.txt`:
   ```text
   # CUDA 12.8 wheels (RTX 50-series / sm_120 need cu128). For CPU-only, remove the next line and the +cu128 suffix.
   --extra-index-url https://download.pytorch.org/whl/cu128
   torch==2.11.0+cu128
   numpy==2.4.6
   scipy==1.17.1
   scikit-learn==1.9.0
   pandas==3.0.5
   pillow==12.3.0
   matplotlib==3.11.1
   spectral==0.25
   openpyxl==3.1.5
   pytest==9.1.1
   ```
   New `requirements-optional.txt`: `psutil`, `umap-learn`.
3. New `environment.yml`: `name: medmamba-ss-trm`, `channels: [conda-forge]`,
   `dependencies: [python=3.11, pip, {pip: [-r requirements.txt]}]`.
4. Replace `replicate_from_txt.txt` with a pointer to `environment.yml` (or delete it), and rewrite
   the README's "Installation" section.
5. New `tests/test_requirements_portable.py`: no `file://` in `requirements*.txt`, and every
   third-party top-level import of the live code (the AST scan used in the audit) is listed in
   `requirements.txt` or `requirements-optional.txt`.

**Verification — host only (network + GPU):** `mamba env create -f environment.yml` → the arch check
prints `2.11.0+cu128 12.8 [… 'sm_120']` → `python -m pytest -q` passes →
`run_experiments.py sweeps/smoke.py profiles` passes. Whether every pinned version exists on PyPI is
**[NOT VERIFIED]**; adjust pins only after a failed resolve, and record why.

**Docs:** README install; `HOW_TO_RUN.md` §1 steps 2–3, §3 (replace the warning with the new
route), §4 table, §9 first row, Unknown 1.

---

### R5 — Persist normalization statistics; add an inference script (P1)

**Evidence:** `config.json` holds no statistics [Artifact: reference-run keys]. They are re-derived
from ≤ 5,000 training patches (`training/npy_data.py:36-51`), so every tool needs the original
`X_train.npy`. The inference snippet in `HOW_TO_RUN.md` §7.6 reproduced a run's own test
probabilities to 6e-8 [Run].

**Fix**

1. `training/train_pipeline.py` after `:336`: when statistics exist,
   `np.savez(exp_dir / "normalization_stats.npz", global_mean=…, global_std=…)` and add
   `"normalization_stats": {"file": "normalization_stats.npz", "source": args.global_stats,
   "sample_cap": args.norm_stats_sample_cap, "seed": args.seed}` to `config.json`. Parity-safe (new
   file and key).
2. `load_global_stats` (R3) prefers this file.
3. New `scripts/predict.py`, built from the verified snippet: `--run_dir`, `--input` (`[N,H,W,C]`
   raw `.npy`), `--checkpoint` (default `best_model.pt`), `--batch_size`, `--device`, `--out`
   (`.npz` with `probs`, `pred`, `class_names`). It checks `C == config["in_channels"]`, takes
   wavelengths from the run's data directory or `--wavelengths`, and processes chunks.

**Tests (CPU):** the statistics file equals a fresh recompute; `scripts/predict.py` on the
synthetic run's `X_test.npy` reproduces `test_predictions.npz` (atol 1e-6); a channel mismatch exits
with a clear message.

**Docs:** `HOW_TO_RUN.md` §7.6 (the script replaces the snippet), §10 row "Normalization
statistics"; `HOW_IT_WORKS.md` §7 row 1; Verified facts.

---

### R6 — `scientifically_qualified` (P2)

**Evidence:** the `--skip_dataset_validation` help (`training/train_cli.py:215-219`, a pinned
verbatim copy, so the text stays) says the flag is recorded in `config.json`. `train_pipeline.py:572-586`
never writes it. The archived v15 wrote a dict (`archive/train_example_v15.py:1557-1569`).

**Fix:** add the same dict to the config in `train_pipeline.py`: `dataset_validated`,
`dataset_validation_level`, `seeded`, `deterministic`, `representation_gates_checked`,
`representation_gates_passed`, `test_evaluated`. **Test (CPU):** present on a synthetic run;
`--skip_dataset_validation` → `dataset_validated: false`. **Docs:** `HOW_IT_WORKS.md` §8.4 row goes.

---

### R7 — ROI filtering (P1 guard; D3 for any real mask)

**Evidence**

- The recorded paper command passes `--roi_min_frac 0.8` (`data/hsi_v8-80_10_10_importance-new/prep_log.txt`) [Artifact].
- `find_roi_geojson` (`prepare_histologyhsi_bc.py:777-784`) searches for files whose name contains
  `<patient>_<tissue>_x10_C<capture>`. The folder `01_03_HSI ROI_Annotations` holds **47
  per-patient files** `<patient>.geojson` with **674 features** named `<tissue>_C<nn>`; 641 of the
  644 captures in the build have a same-named feature [Artifact]. The lookup never matches.
- **new — the deeper problem.** Each feature is a polygon of about 3,000 × 1,900 px at coordinates
  around 40,000–53,000 [Artifact]. `rasterize_roi_mask` (`:754-774`) draws in the cube's own
  1004 × 600 pixel frame. Even with a matching lookup the mask would be empty, and
  `patch_coords` (`:829-842`) would drop every patch (mean coverage < 0.8) [Inferred]. The polygons
  most likely mark each capture's position on the whole-slide image, which this download does not
  include [Inferred; check the Sci Data descriptor].
- Consequence: the paper build has no ROI or background filtering (3,135,496 of 3,164,616 grid
  positions kept) [Artifact]. This is already disclosed (`paper/draft/MANUSCRIPT_AUDIT_v18_r2.md`, F40).

**Fix**

- **R7a (guard, now)** in `prepare_histologyhsi_bc_trainsel.py`, the adapter of the next build
  (`prepare_histologyhsi_bc.py` stays untouched):
  - count the captures that actually received a mask;
  - print `[roi] masks applied to N/644 captures` at the end of extraction, and record
    `roi_masks_applied` in the provenance (`dataset_statistics.json` / manifest `extra`);
  - new flag `--roi_missing {warn,error}`, default `warn` (so the recorded commands still run). When
    `--roi_min_frac > 0` and N == 0, print a multi-line WARNING with the reason above, or fail with `error`.
- **R7b (only if D3 = b):** design an explicit tissue mask from the capture itself (for example a
  threshold on `SyntheticRGBImage.png` saturation or on mean reflectance, like the glass filter in
  `prepare_hmi_lusc_v20.py`). It would be a new opt-in flag, a new build, an ablation and a
  retraining. It changes every HSI number, so it is out of scope until decided.

**Tests (CPU):** extend the synthetic-ENVI fixture (`tests/test_prep_hsi_fixture.py`) with a
per-patient GeoJSON folder → the warning says `0/N`; with `--roi_missing error` the prep exits non-zero.

**Docs:** replace the ROI warning in `HOW_IT_WORKS.md` §5.3 with the deeper reason; `HOW_TO_RUN.md`
§7.2 note and §9 row.

---

### R8 — Documentation references (P1)

**Evidence:** `a94ccc3` is only on the local branches `development-record-before-2026-10-04` and
`refactor/unify-scripts`, not on `origin/master` [Run]. It is cited by `archive/README.md:73-78`,
`documentations/README.md:8,239`, `documentations/02_train_example_v13.md:3` and
`documentations/17_train_and_sweeps.md:6`. `documentations/README.md` still has a quick start built on
`train_example_v16.py` commands. The README does not link the two new guides.

**Fix**

1. Replace each `git show a94ccc3:…` / `git checkout a94ccc3 -- …` with "deleted on 2026-10-01; not
   in the public repository history". `archive/README.md` keeps its list of deleted files.
   `PROJECT_HISTORY_AND_TECHNICAL_EVOLUTION.md` is a historical record; leave it as is.
2. `README.md`: add a "Documentation" line near the top pointing to `HOW_IT_WORKS.md` and
   `HOW_TO_RUN.md` (and fix "Installation" with R4).
3. `documentations/README.md`: replace the stale quick start with a pointer to `HOW_TO_RUN.md` §1.
4. `.gitignore` per D5 (and D6).

**Verification:** `git grep -n "a94ccc3" -- ':!PROJECT_HISTORY_AND_TECHNICAL_EVOLUTION.md' ':!plan'`
prints nothing. The link checker used for the guides passes on the edited files.

---

### R9 — Main guards (P2)

**Evidence:** `scripts/make_fig_bands_v18.py` (rewrites `paper/figures/results/fig_bands_v18.{png,pdf}`)
and `scripts/tables_v18_short.py` have no `if __name__ == "__main__":` [Code]. Nothing imports them;
`paper/cvc2027/make_figures_lnns.py:6,95` only mentions the first in text [Code].

**Fix:** move each module body into `main()` and add the guard.

**Verification (CPU, light):** run each script before and after → same output files (compare PNGs
by decoded-pixel hash; PDFs carry timestamps). `python -c "import scripts.make_fig_bands_v18"` leaves
every output's mtime unchanged. **Docs:** remove the warning in `HOW_TO_RUN.md` §7.7 and the §9 row.

---

### R10 — The default command is not the paper command (P1)

**Evidence [Run]:** `python train.py --data_dir <hsi build>` resolves to batch 32, 200 epochs and
`lambda_sam 0.0`. Every paper run typed
`--batch_size 256 --epochs 12|20 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986`
(`sweeps/paper_runs.py:REF`). `paper_recipe` itself must not change: `tests/test_train_parity.py`
pins it to the archived v18 parser, and `tests/test_sweeps.py` to 27 real runs.

**Fix (D4 = a):** in `training/train_banner.py:print_banner`, when the profile is `paper_recipe`,
the data directory has `wavelengths.npy`, `recon_mode == "latent"` and `lambda_sam` was not typed,
print:
`NOTE: this is not the paper's HSI command - paper runs also type --batch_size 256 --epochs 12|20 --lambda_sam 0.1 (sweeps/paper_runs.py)`.
No test pins the banner text [Run: grep].
**Optional (D4 = b):** a `paper_hsi` profile = the `paper_recipe` layers +
`{"batch_size": 256, "lambda_sam": 0.1, "epochs": 20}`, `name_suffix="_optimal_recon"`, with a test
that it resolves to the `cli_args` of the `ref20` job in `sweeps/paper_runs.py` (minus `run_tag`).

**Tests (CPU):** banner note present or absent in the four cases (`capsys`).

---

### R11 — Band-selection leak (research; tracked elsewhere)

Already covered by `FUTURE_EXPERIMENTS.md` F1–F3 and `sweeps/paper_evidence_upgrade.py` (`prep` is
done; `headline`, `heldout`, `cv` remain). This plan does not repeat those commands. It requires
only that **R1, R2, R3 and R5 land before any of those GPU runs start**, so that long runs can be
resumed and re-evaluated correctly, and that R7a is in the adapter before any new build.

---

### R12 — Small decisions (P2)

- **D5 / D6:** edit `.gitignore` accordingly. For D6 = b: also `git rm --cached
  tests/test_prepare_hmi_lusc_v20.py`, and make the `lusc-prep` / `lusc` stages of
  `sweeps/paper_evidence_upgrade.py` report "script not present" instead of failing.
- **D7 = a:** `HOW_IT_WORKS.md` §5.1.1: state that `--label_source recurrence` is unsupported on
  this download.
- **D8 = a:** `HOW_IT_WORKS.md` §8: state that G6 runs in fp32 by design. (D8 = b would be a new
  `--test_amp {off,match}`, default `off`.)

---

## 4. Execution order and commits

| Step | Commit | Contents | Checks |
|---|---|---|---|
| 0 | — | D1: fast-forward `master`, branch `fix/guide-findings`; record the baseline test count (757) | freeze test with base `f8735d2` |
| 1 | `fix: numpy-free LR schedule values (R1a)` | `train_pipeline.py:118` + `test_checkpoint_loads_weights_only` | e2e parity green |
| 2 | `fix: resume restores history and best metric, validates its path (R1b, R1c)` | `trainerg.py`, `train_pipeline.py`, `tests/test_resume.py` | full suite |
| 3 | `refactor: one model-rebuild and statistics path for training and tools (R3)` | `build_model_from_args`, `training/run_rebuild.py`, 7 scripts, `tests/test_run_rebuild.py` | e2e parity, script tests |
| 4 | `fix: eval_test_split writes a rerun folder with FLOPs (R2)` | 4 scripts → `TrainerG(record_config=False)`, `--out_dir/--in_place`, tests | full suite |
| 5 | `feat: save normalization statistics; add scripts/predict.py (R5)` | pipeline + `predict.py` + tests | full suite |
| 6 | `fix: record scientifically_qualified in config.json (R6)` | pipeline + test | full suite |
| 7 | `fix: ROI guard in the training-only-selection adapter (R7a)` | trainsel adapter + fixture test | prep tests |
| 8 | `feat: banner note when the default command is not the paper command (R10)` | banner (+ optional profile) + tests | parity, `test_profile_names.py` |
| 9 | `chore: main guards for two paper scripts (R9)` | 2 scripts | output comparison |
| 10 | `build: portable requirements and environment.yml (R4)` | env files + test + README | requirements test; **host** install |
| 11 | `docs: update guides, README and references after the fixes (R8, R12)` | `HOW_*.md`, `README.md`, `documentations/`, `archive/README.md`, `.gitignore` | link check, `git grep a94ccc3` |

Steps 1–4 are independent of the decisions and can start immediately.

---

## 5. Final verification

| Check | Where | Expected |
|---|---|---|
| `python -m pytest -q` | CPU | 757 + new tests, all passing |
| freeze invariant with base `f8735d2` | CPU | no frozen file changed |
| `git diff --name-only f8735d2 -- training/trainerg_v1[23].py prepare_histologyhsi_bc.py prepare_pad_ufes_20.py` | CPU | empty |
| CPU smoke run + resume without the environment variable | CPU | history `[1, 2, 3]`; original config backed up |
| re-evaluation of a PAD-224 run (R3) | CPU (ask first) or host | equals its in-run `test_report.json` |
| `run_experiments.py sweeps/smoke.py all` | host | all profiles train; gpu-parity IDENTICAL |
| fresh environment from `environment.yml` | host | arch list contains `sm_120`; tests pass |
| a 2-epoch real HSI run resumed to 3 | host | runs on torch 2.11 without overrides |
| guides | CPU | no row in `HOW_TO_RUN.md` §9 describes a fixed defect as current |

---

## 6. Out of scope

- Re-running any paper experiment: the audits above show that no recorded number depends on R1–R3.
- The band-selection retraining (R11) and any tissue mask (R7b): see D3 and `FUTURE_EXPERIMENTS.md`.
- Flattening the `TrainerG_v4…v13` chain (deferred by the author since 2026-10-01).
- Any change to `paper_recipe`'s defaults.
