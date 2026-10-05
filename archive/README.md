# archive/ — the frozen reference code

Every file here was replaced by a live script, then moved here **byte-for-byte unchanged**
(`git mv`). Nothing live imports them. They stay for two reasons:

- **They are the reference the tests compare against.** Every one is on
  `tests/test_frozen_files_untouched.py:FROZEN_GLOBS`, which requires each file to exist
  and be unchanged. That is what keeps the audited runs (`20260903_180812`,
  `20260904_010453`) and the paper's runs reproducible.
- **Frozen files cite them by name** in their docstrings, and frozen files cannot be edited
  to update a citation.

To run one, put both the repository root and this folder on the path. The files import each
other by their old module names:

```bash
PYTHONPATH=.:archive python archive/train_example_v18.py --data_dir ...
```

They also import old `training/` module names (`training.gates_v16`,
`training.npy_dataset_v16`, …). `RENAMED_MODULES` in `training/__init__.py` keeps those
names working.

They also use the model's names from before the 2026-10-02 rename (`gmedmamba.py` became
`medmamba_ss_trm.py`; `GMedMamba` became `MedMambaSS`, `GMedMambaRecursive` became
`MedMambaSSTRM`). `archive/gmedmamba.py` is the one file here that is not frozen code: it
answers `from gmedmamba import ...` with the renamed objects. The two wrapper classes they
import from `training/` keep their old names as aliases at the end of
`training/reconstruction_head.py` and `training/reconstruction_head_v2.py`.

## What is here

| file | replaced by | checked by |
|---|---|---|
| `train_example_v18.py` | `train.py --profile paper_recipe` (the default) | produced every number in the paper; `test_train_parity.py`, `test_train_e2e_parity.py` |
| `train_example_v16.py`, `_v16_recon.py` | `train.py --profile pipeline_defaults` | the same two tests |
| `train_example_v16_original.py` | `train.py --profile medmamba_protocol_norecon` | the same two tests |
| `train_example_v16_optimal.py` | `train.py --profile pad_ufes_best_norecon` | the same two tests |
| `train_example_v16_optimal_recon.py` | no profile; kept because `train_example_v18.py` is built on it | imported by `train_example_v18.py` |
| `train_example_v15.py` | `training/train_cli.py`, `training/train_preflight.py` (verbatim copies) | `test_live_copies_match_archive.py`; the audited runs `20260903_180812`, `20260904_010453` |
| `train_example_v6.py` | `training/npy_data.py` (verbatim copies) | `test_live_copies_match_archive.py` |
| `train_example_v7.py` | `training/reconstruction_head_v2.py:MedMambaSSTRMRawReconWrapper` (verbatim copy) | `test_live_copies_match_archive.py` |
| `train_example_v14.py` | `train.py` | frozen; nothing imports it |
| `trainerg_v3.py` | `training/trainerg.py` | frozen; nothing imports it |
| `prepare_histologyhsi_bc_v3.py` … `_v8.py` | `prepare_histologyhsi_bc.py` (v8's behaviour) | frozen; equivalence below |
| `prepare_pad_ufes_20_v4.py` … `_v6.py` | `prepare_pad_ufes_20.py` (v6's code) | frozen |

`prepare_histologyhsi_bc_v3.py` (and `_v4.py`, a 1 KB wrapper around it) imports
`training.dataset_audit` inside a `try`. That is why `training/dataset_audit.py` stays in
`training/` although no live script uses it: without it, v3 would print "could not import"
and silently skip writing `dataset_diagnostics.json`.

## How equivalence was checked

- **Training (2026-09-28):** `tests/test_train_parity.py` resolves every `train.py` profile
  against the old entry point's parser. `tests/test_train_e2e_parity.py` trains both sides
  for two epochs on a tiny synthetic set and requires identical histories, predictions and
  gates. On 2026-09-29 the same comparison ran on the GPU (`sweeps/smoke.py all`, logs in
  `logs/smoke/`): identical.
- **Training copies (2026-09-29):** `tests/test_live_copies_match_archive.py` requires the
  source of every copied definition to equal its original here. It also imports the
  `train.py` chain in a fresh interpreter without `archive/` on the path.
- **Preparation (2026-09-23):** every definition in `prepare_histologyhsi_bc.py` was
  AST-compared with its original (identical apart from one `global _find_rgb_image` and one
  dropped self-import of `CAPTURE_RE`). `prepare_histologyhsi_bc_v8.py` and the unified
  script were then run on a synthetic ENVI corpus in five configurations. Every `.npy` was
  byte-identical. Every JSON matched apart from timestamps, the output path and the random
  `uuid4` shard prefix (`training/prep/parallel.py:131`).

## Deleted on 2026-10-01

These were here without being frozen, run or tested. They are deleted from the working
tree but kept in git history. Commit `a94ccc3` is the last one that has them, all under
`archive/`:

```bash
git show a94ccc3:archive/train_example_v13.py          # print one
git checkout a94ccc3 -- archive/train_example_v13.py   # restore one
```

| deleted | what it was |
|---|---|
| `train_example_v8.py` … `_v13.py` | training entry points before v14; v10 introduced the TRM deep-supervision loop |
| `model/gmedmamba_v1.py` … `_v3.py` | three older full copies of `gmedmamba.py` (now `medmamba_ss_trm.py`) |
| `experiment.py`, `trainer.py`, `ddp.py`, `logger.py`, `tb.py`, `report.py`, `data.py`, `lr_finder.py` | the original `Trainer.fit()` pipeline, from before `TrainerG` (were `training/`) |
| `checkpoint.py`, `gmedmamba_io.py` | that pipeline's checkpointing (`checkpoint.py` was `training/`) |
| `aggregate_report.py`, `capacity_search.py`, `hsi_quality_check.py`, `root_cause_report.py` | standalone tools from the v4–v16 plans (were `training/`) |
| `prepare_pad_ufes20.py` | the PAD-UFES-20 preparation before `training.prep` |
| `bench_v15_throughput.py` | replaced by `scripts/gpu_tune.py` |
| `run_multiseed.py`, `run_classification_ablation.py`, `run_full_ablation_checklist.py` | launchers for `train_example_v8/v11.py` (were `training/`); now `run_experiments.py` + a sweep file |
| `v20_queue.sh`, `v20_done.py` | now `run_experiments.py sweeps/paper_evidence_upgrade.py` and `run_experiments.py --find` |
| `short_runs_v18.sh`, `short_runs_v18b.sh` | now `sweeps/paper_runs.py` stages `k2`, `baselines`, `zeroshot` |

Frozen docstrings that cite a deleted file now point into that commit. Two examples:
`training/run_naming.py:10` cites `train_example_v13.py`, and `medmamba_ss_efficient.py:30`
cites `training/capacity_search.py`.
