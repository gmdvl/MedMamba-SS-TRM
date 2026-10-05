# 17 — `train.py` and `run_experiments.py`

Two scripts replace the six `train_example_v16*.py` / `train_example_v18.py` entry points and
the bash queues (`scripts/v20_queue.sh`, `scripts/short_runs_v18*.sh`). The old entry points are
in `archive/`, unchanged, and they remain the reference the parity tests compare against; the
queues were deleted on 2026-10-01 (`git show a94ccc3:archive/v20_queue.sh`).

## One run: `train.py`

```bash
python train.py --data_dir data/hsi_v8-80_10_10_importance-new/hsi    # default profile: paper_recipe
python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --run_tag abl-base
python train.py --help                                                # every flag
```

`--profile` picks the defaults. Any flag you type overrides them. The banner lists each flag you
typed whose value differs from the profile's default.

Each profile is named for what it is for. **A name ending in `_norecon` trains with
`recon_mode none`**: no reconstruction head and no MSE/SAM term. Every other profile reconstructs.
`tests/test_profile_names.py` holds the names to that.

| profile | good for | reconstruction | name before 2026-10-01 | replaced |
|---|---|---|---|---|
| `paper_recipe` *(default)* | **the paper's configuration**: reproducing or extending its numbers. `pad_ufes_best_norecon` plus latent reconstruction. HSI commands still pass `--lambda_sam 0.1` explicitly | latent, `lambda_mse 0.1` | `v18` | `train_example_v18.py` |
| `pad_ufes_best_norecon` | best metrics on PAD-UFES-20 (RGB); `--stage fit/balance/full` | **none** | `optimal` | `train_example_v16_optimal.py` |
| `medmamba_protocol_norecon` | comparing against MedMamba's own recipe: constant LR, fixed 0.5/0.5 stats, fp32, no clipping | **none** | `original` | `train_example_v16_original.py` |
| `pipeline_defaults` | the bare pipeline, v15/v16 parser defaults with nothing layered on top | latent, `lambda_mse 1.0` | `base` | `train_example_v16.py`, `_recon.py` |

`archive/train_example_v16_optimal_recon.py` has no profile of its own. It stays in `archive/`
because `train_example_v18.py` is built on it.
The old names still work. They resolve to the new profile, and the run prints the new name.
Run directory names did not change: `paper_recipe` runs still end in
`_optimal_recon`, `pad_ufes_best_norecon` runs in `_optimal`, and
`medmamba_protocol_norecon` runs in `_medmamba_protocol`. The parity tests pin those names,
and existing run paths contain them.

Each profile is a list of layers in [`training/train_profiles.py`](../training/train_profiles.py).
The layers are applied in order, and a later layer wins. The order is exactly the order in which
the old entry points called `set_defaults`.

Two things are flags now; before, one entry point achieved them by patching another module:

| flag | values | default |
|---|---|---|
| `--lr_schedule` | `warmup_cosine`, `constant` | `constant` on `medmamba_protocol_norecon`, else `warmup_cosine` |
| `--global_stats` | `train_fit`, `medmamba_fixed`, `identity` | `medmamba_fixed` on `medmamba_protocol_norecon`, else `train_fit` |
| `--early_stopping` (= `--early_stopping_enabled`) | `on`, `off` | `off` on every profile |
| `--stop_on_metric_stall` (= `--stop_on_val_acc_stall`) | `on`, `off` | `on` on `pipeline_defaults` only |

**`--early_stop_patience` does nothing unless `--early_stopping on`.** This is how every
optimal-family entry point behaved: `TrainerG_v12` has no patience rule. No paper number is
affected, because every v18 run was 20 epochs or fewer against a patience of 40.

### Where the code went

| file | contents |
|---|---|
| [`training/train_profiles.py`](../training/train_profiles.py) | the configurations, and the defaults resolved from the dataset (stride, subsample fractions, EMA rate) |
| [`training/train_cli.py`](../training/train_cli.py) | one parser: v15's flags plus every flag the v16-line entry points added, and `--profile` |
| [`training/train_pipeline.py`](../training/train_pipeline.py) | `train_example_v16._main`, with its five injection points turned into flags |
| [`training/trainerg.py`](../training/trainerg.py) | `TrainerG`: the v18 / optimal / original / v13 trainer methods in one class; the two stopping knobs are constructor arguments |
| [`training/train_banner.py`](../training/train_banner.py) | the preflight banner, one for every profile |

## Many runs: `run_experiments.py`

```bash
python run_experiments.py sweeps/paper_evidence_upgrade.py --list                # stages, groups, jobs
python run_experiments.py sweeps/paper_evidence_upgrade.py --status              # done / waiting / to run
python run_experiments.py sweeps/paper_evidence_upgrade.py headline --dry_run    # resolve every job, run nothing
python run_experiments.py sweeps/paper_evidence_upgrade.py p1                    # a group of stages, in order
python run_experiments.py sweeps/paper_evidence_upgrade.py depth --jobs 2        # two runs of a stage at once
python run_experiments.py --find v20-head-s42 --data data/hsi_v9-trainsel/hsi   # print a run dir
```

A sweep file is Python. It defines `STAGES` and, optionally, `GROUPS`, and builds jobs with
`train(tag, data, **flags)` and `cmd(name, argv, done_if=...)`:

```python
from run_experiments import train
REF = dict(profile="paper_recipe", batch_size=256, lambda_sam=0.1,
           train_subsample_frac=0.102, val_subsample_frac=0.0986)
STAGES = {"depth": [train(f"depth-n{t}", HSI, **REF, epochs=12, seed=42, trm_n_improve=t)
                    for t in (1, 2, 4)]}
```

- **Resumable.** A train job is done when a run with its `run_tag` and data dir has finished
  (`test_predictions.npz` exists). A cmd job is done when its `done_if` file exists. After an
  interruption, run the same command again.
- **A failure does not stop the sweep.** The failed job is reported, the sweep moves on, and
  the exit code is 1.
- **One process per job.** Logs go to `logs/<sweep>/<job>.log`.
- **`--jobs N` runs jobs of the same stage in parallel.** Stages still run in order.
  Concurrent runs share the GPU, so their wall clock and ms/step are not comparable to a solo
  run's.

| sweep | contents |
|---|---|
| [`sweeps/paper_runs.py`](../sweeps/paper_runs.py) (was `v18.py`) | every run behind the paper's numbers, from doc 15 and the v18 short-run queues. All of them exist, so a normal invocation skips them |
| [`sweeps/paper_evidence_upgrade.py`](../sweeps/paper_evidence_upgrade.py) (was `v20.py`) | the jobs of `plan/manuscript_v18_evidence_upgrade.plan.md`: `scripts/v20_queue.sh`, stage for stage, with the groups `p1` / `p2` and the same environment overrides |
| [`sweeps/smoke.py`](../sweeps/smoke.py) | the GPU check: one short run per profile, plus old-vs-new v18 parity on the GPU |

The two paper sweeps were renamed on 2026-10-02 for what they hold. Their run tags did not
change (`v20-head-s42` and so on), because a job counts as done when a run with its tag exists.
Logs follow the file name, so new logs go to `logs/paper_evidence_upgrade/`, while the logs of
earlier runs stay in `logs/v20/`.

## How it was verified (CPU only, no GPU)

| test | what it pins |
|---|---|
| `tests/test_train_parity.py` | 304 cases: every old entry point × 4 datasets × several flag sets resolve **every argument**, the **run directory name** and the **stopping knobs** exactly as the archived entry point did |
| `tests/test_train_e2e_parity.py` | each profile trains 2 epochs on a tiny synthetic HSI set through the old entry point and through `train.py`. Per-epoch history, test predictions (bitwise), gates, `cli_args` and the LR schedule must all be identical |
| `tests/test_sweeps.py` | every train job in `sweeps/paper_runs.py` / `sweeps/paper_evidence_upgrade.py`, resolved, equals the `cli_args` recorded by the real run it names: 27 paper runs, 3,537 values |
| `tests/test_profile_names.py` | `_norecon` profiles resolve `recon_mode none`, the others reconstruct; every old name resolves exactly like its new one |

**One deliberate change.** `--safe_mode` now really switches an optimal-family run to the
safe loader, fp32 and eager mode. The old "which flags were typed" set also counted every
profile default, so safe mode left those settings untouched.

**Verified on the GPU on 2026-09-29** (RTX 5060 Ti, `logs/smoke/`):

- `gpu-parity` printed **`IDENTICAL`**. The archived `train_example_v18.py` and `train.py --profile paper_recipe` agree on every per-epoch number and every `cli_args` value.
- Every profile trained on CUDA, with `torch.compile` and bf16 where the profile sets them. Gates G5, G7 (recon profiles) and G8 passed, and the PAD runs produced test reports.
- The old run took ~18 min, the new one ~2 min. The old run went first and paid a cold `--deterministic` compile, including a ~14 min recompile during the G5 re-validation; the new run reused those kernels.

To repeat the check:

```bash
python run_experiments.py sweeps/smoke.py all
```

- **`profiles`** runs one 1-epoch job per profile, on the dataset that profile was built for.
  This exercises the GPU-only paths: `torch.compile`, bf16 on CUDA, the fused fast loop,
  pinned-memory workers and spectral checkpointing.
- **`gpu-parity`** trains the archived `train_example_v18.py` and `train.py --profile paper_recipe` on
  the same HSI subsample, both with `--deterministic`, and compares the two histories. It prints
  `IDENTICAL`, `SAME within GPU nondeterminism` or `DIFFERENT` (exit 1).

The runs are tagged `smoke-*`. Delete them from `experiments/` afterwards.
