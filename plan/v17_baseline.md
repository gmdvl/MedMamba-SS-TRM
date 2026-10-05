# v17 baseline — the numbers every stage is measured against

Companion to `plan/GMedMamba v17 — Throughput, Observability & Simplification.plan.md`.
Written at Stage 0, before any v17 code change.

> **Why this file is in `plan/` and not `experiments/`:** `.gitignore:25` ignores
> `experiments/*`, so a baseline written there would not be committed and the next session
> would not find it. `plan/` is tracked.

Reference commit: `fb5c2a9` (tree clean at Stage 0).
Training machine: RTX 5060 Ti, 15,933 MB, **torch 2.11.0+cu128** (sm_120).
The machine the v17 code changes were authored on has **no GPU** (torch 2.14.0+cpu) and
**no `pytest`** — so every stage's Verify block, and every `pytest` invocation in the plan,
must be run on the training machine. The freeze invariant itself
(`test_frozen_files_untouched.py`) is pure `git` and was checked at every stage with an
inline equivalent; it was green at Stage 0.

---

## 1. Recorded production runs

Both from `experiments/`, both `--architecture recursive --trm_mixer mlp --amp bf16`.

| | PAD `20260906_165349` | HSI `20260906_222725` |
|---|---|---|
| data | `data/pad_optimal` (224×224×3) | `hsi_v8-80_10_10_importance/hsi` (11×11×32) |
| entry point | `train_example_v16_optimal.py` | `train_example_v16.py` |
| batch / token grid | 32 / 28×28 = 784 | 256 / 11×11 = 121 |
| steps per epoch | 51 | 958 |
| **epoch_time (median)** | **43.44 s** (min 42.29, max 48.74) | **679.12 s** (min 654.40, max 705.80) |
| **s/step (derived)** | **0.8517** | **0.7089** |
| peak GPU MB | 1,431.8 (9.0% of card) | 874.2 (5.5% of card) |
| epochs run / wall | 63 / 0.776 h | 12 / 2.261 h |
| epoch-1 `train_loss` | `0.7115191263759562` | `0.6947724704917758` |
| epoch-1 `val_loss` | `0.7090832983575216` | `0.39083625780358155` |
| epoch-1 `train_accuracy` | `0.15928659286592867` | `0.769902409781044` |

### The flags these two runs actually used — read this before claiming any speedup

| flag | PAD | HSI |
|---|---|---|
| `--compile` | **off** | **ON** |
| `--loader_mode` | **balanced** | performance |
| `--eval_artifact_stride` | **1** | 5 |
| `--spectral_checkpointing` | auto (→ off, C=3) | auto (→ **on**, C=32) |
| `--trm_checkpoint_core` | on | on |

**This asymmetry is the single most important fact in this file.** The two baselines are
not the same configuration, and three v17 stages only move one of them:

- **Stage 2** (`loader_mode`, `eval_artifact_stride`) → **PAD only**. HSI already had both.
- **Stage 3** (`--compile on` by default) → **PAD only**. HSI already had it.
- **Stage 5** (per-step syncs, GradScaler) → **both**, and it is the only stage that can
  move HSI at all.

The PAD run went through `train_example_v16_optimal.py` — the "best-metrics" entry point —
and still ran **eager, on the slow loader, writing artifacts every epoch**, because none of
those are in `OPTIMAL_DEFAULTS` and nobody typed them. The HSI runs went through *bare*
`train_example_v16.py` with the good flags typed by hand. That inversion is exactly what
Stages 2 and 3 fix: put the measured-good settings in the defaults so they are not a thing
to remember.

---

## 2. A0 sweep — `scripts/gpu_tune_v16.py`, run 2026-09-13

**What this measures:** `_time_cell.step()` (`scripts/gpu_tune_v16.py:175-187`) is
`opt.zero_grad` → forward (deep supervision) → loss → `backward` → `opt.step()`, and
**nothing else**. No gradient-health inspection, no GradScaler, no EMA, no DataLoader,
no augmentation, no validation, no artifact writing. It also **does not** call
`enable_spectral_gradient_checkpointing`, which the real training path turns on at
`in_channels >= 16` — so the HSI rows below describe a configuration no real run uses.

### PAD — 224×224×3, grid 28, bf16, mixer=mlp

| batch | ckpt | compile | ms/step | peak MB | % card | samples/s |
|---|---|---|---|---|---|---|
| 32 | on | off | 437.0 | 1,428 | 9.0% | 73.2 |
| 64 | on | off | 1040.6 | 2,835 | 17.8% | 61.5 |
| 128 | on | off | 2210.0 | 5,682 | 35.7% | 57.9 |
| 32 | **off** | off | 384.7 | **6,863** | 43.1% | 83.2 |
| 64 | off | off | OOM | — | — | — |
| 128 | off | off | OOM | — | — | — |
| 32 | on | **on** | **266.7** | **1,110** | 7.0% | **120.0** |
| 64 | on | on | 602.4 | 2,200 | 13.8% | 106.2 |
| 128 | on | on | 1309.1 | 4,402 | 27.6% | 97.8 |
| 32 | off | on | 236.1 | 5,669 | 35.6% | 135.5 |
| 64 | off | on | 523.0 | 11,381 | 71.4% | 122.4 |
| 128 | off | on | OOM | — | — | — |

### HSI — 11×11×32, grid 11, bf16, mixer=mlp (no spectral checkpointing — see above)

| batch | ckpt | compile | ms/step | peak MB | % card | samples/s |
|---|---|---|---|---|---|---|
| 256 | on | off | 1040.3 | 10,157 | 63.7% | 246.1 |
| 512 | on | off | OOM | — | — | — |
| 256 | on | **on** | **474.2** | 7,763 | 48.7% | **539.9** |
| 512 | on | on | OOM | — | — | — |
| 256 | off | * | OOM | — | — | — |
| 512 | off | * | OOM | — | — | — |

Inductor note on this card: `Not enough SMs to use max_autotune_gemm mode` — so
`--compile_mode max-autotune` partially falls back. Default mode is what was measured and
what the defaults use.

---

## 3. What the sweep settled

1. **`torch.compile` is the win.** 437 → 266.7 ms on PAD (**1.64×**) and it *lowers* peak
   memory (1,428 → 1,110 MB), because Inductor fuses away the intermediates of
   `_rms_norm_lastdim`'s six-op chain, run twice per block, 126 blocks per forward.
2. **`--no_trm_checkpoint_core` is REJECTED.** 437 → 384.7 ms is **12%**, bought with
   1,428 → 6,863 MB (**4.8×**), and it OOMs at batch 64 where the checkpointed path runs
   fine. Keep `trm_checkpoint_core=True`. This contradicts the hint in
   `gmedmamba.gmedmamba_trm`'s docstring, which was written for `trm_mixer="ss2d"`.
3. **Raising `--batch_size` is REJECTED on this card.** Throughput *falls*: 120.0 → 106.2 →
   97.8 samples/s at bs 32/64/128 (compile on, ckpt on). Independently confirmed for HSI by
   `training/torch_compile.py`'s own docstring: 89.5 / 80.2 / 78.6 samples/s at 256 / 512 /
   1024. Keep PAD at 32 and HSI at 256.
4. **`--spectral_chunk_size` needs no work.** Already measured in
   `training/torch_compile.py`: 1024 is the optimum; 4096 and 16384 are 1.19× and 1.37×
   *slower* because the chunk stops fitting in cache.

---

## 4. The gap this plan exists to close

| | bench step | real s/step | unexplained |
|---|---|---|---|
| PAD (compile off, ckpt on, bs32) | 0.437 s | 0.8517 s | **0.415 s/step — 49%** |
| HSI (compile on, ckpt on, bs256) | 0.474 s † | 0.7089 s | 0.235 s/step — 33% † |

† the HSI bench row lacks spectral checkpointing, which the real run has, so its gap is not
a clean subtraction — it is an upper bound on how much is *non-step* overhead.

### What `epoch_time` actually covers — measured, not assumed

`TrainerG_v12.fit:834-855` sets `epoch_start_time` and takes the delta **before** the
artifact block, so the recorded `epoch_time` is

> `params_before` + `_train_one_epoch()` + `_validate_one_epoch()` + `params_after`

and **excludes** `_save_eval_artifacts`, `_save_per_class_metrics`,
`_update_loss_components_file`, `_save_checkpoint`, `_update_history_files` (which
regenerates all 10 matplotlib curves) and `_log_epoch`.

Measured on the PAD run from file mtimes (`config.json` → `history.json`):

| | seconds | per epoch |
|---|---|---|
| real wall clock, 63 epochs | 2,841.4 | **45.10 s** |
| `sum(epoch_time)` | 2,792.3 | 44.32 s |
| **outside `epoch_time`** (artifacts, checkpoint, plots, logging) | **49.1** | **0.78 s — 1.7%** |

**Consequence for Stage 2:** the artifact write-down is worth ~0.78 s/epoch, not the ~2 s
the plan guessed. It is still correct to do — 64 confusion matrices and 630 regenerated
curve PNGs for a 63-epoch run is pure waste — but it is a **1.7% item**, and Stage 2's real
content is the DataLoader change. Do not claim more for it than that.

**Consequence for Stage 5:** the unexplained time is *inside* `epoch_time`, so it is not
artifacts. Breaking the 43.4 s down:

| component | seconds | how known |
|---|---|---|
| timed training step (51 × 0.437) | 22.3 | A0 sweep |
| validation (328 samples ÷ 32 = 11 forward-only batches) | ~1.6 | estimate, ~⅓ of a train step |
| `parameters_to_vector` ×2 on 0.45 M params | <0.02 | negligible |
| **unaccounted, inside the training loop** | **~19.4** | **← Stage 5's target** |

**~19.4 s over 51 steps is ~380 ms per step — 45% of the epoch — spent inside
`_train_one_epoch_deep_supervision` but outside forward/backward/`opt.step()`.** Known
occupants, none of them quantified:

- `inspect_gradients` (`trainerg_v12.py:331`) and `check_parameters_finite` (`:368`), every
  batch: **73 parameter tensors × 4 device→host syncs = 292 pipeline drains per step**
  (`training/numerical_stability.py:99-134`);
- `GradScaler`, enabled unconditionally (`training/trainerg_v4.py:133`) including under
  `--amp bf16`, which needs no loss scaling at all;
- the DataLoader, on `balanced` = 2 workers, `prefetch_factor=1`, **`pin_memory=False`**
  (which makes `non_blocking=True` at `trainerg_v12.py:277` a no-op) and
  `persistent_workers=False` (workers re-forked every 51-step epoch);
- the EMA update (`self.ema_helper.update(self.model)`, `trainerg_v12.py:365`), every step;
- 7 `.item()` calls per batch for the loss accumulators (`:388-395`), each a sync;
- `torch.isnan(loss) or torch.isinf(loss)` in validation (`:480`), 2 more syncs per batch.

(Artifacts are *not* in this list — they are outside `epoch_time` entirely and cost 0.78 s,
as measured above.)

**Stage 1 attributes these ~19.4 s. Stage 5 attacks the largest one.** Until Stage 1 lands,
any split of this region is a guess.

### Stage 1 must not change `epoch_time`'s meaning

Every recorded run in `experiments/` uses the definition above. Stage 1 **adds** fields
(`train_time_s`, `val_time_s`, `artifact_time_s`, `data_wait_s`, `s_per_step`) and leaves
`epoch_time` exactly as it is, so old and new runs stay comparable. `train_time_s +
val_time_s` should come to within a few hundredths of `epoch_time`; the remainder is the
two `parameters_to_vector` calls.

---

## 5. Stage 0 reference run — RUN THIS ON THE TRAINING MACHINE

The bit-reproducibility anchor for Stages 1, 2, 4, 5, 6, 7, 8 and 9.

```bash
cd <repo>
git rev-parse --short HEAD        # must be the v17 Stage 0 commit
python train_example_v16_optimal.py --data_dir ./data/pad_optimal --epochs 3 --seed 42 \
    2>&1 | tee /tmp/v17_stage0_reference.log
```

Then record the run directory name and these values here:

```
run dir            : experiments/________________________________
epoch-1 train_loss : ______________________   <- match to 6 dp after every non-Stage-3 stage
epoch-1 val_loss   : ______________________
epoch-1 train_acc  : ______________________
epoch_time e1/e2/e3: ______ / ______ / ______
peak GPU MB        : ______________________
```

`train_example_v16_optimal.py` defaults to `--epochs 200`; `--epochs 3` is an explicit
override and will be listed under "DEVIATIONS" in the banner. That is expected — it is the
only deviation the reference run may show.

---

## 6. Per-stage results — fill in as stages land

| Stage | PAD s/step | PAD epoch | HSI s/step | HSI epoch | epoch-1 loss matches? | Notes |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.8517 | 43.44 s | 0.7089 | 679.12 s | — | recorded runs above |
| 1 timing | | | | | ☐ 6 dp | records the 21.1 s split |
| 2 loader/stride | | | | | ☐ 6 dp | PAD only |
| 3 compile | | | | | ☐ 5 s.f. | PAD only; HSI already on |
| 4 progress | | | | | ☐ 6 dp | |
| 5 syncs/scaler | | | | | ☐ 6 dp | only stage that moves HSI |
| 6 prep | n/a | n/a | n/a | n/a | ☐ medians byte-identical | |
| 7 TRM default | | | | | ☐ 6 dp | |
| 8 simplify | | | | | ☐ 6 dp | |
| 9 entry points | | | | | ☐ 6 dp | + config.json diff clean |

### What was verified WITHOUT a GPU (authoring machine, torch 2.14.0+cpu, no pytest)

Isolated-logic tests, all passing. These are not a substitute for the Runbook — they cover
the new logic, not its integration with a real model or dataset.

| stage | checked | result |
|---|---|---|
| S1 | `_timed_iter` over `continue`, `break` and empty-loader paths | wait time exact (100 ms / 40 ms), no drift |
| S2 | guard placement in `_update_history_files` | CSV/JSON/spectral written unconditionally; only plots gated |
| S4 | `ProgressReporter`, `format_duration` | line format and ETA correct; `s/it` for slow items |
| S5 | fused vs frozen gradient health, healthy / NaN / Inf | `all_finite`, `first_bad_parameter`, `nan_count`, `inf_count` all identical |
| S5b | forced `_FUSED_OK=False` fallback | identical to the frozen per-tensor path |
| S5 | disabled `GradScaler` through the real call sequence | `scale`/`unscale_`/`step`/`update` all work; `state_dict()` is `{}` |
| S5 | `{}` into an enabled scaler | raises — hence the `load_checkpoint` guard |
| S6 | `_capture_medians` under shuffled input order | byte-identical dict; `None`/`<=1e-6` filtering preserved |
| S7 | `apply_v16_defaults` with and without an explicit `--architecture` | default flips; explicit wins, no note emitted |
| S8 | `build_active` vs `cfg.build()` over 6 seeds | 8 steps → 6, bit-identical output |
| S9 | `Overrides.resolved()` empty / partial / legacy-rebound | correct in all three |
| all | frozen-file invariant (pure `git`, no pytest needed) | green at every commit |

Static checks in place of the suite: no test imports an archived module; `test_numerical_stability.py`
imports the frozen helpers directly (S5 did not touch them); `test_run_naming.py` passes
`architecture` explicitly and never reaches `apply_v16_defaults`; `self.current_epoch` is
initialised at `trainerg_v3.py:100`.

**Still unverified and only a GPU can settle it:** every wall-clock number, `--compile on`
with `--recon_mode latent`, the `config.json` diff for S9, and the prep medians against
real ENVI cubes.

**Revised targets** (the plan's original "HSI 679 → 340 s" assumed HSI was eager; it was
not, so that target is withdrawn):

- **PAD 43.4 s → ~25–30 s.** Stage 3 takes the 22.3 s step portion to ~13.6 s; Stages 2 and
  5 attack the 21.1 s remainder. A firm number is not available until Stage 1 reports.
- **HSI 679 s → unknown, set after Stage 1.** Only Stage 5 applies. If the 292 syncs/step
  cost ~0.2 ms each, that is ~56 s/epoch (8%); this is a hypothesis, not a target.
