# GMedMamba v17 — Throughput, Observability & Simplification

## Context

Training and data prep in this repo are slower than the hardware requires, and — more
importantly — **nothing in the repo can say where the time goes.** `history.json` records
exactly one timing field (`epoch_time`); prep prints `N/M done` with no rate and no ETA
(`training/prep/core.py:400` computes a rate and then discards it). An HSI epoch is 958
steps of total silence over 11 minutes.

The A0 sweep (`scripts/gpu_tune_v16.py`, run 2026-09-13) settled the open questions and
overturned two of the three things that looked obvious:

| config (PAD 224×224×3, grid 28) | ms/step | peak MB | samples/s |
|---|---|---|---|
| bs32, ckpt=on, compile=off | 437.0 | 1,428 | 73.2 |
| bs32, ckpt=**off**, compile=off | 384.7 | **6,863** | 83.2 |
| bs32, ckpt=on, **compile=on** | **266.7** | **1,110** | 120.0 |
| bs64, ckpt=on, compile=on | 602.4 | 2,200 | 106.2 |
| bs128, ckpt=on, compile=on | 1309.1 | 4,402 | 97.8 |

| config (HSI 11×11×32, grid 11) | ms/step | peak MB | samples/s |
|---|---|---|---|
| bs256, ckpt=on, compile=off | 1040.3 | 10,157 | 246.1 |
| bs256, ckpt=on, **compile=on** | **474.2** | 7,763 | 539.9 |

Three conclusions:

1. **`torch.compile` is the win: 1.64× (PAD), 2.19× (HSI), and it *lowers* peak memory.**
   It has been available as `--compile on` since v15 and defaults to `off`.
2. **`--no_trm_checkpoint_core` is a bad trade and is hereby rejected**: 12% faster for
   4.8× the memory, and it OOMs at batch 64. Keep core checkpointing on.
3. **Raising `--batch_size` is a *regression* on this card**: 120.0 → 106.2 → 97.8
   samples/s at bs 32/64/128. Keep PAD at 32, HSI at 256.

And one finding that reframes the whole effort:

> `gpu_tune_v16.py:175-187` times **forward + backward + `opt.step()` only** — no health
> checks, no GradScaler, no EMA, no dataloader, no validation, no artifacts. The bench
> says 437 ms/step for the PAD baseline; the real run measured **~850 ms/step**
> (43.3 s ÷ 51 steps, `experiments/20260906_165349_.../history.json`).
> **~48% of a real epoch happens outside the only thing anyone has ever measured.**

`--compile` halves the measured half. The unmeasured half is untouched by it, and the
largest identified suspect there is `TrainerG_v12._train_one_epoch_deep_supervision`
calling `inspect_gradients` (`:331`) and `check_parameters_finite` (`:368`) **every
batch**: 73 parameter tensors × 4 device→host syncs = **292 full pipeline drains per
step** (`training/numerical_stability.py:99-134`).

### Correction found at Stage 0 — read before trusting the paragraph above

The two baseline runs **were not the same configuration**, and the sweep's HSI column
measures a config nobody runs (`plan/v17_baseline.md` §1):

| flag | PAD `20260906_165349` | HSI `20260906_222725` |
|---|---|---|
| `--compile` | **off** | **ON already** |
| `--loader_mode` | **balanced** | performance |
| `--eval_artifact_stride` | **1** | 5 |

So **Stages 2 and 3 move PAD only** — HSI already had all three settings, typed by hand on
a bare `train_example_v16.py` command line. **Stage 5 is the only stage that can move HSI.**
The sweep's HSI 2.19× was measured against an eager baseline that no real run has used.

The PAD run went through `train_example_v16_optimal.py` — the "best-metrics" entry point —
and still ran eager, on the slow loader, writing artifacts every epoch, because none of
those are in `OPTIMAL_DEFAULTS` and nobody typed them. That inversion *is* the bug Stages 2
and 3 fix: put the measured-good settings in the defaults so they stop being something to
remember.

**Intended outcome (revised):** PAD epoch 43.4 s → ~25–30 s; HSI improvement set after
Stage 1 reports (Stage 5 only); every epoch reports where its seconds went and how long the
run has left; prep reports rate and ETA; and the entry-point layer loses its module-global
rebinding chain.

---

## Non-negotiable constraints

1. **The freeze holds.** `test_frozen_files_untouched.py:FROZEN_GLOBS` covers
   `gmedmamba.py`, `training/losses.py`, `augmentation.py`, `plots.py`,
   `numerical_stability.py`, `eval_utils.py`, `run_naming.py`, **all of `training/prep/`**,
   `train_example_v6/v7/v14/v15.py`, and `prepare_*_v3..v7` / `prepare_pad_ufes_20_v4..v6`.
   No edits. Where a frozen file must behave differently, use the pattern this repo
   already established: **subclass, or rebind the module attribute from a non-frozen
   caller** (`prepare_histologyhsi_bc_v8.py:346` rebinds `_v6._find_rgb_image`;
   `training/plots_v16.py` re-exports frozen primitives).
2. **Editable:** `train_example_v16*.py`, `training/trainerg_v12.py`, `trainerg_v13.py`,
   `training/*_v16.py`, `scripts/*`, `prepare_*_v8.py`, `prepare_*_optimal.py`,
   `prepare_*_original.py`, `documentations/*`.
3. **TRM stays the default; every other architecture stays reachable** via
   `--architecture {split,fullchannel,efficient,recursive}`. No preset is removed.
4. **KISS.** Each stage below is the smallest change that achieves its goal. No new
   abstraction layer is introduced where a dict entry, a flag, or one small module will do.
5. **Behaviour-preserving unless the stage says otherwise.** Any stage that changes
   numbers says so in its header and is opt-out via an existing flag.

---

## Resumability

**This file is the canonical record.** Each stage is self-contained and ends in a commit.
To resume with no prior context, in this order:

1. Read the **Status** table below — it is updated at the end of every stage, in the same
   commit as that stage's code.
2. `git log --oneline --grep='^v17 S'` shows which stages have landed, one commit each.
3. `plan/v17_baseline.md` holds the measurements every stage is compared against, plus the
   per-stage results table filled in as stages land. (It is in `plan/`, not `experiments/`,
   because `.gitignore:25` ignores `experiments/*` — a baseline written there would never
   be committed.)
4. Read the stage section below for the first unticked row and execute it. Nothing earlier
   needs to be re-read or re-derived.

**A note for whoever picks this up:** the machine this plan was written on has no GPU, so
every stage's code changes were made without running them. Each stage's **Verify** block
is therefore a real gate, not a formality — run it on the training machine before ticking
the row.

### Status

| Stage | Goal | Changes numbers? | Code | Verified on GPU |
|---|---|---|---|---|
| 0 | Plan in repo + baseline capture | no | ☑ | ☐ reference run |
| 1 | Per-epoch timing breakdown | no | ☑ | ☐ |
| 2 | Loader / artifact-stride defaults + plots on stride (PAD) | no | ☑ | ☐ 6 dp |
| 3 | `--compile on` by default (PAD) | **yes (~1.7e-7)** | ☑ | ☐ 5 s.f. |
| 4 | Training progress + ETA | no | ☑ | ☐ |
| 5 | Per-step sync + GradScaler removal | no | ☑ | ☐ 6 dp |
| 6 | Prep: parallel pass 1 + progress/ETA | no | ☑ | ☐ medians identical |
| 7 | TRM as the bare-v16 default | no | ☑ | ☐ |
| 8 | Augmentation filter + `archive/` | no | ☑ | ☐ full pytest |
| 9 | Entry-point de-monkeypatching | no | ☑ | ☐ config.json diff |
| 10 | Documentation | n/a | ☑ | n/a |

**Code for every stage is committed** (`git log --grep='^v17 S'`). **No stage has been run
on a GPU** — the machine this was authored on has neither CUDA nor `pytest`. The right-hand
column is the remaining work, and each stage's Verify block below is the gate.

---

## Stage 0 — Plan in repo, baseline captured

**No code changes.** Establishes the numbers every later stage is measured against.

Done in the Stage 0 commit:

- This plan copied to `plan/`.
- `plan/v17_baseline.md` written: both A0 sweep tables verbatim, both recorded production
  runs with exact epoch-1 losses, **the per-run flag table that exposed the
  PAD-eager/HSI-compiled asymmetry**, the 21.1 s unexplained-time analysis, and an empty
  per-stage results table to fill in as stages land.
- Two `.gitignore` collisions found and recorded: `experiments/*` (so the baseline lives in
  `plan/`) and `archive/*` (which Stage 8 must negate, or its `git mv` silently untracks
  the archived files instead of moving them).

**Remaining, on the training machine — the one thing Stage 0 cannot finish without a GPU:**

```bash
python train_example_v16_optimal.py --data_dir ./data/pad_optimal --epochs 3 --seed 42 \
    2>&1 | tee /tmp/v17_stage0_reference.log
```
Record epoch-1 `train_loss` to 6 decimals in `plan/v17_baseline.md` §5. **This is the
bit-reproducibility anchor for Stages 1, 2, 4, 5, 6, 7, 8 and 9** (Stage 3 changes it by
~1e-7 and says so).

**Done when:** both plan files are committed and `plan/v17_baseline.md` §5 is filled in.

---

## Stage 1 — Per-epoch timing breakdown

**Why first:** every remaining stage claims a speedup, and right now nothing can confirm
one. This is the instrument, not an optimisation.

**Files:** `training/trainerg_v12.py` (not frozen).

**Changes** — plain floats, no new class:

- In `fit` (`:830-860`), wrap the three existing calls with `time.time()` deltas:
  `train_time_s`, `val_time_s`, and `artifact_time_s` around the block
  `_save_eval_artifacts` … `_update_history_files` (`:901-906`).
- In `_train_one_epoch_deep_supervision` (`:259`), accumulate `data_wait_s` — time spent
  blocked on `for batch in self.train_loader` — and derive `s_per_step =
  train_time_s / n_batches`.
- Add all five to the dict returned by `_build_epoch_metrics` (`:774`) so they land in
  `history.json` / `history.csv` automatically.
- Extend `TrainerG_v11._log_epoch`'s `[v15]` line (`:922`) with one segment:
  `time: train 22.3s (0.437 s/step) | val 1.6s | artifacts 0.9s | data-wait 3.1s`.

**Verify:**
```bash
python train_example_v16_optimal.py --data_dir ./data/pad_optimal --epochs 2
python -c "import json;r=json.load(open('<run>/history.json'))[0];print({k:r[k] for k in ('epoch_time','train_time_s','val_time_s','artifact_time_s','data_wait_s','s_per_step')})"
pytest test_trainerg_v11.py test_run_directory_layout.py test_frozen_files_untouched.py -q
```

**Done when:** the four components sum to within 5% of `epoch_time`, and the residual is
named in the log line. **The residual is the target of Stage 5** — record it in
`baseline.md`.

---

## Stage 2 — Loader and artifact-stride defaults; stop rewriting everything every epoch

**No numerical change. Moves PAD only** — the HSI run already used `performance` and
`stride 5` (`plan/v17_baseline.md` §1). Item 2 below (the write-down) helps both.

**Files:** `train_example_v16_optimal.py` (`OPTIMAL_DEFAULTS`, `:373`),
`training/trainerg_v12.py`.

1. **`OPTIMAL_DEFAULTS` additions** (two dict entries — `build_arg_parser:648` already
   validates the keys exist and `config_deviations:870` already reports overrides):
   - `"loader_mode": "performance"` — PAD ran on `balanced` (2 workers, `prefetch_factor=1`,
     `persistent_workers=False`, **`pin_memory=False`**, `training/dataloader_config.py:44`).
     Without pinned memory, `x.to(device, non_blocking=True)` at `trainerg_v12.py:277` is a
     no-op and every H2D copy is synchronous; without persistence, the workers are re-forked
     every 51-step epoch. `train_val_policies_v16` already forces
     `persistent_workers=False` on the *validation* policy, so the known E-4 hazard does
     not apply. The HSI run already uses `performance`.
   - `"eval_artifact_stride": 5` — matches the HSI run. The PAD run wrote 63 confusion-
     matrix PNGs, one per epoch.
2. **`_update_history_files` (`trainerg_v12.py:976`)** currently rewrites `history.csv`
   and `history.json` in full **and regenerates all 10 matplotlib curves** every epoch
   (measured here: **0.56 s** for the curve set). Simplest fix that keeps every artifact
   the run produces today:
   - append one row to `history.csv` (write the header only when the file is new);
   - write `history.json` and call `generate_all_training_plots` **on the eval stride, and
     unconditionally once from the `finally:` block in `fit` (`:917`)**.
   No new flag: `--eval_artifact_stride` already exists and already means this.

**Verify:** re-run the Stage 0 reference. Epoch-1 `train_loss` must match to 6 decimals.
Final run directory must contain the same 10 PNGs, a complete `history.csv` with one row
per epoch, and confusion matrices on epochs 1, 5, 10, … `pytest test_run_directory_layout.py
test_g6_test_loader.py -q`.

**Expected:** ~1–2 s/epoch on PAD, plus whatever `data_wait_s` (Stage 1) shows the loader
was costing.

---

## Stage 3 — `--compile on` by default

**Changes numbers by ~1.7e-7** (`training/torch_compile.py` docstring: max relative error
vs the eager scan in fp32, which is the path that matters since
`_SelectiveScanParams.run` upcasts before calling the backend).

**Scope, corrected at Stage 0: this moves PAD only.** The HSI production run already passed
`--compile on` by hand (`plan/v17_baseline.md` §1). The sweep's HSI 2.19× is measured
against an eager baseline nobody has run. What this stage actually buys on PAD: the timed
step portion goes `51 × 0.437 = 22.3 s` → `51 × 0.267 = 13.6 s`, i.e. a **43.4 s epoch →
~34.7 s**, before Stages 2 and 5 touch the other 21.1 s. The real deliverable is that the
good setting stops depending on the operator remembering a flag.

**Files:** `train_example_v16_optimal.py` (one dict entry), `train_example_v16_optimal.py`
(`print_banner`), `scripts/gpu_tune_v16.py`.

1. `OPTIMAL_DEFAULTS["compile_model"] = "on"`. That is the whole change — `set_defaults`
   at `:654` applies it, `config_deviations` reports `--compile off` as a deviation, and
   `train_example_v16.py:578` already calls `enable_torch_compile(base_model, mode=args.compile_mode)`
   before the recon wrapper is built.
2. In `print_banner`, state it loudly and state the caveat once:
   ```
   [compile] torch.compile ON (default): measured 1.64x on PAD / 2.19x on HSI, and
             LOWER peak memory. Numerics differ from the eager path by ~1.7e-7.
             Pass --compile off to reproduce a pre-v17 run exactly.
   ```
3. `enable_torch_compile` already swallows failures and returns a summary string
   (`torch_compile.py:146-164`). **Print that summary from `train_example_v16.py`** — today
   the return value is discarded, so a silent fallback to eager would look like "compile
   does nothing" rather than "compile did not run". `gpu_tune_v16.py:148` already does this.
4. **Fix the benchmark while here:** `gpu_tune_v16.py` never calls
   `enable_spectral_gradient_checkpointing`, which the real path enables at
   `in_channels >= 16`. That is why the HSI rows read 10,157 MB against the real run's 874 MB
   and why bs=512 / ckpt=off "OOM". Add `--spectral_checkpointing on/off/auto` mirroring
   `train_example_v16.py:573-576`, then re-run the HSI sweep.

**Verify:**
```bash
pytest test_torch_compile.py test_recursive_features_parity.py test_recon_gradient.py -q
python train_example_v16_optimal.py --data_dir ./data/pad_optimal --epochs 3 --seed 42
python train_example_v16_optimal_recon.py --data_dir ./data/pad_optimal --epochs 2   # recon path is NOT in the sweep
```
Compare against Stage 0: epoch-1 `train_loss` must agree to **≥5 significant figures**
(not bit-identical — that is the point of this stage). `s_per_step` from Stage 1 must drop
to ~0.27 on PAD. Confirm the banner says compiled, not `FAILED`.

**Watch for:** `--compile on` + `--recon_mode latent` (the `train_example_v16_optimal_recon.py`
default) is untested by the sweep; the decoder is not compiled but the core is. Gate G7
(`train_example_v16.py:620`) runs a backward through the decoder before training and will
catch a break early.

---

## Stage 4 — Training progress and ETA

**No numerical change.** This is the "what is happening / how long is left" requirement.

**New file:** `training/progress_v16.py` — one small class, ~50 lines, no dependencies
(`tqdm` is installed but is **not** in `requirements.txt`, and a carriage-return bar
corrupts a redirected log; plain rate-limited lines are the KISS choice).

```python
class ProgressReporter:
    """Rate-limited one-line progress with an ETA. `emit(done, total, **fields)`
    prints at most once every `min_interval` seconds, and always on the last item."""
```

**Call sites (3):**

1. `trainerg_v12.py:_train_one_epoch_deep_supervision` — inside the batch loop:
   ```
     epoch 12/200  step 640/958  67%  loss 0.291 acc 0.874  0.47 s/step  ETA epoch 2m29s
   ```
2. `trainerg_v12.py:_validate_one_epoch_impl` — same, `validating 7/11`.
3. `trainerg_v12.py:fit`, after `_log_epoch` — the run-level line, which is the one that
   answers "how long is left" and "is this about to early-stop":
   ```
     [progress] epoch 63/200 | best f1_macro 0.3142 @ ep 23 | early-stop 40/40 | elapsed 45m | ETA 1h52m
   ```
   Every input already exists: `self.best_metric_value`, `self.best_epoch`,
   `self.early_stop_patience`, and `epoch_time` history.

Also extend `print_gpu_budget` (`train_example_v16_optimal.py:962`) — which prints the
memory denominator but never the clock — with a **time** budget from the measured
`s_per_step` table:
```
[budget] 51 steps/epoch x 200 epochs = 10,200 steps @ ~0.27 s = ~46m (+ validation)
```

**Scope note:** the per-step line goes in the recursive/deep-supervision loop only. The
non-recursive fallback is `TrainerG_v9._train_one_epoch` in a **frozen** file; those
architectures keep today's behaviour and still get the validation and run-level lines.
Say this in the docstring rather than working around it.

**Verify:** run 2 epochs on PAD and 1 on HSI; confirm ETA converges and the lines survive
`> train.log 2>&1` as readable text. `pytest test_trainerg_v11.py -q`.

---

## Stage 5 — Remove the per-step syncs and the dead GradScaler

**No numerical change** (both are dead work, not different work). Targets the residual
Stage 1 exposed.

**Files:** `training/trainerg_v12.py` only. `training/numerical_stability.py` is **frozen** —
do not edit it; add the replacement helper to `trainerg_v12.py` or a new
`training/grad_health_v16.py` and call it instead.

1. **292 syncs → 1.** `inspect_gradients` (`numerical_stability.py:99-117`) does *three*
   device→host syncs per parameter tensor on the healthy path — `bool(finite_mask.all())`,
   `g_abs.max().item()`, `g_abs.min().item()` — and `check_parameters_finite` (`:125-134`)
   one more. 73 tensors × 4 = 292 pipeline drains per step. Its own docstring claims "one
   boolean reduction per parameter tensor"; the implementation does four.

   Replacement, returning the **same `GradientInspection` dataclass** so nothing downstream
   changes:
   ```python
   grads = [p.grad for p in model.parameters() if p.grad is not None]
   norms  = torch._foreach_norm(grads)          # one fused kernel
   stacked = torch.stack(norms)
   all_finite = bool(stacked.isfinite().all())  # ONE sync
   ```
   `max_abs_gradient` / `min_abs_gradient` are diagnostics only (they feed
   `grad_health` logging, never a control decision) — derive them from `stacked` in the
   same reduction. The per-tensor `first_bad_parameter` / `nan_count` / `inf_count` walk
   is kept **verbatim, on the failure branch only**, where a few extra syncs are free and
   the diagnostic is the whole point.
2. **`check_parameters_finite` (`:368`)** is the post-`optimizer.step()` NaN guard. Same
   treatment: one stacked `isfinite().all()`, with the naming walk on failure. Optionally
   add `--health_check_stride N` (default **1**, so behaviour is unchanged unless asked).
3. **GradScaler.** `training/trainerg_v4.py:133` constructs
   `GradScaler(device=self.device_type)` **unconditionally enabled**, and nothing in the
   v9→v12 chain overrides it. So every `--amp bf16` run scales the loss by 65536, unscales
   73 gradient tensors, runs a `found_inf` reduction and syncs on it — on a dtype with
   fp32's exponent range that needs none of it. Under `--amp off` it does the same to fp32.
   `trainerg_v4.py` is frozen; `TrainerG_v12.__init__` (`:182`) is not:
   ```python
   super().__init__(...)
   _, amp_dtype = self._resolve_amp()
   self.scaler = GradScaler(device=self.device_type,
                            enabled=(amp_dtype is torch.float16 and self.device_type == "cuda"))
   ```
   Scaling by a power of two is exact, so existing results are unaffected. Keep
   `scaler_state` in the checkpoint (`trainerg_v11.py:569`) — a disabled scaler still
   serialises, so old checkpoints keep loading.

**Verify:** Stage 0 reference run, epoch-1 `train_loss` **bit-identical**.
`pytest test_numerical_stability.py test_trainerg_v11.py test_failure_taxonomy.py -q`.
Force the failure path once (`--debug_numerics` with an absurd `--lr`) and confirm
`first_bad_parameter` is still named in `numerical_failure/`.
Compare Stage 1's residual before and after — that delta is this stage's result.

---

## Stage 6 — Prep: parallel pass 1, progress and ETA

**No change to stored bytes.**

**Files:** `prepare_histologyhsi_bc_v8.py`, `prepare_pad_ufes_20_optimal.py`,
`training/progress_v16.py` (reused). `training/prep/` stays frozen throughout.

1. **Parallelise `pass1` (`prepare_histologyhsi_bc_v8.py:242-274`).** It walks **all 644
   captures serially in the parent process**, and `_sample_capture_median` (`:80-128`)
   reads pixels **one at a time in a Python loop** (`:119-126`) — 400 random seeks per
   capture, ~257,600 single-pixel ENVI reads — while pass 2 extraction happily uses
   `--num_workers 8` (`training/prep/parallel.py:104`).

   **This is exactly value-preserving:** `_sample_capture_median` is already called with
   `seed=args.seed` — the *same* seed for every capture — so each capture's RNG stream is
   independent of the others and of iteration order. Dispatch the loop over a
   `ProcessPoolExecutor` using the same `get_context("fork")` idiom as
   `training/prep/parallel.py:104`, honouring `--num_workers`. Every median comes back
   bit-identical.
2. **Prep progress with a rate and an ETA.** `training/prep/core.py:400` computes
   ```python
   rate = state_box["n"] / max(1e-6, now - state_box.get("t0", now))
   ```
   and then never uses it. The file is frozen. Use this repo's own established escape:
   from the non-frozen `main()` of `prepare_histologyhsi_bc_v8.py` and
   `prepare_pad_ufes_20_optimal.py`, **rebind `training.prep.core.run_extraction`** to a
   wrapper that decorates the caller's `on_result` with a `ProgressReporter` — the same
   module-attribute-rebinding pattern `prepare_histologyhsi_bc_v8.py:346` already uses for
   `_v6._find_rgb_image`, and for the same reason. ~10 lines, zero frozen-file edits.
   ```
     [extract] 412/644 items  64%  3.1 items/s  ETA 1m15s  (410 ok, 2 skipped)
   ```
3. **Report pass 1 too** (`[capture-gain] 220/644 captures, 12.4/s, ETA 3m25s`) — today it
   prints only after finishing, and only if captures were clipped (`:271`).
4. **Delete the hand-rolled argv scan** in `prepare_pad_ufes_20_optimal.py:268-279`, which
   re-parses `--tiling` and `--out_dir` out of `sys.argv` because `run_prep` returns only a
   status code, and mishandles `--` and abbreviations. `preprocess_args` (`:179`) already
   receives the parsed namespace and the class already keeps `self._args` — stash it there
   and read it in `main`. Removes 12 lines and a class of bugs.

**Verify:**
```bash
python prepare_histologyhsi_bc_v8.py --root <...> --out_dir /tmp/hsi_v17_check --limit 20 --num_workers 8
```
then diff `capture_gain_report.json`'s `per_capture` medians against the same command at
`--num_workers 1`: **they must be identical**. `pytest test_prep_hsi_fixture.py
test_prep_pad_fixture.py test_prep_core.py test_group_sidecars.py
test_frozen_files_untouched.py -q`.

---

## Stage 7 — TRM as the bare-`train_example_v16.py` default

**No documented command changes behaviour.**

Current state — the requirement is already met on three of four entry points:

| entry point | `--architecture` default | source |
|---|---|---|
| `train_example_v16_optimal_recon.py` | recursive | `OPTIMAL_DEFAULTS` (`_optimal.py:423`) |
| `train_example_v16_optimal.py` | recursive | same |
| `train_example_v16_original.py` | recursive | `MEDMAMBA_PROTOCOL_DEFAULTS` (`:233`) |
| `train_example_v16.py` | **split** | inherited from frozen `train_example_v15.py:209` |

Both documented invocations of bare v16 (`documentations/README.md:120,127`) pass
`--architecture recursive` explicitly, so this flips nothing that anyone runs.

**Change:** in `apply_v16_defaults` (`train_example_v16.py:141`), one call to the existing
`_set` helper — which already skips explicitly-passed flags and records a printed note:
```python
_set("architecture", "recursive",
     "v17 - GMedMamba-R (TRM) is the default; --architecture split|fullchannel|efficient "
     "still selects the hierarchical backbones")
```
`_main` calls `apply_v16_defaults` (`:338`) before `build_run_name` (`:368`), so run naming
picks it up. `build_model`'s `_PRESETS` table (`training/config_presets.py:112`) needs no
change — all four architectures stay reachable and untouched.

**Verify:** `python train_example_v16.py --data_dir ./data/pad_optimal --epochs 1` builds
a recursive model and prints the `[v16 default]` note; the same with
`--architecture split` still builds `split`. `pytest test_run_naming.py
test_v15_model_changes.py -q`.

---

## Stage 8 — Simplification and `archive/`

**No numerical change.**

1. **Skip the no-op augmentation steps.** `AugmentationConfig.build()`
   (frozen, `training/augmentation.py:179-191`) always returns all 8 steps; each checks
   "am I off" inside itself on every sample. With `OPTIMAL_DEFAULTS`, `band_dropout_prob`
   and `spectral_mask_prob` are 0 — two dead Python calls per sample forever. Filter the
   returned `Compose.steps` at the **call site** (`train_example_v16.py:508`, not frozen)
   by probing each step against a dummy patch, or simply by building the list from the
   same flags. Keep it to a few lines.
2. **`archive/`.** These are imported by nothing reachable from any entry point (verified
   by AST reachability from `train_example_v16_optimal_recon.py`):
   `train_example_v8/v9/v10/v11/v12/v13.py`, `prepare_pad_ufes20.py`, `gmedmamba_io.py`,
   `training/checkpoint.py`, and `model/` (three more full copies of `gmedmamba`). ~300 KB.
   `git mv` them to `archive/`, plus `archive/README.md` mapping each to its replacement.

   **TRAP, found at Stage 0 — `.gitignore` line 27 is `archive/*`.** A bare `git mv` into
   `archive/` therefore *untracks* these files rather than moving them: they vanish from
   the repository and survive only in the local working tree and in history. In a
   thesis repo that is backed up by pushing, that is data loss with extra steps. Before
   moving anything, add the negations:
   ```gitignore
   archive/*
   !archive/README.md
   !archive/*.py
   !archive/model/
   !archive/model/*.py
   ```
   then `git status` must show the moved files as renames, not deletions. Verify with
   `git ls-files archive/ | wc -l` after committing.

   **Pre-checks (all already done, re-confirm after the move):**
   - The only real import among them is `training/checkpoint.py:30 → gmedmamba_io`; both
     move together. Every other cross-reference is a **docstring mention**, including in
     frozen files (`training/run_naming.py:10` cites `train_example_v13.py`) — those
     citations still resolve for a reader who follows `archive/README.md`, and the citing
     files cannot be edited anyway.
   - None of the moved files is on `FROZEN_GLOBS`, so
     `test_frozen_list_files_actually_exist` stays green.
   - **Be honest in the commit message: this makes nothing faster.** It is a
     "which file is current" fix.

**Verify:** `pytest -q` (full suite) and one 1-epoch PAD run, both from a clean checkout
after the move.

---

## Stage 9 — Retire the module-global rebinding in the entry points

**No numerical change. Highest risk; deliberately last**, so a regression here cannot be
confused with a regression in Stages 1–8.

**The problem** (`train_example_v16_optimal_recon.py:121-134, 168-192`): composition works
by rebinding module globals in three order-dependent layers —
```python
optimal.install_overrides(args)                 # sets v16.TrainerG_v12 = TrainerG_v12Optimal
v16.build_arg_parser = build_arg_parser         # re-point AGAIN
v16.build_run_name   = build_run_name
v16.TrainerG_v12     = TrainerG_v13Optimal      # re-point AGAIN
recon._v16_build_arg_parser = lambda: optimal.build_arg_parser(argv)   # save/restore dance
optimal.config_deviations = config_deviations   # substituted from inside print_banner
```
Most of that file's docstring exists to explain why this works. The **class** composition
(`TrainerG_v13Optimal(TrainerG_v12Optimal, TrainerG_v13)`, empty body, MRO does the work)
is genuinely good and **stays**. It is the rebinding that goes.

**Change — parameterise, do not redesign:**
```python
# train_example_v16.py
def _main(*, parser=None, trainer_cls=None, run_name_fn=None): ...
def main():                       # unchanged public entry point
    _main()
```
Each derived entry point then passes what it contributes and deletes its
`install_overrides`, its `v16.*` rebinds, and the `recon._v16_build_arg_parser`
save/restore. `config_deviations` becomes a parameter of `print_banner` instead of a
module global substituted at call time.

**Verify — the strictest gate in the plan:**
- `pytest -q` (full suite).
- Re-run the Stage 0 PAD reference through **all three** entry points
  (`_optimal`, `_optimal_recon`, `_original`) and diff `config.json`'s `cli_args`,
  `entry_point`, `trainer` and the run-directory name against a pre-Stage-9 run of the
  same command. All must match exactly.
- Confirm `--stage fit|balance|full` and `--recon_mode none|latent` still select the same
  trainer MRO (`print(type(trainer).__mro__)` under a debug flag, or an added unit test).

---

## Stage 10 — Documentation

Update to reflect the new defaults, the measured numbers, and the new output.

| File | Change |
|---|---|
| `documentations/11_v16_optimal_and_reconstruction.md` | **§4.1 already asks the three questions this plan answered** — paste both A0 sweep tables there. Record `--no_trm_checkpoint_core` and bigger-batch as **measured and rejected**, with the numbers. Add a *time* budget beside the existing GPU budget in §4. |
| `documentations/12_commands_datasets.md` | §0.3 "Disk and time" gets real epoch/run times for both datasets, pre- and post-v17. Every train command in §A.3/§B.3/§C.3/§D reflects the new defaults (compile on, loader `performance`, stride 5) and shows `--compile off` as the reproduce-a-pre-v17-run escape. |
| `documentations/README.md` | Quick-start block; "Version context" gains a v17 row; a short paragraph on the new progress/ETA output and on `history.json`'s new timing fields. |
| `documentations/06_dataset_prep_core.md` | Parallel pass 1, the prep progress/ETA line, and the `run_extraction` rebinding pattern (why, and why not an edit to `training/prep/core.py`). |
| `documentations/01_gmedmamba_model.md` | One note: which code paths in frozen `gmedmamba.py` are unreachable from any v16 entry point (`benchmark_backend`, `gmedmamba_research`, `gmedmamba_sensor_aware`, `_skew`/`_unskew` at the default `scan_directions=4`, the non-classification heads) — so the next reader does not re-derive it and does not try to delete them. |
| `documentations/14_performance_and_progress.md` | **NEW.** The single performance reference: the measurement method, both sweep tables, the "48% of the epoch is outside the timed step" finding and how Stage 1 closed it, the per-stage before/after table, and how to re-measure (`gpu_tune_v16.py` + `history.json`'s timing fields). |
| `archive/README.md` | **NEW** (Stage 8). Each archived file → what replaced it. |
| `plan/GMedMamba v17 — ....plan.md` | Status table fully ticked; final measured results appended. |

---

## Stage 12 — The epoch budget, the last per-step syncs, and the test layout

Three things, landed together because the first is what actually made HSI runs unusable and
the other two are the residual Stage 5 left behind.

**1. `--train_subsample_frac` became a resolved default, not a flag to remember.**
`OPTIMAL_DEFAULTS` hardcoded `train_subsample_frac: 1.0` / `val_subsample_frac: 1.0`, tuned
on PAD-UFES-20 where the full split is 1,626 images. Every good HSI run typed
`--train_subsample_frac 0.1` on the command line instead — that is what `sub01_vsub01` in
the run directory names records — so moving the HSI work onto `train_example_v16_optimal*`
silently reinstated the full split. Measured, from `history.csv`:

| run | fracs | mean epoch |
|---|---|---|
| `20260906_222725_…_sub01_vsub01` | 0.1 / 0.1 | **678 s** |
| `20260913_153150_…` | 1.0 / 1.0 | **12,828 s** |
| `20260914_100028_…_optimal_recon` (SIGKILLed in epoch 2) | 1.0 / 1.0 | **6,557 s** |

`resolve_subsample_frac(n, target)` in `train_example_v16_optimal.py`, next to and shaped
like `resolve_ema_rate`, expresses the budget as a target COUNT (250,000 train / 33,000 val
patches) rather than a fraction, for the reason that function already gives: a fraction is
only right for the corpus it was measured on. PAD sits below both targets and resolves to
1.0, unchanged. HSI resolves to 0.102 / 0.0986.

The coupling that had to come with it: `steps_per_epoch` now takes `subsample_frac`, because
`resolve_ema_rate` divides by it. Feeding it the full-split count for a subsampled run
overstates steps/epoch by `1/frac` and leaves `mu` that much too slow — the exact failure
`resolve_ema_rate` exists to prevent, re-armed by a different route. (On the current HSI
config both clamp to the 0.9995 bound, so today's visible effect is nil.)

**2. 17 syncs → 1**, in `training/trainerg_v12_fast.py` (`TrainerG_v12Fast`) and
`training/ema_v16.py` (`FastEMAHelper`), reached by `--fast_loop on` (default `off` on the
bare entry point, `on` in `OPTIMAL_DEFAULTS` — the `--compile` split). Stage 5 removed 292
syncs and left ~17: 4 in `check_loss_components`, 3 in `inspect_gradients_fused`, 1 for
`float(grad_norm)`, 1 post-step parameter check, 8 `.item()` calls.

The load-bearing details, none of which are approximations:

- `isfinite(total_norm)` **is** `grad_inspect.all_finite and isfinite(grad_norm)` — a NaN or
  Inf propagates through the L2 reduction — so the healthy path never calls
  `inspect_gradients_fused` at all.
- `torch.nn.utils.get_total_norm` + `clip_grads_with_norm_` **are** `clip_grad_norm_` (its
  own docstring says "equivalent to"). Splitting them is what lets the branch happen while
  the gradients are still unclipped, and that matters: `torch.clamp(nan, max=1.0)` is `nan`,
  so clipping against a non-finite norm poisons every gradient and a post-clip diagnostic
  walk would name whatever parameter comes first in iteration order.
- the six component losses accumulate on the device in **float64**, reproducing
  `x.item() * bs` into a Python float operation for operation. `test_fast_loop_v17.py`
  asserts the returned dict and the model weights are **bit-identical** to `TrainerG_v12`,
  not `allclose`.
- Phase 18's post-step parameter check is the one deferral — read one step late inside the
  next probe, drained after the loop. It aborts unconditionally when it fires, so there is
  no skip-and-continue to preserve. Loss and gradient finiteness are **not** deferred:
  that would mean stepping AdamW on a NaN gradient, turning a skippable batch into a dead run.
- fp16 delegates to `TrainerG_v12` (an enabled `GradScaler` syncs on `found_inf` anyway).
- `FastEMAHelper` uses `_foreach_mul_` + `_foreach_add_`, **not** `_foreach_lerp_`: `lerp`
  computes `s + w*(p-s)`, a different expression from the frozen module's `(1-w)*p + w*s`,
  and at `mu=0.999` the ~1000-step memory accumulates the difference. Measured over 2000
  updates, deviation from the frozen expression is 3.1e-05 for `lerp` against 3.6e-07 for
  `mul_`+`add_`, for the same two kernel launches. `best_model.pt` **is** the EMA copy.

`config.json`'s `"trainer"` is now `ov.trainer_cls.__name__` and `trainerg_v13.py` writes
`type(self).__name__` — the provenance defect Stage 9 flagged for "a later stage". Without
it, `--fast_loop` could change the class and nothing in the run would say so.

**3. Tests moved to `tests/`** with `tests/conftest.py` (repo root on `sys.path`) and
`pytest.ini` (`testpaths`, `norecursedirs`). There was no pytest config and no conftest at
all; the suite ran only by accident of pytest's rootdir default, and
`tests/test_group_sidecars.py`'s `from scripts.derive_patch_groups import …` was already
one `--import-mode` flag away from breaking. `pytest` was added to `requirements.txt`.

**Known cost:** frozen `train_example_v15.py:991,1664` prints
`pytest test_representation_sensitivity.py -q` at runtime and cannot be edited. Read it as
`pytest tests/test_representation_sensitivity.py -q`. Recorded in `documentations/README.md`.

### Stage 12b — `torch.compile` scratch off the RAM-backed `/tmp`

Two failures on the first real run of Stage 12, neither in the training code, both worth
recording because `df` and the tracebacks both point away from the cause.

**ENOSPC at gate G7.** `triton/.../make_cubin` → `fsrc.write(src)` →
`OSError: [Errno 28] No space left on device`, with `df /tmp` reporting 2.6 GB free.
`/tmp` here is a **3.1 GB tmpfs**: its `Avail` column is a size cap, not free storage, and
tmpfs pages come from the same pool as anonymous memory. At ~21 GB of 30 GB resident the
write failed while `df` still showed gigabytes — the SIGKILL that started this work, wearing
a different errno. `/tmp/torchinductor_<user>` had reached 164 MB of RAM.
`training/compile_cache_v17.py` now redirects `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` to `<repo>/.cache/compile` when the default is memory-backed, auto-fixing
in the style of `fd_limit_v16.raise_open_file_limit`.

**`AF_UNIX path too long` — a bug that module shipped with.** Its first version also
redirected `TMPDIR`. `multiprocessing`'s resource sharer binds a Unix socket under `TMPDIR`
(`/pymp-XXXXXXXX/listener-XXXXXXXX`, 32 bytes) and `sun_path` holds 108; this repository
path is 63 characters, so `<repo>/.cache/compile/tmp` came to 79 and the address to 111.
**Every DataLoader worker died, and the compiler was fine** — the traceback named
`multiprocessing`, not Inductor. `TMPDIR` is now left alone by default: it is process-wide,
while the thing that actually filled tmpfs is the two persistent caches. `tmpdir=True` opts
in, guarded by `_AF_UNIX_BUDGET`. `tests/test_compile_cache_v17.py` pins both.

**Verify.** `pytest -q` (437 passed, 1 skipped, ~2 min, CPU). Then on the GPU box, the A/B:
two 3-epoch runs at the same seed, `--fast_loop off` then `on`; classification metrics must
be identical, the EMA-derived ones `allclose`. `scripts/gpu_tune_v16.py` gained a
`--spectral_checkpointing` sweep axis so `--no_trm_checkpoint_core` and
`--spectral_checkpointing off` can be decided from one table rather than from docstrings.

---

## Verification summary

**Every stage:**
```bash
pytest test_frozen_files_untouched.py -q        # the invariant
```

**Stages 1, 2, 4–9 — bit-identical:** re-run the Stage 0 reference and match epoch-1
`train_loss` to 6 decimals.
```bash
python train_example_v16_optimal.py --data_dir ./data/pad_optimal --epochs 3 --seed 42
```

**Stage 3 only** — agreement to ≥5 significant figures, not bit-identical.

**Full suite before Stages 8, 9 and 10:**
```bash
pytest -q
```

**End-to-end acceptance:**
```bash
# PAD: 43.44 s/epoch -> target ~25-30 s (Stages 2, 3 and 5 all apply)
python train_example_v16_optimal.py --data_dir ./data/pad_optimal --epochs 5
# HSI: 679.12 s/epoch -> target set after Stage 1 reports; only Stage 5 applies,
#      because this run already had --compile on, performance loader and stride 5
python train_example_v16_optimal.py --data_dir data/hsi_v8-80_10_10_importance/hsi \
    --epochs 2 --batch_size 256 --train_subsample_frac 0.1 --val_subsample_frac 0.1
# prep, parallel pass 1, identical bytes
python prepare_histologyhsi_bc_v8.py --root <...> --out_dir /tmp/hsi_v17_check --limit 20 --num_workers 8
```
Accept when: the PAD epoch target is met and the HSI number is at or below its Stage-1
target; `history.json` carries the five timing fields and they sum to `epoch_time`; a
progress line with an ETA appears in training and in prep; the prep medians are
byte-identical to the serial run; and `pytest -q` is green.

---

## Runbook — everything that still needs a GPU, in order

All stages are committed; none has been executed. Paths below are this machine's real ones
(`data/pad_optimal` = 1,626×224×224×3 float32, 6 classes; `data/hsi_v8-80_10_10_importance/hsi`
= 2,452,086×11×11×32 float16, 3 classes). Each step is independent, so a failure localises
to one stage.

### 0 — the suite and the invariant (2 min)

```bash
cd /data/dante_data/documents/Masters/courses/Thesis/g-medmamba
git log --oneline --grep='^v17 S'          # expect 11 stage commits + 3 follow-ups
pytest -q
pytest test_frozen_files_untouched.py -q
```

### 1 — the anchor: the SAME command, pre-v17 flags (≈3 min)

The Stage 0 bit-reproducibility reference. Record epoch-1 `train_loss` in
`plan/v17_baseline.md` §5.

```bash
python train_example_v16_optimal.py --data_dir ./data/pad_optimal \
    --epochs 3 --seed 42 \
    --compile off --loader_mode balanced --eval_artifact_stride 1 \
    2>&1 | tee /tmp/v17_anchor_prev17.log
```

Expect epoch-1 `train_loss ≈ 0.711519` (the recorded `20260906_165349` value) only if every
other flag matches that run; otherwise this run is its own baseline.

### 2 — the same command with the v17 defaults (≈2 min)

```bash
python train_example_v16_optimal.py --data_dir ./data/pad_optimal \
    --epochs 3 --seed 42 \
    2>&1 | tee /tmp/v17_anchor_v17.log

grep -E '^\[compile\]|^  \[time\]|^  \[progress\]|throughput' /tmp/v17_anchor_v17.log
```

**Pass:** epoch-1 loss agrees with step 1 to **≥5 significant figures**.
**Fail loudly if it matches to 6 decimals** — that means compile did not engage; the
`[compile]` line will say `FAILED` or `unavailable`.

### 3 — read the new instrument (Stage 1's whole point)

```bash
RUN=$(ls -td experiments/*optimal | head -1); echo $RUN
python -c "import json,sys;r=json.load(open(sys.argv[1]))[0];print({k:r.get(k) for k in \
 ('epoch_time','train_time_s','val_time_s','artifact_time_s','data_wait_s','s_per_step')})" \
 $RUN/history.json
```

### 4 — the recon path the A0 sweep never covered (≈3 min)

`--compile on` together with `--recon_mode latent` is untested. Gate G7 runs a backward
through the decoder before training and will fail fast if it is broken.

```bash
python train_example_v16_optimal_recon.py --data_dir ./data/pad_optimal --epochs 2
```

### 5 — Stage 9's gate: all three entry points (≈4 min)

`config.json`'s `cli_args`, `entry_point`, `trainer` and the run-directory name must match
a pre-S9 run of the same command.

```bash
python train_example_v16_original.py --data_dir ./data/pad_optimal --epochs 1
python -c "import json,glob,os;d=sorted(glob.glob('experiments/*'),key=os.path.getmtime)[-1];\
c=json.load(open(d+'/config.json'));print(d);print(c['entry_point'],c['trainer'],c['architecture'])"
```

### 6 — Stage 6's gate: prep medians must not depend on worker count (≈10 min)

`--allow_missing_classes` is required and is not a workaround for a v17 change.
`--limit 20` takes the first 20 captures, which on this corpus are all DCIS, so the
class-coverage check at `training/prep/core.py:334` aborts with
`missing ['healthy', 'IDC']`. That check runs **before** `adapter.pass1` (`:346`), so
without the flag the run dies before the code under test ever executes and no
`capture_gain_report.json` is written. The flag is the escape hatch the error message
itself names, and it is correct here: this is a value-equivalence check between two worker
counts, not a dataset build.

```bash
python prepare_histologyhsi_bc_v8.py \
    --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
    --out_dir /tmp/chk1 --limit 20 --num_workers 1 --allow_missing_classes
python prepare_histologyhsi_bc_v8.py \
    --root /home/dante/Downloads/downloads/HistologyHSI-BC-Recurrence/ \
    --out_dir /tmp/chk8 --limit 20 --num_workers 8 --allow_missing_classes

diff <(python -c "import json;print(json.load(open('/tmp/chk1/hsi/capture_gain_report.json'))['per_capture'])") \
     <(python -c "import json;print(json.load(open('/tmp/chk8/hsi/capture_gain_report.json'))['per_capture'])") \
  && echo "IDENTICAL - parallel pass 1 is value-preserving"
```

### 7 — re-run the HSI sweep (the pre-v17 rows were taken without spectral checkpointing)

```bash
python scripts/gpu_tune_v16.py --input_side 11 --in_channels 32 --num_classes 3 \
    --batch_size 256 --token_grid 11 --compile off,on --spectral_checkpointing auto
```

Expect peak memory near the **874 MB** the production run recorded, not the 10,157 MB the
pre-v17 script reported.

### 8 — the real acceptance runs

```bash
# PAD: baseline 43.44 s/epoch -> target ~25-30 s
python train_example_v16_optimal.py --data_dir ./data/pad_optimal --epochs 5

# HSI: baseline 679.12 s/epoch. Only Stage 5 applies - this run already had
# compile on, the performance loader and stride 5.
python train_example_v16.py --data_dir data/hsi_v8-80_10_10_importance/hsi \
    --architecture recursive --normalization global_zscore --loss weighted_ce \
    --class_weight_power 0.75 --batch_size 256 --lr 1e-4 --epochs 2 --amp bf16 \
    --checkpoint_metric f1_macro --train_subsample_frac 0.1 --val_subsample_frac 0.1 --seed 42
```

### What each result means

| observation | reading |
|---|---|
| step 2 loss matches step 1 to ≥5 s.f. | Stages 1–9 are behaviour-preserving |
| step 2 loss matches to 6 dp | **compile did not engage** — check the `[compile]` line |
| `train_time_s + val_time_s ≈ epoch_time` (±0.02 s) | the accounting closes; `unacc` is the two `parameters_to_vector` calls |
| `s_per_step` ≈ 0.27 on PAD | Stage 3 landed (was 0.437 eager in the sweep) |
| `epoch_time − train − val` large | something runs inside `epoch_time` that Stage 1 does not name — investigate before trusting any speedup |
| `data_wait_s` still large | the loader is the bottleneck, not the model; raise `--num_workers` |
| `[v17 S5] fused ... unavailable` printed | `torch._foreach_norm` missing on this torch; correct but slower, S5's gain is lost |
| PAD epoch 25–30 s | the plan landed. Record in `plan/v17_baseline.md` §6 |

### If something fails

Each stage is one commit — `git revert` the single `v17 S<n>` rather than unwinding the
series. The two with the most surface area:

- **S5** (`grad_health_v16`, GradScaler): if a numerical-instability abort stops naming
  `first_bad_parameter`, the failure branch is not delegating.
- **S9** (`Overrides`): if `config.json` differs, diff `cli_args` first — `entry_point` and
  `trainer` were deliberately left as literals, so any change there is a bug.
