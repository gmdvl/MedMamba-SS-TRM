# 14 — Performance, and knowing what a run is doing

What v17 changed, what it measured, and what it deliberately rejected.

Read `plan/GMedMamba v17 — Throughput, Observability & Simplification.plan.md` for the
stage-by-stage plan and `plan/v17_baseline.md` for the raw numbers.

---

## 1. The two facts that drove v17

**(a) Nothing could say where an epoch went.** A run recorded exactly one timing field,
`epoch_time`, and it does not even cover the whole epoch — `TrainerG_v12.fit` takes its
delta *before* the artifact block. An HSI epoch is 958 optimizer steps over eleven minutes
and printed nothing between one epoch header and the next.

**(b) The only thing ever benchmarked was half the step.** `scripts/gpu_tune.py` times
forward + backward + `opt.step()` and nothing else. On the PAD baseline that is 0.437 s
against a real 0.852 s/step:

| | s/step | share |
|---|---|---|
| timed training step (`gpu_tune.py`) | 0.437 | 51% |
| validation, ~11 forward-only batches | ~0.031 | 4% |
| **inside the training loop, outside the optimiser** | **~0.380** | **45%** |

That 45% was the largest single item in the run and had never been looked at.

---

## 2. Measured on this machine (RTX 5060 Ti, 15,933 MB, torch 2.11.0+cu128)

`scripts/gpu_tune.py`, PAD 224×224×3 at token grid 28, bf16, `mixer=mlp`:

| batch | core ckpt | compile | ms/step | peak MB | samples/s |
|---|---|---|---|---|---|
| 32 | on | off | 437.0 | 1,428 | 73.2 |
| 32 | **off** | off | 384.7 | **6,863** | 83.2 |
| 32 | on | **on** | **266.7** | **1,110** | **120.0** |
| 64 | on | on | 602.4 | 2,200 | 106.2 |
| 128 | on | on | 1309.1 | 4,402 | 97.8 |

### What this settled

**`torch.compile` is the win — 1.64×, and it *lowers* peak memory.** Inductor fuses the
six-op `_rms_norm_lastdim` chain that runs twice per block, 126 blocks per forward. It is
now `OPTIMAL_DEFAULTS["compile_model"] = "on"` (v17 S3).

**`--no_trm_checkpoint_core` is REJECTED.** 12% faster for 4.8× the memory, and it OOMs at
batch 64 where the checkpointed path is fine. `medmamba_ss_trm.medmamba_ss_trm_default`'s docstring hints
otherwise; that hint was written for `trm_mixer="ss2d"` and does not transfer to the
`mlp` mixer every real run uses.

**Raising `--batch_size` is REJECTED on this card.** Throughput *falls*: 120.0 → 106.2 →
97.8 samples/s at bs 32/64/128. `training/torch_compile.py` independently measured the
same on HSI: 89.5 / 80.2 / 78.6 samples/s at 256 / 512 / 1024. Keep PAD at 32, HSI at 256.

**`--spectral_chunk_size` needs no work.** Already measured: 1024 is the optimum; 4096 and
16384 are 1.19× and 1.37× *slower* because the chunk stops fitting in cache.

> **Caveat on the HSI rows of any sweep run before v17.** `gpu_tune.py` never called
> `enable_spectral_gradient_checkpointing`, which the real path enables at
> `in_channels >= 16`. That is why its 11×11×32 rows read 10,157 MB against the 874 MB the
> production run recorded, and why batch 512 "OOM"d. Fixed in v17 S3 —
> `--spectral_checkpointing on|off|auto`. Re-run any HSI sweep taken before that.

---

## 3. What a run prints now

```
  [train] epoch 12 640/958  67%  loss 0.2910 acc 0.8740  0.438 s/it  ETA 2m19s
  [val]   epoch 12 11/11 100%  7/s  ETA 0s
  [time] train 22.31s (0.437 s/step x 51) | val 1.62s | data-wait 3.05s | artifacts 0.78s | unacc +0.02s
  [progress] epoch 63/200 | best f1_macro 0.3142 @ ep 23 | early-stop 40/40 | elapsed 45m | ETA 1h52m
```

- **In a terminal these two `[train]`/`[val]` lines redraw in place**; redirected to a file
  they are whole lines every 2 s. `ProgressReporter` picks by `stream.isatty()`, because a
  carriage-return redraw written through `| tee` is one unreadable mega-line, and every run
  here is launched with `tee`. Lines are padded to the widest drawn so a shortening line
  (`99%` → `100%`) cannot strand characters, and `close()` releases the line when a loop
  `break`s (a numerical-stability abort) so the epoch summary does not print on top of it.
- `[time]` — `unacc` is the closure check (`epoch_time − train − val`, i.e. the two
  `parameters_to_vector` calls). More than a few hundredths of a second there means
  something is running inside `epoch_time` that the line does not name.
- `[progress]` — the ETA takes the **minimum** of epochs-remaining and
  patience-remaining, because whichever limit lands first is the one worth an ETA.
- Five new fields in every `history.json` row: `train_time_s`, `val_time_s`,
  `artifact_time_s`, `data_wait_s`, `s_per_step`. **`epoch_time` keeps its exact pre-v17
  meaning** (`params_before + train + val + params_after`) so runs already in
  `experiments/` stay comparable.
- `data_wait_s` is time blocked in `next()` on the train loader — genuine input
  starvation. It is `None`, not `0.0`, for non-recursive architectures, whose training
  loop is in a frozen file and cannot report it.

Data prep prints a rate and an ETA too:

```
  [capture-gain] 412/644  64%  388 usable  12.40/s  ETA 18s
  [extract] 412/644  64%  3.10/s  ETA 1m15s
```

---

## 4. Defaults that changed in v17

Applied through `OPTIMAL_DEFAULTS`, so they show in the banner and any override is listed
under DEVIATIONS.

| default | was | now | why |
|---|---|---|---|
| `--compile` | `off` | **`on`** | 1.64× on PAD, lower memory. ~1.7e-7 numeric drift. |
| `--loader_mode` | `balanced` | **`performance`** | `balanced` is 2 workers, `prefetch 1`, `pin_memory=False` (making `non_blocking=True` a no-op) and re-forks workers every epoch. |
| `--eval_artifact_stride` | `1` | **`5`** | `1` wrote 64 confusion matrices for a 63-epoch run. |
| `--architecture` (bare `train_example_v16.py` only) | `split` | **`recursive`** | TRM was already the default on the other three entry points. |

All four were *already* what the HSI runs used — typed by hand on the command line. The
PAD runs went through `train_example_v16_optimal.py`, the "best-metrics" entry point, and
still ran eager on the slow loader, because none of it was in the defaults. **That
inversion was the actual defect.**

**To reproduce a pre-v17 run exactly:** `--compile off --loader_mode balanced
--eval_artifact_stride 1`. The banner states which path you are on, in both directions.

---

## 5. Where the 45% went

- **292 device syncs per step → 1.** `inspect_gradients` ran three device-to-host syncs
  *per parameter tensor* on the healthy path — `isfinite(g).all()`, `g.abs().max().item()`,
  `g.abs().min().item()` — and `check_parameters_finite` added a fourth. With 73 parameter
  tensors that is 292 full pipeline drains every step. `training/fused_stability_checks.py`
  replaces both with `torch._foreach_norm` + one stacked reduction: **one** sync.
  `isfinite(norm)` is False exactly when the tensor holds a NaN or Inf, since both
  propagate through the L2 reduction, so this is the same check, not an approximation. The
  per-tensor walk is kept verbatim on the failure branch, where the diagnostic
  (`first_bad_parameter`, `nan_count`, `inf_count`) is the entire point and syncs are free.
  *One deliberate change:* `max_abs_gradient` / `min_abs_gradient` are now per-tensor
  **norms** rather than absolute elements. Both are logged diagnostics; nothing branches
  on them. *And it cannot take a run down with it:* `torch._foreach_norm` is a private API
  and `torch.stack` refuses a mixed-dtype list, either of which would raise at the **first**
  optimizer step. `_fused_norms` probes for the symbol once, catches anything the fused call
  throws, warns once and latches back to the frozen per-tensor implementation for the rest
  of the run — correct, just slower, i.e. exactly the pre-v17 behaviour. Same policy as
  `training/torch_compile.py` and `training/prep_progress.py`: an optimisation must
  never be what kills a run.
- **The GradScaler was on under bf16.** `training/trainerg_v4.py:133` builds
  `GradScaler(device=...)` with no `enabled=`, and nothing in the v9…v11 chain overrode
  it — so every `--amp bf16` run scaled the loss by 65536, unscaled 73 gradient tensors,
  ran a `found_inf` reduction and synced on it, on a dtype with fp32's exponent range.
  Now enabled for fp16 only. Scaling by a power of two is exact, so results are unchanged.
  *Resume note:* a disabled scaler serialises to `{}`, and `{}` into an enabled scaler
  raises. `TrainerG_v12.load_checkpoint` skips exactly that case, so `--amp bf16` → resume
  `--amp fp16` still works and a genuine fp16 → fp16 resume still restores its scale.
- **Plots were regenerated every epoch.** 630 figure renders for a 63-epoch run, of which
  10 survive. Now on the eval stride, with a forced final flush. Measured: 0.78 s/epoch
  for the whole artifact block, **1.7% of wall clock** — worth fixing, not worth
  overselling.
- **Prep pass 1 was serial.** `prepare_histologyhsi_bc_v8.pass1` walked all 644 captures
  in the parent process at ~400 single-pixel ENVI reads each, while pass 2 had used a
  process pool since v6. Now parallel over `--num_workers`. Exactly value-preserving:
  `_sample_capture_median` is called with the *same* seed for every capture, so each
  capture's RNG stream is already independent of order.

---

## 6. Gate G6 no longer forfeits a run to a file-descriptor ceiling (v17 S11)

A finished HSI run died at the very last step with `RuntimeError: Too many open files`, and
lost its held-out test number after training and validating correctly for two hours. The
test split is the **largest loader in the run** — train and val were subsampled to 10 %, the
test split is evaluated whole at 348,894 patches — and it is built **last**, after the other
loaders have already consumed descriptors. PyTorch's default `file_descriptor` sharing
strategy passes one FD per shared tensor, so the ceiling is `RLIMIT_NOFILE`, whose soft
value is commonly 1024 against a hard value of 1,048,576.

Two changes, and deliberately **not** `set_sharing_strategy("file_system")` — that trades
the FD ceiling for shared-memory segments that leak on a hard kill, a bad trade in a repo
whose v12/v13 plans are largely about `/dev/shm` exhaustion and worker SIGBUS:

* `training/fd_limit.py` — `raise_open_file_limit()` lifts `RLIMIT_NOFILE` soft to hard
  at startup, called from `train_example_v16._main` before any loader exists. Plain headroom
  the kernel has already granted the user; nothing in a run ever needed a 1024-descriptor
  budget, it was simply never raised.
* The G6 handler retries **once, single-process**, when `is_fd_exhaustion(e)` matches (a
  bare `RuntimeError` with that message, or `OSError`/`EMFILE`). Same weights, same data,
  same evaluation — only the loading differs — so the resulting G6 is a real number rather
  than a workaround artifact. The evaluation is factored into `_run_test_eval` so the retry
  cannot drift into a subtly different code path.

What you see:

```
WARNING: test-split evaluation ran out of file descriptors with 4 worker(s). Retrying single-process.
[test-eval] single-process retry succeeded.
```

If the retry also fails, `gates.json` records `"G6": false` and the **retry's** exception
plus traceback are written to `<run>/gate_g6_failure.json` — added because a bare
`AssertionError` prints as an empty message, which is how a frozen `NpyDataset`
normalization assert silently cost an entire earlier run its G6.

---

## 7. How to re-measure

```bash
# the kernel-level cost of a configuration, no dataset needed
python scripts/gpu_tune.py --input_side 224 --in_channels 3 \
    --batch_size 32 --token_grid 28 --compile off,on
python scripts/gpu_tune.py --input_side 11 --in_channels 32 \
    --batch_size 256 --token_grid 11 --compile off,on --spectral_checkpointing auto

# the whole-epoch cost, from a real run
python -c "import json,sys;r=json.load(open(sys.argv[1]))[0];print({k:r.get(k) for k in \
 ('epoch_time','train_time_s','val_time_s','artifact_time_s','data_wait_s','s_per_step')})" \
 experiments/<run>/history.json
```

Remember what each one covers: the sweep is the optimiser alone; `history.json` is the
epoch. Before v17 only the first existed, which is how a 45% item stayed invisible.
