# GMedMamba v15 — Representation Collapse Remediation Plan

**Date:** 2026-09-02
**Supersedes for the recursive path:** nothing — this is additive to the v12/v13 stability plans, which
remain correct and in force.
**Triggering evidence:**
`experiments/20260902_111050_hsi_v7-80_10_10-hsi_recursive_hsi_mlp_sup3_bs256_bf16` (HSI, 6 epochs, 2.2 h)
and `experiments/20260901_195001_pad_v6_recursive_rgb_mlp_sup3_bs256_bf16` (RGB, 33 epochs, 17.5 h).
**Audit:** 38 findings, restated inline below under their audit IDs (`A1`…`E6`).

---

## 0. The defect in one paragraph

`gmedmamba.SpectralTokenizer` builds every band token as `value_embed(v) + positional_encoding`.
`value_embed` is a **shared `nn.Linear(1, d_token)`** initialised by `GMedMambaBackbone._init_weights`
at `trunc_normal_(std=0.02)` with a **zeroed bias**, so on inputs in `[0, 1]` its output has magnitude
≈ 0.006. The positional encoding — **identical for every sample in the dataset** — has magnitude ≈ 0.55.
The input signal is therefore **86–93× smaller than the constant it is added to**, and after
`tokens.mean(dim=1)`, the GELU compressor and `stem_norm` (LayerNorm, which divides out whatever scale
survives), the embedding entering the recursive core varies with the input by **0.18 % (HSI) / 0.29 %
(RGB)**. Both runs consequently begin life as *constant functions*; the RGB run's 105,336 optimizer
steps dragged it partly out (macro-F1 0.09 → 0.24), the HSI run's 2,868 steps did not
(`predictions_epoch01.npz`: max-softmax confidence spans **0.00022** across all 334,516 validation
patches; every sample predicted `DCIS`). Everything else in the audit is either a consequence of this,
an amplifier of it, or a reason it was not caught for 19.7 GPU-hours.

---

## 1. Compatibility contract

Every change in this plan obeys these five rules. They exist so that the two completed runs stay
reproducible and citable in the thesis, and so no fix can silently rewrite the meaning of an older run.

| # | Rule |
|---|------|
| **C-1** | **New entry point `train_example_v15.py`.** `train_example_v14.py` is not modified. This is the repository's existing convention (v13's own docstring: *"implemented as a NEW versioned entry point … rather than editing train_example_v12.py in place, so anyone already depending on v12's exact behaviour keeps it unmodified"*), and it matters more than usual here because v15 flips defaults that change results. |
| **C-2** | **New trainer `training/trainerg_v11.py`,** subclassing `TrainerG_v10`. Reporting-integrity fixes (R5) live there. `TrainerG_v10` is untouched. |
| **C-3** | **`gmedmamba.py` and shared `training/*` modules are edited in place, but every behavioural change sits behind a `GMedMambaConfig` field or a keyword argument whose default reproduces today's behaviour exactly.** v14 constructs the same model it does now because it never sets the new fields. v15 sets them. |
| **C-4** | **No checkpoint-format break.** New parameters are additive; `load_checkpoint` must document and tolerate a `strict=False` load when the recorded `config.json` shows a different `spectral_token_fusion`. A checkpoint trained under one tokenizer mode is never silently loaded into another — the mode is compared and a mismatch raises. |
| **C-5** | **Every item lands with a test that fails on the current tree and passes after.** Pure-warning changes (R0.4) may instead land with an assertion on the emitted warning list. No item is "done" on inspection alone. |

Consequence of C-3 worth stating explicitly: **v14 remains a valid entry point for reproducing the two
runs in this plan's evidence section.** Do not delete it, and do not "fix" it.

---

## 2. Acceptance criteria

### 2.1 Shallow baselines — the bar the model must clear

Measured on this working tree with `sklearn.linear_model.LogisticRegression(class_weight='balanced')`,
global per-channel z-score fit on train only, patient-disjoint splits, 25,000 train / 15,000 val patches.
These are *cheap* models. A 442,215-parameter spatial-spectral network that does not beat them has not
demonstrated anything.

| Dataset | Feature set | dim | Accuracy | **Balanced acc.** | **F1 macro** |
|---|---|--:|--:|--:|--:|
| HSI `hsi_v7-80_10_10` (3 cls) | majority class | — | 0.6329 | 0.3333 | — |
| | patch-mean spectrum | 32 | 0.5179 | 0.4863 | 0.4532 |
| | **patch mean + std per band** | 64 | **0.6590** | **0.6007** | **0.5539** |
| | full flattened patch | 3872 | 0.5811 | 0.4427 | 0.4379 |
| | **G-MedMamba recursive, 6 ep (current)** | — | **0.0734** | **0.3333** | **0.0456** |
| RGB `pad_v6` (6 cls) | majority class | — | 0.3612 | 0.1667 | — |
| | patch-mean colour | 3 | 0.2284 | 0.2605 | 0.1918 |
| | **patch mean + std per channel** | 6 | 0.2689 | **0.2997** | 0.2319 |
| | full flattened patch † | 363 | 0.2175 | 0.2397 | 0.1914 |
| | **G-MedMamba recursive, 33 ep (current)** | — | **0.4202** | **0.2571** | **0.2419** |

† lbfgs hit its 1500-iteration limit — treat as a lower bound.

Read these carefully; they set two different bars.

- **HSI: the model is far below the floor.** Balanced accuracy 0.3333 against 0.6007 from **64 numbers**
  (per-band mean and standard deviation of the patch). The +0.114 jump from mean-alone to mean+std says
  a large part of the signal is *within-patch spectral variance* — exactly the spatial-spectral
  structure the model exists to exploit and currently cannot see at all. That the 3872-dim flattened
  probe does *worse* (0.4427) is ordinary overfitting at 25 k samples, and tells you the useful HSI
  signal is low-dimensional summary statistics rather than raw 11×11 pixel layout.
- **RGB: the model is at parity with six numbers.** F1-macro 0.2419 vs 0.2319 and balanced accuracy
  0.2571 vs 0.2997 — it wins one, loses the other, after 442,602 parameters and 17.5 hours. (The probes
  use `class_weight='balanced'`, which trades raw accuracy for balanced accuracy, so the accuracy column
  is not directly comparable; judge on F1-macro and balanced accuracy.) This is consistent with `A1`:
  once the tokenizer collapses a patch to roughly its mean colour, mean colour is what the network can
  learn from.

> **R0.0 — regenerate this table as a committed script before anything else.**
> `scripts/shallow_baseline.py` (new, ~40 lines, no GPU, ~4 min for both datasets). Re-run it on every
> re-prepped dataset and record the output in the run directory. It is the cheapest instrument in this
> plan and it is what turns "the model is bad" into "the model is bad *relative to a known ceiling*" —
> which is the difference between a debugging note and a thesis result.

### 2.2 Gates

Each gate is a hard pass/fail. Do not advance a stage until its gate passes.

| Gate | Definition | Current value | Target |
|---|---|--:|--:|
| **G1** *init sensitivity* | On 64 real patches through a freshly built model: `S = mean_j(std_i(e_ij)) / mean_ij(|e_ij|)` over the stem embedding `e` | 0.0018 (HSI) / 0.0029 (RGB) | **≥ 0.05** |
| **G2** *logit sensitivity* | Same batch: `logits.std(dim=0).mean()` | ~1e-3 *(inferred from the 2.2e-4 confidence spread over 334k val samples)* | **≥ 1e-2** |
| **G3** *epoch-1 non-degeneracy* | Number of distinct predicted classes on the validation split after epoch 1 | 1 of 3 (HSI), 1 of 6 (RGB) | **≥ 2**, and rising |
| **G4** *beats the shallow ceiling* | Validation balanced accuracy and F1-macro vs. §2.1's best shallow probe | HSI 0.3333 / 0.0456 vs **0.6007 / 0.5539**; RGB 0.2571 / 0.2419 vs **0.2997 / 0.2319** | **> shallow best on both** |
| **G5** *reported = reproducible* | Reloading `best_model.pt` and re-evaluating the validation split reproduces the logged best metric to 1e-6 | fails (EMA/live mismatch, `D1`) | **passes** |
| **G6** *test split reported once* | `test_report.json` exists, produced by exactly one pass over `X_test.npy` with the best checkpoint | absent | **present** |

---

## 3. Stage R0 — Guardrails

**Goal:** make the failure impossible to miss, before changing anything that could hide it.
**Nothing in R0 changes training behaviour.** Land it first so every later stage has an instrument.

### R0.1 — Representation-sensitivity test `A1` `new: test_representation_sensitivity.py`
- **Change.** New root-level pytest module, matching the existing `test_*.py` convention. Builds each
  `(architecture, modality)` preset via `training.config_presets.build_model`, feeds 64 patches
  (real ones when the dataset is present, otherwise seeded random ones in `[0,1]`), and asserts **G1**
  and **G2**. A second test asserts two *disjoint* batches produce different mean logits.
- **Why.** This is the whole audit reduced to one assertion. It fails on the current tree — that is the
  point, and it is how R1 is verified.
- **Verify.** `pytest test_representation_sensitivity.py -q` → fails now (record the numbers in the
  test's docstring), passes after R1.2.

### R0.2 — Input-sensitivity stage in the pre-training smoke test `A1` `training/numerical_stability.py`, `train_example_v15.py`
- **Change.** Add `run_sensitivity_check(model, x, min_logit_std=1e-2) -> dict` to
  `training/numerical_stability.py` — a **new function**, so `run_smoke_test`'s contract is untouched
  (C-3). Call it from `run_numerical_smoke_test` in v15 alongside the existing finiteness cycle. A
  failure raises `SystemExit("REPRESENTATION_COLLAPSE: …")` and writes
  `<exp_dir>/numerical_failure/sensitivity_failure.json`.
- **Change.** In v15, default `--numerical_smoke_test` to `on` (v14 defaults to `auto`, which is off
  unless `--safe_mode`/`--debug_numerics`). The check costs one forward pass.
- **Verify.** Unit test with a deliberately constant `nn.Module`; integration check that v15 aborts in
  under a minute on the pre-R1 model.

### R0.3 — Class collapse becomes a stop condition `B6` `training/trainerg_v11.py`, `training/failure_taxonomy.py`
- **Change.** Add `REPRESENTATION_COLLAPSE = "REPRESENTATION_COLLAPSE"` to `FailureClass` (purely
  additive, per that module's own design note). In `TrainerG_v11`, track a streak of epochs in which the
  predicted-label histogram has exactly one non-zero bin; on reaching `--class_collapse_streak`, set
  `stop_reason = _StopReasonStr("TRAINING_ABORTED_REPRESENTATION_COLLAPSE")`, write
  `<exp_dir>/collapse_failure/failure.json`, and stop.
- **Change.** New flag `--on_class_collapse {warn,abort}`, default **`abort`** in v15. v14's
  warn-only behaviour is preserved because v14 does not have the flag and `TrainerG_v10` is untouched.
- **Why.** `TrainerG_v5._check_class_collapse` currently produces warning strings and nothing else;
  `--class_collapse_streak 3` reads like a control but only sets the warning threshold. The HSI run
  spent 2.2 h re-confirming a state that `predictions_epoch01.npz` had already settled.
- **Verify.** Unit test driving `_check_class_collapse` with a synthetic single-class prediction history.

### R0.4 — Repair the metric validators `B1` `B2` `B3` `B4` `B5` `training/eval_utils.py`, `training/metric_validation.py`
Warnings only — no metric value changes, so this is safe to apply to the shared modules (C-3 exemption:
nothing behavioural is gated).

| Sub | Change |
|---|---|
| a | **Delete** the `2·p·r/(p+r)` harmonic-mean check from **both** `eval_utils.sanity_check_metrics` (`:139-145`) and `metric_validation.MetricValidator` (`:95-105`). Macro-F1 is the mean of per-class F1s and is never that harmonic mean; it fired on **32 of 33** RGB epochs. |
| b | **Replace** it with the check that is actually valid: `abs(f1_macro − mean(per_class_f1)) > 1e-6`, plus the range check `0 ≤ f1_macro ≤ 1` (keep). |
| c | **Delete** the `balanced_accuracy > accuracy` rule (`metric_validation.py:110-124`). Balanced accuracy is the unweighted mean of per-class recall and legitimately exceeds accuracy whenever a minority class is over-predicted — precisely the HSI case. The condition it gropes for is already covered correctly by `class_never_predicted`. |
| d | **Re-word** `constant_metric`: drop *"or the metric not actually being recomputed"* when `class_never_predicted` also fired this epoch, and say instead *"the model produced the same prediction for every validation sample"*. |
| e | **De-duplicate.** `MetricValidator` becomes the single rule set. Strip from `sanity_check_metrics` every rule `MetricValidator` also implements; keep only its two unique rules (near-perfect weighted precision with low accuracy; `accuracy == balanced_accuracy` streak). `TrainerG_v5.fit` keeps both calls, but each message is now emitted once. |
| f | Pass `labels=list(range(num_classes))` into `compute_classification_metrics` from `TrainerG_v6._validate_one_epoch` (`:200-205`), and size `per_class_support` from `num_classes` rather than `y_true.max()+1` (`eval_utils.py:55`). Latent today; a class with zero validation support would otherwise silently drop out of the macro average. |

- **Verify.** New `test_metric_validation.py` asserting: a correct multi-class result emits **zero**
  `impossible_f1`; a genuinely inconsistent F1 still emits one; `balanced_accuracy > accuracy` emits
  nothing; a zero-support class keeps its slot in the macro average.

**Exit gate for R0:** `pytest -q` green except `test_representation_sensitivity.py`, which must fail
with the recorded pre-fix numbers. Re-run `validation_report.json` generation on the two existing runs
and confirm the RGB `impossible_f1` count drops 32 → 0.

---

## 4. Stage R1 — The representation fix

**Goal:** pass G1 and G2. This is the blocker; nothing downstream is measurable until it clears.

### R1.1 — A tokenizer that can see its input `A1` `gmedmamba.py:634-668`, `:1196-1203`

Three modes behind one config field, legacy default preserved:

```python
# GMedMambaConfig
spectral_token_fusion: str = "add"        # "add" (legacy) | "scaled" | "concat_mlp"
spectral_pe_gain: float = 1.0             # initial value of the (learnable) PE gain
spectral_value_init_std: float = 0.02     # init std for value_embed.weight
```

| Mode | Definition | Params | Notes |
|---|---|--:|---|
| `add` | `tokens = value_embed(v) + pe` | 64 | Exactly today. v14 keeps it. |
| `scaled` | `tokens = value_embed(v) + pe_gain · pe`, `pe_gain` an `nn.Parameter` init `spectral_pe_gain` (v15: **0.1**); `value_embed.weight` re-initialised at `spectral_value_init_std` (v15: **0.5**) *after* `self.apply(_init_weights)` | 65 | Minimal 2-line change. Brings the ratio from 93:1 against the signal to ≈ 4.5:1 in favour of it. |
| `concat_mlp` | `tokens = W₂(GELU(W₁([v ; pe]))) + pe_gain · pe`, `W₁: Linear(1+d, 2d)`, `W₂: Linear(2d, d)` | ~4.2 k | **Recommended.** Fixes scale *and* rank (below). Still fully channel-agnostic — parameter shapes depend only on `d_token`. |

**Why `concat_mlp` and not just `scaled`.** With a shared `Linear(1, d)` and a zeroed bias, every band
token is `v_c · W` — a *scalar multiple of one fixed direction*. All C bands are collinear, and
`pooled = tokens.mean(dim=1)` then reduces an entire spectrum to a single scalar times `W`. `scaled`
fixes the magnitude but leaves that rank-1 structure intact; the band-conv encoder has to undo it. The
concat-MLP's nonlinearity makes each band's *response direction* depend on its own positional encoding,
which is the property a spectral tokenizer needs and the only one that is genuinely absent today.

> **Do not "fix" this with a LayerNorm on the value branch.** `LayerNorm(v·W)` for `v > 0` is
> `LayerNorm(W)` — a constant, independent of `v`. That is the exact mechanism that would annihilate a
> non-normalized (raw radiance) input path, and applying it here would make things strictly worse.
> Recorded here because it is the obvious-looking fix.

- **Verify.** G1 ≥ 0.05 and G2 ≥ 1e-2 for both modalities, all four architectures
  (`split`/`fullchannel`/`efficient` share this tokenizer and are equally affected).
  Assert `spectral_token_fusion="add"` still reproduces the pre-change forward bit-for-bit on a fixed
  seed — this is the C-3 guarantee and must be a test.

### R1.2 — Optimizer parameter groups `C8` `new: training/optim_groups.py`, `train_example_v15.py:optimizer`
- **Change.** `build_param_groups(model, weight_decay) -> list[dict]`: decay only parameters with
  `ndim >= 2` that are not norm weights; everything else (biases, `LayerNorm`/`RMSNorm` gains,
  `pe_gain`, and **`value_embed.weight`** explicitly) gets `weight_decay=0.0`.
- **Why.** `AdamW(model.parameters(), weight_decay=0.05)` currently decays every norm gain and bias, and
  — critically — the one weight that has to *grow* for the model to become input-sensitive.
- **Verify.** Unit test on group membership; confirm G1 does not decay over a 200-step run.

### R1.3 — Wavelength encoding scale `C2` `gmedmamba.py:621-632`
- **Change.** `continuous_wavelength_encoding(wl_norm, dim, span)` — replace the hardcoded `× 1000.0`
  with `× span`, where `span` is the band count `C`. Add
  `wavelength_encoding_scale: Optional[float] = 1000.0` to `GMedMambaConfig`; `None` ⇒ use `C`; v15
  sets `None`.
- **Why.** `index_positional_encoding` uses `position∈[0,1] × div_term × num_bands`, i.e. effectively
  `band_index × div_term`. Using `C` makes the continuous encoding **reduce exactly to the index
  encoding for a uniformly-spaced sensor** and interpolate sensibly for an irregular one. With `1000.0`
  and 32 bands, adjacent bands differ by ~32 radians in the lowest-frequency component — more than five
  full cycles — so the encoding is pseudo-random noise with no locality, which is the one property a
  positional encoding exists to provide.
- **Why now.** This is dead code today (`C1`), but R2.3 turns it on. Fixing C1 without this would make
  results *worse*, and the cause would be hard to find.
- **Verify.** Test that for uniformly-spaced wavelengths, `continuous_wavelength_encoding` matches
  `index_positional_encoding` to 1e-5.

**Exit gate for R1:** `test_representation_sensitivity.py` passes. Then a **3-epoch, 5 %-subsample HSI
run** must reach validation balanced accuracy **≥ 0.40** (i.e. clearly non-constant and in the shallow
probe's neighbourhood) in ≤ 20 minutes. If it does not, stop and re-measure G1/G2 on the *trained*
model — do not proceed to R2.

---

## 5. Stage R2 — Data pipeline

**Goal:** stop discarding signal before the model ever sees it, and pass G4.

### R2.1 — Expose and record input normalization `A3` `train_example_v6.py:104-155`, `train_example_v15.py`
- **Change.** Add `--normalization {per_sample_minmax,global_zscore}` to v15, mirroring the flag
  `train_example_v6.py:240` already has, and thread `global_mean`/`global_std` from the existing,
  currently-unused `compute_global_channel_stats(x_train_path)` (train split only — never val/test).
  **Default `global_zscore` for HSI, `per_sample_minmax` for RGB** (RGB is already in `[0,1]` with
  meaningful absolute colour; re-stretching each patch by its own extremes is the more questionable
  operation there, so make it an explicit ablation rather than a silent default flip).
- **Change.** Record `normalization`, and the fitted `global_mean`/`global_std`, in `config.json`.
  Neither is recorded anywhere today — the two completed runs do not state how their inputs were scaled.
- **Why.** `NpyDataset` defaults to `per_sample_minmax`; v14 never passes the argument. Each patch is
  rescaled by its own scalar min and max over all `11×11×C` values, both driven by outlier pixels
  (measured HSI: min 2005 ± 1855, max 14820 ± 3862). Cost, measured on the linear probe:
  **balanced accuracy 0.4942 → 0.4499**.
- **Verify.** `shallow_baseline.py` reproduces the −0.044 gap; `config.json` round-trips the stats.

### R2.2 — Band selection that spans the sensor `A4` `prepare_histologyhsi_bc_v6.py:492,532`, `prepare_histologyhsi_bc_v7.py`
`selected_band_indices.npy` is exactly `[0…31]` — the first 32 of 740 bands, **400.5–423.0 nm out of
400.5–938.2 nm**, 4.3 % of the sensor, missing the entire haemoglobin absorption region (≈540–580 nm)
and all NIR scatter. `compute_band_importance` blends normalized variance with mutual information; that
score falls almost monotonically with band index, and `select_bands_by_importance` takes a plain top-k,
so it walks off the front of the array.

- **Change (a) — decorrelated greedy selection.** In `select_bands_by_importance`, after ranking, walk
  the ranking and skip any candidate whose `|corr|` with an already-selected band exceeds
  `--band_max_corr` (default 0.95) or whose index is within `--band_min_gap` of one (default 0).
  Falls back to the plain top-k if the constraint cannot be satisfied, with a warning.
- **Change (b) — a `uniform` method.** `--band_selection uniform` = evenly spaced indices across the
  full range. Cheap, deterministic, and a strictly better baseline than the current output.
- **Change (c) — a coverage guardrail.** `save_selected_bands` / `save_importance_selection_report`
  compute `selected_span / sensor_span` and **fail the prep** below `--band_min_coverage`
  (default 0.30) unless `--allow_narrow_bands` is passed. This is the check that would have caught the
  current dataset at prep time instead of two datasets later.
- **Change (d).** Record `wavelength_min/max/span_nm` and `coverage_frac` in
  `band_selection_report.json`.
- **Verify.** Re-prep to `data/hsi_v8/`; assert coverage ≥ 0.9 and `|corr|` between any two selected
  bands ≤ 0.95. **Re-run `shallow_baseline.py` on the new dataset before training anything** — that
  number is the new ceiling and the whole justification for the re-prep.

### R2.3 — Actually pass the wavelengths to the model `C1` `training/trainerg_v11.py`
- **Change.** v11 stores `self._wl = torch.as_tensor(wavelengths, device=...)` and calls
  `base.forward_deep_supervision(x, wavelengths=self._wl)` (and `self.model(x, wavelengths=...)` on the
  non-DS path) when the model's `forward` accepts the keyword — guarded by `--use_wavelengths`
  (default **on** in v15).
- **Why.** `train_example_v14` loads `wavelengths.npy` and hands it to the trainer, but no trainer in
  the `v4 → v10` chain forwards it into any model call; only the old, unused `training/trainer.py:110`
  does. `use_wavelength_metadata`, `continuous_wavelength_encoding` and `sensor_reference_range` are
  inert, so the sensor-agnostic claim in `gmedmamba.py`'s header is currently **untested**.
- **Depends on R1.3.** Do not enable before it.
- **Verify.** Test that the encoding differs between `wavelengths=None` and a real wavelength vector;
  ablation `--use_wavelengths` on/off in R7.

**Exit gate for R2:** a full HSI run beats the §2.1 shallow ceiling (**G4**). If it does not, the model
— not the data — is still the limit; go to R4 before spending more compute.

---

## 6. Stage R3 — Compute budget and throughput

**Goal:** make an epoch a comparable unit of learning across datasets, and stop paying for nothing.

### R3.1 — Subsample the validation split `A2` `train_example_v15.py`
- **Change.** `--val_subsample_frac` (default 1.0) + `--val_subsample_mode {random,stratified}`
  (default `stratified`), deterministic from `--seed`, applied once, recorded in `config.json` **and in
  the run name**. The subset is fixed for the whole run so epoch-to-epoch metrics stay comparable.
- **Why.** The HSI run validated on **334,516** patches per epoch while training on **122,604** — a
  2.7:1 ratio, plus a full `copy.deepcopy` of the model for the EMA pass, every epoch. A stratified
  20–40 k subset has ample statistical power at these sizes.
- **Change.** Also fix `--train_subsample_frac`'s semantics — see R6.1.

### R3.2 — Batch the spectral pathway `C6` `gmedmamba.py:891-902`
- **Change.** `spectral_chunk_size: int = 1024` in `GMedMambaConfig`; `0` ⇒ process all `N` patches in
  one pass. v15 sets `0` for 11×11 patches.
- **Why.** `chunk_size` is hardcoded at 1024 regardless of batch size. At `bs=256`, `patch_size=1`,
  11×11 patches, `N = 30,976`, so each forward runs **31 sequential Python iterations** of the full
  spectral encoder — each of which contains `SpectralMamba`, whose `_selective_scan_pure_pytorch` is
  itself a Python loop over `C` timesteps. Roughly 1,000 sequential launch groups per forward for HSI.
- **Change.** Enable `enable_spectral_gradient_checkpointing` **independently of
  `--gradient_checkpointing`** via a new `--spectral_checkpointing {auto,on,off}` (default `auto` =
  on when `C ≥ 16`). Peak GPU was **9,695 MB (HSI)** vs **1,714 MB (RGB)** for the same 0.44 M-parameter
  model; `training/spectral_checkpoint.py` exists precisely for this and neither run reached it.
- **Verify.** Measured ms/step and peak MB before/after, both datasets, in the style of
  `findings/finds_20260901_140148.md`.

### R3.3 — Make gradient checkpointing honest `C7` `gmedmamba.py:1457-1472`, `training/grad_checkpoint.py:28-35`
- **Change.** `trm_checkpoint_core: bool = True` in `GMedMambaConfig` (legacy default), wired to a new
  `--trm_checkpoint_core / --no_trm_checkpoint_core`. `RecursiveCore.f` currently checkpoints
  unconditionally, so `--gradient_checkpointing` does not control it and the recompute is always paid.
- **Change.** `enable_gradient_checkpointing` returns 0 for `GMedMambaRecursive` (no
  `.backbone.stages`) and prints *"wrapped 0 backbone stage(s)"* while the preflight reports
  `gradient_checkpointing: True`. Either teach it the recursive backbone or **raise** when it wraps
  zero modules on a model the caller explicitly asked to checkpoint. Prefer raising — a silent no-op on
  an explicit flag is how this went unnoticed.
- **Verify.** Test that both helpers return > 0 or raise, for every architecture.

---

## 7. Stage R4 — Model capacity and dead configuration

**Goal:** spend the 442 k parameters where the task is. Only start once G4 is passing — otherwise you
are tuning a model whose input is still broken.

### R4.1 — Remove the duplicate channel MLP `C4` `gmedmamba.py:1370-1384`, `:1407-1428`
Current split, `trm_dim=128`, `trm_core_layers=2`, `trm_ffn_mult=2.0`:

```
per RecursiveMambaBlock   dwconv 3×3 depthwise              1,280
                          _TokenMLPMixer.pw   (GatedMLP)   98,944
                          .ffn                (GatedMLP)   98,944
× 2 layers                                               = 398,336   of 442,215  (90 %)
spatial mixing, whole model                                  4,608   (four 3×3 depthwise convs)
```

`_TokenMLPMixer` is `dwconv → GatedMLP`, and `RecursiveMambaBlock` then applies a **second** `GatedMLP`
as `ffn`. Each "token mixer + channel MLP" block is really one depthwise conv and two channel MLPs.

- **Change.** `trm_mixer_channel_mlp: bool = True` (legacy) → v15 `False`, making `_TokenMLPMixer` a
  pure spatial operator. Spend the freed budget on `--trm_mixer attention` (on 121 tokens this is cheap
  and gives global mixing per `f` call) or a larger depthwise kernel — decide by ablation (R7).
- **Verify.** Parameter-count test per mode; the R7 ablation decides the default.

### R4.2 — Restore regularization to the recursive core `C3` `gmedmamba.py:1407-1428`
- **Change.** `trm_drop_path: float = 0.0` (legacy) and `trm_dropout: float = 0.0` in
  `GMedMambaConfig`, applied inside `RecursiveMambaBlock` using the `DropPath` and `GatedMLP(dropout=)`
  that already exist in this file. Wire `--drop_path_rate` through for `recursive` instead of accepting
  it, writing it into `config.json`, and ignoring it. Allow `--classifier_dropout` for `recursive` (it
  currently raises for anything but `efficient`).
- **Why.** The recursive block has no `DropPath`, no `LayerScale`, no dropout — the hierarchical
  `GBlock` has all three. With R1.2 removing weight decay from norms and biases, the recursive model
  would otherwise have essentially no regularizer.

### R4.3 — Resolve the halting head `C5` `gmedmamba.py:1487-1494`, `:1601-1622`, `training/trainerg_v10.py:120-127`
Measured from the RGB history: `train_loss − classification_loss = 0.5·halt_loss` moved
**0.3381 → 0.3309** across 33 epochs, i.e. halt BCE ≈ **0.66** against `ln 2 = 0.693` — chance, flat,
and **~20 % of the training objective**. Meanwhile nothing ever reads `q`:
`forward_deep_supervision` always runs all `trm_deep_supervision_steps` segments, and
`trm_halt_exploration_prob` is declared in `GMedMambaConfig` and referenced nowhere in the repository.

- **Change (default).** v15 defaults to `--no_trm_halting`, reclaiming the gradient budget.
- **Change (optional, only if R7 shows value).** Implement the real ACT: at eval, stop segments once
  `sigmoid(q) > τ`; during training, use `trm_halt_exploration_prob` as the exploration floor, which is
  what the field was added for. Either implement it or delete the field — do not leave it half-wired.
- **Verify.** R7 ablation `halting {off, loss-only, full-ACT}` on the RGB dataset.

### R4.4 — API and hygiene cleanups `C9`
- `RecursiveHead.forward` returns `(logits, q)` for a tensor input and a bare `logits` for a dict.
  Split into `forward(y) -> (logits, q)` and `forward_features(dict) -> logits`.
- `GMedMambaRecursive.__init__` mutates the caller's config in place (`cfg.recursive`, `cfg.dims`,
  `cfg.depths`). Deep-copy it first.
- `GMedMambaRecursiveBackbone.forward_features` / `_run_segments` duplicate the recursion that
  `forward_deep_supervision` reimplements, and are unreachable from the training path. Make
  `forward_deep_supervision` the single implementation and have `forward_features` call it.
- `self.last_feature_map` is written on every forward including eval. Set it only under
  `self.training`, or return it rather than stashing it.
- **`--recon_mode` default.** v14's parser default is `latent`; combined with `recursive`,
  `TrainerG_v10` feeds the decoder `base.last_feature_map`, which is **detached** — so MSE + 0.1·SAM add
  ≈ 1.0 to the loss while training only the decoder. **v15 defaults `--recon_mode none` for
  `--architecture recursive`** and warns if `latent` is combined with it.

---

## 8. Stage R5 — Reporting integrity

**Goal:** every number in `history.json` means what its label says. This is what makes the results
publishable.

| Item | Finding | Change |
|---|---|---|
| **R5.1** | `D3` | Log `halt_loss`. `_train_one_epoch_deep_supervision` returns it and `TrainerG_v5.fit`'s `metrics` dict never reads it, so it is absent from `history.json`, `history.csv`, the plots and the printed breakdown — which is why `Loss: 1.6955 (CE: 1.3647, MSE: 0.0000, SAM: 0.0000, GAN: 0.0000)` has 0.33 unaccounted for. Add `halt_loss` to the dict, the CSV header and `_log_epoch`'s breakdown. |
| **R5.2** | `D2` | Compute the generalization gap from **last-segment CE on both sides**. Today train `cls_loss` is the mean over all 3 deep-supervision segments *plus* `0.5·halt_loss`, on augmented data, from the live weights; validation is last-segment CE on clean data from the EMA weights. Hence RGB's `GenGap(loss): -0.2247` every epoch. Add `train_cls_loss_last_segment` and derive the gap from it; if that is not wanted, drop the metric for `recursive` rather than publishing an uninterpretable number. |
| **R5.3** | `D1` | Make the checkpoint match the report. `TrainerG_v10._validate_one_epoch` validates the EMA copy but `_save_checkpoint` saves the **live** weights to `best_model.pt`, so it does not reproduce the logged best; only `best_model_ema.pt` does, and that stores a bare shadow dict. In v11: when EMA drives validation, `best_model.pt` **is** the EMA model (full loadable state), the live weights go to `best_model_live.pt`, and `config.json` records which. Also log `train_accuracy` for the EMA model, or stop reporting `GenGap(acc)` — as it stands it compares two different networks. **Gate G5.** |
| **R5.4** | `D4` | Honour `--amp` in validation. `TrainerG_v6._validate_one_epoch:146-149` hardcodes bf16-if-supported, so `--amp off` only affects training and `amp_mode` in `config.json` does not describe the numerics that produced the reported metrics. Add `amp_mode`-aware resolution in v11 (override `_validate_one_epoch`; do not edit v6). |
| **R5.5** | `D5` | **Test-split evaluation.** New `--eval_test {off,best}` (default `best`) in v15: after `fit`, reload the best checkpoint, run `training.evaluator.evaluate_model` once over `X_test.npy`, and write `test_report.json` with per-class extended metrics (`training/per_class_metrics.py`), calibration (`training/metrics.py`) and efficiency (`training/flops_counter.py`) — all of which already exist and are never invoked. Both datasets ship a patient-disjoint test split (5 held-out HSI patients) that has never been evaluated. **Gate G6.** |
| **R5.6** | `D6` | Retention. `--keep_last_n` (default 3) + always keep best; `--eval_artifact_stride` (default 1) for the per-epoch confusion-matrix PNG, classification report and 334 k-row prediction `.npz`. HSI wrote six byte-identical PNGs. Stop embedding the full history in every checkpoint (5,399,147 → 5,407,403 bytes over 6 epochs) — write it to `history.json` only. |

---

## 9. Stage R6 — Reproducibility and provenance

| Item | Finding | Change |
|---|---|---|
| **R6.1** | `E1` | Decide `--train_subsample_frac`'s semantics and make all three strings agree. The module docstring says *"each epoch"*, the argument help says *"each run"*, the console message says *"per epoch"*, and the code (`:861-877`) picks **one fixed subset before the loop**. At 5 % that is the difference between seeing all 2.45 M HSI patches over 20 epochs and seeing the same 122,604 forever. **Choose per-epoch resampling** (`RandomSampler(num_samples=…)`, or a fresh `Subset` per epoch) and add `--train_subsample_mode {fixed,per_epoch}` with `per_epoch` as the v15 default, so the old behaviour stays reachable for reproducing the HSI run. |
| **R6.2** | `E2` | Put the settings that changed the result into the run name. `build_run_name` emits `{ts}_{dataset}_{arch}_{modality}_{mixer}_sup{N}_bs{N}_{amp}` — no subsample, loss, sampler, lr or seed — while v14's docstring claims the subsample fraction is *"recorded in config.json and in the run directory name"*. The two evidence runs differ in `--loss`, `--sampler`, `--train_subsample_frac` and `--dataset_storage`, and their names do not say so. Append non-default values for a fixed allowlist (`loss`, `sampler`, `normalization`, `train_subsample_frac`, `val_subsample_frac`, `seed` when ≠ 42), e.g. `…_wce_sub05_seed7`. Extend `test_run_naming.py`. |
| **R6.3** | `E3` | Fix augmentation RNG duplication. `AugmentationConfig.build()` closes over a dedicated `random.Random(self.seed)` instance which is fork-duplicated into every worker; `make_worker_init_fn` reseeds the **module-level** `random`, so it cannot reach that instance. All 4 workers therefore draw the same flip/rot90 sequence — and both runs used `--augment_preset custom` with `flip_h/flip_v/rotate90 = 0.5`, all three driven by it. **Fix:** derive a per-item seed inside `AugmentedPatchDataset.__getitem__` from `torch.utils.data.get_worker_info().id`, the epoch and the index, or move every draw onto `torch`'s RNG (which *is* reseeded per worker). Update the caveat in `training/augmentation.py:159-172`. |
| **R6.4** | `E4` | Seed everything, or stop claiming to. `make_worker_init_fn(base_seed: int = 0)` is called with no arguments from `DataLoaderPolicy.dataloader_kwargs`, so worker seeds are `0 + worker_id` regardless of `--seed`; `_main` sets `torch.manual_seed` and `np.random.seed` but not `torch.cuda.manual_seed_all` or `random.seed`; `cudnn.benchmark = True` with no `use_deterministic_algorithms`. **Fix:** thread `--seed` through `dataloader_kwargs`, seed all four RNGs, add `--deterministic` (sets `cudnn.deterministic`, clears `benchmark`, calls `use_deterministic_algorithms(True, warn_only=True)`). **Redefine `scientifically_qualified`** as a dict — `{dataset_validated, seeded, deterministic, test_evaluated}` — instead of a single boolean that currently means only "dataset validation was not skipped". |
| **R6.5** | `E5` | Default `--dataset_validation_level` to `deep` for HSI in v15. `prep_log.txt` records `PHYSICALLY UNREADABLE — deep read failed: OSError(5)/EIO at byte 12946243456` for this exact `X_train.npy` on 2026-09-02 10:14; the array was re-finalized and deep-verified afterwards, but the training run then used `structural`, which probes ~19 rows per file. |
| **R6.6** | `E6` | Small fixes: pass `in_channels`/`modality` into `print_preflight` (it accepts them and is called without them, so the line never prints); read `in_channels` from the dataset **before** `build_model` and without consuming an augmentation RNG draw; make `infer_modality` fall back to a content check (`wavelengths*.npy` glob) and warn loudly when a `*_batches` shard dir or a differently-named wavelength file would make a spectral dataset train silently as `rgb`; note in `config.json` when `lambda_gan` was auto-promoted from 0.0 to 0.05. |

---

## 10. Stage R7 — The experiments

Only after G1–G6 pass. Each row is one run; the ablations are thesis content, not overhead.

### 10.1 Qualification (cheap, gates the rest)

| # | Purpose | Command sketch | Budget |
|---|---|---|--:|
| Q1 | Sensitivity + smoke | `pytest test_representation_sensitivity.py` | seconds |
| Q2 | HSI 3-epoch, 5 % subsample | `v15 --data_dir data/hsi_v8/hsi --epochs 3 --train_subsample_frac 0.05 --val_subsample_frac 0.1` | ~20 min |
| Q3 | RGB 3-epoch | `v15 --data_dir data/pad_v6 --epochs 3` | ~30 min |

### 10.2 Ablation matrix (single-variable, per Phase 12's own rule)

| Axis | Values | Answers |
|---|---|---|
| `spectral_token_fusion` | `add` / `scaled` / `concat_mlp` | How much of the failure was scale and how much was rank? **The headline result.** |
| `normalization` | `per_sample_minmax` / `global_zscore` | Confirms the −0.044 probe gap end-to-end. |
| band set | `hsi_v7` (0–31) / `hsi_v8` (decorrelated) / uniform | Isolates the prep bug from the model bug. |
| `use_wavelengths` | off / on | First real test of the sensor-aware claim (`C1`). |
| `trm_mixer` | `mlp` / `attention` | Spatial-capacity question from R4.1. |
| halting | off / loss-only / full-ACT | Whether the halt head earns its 20 % of the objective. |
| `loss` × `sampler` | `ce`/`weighted_ce` × `none`/`moderate_oversample` | Class-imbalance handling, on a model that can actually learn. |

Run each at a **fixed, comparable step budget** — not a fixed epoch count. With `--train_subsample_frac`
at 0.05 the HSI run took **2,868 optimizer steps** against the RGB run's **105,336** (37×); comparing
them as "6 epochs vs 33 epochs" is meaningless. Report steps alongside epochs everywhere.

### 10.3 Reference models

Add `--architecture split` and one non-Mamba baseline (a small 2-D CNN on 11×11 patches) at the same
step budget, plus the §2.1 shallow probes reported alongside every result. Right now the strongest
comparison in the record is a 64-feature logistic regression: on HSI it beats the network by 0.27
balanced accuracy, and on RGB the network is at parity with six numbers. Any claim the thesis makes for
the architecture has to be made against that table, not against the majority-class baseline.

---

## 11. Sequencing and effort

| Stage | Depends on | Wall-clock | GPU |
|---|---|---|---|
| **R0** Guardrails | — | ~0.5 day | none |
| **R1** Representation fix | R0 | ~0.5 day + Q2/Q3 | ~1 h |
| **R2** Data pipeline | R1 | ~1 day (incl. ~1 h re-prep) | ~1 h |
| **R3** Budget/throughput | R1 | ~3 h | ~0.5 h |
| **R4** Capacity | R2 gate G4 | ~0.5 day | ablation |
| **R5** Reporting | R0 | ~0.5 day | none |
| **R6** Reproducibility | R5 | ~3 h | none |
| **R7** Experiments | all | — | the real budget |

R3, R5 and R6 are independent of R1/R2 and can be done in any order or in parallel. **R1 before
everything that consumes GPU time.**

---

## 12. Explicitly out of scope

Recorded so they are not rediscovered as gaps:

- **Do not rewrite `_selective_scan_pure_pytorch` or add a CUDA/Triton kernel.** `--trm_mixer mlp` is
  already the default and `ss2d` is an ablation-only path; `register_scan_backend` is the intended
  extension point. Not on the critical path for anything in this plan.
- **Do not retrain `--architecture split`/`fullchannel`/`efficient` yet.** They share the tokenizer and
  are equally affected by `A1`, so any comparison made before R1 is meaningless. They come back in R7.3.
- **Do not chase the `SIGBUS` / dataset-integrity machinery.** v12/v13 fixed it and it worked: both runs
  passed validation cleanly and the one real EIO was caught and reported by design.
- **Do not tune learning rate, schedule or batch size before R1.** With a 0.2 %-input-dependent
  embedding, every such sweep measures noise.
- **Do not edit `train_example_v14.py`.** C-1.

---

## 13. Traceability

| Finding | Item |
|---|---|
| A1 tokenizer scale/rank | R0.1, R0.2, **R1.1** |
| A2 step budget | R3.1, R6.1, R7.2 |
| A3 per-sample min-max | **R2.1** |
| A4 band selection | **R2.2** |
| B1 harmonic-mean F1 | R0.4a-b |
| B2 balanced-acc rule | R0.4c |
| B3 warning wording | R0.4d |
| B4 duplicate warnings | R0.4e |
| B5 missing `labels=` | R0.4f |
| B6 collapse never acts | **R0.3** |
| C1 wavelengths unused | R2.3 |
| C2 encoding aliasing | R1.3 |
| C3 no regularization | R4.2 |
| C4 capacity split | R4.1 |
| C5 halting head | R4.3 |
| C6 chunk loop / VRAM | R3.2 |
| C7 checkpointing honesty | R3.3 |
| C8 weight decay groups | R1.2 |
| C9 API hygiene, recon default | R4.4 |
| D1 EMA vs live checkpoint | R5.3 (G5) |
| D2 generalization gap | R5.2 |
| D3 halt_loss unlogged | R5.1 |
| D4 validation AMP | R5.4 |
| D5 no test evaluation | R5.5 (G6) |
| D6 artifact churn | R5.6 |
| E1 subsample semantics | R6.1 |
| E2 run naming | R6.2 |
| E3 augmentation RNG | R6.3 |
| E4 seeding/determinism | R6.4 |
| E5 structural validation | R6.5 |
| E6 preflight & misc | R6.6 |
