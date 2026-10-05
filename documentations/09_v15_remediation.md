# `train_example_v15.py` — the Representation Collapse Remediation

> **Files:** [`train_example_v15.py`](../archive/train_example_v15.py) · [`training/trainerg_v11.py`](../training/trainerg_v11.py) · [`training/optim_groups.py`](../training/optim_groups.py) · [`scripts/shallow_baseline.py`](../scripts/shallow_baseline.py) · [`scripts/bench_v15_throughput.py`](../scripts/bench_v15_throughput.py)
> **One-line summary:** the model could not see its input; this is what was wrong, what changed, and what to run.
> **Supersedes:** nothing. [`07_parameters_v14.md`](07_parameters_v14.md) still describes `train_example_v14.py` exactly, and v14 is unmodified on purpose — it remains the entry point that reproduces the two runs this work is based on.

---

## Table of contents

1. [The defect](#1-the-defect)
2. [The compatibility contract](#2-the-compatibility-contract)
3. [Gates](#3-gates)
4. [What v15 changes, by flag](#4-what-v15-changes-by-flag)
5. [Commands](#5-commands)
6. [Where the plan was wrong](#6-where-the-plan-was-wrong)
7. [The qualification run](#7-the-qualification-run-r1s-exit-gate)
8. [What is NOT done](#8-what-is-not-done)

---

## 1. The defect

`medmamba_ss_trm.SpectralTokenizer` built every band token as `value_embed(v) + positional_encoding`.
`value_embed` is a shared `nn.Linear(1, d_token)` initialised by `MedMambaSSBackbone._init_weights`
at `trunc_normal_(std=0.02)` with a zeroed bias, so on inputs in `[0,1]` its output has magnitude
≈ 0.006. The positional encoding — **identical for every sample in the dataset** — has magnitude
≈ 0.55. The input signal was therefore **86–93× smaller than the constant it was added to**.

Measured on 64 real patches through a freshly built model, the stem embedding varied with the
input by **0.25 %** (gate G1 = 0.0025 HSI / 0.0032 RGB). Both completed runs began life as
constant functions. The HSI run's `predictions_epoch01.npz` shows max-softmax confidence spanning
**0.00022** across all 334,516 validation patches, every sample predicted `DCIS`.

Three more instances of the same mechanism were found downstream while fixing it, and all three
had to be fixed for the model to become input-sensitive end-to-end:

| # | Where | The constant | The signal |
|---|---|---|---|
| R1.1 | `SpectralTokenizer` | positional encoding, \|·\| ≈ 0.55 | value embedding, \|·\| ≈ 0.006 |
| R1.1b | `stem_norm` | `nn.LayerNorm(eps=1e-5)` | context map arrives with variance ≈ 1.1e-5 — **below eps**, so the LayerNorm degenerates into a constant rescale instead of normalising |
| R1.1c | `MedMambaSSTRMBackbone.embed` | 2-D sinusoidal encoding, \|·\| ≈ 0.56 | stem output, \|·\| ≈ 0.003 |
| R1.1d | `RecursiveHead.fc` | — | `trunc_normal_(0.02)` on a `Linear(128, C)` is 4.4× below that layer's own fan-in scale, costing a matching factor of logit sensitivity |

`training/optim_groups.py` (R1.2) then stops `weight_decay=0.05` from shrinking the one weight
that has to *grow*.

**Result** (64 real patches, freshly built `--architecture recursive`):

| | G1 stem sensitivity | G2 logit std |
|---|--:|--:|
| pre-v15, HSI | 0.0025 | 5e-07 |
| **v15, HSI** | **0.058** | **0.024** |
| pre-v15, RGB | 0.0032 | 5e-07 |
| **v15, RGB** | **0.058** | **0.021** |

---

## 2. The compatibility contract

| # | Rule | How it is enforced |
|---|---|---|
| C-1 | `train_example_v14.py` is not modified | v15 is a new file |
| C-2 | `TrainerG_v10` is untouched | `TrainerG_v11` subclasses it; `test_trainerg_v11.py` asserts v10 has none of the v15 knobs |
| C-3 | Every change in a shared module sits behind a field whose **default reproduces today's behaviour exactly** | `test_v15_model_changes.py::test_every_new_config_field_defaults_to_legacy_behaviour`, plus a verified bit-for-bit forward comparison against the pre-change tree for all four architectures × both modalities |
| C-4 | No checkpoint-format break; a checkpoint is never silently loaded into a different tokenizer | `TrainerG_v11.load_checkpoint` compares `spectral_token_fusion` and raises on a mismatch |
| C-5 | Every item lands with a test that fails before and passes after | **201 new tests** — `test_representation_sensitivity.py` (41), `test_v15_model_changes.py` (76), `test_trainerg_v11.py` (35), `test_band_selection_coverage.py` (22), `test_metric_validation.py` (14), plus 13 added to `test_run_naming.py`. Suite total 133 → 334, all green |

`test_representation_sensitivity.py::test_legacy_defaults_still_reproduce_the_defect` asserts the
legacy path **still exhibits the defect** — if it ever starts failing, a v15 default has leaked
into v14's path.

---

## 3. Gates

| Gate | Definition | Pre-v15 | Target | Status |
|---|---|--:|--:|---|
| **G1** | stem sensitivity `mean_j(std_i(e_ij)) / mean_ij(\|e_ij\|)` | 0.0025 / 0.0032 | ≥ 0.05 | **passing** (0.058) |
| **G2** | `logits.std(dim=0).mean()` | ~5e-07 | ≥ 1e-2 | **passing** (0.021–0.024) for `recursive` |
| **G3** | ≥ 2 distinct predicted classes after epoch 1 | 1 of 3 | ≥ 2 | enforced as a stop condition (`--on_class_collapse abort`) |
| **G4** | beats the shallow ceiling | below it | above | **not yet** — a 432-step qualification run reaches balanced accuracy 0.573 on the held-out HSI test split against the probe's 0.608 (§7); closing it needs the R7 budget |
| **G5** | `best_model.pt` reproduces the logged best metric | failed | passes | **passing**, verified end-to-end |
| **G6** | `test_report.json` from one pass over `X_test.npy` | absent | present | **passing** |

Run G1/G2 with `pytest test_representation_sensitivity.py -q` (seconds, no GPU). The shallow
ceiling for G4 comes from `python scripts/shallow_baseline.py --data_dir <dir>` (~4 min, no GPU):

| dataset | best shallow probe | balanced acc. | F1-macro |
|---|---|--:|--:|
| `hsi_v7-80_10_10` | per-band mean + std (64 numbers) | **0.6078** | **0.5612** |
| `pad_v6` | per-channel mean + std (6 numbers) | **0.2899** | **0.2217** |

The HSI run reached 0.3333 / 0.0456 — far *below* 64 numbers. The RGB run reached 0.2571 / 0.2419
— at parity with six numbers, after 442,602 parameters and 17.5 GPU-hours.

---

## 4. What v15 changes, by flag

Defaults are v15's; the value in brackets is what the shared module still defaults to, so v14 is
unaffected.

### R0 — guardrails (no behavioural change to training)

| Flag | Default | What it does |
|---|---|---|
| `--sensitivity_check` | `on` | asserts G1/G2 on one real batch before training; aborts with `REPRESENTATION_COLLAPSE` |
| `--min_stem_sensitivity` / `--min_logit_std` | `0.05` / `1e-2` | the gate thresholds |
| `--on_class_collapse` | `abort` [warn] | stop when the validation split has been assigned one class for `--class_collapse_streak` epochs |
| `--numerical_smoke_test` | `on` [auto] | one forward/backward/step cycle |

R0.4 also repaired the metric validators. The harmonic-mean F1 rule fired on **32 of the RGB
run's 33 epochs** on entirely correct metrics (macro-F1 is the *mean* of the per-class F1s, never
the harmonic mean of the macro averages); the `balanced_accuracy > accuracy` rule fired on
exactly the case where it is legitimate. Re-running the validators over both completed runs now
gives **0** `impossible_f1` and preserves the 21 real `class_never_predicted` findings.

### R1 — the representation fix

| Flag | Default | Legacy |
|---|---|---|
| `--spectral_token_fusion` | `concat_mlp` | `add` |
| `--spectral_pe_gain` | `0.1` | `1.0` |
| `--spectral_value_init_std` | `0.5` | `0.02` |
| `--spectral_ctx_norm` | on | off |
| `--spatial_pe_gain` | `0.1` | `1.0` |
| `--classifier_init` | `fan_in` | `shared` |
| `--wavelength_scale` | unset ⇒ band count | `1000.0` |
| `--weight_decay_groups` | on | — |

`concat_mlp` fixes scale **and rank**: with a shared `Linear(1,d)` and a zeroed bias every band
token is a scalar multiple of one fixed direction, so all `C` bands are collinear and
`tokens.mean(dim=1)` reduces a whole spectrum to a single scalar. Do **not** try to fix this with
a LayerNorm on the value branch — `LayerNorm(v·W)` for `v > 0` is `LayerNorm(W)`, a constant.

### R2 — data pipeline

| Flag | Default | Note |
|---|---|---|
| `--normalization` | `auto` ⇒ `global_zscore` (HSI) / `per_sample_minmax` (RGB) | neither completed run records how its inputs were scaled; v15 records the mode **and the fitted statistics** in `config.json` |
| `--use_wavelengths` | on | v14 loaded `wavelengths.npy` and no trainer in the v4→v10 chain ever forwarded it into a model call |
| `--sensor_range` | unset | fixed physical range for cross-sensor normalization |

Band selection (`prepare_histologyhsi_bc_v6/v7.py`) gained `--band_max_corr` (0.95),
`--band_min_gap`, `--band_selection uniform`, and a **coverage guardrail**: `--band_min_coverage`
(0.30) fails the prep unless `--allow_narrow_bands` is passed. `data/hsi_v7`'s
`selected_band_indices.npy` is exactly `[0..31]` — 400.5–423.1 nm out of 400.5–938.2 nm,
**4.2 % of the sensor**, missing the entire haemoglobin absorption region and all NIR scatter.
The guardrail reproduces that number and refuses it.

### R3 — compute budget

| Flag | Default | Note |
|---|---|---|
| `--val_subsample_frac` / `--val_subsample_mode` | `1.0` / `stratified` | the HSI run validated on 334,516 patches while training on 122,604 |
| `--spectral_chunk_size` | `1024` | **see §6** — the plan's recommended `0` is a 35 % regression |
| `--spectral_checkpointing` | `auto` (on for C ≥ 16) | 10,243 → 1,007 MB on HSI for +15 % step time |
| `--trm_checkpoint_core` | on | now a real switch; off OOMs a 16 GB card on HSI |

### R4 — capacity

`--trm_mixer_channel_mlp` (on), `--trm_drop_path`, `--trm_dropout`, `--classifier_dropout` (now
accepted for `recursive`), `--trm_halt_threshold` (unset). `--recon_mode` defaults to `none` for
`recursive`: the latent decoder reads a **detached** feature map, so it added ≈ 1.0 to the
reported loss while training only itself. `--no_trm_halting` is the default — the halt BCE sat at
≈ 0.66 against `ln 2 = 0.693` for 33 epochs while being ~20 % of the objective.

### R5 / R6 — reporting and reproducibility

`--eval_test` (`best`), `--verify_best_checkpoint` (on), `--keep_last_n` (3),
`--eval_artifact_stride` (1), `--train_subsample_mode` (`per_epoch`), `--deterministic` (off).

`history.json` now carries `halt_loss`, `train_cls_loss_last_segment`,
`optimizer_steps_this_epoch`/`_total`, and a generalization gap computed from **last-segment CE on
both sides** (the RGB run reported an identical `GenGap(loss): -0.2247` on all 33 epochs because
it differenced a 3-segment mean plus half a halt loss against a single-segment CE).

Run directories now carry the settings that changed the result:
`…_bs256_bf16_wce_sub05_seed7`.

---

## 5. Commands

```bash
# Q1 - the gates. Seconds, no GPU, no dataset needed.
pytest test_representation_sensitivity.py -q

# the ceiling every result must be read against. ~4 min, no GPU.
python scripts/shallow_baseline.py --data_dir data/hsi_v7-80_10_10/hsi data/pad_v6

# Q2 - HSI qualification.
PYTHONPATH=.:archive python archive/train_example_v15.py --data_dir data/hsi_v7-80_10_10/hsi --epochs 3 \
    --architecture recursive --train_subsample_frac 0.015 --val_subsample_frac 0.05 \
    --loss weighted_ce --loader_mode performance --dataset_storage mmap

# Q3 - RGB qualification.
PYTHONPATH=.:archive python archive/train_example_v15.py --data_dir data/pad_v6 --epochs 3 --architecture recursive

# the headline R7 ablation: how much of the failure was scale, how much was rank?
for f in add scaled concat_mlp; do
    PYTHONPATH=.:archive python archive/train_example_v15.py --data_dir data/pad_v6 --epochs 10 \
        --architecture recursive --spectral_token_fusion $f
done

# a band selection that spans the sensor (R2.2), then re-measure the ceiling
python prepare_histologyhsi_bc_v7.py --root <HistologyHSI> --out_dir data/hsi_v8 \
    --band_selection uniform --num_bands 32 --num_workers 8
python scripts/shallow_baseline.py --data_dir data/hsi_v8/hsi
```

---

## 6. Where the plan was wrong

**R3.2's recommended `--spectral_chunk_size 0` is a 35 % throughput regression for zero memory
benefit.** The plan reasoned that the hardcoded 1024 meant 31 sequential Python iterations per
forward and that removing them would be faster. Measured on an RTX 5060 Ti at bs=256/bf16, HSI
runs 1,175 ms/step at 1024 against 1,592 ms/step at 0, with a clear U-shaped optimum at 1024 and
**flat peak memory across the whole sweep**; the ordering holds at bs=128. The model was never
launch-bound at this size.

Worse, the two R3.2 recommendations interact badly: `chunk_size=0` makes the batch a single chunk,
so `--spectral_checkpointing` has one activation to discard and saves ~6 % instead of ~90 %.
Applying both of the plan's recommendations together gives the slowest configuration *and* still
10 GB of peak memory.

**v15 therefore ships `--spectral_chunk_size 1024`.** R3.2's actual defect — that the value was
hardcoded and could not be chosen — is fixed, with a test asserting the result is
chunk-size-independent. Full numbers: [`findings/finds_20260902_v15_r3.md`](../findings/finds_20260902_v15_r3.md).

A second, smaller correction: R1.1b cannot use `_rms_norm_lastdim`, whose `+1e-6` is an
**absolute** floor on a mean-square and is therefore useless at the 1e-5 scale the context map
actually arrives at (it produces an output RMS of 0.0095 instead of 1.0). `_rms_norm_scale_invariant`
divides by the RMS itself.

---

## 7. The qualification run (R1's exit gate)

`--data_dir data/hsi_v7-80_10_10/hsi --epochs 3 --train_subsample_frac 0.015
--val_subsample_frac 0.05 --loss weighted_ce`, 432 optimizer steps, ~11 min on an RTX 5060 Ti.
Time-boxed to the plan's 20-minute budget, which on this dataset (4.9 M train patches, roughly
double the one the plan sized against) means 1.5 % per epoch rather than 5 %.

| | steps | balanced acc. | F1-macro | classes predicted |
|---|--:|--:|--:|--:|
| pre-v15 HSI run (6 epochs, 2.2 GPU-h) | 2,868 | 0.3333 | 0.0456 | **1 of 3** |
| v15 epoch 1 | 144 | 0.3853 | **0.3175** | 3 of 3 |
| v15 epoch 2 | 288 | 0.3856 | 0.2374 | 3 of 3 |
| v15 epoch 3 | 432 | **0.3965** | 0.2118 | 3 of 3 |

**F1-macro 0.0456 → 0.3175 at one seventh of the optimizer steps**, and the model predicts all
three classes from epoch 1. That is R1 working.

Two honest qualifications:

* The plan's numeric gate is *balanced accuracy ≥ 0.40 after 3 epochs*. This run reached
  **0.3965** — just short, at 15 % of the plan's intended step budget.
* The run is **overfitting hard**, and the newly-correct instrumentation is what says so: the
  matched generalization gap (R5.2) goes +0.35 → +1.24 → +1.89 while train CE falls 0.85 → 0.48
  and validation CE rises 1.20 → 2.36. The old `GenGap(loss)` could not have shown this — it
  differenced a 3-segment mean plus half a halt loss against a single-segment CE, and reported the
  same number every epoch.

The indicated next lever is therefore regularization (`--trm_drop_path`, `--trm_dropout`,
`--classifier_dropout`, all newly available for `recursive` via R4.2) and more data per epoch —
not more steps at these settings.

**Gate G5 passed** (`f1_macro` reloaded from `best_model.pt` to a delta of exactly 0.0), and
**gate G6 produced the first held-out test number this project has ever recorded** — one pass
over the 348,894 patches of the five held-out HSI patients, with the EMA weights of the best
epoch:

```
n = 348,894    accuracy 0.3178    balanced accuracy 0.5728    F1-macro 0.3275
per-class recall  [0.800, 0.819, 0.099]        ECE 0.1199
true  histogram   [ 83,538   24,570  240,786]
pred  histogram   [133,667  191,285   23,942]
```

Read that against §3's ceiling: **balanced accuracy 0.5728 against the 64-number shallow probe's
0.6078**, after 432 optimizer steps on 1.5 % of the training split. The model is now in the same
regime as the probe rather than three tenths below it. It is also clearly mis-calibrated toward
the two minority classes — `--loss weighted_ce` on a test split whose majority class is the one
the training split under-weights — which is a real, addressable finding, and precisely the kind of
thing that was invisible while every prediction was `DCIS`.

`--eval_artifact_stride 3` wrote per-epoch artifacts for epochs 1 and 3 only (R5.6), and
`--keep_last_n 1` left one per-epoch checkpoint plus `best_model.pt`, `best_model_live.pt` and
`best_model_ema.pt`.

---

## 8. What is NOT done

* **G4 — beating the shallow ceiling.** That is a full training run per dataset, i.e. the R7 GPU
  budget. Everything needed to run it and to report it honestly is in place; the number is not.
* **The R7 ablation matrix and reference models** (§10 of the plan). Every axis is now a CLI flag,
  and `scripts/bench_v15_throughput.py` gives the step-budget arithmetic, but no ablation has been
  run.
* **The `data/hsi_v8` re-prep.** The guardrail, the `uniform` method and the decorrelated greedy
  walk are implemented and tested against `hsi_v7`'s actual selection; running the prep needs the
  raw HistologyHSI-BC-Recurrence source tree.
* **G2 for `split` / `fullchannel` / `efficient`.** They clear G1 everywhere (0.06–0.09 against a
  pre-v15 0.003) and land at 4e-3 to 1.6e-2 on G2 — a 25–100× improvement that does not uniformly
  reach 1e-2. The remaining attenuation is in the hierarchical backbone
  (`layerscale_init=1e-4` makes every residual branch near-identity at init), not in the
  tokenizer, and the plan defers these architectures to R7.3 anyway. They are held to G1 in full
  and to a 1e-3 G2 floor by `test_representation_sensitivity.py`.
