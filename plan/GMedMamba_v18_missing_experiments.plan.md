# GMedMamba v18 — Missing Experiments for Model Comparison

**Date:** 2026-09-15 · **Against:** `paper/draft/MedMamba-SS-TRM_manuscript_v8.md` (938 lines)
**Reference artefact shape:** `experiments/20260914_181738_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon`
**Constraint:** every run must be fairly quick, must carry reconstruction metrics **and** images, and must report its own wall clock.

This plan is a **gap analysis plus an execution order**, not a rewrite of the manuscript. Everything
below is either *(verified today)* — measured in this session against the repository as it stands — or
*(from artefact)* — read out of a run directory. Nothing is carried on the earlier manuscript's authority.

---

## 0. What was checked

| Source | What was read |
| --- | --- |
| `paper/draft/MedMamba-SS-TRM_manuscript_v8.md` | all 938 lines |
| `experiments/20260914_181738…_optimal_recon` | `config.json`, `test_report.json`, `history.csv`, `experiment_report.json`, `reconstruction/{status,provenance,samples_metrics}.json`, directory tree |
| `experiments/20260906_222725…` / `20260907_025027…` / `20260909_221520…` | the manuscript's §7.2 headline pair and its replicate |
| `data/hsi_v8-80_10_10_importance{,-new}/prep_log.txt` | the argv each build was produced from |
| `training/config_presets.py`, `training/reconstruction_head.py`, `training/latent_space.py`, `training/trainerg_v1{1,2}.py`, `training/flops_counter.py` | model construction, decoder shape, diagnostics call sites |
| `train_example_v16{,_optimal,_optimal_recon}.py`, `train_example_v15.py` | CLI surface, `OPTIMAL_DEFAULTS`, override composition |
| `documentations/12_commands_datasets.md` | the six ablations already specified but never run |
| `medmamba-original/MedMamba/train_hsi.py` + `runs_hsi/` | baseline entry point and the one HSI baseline run that exists |
| CPU instantiation of `build_model` | parameter counts for all four architectures, both modalities |

No GPU is available in this session, so nothing here was trained. Parameter counts *were* recomputed.

---

## 1. Findings that block a clean comparison

Seven, ordered by how much they cost the paper. F1–F3 are defects; F4–F7 are missing controls that
the paper already names but whose blocking causes it does not.

### F1 — The headline RQ0 pair is built on a pre-fix RGB build *(verified today)*

`data/hsi_v8-80_10_10_importance/prep_log.txt` line 1 records the build both §7.2 runs consume:

```
2026-09-04T14:26:06  prep start: adapter=histology_hsi_bc_v8 argv=--root … --band_selection importance
--num_bands 32 --band_min_gap 8 --band_max_corr 0.95 --band_min_coverage 0.30 --split 80_10_10
--seed 42 --num_workers 8 --verify_level deep
```

There is **no `--rgb_source` in that argv**, and there could not have been: the flag was introduced
in commit `3cca0d0` on **2026-09-12**, eight days after the build. That build therefore used the
legacy directory-iteration-order path — the one §6 itself describes as "an RGB arm built from a
mixture of two optical sources, 29.5 % of it resized from a different field of view, and not
reproducible across filesystems."

`documentations/12_commands_datasets.md` §B.3 states this outright: *"Measured under `legacy`: HSI
0.9073 balanced / 0.8580 macro-F1 vs RGB 0.7571 / 0.7282."* The manuscript at line 403 says the
opposite — *"Only `synthetic` makes 'HSI vs. RGB' a single-variable comparison, and it is what §7.2
uses."*

**Consequence.** The paper's single largest empirical claim (+15.0 points balanced accuracy, the
whole of RQ0) rests on the arm the paper's own §6 spends a paragraph discrediting. The margin may
well survive — a mixed-optics RGB arm is if anything *handicapped*, so the true spectral margin is
probably smaller, not larger — but as it stands the sentence at line 403 is not supported by the
artefact. **This is E1, and it is the highest-value run in this plan.**

The corrected build already exists: `data/hsi_v8-80_10_10_importance-new/` was prepared 2026-09-13
with `--rgb_source synthetic`, same seed, same split, same band selection, identical sample counts
(2,452,086 / 334,516 / 348,894). Only its HSI arm has been trained.

### F2 — The "61× smaller" ratio compares two different tasks *(verified today)*

Instantiating `build_model` on CPU at the current presets and the v15 representation overrides:

| Architecture | Modality | Classes | Trainable parameters |
| --- | --- | ---: | ---: |
| `recursive` | hsi | 3 | **446,409** ✓ matches `config.json` |
| `split` (hierarchical) | hsi | 3 | **2,773,007** |
| `fullchannel` | hsi | 3 | 3,513,451 |
| `efficient` (gated) | hsi | 3 | 3,685,483 |
| `split` (hierarchical) | rgb | 6 | **27,425,314** |
| `split` (hierarchical) | rgb | 3 | 27,423,055 |
| `recursive` | rgb | 6 | **446,796** ✓ matches the PAD run |

The 27.43 M figure is the **RGB/PAD** preset — `dims=(96,192,384,768)`, `depths=(2,2,4,2)`,
`patch_size=4` (`training/config_presets.py:31`). The HSI preset is three stages at
`(64,128,256)` with `patch_size=1` (`config_presets.py:29`) and comes to 2.77 M.

§3.6's table already says this correctly ("MedMamba-SS hierarchical, HSI, 3 cl. | 2.77 M"). But line
271, line 664, the abstract (line 15) and the conclusion (line 816) all state ≈ 61× **"on the same
task"**, pairing the recursive *HSI* model against the hierarchical *RGB* one.

**On the task the substitution is actually evaluated on, the ratio is 6.2×, not 61×.** The 61×
figure is valid only for PAD-UFES-20 whole-image RGB (27,425,314 / 446,796 = 61.4×), where the
recursive model *is* also evaluated — so the fix is to attach each ratio to its task, not to drop one.

Two side effects worth having: both rows are now *(verified)* rather than *(prior)*, and the
`fullchannel` / `efficient` rows are new numbers the paper does not have.

### F3 — Three artefact classes are silently absent from every run

**F3a — FLOPs. Every `test_report.json` in the repository carries `"flops": {"error": …}`.**
`training/trainerg_v11.py:717` calls

```python
report["flops"] = count_flops(self.model, tuple(sample_x.shape[1:]), device=str(self.device))
```

`sample_x.shape[1:]` strips the batch dimension, so `count_flops` builds a 3-D `torch.randn` and the
model's forward raises `ValueError: not enough values to unpack (expected 4, got 3)`, which the
surrounding `except` converts into the stored error string. It needs `(1, *sample_x.shape[1:])`.
This is why §7.6's entire FLOP panel is tagged *(prior)* and why the paper's most transferable
result — the cost inversion — is the least evidenced one in it.

**F3b — Latent space. `latent_space/` is empty in every reconstruction run.**
`training/latent_space.py:62` iterates `for x, y in loader:`, but with `--recon_mode latent` the
loader yields 3-tuples `(x, y, target_cube)`. The `ValueError` propagates to
`training/trainerg_v12.py:1170-1178`, which logs `[diagnostics] latent-space visualization failed`
and continues. Confirmed across four runs:

| Run | `recon_mode` | files in `latent_space/` |
| --- | --- | ---: |
| `20260914_181738…` | `latent` | **0** |
| `20260904_230812…` | `latent` | **0** |
| `20260906_222725…` (headline HSI) | `none` | 1 |
| `20260907_025027…` (headline RGB) | `none` | 1 |

Since the user's requirement is that *every* new experiment carry reconstruction artefacts, every
new experiment would also lose its embedding plot. That plot is the natural qualitative figure for
"what does 32 bands buy that 3 does not" and for "does the recursive core separate classes the way
the hierarchical one does" — exactly the understanding this plan is meant to add.

**F3c — Run-to-run spread is measured but mislabelled.** `documentations/12_commands_datasets.md`
§B.3 calls `20260909_221520…` "a second RGB seed". Its `config.json` records `seed: 42` — the same
seed as `20260907_025027…`. It is a **nondeterminism replicate** (`deterministic: false`, with
`torch.compile` on), and as such it is a useful number the paper does not use:

| | balanced acc. | macro F1 |
| --- | ---: | ---: |
| RGB, seed 42, run 1 | 0.7571 | 0.7282 |
| RGB, seed 42, run 2 | 0.7615 | 0.7332 |
| **spread** | **0.44 pt** | **0.005** |

That is a 0.44-point run-to-run floor against a 15.0-point RQ0 margin — evidence *for* the margin
that §9.1 currently does not claim, because it thinks the second run is a different seed. (It says
nothing about *seed* variance, which still needs E7.)

### F4 — The reconstruction decoder is **not** band-count agnostic *(verified today)*

`training/reconstruction_head.py:96`:

```python
self.out_conv = nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)   # out_channels = C
```

with `hidden = max(32, latent_channels // 2) = 64` for `dims[-1] = 128`
(`reconstruction_head.py:119`). So the decoder costs `64·C·9 + C` parameters in its output layer:

| | backbone | decoder | total |
| --- | ---: | ---: | ---: |
| 32 bands | 446,409 | 129,440 | **575,849** ✓ `test_report.json → efficiency.num_params` |
| 3 bands | 446,409 | 112,707 | 559,116 *(derived)* |

Proposition 1 is untouched — it is a claim about the classifier — but two things follow that the
manuscript must state before publishing any recon-bearing comparison:

1. **Quote `backbone_num_params`, never `efficiency.num_params`, for the RQ1 claim.** In a
   reconstruction run the `efficiency` block reports the *wrapped* model, so §7.1's "Parameter
   memory 1.786 MB" row becomes 2.303 MB and the "difference 0" row becomes a difference of 16,733.
   The artefact already separates the two fields; the paper must pick the right one and say which.
2. **The agnosticism claim has a stated boundary**, which is more honest than not having one: the
   classifier is exactly band-count-agnostic, the auxiliary decoder is linear in band count. A
   shared per-band output projection would close it, and that is a design note, not this plan's work.

### F5 — §7.3's "single largest caveat" has a one-command fix that was never run

The only MedMamba HSI baseline in existence is
`medmamba-original/MedMamba/runs_hsi/20260905_051406_hsi_v8-80_10_10_importance_undersample_…`,
trained on the **class-balanced** build (368,550 patches at exactly 122,850 per class) while
MedMamba-SS-TRM trained on the natural 13.1:1 build. §7.3 correctly refuses to read the result as an
architectural win because of it. The natural-distribution build has existed since 2026-09-04 and the
corrected one since 2026-09-13; nothing prevents the matched run except that `train_hsi.py` has no
subsample flag, so it would train the full 2.45 M patches per epoch. At the measured 155 s / 368,550
patches that is ≈ 1,030 s/epoch — and MedMamba reaches its best validation macro-F1 at **epoch 3 of
30** on this dataset, so five epochs is a generous budget. **≈ 1.4 h.**

One genuine mismatch survives and should be fixed in the same pass: `train_hsi.py`'s `--normalize`
accepts only `none | per_patch_zscore | per_sample_minmax`. Our runs use `global_zscore`. That repo
is not frozen; adding the third mode is ~10 lines and removes a named confound.

### F6 — The recursion-depth question is answerable with one flag and has never been asked

§7.8 says it plainly: *"nothing here establishes that 63 core applications are necessary, or that the
same accuracy could not be had at 21. That is the most obvious missing experiment in the paper, and
it is cheap."* §8.2 then concedes the paper cannot say whether TRM's benefit transferred.

Core applications = `(trm_n_latent + 1) × trm_n_improve × trm_deep_supervision_steps`. At the
reported `n_latent=6, sup=3`, varying `--trm_n_improve` gives exactly the 21 / 42 / 63 / 84 ladder
§7.6 already priced in FLOPs. **Vary `n_improve`, not `deep_supervision_steps`** — the latter also
changes the number of supervised segments in the objective, which confounds depth with supervision
density. The paper's §7.6 sentence "varying the segment count and inner loop" suggests both axes were
mixed in the FLOP sweep; for the *quality* sweep only one may move.

### F7 — Behavioural band-count agnosticism can be tested without preparing a single new dataset

`documentations/12_commands_datasets.md` §G specifies the band sweep as four fresh preparations
(~1.8 h each, ~43 GB) and then notes: *"If disk is tight, prep only `--num_bands 32` and subset
channels at training time instead."* **There is no flag that does that** — no `--band_subset`,
`--keep_bands` or equivalent anywhere in the v15/v16 parser or in `training/npy_dataset_v16.py`.

That missing 30-line wrapper is what stands between the repository and the cheapest experiment in
this plan: the model is band-count-agnostic *by construction*, so the **already-trained 32-band
checkpoint can be evaluated at 16, 8, 4 and 2 bands with no training at all** — minutes, not hours.
That is a direct behavioural test of RQ1's open half, on the existing best checkpoint.

---

## 2. Prerequisites — three read-only scripts, no GPU, no frozen file touched

The freezing rule holds: `training/trainerg_v11.py`, `training/trainerg_v12.py` and
`training/latent_space.py` are all on the audited import path. None is edited. Each prerequisite is a
**new standalone script** that consumes a finished run directory — which has the side benefit that all
three retro-fit the 81 runs already on disk, including the manuscript's headline pair.

`scripts/eval_test_split_v16.py` is the template for all three: it rebuilds the loader from a run's
own `config.json` and calls the trainer's evaluation path without importing anything for mutation.

### P1 — `scripts/flops_report_v16.py` *(closes F3a)*

```
python scripts/flops_report_v16.py --run_dir experiments/<run>        # writes <run>/flops_report.json
```

Rebuilds the model from `config.json` via `training.config_presets.build_model`, calls
`count_flops(model, (1, C, H, W), device="cpu")`, adds
`estimate_selective_scan_flops(...)` for the scan term the counter excludes, and records core
applications, parameter memory and the measured `latency_bs1_ms_mean` already in `test_report.json`.
CPU-only — the pure-PyTorch scan backend at `gmedmamba.py:81` means no CUDA is needed.

*Run it over every architecture from §1 F2 in one pass.* That converts the whole of §7.6's efficiency
panel from *(prior)* to *(verified)* — including, for the first time, a hierarchical-vs-recursive FLOP
comparison **on the same task**, which is what F2 showed was missing.

**Cost:** ~1 h to write, minutes to run over all runs. **No GPU.**

### P2 — `scripts/latent_space_v16.py` *(closes F3b)*

```
python scripts/latent_space_v16.py --run_dir experiments/<run> [--split validation|test]
```

Rebuilds the loader with `NpyDatasetV16`, tolerates both 2-tuple and 3-tuple batches (the one line
`training/latent_space.py:62` gets wrong), extracts penultimate features through
`forward_features`, and calls `training.plots.plot_latent_space`. Forwards `wavelengths` —
`trainerg_v12.py:1173` does not, so even the two runs that produced a plot produced it without them.

**Cost:** ~1 h to write, ~2 min per run (capped at 2,000 samples). **GPU optional** (CPU is fine at
that cap).

### P3 — `scripts/eval_band_decimation_v16.py` *(closes F7, enables E8)*

```
python scripts/eval_band_decimation_v16.py --run_dir experiments/<hsi run> --keep_bands 32,16,8,4,2
```

A `torch.utils.data.Dataset` wrapper that slices the channel axis of an existing build and slices
`wavelengths.npy` identically (uniform decimation over the 32 selected centres — note the 219.0 nm
gap, so "uniform in index" is not "uniform in nm"; record both). For each band count it calls the
same `evaluate_test_split` the entry point would, writing
`band_decimation/test_report_C{n}.json` plus one summary table.

**Cost:** ~2 h to write, ~10 min per checkpoint for the whole ladder. **GPU for the forward passes.**

---

## 3. The experiment set

**Common template.** Every arm below is the reference run's command with **one** flag changed. The
reference command, reconstructed field-by-field from
`experiments/20260914_181738…/config.json → cli_args` (everything else is `OPTIMAL_DEFAULTS` +
`RECON_DEFAULTS`):

```bash
python train_example_v16_optimal_recon.py \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 20 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --seed 42
```

which resolves to `recursive`, `trm_dim 128`, `trm_core_layers 2`, `trm_n_latent 6`,
`trm_n_improve 3`, `trm_deep_supervision_steps 3`, `trm_mixer mlp`, halting off,
`recon_mode latent`, `lambda_mse 0.1`, `lambda_sam 0.1`, `loss focal_weighted` at `gamma 1.5`,
`class_weight_power 0.75`, `sampler none`, `global_zscore`, `augment_preset custom`, `bf16`,
`compile on`, `trm_ema_rate 0.9995` (resolved from steps/epoch), `checkpoint_metric f1_macro`,
`eval_test best`.

**Epoch budget.** The reference run selected epoch **7 of 20** and its validation macro-F1 declined
monotonically afterwards (0.699 → 0.644). **Ablation arms run 12 epochs, not 20** — 2.20 h instead
of 3.66 h, with five epochs of headroom past the observed optimum. Only the RQ0 pair (E1) runs the
full 20, so it stays byte-comparable to the reference run.

**Measured cost basis** *(from the reference run's `history.csv`)*: median `epoch_time` 653.1 s,
`train_time_s` 626.9 s, `val_time_s` 25.8 s, `s_per_step` 0.64, 978 steps/epoch, peak 843 MB VRAM,
20 epochs = 3.66 h, 12 epochs = 2.20 h. Depth-arm estimates below scale `train_time_s` by
§7.6's measured 95.7 MFLOP/core-application on a 157 MFLOP intercept.

---

### E1 — The matched HSI/RGB pair, on the corrected build ★ highest value

**Closes:** F1, and re-establishes RQ0 under the current best recipe. **Cost: ≈ 1.9 h.**

```bash
python train_example_v16_optimal_recon.py \
    --data_dir data/hsi_v8-80_10_10_importance-new/rgb \
    --batch_size 256 --epochs 20 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --seed 42
```

One argument differs from the reference run: `--data_dir`. The HSI arm already exists
(`20260914_181738…`, test 96.22 % acc / 94.31 % balanced / 0.9003 macro-F1, 3.66 h) — **do not re-run
it.**

Three decisions to make deliberately, because each is a way to accidentally add a second variable:

* **Keep `--lambda_sam 0.1` on the RGB arm.** Spectral angle over three broad channels is a weaker
  quantity than over 32 narrow ones, and the entry point's own docstring suggests 0.0 for RGB — but
  changing it makes the pair differ in *two* arguments and forfeits the single-variable claim that is
  the entire point of §7.2. Keep it identical and note the interpretation in the paper.
* **`--spectral_checkpointing auto` engages at C ≥ 16**, so the HSI arm pays recompute and the RGB arm
  does not. That is the §7.6 finding, not a band-count effect; report the time and memory columns with
  that sentence attached, or the cost comparison will be misread again.
* **Parameter counts will not match** (575,849 vs 559,116) because of F4. Report
  `backbone_num_params` (446,409 both) for the RQ1 claim and the wrapped totals for the cost table,
  labelled.

**What it produces that the current pair cannot:** reconstruction metrics and images on both arms,
`epoch_time`/`train_time_s`/`val_time_s`/`s_per_step` on both arms in the same schema, and a clean
statement of which RGB rendering was used.

### E2 — Recursion-depth quality sweep

**Closes:** F6, the paper's own priority #2, and §8.2's open question against the TRM literature.
**Cost: ≈ 5.2 h for three arms.**

```bash
for T in 1 2 4; do
  python train_example_v16_optimal_recon.py \
      --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
      --batch_size 256 --epochs 12 \
      --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
      --trm_n_improve $T --seed 42
done
```

| `--trm_n_improve` | core applications | est. GFLOPs | est. s/epoch | 12 epochs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21 | 2.17 | ≈ 246 | **0.82 h** |
| 2 | 42 | 4.18 | ≈ 450 | **1.50 h** |
| 3 *(reference, done)* | 63 | 6.19 | 653 *(measured)* | 2.20 h |
| 4 | 84 | 8.20 | ≈ 857 | **2.86 h** |

Hold `trm_deep_supervision_steps` at 3 throughout (see F6). Tabulate with
`python scripts/compare_runs_v16.py 'experiments/*sup3*' --per_patient` and plot macro-F1 against
measured wall clock — the negative result (flat quality, linear cost) would be as publishable as a
positive one, and is what §8.2 promises the base literature.

### E3 — Controlled backbone match, on the same task

**Closes:** F2, the paper's priority #3, and turns §3.6's *(prior)* rows into *(verified)* ones.
**Cost: ≈ 1.5 h for two arms** *(estimated — measure it; the hierarchical arm has never been timed
on this data).*

```bash
for ARCH in split efficient; do
  python train_example_v16_optimal_recon.py \
      --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
      --batch_size 256 --epochs 12 \
      --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
      --architecture $ARCH --seed 42
done
```

Both arms work through this entry point: `build_model_optimal`
(`train_example_v16_optimal.py:919`) drops the inapplicable `--classifier_dropout` with a printed
warning instead of raising, and it is wired in as `model_fn` through
`train_example_v16_optimal_recon.build_overrides`. Plain `train_example_v16.py` would raise — use
the optimal-recon entry point, as above.

This is the comparison that makes RQ2 an architecture claim rather than a capacity claim: 446,409
against 2,773,007 on identical data, identical recipe, identical test patches, both carrying
reconstruction artefacts. Pair it with P1 so the FLOP and latency columns are measured on the same
two models.

### E4 — Reconstruction ablation

**Closes:** §7.7's explicit boundary — *"no part of the classification result can be attributed to
this objective"* — and the paper's priority #6. **Cost: ≈ 2.2 h.**

```bash
python train_example_v16_optimal_recon.py \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --recon_mode none --seed 42
```

`--recon_mode none` removes the decoder entirely (back to 446,409 parameters) and is the correct
control. Do **not** use `--recon_mode latent --lambda_mse 0 --lambda_sam 0`: the entry point warns
about it for good reason — the decoder would exist, receive no gradient, and write meaningless
metrics.

This is the one arm in the plan that by construction produces no reconstruction images. That is the
experiment, not an omission; `reconstruction/status.json` records the skip.

### E5 — MedMamba baseline on the same training distribution

**Closes:** F5, §7.3's stated largest caveat. **Cost: ≈ 1.4 h** (5 epochs × ~1,030 s).

```bash
cd ../medmamba-original/MedMamba
python train_hsi.py \
    --data /data/dante_data/.../data/hsi_v8-80_10_10_importance-new/hsi \
    --epochs 5 --batch_size 256 --lr 1e-4 --amp bf16 \
    --class_weights --normalize global_zscore \
    --checkpoint_metric f1_macro --patch_size 1 \
    --dims 64,128,256,512 --depths 1,1,2,1 --seed 42 \
    --run_name matched_natural_v8new
```

Two edits to that (unfrozen) script first: add `global_zscore` to `--normalize`'s choices so the
normalization matches, and a `--train_subsample_frac` if you would rather match *samples seen* than
epochs. Match on samples seen if you can — the reference run sees 250,113 × 20 ≈ 5.0 M patch
presentations, so ~2 full epochs of the natural build; 5 epochs gives MedMamba 2.4× that, which is
generous rather than handicapping and should be stated as such.

Matched after this run: training-split build, class balance, normalization, batch size, precision,
seed, learning rate, checkpoint rule, GPU, test set. Still unmatched and still worth naming: EMA,
augmentation, the auxiliary objective, and the subsample policy.

### E6 — Wavelength vs. index positional encoding

**Closes:** RQ1's encoding half, untested since v15 built the mechanism. **Cost: ≈ 2.2 h.**

```bash
python train_example_v16_optimal_recon.py \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --no_use_wavelengths --seed 42
```

`--no_use_wavelengths` (`train_example_v15.py:449`) is the true index-encoding arm.
`--wavelength_scale None` is **not** — it sets the angular span to the band count, which reduces to
the index encoding only for a *uniformly spaced* sensor, and this one is emphatically not: eighteen
bands at or below 633.3 nm, then a **219.0 nm gap**, then fourteen from 852.3–938.2 nm. That gap is
the whole reason the encoding claim is interesting, and it is why this ablation has a real chance of
showing a difference rather than a null.

### E7 — Seed replication of the RQ0 pair

**Closes:** §9.1's binding limitation, the paper's priority #1. **Cost: ≈ 8.2 h — run overnight.**

```bash
for S in 1 7; do
  for D in hsi rgb; do
    python train_example_v16_optimal_recon.py \
        --data_dir data/hsi_v8-80_10_10_importance-new/$D \
        --batch_size 256 --epochs 20 \
        --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
        --seed $S
  done
done
```

Two extra seeds, not five. With the 0.44-point nondeterminism floor from F3c already in hand, three
seeds per arm is enough to say whether a 15-point margin is stable; five would cost 20 h for a
precision the five-patient test split cannot support anyway. **Run E1 first** — replicating the
wrong RGB build five times is the one way to make this worse.

### E8 — Zero-shot band decimation on the existing checkpoint

**Closes:** RQ1's behavioural half, at the lowest cost in the plan. **Cost: ≈ 10 min, no training.**

```bash
python scripts/eval_band_decimation_v16.py \
    --run_dir experiments/20260914_181738…_optimal_recon \
    --keep_bands 32,16,8,4,2
```

Needs P3. This asks the question the architecture was built to make askable: a model whose parameter
set does not know how many bands it is reading should degrade gracefully when bands are removed *at
test time*. It is the difference between §7.1's structural result — arithmetic, and the paper says so
— and a behavioural one.

Report it as a curve with the retrained arms absent and flagged: zero-shot transfer and retraining at
C bands are different claims, and only the first is nearly free. If the curve is flat to 16 bands,
that is a filter-wheel-camera result and is worth more than the 32-vs-3 comparison it sits next to.

---

## 4. Order and budget

| # | Experiment | GPU | Closes | Manuscript effect |
| --- | --- | ---: | --- | --- |
| P1 | FLOPs script | — | F3a | §7.6 panel *(prior)* → *(verified)* |
| P2 | Latent-space script | — | F3b | new qualitative figure; retro-fits all runs |
| P3 | Band-decimation script | — | F7 | enables E8 |
| E1 | **RGB twin, corrected build** | **1.9 h** | F1 | §7.2 becomes what §6 claims it is |
| E3 | Backbone match (`split`, `efficient`) | 1.5 h | F2 | RQ2 becomes architectural; §3.6 verified |
| E5 | MedMamba, matched distribution | 1.4 h | F5 | §7.3's largest caveat removed |
| E8 | Zero-shot band decimation | 0.2 h | F7 | RQ1 behavioural half opened |
| E2 | Depth sweep (21 / 42 / 84) | 5.2 h | F6 | §7.8's "most obvious missing experiment" |
| E4 | Reconstruction ablation | 2.2 h | — | §7.7 attributable |
| E6 | Encoding ablation | 2.2 h | — | RQ1 encoding half |
| E7 | Seeds 1, 7 on the E1 pair | 8.2 h | — | §9.1's binding limitation |
| | **Total** | **≈ 22.8 h** | | |

Roughly two overnight blocks plus a working day. The first four GPU rows — **5.0 h total** — close
the three defects and the two largest caveats; if only one day is available, stop there.

**Do E1 before E7.** Everything else is independent and can be reordered freely.

---

## 5. Reporting contract

Every run above must land a directory matching
`experiments/20260914_181738…_optimal_recon` field for field. That shape is produced automatically by
`train_example_v16_optimal_recon.py`; what follows is the checklist for confirming it did, plus the
two additions this plan introduces.

**Classification and provenance** — `config.json` (with `backbone_num_params`, full `cli_args`,
`scheduler`, `gate_g7`, `recon_artifacts`), `test_report.json`, `gates.json`,
`checkpoint_reproducibility.json`, `leakage_report.json`, `split_drift_report.json`,
`class_imbalance_report.json`, `dataset_{integrity,split}_report.json`, `per_patient_metrics.json`,
`per_capture_metrics.csv`, `classification_reports/`, `confusion_matrix/`, `predictions/`,
`confidence_analysis/`.

**Reconstruction — metrics and images** (`reconstruction/`): `status.json`, `provenance.json`,
`samples_metrics.json`, `samples.npz`, `metrics_curves.png`,
`reconstruction_{best,median,worst,samples}.png`, `hist_{rmse,mae,sam_deg,sid,peak_position_error}.png`,
`samples/` (3 per class at even SAM quantiles). Per-epoch spectral metrics land in `history.csv`
columns 38–48 (`spectral_{rmse,mae,sam_deg,sid,pearson_correlation,cosine_similarity,peak_position_error,psnr,ssim}`).

> Keep `provenance.json`'s own warning in front of you when writing the paper: `ssim2d` is windowed
> 2-D spatial SSIM; `ssim_spectral` is per-pixel across the band axis. They are different quantities.
> §7.7 gets this right today — do not lose it when adding rows.

**Run time — required on every row.** `history.csv` already carries `epoch_time`, `train_time_s`,
`val_time_s`, `artifact_time_s`, `data_wait_s`, `s_per_step` and `GPU_memory_MB` per epoch, and
`reconstruction/status.json` carries `elapsed_s` for the export. Report, for every arm:

| Field | Source | Reference run |
| --- | --- | ---: |
| Total wall clock | Σ `history.csv:epoch_time` | 13,174.6 s = **3.66 h** |
| Median s/epoch | `history.csv:epoch_time` | 653.1 s |
| Median s/step | `history.csv:s_per_step` | 0.64 s |
| Train / val split | `train_time_s` / `val_time_s` | 626.9 / 25.8 s |
| Peak training VRAM | `history.csv:GPU_memory_MB` | 843 MB |
| Time to selected epoch | Σ `epoch_time` to `best_epoch` | epoch 7 = **1.30 h** |
| Inference latency / throughput | `test_report.json:efficiency` | 13.20 ms bs1, 1,159 patch/s |
| Recon export | `reconstruction/status.json:elapsed_s` | 29.5 s |

Two columns the pre-2026-09-13 runs cannot supply in this schema — their `history.csv` headers are
space-padded (`" epoch_time"`), so any comparison script must strip keys before lookup. `history.json`
is the safer source for those runs.

**Additions from this plan:** `flops_report.json` (P1) and a populated `latent_space/` (P2) in every
run directory, including retro-fitted older ones.

**Tabulation:** `python scripts/compare_runs_v16.py 'experiments/<glob>' --per_patient --csv out.csv`.
Per-patient is not optional on the histology corpus — the test split is 348,894 patches from **five
people**, and per-patient macro recall on the existing runs spans 0.64 to 1.00.

---

## 6. Claims to correct in the manuscript once these land

Independent of any run, from §1's findings:

1. **Line 403** — "…and it is what §7.2 uses" is not supported by
   `data/hsi_v8-80_10_10_importance/prep_log.txt`. Either re-point §7.2 at E1, or state that the
   reported pair used the legacy source and that the corrected pair is E1. (F1)
2. **Lines 15, 60, 271, 664, 754, 816** — "≈ 61× … on the same task" pairs a recursive HSI model with
   a hierarchical RGB one. On HSI the ratio is **6.2×** (446,409 vs 2,773,007); 61× is the PAD-RGB
   ratio (446,796 vs 27,425,314). Attach each to its task. §3.6's table is already right. (F2)
3. **§3.6** — the 2.77 M and 27.43 M rows are now *(verified)*, not *(prior)*, and `fullchannel`
   (3,513,451) and `efficient` (3,685,483) can be added. (F2)
4. **§9.1 / doc 12 §B.3** — the "second RGB seed" is seed 42 twice. Report it as a
   **nondeterminism replicate with a 0.44-point spread**, which strengthens RQ0 rather than weakening
   it, and note that seed variance remains unmeasured until E7. (F3c)
5. **§7.1** — when reconstruction is on, quote `backbone_num_params`, not
   `test_report.json:efficiency.num_params`; the decoder's output convolution is linear in band
   count, so the wrapped model is *not* band-count-agnostic and the boundary should be stated. (F4)
6. **§7.6 / §7.8** — every FLOP figure is currently *(prior)* because the counter has never run
   (`trainerg_v11.py:717`). P1 fixes that for all runs retroactively. (F3a)

---

## 7. Notes for whoever runs this

* **Nothing frozen is edited.** P1–P3 are new standalone scripts; every experiment is an existing
  entry point with different flags. The one file that *is* edited is
  `medmamba-original/MedMamba/train_hsi.py` (E5), which is in the other repository and is not frozen.
* **Disk.** 531 GB free on `/data`; no new dataset preparation is required by any experiment here.
  That is deliberate — §G's four-build band sweep (~7.2 h prep, ~43 GB) is replaced by E8 plus,
  optionally, retrained arms later.
* **Directory names still do not identify contents.** Appendix A's warning applies to everything
  produced here; establish identity from `config.json` and record the mapping as you go.
* **`--seed 42` and `deterministic: false`.** Two identical commands differ by ~0.44 points. Report
  a margin as real only if it clears that, and never re-run an arm expecting a bitwise match.
