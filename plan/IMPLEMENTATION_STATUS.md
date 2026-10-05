# GMedMamba Improvement Plan — Implementation Status

> **Superseded.** Moved from the repository root by v17 S12. This is the Stage-A status
> report for the `trainerg_v4`/`train_example_v6` era; the current plans are
> `plan/GMedMamba v15/v16/v17*`. Kept because
> `training/run_full_ablation_checklist.py:28` cites its Stage C/D sections.

This covers **Stage A ("Correctness")** from the plan's own §51 staging —
the stage the plan itself says to complete before judging architectural
performance. It's delivered as new files that layer on top of your existing
code (`gmedmamba.py`, `trainerg_v4.py`, `train_example_v6.py`,
`prepare_histologyhsi_bc_v4.py`) **without editing any of them**, matching
the pattern your repo already uses in `training/grad_checkpoint.py`
(wrap/patch at runtime, keep the architecture file untouched).

Stages B–G (classification-strategy experiments, the full architecture
redesign, efficiency work, generalization experiments, and the
MedMamba/multi-seed scientific validation) are **not** implemented here —
see "What's not done" below for why and what each would need.

## New files

| File | Plan phase(s) |
|---|---|
| `training/class_coverage.py` | Phase 3 (class coverage), Phase 11 (breakdown) |
| `training/gradient_health.py` | Phase 16 (instability counters), Phase 17 (invalid epoch) |
| `training/activation_patch.py` | Phase 19–21 (LeakyReLU) |
| `training/reconstruction_head.py` | Phase 34 (latent reconstruction), Phase 35 (auxiliary loss), Phase 39 (spectral dropout) |
| `training/trainerg_v5.py` | Phase 13 (macro-F1 selection), Phase 15 (class collapse), Phase 16/17 integration |
| `training/hsi_quality_check.py` | Phase 5 (HSI spectral QC) |
| `train_example_v7.py` | wires all of the above into one entry point |

## What each fixes, concretely

- **Phase 13 (critical bug)** — `TrainerG_v4.fit()` picks the "best"
  checkpoint via `val_accuracy > best_val_acc`. On an imbalanced set this is
  exactly the failure mode Phase 10 describes (36.73% accuracy / 21.01%
  balanced accuracy / 3-of-6 classes at zero recall) — a majority-class
  predictor can still "win" on accuracy. `TrainerG_v5` selects on
  `--checkpoint_metric` (default `f1_macro`) instead.

- **Phase 17 (critical bug)** — `TrainerG_v4` aborts the *entire run* on
  the first NaN/Inf training loss (`self.is_stopped = True`, loop breaks).
  `TrainerG_v5` skips only the offending batch and marks the *epoch*
  invalid only if every batch in it failed — invalid epochs are logged,
  excluded from best-checkpoint selection, but training continues.

- **Phase 34 (architectural bug)** — both existing reconstruction wrappers
  (`GMedMambaReconWrapper`, `GMedMambaGANWrapper`) run `self.recon_head(x)`
  directly on the raw input, not on anything the backbone computed. The
  "reconstruction loss" therefore can't actually train the encoder to
  produce a better shared representation — it only trains a decoder that
  never talks to the backbone. `GMedMambaLatentReconWrapper` decodes from
  `backbone.forward_features(x)["feature_map"]` instead, so
  `lambda_mse`/`lambda_sam` gradients now flow into the encoder.

- **Phase 3** — nothing currently stops training on a split that's missing
  a class outright; you'd only discover it via `sanity_check_metrics`
  warnings *after* wasting a training run. `verify_class_coverage()` fails
  fast, before the first batch.

- **Phase 15/16** — `TrainerG_v4` has per-metric constant-streak and
  NaN/Inf-*metric* detection (`eval_utils.sanity_check_metrics`,
  `metric_validation.MetricValidator`), but nothing tracks *optimizer-step*
  health (`skip_ratio`, `nonfinite_gradients`, …) or flags a specific class
  recall sitting at exactly 0 for several epochs running. Added.

## How to validate the fixes (Phase 44-style A/B)

```bash
# Old behavior (Phase-34-broken reconstruction, accuracy-based selection, ReLU):
python train_example_v7.py --data_dir ./data/hsi --epochs 30 \
    --recon_mode raw_input --checkpoint_metric accuracy --activation relu

# Stage-A-fixed behavior (all defaults):
python train_example_v7.py --data_dir ./data/hsi --epochs 30
```

Compare `experiment_report.json`'s `best_val_accuracy`/history against
`spectral_rmse`/`spectral_sam_deg` trends and per-class recall in both runs.
`run_ablations_v2.py` can be extended with a `stageA` axis the same way its
existing `reconstruction`/`gan`/`loss_weights` axes are defined, if you want
this wired into the existing sweep tooling.

## What's not done, and why

The remaining phases are genuinely large, separable pieces of work; doing
them without dedicated review/testing would risk silently degrading a
codebase that already has substantial working infrastructure (dataset
integrity checks, spectral metrics, GAN training, band selection, ablation
sweeps — all already present and left untouched here).

- **Phase 7–9 (configurable normalization / train-only augmentation)** —
  no augmentation code exists anywhere in the repo yet (spectral
  noise/scaling/offset/band-dropout/masking, spatial flips/rotations). This
  is a full new module plus a Dataset-level integration point in
  `NpyDataset.__getitem__`, needed before Stage B's classification-fix
  experiments (weighted CE, focal loss, balanced sampling) are meaningful.
- **Phase 22–33 (remove the hard spectral/spatial channel split)** — this
  is the plan's own "primary architectural experiment": `GBlock` currently
  splits `hidden_dim` in half, running `SS2D` on one half and a conv branch
  on the other (`gmedmamba.py`'s `GBlock.forward`, `left, right =
  x.chunk(2, dim=-1)`). Replacing that with the proposed full-channel
  dual-branch-with-cross-interaction design changes the model's parameter
  count and shapes throughout `GStage`/`GMedMambaBackbone`, so it needs
  its own file (e.g. `gmedmamba_fullchannel.py`) and a real training run to
  validate before it can replace anything — not something to hand you
  untested.
- **Phase 40–41 (parameter targets), Phase 45–47 (MedMamba comparison,
  multi-seed)** — these are experiment-design/compute tasks, not code the
  repo is missing; `run_ablations_v2.py` already has the right shape to
  extend for this once Stage C exists to sweep over.
- **Phase 4 preprocessing_report.json/.txt (full field list) and Phase 5's
  integration into `prepare_histologyhsi_bc_v4.py` itself** —
  `training/hsi_quality_check.py` covers Phase 5's checks standalone (run
  it against `X_train.npy` after prep), but wasn't spliced into the prep
  script's own `dataset_statistics.json` to avoid editing that large,
  already-tested file blind.

---

## Stage B ("Classification") — done

Items 12–14 (macro-F1 checkpointing, per-class monitoring, class-collapse
detection) were already covered by Stage A's `trainerg_v5.py`. This adds
items 8–11:

| File | Plan phase(s) |
|---|---|
| `training/class_imbalance.py` | Phase 11 (class_count/percentage per split) |
| `training/losses.py` | Phase 12.B (weighted CE), Phase 12.C (focal loss) |
| `training/samplers.py` | Phase 12.D (balanced sampler), Phase 12.E (moderate oversampling) |
| `training/trainerg_v6.py` | plugs a configurable criterion into TrainerG_v5's train/val loops |
| `train_example_v8.py` | `--loss {ce,weighted_ce,focal,focal_weighted}`, `--sampler {none,balanced,moderate_oversample}` |
| `run_classification_ablation.py` | Phase 12's required controlled A–E comparison |

**Phase 12 is explicit that these should be tested one at a time, not
stacked** ("Do not combine all strategies immediately. Identify which
intervention actually solves class collapse.") — `train_example_v8.py`
exposes `--loss` and `--sampler` as single-choice flags for exactly that
reason; combining e.g. `--loss weighted_ce --sampler balanced` is possible
but is then *your* deliberate second-order experiment, not something
either script does implicitly.

Run the full Phase 12 comparison:

```bash
python run_classification_ablation.py --data_dir ./data/hsi \
    --out_dir ./ablations_classification --epochs 20
```

This runs options A–E as five separate `train_example_v8.py` subprocesses
(same pattern as `run_ablations_v2.py`) and writes
`classification_ablation_results.csv` with, per option, `final_f1_macro`,
`final_balanced_accuracy`, and `n_zero_recall_classes_final_epoch` — the
last one is the most direct evidence Phase 10's class-collapse problem
(three of six classes at zero recall) is actually fixed rather than just
statistically improved on average.

---

## Stage C ("Architecture") — items 15–20

Item 15 (LeakyReLU) was already done in Stage A. Items 16–20 map to the
plan's Phases 22–27 ("remove the hard channel split" / full-channel
spectral-spatial block / cross-branch interaction / lightweight fusion).

| File | Plan phase(s) |
|---|---|
| `gmedmamba_fullchannel.py` | Phase 22–27 (full-channel dual-branch block), Phase 24-C (depthwise+pointwise, not dense Conv2d) |
| `train_example_v9.py` | `--architecture {split, fullchannel}` on top of Stage A+B |

**Read `gmedmamba_fullchannel.py`'s module docstring before using it** — the
plan describes removing a spectral-vs-spatial channel split
("C spectral channels + 1 spatial channel") that does not literally exist
in this codebase: `SpectralPathway` already processes every wavelength band
densely and fuses into the backbone per-stage via a pluggable fusion
strategy, which already satisfies Phase 22's stated goal. The one hard
50/50 split that does exist is inside `GBlock`, between its two *spatial*
operators (a long-range `SS2D` scan and a local conv branch — a MedMamba
"SS-Conv-SSM" design choice, not a spectral/spatial split). `FullChannelGBlock`
removes that split: both operators now see the full channel width, combined
via lightweight cross-branch interaction (Phase 27) and a learned 2-way
gate (Phase 26) instead of concatenation + `channel_shuffle`. If your intent
was instead to change `SpectralPathway`'s channel allocation, no change is
needed there — it already does full-channel spectral processing.

Built as a subclass hierarchy (`FullChannelGStage(GStage)`,
`FullChannelGMedMambaBackbone(GMedMambaBackbone)`, `GMedMambaFullChannel(GMedMamba)`)
that overrides only `__init__` to swap the block type — every `forward()`
above `FullChannelGBlock` is inherited **unchanged**, so it's a fully
compatible drop-in: `base_model.cfg`, `base_model.backbone.forward_features(...)`,
and `base_model.head(...)` all exist with the same shapes, meaning every
Stage A/B wrapper works with it untouched. Verified concretely, not just by
inspection:

```
$ python gmedmamba_fullchannel.py
  original      params=2.770M  out=(2, 6)
  full-channel  params=3.510M  out=(2, 6)
  loss=1.6943  params_without_grad=0
  keys: ['band_weights', 'feature_map', 'pooled', 'spectral_context', 'stage_ctx_maps', 'stage_feature_maps', 'stage_pools']
  All smoke tests passed.
```

...and an end-to-end integration check composing it with Stage A's
LeakyReLU patch (0 ReLUs found to replace — correct, `FullChannelGBlock`
already uses `LeakyReLU` internally, so there are zero `nn.ReLU` modules
left anywhere in the full-channel architecture) and latent-reconstruction
wrapper: forward + backward pass, zero missing gradients.

Run the A/B yourself:

```bash
python train_example_v9.py --data_dir ./data/hsi --epochs 30 --architecture split        # baseline
python train_example_v9.py --data_dir ./data/hsi --epochs 30 --architecture fullchannel  # Phase 22-27
```

Compare `backbone_num_params` in each run's `config.json` alongside
`final_f1_macro`/`final_balanced_accuracy` — per Phase 46, a param-count
difference alone doesn't tell you the architecture change helped; you need
both numbers side by side.

---

## Stage D ("Efficiency") — items 21–26

| File | Plan phase(s) |
|---|---|
| `training/compact_head.py` | Phase 25 (compact classifier), Phase 33 (no large concatenation) |
| `training/lightweight_attention.py` | Phase 24 (ECA/SE-gate/no-attention fusion options) |
| `training/capacity_search.py` | Phase 26/40/41 (parameter-target sweep vs. the plan's 27.42M baseline) |
| `gmedmamba_efficient.py` | wires the above onto Stage C's full-channel backbone |
| `train_example_v10.py` | `--architecture {split,fullchannel,efficient}`, `--fusion_type` |

Item 21 ("replace dense Conv3D") is N/A — this codebase never had a dense
Conv3D spectral branch to begin with (`FullChannelGBlock`'s local branch
was already depthwise+pointwise from Stage C). Item 24's "full attention vs
lightweight vs no attention" comparison point is `--fusion_type
{film|gated|cross_attention|multiplicative|residual|se_gate|eca|none}` —
same reinterpretation caveat as Stage C applies (see
`training/lightweight_attention.py`'s docstring: this codebase's spectral
inter-channel attention was already lightweight; the genuinely "full
attention" option lives at the fusion slot, not inside the spectral
pathway). Items 22 (spectral width) and 23 (spectral blocks per stage) are
directly configurable on the existing `GMedMambaConfig` via
`d_token`/`compression_dims` and `spectral_depth` — no new class needed;
`gmedmamba_efficient.py`'s `scale_spectral_width()` helper does the Phase
22 100/75/50/25% grid.

`gmedmamba_efficient.py` is built via a small refactor of
`gmedmamba_fullchannel.py` (added `_fusion_factory`/`_block_factory`/
`_stage_factory`/`_backbone_factory`/`_classification_head_factory`
extension hooks — each Stage D class overrides exactly one hook and
inherits everything else). Actually executed, not just inspected:

```
$ python gmedmamba_efficient.py
=== Param comparison: split (Stage A/B baseline) vs fullchannel (Stage C) vs efficient (Stage D) ===
  split                params=2.770M  out=(2, 6)
  fullchannel          params=3.510M  out=(2, 6)
  efficient/se_gate    params=3.446M  out=(2, 6)

=== Fusion-type sweep on the efficient architecture (Phase 24) ===
  fusion_type=se_gate    params=3.446M
  fusion_type=eca        params=3.402M
  fusion_type=none       params=3.344M
  fusion_type=film       params=3.461M

=== Backward pass / gradient-flow check (efficient, se_gate) ===
  loss=1.4931  params_without_grad=0

=== Phase 22 spectral-width scaling ===
  fraction=1.0   d_token=32 compression_dims=(128, 64) params=3.461M
  fraction=0.75  d_token=24 compression_dims=(96, 48) params=3.447M
  fraction=0.5   d_token=16 compression_dims=(64, 32) params=3.436M
  fraction=0.25  d_token=8 compression_dims=(32, 16) params=3.430M
All smoke tests passed.
```

The compact head's savings are small in this tiny-config smoke test
(pooled_dim there is already small); they scale up substantially on the
plan's actual ~27.42M `gmedmamba_tiny`-sized configs, where
`ClassificationHead`'s `sum(dims)+d_ctx -> half -> num_classes` MLP is a
much larger fraction of total parameters.

`training/capacity_search.py` was also run for real:

```
$ python -m training.capacity_search
Baseline (plan's reported GMedMamba-Tiny): 27.42M params
  target_20.0M: None
  target_17.0M: None
  target_15.0M: {'d_token': 12, 'depths': (1, 1, 2, 1), 'params_m': 15.119}
```

`None` for 20M/17M is an honest result, not a bug — the default grid
(scaling `d_token`/depth only, holding `dims` fixed at the plan's own
27.42M baseline shape) doesn't densely cover that range; widen
`d_token_choices`/`depth_scale_choices`, or also scale `dims` and pass
`model_cls=GMedMambaEfficient` (compact head + lightweight fusion already
save several M) if you need tighter coverage — the function is designed to
be re-run with different grids per Phase 26's "search," not to hit every
target from one default call.

---

## Stage E ("Generalization") — items 27–33

| File | Plan phase(s) |
|---|---|
| `training/augmentation.py` | Phase 9 (spectral + spatial augmentation), Phase 8 (train-only) |
| `training/config_presets.py` | plumbing so `drop_path_rate`/`classifier_dropout` overrides actually reach the model |
| `training/trainerg_v7.py` | Phase 38 (patience-based early stopping on macro-F1) |
| `train_example_v11.py` | `--augment_preset`, `--weight_decay`, `--drop_path_rate`, `--classifier_dropout`, `--early_stop_patience` |

**Phase 9 was the single biggest actual gap in this codebase** — no
augmentation of any kind existed anywhere before this. `training/augmentation.py`
implements all of it: spectral noise/scale/offset, scattered band dropout,
contiguous spectral-region masking, plus spatial flips/90°-rotation/crop-
rescale, composed via `AugmentationConfig` (every knob defaults to off) and
three presets (`none`/`light`/`medium`). `AugmentedPatchDataset` wraps a
dataset generically — `train_example_v11.py` wraps only `train_ds`, and
`val_ds` is built and passed to the val `DataLoader` completely separately,
so Phase 8's "no validation/test sample should be altered by training
augmentation" is structural, not just a documented intention. Read the
module docstring for one honest caveat: with `num_workers > 0`, forked
worker processes each get their own copy of the shared augmentation RNG
seed (a diversity nit, not a correctness bug — labels are never touched).

Two overrides genuinely didn't reach the model before this stage, not just
weren't exposed on the CLI: `drop_path_rate` (Phase 32) and classifier
dropout (Phase 31) — the convenience constructors (`gmedmamba_hsi_small`,
etc.) build `GMedMambaConfig` internally and only forward `**kwargs` to the
model class's `task`/`num_classes`, with no path to the config object
itself. `training/config_presets.py` fixes this by building the same
preset configs as data you can override before construction — verified for
real, not just by inspection:

```
default drop_path_rate: 0.1
override drop_path_rate: 0.3
classifier dropout p: 0.5
correctly rejected classifier_dropout on split arch: --classifier_dropout only applies to --architecture efficient (gmedmam...
forward pass OK: (2, 4)
ALL config_presets CHECKS PASSED
```

Phase 30 (weight decay) was already wired into every AdamW optimizer call
in every prior entry point, just hardcoded at `0.05` instead of a CLI flag
— `--weight_decay` fixes that. Phase 37's other knobs (dropout, drop path)
were likewise already present in `GMedMambaConfig`/`ClassificationHead`
internals, just not reachable — same fix.

Phase 38's `TrainerG_v7` keeps `TrainerG_v4`'s inherited divergence/stall
heuristics as a safety net and adds real patience-based stopping on
`self.checkpoint_metric` (the SAME metric Stage A's Phase 13 fix already
uses for best-checkpoint selection, so "what's checkpointed" and "what
early-stopping watches" can never silently disagree). It reports a stop
reason string (`EARLY_STOPPED_PATIENCE_F1_MACRO`) through a tiny `str`
subclass (`_StopReasonStr`) rather than editing `trainerg_v4.py`'s closed
`StopReason` `Enum` to add a member — verified that `.value` access (the
only thing any downstream code reads) works transparently, and the
patience/reset arithmetic was verified against the exact comparison logic
used in the real method.

---

## Stage F ("Scientific Validation") — items 34–42

| File | Plan phase(s) |
|---|---|
| `training/extra_metrics.py` + `training/trainerg_v8.py` | Phase 48 — fills a real gap: Cohen's Kappa/MCC weren't computed anywhere in the live per-epoch training loop |
| `training/flops_counter.py` | Phase 0/48 efficiency benchmark gap — no FLOPs/MACs measurement existed anywhere |
| `run_multiseed.py` | Phase 47 (5+ seeds, mean±std) |
| `training/capacity_search.py`'s new `find_param_matched_configs()` | Phase 46 (parameter-matched cross-architecture comparison) |
| `run_full_ablation_checklist.py` | Phase 44's explicit 11-item on/off checklist |
| `training/aggregate_report.py` | Phase 49 (parameters-vs-metric thesis figure, cross-experiment summary table) |
| `train_example_v11.py` (updated) | now uses `TrainerG_v8` so Cohen's Kappa/MCC appear by default |

**Two genuine correctness/coverage gaps found and fixed, not just new
orchestration:** `training/eval_utils.py` (what the live trainer calls
every epoch) never computed Cohen's Kappa or MCC — `training/metrics.py`
has both, but that's a completely separate module only used by the
standalone `evaluate.py` path. `TrainerG_v8` closes this with a one-method
override (confirmed additive, not a duplicate validation loop) and
`train_example_v11.py` now uses it. Verified against real predictions:

```
{'cohen_kappa': 0.6969..., 'matthews_corrcoef': 0.6969...}
```

Likewise, nothing in this repo measured FLOPs/MACs before
`training/flops_counter.py` — run against the real `gmedmamba_hsi_small`
model:

```
conv_linear_macs: 126,305,920
conv_linear_flops_approx: 252,611,840
by layer type: {'Conv1d': 44605440, 'Conv2d': 16178816, 'Linear': 65521664}
selective_scan_flops_counted: False
estimated selective-scan FLOPs (one SS2D call, 11x11 patch): 1,982,464
```

Read the module docstring: this deliberately does NOT claim to count the
custom selective-scan einsum ops (the actual core Mamba compute) — doing so
silently and calling the result "total FLOPs" would be exactly the kind of
partial-number-presented-as-complete that `training/spectral_metrics.py`
already refuses to do elsewhere in this repo. `estimate_selective_scan_flops()`
gives the analytical formula instead, so nothing is hidden.

`training/config_presets.py` also gained a `dynamic_band_selection`
override (Phase 44's "spectral gating on/off" checklist item) — verified
both settings actually reach the config and produce working forward passes.

`run_full_ablation_checklist.py` operationalizes Phase 44's literal
11-item checklist as one-flag-at-a-time `train_example_v11.py` runs. Two
items are honestly reported as not fully available rather than faked:
**spectral gating** (the new `dynamic_band_selection` toggle isn't yet
wired to a CLI flag — a quick follow-up, not a design gap) and
**cross-branch fusion on/off** (`FullChannelGBlock` applies Phase 27's
interaction unconditionally; toggling it needs one more small variant class
using the same `_block_factory` hook pattern already established in Stage
C/D — not built here to keep this response scoped).

`run_multiseed.py` (Phase 47) and `training/aggregate_report.py` (Phase 49)
were both run against real/synthetic data end-to-end — the aggregate report
was verified against three synthetic experiment dirs (CSV row count and
both PNG plots confirmed to exist and match input data), since running
actual multi-epoch training here isn't practical without a real dataset.
`training/capacity_search.py`'s `find_param_matched_configs()` was run for
real across `GMedMamba`/`GMedMambaFullChannel`/`GMedMambaEfficient` at a
4M-parameter target — `split` correctly returned `None` (its default grid
tops out below that target at this tiny config size) rather than a
fabricated match, while `fullchannel`/`efficient` found real matches within
tolerance.

---

## Overall status

**Stages A–F are all implemented** (with the two Phase-44 checklist items
above flagged as partial). **Stage G (frozen MedMamba transfer) remains
explicitly deferred**, per the plan's own instruction — nothing here
touches it.

What would still be worth doing, roughly in priority order, if you want to
keep going:
1. Wire `dynamic_band_selection` and a `--no_cross_branch_interaction`
   architecture variant into the CLI/checklist (closes Stage F's two
   partial items).
2. Actually run the pipeline end-to-end on real data — everything here has
   been verified with real forward/backward passes, synthetic-data
   integration tests, and (where training wasn't practical in this
   environment) exact-logic unit tests, but no multi-epoch training run on
   an actual HistologyHSI-BC-Recurrence-derived dataset has happened. That's
   the real next step before any of Phase 44-49's actual scientific
   conclusions can be drawn — this plan produces the instrumented pipeline
   to run those experiments with, not the experimental results themselves.
3. If FLOPs must include the selective-scan cost precisely (not the
   analytical estimate), instrument `_selective_scan_pure_pytorch` directly
   with a per-call counter rather than extending the hook-based approach.


---

## Post-Stage-F fix: dataset preparation (a real bug + a second dataset)

Reported failure:

```
ValueError: Class coverage check failed (Main Development Plan, Phase 3) -
at least one split has ZERO samples for a required class:
  - validation: missing ['0', '2']
```

**Root cause**: `prepare_histologyhsi_bc_v4.py`'s `patient_split_three()`
does a single global shuffle of ALL patients, then slices off the first N
as test/val - patient-level (Phase 2, correct), but NOT stratified by
class. When a class has a small pool of distinct patients, plain random
slicing can put every one of them into `train`, leaving validation with
zero samples of that class. The Phase-3 coverage check (built in Stage A)
correctly *caught* this - but catching it isn't fixing it, and it was only
catching it at TRAIN time, after the (potentially very long) prep run had
already finished.

### The fix

| File | What it does |
|---|---|
| `training/dataset_prep_common.py` | `stratified_patient_split_three()` - class-stratified, patient-level split; `check_prep_class_coverage()` - runs the same Stage-A coverage check, but at PREP time, before any extraction |
| `prepare_histologyhsi_bc_v5.py` | v4 + the stratified split wired in as the new default (`--split_strategy legacy_random` restores old behavior byte-for-byte if needed), coverage check runs immediately after the split, before Pass-1 band selection or any shard extraction |
| `prepare_pad_ufes20.py` | new dataset support (see below), uses the same fix |

**How the split fix works**: each patient is grouped by their RAREST
class (the class with the fewest total patients), and groups are processed
rarest-first, with `train`/`val`/`test` patient *counts* computed directly
per group rather than sliced from one global shuffle. A class gets left out
of a split only when its patient pool is genuinely too small to avoid
that (fewer than 2 patients for a 2-way split, fewer than 3 for a 3-way
split) - and that gets reported as an explicit warning (Phase 3's own
escape hatch: "if this is impossible because the source dataset genuinely
lacks enough independent patients... explicitly report the limitation"),
not silently produced or crashed on later.

Verified against the user's exact failure shape (a class with a small
patient pool getting excluded from validation under naive shuffling) across
20 seeds at the default `80_20` split — 0/20 failures — and against a
tighter 3-way `80_10_10` split with a class down to only 3 patients — 0/30
failures. A genuinely unsplittable case (a class with exactly 1 total
patient) is correctly reported as a warning rather than silently dropped:

```
class 3: only 1 patient - validation AND test splits will have NO samples
of this class (assigned to train only)
```

`prepare_histologyhsi_bc_v5.py`'s `--allow_missing_classes` /
`--split_strategy legacy_random` flags mirror `train_example_v11.py`'s own
flags of the same name, so both ends of the pipeline agree on how to
handle an edge case rather than one silently overriding the other.

### Second dataset: PAD-UFES-20

`prepare_pad_ufes20.py` is a new, independent script (this dataset - 2,298
smartphone dermoscopy photos + a `metadata.csv`, not gigapixel ENVI HSI
cubes - doesn't need `prepare_histologyhsi_bc_v5.py`'s streaming-shard/
resumable-manifest machinery, so it's much simpler, not a copy-paste of the
HSI script). It reuses the SAME `stratified_patient_split_three()` /
`check_prep_class_coverage()` fix, since PAD-UFES-20 has the identical risk
profile (patients can have multiple lesions of different diagnoses; MEL and
SCC are minority classes with few distinct patients in the real dataset).

Verified end-to-end against a fabricated dataset matching PAD-UFES-20's
real shape (six classes: ACK/BCC/MEL/NEV/SCC/SEK, realistic class
imbalance with MEL/SCC as the minority classes, multiple images per
patient):
- A deliberately harsh case (MEL/SCC each down to 1 patient) correctly
  aborts before loading any images, with the exact same coverage-report
  format as the HSI path.
- `--allow_missing_classes` completes the full pipeline and writes valid
  `X_train.npy`/`X_val.npy`/`X_test.npy` + `class_names.json`.
- A realistic case (every class ≥3 patients) completes cleanly with
  **no flags needed** - `[class-coverage] every class present in every
  split - OK`.
- The written output was fed straight through the EXISTING (unmodified)
  `train_example_v6.discover_data()` → `NpyDataset` → `GMedMambaEfficient`
  forward pass, confirming RGB modality auto-detection (`wavelengths=None`)
  and the full training pipeline works on this dataset with **zero changes**
  to any Stage A-F file - only a new prep script was needed.

### If you're re-running an already-prepared HistologyHSI dataset

A dataset directory produced by `prepare_histologyhsi_bc_v4.py` (or the old
random split) doesn't need to be regenerated from raw data necessarily -
but if `train_example_v11.py` is currently refusing it for missing classes,
the underlying `.npy` files were built from a bad split and need
re-running through `prepare_histologyhsi_bc_v5.py` (same `--root`/`--seed`,
default `--split_strategy stratified`) to actually fix the class
distribution, not just to pass the check with `--allow_missing_classes`
tacked on as a band-aid.
