# G-MedMamba v16 — Audit Remediation Plan

## Context

The v15 audit (4 Sep 2026) found 18 defects across two training runs
(`20260903_180812` global_zscore, `20260904_010453` per_sample_minmax) on
`hsi_v7-80_10_10_importance`. Three of them invalidate published numbers:

1. **The HSI validation collapse is an acquisition fault, not overfitting.**
   Patient 68's IDC captures were acquired at ~50 % intensity, they sit only in
   validation, and `global_zscore` (a fixed train-fit affine) passes that
   exposure straight into the model. Dark-slide IDC recall goes 0.980 → 0.079
   over 7 epochs. `per_sample_minmax` hides it by construction, which is why the
   RGB arm looked healthy.
2. **The reconstruction pathway has never trained**, on either modality, on the
   current architecture. Six independent defects stack; fixing any one alone
   still yields zeros. This blocks a top-level project objective
   (`plan/MedMamba_Main_Development_Plan.md:13`, "produce scientifically valid
   reconstruction metrics") and all of Phase 4.
3. **Neither audited run is reportable.** Run A picked its checkpoint (epoch 1)
   using a metric that a mis-exposed slide was destroying; Run B has a
   trustworthy validation signal and a weaker model, because per-sample min-max
   discards absolute brightness on the well-exposed 96 % of the data too.

Intended outcome: one HSI run whose validation signal is trustworthy, whose
checkpoint selection is therefore meaningful, whose reconstruction metrics are
real and in reflectance units, and whose per-patient breakdown makes an
acquisition fault visible before it costs a run.

## Decisions taken

| Decision | Choice |
|---|---|
| A-1 acquisition fault | **Both** — `per_patch_zscore` in training now, *and* a re-prep with per-capture gain correction |
| Reconstruction metric units | **Reflectance** — plumb per-patch normalization stats through the DataLoader |
| RGB arm (C-1) | **MIL aggregation** via a patch→image sidecar; no PAD re-prep |
| Delivery | **New generation** — `train_example_v16.py` + `TrainerG_v12`; v15 / v11 frozen |

### The freezing rule (read this before touching anything)

`train_example_v15.py` and `training/trainerg_v11.py` are **frozen** so the two
audited runs stay bit-reproducible. Every earlier generation
(`train_example_v6…v14.py`, `trainerg_v3…v10.py`, `prepare_histologyhsi_bc_v3…v7.py`)
is frozen too, as it already was.

Several defects live in *shared* modules — `NpyDataset` (inside
`train_example_v6.py`), `plots.py`, `reconstruction_head.py`, `gmedmamba.py`.
**Every one of those is on v15/v11's import path** (verified below), so the rule
is absolute:

> **No file that `train_example_v15.py` or `TrainerG_v11` imports may be edited
> at all** — not even additively. New behaviour goes in a new module that
> *imports and subclasses* the frozen one. v15's bytes do not change, so v15's
> numbers cannot.

The only files exempt are those nothing in the v15 path imports, and they are
named explicitly in the third table.

**Import-path audit** (`grep` over `train_example_v15.py` and
`trainerg_v3…v11.py`). These are imported, therefore frozen:

`train_example_v6.py` (v15:98) · `train_example_v7.py` (v15:101) ·
`gmedmamba.py` + `gmedmamba_ema.py` · `training/numerical_stability.py`
(v15:103, v9, v10, v11) · `training/gan.py` (v15:104, v4, v5, v6, v9, v10, v11) ·
`training/reconstruction_head.py` (v15:108) · `training/losses.py` (v15:109) ·
`training/augmentation.py` (v15:111) · `training/dataloader_config.py` (v15:117) ·
`training/run_naming.py` (v15:130) · `training/eval_utils.py` (v4, v5, v6, v11) ·
`training/plots.py` (v4) · `training/spectral_recon_metrics.py` (v4, v6, v11) ·
`training/per_class_metrics.py`, `training/metrics.py`, `training/evaluator.py`,
`training/metric_validation.py`, `training/latent_space.py`,
`training/flops_counter.py`, `training/extra_metrics.py`,
`training/gradient_health.py`, `training/failure_taxonomy.py`, and the rest of
v15's import block.

### File manifest

**NEW — write these:**

| File | Purpose |
|---|---|
| `train_example_v16.py` | Entry point. Start from v15, not from scratch. |
| `training/trainerg_v12.py` | `TrainerG_v12(TrainerG_v11)` — override only what is broken. |
| `training/normalization.py` | All three modes + their inverses (Stage 0.1). |
| `prepare_histologyhsi_bc_v8.py` | Gain-corrected reflectance prep (Stage 1.5). |
| `scripts/derive_patch_groups.py` | Patch → patient/capture/image sidecars (Stage 0.3). |
| `scripts/tune_class_weight_power.py` | Validation-only sweep (Stage 3.3). |
| `test_*.py` × 10 | See Verification. |

**NEW — because the module that needed changing is frozen.** Each one imports
and subclasses (or delegates to) its frozen counterpart; none copies it.

| New file | Frozen module it replaces for v16 | What changes |
|---|---|---|
| `training/npy_dataset_v16.py` | `train_example_v6.NpyDataset` | `NpyDatasetV16(NpyDataset)` — third normalization mode, optional `norm_stats` third return value |
| `training/recursive_features.py` | `gmedmamba.GMedMambaRecursive.forward_deep_supervision` | a segment loop that returns the **live** feature map (R-2) |
| `training/reconstruction_head_v2.py` | `training/reconstruction_head.py` | `LatentReconstructionDecoderV2(LatentReconstructionDecoder)` — configurable output activation (R-3) |
| `training/spectral_recon_metrics_v2.py` | `training/spectral_recon_metrics.py` | non-negativity guard, safe cosine (R-7) |
| `training/gates_v16.py` | `training/numerical_stability.py` | gates G7 + G8, margin reporting (R-1, A-3, C-2) |
| `training/plots_v16.py` | `training/plots.py` | non-zero recon gate + Phase-4 recon figures (R-6) |
| `training/augmentation_v16.py` | `training/augmentation.py` | `AugmentedPatchDatasetV16` — shared-memory epoch counter (E-4) |
| `training/dataloader_config_v16.py` | `training/dataloader_config.py` | non-persistent validation workers (E-4 follow-on) |
| `training/run_naming_v16.py` | `training/run_naming.py` | `ppz` token, `_frac_token` fix |
| `scripts/shallow_baseline_v16.py` | `scripts/shallow_baseline.py` | thin wrapper adding the third mode to gate G4 |

**EXEMPT — nothing in the v15 path imports these, so they may be edited:**

| File | Change | Why it is safe |
|---|---|---|
| `training/aggregate_report.py` | folder-first, root-fallback lookup for per-epoch artifacts (Stage 5) | standalone CLI; `grep` finds no importer |
| `documentations/10_commands_v15.md:90` | correct the `hsi_all_calibrated` claim (Stage 1.5) | prose |

Everything else that used to be on this list moved to the second NEW table.
There is now **no** edit to any file v15 or v11 imports.

**FROZEN — do not edit, reuse as-is:**

`train_example_v6…v15.py` · `training/trainerg_v3…v11.py` · `gmedmamba*.py` ·
`prepare_histologyhsi_bc_v3…v7.py` · `prepare_pad_ufes_20_v4…v6.py` ·
`training/gan.py` · `training/eval_utils.py` · `training/losses.py` ·
`training/numerical_stability.py` · `training/reconstruction_head.py` ·
`training/spectral_recon_metrics.py` · `training/plots.py` ·
`training/augmentation.py` · `training/dataloader_config.py` ·
`training/run_naming.py` · `training/prep/*` · `scripts/shallow_baseline.py`

**Enforcement.** Before committing, `git diff --stat` must show **zero** changes
to any path in this list. Add it to the plan's definition of done — this is the
one invariant that keeps the audited runs reproducible.

### Reuse, don't reimplement

v16 / v12 must call these rather than writing their own. Every one of them is
already correct; the defects are in what calls them, not in them.

- `training/eval_utils.py` — `write_classification_report`, `save_confusion_matrix`,
  `save_prediction_distribution`. **All three already take a destination
  argument**, which is what makes Stage 5's run-directory fix a one-line override.
- `training/trainerg_v4.py` — `_update_loss_components_file` (`:596`),
  `_save_per_class_metrics`, `_check_stopping_rules` (`:627`). Inherited and
  correct; v5 and v11 simply stopped calling the first two (R-6).
- `training/gan.py` — `SAMLoss`, `safe_cosine_similarity` (`:56`),
  `SpectralDiscriminator`, `generator_adversarial_loss`, `discriminator_loss`.
- `training/spectral_recon_metrics.py` — `compute_all_spectral_metrics`,
  `compute_per_sample_spectral_metrics`.
- `training/losses.py` — `build_criterion`, `compute_class_weights` (already
  carries the staged `class_weight_power` work).
- `training/prep/` — `run_prep`, `DatasetAdapter`, `ShardWriter`,
  `ItemResult.extra` (a live but always-empty dict, already persisted verbatim
  to `_progress.json` at `core.py:390-392` — no schema change needed).
- `training/dataset_prep_common.py` — `stratified_patient_split_three`.
- `training/numerical_stability.py` — `stem_sensitivity`, `run_sensitivity_check`.
- `training/plots.py` — `plot_confusion_matrix`, `generate_all_training_plots`.
- `training/npy_integrity.py`, `training/dataloader_config.py`,
  `training/run_naming.py`, `training/checkpoint.py`.

### `TrainerG_v12` inheritance shape

Subclass `TrainerG_v11` and override only these — everything else is inherited:

| Override | Why |
|---|---|
| `_train_one_epoch_deep_supervision` | R-5 single decode path |
| `_validate_one_epoch_impl` | R-4 return the components, R-5 same path, R-3/R-7 reflectance metrics |
| `_forward_with_recon` *(new)* | R-2/R-5 the one shared decode helper |
| `_check_stopping_rules` | E-1 watch the checkpoint metric |
| `_save_eval_artifacts` | Stage 5 run-directory layout |
| `fit` | R-4 index components directly, R-6 call the artifact writers, E-2 step-axis scheduler |

`fit` has to be copied — v11 assembles the metrics dict inline (`:820-845`),
which is exactly why v11 copied it from v5, which copied it from v4. **Break the
chain**: in v12, extract the dict assembly into `_build_epoch_metrics(...)` so
v13 can override that instead of copying `fit` a fourth time.

---

## Stage 0 — Shared foundations

Everything else depends on these three. Build and test them first.

### 0.1 `training/normalization.py` (new)

Single home for every normalization mode and, crucially, its **inverse**.

```python
NORMALIZATION_MODES = ("per_sample_minmax", "global_zscore", "per_patch_zscore")

def normalize_patch(patch_hwc, mode, global_mean=None, global_std=None):
    """-> (normalized_hwc, offset_c, scale_c) with raw == normalized * scale + offset.
    offset/scale are ALWAYS per-channel float32 [C], whatever the mode, so the
    collated batch has a uniform [B, 2, C] shape."""

def denormalize(x_bchw, norm_stats_b2c):   # -> reflectance-domain tensor
```

- `per_sample_minmax` → `offset = p_min`, `scale = p_max - p_min` (scalars broadcast to `[C]`)
- `global_zscore`     → `offset = global_mean`, `scale = global_std`
- `per_patch_zscore`  → `offset = patch.mean()`, `scale = patch.std()` over all H×W×C
  (the audit's definition; measured 0.07 σ on normal validation IDC vs 0.13 σ for
  `global_zscore`, and 0.35 σ on dark vs 2.79 σ — it wins on **both** columns)

Port the existing math verbatim from `train_example_v6.py:137-144` so the two
legacy modes stay bit-identical, and add a `test_normalization.py` that asserts
`denormalize(normalize(x)) == x` to float32 tolerance for all three modes.

### 0.2 `training/npy_dataset_v16.py` (new)

`train_example_v6.NpyDataset` is imported by v15 (`train_example_v15.py:98`) and
is therefore frozen. Subclass it instead — no copy, no edit:

```python
from train_example_v6 import NpyDataset          # frozen, imported not edited

class NpyDatasetV16(NpyDataset):
    """Adds `per_patch_zscore` and the optional third return value.
    Everything else - storage_mode, mmap, the length assert, the flip
    augmentation - is inherited unchanged."""

    def __init__(self, *a, normalization="per_sample_minmax",
                 return_norm_stats: bool = False, **kw):
        # the parent asserts a 2-mode whitelist at :119, so construct it with a
        # legal mode and set the real one afterwards
        super().__init__(*a, normalization=("per_sample_minmax"
                                            if normalization == "per_patch_zscore"
                                            else normalization), **kw)
        self.normalization = normalization
        self.return_norm_stats = return_norm_stats

    def __getitem__(self, idx):        # full override; does NOT call super()
        ...  # nan_to_num -> training.normalization.normalize_patch -> flip -> CHW
```

`__getitem__` is overridden rather than wrapped because the parent normalizes
and converts to a tensor in one pass (`:133-154`); re-deriving `norm_stats` from
its output is not possible. Port those 22 lines into the subclass and assert in
`test_normalization.py` that the two legacy modes are bit-identical to the
parent's output for the same input.

Because the parent is untouched, `AugmentedPatchDataset` still wraps it fine —
but it must pass a 3-tuple through, which is why Stage 3.4 introduces
`AugmentedPatchDatasetV16`.

### 0.3 `scripts/derive_patch_groups.py` (new, read-only, no re-prep)

Unblocks A-4, A-5 and C-1 **today**, on the existing on-disk datasets.

Row order in the unified `X_*.npy` is deterministic and sorted by item key
(`training/prep/manifest.py:collect_shard_entries(deterministic=True)`, called
from `training/prep/core.py:208`). So a cumulative sum over
`sorted(_progress.json["completed_items"])`, filtered by split, using
`counts[modality]`, reconstructs the provenance of every patch row exactly.

Emits alongside the modality dir:

- `groups_{split}.npy` — patient id per patch (`int32`)
- `captures_{split}.npy` — capture index per patch (`int32`)
- `capture_index.json` — index → `{key, patient, tissue, capture, split, n}`

For `data/pad_v6`, the same script emits `images_{split}.npy` (clinical image id).

**Self-check, mandatory:** assert `sum(counts) == len(y_{split})` per split and
that the derived per-group class labels are single-valued. If either fails, the
sidecar is not written — a silently misaligned group vector is worse than none.

---

## Stage 1 — Data truth (A-1 … A-5)

### 1.1 `per_patch_zscore` reaches the CLI — A-2

Touch points (all four, or the run name lies):

| File | Status | Change |
|---|---|---|
| `training/normalization.py` | new | canonical implementation (Stage 0.1) |
| `training/npy_dataset_v16.py` | new | applies it (Stage 0.2) |
| `train_example_v16.py` | new | `--normalization` choices gain `per_patch_zscore`; `auto` resolves HSI → `per_patch_zscore` |
| `training/run_naming_v16.py` | new | own `_NAME_ALLOWLIST` with `"per_patch_zscore": "ppz"`, plus a fixed `_frac_token` |
| `scripts/shallow_baseline_v16.py` | new | thin wrapper so gate G4's ceiling is measured under the same transform |

`training/run_naming.py:198` and `scripts/shallow_baseline.py:108-119` are the
two places that carry a *second* copy of the mode list. Both are frozen
(`run_naming` is imported at `train_example_v15.py:130`), so v16 gets its own:

- `training/run_naming_v16.py` imports `dataset_slug`, `infer_modality` and
  `modality_warnings` from the frozen module unchanged, and redefines only
  `_NAME_ALLOWLIST` + `_settings_suffix` + `build_run_name`. `_settings_suffix`
  reads the module-level tuple directly (`run_naming.py:213`), so it cannot be
  parameterised from outside — redefining is the only option.
- `scripts/shallow_baseline_v16.py` imports the frozen script as a module and
  substitutes its normalization function, rather than duplicating 259 lines.

**`_frac_token` bug, fixed in the v16 copy only** (`run_naming.py:207-211`):
`0.1` renders as `"1"`, so a directory named `sub1_vsub1` actually means *0.1*,
not 1.0. Prepend `"0"` when the value is < 1. Fixing it in place would rename
future v15 runs, which is exactly what the freeze forbids.

### 1.2 Gate G8 — post-normalization split drift — A-3

New preflight in **`training/gates_v16.py`** (new file — `numerical_stability.py`
is imported at `train_example_v15.py:103` and by v9/v10/v11, so it is frozen),
run from v16 before `trainer.fit`. This is the check that would have caught
patient 68 in seconds.

- Sample `--drift_check_patches` (default 8000) patches per split.
- Apply the **resolved** normalization.
- Report, per split: raw mean, post-norm mean, post-norm std, and fraction below
  the train 1st percentile.
- **Fail** when `|mean_split| > 0.25` or `std_split` outside `[0.75, 1.25]`, in
  train-group sigmas. Under `global_zscore` today, validation is −0.303 / 1.364
  / 8.6 % — a clean fail.
- With `groups_{split}.npy` present, additionally report the **per-patient**
  post-norm mean and name the outliers explicitly. Patient 68 should print by
  name.
- Writes `<exp_dir>/split_drift_report.json`; `--on_split_drift {abort,warn,off}`,
  default `abort`.

### 1.3 Per-patient / per-capture breakdown — A-4

`TrainerG_v12` accepts optional `val_groups` / `test_groups` vectors and, in
`_save_eval_artifacts` and `evaluate_test_split`, writes:

- `per_patient_metrics.json` — recall / precision / F1 per class per patient
- `per_capture_metrics.csv` — same per capture, plus each capture's mean raw
  intensity (from `capture_index.json`)

Gate the checkpoint report on it: if any single patient's recall for any class
drops below `--min_patient_recall` (default off; suggest 0.5) while the macro
metric rises, log a `PATIENT_OUTLIER` warning into `history.json`.

### 1.4 Patient-aware validation subsample — A-5

`_subsample_indices` (`train_example_v15.py:931-954`, ported into v16) gains a
`groups` argument and a `stratified_group` mode: draw the per-class fraction
*within each patient*, so a 10 % subset keeps every patient represented in
proportion. Default for v16 when a group vector is available.

Note the structural limit honestly in the write-up: **the validation split is 5
patients.** Sub-sampling cannot fix that. Recommended (optional, Stage 5):
a patient-grouped 5-fold CV over the 40 non-test patients for the headline
number, with the 5-patient test split touched exactly once.

### 1.5 Re-prep: `prepare_histologyhsi_bc_v8.py` — A-1 at the source

New adapter, delegating HSI IO to the frozen v6/v7 helpers as v7 already does.

**Design: store per-capture-gain-corrected reflectance, not raw DN.** This one
change closes A-1 at the source *and* gives R-3/R-7 the reflectance units they
need *and* halves storage.

New flags (all added to `compat_keys` — `prepare_histologyhsi_bc_v7.py:108-118`
— so a resumed run can never mix corrected and uncorrected shards):

- `--capture_gain {none, median_ratio}` (default `median_ratio`)
  Pass 1 already walks the captures; extend it to sample ROI pixels per capture
  and record a median intensity. Per capture, `gain = corpus_median / capture_median`,
  clipped to `[0.5, 2.0]` with anything outside logged as a suspect acquisition
  rather than silently stretched. Applied in `_hsi_stream`
  (`prepare_histologyhsi_bc_v7.py:267-276`), between the flat-field call and the
  band slice.
- `--hsi_value_scale <float|auto>` — divide by a recorded constant so cubes land
  in reflectance-like `[0, ~1.5]`. Recorded in `dataset_manifest.json` via the
  existing `provenance()` hook (`training/prep/adapter.py:118`).
- `--emit_group_sidecars` (default on) — write `groups_*.npy` / `captures_*.npy`
  / `capture_index.json` directly, so Stage 0.3 becomes a fallback rather than
  the only source.

Two existing bugs to fix while in here:

- **`hsi_all_calibrated` is inverted.** `get_calibration_means`
  (`prepare_histologyhsi_bc_v6.py:314-315`) returns `(None, None)` precisely
  *because* the cube is already calibrated, and the probe at
  `prepare_histologyhsi_bc_v7.py:179-181` reads that as "uncalibrated". Note
  carefully: fixing the flag **alone changes nothing**, because
  `_FLOAT16_SAFE_MAX = 60000.0` and the corpus `hsi_value_max` is `64829.0`, so
  `:239` still selects float32. It is `--hsi_value_scale` that unlocks float16
  and takes `X_train.npy` from 38 GB to ~19 GB. Also correct
  `documentations/10_commands_v15.md:90`, which repeats the wrong conclusion.
- Per-capture stats (`gain`, `median_intensity`, `n_sampled`) go into
  `ItemResult.extra` (`training/prep/parallel.py:36`) — a live but always-empty
  dict already persisted verbatim to `_progress.json`
  (`training/prep/core.py:390-392`). Zero schema change.

**Sequencing:** the re-prep is hours of compute and a ~19 GB write. Nothing else
in this plan blocks on it — Stage 0.3 gives the group sidecars now, and
`per_patch_zscore` gives a correct run now. Kick the re-prep off in the
background and let it land for the final numbers.

---

## Stage 2 — The reconstruction pathway (R-1 … R-7)

Six defects stack. Fix in this order; each is independently testable.

### 2.1 R-2 — the decoder must read a live tensor

`gmedmamba.py` is frozen — it is the model every trainer imports. The fix goes
in **`training/recursive_features.py`** (new), and it needs no patch at all,
because `forward_deep_supervision` (`gmedmamba.py:1942-1982`) is built entirely
from public attributes:

```python
def forward_deep_supervision_with_features(model, x, wavelengths=None, sensor_range=None):
    """`GMedMambaRecursive.forward_deep_supervision`, returning the LIVE final
    feature map as a third value. Reads only public API - `backbone.embed`,
    `backbone.core.{init_states,one_segment}`, `model.head`, `model.cfg` - so
    `gmedmamba.py` is untouched."""
    x_emb, _, _ = model.backbone.embed(x, wavelengths, sensor_range)
    core = model.backbone.core
    y, z = core.init_states(x_emb)
    ...                                   # same ACT / halting / detach-between-segments loop
    return logits_list, q_list, y         # y still attached to the graph
```

This is the repo's documented extension pattern — "`gmedmamba.py` stays
framework-free; gradient checkpointing, reconstruction heads and preset
overriding all wrap or monkey-patch the model at runtime"
(`documentations/README.md`), and it is exactly what the newest module,
`training/torch_compile.py`, already does.

Why this works: the `.detach()` at `:1981` exists only because
`EMAHelper.ema_copy` calls `copy.deepcopy(module)` every epoch and deepcopy
refuses a *module attribute* holding a live autograd graph. Returning `y` from a
free function sidesteps that entirely — nothing is stored on the module, so
deepcopy stays safe and the auxiliary loss becomes real.

**Drift risk, and its guard.** This duplicates ~25 lines of ACT/halting logic.
`test_recursive_features_parity.py` asserts that, in eval mode with a fixed
seed, `forward_deep_supervision_with_features(m, x)[:2]` is elementwise equal to
`m.forward_deep_supervision(x)`. If someone edits the frozen loop, that test
fails loudly instead of the two silently diverging.

### 2.2 R-5 — one decode path for train and validation

Today training decodes `base.last_feature_map` (from `forward_deep_supervision`)
while validation decodes inside `GMedMambaLatentReconWrapper.forward`, which
re-runs `backbone.forward_features` → `run_segments` — **a different segment
recursion**. The two MSE numbers would not be comparable even after 2.1.

`TrainerG_v12` gets one helper used by **both** loops:

```python
def _forward_with_recon(self, model, x):
    """-> (logits_list, q_list, x_recon). One path, so train MSE and val MSE
    measure the same thing."""
```

- recursive base + decoder → `forward_deep_supervision(..., return_features=True)`,
  then `decoder(feat, target_hw=x.shape[-2:])`
- otherwise → existing `_model_forward` (`trainerg_v11.py:133`)

`_train_one_epoch_deep_supervision` (v12 override of `:178`) and
`_validate_one_epoch_impl` (v12 override of `:362`) both call it. Delete the
`base.last_feature_map` read at `:220` from the v12 path.

### 2.3 R-3 — the decoder output range must be able to represent the input

`training/reconstruction_head.py` is imported at `train_example_v15.py:108`, so
it is frozen. **`training/reconstruction_head_v2.py`** (new) subclasses it — the
only difference is the final line of `forward`:

```python
from training.reconstruction_head import LatentReconstructionDecoder, SpectralDropout

class LatentReconstructionDecoderV2(LatentReconstructionDecoder):
    def __init__(self, *a, out_activation: str = "linear", **kw):
        super().__init__(*a, **kw)          # proj / out_conv inherited verbatim
        self.out_activation = out_activation

    def forward(self, feature_map, target_hw):
        x = feature_map.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.out_conv(self.proj(x))
        return torch.sigmoid(x) if self.out_activation == "sigmoid" else x
```

`GMedMambaLatentReconWrapperV2` alongside it builds the V2 decoder and, for a
recursive base, decodes via `recursive_features` (Stage 2.2) instead of
`backbone.forward_features`.

`"sigmoid"` | `"linear"`. v16 selects `linear` for any
z-score mode and `sigmoid` for min-max, and **raises at startup** on a mismatch
rather than training into a floor. Under `global_zscore` the target is ~N(0,1)
while `x_recon ∈ (0,1)`, which floors `F.mse_loss` near 1.0 no matter how good
the decoder is — this, not only the detach, is the "adds ~1.0 to the loss" noted
in the v15 plan (`train_example_v15.py:678`).

### 2.4 R-3 / R-7 — metrics in reflectance units

The loss stays in normalized space (scale-uniform, which is what an auxiliary
loss wants). The **reported** metrics do not.

In `TrainerG_v12._validate_one_epoch_impl`, before
`compute_all_spectral_metrics` (`trainerg_v11.py:408`):

```python
x_r      = denormalize(x, norm_stats)
recon_r  = denormalize(x_recon, norm_stats)
metrics  = compute_all_spectral_metrics(x_r, recon_r, wavelengths=wl_tensor)
```

**`training/spectral_recon_metrics_v2.py`** (new — the v1 module is imported by
`trainerg_v4/v6/v11`, so it is frozen) redefines `sam_deg`, `sid`,
`peak_position_error` and `compute_all_spectral_metrics` with a
`require_nonnegative=True` guard that **raises** if handed data with a
meaningful negative fraction, instead of silently producing an arbitrary angle.
`rmse`, `mae`, `psnr`, `ssim`, `pearson_correlation` and `cosine_similarity` are
imported from the frozen module unchanged — they are correct in any domain. Spectral angle is only meaningful for non-negative reflectance spectra,
and SID — which renormalizes each spectrum to a probability distribution via
`clamp(min=1e-8)` — is worse on z-scored input. Small negatives from decoder
error are clamped at zero and the clamped fraction is reported.

The v2 module also drops the product-of-norms `+eps` form that
`spectral_recon_metrics.sam_deg:44-53` uses and that `training/gan.py:20-45`
documents as numerically unsafe — it imports `safe_cosine_similarity` from
`gan.py` (frozen, imported not edited) instead of reimplementing it.

### 2.5 R-4 — validation MSE/SAM/GAN stop being structural zeros

`_validate_one_epoch_impl` returns a dict with **no** `mse_loss` / `sam_loss` /
`gan_loss` key (`trainerg_v11.py:437-444`): MSE is computed at `:389` and
discarded, SAM and GAN are never computed in validation at all. `fit()` then
reads them with `val_metrics.get("mse_loss", 0.0)` (`:837-839`), so all three are
literal zeros in `history.json`, `history.csv` and the plots for **every run
since TrainerG_v6**, even with reconstruction fully enabled. `TrainerG_v4`
indexed them directly and worked; v6 dropped them from the return dict and the
`.get(…, 0.0)` default converted the resulting `KeyError` into a silent zero.

v12: accumulate `total_mse` / `total_sam` / `total_gan` in the validation loop
and return all three, then **index them directly** — `val_metrics["mse_loss"]`,
no `.get` default. A missing key must fail loudly. That defaulting is what hid
this for four trainer versions, and deleting it is half the fix.

### 2.6 R-6 — the artifacts get written

- `TrainerG_v12.fit` calls `_update_loss_components_file(metrics)`
  (`trainerg_v4.py:596`) and `_save_per_class_metrics`. Both are inherited and
  correct; `TrainerG_v5.fit` and `TrainerG_v11.fit` simply stopped calling them.
- **`training/plots_v16.py`** (new) redefines only `generate_all_training_plots`,
  importing every individual `plot_*` primitive from the frozen `training/plots.py`.
  The one change is the gate at `plots.py:231`: `any(r.get("MSE_loss") is not
  None ...)` becomes the non-zero test already used for SAM at `plots.py:75`,
  `any(v not in (None, 0) for v in series)`. `MSE_loss` is always a float (0.0
  when `decoder is None`), never `None`, so the guard is currently always true
  and every run gets a meaningless flat-zero `reconstruction_loss.png` beside an
  empty `reconstruction/` directory.
  `TrainerG_v12._generate_rich_diagnostics` calls the v16 version;
  `trainerg_v4.py:712` and every v11 run keep calling the frozen one.
- `plots_v16.py` also adds the Phase-4 visualisations into `reconstruction/`:
  original-vs-reconstructed spectra (best / median / worst by SAM) and a
  histogram per metric. `recon_true_means` / `recon_pred_means` are already
  collected and thrown away (`trainerg_v11.py:418-419`).

### 2.7 R-1 — remove the override, replace it with a gate

`train_example_v15.py:672-684` force-sets `recon_mode="none"` for
`architecture == "recursive"`, zeroes `--lambda_mse` / `--lambda_sam`
(`:1591-1592`) and discards `--use_gan` — with no warning, while the parser still
advertises defaults of 1.0 and 0.1. Both the HSI and RGB runs use
`--architecture recursive`, so **both** got reconstruction switched off.

In v16: delete the override; `--recon_mode` defaults to `latent` for recursive.
Replace it with a gate so R-2 can never silently regress:

> **Gate G7 — reconstruction gradient reaches the encoder.** One preflight
> batch. Zero the classification term, backprop
> `lambda_mse*mse + lambda_sam*sam` alone, and assert that parameters *inside
> the recursive core* (not just the decoder) received finite, non-zero gradient.
> Abort with `RECONSTRUCTION_DETACHED` otherwise.

G7 lives in `training/gates_v16.py` and is the structural guarantee that R-2
stays fixed. v16 calls it right after its own copy of
`run_representation_sensitivity_check` (ported from
`train_example_v15.py:956-993` into the new entry point) and writes the result
into `gates.json`.

v16 also drops the ungated `lambda_gan` passthrough that v15 has at `:1591-1593`
and `:829-831` — harmless there only because `TrainerG_v4.__init__:99` re-derives
`use_gan` from `discriminator is not None`. Nothing is edited in v15; v16 simply
does not reproduce it.

---

## Stage 3 — Training-loop correctness (E-1 … E-4)

### 3.1 E-1 — `VAL_DIVERGENCE` watches the wrong metric

`training/trainerg_v4.py:627-642`: the rule fires when raw `val_accuracy` falls
and `val_loss` rises for 3 consecutive epochs, threshold hardcoded at `:635`,
no CLI flag. Under weighted CE the objective is balanced performance, so raw
accuracy is the wrong signal — and this is exactly the rule that killed Run A at
epoch 7. (`trainerg_v3.py:342-345` uses 2, not 3 — an undocumented discrepancy.)

v12 overrides `_check_stopping_rules` to watch `self.checkpoint_metric`
(macro-F1 / balanced accuracy) and adds `--val_divergence_patience`
(default 3, `0` disables). Recommendation for v16 runs: disable it and rely on
`--early_stop_patience` (`training/trainerg_v7.py:43-81`), which already
monitors the right metric.

### 3.2 E-2 — cosine LR on the wrong axis, and no warmup

`train_example_v15.py:1472` is `CosineAnnealingLR(T_max=args.epochs)`, stepped
once per epoch (`trainerg_v11.py:809`). With
`--train_subsample_frac 0.1 --train_subsample_mode per_epoch`, an "epoch" is a
tenth of a pass — so 20 "epochs" is 2 real passes, and the cosine is tied to
wall-clock epochs rather than optimizer steps. If early stopping fires first,
the cosine never reaches its minimum. There is no warmup anywhere in the repo.

v16: build the schedule on the **step** axis —
`total_steps = epochs * ceil(kept / batch_size)`, `LambdaLR` with a linear
warmup of `--warmup_steps` (default 3 % of total, floor 200) into cosine decay.
v12 steps the scheduler in the batch loop when constructed with
`scheduler_interval="step"`; `"epoch"` remains available and is what v11 did.

### 3.3 E-3 — `class_weight_power` was tuned on test

The help text at `train_example_v15.py:185-197` and `training/losses.py` record
that 0.75 was chosen by re-deciding **saved test-split probabilities**. That is a
test-set-informed hyperparameter and it contaminates every number reported from
it.

- `scripts/tune_class_weight_power.py` (new): identical sweep, restricted to
  saved **validation** probabilities, writing `class_weight_power_sweep.json`.
- v16 records the tuning provenance in `config.json`
  (`{"value": …, "tuned_on": "validation", "sweep": "<path>"}`) and refuses to
  start if it is `test`.
- Re-run the sweep on validation and take whatever it says — do not carry 0.75
  over. Also record in the write-up that the published 0.75 figure was test-tuned
  and has been superseded.
- Bonus, already documented in the staged `training/losses.py` diff:
  `--weight_method balanced` and `inverse` are **identical** after the mean-1
  renormalization. Keep the flag, keep the docstring saying so.

### 3.4 E-4 — `set_epoch` is inert under `persistent_workers=True`

`training/augmentation.py:270-274` documents it, and the documented run command
(`documentations/10_commands_v15.md:113`) uses `--loader_mode performance`,
which sets `persistent_workers=True` (`training/dataloader_config.py:46`).
Workers are never re-forked, so the epoch never reaches them and the augmentation
stream repeats identically every epoch — the R6.3 fix has never activated in any
real run.

Fix, in **`training/augmentation_v16.py`** (new — `augmentation.py` is imported
at `train_example_v15.py:111`):

```python
from training.augmentation import AugmentedPatchDataset   # frozen

class AugmentedPatchDatasetV16(AugmentedPatchDataset):
    """set_epoch that survives persistent workers, and passes a 3-tuple through."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._epoch_shared = torch.zeros(1, dtype=torch.int64).share_memory_()
    def set_epoch(self, epoch):        # main process writes
        self._epoch_shared[0] = int(epoch)
```

The tensor is allocated **before** the workers fork, so writes from the main
process are visible to persistent workers. `__getitem__` reads
`int(self._epoch_shared[0])` for its per-item seed, and passes through a
`norm_stats` third element when the wrapped dataset returns one (Stage 0.2).
Fall back to the parent's attribute when sharing is unavailable, and log which
path is live.

While here, in **`training/dataloader_config_v16.py`** (new — the v1 is imported
at `train_example_v15.py:117`): `dataloader_config.py:114-131`'s docstring
promises validation loaders are "never persistent", but `:126` keeps
`train_policy.persistent_workers` whenever `val_workers > 0`. The latest run
holds 3 idle worker processes for its whole life. `train_val_policies_v16` calls
the frozen `train_val_policies` and then applies
`replace(val_policy, persistent_workers=False)` — a four-line wrapper, not a
copy.

---

## Stage 4 — The RGB arm (C-1, C-2)

### 4.1 C-1 — make the existing PAD run interpretable

PAD-UFES-20 is patchified to **11×11×3** (740,800 train patches, ~400 per
clinical image) and every patch inherits the whole-lesion diagnosis. An 11×11
crop of a clinical skin photograph is a few percent of the lesion width and
carries essentially only colour — it cannot separate melanoma from nevus. (The
same 11×11 is reasonable for HSI histology at 10×, where it is a genuine
histological texture unit. The patch size is not wrong; reusing it across
modalities is.)

- `scripts/derive_patch_groups.py` emits `images_{split}.npy` for `data/pad_v6`.
- v16 gains `--eval_group_aggregation {none, mean_prob, majority}` (default
  `none`). When set, validation and gate G6 additionally report **image-level**
  metrics by aggregating patch posteriors per clinical image.
- Report it explicitly as a patch-MIL result. **It is not a like-for-like
  modality comparison** and must not be presented as the HSI run's control.
  Note in the write-up that run `20260903_010639` reached balanced accuracy
  0.2715 after 28 epochs against a 6-class chance of 0.167.

### 4.2 C-2 — gates must report margin, not pass/fail

At initialization: RGB `stem_sensitivity 0.104` / `logit_std 0.0243`; HSI `0.533`
/ `0.250`. RGB clears the G1/G2 thresholds (0.05, 0.01) but by a factor that
should have been flagged — the representation is ~10× weaker.

- `training/gates_v16.py` wraps the frozen
  `numerical_stability.run_sensitivity_check`, keeps its verdict, and adds
  `{"value", "threshold", "margin": value/threshold}` per gate, **warning below
  3×**. Written into `gates.json` and the preflight banner. A wrapper, not a
  reimplementation — the sensitivity math itself is correct and stays put.
- The wrapper also gates on `disjoint_batch_mean_logit_delta`, which the frozen
  function measures at `:482-485` and never puts into `failures`.
- Document that `SpectralPathway` is built around a 32+ band spectral sequence
  and that its tokenizer, scan and compressor are close to degenerate at C = 3.

---

## Stage 5 — Run-directory hygiene (quality-of-life)

Not an audit finding — a usability fix. A 14-epoch run currently drops **21
per-epoch files** into the run root: `confusion_matrix_epoch_01..14.png`,
`classification_report_epoch01..14.txt` and `confusion_matrix.npy`, mixed in
with `config.json`, `history.csv`, `gates.json` and the checkpoints. Every other
artifact family already has a folder (`plots/`, `predictions/`,
`confidence_analysis/`, `latent_space/`, `reconstruction/`, `checkpoints/`);
these two are the exception.

**Rule: the run root holds only run-level files. Anything per-epoch lives in a
folder.**

`TrainerG_v12.__init__` extends the `self.dirs` dict (`trainerg_v4.py:143-152`):

```python
self.dirs["confusion_matrix"]      = self.exp_dir / "confusion_matrix"
self.dirs["classification_reports"] = self.exp_dir / "classification_reports"
```

`TrainerG_v12._save_eval_artifacts` then passes those instead of `self.exp_dir`:

```python
write_classification_report(
    labels, preds,
    self.dirs["classification_reports"] / f"classification_report_epoch{epoch:02d}.txt",
    class_names=self.class_names)
save_confusion_matrix(labels, preds, str(self.dirs["confusion_matrix"]), epoch,
                      class_names=self.class_names)
```

`training/eval_utils.py:101,113` already take `out_path` / `out_dir`, so **no
shared module changes** — this is a pure v12 override, and v11 runs keep their
current layout.

Resulting root, after this and the Stage 2 additions:

```
config.json  history.{json,csv}  experiment_report.json  gates.json
best_model{,_live,_ema}.pt  loss_components.{csv,json}
checkpoint_reproducibility.json  test_report.json  test_predictions.npz
split_drift_report.json  per_patient_metrics.json          <- new, run-level
dataset_*.json  class_*.json  leakage_report.json  validation_report.json
system_memory_startup.json
checkpoints/  plots/  predictions/  confidence_analysis/  latent_space/
reconstruction/  confusion_matrix/  classification_reports/   <- 2 new folders
per_capture_metrics.csv                                       <- new, run-level
```

Consumers:

- `training/aggregate_report.py` — a standalone CLI that nothing in the v15
  import path imports (`grep` confirms), so it is **exempt** and may be edited
  directly. Make any glob of `confusion_matrix_epoch_*.png` /
  `classification_report_epoch*.txt` try the folder first and fall back to the
  root, so **existing v11 runs stay readable**.
- `training/trainerg_v4.py:712` (`_generate_rich_diagnostics`) already routes
  through `self.dirs[...]`, so v12 gets the new layout for free by extending the
  dict. Nothing to change in the frozen file.

---

## Stage 6 — What to run

Nothing here is reportable until Stages 0–3 land and G7 + G8 pass.

| # | Run | Purpose | Blocks on |
|---|---|---|---|
| 1 | `per_patch_zscore`, `recon_mode none` | Isolate the normalization change against Run A / Run B. Expect the collapse gone *and* test macro-F1 above Run B's 0.8130. | Stages 0, 1.1-1.4, 3, 5 |
| 2 | `per_patch_zscore`, `recon_mode latent`, linear head | First real reconstruction numbers, in reflectance units. G7 must pass. | + Stage 2 |
| 3 | Ablation: CE / CE+MSE / CE+MSE+SAM / +GAN | Plan Phase 11's stated ablation, finally runnable. | + run 2 |
| 4 | Re-prep dataset, best config from 1-3 | Final headline numbers on gain-corrected reflectance. | + Stage 1.5 |
| 5 | PAD-UFES-20 with `--eval_group_aggregation mean_prob` | Image-level RGB result, reported as patch-MIL. | + Stage 4 |

Test split discipline: `--eval_test` (gate G6) runs **once per configuration**,
after the checkpoint is selected on validation. Every hyperparameter — including
`class_weight_power` — is decided on validation only.

---

## Findings register → work items

| ID | Finding | Work item |
|---|---|---|
| A-1 | Patient 68 IDC at ~50 % intensity | 1.1 (normalize) + 1.5 (re-prep gain correction) |
| A-2 | `auto` gives HSI a fixed train-fit affine | 0.1, 1.1 |
| A-3 | No preflight gate on post-norm split drift | 1.2 (gate G8) |
| A-4 | No per-patient / per-capture breakdown | 0.3, 1.3 |
| A-5 | Val split is 5 patients; subsample stratifies by class | 0.3, 1.4 |
| R-1 | Reconstruction force-disabled on `recursive` | 2.7 (+ gate G7) |
| R-2 | Decoder reads a detached feature map | 2.1 |
| R-3 | Sigmoid output vs z-scored input → MSE floor ~1.0 | 2.3, 2.4 |
| R-4 | Validation MSE/SAM/GAN structurally zero since v6 | 2.5 |
| R-5 | Train and val decode from different feature maps | 2.2 |
| R-6 | `loss_components.csv` never written; flat-zero plot | 2.6 |
| R-7 | SAM/SID undefined on z-scored cubes | 2.4 |
| E-1 | `VAL_DIVERGENCE` watches raw accuracy, hardcoded 3 | 3.1 |
| E-2 | Cosine LR on the epoch axis, no warmup | 3.2 |
| E-3 | `class_weight_power` tuned on test probabilities | 3.3 |
| E-4 | `set_epoch` inert under `persistent_workers=True` | 3.4 |
| C-1 | RGB patches 11×11 with whole-lesion labels | 0.3, 4.1 |
| C-2 | G1/G2 pass on RGB with a 10× weaker representation | 4.2 |

---

## Verification

Environment: `PYTHONPATH=/data/dante_data/documents/Masters/courses/Thesis/g-medmamba /home/dante/.local/share/mamba/envs/gmedmamba/bin/python`

### New tests (root-level `test_*.py`, matching repo convention)

| File | Asserts |
|---|---|
| `test_normalization.py` | round-trip `denormalize(normalize(x)) == x` for all 3 modes; `per_sample_minmax` / `global_zscore` bit-identical to `train_example_v6.py:137-144` |
| `test_recon_gradient.py` | **the load-bearing one** — with `recon_mode=latent`, backprop of the recon term alone puts non-zero grad on recursive-core params; and `copy.deepcopy(model)` still succeeds after a forward (the EMA constraint that caused R-2) |
| `test_recon_metrics_reported.py` | `_validate_one_epoch_impl` returns non-zero `mse_loss`/`sam_loss`; `fit` writes them to history; `KeyError` if a key is missing |
| `test_split_drift_gate.py` | G8 fails a synthetic split shifted 2.8 σ; passes a matched one |
| `test_group_sidecars.py` | derived `groups_*.npy` length matches `y_*.npy`; per-group labels single-valued |
| `test_scheduler_warmup.py` | LR rises over warmup then decays; total steps match `epochs × steps_per_epoch` |
| `test_augmentation_persistent_workers.py` | two epochs differ with `persistent_workers=True` |
| `test_run_directory_layout.py` | after two epochs, no `confusion_matrix_epoch_*.png` or `classification_report_epoch*.txt` at the run root; both folders exist and are populated |
| `test_recursive_features_parity.py` | `recursive_features.forward_deep_supervision_with_features(m, x)[:2]` is elementwise equal to the frozen `m.forward_deep_supervision(x)` in eval mode — the guard against the duplicated segment loop drifting |
| `test_frozen_files_untouched.py` | **the invariant** — `git diff --name-only <base>` intersected with the FROZEN list is empty |

### Regression guard (non-negotiable)

```
pytest test_v15_model_changes.py test_representation_sensitivity.py \
       test_class_weight_power.py test_torch_compile.py test_run_naming.py -q
```

All must stay green — necessary, but no longer sufficient.

**The freeze check is the real proof.** Green tests show v15 still *works*; only
this shows it is *unchanged*:

```sh
git diff --name-only HEAD -- \
  train_example_v6.py train_example_v7.py train_example_v14.py train_example_v15.py \
  gmedmamba.py gmedmamba_ema.py gmedmamba_fullchannel.py gmedmamba_efficient.py \
  'training/trainerg_v*.py' training/gan.py training/eval_utils.py training/losses.py \
  training/numerical_stability.py training/reconstruction_head.py \
  training/spectral_recon_metrics.py training/plots.py training/augmentation.py \
  training/dataloader_config.py training/run_naming.py 'training/prep/*' \
  scripts/shallow_baseline.py 'prepare_histologyhsi_bc_v*.py' 'prepare_pad_ufes_20_v*.py'
```

**Must print nothing.** `test_frozen_files_untouched.py` runs the same check in
CI. Note `training/losses.py` is on the list but already carries *your*
uncommitted `class_weight_power` work — set the diff base past that commit, not
at `HEAD`, once it is committed.

### End-to-end smoke, before any long run

```
<python> train_example_v16.py \
  --data_dir data/hsi_v7-80_10_10_importance/hsi \
  --architecture recursive --normalization per_patch_zscore \
  --recon_mode latent --epochs 2 \
  --train_subsample_frac 0.002 --val_subsample_frac 0.01 \
  --sensitivity_check on --eval_test off
```

Check, in order:

1. Preflight prints `normalization=per_patch_zscore` and G7/G8 as **PASS** with margins.
2. `split_drift_report.json` names patient 68 and its post-norm mean is now within tolerance.
3. `history.csv` has **non-zero** `val_mse_loss` and `val_sam_loss`.
4. `loss_components.csv` exists.
5. `reconstruction/` is non-empty and `reconstruction_loss.png` is not flat-zero.
6. `per_patient_metrics.json` exists and lists 5 validation patients.
7. The run root has **no** `confusion_matrix_epoch_*.png` and no
   `classification_report_epoch*.txt`; `confusion_matrix/` and
   `classification_reports/` exist and are populated.
8. Re-run identically with `--seed 42` → byte-identical `history.csv`.

Then gate G4 against the ceiling:
`<python> scripts/shallow_baseline.py --data_dir data/hsi_v7-80_10_10_importance/hsi --normalization per_patch_zscore`

### The headline check

Run 1 must show, on the 2,848 patient-68 IDC validation patches, dark-slide
recall **staying** near 0.94 across all epochs (not 0.980 → 0.079), with test
macro-F1 **above** Run B's 0.8130. If recall holds but test macro-F1 does not
beat 0.8130, `per_patch_zscore` has not bought back what `per_sample_minmax`
cost, and Stage 1.5's gain-corrected re-prep becomes load-bearing rather than
belt-and-braces.
