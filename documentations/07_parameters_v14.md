# `train_example_v14.py` — Complete Parameter Reference

> **File:** [`train_example_v14.py`](../train_example_v14.py) · ~910 lines · training entry point (Stage G + the TRM recursive variant)
> **One‑line summary:** every CLI flag `train_example_v14.py` accepts — what it does, what its options mean, when to change it, and **which architectures actually honour it**.
> **Supersedes:** [`02_train_example_v13.md`](02_train_example_v13.md) §6 for v14. That reference predates `--architecture recursive` and all nine `--trm_*` flags, and its `--seed` default is stale. Everything else in `02_…` (the 10 startup gates, failure taxonomy, exit codes) still applies unchanged.

---

## Table of contents

1. [How to read this document](#1-how-to-read-this-document)
2. [Flag reference](#2-flag-reference)
3. [Architecture applicability matrix](#3-architecture-applicability-matrix)
4. [Cross‑flag interactions that surprise](#4-cross-flag-interactions-that-surprise)
5. [The recursive / TRM cost model](#5-the-recursive--trm-cost-model)
6. [Defaults changed in this revision](#6-defaults-changed-in-this-revision)
7. [Known gaps and dead flags](#7-known-gaps-and-dead-flags)

For ready‑to‑run commands, see [`08_commands.md`](08_commands.md).

---

## 1. How to read this document

Three things are worth knowing before the tables.

**Not every flag applies to every architecture.** `train_example_v14.py` drives four different
model families through one parser. Roughly a dozen flags are **silent no‑ops** on some of
them — accepted, recorded in `config.json`, printed in the preflight, and doing nothing.
Section 3 is the authoritative map; check it before concluding a flag "had no effect".

**Parameter count is not the cost model.** `--architecture recursive` builds a 0.44 M‑parameter
model that is *slower and more memory‑hungry per step* than the 27.4 M‑parameter `split`. It
trades parameters for recursion depth. Section 5 explains the arithmetic.

**Five defaults changed** relative to `train_example_v13.py`. If you are comparing an old
`config.json` against a new one, section 6 lists exactly what moved and why.

Legend used throughout: **S** = `split`, **F** = `fullchannel`, **E** = `efficient`,
**R** = `recursive`.

---

## 2. Flag reference

Grouping follows the parser's own comment blocks in
[`build_arg_parser()`](../train_example_v14.py).

### 2.1 Core

| flag | type | default | choices | what it does | when to change it |
|---|---|---|---|---|---|
| `--data_dir` | str | *required* | — | Directory holding `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`. `X_val.npy`/`y_val.npy`, `wavelengths.npy` and `class_names.json` are optional. | Always. |
| `--epochs` | int | `10` | — | Maximum epochs. `CosineAnnealingLR`'s `T_max` is set from this, so changing it changes the LR schedule shape, not just its length. | Set once, up front — restarting with a different value gives a different schedule. |
| `--batch_size` | int | **`256`** | — | Samples per step, for both train and validation loaders. | 256 is the measured sweet spot on 11×11 patches. Lower only for VRAM; raising to 512 buys little (see §4.5). |
| `--lr` | float | `1e-4` | — | AdamW learning rate. | `3e-4` trains faster but was measured to collapse to a single class in epoch 1 on `data/pad_v6/`. Stay at `1e-4` unless you are watching for collapse. |
| `--disc_lr` | float | `1e-4` | — | Separate LR for the GAN discriminator. | Only relevant with `--use_gan`. |
| `--resume` | str | `None` | — | Path to a **checkpoint file**, e.g. `experiments/<run>/checkpoints/latest.pt`. The experiment directory is derived as that path's `.parent.parent`. | Continuing a run. See §4.6 for the depth trap. |
| `--seed` | int | `42` | — | Seeds `torch`, `numpy`, the sampler, augmentation and `--train_subsample_frac`. | Multi‑seed runs. **Does not reach DataLoader workers** — see §7. |

### 2.2 Architecture

| flag | type | default | choices | what it does | when to change it |
|---|---|---|---|---|---|
| `--architecture` | str | `split` | `split`, `fullchannel`, `efficient`, `recursive` | Selects the model family and, with the auto‑detected modality, the `(architecture, modality)` preset in [`training/config_presets.py`](../training/config_presets.py). | See the table below. |
| `--fusion_type` | str | `se_gate` | `film`, `gated`, `cross_attention`, `multiplicative`, `residual`, `se_gate`, `eca`, `none` | How the spectral context modulates the spatial features. | **`efficient` only** — silently ignored elsewhere (§3). The last three values (`se_gate`, `eca`, `none`) currently **fail validation everywhere**, including on `efficient`, where `se_gate` is the default — see §7.1. |

What each architecture is:

| value | params (RGB) | what it is | measured ms/step (bs 256, bf16) |
|---|---|---|---|
| `split` | 27.4 M | The 4‑stage hierarchical backbone, `patch_size=4`. The established baseline. | ~58 (at bs 150) |
| `fullchannel` | ~27 M | As `split`, but the spectral pathway sees all channels jointly rather than split into groups. | not benchmarked |
| `efficient` | — | **Currently broken — see §7.1.** Intended to be compact heads + the extended fusion registry (`se_gate`, `eca`, `none`); in fact it builds exactly the same model as `fullchannel`, and crashes with its own default `--fusion_type`. | n/a |
| `recursive` | **0.44 M** | TRM‑style ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)) weight‑shared core applied recursively, `patch_size=1`. Fewest parameters, **most compute** — see §5. | ~500 |

### 2.3 TRM recursive variant

All nine flags below are **`recursive` only**; they are silent no‑ops on S/F/E because
`trm_kwargs` is left `None`.

| flag | type | default | choices | what it does | when to change it |
|---|---|---|---|---|---|
| `--trm_mixer` | str | **`mlp`** | `ss2d`, `mlp`, `attention` | The token mixer inside the shared core. `mlp` = depthwise 3×3 conv + GEGLU. `attention` = plain multi‑head self‑attention over the `Hp*Wp` tokens. `ss2d` = the selective scan. | **Leave at `mlp`.** `ss2d` routes through `_selective_scan_pure_pytorch`, a Python loop over all 121 timesteps that the core calls 168× per step — **measured 39× slower** (66.9 s/step vs 1.7 s). Only for a deliberate ablation, or after registering a real CUDA kernel via `medmamba_ss_trm.register_scan_backend()`. |
| `--trm_deep_supervision_steps` | int | **`3`** | — | ACT segments per batch. Each produces its own logits; the classification loss is their mean. | Raising it multiplies the whole step cost linearly (502 ms at 3 → 635 ms at 4). Lower to 2 for fast sweeps. |
| `--trm_n_latent` | int | `6` | — | Latent `z` updates per improve step (TRM `L_cycles`). | Part of the depth budget (§5). 3 for fast sweeps. |
| `--trm_n_improve` | int | `3` | — | Improve steps per segment (TRM `H_cycles`). `n_improve - 1` of them run under `torch.no_grad()`. | Part of the depth budget (§5). |
| `--trm_dim` | int | `128` | — | Fixed working width `d` of the recursive core. | Must be divisible by 4 (2‑D sinusoidal PE). The main capacity knob. |
| `--trm_core_layers` | int | `2` | — | Layers inside the shared core `f`. | Multiplies mixer calls per `f` application. |
| `--trm_ema_rate` | float | `0.999` | — | EMA decay over weights. Validation and `best_model_ema.pt` use the averaged copy; `0` disables. | TRM reports EMA matters for stability on small datasets. Leave on. |
| `--trm_halting` / `--no_trm_halting` | flag | on | — | Adds a learned halt head and a BCE halting loss (weight 0.5). | `--no_trm_halting` to isolate the classification objective. The head is always built; only the loss term is gated. |

Four `MedMambaSSTRMConfig` fields have **no CLI flag** — `trm_two_state`, `trm_ffn_mult`,
`trm_halt_exploration_prob`, `recursive` — and can only be changed by editing
[`training/config_presets.py`](../training/config_presets.py).

### 2.4 Reconstruction and GAN

| flag | type | default | choices | what it does | when to change it |
|---|---|---|---|---|---|
| `--recon_mode` | str | `latent` | `latent`, `raw_input`, `none` | `latent` wraps the model so a decoder reconstructs the input from the backbone's learned feature map. `raw_input` uses the older decoder that reads the raw input. `none` is classification only. | Use `none` for a pure classification benchmark — it removes the decoder, MSE and SAM from every step (~24% of step time). **`raw_input` + `recursive` silently disables deep supervision** (§4.1). |
| `--lambda_mse` | float | `1.0` | — | Weight on the reconstruction MSE. | — |
| `--lambda_sam` | float | `0.1` | — | Weight on the Spectral Angle Mapper loss. | **Set to 0 on RGB.** SAM measures the angle between spectral signature vectors; over C=3 that is hue similarity, not a spectral quantity. Applied by default. |
| `--lambda_gan` | float | `0.0` | — | Adversarial loss weight. Any value > 0 also enables the discriminator. | — |
| `--use_gan` | flag | off | — | Attaches a `SpectralDiscriminator`. If `--lambda_gan` is still 0 it is bumped to 0.05. | Rarely; adds a second optimizer and backward per step. |
| `--spectral_dropout` | float | `0.0` | — | Train‑time random zeroing of input bands, inside the recon wrapper. | Ablation only; needs `--recon_mode latent`/`raw_input` to have an effect. |

### 2.5 Loss, class imbalance and sampling

| flag | type | default | choices | what it does | when to change it |
|---|---|---|---|---|---|
| `--loss` | str | `ce` | `ce`, `weighted_ce`, `focal`, `focal_weighted` | Classification objective. The `*_weighted` variants compute per‑class weights from `y_train`. | Use `weighted_ce` when the split is imbalanced **and** you are checkpointing on a macro metric. Do not stack it with a sampler (§4.3). |
| `--focal_gamma` | float | `2.0` | — | Focal loss focusing parameter. | Only with `focal`/`focal_weighted`. Note focal shrinks the reported loss magnitude — do not compare it against a CE curve. |
| `--weight_method` | str | `balanced` | `balanced`, `inverse` | How class weights are derived. | — |
| `--sampler` | str | `none` | `none`, `balanced`, `moderate_oversample` | `balanced` = `WeightedRandomSampler` to full balance (forces `shuffle=False`). `moderate_oversample` = duplicate minority indices up to a percentile of the class counts. | `moderate_oversample` is the gentler option. It **lengthens the epoch** (§4.3). |
| `--oversample_target_percentile` | float | `50.0` | — | The percentile of class counts that `moderate_oversample` tops minority classes up to. 50 = median. | Raise toward 100 to approach full balancing. |
| `--train_subsample_frac` | float | `1.0` | — | Train on a random fraction of the training split each epoch. Applied *after* `--sampler`, deterministic given `--seed`. | When an "epoch" is dominated by redundancy — `data/pad_v6/` is 674 patches tiled from each of 1,099 images, `data/hsi_v7/hsi/` is 2.59 M patches. **Report the value alongside any metric.** |

### 2.6 Augmentation

`--augment_preset` selects a bundle; `custom` reads the eleven individual knobs below it.

| flag | type | default | what it does |
|---|---|---|---|
| `--augment_preset` | str | `none` | `none`, `light`, `medium`, `custom`. **`light`/`medium` are HSI presets** — on RGB they destroy a colour channel (§4.4). |
| `--spectral_noise_std` | float | `0.0` | Additive Gaussian noise on band values. |
| `--spectral_scale_range` | float | `0.0` | Random multiplicative gain. |
| `--spectral_offset_std` | float | `0.0` | Random additive offset. |
| `--band_dropout_prob` / `--band_dropout_max_frac` | float | `0.0` / `0.1` | Zero a scattered subset of bands. **On C=3 this always zeroes exactly one RGB channel** (§4.4). |
| `--spectral_mask_prob` / `--spectral_mask_max_width_frac` | float | `0.0` / `0.1` | Zero one *contiguous* wavelength region (dead detector segment). |
| `--flip_h_prob` / `--flip_v_prob` / `--rotate90_prob` | float | `0.0` | Geometric augmentation. **Safe on every modality** — tissue patches have no canonical orientation. |
| `--crop_scale_max_frac` | float | `0.0` | Random crop‑and‑rescale. |

For RGB, the safe custom bundle is
`--augment_preset custom --flip_h_prob 0.5 --flip_v_prob 0.5 --rotate90_prob 0.5`.

### 2.7 Optimizer, regularization, checkpoint selection

| flag | type | default | what it does | when to change it |
|---|---|---|---|---|
| `--weight_decay` | float | `0.05` | AdamW weight decay. | — |
| `--drop_path_rate` | float | `None` | Stochastic depth, linearly scheduled across backbone blocks. | **Silent no‑op on `recursive`** (§3) — yet it is still printed in the `[architecture]` line as if it applied. |
| `--classifier_dropout` | float | `None` | Dropout in the classification head. | **`efficient` only, and it raises `ValueError` elsewhere.** On `efficient` itself it is accepted and then silently discarded, because the head actually built is `ClassificationHead` (hardcoded dropout 0.1), not `CompactClassificationHead` — §7.1. |
| `--activation` / `--leaky_slope` | str / float | `leakyrelu` / `0.01` | Replaces `nn.ReLU` with `nn.LeakyReLU`. | **Only does anything on `split`** — it is the only architecture with `nn.ReLU` modules. Expect `replaced 0 nn.ReLU module(s)` elsewhere; that is normal. |
| `--gradient_checkpointing` | flag | off | Recompute activations in backward instead of storing them. | Trades ~30% time for a large memory saving. **Partial no‑op on `recursive`** (§3). |
| `--checkpoint_metric` | str | `f1_macro` | Which validation metric selects `best_model.pt` and drives early stopping. | **Unvalidated free string** — a typo silently disables both (§4.2). Valid: `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `precision_weighted`, `cohen_kappa`, … |
| `--early_stop_patience` | int | `None` | Stop after N epochs without improvement in `--checkpoint_metric`. | Set it. 50 epochs over a heavily tiled dataset is deep into overfitting. |
| `--class_collapse_streak` | int | `3` | Warn when a class has had 0 recall for N consecutive epochs. | **Warning only — never stops the run.** |

### 2.8 DataLoader memory policy

| flag | type | default | choices | what it does |
|---|---|---|---|---|
| `--loader_mode` | str | **`balanced`** | `safe`, `balanced`, `performance` | Named bundle of the four fields below. `safe` = 0/None/False/False. `balanced` = 2/1/False/False. `performance` = 4/2/True/True. |
| `--num_workers` | int | `None` | — | Overrides **only** the worker count of the selected mode (§4.5). |
| `--prefetch_factor` | int | `None` | — | Batches prefetched per worker. Ignored when `num_workers == 0`. |
| `--persistent_workers` / `--no_persistent_workers` | flag | `None` | — | Keep workers alive between epochs. |
| `--pin_memory` / `--no_pin_memory` | flag | `None` | — | Pinned host memory for faster H2D. No effect on CPU runs. |
| `--cpu_threads` | int | `None` | — | Caps OMP/MKL/torch threads in the main process, against oversubscription. |

Validation always gets `min(workers, workers − 1)` workers and never persistence.

### 2.9 Dataset integrity and storage

| flag | type | default | choices | what it does | when to change it |
|---|---|---|---|---|---|
| `--dataset_storage` | str | `auto` | `auto`, `mmap`, `ram` | `auto` loads into RAM only if the header‑declared payload fits comfortably in available RAM, else mmap. | **Leave on `auto`.** An explicit `ram` on a 40 GB array will thrash or die; `auto` correctly picks mmap there (verified on `data/hsi_v7/hsi/`). |
| `--dataset_validation_level` | str | `structural` | `structural`, `deep` | `structural` = header + on‑disk size + ~19 row probes per file. `deep` = additionally read every `X_*.npy` end to end. | `deep` only when you suspect bad sectors — it is a full sequential read of every file at startup (46 GB for `hsi_v7/hsi`). |
| `--skip_dataset_validation` | flag | off | — | Skips the integrity gate. Records `scientifically_qualified=false` in `config.json`. | Never, in practice. |
| `--manifest_check` | str | `auto` | `auto`, `off`, `strict` | Cross‑checks every file against `dataset_manifest.json`. `strict` also requires the manifest to exist. | `strict` for a qualification run. |
| `--check_leakage` / `--no_check_leakage` | flag | on | — | Patient‑level train/val/test overlap check. | Keep on. |
| `--abort_on_leakage` / `--allow_leakage` | flag | abort | — | Whether leakage is fatal. | Keep aborting. |
| `--check_class_coverage` / `--no_check_class_coverage` | flag | on | — | Verifies every class appears in every split. | Keep on. |
| `--allow_missing_classes` | flag | off | — | Downgrade a missing class from fatal to a warning. | Only as a documented dataset limitation. |

### 2.10 Numerical stability and AMP

| flag | type | default | choices | what it does | when to change it |
|---|---|---|---|---|---|
| `--amp` | str | **`bf16`** | `off`, `bf16`, `fp16`, `auto` | Mixed precision for the training pass. | `bf16` is ~3.8× faster than `off` and, unlike fp16, has fp32 exponent range so it needs no loss scaling. It is **auto‑downgraded to `off`** on a GPU without bf16 support. Use `off` when debugging non‑finite losses. |
| `--max_gradient_norm` | float | `1.0` | — | Gradient clipping norm. | — |
| `--max_gradient_skip_ratio` | float | `0.10` | — | Epoch is "unhealthy" above this fraction of skipped updates. | — |
| `--max_consecutive_bad_batches` | int | `3` | — | **Aborts the run** after N consecutive non‑finite batches. | This is the main way a run dies mid‑training. Far more likely with `--amp fp16` than `bf16`. |
| `--max_consecutive_unhealthy_epochs` | int | `2` | — | Aborts after N consecutive unhealthy epochs. | — |
| `--debug_numerics` | flag | off | — | Per‑batch non‑finite diagnostics. | When chasing a `TRAINING_ABORTED_NUMERICAL_INSTABILITY`. |
| `--numerical_smoke_test` | str | `auto` | `auto`, `on`, `off` | One forward/backward/step on a throwaway model copy before training. `auto` = on with `--safe_mode`/`--debug_numerics`. | `on` before a long run — it costs seconds and catches a dead configuration. |

### 2.11 One‑shot modes

| flag | type | default | what it does |
|---|---|---|---|
| `--loader_test` / `--loader_test_batches` | flag / int | off / `5` | Build the dataset + DataLoader, read N batches, report shape/dtype/timing/RAM, exit **without building the GPU model**. The fastest way to check a new `--data_dir`. |
| `--safe_mode` | flag | off | Forces `loader_mode=safe`, `amp=off`, `gradient_checkpointing=on`, validation on, `debug_numerics=on`, 0 workers. **Any of these you pass explicitly now wins** — `--safe_mode --amp bf16` gives you bf16 (this silently failed before). `--skip_dataset_validation` remains the one non‑overridable field. |

---

## 3. Architecture applicability matrix

The column marks what the flag does for that architecture: **yes** = applies, **no‑op** =
accepted and silently ignored, **error** = raises.

Note the `efficient` column describes *intent*. As shipped, `efficient` constructs the same
model as `fullchannel` (§7.1), so its column is currently aspirational.

| flag | S `split` | F `fullchannel` | E `efficient` | R `recursive` |
|---|---|---|---|---|
| `--fusion_type` | **no‑op** | **no‑op** | yes | **no‑op** |
| `--drop_path_rate` | yes | yes | yes | **no‑op** |
| `--classifier_dropout` | **error** | **error** | yes | **error** |
| `--activation` / `--leaky_slope` | yes | **no‑op** | **no‑op** | **no‑op** |
| `--gradient_checkpointing` | yes | yes | yes | **partial** |
| all nine `--trm_*` | **no‑op** | **no‑op** | **no‑op** | yes |
| everything else (loss, sampler, augmentation, loader, AMP, dataset, stability) | yes | yes | yes | yes |

Why each of these is the way it is:

- **`--fusion_type`** — the call site passes it only when `architecture == "efficient"`, else
  `None`. `build_model` *would* raise, but the ternary pre‑empts that. Worse, the value is
  **still written into `config.json` as `"fusion_type": "eca"`** even when it was ignored, so
  the run's own provenance record is misleading. Read `--fusion_type` from `config.json` only
  when `architecture == "efficient"`.
- **`--drop_path_rate`** — `RecursiveMambaBlock` contains no `DropPath` module and the
  recursive backbone never builds a depth schedule. The value is nonetheless accepted, stored
  on the config, **printed in the `[architecture] … (drop_path_rate=0.2)` log line**, and
  written to `config.json`. It changes nothing.
- **`--classifier_dropout`** — the only misapplied flag that fails loudly, because it is passed
  through unguarded. `medmamba_ss_trm.ClassificationHead`'s dropout is hardcoded at 0.1.
- **`--activation`** — the patch replaces `nn.ReLU` modules, and the only ones in the codebase
  are in `split`'s `GBlock.conv_branch`. `fullchannel`/`efficient` already hardcode
  `LeakyReLU`; `recursive` uses SiLU and GELU. `[activation-patch] replaced 0 nn.ReLU
  module(s)` is the expected output on F/E/R, not a bug. `--activation relu` is a no‑op
  everywhere — there is no reverse patch, it just means "skip the patch".
- **`--gradient_checkpointing` on `recursive`** — *partial*. The backbone half early‑outs
  (the recursive backbone has no `.stages`), the spectral half still fires, and
  `RecursiveCore.f` **checkpoints itself unconditionally** whenever training with grad. So
  core checkpointing is always on for `recursive` and the flag cannot turn it off. This is
  load‑bearing, not incidental — see §5.

---

## 4. Cross‑flag interactions that surprise

### 4.1 `--recon_mode raw_input` + `--architecture recursive` silently disables deep supervision

`MedMambaSSTRMRawReconWrapper` does not expose the per‑segment logits list through its `forward`,
so the trainer's `_deep_supervision_base()` returns `None` and training falls back to the
ordinary single‑forward path. The model still trains, and nothing warns you — it just is not
the TRM training procedure the `--trm_deep_supervision_steps` and `--trm_halting` flags
describe. Use `latent` or `none` with `recursive`.

With `--recon_mode latent` + `recursive`, note the decoder reads the **detached** final
feature map, so reconstruction trains the decoder but no longer backpropagates into the
recursive core.

### 4.2 `--checkpoint_metric` typos fail silently, twice

The value is a free string, looked up with `metrics.get(...)`. A typo returns `None`, and both
the checkpoint‑selection path and the early‑stopping path then skip without complaint. The
result is a run where **`best_model.pt` is never written and `--early_stop_patience` never
fires**, with no error anywhere in the log. Check the first epoch's output actually says
`Saved: … (BEST)`.

### 4.3 `--sampler` and `--loss weighted_ce` both correct imbalance — pick one

`moderate_oversample` duplicates minority indices up to the median class count; `weighted_ce`
scales the per‑class loss. Doing both double‑corrects and can push the model to over‑predict
the minority class. Choose sampler‑level *or* loss‑level rebalancing.

Also note `moderate_oversample` only ever **adds** indices, so it **lengthens** the epoch —
on `data/pad_v6/` from 740,800 to ~817,200 samples (+10%).

### 4.4 `light` / `medium` augmentation destroys an RGB channel

`band_dropout` computes `n_drop = max(1, round(C * max_frac * random()))`. With C=3 and
`max_frac=0.05`, the rounded term is always 0, so `max(1, 0)` → **1**: whenever it fires
(p=0.1 for `light`, p=0.2 for `medium`) it zeroes one of the three RGB channels. For
pigmented‑lesion classification, where colour is the primary diagnostic cue, that is
destructive, not regularising. `spectral_mask` and `spectral_noise` are equally
HSI‑oriented. On RGB, use the geometric flags only.

### 4.5 `--num_workers` alone leaves the rest of the loader at the mode's defaults

The four loader fields are resolved independently: passing `--num_workers 8` with the default
mode gives you 8 workers **and the mode's** prefetch/persistence/pinning. Under the old `safe`
default that meant 8 non‑persistent, non‑pinned workers re‑forked every epoch. Pass
`--loader_mode performance` to set all four coherently, or override each field explicitly.

Related: raising `--batch_size` past ~256 amortises less than expected, because
`SpectralPathway` processes pixel‑spectra in fixed chunks of 1024, so its sequential chunk
count grows *linearly* with the batch (B=150 → 18 chunks, B=512 → 61).

### 4.6 `--resume` expects a path two levels deep

The experiment directory is computed as `Path(--resume).resolve().parent.parent`, i.e. the
argument must be `<exp_dir>/checkpoints/<file>.pt`. A path one level deep silently yields
`experiments/` itself as the experiment directory and scatters artifacts there. There is no
existence check at that point either: a typo'd path gives you a fresh run writing into a
garbage directory, because the "resume path not found, starting fresh" warning happens later
and independently.

---

## 5. The recursive / TRM cost model

`--architecture recursive` has 0.44 M parameters against `split`'s 27.4 M, and is **slower**.
Weight sharing reduces storage, not work.

The shared core `f` is applied:

```
f-calls per improve step  = trm_n_latent + 1
f-calls per segment       = × trm_n_improve
f-calls per training step = × trm_deep_supervision_steps
```

At the **old** defaults (`n_latent 6`, `n_improve 3`, `sup 4`) that is `7 × 3 × 4` = **84 core
applications per training step**, each running `trm_core_layers` = 2 mixer blocks — 168 mixer
calls. At the **current** defaults (`sup 3`) it is 63. `split`, by contrast, makes one pass
over a 3×3 token grid (`patch_size=4` on an 11×11 patch), while `recursive` makes dozens of
passes over 121 tokens (`patch_size=1`).

With `--trm_mixer ss2d`, each of those 168 calls additionally runs a Python loop over all 121
timesteps issuing ~5 CUDA kernels apiece — on the order of 100,000 kernel launches per step.
The GPU idles while one CPU core issues them. Measured: **66.9 s/step**, versus 1.7 s with
`mlp`.

**Checkpointing is mandatory, not optional.** Measured at batch 256 with `mlp` + bf16:

| `--trm_deep_supervision_steps` | core checkpointed | result |
|---|---|---|
| 3 | yes (current behaviour) | 502 ms/step, 1.80 GB |
| 3 | no | **CUDA OOM** on a 16 GB card |
| 4 | yes | 635 ms/step, 1.94 GB |
| 4 | no | **CUDA OOM** |

A 0.44 M‑parameter model OOMs 16 GB without checkpointing. That is the clearest statement of
what this architecture actually costs.

---

## 6. Defaults changed in this revision

Five parser defaults moved. All are throughput changes; no objective, sampling or
augmentation default was touched, because those are dataset‑specific choices that belong in a
command, not a global default.

| flag | was | now | why |
|---|---|---|---|
| `--trm_mixer` | `ss2d` | **`mlp`** | 39× measured. `medmamba_ss_trm.py`'s own docstring already recommended `mlp`; only the default disagreed. |
| `--amp` | `off` | **`bf16`** | 3.8× measured. `off` was correct while the v13 plan was debugging non‑finite gradients; that work is done. Auto‑downgrades on unsupported hardware. |
| `--batch_size` | `128` | **`256`** | Measured throughput sweet spot on 11×11 patches. |
| `--trm_deep_supervision_steps` | `4` | **`3`** | 1.27×, keeps three supervision signals. |
| `--loader_mode` | `safe` | **`balanced`** | `safe` means 0 workers. `balanced` is 2 workers with prefetch 1; `performance` remains opt‑in per the v12 plan. |

Net effect on a bare‑defaults `--architecture recursive` run over `data/pad_v6/`:
**~92 h/epoch → ~26 min/epoch.**

Unchanged and deliberately so: `--recon_mode latent`, `--loss ce`, `--sampler none`,
`--augment_preset none`, `--lr 1e-4`, `--dataset_storage auto`.

`--safe_mode` still forces the old conservative bundle, and now honours explicit overrides.

---

## 7. Known gaps and dead flags

### 7.1 `--architecture efficient` does not build the efficient model

Verified by construction, not inferred. Three linked defects:

```
>>> build_model("efficient", "rgb", 6, fusion_type="se_gate")     # its OWN default
AssertionError: unknown fusion_type 'se_gate'.
                Valid: ['cross_attention', 'film', 'gated', 'multiplicative', 'residual']

>>> a = build_model("fullchannel", "rgb", 6)
>>> b = build_model("efficient",   "rgb", 6, fusion_type="film")
>>> type(b.backbone).__name__, type(b.head).__name__
('FullChannelMedMambaSSBackbone', 'ClassificationHead')       # not the Efficient* classes
>>> sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
True                                                          # 34,995,373 both
>>> a.state_dict().keys() == b.state_dict().keys()
True                                                          # structurally identical
```

**Root cause.** `medmamba_ss_efficient.py` follows a factory-override pattern — `MedMambaSSEfficient`
sets `_backbone_factory = EfficientMedMambaSSBackbone` and
`_classification_head_factory = CompactClassificationHead`. But `MedMambaSSFullChannel.__init__`
never reads those attributes: it hardcodes `self.backbone = FullChannelMedMambaSSBackbone(self.cfg)`
and `self.head = ClassificationHead(...)`. Both override hooks are dead code.

**Consequences.**

1. `--architecture efficient` is a **silent alias for `--architecture fullchannel`**. Any result
   attributed to the "efficient" architecture is a `fullchannel` result.
2. The extended fusion types exist only in `validate_extended_config`, which is reached through
   `EfficientMedMambaSSBackbone._validate_config` — a class that is never instantiated. So
   `se_gate`, `eca` and `none` fail `MedMambaSSTRMConfig.validate()` everywhere. Since `se_gate` is
   the parser default, **`--architecture efficient` crashes out of the box**; it only runs if you
   also pass one of the five base fusion types.
3. `--classifier_dropout` is gated to `efficient` (raising elsewhere) and then discarded there,
   because `CompactClassificationHead` is never built.

**Not fixed here** — this is a model-assembly bug in `medmamba_ss_fullchannel.py`, outside the
scope of the defaults/documentation work this revision covers. The fix is to make
`MedMambaSSFullChannel.__init__` dispatch through `self._backbone_factory` and
`self._classification_head_factory` instead of the hardcoded classes. Until then, treat
`--architecture efficient` as unavailable and use `split`, `fullchannel` or `recursive`.

### 7.2 Other known gaps

Behaviours worth knowing about that are not bugs this reference can fix:

| what | where | effect |
|---|---|---|
| `--seed` never reaches DataLoader workers | `make_worker_init_fn()` is called with no arguments, so `base_seed` is always 0 | Worker‑side augmentation RNG is identical across seeds. Multi‑seed runs vary less than they appear to. |
| Validation ignores `--amp` | the validation pass uses the legacy "bf16‑if‑supported" logic rather than `amp_mode` | `--amp off` still validates in bf16. Documented in the trainer's own comment. |
| `GradScaler` is always constructed | never gated on `amp_mode` | Runs even for `--amp off` and `--amp bf16`, where dynamic loss scaling is unnecessary. Harmless — bf16 cannot overflow the way fp16 does — but it adds a `found_inf` reduction and a sync per step. |
| Per‑batch full‑model numeric audits | `inspect_gradients` + `check_parameters_finite` run every batch unconditionally | ~400 forced device syncs per step. Measured cost 2.5% on the 0.44 M recursive model; larger on `split`. |
| `X_val.npy` missing → **test set used as validation, silently** | `discover_data` falls back to `X_test`/`y_test` with no warning | Affects `data/hsi_v7/hsi/` and `data/hsi_v7/rgb/`. Checkpoint selection then happens on test data. See [`08_commands.md`](08_commands.md) §B. |
| Modality detection is a single filename check | `wavelengths.npy` at the top level of `--data_dir` | `selected_wavelengths.npy` and `wavelengths_full.npy` do **not** count. A spectral dataset with a differently named wavelength file trains silently as RGB. |
| Wavelengths never reach the model | the trainer calls `self.model(x)` without them | `--architecture` presets are wavelength‑capable, but wavelength metadata is currently used only for reconstruction spectral metrics. |
