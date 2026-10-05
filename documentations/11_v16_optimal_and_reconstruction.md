# v16 — Current Scripts, Measured Results, GPU Budget, and Reconstruction

> **2026-09-28:** the commands below were rewritten from `train_example_v16*.py` / `train_example_v18.py` / `scripts/v20_queue.sh` to `train.py --profile …` / `run_experiments.py`, with the same flags. The two resolve every argument identically (`tests/test_train_parity.py`; see [`17_train_and_sweeps.md`](17_train_and_sweeps.md)). The originals are in `archive/`. A pre-2026-09-28 `train_example_v16_optimal_recon.py` command is `--profile paper_recipe`. **2026-10-01:** the profiles were renamed for what they are for (`v18` → `paper_recipe`, `optimal` → `pad_ufes_best_norecon`, `original` → `medmamba_protocol_norecon`, `base` → `pipeline_defaults`); `_norecon` marks the two without reconstruction. The old names still work.


> **Documents the files that are actually current**, which docs 01–10 do not:
> [`train_example_v16_optimal.py`](../train_example_v16_optimal.py),
> [`train_example_v16_original.py`](../train_example_v16_original.py),
> [`train_example_v16_recon.py`](../train_example_v16_recon.py),
> [`train_example_v16_optimal_recon.py`](../train_example_v16_optimal_recon.py),
> [`prepare_pad_ufes_20_optimal.py`](../prepare_pad_ufes_20_optimal.py),
> [`prepare_histologyhsi_bc_v8.py`](../prepare_histologyhsi_bc_v8.py),
> [`training/trainerg_v13.py`](../training/trainerg_v13.py),
> [`training/recon_artifacts.py`](../training/recon_artifacts.py).
>
> **Hardware for every measurement below:** NVIDIA GeForce RTX 5060 Ti, 16,651 MB
> VRAM (Blackwell, `sm_120`), torch 2.11.0+cu128, CUDA 12.8, 32 GB RAM, 16 cores,
> 16 GB `/dev/shm`.
>
> **Everything numeric here is read out of `experiments/`**, not estimated. The run
> directory is named in each table so any row can be checked.

---

## 0. Which file is current

Docs 01–10 stop at v15 / v13 / v6-prep. This is the current set:

| role | current file | what docs 01–10 say | status of the older doc |
|---|---|---|---|
| model | `medmamba_ss_trm.py` | [`01_medmamba_ss_trm_model.md`](01_medmamba_ss_trm_model.md) | **still accurate** — `medmamba_ss_trm.py` is FROZEN |
| training core | `train_example_v16.py` (+ `TrainerG_v12`) | [`10_commands_v15.md`](10_commands_v15.md) §"v16" | accurate |
| training, best metrics | **`train_example_v16_optimal.py`** | — | **undocumented before this file** |
| training, protocol fidelity | **`train_example_v16_original.py`** | — | **undocumented before this file** |
| training + recon artifacts | **`train_example_v16_recon.py`** | — | **undocumented before this file** |
| both of the above two | **`train_example_v16_optimal_recon.py`** | — | **undocumented before this file** |
| HSI prep | **`prepare_histologyhsi_bc_v8.py`** | [`03_…_v6.md`](03_prepare_histologyhsi_bc_v6.md) documents v6 | v6 doc still describes the shared core; v8's three additions are §2 below |
| PAD prep | **`prepare_pad_ufes_20_optimal.py`** | [`04_…_v5.md`](04_prepare_pad_ufes_20_v5.md) documents v5 | v5 doc describes the shared core; the two changed defaults are §3 below |
| CLI reference | `07_parameters_v14.md` | v14 flags | **stale** — every v15/v16 flag is missing; read the `--help` of the entry point you are running |

The repo convention is **new versioned files, never in-place edits**. `test_frozen_files_untouched.py`
enforces it: `medmamba_ss_trm.py`, `train_example_v15.py`, every `training/trainerg_v3..v11.py`,
`training/dataloader_config.py`, `training/augmentation.py`, `training/losses.py`, all
`training/prep/`, and every earlier prep generation are FROZEN and must not be edited. The
`*_optimal` / `*_original` / `*_recon` entry points therefore work by **importing
`train_example_v16.py` and passing `v16.main()` a `v16.Overrides` value** (parser, run-name
builder, model builder, trainer class) — not by copying its `_main`. Before v17 S9 they
*rebound* those four names inside v16's module namespace, up to three layers deep, each
layer overwriting the last; `build_overrides` replaced that with a value the composition can
be read off. `install_overrides` survives as a deprecated shim, and
`train_example_v16_original.py` still uses it, for the reason given in its own docstring.

---

## 1. Does it give good results? — measured

### 1.1 HSI vs RGB, matched pair (the RQ0 answer)

Same dataset root, same architecture, same everything except the modality. This is a
genuine controlled comparison — batch size, LR, epochs, seed, loss, class-weight power,
normalization and all TRM hyperparameters are identical:

| | HSI | RGB | RGB (repeat) |
|---|---|---|---|
| run | `20260906_222725_…-hsi_…` | `20260907_025027_…-rgb_…` | `20260909_221520_…-rgb_…` |
| data | `data/hsi_v8-80_10_10_importance/hsi/` | `…/rgb/` | `…/rgb/` |
| batch / lr / epochs / seed | 256 / 1e-4 / 20 / 42 | 256 / 1e-4 / 20 / 42 | 256 / 1e-4 / 20 / 42 |
| **test accuracy** | **0.9436** | 0.8904 | 0.8932 |
| **balanced accuracy** | **0.9073** | 0.7571 | 0.7615 |
| **macro-F1** | **0.8580** | 0.7282 | 0.7332 |

`n_test = 348,894` patches, 3 classes (healthy / DCIS / IDC). Per-class F1 for HSI:
healthy 0.871, DCIS 0.704, IDC 0.999 — DCIS is the hard class (24,570 test patches,
precision 0.565 against recall 0.934).

**HSI beats RGB by +14.6 points of balanced accuracy and +12.6 of macro-F1.** The two RGB
repeats differ by 0.005 macro-F1, so the gap is two orders of magnitude larger than the
run-to-run spread. This is the strongest result in the repository and it is the one the
thesis' central claim rests on.

> Caveat worth stating in the write-up: both runs use `--train_subsample_frac 0.1`, i.e.
> 10% of the training patches. The comparison is matched and therefore valid, but neither
> number is this architecture's ceiling on the full split.
>
> **Second caveat, and the more serious one: the `rgb/` arm of this dataset was built before
> `--rgb_source` existed**, so it is the `legacy` mixture of two optical sources with 29.5 %
> of it resized from a different field of view — see
> [doc 12 §B.1](12_commands_datasets.md#b1---rgb_source--read-this-before-preparing). A
> clean re-prep (`data/hsi_v8-80_10_10_importance-new`, `rgb_source: synthetic`) exists and
> its HSI arm has been run (`20260913_121925`, 0.9231 / 0.8822 / 0.8182 at 10 % subsample);
> **the matched RGB arm on it has not**, so the defensible version of this table does not
> exist yet. Quote these numbers as the `legacy`-RGB pair until it does.

### 1.2 PAD-UFES-20 (RGB dermatology) — honest summary

| run | config | test acc | balanced | macro-F1 |
|---|---|---|---|---|
| `20260906_183513_pad_optimal-undersample_…` | whole 224, focal_weighted, cwp 1.0, lr 1e-3 | 0.4564 | **0.5020** | **0.4213** |
| `20260906_121708_optimal_mil9-undersample_…` | mil9 tiles | 0.4199 | 0.4097 | 0.3768 |
| `20260906_002612_…pad_original_with-shallow_…` | whole 224, 60/10/30 | 0.5321 | 0.3216 | 0.3218 |
| whole-image linear probe (`shallow_baseline_v16.py`) | — | 0.4735 | 0.3365 | 0.3252 |
| constant (majority-class) predictor | — | 0.3459 | 0.1667 | 0.0857 |

Chance is 0.1667 balanced. The best PAD run clears the whole-image linear probe it was
built to beat (0.502 vs 0.337 balanced, 0.421 vs 0.325 macro-F1), which is the target
`train_example_v16_optimal.py`'s docstring sets. **But** the test split holds 344 images
with MEL at ~7 of them, so macro-F1 here carries a very wide error bar — report balanced
accuracy and per-class support beside it, always.

Runs on the old **11×11 patch grid (`data/pad_v6`) never exceeded 0.258 balanced / 0.228
macro-F1**, and a linear probe over all 363 pixels of such a crop scores *below* one over
3 mean channel values. An 11×11 window is 0.24% of the frame and usually misses the
lesion; that regime is an input problem, not an optimisation one, and
`prepare_pad_ufes_20_optimal.py --tiling whole` is the fix.

**Verdict.** HSI: yes, clearly good, and the HSI-vs-RGB claim is solid. PAD RGB: the
pipeline is working correctly and beats every trivial and linear baseline, but ~0.50
balanced accuracy on 6 classes is a weak absolute result, driven by 2,298 images with
MEL at 52 of them.

---

## 2. `prepare_histologyhsi_bc_v8.py` — what it adds over v6/v7

A `DatasetAdapter` subclass of v7's `HistologyHsiAdapter`. Three additions, all aimed at
storing **gain-corrected reflectance** rather than raw sensor DN:

| flag | default | what it does |
|---|---|---|
| `--capture_gain {none,median_ratio}` | `median_ratio` | per-capture flat-field residual correction. Pass 1 samples `--gain_sample_pixels` (400) ROI pixels per capture, takes a median, and sets `gain = corpus_median / capture_median`, clipped to `[--gain_clip_lo, --gain_clip_hi]` = `[0.5, 2.0]`. Anything clipped is logged as a **SUSPECT ACQUISITION**. This closes the patient-68 exposure fault (IDC captures acquired at ~50% intensity) at the source instead of leaving gate G8 to catch it at train time. |
| `--hsi_value_scale` | `auto` | divides by a recorded constant so cubes land in reflectance-like `[0, ~1.5]`. `auto` = 99th percentile of the **gain-corrected** pass-1 sample, which is what makes `float16` storage safe and roughly halves the corpus on disk. |
| `--emit_group_sidecars` | on | writes `groups_*.npy` / `captures_*.npy` / `capture_index.json` during `finalize_modality`, so `scripts/derive_patch_groups.py` becomes a fallback rather than the only source. |
| `--rgb_source {synthetic,camera,legacy}` | `synthetic` | **which of the two PNGs beside each cube becomes the RGB arm.** The frozen `_find_rgb_image` scored `RGBImage.png` and `SyntheticRGBImage.png` equally and let `os.listdir` order decide, so the RGB arm was a *mixture* of two optical sources with 29.5 % of it NN-resized from a different field of view. `synthetic` is cube-aligned in 644/644 captures and is the only choice that makes HSI-vs-RGB single-variable; `legacy` reproduces the old mixture. **The `rgb/` arm behind the §1.1 numbers was built under `legacy`** — see [doc 12 §B.1](12_commands_datasets.md#b1---rgb_source--read-this-before-preparing). |

It also fixes an inverted flag in the frozen v6/v7 calibration probe (re-derived here, not
edited there): `get_calibration_means` returns `(None, None)` *because* a cube has no
reference captures — v6/v7 read that as "already calibrated". v8 ties `hsi_all_calibrated`,
and hence the float16 decision, to whether the **actual stored values** fit under
`_FLOAT16_SAFE_MAX`. Per-capture stats land in `capture_gain_report.json`.

---

## 3. `prepare_pad_ufes_20_optimal.py` — two defaults and one option

A subclass of the FROZEN `prepare_pad_ufes_20_v6.PadUfes20Adapter`. Almost every extraction
decision in v6 was already right; what was wrong was the command line the manuscript's
datasets were built with.

1. **`--split` default `70/15/15`, not the core's `80_20`.** `80_20` is a *two-way* spec —
   no `X_val.npy` is written, `train_example_v16.discover_data` falls back to the test split
   as validation, and gate G6 then refuses to report a test number at all. The only number
   such a run can report is the one it selected on.
2. **`--tiling` (new), default `whole`** — one 224×224 sample per clinical image. This is the
   single largest measured effect on this dataset (see §1.2). `mil9` / `mil4` cut 112×112
   tiles for use with `--eval_group_aggregation mean_prob`, and write the `images_*.npy`
   sidecar that aggregation needs.

Deliberately unchanged: `--img_size 224`, bilinear resize of the full frame, EXIF-corrected
RGB float32 in [0,1]; `--balance_classes none` (imbalance is corrected in the loss, not by
discarding 86% of the data); and **patient-level grouping**, which is not optional — the
MedMamba paper's published PAD numbers use an image-level split that shares patients across
splits and inflates results.

---

## 4. GPU budget on this machine — the headroom nobody was using

`history.json`'s `GPU_memory_MB` column, read across `experiments/`:

| dataset / token grid | batch | amp | `GPU_memory_MB` | of 16,651 | s/epoch | ms/step |
|---|---|---|---|---|---|---|
| 11×11 (121 tok) HSI | 256 | bf16 | 874 | **5.2 %** | 680 | 710 |
| 11×11 (121 tok) RGB | 32 | bf16 | 201 | **1.2 %** | 595 | 78 |
| 28×28 (784 tok) PAD | 32 | bf16 | 1,432 | **8.6 %** | 46 | 900 |
| 28×28 (784 tok) PAD | 64 | **off (fp32)** | 3,674 | 22.1 % | 41 | 1,845 |

**Every current configuration uses 1–9 % of a 16 GB card.** Net of a ~200 MB floor
(CUDA context + the 0.45 M-parameter model + optimizer state), activations cost
**~0.049 MB per token-row**, where a token-row is one token of one sample. At batch 32:

| `--target_token_grid` | stem stride @224 | tokens | estimated VRAM | fits? |
|---|---|---|---|---|
| 28 *(default)* | 8 | 784 | ~1.4 GB | yes |
| 56 | 4 | 3,136 | ~5.3 GB | **yes** |
| 112 | 2 | 12,544 | ~20.5 GB | no |

The batch-64 row is the only one here that is **not** bf16 — it comes from
`20260905_235026`, an `--amp off` protocol run — so do not read it as twice the bf16 cost of
the row above it. The bf16 batch-64 figure is 2,200 MB — the `64 | on | on` row of the
§4.1 sweep, which is also compiled, and compile lowers the peak (1,428 → 1,110 MB at
batch 32). So the 28×28
default is **not** a ceiling this card imposes at any batch size these runs use, and
`train_example_v16_optimal.py` uses 32. The reason to keep 28 is **wall
clock, not memory**: token count scales step cost roughly linearly, so `--target_token_grid 56`
turns a ~2.5 h, 200-epoch PAD run into ~10 h. It is nevertheless the most promising
unexplored axis on PAD, because a stride of 8 discards 98 % of the pixels before the
recursive core sees anything, on a task whose open problem is that the input does not carry
the lesion.

`train_example_v16_optimal.py` now prints a `[gpu-budget]` block at startup with the measured
free VRAM, this estimate against it, and the percentage of the card that will sit idle.
**It changes nothing automatically** — `--target_token_grid` is the knob.

### 4.1 Measuring instead of estimating — `scripts/gpu_tune.py`

The two cost models in the repo disagree with each other and with the measurements, so
there is now a script that just measures. It builds the **real** model through the same
`training.config_presets.build_model` call `train_example_v16.py` makes, runs a few
synthetic training steps, and reports median ms/step and peak VRAM per sweep cell. An OOM
is reported as a row rather than a crash, and nothing is written to `experiments/`.

```bash
# the two knobs with real headroom, on the PAD whole-image input
python scripts/gpu_tune.py --input_side 224 --batch_size 16,32,64 --token_grid 28,56

# does torch.compile pay for itself on the 63-application core?
python scripts/gpu_tune.py --input_side 11 --in_channels 32 --num_classes 3 \
    --batch_size 256 --token_grid 11 --compile off,on

# is gradient checkpointing still worth paying for with 90% of the card idle?
python scripts/gpu_tune.py --input_side 224 --batch_size 32 \
    --token_grid 28 --checkpoint_core on,off
```

#### The answers (v17, RTX 5060 Ti, torch 2.11.0+cu128, bf16, `mixer=mlp`)

Those three questions have now been run. PAD 224×224×3 at token grid 28:

| batch | core ckpt | compile | ms/step | peak MB | samples/s |
|---|---|---|---|---|---|
| 32 | on | off | 437.0 | 1,428 | 73.2 |
| 32 | **off** | off | 384.7 | **6,863** | 83.2 |
| 32 | on | **on** | **266.7** | **1,110** | **120.0** |
| 64 | on | on | 602.4 | 2,200 | 106.2 |
| 128 | on | on | 1309.1 | 4,402 | 97.8 |

* **`torch.compile` pays for itself: 1.64×, and it *lowers* peak memory** (1,428 → 1,110 MB).
  It is now the default (`OPTIMAL_DEFAULTS["compile_model"] = "on"`, v17 S3).
* **Gradient checkpointing is still worth paying for — the opposite of what "90% of the card
  is idle" suggests.** Turning it off buys 12% for 4.8× the memory and OOMs at batch 64.
  Keep it on. (The hint in `medmamba_ss_trm.medmamba_ss_trm_default`'s docstring was written for
  `trm_mixer="ss2d"` and does not transfer to the `mlp` mixer every real run uses.)
* **More batch is slower here.** 120.0 → 106.2 → 97.8 samples/s at bs 32/64/128. The same
  was already measured for HSI in `training/torch_compile.py`: 89.5 / 80.2 / 78.6 at
  256 / 512 / 1024. The "memory headroom" the §4 budget prints is real and is **not**
  convertible into throughput on this card.

> **HSI rows from a sweep run before v17 are wrong.** The script never called
> `enable_spectral_gradient_checkpointing`, which the real path enables at
> `in_channels >= 16` — so its 11×11×32 rows read 10,157 MB against the 874 MB the
> production run recorded, and batch 512 "OOM"d spuriously. Fixed by
> `--spectral_checkpointing on|off|auto` (v17 S3); re-run any earlier HSI sweep.

> **The sweep is the optimiser, not the epoch.** It times forward + backward + `opt.step()`
> only. On the PAD baseline that is 0.437 s of a 0.852 s real step — the other 45% is
> inside the training loop and was invisible until v17 added the `history.json` timing
> fields. See [`14_performance_and_progress.md`](14_performance_and_progress.md).

Two fidelity details that make the numbers transferable, both easy to get wrong:

* It times **`forward_deep_supervision` with a loss on every segment**, matching
  `TrainerG_v10._train_one_epoch` (`v10:116`). Timing plain `forward` instead would
  under-measure the step by roughly `trm_deep_supervision_steps`, because the core detaches
  `(y, z)` between segments, so a loss on the returned last-segment logits backprops through
  one segment only.
* `--compile on` uses the repo's own `enable_torch_compile`, which rebinds
  `RecursiveCore._f_forward` and the selective scan **in place** — a bare
  `torch.compile(model)` would wrap `forward`, which the timed path bypasses, and the
  `compile=on` row would silently measure nothing. `enable_torch_compile` also swallows
  failures and returns a summary string, so the script prints that summary and refuses the
  cell if compilation did not actually engage.

**`--checkpoint_core` looked like the cheapest open win, and it is not one.**
`cfg.trm_checkpoint_core` defaults to `True`, so every one of the ~63 `RecursiveCore.f`
applications is recomputed during backward. The default was set when the peak "was observed
to exceed 13 GB for a batch of 128 on an 11×11 grid" — but the measured peak at batch **256**
on that grid is 874 MB, and an early CPU run of the sweep put checkpointing at **+29 % wall
clock**, which made `--no_trm_checkpoint_core` look like close to free speed. **Measured on
the GPU in v17 it is not:** 12 % faster for 4.8× the memory, and it OOMs at batch 64 where
the checkpointed path is fine (the `off` row in the table above). Keep it on. The CPU figure
did not transfer, and neither does the hint in `medmamba_ss_trm.medmamba_ss_trm_default`'s docstring, which
was written for `trm_mixer="ss2d"`.

Two further notes specific to this hardware:

* **The GPU is Blackwell (`sm_120`).** It needs CUDA 12.8+ / a `cu128` torch build. The
  runs above used **torch 2.11.0+cu128**, which is correct. A `cu124` build (e.g. torch 2.5.x)
  does **not** contain `sm_120` kernels and will fail on this card — check
  `torch.cuda.get_arch_list()` before blaming the code.
* **TF32 and `cudnn.benchmark` are already on** (`train_example_v16.py:402-405`), and
  `--amp bf16` is the default and is supported natively. **`--compile on` is the default
  since v17 S3** — this workload (many tiny kernels, ~63 sequential core applications, fixed
  shapes) is exactly the shape `torch.compile` helps, and it was benchmarked: 1.64× on PAD
  and *lower* peak memory (§4.1). It is no longer an open experiment. Numerics differ from
  eager by ~1.7e-7; pass `--compile off` to reproduce a pre-v17 run exactly.

---

## 5. Reconstruction — what exists, what did not, and how to get it

### 5.1 The state of things

`--recon_mode latent` trains an auxiliary decoder and the MSE/SAM terms are computed
correctly, in reflectance units. But `TrainerG_v12` **mean-pools `x_recon` over `(H, W)` and
keeps only `[N, C]` mean spectra**; the `[B, C, H, W]` cube is freed at the end of the batch.

Consequently, of the 60 runs in `experiments/`:

* 8 runs trained a decoder that demonstrably learned (train MSE falling, e.g.
  `20260904_230812` HSI 0.745 → 0.207 over 30 epochs; `20260906_031815` PAD RGB
  0.599 → 0.055 over 10);
* 4 of those have a `reconstruction/` directory, containing **only figures** — five metric
  histograms (MAE, RMSE, SAM, SID, peak-position error) and best/median/worst **spectrum
  line plots**;
* **zero runs contain a reconstructed image, a reconstructed cube, or a machine-readable
  reconstruction metrics file.** There is no `samples.npz`, no `samples_metrics.json`,
  no `status.json` anywhere in `experiments/`.

Three further gaps in the frozen path: `training/spectral_recon_metrics.ssim` is *spectral*
SSIM, so **nothing measured spatial fidelity at all**; the frozen figures are written from the
*last* epoch's validation metrics rather than from the checkpoint actually shipped; and the
histogram caption claims "across all validation samples" while the caller caps at the first
`recon_sample_cap` patches in split order.

### 5.2 What now closes it

`training/recon_artifacts.py` + `training/trainerg_v13.py` (`TrainerG_v13(TrainerG_v12)`)
add the missing half. **18/18 unit tests in `test_recon_artifacts.py` pass.** Written under
`<run>/reconstruction/`:

| file | contents |
|---|---|
| `samples.npz` | `x_true` / `x_recon` cubes `[N,C,H,W]` in reflectance units + labels, split indices, tags, wavelengths, class names |
| `samples_metrics.json` | per-sample SAM, RMSE, PSNR, SID, **spectral** SSIM, **2-D** SSIM (new — the spatial metric that did not exist), peak-position error |
| `provenance.json` | which checkpoint, the selection rule, the RGB-composite band choice, split statistics |
| `status.json` | written on **every** path, including skips — so "no decoder" is recorded rather than leaving an empty directory |
| `reconstruction_samples.png`, `samples/NN_<tag>.png` | contact sheet + one four-panel figure per sample |
| `metrics_curves.png` | per-epoch SAM / RMSE / PSNR / spectral SSIM |

Samples are chosen at **evenly spaced SAM quantiles within each class**, so the worst case is
in the figure by construction and the selection rule is recorded — a figure cannot be
mistaken for a random draw or for a cherry-pick.

### 5.3 Entry points

| entry point | best-metrics config | recon artifacts |
|---|---|---|
| `train_example_v16.py` | no | no |
| `train_example_v16_optimal.py` | yes | no |
| `train_example_v16_recon.py` | no | yes |
| **`train_example_v16_optimal_recon.py`** | **yes** | **yes** |

The first three are mutually exclusive because each contributes the same field of
`v16.Overrides` — `trainer_cls` (before v17 S9, the same module global `v16.TrainerG_v12`).
`train_example_v16_optimal_recon.py` composes them —
`TrainerG_v13Optimal(TrainerG_v12Optimal, TrainerG_v13)`, an empty class whose MRO gives
each side exactly what it contributes — and flips two defaults, `recon_mode=latent` and
`lambda_mse=0.1`.

**Verification status.** The path has been exercised end-to-end, on CPU, against a small
synthetic 3-class RGB dataset (96/48/48 samples of 11×11×3, `--epochs 2 --batch_size 16
--amp off`). `train_example_v16_optimal_recon.py` completed, gate G7 passed
("reconstruction gradient reaches 20 recursive-core parameters"), and the run wrote
**6 cubes + 7 figures**:

```text
reconstruction/samples.npz          x_true, x_recon  (6, 3, 11, 11) float32
                                    + label, dataset_index, tag, wavelengths, class_names
reconstruction/samples_metrics.json per sample: rmse mae psnr ssim_spectral ssim2d
                                    pearson_correlation peak_position_error sam_deg sid
reconstruction/status.json          written:true, split, checkpoint, denormalized:true,
                                    rgb_bands, elapsed_s, timestamp
reconstruction/provenance.json      + reconstruction_samples.png, metrics_curves.png,
                                    5 histograms, samples/NN_class<k>_q<Q>.png
```

Selection came out as best+worst per class (`q0.00` / `q1.00` × 3 classes), as designed.

> That run validates the **plumbing, not reconstruction quality** — 12 total optimizer steps
> against 11 warmup steps on random pixels leaves the decoder essentially untrained (its
> per-sample Pearson correlations are ~0). **No real dataset has produced artifacts yet.**
> The commands in §6.3 are what produce the first meaningful ones; the
> `export_recon_samples.py` route against `20260904_230812` is the quickest, because
> that decoder actually trained (MSE 0.745 → 0.207 over 30 epochs).

---

## 6. Commands

### 6.1 Prepare

```bash
# HSI — gain-corrected reflectance (the dataset behind the §1.1 result)
python prepare_histologyhsi_bc.py \
    --root /path/to/HistologyHSI-BC-Recurrence --out_dir ./data/hsi_v8-80_10_10_importance \
    --patch_size 11 --stride 11 --label_source tissue --modality both \
    --split 80_10_10 --band_selection importance --num_bands 32 --num_workers 8

# PAD-UFES-20 — whole clinical images, patient-grouped, three-way split
python prepare_pad_ufes_20_optimal.py \
    --root /path/to/PAD-UFES-20 --out_dir ./data/pad_optimal --num_workers 8

# the linear-probe bar the network has to clear (writes shallow_baseline.json next to the data)
python scripts/shallow_baseline_v16.py --data_dir ./data/pad_optimal
```

### 6.2 Train

```bash
# HSI — reproduces the §1.1 headline (0.9436 / 0.9073 / 0.8580)
python train.py --profile pipeline_defaults \
    --data_dir data/hsi_v8-80_10_10_importance/hsi/ --architecture recursive \
    --normalization global_zscore --loss weighted_ce --class_weight_power 0.75 \
    --batch_size 256 --lr 1e-4 --epochs 20 --amp bf16 --checkpoint_metric f1_macro \
    --train_subsample_frac 0.1 --val_subsample_frac 0.1 --seed 42

# the matched RGB arm: identical flags, only --data_dir changes
python train.py --profile pipeline_defaults --data_dir data/hsi_v8-80_10_10_importance/rgb/ ...

# PAD — best-metrics configuration. Run the ladder first; --stage fit proves the model
# can fit at all before any regulariser or class weighting is switched on.
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --stage fit
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --stage balance
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal

# spend the idle 91% of the card on spatial detail instead (~4x the wall clock)
python train.py --profile pad_ufes_best_norecon --data_dir ./data/pad_optimal --target_token_grid 56
```

### 6.3 Reconstruction

```bash
# NEW RUN, with artifacts. HSI is where SAM means something, so give it a weight.
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance/hsi/ \
    --lambda_mse 0.1 --lambda_sam 0.1 --recon_sample_per_class 5

# PAD, best-metrics config + artifacts
python train.py --profile paper_recipe --data_dir ./data/pad_optimal

# EXISTING FINISHED RUN — the decoder is still in best_model.pt, so the cubes are
# recoverable without retraining. This is the fastest way to a first samples.npz.
python scripts/export_recon_samples.py \
    --run_dir experiments/20260904_230812_portance_undersample-hsi_recursive_hsi_mlp_sup3_bs256_bf16_wce_modover_ppz_sub01_vsub02 \
    --split val --recon_sample_per_class 3

# --split test gives a HELD-OUT reconstruction number, which no training run can produce
# (the frozen evaluator builds its test loader with return_norm_stats=False and therefore
# cannot denormalize, so it cannot report reflectance-unit reconstruction error at all).
python scripts/export_recon_samples.py --run_dir <run> --split test
```

Runs that still carry a usable decoder: `20260904_230812` (30 ep, HSI — the best one),
`20260904_164643` (6 ep, HSI), `20260906_031815` (10 ep, PAD RGB), `20260904_121119`
(2 ep, HSI).

---

## 7. Reading a run before believing it

In order, from the preflight:

* `[baseline]` — the constant-predictor scores and the linear-probe ceiling. A network
  below the probe has an **optimisation** problem; a network at the prior-entropy CE floor
  has learned the class marginal and nothing else.
* `[budget]` — optimizer steps. Whole-image PAD is ~51 steps/epoch; underfitting is the
  default failure there.
* `[trm-cost]` / `[gpu-budget]` — token-rows and VRAM. Where the two disagree, believe
  `[gpu-budget]`; where either disagrees with `GPU_memory_MB` in `history.json`, believe
  `history.json`.
* `[ema]` — **the trap.** `EMAHelper` seeds its shadow with the *random initialisation* and
  never bias-corrects, and `TrainerG_v10._validate_one_epoch` validates the EMA copy. A
  stale EMA and a collapsed model are indistinguishable in the epoch log — both show
  balanced accuracy pinned at exactly `1/k`. `train_example_v16_optimal.py` resolves the
  decay from steps/epoch so the horizon is a fixed number of *epochs*; the block prints how
  much of the random init is still in the validation model at each epoch.

And in the results: balanced accuracy pinned at exactly `1/num_classes` means a one-class
predictor. On PAD that is 0.1667, on the histology corpus 0.3333.
