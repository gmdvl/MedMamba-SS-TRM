# 18 — The ten best MedMamba-SS-TRM runs, and how to re-run them

Compiled 2026-10-04 from the run folders in `experiments/`. Every number below is read from
each run's `test_report.json` and `config.json`; nothing is re-computed.

To re-run the top three on any dataset: [`scripts/run_top3.py`](../scripts/run_top3.py)
(see [§5](#5-re-running-the-top-three-scriptsrun_top3py)).

## 1. What was ranked

- **Model:** MedMamba-SS-TRM only — `architecture: recursive`, the `MedMambaSSTRM` class in
  [`medmamba_ss_trm.py`](../medmamba_ss_trm.py) (was `GMedMambaRecursive` in `gmedmamba.py`).
  The comparison models are excluded: HybridSN, SpectralFormer, the MedMamba-HSI baseline and
  the hierarchical MedMamba-SS arms (`split` / `fullchannel`).
- **Reconstruction images required:** the run must have `reconstruction/reconstruction_samples.png`.
  All ten below have the full set of 19 PNGs (`reconstruction_samples/best/median/worst.png`,
  error histograms, `metrics_curves.png`, and per-sample images in `reconstruction/samples/`).
- **Ranked by** test balanced accuracy. 20 runs qualify; these are the top ten.
- **Test split:** HistologyHSI-BC, 348,894 patches, 3 classes (healthy / DCIS / IDC). Each
  run reports the EMA weights from its best epoch, chosen by validation macro-F1.

## 2. The table

All ten trained on `data/hsi_v8-80_10_10_importance-new/hsi` (32 bands) with the
446,409-parameter backbone, and all finished every epoch they were set to run
(`TRAINING_COMPLETED`). The command column is the **current** command: `train.py` with
`--profile paper_recipe` (the profile was called `v18` until 2026-10-01; the old name still works).

| # | Run (`experiments/…`) | `run_tag` | Bal. acc | Macro-F1 | Acc | DCIS F1 | κ | ROC-AUC | ECE | Best ep. / epochs | GFLOPs | Current command to replicate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `20260915_150351` | `depth-n2` | **0.9441** | 0.8992 | 0.9615 | 0.785 | 0.9177 | 0.9966 | 0.054 | 5 / 12 | 4.22 | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 12 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --trm_n_improve 2 --run_tag depth-n2` |
| 2 | `20260915_031356` | `ref20` | 0.9437 | **0.9036** | **0.9638** | **0.792** | **0.9226** | 0.9957 | 0.056 | 7 / 20 | 6.23 | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 20 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --run_tag ref20` |
| 3 | `20260914_181738` | *(none)* | 0.9431 | 0.9003 | 0.9622 | 0.786 | 0.9192 | 0.9960 | 0.054 | 7 / 20 | not recorded (same model as #2) | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 20 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --num_workers 2 --no_pin_memory --no_persistent_workers --dataset_validation_level structural` |
| 4 | `20260915_183538` | `depth-n4` | 0.9386 | 0.8967 | 0.9609 | 0.778 | 0.9164 | 0.9962 | 0.051 | 5 / 12 | 8.24 | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 12 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --trm_n_improve 4 --run_tag depth-n4` |
| 5 | `20260915_123136` | `depth-n1` | 0.9378 | 0.8932 | 0.9581 | 0.775 | 0.9106 | 0.9958 | 0.066 | 4 / 12 | 2.21 | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 12 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --trm_n_improve 1 --run_tag depth-n1` |
| 6 | `20260919_205302` | `rq0-s23` | 0.9378 | 0.8976 | 0.9615 | 0.779 | 0.9175 | 0.9961 | 0.051 | 5 / 20 | 6.23 | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 20 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 23 --run_tag rq0-s23` |
| 7 | `20260915_031555` | `abl-base` | 0.9353 | 0.8920 | 0.9589 | 0.769 | 0.9121 | 0.9959 | 0.053 | 5 / 12 | 6.23 | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 12 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --run_tag abl-base` |
| 8 | `20260916_155129` | `rq0-s7` | 0.9341 | 0.8962 | 0.9612 | 0.775 | 0.9169 | 0.9949 | 0.049 | 6 / 20 | 6.23 | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 20 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 7 --run_tag rq0-s7` |
| 9 | `20260915_202942` | `enc-index` | 0.9291 | 0.8804 | 0.9530 | 0.748 | 0.8998 | 0.9958 | 0.054 | 5 / 12 | 6.23 | `python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi --batch_size 256 --epochs 12 --lambda_sam 0.1 --train_subsample_frac 0.102 --val_subsample_frac 0.0986 --seed 42 --no_use_wavelengths --run_tag enc-index` |
| 10 | `20260916_013913` | `enc-index` (repeat) | 0.9287 | 0.8798 | 0.9528 | 0.747 | 0.8993 | 0.9958 | 0.054 | 5 / 12 | 6.23 | same command as #9 |

Full run folder names:

1. `experiments/20260915_150351_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_depth-n2`
2. `experiments/20260915_031356_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_ref20`
3. `experiments/20260914_181738_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon`
4. `experiments/20260915_183538_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_depth-n4`
5. `experiments/20260915_123136_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_depth-n1`
6. `experiments/20260919_205302_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_seed23_optimal_recon_rq0-s23`
7. `experiments/20260915_031555_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_abl-base`
8. `experiments/20260916_155129_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_seed7_optimal_recon_rq0-s7`
9. `experiments/20260915_202942_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_enc-index`
10. `experiments/20260916_013913_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_enc-index`

What each run changes relative to #2 (`ref20`):

| # | `run_tag` | Change vs `ref20` | What the run was for |
|---|---|---|---|
| 1 | `depth-n2` | `--trm_n_improve 2 --epochs 12` | recursion-depth sweep (42 core applications) |
| 2 | `ref20` | — | the reference / headline run |
| 3 | *(none)* | none in the model or training; trained with `train_example_v16_optimal_recon.py` and different loader / dataset-check settings | the run `ref20` reproduces |
| 4 | `depth-n4` | `--trm_n_improve 4 --epochs 12` | depth sweep (84 core applications) |
| 5 | `depth-n1` | `--trm_n_improve 1 --epochs 12` | depth sweep (21 core applications) |
| 6 | `rq0-s23` | `--seed 23` | HSI-vs-RGB seed study |
| 7 | `abl-base` | `--epochs 12` | 12-epoch baseline for the ablations |
| 8 | `rq0-s7` | `--seed 7` | HSI-vs-RGB seed study |
| 9 | `enc-index` | `--no_use_wavelengths --epochs 12` | wavelength-encoding ablation |
| 10 | `enc-index` | same as #9 | repeat of #9 |

## 3. Parameters all ten share

Everything not in the "change" column above is this recipe (from `ref20`'s `config.json`,
`cli_args`):

| Group | Parameters |
|---|---|
| Data | `data_dir data/hsi_v8-80_10_10_importance-new/hsi` (32 bands, 3 classes); `normalization global_zscore`; `patch_size 1`; `train_subsample_frac 0.102` redrawn every epoch (250,113 of 2,452,086 patches); `val_subsample_frac 0.0986` stratified (32,985 of 334,516) |
| Model | `architecture recursive`; `trm_dim 128`; `trm_core_layers 2`; `trm_mixer mlp`; `trm_n_latent 6`; `trm_n_improve 3`; `trm_deep_supervision_steps 3`; halting off; `spectral_token_fusion concat_mlp`; `use_wavelengths true`; `classifier_dropout 0.1`; `activation leakyrelu` |
| Loss | `loss focal_weighted`, `focal_gamma 1.5`; `class_weight_power 0.75`; `weight_method balanced`; `sampler none`; reconstruction `recon_mode latent`, `lambda_mse 0.1`, `lambda_sam 0.1` |
| Optimization | `lr 3e-4`; `weight_decay 0.05`; `batch_size 256`; `amp bf16`; compile on; `max_gradient_norm 1.0`; linear warmup into cosine decay, stepped per batch (`scheduler_interval step`; 19,560 steps with 587 warmup steps at 20 epochs); EMA `trm_ema_rate 0.9995`; `seed 42` |
| Checkpointing | `checkpoint_metric f1_macro`; `eval_test best`; early stopping off (`early_stop_patience 40` is recorded but was never active) |
| Augmentation | horizontal flip, vertical flip and 90° rotation, each p = 0.5; spectral noise std 0.02, offset std 0.05, scale range ±0.15 |
| Reconstruction output | `recon_render_png on`; `recon_save_samples on`; `recon_sample_per_class 3`; `recon_max_cubes 64` |

`--lambda_sam 0.1` must always be typed: the profile does not set it.

## 4. How the commands were checked

- **Each command reproduces its run's recorded parameters.** Every command in §2 was resolved
  with the current `training.train_cli.resolve` and compared with the run's recorded
  `cli_args`. Every value matches.
- **Three flags are newer than these runs.** `lr_schedule warmup_cosine`,
  `global_stats train_fit` and `early_stopping off` are not in the old `config.json` files.
  [`tests/test_train_parity.py`](../tests/test_train_parity.py) checks that `paper_recipe`
  resolves them to what the old `train_example_v18.py` did.
- **Run #3 uses `paper_recipe`.** It was trained with `train_example_v16_optimal_recon.py`,
  which has no profile of its own; that entry point's defaults changed on 2026-09-28, so it no
  longer matches this run. `paper_recipe` plus the four loader / dataset-check flags matches
  exactly; those flags affect only data loading and the startup dataset check, not training.

## 5. Re-running the top three: `scripts/run_top3.py`

```bash
python scripts/run_top3.py --dry_run                                # print commands + resolved config
python scripts/run_top3.py                                          # all three, on the paper build
python scripts/run_top3.py --data_dir data/hsi_v9-trainsel/hsi      # all three, on another dataset
python scripts/run_top3.py --only ref20 --data_dir data/pad_optimal # one experiment
python scripts/run_top3.py --status                                 # which are done / still to run
```

| Option | Default | Meaning |
|---|---|---|
| `--data_dir` | `data/hsi_v8-80_10_10_importance-new/hsi` | dataset build to train on |
| `--only` | all three | any of `depth-n2`, `ref20`, `ref20-v16` (= #1, #2, #3) |
| `--subsample` | `auto` | `auto` sizes each epoch for `--data_dir` (~250k train / ~33k validation patches, or the whole split if smaller); `paper` types 0.102 / 0.0986 regardless of dataset |
| `--tag_prefix` | `top3-` | run tags become `top3-depth-n2`, `top3-ref20`, `top3-ref20-v16` |
| `--dry_run` / `--status` | — | run nothing |
| `--log_dir` | `logs/top3/<data dir>/` | one log per run |

- **On the paper build, both `--subsample` modes give the original runs' exact `cli_args`.**
  This was checked for all three experiments: `auto` resolves to the same 0.102 / 0.0986.
- **On another dataset, use `auto`.** The paper's fractions only make sense for a split of
  about 2.45 M training patches; on a smaller dataset they would train on a tenth of it.
  The profile also sets `patch_size` and the EMA rate from the dataset's size.
- **The script is resumable.** A run that already finished with the same tag on the same
  `--data_dir` is skipped, so after an interruption run the same command again.
- **Runs go one at a time.** With a 2.6 GB RAM load per job, the host fits two at most.
- **The `top3-` tags keep these runs separate** from the paper's own `ref20` / `depth-n2` runs,
  which [`sweeps/paper_runs.py`](../sweeps/paper_runs.py) looks up by tag.
- **Use the CUDA environment:** `~/.local/share/mamba/envs/gmedmamba` (torch 2.11, cu128).
- **Each new run writes its own `reconstruction/` folder**, since the commands keep
  `recon_mode latent` and the PNG/sample settings.

## 6. Reading the ranking

- **#1–#3 are tied.** They are 0.1 points apart in balanced accuracy, and re-running the same
  seed moves it by about 0.06 points (#2 against #3). By macro-F1, accuracy and DCIS F1,
  #2 `ref20` is first. Runs are not deterministic (`deterministic: false`), so a re-run will
  not match to the last digit.
- **#1 is the cheapest of the three.** `depth-n2` needs 4.22 GFLOPs against 6.23 and only
  12 epochs. The recursion-depth sweep is flat: more core applications do not help.
- **The list is mostly seeds and ablations of one recipe.** #6 and #8 are `ref20` with other
  seeds; #9 and #10 are one configuration run twice. Seeds 1 and 13 fall outside the top ten
  (seed 13 is 11th at 0.9276).
- **These are test-split numbers only.** On the held-out validation patients this model family
  scores about 72 % balanced accuracy, below HybridSN; pooled over both, HybridSN leads
  (87.1 against 83.2). The test-split lead does not hold across patients.
