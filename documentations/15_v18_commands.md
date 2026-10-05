# v18 — Every command needed to complete the paper

> **2026-09-28:** the commands below were rewritten from `train_example_v16*.py` / `train_example_v18.py` / `scripts/v20_queue.sh` to `train.py --profile …` / `run_experiments.py`, with the same flags. The two resolve every argument identically (`tests/test_train_parity.py`; see [`17_train_and_sweeps.md`](17_train_and_sweeps.md)). The originals are in `archive/`. A pre-2026-09-28 `train_example_v16_optimal_recon.py` command is `--profile paper_recipe`. **2026-10-01:** the profiles were renamed for what they are for (`v18` → `paper_recipe`, `optimal` → `pad_ufes_best_norecon`, `original` → `medmamba_protocol_norecon`, `base` → `pipeline_defaults`); `_norecon` marks the two without reconstruction. The old names still work.


Companion to `plan/GMedMamba_v18_missing_experiments.plan.md` (the gap analysis) and
`/home/dante/.claude/plans/build-very-detailed-strong-temporal-umbrella.md` (the implementation
plan). **This file is the execution sheet**: every command, every flag, what it writes, how long it
takes, and which manuscript claim it closes.

Run everything from the repository root unless a block says otherwise.

---

## 0. Three things that will silently ruin a run if you skip them

**0.1 `--lambda_sam 0.1` is mandatory on every HSI command.**
`RECON_DEFAULTS` (`train_example_v16_optimal_recon.py:101`) sets only `lambda_mse`;
`OPTIMAL_DEFAULTS` (`train_example_v16_optimal.py:432`) leaves `lambda_sam` at **0.0**. The reference
run's `config.json` records `lambda_sam: 0.1`, so it was typed on the command line. Omit it and the
spectral-angle term vanishes from the objective with no warning — on HSI that is the half of the
reconstruction loss that means anything. Verified against the reference config: with
`--lambda_sam 0.1`, 126 of 130 `cli_args` fields match at parse time and the other four
(`num_workers`, `persistent_workers`, `pin_memory`, `recon_out_activation`) resolve at runtime to the
reference values.

**0.2 `--epochs` sets the LR schedule, so ablations need their own baseline.**
`train_example_v16.py:709` computes `total_steps = args.epochs * steps_per_epoch`, and the cosine
decays over exactly that. A 12-epoch run is a *different schedule*, not a truncated 20-epoch one.

| Block | `--epochs` | Members | Baseline |
| --- | ---: | --- | --- |
| Headline | 20 | RQ0 pair + seed replicates | `20260914_181738…` (exists) |
| Ablation | 12 | depth, architecture, recon, encoding, bands | **B0**, run it first |

**0.3 `--run_tag` is mandatory for E2, E4 and E6.**
`run_naming_v16.py:_NAME_ALLOWLIST` covers `loss`, `sampler`, `normalization`, both subsample
fractions and `seed` — not `trm_n_improve`, `recon_mode` or `use_wavelengths`. Without a tag those
arms differ from their baseline **only by timestamp**. `tests/test_run_tag_v18.py` pins this.

---

## 1. Prerequisites — already done, no GPU

These landed with v18 and need no re-running unless the code changes.

```bash
python -m pytest tests/ -q
# expect: 490 passed, 1 skipped   (includes test_frozen_files_untouched.py)
```

New files: `train_example_v18.py`, `training/flops_report.py`,
`training/latent_space_v18.py`, `scripts/flops_report.py`, `scripts/latent_space.py`,
`scripts/eval_band_decimation.py`, `scripts/make_band_subset_build.py`, three test files,
plus `global_zscore` in `medmamba-original/MedMamba/train_hsi.py`. Nothing frozen was edited.

---

## 2. Retro-fit the finished runs — no GPU, ~30 min

### 2.1 FLOPs for every run

```bash
for r in experiments/2026*/; do
  [ -f "$r/config.json" ] || continue
  python scripts/flops_report.py --run_dir "$r"
done
```

Writes `<run>/flops_report.json`. **Already run for the five cited runs**, and the numbers
reproduce §7.6's *(prior)* panel almost exactly — which is the best possible outcome, since it means
the panel was right and merely unevidenced:

| | §7.6 *(prior)* | measured |
| --- | ---: | ---: |
| Conv/linear FLOPs | 6,187 M | **6,187.5 M** |
| Scan FLOPs | 16 M | **15.86 M** |
| Total | 6,203 M | **6,203.3 M** |

### 2.2 The architecture comparison the paper never had

```bash
python scripts/flops_report.py \
    --architectures split,fullchannel,efficient,recursive \
    --modality hsi --num_classes 3 --in_channels 32 \
    --wavelengths data/hsi_v8-80_10_10_importance-new/hsi/wavelengths.npy \
    --out findings/flops_architectures_hsi.json

python scripts/flops_report.py \
    --architectures split,recursive \
    --modality rgb --num_classes 6 --in_channels 3 \
    --patch_hw 224,224 --patch_size 8 \
    --out findings/flops_architectures_pad.json
```

`--patch_size 8` on the PAD command is **not optional**: the PAD whole-image runs resolve it from
`--target_token_grid 28` and record `patch_size: 8`, against the recursive preset's 1. Counting at
the preset value overstates the recursive model's PAD cost by 64×.

**Results (verified this session):**

| Task | Architecture | Params | Total GFLOPs |
| --- | --- | ---: | ---: |
| HSI 32-band, 3-class, 11×11 | `split` (hierarchical) | 2,773,007 | 0.278 |
| | `fullchannel` | 3,513,451 | 0.305 |
| | `efficient` | 3,685,483 | 0.312 |
| | **`recursive`** | **446,409** | **6.203** |
| PAD RGB, 6-class, 224², ps 8 | `split` | 27,425,314 | 2.396 |
| | **`recursive`** | **446,796** | **39.211** |

**The cost inversion, same-task, for the first time: 6.2× fewer parameters and 22.3× more FLOPs on
HSI; 61.4× fewer parameters and 16.4× more FLOPs on PAD.** §7.6 currently compares against the
MedMamba baseline (191×, *(prior)*); these two rows are the comparison RQ2 actually claims.

### 2.3 Latent-space plots

```bash
for r in experiments/20260914_181738*/ experiments/20260906_222725*/ \
         experiments/20260907_025027*/ experiments/20260904_230812*/; do
  python scripts/latent_space.py --run_dir "$r" --max_samples 900 --batch_size 64
done
```

Writes `<run>/latent_space/latent_tsne.png` (and `latent_umap.png` if `umap-learn` is installed —
`pip install umap-learn`, optional).

Already run for the reference run. The figure corroborates §7.2 exactly: **IDC forms its own
isolated island; healthy and DCIS overlap heavily** — the entire margin lives at that boundary,
which is what the per-class F1 table says numerically. This is the qualitative figure §7.2 lacks.

> Two things this script fixes that the trainer does not. It samples **stratified** rather than
> taking the first N off an unshuffled loader — `X_*.npy` is written capture by capture and is
> class-ordered, so the naive path plots one class and looks entirely plausible (the two runs that
> did produce a plot produced exactly that). And it loads the checkpoint **strictly**, under the key
> `model_state` that `TrainerG_v11._save_checkpoint` actually writes.

---

## 3. The reference command

Every training command below is this one with the stated flags changed and **nothing else**.

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 20 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --seed 42 --run_tag ref20
```

Resolves to: `recursive`, `trm_dim 128`, `trm_core_layers 2`, `trm_n_latent 6`, `trm_n_improve 3`,
`trm_deep_supervision_steps 3`, `trm_mixer mlp`, halting off, `recon_mode latent`,
`lambda_mse 0.1`, `lambda_sam 0.1`, `loss focal_weighted` γ 1.5, `class_weight_power 0.75`,
`sampler none`, `global_zscore`, `augment_preset custom`, `bf16`, `compile on`,
`trm_ema_rate 0.9995`, `checkpoint_metric f1_macro`, `eval_test best`, `early_stop_patience 40`.

**Do not run it** — `experiments/20260914_181738…` *is* this run (20 epochs, 3.66 h, test 96.22 % /
94.31 % balanced / 0.9003 macro-F1, best epoch 7).

---

## 4. Tier 1 — the defects and the two largest caveats (6.6 h GPU)

### B0 · 12-epoch ablation baseline — 2.18 h

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --seed 42 --run_tag abl-base
```

Check before starting the block: `config.json → scheduler.total_steps == 11736`.

### E1 · RGB twin on the corrected build ★ — 1.9 h

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/rgb \
    --batch_size 256 --epochs 20 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --seed 42 --run_tag rq0-rgb
```

**Closes:** the §7.2 RQ0 pair currently rests on `data/hsi_v8-80_10_10_importance/`, prepared
2026-09-04, while `--rgb_source` landed 2026-09-12 (commit `3cca0d0`). Manuscript line 403 claims
the synthetic rendering; `documentations/12_commands_datasets.md` §B.3 says *"Measured under
`legacy`"*. This run uses the corrected build.

One argument differs from the reference: `--data_dir`. Keep `--lambda_sam 0.1` even though SAM over
three broad channels is a weaker quantity — changing it makes the pair differ in two arguments and
forfeits the single-variable claim. Expect the parameter counts **not** to match (575,849 vs
559,116): the decoder's `out_conv` is C-wide (`training/reconstruction_head.py:96`). Quote
`backbone_num_params` (446,409 both) for RQ1.

### E3a · hierarchical backbone, same task — 0.67 h ‡

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture split --seed 42 --run_tag arch-split
```

Works here because `build_model_optimal` drops the inapplicable `--classifier_dropout` with a
warning instead of raising; plain `train_example_v16.py` would abort before epoch 1.
Read against B0 (not against the 20-epoch reference).

### E5 · MedMamba on the same training distribution — 1.43 h

```bash
cd ../medmamba-original/MedMamba
python train_hsi.py \
    --data /data/dante_data/documents/Masters/courses/Thesis/g-medmamba/data/hsi_v8-80_10_10_importance-new/hsi \
    --epochs 5 --batch_size 256 --lr 1e-4 --amp bf16 \
    --class_weights --normalize global_zscore \
    --norm_stats_sample_cap 5000 \
    --checkpoint_metric f1_macro --patch_size 1 \
    --dims 64,128,256,512 --depths 1,1,2,1 --seed 42 \
    --run_name matched_natural_v8new
cd -
```

`--normalize global_zscore` exists as of v18 and is fitted on the training split only, ported
verbatim from `train_example_v6.py:60-75` so both pipelines fit the statistics identically.

**Closes:** §7.3's stated largest caveat — the existing baseline trained on the *undersampled*
build. Five epochs is generous: MedMamba peaks at **epoch 3 of 30** here, and the reference run sees
250,113 × 20 ≈ 5.0 M patch presentations ≈ 2.0 full epochs of this build. Say that rather than
claiming a matched budget.

### E8 · zero-shot band decimation — 0.5 h

```bash
# identity check FIRST - C=32 must reproduce the run's own test_report.json
python scripts/eval_band_decimation.py \
    --run_dir experiments/20260914_181738_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon \
    --keep_bands 32

# then the curve
python scripts/eval_band_decimation.py \
    --run_dir experiments/20260914_181738_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon \
    --keep_bands 32,16,8,4,2
```

Writes `<run>/band_decimation/C{n}/test_report.json` and `band_decimation/summary.{json,csv}` —
never the run's own `test_report.json`. The script prints an explicit `[identity]` verdict.

> **A 512-patch CPU smoke test already ran, and the result is strong enough to plan around.**
> C=32 reproduced the model (95.31 % acc / 93.05 % balanced / 0.8646 macro-F1 on those patches),
> while **C=16, 8, 4 and 2 all collapsed to predicting a single class** (~25 % acc, balanced
> accuracy ≈ 1/3). Structural band-count agnosticism does **not** imply behavioural agnosticism —
> the model instantiates identically at any C but its learned representation does not survive
> removing bands at test time. Confirm on the full test split before writing it up, and read it
> against E10's retrained arms, which answer the different question of whether a model *trained* at
> C works.

---

## 5. Tier 2 — the ablations the paper names (13.9 h GPU)

### E2 · recursion-depth quality sweep — 5.18 h

```bash
for T in 1 2 4; do
  python train.py --profile paper_recipe \
      --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
      --batch_size 256 --epochs 12 \
      --lambda_sam 0.1 \
      --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
      --trm_n_improve $T --seed 42 --run_tag depth-n$T
done
```

| `--trm_n_improve` | core applications | est. s/epoch | 12 epochs |
| ---: | ---: | ---: | ---: |
| 1 | 21 | ≈ 246 | 0.82 h |
| 2 | 42 | ≈ 449 | 1.50 h |
| 3 *(= B0)* | 63 | 653 measured | 2.18 h |
| 4 | 84 | ≈ 857 | 2.86 h |

Vary `trm_n_improve`, **hold `trm_deep_supervision_steps` at 3** — the latter also changes the number
of supervised segments, confounding depth with supervision density.
`tests/test_flops_report_v18.py` pins the 21/42/63/84 ladder.

**Closes:** §7.8's "most obvious missing experiment in the paper, and it is cheap", and §8.2's open
question against the TRM literature.

### E3b/E3c · the rest of the backbone family — 1.34 h ‡

```bash
# fullchannel takes no --fusion_type
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture fullchannel --seed 42 --run_tag arch-fullchannel

# efficient REQUIRES --fusion_type: see the warning below
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture efficient --fusion_type gated \
    --seed 42 --run_tag arch-efficient
```

> **`--fusion_type gated` is mandatory on the `efficient` arm, not optional.** The argparse
> *default* is `se_gate` (`train_example_v15.py:211`), and `build_model` rejects it:
> `AssertionError: unknown fusion_type 'se_gate'. Valid: ['cross_attention', 'film', 'gated',
> 'multiplicative', 'residual']`. Because it is a default rather than something you type, an
> `efficient` run with no `--fusion_type` dies after the dataset gates and before epoch 1, leaving
> a run directory holding only the gate reports and `failure_class.json`. This is what killed
> `20260915_151926…_arch-efficient`. `se_gate` is registered on `MedMambaSSEfficient`'s own fusion
> list, which is why it appears in older `config.json` files, but it never reaches `build_model`.

### E4 · reconstruction ablation — 2.07 h

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --recon_mode none --seed 42 --run_tag recon-off
```

No `--lambda_sam` here: `--recon_mode none` removes the decoder entirely (back to 446,409
parameters), which is the correct control. **Do not** use `--recon_mode latent --lambda_mse 0
--lambda_sam 0` — the decoder would exist, get no gradient, and write meaningless metrics; the entry
point warns about exactly this. This is the one arm that produces no reconstruction images by
construction; `reconstruction/status.json` records the skip.

**Closes:** §7.7's boundary — *"no part of the classification result can be attributed to this
objective"*.

### E6 · wavelength vs index encoding — 2.18 h

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --no_use_wavelengths --seed 42 --run_tag enc-index
```

`--no_use_wavelengths` is the true index arm — with `wavelengths=None` the tokenizer takes the
`index_positional_encoding` branch at `medmamba_ss_trm.py:836`. `--wavelength_scale None` is **not** an
index arm: it sets the angular span to the band count, which reduces to the index encoding only for a
*uniformly spaced* sensor, and this one has a **219.0 nm gap** between bands 18 and 19.

### E10 · retrained band arms at C = 16, 8 — 1.5 h CPU + 3.09 h GPU

```bash
python scripts/make_band_subset_build.py \
    --src data/hsi_v8-80_10_10_importance-new/hsi --keep 16 --out data/hsi_v8-bands16
python scripts/make_band_subset_build.py \
    --src data/hsi_v8-80_10_10_importance-new/hsi --keep 8  --out data/hsi_v8-bands8

for N in 16 8; do
  python train.py --profile paper_recipe \
      --data_dir data/hsi_v8-bands${N} \
      --batch_size 256 --epochs 12 \
      --lambda_sam 0.1 \
      --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
      --seed 42 --run_tag bands$N
done
```

The builder slices the parent build (verified bitwise identical on the kept bands over 2,000 random
patches) rather than re-preparing from raw cubes — minutes instead of ~1.8 h per arm, and it uses the
**same band subsets E8 evaluates zero-shot**, so the two read as one curve with two conditions.
Disk: ~11.5 GB at C=16, ~5.8 GB at C=8.

At C=8, `--spectral_checkpointing auto` no longer engages (`min_channels 16`): expect lower step time
and higher peak memory. That is the §7.6 mechanism again, and a second instance worth citing.

---

## 6. Tier 3 — seeds and the PAD ratio (14.0 h GPU)

### E7 · seed replicates of the RQ0 pair — 11.02 h, overnight

```bash
for S in 1 7; do
  for D in hsi rgb; do
    python train.py --profile paper_recipe \
        --data_dir data/hsi_v8-80_10_10_importance-new/$D \
        --batch_size 256 --epochs 20 \
        --lambda_sam 0.1 \
        --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
        --seed $S --run_tag rq0-s$S
  done
done
```

20 epochs, matching the headline block. **Run E1 first** — replicating the wrong RGB build is the one
way to make this worse. Three seeds total, against the 0.44-point nondeterminism floor already
measured (two byte-identical seed-42 runs: 0.7571 vs 0.7615 balanced accuracy).

### E9 · hierarchical MedMamba-SS on PAD whole-image — ≈3 h ‡

```bash
python train.py --profile pad_ufes_best_norecon \
    --data_dir data/pad_optimal-224 \
    --architecture split --seed 42
```

Makes the **61.4× parameter ratio verified on the task it is actually true for**, alongside E3a's
6.2× on HSI. Uses `train_example_v16_optimal.py`, not v18 — reconstruction on a 6-class RGB task is
not wanted here.

---

## 7. After every run — verification

```bash
python - "$RUN" <<'PY'
import json, sys, statistics as st
r = sys.argv[1].rstrip("/")
c = json.load(open(f"{r}/config.json")); a = c["cli_args"]
g = json.load(open(f"{r}/gates.json"))
t = json.load(open(f"{r}/test_report.json"))
h = json.load(open(f"{r}/history.json"))
et = [float(e["epoch_time"]) for e in h]
print("run       ", r.split("/")[-1])
print("entry     ", c["entry_point"], "| trainer", c["trainer"])
print("params    ", c["backbone_num_params"], "| steps", c["scheduler"]["total_steps"],
      "| epochs", a["epochs"], "| lambda_sam", a["lambda_sam"], "| run_tag", a.get("run_tag"))
print("gates     ", {k: v for k, v in g.items() if isinstance(v, bool)})
print("flops     ", "OK" if "error" not in t.get("flops", {}) else "MISSING")
print("test      ", {k: round(t["sklearn_metrics"][k], 4)
                     for k in ("accuracy", "balanced_accuracy", "f1_macro")})
print("wallclock ", f"{sum(et)/3600:.2f} h   median {st.median(et):.1f} s/epoch")
print("best epoch", json.load(open(f"{r}/experiment_report.json"))["completion_status"]["best_epoch"])
PY

ls "$RUN"/reconstruction/   # figures + samples/ + status/provenance/samples_metrics.json
ls "$RUN"/latent_space/     # latent_tsne.png (+ latent_umap.png if umap-learn present)
```

Expected values, verified on an end-to-end smoke run through `train_example_v18.py`:

* `entry` is `train_example_v18.py`, and `trainer` is **`TrainerG_v18Fast`**, not `TrainerG_v18`.
  That is correct: `--fast_loop on` is an `OPTIMAL_DEFAULTS` value and
  `training/trainerg_v12_fast.make_fast` (l. 421) builds
  `type("TrainerG_v18Fast", (TrainerG_v12Fast, TrainerG_v18), …)`. `TrainerG_v12Fast` overrides only
  `__init__`, `_abort_on_nonfinite_parameters` and `_train_one_epoch_deep_supervision`, so both v18
  overrides are still reached through the MRO — confirmed by the `[flops-v18]` and `[latent-v18]`
  lines in the run log.
* `flops` is `OK`. For a v18 run the count lives **inside `test_report.json["flops"]`**, written by
  `TrainerG_v18.evaluate_test_split`; there is no separate `flops_report.json` unless you also ran
  `scripts/flops_report.py` over the run. Check `test_report.json["flops"]["input_shape"]` —
  it must start with `1` (the batch axis the frozen caller dropped).
* `--lambda_sam` and `scheduler.total_steps` are on the list deliberately: they are the two fields
  that silently differ if §0.1 or §0.2 was skipped.

Both v18 overrides announce themselves in the log, so grep is the fastest check of all:

```bash
grep -E "^\[v18\]|flops-v18|latent-v18" "$RUN"/../*.log   # or wherever you tee'd the run
```

### Cross-run tables

```bash
python scripts/compare_runs.py 'experiments/*rq0*'   --per_patient --csv findings/rq0.csv
python scripts/compare_runs.py 'experiments/*depth*' --per_patient --csv findings/depth.csv
python scripts/compare_runs.py 'experiments/*arch*'  --per_patient --csv findings/arch.csv
python scripts/compare_runs.py 'experiments/*bands*' --per_patient --csv findings/bands.csv
python scripts/compare_runs.py 'experiments/*abl-base*' 'experiments/*recon-off*' \
       'experiments/*enc-index*' --per_patient --csv findings/ablations.csv
```

`--per_patient` is not optional on the histology corpus: 348,894 test patches come from **five
people**, and per-patient macro recall on existing runs spans 0.64 to 1.00.

---

## 8. Run order and budget

| Order | Run | GPU | Closes |
| ---: | --- | ---: | --- |
| 1 | §2 retro-fit | — | §3.6 + §7.6 → *(verified)*; latent figure |
| 2 | **B0** | 2.18 h | the ablation baseline everything else needs |
| 3 | **E1** | 1.90 h | the RQ0 build defect |
| 4 | **E8** | 0.50 h | RQ1 behavioural half |
| 5 | **E3a** | 0.67 h | RQ2 becomes architectural |
| 6 | **E5** | 1.43 h | §7.3's largest caveat |
| 7 | E2 | 5.18 h | §7.8's "most obvious missing experiment" |
| 8 | E4 | 2.07 h | §7.7 attributable |
| 9 | E6 | 2.18 h | RQ1 encoding half |
| 10 | E10 | 3.09 h | band curve, trained condition |
| 11 | E3b/E3c | 1.34 h | backbone family |
| 12 | E7 | 11.02 h | §9.1's binding limitation |
| 13 | E9 | 3.00 h | the PAD parameter ratio |
| | **Total** | **≈ 34.6 h** | |

Rows 1–6 are **6.7 h** and close all three defects plus both largest caveats. ‡ marks estimates with
no measured precedent — measure epoch 1 and re-budget if it is more than 2× off.

---

## 9. Manuscript v9 — the corrections that need no run

`cp` v8 → `paper/draft/MedMamba-SS-TRM_manuscript_v9.{tex,md}`, keep v8 intact, change both files
together.

| # | md line | Change |
| --- | --- | --- |
| 1 | 15, 60, 271, 664, 754, 816 | Attach each ratio to its task: **6.2×** vs hierarchical on HSI, **61.4×** on PAD-RGB. §3.6's table is already right; the prose is not. |
| 2 | 403 | The §7.2 pair used the **legacy** RGB source. Re-point at E1 or say so. |
| 3 | §3.6 table | 2.77 M and 27.43 M → *(verified)*; add `fullchannel` 3,513,451 and `efficient` 3,685,483. |
| 4 | §7.6, §7.8 | *(prior)* → *(verified)* from `flops_report.json`; add the same-task hierarchical column (0.278 vs 6.203 GF) and core applications. |
| 5 | §7.6 | The HSI/RGB wall-clock gap is **not** arithmetic: 6.2033 vs 6.0517 GF is 2.5 %. It is the `--spectral_checkpointing auto` threshold, now measured rather than asserted. |
| 6 | §9.1, doc 12 §B.3 | `20260909_221520` is seed 42 twice — a **0.44-point nondeterminism floor**, not a second seed. It supports RQ0. |
| 7 | §7.1 | Quote `backbone_num_params`, never `efficiency.num_params`: the decoder's output conv is C-wide, so a recon-wrapped model is not band-count agnostic (575,849 at C=32 vs 559,116 at C=3). |
| 8 | §6 | State the two-block epoch budget and why (`train_example_v16.py:709`). |

Compile check:
`flatpak-spawn --host tectonic -X compile --keep-logs --synctex <abs path>/MedMamba-SS-TRM_manuscript_v9.tex`
(v8 gives 22 pp, 0 errors, 0 undefined refs).

---

## 10. Results (runs of 2026-09-15/16, analysed 2026-09-19)

Arms are identified by `cli_args.run_tag`. Reproduce any table with
`python scripts/compare_runs.py 'experiments/*<tag>*' --per_patient`.

### 10.1 RQ0 — the margin is ~4× smaller than manuscript v8 claims

Three seeds, 20 epochs, corrected `--rgb_source synthetic` build, one flag apart:

| seed | HSI bal | RGB bal | Δ bal | HSI F1 | RGB F1 | Δ F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9437 | 0.8851 | +5.86 | 0.9036 | 0.8217 | +0.082 |
| 1 | 0.8961 | 0.8818 | +1.43 | 0.8077 | 0.8202 | **−0.013** |
| 7 | 0.9341 | 0.8897 | +4.45 | 0.8962 | 0.8311 | +0.065 |

**Mean Δ balanced accuracy +3.91 ± 2.26 pts; Δ macro-F1 +0.045 ± 0.050.** Manuscript v8 reports
**+15.02 pts / +0.1298** from one seed on the **legacy** RGB build.

What survives: balanced accuracy favours HSI at all three seeds. What does not: the macro-F1
standard deviation exceeds its mean, and at seed 1 the ordering **inverts**. The HSI arm is also
far less seed-stable (balanced-accuracy range 4.8 pts) than the RGB arm (0.8 pts). The same-seed
nondeterminism floor is ~0.06 pts (`ref20` vs `20260914_181738`), so the seed-1 result is a real
seed effect, not noise.

### 10.2 E2 — recursion depth is flat

12 epochs, only `--trm_n_improve` varies; core applications = (6+1)·n·3.

| core applications | balanced acc. | macro-F1 | DCIS F1 | GFLOPs (measured) |
| ---: | ---: | ---: | ---: | ---: |
| 21 | 0.9378 | 0.8932 | 0.774 | 2.215 |
| 42 | **0.9441** | **0.8992** | **0.785** | 4.224 |
| 63 *(reported config)* | 0.9353 | 0.8920 | 0.769 | 6.234 |
| 84 | 0.9386 | 0.8967 | 0.778 | 8.245 |

Quality is non-monotone and the reported 63 is the **worst** of the four, while costing **2.8× the
arithmetic of 21**. This answers §7.8's "most obvious missing experiment" in the negative:
recursion depth is not doing work on this task, so TRM's depth benefit did not transfer.

Fitting the four points gives **95.72 MFLOP per core application** on a 204 MFLOP intercept,
reproducing §7.6's *(prior)* 95.7 MFLOP/application from measured data for the first time.

> **Cost must be read off the FLOP column, not the clock.** These runs were executed 2–4 at a
> time on one GPU: `ref20`, `abl-base` and `rq0-rgb` overlapped, as did `arch-fullchannel`,
> `depth-n4` and `enc-index`. The contention is large enough to swamp the effect being measured —
> the two `enc-index` runs have **byte-identical configurations and differ 2.0× in wall-clock**
> (1251.7 vs 638.8 s/epoch). Only the two `rq0-s7` runs had the card to themselves, and they give
> the one clean timing in the set: **621.3 s/epoch at C=32 against 330.9 s/epoch at C=3, a 1.88×
> ratio** consistent with the 1.96× §7.6 reports for the headline pair. Re-time any arm you intend
> to quote as a cost, alone on the card.

### 10.3 E4 / E6 — both now attributable

Against the same 12-epoch B0 baseline (0.9353 balanced / 0.8920 F1 / 0.769 DCIS F1):

| arm | balanced acc. | macro-F1 | Δ F1 vs B0 |
| --- | ---: | ---: | ---: |
| `--recon_mode none` | 0.9225 | 0.8708 | −0.021 |
| `--no_use_wavelengths` | 0.9291 | 0.8804 | −0.012 |
| `--no_use_wavelengths` (repeat) | 0.9287 | 0.8798 | −0.012 |

Auxiliary reconstruction is worth **+0.021 macro-F1**, closing §7.7's "cannot be attributed"
boundary. Wavelength encoding is worth **+0.012**, answering RQ1's encoding half; the two index
runs agree to 0.0006, so it is above the noise floor.

### 10.4 E3 — a failed comparison, not a result

| arm | params | balanced acc. | macro-F1 | best epoch |
| --- | ---: | ---: | ---: | ---: |
| `recursive` (B0) | 446,409 | 0.9353 | 0.8920 | 5 |
| `split` | 2,773,007 | 0.3342 | 0.2741 | 1 |
| `fullchannel` | 3,513,451 | 0.4085 | 0.3974 | 1 |

0.3342 is chance for three classes, and both hierarchical arms peaked at epoch 1. **This does not
show the recursive backbone is better — it shows `OPTIMAL_DEFAULTS` does not transfer to the
hierarchical architectures.** Report it as a failed comparison; it needs its own LR and
regularisation search before it means anything.

### 10.5 E9 — PAD hierarchical

`--architecture split` on `pad_optimal-224`, 200 epochs, two runs: accuracy 0.535 / 0.541,
balanced 0.398 / 0.384, macro-F1 0.386 / 0.377, at 27,425,314 parameters and 2.396 GFLOPs.
Against the manuscript's recursive PAD row (446,796 parameters): accuracy 0.4564, balanced
0.5020, macro-F1 0.4213. The 61× larger model wins raw accuracy by ~8 points and **loses**
balanced accuracy by ~11 and macro-F1 by ~0.04.

### 10.6 E8 + E10 — band count: retrainable, but not transferable

Both conditions over the **same** band subsets, so they read as one curve. E8 is zero-shot from
the 32-band `ref20` checkpoint (full 348,894-patch test split); E10 retrains at that band count
under B0's 12-epoch schedule.

| bands | zero-shot bal | zero-shot F1 | retrained bal | retrained F1 |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.9437 | 0.9037 | 0.9353 *(B0)* | 0.8920 *(B0)* |
| 16 | 0.3845 | 0.1854 | **0.9159** | **0.8636** |
| 8 | 0.3329 | 0.1289 | **0.9014** | **0.8684** |
| 4 | 0.3940 | 0.1671 | — | — |
| 2 | 0.4091 | 0.1639 | — | — |

The C=32 row is the identity check and reproduces `ref20`'s own `test_report.json`
(0.9639 / 0.9437 / 0.9037 against 0.9638 / 0.9437 / 0.9036; macro-F1 delta 2.5e-05, which is
bf16 evaluation noise over 348,894 patches).

**Two claims, and they point opposite ways.**

*Zero-shot transfer fails completely.* Remove a single band from the set the model was trained on
and it collapses to near-chance, with **IDC F1 exactly 0.000 at every reduced band count** — the
class both arms otherwise solve perfectly. Structural band-count agnosticism (§7.1) does **not**
imply behavioural agnosticism, and the manuscript should say so explicitly rather than leaving
the behavioural half merely "untested".

*Retraining at a lower band count works, and works well.* At 8 bands the model reaches 0.8684
macro-F1 against the 32-band baseline's 0.8920 — **97 % of the macro-F1 on a quarter of the
spectrum** — and 8 bands slightly outperforms 16. That is the deployable result: a filter-wheel
camera with eight well-placed bands would lose little here.

So the honest formulation is: the architecture is band-count agnostic **by construction and under
retraining**, but a trained instance is **sensor-specific**.

### 10.7 E5 — MedMamba on the matched training distribution

`train_hsi.py` with `--normalize global_zscore` on `hsi_v8-80_10_10_importance-new/hsi`
(`runs_hsi/matched_natural_v8new`), 3,648,995 parameters:

| | MedMamba-SS-TRM (`ref20`) | MedMamba-HSI (matched) |
| --- | ---: | ---: |
| Parameters | **446,409** | 3,648,995 |
| Test accuracy | **0.9638** | 0.9514 |
| Test balanced accuracy | **0.9437** | 0.9102 |
| Test macro-F1 | **0.9036** | 0.8718 |

§7.3's "single largest caveat" — the old baseline trained on the *undersampled* build — is now
closed: same build, same normalization, same split. MedMamba-SS-TRM leads by **3.4 points of
balanced accuracy and 0.032 macro-F1 at 8.2× fewer parameters**.

### 10.8 The hierarchical collapse is not a learning rate

Two 3-epoch probes on `split` (§11.4) at `1e-4` and `3e-5`, against the original `3e-4`:

| lr | test bal | test macro-F1 | best epoch |
| --- | ---: | ---: | ---: |
| 3e-4 *(E3a)* | 0.3342 | 0.2741 | 1 |
| 1e-4 | 0.3506 | 0.3074 | 1 |
| 3e-5 | 0.3439 | 0.2992 | 1 |

All three behave identically, so **learning rate is ruled out**. Two other hypotheses are also
dead: EMA is not applied to non-recursive architectures at all
(`train_example_v16.py:802`), and the data is demonstrably learnable by a hierarchical model —
E5's MedMamba-HSI reaches 0.9102 balanced accuracy on the same build through its own pipeline.

The loss traces localise it without needing another run. The `split` arms **do** learn on train
(train accuracy 0.735 → 0.906, classification loss 0.236 → 0.080 over six epochs) while
validation macro-F1 freezes at exactly 0.2583 from epoch 2 and validation loss rises
monotonically. And the gradient norm runs **302 → 1002** against the recursive arm's 13.1 → 0.24,
so with `--max_gradient_norm 1.0` essentially every hierarchical step is clipped by a factor of
~300–1000.

**E3 is therefore an unresolved pipeline incompatibility, not a backbone comparison, and the
manuscript must not read it as evidence that recursion beats hierarchy.** The next probe, if one
is wanted, is `--recon_mode none` and/or `--max_gradient_norm 5.0` for three epochs — the
auxiliary decoder and the clipping threshold are the two untested suspects.

### 10.9 Still to run

Nothing in the v18 plan remains, with one deliberate exception.

* **E3b** (`efficient`) — still unrun, and **recommended to stay unrun** until §10.8 is resolved.
  Both other members of its family collapse identically at three learning rates; a third
  collapsed row costs 6–10 h and adds nothing. If §10.8's `--recon_mode none` probe finds the
  cause, run E3a/E3b/E3c together at the corrected setting.

Everything else — B0, E1, E2, E4, E5, E6, E7, E8, E9, E10 and all three prerequisite scripts — is
complete. The remaining work is the manuscript.

---

## 11. What is left to run (as of 2026-09-19)

**Run these ONE AT A TIME.** §10.2 shows why: the 2026-09-15/16 batch was executed 2–4 runs at a
time and the contention swamped the wall-clock measurements. Nothing below depends on anything
else below, so there is no reason to overlap them.

Prerequisites are already done: `data/hsi_v8-bands16` and `data/hsi_v8-bands8` exist, are bitwise
correct on the kept bands, and their manifests were regenerated on 2026-09-19 —
`verify_against_manifest` returns **CLEAN** for both, so the `DATASET_TRUNCATED` abort that killed
all four earlier attempts will not recur.

### 11.1 E8 — full-split band decimation (evaluation only, ~30 min) ★ do this first

```bash
# 1. identity check: C=32 MUST reproduce ref20's own test_report.json
python scripts/eval_band_decimation.py \
    --run_dir experiments/20260915_031356_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_ref20 \
    --keep_bands 32

# expected: acc 0.9638  bal 0.9437  macroF1 0.9036
#           and a line "[identity] C=32 ... macro-F1 MATCHES"
# if it does NOT match, stop - the slicing or wavelength handling is wrong.

# 2. the curve
python scripts/eval_band_decimation.py \
    --run_dir experiments/20260915_031356_10_10_importance-new-hsi_recursive_hsi_mlp_sup3_bs256_bf16_focalw_zscore_sub0102_vsub0986_optimal_recon_ref20 \
    --keep_bands 32,16,8,4,2
```

Writes `<run>/band_decimation/C{n}/test_report.json` and `band_decimation/summary.{json,csv}`.
Never touches the run's own `test_report.json`.

**What to expect.** A 512-patch CPU smoke test found C=32 reproducing the model (95.31 % accuracy
on those patches) while **C=16, 8, 4 and 2 all collapsed to predicting one class** (~25 %
accuracy, balanced accuracy ≈ 1/3). If that holds on the full split it is a substantive RQ1
result — structural band-count agnosticism without behavioural agnosticism — and it is the
cheapest finding in the whole set.

### 11.2 E10 — retrained band arms (2 runs, ~2–5 h each)

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-bands16 \
    --batch_size 256 --epochs 12 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --seed 42 --run_tag bands16

python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-bands8 \
    --batch_size 256 --epochs 12 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --lambda_sam 0.1 \
    --seed 42 --run_tag bands8
```

Read against B0 (`abl-base`, 0.9353 balanced / 0.8920 macro-F1) — same 12-epoch schedule — and
against E8's zero-shot points at the same band counts. The gap between the two conditions is what
retraining buys. At C=8 `--spectral_checkpointing auto` no longer engages (`min_channels 16`), so
expect a lower step time and a higher peak memory; that is the §7.6 mechanism, not a band effect.

### 11.3 E3b — the `efficient` arm, and a warning

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture efficient --fusion_type gated \
    --seed 42 --run_tag arch-efficient
```

`--fusion_type gated` is mandatory (§E3b). The command is verified to parse and the model builds
at 3,685,483 parameters.

> **This will probably collapse, and it will cost 6–10 h to find that out.** Both hierarchical
> arms already did — `split` to 0.3342 balanced accuracy (chance) and `fullchannel` to 0.4085,
> each peaking at epoch 1 — and `efficient` is the third member of the same family under the same
> recursive-tuned recipe. Spending the GPU time to collect a third collapsed row is poor value.
> **Fix the recipe first** (§11.4), then run all three arms together.

### 11.4 Recommended instead of 11.3 — a short LR probe on the hierarchical family

The collapse is a recipe-transfer failure, not an architectural result. `OPTIMAL_DEFAULTS` uses
`lr 3e-4`, tuned for a 0.45 M recursive core; the MedMamba baseline that trains fine on this data
uses `1e-4`. Three cheap epochs will say whether that is the whole story:

```bash
for LR in 1e-4 3e-5; do
  python train.py --profile paper_recipe \
      --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
      --batch_size 256 --epochs 3 \
      --lambda_sam 0.1 \
      --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
      --architecture split --lr $LR \
      --seed 42 --run_tag probe-split-lr${LR}
done
```

~1.5 h per probe at `split`'s measured 0.485 h/epoch. Success looks like validation macro-F1 rising
past epoch 1 rather than peaking there. If one of them trains, re-run E3a/E3b/E3c at that LR and
**say in the paper that the hierarchical arms needed their own learning rate** — that is itself a
reportable asymmetry between the two backbones, and a more honest result than either the collapse
or a silent recipe change.

### 11.5 After each run

```bash
python scripts/flops_report.py --run_dir <run> --write_into_test_report
python scripts/latent_space.py --run_dir <run> --max_samples 900 --batch_size 64
```

The first is only needed for runs that did **not** go through `train_example_v18.py` (it writes
`flops_report.json` and repairs the inline block); v18 runs count their own FLOPs correctly as of
the 2026-09-19 compiled-core fix. Then tabulate:

```bash
python scripts/compare_runs.py 'experiments/*bands*' --per_patient --csv findings/bands.csv
python scripts/compare_runs.py 'experiments/*arch*'  --per_patient --csv findings/arch.csv
```

---

## 12. GPU commands remaining (2026-09-19)

**Nothing here is required.** Every experiment in the v18 plan is complete (§10.9). Both tracks
below are optional and independent; run them one at a time (§10.2 — concurrent runs made the
15–16 Sept wall-clock unusable).

### Track A — make E3 a real comparison (recommended)

E3 is currently unusable: `split` and `fullchannel` collapse to ~chance, and §10.8 ruled out
learning rate, EMA and the data. Two suspects remain untested. Each probe is 3 epochs, ~0.5 h.

**A1 — is it the auxiliary decoder?**

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 3 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture split --recon_mode none \
    --seed 42 --run_tag probe-split-norecon
```

No `--lambda_sam` here: `--recon_mode none` removes the decoder, so the SAM/MSE terms do not exist.

**A2 — is it the gradient clipping?**

```bash
python train.py --profile paper_recipe \
    --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 3 \
    --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture split --max_gradient_norm 5.0 \
    --seed 42 --run_tag probe-split-clip5
```

`split` runs at gradient norm 302 → 1002 against the recursive arm's 13.1 → 0.24, so at the
default `--max_gradient_norm 1.0` essentially every step is scaled down by ~300–1000×.

**Decision rule.** Read `history.json`:

```bash
python - "experiments/<probe run>" <<'PY'
import json, sys
h = json.load(open(sys.argv[1].rstrip("/") + "/history.json"))
for e in h:
    print(f"ep{e['epoch']}  val_f1 {e.get('f1_macro',0):.4f}  val_loss {e.get('val_loss',0):.4f} "
          f"train_acc {e.get('train_accuracy',0):.4f}  gradnorm {e.get('gradient_norm',0):.1f}")
PY
```

*Fixed* = validation `f1_macro` rises past epoch 1 and `val_loss` falls. *Still broken* = `f1_macro`
pinned near 0.2583 with `val_loss` rising, the signature in §10.8.

> A 3-epoch run compresses the cosine schedule into 3 epochs, so it is a diagnostic, not a result.
> It is sufficient because the collapse signature appears by epoch 2 in every affected run.

**A3–A5 — only if a probe fixes it.** Re-run the whole family at 12 epochs with the corrected
setting, substituting `<FIX>` with whichever flag worked (`--recon_mode none` or
`--max_gradient_norm 5.0`):

```bash
python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture split <FIX> --seed 42 --run_tag arch-split-fixed

python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture fullchannel <FIX> --seed 42 --run_tag arch-fullchannel-fixed

python train.py --profile paper_recipe --data_dir data/hsi_v8-80_10_10_importance-new/hsi \
    --batch_size 256 --epochs 12 --lambda_sam 0.1 \
    --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
    --architecture efficient --fusion_type gated <FIX> --seed 42 --run_tag arch-efficient
```

`--fusion_type gated` is mandatory on the `efficient` arm (§E3b); the argparse default `se_gate`
is rejected by `build_model`. Compare against B0 (`abl-base`, 0.9353 balanced / 0.8920 macro-F1) —
same 12-epoch schedule. Budget ~6 h, ~10 h and ~6–10 h respectively at the measured per-epoch
times for these architectures.

**If neither probe fixes it**, stop. E3 stays a documented pipeline incompatibility (§10.8), which
is a legitimate thing for the paper to report, and `efficient` is not worth 6–10 h to confirm a
third time.

### Track B — two more RQ0 seeds (optional)

RQ0 is now the paper's weakest claim: **+3.91 ± 2.26 points** of balanced accuracy over three
seeds, with the macro-F1 ordering inverting at seed 1 (§10.1). Two more seeds would roughly halve
the standard error on the claim the abstract leads with.

```bash
for S in 13 23; do
  for D in hsi rgb; do
    python train.py --profile paper_recipe \
        --data_dir data/hsi_v8-80_10_10_importance-new/$D \
        --batch_size 256 --epochs 20 \
        --lambda_sam 0.1 \
        --train_subsample_frac 0.102 --val_subsample_frac 0.0986 \
        --seed $S --run_tag rq0-s$S
  done
done
```

20 epochs, matching the headline block. ~4 runs, ~2–4 h each.

### After every run in either track

```bash
python scripts/latent_space.py --run_dir <run> --max_samples 900 --batch_size 64
python scripts/compare_runs.py 'experiments/*arch*' --per_patient --csv findings/arch.csv
python scripts/compare_runs.py 'experiments/*rq0*'  --per_patient --csv findings/rq0.csv
```

FLOPs need no separate step — `train_example_v18.py` counts its own correctly as of the
2026-09-19 compiled-core fix.

---

## 13. Pre-manuscript audit (2026-09-20)

Everything below was checked against the repository, not assumed.

### 13.1 Experiments — complete. One item deliberately skipped.

| Item | Status |
| --- | --- |
| P1–P3 scripts, B0, E1, E2, E4, E5, E6, E7, E8, E9, E10 | complete |
| E7 extended to 5 seeds (42, 1, 7, 13, 23) | complete |
| Track A probes (`--recon_mode none`, `--max_gradient_norm 5.0`) | complete — **both failed** |
| E3b `efficient` | **skipped on purpose — do not run** |

**Track A is closed and the answer is negative.** Both probes leave validation macro-F1 pinned at
exactly **0.2583 for all three epochs**, identical to the untouched arm. Neither the auxiliary
decoder nor the clipping threshold is the cause, on top of learning rate, EMA and the data already
ruled out in §10.8. Per the §12 decision rule, A3–A5 and E3b stay unrun: `efficient` is the fourth
member of a family where four separate interventions have now failed, and 6–10 h to confirm a
third collapse buys nothing. **E3 is a documented pipeline incompatibility and the manuscript
should report it as one.**

### 13.2 RQ0 at five seeds — the claim is now much stronger than at three

| seed | HSI bal | RGB bal | Δ bal | HSI F1 | RGB F1 | Δ F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8961 | 0.8818 | +1.43 | 0.8077 | 0.8202 | −0.0125 |
| 7 | 0.9341 | 0.8897 | +4.45 | 0.8962 | 0.8311 | +0.0651 |
| 13 | 0.9276 | 0.8879 | +3.97 | 0.8689 | 0.8467 | +0.0222 |
| 23 | 0.9378 | 0.8889 | +4.88 | 0.8976 | 0.8365 | +0.0611 |
| 42 | 0.9437 | 0.8851 | +5.86 | 0.9036 | 0.8217 | +0.0820 |

**Δ balanced accuracy +4.12 ± 1.66 pts (sem 0.74), positive at 5/5 seeds.**
**Δ macro-F1 +0.0436 ± 0.0382 (sem 0.0171), positive at 4/5.**

Balanced accuracy is now a defensible claim — every seed agrees and the mean is 5.6 standard
errors from zero. Macro-F1 remains the weaker of the two and should be reported with its
inversion at seed 1 stated, not buried. Against manuscript v8's +15.02 / +0.1298 from one seed on
the legacy RGB build.

### 13.3 The manuscript itself has not been updated at all

The 2026-09-16 rewrite restructured and regenerated the LaTeX (938 → 1,093 lines) but **changed no
numbers**. The current `.md` and `.tex` still carry, verbatim:

* `90.73 %` / `75.71 %` balanced accuracy and `+15.02` (abstract, §7.2, §7.3)
* `0.8580` / `0.7282` macro-F1 and `+0.1298`
* nine occurrences of `27.43 M` / `61×`
* **zero** references to any run from 2026-09-14 onward

`.tex` and `.md` agree with each other, so both need the same edits.

### 13.4 Two provenance corrections — §7.4 is NOT unevidenced

Manuscript §7.4 states "No artefacts for these runs survive in the repository; the whole subsection
is *(prior)*", and Appendix A repeats it. **That is wrong. All four rows are recoverable:**

| §7.4 row | artefact | matches |
| --- | --- | --- |
| MedMamba-SS-TRM, patch, 25.80 / 25.73 / 0.2121 | `experiments/20260905_034435_pad_v6-undersample…` | exact |
| MedMamba-SS-TRM, matched, 30.72 / 25.80 / 0.2281 | `experiments/20260905_133922_pad_v6-undersample…` | exact |
| MedMamba, patch, 27.55 / 24.78 / 0.2158 | `medmamba-original/MedMamba/runs_hsi/20260905_102039_pad_v6-undersample…` | exact |
| MedMamba-SS-TRM, as run | `experiments/20260905_022614_pad_v6-undersample…` | 26.42 / 25.51 / 0.2120 |

§7.4 can be re-tagged *(verified)* and Appendix A's "artefacts referenced but not present" list
corrected.

### 13.5 Three things v8 could not claim that are now available

* **§7.7 reconstruction on the headline configuration.** v8's table comes from
  `20260904_230812` — a different config (`per_patch_zscore`, undersampled build), stated as a
  caveat. `ref20` has its own: SAM 9.466° → 5.342°, RMSE 0.1928 → 0.0957, PSNR 14.76 → 21.80,
  band SSIM 0.7148 → 0.8968 over 20 epochs. The caveat disappears.
* **Shallow probes under matched normalization.** v8 notes the probes used `per_patch_zscore`
  while the runs used `global_zscore`, "so the probe is a floor rather than a matched control".
  The new build carries `global_zscore` probes: majority 0.6341 / 0.3333, mean 0.7023 / 0.6755 /
  0.6131, **mean+std 0.8181 / 0.7815 / 0.7109**, flat 0.7004 / 0.5616 / 0.5473. Now a matched
  control, and the "flat beats nothing" observation survives (3,872 dims lose to 64).
* **Five seeds** — the rating document's lowest score is "Experimental rigour 5.5 — one seed, five
  test patients, one primary dataset". Two thirds of that sentence is now obsolete.

### 13.6 Figures — four encode numbers that changed, and there is no generation script

`paper/figures/` holds all nine referenced figures as `.svg` + `.pdf`. **No script generates
them**; they are hand-authored, so updating is manual.

| figure | action |
| --- | --- |
| `fig7_rq0_matched.svg` | **must change** — the RQ0 panel it draws is +15.02/+0.1298; now +4.12 ± 1.66 over five seeds |
| `fig4_cost.svg` | **must change** — FLOPs are measured now, and the same-task hierarchical comparison (0.278 vs 6.203 GF) did not exist |
| `fig8_recursion_schedule.svg` | **should change** — the depth axis now has quality data attached (flat) |
| `fig5_protocol.svg` | **check** — it depicts "the controlled comparison this work specifies", which has now been run |
| `fig2`, `fig3`, `fig6`, `fig10`, `fig1` | structural/architectural — no change expected |

### 13.7 Nothing missing in

Run artefacts (every cited run has `test_report.json`, working FLOPs, `latent_space/`,
reconstruction figures, per-patient metrics), validity gates, the test suite (492 passed,
1 skipped), the tooling, the datasets, or the LaTeX toolchain.

### 13.8 Verdict

**No experiment, artefact or measurement blocks the manuscript.** The remaining work is the v9
revision plus four figure updates.
