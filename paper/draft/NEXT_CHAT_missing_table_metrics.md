# Handoff — fill the missing metrics in Tables V, VII, VIII and IX of the v11 manuscript

> **STATUS: COMPLETED 2026-09-25 (same session).** Every cell in Section 1 is filled; Tables VII and IX
> are now generated (`tables_v19.py::ablations()` / `pad()`, placeholders `{{ABLATION_TABLE}}` /
> `{{PAD_TABLE}}` in `q4b.md`, filled into `q4b_filled.md`, which `rebuild.py` reads). Table VIII's last
> column is now "ECE, temperature fitted on the other set" (cross-fit). `heldout_eval_v19.py` needed one fix
> for hierarchical runs (drop `classifier_dropout`, as training did). Findings while filling: MedMamba-SS
> leads PAD accuracy, specificity, κ and ROC-AUC (0.798); for the 3-band model a test-fitted temperature
> *lowers* validation ECE (0.070 → 0.057).
>
> **Section 3 items 1–3 DONE 2026-09-25 (second session).** Table VIII is now the calibration table for
> every model (`tables_v19.py::calibration()`): MedMamba-SS-TRM 5-seed rows + MedMamba, HybridSN,
> SpectralFormer and the two mean+s.d. probes at seed 42, with MCE and the cross-fitted temperature for all
> (`analysis_v19.py::temperature_crossfit`). Item 1 is folded into it (Table V's note points to VI-H) rather
> than adding a duplicate ECE column. MCE is computed over bins holding ≥ 0.1 % of patches (`mce_floor`),
> because MedMamba's standard MCE (0.474) comes from a 7-patch bin; the note says so. New finding: all six
> models with validation predictions have T > 1 on validation and T < 1 on test, so the transfer failure
> is a split property, not ours. **Item 4 (Table VII val/pooled, ~4 GPU-h) is still open; only on request.**
> Pre-change backup: `paper/source/v11_backup_before_calib_table_2026-09-25.tar.gz`.

**Written 2026-09-25** at the end of the session that produced
`paper/draft/MedMamba-SS-TRM_manuscript_v11.md`. Hand this file to a new chat as its first message; it
contains everything needed, and the new chat should not need to re-derive anything below.

**Goal.** Every "—" cell in four tables of the manuscript is either filled with a verified number or kept
with a stated reason. No other change to the paper's claims is in scope unless a filled number contradicts
the text, in which case the text must be updated to match.

---

## 0. Read this first

### 0.1 What the project is

A Master's thesis on hyperspectral breast-histology classification. Two models: **MedMamba-SS** (MedMamba
with a band-count-agnostic spectral pathway) and **MedMamba-SS-TRM** (the same pathway with one recursively
applied core in place of the hierarchy). Code identifiers still use the old names (`gmedmamba.py`,
`GMedMamba`, `GMedMambaRecursive`); the paper must only use the new names.

### 0.2 The manuscript is generated — never edit the output file

`paper/draft/MedMamba-SS-TRM_manuscript_v11.md` is **built** from parts in `paper/source/v11_parts/`:

| File | Content | Edit? |
| --- | --- | --- |
| `q1.md` | front matter, I Introduction, II Literature Review, III Datasets | yes |
| `q2.md` | IV Methodology (eqs. 1–23, four Mermaid diagrams, Algorithm 1) | yes |
| `q3.md` | V Experimental Setup (eqs. 24–29, Table IV hyperparameters, baselines) | yes |
| `q4a.md` | VI-A comparison (Table V), VI-B learning curves, VI-C modality (Table VI) — contains `{{COMPARISON_ROWS}}`, `{{MODALITY_ROWS}}`, `{{MODALITY_TEXT}}` | yes |
| `q4b.md` | VI-D bands, VI-E ablation (**Table VII, hand-written**), VI-F depth, VI-G error, VI-J reconstruction, VI-K skin lesions (**Table IX, hand-written**) | yes |
| `q4h.md` | VI-H calibration (Table VIII) — `{{CALIB_ROWS}}`, `{{CALIB_VAL}}`, `{{CALIB_TEMP}}` | yes |
| `q4c.md` | VI-I explainability | yes |
| `q4l.md` | VI-L consolidated discussion | yes |
| `q5_unfilled.md` | VII Conclusion (`CONCLUSION PENDING` + Future Work list), appendices | yes |
| `fill_v11.py` | fills every placeholder and writes the abstract and conclusion from `paper/source/v19_analysis.json` | yes |
| `build_v11.py` | numbers citations `[@key]` (IEEE first appearance), tables `{{T:key}}` (Roman), figures `{{F:key}}` | only for new references |
| `rebuild.py` | runs everything | no |
| `q4.md`, `q5.md`, `*_filled.md`, `abstract.md` | **regenerated** by `rebuild.py` | **never** |

Rebuild (from the repo root, sandbox Python is fine):

```
python3 paper/source/v11_parts/rebuild.py
```

It prints the table and figure order, the citation count and the word count. The current build is
9,386 prose words, 9 tables, 14 figures, 33 references, equations (1)–(29).

### 0.3 Where numbers come from — never type a result by hand

```
run artefacts ──> scripts/analysis_v19.py ──> paper/source/v19_analysis.json
                                                   │
                  scripts/tables_v19.py  <─────────┤   (prints Markdown tables)
                  paper/source/v11_parts/fill_v11.py  (fills placeholders)
                  scripts/make_figures_v19.py ──> paper/figures/results/*.png|pdf
```

Tables V, VI and VIII are generated. **Tables VII and IX are hand-written in `q4b.md`** and are the main
reason for this handoff; convert them to generated tables (Sections 2.2 and 2.4).

### 0.4 Rules the paper must keep (set by the user)

- IEEE structure and tone. No reference to earlier drafts or versions, no code, script, file or path names,
  no flag names, no "RQ" labels anywhere in the manuscript text.
- Names: MedMamba-SS and MedMamba-SS-TRM only.
- Contribution hierarchy: the two models are the only numbered contributions.
- Report every ranking on test, validation **and** pooled; do not reintroduce "outperforms" claims that the
  pooled column contradicts.
- **Freezing rule:** never edit a file that an audited training entry point imports
  (`gmedmamba.py`, `train_example_v*.py`, `training/trainerg_v*.py`, `scripts/eval_test_split_v16.py`, …).
  New behaviour goes in new files. All v19 scripts listed in Section 4 are new and may be edited.

### 0.5 Environment

- **Sandbox** (where the chat's Bash runs): Python 3.13 with numpy, scikit-learn, scipy and matplotlib.
  **No torch, no GPU.** CPU analysis and plotting run here.
- **Host** (has the GPU, an RTX 5060 Ti 16 GB): reach it with `flatpak-spawn --host`. The CUDA env is
  `~/.local/share/mamba/envs/gmedmamba/bin/python` (torch 2.11 + cu128). Example:

  ```
  flatpak-spawn --host sh -c "cd /data/dante_data/documents/Masters/courses/Thesis/g-medmamba && \
      ~/.local/share/mamba/envs/gmedmamba/bin/python scripts/heldout_eval_v19.py --run_dirs <dirs> \
      > experiments/v19_logs/<name>.log 2>&1"
  ```

  Run long jobs with the Bash tool's `run_in_background`.
- The claude.ai scratchpad is **wiped between user turns**; keep working files in the repo.

### 0.6 Pitfalls already hit

- `pkill -f <pattern>` inside `flatpak-spawn --host sh -c "..."` kills its own shell (exit 144), because
  the pattern appears in its own command line. Use `pgrep` to get PIDs and `kill` them.
- Full-split evaluation of a **recursive** checkpoint takes about 24 minutes alone on the GPU (63 core
  applications per patch) and about 36 minutes when three run in parallel; it is GPU-bound. The
  **hierarchical** MedMamba-SS is about 22× cheaper per patch.
- `reconstruction/samples.npz` needs `np.load(..., allow_pickle=True)`.
- Balanced accuracy on a confidence-ranked subset is meaningless below about 90 % coverage (only IDC remains).
- The recursive PAD-UFES-20 whole-image run (`experiments/20260906_183513_…`) used learning rate 1e-3,
  not 3e-4; do not state 3e-4 for it.

---

## 1. Every missing cell at a glance

| # | Table | Row | Column(s) | Why it is missing | Data on disk? | Fix | GPU? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | V | Majority class | Test F1, Val F1, Pooled F1 | The probe script sets a constant predictor's F1 to `None` by design; the row is hard-coded in `tables_v19.py` | computable from labels | compute macro-F1 of "always IDC" | no |
| 2 | V | MedMamba-SS | Test κ | `analysis_v19.py` keeps only accuracy, BA and F1 for this row (`test_sklearn`) | yes, `test_predictions.npz` and `evaluator_metrics.cohen_kappa` | read or recompute | no |
| 3 | V | MedMamba-SS | Val BA/F1, Pooled BA/F1 | `heldout_eval_v19.py` was never run on the hierarchical checkpoints (skipped because they collapsed) | no | run full-validation evaluation | **yes, minutes** |
| 4 | VII | index encoding (×2), 16 bands, 8 bands, hierarchical | DCIS F1 | Table hand-written from an older brief that lacked these values | **yes**, in `v19_analysis.json` | generate the table | no |
| 5 | VII | full model | Δ macro-F1 | reference row, "—" by design | — | keep | no |
| 6 | VIII | Validation rows | ECE after temperature scaling | T is fitted **on** validation, so a validation value would be in-sample | computable | cross-fit: fit T on test, apply to validation | no |
| 7 | IX | MedMamba-SS | Macro ROC-AUC | Carried over as "—" from an older table and never looked up | **yes**, 0.7984 | read from run | no |

Optional additions that reviewers would also expect are listed in Section 3.

---

## 2. Per-table detail

### 2.1 Table V — comparison on HistologyHSI-BC-Recurrence

Generated by `scripts/tables_v19.py::comparison()` from `v19_analysis.json["comparison"]` and
`["modality"]["per_seed"]["42"]["hsi"]`; filled into `q4a.md` via `{{COMPARISON_ROWS}}`.

**Cell 1 — majority-class F1.** Predicting IDC everywhere (verified in this session):

| | Test | Validation | Pooled |
| --- | ---: | ---: | ---: |
| Accuracy | 0.6901 | 0.6328 | 0.6621 |
| Macro-F1 | **0.2722** | **0.2584** | **0.2656** |
| κ | 0 | 0 | 0 |

Compute it in `analysis_v19.py` (add a `"Majority"` entry to `comparison` with `test`, `val_full`,
`pooled` from `classification_metrics` on a constant prediction) and let `comparison()` read it instead of
the hard-coded row. Note: the hierarchical MedMamba-SS test macro-F1 (0.2741) is essentially the majority
value, which the text can state as evidence that it predicts one class.

**Cell 2 — MedMamba-SS test κ.** `experiments/20260915_083810_10_10_importance-new-hsi_split_hsi_…/`
has `test_predictions.npz` (`true_label`, `predicted_label`, `probs`) and
`test_report.json → evaluator_metrics.cohen_kappa = 0.0025` (verified). The full-channel variant
(`experiments/20260915_152009_…fullchannel…`) has κ 0.1311. In `analysis_v19.py`, replace the
`test_sklearn` extraction for these two runs with `classification_metrics(labels, probs)` from their
`test_predictions.npz`.

**Cell 3 — MedMamba-SS validation and pooled.** Run on the host GPU:

```
~/.local/share/mamba/envs/gmedmamba/bin/python scripts/heldout_eval_v19.py \
    --run_dirs experiments/20260915_083810_* experiments/20260915_152009_*
```

It writes `<run>/heldout_v19/validation/test_predictions.npz` (the file name is the evaluator's; the split
is validation). **This script has only been exercised on recursive runs.** `_rebuild_model`
(from the frozen `scripts/eval_test_split_v16.py`) handles every architecture and the trainer is built with
`ema_rate=None` for non-recursive runs, so it should work; check the first log lines and that the reported
test-equivalent numbers make sense (expect near-chance validation BA). If it fails, fix it in
`heldout_eval_v19.py` (a new file), not in any frozen file. Then add both runs to `analysis_v19.py`'s
comparison block with `val_full`, `pooled`, `test_per_patient`, `val_per_patient`, exactly as done for
`hybridsn`. Mark in the table that MedMamba-SS is the **12-epoch** budget while the other networks are 20
epochs (the row label already says so).

### 2.2 Table VII — component ablations (hand-written in `q4b.md`)

The DCIS F1 values exist in `v19_analysis.json["ablations"][key]["per_class_f1"][1]` (verified):

| Variant | Run prefix | BA | Macro-F1 | DCIS F1 |
| --- | --- | ---: | ---: | ---: |
| full (12-epoch baseline) | `20260915_031555` | 0.9353 | 0.8920 | 0.769 |
| reconstruction off | `20260915_160442` | 0.9225 | 0.8708 | 0.733 |
| index encoding, run a | `20260915_202942` | 0.9291 | 0.8804 | **0.748** |
| index encoding, run b | `20260916_013913` | 0.9287 | 0.8798 | **0.747** |
| 16 bands, retrained | `20260919_015237` | 0.9159 | 0.8636 | **0.717** |
| 8 bands, retrained | `20260919_020050` | 0.9014 | 0.8684 | **0.715** |
| 21 core applications | `20260915_123136` | 0.9378 | 0.8932 | 0.774 |
| hierarchical MedMamba-SS | `20260915_083810` | 0.3342 | 0.2741 | **0.001** |
| full-channel MedMamba-SS (not in table) | `20260915_152009` | 0.4085 | 0.3974 | 0.228 |

The 21-application row comes from `v19_analysis.json["depth"]["21"]` (it has `bal`, `f1`, `dcis_f1`).

Do this: add `tables_v19.py::ablations()` that prints the rows from `analysis_v19.json`, replace the
hand-written table body in `q4b.md` with `{{ABLATION_ROWS}}`, and make `fill_v11.py` substitute it (note:
`rebuild.py` reads `q4b.md` directly, so the substitution must happen on the text `rebuild.py` assembles —
simplest is to have `fill_v11.py` write a `q4b_filled.md` and point `rebuild.py` at it). Keep the Δ column
for the reference row as "—" (it is the reference). A numeric Δ of −0.618 may replace "does not train" for
the hierarchical row, with the words kept in the text.

### 2.3 Table VIII — calibration

Generated by `fill_v11.py` from `v19_analysis.json["modality"]["summary"]` and
`["modality"]["per_seed"][seed][mod]["temperature"]`. The temperature is fitted by `fit_temperature` in
`analysis_v19.py` on each run's **full-validation** probabilities and applied to test.

**Cell 6 — validation "ECE after temperature scaling".** The value is "—" on purpose: applying a
validation-fitted temperature back to validation would be an in-sample number. Two correct ways to fill it:

1. **Cross-fit (recommended):** fit T on the test probabilities and apply it to validation. Add to the
   temperature block in `analysis_v19.py`:
   `t_test = fit_temperature(probs_t, lab_t); vs = classification_metrics(lab_v, apply_temperature(probs_v, t_test))`
   and store `val_ece_crossfit`, `T_test`. Then fill the validation rows with it and rename the column
   "ECE after temperature scaling fitted on the other held-out set". Expect the same story as on test
   (currently: fitting on validation gives T ≈ 1.53 for 32 bands and 1.99 for 3 bands, and test ECE rises
   from 0.056 to 0.095 and from 0.094 to 0.222).
2. In-sample, clearly labelled as such.

Update `{{CALIB_TEMP}}` text in `fill_v11.py` if the cross-fit numbers change the story.

### 2.4 Table IX — whole-image PAD-UFES-20 (hand-written in `q4b.md`)

**Cell 7 — MedMamba-SS macro ROC-AUC = 0.7984** (verified), from
`experiments/20260916_032044_pad_optimal-224_split_rgb_bs32_bf16_focalw_zscore_optimal/test_report.json →
evaluator_metrics.macro_roc_auc`. This is the run whose accuracy / BA / F1 (53.49 / 39.82 / 0.3855) are in
the table. A second run of the same configuration, `20260916_084319_…`, has 0.8062 (acc 0.5407, BA 0.3835,
F1 0.3770).

Sources of every column (all verified this session):

| Column | Source | acc | BA | F1 | precision | specificity | ROC-AUC | κ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MedMamba-SS-TRM | `experiments/20260906_183513_pad_optimal-undersample_recursive_…/test_report.json → evaluator_metrics` | 0.4564 | 0.5020 | 0.4213 | 0.4029 | 0.8865 (paper) | 0.7934 | 0.3054 |
| MedMamba-SS | `experiments/20260916_032044_…split…/test_report.json → evaluator_metrics` (+ `test_predictions.npz`) | 0.5349 | 0.3982 | 0.3855 | 0.3750 | 0.8923 (computed from predictions) | **0.7984** | 0.3516 |
| MedMamba-T | `../medmamba-original/MedMamba/runs/PAD-UFES-20-paper/results/test_metrics.json` (+ `test_probabilities.npy`, `test_labels.npy`, `confusion_matrix.csv`) | 0.4718 | 0.3387 | 0.3353 | 0.3446 | 0.8791 | 0.7145 | compute from `test_probabilities.npy`/`test_labels.npy` |

The MedMamba-T column is a verifiable run (the sibling repository), not merely auxiliary; its split is
image-level 60/10/30 and it has 14.47 M parameters. The MedMamba-SS-TRM specificity (88.65 %) comes from
the older manuscript; recompute it from `20260906_183513_…/test_predictions.npz` for consistency.

Do this: add a `pad` block to `analysis_v19.py` that reads the three sources and recomputes precision,
specificity, κ and ROC-AUC from predictions where available; add `tables_v19.py::pad()`; replace the
hand-written Table IX in `q4b.md` with a placeholder filled by `fill_v11.py`. Consider adding the macro
precision, specificity and κ rows (reviewers compare against the published PAD table, which reports
precision, sensitivity, specificity, F1, overall accuracy and AUC).

---

## 3. Optional additions a reviewer would expect (not required by the user's request)

1. **Test ECE column in Table V.** `classification_metrics` already computed `ece`, `brier_score`, `nll` for
   every model with probabilities (`comparison[*]["test"]`); MedMamba has test probabilities
   (`../medmamba-original/MedMamba/runs_hsi/matched_natural_v8new/predictions_test.npz`, keys `labels`,
   `predictions`, `probabilities`).
2. **Baselines in Table VIII.** HybridSN and SpectralFormer have test and full-validation probabilities
   (`experiments/*_hybridsn_baseline_s42/{test,val}_predictions.npz`,
   `experiments/*_spectralformer_baseline_s42/{test,val}_predictions.npz`); the probes too
   (`data/hsi_v8-80_10_10_importance-new/{hsi,rgb}/shallow_probe_heldout_v19_<kind>_{test,val}.npz`).
3. **MCE** (maximum calibration error) is returned by `training.metrics.calibration_metrics` but not shown.
4. **Validation and pooled columns for Table VII** require full-validation evaluation of the eight ablation
   checkpoints (recursive, about 24–36 minutes each on the GPU, roughly 4 hours in total). This is Future
   Work item 7 in the paper; only do it if the user asks.

---

## 4. File map

| Path | Role |
| --- | --- |
| `paper/draft/MedMamba-SS-TRM_manuscript_v11.md` | the manuscript (generated) |
| `paper/draft/MANUSCRIPT_RATING_v11.md` | current ratings (overall 7.7, reviewer 7.0, Saad-fit 8.7, composite 7.6) |
| `paper/source/v11_parts/` | manuscript parts and build scripts (Section 0.2) |
| `paper/source/v19_analysis.json` | every number the paper reports |
| `paper/figures/results/` | the ten result figures (PNG + PDF) |
| `scripts/analysis_v19.py` | builds `v19_analysis.json` from run artefacts (runs in the sandbox in ~30 s) |
| `scripts/tables_v19.py` | prints Markdown tables from the JSON |
| `scripts/make_figures_v19.py` | all result figures; `python3 scripts/make_figures_v19.py <name …>` |
| `scripts/heldout_eval_v19.py` | full-validation predictions and XAI export for a finished run (**GPU**) |
| `scripts/train_hsi_baseline_v19.py` | trains HybridSN / SpectralFormer under the reported recipe (**GPU**) |
| `scripts/shallow_probe_heldout_v19.py` | refits the probes and scores full test and validation |
| `training/heldout_metrics_v19.py` | `classification_metrics`, `per_patient`, patient bootstrap |
| `training/hsi_baselines_v19.py` | HybridSN and SpectralFormer models |
| `experiments/v19_logs/` | logs of every v19 job |
| `experiments/v19_logs/modality_runs.txt` | the ten seed runs (note: the list starts with a space) |

Key run directories (prefix → content):

| Prefix | Run |
| --- | --- |
| `20260915_031356` / `20260915_031947` | seed-42 MedMamba-SS-TRM, 32-band / 3-band (20 epochs) |
| `20260916_084156`, `20260916_155129`, `20260919_150920`, `20260919_205302` | seeds 1, 7, 13, 23, 32-band |
| `20260916_131540`, `20260916_192436`, `20260919_184739`, `20260920_003948` | seeds 1, 7, 13, 23, 3-band |
| `20260915_031555` | 12-epoch baseline for every ablation |
| `20260925_181908_…hybridsn_baseline_s42`, `20260925_184306_…spectralformer_baseline_s42` | the two HSI baselines |
| `../medmamba-original/MedMamba/runs_hsi/matched_natural_v8new` | MedMamba HSI baseline |

## 5. Definition of done

1. Every cell in Section 1 filled or explicitly kept with its reason stated in the caption or a table note.
2. Tables VII and IX generated, not hand-written; no result number typed by hand.
3. `python3 paper/source/v11_parts/rebuild.py` succeeds and prints no unresolved placeholders.
4. Compliance scan prints nothing:
   `grep -n -i -E "gmedmamba|\.py\b|experiments/|\bRQ[0-9]?\b|legacy|\bv1[0-9]\b|\{\{|\[@|PENDING" paper/draft/MedMamba-SS-TRM_manuscript_v11.md`
5. If any filled number changes a sentence (for example MedMamba-SS validation results in Section VI-A or
   VI-E), the sentence is updated.
6. If the Mermaid blocks were touched, render them with the Mermaid CLI on the host
   (`npx -y -p @mermaid-js/mermaid-cli mmdc` with Puppeteer pointed at `/usr/bin/brave`,
   `PUPPETEER_SKIP_DOWNLOAD=1`; output into a folder inside the repo, delete afterwards).
