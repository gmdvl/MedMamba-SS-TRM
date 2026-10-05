# Manuscript v18 — Evidence Upgrade to Reach ≥ 8/10 (Saad-fit ≈ 9/10)

> **2026-09-26 15:45 — DEADLINE MODE (7 h).** The user asked to finish the paper today with verified evidence only.
> - **v18 = the deadline paper**, built from the v8-build evidence (as in v17) plus today's verified additions: `paper/source/v18_parts/`, audit `paper/draft/V18_EVIDENCE_AUDIT.md`, deferred work in `FUTURE_EXPERIMENTS.md` (repo root).
> - The new-build text (CV, HMI-LUSC, training-only selection with `⟦R:⟧` placeholders) was moved to `paper/source/v19_parts_pending/` (`build_v19.py`). It becomes **v19** once F1–F8 of `FUTURE_EXPERIMENTS.md` have run; the §8 S5 blueprint below applies to v19.
> - **Done 17:26:** HybridSN and SpectralFormer at 5 seeds (v8 build); results in Table X and VI-A. K = 2 stopped (≈ 3.7 h). The first launch was OOM-killed at 16:14 (flatpak-session cgroup), so launch with `systemd-run --user` and at most 2 jobs.
> - ≤ 2 h runs: `scripts/short_runs_v18.sh` (launched 15:59 on the host, env `gmedmamba`), aggregated by `scripts/analysis_v18_short.py` → `paper/source/v18_short_analysis.json`.


**Resume here.** This file is the single source of truth for this task. Every stage has a status
(`TODO`, `DOING`, `DONE`, `WAITING-GPU`, `BLOCKED`) and names its outputs. A new session should:

1. read this file top to bottom;
2. run the **status probe** at the end (§7), which lists which GPU outputs exist;
3. continue with the first stage that is not `DONE`.

- Started: 2026-09-26.
- Input manuscript: `paper/draft/MedMamba-SS-TRM_manuscript_v17.md` (parts in `paper/source/v17_parts/`).
- Input rating: `paper/draft/MANUSCRIPT_RATING_v17.md` (7.5 / 7.0 / 8.8 / composite 7.6).
- Output manuscript: `paper/draft/MedMamba-SS-TRM_manuscript_v18.md` (parts in `paper/source/v18_parts/`) plus supplement `paper/draft/MedMamba-SS-TRM_supplement_v18.md`.
- GPU command sheet: `documentations/16_v20_commands.md`.

---

## 0. Ground rules (read before changing anything)

- **Never fabricate or predict a result.** Every number in v18 comes from a run directory or a JSON produced by `scripts/analysis_v20.py`.
- **A rating is a measurement, not a target.** The ≥ 8 / ≈ 9 goal is met by producing the evidence the v17 rating asked for, not by scoring leniently. §1 states which rating dimension each experiment can move. Whether it moves depends on the result.
- **Frozen files.** Never edit a file that an audited entry point imports (see memory `gmedmamba-freezing-rule`). New behaviour goes into new `*_v20.py` files that import and subclass.
- **No heavy Python in the agent sandbox.** There is no GPU and RAM is tight. All training, preparation, probes and held-out evaluation run on the user's host.
- **One variable at a time.** Every new run is the reference command of `15_v18_commands.md` §3 with only the stated flags changed. The data build is the one deliberate global change: `data/hsi_v9-trainsel`.

---

## 1. Why the v17 scores are where they are, and what can move them

| Rating (v17) | Binding weaknesses (from `MANUSCRIPT_RATING_v17.md`) | What can move it | Movable by text alone? |
| --- | --- | --- | --- |
| Overall 7.5 | Leak and test exposure (rigour 6.0); weak positives (5.0); novelty narrowed by CARL (7.0); length | G1–G3 (clean build, headline re-run), G10 (patient CV), T1 (cut length) | Only length/writing (≈ +0.1) |
| Engineer 7.0 | Significance 4.5; statistical validity 6.0; baselines 6.5; ablations 6.5 | G5 (multi-seed baselines), G6 (non-recursive control), G10 (CV), G7 (MedMamba-SS eval-mode) | **No** |
| Saad-fit 8.8 | Group's histology papers uncited; separate Discussion; one HSI dataset | T2 (read + cite), T3 (shorten VII), G11 (second HSI dataset) | Partly (≈ +0.2) |

**Honest ceiling.** Text alone cannot take the engineer rating to 8: its weakest dimensions are evidence, not prose. Even with all GPU runs done, "significance of the positive results" rises only if MedMamba-SS-TRM is competitive with HybridSN and the colour probe on a clean, patient-level evaluation. If it is not, the paper can still reach ≈ 8 by being a rigorous, well-controlled study with a clear negative or mixed finding (rigour, statistics, ablations and baselines all go to ≥ 8), but the headline claim must then change. The re-rating (S6) will score what the evidence shows.

---

## 2. Stages

| Stage | What | Status | Output |
| --- | --- | --- | --- |
| **S0** | This plan and tracker | DONE | this file |
| **S1** | New code (CPU-testable, no frozen edits) | DONE (C1–C8; 14 tests pass) | see §3 |
| **S2** | GPU command sheet with every command | DONE (`16_v20_commands.md` + resumable `scripts/v20_queue.sh`) | `documentations/16_v20_commands.md` |
| **S3** | Text-only manuscript improvements into v18 parts (numbers still v17's) | DONE (T1, T2, T4, T5; T3 deferred to S5) | `paper/source/v18_parts/`, supplement |
| **S4** | User runs GPU/host commands (G1–G12) | WAITING-GPU | `data/hsi_v9-trainsel/`, `experiments/*_v20-*` |
| **S5** | Aggregate results (`analysis_v20.py`), regenerate figures/tables, write numbers into v18 | TODO (after S4) | `paper/source/v20_analysis.json`, v18 manuscript |
| **S6** | Re-rate v18 | TODO (after S5) | `paper/draft/MANUSCRIPT_RATING_v18.md` |

---

## 3. S1 — code to write (status per file)

| ID | File | Purpose | Status |
| --- | --- | --- | --- |
| C1 | `prepare_histologyhsi_bc_trainsel.py` | Band selection **and** the capture-gain reference computed on training patients only; held-out captures gain-corrected against the training median; split reused from the v8 build | DONE |
| C2 | `scripts/eval_band_decimation_fixed_eta_v20.py` | Zero-shot band decimation with η and the wavelength normalization range held at their 32-band values | DONE |
| C3 | `scripts/check_eval_mode_v20.py` | MedMamba-SS collapse diagnosis: predictions of a saved hierarchical checkpoint in `eval()` vs `train()` mode (BatchNorm statistics) on a class-stratified test sample | DONE |
| C4 | `scripts/export_val_subset_indices_v20.py` + `--val_subset_indices` in `../medmamba-original/MedMamba/train_hsi.py` | Removes the MedMamba selection asymmetry: it selects on the same 32,985-patch subset as every other network | DONE |
| C5 | `scripts/analysis_v20.py` + `paper/source/v20_runs.json` | Registry-driven aggregation (seeds, baselines, ablations, controls, CV folds) reusing `analysis_v19` metric code; no hard-coded run prefixes | DONE |
| C6 | `tests/test_*_v20.py` (5 files) | CPU tests on synthetic fixtures; 14 pass with the frozen-file guard | DONE |
| C7 | `prepare_hmi_lusc_v20.py` | HMI-LUSC → repo .npy layout; calibration, mask purity, glass filter, patient 5-fold CV | DONE |
| C8 | `scripts/v20_done.py`, `scripts/v20_queue.sh` | Resumable queue: skips finished jobs by run tag | DONE |

---

## 4. S4 — experiments (the user runs these; commands in `16_v20_commands.md`)

Priority P1 = required for the ≥ 8 target; P2 = strongly recommended; P3 = optional.

| ID | Pri | Experiment | Closes (rating item) | Est. host time | Status |
| --- | --- | --- | --- | --- | --- |
| G1 | P1 | Re-prep `data/hsi_v9-trainsel` (training-only band selection and gain reference, same split) | Leak (all ratings) | ≈ 2 h CPU | TODO |
| G2 | P1 | Headline pair on G1: 5 seeds × {hsi, rgb}, 20 epochs | Rigour, significance | ≈ 28 h GPU | TODO |
| G3 | P1 | Held-out evaluation (full validation + XAI) for G2 runs; probes on G1 | Pooled/patient analysis | ≈ 3 h GPU + 1 h CPU | TODO |
| G4 | P1 | MedMamba-SS eval-mode check (C3) on the existing hierarchical checkpoint | Contribution-1 framing | ≈ 5 min | TODO |
| G5 | P1 | Baselines on G1 at 5 seeds: HybridSN, SpectralFormer; MedMamba with subset selection (C4) | Baseline quality, significance | ≈ 20 h GPU | TODO |
| G6 | P1 | Non-recursive control (K = 2, same 446,409 parameters) + 12-epoch baseline, 3 seeds | Recursion isolated | ≈ 8 h GPU | TODO |
| G7 | P2 | Ablations on G1 at 3 seeds: recon-off, index encoding | Ablation coverage | ≈ 13 h GPU | TODO |
| G8 | P2 | Zero-shot decimation, both variable and fixed η (C2), on G2 seed 42 | Band-agnostic behaviour | ≈ 1 h GPU | TODO |
| G9 | P2 | Retrained 16/8-band arms and depth sweep T = 1, 2, 4 on G1 | Consistent build for every number | ≈ 8 h GPU | TODO |
| G10 | P1 | Patient-level 5-fold CV with training-only band selection per fold, HSI + RGB, frozen recipe | Test exposure; statistics | ≈ 5 × 7.5 h | TODO |
| G11 | P2 | Second HSI **histology** dataset: HMI-LUSC (10 patients, 61 bands 450–750 nm, pixel masks), patient 5-fold CV, MedMamba-SS-TRM + HybridSN + SpectralFormer | Saad-fit (datasets), band-count agnosticism on a second sensor | ≈ 25 h GPU | TODO |
| G12 | P3 | Hierarchical MedMamba-SS re-run, only if G4 finds an eval-mode defect | Contribution 1 | ≈ 1 h GPU | CONDITIONAL |

---

## 5. S3 — text-only changes (can be done before any GPU result)

| ID | Change | Status |
| --- | --- | --- |
| T1 | Move Table X (configuration), the calibration detail, Appendix A, learning curves, t-SNE and reconstruction figures to a supplement; shorten number-dense prose | DONE: configuration table and safeguards moved to `s1_supplement.md`; curves, t-SNE, reconstruction and calibration detail move in S5 (their numbers change) |
| T2 | Read the three uncited histology papers of Saad B. Ahmed's group; cite only those that are relevant and readable | DONE: cited Fatima et al. 2025 (Discover Appl. Sci. 7:1006, read) and Fatima et al. 2026 (Sci. Rep. 16:9593, abstract read); the ICAIB 2025 breast paper cannot be found online, so it is not cited; PatchGraph-MTFormer dropped (group total 7 of 57) |
| T3 | Shorten Section VII: keep rankings and Limitations; move per-model interpretation to the end of Section VI | DEFERRED to S5 (content depends on results) |
| T4 | Title footnote on the name (lineage, no spatial SSM in the reported core) | DONE (note under the author block) |
| T5 | Placeholders `⟦R:key⟧` for every number that G1–G11 will replace, so S5 is a mechanical fill | DONE: `⟦R:…⟧` placeholders plus `<!-- S5-PENDING -->` markers; `build_v18.py` prints the count |

---

## 6. Decisions log

| Date | Decision | Why |
| --- | --- | --- |
| 2026-09-26 | Keep the v8 patient split (`--split_file`) for G1 | Only band selection and the gain reference change, so the leak's effect is isolated |
| 2026-09-26 | Keep ROI masking as in v8 (inactive) for G1 | Changing it would be a second variable; disclosed in III-A. A masked build can be a later, separate experiment |
| 2026-09-26 | Non-recursive control = K = 2 (`--trm_n_latent 1 --trm_n_improve 1 --trm_deep_supervision_steps 1`) | The model asserts n ≥ 1, so K = 1 needs new model code. K = 2 keeps all 446,409 parameters and removes the recursion depth (63 → 2) |
| 2026-09-26 | Second dataset = HMI-LUSC | HMI-LUSC is histology, has a different sensor with wavelengths, and has pixel masks |
| 2026-09-26 | Breast CV dry run: 45/45 patients tested once, all 7 DCIS patients tested, class coverage OK in every fold | Verified with `--dry_run` in the sandbox |
| 2026-09-26 | SS2D-mixer core not re-run | Measured 39× slower (91.8 h/epoch in the benchmark config); handled by a title footnote instead |

---

## 7. Status probe (run from the repo root on the host)

```bash
ls -d data/hsi_v9-trainsel/{hsi,rgb} 2>/dev/null
ls experiments | grep -E "_v20-" | sort
ls data | grep -E "cv5-f[0-4]" 2>/dev/null
python -c "import json;print(json.load(open('paper/source/v20_runs.json'))['status'])" 2>/dev/null
```

---

## 8. S5 blueprint (how to finish once the runs exist)

1. `python scripts/analysis_v20.py` on the host, then read `paper/source/v20_analysis.json` here.
2. Regenerate figures: copy `scripts/make_figures_v19.py` to `make_figures_v20.py`, pointing it at `data/hsi_v9-trainsel` and `v20_analysis.json` (same palette). New figures: CV per-patient recall (45 patients, both arms), K = 2 vs 63 control across seeds, zero-shot variable vs fixed η, HMI-LUSC per-fold bars.
3. Replace every `⟦R:…⟧` and `S5-PENDING` block in `paper/source/v18_parts/`; `build_v18.py` must report `[pending] 0`.
4. Rewrite Abstract, contributions, Section VI (planned order in the S5-PENDING comment), VII (short: rankings + limitations), VIII. Remove every claim the new evidence does not support, and never reintroduce the phrasings listed in memory `gmedmamba-manuscript-v14`.
5. Numeric audit: every number in v18 must trace to `v20_analysis.json`, a run directory, or unchanged method facts (parameter counts, FLOPs, dataset counts).
6. Mermaid render check (host mermaid-cli, see memory `gmedmamba-manuscript-v10`), then S6 re-rating.

**Rule that protects the CV claim:** after the first CV fold starts, the training recipe is frozen. Any recipe change would require re-running every fold.
