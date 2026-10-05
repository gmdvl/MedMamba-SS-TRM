# Manuscript v18 — Evidence Audit, Experiment Triage and Decision Log

- **Date:** 2026-09-26, under a 7-hour completion deadline.
- **Manuscript:** `paper/draft/MedMamba-SS-TRM_manuscript_v18.md` and `paper/draft/MedMamba-SS-TRM_supplement_v18.md`, built from `paper/source/v18_parts/` with `build_v18.py`.
- **Evidence base:** every hyperspectral result is from the build `data/hsi_v8-80_10_10_importance-new`, as in v17. Nothing trained on the new training-only build (`data/hsi_v9-trainsel`) finished, so no result from it is reported. Only the build's band statistics are used (Sec. III-A).
- **Deferred work:** `FUTURE_EXPERIMENTS.md` at the repository root.

**Evidence levels:**

| Level | Meaning |
| --- | --- |
| L1 | Directly verified: a run file, JSON, source code or the published paper |
| L2 | Derived from L1 data by a reproducible calculation |
| L3 | Interpretation |
| L4 | Unknown |

---

## 1. Paper completion status

| Part | Status |
| --- | --- |
| Architecture, method and equations (Sec. IV) | Unchanged from v17. It was verified against the code in the v12–v16 audits. |
| Results on the v8 build (Sec. VI) | Unchanged from v17. Traced to `paper/source/v19_analysis.json` and the run directories (v16 resolution log §5). |
| **New today, verified** | (1) The training-only band re-selection bounds the leak's effect on the input bands (Sec. III-A, VII-E). (2) The MedMamba-SS "chance" result is partly a BatchNorm evaluation-mode artefact (Sec. VI-D). (3) The CARL description was corrected to what its full text states (Sec. II-E, Table I). (4) Two Saad-group papers are cited. (5) HMI-LUSC is cited as the second-dataset target (Future Work). |
| **New today, from ≤ 2 h runs (finished and validated)** | (6) HybridSN and SpectralFormer at five seeds, seed-paired with MedMamba-SS-TRM (Table X, VI-A): HybridSN leads pooled and on validation at every seed. (7) Zero-shot band removal with η fixed: 0.9086 BA at 16 bands against 0.3845 (VI-C, Fig. 9 regenerated as `fig_bands_v18`). |
| Stopped or deferred | K = 2 non-recursive control (measured ≈ 3.7 h), MedMamba matched selection, every new-build and cross-validation run, HMI-LUSC; see `FUTURE_EXPERIMENTS.md` |
| Moved to the supplement | The training-configuration table (S1), the implementation safeguards (S2), and the learning-curve, t-SNE and reconstruction figures (S1–S3). The Discussion was condensed from 6 to 4 subsections. |
| Incomplete, and disclosed in the paper | Band selection saw held-out labels (Sec. III-A). Test scores were consulted during recipe development (Sec. V-C). The baselines and most ablations are single runs. MedMamba's checkpoint selection is asymmetric. There is one hyperspectral dataset. |

---

## 2. Evidence matrix (main claims)

| Paper claim | Evidence | Level | Verified? | Missing evidence | ≤ 2 h? | Action |
| --- | --- | --- | --- | --- | --- | --- |
| MedMamba-SS-TRM has 446,409 parameters at 32 and at 3 bands | `backbone_num_params` in both seed-42 `config.json` files | L1 | Yes | — | — | Keep |
| Core arithmetic 95.72 MFLOPs per application; 6.203 GF total; hierarchy 0.278 GF (6.2× fewer parameters, 22.3× more arithmetic) | `scripts/flops_report_v16.py` outputs; `test_report.json["flops"]`; eq. (23) | L1/L2 | Yes | — | — | Keep |
| Test BA 92.78 ± 1.87 over five seeds; first at 4 of 5 seeds | `v19_analysis.json` → `modality.summary.test_hsi` | L1/L2 | Yes | — | — | Keep |
| Pooled BA: HybridSN 87.14, RGB probe 86.50, MedMamba-SS-TRM 84.49 | `v19_analysis.json` → `comparison` | L1 | Yes | Baselines are single runs | Partly (F10) | Keep; baseline seeds launched |
| HSI − RGB: +4.12 ± 1.66 test, −5.81 ± 4.78 validation, −1.04 pooled | `v19_analysis.json` → `modality.summary.*_delta` | L2 | Yes | New-build retrain (F1) | No | Keep with caveat |
| Patient 304 explains about two-thirds of the validation reversal | `v19_analysis.json` per-patient data (v14 memory, recomputed in v16) | L2 | Yes | — | — | Keep |
| Retrained at 8 bands: 96 % of BA | run test reports `20260919_020050` vs `20260915_031555` | L2 | Yes | Single seed | No | Keep, as single-run |
| Zero-shot band removal collapses with η = C, but with η fixed at 32 the checkpoint keeps 0.9086 BA at 16 bands (0.5599 at 8); identity check passes at 32 | `20260915_031356_*/band_decimation/summary.json`, `.../band_decimation_fixed_eta/summary.json` (2026-09-26, 17:26–17:58) | L1 | Yes | More checkpoints and decimations | Done | **Updated**: VI-C, Fig. 9 (`fig_bands_v18`), Abstract, Contribution 1, VII-B, VII-C, VIII |
| Depth flat (21 applications match 63) | depth-run test reports | L1 | Yes | Seeds | No | Keep, as single-run |
| MedMamba-SS collapse is a BN evaluation-mode artefact; ≤ 0.49 BA with batch statistics | `experiments/20260915_083810_*/eval_mode_check_v20.json`, `..._152009_*/eval_mode_check_v20.json` (run 2026-09-26 on CPU, 9,000 stratified test patches, shuffled batches) | L1 | Yes | Full split, more checkpoints | — | **Added (Sec. VI-D)**; scope stated |
| Training-only re-selection: same layout, max shift 5.1 nm, 8 of 32 identical, gain reference +0.25 % | `data/hsi_v9-trainsel/{selected_wavelengths.npy, pass1_scope_report.json, capture_gain_report.json}`; `logs/v20/prep.log` | L1/L2 | Yes | Retraining (F1) | No | **Added (Sec. III-A)**; the paper says the effect on results is not measured |
| CARL: wavelength positional encoding, attention distillation, porcine organs, 19 classes, 100 channels over 500–1,000 nm | arXiv 2504.19223 HTML full text (fetched 2026-09-26) | L1 | Yes | Subject-disjointness | — | **Corrected**: "subject-disjoint" removed |
| Fatima et al. 2025 review: scarce annotations, compute cost and generalization are the open problems | PDF, Sec. 8.4 (read 2026-09-26) | L1 | Yes | — | — | Added (Sec. I) |
| Fatima et al. 2026: incremental learning across magnifications with SegFormer and distillation | Europe PMC record and abstract (2026-09-26) | L1 | Yes, abstract only | Full text | — | Added (Sec. II-D); claim limited to the abstract |
| HMI-LUSC: 10 patients, 62 images, 61 bands 450–750 nm, H&E 10×, pixel tumour masks, baselines not reported as patient-disjoint | PMC13003143 (2026-09-26); raw headers on disk | L1 | Yes | — | — | Cited in Future Work only |
| Test scores of development runs were recorded and compared | Run-ranking notes (v15 audit F1) | L1 (existence) | **Partly** | Which decisions used them | No (author input) | **CLAIM REQUIRES VERIFICATION** — kept as the conservative disclosure in Sec. V-C and VII-E |
| BAT-Former, DOFA, ChannelViT and MelanoSpec-SSM descriptions | Verified in earlier sessions (v12/v16 logs) | L1 (earlier) | Not re-checked today | — | — | Keep; flagged for a final citation check |
| Hardware: RTX 5060 Ti 16 GB, PyTorch 2.11 + CUDA 12.8 | `config.json` (`torch_version` 2.11.0+cu128, `device` cuda); `nvidia-smi` | L1 | Yes | — | — | Keep |

---

## 3. Experiment triage

Runtimes are measured values from existing runs (see `FUTURE_EXPERIMENTS.md`, "Measured runtimes").

| Experiment | Purpose | Estimated runtime | Evidence value | Run now? | Reason |
| --- | --- | ---: | --- | --- | --- |
| MedMamba-SS eval-mode diagnostic | Explain the "chance" result | 3 min (measured) | High | **Ran (CPU, sandbox)** | Existing checkpoints; light |
| Band re-selection check | Bound the leak's effect on the input | 0 (build existed) | High | **Done (analysis only)** | Build finished at 12:49 |
| HybridSN + SpectralFormer, seeds 1, 7, 13, 23 | Seed-paired baselines | 4 × 24 + 4 × 6 min, sequential | High | **Done 17:26** (after one OOM restart) | Measured runtimes fit in 2 h |
| K = 2 non-recursive control, seed 42 | Isolate recursion | measured 1.12 s/it → ≈ 3.7 h | High | **Launched 15:59, stopped 16:02** | Measured runtime exceeds 2 h; deferred (F5) |
| Zero-shot with fixed η, 32/16/8 bands | Separate the encoding's share of the zero-shot failure | ≈ 30 min (≈ 10 min per band count, measured) | Medium-high | **Done 17:58** (third attempt, alone) | Killed by OOM at 16:14 and stopped at 16:27 to protect the baselines; run alone after they finished |
| Headline pair on the new build (F1) | Remove the band-selection caveat | ≈ 60 h | Critical | No | Category C |
| Patient 5-fold CV (F3) | Remove test exposure | 40–70 h | Critical | No | Category C |
| MedMamba with matched selection (F4) | Remove the baseline asymmetry | 3 × 6.26 h | High | No | Category C |
| HMI-LUSC CV (F8) | Second dataset | ≫ 2 h (54 min per epoch observed) | High | No | Category C |
| Multi-seed ablations and controls (F5, F6) | Replication | 5.5 h per run | High/Medium | No | Category C |
| New-build depth, bands and hierarchy (F7) | Consistency | 2.4–5.8 h per run | Medium | No | Category C |
| MedMamba-SS with batch-independent normalization (F9) | Follow-up to the diagnostic | code + 5.8 h | Medium | No | Needs code |

---

## 4. Experiment decision log

| Time (EDT) | Decision / event | Evidence |
| --- | --- | --- |
| 12:10–12:49 | `data/hsi_v9-trainsel` prepared with training-only Pass 1 (queue from the previous session) | `logs/v20/prep.log`, `pass1_scope_report.json` |
| 12:11 | All P2 jobs failed at start: they began before the build existed ("Missing required file ... X_train.npy") | `logs/v20/*.log` |
| 12:15–14:39 | HMI-LUSC fold 0 and the new-build seed-1 headline run ran in parallel; they reached epoch 4 and epoch 4–5 when the logs stop | `logs/v20/lusc-f0-trm.log`, `logs/v20/head-s1-hsi.log` |
| 12:50 | Eval-mode check failed: `_rebuild_model` passed `classifier_dropout` to `split` | `logs/v20/evalmode_*.log` |
| ≈ 15:45 | **Decision:** stop all long runs; report on the v8 build only | 7 h deadline; measured runtimes |
| ≈ 15:47 | Fixed `check_eval_mode_v20.py` (mirrors `build_model_optimal`) and added a regression test | `tests/test_check_eval_mode_v20.py` (4 pass) |
| ≈ 15:50 | First diagnostic used class-ordered batches: rejected as a biased test, fixed (seeded shuffle) and rerun | script diff; `eval_mode_check_v20.json` |
| ≈ 15:55 | CARL subject-disjointness not found in the accessible full text: wording corrected | arXiv HTML |
| 15:59 | Launched `scripts/short_runs_v18.sh` on the host GPU (env `gmedmamba`, torch 2.11.0+cu128). **Hard stop 17:59** | `logs/v18short/started.txt` |

| 16:02 | **K = 2 control stopped.** Measured 1.12 s/it with the other two chains running, so 978 it × 12 epochs ≈ 3.7 h of training before validation, which does not fit the 2 h rule. Deferred to F5 (now including seed 42). No partial result is used | `logs/v18short/k2-control.log` |

| 16:06–16:13 | SpectralFormer s1 and s7 finished (test BA 0.8175, 0.8940) | `experiments/20260926_155915_*`, `20260926_160639_*` |
| 16:14:08 | I started HybridSN s13 and s23 in parallel to beat the hard stop | `logs/v18short/hybridsn-s*-parallel.log` |
| 16:14:25 | **systemd-oomd killed every job launched through flatpak-spawn** (memory and swap above 90 %). My parallel launch caused it. Lost: SpectralFormer s13 (epoch 3), both HybridSN runs (data loading), fixed-η (inside C = 32) | host journal (`systemd-oomd ... flatpak-session-helper.service ... oom-kill`) |
| 16:26 | Relaunched as a separate systemd unit `v18-short-b` (`scripts/short_runs_v18b.sh`): one baseline chain plus fixed-η | `logs/v18short/restarted.txt` |
| 16:27 | Swap rose by about 2 GB/min toward the kill threshold. **Fixed-η stopped** to protect the baseline chain; deferred (F11) | `free -m` samples 19.2 → 21.2 GB in 40 s |

| 16:26–17:26 | Baseline chain (relaunch) finished: SpectralFormer s13 and s23; HybridSN s1, s7, s13, s23 (≈ 12 min each when run alone) | `experiments/20260926_16*_baseline_s*`, `logs/v18short/*-b.log` |
| 17:26 | Fixed-η started **alone** (`v18-fixed-eta` unit, `timeout` at 17:59) | `logs/v18short/fixed-eta-c.log` |

| 17:58:49 | Fixed-η finished inside the hard stop; `[identity] C=32 ... OK` (0.90365 vs 0.90363) | `logs/v18short/fixed-eta-c.log` |
| 17:59 | **Experimentation stopped.** No job running (`systemctl --user is-active` inactive for both units; GPU holds only desktop processes) | `nvidia-smi` |

Rows for results and the hard stop are appended in §6.

---

## 5. Integrity checks performed

- **Numeric diff v17 → v18** (main + supplement, citation numbers excluded). The added numbers are exactly the band re-selection values, the eval-mode diagnostic values, the CARL facts and the HMI-LUSC facts listed in §2. The only removed numbers belong to the rewritten Future Work paragraph ("2024", "21"). No v17 result changed.
- **Build:** 0 placeholders and 0 pending markers. Tables and figures are numbered in placement order. The supplement tables are S1 and S2.
- **Commands** in `FUTURE_EXPERIMENTS.md`: arguments were checked against the parsers and existing `config.json` files, and unverifiable ones are marked "COMMAND NOT VERIFIED".
- **Tests:** 15 pass (the v20 tests plus the frozen-file guard); no frozen file was edited.
- **Final build (18:0x):** 16 tables and 12 figures in placement order; supplement S-A to S-C with Tables S1–S2 and Figs. S1–S3. Every section and supplementary cross-reference resolves, every image file exists, and the 5 Mermaid diagrams are identical to v17 (already render-checked). Abstract 249 words. No placeholders.
- **Speculative-wording scan** of all new sentences: only "expected calibration error" (a metric name) and one Future Work recommendation ("should be retrained").

---

## 6. Short-run results (appended when available)

Source: `paper/source/v18_short_analysis.json` (from `scripts/analysis_v18_short.py`, same metric code as the published tables). Check: the seed-42 rows reproduce Table X exactly (HybridSN 88.85 / 84.94 / 87.14; SpectralFormer 82.45 / 80.73 / 81.78).

| Model, five seeds | Test BA (%) | Val. BA (%) | Pooled BA (%) | MedMamba-SS-TRM − model, paired by seed: test / val. / pooled (points) | MedMamba-SS-TRM ahead at (test / val. / pooled) |
| --- | ---: | ---: | ---: | --- | --- |
| HybridSN | 89.40 ± 0.47 | 84.37 ± 1.04 | 87.12 ± 0.66 | +3.38 ± 2.14 / −8.19 ± 4.44 / −2.62 ± 1.34 | 4/5 / 0/5 / 0/5 |
| SpectralFormer | 83.02 ± 3.98 | 79.29 ± 1.29 | 81.30 ± 2.33 | +9.77 ± 4.25 / −3.11 ± 5.64 / +3.19 ± 2.94 | 5/5 / 1/5 / 4/5 |
| MedMamba-SS-TRM (existing) | 92.78 ± 1.87 | 76.18 ± 4.59 | 84.49 ± 1.45 | — | — |

**Paper changes driven by this result:**
- Table X gains two five-seed rows.
- VI-A gains a "Seed-paired baselines" paragraph.
- V-E, VII-A (Table XVI), VII-B and VII-C are updated, together with the Abstract, Contribution 2 and the Conclusion.
- The pooled statement is now "HybridSN leads at every seed", where it was previously a single-seed statement.

**Zero-shot band removal, seed-42 checkpoint** (`v18_short_analysis.json → zero_shot`):

| Bands | η = C (as defined): BA / macro-F1 | η fixed at 32: BA / macro-F1 |
| ---: | --- | --- |
| 32 | 0.9437 / 0.9037 | 0.9437 / 0.9037 (identity) |
| 16 | 0.3845 / 0.1854 | **0.9086 / 0.8256** |
| 8 | 0.3329 / 0.1289 | 0.5599 / 0.4983 |

Derived (L2): both decimated sets keep the end bands (400.48 and 938.16 nm, from the `wavelengths_nm` of `band_decimation/summary.json`), so the normalization range is unchanged and the difference is due to η alone.

**Not obtained today:** the K = 2 control (stopped, F5) and MedMamba seeds (F4).

---

## 7. Post-deadline edits (2026-09-26, on request)

- **Reconstruction figure** moved back from the supplement to the main text (now Fig. 13, Sec. VI-J). It had been moved only to shorten the paper. The supplement keeps Figs. S1 (learning curves) and S2 (t-SNE).
- **Mermaid diagrams simplified** (Figs. 3–6): each is now 4–7 short-labelled nodes per panel, with the same colour code. Detail was moved into the captions and the existing change tables (Tables V, VI, VIII, IX). Content was checked against Sec. IV. All five diagrams re-rendered with mermaid-cli. Table and figure numbering is still in placement order: captions that referred forward to later tables were reworded.
