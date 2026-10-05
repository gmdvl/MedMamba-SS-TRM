# Manuscript Rating — MedMamba-SS / MedMamba-SS-TRM, v18 (revised)

- **Subject:** `paper/draft/MedMamba-SS-TRM_manuscript_v18.md` and `_supplement_v18.md`, after the easy audit fixes
  - 16,534 prose words
  - 16 tables and 13 figures (5 of them Mermaid diagrams)
  - 29 numbered equations, 1 proposition, 1 algorithm
  - 57 references
- **Rated:** 2026-09-26. The ratings count both audits:
  - `MANUSCRIPT_AUDIT_v18.md`: round 1, 39 findings, 28 fixed or partly fixed.
  - `MANUSCRIPT_AUDIT_v18_r2.md`: round 2, 25 findings open, 4 of them high.
- **Scale:** 1–10.
  - 5 = publishable somewhere with work.
  - 7 = solid; accept at a good applied venue.
  - 8.5+ = strong top-venue paper.
- **Weights for the mix:** overall 30 %, senior researcher/engineer 45 %, Saad-fit 25 %. These are the same weights as every earlier rating.

| Rating | v17 | v18 (first) | **v18 revised** | Change |
| --- | ---: | ---: | ---: | ---: |
| 1. Overall | 7.5 | 7.4 | **7.5** | +0.1 |
| 2. Senior researcher and AI/ML engineer | 7.0 | 7.0 | **7.1** | +0.1 |
| 3. Structural fit to Saad B. Ahmed's papers | 8.8 | 8.6 | **8.6** | 0 |
| 4. Mix (30 / 45 / 25) | 7.6 | 7.5 | **7.6** | +0.1 |

**In one line:** the fixes removed every internal contradiction the audit found, and they put the paper's claims at exactly the strength the evidence supports. That is worth about +0.1 everywhere. The ceiling is still set by evidence that no amount of editing can change:
- Band selection saw held-out labels.
- No patient set is untouched by development.
- Ten held-out patients decide every comparison.
- MedMamba-SS does not train on the hyperspectral data.

---

## 1. Overall — **7.5 / 10**

| Dimension | First v18 | Now | Note |
| --- | ---: | ---: | --- |
| Novelty and positioning | 7.0 | 7.0 | Unchanged. The spectral selective scan in a medical state-space host is new; MelanoSpec-SSM [19] is still not placed in Table I. |
| Technical soundness | 8.5 | 8.5 | Every formula checks. Eq. (15) is now unambiguous. Fig. 5 vs Fig. 6 colour-code clash (round-2 F42). |
| Experimental rigour | 6.5 | 6.5 | No new experiments |
| Calibration of claims | 9.0 | 9.5 | Post-hoc labels, untuned-baseline qualifiers, a Conclusion that separates design from findings, and corrected statistics |
| Writing and structure | 7.5 | 7.7 | Table contradictions gone; one duplicated rationale removed. Repetition and five names for the 3-band arm remain. |
| Reproducibility | 7.0 | 7.5 | Split IDs, the scaling constant and the augmentation magnitudes are now in Table S1. Missing: dataset version, code, the ROI note. |
| Significance | 6.0 | 6.0 | Unchanged. HybridSN leads pooled; the HSI results are provisional. |

### Good

- **Every claim now sits at the right strength.**
  - The abstract says "single-run MedMamba (own recipe)", "untuned HybridSN and SpectralFormer", "one checkpoint, tested post hoc" and "provisional".
  - The Conclusion separates what holds by construction (identical parameters at 3 and 32 bands; 6.2× fewer parameters for 22.3× more arithmetic) from what the experiments add.
  Few manuscripts are this exact about their own scope.
- **The paper is internally consistent.** Two audits recomputed every table total, parameter formula, cost-model value, per-seed statistic, bootstrap count and patient recall. Nothing is fabricated, and the round-1 contradictions (Table XVI, two vs three patients, PSNR) are resolved.
- **The architecture is well explained.** Two base models, two single substitutions, side-by-side diagrams, change tables, and a tensor-shape table.
- **The diagnostic work is excellent.** The batch-norm evaluation-mode diagnosis, the fixed-η check with an identity control, the per-patient error attribution, and the cross-set calibration analysis each explain a result rather than excuse it.
- **The preprocessing is now reproducible in the supplement.** Split patient IDs, the per-capture gain, the global constant with its derivation, and augmentation magnitudes. The fixes also surfaced one new, honest disclosure: the scaling constant, like the gain reference, uses captures from all patients.

### Bad

- **Every hyperspectral number is still provisional.** The band-selection leak is bounded on the input bands (≤ 5.1 nm) but not on the results.
- **No held-out set is untouched.** Validation selects the checkpoints, the test patients were consulted during development, and seed 42 is the best test seed.
- **Contribution 1 still claims a model that does not work on the target data.** MedMamba-SS reaches ≤ 0.49 BA on hyperspectral patches even without the batch-norm artefact. Its evidence comes from MedMamba-SS-TRM.
- **The accuracy story is weak on the fairest comparison.** Pooled over all ten non-training patients, HybridSN (0.57 M params) leads at every seed, and a six-feature colour probe also leads.
- **Small items remain.** A silently ineffective ROI step is described as a choice (round-2 F40), a colour-code clash between Figs. 5 and 6, a missing qualifier in VII-B, and residual repetition.

### To improve

1. **Apply the round-2 small fixes** (F41–F48), none of which needs new information. Clarify the ROI step (F40). Hours, no GPU.
2. **Decide Contribution 1.** Re-scope it to "the band-count-agnostic spectral pathway", with MedMamba-SS as its hierarchical host, and adjust the title to match. This is the last cheap step with a real rating effect.
3. **Retrain on the training-only band build** (`hsi_v9-trainsel`). This removes "provisional".
4. **Patient-level 5-fold CV with the recipe frozen.** It removes the test-exposure and ten-patient problems at once.

---

## 2. As a senior research-paper expert and AI/ML engineer — **7.1 / 10**

*This rating reviews the paper as if for a strong applied venue. Data integrity and claim–evidence alignment weigh most.*

| Dimension | First v18 | Now | Note |
| --- | ---: | ---: | --- |
| Claim–evidence alignment | 7.0 | 7.5 | Abstract, Conclusion and question 3 now match the evidence. Contribution 2 still lacks the qualifier (round-2 F44). |
| Baseline quality | 7.5 | 7.5 | Seed-paired HybridSN and SpectralFormer. MedMamba uses its own recipe and selection. No channel-adaptive baseline. |
| Ablation coverage | 6.0 | 6.0 | Single-seed ablations; no non-recursive control |
| Statistical validity | 6.5 | 6.5 | The misstatements are corrected (8-band loss vs √2σ; leave-one-out wording). n = 10 patients is unchanged. |
| Data and evaluation integrity | 5.5 | 5.5 | More fully disclosed (the scaling constant), but not repaired |
| Engineering credibility | 9.0 | 9.0 | Validity gates, checkpoint re-scoring to 0.0, the failure catalogue |
| Efficiency analysis | 8.5 | 8.5 | The analytic cost model equals measurement. The SS2D benchmark is now correctly attributed to RGB patches. |
| Significance | 5.5 | 5.5 | Unchanged |
| Reproducibility | 7.0 | 7.5 | Split, preprocessing and augmentation are documented. Code, dataset version and the ROI note are missing. |

**Verdict:**
- Major revision at a main-track venue.
- Minor revision at an applied or medical-imaging journal, provided "provisional" stays and Contribution 1 is re-scoped.
- Strong as an M.Sc. thesis chapter.

### Good

- **A reviewer can trust the numbers.** Two independent audits recomputed them, and each traces to a run file or analysis JSON. The SS2D benchmark, the PSNR aggregation and the Eq. (15) residual were checked against the code and logs, and each was corrected where the text was imprecise.
- **The paired-seed evidence is reported against the author's interest.** HybridSN leads pooled at 5/5 seeds, and the paper says so in the abstract. Reviewers reward this.
- **The diagnostics follow good experimental method:** one variable changed at a time, with controls (the batch-norm check with eval vs batch statistics vs train mode; fixed-η with an identity check at 32 bands).
- **The efficiency analysis is correct and transferable.** 95.72 MFLOPs per core application, derived analytically and matched by measurement, and the explicit point that parameter count measures storage, not compute.

### Bad

- **No number was untouched by development.** This is still a reviewer's first objection.
- **The architecture's accuracy benefit is not shown on the fairest comparison.** HybridSN and a colour probe lead pooled.
- **Recursion is never isolated.** Depth is flat (21 applications ≈ 63), halting is disabled, deep supervision is restructured, and no same-size non-recursive core was trained.
- **The ablations are single-seed**, with effects below the seed spread.
- **The baselines are asymmetric:**
  - MedMamba selected epoch 1 on the full validation split, under a different loss and learning rate.
  - HybridSN and SpectralFormer are untuned.
  - The proposed recipe was developed with test exposure.
- **A reproducibility trap.** The recorded prep command requests ROI filtering that silently did nothing. Anyone rerunning a corrected pipeline would get a different dataset.

### To improve

1. **Frozen-recipe patient CV on the training-only band build** (40–70 h measured). It moves integrity, statistics and significance together.
2. **A non-recursive control at matched parameters, at three or more seeds** (≈ 3.7 h per run). Without it, "recursion" is a cost, not a contribution.
3. **Fixed-η evaluation on the four other checkpoints** (≈ 2 h of GPU evaluation). It turns a post-hoc anecdote into a mean ± s.d.
4. **Give MedMamba matched selection** (the same 9.86 % subset, EMA and loss), and **add one channel-adaptive baseline** (BAT-Former).
5. **Release the code, or give an anonymized link for review.** State the ROI step and the dataset version.

---

## 3. Structural fit to Saad B. Ahmed's publication pattern — **8.6 / 10**

*Basis:* his Google Scholar profile, fetched 2026-09-26, and the section structure of two of his papers read in full earlier (MedFormer-UR and the bidirectional-Mamba crop-field paper). This compares observable structure and is not a claim about his personal opinion.
- **Profile:** Lakehead University; interests Explainable AI, Intelligent Systems, Quantum Computing, Early Medical Diagnosis.
- **Recent work:** BAT-Former, QuantFormer, SLIC melanoma pseudo-labels, MedFormer-UR, skin-histology incremental learning, the dermatopathology review, and *Interpretable Deep Transfer Learning Framework for Breast Cancer Histopathology Classification* (ICAIB 2025).

| Element | His pattern | This manuscript | Fit |
| --- | --- | --- | ---: |
| Section order | Intro → Literature Review → **Gaps Identified** → **Dataset** → Methodology → Experimental Results → Conclusion | The same, plus Experimental Setup and a Discussion section | 9.5 |
| Mathematical density | 20–25 equations | 29 equations, a proposition, closed-form parameter counts | 9.5 |
| Architecture diagrams | 5+ | 5 Mermaid diagrams + change tables | 9.0 |
| Algorithm block | Common | Algorithm 1, now unambiguous | 9.0 |
| Comparison table | 6+ published methods | 3 published networks + 4 probes | 7.5 |
| Ablation with deltas | Always | Table XI with Δ macro-F1 | 8.5 |
| Error analysis | Confusion matrix, per-class, learning curves | Confusion matrices and per-patient table in the main text; learning curves in the supplement | 8.5 |
| Calibration and uncertainty | Central (MedFormer-UR: ECE, MCE, Brier, NLL) | All four, plus temperature transfer and selective prediction | 9.5 |
| Explainability | Attribution-style XAI | Band-gate profile and t-SNE | 8.0 |
| Topic | HSI, medical diagnosis, Mamba, band-aware design | All four | 9.5 |
| Contributions | About 4 numbered | 2 numbered + a supporting-components paragraph | 8.0 |
| Limitations and hedging | No limitations section; findings stated plainly | A Limitations subsection; qualifiers now even more consistent | 6.5 |
| Length | About 12–15k words | 16.5k prose words | 7.0 |
| His group's literature | Builds on his line | 7 of 57 references (the BAT-Former reference now correctly attributed). The breast-histopathology ICAIB 2025 paper is uncited. | 8.0 |

**Why unchanged:** the fixes were about accuracy and consistency, not structure. The one change that matters here cuts both ways. The added qualifiers make the paper more exact but more hedged, and his house style is to state results plainly.

### Good

- **The skeleton is his, down to the heading words:** "Literature Review" by method family ending in "Gaps Identified", then a standalone Datasets section before Methodology.
- **The Methodology is as formal as his and more checkable.** The parameter counts are closed-form, with $C$ visibly absent, and the cost model is analytic and matched by measurement.
- **The results cover every element his papers use:** a comparison table, an ablation with deltas, confusion matrices, and a calibration table with exactly MedFormer-UR's metric set, plus selective prediction, which is MedFormer-UR's uncertainty-routing idea.
- **The topic sits where his current papers meet.** Band-aware design (BAT-Former), a bidirectional spectral Mamba (the crop-field paper), and medical HSI and histology (SLIC melanoma, the dermatopathology review, skin histology). All of these are cited where they are relevant.

### Bad

- **More hedged and longer than his template.** A full Limitations subsection, qualifiers in most results paragraphs, and 16.5k words against his roughly 12–15k.
- **The comparison table is thinner than his.** Three published networks, where he uses six or more. His own BAT-Former is discussed but not run.
- **The most relevant paper from his group is uncited:** the ICAIB 2025 breast-histopathology paper, the same organ and task family. Correctly, it was not cited unread.
- **The explainability is lighter than his XAI work.** A gate average and t-SNE, but no attribution maps.
- **Two contributions against his usual four.**

### To improve

1. **Find, read and cite the ICAIB 2025 breast-histopathology paper** where it fits.
2. **Add BAT-Former as a baseline.** This is also the reviewer rating's "channel-adaptive baseline".
3. **Expand the comparison table to five or six published methods**, for example a 3-D CNN, Vision Mamba or VMamba, and BAT-Former.
4. **Move the learning curves into the main text, and add one attribution method over bands** (occlusion or integrated gradients). This also tests the causal reading of the band gate that the paper says is missing.
5. **Cut to about 13–14k words.** Remove the remaining repetitions (MedMamba-SS "chance" appears about 7 times, "storage, not compute" about 5), and keep the hedging in the Limitations subsection rather than in every paragraph.

---

## 4. Mix of all ratings — **7.6 / 10**

0.30 × 7.5 + 0.45 × 7.1 + 0.25 × 8.6 = 2.25 + 3.195 + 2.15 = **7.60**

### The good, in one place

An exact, honest and well-built paper:
- A two-step lineage from two named models.
- A spectral front end whose band-count independence is proven, measured and partly transferable (retraining at 8 bands; zero-shot at 16 bands under a fixed encoding scale).
- A quantified, general lesson that weight-shared recursion trades storage for compute.
- Engineering and diagnostics above the norm.

After two audits, every claim is stated at the strength of its evidence, and the structure matches the supervisor's template almost section for section.

### The bad, in one place

The evidence can't yet carry the accuracy claims. The input bands saw held-out labels, no patient set was untouched by development, and ten patients with one DCIS patient per set decide every comparison. On the fairest comparison, pooled over all ten, a smaller HybridSN and a six-feature colour probe beat the proposed model. MedMamba-SS doesn't train on the target data, and recursion is never isolated. The paper is now candid about all of this. Candour raises trust, not significance.

### What to do next, in order of rating gain per hour

| # | Action | Cost | Expected effect |
| ---: | --- | --- | --- |
| 1 | Round-2 small fixes (F41–F48) and the ROI clarification (F40) | hours, no GPU | Overall +0.1 |
| 2 | Re-scope Contribution 1 and the title | an afternoon, no GPU | Reviewer +0.2, Overall +0.1 |
| 3 | Fixed-η on four more checkpoints | ≈ 2 h of GPU evaluation | Reviewer +0.1 |
| 4 | Cite the ICAIB 2025 paper; add BAT-Former; expand the comparison table | days; ≈ 1 h of GPU per seed | Saad-fit +0.3, Reviewer +0.2 |
| 5 | Retrain on the training-only band build | ≈ 60 h for five seeds | Reviewer +0.3–0.5 |
| 6 | Frozen-recipe patient 5-fold CV | 40–70 h | Reviewer +0.5 or more, if the ranking holds |
| 7 | Non-recursive control, ≥ 3 seeds | ≈ 3.7 h per run | Reviewer +0.2 |

Items 1–4 are likely to bring the mix to about **7.9**. Crossing 8 on the reviewer rating needs items 5–6. That holds only if the retrained, cross-validated results keep the test-set ranking. If they don't, the paper's contribution becomes the architecture and the cost analysis, and it should be framed that way.

---

## Sources

- [Saad B. Ahmed — Google Scholar profile, sorted by date](https://scholar.google.com/citations?hl=es&user=6QZM-vIAAAAJ&view_op=list_works&sortby=pubdate) (fetched 2026-09-26)
- [MedFormer-UR (arXiv 2604.08868)](https://arxiv.org/abs/2604.08868) and [Quantum Enhanced Multi-Scale CNN with Bi-Directional Mamba for Crop Field Analysis (arXiv 2606.17222)](https://arxiv.org/abs/2606.17222): full texts read earlier for section structure
- `paper/draft/MANUSCRIPT_AUDIT_v18.md` (round 1 and its resolution log) and `paper/draft/MANUSCRIPT_AUDIT_v18_r2.md` (round 2): the findings counted in all four ratings
- `paper/draft/MANUSCRIPT_RATING_v18.md`: the first v18 rating, kept for comparison
