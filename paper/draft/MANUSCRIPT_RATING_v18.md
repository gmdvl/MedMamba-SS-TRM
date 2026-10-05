# Manuscript Rating — MedMamba-SS / MedMamba-SS-TRM, v18

- **Subject:** `paper/draft/MedMamba-SS-TRM_manuscript_v18.md` and `_supplement_v18.md`
  - 16,270 prose words
  - 16 tables and 13 figures (5 of them Mermaid diagrams)
  - 29 numbered equations, 1 proposition, 1 algorithm
  - 57 references
- **Rated:** 2026-09-26, after the forensic audit `MANUSCRIPT_AUDIT_v18.md` (39 findings: 0 critical, 5 high). The ratings count that audit's findings.
- **Scale:** 1–10.
  - 5 = publishable somewhere with work.
  - 7 = solid; accept at a good applied venue.
  - 8.5+ = strong top-venue paper.
- **Previous:** v17 (same four ratings, same weights): overall 7.5, reviewer/engineer 7.0, Saad-fit 8.8, composite 7.6.

| Rating | v17 | **v18** |
| --- | ---: | ---: |
| 1. Overall | 7.5 | **7.4** |
| 2. Senior researcher and AI/ML engineer | 7.0 | **7.0** |
| 3. Structural fit to Saad B. Ahmed's papers | 8.8 | **8.6** |
| 4. Composite (30 / 45 / 25) | 7.6 | **7.5** |

**In one line:** v18's new evidence is real: seed-paired baselines, the fixed-η zero-shot result, the batch-norm diagnosis of MedMamba-SS and the bound on the band re-selection. But part of it points *against* the proposed model: HybridSN now leads the pooled result at every seed. The audit also found stale text in two tables. The net effect on every rating is roughly zero. The rating will move on data, not on writing.

---

## 1. Overall — **7.4 / 10**

| Dimension | Score | Note |
| --- | ---: | --- |
| Novelty and positioning | 7.0 | Similar in purpose to DOFA and CARL. New in its spectral selective scan and its medical state-space host. Gap 2 has not been checked against MelanoSpec-SSM [19]. |
| Technical soundness | 8.5 | Every parameter formula, the cost model and all 29 equations check exactly (audit §0) |
| Experimental rigour | 6.5 | Five seeds on the headline pair, seed-paired baselines, patient bootstrap. But ten held-out patients, one DCIS patient per set, and single-seed ablations. |
| Calibration of claims | 9.0 | Unusually candid. It loses points for the post-hoc fixed-η result in the abstract and a circular Conclusion. |
| Writing and structure | 7.5 | Clear IEEE flow. 16.3k words, heavy repetition, and two stale table statements. |
| Reproducibility | 7.0 | Full model and recipe tables. Missing: the cohort (644 vs 677 images), the split IDs, and code during review. |
| Significance | 6.0 | Main HSI results provisional (band-selection leak). HybridSN leads pooled. The storage-for-compute lesson transfers. |

### Good

- **The paper has a real spine.** Two named base models and two single-component substitutions, each with a side-by-side diagram and a change table (Tables V, VI and VIII). A reader always knows what changed and why.
- **The formal core is exact.** Proposition 1, Table IV, Table VII and Eq. (23) all recompute to the digit. The measured FLOPs lie exactly on the analytic line (95.72 MFLOPs per application). The "parameters measure storage, not compute" result (6.2× fewer parameters, 22.3× more arithmetic) is exact, and it transfers to anyone using weight-shared recursion.
- **v18 turned two weak points into findings.**
  - Zero-shot band transfer no longer "fails". The failure is traced to the encoding scale η: fixing it restores 0.909 BA at 16 bands, with an identity check at 32 bands.
  - The MedMamba-SS "chance" result is diagnosed as a batch-normalization artefact that still leaves the model at ≤ 0.49 BA.
  Both are small, careful experiments that explain rather than excuse.
- **The honesty is exceptional.** The paper discloses, in the abstract, that band selection saw held-out labels, that the test patients were consulted during recipe development, and that HybridSN beats the proposed model pooled at every seed. Few papers report their own adverse evidence this plainly.
- **The per-patient analysis is the best section of the results.** It shows that the HSI-vs-RGB reversal is mostly one patient (304). It confirms this with a network-free linear probe. It ties the errors to patients 136, 304 and 65 rather than to the architecture.

### Bad

- **Every hyperspectral number is still provisional.** The band-selection leak is bounded on the input (≤ 5.1 nm shift) but not on the results, and nothing was retrained.
- **No held-out set is untouched.** The validation patients choose the checkpoints. The test patients were consulted during development. Seed 42, used for every single-seed analysis, is the best test seed (r = −0.95 between test and validation across seeds).
- **Contribution 1 has no evidence of its own on the target data.** MedMamba-SS doesn't train on hyperspectral patches. Its claimed properties are measured inside MedMamba-SS-TRM, and its hierarchy additions are unablated.
- **Question 2 is answered on cost only.** The hierarchy being replaced doesn't train. There is no non-recursive control. Depth buys nothing (21 applications ≈ 63). The Conclusion then presents properties that hold by construction as "structural results".
- **The comparative claims flip with the patient set.** The model is first on 5 test patients and among the weaker networks on 5 validation patients. HybridSN and a 6-feature colour probe lead pooled.
- **Stale and inconsistent text.** Table XVI's title says every baseline is a single run. Its validation row names a leader that scores below its runner-up. Sections VI-G and VIII disagree on two vs three patients.

### To improve

1. **Fix the audit's text defects.** Hours of work, no GPU: Table XVI, "two/three patients", the PSNR aggregation, the Fig. 7 caption, ref [27] and the Algorithm 1 line break.
2. **Re-scope Contribution 1** to "the band-count-agnostic spectral pathway", with MedMamba-SS as its hierarchical host that fails on HSI patches. Rewrite the Conclusion so it separates design properties from empirical findings. This costs nothing and removes the two most exploitable lines of attack.
3. **Retrain the headline pair on the training-only band build** (`hsi_v9-trainsel`). This single experiment removes "provisional" from the abstract.
4. **Patient-level 5-fold CV with the recipe frozen.** It tests all 45 patients and all 7 DCIS patients, and it ends the test-exposure problem.

---

## 2. As a senior research-paper expert and AI/ML engineer — **7.0 / 10**

*This rating reviews the paper as if for a strong applied venue. Data integrity and claim–evidence alignment weigh most, engineering credibility counts, and rhetoric is discounted.*

| Dimension | Score | Note |
| --- | ---: | --- |
| Claim–evidence alignment | 7.0 | Local claims are well hedged. The abstract, Contribution 1 and the Conclusion outrun the evidence. |
| Baseline quality | 7.5 | HybridSN and SpectralFormer are seed-paired (new). MedMamba uses its own recipe and selection. No channel-adaptive baseline. |
| Ablation coverage | 6.0 | Component deltas, a depth sweep and band retraining. All single-seed, and there is no non-recursive control. |
| Statistical validity | 6.5 | Five seeds and a patient bootstrap. But n = 10 patients, the bootstrap is conditional on DCIS, and one comparison uses the wrong s.d. (audit F9). |
| Data and evaluation integrity | 5.5 | Band-selection leak and test exposure. Both are disclosed, and the leak is bounded on the input. |
| Engineering credibility | 9.0 | Validity gates, checkpoint re-scoring to 0.0, the silent-failure catalogue, the batch-norm diagnosis, the fixed-η identity check |
| Efficiency analysis | 8.5 | Analytic cost model = measurement. Memory and parameter memory reported. No latency beyond the SS2D benchmark. |
| Significance | 5.5 | The HSI advantage is patient-dependent. The proposed model is beaten pooled. The hierarchical model doesn't train. |
| Reproducibility | 7.0 | Model and recipe complete. Cohort and split not reconstructible from the paper. |

**Verdict:**
- Major revision at a main-track venue.
- Weak accept to minor revision at an applied or medical-imaging journal, if the text defects are fixed and "provisional" stays.
- Strong as an M.Sc. thesis chapter.

### Good

- **The engineering is trustworthy.** A reviewer who reran this would likely get the same numbers. Every run passes leakage, drift, sensitivity and reconstruction-gradient gates, and the selection metric recomputed from the saved checkpoint differs by 0.0. The audit recomputed dozens of derived values and found no arithmetic error.
- **The paired-seed baselines are the right experiment, reported the right way.** The paper reports them even though they hurt the pooled ranking. The seed-by-seed win counts (4/5 on test, 0/5 on validation, 0/5 pooled vs HybridSN) are more informative than a mean ± s.d.
- **The diagnostic work is excellent.**
  - The batch-norm check: eval mode vs batch statistics vs train mode on a stratified, shuffled sample, with the running-stat mismatch quantified.
  - The fixed-η check: one variable changed, with an identity control.
  Both follow the pattern "form a hypothesis, isolate one variable, include a control".
- **The calibration analysis is unusually deep.** Cross-set temperature transfer shows that miscalibration follows the patient split across all six models. That is a real, reusable observation about small-cohort evaluation.

### Bad

- **The central quality claims don't have an independent test set.** Selection happens on validation, and development touched test. A reviewer's first question: "which number was never seen during development?" The answer is none.
- **The proposed model does not win where it matters most.** Pooled over all ten non-training patients, which is the fairest comparison, HybridSN (0.57 M params) leads at every seed by 2.6 points, and a 6-feature logistic regression on colour statistics also leads. The paper says so. A reviewer will conclude the architecture's accuracy benefit is not shown.
- **Recursion is costed, not justified.** Depth is flat, halting is disabled, deep supervision is restructured, and no same-size non-recursive core was trained. What TRM contributes beyond "a small convolutional network run 63 times" is not isolated.
- **The ablations are single-seed, and their effects are below the seed spread.** The paper says this. The result is that the reconstruction objective and the wavelength encoding have no demonstrated effect.
- **The baseline asymmetries remain:**
  - MedMamba was selected at epoch 1 on the full validation split, under a different loss and learning rate.
  - HybridSN and SpectralFormer are untuned and adapted (no PCA; 11 × 11 windows).
  - The proposed recipe was developed with test exposure.
- **The fixed-η result is post hoc** (one checkpoint, test split, decided after seeing the failure), yet it appears in the abstract.

### To improve

1. **Patient-level CV with a frozen recipe**, on the training-only band build. This addresses the integrity, statistics and significance scores at once. Measured cost: 40–70 h.
2. **A K = 2 or non-recursive control at matched parameters**, at three or more seeds (≈ 3.7 h per run measured). Without it, the reviewer can say the recursion is decoration.
3. **Run the fixed-η evaluation on the other four checkpoints** (≈ 30 min each, evaluation only) and report mean ± s.d. It is cheap, and it turns a post-hoc anecdote into a result.
4. **Give MedMamba matched selection** (9.86 % subset, EMA, the same loss), or drop the "ahead of MedMamba" phrasing from the abstract.
5. **Add one channel-adaptive baseline** (BAT-Former or a ChannelViT-style model). The paper's positioning is against these models, and none is run.

---

## 3. Structural fit to Saad B. Ahmed's publication pattern — **8.6 / 10**

*Basis, stated so the rating can be checked.* This compares the manuscript with the **observable structure** of his work. It is not a claim about his personal opinion.

- **Profile, re-fetched 2026-09-26:** Assistant Professor, Lakehead University. Interests: Explainable AI, Intelligent Systems, Quantum Computing, Early Medical Diagnosis. 1,626 citations, h-index 20.
- **Recent output (2025–2026):**
  - Hyperspectral: BAT-Former, QuantFormer, SLIC melanoma pseudo-labels, the quantum-enhanced CNN with bidirectional Mamba for crop fields, PatchGraph-MTFormer.
  - Medical: MedFormer-UR, skin-histology incremental learning (*Sci. Rep.*), the dermatopathology review, and *Interpretable Deep Transfer Learning Framework for Breast Cancer Histopathology Classification* (ICAIB 2025).
- **Template:** taken from the two full texts read earlier (MedFormer-UR, the crop-field Mamba paper).

| Element | His pattern | v18 | Fit |
| --- | --- | --- | ---: |
| Section order | Intro → Literature Review by family → **Gaps Identified** → **Dataset section** → Methodology → Experimental Results → Conclusion | Exactly this, plus Experimental Setup and a Discussion section | 9.5 |
| Mathematical density | 20–25 numbered equations | 29 equations, Proposition 1, parameter formulas in closed form | 9.5 |
| Architecture figures | 5+ diagrams in Methodology | 5 Mermaid diagrams + change tables | 9.0 |
| Algorithm block | Common | Algorithm 1 | 9.0 |
| Comparison table | 6+ prior published methods | 3 published networks (MedMamba, HybridSN, SpectralFormer) + 4 probes | 7.5 |
| Ablation with deltas | Always | Table XI, Δ macro-F1 per component | 8.5 |
| Error analysis | Confusion matrix, per-class, learning curves | Confusion matrices and per-patient table in the main text; learning curves only in the supplement | 8.5 |
| Calibration and uncertainty | Central in MedFormer-UR (ECE, MCE, Brier, NLL, routing) | Table XIV with all four, plus temperature transfer and selective prediction | 9.5 |
| Explainability | A profile interest; his XAI papers use attribution maps | Band-gate profile and t-SNE; no attribution method | 8.0 |
| Topic | HSI, medical diagnosis, Mamba, band-aware design | All four | 9.5 |
| Contributions | ~4 numbered items | 2 numbered model contributions + a supporting-components paragraph | 8.0 |
| Limitations and hedging | No limitations section; states findings and moves on | Discussion with Limitations and Future Work; hedging in most results paragraphs | 6.5 |
| Length | ~12–15k words | 16.3k prose words + supplement | 7.0 |
| His group's literature | Builds on his own line | 7 of 57 references. The most relevant one (ICAIB 2025, breast histopathology) is uncited. | 8.0 |

### Good

- **The skeleton is his.** "Literature Review" organized by method family, ending in "F. Gaps Identified", then a standalone "III. Datasets" before "IV. Methodology". That is his order, and his heading words.
- **The Methodology is as formal as his, and more checkable.** 29 equations, closed-form parameter counts in which $C$ visibly does not appear, and an analytic cost model verified against measurement.
- **The results section hits every element he uses:**
  - A comparison table.
  - An ablation table with deltas.
  - Confusion matrices.
  - A calibration table with ECE, MCE, Brier and NLL, the exact set MedFormer-UR emphasizes.
  - Selective prediction, which is the uncertainty-routing idea of MedFormer-UR applied here.
- **The topic sits at the centre of his current line.** BAT-Former is band-aware, the crop-field paper is a bidirectional spectral Mamba, and the SLIC and dermatopathology papers are medical HSI and histology. The band-count-agnostic spectral scan is the idea that connects them, and the paper cites all of these where they are relevant.

### Bad

- **The paper is more hedged and longer than his template.** His papers state results and conclude. This one qualifies nearly every results paragraph ("single run", "within the seed spread", "not established") and adds a full Limitations and Future Work block. The honesty is right for review. In his house style it reads as tentative.
- **The comparison table is thin by his standard.** He compares against six or more published methods. Here there are three published networks and four probes. His own BAT-Former, the closest band-aware design, is discussed but not run.
- **The most relevant paper from his group is uncited.** *Interpretable Deep Transfer Learning Framework for Breast Cancer Histopathology Classification* (ICAIB 2025) is breast histopathology, the same organ and task family. It was earlier judged "unfindable", so it was not cited unread, which is correct. A reader from his group will notice its absence.
- **The explainability is lighter than his XAI work.** A band-gate average and a t-SNE are not attribution maps. His "Explainable …" papers use Grad-CAM-style visual explanations.
- **Only two numbered contributions.** His papers usually list about four. The deliberate two-contribution framing, with "supporting components" as prose, departs from that.

### To improve

1. **Find and read the ICAIB 2025 breast-histopathology paper**, via his Lakehead page, the proceedings or the author directly. Cite it where it fits: breast histology, transfer learning, interpretability. Never cite it unread.
2. **Add BAT-Former as a baseline.** It is his group's band-aware model, and the paper already positions against it. This raises both this rating and the reviewer rating's baseline score.
3. **Expand the comparison table to five or six published methods**, for example a 3-D CNN, Vision Mamba or VMamba, and BAT-Former, under the same split.
4. **Move the learning curves (Fig. S1) into the main text**, and add one attribution-style explanation (e.g. per-band occlusion, or integrated gradients over bands). This matches his XAI emphasis and would also test the band gate's causal reading, which the paper says is missing.
5. **Cut to about 13–14k words.** Remove the repeated passages (audit F34) and collect the hedging into the Limitations subsection rather than repeating it in every results paragraph.

---

## 4. Composite — **7.5 / 10**

Weighted: overall 30 %, senior reviewer/engineer 45 %, structural fit 25 %. The reviewer view has the most weight because it decides acceptance.

0.30 × 7.4 + 0.45 × 7.0 + 0.25 × 8.6 = 2.22 + 3.15 + 2.15 = **7.52**

### The good, in one place

An honest, technically exact paper. It has a clean two-step lineage from two named models, a spectral front end whose band-count independence is proven and measured, and a quantified, transferable finding that weight-shared recursion trades storage for compute. Its engineering and diagnostic discipline (validity gates, the batch-norm diagnosis, fixed-η with an identity control, the cross-set calibration analysis) is above the norm for applied ML. Its structure matches the supervisor's template almost section for section.

### The bad, in one place

The evidence cannot yet carry the accuracy claims:
- The input bands saw held-out labels.
- No patient set is untouched by development.
- Ten patients with one DCIS patient per set decide every comparison.
- Pooled over all of them, a 0.57 M-parameter HybridSN and a six-feature colour probe beat the proposed model.

MedMamba-SS, a headline contribution, doesn't train on the target data, and recursion is never isolated from model size. What survives every caveat is architectural and arithmetic, not diagnostic accuracy.

### What to do, in order of rating gain per hour

| # | Action | Cost | Main rating moved |
| ---: | --- | --- | --- |
| 1 | Fix the audit's text defects (Table XVI, two/three patients, PSNR, Fig. 7 caption, ref [27], Algorithm 1) | hours, no GPU | Overall (writing) |
| 2 | Re-scope Contribution 1 to the spectral pathway; split the Conclusion into design properties vs findings | hours, no GPU | Reviewer (claim–evidence), Overall |
| 3 | Fixed-η on the four remaining checkpoints | ≈ 2 h of GPU evaluation | Reviewer (statistics) |
| 4 | Find and cite the ICAIB 2025 paper; add BAT-Former as a baseline | days; ≈ 1 h of GPU per seed | Saad-fit, Reviewer (baselines) |
| 5 | Retrain the headline pair on `hsi_v9-trainsel` | ≈ 60 h for five seeds | Reviewer (integrity, significance) |
| 6 | Patient 5-fold CV, frozen recipe | 40–70 h | Reviewer (integrity, statistics, significance) |
| 7 | Non-recursive control at matched parameters, ≥ 3 seeds | ≈ 3.7 h per run | Reviewer (ablation), Overall |

Items 1–4 are likely to bring the composite to about **7.8**. Items 5–6 are what can take the reviewer rating past 8. That holds only if the retrained, cross-validated results keep the test-set ranking. If they don't, the paper's contribution becomes the architecture and the cost analysis, and that must be stated.

---

## Sources for the structural-fit rating

- [Saad B. Ahmed — Google Scholar profile, sorted by date](https://scholar.google.com/citations?hl=es&user=6QZM-vIAAAAJ&view_op=list_works&sortby=pubdate) (fetched 2026-09-26)
- [MedFormer-UR: Uncertainty-Routed Transformer for Medical Image Classification (arXiv 2604.08868)](https://arxiv.org/abs/2604.08868): full text read earlier for section structure
- [Quantum Enhanced Multi-Scale CNN with Bi-Directional Mamba for Crop Field Analysis (arXiv 2606.17222)](https://arxiv.org/abs/2606.17222): full text read earlier for section structure
- `paper/draft/MANUSCRIPT_AUDIT_v18.md`: the findings counted in all four ratings
