## VII. Discussion

### A. Summary of Rankings

Table {{T:scorecard}} collects the rankings of Section VI in one place: where the proposed models lead, and where they do not. The rest of this section interprets them for each proposed model and for the modality comparison, and then states the limitations of the evidence.

**TABLE {{T:scorecard}}**
**Summary of Rankings (Seed 42 Unless Stated; Every Baseline Is a Single Run)**

| Criterion | Leader | Runner-up | Section |
| --- | --- | --- | --- |
| Test balanced accuracy, five patients | **MedMamba-SS-TRM** (five seeds 92.78 ± 1.87 %, first at four of five; 94.37 % at seed 42) | MedMamba (91.02 %) | VI-A |
| Test macro-F1, five patients | Level: MedMamba-SS-TRM (five seeds 0.875 ± 0.040) and MedMamba (0.872) | — | VI-A |
| Test accuracy and Cohen's κ, five patients | MedMamba (95.14 %, 0.896) | MedMamba-SS-TRM (five seeds 94.66 %, 0.888) | VI-A |
| Test selective prediction among networks, AURC (lower is better), seed 42 only | Level: MedMamba-SS-TRM (0.0035) and MedMamba (0.0038); MedMamba more accurate on the 90 % most confident patches | — | VI-H |
| Parameters identical at 3 and 32 bands | **MedMamba-SS-TRM** (446,409 at both) | — (every baseline changes with $C$) | VI-C |
| Pooled balanced accuracy, all models, ten patients | HybridSN (87.14 %) | RGB colour probe (86.50 %); MedMamba-SS-TRM five seeds 84.49 % | VI-A |
| Pooled macro-F1, all models, ten patients | HybridSN (0.813) | RGB colour probe (0.807); MedMamba-SS-TRM five seeds 0.795 | VI-A |
| Validation balanced accuracy, five patients | HybridSN (84.94 %) | RGB colour probe (84.90 %); MedMamba-SS-TRM five seeds 76.18 % | VI-A |
| Test ECE | MedMamba (0.013) | 64-feature probe (0.025); MedMamba-SS-TRM 0.056 | VI-H |
| Whole-image PAD-UFES-20 (confounded single runs; differences within run-to-run variation) | MedMamba-SS-TRM on balanced accuracy and macro-F1 (50.2 %, 0.421); MedMamba-SS on accuracy, κ and ROC-AUC (53.5 %, 0.352, 0.798) | — | VI-K |

### B. MedMamba-SS and the Spectral Pathway

The spectral pathway removes the band count from every parameter shape by construction. Measured inside MedMamba-SS-TRM, the same parameters serve 3 and 32 bands, a model retrained at 8 bands keeps most of its accuracy, and the learned band gate is highest at 577–633 nm, near where the classes differ most (Sections VI-C and VI-I). Single-seed ablations are consistent with small gains from the wavelength encoding and from the reconstruction objective, within the seed-to-seed spread. Independence holds for the architecture, not for a trained instance: a model trained on 32 bands does not transfer to fewer without retraining, partly because its encoding changes with the band set. MedMamba-SS itself trains on whole RGB images, well above the majority-class rate, but it stays at chance on hyperspectral patches under the recipe used for the recursive model, a failure that survives changes of learning rate, clipping threshold and auxiliary loss and whose cause we have not identified. That failure is the main open problem for MedMamba-SS, and until it is resolved the additions of MedMamba-SS to the hierarchy (per-stage conditioning, context updater, modified block and head) remain unablated.

### C. MedMamba-SS-TRM and What Transfers from TRM

The recursive substitution reduces parameters 6.2× on hyperspectral input while keeping the spectral pathway and the task interface. The resulting 0.45 M-parameter model has the highest test balanced accuracy of the models compared at four of five seeds. Against its base architecture MedMamba (3.65 M), trained under a different recipe, it is ahead on test and pooled balanced accuracy, level on test macro-F1 and behind on test accuracy and κ; these differences are small relative to the seed-to-seed spread, and MedMamba is a single run. In the reported configuration the recursion also replaces MedMamba's spatial selective scan with a convolutional mixer (Section IV-E), so the comparison with MedMamba measures that change as well; an SS2D mixer inside the core is implemented but was too slow to train at this scale (Section V-C).

TRM's schedule, with two carried states and deep supervision inside one forward pass, trains stably on noisy patch classification; whether these components help, compared with a non-recursive core of the same size, is not tested here. Three expectations that one might carry over from TRM do not hold on this task. Its halting head, trained against "the current answer is correct", does not learn on three-way tissue labels (Appendix A). Its depth benefit is not measurable: 21 core applications match 63, and at 21 there is no gradient-free prelude. And the common reading of "tiny" as "cheap", which TRM does not claim since it reports no arithmetic, does not hold: a weight-shared recursive model trades storage for compute (Section VI-E), at inference as well as in training.

### D. Dependence on the Held-Out Patients

Whether hyperspectral input beats RGB input, and whether MedMamba-SS-TRM beats HybridSN or a logistic regression on per-patch statistics, depends on which five patients are held out. On the test patients the 32-band model has the highest balanced accuracy and beats its RGB twin at every seed; on the validation patients it is among the weaker networks and loses to its RGB twin at every seed. Per patient the picture is simpler: 32 bands give the higher recall for seven of ten patients, tie on one, and fail badly on one (patient 304); and each held-out set contains DCIS from a single patient. With five patients per held-out set, the between-patient variation is larger than most of the differences among the networks that train. The errors themselves are concentrated in patients: healthy captures of patient 136 on test and of patient 304 on validation are predicted as DCIS at every seed, and patient 65's IDC captures are missed almost entirely (Table {{T:perpatient}}). The pooled ten-patient comparison, in which HybridSN and a six-feature colour probe lead, uses the most patients, but half of them also selected the networks' checkpoints, so it does not replace patient-level cross-validation.

### E. Limitations

- **Band selection.** The importance score that chose the 32 input bands used tissue labels from 64 captures drawn before the split, 16 of them from held-out patients (Section III-A). The patient-disjoint split does not cover this step, and the preparation must be repeated with band selection restricted to training captures before the hyperspectral results, the band-gate profile and the modality comparison can be taken at face value.
- **Held-out patients.** All hyperspectral results rest on ten non-training patients. DCIS is represented by five training patients and by one patient per held-out set, and the validation patients also select each network's checkpoint. The test patients play no part in checkpoint selection, but test scores of development runs were recorded and compared while the recipe was developed (Section V-C), so their independence from model development cannot be demonstrated. Patient-level cross-validation, or a new held-out set, is the next experiment.
- **Patient-concentrated errors.** The dominant errors come from individual patients at every seed (Table {{T:perpatient}}). Whether DCIS was partly learned as patient or slide appearance, or whether capture-level labels are noisy on these captures, has not been tested; the region annotations of the collection could be used to check the latter. Patient 65's ten IDC captures are smaller than all others and are almost never classified as IDC, for a reason we have not identified.
- **Labels and unit of analysis.** Every patch inherits its capture's tissue label without a pixel-level mask, so labels are noisy at patch level, and evaluation is per patch, not per slide or patient, the unit at which a diagnosis is made. DCIS is also the diagnosis on which pathologists agree least often [@elmore]. The per-capture intensity gain uses a reference median over all captures, which involves no labels but is not restricted to training data. The data come from one institution and one acquisition system, and the results are computational; nothing here constitutes clinical validation.
- **Statistics and controls.** Every baseline and ablation is a single run, and the ablation effects are smaller than the seed-to-seed spread. Seed 42, used for every single-seed analysis, is the best test seed and the worst validation seed of MedMamba-SS-TRM. MedMamba was trained with its own recipe and selected on the full validation split, and HybridSN and SpectralFormer were not tuned. No non-recursive core of the same size was trained, so the contribution of recursion itself, as opposed to the rest of the design, is not isolated.
- **Modality comparison.** The RGB input is the collection's 8-bit synthetic rendering, and the two arms also differ in the reconstruction target and in the spectral position encoding (index for the 3-band arm).
- **Hierarchy.** MedMamba-SS does not train on hyperspectral patches under our recipe, for an unidentified reason that may include an evaluation-mode defect, so on those data the recursive substitution is supported on parameters, arithmetic and interface, not on a head-to-head comparison.
- **Band transfer.** The zero-shot failure is partly built into the encoding, and no channel-adaptive baseline was trained.
- **PAD-UFES-20.** The whole-image comparison is confounded by training set, recipe and loss weighting, its metrics vary by several points between identical runs, the MedMamba-SS-TRM configuration was chosen among 33 runs whose test scores were recorded, and our run of the MedMamba baseline is weaker than the published one.

### F. Future Work

The most direct next steps close the gaps listed above: repeating band selection on training captures only, patient-level cross-validation, replicating the ablations and baselines across seeds, testing whether the collapse of MedMamba-SS on hyperspectral patches comes from a defect in evaluation mode (Section VI-D), and a zero-shot test with the wavelength encoding fixed at its 32-band values (Section VI-C). Beyond these, training on random band subsets [@channelvit], with the wavelength encoding fixed to a reference range, may make a trained model transferable between band counts, which would turn the zero-shot failure into a strength. A second hyperspectral dataset, such as the HSIDermoscopy skin-lesion data used in [@hsi_melanoma] or the HyperLeaf2024 data used in [@batformer] and [@patchgraph], would test band-count agnosticism beyond one sensor, and a direct comparison with BAT-Former on it would place the spectral pathway against the closest band-aware design. A non-recursive core of the same size, trained under the same recipe, would isolate the contribution of recursion. Selecting bands instead of decimating them, recursion depths below 21 and an SS2D mixer with a fused scan kernel inside the core complete the list.

---

## VIII. Conclusion

We extended MedMamba [@medmamba] in two steps, each a single substitution. MedMamba-SS replaces the convolutional patch embedding with a spectral pathway whose parameters do not depend on the number of bands and conditions every stage of the SS-Conv-SSM hierarchy on it; MedMamba-SS-TRM replaces that hierarchy with one weight-shared core applied recursively in the manner of TRM [@trm], keeping the spectral pathway and the task interface.

The structural results hold whichever patients are held out. The classification path has exactly the same parameter count at 3 and at 32 bands (446,409). And the recursive core reduces parameters 6.2× while multiplying arithmetic 22.3×, so parameter count in a weight-shared recursive model measures storage, not compute.

Further findings come from single runs on the five test patients. A model retrained at eight bands keeps most of its accuracy, whereas a trained model does not transfer to fewer bands, partly because its wavelength encoding changes with the band set. The wavelength encoding and the reconstruction objective are associated with small gains within the seed-to-seed spread, and recursion depth beyond 21 applications buys nothing measurable.

The comparisons with other models and with RGB input depend on the patients. MedMamba-SS-TRM has the highest balanced accuracy on the five test patients (92.8 ± 1.9 % over five seeds, first at four of five against single-run baselines) and is among the weaker networks on the five validation patients; pooled over all ten, HybridSN and a six-feature colour probe lead. The advantage of 32-band over 3-band input follows the same split (+4.1 points on test, −5.8 on validation, −1.0 pooled), and a linear probe shows the same pattern, so the reversal is not specific to the network. The largest errors are concentrated in three patients. All hyperspectral results also carry the band-selection caveat of Section III-A.

Together, these results support the two architectural claims, a front end whose parameters do not depend on the band count and a recursive backbone that trades storage for compute, while the classification benefit of hyperspectral over RGB input for breast histology remains to be settled on more patients, with band selection restricted to training data.

---

## Appendix A: Implementation Safeguards

The failure modes in Table {{T:safeguards}} degrade the accuracy of models built on the spectral pathway or the recursive core without raising an error. The reported configuration contains a remedy for each, and the representation-sensitivity and reconstruction-gradient gates of Section V-C abort a run in which the first or the last of them recurs.

**TABLE {{T:safeguards}}**
**Silent Failure Modes and Their Remedies**

| Failure mode | Mechanism and measurement | Remedy |
| --- | --- | --- |
| Representation collapse | A linear value embedding initialized at s.d. 0.02 produced tokens of magnitude ≈ 0.006 beside a positional encoding of ≈ 0.55, so the band values were 86–93× smaller than a term that is constant for a given sensor; stem sensitivity measured 0.0018 (HSI) and 0.0029 (RGB) in diagnostic runs. The same defect arises in the recursive model's 2-D encoding and the classifier's initialization. | Concat-MLP tokenizer (7), value initialization s.d. 0.5, positional gains 0.1, fan-in classifier initialization; sensitivity gate (stem sensitivity ≈ 0.5 against a floor of 0.05) |
| Normalization below its $\epsilon$ | Initialization at s.d. 0.02, about 9× below fan-in scale, shrank activations through the pathway's successive projections. The context reached the stem at magnitude ≈ $5 \times 10^{-5}$, a mean square of ≈ $2.5 \times 10^{-9}$, orders of magnitude below the constants that layer normalization ($10^{-5}$) and a standard RMS normalization ($10^{-6}$) add to the mean square, so both become near-constant rescalings; a standard RMS normalization gives an output RMS of about 0.01 instead of 1 at input scale $10^{-5}$. | Scale-invariant RMS normalization $\rho$ in (12) and (18), output RMS 1.000 at any scale |
| Halting head without effect | With ACT enabled, the halting head's cross-entropy was about 20 % of the objective and stayed at ≈ 0.66 without decreasing for 33 epochs in a diagnostic run. | Halting disabled; when enabled, segments stop once every sample in the batch exceeds the halting threshold |
| Detached reconstruction | A decoder reading a detached feature map trains itself but cannot shape the representation. | Decoder reads the live answer state returned by the forward pass; reconstruction-gradient gate |

A linear value branch has a second, independent defect: at the tokenizer output every band value lies along one vector (Section IV-C). A layer normalization on the value branch does not fix it, because $\mathrm{LN}(v w) = \mathrm{LN}(w)$ for $v > 0$; the concat-MLP tokenizer does, and it keeps Proposition 1 intact because its weight shapes depend only on $d_t$.

## Appendix B: Reproducibility

Both proposed models are defined in one implementation and selected by one configuration switch. Every run records its configuration, per-epoch history, gate reports, per-patient and per-capture metrics, arithmetic counts and a checkpoint-reproducibility record, so a run can be repeated from its recorded configuration; runs are not bitwise deterministic (Section V-E). Data preparation records its command line, the selected band indices, the per-capture gains and the patient split. The held-out analyses of Section VI (full-validation predictions, patient bootstrap, probes on both held-out sets, band-gate and latent-space exports) are computed from saved checkpoints and predictions without retraining. Both datasets are public under the CC BY 4.0 license [@hsibc_data], [@pad_data]; the MedMamba baseline uses the reference implementation [@medmamba_code]. Code and trained checkpoints will be released on acceptance.

---

## Acknowledgment

The hyperspectral histology data used in this work are from the HistologyHSI-BC-Recurrence collection [@hsibc_data], described in [@hsibc_paper] and accessed through The Cancer Imaging Archive (TCIA); they are used under the Creative Commons Attribution 4.0 International (CC BY 4.0) license and the TCIA Data Usage Policy. The skin-lesion photographs are from the PAD-UFES-20 dataset [@pad_data], described in [@pad] and accessed through Mendeley Data, and are likewise used under CC BY 4.0. The band selection, patch extraction and patient splits derived from them are described in Section III. The author thanks the creators of both datasets for making them publicly available.

The author also thanks the authors of MedMamba and of the Tiny Recursive Model for releasing reference implementations [@medmamba_code], [@trm_code].

A generative AI assistant (Claude, Anthropic) was used to help draft and revise the text of this manuscript and to help write and debug code for the experiments, analyses and figures. The author directed this work, checked the text, numbers and references against the experimental records, and takes full responsibility for the content.

---

## References
