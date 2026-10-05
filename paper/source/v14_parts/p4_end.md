## VII. Conclusion

We extended MedMamba [@medmamba] in two steps. MedMamba-SS replaces the convolutional patch embedding with a spectral pathway whose parameters do not depend on the number of bands and conditions every stage of the SS-Conv-SSM hierarchy on it; MedMamba-SS-TRM replaces that hierarchy with one weight-shared core applied recursively in the manner of TRM [@trm], keeping the spectral pathway and the task interface.

Two results are structural and hold whichever patients are held out. The classification path has exactly the same parameter count at 3 and at 32 bands (446,409). And the recursive core reduces parameters 6.2× while multiplying arithmetic 22.3×, with the cost of each core application given by the cost model, so parameter count in a recursive model measures storage, not compute.

Further findings were measured on the five test patients, most at a single seed. A model retrained at eight bands keeps 97 % of its macro-F1, whereas a trained model does not transfer to fewer bands, partly because its wavelength encoding changes with the band set. The band gate peaks in the 535–633 nm range where the class-mean spectra differ most. The wavelength encoding and the reconstruction objective are associated with small gains that lie within the seed-to-seed spread, and recursion depth beyond 21 applications buys nothing measurable.

Two results depend on the patients. MedMamba-SS-TRM has the highest balanced accuracy on the five test patients (92.8 ± 1.9 % over five seeds, first at four of five against single-run baselines) and is among the weaker networks on the five validation patients; pooled over all ten, HybridSN and a six-feature colour probe lead, and its advantage over MedMamba, trained under a different recipe, is small relative to the spread over seeds. The advantage of 32-band over 3-band input follows the same split (+4.1 points on test, −5.8 on validation, −1.0 pooled): 32 bands give the higher recall for most held-out patients but fail badly on one, which accounts for about two-thirds of the validation reversal, and a linear probe shows the same pattern, so the reversal is not specific to the network.

**Limitations.**

- **Band selection.** The importance score that chose the 32 input bands used tissue labels from 64 captures drawn before the split, 16 of them from held-out patients (Section III-A). The patient-disjoint split does not cover this step, and the preparation must be repeated with band selection restricted to training captures before the hyperspectral results, the band-gate profile and the modality comparison can be taken at face value.
- **Held-out patients.** All hyperspectral results rest on ten non-training patients. DCIS is represented by five training patients and by one patient per held-out set, and the validation patients also select each network's checkpoint, so only the five test patients are independent of model selection. Patient-level cross-validation is the next experiment.
- **Labels and unit of analysis.** Every patch inherits its capture's tissue label without a pixel-level mask, so labels are noisy at patch level, and evaluation is per patch, not per slide or patient, the unit at which a diagnosis is made. DCIS is also the diagnosis on which pathologists agree least often [@elmore]. The data come from one institution and one acquisition system, and the results are computational; nothing here constitutes clinical validation.
- **Statistics.** Every baseline and ablation is a single run, and the ablation effects are smaller than the seed-to-seed spread. MedMamba was trained with its own recipe, and HybridSN and SpectralFormer were not tuned.
- **Modality comparison.** The RGB input is the collection's 8-bit synthetic rendering, and the two arms also differ in the reconstruction target.
- **Hierarchy.** MedMamba-SS does not train on hyperspectral patches under our recipe, for an unidentified reason that may include an evaluation-mode defect, so on those data the recursive substitution is supported on parameters, arithmetic and interface, not on a head-to-head comparison.
- **Band transfer.** The zero-shot failure is partly built into the encoding, and no channel-adaptive baseline was trained.
- **PAD-UFES-20.** The whole-image comparison is confounded by training set, recipe and loss weighting, its metrics vary by several points between identical runs, and our run of the MedMamba baseline is weaker than the published one.

**Future work.** Training on random band subsets [@channelvit], with the wavelength encoding fixed to a reference range, may make a trained model transferable between band counts, which would turn the zero-shot failure of Section VI-D into a strength. A second hyperspectral dataset, such as the HSIDermoscopy skin-lesion data used in [@hsi_melanoma] or the HyperLeaf2024 data used in [@batformer] and [@patchgraph], would test band-count agnosticism beyond one sensor, and a direct comparison with BAT-Former on it would place the spectral pathway against the closest band-aware design. Selecting bands instead of decimating them, recursion depths below 21 and a non-recursive core of the same size complete the list.

---

## Appendix A: Implementation Safeguards

Four failure modes degrade the accuracy of models built on the spectral pathway or the recursive core without raising an error. The reported configuration contains a remedy for each, and the representation-sensitivity and reconstruction-gradient gates of Section V-C abort a run in which the first or the last of them recurs.

| Failure mode | Mechanism and measurement | Remedy |
| --- | --- | --- |
| Representation collapse | A linear value embedding initialized at s.d. 0.02 produced tokens of magnitude ≈ 0.006 beside a positional encoding of ≈ 0.55, so the band values were 86–93× smaller than a term that is constant for a given sensor; stem input dependence measured 0.18 % (HSI) and 0.29 % (RGB) in diagnostic runs. The same defect arises in the recursive model's 2-D encoding and the classifier's initialization. | Concat-MLP tokenizer (7), value initialization s.d. 0.5, positional gains 0.1, fan-in classifier initialization; sensitivity gate (stem sensitivity ≈ 0.5 against a floor of 0.05) |
| Normalization below its $\epsilon$ | Initialization at s.d. 0.02, about 9× below fan-in scale, shrank activations through the pathway's successive projections. The context reached the stem at magnitude ≈ $5 \times 10^{-5}$, a mean square of ≈ $2.5 \times 10^{-9}$, orders of magnitude below the constants that layer normalization ($10^{-5}$) and the earlier RMS normalization ($10^{-6}$) add to the mean square, so both became near-constant rescalings; the earlier RMS normalization gave an output RMS of 0.0095 instead of 1 at input scale $10^{-5}$. | Scale-invariant RMS normalization $\rho$ in (12) and (18), output RMS 1.000 at any scale |
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
