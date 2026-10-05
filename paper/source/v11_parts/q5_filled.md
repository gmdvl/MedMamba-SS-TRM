---

## VII. Conclusion

We extended MedMamba in two steps. MedMamba-SS replaces the convolutional patch embedding with a spectral pathway whose parameters do not depend on the number of bands and conditions every stage of the SS-Conv-SSM hierarchy on it; MedMamba-SS-TRM replaces that hierarchy with one weight-shared core applied recursively in the manner of TRM.

Four results do not depend on which patients are held out. The parameter count is exactly the same at 3 and at 32 bands, and a model retrained at eight bands keeps 97 % of its macro-F1, whereas a trained model cannot be moved to fewer bands. The spectral pathway learns to weight the 535–633 nm stain-absorption region where the classes differ, and its wavelength encoding and reconstruction objective each improve classification. The recursive core reduces parameters 6.2×, and with 8.2× fewer parameters than MedMamba it outperforms it on the test patients and pooled over all ten and matches it on the validation patients. And it multiplies arithmetic 22.3×, exactly as the cost model predicts, while recursion depth beyond 21 applications buys nothing measurable, so parameter count in a recursive model measures storage rather than compute.

Two results depend on the patients. MedMamba-SS-TRM is the most accurate model on the five test patients and among the least accurate networks on the five validation patients; pooled over all ten, HybridSN and a six-feature colour probe lead. The advantage of 32-band over 3-band input follows the same split (+4.1 points on test, −5.8 on validation, −1.0 pooled): 32 bands give the higher recall for most held-out patients but fail badly on one, and a linear probe shows the same pattern, so it is a property of the patients rather than of the network.

**Limitations.** All hyperspectral results rest on ten non-training patients, and each held-out set contains DCIS from a single patient. The baselines are single-seed, and MedMamba's recipe differs from the others'. The RGB input is the collection's own synthetic rendering. The hierarchical MedMamba-SS does not train under the recursive recipe on hyperspectral data, so the substitution is supported on parameters, arithmetic and interface rather than on a head-to-head comparison. The whole-image PAD-UFES-20 comparison is confounded by training-set size and loss weighting.

**Future work.** In priority order, the experiments that would close the gaps above:

1. **Patient-level cross-validation.** Five or more folds over all 45 patients, stratified so that every held-out fold contains DCIS from several patients, for every model in Table {{T:comparison}}. This is the only experiment that can rank the models and settle whether 32-band input helps.
2. **The failure on patient 304.** Its captures, staining and spectra should be examined, and spectral normalization or stain-invariant augmentation tested, since this one patient decides the sign of the validation result.
3. **Multi-seed baselines.** At least three seeds each for HybridSN, SpectralFormer and MedMamba, which are single-seed here.
4. **MedMamba under the shared recipe.** Twenty epochs of 10.2 % subsets with the focal loss, optimizer, schedule and augmentation of Table {{T:hyper}}, in place of its five full epochs of class-weighted cross-entropy at learning rate $10^{-4}$.
5. **Sensitivity to the selection rule.** Fixed-epoch and last-epoch comparisons beside validation-based selection, since HybridSN and SpectralFormer select their first and third epochs and MedMamba-SS-TRM its seventh, and a selection split with more than five patients.
6. **A head-to-head hierarchical comparison.** A recipe under which MedMamba-SS trains on hyperspectral data (its gradient norm of 302 to 1,002 against a clipping threshold of 1.0 points to per-architecture clipping or learning rate), followed by evaluation on both held-out sets.
7. **The ablations not yet run.** The spectral pathway against a convolutional stem inside the same backbone; FiLM against the four implemented fusion alternatives; one carried state against two; and every ablation of Table {{T:ablation}} evaluated on the validation patients and pooled, not only on the test patients.
8. **Recursion depth below 21 core applications** (7, 3 and 1), to find where quality begins to fall, and adaptive depth with a halting target defined for soft labels.
9. **Band selection and sensor transfer.** Selecting the best 8 and 16 of the 740 bands instead of decimating the selected 32; training with band dropout so that a trained model transfers to fewer bands; and a second hyperspectral dataset from a different instrument.
10. **A controlled RGB arm.** RGB rendered from the cubes with CIE colour-matching functions, replacing the collection's synthetic rendering.
11. **Calibration under patient shift.** Per-patient or shift-robust calibration, since for the 32-band model a temperature fitted on either held-out set worsens calibration on the other.
12. **PAD-UFES-20.** Retraining MedMamba-SS-TRM on all 1,626 training images with MedMamba-SS's loss weighting, to remove the confounds of Table {{T:pad}}, and an image-level (multiple-instance) loss for patch models [@mil].
13. **Efficiency.** A compiled selective-scan kernel, which would make the SS2D mixer practical, and wall-clock timing on an uncontended GPU.

---

## Appendix A: Implementation Safeguards

Four mechanisms found while building the models degrade accuracy without raising an error. The reported configuration contains a remedy for each, and three are guarded by a gate that aborts the run.

| Failure mode | Mechanism and measurement | Remedy |
| --- | --- | --- |
| Representation collapse | A linear value embedding initialized at s.d. 0.02 produced tokens of magnitude ≈ 0.006 beside a positional encoding of ≈ 0.55, so the input was 86–93× smaller than a per-dataset constant; stem input dependence measured 0.18 % (HSI) and 0.29 % (RGB) *(auxiliary)*. The same defect recurred in the recursive model's 2-D encoding and the classifier's initialization. | Concat-MLP tokenizer (7), value initialization s.d. 0.5, positional gains 0.1, fan-in classifier initialization; sensitivity gate (stem sensitivity ≈ 0.5 against a floor of 0.05) |
| Normalization below its $\epsilon$ | Initialization at s.d. 0.02, about 9× below fan-in scale, shrank activations about 100× per projection; the context reached the stem at ≈ $5 \times 10^{-5}$, below layer normalization's $\epsilon = 10^{-5}$, so the normalization became a constant rescaling (output RMS 0.0095 at input scale $10^{-5}$). | Scale-invariant RMS normalization $\rho$ in (12) and (18), output RMS 1.000 at any scale |
| Halting head without effect | With ACT enabled, the halting head's cross-entropy was about 20 % of the objective and stayed at ≈ 0.66 against $\ln 2 = 0.693$ for 33 epochs *(auxiliary)*. | Halting disabled; when enabled, a threshold genuinely stops segments once every sample has halted |
| Detached reconstruction | A decoder reading a detached feature map trained itself but could not shape the representation. | Decoder reads the live answer state returned by the forward pass; reconstruction-gradient gate |

A linear value branch has a second, independent defect: every band token is a multiple of one vector, so averaging over bands reduces a spectrum to a scalar. A layer normalization on the value branch does not fix it, because $\mathrm{LN}(v w) = \mathrm{LN}(w)$ for $v > 0$; the concat-MLP tokenizer does, and it keeps Proposition 1 intact because its weight shapes depend only on $d_t$.

## Appendix B: Reproducibility

Both models are defined in one implementation and selected by one configuration switch. Every run records its configuration, per-epoch history, gate reports, per-patient and per-capture metrics, arithmetic counts and a checkpoint-reproducibility record, and every configuration option introduced during development defaults to the behaviour that preceded it, so a recorded run can be reproduced from its configuration. The held-out analyses of Section VI (full-validation predictions, patient bootstrap, probes on both held-out sets, band-gate and latent-space exports) are computed from saved checkpoints and predictions without retraining. Both datasets are public [@hsibc_data], [@pad]; the MedMamba baseline uses the reference implementation [@medmamba_code]. Code and trained checkpoints will be released on acceptance.

---

## Acknowledgment

The author thanks the authors of MedMamba and of the Tiny Recursive Model for releasing reference implementations [@medmamba_code], [@trm_code].

---

## References

