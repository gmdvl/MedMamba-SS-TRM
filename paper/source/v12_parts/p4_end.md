## VII. Conclusion

We extended MedMamba [@medmamba] in two steps. MedMamba-SS replaces the convolutional patch embedding with a spectral pathway whose parameters do not depend on the number of bands and conditions every stage of the SS-Conv-SSM hierarchy on it; MedMamba-SS-TRM replaces that hierarchy with one weight-shared core applied recursively in the manner of TRM [@trm], keeping the spectral pathway and the task interface.

Four results do not depend on which patients are held out. The parameter count is exactly the same at 3 and at 32 bands, and a model retrained at eight bands keeps 97 % of its macro-F1, whereas a trained model cannot be moved to fewer bands. The spectral pathway learns to weight the 535–633 nm stain-absorption region where the classes differ, and its wavelength encoding and the reconstruction objective each improve classification. The recursive core reduces parameters 6.2×, and with 8.2× fewer parameters than MedMamba it outperforms it on the test patients and pooled over all ten and matches it on the validation patients. And it multiplies arithmetic 22.3×, exactly as the cost model predicts, while recursion depth beyond 21 applications buys nothing measurable, so parameter count in a recursive model measures storage rather than compute.

Two results depend on the patients. MedMamba-SS-TRM is the most accurate model on the five test patients and among the least accurate networks on the five validation patients; pooled over all ten, HybridSN and a six-feature colour probe lead. The advantage of 32-band over 3-band input follows the same split (+4.1 points on test, −5.8 on validation, −1.0 pooled): 32 bands give the higher recall for most held-out patients but fail badly on one, and a linear probe shows the same pattern, so it is a property of the patients rather than of the network.

**Limitations and future work.** All hyperspectral results rest on ten non-training patients; the pooled analysis uses both held-out sets but does not replace patient-level cross-validation, which is the next experiment, and each held-out set contains DCIS from a single patient. The baselines are single-seed, and MedMamba's recipe differs from the others'. The RGB input is the collection's own synthetic rendering. The hierarchical MedMamba-SS does not train under the recursive recipe on hyperspectral patches, so on those data the recursive substitution is supported on parameters, arithmetic and interface rather than on a head-to-head comparison; a recipe search for the hierarchy is needed to close that comparison. The whole-image PAD-UFES-20 comparison is confounded by training-set size and loss weighting. Selecting bands rather than decimating them, a second hyperspectral dataset, and recursion depths below 21 are the natural extensions.

---

## Appendix A: Implementation Safeguards

Four failure modes degrade the accuracy of models built on the spectral pathway or the recursive core without raising an error. The reported configuration contains a remedy for each, and three are guarded by a gate that aborts the run.

| Failure mode | Mechanism and measurement | Remedy |
| --- | --- | --- |
| Representation collapse | A linear value embedding initialized at s.d. 0.02 produced tokens of magnitude ≈ 0.006 beside a positional encoding of ≈ 0.55, so the input was 86–93× smaller than a per-dataset constant; stem input dependence measured 0.18 % (HSI) and 0.29 % (RGB) in diagnostic runs. The same defect arises in the recursive model's 2-D encoding and the classifier's initialization. | Concat-MLP tokenizer (7), value initialization s.d. 0.5, positional gains 0.1, fan-in classifier initialization; sensitivity gate (stem sensitivity ≈ 0.5 against a floor of 0.05) |
| Normalization below its $\epsilon$ | Initialization at s.d. 0.02, about 9× below fan-in scale, shrank activations about 100× per projection; the context reached the stem at ≈ $5 \times 10^{-5}$, below layer normalization's $\epsilon = 10^{-5}$, so the normalization became a constant rescaling (output RMS 0.0095 at input scale $10^{-5}$). | Scale-invariant RMS normalization $\rho$ in (12) and (18), output RMS 1.000 at any scale |
| Halting head without effect | With ACT enabled, the halting head's cross-entropy was about 20 % of the objective and stayed at ≈ 0.66 against $\ln 2 = 0.693$ for 33 epochs in a diagnostic run. | Halting disabled; when enabled, segments stop once every sample in the batch exceeds the halting threshold |
| Detached reconstruction | A decoder reading a detached feature map trains itself but cannot shape the representation. | Decoder reads the live answer state returned by the forward pass; reconstruction-gradient gate |

A linear value branch has a second, independent defect: every band token is a multiple of one vector, so averaging over bands reduces a spectrum to a scalar. A layer normalization on the value branch does not fix it, because $\mathrm{LN}(v w) = \mathrm{LN}(w)$ for $v > 0$; the concat-MLP tokenizer does, and it keeps Proposition 1 intact because its weight shapes depend only on $d_t$.

## Appendix B: Reproducibility

Both proposed models are defined in one implementation and selected by one configuration switch. Every run records its configuration, per-epoch history, gate reports, per-patient and per-capture metrics, arithmetic counts and a checkpoint-reproducibility record, so a run can be reproduced from its recorded configuration alone. The held-out analyses of Section VI (full-validation predictions, patient bootstrap, probes on both held-out sets, band-gate and latent-space exports) are computed from saved checkpoints and predictions without retraining. Both datasets are public [@hsibc_data], [@pad]; the MedMamba baseline uses the reference implementation [@medmamba_code]. Code and trained checkpoints will be released on acceptance.

---

## Acknowledgment

The author thanks the authors of MedMamba and of the Tiny Recursive Model for releasing reference implementations [@medmamba_code], [@trm_code].

---

## References
