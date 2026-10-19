## VII. Discussion

### A. What We Learned

**A spectral front end takes the band count out of the architecture.** Building the front end only from operations that share their weights across bands was enough to make the parameter count independent of the band count (Section VI-B), and a model retrained on a quarter of the bands lost little. Transfer without retraining depends on one detail of the wavelength code: tying its scale to the band count breaks transfer, and fixing the scale restores it at half of the bands. A code whose scale does not depend on $C$, perhaps combined with training on random band subsets [@channelvit], is the natural next design to test.

**TRM's recursion carries over, but not all of TRM's promises.** A 0.45 M-parameter recursive core trains stably on noisy, imbalanced patch classification, where the hierarchical MedMamba-SS did not train under the same recipe (Section VI-D). Three expectations one might take from TRM do not hold on our task: the halting signal does not learn, depth beyond 21 core applications buys nothing measurable, and reusing weights saves storage but not computation (Section VI-C). At 22.3× the arithmetic of the hierarchy, the parameter count of a recursive model says little about its cost. Whether recursion itself helps, compared with a non-recursive core of the same size, we have not tested.

**Patients matter more than seeds.** The largest effects in this study are not architectural. The gap between the two held-out patient sets is larger than any difference between seeds or between variants, and the errors that create it come from three patients (Section VI-E). With DCIS present in only one patient per held-out set, ten held-out patients are too few to pin down how well the model generalizes.

### B. Limitations

- **Band selection.** Labels of 16 captures from held-out patients entered the selection of the 32 input bands (Section III). A training-only selection exists, but no model has been retrained on it.
- **Held-out patients.** All results rest on ten non-training patients with one DCIS patient per set. The validation patients also choose each checkpoint, and test scores of development runs were recorded while the recipe was developed.
- **Labels and scope.** Labels are per capture, without pixel masks, and evaluation is per patch rather than per patient, the unit at which a diagnosis is made. DCIS is also the diagnosis on which pathologists agree least often [@elmore]. The data come from one institution and one camera, and nothing here amounts to clinical validation.
- **Statistics.** The component studies are single runs whose effects lie within the spread between seeds, and no equal-size non-recursive control was trained.
- **Hierarchy.** MedMamba-SS does not train on these patches under our recipe, so its additions to MedMamba's stages remain unevaluated on hyperspectral data.
- **Scope of claims.** This paper evaluates our models only. It does not compare them with other hyperspectral or channel-adaptive classifiers [@channelvit], [@dofa], [@batformer], [@carl], and it makes no claim of advantage over them.

### C. Future Work

The limitations point to four experiments whose inputs are already prepared: retraining on the training-only band selection; patient-level five-fold cross-validation with band selection repeated inside each fold, which tests every one of the 45 patients, including all seven with DCIS; a second hyperspectral histology dataset from a different camera, HMI-LUSC (lung squamous cell carcinoma, 10 patients, 61 bands from 450 to 750 nm) [@hmi_lusc]; and multi-seed component studies that include an equal-size non-recursive control. Beyond these, a band-count-independent wavelength code and a fused SS2D kernel, which would give the recursive core a spatial state-space mixer again, are the next design steps.

---

## VIII. Conclusion

We set out to build a hyperspectral tissue classifier that does not depend on how many bands a camera records and that stays small enough for a small cohort. Following MedMamba [@medmamba] and the Tiny Recursive Model [@trm], we built it in two steps: a spectral–spatial front end, which gives MedMamba-SS its name, and a recursive core, which gives MedMamba-SS-TRM its name. Two properties hold by design and were confirmed by measurement: the model has the same 446,409 parameters at every band count, and its recursion trades 6.2× fewer parameters for 22.3× more arithmetic. Trained on breast histology from 45 patients, the model reaches 92.8 % balanced accuracy on the test patients, keeps most of its accuracy when retrained on a quarter of the bands, and transfers to half of its bands without retraining once its wavelength code is held fixed. Its weaknesses are as instructive as its strengths: accuracy on the validation patients is much lower, the errors concentrate in three patients, and depth beyond 21 core applications adds nothing. Settling how well such a model generalizes now needs more patients, band selection restricted to training data, and a second camera.

---

## Acknowledgment

The hyperspectral histology data are from the HistologyHSI-BC-Recurrence collection [@hsibc_data], described in [@hsibc_paper] and accessed through The Cancer Imaging Archive under the CC BY 4.0 license and the TCIA Data Usage Policy. The author thanks the creators of the collection for making it public, and the authors of MedMamba and of the Tiny Recursive Model for releasing their code [@medmamba_code], [@trm_code]. Code and trained checkpoints will be released on acceptance.

A generative AI assistant (Claude, Anthropic) was used to help draft and revise the text of this manuscript and to help write and debug code for the experiments, analyses and figures. The author directed this work, checked the text, numbers and references against the experimental records, and takes full responsibility for the content.

---

## References

