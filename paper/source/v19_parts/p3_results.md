## VI. Results

We now follow the trained model through six questions, from whether it learns at all to what it attends to. Unless stated otherwise, results are on the five test patients; we also report the five validation patients, scored on their full split, and all ten held-out patients pooled.

### A. Does the Model Learn?

Table {{T:seeds}} lists every seed. On the five test patients MedMamba-SS-TRM reaches 92.8 ± 1.9 % balanced accuracy, on the five validation patients 76.2 ± 4.6 %, and pooled over all ten 84.5 ± 1.4 %. Two patterns stand out. First, the gap between the two patient sets averages 16.6 points (6.6 to 22.3, depending on the seed), several times the standard deviation between seeds on either set. Second, the seeds trade one set against the other: the seed that is best on the test patients (42) is the worst on the validation patients, and across the five seeds the two scores are strongly anti-correlated (Pearson $r = -0.95$, descriptive only with five points). Section VI-E traces the gap to a few patients. Validation scores are also slightly optimistic, because the validation patients choose each run's checkpoint.

<!-- tex: wide; cols=lrrrrrr -->
**TABLE {{T:seeds}}**
**MedMamba-SS-TRM at Five Seeds: Test (5 Patients), Validation (5 Patients) and Pooled (10 Patients)**

| Seed | Test BA (%) | Test macro-F1 | Val. BA (%) | Val. macro-F1 | Pooled BA (%) | Pooled macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 89.61 | 0.808 | 83.00 | 0.749 | 86.32 | 0.777 |
| 7 | 93.41 | 0.896 | 74.37 | 0.722 | 83.94 | 0.808 |
| 13 | 92.76 | 0.869 | 78.66 | 0.735 | 85.74 | 0.799 |
| 23 | 93.78 | 0.898 | 72.77 | 0.700 | 83.26 | 0.796 |
| 42 | 94.37 | 0.904 | 72.11 | 0.694 | 83.21 | 0.795 |
| **Mean ± s.d.** | **92.78 ± 1.87** | **0.875 ± 0.040** | **76.18 ± 4.59** | **0.720 ± 0.023** | **84.49 ± 1.45** | **0.795 ± 0.011** |

*BA: balanced accuracy (10). Test: 348,894 patches; validation: 334,516; pooled: both. Chance level is 33.33 %.*

Fig. {{F:learning}} shows how the seed-42 run trains. Training loss keeps falling, but validation loss is lowest at epoch 9 and rises afterwards, and validation macro-F1 peaks at epoch 7: the model keeps fitting the 35 training patients after it stops improving on the 5 validation patients, and checkpoint selection keeps the epoch before that divergence. So the model does learn, but how well it does depends on which patients it is tested on.

<!-- tex: width=1.0 -->
![figure](../v19/figures/fig_learning_hsi.png)

*Fig. {{F:learning}}. Training of the seed-42 model. (a) Classification loss on the training and validation subsets (log scale). (b) Validation macro-F1; the dot marks the selected checkpoint.*

### B. Does It Really Ignore the Number of Bands?

Property 1 predicts that the parameter count does not change with the band count, and it does not: trained at 32, 16 and 8 bands, the model has 446,409 parameters each time. What changes is the size of the intermediate activations; peak inference memory, with the decoder attached, is 161, 100 and 69 MB.

The harder question is whether a trained model still works on a different set of bands. Fig. {{F:bands}} answers it in two ways on the test patients. *Retraining* at a lower band count works: with 8 of the 32 bands, chosen evenly by index, the model keeps 96 % of its 12-epoch balanced accuracy (0.901 against 0.935). Applying a 32-band checkpoint to fewer bands *without* retraining (zero-shot) fails with the wavelength code as defined in (3): at every reduced band count balanced accuracy falls to 0.33–0.41, near chance. The cause is the scale $\eta = C$. The subsets keep both end bands, so the rescaling in (3) is unchanged, but $\eta$ changes with $C$, so every remaining band receives a code the model never saw in training. Holding $\eta$ at its training value of 32, and changing nothing else, restores most of the accuracy at 16 bands (0.909 instead of 0.385), though not at 8 (0.560). These are post-hoc evaluations of a single checkpoint. The architecture is therefore band-count agnostic by construction and under retraining, and a trained model transfers to half of its bands once the scale of its code is fixed, but not to a quarter of them.

<!-- tex: width=1.0 -->
![figure](../v19/figures/fig_bands_hsi.png)

*Fig. {{F:bands}}. Test performance against the number of bands $C$ (subsets chosen evenly by band index, both end bands kept). Zero-shot, $\eta = C$: the 32-band checkpoint with the code as defined in (3). Zero-shot, $\eta$ fixed at 32: the same checkpoint with the code's scale held at its training value. Retrained: a model trained at that band count (12 epochs; the 32-band point is the 12-epoch reference run).*

### C. What Does Recursion Buy, and What Does It Cost?

Fig. {{F:depth}}(a) varies only the number of improvement steps $T$, from 1 to 4, which sets the number of core applications to 21, 42, 63 or 84. Quality is flat: balanced accuracy stays between 0.935 and 0.944, and the reported configuration of 63 applications is no better than 21. With $T = 1$ there is no gradient-free step at all, so on this task neither depth beyond 21 applications nor the gradient-free refinement buys anything measurable.

The cost behaves as (9) predicts (both count the same operations, so this is a consistency check rather than an independent test). Fig. {{F:depth}}(b) shows the measured arithmetic on a straight line of 95.72 MFLOPs per core application, from 2.18 GFLOPs at 21 applications to 8.21 GFLOPs at 84. Fig. {{F:depth}}(c) sets this against the hierarchy: MedMamba-SS-TRM stores 6.2× fewer parameters than MedMamba-SS (446,409 against 2,773,007) but needs 22.3× more arithmetic per patch (6.20 against 0.28 GFLOPs), at inference as well as in training. A weight-shared recursive model is small to store, not cheap to run.

<!-- tex: wide; width=1.0 -->
![figure](../v19/figures/fig_recursion_cost_hsi.png)

*Fig. {{F:depth}}. (a) Test quality against the number of core applications $K$ (12 epochs, seed 42). (b) Measured arithmetic per patch, decoder excluded, against the cost model (9). (c) Parameters against arithmetic for MedMamba-SS-TRM and the two hierarchical MedMamba-SS variants (log scales).*

### D. Which Parts Matter?

Table {{T:ablation}} removes or changes one component at a time (12 epochs, seed 42). Removing the reconstruction objective lowers balanced accuracy by 1.3 points, mostly through DCIS, and replacing the wavelength code with a plain band-index code lowers it by 0.6 points, with two independent runs agreeing closely. Both effects are smaller than the 1.9-point spread between seeds, so they suggest, but do not establish, that each component helps. If the wavelength code does help, a likely reason is the 219 nm gap, which an index code treats as an ordinary step between neighbours.

<!-- tex: wide; x=Xrrrr -->
**TABLE {{T:ablation}}**
**Component Study (Test Patients, 12 Epochs, Seed 42)**

| Variant | BA | Macro-F1 | DCIS F1 | Δ BA (points) |
| --- | ---: | ---: | ---: | ---: |
| MedMamba-SS-TRM, full | **0.9353** | **0.8920** | **0.769** | — |
| − reconstruction objective | 0.9225 | 0.8708 | 0.733 | −1.28 |
| Band-index code instead of wavelength code (two runs) | 0.9291 / 0.9287 | 0.8804 / 0.8798 | 0.748 / 0.747 | −0.62 / −0.66 |
| 16 bands, retrained | 0.9159 | 0.8636 | 0.717 | −1.95 |
| 8 bands, retrained | 0.9014 | 0.8684 | 0.715 | −3.39 |
| 21 core applications instead of 63 | 0.9378 | 0.8932 | 0.774 | +0.25 |
| Hierarchical backbone (MedMamba-SS) | 0.3342 | 0.2741 | 0.001 | −60.12 |
| Hierarchical backbone, full-channel variant | 0.4085 | 0.3974 | 0.228 | −52.68 |

*The seed-to-seed standard deviation of test BA at 20 epochs is 1.87 points (Table {{T:seeds}}).*

The last two rows put the hierarchical MedMamba-SS back in place of the recursive core, and both hierarchical variants stay near chance. For the standard variant, training accuracy rises from 0.74 to 0.91 while the validation output stays constant, a typical sign of a layer that behaves differently in training and in evaluation. The convolutional branch inherited from MedMamba uses batch normalization, and its stored statistics turned out to be far from those of test batches (channel means up to 3.8 standard deviations apart). Normalizing with the statistics of each test batch instead stops the standard variant from predicting IDC everywhere, but it still reaches only 0.49 balanced accuracy on a class-balanced sample of 9,000 test patches. Lower learning rates, removing the decoder and looser gradient clipping did not help either. Under a recipe developed for the recursive model, therefore, only the recursive model trained on 11 × 11 hyperspectral patches; why the hierarchy learns so little here is an open question.

### E. Where Does It Fail?

Fig. {{F:error}}(a) shows the test confusion matrix at seed 42. IDC is recognized almost perfectly, and at every seed the test precision for IDC rounds to 1.000, so the errors are almost all healthy patches predicted as DCIS (14.3 % of healthy test patches at seed 42).

<!-- tex: width=1.0 -->
![figure](../v19/figures/fig_error_hsi.png)

*Fig. {{F:error}}. Errors and confidence of the seed-42 model. (a) Test confusion matrix, row-normalized, with patch counts. (b) Reliability on the test and validation patients over 15 confidence bins; bins with at most 200 patches are not drawn, and the dashed diagonal is perfect calibration.*

Table {{T:patients}} shows that these errors are concentrated in individual patients, at every seed. On the test set, patient 136, the only test patient with DCIS, has a healthy recall of only 0.24–0.50 across seeds, and at seed 42 its healthy captures produce 83 % of all healthy-patch errors; the other test patients reach 0.80–1.00. On the validation set, patient 304 has a healthy recall of 0.01–0.30, and the IDC captures of patient 65 are almost never recognized. These are the ten smallest captures in the collection (250 × 1,004 pixels instead of 600 × 1,004), and why the model misses them is not known. The other seven held-out patients reach a mean macro recall of 0.85 or more. Two explanations fit these patterns, and neither has been tested. DCIS is learned from only five training patients, so the model may associate the appearance of particular slides with DCIS; and labels are assigned per capture without pixel masks, so a capture labelled healthy may contain tissue that resembles DCIS.

<!-- tex: wide; cols=llrrrrr -->
**TABLE {{T:patients}}**
**Recall per Held-Out Patient and Class, Mean (Minimum–Maximum) Over Five Seeds**

| Split | Patient | Patches | Healthy | DCIS | IDC | Macro recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Test | 136 | 88,452 | 0.40 (0.24–0.50) | 0.97 (0.95–0.99) | 0.93 (0.74–1.00) | 0.770 |
| Test | 141 | 73,710 | 0.90 (0.80–0.94) | — | 1.00 (0.99–1.00) | 0.946 |
| Test | 154 | 49,140 | — | — | 1.00 (0.99–1.00) | 0.997 |
| Test | 213 | 73,710 | 0.99 (0.97–0.99) | — | 1.00 (1.00–1.00) | 0.993 |
| Test | 229 | 63,882 | 0.99 (0.98–1.00) | — | 1.00 (1.00–1.00) | 0.995 |
| Val. | 65 | 44,590 | 0.99 (0.97–1.00) | — | 0.00 (0.00–0.00) | 0.495 |
| Val. | 68 | 73,710 | 0.98 (0.91–1.00) | — | 1.00 (0.99–1.00) | 0.987 |
| Val. | 197 | 98,280 | 0.94 (0.87–0.97) | 0.62 (0.50–0.88) | 1.00 (1.00–1.00) | 0.853 |
| Val. | 238 | 44,226 | — | — | 1.00 (1.00–1.00) | 1.000 |
| Val. | 304 | 73,710 | 0.15 (0.01–0.30) | — | 0.98 (0.95–1.00) | 0.566 |

*—: the patient has no patches of that class. Macro recall is the mean over the classes the patient has, averaged over seeds.*

Calibration tells a related story (Fig. {{F:error}}(b), Table {{T:calib}}). On the test patients the model is *under*-confident: above a confidence of 0.5, accuracy exceeds confidence in every bin. On the validation patients it is *over*-confident. Both calibration errors are small on average but point in opposite directions, so temperature scaling [@guo_calib], the standard fix, does not transfer between the two patient sets: a temperature fitted on one set makes the other worse. Confidence still ranks errors well. At seed 42 the 90 % most confident test patches are 98.4 % correct (87.1 % on the validation patients), and the area under the risk–coverage curve [@aurc] is 0.0035, which matters if uncertain cases are to be referred to a pathologist [@selective], [@medformer_ur].

<!-- tex: wide; cols=lrrrrr -->
**TABLE {{T:calib}}**
**Calibration Over Five Seeds (Mean ± S.D.)**

| Held-out patients | ECE | Brier score | Macro ROC-AUC | Temperature fitted on the other set | ECE after applying it |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 0.056 ± 0.007 | 0.097 ± 0.033 | 0.995 ± 0.001 | 1.53 ± 0.14 | 0.095 ± 0.015 |
| Validation | 0.059 ± 0.003 | 0.227 ± 0.017 | 0.945 ± 0.003 | 0.44 ± 0.09 | 0.094 ± 0.010 |

*A temperature above 1 softens the predicted probabilities and one below 1 sharpens them. Lower is better for ECE and Brier score.*

### F. What Does the Model Look At, and What Does It Remember?

The band gate (5) weights every band of every patch, so averaging it shows which wavelengths the pathway emphasizes (Fig. {{F:gate}}(a)). The gate rises from about 0.55 below 500 nm to a peak at 577 nm, stays high up to 633 nm, and drops to about 0.54 over the near-infrared bands. Its peak sits next to the 535 nm band, where the class-mean spectra differ most (Fig. {{F:gate}}(b)). Three cautions apply. The gate was never told which bands matter, but the 32 bands it weights were themselves chosen using labels (Section III); a gate weight is a learned emphasis, not proof that a band drives the prediction; and because every DCIS test patch comes from patient 136, class contrasts here are also patient contrasts.

<!-- tex: wide; width=0.92 -->
![figure](../figures/results/fig_xai_bands.png)

*Fig. {{F:gate}}. (a) Band-gate weights $a_c$ of (5), averaged within each patch, for 9,000 class-balanced test patches (seed 42), with ±1 s.d. bands. (b) Difference between each class's mean spectrum and the overall mean, for the same patches.*

Fig. {{F:recon}} shows what the final answer state still knows about the input. The auxiliary decoder rebuilds the 32-band spectra of validation patches with a mean spectral angle of 6.2° at the selected epoch (9.5° after the first epoch, 5.3° at epoch 20). Typical reconstructions follow the input spectrum closely, including the absorption minimum near 535 nm, and keep the spatial pattern of the patch. The worst case, an IDC patch with an angle of 20.6°, is a saturated, nearly uniform patch without the usual stain absorption, which the decoder pulls toward a typical tissue spectrum. Because the answer state (11 × 11 × 128) is larger than the input (32 × 11 × 11), good reconstruction is not surprising. Its value is as evidence that the representation keeps the shape of the spectrum, and as an auxiliary objective that, by Table {{T:ablation}}, may help with DCIS.

<!-- tex: wide; width=0.74 -->
![figure](../figures/results/fig_reconstruction.png)

*Fig. {{F:recon}}. Spectral reconstruction of validation patches (seed 42): one median and one worst patch per class, ranked by spectral angle (11). Left: three-band composites of input and reconstruction on a common scale, and the per-pixel spectral angle. Right: patch-mean spectra; no line is drawn across the 219 nm gap.*

---
