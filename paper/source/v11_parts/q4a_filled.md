---

## VI. Experimental Results and Discussion

All results are on HistologyHSI-BC-Recurrence unless stated. Because the validation and test sets each contain five patients, every comparison is reported on the test patients, on the validation patients (full split, scored at the checkpoint selected on the validation subset), and pooled over all ten non-training patients.

### A. Comparison with Existing Models

Table {{T:comparison}} and Fig. {{F:comparison}} compare MedMamba-SS-TRM with three shallow probes, MedMamba, HybridSN, SpectralFormer and the hierarchical MedMamba-SS.

**TABLE {{T:comparison}}**
**Comparison on HistologyHSI-BC-Recurrence (Seed 42). Test: 348,894 Patches from 5 Patients; Validation: 334,516 Patches from 5 Patients; Pooled: All 10**

| Model | Params | Test Acc. (%) | Test BA (%) | Test F1 | Test κ | Val. BA (%) | Val. F1 | Pooled BA (%) | Pooled F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority class (always IDC) | — | 69.01 | 33.33 | 0.272 | 0.000 | 33.33 | 0.258 | 33.33 | 0.266 |
| Logistic regression, band means | 32 feat. | 82.13 | 82.68 | 0.726 | 0.656 | 68.39 | 0.619 | 75.81 | 0.671 |
| Logistic regression, band means + s.d. | 64 feat. | 93.07 | 90.28 | 0.843 | 0.854 | 77.99 | 0.710 | 84.31 | 0.771 |
| Logistic regression, raw patch | 3,872 feat. | 79.66 | 62.02 | 0.623 | 0.551 | 56.92 | 0.555 | 59.86 | 0.587 |
| Logistic regression, RGB means + s.d. (RGB build) | 6 feat. | 93.05 | 87.58 | 0.830 | 0.853 | 84.90 | 0.783 | 86.50 | 0.807 |
| MedMamba [@medmamba] | 3.649 M | 95.14 | 91.02 | 0.872 | 0.896 | 72.30 | 0.689 | 81.67 | 0.774 |
| HybridSN [@hybridsn] | 0.570 M | 93.56 | 88.85 | 0.841 | 0.863 | 84.94 | 0.785 | 87.14 | 0.813 |
| SpectralFormer [@spectralformer] | 0.121 M | 92.01 | 82.45 | 0.804 | 0.826 | 80.73 | 0.766 | 81.78 | 0.784 |
| MedMamba-SS, 12-epoch budget | 2.773 M | 69.06 | 33.42 | 0.274 | 0.002 | 33.30 | 0.259 | 33.36 | 0.267 |
| MedMamba-SS, full-channel variant, 12-epoch budget | 3.513 M | 70.01 | 40.85 | 0.397 | 0.131 | 30.20 | 0.269 | 35.64 | 0.329 |
| **MedMamba-SS-TRM** | **0.446 M** | 96.38 | 94.37 | 0.904 | 0.923 | 72.11 | 0.694 | 83.21 | 0.795 |

*BA: balanced accuracy (26); F1: macro-F1 (27). MedMamba's pooled scores are computed exactly from its per-class validation counts, since its validation predictions were not stored. Section VI-H compares the calibration of these models.*

![Fig. comparison](../figures/results/fig_comparison.png)

*Fig. {{F:comparison}}. Balanced accuracy of every model on the test patients, the validation patients and all ten held-out patients pooled (seed 42).*

**On the test patients, MedMamba-SS-TRM is the most accurate model**, with 94.37 % balanced accuracy and 0.904 macro-F1, ahead of MedMamba (91.02 %, 0.872) with 8.2× fewer parameters, of HybridSN (88.85 %, 0.841) and SpectralFormer (82.45 %, 0.804), and of the strongest probe, a logistic regression on the 64 per-band means and standard deviations (90.28 %, 0.843). **On the validation patients the order reverses**: HybridSN (84.94 %) and a six-feature RGB colour probe (84.90 %) lead, and MedMamba-SS-TRM (72.11 %) and MedMamba (72.30 %) trail the 64-feature probe (77.99 %). **Pooled over all ten patients**, HybridSN leads (87.14 %, 0.813) and the six-feature RGB probe follows (86.50 %, 0.807). Among models that read the 32-band input, MedMamba-SS-TRM is second on pooled macro-F1 (0.795) and third on pooled balanced accuracy (83.21 %, behind the 64-feature probe at 84.31 %), and it leads its base architecture MedMamba on both pooled metrics (+1.5 points, +0.021 macro-F1). The hierarchical MedMamba-SS does not train under this recipe and stays at chance on every held-out set; its test macro-F1 (0.274) is that of predicting IDC everywhere (0.272) (Section VI-E).

Three observations follow. First, the ranking of models depends on which five patients are held out more than on the architecture, partly for a structural reason, since each held-out set contains DCIS from a single patient: the gap between a network's test and validation balanced accuracy ranges from 1.7 points (SpectralFormer) to 22.3 points (MedMamba-SS-TRM), larger than most differences between networks. Second, simple spectral statistics are a strong baseline on this dataset, and the flattened raw patch is a weak one (62.02 %), so the information is mainly spectral rather than spatial at 11 × 11. Third, the two models that do best on validation, HybridSN and SpectralFormer, select their first and third epoch, whereas MedMamba-SS-TRM selects its seventh; on these data, models that stop earlier fit the validation patients better and the test patients less well.

### B. Learning Curves

Fig. {{F:learning}} shows the training dynamics of the seed-42 pair. Training classification loss falls monotonically for both inputs (32 bands: 0.128 to 0.034). Validation loss for the 32-band model reaches its minimum at epoch 9 (0.101) and then rises to 0.169 at epoch 20, and validation macro-F1 peaks at epoch 7 and declines slowly; the model keeps fitting the 35 training patients after it stops improving on the five validation patients, and the checkpoint rule selects the epoch before that divergence. The 3-band model peaks earlier (epoch 4). The spectral-angle reconstruction loss falls steadily on both splits for the 32-band model (validation 1.27 to 0.55) and is lower for the 3-band model, whose three-channel target is easier to reconstruct.

![Fig. learning](../figures/results/fig_learning_curves.png)

*Fig. {{F:learning}}. Training dynamics at seed 42. (a) Training and validation classification loss (log scale). (b) Validation macro-F1; dots mark the selected checkpoints. (c) Training and validation spectral-angle loss.*

### C. Hyperspectral Versus RGB Input

The two arms of this comparison use the same model (446,409 parameters), recipe, patient split and patch coordinates, and differ only in the input directory (Section III-B); each is repeated at five seeds. Table {{T:modality}} gives balanced accuracy on each held-out set.

**TABLE {{T:modality}}**
**Balanced Accuracy of the 32-Band (HSI) and 3-Band (RGB) Arms, Five Seeds; Δ in Points**

| Seed | Test HSI | Test RGB | Δ | Val. HSI | Val. RGB | Δ | Pooled HSI | Pooled RGB | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 89.61 | 88.18 | +1.43 | 83.00 | 84.81 | −1.81 | 86.32 | 86.73 | −0.41 |
| 7 | 93.41 | 88.97 | +4.45 | 74.37 | 80.41 | −6.05 | 83.94 | 84.89 | −0.95 |
| 13 | 92.76 | 88.79 | +3.97 | 78.66 | 78.94 | −0.28 | 85.74 | 83.99 | +1.75 |
| 23 | 93.78 | 88.89 | +4.88 | 72.77 | 82.39 | −9.62 | 83.26 | 85.86 | −2.59 |
| 42 | 94.37 | 88.51 | +5.86 | 72.11 | 83.41 | −11.31 | 83.21 | 86.17 | −2.96 |
| **Mean** | **92.78** | **88.67** | **+4.12 ± 1.66** | **76.18** | **81.99** | **−5.81 ± 4.78** | **84.49** | **85.53** | **−1.04 ± 1.89** |

On the test patients, balanced accuracy favours 32-band input at five of five seeds, by 4.12 ± 1.66 points (macro-F1 +0.044 ± 0.038, positive at four of five), and the gain lies in healthy tissue and DCIS (mean per-class F1 0.892 against 0.844 and 0.739 against 0.656; IDC 0.993 against 0.994). On the validation patients, scored on the full split, the sign reverses: 3-band input is better at five of five seeds, by 5.81 ± 4.78 points. Pooled over all ten patients the difference is −1.04 ± 1.89 points (positive at one of five seeds). Resampling patients, the 95 % interval of the seed-mean difference is [+1.6, +4.8] points on the test patients, [−11.8, −1.4] points on the validation patients and [−6.6, +3.9] points pooled.

Fig. {{F:modality}}(b) locates the reversal. Averaged over seeds, 32-band input gives the higher macro recall for seven of the ten held-out patients (patient 136 is tied at 0.770), and the validation reversal comes almost entirely from one patient, 304, whose macro recall is 0.566 with 32 bands against 0.773 with 3. The same patient drives the reversal without any network: a logistic regression on per-band means and standard deviations scores 90.28 % on the test patients with 32 bands against 87.58 % with 3, and 77.99 % against 84.90 % on the validation patients, where patient 304 again favours 3 bands (0.672 against 0.836). The 3-band model is also the more stable across seeds (test balanced accuracy s.d. 0.3 against 1.9 points). The comparison therefore does not establish that 32-band input is better for this task. It shows that 32 bands help most patients and fail badly on one, and that ten patients, with DCIS present in only one patient per held-out set, are too few to settle the question.

![Fig. modality](../figures/results/fig_modality.png)

*Fig. {{F:modality}}. (a) HSI − RGB balanced accuracy per seed on the test patients, the validation patients and all ten pooled. (b) Macro recall of every held-out patient (split, number of patches), averaged over the five seeds, for both inputs.*

