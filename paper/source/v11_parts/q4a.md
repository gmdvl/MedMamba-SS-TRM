---

## VI. Experimental Results and Discussion

All results are on HistologyHSI-BC-Recurrence unless stated. Because the validation and test sets each contain five patients, every comparison is reported on the test patients, on the validation patients (full split, scored at the checkpoint selected on the validation subset), and pooled over all ten non-training patients.

### A. Comparison with Existing Models

Table {{T:comparison}} and Fig. {{F:comparison}} compare MedMamba-SS-TRM with three shallow probes, MedMamba, HybridSN, SpectralFormer and the hierarchical MedMamba-SS.

**TABLE {{T:comparison}}**
**Comparison on HistologyHSI-BC-Recurrence (Seed 42). Test: 348,894 Patches from 5 Patients; Validation: 334,516 Patches from 5 Patients; Pooled: All 10**

| Model | Params | Test Acc. (%) | Test BA (%) | Test F1 | Test κ | Val. BA (%) | Val. F1 | Pooled BA (%) | Pooled F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{{COMPARISON_ROWS}}

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
{{MODALITY_ROWS}}

{{MODALITY_TEXT}}

![Fig. modality](../figures/results/fig_modality.png)

*Fig. {{F:modality}}. (a) HSI − RGB balanced accuracy per seed on the test patients, the validation patients and all ten pooled. (b) Macro recall of every held-out patient (split, number of patches), averaged over the five seeds, for both inputs.*

