### H. Calibration

Table {{T:calib}} reports calibration of MedMamba-SS-TRM over the five seeds of each arm and, at seed 42, of the other networks and the two strongest probes of Table {{T:comparison}}.

**TABLE {{T:calib}}**
**Calibration on the Test and Validation Patients. MedMamba-SS-TRM: Mean ± S.D. Over Five Seeds; Other Models: Seed 42**

| Held-out set | Model | ECE (29) | MCE | Brier | NLL | Macro ROC-AUC | ECE, temperature fitted on the other set |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | **MedMamba-SS-TRM, 32 bands** | 0.056 ± 0.007 | 0.202 ± 0.034 | 0.097 ± 0.033 | 0.167 ± 0.062 | 0.9953 ± 0.0009 | 0.095 ± 0.015 |
| Test | MedMamba-SS-TRM, 3 bands | 0.094 ± 0.017 | 0.273 ± 0.020 | 0.134 ± 0.016 | 0.232 ± 0.033 | 0.9900 ± 0.0006 | 0.222 ± 0.028 |
| Test | MedMamba [@medmamba] | 0.013 | 0.067 | 0.069 | 0.113 | 0.9924 | — |
| Test | HybridSN [@hybridsn] | 0.062 | 0.245 | 0.112 | 0.182 | 0.9883 | 0.326 |
| Test | SpectralFormer [@spectralformer] | 0.035 | 0.182 | 0.120 | 0.202 | 0.9756 | 0.110 |
| Test | Logistic regression, band means + s.d. (64 feat.) | 0.025 | 0.084 | 0.105 | 0.187 | 0.9912 | 0.373 |
| Test | Logistic regression, RGB means + s.d. (6 feat.) | 0.035 | 0.134 | 0.115 | 0.186 | 0.9839 | 0.134 |
| Validation | **MedMamba-SS-TRM, 32 bands** | 0.059 ± 0.003 | 0.177 ± 0.088 | 0.227 ± 0.017 | 0.393 ± 0.062 | 0.9451 ± 0.0029 | 0.094 ± 0.010 |
| Validation | MedMamba-SS-TRM, 3 bands | 0.070 ± 0.014 | 0.219 ± 0.027 | 0.240 ± 0.013 | 0.812 ± 0.048 | 0.9505 ± 0.0044 | 0.057 ± 0.004 |
| Validation | HybridSN [@hybridsn] | 0.076 | 0.230 | 0.200 | 1.827 | 0.9350 | 0.060 |
| Validation | SpectralFormer [@spectralformer] | 0.052 | 0.167 | 0.200 | 0.601 | 0.9472 | 0.040 |
| Validation | Logistic regression, band means + s.d. (64 feat.) | 0.071 | 0.107 | 0.289 | 1.941 | 0.9128 | 0.075 |
| Validation | Logistic regression, RGB means + s.d. (6 feat.) | 0.057 | 0.274 | 0.211 | 0.670 | 0.9628 | 0.051 |

*MCE: largest $|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)|$ over the bins of (29) that hold at least 0.1 % of the patches. MedMamba-SS-TRM: mean ± s.d. over five seeds; other models: one run at seed 42. MedMamba's validation probabilities were not stored, so it has no validation row and no transferred temperature. Without that floor, MedMamba's test MCE would be 0.474, set by a bin of 7 patches; that of two MedMamba-SS-TRM validation runs would be set by a bin of fewer than 335 patches.*

On the test patients the 32-band model is better calibrated than its 3-band twin at every seed (ECE 0.056 ± 0.007 against 0.094 ± 0.017; Brier 0.097 against 0.134), and on the validation patients as well (ECE 0.059 against 0.070). The reliability diagram (Fig. {{F:error}}(c)) shows the direction of the error on the test patients: above a confidence of 0.5, accuracy exceeds confidence in every bin, so both models are under-confident, as expected from EMA weights and a focal loss that down-weights confident predictions. Temperature scaling fitted on each run's validation predictions does not repair this. The fitted temperature is above one (T = 1.53 ± 0.14 for 32 bands, 1.99 ± 0.11 for 3 bands), because the models are over-confident on the validation patients, and applied to the test patients it raises test ECE from 0.056 to 0.095 (32 bands) and raises it from 0.094 to 0.222 (3 bands). In the other direction, a temperature fitted on the test patients is below one (T = 0.44 and 0.34); applied to the validation patients it raises validation ECE from 0.059 to 0.094 (32 bands) and lowers it from 0.070 to 0.057 (3 bands; last column of Table {{T:calib}}). So for the 32-band model no single temperature fits both held-out sets, whereas for the 3-band model the temperature fitted on the test patients also lowers validation ECE. Among the other models, MedMamba (0.013), the 64-feature probe (0.025), SpectralFormer (0.035) and the RGB probe (0.035) are better calibrated than the 32-band MedMamba-SS-TRM on the test patients and HybridSN (0.062) is worse; on the validation patients SpectralFormer (0.052) and the RGB probe (0.057) are better and the 64-feature probe (0.071) and HybridSN (0.076) are worse. The transferred temperature fails for HybridSN, SpectralFormer, the 64-feature probe and the RGB probe as well: fitted on the validation patients it is above one for each of them (T = 1.70 to 5.51) and raises test ECE, most for the 64-feature probe (0.025 to 0.373) and HybridSN (0.062 to 0.326); fitted on the test patients it lowers validation ECE for HybridSN, SpectralFormer and the RGB probe and raises it for the 64-feature probe. All six models with validation predictions are over-confident on the validation patients and under-confident on the test patients, so the miscalibration follows the patient split rather than the architecture, and on these data a temperature fitted on five patients does not transfer to five others.

**Selective prediction.** Ranking test patches by confidence, the 32-band MedMamba-SS-TRM has the lowest area under the risk–coverage curve (AURC 0.0035, against 0.0038 for MedMamba, 0.0068 for HybridSN, 0.0104 for SpectralFormer and 0.0103 for its RGB twin), with 98.4 % accuracy on the 90 % most confident patches. On the validation patients the order reverses (AURC 0.033, against 0.024 for HybridSN and 0.018 for SpectralFormer), the same patient dependence as in Section VI-A.

