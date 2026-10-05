### H. Calibration

Table {{T:calib}} reports calibration of MedMamba-SS-TRM over the five seeds of each arm and, at seed 42, of the other networks and the two strongest probes of Table {{T:comparison}}.

**TABLE {{T:calib}}**
**Calibration on the Test and Validation Patients. MedMamba-SS-TRM: Mean ± S.D. Over Five Seeds; Other Models: Seed 42**

| Held-out set | Model | ECE (29) | MCE | Brier | NLL | Macro ROC-AUC | ECE, temperature fitted on the other set |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{{CALIB_ROWS}}

{{CALIB_NOTE}}

On the test patients the 32-band model is better calibrated than its 3-band twin at every seed (ECE 0.056 ± 0.007 against 0.094 ± 0.017; Brier 0.097 against 0.134){{CALIB_VAL}}. The reliability diagram (Fig. {{F:error}}(c)) shows the direction of the error on the test patients: above a confidence of 0.5, accuracy exceeds confidence in every bin, so both models are under-confident, as expected from EMA weights and a focal loss that down-weights confident predictions. {{CALIB_TEMP}} {{CALIB_BASE}}

**Selective prediction.** Ranking test patches by confidence, the 32-band MedMamba-SS-TRM has the lowest area under the risk–coverage curve (AURC 0.0035, against 0.0038 for MedMamba, 0.0068 for HybridSN, 0.0104 for SpectralFormer and 0.0103 for its RGB twin), with 98.4 % accuracy on the 90 % most confident patches. On the validation patients the order reverses (AURC 0.033, against 0.024 for HybridSN and 0.018 for SpectralFormer), the same patient dependence as in Section VI-A.

