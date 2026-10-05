### D. Band-Count Agnosticism

**Structure.** The same MedMamba-SS-TRM on the same dataset and label space has exactly 446,409 parameters at 32 bands and at 3 bands, as Proposition 1 predicts; its parameter memory is 1.786 MB in both cases. What depends on the band count is activation size (peak inference memory 161.5 MB against 52.6 MB) and, marginally, arithmetic (6.203 against 6.052 GFLOPs), because 97 % of the arithmetic is the recursion over a fixed 121-token grid. The optional reconstruction decoder is the one band-dependent module: its output layer has one channel per band (575,849 parameters with the decoder at 32 bands, 559,116 at 3), so all parameter counts in this paper exclude it.

**Behaviour.** Fig. {{F:bands}} evaluates the seed-42 model on subsets of its bands, uniformly decimated in index, without retraining (*zero-shot*), and trains fresh models at 16 and 8 bands under the 12-epoch budget (*retrained*). Zero-shot transfer fails: at every reduced band count balanced accuracy falls to 0.33–0.41 and IDC F1 to exactly 0.000. Retraining costs little: at eight bands the model reaches 0.9014 balanced accuracy and 0.8684 macro-F1, 97 % of the 32-band macro-F1 (0.8920) on a quarter of the spectrum. The architecture is therefore band-count agnostic by construction and under retraining, while a trained model is specific to its sensor. For acquisition design this suggests that a filter camera with about eight well-chosen bands would lose little on this task.

![Fig. bands](../figures/results/fig_bands.png)

*Fig. {{F:bands}}. Test performance against the number of bands. Zero-shot: the 32-band model evaluated on decimated input. Retrained: a model trained at that band count (12-epoch budget; the 32-band point is the 12-epoch baseline).*

### E. Ablation Study

Table {{T:ablation}} removes one component at a time from MedMamba-SS-TRM at the 12-epoch budget.

**TABLE {{T:ablation}}**
**Component Ablations (Test, 12-Epoch Budget, Seed 42)**

{{ABLATION_TABLE}}

Every spectral component contributes. The auxiliary reconstruction objective adds 1.3 points of balanced accuracy and 0.021 macro-F1, most of it in DCIS (0.733 → 0.769). The wavelength encoding adds 0.6 points and 0.012 macro-F1 over an index encoding; the two index-encoding runs agree to 0.0006, well above the run-to-run floor, and the plausible source is the 219 nm gap, which an index encoding treats as a step between neighbours. The recursion depth, in contrast, can be cut to a third with no loss (Section VI-F).

The hierarchical backbone does not train under the recursive model's recipe on this data: both MedMamba-SS variants sit at chance (0.334 and 0.409 balanced accuracy) with their best epoch at 1. Lowering the learning rate to $10^{-4}$ or $3 \times 10^{-5}$, removing the reconstruction decoder and raising the clipping threshold to 5.0 all leave it at chance. The model does learn on the training split (training accuracy 0.735 → 0.906), while validation macro-F1 freezes at 0.2583 and its gradient norm (302 → 1,002) is clipped by two to three orders of magnitude at every step. The same hierarchy trains normally on PAD-UFES-20 (Section VI-K), so the incompatibility is specific to this configuration; it means the recursive substitution is established on parameters, arithmetic and interface, and its quality is compared against MedMamba rather than against MedMamba-SS.

### F. Recursion Depth and Computational Cost

Fig. {{F:depth}} varies only the number of improvement steps. Quality is flat and non-monotonic over a fourfold range of depth (balanced accuracy 0.9378, 0.9441, 0.9353 and 0.9386 at 21, 42, 63 and 84 core applications), and the reported configuration of 63 is the lowest of the four. The arithmetic, measured by forward hooks, is 2.215, 4.224, 6.234 and 8.245 GFLOPs: a straight line of 95.72 MFLOPs per application, exactly the slope that the cost model (23) predicts, on an intercept of 0.204 GFLOPs for everything outside the core.

![Fig. depth](../figures/results/fig_depth_cost.png)

*Fig. {{F:depth}}. (a) Test quality against the number of core applications (12-epoch budget). (b) Measured arithmetic against the cost model (23). (c) Parameters against arithmetic for the hyperspectral configurations.*

On the same input, MedMamba-SS-TRM holds 6.2× fewer parameters than MedMamba-SS (446,409 against 2,773,007) and needs 22.3× more arithmetic (6.203 against 0.278 GFLOPs; Fig. {{F:depth}}(c)); on the six-class RGB configuration the ratios are 61.4× and 16.4×. Because the forward pass always runs all three segments, inference pays this cost too (14.7 ms per patch at batch size 1). Parameter count in a weight-shared recursive model therefore measures storage, not compute, and the depth that this compute buys is not measurable on this task: 21 applications match 63 at a third of the arithmetic.

### G. Error Analysis

Fig. {{F:error}} shows the seed-42 test confusion matrices. Both arms classify IDC almost perfectly (100.0 % and 99.0 % recall). The errors are healthy tissue predicted as DCIS: 14.3 % of healthy patches with 32 bands and 27.5 % with 3 bands. DCIS recall is high in both (97.4 % and 94.0 %), so the difference between the arms is mostly the false-positive rate for DCIS on healthy tissue, which is also where the class-mean spectra of Fig. {{F:dataset}}(a) differ.

![Fig. error](../figures/results/fig_error_analysis.png)

*Fig. {{F:error}}. (a, b) Row-normalized test confusion matrices at seed 42 with patch counts. (c) Reliability diagram (15 bins with more than 200 patches).*

Per patient, test macro recall averaged over the five seeds is 0.95–1.00 for four of the five test patients in the 32-band arm and 0.77 for the fifth (patient 136), which is equally difficult for both arms and is the only test patient with DCIS. The reliability curves lie above the diagonal: both models are under-confident, which is consistent with EMA weights and focal loss, and which Section VI-H quantifies.

### J. Reconstruction

The auxiliary decoder reconstructs the 32-band input from the core's final answer state. On the validation subset its spectral angle falls from 9.47° after the first epoch to 6.16° at the selected epoch 7 and 5.34° at epoch 20, with RMSE 0.1109 and PSNR 20.3 dB at the selected epoch. Fig. {{F:recon}} shows typical (median spectral angle) and worst reconstructions for each class. Median reconstructions follow the input spectrum closely across both spectral regions, including the absorption minimum near 535 nm, and preserve the spatial pattern of the patch (2-D SSIM 0.66–0.80). The worst cases are informative: the worst IDC patch (20.6°) is a saturated, nearly uniform patch whose spectrum lacks the stain absorption minimum, an atypical input that the decoder maps back toward the typical tissue spectrum. That the answer state supports this reconstruction shows that it retains detailed spectral information rather than only what a three-way decision needs, and the ablation of Section VI-E shows that asking it to do so improves classification.

![Fig. recon](../figures/results/fig_reconstruction.png)

*Fig. {{F:recon}}. Spectral reconstruction on validation patches, one median and one worst patch per class (ranked by spectral angle). Left: composites (633/562/463 nm) of input and reconstruction on a common intensity scale, and the per-pixel spectral angle. Right: patch-mean spectra; no line is drawn across the 219 nm gap.*

### K. Skin Lesions

On PAD-UFES-20, 11 × 11 patches cut from whole photographs inherit the photograph's label although most contain no lesion, which is a multiple-instance setting [@mil]. At patch level MedMamba-SS-TRM therefore scores only 25.7 % balanced accuracy; averaging its patch predictions over each image raises this to 35.6 % (MedMamba: 24.8 % → 31.4 %). Under a patch recipe matched to MedMamba in every setting except the architecture, MedMamba-SS-TRM leads on accuracy (30.7 % against 27.6 %), balanced accuracy (25.8 % against 24.8 %) and macro-F1 (0.228 against 0.216) with 8.2× fewer parameters.

On whole 224 × 224 images under a patient-disjoint split (Table {{T:pad}}), MedMamba-SS-TRM reaches the highest balanced accuracy (50.2 %), macro precision and macro-F1 (0.421) of the three models, with 61× fewer parameters than MedMamba-SS and 32× fewer than MedMamba-T, while MedMamba-SS leads accuracy, specificity, Cohen's κ and ROC-AUC (0.798; 0.806 in a repeat run of the same configuration). The comparison is confounded, and the table lists how. Placed against the published PAD-UFES-20 comparison of [@medmamba], which uses an image-level split and different test images, its sensitivity (50.2 %) is higher than all eleven published rows and its macro-F1 (42.1 %) is second only to NesT-Tiny (42.3 %), while its overall accuracy (45.6 %) is the lowest.

**TABLE {{T:pad}}**
**Whole-Image PAD-UFES-20, Test Split**

{{PAD_TABLE}}
