## V. Experimental Setup

### A. Loss Function

Classification uses the focal loss [@focal] with class weights, which down-weights well-classified patches and counteracts the 13 : 1 class imbalance. For a patch of class $k$ with predicted probability $p_k$,

$$\mathcal{L}_{\text{FL}} = -\,w_k\,(1 - p_k)^{\gamma} \log p_k, \qquad \gamma = 1.5, \tag{24}$$

$$w_k = \frac{\tilde{w}_k}{\sum_j \tilde{w}_j}\, K_{\text{cls}}, \qquad \tilde{w}_k = \Big(\frac{N}{K_{\text{cls}} N_k}\Big)^{0.75}, \tag{25}$$

where $N_k$ is the number of training patches of class $k$ and the exponent 0.75 softens full inverse-frequency weighting. For MedMamba-SS-TRM, (24) enters the deep-supervision objective (22).

### B. Evaluation Metrics

Because the classes are imbalanced, balanced accuracy and macro-F1 are the primary metrics:

$$\mathrm{BA} = \frac{1}{K_{\text{cls}}} \sum_{k} \frac{\mathrm{TP}_k}{\mathrm{TP}_k + \mathrm{FN}_k}, \tag{26}$$

$$\mathrm{F1}_{\text{macro}} = \frac{1}{K_{\text{cls}}} \sum_{k} \frac{2\,P_k R_k}{P_k + R_k}. \tag{27}$$

Reconstruction is scored by the spectral angle between a reconstructed spectrum $\hat{x}$ and its target $x$, averaged over pixels, together with RMSE, PSNR and 2-D SSIM:

$$\mathrm{SAM}(\hat{x}, x) = \arccos \frac{\langle \hat{x}, x \rangle}{\lVert \hat{x} \rVert\, \lVert x \rVert}. \tag{28}$$

Calibration matters for a clinical classifier whose confidence may be used to defer uncertain cases to a pathologist [@medformer_ur]. It is measured by the expected calibration error over $M = 15$ confidence bins $B_m$, together with the Brier score:

$$\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|. \tag{29}$$

We also report accuracy, macro precision, Cohen's $\kappa$ and macro ROC-AUC, and per-patient macro recall, the mean recall over the classes a patient has.

### C. Implementation Details

All models were trained on one NVIDIA RTX 5060 Ti (16 GB) with PyTorch 2.11 [@pytorch] and CUDA 12.8. MedMamba-SS-TRM uses width 128, two core blocks, $n = 6$, $T = 3$, $N_{\text{sup}} = 3$, AdamW [@adamw] at learning rate $3 \times 10^{-4}$ with weight decay 0.05 (not applied to normalization weights and biases), batch size 256, bf16 precision, gradient clipping at 1.0, and a 587-step linear warmup into cosine decay over 19,560 steps, stepped per optimizer step. Inputs are z-scored with per-band statistics fitted on training data, and training uses geometric (flips, 90° rotations, crops) and spectral (noise, scaling, offset) augmentation. Each epoch trains on a fresh random 10.2 % of the training split (250,113 patches, 978 steps) and validates on a fixed patient-stratified 9.86 % of the validation split (32,985 patches); the checkpoint with the best validation macro-F1 is tested on the full test split. Headline runs train for 20 epochs; every ablation trains for 12 epochs (11,736 steps) and is compared with a 12-epoch baseline, because the cosine schedule makes a shorter run a different schedule rather than a truncation. Table {{T:config}} lists the full configuration.

**TABLE {{T:config}}**
**Training Configuration of MedMamba-SS-TRM**

| Group | Setting |
| --- | --- |
| Architecture | Width 128; 2 core blocks; $n = 6$, $T = 3$, $N_{\text{sup}} = 3$ ($K = 63$); depthwise-convolution mixer with GEGLU channel MLP; two carried states; core gradient checkpointing; halting off; drop-path 0; core dropout 0; classifier dropout 0.1 |
| Spectral pathway | $d_t = 32$; $d_{\text{ctx}} = 64$; three residual spectral blocks and one spectral scan (state 8); soft band gate; compressor 128 → 64 → 64; concat-MLP tokenizer, value initialization s.d. 0.5; spectral positional gain 0.1; wavelength encoding with $\sigma = C$; chunk size 1,024; spectral checkpointing for $C \geq 16$ |
| Stem and head | Scale-invariant RMS normalization of the context; linear projection with layer normalization; 2-D positional gain 0.1; fan-in classifier initialization |
| Optimization | AdamW, $3 \times 10^{-4}$, weight decay 0.05 (not on norms and biases); batch 256; bf16; clipping at 1.0; 587 warmup steps into cosine decay over 19,560 steps (12-epoch ablations: 352 and 11,736); EMA 0.9995; graph compilation |
| Loss | Focal, $\gamma = 1.5$, class-weight exponent 0.75, mean over three segments; reconstruction decoder on the answer state, $\lambda_{\text{mse}} = \lambda_{\text{sam}} = 0.1$, linear output |
| Data | Global z-score fitted on training data; random 10.2 % training subset per epoch (250,113 patches); fixed patient-stratified 9.86 % validation subset (32,985); flips, 90° rotations, crops, spectral noise/scale/offset; no sampler |
| Selection | Validation macro-F1 on EMA weights; early-stopping patience 40; seeds 1, 7, 13, 23, 42 |

Three implementation choices were fixed by dedicated measurements. With the SS2D mixer the recursive core ran a Python-level scan over 121 positions 168 times per step and took 91.8 hours for one epoch; the convolutional mixer is 39× faster under otherwise identical settings. Without gradient checkpointing in the core, the 0.45 M-parameter model does not fit in 16 GB on hyperspectral input. And evaluating the spectral pathway in chunks of 1,024 positions, with per-chunk checkpointing for $C \geq 16$, cuts its peak memory by 90 % (10.2 GB to 1.0 GB) for a 15 % increase in step time.

Every run passes validity gates before a result is recorded: data integrity and patient-level leakage checks, a representation-sensitivity check (stem sensitivity 0.504 and 0.552 against a floor of 0.05 for the seed-42 hyperspectral and RGB runs), a split-drift check, a check that the reconstruction gradient reaches the core, and recomputation of the selection metric from the saved checkpoint (difference 0.0 in both runs).

### D. Baselines

All baselines are trained on the same hyperspectral build and patient split and scored on the same full test and validation splits.

- **Shallow probes.** Class-balanced logistic regressions [@sklearn] on per-band means (32 features), per-band means and standard deviations (64), and the flattened patch (3,872), fitted on 25,000 training patches under the same normalization.
- **MedMamba** [@medmamba]. The reference implementation with a 32-band stem, patch size 1 and widths (64, 128, 256, 512), depths (1, 1, 2, 1): 3,648,995 parameters, trained for five full epochs with class-weighted cross-entropy at learning rate $10^{-4}$.
- **HybridSN** [@hybridsn] (569,843 parameters) and **SpectralFormer** [@spectralformer] (patch-wise, cross-layer adaptive fusion; 121,428 parameters), re-implemented for 11 × 11 × 32 patches and trained with MedMamba-SS-TRM's 20-epoch recipe: the same loss, optimizer, schedule, subsampling, augmentation and selection rule, without EMA or reconstruction.
- **MedMamba-SS** (2,773,007 parameters) under MedMamba-SS-TRM's recipe at the 12-epoch budget.

HybridSN and SpectralFormer are included because MedMamba, the base architecture, has no mechanism that reads the spectral axis, so on its own it cannot show whether the spectral pathway improves on established ways of modelling spectra. The two models represent the two main families of spectral–spatial classifiers: HybridSN convolves jointly over bands and space, and SpectralFormer treats groups of neighbouring bands as Transformer tokens, which is the closest published design to our band-token pathway. Both are small (0.57 M and 0.12 M parameters), so they compare with MedMamba-SS-TRM (0.45 M) at similar storage, and neither is band-count agnostic: HybridSN's 2-D convolution and SpectralFormer's position embedding have shapes that depend on the number of bands.

### E. Protocol

The hyperspectral-versus-RGB comparison repeats both arms at five seeds (1, 7, 13, 23, 42); the two arms differ only in the input directory, and both have 446,409 parameters. Every other configuration uses seed 42. Two pairs of runs with identical configuration and seed differ by 0.04 and 0.06 points of balanced accuracy, which is the run-to-run floor. Because the validation and test sets each contain only five patients, we report both, and we evaluate every seed's selected checkpoint on the *full* validation split, so that all ten non-training patients can be analysed. Patient-level 95 % intervals resample patients with replacement.

---

## VI. Experimental Results and Discussion

All results are on HistologyHSI-BC-Recurrence unless stated. Because the validation and test sets each contain five patients, every comparison is reported on the test patients, on the validation patients (full split, scored at the checkpoint selected on the validation subset), and pooled over all ten non-training patients. The section first ranks MedMamba-SS-TRM against existing models (Sections VI-A and VI-B), then examines the spectral input and the spectral pathway (Sections VI-C and VI-D), the components and the cost of both proposed models (Sections VI-E and VI-F), the errors, calibration and interpretability of MedMamba-SS-TRM (Sections VI-G to VI-J), and both models on skin lesions (Section VI-K).

### A. Comparison with Existing Models

Table {{T:comparison}} and Fig. {{F:comparison}} compare MedMamba-SS-TRM and MedMamba-SS with three shallow probes, a colour probe on the paired RGB build, MedMamba, HybridSN and SpectralFormer.

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

![figure](../figures/results/fig_comparison.png)

*Fig. {{F:comparison}}. Balanced accuracy of every model on the test patients, the validation patients and all ten held-out patients pooled (seed 42).*

**On the test patients, MedMamba-SS-TRM is the most accurate model**, with 94.37 % balanced accuracy and 0.904 macro-F1, ahead of MedMamba (91.02 %, 0.872) with 8.2× fewer parameters, of HybridSN (88.85 %, 0.841) and SpectralFormer (82.45 %, 0.804), and of the strongest probe, a logistic regression on the 64 per-band means and standard deviations (90.28 %, 0.843). **On the validation patients the order reverses**: HybridSN (84.94 %) and a six-feature RGB colour probe (84.90 %) lead, and MedMamba-SS-TRM (72.11 %) and MedMamba (72.30 %) trail the 64-feature probe (77.99 %). **Pooled over all ten patients**, HybridSN leads (87.14 %, 0.813) and the six-feature RGB probe follows (86.50 %, 0.807). Among models that read the 32-band input, MedMamba-SS-TRM is second on pooled macro-F1 (0.795) and third on pooled balanced accuracy (83.21 %, behind the 64-feature probe at 84.31 %), and it leads its base architecture MedMamba on both pooled metrics (+1.5 points, +0.021 macro-F1). The hierarchical MedMamba-SS does not train under this recipe and stays at chance on every held-out set; its test macro-F1 (0.274) is that of predicting IDC everywhere (0.272) (Section VI-E).

Three observations follow. First, the ranking of models depends on which five patients are held out more than on the architecture, partly for a structural reason, since each held-out set contains DCIS from a single patient: the gap between a network's test and validation balanced accuracy ranges from 1.7 points (SpectralFormer) to 22.3 points (MedMamba-SS-TRM), larger than most differences between networks. Second, simple spectral statistics are a strong baseline on this dataset, and the flattened raw patch is a weak one (62.02 %), so the information is mainly spectral rather than spatial at 11 × 11. Third, the two models that do best on validation, HybridSN and SpectralFormer, select their first and third epoch, whereas MedMamba-SS-TRM selects its seventh; on these data, models that stop earlier fit the validation patients better and the test patients less well.

### B. Learning Curves

Fig. {{F:curves}} shows the training dynamics of the seed-42 pair of MedMamba-SS-TRM models. Training classification loss falls monotonically for both inputs (32 bands: 0.128 to 0.034). Validation loss for the 32-band model reaches its minimum at epoch 9 (0.101) and then rises to 0.169 at epoch 20, and validation macro-F1 peaks at epoch 7 and declines slowly; the model keeps fitting the 35 training patients after it stops improving on the five validation patients, and the checkpoint rule selects the epoch before that divergence. The 3-band model peaks earlier (epoch 4). The spectral-angle reconstruction loss falls steadily on both splits for the 32-band model (validation 1.27 to 0.55) and is lower for the 3-band model, whose three-channel target is easier to reconstruct.

![figure](../figures/results/fig_learning_curves.png)

*Fig. {{F:curves}}. Training dynamics at seed 42. (a) Training and validation classification loss (log scale). (b) Validation macro-F1; dots mark the selected checkpoints. (c) Training and validation spectral-angle loss.*

### C. Hyperspectral Versus RGB Input

The two arms of this comparison use the same model (446,409 parameters), recipe, patient split and patch coordinates, and differ only in the input directory (Section III-B); each is repeated at five seeds. Table {{T:modality}} gives balanced accuracy on each held-out set.

**TABLE {{T:modality}}**
**Balanced Accuracy of the 32-Band (HSI) and 3-Band (RGB) Arms of MedMamba-SS-TRM, Five Seeds; Δ in Points**

| Seed | Test HSI | Test RGB | Δ | Val. HSI | Val. RGB | Δ | Pooled HSI | Pooled RGB | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 89.61 | 88.18 | +1.43 | 83.00 | 84.81 | −1.81 | 86.32 | 86.73 | −0.41 |
| 7 | 93.41 | 88.97 | +4.45 | 74.37 | 80.41 | −6.05 | 83.94 | 84.89 | −0.95 |
| 13 | 92.76 | 88.79 | +3.97 | 78.66 | 78.94 | −0.28 | 85.74 | 83.99 | +1.75 |
| 23 | 93.78 | 88.89 | +4.88 | 72.77 | 82.39 | −9.62 | 83.26 | 85.86 | −2.59 |
| 42 | 94.37 | 88.51 | +5.86 | 72.11 | 83.41 | −11.31 | 83.21 | 86.17 | −2.96 |
| **Mean** | **92.78** | **88.67** | **+4.12 ± 1.66** | **76.18** | **81.99** | **−5.81 ± 4.78** | **84.49** | **85.53** | **−1.04 ± 1.89** |

On the test patients, balanced accuracy favours 32-band input at five of five seeds, by 4.12 ± 1.66 points (macro-F1 +0.044 ± 0.038, positive at four of five), and the gain lies in healthy tissue and DCIS (mean per-class F1 0.892 against 0.844 and 0.739 against 0.656; IDC 0.993 against 0.994). On the validation patients, scored on the full split, the sign reverses: 3-band input is better at five of five seeds, by 5.81 ± 4.78 points. Pooled over all ten patients the difference is −1.04 ± 1.89 points (positive at one of five seeds). Resampling patients, the 95 % interval of the seed-mean difference is [+1.6, +4.8] points on the test patients, [−11.8, −1.4] points on the validation patients and [−6.6, +3.9] points pooled.

Fig. {{F:modality}}(b) locates the reversal. Averaged over seeds, 32-band input gives the higher macro recall for seven of the ten held-out patients, patient 136 is tied at 0.770, and the remaining two are validation patients: patient 197 favours 3 bands narrowly (0.853 against 0.866), and patient 304 favours them by a wide margin, with macro recall 0.566 with 32 bands against 0.773 with 3. The validation reversal therefore comes almost entirely from patient 304. The same patient drives the reversal without any network: a logistic regression on per-band means and standard deviations scores 90.28 % on the test patients with 32 bands against 87.58 % with 3, and 77.99 % against 84.90 % on the validation patients, where patient 304 again favours 3 bands (0.672 against 0.836). The 3-band model is also the more stable across seeds (test balanced accuracy s.d. 0.3 against 1.9 points). The comparison therefore does not establish that 32-band input is better for this task. It shows that 32 bands help most patients and fail badly on one, and that ten patients, with DCIS present in only one patient per held-out set, are too few to settle the question.

![figure](../figures/results/fig_modality.png)

*Fig. {{F:modality}}. (a) HSI − RGB balanced accuracy per seed on the test patients, the validation patients and all ten pooled. (b) Macro recall of every held-out patient (split, number of patches), averaged over the five seeds, for both inputs.*

### D. Band-Count Agnosticism

**Structure.** The same MedMamba-SS-TRM on the same dataset and label space has exactly 446,409 parameters at 32 bands and at 3 bands, as Proposition 1 predicts for any model built on the spectral pathway; its parameter memory is 1.786 MB in both cases. What depends on the band count is activation size (peak inference memory 161.5 MB against 52.6 MB) and, marginally, arithmetic (6.203 against 6.052 GFLOPs), because 97 % of the arithmetic is the recursion over a fixed 121-token grid. The optional reconstruction decoder is the one band-dependent module: its output layer has one channel per band (575,849 parameters with the decoder at 32 bands, 559,116 at 3), so all parameter counts in this paper exclude it.

**Behaviour.** Fig. {{F:bands}} evaluates the seed-42 model on subsets of its bands, uniformly decimated in index, without retraining (*zero-shot*), and trains fresh models at 16 and 8 bands under the 12-epoch budget (*retrained*). Zero-shot transfer fails: at every reduced band count balanced accuracy falls to 0.33–0.41 and IDC F1 to exactly 0.000. Retraining costs little: at eight bands the model reaches 0.9014 balanced accuracy and 0.8684 macro-F1, 97 % of the 32-band macro-F1 (0.8920) on a quarter of the spectrum. The architecture is therefore band-count agnostic by construction and under retraining, while a trained model is specific to its sensor. Training on random band subsets, as ChannelViT's hierarchical channel sampling does for channels [@channelvit], targets exactly this failure and is the natural remedy to test. For acquisition design this suggests that a filter camera with about eight well-chosen bands would lose little on this task.

![figure](../figures/results/fig_bands.png)

*Fig. {{F:bands}}. Test performance against the number of bands. Zero-shot: the 32-band model evaluated on decimated input. Retrained: a model trained at that band count (12-epoch budget; the 32-band point is the 12-epoch baseline).*

### E. Ablation Study

Table {{T:ablation}} removes one component at a time from MedMamba-SS-TRM at the 12-epoch budget, including the substitution that defines it: the last two rows put the hierarchical MedMamba-SS backbone back in place of the recursive core.

**TABLE {{T:ablation}}**
**Component Ablations (Test, 12-Epoch Budget, Seed 42)**

| Variant | Balanced accuracy | Macro-F1 | DCIS F1 | Δ macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| MedMamba-SS-TRM, full | **0.9353** | **0.8920** | **0.769** | — |
| − reconstruction objective ($\lambda = 0$) | 0.9225 | 0.8708 | 0.733 | −0.021 |
| − wavelength encoding (index encoding; two runs) | 0.9291 / 0.9287 | 0.8804 / 0.8798 | 0.748 / 0.747 | −0.012 |
| 16 bands (retrained) | 0.9159 | 0.8636 | 0.717 | −0.028 |
| 8 bands (retrained) | 0.9014 | 0.8684 | 0.715 | −0.024 |
| 21 core applications instead of 63 | 0.9378 | 0.8932 | 0.774 | +0.001 |
| hierarchical backbone (MedMamba-SS) | 0.3342 | 0.2741 | 0.001 | −0.618 |
| hierarchical backbone, full-channel variant | 0.4085 | 0.3974 | 0.228 | −0.495 |

Every spectral component contributes. The auxiliary reconstruction objective adds 1.3 points of balanced accuracy and 0.021 macro-F1, most of it in DCIS (0.733 → 0.769). The wavelength encoding of the spectral pathway adds 0.6 points and 0.012 macro-F1 over an index encoding; the two index-encoding runs agree to 0.0006, well above the run-to-run floor, and the plausible source is the 219 nm gap, which an index encoding treats as a step between neighbours. The recursion depth, in contrast, can be cut to a third with no loss (Section VI-F).

The hierarchical MedMamba-SS does not train under the recursive model's recipe on this data: both variants sit at chance (0.334 and 0.409 balanced accuracy) with their best epoch at 1. Lowering the learning rate to $10^{-4}$ or $3 \times 10^{-5}$, removing the reconstruction decoder and raising the clipping threshold to 5.0 all leave it at chance. The model does learn on the training split (training accuracy 0.735 → 0.906), while validation macro-F1 freezes at 0.2583 and its gradient norm (302 → 1,002) is clipped by two to three orders of magnitude at every step. The same hierarchy trains normally on PAD-UFES-20 (Section VI-K), so the incompatibility is specific to this configuration rather than to the architecture. On hyperspectral patches, therefore, the recursive substitution is established on parameters, arithmetic and interface, and its quality is compared against MedMamba rather than against MedMamba-SS.

### F. Recursion Depth and Computational Cost

Fig. {{F:depth}} varies only the number of improvement steps. Quality is flat and non-monotonic over a fourfold range of depth (balanced accuracy 0.9378, 0.9441, 0.9353 and 0.9386 at 21, 42, 63 and 84 core applications), and the reported configuration of 63 is the lowest of the four. The arithmetic, measured by forward hooks, is 2.215, 4.224, 6.234 and 8.245 GFLOPs: a straight line of 95.72 MFLOPs per application, exactly the slope that the cost model (23) predicts, on an intercept of 0.204 GFLOPs for everything outside the core.

![figure](../figures/results/fig_depth_cost.png)

*Fig. {{F:depth}}. (a) Test quality against the number of core applications (12-epoch budget). (b) Measured arithmetic against the cost model (23). (c) Parameters against arithmetic for the hyperspectral configurations.*

On the same input, MedMamba-SS-TRM holds 6.2× fewer parameters than MedMamba-SS (446,409 against 2,773,007) and needs 22.3× more arithmetic (6.203 against 0.278 GFLOPs; Fig. {{F:depth}}(c)); on the six-class RGB configuration the ratios are 61.4× and 16.4×. Because the forward pass always runs all three segments, inference pays this cost too (14.7 ms per patch at batch size 1). Parameter count in a weight-shared recursive model therefore measures storage, not compute, and the depth that this compute buys is not measurable on this task: 21 applications match 63 at a third of the arithmetic.

### G. Error Analysis

Fig. {{F:errors}} shows the seed-42 test confusion matrices of the two MedMamba-SS-TRM arms. Both classify IDC almost perfectly (100.0 % and 99.0 % recall). The errors are healthy tissue predicted as DCIS: 14.3 % of healthy patches with 32 bands and 27.5 % with 3 bands. DCIS recall is high in both (97.4 % and 94.0 %), so the difference between the arms is mostly the false-positive rate for DCIS on healthy tissue, which is also where the class-mean spectra of Fig. {{F:dataset}}(a) differ.

![figure](../figures/results/fig_error_analysis.png)

*Fig. {{F:errors}}. (a, b) Row-normalized test confusion matrices at seed 42 with patch counts. (c) Reliability diagram (15 bins with more than 200 patches).*

Per patient, test macro recall averaged over the five seeds is 0.95–1.00 for four of the five test patients in the 32-band arm and 0.77 for the fifth (patient 136), which is equally difficult for both arms and is the only test patient with DCIS. The reliability curves lie above the diagonal: both models are under-confident, which is consistent with EMA weights and focal loss, and which Section VI-H quantifies.

### H. Calibration

Table {{T:calibration}} reports calibration of MedMamba-SS-TRM over the five seeds of each arm and, at seed 42, of the other networks and the two strongest probes of Table {{T:comparison}}.

**TABLE {{T:calibration}}**
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

On the test patients the 32-band model is better calibrated than its 3-band twin at every seed (ECE 0.056 ± 0.007 against 0.094 ± 0.017; Brier 0.097 against 0.134), and on the validation patients as well (ECE 0.059 against 0.070). The reliability diagram (Fig. {{F:errors}}(c)) shows the direction of the error on the test patients: above a confidence of 0.5, accuracy exceeds confidence in every bin, so both models are under-confident, as expected from EMA weights and a focal loss that down-weights confident predictions.

Temperature scaling fitted on each run's validation predictions does not repair this. The fitted temperature is above one (T = 1.53 ± 0.14 for 32 bands, 1.99 ± 0.11 for 3 bands), because the models are over-confident on the validation patients, and applied to the test patients it raises test ECE from 0.056 to 0.095 (32 bands) and from 0.094 to 0.222 (3 bands). In the other direction, a temperature fitted on the test patients is below one (T = 0.44 and 0.34); applied to the validation patients it raises validation ECE from 0.059 to 0.094 (32 bands) and lowers it from 0.070 to 0.057 (3 bands; last column of Table {{T:calibration}}). So for the 32-band model no single temperature fits both held-out sets, whereas for the 3-band model the temperature fitted on the test patients also lowers validation ECE.

Among the other models, MedMamba (0.013), the 64-feature probe (0.025), SpectralFormer (0.035) and the RGB probe (0.035) are better calibrated than the 32-band MedMamba-SS-TRM on the test patients and HybridSN (0.062) is worse; on the validation patients SpectralFormer (0.052) and the RGB probe (0.057) are better and the 64-feature probe (0.071) and HybridSN (0.076) are worse. The transferred temperature fails for HybridSN, SpectralFormer, the 64-feature probe and the RGB probe as well: fitted on the validation patients it is above one for each of them (T = 1.70 to 5.51) and raises test ECE, most for the 64-feature probe (0.025 to 0.373) and HybridSN (0.062 to 0.326); fitted on the test patients it lowers validation ECE for HybridSN, SpectralFormer and the RGB probe and raises it for the 64-feature probe. All six models with validation predictions are over-confident on the validation patients and under-confident on the test patients, so the miscalibration follows the patient split rather than the architecture, and on these data a temperature fitted on five patients does not transfer to five others.

**Selective prediction.** When a classifier abstains on its least confident inputs [@medformer_ur], its value depends on how well confidence ranks errors. Ranking test patches by confidence, the 32-band MedMamba-SS-TRM has the lowest area under the risk–coverage curve (AURC 0.0035, against 0.0038 for MedMamba, 0.0068 for HybridSN, 0.0104 for SpectralFormer and 0.0103 for its RGB twin), with 98.4 % accuracy on the 90 % most confident patches. On the validation patients the order reverses (AURC 0.033, against 0.024 for HybridSN and 0.018 for SpectralFormer), the same patient dependence as in Section VI-A.

### I. Explainability: What the Spectral Pathway Attends To

The band gate (10) weights every band of every patch before pooling, so its values show which wavelengths the pathway emphasizes. Fig. {{F:xai}}(a) averages them over 9,000 class-stratified test patches for the seed-42 hyperspectral model. The gate is not uniform. It rises from about 0.55 below 500 nm to a maximum at 577 nm and stays high to 633 nm, then falls to about 0.54 in the near-infrared block. The maximum is highest for DCIS (0.67 at 577 nm), then IDC (0.65) and healthy tissue (0.61), and its between-class spread is largest in the same 535–633 nm region where the input spectra differ most (Fig. {{F:xai}}(b)): at 535 nm healthy tissue reflects 0.17 above the overall mean and DCIS 0.15 below it, in the range where the haematoxylin and eosin stains absorb. The pathway has therefore learned, without supervision on bands, to weight the part of the spectrum that separates the classes, and to weight it most for the minority class whose separation from healthy tissue drives the modality effect. The gate is a learned weighting rather than a causal attribution, and the model still receives every band.

![figure](../figures/results/fig_xai_bands.png)

*Fig. {{F:xai}}. (a) Band-gate weights $a_c$ of (10), averaged within each patch, for 9,000 class-stratified test patches (3,000 per class). (b) Difference between each class's mean spectrum and the overall mean for the same patches.*

Fig. {{F:tsne}} embeds the pooled answer state, the vector the classifier reads, for the same patches with t-SNE [@tsne]. With 32 bands the three classes form three separate groups, healthy and DCIS adjacent but distinct. With 3 bands IDC stays separate, while healthy and DCIS patches intermix along a shared boundary, which is the confusion that Fig. {{F:errors}} counts (27.5 % of healthy patches predicted DCIS, against 14.3 % with 32 bands).

![figure](../figures/results/fig_tsne.png)

*Fig. {{F:tsne}}. t-SNE of the pooled answer state of MedMamba-SS-TRM for 9,000 class-stratified test patches (seed 42). (a) 32-band model. (b) 3-band model.*

### J. Reconstruction

The auxiliary decoder reconstructs the 32-band input from the core's final answer state. On the validation subset its spectral angle falls from 9.47° after the first epoch to 6.16° at the selected epoch 7 and 5.34° at epoch 20, with RMSE 0.1109 and PSNR 20.3 dB at the selected epoch. Fig. {{F:recon}} shows typical (median spectral angle) and worst reconstructions for each class. Median reconstructions follow the input spectrum closely across both spectral regions, including the absorption minimum near 535 nm, and preserve the spatial pattern of the patch (2-D SSIM 0.66–0.80). The worst cases are informative: the worst IDC patch (20.6°) is a saturated, nearly uniform patch whose spectrum lacks the stain absorption minimum, an atypical input that the decoder maps back toward the typical tissue spectrum. That the answer state supports this reconstruction shows that it retains detailed spectral information rather than only what a three-way decision needs, and the ablation of Section VI-E shows that asking it to do so improves classification.

![figure](../figures/results/fig_reconstruction.png)

*Fig. {{F:recon}}. Spectral reconstruction on validation patches, one median and one worst patch per class (ranked by spectral angle). Left: composites (633/562/463 nm) of input and reconstruction on a common intensity scale, and the per-pixel spectral angle. Right: patch-mean spectra; no line is drawn across the 219 nm gap.*

### K. Skin Lesions

On PAD-UFES-20, 11 × 11 patches cut from whole photographs inherit the photograph's label although most contain no lesion, which is a multiple-instance setting [@mil]. At patch level MedMamba-SS-TRM therefore scores only 25.7 % balanced accuracy; averaging its patch predictions over each image raises this to 35.6 % (MedMamba: 24.8 % → 31.4 %). Under a patch recipe matched to MedMamba in every setting except the architecture, MedMamba-SS-TRM leads on accuracy (30.7 % against 27.6 %), balanced accuracy (25.8 % against 24.8 %) and macro-F1 (0.228 against 0.216) with 8.2× fewer parameters.

Whole 224 × 224 images are where both proposed models can be compared directly (Table {{T:pad}}). MedMamba-SS trains normally here and leads accuracy (53.5 %), specificity, Cohen's κ (0.352) and ROC-AUC (0.798; 0.806 in a repeat run of the same configuration), which locates its failure on hyperspectral patches (Section VI-E) in that configuration rather than in the hierarchy. MedMamba-SS-TRM reaches the highest balanced accuracy (50.2 %), macro precision and macro-F1 (0.421) of the three models, with 61× fewer parameters than MedMamba-SS and 32× fewer than MedMamba-T. The comparison is confounded, and the table lists how: the two proposed models share the patient-disjoint split and test images but not the training set or the class weighting, and the MedMamba-T run uses the reference image-level protocol with a different test set. Placed against the published PAD-UFES-20 comparison of [@medmamba], which uses an image-level split and different test images, the sensitivity of MedMamba-SS-TRM (50.2 %) is higher than all eleven published rows and its macro-F1 (42.1 %) is second only to NesT-Tiny (42.3 %), while its overall accuracy (45.6 %) is the lowest.

**TABLE {{T:pad}}**
**Whole-Image PAD-UFES-20, Test Split**

| | MedMamba-SS-TRM | MedMamba-SS | MedMamba-T |
| --- | ---: | ---: | ---: |
| Parameters | **0.45 M** | 27.4 M | 14.5 M |
| Split | patient-disjoint | patient-disjoint | image-level |
| Training images | 234 (class-undersampled) | 1,626 | 1,378 |
| Test images | 344 | 344 | 691 |
| Loss | focal, weights $\propto n_k^{-1}$ | focal, weights $\propto n_k^{-0.75}$ | cross-entropy |
| Accuracy | 45.6 % | **53.5 %** | 47.2 % |
| Balanced accuracy (macro sensitivity) | **50.2 %** | 39.8 % | 33.9 % |
| Macro precision | **40.3 %** | 37.5 % | 34.5 % |
| Macro specificity | 88.6 % | **89.2 %** | 87.9 % |
| Macro-F1 | **0.421** | 0.385 | 0.335 |
| Cohen's κ | 0.305 | **0.352** | 0.271 |
| Macro ROC-AUC | 0.793 | **0.798** | 0.714 |

*MedMamba-T: a run of the reference implementation [@medmamba_code] under its own image-level protocol.*

### L. Consolidated Discussion

Table {{T:scorecard}} collects the rankings of Sections VI-A to VI-K in one place: where the proposed models lead, and where they do not.

**TABLE {{T:scorecard}}**
**Summary of Rankings (Seed 42 Unless Stated)**

| Criterion | Leader | Runner-up | Section |
| --- | --- | --- | --- |
| Test balanced accuracy, five patients | **MedMamba-SS-TRM** (94.37 %) | MedMamba (91.02 %) | VI-A |
| Test macro-F1, five patients | **MedMamba-SS-TRM** (0.904) | MedMamba (0.872) | VI-A |
| Test selective prediction, AURC (lower is better) | **MedMamba-SS-TRM** (0.0035) | MedMamba (0.0038) | VI-H |
| Pooled balanced accuracy against the base model, ten patients | **MedMamba-SS-TRM** (83.21 %) | MedMamba (81.67 %) | VI-A |
| Parameters identical at 3 and 32 bands | **MedMamba-SS-TRM** (446,409 at both) | — (every baseline changes with $C$) | VI-D |
| Whole-image PAD-UFES-20 balanced accuracy and macro-F1 | **MedMamba-SS-TRM** (50.2 %, 0.421) | MedMamba-SS (39.8 %, 0.385) | VI-K |
| Whole-image PAD-UFES-20 accuracy, Cohen's κ, ROC-AUC | **MedMamba-SS** (53.5 %, 0.352, 0.798) | MedMamba-SS-TRM (45.6 %, 0.305, 0.793) | VI-K |
| Pooled balanced accuracy, all models, ten patients | HybridSN (87.14 %) | RGB colour probe (86.50 %); MedMamba-SS-TRM 83.21 % | VI-A |
| Validation balanced accuracy, five patients | HybridSN (84.94 %) | RGB colour probe (84.90 %); MedMamba-SS-TRM 72.11 % | VI-A |
| Test ECE | MedMamba (0.013) | 64-feature probe (0.025); MedMamba-SS-TRM 0.056 | VI-H |

**What MedMamba-SS delivers.** The spectral pathway makes MedMamba independent of the band count by construction: the same parameters serve 3 and 32 bands, a model retrained at 8 bands keeps 97 % of its macro-F1, and the learned band gate concentrates on the 535–633 nm stain-absorption region where the classes differ. Its wavelength encoding and the reconstruction objective each add measurably to classification. Independence holds for the architecture, not for a trained instance: a model trained on 32 bands does not transfer to fewer without retraining. The conditioned hierarchy trains normally on whole RGB images, where it has the highest accuracy, κ and ROC-AUC of the three models compared, but not on hyperspectral patches under the recipe tuned for the recursive model; that incompatibility, which survives changes of learning rate, clipping threshold and auxiliary loss, is the main open problem for MedMamba-SS.

**What MedMamba-SS-TRM delivers.** The recursive substitution reduces parameters 6.2× on hyperspectral input and 61.4× on RGB images while keeping the spectral pathway and the task interface, and the resulting 0.45 M-parameter model leads its base architecture, MedMamba (3.65 M), on the test patients, on the validation patients' macro-F1 and pooled over all ten patients. It is the most accurate model of all on the test patients, the best-ranked model under selective prediction there, and second among 32-band models on pooled macro-F1.

**What depends on the patients.** Whether hyperspectral input beats RGB input, and whether MedMamba-SS-TRM beats HybridSN or a logistic regression on spectral statistics, depends on which five patients are held out. On the test patients the 32-band model is the best of all models and beats its RGB twin at every seed; on the validation patients it is among the weakest networks and loses to its RGB twin at every seed. Per patient the picture is simpler: 32 bands give the higher recall for seven of ten patients, tie on one, and fail badly on one (patient 304), and each held-out set contains DCIS from a single patient. With five patients per held-out set, the between-patient variation is larger than any architectural effect in this paper, and the pooled ten-patient comparison, in which HybridSN and a six-feature colour probe lead, is the most reliable single ranking available.

**What transfers from TRM, and what does not.** The recursion itself transfers: two carried states, a gradient-free prelude and deep supervision train stably on noisy patch classification. Three of TRM's premises do not. Its halting head, trained against "the current answer is correct", stays at chance on soft three-way tissue labels (Appendix A). Its depth benefit is not measurable here: 21 core applications match 63. And a tiny recursive model is not a cheap one: 63 applications cost 22.3× the arithmetic of the hierarchy they replace, at inference as well as in training, as the cost model (23) predicts for any model built this way.

---

