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

In the objective (22) the angle is computed in radians on the z-scored input; reported spectral angles, RMSE and PSNR are computed on reflectance, after the normalization is reversed, and angles are given in degrees.

Calibration matters for a classifier whose confidence may be used to abstain on uncertain cases [@selective] and refer them to a pathologist [@medformer_ur]. It is measured by the expected calibration error [@naeini], [@guo_calib] over $M = 15$ confidence bins $B_m$ of the $N_{\text{eval}}$ evaluated patches, together with the Brier score [@brier]:

$$\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N_{\text{eval}}} \big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|. \tag{29}$$

We also report accuracy, macro precision, Cohen's $\kappa$ and macro ROC-AUC, and per-patient macro recall, the mean recall over the classes a patient has.

### C. Implementation Details

All models were trained on one NVIDIA RTX 5060 Ti (16 GB) with PyTorch 2.11 [@pytorch] and CUDA 12.8. MedMamba-SS-TRM uses width 128, two core blocks, $n = 6$, $T = 3$, $N_{\text{sup}} = 3$, AdamW [@adamw] at learning rate $3 \times 10^{-4}$ with weight decay 0.05 (not applied to normalization weights and biases), batch size 256, bf16 precision, gradient clipping at 1.0, and a 587-step linear warmup into cosine decay over 19,560 steps, stepped per optimizer step. Inputs are z-scored with per-band means and standard deviations fitted over the training split, and training uses geometric (flips, 90° rotations, crops) and spectral (noise, scaling, offset) augmentation. Each epoch trains on a fresh random 10.2 % of the training split (250,113 patches, 978 steps) and validates on a fixed patient-stratified 9.86 % of the validation split (32,985 patches); the checkpoint with the best validation macro-F1 is tested on the full test split. The recipe was developed over earlier runs on the same patient split; checkpoints were always selected on validation data, but the test scores of those development runs were recorded and compared alongside their validation scores, so the test patients are independent of checkpoint selection but not demonstrably of recipe development. Headline runs train for 20 epochs; every ablation trains for 12 epochs (11,736 steps) and is compared with a 12-epoch baseline, because the cosine schedule makes a shorter run a different schedule rather than a truncation. Supplementary Table {{ST:config}} lists the full configuration.

Three implementation choices were fixed by dedicated measurements. In a benchmark configuration on RGB skin-lesion patches (740,800 training patches per epoch) with four supervision segments (84 core applications, hence 168 block calls per step), fp32 precision, batch size 150 and reconstruction on, the SS2D mixer, whose selective scan ran sequentially over the 121 positions without a fused kernel, took 66.9 s per step, or 91.8 hours per epoch of that configuration; replacing only the mixer with the convolutional one made the step 39× faster. Without gradient checkpointing in the core, the 0.45 M-parameter model does not fit in 16 GB on hyperspectral input. And evaluating the spectral pathway in chunks of 1,024 positions, with per-chunk checkpointing for $C \geq 16$, cuts its peak memory by 90 % (10.2 GB to 1.0 GB) for a 15 % increase in step time.

Every run passes validity gates before a result is recorded: data integrity and patient-level leakage checks, a representation-sensitivity check (stem sensitivity 0.504 and 0.552 against a floor of 0.05 for the seed-42 hyperspectral and RGB runs), a split-drift check, a check that the reconstruction gradient reaches the core, and recomputation of the selection metric from the saved checkpoint (difference 0.0 in both runs).

### D. Baselines

All baselines are trained on the same hyperspectral build and patient split and scored on the same full test and validation splits.

- **Shallow probes.** Class-balanced logistic regressions [@sklearn] on per-band means (32 features), per-band means and standard deviations (64), and the flattened patch (3,872), fitted on 25,000 training patches under the same normalization.
- **MedMamba** [@medmamba]. The reference implementation with a 32-band stem, patch size 1 and widths (64, 128, 256, 512), depths (1, 1, 2, 1): 3,648,995 parameters, trained for five full epochs (about 12.3 M patch presentations, against 5.0 M for MedMamba-SS-TRM's 20 subsampled epochs) with class-weighted cross-entropy, learning rate $10^{-4}$, weight decay $10^{-4}$, batch size 256 and bf16, on the same z-scored inputs, with the checkpoint chosen by macro-F1 on the full validation split and no EMA. This differs from the other networks, which select on a 9.86 % validation subset: MedMamba's checkpoint (epoch 1) was chosen on the same set on which its validation and pooled scores are reported.
- **HybridSN** [@hybridsn] (569,843 parameters): three 3-D convolutions (8, 16 and 32 filters; spectral kernels 7, 5 and 3; spatial 3 × 3), one 2-D convolution with 64 filters and a 256–128 fully connected head with dropout 0.4, applied without padding to the 32 selected bands of the 11 × 11 patch (the original uses 30 PCA components of 25 × 25 windows). **SpectralFormer** [@spectralformer] (121,428 parameters): the patch-wise variant with cross-layer adaptive fusion and the official defaults (group-wise embedding of each band with its neighbours, width 64, depth 5, four heads). Both are trained with MedMamba-SS-TRM's 20-epoch recipe: the same loss, optimizer, schedule, subsampling, augmentation and selection rule, without EMA or reconstruction. No hyperparameter was tuned for either, whereas the recipe was developed for MedMamba-SS-TRM (Section V-C).
- **MedMamba-SS** (2,773,007 parameters) under MedMamba-SS-TRM's recipe at the 12-epoch budget.

HybridSN and SpectralFormer are included because MedMamba, the base architecture, has no mechanism that reads the spectral axis, so on its own it cannot show whether the spectral pathway improves on established ways of modelling spectra. The two models represent the two main families of spectral–spatial classifiers: HybridSN convolves jointly over bands and space, and SpectralFormer treats groups of neighbouring bands as Transformer tokens, which makes it the closest classic hyperspectral classifier to our band-token pathway. The channel-adaptive models of Section II-E are closer in purpose but are not among the baselines. Both are small (0.57 M and 0.12 M parameters), so they compare with MedMamba-SS-TRM (0.45 M) at similar storage, and neither is band-count agnostic: HybridSN's 2-D convolution and SpectralFormer's position embedding have shapes that depend on the number of bands.

### E. Protocol

The hyperspectral-versus-RGB comparison repeats both arms at five seeds (1, 7, 13, 23, 42); the two arms share one configuration and differ in the input build (Section III-B), and hence in the reconstruction target, and both have 446,409 parameters. HybridSN and SpectralFormer are also trained at the same five seeds, so that they can be compared with MedMamba-SS-TRM seed by seed (Section VI-A); every other configuration, including MedMamba and every ablation, uses seed 42. On this dataset, two pairs of runs with identical configuration and seed differ by 0.04 and 0.06 points of balanced accuracy; this is the nondeterminism floor for a fixed seed here (identical runs on the 344-image PAD-UFES-20 test set differ far more, Section VI-K). Changing the seed moves test balanced accuracy far more (s.d. 1.9 points for the 32-band arm, Section VI-A), and every single-seed comparison below should be read against that spread. Because the validation and test sets each contain only five patients, we report both, and we evaluate every seed's selected checkpoint on the *full* validation split, so that all ten non-training patients can be analysed. The validation patients also choose each network's checkpoint, so validation and pooled scores of the networks are not independent of model selection and are optimistically biased; the probes involve no selection. Patient-level 95 % intervals resample the held-out patients with replacement (2,000 draws) and report the 2.5th and 97.5th percentiles of the mean over seeds of the balanced-accuracy difference. A draw that loses a class entirely, which happens whenever the single DCIS patient is left out, is discarded (1,372 of 2,000 draws kept on the test patients, 1,355 on the validation patients and 1,808 pooled); with five patients per set these intervals are coarse and condition on the DCIS patient being present.

---

## VI. Experimental Results

All results are on HistologyHSI-BC-Recurrence unless stated. Because the validation and test sets each contain five patients, every comparison is reported on the test patients, on the validation patients (full split, scored at the checkpoint selected on the validation subset, so optimistic for the networks), and pooled over all ten non-training patients. The section first compares MedMamba-SS-TRM with existing models and follows its training (Sections VI-A and VI-B). It then examines the properties that the two substitutions were designed to provide: independence from the band count (Section VI-C), the contribution of individual components, including the hierarchical backbone of MedMamba-SS (Section VI-D), and the cost and depth of the recursion (Section VI-E). Because the band count no longer fixes the architecture, one model can be trained on either input, and Section VI-F uses this to compare hyperspectral with RGB input. Sections VI-G to VI-J analyse the errors, calibration, band weighting and reconstruction of MedMamba-SS-TRM, and Section VI-K evaluates both proposed models on skin lesions.

### A. Comparison with Existing Models

Table {{T:comparison}} and Fig. {{F:comparison}} compare MedMamba-SS-TRM and MedMamba-SS with three shallow probes, a colour probe on the paired RGB build, MedMamba, HybridSN and SpectralFormer.

**TABLE {{T:comparison}}**
**Comparison on HistologyHSI-BC-Recurrence (Seed 42 Unless Stated). Test: 348,894 Patches from 5 Patients; Validation: 334,516 Patches from 5 Patients; Pooled: All 10**

| Model | Params | Test Acc. (%) | Test BA (%) | Test F1 | Test κ | Val. BA (%) | Val. F1 | Pooled BA (%) | Pooled F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority class (always IDC) | — | 69.01 | 33.33 | 0.272 | 0.000 | 33.33 | 0.258 | 33.33 | 0.266 |
| Logistic regression, band means | 32 feat. | 82.13 | 82.68 | 0.726 | 0.656 | 68.39 | 0.619 | 75.81 | 0.671 |
| Logistic regression, band means + s.d. | 64 feat. | 93.07 | 90.28 | 0.843 | 0.854 | 77.99 | 0.710 | 84.31 | 0.771 |
| Logistic regression, raw patch | 3,872 feat. | 79.66 | 62.02 | 0.623 | 0.551 | 56.92 | 0.555 | 59.86 | 0.587 |
| Logistic regression, RGB means + s.d. (RGB build) | 6 feat. | 93.05 | 87.58 | 0.830 | 0.853 | 84.90 | 0.783 | 86.50 | 0.807 |
| MedMamba [@medmamba] | 3.649 M | 95.14 | 91.02 | 0.872 | 0.896 | 72.30 | 0.689 | 81.67 | 0.774 |
| HybridSN [@hybridsn] | 0.570 M | 93.56 | 88.85 | 0.841 | 0.863 | 84.94 | 0.785 | 87.14 | 0.813 |
| HybridSN, five seeds (mean ± s.d.) | 0.570 M | 93.45 ± 0.48 | 89.40 ± 0.47 | 0.841 ± 0.008 | 0.861 ± 0.010 | 84.37 ± 1.04 | 0.785 ± 0.017 | 87.12 ± 0.66 | 0.813 ± 0.013 |
| SpectralFormer [@spectralformer] | 0.121 M | 92.01 | 82.45 | 0.804 | 0.826 | 80.73 | 0.766 | 81.78 | 0.784 |
| SpectralFormer, five seeds (mean ± s.d.) | 0.121 M | 92.28 ± 1.23 | 83.02 ± 3.98 | 0.817 ± 0.025 | 0.829 ± 0.030 | 79.29 ± 1.29 | 0.761 ± 0.004 | 81.30 ± 2.33 | 0.788 ± 0.013 |
| MedMamba-SS, 12-epoch budget | 2.773 M | 69.06 | 33.42 | 0.274 | 0.002 | 33.30 | 0.259 | 33.36 | 0.267 |
| MedMamba-SS, full-channel variant, 12-epoch budget | 3.513 M | 70.01 | 40.85 | 0.397 | 0.131 | 30.20 | 0.269 | 35.64 | 0.329 |
| **MedMamba-SS-TRM** | **0.446 M** | 96.38 | 94.37 | 0.904 | 0.923 | 72.11 | 0.694 | 83.21 | 0.795 |
| MedMamba-SS-TRM, five seeds (mean ± s.d.) | 0.446 M | 94.66 ± 2.60 | 92.78 ± 1.87 | 0.875 ± 0.040 | 0.888 ± 0.051 | 76.18 ± 4.59 | 0.720 ± 0.023 | 84.49 ± 1.45 | 0.795 ± 0.012 |

*BA: balanced accuracy (26); F1: macro-F1 (27). Rows marked "five seeds" are means over seeds 1, 7, 13, 23 and 42; all other rows are single runs at seed 42. Validation and pooled scores of the networks are optimistic, because the validation patients select their checkpoints (Section V-E). MedMamba selected its checkpoint on the full validation split, the others on a subset (Section V-D). MedMamba's pooled scores are computed exactly from its per-class validation counts, since its validation predictions were not stored. Section VI-H compares the calibration of these models.*

![figure](../figures/results/fig_comparison.png)

*Fig. {{F:comparison}}. Balanced accuracy on the test patients, the validation patients and all ten held-out patients pooled, at seed 42, for the models of Table {{T:comparison}} that train, and for the 3-band (RGB) arm of MedMamba-SS-TRM. The majority-class row, the raw-patch probe and both MedMamba-SS variants are omitted; five-seed results are in Table {{T:comparison}}.*

On the test patients MedMamba-SS-TRM has the highest balanced accuracy (92.78 ± 1.87 % over five seeds, 94.37 % at seed 42), ahead of MedMamba (91.02 %), the strongest probe, a logistic regression on the 64 per-band means and standard deviations (90.28 %), HybridSN and SpectralFormer. The ranking holds at four of five seeds; at seed 1 (89.61 %) MedMamba and the 64-feature probe are ahead. MedMamba and the probes are single runs; HybridSN and SpectralFormer are also compared seed by seed below. On the other test metrics the five-seed mean does not lead MedMamba: macro-F1 is level (0.875 against 0.872), and accuracy (94.66 % against 95.14 %) and Cohen's κ (0.888 against 0.896) are lower. On the validation patients the order reverses. HybridSN (84.94 %) and a six-feature RGB colour probe (84.90 %) lead, and MedMamba-SS-TRM (76.18 ± 4.59 % over five seeds) and MedMamba (72.30 %) trail the 64-feature probe (77.99 %). Pooled over all ten patients, HybridSN (87.14 %) and the RGB probe (86.50 %) lead; MedMamba-SS-TRM reaches 84.49 ± 1.45 % over five seeds, level with the 64-feature probe (84.31 %) and ahead of MedMamba (81.67 %). MedMamba was trained under a different recipe (Section V-D), and differences of this size between single runs lie within the seed-to-seed spread. The hierarchical MedMamba-SS stays at chance on every held-out set; its test macro-F1 (0.274) is that of predicting IDC everywhere (0.272) (Section VI-D).

Seed 42, at which the single-run baselines and every single-seed analysis below were run, is MedMamba-SS-TRM's best seed on the test patients and its worst on the validation patients. Across the five seeds, test and validation balanced accuracy are strongly anti-correlated (Pearson r = −0.95; n = 5, descriptive only): seeds trade performance between the two patient sets, and the seed-42 analyses show the model at its most favourable on the test patients.

**Seed-paired baselines.** Because a single seed can favour either model, HybridSN and SpectralFormer were also trained at the four other seeds of the headline model (1, 7, 13 and 23) under the same recipe, so that each can be compared with MedMamba-SS-TRM seed by seed (Table {{T:comparison}}, rows marked five seeds). On the test patients MedMamba-SS-TRM is ahead of SpectralFormer at every seed, by 9.77 ± 4.25 points of balanced accuracy, and ahead of HybridSN at four of five seeds, by 3.38 ± 2.14 points; at seed 1 the two are level (89.61 % against 89.65 %). On the validation patients HybridSN is ahead at every seed, by 8.19 ± 4.44 points, and SpectralFormer at four of five, by 3.11 ± 5.64 points. Pooled over all ten held-out patients, HybridSN is ahead of MedMamba-SS-TRM at every seed, by 2.62 ± 1.34 points (87.12 ± 0.66 % against 84.49 ± 1.45 %), whereas MedMamba-SS-TRM is ahead of SpectralFormer at four of five seeds, by 3.19 ± 2.94 points. The seed-paired comparison therefore confirms the pattern of the single runs and makes the pooled ranking firmer: the test patients favour the recursive model, the validation patients the baselines, and over all ten patients HybridSN leads at every seed. Both baselines also vary less between seeds than MedMamba-SS-TRM on the validation patients (s.d. 1.04 and 1.29 against 4.59 points).

The ranking of models depends on which five patients are held out more than on the architecture, partly for a structural reason, since each held-out set contains DCIS from a single patient: among the networks that train, the gap between test and validation balanced accuracy ranges from 1.7 points (SpectralFormer) to 22.3 points (MedMamba-SS-TRM at seed 42), larger than most differences between networks. Per-patch summary statistics are also a strong baseline on this dataset. A linear model on 64 band statistics, or on six colour statistics, is competitive with every network, whereas a linear model on the 3,872 raw values of the flattened patch is weak (62.02 %). A linear model cannot exploit translation-invariant spatial structure, so this does not show that the patches carry no spatial information; it shows that most of the linearly accessible information lies in per-patch statistics. Finally, the two networks that do best on validation, HybridSN and SpectralFormer, select their first and third epoch, whereas MedMamba-SS-TRM selects its seventh. Because every network's checkpoint is the one its validation data preferred, this pattern cannot separate a property of the models from the selection rule.

### B. Learning Curves

Supplementary Fig. {{SF:curves}} shows the training dynamics of the seed-42 pair of MedMamba-SS-TRM models. Training classification loss falls steadily for both inputs (32 bands: 0.128 to 0.034). Validation loss for the 32-band model reaches its minimum at epoch 9 (0.101) and then rises to 0.169 at epoch 20, and validation macro-F1 peaks at epoch 7 and declines slowly; the model keeps fitting the 35 training patients after it stops improving on the five validation patients, and the checkpoint rule selects the epoch before that divergence. The 3-band model peaks earlier (epoch 4). The spectral-angle reconstruction loss, in radians on z-scored spectra, falls steadily on both splits for the 32-band model (validation 1.27 to 0.55) and is lower for the 3-band model; the two losses are computed over different numbers of channels and are not directly comparable.

### C. Band-Count Agnosticism

**Structure.** The same MedMamba-SS-TRM on the same dataset and label space has exactly 446,409 parameters at 32 bands and at 3 bands, as Proposition 1 predicts for the classification path; its parameter memory is 1.786 MB in both cases. What depends on the band count is activation size (peak inference memory 161.5 MB against 52.6 MB, measured with the reconstruction decoder attached) and, marginally, arithmetic (6.203 against 6.052 GFLOPs), because 97 % of the arithmetic is the recursion over a fixed 121-token grid. The optional reconstruction decoder is the one band-dependent module: its output layer has one channel per band (575,849 parameters with the decoder at 32 bands, 559,116 at 3), so all parameter and arithmetic counts in this paper exclude it.

**Behaviour.** Fig. {{F:bands}} evaluates the seed-42 model on subsets of its bands, uniformly decimated in index, without retraining (*zero-shot*), and trains fresh models at 16 and 8 bands under the 12-epoch budget (*retrained*). With the encoding as defined in (5)–(6), zero-shot transfer fails: at every reduced band count balanced accuracy falls to 0.33–0.41 and IDC F1 to exactly 0.000. Most of this failure at 16 bands comes from the encoding. Index decimation keeps both end bands, so the normalization (5) is unchanged, but $\eta = C$ in (6) changes with the band count, so every retained band receives a code that the 32-band model never saw in training. Evaluating the same checkpoint with $\eta$ held at its training value of 32, and nothing else changed, separates the two effects. At 16 bands balanced accuracy rises from 0.3845 to 0.9086 and macro-F1 from 0.1854 to 0.8256 (per-class F1 0.828, 0.693 and 0.956 for healthy, DCIS and IDC); at 8 bands it rises only from 0.3329 to 0.5599, and macro-F1 from 0.1289 to 0.4983. With all 32 bands the fixed encoding reproduces the model's own test result (macro-F1 0.90365 against 0.90363). These are single, post-hoc evaluations of the seed-42 checkpoint on the test patients. Retrained at eight bands, the model reaches 0.9014 balanced accuracy and 0.8684 macro-F1, 96 % and 97 % of the 12-epoch 32-band values (0.9353 and 0.8920) with a quarter of the bands. The loss of 3.4 points of balanced accuracy is about 1.3 times the standard deviation expected for a difference between two single runs (√2 × 1.9 ≈ 2.7 points, taking the 20-epoch seed spread as a proxy for these 12-epoch runs), so it is suggestive but not established. At 16 bands balanced accuracy is higher (0.9159) and macro-F1 lower (0.8636) than at eight; these are single runs. The architecture is therefore band-count agnostic by construction and under retraining. A trained model transfers zero-shot to half of its bands once its encoding is held at the training scale, but not to a quarter of them, so a band-count-independent choice of $\eta$ is a candidate design that retraining would have to confirm, and larger reductions need another remedy. Training on random band subsets, as ChannelViT's hierarchical channel sampling does for channels [@channelvit], is one to test. The eight bands were decimated in index from a label-selected set of narrow bands, so the result does not carry over directly to a filter camera with broader bands.

![figure](../figures/results/fig_bands.png)

*Fig. {{F:bands}}. Test performance against the number of bands. Zero-shot, $\eta = C$: the 32-band model evaluated on decimated input with the encoding as defined in (6). Zero-shot, $\eta$ fixed at 32: the same checkpoint with the encoding scale held at its training value (evaluated at 32, 16 and 8 bands). Retrained: a model trained at that band count (12-epoch budget; the 32-band point is the 12-epoch baseline).*

### D. Ablation Study

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

At a single seed, removing the auxiliary reconstruction objective lowers balanced accuracy by 1.3 points and macro-F1 by 0.021, most of it in DCIS (F1 0.769 with the objective, 0.733 without), and replacing the wavelength encoding with an index encoding lowers them by 0.6 points and 0.012. The two index-encoding runs agree with each other to 0.0006, but both effects are smaller than the seed-to-seed standard deviation of test balanced accuracy (1.9 points at 20 epochs, Section VI-A), so they are consistent with a contribution of each component but do not establish one; replication across seeds is needed. If the wavelength encoding does help, one candidate mechanism is the 219 nm gap, which an index encoding treats as a step between neighbours. The recursion depth can be cut to a third with no measurable loss (Section VI-E).

The hierarchical MedMamba-SS stays at chance under the recursive model's recipe on this data: both variants (0.334 and 0.409 balanced accuracy) select their first epoch. Lowering the learning rate to $10^{-4}$ or $3 \times 10^{-5}$, removing the reconstruction decoder and raising the clipping threshold to 5.0 all leave it at chance. Its training accuracy rises (0.735 → 0.906) while validation macro-F1 stays at 0.2583, the value of a constant prediction, and its gradient norm (302 → 1,002) is clipped by two to three orders of magnitude at every step. Rising training accuracy with a constant evaluation output is the signature of a module that behaves differently in evaluation mode, and the convolutional branch $\Phi$ contains batch normalization. We therefore scored each saved hierarchical checkpoint on the same class-stratified sample of 9,000 test patches (3,000 per class, batches in random order) three ways: in evaluation mode, with batch normalization using the statistics of each test batch instead of its running statistics, and with the whole network in training mode. For the standard variant, evaluation mode predicts IDC for 8,994 of the 9,000 patches (balanced accuracy 0.334); with batch statistics the prediction no longer collapses and balanced accuracy rises to 0.490 (0.487 in training mode). Its running statistics are far from those of the test batches: in the first stage's convolutional branch the channel means differ by up to 3.8 running standard deviations and the variances by a factor of up to 4.2. For the full-channel variant, evaluation mode makes no difference (0.405, 0.388 and 0.405). The constant prediction of the standard variant is therefore an artefact of mismatched batch-normalization statistics, but even without it neither hierarchical variant exceeds 0.49 on this sample, far below MedMamba-SS-TRM. Why MedMamba-SS learns so little from 11 × 11 hyperspectral patches under this recipe remains unexplained; these diagnostics were run on one sample and one checkpoint per variant. The same architecture family trains on PAD-UFES-20 (Section VI-K), in a configuration that differs in input size, stage layout, patch size and recipe. On hyperspectral patches, therefore, the recursive substitution is established on parameters, arithmetic and interface, and its quality is compared against MedMamba rather than against MedMamba-SS.

### E. Recursion Depth and Computational Cost

Fig. {{F:depth}} varies only the number of improvement steps, $T = 1$ to 4. Quality is flat and non-monotonic over a fourfold range of depth (balanced accuracy 0.9378, 0.9441, 0.9353 and 0.9386 at 21, 42, 63 and 84 core applications), and the reported configuration of 63 is the lowest of the four; at a single seed these differences are within noise. At 21 applications ($T = 1$) there is no gradient-free prelude, so the prelude, too, contributes nothing measurable here. The arithmetic, counted as in Section IV-F, is 2.183, 4.193, 6.203 and 8.213 GFLOPs: a straight line of 95.72 MFLOPs per application, the slope of the cost model (23), which counts the same operations (Section IV-F), on an intercept of 0.173 GFLOPs for everything outside the core.

![figure](../figures/results/fig_recursion_cost.png)

*Fig. {{F:depth}}. (a) Test quality against the number of core applications (12-epoch budget). (b) Measured arithmetic, without the reconstruction decoder, against the cost model (23). (c) Parameters against arithmetic for the hyperspectral configurations; full-channel is the hierarchical variant of Section IV-D.*

On the same input, MedMamba-SS-TRM holds 6.2× fewer parameters than MedMamba-SS (446,409 against 2,773,007) and needs 22.3× more arithmetic (6.203 against 0.278 GFLOPs; Fig. {{F:depth}}(c)); on whole 224 × 224 RGB images, where MedMamba-SS uses MedMamba-T's layout and MedMamba-SS-TRM keeps $d = 128$, the ratios are 61.4× and 16.4×, figures that reflect those two configurations as much as the substitution. Because the forward pass always runs all three segments, inference pays this cost too. Parameter count in a weight-shared recursive model therefore measures storage, not compute, and the depth that this compute buys is not measurable on this task: 21 applications match 63 at a third of the arithmetic.

### F. Hyperspectral Versus RGB Input

Because the architecture has the same parameters at 3 and at 32 bands (Section VI-C), the same model can be trained on the paired RGB build. The two arms of this comparison use the same model (446,409 parameters), recipe, patient split and patch coordinates, and differ in the input build and therefore in the reconstruction target and the spectral position encoding, wavelength for 32 bands and index for 3 (Section III-B); each is repeated at five seeds. Table {{T:modality}} gives balanced accuracy on each held-out set.

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

On the test patients, balanced accuracy favours 32-band input at five of five seeds, by 4.12 ± 1.66 points (macro-F1 +0.044 ± 0.038, positive at four of five), and the gain lies in healthy tissue and DCIS (mean per-class F1 0.892 against 0.844 and 0.739 against 0.656; IDC 0.993 against 0.994). Per patient (Table {{T:perpatient}}), it comes from the healthy captures of patients 141, 213 and 229 (seed-mean healthy recall 0.90–0.99 against 0.78–0.94) and from the DCIS captures of patient 136 (0.97 against 0.91); on patient 136's healthy captures both arms reach only 0.40. On the validation patients, scored on the full split, the sign reverses: 3-band input is better at five of five seeds, by 5.81 ± 4.78 points. Pooled over all ten patients the difference is −1.04 ± 1.89 points (positive at one of five seeds). Resampling patients (Section V-E), the 95 % interval of the seed-mean difference is [+1.6, +4.8] points on the test patients, [−11.8, −1.4] points on the validation patients and [−6.6, +3.9] points pooled.

Fig. {{F:modality}}(b) locates the reversal. Averaged over seeds, 32-band input gives the higher macro recall for seven of the ten held-out patients, patient 136 is tied at 0.770, and the remaining two are validation patients: patient 197 favours 3 bands narrowly (0.853 against 0.866), and patient 304 favours them by a wide margin, with macro recall 0.566 with 32 bands against 0.773 with 3. Removing patient 304 shrinks the validation difference from −5.81 ± 4.78 to −1.91 ± 5.50 points (32 bands better at two of five seeds), so most of the validation reversal disappears without patient 304, and the remainder lies within the spread over seeds. The same patient reverses the comparison without any network: a logistic regression on per-band means and standard deviations scores 90.28 % on the test patients with 32 bands against 87.58 % with 3, and 77.99 % against 84.90 % on the validation patients, where patient 304 again favours 3 bands (0.672 against 0.836). Patient 65, also in validation, is the hardest held-out patient for both inputs (macro recall 0.495 and 0.481): its IDC captures, the ten smaller captures of Table {{T:splitclass}}, are classified as IDC at a recall of 0.000–0.002 at every seed in both arms. The 3-band model is also the more stable across seeds (test balanced accuracy s.d. 0.3 against 1.9 points). The comparison therefore does not establish that 32-band input is better for this task. It shows that 32 bands help most patients and fail badly on one, and that ten patients, with DCIS present in only one patient per held-out set, are too few to decide between the two inputs. The 32 input bands were also chosen with labels that include held-out patients (Section III-A), which could favour the 32-band arm on either held-out set.

![figure](../figures/results/fig_modality.png)

*Fig. {{F:modality}}. (a) HSI − RGB balanced accuracy per seed on the test patients, the validation patients and all ten pooled. (b) Macro recall of every held-out patient (split, number of patches), averaged over the five seeds, for both inputs.*

### G. Error Analysis

Fig. {{F:errors}} shows the seed-42 test confusion matrices of the two MedMamba-SS-TRM arms. Both classify IDC almost perfectly (100.0 % and 99.0 % recall), and test IDC precision is 1.000 at every seed in both arms, so the healthy-tissue errors are DCIS predictions: 14.3 % of healthy patches with 32 bands and 27.5 % with 3 bands at seed 42.

![figure](../figures/results/fig_error_analysis.png)

*Fig. {{F:errors}}. (a, b) Row-normalized test confusion matrices at seed 42 with patch counts. (c) Reliability diagram over 15 bins; bins with at most 200 patches are not drawn. The ECE values in the legend are for seed 42; Table {{T:calibration}} gives five-seed means.*

**TABLE {{T:perpatient}}**
**Recall per Held-Out Patient and Class, Mean (Minimum–Maximum) Over Five Seeds, for the 32-Band (HSI) and 3-Band (RGB) Arms of MedMamba-SS-TRM**

| Split | Patient | Healthy, HSI | Healthy, RGB | DCIS, HSI | DCIS, RGB | IDC, HSI | IDC, RGB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 136 | 0.40 (0.24–0.50) | 0.40 (0.35–0.46) | 0.97 (0.95–0.99) | 0.91 (0.87–0.94) | 0.93 (0.74–1.00) | 1.00 (0.99–1.00) |
| Test | 141 | 0.90 (0.80–0.94) | 0.78 (0.72–0.84) | — | — | 1.00 (0.99–1.00) | 0.99 (0.97–1.00) |
| Test | 154 | — | — | — | — | 1.00 (0.99–1.00) | 0.99 (0.97–1.00) |
| Test | 213 | 0.99 (0.97–0.99) | 0.94 (0.93–0.96) | — | — | 1.00 (1.00–1.00) | 0.98 (0.97–1.00) |
| Test | 229 | 0.99 (0.98–1.00) | 0.90 (0.89–0.93) | — | — | 1.00 (1.00–1.00) | 0.99 (0.97–1.00) |
| Val. | 65 | 0.99 (0.97–1.00) | 0.96 (0.91–0.98) | — | — | 0.00 (0.00–0.00) | 0.00 (0.00–0.00) |
| Val. | 68 | 0.98 (0.91–1.00) | 0.94 (0.88–0.97) | — | — | 1.00 (0.99–1.00) | 0.99 (0.98–1.00) |
| Val. | 197 | 0.94 (0.87–0.97) | 0.87 (0.84–0.90) | 0.62 (0.50–0.88) | 0.73 (0.65–0.82) | 1.00 (1.00–1.00) | 0.99 (0.98–1.00) |
| Val. | 238 | — | — | — | — | 1.00 (1.00–1.00) | 0.98 (0.96–1.00) |
| Val. | 304 | 0.15 (0.01–0.30) | 0.56 (0.50–0.62) | — | — | 0.98 (0.95–1.00) | 0.98 (0.95–1.00) |

*—: the patient has no patches of that class. Validation recalls are on the full validation split.*

The errors are concentrated in individual patients, at every seed and in both arms (Table {{T:perpatient}}). On the test set, patient 136, the only test patient with DCIS, has healthy recall of 0.24–0.50 across seeds with 32 bands, and at seed 42 its four healthy captures produce 83 % of the 32-band arm's healthy errors (9,877 of 11,937 patches); the other test patients reach healthy recall of 0.80–1.00 with 32 bands. On the validation set, patient 304, who has no DCIS, has healthy recall of 0.01–0.30 with 32 bands against 0.50–0.62 with 3, and produces 84 % of the 32-band arm's validation healthy errors at seed 42. The high DCIS recall on the test set is measured on patient 136 alone; on the validation DCIS patient, 197, it is 0.50–0.88. Two explanations fit these patterns, and neither has been tested. DCIS is learned from five training patients, so the model may associate slide- or patient-level appearance with DCIS; and labels are assigned per capture without a pixel mask, so healthy captures of a patient with DCIS may contain tissue resembling DCIS. The class-mean spectral difference at 535–633 nm (Fig. {{F:dataset}}(a)) may contribute as well, but on its own it does not explain why the errors concentrate in these two patients.

The reliability curves lie above the diagonal: both models are under-confident on the test patients (Section VI-H).

### H. Calibration and Selective Prediction

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

On the test patients the 32-band model is better calibrated than its 3-band twin at every seed (ECE 0.056 ± 0.007 against 0.094 ± 0.017; Brier 0.097 against 0.134), and on the validation patients as well (ECE 0.059 against 0.070). The reliability diagram (Fig. {{F:errors}}(c)) shows the direction of the error on the test patients: above a confidence of 0.5, accuracy exceeds confidence in every bin, so both models are under-confident. One possible contributor is the focal loss, which down-weights confident predictions; we have not isolated it.

Temperature scaling [@guo_calib] fitted on each run's validation predictions does not repair this. The fitted temperature is above one ($\tau$ = 1.53 ± 0.14 for 32 bands, 1.99 ± 0.11 for 3 bands), because the models are over-confident on the validation patients, and applied to the test patients it raises test ECE from 0.056 to 0.095 (32 bands) and from 0.094 to 0.222 (3 bands). In the other direction, a temperature fitted on the test patients is below one ($\tau$ = 0.44 and 0.34); applied to the validation patients it raises validation ECE from 0.059 to 0.094 (32 bands) and lowers it from 0.070 to 0.057 (3 bands; last column of Table {{T:calibration}}). So for the 32-band model no single temperature fits both held-out sets, whereas for the 3-band model the temperature fitted on the test patients also lowers validation ECE.

Among the other models, MedMamba (0.013), the 64-feature probe (0.025), SpectralFormer (0.035) and the RGB probe (0.035) are better calibrated than the 32-band MedMamba-SS-TRM on the test patients and HybridSN (0.062) is worse; on the validation patients SpectralFormer (0.052) and the RGB probe (0.057) are better and the 64-feature probe (0.071) and HybridSN (0.076) are worse. The transferred temperature fails for HybridSN, SpectralFormer, the 64-feature probe and the RGB probe as well: fitted on the validation patients it is above one for each of them ($\tau$ = 1.70 to 5.51) and raises test ECE, most for the 64-feature probe (0.025 to 0.373) and HybridSN (0.062 to 0.326); fitted on the test patients it is below one for each of them ($\tau$ = 0.36 to 0.77) and lowers validation ECE for HybridSN, SpectralFormer and the RGB probe and raises it for the 64-feature probe. All six models with validation predictions are therefore over-confident on the validation patients and under-confident on the test patients. Because the direction is the same for every model, it is consistent with an effect of the patient sets rather than of the architecture, and on these data a temperature fitted on five patients does not transfer to five others.

**Selective prediction.** When a classifier abstains on its least confident inputs [@selective], [@medformer_ur], its value depends on how well confidence ranks errors. Ranking test patches by confidence at seed 42, the 32-band MedMamba-SS-TRM has the lowest area under the risk–coverage curve [@aurc] among the networks, marginally below MedMamba's (AURC 0.0035, against 0.0038 for MedMamba, 0.0068 for HybridSN, 0.0104 for SpectralFormer and 0.0103 for its RGB twin; the probes were not scored), with 98.42 % accuracy on the 90 % most confident patches, against 98.49 % for MedMamba. This ranking is from one seed, MedMamba-SS-TRM's best on the test patients (Section VI-A). On the validation patients the order reverses (AURC 0.033, against 0.024 for HybridSN and 0.018 for SpectralFormer), the same patient dependence as in Section VI-A.

### I. Explainability: What the Spectral Pathway Attends To

The band gate (10) weights every band of every patch before pooling, so its values show which wavelengths the pathway emphasizes. Fig. {{F:xai}}(a) averages them over 9,000 class-stratified test patches for the seed-42 hyperspectral model. The gate is not uniform, but its modulation is moderate. It rises from about 0.55 below 500 nm to a peak at 577 nm and stays high to 633 nm, then falls to about 0.54 in the near-infrared block. The peak is highest for DCIS (0.67 at 577 nm), then IDC (0.65) and healthy tissue (0.61), although the ±1 s.d. bands of the three classes overlap across patches (Fig. {{F:xai}}(a)), and its between-class spread is largest in the same 535–633 nm region where the input spectra differ most (Fig. {{F:xai}}(b)): at 535 nm healthy tissue reflects 0.17 above the overall mean and DCIS 0.15 below it, near the absorption minimum of the class-mean spectra. The class means also differ in the near-infrared block, by about 0.11 between DCIS and healthy tissue, where the gate is low. The gate thus puts its highest weights on the visible bands at 577–633 nm, next to the 535 nm band where the input classes differ most and where the gate is still rising; only four selected bands lie between 496 and 852 nm (Section III-A), so this localization is coarse. Every DCIS test patch comes from patient 136, so the class contrasts of Fig. {{F:xai}} and Supplementary Fig. {{SF:tsne}} are also contrasts between patients. The gate was not given band-level supervision, but the 32 bands it weights were themselves chosen by mutual information with the labels (Section III-A). The gate is a learned weighting, not a causal attribution: it does not show that these bands drive the predictions, the model still receives every band, and the profile comes from one seed.

![figure](../figures/results/fig_xai_bands.png)

*Fig. {{F:xai}}. (a) Band-gate weights $a_c$ of (10), averaged within each patch, for 9,000 class-stratified test patches (3,000 per class). (b) Difference between each class's mean spectrum and the overall mean for the same patches.*

Supplementary Fig. {{SF:tsne}} embeds the pooled answer state, the vector the classifier reads, for the same patches with t-SNE [@tsne]. With 32 bands the three classes form three separate groups, healthy and DCIS adjacent but distinct. With 3 bands IDC stays separate, while healthy and DCIS patches intermix along a shared boundary. This agrees in direction with the higher healthy-to-DCIS confusion of the 3-band arm (Section VI-G), but t-SNE geometry is not a measure of separability.

### J. Reconstruction

The auxiliary decoder reconstructs the 32-band input from the core's final answer state. On the validation subset its spectral angle, computed on reflectance, falls from 9.47° after the first epoch to 6.16° at the selected epoch 7 and 5.34° at epoch 20, with RMSE 0.1109 and PSNR 20.3 dB at the selected epoch (peak value 1; both are means of per-batch values weighted by batch size, so the PSNR exceeds the 19.1 dB implied by the mean RMSE). Fig. {{F:recon}} shows typical (median spectral angle) and worst reconstructions for each class. Median reconstructions follow the input spectrum closely across both spectral regions, including the absorption minimum near 535 nm, and preserve the spatial pattern of the patch (2-D SSIM 0.66–0.80). The worst cases are informative: the worst IDC patch (20.6°) is a saturated, nearly uniform patch whose spectrum lacks the stain absorption minimum, an atypical input that the decoder maps back toward the typical tissue spectrum. The answer state is a 121 × 128 map, larger than the 32 × 11 × 11 input, so its capacity to support this reconstruction is expected; the single-seed ablation of Section VI-D is consistent with the reconstruction objective helping classification, but does not establish it.

![figure](../figures/results/fig_reconstruction.png)

*Fig. {{F:recon}}. Spectral reconstruction on validation patches, one median and one worst patch per class (ranked by spectral angle). Left: composites (633/562/463 nm) of input and reconstruction on a common intensity scale, and the per-pixel spectral angle. Right: patch-mean spectra; no line is drawn across the 219 nm gap.*

### K. Skin Lesions

On PAD-UFES-20, 11 × 11 patches cut from whole photographs inherit the photograph's label although most contain no lesion, which is a multiple-instance setting [@mil]. At patch level, under the recipe tuned for it, MedMamba-SS-TRM scores only 25.7 % balanced accuracy (chance is 16.7 %), which is consistent with this label noise; averaging its patch predictions over each image raises this to 35.6 % (MedMamba: 24.8 % → 31.4 %). In a separate run under a patch recipe matched to MedMamba in every setting except the architecture, MedMamba-SS-TRM is ahead on accuracy (30.7 % against 27.6 %), balanced accuracy (25.8 % against 24.8 %) and macro-F1 (0.228 against 0.216) with 8.2× fewer parameters, but these are single runs a few points above chance, and we do not rank the models on them.

Whole 224 × 224 images are where both proposed models can be compared directly (Table {{T:pad}}). MedMamba-SS trains here, well above the majority-class accuracy of 34.6 %, so its failure on hyperspectral patches (Section VI-D) is not a failure of the hierarchy on every input; it has the highest accuracy (53.5 %), specificity, Cohen's κ (0.352) and ROC-AUC (0.798) of the three runs. MedMamba-SS-TRM has the highest balanced accuracy (50.2 %) and macro-F1 (0.421), with 61× fewer parameters than MedMamba-SS and 32× fewer than MedMamba-T. The reported MedMamba-SS-TRM configuration is one of 33 recursive PAD-UFES-20 runs with different recipes whose test scores were recorded during development, so its scores may be optimistically biased. A repeat of the MedMamba-SS run with the same configuration and seed gives accuracy 54.1 %, balanced accuracy 38.4 %, macro precision 54.8 %, macro-F1 0.377, κ 0.343 and ROC-AUC 0.806: macro precision alone moves by 17 points between two identical runs, so differences of a few points in Table {{T:pad}} are within run-to-run variation on this 344-image test set. The comparison is also confounded, and the table lists how. The two proposed models share the patient-disjoint split and test images but not the training set, recipe or class weighting: MedMamba-SS-TRM was trained on a class-undersampled set of 234 images for 150 epochs at learning rate $10^{-3}$, MedMamba-SS on all 1,626 training images for 200 epochs at $3 \times 10^{-4}$ (batch 32, patch size $p = 8$ and focal loss with $\gamma = 1.5$ for both, no reconstruction). Class undersampling is expected to raise macro sensitivity at the expense of accuracy, which is consistent with this pattern, but the two runs differ in more than the sampling. The MedMamba-T run uses the reference image-level protocol with a different test set, and it reaches 47.2 % accuracy and ROC-AUC 0.714, below the 58.8 % and 0.808 listed for this dataset in the MedMamba repository [@medmamba_code]; our reproduction of the baseline is therefore weaker than the published one.

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

*MedMamba-T: our run of the reference implementation [@medmamba_code] under its own image-level protocol. MedMamba-SS: first of two identical-configuration runs; the repeat is reported in the text. All columns are single runs; bold marks the best of the three in each row, and differences of a few points are within run-to-run variation.*

---

