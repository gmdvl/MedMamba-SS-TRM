# MedMamba-SS and MedMamba-SS-TRM: Band-Count-Agnostic Spectral–Spatial Medical Image Classification with a Weight-Shared Recursive Backbone

**Gabriel M. Del Valle Lopez**
Department of Computer Science

---

***Abstract*—MedMamba classifies medical images with a block that divides its channels between a convolutional branch and a two-dimensional selective scan, but its first layer holds a weight for every input channel, so it reads a spectrum as unordered channels whose number is fixed by the sensor. We extend it in two steps. MedMamba-SS replaces the patch embedding with a spectral pathway in which no parameter depends on the number of bands (a wavelength-keyed tokenizer shared across bands, a spectral selective scan and pooling over bands) and conditions every stage of the hierarchy on its output. MedMamba-SS-TRM replaces the hierarchy with one two-layer core applied recursively, following the Tiny Recursive Model. On hyperspectral breast histology from 45 patients, 32-band and 3-band models have exactly 446,409 parameters, retraining at eight bands keeps 97 % of macro-F1, and the learned band gate concentrates on the 535–633 nm stain-absorption region. MedMamba-SS-TRM is the most accurate model on the five test patients (94.4 % balanced accuracy) and outperforms MedMamba pooled over all ten held-out patients with 8.2× fewer parameters, but HybridSN leads pooled (87.1 % against 83.2 %). The advantage of 32 over 3 bands is patient-dependent too: +4.1 points on the test patients at five of five seeds, −5.8 on the validation patients, driven by one patient, and −1.0 pooled. The recursive model is 6.2× smaller than the hierarchy it replaces but needs 22.3× more arithmetic, as an analytic cost model predicts, and 21 to 84 core applications give the same quality.**

***Index Terms*—Band-count agnosticism, explainability, hyperspectral imaging, MedMamba, medical image classification, recursive networks, state-space models, weight sharing.**

---

## I. Introduction

Distinguishing ductal carcinoma *in situ* (DCIS) from invasive ductal carcinoma (IDC) and from healthy breast tissue governs treatment, and it is made on morphology that varies between laboratories, stains and observers. Hyperspectral imaging (HSI) adds a physically different measurement: each pixel carries a spectrum of tens of narrow bands rather than three integrated colour values, and absorption in stained tissue depends on its molecular composition. A classifier that exploits this must read the spectral axis as an ordered physical quantity, and, because instruments sample different numbers of bands, it should not have to be redesigned for every sensor.

This work builds on two published architectures, and both of its contributions are extensions of them. **MedMamba** [@medmamba] brought selective state-space models to medical image classification. Its SS-Conv-SSM block splits the channels between a convolutional branch for local detail and a two-dimensional selective scan for long-range context at a cost linear in the number of positions, and a four-stage hierarchy of these blocks was evaluated on sixteen datasets covering ten imaging modalities. As published, it is a poor fit for hyperspectral input in two respects: its first layer is a strided convolution with a weight for every input channel, so its parameter count is tied to the sensor, and band order and wavelength are invisible to it. The **Tiny Recursive Model** (TRM) [@trm] showed that a single two-layer network, applied recursively to an answer state and a latent state, can outperform networks with several times its parameters on reasoning puzzles, decoupling effective depth from stored parameters. It has been tested only on small grids with exact answers, and its arithmetic cost has not been reported.

We extend these models in two steps, each of which substitutes one component (Fig. {{F:lineage}}). **MedMamba-SS** (spectral–spatial) replaces MedMamba's convolutional patch embedding with a spectral pathway in which no parameter shape depends on the number of bands, and conditions every stage of the inherited hierarchy on that pathway's output; the SS-Conv-SSM blocks, patch merging and stage layout are kept. **MedMamba-SS-TRM** then replaces the hierarchy with one weight-shared two-layer core applied recursively under TRM's schedule, and keeps the spectral pathway and the task interface unchanged. The first substitution addresses the sensor: a band-count-agnostic front end lets one architecture, with one parameter count, serve 3-band and 32-band input. The second addresses scale: hyperspectral cohorts are small, and if depth can come from reusing weights rather than from storing them, the model can be several times smaller.

```mermaid
flowchart LR
    MM["MedMamba [1]<br/>convolutional patch embedding,<br/>four stages of SS-Conv-SSM blocks,<br/>classifier on the last stage"]
    TRM["Tiny Recursive Model [2]<br/>one two-layer network,<br/>answer and latent states,<br/>deep supervision"]
    SS["MedMamba-SS<br/>spectral pathway replaces the<br/>patch embedding and conditions<br/>every stage of the hierarchy"]
    SSTRM["MedMamba-SS-TRM<br/>one weight-shared core, applied<br/>63 times, replaces the hierarchy;<br/>spectral pathway kept"]
    MM -->|"extension 1:<br/>spectral-spatial front end"| SS
    SS -->|"extension 2:<br/>recursive backbone"| SSTRM
    TRM -->|"recursion schedule, states,<br/>deep supervision, EMA"| SSTRM

    classDef inherit fill:#e8eaed,stroke:#6b7280,color:#111827
    classDef fromtrm fill:#bfdbfe,stroke:#1d4ed8,stroke-width:2px,color:#111827
    classDef newmod fill:#bbf7d0,stroke:#15803d,stroke-width:2px,color:#111827
    class MM inherit
    class TRM fromtrm
    class SS,SSTRM newmod
```

*Fig. {{F:lineage}}. The two base models (grey, blue) and the two proposed models (green). Each proposed model differs from its predecessor by one substitution; Sections IV-D and IV-E give the component-level changes.*

Three questions organize the evaluation. The first is whether hyperspectral input improves classification over RGB input when the model, the training recipe, the patient split and the patch locations are identical and only the input differs. The second is whether a model derived from MedMamba can be made independent of the number of bands, both in its parameters and in its behaviour when the band count changes. The third is whether one recursively applied core can replace a hierarchical backbone, at what computational cost, and with what benefit from recursion depth.

The main contributions of this work are the two models:

1. **MedMamba-SS**, a spectral–spatial extension of MedMamba (Section IV-D). A band-count-agnostic spectral pathway replaces the patch embedding, and FiLM conditioning [@film], per-stage context selection and a context updater feed its output into every stage of the SS-Conv-SSM hierarchy. We prove that no parameter of the pathway, or of any model built on it, depends on the band count (Proposition 1), and verify it exactly on the recursive model, which shares the pathway: its 32-band and 3-band instances have identically 446,409 parameters. The pathway's wavelength encoding improves classification, and its learned band gate concentrates on the stain-absorption region where the tissue classes differ. The hierarchy trains normally on whole-image skin-lesion photographs, where it has the highest accuracy, Cohen's κ and ROC-AUC of the three models compared, but not on hyperspectral patches under the recipe tuned for the recursive model (Section VI-E).

2. **MedMamba-SS-TRM**, which replaces the hierarchy with one weight-shared core applied 63 times per forward pass (Section IV-E) and has 6.2× fewer parameters than MedMamba-SS on the hyperspectral task. It is the most accurate model in our comparison on the five test patients, and it outperforms its base architecture, MedMamba, pooled over all ten held-out patients with 8.2× fewer parameters, although HybridSN is more accurate pooled. An analytic cost model predicts its measured arithmetic to four significant figures and makes explicit that a weight-shared recursive model trades storage for compute.

Three supporting steps make these contributions measurable, and we report them as the means to that end rather than as contributions in their own right: a paired hyperspectral/RGB data preparation that samples both modalities at the same patch coordinates, so that the modality comparison changes a single variable (Section III-B); an evaluation over all ten non-training patients with patient-level bootstrap intervals, needed because each five-patient held-out set contains DCIS from a single patient (Sections V-E and VI-C); and four implementation safeguards without which the spectral pathway fails to train, silently (Appendix A). Explainability and reconstruction analyses (Sections VI-I and VI-J) show what the spectral pathway attends to.

Section II reviews related work and identifies the gaps this paper addresses. Section III describes the datasets, Section IV the base models and the two extensions, Section V the experimental setup and Section VI the results. Section VII concludes.

---

## II. Literature Review

### A. Convolutional and Transformer Classifiers

Residual convolutional networks made deep models trainable and remain strong, data-efficient baselines [@resnet], [@convnext]; Vision Transformers relate all patch tokens through self-attention at quadratic cost [@vit], and Swin restores a hierarchy with shifted windows [@swin]. In both families the first layer treats input channels as an unordered set.

### B. State-Space Models and MedMamba

Mamba made state-space parameters input-dependent, giving a sequence model with near-linear cost [@mamba]. Vision Mamba [@vim] and VMamba [@vmamba] extended it to images; VMamba's two-dimensional selective scan (SS2D) traverses a feature map in four directions. MedMamba [@medmamba] combined SS2D with a convolutional branch in the SS-Conv-SSM block and evaluated it on sixteen medical datasets covering ten modalities, with variants from 15.2 M (MedMamba-T) to 48.1 M (MedMamba-B) parameters. It is the base architecture of both models in this paper.

### C. Recursive and Weight-Shared Networks

Universal Transformers [@ut], ALBERT [@albert] and deep equilibrium models [@deq] reuse one block in place of many, and Adaptive Computation Time [@act] learns when to stop. TRM [@trm] simplifies the Hierarchical Reasoning Model (HRM) [@hrm] to one two-layer network with two carried states and deep supervision, and with 5–7 M parameters it improves on HRM's 27 M on Sudoku-Extreme (87.4 % against 55.0 %), Maze-Hard and ARC-AGI. Its benchmarks are small grids with exact answers, and it reports parameters and accuracy but not arithmetic. It is the source of the recursion in MedMamba-SS-TRM.

### D. Hyperspectral Image Classification

HybridSN combines 3-D and 2-D convolutions over hyperspectral patches [@hybridsn], and SpectralFormer embeds groups of neighbouring bands as Transformer tokens [@spectralformer]. Both come from remote sensing, where evaluation is typically pixel-wise within one scene, and both have parameters whose shape depends on the band count. Medical hyperspectral imaging is a smaller literature [@hsi_review], in which patient-disjoint evaluation is uncommon.

### E. Gaps Identified

1. **No representation of the spectral axis in MedMamba.** Band order is invisible to its first layer, and its parameters scale with the band count, so it cannot move between sensors without redesign.
2. **No sensor-independent spectral classifier.** HybridSN and SpectralFormer are sized for one band grid; independence from the band count is neither claimed nor tested.
3. **No test of recursion outside reasoning puzzles.** Whether TRM's mechanism, its halting head and its depth benefit carry over to noisy medical classification is unknown, and its arithmetic cost has not been reported.
4. **No controlled hyperspectral-versus-RGB comparison** in which the model, recipe, patient split and patch locations are identical and only the input differs.

MedMamba-SS addresses the first two gaps and MedMamba-SS-TRM the third; the paired data preparation of Section III-B makes the fourth measurable.

---

## III. Datasets

### A. HistologyHSI-BC-Recurrence

The HistologyHSI-BC-Recurrence collection [@hsibc_data], [@hsibc_paper] contains hyperspectral captures of stained breast histology slides. We classify three tissue types: healthy, DCIS and IDC. From 644 captures of 45 patients, one preparation pass selects 32 of the 740 bands by importance and cuts 11 × 11 patches under a patient-disjoint split of 35 training, 5 validation and 5 test patients (Table {{T:datasets}}). The selected band centres span 400.5–938.2 nm unevenly: eighteen lie at or below 633.3 nm and, after a 219.0 nm gap, fourteen lie between 852.3 and 938.2 nm. Fig. {{F:dataset}}(a) shows that the class-mean spectra differ most between 535 and 633 nm, where healthy tissue reflects more than DCIS and IDC, and that DCIS and IDC are close throughout. The held-out sets are small in a way that matters for evaluation: in each of them all DCIS patches come from a single patient (patient 197 in validation, patient 136 in test), so one third of balanced accuracy on either set is measured on one patient.

**TABLE {{T:datasets}}**
**Properties of the Two Datasets**

| | HistologyHSI-BC-Recurrence | PAD-UFES-20 |
| --- | --- | --- |
| Modality | Hyperspectral (32 of 740 bands, 400.5–938.2 nm) and paired synthetic RGB | Smartphone RGB photographs |
| Classes | healthy, DCIS, IDC | BCC, ACK, NEV, SEK, SCC, MEL |
| Sample | 11 × 11 patch | 224 × 224 image |
| Patients (train / val / test) | 35 / 5 / 5 | 1,373 in total, patient-disjoint 70/15/15 |
| Samples (train / val / test) | 2,452,086 / 334,516 / 348,894 | 1,626 / 328 / 344 |
| Class counts, train | 722,358 / 122,850 / 1,606,878 | 613 / 508 / 172 / 158 / 136 / 39 |
| Class counts, test | 83,538 / 24,570 / 240,786 | 119 / 114 / 36 / 40 / 28 / 7 |
| Largest : smallest class, train | 13.1 : 1 | 15.7 : 1 |

![figure](../figures/results/fig_dataset.png)

*Fig. {{F:dataset}}. HistologyHSI-BC-Recurrence. (a) Class-mean reflectance over the 32 selected bands, 30,000 training patches; no line is drawn across the 219 nm gap, where no band was selected. (b) One test patch per class as a hyperspectral composite (633, 562 and 463 nm) and as the paired synthetic RGB patch.*

### B. Paired Hyperspectral and RGB Builds

A modality comparison is informative only if the two arms differ in modality alone. The preparation therefore writes two builds from the same captures, paired at the patch level: the RGB image is resized to the cube's spatial size where necessary and sampled at the *same patch coordinates*, and the patient and capture identifiers are identical in both builds. Each capture folder ships two RGB images: a synthetic rendering computed from the cube, aligned with it in all 644 captures, and a frame from a separate wide-field camera whose field of view differs in 228 captures. We use the synthetic rendering throughout, so the two builds differ only in the number of spectral channels.

### C. PAD-UFES-20

PAD-UFES-20 [@pad] contains 2,298 smartphone photographs of skin lesions from 1,373 patients in six classes, with melanoma at 2.3 %. It is the dataset on which MedMamba reports a public comparison, and we use it to test both proposed models on a second modality under a patient-disjoint split (the published MedMamba results use an image-level split, which places photographs of the same lesion in training and test).

---

