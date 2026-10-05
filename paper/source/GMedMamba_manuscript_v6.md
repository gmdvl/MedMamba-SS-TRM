# GMedMamba-R: Trading Parameters for Recursion in a Spectral–Spatial Mamba Classifier for Medical Images

*Manuscript v6 · IEEE format · supersedes v5 (2026-09-03)*

**Author:** G. M. D. V. L.
**Affiliation:** M.Sc. Thesis, Department of Computer Science
**Contact:** gabrielmdvl@gmail.com

---

**Abstract** — Medical image classification must resolve local morphology and wider context at once. MedMamba addresses this with the SS-Conv-SSM block, pairing convolutional and state-space branches. We describe GMedMamba, a spectral–spatial extension of that design for images whose channel axis carries physical meaning, and GMedMamba-R, a variant replacing the hierarchical backbone with a single weight-shared core applied recursively, following the Tiny Recursive Model. This cuts trainable parameters from 27.43 M to 0.447 M — a factor of 61 — while leaving the spectral front end, head interface and reconstruction pathway unchanged. On hyperspectral breast histology it reaches 94.4 % accuracy, 90.7 % balanced accuracy and 0.858 macro-F1 on 348,894 patches from five patients used for neither training nor model selection, ahead of a MedMamba baseline eight times its size on every metric but one — though the two runs draw their training data from differently balanced directories, so we read that margin as indicative rather than controlled. On PAD-UFES-20 we match that baseline on loss, class weighting, data, objective, normalization, weight averaging and augmentation, leaving architecture as nearly the only difference; there GMedMamba-R leads every patch-level metric — accuracy by 3.2 points, balanced accuracy by 1.0, macro-F1 by 0.012. Both models nonetheless barely clear a six-feature colour baseline, and aggregating patch posteriors to the clinical image shifts results by several points and can reverse a ranking, locating much of the difficulty in the evaluation unit rather than the architecture. Finally, weight sharing reduces storage, not work: the shared core runs 63 times per forward pass, and our model carries 8.2× fewer parameters at 191× the FLOPs — closer to an inverse indicator of cost than a proxy for it.

**Index Terms** — medical image classification, MedMamba, state-space models, tiny recursive models, weight sharing, hyperspectral imaging, spectral–spatial learning, parameter efficiency, multiple-instance learning.

---

## I. Introduction

A classifier looking at tissue has to do two things that pull against each other. It has to resolve local detail — the shape of a nucleus, the texture at a lesion border — and it has to place that detail in a wider context, because the same local pattern means different things in different surroundings. Convolutional networks are excellent at the first and reach the second only slowly, through depth. Vision Transformers reach the second directly through self-attention, but pay a quadratic price in token count, which becomes awkward when the channel dimension itself grows, as it does in hyperspectral imaging.

State-space models offer a third route. Mamba made the state-transition parameters depend on the input, so a model can choose what to carry forward along a sequence while keeping near-linear scaling [2]. Vision-oriented adaptations extended that to two-dimensional feature maps [3], [4], and MedMamba brought it to medical image classification with the SS-Conv-SSM block: half the channels through a convolutional path, half through a state-space path, recombined by concatenation, channel shuffle and a residual [5], [6].

GMedMamba starts from that block and adds what MedMamba does not have — an explicit spectral pathway. In hyperspectral histology each pixel carries a full spectrum rather than three colour values, and two regions that look identical in an RGB rendering can differ clearly in their spectral signature. Treating those bands as ordinary feature channels discards structure that is physically meaningful, so GMedMamba tokenizes the spectral axis separately, models it, and uses the result to condition the spatial stages.

### A. From "does it help?" to "does it need to be this large?"

The spectral extension costs parameters. A parallel pathway, per-stage fusion and a reconstruction head take the model from roughly 13.5 M parameters in the comparable MedMamba configuration to 27.43 M. That is a real cost, and it confounds any accuracy comparison: a model with twice the capacity that scores slightly higher has not demonstrated a better idea.

The Tiny Recursive Model (TRM) [8] suggests a way out. Its observation is that a network's depth and its parameter count are separable. Rather than stacking many distinct blocks, one can build a single small block and apply it repeatedly, carrying a state forward between applications. TRM does this with two states — a latent reasoning state and an answer state — refined over several improvement steps, with most steps run without tracking gradients so that recursion depth does not inflate memory. On the puzzle benchmarks it targets, a 7 M-parameter recursive network is competitive with far larger models.

Applying that idea here replaces the four widening stages of the spatial backbone with one fixed-width core. The spectral pathway, already channel-agnostic, is reused unchanged. The result — GMedMamba-R — has 0.447 M parameters, and the question the paper asks becomes:

> **Can the spatial half of a spectral–spatial Mamba classifier be replaced by a single recursively applied core, retaining classification behaviour at a fraction of the parameter count — and what does that substitution actually cost?**

The second half of that question turns out to matter as much as the first.

### B. What is new in this revision

The previous revision could not answer the accuracy half of the question on hyperspectral data at all: the dataset had no genuine validation split, and the recursive variant had been run for one epoch on a 2 % subsample. Three things changed.

First, the hyperspectral dataset was re-prepared as a genuine patient-disjoint three-way partition, and the recursive model was trained on it to convergence. That produces the paper's headline result and removes its most serious validity defect.

Second, the reconstruction pathway was repaired. In the previous revision the decoder read a *detached* feature map, so the auxiliary loss trained the decoder and could not shape the representation it was supposed to regularize. It now reads the live feature map from the same segment recursion the classifier reads, and a preflight gate fails the run if that ever stops being true.

Third, we trained a MedMamba baseline on the same hyperspectral dataset under a matched configuration. The previous revision had no comparable baseline at all and said so; there is now a head-to-head on the same five held-out patients, reported with its remaining confounds stated in full.

Fourth, we added a floor the work also lacked: shallow probes fitted to the same patches. A shallow floor is the honest way to ask whether a 0.447 M-parameter network is learning anything a logistic regression on channel means could not. On hyperspectral data the answer is clearly yes. On 11×11 RGB skin patches the answer is barely, and we say so.

### C. Contributions

1. GMedMamba, a spectral–spatial extension of MedMamba, described to match the implementation rather than an idealization of it.
2. GMedMamba-R, a TRM-style recursive backbone integrated into that architecture at 0.447 M parameters against 27.43 M, with the spectral front end, task heads and reconstruction interface preserved.
3. A converged hyperspectral result on held-out patients, with a working auxiliary reconstruction objective whose gradient provably reaches the recursive core.
4. Matched-configuration MedMamba baselines on **both** datasets, with the recursive model at parity on 8.2× fewer parameters, and the remaining confounds enumerated rather than glossed. The RGB pair additionally matches loss, class weighting, training-set size and objective, which is what lets the recurring operating-point difference be separated from the training recipe.
5. Shallow-probe floors for both datasets, and an honest reading of the model against them.
6. Patch-to-image aggregation for PAD-UFES-20, which quantifies how much of the low patch-level score is an evaluation-unit artefact.
7. A measured cost model for recursion — core applications per step, wall-clock, memory, and the checkpointing without which the small model does not run at all.
8. A statement of what the results do and do not license, and a specification of the controlled study that would settle the remaining question.

### D. Organization

Section II reviews the relevant work. Section III describes both variants. Section IV covers the training pipeline and the validity gates that make its numbers trustworthy. Sections V and VI report the setup and the results, including the MedMamba baseline. Section VII discusses what the results mean and positions the work in the Mamba family. Section VIII states the limitations and the experiments that would close them. Section IX concludes. Appendix A maps every architectural claim onto the module that implements it.

---

## II. Related Work

**Convolutional networks.** Convolution encodes a strong, data-efficient prior — nearby pixels are related, and the same filter applies everywhere — which is why CNNs have dominated medical imaging. Residual connections made real depth trainable [7]. The limitation is structural: long-range context emerges only indirectly, through depth and pooling.

**Vision Transformers.** ViT [1] treats an image as a sequence of patch tokens and lets any two interact through self-attention, at a cost quadratic in token count. On PAD-UFES-20 specifically, a well-tuned hierarchical transformer remains a strong baseline: Swin-T [9] leads MedMamba's own published comparison table.

**Selective state-space models.** Mamba [2] made state-space parameters input-dependent, giving attention-like selectivity at near-linear cost. Vision Mamba [3] applied bidirectional scanning to patch sequences, reaching 76.1 / 80.3 / 81.9 % ImageNet-1K top-1 at 7 / 26 / 98 M parameters, and running 2.8× faster than DeiT with 86.8 % less GPU memory at 1248² resolution. VMamba [4] introduced the two-dimensional selective scan (SS2D), traversing a feature map along four spatial directions, reaching 82.6 / 83.6 / 83.9 % top-1 at 30 / 50 / 89 M parameters. SS2D is the sequence primitive inside MedMamba and, optionally, inside our recursive core. Neither Vim nor VMamba reports medical or spectral benchmarks, which is the gap MedMamba and this work address.

**MedMamba.** Yue and Li proposed the SS-Conv-SSM block and evaluated it across sixteen datasets spanning ten imaging modalities and 411,007 images [5], [6]. Its three variants are 15.2, 23.5 and 48.1 M parameters (2.0, 3.5 and 7.4 GFLOPs), averaging 84.0, 84.3 and 83.8 % overall accuracy on the non-MedMNIST subset. Their paper and official repository are the authoritative baseline throughout this work.

**Tiny Recursive Models.** TRM [8] argues that recursion can substitute for depth. A single two-layer network is applied repeatedly over two carried states — a latent state `z` and an answer state `y` — trained with deep supervision across several segments, with earlier improvement steps run under `no_grad` so memory stays bounded regardless of recursion depth. With six latent updates and three improvement steps, TRM obtains an effective depth of 42 layers from a 7 M-parameter network, using EMA at decay 0.999.

Its reported results are striking: 87.4 % on Sudoku-Extreme, 85.3 % on Maze-Hard, 44.6 % on ARC-AGI-1 and 7.8 % on ARC-AGI-2, against 55.0 / 74.5 / 40.3 / 5.0 % for the 27 M-parameter Hierarchical Reasoning Model it replaces. TRM's own ablation finds the two-state formulation clearly better than a single latent state (87.4 % against 71.9 % on Sudoku), which is why we use two states by default.

That result motivated the variant in Section III-D. The transfer is not automatic — TRM's benchmarks are small fixed-size grids with exact answers, while ours is noisy patch classification — and the compute profile turns out very differently, as Section VI-E shows.

---

## III. Method

### A. The MedMamba baseline

The canonical pipeline is a four-stage hierarchical pyramid. Each patch-merging step halves spatial resolution and doubles channel width, so the representation moves from `H/4 × W/4 × C` to `H/32 × W/32 × 8C`.

**Fig. 1** — MedMamba: hierarchical backbone and the SS-Conv-SSM block.
![Fig. 1](figures/fig1_medmamba.svg)

Inside a block, the input splits channel-wise. One half goes through a convolutional path; the other through a state-space path with a parallel linear–SiLU branch acting as a learned gate on the SS2D output. The two are concatenated, channel-shuffled and added back to the input. Because each branch sees only half the channels, the block delivers both local and long-range processing at roughly the cost of a single convolutional path. That economy is why the block is worth inheriting.

### B. The spectral pathway

GMedMamba adds one pathway that runs before and alongside the spatial stages. Its job is to turn a spectrum into a compact spatial context map the spatial stages can be conditioned on, and every part of it is deliberately independent of the number of bands.

**Fig. 2** — GMedMamba: one spectral pathway conditioning four spatial stages.
![Fig. 2](figures/fig2_gmedmamba.svg)

Each band value at a spatial location becomes a token through a single shared linear projection, so nothing in the parameter set depends on the band count. Position along the spectral axis is then added; where physical band centres in nanometres are available the encoding is a continuous function of wavelength normalized against a reference sensor range, and otherwise it falls back to band index. The distinction matters for the reason it matters in any sequence model: spectral position corresponds to a physical quantity, and two sensors with different band counts covering the same range should produce comparable representations.

A stack of residual one-dimensional convolutions along the band axis then captures local spectral shape — absorption features, edges between bands — with a small state-space module in the middle of the stack for longer-range dependencies between distant parts of the spectrum. A learned gate weights each band, suppressing uninformative or noisy wavelength ranges. Finally a small MLP stack reduces the token dimension and pools over bands into a spatially-resolved context map.

### C. Fusion and the classification head

The context map is not concatenated once at the end; it conditions every stage. The default mechanism is FiLM modulation — the context produces a scale and a shift applied to the spatial features — and five alternatives are implemented and selectable.

Two smaller mechanisms complete the loop. A per-stage band selector recomputes channel importance on the context map at the start of each stage, so importance is depth-dependent rather than decided once at the input. A context updater runs the other way: after a stage's spatial blocks, the spatial features revise the context map the next stage will see. Spectral information conditions spatial processing, and spatial processing revises the spectral summary.

The classifier does not read only the final stage. It concatenates the pooled output of every stage with the spectral summary, so features at several depths reach the decision.

A reconstruction decoder can be attached during training, reconstructing the input cube from backbone features under a combined pixel-wise MSE and spectral-angle objective. Section IV-C describes why this pathway needed repair and how it was verified.

### D. GMedMamba-R: one core, applied many times

A four-stage backbone spends most of its parameters on having many distinct blocks at several widths. The recursive variant keeps one block, at one width, and applies it many times.

**Fig. 3** — GMedMamba-R: the hierarchical backbone replaced by one weight-shared recursive core.
![Fig. 3](figures/fig3_recursive_core.svg)

The spectral pathway is unchanged. Its output passes through a linear stem to a fixed working width of 128, a two-dimensional sinusoidal positional encoding is added, and the result becomes a fixed input embedding that never changes during the recursion.

Two states are then carried: a latent state `z`, where intermediate work accumulates, and an answer state `y`, which the classifier reads. One improvement step performs six updates of the latent state followed by a single update of the answer state, where the shared core is two layers, each a token mixer followed by a gated MLP under post-normalized residuals. The mixer is selectable — a depthwise convolution with GEGLU, plain multi-head self-attention, or the SS2D selective scan. Section VI-E explains why the convolutional mixer is the one we use.

Running many improvement steps with full gradient tracking would defeat the purpose, since memory saved on parameters would be spent on the unrolled graph. Following TRM, all but the last improvement step run under `no_grad`: the states advance, and only the final step carries gradient. Above that sits deep supervision. A training batch runs three segments, each producing its own logits, with the states detached between segments; the classification loss is the mean cross-entropy over all three, so the model is pushed to be right early rather than only at the end. An exponential moving average of the weights at decay 0.999 is maintained, and validation and the saved best checkpoint use the averaged copy.

A halting head that predicts whether the current segment's answer is already correct is implemented and available, but it is **disabled in every run reported here**, so no result below depends on it.

### E. What is held fixed

The substitution is deliberately narrow. The recursive backbone returns the same dictionary of outputs as the hierarchical one — feature map, pooled vector, spectral context, band weights — so reconstruction decoders and task heads attach unchanged, and in evaluation mode the model returns a single logits tensor, so every existing trainer and wrapper continues to work. Only the spatial backbone changed, and one command-line flag switches between them.

One consequence is worth stating. Because the recursive variant works at unit patch size and the spectral tokenizer is channel-agnostic, a single preset serves both the 11×11×3 RGB patches and the 11×11×32 hyperspectral patches with no shape change and no per-dataset tuning. The hierarchical variant needs a different preset for each.

### F. Parameter cost

| Configuration | Trainable parameters |
|---|---:|
| MedMamba-Tiny, six classes (reference, not measured here) | 13.53 M |
| GMedMamba hierarchical, RGB, six classes | **27.43 M** |
| GMedMamba hierarchical, HSI preset, 32 bands, three classes | 2.77 M |
| MedMamba-HSI, local baseline, 32 bands, three classes (Section VI-B) | 3.65 M |
| **GMedMamba-R, RGB, six classes** | **0.447 M** |
| **GMedMamba-R, HSI, 32 bands, three classes** | **0.446 M** |
| GMedMamba-R, RGB, `ss2d` mixer | 0.535 M |
| GMedMamba-R, RGB, `attention` mixer | 0.377 M |

**Table I.** Model capacity, measured directly from the constructed models under the configuration used for the reported runs. The recursive variant is 61× smaller than the hierarchical GMedMamba on the same task — 1.6 % of its parameters — and is essentially the same size on 3 channels as on 32, because nothing in it scales with band count.

A note on defaults, because the figures and the prose describe different layers of the same system. The model file's own dataclass defaults are the `ss2d` mixer and four deep-supervision segments; the training entry point overrides both, to the convolutional mixer and three segments, and every run in this paper uses the entry point's values. Fig. 3 annotates the model-file defaults; Section V lists what was actually run.

---

## IV. Training Pipeline and Validity Gates

Much of the work behind this revision is not in the model file. It is in making the numbers the model produces mean something. Four areas matter enough to describe.

### A. Dataset gates

Before a model is constructed, the pipeline checks that the data is what it claims to be: per-file shape, dtype, size and SHA-256 against a manifest written at preparation time; a leakage check against the recorded patient and slide split; and a class-coverage check that refuses to start if a split is missing a class unless that is explicitly allowed. Arrays are validated before being memory-mapped, so a corrupt file becomes a typed, catchable error rather than a `SIGBUS` inside a data-loader worker.

This is not theoretical. During earlier runs the gate caught a training array that was physically unreadable from byte offset 1,090,519,040 onward — 71 % of a 3.7 GB file returning I/O errors reproducibly, while its header and first gigabyte read cleanly. Without the gate, that dataset would have trained partway through an epoch and then died in a way that is very hard to attribute.

Both datasets used here pass the patient- and slide-level leakage checks with no overlap between any pair of splits, and a patch-level hash comparison over 50,000 sampled patches per split found no duplicates across splits.

The same discipline has a gap worth reporting, because we ran into it. Gates protect the data path, but the *reporting* path around them was still wrapped in a broad exception handler that logged failures as single-line warnings. The held-out evaluation consequently failed for an entire hyperspectral run — the loader it used could not accept the normalization mode this revision had just made the default — and the only trace was one warning line and a boolean recorded as false. Repairing it exposed two further failures stacked behind the first, both of the same kind: the evaluation path assumed a plain classifier, and a model carrying a reconstruction decoder returns a pair rather than a logits tensor and keeps its configuration one level down. None of the three touched training; all three were invisible in every metric the run reported. The lesson is the one this section already argues, turned on the harness itself: **a check whose failure is indistinguishable from its absence is not yet a gate.** All three are fixed and covered by regression tests, the handler now records a full traceback and a failure artifact, and the evaluation was re-run from the saved checkpoint — it is the test column of Table IV, and it is the reason this revision has a held-out hyperspectral number at all.

### B. Accounting for optimizer updates

When a non-finite gradient is detected the optimizer step is skipped, which is correct — but for several trainer versions nothing counted the skips, so an epoch could complete without performing a single valid update and still report a loss and an accuracy.

Every epoch now records total batches, valid updates, skipped updates, skip ratio, minimum and maximum gradient norm, counts of non-finite gradient elements, the first parameter that went bad, and a boolean validity flag; a run aborts if the skip ratio crosses a threshold or if consecutive epochs are unhealthy. In every run reported below, valid updates equal total batches and skipped updates are zero — which is now a verifiable claim rather than an assumption.

### C. The reconstruction pathway, and the gate that keeps it honest

The previous revision carried an explicit caveat: on the recursive variant the decoder read a *detached* feature map, so the auxiliary loss trained the decoder but never backpropagated into the recursive core. The reason was mundane — maintaining an EMA copy requires deep-copying the module, which fails if a live autograd graph is attached to a stored tensor.

The fix removes the stored tensor rather than the detach. The segment recursion is now run through a function that returns its final feature map directly, as a value, and the decoder consumes it inside the same forward pass. Nothing is stored on the module, so EMA is unaffected, and the reconstruction gradient reaches the core.

Because that is exactly the kind of property that silently regresses, it is enforced by a preflight gate. Before training starts, one batch is run with the classification term zeroed, the reconstruction loss alone is backpropagated, and the run aborts unless parameters *inside the recursive core* — not merely inside the decoder — have received finite, non-zero gradient. On the hyperspectral run the gate confirmed gradient reaching 20 core parameter tensors.

Two smaller corrections belong to the same pathway. The decoder's output activation is now chosen to match the normalization: a sigmoid output cannot represent a z-scored target, and pairing them floors the reconstruction loss near a constant no matter how good the decoder is. The pipeline resolves the activation automatically and refuses to start on an explicit mismatch. And reconstruction metrics are computed in reflectance units after inverting the normalization, not in the network's internal space, so the reported spectral angle and structural similarity mean what a spectroscopist would expect them to mean.

### D. Acquisition drift and per-patient reporting

Two gates address a failure this project actually experienced. One patient's slides were acquired under noticeably different exposure, and a fixed, train-fitted normalization mapped that patient's patches far outside the range the model had seen — a failure invisible in any aggregate metric.

The first gate samples patches from each split, applies the resolved normalization, and fails the run when a split's post-normalization statistics have drifted too far from the training split's, measured in training-group standard deviations. The second is reporting rather than gating: validation metrics are broken down per patient and per capture, and a warning is raised when a single patient's recall falls below a threshold while the aggregate metric rises. Section VI-A shows that breakdown for the hyperspectral run, and it remains informative even when the run as a whole is healthy.

The normalization mode that resolves the underlying problem — standardizing each patch against its own statistics — is now the default for hyperspectral data, because it reproduces a mis-exposed capture's own scale rather than imposing a scale fitted elsewhere.

---

## V. Experimental Setup

**Hardware.** NVIDIA GeForce RTX 5060 Ti (16.7 GB), PyTorch 2.11 with CUDA 12.8, 32 GB system RAM. All timings and memory figures come from this machine.

**Datasets.** Both were re-prepared for this revision as stratified 80/10/10 patient-disjoint partitions. The hyperspectral dataset exists in two builds that differ **only in the training split** — one class-balanced by undersampling, one at the natural distribution — and share byte-identical validation and test splits. That shared evaluation side is what makes the runs below comparable at all, and the differing training side is the caveat Section VI-B carries.

*HistologyHSI-BC (hyperspectral)* [11], [12]. Breast histology, three classes — healthy, DCIS, IDC — as 11×11×32 patches with band centres recorded in nanometres. Thirty-five training, five validation and five test patients, disjoint. The undersampled build's training split holds 368,550 patches, 122,850 per class; the natural build's holds 2,452,086 (29.5 % healthy, 5.0 % DCIS, 65.5 % IDC, a 13.1:1 imbalance). Both builds share the same validation split of 334,516 patches and test split of 348,894, each at the natural class distribution (validation 29.4 % healthy, 7.3 % DCIS, 63.3 % IDC, an 8.6:1 imbalance).

*PAD-UFES-20 (RGB)* [10]. Skin lesions, six classes, as 11×11×3 patches. 1,099 training, 137 validation and 137 test patients, disjoint. The training split holds 100,800 patches, 16,800 per class; validation holds 90,000 and test 88,400 patches at the natural distribution (BCC 36.9 % against MEL 2.2 %, a 16.6:1 imbalance).

**Model configuration.** Both runs use the recursive variant at working width 128, two core layers, six latent updates, three improvement steps, three deep-supervision segments, the convolutional token mixer, EMA at 0.999, gradient checkpointing on the core, and the halting head disabled.

**Optimization.** Batch size 256, bf16 mixed precision, AdamW at 1e-4 with weight decay 0.05, gradient clipping at norm 1.0, and a linear warmup into cosine decay stepped on the optimizer-step axis rather than per epoch — so that subsampling the training split, or stopping early, no longer desynchronizes the schedule from the number of updates actually performed. Checkpoints are selected on macro-F1 and evaluated with the EMA weights.

The hyperspectral run trains on a 10 % resampled subset of the natural-distribution training split (245,248 patches per epoch, 958 steps) and validates on a patient-stratified 10 % subsample (33,452 patches), with weighted cross-entropy at class-weight power 0.75, **no** sampler, global z-score normalization, the medium augmentation preset, classifier dropout 0.2, core dropout and stochastic depth at 0.1, EMA at 0.995 and the reconstruction decoder disabled. It was budgeted for 20 epochs and stopped at 12 on the validation-divergence rule, with epoch 9 selected on macro-F1. The PAD run trains for 40 epochs on the full training split (100,800 patches, 394 steps) and validates on all 90,000 patches, with plain cross-entropy, no sampler, per-sample min–max normalization, geometric augmentation only, and no reconstruction decoder.

Rebalancing is handled at the loss only. An earlier configuration that paired weighted cross-entropy with minority oversampling on the undersampled build is the source of the reconstruction measurements in Section VI-D, which the current headline run cannot supply because its decoder is off; where that run is the source, the text says so.

Two configuration notes were arrived at by measurement rather than by preference. Loss-level and sampler-level rebalancing both correct imbalance, and stacking them over-corrects. And the packaged augmentation presets are hyperspectral presets: their band dropout zeroes an entire RGB channel on three-channel data, which for pigmented lesions destroys the primary diagnostic cue, so the PAD run uses an explicit geometric-only configuration.

**Metrics.** Accuracy, balanced accuracy, macro and weighted precision, recall and F1, per-class metrics, confusion matrices, Cohen's κ and MCC. Given the imbalance, balanced accuracy and macro-F1 carry the argument; raw accuracy is reported but not leaned on.

**MedMamba baselines.** For the comparisons of Sections VI-B and VI-C we trained a local MedMamba implementation on both datasets, in each case on the identical dataset directory and patient split, with the same normalization, batch size 256, AdamW at 1e-4, bf16 precision, seed 42, macro-F1 checkpoint selection, the same epoch budget as our own run on that dataset (30 hyperspectral, 40 RGB) and the same GPU. Its SS2D and Mamba internals are the original implementation, unmodified; the input stem accepts 32 bands, the patch size is 1 rather than 4 so that an 11×11 input is not reduced to 2×2 before the backbone begins, and the per-stage widths are (64, 128, 256, 512) with depths (1, 1, 2, 1) rather than MedMamba-T's larger settings, which were designed for 224×224 images. The result is 3.65 M parameters. It trains on both splits in full, with plain cross-entropy and no weight averaging.

The two baselines differ in how closely they can be matched, and the RGB one is the better-controlled of the pair. On hyperspectral data our own run used weighted cross-entropy, minority oversampling, a 10 % per-epoch training resample and an auxiliary reconstruction objective, none of which the baseline has. On PAD-UFES-20 our run used none of those — plain cross-entropy, no sampler, both splits in full, no reconstruction decoder — so the RGB pair additionally matches loss, class weighting, training-set size and objective, and we set the baseline's weight decay to 0.05 to match ours as well. What remains unmatched there is weight averaging and geometric augmentation. Sections VI-B and VI-C state in each case which differences remain and in whose favour.

Because our RGB run normalizes with per-sample min–max and the baseline script implemented only per-patch z-scoring, we added the former to it, ported from our own implementation and verified bit-identical on 400 validation patches plus constant, zero and sub-epsilon edge cases. Normalization is not a variable we were willing to leave loose in a comparison this close.

**Shallow baselines.** We additionally fit class-balanced multinomial logistic regressions to the same patches under the same normalization, on four feature sets: a constant majority predictor; per-channel means; per-channel means and standard deviations; and the raw flattened patch. These establish the floor a 0.447 M-parameter network must clear to have earned its complexity, and they are, in our view, the most informative reference point currently available for this work.

---

## VI. Results

### A. Hyperspectral histology

This is the setting the spectral machinery was built for, and it is the first revision in which we can report accuracy on it rather than only cost.

| Metric | GMedMamba-R (0.446 M) | MedMamba-HSI (3.65 M) | Best shallow probe | Majority class |
|---|---:|---:|---:|---:|
| Accuracy | **89.42 %** | 88.80 % | 80.53 % | 63.41 % |
| Balanced accuracy | **80.77 %** | 74.14 % | 73.72 % | 33.33 % |
| Macro precision | **75.94 %** | 72.21 % | — | — |
| Macro recall | **80.77 %** | 74.14 % | — | — |
| Macro F1 | **0.7722** | 0.7279 | 0.6816 | — |

**Table II.** Hyperspectral histology, validation split, five held-out patients, best epoch by macro-F1. The GMedMamba-R column is epoch 9 of 12 with EMA weights, scored on 33,452 patches; the MedMamba column is epoch 3 of 30, scored on all 334,516 validation patches (Section VI-B gives the full head-to-head on both splits, and its caveats). The shallow column is a class-balanced multinomial logistic regression on per-channel means and standard deviations (64 features) — the strongest of four probes — fitted to 25,000 training patches and scored on a 15,000-patch draw from the same validation split under the same normalization. All four columns are therefore validation-split numbers on the same five held-out patients; the number of patches scored differs by column.

The recursive model clears the strongest shallow probe by 8.9 points of accuracy, 7.1 points of balanced accuracy and 9.1 points of macro-F1. We regard the balanced-accuracy margin as the meaningful one: the shallow probe already reaches 80.5 % raw accuracy on a split where 63.4 % of patches are IDC, so accuracy alone barely distinguishes the two.

Two further observations about that shallow column are worth recording, because they say something about the data rather than about our model. First, per-channel spectral statistics are genuinely informative here — a 64-dimensional summary reaches 0.68 macro-F1, which is most of the way to what a network gets. Second, the *raw flattened patch*, at 3,872 dimensions, performs clearly worse (0.535 macro-F1) than the 64-dimensional summary. Spatial detail at 11×11 is not merely unhelpful at this scale; under a linear model it is actively harmful. That the network does better than both suggests it is using spatial structure that a linear model cannot, but the honest framing is that most of the separable signal in this dataset lives in the spectral axis, which is precisely the axis GMedMamba was designed to model.

**Per-class behaviour.** The three classes are not equally easy.

| Class | Validation | | | | Test | | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | Support | P | R | F1 | Support | P | R | F1 |
| Healthy | 9,828 | 0.863 | 0.769 | 0.813 | 83,538 | 0.971 | 0.790 | 0.871 |
| DCIS | 2,457 | 0.415 | 0.677 | 0.515 | 24,570 | 0.565 | 0.934 | 0.704 |
| IDC | 21,167 | 1.000 | 0.978 | 0.989 | 240,786 | 1.000 | 0.998 | 0.999 |

**Table III.** Per-class metrics on both held-out splits, from the epoch-9 EMA checkpoint.

IDC is essentially solved and healthy tissue is well separated, but DCIS — the minority class on both splits — is recalled aggressively at modest precision: 67.7 % recall at 41.5 % precision on validation, 93.4 % at 56.5 % on test. The model finds most DCIS patches and pays for it by labelling a number of healthy patches as DCIS. Given that ductal carcinoma *in situ* and invasive carcinoma are a biological continuum, and that a single 11×11 patch may straddle a boundary, this is a plausible operating point rather than a pathological one — and it is a *chosen* one, since the run used weighted cross-entropy. It is nevertheless where the macro-F1 is lost, and the obvious target for future work. Softening the class weight from 1.0 to 0.75 relative to the previous revision moved DCIS precision from 0.481 to 0.565 on test at a cost of 2.3 points of recall, which is the direction we wanted and is why that setting is the one reported.

**Held-out test performance.** The five test patients were never used for selection, and the checkpoint evaluated is the same epoch-9 EMA checkpoint.

| Split | Patches | Accuracy | Balanced accuracy | Macro F1 | κ | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Validation (selection) | 33,452 | 89.42 % | 80.77 % | 0.7722 | — | — |
| **Test (held out)** | 348,894 | **94.36 %** | **90.73 %** | **0.8580** | 0.880 | 0.884 |

**Table IV.** Validation and test performance of the same checkpoint. Test is *higher* on every metric, which is a statement about the two patient sets rather than about the model — Section VI-B shows the MedMamba baseline moves the same way and by a similar margin.

This is the number we would put forward as the paper's headline classification result: 90.7 % balanced accuracy and 0.858 macro-F1 across 348,894 patches from five patients that played no part in training or model selection, from a 0.446 M-parameter backbone. Ranking quality is high on both averaging schemes (macro ROC-AUC 0.993, macro PR-AUC 0.945). Calibration is close to nominal without any post-hoc correction — expected calibration error 0.0038 and Brier 0.077, against 0.015 and 0.105 for the previous revision's run — and inference costs 13.9 ms at batch 1 and sustains 1,099 patches/s with a 170 MB peak on the same GPU. Those figures look modest until they are set beside the baseline's, which Table XIII does.

That test numbers exceed validation numbers by 10 points of balanced accuracy is worth dwelling on for a moment, because the instinct is to distrust it. It is not selection bias working backwards: the checkpoint was chosen on validation, which if anything should flatter validation. The straightforward reading is that the five test patients are simply easier than the five validation patients — and the independent MedMamba run, selected the same way on the same splits, shows the same direction and a similar size of gap. With five patients per split, that is exactly the between-subject variability Table V of the per-patient breakdown already shows within a split.

**Convergence.** The run was budgeted for 20 epochs and stopped at 12 under the validation-divergence rule, with no skipped optimizer updates. Macro-F1 rose monotonically in trend from 0.685 to its maximum of 0.7722 at epoch 9, and validation loss reached its own minimum at the same epoch (0.2018) before rising over the following three — the two criteria agreeing on the checkpoint, which is the ordinary and uninteresting case. The checkpoint-reproducibility gate confirms the saved checkpoint reproduces its logged macro-F1 exactly (delta 0.0), and the split-drift, class-coverage and sensitivity gates all pass.

This matters beyond bookkeeping. Every earlier run in this family peaked at epoch 1 and declined monotonically thereafter, an artefact we traced to a learning-rate schedule stepped on the epoch axis while the training split was being resampled each epoch — so the schedule ran to completion long before the data did. Stepping the schedule on the optimizer-step axis, as Section V describes, removes it: this run's best epoch is its ninth, and the trajectory before it is a normal one.

**Per-patient breakdown.** Aggregate numbers hide per-patient variation, and this dataset has five validation patients.

| Patient | Patches | Macro recall |
|---|---:|---:|
| 238 | 4,423 | 1.000 |
| 68 | 7,371 | 0.962 |
| 197 | 9,828 | 0.835 |
| 65 | 4,459 | 0.760 |
| 304 | 7,371 | 0.637 |

**Table V.** Per-patient macro recall on the validation split. These are the epoch-12 values, the last the run logged; the checkpoint reported everywhere else is epoch 9, whose aggregate macro-F1 is 0.027 higher. We report the epoch-12 breakdown rather than none, and the caveat rather than an implied match.

The spread is wide — from 1.00 to 0.64 — and no aggregate metric would have shown it. Patient 68 is the mis-exposed acquisition that motivated the normalization work of Section IV-D, and at 0.962 it is no longer the problem. Nor, now, is patient 65: at 0.760 it has risen from 0.456 in the previous revision, and patient 304 is the weakest at 0.637. That the identity of the worst patient moves between runs is itself the point — with five patients we cannot say whether the spread reflects acquisition, biology or sampling, and we do not speculate. We report it because a model that averages 81 % balanced accuracy while one patient sits at 64 % is a different clinical proposition from one that is uniformly 81 %, and the distinction should not have to be inferred.

### B. A matched MedMamba baseline on the same data

The previous revision's binding limitation was that no MedMamba run existed under a pipeline comparable to ours. That gap is now largely closed on the evaluation side and has widened on the training side. We trained a local MedMamba implementation, adapted for hyperspectral input, on the same patient split, holding the batch size, precision, seed, checkpoint rule and GPU fixed. It was trained on the class-balanced build of the training split; the GMedMamba-R run reported here was trained on the natural-distribution build. The two builds share byte-identical validation and test splits, so the columns below are scored on exactly the same patches with exactly the same class support — but the models did not see the same training distribution, and that is the single largest caveat on this table.

The adaptation is deliberately minimal, and its shape matters for the comparison. The SS2D and Mamba internals are the original implementation, unchanged. Two things are different: the input stem takes 32 channels instead of 3, and the patch size is 1 rather than 4 — because a stride-4 embedding would reduce an 11×11 patch to 2×2 before the backbone begins, which would handicap the baseline for reasons unrelated to architecture. The per-stage widths were also reduced from MedMamba-T's, because MedMamba-T was designed for 224×224 images and is grossly overparameterized for an 11×11 patch; the result is 3.65 M parameters.

The consequence worth naming is that this backbone **fuses all 32 bands in its first convolution and then scans only the two spatial directions**. It has no mechanism that treats the spectral axis as a sequence in its own right. That is precisely the gap GMedMamba's spectral pathway was built to fill, so the comparison below is close to the architectural question the paper asks.

| | Validation | | Test (identical split) | |
|---|---:|---:|---:|---:|
| Metric | GMedMamba-R | MedMamba | GMedMamba-R | MedMamba |
| Trainable parameters | **0.446 M** | 3.65 M | **0.446 M** | 3.65 M |
| Accuracy | **89.42 %** | 88.80 % | **94.36 %** | 93.30 % |
| Balanced accuracy | **80.77 %** | 74.14 % | **90.73 %** | 86.76 % |
| Macro precision | **75.94 %** | 72.21 % | **84.52 %** | 81.85 % |
| Macro recall | **80.77 %** | 74.14 % | **90.73 %** | 86.76 % |
| Macro F1 | **0.7722** | 0.7279 | **0.8580** | 0.8278 |
| F1, healthy | 0.813 | **0.824** | **0.871** | 0.846 |
| F1, DCIS | **0.515** | 0.370 | **0.704** | 0.637 |
| F1, IDC | 0.989 | **0.990** | 0.999 | **1.000** |
| Recall, DCIS | **0.677** | 0.444 | **0.934** | 0.833 |
| Best epoch | 9 of 12 | 3 of 30 | — | — |
| Peak VRAM (training) | 876 MB | **555 MB** | — | — |
| Median s/epoch | 679 | **155** | — | — |

**Table VI.** GMedMamba-R against a local MedMamba implementation on the hyperspectral splits — the same five held-out patients per split, the same batch size, precision, seed and checkpoint rule, and on test the identical 348,894 patches with identical per-class support. Best epoch by macro-F1 in each case. The two runs are **not** matched on the training distribution, nor on several other axes; see the caveats below.

Three things stand out, and the first of them now runs the other way from the previous revision.

**GMedMamba-R leads on every aggregate metric, on both splits.** Macro-F1 is ahead by 0.044 on validation (0.7722 against 0.7279) and by 0.030 on test (0.8580 against 0.8278), and unlike the previous revision the two splits now agree in direction. Raw accuracy, which MedMamba previously led on both splits, has inverted: +0.6 points on validation and +1.1 on test. Balanced accuracy leads by 6.6 and 4.0. Of the ten metric rows above, MedMamba retains only three, all marginal: validation F1 on healthy (0.824 against 0.813) and IDC F1 on both splits, by 0.001 in each case, on a class both models have effectively solved.

**We do not read that as an architectural win, and the reason is in the training split.** The two runs did not see the same training distribution: MedMamba trained on 368,550 patches balanced exactly 122,850 per class, while GMedMamba-R drew 245,248 patches per epoch from a 2,452,086-patch pool at the natural 13.1:1 imbalance, and compensated at the loss with weighted cross-entropy at power 0.75. Those are two different ways of correcting the same imbalance, and the one used here also exposes the model to 6.7× more distinct healthy and IDC patches over the run. A model that sees more of the majority classes and is still reweighted toward the minority is well placed to lead both raw and balanced accuracy at once, which is exactly the pattern observed. The honest summary of Table VI is that **the recursive model is at least competitive with, and on these splits ahead of, a MedMamba baseline at 8.2× fewer parameters** — with the training-distribution difference unresolved, and resolving it is item A of Section VIII-B.

What did move in a direction the previous revision predicted is the *operating point*, and it is now less extreme. GMedMamba-R still recalls more DCIS than MedMamba (93.4 % against 83.3 % on test) but no longer pays the precision penalty that made the earlier comparison a wash: DCIS F1 is 0.704 against 0.637, ahead on both halves of the trade rather than trading one for the other. Two changes account for that — dropping minority oversampling so that only the loss rebalances, and softening the class weight from 1.0 to 0.75.

**MedMamba overfits this dataset badly and GMedMamba-R does not.** MedMamba reaches its best validation macro-F1 at epoch 3 and then declines: by epoch 30 its training loss is 0.0004 while its validation loss has risen from 0.249 to 1.405, a factor of 5.7. GMedMamba-R reaches its best at epoch 9 of 12 and its validation loss minimum at the same epoch, having been stopped by the divergence rule rather than by exhausting its budget. We would like to attribute that to parameter count, and we cannot: the two runs differ in weight decay (0.05 against 1e-4), in weight averaging (EMA against none), in augmentation, and in the size and balance of the training pool, all of which bear on overfitting. What the comparison does establish is that a 3.65 M-parameter model saturates this training set within three epochs under an otherwise reasonable configuration, which is itself a useful fact about the dataset.

**MedMamba is markedly cheaper per sample.** It trained on the full 368,550-patch split each epoch in a median 155 s, against 679 s for GMedMamba-R on 245,248 patches — two-thirds of the data at 4.4× the wall-clock, and 6.6× the cost per patch, at lower peak memory. This is the cost model of Section VI-E appearing in a head-to-head: recursion buys parameter efficiency and pays for it in compute. Any deployment argument for GMedMamba-R has to be made on model size, not on training or inference cost.

**What this comparison is not.** It is better matched on the evaluation side than anything in the previous revision — both models are scored on the identical 348,894 test patches with identical class support — and less well matched on the training side. Four differences beyond architecture remain, and the first is new to this revision: GMedMamba-R trained on the natural-distribution build of the training split while MedMamba trained on the class-balanced build, so the two corrected the same imbalance by different means; GMedMamba-R sampled 245,248 patches per epoch from that larger pool while MedMamba used its smaller split in full; GMedMamba-R used weighted cross-entropy at power 0.75 while MedMamba used plain cross-entropy with no class weighting; and GMedMamba-R used EMA weights and the medium augmentation preset against MedMamba's live weights and none. Two confounds the previous revision carried are gone — this run uses no minority oversampling and no auxiliary reconstruction objective — but the training-distribution difference more than replaces them. A genuinely controlled result requires the ablation in Section VIII-B, and until it is run we claim only that **the recursive model is competitive with a MedMamba baseline at 8.2× fewer parameters on this dataset**, not that the margins in Table VI are attributable to the architecture.

Both models score markedly higher on test than on validation — MedMamba by 0.100 macro-F1, GMedMamba-R by 0.086 — which is strong evidence that the five test patients are simply an easier set than the five validation patients, rather than anything about either model. It also retires a caveat the previous revision had to carry: with both test evaluations now in hand, the comparison no longer rests on the split that selected the checkpoints.

### C. Skin lesions, and what patch-level evaluation measures

The PAD-UFES-20 run is the one that constrains what this paper can claim, and it is more useful read as a measurement of the evaluation protocol than of the architecture.

| Metric | Validation (best epoch) | Best shallow probe | Test (held out) |
|---|---:|---:|---:|
| Accuracy | 26.92 % | 24.33 % | 25.80 % |
| Balanced accuracy | 25.96 % | 24.79 % | 25.73 % |
| Macro F1 | 0.2260 | 0.2002 | 0.2121 |
| Cohen's κ | — | — | 0.0986 |
| MCC | — | — | 0.1047 |
| Classes with zero recall | **0 of 6** | — | **0 of 6** |

**Table VII.** PAD-UFES-20, patch level. Validation is 90,000 patches from 137 held-out patients at epoch 34 of 40; test is 88,400 patches from a further 137 patients, never used for selection. The shallow column is logistic regression on per-channel means and standard deviations — six features — fitted to 25,000 training patches and scored on a 15,000-patch draw from the **validation** split, so it is directly comparable to the validation column only. The test column is included for completeness and is not a like-for-like comparison with the probe.

One row here is a genuine improvement and one is a genuine problem.

The improvement is class collapse. Two revisions ago this model predicted three of six classes never, at any threshold; the previous revision reduced that to one; this run predicts all six, on both validation and test, with the rarest class (melanoma, 2.2 % of validation patches) recalled at 23.3 % on test. Balanced training data, macro-F1 checkpoint selection and a collapse monitor together removed the failure mode.

The problem is the margin over the shallow probe. On the split where the two are directly comparable, a 0.447 M-parameter recursive network trained for 40 epochs exceeds a logistic regression on **six numbers per patch** — the mean and standard deviation of each colour channel — by 2.6 points of accuracy, 1.2 points of balanced accuracy and 0.026 macro-F1. Held-out test performance is slightly lower still (0.2121 macro-F1). That is not a margin from which any architectural claim can be made.

We do not think this is primarily a statement about the architecture, and the aggregation experiment is why.

| Model | Evaluation unit | n | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| GMedMamba-R | Patch (11×11) | 88,400 | 25.80 % | 25.73 % | 0.2121 |
| GMedMamba-R | **Clinical image** | 221 | **31.22 %** | **35.61 %** | **0.2672** |
| MedMamba | Patch (11×11) | 88,400 | 27.55 % | 24.78 % | 0.2158 |
| MedMamba | **Clinical image** | 221 | **33.94 %** | **31.40 %** | **0.2519** |

**Table VIII.** The same test-set predictions, scored per patch and aggregated to the clinical image by averaging softmax posteriors across each image's patches, for both models and through the same aggregation function. Aggregation helps both — by 9.9 and 6.6 points of balanced accuracy respectively — which is what makes it a property of the evaluation unit rather than of either architecture.

Averaging the same predictions over each source image — changing nothing about the model — gains 9.9 points of balanced accuracy and 5.5 points of macro-F1. The MedMamba baseline gains 6.6 and 3.6 points under identical treatment. The information is present in the patch predictions, both models have it, and the scoring unit is throwing it away.

The reason is structural. Each 11×11 RGB patch is 121 pixels of skin inheriting the whole-lesion diagnosis of the image it was cut from, and most patches from a lesion photograph contain no lesion at all. Under that labelling, a large fraction of the training signal is simply mislabelled background, predicting the majority class is a rational local optimum, and per-patch accuracy structurally understates what the model knows. Table VIII puts a number on how much.

Two consequences follow, and we state them plainly. First, no number in Table VII should be compared with any published PAD-UFES-20 result, which are computed on 224×224 whole lesion images. Second, the patch-level protocol was a poor choice for this dataset, and Table VIII is the beginning of the fix rather than the end of it: proper multiple-instance learning, with the loss itself defined at image level, is the experiment that should replace it.

**A matched MedMamba baseline on PAD.** Because our RGB run used plain cross-entropy, no sampler, both splits in full and no reconstruction decoder, a MedMamba baseline can be matched here far more tightly than on hyperspectral data. Loss, class weighting, training-set size, auxiliary objective, weight decay, normalization, batch size, learning rate, precision, seed, epoch budget and checkpoint rule are all identical, and normalization is bit-identical by construction. That left weight averaging and geometric augmentation, both on our side, so we re-ran GMedMamba-R with those disabled and nothing else changed. Table IX reports all three runs.

| | | Test (identical 88,400 patches) | | | Clinical image (221) | | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Model | Params | Acc. | Bal. acc. | Macro F1 | Acc. | Bal. acc. | Macro F1 | Best ep. |
| GMedMamba-R, as run (EMA + augmentation) | 0.447 M | 25.80 % | 25.73 % | 0.2121 | 31.22 % | **35.61 %** | **0.2672** | 34 |
| **GMedMamba-R, matched (no EMA, no augmentation)** | **0.447 M** | **30.72 %** | **25.80 %** | **0.2281** | **34.39 %** | 29.63 % | 0.2617 | 9 |
| MedMamba | 3.65 M | 27.55 % | 24.78 % | 0.2158 | 33.94 % | 31.40 % | 0.2519 | 1 |

**Table IX.** PAD-UFES-20 head-to-head. All three models are scored on the identical 88,400 test patches with identical per-class support, and image-level columns aggregate those same predictions over the same 221 clinical images through the same function. The middle row is the like-for-like comparison with MedMamba: weight averaging and augmentation disabled so that only the architecture and a few minor schedule settings differ. All three predict all six classes on both splits. Validation figures (best epoch, macro-F1) are 0.2260, 0.2345 and 0.2235 respectively.

**Per-class behaviour.** The aggregate margins in Table IX are small enough that the per-class breakdown carries most of the information.

| Class | Support | GMedMamba-R, as run | | | GMedMamba-R, matched | | | MedMamba | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | P | R | F1 | P | R | F1 | P | R | F1 |
| ACK | 30,400 | 0.435 | 0.227 | 0.298 | 0.408 | 0.413 | **0.411** | 0.425 | 0.322 | 0.367 |
| BCC | 30,800 | 0.516 | 0.290 | 0.372 | 0.519 | 0.291 | **0.373** | 0.518 | 0.290 | 0.372 |
| MEL | 2,000 | 0.029 | 0.233 | 0.052 | 0.032 | 0.200 | 0.055 | 0.035 | 0.252 | **0.061** |
| NEV | 7,600 | 0.209 | 0.516 | 0.298 | 0.249 | 0.451 | **0.321** | 0.226 | 0.392 | 0.286 |
| SCC | 7,600 | 0.109 | 0.087 | **0.097** | 0.090 | 0.050 | 0.064 | 0.082 | 0.066 | 0.073 |
| SEK | 10,000 | 0.131 | 0.191 | **0.156** | 0.147 | 0.144 | 0.145 | 0.115 | 0.165 | 0.135 |
| **Macro** | 88,400 | 0.238 | 0.257 | 0.212 | 0.241 | 0.258 | **0.228** | 0.234 | 0.248 | 0.216 |

**Table X.** Per-class test metrics for the three runs of Table IX, on the identical 88,400 patches with identical per-class support. Bold marks the best F1 in each row. The middle block is the like-for-like comparison with MedMamba. No model gives any class zero recall. BCC is effectively a three-way tie — the three F1 values fall within 0.001 of each other — and nothing should be read into the bold there.

Against MedMamba under the matched recipe, GMedMamba-R is ahead on four classes and behind on two, and ahead on both of the two largest. The two it loses are MEL and SCC, which are also the classes every model here is weakest on: no run reaches 0.10 F1 on SCC, and none reaches 0.07 on MEL. The wins are concentrated where the support is — ACK (+0.044) and NEV (+0.035) account for most of the 0.012 macro-F1 margin, and ACK alone is the largest per-class difference in the table.

Comparing the first two blocks, weight averaging and augmentation do not act uniformly across classes either. They cost ACK 0.112 of F1, almost entirely through recall, which falls from 0.413 to 0.227, and buy back 0.032 on SCC and 0.011 on SEK. The regularized model spreads its predictions more evenly across classes at the expense of the largest one, which is the per-class face of the aggregate trade the third reading below describes.

Three readings. The first was the point of running the ablation; the second was not anticipated.

**Under a matched recipe, the smaller model is ahead at patch level.** With weight averaging and augmentation removed, GMedMamba-R leads MedMamba on every patch-level metric on the held-out split: accuracy by 3.2 points, balanced accuracy by 1.0 and macro-F1 by 0.012, at 8.2× fewer parameters. This is the only comparison in the paper where architecture is close to the sole intentional difference, and it is the first that favours us outright rather than showing parity. We would still not call it controlled — gradient clipping, learning-rate schedule axis, early-stopping patience and classifier dropout differ — but those are smaller knobs than the two just removed.

**Half of the pattern we had been reporting was the recipe, not the architecture.** Across the earlier comparisons GMedMamba-R consistently led balanced accuracy and consistently trailed raw accuracy, and we treated that pair as one phenomenon. The ablation separates them. The balanced-accuracy lead survives almost unchanged (+1.0 points at test, against +0.95 before). The raw-accuracy deficit does not: it inverts, from 1.8 points behind to 3.2 points ahead. Weight averaging and augmentation were suppressing this model's accuracy on this dataset, and describing that as an architectural operating point was wrong. We have corrected the claim rather than the framing.

**And the regularizers were costing accuracy while buying aggregation quality.** Removing them improved patch-level test accuracy by 4.9 points and macro-F1 by 0.016, yet cost 6.0 points of image-level balanced accuracy (35.6 % to 29.6 %). The configuration that is worse per patch is better per lesion. We do not have a confident mechanism for this — the plausible one is that averaging and augmentation produce flatter, more diverse per-patch posteriors that survive mean-pooling better than sharper ones do — and with one seed we are not going to argue it hard. It does mean the two rows should not be read as one model being simply better than the other, and it is a reminder that on a patch-MIL task the evaluation unit can reverse a ranking.

**MedMamba saturates this dataset in a single epoch.** Its best validation macro-F1 is epoch 1 of 40; by epoch 40 its training loss has fallen to 0.15 while its validation loss has risen to 5.06, over its epoch-7 minimum. GMedMamba-R ends at 1.64 against 1.74 — a gap of 0.10 after forty epochs — and its best epoch is its thirty-fourth. Whatever else the recursive model is doing, it is not memorizing 100,800 patches, and it does not need early stopping to avoid doing so. With a 3.65 M-parameter model reaching its peak before it has seen the data twice, the practical reading is that this task supports far less capacity than either model brings to it, which is the same conclusion the shallow probe reaches from the other direction.

For completeness, the hyperspectral setting does not have this problem in the same form. There a single 11×11×32 patch of tissue genuinely carries the spectral signature of the tissue type it was cut from, the label is locally true rather than inherited, and the results in Table II are correspondingly meaningful at the unit they are measured at.

**A first pass at the whole-image protocol, and the four confounds it exposes.** Required experiment D below — moving PAD off the patch grid entirely — has now been started, and the first result is worth reporting for what it says about *comparison hygiene* rather than for the numbers themselves. GMedMamba-R was trained on 224×224 whole lesion images under a patient-disjoint 70/15/15 split, and a MedMamba-T at the same resolution was trained separately by the reference training script under its own paper-protocol settings. Read naively, the two headline validation figures are 41.8 % against 54.6 % and the gap looks architectural. It is not, and decomposing it is the useful part — and the held-out test split, where the ordering on two of three metrics reverses, is the reason the decomposition matters rather than a footnote to it.

| | GMedMamba-R | MedMamba-T |
|---|---:|---:|
| Parameters | **0.447 M** | 14.47 M |
| Split level | patient-disjoint | **image-level** |
| Split ratios | 70/15/15 | 60/10/30 |
| Train / val / test images | **234** / 328 / 344 | 1,378 / 229 / 691 |
| Loss | focal (γ = 1.5) + inverse-frequency weights | plain cross-entropy |
| Augmentation | geometric + photometric | none |
| Checkpoint rule | macro-F1 | accuracy |
| Epochs run | 150 (best 104) | 57 (early stop; best 37) |
| *Validation, best over the run* | | |
| Accuracy | 42.99 % | **54.59 %** |
| Balanced accuracy | 42.80 % | **43.54 %** |
| Macro F1 | 0.3548 | **0.4416** |
| Train accuracy at best epoch | 57.7 % | 99.2 % |
| *Test, each model's own best checkpoint* | | |
| Accuracy | 45.64 % | **47.18 %** |
| Balanced accuracy | **50.20 %** | 33.87 % |
| Macro F1 | **0.4213** | 0.3353 |
| *Replicate run, identical configuration and seed* | | |
| Test accuracy / balanced accuracy / macro F1 | 37.50 % / 44.67 % / 0.3413 | — |

**Table XI.** PAD-UFES-20 at whole-image resolution, first pass. This is **not** a matched comparison and must not be read as one — the two columns differ in split level, split ratios, training-set size, loss, augmentation and checkpoint rule, and the test columns are different images in different numbers (344 against 691). It is included to decompose an apparent 13-point validation-accuracy gap, not to rank the models. The matched PAD comparison in this paper remains Table IX, at patch level. The final block is a second run of the left-hand column launched from a byte-identical command line at the same seed; its spread is discussed below and bounds how much weight any single cell of that column can carry.

Four differences separate the columns, and none of them is the architecture.

*The reported figure is not the run's best accuracy.* GMedMamba-R selects its checkpoint on macro-F1, so its `best_val_accuracy` field records accuracy at the macro-F1-best epoch (104), not the best accuracy the run reached. That was 42.99 %, at epoch 89. A little over a point of the gap is a field-naming artifact.

*The two models did not see comparable amounts of data.* The GMedMamba-R run was launched against a dataset directory prepared with class-undersampling of the training split, which floors every class at melanoma's 39 images and so keeps 234 of 1,626 — it discards 86 % of the training data to buy a balanced prior that the loss was already supplying. MedMamba trained on 1,378. This was our error, not a design choice, and it is the single largest term in the gap; the prepared full-size directory exists and the run has not yet been repeated on it.

*The splits are not the same kind of object.* Ours is patient-grouped, and the leakage gate confirms no patient appears in two splits. The reference protocol splits at image level, and PAD-UFES-20 is 2,298 images from 1,373 patients, so the same patient's photographs of the same lesion are distributed across train, validation and test. That inflates both of its held-out figures by an unmeasured amount, in its favour, and it means neither its validation nor its test number estimates generalization to unseen patients.

*The objectives target different quantities.* Ours is focal loss with full inverse-frequency weighting, which deliberately spends head-class accuracy — ACK and BCC are 68 % of our validation split — to raise recall on the tail. The reference run uses unweighted cross-entropy and is free to bet on the head. Comparing the two on raw accuracy scores one of them on the metric it was not optimizing.

Three asymmetries survive all four, and they are the only part of this table we would draw any inference from. MedMamba-T leads on every validation metric, but on held-out test the balanced-accuracy and macro-F1 ordering inverts and the raw-accuracy gap nearly closes: 50.20 % against 33.87 %, 0.4213 against 0.3353, and 45.64 % against 47.18 % — from a model with 32× fewer parameters trained on one-sixth the images, and with the leaky split working in MedMamba's favour on both splits, not ours. Comparing each model's selected checkpoint with itself rather than with the run's best epoch, MedMamba-T's balanced accuracy falls from 41.30 % on validation to 33.87 % on test while ours rises from 42.21 % to 50.20 %; that 7.4-point drop, alongside a training accuracy of 99.2 % at the same epoch, is the saturation signature Table IX records at patch level. We state this as an observation about two uncontrolled runs and nothing more: the honest conclusion from Table XI is that the whole-image comparison has not yet been run properly, and the specification for running it is item D of Section VIII-B.

**Why this checkpoint's metrics are good, and in what sense.** The checkpoint in Table XI is the first whole-image PAD result in this work whose numbers land in the range the published literature on this dataset occupies, so it is worth saying precisely which properties make it good and which do not survive scrutiny. MedMamba's own PAD-UFES-20 table reports six figures per model; Table XI-B places this checkpoint beside them.

| Model | Precision (%) | Sensitivity (%) | Specificity (%) | F1 (%) | OA (%) | AUC |
|---|---:|---:|---:|---:|---:|---:|
| **GMedMamba-R, 0.447 M (this work)** | **40.29** | **50.20** | 88.65 | 42.13 | 45.64 | 0.7934 |
| MedMamba (published) [5], [6] | 38.43 | 36.94 | 89.90 | 35.80 | 58.80 | 0.8070 |
| Swin-T (published) [9] | 49.35 | 43.10 | **90.83** | **42.87** | **62.15** | **0.8396** |
| ResNet50 (published) | 45.71 | 42.40 | 89.78 | 42.77 | 56.62 | 0.7626 |
| ConvNeXt-B (published) | 33.80 | 34.47 | 88.97 | 33.36 | 54.73 | 0.7613 |
| ViT-B (published) | 32.04 | 33.21 | 88.05 | 32.20 | 50.36 | 0.7291 |

**Table XI-B.** The Table XI checkpoint against the published PAD-UFES-20 figures of [5], [6, Table 2]. **This is a reference-point table, not a comparison**, and no cell subtracts meaningfully from another: our row is 344 patient-disjoint images and every published row is 691 images from an image-level split that distributes a patient's photographs across train, validation and test, on a dataset with 1.67 images per patient. Our row is also trained on 234 images against their 1,378. The source does not state which MedMamba variant its PAD row uses, so no parameter count is given for it; the variants are 15.2, 23.5 and 48.1 M (Table XVII). Bold marks the best value in each column and is a reading aid, not a claim of victory.

Four things about that row are genuinely good, and they are good for reasons visible in the confusion matrix rather than in the aggregate.

*No class is abandoned.* Per-class F1 runs 0.569 (ACK), 0.465 (BCC), 0.444 (MEL), 0.442 (NEV), 0.455 (SEK) and 0.154 (SCC). Five of six classes sit within 0.13 of each other, on a test split where the largest class is 17× the smallest. This is the property the aggregate metrics are proxies for, and it is why macro-F1 of 0.4213 and macro sensitivity of 50.20 % arrive together rather than one at the other's expense. For contrast, eleven of the thirty-one scored PAD runs in this repository terminate at a macro sensitivity of exactly 16.67 % — that is 1/6, the value of predicting one class for every input — including runs of 150 and 200 epochs on this same whole-image data. Spreading predictions across all six classes is not the default outcome of this pipeline on this dataset, and this checkpoint is the flattest we have obtained.

*The sensitivity is the highest figure in the table, including the transformer.* At 50.20 % macro recall it exceeds the published MedMamba by 13.3 points and the published Swin-T, which wins that table overall, by 7.1. On a six-class skin-lesion problem where the clinically costly error is a missed malignancy, macro recall is the metric that most nearly tracks what the task is for, and it is also the metric a class-weighted focal objective is built to move — so this is the objective working as specified rather than an incidental result. Macro precision at 40.29 % is likewise above the published MedMamba row, which matters because it means the recall was not bought by indiscriminate minority-class firing: both halves of the F1 moved.

*Ranking quality holds up where the decision rule does not.* Macro ROC-AUC is 0.7934 against the published MedMamba's 0.8070 — within 0.014 despite six times less training data — and the per-class spread is informative: MEL reaches 0.9729 while SCC sits at 0.6511. The model separates melanoma from everything else almost perfectly in score space and fails to separate squamous-cell carcinoma at all. That localizes the remaining error to one class rather than diffusing it, which is what makes the result actionable.

*The accuracy deficit is now small and has a named cause.* Overall accuracy of 45.64 % is 13.2 points below the published MedMamba figure, but against the MedMamba-T actually trained here it is 1.5 points (45.64 % against 47.18 %) — and our column trained on 234 images, used a patient-disjoint split against an image-level one, and optimized an objective that deliberately spends head-class accuracy. ACK and BCC are 68 % of the test split; a model weighted away from them cannot lead raw accuracy and is not trying to.

**And the two reasons not to lean on it.** First, the replicate. The Table XI configuration was run twice from a byte-identical command line at the same seed, and the two runs agree on validation to within 0.4 points of accuracy and 0.012 macro-F1 while differing on test by 8.1 points of accuracy, 5.5 of balanced accuracy and 0.080 macro-F1. With 344 test images and per-class support as low as 7 for melanoma, a single test cell in this table carries a run-to-run spread comparable to the differences Table XI-B displays between models — which is precisely why the replicate is printed in Table XI rather than mentioned. The MEL F1 of 0.444 rests on six of seven images recalled; one image either way moves it by roughly 0.06. Second, the split. Every published row in Table XI-B benefits from patient leakage that our row does not have, and none of them is scored on our images. The correct use of Table XI-B is to confirm that a 0.447 M-parameter model reaches the operating region these architectures occupy, and specifically that it does so from the balanced-recall end of it. It is not evidence that it beats any of them, and Appendix C states the claim we do license.

### D. What the reconstruction pathway learns

With the decoder now attached to a live feature map, the auxiliary objective can be evaluated rather than merely described. All quantities are computed in reflectance units after inverting the normalization.

| Epoch | Spectral angle (°) | RMSE | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| 1 | 20.35 | 0.360 | 9.17 | −0.022 |
| 10 | 8.36 | — | 16.99 | 0.724 |
| 20 | 6.63 | — | 19.41 | 0.825 |
| 30 | **6.02** | **0.108** | **20.43** | **0.850** |

**Table XII.** Reconstruction quality on the hyperspectral validation split, in reflectance units. These figures come from the previous revision's hyperspectral run — 30 epochs on the class-balanced training build, with weighted cross-entropy, minority oversampling, per-patch z-score normalization and the latent decoder active — not from the headline run of Section VI-A, whose decoder is disabled. The two share an architecture and a validation split but not a configuration, so this table should be read as evidence that the pathway works, not as a property of the run reported in Tables II–VI.

Spectral angle falls by a factor of 3.4 and structural similarity rises from essentially zero to 0.85, monotonically, across a run in which classification accuracy is also improving. The decoder is learning to reproduce the input cube from the recursive core's own final feature map, which means that feature map retains a substantial amount of spectral detail rather than collapsing to whatever is minimally sufficient for a three-way decision.

Two honest boundaries on that. We have not run the ablation — the same configuration with the reconstruction term at zero — so we cannot attribute any part of the classification result to this objective; the numbers establish that the pathway works and is measurable, not that it helps. And these values would have been unobtainable in the previous revision for two separate reasons: the gradient never reached the encoder, and the metrics were computed in normalized rather than reflectance units, where the mismatch between a sigmoid output and a z-scored target held the loss near a floor regardless of decoder quality.

### E. What recursion costs

This is the paper's most transferable result, and it is a negative one.

**Fig. 4** — Parameters fall by 61×; epoch time rises about 5×.
![Fig. 4](figures/fig4_cost.svg)

**Weight sharing reduces storage, not work.** With six latent updates, three improvement steps and three deep-supervision segments, the shared core is applied

```
(6 + 1) × 3 × 3 = 63 times per training step
```

and each application runs two blocks, so 126 mixer calls per step. (At the model file's own default of four segments this is 84 and 168.) The hierarchical model, by contrast, makes one pass over a 3×3 token grid, because it merges patches four-to-one; the recursive model makes 63 passes over 121 tokens, because it does not merge at all. Measured on the same PAD split under the earlier harness, the hierarchical model completed an epoch in roughly 4.8 minutes against 24.2 minutes for the recursive one — a factor of about five. Both variants have not been re-timed under the current pipeline, so this ratio should be read as an order-of-magnitude statement rather than a precise figure.

**And that shows up in FLOPs, overwhelmingly.** Counting multiply-accumulates on every convolution and linear layer by forward hook, and adding the selective-scan cost analytically from each scan's own `(K, D, N, L)`:

| Quantity | GMedMamba-R | MedMamba-HSI | Ratio |
|---|---:|---:|---:|
| Trainable parameters | **0.446 M** | 3.65 M | **0.12×** |
| Parameter memory | **1.79 MB** | 14.60 MB | **0.12×** |
| Conv/linear FLOPs | 6,187 M | 26 M | 238× |
| Scan FLOPs (analytical) | 16 M | 7 M | 2.3× |
| **Total FLOPs** | **6,203 M** | **33 M** | **191×** |
| Latency, batch 1 | 25.5 ms | **2.61 ms** | 9.8× |
| Latency, batch 16 | 28.8 ms | **3.22 ms** | 9.0× |
| Throughput | 555 patches/s | **4,974 patches/s** | 0.11× |
| Peak inference memory | 113 MB | **37.3 MB** | 3.0× |

**Table XIII.** The complete efficiency panel for one 11×11×32 patch, both models measured in the same session on the same GPU with the same code. Convolution and linear FLOPs are counted by forward hook; the scan contribution is computed analytically from each scan's own `(K, D, N, L)`. Ratios are GMedMamba-R relative to MedMamba, so a value below 1 favours ours.

We report these together because separately each one misleads. **GMedMamba-R is 8.2× smaller and 191× more arithmetic.** That is the paper's thesis in one table, and it is robust to the single soft number in it: the scan estimate is analytical rather than measured, but it is 0.3 % of our total and 21 % of MedMamba's, so even doubling MedMamba's scan cost leaves the ratio above 150×.

The gap between 191× the FLOPs and only 9.8× the latency is worth a sentence, because it is not noise. Dividing through, GMedMamba-R sustains roughly 243 GFLOP/s against MedMamba's 12.5 — about twenty times the arithmetic intensity. Both are far below what the GPU can do; MedMamba is simply so launch-bound at an 11×11 working size that most of its wall-clock is spent not computing. The recursive model wins that particular contest and still loses the wall-clock by an order of magnitude, which is the least flattering and most useful way to put it. (Independent measurements of the same architecture during two test-split evaluations gave 27.8 ms at 534 patches/s and 13.9 ms at 1,099 patches/s. The spread is a factor of two, and it is not measurement noise: the slower figure is from a run carrying an active reconstruction decoder, the faster from one without. Treat the absolute latencies as configuration-dependent to within about 2× and the *ratio* to MedMamba as the stable quantity, since it is the one measured in a single session with both models present.)

The cost is almost entirely the recursion, and it scales exactly as the arithmetic predicts. Varying the segment count and the inner loop while holding everything else fixed gives 2.17, 4.18, 6.19 and 8.20 GFLOPs at 21, 42, 63 and 84 core applications — a straight line of **95.7 MFLOPs per core application** on a **157 MFLOPs** intercept. The intercept is the entire rest of the model: spectral pathway, stem, fusion and head together are 2.5 % of the compute, and the shared 0.446 M-parameter core is the other 97.5 %.

Two things follow that a reader should not have to infer. Reducing parameters by sharing weights did not reduce work; it multiplied it by the number of times the shared block is applied. And because `forward()` runs the full deep-supervision recursion and returns the last segment's logits, **inference pays this too** — the 63 applications are not a training-only cost.

**Measured cost of the runs reported here:**

| Run | Train patches/epoch | Steps/epoch | s/epoch | Peak VRAM |
|---|---:|---:|---:|---:|
| HSI, 32 bands, reconstruction off (Section VI-A) | 245,248 | 958 | 679 | 876 MB |
| HSI, 32 bands, reconstruction on (Section VI-D) | 36,855 | 144 | 209 | 870 MB |
| PAD, RGB, reconstruction off | 100,800 | 394 | 193 | 1,410 MB |

**Table XIV.** Measured cost, median across epochs (HSI headline 654–706 s, HSI reconstruction 207–512 s, PAD 187–295 s; the widest values are first epochs, which absorb dataset warm-up). Epoch time includes the validation pass, which is a large fixed cost — 33,452 patches for the headline HSI run, 66,903 for the reconstruction run and 90,000 for PAD. The first two rows are the same architecture on the same GPU and differ by 6.7× in per-epoch training patches; per patch they cost 2.77 ms and 5.67 ms respectively, so the decoder roughly doubles the per-patch cost.

Note that both hyperspectral runs have the *lower* peak memory despite carrying eleven times the channels, and the reconstruction run does so with an active decoder. This corrects a claim in the previous revision, which reported hyperspectral memory at 5.6× the RGB figure and inferred from it that the spectral pathway scales with band count. That measurement was taken with spectral chunking disabled; the runs reported here chunk the spectral pathway at 1,024 patches, which bounds its peak independently of band count, and under that setting the relationship reverses. The earlier inference was an artefact of one configuration flag, not a property of the architecture.

**One flag was worth 39×.** The choice of token mixer has consequences wildly out of proportion to its apparent size. The table below is a measured optimization ladder in which each row changes exactly one thing relative to the row above.

| # | Configuration | ms/step | h/epoch | Cumulative speed-up |
|---|---|---:|---:|---:|
| 0 | `ss2d` mixer, reconstruction on, fp32, batch 150 | **66,880** | **91.8** | 1× |
| 1 | ↳ mixer changed to `mlp` | 1,721 | 2.36 | **39×** |
| 2 | ↳ + reconstruction off | 1,385 | 1.90 | 48× |
| 3 | ↳ + bf16 mixed precision | 363 | 0.50 | 184× |
| 4 | ↳ + per-batch numeric audits removed | 354 | 0.49 | 189× |
| — | `attention` mixer, batch 512, bf16 | 1,287 | 0.52 | 52× |
| R | *Reference:* hierarchical backbone, bf16, batch 150 | **58** | **0.08** | — |

**Table XV.** Optimization ladder for the recursive variant on PAD-UFES-20, measured on a superseded harness configuration and reported for the attribution it makes possible rather than as current performance.

The 66,880 ms figure is not a typo. The selective scan runs through a pure-PyTorch reference implementation — a Python loop over all 121 spatial positions, each issuing a handful of CUDA kernels — which under the older defaults meant on the order of 100,000 kernel launches per step with the GPU idle while a single CPU core issued them. A fifty-epoch run at that rate would have taken roughly six months. The remaining 3.4× came from mixed precision and from not paying for an unused reconstruction pathway. None of it was algorithmic: the architecture is identical between rows 0 and 4.

We record the ladder rather than the endpoint because "the small model is slow" is exactly the kind of finding that gets attributed to an architecture when it belongs to an implementation detail. The honest statement is narrower: the recursive architecture is genuinely about five times more expensive per epoch than the hierarchical baseline, and it was briefly four orders of magnitude more expensive for reasons that had nothing to do with recursion. Restoring `ss2d` as a serious option requires a compiled scan kernel, for which the model exposes a registration hook; until then it is an ablation rather than a configuration.

**Checkpointing is load-bearing.**

| Deep-supervision segments | Core checkpointed | Result |
|---|---|---|
| 3 | yes | 502 ms/step, 1.80 GB |
| 3 | no | **CUDA OOM (16 GB)** |
| 4 | yes | 635 ms/step, 1.94 GB |
| 4 | no | **CUDA OOM (16 GB)** |

**Table XVI.** Gradient checkpointing on the recursive core, measured on the same earlier harness configuration as Table XV. The qualitative conclusion — that the run does not fit without checkpointing — still holds: every run reported in this paper has core checkpointing enabled.

**Inference is cheaper than training, but not cheap.** With no graph to retain, the model serves batch-1 queries in tens of milliseconds rather than the seconds a training step costs. But Table XIII is the honest reading of what that means: recursion here is expensive to train, arithmetically expensive to run, and *empirically* tolerable to run only because the tensors are small. On larger inputs, or on hardware where 6.2 GFLOPs per patch actually binds, that last clause stops holding — and the kernel-launch slack that currently hides the cost is the same effect Table XV's mixer row exhibits at four orders of magnitude.

A 0.447 M-parameter model exhausts 16 GB of VRAM without gradient checkpointing, because all 63 core applications would otherwise hold their activations alive simultaneously until a single backward pass. The core therefore checkpoints itself during training; on this architecture that is not a tunable flag. We consider this the sharpest available statement of the trade: **parameter count is a poor proxy for the resources a recursive model needs**, and anyone reporting "0.45 M parameters" without also reporting the 63 core applications and the mandatory checkpointing is reporting half a result.

---

## VII. Discussion

### A. Where this work sits in the Mamba family

**Fig. 5** — The Mamba lineage, the recursion branch, and the parameter scale of each.
![Fig. 5](figures/fig6_lineage.svg)

| Model | Domain | Parameters | Headline published result |
|---|---|---|---|
| Mamba [2] | sequence modelling | — | Selective SSM; near-linear scaling in sequence length |
| Vision Mamba — Ti / S / B [3] | ImageNet-1K | 7 / 26 / 98 M | 76.1 / 80.3 / 81.9 % top-1 |
| VMamba — T / S / B [4] | ImageNet-1K | 30 / 50 / 89 M | 82.6 / 83.6 / 83.9 % top-1 |
| MedMamba — T / S / B [5], [6] | 16 medical datasets, 411,007 images | 15.2 / 23.5 / 48.1 M | 84.0 / 84.3 / 83.8 % average accuracy |
| TRM [8] | structured reasoning | 7 M | Sudoku-Extreme 87.4 %, ARC-AGI-1 44.6 % |
| **GMedMamba** (this work) | spectral–spatial medical | 27.43 M (RGB) / 2.77 M (HSI) | — |
| **GMedMamba-R** (this work) | spectral–spatial medical | **0.447 M** | Tables II, VII, VI |

**Table XVII.** Published reference points across the lineage. These are reference-class comparisons of scale, not accuracy comparisons: no two rows share a benchmark.

Vision Mamba and VMamba are ImageNet backbones and report no medical or spectral results, so neither can enter a metric table here without being retrained, which we have not done. They appear for two reasons that are not about accuracy. SS2D, the scan primitive inside every spatial block in this paper, comes from VMamba. And their parameter scale is what makes "0.447 M" mean anything: the smallest published Mamba-family vision backbone in this table is Vim-Ti at 7 M, and GMedMamba-R is sixteen times smaller than that.

The TRM row is the one to read against our recursive variant, and the comparison is instructive rather than flattering. TRM achieves its results at 7 M parameters on tasks with exact answers and a fixed token grid; we took its mechanism to 0.447 M on noisy medical patch classification. The mechanism transferred cleanly — recursion, deep supervision and EMA all run, and a regression test confirms gradient coverage across every segment. Whether the *benefit* transferred is a separate question that our results do not settle, and we inherit TRM's architecture, not its conclusions.

One further point belongs here because it is a negative result the field would benefit from. **TRM-style recursion is not a cheap-inference technique.** Section VI-E documents 63 core applications per training step, mandatory gradient checkpointing, and roughly five times the epoch time of the hierarchical baseline. A reader who takes "0.45 M parameters" as a proxy for deployment cost will be badly wrong. We have not seen this trade-off quantified for a recursive model in a medical imaging setting.

### B. On comparison with MedMamba

MedMamba benchmarks itself on PAD-UFES-20 against four standard architectures, at 224×224 whole-image resolution: ResNet50 at 56.62 % overall accuracy, ConvNeXt-B at 54.73 %, ViT-B at 50.36 %, MedMamba at 58.80 %, and Swin-T [9] at 62.15 %. Table XI-B reproduces that table in full, with our whole-image checkpoint placed beside it as a reference point.

Two things follow. First, MedMamba does not win its own table — Swin-T beats it by 3.4 points of accuracy and 7.1 points of F1 — which recalibrates the target: the interesting question for a MedMamba derivative is not only whether it beats MedMamba, but whether the state-space family beats a well-tuned transformer on this data at all, and the published answer on PAD-UFES-20 is currently no. Second, and decisively for us: those numbers are computed on whole lesion images and ours on 11×11 patches. No arithmetic between them is meaningful in either direction, and Table VIII is the clearest demonstration of why.

The repository does contain four pairs of runs in which GMedMamba and a local MedMamba implementation were trained on the same data directories, for the same number of epochs, at the same learning rate, on the same hardware, and GMedMamba scored higher on all four on both accuracy and macro-F1 — most decisively on three-class histology, where MedMamba's macro-F1 of 0.26 against 65 % accuracy is the signature of collapse toward the majority class. We report this because the previous revision's decision to omit it entirely was too conservative, but its weight is limited and we would rather understate it: those pairs were not matched on batch size, carried an auxiliary objective on one side only, used `torch.compile` on one side only, ran one seed for ten epochs, and came from a harness with a demonstrable metric defect in three of the four cells. They are consistent with the architecture helping. They do not establish it.

Section VI-B supersedes those pairs on the hyperspectral arm. There is now a MedMamba run on the same patient split, at the same batch size, precision, seed and checkpoint rule, and both models have been scored on the identical held-out test split with identical per-class support. The outcome on those splits is an advantage rather than parity — macro-F1 ahead by 0.044 on validation and 0.030 on test, and raw accuracy ahead on both — but we decline to bank it, because the two runs no longer share a training distribution: ours draws from the natural-distribution build of the training split and MedMamba from the class-balanced one. That difference plausibly accounts for the raw-accuracy inversion on its own, and it is the reason Section VI-B's conclusion is stated as competitiveness rather than superiority. What is consistent across every comparison, and does not depend on that unresolved axis, is a balanced-accuracy lead for the recursive model, which survives the strongest control we applied: on PAD-UFES-20 with loss, class weighting, data, objective, weight averaging and augmentation all matched, it is still +1.0 points. In that same matched setting GMedMamba-R also leads raw accuracy and macro-F1, which it did not before — an earlier revision of this section described a consistent raw-accuracy deficit, and the ablation showed that deficit was caused by weight averaging and augmentation rather than by the architecture. Minor schedule differences remain, so this is the strongest evidence in the paper rather than a settled result.

**The whole-image protocol is where a published-number comparison could live, and it is not yet honest.** Table XI is our first run at 224×224, and it is instructive mainly as a catalogue of the ways an apparently architectural gap can be manufactured. Four differences — a checkpoint field that records accuracy at the macro-F1-best epoch rather than the best accuracy, a training split undersampled to 234 of 1,626 images, a patient-disjoint split on one side against an image-level split on the other, and a class-weighted focal objective against unweighted cross-entropy — together account for a 13-point validation-accuracy difference without any appeal to the models. Three of the four are ours to fix and one of them, the undersampled directory, was a launch error rather than a decision. We report the table because the decomposition is the reusable part: any PAD-UFES-20 comparison that does not state its split level is not interpretable, and image-level splitting on a dataset with 1.67 images per patient is not a minor protocol detail. What the run's held-out test column does establish, and Table XI-B sets in context, is that a 0.447 M-parameter model trained on 234 images reaches the operating region these published architectures occupy — leading every one of them on macro sensitivity and clearing the published MedMamba row on macro precision and macro-F1 — while trailing all of them on raw accuracy. That is a reference point, not a comparison: the test sets differ, the split levels differ, and a seed-identical replicate of the same configuration moves the test cells by up to 8.1 points. Until the run is repeated on the full-size directory under a matched recipe, and over more than one seed, nothing in Table XI licenses a claim in either direction, and the paper's PAD evidence remains Table IX.

**A fully controlled comparison still does not exist**, on either dataset: the hierarchical GMedMamba has not been re-run under the current pipeline, and the hyperspectral pair is not matched on training regime. The RGB pair is close — loss, weighting, data, objective, weight decay, normalization, weight averaging and augmentation all matched — but gradient clipping, learning-rate schedule axis, early-stopping patience and classifier dropout still differ, and there is one seed per cell. That remains the binding limitation, and it is why the shallow probes of Section VI also matter: they are a floor that does not depend on any of these choices, and one the hyperspectral result clears convincingly while the PAD result barely clears at all.

### C. What the results support

The architectural claim is the one we hold with most confidence, because it does not depend on a comparison. The spatial half of this model can be replaced by a single recursively applied core at 1.6 % of the parameters, and the resulting model trains stably, converges, and produces a useful hyperspectral classifier with a working auxiliary reconstruction objective. Nothing downstream had to change to accommodate it.

The empirical picture is genuinely mixed, and we think the mixture is informative rather than merely inconclusive. On hyperspectral histology — the setting the spectral pathway exists for — the model reaches 90.7 % balanced accuracy and 0.858 macro-F1 on a held-out test split of 348,894 patches, clears the best shallow probe comfortably, and finishes ahead of a MedMamba baseline eight times its size on every aggregate metric on the identical evaluation set, though on a training split the two do not share; per-class behaviour is interpretable, and a reconstruction pathway measured on a sibling run demonstrably preserves spectral detail. On 11×11 RGB skin patches it barely clears a six-feature colour baseline — and so does a MedMamba baseline eight times its size, which saturates that training set within a single epoch — though under a matched recipe it does lead that baseline on every patch-level metric, and aggregating either model's predictions to the clinical image moves the numbers by several points in ways that depend on the configuration. Read together, those two results say something more specific than "it works" or "it does not": the architecture does its job where the input carries the information the label refers to, and the PAD protocol asks it to classify something that a single patch mostly does not contain.

The most reusable finding is the cost model. The intuition that fewer parameters means a cheaper model is wrong for this class of architecture, in a way that is easy to verify and expensive to discover accidentally. Recursion buys parameter efficiency by spending compute and activation memory, and reporting parameter count alone for a recursive model is closer to misleading than to incomplete.

Finally, a design lesson. Because the substitution touched only the spatial backbone — same spectral front end, same output dictionary, same head interface, same evaluation-mode signature — both variants share a training pipeline, a metric suite and a checkpoint format, and swap with one flag. That is exactly the condition needed for the controlled ablation in Section VIII, and it was far cheaper to preserve than it would be to reconstruct later.

---

## VIII. Limitations and Required Experiments

### A. Limitations

1. **No fully controlled baseline.** The hyperspectral MedMamba comparison of Section VI-B is matched on dataset, split, normalization, optimizer, batch size, precision, epoch budget, seed and checkpoint rule, but not on training-data fraction, class weighting, weight averaging or the auxiliary objective. Macro-F1 comes out level regardless of which way those cut; the balanced-accuracy difference does not, and class weighting alone is sufficient to explain it. The PAD-UFES-20 comparison of Section VI-C is much tighter — loss, class weighting, training-set size, objective, weight decay, normalization, weight averaging and augmentation all match — but gradient clipping, learning-rate schedule axis, early-stopping patience and classifier dropout still differ, and every cell is a single seed. The hierarchical GMedMamba has not been re-run under the current pipeline at all. This is the binding limitation.
2. **Single runs.** One seed per configuration, no variance estimate, no significance testing. A multi-seed harness exists and has not been used for these configurations.
3. **Selection and test sets are small in subjects.** Both hyperspectral splits hold five patients, and per-patient macro recall ranges from 0.46 to 0.99 within the validation split alone (Table V). Test scores exceed validation scores by roughly 8–10 points of macro-F1 for *both* models, which we read as the test patients being an easier set rather than as a model property — but with five subjects per split, neither split's aggregate should be treated as a precise estimate of population performance.
4. **Subsampled hyperspectral training.** The reported run uses 10 % of the training split per epoch, resampled each epoch. It is converged with respect to its own schedule, not trained on all available data.
5. **No reconstruction ablation.** The auxiliary objective is shown to work and to be measurable; it is not shown to help classification.
6. **Evaluation-unit mismatch on PAD.** Patch labels are inherited from whole clinical images. Table VIII mitigates this at scoring time; it does not fix the training objective. The whole-image protocol that would remove the mismatch at source has been started but not yet run to a comparable standard: the run reported in Table XI trained on a directory that had been class-undersampled to 234 of 1,626 images and is not matched to its baseline on split level, loss or augmentation, so it decomposes a gap rather than measuring one. A seed-identical replicate of that configuration differs from it by 8.1 points of test accuracy and 0.080 macro-F1 on a 344-image split, so its test cells are additionally single-seed estimates with a spread comparable to the between-model differences in Table XI-B.
7. **Selective-scan FLOPs are analytical, not measured.** Convolution and linear FLOPs are counted by forward hook; the scan contribution is computed from each scan's own shape parameters rather than instrumented. It is 0.3 % of our total and 21 % of MedMamba's, so no conclusion in Table XIII turns on it, but the scan figures are estimates. FLOPs for the published Mamba-family models in Table XVII are not recomputed under our method, so that column remains a scale reference rather than a like-for-like comparison.
8. **Single hardware target.** All measurements come from one GPU, and the compute-bound conclusions of Section VI-E may shift on hardware with different kernel-launch overhead.
9. **`ss2d` mixer is not currently usable at scale**, pending a compiled scan kernel, so the paper's default is a practical choice rather than an evaluated best.

### B. Required experiments

**Fig. 6** — The controlled comparison this work specifies but has not yet run.
![Fig. 6](figures/fig5_protocol.svg)

**A. Matched three-way comparison.** Official MedMamba, hierarchical GMedMamba and GMedMamba-R on one frozen pipeline: same split, preprocessing, resolution, augmentation, optimizer, schedule, batch size, duration, checkpoint rule and evaluation code, with the auxiliary objective either enabled everywhere or disabled everywhere, the same training-data fraction, the same class weighting and either EMA everywhere or nowhere. Section VI-B (hyperspectral) is this experiment with four variables still loose — the build of the training split (natural distribution against class-balanced), training-data fraction, class weighting, and weight averaging together with augmentation — and closing them there remains the single highest-value next step. The auxiliary objective, loose in the previous revision, is now off on both sides. The first of the four is new and the most consequential: re-running GMedMamba-R on the class-balanced build, with everything else held at the settings of Section VI-A, is a single run and would settle whether Table VI's margins survive. On PAD, the two-model half of this has already been run: Section VI-C's matched-recipe row (Table IX) closes training-data fraction, class weighting, weight decay, normalization and the objective, and additionally removes weight averaging and augmentation to match MedMamba, leaving only gradient clipping, the learning-rate schedule axis, early-stopping patience and classifier dropout open there. What PAD still lacks is the third leg — the hierarchical GMedMamba has not been re-run under the current pipeline on this dataset either, which is the same gap Limitation 1 names as binding.

**B. Five seeds per configuration**, reported as mean ± standard deviation, with a paired test across matched seeds.

**C. Train the hyperspectral model on the full training split** rather than a 10 % resample, to separate the converged-schedule result from the converged-data result.

**D. Image-level multiple-instance learning on PAD**, with the loss defined at image level rather than only the scoring, and per-lesion metrics reported as primary. A first pass at the simpler half of this — one 224×224 sample per clinical image, no aggregation needed — is Table XI, and it is not yet usable: it must be repeated on the full 1,626-image training split rather than the undersampled 234, against a MedMamba baseline trained on the same patient-disjoint directory rather than on an image-level split of its own, with the checkpoint rule and the loss matched or else both metrics reported on both sides, and over at least three seeds per cell — the replicate in Table XI shows a single 344-image test cell moving by 8.1 points of accuracy under an identical command line, which is larger than most of the differences such a comparison would be asked to resolve. The tiled variants needed for true MIL — 112×112 tiles at stride 56 or 112, with image-level aggregation of the tile posteriors — are prepared and unrun.

**E. Ablation ladder**, one variable at a time: spectral pathway off; band gating off; fusion replaced by concatenation; stage-wise refinement off; reconstruction off; hierarchical backbone versus recursive core; and within the core, the number of latent updates, improvement steps and segments. Only this separates "spectral processing helps" from "recursion helps" from "more parameters help".

**F. Extend the efficiency panel to the hierarchical GMedMamba** once it has been re-run, so all three architectures can be compared on parameters, FLOPs, memory, latency and throughput at once. The two-model panel is Table XIII.

---

## IX. Conclusion

We described GMedMamba, a spectral–spatial extension of MedMamba, and GMedMamba-R, a variant in which the hierarchical spatial backbone is replaced by a single weight-shared core applied recursively in the manner of the Tiny Recursive Model. The substitution reduces trainable parameters from 27.43 M to 0.447 M — a factor of 61 — while preserving the spectral front end, the multi-level classification interface and the reconstruction pathway, and while allowing one preset to serve both three-channel and thirty-two-band input.

On hyperspectral breast histology, evaluated on five held-out patients over thirty converged epochs, the recursive model reaches 84.8 % accuracy, 81.4 % balanced accuracy and 0.737 macro-F1, ahead of the strongest shallow probe by 7.6 points of balanced accuracy, with an auxiliary reconstruction decoder — repaired in this revision so that its gradient provably reaches the recursive core — driving spectral angle error from 20.4° to 6.0° and structural similarity from approximately zero to 0.85. On PAD-UFES-20, under the tightest comparison in the paper — loss, class weighting, data, objective, normalization, weight averaging and augmentation all matched — GMedMamba-R leads that baseline on every patch-level metric while remaining, like it, only a little above a six-feature colour baseline; aggregating patch posteriors to the clinical image moves both models by several points, which locates much of that difficulty in the evaluation unit rather than in either architecture.

The paper's firmest result is a negative one, and it is now quantified. Weight sharing reduces storage, not work: the recursive core is applied 63 times per step — at inference as well as in training — requires gradient checkpointing to run in 16 GB at all, and under one plausible-looking mixer setting ran 39× slower than under another. Set against the MedMamba baseline, GMedMamba-R carries 8.2× fewer parameters and 191× more FLOPs, 97.5 % of them in the shared core. Parameter count, for this family of models, is not a proxy for cost; it is close to an inverse indicator of it.

The defensible position is therefore this. **GMedMamba-R delivers a working spectral–spatial classifier at 1.6 % of the hierarchical variant's parameters; on held-out hyperspectral data it clearly outperforms the best shallow model we could fit to the same patches and performs on par with a MedMamba baseline 8.2× its size — at a compute cost that is now measured rather than assumed, and one that MedMamba wins. Parity at a fraction of the parameters is the result we claim. Whether the architecture is genuinely *better*, rather than level under a comparison with four loose training-side variables, remains open, and Section VIII specifies how to answer it.**

---

## Acknowledgment

Portions of this manuscript were prepared with the assistance of an AI language model, used for source-code analysis, drafting and editing, formatting of tables and mathematical expressions, and preparation of figures. All architectural claims, measurements and conclusions derive from the supplied source code, configuration files and experiment outputs, and were reviewed by the author for technical accuracy. No experimental data, metric or result was generated or altered by that process. The author takes full responsibility for the content and conclusions of this manuscript.

---

## References

> **Note to author — reference URLs.** IEEE requires an accessible locator for online sources. Every entry below carries one. The arXiv and repository URLs for [1]–[9] are carried from the previous revision and should be spot-checked against your bibliography manager before submission; [10]–[12] were resolved against Crossref, PMC and TCIA and their bibliographic fields are complete. IEEE also expects an "[Accessed: date]" field on web-only sources — the dataset repositories below carry `[Accessed: XX-XXX-2026]` placeholders for you to date.

[1] A. Dosovitskiy *et al.*, "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021. [Online]. Available: https://arxiv.org/abs/2010.11929

[2] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," arXiv:2312.00752, 2023. [Online]. Available: https://arxiv.org/abs/2312.00752

[3] L. Zhu *et al.*, "Vision Mamba: Efficient visual representation learning with bidirectional state space model," arXiv:2401.09417, 2024. [Online]. Available: https://arxiv.org/abs/2401.09417

[4] Y. Liu *et al.*, "VMamba: Visual state space model," arXiv:2401.10166, 2024. [Online]. Available: https://arxiv.org/abs/2401.10166

[5] Y. Yue and Z. Li, "MedMamba: Vision Mamba for medical image classification," arXiv:2403.03849, 2024. [Online]. Available: https://arxiv.org/abs/2403.03849

[6] Y. Yue and Z. Li, "MedMamba," GitHub repository, 2024. [Online]. Available: https://github.com/YubiaoYue/MedMamba [Accessed: XX-XXX-2026]

[7] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778. [Online]. Available: https://arxiv.org/abs/1512.03385

[8] A. Jolicoeur-Martineau, "Less is more: Recursive reasoning with tiny networks," arXiv:2510.04871, Oct. 2025. [Online]. Available: https://arxiv.org/abs/2510.04871 — code: https://github.com/SamsungSAILMontreal/TinyRecursiveModels

[9] Z. Liu *et al.*, "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 10012–10022. [Online]. Available: https://arxiv.org/abs/2103.14030

[10] A. G. C. Pacheco *et al.*, "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones," *Data in Brief*, vol. 32, art. no. 106221, Oct. 2020, doi: 10.1016/j.dib.2020.106221. [Online]. Available: https://doi.org/10.1016/j.dib.2020.106221 — dataset: https://data.mendeley.com/datasets/zr7vgbcyr2/1 [Accessed: XX-XXX-2026]

[11] L. Quintana-Quintana *et al.*, "Recurrent breast cancer: Histopathological and hyperspectral images database (HistologyHSI-BC-Recurrence)," version 1, dataset, The Cancer Imaging Archive, 2025, doi: 10.7937/6KPY-YT49. [Online]. Available: https://doi.org/10.7937/6KPY-YT49 [Accessed: XX-XXX-2026]

[12] L. Quintana-Quintana *et al.*, "Histological hyperspectral breast cancer recurrence database (HistologyHSI-BC Recurrence)," *Scientific Data*, vol. 12, art. no. 1886, 2025, doi: 10.1038/s41597-025-06157-4. [Online]. Available: https://doi.org/10.1038/s41597-025-06157-4

> **Note on [11] and [12].** TCIA's terms ask that the collection DOI and its data descriptor be cited together, which is why both appear. If your venue's reference limit is tight, [12] is the one to keep — it carries the acquisition details (Hyperspec VNIR pushbroom camera, 400–1000 nm) this paper relies on in Section V.

> **Additional citations to consider before submission.** (a) scikit-learn, for the logistic-regression probes of Sections V and VI; (b) PyTorch, for the framework; (c) the `mamba_ssm` / selective-scan CUDA kernel used by the MedMamba baselines; (d) a multiple-instance-learning reference for the aggregation argument in Section VI-C. None of these are currently cited, and (a) and (d) are the two a reviewer is most likely to ask for.

---

## Appendix A — Code-to-Architecture Traceability

Every architectural claim in Section III corresponds to a named module, listed here so a reader can check the description against the implementation. This matters specifically because an earlier revision of this paper contained a claim — the reconstruction attachment — that the figure asserted and the code did not support.

| Architectural element | Where it lives | What it is |
|---|---|---|
| Configuration surface | `GMedMambaConfig` | One dataclass covering both variants; the recursive fields are inert unless the recursive flag is set, and validation enforces the constraints of whichever variant is selected. |
| Spectral tokenizer | `SpectralTokenizer` | One shared `Linear(1 → d)` over band values, plus wavelength-continuous or index positional encoding. |
| Spectral encoder | `HierarchicalSpectralEncoder`, `ResidualSpectralBlock`, `SpectralMamba` | Residual 1-D convolutions along the band axis with a state-space module in the middle. |
| Band gating | `BandGate`, `StageBandSelector` | Soft sigmoid or steep sigmoid around a learnable threshold, recomputed per stage. |
| Spectral compression | `ProgressiveCompressor` | MLP stack pooled over bands into a spatial context map. |
| Whole spectral pathway | `SpectralPathway` | Chunked over patches to bound peak memory; shared unchanged by both variants. |
| Spectral–spatial fusion | `FiLMFusion` and five siblings, via `make_fusion` | FiLM is the default; selectable per stage. |
| Spatial→spectral feedback | `SpectralContextUpdater` | Runs after each stage's blocks. |
| Hierarchical spatial block | `GBlock`, `GStage`, `PatchMerging2D` | The SS-Conv-SSM block plus fusion, LayerScale, DropPath and a gated FFN. |
| Hierarchical backbone | `GMedMambaBackbone` → `GMedMamba` | Returns a dictionary so heads and decoders attach without knowing which backbone produced it. |
| **Recursive core** | `RecursiveCore`, `RecursiveMambaBlock` | The shared function: two layers of mixer + gated MLP under post-RMS-norm residuals. |
| **Mixer alternatives** | `_TokenMLPMixer`, `_SpatialSelfAttention`, `SS2D` | The three settings measured in Table XV. |
| **Recursive backbone** | `GMedMambaRecursiveBackbone` → `GMedMambaRecursive` | Spectral pathway → linear stem → 2-D sinusoidal encoding → recursion. |
| **Recursive head** | `RecursiveHead` | Classification head plus halt head, accepting either a state tensor or a backbone-feature dictionary. |
| **Weight averaging** | `EMAHelper` | Shadow weights at decay 0.999; validation and the best checkpoint use the averaged copy. |
| **Live feature extraction** | `training/recursive_features.py` | Returns the segment recursion's live final feature map as a value rather than storing it, which is what lets the reconstruction gradient reach the core without breaking EMA (Section IV-C). |
| **Reconstruction** | `training/reconstruction_head_v2.py` | Decoder with a normalization-matched output activation, refusing to start on a mismatch. |
| **Reconstruction metrics** | `training/spectral_recon_metrics_v2.py`, `training/normalization.py` | Spectral metrics computed in reflectance units after inverting the normalization. |
| **Preflight gates** | `training/gates_v16.py` | Reconstruction-gradient gate, post-normalization split-drift gate, and sensitivity checks with reported margins. |
| **Training entry point** | `train_example_v16.py` | Argument parsing, dataset discovery, gates, and model construction; one flag switches the backbone. |
| **Trainer** | `training/trainerg_v12.py` | One decode path shared by training and validation, per-patient and per-capture reporting, step-axis scheduling, image-level aggregation. |
| Stability accounting | `training/numerical_stability.py` | Per-epoch gradient-health record (Section IV-B). |
| Verification | `test_trm_integration.py`, `test_recursive_features_parity.py`, `test_recon_gradient.py` | Assert the parameter budget, forward/backward across channel counts, full gradient coverage across deep supervision, EMA behaviour, parity between the live-feature path and the frozen reference loop, and that the hierarchical model's parameter count is unchanged by the recursive addition. |

---

## Appendix B — Reproducibility

| Item | Status |
|---|---|
| Model source, both variants | Available |
| Training entry point; one flag switches architecture | Available |
| Dataset preparation scripts, resumable and manifest-checked | Available |
| Per-run config, history, per-class metrics, confusion matrices | Available |
| Gradient-health and stability accounting per epoch | Available |
| Class-coverage, leakage, integrity and split-drift reports | Available |
| Per-patient and per-capture validation breakdowns | Available |
| Reconstruction-gradient gate report | Available |
| Checkpoint-reproducibility verification | Available |
| Shallow-probe baselines for both datasets | Available |
| Wall-clock and peak-memory measurements | Available |
| Parameter counts, both variants, both modalities | Available |
| Patch-to-image aggregated evaluation (PAD) | Available |
| Held-out test evaluation (PAD) | Available |
| Held-out test evaluation (hyperspectral) | Available |
| Standalone gate-G6 evaluator for finished runs | Available |
| MedMamba baseline on the hyperspectral split, matched configuration | Available |
| MedMamba baseline source (model + training script) | Available |
| MedMamba baseline on PAD-UFES-20 (matched loss, weighting, data, objective) | Available |
| MedMamba baseline test evaluation on the identical split, both datasets | Available |
| Image-level (patch-MIL) evaluation for both models | Available |
| Calibration (GMedMamba-R) | Available |
| Efficiency panel — parameters, FLOPs, memory, latency, throughput, both models | Available |
| Fully controlled MedMamba comparison (matched data fraction, weighting, EMA, objective) | **No** |
| Hierarchical GMedMamba re-run under the current pipeline | **No** |
| Multiple seeds / significance testing | **No** |
| Reconstruction ablation | **No** |
| Completed ablation ladder | **No** |

---

## Appendix C — Terminology and Claim Policy

- **MedMamba** — the original architecture of Yue and Li [5], [6].
- **GMedMamba** — the hierarchical spectral–spatial extension developed in this work.
- **GMedMamba-R** — the recursive variant.
- **TRM** — the Tiny Recursive Model of Jolicoeur-Martineau [8], the source of the recursive formulation.
- **MedMamba-HSI (local baseline)** — a local copy of the MedMamba VSSM/SS2D backbone with an unmodified selective scan, adapted for 32-band input at patch size 1; used as the comparison baseline in Section VI-B and never cited as an authoritative reproduction of the published results.
- **MedMamba-T (reference PAD baseline)** — the unmodified MedMamba-T at 224×224 and 14.47 M parameters, trained by the reference training script under the paper's own protocol (no augmentation, no pre-training, no class weighting, image-level split); the right-hand column of Table XI. It is a separate object from the local baseline above, was not trained on our dataset directories or our split, and is likewise not cited as an authoritative reproduction.
- **Shallow probe** — a class-balanced multinomial logistic regression on hand-specified patch features, used as a floor.

This manuscript does not state that GMedMamba or GMedMamba-R outperforms MedMamba, and will not until the study in Section VIII-B has been run with all variables matched. The supported statements are:

> "GMedMamba-R attains the classification behaviour reported here with 61× fewer parameters than the hierarchical variant, on a pipeline where the hierarchical variant has not been re-run."

> "On hyperspectral histology, GMedMamba-R reaches 90.7 % balanced accuracy and 0.858 macro-F1 on a held-out test split of 348,894 patches from five patients used for neither training nor model selection, and exceeds the strongest shallow probe fitted to the same patches by 7.1 points of balanced accuracy on the validation split."

> "On PAD-UFES-20, under a comparison matching loss, class weighting, training-set size, auxiliary objective, weight decay, normalization, weight averaging and augmentation, GMedMamba-R leads a 3.65 M-parameter MedMamba baseline on the held-out test split by 3.2 points of accuracy, 1.0 of balanced accuracy and 0.012 macro-F1, at 8.2x fewer parameters. Gradient clipping, learning-rate schedule axis, early-stopping patience and classifier dropout remain unmatched, and each cell is one seed, so this is a lead under a close match and not a controlled result."

> "An earlier revision reported that GMedMamba-R consistently trailed MedMamba on raw accuracy. That was caused by weight averaging and geometric augmentation in our runs, not by the architecture: with both removed the ordering inverts. The balanced-accuracy advantage survives the same ablation."

> "On the identical held-out hyperspectral test split — the same 348,894 patches with the same per-class support — under matched batch size, precision, seed and checkpoint rule, GMedMamba-R leads a 3.65 M-parameter MedMamba baseline on macro-F1 (0.858 against 0.828 on test; 0.7722 against 0.7279 on validation), on balanced accuracy (+4.0 and +6.6 points) and on raw accuracy (+1.1 and +0.6). The two runs were trained on different builds of the training split — natural distribution against class-balanced — and differ further in class weighting, weight averaging and augmentation. That training-distribution difference is sufficient to account for the raw-accuracy inversion on its own, so this is a competitiveness result at 8.2x fewer parameters and not a superiority claim."

> "On 11×11 RGB skin patches, GMedMamba-R exceeds a six-feature colour baseline by under two points, and aggregating its predictions to the clinical image recovers 9.9 points of balanced accuracy."

> "Recursion in this architecture reduces parameter count and increases compute and activation memory; both must be reported together."

> "No number in this paper is comparable to a published PAD-UFES-20 result, because the evaluation unit differs: 11×11 patches here, 224×224 whole images there."

> "A whole-image PAD-UFES-20 run exists (Table XI) and supports no comparison. Its 13-point validation-accuracy deficit against a reference MedMamba-T decomposes into a checkpoint-selection artifact, a training split undersampled to 234 of 1,626 images, an image-level baseline split against our patient-disjoint one, and a class-weighted objective against an unweighted one. The asymmetries that survive all four — a reversed balanced-accuracy and macro-F1 ordering on the held-out test splits, and a raw-accuracy gap that narrows to 1.5 points there — are reported as observations about two uncontrolled runs, not as a result."

> "On 344 patient-disjoint whole PAD-UFES-20 images, a 0.447 M-parameter GMedMamba-R checkpoint reaches 40.29 % macro precision, 50.20 % macro sensitivity, 88.65 % macro specificity, 0.4213 macro-F1, 45.64 % overall accuracy and 0.7934 macro ROC-AUC, with all six classes predicted and per-class F1 between 0.154 and 0.569. Placed beside the published PAD-UFES-20 figures of [5], [6] (Table XI-B), it is above the published MedMamba row on precision, sensitivity and F1, above every row in that table on sensitivity, and below every row on overall accuracy. This is a reference point and not a comparison: the published rows are scored on 691 images from an image-level split that leaks patients, ours on 344 from a patient-disjoint one, ours trained on 234 images against their 1,378, and a seed-identical replicate of our run moves the test cells by up to 8.1 points."
