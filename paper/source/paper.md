# Band-count-agnostic spectral–spatial classification with a recursive Mamba core: trading parameters for recursion

*M.Sc. thesis manuscript · IEEE format · consolidates GMedMamba manuscript v6 with a re-audit of the supplied run artefacts*

**Author:** G. M. D. V. L. · M.Sc. Thesis, Department of Computer Science

---

## Abstract

Medical image classification has to resolve local morphology and wider context at once, and in
hyperspectral imaging it also has to read a channel axis that carries physical meaning rather
than three colour values. MedMamba addresses the first with the SS-Conv-SSM block; it treats the
channel axis as ordinary features, so spectral order is discarded and the parameter count is tied
to the sensor. We describe GMedMamba, which adds a spectral pathway whose parameter set is
independent of band count — one shared per-band projection, positional encoding continuous in
wavelength, a residual 1-D spectral encoder with a state-space module, a learned band gate and
pooling into a spatial context map — and GMedMamba-R, which replaces the four-stage hierarchical
backbone with a single weight-shared core applied recursively after the Tiny Recursive Model.
The substitution cuts trainable parameters from 27.43 M to 0.447 M, a factor of 61, leaving the
spectral front end, head interface and reconstruction pathway untouched, and lets one preset
serve both 11×11×3 and 11×11×32 input: instantiating the identical model at 3 and 32 bands gives
446,796 and 446,409 parameters, a difference of 387 that is exactly the classifier head's growth
from three to six outputs and nothing attributable to the sensor. On hyperspectral breast
histology the recursive model reaches 90.7 % balanced accuracy and 0.858 macro-F1 on 348,894
patches from five patients used for neither training nor selection, ahead of a MedMamba baseline
eight times its size on the identical test patches — though the two runs draw training data from
differently balanced builds, so we read that as competitiveness, not superiority. On PAD-UFES-20
patches, with loss, class weighting, data, objective, normalization, weight averaging and
augmentation matched, it leads that baseline by 3.2 points of accuracy, 1.0 of balanced accuracy
and 0.012 macro-F1 — yet both models barely clear a six-feature colour probe, and aggregating
patch posteriors to the clinical image moves results by up to 9.9 points and reverses a ranking,
locating much of the difficulty in the evaluation unit. Weight sharing reduces storage, not work:
the shared core runs 63 times per forward pass, at inference as well as in training, requires
gradient checkpointing to fit in 16 GB at all, and carries 8.2× fewer parameters at 191× the
FLOPs — closer to an inverse indicator of cost than a proxy for it.

**Index Terms** — medical image classification, MedMamba, state-space models, tiny recursive
models, weight sharing, hyperspectral imaging, spectral–spatial learning, band-count agnosticism,
parameter efficiency, multiple-instance learning, patient-disjoint evaluation, computational cost.

---

## Note on provenance and verification

This manuscript consolidates GMedMamba manuscript v6 with an independent audit of the four run
directories supplied alongside it. Because the two do not agree everywhere, each result below is
marked with its provenance, and the reader should treat the two classes differently.

**Verified.** Three of the four supplied runs reproduce the manuscript's figures exactly, to
every decimal checked. The MedMamba-HSI baseline (`hsi_runs/gmedmamba/`) matches on test metrics,
per-class F1, best epoch, median epoch time (155.1 s) and peak memory (554.78 MB). The
whole-image PAD-UFES-20 pair (`rgb_runs/gmedmamba/`, `rgb_runs/medmamba/`) matches on splits,
selected epochs, test metrics, macro specificity and all per-class F1 values. Where a number is
drawn from these runs it is marked *(verified)*.

**Not supplied.** The hyperspectral GMedMamba-R headline run of Section VI-A is **not** the run in
`hsi_runs/medmamba/`. The supplied directory holds a 7-epoch run selecting epoch 1 at 87.56 %
balanced accuracy and 0.8535 macro-F1; the manuscript reports a 12-epoch run selecting epoch 9 at
90.73 % and 0.8580. The configurations are otherwise identical, and the supplied run's
`history.csv` shows the learning rate decaying on the epoch axis — the exact defect the
manuscript describes fixing by moving the schedule to the optimizer-step axis. The supplied run
is therefore the superseded one, and the headline run's artefacts were not provided. Section
VI-A carries this as an open item rather than resolving it silently. The PAD **patch-level** runs
of Tables VII–X, the shallow probes, the reconstruction measurements and the FLOPs panel likewise
have no supplied artefacts and are reported on the manuscript's authority.

A caution the audit also produced: the directory names in the supplied artefacts do not identify
their contents. `hsi_runs/medmamba/` holds a *recursive* run and `hsi_runs/gmedmamba/` holds the
*hierarchical MedMamba* baseline. Identity was established from each `config.json`, and Appendix B
gives the mapping.

## I. Introduction

A classifier looking at tissue has to do two things that pull against each other. It must resolve
local detail — the shape of a nucleus, the texture at a lesion border — and it must place that
detail in a wider context, because the same local pattern means different things in different
surroundings. Convolutional networks are excellent at the first and reach the second only slowly,
through depth. Vision Transformers reach the second directly through self-attention [1], at a
cost quadratic in token count, which becomes awkward when the channel dimension itself grows, as
it does in hyperspectral imaging.

Breast cancer histopathology is where this bites clinically. A pathologist decides whether the
architecture in front of them is benign, a carcinoma confined to the duct, or one that has
breached it, and the distinction between ductal carcinoma *in situ* and invasive ductal carcinoma
governs treatment. It is drawn on morphology that varies between laboratories, between stains and
between observers, and case volumes have grown faster than the number of pathologists. At the
margins of a diagnosis, throughput and consistency degrade together.

Hyperspectral imaging offers a physically different measurement. Instead of three integrated
colour values per pixel, a cube records tens of narrow bands, so each pixel carries a spectrum.
Absorption and scattering in tissue depend on molecular composition — haemoglobin, water,
collagen, nuclear density, stain uptake all have wavelength-dependent signatures — and an RGB
sensor integrates those signatures away. Two regions a camera renders identically can differ in
the near-infrared. Whether that difference is diagnostically useful in stained histology is an
empirical question this paper does not settle; the physical argument establishes only that the
information is present before the sensor discards it.

State-space models offer a third computational route. Mamba made the state-transition parameters
depend on the input, so a model can choose what to carry forward along a sequence while keeping
near-linear scaling [2]. Vision-oriented adaptations extended that to two-dimensional feature
maps [3], [4], and MedMamba brought it to medical image classification with the SS-Conv-SSM block:
half the channels through a convolutional path, half through a state-space path, recombined by
concatenation, channel shuffle and a residual [5], [6]. GMedMamba starts from that block and adds
what MedMamba does not have — an explicit spectral pathway.

### A. From "does it help?" to "does it need to be this large?"

The spectral extension costs parameters. A parallel pathway, per-stage fusion and a reconstruction
head take the model from roughly 13.5 M parameters in the comparable MedMamba configuration to
27.43 M. That is a real cost, and it confounds any accuracy comparison: a model with twice the
capacity that scores slightly higher has not demonstrated a better idea.

There are two ways the parameter count of such a model can be coupled to things it should not be.
The first is the sensor. Models built for three-channel input treat bands as ordinary feature
channels, which discards spectral ordering — a first-layer convolution over 32 channels is
invariant to permuting them, so the fact that band 12 lies between bands 11 and 13 in wavelength,
and that the gap between bands 17 and 18 in our data is 219 nm, is information the model never
receives — and makes the stem's width a function of the instrument. Change the sensor and the
architecture is redesigned, not merely retrained. The second is depth. A four-stage hierarchy buys
effective depth by storing distinct blocks at four widths, so depth and storage are welded
together.

The Tiny Recursive Model (TRM) [8] suggests a way out of the second. Its observation is that a
network's depth and its parameter count are separable: rather than stacking many distinct blocks,
build one small block and apply it repeatedly, carrying a state forward between applications. TRM
does this with two states — a latent reasoning state and an answer state — refined over several
improvement steps, with most steps run without tracking gradients so recursion depth does not
inflate memory. On the puzzle benchmarks it targets, a 7 M-parameter recursive network is
competitive with far larger models.

The two moves compose, and that is the point. Decouple the front end from the sensor with a shared
per-band projection and a wavelength-continuous positional encoding, and nothing in the spectral
pathway scales with band count — which exposes the backbone as the remaining parameter bottleneck.
Replace that backbone with one recursively applied core, and depth becomes a property of how many
times the core is called rather than of how many blocks are stored. The result, GMedMamba-R, has
0.447 M parameters, and its size is set by neither the sensor nor the depth.

### B. Research questions

**RQ0.** Does hyperspectral input improve breast-cancer detection relative to RGB under an
identical model and protocol?

**RQ1.** Is a spectral–spatial model derived from MedMamba band-count-agnostic — structurally and
behaviourally — with no architectural change and no per-dataset retuning?

**RQ2.** Can a single weight-shared core applied recursively replace a four-stage hierarchical
backbone at an order of magnitude fewer trainable parameters, and what does that substitution
actually cost?

The answers are uneven, and the paper is organized around saying so precisely. RQ0 turns out not
to be evaluable with the runs that exist, because the two arms are different diseases rather than
two renderings of one acquisition — a limitation of the experimental design, reported as such in
Section VI-A rather than papered over with a cross-dataset comparison. RQ1 is answered exactly in
its structural half and left open in its behavioural half. RQ2 is answered in all three of its
parts, including the part that is unflattering, and the unflattering part is the most transferable
result here.

### C. Contributions

1. **GMedMamba**, a spectral–spatial extension of MedMamba, described to match the implementation
   rather than an idealization of it.
2. **GMedMamba-R**, a TRM-style recursive backbone integrated into that architecture at 0.447 M
   parameters against 27.43 M, with the spectral front end, task heads and reconstruction
   interface preserved and one flag switching between variants.
3. **A structural band-count-agnosticism result that is exact rather than statistical**:
   instantiating the identical model at 32 and 3 bands gives 446,409 and 446,796 parameters,
   differing by the 387 parameters of a larger classifier head and by nothing attributable to the
   sensor.
4. **A converged hyperspectral result on held-out patients**, with a working auxiliary
   reconstruction objective whose gradient provably reaches the recursive core.
5. **Matched-configuration MedMamba baselines on both datasets**, with the recursive model at
   parity or ahead on 8.2× fewer parameters, and the remaining confounds enumerated rather than
   glossed.
6. **Shallow-probe floors** for both datasets, and an honest reading of the model against them —
   convincing on hyperspectral data, barely clearing on RGB patches.
7. **Patch-to-image aggregation**, quantifying how much of the low patch-level score is an
   evaluation-unit artefact, including a case where the ranking between models reverses.
8. **A measured cost model for recursion** — core applications per step, FLOPs, wall-clock,
   memory, and the gradient checkpointing without which the small model does not run at all.
9. **A specification of the controlled study** that would settle what remains open, written so
   someone could run it.

### D. Scope guard

This work does not claim clinical utility, does not claim state of the art, and does not claim a
causal effect of spectral information on diagnosis. It claims a band-count-agnostic front end
proved by construction, a parameter-efficiency result with its compute cost stated in the same
breath, and an evaluation protocol whose validity gates are checkable. Every hyperspectral number
rests on five held-out patients and a single seed.

Section II positions the work, Section III describes both variants, Section IV the training
pipeline and validity gates, Section V the experimental setup, Section VI the results, Section VII
their interpretation, Section VIII the limitations and the experiments that would close them, and
Section IX concludes.

---

## II. Related work

**Convolutional networks.** Convolution encodes a strong, data-efficient prior — nearby pixels are
related, and the same filter applies everywhere — which is why CNNs have dominated medical
imaging, and residual connections made real depth trainable [7]. *Gap:* long-range context emerges
only indirectly, through depth and pooling, and the channel axis is unordered in the first layer.

**Vision Transformers.** ViT [1] treats an image as a sequence of patch tokens and lets any two
interact through self-attention, at a cost quadratic in token count. On PAD-UFES-20 specifically, a
well-tuned hierarchical transformer remains strong: Swin-T [9] leads MedMamba's own published
comparison table. *Gap:* the quadratic cost becomes awkward exactly where the channel dimension
grows, and neither family offers a mechanism for an ordered channel axis.

**Selective state-space models.** Mamba [2] made state-space parameters input-dependent, giving
attention-like selectivity at near-linear cost. Vision Mamba [3] applied bidirectional scanning to
patch sequences, reaching 76.1 / 80.3 / 81.9 % ImageNet-1K top-1 at 7 / 26 / 98 M parameters, and
running 2.8× faster than DeiT with 86.8 % less GPU memory at 1248² resolution. VMamba [4]
introduced the two-dimensional selective scan (SS2D), traversing a feature map along four spatial
directions, reaching 82.6 / 83.6 / 83.9 % top-1 at 30 / 50 / 89 M parameters. SS2D is the sequence
primitive inside MedMamba and, optionally, inside our recursive core. *Gap:* neither Vim nor VMamba
reports medical or spectral benchmarks.

**MedMamba.** Yue and Li proposed the SS-Conv-SSM block and evaluated it across sixteen datasets
spanning ten imaging modalities and 411,007 images [5], [6]. Its three variants are 15.2, 23.5 and
48.1 M parameters (2.0, 3.5 and 7.4 GFLOPs), averaging 84.0, 84.3 and 83.8 % overall accuracy on
the non-MedMNIST subset. It is the authoritative baseline throughout this work. *Gap:* the spectral
axis is handled as an ordinary channel axis, so spectral order is unavailable to the model and the
stem's parameter count follows the sensor — channel-agnostic only in the trivial sense that the
stem can be resized.

**Hyperspectral classification.** Spectral–spatial models read the band axis explicitly, through
1-D convolutions along wavelength, 3-D convolutions over the joint cube, or spectral attention.
Most originate in remote sensing, where evaluation is typically pixel-wise within one scene.
*Gap:* patient-disjoint evaluation is rare, and architectures are commonly sized for one
instrument's band grid, so band-count agnosticism is neither claimed nor tested.

**Tiny Recursive Models.** TRM [8] argues that recursion can substitute for depth. A single
two-layer network is applied repeatedly over two carried states — a latent state `z` and an answer
state `y` — trained with deep supervision across several segments, with earlier improvement steps
run under `no_grad` so memory stays bounded regardless of recursion depth. With six latent updates
and three improvement steps, TRM obtains an effective depth of 42 layers from a 7 M-parameter
network, using EMA at decay 0.999. Its reported results are striking: 87.4 % on Sudoku-Extreme,
85.3 % on Maze-Hard, 44.6 % on ARC-AGI-1 and 7.8 % on ARC-AGI-2, against 55.0 / 74.5 / 40.3 / 5.0 %
for the 27 M-parameter Hierarchical Reasoning Model it replaces. TRM's own ablation finds the
two-state formulation clearly better than a single latent state (87.4 % against 71.9 % on Sudoku),
which is why we use two states by default. *Gap:* TRM's benchmarks are small fixed-size grids with
exact answers, ours is noisy patch classification; the mechanism's transfer is not automatic, and
the compute profile turns out very differently, as Section VI-F shows.

## III. Method

### A. Preliminaries: the inherited MedMamba backbone

The canonical pipeline is a four-stage hierarchical pyramid. An input image `[B, 3, H, W]` passes
through a non-overlapping 4×4 patch embedding to `[B, H/4, W/4, C]` with `C = 96` in the tiny
configuration. Each patch-merging step then halves spatial resolution and doubles channel width,
so the representation moves from `H/4 × W/4 × C` to `H/32 × W/32 × 8C` across widths 96, 192, 384
and 768, with stage depths (2, 2, 4, 2). A final layer normalization, global average pool and
linear classifier produce the logits.

**Fig. 1** — MedMamba: hierarchical backbone and the SS-Conv-SSM block. `figures/fig1_medmamba.svg`

Inside a block the input splits channel-wise. One half goes through a convolutional path —
convolution, normalization, activation — capturing local texture cheaply. The other half is
normalized and passed to SS2D, the two-dimensional selective scan, which flattens the feature map
along four traversal orders (→, ←, ↓, ↑), runs an input-dependent state-space recurrence along
each and merges the four results; a parallel linear–SiLU branch gates the SS2D output. The two
halves are concatenated, a channel shuffle mixes information across the split so the next block's
halves are not the same halves, and the block input is added back as a residual.

Because each branch sees only half the channels, the block delivers both local and long-range
processing at roughly the cost of a single convolutional path. That economy is why the block is
worth inheriting rather than building on a ViT block. The published variants are 15.2, 23.5 and
48.1 M parameters at 2.0, 3.5 and 7.4 GFLOPs; the baselines trained here report their own measured
counts, and no published figure is reused as if it were measured.

### B. The spectral pathway

GMedMamba adds one pathway that runs before and alongside the spatial stages. Its job is to turn a
spectrum into a compact spatial context map the spatial stages can be conditioned on, and every
part of it is deliberately independent of the number of bands. That independence is the paper's
structural claim for RQ1, and it is proved here by construction rather than by experiment, so each
component is given with the reason it satisfies the constraint.

**Fig. 2** — GMedMamba: one spectral pathway conditioning four spatial stages. `figures/fig2_gmedmamba.svg`

*Tokenization.* After patchification each spatial position carries a vector of band values.
`SpectralTokenizer` lifts every scalar band value to a token of width `d_token` through **one
shared linear projection applied to all bands**, giving `[B, ·, C_bands, d_token]`. The projection
maps 1 → `d_token`, so its weight tensor has shape `[d_token, 1]` whether `C_bands` is 3 or 32. A
per-band projection — the obvious alternative — would put `C_bands` into the weight shape and
destroy the property.

*Spectral position.* Order along the band axis is supplied by an additive positional encoding
rather than by parameters. Where physical band centres in nanometres are available,
`normalize_wavelengths` maps them to [0, 1] against a reference sensor range and
`continuous_wavelength_encoding` evaluates a sinusoidal basis on that continuous value; otherwise
`index_positional_encoding` falls back to band index. The distinction matters for the reason it
matters in any sequence model: spectral position corresponds to a physical quantity, and two
sensors that sample 550 nm at different indices should produce the same encoding there. Both
encodings are computed, not learned, so neither adds parameters and neither is tied to a band
count. The difference between them is the intended A-PE ablation, which has not been run.

*Spectral encoding.* `HierarchicalSpectralEncoder` stacks residual one-dimensional convolutions
along the band axis, capturing local spectral shape — absorption features, band-to-band edges.
Convolutions share weights across positions by construction, so a kernel of width `k` costs the
same sliding over 3 bands or 32. A small state-space module (`SpectralMamba`) sits mid-stack for
longer-range dependencies between distant parts of the spectrum, which a stack of short kernels
would need many layers to reach.

*Band gating and compression.* `BandGate` learns a weight per band, suppressing uninformative or
noisy wavelength ranges; it is produced by a projection from the token width rather than stored as
a length-`C_bands` vector, which is what keeps it band-agnostic. `ProgressiveCompressor` then
reduces the token dimension through an MLP stack and **pools over the band axis**, emitting a
spatial context map `ctx_map [B, Hp, Wp, d_ctx]`. That pooling is where the band axis disappears:
every tensor downstream has a shape containing no reference to `C_bands`.

The whole pathway is chunked over patches (`spectral_chunk_size = 1024`) to bound peak memory
independently of band count — a detail that matters more than it looks, and Section VI-F records
the measurement that corrected an earlier claim about it.

### C. Fusion and the classification head in the hierarchical variant

The context map is not concatenated once at the end; it conditions every stage. The default
mechanism is FiLM modulation — the context produces a scale and a shift applied to the spatial
features — and five alternatives are implemented and selectable through `make_fusion`.

Two smaller mechanisms complete the loop. A per-stage band selector recomputes channel importance
on the context map at the start of each stage, so importance is depth-dependent rather than decided
once at the input. A context updater runs the other way: after a stage's spatial blocks, the spatial
features revise the context map the next stage will see. Spectral information conditions spatial
processing, and spatial processing revises the spectral summary.

The classifier does not read only the final stage; it concatenates the pooled output of every stage
with the spectral summary, so features at several depths reach the decision. A reconstruction
decoder can be attached during training, reconstructing the input cube from backbone features under
a combined pixel-wise MSE and spectral-angle objective. Section IV-C describes why this pathway
needed repair and how it was verified.

This variant is described for contrast. Its spatial parameters live in four distinct stage stacks at
four widths, and that is the cost the recursive variant is designed to remove.

### D. The recursive variant

A four-stage backbone spends most of its parameters on having many distinct blocks at several
widths. GMedMamba-R keeps one block, at one width, and applies it many times. The spectral pathway
of Section III-B is reused unchanged.

**Fig. 3** — GMedMamba-R: the hierarchical backbone replaced by one weight-shared recursive core.
`figures/fig3_recursive_core.svg`

*Stem.* The context map is normalized, projected by a single linear layer to width `d = 128`,
normalized again, and given a two-dimensional sinusoidal positional encoding scaled by a
configurable gain (0.1 in the reported runs). The result, `x_emb`, is computed once and **held
fixed for the entire recursion**. It has to be fixed: it is the only term in the loop that still
refers to the input, so if it drifted with the states the recursion would gradually lose contact
with the image it is supposed to be classifying. Every core application re-reads the same `x_emb`.

*Two carried states.* A latent state `z`, where intermediate work accumulates, and an answer state
`y`, which the classifier reads, both initialized from non-trainable buffers broadcast to
`[B, Hp, Wp, d]`. Separating them means the head is not forced to read a tensor that is
mid-computation. TRM's own ablation finds the two-state formulation clearly better than a single
latent state, which is why it is the default here.

*The shared core.* `f` consists of `trm_core_layers = 2` blocks. Each applies a token mixer and then
a gated MLP under post-normalized residuals: `h ← RMSNorm(h + mixer(h))`, then
`h ← RMSNorm(h + GatedMLP(h))`. Three mixers are selectable — a depthwise convolution with GEGLU, a
plain multi-head self-attention, and the SS2D selective scan. **Every run reported here uses the
depthwise-convolution/GEGLU mixer**, and Section VI-F explains why that is a practical necessity
rather than a preference: under the pure-PyTorch reference scan, `ss2d` ran 39× slower.

*The recursion.* One improvement step performs `trm_n_latent = 6` latent updates,
`z ← f(z + y + x_emb)`, then a single answer update `y ← f(y + z)` — seven core applications. One
segment performs `trm_n_improve = 3` improvement steps, the first two under `torch.no_grad()` and
only the last carrying gradient: 21 applications. A forward pass runs
`trm_deep_supervision_steps = 3` segments with the states detached between them, giving **63
applications of the same 2-layer core per forward pass**, and 126 mixer calls. The head is called
after each segment and training minimizes the mean cross-entropy over the per-segment logits, so
the model is pushed to be right early rather than only at the end.

The `no_grad` prelude and the inter-segment detach are not incidental. Without them the autograd
graph would grow linearly in the 63 applications, and the memory saved by storing one core instead
of four stage stacks would immediately be spent on activations. Bounding the graph to a single
improvement step per segment is what makes the parameter saving survive into training.

*Weight averaging and checkpointing.* An exponential moving average of the weights is maintained;
validation and the saved best checkpoint use the averaged copy. Every core application is
gradient-checkpointed, recomputing activations during the backward pass instead of storing them.
This is a compute-for-memory trade, not a free saving, and Section VI-F shows it is not optional:
without it the model OOMs on a 16 GB card.

*Halting.* An ACT-style halting head that predicts whether the current segment's answer is already
correct is implemented and available, but it is **disabled in every run reported here**, so no
result depends on it and the segment count is fixed rather than learned.

### E. What is held fixed: the substitution contract

The substitution is deliberately narrow. The recursive backbone returns the same dictionary of
outputs as the hierarchical one — feature map, pooled vector, spectral context, band weights — so
reconstruction decoders and task heads attach unchanged, and in evaluation mode the model returns a
single logits tensor, so every existing trainer and wrapper continues to work. Only the spatial
backbone changed, and one command-line flag switches between them. That narrowness is what would
let a comparison isolate the backbone, and it is the precondition for the controlled ablation of
Section VIII-B.

One consequence is worth stating because it is the structural claim in operational form. Because
the recursive variant works at unit patch size and the spectral tokenizer is band-agnostic, **a
single preset serves both the 11×11×3 RGB patches and the 11×11×32 hyperspectral patches with no
shape change and no per-dataset tuning.** The hierarchical variant needs a different preset for
each.

### F. Parameter cost

| Configuration | Trainable parameters |
| --- | ---: |
| MedMamba-Tiny, six classes (reference, not measured here) | 13.53 M |
| GMedMamba hierarchical, RGB, six classes | **27.43 M** |
| GMedMamba hierarchical, HSI preset, 32 bands, three classes | 2.77 M |
| MedMamba-HSI, local baseline, 32 bands, three classes *(verified)* | 3.65 M |
| MedMamba-T, reference PAD baseline, 224×224 | 14.47 M |
| **GMedMamba-R, RGB, 3 bands, six classes** *(verified: 446,796)* | **0.447 M** |
| **GMedMamba-R, HSI, 32 bands, three classes** *(verified: 446,409)* | **0.446 M** |
| GMedMamba-R, RGB, `ss2d` mixer | 0.535 M |
| GMedMamba-R, RGB, `attention` mixer | 0.377 M |

**Table I.** Model capacity, measured directly from the constructed models under the configuration
used for the reported runs. The recursive variant is **61× smaller** than the hierarchical
GMedMamba on the same task — 1.6 % of its parameters — and **8.2× smaller** than the MedMamba-HSI
baseline it is compared against in Section VI-C. Both ratios are storage only, and neither means
anything without the compute figures of Section VI-F, which move the other way by 191×.

The two rows in bold carry the structural RQ1 result and deserve to be read together. They are the
same model under the same preset, differing only in the input it accepts and the number of classes
it emits, and they differ by **387 parameters**: exactly three additional classifier outputs at 128
weights plus one bias each. Subtracting that term leaves the two identical. A 10.7-fold change in
band count contributed nothing. What remains band-dependent is the *activation* size during the
spectral stage — and hence the compute — not the storage, a distinction Section VI-F develops.

A note on defaults, because the figures and the prose describe different layers of the same system.
The model file's own dataclass defaults are the `ss2d` mixer, four deep-supervision segments,
halting enabled and EMA at 0.999. The training entry point overrides all of these, to the
convolutional mixer, three segments, halting disabled and EMA at 0.995 on the hyperspectral arm
(and disabled entirely on the whole-image RGB run). Every run in this paper uses the entry point's
values.

<MISSING_START>
**What is missing:** Fig. 3 currently annotates the model-file defaults (`ss2d`, four segments,
halting on, EMA 0.999) rather than the values actually run (`mlp`, three segments, halting off,
EMA 0.995 / disabled).
**Prompt:** "Redraw `figures/fig3_recursive_core.svg` so every annotated constant is the value used
in the reported runs: token mixer = depthwise-conv/GEGLU (`mlp`), deep-supervision segments = 3,
improvement steps = 3 (first 2 under no_grad), latent updates = 6, core layers = 2, total core
applications = 63, ACT halting = DISABLED, EMA = 0.995 (HSI) / disabled (RGB whole-image). Mark
the gradient boundary explicitly: the no_grad prelude and the inter-segment detach."
**How to obtain it:** Edit the SVG directly, or regenerate it from the figure script. Cross-check
each constant against `GMedMambaConfig` in `models/gmedmamba.py` (lines 260–283) for the file
defaults and against each run's `config.json` `cli_args` block for the values actually used. The
mismatch is already documented in the "Defaults vs what was run" table above.
<MISSING_END>

## IV. Training pipeline and validity gates

Much of the work behind this revision is not in the model file. It is in making the numbers the
model produces mean something.

### A. Dataset gates

Before a model is constructed, the pipeline checks that the data is what it claims to be: per-file
shape, dtype, size and SHA-256 against a manifest written at preparation time; a leakage check
against the recorded patient and slide split; and a class-coverage check that refuses to start if a
split is missing a class unless that is explicitly allowed. Arrays are validated before being
memory-mapped, so a corrupt file becomes a typed, catchable error rather than a `SIGBUS` inside a
data-loader worker.

This is not theoretical. During earlier runs the gate caught a training array physically unreadable
from byte offset 1,090,519,040 onward — 71 % of a 3.7 GB file returning I/O errors reproducibly,
while its header and first gigabyte read cleanly. Without the gate that dataset would have trained
partway through an epoch and then died in a way that is very hard to attribute.

Both datasets pass the patient- and slide-level leakage checks with no overlap between any pair of
splits *(verified)*. A patch-level content-hash comparison over 50,000 sampled patches per split
found no cross-split duplicates on the hyperspectral arm *(verified)*. On the supplied whole-image
RGB run all 234 / 328 / 344 images were hashed: no cross-split duplicates, but one duplicate group
within validation and two within test — a within-split redundancy that slightly reduces the
effective independence of those small sets without constituting leakage *(verified)*. A run
configured with `abort_on_leakage` halts before training if a cross-split duplicate is found.

### B. Accounting for optimizer updates

The learning-rate schedule is stepped on the **optimizer-step axis** rather than per epoch. This is
a small change with a large consequence, and it is the reason the hyperspectral results in this
revision are usable at all. When the training split is resampled each epoch and the schedule is
stepped per epoch, the schedule runs to completion long before the data does: every earlier run in
this family peaked at epoch 1 and declined monotonically thereafter. Stepping on the step axis
removes the artefact.

The supplied `hsi_runs/medmamba/` directory is an instance of the pre-fix behaviour and is
discussed in Section VI-B. Per-epoch gradient-health accounting — total batches, valid updates,
skipped updates, non-finite losses and gradients, gradient-norm range and a stability verdict — is
recorded for every epoch, and the reported hyperspectral run skipped no optimizer updates
*(verified)*.

### C. The reconstruction pathway, and the gate that keeps it honest

In an earlier revision the decoder read a *detached* feature map, so the auxiliary loss trained the
decoder and could not shape the representation it was supposed to regularize. It now reads the live
feature map from the same segment recursion the classifier reads, returned as a value rather than
stored, which is what lets the gradient reach the core without breaking EMA. A preflight gate (G7)
fails the run if that ever stops being true. Reconstruction metrics are computed in reflectance
units after inverting the normalization; computing them in normalized units, as the earlier
revision did, held the loss near a floor regardless of decoder quality because a sigmoid output
cannot match a z-scored target.

The reconstruction decoder is **disabled** in every run whose artefacts were supplied
(`recon_mode: none`, and `gates.json` records G7 as null), so no supplied result depends on it. The
measurements in Section VI-E come from a sibling run.

### D. Acquisition drift and per-patient reporting

Normalization statistics are fitted on the training split only and recorded as such
(`"fitted_on": "train split only"` *(verified)*). A post-normalization split-drift gate (G8)
normalizes samples from each split using the training statistics and compares post-normalization
moments; on the supplied whole-image RGB run the test split sits 0.105 σ from the training mean
with a standard-deviation ratio of 0.991, and it passes *(verified)*. Representation-sensitivity
gates (G1, G2, G9) verify that the stem responds to input perturbation, that logits have
non-degenerate spread, and that disjoint batches produce different logits, so a model cannot pass by
emitting a constant; the supplied RGB run clears its thresholds by factors of 10.6, 13.8 and 874
*(verified)*. A failure aborts the run.

The trainer reports per-patient and per-capture breakdowns alongside aggregates, which is what makes
Table V possible. Checkpoint reproducibility (G5) recomputes the selection metric from the saved
checkpoint and compares it with the value logged at selection time under a 1e-6 tolerance; both
supplied recursive runs reproduce exactly, at delta 0.0 *(verified)*.

---

## V. Experimental setup

**Hardware.** NVIDIA GeForce RTX 5060 Ti (16.7 GB), PyTorch 2.10/2.11 with CUDA 12.8, 32 GB system
RAM. All timings and memory figures come from this machine.

**Datasets.** Both were prepared as stratified 80/10/10 patient-disjoint partitions. The
hyperspectral dataset exists in two builds differing **only in the training split** — one
class-balanced by undersampling, one at the natural distribution — sharing byte-identical validation
and test splits. That shared evaluation side is what makes the runs of Section VI-C comparable at
all, and the differing training side is that section's principal caveat.

*HistologyHSI-BC (hyperspectral)* [11], [12]. Breast histology, three classes — healthy, DCIS, IDC —
as 11×11×32 patches with band centres recorded in nanometres, from a Hyperspec VNIR pushbroom camera.
The 32 centres run 400.5–938.2 nm and are unevenly spaced: sixteen fall below 640 nm and, after a
219 nm gap, the remaining fourteen occupy 852–938 nm *(verified)*. That gap is precisely where an
index-based encoding would place bands 17 and 18 adjacent and a wavelength encoding would not.
Thirty-five training, five validation and five test patients, disjoint *(verified)*. The undersampled
build's training split holds 368,550 patches, 122,850 per class; the natural build's holds 2,452,086
(29.5 % healthy, 5.0 % DCIS, 65.5 % IDC — a 13.1:1 imbalance) *(verified)*. Both share the same
validation split of 334,516 patches and test split of 348,894, at the natural distribution
(validation 29.4 / 7.3 / 63.3 %, an 8.6:1 imbalance) *(verified)*.

*PAD-UFES-20 (RGB)* [10]. Clinical smartphone photographs of skin lesions, six classes — ACK, BCC,
MEL, NEV, SCC, SEK. Used in two protocols. At **patch level**: 11×11×3 patches, 1,099 training, 137
validation and 137 test patients, with 100,800 training patches (16,800 per class), 90,000
validation and 88,400 test at the natural distribution (BCC 36.9 % against MEL 2.2 %, a 16.6:1
imbalance). At **whole-image level**: 224×224 images under a patient-disjoint 70/15/15 split, 234
training / 328 validation / 344 test *(verified)*.

**A caution about the two arms.** The hyperspectral arm is breast histopathology and the RGB arm is
dermatology. They are not two renderings of one acquisition, nor two acquisitions of one tissue.
They differ in organ, disease, label cardinality, acquisition device and sample count. The RGB arm is
used here as a second sensor with a different band count — which is what the band-count-agnosticism
claim requires — and not as the RGB half of a spectral-versus-colour comparison. Section VI-A states
the consequence for RQ0.

**Model configuration.** Both recursive runs use working width 128, two core layers, six latent
updates, three improvement steps, three deep-supervision segments, the convolutional token mixer,
gradient checkpointing on the core, and the halting head disabled *(verified)*.

**Optimization.** Batch size 256 on the hyperspectral arm, bf16 mixed precision, AdamW at 1e-4 with
weight decay 0.05, gradient clipping at norm 1.0, and a linear warmup into cosine decay stepped on
the optimizer-step axis. Checkpoints are selected on macro-F1.

The hyperspectral run trains on a 10 % resampled subset of the natural-distribution training split
(≈245,200 patches per epoch, 958 steps *(verified)*) and validates on a patient-stratified 10 %
subsample (33,452 patches *(verified)*), with weighted cross-entropy at class-weight power 0.75, no
sampler, global z-score normalization, the medium augmentation preset, classifier dropout 0.2, EMA at
0.995 and the reconstruction decoder disabled *(verified)*. It was budgeted for 20 epochs
*(verified)* and stopped at 12 on the validation-divergence rule, with epoch 9 selected on macro-F1.

The PAD patch-level run trains for 40 epochs on the full 100,800-patch training split (394 steps) and
validates on all 90,000 patches, with plain cross-entropy, no sampler, per-sample min–max
normalization, geometric augmentation only, and no reconstruction decoder. The PAD whole-image run
trains for 150 epochs at learning rate 1e-3, batch size 32, focal loss at γ = 1.5 with
inverse-frequency weights, geometric and photometric augmentation, EMA disabled, selecting epoch 104
*(verified)*.

Rebalancing is handled at the loss only. Loss-level and sampler-level rebalancing both correct
imbalance and stacking them over-corrects. And the packaged augmentation presets are hyperspectral
presets: their band dropout zeroes an entire RGB channel on three-channel data, which for pigmented
lesions destroys the primary diagnostic cue, so the PAD runs use an explicit geometric-only
configuration.

**MedMamba baselines.** A local MedMamba implementation was trained on both datasets, in each case on
the identical dataset directory and patient split, with the same normalization, batch size 256, AdamW
at 1e-4, bf16, seed 42, macro-F1 checkpoint selection, the same epoch budget as our own run on that
dataset, and the same GPU. Its SS2D and Mamba internals are the original implementation, unmodified.
Two things differ: the input stem accepts 32 bands, and the patch size is 1 rather than 4 — a stride-4
embedding would reduce an 11×11 patch to 2×2 before the backbone begins, handicapping the baseline
for reasons unrelated to architecture. Per-stage widths are (64, 128, 256, 512) with depths (1, 1, 2, 1)
rather than MedMamba-T's larger settings, which were designed for 224×224 images. The result is
**3.65 M parameters** *(verified: 3,648,995)*.

The consequence worth naming is that this backbone **fuses all 32 bands in its first convolution and
then scans only the two spatial directions**. It has no mechanism that treats the spectral axis as a
sequence in its own right — precisely the gap GMedMamba's spectral pathway was built to fill.

Because our RGB run normalizes with per-sample min–max and the baseline script implemented only
per-patch z-scoring, the former was added to it, ported from our implementation and verified
bit-identical on 400 validation patches plus constant, zero and sub-epsilon edge cases.

**Shallow baselines.** Class-balanced multinomial logistic regressions were fitted to the same
patches under the same normalization, on four feature sets: a constant majority predictor;
per-channel means; per-channel means and standard deviations; and the raw flattened patch. These
establish the floor a 0.447 M-parameter network must clear to have earned its complexity, and they
are the most informative reference point currently available for this work.

**Metrics and analysis plan.** Accuracy, balanced accuracy, macro and weighted precision, recall and
F1, per-class metrics, confusion matrices, Cohen's κ, MCC and calibration. Given the imbalance,
**balanced accuracy and macro-F1 carry the argument**; raw accuracy is reported but not leaned on.
The aggregation unit is stated on every reported metric. Quality equivalence for RQ2 is assessed
against an **equivalence margin of 2.0 points** of macro-F1 and balanced accuracy, fixed before test
numbers were read into prose. Every pairwise comparison enumerates its matched and unmatched
dimensions; where the unmatched list is non-empty the comparison is labelled *indicative* and the
unmatched dimensions are named alongside the number.

**Seeds.** One seed (42) per configuration. There is no variance estimate anywhere in this work and
no significance test is performed. Section VIII-A treats this as a binding limitation, and Section
VI-D reports a seed-identical replicate whose spread bounds how much weight any single test cell can
carry.

## VI. Results

### A. RQ0: not evaluable with the runs that exist

RQ0 asked whether hyperspectral input improves breast-cancer detection relative to RGB under an
identical model and protocol. It cannot be answered here, and the reason is not a matter of degree.

The two arms are different datasets in the strong sense. HistologyHSI-BC is breast histopathology
with three tissue classes imaged as 32-band VNIR cubes; PAD-UFES-20 is dermatological photography
with six lesion classes imaged in three colour channels. They differ in organ, disease, label
cardinality, acquisition device and sample count. A difference in macro-F1 between the two arms
measures the difference between the two problems, and the spectral axis is confounded with
everything else that separates them. The design that would answer RQ0 — rendering RGB images from
the same cubes and training the identical model on both — was not run.

This is a gap in the experimental design, not a negative result, and no sentence in this paper should
be read as evidence for or against the utility of hyperspectral imaging in breast-cancer detection.

<MISSING_START>
**What is missing:** the entire RQ0 experiment — a same-acquisition HSI-versus-RGB comparison.
**Prompt:** "Render three-channel RGB images from the HistologyHSI-BC cubes themselves, either by
integrating the 32 bands against a published CIE camera-response model or by selecting three fixed
band centres near 600/550/450 nm. Then train GMedMamba-R twice with architecture, loss, optimizer,
schedule, augmentation, normalization policy, split and seed set held identical, varying only the
input: 11×11×32 cubes against 11×11×3 renderings. Report the paired difference in balanced accuracy
and macro-F1 across at least five seeds, on the same 348,894-patch test split from the same five
held-out patients."
**How to obtain it:** The rendering is a preprocessing step over the existing
`data/hsi_v7-80_10_10_importance/hsi/X_*.npy` arrays — a weighted sum along the band axis producing
`[N, 11, 11, 3]`. No architectural change is needed on the model side: the spectral tokenizer already
accepts any band count, which is exactly what RQ1 establishes, so the same preset runs on both. Budget
two runs × five seeds at roughly 1,150 s/epoch × 12 epochs each. This is the single experiment that
would let the paper's title question be asked at all.
<MISSING_END>

The remaining subsections report what the runs do support: the structural band-count claim, for which
the RGB arm serves as a second sensor; the hyperspectral classification result and its baseline; the
evaluation-unit effect; and the capacity-versus-cost trade.

### B. RQ1: band-count agnosticism

Three distinct claims are involved and they are kept apart. The **structural** claim concerns
parameter count as a function of band count, measured from constructed models. The **behavioural**
claim concerns how metrics degrade as bands are removed, and requires trained runs at several band
counts. The **encoding** claim concerns wavelength-based against index-based spectral position with
everything else fixed. Only the first is supported by evidence here.

*Structural — supported exactly.* The recursive model instantiated for 32-band hyperspectral input
with three classes has **446,409** trainable parameters; instantiated for 3-band RGB input with six
classes it has **446,796** *(both verified from the supplied `test_report.json` efficiency blocks)*.
The difference is **387**, exactly the classifier head's growth from three to six outputs: three
additional output units at 128 weights plus one bias each. Subtracting that term leaves the two models
identical. **A 10.7-fold change in band count contributed nothing to the parameter count.**

The mechanism is the one built in Section III-B, and it is worth naming which terms could have varied
and did not: the tokenizer holds one 1→`d_token` projection shared across bands; the spectral
positional encoding is evaluated, not learned; the 1-D convolutions share kernels across band
positions; the band gate is a projection from token width rather than a length-`C_bands` vector; and
the compressor pools over the band axis before the stem, so every downstream tensor has a shape
containing no reference to the band count.

This satisfies the structural prediction as stated, and it is the cleanest result in the paper because
it is exact rather than statistical — it does not depend on a seed, a split or a metric. It is also
the narrowest: it is a fact about how the parameters are laid out, and it implies nothing about whether
the representation is any good at a band count the model was not trained on. Structural agnosticism was
cheap to prove and was proved; behavioural agnosticism is expensive and was not attempted. The first
must not be allowed to stand for the second.

*Behavioural — untested.* The two band counts that were trained, 32 and 3, sit on different datasets
with different diseases and different label spaces, so they do not constitute two points on a
degradation curve and cannot be plotted as one.

<MISSING_START>
**What is missing:** the band-count sweep (prediction P1b) and Fig. 7.
**Prompt:** "Subsample the 32 HSI bands to 24, 16, 12, 8, 4 and 2 by uniform decimation over
*wavelength* (not index), holding the training recipe of Section V fixed, and train one GMedMamba-R
run per band count at three seeds each. Plot test macro-F1 and balanced accuracy against band count as
Fig. 7, with the shallow-probe floor drawn as a horizontal line, and state the band count at which
performance is no longer distinguishable from that floor. Then run the transfer test the structural
claim actually implies: train at 32 bands and evaluate at 16 *without retraining*, which the
wavelength encoding is supposed to make possible, and report the drop."
**How to obtain it:** Decimation is an index slice over the last axis of the existing patch arrays,
paired with the matching slice of the `wavelengths_nm` list in `dataset_split_report.json` so the
positional encoding still receives physical centres. The model needs no change — pass the shorter
wavelength vector to `forward(wavelengths=...)`. Seven band counts × three seeds ≈ 21 runs at ~3.8 h
each. The zero-retraining transfer test is the higher-value half and is a single evaluation pass, not
a training run.
<MISSING_END>

Two observations from the trained runs bear on the spectral pathway without answering the behavioural
question. On the hyperspectral run the band-importance head concentrates its top-quartile selection on
bands 22–30, roughly 880–930 nm in the near-infrared, with a selection stability of 0.912 and selection
frequency above 0.94 for bands 24–28; only 25.5 % of spectral energy is retained in the selected
quartile, at a redundancy score of 0.088 *(verified)*. On the whole-image RGB run the gate concentrates
on a single channel, index 0, selected in 83.7 % of samples *(verified)*. Both are consistent with a
gate doing something non-uniform. Neither establishes that the selected bands are the informative ones
— a gate can concentrate on a range for reasons of scale or noise as easily as for reasons of signal.

*Encoding — untested.* Both encodings are implemented, and the wavelength encoding is the one used in
the hyperspectral run (`use_wavelengths: true`, 32 centres forwarded to the model *(verified)*).

<MISSING_START>
**What is missing:** the wavelength-versus-index positional-encoding ablation (prediction P1c), in
both the matched-grid and mismatched-grid cases.
**Prompt:** "Train matched pairs of GMedMamba-R differing only in `use_wavelengths` (true → continuous
nm encoding, false → band-index encoding), everything else held at the Section V hyperspectral
configuration, three seeds per arm. Report two cases separately: (i) matched grid — train and test both
on the same 32 centres, where the prediction is that the two encodings are equivalent; (ii) mismatched
grid — train on one decimation of the band set and test on a different decimation covering the same
range, where the prediction is that the wavelength encoding degrades gracefully and the index encoding
does not. Report the delta in macro-F1 for each case."
**How to obtain it:** `use_wavelengths` is already a CLI flag and both code paths exist
(`continuous_wavelength_encoding` and `index_positional_encoding` in `models/gmedmamba.py`). The
mismatched-grid case reuses the decimation tooling from the sweep above. This omission is more
consequential than it looks: the wavelength encoding is the mechanism by which the model is supposed to
transfer across sensors, and transfer is the practical content of "band-count-agnostic". Without it,
agnosticism is demonstrated as a property of the parameter layout and only asserted as a property of
the representation.
<MISSING_END>

*Limit of the claim.* Agnosticism is demonstrated over band counts 3 and 32, on two sensors, and only
structurally. Nothing here shows that a model trained at one band count performs at another.

### C. Hyperspectral histology, and a matched MedMamba baseline

This is the setting the spectral machinery was built for.

| Metric | GMedMamba-R (0.446 M) | MedMamba-HSI (3.65 M) | Best shallow probe | Majority class |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | **89.42 %** | 88.80 % | 80.53 % | 63.41 % |
| Balanced accuracy | **80.77 %** | 74.14 % | 73.72 % | 33.33 % |
| Macro precision | **75.94 %** | 72.21 % | — | — |
| Macro recall | **80.77 %** | 74.14 % | — | — |
| Macro F1 | **0.7722** | 0.7279 | 0.6816 | — |

**Table II.** Hyperspectral histology, **validation** split, five held-out patients, best epoch by
macro-F1, patch-level. The GMedMamba-R column is epoch 9 of 12 with EMA weights on 33,452 patches; the
MedMamba column is epoch 3 of 30 on all 334,516 validation patches *(verified)*. The shallow column is a
class-balanced multinomial logistic regression on per-channel means and standard deviations (64
features) — the strongest of four probes — fitted to 25,000 training patches and scored on a
15,000-patch draw from the same validation split under the same normalization. These are validation
numbers, used here to position the model against its floor, and are never quoted as the paper's result.

The recursive model clears the strongest shallow probe by 8.9 points of accuracy, 7.1 points of
balanced accuracy and 9.1 points of macro-F1. The balanced-accuracy margin is the meaningful one: the
probe already reaches 80.5 % raw accuracy on a split where 63.4 % of patches are IDC, so accuracy alone
barely distinguishes the two.

Two observations about that shallow column say something about the data rather than about our model.
Per-channel spectral statistics are genuinely informative — a 64-dimensional summary reaches 0.68
macro-F1, most of the way to what a network gets. And the *raw flattened patch*, at 3,872 dimensions,
performs clearly worse (0.535 macro-F1) than the 64-dimensional summary. Spatial detail at 11×11 is not
merely unhelpful at this scale; under a linear model it is actively harmful. That the network beats both
suggests it uses spatial structure a linear model cannot, but the honest framing is that most of the
separable signal in this dataset lives in the spectral axis — precisely the axis GMedMamba was designed
to model.

**Held-out test performance.** The five test patients were never used for selection.

| Split | Patches | Accuracy | Balanced accuracy | Macro F1 | κ | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (selection) | 33,452 | 89.42 % | 80.77 % | 0.7722 | — | — |
| **Test (held out)** | 348,894 | **94.36 %** | **90.73 %** | **0.8580** | 0.880 | 0.884 |

**Table III.** Validation and test performance of the same epoch-9 EMA checkpoint, patch-level.

This is the paper's headline classification result: **90.7 % balanced accuracy and 0.858 macro-F1
across 348,894 patches from five patients that played no part in training or model selection, from a
0.446 M-parameter model.** Ranking quality is high on both averaging schemes (macro ROC-AUC 0.993,
macro PR-AUC 0.945). Calibration is close to nominal without post-hoc correction — ECE 0.0038, Brier
0.077 — and inference costs 13.9 ms at batch 1, sustaining 1,099 patches/s at a 170 MB peak.

That test numbers exceed validation numbers by ten points of balanced accuracy invites distrust, so it
is worth dwelling on. It is not selection bias working backwards: the checkpoint was chosen on
validation, which if anything should flatter validation. The straightforward reading is that the five
test patients are simply easier than the five validation patients — and the independent MedMamba run,
selected the same way on the same splits, moves the same direction by a similar margin, which is what
makes the reading credible rather than convenient.

<MISSING_START>
**What is missing:** the run artefacts for the hyperspectral headline result of Tables II and III.
**Prompt:** "Locate and supply the experiment directory for the 12-epoch GMedMamba-R hyperspectral run
that selected epoch 9 — the run whose LR schedule is stepped on the optimizer-step axis. Required files:
`test_report.json`, `validation_report.json`, `history.csv`, `config.json`,
`checkpoint_reproducibility.json`, `gates.json`, `dataset_split_report.json`, `leakage_report.json`,
`split_drift_report.json` and `test_predictions.npz`. If that run no longer exists, re-run it from the
Section V configuration with the step-axis scheduler and regenerate every number in Tables II, III, IV
and V from the new artefacts."
**How to obtain it:** The supplied `hsi_runs/medmamba/` directory is **not** this run. It holds a
7-epoch run selecting epoch 1 at 87.56 % balanced accuracy and 0.8535 macro-F1 against the manuscript's
90.73 % and 0.8580, with per-class DCIS recall 0.781 against 0.934. Its `config.json` matches the
Section V configuration in every field checked (20-epoch budget, 10 % resample, 958 steps, 33,452-patch
validation subsample, weighted CE at power 0.75, global z-score, EMA 0.995, recon off), but its
`history.csv` shows the learning rate decaying on the **epoch** axis (1e-4 → 9.938e-5 → 9.755e-5 → …),
which is the exact defect Section IV-B describes fixing. It is therefore the superseded pre-fix run.
Until the correct artefacts are supplied, every figure in Tables II–V rests on the manuscript alone and
cannot be independently checked. **This is the single highest-priority gap in the paper**, because it is
the headline result.
<MISSING_END>

**Per-class behaviour.** The three classes are not equally easy.

| Class | Validation support | P | R | F1 | Test support | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Healthy | 9,828 | 0.863 | 0.769 | 0.813 | 83,538 | 0.971 | 0.790 | 0.871 |
| DCIS | 2,457 | 0.415 | 0.677 | 0.515 | 24,570 | 0.565 | 0.934 | 0.704 |
| IDC | 21,167 | 1.000 | 0.978 | 0.989 | 240,786 | 1.000 | 0.998 | 0.999 |

**Table IV.** Per-class metrics on both held-out splits, from the epoch-9 EMA checkpoint.

IDC is essentially solved and healthy tissue is well separated, but DCIS — the minority class on both
splits — is recalled aggressively at modest precision: 93.4 % recall at 56.5 % precision on test. The
model finds most DCIS patches and pays by labelling healthy patches as DCIS. Given that carcinoma *in
situ* and invasive carcinoma are a biological continuum, and that a single 11×11 patch may straddle a
boundary, this is a plausible operating point rather than a pathological one — and a *chosen* one, since
the run used weighted cross-entropy. It is nevertheless where macro-F1 is lost. Softening the class
weight from 1.0 to 0.75 moved DCIS precision from 0.481 to 0.565 on test at a cost of 2.3 points of
recall, which is why that setting is the one reported.

**Per-patient breakdown.** Aggregates hide per-patient variation, and this dataset has five validation
patients.

| Patient | Patches | Macro recall |
| --- | ---: | ---: |
| 238 | 4,423 | 1.000 |
| 68 | 7,371 | 0.962 |
| 197 | 9,828 | 0.835 |
| 65 | 4,459 | 0.760 |
| 304 | 7,371 | 0.637 |

**Table V.** Per-patient macro recall, validation split. These are epoch-12 values, the last the run
logged; the checkpoint reported everywhere else is epoch 9, whose aggregate macro-F1 is 0.027 higher. We
report the epoch-12 breakdown with the caveat rather than reporting none.

The spread is wide — 1.00 to 0.64 — and no aggregate would have shown it. That the identity of the worst
patient moves between runs is itself the point: with five patients we cannot say whether the spread
reflects acquisition, biology or sampling, and we do not speculate. A model averaging 81 % balanced
accuracy while one patient sits at 64 % is a different clinical proposition from one uniformly at 81 %,
and the distinction should not have to be inferred.

**The matched baseline.** A local MedMamba implementation was trained on the same patient split, holding
batch size, precision, seed, checkpoint rule and GPU fixed. It trained on the class-balanced build of the
training split; the GMedMamba-R run trained on the natural-distribution build. The two builds share
byte-identical validation and test splits, so the columns below are scored on exactly the same patches
with exactly the same class support — but the models did not see the same training distribution, and that
is the single largest caveat on this table.

| | Validation | | Test (identical split) | |
| --- | ---: | ---: | ---: | ---: |
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

**Table VI.** GMedMamba-R against the local MedMamba implementation on the hyperspectral splits,
patch-level. The MedMamba column is fully *verified* against `hsi_runs/gmedmamba/`: test accuracy
93.30 %, balanced accuracy 86.76 %, macro-F1 0.8278, per-class F1 0.846/0.637/1.000, best epoch 3 of 30,
median 155.1 s/epoch and peak 554.78 MB all reproduce exactly.

**GMedMamba-R leads on every aggregate metric, on both splits.** Macro-F1 is ahead by 0.044 on validation
and 0.030 on test, and the two splits agree in direction. Balanced accuracy leads by 6.6 and 4.0 points.
Of the ten metric rows, MedMamba retains three, all marginal: validation F1 on healthy and IDC F1 on both
splits by 0.001, on a class both models have effectively solved.

**We do not read that as an architectural win, and the reason is in the training split.** MedMamba trained
on 368,550 patches balanced exactly 122,850 per class, while GMedMamba-R drew ≈245,200 patches per epoch
from a 2,452,086-patch pool at the natural 13.1:1 imbalance and compensated at the loss with weighted
cross-entropy at power 0.75. Those are two different ways of correcting the same imbalance, and the one
used here also exposes the model to 6.7× more distinct healthy and IDC patches over the run. A model that
sees more of the majority classes and is still reweighted toward the minority is well placed to lead both
raw and balanced accuracy at once, which is exactly the pattern observed. **The comparison is indicative,
not controlled**, and the unmatched dimensions are: training-split build (natural against class-balanced),
training-data fraction, class weighting, weight averaging (EMA against none) and augmentation. Matched are
the test set, patient split, batch size, precision, seed, learning rate, epoch-budget policy, GPU and
checkpoint rule. Against the 2.0-point equivalence margin fixed in Section V, the test balanced-accuracy
difference (+4.0) and macro-F1 difference (+3.0 points of F1×100) both fall outside the margin in
GMedMamba-R's favour — but with five unmatched dimensions and one seed, that margin is not attributable to
the backbone.

**MedMamba overfits this dataset badly and GMedMamba-R does not.** MedMamba reaches its best validation
macro-F1 at epoch 3 and then declines: by epoch 30 its training loss is 0.0004 while its validation loss has
risen from 0.33 to 1.405 *(verified)*. GMedMamba-R reaches its best at epoch 9 of 12 with its validation
loss minimum at the same epoch. We would like to attribute that to parameter count and cannot: the runs
differ in weight decay (0.05 against 1e-4), weight averaging, augmentation, and the size and balance of the
training pool. What the comparison does establish is that a 3.65 M-parameter model saturates this training
set within three epochs, which is a useful fact about the dataset.

**MedMamba is markedly cheaper per sample.** It trained on the full 368,550-patch split each epoch in a
median 155 s, against 679 s for GMedMamba-R on ≈245,200 patches — two-thirds of the data at 4.4× the
wall-clock and **6.6× the cost per patch**, at lower peak memory. Recursion buys parameter efficiency and
pays for it in compute. Any deployment argument for GMedMamba-R has to be made on model size, not on
training or inference cost.

### D. Skin lesions, and what the evaluation unit measures

The PAD-UFES-20 results constrain what this paper can claim, and are more useful read as a measurement of
the evaluation protocol than of the architecture.

| Metric | Validation (best epoch) | Best shallow probe | Test (held out) |
| --- | ---: | ---: | ---: |
| Accuracy | 26.92 % | 24.33 % | 25.80 % |
| Balanced accuracy | 25.96 % | 24.79 % | 25.73 % |
| Macro F1 | 0.2260 | 0.2002 | 0.2121 |
| Cohen's κ | — | — | 0.0986 |
| MCC | — | — | 0.1047 |
| Classes with zero recall | **0 of 6** | — | **0 of 6** |

**Table VII.** PAD-UFES-20, **patch** level. Validation is 90,000 patches from 137 held-out patients at
epoch 34 of 40; test is 88,400 patches from a further 137 patients. The shallow column is logistic
regression on per-channel means and standard deviations — **six numbers per patch** — fitted to 25,000
training patches and scored on a 15,000-patch draw from the *validation* split, so it is directly
comparable to the validation column only.

One row is a genuine improvement and one is a genuine problem. The improvement is class collapse: two
revisions ago this model predicted three of six classes never, at any threshold; the previous revision
reduced that to one; this run predicts all six, on both splits, with melanoma at 2.2 % prevalence recalled
at 23.3 % on test. Balanced training data, macro-F1 checkpoint selection and a collapse monitor together
removed the failure mode.

The problem is the margin over the shallow probe. On the split where the two are comparable, a
0.447 M-parameter recursive network trained for 40 epochs exceeds a logistic regression on six numbers per
patch by 2.6 points of accuracy, 1.2 points of balanced accuracy and 0.026 macro-F1. **That is not a margin
from which any architectural claim can be made.**

We do not think this is primarily a statement about the architecture, and the aggregation experiment is why.

| Model | Evaluation unit | n | Accuracy | Balanced accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| GMedMamba-R | Patch (11×11) | 88,400 | 25.80 % | 25.73 % | 0.2121 |
| GMedMamba-R | **Clinical image** | 221 | **31.22 %** | **35.61 %** | **0.2672** |
| MedMamba | Patch (11×11) | 88,400 | 27.55 % | 24.78 % | 0.2158 |
| MedMamba | **Clinical image** | 221 | **33.94 %** | **31.40 %** | **0.2519** |

**Table VIII.** The same test-set predictions, scored per patch and aggregated to the clinical image by
averaging softmax posteriors across each image's patches, for both models through the same aggregation
function. Nothing about either model changes; only the unit changes.

Averaging the same predictions over each source image gains 9.9 points of balanced accuracy and 5.5 points
of macro-F1 for GMedMamba-R, and 6.6 and 3.6 for MedMamba. The information is present in the patch
predictions, both models have it, and the scoring unit is throwing it away. **A ranking also changes**: at
patch level MedMamba leads on raw accuracy (27.55 % against 25.80 %) while GMedMamba-R leads on balanced
accuracy; at image level GMedMamba-R leads balanced accuracy by 4.2 points and macro-F1 by 0.015 while
MedMamba leads raw accuracy. Which model is "better" depends on the unit and the metric, which is the
subsection's headline rather than a footnote.

The reason is structural. Each 11×11 RGB patch is 121 pixels of skin inheriting the whole-lesion diagnosis
of the image it was cut from, and most patches from a lesion photograph contain no lesion at all. Under
that labelling a large fraction of the training signal is mislabelled background, predicting the majority
class is a rational local optimum, and per-patch accuracy structurally understates what the model knows.

Two consequences follow. No number in Table VII should be compared with any published PAD-UFES-20 result,
which are computed on 224×224 whole lesion images. And the patch-level protocol was a poor choice for this
dataset: Table VIII is the beginning of a fix rather than the end of one, since proper multiple-instance
learning defines the *loss* at image level, not just the scoring.

For contrast, the hyperspectral setting does not have this problem in the same form. A single 11×11×32 patch
of tissue genuinely carries the spectral signature of the tissue type it was cut from, the label is locally
true rather than inherited, and the results of Section VI-C are correspondingly meaningful at the unit they
are measured at.

**A matched baseline on PAD.** Because the RGB patch run used plain cross-entropy, no sampler, both splits
in full and no reconstruction decoder, a MedMamba baseline can be matched here far more tightly than on
hyperspectral data. Loss, class weighting, training-set size, auxiliary objective, weight decay,
normalization, batch size, learning rate, precision, seed, epoch budget and checkpoint rule are all
identical, and normalization is bit-identical by construction. That left weight averaging and geometric
augmentation, both on our side, so GMedMamba-R was re-run with those disabled and nothing else changed.

| | | Test (identical 88,400 patches) | | | Clinical image (221) | | | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model | Params | Acc. | Bal. acc. | Macro F1 | Acc. | Bal. acc. | Macro F1 | Best ep. |
| GMedMamba-R, as run (EMA + augmentation) | 0.447 M | 25.80 % | 25.73 % | 0.2121 | 31.22 % | **35.61 %** | **0.2672** | 34 |
| **GMedMamba-R, matched (no EMA, no augmentation)** | **0.447 M** | **30.72 %** | **25.80 %** | **0.2281** | **34.39 %** | 29.63 % | 0.2617 | 9 |
| MedMamba | 3.65 M | 27.55 % | 24.78 % | 0.2158 | 33.94 % | 31.40 % | 0.2519 | 1 |

**Table IX.** PAD-UFES-20 head-to-head, all three scored on the identical 88,400 test patches with identical
per-class support, image-level columns aggregating those same predictions over the same 221 images through
the same function. The middle row is the like-for-like comparison. All three predict all six classes.

**Under a matched recipe, the smaller model is ahead at patch level.** With weight averaging and augmentation
removed, GMedMamba-R leads MedMamba on every patch-level metric on the held-out split: accuracy by 3.2
points, balanced accuracy by 1.0 and macro-F1 by 0.012, at 8.2× fewer parameters. **This is the only
comparison in the paper where architecture is close to the sole intentional difference.** It is still not
controlled — gradient clipping, learning-rate schedule axis, early-stopping patience and classifier dropout
differ, and each cell is one seed — but those are smaller knobs than the two just removed.

**Half of a pattern we had been reporting was the recipe, not the architecture.** Across earlier comparisons
GMedMamba-R consistently led balanced accuracy and consistently trailed raw accuracy, and we treated that
pair as one phenomenon. The ablation separates them. The balanced-accuracy lead survives almost unchanged
(+1.0 points, against +0.95 before). The raw-accuracy deficit does not: it inverts, from 1.8 points behind
to 3.2 ahead. Weight averaging and augmentation were suppressing this model's accuracy on this dataset, and
describing that as an architectural operating point was wrong. We have corrected the claim rather than the
framing.

**And the regularizers were costing accuracy while buying aggregation quality.** Removing them improved
patch-level test accuracy by 4.9 points and macro-F1 by 0.016, yet cost 6.0 points of image-level balanced
accuracy (35.6 % → 29.6 %). The configuration that is worse per patch is better per lesion. We do not have a
confident mechanism — the plausible one is that averaging and augmentation produce flatter, more diverse
per-patch posteriors that survive mean-pooling better than sharper ones do — and with one seed we will not
argue it hard. It does mean the two rows are not one model being simply better than the other.

**MedMamba saturates this dataset in a single epoch.** Its best validation macro-F1 is epoch 1 of 40; by
epoch 40 its training loss has fallen to 0.15 while its validation loss has risen to 5.06. GMedMamba-R ends
at 1.64 against 1.74 — a gap of 0.10 after forty epochs — with its best at epoch 34. Whatever else the
recursive model is doing, it is not memorizing 100,800 patches. With a 3.65 M-parameter model peaking before
it has seen the data twice, this task supports far less capacity than either model brings to it — the same
conclusion the shallow probe reaches from the other direction.

<MISSING_START>
**What is missing:** run artefacts for every PAD **patch-level** run — Tables VII, VIII, IX and X, plus both
shallow probes.
**Prompt:** "Supply the experiment directories for the three PAD-UFES-20 patch-level runs (GMedMamba-R as
run at epoch 34; GMedMamba-R matched with EMA and augmentation disabled at epoch 9; MedMamba at epoch 1),
each with `test_report.json`, `history.csv`, `config.json` and `test_predictions.npz`; the image-level
aggregation output over the 221 clinical images; and the fitted shallow-probe results for both datasets
(all four feature sets: majority, channel means, means+stds, raw flattened patch) with their scoring
scripts."
**How to obtain it:** The supplied `rgb_runs/` directories hold the **whole-image** 224×224 runs of Table XI,
not these patch-level runs — a different protocol on a different split (234/328/344 images against
100,800/90,000/88,400 patches). The aggregation numbers in Table VIII can be recomputed from
`test_predictions.npz` plus a patch→image index if those are supplied. The shallow probes are a scikit-learn
`LogisticRegression(class_weight='balanced', multi_class='multinomial')` fit over features derived from the
same patch arrays under the same normalization; recording the fit script and its random state would make
them reproducible.
<MISSING_END>

**Per-class behaviour.** The aggregate margins are small enough that the per-class breakdown carries most of
the information.

| Class | Support | as run P | R | F1 | matched P | R | F1 | MedMamba P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACK | 30,400 | 0.435 | 0.227 | 0.298 | 0.408 | 0.413 | **0.411** | 0.425 | 0.322 | 0.367 |
| BCC | 30,800 | 0.516 | 0.290 | 0.372 | 0.519 | 0.291 | **0.373** | 0.518 | 0.290 | 0.372 |
| MEL | 2,000 | 0.029 | 0.233 | 0.052 | 0.032 | 0.200 | 0.055 | 0.035 | 0.252 | **0.061** |
| NEV | 7,600 | 0.209 | 0.516 | 0.298 | 0.249 | 0.451 | **0.321** | 0.226 | 0.392 | 0.286 |
| SCC | 7,600 | 0.109 | 0.087 | **0.097** | 0.090 | 0.050 | 0.064 | 0.082 | 0.066 | 0.073 |
| SEK | 10,000 | 0.131 | 0.191 | **0.156** | 0.147 | 0.144 | 0.145 | 0.115 | 0.165 | 0.135 |
| **Macro** | 88,400 | 0.238 | 0.257 | 0.212 | 0.241 | 0.258 | **0.228** | 0.234 | 0.248 | 0.216 |

**Table X.** Per-class test metrics for the three runs of Table IX. Bold marks the best F1 per row. BCC is
effectively a three-way tie — the three values fall within 0.001 — and nothing should be read into the bold
there.

Against MedMamba under the matched recipe, GMedMamba-R is ahead on four classes and behind on two, and ahead
on both of the two largest. The two it loses are MEL and SCC, also the classes every model here is weakest on:
no run reaches 0.10 F1 on SCC, none 0.07 on MEL. The wins are concentrated where the support is — ACK (+0.044)
and NEV (+0.035) account for most of the 0.012 macro-F1 margin. Comparing the first two blocks, weight
averaging and augmentation cost ACK 0.112 of F1, almost entirely through recall (0.413 → 0.227), and buy back
0.032 on SCC and 0.011 on SEK: the regularized model spreads predictions more evenly across classes at the
expense of the largest one.

### E. Whole-image PAD-UFES-20: a first pass, and four confounds

Moving PAD off the patch grid entirely has been started, and the first result is worth reporting for what it
says about comparison hygiene rather than for the numbers. Both columns below are *verified* against the
supplied `rgb_runs/` directories.

| | GMedMamba-R | MedMamba-T |
| --- | ---: | ---: |
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
| *Replicate, identical configuration and seed* | | |
| Test acc / bal. acc / macro F1 | 37.50 % / 44.67 % / 0.3413 | — |

**Table XI.** PAD-UFES-20 at whole-image resolution, first pass. **This is not a matched comparison and must
not be read as one** — the columns differ in split level, split ratios, training-set size, loss, augmentation
and checkpoint rule, and the test columns are different images in different numbers (344 against 691). It is
included to decompose an apparent 13-point validation-accuracy gap, not to rank the models. Every cell in both
columns reproduces exactly from `rgb_runs/gmedmamba/` and `rgb_runs/medmamba/` *(verified)*.

Four differences separate the columns, and none is the architecture.

*The reported figure is not the run's best accuracy.* GMedMamba-R selects on macro-F1, so its
`best_val_accuracy` field records accuracy at the macro-F1-best epoch (104), not the best accuracy reached —
42.99 % at epoch 89 *(verified)*. A little over a point of the gap is a field-naming artefact.

*The two models did not see comparable data.* The GMedMamba-R run was launched against a directory prepared
with class-undersampling of the training split, flooring every class at melanoma's 39 images and keeping 234
of 1,626 — discarding 86 % of the training data to buy a balanced prior the loss was already supplying.
MedMamba trained on 1,378. This was an error, not a design choice, and it is the single largest term in the
gap.

*The splits are not the same kind of object.* Ours is patient-grouped and the leakage gate confirms no patient
appears in two splits *(verified)*. The reference protocol splits at image level, and PAD-UFES-20 is 2,298
images from 1,373 patients, so the same patient's photographs of the same lesion are distributed across train,
validation and test. That inflates both of its held-out figures by an unmeasured amount, in its favour, and
means neither of its numbers estimates generalization to unseen patients.

*The objectives target different quantities.* Ours is focal loss with full inverse-frequency weighting, which
deliberately spends head-class accuracy — ACK and BCC are 68 % of the split — to raise tail recall. The
reference run uses unweighted cross-entropy and is free to bet on the head.

Three asymmetries survive all four. MedMamba-T leads every validation metric, but on held-out test the
balanced-accuracy and macro-F1 ordering **inverts** and the raw-accuracy gap nearly closes: 50.20 % against
33.87 %, 0.4213 against 0.3353, and 45.64 % against 47.18 % — from a model with 32× fewer parameters trained
on one-sixth the images, with the leaky split working in MedMamba's favour on both splits. Comparing each
model's selected checkpoint with itself, MedMamba-T's balanced accuracy falls from 41.30 % on validation to
33.87 % on test while ours rises from 42.21 % to 50.20 % *(verified)*; that 7.4-point drop alongside 99.2 %
training accuracy is the saturation signature Table IX records at patch level. We state this as an observation
about two uncontrolled runs and nothing more.

**Reference-point placement.** The Table XI checkpoint is the first whole-image PAD result here whose numbers
land in the range the published literature occupies.

| Model | Precision (%) | Sensitivity (%) | Specificity (%) | F1 (%) | OA (%) | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GMedMamba-R, 0.447 M (this work)** | **40.29** | **50.20** | 88.65 | 42.13 | 45.64 | 0.7934 |
| MedMamba (published) [5], [6] | 38.43 | 36.94 | 89.90 | 35.80 | 58.80 | 0.8070 |
| Swin-T (published) [9] | 49.35 | 43.10 | **90.83** | **42.87** | **62.15** | **0.8396** |
| ResNet50 (published) | 45.71 | 42.40 | 89.78 | 42.77 | 56.62 | 0.7626 |
| ConvNeXt-B (published) | 33.80 | 34.47 | 88.97 | 33.36 | 54.73 | 0.7613 |
| ViT-B (published) | 32.04 | 33.21 | 88.05 | 32.20 | 50.36 | 0.7291 |

**Table XI-B.** The Table XI checkpoint beside the published PAD-UFES-20 figures of [5], [6, Table 2]. **This
is a reference-point table, not a comparison**, and no cell subtracts meaningfully from another: our row is 344
patient-disjoint images and every published row is 691 images from an image-level split that distributes a
patient's photographs across train, validation and test, on a dataset with 1.67 images per patient. Ours trained
on 234 images against their 1,378. Bold is a reading aid, not a claim of victory. Our row is *verified* — macro
precision 40.29 %, sensitivity 50.20 %, specificity 88.65 % (recomputed from per-class specificity), F1 42.13 %,
OA 45.64 %, AUC 0.7934 all reproduce from `rgb_runs/gmedmamba/test_report.json`.

Four things about that row are genuinely good. *No class is abandoned* — per-class F1 runs 0.569 (ACK), 0.465
(BCC), 0.444 (MEL), 0.442 (NEV), 0.455 (SEK) and 0.154 (SCC) *(verified)*; five of six sit within 0.13 of each
other on a split where the largest class is 17× the smallest. *The sensitivity is the highest in the table*,
exceeding the published MedMamba by 13.3 points and Swin-T by 7.1; on a problem where the costly error is a
missed malignancy, macro recall most nearly tracks what the task is for — and macro precision is likewise above
the published MedMamba row, so the recall was not bought by indiscriminate minority firing. *Ranking quality
holds where the decision rule does not* — macro ROC-AUC 0.7934 against the published 0.8070, with MEL at 0.9729
and SCC at 0.6511 *(verified)*, localizing the remaining error to one class. *The accuracy deficit has a named
cause* — 13.2 points below the published MedMamba, but only 1.5 points below the MedMamba-T actually trained
here, from a run on 234 images with a patient-disjoint split and an objective that deliberately spends
head-class accuracy.

**And two reasons not to lean on it.** The configuration was run twice from a byte-identical command line at the
same seed; the two agree on validation to within 0.4 points of accuracy and 0.012 macro-F1 while differing on
test by **8.1 points of accuracy, 5.5 of balanced accuracy and 0.080 macro-F1**. With 344 test images and MEL
support of 7, a single test cell carries a run-to-run spread comparable to the between-model differences in
Table XI-B. The MEL F1 of 0.444 rests on six of seven images recalled; one image either way moves it by roughly
0.06. And every published row benefits from patient leakage that ours does not have, on images that are not
ours. The correct use of Table XI-B is to confirm that a 0.447 M-parameter model reaches the operating region
these architectures occupy, from the balanced-recall end of it. It is not evidence that it beats any of them.

### F. RQ2: capacity, quality and cost

Three axes, always reported together. This is the paper's most transferable result, and it is a negative one.

**Capacity.** GMedMamba-R holds 0.447 M trainable parameters against 27.43 M for the hierarchical GMedMamba on
the same task — a factor of **61**, or 1.6 % — and against 3.65 M for the MedMamba-HSI baseline, a factor of
**8.2** *(verified)*. The relationship is a ratio rather than a subtraction because the models do not differ by
a removable component: the hierarchical variants store distinct block weights at four widths, while
GMedMamba-R stores one 2-layer core at width 128 and calls it 63 times. Storage scales with the number of
distinct blocks in one case and is constant in the number of applications in the other. Against the
hierarchical variant the order-of-magnitude prediction is met comfortably; against the local MedMamba baseline
it is not, at 8.2×, and both are reported rather than only the flattering one. Neither figure means anything
without the compute figures below, which move in the opposite direction by more than either ratio.

**Quality.** Section VI-C places the recursive variant ahead of the MedMamba-HSI baseline on every aggregate
metric on the identical hyperspectral test split, and Section VI-D places it ahead on every patch-level metric
under the tightest match in the paper. Against the 2.0-point equivalence margin, quality is retained and
exceeded in both settings — indicatively in the first, and under a close match in the second.

**Cost — the inversion.** With six latent updates, three improvement steps and three deep-supervision segments,
the shared core is applied

```
(6 + 1) × 3 × 3 = 63 times per forward pass
```

and each application runs two blocks, so 126 mixer calls. The hierarchical model makes one pass over a 3×3 token
grid, because it merges patches four-to-one; the recursive model makes 63 passes over 121 tokens, because it does
not merge at all.

| Quantity | GMedMamba-R | MedMamba-HSI | Ratio |
| --- | ---: | ---: | ---: |
| Trainable parameters | **0.446 M** | 3.65 M | **0.12×** |
| Parameter memory | **1.79 MB** | 14.60 MB | **0.12×** |
| Conv/linear FLOPs | 6,187 M | 26 M | 238× |
| Scan FLOPs (analytical) | 16 M | 7 M | 2.3× |
| **Total FLOPs** | **6,203 M** | **33 M** | **191×** |
| Latency, batch 1 | 25.5 ms | **2.61 ms** | 9.8× |
| Latency, batch 16 | 28.8 ms | **3.22 ms** | 9.0× |
| Throughput | 555 patches/s | **4,974 patches/s** | 0.11× |
| Peak inference memory | 113 MB | **37.3 MB** | 3.0× |

**Table XII.** The complete efficiency panel for one 11×11×32 patch, both models measured in the same session
on the same GPU with the same code. Convolution and linear FLOPs are counted by forward hook; the scan
contribution is computed analytically from each scan's own `(K, D, N, L)`. Ratios are GMedMamba-R relative to
MedMamba, so a value below 1 favours ours.

We report these together because separately each misleads. **GMedMamba-R is 8.2× smaller and 191× more
arithmetic.** That is the paper's thesis in one table, and it is robust to the single soft number in it: the
scan estimate is analytical rather than measured, but it is 0.3 % of our total and 21 % of MedMamba's, so even
doubling MedMamba's scan cost leaves the ratio above 150×.

The gap between 191× the FLOPs and only 9.8× the latency is not noise. Dividing through, GMedMamba-R sustains
roughly 243 GFLOP/s against MedMamba's 12.5 — about twenty times the arithmetic intensity. Both are far below
what the GPU can do; MedMamba is simply so launch-bound at an 11×11 working size that most of its wall-clock is
spent not computing. The recursive model wins that particular contest and still loses the wall-clock by an order
of magnitude, which is the least flattering and most useful way to put it.

The cost is almost entirely the recursion, and it scales exactly as the arithmetic predicts. Varying the segment
count and inner loop while holding everything else fixed gives 2.17, 4.18, 6.19 and 8.20 GFLOPs at 21, 42, 63 and
84 core applications — a straight line of **95.7 MFLOPs per core application** on a **157 MFLOPs** intercept. The
intercept is the entire rest of the model: spectral pathway, stem, fusion and head together are 2.5 % of the
compute, and the shared 0.446 M-parameter core is the other 97.5 %.

Two things follow that a reader should not have to infer. Reducing parameters by sharing weights did not reduce
work; it multiplied it by the number of times the shared block is applied. And because `forward()` runs the full
deep-supervision recursion and returns the last segment's logits, **inference pays this too** — the 63
applications are not a training-only cost.

**Measured cost of the runs.**

| Run | Train patches/epoch | Steps/epoch | s/epoch | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| HSI, 32 bands, reconstruction off (Section VI-C) | ≈245,200 | 958 | 679 | 876 MB |
| HSI, 32 bands, reconstruction on (Section VI-G) | 36,855 | 144 | 209 | 870 MB |
| PAD patch, RGB, reconstruction off | 100,800 | 394 | 193 | 1,410 MB |

**Table XIII.** Measured cost, median across epochs. Epoch time includes the validation pass, a large fixed cost.
The first two rows are the same architecture on the same GPU differing by 6.7× in per-epoch training patches; per
patch they cost 2.77 ms and 5.67 ms, so the decoder roughly doubles per-patch cost.

Both hyperspectral runs have the *lower* peak memory despite eleven times the channels. This corrects an earlier
claim that hyperspectral memory ran at 5.6× the RGB figure and that the spectral pathway therefore scales with
band count. That measurement was taken with spectral chunking disabled; chunking the pathway at 1,024 patches
bounds its peak independently of band count, and under that setting the relationship reverses. The earlier
inference was an artefact of one configuration flag, not a property of the architecture — and it is the compute
counterpart to the structural parameter result of Section VI-B.

**One flag was worth 39×.** The choice of token mixer has consequences wildly out of proportion to its apparent
size.

| # | Configuration | ms/step | h/epoch | Cumulative speed-up |
| --- | --- | ---: | ---: | ---: |
| 0 | `ss2d` mixer, reconstruction on, fp32, batch 150 | **66,880** | **91.8** | 1× |
| 1 | ↳ mixer changed to `mlp` | 1,721 | 2.36 | **39×** |
| 2 | ↳ + reconstruction off | 1,385 | 1.90 | 48× |
| 3 | ↳ + bf16 mixed precision | 363 | 0.50 | 184× |
| 4 | ↳ + per-batch numeric audits removed | 354 | 0.49 | 189× |
| — | `attention` mixer, batch 512, bf16 | 1,287 | 0.52 | 52× |
| R | *Reference:* hierarchical backbone, bf16, batch 150 | **58** | **0.08** | — |

**Table XIV.** Optimization ladder for the recursive variant on PAD-UFES-20, measured on a superseded harness
configuration and reported for the attribution it makes possible rather than as current performance.

The 66,880 ms figure is not a typo. The selective scan runs through a pure-PyTorch reference implementation — a
Python loop over all 121 spatial positions, each issuing a handful of CUDA kernels — which under the older
defaults meant on the order of 100,000 kernel launches per step with the GPU idle while a single CPU core issued
them. A fifty-epoch run at that rate would have taken roughly six months. None of the recovery was algorithmic:
the architecture is identical between rows 0 and 4.

We record the ladder rather than the endpoint because "the small model is slow" is exactly the kind of finding
that gets attributed to an architecture when it belongs to an implementation detail. The honest statement is
narrower: the recursive architecture is genuinely about five times more expensive per epoch than the hierarchical
baseline, and it was briefly four orders of magnitude more expensive for reasons that had nothing to do with
recursion. Restoring `ss2d` as a serious option requires a compiled scan kernel, for which the model exposes a
registration hook; until then it is an ablation rather than a configuration.

**Checkpointing is load-bearing.**

| Deep-supervision segments | Core checkpointed | Result |
| --- | --- | --- |
| 3 | yes | 502 ms/step, 1.80 GB |
| 3 | no | **CUDA OOM (16 GB)** |
| 4 | yes | 635 ms/step, 1.94 GB |
| 4 | no | **CUDA OOM (16 GB)** |

**Table XV.** Gradient checkpointing on the recursive core, measured on the same earlier harness configuration as
Table XIV. The qualitative conclusion — that the run does not fit without checkpointing — still holds: every run
in this paper has core checkpointing enabled.

A 0.447 M-parameter model exhausts 16 GB of VRAM without gradient checkpointing, because all 63 core applications
would otherwise hold their activations alive simultaneously until a single backward pass. On this architecture
checkpointing is not a tunable flag. This is the sharpest available statement of the trade: **parameter count is a
poor proxy for the resources a recursive model needs**, and anyone reporting "0.45 M parameters" without also
reporting the 63 core applications and the mandatory checkpointing is reporting half a result.

**Where the trade is favourable.** A parameter count near 0.45 M with storage constant in band count suits
deployment where model storage or distribution binds — many sensor configurations served from one artefact, or
update bandwidth to edge devices. It is unfavourable wherever throughput or training cost binds: at 191× the
FLOPs, 9.8× the batch-1 latency and 6.6× the training cost per patch, the recursive variant is the slower model on
both sides, and no result here suggests otherwise.

<MISSING_START>
**What is missing:** measured FLOPs for the hierarchical GMedMamba, and a with/without-checkpointing measurement
under the current harness.
**Prompt:** "Extend the efficiency panel of Table XII to three architectures — GMedMamba-R, hierarchical
GMedMamba and MedMamba-HSI — measured in one session on one GPU with one script, reporting parameters, parameter
memory, conv/linear FLOPs by forward hook, analytical scan FLOPs, batch-1 and batch-16 latency, throughput and
peak inference memory. Separately, re-measure Table XV (checkpointing on/off, 3 and 4 segments) under the current
pipeline rather than the superseded harness, reporting ms/step and peak VRAM for each cell."
**How to obtain it:** The FLOP counter already exists — it produced Table XII — and needs the hierarchical variant
constructed alongside the other two. Note that the supplied `test_report.json` files record
`"flops": {"error": "ValueError: not enough values to unpack (expected 4, got 3)"}`, so the in-run profiler path
is broken even though the standalone panel works; fixing that unpack error would let every future run carry its
own FLOPs. The checkpointing re-measurement is four short runs, since each only needs to reach a stable step time
or OOM.
<MISSING_END>

### G. What the reconstruction pathway learns

With the decoder attached to a live feature map, the auxiliary objective can be evaluated rather than merely
described. All quantities are in reflectance units after inverting the normalization.

| Epoch | Spectral angle (°) | RMSE | PSNR (dB) | SSIM |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20.35 | 0.360 | 9.17 | −0.022 |
| 10 | 8.36 | — | 16.99 | 0.724 |
| 20 | 6.63 | — | 19.41 | 0.825 |
| 30 | **6.02** | **0.108** | **20.43** | **0.850** |

**Table XVI.** Reconstruction quality on the hyperspectral validation split. These figures come from a sibling run
— 30 epochs on the class-balanced training build, with weighted cross-entropy, minority oversampling, per-patch
z-score normalization and the latent decoder active — **not** from the headline run of Section VI-C, whose decoder
is disabled. The two share an architecture and a validation split but not a configuration, so this is evidence
that the pathway works, not a property of the reported run.

Spectral angle falls by a factor of 3.4 and structural similarity rises from essentially zero to 0.85,
monotonically, across a run in which classification accuracy is also improving. The decoder learns to reproduce
the input cube from the recursive core's own final feature map, which means that feature map retains substantial
spectral detail rather than collapsing to whatever is minimally sufficient for a three-way decision.

Two honest boundaries. The ablation — the same configuration with the reconstruction term at zero — has not been
run, so no part of the classification result can be attributed to this objective; the numbers establish that the
pathway works and is measurable, not that it helps. And these values would have been unobtainable in the previous
revision for two separate reasons: the gradient never reached the encoder, and the metrics were computed in
normalized rather than reflectance units, where the mismatch between a sigmoid output and a z-scored target held
the loss near a floor regardless of decoder quality.

### H. Ablations

Three factors have measurements and four do not, and the distinction is kept explicit.

| Factor | Levels tested | Effect | Source |
| --- | --- | --- | --- |
| Token mixer | `mlp`, `ss2d`, `attention` | 39× wall-clock between `mlp` and `ss2d`; params 0.447 / 0.535 / 0.377 M. **Speed only** — no matched accuracy comparison | Tables I, XIV |
| Gradient checkpointing | on / off | off ⇒ CUDA OOM at 16 GB, both 3 and 4 segments | Table XV |
| EMA + augmentation (jointly) | on / off | patch-level test accuracy +4.9 pts and macro-F1 +0.016 when **off**; image-level balanced accuracy −6.0 pts | Table IX |
| Recursion depth (`n_latent`, `n_improve`) | 21 / 42 / 63 / 84 core applications | FLOPs scale linearly at 95.7 MFLOPs per application. **Compute only** — no accuracy measured | Section VI-F |
| Deep-supervision segments | 3 (run), 4 (file default) | compute only, as above | Section VI-F |
| Two-state vs single-state carry | — | not run here; TRM reports 87.4 % against 71.9 % on Sudoku | [8] |
| Auxiliary reconstruction | on / off | pathway shown to work (Table XVI); **effect on classification not measured** | Section VI-G |

**Table XVII.** Ablation status. Only the first three rows carry a measured effect on this work's tasks, and only
the third carries an effect on a classification metric.

The mixer and depth rows measure compute, not quality: nothing here establishes that 63 core applications are
necessary, or that the same accuracy could not be had at 21. EMA and augmentation are the one factor with a
measured accuracy effect, and it is not a simple one — the setting that helps per patch hurts per lesion.

<MISSING_START>
**What is missing:** the accuracy half of the ablation ladder. Every recursion constant in Section III-D — six
latent updates, three improvement steps, three segments, two core layers, the `mlp` mixer, the two-state carry —
is currently an unvalidated design choice rather than a tuned one.
**Prompt:** "Run a one-variable-at-a-time ablation ladder on the hyperspectral configuration of Section V, three
seeds per cell, reporting test balanced accuracy, macro-F1, trainable parameters and total FLOPs for every cell:
(a) latent updates ∈ {2, 4, 6, 8}; (b) improvement steps ∈ {1, 2, 3, 4}; (c) deep-supervision segments ∈ {1, 2, 3,
4}; (d) core layers ∈ {1, 2, 3}; (e) token mixer ∈ {mlp, attention} at matched wall-clock budget, and ss2d if a
compiled scan kernel becomes available; (f) two-state against single-state carry; (g) EMA on/off and augmentation
on/off **separately**, since Table IX varied them jointly; (h) auxiliary reconstruction on/off. Report which
effects exceed the seed spread and which do not, and report factors inside the spread as such rather than
narrating them as trends."
**How to obtain it:** Every one of these is an existing CLI flag on `train_example_v16.py`
(`--trm_n_latent`, `--trm_n_improve`, `--trm_deep_supervision_steps`, `--trm_core_layers`, `--trm_mixer`,
`--trm_ema_rate`, `--augment_preset`, `--recon_mode`), so the ladder is a scheduling exercise rather than an
implementation one. Item (g) is the highest value: Table IX confounds EMA with augmentation and the paper
currently cannot say which of the two caused the accuracy inversion. Item (f) is the cheapest check of an
inherited assumption. At ~3.8 h per hyperspectral run, the full ladder at three seeds is roughly 250 GPU-hours;
restricting (a)–(d) to the hyperspectral arm and (e)–(h) to PAD patches would cut that substantially.
<MISSING_END>

### I. Unexpected observations

Reported without interpretation.

Test scores exceed validation scores on the hyperspectral arm for **both** models — GMedMamba-R by 0.086 macro-F1
and MedMamba by 0.100 — which is why Section VI-C reads it as a property of the patient sets rather than of either
model.

A seed-identical replicate of the whole-image PAD configuration differs from its twin by 8.1 points of test
accuracy and 0.080 macro-F1 while agreeing on validation to within 0.4 points *(verified against the manuscript's
replicate row)*.

Eleven of the thirty-one scored PAD runs in the source repository terminate at a macro sensitivity of exactly
16.67 % — that is 1/6, the value of predicting one class for every input — including runs of 150 and 200 epochs on
whole-image data. Spreading predictions across all six classes is not the default outcome of this pipeline on this
dataset.

The supplied `hsi_runs/medmamba/` run peaks at epoch 1 and declines monotonically thereafter, with training accuracy
rising from 0.851 to 0.924 across the same seven epochs *(verified)* — the artefact Section IV-B attributes to a
schedule stepped on the epoch axis while the training split is resampled each epoch.

## VII. Discussion

The band-count-agnostic front end works exactly as designed and costs nothing in parameters; the recursive
backbone delivers a working spectral–spatial classifier at 1.6 % of the hierarchical variant's parameters and
is competitive with or ahead of a MedMamba baseline 8.2× its size — and it pays for that with 191× the FLOPs,
mandatory gradient checkpointing and roughly five times the epoch time, which is the finding most likely to
transfer.

### A. Where this work sits in the Mamba family

| Model | Domain | Parameters | Headline published result |
| --- | --- | --- | --- |
| Mamba [2] | sequence modelling | — | Selective SSM; near-linear scaling in sequence length |
| Vision Mamba — Ti / S / B [3] | ImageNet-1K | 7 / 26 / 98 M | 76.1 / 80.3 / 81.9 % top-1 |
| VMamba — T / S / B [4] | ImageNet-1K | 30 / 50 / 89 M | 82.6 / 83.6 / 83.9 % top-1 |
| MedMamba — T / S / B [5], [6] | 16 medical datasets, 411,007 images | 15.2 / 23.5 / 48.1 M | 84.0 / 84.3 / 83.8 % average accuracy |
| TRM [8] | structured reasoning | 7 M | Sudoku-Extreme 87.4 %, ARC-AGI-1 44.6 % |
| **GMedMamba** (this work) | spectral–spatial medical | 27.43 M (RGB) / 2.77 M (HSI) | — |
| **GMedMamba-R** (this work) | spectral–spatial medical | **0.447 M** | Tables III, VI, IX |

**Table XVIII.** Published reference points across the lineage. These are reference-class comparisons of scale,
not accuracy comparisons: no two rows share a benchmark.

Vision Mamba and VMamba are ImageNet backbones reporting no medical or spectral results, so neither can enter a
metric table here without being retrained. They appear for two reasons that are not about accuracy. SS2D, the
scan primitive inside every spatial block in this paper, comes from VMamba. And their parameter scale is what
makes "0.447 M" mean anything: the smallest published Mamba-family vision backbone in this table is Vim-Ti at
7 M, and GMedMamba-R is sixteen times smaller than that.

The TRM row is the one to read against our recursive variant, and it is instructive rather than flattering. TRM
achieves its results at 7 M parameters on tasks with exact answers and a fixed token grid; we took its mechanism
to 0.447 M on noisy medical patch classification. The mechanism transferred cleanly — recursion, deep
supervision and EMA all run, and a regression test confirms gradient coverage across every segment. Whether the
*benefit* transferred is a separate question our results do not settle. We inherit TRM's architecture, not its
conclusions.

One further point belongs here because it is a negative result the field would benefit from. **TRM-style
recursion is not a cheap-inference technique.** Section VI-F documents 63 core applications per forward pass at
inference as well as in training, mandatory gradient checkpointing, and 191× the FLOPs of the baseline. A reader
who takes "0.45 M parameters" as a proxy for deployment cost will be badly wrong. We have not seen this trade-off
quantified for a recursive model in a medical imaging setting.

### B. The research questions in order

*RQ0 — does hyperspectral input beat RGB?* **Not evaluable.** The prediction required architecture, loss,
optimizer, split policy and augmentation held fixed across two renderings of one acquisition. What exists is
breast histopathology against skin lesions: two diseases, two organs, three classes against six, 32 bands against
3. This is a design gap, not a negative result, and Section VIII-B item A specifies the run that would close it.

*RQ1 — is the model band-count-agnostic?* **Partially supported, and the partition matters.** The structural
prediction got an exact confirmation: 446,409 parameters at 32 bands and 446,796 at 3, differing by the 387
parameters of a larger classifier head and by nothing attributable to the sensor. That is as clean as a
structural claim gets — arithmetic, not statistics, independent of seed, split and metric. The behavioural
prediction, graceful degradation under band subsampling, is **untested**: no sweep was run and the two trained
band counts sit on different datasets. The encoding prediction, wavelength against index, is **untested** in both
the matched-grid and mismatched-grid cases.

The warning the study design carried is exactly the one that materialized. Structural agnosticism was cheap to
prove and was proved; behavioural agnosticism was expensive and was not attempted. The claim is correspondingly
narrow: the parameter set does not know how many bands it is reading, which is a *precondition* for sensor
transfer and not a demonstration of it.

*RQ2 — can recursion replace depth?* **Supported on all three axes, with the third being the interesting one.**
Capacity: 61× fewer parameters than the hierarchical GMedMamba, which exceeds an order of magnitude comfortably,
and 8.2× fewer than the MedMamba-HSI baseline, which does not — both are reported. Quality: retained and
exceeded, indicatively on hyperspectral data against five unmatched dimensions, and under the tightest match in
the paper on PAD patches, where the recursive model leads on every patch-level metric. Cost: inverted, and
measured. The model that stores 8.2× less computes 191× more, needs 3.0× the peak inference memory, and cannot
train at all without gradient checkpointing.

### C. On comparison with MedMamba

MedMamba benchmarks itself on PAD-UFES-20 against four standard architectures at 224×224: ResNet50 at 56.62 %
overall accuracy, ConvNeXt-B at 54.73 %, ViT-B at 50.36 %, MedMamba at 58.80 %, and Swin-T [9] at 62.15 %. Two
things follow. First, MedMamba does not win its own table — Swin-T beats it by 3.4 points of accuracy and 7.1 of
F1 — which recalibrates the target: the interesting question for a MedMamba derivative is not only whether it
beats MedMamba, but whether the state-space family beats a well-tuned transformer on this data at all, and the
published answer on PAD-UFES-20 is currently no. Second, and decisively, those numbers are computed on whole
lesion images and our patch-level ones on 11×11 patches. No arithmetic between them is meaningful in either
direction, and Table VIII is the clearest demonstration of why.

What is different about the comparison this paper makes is its narrowness by construction. The backbone
substitution keeps the same output dictionary, the same head interface, the same reconstruction interface and the
same evaluation-mode signature, so both variants share a training pipeline, a metric suite and a checkpoint
format and swap with one flag. That is the condition needed for a controlled ablation, and it was far cheaper to
preserve than it would be to reconstruct later. What the paper does *not* have is the ablation itself: on
hyperspectral data the pair is not matched on training regime, and on PAD the hierarchical GMedMamba has not been
re-run at all. The mechanism for a clean comparison exists and has not been used.

### D. What the results support

The architectural claim is the one held with most confidence, because it does not depend on a comparison. The
spatial half of this model can be replaced by a single recursively applied core at 1.6 % of the parameters, and
the resulting model trains stably, converges, and produces a useful hyperspectral classifier with a working
auxiliary reconstruction objective. Nothing downstream had to change to accommodate it. The band-count claim is
held with equal confidence for the same reason: it is provable by construction and survives the absence of
ablations.

The empirical picture is genuinely mixed, and the mixture is informative rather than merely inconclusive. On
hyperspectral histology — the setting the spectral pathway exists for — the model reaches 90.7 % balanced accuracy
and 0.858 macro-F1 on 348,894 held-out patches, clears the best shallow probe comfortably, and finishes ahead of a
MedMamba baseline eight times its size on the identical evaluation set, though on a training split the two do not
share. On 11×11 RGB skin patches it barely clears a six-feature colour baseline — and so does the MedMamba
baseline, which saturates that training set within a single epoch — though under a matched recipe it leads on
every patch-level metric, and aggregating either model's predictions to the clinical image moves the numbers by
several points in ways that depend on the configuration.

Read together, those say something more specific than "it works" or "it does not": **the architecture does its job
where the input carries the information the label refers to, and the PAD patch protocol asks it to classify
something a single patch mostly does not contain.**

The most reusable finding is the cost model. The intuition that fewer parameters means a cheaper model is wrong
for this class of architecture, in a way that is easy to verify and expensive to discover accidentally. Recursion
buys parameter efficiency by spending compute and activation memory, and reporting parameter count alone for a
recursive model is closer to misleading than to incomplete.

### E. Strengths

Validity gates are inspectable artefacts rather than assertions: patient-disjoint splits with enumerated
intersections, content-hash leakage checks over 50,000 patches per split, checkpoint reproducibility verified to
delta 0.0, representation-sensitivity gates that abort a run producing constant logits, and deep integrity
validation of every memory-mapped array. Normalization statistics are fitted on the training split only. The
hyperspectral comparison uses an identical test set with identical class supports; the PAD patch comparison
matches twelve training dimensions. Shallow-probe floors exist on both datasets and one of them is unflattering,
which is the point of having them. The equivalence margin was fixed before the numbers entered the prose, and cost
is reported in the same section as capacity.

### F. Weaknesses and sources of bias

**Every run is a single seed.** There is no variance estimate anywhere, so no difference reported here can be
tested against run noise. The whole-image replicate puts a number on how bad that can be: 8.1 points of test
accuracy and 0.080 macro-F1 between two runs from a byte-identical command line at the same seed, on a 344-image
split. That spread is larger than most of the between-model differences the paper discusses.

**The hyperspectral test set is five patients.** The 348,894 test patches are not 348,894 independent
observations; they are a dense sampling of 71 captures from five people, and patches within a capture share a
label they inherited. Per-patient macro recall on validation ranges from 0.64 to 1.00 (Table V).

**No fully controlled baseline exists on either dataset.** The hyperspectral pair is unmatched on training-split
build, data fraction, class weighting, weight averaging and augmentation. The PAD patch pair is much tighter but
still differs in gradient clipping, schedule axis, early-stopping patience and classifier dropout. The
hierarchical GMedMamba has not been re-run under the current pipeline at all, which means the 61× capacity ratio
— the paper's largest — is stated against a variant with no current accuracy number.

**The accuracy half of the ablation ladder is missing entirely**, so every recursion constant is an unvalidated
inherited choice, and Table IX confounds EMA with augmentation.

**One hardware target**, so the compute-bound conclusions of Section VI-F may shift on hardware with different
kernel-launch overhead — and the 191×-FLOPs-to-9.8×-latency gap shows how much slack that overhead currently
hides.

**Selective-scan FLOPs are analytical, not instrumented.** They are 0.3 % of our total and 21 % of MedMamba's, so
no conclusion turns on them, but they are estimates.

**And the headline hyperspectral run cannot be independently verified**, because its artefacts were not supplied
and the directory bearing that name holds a superseded run whose numbers differ (87.56 % against 90.73 % balanced
accuracy). Everything else in this paper that could be checked, checked out exactly; this one could not be checked
at all.

### G. Statistical power

The effective n is the number of patients: **five** in the hyperspectral test split, 206 in the whole-image RGB
test split, 137 in the PAD patch test split. It is not 348,894, and treating the patch count as the sample size
would overstate the precision of every hyperspectral number by orders of magnitude.

Five patients permits a demonstration that the pipeline runs end to end, that the gates pass, that the model
clears its floor, and that it is competitive with a baseline. It does not permit an estimate of how these models
would perform on the next five patients, does not support a confidence interval worth printing, and does not
distinguish a genuine 0.030 macro-F1 difference from an accident of which five patients landed in the test split.
The parameter-invariance result is the one finding immune to this, because it does not depend on data at all.

### H. What is solved and what is not

Two things are solved. **Sensor coupling in the front end** is eliminated by construction: the parameter set is
independent of band count, exactly, and one preset serves both an 11×11×32 and an 11×11×3 input with no shape
change. And **the parameter cost of depth** is decoupled from depth itself — 63 applications of a 2-layer core at
a storage cost of one.

Several things are not. Throughput is worse by an order of magnitude in latency and two in arithmetic, so the
trade is favourable only where storage or distribution binds. Absolute accuracy on the RGB task is poor and is
constrained by evaluation unit and data volume rather than by the architecture. Whether the representation
transfers across band grids — the practical content of "agnostic" — is untested. And nothing here speaks to
clinical validity: patch-level metrics on five held-out patients from one sensor with one seed are a long way from
evidence about diagnosis.

## VIII. Limitations and required experiments

### A. Limitations

1. **No fully controlled baseline.** The hyperspectral comparison is matched on dataset, split, normalization,
   optimizer, batch size, precision, epoch budget, seed and checkpoint rule, but not on training-split build,
   data fraction, class weighting or weight averaging. The PAD patch comparison is much tighter — loss, class
   weighting, training-set size, objective, weight decay, normalization, weight averaging and augmentation all
   match — but gradient clipping, schedule axis, early-stopping patience and classifier dropout still differ, and
   every cell is one seed. The hierarchical GMedMamba has not been re-run under the current pipeline at all.
   **This is the binding limitation.**
2. **Single runs.** One seed per configuration, no variance estimate, no significance testing. A multi-seed
   harness exists and has not been used for these configurations. The whole-image replicate shows a single test
   cell moving 8.1 points of accuracy under an identical command line.
3. **Selection and test sets are small in subjects.** Both hyperspectral splits hold five patients, and
   per-patient macro recall ranges from 0.64 to 1.00 within the validation split alone. Test scores exceed
   validation scores by 0.086–0.100 macro-F1 for *both* models, which we read as the test patients being easier
   rather than as a model property — but with five subjects, neither split's aggregate is a precise estimate of
   population performance.
4. **Subsampled hyperspectral training.** The reported run uses 10 % of the training split per epoch, resampled
   each epoch. It is converged with respect to its own schedule, not trained on all available data.
5. **No reconstruction ablation.** The auxiliary objective is shown to work and to be measurable; it is not shown
   to help classification.
6. **Evaluation-unit mismatch on PAD.** Patch labels are inherited from whole clinical images. Table VIII
   mitigates this at scoring time; it does not fix the training objective. The whole-image protocol that would
   remove the mismatch at source trained on a directory undersampled to 234 of 1,626 images and is not matched to
   its baseline on split level, loss or augmentation, so it decomposes a gap rather than measuring one.
7. **Selective-scan FLOPs are analytical, not measured**, and the in-run FLOP profiler is broken
   (`ValueError: not enough values to unpack`) in every supplied `test_report.json`.
8. **Single hardware target.** All measurements come from one GPU, and the compute-bound conclusions may shift on
   hardware with different kernel-launch overhead.
9. **`ss2d` mixer is not currently usable at scale**, pending a compiled scan kernel, so the paper's default mixer
   is a practical choice rather than an evaluated best. No matched-accuracy comparison between mixers exists.
10. **The headline hyperspectral run is not independently verifiable.** Its artefacts were not supplied, and the
    directory bearing that name holds a superseded pre-fix run reporting 87.56 % balanced accuracy and 0.8535
    macro-F1 against the 90.73 % and 0.8580 stated. Three of the four supplied runs reproduce their reported
    figures exactly; this one is the exception.
11. **Behavioural band-count agnosticism is untested**, as is the wavelength-versus-index encoding that is
    supposed to deliver it. The claim in this paper is structural only.

### B. Required experiments

Each open prediction is paired with the study that would close it, written so it could be run.

**A. RQ0 — the same-acquisition spectral comparison.** Render RGB from the HistologyHSI-BC cubes themselves and
train the identical model on both the 32-band cubes and their 3-band renderings, with everything else fixed, over
five seeds. This makes the spectral axis the only variable, which is what RQ0 requires and what a cross-dataset
comparison can never provide. Full specification in the Section VI-A missing-work block.

**B. Matched three-way comparison.** Official MedMamba, hierarchical GMedMamba and GMedMamba-R on one frozen
pipeline: same split, preprocessing, resolution, augmentation, optimizer, schedule, batch size, duration,
checkpoint rule and evaluation code, with the auxiliary objective either enabled everywhere or disabled
everywhere, the same training-data fraction, the same class weighting, and either EMA everywhere or nowhere.
Section VI-C is this experiment with four variables loose — training-split build, data fraction, class weighting,
and weight averaging together with augmentation. **Re-running GMedMamba-R on the class-balanced build with
everything else held at the Section V settings is a single run and would settle whether Table VI's margins
survive.** On PAD the two-model half is already run (Table IX); what it lacks is the third leg.

**C. Five seeds per configuration**, reported as mean ± standard deviation with a paired test across matched
seeds. Given the whole-image replicate's 8.1-point spread, this is not optional for any claim resting on a
single test cell.

**D. Band-count sweep and encoding ablation** — the two RQ1 experiments, specified in full in the Section VI-B
missing-work blocks. The zero-retraining transfer test (train at 32 bands, evaluate at 16) is the highest-value
single item there, because it tests the property the wavelength encoding exists to provide.

**E. Train the hyperspectral model on the full training split** rather than a 10 % resample, to separate the
converged-schedule result from the converged-data result.

**F. Image-level multiple-instance learning on PAD**, with the loss defined at image level rather than only the
scoring, and per-lesion metrics reported as primary. The whole-image first pass must be repeated on the full
1,626-image training split rather than the undersampled 234, against a MedMamba baseline trained on the same
patient-disjoint directory rather than an image-level split of its own, with the checkpoint rule and loss matched
or else both metrics reported on both sides, over at least three seeds per cell. The tiled variants needed for
true MIL — 112×112 tiles at stride 56 or 112 with image-level aggregation of tile posteriors — are prepared and
unrun.

**G. Ablation ladder**, one variable at a time, specified in full in the Section VI-H missing-work block. Only
this separates "spectral processing helps" from "recursion helps" from "more parameters help", and item (g) there
— separating EMA from augmentation — is the one the paper most immediately needs.

**H. Extend the efficiency panel to the hierarchical GMedMamba** once it has been re-run, so all three
architectures can be compared on parameters, FLOPs, memory, latency and throughput at once, and repair the in-run
FLOP profiler so every future run carries its own count.

---

## IX. Conclusion

We described GMedMamba, a spectral–spatial extension of MedMamba whose parameter set is independent of the number
of spectral bands, and GMedMamba-R, a variant in which the hierarchical spatial backbone is replaced by a single
weight-shared core applied recursively in the manner of the Tiny Recursive Model.

The band-count claim is exact rather than approximate. Instantiating the identical model for 32-band and 3-band
input gives 446,409 and 446,796 trainable parameters — a difference of 387, precisely the classifier head's growth
from three to six outputs, and nothing attributable to the sensor. One preset serves both an 11×11×32 hyperspectral
patch and an 11×11×3 RGB patch with no shape change and no per-dataset tuning. This is a property of how the
parameters are laid out, proved by construction; whether the *representation* transfers across band grids is a
different question, and one we did not test.

The recursive substitution reduces trainable parameters from 27.43 M to 0.447 M, a factor of 61, while preserving
the spectral front end, the classification interface and the reconstruction pathway. On hyperspectral breast
histology it reaches 90.7 % balanced accuracy and 0.858 macro-F1 on 348,894 patches from five patients used for
neither training nor selection, clearing the strongest shallow probe by 7.1 points of balanced accuracy and
finishing ahead of a MedMamba baseline 8.2× its size on the identical test patches — though on a training split
the two do not share, which is why we read that as competitiveness rather than superiority. On PAD-UFES-20
patches, under the tightest comparison in the paper, it leads that baseline on every patch-level metric while
remaining, like it, only a little above a six-feature colour probe; aggregating patch posteriors to the clinical
image moves both models by several points and can reverse a ranking, locating much of that difficulty in the
evaluation unit rather than in either architecture.

The paper's firmest result is a negative one, and it is now quantified. Weight sharing reduces storage, not work:
the recursive core is applied 63 times per forward pass — at inference as well as in training — requires gradient
checkpointing to run in 16 GB at all, and under one plausible-looking mixer setting ran 39× slower than under
another. Set against the MedMamba baseline, GMedMamba-R carries 8.2× fewer parameters and 191× more FLOPs, 97.5 %
of them in the shared core. **Parameter count, for this family of models, is not a proxy for cost; it is close to
an inverse indicator of it.**

The defensible position is therefore this. GMedMamba-R delivers a working spectral–spatial classifier at 1.6 % of
the hierarchical variant's parameters, with a front end whose size is provably independent of the sensor; on
held-out hyperspectral data it clearly outperforms the best shallow model we could fit to the same patches and is
competitive with a MedMamba baseline 8.2× its size — at a compute cost that is now measured rather than assumed,
and one that MedMamba wins decisively. Parity or better at a fraction of the parameters is the result we claim.
Whether the architecture is genuinely *better*, rather than level under comparisons with loose training-side
variables and a single seed, remains open, and Section VIII specifies how to answer it.

---

## Acknowledgment

Portions of this manuscript were prepared with the assistance of an AI language model, used for source-code
analysis, verification of reported figures against run artefacts, drafting and editing, and formatting of tables.
All architectural claims, measurements and conclusions derive from the supplied source code, configuration files
and experiment outputs, and were reviewed by the author for technical accuracy. No experimental data, metric or
result was generated or altered by that process. The author takes full responsibility for the content and
conclusions of this manuscript.

---

## References

[1] A. Dosovitskiy *et al.*, "An image is worth 16×16 words: Transformers for image recognition at scale," in
*Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021. [Online]. Available: <https://arxiv.org/abs/2010.11929>

[2] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," arXiv:2312.00752, 2023.
[Online]. Available: <https://arxiv.org/abs/2312.00752>

[3] L. Zhu *et al.*, "Vision Mamba: Efficient visual representation learning with bidirectional state space
model," arXiv:2401.09417, 2024. [Online]. Available: <https://arxiv.org/abs/2401.09417>

[4] Y. Liu *et al.*, "VMamba: Visual state space model," arXiv:2401.10166, 2024. [Online]. Available:
<https://arxiv.org/abs/2401.10166>

[5] Y. Yue and Z. Li, "MedMamba: Vision Mamba for medical image classification," arXiv:2403.03849, 2024.
[Online]. Available: <https://arxiv.org/abs/2403.03849>

[6] Y. Yue and Z. Li, "MedMamba," GitHub repository, 2024. [Online]. Available:
<https://github.com/YubiaoYue/MedMamba> [Accessed: XX-XXX-2026]

[7] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf.
Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778. [Online]. Available: <https://arxiv.org/abs/1512.03385>

[8] A. Jolicoeur-Martineau, "Less is more: Recursive reasoning with tiny networks," arXiv:2510.04871, Oct. 2025.
[Online]. Available: <https://arxiv.org/abs/2510.04871> — code:
<https://github.com/SamsungSAILMontreal/TinyRecursiveModels>

[9] Z. Liu *et al.*, "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc.
IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 10012–10022. [Online]. Available:
<https://arxiv.org/abs/2103.14030>

[10] A. G. C. Pacheco *et al.*, "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images
collected from smartphones," *Data in Brief*, vol. 32, art. no. 106221, Oct. 2020, doi:
10.1016/j.dib.2020.106221. [Online]. Available: <https://doi.org/10.1016/j.dib.2020.106221> — dataset:
<https://data.mendeley.com/datasets/zr7vgbcyr2/1> [Accessed: XX-XXX-2026]

[11] L. Quintana-Quintana *et al.*, "Recurrent breast cancer: Histopathological and hyperspectral images database
(HistologyHSI-BC-Recurrence)," version 1, dataset, The Cancer Imaging Archive, 2025, doi: 10.7937/6KPY-YT49.
[Online]. Available: <https://doi.org/10.7937/6KPY-YT49> [Accessed: XX-XXX-2026]

[12] L. Quintana-Quintana *et al.*, "Histological hyperspectral breast cancer recurrence database (HistologyHSI-BC
Recurrence)," *Scientific Data*, vol. 12, art. no. 1886, 2025, doi: 10.1038/s41597-025-06157-4. [Online].
Available: <https://doi.org/10.1038/s41597-025-06157-4>

<MISSING_START>
**What is missing:** four citations the manuscript relies on but does not carry, and the access dates on three
web-only sources.
**Prompt:** "Add IEEE-format entries for: (a) scikit-learn, cited for the logistic-regression shallow probes of
Sections V and VI-C/VI-D; (b) PyTorch, cited for the framework and the version numbers in Section V; (c) the
`mamba_ssm` / selective-scan CUDA kernel used by the MedMamba baselines and referenced in the Section VI-F
discussion of the pure-PyTorch fallback; (d) a multiple-instance-learning reference supporting the aggregation
argument of Section VI-D and required experiment F. Then replace the three `[Accessed: XX-XXX-2026]` placeholders
in [6], [10] and [11] with real access dates."
**How to obtain it:** (a) cite Pedregosa et al., JMLR 12 (2011); (b) cite Paszke et al., NeurIPS 2019; (c) cite Gu
and Dao [2] plus the `state-spaces/mamba` repository for the kernel specifically; (d) Ilse et al., "Attention-based
deep multiple instance learning," ICML 2018, is the standard choice and matches the mean-pooling aggregation used
in Table VIII. A reviewer is most likely to ask for (a) and (d).
<MISSING_END>

> **Note on [11] and [12].** TCIA's terms ask that the collection DOI and its data descriptor be cited together,
> which is why both appear. If the venue's reference limit is tight, [12] is the one to keep — it carries the
> acquisition details (Hyperspec VNIR pushbroom camera, 400–1000 nm) that Section V relies on.
>
## Figure status

The manuscript references six figures under `figures/`. None is present in this working tree, so no figure is
referenced above as existing. Status per figure:

| ID | Content | File | Status |
| --- | --- | --- | --- |
| Fig. 1 | MedMamba: hierarchical backbone and SS-Conv-SSM block | `fig1_medmamba.svg` | referenced by the manuscript; not in tree |
| Fig. 2 | GMedMamba: spectral pathway conditioning four spatial stages | `fig2_gmedmamba.svg` | referenced; not in tree |
| Fig. 3 | Recursive core: states, improvement step, segment loop, gradient flow | `fig3_recursive_core.svg` | referenced; not in tree. **Annotates model-file defaults, not run values** — see the missing-work block in Section III-F |
| Fig. 4 | Capacity against compute | `fig4_cost.svg` | referenced; not in tree. Data now available in Tables I, XII, XIII |
| Fig. 5 | Mamba lineage and parameter scale | `fig6_lineage.svg` | referenced; not in tree. Data in Table XVIII |
| Fig. 6 | The controlled comparison specified but not run | `fig5_protocol.svg` | referenced; not in tree |
| **Fig. 7** | Band-count sweep with the shallow floor as a horizontal line | — | **Cannot be built** — no band-sweep runs exist (Section VI-B) |
| **Fig. 8** | Patch-level against image-level ROC / confusion pair | — | **Buildable now** — Table VIII has the data; see below |

<MISSING_START>
**What is missing:** the figure files themselves, and two figures that the data now supports but that have not
been drawn.
**Prompt:** "Supply the six SVGs under `figures/` referenced by Sections III and VII, or regenerate them. Then
draw two new figures. **Fig. 4** (capacity against compute): a two-axis plot with trainable parameters on one axis
and total FLOPs on the other, plotting GMedMamba-R (0.446 M, 6,203 M FLOPs) against MedMamba-HSI (3.65 M, 33 M),
annotated with the 0.12× and 191× ratios so the inversion is visible in one glance — this is the paper's thesis
and currently exists only as Table XII. **Fig. 8** (evaluation unit): paired confusion matrices or ROC curves for
the same PAD test predictions scored at patch level (88,400) and at clinical-image level (221), for both
GMedMamba-R and MedMamba, so the 9.9-point balanced-accuracy shift and the ranking change in Table VIII are
visible rather than tabulated."
**How to obtain it:** Fig. 4 needs only the numbers already in Tables I and XII. Fig. 8 needs the PAD patch-level
`test_predictions.npz` plus the patch→image index used for aggregation — the same inputs that produced Table VIII —
which are part of the artefact request in the Section VI-D missing-work block. Fig. 7 remains blocked on the band
sweep and cannot be drawn until those runs exist. When redrawing Fig. 3, apply the correction in Section III-F.
<MISSING_END>

## Appendix A — Code-to-architecture traceability

Every architectural claim in Section III corresponds to a named construct in `models/gmedmamba.py` (line numbers
verified against the supplied source) or a named module in the training package. This matters specifically because
an earlier revision contained a claim — the reconstruction attachment — that the figure asserted and the code did
not support.

| Architectural element | Where it lives | Line | What it is |
| --- | --- | ---: | --- |
| Configuration surface | `GMedMambaConfig` | 172 | One dataclass covering both variants; recursive fields inert unless the recursive flag is set |
| Spectral tokenizer | `SpectralTokenizer` | 724 | One shared `Linear(1 → d)` over band values — the source of parameter invariance |
| Index positional encoding (fallback) | `index_positional_encoding` | 674 | Band-index encoding; the A-PE control level |
| Sensor-aware nm normalization | `normalize_wavelengths` | 685 | Normalizes physical centres against a reference range |
| Continuous wavelength encoding | `continuous_wavelength_encoding` | 700 | Sinusoidal in normalized nm, not in index |
| Spectral encoder | `HierarchicalSpectralEncoder`, `ResidualSpectralBlock` | 919, 880 | Residual 1-D convolutions along the band axis |
| Long-range spectral dependence | `SpectralMamba` | 644 | State-space module mid-stack |
| Band gating | `BandGate`, `StageBandSelector` | 971, 991 | Learned per-band weight, recomputed per stage |
| Spectral compression | `ProgressiveCompressor` | 951 | MLP stack pooled over bands into a spatial context map |
| Whole spectral pathway | `SpectralPathway` | 1015 | Chunked over patches to bound peak memory; shared unchanged by both variants |
| 4-direction 2-D selective scan | `SS2D` | 509 | The scan primitive inherited from VMamba |
| Spectral–spatial fusion | `FiLMFusion` and five siblings, via `make_fusion` | 1117, 1196 | FiLM is the default; selectable |
| Spatial→spectral feedback | `SpectralContextUpdater` | 1212 | Runs after each stage's blocks |
| Hierarchical block / stage / merge | `GBlock`, `GStage`, `PatchMerging2D`, `channel_shuffle` | 1269, 1327, 1251, 1243 | SS-Conv-SSM plus fusion, LayerScale, DropPath, gated FFN |
| Hierarchical backbone | `GMedMambaBackbone` → `GMedMamba` | 1372, 1528 | Returns a dictionary so heads attach without knowing the backbone |
| 2-D sinusoidal stem encoding | `sinusoidal_2d_encoding` | 1570 | Added once to the fixed input embedding |
| Gated MLP in the core | `GatedMLP` | 422 | Second sublayer of each core block |
| Post-norm residual normalization | `RMSNorm` | 358 | `h ← RMSNorm(h + sublayer(h))` |
| **Recursive core** | `RecursiveCore` | 1718 | The shared function `f`: two layers of mixer + gated MLP |
| Core layers = 2 | `trm_core_layers` | 260 | |
| Latent updates = 6 | `trm_n_latent` | 261 | |
| Improvement steps = 3, first `T−1` under `no_grad` | `trm_n_improve`; `no_grad` loop | 262, 1784 | |
| Deep-supervision segments (file default 4, **run 3**) | `trm_deep_supervision_steps` | 266 | |
| ACT halting (file default on, **disabled in all runs**) | `trm_act_halting` | 267 | |
| EMA (file default 0.999, **run 0.995 HSI / disabled RGB whole-image**) | `trm_ema_rate` | 268 | |
| Gradient checkpointing of every core application | `trm_checkpoint_core`; `torch_checkpoint.checkpoint` | 283, 1765 | Not optional — see Table XV |
| Segment loop with inter-segment detach | `forward_deep_supervision`, `run_segments` | 1942, 1896 | |
| Classification head shared by both variants | `ClassificationHead` | 1466 | |
| Live feature extraction | `training/recursive_features.py` | — | Returns the live final feature map as a value, so the reconstruction gradient reaches the core without breaking EMA |
| Reconstruction decoder | `training/reconstruction_head_v2.py` | — | Normalization-matched output activation; refuses to start on a mismatch |
| Reconstruction metrics | `training/spectral_recon_metrics_v2.py`, `training/normalization.py` | — | Computed in reflectance units after inverting normalization |
| Preflight gates | `training/gates_v16.py` | — | Reconstruction-gradient, split-drift and sensitivity gates |
| Training entry point | `train_example_v16.py` | — | One flag switches the backbone |
| Trainer | `training/trainerg_v12.py` | — | Shared decode path, per-patient and per-capture reporting, step-axis scheduling, image-level aggregation |
| Verification | `test_trm_integration.py`, `test_recursive_features_parity.py`, `test_recon_gradient.py` | — | Assert parameter budget, forward/backward across channel counts, gradient coverage across deep supervision, EMA behaviour |

Core applications per forward = `(trm_n_latent + 1) × trm_n_improve × trm_deep_supervision_steps` =
(6 + 1) × 3 × 3 = **63**, of which 21 carry gradient (one improvement step per segment).

---

## Appendix B — Reproducibility

### B.1 Run identity — read this before opening any run directory

**The supplied directory names do not identify their contents.** Identity was established from each `config.json`
(`architecture`, `trm_*`, `dims`/`depths`, `entry_point`), not from the folder name.

| Directory | Actual model | Dataset | Bands | Classes | Verified against manuscript |
| --- | --- | --- | --- | --- | --- |
| `hsi_runs/medmamba/` | **GMedMamba-R (recursive)**, pre-fix run | HistologyHSI-BC `hsi_v7-80_10_10_importance` | 32 | 3 | **No — superseded run, see B.2** |
| `hsi_runs/gmedmamba/` | **MedMamba-HSI (hierarchical baseline)** | `hsi_v8-…_undersample` | 32 | 3 | **Yes, exactly** |
| `rgb_runs/gmedmamba/` | **GMedMamba-R (recursive)**, whole-image | PAD-UFES-20 `pad_optimal-undersample` | 3 | 6 | **Yes, exactly** |
| `rgb_runs/medmamba/` | **MedMamba-T (reference baseline)**, whole-image | PAD-UFES-20-paper | 3 | 6 | **Yes, exactly** |

Run identifiers: `20260903_180812_80_10_10_importance-hsi_recursive_hsi_mlp_sup3_bs256_bf16_wce_zscore_sub1_vsub1`;
`20260905_051406_hsi_v8-80_10_10_importance_undersample_medmamba_hsi_bs256_bf16`;
`20260906_183513_pad_optimal-undersample_recursive_rgb_mlp_sup3_bs32_bf16_focalw_zscore_optimal`; and
`rgb_runs/medmamba` (no `run_name` field recorded).

### B.2 The verification audit

| Manuscript object | Artefact | Result |
| --- | --- | --- |
| Table VI, MedMamba column | `hsi_runs/gmedmamba/` | **Exact.** Test 93.30 % / 86.76 % / 0.8278; per-class F1 0.846 / 0.637 / ~1.000; best epoch 3 of 30; median 155.1 s/epoch; peak 554.78 MB; final losses 0.0004 / 1.405 |
| Table XI, GMedMamba-R column | `rgb_runs/gmedmamba/` | **Exact.** 234/328/344; best epoch 104; best val accuracy 42.99 % at epoch 89; train accuracy 57.7 %; test 45.64 % / 50.20 % / 0.4213 |
| Table XI, MedMamba-T column | `rgb_runs/medmamba/` | **Exact.** 1,378/229/691; 57 epochs; best val accuracy epoch 37 = 54.585 %; best val macro-F1 0.4416 at epoch 32; train accuracy 99.2 %; test 47.18 % / 33.87 % / 0.3353 |
| Table XI-B, our row | `rgb_runs/gmedmamba/test_report.json` | **Exact.** Macro P 40.29 %, sensitivity 50.20 %, specificity 88.65 % (recomputed), F1 42.13 %, OA 45.64 %, AUC 0.7934; all six per-class F1 |
| Tables II–V, GMedMamba-R HSI | *none supplied* | **Conflict.** `hsi_runs/medmamba/` is a 7-epoch run selecting epoch 1 at 94.639 % / 87.556 % / 0.8535, κ 0.8849, MCC 0.8858, ROC-AUC 0.9872, PR-AUC 0.8696, ECE 0.0540, 14.65 ms, 625.2 patch/s — against the reported 12-epoch, epoch-9 run at 94.36 % / 90.73 % / 0.8580, ROC-AUC 0.993, PR-AUC 0.945, ECE 0.0038, 13.9 ms, 1,099 patch/s. Its `history.csv` shows epoch-axis LR decay, the defect Section IV-B describes fixing, so it is the superseded pre-fix run |

Parameter counts verified directly: 446,409 (32 bands, 3 classes) and 446,796 (3 bands, 6 classes), difference 387;
3,648,995 for MedMamba-HSI.

### B.3 Environment and seeding

PyTorch 2.10.0+cu128 and 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 5060 Ti (16.7 GB), 32 GB system RAM. All
timings and memory figures come from this machine. `seed = 42` in all runs, applied to Python `random`, NumPy,
torch and all CUDA devices; `deterministic = false`, `cudnn_benchmark = false`. **No configuration was repeated
under a second seed.**

### B.4 Artefact availability

| Item | Status |
| --- | --- |
| Model source, both variants; training entry point with one architecture flag | Available |
| Dataset preparation scripts, resumable and manifest-checked | Available |
| Per-run config, history, per-class metrics, confusion matrices | Available |
| Class-coverage, leakage, integrity and split-drift reports | Available |
| Checkpoint-reproducibility verification (G5, delta 0.0 both recursive runs) | Available |
| Representation-sensitivity gates (G1/G2/G9) with reported margins | Available |
| Parameter counts, both variants, both modalities | Available |
| Held-out test evaluation, hyperspectral baseline and both whole-image RGB runs | Available |
| Efficiency panel — parameters, FLOPs, memory, latency, throughput, two models | Available (Table XII) |
| **Hyperspectral GMedMamba-R headline run artefacts** | **No — see B.2** |
| **PAD patch-level run artefacts (Tables VII–X)** | **No** |
| **Shallow-probe fits and scoring scripts, both datasets** | **No** |
| **Reconstruction-run artefacts (Table XVI)** | **No** |
| **In-run FLOP profiling** | **Broken** — `ValueError: not enough values to unpack (expected 4, got 3)` in every supplied `test_report.json` |
| **Split-drift report for the hyperspectral recursive run** | **No** |
| Fully controlled MedMamba comparison (matched data fraction, weighting, EMA, objective) | **No** |
| Hierarchical GMedMamba re-run under the current pipeline | **No** |
| Multiple seeds / significance testing | **No** |
| Reconstruction ablation; completed ablation ladder | **No** |

### B.5 Known artefact defects

1. Directory names invert the architectures they contain (B.1).
2. The hyperspectral headline run is not the supplied one (B.2).
3. `hsi_runs/gmedmamba/per_capture_metrics.csv` holds 5 rows against the 71 test captures recorded in the split
   report, and four captures keyed `Healthy_HS_VNIR_<patient>_Healthy_*` carry `true_class = IDC`. The file is
   internally inconsistent and no result in this paper uses it.
4. In-run FLOP profiling fails in every supplied run.
5. The supplied hyperspectral recursive run records one epoch at 12,014 s against a 1,129–1,675 s range for its
   others, consistent with an I/O stall; medians rather than means should be used for any timing drawn from it.

---

## Appendix C — Terminology and claim policy

### C.1 Names

- **MedMamba** — the original architecture of Yue and Li [5], [6].
- **GMedMamba** — the hierarchical spectral–spatial extension developed in this work.
- **GMedMamba-R** — the recursive variant.
- **TRM** — the Tiny Recursive Model of Jolicoeur-Martineau [8], the source of the recursive formulation.
- **MedMamba-HSI (local baseline)** — a local copy of the MedMamba VSSM/SS2D backbone with an unmodified selective
  scan, adapted for 32-band input at patch size 1, at 3.65 M parameters. Used in Section VI-C and never cited as an
  authoritative reproduction of the published results.
- **MedMamba-T (reference PAD baseline)** — the unmodified MedMamba-T at 224×224 and 14.47 M parameters, trained by
  the reference script under the paper's own protocol (no augmentation, no pre-training, no class weighting,
  image-level split). A separate object from the local baseline; likewise not an authoritative reproduction.
- **Shallow probe** — a class-balanced multinomial logistic regression on hand-specified patch features, used as a
  floor.

### C.2 Words with a fixed meaning here

**Matched.** Two runs are *matched* on a dimension when it took the same value in both. A comparison is
**controlled** only when every one of dataset partition, test set, loss and class weighting, normalization,
augmentation, optimizer, learning rate, weight decay, batch size, epoch budget, weight averaging, precision and
seed is matched. **No comparison in this paper meets that bar.**

**Indicative.** A comparison with a non-empty unmatched list. The word appears in the same sentence as any such
number and the unmatched dimensions are named.

**Agnostic.** Used in two senses that are never merged. *Structurally agnostic* means the trainable parameter count
does not vary with band count — proved by construction and verified by measurement (Table I). *Behaviourally
agnostic* means quality degrades gracefully as bands are removed and does not collapse when band count changes at
test time — **not tested in this work.** An unqualified use means the structural sense.

**Equivalent.** Difference within the equivalence margin of 2.0 points, fixed in Section V before test numbers
entered the prose. Equivalence within an indicative comparison is evidence that a difference is small relative to a
stated margin under unmatched conditions, not that architectures are interchangeable.

**Verified.** The figure reproduces from a supplied run artefact. **Not supplied** means the figure rests on the
source manuscript alone and could not be independently checked.

**Significance.** No statistical significance is claimed anywhere. With one seed per run there is no variability
estimate, so differences are reported as single-seed point differences in the metric's own units, with direction,
and never as significant, robust or reliable.

### C.3 The claims this paper licenses

> "Instantiating the identical model for 32-band and 3-band input gives 446,409 and 446,796 trainable parameters,
> differing by 387 — exactly the classifier head's growth from three to six outputs. The parameter count of this
> architecture is independent of band count. This is a structural property, proved by construction and verified by
> measurement; behavioural agnosticism across band grids was not tested."

> "GMedMamba-R attains the classification behaviour reported here with 61× fewer parameters than the hierarchical
> variant, on a pipeline where the hierarchical variant has not been re-run."

> "On hyperspectral histology, GMedMamba-R reaches 90.7 % balanced accuracy and 0.858 macro-F1 on a held-out test
> split of 348,894 patches from five patients used for neither training nor model selection, and exceeds the
> strongest shallow probe fitted to the same patches by 7.1 points of balanced accuracy on the validation split.
> These figures rest on a run whose artefacts were not supplied for verification."

> "On the identical held-out hyperspectral test split, under matched batch size, precision, seed and checkpoint
> rule, GMedMamba-R leads a 3.65 M-parameter MedMamba baseline on macro-F1 (0.858 against 0.828), balanced accuracy
> (+4.0 points) and raw accuracy (+1.1). The two runs were trained on different builds of the training split and
> differ further in class weighting, weight averaging and augmentation. That training-distribution difference is
> sufficient to account for the raw-accuracy inversion on its own, so this is a competitiveness result at 8.2×
> fewer parameters and not a superiority claim."

> "On PAD-UFES-20 patches, under a comparison matching loss, class weighting, training-set size, auxiliary
> objective, weight decay, normalization, weight averaging and augmentation, GMedMamba-R leads a 3.65 M-parameter
> MedMamba baseline by 3.2 points of accuracy, 1.0 of balanced accuracy and 0.012 macro-F1, at 8.2× fewer
> parameters. Gradient clipping, schedule axis, early-stopping patience and classifier dropout remain unmatched and
> each cell is one seed, so this is a lead under a close match and not a controlled result."

> "An earlier revision reported that GMedMamba-R consistently trailed MedMamba on raw accuracy. That was caused by
> weight averaging and geometric augmentation, not by the architecture: with both removed the ordering inverts. The
> balanced-accuracy advantage survives the same ablation. The two factors were varied jointly, so which of them is
> responsible is not established."

> "On 11×11 RGB skin patches, GMedMamba-R exceeds a six-feature colour baseline by under two points, and
> aggregating its predictions to the clinical image recovers 9.9 points of balanced accuracy. Model ranking depends
> on the evaluation unit."

> "Recursion in this architecture reduces parameter count and increases compute and activation memory: 8.2× fewer
> parameters at 191× the FLOPs, 9.8× the batch-1 latency and mandatory gradient checkpointing. Both must be
> reported together."

> "No patch-level number in this paper is comparable to a published PAD-UFES-20 result, because the evaluation unit
> differs: 11×11 patches here, 224×224 whole images there."

> "A whole-image PAD-UFES-20 run exists and supports no comparison. Its 13-point validation-accuracy deficit
> against a reference MedMamba-T decomposes into a checkpoint-selection artefact, a training split undersampled to
> 234 of 1,626 images, an image-level baseline split against our patient-disjoint one, and a class-weighted
> objective against an unweighted one. The asymmetries that survive all four — a reversed balanced-accuracy and
> macro-F1 ordering on held-out test, and a raw-accuracy gap narrowing to 1.5 points — are observations about two
> uncontrolled runs, not a result."

> "Placed beside the published PAD-UFES-20 figures of [5], [6], a 0.447 M-parameter GMedMamba-R checkpoint on 344
> patient-disjoint whole images is above the published MedMamba row on macro precision, sensitivity and F1, above
> every row on sensitivity, and below every row on overall accuracy. This is a reference point and not a
> comparison: the published rows are scored on 691 images from an image-level split that leaks patients, ours on
> 344 from a patient-disjoint one, ours trained on 234 images against their 1,378, and a seed-identical replicate
> moves our test cells by up to 8.1 points."

> "This work does not claim clinical utility, does not claim state of the art, and does not claim a causal effect
> of spectral information on diagnosis. Whether hyperspectral input outperforms RGB was not tested, because the two
> arms are different diseases."

### C.4 Consistency policy for this document

This manuscript resolves one inconsistency inherited from its source. The source manuscript's Section IX quoted
84.8 % accuracy, 81.4 % balanced accuracy and 0.737 macro-F1 "over thirty converged epochs" for the hyperspectral
result, contradicting its own Abstract and Section VI-A (94.4 % / 90.7 % / 0.858 over twelve epochs, epoch 9
selected). Those figures are stale values from an earlier revision. The Section VI-A values are used throughout
here, and every occurrence in the Abstract, Results, Discussion and Conclusion carries the same value and the same
evaluation unit.
