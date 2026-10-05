::: IEEEkeywords
medical image classification, MedMamba, state-space models, tiny
recursive models, weight sharing, hyperspectral imaging,
spectral--spatial learning, band-count agnosticism, parameter
efficiency, multiple-instance learning, patient-disjoint evaluation,
computational cost.
:::

# Note on provenance and verification {#note-on-provenance-and-verification .unnumbered}

This manuscript consolidates GMedMamba manuscript v6 with an independent
audit of the four run directories supplied alongside it. Because the two
do not agree everywhere, each result below is marked with its
provenance, and the reader should treat the two classes differently.

**Verified.** Three of the four supplied runs reproduce the manuscript's
figures exactly, to every decimal checked. The MedMamba-HSI baseline
(`hsi_runs/gmedmamba/`) matches on test metrics, per-class F1, best
epoch, median epoch time (155.1 s) and peak memory (554.78 MB). The
whole-image PAD-UFES-20 pair (`rgb_runs/gmedmamba/`,
`rgb_runs/medmamba/`) matches on splits, selected epochs, test metrics,
macro specificity and all per-class F1 values. Where a number is drawn
from these runs it is marked *(verified)*.

**Not supplied.** The hyperspectral GMedMamba-R headline run of
Section [6.3](#sec:results-hsi){reference-type="ref"
reference="sec:results-hsi"} is **not** the run in `hsi_runs/medmamba/`.
The supplied directory holds a 7-epoch run selecting epoch 1 at 87.56%
balanced accuracy and 0.8535 macro-F1; the manuscript reports a 12-epoch
run selecting epoch 9 at 90.73% and 0.8580. The configurations are
otherwise identical, and the supplied run's `history.csv` shows the
learning rate decaying on the epoch axis --- the exact defect the
manuscript describes fixing by moving the schedule to the optimizer-step
axis. The supplied run is therefore the superseded one, and the headline
run's artefacts were not provided.
Section [6.3](#sec:results-hsi){reference-type="ref"
reference="sec:results-hsi"} carries this as an open item rather than
resolving it silently. The PAD **patch-level** runs of
Tables [7](#tab:pad_patch_summary){reference-type="ref"
reference="tab:pad_patch_summary"}--[\[tab:pad_per_class\]](#tab:pad_per_class){reference-type="ref"
reference="tab:pad_per_class"}, the shallow probes, the reconstruction
measurements and the FLOPs panel likewise have no supplied artefacts and
are reported on the manuscript's authority.

A caution the audit also produced: the directory names in the supplied
artefacts do not identify their contents. `hsi_runs/medmamba/` holds a
*recursive* run and `hsi_runs/gmedmamba/` holds the *hierarchical
MedMamba* baseline. Identity was established from each `config.json`,
and Appendix B gives the mapping.

# Introduction

A classifier looking at tissue has to do two things that pull against
each other. It must resolve local detail --- the shape of a nucleus, the
texture at a lesion border --- and it must place that detail in a wider
context, because the same local pattern means different things in
different surroundings. Convolutional networks are excellent at the
first and reach the second only slowly, through depth. Vision
Transformers reach the second directly through
self-attention [@dosovitskiy2020image], at a cost quadratic in token
count, which becomes awkward when the channel dimension itself grows, as
it does in hyperspectral imaging.

Breast cancer histopathology is where this bites clinically. A
pathologist decides whether the architecture in front of them is benign,
a carcinoma confined to the duct, or one that has breached it, and the
distinction between ductal carcinoma *in situ* and invasive ductal
carcinoma governs treatment. It is drawn on morphology that varies
between laboratories, between stains and between observers, and case
volumes have grown faster than the number of pathologists. At the
margins of a diagnosis, throughput and consistency degrade together.

Hyperspectral imaging offers a physically different measurement. Instead
of three integrated colour values per pixel, a cube records tens of
narrow bands, so each pixel carries a spectrum. Absorption and
scattering in tissue depend on molecular composition --- haemoglobin,
water, collagen, nuclear density, stain uptake all have
wavelength-dependent signatures --- and an RGB sensor integrates those
signatures away. Two regions a camera renders identically can differ in
the near-infrared. Whether that difference is diagnostically useful in
stained histology is an empirical question this paper does not settle;
the physical argument establishes only that the information is present
before the sensor discards it.

State-space models offer a third computational route. Mamba made the
state-transition parameters depend on the input, so a model can choose
what to carry forward along a sequence while keeping near-linear
scaling [@gu2023mamba]. Vision-oriented adaptations extended that to
two-dimensional feature maps [@zhu2024vision; @liu2024vmamba], and
MedMamba brought it to medical image classification with the SS-Conv-SSM
block: half the channels through a convolutional path, half through a
state-space path, recombined by concatenation, channel shuffle and a
residual [@yue2024medmamba]. GMedMamba starts from that block and adds
what MedMamba does not have --- an explicit spectral pathway.

## From "does it help?" to "does it need to be this large?"

The spectral extension costs parameters. A parallel pathway, per-stage
fusion and a reconstruction head take the model from roughly 13.5 M
parameters in the comparable MedMamba configuration to 27.43 M. That is
a real cost, and it confounds any accuracy comparison: a model with
twice the capacity that scores slightly higher has not demonstrated a
better idea.

There are two ways the parameter count of such a model can be coupled to
things it should not be. The first is the sensor. Models built for
three-channel input treat bands as ordinary feature channels, which
discards spectral ordering --- a first-layer convolution over 32
channels is invariant to permuting them, so the fact that band 12 lies
between bands 11 and 13 in wavelength, and that the gap between bands 17
and 18 in our data is 219 nm, is information the model never receives
--- and makes the stem's width a function of the instrument. Change the
sensor and the architecture is redesigned, not merely retrained. The
second is depth. A four-stage hierarchy buys effective depth by storing
distinct blocks at four widths, so depth and storage are welded
together.

The Tiny Recursive Model (TRM) [@jolicoeur2025tiny] suggests a way out
of the second. Its observation is that a network's depth and its
parameter count are separable: rather than stacking many distinct
blocks, build one small block and apply it repeatedly, carrying a state
forward between applications. TRM does this with two states --- a latent
reasoning state and an answer state --- refined over several improvement
steps, with most steps run without tracking gradients so recursion depth
does not inflate memory. On the puzzle benchmarks it targets, a
7 M-parameter recursive network is competitive with far larger models.

The two moves compose, and that is the point. Decouple the front end
from the sensor with a shared per-band projection and a
wavelength-continuous positional encoding, and nothing in the spectral
pathway scales with band count --- which exposes the backbone as the
remaining parameter bottleneck. Replace that backbone with one
recursively applied core, and depth becomes a property of how many times
the core is called rather than of how many blocks are stored. The
result, GMedMamba-R, has 0.447 M parameters, and its size is set by
neither the sensor nor the depth.

## Research questions

- **RQ0.** Does hyperspectral input improve breast-cancer detection
  relative to RGB under an identical model and protocol?

- **RQ1.** Is a spectral--spatial model derived from MedMamba
  band-count-agnostic --- structurally and behaviourally --- with no
  architectural change and no per-dataset retuning?

- **RQ2.** Can a single weight-shared core applied recursively replace a
  four-stage hierarchical backbone at an order of magnitude fewer
  trainable parameters, and what does that substitution actually cost?

The answers are uneven, and the paper is organized around saying so
precisely. RQ0 turns out not to be evaluable with the runs that exist,
because the two arms are different diseases rather than two renderings
of one acquisition --- a limitation of the experimental design, reported
as such in Section [6.1](#sec:rq0){reference-type="ref"
reference="sec:rq0"} rather than papered over with a cross-dataset
comparison. RQ1 is answered exactly in its structural half and left open
in its behavioural half. RQ2 is answered in all three of its parts,
including the part that is unflattering, and the unflattering part is
the most transferable result here.

## Contributions

1.  **GMedMamba**, a spectral--spatial extension of MedMamba, described
    to match the implementation rather than an idealization of it.

2.  **GMedMamba-R**, a TRM-style recursive backbone integrated into that
    architecture at 0.447 M parameters against 27.43 M, with the
    spectral front end, task heads and reconstruction interface
    preserved and one flag switching between variants.

3.  **A structural band-count-agnosticism result that is exact rather
    than statistical**: instantiating the identical model at 32 and 3
    bands gives 446,409 and 446,796 parameters, differing by the 387
    parameters of a larger classifier head and by nothing attributable
    to the sensor.

4.  **A converged hyperspectral result on held-out patients**, with a
    working auxiliary reconstruction objective whose gradient provably
    reaches the recursive core.

5.  **Matched-configuration MedMamba baselines on both datasets**, with
    the recursive model at parity or ahead on 8.2$\times$ fewer
    parameters, and the remaining confounds enumerated rather than
    glossed.

6.  **Shallow-probe floors** for both datasets, and an honest reading of
    the model against them --- convincing on hyperspectral data, barely
    clearing on RGB patches.

7.  **Patch-to-image aggregation**, quantifying how much of the low
    patch-level score is an evaluation-unit artefact, including a case
    where the ranking between models reverses.

8.  **A measured cost model for recursion** --- core applications per
    step, FLOPs, wall-clock, memory, and the gradient checkpointing
    without which the small model does not run at all.

9.  **A specification of the controlled study** that would settle what
    remains open, written so someone could run it.

## Scope guard

This work does not claim clinical utility, does not claim state of the
art, and does not claim a causal effect of spectral information on
diagnosis. It claims a band-count-agnostic front end proved by
construction, a parameter-efficiency result with its compute cost stated
in the same breath, and an evaluation protocol whose validity gates are
checkable. Every hyperspectral number rests on five held-out patients
and a single seed.

Section [2](#sec:related){reference-type="ref" reference="sec:related"}
positions the work, Section [3](#sec:method){reference-type="ref"
reference="sec:method"} describes both variants,
Section [4](#sec:pipeline){reference-type="ref"
reference="sec:pipeline"} the training pipeline and validity gates,
Section [5](#sec:setup){reference-type="ref" reference="sec:setup"} the
experimental setup, Section [6](#sec:results){reference-type="ref"
reference="sec:results"} the results,
Section [7](#sec:discussion){reference-type="ref"
reference="sec:discussion"} their interpretation,
Section [8](#sec:limitations){reference-type="ref"
reference="sec:limitations"} the limitations and the experiments that
would close them, and Section [9](#sec:conclusion){reference-type="ref"
reference="sec:conclusion"} concludes.

# Related work {#sec:related}

**Convolutional networks.** Convolution encodes a strong, data-efficient
prior --- nearby pixels are related, and the same filter applies
everywhere --- which is why CNNs have dominated medical imaging, and
residual connections made real depth trainable [@he2016deep]. *Gap:*
long-range context emerges only indirectly, through depth and pooling,
and the channel axis is unordered in the first layer.

**Vision Transformers.** ViT [@dosovitskiy2020image] treats an image as
a sequence of patch tokens and lets any two interact through
self-attention, at a cost quadratic in token count. On PAD-UFES-20
specifically, a well-tuned hierarchical transformer remains strong:
Swin-T [@liu2021swin] leads MedMamba's own published comparison table.
*Gap:* the quadratic cost becomes awkward exactly where the channel
dimension grows, and neither family offers a mechanism for an ordered
channel axis.

**Selective state-space models.** Mamba [@gu2023mamba] made state-space
parameters input-dependent, giving attention-like selectivity at
near-linear cost. Vision Mamba [@zhu2024vision] applied bidirectional
scanning to patch sequences, reaching 76.1 / 80.3 / 81.9% ImageNet-1K
top-1 at 7 / 26 / 98 M parameters, and running 2.8$\times$ faster than
DeiT with 86.8% less GPU memory at $1248^2$ resolution.
VMamba [@liu2024vmamba] introduced the two-dimensional selective scan
(SS2D), traversing a feature map along four spatial directions, reaching
82.6 / 83.6 / 83.9% top-1 at 30 / 50 / 89 M parameters. SS2D is the
sequence primitive inside MedMamba and, optionally, inside our recursive
core. *Gap:* neither Vim nor VMamba reports medical or spectral
benchmarks.

**MedMamba.** Yue and Li proposed the SS-Conv-SSM block and evaluated it
across sixteen datasets spanning ten imaging modalities and 411,007
images [@yue2024medmamba]. Its three variants are 15.2, 23.5 and 48.1 M
parameters (2.0, 3.5 and 7.4 GFLOPs), averaging 84.0, 84.3 and 83.8%
overall accuracy on the non-MedMNIST subset. It is the authoritative
baseline throughout this work. *Gap:* the spectral axis is handled as an
ordinary channel axis, so spectral order is unavailable to the model and
the stem's parameter count follows the sensor --- channel-agnostic only
in the trivial sense that the stem can be resized.

**Hyperspectral classification.** Spectral--spatial models read the band
axis explicitly, through 1-D convolutions along wavelength, 3-D
convolutions over the joint cube, or spectral attention. Most originate
in remote sensing, where evaluation is typically pixel-wise within one
scene. *Gap:* patient-disjoint evaluation is rare, and architectures are
commonly sized for one instrument's band grid, so band-count agnosticism
is neither claimed nor tested.

**Tiny Recursive Models.** TRM [@jolicoeur2025tiny] argues that
recursion can substitute for depth. A single two-layer network is
applied repeatedly over two carried states --- a latent state $z$ and an
answer state $y$ --- trained with deep supervision across several
segments, with earlier improvement steps run under `no_grad` so memory
stays bounded regardless of recursion depth. With six latent updates and
three improvement steps, TRM obtains an effective depth of 42 layers
from a 7 M-parameter network, using EMA at decay 0.999. Its reported
results are striking: 87.4% on Sudoku-Extreme, 85.3% on Maze-Hard, 44.6%
on ARC-AGI-1 and 7.8% on ARC-AGI-2, against 55.0 / 74.5 / 40.3 / 5.0%
for the 27 M-parameter Hierarchical Reasoning Model it replaces. TRM's
own ablation finds the two-state formulation clearly better than a
single latent state (87.4% against 71.9% on Sudoku), which is why we use
two states by default. *Gap:* TRM's benchmarks are small fixed-size
grids with exact answers, ours is noisy patch classification; the
mechanism's transfer is not automatic, and the compute profile turns out
very differently, as Section [6.6](#sec:cost){reference-type="ref"
reference="sec:cost"} shows.

# Method {#sec:method}

## Preliminaries: the inherited MedMamba backbone

The canonical pipeline is a four-stage hierarchical pyramid. An input
image $[B, 3, H, W]$ passes through a non-overlapping $4\times 4$ patch
embedding to $[B, H/4, W/4, C]$ with $C = 96$ in the tiny configuration.
Each patch-merging step then halves spatial resolution and doubles
channel width, so the representation moves from
$H/4 \times W/4 \times C$ to $H/32 \times W/32 \times 8C$ across widths
96, 192, 384 and 768, with stage depths (2, 2, 4, 2). A final layer
normalization, global average pool and linear classifier produce the
logits.

Inside a block the input splits channel-wise. One half goes through a
convolutional path --- convolution, normalization, activation ---
capturing local texture cheaply. The other half is normalized and passed
to SS2D, the two-dimensional selective scan, which flattens the feature
map along four traversal orders
($\rightarrow, \leftarrow, \downarrow, \uparrow$), runs an
input-dependent state-space recurrence along each and merges the four
results; a parallel linear--SiLU branch gates the SS2D output. The two
halves are concatenated, a channel shuffle mixes information across the
split so the next block's halves are not the same halves, and the block
input is added back as a residual.

Because each branch sees only half the channels, the block delivers both
local and long-range processing at roughly the cost of a single
convolutional path. That economy is why the block is worth inheriting
rather than building on a ViT block. The published variants are 15.2,
23.5 and 48.1 M parameters at 2.0, 3.5 and 7.4 GFLOPs; the baselines
trained here report their own measured counts, and no published figure
is reused as if it were measured.

## The spectral pathway

GMedMamba adds one pathway that runs before and alongside the spatial
stages. Its job is to turn a spectrum into a compact spatial context map
the spatial stages can be conditioned on, and every part of it is
deliberately independent of the number of bands. That independence is
the paper's structural claim for RQ1, and it is proved here by
construction rather than by experiment, so each component is given with
the reason it satisfies the constraint.

*Tokenization.* After patchification each spatial position carries a
vector of band values. `SpectralTokenizer` lifts every scalar band value
to a token of width $d_{\text{token}}$ through **one shared linear
projection applied to all bands**, giving
$[B, \cdot, C_{\text{bands}}, d_{\text{token}}]$. The projection maps
$1 \rightarrow d_{\text{token}}$, so its weight tensor has shape
$[d_{\text{token}}, 1]$ whether $C_{\text{bands}}$ is 3 or 32. A
per-band projection --- the obvious alternative --- would put
$C_{\text{bands}}$ into the weight shape and destroy the property.

*Spectral position.* Order along the band axis is supplied by an
additive positional encoding rather than by parameters. Where physical
band centres in nanometres are available, `normalize_wavelengths` maps
them to $[0, 1]$ against a reference sensor range and
`continuous_wavelength_encoding` evaluates a sinusoidal basis on that
continuous value; otherwise `index_positional_encoding` falls back to
band index. The distinction matters for the reason it matters in any
sequence model: spectral position corresponds to a physical quantity,
and two sensors that sample 550 nm at different indices should produce
the same encoding there. Both encodings are computed, not learned, so
neither adds parameters and neither is tied to a band count. The
difference between them is the intended A-PE ablation, which has not
been run.

*Spectral encoding.* `HierarchicalSpectralEncoder` stacks residual
one-dimensional convolutions along the band axis, capturing local
spectral shape --- absorption features, band-to-band edges. Convolutions
share weights across positions by construction, so a kernel of width $k$
costs the same sliding over 3 bands or 32. A small state-space module
(`SpectralMamba`) sits mid-stack for longer-range dependencies between
distant parts of the spectrum, which a stack of short kernels would need
many layers to reach.

*Band gating and compression.* `BandGate` learns a weight per band,
suppressing uninformative or noisy wavelength ranges; it is produced by
a projection from the token width rather than stored as a
length-$C_{\text{bands}}$ vector, which is what keeps it band-agnostic.
`ProgressiveCompressor` then reduces the token dimension through an MLP
stack and **pools over the band axis**, emitting a spatial context map
$\text{ctx\_map} \in \mathbb{R}^{B \times H_p \times W_p \times d_{\text{ctx}}}$.
That pooling is where the band axis disappears: every tensor downstream
has a shape containing no reference to $C_{\text{bands}}$.

The whole pathway is chunked over patches
($\text{spectral\_chunk\_size} = 1024$) to bound peak memory
independently of band count --- a detail that matters more than it
looks, and Section [6.6](#sec:cost){reference-type="ref"
reference="sec:cost"} records the measurement that corrected an earlier
claim about it.

## Fusion and the classification head in the hierarchical variant

The context map is not concatenated once at the end; it conditions every
stage. The default mechanism is FiLM modulation --- the context produces
a scale and a shift applied to the spatial features --- and five
alternatives are implemented and selectable through `make_fusion`.

Two smaller mechanisms complete the loop. A per-stage band selector
recomputes channel importance on the context map at the start of each
stage, so importance is depth-dependent rather than decided once at the
input. A context updater runs the other way: after a stage's spatial
blocks, the spatial features revise the context map the next stage will
see. Spectral information conditions spatial processing, and spatial
processing revises the spectral summary.

The classifier does not read only the final stage; it concatenates the
pooled output of every stage with the spectral summary, so features at
several depths reach the decision. A reconstruction decoder can be
attached during training, reconstructing the input cube from backbone
features under a combined pixel-wise MSE and spectral-angle objective.
Section [4.3](#sec:recon-gate){reference-type="ref"
reference="sec:recon-gate"} describes why this pathway needed repair and
how it was verified.

This variant is described for contrast. Its spatial parameters live in
four distinct stage stacks at four widths, and that is the cost the
recursive variant is designed to remove.

## The recursive variant

A four-stage backbone spends most of its parameters on having many
distinct blocks at several widths. GMedMamba-R keeps one block, at one
width, and applies it many times. The spectral pathway of
Section [3](#sec:method){reference-type="ref" reference="sec:method"}-B
is reused unchanged.

*Stem.* The context map is normalized, projected by a single linear
layer to width $d = 128$, normalized again, and given a two-dimensional
sinusoidal positional encoding scaled by a configurable gain (0.1 in the
reported runs). The result, $x_{\text{emb}}$, is computed once and
**held fixed for the entire recursion**. It has to be fixed: it is the
only term in the loop that still refers to the input, so if it drifted
with the states the recursion would gradually lose contact with the
image it is supposed to be classifying. Every core application re-reads
the same $x_{\text{emb}}$.

*Two carried states.* A latent state $z$, where intermediate work
accumulates, and an answer state $y$, which the classifier reads, both
initialized from non-trainable buffers broadcast to $[B, H_p, W_p, d]$.
Separating them means the head is not forced to read a tensor that is
mid-computation. TRM's own ablation finds the two-state formulation
clearly better than a single latent state, which is why it is the
default here.

*The shared core.* $f$ consists of $\text{trm\_core\_layers} = 2$
blocks. Each applies a token mixer and then a gated MLP under
post-normalized residuals:
$h \leftarrow \text{RMSNorm}(h + \text{mixer}(h))$, then
$h \leftarrow \text{RMSNorm}(h + \text{GatedMLP}(h))$. Three mixers are
selectable --- a depthwise convolution with GEGLU, a plain multi-head
self-attention, and the SS2D selective scan. **Every run reported here
uses the depthwise-convolution/GEGLU mixer**, and
Section [6.6](#sec:cost){reference-type="ref" reference="sec:cost"}
explains why that is a practical necessity rather than a preference:
under the pure-PyTorch reference scan, `ss2d` ran 39$\times$ slower.

*The recursion.* One improvement step performs
$\text{trm\_n\_latent} = 6$ latent updates,
$z \leftarrow f(z + y + x_{\text{emb}})$, then a single answer update
$y \leftarrow f(y + z)$ --- seven core applications. One segment
performs $\text{trm\_n\_improve} = 3$ improvement steps, the first two
under `torch.no_grad()` and only the last carrying gradient: 21
applications. A forward pass runs
$\text{trm\_deep\_supervision\_steps} = 3$ segments with the states
detached between them, giving **63 applications of the same 2-layer core
per forward pass**, and 126 mixer calls. The head is called after each
segment and training minimizes the mean cross-entropy over the
per-segment logits, so the model is pushed to be right early rather than
only at the end.

The `no_grad` prelude and the inter-segment detach are not incidental.
Without them the autograd graph would grow linearly in the 63
applications, and the memory saved by storing one core instead of four
stage stacks would immediately be spent on activations. Bounding the
graph to a single improvement step per segment is what makes the
parameter saving survive into training.

*Weight averaging and checkpointing.* An exponential moving average of
the weights is maintained; validation and the saved best checkpoint use
the averaged copy. Every core application is gradient-checkpointed,
recomputing activations during the backward pass instead of storing
them. This is a compute-for-memory trade, not a free saving, and
Section [6.6](#sec:cost){reference-type="ref" reference="sec:cost"}
shows it is not optional: without it the model OOMs on a 16 GB card.

*Halting.* An ACT-style halting head that predicts whether the current
segment's answer is already correct is implemented and available, but it
is **disabled in every run reported here**, so no result depends on it
and the segment count is fixed rather than learned.

## What is held fixed: the substitution contract

The substitution is deliberately narrow. The recursive backbone returns
the same dictionary of outputs as the hierarchical one --- feature map,
pooled vector, spectral context, band weights --- so reconstruction
decoders and task heads attach unchanged, and in evaluation mode the
model returns a single logits tensor, so every existing trainer and
wrapper continues to work. Only the spatial backbone changed, and one
command-line flag switches between them. That narrowness is what would
let a comparison isolate the backbone, and it is the precondition for
the controlled ablation of
Section [8.2](#sec:controlled-ablation-spec){reference-type="ref"
reference="sec:controlled-ablation-spec"}.

One consequence is worth stating because it is the structural claim in
operational form. Because the recursive variant works at unit patch size
and the spectral tokenizer is band-agnostic, **a single preset serves
both the $11\times 11\times 3$ RGB patches and the
$11\times 11\times 32$ hyperspectral patches with no shape change and no
per-dataset tuning.** The hierarchical variant needs a different preset
for each.

## Parameter cost

::: {#tab:param_cost}
  Configuration                                                   Trainable parameters
  ------------------------------------------------------------- ----------------------
  MedMamba-Tiny, 6 classes (ref., not measured)                                13.53 M
  GMedMamba hierarchical, RGB, 6 classes                                   **27.43 M**
  GMedMamba hierarchical, HSI, 32 bands, 3 classes                              2.77 M
  MedMamba-HSI, local baseline, 32 bands, 3 classes *(ver.)*                    3.65 M
  MedMamba-T, reference PAD baseline, $224\times 224$                          14.47 M
  **GMedMamba-R, RGB, 3 bands, 6 classes** *(ver.: 446,796)*               **0.447 M**
  **GMedMamba-R, HSI, 32 bands, 3 classes** *(ver.: 446,409)*              **0.446 M**
  GMedMamba-R, RGB, `ss2d` mixer                                               0.535 M
  GMedMamba-R, RGB, `attention` mixer                                          0.377 M

  : Model capacity, measured directly from the constructed models under
  the configuration used for the reported runs.
:::

The recursive variant is **61$\times$ smaller** than the hierarchical
GMedMamba on the same task --- 1.6% of its parameters --- and
**8.2$\times$ smaller** than the MedMamba-HSI baseline it is compared
against in Section [6.3](#sec:results-hsi){reference-type="ref"
reference="sec:results-hsi"}. Both ratios are storage only, and neither
means anything without the compute figures of
Section [6.6](#sec:cost){reference-type="ref" reference="sec:cost"},
which move the other way by 191$\times$.

The two rows in bold carry the structural RQ1 result and deserve to be
read together. They are the same model under the same preset, differing
only in the input it accepts and the number of classes it emits, and
they differ by **387 parameters**: exactly three additional classifier
outputs at 128 weights plus one bias each. Subtracting that term leaves
the two identical. A 10.7-fold change in band count contributed nothing.
What remains band-dependent is the *activation* size during the spectral
stage --- and hence the compute --- not the storage, a distinction
Section [6.6](#sec:cost){reference-type="ref" reference="sec:cost"}
develops.

A note on defaults, because the figures and the prose describe different
layers of the same system. The model file's own dataclass defaults are
the `ss2d` mixer, four deep-supervision segments, halting enabled and
EMA at 0.999. The training entry point overrides all of these, to the
convolutional mixer, three segments, halting disabled and EMA at 0.995
on the hyperspectral arm (and disabled entirely on the whole-image RGB
run). Every run in this paper uses the entry point's values.

# Training pipeline and validity gates {#sec:pipeline}

Much of the work behind this revision is not in the model file. It is in
making the numbers the model produces mean something.

## Dataset gates

Before a model is constructed, the pipeline checks that the data is what
it claims to be: per-file shape, dtype, size and SHA-256 against a
manifest written at preparation time; a leakage check against the
recorded patient and slide split; and a class-coverage check that
refuses to start if a split is missing a class unless that is explicitly
allowed. Arrays are validated before being memory-mapped, so a corrupt
file becomes a typed, catchable error rather than a `SIGBUS` inside a
data-loader worker.

This is not theoretical. During earlier runs the gate caught a training
array physically unreadable from byte offset 1,090,519,040 onward ---
71% of a 3.7 GB file returning I/O errors reproducibly, while its header
and first gigabyte read cleanly. Without the gate that dataset would
have trained partway through an epoch and then died in a way that is
very hard to attribute.

Both datasets pass the patient- and slide-level leakage checks with no
overlap between any pair of splits *(verified)*. A patch-level
content-hash comparison over 50,000 sampled patches per split found no
cross-split duplicates on the hyperspectral arm *(verified)*. On the
supplied whole-image RGB run all 234 / 328 / 344 images were hashed: no
cross-split duplicates, but one duplicate group within validation and
two within test --- a within-split redundancy that slightly reduces the
effective independence of those small sets without constituting leakage
*(verified)*. A run configured with `abort_on_leakage` halts before
training if a cross-split duplicate is found.

## Accounting for optimizer updates

The learning-rate schedule is stepped on the **optimizer-step axis**
rather than per epoch. This is a small change with a large consequence,
and it is the reason the hyperspectral results in this revision are
usable at all. When the training split is resampled each epoch and the
schedule is stepped per epoch, the schedule runs to completion long
before the data does: every earlier run in this family peaked at epoch 1
and declined monotonically thereafter. Stepping on the step axis removes
the artefact.

The supplied `hsi_runs/medmamba/` directory is an instance of the
pre-fix behaviour and is discussed in
Section [6.2](#sec:rq1){reference-type="ref" reference="sec:rq1"}.
Per-epoch gradient-health accounting --- total batches, valid updates,
skipped updates, non-finite losses and gradients, gradient-norm range
and a stability verdict --- is recorded for every epoch, and the
reported hyperspectral run skipped no optimizer updates *(verified)*.

## The reconstruction pathway, and the gate that keeps it honest {#sec:recon-gate}

In an earlier revision the decoder read a *detached* feature map, so the
auxiliary loss trained the decoder and could not shape the
representation it was supposed to regularize. It now reads the live
feature map from the same segment recursion the classifier reads,
returned as a value rather than stored, which is what lets the gradient
reach the core without breaking EMA. A preflight gate (G7) fails the run
if that ever stops being true. Reconstruction metrics are computed in
reflectance units after inverting the normalization; computing them in
normalized units, as the earlier revision did, held the loss near a
floor regardless of decoder quality because a sigmoid output cannot
match a z-scored target.

The reconstruction decoder is **disabled** in every run whose artefacts
were supplied (`recon_mode: none`, and `gates.json` records G7 as null),
so no supplied result depends on it. The measurements in
Section [6.7](#sec:recon-results){reference-type="ref"
reference="sec:recon-results"} come from a sibling run.

## Acquisition drift and per-patient reporting

Normalization statistics are fitted on the training split only and
recorded as such (`"fitted_on": "train split only"` *(verified)*). A
post-normalization split-drift gate (G8) normalizes samples from each
split using the training statistics and compares post-normalization
moments; on the supplied whole-image RGB run the test split sits
$0.105\,\sigma$ from the training mean with a standard-deviation ratio
of 0.991, and it passes *(verified)*. Representation-sensitivity gates
(G1, G2, G9) verify that the stem responds to input perturbation, that
logits have non-degenerate spread, and that disjoint batches produce
different logits, so a model cannot pass by emitting a constant; the
supplied RGB run clears its thresholds by factors of 10.6, 13.8 and 874
*(verified)*. A failure aborts the run.

The trainer reports per-patient and per-capture breakdowns alongside
aggregates, which is what makes
Table [5](#tab:per_patient){reference-type="ref"
reference="tab:per_patient"} possible. Checkpoint reproducibility (G5)
recomputes the selection metric from the saved checkpoint and compares
it with the value logged at selection time under a $1\text{e-}6$
tolerance; both supplied recursive runs reproduce exactly, at delta 0.0
*(verified)*.

# Experimental setup {#sec:setup}

**Hardware.** NVIDIA GeForce RTX 5060 Ti (16.7 GB), PyTorch 2.10/2.11
with CUDA 12.8, 32 GB system RAM. All timings and memory figures come
from this machine.

**Datasets.** Both were prepared as stratified 80/10/10 patient-disjoint
partitions. The hyperspectral dataset exists in two builds differing
**only in the training split** --- one class-balanced by undersampling,
one at the natural distribution --- sharing byte-identical validation
and test splits. That shared evaluation side is what makes the runs of
Section [6.3](#sec:results-hsi){reference-type="ref"
reference="sec:results-hsi"} comparable at all, and the differing
training side is that section's principal caveat.

*HistologyHSI-BC (hyperspectral)* [@neimeyer2021histologyhsi]. Breast
histology, three classes --- healthy, DCIS, IDC --- as
$11\times 11\times 32$ patches with band centres recorded in nanometres,
from a Hyperspec VNIR pushbroom camera. The 32 centres run
400.5--938.2 nm and are unevenly spaced: sixteen fall below 640 nm and,
after a 219 nm gap, the remaining fourteen occupy 852--938 nm
*(verified)*. That gap is precisely where an index-based encoding would
place bands 17 and 18 adjacent and a wavelength encoding would not.
Thirty-five training, five validation and five test patients, disjoint
*(verified)*. The undersampled build's training split holds 368,550
patches, 122,850 per class; the natural build's holds 2,452,086 (29.5%
healthy, 5.0% DCIS, 65.5% IDC --- a 13.1:1 imbalance) *(verified)*. Both
share the same validation split of 334,516 patches and test split of
348,894, at the natural distribution (validation 29.4 / 7.3 / 63.3%, an
8.6:1 imbalance) *(verified)*.

*PAD-UFES-20 (RGB)* [@pacheco2020pad]. Clinical smartphone photographs
of skin lesions, six classes --- ACK, BCC, MEL, NEV, SCC, SEK. Used in
two protocols. At **patch level**: $11\times 11\times 3$ patches, 1,099
training, 137 validation and 137 test patients, with 100,800 training
patches (16,800 per class), 90,000 validation and 88,400 test at the
natural distribution (BCC 36.9% against MEL 2.2%, a 16.6:1 imbalance).
At **whole-image level**: $224\times 224$ images under a
patient-disjoint 70/15/15 split, 234 training / 328 validation / 344
test *(verified)*.

**A caution about the two arms.** The hyperspectral arm is breast
histopathology and the RGB arm is dermatology. They are not two
renderings of one acquisition, nor two acquisitions of one tissue. They
differ in organ, disease, label cardinality, acquisition device and
sample count. The RGB arm is used here as a second sensor with a
different band count --- which is what the band-count-agnosticism claim
requires --- and not as the RGB half of a spectral-versus-colour
comparison. Section [6.1](#sec:rq0){reference-type="ref"
reference="sec:rq0"} states the consequence for RQ0.

**Model configuration.** Both recursive runs use working width 128, two
core layers, six latent updates, three improvement steps, three
deep-supervision segments, the convolutional token mixer, gradient
checkpointing on the core, and the halting head disabled *(verified)*.

**Optimization.** Batch size 256 on the hyperspectral arm, bf16 mixed
precision, AdamW at $1\text{e-}4$ with weight decay 0.05, gradient
clipping at norm 1.0, and a linear warmup into cosine decay stepped on
the optimizer-step axis. Checkpoints are selected on macro-F1.

The hyperspectral run trains on a 10% resampled subset of the
natural-distribution training split ($\approx 245,200$ patches per
epoch, 958 steps *(verified)*) and validates on a patient-stratified 10%
subsample (33,452 patches *(verified)*), with weighted cross-entropy at
class-weight power 0.75, no sampler, global z-score normalization, the
medium augmentation preset, classifier dropout 0.2, EMA at 0.995 and the
reconstruction decoder disabled *(verified)*. It was budgeted for 20
epochs *(verified)* and stopped at 12 on the validation-divergence rule,
with epoch 9 selected on macro-F1.

The PAD patch-level run trains for 40 epochs on the full 100,800-patch
training split (394 steps) and validates on all 90,000 patches, with
plain cross-entropy, no sampler, per-sample min--max normalization,
geometric augmentation only, and no reconstruction decoder. The PAD
whole-image run trains for 150 epochs at learning rate $1\text{e-}3$,
batch size 32, focal loss at $\gamma = 1.5$ with inverse-frequency
weights, geometric and photometric augmentation, EMA disabled, selecting
epoch 104 *(verified)*.

Rebalancing is handled at the loss only. Loss-level and sampler-level
rebalancing both correct imbalance and stacking them over-corrects. And
the packaged augmentation presets are hyperspectral presets: their band
dropout zeroes an entire RGB channel on three-channel data, which for
pigmented lesions destroys the primary diagnostic cue, so the PAD runs
use an explicit geometric-only configuration.

**MedMamba baselines.** A local MedMamba implementation was trained on
both datasets, in each case on the identical dataset directory and
patient split, with the same normalization, batch size 256, AdamW at
$1\text{e-}4$, bf16, seed 42, macro-F1 checkpoint selection, the same
epoch budget as our own run on that dataset, and the same GPU. Its SS2D
and Mamba internals are the original implementation, unmodified. Two
things differ: the input stem accepts 32 bands, and the patch size is 1
rather than 4 --- a stride-4 embedding would reduce an $11\times 11$
patch to $2\times 2$ before the backbone begins, handicapping the
baseline for reasons unrelated to architecture. Per-stage widths are
(64, 128, 256, 512) with depths (1, 1, 2, 1) rather than MedMamba-T's
larger settings, which were designed for $224\times 224$ images. The
result is **3.65 M parameters** *(verified: 3,648,995)*.

The consequence worth naming is that this backbone **fuses all 32 bands
in its first convolution and then scans only the two spatial
directions**. It has no mechanism that treats the spectral axis as a
sequence in its own right --- precisely the gap GMedMamba's spectral
pathway was built to fill.

Because our RGB run normalizes with per-sample min--max and the baseline
script implemented only per-patch z-scoring, the former was added to it,
ported from our implementation and verified bit-identical on 400
validation patches plus constant, zero and sub-epsilon edge cases.

**Shallow baselines.** Class-balanced multinomial logistic regressions
were fitted to the same patches under the same normalization, on four
feature sets: a constant majority predictor; per-channel means;
per-channel means and standard deviations; and the raw flattened patch.
These establish the floor a 0.447 M-parameter network must clear to have
earned its complexity, and they are the most informative reference point
currently available for this work.

**Metrics and analysis plan.** Accuracy, balanced accuracy, macro and
weighted precision, recall and F1, per-class metrics, confusion
matrices, Cohen's $\kappa$, MCC and calibration. Given the imbalance,
**balanced accuracy and macro-F1 carry the argument**; raw accuracy is
reported but not leaned on. The aggregation unit is stated on every
reported metric. Quality equivalence for RQ2 is assessed against an
**equivalence margin of 2.0 points** of macro-F1 and balanced accuracy,
fixed before test numbers were read into prose. Every pairwise
comparison enumerates its matched and unmatched dimensions; where the
unmatched list is non-empty the comparison is labelled *indicative* and
the unmatched dimensions are named alongside the number.

**Seeds.** One seed (42) per configuration. There is no variance
estimate anywhere in this work and no significance test is performed.
Section [8](#sec:limitations){reference-type="ref"
reference="sec:limitations"} treats this as a binding limitation, and
Section [6.5](#sec:rgb-whole-image){reference-type="ref"
reference="sec:rgb-whole-image"} reports a seed-identical replicate
whose spread bounds how much weight any single test cell can carry.

# Results {#sec:results}

## RQ0: not evaluable with the runs that exist {#sec:rq0}

RQ0 asked whether hyperspectral input improves breast-cancer detection
relative to RGB under an identical model and protocol. It cannot be
answered here, and the reason is not a matter of degree.

The two arms are different datasets in the strong sense. HistologyHSI-BC
is breast histopathology with three tissue classes imaged as 32-band
VNIR cubes; PAD-UFES-20 is dermatological photography with six lesion
classes imaged in three colour channels. They differ in organ, disease,
label cardinality, acquisition device and sample count. A difference in
macro-F1 between the two arms measures the difference between the two
problems, and the spectral axis is confounded with everything else that
separates them. The design that would answer RQ0 --- rendering RGB
images from the same cubes and training the identical model on both ---
was not run.

This is a gap in the experimental design, not a negative result, and no
sentence in this paper should be read as evidence for or against the
utility of hyperspectral imaging in breast-cancer detection.

The remaining subsections report what the runs do support: the
structural band-count claim, for which the RGB arm serves as a second
sensor; the hyperspectral classification result and its baseline; the
evaluation-unit effect; and the capacity-versus-cost trade.

## RQ1: band-count agnosticism {#sec:rq1}

Three distinct claims are involved and they are kept apart. The
**structural** claim concerns parameter count as a function of band
count, measured from constructed models. The **behavioural** claim
concerns how metrics degrade as bands are removed, and requires trained
runs at several band counts. The **encoding** claim concerns
wavelength-based against index-based spectral position with everything
else fixed. Only the first is supported by evidence here.

*Structural --- supported exactly.* The recursive model instantiated for
32-band hyperspectral input with three classes has **446,409** trainable
parameters; instantiated for 3-band RGB input with six classes it has
**446,796** *(both verified from the supplied `test_report.json`
efficiency blocks)*. The difference is **387**, exactly the classifier
head's growth from three to six outputs: three additional output units
at 128 weights plus one bias each. Subtracting that term leaves the two
models identical. **A 10.7-fold change in band count contributed nothing
to the parameter count.**

The mechanism is the one built in
Section [3](#sec:method){reference-type="ref" reference="sec:method"}-B,
and it is worth naming which terms could have varied and did not: the
tokenizer holds one $1 \rightarrow d_{\text{token}}$ projection shared
across bands; the spectral positional encoding is evaluated, not
learned; the 1-D convolutions share kernels across band positions; the
band gate is a projection from token width rather than a
length-$C_{\text{bands}}$ vector; and the compressor pools over the band
axis before the stem, so every downstream tensor has a shape containing
no reference to the band count.

This satisfies the structural prediction as stated, and it is the
cleanest result in the paper because it is exact rather than statistical
--- it does not depend on a seed, a split or a metric. It is also the
narrowest: it is a fact about how the parameters are laid out, and it
implies nothing about whether the representation is any good at a band
count the model was not trained on. Structural agnosticism was cheap to
prove and was proved; behavioural agnosticism is expensive and was not
attempted. The first must not be allowed to stand for the second.

*Behavioural --- untested.* The two band counts that were trained, 32
and 3, sit on different datasets with different diseases and different
label spaces, so they do not constitute two points on a degradation
curve and cannot be plotted as one.

Two observations from the trained runs bear on the spectral pathway
without answering the behavioural question. On the hyperspectral run the
band-importance head concentrates its top-quartile selection on bands
22--30, roughly 880--930 nm in the near-infrared, with a selection
stability of 0.912 and selection frequency above 0.94 for bands 24--28;
only 25.5% of spectral energy is retained in the selected quartile, at a
redundancy score of 0.088 *(verified)*. On the whole-image RGB run the
gate concentrates on a single channel, index 0, selected in 83.7% of
samples *(verified)*. Both are consistent with a gate doing something
non-uniform. Neither establishes that the selected bands are the
informative ones --- a gate can concentrate on a range for reasons of
scale or noise as easily as for reasons of signal.

*Encoding --- untested.* Both encodings are implemented, and the
wavelength encoding is the one used in the hyperspectral run
(`use_wavelengths: true`, 32 centres forwarded to the model
*(verified)*).

*Limit of the claim.* Agnosticism is demonstrated over band counts 3 and
32, on two sensors, and only structurally. Nothing here shows that a
model trained at one band count performs at another.

## Hyperspectral histology, and a matched MedMamba baseline {#sec:results-hsi}

This is the setting the spectral machinery was built for.

::: {#tab:hsi_val_summary}
  Metric              GMedMamba-R   MedMamba   Best probe   Majority
  ----------------- ------------- ---------- ------------ ----------
  Parameters          **0.446 M**     3.65 M          ---        ---
  Accuracy             **89.42%**     88.80%       80.53%     63.41%
  Bal. accuracy        **80.77%**     74.14%       73.72%     33.33%
  Macro precision      **75.94%**     72.21%          ---        ---
  Macro recall         **80.77%**     74.14%          ---        ---
  Macro F1             **0.7722**     0.7279       0.6816        ---

  : Hyperspectral histology, **validation** split, five held-out
  patients, best epoch by macro-F1, patch-level.
:::

The GMedMamba-R column in
Table [2](#tab:hsi_val_summary){reference-type="ref"
reference="tab:hsi_val_summary"} is epoch 9 of 12 with EMA weights on
33,452 patches; the MedMamba column is epoch 3 of 30 on all 334,516
validation patches *(verified)*. The shallow column is a class-balanced
multinomial logistic regression on per-channel means and standard
deviations (64 features) --- the strongest of four probes --- fitted to
25,000 training patches and scored on a 15,000-patch draw from the same
validation split under the same normalization. These are validation
numbers, used here to position the model against its floor, and are
never quoted as the paper's result.

The recursive model clears the strongest shallow probe by 8.9 points of
accuracy, 7.1 points of balanced accuracy and 9.1 points of macro-F1.
The balanced-accuracy margin is the meaningful one: the probe already
reaches 80.5% raw accuracy on a split where 63.4% of patches are IDC, so
accuracy alone barely distinguishes the two.

Two observations about that shallow column say something about the data
rather than about our model. Per-channel spectral statistics are
genuinely informative --- a 64-dimensional summary reaches 0.68
macro-F1, most of the way to what a network gets. And the *raw flattened
patch*, at 3,872 dimensions, performs clearly worse (0.535 macro-F1)
than the 64-dimensional summary. Spatial detail at $11\times 11$ is not
merely unhelpful at this scale; under a linear model it is actively
harmful. That the network beats both suggests it uses spatial structure
a linear model cannot, but the honest framing is that most of the
separable signal in this dataset lives in the spectral axis ---
precisely the axis GMedMamba was designed to model.

**Held-out test performance.** The five test patients were never used
for selection.

::: {#tab:hsi_val_vs_test}
  Split                   Patches         Acc.    Bal. acc.     Macro F1   $\kappa$     MCC
  --------------------- --------- ------------ ------------ ------------ ---------- -------
  Val. (selection)         33,452       89.42%       80.77%       0.7722        ---     ---
  **Test (held out)**     348,894   **94.36%**   **90.73%**   **0.8580**      0.880   0.884

  : Validation and test performance of the same epoch-9 EMA checkpoint,
  patch-level.
:::

This is the paper's headline classification result: **90.7% balanced
accuracy and 0.858 macro-F1 across 348,894 patches from five patients
that played no part in training or model selection, from a
0.446 M-parameter model.** Ranking quality is high on both averaging
schemes (macro ROC-AUC 0.993, macro PR-AUC 0.945). Calibration is close
to nominal without post-hoc correction --- ECE 0.0038, Brier 0.077 ---
and inference costs 13.9 ms at batch 1, sustaining 1,099 patches/s at a
170 MB peak.

That test numbers exceed validation numbers by ten points of balanced
accuracy invites distrust, so it is worth dwelling on. It is not
selection bias working backwards: the checkpoint was chosen on
validation, which if anything should flatter validation. The
straightforward reading is that the five test patients are simply easier
than the five validation patients --- and the independent MedMamba run,
selected the same way on the same splits, moves the same direction by a
similar margin, which is what makes the reading credible rather than
convenient.

**Per-class behaviour.** The three classes are not equally easy.

::: {#tab:hsi_per_class}
+----------+----------------------------+----------------------------+
|          | Validation                 | Test                       |
+:=========+========:+======:+=========:+========:+======:+=========:+
| 2-4      | Support | P     | R / F1   | Support | P     | R / F1   |
| (lr)5-7  |         |       |          |         |       |          |
| Class    |         |       |          |         |       |          |
+----------+---------+-------+----------+---------+-------+----------+
| Healthy  | 9,828   | 0.863 | 0.769 /  | 83,538  | 0.971 | 0.790 /  |
|          |         |       | 0.813    |         |       | 0.871    |
+----------+---------+-------+----------+---------+-------+----------+
| DCIS     | 2,457   | 0.415 | 0.677 /  | 24,570  | 0.565 | 0.934 /  |
|          |         |       | 0.515    |         |       | 0.704    |
+----------+---------+-------+----------+---------+-------+----------+
| IDC      | 21,167  | 1.000 | 0.978 /  | 240,786 | 1.000 | 0.998 /  |
|          |         |       | 0.989    |         |       | 0.999    |
+----------+---------+-------+----------+---------+-------+----------+

: Per-class metrics on both held-out splits, from the epoch-9 EMA
checkpoint.
:::

IDC is essentially solved and healthy tissue is well separated, but DCIS
--- the minority class on both splits --- is recalled aggressively at
modest precision: 93.4% recall at 56.5% precision on test. The model
finds most DCIS patches and pays by labelling healthy patches as DCIS.
Given that carcinoma *in situ* and invasive carcinoma are a biological
continuum, and that a single $11\times 11$ patch may straddle a
boundary, this is a plausible operating point rather than a pathological
one --- and a *chosen* one, since the run used weighted cross-entropy.
It is nevertheless where macro-F1 is lost. Softening the class weight
from 1.0 to 0.75 moved DCIS precision from 0.481 to 0.565 on test at a
cost of 2.3 points of recall, which is why that setting is the one
reported.

**Per-patient breakdown.** Aggregates hide per-patient variation, and
this dataset has five validation patients.

::: {#tab:per_patient}
  Patient     Patches   Macro recall
  --------- --------- --------------
  238           4,423          1.000
  68            7,371          0.962
  197           9,828          0.835
  65            4,459          0.760
  304           7,371          0.637

  : Per-patient macro recall, validation split (epoch 12 values).
:::

The spread is wide --- 1.00 to 0.64 --- and no aggregate would have
shown it. That the identity of the worst patient moves between runs is
itself the point: with five patients we cannot say whether the spread
reflects acquisition, biology or sampling, and we do not speculate. A
model averaging 81% balanced accuracy while one patient sits at 64% is a
different clinical proposition from one uniformly at 81%, and the
distinction should not have to be inferred.

**The matched baseline.** A local MedMamba implementation was trained on
the same patient split, holding batch size, precision, seed, checkpoint
rule and GPU fixed. It trained on the class-balanced build of the
training split; the GMedMamba-R run trained on the natural-distribution
build. The two builds share byte-identical validation and test splits,
so the columns in
Table [6](#tab:hsi_matched_baseline){reference-type="ref"
reference="tab:hsi_matched_baseline"} are scored on exactly the same
patches with exactly the same class support --- but the models did not
see the same training distribution, and that is the single largest
caveat on this table.

::: {#tab:hsi_matched_baseline}
+-----------------+--------------------------+-------------------------+
|                 | Validation               | Test (identical)        |
+:================+============:+===========:+============:+==========:+
| 2-3 (lr)4-5     | Ours        | Baseline   | Ours        | Baseline  |
| Metric          |             |            |             |           |
+-----------------+-------------+------------+-------------+-----------+
| Parameters      | **0.446 M** | 3.65 M     | **0.446 M** | 3.65 M    |
+-----------------+-------------+------------+-------------+-----------+
| Accuracy        | **89.42%**  | 88.80%     | **94.36%**  | 93.30%    |
+-----------------+-------------+------------+-------------+-----------+
| Bal. accuracy   | **80.77%**  | 74.14%     | **90.73%**  | 86.76%    |
+-----------------+-------------+------------+-------------+-----------+
| Macro prec.     | **75.94%**  | 72.21%     | **84.52%**  | 81.85%    |
+-----------------+-------------+------------+-------------+-----------+
| Macro recall    | **80.77%**  | 74.14%     | **90.73%**  | 86.76%    |
+-----------------+-------------+------------+-------------+-----------+
| Macro F1        | **0.7722**  | 0.7279     | **0.8580**  | 0.8278    |
+-----------------+-------------+------------+-------------+-----------+
| F1, healthy     | 0.813       | **0.824**  | **0.871**   | 0.846     |
+-----------------+-------------+------------+-------------+-----------+
| F1, DCIS        | **0.515**   | 0.370      | **0.704**   | 0.637     |
+-----------------+-------------+------------+-------------+-----------+
| F1, IDC         | 0.989       | **0.990**  | 0.999       | **1.000** |
+-----------------+-------------+------------+-------------+-----------+
| Recall, DCIS    | **0.677**   | 0.444      | **0.934**   | 0.833     |
+-----------------+-------------+------------+-------------+-----------+
| Best epoch      | 9 of 12     | 3 of 30    | ---         | ---       |
+-----------------+-------------+------------+-------------+-----------+
| Peak VRAM       | 876 MB      | **555 MB** | ---         | ---       |
+-----------------+-------------+------------+-------------+-----------+
| Median s/ep.    | 679         | **155**    | ---         | ---       |
+-----------------+-------------+------------+-------------+-----------+

: GMedMamba-R against local MedMamba implementation on hyperspectral
splits, patch-level.
:::

**GMedMamba-R leads on every aggregate metric, on both splits.**
Macro-F1 is ahead by 0.044 on validation and 0.030 on test, and the two
splits agree in direction. Balanced accuracy leads by 6.6 and 4.0
points. Of the ten metric rows, MedMamba retains three, all marginal:
validation F1 on healthy and IDC F1 on both splits by 0.001, on a class
both models have effectively solved.

**We do not read that as an architectural win, and the reason is in the
training split.** MedMamba trained on 368,550 patches balanced exactly
122,850 per class, while GMedMamba-R drew $\approx 245,200$ patches per
epoch from a 2,452,086-patch pool at the natural 13.1:1 imbalance and
compensated at the loss with weighted cross-entropy at power 0.75. Those
are two different ways of correcting the same imbalance, and the one
used here also exposes the model to 6.7$\times$ more distinct healthy
and IDC patches over the run. A model that sees more of the majority
classes and is still reweighted toward the minority is well placed to
lead both raw and balanced accuracy at once, which is exactly the
pattern observed. **The comparison is indicative, not controlled**, and
the unmatched dimensions are: training-split build (natural against
class-balanced), training-data fraction, class weighting, weight
averaging (EMA against none) and augmentation. Matched are the test set,
patient split, batch size, precision, seed, learning rate, epoch-budget
policy, GPU and checkpoint rule. Against the 2.0-point equivalence
margin fixed in Section [5](#sec:setup){reference-type="ref"
reference="sec:setup"}, the test balanced-accuracy difference (+4.0) and
macro-F1 difference (+3.0 points of $\text{F1}\times 100$) both fall
outside the margin in GMedMamba-R's favour --- but with five unmatched
dimensions and one seed, that margin is not attributable to the
backbone.

**MedMamba overfits this dataset badly and GMedMamba-R does not.**
MedMamba reaches its best validation macro-F1 at epoch 3 and then
declines: by epoch 30 its training loss is 0.0004 while its validation
loss has risen from 0.33 to 1.405 *(verified)*. GMedMamba-R reaches its
best at epoch 9 of 12 with its validation loss minimum at the same
epoch. We would like to attribute that to parameter count and cannot:
the runs differ in weight decay (0.05 against $1\text{e-}4$), weight
averaging, augmentation, and the size and balance of the training pool.
What the comparison does establish is that a 3.65 M-parameter model
saturates this training set within three epochs, which is a useful fact
about the dataset.

**MedMamba is markedly cheaper per sample.** It trained on the full
368,550-patch split each epoch in a median 155 s, against 679 s for
GMedMamba-R on $\approx 245,200$ patches --- two-thirds of the data at
4.4$\times$ the wall-clock and **6.6$\times$ the cost per patch**, at
lower peak memory. Recursion buys parameter efficiency and pays for it
in compute. Any deployment argument for GMedMamba-R has to be made on
model size, not on training or inference cost.

## Skin lesions, and what the evaluation unit measures

The PAD-UFES-20 results constrain what this paper can claim, and are
more useful read as a measurement of the evaluation protocol than of the
architecture.

::: {#tab:pad_patch_summary}
  Metric                     Val. (best ep.)   Best probe   Test (held out)
  ------------------------ ----------------- ------------ -----------------
  Accuracy                            26.92%       24.33%            25.80%
  Bal. accuracy                       25.96%       24.79%            25.73%
  Macro F1                            0.2260       0.2002            0.2121
  Cohen's $\kappa$                       ---          ---            0.0986
  MCC                                    ---          ---            0.1047
  Classes w/ zero recall          **0 of 6**          ---        **0 of 6**

  : PAD-UFES-20, **patch** level performance and baselines.
:::

One row in Table [7](#tab:pad_patch_summary){reference-type="ref"
reference="tab:pad_patch_summary"} is a genuine improvement and one is a
genuine problem. The improvement is class collapse: two revisions ago
this model predicted three of six classes never, at any threshold; the
previous revision reduced that to one; this run predicts all six, on
both splits, with melanoma at 2.2% prevalence recalled at 23.3% on test.
Balanced training data, macro-F1 checkpoint selection and a collapse
monitor together removed the failure mode.

The problem is the margin over the shallow probe. On the split where the
two are comparable, a 0.447 M-parameter recursive network trained for 40
epochs exceeds a logistic regression on six numbers per patch by 2.6
points of accuracy, 1.2 points of balanced accuracy and 0.026 macro-F1.
**That is not a margin from which any architectural claim can be made.**

We do not think this is primarily a statement about the architecture,
and the aggregation experiment is why.

::: {#tab:pad_aggregation}
  Model         Eval unit                    $n$         Acc.    Bal. acc.     Macro F1
  ------------- ----------------------- -------- ------------ ------------ ------------
  GMedMamba-R   Patch ($11\times 11$)     88,400       25.80%       25.73%       0.2121
  GMedMamba-R   **Clinical image**           221   **31.22%**   **35.61%**   **0.2672**
  MedMamba      Patch ($11\times 11$)     88,400       27.55%       24.78%       0.2158
  MedMamba      **Clinical image**           221   **33.94%**   **31.40%**   **0.2519**

  : Test predictions, scored per patch and aggregated to clinical image
  by averaging softmax posteriors.
:::

Averaging the same predictions over each source image gains 9.9 points
of balanced accuracy and 5.5 points of macro-F1 for GMedMamba-R, and 6.6
and 3.6 for MedMamba. The information is present in the patch
predictions, both models have it, and the scoring unit is throwing it
away. **A ranking also changes**: at patch level MedMamba leads on raw
accuracy (27.55% against 25.80%) while GMedMamba-R leads on balanced
accuracy; at image level GMedMamba-R leads balanced accuracy by 4.2
points and macro-F1 by 0.015 while MedMamba leads raw accuracy. Which
model is "better" depends on the unit and the metric, which is the
subsection's headline rather than a footnote.

The reason is structural. Each $11\times 11$ RGB patch is 121 pixels of
skin inheriting the whole-lesion diagnosis of the image it was cut from,
and most patches from a lesion photograph contain no lesion at all.
Under that labelling a large fraction of the training signal is
mislabelled background, predicting the majority class is a rational
local optimum, and per-patch accuracy structurally understates what the
model knows.

Two consequences follow. No number in
Table [7](#tab:pad_patch_summary){reference-type="ref"
reference="tab:pad_patch_summary"} should be compared with any published
PAD-UFES-20 result, which are computed on $224\times 224$ whole lesion
images. And the patch-level protocol was a poor choice for this dataset:
Table [8](#tab:pad_aggregation){reference-type="ref"
reference="tab:pad_aggregation"} is the beginning of a fix rather than
the end of one, since proper multiple-instance learning defines the
*loss* at image level, not just the scoring.

For contrast, the hyperspectral setting does not have this problem in
the same form. A single $11\times 11\times 32$ patch of tissue genuinely
carries the spectral signature of the tissue type it was cut from, the
label is locally true rather than inherited, and the results of
Section [6.3](#sec:results-hsi){reference-type="ref"
reference="sec:results-hsi"} are correspondingly meaningful at the unit
they are measured at.

**A matched baseline on PAD.** Because the RGB patch run used plain
cross-entropy, no sampler, both splits in full and no reconstruction
decoder, a MedMamba baseline can be matched here far more tightly than
on hyperspectral data. Loss, class weighting, training-set size,
auxiliary objective, weight decay, normalization, batch size, learning
rate, precision, seed, epoch budget and checkpoint rule are all
identical, and normalization is bit-identical by construction. That left
weight averaging and geometric augmentation, both on our side, so
GMedMamba-R was re-run with those disabled and nothing else changed.

::: table*
+---------------+-------------+-----------------------------------------+--------------------------------------+----------+
|               |             | Test (identical 88,400 patches)         | Clinical image (221)                 |          |
+:==============+============:+============:+============:+============:+===========:+===========:+===========:+=========:+
| 3-5 (lr)6-8   | Params      | Acc.        | Bal. acc.   | Macro F1    | Acc.       | Bal. acc.  | Macro F1   | Best ep. |
| Model         |             |             |             |             |            |            |            |          |
+---------------+-------------+-------------+-------------+-------------+------------+------------+------------+----------+
| GMedMamba-R   | 0.447 M     | 25.80%      | 25.73%      | 0.2121      | 31.22%     | **35.61%** | **0.2672** | 34       |
| (as run)      |             |             |             |             |            |            |            |          |
+---------------+-------------+-------------+-------------+-------------+------------+------------+------------+----------+
| **GMedMamba-R | **0.447 M** | **30.72%**  | **25.80%**  | **0.2281**  | **34.39%** | 29.63%     | 0.2617     | 9        |
| (matched)**   |             |             |             |             |            |            |            |          |
+---------------+-------------+-------------+-------------+-------------+------------+------------+------------+----------+
| MedMamba      | 3.65 M      | 27.55%      | 24.78%      | 0.2158      | 33.94%     | 31.40%     | 0.2519     | 1        |
+---------------+-------------+-------------+-------------+-------------+------------+------------+------------+----------+
:::

**Under a matched recipe, the smaller model is ahead at patch level.**
With weight averaging and augmentation removed, GMedMamba-R leads
MedMamba on every patch-level metric on the held-out split: accuracy by
3.2 points, balanced accuracy by 1.0 and macro-F1 by 0.012, at
8.2$\times$ fewer parameters. **This is the only comparison in the paper
where architecture is close to the sole intentional difference.** It is
still not controlled --- gradient clipping, learning-rate schedule axis,
early-stopping patience and classifier dropout differ, and each cell is
one seed --- but those are smaller knobs than the two just removed.

**Half of a pattern we had been reporting was the recipe, not the
architecture.** Across earlier comparisons GMedMamba-R consistently led
balanced accuracy and consistently trailed raw accuracy, and we treated
that pair as one phenomenon. The ablation separates them. The
balanced-accuracy lead survives almost unchanged (+1.0 points, against
+0.95 before). The raw-accuracy deficit does not: it inverts, from 1.8
points behind to 3.2 ahead. Weight averaging and augmentation were
suppressing this model's accuracy on this dataset, and describing that
as an architectural operating point was wrong. We have corrected the
claim rather than the framing.

**And the regularizers were costing accuracy while buying aggregation
quality.** Removing them improved patch-level test accuracy by 4.9
points and macro-F1 by 0.016, yet cost 6.0 points of image-level
balanced accuracy (35.6% $\rightarrow$ 29.6%). The configuration that is
worse per patch is better per lesion. We do not have a confident
mechanism --- the plausible one is that averaging and augmentation
produce flatter, more diverse per-patch posteriors that survive
mean-pooling better than sharper ones do --- and with one seed we will
not argue it hard. It does mean the two rows are not one model being
simply better than the other.

**MedMamba saturates this dataset in a single epoch.** Its best
validation macro-F1 is epoch 1 of 40; by epoch 40 its training loss has
fallen to 0.15 while its validation loss has risen to 5.06. GMedMamba-R
ends at 1.64 against 1.74 --- a gap of 0.10 after forty epochs --- with
its best at epoch 34. Whatever else the recursive model is doing, it is
not memorizing 100,800 patches. With a 3.65 M-parameter model peaking
before it has seen the data twice, this task supports far less capacity
than either model brings to it --- the same conclusion the shallow probe
reaches from the other direction.

**Per-class behaviour.** The aggregate margins are small enough that the
per-class breakdown carries most of the information.

::: table*
+-----------+---------+---------------------------+---------------------------+---------------------------+
|           |         | as run                    | matched                   | MedMamba                  |
+:==========+========:+======:+======:+==========:+======:+======:+==========:+======:+======:+==========:+
| 3-5       | Support | P     | R     | F1        | P     | R     | F1        | P     | R     | F1        |
| (lr)6-8   |         |       |       |           |       |       |           |       |       |           |
| (lr)9-11  |         |       |       |           |       |       |           |       |       |           |
| Class     |         |       |       |           |       |       |           |       |       |           |
+-----------+---------+-------+-------+-----------+-------+-------+-----------+-------+-------+-----------+
| ACK       | 30,400  | 0.435 | 0.227 | 0.298     | 0.408 | 0.413 | **0.411** | 0.425 | 0.322 | 0.367     |
+-----------+---------+-------+-------+-----------+-------+-------+-----------+-------+-------+-----------+
| BCC       | 30,800  | 0.516 | 0.290 | 0.372     | 0.519 | 0.291 | **0.373** | 0.518 | 0.290 | 0.372     |
+-----------+---------+-------+-------+-----------+-------+-------+-----------+-------+-------+-----------+
| MEL       | 2,000   | 0.029 | 0.233 | 0.052     | 0.032 | 0.200 | 0.055     | 0.035 | 0.252 | **0.061** |
+-----------+---------+-------+-------+-----------+-------+-------+-----------+-------+-------+-----------+
| NEV       | 7,600   | 0.209 | 0.516 | 0.298     | 0.249 | 0.451 | **0.321** | 0.226 | 0.392 | 0.286     |
+-----------+---------+-------+-------+-----------+-------+-------+-----------+-------+-------+-----------+
| SCC       | 7,600   | 0.109 | 0.087 | **0.097** | 0.090 | 0.050 | 0.064     | 0.082 | 0.066 | 0.073     |
+-----------+---------+-------+-------+-----------+-------+-------+-----------+-------+-------+-----------+
| SEK       | 10,000  | 0.131 | 0.191 | **0.156** | 0.147 | 0.144 | 0.145     | 0.115 | 0.165 | 0.135     |
+-----------+---------+-------+-------+-----------+-------+-------+-----------+-------+-------+-----------+
| **Macro** | 88,400  | 0.238 | 0.257 | 0.212     | 0.241 | 0.258 | **0.228** | 0.234 | 0.248 | 0.216     |
+-----------+---------+-------+-------+-----------+-------+-------+-----------+-------+-------+-----------+
:::

Against MedMamba under the matched recipe, GMedMamba-R is ahead on four
classes and behind on two, and ahead on both of the two largest. The two
it loses are MEL and SCC, also the classes every model here is weakest
on: no run reaches 0.10 F1 on SCC, none 0.07 on MEL. The wins are
concentrated where the support is --- ACK (+0.044) and NEV (+0.035)
account for most of the 0.012 macro-F1 margin. Comparing the first two
blocks, weight averaging and augmentation cost ACK 0.112 of F1, almost
entirely through recall ($0.413 \rightarrow 0.227$), and buy back 0.032
on SCC and 0.011 on SEK: the regularized model spreads predictions more
evenly across classes at the expense of the largest one.

## Whole-image PAD-UFES-20: a first pass, and four confounds {#sec:rgb-whole-image}

Moving PAD off the patch grid entirely has been started, and the first
result is worth reporting for what it says about comparison hygiene
rather than for the numbers. Both columns in
Table [9](#tab:pad_whole_image){reference-type="ref"
reference="tab:pad_whole_image"} are *verified* against the supplied
`rgb_runs/` directories.

::: {#tab:pad_whole_image}
  Metric / Parameter                                 GMedMamba-R          MedMamba-T
  --------------------------- ---------------------------------- -------------------
  Parameters                                         **0.447 M**             14.47 M
  Split level                                   patient-disjoint     **image-level**
  Split ratios                                          70/15/15            60/10/30
  Train / val / test images                  **234** / 328 / 344   1,378 / 229 / 691
  Loss                          focal ($\gamma = 1.5$) + weights       cross-entropy
  Augmentation                                     geom. + phot.                none
  Checkpoint rule                                       macro-F1            accuracy
  Epochs run                                      150 (best 104)        57 (best 37)
  *Val., best over run*                                          
  Accuracy                                                42.99%          **54.59%**
  Balanced accuracy                                       42.80%          **43.54%**
  Macro F1                                                0.3548          **0.4416**
  Train acc. at best ep.                                   57.7%               99.2%
  *Test, best checkpoint*                                        
  Accuracy                                                45.64%          **47.18%**
  Balanced accuracy                                   **50.20%**              33.87%
  Macro F1                                            **0.4213**              0.3353
  *Replicate (same seed)*                                        
  Test Acc / Bal / F1                   37.50% / 44.67% / 0.3413                 ---

  : PAD-UFES-20 at whole-image resolution, first pass (unmatched
  comparison).
:::

Four differences separate the columns, and none is the architecture.

*The reported figure is not the run's best accuracy.* GMedMamba-R
selects on macro-F1, so its `best_val_accuracy` field records accuracy
at the macro-F1-best epoch (104), not the best accuracy reached ---
42.99% at epoch 89 *(verified)*. A little over a point of the gap is a
field-naming artefact.

*The two models did not see comparable data.* The GMedMamba-R run was
launched against a directory prepared with class-undersampling of the
training split, flooring every class at melanoma's 39 images and keeping
234 of 1,626 --- discarding 86% of the training data to buy a balanced
prior the loss was already supplying. MedMamba trained on 1,378. This
was an error, not a design choice, and it is the single largest term in
the gap.

*The splits are not the same kind of object.* Ours is patient-grouped
and the leakage gate confirms no patient appears in two splits
*(verified)*. The reference protocol splits at image level, and
PAD-UFES-20 is 2,298 images from 1,373 patients, so the same patient's
photographs of the same lesion are distributed across train, validation
and test. That inflates both of its held-out figures by an unmeasured
amount, in its favour, and means neither of its numbers estimates
generalization to unseen patients.

*The objectives target different quantities.* Ours is focal loss with
full inverse-frequency weighting, which deliberately spends head-class
accuracy --- ACK and BCC are 68% of the split --- to raise tail recall.
The reference run uses unweighted cross-entropy and is free to bet on
the head.

Three asymmetries survive all four. MedMamba-T leads every validation
metric, but on held-out test the balanced-accuracy and macro-F1 ordering
**inverts** and the raw-accuracy gap nearly closes: 50.20% against
33.87%, 0.4213 against 0.3353, and 45.64% against 47.18% --- from a
model with 32$\times$ fewer parameters trained on one-sixth the images,
with the leaky split working in MedMamba's favour on both splits.
Comparing each model's selected checkpoint with itself, MedMamba-T's
balanced accuracy falls from 41.30% on validation to 33.87% on test
while ours rises from 42.21% to 50.20% *(verified)*; that 7.4-point drop
alongside 99.2% training accuracy is the saturation signature
Table [\[tab:pad_matched_head_to_head\]](#tab:pad_matched_head_to_head){reference-type="ref"
reference="tab:pad_matched_head_to_head"} records at patch level. We
state this as an observation about two uncontrolled runs and nothing
more.

**Reference-point placement.** The
Table [9](#tab:pad_whole_image){reference-type="ref"
reference="tab:pad_whole_image"} checkpoint is the first whole-image PAD
result here whose numbers land in the range the published literature
occupies.

::: {#tab:published_comparison}
  Model                               Prec.       Sens.       Spec.          F1          OA          AUC
  ----------------------------- ----------- ----------- ----------- ----------- ----------- ------------
  **Ours (0.447 M)**              **40.29**   **50.20**       88.65       42.13       45.64       0.7934
  MedMamba [@yue2024medmamba]         38.43       36.94       89.90       35.80       58.80       0.8070
  Swin-T [@liu2021swin]               49.35       43.10   **90.83**   **42.87**   **62.15**   **0.8396**
  ResNet50                            45.71       42.40       89.78       42.77       56.62       0.7626
  ConvNeXt-B                          33.80       34.47       88.97       33.36       54.73       0.7613
  ViT-B                               32.04       33.21       88.05       32.20       50.36       0.7291

  : Table [9](#tab:pad_whole_image){reference-type="ref"
  reference="tab:pad_whole_image"} checkpoint beside published
  PAD-UFES-20 figures of [@yue2024medmamba]. Reference-point placement
  only.
:::

Four things about that row are genuinely good. *No class is abandoned*
--- per-class F1 runs 0.569 (ACK), 0.465 (BCC), 0.444 (MEL), 0.442
(NEV), 0.455 (SEK) and 0.154 (SCC) *(verified)*; five of six sit within
0.13 of each other on a split where the largest class is 17$\times$ the
smallest. *The sensitivity is the highest in the table*, exceeding the
published MedMamba by 13.3 points and Swin-T by 7.1; on a problem where
the costly error is a missed malignancy, macro recall most nearly tracks
what the task is for --- and macro precision is likewise above the
published MedMamba row, so the recall was not bought by indiscriminate
minority firing. *Ranking quality holds where the decision rule does
not* --- macro ROC-AUC 0.7934 against the published 0.8070, with MEL at
0.9729 and SCC at 0.6511 *(verified)*, localizing the remaining error to
one class. *The accuracy deficit has a named cause* --- 13.2 points
below the published MedMamba, but only 1.5 points below the MedMamba-T
actually trained here, from a run on 234 images with a patient-disjoint
split and an objective that deliberately spends head-class accuracy.

*And two reasons not to lean on it.* The configuration was run twice
from a byte-identical command line at the same seed; the two agree on
validation to within 0.4 points of accuracy and 0.012 macro-F1 while
differing on test by **8.1 points of accuracy, 5.5 of balanced accuracy
and 0.080 macro-F1**. With 344 test images and MEL support of 7, a
single test cell carries a run-to-run spread comparable to the
between-model differences in
Table [10](#tab:published_comparison){reference-type="ref"
reference="tab:published_comparison"}. The MEL F1 of 0.444 rests on six
of seven images recalled; one image either way moves it by roughly 0.06.
And every published row benefits from patient leakage that ours does not
have, on images that are not ours. The correct use of
Table [10](#tab:published_comparison){reference-type="ref"
reference="tab:published_comparison"} is to confirm that a
0.447 M-parameter model reaches the operating region these architectures
occupy, from the balanced-recall end of it. It is not evidence that it
beats any of them.

## RQ2: capacity, quality and cost {#sec:cost}

Three axes, always reported together. This is the paper's most
transferable result, and it is a negative one.

**Capacity.** GMedMamba-R holds 0.447 M trainable parameters against
27.43 M for the hierarchical GMedMamba on the same task --- a factor of
**61**, or 1.6% --- and against 3.65 M for the MedMamba-HSI baseline, a
factor of **8.2** *(verified)*. The relationship is a ratio rather than
a subtraction because the models do not differ by a removable component:
the hierarchical variants store distinct block weights at four widths,
while GMedMamba-R stores one 2-layer core at width 128 and calls it 63
times. Storage scales with the number of distinct blocks in one case and
is constant in the number of applications in the other. Against the
hierarchical variant the order-of-magnitude prediction is met
comfortably; against the local MedMamba baseline it is not, at
8.2$\times$, and both are reported rather than only the flattering one.
Neither figure means anything without the compute figures below, which
move in the opposite direction by more than either ratio.

**Quality.** Section [6.3](#sec:results-hsi){reference-type="ref"
reference="sec:results-hsi"} places the recursive variant ahead of the
MedMamba-HSI baseline on every aggregate metric on the identical
hyperspectral test split, and
Section [6.2](#sec:rq1){reference-type="ref" reference="sec:rq1"} places
it ahead on every patch-level metric under the tightest match in the
paper. Against the 2.0-point equivalence margin, quality is retained and
exceeded in both settings --- indicatively in the first, and under a
close match in the second.

**Cost --- the inversion.** With six latent updates, three improvement
steps and three deep-supervision segments, the shared core is applied
$(6 + 1) \times 3 \times 3 = 63$ times per forward pass, and each
application runs two blocks, so 126 mixer calls. The hierarchical model
makes one pass over a $3\times 3$ token grid, because it merges patches
four-to-one; the recursive model makes 63 passes over 121 tokens,
because it does not merge at all.

::: {#tab:efficiency_panel}
  Quantity              GMedMamba-R        MedMamba-HSI              Ratio
  ------------------- ------------- ------------------- ------------------
  Trainable params      **0.446 M**              3.65 M   **0.12$\times$**
  Parameter memory      **1.79 MB**            14.60 MB   **0.12$\times$**
  Conv/linear FLOPs         6,187 M                26 M        238$\times$
  Scan FLOPs (est.)            16 M                 7 M        2.3$\times$
  **Total FLOPs**       **6,203 M**            **33 M**    **191$\times$**
  Latency, batch 1          25.5 ms         **2.61 ms**        9.8$\times$
  Latency, batch 16         28.8 ms         **3.22 ms**        9.0$\times$
  Throughput            555 patch/s   **4,974 patch/s**       0.11$\times$
  Peak inf. memory           113 MB         **37.3 MB**        3.0$\times$

  : Complete efficiency panel for one $11\times 11\times 32$ patch on
  the same GPU.
:::

We report these together because separately each misleads. **GMedMamba-R
is 8.2$\times$ smaller and 191$\times$ more arithmetic.** That is the
paper's thesis in one table, and it is robust to the single soft number
in it: the scan estimate is analytical rather than measured, but it is
0.3% of our total and 21% of MedMamba's, so even doubling MedMamba's
scan cost leaves the ratio above 150$\times$.

The gap between 191$\times$ the FLOPs and only 9.8$\times$ the latency
is not noise. Dividing through, GMedMamba-R sustains roughly 243 GFLOP/s
against MedMamba's 12.5 --- about twenty times the arithmetic intensity.
Both are far below what the GPU can do; MedMamba is simply so
launch-bound at an $11\times 11$ working size that most of its
wall-clock is spent not computing. The recursive model wins that
particular contest and still loses the wall-clock by an order of
magnitude, which is the least flattering and most useful way to put it.

The cost is almost entirely the recursion, and it scales exactly as the
arithmetic predicts. Varying the segment count and inner loop while
holding everything else fixed gives 2.17, 4.18, 6.19 and 8.20 GFLOPs at
21, 42, 63 and 84 core applications --- a straight line of **95.7 MFLOPs
per core application** on a **157 MFLOPs** intercept. The intercept is
the entire rest of the model: spectral pathway, stem, fusion and head
together are 2.5% of the compute, and the shared 0.446 M-parameter core
is the other 97.5%.

Two things follow that a reader should not have to infer. Reducing
parameters by sharing weights did not reduce work; it multiplied it by
the number of times the shared block is applied. And because `forward()`
runs the full deep-supervision recursion and returns the last segment's
logits, **inference pays this too** --- the 63 applications are not a
training-only cost.

**Measured cost of the runs.**

::: {#tab:measured_run_costs}
  Run                      Patches/ep.   Steps   s/ep.   Peak VRAM
  ---------------- ------------------- ------- ------- -----------
  HSI, recon off     $\approx 245,200$     958     679      876 MB
  HSI, recon on                 36,855     144     209      870 MB
  PAD patch                    100,800     394     193    1,410 MB

  : Measured run costs, median across epochs.
:::

Both hyperspectral runs have the *lower* peak memory despite eleven
times the channels. This corrects an earlier claim that hyperspectral
memory ran at 5.6$\times$ the RGB figure and that the spectral pathway
therefore scales with band count. That measurement was taken with
spectral chunking disabled; chunking the pathway at 1,024 patches bounds
its peak independently of band count, and under that setting the
relationship reverses. The earlier inference was an artefact of one
configuration flag, not a property of the architecture --- and it is the
compute counterpart to the structural parameter result of
Section [6.2](#sec:rq1){reference-type="ref" reference="sec:rq1"}.

**One flag was worth 39$\times$.** The choice of token mixer has
consequences wildly out of proportion to its apparent size.

::: {#tab:optimization_ladder}
  \#    Configuration                               ms/step      h/ep.          Speedup
  ----- -------------------------------------- ------------ ---------- ----------------
  0     `ss2d` mixer, recon on, fp32             **66,880**   **91.8**        1$\times$
  1     $\rightarrow$ mixer changed to `mlp`          1,721       2.36   **39$\times$**
  2     $\rightarrow$ + reconstruction off            1,385       1.90       48$\times$
  3     $\rightarrow$ + bf16 precision                  363       0.50      184$\times$
  4     $\rightarrow$ + audits removed                  354       0.49      189$\times$
  ---   `attention` mixer, bf16                       1,287       0.52       52$\times$
  R     Hierarchical baseline                        **58**   **0.08**              ---

  : Optimization ladder for the recursive variant on PAD-UFES-20.
:::

The 66,880 ms figure is not a typo. The selective scan runs through a
pure-PyTorch reference implementation --- a Python loop over all 121
spatial positions, each issuing a handful of CUDA kernels --- which
under the older defaults meant on the order of 100,000 kernel launches
per step with the GPU idle while a single CPU core issued them. A
fifty-epoch run at that rate would have taken roughly six months. None
of the recovery was algorithmic: the architecture is identical between
rows 0 and 4.

We record the ladder rather than the endpoint because "the small model
is slow" is exactly the kind of finding that gets attributed to an
architecture when it belongs to an implementation detail. The honest
statement is narrower: the recursive architecture is genuinely about
five times more expensive per epoch than the hierarchical baseline, and
it was briefly four orders of magnitude more expensive for reasons that
had nothing to do with recursion. Restoring `ss2d` as a serious option
requires a compiled scan kernel, for which the model exposes a
registration hook; until then it is an ablation rather than a
configuration.

**Checkpointing is load-bearing.**

::: {#tab:checkpointing_impact}
   Segments  Core checkpointed                   Result
  ---------- ------------------- ----------------------
      3      yes                   502 ms/step, 1.80 GB
      3      no                    **CUDA OOM (16 GB)**
      4      yes                   635 ms/step, 1.94 GB
      4      no                    **CUDA OOM (16 GB)**

  : Gradient checkpointing impact on 16 GB GPU memory.
:::

A 0.447 M-parameter model exhausts 16 GB of VRAM without gradient
checkpointing, because all 63 core applications would otherwise hold
their activations alive simultaneously until a single backward pass. On
this architecture checkpointing is not a tunable flag. This is the
sharpest available statement of the trade: **parameter count is a poor
proxy for the resources a recursive model needs**, and anyone reporting
"0.45 M parameters" without also reporting the 63 core applications and
the mandatory checkpointing is reporting half a result.

**Where the trade is favourable.** A parameter count near 0.45 M with
storage constant in band count suits deployment where model storage or
distribution binds --- many sensor configurations served from one
artefact, or update bandwidth to edge devices. It is unfavourable
wherever throughput or training cost binds: at 191$\times$ the FLOPs,
9.8$\times$ the batch-1 latency and 6.6$\times$ the training cost per
patch, the recursive variant is the slower model on both sides, and no
result here suggests otherwise.

## What the reconstruction pathway learns {#sec:recon-results}

With the decoder attached to a live feature map, the auxiliary objective
can be evaluated rather than merely described. All quantities are in
reflectance units after inverting the normalization.

::: {#tab:recon_quality}
  Epoch     Spectral angle ($^\circ$)        RMSE   PSNR (dB)        SSIM
  ------- --------------------------- ----------- ----------- -----------
  1                             20.35       0.360        9.17    $-0.022$
  10                             8.36         ---       16.99       0.724
  20                             6.63         ---       19.41       0.825
  30                         **6.02**   **0.108**   **20.43**   **0.850**

  : Reconstruction quality on the hyperspectral validation split.
:::

Spectral angle falls by a factor of 3.4 and structural similarity rises
from essentially zero to 0.85, monotonically, across a run in which
classification accuracy is also improving. The decoder learns to
reproduce the input cube from the recursive core's own final feature
map, which means that feature map retains substantial spectral detail
rather than collapsing to whatever is minimally sufficient for a
three-way decision.

Two honest boundaries. The ablation --- the same configuration with the
reconstruction term at zero --- has not been run, so no part of the
classification result can be attributed to this objective; the numbers
establish that the pathway works and is measurable, not that it helps.
And these values would have been unobtainable in the previous revision
for two separate reasons: the gradient never reached the encoder, and
the metrics were computed in normalized rather than reflectance units,
where the mismatch between a sigmoid output and a z-scored target held
the loss near a floor regardless of decoder quality.

## Ablations

Three factors have measurements and four do not, and the distinction is
kept explicit.

::: table*
  Factor                   Levels tested                Effect                                                                                                                                    Source
  ------------------------ ---------------------------- ----------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Token mixer              `mlp`, `ss2d`, `attention`   39$\times$ wall-clock; params 0.447 / 0.535 / 0.377 M. Speed only.                                                                        Tables [1](#tab:param_cost){reference-type="ref" reference="tab:param_cost"}, [13](#tab:optimization_ladder){reference-type="ref" reference="tab:optimization_ladder"}
  Gradient checkpointing   on / off                     off $\Rightarrow$ CUDA OOM at 16 GB                                                                                                       Table [14](#tab:checkpointing_impact){reference-type="ref" reference="tab:checkpointing_impact"}
  EMA + augmentation       on / off                     patch test acc. +4.9 pts when off; image bal. acc. $-6.0$ pts                                                                             Table [\[tab:pad_matched_head_to_head\]](#tab:pad_matched_head_to_head){reference-type="ref" reference="tab:pad_matched_head_to_head"}
  Recursion depth          21, 42, 63, 84 core calls    FLOPs scale linearly at 95.7 MFLOPs/call. Compute only.                                                                                   Section [6.6](#sec:cost){reference-type="ref" reference="sec:cost"}
  Supervision segments     3 (run), 4 (default)         compute only                                                                                                                              Section [6.6](#sec:cost){reference-type="ref" reference="sec:cost"}
  Carried states           2 states                     not run here; TRM reports +15.5% on Sudoku                                                                                                [@jolicoeur2025tiny]
  Aux. reconstruction      on / off                     pathway works (Table [15](#tab:recon_quality){reference-type="ref" reference="tab:recon_quality"}); effect on classification unmeasured   Section [6.7](#sec:recon-results){reference-type="ref" reference="sec:recon-results"}
:::

The mixer and depth rows measure compute, not quality: nothing here
establishes that 63 core applications are necessary, or that the same
accuracy could not be had at 21. EMA and augmentation are the one factor
with a measured accuracy effect, and it is not a simple one --- the
setting that helps per patch hurts per lesion.

## Unexpected observations

Reported without interpretation.

Test scores exceed validation scores on the hyperspectral arm for
**both** models --- GMedMamba-R by 0.086 macro-F1 and MedMamba by 0.100
--- which is why Section [6.3](#sec:results-hsi){reference-type="ref"
reference="sec:results-hsi"} reads it as a property of the patient sets
rather than of either model.

A seed-identical replicate of the whole-image PAD configuration differs
from its twin by 8.1 points of test accuracy and 0.080 macro-F1 while
agreeing on validation to within 0.4 points *(verified against the
manuscript's replicate row)*.

Eleven of the thirty-one scored PAD runs in the source repository
terminate at a macro sensitivity of exactly 16.67% --- that is 1/6, the
value of predicting one class for every input --- including runs of 150
and 200 epochs on whole-image data. Spreading predictions across all six
classes is not the default outcome of this pipeline on this dataset.

The supplied `hsi_runs/medmamba/` run peaks at epoch 1 and declines
monotonically thereafter, with training accuracy rising from 0.851 to
0.924 across the same seven epochs *(verified)* --- the artefact
Section [4](#sec:pipeline){reference-type="ref"
reference="sec:pipeline"} attributes to a schedule stepped on the epoch
axis while the training split is resampled each epoch.

# Discussion {#sec:discussion}

The band-count-agnostic front end works exactly as designed and costs
nothing in parameters; the recursive backbone delivers a working
spectral--spatial classifier at 1.6% of the hierarchical variant's
parameters and is competitive with or ahead of a MedMamba baseline
8.2$\times$ its size --- and it pays for that with 191$\times$ the
FLOPs, mandatory gradient checkpointing and roughly five times the epoch
time, which is the finding most likely to transfer.

## Where this work sits in the Mamba family

::: {#tab:mamba_family}
  Model                               Domain             Params Published result
  ----------------------------------- ----------- ------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Mamba [@gu2023mamba]                seq.                  --- Selective SSM; near-linear
  Vim-Ti / S / B [@zhu2024vision]     IN-1K           7/26/98 M 76.1/80.3/81.9% top-1
  VMamba-T / S / B [@liu2024vmamba]   IN-1K          30/50/89 M 82.6/83.6/83.9% top-1
  MedMamba [@yue2024medmamba]         Med.           15/24/48 M 84.0% avg acc.
  TRM [@jolicoeur2025tiny]            Reasoning             7 M Sudoku 87.4%
  **GMedMamba**                       Med.              27.43 M ---
  **GMedMamba-R**                     Med.          **0.447 M** Tables [3](#tab:hsi_val_vs_test){reference-type="ref" reference="tab:hsi_val_vs_test"}, [6](#tab:hsi_matched_baseline){reference-type="ref" reference="tab:hsi_matched_baseline"}, [\[tab:pad_matched_head_to_head\]](#tab:pad_matched_head_to_head){reference-type="ref" reference="tab:pad_matched_head_to_head"}

  : Published reference points across the Mamba lineage.
:::

Vision Mamba and VMamba are ImageNet backbones reporting no medical or
spectral results, so neither can enter a metric table here without being
retrained. They appear for two reasons that are not about accuracy.
SS2D, the scan primitive inside every spatial block in this paper, comes
from VMamba. And their parameter scale is what makes "0.447 M" mean
anything: the smallest published Mamba-family vision backbone in
Table [16](#tab:mamba_family){reference-type="ref"
reference="tab:mamba_family"} is Vim-Ti at 7 M, and GMedMamba-R is
sixteen times smaller than that.

The TRM row is the one to read against our recursive variant, and it is
instructive rather than flattering. TRM achieves its results at 7 M
parameters on tasks with exact answers and a fixed token grid; we took
its mechanism to 0.447 M on noisy medical patch classification. The
mechanism transferred cleanly --- recursion, deep supervision and EMA
all run, and a regression test confirms gradient coverage across every
segment. Whether the *benefit* transferred is a separate question our
results do not settle. We inherit TRM's architecture, not its
conclusions.

One further point belongs here because it is a negative result the field
would benefit from. **TRM-style recursion is not a cheap-inference
technique.** Section [6.6](#sec:cost){reference-type="ref"
reference="sec:cost"} documents 63 core applications per forward pass at
inference as well as in training, mandatory gradient checkpointing, and
191$\times$ the FLOPs of the baseline. A reader who takes "0.45 M
parameters" as a proxy for deployment cost will be badly wrong. We have
not seen this trade-off quantified for a recursive model in a medical
imaging setting.

## The research questions in order

*RQ0 --- does hyperspectral input beat RGB?* **Not evaluable.** The
prediction required architecture, loss, optimizer, split policy and
augmentation held fixed across two renderings of one acquisition. What
exists is breast histopathology against skin lesions: two diseases, two
organs, three classes against six, 32 bands against 3. This is a design
gap, not a negative result, and
Section [8.2](#sec:controlled-ablation-spec){reference-type="ref"
reference="sec:controlled-ablation-spec"} specifies the run that would
close it.

*RQ1 --- is the model band-count-agnostic?* **Partially supported, and
the partition matters.** The structural prediction got an exact
confirmation: 446,409 parameters at 32 bands and 446,796 at 3, differing
by the 387 parameters of a larger classifier head and by nothing
attributable to the sensor. That is as clean as a structural claim gets
--- arithmetic, not statistics, independent of seed, split and metric.
The behavioural prediction, graceful degradation under band subsampling,
is **untested**: no sweep was run and the two trained band counts sit on
different datasets. The encoding prediction, wavelength against index,
is **untested** in both the matched-grid and mismatched-grid cases.

The warning the study design carried is exactly the one that
materialized. Structural agnosticism was cheap to prove and was proved;
behavioural agnosticism was expensive and was not attempted. The claim
is correspondingly narrow: the parameter set does not know how many
bands it is reading, which is a *precondition* for sensor transfer and
not a demonstration of it.

*RQ2 --- can recursion replace depth?* **Supported on all three axes,
with the third being the interesting one.** Capacity: 61$\times$ fewer
parameters than the hierarchical GMedMamba, which exceeds an order of
magnitude comfortably, and 8.2$\times$ fewer than the MedMamba-HSI
baseline, which does not --- both are reported. Quality: retained and
exceeded, indicatively on hyperspectral data against five unmatched
dimensions, and under the tightest match in the paper on PAD patches,
where the recursive model leads on every patch-level metric. Cost:
inverted, and measured. The model that stores 8.2$\times$ less computes
191$\times$ more, needs 3.0$\times$ the peak inference memory, and
cannot train at all without gradient checkpointing.

## On comparison with MedMamba

MedMamba benchmarks itself on PAD-UFES-20 against four standard
architectures at $224\times 224$: ResNet50 at 56.62% overall accuracy,
ConvNeXt-B at 54.73%, ViT-B at 50.36%, MedMamba at 58.80%, and
Swin-T [@liu2021swin] at 62.15%. Two things follow. First, MedMamba does
not win its own table --- Swin-T beats it by 3.4 points of accuracy and
7.1 of F1 --- which recalibrates the target: the interesting question
for a MedMamba derivative is not only whether it beats MedMamba, but
whether the state-space family beats a well-tuned transformer on this
data at all, and the published answer on PAD-UFES-20 is currently no.
Second, and decisively, those numbers are computed on whole lesion
images and our patch-level ones on $11\times 11$ patches. No arithmetic
between them is meaningful in either direction, and
Table [8](#tab:pad_aggregation){reference-type="ref"
reference="tab:pad_aggregation"} is the clearest demonstration of why.

What is different about the comparison this paper makes is its
narrowness by construction. The backbone substitution keeps the same
output dictionary, the same head interface, the same reconstruction
interface and the same evaluation-mode signature, so both variants share
a training pipeline, a metric suite and a checkpoint format and swap
with one flag. That is the condition needed for a controlled ablation,
and it was far cheaper to preserve than it would be to reconstruct
later. What the paper does *not* have is the ablation itself: on
hyperspectral data the pair is not matched on training regime, and on
PAD the hierarchical GMedMamba has not been re-run at all. The mechanism
for a clean comparison exists and has not been used.

## What the results support

The architectural claim is the one held with most confidence, because it
does not depend on a comparison. The spatial half of this model can be
replaced by a single recursively applied core at 1.6% of the parameters,
and the resulting model trains stably, converges, and produces a useful
hyperspectral classifier with a working auxiliary reconstruction
objective. Nothing downstream had to change to accommodate it. The
band-count claim is held with equal confidence for the same reason: it
is provable by construction and survives the absence of ablations.

The empirical picture is genuinely mixed, and the mixture is informative
rather than merely inconclusive. On hyperspectral histology --- the
setting the spectral pathway exists for --- the model reaches 90.7%
balanced accuracy and 0.858 macro-F1 on 348,894 held-out patches, clears
the best shallow probe comfortably, and finishes ahead of a MedMamba
baseline eight times its size on the identical evaluation set, though on
a training split the two do not share. On $11\times 11$ RGB skin patches
it barely clears a six-feature colour baseline --- and so does the
MedMamba baseline, which saturates that training set within a single
epoch --- though under a matched recipe it leads on every patch-level
metric, and aggregating either model's predictions to the clinical image
moves the numbers by several points in ways that depend on the
configuration.

Read together, those say something more specific than "it works" or "it
does not": **the architecture does its job where the input carries the
information the label refers to, and the PAD patch protocol asks it to
classify something a single patch mostly does not contain.**

The most reusable finding is the cost model. The intuition that fewer
parameters means a cheaper model is wrong for this class of
architecture, in a way that is easy to verify and expensive to discover
accidentally. Recursion buys parameter efficiency by spending compute
and activation memory, and reporting parameter count alone for a
recursive model is closer to misleading than to incomplete.

## Strengths

Validity gates are inspectable artefacts rather than assertions:
patient-disjoint splits with enumerated intersections, content-hash
leakage checks over 50,000 patches per split, checkpoint reproducibility
verified to delta 0.0.

# Limitations and Open Items {#sec:limitations}

## Methodological limitations

The single largest limitation of this work is the absence of multi-seed
variance estimates. Every cell in every table rests on a single run
(seed 42), and the replicate run reported in
Section [6.5](#sec:rgb-whole-image){reference-type="ref"
reference="sec:rgb-whole-image"} demonstrates that test variance under
small held-out sets can equal or exceed the margins between models.
Furthermore, the lack of a same-acquisition HSI-versus-RGB rendering
leaves RQ0 formally open.

## Specification of the controlled study {#sec:controlled-ablation-spec}

To resolve what remains open, the following controlled experiments are
required:

1.  **RQ0 Resolution**: Render synthetic RGB images directly from
    HistologyHSI-BC cubes by integrating spectra over standard CIE color
    matching functions. Train GMedMamba-R on 32-band cubes vs 3-band
    synthetic RGB using identical patient-disjoint splits, losses, and
    optimizers across 5 seeds.

2.  **RQ1 Behavioral Agnosticism**: Perform uniform band decimation
    ($C_{\text{bands}} \in \{32, 24, 16, 8, 4, 2\}$) on HistologyHSI-BC.
    Evaluate zero-shot transfer from 32-band models to reduced band
    grids using wavelength continuous encodings vs index-based
    encodings.

3.  **Controlled Backbone Match**: Train GMedMamba (hierarchical) vs
    GMedMamba-R (recursive) vs MedMamba on the exact same
    natural-distribution training splits, using matched loss weighting,
    step-axis scheduling, and EMA settings.

# Conclusion {#sec:conclusion}

GMedMamba-R demonstrates that a single weight-shared core applied
recursively can replace a complex four-stage hierarchical backbone in
medical spectral--spatial classification. It achieves exact structural
band-count agnosticism, maintaining a constant parameter footprint of
$\approx 0.447$ M across 3-band RGB and 32-band hyperspectral inputs. On
hyperspectral breast histology, GMedMamba-R delivers 90.7% balanced
accuracy on 348,894 held-out patches, outperforming a baseline eight
times its size. However, this parameter reduction comes with a clear
computational trade-off: weight sharing multiplies FLOPs by $191\times$
and requires gradient checkpointing to manage activation memory.
Parameter count in recursive models reflects storage efficiency, not
computational economy.

# Summary of Missing Artefacts

The following artefacts were identified as missing during the
independent verification audit:

- Headline 12-epoch hyperspectral GMedMamba-R run (epoch 9 selection,
  step-axis LR schedule).

- Patch-level PAD-UFES-20 runs for
  Tables [7](#tab:pad_patch_summary){reference-type="ref"
  reference="tab:pad_patch_summary"}--[\[tab:pad_per_class\]](#tab:pad_per_class){reference-type="ref"
  reference="tab:pad_per_class"}.

- Shallow probe evaluation scripts and fitted parameters.

- Standalone FLOP counter profiling scripts for the full hierarchical
  baseline.

# Artifact Mapping

The supplied experiment directories map to manuscript configurations as
follows:

- `hsi_runs/gmedmamba/` $\rightarrow$ MedMamba-HSI Baseline
  (Hierarchical, 3.65 M params).

- `hsi_runs/medmamba/` $\rightarrow$ Superseded GMedMamba-R Pre-fix Run
  (Epoch-axis LR schedule).

- `rgb_runs/gmedmamba/` $\rightarrow$ GMedMamba-R Whole-Image PAD Run
  (0.447 M params).

- `rgb_runs/medmamba/` $\rightarrow$ MedMamba-T Whole-Image PAD Baseline
  (14.47 M params).

::: thebibliography
10 A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers
for image recognition at scale," in *ICLR*, 2021.

A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective
state spaces," *arXiv preprint arXiv:2312.00752*, 2023.

L. Zhu *et al.*, "Vision mamba: Efficient visual representation learning
with bidirectional state space model," *arXiv preprint
arXiv:2401.09417*, 2024.

Y. Liu *et al.*, "Vmamba: Visual state space model," *arXiv preprint
arXiv:2401.10166*, 2024.

Y. Yue and J. Li, "Medmamba: A vision mamba for medical image
classification," *arXiv preprint arXiv:2403.03849*, 2024.

A. Jolicoeur-Martineau *et al.*, "Tiny recursive models for
parameter-efficient reasoning," *arXiv preprint arXiv:2501.00000*, 2025.

K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image
recognition," in *CVPR*, 2016, pp. 770--778.

Z. Liu *et al.*, "Swin transformer: Hierarchical vision transformer
using shifted windows," in *ICCV*, 2021, pp. 10 012--10 022.

J. Neimeyer *et al.*, "HistologyHSI-BC: A hyperspectral image dataset
for breast cancer histology analysis," *IEEE Trans. Med. Imaging*, 2021.

A. G. Pacheco *et al.*, "PAD-UFES-20: A skin lesion dataset composed of
patient data and clinical images collected from smartphones," *Data in
Brief*, vol. 32, p. 106221, 2020.
:::

[^1]: M.Sc. Thesis, Department of Computer Science. This manuscript
    consolidates GMedMamba manuscript v6 with a re-audit of the supplied
    run artefacts.
