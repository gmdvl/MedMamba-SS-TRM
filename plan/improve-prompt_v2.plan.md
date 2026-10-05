# G-MedMamba v2 Development Roadmap

This roadmap focuses **only on the model architecture**, not on the training pipeline, dataset loading, or inference code. The goal is to create a **general-purpose spectral–spatial foundation model** that accepts images with **any number of spectral bands** while remaining faithful to the core ideas of MedMamba.

---

# Overall Goals

The final model should satisfy the following design objectives:

* Accept grayscale, RGB, multispectral, and hyperspectral images through the same architecture.
* Never hardcode the number of spectral bands.
* Preserve wavelength ordering and spectral locality.
* Explicitly model both spatial and spectral dependencies.
* Be modular, configurable, and extensible.
* Remain compatible with future optimized selective-scan implementations (CUDA, Triton, etc.).

---

# Phase 1 — Redesign the Universal Input Layer (Highest Priority)

## Objective

Replace the current `LazyLinear`-based embedding with a **true channel-agnostic spectral tokenizer**.

### Problem

`LazyLinear` only infers the input dimension once. A model initialized with 31 bands cannot later process 826 bands without rebuilding the layer.

### Desired Behavior

The same trained model should process:

```text
1 band
3 bands
31 bands
128 bands
826 bands
```

without changing any parameters.

### Proposed Architecture

```text
Input Cube (B,C,H,W)

↓

Treat each wavelength as an individual spectral token

↓

Shared Spectral Encoder

↓

Spectral Token Embeddings

↓

Adaptive Aggregation

↓

Spatial Backbone
```

### Tasks

* Remove `LazyLinear`.
* Introduce wavelength-wise tokenization.
* Use shared weights across wavelengths.
* Keep the embedding independent of `C`.
* Support arbitrary channel counts during both training and inference.

### Expected Benefits

* Truly channel-agnostic.
* No architecture rebuilding.
* Better transfer across sensors.

---

# Phase 2 — Replace Global Spectral Pooling

## Objective

Preserve local spectral information instead of averaging the entire image before spectral processing.

### Current

```text
Image

↓

Global Mean

↓

Spectral Branch
```

### Problem

Spatially distinct tissues are merged into a single spectral profile.

### Proposed

```text
Image

↓

Patch-wise Spectral Tokens

↓

Spectral Mamba

↓

Global Aggregation
```

### Tasks

* Extract spectral tokens for each spatial patch.
* Apply Spectral Mamba before global pooling.
* Aggregate only after spectral modeling.

### Expected Benefits

* Better tissue discrimination.
* Preserve heterogeneous spectral signatures.
* Stronger HSI performance.

---

# Phase 3 — Hierarchical Spectral Encoder

## Objective

Replace the shallow spectral encoder with a deep residual encoder.

### Current

```text
1×1
3×1
5×1
```

### Proposed

```text
Residual Spectral Block

↓

Residual Spectral Block

↓

Spectral Mamba

↓

Residual Spectral Block
```

### Tasks

* Add residual connections.
* Add normalization.
* Add configurable depth.
* Allow different kernel sizes.

### Benefits

* Better feature extraction.
* Improved gradient flow.
* Higher representation capacity.

---

# Phase 4 — True Spectral Convolutions

## Objective

Learn local wavelength relationships directly from the image cube.

### Current

Convolutions operate on the pooled spectral profile.

### Proposed

```text
Cube

↓

1D Spectral Conv

↓

Residual

↓

Spectral Tokens
```

### Tasks

* Introduce spectral convolutions before tokenization.
* Make kernel size configurable.
* Preserve wavelength ordering.

### Benefits

* Better neighboring wavelength modeling.
* Improved spectral denoising.

---

# Phase 5 — Multi-Scale Spectral Processing

## Objective

Capture both narrow and broad spectral patterns.

### Proposed

```text
Kernel 3

Kernel 5

Kernel 9

Dilated Kernel

↓

Fusion
```

### Tasks

* Add parallel spectral branches.
* Fuse using attention or gating.
* Make branches configurable.

### Benefits

* Capture multiple spectral scales.
* Better robustness across sensors.

---

# Phase 6 — Bidirectional Spectral–Spatial Interaction

## Objective

Allow spatial features to update spectral representations.

### Current

```text
Spectral

↓

Spatial
```

### Proposed

```text
Spectral

↓

Spatial

↓

Updated Spectral Context

↓

Next Stage
```

### Tasks

After every stage:

* Pool spatial features.
* Update spectral context.
* Feed updated context into the next stage.

### Benefits

* Dynamic interaction.
* Better contextual understanding.
* Improved feature refinement.

---

# Phase 7 — Stage-Wise Spectral Refinement

## Objective

Recompute spectral representations throughout the network.

### Current

```text
Input

↓

Spectral Branch

↓

Done
```

### Proposed

```text
Stage 1

↓

Spectral Update

↓

Stage 2

↓

Spectral Update

↓

Stage 3
```

### Tasks

* Add lightweight spectral refinement blocks.
* Recompute spectral attention.
* Update FiLM parameters.

### Benefits

* Spectral information evolves with depth.
* Better semantic alignment.

---

# Phase 8 — Dynamic Band Selection v2

## Objective

Make band importance adaptive throughout the network.

### Current

Single band weighting.

### Proposed

```text
Input

↓

Band Selection

↓

Stage 1

↓

Band Selection

↓

Stage 2
```

### Tasks

* Compute band weights per stage.
* Support optional sparse selection.
* Add learnable thresholds.

### Benefits

* Context-aware band importance.
* Better computational efficiency.

---

# Phase 9 — Generalized SS2D

## Objective

Extend spatial scanning beyond the original four directions.

### Current

```text
↓

←

↑

↓
```

### Proposed

```text
↓

↑

←

→

↘

↖

↗

↙
```

### Tasks

* Add diagonal scans.
* Make scan directions configurable.
* Learn direction importance.

### Benefits

* Better texture modeling.
* Improved pathology feature extraction.

---

# Phase 10 — Multi-Scale SS2D

## Objective

Capture different spatial receptive fields.

### Proposed

```text
SS2D (3×3)

SS2D (5×5)

SS2D (7×7)

↓

Fusion
```

### Tasks

* Multiple depthwise convolutions.
* Separate scan branches.
* Fuse outputs.

### Benefits

* Better local/global balance.
* Improved feature richness.

---

# Phase 11 — Cross-Attention Fusion

## Objective

Replace simple FiLM modulation with bidirectional cross-attention.

### Current

```text
Spectral

↓

FiLM

↓

Spatial
```

### Proposed

```text
Spatial Queries

↓

Cross Attention

↓

Spectral Keys

↓

Cross Attention

↓

Spatial Output
```

### Tasks

* Implement lightweight cross-attention.
* Maintain linear complexity where possible.
* Allow configurable fusion strategies.

### Benefits

* Richer spectral–spatial interaction.
* Better information exchange.

---

# Phase 12 — Progressive Spectral Compression

## Objective

Avoid aggressive one-step dimensionality reduction.

### Proposed

```text
826

↓

512

↓

256

↓

128

↓

64
```

### Tasks

* Residual compression stages.
* Learnable compression ratios.
* Configurable bottleneck depth.

### Benefits

* Preserve information.
* Better compression quality.

---

# Phase 13 — General Backbone API

## Objective

Separate the backbone from downstream tasks.

### Proposed

```text
Backbone

↓

Classification Head

Segmentation Head

Regression Head

Detection Head

Embedding Head
```

### Tasks

* Return feature maps.
* Return pooled embeddings.
* Allow interchangeable heads.

### Benefits

* Reusable foundation model.
* Easier transfer learning.

---

# Phase 14 — Configurable Architecture

## Objective

Expose every architectural choice through configuration.

Example:

```yaml
model:
  embedding:
    type: spectral_tokens

  spectral:
    depth: 4
    multiscale: true
    dynamic_band_selection: true

  spatial:
    scan_directions: 8
    multiscale: true

  fusion:
    type: cross_attention

  normalization:
    type: rmsnorm

  residual:
    layerscale: true
```

### Benefits

* Easier experimentation.
* Cleaner ablation studies.
* Better reproducibility.

---

# Phase 15 — Future-Proof Selective Scan Backend

## Objective

Allow multiple scan implementations without changing the architecture.

### Interface

```text
SS2D

↓

Backend

├── Pure PyTorch
├── CUDA
├── Triton
├── Flash Scan (future)
```

### Tasks

* Define backend interface.
* Keep the same API.
* Auto-select the fastest available implementation.

### Benefits

* Portable development.
* Optimized deployment.
* Easy benchmarking.

---

# Suggested Development Order

| Priority | Phase                                                | Difficulty | Expected Impact |
| -------- | ---------------------------------------------------- | :--------: | :-------------: |
| ⭐⭐⭐⭐⭐    | Phase 1 – Universal channel-agnostic tokenizer       |    High    |    Very High    |
| ⭐⭐⭐⭐⭐    | Phase 2 – Patch-wise spectral processing             |   Medium   |    Very High    |
| ⭐⭐⭐⭐⭐    | Phase 3 – Hierarchical spectral encoder              |   Medium   |       High      |
| ⭐⭐⭐⭐⭐    | Phase 4 – True spectral convolutions                 |   Medium   |       High      |
| ⭐⭐⭐⭐☆    | Phase 5 – Multi-scale spectral processing            |   Medium   |       High      |
| ⭐⭐⭐⭐⭐    | Phase 6 – Bidirectional spectral–spatial interaction |    High    |    Very High    |
| ⭐⭐⭐⭐⭐    | Phase 7 – Stage-wise spectral refinement             |   Medium   |    Very High    |
| ⭐⭐⭐⭐⭐    | Phase 8 – Dynamic band selection v2                  |   Medium   |    Very High    |
| ⭐⭐⭐⭐☆    | Phase 9 – Eight-direction SS2D                       |   Medium   |       High      |
| ⭐⭐⭐⭐☆    | Phase 10 – Multi-scale SS2D                          |    High    |       High      |
| ⭐⭐⭐⭐⭐    | Phase 11 – Cross-attention fusion                    |    High    |    Very High    |
| ⭐⭐⭐⭐☆    | Phase 12 – Progressive spectral compression          |   Medium   |       High      |
| ⭐⭐⭐☆☆    | Phase 13 – General backbone API                      |     Low    |      Medium     |
| ⭐⭐⭐☆☆    | Phase 14 – Configurable architecture                 |     Low    |      Medium     |
| ⭐⭐⭐☆☆    | Phase 15 – Backend abstraction                       |   Medium   |      Medium     |

## Recommendation

One suggestion from the previous discussion deserves to be revised: I would **not** replace FiLM fusion with cross-attention outright. FiLM is lightweight, stable, and computationally efficient. A better approach is to make the fusion mechanism **pluggable** so the model can use different strategies depending on the application:

* **FiLM** as the default for efficiency and simplicity.
* **Cross-attention** as an optional, higher-capacity fusion module for research and larger models.
* **Gated additive fusion** as another intermediate option.

Similarly, rather than hardcoding eight scan directions into SS2D, make the scan topology configurable. The original four-direction scan remains a strong default, while diagonal or learned scan directions can be enabled as optional variants for experimentation.

This approach keeps G-MedMamba faithful to the MedMamba philosophy while evolving it into a flexible, modality-agnostic backbone that can scale from lightweight RGB models to large hyperspectral systems without changing its overall architecture.
