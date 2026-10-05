# G-MedMamba v3 Architecture Roadmap

## A Detailed Model Development Plan for a Universal Spectral–Spatial Foundation Model

This roadmap is **only about the model architecture**. It intentionally excludes dataset handling, preprocessing, training scripts, optimization loops, inference code, and deployment. The goal is to evolve G-MedMamba into a **sensor-agnostic backbone** that processes grayscale, RGB, multispectral, and hyperspectral images through a unified architecture.

---

# Guiding Principles

Every architectural change should satisfy the following principles:

1. **Sensor-agnostic**: The backbone must not assume a fixed number of input channels.
2. **Wavelength-aware**: Spectral bands are ordered physical measurements, not arbitrary feature channels.
3. **Local-to-global spectral modeling**: Learn local spectral patterns before forming global representations.
4. **Progressive refinement**: Spectral and spatial representations should evolve together throughout the network.
5. **Modularity**: Every major component should be replaceable or configurable for experimentation.
6. **Scalability**: The architecture should support lightweight and large variants without redesign.

---

# Phase 1 — Universal Spectral Tokenization (Highest Priority)

## Objective

Create a channel-count-independent input representation.

### Current Limitation

The model embeds the entire spectral dimension at once, so the embedding is still influenced by the input dimensionality.

### Target Architecture

```text
Input Cube (B × C × H × W)
        │
Per-Wavelength Representation
        │
Spectral Tokenizer
        │
Spectral Tokens
        │
Shared Spectral Encoder
```

### Design Requirements

* The tokenizer must operate identically regardless of whether `C = 1`, `3`, `31`, `128`, or `826`.
* Each wavelength should be treated as an individual token with shared processing weights.
* The embedding should not require reinitialization when the number of channels changes.

### Tasks

* Design a wavelength-token abstraction.
* Replace channel-dependent embedding with token generation.
* Preserve the physical order of wavelengths.
* Support variable-length token sequences.

### Deliverables

* `SpectralTokenizer` module.
* Configurable token dimension.
* Unit tests with varying channel counts.

---

# Phase 2 — Patch-Wise Spectral Modeling

## Objective

Preserve local spectral signatures before any global aggregation.

### Current Limitation

Global pooling removes spatial variability too early.

### Target Architecture

```text
Input Cube
      │
Patch Extraction
      │
Patch Spectral Tokens
      │
Spectral Mamba
      │
Patch Spectral Features
```

### Tasks

* Generate spectral tokens per spatial patch.
* Delay global pooling until after spectral modeling.
* Allow configurable patch sizes and overlap.

### Deliverables

* Patch extraction module.
* Patch-wise spectral encoder.
* Configurable aggregation strategy.

---

# Phase 3 — Deep Residual Spectral Encoder

## Objective

Increase spectral representation capacity.

### Target Architecture

```text
Spectral Block
      │
Residual
      │
Spectral Block
      │
Residual
      │
Spectral Mamba
      │
Residual
```

### Design Features

* Residual connections.
* Configurable depth.
* Configurable kernel sizes.
* Normalization after each block.
* Optional squeeze-and-excitation.

### Deliverables

* Residual spectral block.
* Stackable encoder.
* Configurable depth parameter.

---

# Phase 4 — True Spectral Convolution

## Objective

Model local wavelength correlations directly from the image cube.

### Current Limitation

Spectral operations mainly occur after pooling or projection.

### Target Architecture

```text
Cube
 │
1D Spectral Conv
 │
Residual
 │
Normalization
 │
Activation
 │
Projection
```

### Tasks

* Apply convolutions along the spectral axis.
* Preserve spatial dimensions.
* Support multiple kernel sizes and dilation rates.

### Deliverables

* Spectral convolution module.
* Configurable kernel library.

---

# Phase 5 — Multi-Scale Spectral Branch

## Objective

Capture spectral structures at multiple scales.

### Target Architecture

```text
Kernel 3
Kernel 5
Kernel 9
Dilated Kernel
      │
Attention Fusion
```

### Tasks

* Implement parallel spectral branches.
* Add configurable fusion strategies.
* Support branch weighting.

### Deliverables

* Multi-scale spectral encoder.
* Branch attention module.

---

# Phase 6 — Bidirectional Spectral–Spatial Interaction

## Objective

Allow both streams to influence each other throughout the network.

### Current

```text
Spectral
    │
Spatial
```

### Proposed

```text
Spectral
   ↕
Spatial
   ↕
Spectral
```

### Tasks

At every stage:

* Extract spatial context.
* Update spectral features.
* Feed updated spectral context into the next stage.
* Optionally update spatial features using spectral context.

### Deliverables

* Spectral update block.
* Spatial update block.
* Bidirectional interaction interface.

---

# Phase 7 — Stage-Wise Spectral Refinement

## Objective

Continuously improve spectral representations with increasing semantic information.

### Target Architecture

```text
Stage 1
   │
Spectral Update
   │
Stage 2
   │
Spectral Update
   │
Stage 3
```

### Tasks

* Recompute spectral attention after each stage.
* Refresh FiLM or gating parameters.
* Maintain lightweight computation.

### Deliverables

* Spectral refinement module.
* Stage hooks.

---

# Phase 8 — Dynamic Band Selection v2

## Objective

Make band importance dependent on network depth and context.

### Target Architecture

```text
Input
 │
Band Selection
 │
Stage 1
 │
Band Selection
 │
Stage 2
```

### Tasks

* Compute band importance per stage.
* Support sparse and dense modes.
* Add learnable thresholds.

### Deliverables

* Stage-aware band selector.
* Optional sparsity regularization.

---

# Phase 9 — Generalized SS2D

## Objective

Extend scanning flexibility while preserving MedMamba's linear-complexity philosophy.

### Proposed Scan Modes

* Horizontal
* Vertical
* Reverse Horizontal
* Reverse Vertical
* Diagonal
* Anti-diagonal
* Custom learned directions

### Tasks

* Make scan topology configurable.
* Learn direction weights.
* Support disabling directions.

### Deliverables

* Scan registry.
* Direction weighting module.

---

# Phase 10 — Multi-Scale SS2D

## Objective

Improve spatial feature extraction across different object sizes.

### Target Architecture

```text
SS2D (3×3)
SS2D (5×5)
SS2D (7×7)
Dilated SS2D
      │
Fusion
```

### Tasks

* Parallel SS2D branches.
* Shared or independent parameters.
* Adaptive branch weighting.

### Deliverables

* Multi-scale SS2D block.

---

# Phase 11 — Pluggable Fusion Framework

## Objective

Support multiple spectral–spatial fusion strategies.

### Candidate Modules

* FiLM
* Gated additive fusion
* Cross-attention
* Multiplicative modulation
* Residual fusion

### Tasks

* Standardize fusion API.
* Benchmark fusion methods.
* Allow per-stage selection.

### Deliverables

* Fusion interface.
* Multiple fusion implementations.

---

# Phase 12 — Progressive Spectral Compression

## Objective

Compress spectral information gradually.

### Proposed Pipeline

```text
826
 │
512
 │
256
 │
128
 │
64
```

### Tasks

* Residual compression blocks.
* Configurable compression schedules.
* Learnable bottleneck widths.

### Deliverables

* Progressive compression module.

---

# Phase 13 — Multi-Level Backbone Outputs

## Objective

Expose intermediate features for downstream tasks.

### Outputs

* Stage 1 feature map.
* Stage 2 feature map.
* Stage 3 feature map.
* Final embedding.
* Spectral embedding.
* Attention maps (optional).

### Deliverables

* Unified output structure.
* Optional feature selection.

---

# Phase 14 — Continuous Spectral Positional Encoding

## Objective

Represent wavelengths using their physical values.

### Current Limitation

Positional encoding is index-based.

### Proposed

```text
450.2 nm
451.8 nm
453.4 nm
```

instead of

```text
Band 1
Band 2
Band 3
```

### Tasks

* Accept wavelength arrays as optional input.
* Encode wavelengths with continuous functions or learnable embeddings.
* Fall back to index-based encoding if wavelength metadata is unavailable.

### Deliverables

* Continuous wavelength encoder.
* Backward-compatible positional encoding.

---

# Phase 15 — Sensor-Aware Wavelength Encoding

## Objective

Generalize across different sensors with different sampling intervals.

### Design

```text
Sensor Metadata
      │
Wavelength Encoder
      │
Spectral Tokens
```

### Tasks

* Support irregular wavelength spacing.
* Normalize wavelength ranges.
* Learn sensor-independent spectral representations.

### Deliverables

* Sensor metadata interface.
* Wavelength normalization module.

---

# Phase 16 — Backbone Configuration Framework

## Objective

Expose all architectural decisions through configuration.

Example:

```yaml
model:
  tokenizer:
    type: spectral_tokens

  spectral:
    encoder_depth: 4
    multiscale: true

  interaction:
    bidirectional: true

  ss2d:
    directions:
      - horizontal
      - vertical
      - diagonal

  fusion:
    type: film

  compression:
    schedule:
      - 512
      - 256
      - 128
```

### Deliverables

* Complete configuration schema.
* Validation utilities.

---

# Phase 17 — Backend Abstraction

## Objective

Support multiple selective scan implementations without changing the architecture.

### Supported Backends

* Pure PyTorch
* CUDA
* Triton
* Future optimized kernels

### Tasks

* Standardize backend API.
* Automatic backend selection.
* Benchmarking utilities.

---

# Suggested Implementation Order

The order below prioritizes architectural foundations before optional enhancements.

| Milestone | Focus                                                       | Reason                                                            |
| --------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| **M1**    | Universal spectral tokenizer                                | Enables true sensor-agnostic inputs.                              |
| **M2**    | Patch-wise spectral modeling                                | Preserves local spectral information.                             |
| **M3**    | Deep residual spectral encoder + true spectral convolutions | Strengthens spectral feature extraction.                          |
| **M4**    | Multi-scale spectral branch                                 | Captures spectral features at different scales.                   |
| **M5**    | Bidirectional interaction + stage-wise refinement           | Allows spectral and spatial representations to co-evolve.         |
| **M6**    | Dynamic band selection v2                                   | Makes spectral emphasis context-dependent.                        |
| **M7**    | Generalized and multi-scale SS2D                            | Extends MedMamba's spatial modeling while remaining configurable. |
| **M8**    | Pluggable fusion framework + progressive compression        | Improves flexibility and efficiency.                              |
| **M9**    | Multi-level outputs                                         | Makes the backbone reusable for multiple tasks.                   |
| **M10**   | Continuous wavelength encoding + sensor-aware metadata      | Improves cross-sensor transfer and scientific correctness.        |
| **M11**   | Configuration framework + backend abstraction               | Simplifies experimentation and future optimization.               |

## Important Design Recommendation

One aspect of the earlier roadmap should be refined. The goal of "accept any image with any number of spectral dimensions" should be interpreted carefully.

A **single architecture** should absolutely support arbitrary numbers of channels. However, expecting **one trained set of weights** to perform optimally across sensors with widely different spectral ranges (e.g., RGB at 450–650 nm and HSI at 400–1000 nm with hundreds of bands) is a much stronger research problem. Achieving robust cross-sensor performance typically requires providing the model with wavelength metadata or learning wavelength-aware representations, which is exactly why Phases 14 and 15 are so valuable.

With those additions, G-MedMamba would not simply be "MedMamba for HSI"; it would become a **general spectral–spatial foundation backbone** whose architectural assumptions are based on the physics of spectral imaging rather than the fixed channel layouts used by conventional vision models.
