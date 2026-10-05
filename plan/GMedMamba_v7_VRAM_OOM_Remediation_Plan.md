# GMedMamba v7 — VRAM/OOM Remediation and Memory Management Implementation Plan

## 0. Document Purpose

This document is the implementation plan for resolving the CUDA out-of-memory (OOM) failure observed while training the GMedMamba v7 codebase with `train_example_v11.py`.

The goal is to:

1. eliminate the immediate OOM on a ~15.5 GiB GPU;
2. make VRAM use predictable and measurable;
3. preserve the intended GMedMamba architecture and experimental validity;
4. use memory-saving execution techniques before changing model capacity;
5. retain an effective batch size of 128 where practical through gradient accumulation;
6. provide controlled switches for ablation and debugging;
7. prevent future regressions by adding automated memory-budget checks and telemetry.

The implementation order is intentionally strict. **Do not begin by shrinking the model architecture.** First fix activation retention, batch execution, checkpointing, and unnecessary outputs.

---

# 1. Current Failure and Exact Configuration

## 1.1 Observed error

The failure is:

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 15.51 GiB of which 149.62 MiB is free.
Process 1839 has 322.23 MiB memory in use.
Including non-PyTorch memory, this process has 11.65 GiB memory in use.
Of the allocated memory 11.47 GiB is allocated by PyTorch,
and 10.27 MiB is reserved by PyTorch but unallocated.
```

The 20 MiB allocation is the final failed allocation, not the size of the underlying problem.

The important observations are:

- GPU capacity: approximately **15.51 GiB**;
- only approximately **149.62 MiB free** at failure;
- PyTorch allocated approximately **11.47 GiB**;
- only approximately **10.27 MiB** was reserved but unallocated;
- therefore allocator fragmentation is **not the primary diagnosis**.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` may still be supported as a secondary allocator mitigation, but it must not be treated as the principal fix.

---

## 1.2 Relevant training configuration

The failing configuration is approximately:

```text
architecture = efficient
modality = hsi

dims = (64, 128, 256)
depths = (2, 2, 2)

d_state = 8
d_ctx = 64

patch_size = 1
spectral_depth = 3

d_token = 32
compression_dims = (128, 64)

dynamic_band_selection = True
band_selection_mode = "soft"

stage_spectral_refinement = True

use_ffn = True
layerscale = True
drop_path = 0.1

fusion_type = "se_gate"

recon_mode = latent

batch_size = 128
num_workers = 4
prefetch_factor = 4

lambda_mse = 1.0
lambda_sam = 0.1
```

The immediate target hardware is a GPU with approximately:

```text
15.51 GiB total VRAM
```

---

# 2. Diagnosis

## 2.1 Primary cause: physical batch size is too large

The HSI model operates on a batch of 128 samples.

For an HSI patch of approximately:

```text
128 spectral bands
11 × 11 spatial pixels
```

the input contains:

```text
128 × 128 × 11 × 11
= 1,986,560
```

values per physical batch.

The input itself is not sufficient to explain the OOM. The problem is the number and size of intermediate tensors required for:

- spectral processing;
- spectral Mamba blocks;
- spatial Mamba/SSM processing;
- convolution branches;
- cross-branch interactions;
- FFNs;
- stage spectral refinement;
- fusion;
- latent reconstruction;
- classification;
- backward propagation.

A physical batch of 128 is therefore too aggressive for this architecture on a 15.5 GiB GPU.

### Required direction

Use a smaller **physical batch** and recover the desired effective batch through gradient accumulation.

Initial target:

```text
physical batch = 16
gradient accumulation = 8
effective batch = 128
```

Fallbacks:

```text
physical batch = 8
gradient accumulation = 16
effective batch = 128
```

or:

```text
physical batch = 4
gradient accumulation = 32
effective batch = 128
```

---

# 3. Critical Implementation Problem: Spectral Chunking Does Not Fully Save Training Memory

## 3.1 Current behavior

The spectral pathway uses chunking similar to:

```python
chunk_size = 1024

ctx_list, bw_list = [], []

for i in range(0, N, chunk_size):
    ...
    ctx_chunk, bw_chunk = self._process_patch_chunk(...)
    ctx_list.append(ctx_chunk)
    bw_list.append(...)
```

The number of spectral-spatial locations for the current configuration is approximately:

```text
B × H × W
=
128 × 11 × 11
=
15,488
```

With:

```text
chunk_size = 1024
```

this produces approximately:

```text
15,488 / 1,024 ≈ 16 chunks
```

## 3.2 Why this is a problem during training

Chunking reduces the size of each individual operation, but the current implementation stores all gradient-connected chunk outputs until concatenation.

Conceptually:

```text
chunk 1 → graph 1 retained
chunk 2 → graph 2 retained
chunk 3 → graph 3 retained
...
chunk 16 → graph 16 retained

              ↓

        torch.cat(...)
```

Therefore the implementation can retain many spectral encoder autograd graphs simultaneously.

This means:

> **The existing chunking mechanism is not equivalent to true training-time activation streaming.**

This is the highest-priority code-level issue to address.

---

# 4. Overall Implementation Priority

Implement changes in this exact order.

## P0 — Mandatory

1. Add reproducible memory profiling.
2. Fix spectral chunk training-time graph retention.
3. Add checkpointing for spectral chunks/encoder.
4. Reduce physical HSI batch size.
5. Add gradient accumulation.
6. Disable unnecessary diagnostic outputs during training.
7. Ensure AMP/BF16 is used consistently.
8. Add explicit peak-VRAM telemetry.

## P1 — Strongly recommended

9. Add spatial-block activation checkpointing.
10. Profile latent reconstruction separately.
11. Make spectral chunk size configurable.
12. Add automatic/safe batch-size discovery.
13. Add configurable memory budget.
14. Add allocator configuration as a secondary mitigation.
15. Optimize DataLoader behavior only after GPU activation memory is fixed.

## P2 — Diagnostic/experimental

16. Benchmark FFN enabled vs disabled.
17. Benchmark stage spectral refinement enabled vs disabled.
18. Benchmark different reconstruction widths.
19. Benchmark different spectral chunk sizes.
20. Benchmark model-capacity reductions only if execution-level optimizations are insufficient.

---

# 5. Phase 0 — Establish a Baseline

## 5.1 Do not modify architecture first

Before changing model dimensions, create a reproducible baseline.

Run the exact failing configuration and capture:

- maximum allocated memory;
- maximum reserved memory;
- allocated memory immediately before forward;
- allocated memory after spectral pathway;
- allocated memory after each backbone stage;
- allocated memory after reconstruction;
- allocated memory after loss;
- peak memory after backward;
- peak memory after optimizer step;
- whether OOM occurs during forward, loss, backward, or optimizer step.

## 5.2 Required helper

Create a memory utility, for example:

```text
utils/memory.py
```

or the project's existing utilities module if an appropriate location already exists.

Implement:

```python
def cuda_memory_snapshot(tag: str):
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()

    print(
        f"[VRAM] {tag}: "
        f"allocated={allocated / 1024**3:.3f} GiB, "
        f"reserved={reserved / 1024**3:.3f} GiB, "
        f"peak_allocated={max_allocated / 1024**3:.3f} GiB, "
        f"peak_reserved={max_reserved / 1024**3:.3f} GiB"
    )
```

Do not call `torch.cuda.empty_cache()` between every operation. That hides useful memory behavior and can hurt performance.

## 5.3 Reset peak counters

At the beginning of each profiled training iteration:

```python
torch.cuda.reset_peak_memory_stats()
```

Then record:

```text
forward
loss
backward
optimizer
```

---

# 6. Phase 1 — Fix Spectral Chunk Graph Retention

## 6.1 Required objective

The implementation must ensure that spectral chunking does not retain all intermediate encoder activations for the complete batch.

The target behavior is:

```text
chunk
 ↓
checkpointed computation
 ↓
only required output retained
 ↓
next chunk
```

rather than:

```text
chunk 1 → complete graph retained
chunk 2 → complete graph retained
...
chunk 16 → complete graph retained
 ↓
concatenate
```

## 6.2 Do not detach outputs

Do NOT solve the problem by doing:

```python
ctx_chunk = ctx_chunk.detach()
```

or:

```python
ctx = ctx.detach()
```

inside the training path.

That would break gradient flow and change the optimization behavior.

The objective is to reduce **stored activations**, not remove autograd.

---

# 7. Phase 1A — Add Checkpointed Spectral Encoder

Use PyTorch activation checkpointing around the expensive spectral encoder computation.

Conceptually:

```python
from torch.utils.checkpoint import checkpoint
```

Then structure the expensive operation as a function whose output can be recomputed during backward.

For example:

```python
def _spectral_encoder_forward(self, tokens, wavelengths=None):
    ...
    return encoded
```

Then:

```python
encoded = checkpoint(
    self._spectral_encoder_forward,
    tokens,
    wavelengths,
    use_reentrant=False,
)
```

The exact implementation must follow the existing function signatures in v7.

## 7.1 Important checkpoint requirements

Test all supported cases:

```text
wavelengths = None
```

and:

```text
wavelengths = tensor/vector of wavelengths
```

Ensure:

- gradients remain enabled;
- model parameters receive gradients;
- output values match the non-checkpointed version within numerical tolerance;
- no unsupported non-Tensor checkpoint arguments are accidentally passed;
- the modern `use_reentrant=False` mode is used where supported.

---

# 8. Phase 1B — Make Spectral Chunk Size Configurable

Remove the hard-coded:

```python
chunk_size = 1024
```

and expose:

```text
--spectral_chunk_size
```

Recommended choices:

```text
256
512
1024
2048
```

Initial production value:

```text
512
```

## 8.1 Expected effect

With approximately 15,488 spectral-spatial locations:

```text
1024 → approximately 16 chunks
512  → approximately 31 chunks
256  → approximately 61 chunks
```

The purpose is not simply to increase the number of chunks.

The combination should be:

```text
smaller chunk
+
checkpointed computation
=
lower activation peak
```

Without checkpointing, simply reducing the chunk size may not provide the desired training-memory reduction because the graphs can still accumulate.

---

# 9. Phase 1C — Avoid Unnecessary `band_weights` Retention

The spectral pathway constructs band-selection weights.

Separate:

```text
band weights required for model computation
```

from:

```text
band weights returned only for diagnostics/visualization
```

Introduce:

```text
return_band_weights = False
```

during normal training.

Only enable:

```text
return_band_weights = True
```

for:

- visualization;
- band-selection analysis;
- ablation;
- debugging.

Do not detach band weights merely to hide the problem if they are required by the loss. Determine whether the training objective actually consumes them first.

---

# 10. Phase 2 — Reduce Physical Batch Size

## 10.1 New HSI starting configuration

Change HSI training from:

```text
batch_size = 128
```

to:

```text
batch_size = 16
```

This is the first target for the 15.51-GiB GPU.

## 10.2 Do not confuse physical and effective batch size

The physical batch is the number of samples simultaneously resident on the GPU.

The effective batch can remain 128 through gradient accumulation.

Target:

```text
physical batch = 16
accumulation = 8
effective batch = 128
```

Formula:

```text
effective_batch =
    physical_batch × gradient_accumulation_steps
```

---

# 11. Phase 3 — Implement Correct Gradient Accumulation

Add:

```text
--grad_accum_steps
```

with:

```text
default = 8
```

when:

```text
batch_size = 16
```

## 11.1 Correct training pattern

The intended logic is:

```python
optimizer.zero_grad(set_to_none=True)

for accumulation_idx in range(grad_accum_steps):
    outputs = model(...)
    loss = compute_loss(...) / grad_accum_steps
    scaler.scale(loss).backward()

scaler.step(optimizer)
scaler.update()
```

The actual code must preserve the project's existing:

- AMP;
- scaler;
- scheduler;
- optimizer;
- gradient clipping;
- logging.

## 11.2 Critical requirement

Scale the loss:

```python
loss = loss / grad_accum_steps
```

before backward.

Otherwise gradients become approximately `grad_accum_steps` times too large.

## 11.3 Optimizer step semantics

The optimizer should step once per effective batch:

```text
8 physical mini-batches
        ↓
8 backward calls
        ↓
1 optimizer.step()
```

Do not call `optimizer.step()` after every physical batch if the goal is a true effective batch of 128.

---

# 12. Phase 3A — Scheduler Semantics

Check whether the learning-rate scheduler currently advances:

```text
per physical batch
```

or:

```text
per optimizer step
```

After gradient accumulation, it should normally advance according to **optimizer updates**, not raw DataLoader iterations, unless the existing experimental protocol explicitly requires otherwise.

Record the change because it affects reproducibility.

---

# 13. Phase 3B — Gradient Clipping

If gradient clipping exists, apply it immediately before the optimizer step:

```python
scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm
)
```

then:

```python
scaler.step(optimizer)
scaler.update()
```

Do not clip each accumulation micro-batch independently unless that behavior is explicitly intended.

---

# 14. Phase 4 — Disable Unnecessary Intermediate Outputs During Training

The model currently maintains structures conceptually equivalent to:

```text
stage_pools
stage_feature_maps
stage_ctx_maps
```

These should become optional.

Add:

```text
return_intermediates = False
```

or an equivalent configuration.

## 14.1 Training mode

Normal training should return only what the trainer actually needs:

```text
classification output
+
reconstruction output
+
required loss information
```

Do not construct lists of all stage tensors if they are not consumed.

## 14.2 Debug/evaluation mode

Allow:

```text
return_intermediates = True
```

for:

- feature visualization;
- debugging;
- research analysis;
- ablation.

This preserves functionality without imposing its memory cost on every training iteration.

---

# 15. Phase 5 — Activation Checkpointing for Spatial Backbone

After spectral checkpointing and physical batch reduction are working, add checkpointing to the expensive spatial blocks.

The main candidate is:

```text
FullChannelGBlock
```

because the efficient architecture still uses full channel width in both major branches.

At the later stage the block operates around:

```text
256 channels
```

and simultaneously maintains:

- SSM branch;
- convolution branch;
- cross-branch interactions;
- gating;
- residual/intermediate tensors;
- FFN activations.

## 15.1 Checkpoint target order

Use this order:

1. spectral encoder;
2. `ResidualSpectralBlock`;
3. `FullChannelGBlock`;
4. entire stages only if still necessary.

Do not checkpoint everything immediately. Measure after each level.

---

# 16. Phase 5A — Add Configurable Checkpointing

Expose:

```text
--gradient_checkpointing
```

with modes such as:

```text
none
spectral
spatial
all
```

Recommended initial configuration:

```text
gradient_checkpointing = spectral
```

Then test:

```text
spectral
```

followed by:

```text
spectral + spatial
```

Use the least expensive checkpointing configuration that fits the VRAM budget.

---

# 17. Phase 6 — AMP/BF16

Ensure mixed precision is explicit and consistently applied.

Prefer BF16 when the GPU supports it and the existing numerical behavior is acceptable.

Expose:

```text
--amp
--amp_dtype {bf16,fp16}
```

or use the project's existing equivalent.

Record in the experiment report:

```text
precision = bf16
```

or:

```text
precision = fp16
```

Do not silently change precision between experiments.

---

# 18. Phase 6A — Numerical Validation After AMP/Checkpointing

Before doing a long training run, compare:

```text
AMP off / checkpoint off
```

against:

```text
AMP on / checkpoint on
```

for a small deterministic batch.

Compare:

- classification logits;
- reconstruction output;
- total loss;
- MSE loss;
- SAM loss;
- gradients;
- parameter update direction.

Small floating-point differences are expected. Large deviations are not.

---

# 19. Phase 7 — Latent Reconstruction Profiling

The latent reconstruction path should be profiled separately.

Current conceptual path:

```text
[B, 256, H', W']
        ↓
interpolate
        ↓
decoder hidden representation
        ↓
convolution layers
        ↓
[B, 128, H, W]
```

The decoder contributes to peak memory because its output participates in:

```text
MSE
+
SAM
```

and therefore must remain differentiable.

## 19.1 Do not optimize blindly

Run:

```text
backbone + classification
```

versus:

```text
backbone + classification + latent reconstruction
```

with the same batch.

Measure:

```text
peak VRAM difference
```

before changing reconstruction architecture.

---

# 20. Phase 7A — Optional Reconstruction Width

If reconstruction is demonstrated to be a substantial memory contributor, expose:

```text
--recon_hidden
```

Candidate values:

```text
128
64
32
```

Keep:

```text
128
```

as the research/default architecture unless evidence requires otherwise.

Any reduction in reconstruction width must be documented as an architectural variant.

---

# 21. Phase 8 — FFN Diagnostic Switch

The current configuration has:

```text
use_ffn = True
```

Do not disable it in the main model merely to fit VRAM.

Instead expose:

```text
--use_ffn
```

or:

```text
--ffn {on,off}
```

Then run a controlled memory comparison:

```text
FFN ON
vs
FFN OFF
```

Record:

- peak VRAM;
- throughput;
- training loss;
- validation metrics.

Only use FFN-off as an architecture ablation unless the project explicitly decides otherwise.

---

# 22. Phase 9 — Stage Spectral Refinement Diagnostic

The current configuration has:

```text
stage_spectral_refinement = True
```

This should remain enabled in the main model.

Expose a switch for diagnostic comparison:

```text
--stage_spectral_refinement
```

Benchmark:

```text
enabled
vs
disabled
```

Record memory and accuracy effects.

Do not permanently remove it just to solve the OOM.

---

# 23. Phase 10 — Memory Budget

Introduce a configurable memory target.

For a 15.51-GiB GPU, target a maximum peak around:

```text
13.0–13.5 GiB
```

rather than trying to use all 15.5 GiB.

This leaves room for:

- CUDA runtime;
- non-PyTorch allocations;
- temporary kernels;
- cuDNN/cuBLAS workspaces;
- allocator overhead;
- validation;
- future code changes.

Expose:

```text
--max_vram_gib
```

or:

```text
--max_vram_fraction
```

Example:

```text
max_vram_fraction = 0.87
```

The exact default should be validated on the target GPU.

---

# 24. Phase 11 — Automatic Safe Batch Size

Add:

```text
--batch_size auto
```

The auto-tuner must test the **actual training path**, not only inference.

Each candidate must perform:

```text
forward
loss
backward
optimizer-compatible gradient calculation
```

Candidate search:

```text
128
64
32
16
8
4
```

Stop when the batch fits within the memory target.

Do not automatically choose a batch whose peak is within a few MiB of total VRAM.

---

# 25. Phase 11A — Auto Batch and Gradient Accumulation

If the requested effective batch is:

```text
128
```

and the auto-selected physical batch is:

```text
16
```

automatically derive:

```text
grad_accum_steps = 8
```

If:

```text
physical batch = 8
```

derive:

```text
grad_accum_steps = 16
```

Require:

```text
effective_batch % physical_batch == 0
```

unless the trainer explicitly supports a partial final accumulation.

---

# 26. Phase 12 — DataLoader Memory Management

The current DataLoader settings include:

```text
num_workers = 4
prefetch_factor = 4
```

These primarily affect host RAM and pinned-memory pressure rather than the fundamental GPU activation OOM.

Do not treat them as the primary fix.

After GPU memory is stable, benchmark:

```text
num_workers = 0, 2, 4
```

and:

```text
prefetch_factor = 2, 4
```

if applicable.

Avoid excessive prefetching.

If `pin_memory=True`, retain it only if it improves transfer performance without causing system-memory pressure.

---

# 27. Phase 13 — CUDA Allocator Configuration

As a secondary safeguard, test:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

However, the current error indicates that allocator fragmentation is not the main cause.

Do not use allocator configuration to hide an activation-memory problem.

The correct order is:

```text
reduce actual peak activation memory
        ↓
then evaluate allocator behavior
```

---

# 28. Phase 14 — Memory-Safe Training Loop

The final training loop should have this conceptual structure:

```text
initialize model
initialize optimizer
initialize AMP
initialize checkpoint configuration

for epoch:

    optimizer.zero_grad(set_to_none=True)

    for micro_batch in loader:

        move only required tensors to GPU

        autocast:
            forward
            loss

        loss /= grad_accum_steps

        backward

        if accumulation boundary:
            unscale gradients
            clip gradients if enabled
            optimizer.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        log memory at configured intervals

    validation
```

Important:

- Do not call `retain_graph=True` unless absolutely required.
- Do not store tensors requiring gradients in Python lists for logging.
- Convert logged scalar tensors to Python numbers with `.item()`.
- Do not retain model outputs across iterations unnecessarily.
- Do not append full GPU tensors to experiment-history structures.
- Detach tensors used solely for visualization/logging.

---

# 29. Tensor Retention Audit

Search the entire v7 training code for:

```text
retain_graph=True
```

and remove it unless a specific backward dependency requires it.

Search for patterns like:

```python
history.append(loss)
```

and replace with:

```python
history.append(loss.item())
```

when only the scalar value is needed.

Search for:

```python
predictions.append(output)
```

and determine whether:

```python
output.detach().cpu()
```

is required.

Never retain full GPU tensors for epoch-level history unless explicitly necessary.

---

# 30. Optimizer Memory Audit

Optimizer state is separate from activation memory.

Check whether the optimizer is Adam/AdamW or another stateful optimizer.

For Adam-like optimizers, parameters may have:

- parameter tensor;
- gradient tensor;
- first moment;
- second moment.

Record:

```text
model parameter memory
optimizer state memory
activation memory
```

separately if possible.

Do not change optimizer merely to solve this OOM unless profiling demonstrates that optimizer state is a significant component.

---

# 31. Memory Telemetry Requirements

Every training experiment should be able to report:

```text
GPU total
GPU free at start

model parameters
optimizer state
input batch size

peak allocated
peak reserved

spectral peak
stage 1 peak
stage 2 peak
stage 3 peak

reconstruction peak
loss peak
backward peak
optimizer-step peak
```

At minimum, record:

```text
peak_allocated_gib
peak_reserved_gib
```

for the full iteration.

---

# 32. Required Diagnostic Checkpoints

Add memory probes after:

```text
1. batch transferred to GPU
2. spectral pathway
3. Stage 1
4. Stage 2
5. Stage 3
6. fusion
7. reconstruction
8. loss calculation
9. backward
10. optimizer step
```

This identifies the true peak rather than relying on the location where CUDA finally reports OOM.

---

# 33. Test Matrix

Use the exact failing model configuration.

## Test A — Baseline

```text
batch = 128
checkpoint = off
reconstruction = latent
```

Expected:

```text
OOM
```

Purpose:

Reproduce the failure.

---

## Test B — Batch reduction

```text
batch = 16
grad_accum = 8
checkpoint = off
reconstruction = latent
```

Purpose:

Measure batch-size contribution.

---

## Test C — Spectral checkpointing

```text
batch = 16
grad_accum = 8
spectral_checkpoint = on
reconstruction = latent
```

Purpose:

Measure the effect of fixing spectral activation retention.

---

## Test D — Smaller spectral chunks

```text
batch = 16
grad_accum = 8
spectral_checkpoint = on
spectral_chunk_size = 512
```

Purpose:

Find the best memory/performance tradeoff.

---

## Test E — Spatial checkpointing

```text
batch = 16
grad_accum = 8
spectral_checkpoint = on
spatial_checkpoint = on
```

Purpose:

Measure spatial backbone activation cost.

---

## Test F — Reconstruction contribution

Compare:

```text
recon = none
```

against:

```text
recon = latent
```

with all other settings identical.

---

## Test G — Batch 32

If Test E fits comfortably:

```text
batch = 32
grad_accum = 4
```

Target:

```text
effective batch = 128
```

---

## Test H — Batch 64

Only if Test G leaves sufficient memory:

```text
batch = 64
grad_accum = 2
```

---

## Test I — Batch 128

Do not make this a requirement.

```text
batch = 128
grad_accum = 1
```

Only attempt it if profiling shows the optimized model fits comfortably under the memory budget.

The research protocol should not depend on forcing batch 128 into 15.5 GiB VRAM.

---

# 34. Acceptance Criteria

A fix is accepted only if all of the following are true.

## Memory

The complete training iteration fits within the target:

```text
peak VRAM ≤ approximately 13.0–13.5 GiB
```

on the 15.51-GiB GPU.

## Correctness

Training completes:

```text
forward
loss
backward
optimizer step
```

without OOM.

## Gradient correctness

Checkpointing and accumulation do not silently remove gradients.

Verify:

```text
parameter.grad is not None
```

for expected trainable parameters.

## Numerical behavior

Checkpointed and non-checkpointed outputs match within an appropriate floating-point tolerance.

## Effective batch

The intended effective batch is maintained:

```text
physical batch × accumulation = effective batch
```

## No accidental graph retention

No training-history or diagnostic list should retain full gradient-connected GPU tensors across iterations.

## Performance

Record:

```text
iterations/sec
samples/sec
seconds/epoch
```

because checkpointing trades memory for recomputation.

---

# 35. Architecture Preservation Rules

The main GMedMamba experiment should preserve:

```text
dims = (64,128,256)
depths = (2,2,2)
d_state = 8
d_ctx = 64
d_token = 32
spectral_depth = 3
dynamic band selection = True
stage spectral refinement = True
FFN = True
latent reconstruction = True
fusion = se_gate
```

unless a separate architecture ablation is intentionally being performed.

The following are execution/memory changes and should not be considered architectural changes:

```text
smaller physical batch
gradient accumulation
activation checkpointing
AMP/BF16
spectral chunking
optional diagnostic outputs
memory telemetry
allocator configuration
```

---

# 36. Architecture-Change Escalation Policy

Only modify architecture if all of the following have been attempted:

1. physical batch reduction;
2. gradient accumulation;
3. spectral checkpointing;
4. spatial checkpointing;
5. unnecessary-output removal;
6. AMP/BF16;
7. reconstruction profiling;
8. safe chunk sizing.

Then consider, in order:

```text
1. recon_hidden: 128 → 64
2. FFN diagnostic
3. stage spectral refinement diagnostic
4. d_token reduction
5. spectral depth reduction
6. compression dimension reduction
7. stage dimension reduction
```

Every such change must be labeled as an architecture variant.

---

# 37. Recommended Initial Production Configuration

For the 15.51-GiB GPU, start with:

```text
architecture = efficient
modality = hsi

dims = (64,128,256)
depths = (2,2,2)

d_state = 8
d_ctx = 64

patch_size = 1
spectral_depth = 3

d_token = 32
compression_dims = (128,64)

dynamic_band_selection = True
band_selection_mode = soft

stage_spectral_refinement = True

use_ffn = True
layerscale = True
drop_path = 0.1

fusion_type = se_gate
recon_mode = latent

batch_size = 16
grad_accum_steps = 8

spectral_chunk_size = 512

gradient_checkpointing = spectral
```

After validation, test:

```text
gradient_checkpointing = all
```

if necessary.

---

# 38. Recommended Escalation Configurations

## Configuration 1 — Conservative

```text
batch = 16
accumulation = 8
spectral_chunk = 512
checkpoint = spectral
```

## Configuration 2 — More memory-safe

```text
batch = 8
accumulation = 16
spectral_chunk = 512
checkpoint = spectral + spatial
```

## Configuration 3 — Maximum memory conservation

```text
batch = 4
accumulation = 32
spectral_chunk = 256
checkpoint = all
```

Use Configuration 3 only if necessary because recomputation will increase training time.

---

# 39. What Not To Do

Do not:

```text
- simply call torch.cuda.empty_cache() every iteration;
- detach spectral outputs to prevent OOM;
- disable gradients;
- use torch.no_grad() during training;
- reduce model dimensions before profiling;
- assume the 20 MiB allocation is the real problem;
- assume allocator fragmentation is the main problem;
- retain full GPU tensors for logging;
- use batch 128 as a hard requirement;
- silently change the effective batch size;
- silently remove FFNs or spectral refinement from the research model.
```

---

# 40. Implementation Checklist

## Code changes

- [ ] Add VRAM telemetry utility.
- [ ] Add `spectral_chunk_size`.
- [ ] Fix spectral chunk autograd retention.
- [ ] Add spectral activation checkpointing.
- [ ] Add `return_intermediates`.
- [ ] Add `return_band_weights`.
- [ ] Add physical batch configuration.
- [ ] Add `grad_accum_steps`.
- [ ] Correctly scale accumulated loss.
- [ ] Correct optimizer-step frequency.
- [ ] Correct scheduler-step frequency.
- [ ] Preserve gradient clipping semantics.
- [ ] Add spatial checkpointing.
- [ ] Expose checkpointing modes.
- [ ] Expose AMP dtype.
- [ ] Add reconstruction profiling.
- [ ] Add memory budget.
- [ ] Add optional automatic batch-size detection.
- [ ] Audit `retain_graph=True`.
- [ ] Audit Python lists storing GPU tensors.
- [ ] Audit experiment logging for graph retention.

## Validation

- [ ] Reproduce baseline OOM.
- [ ] Verify batch 16 fits.
- [ ] Verify accumulated effective batch = 128.
- [ ] Verify checkpointed output numerically.
- [ ] Verify gradients.
- [ ] Verify optimizer step.
- [ ] Verify validation.
- [ ] Verify checkpoint saving/loading.
- [ ] Verify inference.
- [ ] Verify reconstruction.
- [ ] Verify band-selection diagnostics.
- [ ] Verify no hidden GPU-tensor accumulation.

## Reporting

- [ ] Record peak allocated VRAM.
- [ ] Record peak reserved VRAM.
- [ ] Record training throughput.
- [ ] Record epoch duration.
- [ ] Record effective batch size.
- [ ] Record physical batch size.
- [ ] Record checkpoint mode.
- [ ] Record spectral chunk size.
- [ ] Record precision.
- [ ] Record GPU model.
- [ ] Record software/PyTorch version.

---

# 41. Final Target Architecture/Execution Separation

The final implementation should conceptually separate:

```text
GMedMamba architecture
```

from:

```text
GMedMamba execution policy
```

Architecture:

```text
spectral pathway
+
spatial backbone
+
spectral refinement
+
fusion
+
classification
+
latent reconstruction
```

Execution policy:

```text
physical batch
+
gradient accumulation
+
activation checkpointing
+
mixed precision
+
spectral chunk size
+
diagnostic-output policy
+
memory budget
```

This is important for the thesis because the memory optimization can then be reported as an implementation/execution improvement rather than an undocumented architecture modification.

---

# 42. Final Diagnosis

The OOM should be treated as an **activation-memory problem**, not primarily a CUDA allocator problem.

The highest-priority issue in the v7 implementation is the interaction between:

```text
batch_size = 128
```

and the spectral pathway's chunking behavior:

```text
15,488 spectral-spatial locations
÷
1,024 chunk size
≈
16 chunks
```

where gradient-connected chunk outputs are retained until concatenation.

The full-channel spatial blocks add substantial activation memory because both branches operate at full channel width and include cross-branch interactions and FFNs.

Latent reconstruction adds another gradient-connected path, but it should be profiled rather than assumed to be the main cause.

Therefore the correct remediation order is:

```text
1. Profile exact memory peaks
        ↓
2. Fix spectral chunk autograd retention
        ↓
3. Checkpoint spectral computation
        ↓
4. Reduce physical batch 128 → 16
        ↓
5. Restore effective batch 128 using accumulation
        ↓
6. Remove unnecessary intermediate outputs
        ↓
7. Add AMP/BF16
        ↓
8. Checkpoint spatial blocks if needed
        ↓
9. Profile reconstruction
        ↓
10. Add automatic memory/batch management
        ↓
11. Only then consider architectural reductions
```

The objective is not merely to make one run stop crashing. The objective is to make GMedMamba v7 **memory-bounded, reproducible, diagnosable, and capable of maintaining the intended effective training batch without unnecessarily changing the model architecture**.
