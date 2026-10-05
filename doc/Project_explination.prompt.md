Act as a **senior AI/ML researcher, software architect, and production-level software engineer** with strong expertise in deep learning, computer vision, medical imaging, PyTorch, model architecture design, and technical documentation.

Your task is to create a **very detailed Markdown documentation file** that serves as the definitive technical history and architecture document for this project.

 ## Source Projects / Inspirations

 The project was originally inspired by:

 - **MedMamba:** https://github.com/YubiaoYue/MedMamba
- **Tiny Recursive Models (TRM):** https://github.com/SamsungSAILMontreal/TinyRecursiveModels

 You must inspect and understand these repositories before writing the documentation.

 The documentation must clearly distinguish between:

 1. What was inherited or inspired by the original MedMamba implementation.
2. What was modified, redesigned, optimized, or newly implemented in this project.
3. What was inspired by or adapted from the Tiny Recursive Models (TRM) research/codebase.
4. What is completely original to this project.
5. What is an implementation detail versus an architectural or research-level change.

 Do **not** assume that an implementation is original simply because it differs syntactically from the source repositories. Analyze the underlying concepts, architecture, algorithms, and behavior.

---

 # Main Objective

 Generate a comprehensive Markdown file documenting the **entire evolution of this project**, from the original MedMamba inspiration through all subsequent architectural, implementation, training, optimization, and TRM-related changes.

 The document should be useful to:

 - A senior ML engineer joining the project.
- A researcher trying to understand the architectural contributions.
- A developer maintaining or extending the codebase.
- Someone reviewing the project for reproducibility.
- Someone comparing this implementation against MedMamba and TRM.
- Someone trying to determine exactly which components are inherited, modified, adapted, or novel.

 The final document should be detailed enough that another experienced engineer could understand the project's architecture and development history **without having to reverse-engineer the repository from scratch**.

---

 # Repository Analysis

 Before writing the document, inspect the current repository thoroughly.

 Analyze, where applicable:

 - Directory structure
- Python modules
- Model definitions
- Neural network blocks
- Training scripts
- Dataset implementations
- Configuration files
- Loss functions
- Optimizers
- Schedulers
- Augmentation pipelines
- Evaluation code
- Checkpointing
- Logging
- Inference code
- CLI interfaces
- Utility functions
- Tests
- Experiment scripts
- Model configuration
- Hyperparameters
- Any TRM-related implementation
- Any modifications to the original MedMamba architecture

 Trace imports and dependencies so that you understand how the components actually interact.

 Do not document files merely based on their filenames. Follow the code and explain what each important component actually does.

---

 # Required Documentation Structure

 Create a single detailed Markdown document with a structure similar to the following.

 ## 1\. Project Overview

 Explain:

 - What the project does.
- The primary problem it solves.
- The model architecture at a high level.
- The role of MedMamba.
- The role of TRM.
- The major architectural changes made in this project.
- The overall evolution of the system.

 Include a concise architecture summary before going into implementation details.

---

 ## 2\. Original Inspiration: MedMamba

 Document the original MedMamba architecture as it relates to this project.

 Explain:

 - The original motivation.
- Core architecture.
- Major building blocks.
- Mamba-related components.
- Vision processing pipeline.
- Feature extraction.
- Downsampling.
- Classification/segmentation head, where applicable.
- Data flow.
- Important design decisions.

 Clearly identify which parts of the current implementation originate from MedMamba.

 Where useful, provide:

```
Original MedMamba
        ↓
Component
        ↓
Current Project Equivalent
        ↓
Modification
```

 Do not simply summarize the GitHub README. Analyze the actual implementation.

---

 # 3\. Baseline Architecture

 Describe what the project looked like when starting from the original inspiration.

 Document:

 - Initial model structure.
- Initial training pipeline.
- Initial data pipeline.
- Initial losses.
- Initial optimization strategy.
- Initial configuration.
- Initial limitations or bottlenecks.

 Explain the baseline clearly enough that later changes can be compared against it.

---

 # 4\. Complete Change History

 Create a detailed chronological or logical history of **every significant change made to the project**.

 For each change, document:

 ### Change Name

 **Problem:**\
 What problem or limitation motivated the change?

 **Original behavior:**\
 How did the system work before?

 **New behavior:**\
 How does it work now?

 **Implementation:**\
 Which files/classes/functions were changed?

 **Architectural impact:**\
 Did the change modify the model architecture?

 **Training impact:**\
 Did it affect optimization, convergence, stability, memory usage, or compute?

 **Inference impact:**\
 Did it affect latency, memory, throughput, or accuracy?

 **Reasoning:**\
 Why was this approach chosen?

 **Trade-offs:**\
 What did the change improve and what did it potentially make worse?

 **Relationship to source projects:**

 - MedMamba-derived
- TRM-derived
- Adapted from existing research
- Independent/original
- Engineering optimization

 Repeat this for every major modification.

---

 # 5\. Architecture Changes

 Provide a dedicated deep dive into all architectural changes.

 For every modified or newly introduced module:

 - Explain its purpose.
- Explain its inputs and outputs.
- Explain tensor shapes.
- Explain the mathematical operation where relevant.
- Explain how it interacts with surrounding modules.
- Explain why it exists.
- Explain what it replaced.
- Explain whether it is derived from MedMamba, TRM, another source, or independently designed.

 Include diagrams using Mermaid when useful.

 For example:

 Mermaid flowchart: Input, Feature Extraction, Modified Block, Recursive Reasoning, Output

---

 # 6\. TRM Integration

 Create a substantial section dedicated specifically to the **Tiny Recursive Model implementation**.

 Use the official/reference repository:

 https://github.com/SamsungSAILMontreal/TinyRecursiveModels

 Explain the TRM architecture and principles relevant to this implementation.

 Document:

 - Why TRM was introduced into this project.
- What problem recursive reasoning is intended to solve.
- The original TRM design principles.
- Which parts of TRM were implemented.
- Which parts were adapted.
- Which parts were intentionally changed.
- Which parts were not used.
- Why those decisions were made.

 Trace the implementation from the source TRM concepts to the actual project code.

---

 # 7\. TRM Implementation Mapping

 Create a detailed mapping table.

 Example:

 | TRM Concept | Original TRM | Current Implementation | Modification | Reason |
| --- | --- | --- | --- | --- |
| Recursive block | ... | ... | ... | ... |
| Hidden state | ... | ... | ... | ... |
| Recursion | ... | ... | ... | ... |
| Output head | ... | ... | ... | ... |

 Be precise.

 Do not claim something is copied from TRM unless the implementation or underlying algorithm supports that conclusion.

 Distinguish between:

 - Direct implementation
- Conceptual inspiration
- Architectural adaptation
- Implementation rewrite
- Independent implementation

---

 # 8\. Recursive Reasoning Mechanism

 Explain the recursive mechanism in depth.

 Cover:

 - State representation.
- Initialization.
- Recursion loop.
- State updates.
- Hidden representations.
- Input injection.
- Output prediction.
- Halting/stopping behavior if applicable.
- Gradient flow.
- Number of recursive iterations.
- Training versus inference behavior.
- Computational complexity.
- Memory implications.

 Include equations where useful.

 For example, if the architecture can be represented as:

```
z_{t+1} = f(x, z_t)
```

 explain exactly what `x`, `z_t`, and `f` represent in this implementation.

 Do not invent equations. Derive them from the actual code.

---

 # 9\. Full Model Architecture

 Document the complete current architecture from input to output.

 Include:

 - Input shape.
- Embedding/stem.
- Encoder stages.
- Mamba components.
- Attention/components if present.
- Downsampling.
- Feature dimensions.
- Recursive/TRM components.
- Decoder/head if present.
- Output dimensions.

 Provide tensor-shape flow wherever possible.

 For example:

```
Input
[B, C, H, W]
      ↓
Stem
[B, C1, H1, W1]
      ↓
Stage 1
[B, C2, H2, W2]
      ↓
...
      ↓
TRM / Recursive Reasoning
[B, Cn, Hn, Wn]
      ↓
Head
[B, num_classes]
```

 Use the actual dimensions found in the code rather than placeholders.

---

 # 10\. Data Pipeline

 Document the complete data pipeline.

 Include:

 - Dataset format.
- Dataset loading.
- Preprocessing.
- Resizing/cropping.
- Normalization.
- Augmentation.
- Train/validation/test splits.
- Sampling strategy.
- Class balancing.
- Batch construction.
- Any special handling for medical images.

 Explain why each important transformation exists.

---

 # 11\. Training Pipeline

 Document the training system in detail.

 Include:

 - Loss functions.
- Optimizer.
- Learning-rate schedule.
- Warmup.
- Weight decay.
- Gradient clipping.
- Mixed precision.
- Gradient accumulation.
- EMA if present.
- Checkpointing.
- Early stopping.
- Validation.
- Metrics.
- Random seeds.
- Reproducibility considerations.

 Explain how the training pipeline differs from the original inspiration.

---

 # 12\. Hyperparameters

 Create a comprehensive hyperparameter reference.

 Include:

 | Parameter | Value | Purpose | Where Defined |
| --- | --- | --- | --- |
| Batch size | ... | ... | ... |
| Learning rate | ... | ... | ... |
| Weight decay | ... | ... | ... |
| Recursion steps | ... | ... | ... |
| Hidden dimension | ... | ... | ... |

 Separate:

 - Model hyperparameters
- Training hyperparameters
- Data hyperparameters
- TRM hyperparameters
- Runtime/inference parameters

---

 # 13\. Computational Complexity

 Analyze the computational characteristics of the architecture.

 Discuss, where possible:

 - Parameter count.
- FLOPs.
- Memory complexity.
- Activation memory.
- Training cost.
- Inference cost.
- Effect of recursion depth.
- Effect of input resolution.
- Scaling behavior.

 Compare the current implementation conceptually against:

 - Original MedMamba
- TRM/reference implementation

 Do not fabricate benchmark numbers. If exact measurements are unavailable, explicitly state that.

---

 # 14\. Performance and Experiments

 Document all experiments found in the repository.

 For each experiment include:

 - Configuration.
- Dataset.
- Model variant.
- Training setup.
- Number of epochs/steps.
- Important hyperparameters.
- Metrics.
- Results.
- Checkpoint.
- Purpose of the experiment.

 If results are unavailable, explicitly mark them as unavailable rather than guessing.

 Where possible, create comparison tables.

---

 # 15\. Ablation Analysis

 Identify existing or implied ablations.

 Discuss the effect of:

 - Removing TRM.
- Changing recursion depth.
- Removing architectural modifications.
- Changing Mamba components.
- Changing normalization.
- Changing feature dimensions.
- Changing training strategy.

 Only report actual experimental results if they exist.

 For hypothetical ablations, clearly label them as **proposed experiments**, not completed experiments.

---

 # 16\. Code-Level Change Map

 Create a detailed mapping of important source files.

 Example:

 | File | Component | Original Source | Current Role | Major Changes |
| --- | --- | --- | --- | --- |
| `...` | ... | MedMamba | ... | ... |
| `...` | ... | TRM | ... | ... |
| `...` | ... | Original | ... | ... |

 Include classes and important functions where useful.

---

 # 17\. Dependency / Attribution Analysis

 Clearly document the project's intellectual and implementation lineage.

 Create categories such as:

 ### MedMamba-derived

 List components that originate from or are strongly based on MedMamba.

 ### TRM-derived

 List components based on TRM concepts or implementation.

 ### Other external research

 List other identifiable external inspirations.

 ### Original / Project-specific

 List components that appear to have been independently designed for this project.

 For each external source, explain exactly what was used or inspired by it.

 Avoid overstating originality.

---

 # 18\. Important Engineering Decisions

 Document important decisions that are not necessarily visible from the architecture diagram.

 Examples:

 - Why a particular normalization was chosen.
- Why recursion happens at a particular feature level.
- Why a particular hidden dimension was selected.
- Why a component was removed.
- Why a particular implementation strategy was used.
- Why a simpler implementation was preferred over a more complex one.

 For each decision explain:

 **Decision → Motivation → Alternatives → Trade-off → Result**

---

 # 19\. Known Limitations

 Be honest and critical.

 Document:

 - Architectural limitations.
- Training limitations.
- Dataset limitations.
- Computational limitations.
- Reproducibility limitations.
- Potential bugs or fragile assumptions.
- Areas where the implementation diverges from the reference papers/repositories.
- Areas requiring additional validation.

 Do not hide weaknesses.

---

 # 20\. Potential Future Improvements

 Separate these into:

 ### High priority

 Changes likely to have meaningful impact.

 ### Medium priority

 Useful engineering or research improvements.

 ### Experimental

 Research ideas that require validation.

 Do not present speculative ideas as established improvements.

---

 # 21\. Reproducibility Guide

 Explain exactly how another engineer could reproduce the current system.

 Include:

 - Environment requirements.
- Dependencies.
- Dataset preparation.
- Configuration.
- Training command.
- Evaluation command.
- Inference command.
- Checkpoint handling.
- Random seeds.
- Hardware assumptions.

 Only provide commands that actually exist or can be verified from the repository.

---

 # 22\. Final Architecture Summary

 End with a concise but technically precise summary of the final system.

 Explain:

 1. Where the project started.
2. What MedMamba contributed.
3. What changed.
4. Why those changes were made.
5. How TRM was integrated.
6. What the final architecture looks like.
7. Which components are inherited, adapted, or original.
8. What the most important technical contributions are.
9. What remains uncertain or requires further experimentation.

---

 # Critical Requirements

 Follow these rules strictly:

 - **Inspect the actual code before documenting it.**
- Do not hallucinate files, classes, functions, algorithms, experiments, or results.
- Do not infer that something exists merely because it would be typical for this architecture.
- If something cannot be verified, explicitly state **"Not verified from the repository."**
- Distinguish facts from interpretation.
- Distinguish completed work from proposed future work.
- Distinguish source inspiration from direct code reuse.
- Do not falsely claim originality.
- Do not falsely claim direct copying.
- When comparing against MedMamba or TRM, explain the technical difference rather than merely saying "modified."
- Prefer concrete file/class/function references.
- Use actual tensor dimensions from the implementation when available.
- Use equations only when they accurately describe the implementation.
- Use Mermaid diagrams where they improve understanding.
- Use Markdown tables for comparisons and mappings.
- Avoid vague statements such as "optimized the model" without explaining exactly what was optimized and how.
- Preserve technical nuance.
- Be critical rather than promotional.

 ## Source References

 Include references to:

 - MedMamba: https://github.com/YubiaoYue/MedMamba
- Tiny Recursive Models: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- Any additional papers, repositories, or technical sources that are demonstrably relevant.

 Clearly separate **source references** from **project-specific implementation details**.

 ## Final Deliverable

 Return the complete documentation as a single Markdown document.

 The document should read like a combination of:

 - Technical architecture specification
- Research implementation report
- Engineering change log
- Reproducibility guide
- Attribution/provenance document

 Prioritize **accuracy, completeness, traceability, and technical depth** over brevity.