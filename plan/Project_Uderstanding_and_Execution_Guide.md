Act as a team of senior experts in:

* AI/ML research and engineering
* Deep learning and computer vision
* Python/software engineering
* Research software reproducibility
* Academic technical writing
* Academic presentation and documentation

Your task is to **thoroughly analyze the entire project before writing anything** and produce a clear, step-by-step guide explaining **what the project does, how it works, and exactly how to run it**.

The final document must be understandable to **any technically capable reader who has never seen this project before**. It must be sufficiently detailed that another researcher or developer can follow it without needing to guess missing steps.

## 1. IMPORTANT ANALYSIS RULES

Before writing the guide:

1. Inspect the complete project structure.
2. Inspect all relevant source-code files.
3. Inspect configuration files.
4. Inspect dependency/environment files.
5. Inspect scripts used for training, testing, evaluation, preprocessing, inference, or data preparation.
6. Inspect README files and existing documentation.
7. Inspect model definitions and architectural components.
8. Inspect dataset-loading and preprocessing logic.
9. Inspect configuration parameters and their defaults.
10. Inspect command-line arguments and execution entry points.
11. Trace the actual execution flow through the code.
12. Identify dependencies between components.
13. Identify what files are required to execute each workflow.
14. Identify expected input and output formats.
15. Identify hardware/software requirements.
16. Identify any assumptions made by the implementation.

**Do not infer behavior that is not supported by the project files.**

If something cannot be verified from the project:

* Clearly mark it as **NOT VERIFIED**.
* Explain what information is missing.
* Do not invent commands, paths, parameters, results, architecture details, or functionality.
* Distinguish clearly between:

  * **Verified from source code**
  * **Verified from configuration/documentation**
  * **Inferred**
  * **Unknown / Not Verified**

The goal is **zero hallucination and maximum reproducibility**.

---

# PART A — HOW THE PROJECT WORKS

Create a complete, step-by-step explanation of how the project works.

## A1. Project Overview

Explain:

* What the project is
* What problem it solves
* The project's primary objective
* The expected inputs
* The expected outputs
* The major technologies/frameworks used
* The overall workflow

Start with a simple explanation suitable for someone seeing the project for the first time.

Then provide a more technical explanation.

---

## A2. Project Architecture

Explain the complete architecture of the project.

Show the major components and their relationships.

Use a clear flow such as:

```text
Input
  ↓
Data Loading
  ↓
Preprocessing
  ↓
Dataset / DataLoader
  ↓
Model
  ↓
Training / Inference
  ↓
Evaluation
  ↓
Results / Outputs
```

Replace this example with the **actual architecture discovered in the project**.

For each component explain:

* What it does
* Which file implements it
* Which class/function implements it
* What it receives as input
* What it produces as output
* Which component calls it
* Which component it calls next

---

## A3. Directory and File Structure

Explain the project's directory structure.

For every important directory/file, provide:

| Path | Purpose | Important Components | Used By |
| ---- | ------- | -------------------- | ------- |

Do not document files that have no meaningful relevance unless necessary for understanding execution.

Clearly identify:

* Entry points
* Configuration files
* Model files
* Dataset files
* Training scripts
* Evaluation scripts
* Utility modules
* Output/result directories
* Checkpoints
* Logs
* Documentation

---

## A4. End-to-End Execution Flow

Trace the actual execution flow from beginning to end.

Explain exactly what happens when the main workflow is executed.

For example:

```text
Command
  ↓
Entry-point script
  ↓
Argument parsing
  ↓
Configuration loading
  ↓
Dataset initialization
  ↓
Data preprocessing
  ↓
Model initialization
  ↓
Optimizer / scheduler initialization
  ↓
Training loop
  ↓
Validation
  ↓
Checkpointing
  ↓
Testing
  ↓
Metrics
  ↓
Results
```

Use the project's actual implementation rather than assuming this structure.

For every stage identify the relevant file, class, or function.

---

# PART B — DATA PIPELINE

Explain the complete data pipeline.

## B1. Required Dataset

Explain:

* What dataset is expected
* Where it must be located
* Expected directory structure
* Required files
* Expected file formats
* Labels/classes
* Metadata requirements
* Train/validation/test organization

If the project supports multiple datasets, document each one separately.

---

## B2. Data Loading

Explain:

1. Which file loads the data.
2. Which class/function performs loading.
3. What format the raw data has.
4. What transformations are applied.
5. What tensor/array shape is produced.
6. What the dimensions represent.

For example:

```text
[B, C, H, W]

B = batch size
C = channels / spectral bands
H = height
W = width
```

Only provide dimensions that are verified from the code.

---

## B3. Preprocessing

Document every preprocessing step in execution order.

For each step explain:

* Purpose
* Implementation file/function
* Input
* Output
* Parameters
* Whether it occurs during training, validation, testing, or all three

Include normalization, resizing, cropping, augmentation, band selection, channel conversion, patch extraction, masking, etc., when actually present.

---

# PART C — MODEL / ALGORITHM

Provide a complete explanation of the model or algorithm.

## C1. Model Overview

Explain:

* Model name
* Purpose
* Input format
* Output format
* Number of classes/outputs
* Major architectural components

---

## C2. Layer-by-Layer Architecture

Document the architecture in execution order.

Use a table:

| # | Component | Input Shape | Operation | Output Shape | Purpose |
| - | --------- | ----------- | --------- | ------------ | ------- |

For every important layer/module explain what it actually does.

Do not describe a component based solely on its name. Verify its implementation.

---

## C3. Important Algorithms

Explain important algorithms or mechanisms implemented by the project.

Examples may include:

* Mamba/state-space modules
* Attention
* Convolution
* Feature extraction
* Band selection
* Recursive processing
* Reconstruction
* Classification
* Loss functions
* Regularization
* Sampling
* Augmentation

Only include mechanisms actually present in the project.

---

# PART D — TRAINING WORKFLOW

Explain exactly how training works.

Include:

1. Dataset initialization
2. DataLoader creation
3. Model initialization
4. Device selection
5. Loss function
6. Optimizer
7. Learning-rate scheduler
8. Mixed precision
9. Forward pass
10. Loss calculation
11. Backpropagation
12. Gradient handling
13. Parameter update
14. Validation
15. Checkpointing
16. Logging
17. Early stopping, if implemented
18. Final model saving

Explain the training loop step by step.

---

# PART E — EVALUATION

Explain how the project evaluates the model.

Document:

* Evaluation script
* Dataset used
* Metrics
* Metric formulas when useful
* Prediction generation
* Checkpoint selection
* Output files
* Result interpretation

Clearly distinguish between metrics that are:

* Actually calculated by the code
* Reported in documentation
* Mentioned but not implemented

---

# PART F — HOW TO INSTALL AND RUN THE PROJECT

This section must be **fully reproducible**.

## F1. System Requirements

Document verified requirements:

* Operating system
* Python version
* CUDA version
* GPU requirements
* CPU requirements
* RAM requirements
* Storage requirements
* Required external software

Do not invent minimum hardware requirements.

---

## F2. Environment Setup

Provide exact commands for creating and activating the environment.

Example:

```bash
python --version
```

Then provide the project's actual installation commands.

Explain what each command does.

---

## F3. Dependencies

List:

* Python packages
* System dependencies
* CUDA dependencies
* Model-specific dependencies
* Optional dependencies

Identify dependency versions whenever they are specified by the project.

---

## F4. Dataset Setup

Provide exact instructions for:

1. Obtaining the dataset
2. Creating required directories
3. Placing files
4. Verifying the dataset
5. Running any preprocessing/conversion scripts

Show the expected final directory structure.

---

## F5. Configuration

Explain every important configuration parameter required to run the project.

Use:

| Parameter | Location | Default | Meaning | Recommended/Required Value |
| --------- | -------- | ------: | ------- | -------------------------- |

Do not invent recommended values.

Clearly distinguish:

* Required parameters
* Optional parameters
* Defaults
* Parameters that should normally remain unchanged

---

# PART G — EXACT COMMANDS

Provide complete commands for every major workflow.

At minimum, document:

### 1. Environment setup

```bash
...
```

### 2. Dataset preparation

```bash
...
```

### 3. Training

```bash
...
```

### 4. Validation

```bash
...
```

### 5. Testing

```bash
...
```

### 6. Inference

```bash
...
```

### 7. Evaluation

```bash
...
```

### 8. Any additional project-specific workflow

```bash
...
```

For every command explain:

* Where it should be executed
* Required working directory
* Required environment
* Required arguments
* Expected output
* Where results will be saved

**Never provide placeholder commands as if they are executable.**

If an exact command cannot be verified, label it:

> NOT VERIFIED — DO NOT EXECUTE WITHOUT CONFIRMATION

---

# PART H — EXPECTED OUTPUTS

Explain what the user should see after running each major command.

Include:

* Terminal output
* Logs
* Checkpoints
* Metrics
* Generated files
* Result directories
* Reports
* Images/plots
* Predictions

Provide example output only when it is directly supported by the project or clearly label it as an example.

---

# PART I — TROUBLESHOOTING

Create a troubleshooting section based on issues that can be identified from the project.

For each problem provide:

| Problem | Likely Cause | Verification | Solution |
| ------- | ------------ | ------------ | -------- |

Include common issues involving:

* Missing dependencies
* Incorrect Python version
* CUDA/GPU incompatibility
* Missing datasets
* Incorrect paths
* Incorrect tensor dimensions
* Missing checkpoints
* Import errors
* Configuration errors
* Memory errors
* Permission errors

Do not invent project-specific errors that have no evidence.

---

# PART J — QUICK START

Create a short **Quick Start** section for someone who simply wants to run the project.

It must contain only the essential steps:

```text
1. Clone/open project
2. Create environment
3. Install dependencies
4. Prepare dataset
5. Configure project
6. Run training
7. Run evaluation
8. Find results
```

Include the exact verified commands.

---

# PART K — RESEARCH REPRODUCIBILITY

Because this project may be used for academic research, document everything required to reproduce an experiment.

Include:

* Dataset version
* Data split
* Random seed
* Model configuration
* Hyperparameters
* Batch size
* Learning rate
* Number of epochs
* Optimizer
* Scheduler
* Hardware
* Software environment
* Checkpoint used
* Evaluation procedure
* Metrics
* Output files

Clearly identify which reproducibility information is available and which is missing.

---

# PART L — ACADEMIC / PRESENTATION EXPLANATION

Provide a concise explanation that could be used when presenting the project to:

* A professor
* Research supervisor
* Research team
* Conference audience
* Technical reviewer

Answer:

1. What is the problem?
2. What does the project do?
3. How does it work?
4. What is the model/algorithm?
5. What data does it use?
6. What happens during training?
7. How is performance evaluated?
8. What are the outputs?
9. What are the important technical contributions?
10. What limitations or unverified aspects remain?

Do not claim something is a research contribution unless the project evidence supports that claim.

---

# FINAL SECTION — VERIFIED FACTS AND UNKNOWN INFORMATION

End the document with two explicit sections.

## Verified Facts

List important facts confirmed directly from the project.

## Unknown / Not Verified

List anything that could not be established from the available project files.

This section is mandatory.

---

# WRITING REQUIREMENTS

The final document must be:

* Extremely clear
* Step-by-step
* Technically accurate
* Reproducible
* Structured logically
* Easy to navigate
* Suitable for both technical and non-expert readers
* Free of unnecessary jargon
* Free of unsupported claims

When technical terminology is necessary, explain it the first time it appears.

Use:

* Headings
* Numbered steps
* Tables
* Code blocks
* File paths
* Command examples
* Architecture diagrams using Markdown
* Input/output descriptions
* Clear warnings where necessary

Do not bury critical instructions inside long paragraphs.

---

# ZERO-HALLUCINATION POLICY

This is mandatory.

**Never fabricate:**

* Commands
* File names
* Paths
* Functions
* Classes
* Parameters
* Hyperparameters
* Dataset structure
* Model architecture
* Results
* Metrics
* Hardware requirements
* Dependencies
* Research contributions

If something cannot be verified, explicitly state:

> **NOT VERIFIED**

and explain what would be required to verify it.

The final guide must reflect **what the project actually does**, not what you believe the project was intended to do.

## FINAL QUALITY CHECK

Before producing the final document, verify that:

* Every major workflow has been traced through the code.
* Every execution command is supported by the project.
* Important files are identified.
* Dataset requirements are documented.
* Model architecture is documented.
* Training flow is documented.
* Evaluation flow is documented.
* Outputs are documented.
* Dependencies are documented.
* Configuration parameters are documented.
* Unknown information is explicitly identified.
* No unsupported claims have been introduced.
* A new researcher could follow the guide without guessing what to do next.

The final result should function as a **complete Project User Guide + Technical Architecture Guide + Reproducibility Guide**.
