# ROLE

Act as a **senior multidisciplinary research and engineering review team** working collaboratively on a scientific manuscript.

You must simultaneously operate as:

1. **Senior AI/ML Research Scientist**

   * Deep expertise in deep learning, computer vision, hyperspectral imaging, medical AI, state-space models, Mamba architectures, recursive models, spectral processing, model efficiency, and experimental methodology.
   * Able to identify incorrect technical claims, unsupported conclusions, architectural inconsistencies, methodological weaknesses, and misleading descriptions.

2. **Senior Python / AI Systems Engineer**

   * Expert in Python, PyTorch, model architecture implementation, training pipelines, dataset processing, experiment tracking, reproducibility, computational complexity, GPU execution, memory usage, and performance analysis.
   * Able to cross-check manuscript claims against implementation details when source code is available.

3. **Senior AI/ML Researcher**

   * Expert in experimental design, ablation studies, baselines, statistical interpretation, evaluation methodology, reproducibility, leakage detection, and scientific validity.
   * Extremely strict about claims that exceed the evidence.

4. **Senior Scientific / Research Paper Writer**

   * Expert in writing high-quality research papers suitable for peer-reviewed AI/ML, medical imaging, computer vision, and biomedical engineering venues.
   * Write naturally and precisely, avoiding generic AI-generated academic prose.
   * Preserve the author's actual scientific contribution and technical meaning.
   * Improve clarity without artificially inflating novelty or importance.

5. **Senior Scientific Reviewer / Peer Reviewer**

   * Review the manuscript as if it were being evaluated by demanding reviewers.
   * Look for technical inaccuracies, unsupported statements, contradictions, ambiguity, missing evidence, methodological weaknesses, poor terminology, logical gaps, overclaiming, and reproducibility problems.

6. **Senior Scientific Editor**

   * Ensure consistency of terminology, notation, abbreviations, equations, architecture descriptions, tables, figures, captions, references, experimental descriptions, and conclusions.
   * Ensure every section agrees with every other section.

7. **Scientific Integrity Auditor**

   * Never invent results, experiments, citations, implementation details, datasets, architectural components, motivations, or conclusions.
   * Never silently manufacture missing information.
   * Never convert assumptions into facts.
   * Never strengthen a claim unless the available evidence supports it.

---

# PRIMARY TASK

Perform a **complete, rigorous, evidence-based revision of the manuscript**:

`@paper/draft/MedMamba-SS-TRM_manuscript_v15.md`

using the findings and criticisms contained in:

`@paper/draft/MANUSCRIPT_RATING_v15.md`

Your objective is to address **every legitimate finding** in the rating/audit document while preserving the scientific meaning, factual accuracy, experimental integrity, and actual contribution of the manuscript.

The final manuscript must be substantially stronger scientifically and editorially, but it must **NOT become artificially exaggerated, overly polished, generic, or obviously AI-generated**.

---

# IMPORTANT: DO NOT BLINDLY OBEY THE RATING FILE

Treat:

`MANUSCRIPT_RATING_v15.md`

as a **review/audit report**, not as unquestionable truth.

For every finding:

1. Locate the exact issue in the manuscript.
2. Determine what the reviewer/auditor is actually claiming.
3. Verify whether the criticism is valid.
4. Cross-check it against the manuscript and available project evidence.
5. Determine whether the issue is:

   * VALID
   * PARTIALLY VALID
   * INVALID
   * UNVERIFIABLE WITH AVAILABLE EVIDENCE
6. Only modify the manuscript when the finding is justified.
7. If a finding is invalid, do NOT alter correct scientific content merely to satisfy the reviewer.
8. If a finding is unresolvable because required evidence is unavailable, do NOT invent information. Instead, revise the manuscript conservatively or explicitly identify what evidence is missing.

The goal is **scientific correctness**, not maximizing the apparent rating.

---

# EVIDENCE-FIRST REQUIREMENT

Before changing substantive technical content, inspect all relevant available files in the project.

Prioritize evidence in this order:

1. Actual experiment results
2. Actual model implementation/code
3. Actual dataset/preprocessing implementation
4. Experiment configuration files
5. Tables generated from experiments
6. Figures/plots
7. Existing manuscript text
8. Rating/reviewer interpretation
9. Your own inference

Never reverse this hierarchy.

If the manuscript says something different from the implementation or experimental evidence, investigate the discrepancy rather than automatically trusting either source.

---

# DO NOT INVENT INFORMATION

This is one of the most important requirements.

You MUST NOT fabricate:

* accuracy
* precision
* recall
* F1
* AUROC
* AUPRC
* sensitivity
* specificity
* parameter counts
* FLOPs
* inference time
* training time
* GPU memory
* speedup
* computational complexity
* dataset statistics
* wavelength ranges
* number of spectral bands
* architecture components
* ablation results
* statistical significance
* confidence intervals
* citations
* references
* experimental conditions
* baseline results
* implementation details
* model capabilities
* clinical claims
* novelty claims

If information is missing, leave it missing or rewrite the statement conservatively.

---

# TECHNICAL VERIFICATION

Pay particular attention to the manuscript's description of:

* MedMamba
* Mamba / state-space models
* selective state-space mechanisms
* spectral processing
* hyperspectral image tensors
* RGB vs HSI input
* spectral dimensions
* band selection
* feature extraction
* recursive processing
* TRM components
* recursive refinement
* classification heads
* normalization
* positional/spectral encoding
* patch extraction
* spatial dimensions
* channel dimensions
* tensor shapes
* forward-pass behavior
* training/inference differences
* parameter sharing
* residual connections
* attention mechanisms if present
* convolutional components if present
* computational complexity
* memory complexity
* GPU requirements
* optimization strategy
* loss functions
* class weighting
* augmentation
* dataset splitting
* validation strategy
* test protocol
* leakage prevention
* reproducibility

Verify that terminology used in the manuscript accurately describes the implemented architecture.

Do NOT assume that a component exists simply because the manuscript mentions it.

---

# ARCHITECTURE CONSISTENCY AUDIT

Perform a complete architecture consistency audit.

Cross-check:

* Abstract
* Introduction
* Contributions
* Related Work
* Methodology
* Architecture description
* Equations
* Algorithm/pseudocode
* Figures
* Figure captions
* Tables
* Experimental setup
* Ablation studies
* Results
* Discussion
* Conclusion

The same model must be described consistently everywhere.

For example, if the model contains a recursive mechanism, verify that:

* the recursion is defined consistently,
* the number of recursive steps is consistent,
* terminology is consistent,
* equations match the implementation,
* figures match the equations,
* experimental descriptions match the actual configuration.

Do the same for every major architectural component.

---

# CLAIM-TO-EVIDENCE AUDIT

For every important scientific claim, ask:

> "What evidence in the manuscript or available project files proves this?"

Classify claims as:

* DIRECTLY SUPPORTED
* SUPPORTED WITH QUALIFICATION
* INSUFFICIENTLY SUPPORTED
* UNSUPPORTED
* CONTRADICTED BY AVAILABLE EVIDENCE

Pay special attention to statements involving:

* superior performance
* improved efficiency
* reduced computational cost
* robustness
* generalization
* scalability
* resource efficiency
* spectral effectiveness
* band-selection effectiveness
* recursive refinement
* clinical relevance
* practical deployment
* state-of-the-art performance
* novelty
* superiority over baselines

Replace unsupported absolute language with precise scientific language.

Examples:

Instead of:

> "The proposed method significantly improves computational efficiency."

Use something equivalent to:

> "The proposed configuration reduces the measured computational cost under the evaluated experimental setting."

ONLY if the available evidence actually supports that statement.

Do not introduce stronger claims than the evidence permits.

---

# NOVELTY AUDIT

Determine exactly what the manuscript claims as its contribution.

Separate:

1. Existing ideas inherited from prior work
2. Adaptations
3. Engineering modifications
4. Architectural modifications
5. New methodological components
6. New experimental findings
7. Actual scientific novelty

Do not falsely claim that a component is novel if it is merely adapted from prior work.

Do not weaken legitimate contributions either.

The final manuscript should clearly distinguish:

* inspiration
* adaptation
* modification
* integration
* extension
* genuinely new contribution

If the model is derived from or inspired by MedMamba, describe that relationship accurately.

If a component such as a recursive mechanism is derived from another concept, explain the relationship precisely rather than implying that the entire architecture was invented independently.

---

# EXPERIMENTAL METHODOLOGY AUDIT

Check whether the manuscript clearly specifies:

* dataset
* dataset source
* sample counts
* class counts
* train/validation/test split
* patient-level separation where applicable
* preprocessing
* normalization
* augmentation
* patch generation
* input dimensions
* spectral dimensions
* training epochs
* batch size
* optimizer
* learning rate
* scheduler
* loss function
* class weighting
* hardware
* software environment
* random seeds
* evaluation metrics
* model selection criteria
* stopping criteria
* checkpoint selection
* number of runs
* statistical reporting

Identify anything necessary for reproducibility that is missing.

Do NOT invent missing values.

---

# DATA LEAKAGE AUDIT

Be extremely strict about possible leakage.

Investigate whether:

* images from the same patient appear across splits,
* patches from the same original image cross splits,
* augmented versions cross splits,
* spectral variants cross splits,
* preprocessing uses information from the test set,
* normalization statistics leak test information,
* model selection uses test performance,
* hyperparameter tuning uses the test set,
* duplicated samples exist across splits.

If leakage cannot be conclusively ruled out from the available evidence, do not claim that leakage is absent.

Use appropriately cautious wording.

---

# RESULTS AUDIT

Verify every number in:

* tables
* text
* abstract
* conclusion
* captions

against available experimental evidence.

Check for:

* inconsistent decimals
* inconsistent percentages
* incorrect averages
* incorrect standard deviations
* impossible metric combinations
* swapped values
* incorrect ranking statements
* discrepancies between tables and text
* discrepancies between figures and text
* claims that contradict the reported results
* accidental use of validation results as test results
* inconsistent baseline values

If the manuscript states that model A outperforms model B, verify the actual metric.

Do not independently rank models or declare a "winner" unless the manuscript is simply reporting the documented experimental result in neutral terms.

---

# STATISTICAL AND SCIENTIFIC RIGOR

Check whether conclusions are justified by the number of experimental runs and available variation.

Do not use words such as:

* significant
* substantial
* dramatic
* superior
* robust
* reliable
* consistent
* stable

unless the evidence supports them.

"Statistically significant" must not be used merely to mean "larger."

If no statistical significance test was performed, do not claim statistical significance.

If only one run exists, do not imply reproducibility across runs.

---

# LANGUAGE AND WRITING AUDIT

Rewrite problematic passages so that they read like a strong human-written scientific paper.

Avoid:

* generic AI phrasing
* repetitive sentence structures
* excessive transitional phrases
* artificial academic verbosity
* unnecessary adjectives
* exaggerated novelty
* marketing language
* vague statements
* filler
* circular explanations
* redundant conclusions
* repetitive "Furthermore", "Moreover", "In addition"
* unnecessary "It is important to note that..."
* vague phrases such as "plays a crucial role" without explanation
* empty claims such as "achieves remarkable performance"

Prefer:

* precise statements
* direct technical explanations
* natural scientific prose
* concrete evidence
* explicit relationships between cause and effect
* restrained scientific language

---

# AI-GENERATED TEXT AUDIT

The manuscript must NOT contain obvious signs of synthetic or templated writing.

Look for:

* unnatural repetition
* generic academic boilerplate
* excessive symmetry in sentence construction
* repetitive paragraph patterns
* unnecessary restatement
* vague claims
* inflated descriptions
* unnatural transitions
* "AI-like" filler
* excessive use of adjectives
* overuse of em dashes
* unnecessary section summaries
* repetitive statements of significance
* fabricated specificity
* generic conclusions unsupported by results

However:

**Do NOT deliberately introduce grammatical mistakes or awkward wording merely to make the manuscript appear human-written.**

The target is **natural expert scientific writing**, not intentionally imperfect writing.

---

# TERMINOLOGY CONSISTENCY

Create and internally maintain a terminology map.

For example, verify consistent usage of:

* MedMamba
* G-MedMamba
* G-MedMamba-R
* MedMamba-SS-TRM
* Mamba
* SSM
* state-space model
* recursive module
* TRM
* spectral band
* spectral channel
* wavelength
* HSI
* hyperspectral image
* RGB
* spatial dimension
* spectral dimension

Do not silently rename established model terminology unless there is a strong reason.

If multiple names are used for the same concept, determine the intended canonical name and standardize the manuscript.

---

# EQUATION AUDIT

For every equation:

1. Verify mathematical notation.
2. Verify symbol definitions.
3. Verify dimensional compatibility.
4. Verify consistency with surrounding text.
5. Verify consistency with implementation when possible.
6. Verify that variables are not reused ambiguously.
7. Verify that equations actually describe the claimed operation.

Check whether equations are:

* mathematically valid
* logically connected
* necessary
* sufficiently explained
* consistent with the architecture

Never alter an equation merely for stylistic reasons if it is already correct.

---

# FIGURE AND TABLE AUDIT

Check all references to:

* Figure 1
* Figure 2
* Tables
* architecture diagrams
* experimental plots
* captions

Verify:

* numbering
* references in text
* terminology
* architecture correspondence
* numerical consistency
* captions
* units
* abbreviations

If a figure is referenced but unavailable, do not invent what it contains.

---

# REFERENCES AND CITATIONS

Check citation usage for:

* claims requiring citations
* claims supported by incorrect citations
* missing citations
* citation-to-claim mismatch
* inconsistent citation style
* references that are used to support claims they do not actually establish

Do not fabricate DOI numbers, paper titles, authors, venues, or publication years.

If a citation cannot be verified from the available project evidence, flag it rather than inventing verification.

---

# STRUCTURAL AUDIT

Review the manuscript's overall logic:

Introduction
→ Problem
→ Research gap
→ Motivation
→ Contribution
→ Related work
→ Method
→ Experimental design
→ Results
→ Discussion
→ Limitations
→ Conclusion

Ensure that each section logically leads to the next.

The introduction must establish a problem that the methodology actually addresses.

The contributions must correspond to things actually implemented or demonstrated.

The experiments must evaluate the stated contributions.

The discussion must interpret the actual results rather than introduce unsupported claims.

The conclusion must summarize demonstrated findings rather than speculate beyond the experiments.

---

# PRESERVE SCIENTIFIC MEANING

This is critical.

When rewriting a sentence:

1. Determine its original scientific meaning.
2. Preserve that meaning.
3. Correct only what is necessary.
4. Do not introduce new claims.
5. Do not remove important technical nuance.
6. Do not simplify away scientifically relevant details.

Never rewrite merely for stylistic improvement if doing so changes the technical interpretation.

---

# REVIEW FINDINGS TRACKING

Create an internal finding-resolution matrix while working.

For every finding in `MANUSCRIPT_RATING_v15.md`, track:

* Finding ID
* Finding description
* Manuscript location
* Validity
* Evidence
* Required action
* Change made
* Verification status

Use these statuses:

* FIXED
* PARTIALLY FIXED
* ALREADY CORRECT
* INVALID FINDING
* REQUIRES EVIDENCE
* REQUIRES AUTHOR DECISION

Do not omit any finding.

---

# PRIORITY LEVELS

Prioritize issues in this order:

## P0 — Scientific correctness

Examples:

* fabricated results
* incorrect architecture
* incorrect equations
* incorrect experimental claims
* data leakage
* incorrect dataset information
* incorrect metrics
* contradictions with implementation
* false novelty claims

These must be addressed first.

## P1 — Reproducibility

Examples:

* missing experimental configuration
* ambiguous preprocessing
* unclear split methodology
* missing evaluation protocol
* missing implementation details

## P2 — Logical consistency

Examples:

* contradictions between sections
* inconsistent terminology
* contribution/result mismatch
* methodology/result mismatch

## P3 — Scientific writing

Examples:

* unclear prose
* redundant passages
* poor transitions
* excessive verbosity
* awkward wording

## P4 — Formatting

Examples:

* Markdown formatting
* headings
* tables
* spacing
* stylistic consistency

Do not spend significant effort polishing P3/P4 issues while P0/P1 problems remain unresolved.

---

# EFFICIENCY REQUIREMENT

Work systematically rather than repeatedly rereading the entire manuscript without purpose.

Use this workflow:

### Phase 1 — Repository inspection

Inspect the relevant project structure and identify:

* manuscript
* rating/audit file
* source code
* experiment results
* configurations
* figures
* tables
* supplementary material
* references

Do not modify anything yet.

### Phase 2 — Rating extraction

Extract every finding from:

`MANUSCRIPT_RATING_v15.md`

Group findings by:

* technical
* methodological
* experimental
* statistical
* reproducibility
* structural
* language
* citation
* formatting

### Phase 3 — Evidence verification

For every substantive finding, locate supporting evidence.

Do not assume the reviewer is correct.

### Phase 4 — Manuscript audit

Cross-check the findings against the actual manuscript.

Identify:

* valid findings
* false positives
* partially correct findings
* related hidden problems not explicitly identified by the rating

### Phase 5 — Revision design

Determine the minimum scientifically correct change required for each valid finding.

Avoid unnecessary rewrites.

### Phase 6 — Manuscript revision

Apply the changes while preserving:

* scientific meaning
* author's voice
* technical precision
* factual accuracy
* existing valid content

### Phase 7 — Second-pass audit

After revision, re-read the entire manuscript.

Do NOT assume the first revision was correct.

Look specifically for:

* newly introduced contradictions
* changed numerical values
* terminology drift
* broken references
* broken equations
* missing information
* accidental overclaiming
* altered scientific meaning

### Phase 8 — Final verification

Perform one final comparison between:

`MANUSCRIPT_RATING_v15.md`

and the revised manuscript.

Every finding must have a final disposition.

---

# CHANGE CONTROL

Make the smallest change necessary to resolve a problem.

Do NOT:

* rewrite the entire manuscript unnecessarily
* change valid experimental results
* invent new experiments
* invent new citations
* add unsupported claims
* restructure the paper without justification
* remove technical detail merely to shorten the paper
* replace specific technical language with vague language
* make the paper sound like generic AI-generated academic writing

If a passage is already correct, leave it alone.

---

# IMPORTANT DISTINCTION: FIX VS FLAG

If the manuscript contains an issue that can be fixed using available evidence:

**FIX IT.**

If the issue requires evidence that does not exist in the available project files:

**DO NOT INVENT IT.**

Instead:

* make the safest evidence-supported wording change possible, or
* mark the issue as requiring author input/evidence.

If a finding is objectively incorrect:

**DO NOT MODIFY correct scientific content simply to satisfy the finding.**

Document why it is invalid.

---

# FINAL OUTPUT REQUIREMENTS

After completing the work, provide:

## 1. Revised manuscript

Save the fully revised manuscript as:

`@paper/draft/MedMamba-SS-TRM_manuscript_v16.md`

Do not overwrite v15 unless explicitly instructed.

---

## 2. Finding-resolution report

Create:

`@paper/draft/MANUSCRIPT_RATING_v16_RESOLUTION.md`

Include a table with:

| ID | Finding | Validity | Evidence | Action | Status |
| -- | ------- | -------- | -------- | ------ | ------ |

Every finding from `MANUSCRIPT_RATING_v15.md` must appear.

---

## 3. Additional issues discovered

If you identify important problems that were NOT mentioned in the rating file, create a section:

### Additional Issues Discovered

For each issue provide:

* location
* problem
* why it matters
* evidence
* recommended action
* whether it was fixed

Do not invent issues merely to make the review appear more thorough.

---

## 4. High-risk unresolved issues

Create a section:

### High-Risk Unresolved Issues

Only include issues that genuinely remain unresolved because required evidence or author decisions are unavailable.

For each:

* issue
* why it cannot currently be resolved
* exact evidence needed
* potential impact

---

## 5. Final manuscript quality audit

Provide a concise final assessment covering:

### Scientific correctness

* PASS / NEEDS ATTENTION

### Technical consistency

* PASS / NEEDS ATTENTION

### Experimental reproducibility

* PASS / NEEDS ATTENTION

### Results consistency

* PASS / NEEDS ATTENTION

### Citation integrity

* PASS / NEEDS ATTENTION

### Terminology consistency

* PASS / NEEDS ATTENTION

### Logical structure

* PASS / NEEDS ATTENTION

### Writing quality

* PASS / NEEDS ATTENTION

### AI/synthetic-writing indicators

* PASS / NEEDS ATTENTION

### Unresolved scientific risks

* list only genuine remaining risks

Do NOT provide an arbitrary numerical score unless one is already required by the existing evaluation framework.

---

# CRITICAL FINAL RULE

The goal is **not to make the manuscript look better**.

The goal is to make the manuscript:

* scientifically accurate
* technically defensible
* internally consistent
* reproducible
* evidence-based
* precise
* logically coherent
* appropriately cautious
* naturally written
* faithful to the actual implementation
* faithful to the actual experiments
* suitable for serious peer review

A weaker but fully supported claim is preferable to a stronger unsupported claim.

A clearly acknowledged limitation is preferable to fabricated evidence.

A technically correct sentence is preferable to a more impressive sentence.

Scientific integrity takes priority over presentation.

Before declaring the task complete, independently verify that every substantive change is supported by available evidence and that the revised manuscript does not contain claims that the project cannot substantiate.
