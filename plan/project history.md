# MASTER PROMPT — Comprehensive Project History Reconstruction, Technical Audit, and Engineering Evolution Documentation

## 1. ROLE AND EXPERTISE

Act as a multidisciplinary team of highly experienced senior professionals working collaboratively, combining the expertise of:

* Senior AI/ML Research Engineer
* Senior Software Architect
* Senior Python Engineer
* Senior Machine Learning Systems Engineer
* Senior Codebase Auditor
* Senior Technical Documentation Engineer
* Senior Research and Development Analyst

Your collective responsibility is to perform an exhaustive investigation of the entire project and reconstruct its technical and developmental history with maximum accuracy, completeness, and traceability.

Approach this task as a formal engineering investigation, not as a superficial code review or generic project summary.

You must understand not only what the project currently contains, but also:

* How it evolved.
* What was changed.
* Why changes were introduced.
* What problems each change was intended to solve.
* How the implementation changed technically and logically.
* What improvements were introduced.
* What architectural or methodological decisions were made.
* How different components, modules, and subsystems evolved.
* What limitations, regressions, unresolved issues, or technical debt remain.

The final document must function as a comprehensive historical and technical record that another senior engineer could use to understand the project's evolution without needing to reconstruct everything from scratch.

---

## 2. PRIMARY OBJECTIVE

Conduct a comprehensive investigation of the complete project and generate a detailed Markdown document named:

`PROJECT_HISTORY_AND_TECHNICAL_EVOLUTION.md`

The document must provide a chronological and technically meaningful record of the project's development, including:

1. Project origins and initial structure, where evidence is available.
2. Historical development stages.
3. Major and minor modifications.
4. Architectural changes.
5. Logical and algorithmic modifications.
6. Code-level implementation changes.
7. Performance and resource-efficiency improvements.
8. Bug fixes and corrective modifications.
9. Changes to data processing, training, evaluation, and inference pipelines, where applicable.
10. Configuration and dependency changes.
11. Refactoring and code organization.
12. Experimental findings that influenced development.
13. Failed approaches and reverted changes, when verifiable.
14. Current implementation and its relationship to earlier versions.
15. Outstanding problems, limitations, and recommended future work.

Do not merely enumerate file modifications.

Explain the engineering significance of each meaningful change.

---

## 3. STRICT INVESTIGATION REQUIREMENTS

### 3.1 Complete Project Inspection

Inspect the entire accessible project before producing conclusions.

Investigate, where present:

* Source code.
* Python modules and packages.
* Model architectures.
* Training and evaluation scripts.
* Data preprocessing and dataset loaders.
* Configuration files.
* Dependency and environment definitions.
* Documentation.
* README files.
* Experiment logs.
* Metrics and result reports.
* Checkpoints and model artifacts.
* Test suites.
* Git history, branches, tags, and commit messages.
* Existing development notes and historical documentation.
* Scripts responsible for automation, benchmarking, and reporting.

Do not restrict the investigation to the main entry point or README.

Identify relationships between modules and trace important implementation paths across the codebase.

### 3.2 Historical Reconstruction

Use Git history as a primary source of chronological evidence whenever available.

Investigate:

* Commit history.
* Commit dates.
* Commit messages.
* File additions and deletions.
* Code diffs.
* Renamed or reorganized modules.
* Changes in function signatures.
* Changes in class definitions.
* Changes in model layers and data flow.
* Changes in configuration defaults.
* Changes in dependencies.
* Changes in experimental methodology.

Where Git history is unavailable or incomplete, reconstruct only what can be supported by remaining evidence.

**Never fabricate historical events, dates, motivations, implementation details, or previous versions.**

Distinguish clearly between:

* Verified historical changes.
* Changes inferred from available evidence.
* Current-state observations.
* Unknown or unrecoverable history.

### 3.3 Technical and Logical Analysis

For every significant modification, investigate both its implementation and its purpose.

Explain:

**WHAT changed**

* Previous implementation, if recoverable.
* New implementation.
* Affected files, classes, functions, or components.

**WHY it changed**

* Original problem or limitation.
* Engineering motivation.
* Research or experimental motivation, if documented.
* Evidence supporting the reason.

**HOW it changed**

* Implementation strategy.
* Algorithmic or architectural differences.
* Changes in data flow.
* Changes in computational behavior.
* Dependencies on other modules.

**WHAT IMPROVED**

* Correctness.
* Maintainability.
* Computational efficiency.
* Memory consumption.
* Training stability.
* Model capability.
* Modularity.
* Reproducibility.
* Observability.
* Reliability.

**WHAT THE CONSEQUENCES WERE**

* New capabilities.
* Removed limitations.
* Trade-offs.
* Potential regressions.
* Compatibility implications.
* Remaining limitations.

Do not claim an improvement merely because code was modified. Require supporting evidence or clearly label it as an intended improvement rather than a demonstrated result.

---

## 4. ZERO-HALLUCINATION AND EVIDENCE POLICY

This is a strict requirement.

1. Never invent facts, code behavior, historical events, experimental results, motivations, or architectural details.
2. Never assume that a change was successful simply because it was implemented.
3. Never confuse intended functionality with verified functionality.
4. Never claim performance improvements without benchmark evidence.
5. Never claim accuracy improvements without actual experimental results.
6. Never assume that a deleted file represents a failed approach.
7. Never infer developer intentions from code alone as established fact.
8. Never fabricate commit hashes, dates, file paths, line numbers, or version identifiers.
9. If information cannot be established, explicitly state `Not verifiable from available project evidence`.
10. Preserve contradictory evidence rather than silently resolving it through assumptions.

Every significant historical claim must be traceable to at least one available source, such as:

* Git commit or diff.
* File path and relevant code.
* Experiment report.
* Configuration change.
* Test result.
* Existing documentation.

Use exact commit hashes and dates when available.

For claims about current code, cite the relevant file path and, where practical, class, function, or line range.

Maintain a clear distinction between historical evidence and technical interpretation.

---

## 5. REQUIRED DOCUMENT STRUCTURE

Generate the Markdown document using the following structure.

# Project History and Technical Evolution

## 1. Executive Summary

* Project purpose.
* Current technical state.
* Main development milestones.
* Most significant engineering changes.
* Overall evolution.

## 2. Project Overview

* Original objectives, if recoverable.
* Current objectives.
* Core technologies.
* Main subsystems.
* High-level architecture.

## 3. Initial Project State

* Initial structure.
* Initial implementation.
* Original architecture.
* Initial limitations.
* Known baseline behavior.

Clearly identify unavailable historical information.

## 4. Chronological Development Timeline

Organize changes chronologically.

For every identifiable development milestone, include:

### Milestone: [Name or Date]

* **Date / Commit:**
* **Affected components:**
* **Previous state:**
* **Modification:**
* **Technical explanation:**
* **Logical motivation:**
* **Problem addressed:**
* **Resulting behavior:**
* **Demonstrated improvements:**
* **Trade-offs or regressions:**
* **Evidence:**

Group related commits into meaningful engineering milestones when appropriate, but preserve their individual traceability.

## 5. Architectural Evolution

Document how the architecture changed over time.

Include:

* Original architecture.
* Intermediate architectural stages.
* Current architecture.
* Module responsibilities.
* Component interactions.
* Data flow.
* Dependency relationships.
* Architectural decisions.
* Reasons for restructuring, when documented.

Use Mermaid diagrams where they materially clarify the evolution.

## 6. Detailed Component-by-Component History

For every significant subsystem, document:

* Original purpose.
* Historical implementation.
* Modifications.
* Current implementation.
* Technical rationale.
* Dependencies.
* Improvements.
* Remaining limitations.

Do not omit important components simply because they are secondary to the main application.

## 7. Algorithmic and Logical Evolution

Identify changes that affect actual computational or application logic.

Explain:

* Previous logic.
* New logic.
* Mathematical or algorithmic differences, where applicable.
* Changes in execution flow.
* Changes in inputs and outputs.
* Implications for correctness, performance, and behavior.

Distinguish algorithmic changes from simple refactoring.

## 8. Performance and Resource Optimization History

Document all identifiable optimization efforts.

For each:

* Original bottleneck.
* Optimization introduced.
* Implementation details.
* Expected benefit.
* Measured benefit, if available.
* Benchmark conditions.
* Trade-offs.
* Verification status.

Do not present unmeasured optimization intentions as proven performance gains.

## 9. Bug Fixes and Corrective Engineering

Document significant defects and their resolutions.

Include:

* Observed problem.
* Root cause, if established.
* Affected components.
* Corrective implementation.
* Verification method.
* Remaining concerns.

## 10. Experimental and Evaluation History

Where experiments exist, document:

* Experiment objectives.
* Configurations.
* Methodological changes.
* Metrics.
* Observed outcomes.
* Conclusions supported by results.
* Decisions influenced by the findings.

Separate measured results from interpretations.

## 11. Failed, Reverted, and Deprecated Approaches

Record identifiable:

* Reverted commits.
* Deprecated implementations.
* Replaced architectures.
* Abandoned experimental branches.
* Known unsuccessful configurations.

Explain why they were abandoned only when evidence supports the explanation.

## 12. Dependency and Environment Evolution

Document:

* Python versions.
* Framework versions.
* CUDA and hardware compatibility changes.
* Dependency additions, removals, and upgrades.
* Environment configuration.
* Reproducibility implications.

## 13. Current Project State

Describe the verified state of the project at the time of inspection.

Include:

* Current architecture.
* Main entry points.
* Core modules.
* Implemented capabilities.
* Available tests.
* Known limitations.
* Unresolved issues.

## 14. Technical Debt and Outstanding Issues

Identify:

* Incomplete implementations.
* Duplicated logic.
* Fragile dependencies.
* Compatibility risks.
* Missing tests.
* Inconsistent configurations.
* Performance bottlenecks.
* Documentation gaps.
* Reproducibility risks.

Classify each finding as:

* Confirmed.
* Suspected.
* Requires verification.

## 15. Future Development Recommendations

Provide technically justified recommendations based on actual project findings.

For each recommendation, include:

* Problem or opportunity.
* Proposed modification.
* Expected benefit.
* Risk or trade-off.
* Priority.
* Required validation.

Do not present speculative features as existing project capabilities.

## 16. Complete Change Registry

Create a consolidated table:

| ID | Date/Commit | Component | Change Type | Description | Motivation | Impact | Evidence |
| -- | ----------- | --------- | ----------- | ----------- | ---------- | ------ | -------- |

Include all meaningful identified changes, not just major milestones.

## 17. Final Engineering Assessment

Provide an objective assessment of:

* Development maturity.
* Architectural consistency.
* Code quality.
* Maintainability.
* Performance engineering.
* Testing and reproducibility.
* Historical traceability.
* Remaining technical risks.

Clearly distinguish verified strengths from subjective engineering assessments.

## 18. Investigation Limitations

Explicitly document:

* Missing Git history.
* Unavailable commits or branches.
* Missing experiment logs.
* Unrecoverable previous implementations.
* Any areas that could not be fully inspected.

---

## 6. DOCUMENTATION QUALITY REQUIREMENTS

The final document must be:

* Technically detailed.
* Chronologically coherent.
* Professionally written.
* Clear and unambiguous.
* Evidence-based.
* Consistent in terminology.
* Free of fabricated details.
* Free of generic AI-generated filler.
* Useful for future development, debugging, research, and maintenance.

Avoid repetitive descriptions and meaningless statements such as "the code was improved" without explaining the actual modification.

Explain complex changes at a level appropriate for senior engineers.

Preserve technical specificity, including module names, class names, function names, algorithms, parameters, and implementation differences whenever evidence is available.

Do not compress the history into a superficial executive summary.

Completeness and accuracy take priority over brevity.

---

## 7. EXECUTION PROTOCOL

Follow this sequence:

### Phase 1 — Discovery

Inspect the project structure, Git metadata, documentation, source files, and available experimental artifacts.

### Phase 2 — Historical Reconstruction

Build an internal chronological registry of identifiable changes and their supporting evidence.

### Phase 3 — Technical Investigation

Analyze significant changes across architecture, implementation, algorithms, performance, and system behavior.

### Phase 4 — Cross-Verification

Verify dates, commit references, file paths, implementation claims, and stated outcomes.

Identify contradictions and unresolved historical gaps.

### Phase 5 — Documentation

Generate the complete `PROJECT_HISTORY_AND_TECHNICAL_EVOLUTION.md`.

### Phase 6 — Final Audit

Review the generated document against the actual project.

Verify:

* No fabricated facts.
* No unsupported claims.
* No major subsystem omitted.
* No meaningful identified change omitted from the registry.
* No incorrect chronology.
* No confusing intended improvements with measured outcomes.
* No contradictions between the timeline and current-state analysis.
* All major conclusions have traceable evidence.

Correct any findings before declaring the document complete.

---

## 8. FINAL DELIVERABLE

Create the Markdown file directly in the project root unless another output location is explicitly specified.

Do not modify application source code, experiment results, configurations, or Git history.

The only intended deliverable is the historical documentation file.

After completing the task, provide a concise completion report stating:

1. Exact output file path.
2. Scope of the investigation.
3. Historical period covered, if verifiable.
4. Number of meaningful changes documented, if countable.
5. Major areas investigated.
6. Important historical gaps or limitations.
7. Whether the final documentation audit was completed.

**FINAL DIRECTIVE: Do not produce a generic project overview. Reconstruct the actual engineering evolution of the project, explain the technical and logical significance of its changes, and preserve an auditable historical record grounded entirely in available evidence.**
