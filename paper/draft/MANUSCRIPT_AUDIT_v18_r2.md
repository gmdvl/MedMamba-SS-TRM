# Pre-Submission Forensic and Scientific Audit — Manuscript v18, Round 2

- **Document audited:** `paper/draft/MedMamba-SS-TRM_manuscript_v18.md` and `_supplement_v18.md`, as rebuilt after the easy fixes (resolution log in `MANUSCRIPT_AUDIT_v18.md`)
- **Protocol:** `doc/paper_analysis.prompt.md`
- **Date:** 2026-09-26
- **Round 1:** `MANUSCRIPT_AUDIT_v18.md`, findings 1–39, which are not restated here.
- **This round:**
  1. Confirms each round-1 fix in the built text.
  2. Carries forward the open round-1 findings.
  3. Reports new findings 40–48, some of them introduced or exposed by the fixes.
- **New sources checked in this round:**
  - The headers of all 644 calibrated cubes in the raw collection.
  - The data build's `prep_log.txt` and `dataset_manifest.json`.
  - All image paths in the manuscript and supplement (every file exists).
- **The manuscript was not edited in this round.**

---

## A. Round-1 fixes confirmed in the built text

All 23 fixes marked "Fixed" in the round-1 resolution log are present in the built manuscript. The numeric diff against the pre-fix build is clean: no result number changed.

The build also passes these checks:
- Tables and figures are numbered in placement order (16 tables, 13 figures).
- 57 references with unchanged numbering.
- The abstract is 250 words.
- No pending markers.
- Every figure file exists.

## B. Round-1 findings still open

The full finding text is in `MANUSCRIPT_AUDIT_v18.md`.

| # | Severity now | Type | Status | Required resolution |
| ---: | --- | --- | --- | --- |
| 1 | HIGH | Methodological | Open | `PROVIDE EXPERIMENTAL EVIDENCE`: retrain on the training-only band build |
| 2 | HIGH | Methodological | Open | `VERIFY` which recipe decisions used test scores; frozen-recipe patient CV |
| 3 | HIGH | Scientific | Open | `PROVIDE EXPERIMENTAL EVIDENCE`: patient-level CV |
| 4 | HIGH | Scientific | Open | `CONFIRM INTENDED MEANING`: re-scope Contribution 1 and the title, or show MedMamba-SS training on HSI |
| 5 | MEDIUM (was HIGH) | Logical | Mostly resolved: the Conclusion now separates design properties from findings and says recursion vs a same-size core is untested | Residual: `PROVIDE EXPERIMENTAL EVIDENCE` (a non-recursive control); question 2 is still posed as "replace", but answered only on cost |
| 11 | MEDIUM | Citation | Open | `VERIFY` [19]'s band handling and the name "MelanoSpec-SSM"; add it to Table I |
| 15 | MEDIUM | Technical | Open | `CONFIRM INTENDED MEANING`: the naming note under the author line |
| 16 | MEDIUM | Methodological (REPRODUCIBILITY CRITICAL) | Open | `PROVIDE DATA`: the TCIA version, and why 33 images and 2 patients are absent |
| 17 | MEDIUM | Methodological | Partly resolved (split IDs, constant, augmentation added) | `PROVIDE DATA`: code access for review; encoder and decoder widths; probe hyperparameters |
| 19 | REVIEW | Citation | Open | `CONFIRM INTENDED MEANING`: the supervisor-group citation cluster |
| 30 | LOW | Citation | Partly resolved (repository wording) | `PROVIDE SOURCE`: the "image-level split" claim about MedMamba's published PAD results (III-C) |
| 31 | REVIEW | Citation | Open | `VERIFY` [29], [30], [2] and [5]–[8] against the full texts |
| 34 | LOW | Redundancy | Partly resolved | `REWRITE`: MedMamba-SS "chance" restated in about 7 places; "storage, not compute" in about 5 |
| 35 | REVIEW | AI Artifact | Partly resolved | `KEEP` or light `REWRITE` |
| 36 | REVIEW | AI Artifact (disclosure) | Open | `VERIFY` the venue policy |
| 38 | LOW | Markdown | Open | `PROVIDE DATA`: institution and e-mail |

---

## C. New findings

## FINDING 40

**Severity:** `MEDIUM`
**Audit type:** `Methodological`
**Location:** Section III-A → "Patient split and patches"

**Original text:**
> "no pixel-level mask is applied (the region annotations distributed with the collection were not used), so a patch can contain stroma or background from a capture labelled DCIS or IDC."

**Problem:** The sentence describes the data correctly: every full-size capture yields all 4,914 patches (Table III). But it implies a deliberate choice. The recorded preparation command requested ROI filtering: `--roi_min_frac 0.8`, with `roi_root` resolved to the collection's `01_03_HSI ROI_Annotations` folder (`prep_log.txt`, `dataset_manifest.json`). The filtering silently matched nothing, because the annotations are per-patient files and the lookup searched for per-capture names (project notes, v14 audit).

**Why it matters:** A reader, or anyone rerunning the recorded command with a corrected pipeline, would get a different, ROI-filtered dataset. The paper's methods and the recorded command disagree.

**Evidence:** `data/hsi_v8-80_10_10_importance-new/prep_log.txt`, line 1 (argv). The manifest `extra.cli_args.roi_min_frac = 0.8` and `roi_root`. Table III patch counts equal full tiling.

**Confidence:** `Confirmed`
**Research impact:** `Moderate`

**Required resolution:** `CLARIFY`, e.g. "ROI filtering was requested in preparation but matched no annotation files, so every patch of every capture was kept". Make sure the released code and command document this.

**Human approval required:** `YES`

---

## FINDING 41

**Severity:** `LOW`
**Audit type:** `Numerical` (dataset)
**Location:** Table III footnote; Section VII-C → "Patient-concentrated errors"

**Original text:**
> "The ten smaller captures (2,002 patches each) are all in validation and are exactly the IDC captures of patient 65; we have not determined why they are smaller."
> "Patient 65's ten IDC captures are smaller than all others and are almost never classified as IDC, for a reason we have not identified."

**Problem:** The size can now be stated. In the distributed collection, exactly ten calibrated cubes (`calibrated.hdr`) have 250 lines × 1,004 samples: `HS_VNIR_65_IDC_x10_C01`–`C10`. All other 634 are 600 × 1,004. 250 × 1,004 tiles into 22 × 91 = 2,002 patches of 11 × 11.

**Why it matters:** "Not determined" invites the suspicion of a preprocessing error. The headers show the size comes from the dataset as distributed. The *reason* for the shorter acquisition, and whether it explains the 0.000–0.002 IDC recall, remains unknown.

**Evidence:** The `lines` field of all 644 `calibrated.hdr` files: 634 × 600 and 10 × 250 (patient 65 IDC only).

**Confidence:** `Confirmed`
**Research impact:** `Minor`

**Required resolution:** `CORRECT`, e.g. "are 250 × 1,004-pixel cubes in the distributed collection (all others are 600 × 1,004); we have not determined why they were acquired smaller, or whether this relates to their classification".

**Human approval required:** `NO` (a factual addition)

---

## FINDING 42

**Severity:** `LOW`
**Audit type:** `Technical` (figure consistency)
**Location:** Section IV intro (colour code); Fig. 5; Fig. 6; Table VIII

**Original text:**
> "blue for taken from TRM as published" (IV intro)
> Fig. 5 caption: "In (b), the recursive core (blue) follows TRM's schedule (Fig. 6)"

**Problem:** Fig. 5 colours the shared core blue, which the paper's colour code defines as "taken from TRM as published". But Fig. 6 colours the network $f$ amber ("used with modifications"), and Table VIII lists its mixer and feed-forward as replaced or modified. The two figures classify the same component differently.

**Why it matters:** This is a minor internal contradiction in the paper's own colour code.

**Evidence:** The Mermaid `class bCORE fromtrm` (Fig. 5) vs `class tF,tSUP,bF,bSUP changed` (Fig. 6).

**Confidence:** `Confirmed`
**Research impact:** `Minor`

**Required resolution:** `CORRECT`. Colour the Fig. 5 core amber, or say in the caption that blue refers to the recursion schedule only, with $f$ modified (Fig. 6).

**Human approval required:** `NO`

---

## FINDING 43

**Severity:** `LOW`
**Audit type:** `Meaning Preservation`
**Location:** Section VII-B → "The spectral pathway and MedMamba-SS"

**Original text:**
> "A trained instance transfers only partly: evaluated zero-shot on 16 bands, the 32-band model collapses when its encoding scale follows the band count but keeps 0.909 balanced accuracy when the scale is held at its training value, whereas at 8 bands it reaches at most 0.560 (Section VI-C)."

**Problem:** Round-1 fix 8 added "post hoc, one checkpoint" to the Abstract, Contribution 1, VI-C and VIII. VII-B was not updated, so the Discussion now states the same result with fewer qualifiers than every other section.

**Why it matters:** The qualifiers are inconsistent across sections. Of all the sections, only VII-B still presents the result without its scope.

**Evidence:** The Abstract, I ¶7, VI-C and VIII all now carry the qualifier.

**Confidence:** `Confirmed`
**Research impact:** `Minor`

**Required resolution:** `CORRECT`: "…keeps 0.909 balanced accuracy in a post-hoc evaluation of one checkpoint when the scale is held…".

**Human approval required:** `NO` (aligns with already-approved wording)

---

## FINDING 44

**Severity:** `LOW`
**Audit type:** `Methodological`
**Location:** Section I → Contribution 2

**Original text:**
> "On the five test patients it has the highest balanced accuracy of the models compared at four of five seeds (92.8 ± 1.9 %), with HybridSN and SpectralFormer compared seed by seed;"

**Problem:** The abstract now qualifies the comparison ("single-run MedMamba (own recipe)", "untuned HybridSN and SpectralFormer"). Contribution 2 makes the same claim without these qualifiers. The asymmetric tuning (round-1 Finding 13) is disclosed only in V-C and V-D.

**Why it matters:** The Contributions list is where reviewers look for the claims. It should carry the same scope as the abstract.

**Evidence:** The Abstract; V-D.

**Confidence:** `Confirmed`
**Research impact:** `Minor`

**Required resolution:** `CLARIFY`: add "(MedMamba: one run under its own recipe; HybridSN and SpectralFormer untuned)".

**Human approval required:** `YES`

---

## FINDING 45

**Severity:** `LOW`
**Audit type:** `Writing` (terminology)
**Location:** Sections VI-A, VI-F, VI-H, Fig. 7 caption

**Original text:**
> "0.0103 for its RGB twin" (VI-H)
> "better calibrated than its 3-band twin" (VI-H)

**Problem:** The 3-band input arm has five names:
- "3-band arm" (5 uses)
- "RGB arm" (1)
- "RGB twin" (1)
- "3-band twin" (1)
- "RGB build" (4, for the data)

III-B defines "32-band" and "3-band" as the terms, and "twin" is informal.

**Why it matters:** Inconsistent terms for one experimental arm make comparisons harder to follow.

**Evidence:** Term counts in the built manuscript.

**Confidence:** `Confirmed`
**Research impact:** `Minor`

**Required resolution:** `CORRECT`: use "3-band (RGB) arm" for the model and "RGB build" for the data throughout.

**Human approval required:** `NO`

---

## FINDING 46

**Severity:** `LOW`
**Audit type:** `Numerical` (clarity)
**Location:** Section III-A, "Cohort and preprocessing" vs. "Band selection"

**Original text:**
> "(8,618.75, the 99th percentile of the gain-corrected capture medians of 65 captures, 10 % of the collection)"
> "computed on up to 2,000 random pixel spectra from each of 64 captures (10 % of all captures)"

**Problem:** The same subsection gives two "10 %" capture samples, with 65 and 64 captures. They are different samples in the preparation code:
- The constant uses the first ⌈0.1 × 644⌉ = 65 items.
- Band statistics use a seeded random 10 % sample of 64.

The text does not say they differ.

**Why it matters:** A careful reader will see 64 and 65 as an inconsistency.

**Evidence:** The archived v8 preparation script (`items[: max(3, ceil(len(items) * 0.1))]`); `V18_EVIDENCE_AUDIT.md` and the project notes on the 64-capture band sample.

**Confidence:** `Confirmed`
**Research impact:** `Minor`

**Required resolution:** `CLARIFY`: say the constant's 65 captures are a separate sample from the 64 used for band selection. Alternatively, drop "10 % of the collection" from the constant's description.

**Human approval required:** `NO`

---

## FINDING 47

**Severity:** `LOW`
**Audit type:** `Meaning Preservation`
**Location:** Section VI-A, ¶3

**Original text:**
> "The ranking holds at four of five seeds; at seed 1 (89.61 %) MedMamba and the 64-feature probe are ahead."

**Problem:** At seed 1, HybridSN (89.65 %) is also ahead, by 0.04 points. The next paragraph calls the two "level". The sentence lists only two of the three models ahead of MedMamba-SS-TRM at that seed.

**Why it matters:** This is a small omission, but a check of the numbers would find it.

**Evidence:** VI-A "Seed-paired baselines": "at seed 1 the two are level (89.61 % against 89.65 %)".

**Confidence:** `Confirmed`
**Research impact:** `Minor`

**Required resolution:** `CORRECT`, e.g. "…MedMamba and the 64-feature probe are ahead, and HybridSN is level (89.65 %)".

**Human approval required:** `NO`

---

## FINDING 48

**Severity:** `LOW`
**Audit type:** `Writing`
**Location:** Supplementary Table S1 → Selection

**Original text:**
> "early-stopping patience 40 epochs (never reached)"

**Problem:** The runs last 20 epochs (ablations 12), so a 40-epoch patience cannot trigger. "Never reached" states the obvious, and it obscures the actual rule: no early stopping.

**Why it matters:** This is a minor clarity problem.

**Evidence:** V-C (20 and 12 epochs).

**Confidence:** `Confirmed`
**Research impact:** `Minor`

**Required resolution:** `CORRECT`: "no early stopping (patience 40 exceeds the 20-epoch budget)".

**Human approval required:** `NO`

---

# REQUIRED RESOLUTIONS

| # | Severity | Action | Required information / alternative |
| ---: | --- | --- | --- |
| 1 | HIGH | `PROVIDE EXPERIMENTAL EVIDENCE` | Training-only-build retrain. **Alternative:** keep "provisional". |
| 2 | HIGH | `VERIFY` + `PROVIDE EXPERIMENTAL EVIDENCE` | Which decisions used test scores; frozen-recipe CV |
| 3 | HIGH | `PROVIDE EXPERIMENTAL EVIDENCE` | Patient-level CV |
| 4 | HIGH | `CONFIRM INTENDED MEANING` | Contribution 1 scope and title |
| 5 | MEDIUM | `PROVIDE EXPERIMENTAL EVIDENCE` | Non-recursive control; or rephrase question 2 as a cost question |
| 11 | MEDIUM | `VERIFY` + `PROVIDE CITATION` | [19] full text |
| 15 | MEDIUM | `CONFIRM INTENDED MEANING` | Naming note |
| 16 | MEDIUM | `PROVIDE DATA` | TCIA version; exclusions |
| 17 | MEDIUM | `PROVIDE DATA` | Code access; remaining hyperparameters |
| 40 | MEDIUM | `CLARIFY` | ROI filtering requested but ineffective |
| 19 | REVIEW | `CONFIRM INTENDED MEANING` | Self-group citations |
| 30 | LOW | `PROVIDE SOURCE` | Image-level split claim |
| 31 | REVIEW | `VERIFY` | Full-text citation checks |
| 34 | LOW | `REWRITE` | Remaining repetition |
| 35 | REVIEW | `KEEP` / `REWRITE` | Style |
| 36 | REVIEW | `VERIFY` | Venue AI policy |
| 38 | LOW | `PROVIDE DATA` | Affiliation |
| 41 | LOW | `CORRECT` | State the 250 × 1,004 size |
| 42 | LOW | `CORRECT` | Fig. 5 core colour |
| 43 | LOW | `CORRECT` | VII-B qualifier |
| 44 | LOW | `CLARIFY` | Contribution 2 qualifier |
| 45 | LOW | `CORRECT` | One term for the 3-band arm |
| 46 | LOW | `CLARIFY` | 64 vs 65 captures |
| 47 | LOW | `CORRECT` | Seed-1 HybridSN |
| 48 | LOW | `CORRECT` | Early-stopping wording |

---

# FINAL AUDIT REPORT

## Document Status

`MAJOR SCIENTIFIC REVISION REQUIRED`

This status is unchanged from round 1. The text-level contradictions are fixed, and no integrity or fabrication problem was found. Four HIGH findings remain open, and they need experiments or author decisions, not editing:
- Band selection saw held-out labels.
- The test set was used during recipe development.
- Ten held-out patients, with one DCIS patient per set.
- Contribution 1's model does not train on the target data.

## Finding Summary (open findings after round 2: 16 carried forward + 9 new)

| Category | Count |
| --- | ---: |
| Hallucination/Fabrication | 0 |
| AI/Synthetic Indicators | 1 |
| Explicit AI Indications | 1 |
| Scientific Issues | 2 |
| Methodological Issues | 6 |
| Logical Issues | 1 |
| Numerical Issues | 2 |
| Citation Issues | 4 |
| Technical Issues | 2 |
| Meaning Preservation Issues | 2 |
| Grammar/Language Issues | 2 |
| Markdown/Syntax Issues | 1 |
| Redundancy Issues | 1 |
| **Total open** | **25** |

By severity: 4 high, 6 medium, 11 low and 4 review. Round 1 had 39 open findings, including 5 high.

---

# SENIOR RESEARCHER ASSESSMENT

**Research question.** The three questions are clear. Question 3 now states honestly what differs between the arms. Question 2 still asks whether the core can *replace* the hierarchy, but the paper can only answer it on cost.

**Research gap.** Gap 3 now matches what was tested. Gap 2 still needs the [19] check.

**Methodology.** It is formally exact. Preprocessing is now fully described (the gain, the constant, the augmentation), except for the silently ineffective ROI step (Finding 40).

**Experimental design.** Unchanged since round 1:
- Band selection saw held-out labels.
- No held-out set is untouched.
- Most analyses are single-seed.
- No non-recursive control.

**Results.** Every number checked traces to its source. The round-1 statistical misstatements (8-band loss, "two-thirds", PSNR) are corrected.

**Discussion.** It separates evidence from interpretation well. One qualifier is still missing (Finding 43).

**Limitations.** Specific and complete. The round-1 fixes also added the scaling-constant disclosure.

**Contribution.** Unchanged. The spectral pathway and the quantified storage-for-compute trade-off are demonstrated. MedMamba-SS as a hyperspectral classifier is not.

**Reproducibility.** Improved: the split IDs, the constant and the augmentation are now in the supplement. Still missing: the dataset version, the exclusions, code access, and a note on the ROI step.

**Academic writing.** Precise, with residual repetition and inconsistent terms for the 3-band arm.

**Internal consistency.** The round-1 table contradictions are resolved. Three small inconsistencies remain (Findings 42, 43 and 47).

**Research integrity.** No fabrication. Disclosures are thorough, and ref [27]'s author order is now correct.

---

# AI-INDICATION SUMMARY

### Explicit AI indications
There is one: the Acknowledgment names Claude (Anthropic) as an assistant for drafting and code. This is a proper disclosure. No leftover assistant phrasing, prompts or editing instructions were found.

### Synthetic-writing indicators
- Recurring "X, not Y" antitheses.
- Repeated restatements of the MedMamba-SS result and of "storage, not compute".
- A uniform hedging cadence.

Round 1 removed the clearest meta-commentary sentence.

### Evidence of AI authorship
The document's own acknowledgment discloses AI assistance. Beyond that, **no direct evidence of AI authorship was established by this audit.**

---

# HIGHEST-PRIORITY ACTIONS

| Priority | Finding | Problem | Required action | Evidence needed |
| --- | ---: | --- | --- | --- |
| HIGH | 1 | Held-out labels in band selection | Retrain on `hsi_v9-trainsel` | New-build results for the headline pair |
| HIGH | 2 | Test set used during recipe development | List the decisions that used test scores; frozen-recipe CV | Run-log review; CV results |
| HIGH | 3 | 10 held-out patients; one DCIS patient per set | Patient-level CV | CV results |
| HIGH | 4 | MedMamba-SS not shown on HSI | Re-scope Contribution 1 and the title, or fix training | Author decision; F9 run |
| MEDIUM | 40 | ROI filtering requested but silently ineffective | Clarify III-A | None (evidence in the prep log) |
| MEDIUM | 16, 17 | Dataset version; code access | Add to the paper and supplement | TCIA version, exclusion rule, code link |
| MEDIUM | 11 | Closest prior SSM work not characterized | Full-text check; add to Table I | [19] full text |
| LOW | 41–48 | Small consistency and clarity items | Apply the listed corrections | None; all evidence is in hand |

---

**STOP.** The manuscript was not modified in this round. Findings 41, 42, 43, 45, 46, 47 and 48 can be applied without new information. Findings 40 and 44 need the author's approval of the wording.
