# One model. Any camera. A twelfth of the weights.
### MedMamba → MedMamba-SS → MedMamba-SS-TRM

**47 slides** (22 of them backup) · **19:59** as written.

Assumes the room has seen **MedMamba** — Part 1 is one slide. Assumes nothing about **TRM**, which is taught from scratch and is not ours.

The spine is a chain: **MedMamba → what we changed → TRM → how we integrated it → does it work.** There is a progress map on every divider.

Built by `build_pptx_v2.py` → `PRESENTATION_MedMamba_SS_TRM_v2.pptx`. The speaker script is [`PRESENTATION_SPEAKER_SCRIPT_v2.md`](PRESENTATION_SPEAKER_SCRIPT_v2.md); both come from the same source, so they cannot drift.

---

**1.** One model. Any camera. A twelfth of the weights.

**2.** The job  ·  `fig23_the_task.svg`
  — 69 % of the test set is a class everybody already gets right. So every number today is balanced accuracy or macro-F1 — never plain accuracy.

**3.** The spine  ·  `fig21_story_map.svg`
  — Two numbers are frozen into the weights at build time. Part 2 deletes the first. Part 4 deletes the second. Part 3 is the tool Part 4 needs.


## Part 1 — MedMamba

**4.** *section divider*

**5.** Part 1 · MedMamba, in one slide  ·  `fig38_medmamba_clear.svg`
  — Two operations, repeated: patch merging shrinks the grid, the block does the work.


## Part 2 — → MedMamba-SS

**6.** *section divider*

**7.** Part 2 · the problem  ·  `fig04_the_problem.svg`
  — Not "performs badly on another sensor" — the weights are a different shape, so they cannot load at all.

**8.** Part 2 · why C disappears ★  ·  `fig06_why_c_vanishes.svg`
  — Feed the bands in one at a time, and no weight ever learns how many there are.

**9.** Part 2 · the proof  *(cuttable 1)*  ·  `fig25_param_proof.svg`
  — 446,409 = 446,409. Not "about the same". Identical — this is arithmetic, not a measurement.

**10.** Part 2 · the hinge  ·  `fig26_where_params_live.svg`
  — Everything Part 2 did is 0.1 % of the model. So: can a backbone reuse the same weights instead of storing twelve sets?


## Part 3 — TRM

**11.** *section divider*

**12.** Part 3 · the paper TRM starts from  ·  `fig39_hrm.svg`
  — TRM is HRM with things deleted — and every deletion made it better.

**13.** Part 3 · the whole idea, in a picture ★  ·  `fig19_trm_maze.svg`
  — The model is never asked "solve this". It is asked "here is a wrong answer — make it less wrong". That is why a tiny network is enough.

**14.** Part 3 · the same thing, precisely ★  ·  `fig28_code.svg`
  — One shared network: 87.4 % at 5 M. Two separate ones: 82.4 % at 10 M. Half the size and five points better.

**15.** Part 3 · the paper’s real result ★  ·  `fig10_nograd.svg`
  — 31 points — the largest single effect in the paper. And TRM’s version costs MORE memory, not less.


## Part 4 — → MedMamba-SS-TRM

**16.** *section divider*

**17.** Part 4 · where TRM lands  *(cuttable 2)*  ·  `fig36_integration.svg`
  — One of the three parts is replaced. The spectral pathway and the head are reused byte for byte — which is the only reason the comparison is one flag.

**18.** Part 4 · what x, y and z actually are here ★  ·  `fig37_trm_mapping.svg`
  — x is computed once and held fixed. y is what the head reads. z is the working.

**19.** Part 4 · what we changed, and why  ·  `fig31_what_we_changed.svg`
  — We implemented TRM’s recursion, not TRM’s blocks. They use attention; every real run of ours uses a convolution.


## Part 5 — Results

**20.** *section divider*

**21.** Part 5 · does the spectrum earn its place ★  ·  `fig14_results.svg`
  — +15.0 points of balanced accuracy, and essentially all of it is the healthy / DCIS boundary — the call that actually matters.

**22.** Part 5 · the catch ★  ·  `fig15_the_catch.svg`
  — Parameter count in a recursive model measures storage, and is close to an inverse indicator of compute. The most transferable result here.

**23.** Part 5 · what we have not measured  *(cuttable 3)*  ·  `fig34_not_measured.svg`
  — Every result is a single seed. The hyperspectral evaluation is five patients. Neither is a detail.

**24.** Takeaways  ·  `fig35_takeaways.svg`

**25.** Thank you — questions?

**26.** Backup · not presented

**27.** Backup · the reference architecture  ·  `fig17_gmedmamba_r_architecture.svg`

**28.** Backup · SS2D, if anyone wants it  ·  `fig02_ss2d.svg`
  — SS2D = read the image as a line, four different ways, and combine. That is the only piece of MedMamba jargon in this talk.

**29.** Backup · the recursion, one row per line  ·  `fig09_trm_loop.svg`

**30.** Backup · the full TRM ablation
  — Depth is not the knob. HRM across effective depths 9 → 168 only moves 46.4 % → 62.3 %, and is worse at the deepest setting than at depth 48.

**31.** Backup · a diagram that does not describe the code  ·  `fig18_then_vs_now.svg`
  — If you have seen the older G-MedMamba diagram: three of the blocks on it were never built. Treat it as a design proposal, not architecture.

**32.** Backup · the questions I expect

**33.** Backup · Three names you will hear  ·  `fig22_three_models.svg`
  — Each model is the one before it with exactly one thing replaced. That is deliberate — it is what lets us say which change caused what.

**34.** Backup · the camera  ·  `fig24_hsi_vs_rgb.svg`
  — A normal camera gives you 3 numbers per pixel. Ours gives 32 — and the whole first half of this talk is about making a model that does not care which.

**35.** Backup · MedMamba, in 60 seconds  ·  `fig01_medmamba_simple.svg`
  — You know this. The only thing we need from it today is the red box — the patch embedding, where the channel count lives.

**36.** Backup · the fix  ·  `fig05_stem_swap.svg`
  — The honest cost, up front: the backbone no longer sees raw pixels. It sees a summary of each patch’s spectrum, so texture inside a patch is gone.

**37.** Backup · conditioning every stage  ·  `fig03_block.svg`
  — Badge ① is the one that matters: the spectrum conditions the block at every stage, not just at the input. Grey is MedMamba’s and untouched.

**38.** Backup · it was not free  ·  `fig07_magnitude_bug.svg`
  — The first version of the channel-agnostic front end could not see its own input — and nothing crashed, and the loss still went down.

**39.** Backup · why it improves  ·  `fig20_trm_maze_refine.svg`
  — The model is never asked "solve this". It is asked "here is a wrong answer — make it less wrong". That is why a tiny network is enough.

**40.** Backup · the two things it carries  ·  `fig27_y_and_z.svg`
  — 87.4 % is your reference number for the rest of Part 3. Every table from here is the same model with one thing put back — so just watch whether the number drops.

**41.** Backup · the idea that licenses the last one  ·  `fig11_deep_supervision.svg`
  — 672 layers of effective depth, out of 2 layers of weights. And deep supervision is what makes the no-grad rounds legitimate — you cannot take one without the other.

**42.** Backup · does it work  ·  `fig29_trm_benchmarks.svg`
  — Quote this paper as "tiny model beats the frontier" and you will be corrected, and you will deserve it. The abstract says "most LLMs" and means it.

**43.** Backup · what to carry forward  ·  `fig30_trm_carry.svg`
  — Everything from here is ours again.

**44.** Backup · the swap  ·  `fig13_two_backbones.svg`
  — 12 blocks stored once, versus 2 blocks stored once and called 126 times — behind an identical interface. One config flag picks which.

**45.** Backup · the finished model  ·  `fig32_pipeline_simple.svg`
  — Notice what is missing: no stages, no patch merging, no downsampling. Width 128 from the stem all the way to the head.

**46.** Backup · where each piece came from  ·  `fig16_lineage.svg`
  — The start states are buffers, not parameters — because the first round that reads them is inside the no-grad block. Make them parameters and they silently never train.

**47.** Backup · against the baseline  ·  `fig33_vs_medmamba.svg`
  — Leads every aggregate metric at an eighth of the parameters — but the two models trained on differently balanced splits, so read it as indicative, not controlled.

