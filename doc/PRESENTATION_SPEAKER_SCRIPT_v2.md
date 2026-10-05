# Speaker script — *One model. Any camera. A twelfth of the weights.*

Companion to [`PRESENTATION_MedMamba_SS_TRM_v2.pptx`](PRESENTATION_MedMamba_SS_TRM_v2.pptx). **Generated from `build_pptx_v2.py`** — edit the script there and the deck, the notes and this file all move together.

Written to be **said out loud**, not read off the screen. Short sentences, no undefined jargon, and every likely objection answered before it is asked.

Built to a **20-minute** budget, questions not included. Part 1 is deliberately **one slide** — the room already knows MedMamba. Part 3 is deliberately **the longest** — the room does not know TRM, and Part 4 is unreadable without it.

Every note is written to a word count. **If you add a sentence, take one out.**

| | |
|---|---|
| Runtime | **19:59** as written, measured from this script at 140 words/minute plus 10 % for pointing and pausing |
| Audience | assumes **MedMamba**; assumes nothing about **TRM** |
| Spine | MedMamba → what we changed → TRM → how we integrated it → does it work |
| `[brackets]` | stage directions. Do not say these |

## Five delivery rules

1. **Never read a number that is already on the slide.** Point at it and say what it *means*.
2. **Say "none of this is ours" at both ends of Part 3.** Once going in, once coming out. The whole talk’s credibility rests on the room knowing which half is which.
3. **The 7 starred slides carry the talk.** If you are behind, take the time from anywhere else.
4. **The backup deck is large on purpose** — 22 slides. The results against MedMamba, the cost breakdown, the ablations and the full architecture are all one jump away.
5. **Pause after "that is the whole trick" and after "thirty-one points".** Count two.

## Timing

| # | slide | time | |
|---|---|---:|---|
| 2 | The job | 0:43 |  |
| 3 | The spine | 0:46 |  |
| | **Part 1 — MedMamba** | **1:20** | |
| 5 | Part 1 · MedMamba, in one slide | 1:16 |  |
| | **Part 2 — → MedMamba-SS** | **3:57** | |
| 7 | Part 2 · the problem | 0:48 |  |
| 8 | Part 2 · why C disappears | 1:25 | ★ never cut |
| 9 | Part 2 · the proof | 0:51 | **cuttable (1)** |
| 10 | Part 2 · the hinge | 0:48 |  |
| | **Part 3 — TRM** | **5:32** | |
| 12 | Part 3 · the paper TRM starts from | 1:09 |  |
| 13 | Part 3 · the whole idea, in a picture | 1:35 | ★ never cut |
| 14 | Part 3 · the same thing, precisely | 1:16 | ★ never cut |
| 15 | Part 3 · the paper’s real result | 1:26 | ★ never cut |
| | **Part 4 — → MedMamba-SS-TRM** | **3:03** | |
| 17 | Part 4 · where TRM lands | 0:42 | **cuttable (2)** |
| 18 | Part 4 · what x, y and z actually are here | 1:17 | ★ never cut |
| 19 | Part 4 · what we changed, and why | 0:56 |  |
| | **Part 5 — Results** | **4:03** | |
| 21 | Part 5 · does the spectrum earn its place | 1:12 | ★ never cut |
| 22 | Part 5 · the catch | 1:01 | ★ never cut |
| 23 | Part 5 · what we have not measured | 0:44 | **cuttable (3)** |
| 24 | Takeaways | 0:40 |  |
| 25 | Thank you — questions? | 0:20 |  |
| | **full** | **19:59** | |

> Backup slides are not counted. They are not in the talk.

### Getting shorter

This deck is already the short cut. If the clock is still against you, drop these three, in this order — each is self-contained and nothing later refers back to it. The slide footer names its cut number so you can see it while presenting.

| | cut | saves | leaves |
|---|---|---:|---:|
| 1 | Part 2 · the proof | −0:51 | **19:07** |
| 2 | Part 4 · where TRM lands | −0:42 | **18:25** |
| 3 | Part 5 · what we have not measured | −0:44 | **17:41** |

**Three honest configurations:**

| you have | give |
|---|---|
| **~20 min** | everything as written |
| **~18 min** | cuts 1-3, if the clock is against you |
| **~7 min, TRM only** | Part 3 alone. It mentions nothing of ours and stands up without the rest |

**Never cut** the {nstar} starred slides, and never shorten by gutting the middle of Part 3 — a room that saw the precise version without the maze has not understood it, and Part 4 then lands on nothing.

---

## 1 · One model. Any camera. A twelfth of the weights.

Good morning.

The talk in one sentence: we took MedMamba, made it read a camera with any number of channels,
then shrank it sixty-fold by using the same two layers over and over instead of storing twelve.

I assume you know MedMamba, so Part one is one slide. I assume nothing about TRM, the paper in the
middle — that gets taught, and none of it is ours.

Twenty minutes.

## 2 · The job  —  0:43

*Figure: `figures/fig23_the_task.svg`*

Sort breast tissue into three classes from a microscope image.

Healthy. D-C-I-S — carcinoma still contained inside the milk duct. I-D-C — carcinoma that has
broken out. The middle one is the call that matters: contained or invasive changes how the patient
is treated.

Now the percentages, because they set a trap. Invasive is sixty-nine percent of our test data and
every model gets it essentially perfect. So every number today is balanced accuracy or macro-F1,
which weigh all three classes equally.

And our camera gives thirty-two wavelength bands per pixel, not three colours.

> **On the slide:** **69 % of the test set is a class everybody already gets right.** So every number today is balanced accuracy or macro-F1 — never plain accuracy.

## 3 · The spine  —  0:46

*Figure: `figures/fig21_story_map.svg`*

This slide holds the talk together.

MedMamba has two numbers frozen into its weights at build time, before it sees any data.

C — how many channels it can read, set by the shape of its first layer. The cost is not "works
less well on another camera": the weights will not load. A wall, not a slope.

And twelve — how many distinct blocks it stores. Twenty-seven million numbers, about ninety-nine
percent of the model.

Part two deletes the first, Part four the second. Part three is someone else's paper, and it is
here because Part four needs it.

> **On the slide:** **Two numbers are frozen into the weights at build time.** Part 2 deletes the first. Part 4 deletes the second. Part 3 is the tool Part 4 needs.

---

# PART 1 — MedMamba

## 4 ·   —  0:04

Part one. One slide, because you already know this.

## 5 · Part 1 · MedMamba, in one slide  —  1:16

*Figure: `figures/fig38_medmamba_clear.svg`*

A reminder, and the only MedMamba slide. Two operations, repeated.

Top row is the shape. Cut the image into four-by-four patches; each patch becomes ninety-six
numbers, giving a fifty-six by fifty-six grid of cells. Then four stages — and between each one sits
patch merging: glue every two-by-two group of cells into one, then halve the result. Grid halves,
numbers per cell double. Four times over.

Bottom row is the block, and this is the part that does the work. Split a cell's ninety-six numbers
in half. The first forty-eight go through SS2D, which scans the whole grid in four directions
carrying a memory along — the long-range operator. The other forty-eight go through a three-by-three
convolution, which sees only neighbours — local detail. Glue them back, then shuffle the channels so
the halves swap jobs next block.

A long-range operator and a local one for about the price of one. We kept all of it.

[point at the red box]

The only thing we change is that red box.

> **On the slide:** **Two operations, repeated:** patch merging shrinks the grid, the block does the work.

---

# PART 2 — → MedMamba-SS

## 6 ·   —  0:02

Part two. From MedMamba to MedMamba-SS.

## 7 · Part 2 · the problem  —  0:48

*Figure: `figures/fig04_the_problem.svg`*

That first layer is a convolution, and the input-channel count lives inside its
weight tensor. Three on the left, thirty-two on the right. Different shapes.

Say it precisely: it is not that a model trained on one sensor performs badly on another. The file
will not load.

Second consequence, subtler and more important. Nothing in that model represents which wavelength a
channel is. Band seven is just index seven — and ours are not evenly spaced; there is a
two-hundred-and-nineteen nanometre jump between two adjacent ones.

A spectral model that cannot represent a wavelength is not a spectral model. So we deleted that
layer.

> **On the slide:** Not "performs badly on another sensor" — **the weights are a different shape, so they cannot load at all.**

## 8 · Part 2 · why C disappears  —  1:25  ★

*Figure: `figures/fig06_why_c_vanishes.svg`*

Slow down. This is the central trick of the first half.

One patch has a spectrum — one number per band. Thirty-two if hyperspectral, three if RGB.

The normal thing is to feed all of them into one layer at once. That layer needs a slot for each,
and that is where C gets welded in.

So we feed them in one at a time.

There is a single small layer: linear, one number to thirty-two. It takes one band's value and
returns a vector. Its weight is thirty-two numbers. There is no C in it — there cannot be, because
it only ever sees one value, and cannot know how many times it will be called.

Thirty-two bands gives thirty-two tokens. Three gives three. Same weights.

Then we average. Thirty-two vectors averaged is one vector; three averaged is one vector. Same size.
From there on, nothing downstream knows how many bands there were.

That is the whole trick.

[pause — count two]

One loose end: averaging throws away order, so every token is first tagged with its wavelength in
nanometres. A physical quantity, not an index.

> **On the slide:** Feed the bands in **one at a time**, and no weight ever learns how many there are.

## 9 · Part 2 · the proof  —  0:51  *(cuttable)*

*Figure: `figures/fig25_param_proof.svg`*

The receipt.

We ran the same model twice and diffed the two config files field by field. They differ in exactly
one place: which directory the data comes from. One points at the thirty-two-band cubes, the other
at a three-channel rendering of the same captures.

Four hundred and forty-six thousand, four hundred and nine parameters. Both times.

I like this more than any other number here, because it is arithmetic, not statistics — it does not
depend on seed, split or metric.

But it is narrow. Storage does not depend on band count. Whether the model is any good at a band
count it never trained on, we did not test.

> **On the slide:** **446,409 = 446,409.** Not "about the same". Identical — this is arithmetic, not a measurement.

## 10 · Part 2 · the hinge  —  0:48

*Figure: `figures/fig26_where_params_live.svg`*

C is gone. Now look at where the parameters are — this is the hinge.

Everything I just spent five minutes on is thirty-eight thousand parameters: the orange sliver.
Everything else is twenty-seven point four million.

And plainly: our addition made the model bigger. MedMamba was three and a half million;
MedMamba-S-S is twenty-seven. A model with more capacity that scores higher has not demonstrated a
better idea.

Which sets up one question. Can a backbone reuse the same weights instead of storing twelve sets?

That has an answer in a paper from October twenty twenty-five. Nothing in the next seven minutes is
ours.

> **On the slide:** Everything Part 2 did is **0.1 % of the model**. So: can a backbone **reuse** the same weights instead of storing twelve sets?

---

# PART 3 — TRM

## 11 ·   —  0:04

Part three. Nothing in the next seven minutes is ours.

## 12 · Part 3 · the paper TRM starts from  —  1:09

*Figure: `figures/fig39_hrm.svg`*

One minute on the paper TRM starts from, because TRM is defined
by what it deleted.

HRM — the Hierarchical Reasoning Model. Two small recurrent networks at two speeds: a slow one
updating rarely, a fast one updating often, passing state back and forth. Four layers each,
twenty-seven million parameters. On about a thousand examples it beat the large language models on
Sudoku, mazes and ARC.

But it justified its design with two things that should make an engineer uneasy. Biology — arguments
about timescales in the brain. And a fixed-point theorem: assume the loop settles, and you only
backpropagate the last two steps of six.

TRM says the loop never settles. Hold that — it comes back in two slides.

So TRM is a list of deletions. Two networks to one. Four layers to two. Biology and theorem gone.
Twenty-seven million parameters to seven — and every benchmark goes up.

> **On the slide:** **TRM is HRM with things deleted** — and every deletion made it better.

## 13 · Part 3 · the whole idea, in a picture  —  1:35  ★

*Figure: `figures/fig19_trm_maze.svg`*

Picture first; the maths is the next slide.

You are given a maze — a photograph of one. That is x, the question. You look at it once and it
never changes.

You draw a line, start to finish, your best guess, committed, in pencil. That is y, your current
answer — and it is a real answer at every moment. Stop me halfway and I point at the line.

Now you look again. That stretch was a dead end. So you rub out that stretch and redraw just that
stretch. You do not start again.

f, the circle, is you. One brain — the only thing here with any weights in it. And before each
stroke you think: you rule out corridors, none of it on the paper. That is z, the working.

[slow down]

Here is the idea that makes it work. The model is never asked "solve this maze". It is asked: "here
is a line that is wrong in places — make it less wrong."

A model that must be right in one shot needs the capacity to be right in one shot. A model that only
has to make an answer less wrong does not. It borrows what it is missing from time.

[pause — count two]

> **On the slide:** The model is **never** asked "solve this". It is asked "here is a wrong answer — make it less wrong". **That is why a tiny network is enough.**

## 14 · Part 3 · the same thing, precisely  —  1:16  ★

*Figure: `figures/fig28_code.svg`*

Same idea, exactly. Three lines, and it is the whole algorithm.

Six times: z equals f of z plus y plus x — update the working. Then once: y equals f of y plus z —
update the answer. That is one round.

Three things to notice.

One. Both calls are the same network. Same weights, called again. That is the entire parameter
saving.

Two. The inputs are added, not stuck side by side. So f's input is the same shape on call one and
call sixty-three. Concatenate instead and the input grows every round.

Three, my favourite. Look at what is missing from the answer line. The question. x is in the
scribble line, not the stroke line — the only thing telling one shared network which job it is on.
See x, you are thinking. Do not, you are answering.

Which is why a second network is unnecessary, and they measured it: eighty-two point four at ten
million, against eighty-seven point four at five.

> **On the slide:** One shared network: **87.4 % at 5 M**. Two separate ones: **82.4 % at 10 M**. Half the size and five points better.

## 15 · Part 3 · the paper’s real result  —  1:26  ★

*Figure: `figures/fig10_nograd.svg`*

The problem with running something twenty-one times: gradients through all of
it means storing twenty-one sets of activations, and you have thrown away the saving you came for.

HRM's answer was that fixed-point theorem — backpropagate only the last two steps of six. TRM's
objection: there is no stable point. HRM runs four steps and then asserts it converged. Its own
plots never reach zero.

TRM instead runs three rounds per pass, the first two with gradients off. The last carries gradient
through the whole round.

The justification is not a theorem, it is the training objective. Deep supervision runs the whole
recursion, takes a loss, then hands its answer and working to another run of itself. So every pass
is handed an imperfect answer and graded on improving it — which is exactly what the gradient-free
rounds rely on.

Now the number. Full round backpropagated: eighty-seven point four. HRM's one-step gradient,
everything else identical: fifty-six point five.

[pause — count two]

Thirty-one points — the largest single effect in the paper. And TRM's version costs more memory,
not less.

That is the end of the part that is not ours.

> **On the slide:** **31 points** — the largest single effect in the paper. And TRM’s version costs MORE memory, not less.

---

# PART 4 — → MedMamba-SS-TRM

## 16 ·   —  0:07

Part four. Back to us — and I want to be precise about where TRM goes.

## 17 · Part 4 · where TRM lands  —  0:42  *(cuttable)*

*Figure: `figures/fig36_integration.svg`*

MedMamba-SS is three parts stacked up: the spectral pathway at the front,
the four-stage hierarchy in the middle, the head on top.

We replace exactly one — the middle one.

The spectral pathway is reused byte for byte. So is the head. Only the backbone is swapped, and it
is swapped behind an interface that does not move: same input shape in, same output keys out.

That is what makes this a clean substitution rather than two different models with similar names.
One line in the config decides which backbone runs.

> **On the slide:** One of the three parts is replaced. The spectral pathway and the head are reused **byte for byte** — which is the only reason the comparison is one flag.

## 18 · Part 4 · what x, y and z actually are here  —  1:17  ★

*Figure: `figures/fig37_trm_mapping.svg`*

In the maze, x was a photograph and y a pencil line.
Here they are four tensors.

x is the spectral pathway's output through the stem: a grid of feature vectors, one per patch, a
hundred and twenty-eight wide. Computed once, before the recursion, and held fixed for all
sixty-three calls.

It has to be fixed. x is the only term in the loop still referring to the image. If it drifted with
the states, the recursion would lose contact with what it is classifying.

y is the answer, same shape — literally the grid the head reads. z is the working: same shape, never
decoded.

f is two blocks — a mixer, then a gated MLP, each under a post-normalised residual. Three hundred
and ninety-eight thousand parameters, which is eighty-nine percent of the model.

One trap: the start states are buffers, not parameters. The first round that reads them is inside
the gradient-free block, so a parameter there would never train. Silently, with no warning.

> **On the slide:** `x` is computed **once** and held fixed. `y` is what the head reads. `z` is the working.

## 19 · Part 4 · what we changed, and why  —  0:56

*Figure: `figures/fig31_what_we_changed.svg`*

Five changes. Four small; one with consequences.

Shape — they work on token sequences, we on a two-dimensional grid, because we classify images.

The mixer inside f — they use self-attention, we use a depthwise three-by-three convolution. On an
eleven-by-eleven grid attention is overkill, and their own ablation says the mixer is
task-dependent.

Passes — sixteen became three: sixty-three calls, not three hundred and thirty-six.

Deep supervision is the one with consequences. Ours runs all passes inside one forward, because
every stability check we had is written per batch. The price is memory proportional to passes, which
is why gradient checkpointing stopped being optional.

Halting — off. It sat at chance for thirty-three epochs.

And be precise: we implemented TRM's recursion, not TRM's blocks.

> **On the slide:** We implemented TRM’s **recursion**, not TRM’s **blocks**. They use attention; every real run of ours uses a convolution.

---

# PART 5 — Results

## 20 ·   —  0:05

Part five. And the warning up front: single seed, five patients.

## 21 · Part 5 · does the spectrum earn its place  —  1:12  ★

*Figure: `figures/fig14_results.svg`*

The cleanest experiment in the project. Two runs whose config files differ
in exactly one field — the input directory. Same model, same recipe, same patients, same three
hundred and forty-eight thousand test patches.

Left panel: balanced accuracy, ninety point seven against seventy-five point seven. Fifteen points.

But the right panel carries the meaning. Invasive carcinoma: both score zero point nine nine nine.
Identical — and that is sixty-nine percent of the test set, so it says nothing about which is
better.

The whole margin is the healthy-versus-D-C-I-S boundary, and most of it is D-C-I-S: the seven
percent minority class, and the consequential call.

It survives every threshold-free measure, and it survives an asymmetry against it — the
hyperspectral arm stopped early at twelve epochs while RGB trained all twenty.

What weakens it: the validation gap is two points against fifteen on test, and the RGB arm is the
collection's own rendering.

One seed, five patients.

> **On the slide:** **+15.0 points of balanced accuracy**, and essentially all of it is the healthy / DCIS boundary — the call that actually matters.

## 22 · Part 5 · the catch  —  1:01  ★

*Figure: `figures/fig15_the_catch.svg`*

The result most likely to be useful to someone else — a negative one.

Three numbers, read together. Parameters: eight times smaller, which is the number people quote.
FLOPs: a hundred and ninety-one times more arithmetic. Wall-clock: four point four times slower per
epoch.

Weight sharing did not reduce the work. It multiplied it by the number of times you share.

Two consequences. The forward pass runs the full recursion, so inference pays it too. And without
gradient checkpointing a four-hundred-and-forty-seven thousand parameter model exhausts a
sixteen-gigabyte card, because all sixty-three applications hold their activations alive at once.

So: parameter count in a recursive model reports storage, and is close to an inverse indicator of
compute. Quoting nought point four five million without the sixty-three core applications is half a
result.

> **On the slide:** **Parameter count in a recursive model measures storage, and is close to an inverse indicator of compute.** The most transferable result here.

## 23 · Part 5 · what we have not measured  —  0:44  *(cuttable)*

*Figure: `figures/fig34_not_measured.svg`*

The gaps, because I would rather say them than have them asked.

Does the recursion help our model? We do not know. No depth ablation, no hierarchical-versus-
recursive comparison under matched conditions. One flag away, never run.

Do we inherit TRM's results? No. Theirs are puzzle grids with a verifiable answer; ours is patch
classification on medical images. We inherited the schedule, not the evidence.

Why a convolution and not the scan? Speed — ours is a pure Python loop, thirty-nine times slower at
dataset scale.

And underneath all of it: single seed, five patients.

> **On the slide:** Every result is a **single seed**. The hyperspectral evaluation is **five patients**. Neither is a detail.

## 24 · Takeaways  —  0:40

*Figure: `figures/fig35_takeaways.svg`*

Five things.

Delete C from your weight shapes, and one model serves every sensor.

TRM is two lines: working six times, answer once, most of it with gradients off.

Three is the one I would keep. Refinement beats capacity — a model that only has to make an answer
less wrong can be far smaller than one that has to be right in one shot.

But weight sharing trades storage for compute, steeply. Report both numbers.

And be precise about provenance: TRM's recursion, not TRM's blocks.

## 25 · Thank you — questions?  —  0:20

Everything is in the repository — the manuscript with provenance on every
number, the architecture document with the line-by-line citations, and the figures.

There are backup slides after this: the full architecture, the results against MedMamba itself, and
the TRM ablation table.

Thank you. Questions?

## 26 · Backup · not presented  —  0:01

Backup. Not presented.

## 27 · Backup · the reference architecture  —  0:59

*Figure: `figures/fig17_gmedmamba_r_architecture.svg`*

The full reference sheet, drawn in the MedMamba paper figure's own style so the two can
sit side by side. This is the one to print and pin up rather than read from the back of a room.

Left panel is the spectral pathway — follow it down to "mean over the band axis", which is where
C disappears. Right panel is the recursive core — the schedule on top, what f actually is
underneath.

Two things worth pointing at if the question comes up. The red box under the pipeline is the
auxiliary reconstruction decoder: it exists, it is tested, it is gated, and it is off in every
run we report. And bottom right: every call to f is gradient-checkpointed.

This figure carries the earlier name, G-MedMamba-R.

## 28 · Backup · SS2D, if anyone wants it  —  0:41

*Figure: `figures/fig02_ss2d.svg`*

Only if somebody asks what SS2D is.

State-space models read a sequence one step at a time carrying a memory forward, built so a GPU
can run it fast — near-linear in sequence length where attention is quadratic. The catch is they
read a line, and an image is not a line.

MedMamba's answer: read it four ways and add the readings. The orange is our one change — learn a
weight per direction, so a reading order can be turned down if it is not earning its place.

> **On the slide:** **SS2D** = read the image as a line, four different ways, and combine. That is the only piece of MedMamba jargon in this talk.

## 29 · Backup · the recursion, one row per line  —  0:08

*Figure: `figures/fig09_trm_loop.svg`*

The recursion again, one row per line of code, if the four-line version went too fast
for somebody.

## 30 · Backup · the full TRM ablation  —  0:31

The complete ablation, if somebody wants a row I did not show. Every line is T-R-M with
one thing put back, against eighty-seven point four as published.

The last block is the one I find most telling: they tried actual fixed-point iteration, with
TorchDEQ, and it was both slower and generalized worse. That is the strongest evidence that the
fixed point was never what made H-R-M work.

> **On the slide:** Depth is not the knob. HRM across effective depths 9 → 168 only moves 46.4 % → 62.3 %, and is worse at the deepest setting than at depth 48.

## 31 · Backup · a diagram that does not describe the code  —  0:37

*Figure: `figures/fig18_then_vs_now.svg`*

If anyone has seen an older G-MedMamba diagram in circulation, this is the one, and it
does not describe the code.

It shows two parallel branches, dense three-D convolutions, spectral pooling and a multi-modal
fusion block. Three of those were never built. The source file says outright that this codebase
never had a dense Conv3D spectral branch.

Every row of this figure was checked against the source. Treat that older figure as a design
proposal rather than as architecture.

> **On the slide:** If you have seen the older G-MedMamba diagram: **three of the blocks on it were never built.** Treat it as a design proposal, not architecture.

## 32 · Backup · the questions I expect  —  0:02

Short answers, if I need them.

## 33 · Backup · Three names you will hear  —  1:10

*Figure: `figures/fig22_three_models.svg`*

Three names, because you will hear all three and I do not want
them to blur.

MedMamba is the published baseline. Four stages, twelve distinct blocks, and a first layer with
the channel count welded in.

MedMamba-S-S is ours. It is MedMamba with one thing added — a spectral pathway at the front that
reads the bands properly. S-S is spectral-spatial. After that change, C is gone.

MedMamba-S-S-T-R-M is also ours. It is the middle one with one thing swapped: the four-stage
hierarchy replaced by a single small core called sixty-three times. T-R-M is the paper we
borrowed that from.

Look at the parameter counts. Three and a half million, twenty-seven million, then four hundred
and forty-seven thousand. Yes — our first change made the model bigger. I will come back to that
honestly.

One housekeeping note: some of our older figures call the third one G-MedMamba-R. Same model,
earlier name.

> **On the slide:** Each model is the one before it with **exactly one thing replaced**. That is deliberate — it is what lets us say which change caused what.

## 34 · Backup · the camera  —  1:09

*Figure: `figures/fig24_hsi_vs_rgb.svg`*

Second thing: the camera.

An ordinary camera gives you three numbers per pixel — red, green, blue. Three wide,
overlapping buckets. Anything finer than a bucket is averaged away before you see it.

A hyperspectral camera gives you thirty-two narrow bands per pixel, chosen out of seven hundred
and forty the instrument can capture. So each pixel carries a small curve: an actual spectrum.

That matters because light is absorbed and scattered by tissue according to what the tissue is
made of. A spectrum carries some of that chemistry. Three wide buckets integrate most of it
away.

Whether the extra chemistry actually helps a diagnosis is an empirical question, and Part 5 is
our answer — for one dataset. I am not claiming it in general.

But notice the engineering problem it creates. If your model is built for three channels and the
camera gives you thirty-two, you are stuck.

> **On the slide:** A normal camera gives you **3 numbers per pixel**. Ours gives **32** — and the whole first half of this talk is about making a model that does not care which.

## 35 · Backup · MedMamba, in 60 seconds  —  1:02

*Figure: `figures/fig01_medmamba_simple.svg`*

Most of you already know MedMamba, so this is a reminder, not a tutorial, and it is
the only MedMamba slide in the talk.

Four stages. The picture halves, the features double. Average at the end, one linear layer, done.
Inside each stage is the SS-Conv-SSM block: split the channels, send one half through a
two-dimensional selective scan and the other through convolutions, concatenate and shuffle. A
global operator and a local one for roughly the price of one.

We kept all of that. It is worth inheriting and we did not touch it.

The only thing on this slide that matters for the next forty minutes is the red box on the far
left. The patch embedding. The very first layer.

[point at the red callout]

Everything in Part 2 comes out of that one box.

> **On the slide:** You know this. **The only thing we need from it today is the red box** — the patch embedding, where the channel count lives.

## 36 · Backup · the fix  —  1:22

*Figure: `figures/fig05_stem_swap.svg`*

So we deleted that layer.

Top row is MedMamba. Image, one convolution with C baked in, backbone. Locked to one sensor for
life.

Bottom row is ours. Same image, then a chain of small steps — patchify, tokenize, encode,
average, compress — and then the backbone. The next slide is entirely about those two orange
boxes.

Now the cost, and I am putting it on the slide rather than saving it for a limitations section,
because it is the first thing a careful listener objects to.

The backbone no longer sees pixels. It sees a summary of each patch's spectrum. So texture
inside a patch is gone before the backbone runs.

How bad that is depends entirely on patch size. At patch size one, a patch is a single pixel and
you have lost nothing. At patch size eight, it is sixty-four pixels and you have thrown most of
the image away. We use one on the tissue data, which is why this is a fair trade there.

That is the cost. Here is what it buys.

> **On the slide:** **The honest cost, up front:** the backbone no longer sees raw pixels. It sees a summary of each patch’s spectrum, so texture inside a patch is gone.

## 37 · Backup · conditioning every stage  —  1:16

*Figure: `figures/fig03_block.svg`*

The spectral pathway hands out a context map. This slide is what we do with it.

Grey is MedMamba's block, untouched — you know it, so I am not going to walk it again.

The orange badges are ours, and badge one is the only one that matters. The spectrum conditions
the block at every stage, not just once at the input. Concretely: the context map modulates the
features at all four stages, through a FiLM-style scale and shift, plus a per-stage band selector.

Why that rather than just feeding the spectrum in at the front? Because a signal injected once at
the input has to survive four stages of downsampling and channel mixing to still be doing work at
the top. Conditioning at every stage means it never has to.

Badges two and three are ordinary engineering — a learnable dial on the residual branch, and a
feed-forward block.

And the honest cost of all of this is on the next slide but one.

> **On the slide:** **Badge ① is the one that matters:** the spectrum conditions the block at **every** stage, not just at the input. Grey is MedMamba’s and untouched.

## 38 · Backup · it was not free  —  1:36

*Figure: `figures/fig07_magnitude_bug.svg`*

A minute on the version that did not work, because if I only show
you the clean result I am selling you something.

Every band token gets a value and a wavelength tag. The obvious thing is to add them. One line
of code. That is what we did first.

Then we measured the two things we were adding. The embedded band value came out around zero
point zero zero six. The wavelength tag, around zero point five five. The actual signal was
eighty-six to ninety-three times smaller than a tag that is identical for every patch in the
dataset.

So we measured how much the output moved when the input changed. Zero point two five percent.
That front end was, to a good approximation, ignoring its own input.

Here is the part that should bother you. Nothing crashed. No warning. The loss went down. We
only found it because we had a gate that asserts the embedding must move when the input moves.

The fix is on the slide: stop adding, concatenate and put a small network on top.

So if anyone asks whether deleting C is a free lunch — no. The idea is cheap. Making it actually
see its input was not.

> **On the slide:** The first version of the channel-agnostic front end **could not see its own input** — and nothing crashed, and the loss still went down.

## 39 · Backup · why it improves  —  1:11

*Figure: `figures/fig20_trm_maze_refine.svg`*

Now watch what is actually being asked of the model, because this
is the idea that makes the whole thing work.

It is never asked "solve this maze."

It is asked: "here is a maze, and here is a line that is wrong in places. Make it less wrong."

Follow the panels. A confident start, straight into a dead end. That stretch rubbed out, but the
next one still wrong. Both detours gone, but the line stops short. Then it is right.

Every one of those arrows is the same two layers doing the same job.

[slow down]

A model that has to be right in one shot needs the capacity to be right in one shot. A model
that only has to make an answer less wrong does not. It borrows the capacity it is missing from
time — from being run again.

[pause — count two]

If you remember one sentence from this part, that is it.

> **On the slide:** The model is **never** asked "solve this". It is asked "here is a wrong answer — make it less wrong". That is why a tiny network is enough.

## 40 · Backup · the two things it carries  —  1:29

*Figure: `figures/fig27_y_and_z.svg`*

Now the paper's own language, because you will meet
both letters.

y is the answer — and that is a checkable claim, not a figure of speech. Push y through the
output head and you get an actual Sudoku grid. The paper prints one. Decode y halfway through
and you see a partly solved puzzle, getting better.

z is the working. Decode z through the same head and you get nothing meaningful. Nobody looks at
it — not you, not the loss.

So why exactly two? They tested it, and the bars are the answer.

Take z away and it drops to seventy-one point nine — it forgets how it got where it is, so every
pass starts from scratch. Split z into seven features instead of one and it drops to
seventy-seven point six — you have bought capacity and overfitting and nothing else.

Both, as published: eighty-seven point four.

Hold on to that number, because it makes the rest of this part easy. Every table from here is
the same model with exactly one thing put back. So you never have to remember a table — only
whether the number went down.

> **On the slide:** **87.4 % is your reference number** for the rest of Part 3. Every table from here is the same model with one thing put back — so just watch whether the number drops.

## 41 · Backup · the idea that licenses the last one  —  1:40

*Figure: `figures/fig11_deep_supervision.svg`*

Second idea, and it is what holds up the slide I just
showed you.

One loss at the end of a forty-two-deep chain is a miserable thing to train. So instead: run the
whole recursion, check the answer, take a loss — then hand that answer and that working to
another run of the same recursion as its starting point. Up to sixteen times.

Between passes you detach, so pass three does not backpropagate into pass one.

Now notice what that trains, because this is the bit people miss. Every pass is handed an
imperfect answer and graded on improving it. Which is exactly the property the no-gradient
rounds relied on one slide ago.

So the two ideas hold each other up. Deep supervision is what makes the no-grad prelude
legitimate rather than a hack. You cannot take one without the other, and that is what people
get wrong when they reimplement this.

The arithmetic: two layers, times seven calls, times three rounds, times sixteen passes. Six
hundred and seventy-two layers of effective depth, out of two layers of stored weights.

Last piece: halting. A second tiny head looks at the answer and outputs one number — "is this
already right?". Training-time only, so you stop burning sixteen passes on something you solved
in two.

> **On the slide:** **672 layers of effective depth, out of 2 layers of weights.** And deep supervision is what makes the no-grad rounds legitimate — you cannot take one without the other.

## 42 · Backup · does it work  —  1:16

*Figure: `figures/fig29_trm_benchmarks.svg`*

The receipts, and the limits. I am not going to read the numbers.

Seven million parameters, about a thousand training examples. On Sudoku and mazes, every large
language model in the comparison scores a flat zero. T-R-M scores eighty-seven and eighty-five.

But the bottom row is the one that keeps you honest. Grok-4 beats it on A-R-C. The abstract says
"higher than most large language models" and it means most. Quote this paper as "tiny model
beats the frontier" and somebody will correct you, and you will deserve it.

Three more limits. It is not generative — one input, one deterministic answer. Bigger is worse
and nobody knows why: four layers lose to two, and the authors say outright they suspect
overfitting but have no theory. And none of this is measured on medical images.

When we get to Part four, we inherited their recursion schedule. We did not inherit their
evidence.

And that is the end of the part that is not ours.

> **On the slide:** Quote this paper as "tiny model beats the frontier" and you will be corrected, and you will deserve it. **The abstract says "most LLMs" and means it.**

## 43 · Backup · what to carry forward  —  0:45

*Figure: `figures/fig30_trm_carry.svg`*

Twelve minutes compressed into three cards. If you have been
half-listening, start again now, because these three are all you need for Part four.

One. Refinement beats capacity. The model is never asked to solve the problem, only to improve a
wrong answer — so it never needs the capacity to be right in one shot.

Two. One network, two jobs. Whether it can see the question is the only thing that tells it
which job it is on.

Three. Differentiate a whole round, not one step. That was worth thirty-one points.

Everything from here is ours again.

> **On the slide:** Everything from here is ours again.

## 44 · Backup · the swap  —  1:16

*Figure: `figures/fig13_two_backbones.svg`*

Here is the swap itself.

Same spectral front end. Same head. Same trainer, same metrics, same output keys. One line in
the config decides what goes in the middle.

Top row: four stages, twelve distinct blocks, each stored once. Bottom row: a single core — two
blocks, stored once, called a hundred and twenty-six times.

The structural claim is the one that does not move, and I would defend it anywhere: twelve
distinct blocks stored, against two stored and applied a hundred and twenty-six times, behind an
interface you cannot tell apart from outside.

The multiplier does move, so let me get in front of it. You will see "sixty-one times fewer
parameters" in our write-up. That is against a backbone at MedMamba's widths. Against our own
narrower configuration it is about six. Both are true; they are different comparisons. If I just
said sixty-one and somebody checked, they would find the six — and then wonder what else I had
rounded in my favour.

> **On the slide:** **12 blocks stored once, versus 2 blocks stored once and called 126 times** — behind an identical interface. One config flag picks which.

## 45 · Backup · the finished model  —  1:12

*Figure: `figures/fig32_pipeline_simple.svg`*

The finished model, in five boxes.

A cube goes in, with any number of bands. The spectral pathway turns it into a per-patch summary
— and C disappears at the "mean over the band axis" step. The stem widens it to a hundred and
twenty-eight. The recursive core runs sixty-three times. The head normalises, pools, and picks a
class.

Notice what is not here, especially next to the MedMamba diagram from Part one. No stages. No
patch merging. No downsampling. A hundred and twenty-eight wide from the stem to the head.

Two things to point at. Left: same front end, same head, same trainer as the hierarchical
version, so the comparison between them really is one flag.

Right, in red: every call to f is gradient-checkpointed. That is not a tuning knob. It is the
only reason this fits on the card, and I will show you the number behind that in Part five.

> **On the slide:** Notice what is **missing**: no stages, no patch merging, no downsampling. Width 128 from the stem all the way to the head.

## 46 · Backup · where each piece came from  —  1:12

*Figure: `figures/fig16_lineage.svg`*

What we took, and from where.

Left, from MedMamba: the scan, the block, the patch merging, the channel shuffle, the
initialisation policy. Several near-verbatim.

Middle, from T-R-M: the recursion, the no-gradient prelude, the detaching, the buffer start
states, and the weight-averaging helper — which is a straight copy of their file and says so in
its own docstring.

Right, ours: everything spectral, and the starred one, replacing the stem.

One subtlety worth twenty seconds, because it is my favourite thing in this codebase.

Those start states for z and y are buffers, not learnable parameters. Why does that matter? The
first round that reads them runs inside the no-gradient block. Make them learnable and they
would never receive a gradient. Not once. Silently. No crash, no warning, and the loss curve
would look identical.

T-R-M uses buffers. We use buffers. And our source says why, so the next person does not have to
rediscover it.

> **On the slide:** The start states are **buffers, not parameters** — because the first round that reads them is inside the no-grad block. Make them parameters and they silently never train.

## 47 · Backup · against the baseline  —  1:48

*Figure: `figures/fig33_vs_medmamba.svg`*

Second result: against MedMamba itself.

We trained a MedMamba on the same patient split, holding batch size, precision, seed, checkpoint
rule and GPU fixed. Both are scored on exactly the same test patches.

At an eighth of the parameters, ours leads every aggregate metric. Balanced accuracy by four
points. And on D-C-I-S, the class that matters, F1 of zero point seven against zero point
six four.

And it costs four point four times the wall-clock to get there.

Now the caveat, and take it seriously, because it is large. MedMamba trained on a class-balanced
build. Ours drew from the natural thirteen-to-one imbalance and compensated in the loss instead.
Two different ways of fixing the same problem — and ours also sees more distinct patches over
the run. A model that sees more data and is still reweighted toward the minority class is well
placed to lead both raw and balanced accuracy at once, which is exactly the pattern you are
looking at.

Five dimensions are unmatched. So this is indicative, not controlled. I would not attribute that
margin to the backbone, and the experiment that would settle it is one flag away and has not
been run.

One thing it does establish, about the dataset rather than about us: MedMamba peaks at epoch
three of thirty and then declines. A three-and-a-half million parameter model saturates this
training set in three epochs.

> **On the slide:** Leads **every aggregate metric** at an eighth of the parameters — but the two models trained on differently balanced splits, so read it as **indicative, not controlled**.

