# Speaker script — *One model. Any camera. A twelfth of the weights.*

Companion to [`PRESENTATION_MedMamba_SS_TRM_v2.pptx`](PRESENTATION_MedMamba_SS_TRM_v2.pptx). **Generated from `build_pptx_v2.py`** — edit the script there and the deck, the notes and this file all move together.

Written to be **said out loud**, not read off the screen. Short sentences, no undefined jargon, and every likely objection answered before it is asked.

Part 1 is deliberately **one slide** — the room already knows MedMamba. Part 3 is deliberately **the longest** — the room does not know TRM, and Part 4 is unreadable without it.

| | |
|---|---|
| Runtime | **49:46** as written, measured from this script at 140 words/minute plus 10 % for pointing and pausing |
| Audience | assumes **MedMamba**; assumes nothing about **TRM** |
| Spine | MedMamba → what we changed → TRM → how we integrated it → does it work |
| `[brackets]` | stage directions. Do not say these |

## Four delivery rules

1. **Never read a number that is already on the slide.** Point at it and say what it *means*.
2. **Say "none of this is ours" at both ends of Part 3.** Once going in, once coming out. The whole talk’s credibility rests on the room knowing which half is which.
3. **The 7 starred slides carry the talk.** If you are behind, take the time from anywhere else.
4. **Pause after "that is the whole trick" and after "thirty-one points".** Count two.

## Timing

| # | slide | time | |
|---|---|---:|---|
| 2 | Setting up · the job | 1:14 |  |
| 3 | Setting up · the camera | 1:09 | **cuttable (7)** |
| 4 | The spine of the talk | 1:26 |  |
| 5 | Three names you will hear | 1:10 |  |
| | **Part 1 — MedMamba** | **1:09** | |
| 7 | Part 1 · MedMamba, in 60 seconds | 1:02 |  |
| | **Part 2 — → MedMamba-SS** | **10:26** | |
| 9 | Part 2 · the problem | 1:05 |  |
| 10 | Part 2 · the fix | 1:22 |  |
| 11 | Part 2 · why C disappears | 1:56 | ★ never cut |
| 12 | Part 2 · the proof | 1:24 |  |
| 13 | Part 2 · conditioning every stage | 1:16 | **cuttable (5)** |
| 14 | Part 2 · it was not free | 1:36 | **cuttable (1)** |
| 15 | Part 2 · the hinge | 1:34 |  |
| | **Part 3 — TRM** | **13:18** | |
| 17 | Part 3 · the whole idea, in a picture | 2:08 | ★ never cut |
| 18 | Part 3 · why it improves | 1:11 |  |
| 19 | Part 3 · the two things it carries | 1:29 | **cuttable (6)** |
| 20 | Part 3 · the same thing, precisely | 1:49 | ★ never cut |
| 21 | Part 3 · the paper’s real result | 2:09 | ★ never cut |
| 22 | Part 3 · the idea that licenses the last one | 1:40 |  |
| 23 | Part 3 · does it work | 1:16 | **cuttable (3)** |
| 24 | Part 3 · what to carry forward | 0:45 | **cuttable (8)** |
| | **Part 4 — → MedMamba-SS-TRM** | **9:14** | |
| 26 | Part 4 · where TRM lands | 1:28 |  |
| 27 | Part 4 · what x, y and z actually are here | 2:02 | ★ never cut |
| 28 | Part 4 · the swap | 1:16 |  |
| 29 | Part 4 · what we changed, and why | 1:45 |  |
| 30 | Part 4 · the finished model | 1:12 |  |
| 31 | Part 4 · where each piece came from | 1:12 | **cuttable (2)** |
| | **Part 5 — Results** | **9:26** | |
| 33 | Part 5 · does the spectrum earn its place | 2:16 | ★ never cut |
| 34 | Part 5 · against the baseline | 1:48 | **cuttable (4)** |
| 35 | Part 5 · the catch | 1:54 | ★ never cut |
| 36 | Part 5 · what we have not measured | 1:29 |  |
| 37 | Takeaways | 1:00 |  |
| 38 | Thank you — questions? | 0:33 |  |
| | **full** | **49:46** | |

> Backup slides are not counted. They are not in the talk.

### Getting shorter

Cut in this order. Each one is self-contained, and nothing later in the talk refers back to it. The slide footer names its cut number, so you can see it while presenting.

| | cut | saves | leaves |
|---|---|---:|---:|
| 1 | Part 2 · it was not free | −1:36 | **48:09** |
| 2 | Part 4 · where each piece came from | −1:12 | **46:57** |
| 3 | Part 3 · does it work | −1:16 | **45:40** |
| 4 | Part 5 · against the baseline | −1:48 | **43:51** |
| 5 | Part 2 · conditioning every stage | −1:16 | **42:35** |
| 6 | Part 3 · the two things it carries | −1:29 | **41:06** |
| 7 | Setting up · the camera | −1:09 | **39:56** |
| 8 | Part 3 · what to carry forward | −0:45 | **39:10** |

**Four honest configurations:**

| you have | give |
|---|---|
| **~50 min** | everything as written |
| **~44 min** | cuts 1-4 — the intended default |
| **~39 min** | cuts 1-8, for a real squeeze |
| **~13 min, TRM only** | Part 3 alone. It mentions nothing of ours and stands up without the rest |

**Never cut** the 7 starred slides, and never shorten by gutting the middle of Part 3 — a room that saw the precise version without the maze has not understood it, and Part 4 then lands on nothing.

---

## 1 · One model. Any camera. A twelfth of the weights.

Good morning.

The whole talk in one sentence: we took a published medical image classifier, made it able to
read a camera with any number of colour channels, and then shrank it sixty-fold by using the
same two layers over and over instead of storing twelve different ones.

One assumption and one promise. I am assuming you have seen MedMamba — so Part one is a single
slide. I am assuming nothing about TRM, the paper in the middle; that gets taught from scratch,
and I will say clearly that none of it is ours.

The shape of the talk is a chain. MedMamba. What we changed to get MedMamba-S-S. Then TRM. Then
how we put TRM inside, to get MedMamba-S-S-T-R-M. Then whether any of it worked.

There is a map on every divider so you always know where we are. Let us start with what the model
is for.

## 2 · Setting up · the job  —  1:14

*Figure: `figures/fig23_the_task.svg`*

The job is sorting breast tissue into three classes, from a picture
under a microscope.

Healthy on the left. Then two carcinomas. D-C-I-S, in the middle, is cancer still contained
inside the milk duct. I-D-C, on the right, has broken out into the surrounding tissue.

The middle one is the call that matters. Contained or invasive changes how the patient is
treated — and it is drawn by eye, on shapes that vary between labs, between stains, and between
pathologists.

Now look at the percentages, because they set a trap. Invasive carcinoma is sixty-nine percent
of our test data, and every model we have trained gets it essentially perfect. So a model that
only ever said "invasive" would already score in the sixties.

That is why you will not hear me quote plain accuracy once today. Every number is balanced
accuracy or macro-F1 — both weigh all three classes equally, so a model cannot hide behind the
easy one.

> **On the slide:** **69 % of the test set is a class everybody already gets right.** So we never quote plain accuracy — only balanced accuracy and macro-F1.

## 3 · Setting up · the camera  —  1:09  *(cuttable)*

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

## 4 · The spine of the talk  —  1:26

*Figure: `figures/fig21_story_map.svg`*

This is the slide that holds the talk together. If you keep one
picture, keep this one.

MedMamba is where we start. It is published, it is good, and we kept most of it. But two numbers
are frozen into its weights when you build it, before it sees any data.

The first is C — how many channels it can read, decided by the shape of its very first layer.
And the cost is not "it works less well on another camera". The weights physically will not
load. Different shape, different tensor. It is a wall, not a slope.

The second is twelve — how many distinct blocks it stores. Twelve blocks, each with its own
weights. Twenty-seven million numbers, which turns out to be about ninety-nine percent of the
model.

Part two deletes the first. Part four deletes the second.

Part three is the odd one out. It is somebody else's paper, from October twenty twenty-five, and
I will say this at both ends of it: none of Part three is ours. It is here because Part four does
not make sense without it.

> **On the slide:** **Two numbers are frozen into the weights at build time.** Part 2 deletes the first. Part 4 deletes the second. Part 3 is the tool Part 4 needs.

## 5 · Three names you will hear  —  1:10

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

---

# PART 1 — MedMamba

## 6 ·   —  0:07

Part one, and it is one slide, because most of you already know this model.

## 7 · Part 1 · MedMamba, in 60 seconds  —  1:02

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

---

# PART 2 — → MedMamba-SS

## 8 ·   —  0:09

Part two. From MedMamba to MedMamba-SS.

Two changes, and one of them we can prove rather than measure, which I like.

## 9 · Part 2 · the problem  —  1:05

*Figure: `figures/fig04_the_problem.svg`*

So: that first layer.

It is a convolution, and the number of input channels lives inside its weight tensor. Three on
the left, thirty-two on the right. Different shapes. They do not fit.

Say that precisely, because the vague version is much weaker. It is not that a model trained on
one sensor performs badly on another. The file will not load. You get a shape-mismatch error and
nothing runs.

There is a second consequence, subtler and I think more important. Nothing in that model
represents which wavelength a channel is. Band seven is just "index seven". Swap two bands
around and it cannot tell. And our bands are not even evenly spaced — there is a
two-hundred-and-nineteen nanometre jump between two adjacent ones.

So: a spectral model that cannot represent a wavelength is not really a spectral model.

> **On the slide:** It is not that the model **performs badly** on another sensor. The weights are a different shape, so they **cannot load at all**.

## 10 · Part 2 · the fix  —  1:22

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

## 11 · Part 2 · why C disappears  —  1:56  ★

*Figure: `figures/fig06_why_c_vanishes.svg`*

Slow down here. This is the central trick of the first half.

Take one patch. It has a spectrum — one number per band. Thirty-two if it is hyperspectral,
three if it is RGB.

The normal thing is to feed all of them into one layer at once. That layer then needs a slot for
each one, and that is exactly where C gets welded in.

So we do not do that. We feed them in one at a time.

There is a single small layer: linear, from one number to thirty-two. It takes one band's value
and hands back a vector. Its weight is thirty-two numbers, full stop. There is no C in it. There
cannot be — it only ever sees one value at a time, and it has no way of knowing how many times
it will be called.

So thirty-two bands gives thirty-two tokens. Three bands gives three tokens. Same layer, same
weights, both times.

Then we average them. Thirty-two vectors averaged is one vector; three averaged is one vector.
Same size. And from there on, nothing downstream has any idea how many bands there were.

That is the whole trick.

[pause — count two]

One loose end, because someone always spots it. Averaging throws away order. So before we
average, every band token is tagged with its actual wavelength in nanometres, as a
sine-and-cosine pattern. That is a physical quantity, not an index — so the tag works just as
well for a wavelength the model has never seen.

> **On the slide:** Feed the bands in **one at a time**, and no weight ever learns how many there are.

## 12 · Part 2 · the proof  —  1:24

*Figure: `figures/fig25_param_proof.svg`*

And here is the receipt.

We ran the same model twice. We diffed the two configuration files field by field, and they
differ in exactly one place: which directory the data comes from. One points at the
thirty-two-band cubes; the other at a three-channel rendering of the very same captures.

Four hundred and forty-six thousand, four hundred and nine parameters. Both times.

Not "about the same". Not "the same after you subtract the classifier head". Identical. A
ten-fold change in band count contributed exactly zero parameters.

I like this result more than any other number in the talk, for an odd reason: it is arithmetic,
not statistics. It does not depend on the seed, or the split, or the metric.

But let me say what it is not, because it is narrow. It says storage does not depend on band
count. It says nothing about whether the model is any good at a band count it never trained on
— we did not test that. And memory while running does still depend on band count. Storage is
flat; activations are not.

> **On the slide:** **446,409 = 446,409.** Same model, same dataset, same three classes. The two runs differ in exactly one field: the input directory.

## 13 · Part 2 · conditioning every stage  —  1:16  *(cuttable)*

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

## 14 · Part 2 · it was not free  —  1:36  *(cuttable)*

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

## 15 · Part 2 · the hinge  —  1:34

*Figure: `figures/fig26_where_params_live.svg`*

C is gone. Now look at where the parameters actually are, because this
is the hinge of the talk.

Everything I just spent five minutes on — the whole spectral front end, the thing that made the
model sensor-independent — is thirty-eight thousand parameters. That is the orange sliver, and
on this scale you can barely see it.

Everything else — the four-stage backbone, the fusion, the heads — is twenty-seven point four
million. Ninety-nine point nine percent.

And plainly: our own addition made the model bigger overall. MedMamba was three and a half
million; MedMamba-S-S is twenty-seven. A model with more capacity that scores a bit higher has
not demonstrated a better idea, and I am not going to pretend otherwise.

Which sets up exactly one question. Can a backbone reuse the same weights instead of storing
twelve separate sets?

That has an answer in a paper from October twenty twenty-five — and a bigger answer than we went
looking for, because most of that paper is not about making a model smaller. It is about why
reusing weights makes a model better.

So the next ten minutes are that paper. Nothing in them is ours. I will tell you when we are
back.

> **On the slide:** Everything Part 2 did is **0.1 % of the model**. The backbone is the other 99.9 %.

---

# PART 3 — TRM

## 16 ·   —  0:46

Part three. What is T-R-M?

Say this out loud and mean it: nothing in the next twelve minutes is ours. This is one paper,
by one author, at Samsung's Montreal lab, from October last year. Every number I show you in
this part is theirs, measured on puzzle benchmarks — not on medical images.

I am spending a third of the talk on somebody else's paper on purpose. It is recent enough that
most rooms have not read it, and part four is genuinely meaningless without it.

I will tell you again when we come out the other side.

## 17 · Part 3 · the whole idea, in a picture  —  2:08  ★

*Figure: `figures/fig19_trm_maze.svg`*

Picture first. The maths is three lines and it comes
in three slides.

You are given a maze. Not a description — a photograph of one, on a page in front of you. You
have a pencil and an eraser.

That maze is x. The question. You look at it once and it never changes.

Now you draw a line. Start to finish, your best guess, all the way through, committed, in
pencil. That is y. Your current answer. And the important thing about y is that it is a real
answer at every moment. If I stop you halfway and ask what your answer is, you point at the
line.

Now you look again. That stretch was wrong — dead end. So you rub out that stretch and redraw
just that stretch. You do not start again. You do not get a fresh sheet.

And f, the circle in the middle, is you. One brain. It is the only thing in this picture with
any weights in it. Everything else is paper.

One more piece. Before each stroke, you think — you trace routes with your eyes, you remember
corridors you have ruled out. None of that is on the paper. That is z, the working. Nobody ever
sees it, including the thing that grades you.

Four words, and I will use only these. One update of the working is a scribble. One update of
the line is a stroke. Six scribbles then one stroke is a round. Three rounds is a pass.

And mazes are not a metaphor I picked to be charming. Thirty-by-thirty mazes are one of the four
benchmarks in the paper.

> **On the slide:** **x** is the maze · **y** is the pencil line · **z** is what you worked out but did not draw · **f** is you — and f is the only thing with any weights in it.

## 18 · Part 3 · why it improves  —  1:11

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

## 19 · Part 3 · the two things it carries  —  1:29  *(cuttable)*

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

## 20 · Part 3 · the same thing, precisely  —  1:49  ★

*Figure: `figures/fig28_code.svg`*

Same idea, exactly. Three lines of Python, and it is
the entire algorithm.

Six times: z equals f of z plus y plus x. That is a scribble — update the working. Then once: y
equals f of y plus z. That is a stroke — update the answer. Together, one round.

Three things to notice. All three are easy to miss.

One. Both calls are the same network. Same weights, called again. That is where the entire
parameter saving comes from, and there is genuinely nothing else to it.

Two. The inputs are added, not stuck side by side. That sounds like a detail and it is not.
Because they are added, f's input is the same shape on call one and on call sixty-three.
Concatenate instead and the input grows every round, and deep recursion becomes impossible.

Three, and this is my favourite. Look at what is missing from the answer line. The question. x
is in the scribble line. x is not in the stroke line.

That is the switch. It is the only thing telling one shared network which of its two jobs it is
on. See x, you are thinking. Do not, you are answering.

Which is why a second network is unnecessary — and they measured it. Two networks: eighty-two
point four, at ten million. One shared: eighty-seven point four, at five million. Half the
parameters and five points better.

> **On the slide:** One shared network: **87.4 % at 5 M parameters.** Two separate ones: **82.4 % at 10 M.** Half the size and five points better.

## 21 · Part 3 · the paper’s real result  —  2:09  ★

*Figure: `figures/fig10_nograd.svg`*

Now the problem with running something twenty-one times.

If you want gradients through all twenty-one calls, the framework stores twenty-one sets of
activations. That is a twenty-one-layer memory bill, and you have thrown away the saving you
came for.

The paper T-R-M argues with is called H-R-M, and its answer was a theorem. Assume the recursion
settles to a stable point, invoke the Implicit Function Theorem, and backpropagate through only
the last two steps of six.

T-R-M's objection is blunt, and I think right. There is no stable point. H-R-M runs four forward
steps and then asserts it has converged — it never iterates to an equilibrium, it just stops.
Its own plots never reach zero. The theorem is applied where its condition does not hold.

T-R-M does something else. Three rounds per pass; the first two with gradients off entirely. The
last carries gradient through the whole round, all seven calls. A complete think-then-answer
cycle.

And the justification is not a theorem — it is the training objective. Remember the maze: this
model is trained to take any line and improve it. So running it twice more without gradients
cannot hurt. Those rounds genuinely happen; nothing is stored.

Twenty-one calls forward, seven backward.

And now the number for this whole part. Full round backpropagated: eighty-seven point four.
H-R-M's one-step gradient, everything else identical: fifty-six point five.

[pause — count two]

Thirty-one points — the largest single effect in the paper. And notice which way it goes.
T-R-M's version costs more memory, not less. They spent memory and bought thirty-one points.

So the lesson is not "skip most of the gradient". It is: make the part you do differentiate a
complete round.

> **On the slide:** **31 points** — the largest single effect in the paper. And TRM’s version costs MORE memory than the alternative, not less.

## 22 · Part 3 · the idea that licenses the last one  —  1:40

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

## 23 · Part 3 · does it work  —  1:16  *(cuttable)*

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

## 24 · Part 3 · what to carry forward  —  0:45  *(cuttable)*

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

---

# PART 4 — → MedMamba-SS-TRM

## 25 ·   —  0:16

Part four. Integrating TRM.

Everything in the last twelve minutes was somebody else's. Now we put it inside our model, and I
want to be precise about exactly where it goes and what it touches.

## 26 · Part 4 · where TRM lands  —  1:28

*Figure: `figures/fig36_integration.svg`*

So: how do you actually put TRM inside a medical image classifier?

The answer is that you do not put it anywhere clever. MedMamba-SS is three parts stacked up. The
spectral pathway at the front. The four-stage hierarchy in the middle. The head on top.

We replace exactly one of them. The middle one.

The spectral pathway is reused byte for byte — same code, same weights, same output. The head is
reused byte for byte. Only the backbone is swapped, and it is swapped behind an interface that
does not move: same input shape going in, same output keys coming out.

That is what makes this a clean experiment rather than two different models with similar names.
One line in the config file decides which backbone runs, and everything on either side of it is
identical. Same trainer, same metrics, same checkpoint rule, same gates.

And it is worth saying what this buys us that we have not spent: because the interface holds, the
hierarchical-versus-recursive comparison under matched conditions is genuinely one flag away. We
have not run it. I will come back to that in Part 5.

> **On the slide:** One of the three parts is replaced. The spectral pathway and the head are reused **byte for byte** — which is the only reason the comparison is one flag.

## 27 · Part 4 · what x, y and z actually are here  —  2:02  ★

*Figure: `figures/fig37_trm_mapping.svg`*

And here is the part that is easy to hand-wave, so let me not. In the maze, x was a
photograph and y was a pencil line. In our model they are four specific tensors.

x is the output of the spectral pathway, pushed through the stem. It is a grid of feature
vectors, one per patch position, a hundred and twenty-eight wide. Crucially it is computed once,
before the recursion starts, and then held completely fixed for all sixty-three calls.

It has to be fixed, and the reason is worth twenty seconds. x is the only term left in the loop
that still refers to the actual image. If it drifted along with the states, the recursion would
slowly lose contact with the thing it is supposed to be classifying. It would start refining an
answer to a question it had forgotten.

y is the answer, and it is the same shape. It is literally the feature grid the classification
head reads. Decode it after any of the three segments and you get logits — which is what deep
supervision does.

z is the working. Same shape again, never decoded, never supervised.

And f is two blocks — a token mixer, then a gated MLP, each under a post-normalised residual.
Three hundred and ninety-eight thousand parameters, and that is eighty-nine percent of the entire
model.

Two things on the bottom row. The start states for y and z are buffers, not parameters — I will
say why in three slides. And the arithmetic: seven calls per round, three rounds, three passes.
Sixty-three.

> **On the slide:** `x` is the spectral summary, computed **once** and held fixed. `y` is the grid the head reads. `z` is the working. `f` is 398,592 parameters, and it is the whole core.

## 28 · Part 4 · the swap  —  1:16

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

## 29 · Part 4 · what we changed, and why  —  1:45

*Figure: `figures/fig31_what_we_changed.svg`*

Five changes. Four small; one with consequences.

Shape. They work on sequences of tokens. We work on a two-dimensional grid, because we classify
images.

The mixer inside f. They use self-attention; we use a depthwise three-by-three convolution. On
an eleven-by-eleven grid attention is overkill — and remember their own ablation showed the
mixer is task-dependent. So this is a choice their paper invites.

Passes. Their six and three we kept exactly. Their sixteen passes became three — sixty-three
calls to f rather than three hundred and thirty-six.

Deep supervision is the one with consequences. Theirs does one pass per optimizer step, carrying
state across batches. Ours does all three passes inside one forward. That was deliberate: every
stability check we had — gradient health, class collapse, NaN detection — is written per batch,
and their scheme would have invalidated all of them. The price is memory proportional to the
number of passes, which is exactly why gradient checkpointing stopped being optional.

Halting. Theirs is per sample and on. Ours is whole-batch and off — it sat at chance for
thirty-three epochs while being twenty percent of the objective.

The green bar is what we did not touch: the list Part three said was load-bearing.

And be precise about this, because it is the easiest thing to get caught on. We implemented
T-R-M's recursion. We did not implement T-R-M's blocks.

> **On the slide:** We implemented TRM’s **recursion**, not TRM’s **blocks**. They use attention; every real run of ours uses a convolution.

## 30 · Part 4 · the finished model  —  1:12

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

## 31 · Part 4 · where each piece came from  —  1:12  *(cuttable)*

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

---

# PART 5 — Results

## 32 ·   —  0:23

Part five. Does it work?

Three results and one cost. And a warning I will repeat at the end: everything here is a single
seed, and the hyperspectral evaluation is five patients. Three hundred and forty-eight thousand
patches sounds like a lot, but the effective sample size is five people.

## 33 · Part 5 · does the spectrum earn its place  —  2:16  ★

*Figure: `figures/fig14_results.svg`*

First result, and it is the cleanest experiment
in the project.

Two runs whose configuration files differ in exactly one field: the input directory. One reads
the thirty-two-band cubes, the other a three-channel rendering of the same captures. Same model,
same parameter count, same recipe, same patients, same three hundred and forty-eight thousand
test patches. A single-variable experiment, which is rare and worth something.

Left panel: balanced accuracy, ninety point seven against seventy-five point seven. Macro-F1,
zero point eight six against zero point seven three. Fifteen points.

But the right panel carries the meaning. Invasive carcinoma: both arms score zero point nine
nine nine. Identical — and that is sixty-nine percent of the test set, so it tells you nothing
about which model is better.

The entire margin is the healthy-versus-D-C-I-S boundary, and most of it is D-C-I-S itself — the
seven percent minority class, and the clinically consequential call. F1 goes from zero point
four four to zero point seven zero.

Two things make it harder to dismiss. It survives every averaging scheme and every
threshold-free measure — R-O-C, precision-recall, kappa all move the same way. And it survives
an asymmetry running against it: the hyperspectral arm stopped early at twelve epochs while the
RGB arm trained all twenty. The winning arm had forty percent less training.

What weakens it: the validation gap is only two points against fifteen on test — different five
patients. Per-patient it is two big wins, one tie, two small losses. And the RGB arm is the
collection's own rendering, not one we computed ourselves.

So: on this dataset, the spectrum earns its place, in the class where it matters. One seed, five
patients. Not a claim about hyperspectral imaging in general, and not a clinical result.

> **On the slide:** **+15.0 points of balanced accuracy**, and essentially all of it is the healthy / DCIS boundary — the call that actually matters.

## 34 · Part 5 · against the baseline  —  1:48  *(cuttable)*

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

## 35 · Part 5 · the catch  —  1:54  ★

*Figure: `figures/fig15_the_catch.svg`*

And now the result most likely to be useful to somebody else — and it
is a negative one.

Three numbers, read together, because separately each one misleads.

Parameters: eight times smaller. That is the number people quote. F-L-O-Ps: a hundred and
ninety-one times more arithmetic. Wall-clock: four point four times slower per epoch.

Weight sharing did not reduce the work. It multiplied the work by the number of times you share.
That is the general form — it applies to anything built this way, not just to us.

Two consequences a reader should not have to infer. The forward pass runs the full recursion, so
inference pays this too. There is no cheap evaluation path.

And the bottom-left box: without gradient checkpointing, a four-hundred-and-forty-seven thousand
parameter model exhausts a sixteen-gigabyte card, because all sixty-three applications hold
their activations alive at once. One point four gigabytes with it; nearly seven without, and it
dies at batch sixty-four.

So the sentence at the bottom is what I would take from this whole project. Parameter count in a
recursive model reports storage efficiency, and is close to an inverse indicator of compute.
Anyone who says "zero point four five million parameters" without saying "sixty-three core
applications and mandatory checkpointing" has given you half a result.

Where is the trade good? Where storage or distribution bandwidth binds — many sensor
configurations from one small artefact, or updates to edge devices. It is bad wherever
throughput or training cost binds.

> **On the slide:** **Parameter count in a recursive model measures storage, and is close to an inverse indicator of compute.** This is the most transferable result in the project.

## 36 · Part 5 · what we have not measured  —  1:29

*Figure: `figures/fig34_not_measured.svg`*

Before takeaways, the gaps. I would rather say these
than have them asked, and they are the three best questions you could ask me.

Does the recursion actually help our model? We do not know. No depth ablation, no
hierarchical-versus-recursive comparison under matched conditions. One command-line flag away,
never run. Until it is done, the honest position is that we know what sixty-three core
applications cost and not what they buy.

Do we inherit T-R-M's results? No, and this matters. Theirs are puzzle grids with a verifiable
answer and a thousand-fold augmentation. Ours is patch classification on medical images. We
inherited the schedule, not the evidence.

Why a convolution and not the state-space scan in the core? Speed — our scan is a pure Python
loop, about thirty-nine times slower at dataset scale. It is implemented, tested, and no real
run has used it. So in practice this is a recursive convolutional model with a Mamba-based front
end. I would rather say that first than have you find it.

And two standing weaknesses under all of it. Every result is a single seed. And the
hyperspectral evaluation is five patients.

> **On the slide:** Every result in this talk is a **single seed**. The hyperspectral evaluation is **five patients**. Neither of those is a detail.

## 37 · Takeaways  —  1:00

*Figure: `figures/fig35_takeaways.svg`*

Five things.

One. Delete C from your weight shapes. A shared per-value projection plus a physical wavelength
tag, and one model serves every sensor you own.

Two. T-R-M is two lines. Working six times, answer once, most of it with gradients off.

Three — and this is the one I would keep. Refinement beats capacity. A model that only has to
make an answer less wrong can be far smaller than one that has to be right in one shot. It
borrows what it is missing from time.

Four. But weight sharing trades storage for compute, steeply. Eight times smaller, a hundred and
ninety-one times more arithmetic, and checkpointing went from optional to structural. Report
both numbers, always.

Five. Be precise about provenance. We implemented T-R-M's recursion, not T-R-M's blocks.

## 38 · Thank you — questions?  —  0:33

Everything here is in the repository. The manuscript has all the numbers with their
provenance marked; the architecture document has the line-by-line citations; the figures are in
the doc folder.

There are backup slides after this one — the full reference architecture, the complete T-R-M
ablation table, and one diagram that is in circulation and does not actually describe our code,
which I will explain if anyone has seen it.

Thank you. Questions?

## 39 · Backup · not presented  —  0:01

Backup. Not presented.

## 40 · Backup · the reference architecture  —  0:59

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

## 41 · Backup · SS2D, if anyone wants it  —  0:41

*Figure: `figures/fig02_ss2d.svg`*

Only if somebody asks what SS2D is.

State-space models read a sequence one step at a time carrying a memory forward, built so a GPU
can run it fast — near-linear in sequence length where attention is quadratic. The catch is they
read a line, and an image is not a line.

MedMamba's answer: read it four ways and add the readings. The orange is our one change — learn a
weight per direction, so a reading order can be turned down if it is not earning its place.

> **On the slide:** **SS2D** = read the image as a line, four different ways, and combine. That is the only piece of MedMamba jargon in this talk.

## 42 · Backup · the recursion, one row per line  —  0:08

*Figure: `figures/fig09_trm_loop.svg`*

The recursion again, one row per line of code, if the four-line version went too fast
for somebody.

## 43 · Backup · the full TRM ablation  —  0:31

The complete ablation, if somebody wants a row I did not show. Every line is T-R-M with
one thing put back, against eighty-seven point four as published.

The last block is the one I find most telling: they tried actual fixed-point iteration, with
TorchDEQ, and it was both slower and generalized worse. That is the strongest evidence that the
fixed point was never what made H-R-M work.

> **On the slide:** Depth is not the knob. HRM across effective depths 9 → 168 only moves 46.4 % → 62.3 %, and is worse at the deepest setting than at depth 48.

## 44 · Backup · a diagram that does not describe the code  —  0:37

*Figure: `figures/fig18_then_vs_now.svg`*

If anyone has seen an older G-MedMamba diagram in circulation, this is the one, and it
does not describe the code.

It shows two parallel branches, dense three-D convolutions, spectral pooling and a multi-modal
fusion block. Three of those were never built. The source file says outright that this codebase
never had a dense Conv3D spectral branch.

Every row of this figure was checked against the source. Treat that older figure as a design
proposal rather than as architecture.

> **On the slide:** If you have seen the older G-MedMamba diagram: **three of the blocks on it were never built.** Treat it as a design proposal, not architecture.

## 45 · Backup · the questions I expect  —  0:02

Short answers, if I need them.

