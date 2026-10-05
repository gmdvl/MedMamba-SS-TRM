# Speaker script — *From MedMamba to G‑MedMamba‑R*

Companion to [`PRESENTATION_MedMamba_to_TRM.md`](PRESENTATION_MedMamba_to_TRM.md).
Written to be **said out loud**, not read off the screen.

| | |
|---|---|
| Runtime | **26:55** as written, **24:15** in the intended cut, measured from this script at ~140 words/minute plus 10% pointing time |
| Audience | engineers who have **already seen MedMamba**. No TRM background assumed |
| `[brackets]` | stage directions — where to point, when to pause. Do not say these |
| **bold** | land on this word |

**The talk has one spine, and it is on the roadmap slide.** Two numbers are frozen into MedMamba's weights — `C`, the channel count, and 12, the number of distinct blocks it stores. Part 1 deletes the first. Part 3 deletes the second. Part 2 is the tool Part 3 needs. Every time the room looks lost, go back to that sentence.

**MedMamba itself gets 95 seconds and not a second more.** Slides 1 and 2 are recap for a room that has seen it. Their only job is to leave the red box on the patch‑embed in everyone's head. Resist the urge to teach SS2D; if someone wants it, `fig02_ss2d.svg` is in `doc/figures`.

**Part 2 is half the talk, and that is deliberate.** TRM is a 2025 paper nobody in the room will have read, and Part 3 is meaningless without it. Slides 7–14 mention nothing of ours; they can be lifted out and given alone as a ~13‑minute TRM seminar.

**Appendix A is for you, not for the room.** It is the G‑MedMamba‑R detail the slides deliberately don't carry — the pipeline shapes, the tokenizer's three fusion modes, what our deep‑supervision deviation cost in memory, and the fact that every real run uses a convolution rather than SS2D. Read it once before presenting; most hard questions land there.

**Results are not in this deck.** That is a deliberate scope choice — this is an architecture talk. You have the numbers in Appendix B if asked, and `fig14_results.svg` and `fig15_the_catch.svg` are built and ready to show. Say "ask me afterwards and I'll show you" and mean it.

---

## Six delivery rules

1. **Never read a number that is already on the slide.** Point at it and say what it *means*.
2. **The four starred slides are the talk.** *Why C vanishes*, *A maze solved in pencil*, *The same thing precisely*, *The gradient question*. If you are behind, take the time from anywhere else.
3. **Slide 6 is the hinge.** It is the only thing that explains why the room is about to spend twelve minutes on Sudoku. If you rush one slide in Part 1, do not let it be this one.
4. **Say "none of this is ours" at both ends of Part 2.** Once entering, once leaving. The whole talk's credibility rests on the audience knowing which half is which.
5. **Slides 7 and 8 are the ones that actually teach TRM.** The maze does the work; 9 through 12 only make it precise. If the room looks lost later, go back to the maze rather than pressing on.
6. **Pause after every "that's the whole trick".** Let it land. Count two.

---

## Timing table — measured, not estimated

Every figure below is the word count of the script on that slide, at 140 words per minute,
plus 10% for pointing at figures and pausing. Stage directions are excluded from the count.

| # | slide | time | |
|---|---|---:|---|
| 0 | Roadmap | 0:45 | |
| | **Part 1 — Deleting `C`** | **5:55** | |
| 1 | MedMamba, in one slide | 0:55 | |
| 2 | The three things we hung off its block | 0:40 | |
| 3 | Problem 1 — `C` is welded in | 0:55 | |
| 4 | The fix: delete the stem | 1:00 | |
| 5 | **Why `C` vanishes** | 1:25 | ★ never cut |
| 6 | **Problem 2 — the backbone** | 1:00 | ★ the hinge |
| | **Part 2 — What is TRM?** | **12:55** | |
| 7 | **A maze, solved in pencil** | 2:15 | ★ never cut |
| 8 | Why it gets better each pass | 1:10 | never cut |
| 9 | `y` is the answer, `z` is the working | 1:20 | |
| 10 | **The same thing, precisely** | 1:45 | ★ never cut |
| 11 | **The gradient question** | 2:15 | ★ never cut |
| 12 | Deep supervision, and halting | 1:30 | |
| 13 | What TRM deleted from HRM | 1:15 | **cuttable** |
| 14 | Does it work, and what it doesn't claim | 1:25 | **cuttable** |
| | **Part 3 — What we built** | **7:20** | |
| 15 | Two backbones, one flag | 0:50 | |
| 16 | Where each piece came from | 1:00 | |
| 17 | What we changed, and why | 1:45 | |
| 18 | The whole thing, properly | 1:15 | |
| 19 | What we have not measured | 1:10 | |
| 20 | Takeaways | 0:55 | |
| 21 | Close | 0:25 | |
| | **full** | **26:55** | |

### Getting shorter

Cut in this order. Each is self‑contained, sits at the **end** of a part, and nothing downstream depends on it.

| | cut | saves | leaves |
|---|---|---:|---:|
| 1 | **Does it work** (14) — say "7M parameters, beats every LLM on Sudoku, loses to Grok‑4 on ARC" and move on | −1:25 | 25:30 |
| 2 | **What TRM deleted from HRM** (13) — one sentence: "TRM is a stripped‑down HRM, and every deletion made it better" | −1:15 | **24:15** |
| 3 | **Where each piece came from** (16) — the buffer story is lovely and nothing depends on it | −1:00 | 23:15 |
| 4 | **The block** (2) — keep the split, drop the three badges | −0:40 | 22:35 |
| 5 | Takeaways brisk — five headlines, no elaboration | −0:25 | **22:10** |

**Cuts 1 and 2 are the intended default.** Both slides sit at the very end of Part 2, both are
reference tables rather than teaching, and dropping them lands the talk at **24:15** — which is
the right shape for a 25‑minute slot with questions. Everything below cut 2 is for a real squeeze.

**Do not get shorter by gutting the middle of Part 2.** Slides 7 and 8 are where TRM is actually taught; 10 through 12 only make it precise. A room that saw the precise version without the maze has not understood it, and Part 3 then lands on nothing.

**Five configurations that are honest:**

| you have | give |
|---|---|
| **~25 min** *(the intended shape)* | the script **minus slides 13 and 14** — 24:15, leaving room for questions |
| **~27 min** | everything as written |
| **~22 min** | cuts 1–5 above |
| **~13 min, TRM only** | slides 7–14. They mention nothing of ours and stand alone |
| **~12 min, us only** | Parts 1 and 3, with slides 7 and 8 kept as the only TRM content |

The four ★ slides carry the talk. If you are running behind, take the time from anywhere else.

---
## 0 · Roadmap — 0:45

One idea holds this talk together, and it's on this slide.

MedMamba has **two numbers frozen into its weights**. `C` — how many channels it can read. And **twelve** — how many distinct blocks it stores.

[*point at the two rows*]

Both are decided for you at build time, and both cost you something real. Part one deletes the first. Part three deletes the second.

Part two is the odd one out: it's a paper, not ours, and it's here because part three doesn't work without it.

You've all seen MedMamba, so the first two slides go fast.

---
---

# PART 1 — Deleting `C`

---

## 1 · MedMamba, in one slide — 0:55
Quick recap, and I do mean quick — you've seen this.

[*point along the pipeline*]

Four stages. The picture halves, the features double. Average at the end, one linear layer, done. That's a ResNet's shape, that's Swin's shape.

The only word I'll define is **SS2D**: a state‑space scan — think an RNN built so a GPU can actually run it — applied to an image. An image isn't a line, so there's no obvious place to start, and MedMamba's answer is to read it four ways and add the results.

[*point at the red callout*]

Now — hold that red box. The patch embedding, the very first layer. It is the entire reason Part 1 exists and I'm back on it in ninety seconds.

---

## 2 · The three things we hung off its block — 0:40
The block, and what's ours.

[*trace down the centre*]

Grey is MedMamba's and untouched: split the channels, one half gets the state‑space operator, the other gets convolutions, glue them back and **shuffle**. A global operator and a local one for the price of one.

[*point at the orange badges*]

Orange is ours. Badge **one** is the one that matters: the spectrum conditions the block at **every** stage, not just at the input. Two and three are standard — a learnable dial on the residual branch, and a feed‑forward block.

---

## 3 · Problem 1 — `C` is welded into the first layer — 0:55
So. That first layer.

[*point at the two weight shapes*]

It's a convolution, and the number of input channels sits **inside its weight tensor**. Three on the left, thirty‑two on the right. Different shapes. They do not fit together.

And we don't have three colour channels. We have thirty‑two wavelength bands, chosen out of seven hundred and forty.

[*point at the table, one row at a time*]

One — a model trained on one sensor **cannot load** into a model for another. Not "performs badly". Cannot load.

Two, the subtler one: nothing in that model represents **which wavelength** a channel is. Band seven is just "index seven". Swap two bands round and it can't tell.

[*pause*]

A spectral model that can't represent a wavelength isn't really a spectral model.

---

## 4 · The fix: delete the stem — 1:00
So we deleted it.

[*top row*] That's MedMamba. Image, convolution with `C` in it, backbone. Locked to one sensor.

[*bottom row*] That's ours. Same image. Then a chain of small steps — patchify, tokenize, encode, average, compress — and **then** the backbone.

The two orange boxes are where the work happens, and the next slide is entirely on them.

[*point at the cost line*]

The cost is on the slide rather than in a limitations section, because it's the first thing a careful listener objects to. The backbone **no longer sees pixels**. It sees a per‑patch summary of the spectrum — so texture inside a patch is gone. At patch size one that costs nothing; at patch size eight it's most of the image.

That's a real trade. What we bought for it is the next slide.

---

## 5 · Why `C` vanishes — 1:25  ★

[*slow down here*]

Take one patch. It has a spectrum — one number per band. Thirty‑two numbers if it's hyperspectral, three if it's RGB.

Instead of feeding all of them into one layer, we feed them in **one at a time**.

[*point at the orange box*]

There is a single small layer. `Linear`, one to thirty‑two. It takes **one number** and gives back a vector of thirty‑two. Its weight is thirty‑two numbers. There is no `C` in it. There **can't** be — it only ever sees one value at a time.

[*left panel, then right panel*]

Thirty‑two bands gives thirty‑two tokens. Three bands gives three tokens. Same layer. Same weights.

[*point at the green boxes*]

Then we average them. Thirty‑two vectors averaged is one vector. Three vectors averaged is one vector. **Same size.**

And that's the whole trick.

[*pause — count two*]

One loose end. Averaging throws away the order — which band was which? So before we average, each token is tagged with its actual wavelength in nanometres, as a sine‑and‑cosine pattern. The model can still tell 450 from 700.

[*point at the green bar*]

Four hundred and forty‑six thousand, four hundred and nine parameters at thirty‑two bands. Same number at three. Not "about the same". Identical.

---

## 6 · Problem 2 — the backbone is the expensive part — 1:00  ★ *the hinge*

`C` is gone. And the model is still a four‑stage hierarchical stack — twelve distinct blocks, every one of them stored separately.

[*point at the two rows*]

Look at where the parameters actually are. Everything I just spent five minutes on — the whole spectral front end — is thirty‑eight thousand parameters. The backbone is twenty‑seven **million**.

So: **can a backbone reuse the same weights instead of storing twelve sets?**

[*pause*]

That has an answer in a paper from October last year — and a bigger answer than we went in for, because most of that paper isn't about making a model smaller. It's about why reusing weights makes a model **better**.

So the next twelve minutes are that paper. Nothing in them is ours. I'll tell you when we're back.

---
---

# PART 2 — What is TRM?

> Say it out loud before slide 7: **"Nothing in the next twelve minutes is ours."** Then say it again at the end of Part 2. People conflate the two halves otherwise, and it is the single most damaging thing that can happen to this talk's credibility.

---

## 7 · A maze, solved in pencil — 2:15  ★

Tiny Recursive Model. One author, Samsung's Montreal lab, October last year.

Here's the whole thing — picture first, maths after.

You are given a maze. Not a description of a maze — a **photo** of one, on a page in front of you. You have a pencil and an eraser.

[*point at the left panel*]

That's `x`. The question. You look at it once, and it never changes.

[*point at the middle panel*]

You draw a line. Start to finish, your best guess, all the way through — **committed, in pencil**. That's `y`. Your current answer.

And here's the thing about it: it's a real answer, at every moment. Not a code, not an intermediate. If I stop you halfway and ask "what's your answer?", you point at the line.

[*point at the grey dashes*]

Now you look again. That stretch was wrong — dead end. So you **rub it out and redraw just that stretch**. You don't start the maze again. You don't get a fresh sheet.

[*point at the f circle*]

And `f` is you. One brain. It is the **only** thing in this picture with any weights in it — everything else is just paper.

Before each stroke of the pencil, you think. You trace routes with your eyes, you remember which corridors you've ruled out. **None of that is on the paper.** That's `z` — the working.

[*point at the vocabulary line — slowly*]

Four words, and I'll use only these for the rest of the part. A **scribble** is one update of the working. A **stroke** is one update of the pencil line. Six scribbles then one stroke is a **round**. Three rounds, then you look at the answer, is a **pass**.

[*pause*]

And mazes aren't a metaphor I picked for flavour. Thirty‑by‑thirty mazes are one of the four benchmarks in the paper. This is close to literally what the model does.

---

## 8 · Why it gets better each pass — 1:10

[*point along the four panels*]

Watch what's actually being asked of the model, because this is the bit that makes the whole thing work.

It is **never** asked "solve this maze."

It is asked: "here is a maze, and here is a line that's wrong in places. Make it **less wrong**."

[*pass 1*] A confident start, straight into a dead end.
[*pass 2*] That stretch rubbed out. The next one's still wrong.
[*pass 3*] Both detours gone, but the line stops short.
[*pass 4*] There.

[*slow down*]

Every one of those arrows is the **same two layers**, doing the **same job**.

[*pause*]

A model that has to produce the right answer in one shot needs the capacity to do that. A model that only has to make an answer **less wrong** does not. It borrows the capacity it's missing from **time** — from being run again.

[*pause — count two*]

If you remember one sentence from this part, that's it. Everything on the next three slides exists only to make it possible.

---

## 9 · `y` is the answer, `z` is the working — 1:20

Now the paper's own language, because you'll meet both.

[*point at `y`*]

The concrete claim, and it's genuinely checkable: `y` **is** the answer. Push it through the output head, take the argmax, and you get an actual Sudoku grid — the paper prints one. Decode `y` halfway through and you see a partly‑solved puzzle, getting better. That's the pencil line.

Decode `z` and you get nothing meaningful. It isn't a solution. It's the working.

[*point at the table*]

Now — why exactly **two**? They tested it.

Take `z` away and the model forgets **how** it got where it is. No working to build on. Seventy‑one point nine.

Split `z` into seven separate features? Seventy‑seven point six. You've bought capacity and overfitting and nothing else.

[*point at the 87.4 row — say this deliberately*]

And **eighty‑seven point four is your reference number for the rest of this part**. That's TRM exactly as published, on extreme Sudoku. Every table I show you from here is that same model with **one thing put back** — so all you have to remember is whether the number went down.

---

## 10 · The same thing, precisely — 1:45  ★

Same idea, now exactly. Four lines of Python, not a metaphor.

[*point at the loop, then the line under it*]

Six scribbles: `z` equals `f` of `z` plus `y` plus `x`. Then one stroke: `y` equals `f` of `y` plus `z`. That's a round.

Three things to notice. All three are easy to miss and all three matter.

**One.** Both calls are **the same network**. Same weights, called again. That is where the entire parameter saving comes from. There is nothing else to it.

**Two.** The inputs are **added**, not stuck side by side. That sounds like a detail and it isn't. Because they're added, `f`'s input is the same shape on call one and on call sixty‑three. That's what makes deep recursion possible at all. Concatenate instead, the shape grows every round, and the whole thing falls over.

[*point at the two lines — slowly*]

**Three, and this is my favourite.** Look at what's **missing** from the answer line. The question. `x` is in the scribble line. `x` is **not** in the stroke line.

That's the switch. It is the *only* thing telling one shared network which of its two jobs to do. See `x`, you're thinking. Don't, you're answering.

[*point at the little table*]

Which is why a second network is unnecessary — and they measured it. Two separate networks: eighty‑two point four, ten million parameters. One shared network: eighty‑seven point four, **five** million.

Half the parameters and five points better.

[*pause — count two*]

---

## 11 · The gradient question — 2:15  ★
Now the problem with running something twenty‑one times.

If you want gradients through all twenty‑one, PyTorch has to **store** twenty‑one sets of activations. That's a twenty‑one‑layer memory bill — you've thrown away the saving you came for.

[*point at the HRM paragraph*]

The paper TRM is arguing with is called HRM, and HRM's answer was a fixed‑point theorem. Assume the recursion settles to a stable point, invoke the Implicit Function Theorem, backpropagate through only the **last two of six** steps.

TRM's objection is blunt, and I think correct. **There is no fixed point.** HRM does four forward steps and then *asserts* it's converged — it never iterates to an equilibrium, it just stops. Its own residual plots never reach zero. The theorem is applied where it doesn't hold.

[*point at the three rounds*]

TRM instead: three rounds per pass. The first two run under `no_grad`. The last carries gradient — through the **whole** round, all seven calls. A complete think‑then‑answer cycle.

And the justification isn't a theorem, it's the training objective. Remember the maze: the model is trained to take **any** line and improve it. So running it twice more without gradients can only help. Those rounds genuinely **happen** — the line really does get better in them. Nothing is stored.

[*point at the two numbers, then the table*]

Forward, twenty‑one calls. Backward, seven.

And now the number to take away from this whole part. Full round backpropagated: eighty‑seven point four. HRM's one‑step gradient, everything else identical: **fifty‑six point five**.

[*pause — count two*]

Thirty‑one points. The largest single effect in the paper. And notice it goes the *wrong* way for efficiency — TRM's version **costs more memory** than HRM's. They spent memory and bought thirty‑one points.

So the lesson isn't "skip most of the gradient". It's: make the part you *do* differentiate a complete round.

---

## 12 · Deep supervision, and halting — 1:30
Second idea — and it's the one holding up the last slide.

One loss at the end of a forty‑two‑deep chain is a miserable thing to train. So instead: run the whole recursion, **check the answer**, then hand that answer and that working to another run of the same recursion. Up to sixteen times.

[*point at the scissors*]

Between passes you **detach** — cut the gradient — so pass three doesn't backpropagate into pass one.

Now notice what that trains. Every pass is handed an imperfect answer and graded on improving it — **exactly** the property the `no_grad` rounds relied on a minute ago. These two ideas hold each other up. You can't take one without the other, and that's what people get wrong when they reimplement TRM.

[*point at the arithmetic*]

Layers, times calls per round, times rounds, times passes. **Six hundred and seventy‑two layers of effective depth out of two layers of weights.**

[*point at the halting box*]

Last piece: halting. A tiny second head on `y` outputting one number — *"is this already right?"* Its bias starts at minus five, so it opens at "definitely not done" and has to earn confidence. Training only, so you stop burning sixteen passes on something you solved in two.

---

## 13 · What TRM deleted from HRM — 1:15  *(cuttable)*
A minute of lineage, because you can't see what's clever about TRM without knowing what it deleted.

HRM — Hierarchical Reasoning Model, earlier in twenty twenty‑five. Two transformers, twenty‑seven million parameters, a **thousand** training examples, and it beat the large language models on Sudoku, mazes and ARC. Not narrowly.

But it justified its *design* with two things that should make an engineer uneasy: biology — arguments about brain timescales — and a fixed‑point theorem.

[*point at the table, go down it*]

TRM is a list of deletions. Two networks to one. Four layers to two. Biology gone. Theorem gone. Twenty‑seven million parameters to seven — and **every benchmark goes up**.

[*point at the bottom block*]

And there was already reason to be suspicious. The ARC Prize Foundation audited HRM independently. Deep supervision was worth nineteen percent to thirty‑nine. The recursion — the headline idea, the thing the paper is named after — moved it thirty‑five point seven to thirty‑nine.

Almost nothing. TRM's claim is that the recursion wasn't a weak idea. It was **implemented wrong**.

---

## 14 · Does it work — and what it doesn't claim — 1:25  *(cuttable)*
The receipts, and the limits. I'm not going to read the table.

[*point at the Sudoku column*]

Seven million parameters, a thousand training examples. Every large language model in that column: zero. TRM: eighty‑seven.

[*point at the Grok row — do not skip this*]

But be honest about **this** row. Grok‑4 beats it on ARC. The abstract says "higher than **most** LLMs" and it means most. Quote this paper as "tiny model beats the frontier" and you'll get corrected, and you'll deserve it.

[*point at the TRM‑MLP row*]

And this one, which I love. TRM with an MLP mixer wins Sudoku by thirteen points — and scores **zero point zero** on mazes. Same algorithm. The mixer inside `f` is a task‑dependent choice, not a universal one. Remember that — it comes back when I tell you we used a convolution.

[*point at the last two bullets*]

Two more limits. It is **not** generative — one input, one deterministic answer — so if your problem has several valid answers, this can't express that. And the honest one: **bigger is worse and nobody knows why.** Four layers lose to two. Their words, and unusually candid.

[*pause*]

And that's the end of the part that isn't ours.

---
---

# PART 3 — What we built

---

## 15 · Two backbones, one flag — 0:50
Right — back to us.

Same spectral front end, same head, same trainer, same output keys. One config flag picks what goes in the middle.

[*top row*] Four stages, twelve distinct blocks, each stored once.

[*bottom row*] One core. Two blocks, stored once, **called** a hundred and twenty‑six times.

[*point at the second paragraph*]

The structural claim is the one that doesn't move: twelve blocks stored, versus two stored and applied a hundred and twenty‑six times, behind an identical interface.

The **multiplier** does move, so let me get in front of it. Sixty‑one times is against a MedMamba‑width stack; against our own narrower hyperspectral config it's about six. If I just said "sixty‑one" and somebody checked, they'd find the six.

---

## 16 · Where each piece came from — 1:00
What we took, and from where.

[*point at each column*]

Left, from MedMamba — the scan, the block, the merging, the channel shuffle, the initialisation policy. Several are near‑verbatim.

Middle, from TRM — the recursion, the `no_grad` prelude, the detaching, the buffer start states, and the EMA helper, which is a straight copy of their file and says so in its own docstring.

Right, ours — everything spectral, and the starred one, the stem replacement.

[*point at the callout*]

One subtlety worth twenty seconds, because it's my favourite thing in this codebase.

Those start states for `z` and `y` are **buffers**, not learnable parameters. Why? The first round that reads them is **inside** the `no_grad` block. Make them parameters and they'd never receive a gradient. Ever. Silently. No crash, no warning, and the loss wouldn't move an inch.

---

## 17 · What we changed, and why — 1:45
Five changes, and one of them has consequences.

**Shape.** They work on token sequences. We work on a two‑dimensional grid, because we classify images.

**The mixer inside `f`.** They use self‑attention; we use a depthwise three‑by‑three convolution. On an eleven‑by‑eleven grid attention is overkill — and their own ablation said the mixer is task‑dependent, so this is a choice their paper **invites**.

**Passes.** Six and three are theirs and we kept them. Sixteen passes became three — sixty‑three calls to `f`, not three hundred and thirty‑six.

**Deep supervision** — the one with consequences. Theirs does one pass per optimizer step, carrying state across batches. Ours does **all** passes inside one forward. That was deliberate: every stability check and collapse detector we already had keeps working unchanged. The price is memory — now proportional to the number of passes, which is exactly why gradient checkpointing stopped being optional.

**Halting.** Theirs is per‑sample. Ours is whole‑batch, and **off**: it sat at chance for thirty‑three epochs while being twenty percent of the objective.

[*point at the "what we did not change" line*]

What we didn't touch is the list Part 2 said was load‑bearing. Both lines. One shared network. Additive inputs. A full round backpropagated. Detach between passes.

[*slow down*]

And be precise about this. We implemented TRM's **recursion**. We did not implement TRM's **blocks** — they use attention, we use a convolution. Don't say "we implemented TRM's architecture". You'll get caught.

---

## 18 · The whole thing, properly — 1:15
The reference version — drawn in the MedMamba paper figure's own style, so you can hold them side by side. This is the one to print and pin up; don't try to read it from the back of the room.

[*left panel*] Left panel is Part 1. Follow it down to the green box — "mean over the band axis". That's where `C` disappears.

[*right panel*] Right panel is Part 2. The schedule on top, what `f` actually is underneath.

Notice what's **missing** next to the MedMamba diagram: no stages, no patch merging, no downsampling. Width one‑twenty‑eight from the stem to the head.

[*red box under the pipeline*] Two things to point at. This red box is an auxiliary decoder that reconstructs the input cube. It exists, it's tested, it's gated — and it's **off in every run we've done**.

[*bottom right*] And this line. Every call to `f` is gradient‑checkpointed. Not a tuning knob for us — the only reason this fits on the card.

---

## 19 · What we have not measured — 1:10
Before takeaways — the gaps. I'd rather say these than have them asked.

[*point at row one*]

**Does the recursion actually help our model?** We don't know. No depth ablation, no hierarchical‑versus‑recursive comparison under matched conditions. One command‑line flag away, and never run. It's the first thing I'd do with a spare night.

[*row two*]

**Do we inherit TRM's results?** No, and this matters. Theirs are puzzle grids with a thousand augmentations and a verifiable answer. Ours is patch classification on medical images. We inherited the recursion **schedule**, not the evidence.

[*row three*]

**Why a convolution and not SS2D?** Speed. Our selective scan is a pure‑PyTorch loop, about thirty‑nine times slower at dataset scale. `ss2d` is implemented and no real run has used it.

[*point at the last line*]

And two standing weaknesses. Everything is one seed. And the backbone averages away spatial detail inside each patch — free at patch size one, most of the image at patch size eight.

---

## 20 · Takeaways — 0:55
Five things.

**One.** Delete `C` from your weight shapes and one model serves every sensor. A shared per‑value embedding plus a physical wavelength tag is all it takes.

**Two.** TRM is two lines. Working six times, answer once, most of it under `no_grad`.

**Three** — and this is the one I'd keep. **Refinement beats capacity.** A model that only has to make an answer *less wrong* can be far smaller than one that has to be right in one shot. It borrows what it's missing from time.

**Four.** But weight sharing trades storage for compute, and steeply. It made gradient checkpointing mandatory rather than optional. Report both numbers, always.

**Five.** Be precise about provenance. We implemented TRM's recursion, not TRM's blocks.

---

## 21 · Close — 0:25

Everything here is in the repo. The deep version with line‑by‑line citations is in `PROJECT_ARCHITECTURE_AND_HISTORY.md`, the figures are in `doc/figures`, and the commands that actually run are in the documentations folder.

I've got results — they're not in this deck because this is an architecture talk, but ask me and I'll show you.

Questions?

---
---

# Appendix A — background the slides don't carry

> None of this is on a slide. It is here so that when someone asks, you are answering from
> knowledge rather than from the deck. Read it once before you present.

## A1 · What actually happens to a cube, step by step

This is the pipeline behind the two orange boxes on slide 4. `gmedmamba.py`, class
`GMedMambaRecursiveBackbone`.

| # | step | shape after | what it does |
|---|---|---|---|
| 1 | patchify | `[B, Hp, Wp, C]` | cut the cube into patches; each patch keeps its `C` band values |
| 2 | **tokenize** | `[B, Hp, Wp, C, d]` | the shared `Linear(1 → d)` — **one band value at a time** |
| 3 | wavelength tag | `[B, Hp, Wp, C, d]` | each band token tagged with its nm, as sin/cos |
| 4 | encode | `[B, Hp, Wp, C, d]` | a small Mamba pass **along the band axis** |
| 5 | **mean over bands** | `[B, Hp, Wp, d]` | ← **this is where `C` disappears** |
| 6 | stem | `[B, Hp, Wp, 128]` | one `Linear(d_ctx → 128)` |
| 7 | 2‑D position | `[B, Hp, Wp, 128]` | sinusoidal, scaled by a learnable gain |
| 8 | recursive core | `[B, Hp, Wp, 128]` | 63 calls to `f` |
| 9 | pool + classify | `[B, classes]` | mean over the grid, one linear layer |

Step 5 is the load‑bearing one. Everything before it is per‑band and shape‑agnostic;
everything after it has no idea how many bands there were.

**If asked "what is `d_ctx`?"** — the width of the per‑patch spectral summary handed to the
stem. The stem is the only thing that has to know it, and it is one linear layer.

## A2 · The tokenizer has three fusion modes, and the default is the bad one

`SpectralTokenizer` (`gmedmamba.py:724`) can combine the band value with its wavelength tag
three ways:

- **`add`** — `value_embed(v) + pe`. This is the **legacy default**, and it is the defect:
  the embedded value comes out around 0.006 against a positional tag of about 0.55 — the
  signal is **86–93× smaller than a per‑dataset constant**. Measured input‑dependence of the
  resulting embedding: **0.18% on HSI, 0.29% on RGB.** It is still the default *only* so that
  `train_example_v14.py` keeps reproducing runs already on record.
- **`scaled`** — adds a learnable gain on the tag. Fixes the magnitude. Does not fix the next problem.
- **`concat_mlp`** *(recommended)* — the one to use. With a shared `Linear(1, d)` and a zero
  bias, every band token is `v · W` — a scalar multiple of **one fixed direction** — so all
  bands are collinear and averaging them collapses a whole spectrum to a single scalar. The
  concat‑MLP's nonlinearity makes each band's response *direction* depend on its own
  wavelength, which is the property a spectral tokenizer actually needs.

**Why you should know this:** it is the strongest evidence in the project that the "delete `C`"
idea needed more than the obvious implementation to work. If someone asks whether the
channel‑agnostic stem is a free lunch, this is the honest answer — the first version of it
could not see its own input, and nothing crashed.

## A3 · The recursive core, in code terms

`RecursiveCore` (`gmedmamba.py:1711`). Our configuration:

| knob | ours | TRM | note |
|---|---|---|---|
| `trm_dim` | 128 | 512 | working width, constant end to end |
| `trm_core_layers` | 2 | 2 | the only weights in the core |
| `trm_n_latent` (`n`) | 6 | 6 | working updates per round |
| `trm_n_improve` (`T`) | 3 | 3 | rounds per pass; **2 under `no_grad`** |
| `trm_deep_supervision_steps` | **3** | 16 | passes — this is our main deviation |
| `trm_mixer` | `ss2d` by default, **conv in every real run** | attention | see A5 |
| `trm_ema_rate` | 0.999 | 0.999 | |
| `trm_act_halting` | **off** | on | sat at chance for 33 epochs |

**The arithmetic to have ready:** `(n + 1) × T` = 21 calls to `f` per pass; × 3 passes = **63
calls**; × 2 layers = **126 block applications**. Parameters: **446,409**.

**Where those parameters sit** — summed from the headline run's `model_state`
(`experiments/20260906_222725_…/best_model.pt`) **[verified]**:

| module | parameters | share |
|---|---:|---:|
| `backbone.core` — the 2 recursive blocks | 398,592 | 89.2% |
| `backbone.spectral_pathway` | 38,724 | 8.7% |
| `backbone.stem` | 8,320 | 1.9% |
| `head` (`fc` + `q_head`) | 772 | 0.2% |

This is the number behind slide 6: the entire spectral contribution of Part 1 is **38,724
parameters**, against a hierarchical backbone of ~27 M. Note the state‑dict sum is 446,665
against the config's `backbone_num_params` of 446,409 — a 256‑element `stem_norm` and the head
are counted differently by the two. Quote **446,409**, which is the number on record.

**`init_states` returns buffers, not parameters** — and if someone asks why it matters, the
answer is that the first read of them happens inside `torch.no_grad()`, so an `nn.Parameter`
there would never receive a gradient, silently, forever.

## A4 · Where our deep supervision differs, and what it cost

TRM runs **one pass per optimizer step** and carries `(y, z)` across mini‑batches. We run
**all three passes inside one forward**, then one backward on the averaged loss.

- **Why:** every stability gate we already had — gradient health, class collapse, NaN
  detection — is written per batch. Their scheme would have invalidated all of them.
- **What it cost:** peak memory is now `O(passes)`. Every grad‑carrying pass keeps its whole
  subgraph alive until the single backward, and the SS2D mixer additionally retains one
  activation per scan step. Measured: a batch of 128 on an 11×11 grid exceeded **13 GB**
  before checkpointing. With `trm_checkpoint_core` on it is about **1.4 GB**; off, nearly
  **7 GB**, and it dies at batch 64.

So gradient checkpointing is not a tuning flag in this model. It is structural.

## A5 · "G‑MedMamba‑R" is, in every real run, a recursive *conv* model

`trm_mixer` defaults to `ss2d`, but **no real run has used it.** Our selective scan is pure
PyTorch — a Python loop — and it measured about **39× slower** at dataset scale. Every result
we have used a depthwise 3×3 conv inside `f`.

Say it plainly if asked: *the recursion is TRM's, the spectral front end is Mamba‑based, and
the core mixer is a convolution.* Anyone who reads the config will find this, so say it first.

**Where the time goes:** about **61%** is in the spectral pathway, not the recursive core —
that Python scan loop again. `torch.compile` recovers roughly half of it.

## A6 · The cost of weight sharing, in numbers

Slide 20's fourth takeaway, with the figures behind it. **`fig15_the_catch.svg` draws this** —
show it if the question is asked twice.

- parameters **0.446 M vs 3.65 M** (~8× down) against the matched MedMamba‑HSI baseline
- FLOPs **6,203 M vs 33 M** — about **190× up**. *Order of magnitude, not a measurement:
  the manuscript tags this panel as having no artefact in the repository.*
- wall‑clock **679 s vs 155 s** per epoch — about **4.4× up**
- `forward()` runs the full recursion, so **inference pays it too** — there is no cheap eval path
- peak memory without checkpointing: **1.4 GB → 6.9 GB**, OOM at batch 64

The general form: **weight sharing does not reduce the work, it multiplies it by the number of
times you share.** Anyone quoting "0.45 M parameters" without "63 core applications and
mandatory checkpointing" has given you half a result.

**On the 61× vs 6× discrepancy** (slide 15): 27 M is the hierarchical stack at *MedMamba's*
widths — 96/192/384/768 — and is **[unverified]**. Our own hierarchical arm at the HSI config
is ~2.77 M, which is where the ~6× comes from. The claim that does not move is structural:
12 distinct blocks stored, versus 2 stored and applied 126 times.

## A7 · Things that are built but off, and things that were never built

- **The reconstruction decoder** is real, tested and gated — and **off in every run**. It was
  silently broken for a long time: it read a *detached* feature map, so it trained the decoder
  and could not shape the encoder. Fixed, with a gate that fails the run if it regresses.
  Whether it helps classification has never been measured.
- **An older G‑MedMamba diagram** is in circulation showing two parallel branches, 3‑D
  convolutions, spectral pooling and a multi‑modal fusion block. Three of those were **never
  built**; `gmedmamba_efficient.py:20` says outright *"this codebase never had a dense Conv3D
  spectral branch."* Treat that figure as a design proposal, not architecture. If it comes up
  in questions, say so.

---
---

# Appendix B — likely questions, and honest answers

> Do not bluff any of these. The honest answer is more impressive than a guess, and this
> project's whole credibility rests on it.

**"What do the results look like?"**
*(The most likely question in the room, because the deck deliberately has no results slide.)*
A matched HSI‑vs‑RGB pair whose configs differ in **exactly one field** — `data_dir` — scored on
the same 348,894 test patches from the same five held‑out patients: **balanced accuracy 0.9073
vs 0.7571**, macro‑F1 **0.8580 vs 0.7282** **[verified]**. But two thirds of that test set is
IDC, which *both* models solve completely (F1 0.999), so the entire margin sits on the
healthy‑versus‑DCIS boundary — which happens to be the clinically consequential one. Single
seed, and the effective sample is **five patients**. A strong indication, not a significance
claim. `fig14_results.svg` draws exactly this and is ready to show.

**"How does it compare to MedMamba itself?"**
On the same test set: **0.446 M vs 3.65 M parameters**, balanced accuracy **0.9073 vs 0.8676**,
macro‑F1 **0.8580 vs 0.8278**, and **679 vs 155 seconds per epoch**. Say the caveat in the same
breath: **the comparison is indicative, not controlled.** MedMamba trained on a class‑balanced
build while ours drew from the natural 13:1 imbalance and compensated at the loss. Unmatched:
training‑split build, data fraction, class weighting, EMA, augmentation. Matched: test set,
patient split, batch size, precision, seed, lr, GPU, checkpoint rule.

**"Why does recursion beat just making the network deeper?"**
Nobody knows, and the paper says so outright: *"why recursion helps so much compared to using a larger and deeper network remains to be explained; we suspect it has to do with overfitting, but we have no theory to back this explanation."* Their evidence is circumstantial but consistent — 4 layers lose to 2, Mixture‑of‑Experts loses badly, and both failures look like overfitting on ~1000 examples. Do not offer a mechanism you can't defend.

**"If there's no fixed point, why does the `no_grad` prelude work at all?"**
Because deep supervision trains for it directly. Every pass is handed an arbitrary `(y, z)` and graded on improving it, so "run it twice more without gradients" is not an approximation of anything — it is the model doing the exact job it was trained for. That's the substantive difference from HRM, which needed convergence to a fixed point to justify its gradient. TRM needs nothing to converge.

**"Did they try actual fixed‑point iteration?"**
Yes — TorchDEQ, in their failed‑ideas section. It was slower *and* generalized worse. That's the strongest evidence that the fixed point was never what made HRM work.

**"Is TRM better than an LLM?"**
On these four puzzle benchmarks, mostly yes, at 0.01% of the parameters — and on Sudoku and Maze the LLMs score a flat 0.0. But Grok‑4 beats it on both ARC benchmarks. And TRM is supervised and deterministic: one input, one answer, no sampling. It is not a general‑purpose model and the paper never claims it is.

**"Does our version inherit TRM's results?"**
No, and this matters. Their numbers are on 9×9 and 30×30 puzzle grids with 1000× augmentation and a verifiable answer. Ours is patch classification on medical images. We inherited the *recursion schedule*, not the evidence. This is on slide 19.

**"Is the recursion actually helping, in your model?"**
We don't know. There's no depth ablation and no hierarchical‑versus‑recursive comparison under matched conditions. Both are one command‑line flag away and neither has been run. It's the first thing I'd do with a spare night. This is on slide 19.

**"Why a convolution instead of SS2D in the core?"**
Speed — see A5. About 39× slower at dataset scale. It's implemented and has never been used in a real run. This is on slide 19.

**"Could you just pad every input to 32 channels?"**
You could, and people do. You'd be feeding the model zeros it has to learn to ignore, you'd still have no representation of wavelength, and you'd be capped at whatever you padded to. Ours has no cap.

**"What's the biggest weakness?"**
Two. Everything is one seed. And the backbone averages away spatial detail inside each patch — fine at patch size one, and throwing away ninety‑eight percent of the pixels at patch size eight, which is what we use on the skin‑lesion dataset.

**"What's the reconstruction decoder for, if it's off?"**
See A7.

**"Is the 61× parameter saving real?"**
See A6. The structural claim is exact; the multiplier depends on which hierarchical width you
compare against, and the 27 M figure is unverified.
