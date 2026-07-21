# What a 30M-Parameter Language Model Taught Me About Training LLMs

*I built a decoder-only transformer from scratch and trained it on TinyStories
using a single free GPU. The interesting part wasn't the model — it was
discovering that most of the training machinery people obsess over does nothing
at this scale, that an apparent "scaling floor" was a mirage, and that a
1.8M-parameter model quietly rediscovers the same circuits found in GPT.*

---

## TL;DR

I trained a family of small language models (1.8M–56.6M non-embedding params) from
scratch on [TinyStories](https://arxiv.org/abs/2305.07759), on Kaggle's free T4
GPU, with no high-level modelling libraries. Running controlled experiments, I
found:

- **Most "standard" training tricks are no-ops at this scale.** Warmup length and
  LayerNorm-vs-RMSNorm made no measurable difference to final loss. The one lever
  that mattered was **peak learning rate** — the textbook default was simply too
  conservative.
- **A clean scaling law that turned out to be a trap.** Loss followed a tidy
  power law with an apparent floor at ~1.16 — but doubling the compute budget on
  the largest model punched straight through it. The floor was
  *compute*-limited, not a data ceiling. **Lesson: don't read an irreducible
  floor off an undertrained sweep.**
- **A tiny model rediscovers big-model circuits.** Visualizing attention, the
  1.8M model had crisp *previous-token heads* and *attention sinks* — the same
  interpretable structures documented in GPT-scale models.
- **Specialization beats scale on-distribution.** Judged by an LLM on story
  quality, my 56.6M model beat GPT-2 (124M) at the children's-story task, because
  GPT-2 — though a stronger language model — can't stay in the genre.

Full code and lab notebook: **[github.com/rishipadhye/my-LLM](https://github.com/rishipadhye/my-LLM)**.

---

## Why build this

I wanted to actually *understand* transformers, not import one. So everything —
the BPE tokenizer, multi-head attention, the training loop with AMP and cosine
scheduling, checkpointing — is hand-written. The goal was never a state-of-the-art
model; it was to run **controlled experiments** on a system small enough to
iterate on in an hour, and to see how much of the received wisdom from
GPT-scale training actually transfers to the tiny regime.

The base model: a ~30M-parameter decoder-only transformer (512-wide, 8 layers, 8
heads, 8k-token vocab), trained on TinyStories to **val loss 1.401**. It writes
fluent, coherent little stories. That was the starting point, not the point.

## Finding 1 — Most training tricks do nothing here

I ran three ablations expecting to learn which knobs matter. The answer, three
times running, was "not this one":

| Ablation | Result |
| --- | --- |
| LayerNorm vs RMSNorm | Loss tie (1.601 vs 1.603); LayerNorm ~33% faster (fused kernel) |
| Warmup length (50 / 500 / 2000) | Null (1.590 / 1.599 / 1.601) — within noise |
| Warmup under stress (10× LR, no clipping) | *Still* null; neither run diverged |

The warmup result surprised me most. I removed gradient clipping and cranked the
LR 10× specifically to *make* warmup matter — and it still didn't. The reason is
the optimizer: **AdamW already normalizes each parameter's update by a running
estimate of its own gradient magnitude.** That adaptive step-sizing does the very
job warmup and clipping were invented for, so at this scale they're redundant. A
"null result," but a mechanistic one — and a reminder that a lot of training
folklore is really *"things that matter for plain SGD at billion-parameter
scale."*

## Finding 2 — Learning rate was the only real lever

After three ties, the LR sweep was the first experiment to actually show a difference:

| Peak LR | val loss |
| --- | --- |
| 3e-4 (the default I started with) | 1.601 |
| **1e-3** | **1.559** |
| 3e-3 | 1.557 |

Tripling the LR cut loss by 0.042 — ~20× the run-to-run noise floor — then
plateaued. My original 3e-4 baseline was just too timid. Promoting 1e-3 to the
full 80k-step run carried the gain all the way through (1.43 → **1.401**). The
takeaway I'll keep: **tune the learning rate before touching architecture.**

## Finding 3 — A scaling law, and the floor that wasn't

I trained a 5-point ladder (1.8M → 56.6M non-embedding params) at a fixed token
budget and fit loss against parameters. It's a clean, gently bending power law:

> **L ≈ 1.16 + 0.78 · N^−0.276**   (R² = 0.9975)

![Scaling curve](assets/scaling_loglog.png)

The bend toward ~1.16 looked exactly like an irreducible loss floor — the point
where more parameters stop helping. It's tempting to stop there and write
"TinyStories saturates at 1.16."

So I tested it. I retrained the largest model for **double the budget**. If 1.16
were a real data ceiling, more compute shouldn't help. Instead loss dropped from
1.415 to **1.325** — straight through the "floor," and past even the
fully-trained 30M reference. **The floor was an artifact of the training budget,
not the data.** The real lesson generalizes well beyond this project: *a scaling
curve fit to undertrained points will show you a floor that isn't there.* That
single control run changed the conclusion — which is exactly why I ran it.

## Finding 4 — A tiny model rediscovers GPT's circuits

Loss curves say how well a model does, not what it learned. I added a one-line
hook to capture attention weights and plotted them. Even the **smallest** model
(1.8M params, 4 layers) had learned recognizable, interpretable structure:

![A previous-token head](assets/attn_xs_l0_h0.png)

That bright sub-diagonal is a **previous-token head** — every token attending to
the one before it. Plotting all heads revealed a familiar taxonomy: previous-token
heads, attend-to-self heads, and **attention-sink heads** (nearly every token
dumping attention onto the first token) — the same circuits documented in
GPT-scale interpretability work, reinvented by a model trained only on children's
stories.

I also probed for **induction heads** (the copy/pattern-completion circuit) by
scoring every head on a repeated prompt. Both model sizes showed only a *weak,
above-chance* induction bias (top head ~2× chance) — no dedicated induction
circuit at either scale. That's an honest negative: TinyStories rarely rewards
long-range verbatim copying, so there's little pressure to build one.

## Finding 5 — Specialization beats scale, on-distribution

Finally: are the stories any *good*? I had every model continue 10 prompts, and
scored the continuations 1–5 on six dimensions with an LLM judge
(Llama-3.3-70B), instructed to grade *fitness for a young child's story*.

![Per-dimension judge scores](assets/eval_scores.png)

My 56.6M model (overall **4.02**) beat **GPT-2 (124M, overall 1.60)** — a general
model twice its size — at the children's-story task. GPT-2 collapses on the
*story* dimensions (coherence, staying on-prompt, plot) because it drifts
off-genre into police stations and inconsistent characters. A model specialized
for a narrow distribution beats a bigger general one *at that distribution*.

**An honest caveat**, because it matters: GPT-2's low grammar/fluency scores
almost certainly *understate* its real linguistic quality — my task-framed judge
let the genre-drift bleed into the language scores (a halo effect). The
defensible claim is narrower: GPT-2's weakness here is *task-fit*, not language.
Flagging where my own evaluation is soft is part of doing it honestly.

## What I'd do differently

- **Compute-optimal points.** The scaling ladder is near-converged, not
  Chinchilla-optimal — fine for showing the *shape*, not for a precise exponent.
- **A more rigorous eval.** One generation per prompt, ten prompts, a single
  judge. Multiple samples, more prompts, and multiple judges would tighten it.
- **A fairer prompt for GPT-2.** I fed GPT-2 the same bare story-starters my model
  saw, but GPT-2 was never told it was writing for young children — so some of its
  genre-drift is a prompting artifact, not a capability gap. A system/instruction
  prefix ("Write a simple children's story:") would give it a fair shot at staying
  on-genre and make the comparison cleaner.
- **A dedicated induction probe.** Averaging over many random repeated sequences
  would turn the weak induction signal into a firm result either way.

## Takeaways

The thread running through all of it: **at 30M parameters, most of what people
tune is noise, and the things that actually move the needle are unglamorous** —
the learning rate, the training budget, and the match between model and data. The
most valuable habit this project reinforced wasn't any single result; it was
running the control that could *disprove* the tidy story — the 80k diagnostic
that killed the "data floor," the caveat that keeps the eval honest.

*Built with PyTorch on a single Kaggle T4. Code, configs, and the full lab
notebook (every experiment, including the null results): [github.com/rishipadhye/my-LLM](https://github.com/rishipadhye/my-LLM).*
