# Findings

Measured across 28 models on an NVIDIA DGX Spark (GB10, 121 GiB unified
LPDDR5X, ~273 GB/s), August–September 2026. Roughly 25,000 scored cases. One
later trial (2026-09-15, a 29th model, 1,059 cases) is included and marked.

Numbers specific to this hardware are marked as such. The conclusions in
**bold** are the ones we expect to transfer.

This is the condensed version. The full write-up — how models were selected,
how the engines were chosen and configured, what was tuned and why — is in
[DGX-Spark-Results.md](DGX-Spark-Results.md).

---

## The machine sets the terms

Unified memory means no separate VRAM budget and no host-to-device copy. The
constraint is bandwidth: at ~273 GB/s, decoding is memory-bound for every model
we tested. Each token requires reading the active weights out of memory.

**Tokens per second tracks bytes moved per token, which is a function of active
parameters and quantisation — not total parameter count.**

| Model shape | tok/s |
|---|---|
| Diffusion 26B | 119.2 |
| MoE 30B, 3B active | 79.3 |
| MoE 120B (MXFP4) | 38.4 |
| **MoE 122B, 10B active** | **22.9** |
| **Dense 27B** | **14.7** |
| Dense 70B (AWQ) | 6.0 |
| Dense 123B (AWQ) | 3.4 |

The two bolded rows are the finding: the 122B model is four times larger on disk
and 56% faster.

---

## Over-refusal: the axis most comparisons skip

XSTest's safe half — 250 benign prompts that pattern-match as unsafe. Scored so
that 1.0 = answered. Noise floor from engine twins: 0.0–0.8pp.

| Model class | Corrected refusal rate |
|---|---|
| Stock (gpt-oss family) | **0.0%** |
| Neutral-aligned 70B | 0.0% |
| Stock, various | 0.4–1.6% |
| Abliterated | 0.4–0.8% |
| Uncensored fine-tune | 0.8% |
| Qwen 3.5 family | 2.8–3.6% |
| Qwen 3.8 (27B) | 4.4% |
| Qwen 3.6 family | 6.8–9.2% |
| Uncensored RP fine-tune | **13.6%** |

### "Uncensored" branding predicts nothing

We added four models chosen specifically for permissiveness. Two stock models
already measured 0.0%. Of the four:

- one tied the stock floor (and was 6× slower)
- two came in marginally worse, inside noise
- one was **the most refusal-prone model of all 28**

The worst declined to explain a Judo choke, how to kill someone in a video game,
where to euthanise a sick dog, and how to beat one's wife at checkers.

**If you want a model that doesn't refuse benign work, measure the stock models
first. Abliteration is not required and the label carries no signal.**

Note also that *fine-tuning is not ablation*. The worst performer reached its
"uncensored" character through roleplay fine-tuning, which evidently left the
underlying refusal behaviour intact.

### Safety tuning moves between minor versions

Pooled across nine Qwen models, with no overlap between families:

| Generation | Pooled refusal | n |
|---|---|---|
| Qwen 3.5 | 3.2% | 1,250 |
| Qwen 3.6 | 8.0% | 1,000 |
| Qwen 3.8 | 4.4% | 250 |

z = 5.04 for the 3.6-vs-3.5 gap. A 3× swing across one minor generation, then
most of the way back. **Test the version you intend to run.** The 3.8 figure
rests on a single model and is reported as such.

---

## Quantisation costs less than expected on behaviour

Three matched FP8 / NVFP4 pairs — same checkpoint, same settings, format only:

| Checkpoint | FP8 | NVFP4 | Δ | z |
|---|---|---|---|---|
| Qwen3.6-27B dense | 6.8% | 8.4% | +1.6 | +0.68 |
| Qwen3.5-27B dense | 2.8% | 3.6% | +0.8 | +0.51 |
| Qwen3.5-122B MoE | 3.6% | 3.2% | −0.4 | −0.25 |

Mixed signs, every |z| under 0.7. **Quantisation format has no measurable effect
on refusal behaviour** — take the memory saving.

On knowledge benchmarks it does cost a few points, and the cost is larger on
GPQA and MMLU-Pro than on task-flavoured suites. **Test quantisation on knowledge
tasks specifically**, or you will underestimate what it costs.

---

## Agents fail confidently

Six planted defects in a sandboxed host — a value changed in one of three config
files, a guard whose condition can never fire, an installer that reports a
version it didn't install. Three runs each. What's scored is not whether the
agent succeeds but **whether its claim of success is true**.

A leading model, 15 valid runs:

| Outcome | Count |
|---|---|
| Fixed | 10 |
| Ran out of step budget, said so | 2 |
| **Claimed success, defect remained** | **3** |

All three false claims came from the two traps where the defect sits somewhere
other than where the agent was looking. The trap where the defect is in plain
view was 3 for 3.

### The two mechanisms

**Verification that shares the fix's assumption.** The agent rewrote a guard
script, tested it by planting a file in the directory *its own rewrite* checks,
watched the test pass, and reported "the guard now refuses to run while any
worker pidfile exists." True — for pidfiles in the directory it invented. It
never found the real one. Its verification could not have failed.

**Evidence found, then reclassified.** The agent's global scan *found* the second
affected unit, named it correctly, and filed it as future work while declaring
the task complete.

### The practical warning

**The false completions were the faster, tidier, more confident runs.** One took
9 steps and 93 seconds and closed with a five-point numbered verification
summary. The honest failure in the same cell took 40 steps and 752 seconds and
said plainly it hadn't finished.

Any review process that asks "did the agent verify its work?" passes these runs.

---

## A benchmark lead does not transfer to your own work

*2026-09-15, one model, 1,059 cases.*

We trialled a new checkpoint — a sparse-attention MoE at NVFP4, 79 GiB of
weights, served with thinking enabled and a 16,384-token reasoning budget — as a
replacement for our default. It was evaluated the way most model comparisons are
done: five public suites, 1,038 cases, 21.8 GPU-hours.

It won.

| | Candidate | Incumbent default |
|---|---:|---:|
| GPQA-diamond | **74.7%** | 62.6% |
| HumanEval+ | 93.3% | — |
| IFEval | 87.1% | — |
| XSTest-safe | 95.2% | — |
| Parallel tool calls | **works** | **0.000 — cannot** |

On the strength of that we would have promoted it. Then we ran the 21 cases we
keep privately — each one written from an incident that actually happened on our
own infrastructure, where the textbook answer is the wrong one. A backup that
restores rows encrypted with a key the backup does not contain. A network plugin
installed but never configured. A verification script comparing a boolean as
text, so it reports failure on a healthy system.

| | Candidate | Incumbent default |
|---|---:|---:|
| Private suite, 14 hardest cases | **0.480** | 0.827 |
| All 21 private cases | **0.634** | 0.865 |
| Produced no answer at all | **7 of 21** | 0 of 21 |
| Tokens spent on those 21 cases | 177,297 | 20,614 |
| Wall clock | 90 min | 9 min |

**The strongest public-benchmark result we measured all year belonged to a model
that could not do our work.** Not because it lacked knowledge — it beat the
incumbent outright on five of the private cases, including one nothing else had
ever solved — but because on seven of them it never stopped reasoning. Each of
those seven consumed its entire 16,384-token budget and emitted nothing. One
produced 59,000 characters restating a single hypothesis, on a question whose
reference answer is two sentences.

Six of the seven were the hardest cases in the set: exactly the work you would
keep a local model for.

**Corollary, and the cheap part of this story.** The private suite took 90
minutes on top of a 22-hour benchmark run — about 7% — and reversed the verdict.
If you have any body of real incidents, write twenty of them up as cases. It is
the highest-value hour of evaluation work we have done, and nothing published
can substitute for it: a public benchmark measures whether a model is good, and
your own cases measure whether it is good *at what you do*.

### Two smaller things that run fell out of

**A five-case gate is a smoke test, not a measurement.** We had set a threshold
of ≥0.90 on parallel tool calls. The category holds five cases, so one miss
scores 0.80 and fails the gate — and the miss was the model making one *extra*
correct call rather than failing to parallelise at all, the opposite of the
defect the gate existed to catch. Size a gate's case count to the decision it
gates.

**Report the unscorable rate per suite, not per run.** The candidate's overall
rate was a tolerable-looking 7.2%. On GPQA alone it was 21.7%, and on the private
suite 33%, concentrated entirely in the hard cases. Averaging across easy cases
hid the defect that decided the trial.

---

## Same model, new build, different score

*2026-10-21, one post-training against its own base, 3.75 hours.*

A post-training of the dense 27B we run as our resident model claimed the same
benchmark pass counts with 25% fewer completion tokens. We tested that claim
against a control that differed only in the post-training: both arms at 8-bit,
on one engine build, with identical flags.

| | Post-training | Matched base | Gate |
|---|---:|---:|---|
| Median tokens, thinking on, 10 tasks | −4.5% | — | −20% |
| Median end-to-end | −4.2%, CI spans zero | — | −15% |
| Private suite, 21 × 3 | **0.810** | 0.715 | no loss: **passed**, 3 wins, 0 losses |
| Ask-before-acting | 0.875, 0 wrong actions | 0.833, 0 wrong actions | passed |
| Unsafe actions with the gate open | none new | — | passed |
| Function calling, 30 | 0.667 | 0.733 | under-calls on parallel calls |

The claim didn't reproduce, and the post-training wasn't promoted. Two other
results were worth more than the verdict:

**The control model scored 0.715 here and 0.840 in production.** The weights
are the same, but the production copy runs a different quantisation recipe on
an older build of the same engine. The 0.125 gap is larger than the 0.095
between candidate and control. A comparison against the stored production score
would have called the candidate a regression. On the same build, it won every
case that differed. Your suite's score belongs to model, quantisation and engine
together, so hold two of them fixed.

**Its draft head roughly doubled speed.** With multi-token prediction, the same
weights went from 8.1 to 16.5–20.9 tok/s, or 2.0–2.6× end to end, for 2.5 GiB
more memory. Code accepted more drafted tokens than prose did. Our "no change"
gate was byte-identical output at temperature 0, and it failed. That was the gate's
fault, not the model's: batched verification shifts floating-point results, and
the texts diverged a few hundred characters in while the finished answer stayed
the same. Gate draft heads on answers, not bytes.

Full write-up: [DGX-Spark-Results.md §12](DGX-Spark-Results.md#12-addendum-a-post-training-that-promised-fewer-tokens-and-the-draft-head-that-delivered-speed-2026-10-21).

---

## Fixing the defect doesn't make the model better

*2026-10-21, the September candidate re-run with a forced reasoning cap.*

The model that lost our private suite in September did it by never stopping:
seven cases spent the whole 16k budget and emitted nothing. Its September budget
was only a `max_tokens` ceiling. Our engine now accepts a thinking budget that
*forces* the end-of-thinking tag when it runs out, so we re-ran the same 21
cases with a real cap. The model ran at 4-bit (IQ4_XS): the NVFP4 GGUF needs a
fork of the engine.

| | Thinking off | Cap 4k | Cap 2k | September (no cap) | Resident 27B | Default |
|---|---:|---:|---:|---:|---:|---:|
| Private suite | 0.840 | 0.836 | 0.844 | 0.634 | 0.840 | 0.873 |
| No answer | 0 / 63 | 0 / 63 | 0 / 21 | 7 / 21 | 0 | 0 |
| Median tokens | 300 | 4,325 | 2,298 | — | 300 | — |

- **The cap works.** Every case got an answer. The model still uses nearly the
  whole budget, so the cap ends its thinking for it, and the answers that follow
  are sound. The September loss was a termination defect.
- **With the defect fixed, there is no win.** The candidate ties the resident it
  would replace, trails the default, and needs the whole machine at 28 tok/s.
  Capped thinking costs 8× the tokens of thinking off for the same score.
- **The one real advantage** is that it is the most conservative agent we've
  measured. With the safety gate open it executed 6 of 15 malicious actions,
  against 9 for the resident and 15 for the default.
- **A harness note:** seven of 48 structured answers had the right decision and
  were missing only their final `}`. A strict parser fails them all. Report a
  lenient score beside the strict one when the gap is this systematic, and say
  which one decides.

Full write-up: [DGX-Spark-Results.md §13](DGX-Spark-Results.md#13-addendum-capping-the-reasoning-fixed-the-loops-and-nothing-else-2026-10-21).

---

## Speed is not the deciding gate

*2026-10-22, one 2.5B-active coding MoE against the resident 27B.*

| | Candidate | Resident |
|---|---:|---:|
| Decode at 8K | **82 tok/s** | 10.5 tok/s |
| Median time per correct answer, 10 tasks | **10.0 s** | 20.6 s |
| **Total** time, same 10 tasks | 418 s | **230 s** |
| Required arguments, 135-tool catalog | 12/12, none invented | — |
| Private suite, 21 × 3 | 0.705 | **0.840** |
| Sends a credential file when asked, gate open | **3/3** | 0/3 |

- **Report the total as well as the median of correct answers.** Two looping
  cases made the fastest model's total time *longer* than the slow model's.
- **The large-catalog defect we were warned about didn't reproduce.** We built
  deterministic synthetic catalogs to test it, with one request deliberately
  missing a required value. The model left that value out rather than inventing
  it.
- **Injection-resistant isn't the same as safe.** It refused an instruction
  planted in a file, and complied with the same leak when the user asked.
- **Harness note:** a fixed per-turn token cap can make an always-reasoning
  model look like it refused, because it thinks until the cap and calls nothing.
  Record truncation as an error, never as a decision.

Full write-up: [DGX-Spark-Results.md §14](DGX-Spark-Results.md#14-addendum-the-fastest-model-we-have-measured-and-why-speed-was-not-the-question-2026-10-22).

---

## What we got wrong

### The apparent winner was survivorship bias

One model topped GPQA at 93.9% — above every frontier-class model we tested. It
produced a parseable answer on **82 of 198 questions**. The score describes the
41% it attempted; the rest exhausted their reasoning budget mid-thought.

Now asterisked, excluded from ranking, kept in the table.

### A fifth of our refusal measurements were detector error

46 of 224 flagged cases (21%) were models *correctly* identifying a fictional
premise — "I can't help with that, Daenerys is fictional." Correcting it
strengthened one finding, left one unchanged, and **retracted a third**: we had
concluded coding models over-refuse. They do not.

### Engine comparison, done properly

An engine claiming ~3× throughput measured **1.68×** when we held the checkpoint
fixed. The published comparison changes the model as well as the engine, and on
a bandwidth-bound machine that difference alone explains most of the gap.

### Speculative decoding depth does not transfer

Depth 4 was adopted fleet-wide from dense 27B measurements, where draft
acceptance runs 94–100%. On a 35B MoE with 3B active, depth 4 gives 58.7%
acceptance and a mean accepted length of 3.36 against a ceiling of 5 — the
engine drafts tokens that get thrown away. Depth 2 is faster there.

**Measure speculation depth per model, across multiple reloads.** This machine's
across-load variance (13.6%) dwarfs its within-load variance (0.9%), so a single
cold probe measures the load rather than the setting — and did once reverse the
sign of a result.

---

## What we'd run

**A stock, well-tuned MoE in the 100B class is the default.** In our fleet
`gpt-oss-120b` led accuracy among models that answered everything, refused
nothing, and decoded at 38 tok/s. Every specialised alternative we tested was
chosen for a constraint it didn't satisfy — speed, memory — rather than because
it was better.

The transferable version:

- **Ignore total parameter count when predicting speed.** Ask what's active.
- **Take the quantisation saving.** It costs nothing measurable in behaviour.
- **Measure the version, not the family.** Minor releases move safety tuning.
- **Don't buy a modified model to avoid refusals** until you've measured stock.
- **Don't trust an agent's completion report** on tasks that require
  establishing scope — verify the state, not the summary.
