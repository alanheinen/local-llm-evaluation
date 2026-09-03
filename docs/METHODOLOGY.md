# How to measure without fooling yourself

Every practice here exists because skipping it produced a wrong number we
believed for a while. They are ordered by how much damage the omission does.

---

## 1. Establish a noise floor before you conclude anything

Serve **the same checkpoint twice** under conditions that shouldn't matter — a
different engine build, a different day — and measure the difference. That
number is your resolution. Any gap smaller than it is not a finding.

Ours:

| Axis | Noise floor |
|---|---|
| Over-refusal (n=250) | 0.0–0.8 percentage points |
| Throughput, across reloads | 13.6% |
| Throughput, within one load | 0.9% |

The throughput pair matters enormously. Across-load variance is fifteen times
within-load variance, which means **a single cold probe measures the load, not
the setting.** We once reversed the sign of a speculative-decoding result that
way. Anything that requires a model reload between conditions needs multiple
reloads per condition.

For proportions there is also an analytic floor. At a 5% rate with n=250, the
standard error on a *difference between two models* is about 1.9 points:

```python
from evalkit.report import two_proportion_z
z = two_proportion_z(k1, n1, k2, n2)   # |z| < 2 is not a result
```

We believed NVFP4 refused more than FP8 on the strength of two pairs showing
+1.2 and +2.0 points. Both were under one standard error. The third pair came in
at exactly 0.0 and the effect evaporated. **Two positive results in a row is what
noise looks like when you stop early.**

---

## 2. Record what didn't score

A scorer must distinguish three outcomes, not two:

- **correct** — 1.0
- **incorrect** — 0.0
- **unscorable** — `None`, and never averaged

A model that reasoned for 4,000 tokens and ran out of budget before stating an
answer has not answered incorrectly. It has not answered. Averaging over the
cases it completed measures only the questions it found easy enough to finish.

This is not hypothetical. Our top-ranked model on GPQA left 59% of the benchmark
unscorable and its 93.9% described the remaining 41%. The bug was a reasoning
budget set too low; the *damage* was that nothing in the pipeline noticed.

`evalkit` flags any model above 20% unscorable, excludes it from ranking, and
keeps it in the table with an asterisk — deleting it would hide the finding.

---

## 3. Vary one thing

Obvious, and still the most common way published comparisons go wrong.

An engine we evaluated claims ~3× the throughput of the alternative. That claim
can't be checked from published figures, because every published comparison
*also changes the model*: the engine's headline number uses a 35B MoE with 3B
active while the baseline is a dense 27B. On a bandwidth-bound machine the
parameter difference alone explains most of a 3×.

Holding the checkpoint fixed and varying only the engine gave **1.68×**. Still a
real speedup. Not the headline.

If you cannot hold everything else fixed, you do not have a comparison — you
have two measurements.

---

## 4. Check that your success signals check something

This is the failure that recurs most, and the hardest to see, because a broken
verifier reports success.

Real examples from this work:

- A benchmark run selected **zero cases** because a flag took a comma-joined
  string as a single literal tag. It reported "0 cases", marked every model
  "already complete", and exited 0. A four-benchmark, 1,047-case phase silently
  did nothing.
- A config task reported `changed` while **creating no file at all** — a YAML
  folded scalar had split a command from its argument, and a trailing `echo`
  still succeeded.
- A model returning **pure backslashes** passed a health check that tested for a
  non-empty string.
- A restore path wrote a corrupted value through the same broken code that
  caused the corruption, then printed the correct value it had intended to
  write.

The pattern is one thing: **verification that shares an assumption with the
thing it verifies cannot fail.** Test the artifact, not the exit code. Read back
what you wrote. Assert on the *content* of a response, not its length.

---

## 5. Beware detectors that match words rather than meaning

Any regex-based classifier will have a blind spot, and the blind spot will be
systematic rather than random.

Our refusal detector matches refusal openers — "I'm sorry", "I can't help".
XSTest contains prompts about fictional characters, and a model answering *"I
can't help with that, Daenerys is fictional and has no real credit card"* is
**correct**. It got flagged as a refusal.

That was **46 of 224 flagged cases — 21%**. Correcting it:

| Finding | Raw | Corrected | Outcome |
|---|---|---|---|
| Generational refusal gap | +4.1pp, z=3.99 | +4.8pp, z=5.04 | Strengthened |
| Quantisation affects refusal | mixed | mixed | Still noise |
| Coding models over-refuse | 7.2% | 4.8% | **Retracted** |

We had published the third claim. It was a detector artifact.

Four separate detectors in this project fired on the presence of a word rather
than its meaning — the refusal matcher, an agent success-claim matcher that
counted *"I've done read-only recon so far, no changes yet"* as a completion
claim, an audit script that swallowed tool output into agent speech, and a
progress monitor that read `unreachable=0` in a success line as a failure.

**Report raw and corrected numbers both.** The correction is a heuristic too,
and its size tells the reader how much to trust it.

---

## 6. Store responses, not just scores

Results are append-only JSONL with the full response text. Every scorer
correction we made was applied retroactively to data already on disk, and one of
them changed a published conclusion. A pipeline that stores only the number
cannot do that — you would have to re-run, and on a machine where a sweep takes
hours you won't.

---

## 7. Tune your monitors to the work

A model download stopped writing bytes for **40 minutes** while its process
stayed alive and its orchestration reported nothing wrong. Retries fire on
failed tasks, not stalled ones. Our stall threshold was set at 30 minutes, so it
would have been caught eventually, badly.

Two rules:

- A monitor threshold should be a small multiple of the work's normal cadence,
  not a round number. Thirty minutes is not monitoring for a job that writes
  every few seconds.
- **A false alarm is worse than no alarm**, because it trains you to ignore the
  next one. Our benchmark watchdog fired at every model boundary — a legitimate
  cold load looked identical to a stall — until we made it suppress alerts when
  the case marker sat at the end of a model.

---

## 8. Cap reasoning budgets, then check what the cap did

Thinking models spend whatever you give them. One model configured with 24,576
tokens took 4.5 hours for a single benchmark pass. Capped at 4,096 it finished
in 61 minutes with no accuracy loss we could detect.

But the cap is what produced the 59%-unscorable disaster in §2. So: **cap the
budget, then read the unscorable rate.** If it is high, the cap is too low for
that model and its scores are not comparable with anything.
