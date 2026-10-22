# Evaluating local LLMs on unified-memory hardware

A method and a toolkit for working out which local models are actually worth
running on your machine — and a record of what we found when we did it across
28 models on a 121 GB unified-memory box.

Most local-model comparisons measure accuracy and stop. That leaves out two
things that decide whether a model is usable day to day: **whether it refuses
work it should do**, and **whether an agent built on it tells you the truth
about its own failures**. Both are measurable, both surprised us, and both are
in here.

---

## What you get

- **[docs/DGX-Spark-Results.md](docs/DGX-Spark-Results.md)** — the full write-up. The whole study in the
  order it happened: the machine's binding constraint, how models were chosen,
  engine selection and configuration, tuning, what we measured and why, the
  measurement errors we found in our own instruments, conclusions, and quick
  starts for six use cases.
- **`evalkit`** — a small harness that points at any OpenAI-compatible endpoint,
  fetches public benchmarks, runs them, and produces a report you can defend.
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — how to measure without fooling
  yourself. The practices that mattered, and the specific mistakes they prevent.
- **[docs/FINDINGS.md](docs/FINDINGS.md)** — what we found. Portable conclusions,
  separated from the numbers that are specific to our hardware.
- **[docs/EVALUATING-A-NEW-MODEL.md](docs/EVALUATING-A-NEW-MODEL.md)** — the
  checklist for adding a model to your own fleet.

Nothing here depends on our infrastructure. There are no hostnames, no
orchestration, no vendor SDK. If it serves `/v1/chat/completions`, this works
against it.

---

## Quick start

```bash
pip install -r requirements.txt

export EVALKIT_BASE_URL=http://localhost:8000/v1   # vLLM, llama.cpp, LM Studio…
export EVALKIT_API_KEY=not-needed                  # many local servers ignore this

# 1. Prove the harness works BEFORE trusting anything it tells you.
python -m evalkit.cli selftest

# 2. Fetch a benchmark.
python -m evalkit.cli import xstest

# 3. Register your models.
cp models.example.yaml models.yaml && $EDITOR models.yaml

# 4. Run.
python -m evalkit.cli run --models my-model --suite xstest --run-id first

# 5. Report.
python -m evalkit.cli report --run-id first
```

Step 1 is not ceremony. It checks that the code sandbox can execute a *correct*
solution, that the refusal detector separates a refusal from an answer, and that
your endpoint answers at all. Each of those has silently produced a wrong number
for us at least once — a correct reference solution scored 0 because the sandbox
lacked numpy, and it looked exactly like a model failure.

### Available benchmarks

| Import | Cases | Measures |
|---|---|---|
| `gpqa` | 198 | Graduate-level reasoning (gated on HuggingFace) |
| `mmlu-pro` | 297 | Broad knowledge, 14 subjects |
| `humanevalplus` | 164 | Code that must execute against tests |
| `ifeval` | 388 | Verifiable instruction following |
| `xstest` | 250 | **Over-refusal** — benign prompts that look unsafe |

Datasets are fetched, not vendored: they carry their own licences and they
change. A stale copy in a repo is a silent correctness problem.

---

## The findings that changed what we run

Measured across 28 models on an NVIDIA DGX Spark (GB10, 121 GiB unified memory,
~273 GB/s). The numbers are ours; the shape of them is likely to hold on any
bandwidth-bound machine.

**1. Active parameters, not total, set decode speed.**
A 122B mixture-of-experts model with 10B active decoded at 22.9 tok/s. A dense
27B managed 14.7. The MoE is four times larger on disk and 56% faster, because
only the active experts move per token. "How big is it" is the wrong question.

**2. "Uncensored" branding predicts nothing.**
We added four models chosen specifically for permissiveness — abliterated,
uncensored, roleplay fine-tunes. None beat a stock model. Two stock builds
already measured **0.0%** over-refusal. The worst model in the entire study, at
**13.6%**, was an uncensored roleplay fine-tune that declined to explain a Judo
choke and how to beat one's wife at checkers. If you want a model that doesn't
refuse benign work, measure the stock models first.

**3. Safety tuning moves between minor versions.**
Qwen 3.6 over-refused at 8.0–9.2%; Qwen 3.5 at 2.8–3.6%; Qwen 3.8 reverted to
4.4%. Same vendor, same sizes, a 3× swing and most of the way back. Nothing in a
model card tells you this — you have to measure the version you intend to run.

**4. Agents fail confidently, and the confident failures look better.**
On six planted bugs, a leading model claimed success while the defect remained
in **3 of 15 runs**. Every false claim came with a tidy numbered verification
summary. The honest failure in the same cell took four times as long and said
plainly that it hadn't finished. Polish is not evidence.

**5. Quantisation is nearly free on behaviour, mildly costly on knowledge.**
Three matched FP8/NVFP4 pairs showed no measurable refusal difference (all
inside noise). Accuracy costs a few points, and the cost is larger on knowledge
benchmarks than on task-flavoured suites — so test quantisation on knowledge
tasks specifically.

**6. A benchmark lead does not transfer to your own work.** *(added 2026-09-15)*
A later candidate beat our default by **12.1pp on GPQA** and made parallel tool
calls the default cannot make at all. On twenty-one cases written from incidents
that actually happened on our own infrastructure it scored **0.480 against the
default's 0.827**, and produced no answer at all on 7 of the 21 — it looped until
its reasoning budget ran out, six times on the hardest cases in the set. We did
not promote it. Those private cases cost 90 minutes beside a 22-hour public
benchmark run, and reversed its verdict.

**7. Run the control on the same build, or your comparison measures the build.** *(added 2026-10-21)*
A post-training that claimed 25% fewer tokens cut the median by **4.5%** on our
work, so it failed. The bigger lesson came from its control. The same base model
we already run scored **0.715** on our private cases on a new engine build and
quantisation, against **0.840** in production: a larger gap than the one between
candidate and control. Compared with the stored production score, the candidate
looked like a loss. Against a control on the same build, it won every case that
differed. Separately, its multi-token-prediction draft head made generation
**2.0–2.6× faster** end to end.

**8. Fixing the defect doesn't make the model better.** *(added 2026-10-21)*
The candidate from finding 6 looped on 7 of 21 private cases. Re-run with a
reasoning cap that *forces* an ending, it answered all of them: **0 of 105**
unanswered across three settings. It then scored **0.840**, exactly the resident
27B's score and under the default's 0.873. Capped thinking cost 8× the tokens of
thinking off, for no gain. The loops explained the September loss; removing them
didn't create a win.

The full write-up — selection, engines, tuning, benchmarking, conclusions and
per-use-case quick starts — is in [docs/DGX-Spark-Results.md](docs/DGX-Spark-Results.md). The short
version is [docs/FINDINGS.md](docs/FINDINGS.md).

---

## The thing most benchmark reports get wrong

Our apparent winner scored **93.9%** on GPQA — above every frontier-class model
we tested, which should have been the first clue.

It produced a parseable answer on **82 of 198 questions**. The other 116 ran out
of reasoning budget mid-thought and scored nothing at all. The figure was 93.9%
*of the 41% it attempted*, and averaging over completed cases turned a
truncation bug into a first-place finish.

`evalkit` reports an **unscorable rate** for every model on every benchmark, and
asterisks anything above 20% out of the ranking. It is the single most important
column in the output, and it is the one most harnesses don't have.

---

## Licence

MIT for the code. The benchmark datasets carry their own licences and are
fetched from their canonical sources rather than redistributed here.
