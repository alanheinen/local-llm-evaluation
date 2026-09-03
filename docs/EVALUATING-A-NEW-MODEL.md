# Evaluating a new model

The checklist we arrived at after adding — and in several cases removing —
models from a working fleet. Each step exists because skipping it cost us
somewhere between an hour and a day.

---

## Before you download

**Check the quantisation format is one your stack can load.** Read
`config.json` from the model repo, not the model card:

```bash
curl -s https://huggingface.co/ORG/MODEL/raw/main/config.json | \
  python3 -c "import json,sys; c=json.load(sys.stdin); \
    print('quant:', (c.get('quantization_config') or {}).get('quant_method')); \
    print('max_pos:', c.get('max_position_embeddings'))"
```

Two things to catch here:

- **A `quantization_config` with no `quant_method`** is usually an MLX (Apple
  Silicon) build. It will not load on CUDA. We nearly downloaded 77 GB of one.
  The repo's `library_name: mlx` and its `mlx` tag are the giveaway.
- **`max_position_embeddings`** is the model's real context limit. Never copy a
  context length from another model's config — a Llama-3 checkpoint has 8192,
  and a preset copied from a Llama-3.1 model at 32768 simply refuses to start.

**Check the size against your memory budget**, remembering you need room for the
KV cache, not just the weights.

---

## After it loads, before you benchmark it

**Ask it something and read the answer.** Not "did the endpoint return 200", not
"was the string non-empty" — read it.

A model in our fleet returned a run of backslashes to every chat request while
the raw completion endpoint answered correctly. It passed a health check that
tested for a non-empty response. Had we trusted that, it would have contributed
250 cases of garbage scored as refusals.

If output is degenerate, the usual causes in order of likelihood:

1. **A missing BOS token.** Some checkpoints ship a chat template that omits
   `bos_token`. Servers render chat templates with `add_special_tokens=False` —
   correct, so templates carrying BOS aren't double-counted — and the model then
   receives none at all. Diagnose by tokenizing both paths and comparing:

   ```bash
   curl -s $BASE/tokenize -d '{"model":"M","messages":[{"role":"user","content":"hi"}]}'
   curl -s $BASE/tokenize -d '{"model":"M","prompt":"hi"}'
   ```

   If the chat path is missing a leading BOS id that the completion path has,
   that's it. Fix with a corrected chat template.

2. **A mismatched tool-call parser.** A parser attached to a model that doesn't
   emit that format can corrupt every response. Match the parser to the model's
   actual output format, verified by looking at its chat template — not to its
   family name.

3. **A broken quantisation.** Rarer than the two above. Test with the raw
   completion endpoint: if that is coherent and chat is not, the weights are
   fine and the problem is in the chat path.

---

## Run the cheap measurement first

Order your benchmarks so the one you care about most runs first. If the run has
to be stopped — and long runs do get stopped — you want the result you were
actually after.

For us that meant over-refusal before capability: it is 250 short prompts
against thousands of long ones, and it measures the property that most often
decides whether a model is usable.

```bash
python -m evalkit.cli run --models new-model --suite xstest --run-id new-model-refusal
python -m evalkit.cli run --models new-model --suite gpqa mmlu-pro humanevalplus ifeval
```

---

## Read the unscorable rate before the score

```bash
python -m evalkit.cli report --run-id new-model-refusal
```

If the unscorable column is above ~20%, **the score is not usable** and is not
comparable with models that answered everything. Raise `max_tokens` for that
model in `models.yaml` and re-run.

This is the single check that would have saved us from ranking a model first
that had answered 41% of the questions.

---

## Compare against your existing fleet, not against published numbers

Published benchmark figures are measured on different hardware, different
quantisations, different prompts and often different subsets. They are not
comparable with yours.

The only comparison that means anything is the one you ran yourself, through the
same harness, on the same cases. That is the whole reason to keep a standing
fleet measurement rather than a collection of screenshots.

---

## Decide on the axis you added it for

Be explicit about *why* you are adding a model, and judge it on that.

We added four models for a single property — being permissive. They were not
better at anything else and were not expected to be. When the refusal numbers
came in showing they were no better than stock models on the property they were
chosen for, that settled it, and the capability run was stopped rather than
finished for completeness.

**Keep the measurements when you remove the model.** "This category of model
doesn't deliver what it claims" is a finding, and it is the return on having
spent the time.

---

## A worked example

The four permissive models we added, measured, and removed:

| Model | Class | Over-refusal | Outcome |
|---|---|---|---|
| hermes-3-70b | neutral-aligned | 0.0% | Met the goal — but tied a stock model that runs 6× faster |
| dolphin-2.9-70b | abliterated | 0.8% | Worse than stock, and 27 points behind on GPQA |
| wizardlm-2-8x22b | uncensored | 0.8% | Worse than stock |
| magnum-v4-123b | uncensored RP fine-tune | 13.6% | **Worst of 28 models measured** |

The stock baseline they were meant to beat: **0.0%**.

All four were removed. The finding — that the label predicts nothing — is the
part worth keeping.
