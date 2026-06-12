# Measuring model quality: the other column

The first two docs in this chapter are about trusting a *speed* number. This one
is about the measurement those docs can't make: whether the model is still
*good*. The optimization phase of the roadmap (M15–M19) is built on levers —
quantization, KV-cache compression, speculative decoding — that buy speed by
touching the model's outputs, and a speed column without a quality column next
to it is half a result. This doc is the toolkit for that second column: what to
measure, what each metric catches and misses, and the discipline for reporting
it.

Prerequisites: [Benchmarking methodology](01_benchmarking.md) — this doc reuses
its fixed-environment discipline — and [Chapter 3](../03_numerical_types/README.md)
for what quantization does to a weight.
Next: M15, M19 and M22 in the [Roadmap](../../ROADMAP.md), whose reports carry the
quality column this doc defines.

## 1. Two different ways to be wrong

The benchmark harness already has a correctness gate: a kernel's output is
compared elementwise against a trusted reference (usually PyTorch) within a
tolerance. That catches **numerical error** — the kernel computes the wrong
values. But the M15-class optimizations are wrong in a different way: a
quantized matmul kernel can pass its elementwise tolerance perfectly — it
computes exactly what it's supposed to compute — while the *model built from it*
got measurably dumber, because the quantization itself (not the kernel) threw
away information.

```text
  numerical error      "the kernel computed the wrong numbers"
                       caught by:  elementwise compare vs reference, per kernel

  quality degradation  "the numbers are as intended, but the model is worse"
                       caught by:  nothing at the kernel level — only by
                                   evaluating the whole model's outputs
```

So quality needs its own instruments, all of which evaluate the model end to
end. There are three, in increasing cost: perplexity (§2), distribution
agreement (§3), task evals (§4).

## 2. Perplexity: the cheap, standard first check

**Perplexity** measures how well a model predicts a fixed text corpus. Run the
corpus through the model, record the probability it assigned to each actual
next token, and average the *negative log-likelihoods* (NLL — the `−log` of
the probability assigned to the correct token; low is good). Perplexity is `e`
raised to that average:

```text
  perplexity = exp( mean over tokens of  −ln p(correct token) )
```

A worked micro-example — three tokens, to which the model gave probabilities
0.5, 0.25, 0.125:

```text
  NLLs:  −ln 0.5 = 0.693    −ln 0.25 = 1.386    −ln 0.125 = 2.079
  mean = 1.386          perplexity = e^1.386 = 4.0
```

The intuition: perplexity 4 means the model was, on average, "as unsure as if
it were choosing uniformly among 4 tokens." Lower is better; an optimization
that hurts the model pushes it up.

What makes it useful: it's cheap (one forward pass over a corpus, no
generation), deterministic, and sensitive — small damage moves it. The
conventional corpus is **WikiText-2**, which makes numbers comparable across
papers. What to respect about it:

- **Only deltas mean anything here.** The absolute value depends on the
  tokenizer (a model with a larger vocabulary plays a harder prediction game),
  so cross-model comparisons are fraught. The quality column is always
  *perplexity before vs after the optimization, same model, same corpus*.
- **It misses behavior.** Perplexity tests next-token prediction on prose. A
  model can hold its perplexity while losing instruction-following, math, or
  long-context recall — failures that live in generated *sequences*, not
  per-token probabilities.

## 3. Distribution agreement: comparing the model to itself

The sharpest question for an optimization is not "is the model good?" but **"is
it still the same model?"** — and that can be measured directly, no labeled
data needed. Run the original (say fp16) model and the optimized one on the
same inputs and compare their output distributions position by position:

- **KL divergence** (Kullback–Leibler divergence): a standard measure of how
  different two probability distributions are. For the original's distribution
  `P` and the optimized model's `Q` over the vocabulary:

```text
  KL(P ‖ Q) = Σ over tokens v of   P(v) · ln( P(v) / Q(v) )

  = 0     when the distributions are identical
  grows   as Q drifts from P  (heavily punishing tokens P likes and Q doesn't)
```

  Mean per-position KL over a corpus is a single drift number, and it's more
  sensitive than perplexity: perplexity only looks at the probability of the
  *one* correct token, KL compares the *whole* distribution.

- **Token agreement rate**: greedy-decode both models from the same prompts and
  count how often they pick the same token. Crude but brutally interpretable —
  "the int4 model picks a different token 4% of the time" — and it previews how
  far generated outputs will wander (one different token early can diverge the
  rest of the sequence).

These are my loop metrics for M15-class work: cheap enough to run after every
change, no eval-set design decisions to argue about.

## 4. Task evals: the expensive final gate

The metric that actually answers "is it still good at things": run a benchmark
suite — academic-knowledge QA (MMLU-class), math, code — and compare scores.
The standard tool is EleutherAI's **lm-evaluation-harness**, which wraps
hundreds of such tasks behind one interface.

The catch is statistics. A task score is an accuracy over `N` questions, so it
carries sampling noise of roughly `1/√N`; on a 1,000-question task that's
~±1.5%, which is *larger* than the true damage a good int8 quantization does.
A 0.5-point drop on one task is noise, not signal. So: report `N`, prefer
suites over single tasks, and treat task evals as the **gate** (run once,
before calling an optimization done) rather than the loop metric — §3 is the
loop metric.

## 5. Which signal for which optimization

```text
  optimization           what to measure                          why
  weight quant (M15)     ppl delta + KL/agreement, task gate      lossy by design;
                                                                  measure the loss
  KV-cache quant         same, but on LONG-context inputs         damage concentrates
                                                                  where the cache is big
  speculative decoding   acceptance rate only                     with proper rejection
  (M19)                                                           sampling the output
                                                                  distribution is
                                                                  UNCHANGED — acceptance
                                                                  is a speed metric, not
                                                                  a quality metric
  relaxed speculation    KL + task gate                           the "lossless" claim
  (typical acceptance,                                            no longer holds; now
  aggressive tree drafts)                                         it must be measured
  sparse attention /     long-context retrieval tests             what's at risk is
  SWA-style variants     (needle-in-a-haystack class)             recall of distant
                                                                  tokens specifically
```

The speculative-decoding row is the one worth memorizing: standard speculative
sampling provably preserves the target model's distribution, so its quality
column is "lossless (by construction)" — but the moment a variant relaxes
verification to push acceptance up, it moves to the row below and owes real
measurements.

## 6. The discipline

The same rules as [the benchmarking doc](01_benchmarking.md), translated:

```text
  1. fix the eval set    same corpus / prompts / seeds, recorded in the
                         results JSON like any other environment field
  2. always paired       every quality number is a DELTA vs the unoptimized
                         baseline, measured by the same harness in the same run
  3. loop vs gate        ppl + KL after every change (minutes);
                         task suite once, before the report (hours)
  4. quality column      M15/M19/M22 report tables carry speed AND quality;
                         a speedup with an unmeasured quality cost is not
                         a result, it's a hope
```

## 7. Further reading

- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**
  (EleutherAI) — the standard task-eval tool of §4; the README's task list is
  also a good map of what "model quality" gets operationalized as.
- **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)**
  (Frantar et al., 2022) — the canonical weight-quantization paper, and a model
  of reporting quality the way §5 asks: perplexity deltas at every bit width.
- **[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)**
  (Lin et al., 2023) — the other standard quant method M15 will compare against,
  again argued almost entirely through quality-vs-bits tables.
- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**
  (Leviathan et al., 2022) — the proof behind §5's "lossless by construction"
  row: rejection sampling that exactly preserves the target distribution.

## 8. What to carry forward

```text
numerical error ≠ quality degradation (§1)   -> M15, why correctness.py isn't enough
perplexity deltas, fixed corpus (§2)          -> M15/M22, the cheap loop metric
KL + agreement vs the fp16 model (§3)         -> M15, the sharp loop metric
task evals as the noisy, final gate (§4)      -> M22, the capstone's quality axis
spec decoding is lossless; relaxed isn't (§5) -> M19, what to measure there
```

The one sentence to keep: **a kernel-level correctness check cannot see
model-level damage, so every optimization that touches outputs gets a quality
column — perplexity and KL deltas against the unoptimized model as the loop
metric, a task-eval suite as the final gate — or its speedup is unproven.**
