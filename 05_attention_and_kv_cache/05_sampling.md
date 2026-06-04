# From logits to a token: sampling

[The LM head](03_embedding_and_lm_head.md) left us with *logits* — one raw,
unbounded score per vocabulary token, for the position we're generating. But the
decode loop ([end-to-end §6](../02_cuda_software_stack/02_end_to_end_inference.md#6-stage-3--decode-the-autoregressive-loop))
needs **one token ID** to feed back in. *Sampling* is the step that closes that
gap: scores in, one chosen token out. The end-to-end map drew this as "sample one
token" and `end-to-end §12` said "a sampling kernel picks one token ID from the
logits" — without ever saying how. This is how.

It's the last piece of the forward pass, and the one with actual knobs a user
turns — `temperature`, `top_k`, `top_p` are all this step. Same as the rest of the
chapter, it's the map, not the build: I work out what each knob does to the math
so the M3.5 sampling-kernel build starts from known behaviour.

Prerequisites: [the LM head](03_embedding_and_lm_head.md#3-the-lm-head-the-mirror-operation)
(where logits come from) and [attention §3](01_attention.md#3-the-computation-step-by-step)
(the softmax, reused here on a different input).
Next: M3.5 in the [Roadmap](../ROADMAP.md) — the sampling kernels (top-k, top-p,
temperature, repetition penalty).

## 1. The job, and why it's a GPU step

The logits are a vector as long as the vocabulary — 32,000 to 256,000 numbers
— sitting in VRAM. Sampling turns that vector into one integer token ID. It
runs **on the GPU**, right where the logits already are: shipping a 256k-long
vector back to the CPU every single token, just to pick one entry, would waste the
very bandwidth decode is starved for (`end-to-end §6`). So a *sampling kernel*
does it in place, per token, on the critical path of the loop.

The pipeline is three conceptual steps — turn scores into probabilities, optionally
reshape that distribution, then draw one token:

```text
  logits [vocab]  ──temperature──▶  ──softmax──▶  probabilities [vocab]
                                                        │
                                            ──truncate (top-k / top-p)──
                                                        │
                                                   draw one token  ──▶  token ID
```

## 2. Softmax: logits → a probability distribution

Logits aren't probabilities — they're unbounded scores. The *softmax* function
turns them into a proper distribution (all entries positive, summing to 1):

```text
  p_i  =  e^{z_i} / Σ_j e^{z_j}        z = logits,  p = probabilities
```

It's the same softmax attention uses on its scores (`attention §3`), just applied
to the logit vector instead. The bigger a logit, the bigger its share of the
probability mass, but every token keeps some non-zero probability. (In practice
the kernel subtracts the max logit before exponentiating — `e^{z_i − max z}` — so
the exponentials never overflow; that *numerically stable softmax* is an M3 topic.)

## 3. Temperature: sharpen or flatten the distribution

*Temperature* `T` divides the logits before the softmax:

```text
  p_i  =  e^{z_i / T} / Σ_j e^{z_j / T}
```

It controls how peaky the distribution is. A toy 3-token vocab with logits
`z = [2, 1, 0]`:

```text
  T = 1.0   p = [0.67, 0.24, 0.09]    the model's "natural" distribution
  T = 0.5   p = [0.87, 0.12, 0.02]    SHARPER  — top token dominates (more focused)
  T = 2.0   p = [0.51, 0.31, 0.19]    FLATTER  — long shot tokens get a chance
  T → 0     p = [1.00, 0.00, 0.00]    the limit: always the top token (greedy)
```

So low temperature makes the model confident and repetitive; high temperature
makes it diverse and risky; `T → 0` collapses to *greedy decoding* — just take the
single highest logit (an `argmax`), no randomness at all.

## 4. Truncation: top-k and top-p (nucleus)

Even after softmax, the long tail of the vocabulary holds tens of thousands of
very-low-probability tokens. Their probabilities are each tiny, but there are so
many that *collectively* they carry enough mass to occasionally get drawn — and a
single absurd token can derail the whole generation. *Truncation* cuts the tail
off before drawing. Two schemes, on the same `p = [0.67, 0.24, 0.09, …tail…]`:

```text
  top-k  (k = 2)   keep the k HIGHEST-probability tokens, drop the rest,
                   renormalize, then draw:
                   keep {0.67, 0.24} → renorm → {0.74, 0.26}

  top-p  (p = 0.9) keep the SMALLEST set whose cumulative probability ≥ p,
                   renormalize, then draw  (also called "nucleus sampling"):
                   cumsum 0.67, then 0.91 ≥ 0.9 → keep first 2 → {0.74, 0.26}
```

The difference is *fixed vs. adaptive count*. Top-k always keeps exactly `k`
tokens, whether the model is confident or not. Top-p keeps however many are needed
to reach mass `p` — few when the model is sure (the distribution is peaky), many
when it's uncertain (flat) — which is usually the better behaviour, and why
top-p is the common default. They're often combined: top-k then top-p then sample.

## 5. Repetition penalty, and the greedy/sampling choice

Two smaller knobs round it out. A *repetition penalty* divides (or subtracts from)
the logits of tokens that have **already appeared**, before the softmax, lowering
their odds of being picked again — a cheap guard against the model looping on the
same phrase. And the overarching switch is **greedy vs. sampling**: greedy
(`argmax`, or `T → 0`) is deterministic and reproducible — same prompt, same
output — and good for tasks with one right answer; sampling (with temperature and
truncation) adds the controlled randomness that makes open-ended generation feel
natural. Same model, same logits — only this last step differs.

## 6. Shapes and hardware

Everything here operates on the single `[vocab]`-long logit vector, per generated
token:

```text
  softmax          reduce (max, then Σ) + elementwise exp over [vocab]  -> FP lanes
  temperature      elementwise divide by T                              -> FP lanes
  top-k / top-p    partial sort / selection over [vocab], then renorm   -> FP lanes
  draw             one random pick from the kept distribution           -> FP lanes
```

It's reduction-and-elementwise work plus a selection (the M3 reduction/softmax
machinery, `attention §3`), so it runs on the FP lanes, not the Tensor Cores. The
cost is small next to a layer's matmuls — but the reduction is over a 32k–256k-long
vector, it happens **every** decode token, and it sits on the loop's critical
path, so a sloppy sampling kernel can still show up in token latency.

## 7. What to carry forward

```text
softmax = logits → probabilities (§2)              -> M3, the stable-softmax kernel
temperature sharpens/flattens; T→0 = greedy (§3)   -> M3.5, the sampling kernels
top-k (fixed) vs top-p (adaptive) truncation (§4)  -> M3.5
sampling is a per-token reduction over the vocab (§6) -> M3.5, on the decode path
```

The one sentence to keep: **sampling closes the loop — it turns the LM head's
vocab-long logit vector into the one token ID fed back into decode, by softmaxing
the (temperature-scaled) logits into a distribution, truncating its tail with
top-k or top-p, and drawing one token; greedy decoding is just the `T → 0` corner
where that draw becomes a plain argmax.**
