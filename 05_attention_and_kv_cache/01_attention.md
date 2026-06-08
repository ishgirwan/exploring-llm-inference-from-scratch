# Attention and the KV cache

This is the doc for the operation the
[end-to-end inference walkthrough](../02_cuda_software_stack/02_end_to_end_inference.md)
kept pointing at and deferring: in its §5 layer graph, "attention" was one box
labelled *FlashAttention kernel → Tensor Cores + FP lanes*, and its §7 decode
loop said "this token's Q attends over all past K,V in the KV cache" without
ever saying what Q, K, V are or why the cache is shaped the way it is. This doc
fills in that box.

Like the end-to-end doc, it's a **map**, not the build. I work attention out on
paper here so that when I implement it for real — a toy attention path in M9–M10,
FlashAttention in M16, paged attention in M17 — I already know what the numbers
are supposed to be. Where another doc owns a concept, I link to it rather than
re-derive it.

Prerequisites:
[End to end: a prompt becomes tokens](../02_cuda_software_stack/02_end_to_end_inference.md)
(its §5 transformer-layer graph and §7 decode loop), and
[GPU execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)
(memory-bound vs compute-bound).
Next: M9–M10 in the [Roadmap](../ROADMAP.md) — where this gets built and measured.

## 1. What attention is for

A language model reads a sequence of tokens and, at each position, has to
produce a vector that captures not just *that* token but how it relates to the
others. "It" in a sentence means nothing until you know what it refers to. So
every layer needs a step where each token can **look at the other tokens and
pull in information from them**. That step is attention.

Concretely, attention takes one vector per token in and produces one vector per
token out, where each output vector is a *blend of information gathered from the
other tokens*, weighted by how relevant each one is:

```text
  in:   x1   x2   x3   ...   xN        (one vector per token)
                  |
            attention mixes across positions
                  |
  out:  y1   y2   y3   ...   yN        (each yi = a relevance-weighted
                                        blend of information from x1..xN)
```

The rest of the transformer layer (the norms, the MLP — see
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels))
works on each token independently. Attention is the *only* place tokens talk to
each other. Everything below is the mechanics of "look at the others and pull in
a weighted blend."

## 2. Q, K, V: three projections, one lookup

Attention is built as a *soft dictionary lookup*. A normal dictionary lookup is:
you have a *query*, you match it against *keys*, and you return the *value* of
the key that matches. Attention does the same thing, except "match" is a
similarity score and instead of returning one value it returns a weighted
average of *all* values, weighted by how well each key matched.

To do this, each token's input vector is turned into three different vectors by
three separate learned matrices (the *projections* from
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)):

```text
  query  Q = x · W_Q     "what am I looking for?"
  key    K = x · W_K     "what do I offer, as something to be matched against?"
  value  V = x · W_V     "what information do I hand over if I'm matched?"
```

A *projection* here just means multiplying the token vector by a learned weight
matrix to map it into a new space. `W_Q`, `W_K`, `W_V` are part of the model's
trained weights, living in VRAM like everything else
([end-to-end §3](../02_cuda_software_stack/02_end_to_end_inference.md#3-stage-0--loading-the-model-once-at-startup)).
The shapes, for a sequence of `N` tokens:

```text
  X is [N x d_model]          d_model = the model's hidden width (e.g. 4096)
  Q = X·W_Q  is [N x d_k]     d_k = the per-head query/key dimension
  K = X·W_K  is [N x d_k]
  V = X·W_V  is [N x d_v]      d_v = the per-head value dimension (usually = d_k)
```

These three matmuls are the `Q,K,V projection` box from
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
— ordinary GEMMs on Tensor Cores. With Q, K, V in hand, attention itself is the
five steps in §3.

## 3. The computation, step by step

Given Q, K, V, the output is computed in five steps. The whole thing is one
formula —

```text
  Attention(Q, K, V) = softmax( (Q·Kᵀ) / √d_k  + mask ) · V
```

— but it's much clearer unpacked. I'll note the hardware each step runs on as I
go, the same grounding
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
used for the layer.

**Step 1 — scores: how much each query matches each key.** Take the dot product
of every query with every key. The dot product is a similarity measure: large
when two vectors point the same way. `Q` is `[N × d_k]` and `Kᵀ` is `[d_k × N]`,
so the result is an `N × N` matrix — for every (query token, key token) pair,
one score.

```text
  scores = Q · Kᵀ        shape [N x N]      (a matmul -> Tensor Cores)

  scores[i, j] = how much token i's query matches token j's key
```

**Step 2 — scale by `1/√d_k`.** Divide every score by the square root of `d_k`.
The reason is statistical: a dot product of two `d_k`-dimensional vectors is a
sum of `d_k` products, so its variance grows with `d_k`. Left unscaled, large
`d_k` produces large scores, and large scores push softmax (step 4) into a
regime where one weight is ~1 and the rest ~0 — it stops blending and its
gradient vanishes. Dividing by `√d_k` keeps the scores' variance ~1 regardless
of `d_k`.

```text
  scores = scores / √d_k        (elementwise -> FP lanes)
```

**Step 3 — causal mask.** In text generation a token must not see tokens that
come *after* it (at training and prefill time the whole sequence is present, but
the model is only allowed to use the past). So before softmax we set every
"future" score to −∞, which softmax will turn into a weight of exactly 0. Token
`i` may attend to keys `1..i` and no further. Keep the **lower triangle**
(including the diagonal), mask the **upper triangle**:

```text
  causal mask for N = 4  (• = kept, x = set to −∞)

  query \ key   k1  k2  k3  k4
       q1       •   x   x   x      q1 sees only itself
       q2       •   •   x   x      q2 sees k1, k2
       q3       •   •   •   x      q3 sees k1, k2, k3
       q4       •   •   •   •      q4 sees everything up to itself
```

```text
  scores = scores + mask        (elementwise -> FP lanes)
```

**Step 4 — softmax, across the keys.** Turn each query's row of scores into a
set of weights that are positive and sum to 1. *Softmax* of a vector exponentiates
each entry and divides by the total:

```text
  softmax(z)_j = exp(z_j) / Σ_k exp(z_k)
```

The axis matters and is easy to get wrong: softmax runs **across keys, for a
fixed query** — i.e. along each *row* of the `N × N` matrix, so **each row sums
to 1**. Row `i` becomes "how token `i` distributes its attention over all the
keys it's allowed to see."

```text
  weights = softmax(scores, over the key axis)   shape [N x N], rows sum to 1
                                                 (exp + reduction -> FP lanes)
```

(The naive `exp`-then-divide above can overflow; the numerically stable version
subtracts the row max first. That's a reduction-kernel topic — built in M3 —
and not needed to understand the math here.)

**Step 5 — weighted sum of values.** Each output vector is the attention-weighted
average of *all* the value vectors. This is `weights · V`: `[N × N]` times
`[N × d_v]` gives `[N × d_v]`, one output vector per token.

```text
  output = weights · V        shape [N x d_v]    (a matmul -> Tensor Cores)

  output[i] = Σ_j  weights[i, j] · V[j]
            = the blend token i pulled in from everyone it attended to
```

The shape pipeline, end to end:

```text
  Q [N×d_k]  K [N×d_k]  V [N×d_v]
       \       /
        Q·Kᵀ            -> scores  [N×N]      Tensor Cores
          |  / √d_k                           FP lanes
          |  + causal mask                    FP lanes
          v
        softmax (per row, over keys)          FP lanes
          |
          v  · V
        output          [N×d_v]               Tensor Cores
```

So the two heavy matmuls (`Q·Kᵀ` and `weights·V`) hit the Tensor Cores; the
scale, mask, and softmax in the middle are elementwise/reduction work on the FP
lanes — exactly the split
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
drew for the whole layer, now seen inside the attention box.

## 4. A worked example, with actual numbers

Three tokens, `d_k = d_v = 2`, causal. Concrete vectors (these would be the
outputs of the Q/K/V projections; I'm just picking values to compute with):

```text
  keys                 values
  k1 = [1, 0]          v1 = [10,  0]
  k2 = [0, 1]          v2 = [ 0, 10]
  k3 = [1, 1]          v3 = [ 5,  5]

  queries
  q1 = [1, 0]   q2 = [0, 1]   q3 = [1, 0]
```

**Step 1, scores** `= qi · kj`:

```text
            k1   k2   k3
  q1        1    0    1
  q2        0    1    1
  q3        1    0    1
```

**Steps 2–3, scale by 1/√2 ≈ 0.707 and apply the causal mask** (upper triangle
→ −∞):

```text
            k1      k2      k3
  q1       0.707    −∞      −∞
  q2        0      0.707    −∞
  q3       0.707    0      0.707
```

**Step 4, softmax along each row** (over the keys; each row sums to 1). Walk the
last row, `q3 = [0.707, 0, 0.707]`, in full:

```text
  exp(0.707) = 2.028     exp(0) = 1.000     exp(0.707) = 2.028
  sum = 2.028 + 1.000 + 2.028 = 5.056
  weights(q3) = [2.028, 1.000, 2.028] / 5.056 = [0.401, 0.198, 0.401]   (sums to 1 ✓)
```

The full weight matrix (row 1 has only one allowed key, so weight 1; row 2 splits
over two):

```text
            k1      k2      k3
  q1       1.000    0       0
  q2       0.330   0.670    0
  q3       0.401   0.198   0.401
```

**Step 5, weighted sum of values** for `q3`:

```text
  output(q3) = 0.401·[10,0] + 0.198·[0,10] + 0.401·[5,5]
             = [4.01, 0] + [0, 1.98] + [2.005, 2.005]
             = [6.015, 3.985]   ≈ [6.0, 4.0]
```

Read that out: token 3 attended ~40% to token 1, ~20% to token 2, ~40% to
itself, and its output is that blend of their values. **That last row — one
query attending over all keys — is exactly one decode step** (§6): in decode you
have only the newest token's query and you score it against every stored key.
Hold onto that; it's why the KV cache exists.

## 5. Multi-head attention

One attention computation can only express one kind of relationship at a time
(say, "match the subject"). Real models run several attentions in parallel, each
with its own `W_Q`, `W_K`, `W_V`, so different *heads* can specialise — one
tracks syntax, another tracks the referent of a pronoun, and so on.

The model's hidden width `d_model` is split across `h` heads, each of width
`head_dim = d_model / h` (e.g. 4096 split into 32 heads of 128). Each head runs
the full §3 computation on its slice; the per-head outputs are concatenated back
to width `d_model` and passed through one more learned matrix `W_O`:

```text
  X [N×d_model]
     | split into h heads, each gets its own W_Q,W_K,W_V
     v
  head 1: attention -> [N×head_dim]   \
  head 2: attention -> [N×head_dim]    |  run in parallel,
   ...                                 |  independent
  head h: attention -> [N×head_dim]   /
     | concatenate -> [N×d_model]
     v
  output = concat · W_O   [N×d_model]
```

That final `· W_O` is the **`output proj` box from
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)**
— the same step, named from the attention side here and from the layer-graph
side there. Heads are independent, which is part of why attention parallelises
so well across the GPU's many lanes.

## 6. Prefill vs decode: the same math, two very different shapes

[End-to-end §5 and §7](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
introduced prefill (process the whole prompt at once) and decode (one token at a
time). Attention is where the difference bites hardest, because the *shape* of
the computation changes completely.

```text
  PREFILL — all N prompt tokens at once
     Q is [N × d_k], K is [N × d_k]
     Q·Kᵀ is the full [N × N] matrix      <- the whole triangle, computed once
     -> matrix × matrix, high arithmetic intensity -> COMPUTE-bound

  DECODE — one new token, t tokens already generated
     Q is [1 × d_k]   (only the new token's query)
     K is [t × d_k]   (ALL past keys)
     Q·Kᵀ is [1 × t]                      <- a single row, the "q3 row" of §4
     -> matrix × vector, low arithmetic intensity -> MEMORY-bound (§8)
```

Two things follow, and they're the whole reason for the rest of this doc:

1. In decode, the query is just the one new token, but the keys and values are
   *every token so far*. If we recomputed K and V for the whole history on every
   step, decode would get quadratically slower. We don't — we **cache** them
   (§7).
2. The full `N × N` matrix only ever exists in prefill, and even there it's
   wasteful to write all of it to memory. That's the opening FlashAttention
   exploits (§9).

## 7. The KV cache: what it stores, and why it's shaped that way

The KV cache is the running store of every past token's **K**ey and **V**alue
vectors, kept in VRAM so that each new decode step can attend over the history
without recomputing it
([end-to-end §7](../02_cuda_software_stack/02_end_to_end_inference.md#7-stage-3--decode-the-autoregressive-loop)
introduced it by name). §6 above is the justification: decode needs all past K
and V, but they don't change once computed, so compute each token's K and V
once and keep them.

**Why K and V but not Q.** This is the question that makes the layout make sense.
A past token's *query* was only ever needed to produce that token's own output,
back when it was the current token — and that output is already done. Future
tokens never look at a past token's query. They look at its **key** (to score
against) and its **value** (to blend in). So Q is recomputed fresh for the one
current token each step and immediately discarded; K and V are the only things
worth keeping.

```text
  per decode step for the new token:
     q_new, k_new, v_new = project(x_new)      (compute all three)
     append k_new, v_new to the cache          (keep K, V)
     attention = softmax( q_new · K_cacheᵀ / √d_k ) · V_cache
     (q_new is used once, here, then thrown away)
```

**The layout.** The cache has to be indexed by *which layer*, *which head*,
*which past position* — because every layer and every head has its own separate
K and V. So conceptually it's a tensor with these axes (per sequence):

```text
  KV cache  ≈  [ n_layers, 2, n_kv_heads, seq_len, head_dim ]
                              ^
                              the 2 is K and V

  - n_layers   one independent cache per transformer layer
  - 2          K and V stored side by side
  - n_kv_heads number of key/value heads (see GQA below)
  - seq_len    grows by 1 every decode step  <- the only axis that changes
  - head_dim   the per-head width from §5
```

Every decode step writes one new slice along the `seq_len` axis (the append
above) and reads the whole thing back for the score-and-blend.

**Why the size matters.** Multiply the axes out, times the bytes per element and
the batch size:

```text
  KV bytes = 2 · n_layers · n_kv_heads · head_dim · seq_len · batch · dtype_bytes
```

For a 70B-class model (80 layers, head_dim 128, fp16 = 2 bytes) with 8 KV heads:

```text
  per token  = 2 · 80 · 8 · 128 · 2  ≈ 320 KB
  at 4,096 tokens, one sequence      ≈ 1.3 GB
  at 32,768 tokens, one sequence     ≈ 10.7 GB
```

That is *per sequence*, on top of the weights, and it grows with context length.
It's the largest dynamic structure in inference and the main reason long context
and large batches run out of VRAM.

**GQA — the shrink lever.** Notice `n_kv_heads` in the formula, not `n_heads`.
Modern models use *grouped-query attention* (GQA): many query heads **share** one
K/V head (e.g. 64 query heads but only 8 KV heads), which cuts the cache by that
ratio — 8× here — for almost no quality loss. Plain multi-head attention has
`n_kv_heads = n_heads`; the extreme, one shared KV head, is *multi-query
attention* (MQA). This is purely a cache-size decision, which is why the formula
above is written in `n_kv_heads`.

**Contiguous vs paged.** The simplest cache is one contiguous block per sequence,
sized to the max length — easy, but it wastes VRAM (sequences that finish early,
or never reach max length, leave holes) and fragments as requests come and go.
Real engines store the cache in fixed-size blocks instead, like operating-system
memory pages — vLLM's **PagedAttention**
([end-to-end §10](../02_cuda_software_stack/02_end_to_end_inference.md#10-the-serving-engine-who-drives-the-loop)).
The toy version of that is M17; the layout above is the contiguous mental model
to start from.

## 8. Why decode attention is memory-bound — and it's a *different* wall

[End-to-end §7](../02_cuda_software_stack/02_end_to_end_inference.md#7-stage-3--decode-the-autoregressive-loop)
and
[execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)
already established that decode is memory-bound at batch size 1. But "memory-bound"
hides two *different* bottlenecks, and attention is where the second one lives:

```text
  the projections + MLP   are memory-bound on the WEIGHTS
       bytes read per step ≈ model size (e.g. 140 GB), FIXED — independent of
       how long the context is, and SHAREABLE across a batch (read the weight
       once, use it for every sequence in the batch — end-to-end §10)

  attention decode        is memory-bound on the KV CACHE
       bytes read per step = the whole cache (the §7 formula), which GROWS with
       context length, and is NOT shareable — every sequence has its own cache
```

That distinction is the mechanism behind a fact people observe but rarely
explain: **long-context decode gets slower per token.** The weight traffic is a
fixed cost; the KV-cache traffic climbs with every token already in the context.
And because batching amortizes the weight read but *not* the per-sequence KV read
(`batch · seq_len` in the formula has no divisor), in real batched serving the KV
cache becomes the dominant memory traffic at long context — the point where
inference stops being weight-bandwidth-limited and becomes KV-bandwidth-limited.
(Batching itself — the lever that amortizes the weight read across sequences — is
[Chapter 6](../06_batching/01_batching.md).)

Each decode step still does only ~one multiply-accumulate of math per KV byte it
reads (arithmetic intensity ~1, the same low ratio as the weight matmuls in
[execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)),
so the Tensor Cores sit idle waiting for the cache to stream in. Shrinking that
traffic — GQA (§7), quantizing the cache, paging it — is a whole class of
optimizations precisely because it's the wall that moves.

## 9. The N×N matrix, and what FlashAttention does about it

Back to prefill, where the full `N × N` score matrix does exist (§6). Its size is
the problem: for a sequence of `N` tokens it has `N²` entries *per head, per
layer*. At `N = 8,192` that's ~67 million numbers in a single head's matrix.
Writing that to VRAM and reading it back for the softmax and the `·V` step is a
huge amount of memory traffic — and the matrix is a throwaway intermediate, never
needed again.

```text
  naive attention:
     Q·Kᵀ  -> write the whole [N×N] matrix to VRAM
     read it back -> softmax -> write back
     read it back -> · V -> output
     (the N×N matrix crosses the slow VRAM link three times)
```

**FlashAttention** avoids ever materializing the full matrix. It tiles Q, K, V
into blocks small enough to fit in on-chip shared memory
([execution model §10](../01_hardware_fundamentals/03_gpu_model.md#10-shared-memory-holds-reusable-tiles)),
computes attention block by block, and keeps a running softmax (a running max and
running sum) so partial results from each block can be combined correctly without
the whole row ever existing at once. The `N × N` matrix never touches VRAM; only
the `[N × d]` inputs and output do. That turns attention's memory cost from
`O(N²)` down to `O(N)` and is why it's described as *IO-aware*.

That's the map-altitude version. The actual online-softmax recurrence and the
tiling schedule are what I build in **M16** — this section is just here so the
word "FlashAttention" in
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
now points at a real idea: *don't write the big matrix down.*

## 10. What to carry forward

```text
the Q·Kᵀ → scale → mask → softmax → ·V steps (§3)   -> M9, built for real
multi-head + output projection (§5)                  -> M9
the KV cache layout and append (§7)                  -> M10, the toy KV path
prefill vs decode attention shapes (§6)              -> M11, benchmarked
GQA / cache-size levers (§7)                          -> M11, M15 (quantized cache)
FlashAttention: don't materialize N×N (§9)           -> M16
paged / block-table KV cache (§7)                    -> M17
```

The one sentence to keep: **attention lets each token pull in a softmax-weighted
blend of the values of the tokens it's allowed to see; in decode that means one
new query scored against every cached key, which is why the KV cache exists and
why long-context decode is bound by how fast that cache streams out of VRAM.**
