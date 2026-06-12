# Attention variants: when the KV cache stops growing

[The attention doc](01_attention.md) ends on two facts that drive most of this
repo's optimization story: the KV cache grows linearly with context, and decode
reads all of it every step. Those facts hold for **full attention** — every
token attending over the entire history — and that's what most models do. But a
growing family of architectures bends one or both facts, and several of them now
ship in models a serving engine has to handle. This section is the map of those
variants, organized by the verb each applies to the cache: *shrink* it, *cap*
it, *replace* it, or *read it selectively*.

This is a survey section: each variant gets opened one level, enough to see what
it does to the cache and the decode wall, then re-sealed. The dense, full-attention
model from the rest of the chapter stays the default mental model.

Prerequisites: [Attention and the KV cache](01_attention.md), especially its KV-cache
layout and memory-bound-decode sections.
Next: [Chapter 6 — Batching](../06_batching/README.md).

## 1. The baseline, and the four verbs

What full attention costs, per sequence, at context length `L`:

```text
  cache SIZE   grows ∝ L      every token's K and V, kept forever
  decode READ  grows ∝ L      every step re-reads the whole history
```

The variants, by what they do to that:

```text
  SHRINK   GQA / MQA / MLA      fewer or compressed KV entries per token
                                (still ∝ L, smaller constant)
  CAP      sliding-window       only the last W tokens kept per layer
                                (size and read become constant, ≤ W)
  REPLACE  linear attention /   no cache at all — a fixed-size running
           SSM hybrids          state per layer, updated each token
  SELECT   sparse attention     full cache kept, but each step reads
                                only a chosen subset of it
```

**Shrink** is already covered: GQA/MQA in [the attention
doc](01_attention.md), MLA — which stores one compressed latent vector per
token (512-wide in DeepSeek-V3) instead of per-head K and V — in [Chapter 8's
decode doc](../08_optimizing_inference/01_improving_decode.md). The other three
verbs are this section.

## 2. Cap: sliding-window attention

**Sliding-window attention (SWA)** limits each token to attending over only the
last `W` tokens instead of the full history. Mistral 7B used `W = 4096`. The
cache consequence is the interesting part: entries older than `W` are never
read again, so the cache becomes a **ring buffer** — a fixed `W`-slot array
where the newest entry overwrites the oldest. Size and per-step read both stop
growing at `W`.

The catch is recall: a pure-SWA model literally cannot look at a token more
than `W` back (information can still propagate further through stacked layers,
but directly attending to it is gone). So real models interleave: **Gemma 3
runs five local layers (window 1024) for every one global full-attention
layer.** The global layers preserve long-range recall; the local layers make
five-sixths of the cache constant-size:

```text
  Gemma 3, per repeating block of 6 layers, context L >> 1024:

  layer  1-5   local,  window 1024   cache: 5 × 1024 slots   (constant)
  layer    6   global, full          cache: 1 × L            (grows)

  -> most of the KV memory problem confined to every sixth layer
```

The engine-side consequence: the KV cache is no longer one homogeneous tensor —
different layers have different shapes and different eviction rules, and the
cache manager has to track both kinds.

## 3. Replace: linear attention and SSM hybrids

The more radical move deletes the cache entirely. **Linear attention** and
**state-space models (SSMs)** process the sequence recurrently: each layer keeps
a **fixed-size state** (a matrix of constant shape, independent of `L`), updates
it as each token arrives, and answers queries from the state instead of from a
stored history:

```text
  full attention   decode step reads:  everything so far     (∝ L)
  linear / SSM     decode step reads:  one fixed-size state  (∝ 1)
```

Mamba is the best-known SSM: its *selective* state update lets the model choose
per token what to write into and forget from the state — which is what made the
approach competitive with attention. The structural trade is permanent: a
fixed-size state is a lossy summary, so exact recall of an arbitrary distant
token (verbatim quoting, needle-in-a-haystack lookups) is where these models
historically give ground.

Which is why what actually ships is **hybrids** — mostly-linear stacks with a
few full-attention layers retained as the recall anchor, the same interleaving
logic as Gemma 3's, one level more aggressive:

```text
  Jamba (AI21)    Mamba layers + periodic full-attention layers (+ MoE)
  Qwen3-Next      3 linear-attention layers (Gated DeltaNet, a gated
                  linear-attention design) : 1 full-attention layer;
                  an 80B-total / 3B-active MoE on top
```

For inference the appeal is exactly the two facts of §1: a hybrid's memory and
per-step read are dominated by constant-size states, so very long contexts
(Qwen3-Next ships 262K native) stop being a VRAM and bandwidth catastrophe.

## 4. Select: sparse attention

The newest verb keeps the full cache in VRAM but stops *reading* all of it.
**DeepSeek Sparse Attention (DSA)**, shipped in DeepSeek-V3.2, attaches a
lightweight **indexer** that scores the history per query and selects the
**top-2048** tokens; full attention then runs over only those:

```text
  every step:   indexer skims all L entries cheaply -> picks top-k (k = 2048)
                attention reads k entries, not L

  cache size:   still ∝ L   (nothing is discarded — any token MIGHT be picked)
  decode read:  ∝ k         (constant once L > k)
```

Compared to §2 and §3, selection gives up nothing in *reachability* — any
distant token can still be attended to if the indexer ranks it — at the cost of
the indexer's own (much cheaper) pass and the gamble that its ranking is good.
The bandwidth win lands precisely on the wall [the attention
doc](01_attention.md) identified: long-context decode stops scaling with `L`.

## 5. What this means for the rest of the repo — and the re-seal

The serving-engine view, one row per verb:

```text
  verb      cache size      decode read     engine must handle
  shrink    ∝ L, smaller    ∝ L, smaller    nothing new (same layout, smaller)
  cap       constant ≤ W    constant ≤ W    ring buffers; per-layer cache shapes
  replace   constant        constant        state tensors, not KV pages, for
                                            most layers; hybrid allocators
  select    ∝ L             ∝ k constant    indexer pass + gather-style reads
```

Re-sealing the box: this repo's builds (M10's KV cache, M16's FlashAttention,
M17's paged attention) all assume full attention with GQA — the right default,
since it's what the bulk of served models do and what every variant is defined
*against*. What this section changes is the reading of model configs: when an
engine's release notes mention hybrid allocators, sliding-window support, or
indexer kernels, those are these four verbs showing up in code. The KV-cache
formula from [the attention doc](01_attention.md) isn't wrong — it's the
full-attention row of a table that now has four more rows.

## 6. Further reading

One primary source per verb:

- **[Mistral 7B](https://arxiv.org/abs/2310.06825)** (Jiang et al., 2023) — the
  paper that mainstreamed sliding-window attention and the ring-buffer ("rolling
  buffer") cache of §2.
- **[Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)** (Gemma team,
  2025) — the 5:1 local:global design of §2, with ablations showing how little
  quality the capped layers cost.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)**
  (Gu & Dao, 2023) — the selective SSM of §3; section 2 is a self-contained
  intro to state-space layers.
- **[Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)**
  (AI21, 2024) — the hybrid argument of §3 made explicit, including why a few
  attention layers are kept.
- **[DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)**
  (DeepSeek-AI, 2025) — DSA, the production sparse attention of §4.

## 7. What to carry forward

```text
the four verbs: shrink/cap/replace/select (§1)  -> M12-M14, reading engine release notes
ring-buffer cache for SWA (§2)                  -> M10/M17, cache-layout contrasts
fixed-size state vs growing cache (§3)          -> M11, long-context benchmark framing
sparse read: size ∝ L, read ∝ k (§4)            -> M16-adjacent, attention-kernel variants
full attention + GQA as the default (§5)        -> every M-module, the baseline assumption
```

The one sentence to keep: **full attention's two costs — cache size and decode
read, both growing with context — are now design choices, attacked by shrinking
the cache (GQA/MLA), capping it (sliding windows), replacing it with fixed-size
state (linear/SSM hybrids), or reading it selectively (sparse attention); the
dense full-attention model stays the baseline every variant is measured
against.**
