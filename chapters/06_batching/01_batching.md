# Batching: the throughput lever

Three docs so far have hit the same wall and put up the same "for now" sign.
[End-to-end §7](../02_cuda_software_stack/02_end_to_end_inference.md#7-stage-3--decode-the-autoregressive-loop)
said decode is memory-bound **at batch size 1**.
[Attention §8](../05_attention_and_kv_cache/01_attention.md#8-why-decode-attention-is-memory-bound--and-its-a-different-wall)
and
[MLP §7](../05_attention_and_kv_cache/02_mlp_feedforward.md#7-why-mlp-decode-is-weight-memory-bound--the-first-wall)
both carried that same "at batch size 1" qualifier. This doc is what they were
deferring to: what changes when you stop running one sequence at a time and run
many at once. It's the single biggest throughput lever in serving, which is why
the roadmap gives it a whole topic of its own (M12.5).

Like the rest of these foundation docs it's a map, but it's the one that points
most directly at the serving work ahead — read it as the bridge into M12–M13,
not as theory to finish before M0.

Prerequisites:
[End-to-end §7 and §10](../02_cuda_software_stack/02_end_to_end_inference.md#7-stage-3--decode-the-autoregressive-loop),
[Attention §7–§8](../05_attention_and_kv_cache/01_attention.md#7-the-kv-cache-what-it-stores-and-why-its-shaped-that-way),
[MLP §7](../05_attention_and_kv_cache/02_mlp_feedforward.md#7-why-mlp-decode-is-weight-memory-bound--the-first-wall),
and [execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound).
Next: M11–M13 in the [Roadmap](../../ROADMAP.md).

## 1. The problem batching solves

At batch size 1, decode reads a weight out of VRAM and uses it for exactly one
token's worth of math before discarding it
([MLP §7](../05_attention_and_kv_cache/02_mlp_feedforward.md#7-why-mlp-decode-is-weight-memory-bound--the-first-wall)).
The Tensor Cores spend almost the whole step idle, waiting for the next slab of
weights to stream in. The model's whole weight set (~140 GB for a 70B model) is
dragged across the VRAM bus to produce one token.

```text
  batch 1:  load weight tile ──► 1 multiply ──► throw it away ──► load next
            (the Tensor Cores idle between loads)
```

The waste is glaring: that weight tile, sitting in shared memory, could feed many
multiplies instead of one — *if there were more tokens that needed it right now*.
There are, if you serve more than one request: every sequence being decoded needs
the **same** weights this step. So load each weight once and use it for all of
them. That is batching.

## 2. Batching reuses each weight load — intensity climbs with B

Run `B` sequences' current tokens together. The activations stack into a `B`-row
matrix, so each projection turns from matrix×vector into matrix×matrix:

```text
  batch 1:   W [n×m] · x [m]      = matrix × vector   -> 1 token's output
  batch B:   W [n×m] · X [m×B]    = matrix × matrix   -> B tokens' outputs
             (W is read from VRAM ONCE, used for all B columns)
```

Count the arithmetic intensity — FLOPs per byte loaded
([execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)):

```text
  FLOPs   = 2 · n · m · B      (the matmul, B columns)
  weight bytes read = 2 · n · m   (read ONCE, regardless of B)
  intensity = FLOPs / bytes ≈ B   FLOP/byte
```

So intensity rises roughly *linearly with the batch size* (ignoring the
activation bytes, which are small next to the weights). At `B = 1` it's ~1
FLOP/byte — deep in memory-bound territory. Crank `B` up and each expensive
weight load does `B`× more useful work.

## 3. The roofline view: climbing toward the ridge

[Execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)
introduced the idea that a kernel is memory-bound below some arithmetic-intensity
*ridge point* and compute-bound above it (a few hundred FLOP/byte on an H100).
Batch size walks the projection/MLP work up that slope:

```text
  throughput
  (tokens/s)            ____________  compute-bound ceiling (FLOP/s limit)
       |              /
       |            /
       |          /   <- climbing: intensity ≈ B, weight-bound matmuls
       |        /        do more work per byte as B grows
       |      /
       |____/________________________ intensity (≈ B) ->
          memory-bound          ridge (~hundreds)
```

But be careful not to overclaim. You **approach** the ridge with larger `B`; you
rarely **reach** it in decode, for two reasons the next sections give: the KV
cache caps how large `B` can get (§7), and attention never climbs the slope at
all (§4). So the honest statement is three parts:

```text
  - the projection + MLP work APPROACHES compute-bound as B grows
  - attention stays KV-memory-bound regardless of B            (§4)
  - B is capped by KV-cache VRAM before you hit the ridge      (§7)
```

What rises cleanly with `B` is **throughput** (tokens/sec across all sequences),
because the fixed cost of streaming the weights is now shared. What rises too is
**per-token latency** — a bigger matmul takes a little longer, and a token waits
for its `B−1` batchmates each step. That is the central serving tradeoff:

```text
  bigger batch  ->  higher throughput (good for cost/utilization)
                ->  higher per-token latency (worse for one user's speed)
```

Prefill, by contrast, is *already* compute-bound without any batching trick,
because a single prompt's `N` tokens are processed together — `N` plays the role
of `B`
([end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)).
Decode is the phase that needs batching.

## 4. The subtlety: batching helps the weights, not attention

This is the part that makes batching worth its own doc. Batching amortizes a
weight load across `B` sequences — but **attention has no shared weights to
amortize**. Its memory cost is the KV cache
([attention §7](../05_attention_and_kv_cache/01_attention.md#7-the-kv-cache-what-it-stores-and-why-its-shaped-that-way)),
and every sequence has *its own* cache. Batch `B` sequences and you read `B`
separate caches — the FLOPs and the bytes both go up by `B`, so the ratio doesn't
move:

```text
  projection / MLP   weights shared across the batch
                     intensity ≈ B          -> climbs toward compute-bound

  attention          each sequence reads ITS OWN KV cache
                     FLOPs ×B, KV bytes ×B  -> intensity stays ≈ 1
                                            -> STAYS KV-memory-bound
```

This is exactly the two-walls split from
[attention §8](../05_attention_and_kv_cache/01_attention.md#8-why-decode-attention-is-memory-bound--and-its-a-different-wall),
now seen through the batching lens: batching knocks down the **weight wall** (the
projections and the MLP, most of the model) but leaves the **KV-cache wall**
standing. It's why attention gets its own specialised kernels (FlashAttention,
PagedAttention) — it's the part batching can't fix.

The one exception, worth flagging because it's a whole optimization: when several
sequences **share a prompt prefix** (the same system prompt, or an agent loop
re-sending history), their KV cache for that prefix is identical and can be stored
*once* and reused — *prefix caching* / *RadixAttention*
([end-to-end §10](../02_cuda_software_stack/02_end_to_end_inference.md#10-the-serving-engine-who-drives-the-loop),
SGLang, M13). That's the one way attention work gets amortized across sequences,
and only for the shared part.

## 5. Static batching, and why it wastes the GPU

The naive way to batch: collect `B` requests, run them together until they're all
done, then take the next `B`. The problem is that sequences finish at *different*
times — they generate different numbers of tokens — so the batch runs until the
**longest** one finishes, and every sequence that finished earlier leaves its slot
sitting idle:

```text
  static batch of 4 (each row is one sequence; • = a decode step, _ = idle slot)

  seq A  • • • ✓ _ _ _ _ _ _ _ _      finished at step 4, idle for 8 more
  seq B  • • • • • • • • • • • ✓      the long pole — the batch waits for it
  seq C  • • • • • ✓ _ _ _ _ _ _      finished at step 6, idle for 6 more
  seq D  • • ✓ _ _ _ _ _ _ _ _ _      finished at step 3, idle for 9 more
         ^^^^^^^^^^^^^^^^^^^^^^^^
         the GPU runs all 12 steps at batch 4, but most slots are dead weight
```

Those idle slots are wasted compute and wasted KV-cache VRAM. With realistic,
highly variable output lengths, static batching can leave the GPU half-empty.

## 6. Continuous batching: refill the batch every step

The fix is to stop thinking in whole requests and schedule at the granularity of
**one decode step**. After every step, evict the sequences that just finished and
admit waiting requests into the freed slots — so the batch is continuously topped
up instead of draining to empty. This is *continuous batching* (also called
*iteration-level scheduling* or *in-flight batching*), the core of vLLM
([end-to-end §10](../02_cuda_software_stack/02_end_to_end_inference.md#10-the-serving-engine-who-drives-the-loop)):

```text
  continuous batching (new requests E,F,G admitted as A,D,C finish)

  seq A  • • • ✓
  seq D  • • ✓
  seq E      ▸ • • • • • • ✓        admitted into D's freed slot
  seq B  • • • • • • • • • • • ✓
  seq C  • • • • • ✓
  seq F          ▸ • • • • • • ...  admitted into C's freed slot
  seq G        ▸ • • • • • • ...    admitted into A's freed slot
         ^^^^^^^^^^^^^^^^^^^^^^^^
         the batch stays full — no waiting for the long pole
```

The GPU stays near a full batch the whole time, so the weight-load amortization of
§2 actually holds in practice instead of decaying as sequences drop out. This one
change is the difference between a serving engine and a `for`-loop, and it's the
lever M12.5 builds as a ~200-line scheduler.

## 7. The constraint: KV-cache VRAM caps the batch

If bigger `B` means more throughput, why not make `B` enormous? Because each
concurrent sequence needs its own KV cache in VRAM, and that grows with context
length
([attention §7](../05_attention_and_kv_cache/01_attention.md#7-the-kv-cache-what-it-stores-and-why-its-shaped-that-way)):

```text
  total KV VRAM = per-token KV × context length × B

  so:  max B ≈ (VRAM left after weights) / (per-token KV × context length)
```

Weights take a fixed chunk of VRAM; whatever's left is the KV-cache budget, and
that budget — divided by how much each sequence's cache costs — is the real cap on
batch size. Long contexts make each sequence expensive, so `B` shrinks; this is
why you usually hit the **VRAM wall before the compute ridge** (§3).

That makes packing the KV cache efficiently a direct throughput lever: the tighter
you pack it, the more sequences fit, the bigger `B`, the higher throughput. That's
the whole point of **PagedAttention** — store the cache in small fixed-size blocks
so there's no wasted space reserved for max-length sequences that never get there
([end-to-end §10](../02_cuda_software_stack/02_end_to_end_inference.md#10-the-serving-engine-who-drives-the-loop);
the toy version is M17).

## 8. Loose ends: prefill, and how we measure this

Two threads to pick up properly later:

**Mixed prefill and decode.** Prefill is compute-bound, decode is memory-bound
(§3), and a real server has both happening across its many requests. Naively, a
long prefill blocks everyone's decode for a step. *Chunked prefill* splits a big
prefill into pieces that interleave with ongoing decodes so neither starves
([end-to-end §10](../02_cuda_software_stack/02_end_to_end_inference.md#10-the-serving-engine-who-drives-the-loop)).

**The metrics batching trades between.** Because batching pushes throughput up and
per-token latency up at the same time, you can't talk about it with one number.
The serving metrics that pull apart are *TTFT* (time to first token — dominated by
prefill), *TPOT* / *ITL* (time per output token — the decode step time), and
*throughput* (total tokens/sec across all sequences). M11 defines these and
measures the tradeoff on a real model.

## 9. Further reading

Batching is the biggest throughput lever, so these go deep on both the arithmetic
and the scheduler that exploits it:

- **[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)**
  (Carol Chen / kipply, 2022) — the operational-intensity math behind "intensity ≈ B"
  (§2), and why batching lifts the weight matmuls toward compute-bound but does nothing
  for per-sequence attention (§4).
- **[How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)**
  (Anyscale, 2023) — the diagram-driven contrast of static vs continuous batching (§6),
  with benchmarks across real serving engines.
- **[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)**
  (Yu et al., OSDI 2022) — the paper that introduced iteration-level scheduling and
  selective batching: the origin of the continuous batching in §6.
- **[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**
  (Kwon et al., SOSP 2023) — the vLLM paper, on why KV-cache VRAM is the real cap on
  batch size (§7) and how paging stretches it.
- **[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)** (GeeeekExplorer, 2025) —
  the vLLM-style serving core — scheduler, prefix caching, tensor parallelism, CUDA
  graphs — rebuilt in ~1,200 lines of readable Python at speeds comparable to vLLM's.
  Small enough to read in a sitting: the bridge between this doc's diagrams and a real
  engine's codebase, and the natural warm-up for M12.5's toy scheduler.

## 10. What to carry forward

```text
batching reuses weight loads, intensity ≈ B (§2)     -> M11, measured
the throughput/latency tradeoff (§3)                  -> M11, the metrics
batching helps weights, not KV (§4)                   -> M16-M17, why attention
                                                         needs its own kernels
continuous batching (§6)                              -> M12 (vLLM), M12.5 (build it)
KV VRAM caps the batch (§7)                           -> M12, M17 (PagedAttention)
prefix sharing amortizes attention (§4)               -> M13 (SGLang, RadixAttention)
```

The one sentence to keep: **batching runs many sequences' decode steps together so
each weight load out of VRAM is reused across all of them — which lifts the
projection and MLP work toward compute-bound and is the main throughput lever — but
it can't help attention (every sequence has its own KV cache), and the KV cache's
VRAM footprint is what ultimately caps how large the batch can grow.**
