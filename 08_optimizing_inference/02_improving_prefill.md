# Improving prefill: a different bottleneck, different levers

Prefill is the other half of inference: the one-shot pass that reads the whole
prompt and fills the KV cache before the first output token comes out
([end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)).
The previous doc was about prying decode off the memory wall. Prefill doesn't sit
on that wall at all, so the levers are different — which is exactly why it gets its
own short doc instead of more rows in the decode one.

This is a map of where prefill's time goes and how the M-topics attack it; most of
the mechanics live in [Chapter 6](../06_batching/01_batching.md) and
[end-to-end §10](../02_cuda_software_stack/02_end_to_end_inference.md#10-the-serving-engine-who-drives-the-loop)
already, so this doc is deliberately pointer-heavy.

Prerequisites:
[Batching §3 and §8](../06_batching/01_batching.md#3-the-roofline-view-climbing-toward-the-ridge),
[End-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels),
and [Improving decode §5](01_improving_decode.md#5-the-agent-case-prefix-caching-wraps-the-shared-part).
Next: M11–M16 in the [Roadmap](../ROADMAP.md).

## 1. Prefill is already compute-bound

A prompt's `N` tokens are all known up front, so prefill processes them
*together* in one pass. That makes every weight load feed `N` tokens at once —
`N` plays the exact role the batch size `B` plays in decode
([batching §3](../06_batching/01_batching.md#3-the-roofline-view-climbing-toward-the-ridge)).
With `N` in the hundreds or thousands, arithmetic intensity is already high and the
Tensor Cores are already near saturated: prefill is **compute-bound**, the opposite
of decode.

So the decode playbook doesn't transfer. There's no idle compute to spend on
speculation, and shaving bytes barely helps a workload that isn't waiting on bytes.
The questions that matter for prefill are different:

```text
  prefill is compute-bound, so the levers are about WORK and SCHEDULING:
    - don't redo work you've already done      (§2  prefix reuse)
    - don't let a big prefill block decode      (§3  chunked prefill)
    - don't waste HBM traffic in attention      (§4  FlashAttention)
    - run it where its bottleneck fits          (§5  disaggregation)
```

Why prefill is worth optimizing at all: it owns **TTFT** (time to first token) —
the user-visible wait before *any* output appears. Decode owns the speed *after*
that; prefill owns the silence before it. M11 measures both.

## 2. Don't redo work: prefix reuse

The largest prefill saving isn't making the pass faster — it's not doing it. When
requests share leading tokens (a common system prompt, the same few-shot examples,
an agent re-sending its history), the KV cache for that shared prefix is identical
and can be computed once and reused, skipping the prefix's entire forward pass on
every later request — *prefix caching* / *RadixAttention*
([end-to-end §10](../02_cuda_software_stack/02_end_to_end_inference.md#10-the-serving-engine-who-drives-the-loop)).

Decode §5 already placed this on the escape map
([improving decode §5](01_improving_decode.md#5-the-agent-case-prefix-caching-wraps-the-shared-part));
the point to add here is that prefill is where it *pays off most*. A cached prefix
turns a long, compute-heavy prefill into a near-instant cache hit, collapsing TTFT
for any request that shares it. This is precisely why agent workloads — which
re-send near-identical prompts turn after turn — stress the prefix cache so hard,
and why M13 (SGLang) and M14 (the agent benchmark) center on it.

## 3. Don't let a big prefill block decode

A server runs prefill and decode at the same time, across its many in-flight
requests — and they interfere. A long prefill is a heavy compute burst; run it
whole in one step and every concurrent decode stalls behind it, spiking those
users' per-token latency (TPOT). *Chunked prefill* splits a long prefill into
smaller pieces that interleave with ongoing decodes, so neither phase starves the
other
([batching §8](../06_batching/01_batching.md#8-loose-ends-prefill-and-how-we-measure-this),
[end-to-end §10](../02_cuda_software_stack/02_end_to_end_inference.md#10-the-serving-engine-who-drives-the-loop)).
It doesn't make prefill cheaper; it keeps one request's prefill from holding the
whole batch's decode hostage. That's an M12 (vLLM) behavior to measure.

## 4. Don't waste HBM traffic in attention

Even compute-bound prefill has one part that touches a lot of memory: attention.
Computing `Q·Kᵀ` over a prompt of length `N` forms an `N×N` matrix of attention
scores, and `N×N` grows fast — at `N` = a few thousand, naively writing that matrix
out to HBM (the GPU's main memory) and reading it back dominates the step.

*FlashAttention* avoids ever materializing the `N×N` matrix: it tiles the
computation and accumulates the softmax online, in fast on-chip SRAM, so the big
score matrix never hits HBM. It's IO-aware — it optimizes bytes moved, not FLOPs —
which is why it helps most exactly where `N` is large, i.e. prefill (and
long-context decode, via FlashDecoding,
[improving decode §6](01_improving_decode.md#6-which-lever-for-which-regime)).
M16 builds a simplified version and benchmarks it against the real one.

## 5. Run each phase where its bottleneck fits

The deepest lever follows from §1's observation that prefill and decode have
*opposite* resource profiles — prefill is compute-bound, decode is memory-bound.
Co-locate them on the same GPU and each phase's idle resource is the other's
bottleneck: while a GPU grinds a compute-heavy prefill, its memory bandwidth sits
underused, and vice versa.

*Prefill/decode disaggregation* runs the two phases on **separate** GPU pools — one
sized for compute-bound prefill, one for memory-bound decode — and ships the KV
cache between them. Each pool then runs near the bottleneck it was provisioned for.
This is the most speculative item here: the roadmap files it under "maybe later"
(M32), so treat it as a direction I'm flagging, not a topic I've committed to
building.

## 6. What to carry forward

```text
prefill is compute-bound; N plays B's role (§1)   -> the frame; M11 (TTFT)
prefix reuse skips the shared prefill (§2)         -> M13 (SGLang), M14 (agents)
chunked prefill keeps prefill from blocking        -> M12 (vLLM)
  decode (§3)
FlashAttention cuts attention's HBM traffic,        -> M16, build + benchmark
  matters most at large N (§4)
prefill/decode disaggregation (§5)                  -> M32, "maybe later" — speculative
```

The one sentence to keep: **prefill is already compute-bound — `N` prompt tokens
share every weight load — so its levers aren't about escaping a memory wall but
about not redoing work (prefix reuse), not letting a big prefill block everyone's
decode (chunked prefill), not wasting attention's memory traffic (FlashAttention),
and ultimately running it where its compute-bound profile fits (disaggregation) —
and it matters because prefill owns the time-to-first-token.**
