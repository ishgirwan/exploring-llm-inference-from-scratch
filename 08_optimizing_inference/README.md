# Chapter 8 — Optimizing inference

The earlier chapters keep landing on the same two facts: decode is memory-bound and
prefill is compute-bound. This chapter is the map of what you *do* about that — the
catalogue of inference optimizations, organized not as a list of tricks but by
which bottleneck each one attacks and how it moves the underlying ratio.

It's the most downstream of the foundation chapters: it builds on attention and the
KV cache (Chapter 5) and on batching (Chapter 6), and it's the bridge into the
optimization phase of the roadmap (M15–M19), with the serving-side routes pointing
back at M11–M14. Where Chapter 6 explains the across-users mechanics (continuous
batching, PagedAttention, prefix caching) in full, this chapter places them on the
map and goes deep on the techniques those chapters don't cover — speculative
decoding, multi-token prediction, quantization as a memory lever, and KV-cache
shrinking. A third doc leaves the single GPU: the four ways to split a model
across GPUs (tensor, pipeline, expert, data parallelism), what each costs in
communication, and disaggregation as the split by phase — the bridge to M20,
M31, and M32.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [Improving decode: escaping the memory wall](01_improving_decode.md) | The one ratio decode is bound by (bytes per useful output token) and the two ways to push on it; getting more tokens per weight load across users (batching, cross-ref Ch 6) or within one stream (speculative decoding, Medusa/EAGLE, multi-token prediction, and the "spend the idle compute" framing); moving fewer bytes per token (quantization; GQA/MQA/MLA and KV quant for the KV cache); prefix caching for the agent case; and choosing the lever by concurrency regime (llama.cpp vs. vLLM/SGLang), plus FlashDecoding and CUDA graphs |
| 2 | [Improving prefill: a different bottleneck, different levers](02_improving_prefill.md) | Why prefill is already compute-bound (N prompt tokens play the role of batch size B), so its levers are about work and scheduling, not escaping memory: prefix reuse (don't redo work), chunked prefill (don't block decode), FlashAttention (don't waste attention's HBM traffic at large N), and prefill/decode disaggregation (run each phase where its bottleneck fits); prefill owns time-to-first-token |
| 3 | [Scaling past one GPU](03_scaling_past_one_gpu.md) | Capacity vs speed as the reasons to split; tensor parallelism (two all-reduces per layer, NVLink vs PCIe), pipeline parallelism (capacity, not per-token speed), expert parallelism (MoE's all-to-all), data parallelism (replicas); what NCCL collectives cost; disaggregation revisited as the split by phase, now production practice (Mooncake, vLLM, NVIDIA Dynamo) |

Prerequisites: [Chapter 5 — Anatomy of a forward pass](../05_attention_and_kv_cache/README.md) and [Chapter 6 — Batching](../06_batching/README.md).
Next: M11–M19 in the [Roadmap](../ROADMAP.md) — where these optimizations get built and measured.
