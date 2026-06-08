# Chapter 5 — Anatomy of a forward pass

This chapter is the anatomy of a single forward pass: every operation the
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
map named and then deferred, worked out on paper. Its heart is **attention** and
the **KV cache** (§1) — the only operation where tokens look at each other, and
where inference's memory behaviour gets most interesting: the cache that makes
decode fast is also what grows with context and eventually dominates VRAM
traffic. Around that core the chapter assembles the rest of the pass — the
**MLP** (§2), the model's two ends (**embedding** and the **LM head**, §3), the
elementwise **glue** that threads through every layer (§4), and **sampling**,
which turns the final scores into the next token (§5). Together they explain the
whole `end-to-end §5` layer graph, end to end. The map for the M3.5–M17 build
topics.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [Attention and the KV cache](01_attention.md) | Q/K/V as a soft dictionary lookup; the five-step computation with a worked numeric example; multi-head attention; prefill vs decode shapes; the KV cache layout, size formula, GQA, and paged vs contiguous; why decode attention is KV-cache-bandwidth-bound; what FlashAttention avoids |
| 2 | [The MLP / feed-forward block](02_mlp_feedforward.md) | Position-wise transform vs attention's token mixing; the expand→activate→contract FFN; GELU/SiLU; SwiGLU's 3-matrix gated form and the `d_ff` width; why the MLP is ~2/3–80% of a model's weights; why MLP decode is weight-memory-bound (the "first wall") |
| 3 | [The two ends: embedding & LM head](03_embedding_and_lm_head.md) | The two ends as mirror images; the embedding as a `[vocab × d_model]` lookup (a gather, not a matmul); the LM head as the transposed projection to logits; weight tying; why the LM head is one of the largest single matrices; the prefill last-position shortcut |
| 4 | [The elementwise glue: RMSNorm, RoPE, residuals](04_elementwise_glue.md) | The non-matmul glue on the FP lanes — RMSNorm (rescale to RMS 1) vs LayerNorm; RoPE (rotating Q,K so attention sees relative position); residual adds and the residual stream; why all three are memory-bound and the prime targets for fusion |
| 5 | [From logits to a token: sampling](05_sampling.md) | Logits → one token; softmax to a distribution; temperature (sharpen/flatten, `T→0` = greedy); top-k (fixed) vs top-p/nucleus (adaptive) truncation; repetition penalty; greedy vs sampling; why it's a per-token reduction over the whole vocabulary |

Prerequisites: [Chapter 2 §2 — End to end: a prompt becomes tokens](../02_cuda_software_stack/02_end_to_end_inference.md) and [Chapter 1 §3 — The GPU execution model](../01_hardware_fundamentals/03_gpu_model.md).
Next chapter: [Chapter 6 — Batching](../06_batching/README.md).
