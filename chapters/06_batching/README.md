# Chapter 6 — Batching

Batching is where these foundation chapters hand off to the serving work. The
earlier docs kept qualifying their memory-bound claims with "at batch size 1";
this chapter explains what changes when you run many sequences at once — the
single biggest throughput lever in LLM serving — and why it helps the weight-heavy
matmuls but not attention.

Unlike the chapters before it, this isn't really pre-M0 theory: batching is an
M12-era serving concept. It sits here as the **bridge** — the idea that resolves
the recurring "batch size 1" qualifier and sets up the M11–M13 serving topics
(prefill/decode metrics, vLLM, a toy scheduler, SGLang). A second doc looks up
the stack at the **host side** — the CPU loop of scheduling, bookkeeping,
detokenizing, and grammar masking that must finish inside each GPU step, and
that grows with exactly the request count batching maximizes.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [Batching: the throughput lever](01_batching.md) | Why batch-1 decode wastes the cores; how batching makes arithmetic intensity ≈ B and lifts the projection/MLP work toward compute-bound; why it can't help attention (per-sequence KV cache) except via shared-prefix reuse; static vs. continuous batching; how KV-cache VRAM caps the batch size; the throughput/latency tradeoff and the TTFT/TPOT metrics |
| 2 | [The host side](02_host_side.md) | The per-step CPU budget and the gap-between-steps symptom; a request's path through the engine (API → tokenize → schedule → step loop → detokenize → stream); where the milliseconds go (Python/GIL, launch overhead, block bookkeeping); async engines that hide the host behind the GPU step; structured decoding — logit masking, compiled grammars (Outlines/XGrammar/llguidance), jump-forward decoding |

Prerequisites: [Chapter 5 — Anatomy of a forward pass](../05_attention_and_kv_cache/README.md) and [Chapter 2 §2 — End to end: a prompt becomes tokens](../02_cuda_software_stack/02_end_to_end_inference.md).
Next chapter: [Chapter 7 — Writing and tuning a kernel](../07_writing_and_tuning_kernels/README.md).
