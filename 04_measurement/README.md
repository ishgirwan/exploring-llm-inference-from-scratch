# Chapter 4 — Measurement

How to get a number out of a GPU that we can trust and reproduce.
Benchmarking methodology — CUDA events, synchronisation, percentiles, clock
locking — and the one-time first-run effects that ruin naive measurements if
they are not discarded.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [Benchmarking methodology](01_benchmarking.md) | The rules every benchmark in this repo follows: correctness gates, GPU-event timing, warm-up, percentiles (p50/p95/p99), clock locking, environment recording, results schema, reading GPU utilization (memory vs compute) |
| 2 | [First-run effects](02_first_run_effects.md) | What warm-up actually hides — the caching allocator, JIT compilation, autotuning, cold caches, GPU power states, and CUDA context initialisation |

Prerequisites: [Chapter 1 — Hardware fundamentals](../01_hardware_fundamentals/README.md), [Chapter 2 — The CUDA software stack](../02_cuda_software_stack/README.md), [Chapter 3 — Numerical types](../03_numerical_types/README.md).
Next chapter: [Chapter 5 — Anatomy of a forward pass](../05_attention_and_kv_cache/README.md).
