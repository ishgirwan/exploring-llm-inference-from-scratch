# Chapter 9 — Kernel engineering

Chapters 1–8 are the prerequisite reading before M0: they build the understanding
to follow LLM inference from the first kernel to a serving engine. **This chapter is
different.** It's the conceptual prework for the
[Phase 5 kernel-engineering track](../ROADMAP.md) (M23–M30) — the deeper, optional
phase where the goal stops being "explain the gap to production kernels" and becomes
"close it." Read it when that track begins, or out of curiosity now; it's *not*
required before M0.

The chapter opens up the tools the frontier actually uses to write peak kernels.
[Chapter 7's kernel-language landscape](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md)
ended on *which* language to use — Triton for breadth, CuTe DSL for depth, CUDA/CUTLASS
as the bedrock you read. This chapter goes inside that answer: the CuTe DSL
vocabulary (layouts, tensors, atoms, the hardware glossary), and then FlashAttention
read across all four languages to see what each rung forces you to write.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [CuTe DSL foundations](01_cute_dsl_foundations.md) | The CUDA C++ → CUTLASS → CuTe → CuTe DSL stack; why Python costs no runtime performance (compile-time metaprogramming); **layouts** (`Shape:Stride` as a coordinate→offset map), **tensors** (pointer + layout), **atoms** (one hardware MMA/copy instruction) and the tiled-MMA / Thread-Value layout; the `@jit`/`@kernel` host/device split; and the hardware glossary — MMA, WGMMA, tcgen05, TMEM, DMA, TMA, warp specialization, mbarrier |
| 2 | [FlashAttention Rosetta Stone](02_flashattention_rosetta_stone.md) | The same algorithm (tiled, IO-aware attention with online softmax) read across four languages from one repo — CUDA C++ (`csrc`), CuTe C++ (`hopper`), CuTe DSL (`flash_attn/cute`), Triton (`third_party`) — mapping what each GPU generation forced (WGMMA/TMA/warp specialization → tcgen05/TMEM) and what slides from the compiler to you as you descend the control ladder |
| 3 | [From source to SASS](03_source_to_sass.md) | The compilation pipeline stage by stage — Triton (`TTIR → TTGIR → LLVM IR → PTX → SASS`) vs CuTe DSL (both MLIR-based, sharing the `PTX → ptxas → SASS` back end); what each format looks like (layout-carrying MLIR, virtual PTX, real machine SASS); the PTX→SASS instruction map (`wgmma → HGMMA`, TMA `→ UTMALDG`); how the same GEMM produces different SASS (warp structure) in each pipeline; and how to dump and diff every stage (`TRITON_KERNEL_DUMP`, `cuobjdump -sass`, `nvdisasm`) |

Prerequisites: [Chapter 7 — Writing and tuning kernels](../07_writing_and_tuning_kernels/README.md), especially its [kernel-language landscape](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md); and [Chapter 5 — Anatomy of a forward pass](../05_attention_and_kv_cache/README.md) for the attention algorithm.
Next: the Phase 5 modules (M23–M30) in the [Roadmap](../ROADMAP.md) — where these ideas get built, run, and measured.
