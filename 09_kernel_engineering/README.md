# Chapter 9 — Kernel engineering

Chapters 1–8 are the foundation reading for the build journey (M0–M22): they build
the understanding to follow LLM inference from the first kernel to a serving
engine. **This chapter is different.** It's the conceptual prework for the
[Phase 5 kernel-engineering track](../ROADMAP.md) (M23–M30) — the deeper, optional
phase where the goal stops being "explain the gap to production kernels" and becomes
"close it." Read it when that track begins, or out of curiosity now; it's *not*
required before M0.

The chapter opens up the tools the frontier actually uses to write peak kernels.
[Chapter 7's kernel-language landscape](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md)
ended on *which* language to use — Triton for breadth, CuTe DSL for depth, CUDA/CUTLASS
as the bedrock you read. This chapter goes inside that answer: the CuTe DSL
vocabulary (layouts, tensors, atoms, the hardware glossary), and then FlashAttention
read across all four languages to see what each rung forces you to write. A final
section crosses to the other vendor — AMD's parallel stack (HIP, Composable Kernel,
FlyDSL, AITER) — where the same layout-and-atom skills transfer and the
kernel-contribution gap is wider. A closing section steps back from any single tool to the
*method*: how to practice all of it — change one variable per experiment, and run the
cheapest, highest-value reps first.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [CuTe DSL foundations](01_cute_dsl_foundations.md) | The CUDA C++ → CUTLASS → CuTe → CuTe DSL stack; why Python costs no runtime performance (compile-time metaprogramming); **layouts** (`Shape:Stride` as a coordinate→offset map), **tensors** (pointer + layout), **atoms** (one hardware MMA/copy instruction) and the tiled-MMA / Thread-Value layout; the `@jit`/`@kernel` host/device split; and the hardware glossary — MMA, WGMMA, tcgen05, TMEM, DMA, TMA, warp specialization, mbarrier |
| 2 | [FlashAttention Rosetta Stone](02_flashattention_rosetta_stone.md) | The same algorithm (tiled, IO-aware attention with online softmax) read across four languages from one repo — CUDA C++ (`csrc`), CuTe C++ (`hopper`), CuTe DSL (`flash_attn/cute`), Triton (`third_party`) — mapping what each GPU generation forced (WGMMA/TMA/warp specialization → tcgen05/TMEM) and what slides from the compiler to you as you descend the control ladder |
| 3 | [From source to SASS](03_source_to_sass.md) | The compilation pipeline stage by stage — Triton (`TTIR → TTGIR → LLVM IR → PTX → SASS`) vs CuTe DSL (both MLIR-based, sharing the `PTX → ptxas → SASS` back end); what each format looks like (layout-carrying MLIR, virtual PTX, real machine SASS); the PTX→SASS instruction map (`wgmma → HGMMA`, TMA `→ UTMALDG`); how the same GEMM produces different SASS (warp structure) in each pipeline; how to dump and diff every stage (`TRITON_KERNEL_DUMP`, `cuobjdump -sass`, `nvdisasm`); and the PTX layer as a unifying substrate, with the skeptical checklist for kernel claims |
| 4 | [Reading PTX and acting on it](04_reading_and_optimizing_ptx.md) | The practical profile-and-optimize loop — anatomy of a PTX kernel (registers, memory spaces, the load/compute/store shape); the reading checklist (vector width, cache operators, fast-path markers, coalescing, `.local` spills); profiling with Nsight Compute (Speed-of-Light bound, warp stall reasons — Long Scoreboard vs LG Throttle); the symptom→change→where map; a full end-to-end worked example (read PTX → profile → diagnose → vectorize in source → verify → re-measure); and when inline PTX is actually warranted |
| 5 | [Kernels on AMD: ROCm, Composable Kernel, FlyDSL](05_amd_kernel_track.md) | The AMD mirror of the NVIDIA stack from §1–§4 — HIP (↔CUDA C++), Composable Kernel (↔CUTLASS), **FlyDSL** (↔CuTe DSL: the Python, MLIR-native layout DSL), AITER (↔FlashInfer); the hardware-term map (CU / 64-lane wavefront / LDS / Matrix Core / MFMA); why CUTLASS being NVIDIA-only makes AMD a younger, thinner-staffed *parallel* stack worth contributing to; Triton as the write-once cross-vendor bridge; the rewrite→verify→profile→beat loop anchored on AMD's public Kimi-K2.5 fused-MoE-on-MI300X example; and the per-second cost ladder (with AMD's no-consumer-floor catch) for running it all with no hardware |
| 6 | [How I practice the kernel track](06_how_i_practice.md) | The *method* over the modules, not another tool: the one rule that makes practice compound — change exactly one dimension per experiment (which settles "same kernel on many GPUs" vs "many kernels on one GPU"); the dimensions (optimization level, baseline/provider, kernel type, GPU arch, vendor, model) ranked by value-per-dollar; reading a kernel's improvement history into a *move-list* as the free, zero-GPU rep; study-then-extend vs the model-kernel-by-kernel capstone; and the cheap→expensive sequence (read → GEMM-vs-cuBLAS → attention → arch sweep → model → AMD) as a lens over M23–M30 |

Prerequisites: [Chapter 7 — Writing and tuning kernels](../07_writing_and_tuning_kernels/README.md), especially its [kernel-language landscape](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md); and [Chapter 5 — Anatomy of a forward pass](../05_attention_and_kv_cache/README.md) for the attention algorithm.
Next: the Phase 5 modules (M23–M30) in the [Roadmap](../ROADMAP.md) — where these ideas get built, run, and measured.
