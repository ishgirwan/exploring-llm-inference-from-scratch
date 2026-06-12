# Roadmap

My plan for learning how LLM inference runs on GPUs, from the first kernel up to
real serving optimizations — and, in Phase 5, to writing competitive GPU kernels
myself. It's a plan, not a syllabus. I'll reorder it as I go.

## 1. What this is

I'm learning how LLMs run on GPUs (kernels, memory, profiling, serving engines)
and doing it in the open. This repo is the trail: the code, the numbers, what
confused me, and the explanations I worked out once it stopped confusing me.

It isn't a course. I'm a Python person learning GPU systems from scratch, and
when I explain something here I'm really explaining it to myself. 

I want to understand how the industry implements these
things, rebuild the ideas myself, and measure where my version falls short.

Public for two reasons: unfinished work in the open is harder to abandon, and
the trail might help someone later. 

## 2. How I work: do first, then understand

I bounce off reading-first material, so I chase a working artifact in each topic
instead: a kernel that runs, a benchmark with real numbers. The theory gets
pulled in behind it.

I'm not skipping theory, just changing when it shows up. Write a slow kernel,
measure that it's slow, then go learn the concept that explains why. Theory has
to earn its place by answering a question the code already raised.

The loop, every topic:

```
1. BUILD     Simplest version that works. .
2. VERIFY    Prove it's correct against something I trust (usually PyTorch).
3. MEASURE   Benchmark it properly.
4. PROFILE   See where the time went.
5. EXPLAIN   Learn the concept behind the numbers. Improve the code.
6. WRITE     What I built, what surprised me, what broke.
```

## 3. Layout of a topic

Three folders per topic:

- `lessons/NN_topic.md` — notes, the concept explained to myself
- `labs/NN_topic/` — the code
- `reports/NN_topic.md` — numbers, profiles, what I learned

Notes and reports use the same headings every time:

```
Concept · Why it matters · Implementation · Correctness · Benchmark
Profiler · Comparison · Reflection · Failure log · Hardware notes · Next step
```

The Failure log and "what surprised me" parts matter most. The clean final code
is the least interesting thing in the repo; the wrong turns are where the
learning is, so I'm leaving them in.

## 4. Hardware

I write code locally and run it on a
rented GPU.

| Stage | Topics | Hardware | ~Cost |
| --- | --- | --- | --- |
| Foundations | M0–M7 | Colab free tier (T4) | Free |
| Transformer & serving | M8–M14 | Rented 4090 or L4 | $0.30–0.70/hr |
| Optimization | M15–M19 | Rented 4090 / A100 | $0.40–1.50/hr |
| Scale & capstone | M20–M22 | Rented 2× A100 or H100 | $2–8/hr |
| Kernel engineering | M23–M30 | Rented H100; a Blackwell (B200) slice for M28; an MI300X slice for the optional AMD arm | $2–25/hr |

The Colab T4 covers M0–M7. It's a Turing GPU with no native bf16 and no FP8, so
from M7-ish I rent something stronger. Nsight Systems works
on Colab; Nsight Compute usually doesn't, since it needs permissions Colab locks
down.

`setup/` has a pinned container and provisioning scripts. I always use the
container so benchmarks don't drift as the software underneath changes. See
`BENCHMARKING.md`.

## 5. The plan

Roughly 30 topics in five phases. The early phases are concrete; the later ones
(especially Phase 5) are a guess and I expect to reorder them. Half-numbered items
(M2.5 etc.) are short, one focused note. Time estimates assume ~10 hrs/week and are
optimistic. The fine-grained path — each module broken into small steps, with
the reading braided in before and after each build, the questions to ask inside
each lab, and what each stage hands to the next — lives in
[`LEARNING_PATH.md`](LEARNING_PATH.md).

### Phase 1 — Foundations *(Colab T4 is enough)*

Understand what a GPU kernel is, why GPUs wait on memory instead of computing,
and how to write, check, measure and profile one.

| # | Topic | What I'll build | Concepts |
| --- | --- | --- | --- |
| M0 | Setup & the harness | Container, GPU-info script, the shared benchmark + correctness harness | reproducibility; methodology before kernels |
| M0.5 | Run a real model | Load a ~0.5B chat model on the T4, generate text, crudely time prefill vs decode | the whole journey made concrete on day one; TTFT and tokens/sec felt before they're formally defined |
| M1 | Vector add | The same op in CUDA C++, Triton, PyTorch | host vs device, grid/block/thread, warp, kernel launch |
| M2 | Memory bandwidth | Copy / scale / axpy kernels + bandwidth benchmark | global memory, coalescing, memory-bound kernels |
| M2.5 | Profiling basics | First Nsight Systems trace; roofline model | timelines, achieved vs theoretical bandwidth, arithmetic intensity |
| M3 | Reductions & softmax | Sum / max reduction, stable softmax | shared memory, synchronization, warp reductions, numerical stability |
| M3.5 | Sampling kernels | top-k, top-p, temperature, repetition penalty | why token sampling is a GPU problem |
| M4 | RMSNorm | RMSNorm in PyTorch, CUDA, Triton | vectorized loads, occupancy, register pressure |
| M5 | RoPE | A rotary positional encoding kernel | elementwise kernels, memory layout, fusion |
| M6 | Matmul & Tensor Cores | Naive → tiled → Triton matmul vs `torch.matmul` | GEMM, tiling, Tensor Cores, compute- vs memory-bound |
| M7 | PyTorch custom ops | My kernels as real PyTorch operators | C++/CUDA extensions, operator registration, `torch.compile` |

Checkpoint after M4: tag `v0.1`. Harness + first real LLM kernel, checked,
measured, profiled.

### Phase 2 — Putting a transformer together *(rent a 4090 or L4)*

Assemble the kernels into a transformer block, build a KV cache, benchmark real
serving engines. One rule for M12–M14: I pin the engine versions (and record
them in every results file) the day the module starts — vLLM and SGLang move
monthly, and numbers taken across drifting versions aren't comparable.

| # | Topic | What I'll build | Concepts |
| --- | --- | --- | --- |
| M8 | Advanced profiling | Deep `ncu` analysis of my kernels | warp stalls, occupancy, source-level attribution |
| M9 | Mini transformer block | RMSNorm + QKV + RoPE + attention + MLP + residual | a transformer as a graph of kernels |
| M10 | KV cache (toy) | A contiguous KV-cache attention path | KV cache layout, why decode reuses past keys/values |
| M11 | Prefill vs decode | Benchmarks of a real small model | prefill, decode, TTFT, TPOT/ITL, latency vs throughput, goodput |
| M12 | vLLM baseline | Benchmark scripts + workload generator | PagedAttention, continuous batching, prefix caching, chunked prefill |
| M12.5 | Toy batching scheduler | A ~200-line token-level scheduler | the biggest throughput lever, small enough to read |
| M13 | SGLang baseline | M12's workloads on SGLang | RadixAttention, prefix cache, scheduler differences |
| M14 | Agent workload benchmark | A repeated-prefix workload across engines | shared prefix, KV reuse, why agents stress caches |

Checkpoint after M14: tag `v0.2`. Full kernel stack, a serving benchmark, the
agent angle.

### Phase 3 — Optimization *(rent a 4090 / A100)*

These levers buy speed by touching the model's outputs, so every Phase 3 report
carries a **quality column** next to the speed column — perplexity and KL deltas
against the unoptimized baseline, with a task-eval gate before calling a module
done ([Chapter 4 §3](chapters/04_measurement/03_measuring_model_quality.md) defines the
metrics). The one exception is proper speculative decoding, which is lossless by
construction — there the quality column just says so.

| # | Topic | What I'll build | Concepts |
| --- | --- | --- | --- |
| M15 | Quantization & fused dequant | Quantized linear layer; fused dequant+matmul Triton kernel | INT8/FP8/INT4, weight-only quant, scales |
| M16 | FlashAttention | A simplified IO-aware attention kernel | tiling, on-chip SRAM, avoiding the full attention matrix |
| M17 | Paged attention (toy) | A toy PagedAttention with block tables | KV fragmentation, block tables, indirect memory access |
| M18 | CUDA graphs | Decode with and without graph capture | kernel-launch overhead, static shapes, graph replay |
| M19 | Speculative decoding | A draft/verify loop + acceptance analysis | draft model, verify step, acceptance rate, latency vs quality |

Checkpoint after M19: tag `v0.3`.

### Phase 4 — Scale & capstone *(rent 2× A100 or an H100)*

| # | Topic | What I'll build | Concepts |
| --- | --- | --- | --- |
| M20 | Multi-GPU inference | 1-GPU vs 2-GPU tensor-parallel benchmarks; NCCL traces | tensor parallelism, all-reduce, NVLink vs PCIe |
| M21 | Architecture comparison | The same workload across T4 / 4090 / A100 | Tensor Core generations, bandwidth, dtype support |
| M22 | Capstone | The full picture: model × quant × context × engine × GPU | one honest report pulling it together |

Checkpoint after M22: tag `v1.0`. The inference journey is complete; Phase 5 turns
that understanding into the ability to *author* competitive kernels.

### Phase 5 — Kernel engineering *(rent an H100; a Blackwell slice for M28)*

Phases 1–4 build understanding by reading real kernels and building simplified
ones — the "explain the gap" ethos of §9. Phase 5 raises the ceiling: go down to
the metal in the tools the frontier actually uses, and *close* the gap on at least
one kernel. The on-ramp is deliberate: Triton (already learned, M1–M16) taught the
tile model; **CuTe DSL** is the primary low-level tool here, because it reaches
CUTLASS/CUDA-grade performance with a far gentler learning curve than C++; raw
**CUDA C++ / CUTLASS** is what I read to see what compiles underneath. The key fact
that makes this reachable: CuTe DSL's Python runs once at *compile* time to
generate the kernel — it drives the same hardware primitives (WGMMA, TMA, tcgen05)
as the C++ path, with no Python left in the running kernel, so staying in Python
costs no performance at the ceiling. FlashAttention is the spine that ties all
three languages together.

| # | Topic | What I'll build | Concepts |
| --- | --- | --- | --- |
| M23 | CuTe DSL foundations | Layout / tensor / atom exercises; a copy kernel + GEMV in CuTe DSL, beside a Triton equivalent | layouts, tensors, MMA / copy *atoms*, host `@jit` vs device `@kernel`, the Python→MLIR→PTX JIT |
| M24 | Tiled GEMM, three ways | The same tiled matmul in Triton → CuTe DSL → CUTLASS, each measured vs cuBLAS | the M6 ladder mapped across languages; which rung each tool makes explicit vs hides |
| M25 | FlashAttention Rosetta Stone | An annotated read of FA1→FA4 side by side — CUDA (`csrc`), CuTe C++ (`hopper`), CuTe DSL (`flash_attn/cute`), Triton | what each rung forces you to write: SRAM tiling → WGMMA → TMA → warp specialization → Python codegen |
| M26 | Build FlashAttention | Carry M16's simplified Triton FA further, then rebuild it in CuTe DSL; verify + benchmark vs `flash-attn` | online softmax, on-chip tiling, the producer/consumer split, accumulator precision |
| M27 | Async pipelines & warp specialization | A warp-specialized, TMA-fed, multi-stage pipelined GEMM (Hopper) | TMA (a DMA engine), WGMMA, `mbarrier`, producer/consumer warps, pipeline depth, thread-block clusters + distributed shared memory |
| M28 | Blackwell tensor cores | Port a kernel to Blackwell: tcgen05 MMA (UMMA), Tensor Memory, low precision | 5th-gen Tensor Cores, TMEM, 2-CTA MMA, MXFP8 / NVFP4 microscaling |
| M29 | Fusion & perf portability | A fused kernel (fused MLP, or attention + bias + softmax) autotuned across two arches | epilogue + kernel fusion, persistent kernels, autotuning, why the best config differs per GPU |
| M30 | Beat-the-baseline capstone | Take one kernel and tune it until it matches or beats a strong reference on a target GPU | the honest bar: correctness + speedup vs a *real* baseline (cuBLAS / flash-attn), documented end to end |

Checkpoint after M30: tag `v2.0`. I can author a custom kernel that holds its own
against good hand-written code, and explain exactly why it's fast.

M30 also carries a stretch outcome past the private benchmark: **upstream
something**. A kernel improvement or fix merged into a library other people run —
FlashInfer, vLLM, SGLang, flash-attn, or AITER on the AMD arm, where the
contribution gap is widest ([Chapter 9 §5](chapters/09_kernel_engineering/05_amd_kernel_track.md)) —
is the one artifact that proves the skill to someone who wasn't watching; a
speedup that lives only in my own repo proves it only to me.

Everything in Phase 5 above is NVIDIA. AMD is the thinner-staffed frontier — the same
low-level skills transfer, but the kernel gap is wider and fewer hands close it.
[Chapter 9 §5](chapters/09_kernel_engineering/05_amd_kernel_track.md) maps the AMD stack (HIP ↔
CUDA C++, Composable Kernel ↔ CUTLASS, **FlyDSL** ↔ CuTe DSL, AITER ↔ FlashInfer) and an
optional AMD arm — an MI300X pass over M24/M26/M30 using Triton as the cross-vendor
bridge and FlyDSL as the CuTe-DSL-equivalent depth, on the same per-second rental model.

Two modules sit outside the phases, numbered after them but not gated on Phase 5:

- **M31 — MoE inference** *(planned; slots naturally right after M20)*. A toy
  MoE block — router, top-k dispatch, grouped-GEMM expert matmuls — verified and
  benchmarked, then an expert-parallel pass across two GPUs. Most current open
  frontier models (DeepSeek-V3, Llama 4, Qwen3's larger variants, Kimi K2,
  GPT-OSS) are MoE, so this stopped being optional;
  [Chapter 5's MoE section](chapters/05_attention_and_kv_cache/06_moe.md) is the prework.
- **M32 — disaggregated prefill/decode** *(maybe, if energy remains)*. Industry
  practice already (Mooncake, vLLM, NVIDIA Dynamo —
  [Chapter 8 §3](chapters/08_optimizing_inference/03_scaling_past_one_gpu.md)), but a
  heavy multi-node build for one person; it stays a stretch goal.

### The kernel-track reorder

The phase order above is the full inference journey, and it treats Phase 5 as the
optional summit. §9's second ambition inverts that: when kernel authorship is the
goal that leads, most of Phases 2–4 stops being prerequisite and becomes breadth.
Phase 5's real prerequisites are only Triton fluency (M1–M6), profiler fluency
(M2.5, then M8's Nsight Compute work), and the attention algorithm
([Chapter 5 §1](chapters/05_attention_and_kv_cache/01_attention.md), exercised in M16). So
the same modules admit a second, shorter spine:

```text
  M0–M8     unchanged — the harness, the kernel patterns, the matmul
            ladder, the first Nsight Compute session
  M9–M11    only as far as M16 needs them: a transformer block and the
            prefill/decode shapes, not yet the serving-engine work
  M16       the simplified Triton FlashAttention, pulled forward — the
            bridge between the Triton world and Chapter 9
  M23–M30   the kernel-engineering track, entered directly
```

Deferred is not deleted: the serving engines (M12–M14), the output-touching
optimizations (M15, M17–M19), and scale (M20–M22) stay on the list — a kernel
engineer who knows where kernels sit in a serving stack chooses better targets —
they just come after the kernel spine instead of before it. The checkpoint tags
in §6 bind to modules, not to calendar order, so each still lands when its
modules complete. The cost shape changes too: the H100 rentals of M23+ arrive
months earlier, and the cheap-first sequencing in
[Chapter 9 §6](chapters/09_kernel_engineering/06_how_i_practice.md) is what keeps that
affordable.

## 6. Checkpoints

A 30-topic project is well over a year part-time, and solo projects this size
usually die around month 3. So I tag checkpoints, points where the work adds up
to something whole.

| Tag | After | What I've got |
| --- | --- | --- |
| `v0.1` | M4 | Harness + a real LLM kernel, checked, measured, profiled |
| `v0.2` | M14 | Full kernel stack + serving benchmark + agent angle |
| `v0.3` | M19 | Optimization techniques, implemented and measured |
| `v1.0` | M22 | The full inference journey: model × quant × context × engine × GPU |
| `v2.0` | M30 | A custom kernel that matches/beats a strong baseline — and the skill to write more |

Stop at `v0.2` and I've still finished a real thing. That's the point of tagging.
`v1.0` is the complete inference story; `v2.0` is the kernel-engineering ceiling,
and it's optional — reach it only if the earlier phases leave me wanting to build,
not just understand.

## 7. Repo layout

```
README.md  ROADMAP.md  LEARNING_PATH.md  FAILURES.md  CHANGELOG.md  LICENSE
pyproject.toml  uv.lock     the Python project — uv-managed; ruff + pytest config
chapters/                   the reference library; each chapter has a README.md
                            indexing its sections, readable straight through or
                            pulled section by section as LEARNING_PATH.md calls
                            them; chapters grow as the project does
  01_hardware_fundamentals/   chapter 1, four sections (circuits → memory → GPU
                              execution model → GPU architecture); the running
                              example z = x+y → C[i] = A[i]+B[i] → C = A×B
  02_cuda_software_stack/     chapter 2; driver vs toolkit, host/device, the
                              library layer (cuDNN, cuBLAS, NCCL, CUTLASS, Triton),
                              and the end-to-end inference walkthrough: load →
                              prefill → decode, kernel selection, serving engines
  03_numerical_types/         chapter 3; IEEE 754, fp32/fp16/bf16/tf32/fp8,
                              int8/int4 quantization, accumulator precision
  04_measurement/             chapter 4; benchmarking methodology, the
                              first-run effects warm-up has to hide, and
                              measuring model quality (perplexity, KL, task
                              evals) — the column next to every speed column
  05_attention_and_kv_cache/  chapter 5, anatomy of a forward pass: attention
                              (Q·Kᵀ → softmax → ·V) + the KV cache, the MLP
                              (SwiGLU), the two ends (embedding + LM head), the
                              elementwise glue (RMSNorm, RoPE, residuals), and
                              sampling — every op the end-to-end §5 map deferred;
                              then the two architecture variants that bend the
                              dense map: MoE (router, experts, total vs active
                              params) and attention variants (sliding window,
                              linear/SSM hybrids, sparse attention)
  06_batching/                chapter 6; batching as the throughput lever —
                              intensity ≈ B, continuous batching, why it helps
                              the weight matmuls but not attention, KV-cache
                              VRAM as the batch-size cap; plus the host side —
                              the per-step CPU loop, async engines, structured
                              decoding; the bridge into M11–M13
  07_writing_and_tuning_kernels/  chapter 7; what a kernel is made of and how to
                              write + tune one, using matmul — the optimization
                              ladder (coalescing, tiling, registers, Tensor
                              Cores), autotuning, GPU-arch dependence, and the
                              kernel-language landscape (Triton → CuTe DSL → CUDA);
                              bridge to M1–M6 and the Phase 5 kernel-engineering track
  08_optimizing_inference/    chapter 8; the optimization map — decode's escape
                              routes (speculative decoding, multi-token prediction,
                              quantization, GQA/MQA/MLA KV shrinking) and prefill's
                              levers (prefix reuse, chunked prefill, FlashAttention,
                              prefill/decode disaggregation), each pinned to the
                              bottleneck it attacks; then scaling past one GPU —
                              tensor/pipeline/expert/data parallelism, what the
                              collectives cost, disaggregation as the split by
                              phase; bridge to M11–M20 and M31/M32
  09_kernel_engineering/      chapter 9 (Phase 5 prework, NOT pre-M0 reading); CuTe
                              DSL foundations (layouts, tensors, atoms, the hardware
                              glossary), the FlashAttention Rosetta Stone — one
                              algorithm across CUDA / CuTe C++ / CuTe DSL / Triton —
                              the source-to-SASS compilation pipeline, and reading PTX
                              + the profile-and-optimize loop, and the AMD stack (HIP,
                              Composable Kernel, FlyDSL, AITER), and a practice method
                              (one variable per experiment, cheap-first); bridge to M23–M30
setup/        docker container, provisioning scripts, check_gpu.py
common/       the shared harness — bench.py, correctness.py,
              results_schema.py, plot.py
lessons/      NN_topic.md — notes
labs/         NN_topic/   — code
tests/        correctness tests (some run GPU-free in CI)
benchmarks/   results/ (JSON), plots/, nsight/ (profiler captures)
reports/      NN_topic.md — numbers, profiles, reflections
.github/      CPU-only CI: lint + GPU-free tests
```

I build `common/` first, in M0, before any kernel. Boring, but it's what keeps
every later number trustworthy and comparable.

## 8. Following along

Welcome. Two ways in, depending on how you learn:

**The interleaved track (how I actually move).** Read the short on-ramp —
[Chapter 1 §3 (the GPU execution model)](chapters/01_hardware_fundamentals/03_gpu_model.md),
[Chapter 2 §1 (the stack)](chapters/02_cuda_software_stack/01_the_stack.md), and
[Chapter 4 §1 (benchmarking)](chapters/04_measurement/01_benchmarking.md), ~2 hrs —
then go straight to M0 and follow [`LEARNING_PATH.md`](LEARNING_PATH.md),
which braids the remaining chapter sections into the labs just-in-time:
a section right before the build that needs it, another right after the
measurement it explains. Chapters 5–6 unseal at the transformer-assembly
modules (M9–M14), Chapter 7 rung by rung inside M6, Chapter 8 at the
optimization phase (M15–M20). Chapter 9 is separate either way — prework for
the Phase 5 track (M23–M30); save it for when that track begins.

**The library track.** If reading-first suits you, the eight chapters are
written to be read straight through — start at
[`chapters/01_hardware_fundamentals/README.md`](chapters/01_hardware_fundamentals/README.md) and
follow each chapter's "Next chapter" link through `chapters/08_optimizing_inference/`
(~7–9 hrs), then begin at M0.

Either way: don't trust my numbers, re-run them; every one has a
script behind it. This will be wrong in places. When I find a mistake I fix it
and note it in `FAILURES.md` rather than quietly editing history. Spot
something before I do? Open an issue.

## 9. What I'm aiming for

Two levels of ambition, and Phase 5 is where the second one kicks in.

**Phases 1–4 — understand it honestly.** Industry-standard methodology:
correctness checks, disciplined benchmarking, real profiling, honest reporting. I
build simplified versions of the real kernels, read the production implementations
(FlashInfer, vLLM, FlashAttention), benchmark against them, and explain the gap.
Understanding that gap is the honest version of "I learned this," and for most of
the stack that understanding *is* the goal.

**Phase 5 — close the gap.** I'm no longer content to only explain the gap; I want
to shrink it. A *full* production library is still hundreds of person-months of
hardware-specific work, and I'm not rebuilding all of that. But the skill I'm after
is real and reachable: write kernels at the level the frontier actually uses — down
through CuTe DSL to Hopper's TMA/WGMMA and Blackwell's tcgen05/TMEM — and prove it
by taking at least one kernel and tuning it until it *matches or beats a strong
reference* (cuBLAS, flash-attn) on a target GPU (M30). The end state I'm aiming
for: given a GPU architecture and a model operation, I can translate it into a
hardware-efficient custom kernel that holds its own against good hand-written code,
and know exactly why it's fast. And the proof I want to leave behind is public:
at least one contribution merged into a kernel library other people actually run
(the M30 stretch).
