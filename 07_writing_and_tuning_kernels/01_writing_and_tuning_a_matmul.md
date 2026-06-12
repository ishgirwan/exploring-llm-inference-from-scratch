# Writing and tuning a kernel: a matmul, start to finish

Every doc so far has *named* kernels — "a cuBLAS GEMM," "the attention kernel,"
"a fused elementwise kernel" — without ever opening one up. This doc opens one
up. It takes a single kernel and walks the whole craft: what it's made of, how to
write a correct version, and then the ladder of optimizations that turns a
correct-but-slow kernel into a fast one — plus the *tuning knobs* and the
*autotuning* that find the fastest configuration for a given GPU.

I use **matrix multiply (GEMM)** as the example, for three reasons: it's the LLM
workhorse (every projection and MLP, and the heart of attention, is a GEMM —
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)),
it's the single most-studied optimization target in GPU computing, and every
concept worth knowing — coalescing, tiling, registers, Tensor Cores, occupancy,
autotuning — shows up in it.

**One framing to fix before anything else.** The optimization ladder in Part B is
the story of a *large, compute-bound* GEMM — which is **prefill** (or any
large-batch step). The repo has spent four docs establishing the *other* regime:
**decode is a skinny matrix×vector, memory-bound at ~1 FLOP/byte**
([end-to-end §7](../02_cuda_software_stack/02_end_to_end_inference.md#7-stage-3--decode-the-autoregressive-loop),
[attention §8](../05_attention_and_kv_cache/01_attention.md#8-why-decode-attention-is-memory-bound--and-its-a-different-wall),
[MLP §7](../05_attention_and_kv_cache/02_mlp_feedforward.md#7-why-mlp-decode-is-weight-memory-bound--the-first-wall)),
where tiling-for-reuse does *not* help. §11 returns to that regime and connects it
to batching. Hold the distinction the whole way down.

This is a map of the craft, not a code tutorial — the actual implementation, with
real numbers, is M1–M6. It's the bridge into that kernel-building work.

Prerequisites: [Chapter 1 — Hardware fundamentals](../01_hardware_fundamentals/README.md)
(SM, warps, shared memory, Tensor Cores, the memory hierarchy),
[Chapter 2 — The CUDA software stack](../02_cuda_software_stack/README.md)
(CUDA/Triton, call-vs-build), [Chapter 3 — Numerical types](../03_numerical_types/README.md),
and [Chapter 4 — Measurement](../04_measurement/README.md).
Next: M1 in the [Roadmap](../ROADMAP.md).

---

# Part A — Writing the kernel (get it correct)

## 1. What a kernel is made of

Strip away the specifics and *every* GPU kernel has the same four-part skeleton.
Each thread:

```text
  1. IDENTIFY   which piece of the output this thread is responsible for,
                from its block/thread indices (execution model §2)
  2. LOAD       read that piece's inputs from memory (global -> maybe shared
                -> registers), climbing the hierarchy toward the compute units
  3. COMPUTE    do the arithmetic, on FP/INT lanes or Tensor Cores
  4. STORE      write the result back down to global memory
```

That's it. A kernel is a per-thread recipe for "figure out what I own, fetch it,
compute it, write it back." The reason there's a whole *craft* is that steps 2 and
4 — moving bytes — are far slower than step 3, so almost all optimization is about
**moving fewer bytes, and reusing each byte more** once it's on-chip. Keep that
sentence; it's the through-line of Part B.

The thread hierarchy each kernel maps work onto (full build-up in
[execution model §2–§6](../01_hardware_fundamentals/03_gpu_model.md#2-gpu-threads-map-work-to-data)):

```text
  grid    -> all threads of the launch
    block   -> runs on one SM; cooperates via shared memory + barriers
      warp    -> 32 threads issued together
        thread  -> one lane, one piece of output
```

## 2. The naive matmul — correct, and slow

For `C = A × B` with `A` `[M×K]`, `B` `[K×N]`, `C` `[M×N]`, the simplest kernel
assigns **one thread per output element** `C[row, col]`, and that thread loops
over `k` (the recipe from
[execution model §9](../01_hardware_fundamentals/03_gpu_model.md#9-matrix-multiplication-adds-data-reuse)):

```text
  __global__ void naive_matmul(float* A, float* B, float* C, int M, int N, int K){
      int row = blockIdx.y * blockDim.y + threadIdx.y;   // step 1: identify
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (row < M && col < N) {
          float acc = 0.0f;
          for (int k = 0; k < K; ++k)                    // step 2+3: load+compute
              acc += A[row*K + k] * B[k*N + col];        //   read from GLOBAL each time
          C[row*N + col] = acc;                          // step 4: store
      }
  }
```

It is correct. It is also slow, and §4 explains exactly why: every thread reads a
full row of `A` and a full column of `B` straight from global memory, with no
reuse — so neighbouring threads re-read the same rows and columns over and over,
hammering the slowest level of the hierarchy.

```text
  naive: thread for C[row,col] reads, from GLOBAL memory:
     A[row, 0..K]   (a whole row)        \  K + K loads from the ~500-cycle
     B[0..K, col]   (a whole column)     /  level, for ONE output element
```

## 3. The gate: verify before you tune

Before touching performance, prove the kernel is **correct** against a trusted
reference — `torch.matmul`, or cuBLAS — across a range of shapes and dtypes. This
is not optional bookkeeping; it is the hinge between writing and tuning, for one
blunt reason:

```text
  a fast kernel that computes the wrong answer is worth nothing,
  and a speedup measured on an unverified kernel is meaningless.
```

This is exactly the AI-kernel cautionary tale from
[the stack doc](../02_cuda_software_stack/01_the_stack.md#what-people-actually-use):
the headline "faster than handwritten" results that collapsed were *fast but
wrong*, or fast against a weak baseline. So the workflow — and the M0 ethos of
this whole project — is **write → verify → then tune**, never tune first. The
verification harness ([Benchmarking](../04_measurement/01_benchmarking.md)) is
what makes every later speedup number trustworthy, and it's the real meaning of
"KernelBench-worthy" (§17): correctness first, speed second.

---

# Part B — Tuning: the optimization ladder (the compute-bound regime)

This is the large-GEMM = **prefill / large-batch** story (the framing up top). Each
rung follows the same shape: *what it changes → why it helps → which tuning knob it
introduces.* The goal is to climb off the memory-bound floor toward compute-bound.

## 4. The roofline frame: from memory-bound to compute-bound

Whether matmul is memory- or compute-bound depends on *arithmetic intensity* —
FLOPs per byte loaded
([execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)).
The naive kernel reloads operands per output, so intensity is low → **memory-bound**,
the lanes starving for data. But a large GEMM has enormous *potential* reuse: each
element of `A` participates in `N` outputs, each element of `B` in `M`. Optimization
is the art of *capturing* that reuse so each loaded byte feeds many multiply-adds —
raising intensity until the kernel becomes **compute-bound** and the math units, not
the memory bus, set the speed.

```text
  % of cuBLAS    ┌─────────────────────────────── compute-bound ceiling (cuBLAS)
  (illustrative, │                          ╱▔▔▔  +pipeline/vectorize  ~85%
   FP32, one     │                     ╱▔▔▔       +register tiling     ~60%
   well-known    │                ╱▔▔▔            +shared-mem tiling   ~15%
   walkthrough — │           ╱▔▔▔                 +coalescing          ~10%
   shape/HW      │  ▁▁▁▁▁╱                          naive               ~1%
   specific) ────┴────────────────────────────────────────────────────────
                  memory-bound  ───────────────────►  compute-bound
```

(Numbers are illustrative of the *shape* of the climb, from a well-known FP32
single-GPU walkthrough; exact figures are hardware- and shape-specific, which is
the whole reason tuning is empirical — Part C.)

## 5. Rung 1 — memory coalescing

**What changes:** lay out the thread→data mapping so the 32 threads of a warp read
*consecutive* global addresses. **Why it helps:** the hardware then fuses those 32
reads into one wide memory transaction instead of 32 scattered ones — same bytes,
far fewer transactions ([execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)
introduced coalescing). **Knob introduced:** how threads are mapped to output
elements (which index varies fastest within a warp).

## 6. Rung 2 — shared-memory tiling

**What changes:** the block cooperatively loads a *tile* of `A` and a tile of `B`
from global memory into on-chip **shared memory** once, then every thread in the
block reuses those tiles for many multiply-adds before loading the next pair. This
is the tiled matmul built in
[execution model §9–§10](../01_hardware_fundamentals/03_gpu_model.md#10-shared-memory-holds-reusable-tiles)
— I won't re-derive the concept; the tuning layer is what's new here.

```text
  block computes a BM×BN tile of C, marching over K in steps of BK:

     A tile [BM×BK] ─┐
                     ├─ loaded ONCE into shared memory (~30-cycle level),
     B tile [BK×BN] ─┘   then reused by all BM×BN threads
                          → one global load feeds BK multiply-adds
```

**Why it helps:** turns ~500-cycle global reads into ~30-cycle shared-memory reads
for the reused data — the single biggest jump on the ladder. **Knob introduced:**
the **tile sizes `BM`, `BN`, `BK`** — the first and most important dials.

## 7. Rung 3 — register tiling (thread coarsening)

**What changes:** instead of one output per thread, each thread computes a small
**micro-tile** of outputs (say `TM×TN`, e.g. 8×8), holding the accumulators and the
reused operands in **registers** — the ~1-cycle top of the hierarchy
([GPU architecture §4](../01_hardware_fundamentals/04_gpu_architecture.md#4-the-memory-hierarchy)).

```text
  one thread now owns a TM×TN block of C:
     load a sliver of the shared-mem tile into registers,
     reuse each register value across the whole micro-tile
     → many FLOPs per shared-memory read
```

**Why it helps:** climbs one more level — from shared memory to registers — and
raises FLOPs-per-load again, which is what finally pushes a large GEMM toward
compute-bound. **Knob introduced:** the **thread-tile `TM`, `TN`**. Caution: more
registers per thread means fewer threads fit on the SM (occupancy, §13) — the
central tuning tension.

## 8. Rung 4 — vectorized loads and bank conflicts

**What changes:** load 4 floats at once (`float4`, a 128-bit load) instead of one,
and arrange shared-memory accesses to avoid *bank conflicts*. **Bank conflict:**
shared memory is split into 32 *banks*; if multiple threads in a warp hit the same
bank simultaneously, the accesses serialize. Padding or *swizzling* the
shared-memory layout sidesteps this. **Why it helps:** fewer, wider load
instructions and full shared-memory bandwidth. **Knob introduced:** **vector width**,
and the shared-memory padding/swizzle scheme.

## 9. Rung 5 — latency hiding: pipelining and async copy

**What changes:** overlap memory and compute. While the lanes work on the current
tile, *prefetch* the next tile — classically with **double buffering** (two
shared-memory buffers, ping-pong), and on Ampere-and-newer with **async copy**
(`cp.async`, which streams global→shared without occupying the lanes;
[GPU architecture §"What's coming"](../01_hardware_fundamentals/04_gpu_architecture.md#whats-coming)).
**Why it helps:** keeps the math units fed instead of stalling on each tile load —
the same latency-hiding logic as keeping many warps resident
([execution model §8](../01_hardware_fundamentals/03_gpu_model.md#8-many-warps-hide-memory-latency)),
but now also *within* a thread's work. **Knob introduced:** the number of
**pipeline stages** (how many tiles ahead to prefetch).

## 10. Track switch — Tensor Cores (not "rung 6")

Everything above is the **FP32 CUDA-core ladder**. Tensor Cores are not the next
rung on it — they're a *different compute unit* with their own programming model,
and for the FP16 / BF16 / FP8 dtypes LLMs actually run, you **switch tracks** to
them ([execution model §11](../01_hardware_fundamentals/03_gpu_model.md#11-tensor-cores-accelerate-the-matmul-pattern),
[GPU architecture §5](../01_hardware_fundamentals/04_gpu_architecture.md#5-tensor-cores),
[Numerical types](../03_numerical_types/01_floating_point.md)).

```text
  CUDA-core matmul   one scalar multiply-add per lane per cycle    (FP32 ladder)
  Tensor-core matmul one small MATRIX multiply-accumulate per       (different unit;
                     instruction, over fragments of a tile          fp16/bf16/fp8)
```

You still tile (the shared-memory and register ideas carry over), but the inner
compute is now MMA instructions over *fragments* (small per-warp matrix pieces)
rather than scalar FMAs. This is the jump that gives LLM matmuls their headline
throughput — and it's **architecture-gated**: which dtype a Tensor Core supports
depends on the GPU generation (§15).

---

# Part B′ — The other regime

## 11. Decode is a GEMV, and batching turns it back into a GEMM

The entire ladder assumed a large GEMM with reuse to capture. **Decode breaks that
assumption.** Generating one token is a matrix × *vector*: `M = 1` (one token's
activations) against the weight matrix. There's almost nothing to reuse — each
weight is read once and used for one multiply-add — so intensity is ~1 FLOP/byte
and the kernel is **memory-bound no matter how you tile it**
([end-to-end §7](../02_cuda_software_stack/02_end_to_end_inference.md#7-stage-3--decode-the-autoregressive-loop)).

So the optimization target flips:

```text
  large GEMM (prefill)   capture reuse -> compute-bound -> the Part B ladder
  GEMV (decode, M=1)     no reuse to capture -> memory-bound -> optimize for
                         BANDWIDTH: maximal coalescing, and SPLIT-K (split the
                         K-reduction across blocks to put more SMs to work on
                         one skinny problem)
```

And here is the connection that makes this doc and
[Chapter 6](../06_batching/01_batching.md) one idea: **batching stacks `B`
sequences into the `M` dimension.** A batch of `B` decode tokens is a `B×K` times
`K×N` GEMM — `M` grows from 1 to `B`, reuse reappears, and the memory-bound GEMV
becomes a compute-bound GEMM that the Part B ladder optimizes. The throughput lever
of [batching §2](../06_batching/01_batching.md#2-batching-reuses-each-weight-load--intensity-climbs-with-b)
("intensity ≈ B") and the tiling ladder here are *the same fact from two sides*:
batching manufactures the `M` dimension that tiling needs.

---

# Part C — The tuning knobs and autotuning

## 12. The configuration space

Each rung introduced a dial. Collected, they are the kernel's *configuration*:

```text
  BM, BN, BK     shared-memory tile sizes        (rung 2)
  TM, TN         per-thread register micro-tile  (rung 3)
  vector width   float4 / float2 / scalar loads  (rung 4)
  num stages     pipeline / prefetch depth       (rung 5)
  num warps      warps per block (block size)
```

These aren't independent — they trade against each other through one shared budget,
which is the heart of tuning.

## 13. Occupancy: the central tension

Every block draws from the SM's fixed pools: a register file (~256 KB/SM) and
shared memory (~128 KB/SM, more on newer arches)
([GPU architecture §3–§4](../01_hardware_fundamentals/04_gpu_architecture.md#3-streaming-multiprocessors-sms)).
**Occupancy** is resident warps ÷ the SM's max
([execution model §8](../01_hardware_fundamentals/03_gpu_model.md#8-many-warps-hide-memory-latency)).
The tension:

```text
  bigger tiles / micro-tiles  →  more reuse, more FLOPs per load   (good)
                              →  more registers + shared mem per block
                              →  FEWER blocks/warps fit on the SM   (lower occupancy)
                              →  less latency hiding                (bad)
```

Push register tiling too far and values *spill* to local memory (which lives in
global memory at ~500 cycles —
[GPU architecture §4](../01_hardware_fundamentals/04_gpu_architecture.md#4-the-memory-hierarchy)),
erasing the gain. So there's no universally best config — only the best *balance*
of reuse vs. occupancy for a given shape and GPU. Which is why you don't guess it:

## 14. Autotuning: searching the space

Rather than a human hand-picking `BM, BN, ...`, an **autotuner** enumerates
candidate configurations, benchmarks each on the actual shape and GPU, and keeps the
fastest. This is `Triton @autotune`, the CUTLASS profiler, and cuBLASLt's heuristic
search — and it's exactly the *first-call cost* described in
[end-to-end §8](../02_cuda_software_stack/02_end_to_end_inference.md#8-who-chooses-the-kernel-and-when):
the search runs the first time a shape is seen, then the winner is cached (the JIT /
first-run effect, [First-run effects §2](../04_measurement/02_first_run_effects.md#2-jit-compilation)).
Autotuning is why the same Triton kernel is fast across many shapes without a human
re-tuning each one.

## 15. GPU-architecture dependence

A question worth answering directly: **does the kernel perform differently on
different GPUs, and is the architecture itself a tuning input?** Yes to both — and in
two distinct ways.

**Arch as a hard constraint** — it decides which *features and dtypes even exist*
([GPU architecture §5–§6](../01_hardware_fundamentals/04_gpu_architecture.md#5-tensor-cores)):

```text
  native bf16    needs compute capability ≥ 8.0  (Ampere+)
  fp8            needs ≥ 8.9 / 9.0               (Ada / Hopper)
  async cp.async needs Ampere+                    (rung 5 pipelining)
  TMA            needs Hopper                      (faster async bulk copy)
```

A kernel written for fp8 + TMA simply won't run on a T4. This is why the project
uses a T4 only for early modules and an Ampere-or-newer GPU from M7 on.

**Arch as the dominant tuning input** — even where a config is *valid* on two GPUs,
the *best* config differs, because the resource budgets and units differ:

| GPU | Arch | SMs | Max shared mem/SM | Tensor Core gen | Best dtype |
| --- | --- | --- | --- | --- | --- |
| T4 | Turing (7.5) | 40 | up to ~64 KB | 2nd (fp16, int8) | fp16 |
| A100 | Ampere (8.0) | 108 | up to ~164 KB | 3rd (+bf16, tf32) | bf16 |
| H100 | Hopper (9.0) | 132 | up to ~228 KB | 4th (+fp8, TMA) | fp8/bf16 |

(SM counts from
[GPU architecture §7](../01_hardware_fundamentals/04_gpu_architecture.md#7-the-gpus-this-project-uses).
The shared-memory figures are each arch's *maximum* configurable ceiling — not a
contradiction of the ~128 KB *typical* figure in §13 and
[GPU architecture §4](../01_hardware_fundamentals/04_gpu_architecture.md#4-the-memory-hierarchy);
a kernel opts into more shared memory, up to its arch's cap.) Bigger shared memory
permits bigger tiles;
more SMs change the split-K calculus; a newer Tensor Core changes the inner loop
entirely. **So the autotuner must re-search per architecture** — the optimal H100
config is wrong for a T4. "Tuned for the GPU" is literal.

---

# Part D — How the industry does it, and what "optimal" means

## 16. What people actually reach for

The [call-vs-build and automation ladder from the stack doc](../02_cuda_software_stack/01_the_stack.md#what-people-actually-use)
applies directly:

```text
  cuBLAS         the closed, near-peak GEMM — the BASELINE everyone measures against
  CUTLASS        open C++ templates — gets you most of the way to cuBLAS, customizable
  Triton         write + autotune a kernel fast (~80–95% of peak, fraction of effort)
  torch.compile  auto-GENERATE a fused Triton kernel from PyTorch ops (free first try)
  frontier labs  hand-tuned CUDA/CUTLASS for the biggest kernels + exotic features
```

Beyond the core ladder, the highest-end kernels add tricks this doc only names:
**split-K** (§11), **persistent kernels** (keep a block resident and stream many
tiles through it, cutting launch/scheduling overhead), **epilogue fusion** (fold the
bias/activation/scaling into the GEMM's write-back so the result never round-trips to
VRAM — the fusion idea from
[end-to-end §9](../02_cuda_software_stack/02_end_to_end_inference.md#9-how-a-kernel-is-made-fast)),
and **warp specialization / TMA** on Hopper. These are M6/M16 footnotes, not new
rungs.

## 17. Measuring "optimal" — the honest bar

"Optimal" and "KernelBench-worthy" reduce to two checks, in order:

```text
  1. CORRECT   matches a trusted reference across shapes + dtypes      (the §3 gate)
  2. FAST      speedup vs a REAL baseline (cuBLAS / torch.matmul — NOT the naive
               kernel), across many shapes, with proper warm-up and timing (ch4)
```

The trap, from the AI-kernel discourse: a speedup "vs handwritten" that's really vs
the *naive* baseline, or that's measured on an unverified kernel, is not a real
result. Beating the naive kernel is easy; beating cuBLAS is the bar, and it is
genuinely hard — a production GEMM is hundreds of person-months of
architecture-specific work ([Roadmap §9](../ROADMAP.md)). So this doc is the *map*:
M6 builds simplified versions, measures the gap to cuBLAS honestly, and reads the
real implementations to explain it.

## 18. Further reading

The matmul optimization ladder is well-trodden ground; these are the climbs I
found clearest, from a hand-written CUDA worklog up to production GEMM:

- **[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)**
  (Simon Boehm, 2022) — a worklog that starts naive and climbs to ~95% of cuBLAS, one
  optimization per step; the closest match to this doc's ladder (§5–§12), with code.
- **[Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)**
  (NVIDIA) — the GEMM arithmetic-intensity model: how to tell whether a given shape is
  math- or memory-bound (§5, §11) and where Tensor Cores actually help.
- **[Programming Massively Parallel Processors](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)**
  (Hwu, Kirk & El Hajj, 4th ed., 2022) — the textbook derivation of shared-memory and
  register tiling (§6–§7) and occupancy (§12).
- **[Triton: Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)**
  (Triton) — `@triton.autotune` over block / warp / stage configs, the literal version
  of the config-space search in §12–§14.
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** (NVIDIA) — what the top of the ladder
  looks like in practice: the architecture-specific scheduling and Tensor Core pipelines
  a tutorial kernel delegates or omits (§16).

## 19. What to carry forward

```text
the four-part kernel skeleton (§1)               -> M1, the first kernels
write -> verify -> tune (§3)                       -> M0 harness, every kernel
coalescing, memory-bound kernels (§5)            -> M2 (bandwidth)
shared-memory tiling + register tiling (§6–§7)   -> M6 (naive → tiled matmul)
Tensor Cores for LLM dtypes (§10)                -> M6, M15 (quantized dtypes)
occupancy + autotuning the config space (§12–14) -> M6, M8 (deep profiling)
GPU-arch dependence (§15)                        -> M21 (arch comparison)
the GEMV/decode regime + split-K (§11)           -> M11, and FlashAttention M16
```

The one sentence to keep: **a kernel is "identify your data → load → compute →
store," and tuning it is a disciplined climb up the memory hierarchy — coalesce,
tile into shared memory, tile into registers, vectorize, pipeline, and (for LLM
dtypes) switch to Tensor Cores — capturing enough reuse to turn a memory-bound
matmul compute-bound, with the exact tile configuration chosen empirically per shape
and per GPU, and every speedup gated on correctness first.**
