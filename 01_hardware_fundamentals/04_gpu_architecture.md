# GPU architecture

This doc is the reference companion to [The GPU execution model](03_gpu_model.md).
03 builds up the conceptual story of how a GPU runs work — array example,
thread, block, warp, SM, kernel launch, shared memory, Tensor Cores, HBM. This
doc holds the lookup-style content that doc deliberately keeps out of the
build-up: the memory hierarchy in concrete latency/bandwidth numbers, the
Tensor Core generation/dtype mapping, NVIDIA's compute-capability numbering,
and the specific GPUs this project uses.

Read 03 first. The sections below pick up where 03 left off and assume those
concepts are already in hand.

If words like core, register, L1, L2, cache line, SRAM, DRAM, HBM, or GDDR
are new, start earlier still: [Circuits and cores](01_circuits_and_cores.md)
and [Memory and caches](02_memory_and_caches.md).

## 1. Why a GPU is different from a CPU

The conceptual answer, with diagrams and the CPU/GPU side-by-side chip-level
comparison, lives in [The GPU execution model §1](03_gpu_model.md#1-the-array-example-exposes-parallel-work)
and [§7](03_gpu_model.md#7-cpu-and-gpu-at-the-chip-level-side-by-side).
Short version: a CPU spends transistors on making one instruction stream
fast (branch prediction, out-of-order execution, large caches); a GPU spends
them on many parallel lanes plus high memory bandwidth, and assumes you have
enough independent work to keep all of them busy.

One naming convention worth highlighting here, since later sections use it:
a GPU **kernel** is a single function executed in parallel by thousands of
threads. The CUDA syntax for declaring and launching one (`__global__`,
`<<<num_blocks, threads_per_block>>>`) is walked through in
[03 §2](03_gpu_model.md#2-gpu-threads-map-work-to-data).

## 2. The thread hierarchy

The full build-up — grid → block → warp → thread, including why both block
and warp exist as separate groupings — is in [03 §2–§6](03_gpu_model.md#2-gpu-threads-map-work-to-data).
A one-line refresh:

```text
grid       all threads from one kernel launch
  block      programmer's grouping; runs entirely on one SM;
             cooperates via shared memory and barriers
    warp       hardware's grouping; 32 threads issued together
      thread     one element of work; one SM lane for one cycle
```

One CUDA-side name that 03 does not use: a block is also called a **CTA**
(Cooperative Thread Array) in NVIDIA's hardware documentation, in PTX
assembly, and in Nsight profiler output. "Block" and "CTA" mean exactly the
same thing — the user-chosen group of threads that lives on one SM. The
term shows up often enough in NVIDIA docs and tooling that it is worth
recognising on sight.

## 3. Streaming Multiprocessors (SMs)

What an SM is and what it contains is covered in [03 §4](03_gpu_model.md#4-the-sm-is-the-gpu-execution-unit);
why the SM keeps many warps resident, and the latency-hiding mechanism that
relies on it, is in [03 §8](03_gpu_model.md#8-many-warps-hide-memory-latency).

One metric that 03 only name-drops, called out here because it is the
single most-cited SM-level number in profiler output: **occupancy**. It is
the ratio of resident warps on an SM to the maximum that SM can hold:

```text
occupancy = resident warps on the SM / max resident warps the SM supports
```

A kernel can fall short of the maximum because each block consumes
registers and shared memory from fixed per-SM pools (see [03 §8](03_gpu_model.md#8-many-warps-hide-memory-latency)
for the numbers on H100). If a kernel asks for more registers per thread
or more shared memory per block, fewer blocks fit on the SM at once, and
occupancy drops.

Occupancy shows up as a column in **Nsight Compute**, NVIDIA's per-kernel
profiler (see [Benchmarking §benchmark-vs-profile](../04_measurement/01_benchmarking.md#benchmark-vs-profile)).
High occupancy is neither necessary nor sufficient for high performance,
but very low occupancy usually points to an underutilised SM — either too
few threads launched, or each thread asking for too many resources.

## 4. The memory hierarchy

03 introduces each layer of GPU memory in its own section: registers and
shared memory in [§10](03_gpu_model.md#10-shared-memory-holds-reusable-tiles),
the cross-SM L2 in [§12](03_gpu_model.md#12-l2-connects-the-sms-to-gpu-memory),
and off-chip HBM/GDDR in [§13](03_gpu_model.md#13-hbm-and-gddr-are-large-off-chip-memory).
This section is the side-by-side lookup table — all four layers in one place,
with concrete latency and bandwidth numbers attached.

The "Latency (cycles)" column is in *clock cycles* — one tick of the GPU's
clock. At a typical ~1.4 GHz GPU clock, one cycle is about 0.7 ns, so the
~500-cycle global-memory latency is around 350 ns; the ~30-cycle shared-memory
latency is around 20 ns.

| Level | Scope | Typical size | Latency (cycles) | Bandwidth |
| --- | --- | --- | --- | --- |
| Registers | Per thread | ~256 KB total per SM | ~1 | — |
| Shared memory / L1 | Per block (shared); per SM (L1) | ~128 KB per SM | ~30 | several TB/s |
| L2 cache | Per GPU | 4–60 MB | ~200 | ~3 TB/s |
| Global (HBM / GDDR) | Per GPU | 16–80 GB | ~500 | 300 GB/s – 3 TB/s |

```
  Faster, smaller, on-chip                          on-chip
   ▲    ┌───────────────────────────────────────┐   │
   │    │             REGISTERS                  │  per thread     ~1 cyc
   │    ├───────────────────────────────────────┤
   │    │       SHARED MEMORY  /  L1            │  per block / SM  ~30 cyc
   │    ├───────────────────────────────────────┤
   │    │             L2 CACHE                   │  per GPU        ~200 cyc
   │    ├───────────────────────────────────────┤
   │    │   GLOBAL MEMORY  (HBM / GDDR)          │  per GPU        ~500 cyc
   ▼    └───────────────────────────────────────┘   │
  Slower, larger, off-chip                          off-chip
```

A few practical consequences these numbers point at:

- **Registers** are where the action happens. If a kernel uses too many
  per thread, excess values spill to *local memory* — which despite the
  name lives in global memory, so a spilled access costs the full ~500
  cycles instead of ~1. This is called **register spilling** and is a
  major performance hit.
- **Shared memory** is software-managed. Unlike a CPU cache, the kernel
  decides exactly what gets loaded and when. Almost every fast GPU kernel
  uses it deliberately (see the tiled-matmul walkthrough in [03 §10](03_gpu_model.md#10-shared-memory-holds-reusable-tiles)).
- **L2** is hardware-managed and shared across the whole GPU. Small
  benchmark inputs often fit entirely in L2, which is why microbenchmarks
  systematically overestimate cache hit rates compared to production
  workloads — a subtlety to watch for when interpreting numbers.
- **Global memory** bandwidth is one of the most-cited numbers about a
  GPU. An A100 has ~1.5 TB/s of HBM bandwidth; an H100 has ~3 TB/s. If
  a kernel achieves only 10% of that, the profiler will say so — and
  most LLM-inference kernels are memory-bound, so that fraction directly
  drives end-to-end speed.

## 5. Tensor Cores

What Tensor Cores do (small-tile matrix multiply-accumulate in one
instruction, instead of one scalar MAC per cycle on a regular FP lane) is
walked through in [The GPU execution model §11](03_gpu_model.md#11-tensor-cores-accelerate-the-matmul-pattern),
along with why MAC is the atomic step of matmul. This section is the
generation table — which architecture introduced which Tensor Core, and
which numerical formats each generation added support for.

A short note on the "Volta", "Turing", "Ampere", etc. names that recur across
this and later docs: NVIDIA names each successive GPU architecture family
after a physicist or scientist. The chronological order — Tesla, Fermi,
Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Ada (Lovelace), Hopper,
Blackwell — matches the generations and their compute capability numbers
listed in §6 below ("Compute capability"). Codenames identify the silicon
family and the feature set
(which dtypes, which Tensor Core generation, which instructions).

Tensor Cores have evolved across generations:

| Generation | Architecture | Compute capability | Dtypes added |
| --- | --- | --- | --- |
| 1st | Volta | 7.0 | fp16 |
| 2nd | Turing | 7.5 | int8, int4 |
| 3rd | Ampere | 8.0 / 8.6 | bf16, tf32, fp64 |
| 4th | Ada / Hopper | 8.9 / 9.0 | fp8 |
| 5th | Blackwell | 10.0+ | fp4 |

The "Dtypes added" column lists fp16, bf16, tf32, fp8, fp4, etc. — the
numerical formats LLMs use. See [Numerical types](../03_numerical_types/01_floating_point.md)
for what each one is, how wide its exponent and mantissa are, and where each
shows up in modern LLM workflows.

Hopper also introduced the **Transformer Engine**, hardware-assisted dynamic
scaling between fp8 and fp16 for transformer layers.

This is why the project uses a T4 only for early modules — Turing Tensor Cores
do not support **bf16**, which most modern LLMs are trained in. From M7 onward
the project requires an Ampere-or-newer GPU (compute capability ≥ 8.0).

## 6. Compute capability

NVIDIA versions each GPU architecture with a number called **compute
capability**, formatted as `major.minor`. The major number identifies the
architecture family; the minor identifies the variant within it.

| CC | Architecture | Example GPUs |
| --- | --- | --- |
| 7.0 | Volta | V100 |
| 7.5 | Turing | T4, RTX 2080 |
| 8.0 | Ampere (datacenter) | A100 |
| 8.6 | Ampere (consumer) | RTX 3090 |
| 8.9 | Ada | L4, RTX 4090 |
| 9.0 | Hopper | H100, H200 |
| 10.0+ | Blackwell | B100, B200 |

Compute capability matters because:

- Certain instructions exist only on certain CCs (native bf16 needs ≥ 8.0;
  fp8 needs ≥ 8.9 or 9.0).
- CUDA code compiled for one CC will not necessarily run on others without
  re-JITing. *JIT* = just-in-time compilation: turning code into runnable
  GPU instructions on first use rather than ahead of time; *re-JIT* = the
  driver doing this at load time because the binary's pre-compiled SASS does
  not match the GPU at hand. Detailed treatment in
  [First-run effects §2](../04_measurement/02_first_run_effects.md#2-jit-compilation).
- Libraries like CUTLASS (NVIDIA's open-source C++ template library for
  building fast matmul/GEMM kernels) and FlashInfer (an open-source library
  of fused attention and other transformer kernels used by serving engines)
  ship kernels specialized per CC; an unsupported CC falls back to a slower
  path or fails outright.

## 7. The GPUs this project uses

| GPU | CC | Architecture | SMs | VRAM | Bandwidth | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| T4 | 7.5 | Turing | 40 | 16 GB GDDR6 | 320 GB/s | Free on Colab. No bf16, no fp8. |
| L4 | 8.9 | Ada | 58 | 24 GB GDDR6 | 300 GB/s | Cheap datacenter Ada. |
| RTX 4090 | 8.9 | Ada | 128 | 24 GB GDDR6X | 1008 GB/s | Consumer flagship; high bandwidth. |
| A100 | 8.0 | Ampere | 108 | 40 or 80 GB HBM2e | 1555 / 2039 GB/s | Datacenter workhorse. |
| H100 | 9.0 | Hopper | 132 | 80 GB HBM3 | 3350 GB/s | fp8, Transformer Engine. |

The architecture comparison in M21 is designed to highlight the differences
across this set: memory bandwidth, Tensor Core generation, and dtype support.

A note on the variant labels in that table — "GDDR6", "GDDR6X", "HBM2e",
"HBM3": the numbers and letters are generation markers. Higher numbers mean
newer revisions with more bandwidth and density. The "X" in GDDR6X is a
proprietary signaling variant developed by Micron with NVIDIA, faster than
plain GDDR6. The "e" in HBM2e is JEDEC's marker for an enhanced revision of
HBM2 with higher per-pin signaling. HBM3 is the next generation again.

## What's coming

Topics this doc will pick up as the project gets there:

- Warp scheduling and instruction issue policy in detail
- Asynchronous copies (`cp.async`, TMA) and how Ampere/Hopper overlap memory
  and compute
- Distributed shared memory and Hopper Thread Block Clusters
- NVLink, NVSwitch, and the multi-GPU interconnect hierarchy
- Power, thermal, and clock management in depth
- The relationship between compute throughput (TFLOPS) and memory bandwidth —
  the roofline model
