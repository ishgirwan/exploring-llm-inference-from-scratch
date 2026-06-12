# First-run effects

The first few times a GPU kernel runs in a process, several one-time setup
costs are paid that never recur. These costs can be orders of magnitude larger
than the kernel's steady-state runtime. This document explains what those costs
are and why the benchmarking harness discards the first 10–25 iterations of
every measurement.

Prerequisites: a basic understanding of the [GPU memory
hierarchy](../01_hardware_fundamentals/04_gpu_architecture.md#4-the-memory-hierarchy).

## 1. The caching allocator

PyTorch does not call `cudaMalloc` for every tensor. It runs a **caching
allocator**: a layer that requests large chunks of GPU memory from the driver
and then hands out and reuses pieces of those chunks internally.

The first allocation of a given size triggers a real `cudaMalloc`, which is
synchronous (stalls the CPU until completion), may have to grow the allocator's
pool, and may trigger fragmentation handling — *fragmentation* meaning free
memory exists in total but only in pieces too small to satisfy the request,
so the allocator has to find or create a large-enough contiguous block. This
typically costs milliseconds.

Subsequent allocations of the same size are served from the pool in
microseconds. The first measurement of a kernel is therefore inflated by
allocation costs the second measurement never sees.

## 2. JIT compilation

Most kernels in this repo are compiled **just-in-time** — the kernel source is
not compiled at install time but the first time the kernel is actually called.

- **Triton** kernels are written in Python. On first call with a given set of
  compile-time constants (block sizes, dtypes, etc.), Triton lowers the kernel
  to PTX (Parallel Thread Execution — NVIDIA's portable, GPU-independent
  virtual assembly) and then to SASS (the hardware instruction set for the
  specific GPU the code will run on; NVIDIA uses the acronym without
  consistently expanding it). PTX is what gets shipped inside a CUDA binary;
  SASS is what the GPU actually executes, and the driver produces SASS from
  PTX at load time if a matching SASS is not already cached. This compile
  commonly takes 100 ms to several seconds. The result is cached on disk in
  `~/.triton/cache`, so a fresh process can skip it — but within one process,
  the first call pays the full cost.
- **`torch.compile`** traces a model, lowers it, and generates code on first
  call. The same order of magnitude as Triton, sometimes larger.
- **Hand-written CUDA C++** kernels are normally compiled *ahead of time*
  (AOT — the opposite of JIT; compiled once at build time into a binary that
  ships with the application). If the binary targets the wrong
  [compute capability](../01_hardware_fundamentals/04_gpu_architecture.md#6-compute-capability) for the
  GPU it runs on, the driver re-JITs them at load time. This is invisible
  from the application and appears as a slow first run.

A 500 ms JIT compile shows up as "this kernel is 5000× slower than the
reference" if no warm-up is performed. The reference (PyTorch's built-in op) is
typically precompiled C++ or a cuDNN kernel and pays no JIT cost.

## 3. Autotuning

Some kernels exist as a family of implementations parameterized by block size,
number of warps, *pipeline stages* (how many independent loop iterations
overlap in flight before a barrier — more stages can hide memory latency
better but use more registers per thread), and so on. **Autotuning** runs
several candidate configurations on the actual input shape and picks the
fastest.

- **`@triton.autotune`** runs each candidate on first call for a given input
  shape, times them all, picks the winner, and caches the choice by shape key.
- **cuDNN** behaves similarly when `torch.backends.cudnn.benchmark = True`,
  trying alternative algorithms before committing to one. The first call with
  each new shape is much slower as a result.
- **cuBLAS** uses heuristics (rules of thumb that pick a likely-good
  configuration without trying every option) rather than exhaustive search,
  but still has lazy first-call initialization.

A single "first call" can therefore correspond to many hidden kernel launches
plus the timing logic between them. The cached winner is reused on subsequent
calls and is genuinely fast.

## 4. Cold caches

GPUs have per-SM L1 caches, a chip-wide L2 cache, and an instruction cache. All
start empty.

- First call: input data lives in global memory (HBM or GDDR). Reads pay the
  full memory latency. The kernel's compiled binary is loaded into the
  instruction cache as it executes.
- Subsequent calls: if the input fits in L2 — small benchmark inputs often do —
  reads come from L2 at higher effective bandwidth. The instruction cache is
  already warm.

Cache state also depends on what ran immediately before. A kernel called in
isolation has different cache hit rates from the same kernel called as part of
a real workload. This is one reason microbenchmarks tend to overestimate cache
hit rates compared to production.

## 5. GPU power and clock state

An idle GPU sits in a low-power state (NVIDIA labels these `P0` through `P12`,
where `P0` is full performance and `P8`–`P12` are deep idle). The first compute
work transitions the GPU upward, which takes tens to a few hundred milliseconds.
The clock then ramps from a low frequency to its boost frequency over the first
several iterations.

This effect is independent of all the others and applies even to AOT-compiled
kernels with cached configurations. Forcing the GPU to a fixed clock with
`nvidia-smi -lgc` removes it — see
[Benchmarking §clock-locking](01_benchmarking.md#clock-locking).

## 6. CUDA context initialization

A *CUDA context* is a per-process bundle of GPU state: the loaded kernel
binaries, the caching allocator's pool, the active CUDA streams and events,
and the device handles the process owns. Conceptually it is to a CUDA
program what an OS process control block is to a CPU program — the
bookkeeping the runtime maintains so the GPU knows which work belongs to
which process.

The very first CUDA call in a Python process is special. It triggers driver
initialization, CUDA runtime setup, creation of the per-device CUDA context,
and on-demand loading of libraries such as cuBLAS, cuDNN, and NCCL. The cost is
typically several hundred milliseconds to a couple of seconds, depending on how
many libraries are pulled in.

PyTorch makes this lazy: the cost is paid on the first `torch.cuda.*` call.
Subsequent calls are immediate.

## 7. How warm-up addresses these

The warm-up phase in `common/bench.py` runs the kernel some configurable number
of times (default 10–25) before the timing clock starts, discarding the
results. By the time real measurement begins:

- the allocator pool is sized correctly and does not call `cudaMalloc`
- Triton and `torch.compile` have produced and cached compiled binaries
- autotuning has selected and cached a configuration
- L2 and the instruction cache hold the relevant data
- the GPU has transitioned to a boosted power state
- the CUDA context is fully initialized

The 10–25 iteration band covers a JIT compile, a small autotune sweep, and
clock ramp without wasting too much sweep time. Faster kernels (where
per-iteration cost is small relative to setup) benefit from the higher end of
the range; large kernels can use the lower end.

Visually, what warm-up subtracts from the measurement:

```
  Without warm-up — timing the whole thing:

   cost │ ████  ← JIT compile + cudaMalloc + autotune (iter 1)
        │ ███   ← caching allocator + clock ramp (iter 2–3)
        │ ██    ← cache warm-up (iter 4–5)
        │ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █  ← steady state
        └───────────────────────────────────────────────► iteration
          1 2 3 4 5                            ...


  With warm-up — timing only the steady state:

           ┌── discarded warm-up ──┐  ┌── measured ──────────────┐
   cost │ ████                                                    │
        │ ███                                                     │
        │ ██                                                      │
        │ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █  ← what gets reported
        └───────────────────────────────────────────────► iteration
          1 ........ ~25 ......................
```

The "what gets reported" iterations are the ones the harness times and feeds
into the p50/p95/p99 distribution.

## 8. Diagnostic technique

To distinguish which effect dominates an unexpectedly slow first run, time
iterations 1, 5, 10, and 100 separately and observe the convergence.

| Pattern | Likely cause |
| --- | --- |
| Iteration 1 is 100×–10000× slower; iteration 2+ are normal | JIT compilation |
| Iterations 1–5 progressively faster | Caching allocator + clock ramp |
| Iterations 1–10 noisy; 11+ stable | Autotuning + cache warm-up |
| Iteration 1 very slow even for precompiled code | CUDA context initialization (only if first kernel of the process) |

The shape of the convergence indicates which effect dominates, which in turn
indicates whether the warm-up count should be increased for that kernel.

## 9. Further reading

The primary sources behind each first-run effect, and the tools that hide them:

- **[CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)** (PyTorch)
  — documents the caching allocator (§1) and the lazy CUDA-context initialization (§6)
  that make a first call mysteriously slow.
- **[Triton: Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)**
  (Triton) — shows JIT compilation (§2) and `@triton.autotune` (§3) in action: the
  first call pays to compile and to search the config space.
- **[How to Implement Performance Metrics in CUDA C/C++](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)**
  (Mark Harris, NVIDIA) — the timing discipline that turns warm-up (§7) and the
  iteration-by-iteration diagnostic (§8) into something measurable rather than guessed.
- **[torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html)**
  (PyTorch) — a `Timer` that runs its own warm-up and synchronization, the practical
  antidote to nearly every effect in §1–§7.
