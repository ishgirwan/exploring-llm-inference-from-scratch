# Benchmarking methodology

A benchmark is a claim — "this kernel takes X microseconds" — and is only
useful if it reproduces. GPU benchmarking is harder than it looks. This
document defines the rules every benchmark in this repo follows.

## Why GPU benchmarking is tricky

Three properties of GPU execution make naive benchmarks unreliable.

- **Asynchronous execution.** CUDA calls return to Python before the GPU has
  finished the requested work; the work is queued, not done. A wall-clock timer
  measures queuing time, not execution time.
- **Variable clock speed.** A GPU's core clock boosts when the chip is cool and
  *throttles* (automatically reduces its frequency to stay within thermal or
  power limits) when it is hot or power-limited. Identical code can run
  20–30% faster on the first measurement than on the hundredth.
- **First-run effects.** Memory allocation, just-in-time (JIT) kernel
  compilation, autotuning, cold caches, and clock ramp all happen on the first
  few invocations of a kernel and never again. See
  [First-run effects](02_first_run_effects.md) for the full breakdown.

## The rules

`common/bench.py` enforces these automatically.

1. Correctness before speed. A kernel must pass its correctness test before it
   is benchmarked.
2. Time on the GPU (CUDA events), not on the CPU (wall clock).
3. Synchronize before stopping the clock.
4. Warm up first; discard early runs.
5. Run many times and report percentiles (p50, p95, p99 — defined below).
   Never a bare mean.
6. Lock GPU clocks when permissions allow.
7. Record the environment with every result.
8. Save results as structured JSON in a single schema.

## Correctness

Every kernel is checked in `tests/` against a trusted reference, almost always
the equivalent PyTorch operation, using dtype-aware tolerances.

- **`rtol`** — *relative tolerance.* Bounds the proportional difference per
  element.
- **`atol`** — *absolute tolerance.* Bounds the raw difference per element.

Two values pass when `|a − b| ≤ atol + rtol · |b|`.

Floating-point dtypes have intrinsic precision limits (see
[Numerical types](../03_numerical_types/01_floating_point.md)):

| dtype | what it is | rtol | atol |
| --- | --- | --- | --- |
| **fp32** | 32-bit IEEE float; full precision | 1e-5 | 1e-6 |
| **fp16** | 16-bit half precision; ~3 decimal digits, narrow range | 1e-2 | 1e-3 |
| **bf16** | brain-float 16; same exponent range as fp32 but only ~2–3 mantissa digits | 1e-2 | 1e-2 |

Use `torch.testing.assert_close(actual, expected, rtol=..., atol=...)` with
explicit tolerances. The defaults in `torch.allclose` are simultaneously too
loose for fp32 and too strict for fp16, and silently accept incorrect results.

A bf16 kernel that matches an fp32 reference to `1e-5` is almost certainly
upcasting to fp32 internally and is not exercising the dtype it claims to. The
tolerance encodes the precision the dtype can actually deliver.

Tests must also include **non-contiguous** inputs — tensors whose memory layout
is not packed (for example a transposed view). Most "works in the benchmark,
breaks in a real model" bugs are *stride bugs*, where the kernel assumed
contiguous memory and was handed something else.

## Timing on the GPU

The naive wall-clock timer is wrong:

```python
# WRONG — measures queuing the work, not doing it
t0 = time.perf_counter()
my_kernel(x)            # returns before the GPU has done anything
dt = time.perf_counter() - t0
```

A **CUDA event** is a marker recorded into the GPU's own command stream. Events
time GPU work directly:

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
my_kernel(x)
end.record()
torch.cuda.synchronize()            # wait for the GPU to reach `end`
dt_ms = start.elapsed_time(end)     # milliseconds, measured by the GPU
```

What the two approaches actually measure:

```
  WRONG — wall-clock around an async call

  Python:   t0 ── my_kernel(x) ── t1                   t1 − t0 reported
                       │
                       └─ returns immediately
                          (work just queued)
  GPU:                          ░░░░ kernel actually runs ░░░░ ─► done
                                                                  ▲
                                                       not captured by the timer


  RIGHT — CUDA events with synchronize

  Python:  start ── my_kernel(x) ── end ── sync ── elapsed_time(start, end)
            │                        │       │
            └─ event marker          │       └─ wait until the GPU has reached `end`
                queued for GPU       └─ event marker queued
  GPU:           ░░░░ kernel runs ░░░░ ─► done
                 ▲                       ▲
                 start marker passes      end marker passes
                 (this is when the timer actually starts on the GPU)
```

## Synchronization

`torch.cuda.synchronize()` blocks the CPU thread until the GPU has finished all
queued work. It is required:

- **Before** starting a timer, so earlier work is not included in the next
  measurement.
- **After** the measured work, before reading the result. Without this, the
  clock stops while the GPU is still executing. This is the most common source
  of fictitious speedups in beginner GPU benchmarks.

## Warm-up

The first few invocations of a kernel pay one-time setup costs — memory
allocation, JIT compilation (e.g. **Triton**, a Python domain-specific
language for writing GPU kernels), autotuning, cold caches, and clock ramp —
that never recur. The
harness runs 10–25 warm-up iterations and discards them. The detailed mechanism
behind each cost is in [First-run effects](02_first_run_effects.md).

## Percentiles, not means

Even after warm-up, individual runs vary due to scheduler jitter and clock
drift. The harness runs each measurement 100+ times and reports the
distribution:

- **p50** (median) — the typical case; half of runs were faster.
- **p95 / p99** — the slowest 5% and 1% of runs.

The mean is not reported. In a serving system the tail latency is what an end
user experiences; the mean averages it away. A large gap between p99 and p50
indicates instability — thermal throttling, a noisy co-tenant on shared cloud
hardware, or a real bug.

## Clock locking

GPU clocks vary by tens of percent between cold and hot states. To remove this
confounder, pin the clock for the duration of the benchmark:

```bash
nvidia-smi -lgc <frequency>   # lock graphics clock
nvidia-smi -rgc               # release
```

The harness applies the lock when permissions allow (typically on rented
bare-metal instances; usually not on Colab or other shared environments) and
records whether the lock was applied. Unlocked-clock runs are still useful but
show a wider spread and require a disclaimer in the report.

## Environment

Every result records:

- GPU model and **[compute capability](../01_hardware_fundamentals/04_gpu_architecture.md#6-compute-capability)**
  (the version identifying the GPU's hardware feature set; e.g. `7.5` for a T4,
  `8.9` for an L4)
- NVIDIA driver and CUDA runtime versions (see
  [The CUDA software stack](../02_cuda_software_stack/01_the_stack.md))
- PyTorch, Triton, and where relevant vLLM / SGLang versions (vLLM and SGLang
  are open-source LLM inference engines; both ship behaviour-changing releases
  approximately monthly)
- the `setup/docker/` container tag the run used
- clock-lock state
- input shape and dtype
- a UTC timestamp

The pinned Docker container is mandatory for benchmarks intended to be
reproducible.

## Results schema

Each measured configuration produces one JSON object in `benchmarks/results/`.
The schema is defined in `common/results_schema.py`. Plots in
`benchmarks/plots/` are generated from these files; no plot is hand-drawn. A
single schema is enforced so that results from any two modules — for example
M4 (RMSNorm) and M20 (multi-GPU) — can be loaded and compared by the same code
in the capstone.

```json
{
  "name": "rmsnorm_triton",
  "module": "M4",
  "timestamp_utc": "2026-05-22T10:30:00Z",
  "environment": { "gpu": "NVIDIA L4", "cuda": "12.4", "torch": "2.x.y",
                   "container_tag": "llm-gpu-lab:0.1", "clocks_locked": true },
  "config": { "shape": [4096, 4096], "dtype": "bf16" },
  "correctness": { "passed": true, "rtol": 1e-2, "atol": 1e-2 },
  "timing": { "warmup_iters": 25, "measured_iters": 200,
              "p50_us": 41.2, "p95_us": 43.8, "p99_us": 47.1 },
  "derived": { "achieved_bandwidth_gb_s": 612.0, "pct_of_peak": 78.0 }
}
```

The `correctness.passed` field is part of the result. The timing of a kernel
that failed its correctness test is never plotted as a real number.

`pct_of_peak` is the headline efficiency number: `achieved_bandwidth /
peak_bandwidth × 100`, where the *peak* is the GPU's nameplate maximum
bandwidth (e.g. ~1.55 TB/s for an A100 40 GB, ~3.35 TB/s for an H100). A
well-tuned memory-bound kernel typically reaches 70–90% of peak; under 30%
usually points to a memory-access bug (uncoalesced reads, the wrong stride,
or running on a too-small input that doesn't saturate the bus).

## Benchmark vs profile

Benchmarking answers *how fast?* and produces a number. Profiling answers
*why that fast?* and produces a trace or counter readout.

- **Nsight Systems** — NVIDIA's system-wide profiler. Produces a timeline of
  CPU and GPU activity. Introduced in M2.5.
- **Nsight Compute** — NVIDIA's per-kernel profiler. Reads hardware performance
  counters (instruction throughput, memory traffic, warp stall reasons) and
  requires elevated permissions, which are typically unavailable on shared
  free-tier environments. Introduced in M8.

Benchmark every topic. Profile when a benchmark surprises.

## Is the GPU actually busy?

A GPU's memory being full does **not** mean it is computing. Memory occupancy
and compute activity are independent axes. PyTorch's caching allocator
([First-run effects §1](02_first_run_effects.md#1-the-caching-allocator)) grabs a
large pool from the driver on the first allocation and holds it, so `nvidia-smi`
can report tens of gigabytes in use while the GPU runs no arithmetic at all. The
memory column answers *is the GPU reserved?* — never *is it working?*

The activity number `nvidia-smi` prints — **GPU utilization** (the `GPU-Util`
column) — is a notorious trap. NVIDIA defines it as *the percent of time over the
sample window during which one or more kernels was executing*: it measures
**time, not capacity**. A single kernel using 1% of the GPU's
[SMs](../01_hardware_fundamentals/04_gpu_architecture.md#3-streaming-multiprocessors-sms)
(Streaming Multiprocessors — the GPU's independent compute units) but running
continuously reports **GPU-Util 100%**. So 0% is informative (nothing ran,
genuinely idle), but 100% is *not* evidence the chip is full — only that it was
never completely idle.

Three signals, cheapest to most precise:

- **Power and clocks** (`nvidia-smi -q -d POWER,CLOCK`) — the cheapest *honest*
  signal, because arithmetic costs watts. An idle GPU sits near its power floor
  in a deep idle state ([First-run effects §5](02_first_run_effects.md#5-gpu-power-and-clock-state));
  a working one pulls toward its **TDP** (Thermal Design Power — the maximum
  sustained wattage the cooling is rated for, e.g. 300–700 W on a data-center
  card) with boosted clocks. Memory full but power at the floor ⇒ nothing is
  computing.
- **GPU-Util** — trust only its zero. Read any non-zero value as "a kernel was
  alive," not "the GPU was full."
- **SM, Tensor-Core, and memory-bandwidth activity** — the real capacity numbers,
  read from hardware counters by **Nsight Compute** (per kernel; see
  [Benchmark vs profile](#benchmark-vs-profile)) or **DCGM** (NVIDIA's Data
  Center GPU Manager — a monitoring daemon that exports these as live metrics).
  For an LLM the
  [Tensor-Core](../01_hardware_fundamentals/04_gpu_architecture.md#5-tensor-cores)
  figure matters most: matmuls run there, so low Tensor-Core activity on a
  matmul-bound workload means the GPU is busy but not busy *well*.

The memory half of the same confusion is measurable from PyTorch directly:
`torch.cuda.memory_reserved()` is what the caching allocator took from the driver
(what `nvidia-smi` shows), while `torch.cuda.memory_allocated()` is what live
tensors actually hold. A large gap is reserved-but-empty pool — memory booked,
not even storing data, let alone computing on it.

## Pre-flight checklist

A benchmark result is a draft until every box is checked:

- [ ] Synchronized before stopping the clock.
- [ ] Warm-up runs discarded.
- [ ] Reporting a distribution (p50/p95/p99), not a single run.
- [ ] Clocks locked, or the report discloses they were not.
- [ ] Correctness passed for this exact configuration.
- [ ] GPU otherwise idle — power at its floor, not merely low memory use (see
      [Is the GPU actually busy?](#is-the-gpu-actually-busy)).
- [ ] Same dtype and shape on both sides of any comparison.
- [ ] Environment recorded.

## Reproducing a result

1. Provision a GPU instance with `setup/provision_*.sh` and the pinned
   container.
2. Run the benchmark script in the relevant module's `labs/` directory.
3. Compare the resulting JSON against the committed file.

Absolute numbers differ across hardware. The *shape* of the result — which
kernel is faster, by roughly what factor, and why — should reproduce. A failure
to reproduce that shape is itself a finding and warrants an issue.
