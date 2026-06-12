# Reading PTX and acting on it: the profile-and-optimize loop

[The source-to-SASS doc](03_source_to_sass.md) showed the *pipeline* and what PTX
*is*. This doc is the *skill*: read an actual PTX trace, profile to find the
bottleneck, decide the change, and make it — usually in the source, occasionally in
PTX. It ends on a full worked loop: **understand the PTX → profile → diagnose →
change → re-measure.** This is the prework for **M8** (deep profiling) and the
optimization steps of **M24/M27/M30**.

**Honesty up front.** No local NVIDIA GPU here (remote-only), so the numbers in the
worked example are *illustrative of what you'd expect to see*, not measured — the
**method** is the deliverable; the real numbers come at M8 on a rented GPU, against
the [harness](../04_measurement/README.md). PTX snippets are representative
(simplified) of what `cuobjdump -ptx` emits.

Prerequisites: [source-to-SASS](03_source_to_sass.md) (the formats, the dump
commands), [Chapter 4 — Measurement](../04_measurement/README.md) (benchmarking, the
roofline), and [the matmul tuning ladder](../07_writing_and_tuning_kernels/01_writing_and_tuning_a_matmul.md)
(coalescing, vectorization, occupancy). A bare **§N** means *this doc's* section N.

---

## 1. Anatomy of a PTX kernel

Here is a representative PTX dump of a simple `axpy` kernel (`y[i] = a*x[i] + y[i]`)
— the kind of thing `cuobjdump -ptx` gives you, annotated:

```ptx
.visible .entry axpy(.param .u64 x, .param .u64 y,           // kernel + its params
                     .param .f32 a, .param .u32 n)
{
    .reg .pred  %p<2>;                  // register DECLARATIONS, by type:
    .reg .f32   %f<5>;                  //   %p = predicate (bool), %f = 32-bit float,
    .reg .b32   %r<6>;                  //   %r = 32-bit, %rd = 64-bit (addresses)
    .reg .b64   %rd<7>;

    ld.param.u64   %rd1, [x];           // read params from PARAM memory space
    ld.param.f32   %f1,  [a];
    ld.param.u32   %r1,  [n];

    mov.u32        %r2, %ctaid.x;       // blockIdx.x   ┐
    mov.u32        %r3, %ntid.x;        // blockDim.x   ├ i = blockIdx*blockDim + threadIdx
    mov.u32        %r4, %tid.x;         // threadIdx.x  ┘
    mad.lo.s32     %r5, %r2, %r3, %r4;  // mad = multiply-add: %r5 = %r2*%r3 + %r4

    setp.ge.s32    %p1, %r5, %r1;       // %p1 = (i >= n)         ┐ the bounds check
    @%p1 bra       DONE;                // if %p1, branch to DONE ┘ (predicated branch)

    cvta.to.global.u64 %rd3, %rd1;      // convert generic → GLOBAL address space
    mul.wide.s32   %rd4, %r5, 4;        // byte offset = i * 4 (sizeof float)
    add.s64        %rd5, %rd3, %rd4;    // &x[i]
    ld.global.f32  %f2, [%rd5];         // LOAD x[i] from global memory
    // (… load y[i] into %f3 the same way …)
    fma.rn.f32     %f4, %f1, %f2, %f3;  // a*x[i] + y[i]   (.rn = round-to-nearest)
    st.global.f32  [%rd6], %f4;         // STORE y[i]
DONE:
    ret;
}
```

Every PTX kernel has this shape: declare typed virtual registers, read params, compute
the thread's index, bounds-check, convert addresses to a memory space, load → compute
→ store. Once you can read this, you can read any PTX — it's the same vocabulary
scaled up.

---

## 2. The reading checklist — what to look for

When you read a PTX trace to judge a kernel, scan for these, in order:

```text
  1. REGISTER TYPES      %p pred · %f f32 · %fd f64 · %r b32 · %rd b64 · %rs b16
  2. MEMORY SPACE        .global · .shared · .local(!) · .param · .const
                         — .local means a register SPILL to slow memory: a red flag.
  3. VECTOR WIDTH        ld.global.f32 (4B) vs ld.global.v4.f32 (16B). Wider = fewer,
                         fuller memory transactions. Scalar loads in a hot loop = smell.
  4. CACHE OPERATOR      .ca (cache all levels) · .cg (L2 only, skip L1) · .cs
                         (streaming, won't reuse) · .cv (don't cache) · .nc (read-only
                         data cache, = __ldg). Wrong hint = wasted cache.
  5. THE FAST-PATH MARKERS   is the kernel using the hardware it should?
                         wgmma / mma.sync (Tensor Cores) · cp.async, cp.async.bulk.tensor
                         (async / TMA copy) · mbarrier (warp specialization).
                         Their ABSENCE in a GEMM/attention kernel is the story.
  6. COALESCING          do consecutive threads (%tid.x) hit consecutive addresses?
                         If the address stride per thread isn't the element size, the
                         warp's loads scatter — the classic memory-bound cause.
  7. PREDICATION vs BRANCHES   @%p instr (cheap) vs heavy @%p bra control flow
                         (warp divergence).
```

The two highest-value scans: **`.local` (spills)** and **missing fast-path markers**.
A GEMM whose PTX shows `mma.sync` but no `wgmma`/`cp.async.bulk.tensor` on Hopper is
leaving the async tensor-core path on the table — exactly the Gap-B/A split from
[source-to-SASS §5](03_source_to_sass.md#5-where-in-the-pipeline-the-gap-lives).

---

## 3. Profiling: from a trace to the bottleneck

PTX tells you *what instructions exist*; the profiler tells you *which one is the
wall*. Reading PTX without profiling is guessing — always profile first. The tool is
**Nsight Compute** (`ncu`); the sections to read, in order:

```text
  GPU SPEED OF LIGHT (SOL)   the headline: % of peak COMPUTE vs % of peak MEMORY
                             achieved. Whichever is higher is your bound. This is the
                             roofline (ch1 §15 / ch4) made into one screen.
  MEMORY WORKLOAD ANALYSIS   DRAM throughput, L2/L1 hit rates, sectors-per-request
                             (low = uncoalesced).
  COMPUTE WORKLOAD ANALYSIS  which pipes are busy (FMA, Tensor Core, ALU).
  WARP STATE STATISTICS      WHY warps stall — the stall reasons below.
  OCCUPANCY                  achieved vs theoretical resident warps (ch7 §13).
```

The warp **stall reasons** are the most actionable signal — they name the wall:

| Stall reason | Means | Usually points to |
| --- | --- | --- |
| **Long Scoreboard** | waiting on data from global/local memory | memory latency; not enough loads in flight; uncoalesced |
| **LG Throttle** | the L1 *instruction queue* for local/global memory ops is saturated (issuing them too frequently) | too many narrow memory instructions → vectorize |
| **Short Scoreboard** | waiting on shared-memory / MIO | bank conflicts; over-using shared memory |
| **Barrier** | waiting at `__syncthreads()` | imbalanced work across the block |
| **MIO Throttle** | the MIO pipe (shared/special-function) is saturated | too many SFU/shared ops (e.g. `exp`) |

`★ Insight ─────────────────────────────────────`
- **Long Scoreboard ≠ LG Throttle**, and the difference picks your fix. Long
  Scoreboard = stalled *waiting for the data to arrive* from a load (latency) → hide
  it with more in-flight loads / async copy. LG Throttle = the *instruction queue*
  for local/global memory ops is saturated because you issue them *too frequently*
  (instruction-rate, not latency) → issue *fewer, wider* loads (vectorize). Same
  "it's memory," opposite remedies.
`─────────────────────────────────────────────────`

---

## 4. The diagnosis → change map

Best-practice mapping from a profiler symptom to the fix — and crucially, **where**
you make it (almost always the source; PTX only as a last resort, §6):

| Profiler symptom | Likely cause | The change | Made in |
| --- | --- | --- | --- |
| Memory-bound SOL, Long Scoreboard, DRAM% low | scalar / uncoalesced loads, low memory parallelism | **vectorize** (`v4`), fix coalescing, prefetch / `cp.async` | source (`float4`, async) |
| LG Throttle | too many narrow memory instructions | fewer, **wider** transactions | source (vectorize) |
| Long Scoreboard on read-only inputs | not using the read-only cache | `.nc` load (`__ldg`) | source `__ldg`, or inline PTX |
| Short Scoreboard, low L1/shared bandwidth | shared-memory **bank conflicts** | pad / swizzle the shared layout | source (matmul §8) |
| Compute-bound, FMA pipe hot, no Tensor Cores | scalar math instead of MMA | route the matmul to **Tensor Cores** | source (`tl.dot`, `wmma`) |
| `.local` in PTX, low occupancy | **register spills** / pressure | smaller tiles, fewer live regs | source (tile sizes) |
| MIO Throttle on `exp`-heavy softmax | special-function unit saturated | approximate / fuse (the FA-4 polynomial trick) | source (algorithm) |

Read this with [the matmul ladder](../07_writing_and_tuning_kernels/01_writing_and_tuning_a_matmul.md):
each "change" is a rung you already know, now *triggered by a specific profiler
signal* rather than applied blindly.

---

## 5. End-to-end: a memory-bound kernel, all the way through

The whole loop on one example — the `axpy` kernel from §1, which is memory-bound
(~3 bytes moved per FLOP). Numbers are illustrative of the *shape* of the result.

**Step 1 — read the PTX.** The hot loads are scalar:

```ptx
  ld.global.f32  %f2, [%rd5];      // x[i]  — one 4-byte load per thread
  ld.global.f32  %f3, [%rd7];      // y[i]  — another 4-byte load
```

Each thread moves only 4 bytes per load. Checklist (§2) flags item 3: **scalar
loads in the hot path**.

**Step 2 — profile.** `ncu` says:

```text
  GPU SOL:        Memory 48%   Compute 9%        → clearly MEMORY-bound
  Warp State:     Long Scoreboard dominates       → stalled waiting on loads
  Memory:         DRAM throughput ~48% of peak    → bandwidth left on the table
```

**Step 3 — diagnose.** Memory-bound, far from peak bandwidth, stalling on load
latency. 4-byte loads under-fill the memory transactions and there's too little in
flight. The fix is **wider loads** (more bytes per instruction, fewer instructions).

**Step 4 — make the change.** Vectorize: each thread loads 4 floats at once. The PTX
you want:

```ptx
  ld.global.v4.f32  {%f2,%f3,%f4,%f5}, [%rd5];   // one 16-byte (128-bit) load
```

**Where you make it matters.** You do *not* hand-write this PTX. You change the
**source** — use `float4` in CUDA, or let Triton/`torch.compile` vectorize — and the
compiler emits the `v4` load. You'd reach for inline PTX (§6) only if the compiler
*refused* to emit what you need (e.g. a `.nc` read-only hint it won't add).

**Step 5 — re-measure (correctness first).** Verify against a reference *before*
trusting the speedup ([matmul §3](../07_writing_and_tuning_kernels/01_writing_and_tuning_a_matmul.md#3-the-gate-verify-before-you-tune)),
then re-profile:

```text
  correctness:    allclose vs torch  ✓   ← the gate; a fast wrong kernel is worthless
  GPU SOL:        Memory 80%  (was 48%)   → much closer to the bandwidth roofline
  Warp State:     Long Scoreboard down    → fewer, fuller loads in flight
```

That is the entire craft in one pass: **the PTX showed the symptom (scalar loads),
the profiler proved it was the bottleneck (memory-bound, Long Scoreboard), the fix
followed from the diagnosis (vectorize), it was made in the source, and it was gated
on correctness before the speedup counted.**

---

## 6. When to actually touch PTX

Inline PTX is the surgical last resort, **not** the default — NVIDIA's own guidance
is to use it "sparingly and only after profiling." You drop to it when profiling has
identified a *specific instruction* the compiler won't emit from source: a cache
hint (`.nc`, `.cs`), a particular async/atomic variant, or a brand-new hardware
instruction. The syntax embeds PTX in CUDA C++:

```cpp
  float v;
  asm volatile("ld.global.nc.f32 %0, [%1];"   // force a read-only-cache load
               : "=f"(v)                        // output: "=f" = a float register
               : "l"(ptr));                     // input:  "l"  = a 64-bit register
  // volatile = don't let the compiler move or delete this;
  // YOU are responsible for the memory space being correct (asm can't check it).
```

The decision rule, the whole calibration of this chapter in one line:

```text
  99% of the time:  change the SOURCE (vectorize, tile, route to Tensor Cores).
   1% of the time:  inline PTX — one instruction, after profiling, then VERIFY + measure.
  hand-writing whole kernels in PTX:  essentially never (see source-to-SASS §7).
```

---

## 7. Further reading

The read-PTX → profile → diagnose → fix loop runs on two NVIDIA manuals and a lot
of practice:

- **[Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)**
  (NVIDIA) — defines Speed-of-Light and the warp-stall reasons §3 reads (Long Scoreboard,
  LG Throttle, …); the reference for "the profiler names the wall."
- **[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)** (NVIDIA) — what
  the register types, memory-space qualifiers, and cache operators in §1–§2 mean when you
  read a dump.
- **[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)**
  (NVIDIA) — the ordered optimization priorities behind §4's symptom → change map: profile
  first, fix memory before compute, then chase occupancy.
- **[GPU MODE lectures](https://github.com/gpu-mode/lectures)** (Mark Saroufim & Andreas
  Köpf) — recorded sessions that run this exact profile-and-optimize loop on real kernels.

## 8. What to carry forward

```text
read PTX: register types, memory space, vector width,           -> M8, M24, reading any dump
  cache op, fast-path markers, coalescing (§1–§2)
profile FIRST; SOL = your bound; stall reasons name             -> M8 (ncu), every optimization
  the wall — Long Scoreboard vs LG Throttle (§3)
the symptom→change→where map; the fix is a known rung           -> M24/M27/M30, triggered by data
  triggered by a profiler signal (§4)
the loop: read PTX → profile → diagnose → change in             -> M8/M24, do it for real
  SOURCE → verify → re-measure (§5)
inline PTX is the 1% surgical case, after profiling (§6)         -> M27/M30, only when needed
```

The one sentence to keep: **optimizing a kernel is a disciplined loop — read the PTX
to spot the symptom (scalar loads, missing `wgmma`/TMA, `.local` spills), profile
with `ncu` to prove which one is the bottleneck (the Speed-of-Light bound and the
dominant stall reason, Long Scoreboard vs LG Throttle), pick the fix that signal
implies (vectorize, coalesce, Tensor Cores, async copy), make it in the *source*
(reaching for inline PTX only as a 1% surgical exception), and gate every speedup on
a correctness check first — never hand-write PTX when reading it and changing the
source does the job.**
