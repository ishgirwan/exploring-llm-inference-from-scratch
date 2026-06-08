# Reading a real optimized kernel: Triton's matmul, annotated

[Section 1](01_writing_and_tuning_a_matmul.md) is the *concepts* of writing and
tuning a kernel. This section is the *code* — but rather than build one from
scratch, it does what [Roadmap §9](../ROADMAP.md) says the honest version of
learning looks like: **read a real, optimized open-source kernel and understand
what was done to it.** Then §1's ladder stops being abstract — you can see each
rung in actual source you can pull up and run yourself.

The kernel I picked is the **official Triton matrix-multiplication tutorial**.
It's the right example because it is canonical (the reference everyone learns
Triton matmul from), readable (~100 lines), and — crucially — it ships *already
optimized and already autotuned*, so it demonstrates the exact thing the chapter
is about. The source, to refer back to:

```text
  https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  repo: python/tutorials/03-matrix-multiplication.py   (triton-lang/triton)
```

**Honesty up front.** This project has no local NVIDIA GPU
(it's remote-only), so everything below is an *annotation of the real source*,
not something I ran here. I don't show timing numbers, because I haven't measured
them — the benchmarking happens at M6 on a rented GPU, against the
[harness](../04_measurement/01_benchmarking.md), and every number there will have
a script behind it. Code excerpts are quoted from the Triton tutorial linked
above (current `main`).

Prerequisites: [§1 of this chapter](01_writing_and_tuning_a_matmul.md). This
section assumes its ladder (coalescing → tiling → registers → Tensor Cores →
pipelining) and its tuning-knob vocabulary.

## 1. Why Triton, and what the other languages do here

The [call-vs-build framing from the stack doc](../02_cuda_software_stack/01_the_stack.md#what-people-actually-use)
decides the language for each role:

```text
  PyTorch    torch.matmul — the REFERENCE we check against (and the cuBLAS
             fast path we'd ultimately compare speed to). One line.
  CUDA C++   where the absolute peak lives (cuBLAS / CUTLASS, Tensor-core
             hand-tuning). The naive version in §1 §2 is also CUDA.
  Triton     the readable middle: you write the BLOCK-level algorithm, and
             the compiler does coalescing, shared-memory staging, vectorized
             loads, and pipelining FOR you. Best for SEEING the optimization
             and the autotuning in one short file. ← our walkthrough
```

That last point is the key reason Triton is the right teaching vehicle: several
rungs of §1's ladder (coalescing = rung 1, shared-memory staging = rung 2,
vectorized loads = rung 4, pipelining = rung 5) are *handled by the Triton
compiler* from your block-level description plus a couple of knobs. So the source
shows mostly the rungs that remain *your* decision — the tiling structure, the L2
ordering, and the autotuned configuration — which is exactly the interesting part.

## 2. The reference, and the naive baseline (for contrast)

The oracle you verify against is one line — and it's also the cuBLAS path, the
real speed bar ([§1 §17](01_writing_and_tuning_a_matmul.md#17-measuring-optimal--the-honest-bar)):

```python
c_ref = torch.matmul(a, b)          # correctness oracle + cuBLAS baseline
```

The naive baseline is the one-thread-per-output kernel from
[§1 §2](01_writing_and_tuning_a_matmul.md#2-the-naive-matmul--correct-and-slow):
every thread reads a full row of A and column of B from global memory, no reuse,
memory-bound. Hold it in mind as the "before"; everything below is the "after."

## 3. The optimized kernel, walked rung by rung

Here is the real kernel's signature and decorator (quoted from the tutorial):

```python
@triton.autotune(
    configs=get_cuda_autotune_config(),   # the tuning knobs — §4 below
    key=['M', 'N', 'K'],                   # re-tune when these change
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,                       # matrix pointers
        M, N, K,                                   # dimensions
        stride_am, stride_ak,                      # how to step through A
        stride_bk, stride_bn,                      # ...through B
        stride_cm, stride_cn,                      # ...through C
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,                # the tile sizes (= §1's BM/BN/BK)
        GROUP_SIZE_M: tl.constexpr,                # the L2-ordering knob (§3.1)
        ACTIVATION: tl.constexpr,                  # optional fused epilogue
):
```

Note the correspondence to §1: `BLOCK_SIZE_M/N/K` are §1's tile sizes `BM/BN/BK`,
and they're `tl.constexpr` (compile-time constants) precisely so the autotuner can
compile a specialized kernel per configuration. Now the body, in four parts.

### 3.1 Grouped program ordering — an L2-cache optimization

This is an optimization §1 didn't cover, and it's the first thing the kernel does:

```python
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

Each Triton *program* computes one `BLOCK_SIZE_M × BLOCK_SIZE_N` tile of C. The
naive thing is to order programs row-major. This code instead walks them in
*groups* of `GROUP_SIZE_M` rows, so blocks that run around the same time reuse the
same columns of B — which are then resident in the shared
[L2 cache](../01_hardware_fundamentals/03_gpu_model.md#12-l2-connects-the-sms-to-gpu-memory).

The picture, for the ~9 C-tiles computed around the same time (so their inputs
share L2): grouping makes that co-resident region *squarer*, so it touches fewer
distinct row/column bands of the inputs at once —

```text
  ROW-MAJOR (a 1×9 strip)            GROUPED, GROUP_SIZE_M=3 (a 3×3 square)
    ▓▓▓▓▓▓▓▓▓                          ▓▓▓······
    ·········                          ▓▓▓······
    ·········                          ▓▓▓······
    touches 1 A row-band               touches 3 A row-bands
          + 9 B col-bands                    + 3 B col-bands
    = 10 input bands in flight         = 6 input bands in flight  ← smaller
                                         working set → more L2 hits
```

Same nine output tiles either way; the grouped order just needs fewer distinct
input bands resident at once, so more of them stay in L2. `GROUP_SIZE_M` sets how
square the group is, and it's autotuned (§4).

```text
  why it helps: the tutorial notes this raises an example matmul from ~220 to
  ~245 TFLOPS purely by re-ordering which blocks are co-resident — same math,
  better cache reuse. GROUP_SIZE_M is the knob (autotuned, §4).
```

(That ~220→245 figure is the tutorial's own measurement on its hardware, not
mine — quoted to show the *kind* of gain ordering buys, not as a number for our
setup.)

### 3.2 The block pointers — this is §1's tiling (rung 2)

```python
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

`a_ptrs` is a `[BLOCK_SIZE_M × BLOCK_SIZE_K]` block of pointers, `b_ptrs` is
`[BLOCK_SIZE_K × BLOCK_SIZE_N]` — these *are* the shared-memory tiles from
[§1 rung 2](01_writing_and_tuning_a_matmul.md#6-rung-2--shared-memory-tiling).
In Triton you describe the tile in terms of pointers and the compiler arranges
the actual global→shared staging and coalesced loads (rungs 1–2) for you.

### 3.3 The K-loop — tiling + the Tensor Core track-switch

```python
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)      # ← Tensor Cores
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
```

Three things to read here:

```text
- march over K in BLOCK_SIZE_K steps, loading one tile pair per iteration and
  reusing it — the tiled-matmul recipe (§1 rung 2).
- tl.dot(a, b, accumulator) is the TRACK-SWITCH from §1 §10: for fp16/bf16
  inputs Triton lowers tl.dot onto TENSOR CORES (the MMA instructions), not
  scalar FP lanes. You don't hand-write fragments; tl.dot is the door to them.
- the accumulator is FP32 even though inputs are fp16 — accumulate-in-higher-
  precision, the Tensor-core accumulator-precision point from
  ch3 numerics. Cast back to fp16 only at the end.
```

The `mask=...other=0.0` handles the ragged last K-tile when K isn't a multiple of
`BLOCK_SIZE_K` — a correctness detail, not a speed one.

### 3.4 The masked store

```python
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

Write the tile back to global memory, masking the edges so out-of-range rows/cols
of a partial tile aren't written. That's the `store` of the four-part skeleton
([§1 §1](01_writing_and_tuning_a_matmul.md#1-what-a-kernel-is-made-of)).

What's *not* in this source, because Triton does it: the coalescing (rung 1), the
shared-memory bank handling and vectorized loads (rung 4), and the software
pipelining (rung 5) — that last one is driven by the `num_stages` knob in §4.

## 4. The autotuning — the config search itself

This is the real config list the kernel autotunes over (abbreviated; the full set
is in the source):

```python
def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # ... ~12 more, including bigger BLOCK_SIZE_K configs flagged
        #     "Good config for fp8 inputs"
    ]
```

Every dial here is a §1 tuning knob, now concrete
([§1 §12](01_writing_and_tuning_a_matmul.md#12-the-configuration-space)):

```text
  BLOCK_SIZE_M/N/K   the shared-memory tile (rung 2)
  GROUP_SIZE_M       the L2 ordering (§3.1)
  num_warps          warps per program (block size)
  num_stages         software-pipeline depth (rung 5) — how many K-tiles to
                     prefetch ahead while computing the current one
```

And the decorator wires the search:

```python
@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
```

This is [§1 §14](01_writing_and_tuning_a_matmul.md#14-autotuning-searching-the-space)
made literal: the **first** time the kernel is called for a given `(M, N, K)`,
Triton runs every config, times it, and caches the fastest — the first-call cost
from [end-to-end §8](../02_cuda_software_stack/02_end_to_end_inference.md#8-who-chooses-the-kernel-and-when).
`key=['M','N','K']` means a *new* shape triggers a *fresh* search, because the best
tile for a 4096³ matmul isn't the best for a skinny decode shape
([§1 §11](01_writing_and_tuning_a_matmul.md#11-decode-is-a-gemv-and-batching-turns-it-back-into-a-gemm)) —
and a fresh search per architecture too
([§1 §15](01_writing_and_tuning_a_matmul.md#15-gpu-architecture-dependence)). Note the configs flagged "good for fp8"
with larger `BLOCK_SIZE_K`: the right tile genuinely depends on dtype.

## 5. The launch wrapper

```python
def matmul(a, b):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) *
                         triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION="",
    )
    return c
```

One thing to notice: `BLOCK_SIZE_*`, `GROUP_SIZE_M`, `num_warps`, `num_stages` are
**not passed here** — the autotuner supplies them from the winning config, which is
also why `grid` is a `lambda` of `META` (it can't know the tile size until the
config is chosen).

## 6. How you'd verify and measure it (at M6)

Correctness first, always ([§1 §3](01_writing_and_tuning_a_matmul.md#3-the-gate-verify-before-you-tune)):

```python
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
torch.testing.assert_close(matmul(a, b), torch.matmul(a, b), atol=1e-2, rtol=1e-2)
#                                                            ^ fp16 tolerances, not exact
```

Then speed, against the real baseline (cuBLAS via `torch.matmul`), with proper
warm-up and GPU timing — Triton ships `do_bench` for exactly this:

```python
ms = triton.testing.do_bench(lambda: matmul(a, b))
tflops = 2 * M * N * K / (ms * 1e-3) / 1e12     # 2·M·N·K FLOPs in a matmul
# compare against: triton.testing.do_bench(lambda: torch.matmul(a, b))
```

I am not filling in numbers, because I haven't run it (no local GPU). The *method*
is the point; the measured result is M6's job, on the
[harness](../04_measurement/01_benchmarking.md), where it'll be reproducible.

## 7. What this kernel is, and isn't

Read honestly ([§1 §17](01_writing_and_tuning_a_matmul.md#17-measuring-optimal--the-honest-bar)):
the Triton tutorial matmul is genuinely good — autotuned, Tensor-core-using, with
L2 ordering — and on many shapes it lands close to cuBLAS. It is *not* the
last word: production-peak GEMMs (cuBLAS, CUTLASS) add architecture-specific
scheduling, persistent kernels, TMA on Hopper, and more
([§1 §16](01_writing_and_tuning_a_matmul.md#16-what-people-actually-reach-for)).
The value of reading this one is that **every rung of §1's ladder is visible in it
or visibly delegated to the compiler** — and you can clone the repo, run it on a
rented GPU, and watch the autotuner pick a config.

## 8. What to carry forward

```text
read the real source, map it to the concepts (this doc)   -> M6, do it for real
tl.dot = the Tensor Core door in Triton (§3.3)            -> M6, fp16/bf16/fp8
the autotune config list = §1's knobs, literal (§4)        -> M6, tune + profile
verify (allclose) before do_bench (§6)                     -> M0 harness, always
the gap to cuBLAS is the honest "I learned this" (§7)      -> M6 report
```

The one sentence to keep: **a real optimized kernel like Triton's matmul is §1's
ladder made concrete — block tiling and L2 ordering you write, coalescing and
pipelining the compiler adds, Tensor Cores reached through `tl.dot`, and the whole
thing wrapped in an `@triton.autotune` config list that searches the tile/warp/stage
space per shape — and the way to learn it is to read it, then run it against cuBLAS
and measure the gap.**
