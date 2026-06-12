# FlashAttention across four languages: a Rosetta Stone

The best way to feel what a kernel language actually *costs* and *buys* is to read
the **same algorithm** written at every rung of the ladder. FlashAttention is the
perfect specimen, because of a lucky fact: the
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) repo holds
*one algorithm in four languages*, by the same authors —

```text
  csrc/              FlashAttention-1 / 2   →  raw CUDA C++
  hopper/            FlashAttention-3       →  CuTe C++ / CUTLASS   (WGMMA, TMA, warp spec)
  flash_attn/cute/   FlashAttention-4       →  CuTe DSL (Python)    (pip install flash-attn-4)
  third_party/       Triton backend (AMD)   →  Triton
```

So you can hold the algorithm fixed and watch *only the language change* — and see
exactly which work each rung forces you to write by hand versus hands to a compiler.
This doc is that side-by-side read. It's the conceptual prework for **M25**
(FlashAttention Rosetta Stone) and **M26** (build it yourself) in the
[Phase 5 track](../ROADMAP.md), and it builds directly on the
[CuTe DSL foundations](01_cute_dsl_foundations.md) doc — I use *layout*, *atom*,
*WGMMA*, *TMA*, *warp specialization*, and *tcgen05* here as defined there.

**Honesty up front.** No local NVIDIA GPU (remote-only), so this is an annotation
of public source and papers, not something I ran. Code shapes below are
*illustrative sketches* of structure, not copy-paste source; the real reading and
benchmarking is M25–M26 on rented GPUs. Quoted numbers are the original authors',
labelled as such. A bare **§N** means *this doc's* section N.

Prerequisites: [CuTe DSL foundations](01_cute_dsl_foundations.md);
[Chapter 5 — attention](../05_attention_and_kv_cache/01_attention.md) (Q·Kᵀ →
softmax → ·V, the KV cache); and the
[matmul tuning ladder](../07_writing_and_tuning_kernels/01_writing_and_tuning_a_matmul.md)
(tiling, Tensor Cores, pipelining).

---

## 1. FlashAttention in one screen (the fixed reference point)

Before comparing languages, pin down the algorithm they all implement, so "what
changed" has a baseline. Plain attention computes, for a query block, scores
against *all* keys, softmaxes them, and weights the values:

```text
  S = Q·Kᵀ        (N×N scores)     ← the problem: N×N is huge, and it's written
  P = softmax(S)  (N×N weights)       out to main memory and read back — attention
  O = P·V         (the output)        is bound by that MEMORY traffic, not the FLOPs
```

FlashAttention's two ideas remove the N×N round-trip:

- **IO-awareness** — never materialize the full `S`/`P` matrix in slow **HBM**
  (*High-Bandwidth Memory* — the GPU's main memory). Tile Q, K, V into blocks that
  fit in on-chip **SRAM** (*Static RAM* — the fast shared memory on each SM), and
  compute the
  output block by block, keeping the scores on-chip. This is the same
  tiling-for-reuse idea as
  [the matmul ladder's rung 2](../07_writing_and_tuning_kernels/01_writing_and_tuning_a_matmul.md#6-rung-2--shared-memory-tiling),
  applied to attention.
- **Online (streaming) softmax** — softmax normally needs the max and the sum over
  the *whole* row (you can't normalize until you've seen every score). FlashAttention
  keeps a *running* max `m` and running sum `l`, and **rescales** the
  partial output each time a new K/V block raises the max:

```text
  for each block of K,V:
     s   = Q·Kblockᵀ                       # scores for this block (on-chip)
     m'  = max(m, rowmax(s))               # update running max
     p   = exp(s - m')                     # unnormalized weights, stabilized
     l   = exp(m - m')·l + rowsum(p)       # rescale + extend running sum
     O   = exp(m - m')·O + p·Vblock        # rescale running output, accumulate
     m   = m'
  O = O / l                                # normalize once, at the end
```

That loop is the whole algorithm. **Every implementation below is this loop** —
what differs is how the tiling, the two matmuls (`Q·Kᵀ` and `P·V`), and the data
movement are expressed, and how aggressively the softmax is overlapped with the
matmuls. (Chapter 8 places FlashAttention among prefill's levers;
[improving prefill](../08_optimizing_inference/02_improving_prefill.md) is where it
sits in the serving picture, and it's module **M16** in the roadmap.)

---

## 2. The four implementations at a glance

| Version | Language | Directory | GPU target | What it adds over the previous |
| --- | --- | --- | --- | --- |
| **FA-1** | CUDA C++ | `csrc` | Ampere (A100) | the algorithm itself: SRAM tiling + online softmax + backward recompute |
| **FA-2** | CUDA C++ / CUTLASS | `csrc` | Ampere | better parallelism + fewer non-matmul ops → ~2× FA-1 |
| **FA-3** | CuTe C++ / CUTLASS | `hopper` | Hopper (H100) | WGMMA + TMA + warp specialization + GEMM/softmax overlap + FP8 |
| **FA-4** | CuTe DSL (Python) | `flash_attn/cute` | Hopper + Blackwell | same control in Python; new pipeline + softmax emulation + tcgen05/TMEM |
| (Triton) | Triton | `third_party` | NVIDIA + AMD | portability; the compiler does the scheduling |

Read top to bottom and you're descending the control ladder *and* climbing GPU
generations at the same time — which is the real lesson: **each new architecture
exposed controls that pushed the kernel one rung lower.**

---

## 3. FA-1 / FA-2 — CUDA C++ (`csrc/`): the by-hand baseline

Here you write the §1 loop with raw CUDA: you index threads yourself, stage tiles
into shared memory yourself, and call Tensor Cores through CUTLASS GEMM helpers.

```text
  // sketch of the structure, not real source
  __global__ void fa_kernel(const half* Q,*K,*V, half* O, ...) {
      __shared__ half Ks[BK][D], Vs[BK][D];     // you declare the SRAM tiles
      float m = -inf, l = 0; float acc[D] = {0}; // running softmax state in registers
      for (int kb = 0; kb < num_k_blocks; ++kb) {
          load_to_shared(Ks, Vs, K, V, kb);      // you write the global→shared copy
          __syncthreads();
          mma(s, Qreg, Ks);                       // Q·Kᵀ via Tensor Cores (CUTLASS)
          online_softmax_update(m, l, acc, s, Vs);// the rescale loop, by hand
          __syncthreads();
      }
      write_output(O, acc, l);                    // normalize + store
  }
```

What **FA-2** changed is instructive precisely because it's *not* a language change:
it reorganized the *work*, parallelizing across query blocks over more thread blocks
and splitting work across warps so they communicate less through shared memory, plus
trimming the non-matmul (softmax) operations. Pure algorithm/scheduling craft, same
language — and it roughly doubled throughput (the authors report FA-2 reaching a
large fraction of A100 matmul peak, versus FA-1's ~half of that). The lesson: at
this rung, *you* own every scheduling decision, and the wins come from your
cleverness, not the compiler's.

---

## 4. FA-3 — CuTe C++ / CUTLASS (`hopper/`): what Hopper forced

The H100 added async Tensor Cores and a copy engine that the FA-2 structure
couldn't exploit, so FA-3 dropped to CUTLASS/CuTe to drive them directly. The §1
loop is the same; the *machinery* around it is new (all terms from
[the foundations doc §6](01_cute_dsl_foundations.md#6-the-hardware-vocabulary-defined-once)):

```text
  - WGMMA              the two matmuls (Q·Kᵀ, P·V) issue ASYNC: fire the Tensor
                       Cores and keep going while they work.
  - TMA                a DMA engine streams the next K/V tile into shared memory
                       without using the math lanes.
  - WARP SPECIALIZATION split the warps: PRODUCER warps issue TMA loads, CONSUMER
                       warps issue WGMMA — coordinated by mbarriers.
  - OVERLAP            the softmax of one block runs while the WGMMA of the next
                       block is in flight (hide the non-matmul work).
  - FP8                an 8-bit forward path for even more Tensor-Core throughput.
```

This is the rung where "writing a kernel" becomes "choreographing asynchronous
hardware." You're no longer just tiling — you're assigning *roles* to warps and
scheduling producer/consumer hand-offs. The authors report FA-3 reaching ~75% of
the H100's FP16 matmul peak, well above a Hopper-naive port. CuTe's **atoms** are
how this stays readable: one WGMMA atom, one TMA copy atom, each replicated over a
thread layout (the [tiled-MMA idea](01_cute_dsl_foundations.md#4-atoms-the-smallest-unit-of-hardware-work)).

---

## 5. FA-4 — CuTe DSL (`flash_attn/cute/`): the same control, in Python

FA-4 keeps FA-3's async/warp-specialized strategy but writes it in **CuTe DSL** —
Python that generates the kernel (no Python in the running kernel; the
[foundations doc §1](01_cute_dsl_foundations.md#why-being-python-costs-no-performance)
explains why that's free). On top, it adds Blackwell support and pipeline tricks:

```text
  - PING-PONG PIPELINE   overlaps the MMA of one stage with the softmax of another
                         more aggressively than FA-3's overlap.
  - SOFTMAX EMULATION    the exponential is approximated by a polynomial, because on
                         Blackwell the special-function unit can't keep up with the
                         doubled Tensor-Core throughput (asymmetric scaling).
  - BLACKWELL            tcgen05 MMA, Tensor Memory (TMEM), and 2-CTA MMA — the
                         5th-gen Tensor-Core path.
```

The headline that makes this the *recommended* low-level tool: FA-4's paper reports
it runs **1.2×–3.2× over the Triton implementation** on compute-bound Blackwell
workloads, and compiles **~20–30× faster** than the equivalent C++ CUTLASS — i.e.
FA-3's performance ceiling, at a fraction of FA-3's authoring pain. That trade is
the entire reason Phase 5 invests in CuTe DSL rather than C++. (This is the 2026
endpoint of the
[kernel-language timeline](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md#6-2026-blackwell--b200--the-trend-becomes-unavoidable).)

---

## 6. The Triton version (`third_party/`): the high-abstraction contrast

The repo's Triton path (used for AMD support) is the *opposite* end of the ladder,
and reading it next to the others is the sharpest single lesson in the chapter:

```text
  - you write the §1 loop over TILES; tl.dot does the matmuls.
  - the COMPILER chooses the thread/value layout, the shared-memory staging, the
    coalescing, the pipelining — everything §4 and §5 made explicit.
  - ONE source runs on both NVIDIA and AMD; the flash_attn CUDA/CuTe paths are
    different code per vendor.
```

The published trade-off, from a Triton-attention study: an autotuned Triton kernel
is broadly competitive across platforms while using **under 2% of the lines of code**
of the CUDA implementations — but the hand-written CuTe paths pull ahead at the peak
on the newest hardware (FA-4's 1.2–3.2× over Triton, §5). That *is* the
productivity-vs-control trade of
[the landscape doc](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md),
made concrete in one repo: Triton for portability and 50× less code; CuTe for the
last ~2× on a specific GPU.

---

## 7. What changes as you descend — the synthesis

Hold the algorithm fixed (§1) and tabulate *only what each language makes you say*:

| Concept | Triton | CuTe DSL / CuTe C++ | raw CUDA C++ |
| --- | --- | --- | --- |
| **Tiling** | "this tile" (compiler maps threads) | explicit **layouts** (Shape:Stride) | manual indices + `__shared__` |
| **The two matmuls** | `tl.dot` (compiler picks MMA) | **MMA atom** (WGMMA/tcgen05) you choose | CUTLASS GEMM call by hand |
| **Data movement** | compiler stages global→shared | **TMA copy atom**, you place it | hand-written load loops |
| **Thread→data map** | hidden | **TV layout**, you write it | manual `threadIdx` math |
| **Warp roles / overlap** | hidden | **warp specialization** + mbarriers, explicit | hand-rolled, very hard |
| **Lines of code** | ~1× (smallest) | a few × | tens of × (largest) |
| **Performance ceiling** | ~high, portable | peak on target GPU | peak, most effort |

Reading across a row is the "three knobs" payoff from
[the landscape doc §2](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md#2-the-three-knobs-every-tool-blends):
descending a rung means *specialization*, *explicit scheduling*, and *explicit data
placement* move from the compiler to you. The algorithm never changed — only who
makes the hardware decisions.

---

## 8. How I'll actually read and build it (M25 → M26)

The plan the roadmap encodes, in order:

```text
  M25  READ.   Start with an EDUCATIONAL Triton FA from scratch (far more readable
               than production code) to lock in the §1 loop, then read the four real
               versions, mapping each to the §7 table — what moved from compiler to me.
  M26  BUILD.  Carry M16's simplified Triton FA further, then REBUILD it in CuTe DSL:
               write the layouts and the tiled MMA myself; verify vs torch attention;
               benchmark vs flash-attn and a naive baseline, honestly (the
               correctness-first gate from the matmul doc).
```

Readable entry points to pair with the production source (not affiliated, just good
teaching): the *"Anatomy of a Triton Attention Kernel"* paper
([arXiv 2511.11581](https://arxiv.org/html/2511.11581v1)) and a from-scratch Triton
FA walkthrough. Read those *before* `csrc/` and `hopper/`, which are dense.

---

## 9. Further reading

This doc reads FlashAttention across four languages; these are the papers behind
each rung and the repo that holds them all:

- **[FlashAttention](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022) — the original
  tiling + online-softmax loop (§1) that every later version keeps.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)** (Tri Dao, 2023) — better work
  partitioning across thread blocks and warps; the rung that closed most of the gap to GEMM.
- **[FlashAttention-3](https://arxiv.org/abs/2407.08608)** (Shah et al., 2024) — the Hopper
  rewrite (§4): WGMMA + TMA asynchrony, warp specialization, and FP8, the lowest rung this
  doc reads.
- **[flash-attention (source)](https://github.com/Dao-AILab/flash-attention)** (Dao-AILab)
  — the repository that is the Rosetta Stone of §2: the same algorithm in CUDA, CuTe C++,
  CuTe DSL, and Triton, side by side.

## 10. What to carry forward

```text
the §1 loop (tiling + online softmax) is the WHOLE         -> M16, M25, M26
  algorithm; every version is this loop
one repo holds FA in 4 languages — the Rosetta Stone (§2)   -> M25, read them side by side
each GPU gen pushed the kernel one rung LOWER:              -> M27 (Hopper), M28 (Blackwell)
  Hopper→WGMMA/TMA/warp-spec (§4), Blackwell→tcgen05 (§5)
the §7 table: descending = specialization + scheduling +    -> M26, when I rebuild in CuTe DSL
  data placement move from compiler to me
Triton = ~50× less code, portable; CuTe = last ~2× on        -> the language choice, M26
  the target GPU (§6)
```

The one sentence to keep: **FlashAttention is a single loop — tile Q/K/V into SRAM
and stream an online softmax — and reading it across CUDA C++ (`csrc`), CuTe C++
(`hopper`), CuTe DSL (`flash_attn/cute`), and Triton (`third_party`) shows the
algorithm never changing while *who decides the layouts, the Tensor-Core
instruction, the data movement, and the warp roles* slides from the compiler (Triton,
~50× less code) down to you (CuTe/CUDA, the last ~2× on a specific GPU) — which is
the productivity-vs-control ladder made concrete, and the reason CuTe DSL is the rung
worth mastering.**
