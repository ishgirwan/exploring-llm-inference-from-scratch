# Kernels on AMD: ROCm, Composable Kernel, and FlyDSL

The [Phase 5 kernel-engineering track](../../ROADMAP.md) (M23–M30), and the four
chapter-9 docs before this one, are entirely about NVIDIA: CuTe DSL, Hopper's
TMA/WGMMA, Blackwell's tcgen05. That's a deliberate narrowing — NVIDIA is where the
tooling is deepest and where I learn fastest — but it isn't the whole map. There is a
second vendor, **AMD**, where the *same* low-level skills transfer almost intact, the
kernel-contribution gap is wider, and far fewer people are working on it. This doc
maps the AMD side onto the NVIDIA stack I already built up in
[CuTe DSL foundations](01_cute_dsl_foundations.md), names the AMD-native tool that
mirrors CuTe DSL (**FlyDSL**), and works out how to learn any of it with no hardware.

This is conceptual prework for an *optional AMD arm* of Phase 5 — a parallel pass over
the GEMM / FlashAttention / beat-the-baseline modules on AMD silicon. Like the rest of
Chapter 9, it isn't pre-M0 reading; it's for when (or if) that arm begins.

**Honesty up front.** This project has no local GPU of either brand (it's remote-only),
so nothing here is something I ran. It's an explanation of the public documentation and
the stack's shape; the running and measuring happen on rented GPUs at M-module time.
Every number is from a public source, labelled and dated as such — and cloud prices
move week to week, so treat them as order-of-magnitude, not quotes.

Prerequisites: [CuTe DSL foundations](01_cute_dsl_foundations.md) (the stack picture,
*layouts*, *atoms*, and the hardware glossary I map across here) and the
[kernel-language landscape](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md)
(the productivity-vs-control ladder). A bare **§N** below means *this doc's* section N;
references to other docs are always links.
Next: the [Phase 5 kernel-engineering track (M23–M30)](../../ROADMAP.md) — where the
optional AMD arm this doc proposes would get built and measured.

---

## 1. Why AMD is the thinner-staffed frontier

The dependency chain is the same on both vendors. A new model ships a new attention
variant, a new quantization format, a new mixture-of-experts routing — and each needs
kernels matched to *its* shapes before a serving engine (vLLM, SGLang) can run it
fast. On NVIDIA that gap closes quickly: NVIDIA ships its best inference kernels through
**FlashInfer** and **CUTLASS**, with a large team and a decade of tooling behind them.
On AMD the same gap exists but fewer hands close it, so frameworks wait longer for an
AMD-tuned path to land.

There's a structural reason it's a *separate* effort rather than a recompile. The
obvious idea — "just build CUTLASS for AMD" — does not work: **CUTLASS is NVIDIA-only.**
It depends on libcu++ (`cuda/std/*`) headers that only exist in the NVIDIA toolchain,
so it cannot compile for AMD at all. AMD therefore had to build a *parallel stack from
scratch* — its own portability language, its own template library, its own Python DSL,
its own kernel distribution library. Each of those is younger and smaller than the
NVIDIA original it answers.

```text
  NVIDIA:  big team + 10 yrs of tooling  →  new-model kernels land fast
  AMD:     younger parallel stack, fewer hands  →  the same kernels land slower
                                                    = a wider gap, thinly staffed
```

Reseal it: the gap is real, and its *thinness* is exactly why a learner can matter here.
On NVIDIA a hand-written kernel competes with NVIDIA's own experts; on AMD there are
whole model/operator combinations with no tuned kernel yet, where a correct, well-profiled
contribution is genuinely new work rather than a re-derivation of something a vendor
already shipped.

---

## 2. The stack, mapped: NVIDIA → AMD

The single most useful thing to carry over is that AMD's stack is a rung-for-rung
mirror of the NVIDIA one from
[foundations §1](01_cute_dsl_foundations.md#1-where-cute-dsl-sits-cuda-c--cutlass--cute--cute-dsl).
Learn the mapping once and most of the vocabulary transfers:

| NVIDIA rung | AMD equivalent | What it is |
| --- | --- | --- |
| CUDA C++ | **HIP** (Heterogeneous-compute Interface for Portability) | A C++ runtime + kernel language deliberately shaped like CUDA, so CUDA code ports to AMD with a near-mechanical rename (`cudaMalloc`→`hipMalloc`). Thin; little overhead over native. |
| PTX → SASS | LLVM IR → **gfx ISA** (AMDGCN) | AMD compiles straight through LLVM to the GPU's native assembly (the `gfxNNN` ISA) — there's no separate virtual-ISA layer like PTX. |
| CUTLASS (C++ templates) | **Composable Kernel (CK)** | AMD's tile-based C++ template library, written in HIP C++; the direct CUTLASS analog. AMD-only; now developed in the `ROCm/rocm-libraries` monorepo. |
| CuTe / **CuTe DSL** (Python) | **FlyDSL** (§3) | The Python, MLIR-native layout DSL — the CuTe DSL analog, built to ease porting CUTLASS/CuTe/FlashAttention kernels onto AMD. |
| FlashInfer / cuDNN / cuBLAS | **AITER** (AI Tensor Engine for ROCm) + rocBLAS / hipBLASLt | The library of pre-optimized ops (attention, MoE, GEMM, norm, quant) a serving engine pulls in. AITER is multi-backend: Triton + CK + hand-tuned assembly + HIP. |
| Triton (on CUDA) | **Triton (on ROCm)** | The *same* tile DSL — Triton has a ROCm backend, so a Triton kernel is the one rung that is genuinely write-once, run-on-both. |
| CUDA (the platform) | **ROCm** (historically *Radeon Open Compute*) | AMD's open GPU-compute stack — the platform all of the above sits in. |

And the hardware words change, though the concepts don't. This is the second
translation table, mapping the
[foundations §6 glossary](01_cute_dsl_foundations.md#6-the-hardware-vocabulary-defined-once)
onto AMD's CDNA architecture (the data-center line — MI300/MI350):

| NVIDIA term | AMD term | Note |
| --- | --- | --- |
| SM (Streaming Multiprocessor) | **CU** (Compute Unit) | The core scheduling/compute block. |
| Warp (32 lanes) | **Wavefront (64 lanes on CDNA)** | AMD's SIMD group is *twice as wide* — a real difference that changes how you size tiles. |
| Shared memory | **LDS** (Local Data Share) | The on-chip scratchpad the threads of a block share. |
| Tensor Core | **Matrix Core** | The matrix-multiply unit. |
| MMA / WGMMA / tcgen05 | **MFMA** (Matrix Fused Multiply-Add) | The matrix-core instruction family — the *atom* (in CuTe terms) you build a tiled MMA from. |

```text
  the lesson:  the IDEAS (layouts, atoms, tiling, coalescing, async copy,
               feeding the matrix unit without stalling) are vendor-neutral.
               only the NAMES and a few sizes (64-lane wavefront, LDS budget)
               change. that's why CuTe-DSL fluency ports to FlyDSL.
```

---

## 3. FlyDSL: the CuTe DSL of AMD

This is the rung the whole AMD track turns on, because it's the one that makes AMD
kernel authoring feel like the CuTe DSL work in the rest of this chapter rather than
weeks of hand-tuned HIP C++.

**FlyDSL** is the Python front-end of the *Flexible LaYout DSL* project: a Python DSL
plus an MLIR compiler stack for writing high-performance AMD kernels with explicit
layouts and tiling. Its core is the **`fly` dialect** — an MLIR layer that, like CuTe,
makes layout a first-class object with explicit algebra and coordinate→offset mapping,
then lowers through a composable pipeline to the GPU (ROCDL / the gfx ISA). The pitch is
identical to CuTe DSL's: author matrix-core kernels in *Python*, iterate in seconds
instead of fighting C++ templates, and still reach hand-tuned performance — and it was
built specifically to reduce the friction of bringing CUTLASS / CuTe DSL / FlashAttention
kernels into the AMD ecosystem.

```text
  CuTe DSL  : Python → MLIR → PTX → ptxas → SASS          (NVIDIA)
  FlyDSL    : Python → MLIR (fly dialect) → ROCDL → gfx ISA  (AMD)
              └─ same shape: a Python layout DSL over an MLIR back end ─┘
```

What it targets, and the status caveat:

- **Hardware:** the README's verified targets are `gfx942` (MI300X / MI308X) and
  `gfx950` (MI350X / MI355X) — the CDNA3 / CDNA4 data-center Instinct line — plus the
  newer `gfx1250` (MI450) and the `gfx1201` Radeon AI PRO R9700 workstation card. Every
  target is a data-center or professional part; **no consumer gaming GPU is on the list**
  (this is the fact §6's cost floor turns on).
- **Status:** experimental — installed as a standalone wheel (`pip install flydsl`),
  outside the official ROCm release and *for evaluation only*. So it's the right tool to
  *learn the model* and prototype, with the understanding that the interface can move
  under me (the same "public beta" caveat CuTe DSL carries, one notch earlier).

Reseal it: **FlyDSL is to Composable Kernel what CuTe DSL is to CUTLASS** — the Python
door into the same low-level, layout-explicit model. The *layout* and *atom* concepts
from [foundations §2–§4](01_cute_dsl_foundations.md#2-layouts-the-one-idea-cute-is-built-on)
transfer almost verbatim; what changes is the hardware atom underneath (an MFMA instead
of a WGMMA) and the memory names (LDS instead of shared memory). Learn CuTe DSL first
and FlyDSL is mostly a re-skin, not a new subject.

---

## 4. The portability ladder: how the skills transfer

The [landscape doc](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md)
gave one map for NVIDIA — *Triton for breadth, CuTe DSL for depth, CUDA/CUTLASS as the
bedrock you read*. The AMD map is the same shape with two names swapped, plus one rung
that spans both vendors:

```text
  PORTABLE BRIDGE   Triton            one kernel, both vendors (ROCm backend).
                                      the cheapest way to reach AMD at all, and how
                                      most kernels get there today.
        │
        ▼  (descend for the last ~2× on AMD silicon)
  AMD-NATIVE DEPTH  FlyDSL            the CuTe-DSL-equivalent rung: explicit layouts,
                                      MFMA atoms, LDS choreography, in Python.
        │
        ▼  (the bedrock you read / port)
  AMD BEDROCK       HIP C++ + CK      the CUDA-C++/CUTLASS-equivalent rung: what
                                      compiles underneath, and the target of a
                                      mechanical CUDA→HIP port.
```

The practical consequence for a learner: **Triton is the bridge, FlyDSL is the
commitment.** A Triton kernel written for the NVIDIA modules already *runs* on AMD, so
the cheapest first AMD experiment is "take a Triton kernel I built on NVIDIA, run it on
an MI300X, profile it, see where the 64-lane wavefront and LDS budget change the tuning."
FlyDSL is the deeper investment that mirrors the CuTe DSL rung — taken once the Triton
version has shown where the AMD-specific gap is.

---

## 5. A worked anchor: rewrite a real kernel, then benchmark it

The method this whole track points at — *take a kernel a real model runs, rewrite it in
the layout DSL, verify, profile, and push it past the existing baseline* — already has a
public worked example on AMD: AMD's own write-up of optimizing **Kimi-K2.5's fused
mixture-of-experts kernel on the MI300X using FlyDSL**. That is exactly the loop, end to
end, on the exact tool — a reference to study and then extend.

It's also the AMD mirror of the M30 *beat-the-baseline capstone*. The template generalizes
to any kernel:

```text
  1. PICK    a kernel a real model actually runs (attention, a fused MLP, a
             grouped GEMM) that has a weak or missing AMD path.
  2. REWRITE it in Triton (bridge) or FlyDSL (depth).
  3. VERIFY  numerics against PyTorch — the same correctness bar as every M-module.
  4. PROFILE on the real GPU (rocprof / the ROCm profilers — the AMD analog of the
             Nsight loop in the PTX doc).
  5. BEAT    the existing AITER / CK path, and document exactly why it's faster.
```

Mixture-of-experts shows up here only because it's the kernel the public example happens
to use ([Chapter 5's MoE section](../05_attention_and_kv_cache/06_moe.md) explains the
architecture and why a fused MoE kernel exists); the method is kernel-agnostic. The point is that on AMD, step 5 — *beating the
shipped baseline* — is reachable on more kernels than it is on NVIDIA, because fewer of
them are already tuned to the limit.

---

## 6. Running it with no hardware: the cost floor and ladder

The hardware question is sharper on AMD than anywhere else in this repo, so it gets its
own section. Two rules govern it.

**Rule 1 — never rent a persistent box; bill by the second.** A profiling session is
minutes of GPU time, not hours. Serverless / per-second platforms (Modal, RunPod
serverless) mean a kernel run costs cents: spin up, profile, tear down. This extends the
[ROADMAP §4 hardware table](../../ROADMAP.md) — same rented-GPU model, but pushed to its
per-second limit so the kernel-engineering phase stays affordable.

**Rule 2 — respect the floor; it's higher on AMD.** On NVIDIA a cheap consumer card is
still useful: a rented RTX 4090 (~\$0.30/hr on marketplace hosts, mid-2026) runs CUDA,
Triton, and correctness checks fine — only the Hopper/Blackwell *tensor-core* paths
(TMA, WGMMA, tcgen05) need the real data-center silicon. On AMD there is **no cheap
consumer-gaming rung at all**. FlyDSL's targets are all data-center Instinct or
professional cards — MI300X / MI308X (`gfx942`), MI350X / MI355X (`gfx950`), the newer
MI450 (`gfx1250`), and the Radeon AI PRO R9700 workstation card (`gfx1201`) — and none is
a \$300 gaming GPU like the 4090. Of those, the one actually rentable by the hour today
is the data-center MI300X, so AMD-native kernel work means renting a real Instinct GPU
from day one — briefly, but unavoidably.

The ladder, with prices labelled as listed mid-2026 and understood to move:

| Rung | Hardware | ~Rate (mid-2026, fluctuates) | What it's for |
| --- | --- | --- | --- |
| 0 | Rented consumer NVIDIA (RTX 4090) | ~\$0.30/hr (Vast.ai marketplace) | CUDA/Triton/Python fundamentals, correctness — *not* modern tensor-core paths |
| 1 | Rented **H100** (SXM) | ~\$2–2.7/hr (RunPod) | The main CuTe DSL learning + profiling target (Hopper) |
| 2 | **B200** slice | ~\$2.1–4/hr (neocloud/RunPod) | Blackwell paths (tcgen05/TMEM), FA-4-class kernels |
| 3 | **MI300X** | ~\$1.7–2/hr (RunPod, TensorWave) | The AMD arm — FlyDSL, AITER/CK comparisons |

The discipline that keeps it to pocket money: develop and debug on the cheapest rung
that can run the kernel, rent the expensive architecture *only* for the final
profile-and-benchmark, and kill it the instant the run finishes. Per-second billing makes
"ten minutes on an MI300X" a real, sub-dollar thing — which is the only reason an AMD
kernel track is feasible at all without owning the silicon.

---

## 7. Further reading

The AMD kernel stack is younger and moves fast; these are the sources I lean on for the
mapping in §2–§5 (AMD-published unless noted):

- **[ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)** and the
  **[FlyDSL announcement blog](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html)**
  (AMD) — the Python layout DSL of §3, the `fly` dialect, the target list, and the
  nightly-wheel status.
- **[Optimizing Kimi-K2.5's Fused MoE on MI300X with FlyDSL](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)**
  (AMD) — the worked rewrite-and-benchmark loop of §5, on the exact tool.
- **[AITER: AI Tensor Engine for ROCm](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)**
  (AMD) — the multi-backend op library (Triton/CK/ASM/HIP) that §2 maps onto FlashInfer.
- **[Composable Kernel User Guide](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/index.html)**
  (AMD) — the tile-based C++ template library, the CUTLASS analog of §2.
- **[What is HIP?](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html)**
  (AMD) — the CUDA-shaped portability layer and the CUDA→HIP port path.

## 8. What to carry forward

```text
the AMD stack mirrors the NVIDIA one rung-for-rung (§2)   -> any AMD pass of M24/M26
  HIP↔CUDA, CK↔CUTLASS, FlyDSL↔CuTe DSL, AITER↔FlashInfer
CUTLASS is NVIDIA-only → AMD is a separate parallel       -> why the AMD gap is wide
  stack, younger and thinner-staffed (§1)                      and worth contributing to
the hardware words change, the ideas don't: CU/wavefront  -> reading any AMD kernel;
  (64)/LDS/Matrix Core/MFMA (§2)                               re-tuning tiles for AMD
FlyDSL = Python layout DSL over MLIR, the CuTe DSL         -> the AMD-native depth rung,
  analog; experimental; data-center/pro parts (§3)             mirrors M23/M24
Triton is the write-once, run-on-both bridge; FlyDSL       -> first AMD experiment =
  is the deeper commitment (§4)                                run a Triton kernel on MI300X
rewrite → verify → profile (rocprof) → beat AITER/CK (§5)  -> the AMD mirror of M30
per-second billing + AMD has no consumer floor: rent       -> the cost model for the
  MI300X from day one, briefly (§6)                            whole AMD arm
```

The one sentence to keep: **AMD's kernel stack is a rung-for-rung mirror of NVIDIA's —
HIP for CUDA C++, Composable Kernel for CUTLASS, **FlyDSL** for CuTe DSL, AITER for
FlashInfer — so the layout-and-atom skills from the NVIDIA track transfer almost intact
(only the names and a 64-lane wavefront change), and because CUTLASS is NVIDIA-only,
AMD's parallel stack is younger and thinner-staffed, which makes a correct, well-profiled
kernel there genuinely new work rather than a re-derivation — reachable with no hardware
only because per-second MI300X rentals make brief profiling runs cost cents.**
