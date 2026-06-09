# From source to SASS: the compilation pipeline, stage by stage

[The foundations doc](01_cute_dsl_foundations.md)
said every kernel language compiles down to **SASS**, the GPU's machine code, and
[§7](01_cute_dsl_foundations.md#7-why-triton-and-cute-can-differ-in-speed-the-two-gaps)
said the performance gap is *which* SASS gets generated, not *whether* you reach it.
This doc opens up that pipeline: the stages a kernel passes through, what each
intermediate format actually *looks like* (the real syntax, and what language it's
written in), and how the **same** kernel diverges through the Triton versus CuTe DSL
pipelines for one problem. Being able to dump and *diff* these stages is the
inspection skill that **M23/M24** are built on.

**Honesty up front.** No local NVIDIA GPU here (remote-only), so I haven't run these
pipelines myself — the code snippets are *representative* of the syntax (simplified,
the kind you'd actually see), not captured output. PTX is documented and stable;
SASS is architecture-specific and not officially documented, so its mnemonics are
shown as the real-but-version-dependent forms you'd find disassembling a Hopper
kernel. The actual dumping and diffing is M23/M24 on a rented GPU.

Prerequisites: [CuTe DSL foundations](01_cute_dsl_foundations.md) (the stack, the
hardware glossary, the two gaps). A bare **§N** means *this doc's* section N.

---

## 1. The pipeline: everyone ends at SASS

A GPU executes exactly one thing: **SASS** (*Streaming ASSembler* — the binary
machine code for one specific GPU architecture). Every kernel language, no matter
how high-level, must reach it. The surprise is how *similar* the back ends are:

```text
  Triton:    Python(@triton.jit) ─► TTIR ─► TTGIR ─► LLVM IR ─► PTX ─► ptxas ─► SASS
                                     │        │                          (in cubin)
                                     │        └─ adds hardware LAYOUTS (thread/warp →
                                     │           data) + pipelining, coalescing
                                     └─ tile-level, machine-INDEPENDENT

  CuTe DSL:  Python(@jit/@kernel) ─► MLIR (CUTLASS dialects) ─► PTX ─► ptxas ─► SASS
                                     └─ carries your explicit layouts + atoms

  CUDA C++:  .cu ─► (nvcc / LLVM) ─────────────────────────► PTX ─► ptxas ─► SASS
```

Two facts to read off this:

- **Both Triton and CuTe DSL are MLIR-based, and all three share the *same* tail:
  `PTX → ptxas → SASS`.** `ptxas` is NVIDIA's PTX-to-SASS assembler; it's the
  identical back end for everyone. So the divergence that sets performance happens
  *upstream* — in the middle IR, in what each pipeline decides to put into the PTX.
- **`PTX` (*Parallel Thread Execution*)** is a *virtual* assembly language —
  GPU-generation-independent, forward-compatible, shipped inside binaries
  ([stack doc](../02_cuda_software_stack/01_the_stack.md#what-people-actually-use)
  introduced it). `ptxas` then specializes it to the real machine code (SASS) for
  the target GPU. **`cubin`** (CUDA binary) is the final packaged blob.

`MLIR` (*Multi-Level Intermediate Representation*) is a framework for building
compiler IRs out of **dialects** — namespaced mini-languages, each a set of
operations and types. Triton's stages (`TTIR`, `TTGIR`) are MLIR dialects; so are
CuTe DSL's. That shared foundation is why "improve the compiler" means improving
MLIR passes for both.

---

## 2. What each format actually looks like

A tour down the stack, with representative syntax. The same idea — load some data,
multiply-accumulate it — gets re-expressed at each level, lower and more
hardware-specific each time.

### 2.1 TTGIR (Triton GPU IR) — MLIR text that carries *layouts*

This is the stage where the thread→data mapping (what
[§4 of foundations](01_cute_dsl_foundations.md#4-atoms-the-smallest-unit-of-hardware-work)
called a Thread-Value layout) becomes explicit *inside Triton* — as MLIR **layout
attributes** (`#blocked`, `#mma`):

```mlir
#blocked = #triton_gpu.blocked<{sizePerThread=[1,4], threadsPerWarp=[8,4], ...}>
#mma     = #triton_gpu.nvidia_mma<{versionMajor=3, ...}>      // a Hopper MMA layout
  %a = tt.load %aptr : tensor<128x64xf16, #blocked>           // load a tile
  %d = tt.dot %a, %b, %c : ... -> tensor<128x128xf32, #mma>   // → a tensor-core op
```

The point: the layout decisions Triton makes *for* you (and that CuTe makes you
write) live right here, as attributes on tensors. This is where a Gap-A
choice — a worse layout — would show up.

### 2.2 PTX — virtual assembly, human-readable, generation-independent

Typed *virtual* registers (`%f`=float, `%r`=32-bit int, `%rd`=64-bit), one
instruction per line. The plain ops are unremarkable; the *fast* ops only appear if
the source expressed them:

```ptx
  ld.global.f32   %f1, [%rd2];                               // a plain load
  fma.rn.f32      %f4, %f1, %f2, %f3;                         // scalar multiply-add
  // these only show up if the kernel asked for the hardware feature:
  wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 ... ;  // Hopper async tensor-core MMA
  cp.async.bulk.tensor.2d.shared::cluster.global.tile ... ;  // TMA bulk copy
  mbarrier.try_wait.parity.shared.b64 ... ;                  // producer/consumer sync
```

### 2.3 SASS — the real machine code for one GPU

Real hardware registers (`R0`–`R255`), architecture-specific (sm_90 Hopper differs
from sm_100 Blackwell), and **not officially documented** — you read it by
disassembling a `cubin` (§6). Each SASS line is roughly what one PTX instruction
became:

```sass
  LDG.E   R4, [R2]                       // load        ← was  ld.global.f32
  FFMA    R6, R4, R5, R6                  // multiply-add ← was  fma.rn.f32
  HGMMA.64x192x16.F32  R8, ...            // tensor core ← was  wgmma   (Hopper "GMMA" family)
  UTMALDG ...                             // TMA load    ← was  cp.async.bulk.tensor
  LDGSTS.E ...                            // async copy  ← was  cp.async (Ampere style)
  BAR.SYNC                                // barrier
```

(In between, Triton also passes through **LLVM IR** — an `SSA`-form typed IR, *Static
Single Assignment*: every value is written exactly once — with GPU-specific intrinsics
for the hardware ops. It's the bridge from the MLIR dialects to PTX; I list it for
completeness but you rarely read it.)

---

## 3. The instruction correspondence

The whole "which SASS" question reduces to which of these rows the PTX contains. The
plain rows appear in *every* kernel; the asynchronous/tensor-core rows appear only
when the source could express them:

| What it does | PTX | SASS (Hopper, representative) | Appears when |
| --- | --- | --- | --- |
| Load / store | `ld.global` / `st.global` | `LDG` / `STG` | always |
| Multiply-add | `fma.rn.f32` | `FFMA` | always |
| Tensor-core MMA (Ampere) | `mma.sync` | `HMMA` / `IMMA` | tensor-core matmul |
| Tensor-core MMA (Hopper, async) | `wgmma.mma_async` | `HGMMA` / `IGMMA` / `QGMMA` | source uses WGMMA |
| Async copy (Ampere) | `cp.async` | `LDGSTS` | source uses `cp.async` |
| Bulk async tile copy (TMA) | `cp.async.bulk.tensor` | `UTMALDG` | source uses TMA |
| Async-completion barrier | `mbarrier.*` | barrier ops | warp specialization |

The "GMMA" family (`HGMMA`=half, `IGMMA`=int, `QGMMA`=fp8) is the SASS form of
Hopper's WGMMA; `UTMALDG` is the SASS form of a TMA load. These are the exact
instructions you'd grep a SASS dump for to answer "is this kernel using the fast
path?"

---

## 4. How two pipelines diverge for *one* problem

Take a single problem — a tiled GEMM on Hopper — and run it through both pipelines.
The math is identical; the generated code is not, and you can *see* it:

| In the generated code | Triton (classic — hides warp roles) | CuTe DSL (warp-specialized) |
| --- | --- | --- |
| Tensor-core op | `wgmma` → `HGMMA`, **uniform across all warps** | `wgmma` → `HGMMA`, **consumer warps only** |
| Data movement | `cp.async` → `LDGSTS` | `cp.async.bulk.tensor` → `UTMALDG` (TMA), **producer warps only** |
| Warp structure | every warp runs the **same** code | warps **branch by role** (producer vs consumer) + `mbarrier` |
| Net | good SASS — ~80–95% of peak | the async/overlap structure → the last ~2× |

The decisive difference isn't a single instruction — it's the **warp structure**:
CuTe's SASS contains *two different code paths* for producer and consumer warps,
coordinated by barriers, which classic Triton's uniform output — every warp running
the same code (**SPMD**, *Single Program, Multiple Data*) — simply doesn't have
([the expressiveness gap, §7 of foundations](01_cute_dsl_foundations.md#7-why-triton-and-cute-can-differ-in-speed-the-two-gaps)).

**But the gap is problem-dependent.** For a *simple* kernel — a vector add, an
elementwise RMSNorm — the two pipelines produce **nearly identical SASS**, because
both compilers know how to emit good coalesced `LDG`/`STG`/`FFMA`. There's no
warp-specialization to express, so there's nothing to diverge on. The SASS
difference *grows* with kernel complexity and hardware novelty — which is exactly
the shape of Gap B.

---

## 5. Where in the pipeline the gap lives

Mapping the [two gaps](01_cute_dsl_foundations.md#7-why-triton-and-cute-can-differ-in-speed-the-two-gaps)
onto the stages of §1:

```text
  GAP A (compiler quality)   lives in the MIDDLE passes and ptxas:
                             a worse #blocked/#mma layout in TTGIR, a clumsier
                             pipeline, suboptimal register allocation in SASS.
                             → improvable WITHOUT changing the source.

  GAP B (expressiveness)     lives at the SOURCE → PTX boundary:
                             if the kernel never expressed warp specialization,
                             the PTX has no wgmma-in-producer/consumer structure
                             and no cp.async.bulk.tensor → the SASS literally
                             cannot contain UTMALDG + role-split warps.
                             → un-improvable until the LANGUAGE can say it.
```

So when you diff two SASS dumps and one has `UTMALDG` + role-split warps and the
other doesn't, you're not looking at a smarter `ptxas` — you're looking at a PTX
that *contained* the instruction versus one that *couldn't*. **The diff is Gap B,
made visible.**

---

## 6. Inspecting it yourself

None of this is theoretical — every stage is dumpable, and doing so is the M23/M24
skill:

```text
  Triton IRs + PTX:   TRITON_KERNEL_DUMP=1  TRITON_DUMP_DIR=./dump
                        → writes  kernel.ttir  .ttgir  .llir  .ptx
  PTX from any cubin: cuobjdump -ptx   a.cubin        (Triton, CuTe, or C++ — any)
  SASS:               cuobjdump -sass  a.cubin    or   nvdisasm a.cubin
```

The exercise that makes the whole chapter click: compile your Triton FlashAttention
and your CuTe DSL one, `cuobjdump -sass` both, and **diff them**. The
`UTMALDG`/`HGMMA`/producer-consumer structure that appears in one and not the other
*is* the performance gap, in machine code you can point at.

---

## 7. The PTX layer as a unifying substrate (and a caution)

§1 showed every DSL converges at PTX. That convergence is itself something people
build on: because Triton, CuTe, TileLang, and ThunderKittens *all* compile to PTX,
you can extract and compare the PTX each emits for the same op — and even try to
*combine* the optimizations each one captured, since at the PTX layer they all
coexist. A 2026 effort
([Standard Kernel](https://standardkernel.com/blog/reimagining-kernel-generation-at-the-ptx-layer-learning-from-and-outperforming-dsls/))
automates exactly this: a program-analysis + LLM hybrid that mines the PTX of
several DSLs and merges their optimizations, reporting kernels competitive with — and
in places slightly ahead of — CUTLASS / Triton / TileLang on an H100. It is the
automated, scaled-up form of the dump-and-diff in §6.

Two things to keep in proportion, straight from the verify-first ethos
([matmul §17](../07_writing_and_tuning_kernels/01_writing_and_tuning_a_matmul.md#17-measuring-optimal--the-honest-bar)):

```text
  CORRECTNESS FIRST  the post reports only timings — no correctness methodology.
                     Combining PTX across sources is exactly where fast-but-WRONG
                     kernels appear; a speedup on an unverified kernel is not a result.
  READ THE MARGINS   the GEMM wins are thin (~5% over CUTLASS, and a tie — "first in
                     5 of 10 trials" — at the large size that matters); the headline
                     ~67% is a small memory-bound op vs ONE baseline. Thin wins can
                     live inside autotuning noise.
```

And a calibration for *your* learning: even this PTX-focused effort finds "LLMs
alone still fall short" at the PTX layer and bolts on program analysis — so
pure-LLM PTX generation isn't here yet. The durable skill is the opposite of
hand-writing PTX: **read** the PTX your kernels generate, profile, and change at the
*source* level, reaching for inline PTX only surgically.
[The next doc](04_reading_and_optimizing_ptx.md) walks that loop end to end.

---

## 8. What to carry forward

```text
everyone ends at SASS; Triton & CuTe share PTX→ptxas→SASS (§1)  -> M23, the mental model
the four formats: TTGIR(layouts) → PTX(virtual asm) →            -> M23, reading dumps
  LLVM IR → SASS(real machine code) (§2)
the PTX→SASS instruction map (wgmma→HGMMA, TMA→UTMALDG) (§3)      -> M24, grep the fast path
same problem, different SASS = different WARP STRUCTURE (§4)       -> M24/M26, the GEMM diff
Gap A = middle passes/ptxas; Gap B = source→PTX boundary (§5)     -> attributing the gap
dump + diff: TRITON_KERNEL_DUMP, cuobjdump -sass, nvdisasm (§6)   -> M23/M24, do it for real
the PTX layer is a shared substrate to mine/compare —            -> M24/M25, verify first
  but verify correctness + read the margins (§7)
```

The one sentence to keep: **every kernel language compiles down to the same SASS
machine code through the same `ptxas` back end — Triton and CuTe DSL even share an
MLIR front — so the performance difference isn't *whether* you reach SASS but *which*
SASS gets generated, which is set upstream by what the source could express: a
warp-specialized CuTe kernel emits PTX with `wgmma`+`cp.async.bulk.tensor` that
becomes `HGMMA`+`UTMALDG` in a producer/consumer warp structure, while a classic
Triton kernel emits uniform-SPMD PTX that can't — and you can dump both with
`cuobjdump -sass` and see the gap as a literal diff.**
