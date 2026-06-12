# CuTe DSL foundations: layouts, tensors, atoms

[Chapter 7](../07_writing_and_tuning_kernels/README.md) ended on a question:
*which* language do you write a kernel in? Its
[kernel-language landscape doc](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md)
gave the answer — Triton for breadth, **CuTe DSL** for depth, CUDA/CUTLASS as the
bedrock you read. This doc opens up CuTe DSL itself: the small set of ideas —
*layouts*, *tensors*, *atoms* — that the whole thing is built on, plus the hardware
vocabulary (MMA, DMA, TMA, WGMMA, tcgen05) that every low-level kernel doc throws
around without defining. It's the vocabulary the rest of this chapter assumes.

This is the conceptual prework for **M23** (CuTe DSL foundations) in the
[Phase 5 kernel-engineering track](../ROADMAP.md). Unlike Chapters 1–8, this chapter
isn't pre-M0 reading — it's for when the kernel-engineering track begins (or out of
curiosity now).

**Honesty up front.** This project has no local NVIDIA GPU (it's remote-only), so
nothing here is something I ran. It's an explanation of the concepts and the public
documentation; the running and measuring happens in the M23–M30 labs on rented
GPUs. Where I quote a number it's from a public source, labelled as such.

Prerequisites: the
[kernel-language landscape](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md)
(the productivity-vs-control ladder, the "three knobs") and
[Chapter 1 — Hardware fundamentals](../01_hardware_fundamentals/README.md) (SM,
warps, shared memory, Tensor Cores). A bare **§N** below means *this doc's* section
N; references to other docs are always links.

---

## 1. Where CuTe DSL sits: CUDA C++ → CUTLASS → CuTe → CuTe DSL

The four names get used interchangeably, and they shouldn't be. They're a stack,
each layer built on the one above:

```text
  CUDA C++     The base GPU language: C++ plus GPU keywords (__global__, __shared__).
               You write threads, shared memory, sync by hand.
               Compiles:  CUDA C++ → PTX (virtual asm) → SASS (real machine code).
                 │
                 │  CUTLASS is a C++ template LIBRARY written IN CUDA C++
                 ▼
  CUTLASS      Not a language — a toolkit of GEMM/attention building blocks, so you
               don't hand-roll tiling, Tensor-Core use, and pipelining every time.
                 │
                 │  its foundational abstraction layer is called CuTe
                 ▼
  CuTe         The algebra at CUTLASS's core: Layouts, Tensors, Atoms (§2–§4).
               CUTLASS 3.x rewrote itself on top of CuTe.
                 │
                 │  CuTe DSL = CuTe exposed in Python (CUTLASS 4.0, 2025)
                 ▼
  CuTe DSL     A Python front-end giving the SAME CuTe abstractions and the SAME
               hardware control as the C++ version — without C++ template metaprogramming.
```

Two things to fix in the mental model:

- **CUDA C++ is the base, not a sibling.** CUTLASS is *written in* CUDA C++; CuTe
  DSL *compiles down to* the same PTX → SASS that CUDA C++ produces. They're
  different doors into the same room, not unrelated languages.
- **CuTe DSL is *low* on the control ladder, despite being Python.** This is the
  trap the
  [landscape doc](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md)
  warns about: "Python" is the host syntax, not the abstraction level. Unlike
  Triton, which *hides* layouts and warp choreography, CuTe DSL hands them to you.

### Why being Python costs no performance

The intuition "Python is slow, so CuTe DSL must be slow" is wrong, and it's worth
killing precisely:

```text
  CuTe DSL Python code  ──runs ONCE, at compile time──►  generates a kernel
                                                          (Python → MLIR → PTX → SASS)
  the kernel that runs on the GPU  ──►  pure SASS, ZERO Python inside it
```

The Python is *metaprogramming* — it runs on the CPU, once, to emit a GPU kernel
(exactly like Triton, exactly like C++ templates). What executes on the GPU has no
Python in it. So CuTe DSL drives the same hardware primitives (WGMMA, TMA, tcgen05;
§6) as the C++ path and reaches the same performance ceiling. **FlashAttention-4 is
written in CuTe DSL** and its paper reports it both beats Triton on a B200 and
compiles ~20–30× faster than the equivalent C++ CUTLASS — same speed, far less
authoring pain. That's the whole reason this rung exists.

---

## 2. Layouts: the one idea CuTe is built on

Everything in CuTe is a **layout**. A layout is how you describe *where every
element of a tensor lives* — in global memory, in shared memory, or spread across
the threads of a warp. Formally it's a pair, written `Shape:Stride`:

```text
  Layout = (Shape, Stride)          e.g.   (3,4):(4,1)
           │       │
           │       └─ Stride: how far to step in memory per +1 in each coordinate
           └───────── Shape:  the logical size in each dimension
```

A layout is a *function*: give it a logical coordinate, it returns a 1-D memory
offset, by taking the dot product of the coordinate with the stride.

```text
  offset(i, j) = i · stride₀ + j · stride₁

  Layout (3,4):(4,1)   →   offset(i,j) = 4i + j      (this is ROW-major)

    logical coord → offset           the 3×4 tile in memory:
      (0,0) → 0    (0,3) → 3            0   1   2   3
      (1,0) → 4    (1,2) → 6           4   5   6   7
      (2,3) → 11                        8   9  10  11
```

The stride is what sets the memory order. Flip it and you flip the layout:

```text
  (3,4):(4,1)   stride 1 ACROSS a row    → rows are contiguous   = ROW-major
  (3,4):(1,3)   stride 1 DOWN  a column  → columns are contiguous = COLUMN-major
```

That's the whole black box, opened one level. **Reseal it like this:** you don't
need the full layout algebra (composition, tiling, complementation) to read Phase 5
— you need to know that *a layout is a first-class object that makes data placement
explicit*. That explicitness is exactly the control Triton hides and CuTe exposes.
When a CuTe kernel is fast, it's usually because someone chose layouts that make the
loads coalesced (Tensor-Core–friendly fragments, bank-conflict-free shared memory)
— the same goals as
[the matmul tuning ladder](../07_writing_and_tuning_kernels/01_writing_and_tuning_a_matmul.md#6-rung-2--shared-memory-tiling),
now expressed as layout algebra instead of left to a compiler.

---

## 3. Tensors: a pointer plus a layout

A **tensor** in CuTe is the simplest possible thing once you have layouts: a
*pointer* (where the data starts) plus a *layout* (how to index it).

```text
  Tensor = pointer + Layout
           │          │
           │          └─ the (Shape,Stride) map from §2
           └──────────── the base address in some memory (global / shared / register)
```

This is why CuTe code reads as "make a tensor over this shared-memory buffer with
this layout, then slice it." The *same* data can be viewed through different
layouts (a global-memory tile and its shared-memory copy have different strides),
and the layout is what records the difference. The tensor carries no data of its
own — it's a typed *view*.

---

## 4. Atoms: the smallest unit of hardware work

This is the term you asked about, and it's the heart of the low-level model. An
**atom** is *one hardware instruction's worth of work* — the smallest indivisible
operation the hardware exposes, wrapped so CuTe can compose it.

```text
  MMA atom    one Tensor-Core matrix-multiply-accumulate instruction
              (e.g. one WGMMA on Hopper, one tcgen05 MMA on Blackwell — §6)
  Copy atom   one hardware copy instruction
              (e.g. one TMA bulk-copy, or one cp.async — §6)
```

You rarely compute a whole tile with a single atom — a tile is bigger than one
instruction. So CuTe builds a **tiled MMA** / **tiled copy**: take one atom and
*replicate it across a layout of threads* to cover the whole tile.

```text
  one MMA atom  ×  a thread layout  =  tiled MMA  (covers a BM×BN×BK tile)
       ↑                  ↑
   one instruction   which thread does which fragment
```

The "which thread does which fragment" part is a layout too — CuTe calls it a
**Thread-Value (TV) layout**: a map from a `(thread_id, value_id)` pair to the
logical tile coordinate that thread+value owns.

```text
  TV layout:  (thread, value)  →  (row, col) in the tile
              "thread 5's 2nd value handles element (1, 10)"
```

That single concept is the cleanest way to see the Triton-vs-CuTe difference:

```text
  Triton    you say "compute this tile"; the COMPILER builds the TV layout
            (decides which thread touches which element) — hidden from you.
  CuTe DSL  YOU write the TV layout — you assign threads to data explicitly.
            More control, more to get right.
```

That extra control is the cost *and* the point: it's how you hand-tune the data
movement and Tensor-Core fragment assignment that a compiler would otherwise guess.

---

## 5. Host vs device: `@jit` and `@kernel`

CuTe DSL splits your Python into two worlds with two decorators (this is NVIDIA's
documented model):

```text
  @jit      HOST code. Runs on the CPU at compile time. Sets up layouts/tensors,
            picks tile sizes, and LAUNCHES kernels. Can call other @jit functions
            (inlined) or @kernel functions (a real GPU launch via the driver).
  @kernel   DEVICE code. The GPU kernel itself, dynamically compiled to a GPU
            symbol. This is what runs on the SMs.
```

Mapping it to the §1 picture: the `@jit` host function is where the *metaprogramming*
happens — it runs once, builds the kernel, and hands it to the driver; the `@kernel`
body is what becomes SASS and executes on the GPU. The full path:

```text
  your Python (@jit/@kernel)  →  MLIR (a compiler IR)  →  PTX  →  ptxas  →  SASS
```

`MLIR` (*Multi-Level Intermediate Representation*) is the compiler framework CuTe
DSL lowers through; `ptxas` is NVIDIA's PTX→SASS assembler (the same back end the
CUDA C++ path uses).

---

## 6. The hardware vocabulary, defined once

These are the terms that show up unexplained in every kernel-engineering source.
Defined here so the FlashAttention and pipeline docs can just use them. The
"Tensor-Core matmul" and "async copy" ideas trace back to
[execution model §11](../01_hardware_fundamentals/03_gpu_model.md#11-tensor-cores-accelerate-the-matmul-pattern)
and
[the matmul ladder's pipelining rung](../07_writing_and_tuning_kernels/01_writing_and_tuning_a_matmul.md#9-rung-5--latency-hiding-pipelining-and-async-copy).

| Term | Stands for | What it is | GPU gen |
| --- | --- | --- | --- |
| **MMA** | Matrix Multiply-Accumulate | One Tensor-Core instruction: multiply two small matrix *fragments*, add into an accumulator | all Tensor-Core GPUs |
| **WGMMA** | Warpgroup MMA | An *async* MMA issued by a whole warpgroup (4 warps); fire it and the Tensor Cores work in the background | Hopper |
| **tcgen05** | Tensor Core gen-5 | Blackwell's MMA instruction set (a.k.a. UMMA); operands come from shared memory and a new on-chip **TMEM** | Blackwell |
| **TMEM** | Tensor Memory | A memory pathway on Blackwell sitting close to the Tensor Cores, feeding their higher throughput; allocated/managed explicitly | Blackwell |
| **DMA** | Direct Memory Access | Hardware that moves data between memory regions *without* the compute cores doing the copy — they stay free to compute | general concept |
| **TMA** | Tensor Memory Accelerator | Hopper's DMA *engine* for bulk async tile copies global↔shared (a copy atom, §4) | Hopper+ |
| **cp.async** | async copy | The earlier (Ampere) async global→shared copy that doesn't occupy the math lanes | Ampere+ |
| **Warp specialization** | — | Splitting a block's warps into roles — *producer* warps that only issue TMA copies, *consumer* warps that only issue MMAs | Hopper+ |
| **mbarrier** | memory barrier | The hardware barrier producers/consumers use to signal "the tile I copied is ready" / "the MMA finished" | Hopper+ |
| **2-CTA MMA** | two-CTA MMA | Two cooperating thread blocks (*CTAs* — Cooperative Thread Arrays, the formal name for a block), one per SM, share operands for one MMA, cutting per-SM shared-memory bandwidth | Blackwell |

The pattern to notice: every Hopper/Blackwell term is about **doing the matmul
asynchronously and feeding it without stalling** — async Tensor Cores (WGMMA,
tcgen05), a DMA engine to feed them (TMA), specialized warps to overlap the two
(producer/consumer), and barriers to coordinate (mbarrier). That async-and-overlap
story is what the [FlashAttention Rosetta Stone](02_flashattention_rosetta_stone.md)
shows being assembled.

---

## 7. Why Triton and CuTe can differ in speed: the two gaps

§1 said CuTe DSL and CUTLASS C++ reach the same performance ceiling. The fair next
question: then why does a *Triton* FlashAttention run slower than the CuTe one
(FA-4 is 1.2–3.2× over Triton on Blackwell)? If both are "just Python that generates
a kernel," what's the difference — and can the Triton compiler simply be improved
until it closes the gap? The answer turns on splitting one apparent gap into two:

```text
  GAP A — compiler quality
    For a FIXED programming model, the compiler's heuristics aren't as good as an
    expert's hand-tuning (a worse tile size, a clumsier pipeline depth).
    → CLOSEABLE by a better compiler (cost models, autotuning).

  GAP B — expressiveness
    The programming model CANNOT SAY the optimization. The control isn't in the
    language, so no compiler — however smart — can emit it.
    → NOT closeable by a better compiler. You must change the LANGUAGE.
```

A clarification, since it trips people up: **all** of these — Triton, CuTe DSL,
CUDA C++ — compile down to **SASS**, the GPU's actual machine code. None of them
"stops short" of the hardware; there is no other way to run on a GPU. So the gap is
*which* SASS gets generated, not *whether* you reach it.

**The deep reason for Gap B: a high-level language is a lossy compression of intent.**
The compiler reconstructs *how* from *what* — and only from the information you
actually supplied. A Triton kernel says "here are tiles, do `tl.dot`, do the
softmax." It does **not** say "make these 4 warps TMA producers, those 8 WGMMA
consumers, and sync them with these mbarriers in this ping-pong order." Triton's
model — tile-based **SPMD** (*Single Program, Multiple Data* — every thread runs the
same code on a different tile, so all threads share one role) — has no syntax for
that (the [Thread-Value layout and warp roles of §4](#4-atoms-the-smallest-unit-of-hardware-work)
are precisely what it hides). That information simply isn't in the program, so no
amount of compiler cleverness can emit it. **You can't lower what the model can't
express.**

Mapping FlashAttention's wins to the two gaps:

| Optimization | Gap | Can a better Triton *compiler* do it? |
| --- | --- | --- |
| Coalescing, basic tiling, pipelining | A | **Yes** — Triton already gets ~80–95% of peak |
| Best tile/stage config for a shape | A | Yes — that's `@triton.autotune` |
| Warp specialization (producer/consumer) | B | **No** — not expressible in classic Triton |
| Explicit TMA + mbarrier ping-pong pipeline | B | No — needs language-level control |
| Polynomial-emulated softmax on Blackwell | — | No — it's an *algorithm* change, not compilation |

The first two rows are why Triton gets you most of the way; the rest are the last ~2×.

**Why the gap is biggest on brand-new hardware.** When a GPU ships, the compiler
hasn't learned to schedule its new instructions (WGMMA at Hopper launch, tcgen05 at
Blackwell launch), so humans hand-write CuTe to exploit them first. Over the next
year the Gap-A part gets absorbed into the compiler and Triton becomes competitive
again — until the next architecture reopens it. So part of the gap is *temporary*
(compiler immaturity, which closes) and part is *structural* (new features live in
the low-level model first).

**How you bridge it — and the catch.** Two levers, one per gap:

```text
  close Gap A  →  improve the compiler / autotuner          (stay high-level)
  close Gap B  →  EXTEND the language to expose the control
                  …which makes the high-level tool LOWER-level — i.e. more like CuTe
```

That second lever is exactly what **Gluon** and **TLX** are
([landscape doc §5](../07_writing_and_tuning_kernels/03_kernel_languages_landscape.md#5-2025--the-tile-dsl-swarm)):
they add warp specialization and explicit layouts to Triton, closing Gap B by
handing Triton CuTe-like controls. So "can we just make Triton as fast as CuTe?" —
yes, but mostly by making Triton low-level enough to *say* what CuTe says.
**The gap is the price of abstraction: a high-level program carries less information
than a hand-tuned one, so you either make the compiler infer the rest (Gap A,
asymptotic) or supply it yourself (Gap B — which means descending the ladder).**

### Where Gluon and TLX sit relative to CuTe

Close, but not identical — and not even identical to each other. The fine ordering,
inside the "explicit-control" band:

```text
  TLX        MID/low: you ANNOTATE a warp-specialization strategy; the COMPILER
             realizes it (warp-group granularity). Stays in the Triton compiler.
  Gluon      LOW: like CUDA C++, you both DEFINE the strategy AND write it out in
             low-level code. Stays in the Triton compiler/IR.
  CuTe DSL   LOWEST / most complete: exposes the FULL memory hierarchy — global,
             shared, register, and Blackwell's TMEM — plus the full layout algebra
             and the native MMA/copy atoms. NVIDIA's own model; gets new hardware FIRST.
```

Gluon is about as low as CuTe in control; TLX sits a notch higher (the compiler
still helps realize your warp-spec strategy); CuTe goes deepest and most complete.
The real dividing line is heritage: **Gluon/TLX are *extensions that keep the Triton
compiler*** — you descend without leaving Triton — **while CuTe DSL is the separate,
more complete low-level model at CUTLASS's foundation**, which is why the newest
atoms (Blackwell tcgen05/TMEM) appear there first. For a Triton-first learner,
Gluon/TLX is the gentler descent; CuTe DSL is the deeper, hardware-leading commitment.

---

## 8. Further reading

CuTe DSL is new and moves fast; these are the sources I rely on for layouts, atoms,
and the hardware underneath:

- **[CUTLASS](https://github.com/NVIDIA/cutlass)** (NVIDIA) — home of CuTe and the Python
  CuTe DSL: the layout / tensor / atom abstractions of §2–§4, with the C++ source the DSL
  mirrors.
- **[Colfax Research — CUTLASS tutorials](https://research.colfax-intl.com/category/papers/tutorials/)**
  (Colfax) — the clearest worked tutorials on CuTe layouts, tiled MMA/copy, and Hopper
  WGMMA GEMMs; the step-by-step companion to §2–§4.
- **[NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)**
  (NVIDIA) — the hardware the atoms in §4 map onto: WGMMA, TMA, and the asynchronous units
  a CuTe kernel orchestrates.
- **[Triton](https://triton-lang.org/)** (OpenAI) — the tile-level language §1 contrasts
  CuTe DSL against; reading both shows what each makes explicit versus hides.

## 9. What to carry forward

```text
the stack: CUDA C++ → CUTLASS → CuTe → CuTe DSL (§1)   -> M23, M24 (three-way GEMM)
Python is compile-time; no runtime perf tax (§1)        -> the reason Phase 5 uses CuTe DSL
a Layout is Shape:Stride, a coord→offset map (§2)        -> M23, reading any CuTe kernel
a Tensor is pointer + Layout (§3)                        -> M23
an Atom is one hardware instruction; tiled MMA/copy      -> M24, M27 (pipelines)
  = atom × thread layout; the TV layout (§4)
@jit host vs @kernel device; MLIR→PTX→SASS (§5)          -> M23, first CuTe DSL kernel
the hardware glossary: WGMMA, TMA, tcgen05, TMEM,        -> M27 (Hopper), M28 (Blackwell)
  warp specialization, mbarrier (§6)
the two gaps: A=compiler quality (closeable), B=         -> M24, M26 — attribute each
  expressiveness (needs a lower language); Gluon/TLX (§7)      % of the Triton↔CuTe gap
```

The one sentence to keep: **CuTe DSL is Python that runs once at compile time to
generate a kernel with no runtime tax, built on three ideas — a *layout* (`Shape:Stride`)
that makes data placement an explicit coordinate→offset map, a *tensor* (pointer +
layout) that views memory through it, and an *atom* (one hardware MMA or copy
instruction) that you replicate across a thread layout to cover a tile — which
together hand you the layouts, Tensor-Core fragments, and warp choreography that
Triton hides, at the same performance ceiling as CUDA C++.**
