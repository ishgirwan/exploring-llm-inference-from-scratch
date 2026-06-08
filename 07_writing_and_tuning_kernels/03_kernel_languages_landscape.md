# Choosing a kernel language: the 2021 → 2026 landscape

[Section 1](01_writing_and_tuning_a_matmul.md) is the *craft* of tuning a kernel.
[Section 2](02_reading_a_real_kernel.md) reads *one* real kernel — and it's in
*one* language, Triton. Both leave the same question hanging, the one that made me
write this doc: **when I sit down to write a kernel, which language do I even
use?** A single tweet's worth of jargon — CuTe DSL, TileLang, ThunderKittens,
Gluon, TLX, cuTile — and no obvious answer for which is the 2026 default.

This doc is the map. It does three things: lays out the *one axis* that orders
every kernel-authoring tool, walks the *timeline* of how that axis went from a
clean two-way choice to a crowded spectrum (and why the hardware forced it), and
ends on **what I'll actually reach for, and in what order**. It's a survey, not a
build — the building is M1–M6 and M16 in the [Roadmap](../ROADMAP.md).

**Honesty up front.** This project has no local NVIDIA GPU (it's remote-only), so
nothing below is benchmarked by me. Where I quote a performance number it's from a
public source and labelled as such — the same rule as
[Section 2](02_reading_a_real_kernel.md), where every figure is the original
author's measurement, not mine. The facts here are drawn from public docs, papers,
and project repos; I write them as flat statements only where I could confirm them.

Prerequisites: [§1](01_writing_and_tuning_a_matmul.md) (the optimization ladder
and its tuning-knob vocabulary) and the
[CUDA software stack doc](../02_cuda_software_stack/01_the_stack.md) (the
*call-vs-build* framing, and the definitions of **DSL**, **PTX**, **SASS**).
Next: M1 in the [Roadmap](../ROADMAP.md).

Throughout this doc, a bare **§N** means *this doc's* section N. References to the
other sections of this chapter are always written as links.

---

# Part A — The one axis that orders every tool

## 1. The ladder: productivity vs. control

Every tool for getting a GPU to run your math sits on a single spectrum. At the
top you describe *what* you want and a compiler decides *how*; at the bottom you
dictate every byte's movement yourself. Higher = faster to write, more left on the
table; lower = peak speed, far more expert-hours.

```text
  HIGHEST ABSTRACTION  (write fast — the compiler makes the hardware decisions)
  │
  │  PyTorch eager / torch.compile     you don't write a kernel; the framework
  │                                    (or its compiler) emits one for you
  │  Triton · Helion                   TILE languages: you think in tiles of
  │                                    data, the compiler assigns threads,
  │                                    stages shared memory, pipelines
  │  CuTe DSL · cuTile · TileLang ·    EXPLICIT-LAYOUT DSLs: you also control
  │    ThunderKittens · Gluon · TLX    data layout, the tensor-core instruction,
  │                                    and which warp does what
  │  CUTLASS  (C++ templates)          industrial GEMM/attention building blocks
  │  CUDA C++                          raw threads, shared memory, you manage all
  │  PTX / SASS                        virtual / real GPU assembly — last resort
  │
  LOWEST ABSTRACTION   (peak speed — you make every hardware decision yourself)
```

A few terms, defined once, since the rest of the doc leans on them:

- **DSL** — *Domain-Specific Language*: a small programming language built for one
  job (here, writing GPU kernels), rather than a general one. Triton, CuTe DSL,
  and TileLang are all DSLs embedded in Python. (Fuller intro in the
  [stack doc](../02_cuda_software_stack/01_the_stack.md#what-people-actually-use).)
- **Tile** — a small rectangular block of a tensor (say 128×64) that a group of
  threads loads and computes together. "Thinking in tiles" is the shared mental
  model of the whole middle of the ladder — it's the shared-memory tiling of
  [§1 rung 2](01_writing_and_tuning_a_matmul.md#6-rung-2--shared-memory-tiling),
  promoted to the *unit you program in*.
- **Layout** — the mapping from a tensor's logical indices (`A[i,j]`) to where
  those elements physically sit in registers, shared memory, or global memory. The
  explicit-layout DSLs exist to let you *control* this mapping; the tile languages
  hide it.
- **MMA** — *Matrix Multiply-Accumulate*: one Tensor-Core instruction that
  multiplies two small matrix fragments and adds the result, instead of one scalar
  multiply-add per lane (the track-switch of
  [§1 §10](01_writing_and_tuning_a_matmul.md#10-track-switch--tensor-cores-not-rung-6)).

The crucial reading of the ladder: **you descend a rung only when the rung above
leaves performance you actually need on the table.** This is the same
*call-vs-build* and *automation* spine from the
[stack doc](../02_cuda_software_stack/01_the_stack.md#what-people-actually-use) —
`torch.compile` (free), then Triton (cheap custom), then CUTLASS/CUDA (expert
only) — now extended downward with the new middle-rung DSLs that 2024–2026 added.

## 2. The three knobs every tool blends

The intimidating part is the *number* of names in the middle. They collapse into
one idea once you ask what each one actually adds over plain Triton. Every new
kernel language since ~2024 turns some mix of exactly **three knobs**:

```text
  1. SPECIALIZATION       give different warps different JOBS — e.g. some warps
                          only fetch data (producers), others only do the math
                          (consumers). "Warp specialization."
  2. EXPLICIT SCHEDULING   let YOU say which instruction issues when, which tile
                          is in shared memory vs registers, when async copies
                          land — instead of the compiler guessing.
  3. COMPILER ORCHESTRATION  the opposite trade: hand even MORE to a compiler
                          (fuse a whole subgraph, auto-pick layouts) so you
                          write even less.
```

Plain Triton turns none of these to the max — it hides specialization and
scheduling so you can be productive, and that's exactly its ceiling on the newest
hardware. Every tool in §5 is "Triton, but with knob 1 and/or 2 cranked," or
"Triton, but with knob 3 cranked." Hold these three labels; the timeline below is
just different tools picking different settings of them.

```text
  black box re-sealed: you do NOT need to memorize the tool list. You need the
  axis (§1) and these three knobs. Given a new kernel-language name, ask only:
  which rung, and which knob does it turn? Everything else is syntax.
```

---

# Part B — Why the middle filled in (2021 → 2026)

The ladder didn't always have a crowded middle. It filled in one GPU generation at
a time, each new architecture exposing a capability the rung above couldn't reach.
This timeline is the *why* behind §1 — read it as evidence that the spectrum is a
response to hardware, not fashion.

## 3. 2021–2023 — the clean binary

Two choices, no middle:

```text
  Triton          the PRODUCTIVITY default for custom ML kernels — write a
                  block-level algorithm in Python, get coalescing + tiling +
                  pipelining for free (the kernel walked in the Triton
                  walkthrough of this chapter).
  CUDA / CUTLASS  the EXPERT-PERFORMANCE default — hand-written C++ for the last
                  few percent (cuBLAS internals, the first FlashAttention).
```

If Triton was fast enough, you used it; if you needed peak, you dropped all the
way to C++. Nothing lived in between, because nothing needed to — pre-Hopper
hardware was simple enough that Triton's "describe tiles, let the compiler
schedule" model captured most of the available performance.

## 4. 2024 (Hopper / H100) — the gap opens

The H100 added hardware features that **don't fit Triton's hide-the-schedule
model**, and the gap between the two rungs suddenly had performance in it. The
proof point: **FlashAttention-3**, the fastest attention kernel of the era,
couldn't be written in first-generation Triton — it dropped to CUTLASS/CuTe and
used the new instructions directly. The features that forced this:

- **WGMMA** — *Warpgroup Matrix Multiply-Accumulate*: an MMA instruction issued by
  a whole *warpgroup* (4 warps = 128 threads) that runs **asynchronously** — the
  threads fire it and move on while the Tensor Cores work in the background. Triton
  couldn't express "issue this, then go do other work."
- **TMA** — *Tensor Memory Accelerator*: a dedicated engine that bulk-copies a tile
  from global to shared memory on its own, without tying up the compute lanes (the
  async-copy idea of
  [§1 rung 5](01_writing_and_tuning_a_matmul.md#9-rung-5--latency-hiding-pipelining-and-async-copy),
  now a hardware unit; it's the Hopper feature flagged in
  [§1 §15](01_writing_and_tuning_a_matmul.md#15-gpu-architecture-dependence)).
- **Warp specialization** — splitting a block's warps into TMA "producers" that
  only fetch and Tensor-Core "consumers" that only compute, coordinated by hardware
  barriers (knob 1 from §2).
- **FP8** — 8-bit floats, double the Tensor-Core throughput of FP16 but with
  scaling the kernel must manage by hand
  ([numerical types](../03_numerical_types/01_floating_point.md)).

These are precisely the controls Triton hides. So 2024 is the year peak attention
left the productivity rung. The wider ecosystem followed: the public GPU-kernel
education world of this period (the **GPU Mode** community, the **FlashInfer**
attention-serving library, JAX's **Mosaic GPU** tile backend) organized around
CUTLASS, FlashAttention-3, and **CuTe** — the layout algebra at CUTLASS's core,
which §5 turns into its own language.

## 5. 2025 — the tile-DSL swarm

With a real performance gap exposed, a wave of new languages rushed to fill it —
each one Triton-grade ergonomics with one or more of the §2 knobs cranked. The
ones I could confirm as real, active projects:

| Tool | Rung / form | What it adds (which knob) |
| --- | --- | --- |
| **CuTe DSL** | explicit-layout, Python | A Python front-end to CUTLASS's **CuTe** layout algebra — explicit control of layouts, MMA atoms (single hardware matrix-multiplies), and the thread/data hierarchy, JIT-compiled via **MLIR** (a compiler-building framework) to PTX. Knobs 1 + 2, without C++. |
| **TileLang** | explicit-layout, Python | Tile programming with fine control over shared memory, async copies, and warp-level primitives. Knob 2. |
| **ThunderKittens** | explicit-layout, C++ embedded | Small library built around 16×16 register tiles; makes FlashAttention-class kernels readable. Knobs 1 + 2. |
| **Helion** | tile, Python (higher than Triton) | Compiles *down to* autotuned Triton; you write closer to PyTorch (`hl.tile`) and it searches tile sizes/layout for you. Knob 3. |
| **Gluon** | lower-level Triton dialect | The Triton project's *own* lower gear — same tile/SPMD model, but layouts and lowering are exposed to you. Knobs 1 + 2, inside Triton. |
| **TLX** | lower-level Triton extension | *Triton Low-level eXtensions* (from Meta): warp-aware intrinsics — warp specialization, register-backed accumulators, fine-grained sync — bolted onto Triton. Knobs 1 + 2, inside Triton. |

> **MLIR** (*Multi-Level Intermediate Representation*) and **SPMD** (*Single
> Program, Multiple Data* — every program instance runs the same code on a
> different tile) are the two acronyms in that table worth keeping; the rest are
> defined in §1.

The single most useful pattern in this list: **Gluon and TLX both descend *within*
Triton.** Two independent groups — the Triton project and Meta — concluded that
when Triton's abstraction runs out of control, the answer isn't to abandon it but
to add a lower gear. That means "Triton vs. CuTe" is no longer a hard either/or:
you can keep your Triton kernel and reach for more control without switching
languages. They all fit one sentence, the timeline's own: *more specialization,
more explicit scheduling, and/or more compiler-managed orchestration* — knobs 1,
2, 3.

## 6. 2026 (Blackwell / B200) — the trend becomes unavoidable

Blackwell pushed the hardware asymmetry further — Tensor-Core throughput roughly
doubles while other units lag — so a kernel that doesn't explicitly schedule
around that imbalance leaves *even more* on the table. The rung-by-rung evidence,
all confirmable today:

- **FlashAttention-4 is written in CuTe DSL.** The flagship attention kernel for
  Hopper *and* Blackwell is a Python CuTe DSL implementation
  (`flash_attn/cute/` in Dao-AILab's repo, `pip install flash-attn-4`). Its own
  paper reports it both **beats Triton on a B200** (in BF16, a 16-bit float) and
  compiles ~20–30× faster than the equivalent C++ CUTLASS — the clearest single signal of
  why this rung exists: near-CUDA speed at near-Triton authoring cost.
- **DeepSeek ships its production kernels in TileLang.** Its open-sourced
  `TileKernels` library — the low-precision (FP8/FP4) kernels it runs
  internally — is written in TileLang (plus some CUDA), targeting Hopper and
  Blackwell.
- **NVIDIA is making a tile model official.** **cuTile** (a NVIDIA Python tile DSL)
  and **TileIR** (a new virtual ISA — an instruction set the compiler targets, one
  level above raw PTX) turn the community's tile idea into a first-class NVIDIA
  compiler substrate, even usable as a Triton backend.
- **ThunderKittens 2.0** (Jan 2026) adds full Blackwell support plus the new
  4-bit/8-bit formats **MXFP8** and **NVFP4** (block-scaled low-precision types
  that wring more throughput out of Blackwell's Tensor Cores).

The pattern across all four: the year's important kernels split *by rung*, not by
tool. Peak attention → CuTe DSL; a production kernel suite → TileLang; everyday
fused ops → Triton/Helion. Nobody picked one language. They each picked the rung
their kernel needed.

---

# Part C — The decision

## 7. The map: which tool, which rung, when

Collapsing Parts A and B into the table I'd actually consult:

| Tool | Rung | Exposes | I'd reach for it when… |
| --- | --- | --- | --- |
| `torch.compile` | top | nothing — auto-generates Triton | I want fusion for free before writing any kernel |
| **Triton** | tile | tiles; compiler schedules | I need a real custom/fused kernel quickly (the [Section 2](02_reading_a_real_kernel.md) walkthrough) |
| **Helion** | tile (higher) | `hl.tile` → autotuned Triton | I want Triton's speed with even less boilerplate |
| **Gluon / TLX** | tile (lower) | layouts, warp specialization, *inside Triton* | a Triton kernel hit its ceiling and I want more control without leaving |
| **CuTe DSL** | explicit-layout | layouts, MMA atoms, TMA, warp roles (Python) | peak Hopper/Blackwell attention or GEMM — the 2026 frontier |
| **TileLang / ThunderKittens** | explicit-layout | same controls, different syntax | following the DeepSeek (TileLang) or readable-FA (TK) lineages |
| **CUTLASS / CUDA C++** | bedrock | everything, by hand | a feature no DSL exposes yet, or the absolute last few percent |

## 8. What I'll actually use, and why

For *this* project — learning LLM inference deeply, not betting on an ecosystem —
the path is deliberate, not "try them all":

```text
  1. TRITON first.   It's still the 2026 productivity default, and the tile/block
                     mental model under it is shared by EVERY rung below. I already
                     have a Triton matmul walked in the Triton walkthrough of this
                     chapter; getting fluent in its
                     autotuning and tile structure is the highest-ROI step.
  2. CuTe DSL for depth.  Confirmed as the frontier (FlashAttention-4 is written
                     in it), and the natural "level 2" when a kernel needs TMA /
                     warp specialization / explicit MMA that Triton hides. This is
                     the M16 (FlashAttention) target.
  3. READ CUDA / CUTLASS.  The bedrock I read to understand what the DSLs compile
                     into — written, only at the frontier or for a missing feature.
```

Two things I'll *recognize but not invest in* yet: **Gluon/TLX** (the lower-gear
escape hatch if I'd rather descend inside Triton than switch to CuTe — arguably the
more natural path for a Triton-first learner), and **TileLang** (the DeepSeek
lineage; worth watching, not my default). I'll skip **Mojo** (a whole separate
language, smaller ecosystem) and **cuTile** (too new) — neither serves a learning
track whose goal is understanding rather than shipping.

The honest 2026 summary: **there is no single winner because the tools occupy
different rungs.** Triton for breadth, CuTe DSL for depth, CUDA/CUTLASS for the
bedrock — and real stacks blend all three. For learning, the rung matters less than
the four ideas every one of them shares: tiles, layouts, async data movement, and
warp roles. Master those in one tool and the rest become syntax.

## 9. What to carry forward

```text
the productivity↔control axis (§1)              -> picking a language for any kernel
the three knobs: specialization / scheduling /
  orchestration (§2)                            -> decode any new kernel-language name
Hopper opened the gap: WGMMA, TMA, warp
  specialization, FP8 (§4)                      -> M16, why FlashAttention drops low
Triton first, CuTe DSL for depth (§8)           -> M1-M6 (Triton), M16 (CuTe DSL)
CuTe DSL = the 2026 frontier (FA-4) (§6)        -> M16, FlashAttention from real source
Gluon/TLX = descend WITHIN Triton (§5)          -> the escape hatch before switching
```

The one sentence to keep: **kernel-writing tools form a single
productivity-versus-control ladder — `torch.compile` and Triton at the top where a
compiler makes the hardware decisions, CUDA/CUTLASS at the bottom where you make
them all, and a crowded 2024–2026 middle (CuTe DSL, TileLang, ThunderKittens,
Gluon, TLX, cuTile) that exists because Hopper and Blackwell exposed
controls — warp specialization, TMA, explicit MMA — that Triton hides; so I learn
Triton first for the shared tile model, reach for CuTe DSL when I need that hidden
control (it's where FlashAttention-4 lives), and read CUDA/CUTLASS to see what the
DSLs compile into.**
