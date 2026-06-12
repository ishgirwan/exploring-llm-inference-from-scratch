# How I practice the kernel track: one variable at a time, cheapest dimension first

The [Phase 5 modules](../../ROADMAP.md) (M23–M30) list *what* to build. This doc is the
*how*: the method I use to practice them so the hours compound into skill instead of
scattering, and the order I run them in so the cheap, high-value reps come first. It's a
lens over the existing M-modules, not a change to them — read it alongside the roadmap.

The question it settles is the one I kept circling: do I take *one kernel and run it on
many GPUs*, or *many kernels on one GPU*? It turns out a single rule makes that choice for
me, and the same rule orders everything else.

**Honesty up front.** No local GPU here (remote-only), so none of this is measured — it's
the method I'll apply once the M-modules run on rented GPUs. Named optimizations and
numbers are from public sources, labelled as such.

Prerequisites: [Chapter 4 — Measurement](../04_measurement/README.md) (the
benchmark-and-attribute discipline this whole method rests on) and
[CuTe DSL foundations](01_cute_dsl_foundations.md) (the tools the practice uses). A bare
**§N** means *this doc's* section N; references to other docs are links.
Next: the [Phase 5 modules (M23–M30)](../../ROADMAP.md), which this method sequences.

---

## 1. The one rule: change exactly one dimension per experiment

Every kernel experiment has a stack of variables I *could* change at once — the kernel,
the GPU, the vendor, the precision, the baseline I measure against. The rule that makes
practice productive is to **hold all of them fixed but one.**

```text
  vary ONE thing       →  a 1.4× change has ONE cause. I can attribute it. I LEARN.
  vary TWO things       →  a 1.4× change has two possible causes. I can attribute
   (new kernel AND          NOTHING. the run gave me a number, not a reason — and a
    new GPU at once)        number I can't explain is wasted GPU time.
```

This dissolves the "one kernel on many GPUs *or* many kernels on one GPU?" question: both
are fine — they're just two *different single-variable sweeps*. "Same kernel, swap the
GPU" isolates the architecture; "same GPU, swap the kernel" isolates the algorithm. What I
never do is swap both at once, because then a slowdown could be either and I've paid to
learn nothing. This is the
[measurement chapter's](../04_measurement/README.md) attribute-the-cause discipline,
applied to *which experiments I set up* — not just to how I time them.

---

## 2. The dimensions, ranked by value-per-dollar

The variables aren't equal. Some teach the transferable craft cheaply; others have the
highest ceiling but make terrible *entry* points because they secretly change two things
at once. Ranked by what to sweep first:

| Sweep this dimension | What it teaches | Cost | When |
| --- | --- | --- | --- |
| **Optimization level** — one kernel, one GPU, rung by rung | The craft itself: profiling, the tuning ladder, reading SASS | Cheap — short rented-H100 sessions | **First; this is the whole game** |
| **Baseline / kernel provider** — mine vs cuBLAS / FlashInfer / Triton | What "good" even *is* — the bar I'm chasing | ~Free — it's the measuring stick | Always — not a phase; it runs under everything |
| **Kernel type** — GEMM → attention → a fused op | Breadth; the patterns that recur | Cheap — same GPU | Second |
| **GPU architecture** — A100 → H100 → B200 | What's architecture-specific vs universal | Medium — bigger rentals, short runs | Third |
| **Vendor** — NVIDIA → AMD | Portability, and the contribution frontier | High floor — MI300X + a new toolchain | Later |
| **Model architecture** — a whole model, kernel by kernel | Where time *actually* goes (rarely where I'd guess) | Medium | Capstone |

The pattern to notice: **the highest-value dimension is also the cheapest** — driving one
kernel up the ladder on one GPU. Vendor and model-architecture have the highest *ceiling*
but the worst *entry*, because each changes two things at once (a new toolchain *and* new
hardware, or dozens of kernels *and* their interactions). They violate §1 by construction,
so they wait until the single-kernel loop is automatic.

---

## 3. The cheapest rep: read how a kernel already got faster

The single cheapest practice — **zero GPU cost** — is studying how an existing kernel was
improved over time. A fast kernel today is the end of a chain of named optimizations, each
one added to fix a specific bottleneck. Reading that chain hands me the moves before I
rent anything.

The output of this rep is a **move-list**: each optimization paired with the bottleneck it
attacked. FlashAttention is the worked example — and reading it across its generations is
exactly [M25, the Rosetta Stone](02_flashattention_rosetta_stone.md):

```text
  move (FlashAttention generation)     bottleneck it attacked
  ──────────────────────────────────   ────────────────────────────────────────────
  tiling + online softmax (FA1)        HBM traffic — never materialize the N×N scores
  better work partitioning (FA2)       wasted non-matmul work; low warp/seq parallelism
  warp specialization + TMA (FA3)      feeding the Tensor Cores without stalling (Hopper)
  polynomial-emulated softmax (FA4)    the exp() / rescale bottleneck on Blackwell
```

Reseal it: I don't need to *write* FA4 to profit from this. The point is to walk into a
paid GPU session already knowing the named moves and *which bottleneck each one is for*, so
I'm applying a known optimization rather than guessing. Reading the history is the warm-up
that makes the paid reps land — and it's free.

---

## 4. Two ways to practice, and where each belongs

Two practice modes — both from how I framed the goal, *understand what exists, then make it
better* — placed by when they pay off:

```text
  STUDY-THEN-EXTEND    take a kernel with a visible history + a strong baseline; replay
   (the daily engine)  the move-list (§3), rebuild a rung myself, then push one step past.
                       cheap, repeatable — this is the loop I run constantly.

  MODEL KERNEL-BY-     take a whole (small) model, profile it end to end, then improve
   KERNEL (capstone)   the ONE kernel the profile says dominates.
                       its real lesson: the bottleneck is rarely where I'd guess — often
                       attention or the elementwise glue, not the big GEMM.
```

Order matters. Study-then-extend is the engine; model-kernel-by-kernel is a *capstone* —
it's M30-shaped and leans on the whole-model assembly from M9–M11. Run early it drowns me
in moving parts; run after I own the single-kernel loop, it's how I find out which kernel
is even *worth* optimizing before I spend a session on it.

---

## 5. The sequence: cheapest-and-highest-value first

Putting §1–§4 together as an order of operations — and mapping each step onto the
M-modules it's a lens over (without changing them):

```text
  STAGE                 WHAT I DO                                     LENS OVER
  ────────────────────  ────────────────────────────────────────────  ──────────────
  0  FREE, no GPU       read one kernel's full evolution; write the   M25
                        move-list (§3)
  1  CHEAP (H100)       ONE kernel — GEMM — naive → matched-vs-cuBLAS, M23, M24
                        rung by rung. don't move on until I beat a
                        real baseline once
  2  CHEAP (H100)       sweep KERNEL TYPE: attention next, same GPU,  M26
                        same loop
  3  MEDIUM             sweep GPU ARCH: the kernel I know, A100 vs     M27, M28
                        H100 (vs a B200 slice). attribute every delta
  4  MEDIUM             CAPSTONE: a small model, profiled end to end;  M29, M30
                        improve the one kernel that dominates
  5  HIGH FLOOR         sweep VENDOR: port my best kernel to MI300X,   the AMD arm
                        find the gap, beat AITER/CK                       (Ch9 §5)
```

**Why GEMM is stage 1.** It's the "hello world" of Tensor Cores, `cuBLAS` is an
unforgiving baseline so I always know exactly how far off I am, and every later kernel
(attention, fused MLP) reuses its tiling and pipelining lessons. It's the *anchor* the
other sweeps move off of — change the GPU under it (stage 3) or the vendor (stage 5), but
keep the kernel I already understand as the constant.

**Why the cost shape is cheap-first.** Stages 0–2 run on the cheapest capable rental for
minutes at a time; only stages 3–5 need the pricier or harder-to-get silicon (a B200
slice, an MI300X), and only for short benchmark runs. The
[AMD doc's cost ladder](05_amd_kernel_track.md#6-running-it-with-no-hardware-the-cost-floor-and-ladder)
is the same rule pushed to its limit: rent the expensive architecture only for the profile
run, kill it the moment the run ends.

---

## 6. Further reading

The method here is "read how it got faster, then rebuild a rung." These are the sources
I lean on for that — each is a kernel with its improvement *visible*:

- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** (Dao-AILab) — the
  repo whose FA1→FA4 history is the move-list of §3; the canonical kernel to read across
  generations.
- **[FlashAttention-2 in CuTe, from scratch](https://blog.echen.io/p/flashattention-2-in-cute-from-scratch/)**
  — a full worked rebuild of one generation; the study-then-extend mode (§4) done in public.
- **[Accelerating MoEs with a Triton cache-aware grouped-GEMM kernel](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)**
  (PyTorch) — a baseline→optimized write-up that names each move and the bottleneck it hit:
  the §3 method applied to a real kernel.
- **[Colfax Research — CUTLASS tutorials](https://research.colfax-intl.com/category/papers/tutorials/)**
  — step-by-step kernel building, the rung-by-rung craft of stage 1.

## 7. What to carry forward

```text
change ONE dimension per experiment or you can't        -> how I set up every
  attribute the result (§1)                                  M23–M30 experiment
dimensions ranked by value/$: optimization-level &       -> the ORDER I run M23–M30
  baseline first, vendor/model-arch last (§2)
read a kernel's history → a move-list (the move +        -> M25 (Rosetta Stone)
  the bottleneck it hit) before paying for GPU (§3)
study-then-extend = the daily engine; model-kernel-      -> M30; capstone modes
  by-kernel = the capstone (§4)
the sequence: read → GEMM-vs-cuBLAS → attention →        -> a lens over M23–M30
  arch sweep → model capstone → AMD (§5)                      + the AMD arm (Ch9 §5)
GEMM is stage 1: hello-world, unforgiving baseline,      -> M23, M24
  reused by every later kernel (§5)
```

The one sentence to keep: **Practice only compounds when each experiment changes exactly
one variable, so I sweep the dimensions one at a time in value-per-dollar order — read a
kernel's improvement history for free to build a move-list, then drive one kernel (GEMM)
up the ladder against cuBLAS on one cheap GPU, and only then vary kernel type, then GPU
architecture, then whole models, and finally vendor (AMD) — because the cheapest dimension
to sweep, optimization level on a single kernel, is also the one that teaches the actual
craft.**
