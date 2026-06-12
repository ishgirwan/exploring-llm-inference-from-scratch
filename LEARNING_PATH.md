# Learning path

[`ROADMAP.md`](ROADMAP.md) says what gets built and why. The chapters are the
reference library. This file is the third thing: **the order I actually move
through both** — reading and building braided together, so every piece of
theory lands next to the lab that exercises it, and every lab ends by raising
the question the next piece of theory answers.

It exists because of a trap I want to avoid: reading eight chapters of theory
with no GPU contact, retaining a fraction of it, and calling that progress. The
chapters are written to survive being read straight through — but the *skill*
is built the other way: small theory dose, immediate practice, measure, explain,
repeat.

## 1. The loop, and what "done" means

Every stage below runs the same iteration:

```text
  1. QUESTION   each stage opens with one, raised by the previous stage
  2. READ       the minimum that lets me act (~30-45 min, sections not chapters)
  3. BUILD      the lab — change or build one thing
  4. ASK        the in-lab questions: predict before measuring, then interrogate
                the numbers until they stop being surprising
  5. READ       the after-reading — highest-value slot, because the fresh
                measurement is what the theory explains
  6. TAKE AWAY  the internalization check: a stage is NOT done when the code
                works; it's done when I can answer its takeaway questions
                cold, away from the screen
  7. CARRY      the question this stage hands to the next
```

The chain those questions form, Phase 1 end to end:

```text
  M0    can I trust a number from a GPU?
   └─►  M0.5  what do a real LLM's numbers look like on my GPU?
         └─►  M1    the model is just kernels — can I write the simplest one?
               └─►  M2    everything was bandwidth — how close to the spec
                          sheet can a kernel actually get?
                     └─►  M3    streaming needs no cooperation — what happens
                                when threads must share results?
                           └─►  M4/M5  can I compose those patterns into real
                                       LLM ops?
                                 └─►  M6    every op so far was memory-bound —
                                            meet the one with arithmetic to
                                            burn; can I make it fast?
                                       └─►  M7    my kernels are loose files —
                                                  how do they become ops a
                                                  model can call?  → v0.1
```

## 2. The build loop (no local GPU)

I write code on a Windows machine with no NVIDIA GPU and run it on a free
Colab T4:

```text
  local:   write code, run the GPU-free tests, commit, push
  Colab:   !git clone (or pull) the repo
           !pip install -r setup/requirements-colab.txt   (pinned versions)
           !python labs/<topic>/run.py --save
           commit the produced results JSON / profiles back
```

- **Pinning on Colab is pip, not Docker.** Colab can't run the pinned container
  the roadmap describes; there, reproducibility = pinned requirements plus the
  environment snapshot every results file records. The container becomes real
  at the rented-GPU stage (M8+).
- **Tests split by hardware.** GPU-needing tests carry a pytest marker and skip
  automatically without CUDA; CI and the local machine run the GPU-free set.
- **Profiling on Colab is Nsight Systems only.** Nsight Compute needs
  permissions Colab locks down, so per-kernel counter work waits for M8 on a
  rented GPU; until then the tools are nsys timelines, bandwidth math, and
  occupancy reasoning.

## 3. The on-ramp (before anything runs)

Three sections, roughly two hours, chosen as the minimum to act — not the
minimum to understand everything:

```text
  Ch1 §3  the GPU execution model     threads/blocks/warps, latency hiding —
                                      the vocabulary every lab uses
  Ch2 §1  the CUDA software stack     driver vs toolkit vs PyTorch — what
                                      "it doesn't run" errors actually mean
  Ch4 §1  benchmarking methodology    the rules the harness encodes — read
                                      before building it
```

Everything else waits until a stage calls it. Chapters 5, 6 and 8 aren't
skipped — they're the before-reading of M9–M14 and M15–M20, months from now,
when they'll land on real context instead of evaporating. Chapter 9 stays
sealed until Phase 5, as the roadmap already says.

## 4. Stage 0 — scaffolding *(no GPU, one or two sittings)*

No opening question; this is the workshop being set up. One real learning
exercise hides in it: **writing `BENCHMARKING.md` — distilling Chapter 4 §1
into the concrete rules `bench.py` will enforce — is the comprehension test
for the on-ramp reading.** If a rule can't be stated crisply enough to encode,
the section gets re-read.

```text
  #    step                                  done when
  0.1  root README.md                        exists, points into ROADMAP        ✓
  0.2  LICENSE                               picked (MIT or Apache-2.0), committed
  0.3  FAILURES.md + CHANGELOG.md            stubs exist, format stated at top
  0.4  BENCHMARKING.md                       Ch4 §1 distilled into enforceable
                                             rules (warm-up count, timing method,
                                             percentiles, what gets recorded)
  0.5  Python skeleton                       pyproject.toml + ruff; common/ as a
                                             package; tests/ with one passing test
  0.6  CI                                    GitHub Actions: lint + GPU-free
                                             tests, green on push
```

## 5. Stage 1 — M0: the harness

**Arriving question: can I trust a number that comes out of a GPU?**

**Read before:** Ch4 §1 (done in the on-ramp; skim again with builder's eyes).

**Build:** `setup/requirements-colab.txt` (pinned) → `setup/check_gpu.py`
(name, compute capability, VRAM, driver/runtime, spec-sheet bandwidth — run on
the T4, output committed as the first GPU artifact) → `common/results_schema.py`
→ `common/bench.py` (CUDA-event timing, warm-up discard, median + p5/p95) →
`common/correctness.py` (dtype-aware tolerances).

**Ask while in it:**
- Before writing the warm-up: time an op *cold*, first call in the process.
  How many orders of magnitude separate it from the steady state?
- Time the same op with `time.time()` and with CUDA events. Where does the
  wall-clock lie come from?
- Run the full benchmark twice end to end. Do the medians agree within their
  spread? If not, what drifted — clocks, thermals, a neighbor on the GPU?

**Read after:** Ch4 §2 (first-run effects) — every weird first-run number just
collected gets a name: caching allocator, JIT, autotune, clocks, context init.

**Take away (answer cold):** why the first run is orders slower; why the CPU
clock can't time an async launch without a sync; why median + spread, not mean.

**Carry forward:** the harness can time anything → *so what does a real LLM
actually do on this GPU?*

**Done when:** the harness, run on the T4, commits a benchmark JSON for plain
`torch.add`.

## 6. Stage 2 — M0.5: run a real model

**Arriving question: what do a real LLM's numbers look like?**

**Read before:** Ch2 §2 (end-to-end inference) — a *skim*: just the stage names
(load → tokenize → prefill → decode), enough to know what to time.

**Build:** load a ~0.5B instruct model fp16 on the T4, generate, measure with
the harness's rules: time-to-first-token, steady tokens/sec, VRAM before and
during. Watch `nvidia-smi` while it runs.

**Ask while in it:**
- Predict before running: prefill on a 500-token prompt vs generating 50
  tokens — which takes longer?
- Is tokens/sec flat as the answer grows, or drifting? Why would it drift?
- The arithmetic rep, the most important one here: model-bytes × tokens/sec ≈
  how many GB/s? Compare to the T4's ~320 GB/s spec. How close is decode to
  just *streaming the weights at full speed*?
- VRAM: weights account for how much? What owns the rest?

**Read after:** Ch2 §2 properly this time — the pipeline just measured, named
layer by layer — and Ch1 §3's memory-bound section, which the GB/s arithmetic
just demonstrated from my own numbers.

**Take away (answer cold):** decode speed ≈ bandwidth / model bytes, and my
own measurement showing it; TTFT and TPOT as two different regimes I've timed,
not two acronyms I've read.

**Carry forward:** that tokens/sec number is produced by kernels → *can I
write even the simplest one?*

**Done when:** `reports/00.5_real_model.md` has measured TTFT, tokens/sec, the
GB/s arithmetic, VRAM numbers, and a failure log.

## 7. Stage 3 — M1: vector add, three ways

**Arriving question: how does my code run on thousands of threads?**

**Read before:** Ch1 §3's core sections (threads, blocks, warps, kernel
launch) — done in the on-ramp; now reread the launch mechanics with intent to
implement.

**Build:** the same `c = a + b` in PyTorch, Triton, CUDA C++ (compiled on
Colab via `torch.utils.cpp_extension`); verify both against PyTorch; benchmark
all three across sizes; plot GB/s vs N.

**Ask while in it:**
- Predict before measuring: CUDA vs Triton vs PyTorch — who wins, by how much?
  (Expect a three-way tie at large N. If one lags badly, it's a bug or an
  unpaid JIT, not a language property — find which.)
- Sweep N from 1K to 100M: where's the elbow in the GB/s curve, and what does
  it say about what dominates at small N?
- Block size 32 vs 256 vs 1024 — measure. Why so flat *here*, when Chapter 7
  §1 makes block size sound critical?
- Remove the bounds guard, run a non-multiple size. What actually happens?

**Read after:** Ch1 §3's latency-hiding and memory-bound sections — the tie
between three "different" implementations is the evidence: vector add is a
memory-pipe problem, and the language can't change the pipe.

**Take away (answer cold):** kernel time = bytes moved / bandwidth for
streaming ops; a launch costs microseconds regardless of size (the elbow);
grid/block indexing without looking it up.

**Carry forward:** everything is bandwidth → *how close to the spec sheet can
a kernel actually get?*

**Done when:** three verified implementations, one GB/s-vs-size plot, the
first full `lessons/` + `labs/` + `reports/` triple.

## 8. Stage 4 — M2 + M2.5: bandwidth, and the first profile

**Arriving question: what fraction of 320 GB/s is reachable, and what eats
the rest?**

**Read before:** Ch1 §2 (memory hierarchy) — where the bytes physically live
and why DRAM rewards contiguous access.

**Build:** copy / scale / axpy kernels + the bandwidth benchmark; first Nsight
Systems trace over the M1/M2 kernels; a hand-drawn roofline note placing every
kernel built so far on it.

**Ask while in it:**
- Copy GB/s vs the spec sheet: expect a healthy fraction, not 100%. What's the
  gap made of?
- Stride the loads by 2, 4, 32. *Predict the bandwidth curve before running* —
  then look at what coalescing actually costs when broken on purpose.
- axpy moves 3 bytes-streams per element to copy's 2 — does measured time
  scale with bytes the way the model predicts?
- In the nsys timeline: find the kernel. What fraction of wall time is kernel
  vs gaps, and what fills the gaps?

**Read after:** Ch7 §1's *coalescing rung only* (the ladder's first move, just
demonstrated by the stride experiment) — the rest of that doc stays sealed
until M6.

**Take away (answer cold):** the achieved-vs-theoretical reflex (never report
GB/s without the % of spec next to it); coalescing as the first thing to check
on any slow kernel; how to place any kernel on a roofline from its bytes and
FLOPs.

**Carry forward:** streaming ops need no thread cooperation → *what happens
when threads must combine results?*

**Done when:** a measured-vs-theoretical bandwidth table, the stride
experiment's curve, one annotated nsys artifact in `benchmarks/nsight/`, and
the roofline note.

## 9. Stage 5 — M3 + M3.5: reductions, softmax, sampling

**Arriving question: how do ten thousand threads add up one number?**

**Read before:** Ch1 §3's shared-memory and synchronization sections; for
M3.5, Ch5 §5 (sampling) — the first Chapter 5 section the path unseals.

**Build:** sum/max reduction (naive atomics, then shared-memory tree, then
warp shuffles), numerically stable softmax, then the sampling kernels (top-k,
top-p, temperature).

**Ask while in it:**
- Atomic-everything vs tree reduction: measure both. At what input size does
  atomic contention start to bite?
- Reduction GB/s vs copy GB/s from M2: a reduction reads everything once — so
  is it still just a streaming problem? How close?
- Sabotage on purpose: remove the max-subtraction from softmax, feed logits
  around 100, watch inf/NaN appear. *Then* read why.
- Warp-shuffle vs shared-memory tree: measurable delta on the T4, or noise?
- Top-k over a 32K vocab: how does its cost compare to the softmax it follows?

**Read after:** Ch3 (floating point — exponent range and why subtracting the
max is exact, read with the NaN still on screen); the rest of Ch5 §5.

**Take away (answer cold):** the shared-memory + `__syncthreads` pattern and
the warp-shuffle shortcut; stability = subtract the max, and *why* it's free;
reductions are still bandwidth problems — cooperation changes the code, not
the bottleneck.

**Carry forward:** these patterns compose → *can I build the actual ops a
transformer layer runs?*

**Done when:** reductions + softmax verified and benched, the NaN experiment
in the failure log, sampling kernels matching `torch.multinomial`-based
references.

## 10. Stage 6 — M4 + M5: RMSNorm and RoPE

**Arriving question: can I build real LLM ops at competitive speed?**

**Read before:** Ch5 §4 (the elementwise glue — RMSNorm, RoPE, residuals: what
these ops *are* in the model and why they're fusion targets).

**Build:** RMSNorm in PyTorch / CUDA / Triton, benchmarked against the
built-in; a RoPE kernel.

**Ask while in it:**
- The gap to `torch`'s RMSNorm: what is it made of? (Launch count? Vector
  width? Fusion?)
- Vectorized `float4` loads: predict, then measure the delta.
- Push registers per thread up (bigger unroll): where does occupancy drop, and
  does speed follow? First contact with the tuning tension Ch7 §1 describes.
- RoPE's layout choice (interleaved pairs vs split halves): which coalesces
  better, and does the measurement agree?

**Read after:** Ch7 §1's occupancy/tuning-knob section (just felt as a real
trade-off, not a definition).

**Take away (answer cold):** memory-access shape decides elementwise-op speed;
the register/occupancy dial and what moving it did; why norm + scale + residual
beg to be one kernel (the fusion instinct, pre-M29).

**Carry forward:** every kernel so far was memory-bound → *meet the op with
arithmetic to burn. Can I make matmul fast?*

**Done when:** RMSNorm within a defensible gap of the built-in (gap explained
in the report), RoPE verified against a reference implementation.

## 11. Stage 7 — M6: the matmul ladder *(the long stage)*

**Arriving question: cuBLAS exists — how far below it does my best effort
land, and what closes the gap?**

This stage is where the braid gets tightest: **Ch7 §1 is read one rung at a
time, between measurements** — never ahead of the rung being climbed. The
discipline is a running table:

```text
  rung   change                     read first (Ch7 §1)      GFLOPs   % cuBLAS
  0      naive triple loop          the kernel skeleton        ?         ?
  1      fix the index order        coalescing rung            ?         ?
  2      shared-memory tile         tiling rung                ?         ?
  3      register tile              register rung              ?         ?
  4      vectorize loads            vectorization rung         ?         ?
  5      tl.dot (Tensor Cores)      Tensor-Core rung           ?         ?
```

**Ask while in it:**
- Before rung 0 runs: write down a predicted % of cuBLAS. (Most people guess
  high by an order of magnitude — the gap *is* the lesson.)
- Before every rung: predict the multiplier. After: explain the miss.
- At rung 2: compute arithmetic intensity by hand. Did the kernel cross the
  roofline ridge from memory-bound to compute-bound? Where does the
  measurement say it is?
- Shrink the batch to 1 (the GEMV shape): watch compute-bound flip back to
  memory-bound — then connect it to M0.5's decode arithmetic. Same fact, now
  seen from inside the kernel.
- After the last rung: read Ch7 §2 (the annotated real Triton matmul) and ask
  the literacy question — *can I now name every move in a kernel someone else
  optimized?*

**Read after:** Ch7 §2 (as above), and Ch7 §3 (the language landscape) as the
closing context for why Triton carried this far and what sits below it.

**Take away (answer cold):** the ladder as a move-list with *personally
measured* multipliers; arithmetic intensity and the ridge point as instinct;
why decode is a GEMV problem and prefill a GEMM problem — derived, this time,
not read.

**Carry forward:** these kernels live in loose files → *how do they become
real operators a model can call?*

**Done when:** the rung table is complete with real numbers, each rung's
report explains its multiplier, and the Triton version sits within a stated
factor of cuBLAS (whatever that factor honestly is on a T4).

## 12. Stage 8 — M7: kernels become PyTorch ops

**Arriving question: what separates a .cu file from an operator a model can
use?**

**Read before:** Ch2 §1 again (driver/toolkit/extension mechanics — the stack
doc becomes a how-to on second read).

**Build:** register the best M1–M6 kernels as PyTorch custom operators;
exercise them under `torch.compile`.

**Ask while in it:** does `torch.compile` leave my op alone, fuse around it,
or refuse it? What does the extension build actually produce, and where does
the driver JIT fit?

**Take away (answer cold):** the operator-registration path; where a custom
kernel sits in the compile stack; what AOT vs JIT compilation means for *my*
artifacts now, not abstractly.

**Carry forward:** Phase 1 closes — tag `v0.1`. The next arc (M8–M14) starts
with a rented GPU and the question the T4 kept deferring: *what does a real
profiler see inside these kernels?* Its before-reading: Chapter 5 (the forward
pass anatomy, sections 1–4) and Chapter 6 — unsealed only now, when the
transformer assembly that needs them begins.

## 13. Beyond v0.1

The braid continues with the same format — Ch5/Ch6 before M9–M14, Ch8 §1–§2
before M15–M19, Ch8 §3 before M20, Ch4 §3 (quality) alongside M15, Chapter 9
at Phase 5 — but the fine-grained breakdown stays **one phase deep on
purpose**: M8+ gets its stage blocks written when v0.1 lands, informed by
whatever Phase 1 actually taught. Breaking down M16 today would be planning
fiction.

## 14. Standing rules

```text
  every number       comes from a committed script, never typed from memory
  predictions        written down BEFORE measuring — a guess that can't be
                     wrong teaches nothing
  results            JSON to benchmarks/results/, plots to benchmarks/plots/,
                     traces to benchmarks/nsight/
  wrong turns        into FAILURES.md as they happen, not retrospectively
  versions           recorded inside every results file, pinned in setup/
  prose              new docs only when a measurement raises a question
                     (the chapters are frozen except for factual fixes)
  done               = the stage's takeaway questions answered cold,
                     not the code merely running
```
