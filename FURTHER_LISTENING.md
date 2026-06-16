# Further listening — what I put on during dead time

Most of this repo is meant to be read at a desk. This file is the opposite: the
audio I queue for workouts, commutes, and chores, to keep the GPU-inference
material turning over when a screen isn't an option. It's the listening companion
to the **Further reading** blocks that close each chapter section — the same
sources where they overlap, re-sorted by the one axis reading lists don't have:
*what survives having no screen.*

It isn't a syllabus. It's the rotation I actually reach for, ordered by how much
it moves my kernel / GPU / inference skills per hour of dead time.

## The one filter that orders everything

Audio-only learning has a hard constraint — can I follow it with my eyes
somewhere else? That single question splits good-on-paper material into three
very different piles:

```text
  CONCEPT talk (architecture, scheduling, "why X is fast")   -> earbuds are fine
  CODE walkthrough (someone narrating a kernel on a screen)  -> eyes required
  DIAGRAM / EQUATION-dense paper read aloud verbatim         -> mostly useless
  the SAME paper as a two-host audio overview (NotebookLM)   -> works, surprisingly
```

So every item below carries a tag:

- **[ears]** — follows fine audio-only; queue it anywhere.
- **[eyes-later]** — worth it, but watch the visual/code parts at a desk first,
  then re-listen to cement. Re-listening dense material is where it actually sticks.
- **[desk-only]** — doesn't survive losing the screen. Listed so I know *not* to
  queue it and waste the slot.

A note on the tools, since two of them carry most of the load below.
**TTS** (text-to-speech) reads a blog or paper aloud in a synthetic voice — fine
for prose, poor at equations and tables. **NotebookLM** (Google's research tool)
does something better for dead time: its *Audio Overview* turns one or more
uploaded sources into a two-host spoken discussion, which survives the no-screen
constraint far better than a literal read-aloud of a paper full of math.

---

## 1. The must-listens

If I only had a handful of hours, in this order:

1. **"How GPU Computing Works" — Stephen Jones (NVIDIA), GTC** — *[eyes-later]*.
   The single best hour for *why a GPU waits on memory instead of computing*. It's
   the spoken version of this repo's whole Chapter 1 thesis. Already the anchor of
   [Chapter 1 §3's reading](chapters/01_hardware_fundamentals/03_gpu_model.md);
   here it's a must-listen. (Slides matter, so watch once, then re-listen.)
2. **GPU MODE** (formerly CUDA MODE) — the reading-group lecture series on CUDA,
   Triton, CUTLASS, FlashAttention, profiling, quantization. *Closest thing to my
   exact goals that exists in audio-adjacent form.* Concept lectures are **[ears]**;
   live kernel walkthroughs are **[eyes-later]**. See §3.
3. **Latent Space podcast** — pure-audio, engineer-first, with recurring
   inference / GPU-infra episodes. The best *commute* fit. See §2.
4. **Stanford CS336 — Language Modeling from Scratch (2025)** — the lecture set
   that covers GPUs/kernels/Triton, MoE, parallelism, inference, and scaling laws
   in one place. **[eyes-later]**. See §3.
5. **Dwarkesh Patel × Sholto Douglas & Trenton Bricken** — the deepest pure-audio
   conversations on how inference, the KV cache, long context, and scaling actually
   behave at the frontier. **[ears]**. See §2.
6. **NotebookLM Audio Overviews of the FlashAttention papers (FA1→FA3)** — turns
   the spine of [Chapter 9](chapters/09_kernel_engineering/02_flashattention_rosetta_stone.md)
   into something I can absorb on a run. See §4.

---

## 2. Podcasts (pure audio — the core of the rotation)

Ranked by how often the episodes land on kernels / GPUs / inference rather than
AI in general. All **[ears]** unless noted.

- **Latent Space: The AI Engineer Podcast** (swyx & Alessio) — the best fit.
  Regularly does inference and GPU-infra episodes (e.g. prefill/decode
  disaggregation and datacenter-scale inference engines). Engineer-first, light
  on hype.
- **Dwarkesh Podcast** — cherry-pick the technical episodes. The conversations
  with **Sholto Douglas & Trenton Bricken** are the standouts for inference,
  context length, and scaling intuition; the **Jeff Dean & Noam Shazeer** episode
  is gold on systems history and hardware/software co-design.
- **SemiAnalysis / Dylan Patel** — datacenter GPUs, HBM (high-bandwidth memory),
  interconnect, and the economics underneath all of it. Not kernel-level, but it's
  the "why does the hardware look like this" layer below
  [Chapter 1](chapters/01_hardware_fundamentals/README.md). Dylan also guests on
  Latent Space and Dwarkesh — those crossovers are a good entry point.
- **The TWIML AI Podcast** (This Week in Machine Learning) — long-running; filter
  by episode title for the systems / serving / efficiency guests.
- **MLOps Community Podcast** — has solid serving/inference-in-production episodes
  (batching, latency, vLLM/SGLang in the wild). Skim the back catalog by title.

---

## 3. Lecture series & talks (YouTube)

The highest-density material, but the most screen-dependent — hence the tags.

- **GPU MODE lectures** (Mark Saroufim & Andreas Köpf) — CUDA, Triton, CUTLASS,
  FlashAttention, NCCL (the multi-GPU collective library), profiling. The
  architecture / profiling / "how Flash works" talks are **[ears]**; the kernel
  walkthroughs are **[eyes-later]**. Their
  [resource-stream](https://github.com/gpu-mode/resource-stream) and
  [lectures repo](https://github.com/gpu-mode/lectures) are already cited from
  [Chapter 7](chapters/07_writing_and_tuning_kernels/02_reading_a_real_kernel.md)
  and [Chapter 9 §4](chapters/09_kernel_engineering/04_reading_and_optimizing_ptx.md).
- **Stanford CS336 — Language Modeling from Scratch (2025)** — *[eyes-later]*.
  Lectures on GPUs/kernels/Triton, mixture-of-experts, parallelism, inference, and
  scaling laws. The conceptual lectures follow by ear; revisit the few slide-heavy
  ones at a desk.
- **Stanford CS149 — Parallel Computing (Kayvon Fatahalian)** — *[eyes-later,
  leaning ears]*. The foundations under CUDA: multicore, the GPU execution model,
  data-parallel thinking, the memory hierarchy. Strong, well-paced lecturing —
  good if the "why" under
  [Chapter 1 §3](chapters/01_hardware_fundamentals/03_gpu_model.md) ever feels thin.
- **Asianometry** — *[ears]*. Semiconductor explainers (HBM, packaging,
  lithography, interconnect), beautifully narrated. Pure commute material for the
  hardware-reality layer.
- **One-off talk: "How GPU Computing Works" (Stephen Jones)** — see §1; the single
  best standalone hour.

---

## 4. Turning the chapters' reading into audio

This is where the **Further reading** I already gathered pays off twice. The
sources below are all cited from the chapters — here they're sorted by whether
they survive being listened to, with the best path for each. Feed the **[ears]**
ones to NotebookLM (Audio Overview) or a TTS app
([article2audio](https://article2audio.app/) for arXiv,
[Listening.com](https://www.listening.com/) for papers with citations/equations,
or Speechify for blogs); leave the **[desk-only]** ones where they are.

### Prose blogs — these read aloud well *[ears]*

- **[Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)**
  (Horace He) — the canonical compute-bound vs memory-bound piece; the spoken
  spine of [Chapter 4 §1](chapters/04_measurement/01_benchmarking.md).
- **[Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)**
  and **[Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)**
  (Lilian Weng) — long-form prose, ideal for TTS; from
  [Chapter 2 §2](chapters/02_cuda_software_stack/02_end_to_end_inference.md) and
  [Chapter 5 §1](chapters/05_attention_and_kv_cache/01_attention.md).
- **[How continuous batching enables 23x throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)**
  (Anyscale) — the throughput-lever story behind
  [Chapter 6 §1](chapters/06_batching/01_batching.md).
- **[Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)**
  (NVIDIA) — a broad spoken survey of the
  [Chapter 8](chapters/08_optimizing_inference/README.md) map.
- **[Making FlashAttention-4 Faster for Inference](https://modal.com/blog/flash-attention-4-faster)**
  (Modal) — the newest rung of the
  [Rosetta Stone](chapters/09_kernel_engineering/02_flashattention_rosetta_stone.md).
- **[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)**
  (kipp.ly) — *[ears, with a caveat]*: it's the back-of-envelope math behind
  [Chapter 6](chapters/06_batching/01_batching.md); a NotebookLM overview handles
  the equations better than literal TTS.

### Papers — best as NotebookLM Audio Overviews *[ears via overview]*

Raw read-aloud chokes on the equations and tables; a two-host overview makes them
commute-able. The high-value set, all already cited in the chapters:

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — the
  foundation ([Chapter 5 §1](chapters/05_attention_and_kv_cache/01_attention.md)).
- **FlashAttention [1](https://arxiv.org/abs/2205.14135) →
  [2](https://arxiv.org/abs/2307.08691) → [3](https://arxiv.org/abs/2407.08608)**
  — the spine of [Chapter 9](chapters/09_kernel_engineering/02_flashattention_rosetta_stone.md);
  listen to the three in sequence to hear the move-list evolve.
- **[PagedAttention / Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)**
  — the KV-cache idea under vLLM ([Chapter 6 §1](chapters/06_batching/01_batching.md)).
- **[Megatron-LM](https://arxiv.org/abs/1909.08053)** — tensor parallelism, behind
  [Chapter 8 §3](chapters/08_optimizing_inference/03_scaling_past_one_gpu.md).
- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**
  — the lossless decode speedup ([Chapter 8 §1](chapters/08_optimizing_inference/01_improving_decode.md)).
- **MoE line: [Switch Transformers](https://arxiv.org/abs/2101.03961) →
  [Mixtral](https://arxiv.org/abs/2401.04088) →
  [DeepSeek-V3](https://arxiv.org/abs/2412.19437)** — the architecture most
  frontier models now use ([Chapter 5 §6](chapters/05_attention_and_kv_cache/06_moe.md)).
- **[Mamba](https://arxiv.org/abs/2312.00752)** — the state-space alternative to
  attention ([Chapter 5 §7](chapters/05_attention_and_kv_cache/07_attention_variants.md)).

### Don't bother listening — these are desk-only *[desk-only]*

Cited and worth reading, but pointless as audio — listing them so I don't try:

- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**
  and **[Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)**
  — the diagrams *are* the explanation.
- **Code repos** — [nanoGPT](https://github.com/karpathy/nanoGPT),
  [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm),
  [CUTLASS](https://github.com/NVIDIA/cutlass),
  [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
  — reading code by ear doesn't work.
- **Interactive / reference** — [Float Toy](https://evanw.github.io/float-toy/),
  [Compiler Explorer](https://godbolt.org/), the
  [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/), and
  [siboehm's CUDA matmul walkthrough](https://siboehm.com/articles/22/CUDA-MMM)
  — all need eyes on the page.

---

## 5. Books — and which ones survive having no screen

I went looking for whether the standard kernel / GPU-programming books are
listenable in dead time. Mostly they are not, and it's worth being honest about
why: the core technical ones are built around code listings, circuit diagrams,
and tables that carry the argument. An audiobook of them would be narration of
"see figure 4.3." So I split them:

**Desk-only — the technical core *[desk-only]*.** No meaningful audio path; these
are bench books.

- **[Programming Massively Parallel Processors](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)**
  (Hwu, Kirk, El Hajj) — the CUDA book this repo leans on
  ([Chapter 1 §3](chapters/01_hardware_fundamentals/03_gpu_model.md)). Code on
  nearly every page.
- **[Computer Architecture: A Quantitative Approach](https://shop.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1)**
  (Hennessy & Patterson) and
  **[Digital Design and Computer Architecture](https://pages.hmc.edu/harris/ddca/)**
  (Harris & Harris) — tables, graphs, and circuit diagrams throughout
  ([Chapter 1 §2](chapters/01_hardware_fundamentals/02_memory_and_caches.md)).

**Listenable — the narrative ones *[ears]*.** These tell a story rather than show
a listing, so the audiobook works on a commute:

- **[Code: The Hidden Language of Computer Hardware and Software](https://www.codehiddenlanguage.com/)**
  (Charles Petzold, 2nd ed.) — builds a computer from relays up in prose. It's the
  gentle, narrative version of what
  [Chapter 1 §1](chapters/01_hardware_fundamentals/01_circuits_and_cores.md) covers
  fast, and it exists as an audiobook. The
  [companion site](https://www.codehiddenlanguage.com/) holds the interactive
  diagrams for when I'm back at a screen.
- **Chip War** (Chris Miller) — *not* in the repo's reading, but the best audio
  context for the hardware-economics layer the
  [SemiAnalysis](https://semianalysis.com/) podcast lives in: a ~12.5-hour
  narrated history of the semiconductor industry. Pure commute listening.

The pattern: a book is listenable in proportion to how much of its argument is
*prose* rather than *figures*. The narrative ones (Petzold, Miller) carry over;
the reference ones (PMPP, Hennessy & Patterson) do not, and pretending otherwise
just wastes the slot.

---

## How I use it

```text
  WORKOUT (low focus)    Latent Space, Dwarkesh, Asianometry, SemiAnalysis,
                         Chip War / Code audiobooks
  COMMUTE (medium focus) CS336 / CS149 lectures, GPU MODE concept talks,
                         NotebookLM overviews of the §4 papers
  DESK first, ears later watch the GPU MODE kernel walkthroughs and the
                         "How GPU Computing Works" talk once, then re-listen
```

The one habit to keep: **the dead-time rotation is for the prose and the
concepts — the talks, the histories, the paper overviews — and the code lives
at the desk; trying to learn a kernel by ear just burns the slot, but
re-listening to one I've already read at a screen is where it finally sticks.**
