# End to end: a prompt becomes tokens

This is the doc I needed before diving into the individual pieces: the whole
path, from a model sitting in a file on disk to text streaming back to a user,
with every software layer and every hardware component named as it gets used.
The earlier docs each zoom into one part — a transistor, an SM, the CUDA
library layer. This one steps back and draws the map they all live on, then
walks a single prompt down through it and back.

It's deliberately a *map*, not a deep dive. Where a piece already has its own
treatment, I link to it rather than re-explain it. The genuinely new ideas here
are three: the **time axis** (a model's life runs load → prefill → decode), how
a **kernel gets chosen** for each operation, and the **serving-engine** layer
that drives the whole loop.

Prerequisites: all of [Chapter 1 — Hardware fundamentals](../01_hardware_fundamentals/README.md)
(especially [The GPU execution model](../01_hardware_fundamentals/03_gpu_model.md))
and [The CUDA software stack](01_the_stack.md) above.
Next chapter: [Chapter 3 — Numerical types](../03_numerical_types/README.md).

## 1. Two axes: the stack and the timeline

Everything in this doc lives at the crossing of two directions, and keeping
them separate is the whole trick to not getting lost.

The **vertical axis** is the software-to-hardware stack: a call starts as
Python at the top and ends as electrons moving in a Tensor Core at the bottom.
Each layer hands work to the one below it. This is the stack from
[§1 of this chapter](01_the_stack.md#the-layers-top-to-bottom), now drawn with
the *inference-specific* names on each rung:

```text
  VERTICAL AXIS — who hands work to whom (top calls down)

  serving engine        vLLM / SGLang: receives requests, batches them,
                        owns the generation loop and the KV cache         (CPU)
     |
  ML framework          PyTorch: defines the model as a graph of ops,
                        dispatches each op to a kernel                     (CPU)
     |
  kernel libraries      cuBLAS (matmul), FlashAttention / FlashInfer
                        (attention), fused custom kernels                 (CPU-side launch)
     |
  CUDA runtime          cudaMalloc, cudaMemcpy, kernel<<<>>> launch        (CPU)
     |
  CUDA driver           turns launches into hardware commands;
                        compiles PTX -> SASS for this exact GPU           (CPU -> GPU)
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  GPU hardware          SMs, warp schedulers, Tensor Cores, FP/INT lanes,
                        shared memory, registers, L2, VRAM                (GPU)
```

The dashed line is the host/device boundary from
[§1](01_the_stack.md#what-a-cuda-program-actually-does): everything above it is
software running on the CPU; everything below it is the GPU chip. The driver is
the rung that straddles the line — it runs on the CPU but its whole job is to
drive the hardware.

The **horizontal axis** is time. A loaded model doesn't do one kind of work; it
does three phases in sequence, and they stress completely different parts of the
hardware:

```text
  HORIZONTAL AXIS — the life of one request, left to right

  [ STAGE 0 ]        [ STAGE 1 ]      [ STAGE 2 ]          [ STAGE 3 ]
  load model    -->  tokenize    -->  prefill        -->   decode loop
  (once, at          the prompt       (process whole        (generate tokens
   startup)          (CPU)            prompt at once)        one at a time)
                                      COMPUTE-bound         MEMORY-bound
                                      Tensor Cores hot      at batch size 1
                                                            (see §6, §9)
                                                                |
                                                                +--loop--+
                                                                         |
                                                            one token out per turn,
                                                            fed back in as next input
```

The rest of the doc walks these four stages left to right. At each stage I pull
in the vertical stack: which software layer is driving, and which hardware
component is actually doing the work. Two things to hold onto from the start:
**Stage 0 happens once** (it's setup, not per-request), and **Stage 3 is a
loop** — almost all the wall-clock time a user waits is spent there.

## 2. Stage 0 — loading the model (once, at startup)

Before a single prompt can be served, the model's *weights* — the billions of
trained numbers that define it — have to get from a file into a place the GPU
can read at full speed. That place is VRAM
([GPU execution model §13](../01_hardware_fundamentals/03_gpu_model.md#13-hbm-and-gddr-are-large-off-chip-memory)).

```text
  disk (.safetensors files)
     |   1. serving engine + framework read the file (CPU)
     v
  CPU RAM
     |   2. cudaMemcpy host -> device, over the PCIe bus
     |      (PCIe = the cable between CPU board and GPU card; see ch2 §1)
     v
  VRAM (HBM or GDDR)   <- weights now live here. e.g. ~140 GB for a
                          70-billion-parameter model in FP16 (2 bytes each)
```

Who does what, top of the stack to bottom:

```text
serving engine   decides which model to load, how to shard it, how much
                 VRAM to reserve for the KV cache (see §9)
framework        reads the weight file, creates tensors, calls .to('cuda')
CUDA runtime     cudaMalloc reserves the VRAM; cudaMemcpy copies into it
driver           moves the bytes across PCIe and into the right VRAM pages
hardware         the bytes come to rest in HBM/GDDR
```

Read top to bottom, that table is a stack of *delegations*: each layer doesn't do
the work itself, it hands the request to the layer below. The single Python line
`tensor.to('cuda')` sets off the whole chain —

```text
  PyTorch        tensor.to('cuda')
                      |  "put this tensor on the GPU"
                      v
  CUDA runtime   cudaMalloc + cudaMemcpy(host -> device)
                      |  "reserve VRAM, then copy these bytes into it"
                      v
  CUDA driver    program the DMA engine; map virtual -> physical pages
                      |  "stream the bytes over PCIe into these VRAM pages"
                      v
  GPU hardware   DMA engine writes the bytes into HBM / GDDR
                      |
                      v
                 weights now resident in VRAM
```

Two terms from that chain. *DMA* (Direct Memory Access) is a hardware copier that
moves bytes across PCIe without the CPU performing the copy — the driver just
programs it and lets it run. A *VRAM page* is one fixed-size chunk of GPU memory
(kilobytes-scale, larger than a CPU's typical 4 KB page): VRAM is not tracked byte
by byte, so the pointer `cudaMalloc` returns is a *virtual* address, and the driver
keeps a *page table* mapping each virtual page to a physical page in the HBM/GDDR
chips. "Into the right VRAM pages" means the driver lands the incoming bytes at the
physical pages backing that tensor's virtual address. The same paging idea returns
one level up in *PagedAttention* ([§9](#9-the-serving-engine-who-drives-the-loop)),
which stores the KV cache in fixed-size blocks — pages — so it need not sit in one
contiguous slab.

This is a one-time cost paid at server startup, and it is pure data movement —
no math, no Tensor Cores. The reason it matters for the rest of the story: once
this is done, **the weights never leave VRAM**. Every token the model ever
generates re-reads these same weights out of VRAM. That single fact is what
makes Stage 3 memory-bound, and §6 returns to it.

A *safetensors* file, named above, is just the common on-disk format for model
weights — a flat block of numbers plus a small header saying the name, shape,
and dtype of each weight tensor. Nothing runs from it directly; it's a
container that gets copied into VRAM.

## 3. Stage 1 — the prompt arrives and becomes tokens

A user sends text. That text hits the **serving engine**, a normal program
running on the CPU (the *host*, [ch2 §1](01_the_stack.md#what-a-cuda-program-actually-does)).
Before the GPU can touch it, the text has to become numbers.

```text
  "Explain memory bandwidth"
        |
        |  tokenizer (runs on the CPU)
        v
  [15496, 4088, 33188, ...]      token IDs: integer indices into the
                                 model's vocabulary
```

*Tokenization* is splitting the string into *tokens* — short, frequently
occurring chunks of text (often whole words or word-pieces) — and replacing
each with its integer ID. The set of all possible tokens is the *vocabulary*
(commonly 32,000–256,000 entries). The whole step is plain CPU string-handling;
the GPU is not involved yet. The output is a short list of integers, the only
form of the prompt the model actually consumes.

## 4. Stage 2 — prefill: a transformer layer is a graph of kernels

Now the GPU starts working. *Prefill* is the phase that processes the entire
prompt in one shot. The N token IDs are sent to the GPU and turned into vectors,
then pushed through every layer of the model.

First, the *embedding lookup*: each token ID is used to pick out one row of the
*embedding table* (a large weights matrix in VRAM, one row per vocabulary
entry). N tokens produce N vectors — an `[N × d_model]` matrix of *activations*
(the running intermediate values flowing through the model; `d_model` is the
model's hidden width, e.g. 4096). The lookup is a gather, not a matmul — and the
embedding table mirrors the *LM head* at the far end of the model (both are
[Chapter 5 §3](../05_attention_and_kv_cache/03_embedding_and_lm_head.md)).

That matrix then flows through a stack of identical *transformer layers* (a
modern model has tens of them). The single most useful thing to see is that
**one transformer layer is not one operation — it's a small graph of separate
kernels**, each a GPU function ([GPU execution model §2](../01_hardware_fundamentals/03_gpu_model.md#2-gpu-threads-map-work-to-data))
launched in turn. Here is one layer, with the kernel for each step and the
hardware that runs it:

```text
  activations in  [N x d_model]
        |
        v   --- ONE TRANSFORMER LAYER (repeated ~32-80 times) ---
        |       (steps run top to bottom; the tag is kernel type, hardware)
        |
        |   RMSNorm             (elementwise, FP lanes)
        |   Q,K,V projection    (GEMM, Tensor Cores)
        |   RoPE                (elementwise, FP lanes)
        |   attention           (FlashAttention, Tensor Cores + FP lanes)
        |   output projection   (GEMM, Tensor Cores)
        |   residual add        (elementwise, FP lanes)
        |   RMSNorm             (elementwise, FP lanes)
        |   MLP up projection   (GEMM, Tensor Cores)
        |   activation          (elementwise, FP lanes)
        |   MLP down projection (GEMM, Tensor Cores)
        |   residual add        (elementwise, FP lanes)
        v
  activations out  [N x d_model]   ->  into the next layer
```

A few terms used there, defined inline:

```text
RMSNorm       rescales each activation vector to a stable size so the
              numbers don't blow up or vanish across many layers
projection    a matrix multiply that maps activations into a new space
              (Q, K, V are the Query/Key/Value matrices attention needs)
RoPE          rotary position encoding: rotates Q and K by an angle that
              depends on the token's position, so attention knows token order
attention     each token looks at every earlier token and pulls in a
              weighted blend of their values — the mechanism that lets a
              model use context
MLP           the feed-forward sub-block: two matmuls with a non-linearity
              between them (the "thinking" capacity of the layer)
residual add  a skip connection: adds the sub-block's input back to its
              output (the value from before that block's RMSNorm), which
              keeps signal flowing cleanly across many stacked layers
GEMM          General Matrix Multiply, C = A×B (ch2 §1) — the matmul kernel
              shape, run on Tensor Cores
```

Every box in that diagram is a forward pointer, and now they all have a
destination. This doc *names* them; **[Chapter 5](../05_attention_and_kv_cache/README.md)
is their anatomy**:

- **attention** (`Q·Kᵀ → softmax → ·V`), multi-head, and the KV cache →
  [§1](../05_attention_and_kv_cache/01_attention.md)
- the **MLP** boxes (up proj, activation, down proj), the layer's other half and
  most of its weights → [§2](../05_attention_and_kv_cache/02_mlp_feedforward.md)
- the **embedding** that produced the input matrix and the **LM head** that
  consumes the final one → [§3](../05_attention_and_kv_cache/03_embedding_and_lm_head.md)
- the **RMSNorm, RoPE, and residual** glue → [§4](../05_attention_and_kv_cache/04_elementwise_glue.md)

One step the diagram folds into the attention box: that box also
*writes this layer's K and V into the KV cache* — the structure §6 leans on to
make decode cheap. What it stores and why it's shaped that way is
[Chapter 5 §7](../05_attention_and_kv_cache/01_attention.md#7-the-kv-cache-what-it-stores-and-why-its-shaped-that-way).

The pattern to carry forward: **the big matmuls (the projections and the MLP)
are GEMMs that run on the Tensor Cores; everything else (norms, RoPE, the
activation function, residual adds) is an elementwise or reduction kernel on the
regular FP lanes.** The compute-heavy work and the glue work run on different
hardware inside the same SM.

Keep three things separate there, because the diagram's `GEMM -> Tensor Cores`
column tempts you to collapse them: a *GEMM* is the **operation** (a matrix
multiply); **cuBLAS** is the **library** that, at each call, picks a concrete
kernel to do it ([§7](#7-who-chooses-the-kernel-and-when)); and the **Tensor
Cores** are the **hardware** that kernel runs on. cuBLAS is neither a kernel nor
a piece of hardware — it's the chooser in the middle. §7 is the section that
makes that "who picks the kernel" question precise.

After the last layer, a final RMSNorm and one more matmul (the *LM head*,
projecting from `d_model` up to vocabulary size) produce the *logits*: one score
per vocabulary token, for each position
([GPU execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)
defined logits). Prefill cares only about the logits at the **last** position —
those decide the first generated token. (The LM head, and why it's often the
same matrix as the embedding, is
[Chapter 5 §3](../05_attention_and_kv_cache/03_embedding_and_lm_head.md); turning
those logits into the actual next token — temperature, top-k, top-p — is
[Chapter 5 §5](../05_attention_and_kv_cache/05_sampling.md).)

**Why prefill is compute-bound.** All N prompt tokens go through together, so
every matmul is matrix × matrix. A weight tile loaded into shared memory feeds
math for all N tokens before it's discarded — high *arithmetic intensity*
(math per byte loaded,
[GPU execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)).
The Tensor Cores stay busy. Prefill is where the GPU's advertised FLOP/s
actually get used.

## 5. Inside one kernel: a matmul, all the way down

Pick one GEMM from the layer above — say the MLP up-projection — and follow it
to the bottom of the stack. This is exactly the
[kernel launch → SM → warp scheduler → lane chain from GPU execution model §6](../01_hardware_fundamentals/03_gpu_model.md#6-putting-the-pieces-together-kernel-launch-to-lane-execution),
now carrying a weight tile instead of an array add. I won't redraw that chain;
I'll add only the matmul-specific layer on top of it.

```text
  framework calls torch matmul (or the engine's fused linear)
        |
        |  cuBLAS picks a kernel + tile size (see §7); CUDA runtime launches it
        v
  GPU: block scheduler hands tiles of the matmul to the ~132 SMs
       (execution model §6)
        |
        v
  one SM, for its assigned output tile:
        |
        |  1. load/store units stream a tile of A (activations) and a tile
        |     of B (weights) from VRAM -> L2 -> into SHARED MEMORY
        |     (the on-chip scratchpad, GPU execution model §10 — NOT VRAM)
        v
  shared memory holds the two tiles
        |
        |  2. warps (execution model §5) feed tile values into the TENSOR
        |     CORES, which do many multiply-accumulates (a×b+c) per
        |     instruction (execution model §11), while other warps hide
        |     the next tile's load latency (execution model §8)
        v
  results accumulate in REGISTERS (fastest storage, per thread)
        |
        |  3. when the output tile is complete, write it back to VRAM
        v
  VRAM holds the output tile  ->  becomes input to the next kernel
```

The thing this picture makes concrete, and the reason the earlier docs spent so
long on shared memory and tiling
([§9–§10](../01_hardware_fundamentals/03_gpu_model.md#9-matrix-multiplication-adds-data-reuse)):
the expensive step is moving bytes across the VRAM link at the top. So the SM
crosses it once to fill shared memory, then reuses those tile values for many
Tensor Core operations before going back. Every box below "shared memory" runs
at on-chip speed; only the top arrow touches slow off-chip memory.

## 6. Stage 3 — decode: the autoregressive loop

Prefill produced the first output token. Now the model must produce the rest,
and it can only make them **one at a time**, because each new token depends on
all the tokens before it. This is *autoregressive* generation, and it is a loop:

```text
  +---------------------------------------------------------------+
  |  DECODE — one turn of the loop = one new token                |
  |                                                               |
  |  1. take the ONE token produced last turn -> embed it         |
  |     (now a single vector, not an N-row matrix)                |
  |                                                               |
  |  2. run it through all layers (same graph as §4), BUT:        |
  |       - every matmul is now matrix x VECTOR (one token)       |
  |       - attention: this token's Q attends over ALL past       |
  |         K,V — which are ALREADY in the KV CACHE in VRAM,       |
  |         not recomputed. Append this token's new K,V.          |
  |                                                               |
  |  3. final norm + LM head -> logits -> sample one token        |
  +---------------------------------------------------------------+
        |
        |  the sampled token ID goes TWO places:
        |
        +---(a) FEEDBACK: back to step 1 as next turn's input -----+
        |       (this edge is the loop; K,V appended to the cache) |
        |                                                          |
        +---(b) SIDE BRANCH: detokenize -> stream text to the user |
                                                                   |
        loop until end-of-sequence token, or max length     <-----+
```

The *KV cache* is the running list of every past token's attention Keys and
Values, kept in VRAM so each new token doesn't re-run attention over the whole
history ([GPU execution model §13](../01_hardware_fundamentals/03_gpu_model.md#13-hbm-and-gddr-are-large-off-chip-memory)
introduced it). It is what turns a would-be O(N²) recompute into "read the cache,
append one entry." Weights, activations, **and** the KV cache all live in VRAM
(§2). The full mechanics — what's cached, the exact layout, the size formula, and
why long-context decode is bound by cache bandwidth — are in
[Chapter 5 §7](../05_attention_and_kv_cache/01_attention.md#7-the-kv-cache-what-it-stores-and-why-its-shaped-that-way).

The feedback edge (a) is what actually generates the next token: the sampled
token ID is fed straight back in as the single input to the next turn, and its freshly computed
K,V are appended to the cache so the turn after that can attend to it. The text
the user sees (b) is a *side branch* — detokenizing for display doesn't feed the
model; the loop would run identically with the screen off.

**Why decode is memory-bound — at batch size 1.** With one token in flight,
every matmul is matrix × vector: each weight is read from VRAM and used for a
single multiply-accumulate, then discarded. Arithmetic intensity is about
1 FLOP per byte — far below the few-hundred-FLOP/byte ratio the hardware wants
([GPU execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)).
So every turn of the loop must stream the **entire** weight set (e.g. 140 GB)
plus the **entire** KV cache out of VRAM, and the Tensor Cores sit mostly idle
waiting. Token speed is set by VRAM bandwidth, not FLOP/s:

```text
  140 GB of weights ÷ 3.35 TB/s (an H100's bandwidth)
        ≈ 42 ms per token  ≈ a ceiling of ~24 tokens/sec
  from bandwidth alone — even with infinitely fast Tensor Cores.
```

That "at batch size 1" qualifier is load-bearing, and §9 is where it gets
lifted.

## 7. Who chooses the kernel, and when

A model has dozens of distinct operations and, for each, many possible kernels —
different tile sizes, different algorithms, one tuned per GPU and per shape. So
"who picks the kernel?" is a real question, and the honest answer is: **several
different deciders, each acting at a different *time*.** Sorting them by *when*
is what makes this stop being hand-wavy.

```text
  WHEN                 WHO                      WHAT THEY DECIDE
  ----                 ---                      ----------------
  build time (human)   engine / model author    which OPERATOR implements
                                                 each layer — e.g. "attention
                                                 uses FlashAttention", "the
                                                 projection is a cuBLAS Linear"

  every call           PyTorch dispatcher       which BACKEND an op routes to,
                                                 by (device, dtype, layout).
                                                 Mechanical lookup, no choice
                                                 of algorithm.

  every call           library heuristics       which concrete GEMM kernel +
                       (cuBLAS / cuBLASLt)       tile config, given the actual
                                                 M,N,K dimensions, dtype, and
                                                 GPU architecture.

  first call per shape  autotuner               benchmarks several candidate
  (then cached)        (Triton @autotune,       configs the FIRST time a shape
                       CUTLASS profiler,         is seen, keeps the winner.
                       cuBLASLt algo search)     This is a JIT / first-run cost.
```

The insight to take away: there is no single chooser. Ask "who picks the kernel"
and the answer depends on *when* you ask — a human wired the operator choice
months ago, the dispatcher routes it on every call, a library heuristic sizes it
per call, and an autotuner may benchmark options the very first time it sees a
new shape.

That last row is why the first run of a kernel is slow and later runs are fast:
the autotuner's search and the driver's PTX→SASS compile
([ch2 §1](01_the_stack.md#the-libraries-on-top)) both happen once, then results
are cached. This is exactly the *first-run effect* that benchmarking has to
warm up past — see
[First-run effects §2](../04_measurement/02_first_run_effects.md#2-jit-compilation).

## 8. How a kernel is made fast

Choosing a kernel assumes fast kernels exist to choose from. Three ideas do most
of the work, and two of them the earlier docs already built:

```text
tiling       load a sub-block into shared memory once, reuse it for many
             multiply-accumulates before going back to VRAM. The core idea
             of §5 above and GPU execution model §9-§10.

fusion       merge several elementwise steps into ONE kernel so intermediate
             values never leave the chip. Instead of "matmul -> write to VRAM
             -> read back -> add bias -> write -> read -> activation", a fused
             kernel does matmul + bias + activation in one launch, keeping the
             intermediate in registers. Fewer VRAM round-trips, fewer launches.

autotuning   for a given shape and GPU, benchmark candidate tile sizes and
             pick the fastest (the §7 first-call step). The best tile size for
             a 4096x4096 matmul on an H100 is not the best for a 512x512 on a
             T4, so the choice is made empirically per case.
```

Fusion is the new one here, and it matters most exactly where the hardware is
starved: the elementwise glue work (norms, activations, residual adds) is itself
memory-bound, so collapsing several such kernels into one removes VRAM
round-trips that would otherwise dominate. This is why serving engines ship
hand-fused kernels for those steps (§9), and a recurring lever in
[GPU execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound).

## 9. The serving engine: who drives the loop

So far the picture has a model and a loop, but something has to *run* that loop —
receive requests, decide what to compute next, manage the KV cache, and keep the
GPU fed. That is the **serving engine**: the top rung of the stack from §1,
running on the CPU. The two this project benchmarks are **vLLM** and **SGLang**
(roadmap M12–M13). At this altitude — an orientation map — here is what they
add over plain PyTorch:

```text
vLLM      PagedAttention      stores the KV cache in fixed-size blocks (like
                              OS memory pages) instead of one contiguous
                              chunk, so VRAM doesn't fragment as sequences
                              grow and shrink
          continuous batching  swaps finished sequences out and new ones in
                              every step, instead of waiting for a whole batch
                              to finish — keeps the GPU busy
          prefix caching      reuses the KV cache of a shared prompt prefix
                              across requests
          chunked prefill     splits a long prefill so it can interleave with
                              ongoing decodes

SGLang    RadixAttention      shares KV-cache prefixes across requests using a
                              radix tree, so common prefixes are computed once
          prefix cache        the same reuse idea, organized for agent-style
                              repeated prompts
```

A tempting overstatement to resist: **the engine does not replace
PyTorch wholesale.** It replaces the *attention path* (with PagedAttention /
FlashInfer-style kernels) and *fuses* some glue ops (RMSNorm, RoPE, the MLP
activation), but the big linear layers still go through **cuBLAS** GEMMs on
Tensor Cores, same as §4. The engine is the conductor and the attention
specialist; cuBLAS is still the workhorse for the projections. (I'll verify the
finer details of each engine's kernel choices when I reach M12–M13 — this doc
stays at map altitude on purpose.)

A step back, since "what they add over PyTorch" invites the obvious question —
why can't PyTorch just do this itself? Because **PyTorch is a compute framework,
not a serving system.** It knows how to run *one forward pass*, and the engines
are built *on top of* it, reusing its tensors, its memory allocator, and most of
its kernels. What PyTorch has no native notion of is everything that lives
*across* requests: a scheduler deciding which of hundreds of in-flight sequences
run this step (continuous batching), and a block allocator plus custom kernel for
KV caches that are not contiguous (PagedAttention). Those two are genuinely beyond
stock PyTorch, and both trace back to the KV cache — the dynamic, per-request
structure that PyTorch's static, contiguous tensors fit badly. Not everything here
is exclusive to the engines, though: CUDA graphs, often cited as an engine trick,
are a *PyTorch* feature the engines merely *apply* to the decode loop (M18). So the
dividing line is scope, not capability — PyTorch optimizes a single forward pass;
the engine optimizes the throughput of a whole stream of requests sharing the GPU.

**This is where the "at batch size 1" qualifier from §6 gets lifted.** Decode is
memory-bound *for a single sequence* because each weight load feeds only one
token's math. Continuous batching changes that: the engine runs many sequences'
decode steps **together**, so one weight load out of VRAM feeds *all* of them at
once. Arithmetic intensity climbs from ~1 FLOP/byte toward the compute-bound
side, and the expensive weight read is amortized across the batch. So "decode is
memory-bound" and "batching makes decode efficient" are not in conflict — the
first is the batch-size-1 baseline, and batching is the lever that moves you off
it. Maximizing how many sequences share each weight load is, per the roadmap,
the single biggest throughput lever in serving (M12.5). The full mechanics —
why arithmetic intensity scales with batch size, static vs. continuous batching,
and how the KV cache caps it — are
[Chapter 6 — Batching](../06_batching/01_batching.md).

## 10. What each hardware component does

A recap of the bottom half of the stack, pinned to where each part showed up in
the flow above:

| Component | Its job in inference | Seen in |
| --- | --- | --- |
| CPU (host) | runs the engine + framework; tokenizes; launches kernels | §3, §9 |
| PCIe bus | carries weights CPU→GPU at load; carries tokens in/out | §2 |
| VRAM (HBM/GDDR) | holds weights, KV cache, activations; the bandwidth that caps decode | §2, §6 |
| L2 cache | shared staging area between VRAM and all SMs | §5 |
| SM | the worker unit; runs each kernel's blocks of threads | §5 |
| warp scheduler | picks a ready warp each cycle; hides VRAM latency | §5 |
| shared memory | on-chip scratchpad holding reused tiles (not VRAM) | §5 |
| registers | fastest per-thread storage; holds matmul accumulators | §5 |
| Tensor Cores | do the big matmuls (projections, MLP, attention) | §4, §5 |
| FP/INT lanes | do the glue: norms, RoPE, activation, residual, sampling | §4 |
| load/store units | stream tiles between VRAM, L2, and shared memory | §5 |

## 11. What each software layer does

And the top half:

| Layer | Its job in inference | Seen in |
| --- | --- | --- |
| serving engine | requests, batching, KV-cache management, the generation loop | §9 |
| ML framework (PyTorch) | the model as a graph of ops; dispatch each to a kernel | §4, §7 |
| kernel libraries | cuBLAS (matmul), FlashAttention (attention), fused kernels | §4, §8 |
| autotuner / dispatcher | pick the concrete kernel + config for each op and shape | §7 |
| CUDA runtime | allocate VRAM, copy data, launch kernels | §2, ch2 §1 |
| CUDA driver | turn launches into hardware commands; compile PTX→SASS | §7, ch2 §1 |

## 12. Follow one token, all the way down and back

Tying both axes together — one full turn of the decode loop, from the engine's
decision down to a Tensor Core and back up to text on a screen:

```text
  1. ENGINE (CPU)      decides this sequence runs this step; gathers its
                       last token ID and a pointer to its KV-cache blocks
  2. FRAMEWORK (CPU)   walks the model graph; for each op, dispatches to a
                       kernel (§7)
  3. RUNTIME (CPU)     launches each kernel: kernel<<<grid,block>>>(...)
  4. DRIVER (CPU->GPU) turns launches into GPU commands; SASS already compiled
  5. GPU: per layer    block scheduler -> SMs; matmuls hit Tensor Cores via
                       shared-memory tiles (§5); attention reads this
                       sequence's KV cache from VRAM and appends one entry;
                       norms/RoPE/activation run on FP lanes
  6. GPU: LM head      final matmul -> logits in VRAM
  7. GPU: sampling     a sampling kernel picks one token ID from the logits
  8. BACK UP           the token ID returns to the engine:
                         (a) fed in as step-1 input for the NEXT turn  (loop)
                         (b) detokenized -> streamed to the user       (branch)
  9. repeat from 1 until end-of-sequence
```

Every box from §1's two diagrams appears here exactly once per token, and the
loop in step 8(a) is the edge that runs thousands of times to produce one
response. The user waits, almost entirely, on that loop turning — which is why
the whole rest of this project is about making each turn move fewer bytes and
keep the SMs busier.

## 13. What to carry into the rest of the project

The map this doc draws is the thing the roadmap topics fill in with real code
and real numbers:

```text
the kernels in §4 (RMSNorm, RoPE, softmax, matmul)   -> M3-M6, built and measured
a transformer layer as a graph of kernels (§4)        -> M9, assembled for real
the KV cache and the decode loop (§6)                 -> M10-M11
prefill vs decode, compute- vs memory-bound (§4, §6)  -> M11, benchmarked
the serving engines (§9)                              -> M12-M13, vLLM & SGLang
batching as the throughput lever (§9)                 -> M12.5, a toy scheduler
how kernels are made fast (§8): fusion, tiling        -> M15-M18 (quant, Flash-
                                                         Attention, paged attn,
                                                         CUDA graphs)
```

The single sentence to keep: **a prompt becomes tokens by loading weights into
VRAM once, computing the whole prompt in a compute-bound prefill, then looping a
memory-bound decode that re-reads those weights for every token — and every
software layer above exists to feed that loop and choose the kernels that make
each turn cheaper.**
