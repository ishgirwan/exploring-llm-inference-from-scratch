# The elementwise glue: RMSNorm, RoPE, and residuals

[Attention](01_attention.md) and [the MLP](02_mlp_feedforward.md) are
the matmul-heavy sub-blocks, and [the two ends](03_embedding_and_lm_head.md)
bracket the stack. But the
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
layer graph had a whole set of *other* boxes between the matmuls — RMSNorm, RoPE,
the activation, the residual adds — all drawn on the **FP lanes** and waved past
as "elementwise glue." This section opens those boxes. The activation already got
its treatment in `MLP §3`; this is the other three: **RMSNorm**, **RoPE**, and the
**residual add**.

They are individually trivial — a few arithmetic operations per number — which is
why no earlier doc stopped on them. But they are not optional, and collectively
they are a real slice of decode latency, for a reason that turns out to be the
whole point: they are *memory-bound*, so they're exactly what kernel **fusion**
([end-to-end §9](../02_cuda_software_stack/02_end_to_end_inference.md#9-how-a-kernel-is-made-fast))
exists to optimize. Understanding why is the payoff of this section.

Same as the rest of the chapter, it's the map, not the build: I work out what
each computes so the M4 (RMSNorm) and M5 (RoPE) kernel builds start from known
math.

Prerequisites:
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
(the layer graph these sit in) and [attention §1](01_attention.md#1-what-attention-is-for)
(where the Q and K that RoPE rotates come from).
Next: M4 and M5 in the [Roadmap](../ROADMAP.md) — the RMSNorm and RoPE kernels,
built and measured.

## 1. What "glue" means, and why it gets its own section

An *elementwise* operation produces each output number from just one (or a few)
input numbers in the same place — no big matrix multiply, no mixing across the
whole vector. The glue ops are all elementwise (or close to it), so they run on
the GPU's regular *FP lanes* — the ordinary floating-point arithmetic units
([execution model §4](../01_hardware_fundamentals/03_gpu_model.md#4-the-sm-is-the-gpu-execution-unit))
— not the Tensor Cores that do the matmuls.

Here's the tension that makes them worth a section. Each glue op does almost no
arithmetic, but it still has to **read its input from VRAM and write its output
back**. So its cost is dominated by memory traffic, not math — it is
*memory-bound*, with arithmetic intensity near 1 FLOP/byte
([execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)).
And there are *many* of them: two RMSNorms, a RoPE, and two residual adds **per
layer**, times 32–80 layers. A naive implementation runs each as its own kernel,
each making a full round trip to VRAM. That's the waste *fusion* removes — merging
glue steps into the neighbouring kernel so the intermediate never leaves the chip
(`end-to-end §9`). The glue is cheap in FLOPs but, done carelessly, expensive in
the one resource decode is starved for: bandwidth.

## 2. RMSNorm: rescale each vector to a stable size

Stacking tens of layers risks each one's output growing or shrinking the numbers a
little, until after many layers they blow up to infinity or vanish to zero.
*Normalization* fixes the scale of each token's vector before it enters a
sub-block, so the numbers stay in a sane range no matter how deep the stack.
Modern LLMs use **RMSNorm** (root-mean-square normalization):

```text
  RMSNorm(x)_i  =  g_i · x_i / sqrt( (1/d) Σ_j x_j²  +  ε )
                   \_gain_/         \___ the RMS (root mean square) of x ___/

  d = d_model (the vector's length);  g = a learned per-channel gain vector;
  ε = a tiny constant so we never divide by zero
```

In words: measure the vector's typical magnitude (its RMS), divide every entry by
it so the vector has RMS 1, then multiply by a learned gain `g` that lets the
model choose each channel's scale back. A toy vector, `d = 4`:

```text
  x = [ 2, -2, 1, -1 ]
  mean of squares = (4 + 4 + 1 + 1) / 4 = 2.5
  RMS = sqrt(2.5) ≈ 1.58
  x / RMS = [ 1.27, -1.27, 0.63, -0.63 ]   <- now has RMS 1 (then scaled by g)
```

**RMSNorm vs LayerNorm.** The older *LayerNorm* also subtracts the vector's mean
first (`(x − μ)/σ`) and adds a learned bias. RMSNorm drops both the
mean-subtraction and the bias — it *only* rescales. It turned out the
re-centering wasn't pulling its weight, so RMSNorm does less work for the same
quality, which is why Llama, Mistral, and most current models use it. It sits
**before** attention and **before** the MLP — the *pre-norm* placement (§4) — plus
one final RMSNorm before the LM head ([the two ends](03_embedding_and_lm_head.md)).

## 3. RoPE: inject token position by rotating Q and K

Attention has a blind spot: the score `Q·Kᵀ` ([attention §3](01_attention.md#3-the-computation-step-by-step))
is a dot product, and a dot product doesn't know *where* either token sits in the
sequence. Without help, "dog bites man" and "man bites dog" look identical to
attention. *Positional encoding* is what tells the model the order, and modern
models use **RoPE** — rotary position encoding.

The idea: take the Query and Key vectors (right after the Q,K,V projection,
`end-to-end §5`), split each into 2-D pairs of coordinates, and **rotate each pair
by an angle proportional to the token's position**. Different pairs rotate at
different speeds (frequencies), so the full rotation encodes position richly.

```text
  one 2-D pair (a, b) of a token at position m, rotated by angle φ = m · θ:

     [ a' ]   [ cos φ   −sin φ ] [ a ]
     [ b' ] = [ sin φ    cos φ ] [ b ]

  same vector (1, 0), different positions (toy, θ = 1 radian per step):
     m = 0:  φ = 0  → (1.00, 0.00)
     m = 1:  φ = 1  → (0.54, 0.84)     <- the SAME token now points elsewhere
     m = 2:  φ = 2  → (−0.42, 0.91)       because it sits at a later position
```

**Why rotation, and why only Q and K.** Rotation has a property no additive scheme
has cleanly: when you later take the dot product of a rotated query at position
`m` with a rotated key at position `n`, the result depends only on the **relative**
distance `m − n`, not on the absolute positions. So attention naturally learns
"how far apart" two tokens are. RoPE touches **Q and K only** — never V — because
position should shape *which* tokens attend to which (the scores), not the values
being blended in. It's a pure elementwise rotation: no weights of its own beyond
the fixed frequencies, no matmul.

## 4. Residual adds: the skip connection that keeps deep stacks trainable

A *residual add* (or *skip connection*) adds a sub-block's **input** back to its
**output**. It's why a layer is written as "add a correction to what came in"
rather than "replace what came in." A modern *pre-norm* layer is two residual
adds wrapped around the two sub-blocks:

```text
  x  ─────────────────────────────────┐  (keep a clean copy)
  |                                    |
  RMSNorm → attention ────── add ──────┘
  |                                    ┐
  RMSNorm → MLP ──────────── add ──────┘
  |
  out      (same width d_model as x, all the way through)
```

As formulas: `x ← x + Attention(RMSNorm(x))`, then `x ← x + MLP(RMSNorm(x))`.

**Why add the input back.** Two reasons, both about depth. First, signal flow: with
the skip path, information (and, during training, gradients) can travel straight
down the stack without being mangled by every sub-block — so 80-layer models stay
trainable. Second, it reframes each block's job: it only has to learn a small
*correction* to the running vector, not reconstruct it from scratch, which is much
easier. That running, added-to vector flowing down the whole model is called the
*residual stream*, and it stays width `d_model` from the embedding to the
final norm. The add itself is the cheapest op in the layer — one elementwise
addition — but structurally it's what makes the depth work.

## 5. Shapes and hardware

All three run on the FP lanes, never the Tensor Cores, and all three are
memory-bound (§1):

```text
  RMSNorm        per-vector reduce (Σx²) + elementwise scale   -> FP lanes
  RoPE           elementwise 2-D rotation of Q and K           -> FP lanes
  residual add   elementwise add of two [N × d_model] tensors  -> FP lanes
```

These are the `RMSNorm`, `RoPE`, and `residual add` boxes from `end-to-end §5` —
the ones it grouped as "everything else is an elementwise kernel on the regular FP
lanes." Because each is memory-bound and there are several per layer, the standard
optimization is to **fuse** them into the matmul kernels on either side, so the
activations don't make extra VRAM round trips (`end-to-end §9`, and the fused
kernels serving engines ship, `end-to-end §10`). The prefill/decode shape split
(`attention §6`) applies but barely matters here: elementwise work scales with the
number of tokens either way and never becomes the matmul-style bottleneck — it's
the matmuls and the KV cache that dominate, and the glue that fusion tucks in
around them.

## 6. Further reading

One primary source per glue op, plus the explainer that makes RoPE click:

- **[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)** (Zhang &
  Sennrich, 2019) — the paper that introduced RMSNorm (§2): why dropping LayerNorm's
  mean-centering keeps the stabilizing effect at lower cost.
- **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)**
  (Su et al., 2021) — the original RoPE paper (§3): encoding position by *rotating* Q and
  K so that attention sees relative distance.
- **[Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)**
  (EleutherAI, 2021) — the diagram-driven walk-through that made RoPE intuitive for me;
  the visual companion to the math in §3.
- **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)** (He
  et al., 2015) — the ResNet paper that introduced the skip connection (§4), and the
  reason very deep stacks stay trainable.

## 7. What to carry forward

```text
RMSNorm = rescale a vector to RMS 1, times a gain (§2)   -> M4, the RMSNorm kernel
RoPE = rotate Q,K by position → relative-position attn (§3) -> M5, the RoPE kernel
residual add = add the sub-block's input back (§4)        -> M9, built into the block
all three are memory-bound FP-lane glue (§5)              -> M15–M18, fusion targets
```

The one sentence to keep: **the glue ops — RMSNorm (rescale to a stable size),
RoPE (rotate Q and K so attention sees relative position), and the residual adds
(keep the deep stack trainable) — are cheap elementwise FP-lane work, but because
they're memory-bound and there are several per layer, fusing them into the
neighbouring matmuls is one of the standard ways a forward pass is made fast.**
