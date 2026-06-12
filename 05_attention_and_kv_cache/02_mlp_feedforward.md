# The MLP / feed-forward block

This finishes the transformer layer. [Attention](01_attention.md) was one of the
two sub-blocks inside the layer graph from
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels);
this is the other one — the *feed-forward network* (FFN), also called the *MLP*
(multi-layer perceptron). Between them they account for essentially all of a
layer's weights and compute, so once attention and the MLP both make sense, the
whole `end-to-end §5` layer graph is explained.

Same as the rest of this chapter, it's the map: I work out what the MLP computes
and why it dominates the weight budget, so that the M9 build and the M15
quantization work start from known numbers.

Prerequisites: [Attention and the KV cache](01_attention.md) (its §6 prefill-vs-decode
shapes and §8 two-walls discussion), and
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels).
Next: M9 in the [Roadmap](../ROADMAP.md).

## 1. What the MLP is for

Attention and the MLP do opposite jobs, and the contrast is the cleanest way to
hold both in your head:

```text
  attention   MIXES across tokens   — each token pulls in information from
                                      the other tokens it's allowed to see
  MLP         transforms each token  — works on every token's vector
              INDEPENDENTLY           on its own, no looking at neighbours
```

The MLP is *position-wise*: the exact same function is applied to each token's
vector separately, with no information crossing between positions. Attention is
the only place tokens interact ([attention §1](01_attention.md#1-what-attention-is-for)); the MLP is where each token's
gathered-up representation gets re-processed. A useful (if loose, and not fully
settled) intuition is that the MLP is where much of the model's learned
*knowledge* lives — it's often interpreted as a large key-value memory that each
token queries — whereas attention is the *routing* that decides which tokens
share information. Take that as an aid to intuition, not a proven claim.

Mechanically, the MLP takes one vector of width `d_model` in and produces one
vector of width `d_model` out, per token, by **expanding to a much wider hidden
size and contracting back**.

## 2. The classic feed-forward block: expand, activate, contract

The original transformer FFN is two matrix multiplies with a non-linearity
between them:

```text
  x  [d_model]
     |
     |  up projection   W_up   [d_model × d_ff]      (matmul)
     v
  h  [d_ff]            <- a much WIDER hidden vector, d_ff ≈ 4 × d_model
     |
     |  activation      (elementwise non-linearity, e.g. GELU)
     v
  h' [d_ff]
     |
     |  down projection W_down [d_ff × d_model]       (matmul)
     v
  y  [d_model]
```

The hidden width `d_ff` (also written `d_intermediate` or `ffn_hidden`) is
conventionally about `4 × d_model` — e.g. `d_model = 4096`, `d_ff = 16384`.

**Why expand then contract.** A single matmul is a linear map; stacking two with
nothing between them collapses to one linear map and buys nothing. The
non-linearity in the middle is what lets the block learn curved, non-linear
functions. Projecting up to a wide `d_ff` first gives that non-linearity a
high-dimensional space to act in (more room to separate features), and the down
projection mixes the result back to `d_model` so the next layer sees a vector of
the original width. Expand → bend → contract.

## 3. Activation functions: the non-linearity in the middle

The *activation* is an elementwise function — it's applied to each entry of the
`d_ff`-wide hidden vector independently. Three that matter, in historical order:

```text
  ReLU(x)  = max(0, x)              zero out negatives. Simple, but the hard
                                    cut at 0 kills gradient for negative inputs.
  GELU(x)  ≈ x · Φ(x)              a smooth ReLU; Φ is the Gaussian CDF
                                    (the probability a standard normal is < x).
                                    Used by GPT-2/BERT-era models.
  SiLU(x)  = x · σ(x)              "Swish"; σ is the sigmoid 1/(1+e^−x).
                                    Smooth, and the building block of SwiGLU (§4).
```

All three are cheap elementwise ops that run on the FP lanes
([execution model §4](../01_hardware_fundamentals/03_gpu_model.md#4-the-sm-is-the-gpu-execution-unit)), not the Tensor Cores — the same glue-work split
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
drew for the whole layer. A quick numeric feel for SiLU:

```text
  SiLU(−2) = −2·σ(−2) = −2·0.119 = −0.24    (negatives are softly suppressed,
  SiLU( 0) =  0·0.5   = 0                     not hard-clipped)
  SiLU( 2) =  2·σ( 2) =  2·0.881 =  1.76      (positives pass through ~linearly)
```

## 4. SwiGLU: the gated FFN modern models use

Most current LLMs (Llama, Mistral, PaLM, and others) replace the two-matrix FFN
with a *gated* variant called **SwiGLU**. It uses **three** weight matrices
instead of two: a *gate*, an *up*, and a *down*.

```text
  x  [d_model]
     |--------------------------+
     |                          |
     | W_gate [d_model×d_ff]     | W_up [d_model×d_ff]      (two matmuls)
     v                          v
  g  [d_ff]                    u  [d_ff]
     |                          |
     | SiLU(g)                  |
     v                          |
   SiLU(g) [d_ff] ---⊙--------- u   (elementwise multiply: the "gate")
                     |
                     v
                  z  [d_ff]
                     |
                     | W_down [d_ff×d_model]                 (matmul)
                     v
                  y  [d_model]
```

As one formula:

```text
  SwiGLU(x) = ( SiLU(x · W_gate)  ⊙  (x · W_up) ) · W_down
            \_______ gate _______/   \__ value _/
  (⊙ is elementwise multiply, over the d_ff entries)
```

**Why gate.** The `SiLU(x·W_gate)` branch produces, per hidden unit, a
*data-dependent multiplier* — a number that scales how much of the `x·W_up`
branch passes through for *this specific input*. So instead of one fixed
non-linearity, the block learns to open and close each hidden channel depending
on the token. In practice this trains better than a plain FFN at the same size,
which is why it won out.

**The width, and a subtlety.** SwiGLU has three `d_model×d_ff` matrices where the
classic FFN had two. To keep the parameter count the same, you shrink `d_ff`. The
*param-preserving baseline* is:

```text
  3 · d_model · d_ff  =  8 · d_model²   →   d_ff = 8/3 · d_model  ≈ 2.67 · d_model
```

That `8/3` is the *reason* models land near 2.7×, not a fixed rule. Real models
pick `d_ff` anywhere in roughly the **2.7×–4× `d_model`** range:

```text
  Llama-2-7B    d_model 4096,  d_ff 11008   = 2.69 × d_model   (≈ the 8/3 baseline)
  Llama-2-70B   d_model 8192,  d_ff 28672   = 3.5  × d_model   (wider than baseline)
```

So treat `8/3` as the param-matched anchor and the actual figure as a per-model
choice.

## 5. Shapes and hardware

Like attention ([attention §3](01_attention.md#3-the-computation-step-by-step)), the MLP splits cleanly across the GPU's two
kinds of arithmetic unit:

```text
  up / gate / down projections   matmuls          -> Tensor Cores
  activation (GELU / SiLU)         elementwise      -> FP lanes
  the SwiGLU gate multiply (⊙)     elementwise      -> FP lanes
```

These are the `MLP up proj`, `activation`, and `MLP down proj` boxes from
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
— the projections are GEMMs through cuBLAS on the Tensor Cores, the activation
and gate are cheap elementwise kernels on the FP lanes. A small shape-walk for
one token, `d_model = 4`, `d_ff = 8` (toy widths):

```text
  x        [4]            one token's vector
  x·W_gate [8]  --SiLU--> [8]   \
  x·W_up   [8] ----------- [8]   |  ⊙ elementwise -> z [8]
                                /
  z·W_down [4]                  the output, back to width 4
```

In prefill all `N` prompt tokens go through together, so each projection is a
matrix×matrix GEMM (compute-bound); in decode it's one token, matrix×vector
(memory-bound) — the same prefill/decode shape split as [attention §6](01_attention.md#6-prefill-vs-decode-the-same-math-two-very-different-shapes), since
the MLP is just matmuls.

## 6. Why the MLP is most of the model's weights

Count the parameters in one layer. Attention has four `d_model×d_model` matrices
— `W_Q`, `W_K`, `W_V`, `W_O` (each is `d_model×d_model` because the `h` heads of
width `head_dim` satisfy `h · head_dim = d_model`):

```text
  attention weights / layer  =  4 · d_model²
```

The classic two-matrix MLP at `d_ff = 4·d_model`:

```text
  MLP weights / layer  =  2 · d_model · d_ff  =  2 · d_model · (4·d_model)  =  8 · d_model²
```

So MLP : attention = `8 : 4` = **2 : 1** — the MLP is **≈2/3 of the layer's
weights** in the classic case. (SwiGLU's three matrices at `d_ff ≈ 8/3·d_model`
come out to the same `8·d_model²`, by construction — §4.)

But `2/3` is a **floor**, not the typical figure, because GQA ([attention §7](01_attention.md#7-the-kv-cache-what-it-stores-and-why-its-shaped-that-way))
shrinks `W_K` and `W_V` — they become `d_model × (n_kv_heads·head_dim)` instead
of `d_model × d_model`, which is much smaller. Worked out for Llama-2-70B
(`d_model 8192`, `n_heads 64`, `head_dim 128`, `n_kv_heads 8`, `d_ff 28672`):

```text
  attention / layer
    W_Q   8192 × 8192        = 67.1M
    W_K   8192 × (8·128=1024) =  8.4M    <- shrunk by GQA
    W_V   8192 × 1024         =  8.4M    <- shrunk by GQA
    W_O   8192 × 8192        = 67.1M
    total                    ≈ 151M

  MLP / layer  (SwiGLU, 3 matrices)
    W_gate 8192 × 28672      = 234.9M
    W_up   8192 × 28672      = 234.9M
    W_down 28672 × 8192      = 234.9M
    total                    ≈ 705M

  MLP share  =  705 / (705 + 151)  ≈  82% of the layer's weights
```

So in a modern GQA model the MLP is not two-thirds but **~80%** of the weights.
This is why quantization (M15) targets the MLP first: it's where the bytes are.

## 7. Why MLP decode is weight-memory-bound — the *first* wall

Attention doc §8 split "memory-bound" into two different walls. The MLP sits
squarely on the **first** one:

```text
  MLP decode is memory-bound on the WEIGHTS
     - bytes read per token ≈ the MLP's weights (the bulk of the model, §6)
     - FIXED: does not grow with context length
     - SHAREABLE across a batch: read each weight once, use it for every
       sequence in the batch (end-to-end §10) — so batching amortizes it
```

This is the *opposite* profile from attention's KV-cache wall ([attention §8](01_attention.md#8-why-decode-attention-is-memory-bound--and-its-a-different-wall)),
which grows with context and is per-sequence. The MLP has no per-token state at
all — it's pure weights in, one vector out — so its only memory cost is streaming
those (large) weights, and that cost is constant per token and falls per-token as
you batch. (How batching turns that into a throughput win is
[Chapter 6](../06_batching/01_batching.md).)

The arithmetic intensity at batch size 1 is the familiar ~1 FLOP/byte: a
projection of `d_model × d_ff` does `2·d_model·d_ff` FLOPs and reads
`2·d_model·d_ff` bytes (fp16), used once for the single token
([execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)). The Tensor Cores idle, waiting on the weight stream —
exactly the decode picture from
[end-to-end §7](../02_cuda_software_stack/02_end_to_end_inference.md#7-stage-3--decode-the-autoregressive-loop),
and since the MLP is most of the weights, it's most of that wait.

## 8. Further reading

Where each piece of the MLP comes from — the papers that defined it, plus one
readable implementation:

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al.,
  2017) — §3.3 defines the position-wise feed-forward network this doc opens with: two
  linear layers with a non-linearity between, applied to each token independently (§2).
- **[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)** (Hendrycks
  & Gimpel, 2016) — the activation that replaced ReLU in transformers, and the baseline
  SwiGLU is measured against (§3).
- **[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)** (Noam Shazeer,
  2020) — the two-page paper that introduced SwiGLU and the gated three-matrix FFN modern
  models use (§4).
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** (Andrej Karpathy) — a ~300-line GPT
  whose `MLP` module is exactly the expand → activate → contract block of §2, worth
  reading before assembling it for real in M9.

## 9. What to carry forward

```text
the up → activation → down structure (§2)        -> M9, built into the layer
GELU / SiLU activation kernels (§3)               -> M4-adjacent elementwise work
SwiGLU's 3-matrix gated form (§4)                 -> M9, the real FFN
MLP is ~2/3–80% of the weights (§6)               -> M15, the main quantization target
MLP decode = weight-bound, batch-shareable (§7)   -> M11-M12, why batching helps it
```

The one sentence to keep: **the MLP transforms each token independently by
expanding to a wide hidden size, applying a (gated) non-linearity, and
contracting back; it holds most of the model's weights, so in decode it's the
biggest contributor to the fixed, batch-shareable weight-bandwidth wall —
the counterpart to attention's growing, per-sequence KV-cache wall.**
