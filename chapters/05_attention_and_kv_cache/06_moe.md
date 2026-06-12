# Mixture of experts: when one MLP becomes many

Everything in this chapter so far described a **dense** model: every weight in
the model participates in every token's forward pass. [The MLP
doc](02_mlp_feedforward.md) worked out that the MLP holds most of those weights,
and [Chapter 8's decode doc](../08_optimizing_inference/01_improving_decode.md)
builds its whole story on streaming them per step. **Mixture of experts (MoE)**
breaks the dense assumption — and most current frontier open models (DeepSeek-V3,
Llama 4, Kimi K2, GPT-OSS, the larger Qwen3 models) are MoE, so the dense map
needs this amendment before it matches the models that engines actually serve.

The idea in one line: replace the single MLP in each layer with **many parallel
MLPs ("experts")** plus a tiny **router** that picks a few of them per token —
so the model can hold far more weights than any one token ever touches.

Prerequisites: [The MLP / feed-forward block](02_mlp_feedforward.md) — an expert
*is* that block — and [Attention and the KV cache](01_attention.md) for the
decode-is-memory-bound framing this doc keeps leaning on.
Next: [Attention variants](07_attention_variants.md) — the other place the dense
map bends — and M31 in the [Roadmap](../../ROADMAP.md), where a toy MoE gets built.

## 1. The dense assumption, and why it binds

In a dense model, making the model smarter means making every token's forward
pass more expensive, because parameters and per-token compute are the same dial:

```text
  dense:  params ↑  →  FLOPs per token ↑  →  bytes read per decode step ↑
          (one dial: every weight is read and used for every token)
```

The MoE bet is that the dial can be split. A model can hold a huge library of
learned transformations, while each token only pays for the few it needs:

```text
  MoE:    TOTAL params      what the model knows   — sets VRAM needed
          ACTIVE params     what one token touches — sets FLOPs and (mostly)
                                                     bytes per decode step
```

DeepSeek-V3 makes the split concrete: **671B total parameters, 37B active per
token** — each token's forward pass touches ~5% of the model.

## 2. The MoE block: a router and many experts

Only the MLP is replaced. Attention, the KV cache, norms, RoPE, embeddings — all
of it stays exactly as the earlier sections described. Inside an MoE layer:

```text
                        x  [d_model]   (one token, after attention)
                        |
                 router W_r [d_model × E]        (a tiny matmul: one row
                        |                         of scores per expert)
                        v
              scores [E]  --softmax-->  probs [E]
                        |
                   keep top-k            (k out of E experts, e.g. 8 of 256)
                        |
        +---------------+----------------+
        v                                v
   expert 3                          expert 41        ... (each expert is a
   SwiGLU FFN                        SwiGLU FFN            full gate/up/down
   [d_model -> d_ff_e -> d_model]    (same shape)          MLP, just narrower)
        |                                |
        +------- weighted sum by the ----+
                 renormalized probs
                        |
                        v
                        y  [d_model]
```

Three parts:

- **The experts.** Each expert is exactly [the SwiGLU
  MLP](02_mlp_feedforward.md) — gate, up, down — but with a much smaller hidden
  width `d_ff_e` than a dense model would use, because there are `E` of them.
- **The router** (also called the *gate*): one small `[d_model × E]` matmul
  producing a score per expert, softmaxed into probabilities. The top-`k`
  experts by probability process the token; the rest are skipped entirely.
- **The combine.** The chosen experts' outputs are summed, weighted by their
  (renormalized) router probabilities.

Many designs add one **shared expert** that every token passes through
regardless of routing (DeepSeek-V3 and Kimi K2 both do), so the routed experts
only have to learn what's *not* common to all tokens.

## 3. A worked example

One token, four experts, top-2. The router matmul produces logits, softmax turns
them into probabilities, and the top two are kept and renormalized:

```text
  router logits      [ 2.0   0.5   1.2  -1.0 ]
  softmax            [ 0.58  0.13  0.26  0.03 ]
  top-2              experts 1 and 3       (0.58 and 0.26)
  renormalize        0.58/(0.58+0.26) = 0.69      0.26/(0.58+0.26) = 0.31

  output  y = 0.69 · expert1(x)  +  0.31 · expert3(x)
```

Experts 2 and 4 do no work for this token — their weights aren't even read.
That last clause is the entire systems story of §5.

## 4. The models that use it

The current open frontier, to make "most models are MoE now" concrete:

```text
  model              total     active    experts            per token
  Mixtral 8x7B       47B       13B       8                  top-2
  DeepSeek-V3        671B      37B       256 + 1 shared     top-8 + shared
  Llama 4 Maverick   400B      17B       128                top-1 + shared
  Qwen3-235B-A22B    235B      22B       128                top-8
  GPT-OSS-120B       117B      5.1B      128                top-4
  Kimi K2            1T        32B       384 + 1 shared     top-8 + shared
```

The trend across rows is *finer granularity*: many small experts with a few
active (DeepSeek calls this fine-grained expert segmentation), rather than
Mixtral's eight big ones. The total/active ratio is the design's whole point —
Kimi K2 holds a trillion parameters and reads ~3% of them per token.

## 5. What MoE does to inference

This is why the section lives in this repo. Every consequence follows from one
sentence: **VRAM must hold the total parameters, but each token only reads the
active ones.**

**Capacity: the ceiling jumps.** All experts must sit in VRAM (the router picks
per token, per layer — there's no useful way to know in advance which experts
a request will want). DeepSeek-V3 at fp8 is ~671 GB of weights before any KV
cache — no single GPU holds it, so MoE at this scale *forces* multi-GPU serving.
[Chapter 8's multi-GPU doc](../08_optimizing_inference/03_scaling_past_one_gpu.md)
covers expert parallelism, the standard answer.

**Decode bandwidth: the bill shrinks to active.** Decode's weight wall — the
weight-memory-bound picture [the MLP doc](02_mlp_feedforward.md) ends on — is
set by bytes read per step.
At batch 1, an MoE reads only the chosen experts — the active params — so a
671B model decodes like a ~37B model streams. That is the trade working as
designed: dense-model knowledge ceiling broken without paying dense-model
bandwidth per token.

**Batching: the lever gets diluted.** [Chapter 6](../06_batching/01_batching.md)
shows batching working because every sequence in the batch needs the *same*
weights this step — one weight read, `B` tokens served. MoE breaks the "same
weights" part: each token routes to its own `k` of `E` experts, so a batch of
`B` tokens scatters into per-expert groups of roughly `B·k/E` tokens each:

```text
  dense:  one MLP,    B tokens          ->  per-matmul batch = B
  MoE:    E experts,  B·k expert-slots  ->  per-expert batch ≈ B·k/E

  DeepSeek-V3 numbers (k=8, E=256):  B=128  ->  ≈4 tokens per expert
```

So an MoE needs a much larger global batch before each expert's matmul climbs
out of the memory-bound regime — and at large `B`, *every* expert gets touched
each step, so the per-step weight traffic heads back toward total params (each
read amortized over only its own small group). The batching math of Chapter 6
still holds; it just applies per expert.

**Kernels: gather, grouped GEMM, scatter.** Serving an MoE step efficiently
means: sort/permute the tokens by assigned expert, run all the small expert
matmuls, and un-permute. Launching `E` separate GEMMs (general matrix-matrix
multiplications) wastes launch overhead and parallelism, so engines use a
**grouped GEMM** — one kernel that processes many small independent matmuls with
different group sizes in a single launch. This is the "fused MoE" kernel —
the same kernel family as the Kimi-K2.5 example in
[Chapter 9's AMD doc](../09_kernel_engineering/05_amd_kernel_track.md).
Routing also brings **load imbalance**: a popular
expert gets a big group while others idle, which training tries to prevent
(balance losses or bias tweaks) and inference has to live with.

## 6. Further reading

The papers behind each layer of the idea, in the order I'd read them:

- **[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)**
  (Shazeer et al., 2017) — the modern MoE blueprint: the softmax router, top-k
  selection, and the load-balancing problem of §5, a decade before it went mainstream.
- **[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)**
  (Fedus, Zoph, Shazeer, 2021) — pushed routing to its top-1 extreme inside a
  transformer and made the total-vs-active params framing of §1 standard.
- **[Mixtral of Experts](https://arxiv.org/abs/2401.04088)** (Jiang et al., 2024) —
  the model that made open-weight MoE real; short, and its §2 is the cleanest
  statement of the top-2 block in §2 of this doc.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)**
  (DeepSeek-AI, 2024) — fine-grained experts, the shared expert, and
  auxiliary-loss-free balancing; also documents the expert-parallel serving
  deployment that §5's capacity point forces.
- **[Accelerating MoEs with a Triton cache-aware grouped-GEMM kernel](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)**
  (PyTorch blog) — what the grouped GEMM of §5 looks like as an actual kernel.

## 7. What to carry forward

```text
total vs active params (§1)                  -> M11+, reading any modern model card
the router/top-k/combine block (§2, §3)      -> M31, the toy MoE build
per-expert batch ≈ B·k/E (§5)                -> M12-M14, why engines sweat MoE batching
grouped GEMM / fused-MoE kernels (§5)        -> M31, and the Ch9 AMD example
all experts resident -> multi-GPU (§5)       -> M20/M31, expert parallelism
```

The one sentence to keep: **MoE replaces each layer's MLP with many small expert
MLPs and a per-token top-k router, splitting "what the model knows" (total
params, which VRAM must hold) from "what one token pays" (active params, which
set per-step compute and bandwidth) — at the price of scattered batches, grouped
GEMMs, and almost-mandatory multi-GPU serving.**
