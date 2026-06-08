# The two ends: embedding and the LM head

This chapter has opened up the *inside* of a transformer layer —
[attention](01_attention.md) and [the MLP](02_mlp_feedforward.md).
But a layer is the *middle* of the model. The stack of identical layers has a
front door and a back door that the
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
map named in passing and then left unexplained: the front door turns token IDs
into the first vectors (the *embedding*), and the back door turns the last
layer's vectors into one score per vocabulary word (the *LM head*, short for
*language-model head*). They bracket the whole stack — and, a nice surprise,
they are often literally the same matrix used twice.

I'm writing this because when I traced a prompt down `end-to-end §5`, these were
the two steps I could *not* actually explain. The map said "embedding lookup (a
large weights matrix in VRAM)" and "LM head (projecting from `d_model` up to
vocabulary size)" without ever saying what either one *is*. This section is that
gap filled.

Same as the rest of the chapter, it's the map, not the build: I work out what
the two ends compute and what shape they are, so the M9 block assembly and the
M11 prefill/decode benchmarks start from known numbers.

Prerequisites:
[end-to-end §5](../02_cuda_software_stack/02_end_to_end_inference.md#5-stage-2--prefill-a-transformer-layer-is-a-graph-of-kernels)
(the layer graph these two bracket) and
[end-to-end §4](../02_cuda_software_stack/02_end_to_end_inference.md#4-stage-1--the-prompt-arrives-and-becomes-tokens)
(tokenization, which produces the token IDs the embedding consumes).
Next: M9 in the [Roadmap](../ROADMAP.md) — where both ends get wired into the
block, and M11, where the LM head's last-position logits drive prefill.

## 1. Where the two ends sit

The model is a sandwich: a stack of identical layers in the middle, with one
fixed step on each side. Everything in `attention §1`–`MLP §6` lives in the
middle band; this section is the two slices of bread.

```text
  token IDs   [15496, 4088, ...]      from the tokenizer (end-to-end §4)
       |
       |  EMBEDDING  — look up one row per token ID            <- FRONT DOOR
       v
  activations [N × d_model]
       |
       |  transformer layer × N   (attention §1 + MLP §2,
       v                           repeated ~32–80 times)
  activations [N × d_model]
       |
       |  final RMSNorm   (one last normalize, see the glue)
       |  LM HEAD        — project each vector to vocab scores  <- BACK DOOR
       v
  logits      [N × vocab]            one score per vocabulary word, per position
       |
       |  sampling        — turn the last position's logits into one token ID
       v
  next token
```

Two terms used there, both defined more fully below: the *vocabulary* is the
fixed set of all possible tokens the model knows (commonly 32,000–256,000
entries, from `end-to-end §4`), and `d_model` is the model's hidden width — the
length of the vector that represents each token as it flows through the layers
(e.g. 4096). The front door maps **one integer → one `d_model` vector**; the back
door maps **one `d_model` vector → one score per vocabulary entry**. They are
mirror images, and §4 below shows they are often the *same weights*.

## 2. The embedding: a table lookup, not a computation

The *embedding table* is a single weights matrix of shape `[vocab × d_model]` —
one row per vocabulary entry, each row a learned `d_model`-long vector. It is one
of the tensors loaded into VRAM at startup
([execution model §13](../01_hardware_fundamentals/03_gpu_model.md#13-hbm-and-gddr-are-large-off-chip-memory)),
shipped inside the model's weight file exactly like every other weight.

The "lookup" is the one step in the whole forward pass that does **no
arithmetic**. A token ID is used directly as a *row index*: token ID 15496 means
"copy out row 15496." No multiply, no non-linearity — just an indexed memory read
(a *gather*: pick out scattered rows by index and stack them). `N` token IDs
produce `N` rows, stacked into the `[N × d_model]` activation matrix the layers
then consume.

```text
  token IDs:  [3, 0, 5]
                 |  gather rows 3, 0, 5
                 v
  embedding table  [vocab × d_model]   (toy: vocab = 6, d_model = 4)
  ┌───────────────────────────┐
  │ row 0  → [ 0.1  -0.2  0.0  0.9 ]   │ ─┐ picked (token 0)
  │ row 1  → [ ...              ]      │  │
  │ row 2  → [ ...              ]      │  │
  │ row 3  → [-0.4   0.3  0.7  0.1 ]   │ ─┤ picked (token 3)
  │ row 4  → [ ...              ]      │  │
  │ row 5  → [ 0.2   0.2 -0.1  0.5 ]   │ ─┘ picked (token 5)
  └───────────────────────────┘
                 |
                 v
  activations  [3 × 4]   the three picked rows, stacked in prompt order
```

Because it is a gather, not a matmul, the embedding runs on the load/store units,
not the Tensor Cores — it moves bytes, it doesn't compute. Its cost is trivial
next to a single layer's matmuls, which is why `end-to-end §5` drew the layer
graph *starting after* the embedding: the lookup is too cheap to be part of the
story.

What each row *means* is learned during training: the vector is the model's
encoding of "what this token is," a point in `d_model`-dimensional space where
tokens used in similar ways sit near each other. The table is not a separate
model that processes the IDs — it is a frozen dictionary the rest of the model
reads.

## 3. The LM head: the mirror operation

After the last layer and one final RMSNorm (covered in
[the glue section](04_elementwise_glue.md)), the model holds, per position, a
`d_model` vector summarizing "everything known about what comes next here." The
*LM head* turns that into a guess over the whole vocabulary with a single matrix
multiply:

```text
  hidden  [d_model]                 one position's final vector
     |
     |  LM head  W_out  [d_model × vocab]      (a matmul → GEMM)
     v
  logits  [vocab]                   one raw score per vocabulary token
```

The output is the *logits*: one raw, unbounded score per vocabulary entry, where
a higher score means the model rates that token more likely to come next
([execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)
defined the term). Logits are not yet probabilities — turning them into a single
chosen token is [sampling](05_sampling.md)'s job.

Unlike the embedding, the LM head genuinely computes: it is a *GEMM* (general
matrix multiply, `end-to-end §5`) and runs on the Tensor Cores, the same workhorse
the projections and the MLP use. One subtlety carried over from `end-to-end §5`:
in **prefill** the model produces a hidden vector at every one of the `N`
positions, but only the **last** position's logits decide the first generated
token — so a serving engine computes the LM head for just that one position and
skips the other `N−1`. In **decode**, there is only ever one position, so its
logits are always what we want (the same prefill/decode shape split as
[attention §6](01_attention.md#6-prefill-vs-decode-the-same-math-two-very-different-shapes)).

## 4. Weight tying: the two ends are often one matrix

Look at the two shapes. The embedding is `[vocab × d_model]`; the LM head is
`[d_model × vocab]`. That is the *same* matrix, one just the transpose of the
other. So a model can store **one** `[vocab × d_model]` matrix and use it both
ways — read a row at the front door, multiply by its transpose at the back door.
This is *weight tying* (or *tied embeddings*), and it is a per-model choice:

```text
  front door (embedding)   read row i           → vector for token i
  back door  (LM head)     multiply by columns  → score for every token
                           (the columns are the same rows, transposed)
```

```text
  GPT-2, Gemma, Llama-3.2-1B/3B   TIED    — one shared matrix, used both ways
  Llama-2, Llama-3-8B             UNTIED  — two separate matrices, trained apart
```

**Why tie.** Two reasons. It halves the parameters spent on the two ends, which
matters because that matrix is large (§5). And there is a pleasing symmetry: the
vector that *means* "cat" on the way in is the same vector that *scores* "cat" on
the way out, so the model learns one consistent representation per token instead
of two. **Why untie.** Larger models can afford two separate matrices and
sometimes train slightly better when the input encoding and the output scoring
are allowed to specialize. Take tying as a common default, not a rule — and a
detail worth checking per model when you count parameters.

## 5. Why the LM head is a surprisingly big matmul

The vocabulary is large, so `[d_model × vocab]` is one of the *biggest single
matrices* in the whole model — often bigger than any one layer's weights:

```text
  GPT-2          768 × 50257    ≈  38.6M   (tied: shared with the embedding)
  Llama-2-7B    4096 × 32000    ≈ 131M
  Llama-3-8B    4096 × 128256   ≈ 525M     <- one matrix, untied, so the two
                                              ends together are ~1.05B params:
                                              over an eighth of an 8B model
```

That last row is the one that surprised me: a model nicknamed "8B" spends more
than a billion of those parameters just on its two ends, because Llama-3 grew the
vocabulary to 128k and keeps the two matrices separate. The lesson for later
chapters: when you tally where a model's bytes live (M15, quantization), the LM
head and embedding are not rounding error.

It is also a real *compute* cost at the back door. In decode, the LM head is a
matrix × vector against the full `[d_model × vocab]` matrix **every single
token** — so like the rest of decode at batch size 1 it is memory-bound (read the
whole big matrix, use it once,
[end-to-end §7](../02_cuda_software_stack/02_end_to_end_inference.md#7-stage-3--decode-the-autoregressive-loop)),
and with a 128k vocabulary that is a non-trivial slice of each token's work.

## 6. Shapes and hardware

The two ends split across the GPU's units the same way the layer did
(`end-to-end §5`):

```text
  embedding lookup    gather (index → rows)   -> load/store units   (no math)
  final RMSNorm       normalize one vector    -> FP lanes           (glue)
  LM head             matmul d_model × vocab   -> Tensor Cores        (a GEMM)
```

Prefill computes the LM head only at the last position; decode computes it at the
one position there is. The embedding is the same cheap gather in both phases — it
never touches the Tensor Cores, because looking up a row is not arithmetic.

## 7. What to carry forward

```text
embedding = a [vocab × d_model] lookup table (§2)   -> M9, the block's front door
LM head   = a [d_model × vocab] GEMM → logits (§3)   -> M9 / M11, the back door
weight tying — often one matrix, used twice (§4)    -> check per model in M15
the LM head is one of the largest matrices (§5)     -> M15, a real quant target
logits feed sampling, not probabilities yet (§3)    -> the next section, sampling
```

The one sentence to keep: **the model's two ends are mirror images that often
share one `[vocab × d_model]` matrix — the embedding reads a row per token ID
(a cheap gather, no math), and the LM head multiplies the final vector back
against that matrix to produce logits (a large GEMM) — bracketing the layer
stack that attention and the MLP fill in.**
