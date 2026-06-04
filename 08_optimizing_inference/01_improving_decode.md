# Improving decode: escaping the memory wall

[Batching §1](../06_batching/01_batching.md#1-the-problem-batching-solves) showed
why decode — the one-token-at-a-time generation loop — is memory-bound: each step
drags the model's whole weight set across the VRAM bus to produce a single token,
and the Tensor Cores (the GPU's matrix-multiply units) sit idle most of the step,
waiting for the next slab of weights. Batching fixes that by running many
sequences' decode steps together so one weight load feeds all of them.

But batching needs *other requests to share the weight load with*. What about the
case where there aren't any? One user, one local model. Or an agent loop where a
single request flows through the model step after step, each token depending on
the one before, with nothing else in flight. Here you can't borrow other streams'
work — so is decode just stuck memory-bound forever?

No. "Memory-bound" is not a dead end; it's a description of *where the slack is*.
This doc is the map of every other way out, organized around the single quantity
decode is actually bound by. The mechanics of the batching-and-serving routes live
in [Chapter 6](../06_batching/01_batching.md) already; this doc places them on the
map and then goes deep on the routes that have no home yet — speculative decoding,
multi-token prediction, quantization as a decode lever, and KV-cache shrinking.

Like the rest of these foundation docs it's a map; read it as the bridge into the
optimization topics M15–M19, not as theory to finish before M0.

Prerequisites:
[Batching §1–§4](../06_batching/01_batching.md#1-the-problem-batching-solves),
[Attention §7–§8](../05_attention_and_kv_cache/01_attention.md#7-the-kv-cache-what-it-stores-and-why-its-shaped-that-way),
[MLP §7](../05_attention_and_kv_cache/02_mlp_feedforward.md#7-why-mlp-decode-is-weight-memory-bound--the-first-wall),
and [Numerical types — integer types](../03_numerical_types/01_floating_point.md#integer-types-int8-int4).
Next: M15–M19 in the [Roadmap](../ROADMAP.md) (and M11–M14 for the serving side).

## 1. The one quantity decode is bound by

When a kernel is memory-bound, its runtime is set by how many bytes it moves, not
how much math it does
([execution model §15](../01_hardware_fundamentals/03_gpu_model.md#15-why-llm-inference-is-often-memory-bound)).
For a decode step the bytes are dominated by the weights (and, in attention, the
KV cache — the stored keys and values of every past token,
[attention §7](../05_attention_and_kv_cache/01_attention.md#7-the-kv-cache-what-it-stores-and-why-its-shaped-that-way)).
So the whole performance story collapses to one ratio:

```text
                 bytes pulled from VRAM
  decode cost ≈  ───────────────────────
                 useful output tokens
```

Every decode optimization lowers that ratio, and there are only two places to push:

```text
  ┌─ push the DENOMINATOR up ─ get more useful tokens out of each weight load
  │     across users:     batching            (Chapter 6 — needs other requests)
  │     within one stream: speculative decoding / multi-token prediction (§3)
  │
  └─ push the NUMERATOR down ─ move fewer bytes per token
        smaller weights:  quantization        (§4)
        smaller KV cache: GQA / MQA / MLA, KV quant (§4)
```

Both moves are the *same move* on the roofline
([batching §3](../06_batching/01_batching.md#3-the-roofline-view-climbing-toward-the-ridge)).
Shift lens from that cost ratio to **arithmetic intensity** — FLOPs per byte, the
quantity the roofline plots — and both *raise* it until the cores stop starving:
more useful work per load lifts the FLOPs on top, fewer bytes per token shrinks the
bytes underneath. (Bytes sat in the numerator of the cost ratio and sit in the
denominator here — same fact seen upside-down, because cutting bytes moved *is*
raising arithmetic intensity, not its opposite.) Hold that one picture and the rest
of this doc just fills in the two branches.

## 2. The route Chapter 6 already covers: more tokens per load, across users

The first way to get more tokens out of one weight load is to have more sequences
that all need that weight *this step*, and run them together. That is batching, and
its full mechanics — why intensity climbs as ≈ B (the batch size), static vs.
continuous batching, and why KV-cache VRAM caps how big B can get — are
[batching §2](../06_batching/01_batching.md#2-batching-reuses-each-weight-load--intensity-climbs-with-b)
through
[batching §7](../06_batching/01_batching.md#7-the-constraint-kv-cache-vram-caps-the-batch).
I won't restate them; the point here is just *where batching sits on the map*: it's
the denominator route, and it needs other requests to fill the batch.

That's exactly the assumption the single-stream case breaks. So the rest of the
denominator route is the interesting one: getting more tokens out of one weight
load **without** other users — from within the single sequence itself.

## 3. More tokens per load, within one stream: speculative decoding

Here is the trick, and it's the prettiest idea in inference. Memory-bound decode
leaves the compute units almost entirely idle — that idleness is *free arithmetic
capacity*. Speculative decoding spends it like this: cheaply **guess** the next few
tokens, then **verify** all of them in a single forward pass of the real model.

- The **guess** comes from something fast: a small "draft" model (a cheaper model
  predicting the same vocabulary), or extra heads bolted onto the big model itself
  (see §3.1). Say it proposes `K` candidate tokens.
- The **verify** step runs the big model **once** over all `K` candidates at the
  same time. That single pass loads the big model's weights *once* and checks `K`
  tokens — structurally the very same matrix×matrix shape batching produces, except
  the "batch" is `K` guessed tokens from *one* sequence instead of tokens from `K`
  different users.

```text
  plain decode (K=1):   load weights ─► 1 token ─► load weights ─► 1 token ─► ...
                        (one sequential step per token)

  speculative (K=4):    draft cheaply ─► [t1 t2 t3 t4] guessed
                        load weights ONCE ─► verify all 4 together
                        keep the longest correct prefix, e.g. t1 t2 t3 ✓ t4 ✗
                        ─► 3 tokens accepted from one big-model weight load
```

The big model accepts the longest prefix of the guess that matches what it would
have produced anyway, and rejects the rest. The fraction it keeps is the
**acceptance rate**; the better the draft predicts the target, the higher it runs
and the more tokens you harvest per weight load. Crucially this is not an
approximation — with the standard verification (speculative sampling), the output
has **the same probability distribution** as decoding from the big model normally
(for greedy decoding, the *identical* tokens). You trade the idle compute for fewer
sequential steps; you do not trade away quality.

This is why speculative decoding only helps a *memory-bound* loop: the free compute
it spends exists precisely because decode wasn't using it. Run it on an
already-compute-bound workload and there's nothing to spend — which is exactly why
it targets the single-stream, batch-1 case that batching can't reach.

### 3.1 Where the cheap guess comes from

The draft is the whole game — a bad guesser means a low acceptance rate and no
speedup. Three ways to produce one:

```text
  separate draft model   a small model of the same vocabulary runs ahead and
                         proposes tokens; the big model verifies (classic form)

  self-drafting heads    extra prediction heads on the big model itself guess the
                         next few tokens, so there's no second model to host
                         (Medusa = a few extra heads + tree-shaped verification;
                          EAGLE = drafting on the model's hidden states for higher
                          acceptance)

  multi-token prediction trained-in: the model learns during training to predict
  (MTP)                  tokens t+1, t+2, … in one shot (e.g. DeepSeek-V3). Those
                         native predictions are a built-in, almost-free draft
```

The distinction worth holding: **MTP is a training-time architecture choice** (the
model is built to emit several tokens at once), while **speculative decoding is the
inference-time draft/verify mechanism** that consumes such guesses. MTP is one way
to feed speculation cheaply; speculation is the loop that turns a guess into
verified output. The roadmap builds a draft/verify loop and measures acceptance
rate at M19.

## 4. Fewer bytes per token: smaller weights, smaller KV cache

The other branch doesn't add work — it shrinks the bill. Since a memory-bound step
costs `bytes / bandwidth`, moving fewer bytes is a near-linear speedup.

**Quantization — smaller weights.** Store each weight in fewer bits: fp16 (2 bytes)
→ int8 (1 byte) → int4 (½ byte), or fp8. Chapter 3 covers the formats and how a
low-bit integer reconstructs an approximate real value through a stored *scale* and
*zero-point*
([numerical types — integer types](../03_numerical_types/01_floating_point.md#integer-types-int8-int4)).
The reason it belongs *here*, as a decode lever, is the bytes link Chapter 3
doesn't draw: weight-only int4 streams ~4× fewer weight bytes per step, so the
weight-bound part of decode runs up to ~4× faster — not because the math got
cheaper (the matmul still accumulates in higher precision,
[accumulator precision](../03_numerical_types/01_floating_point.md#accumulator-precision)),
but because the *thing you were waiting on* — bytes off the VRAM bus — got smaller.
That's M15: a quantized linear layer with a fused dequantize-then-matmul kernel.

**Shrinking the KV cache.** Quantization shrinks the weights; it doesn't touch the
*other* memory cost of decode, the KV cache, which batching also can't help because
each sequence has its own
([batching §4](../06_batching/01_batching.md#4-the-subtlety-batching-helps-the-weights-not-attention)).
You attack that wall by storing fewer KV bytes per token. The architectural levers
share key/value vectors across the model's attention heads instead of giving each
its own:

```text
  MHA   multi-head attention      every query head has its own K and V  (biggest cache)
  GQA   grouped-query attention   heads share K/V in groups             (middle ground)
  MQA   multi-query attention     all query heads share ONE K/V         (smallest of the three)
  MLA   multi-head latent attn.   K/V compressed into a small latent,
                                  decompressed on the fly (DeepSeek)     (smallest cache)
```

Fewer K/V vectors stored means fewer KV bytes streamed every decode step — directly
lowering the numerator of §1's ratio for the attention part. (You can also quantize
the KV cache itself, int8/fp8, for the same reason.) These are model-architecture
choices made at training time, so they're a lever you *select a model for* rather
than bolt on at inference — but they're why a modern model's decode is far less
KV-bound than an old all-MHA one.

## 5. The agent case: prefix caching wraps the shared part

The multi-agent worry — a request threading through several agent steps, or an
agent loop re-sending its whole history each turn — has one more lever specific to
it. Those calls re-send the *same* leading tokens every time: a shared system
prompt, the same tool definitions, the conversation so far. Their KV cache for that
shared prefix is identical, so it can be computed **once** and reused across every
call that shares it — *prefix caching* / *RadixAttention*
([batching §4](../06_batching/01_batching.md#4-the-subtlety-batching-helps-the-weights-not-attention),
[end-to-end §9](../02_cuda_software_stack/02_end_to_end_inference.md#9-the-serving-engine-who-drives-the-loop)).
On the map it's the across-users denominator route again (amortizing work across
calls), but applied to attention's KV cache rather than the weights, and it's the
single biggest reason agent-style workloads behave differently from one-shot
chat. That's exactly what M13 (SGLang) and M14 (the agent workload) measure — and
its larger payoff is actually on the *prefill* side, which is
[the next doc](02_improving_prefill.md).

## 6. Which lever for which regime

The levers aren't ranked; they're matched to *which bottleneck you're actually in*,
and that's set by your concurrency. Two engines sit at the opposite ends and make
the contrast concrete:

```text
  llama.cpp        optimized for ONE user, local, latency-first
                   headline levers: aggressive quantization (GGUF int4/int5) +
                   speculative decoding — the within-one-stream routes (§3, §4),
                   because in that regime there are no other requests to batch with

  vLLM / SGLang    built for MANY concurrent requests, throughput-first
                   headline levers: continuous batching, PagedAttention, prefix
                   caching — the across-users routes (Chapter 6), because the GPU
                   nearly always has other streams in flight
```

The honest qualifier, in the spirit of
[end-to-end §9](../02_cuda_software_stack/02_end_to_end_inference.md#9-the-serving-engine-who-drives-the-loop):
llama.cpp *can* batch — it has parallel slots and continuous batching of its own.
The split isn't "can vs. can't," it's what each is *tuned for*: the single-user
local regime leans on the within-stream levers because that's the regime where the
across-users ones have nothing to work with. Same model, same GPU — the right lever
is a function of how many streams share the machine.

Two more that don't change the ratio but unblock it, noted so they're not mistaken
for omissions:

- **FlashDecoding** — for one long-context sequence, decode-attention can't fill the
  GPU because there's only one new token's worth of work; FlashDecoding splits the
  KV-length dimension across the GPU's many cores so a single sequence still keeps
  them busy. It's the attention-side answer to batch-1 underutilization, built on
  FlashAttention (M16).
- **CUDA graphs** — each decode step launches dozens of tiny kernels, and the CPU
  cost of *launching* them is a real fraction of such a short step. Capturing the
  step into one replayable graph removes that launch overhead
  ([end-to-end §9](../02_cuda_software_stack/02_end_to_end_inference.md#9-the-serving-engine-who-drives-the-loop);
  M18).

## 7. What to carry forward

```text
decode is bound by bytes-per-useful-token (§1)        -> the frame for M15–M19
batching = more tokens/load across users (§2)         -> Chapter 6, M12 / M12.5
speculative decoding / MTP = more tokens/load,        -> M19, build draft/verify,
  within one stream, spending idle compute (§3)          measure acceptance rate
quantization = fewer weight bytes/token (§4)          -> M15, fused dequant+matmul
GQA / MQA / MLA + KV quant = fewer KV bytes/token (§4) -> a model-choice lever; M16–M17
prefix caching amortizes the shared prefix (§5)       -> M13 (SGLang), M14 (agents)
right lever depends on the concurrency regime (§6)    -> M12–M14 vs. local/llama.cpp
```

The one sentence to keep: **memory-bound decode means the compute units are idle,
so every speedup either gets more useful tokens out of each weight load — batching
across users, or speculative decoding / multi-token prediction within one stream —
or moves fewer bytes per token — quantizing the weights, shrinking the KV cache —
and which lever fits is decided by how many request streams are sharing the GPU.**
