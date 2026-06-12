# Scaling past one GPU

The first two docs in this chapter optimize within one GPU. This one covers the
move the biggest models force: splitting inference across several. It stays at
map altitude — the four ways to split a model, what each one costs in
communication, and why the interconnect between the GPUs becomes the new
bottleneck to reason about — because the build modules (M20, M31, M32) are where
the numbers get made.

Prerequisites: [Improving decode](01_improving_decode.md) for the bytes-per-token
framing, and [Chapter 5's MoE section](../05_attention_and_kv_cache/06_moe.md)
for what an expert is.
Next: M20, M31 and M32 in the [Roadmap](../../ROADMAP.md).

## 1. Why one GPU runs out

Two separate limits, and it matters which one is binding:

```text
  CAPACITY    the model + KV cache don't fit
              Llama-70B at fp16 = 140 GB of weights  >  80 GB (one H100)
              DeepSeek-V3 at fp8 ≈ 671 GB            >  8 × 80 GB, even
              (MoE makes this worse: ALL experts must be resident —
               total params, not active, set the VRAM bill)

  SPEED       it fits, but one GPU's bandwidth caps tokens/sec
              decode speed ≈ bandwidth / bytes-per-token (the ratio from
              the decode doc) — more GPUs streaming slices of the weights
              in parallel is more aggregate bandwidth
```

Capacity *forces* multi-GPU; speed merely *invites* it. The price of accepting
either invitation is communication, and the rest of this doc is an accounting
of that price across the four ways to split.

## 2. The four splits

```text
  TENSOR  parallelism (TP)     split each weight MATRIX across GPUs
                               -> all GPUs cooperate on every layer
  PIPELINE parallelism (PP)    split the model by LAYERS into stages
                               -> GPU 1 holds layers 0-19, GPU 2 holds 20-39...
  EXPERT  parallelism (EP)     split an MoE's EXPERTS across GPUs
                               -> each GPU owns a subset of experts
  DATA    parallelism (DP)     don't split the model at all — full REPLICAS,
                               and a router spreads requests across them
```

Real deployments stack them (TP within a node, PP or EP across nodes, DP over
the whole thing), but each is easiest understood alone.

## 3. Tensor parallelism: split the matrices, pay per layer

TP slices the big matmuls. In the standard scheme (from the Megatron-LM
training system, reused unchanged for inference), the MLP's up-projection is
split by *columns* — each of `t` GPUs holds `1/t` of the columns and computes a
slice of the hidden vector — and the down-projection by *rows*, so each GPU's
slice yields a partial sum of the output, and the partial sums are added
together across GPUs. That final add is an **all-reduce**: a collective
operation where every GPU contributes its array and every GPU receives the
elementwise sum.

```text
  GPU 0:  x · W_up[:, 0:h/2] -> activate -> · W_down[0:h/2, :] -> partial y₀
  GPU 1:  x · W_up[:, h/2:h] -> activate -> · W_down[h/2:h, :] -> partial y₁
                                          all-reduce: y = y₀ + y₁  (every GPU
                                          now has y, and can start the next layer)
```

Attention splits the same way (heads are naturally parallel — each GPU takes a
subset of heads, and the KV cache for those heads lives with them), so a
transformer layer costs **two all-reduces in the forward pass: one after
attention, one after the MLP.** That's the bill: every layer, every step,
~`2 · d_model` values per token exchanged, *on the critical path* — the next
layer cannot start until the all-reduce lands.

Which is why TP lives and dies on the interconnect:

```text
  NVLink (H100, within a node)     900 GB/s per GPU, aggregate
  PCIe Gen5 x16                    ~64 GB/s per direction
  NVLink (Blackwell)               1.8 TB/s per GPU
```

Over NVLink the per-layer sync is small against decode's weight streaming; over
PCIe it stacks ~14× more latency per hop and can dominate the step. The rule it
produces: **TP within a node, something coarser across nodes.** And the payoff
when it works: each GPU streams only `1/t` of the weights per step, so TP is
the split that actually divides decode's bytes-per-token — aggregate bandwidth
attacking the memory wall directly.

## 4. Pipeline parallelism: split by layers, pay per boundary

PP gives each GPU a contiguous block of layers; a token's forward pass visits
the stages in order, handing over one `d_model`-wide activation vector per
boundary. The communication is tiny and point-to-point — no collectives — which
is why PP is the split that tolerates slow links and crosses nodes.

Its cost is different in kind: **a stage only works while a token is inside
it.** With a single decode stream, stage 2 idles while stage 1 computes and
vice versa — the *pipeline bubble*. Keeping all stages busy needs several
requests in flight (the serving engine's batch, interleaved across stages), so
PP adds capacity and throughput but does nothing for one request's latency:
every token still walks through every layer sequentially, now with hop
latencies added.

```text
  TP   all GPUs work on the SAME token simultaneously   -> divides per-token time
  PP   GPUs work on DIFFERENT tokens simultaneously     -> divides nothing
                                                           per-token; raises ceiling
```

## 5. Expert parallelism: split the experts, pay in all-to-all

For MoE models, the natural unit is the expert: with 256 experts over 8 GPUs,
each GPU holds 32. But [the MoE section](../05_attention_and_kv_cache/06_moe.md)'s
router picks experts per *token*, and the experts a token wants usually live on
other GPUs. So every MoE layer does an **all-to-all**: each GPU sends each of
its tokens' vectors to the GPUs owning their chosen experts, the experts run,
and a second all-to-all brings the outputs home.

```text
  per MoE layer:  all-to-all (dispatch) -> expert FFNs -> all-to-all (combine)
```

An all-to-all moves more data than TP's all-reduce (whole token vectors, `k`
copies each, to scattered destinations) and its cost depends on routing luck —
balanced routing spreads the traffic, a hot expert concentrates it. This is
the communication pattern DeepSeek-V3-class serving lives inside, and it's why
EP deployments obsess over load balancing and overlap tricks.

## 6. Data parallelism, and what the collectives actually cost

DP is the degenerate split: full model copies, a load balancer, zero
communication. It's the right answer whenever the model *fits* and the goal is
more requests per second — and a reminder that the exotic splits are only for
models that don't.

One level down on the collectives themselves, then re-seal: the library running
them on NVIDIA is **NCCL** (NVIDIA Collective Communications Library), and the
classic all-reduce algorithm is the **ring** — GPUs pass slices around a circle,
summing as they go, so each GPU transmits about `2·(N−1)/N` of the buffer
(≈ 2× the data, regardless of GPU count `N`). The numbers to keep are just:
all-reduce ≈ 2× the buffer through each GPU's link, all-to-all ∝ tokens ×
their destinations, both hideable behind compute only if the kernel work is
long enough to cover them — overlap engineering, not magic.

## 7. Splitting by phase: disaggregation, revisited

[The prefill doc](02_improving_prefill.md) introduced prefill/decode
disaggregation as a lever: run compute-bound prefill and memory-bound decode on
separate GPU pools, ship the finished KV cache from one to the other. Seen from
this doc, it's the fifth split — by *phase* rather than by weights — and the
thing it pays in is **KV-cache transfer**: gigabytes per long request (the
formula in [Chapter 5's attention doc](../05_attention_and_kv_cache/01_attention.md))
moving between pools, which is why disaggregated systems are built around fast
KV movement and caching layers.

It's also no longer speculative as *industry practice*, even though it stays a
maybe-module (M32) for this repo: Mooncake — the serving platform behind the
Kimi service — is a disaggregated, KV-cache-centric architecture in production,
vLLM ships prefill/decode disaggregation support, and NVIDIA's Dynamo framework
(announced at GTC 2025) is built around disaggregated serving as the default
shape of large-scale inference.

## 8. Choosing the split

```text
  model fits on one GPU, want throughput    -> DP replicas, nothing fancier
  doesn't fit, fast links available         -> TP within the node
  doesn't fit in a node                     -> PP (or TP+PP) across nodes
  MoE with many experts                     -> EP for the experts,
                                               TP for attention is common
  decode latency is the product             -> TP (the only split that divides
                                               per-token time)
  prefill interference at scale             -> disaggregation (M32 territory)
```

M20 makes the first cut of this real: the same model TP=1 vs TP=2, with the
all-reduces visible in an Nsight trace, on NVLink vs PCIe.

## 9. Further reading

- **[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)**
  (Shoeybi et al., 2019) — the TP scheme of §3 (column/row splits, the two
  all-reduces per layer); written for training, used verbatim by inference engines.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)**
  (DeepSeek-AI, 2024) — the deployment sections are the best public description
  of expert parallelism (§5) run at production scale.
- **[Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)**
  (Qin et al., 2024; USENIX FAST '25) — disaggregation (§7) as a production
  system, organized entirely around moving and caching KV.
- **[NVIDIA Dynamo](https://developer.nvidia.com/dynamo)** — the open-source
  disaggregated-serving framework; its architecture docs are a current map of
  §7's moving parts (KV-aware routing, KV transfer, pool scheduling).

## 10. What to carry forward

```text
capacity vs speed as the reason to split (§1)   -> M20, framing the benchmark
TP = 2 all-reduces/layer, NVLink-bound (§3)     -> M20, what the NCCL trace shows
PP raises ceiling, not per-token speed (§4)     -> M20+, reading engine configs
EP + all-to-all for MoE (§5)                    -> M31, the multi-GPU half
all-reduce ≈ 2× buffer per link (§6)            -> M20, sanity-checking traces
disaggregation = split by phase (§7)            -> M32, if it happens
```

The one sentence to keep: **multi-GPU inference is choosing what to split —
matrices (TP), layers (PP), experts (EP), replicas (DP), or phases
(disaggregation) — and each choice converts VRAM pressure into a specific
communication pattern on the interconnect, which is the new wall the
profile has to account for.**
