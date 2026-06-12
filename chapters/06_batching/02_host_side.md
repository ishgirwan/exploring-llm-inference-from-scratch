# The host side: the serving stack above the kernels

Everything in this repo so far happens on the GPU. This doc looks up the stack
at the part of a serving engine that runs on the **CPU** — the API server,
tokenizer, scheduler, sampler glue, and detokenizer that wrap every forward
pass — because it can quietly become the bottleneck that no kernel optimization
can fix. It also covers **structured decoding** (forcing output to match a JSON
schema or grammar), the most common feature request that lives exactly on this
CPU/GPU boundary.

This sits in the batching chapter because the host work scales with the thing
batching maximizes: requests in flight. A scheduler juggling 200 sequences has
200 sequences' worth of bookkeeping to finish before the GPU needs its next
instruction.

Prerequisites: [Batching: the throughput lever](01_batching.md), especially its
continuous-batching section, and the request lifecycle from
[Chapter 2's end-to-end doc](../02_cuda_software_stack/02_end_to_end_inference.md).
Next: M11–M14 in the [Roadmap](../../ROADMAP.md), where engines get benchmarked —
with one eye on the host.

## 1. The budget: a decode step is a few milliseconds

A decode step — one forward pass of the batch, one token per sequence — takes
single-digit to low-tens of milliseconds on a modern GPU, depending on model
and batch. That number is the host's whole budget: while the GPU executes step
`N`, the CPU must finish *everything* needed to launch step `N+1` —

```text
  while the GPU runs step N, the CPU must:
    - process arrivals/finishes, decide step N+1's batch   (scheduling)
    - append step N's tokens, build step N+1's input
      tensors and block tables                              (bookkeeping)
    - apply sampling params, pick step N's tokens           (sampling glue)
    - detokenize new tokens, push stream chunks             (output path)

  done in < GPU step time   ->  the GPU never waits  (host is hidden)
  done in > GPU step time   ->  the GPU idles between launches: host-bound
```

When the host loses that race, the symptom is unmistakable in a timeline trace:
**gaps between the decode steps**. The kernels are fast; the engine isn't. And
no amount of kernel work closes a gap that exists because the GPU was waiting
for Python.

## 2. A request's path through the engine

The full round trip, with the GPU loop as just one box in the middle:

```text
  HTTP request ("write me a haiku", sampling params)
      |
      v
  API server          parse, validate, assign request id          [CPU]
      |
  tokenizer           text -> token ids                           [CPU]
      |
  scheduler           admit into the running batch                [CPU]
      |          (continuous batching — the batching doc's refill-every-step)
      v
  ┌─ the step loop ─────────────────────────────────────────────────────┐
  │   forward pass (prefill or decode)                          [GPU]   │
  │   sample next tokens from the logits                        [GPU]   │
  │   detokenize the new token ids -> text fragments            [CPU]   │
  │   stream chunks to clients (SSE — server-sent events,       [CPU]   │
  │      the one-way HTTP streaming protocol chat UIs use)              │
  └──────────────────────────────────────────────────── repeat ─────────┘
      |
      v
  finish: sequence hits a stop token or length cap; scheduler frees its
  KV blocks and admits a waiting request into the freed space
```

Every `[CPU]` line above is per-step, per-request work — which is why host time
grows with batch size even as the GPU's per-token cost shrinks.

## 3. Where the milliseconds go

The classic host costs, roughly in the order engines hit them:

- **Python itself.** Most serving engines are Python at the top. Object
  creation, list shuffling, and the GIL (global interpreter lock — CPython's
  one-thread-at-a-time execution lock) put a ceiling on how much per-step
  bookkeeping a Python loop can do in a few milliseconds. Engines respond by
  moving hot paths to C++/Rust or off the critical path.
- **Kernel-launch overhead.** Each kernel launch costs the CPU microseconds;
  a decode step is hundreds of small kernels; at small batch the launches can
  rival the kernels. This is the cost CUDA graphs erase (M18) — capture the
  whole step once, replay it as one unit.
- **Tokenization at the edges.** Tokenizing a long prompt is real CPU work, and
  detokenization runs every step for every sequence. Cheap individually,
  multiplied by everything.
- **Scheduling and block bookkeeping.** Continuous batching means re-deciding
  the batch every step; paged KV (the batching doc's VRAM cap made concrete in
  M17) means maintaining block tables for every sequence, every step.

## 4. Hiding the host: the async engine

The structural fix mirrors what GPUs do with memory latency: **overlap**. The
engine prepares step `N+1` on the CPU *while* the GPU executes step `N`, so
host time hides behind GPU time instead of adding to it — this is the headline
design change of vLLM's V1 engine rewrite, which moved scheduling and
bookkeeping off the GPU's critical path. The decode loop becomes a pipeline of
two workers (CPU and GPU) rather than an alternation, and the host only shows
up in the trace if its work outgrows a whole GPU step.

The carry-forward for benchmarking: an engine's quality is as much this
overlap as its kernels — two engines with identical kernels can differ
2× at high request rates purely on host architecture.

## 5. Structured decoding: grammar on the critical path

**Structured decoding** (also *constrained* or *guided* decoding) guarantees
the output matches a format — JSON with a given schema, a regex, a context-free
grammar — which agent and tool-calling workloads demand: a tool call that's
almost-JSON is worthless. The mechanism is logit masking:

```text
  every step:
    grammar engine:  given what's been generated, which tokens are
                     legal next?                                   [CPU-ish]
    mask:            set illegal tokens' logits to −∞              [GPU]
    sample:          as normal — but only legal tokens survive
```

The catch is in the first line: "which tokens are legal" must be answered over
a ~100K-token vocabulary, *every step*, *per constrained sequence*, inside the
§1 budget. Naively walking a grammar per token blows it instantly. The
production engines compile their way out — **Outlines** turns regex-class
constraints into a finite-state machine (FSM: a graph of states where each
token is a transition, so "what's legal" is a precomputed table lookup);
**XGrammar** (the default backend in vLLM and SGLang) extends this to full
context-free grammars with the expensive parts precompiled and overlapped with
the GPU step; **llguidance** is the same race run by Guidance/Microsoft. The
elegant extra: when the grammar leaves only one legal continuation (closing
`"}`, a fixed key name), the engine can emit those tokens *without running the
model at all* — SGLang's **jump-forward decoding** — making constrained output
occasionally *faster* than free generation.

The reason this doc covers it at all: structured decoding is the clearest
example of host work that lands directly on the token loop. Done naively it
*is* the bottleneck; done well it's invisible. That's the whole host-side story
in one feature.

## 6. What this means for measuring engines

For the M11–M14 benchmarks, the host earns three habits:

```text
  1. look for gaps      in the Nsight timeline between decode steps —
                        gap = host-bound, and kernel work won't fix it
  2. distrust util %    "GPU utilization 95%" counts a kernel running,
                        not the gaps' cause; step-time variance is the tell
  3. stress the host    high request rate + many short sequences + streaming
                        + structured output is the host-heavy workload;
                        long single prompts barely touch it
```

## 7. Further reading

- **[vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)**
  (vLLM team, 2025) — the engine rewrite of §4, argued almost entirely in terms
  of getting CPU overhead off the critical path.
- **[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)**
  (Zheng et al., 2023) — the compressed-FSM constrained decoding and
  jump-forward trick of §5 (alongside RadixAttention, which Chapter 8 covers).
- **[XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100)**
  (Dong et al., 2024) — how full context-free-grammar constraints get compiled
  and overlapped to near-zero per-token overhead (§5).
- **[Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702)**
  (Willard & Louf, 2023) — the Outlines paper: §5's regex-to-FSM construction,
  short and readable.

## 8. What to carry forward

```text
the per-step CPU budget, and gaps as the tell (§1)  -> M11-M14, reading traces
the request path above the kernels (§2)             -> M12.5, the toy scheduler
launch overhead -> CUDA graphs (§3)                 -> M18
async overlap as engine architecture (§4)           -> M12/M13, explaining engine deltas
logit masking + compiled grammars (§5)              -> M14, agent workloads use this
```

The one sentence to keep: **above the kernels sits a per-step CPU loop —
scheduling, bookkeeping, detokenizing, grammar masking — that must finish
inside the GPU's step time or the GPU idles; engines win or lose on hiding that
loop, and a timeline with gaps between decode steps is the signature of one
that lost.**
