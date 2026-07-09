# SPRINT.md — the eigh sprint

Operational state for the active build. Read this right after `CLAUDE.md`;
[the agentic turn](ROADMAP.md#25-the-agentic-turn-july-2026) is the why. This
file is the single source of truth for "what happens next" — I update it as
work lands, from whichever machine I'm on.

## The target

GPU MODE's live competition: **batched real symmetric eigendecomposition**
(`eigh`) — fp32 in/out, batched 512×512 up to 4096×4096 symmetric matrices,
**B200**, deadline **2026-07-15**. Low-bit internals (FP16/FP8/NVFP4) are
explicitly allowed by the task. Ranking: geometric-mean runtime among
correctness-passing submissions.

Success bar: a correctness-passing, non-trivial submission with full traces and
an honest writeup. Placement is gravy. Acceptable fallback: a passing
`vectoradd_v2`, a working agent + harness system, and the judge dissection —
still a real week's result.

## The ladder

```text
[ ] 0  Machine setup          uv sync · .env from .env.example · modal token new
[ ] 1  Harness core           Modal runner executing the vendored official GPU MODE
                              judge (eval.py / reference.py / utils.py per problem);
                              common/ wraps it: seal + prediction gate + cost ledger
[ ] 2  Agent loop v0          agent/loop.py — bare Anthropic API, model as a CLI flag
                              (default claude-sonnet-5), $5 hard budget cap, max 6
                              repair attempts, JSONL traces committed to git
[ ] 3  vectoradd_v2           agent writes Triton + CUDA (load_inline) fp16 kernels →
                              pass vendored judge on Modal T4 → prediction file →
                              reveal → report → SUBMIT to the practice leaderboard
[ ] 4  eigh judge dissection  read their eval.py line by line; dissection note +
                              prediction file (approach + expected perf vs
                              torch.linalg.eigh on B200)
[ ] 5  eigh attempts          agent iterates on Modal B200 bursts, budget-capped
[ ] 6  Submit + write up      best passing candidate before 2026-07-15; honest report
```

## Decisions (locked)

- **The agent writes kernels; it never writes the judge.** `common/` is
  human-written; the correctness core is GPU MODE's own vendored `eval.py` —
  separation of judge and defendant.
- **Predict before measuring, mechanically enforced**: benchmark numbers stay
  sealed until a prediction file exists; the results JSON embeds the prediction
  file's hash + timestamp, proving prediction preceded measurement.
- **Every attempt is accounted**: prompts, tokens, dollars, outcome — JSONL
  traces committed to git as research data.
- **Modal is the GPU backend**: T4 for vectoradd, B200 bursts for eigh
  (~$30 credits available). GPU MODE runner submissions are free.
- **Submission format** (platform fact): a single `.py` exposing
  `custom_kernel(data)`; output tensors are pre-allocated by the caller;
  CUDA C++ goes in via `torch.utils.cpp_extension.load_inline`.
- CuTe DSL is out of scope for this sprint.

## New-machine setup

```text
git clone https://github.com/ishgirwan/exploring-llm-inference-from-scratch.git
cd exploring-llm-inference-from-scratch
uv sync
cp .env.example .env        # paste ANTHROPIC_API_KEY into .env — never commit it
uv run modal token new      # browser auth; token lands in ~/.modal.toml
```

## Status

- Setup scaffolding done: uv deps (modal, anthropic, python-dotenv) locked,
  `.env.example` in place, `.env` gitignored.
- Modal token + API key: confirm per machine (step 0 of the ladder).
- **Next action: ladder step 1 — the Modal runner for the vectoradd judge.**
