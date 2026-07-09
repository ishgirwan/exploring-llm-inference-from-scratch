# Further reading

The papers, repos, and platforms behind
[the agentic turn](ROADMAP.md#25-the-agentic-turn-july-2026) — the ones I keep
going back to. [`FURTHER_LISTENING.md`](FURTHER_LISTENING.md) is the audio
companion; this is the reading list. Each entry says why it earns its place.

## The field

- [Towards Automated Kernel Generation in the Era of LLMs](https://arxiv.org/pdf/2601.15727)
  — the survey. Its taxonomy maps the whole space, and its open-problems section
  (evaluation robustness, reward hacking, cost of agentic loops, low-bit and
  out-of-distribution kernels) is effectively this project's justification.
- [awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation)
  — curated field map: models, agents, datasets, benchmarks, updated as the
  field moves.

## Competitions

- [GPU MODE KernelBot](https://github.com/gpu-mode/kernelbot) — the competition
  platform: submit a kernel, it verifies correctness and benchmarks runtime on
  real hardware.
- [reference-kernels](https://github.com/gpu-mode/reference-kernels) — the
  official problem sets, including each problem's `eval.py` judge and reference
  implementation; the judges are worth reading as carefully as the problems.
- [gpumode.com](https://www.gpumode.com) — leaderboards and news.
- [kernelbench.com](https://kernelbench.com/) — agentic kernel benchmark
  results: frontier models × coding harnesses on hard kernels, graded against
  hardware rooflines.

## Evaluation rigor

- [Reward Hacking Benchmark](https://arxiv.org/abs/2605.02964) — measures how
  often LLM agents exploit naturalistic shortcuts (skipping verification,
  tampering with eval-relevant code) across frontier models.
- [EvilGenie](https://futuretech.mit.edu/publication/evilgenie-a-reward-hacking-benchmark)
  — a reward-hacking benchmark for code environments.
- [Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/pdf/2507.02825)
  — what it takes for an agent benchmark's numbers to mean anything.

## Kernel-agent frameworks

The systems this project's agent shares a field with — read for their search
strategies and, just as much, for what their evaluation sections leave out:

- [KernelSkill](https://arxiv.org/html/2603.10085v1) — multi-agent framework
  with a dual-level memory architecture.
- [STARK](https://arxiv.org/pdf/2510.16996) — a strategic team of agents
  refining kernels.
- [SpecGen](https://arxiv.org/pdf/2606.17518) — speculative generation to cut
  the cost of the agentic loop itself.
- [CuTeGen](https://arxiv.org/pdf/2604.01489) — agentic generation in CuTe,
  relevant when the CuTe DSL question opens after the sprint.
- [KForge](https://arxiv.org/pdf/2511.13274) — program synthesis across diverse
  AI hardware accelerators.
