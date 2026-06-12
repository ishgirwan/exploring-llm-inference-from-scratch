# Learning LLM inference, from the GPU up

I'm learning how LLMs run on GPUs — kernels, memory, profiling, serving
engines, and eventually writing competitive kernels myself — and doing it in
the open. This repo is the trail: the notes I worked out, the code, the
numbers, and what broke along the way. It isn't a course; when I explain
something here I'm really explaining it to myself.

## Where things stand

The prerequisite reading is written: nine chapters, from how a GPU executes
code up to multi-GPU serving and kernel engineering. The build phase — some 30
modules of kernels, benchmarks, and serving experiments on rented GPUs — is
next, and starts small.

## How to read this

- **[`ROADMAP.md`](ROADMAP.md)** — the master plan: the phases, the modules,
  the hardware, and the two ways to follow along. Start here.
- **[`LEARNING_PATH.md`](LEARNING_PATH.md)** — how I actually move: reading
  and labs braided together, stage by stage, each lab with the questions to
  ask inside it, the takeaway to internalize, and the question it hands to
  the next.
- **Chapters 1–8** — the reference library. Read straight through (~7–9 hrs)
  from [`01_hardware_fundamentals/`](01_hardware_fundamentals/README.md) if
  reading-first suits you, or pull sections just-in-time as the learning path
  calls them.
- **Chapter 9** — kernel-engineering prework for the final phase; read it when
  that phase starts, not first.

Don't trust my numbers — every one has a script behind it; re-run them. When I
find a mistake I fix it and log it rather than quietly editing history.
