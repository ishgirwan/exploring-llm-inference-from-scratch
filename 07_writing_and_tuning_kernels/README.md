# Chapter 7 — Writing and tuning a kernel

The earlier chapters name kernels constantly — a cuBLAS GEMM, the attention
kernel, a fused elementwise kernel — without ever opening one up. This chapter
opens one up, using a matrix multiply as the example: what a kernel is made of,
how to write a correct version, the optimization ladder that makes it fast
(coalescing → shared-memory tiling → register tiling → vectorization →
pipelining → Tensor Cores), the tuning knobs and autotuning that find the best
configuration, and why that configuration changes with the GPU architecture.

It's the bridge into the M1–M6 kernel-building work — a map of the craft, with
the real implementation and measurements deferred to those modules. It builds on
Chapters 1–4 (hardware, the CUDA stack, numerical types, measurement), not on the
attention/batching chapters, though it ties back to batching for the decode case.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [Writing and tuning a matmul](01_writing_and_tuning_a_matmul.md) | The four-part kernel skeleton; the naive matmul; verify-before-tune; the compute-bound optimization ladder (coalescing, shared-memory + register tiling, vectorization, pipelining); Tensor Cores as a track-switch; the decode GEMV regime and how batching turns it back into a GEMM; the tuning-knob/occupancy tension; autotuning; GPU-architecture dependence; industry practice and what "optimal" honestly means |
| 2 | [Reading a real optimized kernel](02_reading_a_real_kernel.md) | An annotated walkthrough of Triton's official matmul tutorial — a canonical open-source kernel that's already optimized and autotuned. Maps each part (grouped L2 ordering, block tiling, `tl.dot` → Tensor Cores, the `@triton.autotune` config list) back to §1's rungs, with the naive baseline for contrast and the verify/measure method (run at M6) |

Prerequisites: [Chapter 1 — Hardware fundamentals](../01_hardware_fundamentals/README.md), [Chapter 2 — The CUDA software stack](../02_cuda_software_stack/README.md), [Chapter 3 — Numerical types](../03_numerical_types/README.md), [Chapter 4 — Measurement](../04_measurement/README.md).
Next chapter: [Chapter 8 — Optimizing inference](../08_optimizing_inference/README.md). The kernel-building labs this chapter maps start at M0–M1 in the [Roadmap](../ROADMAP.md).
