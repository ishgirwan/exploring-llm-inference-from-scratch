# Chapter 3 — Numerical types

LLM inference uses a zoo of floating-point and integer formats — fp32, fp16,
bf16, tf32, fp8, int8, int4. Picking the wrong one trades silently between
speed, memory, and correctness.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [Floating point and quantization](01_floating_point.md) | IEEE 754 storage and the value formula; fp32, fp16, bf16, tf32, fp8 (e4m3 / e5m2); int8 / int4 quantization with scale and zero-point; accumulator precision on Tensor Cores; dtype support by GPU |

Prerequisites: [Chapter 1 §4 — GPU architecture, Tensor Cores](../01_hardware_fundamentals/04_gpu_architecture.md#5-tensor-cores).
Next chapter: [Chapter 4 — Measurement](../04_measurement/README.md).
