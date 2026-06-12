# Chapter 2 — The CUDA software stack

GPU code passes through several layers between Python and the actual hardware
— PyTorch, kernel libraries, the CUDA toolkit, the driver, and finally the
chip. Knowing which layer owns what is the difference between "PyTorch sees
my GPU" and "PyTorch sees my GPU and uses it correctly."

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [The stack](01_the_stack.md) | Driver vs toolkit, host/device, what a CUDA program actually does, PyTorch's bundled runtime, the library layer (cuDNN, cuBLAS, NCCL, CUTLASS, Triton) |
| 2 | [End to end: a prompt becomes tokens](02_end_to_end_inference.md) | The whole inference path with every software layer and hardware component named: load → tokenize → prefill → decode loop, how a kernel is chosen, and the serving engine (vLLM/SGLang) that drives the loop |

Prerequisites: [Chapter 1 — Hardware fundamentals](../01_hardware_fundamentals/README.md).
Next chapter: [Chapter 3 — Numerical types](../03_numerical_types/README.md).
