# Chapter 1 — Hardware fundamentals

The bottom layer: how a chip builds up from transistors all the way to a
complete GPU, and what the names — kernel, thread, warp, SM, HBM, Tensor
Core, compute capability — actually point to.

Sections 1 to 3 share a running example that grows in three steps:

```text
   step 1   z = x + y                  scalar addition
   step 2   C[i] = A[i] + B[i]         the same add across an array
   step 3   C = A × B                  matrix multiplication
```

Each step adds a new hardware concern. By the end of section 3 the running
example has reached matmul — the operation that dominates LLM inference.
Section 4 is the GPU-architecture reference: compute capability, Tensor Core
generations, and the specific GPUs this project uses.

## Sections

| # | Section | What it covers |
| --- | --- | --- |
| 1 | [Circuits and cores](01_circuits_and_cores.md) | Bits, transistors, logic gates, adders, ALU/FPU, registers, instruction pipeline |
| 2 | [Memory and caches](02_memory_and_caches.md) | Arrays, SRAM/DRAM cells, the memory hierarchy, L1/L2/L3, CPU core |
| 3 | [The GPU execution model](03_gpu_model.md) | Threads, blocks, SMs, warps, SIMT, latency hiding, tiled matmul, Tensor Cores, HBM/GDDR |
| 4 | [GPU architecture](04_gpu_architecture.md) | Compute capability, Tensor Core generations, specific GPUs used in this project |

Next chapter: [Chapter 2 — The CUDA software stack](../02_cuda_software_stack/README.md).
