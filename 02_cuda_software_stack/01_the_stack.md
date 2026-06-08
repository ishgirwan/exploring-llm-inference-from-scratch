# The CUDA software stack

GPU code passes through several layers between Python and the actual hardware.
Knowing which layer owns what is the difference between "PyTorch sees my GPU"
and "PyTorch sees my GPU and uses it correctly." This doc maps the layers and
the version constraints between them.

Prerequisites: [GPU architecture](../01_hardware_fundamentals/04_gpu_architecture.md).

## The layers, top to bottom

```
Your code                Python, C++, CUDA C++
   │
ML framework             PyTorch, JAX
   │
Kernel libraries         cuDNN (DL primitives), cuBLAS (BLAS),
                         NCCL (multi-GPU comms), CUTLASS
   │
Kernel languages         Triton, CUDA C++
   │
CUDA toolkit             nvcc compiler, runtime library, headers,
                         profiling tools (Nsight)
   │
CUDA driver              Kernel-mode driver; talks to hardware
   │
GPU hardware             The chip
```

Each layer above depends on the one below being a compatible version.

## Driver vs toolkit

The two version numbers you see most often refer to different things and are
constantly confused.

A note on a name collision before we go further: "kernel" in "kernel-mode
driver" below refers to the *operating-system* kernel — the privileged code
that talks to hardware. This is a completely different "kernel" from a *GPU
kernel* (a function compiled to run on the GPU, introduced in
[The GPU execution model §2](../01_hardware_fundamentals/03_gpu_model.md#2-gpu-threads-map-work-to-data)). Same word, two
unrelated concepts; both show up in this doc.

In one line: **the driver makes the GPU run; the toolkit lets you build programs
for it.** The driver is system software installed into the OS that talks to the
hardware; the toolkit is developer software — compiler, runtime library, headers,
profilers — that you build *against*. They ship and version separately, which is
why there are two numbers at all. The analogy that fixes the compatibility rule
below is building a phone app: the phone's OS is the *driver* (it must be new
enough to run your app), and the app-dev SDK is the *toolkit* (what you compile
against) — you can build with an SDK no newer than the phone's OS supports.

The two version numbers, concretely:

- **Driver version** — what `nvidia-smi` shows in its header. The kernel-mode
  driver installed by the OS. Talks to the hardware. Has a *maximum supported
  CUDA version* — the highest CUDA toolkit it can run.
- **Toolkit version** — what `nvcc --version` shows. The userland CUDA toolkit
  (compiler, headers, runtime library). "Userland" means the part of the
  system where ordinary applications run, as opposed to the OS kernel — the
  opposite of "kernel-mode" above. What CUDA C++ code is built against.

The rule: **toolkit version must be ≤ driver's max supported CUDA version.** A
driver from 2024 generally runs CUDA toolkit code from 2024 and earlier. A
driver from 2023 will not run a 2024 toolkit binary.

`nvidia-smi` shows e.g. `CUDA Version: 12.4`. That is **not** the toolkit
installed on the system. It is the *highest toolkit version this driver
supports*. Confusingly named, and the single most common source of "why doesn't
my GPU work" confusion.

There are really **three** numbers in play, and two of them look identical:

```text
  driver version        e.g. 550.54.15   nvidia-smi header
  "CUDA Version: 12.4"   in nvidia-smi    the MAX toolkit this DRIVER supports
                                          — a driver property, NOT what's installed
  installed toolkit      e.g. 12.1        nvcc --version
```

The middle one is the trap: `nvidia-smi` printing `CUDA Version: 12.4` means the
driver *could* run a 12.4 toolkit, not that one is installed (it may be 12.1, or
absent entirely).

## PyTorch's bundled CUDA runtime

PyTorch wheels (the `.whl` files installed by pip) ship their own CUDA runtime,
cuDNN, NCCL, and a precompiled set of kernels. The CUDA flavour is selected at
install time:

```
pip install torch                                                        # current default
pip install torch --index-url https://download.pytorch.org/whl/cu121     # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu124     # CUDA 12.4
```

The CUDA toolkit installed on the system does **not** affect PyTorch's bundled
runtime. What matters is whether the **driver** is new enough to run PyTorch's
bundled CUDA version. This is why "it worked yesterday" errors are so common on
cloud machines: a different image had a different driver.

Triton, by contrast, requires a system CUDA toolkit because it shells out to
`nvcc` and `ptxas` to compile kernels.

## What a CUDA program actually does

Two terms used throughout this section, and across the rest of the docs:

```text
host    the CPU and its RAM
device  the GPU and its VRAM
```

CUDA APIs and profiler outputs frequently use `H→D` for host-to-device
transfers (CPU → GPU) and `D→H` for the reverse.

A minimal mental model of how a CUDA program runs, hidden behind PyTorch:

1. **Allocate device memory.** `cudaMalloc` reserves space in the GPU's global
   memory. PyTorch's caching allocator wraps this.
2. **Copy host → device.** Data moves from CPU RAM to GPU VRAM over PCIe — the
   *Peripheral Component Interconnect Express* bus that connects the GPU card
   to the CPU motherboard — or over NVLink on multi-GPU systems (NVIDIA's
   proprietary GPU-to-GPU interconnect, much faster than PCIe). Either is
   slow relative to GPU memory bandwidth, and these transfers are often the
   hidden bottleneck for small operations.
3. **Launch a kernel.** Host code calls `kernel<<<grid, block>>>(args)`. The
   launch returns immediately; the GPU schedules and runs the kernel
   asynchronously.
4. **Synchronize.** The host calls `cudaDeviceSynchronize()` (or
   `torch.cuda.synchronize()`) when it needs to know the GPU has finished.
5. **Copy device → host.** Results move back to CPU RAM, if needed.
6. **Free device memory.**

```
  HOST (CPU, Python)                       DEVICE (GPU)
  ──────────────────                       ────────────
       │
       │ 1. cudaMalloc ─────────────────►   reserve VRAM
       │
       │ 2. cudaMemcpy H→D ─────────────►   copy data over PCIe / NVLink
       │
       │ 3. kernel<<<grid,block>>>(…) ──►   schedule kernel
       │      returns immediately               │
       │                                        │  run kernel across
       │  CPU continues other work              │  thousands of threads
       │                                        │
       │ 4. cudaDeviceSynchronize() ────►   wait until done
       │      (CPU blocks here)            ◄──── done
       │
       │ 5. cudaMemcpy D→H ◄────────────    copy results back
       │
       │ 6. cudaFree ───────────────────►   free VRAM
       ▼
```

PyTorch hides all of this. `y = layer(x)` triggers step 1 (caching allocator)
and step 3 (kernels from cuDNN, cuBLAS, or PyTorch's own libraries). Step 4
happens lazily when a value is needed back on CPU. Steps 2 and 5 happen on
`.to('cuda')` and `.cpu()`.

The asynchrony of step 3 is why benchmarks need CUDA events — see
[Benchmarking §timing-on-the-gpu](../04_measurement/01_benchmarking.md#timing-on-the-gpu).

## The libraries on top

- **cuDNN** — NVIDIA's deep-learning primitives (convolutions = sliding-window
  filter operations on image-like tensors; pooling = local downsampling such
  as max-pool; activations = elementwise non-linear functions like ReLU,
  GELU, SiLU). Closed-source. PyTorch uses it for conv-heavy workloads
  (less central to LLMs, which are dominated by plain matmuls).
- **cuBLAS** — NVIDIA's BLAS implementation. *BLAS* (Basic Linear Algebra
  Subprograms) is a long-standing standard set of low-level linear-algebra
  routines: matrix multiplies, vector dot products, matrix-vector products,
  and so on. Many vendors ship their own optimised BLAS — cuBLAS is NVIDIA's
  for the GPU. cuBLAS is the fast path behind `torch.matmul` for most shapes.
- **NCCL** — NVIDIA Collective Communications Library. Multi-GPU collectives
  (all-reduce, all-gather). Used by tensor-parallel and pipeline-parallel
  inference.
- **CUTLASS** — Open-source C++ templates for building fast GEMM (*General
  Matrix Multiply* — BLAS's standard name for `C = αAB + βC`, the central
  operation of dense linear algebra) and related kernels. Not a runtime
  library — a toolkit for *writing* one. Used inside FlashAttention,
  FlashInfer, and many serving engines.
- **Triton** — Python *DSL* (Domain-Specific Language: a small programming
  language built for one problem domain) for writing GPU kernels.
  Open-source, not from NVIDIA. JIT-compiles to PTX. **PTX** (Parallel
  Thread Execution) is
  NVIDIA's portable virtual assembly language — GPU-independent and shipped
  inside CUDA binaries. The driver then compiles PTX down to **SASS**, the
  actual hardware instruction set for the specific GPU the code will run on.
  (NVIDIA uses the acronym SASS without consistently expanding it; you will
  see "shader assembly" and "streaming assembler" in community references.)
  Both PTX and SASS reappear in
  [First-run effects §2](../04_measurement/02_first_run_effects.md#2-jit-compilation).

The five at a glance:

| Library | What it does | Call it / build with it | Source | LLM hot path? |
| --- | --- | --- | --- | --- |
| cuBLAS | matrix multiply (GEMM), linear algebra | call | closed (NVIDIA) | **yes** — every projection + MLP |
| cuDNN | conv / pooling / activation DL primitives | call | closed (NVIDIA) | rarely — LLMs have no convolutions |
| NCCL | cross-GPU communication (all-reduce, all-gather) | call | open (NVIDIA) | only multi-GPU |
| CUTLASS | C++ templates to *build* GEMM/attention kernels | build with | open (NVIDIA) | indirectly — inside FlashAttention etc. |
| Triton | Python language to *write* kernels | build with | open (non-NVIDIA) | indirectly — custom / fused kernels |

Read down the table: **cuBLAS, cuDNN, and NCCL are ready-made functions you
*call*; CUTLASS and Triton are tools you *build* kernels with.** For an LLM on a
single GPU only cuBLAS is on the hot path — cuDNN lights up for vision, NCCL for
multi-GPU, and CUTLASS/Triton when someone hand-writes a kernel (most often
attention, which no NVIDIA call-library provides in the form serving needs).

## What people actually use

The deciding axis in practice is **call vs. build**, not open vs. closed. To
*call*, teams take whatever is fastest regardless of license — closed cuBLAS for
GEMM, open NCCL for comms, open FlashAttention / FlashInfer for attention (the one
hot path where an open kernel, not an NVIDIA call-library, is the standard). To
*build* — when no call-library fits a fused, quantized, or custom kernel — they
write their own, and the open authoring tools win because you have to read and
tune what you write.

How a custom kernel gets written runs along an *automation* axis, from most human
effort to least:

```text
hand-written, hand-tuned   a human writes the kernel AND manually picks the tile
                           sizes, memory layout, and unrolling. Peak speed, most
                           effort. (cuBLAS internals, FlashAttention.)

hand-written, auto-tuned   a human writes the kernel; a search tool benchmarks
                           configurations and keeps the fastest automatically.
                           (Triton @autotune, the CUTLASS profiler.)

compiler-generated         you write high-level PyTorch ops; torch.compile's
                           Inductor backend GENERATES and tunes a Triton kernel —
                           no one hand-writes it. The cheapest real option.

AI-generated               an LLM writes the kernel, often iterating against a
                           benchmark. The frontier — but "beats handwritten"
                           headlines need two checks: beat WHICH baseline, and is
                           the output correct?
```

The auto-tuned row is exactly the first-call cost from
[end-to-end §8](02_end_to_end_inference.md#8-who-chooses-the-kernel-and-when). The
pragmatic ladder: try `torch.compile` first (free fusion), write **Triton** when
you need a real custom kernel quickly, and drop to **CUTLASS / CUDA C++** only at a
scale where the last few percent pays back the expert time. This is M6 and M16
territory; I'll firm it up when I build there.

## Why this project uses a container

The number of moving versions — driver, toolkit, PyTorch wheel, Triton, vLLM,
cuDNN, NCCL — means a benchmark on Tuesday and one on Friday can be measuring
a slightly different stack. The pinned Docker image in `setup/docker/` freezes
everything except the driver, which is part of the host. Recording the driver
version with every benchmark (see
[Benchmarking §environment](../04_measurement/01_benchmarking.md#environment)) closes the
last gap.

## Useful commands

```bash
nvidia-smi                # GPU model, driver, max supported CUDA, VRAM usage
nvidia-smi -q             # verbose: power state, clocks, processes
nvcc --version            # toolkit version (if installed)
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
python -c "import triton; print(triton.__version__)"
```

Next: [End to end: a prompt becomes tokens](02_end_to_end_inference.md) — how all
these layers cooperate to turn one prompt into generated text, from model load
through the decode loop.
