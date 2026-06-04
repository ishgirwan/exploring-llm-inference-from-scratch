# The GPU execution model

Section 3 of the hardware fundamentals chapter. We scale the running example
once more, to the operation that dominates LLM inference:

```text
   C = A × B
```

This doc covers how a Graphics Processing Unit (GPU) runs many independent
operations at once, what the streaming multiprocessor looks like inside, and
why matrix multiplication's data reuse pattern shapes the GPU memory hierarchy.

Prerequisites: [Circuits and cores](01_circuits_and_cores.md) and
[Memory and caches](02_memory_and_caches.md).
Next: [GPU architecture](04_gpu_architecture.md) — compute capability,
specific GPU models, and Tensor Core generations.

## 1. The array example exposes parallel work

The array add has independent elements:

```text
C[0] = A[0] + B[0]
C[1] = A[1] + B[1]
C[2] = A[2] + B[2]
C[3] = A[3] + B[3]
```

Many workers can run the same instruction pattern on different elements.

```text
worker 0 -> C[0] = A[0] + B[0]
worker 1 -> C[1] = A[1] + B[1]
worker 2 -> C[2] = A[2] + B[2]
worker 3 -> C[3] = A[3] + B[3]
```

This is the point where a GPU becomes useful. A GPU spends more chip area on
many arithmetic lanes and high memory bandwidth, and less on making one
instruction stream extremely clever.

```text
CPU: fewer large cores, strong single-thread latency
GPU: many simpler lanes, high total throughput
```

Throughput is total work completed per second.

## 2. GPU threads map work to data

On a GPU, the array add can be written so each thread handles one element:

```text
thread 0 -> C[0] = A[0] + B[0]
thread 1 -> C[1] = A[1] + B[1]
thread 2 -> C[2] = A[2] + B[2]
...
```

A thread is a software instruction stream. In this example, each thread runs
the same code with a different `i`. The function each thread runs is called a
kernel: a function compiled for the GPU and launched in parallel by many
threads at once.

```text
same kernel code:

i = thread_id
C[i] = A[i] + B[i]
```

NVIDIA's GPU programming platform is CUDA. CUDA originally stood for Compute
Unified Device Architecture. In CUDA the program running on the host CPU
writes the kernel as a function marked for the GPU, then launches it with a
special syntax that says how many threads to spawn:

```text
// kernel — this code runs on the GPU, once per thread
__global__ void add_kernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // each thread's index
    if (i < n) C[i] = A[i] + B[i];
}

// launch — this code runs on the host CPU
add_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);
```

The `__global__` keyword marks the function as a kernel — code that runs on
the GPU and is callable from the host CPU. The `<<<...>>>` launch syntax is
how the host triggers it: spawn `num_blocks × threads_per_block` threads in
total, organised into `num_blocks` groups of `threads_per_block` each. Those
groups are the blocks introduced in the next section. Inside the kernel, the
built-in variables `threadIdx.x`, `blockIdx.x`, and `blockDim.x` (the
thread's index within its block, the block's index in the launch, and the
block size) let each thread compute the unique element it should work on.

The GPU may launch thousands or millions of these threads for a large tensor.
A *tensor* is an n-dimensional array of numbers: scalars are 0-d, vectors are
1-d, matrices are 2-d, and the weight buffers and activations in modern models
are often 3-d or 4-d. In memory, a tensor is a flat block of bytes plus shape
and stride metadata that describes how to index into it. That large supply
of threads serves two purposes:

```text
1. cover many tensor elements
2. give the GPU other work to run while some threads wait for memory
```

The second point is central. GPUs tolerate memory latency by keeping many pieces
of work available.

## 3. Blocks give threads local cooperation

Millions of threads need structure. CUDA groups threads into *blocks*. The
block size is something the programmer picks at kernel launch (the
`threads_per_block` argument from §2); common choices are 128, 256, or 512
threads per block.

```text
grid
  |
  +-- block 0
  |     +-- threads 0..255
  |
  +-- block 1
  |     +-- threads 256..511
  |
  +-- block 2
        +-- threads 512..767
```

A block is assigned to one Streaming Multiprocessor (SM), the GPU's main
execution unit.

Threads in the same block can:

```text
share fast on-chip shared memory   (a small SRAM region inside the SM
                                    that every thread in the block can
                                    read and write — full details in §10)
synchronize with each other
```

To *synchronize* here means to wait at a *barrier* until every thread in the
block has reached the same point in the code, then continue. Barriers are how
a block coordinates its threads — for example: "all of you finish loading the
data into shared memory before any of you start reading from it."

This grouping has a hardware reason. Local cooperation inside one SM can use
nearby storage and local control. Whole-GPU cooperation requires wider
coordination and is handled at coarser points.

One thing to flag up front: the block is *not* the only grouping a thread
belongs to. The hardware also bundles threads into *warps* of 32 for
scheduling purposes — see §5. Block is the programmer's grouping; warp is
the hardware's. Both are true of every thread at the same time.

## 4. The SM is the GPU execution unit

An SM is one parallel processing unit of the GPU. A modern GPU has tens to
hundreds of SMs, each with its own schedulers, arithmetic lanes, registers,
and on-chip memory — an SM is roughly analogous to a CPU core, but
restructured to favour many parallel lanes over a single fast instruction
stream. When a kernel launches, the GPU's *block scheduler* hands each
block to one SM; that SM runs every thread in the block to completion
before the block leaves. One SM can hold several blocks at the same time,
depending on how many registers and how much shared memory each block
needs.

The diagram below uses FP for Floating Point and INT for Integer.

Simplified SM:

```text
  +--------------------------------------------------------------+
  | schedulers                                                   |
  |     |          |          |          |                       |
  |     v          v          v          v                       |
  |  FP/INT     FP/INT     FP/INT     FP/INT     ... lanes       |
  |                                                              |
  |  +------------------+     +------------------------------+   |
  |  | Tensor Cores     |     | load/store units             |   |
  |  +------------------+     +------------------------------+   |
  |                                                              |
  |  +--------------------------------------------------------+  |
  |  | large register file                                   |  |
  |  +--------------------------------------------------------+  |
  |                                                              |
  |  +--------------------------------------------------------+  |
  |  | L1 cache / shared memory                              |  |
  |  +--------------------------------------------------------+  |
  +--------------------------------------------------------------+
```

A *lane* is one of the SM's parallel arithmetic units — think of it as the
GPU's equivalent of one ALU slot, with one lane per active thread when a
warp issues. FP/INT lanes handle regular integer and floating-point
arithmetic for scalar and vector work. A *scalar* operation processes one
value; a *vector* operation processes a short, ordered tuple of values
together (for example, four floats at once through one SIMD instruction).

The SM owns the local resources needed by the block:

```text
registers for active threads
arithmetic lanes for regular operations
Tensor Cores for matrix operations    (specialised matrix-multiply units
                                       built into the SM — full details in §11)
load/store units for memory access
L1 cache and shared memory for nearby data
schedulers to choose ready work
```

Many arithmetic lanes share the same surrounding support logic. This is more
area-efficient than giving every lane a full CPU-like control system.

## 5. Warps make scheduling efficient

An SM has many lanes (hundreds). Scheduling one independent instruction
stream for every lane would require a lot of control logic, so the hardware
doesn't try.

Instead, NVIDIA SMs bundle threads within each block into *warps*. A warp is
32 consecutive threads from the same block: threads 0–31 of a block form its
warp 0, threads 32–63 form warp 1, and so on. Warps never cross block
boundaries; every warp belongs to exactly one block.

A warp is the actual unit the SM schedules: in one cycle, a *warp scheduler*
inside the SM picks one ready warp and broadcasts one instruction to all 32
of its lanes (one lane per thread).

For the array add:

```text
one warp runs one instruction:

C[i] = A[i] + B[i]

lane 0:  C[0]  = A[0]  + B[0]
lane 1:  C[1]  = A[1]  + B[1]
lane 2:  C[2]  = A[2]  + B[2]
...
lane 31: C[31] = A[31] + B[31]
```

The instruction is shared across the warp. The data differs per thread. This
model is called Single Instruction, Multiple Threads (SIMT). SIMT is closely
related to the older idea of *Single Instruction, Multiple Data* (SIMD), used
in CPU vector units (AVX, NEON, etc.): SIMD applies one instruction to a
vector of lanes inside a single execution unit; SIMT expresses the same idea
as many threads, each executing one lane's worth of work. The hardware
realisation is essentially the same — many lanes ganged together under one
instruction — but SIMT exposes the parallelism more explicitly in the
programming model (you write code from the perspective of one thread).

The rationale:

```text
one scheduler decision feeds many lanes
control overhead is shared
more chip area can go to arithmetic lanes and memory bandwidth
```

When threads in a warp take different branches, the SM runs the paths in
separate steps with some lanes inactive.

```text
if value > 0:
    path A
else:
    path B

path A step: lanes needing A are active
path B step: lanes needing B are active
```

This is warp divergence. It explains why GPUs prefer regular control flow: a
warp works best when its threads run the same instruction path.

## 6. Putting the pieces together: kernel launch to lane execution

That covers every piece introduced so far — kernel, thread, block, SM,
warp, lane. They form a single chain from CUDA source code on the host CPU
down to one arithmetic lane firing inside the GPU. Drawn in one picture:

```text
  +-------------------------------------------------------------------+
  | host CPU                                                          |
  |                                                                   |
  |   __global__ void add_kernel(...) { ... }                         |
  |   add_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);      |
  +-------------------------------------------------------------------+
                                    |
                                    | kernel launch
                                    v
  +-------------------------------------------------------------------+
  | GPU                                                               |
  |                                                                   |
  |   grid = (num_blocks x threads_per_block) total threads,          |
  |          arranged as num_blocks blocks                            |
  |                                                                   |
  |                       block scheduler                             |
  |                hands each block to one SM                         |
  |              +------+------+------+------+------+                 |
  |              v      v      v      v      v                        |
  |          +-----+ +-----+ +-----+ +-----+ +-----+                  |
  |          | SM0 | | SM1 | | SM2 | | SM3 | | ... |   (many SMs)     |
  |          +-----+ +-----+ +-----+ +-----+ +-----+                  |
  +-------------------------------------------------------------------+
                                    |
                                    | zoom into one SM
                                    v
  +-------------------------------------------------------------------+
  | one SM                                                            |
  |                                                                   |
  |   currently holds one or more resident blocks (see §8):           |
  |                                                                   |
  |   +-------- block A --------+   +-------- block B --------+       |
  |   | warp 0 : threads  0..31 |   | warp 0 : threads  0..31 |       |
  |   | warp 1 : threads 32..63 |   | warp 1 : threads 32..63 |       |
  |   | warp 2 : ...            |   | ...                     |       |
  |   +-------------------------+   +-------------------------+       |
  |                                                                   |
  |   each cycle, a warp scheduler picks one ready warp and           |
  |   broadcasts its next instruction to all 32 of its lanes:         |
  |                                                                   |
  |                         [chosen warp]                             |
  |                              |                                    |
  |                              v                                    |
  |              +---+---+---+- ... -+---+---+                        |
  |              | L0| L1| L2|       |L30|L31|                        |
  |              +---+---+---+- ... -+---+---+                        |
  |                                                                   |
  |   each lane runs that one instruction using its own thread's      |
  |   registers and the SM's L1 / shared memory                       |
  +-------------------------------------------------------------------+
```

The four levels of the hierarchy, top to bottom:

```text
grid       everything one kernel launch creates
  block      the programmer's grouping; runs on one SM
    warp       the hardware's grouping; 32 threads scheduled together
      thread     one element of work; runs on one SM lane for one cycle
```

What you choose in CUDA vs. what the hardware decides:

```text
programmer chooses (in the kernel launch):
  - block size              (the threads_per_block in <<<...>>>)
  - number of blocks        (the num_blocks in <<<...>>>)
  - what each thread does   (the body of the kernel function)

hardware decides (automatically, at runtime):
  - which SM each block lands on        (block scheduler)
  - which warp each thread ends up in   (consecutive 32-thread groups
                                         carved out of each block)
  - which warp's instruction is issued
    next on each SM                     (warp scheduler)
```

So *you* control the work and how it's chunked; *the GPU* controls where
each chunk runs and in what order.

One word in that table is worth pausing on: *decides*. It is tempting to
picture a little scheduling program running somewhere on the chip, choosing
warps each cycle. There isn't one. The block scheduler and the warp
scheduler are not software — they are *fixed-function logic*: circuits whose
behaviour is etched into the silicon when the chip is manufactured, the way a
thermostat is built to trip at a set temperature rather than running code to
decide. The warp scheduler is essentially a bank of status bits (which
resident warps are ready) feeding a selection circuit that outputs "issue
this warp", and it settles to an answer in a fraction of one clock cycle.

It *has* to be built that way. The warp scheduler picks a warp every clock
cycle — on a fast GPU that is roughly once per nanosecond. A software routine
would need many instructions, and many cycles, to make each choice; it would
spend more time deciding than the lanes spend computing. Only dedicated logic
is fast enough to sit inside a per-cycle decision.

There *is* real software inside a modern GPU, but it runs much slower work. A
small processor on the chip (an on-chip microcontroller) runs *firmware* —
code stored in on-die read-only memory or on-board flash, loaded by the
driver — and handles power and clock management, switching between different
applications' work, and similar housekeeping, on a scale of microseconds to
milliseconds, not nanoseconds. The GPU *driver* that compiles and launches
kernels is ordinary software too, but it runs on the host CPU, not on the
GPU. The pattern:

```text
per-cycle decision    (which warp issues now)        -> hardwired logic in silicon
per-block placement   (which SM a block lands on)     -> hardwired logic (block scheduler)
management            (power, clocks, app switching)  -> firmware on an on-chip microcontroller
launch / orchestration (compile, enqueue work)        -> driver software on the host CPU
```

The rule of thumb: the faster and more repetitive a decision is, the further
down it is pushed — from driver software, to on-chip firmware, to logic
frozen into the silicon. The schedulers in this section sit at the bottom of
that ladder.

This closes a loop with §1. The chip area a CPU spends "making one
instruction stream extremely clever" *is* this same kind of hardwired
decision logic: a CPU's out-of-order scheduler is a large circuit choosing
which instruction to run next, every cycle, for one thread. A GPU spends far
less logic per thread — one lean warp scheduler is shared across 32 threads
at once (§5). So "the hardware decides" here and "less chip area on
per-thread cleverness" back in §1 are the same fact seen from two sides.

A natural follow-up question: if the hardware bundles threads into warps
anyway, why does the programmer also need to pick a block size? Why not
just use warps directly?

Because the two groupings serve different jobs:

```text
warp   scheduling unit: which 32 threads fire in the same cycle on the
       same warp scheduler. Size is baked into the hardware (32 on NVIDIA);
       you don't pick it.
block  cooperation unit: which threads can share data via shared memory
       and synchronize at a barrier. Size is your choice, based on how
       many threads your algorithm needs to work together.
```

Threads in different blocks *cannot* share shared memory or synchronize
with each other, even when both blocks happen to be running on the same
SM. So the block is the boundary you draw around "these threads need to
see each other's intermediate results."

Many useful patterns need more than 32 cooperating threads. (A *tile* is
a small rectangular sub-block of a larger array, loaded into on-chip
memory once and then reused — §9 walks through why matmul wants them.
`__syncthreads()` is CUDA's barrier call: every thread in the block waits
at that line until all threads have reached it.)

```text
matmul tile     a 128x128 tile of A and B loaded into shared memory once
                can feed 256+ threads computing different output elements.
                With only 32 threads cooperating, that load would feed 32
                workers and you'd lose most of the reuse benefit.
reduction       summing 1024 numbers cooperatively is fast when 256
                threads each write to shared memory and then read each
                other's partial sums in a tree pattern. With 32 threads
                you'd serialize the work.
tile loading    256 threads each fetching 4 floats fills a 1024-element
                shared-memory tile in one coordinated load, then everyone
                calls __syncthreads() and uses it.
```

A loose analogy:

```text
warp  = meeting room   (fixed capacity, decided by the building)
block = project team   (size you choose based on the work)
```

A team can occupy several meeting rooms at once; the rooms don't know
which team you're on. Likewise a 256-thread block gets carved into 8 warps
by the hardware, but as the programmer you reason about the team, not the
rooms.

There is also a portability angle. Warp size is 32 on NVIDIA, 64 on AMD,
and could change on future hardware. Code written in terms of blocks (with
explicit numbers like 256) stays portable; code written in terms of warps
gets tied to one vendor's current GPU.

The next sections compare GPU hardware to CPU hardware at the chip level
(§7), explain why the SM keeps so many warps resident at once (latency
hiding, §8), and walk through shared memory (§10) and Tensor Cores (§11).

## 7. CPU and GPU at the chip level: side by side

The previous sections drilled into one SM, one warp, one block, and the
kernel that launches them. Step back now and look at how the whole chip
is organised — and what changes if we draw the same picture for a CPU.

Both chips are built from the same building blocks (execution units,
registers, caches, off-chip DRAM) but arrange them very differently.

A CPU chip, simplified:

```text
  +-------------------------------------------------------------------+
  | CPU chip (e.g. Intel Xeon, AMD EPYC, Apple M-series)              |
  |                                                                   |
  |   +-----------+  +-----------+  +-----------+  +-----------+      |
  |   |   core 0  |  |   core 1  |  |   core 2  |  |   core 3  | ...  |
  |   | front-end |  | front-end |  | front-end |  | front-end |      |
  |   |  + ALUs   |  |  + ALUs   |  |  + ALUs   |  |  + ALUs   |      |
  |   |  regs+L1  |  |  regs+L1  |  |  regs+L1  |  |  regs+L1  |      |
  |   |     L2    |  |     L2    |  |     L2    |  |     L2    |      |
  |   +-----+-----+  +-----+-----+  +-----+-----+  +-----+-----+      |
  |         |              |              |              |            |
  |         +--------------+--------------+--------------+            |
  |                                |                                  |
  |                                v                                  |
  |   +-------------------------------------------------------------+ |
  |   | shared L3 cache                                             | |
  |   +-------------------------------------------------------------+ |
  |                                |                                  |
  |                                v                                  |
  |   +-------------------------------------------------------------+ |
  |   | memory controller(s)                                        | |
  |   +-------------------------------------------------------------+ |
  +-------------------------------------------------------------------+
                                   |
                                   v
                       +---------------------------+
                       |  DDR4 / DDR5 DRAM (RAM)   |
                       +---------------------------+
```

A GPU chip, simplified:

```text
  +-------------------------------------------------------------------+
  | GPU chip (e.g. NVIDIA H100, 132 SMs total)                        |
  |                                                                   |
  |  +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+   |
  |  | SM | | SM | | SM | | SM | | SM | | SM | | SM | | SM | | SM |   |
  |  +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+   |
  |  +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+   |
  |  | SM | | SM | | SM | | SM | | SM | | SM | | SM | | SM | | SM |   |
  |  +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+ +----+   |
  |                                                                   |
  |   ...many more SMs in the same grid pattern (132 on H100)...      |
  |                                                                   |
  |   each SM contains schedulers, FP/INT lanes, Tensor Cores,        |
  |   registers, and L1 + shared memory — see §4 above                |
  |                                                                   |
  |                                |                                  |
  |                                v                                  |
  |   +-------------------------------------------------------------+ |
  |   | shared L2 cache                                             | |
  |   +-------------------------------------------------------------+ |
  |                                |                                  |
  |                                v                                  |
  |   +-------------------------------------------------------------+ |
  |   | HBM controllers                                             | |
  |   +-------------------------------------------------------------+ |
  +-------------------------------------------------------------------+
                                   |
                                   v
                       +---------------------------+
                       |   HBM stacks (or GDDR)    |
                       +---------------------------+
```

For the components inside one core / inside one SM, see:

```text
one CPU core   Memory and caches §5 (in the previous doc)
one SM         §4 above
```

The same parts, named on each side:

| Layer | CPU | GPU |
| --- | --- | --- |
| chip | one CPU package | one GPU package |
| parallel unit | core (a few to ~100 per chip) | SM (tens to hundreds per chip) |
| inside the unit | front-end + out-of-order scheduler + a few wide execution units (ALU / FPU / load-store) | warp scheduler(s) + many parallel lanes (FP / INT / Tensor Core) + load/store |
| local storage | registers + L1 + L2 (per core) | registers + L1 / shared memory (per SM) |
| cross-unit cache | shared L3 | shared L2 |
| off-chip DRAM | DDR4 / DDR5 (RAM) | HBM or GDDR (VRAM) |
| optimised for | low latency on one instruction stream | high throughput across many parallel threads |

And how the software pieces you write map onto each side:

| Software piece | CPU | GPU |
| --- | --- | --- |
| program / kernel | a process the OS launches and schedules on cores | a kernel launched from the host as `kernel<<<num_blocks, threads_per_block>>>` |
| unit of work | OS thread — the OS multiplexes it onto cores | GPU thread — runs as one lane of one warp on one SM |
| cooperation unit | (no equivalent below the process) | block — threads share `__shared__` memory and synchronize via `__syncthreads()` |
| hardware scheduling unit | one OS thread per logical core at a time | one warp (32 threads) per warp-scheduler issue |
| fast on-chip storage | hardware-managed cache (implicit; you don't address it directly) | shared memory (explicit; kernel-controlled, sized at launch) |
| off-chip data | RAM | VRAM |

The big-picture takeaway:

```text
CPU: a few large cores, each with deep machinery for one fast
     instruction stream. Caches are big and hardware-managed.
GPU: many simple SMs, each holding many small arithmetic lanes.
     The fast on-chip memory (shared memory) is explicitly managed by
     the kernel; the latency-hiding lever is keeping many warps
     resident, not making one instruction stream faster.
```

That second design only pays off when there is a lot of independent
work — which is exactly the situation in the array add of §1 and the
matmul of §9.

## 8. Many warps hide memory latency

Memory access can take many cycles. The SM keeps multiple warps resident so it
can switch to ready work. A *resident* warp is one whose state — register
contents, program counter, etc. — currently occupies SM resources, so the
scheduler can issue it on any cycle without first loading anything from
elsewhere. The SM has a hardware limit on how many warps can be resident
at once, set by how many registers and how much shared memory each block
consumes.

```text
resident warps on one SM

warp 0 -> waiting for memory
warp 1 -> ready
warp 2 -> ready
warp 3 -> waiting for memory
warp 4 -> ready
```

The scheduler can issue from ready warps:

```text
cycle 0: issue warp 1
cycle 1: issue warp 2
cycle 2: issue warp 4
cycle 3: data for warp 0 arrives, issue warp 0
```

This is why GPU kernels often need many threads. The extra threads provide work
that can run while other warps wait for data.

That raises a concrete question: with all this talk of resident warps, how
many threads can a GPU actually run at once? The numbers all come from the
GPU's published spec sheet.

Three distinct counts to keep separate:

```text
launched   threads created by the kernel<<<...>>> call. Effectively
           unlimited — the GPU queues them in waves as SMs free up.
resident   threads whose state lives on the SM right now (the resident
           warps defined above). Hardware-bounded.
issued     threads firing arithmetic this exact cycle. Even smaller —
           bounded by how many warp schedulers each SM has.
```

The interesting number for performance is *resident*, because that is the
latency-hiding budget: the pool of warps the scheduler can choose from
when others are waiting on memory.

The math, top-down:

```text
threads per warp              = 32           (fixed in NVIDIA hardware)
max resident warps per SM     = from spec    (e.g. 64 on Hopper / Ampere)
max resident threads per SM   = warps x 32
max resident threads on GPU   = SMs x threads_per_SM
```

Concrete example — H100 SXM5:

```text
SMs                            132
max warps per SM               64
max threads per SM             64 x 32      = 2,048
max resident threads on GPU    132 x 2,048  = 270,336

warp schedulers per SM         4
threads firing arithmetic
  in one cycle, whole GPU      132 x 4 x 32 = 16,896
```

So an H100 keeps about 270,000 threads' worth of state alive at once, but
only about 17,000 are running an instruction in any given cycle. The other
~253,000 are *resident but waiting* — typically stalled on memory loads.
That gap is the whole reason GPUs spend transistors on so many resident
warps: keep enough alive that one is always ready when the scheduler
looks.

Real kernels rarely reach the architectural cap. Each block also consumes
registers and shared memory from fixed per-SM pools (e.g. H100 has 65,536
registers and ~228 KB of shared memory per SM), so a kernel with heavy
register use or large shared-memory allocations holds fewer blocks per SM.
The fraction of the cap a kernel actually achieves is called *occupancy*.
The next doc, [GPU architecture](04_gpu_architecture.md), covers compute
capability, the per-GPU spec numbers, and the occupancy metric used by
profilers to summarise how full an SM actually is.

## 9. Matrix multiplication adds data reuse

Array add is simple but has little reuse:

```text
C[i] = A[i] + B[i]
```

Each loaded value is used once.

Matrix multiplication has more reuse:

```text
C = A × B

C[row, col] =
    A[row, 0] * B[0, col] +
    A[row, 1] * B[1, col] +
    A[row, 2] * B[2, col] +
    ...
```

One value from A can be used for several columns of C. One value from B can be
used for several rows of C.

```text
        B columns
        v
      B[0,0] B[0,1] B[0,2]

A[0,0] contributes to C[0,0], C[0,1], C[0,2]
A[1,0] contributes to C[1,0], C[1,1], C[1,2]
```

This reuse creates an opportunity. If we load a small rectangular sub-block of
A and B (a tile) into fast on-chip storage once, the SM can reuse those values
for many multiply-adds before going back to memory.

## 10. Shared memory holds reusable tiles

Shared memory is fast on-chip SRAM controlled by the kernel. It sits inside the
SM and is shared by threads in a block.

A tiled matmul uses it like this:

```text
1. load a tile of A from HBM or GDDR into shared memory
2. load a tile of B from HBM or GDDR into shared memory
3. threads reuse those tiles for many multiply-adds
4. store the result tile of C
```

Data path:

```text
  HBM / GDDR
      |
      | load A tile, B tile
      v
  shared memory on one SM
      |
      | repeated reads by threads in the block
      v
  registers
      |
      v
  FP lanes / Tensor Cores
      |
      v
  registers
      |
      | store C tile
      v
  HBM / GDDR
```

The rationale is reuse. A single expensive global-memory load can feed many
arithmetic operations.

## 11. Tensor Cores accelerate the matmul pattern

Regular FP lanes can do multiply-add operations. Tensor Cores are specialized
for small matrix multiply-accumulate operations.

A multiply-accumulate (MAC) is `a × b + c` performed as a single fused
operation: one multiply and one add, computed together without writing the
intermediate product back to a register. MAC is the atomic step of matrix
multiplication, because each output element of `C = A × B` is itself a sum
of products:

```text
C[row, col] = A[row, 0]*B[0, col] + A[row, 1]*B[1, col] + A[row, 2]*B[2, col] + ...
```

That is a chain of MACs running an accumulator. A regular FP lane does one
scalar MAC per cycle. A Tensor Core does many MACs over a small matrix tile
in one instruction — that is where the dramatic speedup comes from.

The pattern:

```text
D = A × B + C
```

Tensor Cores operate on small tiles of matrices. Libraries and compilers arrange
larger matmuls as many tile operations.

```text
large matrix multiply
        |
        v
many small tile multiplies
        |
        v
Tensor Core instructions
```

LLM inference uses matrix multiplications in attention and feed-forward layers,
so Tensor Cores become a central part of the performance story.

## 12. L2 connects the SMs to GPU memory

Each SM has local registers, L1 cache, and shared memory. The GPU also has a
larger L2 cache shared across SMs.

```text
  SM 0 ----+
  SM 1 ----+
  SM 2 ----+---- shared L2 cache ---- HBM/GDDR
  SM 3 ----+
  SM 4 ----+
```

The shared L2 helps when:

```text
different SMs read nearby or repeated data
stores need a common path toward memory
global memory traffic should be reduced
```

L2 is larger than L1 and reachable by all SMs, so it sits farther away and
serves more traffic. That makes it slower than local SM storage and much faster
than off-chip DRAM.

## 13. HBM and GDDR are large off-chip memory

The model weights, activations, and Key-Value cache (KV cache) usually live in
GPU DRAM. This is often called Video Random-Access Memory (VRAM). The KV
cache is the running list of attention-layer keys and values from every
previously processed token in the current sequence, kept around so each new
generated token does not have to recompute them — it is the largest dynamic
data structure in LLM decoding and a major reason VRAM size matters; a later
module covers it in depth.

VRAM does not mean SRAM. On modern GPUs, VRAM usually means off-chip DRAM:

```text
consumer GPU VRAM     -> usually GDDR DRAM
datacenter GPU VRAM   -> often HBM DRAM
```

VRAM is also not disk storage. When a model is loaded from disk, its weights are
copied into CPU memory and/or GPU VRAM so the GPU can access them quickly.

Consumer GPUs commonly use Graphics Double Data Rate memory (GDDR):

```text
  GDDR layout, simplified top view

  +------+   +------+   +------+
  | DRAM |   | DRAM |   | DRAM |
  +------+   +------+   +------+

             +--------+
             |  GPU   |
             +--------+

  +------+   +------+   +------+
  | DRAM |   | DRAM |   | DRAM |
  +------+   +------+   +------+
```

Datacenter GPUs commonly use High Bandwidth Memory (HBM) stacks close to the
GPU die:

```text
  HBM layout, simplified side view

       HBM stack              GPU die              HBM stack
     +----------+          +----------+          +----------+
     | DRAM die |          |          |          | DRAM die |
     | DRAM die |          |   GPU    |          | DRAM die |
     | DRAM die |          |          |          | DRAM die |
     +----------+          +----------+          +----------+
  --------------------------------------------------------------
                    silicon interposer
```

A silicon interposer is a thin slab of silicon sitting beneath the GPU die
and the HBM stacks. Thousands of fine wires are etched into it, connecting
the GPU and the HBM stacks side-by-side at very high pin density — far more
wires per millimetre than a normal printed circuit board can fit. The
interposer is what makes HBM's "very wide connection" physically possible.

HBM gets high bandwidth from a very wide connection between memory stacks and
the GPU.

```text
bandwidth = bytes per transfer x transfers per second
```

"Transfers per second" depends on the *signaling speed* — how fast bits are
clocked across each individual wire (the data rate per pin). "Bytes per
transfer" depends on *bus width* — how many wires are running in parallel.
GDDR and HBM make opposite trades on this:

```text
GDDR: narrower connection, high signaling speed
      (each wire is clocked very fast; few wires)
HBM:  very wide connection, close physical placement
      (many more wires through the silicon interposer; each clocked slower,
       but multiplied by the wide bus the total bandwidth is higher)
```

This is why datacenter GPUs can feed many SMs at high bandwidth.

DRAM refresh happens only while the memory has power. The memory controller
periodically refreshes rows by reading and rewriting them before the capacitor
charge leaks too far.

```text
system running        -> DRAM/VRAM is powered and refreshed
system sleep          -> memory may stay powered and refreshed
system hibernate      -> memory contents are written to disk, then power can turn off
system powered off    -> DRAM/VRAM loses its contents
```

GPU VRAM follows the same idea. If the GPU is reset or powered off, the contents
of VRAM are lost and must be loaded again.

## 14. Put the whole path together

For the original scalar add:

```text
register x -> ALU -> register z
register y ---^
```

For array add on a GPU:

```text
HBM/GDDR
   |
   v
L2 cache
   |
   v
SM L1 cache
   |
   v
registers for each thread
   |
   v
FP/INT lanes add A[i] + B[i]
   |
   v
register result
   |
   v
store C[i] back toward HBM/GDDR
```

For tiled matmul:

```text
HBM/GDDR
   |
   | load A tile and B tile
   v
shared memory
   |
   | reuse tile values
   v
registers
   |
   v
Tensor Cores
   |
   v
result registers
   |
   v
store C tile to HBM/GDDR
```

The same simple idea has grown:

```text
one add:
  storage -> ALU -> storage

array add:
  memory -> cache -> registers -> lanes -> memory

matmul:
  memory -> shared tiles -> registers -> Tensor Cores -> memory
```

## 15. Why LLM inference is often memory-bound

Two terms that recur in the rest of this project:

```text
memory-bound    math units sit idle waiting for bytes to arrive from memory
compute-bound   memory delivers bytes faster than math can consume them
```

Which side a kernel falls on depends on the ratio of arithmetic operations
performed to bytes loaded — the *arithmetic intensity*. A kernel that does a
lot of math per byte loaded (high intensity) tends to be compute-bound; one
that does a little math per byte (low intensity) tends to be memory-bound.
This balance is formalised by the *roofline model*, picked up in a later doc.

LLM inference repeatedly moves large tensors:

```text
model weights
activations
KV cache entries
logits             (the model's per-vocabulary-token output scores
                    produced at the final layer of each forward pass)
temporary buffers
```

The math units can only run when their input data has arrived.

Low reuse pattern:

```text
load data -> small amount of math -> load more data

math units: work  wait wait  work  wait wait
```

High reuse pattern:

```text
load tile once -> reuse it many times

math units: work work work work work
```

Many later optimizations are ways to improve this path:

```text
coalescing       neighboring threads read neighboring memory
tiling           loaded data is reused on-chip
shared memory    reusable data is placed under kernel control
fusion           intermediate values stay close to compute
quantization     fewer bytes move per value
KV cache layout  decode-time reads become more efficient
```

Coalescing comes up often enough to expand on. When the 32 threads in a warp
read memory at consecutive addresses, the hardware combines those reads into
one wide transaction. Scattered or strided reads require many smaller
transactions and waste bandwidth, even when the same total number of bytes is
moved.

## 16. Abbreviations used across these docs

| Abbreviation | Full form |
| --- | --- |
| ALU | Arithmetic Logic Unit |
| CMOS | Complementary Metal-Oxide Semiconductor |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture |
| DRAM | Dynamic Random-Access Memory |
| FP | Floating Point |
| FPU | Floating-Point Unit |
| GDDR | Graphics Double Data Rate memory |
| GPU | Graphics Processing Unit |
| HBM | High Bandwidth Memory |
| INT | Integer |
| KV cache | Key-Value cache |
| L1 | Level 1 cache |
| L2 | Level 2 cache |
| L3 | Level 3 cache |
| LLM | Large Language Model |
| 1T1C | One-transistor, one-capacitor |
| 6T | Six-transistor |
| SIMT | Single Instruction, Multiple Threads |
| SM | Streaming Multiprocessor |
| SRAM | Static Random-Access Memory |
| SSD | Solid-State Drive |
| VRAM | Video Random-Access Memory |

## 17. What to carry into the GPU architecture doc

The important build-up across all three docs:

```text
bits represent numbers
transistors build logic and storage
adders and ALUs perform arithmetic
registers feed execution units quickly
caches keep nearby array data close
CPU cores optimize a few instruction streams for latency
GPUs use many lanes and many threads for throughput
SMs run thread blocks
warps share one instruction across many threads
shared memory reuses tiles on-chip
L2 and HBM/GDDR feed data to all SMs
```

Three statements in later docs should now have concrete reasons:

```text
GPUs need many threads
  -> ready warps keep SMs busy while other warps wait on memory

Cache and shared memory matter
  -> they reduce expensive trips to off-chip DRAM

LLM inference is often memory-bound
  -> weights and KV cache are large, and math units wait for bytes
```

Next: [GPU architecture](04_gpu_architecture.md) — the reference companion
to this doc. It holds the memory hierarchy as a single latency/bandwidth
table, the SM occupancy definition that pairs with §8 above, the Tensor
Core generation/dtype mapping, NVIDIA's compute-capability numbering, and
the specific GPUs this project uses.
