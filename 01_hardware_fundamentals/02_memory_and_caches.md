# Memory and caches

Section 2 of the hardware fundamentals chapter. We scale the running example
from a single addition to a whole array:

```text
   C[i] = A[i] + B[i]
```

Arrays do not fit in registers, so the chip needs larger storage. This doc
covers how data is physically stored (SRAM and DRAM), why the memory hierarchy
exists, and how a CPU core uses it.

Prerequisites: [Circuits and cores](01_circuits_and_cores.md).
Next: [The GPU execution model](03_gpu_model.md).

## 1. The example grows into arrays

One addition fits in registers. Real work uses many values.

Grow the example:

```text
C[i] = A[i] + B[i]
```

For four elements:

```text
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
C = [6, 8, 10, 12]
```

The same add operation repeats:

```text
C[0] = A[0] + B[0]
C[1] = A[1] + B[1]
C[2] = A[2] + B[2]
C[3] = A[3] + B[3]
```

The ALU still adds two values at a time. The new problem is storage capacity.
Arrays and tensors are too large to live entirely in registers, so the machine
needs larger memory.

```text
registers: active values for current instructions
memory:    larger storage for arrays and tensors
```

## 2. Fast storage and large storage use different circuits

The hardware uses two main memory technologies:

```text
Static Random-Access Memory (SRAM)
Dynamic Random-Access Memory (DRAM)
```

"Random-access" means the hardware can access a chosen location by address.
The important difference is how one bit is stored.

Before looking at SRAM and DRAM cells, it helps to know how memory is arranged.
Memory is laid out as a grid of cells:

```text
                         bit line 0   bit line 1   bit line 2
                             |            |            |
word line row 0 --------- [ cell ] ---- [ cell ] ---- [ cell ]
                             |            |            |
word line row 1 --------- [ cell ] ---- [ cell ] ---- [ cell ]
                             |            |            |
word line row 2 --------- [ cell ] ---- [ cell ] ---- [ cell ]
                             |            |            |
```

The names mean:

```text
cell       one tiny circuit that stores one bit
word line  row-select wire; turns on all cells in one row
bit line   column wire; carries the bit value out of or into a selected cell
```

When memory reads a location, it selects one row with a word line. The selected
cells connect to their bit lines, and the bit lines carry the stored values to
the rest of the chip.

### SRAM: a bit held by a latch

SRAM stores one bit with a small latch circuit. A common SRAM bit cell uses six
transistors, often called a six-transistor (6T) cell.

The core idea is two NOT gates feeding each other. A NOT gate flips one bit:

```text
input 0 -> output 1
input 1 -> output 0
```

```text
  SRAM bit cell idea

                 +-------------------------------+
                 |                               |
                 v                               |
          Q --> [ NOT gate A ] --> Q_bar --> [ NOT gate B ]
          ^                                             |
          |                                             |
          +---------------------------------------------+

  If Q = 1:
      NOT gate A makes Q_bar = 0
      NOT gate B sees 0 and makes Q = 1
      the loop reinforces Q = 1

  If Q = 0:
      NOT gate A makes Q_bar = 1
      NOT gate B sees 1 and makes Q = 0
      the loop reinforces Q = 0
```

That feedback loop is the storage. The bit is stored as one of two stable
electrical states:

```text
store 1: Q = 1, Q_bar = 0
store 0: Q = 0, Q_bar = 1
```

The word "bar" means the opposite value. `Q_bar` is pronounced "Q bar" and is
the complement of `Q`.

The cell also needs a way to connect that stored bit to the memory array:

```text
  SRAM cell with access switches

              bit line                         bit line bar
                 |                                  |
                 |                                  |
            [ access ]                        [ access ]
            [ switch ]                        [ switch ]
                 |                                  |
                 v                                  v
                 Q ---- two cross-coupled NOT gates ---- Q_bar

              both access switches are controlled by the word line
```

SRAM uses two bit lines because the cell stores both the value and its
opposite. Reading a pair of opposite signals is easier and faster than reading
one tiny signal by itself, and writing can force the latch into either state.

The six transistors come from the common Complementary Metal-Oxide
Semiconductor (CMOS) implementation:

```text
2 transistors -> NOT gate A
2 transistors -> NOT gate B
1 transistor  -> access switch from Q to bit line
1 transistor  -> access switch from Q_bar to bit line bar
--------------------------------------------------------
6 transistors per SRAM bit cell
```

How SRAM reads a bit:

```text
1. The memory selects a row by turning on its word line.
2. The access switches connect Q and Q_bar to the two bit lines.
3. The two bit lines show opposite values.
4. A small sensing circuit decides whether Q was 1 or 0.
5. The cross-coupled NOT gates keep holding the value after the read.
```

How SRAM writes a bit:

```text
1. Write circuitry drives the two bit lines to the desired values.
   To write 1: bit line = 1, bit line bar = 0.
   To write 0: bit line = 0, bit line bar = 1.
2. The word line turns on the access switches.
3. The bit lines force Q and Q_bar into the new state.
4. The word line turns off.
5. The NOT-gate loop keeps the new value.
```

Why SRAM is fast:

```text
1. The bit is already held as a strong, stable 0/1 state.
2. Reading mostly connects that stable state to nearby wires.
3. SRAM arrays used for registers and caches are kept small and close to compute.
4. SRAM has no refresh step.
```

Why SRAM holds fewer bits in the same area:

```text
1. One bit commonly uses six transistors.
2. More transistors per bit means a larger physical cell.
3. Larger cells mean fewer bits fit in the same chip area.
```

SRAM is used for small, fast storage:

```text
registers
Level 1 cache (L1)
Level 2 cache (L2)
Level 3 cache (L3) on many Central Processing Units (CPUs)
GPU shared memory
```

### DRAM: a bit held as charge

DRAM stores one bit as charge on a tiny capacitor. A common DRAM bit cell uses
one transistor and one capacitor, often called a one-transistor,
one-capacitor (1T1C) cell.

```text
  DRAM bit cell idea

                 word line
                     |
                     v
  bit line ---- [ access transistor ] ---- storage node ---- capacitor ---- ground

  charged capacitor   -> 1
  uncharged capacitor -> 0
```

How to read this diagram:

```text
The capacitor is the storage bucket.
Charge in the bucket represents 1.
Little or no charge represents 0.
The access transistor opens when the word line selects this cell.
The bit line carries the tiny charge signal to sensing circuits.
```

How DRAM writes a bit:

```text
1. To write 1, the memory drives the bit line high.
2. The word line opens the access transistor.
3. Charge flows into the capacitor.
4. The word line turns off, trapping charge on the capacitor.

To write 0, the same path discharges the capacitor instead.
```

How DRAM reads a bit:

```text
1. The bit line is prepared at a middle voltage.
2. The word line opens the access transistor.
3. The capacitor shares its tiny charge with the bit line.
4. The bit line voltage moves slightly up or down.
5. A sense amplifier turns that tiny change into a clear 1 or 0.
6. The read disturbs the capacitor, so the value is written back.
```

A sense amplifier is a small circuit that detects a tiny voltage difference and
amplifies it into a full digital 0 or 1.

Why DRAM can hold more bits than SRAM:

```text
1. One bit uses one transistor plus one capacitor.
2. That is much smaller than a six-transistor SRAM cell.
3. Smaller cells allow many more bits in the same chip area.
4. Dense arrays make DRAM suitable for gigabytes of memory.
```

Why DRAM is slower:

```text
1. The capacitor stores a tiny amount of charge, so reading it needs sensing.
2. Reading can disturb the stored charge, so the value must be restored.
3. Capacitors leak charge over time, so rows must be refreshed periodically.
4. DRAM arrays are large and often off-chip, so wires and controllers add delay.
```

The key tradeoff:

```text
SRAM: more circuitry per bit -> fast, expensive area, small capacity
DRAM: less circuitry per bit -> dense, cheaper area, larger capacity, slower
```

Why use SRAM near compute instead of putting DRAM there?

```text
1. Distance is only part of latency.
   Even close DRAM still has capacitor sensing, restore, and refresh behavior.

2. Registers and L1 caches need extremely fast access.
   They are read and written constantly by execution units.

3. SRAM presents a strong stored 0/1 quickly.
   DRAM presents a tiny charge that must be sensed and restored.

4. DRAM is dense when built as large arrays.
   Very small near-compute memories would still need sense amplifiers, refresh
   logic, row selection, and control circuitry.

5. Logic chips and DRAM chips are optimized differently.
   DRAM wants dense capacitor cells. CPU and GPU logic wants fast transistors,
   fast wires, and many local connections.
```

There are designs that put DRAM closer to compute. Embedded DRAM puts DRAM on a
logic chip, and HBM places DRAM stacks very close to the GPU package. These
improve capacity or bandwidth, but they still do not replace SRAM for the
closest storage because the DRAM cell itself is slower to read and maintain.

That physical tradeoff creates the memory hierarchy.

One more important point: both SRAM and DRAM are volatile memory.

```text
volatile memory     data survives only while power is supplied
persistent storage  data survives when power is off
```

SRAM and DRAM are volatile. They are used while the program is running. A disk
or Solid-State Drive (SSD) is persistent storage. It stores data differently:

```text
magnetic storage   bits encoded as the orientation of magnetic domains on a
                   spinning platter — what a traditional hard disk drive (HDD)
                   does
flash memory       bits encoded as charge trapped in floating-gate transistors;
                   the trapped charge survives without power — what an SSD does
```

Either is much slower than RAM but cheaper per byte and persistent.

So the hierarchy is:

```text
registers / caches     SRAM, volatile, very fast
CPU RAM                DRAM, volatile, larger
GPU VRAM               DRAM, volatile, larger
SSD / disk             persistent storage, much slower
```

## 3. Cache appears when arrays are reused nearby

In the array example:

```text
C[i] = A[i] + B[i]
```

the code usually reads neighboring elements:

```text
A[0], A[1], A[2], A[3], ...
B[0], B[1], B[2], B[3], ...
```

Accessing DRAM for every element would leave the core waiting often. So the
processor keeps recently used memory in a cache: a small SRAM memory close to
the core.

The cache moves data in chunks called cache lines.

```text
  memory addresses

  ... 1000 1001 1002 1003 1004 1005 1006 1007 ...
      |-----------------------------------------|
                  one cache line
```

If the core asks for `A[0]`, the hardware may fetch the line containing nearby
values too. Then `A[1]`, `A[2]`, and `A[3]` are already close.

Two terms used constantly from here on:

```text
cache hit    the requested address is already in the cache, so the read is fast
cache miss   the address is not in the cache, so the hardware has to fetch from
             a slower memory level (and usually fills a whole line while it is
             there)
```

Reading the array becomes:

```text
read A[0] -> cache miss, fetch line
read A[1] -> cache hit
read A[2] -> cache hit
read A[3] -> cache hit
```

The rationale has a name in textbooks — *locality of reference*. There are
two kinds:

```text
spatial locality   the program is likely to access addresses near one it
                   just accessed (so fetching a whole cache line, not one byte,
                   pays off)
temporal locality  the program is likely to access the same address again
                   soon (so keeping it in the cache pays off)
```

Both kinds shape how caches and the rest of the memory hierarchy are designed.

## 4. L1, L2, and L3 are levels of closeness

A single cache has a size-speed tradeoff. Tiny SRAM can sit very close to a
core. Larger SRAM takes more area and has longer internal paths.

So processors use levels:

```text
  core
    |
    v
  L1 cache     Level 1 cache: small, closest, fastest
    |
    v
  L2 cache     Level 2 cache: larger, farther, slower
    |
    v
  L3 cache     Level 3 cache: common on CPUs, larger again
    |
    v
  DRAM         largest, often off-chip
```

For one element of the array add:

```text
load A[i]
    |
    v
check L1
    |
    v
check L2
    |
    v
fetch from DRAM if needed
    |
    v
place value in a register
    |
    v
ALU adds it to B[i]
```

The closer the value is found, the sooner the ALU can run.

## 5. One CPU core running the array loop

A Central Processing Unit (CPU) core is designed to run a small number of
instruction streams quickly.

Our array add might compile into a loop:

```text
for i in range(n):
    C[i] = A[i] + B[i]
```

The CPU core repeatedly:

```text
load A[i]
load B[i]
add
store C[i]
increment i
check whether loop is done
```

[§6 of the previous doc](01_circuits_and_cores.md#6-instructions-drive-the-hardware)
laid out the basic pipeline: fetch → decode → read registers → execute →
write. A real CPU core wraps that pipeline with extra hardware so the
execution units stay fed even when memory is slow or the next instruction is
unknown.

The new pieces compared to that basic pipeline:

```text
branch predictor        guesses which way the next branch will go so fetch
                        can keep running ahead instead of waiting for the
                        outcome (longer walk-through below)
out-of-order scheduler  watches a window of upcoming instructions and issues
                        whichever one is ready, instead of waiting in source
                        order for a slow instruction to finish (longer
                        walk-through below)
integer ALU(s)          one or more ALUs (Circuits and cores §4) specialised
                        for integer arithmetic; a core often has more than
                        one so the scheduler can issue several integer ops
                        in the same cycle
vector unit / FPU       a vector unit applies one operation to several
                        values at once — a "vector" of 4, 8, 16, ... numbers
                        packed side-by-side in a single wide register (a
                        register large enough to hold multiple values, e.g.
                        a 256-bit register holds eight 32-bit floats); the
                        floating-point unit (FPU,
                        [Circuits and cores §4](01_circuits_and_cores.md#4-the-adding-circuit-becomes-an-arithmetic-logic-unit))
                        handles non-integer arithmetic. The two are often
                        grouped because vector instructions are usually the
                        floating-point ones in numerical code
load/store unit         the only execution unit that talks to memory; it
                        moves values between the register file and L1 cache
register file           bank of registers (Circuits and cores §5) that the
                        execution units read operands from and write results
                        back to
```

Simplified CPU core:

```text
  +-----------------------------------------------------------+
  | front-end                                                 |
  |   branch predictor --> fetch --> decode                   |
  |                                     |                     |
  |                                     v                     |
  |                       out-of-order scheduler              |
  |          dispatches operations in parallel to:            |
  |                                                           |
  |    +---------+   +---------+   +-------------+            |
  |    | integer |   | vector /|   | load/store  |            |
  |    | ALU(s)  |   | FPU     |   | unit        |            |
  |    +----+----+   +----+----+   +------+------+            |
  |         |             |               |                   |
  |         |             |        +------+------+            |
  |         v             v        v             v            |
  |    +-----------------------------+   +-----------+        |
  |    |       register file         |   | L1 cache  |        |
  |    +-----------------------------+   +-----------+        |
  +-----------------------------------------------------------+
```

The drawn arrows show *which* boxes connect; each link actually carries both
reads and writes (an execution unit pulls operands from the register file
and pushes results back into it). The load/store unit is the only one with
an arrow to L1 — that is the rule worth remembering: ALU and FPU results
only reach L1 when a later store instruction moves them there. L1 in turn
connects upward to L2, L3, and DRAM as drawn in §4 above.

Two of the new pieces deserve a longer look.

The *branch predictor* helps with loop and `if` decisions. When the pipeline
fetches the instruction after a branch, it does not yet know which side will
be taken — so it guesses, using a small hardware table that remembers how
each branch went last time, and speculatively runs the predicted path. If the
guess was right, no time is lost; if wrong, the speculative work is discarded
and the pipeline restarts on the correct path.

The *out-of-order scheduler* helps find independent instructions while
another instruction waits for memory. A simple "in-order" core would stall on
a slow load. An out-of-order core looks ahead in the instruction stream,
finds a later instruction that does not depend on the pending load, and runs
it instead. Results are still committed in program order, so the visible
behaviour matches the source code.

These features make one instruction stream progress quickly.

That design uses significant chip area per core, so CPUs usually have a modest
number of large cores.

---

Next: [The GPU execution model](03_gpu_model.md) — what changes when many independent
operations can run at once.
