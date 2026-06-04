# Circuits and cores

Section 1 of the hardware fundamentals chapter. It builds up from individual
transistors to a complete core capable of running one instruction stream.

The first three sections of the chapter share a running example that grows in
three steps:

```text
   step 1   z = x + y                  scalar addition
   step 2   C[i] = A[i] + B[i]         the same add across an array
   step 3   C = A × B                  matrix multiplication
```

This section covers what hardware needs for step 1. Sections 2 and 3 scale it
up for steps 2 and 3; section 4 is the GPU-architecture reference.

Next: [Memory and caches](02_memory_and_caches.md).

## 1. Start with one addition

The computation:

```text
x = 7
y = 5
z = x + y
```

Hardware needs:

```text
storage for x
storage for y
an adding circuit
storage for z
```

Data flow:

```text
  x storage ----+
                v
              +-----+
              | add | ----> z storage
              +-----+
                ^
  y storage ----+
```

This tiny diagram is the seed of the whole series. Cores, registers, caches,
threads, SMs, and HBM all exist to solve the same problem at larger scale:
store values, move values, compute on values, store results.

## 2. Numbers become bits

The machine stores numbers as bits. A bit has two possible values: 0 or 1.

Our example values can be written in binary:

```text
decimal 7   -> binary 0111
decimal 5   -> binary 0101
decimal 12  -> binary 1100
```

So this:

```text
7 + 5 = 12
```

becomes this:

```text
  0111
+ 0101
------
  1100
```

At the physical level, a bit is represented by an electrical state:

```text
logical 0 -> low voltage
logical 1 -> high voltage
```

Two states are useful because they are robust. The chip only has to distinguish
between low and high voltage ranges.

## 3. Transistors make controllable signals

A transistor is the chip's basic switching device. Think of it as a tiny
voltage-controlled switch with three terminals.

```text
             control gate
                 |
                 v
  source ---- [ transistor ] ---- drain

  gate low  -> off (no current)
  gate high -> on  (current flows)
```

What each terminal does:

```text
source   where charge carriers (electrons or holes) come from
drain    where the charge carriers go to
gate     sits above a thin insulating layer between source and drain;
         its voltage controls whether a conductive channel forms
         between them
```

The names are literal. The source supplies the charge, the drain receives it,
the gate gates the flow. The gate itself does not carry the working current —
it controls the channel beneath it by attracting charge with its electric
field.

By itself, a transistor is a switch, not a bit. It controls whether current
flows on a wire. Whether that on/off state ends up encoding logical 1 or
logical 0 depends on how the transistor is wired into a larger circuit; in
fact, in standard CMOS gates, "transistor on" can correspond to *either* a
high or a low output, depending on which transistor in the gate is being
switched. Bits emerge one layer up — at the gate level, the next subsection
— where the gate's *output voltage* is interpreted using the high-voltage =
logical 1, low-voltage = logical 0 convention from §2 above ("Numbers become
bits").

A modern chip has billions of these. The rest of this section shows how those
transistors compose into the building blocks the doc uses later.

One thing the diagram does not yet show: storage. A single transistor only
switches; it does not hold a value on its own. Storage emerges when several
transistors are wired so that the output of one drives the input of another
in a self-reinforcing loop that locks a 0 or a 1 in place. The actual SRAM
and DRAM bit cells — and how they connect back to "storage for x" from §1
above ("Start with one addition") — are walked through in
[Memory and caches §2](02_memory_and_caches.md#2-fast-storage-and-large-storage-use-different-circuits).
The rest of this doc focuses on the compute side: how switches compose into
logic, then arithmetic, then a core.

### Transistors -> logic gate

Wire a few transistors together and you get a logic gate. The simplest useful
one is a NAND (not-and): its output is low only when both inputs are high.

```text
   2-input NAND gate (about 4 transistors inside)

   A ----+
          +-- [ NAND ] -- out      out is LOW only if A AND B are both HIGH
   B ----+
```

The standard CMOS implementation uses four transistors arranged so the
output is pulled high unless both inputs are high — two transistors do the
pulling-down when both inputs are high, two do the pulling-up otherwise.
We do not need to look inside any further; from here on, "NAND" is one
named functional block.

NAND is enough on its own. Any other gate (AND, OR, NOT, XOR) can be built
from NANDs alone, so in principle the whole tower above this layer is just
NANDs.

### Logic gates -> 1-bit adder

A handful of gates wired correctly produces a 1-bit full adder: a circuit
that takes two bits plus a carry-in and produces a sum bit and a carry-out.
A *carry* is exactly the same idea as in grade-school addition — when one
column overflows past what one digit can hold, the extra moves into the next
column to the left. The "carry-in" is what arrives from the column to the
right; the "carry-out" is what leaves toward the column to the left.

```text
   1-bit full adder (about 5 gates inside)

   A bit -----+
              |
   B bit -----+-- [ gates ] -- sum
              |               -- carry out
   carry in --+
```

The internals do not matter for our story. "Add two bits" is now one named
functional block.

### 1-bit adders -> multi-bit adder

Chain four 1-bit adders, passing each adder's carry-out to the next adder's
carry-in, and you can add two 4-bit numbers. This structure is called a
*ripple-carry adder*, because the carry signal ripples from the rightmost
column to the leftmost. That is our 7 + 5:

```text
       carry flows left
             <----------------

       0   1   1   1       7
     + 0   1   0   1       5
     ----------------
       1   1   0   0      12
```

Real chips use 32-bit and 64-bit adders built the same way, with extra logic
for faster carry propagation (so the carry does not have to ripple through
every column before the highest-order sum bit can be known).

### The whole tower

Each layer is built from copies of the previous one:

```text
   chip / GPU         tens of billions of transistors
        ^
   core / SM          millions of transistors
        ^
   ALU                bundle of arithmetic circuits
                      (adder, subtractor, shifter, comparator, AND / OR / XOR)
        ^
   multi-bit adder    chained 1-bit adders
        ^
   1-bit adder        ~5 logic gates wired together
        ^
   logic gate         a few transistors wired together
        ^
   transistor         one switching device
```

A single 64-bit adder contains hundreds of transistors; an ALU contains
thousands; a core contains millions; a GPU contains tens of billions. Each
layer adds capability without needing to know how the layer below it is built.

Build-up so far:

```text
bits represent values
transistors switch electrical signals
logic gates combine signals
adders perform arithmetic on bits
ALUs bundle many arithmetic circuits together
```

## 4. The adding circuit becomes an Arithmetic Logic Unit

A processor needs circuits for many basic operations:

```text
add
subtract
compare       returns whether one value is less than, equal to, or greater
              than another (typically by subtracting and inspecting the
              sign and zero flags of the result)
shift bits    slides all the bits in a value left or right by N positions;
              a left shift by 1 is the same as multiplying by 2
AND / OR / exclusive OR (XOR)
              bit-by-bit logical operations
              XOR returns 1 only when its two inputs differ
```

Those circuits are grouped into an Arithmetic Logic Unit (ALU).

```text
  input A ----+
              v
            +-----+
            | ALU | ----> output
            +-----+
              ^
  input B ----+

  operation control: add, subtract, compare, ...
```

Integer arithmetic uses ALUs. Floating-point arithmetic uses a Floating-Point
Unit (FPU), because floating-point values have a sign, exponent, mantissa,
rounding rules, and special values — see
[Numerical types](../03_numerical_types/01_floating_point.md) for what each
of those parts does and why floating-point arithmetic needs a dedicated unit.

For LLMs, matrix multiplication appears so often that GPUs also include
specialized matrix units called Tensor Cores. We will reach those in
[The GPU execution model](03_gpu_model.md).

## 5. The ALU needs nearby storage

For:

```text
z = x + y
```

the ALU must receive `x` and `y`, then place the result somewhere.

The closest storage is a register.

```text
  register R1 holds x = 7
  register R2 holds y = 5

        R1 ----+
               v
             +-----+
             | ALU | ----> R3 holds z = 12
             +-----+
               ^
        R2 ----+
```

A register is a small physical storage slot close to the execution units. A
register file is a bank of many registers.

```text
  register file

  +------+-------+
  | R0   | value |
  | R1   | 7     |
  | R2   | 5     |
  | R3   | 12    |
  | ...  | ...   |
  +------+-------+
```

Registers sit close to execution units because every arithmetic instruction
needs operands. Shorter paths mean lower *latency* (the time from issuing an
operation to having the result available — measured in clock cycles or
nanoseconds) and less energy per operation (longer wires take more energy
to drive electrically).

## 6. Instructions drive the hardware

The source code:

```text
z = x + y
```

eventually becomes machine instructions. A simplified instruction might be:

```text
ADD R3, R1, R2
```

Meaning:

```text
read R1
read R2
add them
write the result to R3
```

A core runs this through a pipeline:

```text
fetch instruction
      |
      v
decode instruction
      |
      v
read registers
      |
      v
execute in ALU
      |
      v
write result
```

What each stage does:

```text
fetch     read the bits of the instruction from memory
          (using a separate register called the program counter that
          tracks the address of the next instruction)
decode    interpret that bit pattern to identify the operation and the
          register operands — e.g. "this is an ADD, sources are R1 and R2,
          destination is R3"
read      pull the operand values out of the named source registers
execute   run the operation in the ALU using those values
write     store the result in the named destination register
```

A pipeline overlaps instructions. While one instruction is in execute, the
next is already in decode, and the one after that is being fetched:

```text
   cycle:    1       2       3       4       5       6       7
   inst 1:  FETCH   DECODE  READ    EXEC    WRITE
   inst 2:          FETCH   DECODE  READ    EXEC    WRITE
   inst 3:                  FETCH   DECODE  READ    EXEC    WRITE
   inst 4:                          FETCH   DECODE  READ    EXEC
```

At cycle 5 all five stages are busy with different instructions. The pipeline
issues one new instruction per cycle even though each one takes 5 cycles end
to end. This is called instruction-level parallelism.

Now our tiny machine has the main pieces of a core:

```text
  +------------------------------------------------+
  | instruction fetch                              |
  |        |                                       |
  |        v                                       |
  | instruction decode                             |
  |        |                                       |
  |        v                                       |
  | register file <-----> ALU / FPU                |
  |        |                                       |
  |        v                                       |
  | load/store path to memory                      |
  +------------------------------------------------+
```

A core is this bundle: instruction control, nearby storage, execution units, and
paths to memory.

---

Next: [Memory and caches](02_memory_and_caches.md) — what feeds these
circuits when the data does not fit in registers.
