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
   2-input NAND gate (4 transistors inside)

   A ----+
          +-- [ NAND ] -- out      out is LOW only if A AND B are both HIGH
   B ----+
```

This is the one rung where *physics becomes logic*: below it are physical
switches moving current through transistors; above it is Boolean logic working
on abstract 1s and 0s. §3 above ("Transistors make controllable signals")
flagged that a switch only becomes a bit "one layer up, at the gate level" —
this is that layer, so it is the one place we open all the way down to the
transistors. Every rung above NAND is logic built from logic — gates wired into
an adder, adders into an ALU — and the build-up that follows opens each of
those one level deep: far enough to see how the pieces below it wire together,
then sealed back into a named block.

CMOS — the technology nearly all chips use — provides two complementary kinds
of transistor that switch on opposite inputs (the "complementary" in CMOS):

```text
NMOS   conducts (switch closed) when its gate is HIGH
PMOS   conducts (switch closed) when its gate is LOW
```

A gate wires these between two *rails* — supply wires each held at a fixed
voltage:

```text
VDD    the high-voltage rail   (logical 1)
GND    the ground rail, 0 V    (logical 0)
```

- A *pull-up network* of PMOS can connect the output up to VDD (drive it HIGH).
- A *pull-down network* of NMOS can connect the output down to GND (drive it LOW).

For a NAND the four transistors form two PMOS in parallel on top and two NMOS
in series on the bottom:

```text
                  VDD   (high rail = logical 1)
                   |
        +----------+----------+
        |                     |
      [ P1 ]               [ P2 ]        two PMOS, in PARALLEL  (pull-up)
        |                     |
        +----------+----------+
                   |
                   +----------------->  out
                   |
                 [ N1 ]                  two NMOS, in SERIES   (pull-down)
                   |
                 [ N2 ]
                   |
                  GND   (ground rail = logical 0)
```

Each input is a *single* wire that fans out to one PMOS and one NMOS — so the A
wire drives the gates of both P1 and N1, and the B wire drives both P2 and N2.
There are still only two inputs; each just touches two transistors:

```text
   P1 gate = A      N1 gate = A      (the A wire fans out to both)
   P2 gate = B      N2 gate = B      (the B wire fans out to both)
```

Two wiring choices do all the logical work:

```text
parallel pull-up    out reaches VDD if P1 OR  P2 conducts
series  pull-down   out reaches GND if N1 AND N2 both conduct
```

Now walk the four input combinations, remembering PMOS conducts on LOW and
NMOS conducts on HIGH:

```text
 A  B | P1  P2  | N1  N2  | path to VDD?  | path to GND?  | out
 -----+---------+---------+---------------+---------------+-----
 0  0 | on  on  | off off | yes (both)    | no            |  1
 0  1 | on  off | off on  | yes (P1)      | no (N1 open)  |  1
 1  0 | off on  | on  off | yes (P2)      | no (N2 open)  |  1
 1  1 | off off | on  on  | no            | yes (both)    |  0
```

The output sits LOW only on the bottom row, where both inputs are high — which
is exactly the NAND rule. On every other row at least one PMOS pulls the output
up to VDD. (This is the concrete form of the §3 point that "transistor on" can
mean either output level: a conducting PMOS drives the output high, while the
two conducting NMOS together drive it low.)

One behavior falls straight out of the series/parallel arrangement: stacking
the NMOS in series naturally *inverts*, because only all-high inputs complete
the path down to ground, producing a LOW. That is why NAND is the cheap,
natural CMOS primitive at four transistors, while a plain AND needs two more
to flip the output back the other way.

From here on we treat NAND as one named functional block.

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
   1-bit full adder (5 gates inside)

   A bit -----+
              |
   B bit -----+-- [ gates ] -- sum
              |               -- carry out
   carry in --+
```

What the gates inside compute is fixed by how binary addition of three bits
(A, B, and the carry-in) works. Adding three 1-bit values gives a total from 0
to 3, which needs two output bits: the low bit is the *sum*, the high bit is
the *carry-out*.

```text
 A  B  Cin | A+B+Cin | carry-out  sum
 ----------+---------+----------------
 0  0   0  |    0    |     0       0
 0  0   1  |    1    |     0       1
 0  1   0  |    1    |     0       1
 0  1   1  |    2    |     1       0
 1  0   0  |    1    |     0       1
 1  0   1  |    2    |     1       0
 1  1   0  |    2    |     1       0
 1  1   1  |    3    |     1       1
```

Reading the two output columns against the gates from the previous subsection:

```text
sum        is 1 when an ODD number of the inputs are 1   -> A XOR B XOR Cin
carry-out  is 1 when at least TWO of the inputs are 1    -> (A AND B) OR ((A XOR B) AND Cin)
```

(XOR — "exclusive OR" — outputs 1 only when its two inputs differ, so chaining
two XORs yields 1 exactly when an odd number of the three inputs are 1.)

The tidy way to wire that uses two *half-adders*. A half-adder adds just two
bits, with no carry-in: its XOR gives their sum, its AND gives the carry those
two bits generate.

```text
   half-adder (adds two bits)

   X --+--[ XOR ]-- sum     (1 if X and Y differ)
       |
   Y --+--[ AND ]-- carry   (1 only if X and Y are both 1)
```

A full adder is two half-adders plus one OR gate, wired in three stages:

```text
   stage 1   HA1 adds A and B        -> s1 (partial sum),  c1 (carry)
   stage 2   HA2 adds s1 and Cin     -> sum,               c2 (carry)
   stage 3   OR  combines c1 and c2  -> carry-out
```

```text
   1-bit full adder = 2 half-adders + 1 OR   (5 gates)

       A ---+
            +--> [ HA1 ] --s1--+
       B ---+        |         +--> [ HA2 ] --> sum
                     |   Cin --+        |
                    c1                  c2
                     |                  |
                     +----> [ OR ] <----+
                              |
                              v
                          carry-out
```

The carry-out is 1 if *either* half-adder produced a carry — exactly the "at
least two inputs are 1" rule from the truth table. Trace any row of the table
through the three stages and the two outputs match.

Opened once, this is enough. From here up, "add two bits" is one named block — a
*full adder* — drawn as a single box with inputs A, B, Cin and outputs sum and
carry-out.

### 1-bit adders -> multi-bit adder

Chain four 1-bit full adders, passing each adder's carry-out into the next
adder's carry-in, and you can add two 4-bit numbers. Each FA is the block we
just built; only the carry wires connect them. This structure is called a
*ripple-carry adder*, because the carry signal ripples from the rightmost
column (the least significant bit) to the leftmost:

```text
   4-bit ripple-carry adder — four full adders, carry chained right to left

       A3 B3        A2 B2        A1 B1        A0 B0
        | |          | |          | |          | |
        v v          v v          v v          v v
      +-----+  c3   +-----+  c2   +-----+  c1   +-----+
  <---| FA3 |<------| FA2 |<------| FA1 |<------| FA0 |<--- carry-in = 0
  c4  +-----+       +-----+       +-----+       +-----+
        |             |             |             |
        v             v             v             v
        S3            S2            S1            S0
```

Each FA takes one bit of A, one bit of B, and the carry rippling in from its
right; it emits one sum bit and the carry into its left neighbour. The leftmost
carry-out (c4) is the overflow bit. Feeding in our 7 + 5 (A = 0111, B = 0101)
and letting the carry ripple left:

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
The top two rungs — core/SM and chip/GPU — are where the CPU and GPU designs
part ways; §7 below ("From one core to a chip") walks that fork once the core
itself is built.

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

Inside, an ALU is not one clever circuit that "knows" every operation. It is a
bank of separate circuits — the adder built above, plus an AND circuit, an OR
circuit, an XOR circuit, a shifter, a comparator — all wired to the same two
inputs, A and B. Every one of them computes its own result at the same time. A
*multiplexer* then picks which result actually leaves the ALU.

A multiplexer (mux) is a selector circuit: it has several data inputs, a few
*control* inputs, and one output, and the control bits choose which single data
input is passed through to the output — the electronic equivalent of a rotary
switch. In an ALU the control bits are the operation code from the instruction:
"add" selects the adder's output, "and" selects the AND circuit's output, and
so on.

```text
   inside an ALU

       A ---+-----+-----+-----+------+
            |     |     |     |      |
       B ---+-----+-----+-----+------+
            v     v     v     v      v
          +---+ +---+ +---+ +---+ +-----+
          |ADD| |SUB| |AND| |OR | |SHIFT|   ... all compute at once
          +---+ +---+ +---+ +---+ +-----+
            |     |     |     |      |
            +-----+-----+-----+------+
                        |
                        v
                   +---------+
                   |   MUX   | <-- operation-select bits (from the instruction)
                   +---------+
                        |
                        v
                     output     (only the selected circuit's result)
```

Computing every operation and keeping only one looks wasteful, but in hardware
it is cheap and fast: each circuit is small, they all run concurrently, and a
mux selects in less time than it would take to first decide *which* circuit to
run and only then run it. This is an early glimpse of a theme that returns on
the GPU — hardware will happily spend extra transistors to avoid spending time.

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

## 7. From one core to a chip: where CPU and GPU diverge

§6 finished one core: instruction control, registers, execution units, and a
path to memory. A whole chip is many cores on one piece of silicon — and *how*
the core is replicated is the single fork that separates a CPU from a GPU.

```text
                         one core (from §6)
                                |
               +----------------+-----------------+
               |                                  |
        replicate a FEW                   strip the core down to its
        full, heavyweight cores           arithmetic lanes, then replicate
        as-is                             those MANY times, grouped into SMs
               |                                  |
               v                                  v
      +----------------------+        +-----------------------------+
      | CPU                  |        | GPU                         |
      | a handful of cores,  |        | thousands of lanes bundled  |
      | each with deep       |        | into tens-to-hundreds of    |
      | control to run ONE   |        | SMs, sharing control to run |
      | instruction stream   |        | MANY threads at once        |
      | as fast as possible  |        |                             |
      +----------------------+        +-----------------------------+
         optimized for                    optimized for
         latency                          throughput
```

The two designs spend the same silicon budget on opposite things. A CPU pours
area into making a few instruction streams individually fast: large caches and
elaborate per-core control (branch prediction, out-of-order execution) wrapped
around a few strong cores. A GPU pours that area into *many* simple arithmetic
lanes plus wide memory bandwidth, and shares one control unit across a whole
bundle of lanes — a Streaming Multiprocessor (SM) — so that far more of the
chip is doing arithmetic at any moment. (A *lane* is one arithmetic slot, the
GPU's equivalent of a single ALU; an *SM* is the GPU's rough analogue of a CPU
core, restructured to favour many parallel lanes over one fast stream.)

That trade only pays off when there is plenty of independent work to keep the
lanes busy. LLM inference is exactly that case — every matrix multiplication is
thousands of independent multiply-adds — which is why GPUs dominate it.

This doc stops at the fork. The full picture downstream — side-by-side CPU and
GPU chip diagrams, what one SM holds inside, and how the software you write
(threads, blocks, warps) maps onto the lanes — is the subject of
[The GPU execution model](03_gpu_model.md), which opens with this same
chip-area trade and later sets the two chips side by side in "CPU and GPU at
the chip level".

---

Next: [Memory and caches](02_memory_and_caches.md) — what feeds these
circuits when the data does not fit in registers.
