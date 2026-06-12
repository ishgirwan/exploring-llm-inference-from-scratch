# Numerical types in ML

LLM inference uses a zoo of floating-point and integer types. Picking the wrong
one trades silently between speed, memory, and correctness. This doc covers
what each type is, why it exists, and where the gotchas are.

Prerequisites: [GPU architecture §5 Tensor Cores](../01_hardware_fundamentals/04_gpu_architecture.md#5-tensor-cores).

## How a floating-point number is stored

An IEEE 754 floating-point number has three parts:

```
sign  exponent  mantissa
 1       E         M
```

- **Sign** — 1 bit; positive or negative.
- **Exponent** — sets the *range* (how big or small the number can be).
- **Mantissa** — sets the *precision* (how many distinct values exist within
  the range).

A wider exponent extends the range. A wider mantissa adds precision. Total bits
= 1 + E + M.

IEEE 754 is the international standard that defines this encoding and the
rules for rounding, special values, and arithmetic behaviour. Every dtype in
the table below follows it (with one caveat for TF32, noted later).

The bits actually represent this value:

```
value = (−1)^sign × 1.mantissa × 2^(exponent − bias)        (normal values)
```

(This is the formula for *normal* values. *Subnormal* values, used for very
tiny numbers near zero, drop the implicit leading `1` and use a fixed minimum
exponent — they get a brief mention further down, when special values come up.)

What each piece does:

- `(−1)^sign` flips the sign. Sign bit 0 → positive; sign bit 1 → negative.
- `1.mantissa` reads as "binary 1, point, then the mantissa bits." The leading
  `1` is *implicit* — never stored — which buys one extra bit of precision for
  free. So a 23-bit mantissa actually represents 24 bits of significand.
- `2^(exponent − bias)` is the scale. The stored exponent field is always
  non-negative, but real exponents must reach negative values to represent
  small numbers. So the field stores `true_exponent + bias`, and you subtract
  the bias to recover the true exponent.

The bias depends on the exponent field width:

```text
fp32   exponent field = 8 bits   bias = 127
fp16   exponent field = 5 bits   bias = 15
bf16   exponent field = 8 bits   bias = 127  (same as fp32)
```

This is also what justifies the `~decimal digits` column in the table below:
a mantissa with `M` bits gives roughly `log10(2^(M+1))` decimal digits of
precision (the `+1` is the implicit leading `1`). For fp32 that is
`log10(2^24) ≈ 7`; for fp16 `log10(2^11) ≈ 3`; for bf16 `log10(2^8) ≈ 2`.

IEEE 754 also defines a few special values that come up later:

- `±∞` — produced by overflow or division by zero.
- `NaN` ("not a number") — produced by undefined operations like `0 / 0`,
  `∞ − ∞`, `sqrt(-1)`. NaN propagates: any arithmetic involving a NaN returns
  a NaN, which is why a single bad value can poison a whole tensor.
- *Subnormal* (or *denormal*) values — very tiny numbers where the implicit
  leading `1` is dropped, used to fill the gap between the smallest normal
  value and zero. They are the reason fp16 silently underflows to zero so
  easily: once you exceed the subnormal range, the only representable value
  is `0`.

## The dtypes

| dtype | bits | sign | exp | mantissa | range | ~decimal digits |
| --- | --- | --- | --- | --- | --- | --- |
| fp64 | 64 | 1 | 11 | 52 | ±10³⁰⁸ | ~15 |
| fp32 | 32 | 1 | 8 | 23 | ±10³⁸ | ~7 |
| tf32 | 19* | 1 | 8 | 10 | ±10³⁸ | ~4 |
| fp16 | 16 | 1 | 5 | 10 | ±10⁵ | ~3 |
| bf16 | 16 | 1 | 8 | 7 | ±10³⁸ | ~2 |
| fp8 (e4m3) | 8 | 1 | 4 | 3 | ±448 | ~1 |
| fp8 (e5m2) | 8 | 1 | 5 | 2 | ±10⁴ | ~1 |
| int8 | 8 | 1 | — | 7 | ±127 | — |
| int4 | 4 | 1 | — | 3 | ±7 | — |

\* TF32 is stored in 32 bits but only 19 are functional — see below.

## fp32, fp16, bf16: the central trio

Visually side by side, with one character per bit:

```
         sign  exponent              mantissa
         ──    ───────────────       ─────────────────────────────
  fp32   |S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM|       range ±10³⁸   ~7 digits
  fp16   |S|EEEEE|MMMMMMMMMM|                       range ±10⁵    ~3 digits
  bf16   |S|EEEEEEEE|MMMMMMM|                       range ±10³⁸   ~2 digits
```

bf16 keeps fp32's 8-bit exponent (same range, no overflow surprises) and gives
up mantissa width to fit into 16 bits. fp16 splits the bits the other way:
narrow exponent, wider mantissa than bf16.

**fp32** is the default for training and the gold standard for correctness.
Most kernel "is this right?" checks compare against an fp32 reference.

**fp16** halves the memory cost of fp32 and runs ~2× faster on Tensor Cores
from Volta onward. Its exponent is much narrower (5 bits vs 8), so values
larger than ~65k or smaller than ~6×10⁻⁸ overflow or underflow. This causes
training to diverge unless mitigated by loss scaling and causes activations in
large models to clip silently.

**bf16** ("brain float 16") is Google's response: keep fp32's 8-bit exponent
(same range, no overflow problems), spend the saved bits on the mantissa
instead. Same dynamic range as fp32 but only ~2 decimal digits of precision.
Modern LLMs are trained almost exclusively in bf16 because the range issues of
fp16 don't appear and the lost precision is invisible at the scales involved.

The implication for benchmarking: bf16 results cannot match fp32 to better than
~1e-2. If they do, something is silently upcasting. See
[Benchmarking §correctness](../04_measurement/01_benchmarking.md#correctness).

## tf32: the special case

TF32 ("TensorFloat-32") is Ampere-only and exists for one purpose — making
fp32 matmuls faster without changing user code. It uses fp32's 8-bit exponent
and fp16's 10-bit mantissa, stored in 32 bits with the remaining 13 bits
unused. On Ampere Tensor Cores an fp32 matmul is silently performed in TF32 by
default, ~8× faster than a true fp32 matmul on the same Tensor Cores and
usually accurate enough for ML.

Toggle: `torch.backends.cuda.matmul.allow_tf32 = True/False`.

## fp8: e4m3 and e5m2

fp8 comes in two flavours chosen for different uses:

- **e4m3** — 4-bit exponent, 3-bit mantissa. Smaller range (±448), more
  precision. Used for activations and weights in inference.
- **e5m2** — 5-bit exponent, 2-bit mantissa. Larger range (±10⁴), less
  precision. Used for gradients in training.

fp8 requires compute capability ≥ 8.9 (Ada) or 9.0 (Hopper). Hopper's
Transformer Engine dynamically picks scales and switches between fp8 and fp16
per layer; without it, fp8 is harder to use correctly.

## Integer types: int8, int4

Integer quantization stores weights, and sometimes activations, as small
integers with a separate floating-point **scale** (and often a **zero point**)
per group of values:

```
real_value ≈ scale × (int_value − zero_point)
```

A weight matrix in fp16 might be 1 GB; the same matrix int8-quantized with one
scale per 128 values is ~280 MB. The win is memory bandwidth: the linear
layer (a fully-connected layer in a neural network, which at inference time
is just `output = input @ weights + bias`) reads four times less from VRAM.

Variants seen in LLM inference:

- **Weight-only int8 / int4** — weights quantized; activations stay in
  fp16/bf16. Easy, modest speedup.
- **Activation-weight int8 (W8A8)** — both quantized. Larger speedup, harder to
  get right because activations are dynamic.
- **GPTQ, AWQ** — algorithms that choose scales and groupings to minimize
  accuracy loss.

Module M15 covers this in detail.

## Accumulator precision

A matrix multiply `C = A @ B` reads inputs in one dtype and produces outputs in
another, but internally accumulates a running sum in a *third*, usually
higher-precision dtype. This is essential: if the accumulator matched the input
dtype, even a moderately sized matmul would lose precision catastrophically.

Standard practice on Tensor Cores:

| Inputs | Accumulator | Output |
| --- | --- | --- |
| fp16 | fp32 | fp16 or fp32 |
| bf16 | fp32 | bf16 or fp32 |
| fp8 | fp16 or fp32 | fp16, bf16, or fp32 |
| int8 | int32 | int8 or fp16 |

So "the matmul ran in bf16" usually means inputs and output are bf16 but the
accumulator is fp32. Custom kernels that skip this — accumulating in bf16 to
save registers — typically fail correctness tests at moderate matrix sizes.

## Dtype support by GPU

Recap of [GPU architecture §5](../01_hardware_fundamentals/04_gpu_architecture.md#5-tensor-cores)
from the dtype angle:

- **fp32 / fp16** — every GPU since Volta (CC 7.0).
- **int8 / int4** — Turing and newer (CC 7.5+).
- **bf16 / tf32** — Ampere and newer (CC 8.0+).
- **fp8** — Ada and Hopper and newer (CC 8.9+, 9.0+).
- **fp4** — Blackwell and newer (CC 10.0+).

This is why the project's Colab T4 (CC 7.5) is restricted to early modules:
bf16 doesn't exist on Turing, and most modern LLMs are bf16-trained.

## What this implies for correctness tolerances

The tolerances in
[Benchmarking §correctness](../04_measurement/01_benchmarking.md#correctness) come directly
from the mantissa widths above. A dtype with ~3 decimal digits of precision
(fp16) cannot match a reference to better than ~1e-3 absolute. A dtype with ~2
(bf16) cannot match better than ~1e-2. Tighter tolerances are not "more careful
testing" — they are indications of a bug.

## Further reading

The references behind each format and the precision rules — primary sources, plus
one tool that makes the bit layout tangible:

- **[What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)**
  (David Goldberg, 1991) — the classic deep dive into IEEE 754, rounding, and where
  precision is lost; the foundation under "how a floating-point number is stored."
- **[Float Toy](https://evanw.github.io/float-toy/)** (Evan Wallace) — an interactive
  bit-level playground: flip sign / exponent / mantissa bits and watch the value
  change. The fastest way to make the three-part split tangible.
- **[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)** (Micikevicius
  et al., 2022) — the NVIDIA/Arm/Intel paper defining the e4m3 and e5m2 encodings from
  the fp8 section, and why e4m3 bends the IEEE rules to buy range.
- **[Train With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)**
  (NVIDIA) — the practical rules behind accumulator precision: which operations must
  stay in fp32, and why bf16/fp16 inputs with an fp32 accumulator is the safe default.
- **[Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)**
  (Rouhani et al., 2023) — the open standard behind block-scaled fp8/fp4 (MXFP8,
  NVFP4); where the dtype table is heading on Blackwell, picked up again in M28.
