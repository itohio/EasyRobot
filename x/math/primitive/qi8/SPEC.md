# qi8 — int8 NN kernels (bit-exact reference)

Pure-Go int8 kernels. They define the exact output bytes that accelerated
paths (ESP32-S3 PIE, K210) must reproduce. No allocation in any kernel; errors
are package sentinels returned unwrapped.

## Tensors

| Item | Layout / type | Quantisation |
|---|---|---|
| Activations | NHWC, batch 1, `[]int8` | asymmetric: scale, zero point `zp ∈ [-128,127]` (`int8`) |
| Conv weights | OHWI `[OutC][KH][KW][InC]` | per output channel scale; symmetric (`WZP` 0, the usual int8 case; `-128` allowed) or a per-tensor weight zero point `WZP` (asymmetric, TFLite's uint8 models loaded as int8) |
| Depthwise weights | `[KH][KW][C]`, channel multiplier 1 | same |
| FC weights | `[Out][In]` | same |
| Bias | `[]int32` per output channel, or `nil` (zero) | accumulator scale `sIn·sW[o]` |
| Requant | `[]Requant` per output channel | see below |

## Accumulation

`acc = bias[o] + Σ (x − zpIn)·(w − WZP)` in int32. Padded taps contribute 0 (padding
value is `zpIn`). The reduction length K (taps·inC for conv, taps for
depthwise and pools, In for FC) is limited to `MaxTaps = ⌊(2^31−1)/(255·128)⌋ = 65 793`,
or `MaxTapsWZP = ⌊(2^31−1)/(255·255)⌋ = 33 025` when `WZP ≠ 0` (`|w − WZP| ≤ 255`)
(`ErrShape` above it), so the sum cannot wrap; keeping
`|bias| + K·255·128` (`·255` with `WZP`) `< 2^31` is the caller's responsibility (the bias term is
not checked). The generic kernels evaluate `WZP` as
`Σ (x−zpIn)·w − WZP·Σ (x−zpIn)` (conv, FC: the window sum once per output
pixel or call) or `(x−zpIn)·(w−WZP)` per tap (depthwise), only when
`WZP ≠ 0`; the PIE backend runs layers with `WZP ≠ 0` through the generic
loops.

Accelerated paths may evaluate `Σ x·w − zpIn·Σw` (with the padded-tap
correction) instead of subtracting the zero point per tap: within the
`MaxTaps` bound the integer sums are exact, so the bytes are identical.

## Size limits

Every element count — dimensions, kernel sizes, strides, `H·W·C`,
`outH·outW·C`, weight counts — is computed with overflow-checked
multiplication and limited to `MaxElements = 2^30`, so sizes fit a 32-bit
`int` (ESP32-S3) and a huge shape can never wrap to a small value that passes
the length checks. Violations return `ErrShape` before any slice is touched.

## Requantisation

`Requant{Mult, Shift}`: real multiplier `M' = Mult·2^(Shift−31)`,
`Mult ∈ [2^30, 2^31)`, `Shift ∈ [−31, 30]` (total right shift `s = 31−Shift ∈ [1, 62]`),
or the zero value `{0, 0}`, `M' = 0` (TFLite's encoding of a multiplier
below 2^−32, e.g. a dead channel; the channel outputs its zero point).
Anything else is `ErrRequant` (validated on every call).

```
y = clamp(roundHalfAwayFromZero(acc·Mult / 2^s) + zpOut, actMin, actMax)
```

computed exactly in int64 with a single rounding (`|acc·Mult| < 2^62`).
`Requant.Apply(acc)` exposes the rounded product.

`QuantizeMultiplier(real)`: `real = frac·2^exp`, `frac ∈ [0.5,1)` (frexp);
`Mult = round(frac·2^31)` (half away from zero, exact in float64); if it rounds
to `2^31` then `Mult = 2^30, exp++`; `Shift = exp`. Representable reals:
`[2^−32, 2^30)`; non-positive, NaN, Inf or out-of-range → `ErrRequant`.

### Rounding modes (`Round` field of `Conv2DParams`, `DepthwiseConv2DParams`, `FCParams`, `AddParams`)

- `RoundExact` (zero value): the single exact rounding above.
- `RoundTFLite`: TensorFlow Lite's int8 reference kernels (default build,
  double rounding), for models imported from TFLite. `Requant.ApplyTFLite(acc)`
  = `MultiplyByQuantizedMultiplier`: `x = acc·2^max(Shift,0)` in int32
  (wrapping where C overflows), `h = SaturatingRoundingDoublingHighMul(x, Mult)`
  (`(x·Mult + nudge)/2^31` truncated, nudge `2^30` or `1−2^30`: ties toward
  +∞), then `RoundingDivideByPOT(h, max(−Shift,0))` (ties away from zero).
  `(Mult, Shift)` are the same numbers TFLite stores, and `QuantizeMultiplier`
  equals TFLite's `QuantizeMultiplier` on the representable range, so
  `QuantizeMultiplier(float64(sIn)·float64(sW)/float64(sOut))` (float32
  scales widened, TFLite's order) gives TFLite's per-channel multipliers.
  The modes differ by 1 LSB only near .5 ties — not "rarely" in general: on
  the conv/depthwise fixtures 2 of 2 906 outputs (0.07 %; 0.16 % on an
  earlier fixture set) with multipliers derived from real float scales, 348
  of 5 954 (5.8 %) with few-bit multipliers (m·2^−k), and up to ≈ 25 % at a
  multiplier of exactly 0.5 (every negative exact tie).
- **TFLite's FULLY_CONNECTED rounds once, half away from zero** (per-tensor
  and per-channel weights) — i.e. `RoundExact` — while CONV_2D and
  DEPTHWISE_CONV_2D use the double rounding above. Established against the
  LiteRT 2.3.0 interpreter with the reference resolver (`testdata/README.md`):
  150 of 150 FC cases match single rounding, 129 match double rounding.
  Importers therefore set `RoundExact` on fully connected layers.

An unknown `Round` value is `ErrRequant` (checked after the multipliers).
Pools are mode-independent: `AvgPool2D` already rounds as TFLite's int8
kernel does. `Pool2DParams.TiesUp` selects TFLite's uint8 average pool,
`(Σu + n/2)/n` on uint8 values `u`, which on the `u − 128` int8 form is
`floor((Σq + ⌊n/2⌋)/n)`: it differs from half-away rounding only on negative
`.5` ties (even windows).
`Quantize6TFLite`/`ActReLU6TFLite` compute TFLite's ReLU6 bound
(`zp + roundf(6/scale)`, division in float32), which can differ from
`Quantize6` (float64) on a near-.5 quotient.

## Fused activation

Expressed purely by the clamp range `[ActMin, ActMax]` (`ActMin > ActMax` →
`ErrActivation`; note the zero value `[0,0]` is valid and clamps to 0):

- `ActNone() → [-128, 127]`
- `ActReLU(scale, zp) → [zp, 127]` (`scale` is unused; kept for a signature uniform with `ActReLU6`)
- `ActReLU6(scale, zp) → [zp, Quantize6(scale, zp)]`,
  `Quantize6 = clamp(zp + roundHalfAwayFromZero(6/scale), −128, 127)` (float64 from the float32 scale).

Standalone (same scale/zp in and out, may alias): `ReLU(dst, src, zp)`:
`max(x, zp)`; `ReLU6(dst, src, zp, q6)`: `clamp(x, zp, q6)` (`q6 < zp` → `ErrActivation`).

## Element-wise Add

`Add(dst, a, b, AddParams{ZPA, ZPB, ZPOut, RQ, Min, Max, Round, TFLite})`,
equal lengths (0 allowed, else `ErrLength`), `dst` may alias `a` or `b`.
`RoundExact`:

```
acc = (a−ZPA)·MultA + (b−ZPB)·MultB           (exact int64, |acc| < 2^40)
y   = clamp(roundHalfAwayFromZero(acc / 2^s) + ZPOut, Min, Max),  s = 31−Shift
```

`AddRequant{MultA, MultB, Shift}` is two real multipliers with a shared
shift, `realA = MultA·2^(Shift−31) = scaleA/scaleOut` (likewise B). Valid:
both Mults in `[1, 2^31)`, the larger in `[2^30, 2^31)`, `Shift ∈ [−31, 30]`;
else `ErrRequant`; `Min > Max` → `ErrActivation`. One rounding of the exact
sum, as for `Requant`. `QuantizeAddMultipliers(realA, realB)` quantises the
larger like `QuantizeMultiplier` and the smaller at the same shift
(`round(real·2^(31−Shift))`, half away from zero); a smaller multiplier
that rounds to 0 is `ErrRequant`.

`RoundTFLite` (TFLite `kernels/add.cc` + `AddElementwise`; `RQ` unused):
`AddRequantTFLite{A, B, Out}` from `QuantizeAddTFLite(scaleA, scaleB,
scaleOut float32)` (`A = scaleA/(2·max)`, `B` likewise,
`Out = 2·max/(2^20·scaleOut)`, float64 from the float32 scales); all three
`Valid` with `Shift ≤ 0`, else `ErrRequant`.

```
y = clamp(Out.ApplyTFLite(A.ApplyTFLite((a−ZPA)·2^20) + B.ApplyTFLite((b−ZPB)·2^20)) + ZPOut, Min, Max)
```

## Softmax

`Softmax(dst, src, depth, SoftmaxParams{Mult, LeftShift, DiffMin})` over each
row of `depth` elements (lengths equal and a multiple of `depth`, else
`ErrLength`/`ErrShape`; `dst` may alias `src`): TFLite's int8 reference softmax
(`reference_ops::Softmax`, the TFLite Micro kernel) in gemmlowp fixed point —
`d = x − max(row)`; rows entries with `d < DiffMin` give −128; otherwise
`e = exp_on_negative_values(SRDHM(d·2^LeftShift, Mult))` (Q5.26 → Q0.31),
summed after `RoundingDivideByPOT(e, 12)` (Q12.19), reciprocal by
`GetReciprocal` (three Newton–Raphson steps), output
`RoundingDivideByPOT(SRDHM(scale, e), bitsOverUnit + 23) − 128` clamped.
Output quantisation is fixed at scale 1/256, zero point −128.
`QuantizeSoftmax(beta, inScale)` = TFLite's `PreprocessSoftmaxScaling` +
`CalculateInputRadius` (`kScaledDiffIntegerBits` 5). Valid params: Mult in
range, `LeftShift ∈ [0, 30]`, `DiffMin ≤ 0` with `DiffMin·2^LeftShift ≥ −2^31`
(no overflow); else `ErrRequant`. TFLite's default non-reference CPU kernel
uses a lookup table and can differ by 1.

## TFLite operators (LOGISTIC, MUL, MEAN, QUANTIZE, CONCATENATION, PAD, RESIZE_NEAREST_NEIGHBOR)

Generic-only free functions (no prepared layer, no backend seam); none
allocates or panics. Every formula below was derived from the LiteRT 2.3.0
interpreter with `OpResolverType.BUILTIN_REF` and is checked against its
outputs (`tfliteops_test.go`, `testdata/tflite_ops.json.gz`, provenance in
`testdata/tflite_gen/OPS.md`). For the requantising ops `Round` selects
`RoundExact` (one exact rounding of the same `Requant`, half away from zero)
or `RoundTFLite`; **TFLite uses `RoundTFLite` for MUL, MEAN and QUANTIZE.**

- **LOGISTIC** — `QuantizeLogistic(inScale, inZP) → LogisticParams{Table [256]int8}`,
  `Logistic(dst, src, p)` (`Table[uint8(x)]`; equal lengths, may alias).
  TFLite's int8 LOGISTIC (reference and generic kernels) is a float32 lookup
  table, not gemmlowp fixed point:
  `d = float32(inScale·(x−inZP))`, `t = 1/(1+expf(−d))` in float32,
  `q = clamp(roundf(256·t) − 128)`. Output quantisation is fixed at
  scale 1/256, zero point −128 (TFLite rejects others). `expf` is
  `float32(math.Exp(float64))`; a float64 evaluation differs on rare
  entries (14 of 79 872 fixture outputs). Bad scale → `ErrRequant`.
- **MUL** — `Mul(dst, a, b, MulParams{ZPA, ZPB, ZPOut, RQ, Min, Max, Round})`:
  `dst[i] = clamp(ZPOut + R((a[i]−ZPA)·(b[i mod len(b)]−ZPB)), Min, Max)`,
  `R` = `ApplyTFLite` (TFLite) or `Apply`. `b` broadcasts cyclically: same
  length as `a`, its trailing channel vector, or a scalar; `len(b) = 0` or
  not dividing `len(a)` → `ErrLength`. `dst` may alias `a`.
  `QuantizeMul(sA, sB, sOut)` = `QuantizeMultiplier(float64(float32(float32(sA·sB)/sOut)))`
  — TFLite evaluates the real multiplier **in float32** (a float64 product
  gives different outputs on 2 858 fixture outputs). `Min/Max` from the
  fused activation (TFLite `CalculateActivationRangeQuantized`: `zp +
  roundf(f/scale)`, float32 division, for 0, 6, ±1).
- **MEAN** over H and W — `MeanHW(dst, src, in, MeanParams{InZP, OutZP, RQ, Round})`,
  `dst` is `in.C` long. `S_c = Σ_hw (x − InZP)`, `n = H·W`;
  `H·W > (2^31−1)/255` → `ErrShape` (TFLite's int32 sum). `QuantizeMean(sIn, sOut)`
  = `QuantizeMultiplier(float64(sIn)/float64(sOut))`. `RoundTFLite` is
  TFLite's reference `QuantizedMeanOrSum` (used for keep_dims true and false,
  axes `[1,2]` and `[2,1]`): `1/n` is folded into the multiplier in integers,
  `k = min(⌊log2 n⌋, 32, 31+Shift)`, `m = ⌊Mult·2^k / n⌋` (truncating),
  `e = Shift − k`, `y = MultiplyByQuantizedMultiplier(S_c, m, e)` (double
  rounding; `m` may be below 2^30). `RoundExact`: `round(S_c·M'/n)` once,
  exactly. The older requantise-then-divide `integer_ops::Mean` does not match
  the interpreter.
- **QUANTIZE** int8 → int8 — `Requantize(dst, src, RequantizeParams{InZP, OutZP, RQ, Round})`
  (may alias): `clamp(OutZP + R(x − InZP))`, `QuantizeRequantize(sIn, sOut)` =
  `QuantizeMultiplier(float64(sIn)/float64(sOut))` (equal scales → `{2^30, 1}`,
  exact in both modes).
- **CONCATENATION** — `Concat(dst, srcs, outer)`: each `srcs[k]` is `outer`
  rows; `dst` is row 0 of every source, then row 1, … For NHWC batch 1,
  `outer` = H·W (axis C), H (axis W), 1 (axis H). Pure copy: TFLite's int8
  kernel requires every input's scale and zero point to equal the output's
  and no fused activation (it fails to prepare otherwise).
  `outer ≤ 0` → `ErrShape`; lengths not multiples of `outer` or not summing to
  `len(dst)` → `ErrLength`. `dst` must not overlap a source.
- **PAD / PADV2** — `Pad(dst, src, PadParams{In, Top, Bottom, Left, Right, Front, Back, Value})`,
  `OutShape()`; negative padding or oversize → `ErrShape`. Values are
  copied raw (no requantisation even when input and output quantisation
  differ). PAD's `Value` is the output zero point; PADV2's is its constant
  (TFLite requires the constant's zero point and scale to equal the output's).
  No batch padding (TFLite allows it; the shape has batch 1).
- **RESIZE_NEAREST_NEIGHBOR** — `ResizeNearest(dst, src, ResizeParams{In, OutH, OutW, AlignCorners, HalfPixel})`,
  raw copy. Per axis, in float32:
  `scale = AlignCorners && out > 1 ? (in−1)/(out−1) : in/out`,
  `p = (o + (HalfPixel ? 0.5 : 0))·scale`,
  `i = min(AlignCorners ? roundf(p) : floor(p), in−1)` (and `≥ 0` with
  HalfPixel). All four flag combinations are accepted (as in TFLite);
  float64 index math differs from the interpreter.

### LEAKY_RELU and RESIZE_BILINEAR

Same derivation and verification as above (fixtures
`testdata/tflite_ops2.json.gz`, `gen_ops.py --ops2`).

- **LEAKY_RELU** — `QuantizeLeakyReLU(inScale, outScale, alpha) → (identity, alpha Requant)`,
  `LeakyReLU(dst, src, LeakyReLUParams{InZP, OutZP, Identity, Alpha, Round})`
  (equal lengths, may alias). `v = x − InZP`;
  `y = clamp(OutZP + R(v, v ≥ 0 ? Identity : Alpha))`, `R` = `ApplyTFLite`
  (**TFLite: `RoundTFLite`**) or `Apply`. Multipliers as TFLite's
  `LeakyReluPrepare`, **in float32**: `identity = f32(inScale/outScale)`,
  `alpha = f32(f32(inScale·alpha)/outScale)` (float64, or the association
  `inScale·(alpha/outScale)`, differ from the interpreter). In and out
  quantisation are independent. `alpha = 0`, or a multiplier below 2^−32, is
  the zero `Requant` (negative inputs → `OutZP`), as in TFLite; `alpha > 1`
  is supported. A negative alpha (TFLite accepts it with a negative
  multiplier), a non-finite alpha (TFLite aborts on +Inf) or a multiplier
  ≥ 2^30 → `ErrRequant`.
- **RESIZE_BILINEAR** — `ResizeBilinear(dst, src, ResizeParams)`: TFLite's
  `ResizeBilinearInteger` on raw int8 values (no zero points; TFLite does not
  require or apply equal in/out quantisation). Per axis
  `scale10 = (1024·in + out/2)/out`, or `(1024·(in−1) + (out−1)/2)/(out−1)`
  with AlignCorners and `out > 1`; `s = o·scale10 (+ scale10/2 − 512 with
  HalfPixel)`, cells `max(s/1024, 0)` and `min((s+1023)/1024, in−1)`
  (truncating division), `d = s − 1024·lower`;
  `o = Σ q·w` over the four cells with weights `(1024−dy)(1024−dx)`, …,
  `dy·dx`; `y = (o ± 2^19)/2^20` (sign of `o`, truncating; the single-rounding
  `(o + 2^19) >> 20` differs). AlignCorners together with HalfPixel →
  `ErrShape` (TFLite fails to prepare). Also `ErrShape`: any dimension
  above 2^20, or an upscale so large that the last lower cell passes the
  input (e.g. 2 → 3000; TFLite would read out of bounds).

## Padding

- `PaddingSame` (TF "SAME"): `out = ceil(in/stride)`,
  `padTotal = max((out−1)·stride + k − in, 0)`, `padBefore = padTotal/2`.
- `PaddingValid`: `out = (in−k)/stride + 1`, requires `in ≥ k` (else `ErrShape`).

Height and width are independent (`KH/KW`, `StrideH/StrideW`). No dilation.

## Ops

| Function | Notes |
|---|---|
| `Dot(a, b []int8) int32` | raw `Σ a·b` over `len(a)`; panics if `len(b) < len(a)` |
| `FullyConnected(dst, src, w, bias, rq, FCParams)` | GEMV, `dst[o] = requant(bias[o] + Σ (x−zpIn)·w[o][i])` |
| `Conv2D(dst, src, w, bias, rq, Conv2DParams)` | any `KH×KW`, any stride |
| `DepthwiseConv2D(dst, src, w, bias, rq, DepthwiseConv2DParams)` | any `KH×KW`, any stride |
| `MaxPool2D(dst, src, Pool2DParams)` | max over valid (non-padded) cells |
| `AvgPool2D(dst, src, Pool2DParams)` | `roundHalfAwayFromZero(Σq / count)` over valid cells, exact integer (half up with `TiesUp`) |
| `ReLU`, `ReLU6`, `Add`, `Softmax` | above |
| `QuantizeMultiplier`, `QuantizeAddMultipliers`, `QuantizeAddTFLite`, `QuantizeSoftmax`, `Act*`, `Quantize6`, `Quantize6TFLite` | helpers |
| `(*Params).OutShape()` | output shape for `dst` sizing |

Pools keep scale and zero point. Average pooling rounds in the stored
integer domain (TFLite behaviour): on an exact `.5` tie where `mean(q)` and
`mean(q)−zp` have different signs, this differs by 1 LSB from rounding
`mean(real)/scale` and then adding `zp`.

## Errors

`ErrShape` (non-positive dims/kernel/stride, unknown padding, VALID with
`k > in`, any count above `MaxElements`, reduction length above `MaxTaps` or `MaxTapsWZP`),
`ErrLength` (slice length ≠ shape, bias not `nil`/`OutC`, requant ≠ `OutC`),
`ErrRequant`, `ErrActivation`, `ErrNotPrepared` (prepared layers only).

## Prepared layers and backends

Two tiers:

- **Free functions** (`Conv2D`, …, `ReLU6`, `Dot`) validate on every call
  and always run the generic pure-Go code. They are the reference.
- **Prepared layers** (`Conv2DLayer`, `DepthwiseConv2DLayer`, `FCLayer`,
  `MaxPool2DLayer`, `AvgPool2DLayer`, `ReLULayer`, `ReLU6Layer`; all
  implement `Op{InShape, OutShape, Run}`; `AddLayer` has the same contract
  with two inputs, `Run(dst, a, b)`, so it is not an `Op`): `Prepare()` does all validation
  once and returns exactly the error the free function would (given correctly
  sized `dst`/`src`), caching the resolved geometry. `Run(dst, src)` checks
  only the two lengths (`ErrLength`) and returns `ErrNotPrepared` if the last
  `Prepare` failed, was never called, or an exported field (params or the
  `W`/`Bias`/`RQ`/`Scratch` slice headers) changed since. Contents of `W`,
  `Bias`, `RQ` must not be modified after `Prepare`. `ReLULayer`/`ReLU6Layer`/`AddLayer`
  have shape `{1, 1, Len}` with `Len ∈ [1, MaxElements]`; FC shapes are
  `{1, 1, In}` → `{1, 1, Out}`. `Prepare` and `Run` never allocate.

**Scratch.** Weighted layers take a caller `Scratch` of at least
`ScratchLen()` bytes for backend-repacked/aligned weights; the slice need
not be aligned. The generic backend needs none (`ScratchLen() == 0`,
`Scratch` ignored, nil works); the ESP32-S3 PIE backend needs it (a shorter
`Scratch` is `ErrLength` from `Prepare`). Besides the packed weights,
`Scratch` holds per-`Run` state (gathered patch, accumulators, tap table),
so one prepared layer must not run concurrently with itself.

**In-place weights.** To keep one copy of large weights, copy them into
`InPlaceW()` (the first `len(W)` bytes of the backend's aligned weight
region of `Scratch`; nil when the backend needs no `Scratch`, `P` is invalid
or `Scratch` is short), set `W` to it and call `Prepare`: the backend packs
them in place. While prepared, `W`'s bytes are in the backend layout; the
next `Prepare` restores the original layout first and packs again, so
`Prepare` may be repeated (also after a field change or a failed
`Prepare`). To load new weights, call `Reset` (restores `W`'s original
layout and leaves the layer unprepared), write `W`, then `Prepare`; writing
`W` while prepared is not detected and corrupts it. `W` overlapping the
packed region anywhere but at `InPlaceW()` is `ErrLength`; `W` outside it
is a separate copy and is never modified.

Prepared layers run through a per-op CPU backend chosen **at compile time
only** — build tags, `GOARCH` and the TinyGo target select one `impl_*.go` per
binary; there is no runtime CPU detection, no `x/sys/cpu` dispatch and no
function-pointer switching. `Backend()` names the build: `"generic"`,
`"esp32s3-pie"` (`esp32s3 && !qi8generic`: conv/FC dot products in ACCX,
depthwise in QACC; pools, activations, Add and layers with a weight zero
point generic); `-tags qi8generic`
forces generic on any target. Planned: baseline `arm64` = NEON, `-tags
dotprod` adds SDOT; RVV likewise by tag. Every per-op backend must be
**byte-identical** to the free-function reference in the same rounding mode.
(The rounding mode is a per-layer parameter, not a backend choice.)
The test-only tag `qi8pieemu` builds the PIE backend's Go side on the host
with pure-Go models of its C kernels (`pie_emu.go`, which also asserts the
kernels' alignment and read-ahead preconditions), so `go test -tags
qi8pieemu` runs the layer suite against it; `make qi8-pie` does that and
compiles the PIE build with TinyGo.

Whole-model offload (K210 KPU and similar) is outside this interface: it runs
a compiled model in a model runner (e.g. `drivers/kpu` or a `qnet` backend)
and is compared against the reference within a tolerance, not bit-exactly.

## Verification

`reference_test.go` dequantises to exact reals (512-bit `big.Float`, all
values dyadic, exactness asserted), evaluates the op, quantises with exact
rationals (half away from zero) and clamps. Kernel output must be
bit-identical on seeded random cases (odd sizes, strides 1–3, non-square
kernels, SAME/VALID edges, zp ±128/127, extreme Mult/Shift, saturation),
including asymmetric weights (`TestWeightZeroPointReference`) and the
ties-up average pool (`TestAvgPoolTiesUpReference`).
`TestNoAllocs`/`TestLayerNoAllocs` assert `AllocsPerRun == 0`. TinyGo
`-print-allocs=qi8` is empty for `xiao-esp32s3` and the UnitV target.
`layer_test.go` checks every prepared layer byte-for-byte against its free
function on seeded random cases, and its error/`ErrNotPrepared` behaviour,
on whichever backend the build selects (generic, or the PIE emulation);
`inplace_test.go` covers in-place packing, misplaced `W`, repeated
`Prepare` and weight reload through `Reset`.
