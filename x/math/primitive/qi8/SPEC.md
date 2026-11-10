# qi8 — int8 NN kernels (bit-exact reference)

Pure-Go int8 kernels. They define the exact output bytes that accelerated
paths (ESP32-S3 PIE, K210) must reproduce. No allocation in any kernel; errors
are package sentinels returned unwrapped.

## Tensors

| Item | Layout / type | Quantisation |
|---|---|---|
| Activations | NHWC, batch 1, `[]int8` | asymmetric: scale, zero point `zp ∈ [-128,127]` (`int8`) |
| Conv weights | OHWI `[OutC][KH][KW][InC]` | symmetric (zp 0), per output channel scale; `-128` allowed |
| Depthwise weights | `[KH][KW][C]`, channel multiplier 1 | same |
| FC weights | `[Out][In]` | same |
| Bias | `[]int32` per output channel, or `nil` (zero) | accumulator scale `sIn·sW[o]` |
| Requant | `[]Requant` per output channel | see below |

## Accumulation

`acc = bias[o] + Σ (x − zpIn)·w` in int32. Padded taps contribute 0 (padding
value is `zpIn`). The reduction length K (taps·inC for conv, taps for
depthwise and pools, In for FC) is limited to `MaxTaps = ⌊(2^31−1)/(255·128)⌋ = 65 793`
(`ErrShape` above it), so `Σ (x − zpIn)·w` cannot wrap; keeping
`|bias| + K·255·128 < 2^31` is the caller's responsibility (the bias term is
not checked).

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
`Mult ∈ [2^30, 2^31)`, `Shift ∈ [−31, 30]` (total right shift `s = 31−Shift ∈ [1, 62]`).
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
Pools are mode-independent: `AvgPool2D` already rounds as TFLite does.
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
| `AvgPool2D(dst, src, Pool2DParams)` | `roundHalfAwayFromZero(Σq / count)` over valid cells, exact integer |
| `ReLU`, `ReLU6`, `Add` | above |
| `QuantizeMultiplier`, `QuantizeAddMultipliers`, `QuantizeAddTFLite`, `Act*`, `Quantize6`, `Quantize6TFLite` | helpers |
| `(*Params).OutShape()` | output shape for `dst` sizing |

Pools keep scale and zero point. Average pooling rounds in the stored
integer domain (TFLite behaviour): on an exact `.5` tie where `mean(q)` and
`mean(q)−zp` have different signs, this differs by 1 LSB from rounding
`mean(real)/scale` and then adding `zp`.

## Errors

`ErrShape` (non-positive dims/kernel/stride, unknown padding, VALID with
`k > in`, any count above `MaxElements`, reduction length above `MaxTaps`),
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
depthwise in QACC; pools, activations and Add generic); `-tags qi8generic`
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
kernels, SAME/VALID edges, zp ±128/127, extreme Mult/Shift, saturation).
`TestNoAllocs`/`TestLayerNoAllocs` assert `AllocsPerRun == 0`. TinyGo
`-print-allocs=qi8` is empty for `xiao-esp32s3` and the UnitV target.
`layer_test.go` checks every prepared layer byte-for-byte against its free
function on seeded random cases, and its error/`ErrNotPrepared` behaviour,
on whichever backend the build selects (generic, or the PIE emulation);
`inplace_test.go` covers in-place packing, misplaced `W`, repeated
`Prepare` and weight reload through `Reset`.
