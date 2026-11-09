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

TFLite computes `RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(acc, Mult), −Shift)`
(two roundings); it can differ from this single rounding by 1 LSB on rare
ties. A TFLite-compatible mode is a possible later option.

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

`Add(dst, a, b, AddParams{ZPA, ZPB, ZPOut, RQ, Min, Max})`, equal lengths (0
allowed, else `ErrLength`), `dst` may alias `a` or `b`:

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
that rounds to 0 is `ErrRequant`. TFLite's Add (`left_shift` 20, three
doubling-high-muls) is a different rounding; it belongs to the
TFLite-compatible mode.

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
| `QuantizeMultiplier`, `QuantizeAddMultipliers`, `Act*`, `Quantize6` | helpers |
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
  `{1, 1, In}` → `{1, 1, Out}`. Weighted layers take an optional caller
  `Scratch` of `ScratchLen()` bytes for backend-repacked/aligned weights
  (`0` and ignored for the generic backend); `Prepare` and `Run` never
  allocate.

Prepared layers run through a per-op CPU backend chosen at build time
(`impl_*.go`; `Backend()` names it: `"generic"`, later `"esp32s3-pie"`,
RVV, …; `-tags qi8generic` forces generic). Every per-op backend must be
**byte-identical** to the free-function reference in the same rounding mode.

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
function on seeded random cases, and its error/`ErrNotPrepared` behaviour.
