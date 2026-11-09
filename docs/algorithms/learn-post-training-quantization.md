---
type: algorithm
status: partial
tags: [easyrobot, learn, nn]
workstream: [W9, W5]
module: "[[learn]]"
summary: "5 pass, TestQuantizeModel skipped; per-channel/KL missing"
source: [x/math/learn/quantization.go]
tests: [x/math/learn/quantization_test.go]
test_result: pass
complete: no
correct: no
evidence: "5 quant tests pass, TestQuantizeModel skipped (quantization_test.go:145); probe shows per-channel unsupported, KL is percentile, output stored as FP32 (2026-11-09)"
---
# learn post-training quantization (affine int8/uint8)

**What it does:** calibrates a float range (min-max / percentile / "KL"), derives affine params `real = scale·(q − zp)`, quantizes/dequantizes tensors, and computes per-parameter params for a model.

**Where:** `x/math/learn/quantization.go` — `Calibrator` :57-69, `NewCalibrator` :117-147, `AddSample/AddTensor` :150-180, `ComputeParams` :183-249, `computeParamsKL` :253-290, `computePercentileRange` :293-323, `QuantizeTensor` :327-360, `DequantizeTensor` :363-381, `QuantizeModel` :386-419. Options `WithScheme/WithBits/WithCalibrationMethod/WithPercentile` :89-114.

**Inputs → outputs:**
- Range: symmetric (and anything not asymmetric) → signed `[-2^(b-1), 2^(b-1)-1]` e.g. [-128,127]; asymmetric → `[0, 2^b-1]` (:119-126).
- Symmetric: `scale = max(|min|,|max|)/qmax`, `zp=0` (:227-229) — standard (TFLite-style, -128 unused on the positive side).
- Asymmetric: `scale=(max−min)/(qmax−qmin)`, `zp = qmin − int32(min/scale)` (:233-234) — truncation, not `round`; zp not clamped to [qmin,qmax]; range not widened to include 0, so for all-positive data zp < 0.
- Quantize: `q = clamp(round(x/scale)+zp)` (:346-355). Dequantize: `scale·(q−zp)` (:376).

## Completeness review

- Complete: **no**.
  - `QuantPerChannel`/`QuantPerTensor` are enum values (:26,29) but `ComputeParams` returns `unsupported scheme` for them (:236-237; probe: `perchannel err: quantization: unsupported scheme 2`). Per-channel is not implemented; scheme and granularity are conflated in one enum.
  - `CalibKLDivergence` is not KL: computes min/max then discards it and uses the percentile range (:270-273). Comment claims TFLite algorithm (:252).
  - `QuantizeTensor` stores quantized ints in a `DTFP32` tensor (:341) → no memory saving; nothing emits int8 storage or an int8 inference path.
  - `QuantizeModel` only computes params per parameter (key = `fmt.Sprintf("%v", paramIdx)`, :414); it neither quantizes weights nor calibrates activations.
- Tested: `TestQuantizationParams` (symmetric scale = r/127, zp=0), `TestQuantizationAsymmetric` (scale), `TestQuantizeDequantize` (reconstruction error ≤ scale bound), `TestQuantizationPercentile` (scale <1 with outliers), `TestQuantizationZeroRange` (scale>0 only) — pass. **`TestQuantizeModel` is skipped: `t.Skip("Requires actual model implementation to test fully")` at quantization_test.go:145** — placeholder, body empty. No tests for per-channel, KL, asymmetric zp value, clamping.
- Correct: **no** for asymmetric zero-point (truncation instead of round; probe [-1.003,2] gives zp=85 which happens to match round, but e.g. min/scale=−85.6 → 85 instead of 86, biasing exact-zero representation) and for the advertised KL/per-channel modes. Symmetric path correct for its scope.

## Other defects

- Zero-range branch (:213-219): `scale=1`, zp = midpoint (127 for uint8). Constant tensors with |value| > ~127 or < 0.5 lose all precision (probe: constant 5 round-trips only because 5 is an integer within range).
- `computePercentileRange` uses insertion sort O(n²) (:298-309) and copies all samples; `AddSample` stores every value for percentile/KL (:163) — unbounded memory, unusable on-device and slow for real layers (use histogram).
- `NewCalibrator` has duplicated identical branches (:138-144); `bits` not validated (bits ≤0 or ≥32 overflow).
- `QuantizeTensor` uses `At/SetAt` with index iterator per element (:344-356) — slow, allocates index slices.

## TODOs

- [ ] Implement real per-channel quantization (separate granularity from scheme) — x/math/learn/quantization.go:24-29,236 #p2
- [ ] Round (and clamp) asymmetric zero point; nudge range to include 0 — x/math/learn/quantization.go:234,283 #p1 #backlog
- [ ] Replace fake KL with histogram KL (TensorRT/TFLite style) or remove the mode — x/math/learn/quantization.go:270-273 #p2
- [ ] Store quantized tensors as int8/uint8 DType — x/math/learn/quantization.go:341 #p2
- [ ] Implement and test `QuantizeModel` (weights + activation calibration); unskip test — x/math/learn/quantization_test.go:145 #p2
- [ ] Histogram-based percentile instead of storing/sorting all samples — x/math/learn/quantization.go:163,298 #p2
- [ ] Fix zero-range params (scale from |value|, zp=0 symmetric) — x/math/learn/quantization.go:213-219 #p3

## Links

- Module: [[learn]] · [[tensor]] · [[W9-go-nn-framework]] · [[W5-feature-tensor-experiments]]
