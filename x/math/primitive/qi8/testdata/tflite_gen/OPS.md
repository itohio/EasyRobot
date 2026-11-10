# `tflite_ops.json.gz` — interpreter fixtures for the TFLite operators

Fixtures for `tfliteops_test.go` (LOGISTIC, MUL, MEAN, QUANTIZE int8→int8,
CONCATENATION, PAD/PADV2, RESIZE_NEAREST_NEIGHBOR). Unlike `tflite_ref.json.gz`
there is no port in the loop: `gen_ops.py` draws random inputs (seed
20261110), builds a one-operator `.tflite` model per case with the schema
object API (the `M` builder of `interp_check.py`) and stores the output of
**LiteRT 2.3.0 (`ai-edge-litert`) with `OpResolverType.BUILTIN_REF`** (the
reference kernels). `tflref.py` is used only to *select* inputs on which
candidate arithmetics disagree (float32 vs float64 multiplier, single vs double
rounding, truncating vs rounded 1/n fold, float32 vs float64 logistic table);
it never computes an expected value.

Regenerate (byte-identical; venv with `ai-edge-litert==2.3.0 numpy flatbuffers`):

```sh
cd tflite_gen && .venv/bin/python gen_ops.py /tmp/ops.json && gzip -9 -n -c /tmp/ops.json > ../tflite_ops.json.gz
```

## Contents and results (2026-11-10)

| op | cases | outputs | Go mismatches | discrimination (outputs a wrong variant gets wrong) |
|---|---|---|---|---|
| LOGISTIC | 312 (300 random (scale, zp) + 12 targeted) × 256 inputs | 79 872 | 0 | float64 table: 14 |
| MUL | 300 (111 same-shape, 80 channel, 109 scalar `b`; 75 each of none/relu/relu6/relu_n1_to_1) | 8 044 | 0 | `RoundExact`: 3 349; float64 multiplier: 2 858 |
| MEAN | 250 (H, W ∈ 1…7 incl. 1×1, 7×7, odd H·W; C ≤ 40; keep_dims both; axes [1,2] and [2,1]) | 5 025 | 0 | `RoundExact`: 181; requantise-then-divide: 163; rounded 1/n fold: 44 |
| QUANTIZE | 240 × 256 inputs (ratio <1: 93, =1: 62, >1: 85) | 61 440 | 0 | `RoundExact`: 1 896 |
| CONCATENATION | 120 (2–4 inputs × axis 1–3, all 9 combinations; some negative axes) | 9 400 | 0 | — |
| PAD/PADV2 | 120 (half each; H, W, C padding; some PAD with differing output quantisation) | 24 871 | 0 | — |
| RESIZE_NEAREST_NEIGHBOR | 160 (40 per align_corners × half_pixel_centers; up/down, odd sizes, 2×/3×) | 43 857 | 0 | float64 index math: 87 |

Extra one-off check (scratch, not a fixture): the float32 LOGISTIC table
formula against the interpreter for 3 000 random (scale ∈ 2^[−12, 3], zp)
pairs × 256 inputs: 0 mismatches; the float64 formula: 2.

## Interpreter behaviour found

- LOGISTIC int8: float32 lookup table (`LUTPopulate`, `1/(1+expf(−x))`),
  output fixed at scale 1/256, zp −128.
- MUL int8: `MultiplyByQuantizedMultiplier` (double rounding) of
  `(a−zpA)(b−zpB)`; real multiplier `sA·sB/sOut` computed in float32.
- MEAN int8 (axes H, W; keep_dims either way): reference `QuantizedMeanOrSum`
  with the 1/n factor folded into the multiplier by truncating integer
  division, then `MultiplyByQuantizedMultiplier`.
- QUANTIZE int8→int8: `MultiplyByQuantizedMultiplier` with
  `QuantizeMultiplier(float64(sIn)/float64(sOut))`.
- CONCATENATION int8: prepare fails unless every input's scale and zero point
  equal the output's (`concatenation.cc:235/237`) and the fused activation is
  NONE (`concatenation.cc:143`); the kernel is a copy.
- PAD int8: pads with the output zero point, copies values raw even when the
  input and output quantisation differ; PADV2 rejects a constant whose zero
  point differs from the output's (`pad.cc:450`). Batch padding is accepted
  by TFLite (not supported by `qi8.Pad`).
- RESIZE_NEAREST_NEIGHBOR: `GetNearestNeighbor` in float32; all four flag
  combinations accepted; raw copy even with differing quantisation.

## Mutation checks (each applied alone, tests run, then reverted)

| mutation | caught by |
|---|---|
| LOGISTIC: dequantise in float64 instead of float32 | `TestLogisticTFLite` (3 outputs) |
| MUL: `sA·sB/sOut` in float64 | `TestMulTFLite` (2 858) |
| MEAN: rounded instead of truncating 1/n fold | `TestMeanTFLite` (44), `TestTFLiteOpsHand` |
| MEAN exact: ties toward zero | `TestMeanReference` (4), `TestMeanExactRounding` (44) |
| QUANTIZE: `Round` ignored (always exact) | `TestRequantizeTFLite` (1 896) |
| CONCATENATION: source-major copy (no row interleave) | `TestConcatTFLite` (4 286), `TestTFLiteOpsHand` |
| PAD: Front/Back channel padding swapped | `TestPadTFLite` (2 453), `TestTFLiteOpsHand` |
| RESIZE: index math in float64 | `TestResizeNearestTFLite` (87) |

The truncating-fold mutation was first caught only by the hand test; targeted
MEAN cases (every fifth case) were added so the interpreter data pins it too.

# `tflite_ops2.json.gz` — LEAKY_RELU and RESIZE_BILINEAR

Same method as above (`gen_ops.py --ops2`, seed 20261111; set 1 is
unchanged byte for byte):

```sh
cd tflite_gen && .venv/bin/python gen_ops.py --ops2 /tmp/ops2.json && gzip -9 -n -c /tmp/ops2.json > ../tflite_ops2.json.gz
```

## Contents and results (2026-11-10)

| op | cases | outputs | Go mismatches | discrimination (outputs a wrong variant gets wrong) |
|---|---|---|---|---|
| LEAKY_RELU | 320 (scale, zp, alpha) triples × 256 inputs: 97 equal in/out scales, 223 unequal; 24 with alpha = 0, 71 with alpha > 1; 80 few-bit (tie) cases, 80 targeted float32-vs-float64 cases | 81 920 | 0 | `RoundExact`: 2 577; float64 multipliers: 320; float32 `inScale·(alpha/outScale)`: 7 |
| RESIZE_BILINEAR | 216 (72 per accepted flag combination; 163 upsampling, 94 downsampling on some axis, 97 with a 1-pixel dimension, C up to 40) | 51 875 | 0 (kernel and an independent int64 transcription) | single rounding `(o + 2^19) >> 20`: 1 219 |

The exact-mode `LeakyReLU` is also checked against a big-rational reference
(1 500 random multiplier pairs incl. zero alpha, 384 000 outputs, 0 mismatches).

## Interpreter behaviour found

- LEAKY_RELU int8: `MultiplyByQuantizedMultiplier` (double rounding) with the
  identity multiplier for `x − zpIn ≥ 0`, the alpha multiplier otherwise;
  both multipliers computed in float32 (`inScale/outScale`,
  `(inScale·alpha)/outScale`). Unequal input/output quantisation accepted.
  alpha = 0 and alpha below 2^−32 relative to the scales give the zero
  multiplier (output zero point for negative inputs); alpha > 1 works;
  negative alpha is accepted (negative multiplier; on .5 ties the results do
  not follow `MultiplyByQuantizedMultiplier` with that multiplier, e.g.
  alpha −2.5 gives 125 where the formula gives 126) — unsupported by qi8;
  alpha = NaN outputs the zero point; alpha = +Inf aborts the interpreter
  (exit 134); alpha 3e9 (multiplier ≥ 2^30) is saturated and overflows.
- RESIZE_BILINEAR int8: reference `ResizeBilinearInteger` (10-bit scales,
  `ComputeInterpolationValuesInteger`, int64 four-term sum,
  `(o ± 2^19)/2^20`). align_corners together with half_pixel_centers fails to
  prepare ("If half_pixel_centers is True, align_corners must be False.");
  the other three combinations work. Input and output quantisation may differ;
  raw values are interpolated and copied (no requantisation).

## Mutation checks

| mutation | caught by |
|---|---|
| LEAKY_RELU: alpha multiplier as float32 `inScale·(alpha/outScale)` | `TestLeakyReLUTFLite` (7 outputs) |
| RESIZE_BILINEAR: single rounding `(o + 2^19) >> 20` | `TestResizeBilinearTFLite` (1 219), `TestTFLiteOps2Hand` |
