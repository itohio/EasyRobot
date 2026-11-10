# qi8 test data

## `tflite_ref.json.gz` — TFLite-mode fixtures (`tflite_test.go`)

Expected outputs of TensorFlow Lite's int8 reference kernels for random
cases, computed offline by `tflite_gen/gen_fixtures.py` from
`tflite_gen/tflref.py`, and **cross-checked bit for bit against the real
interpreter**: `tflite_gen/interp_check.py` turns every operator case into a
one-operator `.tflite` model and runs it in LiteRT (`ai-edge-litert` 2.3.0,
`OpResolverType.BUILTIN_REF`, i.e. the reference kernels). Result (2026-11-09):
0 mismatches for Conv2D (150 cases, 3 395 outputs), DepthwiseConv2D (150,
5 465), FullyConnected (150, 810), Add (200, 6 612), AveragePool (80, 2 177),
Softmax (300, 7 824) and the fused ReLU6 range (200 models). The 310
requantisation rows test `ApplyTFLite` on arbitrary 31-bit multipliers, which
float32 tensor scales cannot express; they are checked against the port only
and through the Conv2D/depthwise cases.

`tflref.py` is a Python port, in exact integers with C semantics, of:

- `tensorflow/lite/kernels/internal/common.h`: `MultiplyByQuantizedMultiplier`,
  `MultiplyByQuantizedMultiplierSmallerThanOneExp`; gemmlowp
  `SaturatingRoundingDoublingHighMul`, `RoundingDivideByPOT`, the fixed-point
  `exp_on_negative_values` and `one_over_one_plus_x_for_x_in_0_1`
- `quantization_util.cc`: `QuantizeMultiplier`, `PreprocessSoftmaxScaling`,
  `CalculateInputRadius`; `kernel_util.cc`: `CalculateActivationRangeQuantized`
- `reference/integer_ops/` `conv.h`, `depthwise_conv.h`, `add.h`, `pooling.h`
  (AveragePool), `reference/softmax.h`; `kernels/add.cc` (int8 Add parameters,
  left shift 20)
- FULLY_CONNECTED: the interpreter rounds once, half away from zero (per-tensor
  and per-channel weights), unlike the double rounding of Conv2D. The port
  first used double rounding; 11 of 810 FC outputs differed from the
  interpreter, and single rounding matches all 150 cases, so the port follows
  the interpreter.

Contents: 310 requantisation rows (every shift, 5 multipliers, 136
accumulators), 150 Conv2D, 150 DepthwiseConv2D, 150 FullyConnected (half with
per-channel weights), 200 Add, 80 AveragePool, 200 ReLU6 ranges and 300
Softmax cases. Every Conv2D/depthwise/FC case derives its multipliers from
float32 scales the TFLite way; half use few-bit multipliers (power-of-two
input/output scales), a third have inputs near the zero point, so .5 ties
(where the rounding modes differ) are common.

Regenerate (deterministic, Python 3 + numpy):

```sh
cd tflite_gen && python3 gen_fixtures.py /tmp/t.json && gzip -9 -n -c /tmp/t.json > ../tflite_ref.json.gz
```

Cross-check (venv with `ai-edge-litert==2.3.0 numpy flatbuffers`):

```sh
cd tflite_gen && python gen_fixtures.py /tmp/t.json && python interp_check.py /tmp/t.json
```
