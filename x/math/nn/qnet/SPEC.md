# qnet — int8 network runner on qi8

Tiny graph + runner for int8 CNNs on microcontrollers (ESP32-S3, K210) and
the host. Imports only the standard library and `x/math/primitive/qi8` (never
`x/math/nn` or `x/math/tensor`). Planning and `NewRunner`/`Init` allocate
once; `Run` never allocates. Errors are sentinels returned unwrapped.
Numerics are exactly qi8's (see `qi8/SPEC.md`), including its rounding modes.

## Data

| Type | Meaning |
|---|---|
| `Quant{Scale float32; ZP int8}` | activation quantisation, `real = Scale·(q − ZP)` |
| `Kind` | `Conv2D`, `DepthwiseConv2D`, `FullyConnected`, `MaxPool2D`, `AvgPool2D`, `ReLU`, `ReLU6`, `Add`, `Reshape`, `Softmax`, `Logistic`, `Mul`, `Mean`, `Requantize`, `Concat`, `Pad`, `ResizeNearest`, `Const`, `LeakyReLU`, `ResizeBilinear` |
| `Layer` | plain data: `Kind`, `From []int`, `In/Out qi8.Shape`, `InQ/OutQ/In2Q Quant`, geometry `KH, KW, StrideH, StrideW, Padding`, `TiesUp` (AvgPool2D: round ties up, TFLite's uint8 pool), fused clamp `ActMin/ActMax`, `Round qi8.Rounding`, `Weights []int8` (Const: its data), `WZP int8` (weights' zero point, 0 = symmetric), `Bias []int32` (nil = 0), `Requant []qi8.Requant`, `AddRQ`/`AddTFLite`, `SoftmaxParams`, `LogisticParams`, `RQ qi8.Requant` (Mul/Mean/Requantize, LeakyReLU identity), `RQAlpha` (LeakyReLU alpha), `Axis` (Concat: 0 H, 1 W, 2 C), `Pads [6]int` + `PadValue`, `AlignCorners`/`HalfPixel` (both resizes) |
| `Graph{Layers []Layer; Outputs []int}` | layers in execution order; layer i reads the layers in `From`; `Outputs` are the graph outputs (nil: the last layer) |

`From` lists producers: indices below the layer, or −1 for the graph input;
nil means the previous layer (the input for layer 0). Single-input kinds take
at most one entry, `Add` and `Mul` exactly two, `Concat` one or more,
`Const` none (and it must not be layer 0). The graph input's
shape/quantisation are `Layers[0].In/InQ`; the graph outputs are the
`Outputs` layers' `Out` (`NumOutputs()`), by default the last layer.

The qi8 parameter structs are derived, never stored: `Conv2DParams()`,
`DepthwiseParams()`, `FCParams()` (`In = In.Len()`, `Out = Out.C`),
`PoolParams()`, `AddParams()`, `MulParams()` (zero points from `InQ`,
`In2Q`, `OutQ`), `MeanParams()`, `RequantizeParams()`, `PadParams()`,
`ResizeParams()`, `LeakyReLUParams()`.
`Round` reaches every requantising kernel (`qi8.RoundTFLite` for imported
TFLite models). `Out.C` is the channel count of Conv2D/FC; the `Out` of
`Reshape` (same length), `Concat` and `Const` and `Out.H/W` of the resizes
are given; everything else in `Out` is implied (`OutShape()`).
Global average pooling is `AvgPool2D` with `KH×KW = In.H×In.W`, VALID.
Standalone `ReLU6` uses `qi8.Quantize6(InQ.Scale, InQ.ZP)`. `Add` is
`qi8.Add` with `AddRQ` (`RoundExact`) or `AddTFLite` (`RoundTFLite`);
`Softmax` is `qi8.Softmax` over C with `OutQ = SoftmaxOutQuant {1/256, −128}`.
`Logistic` is `qi8.Logistic` (`OutQ = LogisticOutQuant {1/256, −128}`);
`Mul` is `qi8.Mul` (second input `In`-shaped, `1×1×In.C` or `1×1×1`,
broadcast cyclically); `Mean` is `qi8.MeanHW` (`Out = 1×1×C`); `Requantize`
is `qi8.Requantize`; `Concat` is `qi8.Concat` with `outer` = the product of
the dimensions before `Axis`; `Pad` is `qi8.Pad`; `ResizeNearest` and
`ResizeBilinear` are `qi8.ResizeNearest`/`qi8.ResizeBilinear`; `LeakyReLU` is
`qi8.LeakyReLU`; `Const` outputs its `Weights`.
Weight layouts are qi8's (OHWI, `[KH][KW][C]`, `[Out][In]`); `WZP` reaches
the weighted kernels' `qi8` params and `TiesUp` `PoolParams()`. An
accelerator backend that claims a weighted layer must honour `WZP` (or
decline layers with `WZP ≠ 0`).

## Validation (`Graph.Check() (layer int, err error)`, `Validate()`)

Per layer: known kind (`ErrKind`); `From` in range, earlier than the layer,
the right count (`ErrShape`); `Out == OutShape()` and valid geometry
(`ErrShape`); positive finite scales (`ErrQuant`); `In`/`InQ` (and the second
input's shape/`In2Q` for Add and Mul) equal the producers' `Out`/`OutQ`
(`ErrShape`/`ErrQuant`); `Concat` inputs agree with `In` off `Axis`, sum to
`Out` on it and are quantised as `OutQ`; `Const` has `In = Out`, `InQ =
OutQ`; pools, ReLUs and Reshape keep quantisation (Pad and the resizes work on raw values under any `OutQ`, as TFLite; LeakyReLU, Mean and Requantize requantise) and
Softmax/Logistic output `SoftmaxOutQuant`/`LogisticOutQuant` (`ErrQuant`);
`ActMin ≤ ActMax` (`ErrActivation`); `len(Weights)`, `len(Requant)` equal
`ParamCounts` (Const: `Out.Len()`), `Bias` nil or equal, all requants (and
`RQ`/`RQAlpha`, which may also be zero) `Valid`, valid `Round`, Add multipliers and
softmax/logistic scaling (qi8 validation), no weights or `WZP` on other
parameter-free kinds, `TiesUp` only on `AvgPool2D` (`ErrParams`); `Outputs` non-empty and in range
(`ErrShape`, index −1). Empty graph: `ErrEmpty`.

## Parameters

`Layer.ParamCounts()` / `ParamCounts(g)` give weight, bias and requant entry
counts (Conv2D `OutC·KH·KW·InC, OutC, OutC`; depthwise `KH·KW·C, C, C`; FC
`Out·In, Out, Out`). `BindParams(g, w, b, rq)` slices three backing arrays of
exactly those lengths into the layers in order (3-index slices, no copy);
`b` may be nil.

## Budget

`Graph.Budget(i) LayerBudget{In, Out, Weights, Params, MACs}`: activation
bytes, weight bytes, bias (4 B) + requant (8 B) bytes, nominal MACs including
padded taps (= weights × output pixels; other kinds 0). Totals:
`WeightBytes` (Const data counts as weights), `ParamBytes`, `MACs`.

## Plan (`PlanMemory(g, backends...) (*MemPlan, error)`)

Validates shapes (parameters need not be bound), then:

- **Segments.** At each layer the backends are asked in order
  (`Claim(layers) int`); the first non-zero claim becomes a segment. A claim
  must be a chain of single-input layers, each after the first reading its
  predecessor, with no intermediate output read outside the chain or
  returned by the graph (`ErrClaim`, also for counts < 0 or past the end).
  Unclaimed layers run alone on the built-in CPU kernels.
- **Persistent region** (arena start): the qi8 backend scratch of each CPU
  conv/depthwise/FC layer (`ScratchLen`, non-zero only for the ESP32-S3 PIE
  build), 16-byte aligned. `ScratchSlice(arena, i)` returns it; placing the
  layer's weights at its start before `NewRunner` lets qi8 pack them in place.
- **Activations.** Every segment output is a tensor alive from its segment
  to the last segment that reads it (graph outputs to the end); Reshape
  outputs alias their input (a Reshape of the graph input aliases the
  caller's buffer, `Out[i].Off = −1`); Const outputs (and Reshapes of them)
  are the Const's `Weights` (`Off = −2`). Tensors are placed greedily, largest
  first (ties: earlier first), at the lowest 16-aligned offset that overlaps
  no placed tensor with an overlapping lifetime. A tensor never shares bytes
  with a live input of its segment.
- `MemPlan{Segments, Scratch, Out []Region, Persistent, Activations}`;
  `Size() = 15 + Persistent + Activations` (slack to align any buffer).

For sequential graphs the plan equals the ping-pong bound
`max(In+Out)` (rounded to 16): MobileNetV1-0.25 needs 55 296 B of
activations at 96², 98 304 B at 128², 301 056 B at 224².

## Runner

`NewRunner(g, arena, backends...)` / `Init` validates `g` (parameters bound),
plans it, checks `len(arena) ≥ Size()` (`ErrArena`) and prepares every CPU
layer as a qi8 prepared layer over its scratch (allocating the plan and the
layer structs once). `Run(input)` executes all segments (`ErrNotReady`
before a successful `Init`, `ErrInput` on a length mismatch) and returns
graph output 0, aliasing the arena until the next `Run`; `Output(input, k)`
returns output k. `RunSegment(input, k)` and `SegmentOutput(input, k)` run
and expose one segment (profiling).

```go
type Backend interface {
    Name() string
    Claim(layers []Layer) int                  // chain run from layers[0] as one segment; 0 = none
    Run(layers []Layer, dst, src []int8) error // NHWC in (first In) → out (last Out); no allocation
}
```

`RunLayer(l, dst, srcs...)` runs one layer with the qi8 free functions
(the reference the Runner is tested against; `srcs` are the producers'
outputs in `From` order, none for Const).

## MobileNetV1

`MobileNetV1(alpha, H, W, classes, layers)` fills
`layers[:MobileNetV1Layers(classes)]` (28, or 29 with FC): conv 3×3/2 →
`int(32α)`, 13 × (depthwise 3×3 + pointwise 1×1) with pointwise channels
`int(α·{64, 128/2, 128, 256/2, 256, 512/2, 512×5, 1024/2, 1024})` (`/2` =
depthwise stride 2), SAME padding, ReLU6 fused into every conv, global
average pool, optional FC to `classes` (no activation). Quantisation:
input `MobileNetInputQuant {1/128, 0}`, activations `MobileNetActQuant
{6/255, −128}`, logits `MobileNetLogitQuant {1/16, 0}`. Params are left nil.

`SyntheticParams(g, seed)` fills bound params deterministically (SplitMix64):
weights uniform in [−127, 127]; per channel requant `48/σ` jittered ±25 %,
`σ = sqrt(fanIn)·73.3·xRMS` (xRMS 73.9 for the first layer, `48/√2` after);
bias uniform in ±σ/4. Result: every conv output ≈ half at the ReLU floor,
< 1 % at the ReLU6 ceiling.

α = 0.25: 207 600 weight bytes (+256 000 for FC-1000), 2 736 (+1 000) bias
and requant entries; MACs 7.49 M at 96², 13.31 M at 128², 41.0 M at 224² with
FC-1000 (the paper's 41 M).

## TFLite import (`qnet/tfl`)

`tfl.Load(model []byte) (Graph, error)` converts the first subgraph of an
int8 TFLite flatbuffer (pure Go, no reflection, bounds-checked: malformed
input is `ErrFormat`, never a panic). Conv2D, depthwise and Add get
`qi8.RoundTFLite`, FullyConnected `qi8.RoundExact` (TFLite's FC kernel
rounds once, half away from zero); multipliers are `QuantizeMultiplier(float64(sIn)·
float64(sW)/float64(sOut))` per channel (below 2^-32: the zero multiplier, as
TFLite); fused ReLU/ReLU6/ReLU_N1_TO_1 ranges use TFLite's float32 rounding;
Add uses `QuantizeAddTFLite`, Softmax `QuantizeSoftmax`. MUL, MEAN and
QUANTIZE (int8 → int8) get `qi8.RoundTFLite` with `QuantizeMul` (float32
`sA·sB/sOut`, as TFLite), `QuantizeMean` and `QuantizeRequantize`; LOGISTIC
gets `QuantizeLogistic` (TFLite's float32 table); LEAKY_RELU gets
`qi8.RoundTFLite` with `QuantizeLeakyReLU`. These choices are fixed
by the LiteRT interpreter (qi8 `testdata/tflite_gen/OPS.md`). Weights alias the
model bytes (it may live in flash); bias aliases them when 4-byte aligned
(little-endian), else it is copied.

Operators: CONV_2D, DEPTHWISE_CONV_2D (multiplier 1, or one input channel
with any multiplier → Conv2D with transposed weights), FULLY_CONNECTED (a
Reshape follows when the declared output shape differs), AVERAGE_POOL_2D and
MAX_POOL_2D (no fused activation, same quantisation), RESHAPE (alias),
SOFTMAX, ADD (equal shapes), MUL (equal shapes, or the smaller operand per
channel or scalar), LOGISTIC (output `{1/256, −128}`), MEAN (axes H and W of
a rank-4 tensor, keep_dims either way), CONCATENATION (any non-batch axis,
no fused activation, int8 inputs quantised as the output — LiteRT rejects
anything else; uint8 inputs may differ, see below), PAD and PADV2 (constant int32/int64 paddings, no batch
padding; PAD fills with the output zero point, PADV2 with its constant,
which must share the output's quantisation), RESIZE_NEAREST_NEIGHBOR and
RESIZE_BILINEAR (rank 4, both flags), LEAKY_RELU, QUANTIZE (int8 → int8 → `Requantize`). A constant int8
operand of ADD, MUL or CONCATENATION becomes a `Const` layer (not for the
first operator). I/O cuts: a float or uint8 graph input whose QUANTIZE is
the first operator, and DEQUANTIZEs of graph outputs, are removed — the
graph takes the int8 input (`Layers[0].InQ`) and returns int8 outputs; the
caller converts. Requirements: int8 or uint8 activations with one scale/zero point,
symmetric int8 weights per channel or per tensor, int32 bias, batch 1, no
dilation, one input, one or more outputs (`Graph.Outputs` in model order;
nil when the only output is the last layer) with the last operator's among
them, one subgraph. Anything else returns an `*OpError{Index, Op, Reason}`
naming the operator (wraps `ErrUnsupported`).

uint8 models (TF's "quant" exports) load as int8: every uint8 value and zero
point minus 128 (exact). Per-tensor uint8 weights may be asymmetric: their
zero point minus 128 becomes `WZP` (the weights are converted into a copy);
multipliers are `float64(float32(sIn·sW))/float64(sOut)` as TFLite's uint8
path; a uint8 AVERAGE_POOL_2D sets `TiesUp`; CONV_2D/DEPTHWISE_CONV_2D round
as `RoundTFLite`, FULLY_CONNECTED as `RoundExact` (as for int8, both checked
against the interpreter). The caller feeds `x − 128` and reads `y + 128`.
uint8 is accepted on CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED,
AVERAGE_POOL_2D, MAX_POOL_2D, RESHAPE, SOFTMAX, CONCATENATION and LOGISTIC;
mixing int8 and uint8 in one operator, uint8 on any other operator,
per-channel uint8 weights and QUANTIZE to a non-int8 type are refused, the
reason naming the tensor. A uint8 CONCATENATION input quantised unlike the
output gets a `Rescale` map (`qi8.QuantizeConcatRescale`: TFLite's
`ConcatenationWithScaling`, `round(u·s + b)` as one fused multiply-add, as
LiteRT's arm64 build computes it).

`LoadDetection` is `Load` for SSD exports (TF Object Detection API) ending
in the custom `TFLite_Detection_PostProcess` operator, which must be last:
the graph stops before it, its box-encoding and class-score inputs become
`Graph.Outputs` (in that order), and its anchors (uint8 dequantised as
TFLite does, or float32) and FlexBuffers options come back as
`PostProcess`; `x/vision/detect/ssd` decodes the outputs with them. `Load`
keeps refusing every custom operator.

## Verification

Runner output is bit-identical to `RunLayer` layer by layer (sequential
`smallLayers`, MobileNetV1 at 96²/128² with and without FC, a residual
`resLayers` graph with Add from a layer and from the graph input, Reshape and
Softmax, in both rounding modes) and with a test backend fusing every
depthwise+pointwise pair. `checkPlan` verifies independently that no two
tensors with overlapping lifetimes share bytes and that Reshape aliases its
input; a planner that ignores lifetimes fails it. `AllocsPerRun == 0` for
`MobileNetV1`, `BindParams`, `SyntheticParams` and `Run`. `opsLayers`
covers every added kind (Pad, ResizeNearest, Logistic, Const, Mul per
channel/element-wise/scalar, Requantize, Concat on C and on W from one
producer twice, Mean, Reshape) in both rounding modes against `RunLayer`,
Const regions (`Off −2`) and zero allocations, plus 22 validation cases.
`tfl` tests build models in memory (`builder_test.go`) and check
conversion, aliasing, multipliers, the depthwise-multiplier transposition
(against a direct computation), Add/Softmax/Reshape, the zero multiplier,
every rejection path and truncated files; `testdata/opsnet.tflite`
(`gen_opsnet.py`: float I/O, three outputs, every added operator, constants)
equals the LiteRT interpreter on every operator output for 64 inputs
(`opsnet.json.gz`); `yoloLayers` checks LeakyReLU and ResizeBilinear against
`RunLayer` in both rounding modes. `testdata/u8net.tflite` (`gen_u8net.py`:
a uint8 model with asymmetric weights and every uint8 operator, tie-heavy
scales and inputs) equals the interpreter on every operator output for 64
inputs (`u8net.json.gz`) with `AllocsPerRun == 0`. `testdata/ssdtail.tflite`
(`gen_ssdtail.py`: an SSD export's tail — uint8 box and class heads,
CONCATENATIONs of differently quantised inputs, LOGISTIC, the custom
post-process with uint8 anchors) loads with `LoadDetection`; every layer
equals the interpreter and `ssd.Decoder` returns the custom operator's
detections (count, classes, scores; boxes to one float32 step) for 64 inputs
(`ssdtail.json.gz`).
