# qnet — sequential int8 network runner on qi8

Tiny graph + runner for int8 CNNs on microcontrollers (ESP32-S3, K210) and
the host. Imports only the standard library and `x/math/primitive/qi8` (never
`x/math/nn` or `x/math/tensor`). Nothing allocates except `NewRunner` (one
`Runner`); errors are sentinels returned unwrapped. Numerics are exactly qi8's
(see `qi8/SPEC.md`).

## Data

| Type | Meaning |
|---|---|
| `Quant{Scale float32; ZP int8}` | activation quantisation, `real = Scale·(q − ZP)` |
| `Kind` | `Conv2D`, `DepthwiseConv2D`, `FullyConnected`, `MaxPool2D`, `AvgPool2D`, `ReLU`, `ReLU6` |
| `Layer` | plain data: `Kind`, `In/Out qi8.Shape`, `InQ/OutQ Quant`, geometry `KH, KW, StrideH, StrideW, Padding`, fused clamp `ActMin/ActMax`, `Weights []int8`, `Bias []int32` (nil = 0), `Requant []qi8.Requant` |
| `Graph{Layers []Layer}` | sequential: `Layers[i].Out` feeds `Layers[i+1].In` |

The qi8 parameter structs are derived, never stored: `Conv2DParams()`,
`DepthwiseParams()`, `FCParams()` (`In = In.Len()`, `Out = Out.C`),
`PoolParams()`. `Out.C` is the channel count of Conv2D/FC; everything else in
`Out` is implied (`OutShape()`). Global average pooling is `AvgPool2D` with
`KH×KW = In.H×In.W`, VALID. Standalone `ReLU6` uses
`qi8.Quantize6(InQ.Scale, InQ.ZP)`. Weight layouts are qi8's (OHWI,
`[KH][KW][C]`, `[Out][In]`).

## Validation (`Graph.Check() (layer int, err error)`, `Validate()`)

Per layer: known kind (`ErrKind`); `Out == OutShape()` and valid geometry
(`ErrShape`); positive finite scales (`ErrQuant`); `In`/`InQ` equal the
previous `Out`/`OutQ` (`ErrShape`/`ErrQuant`); pools and ReLUs keep
quantisation (`ErrQuant`); `ActMin ≤ ActMax` (`ErrActivation`);
`len(Weights)`, `len(Requant)` equal `ParamCounts`, `Bias` nil or equal, all
requants `Valid`, no params on parameter-free kinds (`ErrParams`). Empty graph:
`ErrEmpty`.

## Parameters

`Layer.ParamCounts()` / `ParamCounts(g)` give weight, bias and requant entry
counts (Conv2D `OutC·KH·KW·InC, OutC, OutC`; depthwise `KH·KW·C, C, C`; FC
`Out·In, Out, Out`). `BindParams(g, w, b, rq)` slices three backing arrays of
exactly those lengths into the layers in order (3-index slices, no copy);
`b` may be nil.

## Budget

`Graph.Budget(i) LayerBudget{In, Out, Weights, Params, MACs}`: activation
bytes, weight bytes, bias (4 B) + requant (8 B) bytes, nominal MACs including
padded taps (= weights × output pixels; pools and ReLUs 0). Totals:
`WeightBytes`, `ParamBytes`, `MACs`, `ArenaSize`.

## Backends and planning

```go
type Backend interface {
    Name() string
    Claim(layers []Layer) int               // layers run from layers[0] as one segment; 0 = none
    Run(layers []Layer, dst, src []int8) error // NHWC in (first In) → out (last Out); no allocation
}
```

`Generic` claims any known kind, one layer per segment, and runs it with
`RunLayer` (qi8). `Runner.Init(g, arena, backends...)` validates `g`, then
greedily plans: at each layer the backends are asked in order, the first
non-zero claim becomes the next segment (`ErrClaim` if it runs past the end
or is negative, `ErrUnclaimed` if nobody claims, `ErrCapacity` beyond
`MaxSegments` = 64). The plan lives in a fixed array; `Plan()` exposes it.

## Arena

Ping-pong in one buffer: segment k writes its output at the arena end (k
even) or start (k odd); segment 0 reads the caller's input directly, later
segments read the previous output. Requirement = max over segments of
`first.In + last.Out` bytes (segment 0: `last.Out`); `ErrArena` if smaller.
`Graph.ArenaSize()` is that value for the all-Generic plan;
`Runner.ArenaSize()` for the actual plan (fused segments can need more, since
their intermediates live in the backend). Backends keep their own scratch.

`Runner.Run(input)`: `ErrNotReady` before a successful `Init`, `ErrInput`
on a length mismatch; output aliases the arena until the next `Run`.

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
FC-1000 (the paper's 41 M); arena 55 296 B at 96², 98 304 B at 128².

## Verification

Runner output is bit-identical to calling the qi8 kernels layer by layer
(96² and 128², with and without FC), also with a test backend fusing every
depthwise+pointwise pair. `AllocsPerRun == 0` for `MobileNetV1`,
`BindParams`, `SyntheticParams`, `Runner.Init`, `Runner.Run`. TinyGo
`-print-allocs='qnet|qi8'` is empty for `xiao-esp32s3` and the UnitV target
(probe with statically sized globals; not part of the repo).
