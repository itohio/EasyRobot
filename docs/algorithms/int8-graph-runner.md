---
type: algorithm
status: tested
tags: [easyrobot, nn, int8, runtime, mobilenet, tflite, embedded]
workstream: W9
module: "[[nn]]"
summary: "Int8 graph (layers with producers and multiple outputs: residual Add, Mul, Concat, Const, Pad, resize, Mean, Logistic, Softmax) + greedy activation-memory planner + runner on qi8 prepared layers (PIE on the S3), TFLite int8 and uint8 importer (pure Go; BlazeFace and TF's uint8 MobileNetV1-0.25 quant models load bit-exact), MobileNetV1-0.25 builder; builds for ESP32-S3 and K210."
source: [x/math/nn/qnet/layer.go, x/math/nn/qnet/graph.go, x/math/nn/qnet/plan.go, x/math/nn/qnet/runner.go, x/math/nn/qnet/mobilenet.go, x/math/nn/qnet/tfl/flatbuf.go, x/math/nn/qnet/tfl/schema.go, x/math/nn/qnet/tfl/model.go, x/math/nn/qnet/tfl/load.go]
tests: [x/math/nn/qnet/qnet_test.go, x/math/nn/qnet/ops_test.go, x/math/nn/qnet/mobilenet_test.go, x/math/nn/qnet/tfl/load_test.go, x/math/nn/qnet/tfl/opsnet_test.go, x/math/nn/qnet/tfl/u8net_test.go, x/math/nn/qnet/tfl/builder_roundtrip_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/nn/qnet/... → ok (2026-11-12); runner bit-identical to layer-by-layer qi8 (sequential, residual, every added kind in both rounding modes, MobileNetV1, fused test backend); planner invariants checked independently; testdata/opsnet.tflite and testdata/u8net.tflite equal the LiteRT interpreter on every operator output (64 inputs each); BlazeFace, vww_96_int8 and TF's mobilenet_v1_0.25_{128,224}_quant equal the interpreter on every operator output (scratch checks, evidence ER-NN-OPS, ER-QU8)"
---
# Int8 graph runner (qnet), memory planner, TFLite import, MobileNetV1-0.25

**What it does:** runs int8 networks — sequential or residual — on the [[int8-nn-kernels]]: plans the activation memory once, prepares every layer as a qi8 prepared layer (so the ESP32-S3 build runs PIE kernels), and executes without allocating. Accelerator backends (K210 KPU) can claim chains of layers. Imported TFLite int8 models become graphs that reproduce TFLite's reference kernels.

**Where:** `x/math/nn/qnet` — `layer.go` (`Layer`, `Kind`, `Quant`, derived qi8 params, `RunLayer`), `graph.go` (`Check/Validate`, `Budget`, `ParamCounts`, `BindParams`), `plan.go` (`PlanMemory`, `MemPlan`, `ScratchSlice`), `runner.go` (`Backend`, `Runner`, `Run`, `RunSegment`), `mobilenet.go`; `x/math/nn/qnet/tfl` — `flatbuf.go` (bounds-checked FlatBuffers reader), `schema.go`, `model.go`, `load.go` (`Load`). Semantics in the package `SPEC.md`.

**Method:**
- A `Layer` is plain data: kind, producers (`From`; nil = previous layer, −1 = graph input), NHWC shapes, quantisation (second input `In2Q`), geometry, clamp, rounding mode, weights/bias/requant, Add multipliers, softmax/logistic scaling, `RQ`/`RQAlpha` (Mul/Mean/Requantize/LeakyReLU), concat axis, pads, resize flags. Kinds: conv, depthwise, FC, max/avg pool, ReLU, ReLU6, LeakyReLU, Add, Reshape (alias), Softmax, Logistic, Mul (per-channel/scalar broadcast), Mean, Requantize, Concat (N inputs), Pad, ResizeNearest, ResizeBilinear, Const (data in `Weights`). `Graph.Outputs` lists several graph outputs (nil = last layer); `Runner.Output(k)` returns them.
- `PlanMemory` (ER-NN-PIPELINE step 5, §D2): backends claim single-input chains; unclaimed layers run on the built-in CPU path. The persistent region holds each CPU layer's qi8 scratch (PIE-packed weights; empty on generic builds); `ScratchSlice` lets a caller place weights there so they are packed in place. Activations: each segment output lives from its segment to its last reader (graph outputs to the end); tensors are placed greedily (largest first, lowest 16-aligned offset free of tensors with overlapping lifetimes); Reshape aliases, Const outputs are their data. Sequential graphs reach the ping-pong bound (MobileNetV1-0.25: 55 296 B at 96²).
- `Runner` prepares qi8 layers (`Conv2DLayer` … `AddLayer`) over the arena once; `Run` / `RunSegment` allocate nothing.
- `tfl.Load` (§D1): CONV_2D, DEPTHWISE_CONV_2D (multiplier 1, or one input channel with any multiplier → Conv2D), FULLY_CONNECTED, AVERAGE/MAX_POOL_2D, RESHAPE, SOFTMAX, ADD, MUL, LOGISTIC, LEAKY_RELU, MEAN (H, W), CONCATENATION, PAD, PADV2, RESIZE_NEAREST_NEIGHBOR, RESIZE_BILINEAR, QUANTIZE (int8 → int8); `qi8.RoundTFLite` for conv/depthwise/Add/Mul/Mean/Requantize/LeakyReLU and `qi8.RoundExact` for FC (TFLite's FC rounds once, half away from zero); TFLite's float32-scale multiplier derivations and activation ranges; a multiplier below 2^-32 becomes qi8's zero multiplier (TFLite's encoding of dead channels — found in `vww_96_int8`); constant operands of ADD/MUL/CONCATENATION become Const layers; a float/uint8 input QUANTIZE and output DEQUANTIZEs are cut (the caller converts); several outputs map to `Graph.Outputs`; weights alias the model bytes. Anything else → `*OpError` naming the operator.
- `MobileNetV1(alpha, h, w, classes, layers)` and `SyntheticParams` as before (benchmarks, tests).

**Real models (host):** TFLM `person_detect.tflite` and MLPerf Tiny `vww_96_int8.tflite` (both MobileNetV1-0.25 96×96, 31 operators) load, plan to 55 296 B of activations, and **equal the LiteRT 2.3.0 interpreter (reference resolver) on every operator output** (FNV chained over all 31 op outputs, on TFLM's two test images per model) as well as an independent Python port with its own flatbuffer parser. On TFLM's test images person_detect classifies person 113 vs −113 and no-person 57 vs −57 (person / no-person scores), the outcome TFLM's own test asserts. Evidence: `~/src/docs/evidence/ER-NN-P4b.md`.

**Detection models (host):** ST's BlazeFace front 128 int8 (Apache-2.0; float I/O, four outputs; 71 layers incl. 11 PAD, 16 ADD) loads and **equals the LiteRT interpreter on every operator output** for 3 inputs (213 op outputs); it plans to 393 216 B of activations (more than the S3's internal heap, so it waits for PSRAM placement). ST's YOLO models are under SLA0044 and are used only as local load checks, never for reported results; a permissively licensed YOLO (YOLOX-nano, Apache-2.0) awaits the owner's download approval. Evidence: `~/src/docs/evidence/ER-NN-OPS.md`.

**uint8 "quant" models (ER-QU8):** TF's official `mobilenet_v1_0.25_128_quant` and `_224_quant` (uint8 per-tensor activations, asymmetric uint8 weights) load as int8 — every value and zero point minus 128, exact; the weights' zero point becomes `Layer.WZP`, the uint8 average pool's tie rule `Layer.TiesUp`, and multipliers take TFLite's uint8 form (`sIn·sW` in float32) — and **equal the LiteRT interpreter on every operator output** (31 per model, 7 inputs each: TFLM's two test images resized, a gradient, 4 random). Random one-operator uint8 models (1 650, incl. tie-stress cases) agree too; a uint8 FULLY_CONNECTED rounds once like the int8 one. uint8 is accepted on CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, AVERAGE/MAX_POOL_2D, RESHAPE and SOFTMAX; mixed int8/uint8 operators, other uint8 operators and per-channel uint8 weights are refused naming the tensor. Evidence: `~/src/docs/evidence/ER-QU8.md`.

## Completeness review

- Complete: no — one input; TFLite ops beyond the list above (e.g. SUB, HARD_SWISH, STRIDED_SLICE, TRANSPOSE) are rejected by name; no grouped depthwise (multiplier > 1 on multi-channel input), dilation, uint8 beyond the operators above, broadcasting other than per channel/scalar Mul, MEAN over other axes; no KPU backend yet; weights in PSRAM not yet planned (flash works by aliasing); BlazeFace's activations exceed the S3's internal RAM.
- Tested: validation of every rule incl. DAG cases (19 + 22 cases), parameters, budget, plan invariants (`checkPlan`: lifetimes vs offsets, Reshape alias, Const data regions, region bounds; catches a planner that ignores lifetimes), runner = reference for sequential, residual (both rounding modes), every added kind (`opsLayers`, `yoloLayers`, both modes), MobileNetV1 96/128 ± FC, fused test backend, Reshape of the input; claim errors (past end, over an Add, intermediate read outside); `AllocsPerRun` 0 for `Run`. `tfl`: in-memory model builder with a round-trip test, conversion of a full classifier chain, weight aliasing, multipliers, activation ranges, depthwise-multiplier transposition against a direct computation (a missing transpose is caught), residual Add from the graph input, zero multiplier, rejection paths, every truncation of a model; `opsnet.tflite` (every added operator, float I/O, three outputs) equals the interpreter on all operator outputs for 64 inputs — wrong rounding for Mul, Mean or Requantize, reversed concat order, a wrong pad value or a dropped resize flag each fail it.
- Correct: yes on these tests, including equality with the TFLite interpreter on the permissively licensed real models (person_detect, vww_96_int8, BlazeFace, TF's mobilenet_v1_0.25_128/224_quant).

## TODOs

- [x] Remaining §D1 operators (Logistic, Concatenation, ResizeNearest, Pad, Quantize/Dequantize, Mean, Mul) for YOLO and face detection — x/math/nn/qnet/tfl/load.go #p2
- [x] RESIZE_BILINEAR and LEAKY_RELU (YOLOX-nano, tiny YOLOv2) — x/math/nn/qnet/tfl/load.go #p2
- [ ] Place activations and weights in PSRAM on the S3 for models beyond SRAM (BlazeFace: 393 216 B of activations; MobileNetV2, YOLO) — x/math/nn/qnet/plan.go #p2
- [ ] K210 KPU backend implementing `Backend` (k210 peer) — x/math/nn/qnet/runner.go #p2

## Links

- Module: [[nn]] · [[int8-nn-kernels]] · [[primitive-kernels]] · [[W9-go-nn-framework]]
