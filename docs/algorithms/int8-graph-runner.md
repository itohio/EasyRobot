---
type: algorithm
status: tested
tags: [easyrobot, nn, int8, runtime, mobilenet, embedded]
workstream: W9
module: "[[nn]]"
summary: "Sequential int8 graph + allocation-free runner with pluggable backends (KPU/SIMD/generic segments) and a MobileNetV1-0.25 builder; builds for ESP32-S3 and K210."
source: [x/math/nn/qnet/layer.go, x/math/nn/qnet/graph.go, x/math/nn/qnet/runner.go, x/math/nn/qnet/mobilenet.go]
tests: [x/math/nn/qnet/qnet_test.go, x/math/nn/qnet/mobilenet_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/nn/qnet → ok (2026-11-09); runner output bit-identical to layer-by-layer qi8 at 96×96 and 128×128 (with/without FC-1000) and with a fused test backend; AllocsPerRun 0 for MobileNetV1, BindParams, SyntheticParams, Runner.Init, Run; tinygo 0.43 -print-allocs='qnet|qi8' empty for xiao-esp32s3 and UnitV (ER-NN-KERNELS N3)"
---
# Int8 graph runner (qnet) and MobileNetV1-0.25

**What it does:** runs a sequential int8 network on the [[int8-nn-kernels]] with no allocation, and splits it into segments that different backends execute: a K210 KPU backend can claim runs of conv/depthwise layers, a SIMD backend single layers, and the generic qi8 backend the rest.

**Where:** `x/math/nn/qnet` — `layer.go` (`Layer`, `Kind`, `Quant`, on-demand qi8 params, `RunLayer`), `graph.go` (`Graph.Validate/Check`, `Budget`, `ArenaSize`, `WeightBytes`, `MACs`, `ParamCounts`, `BindParams`), `runner.go` (`Backend`, `Generic`, `Runner.Init/NewRunner`, `Run`, `Plan`), `mobilenet.go` (`MobileNetV1`, `MobileNetV1Layers`, `SyntheticParams`); semantics in the package `SPEC.md`.

**Method:**
- A `Layer` is plain data (kind, NHWC in/out shapes, in/out quantisation, kernel/stride/padding, clamp range, weights/bias/requant slices) so a hardware backend can inspect and compile it.
- `Backend.Claim(layers)` returns how many consecutive layers it runs as one segment; `Run(layers, dst, src)` executes a segment. `Runner.Init` plans once (backends tried in order, at most 64 segments) and checks the arena.
- One arena used from both ends: segment k writes at the end (k even) or the start (k odd), so the arena needs the largest segment input + output; segment 0 reads the caller's input directly. A fused segment can need more than the per-layer maximum, so `Runner.ArenaSize()` reports the size for the chosen plan.
- `MobileNetV1(alpha, h, w, classes, layers)` fills a caller-provided `[]Layer`: conv 3×3/2 → 8 ch, 13 depthwise-separable blocks (pointwise 16, 32/2, 32, 64/2, 64, 128/2, 128×5, 256/2, 256), ReLU6 fused, global average pool, optional FC. Parameters are bound from caller-owned backing arrays (`BindParams`); `SyntheticParams` fills them deterministically with fan-in-scaled requantisation so activations stay spread.

**Budget (alpha 0.25, feature extractor):**

| | 96×96 | 128×128 |
|---|---|---|
| weights | 207 600 B (463 600 with FC-1000) | same |
| bias + requant | 32 832 B (44 832 with FC-1000) | same |
| MACs | 7.49 M (7.75 M with FC) | 13.31 M (13.57 M with FC) |
| arena (largest in + out) | 55 296 B (layer 2: 18 432 in + 36 864 out) | 98 304 B (layer 2: 32 768 + 65 536) |

At 224×224 with FC-1000 it counts 41.0 M MACs, matching the MobileNet paper.

## Completeness review

- Complete: no — no real-weight import (TFLite or other), no flash/const placement of parameters, no KPU or SIMD backend yet (interface ready), sequential graphs only.
- Tested: validation (18 error cases with exact error and layer index), parameter counts and binding, budget values, planning errors (unclaimed, over/negative claim, capacity, unbound params, input, not ready), arena exact-size check, a fusing test backend (plan string, output bit-identical to all-generic), runner bit-identical to layer-by-layer qi8 at 96 and 128 with/without FC, activation spread of synthetic parameters, allocations 0. TinyGo build of a 96×96 probe with static globals for both chips, `-print-allocs` clean.
- Host speed: about 5.7 ms/frame at 96×96 and 11 ms at 128×128 (feature extractor, Apple M5, heavily loaded machine — up to 3× variance).
- Correct: yes on these tests.

## TODOs

- [ ] Parameters in flash: emit const weight arrays (TinyGo places never-written globals in rodata) so ESP32-S3 SRAM holds only arena + input — x/math/nn/qnet #p2
- [ ] Import real int8 weights (TFLite flatbuffer or a simple export) once downloads are approved — x/math/nn/qnet #p2
- [ ] K210 KPU backend implementing `Backend` (k210 peer) — x/math/nn/qnet/runner.go #p2
- [ ] ESP32-S3 PIE backend (ER-NN-KERNELS N2) — x/math/nn/qnet #p2

## Links

- Module: [[nn]] · [[int8-nn-kernels]] · [[primitive-kernels]] · [[W9-go-nn-framework]]
