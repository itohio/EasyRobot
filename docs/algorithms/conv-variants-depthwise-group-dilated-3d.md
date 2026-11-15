---
type: algorithm
status: implemented
tags: [easyrobot, nn, tensor, conv]
workstream: [W9]
module: "[[tensor]]"
summary: "forward correct by equivalence probes; no gradients, no `dst`"
source: [x/math/tensor/eager/conv.go, x/math/primitive/fp32/tensor.go]
tests: [x/math/tensor/eager/conv_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → TestDepthwiseConv2D/TestGroupConv2D/TestDilatedConv2D/TestConv3D pass (shape-only); probe (2026-11-09): all equivalences ≤1.9e-6"
---
# DepthwiseConv2D, GroupConv2D, DilatedConv2D, Conv3D

**What it does:** Forward-only direct convolutions (NCHW / NCDHW) for depthwise (multiplier 1), grouped, dilated (atrous) and 3D cases.

**Where:**
- Eager `x/math/tensor/eager/conv.go`: `DepthwiseConv2D` 1247-1340, `GroupConv2D` 1348-1434, `DilatedConv2D` 1442-1527, `Conv3D` 1536-1617.
- Primitives `x/math/primitive/fp32/tensor.go`: `DepthwiseConv2D` 1429-1481, `GroupConv2D` 1491-1560, `DilatedConv2D` 1570-1626, `Conv3D` 1707-1768. `SeparableConv2D` (1877-1936) has no eager wrapper.

**Inputs → outputs:**

| Op | Kernel | Bias | Output |
|---|---|---|---|
| Depthwise | `[C,1,kH,kW]` or `[C,kH,kW]` | `[C]` | `[B,C,OH,OW]` |
| Group | `[Cout,Cin/g,kH,kW]` (Cin%g and Cout%g must be 0) | `[Cout]` | `[B,Cout,OH,OW]` |
| Dilated | `[Cout,Cin,kH,kW]` | `[Cout]` | uses effective kernel `(k−1)d+1` |
| Conv3D | `[Cout,Cin,kD,kH,kW]` on `[B,Cin,D,H,W]` | `[Cout]` | stride/pad of length 3 |

- None of these take a dst; all always allocate the result and return `*Tensor`.
- No backward/gradients.
- FP32 only.

## Completeness review

- Complete: yes, for forward inference.
  - No depth multiplier >1. No gradients, so these layers cannot be trained.
  - No dst parameter, unlike `Conv2D`.
- Tested: eager `TestDepthwiseConv2D`, `TestGroupConv2D`, `TestDilatedConv2D` and `TestConv3D` (conv_test.go:323-475) check **shape and length only**. Pass (2026-11-09).
- Correct: yes for contiguous inputs, by probe (p2) against the [[conv2d-kernel]] primitive, which itself matches a float64 naive reference with error 1.9e-6:

  | Equivalence checked | maxAbsErr |
  |---|---|
  | GroupConv2D(groups=1) vs Conv2D | 9.5e-7 |
  | GroupConv2D(groups=2) vs block-diagonal Conv2D | 1.9e-6 |
  | DepthwiseConv2D vs GroupConv2D(groups=C) | 0 |
  | DilatedConv2D(d=1) vs Conv2D | 9.5e-7 |
  | DilatedConv2D(d=2) vs Conv2D with zero-inserted 5×5 kernel | 9.5e-7 |
  | Conv3D(D=1, kD=1) vs Conv2D | 9.5e-7 |

  View offsets are ignored, as in all eager ops.

## Embedded suitability

- `DepthwiseConv2D` with a 4D kernel copies it into a fresh `make([]float32, C·kH·kW)` on every call (conv.go:1311-1318). The layouts are identical, so the copy is pointless.
- Output always allocated. `GroupConv2D` recomputes `srcBatchOffset` in the innermost channel loop.
- float32 only, no generics or goroutines. Depthwise is the MobileNet workhorse on ESP32/K230 and has no int8 path.

## TODOs

- [ ] Add value tests (the probe equivalences above make good test oracles) — x/math/tensor/eager/conv_test.go:323-475 #p2
- [ ] Add dst parameters and stop the per-call kernel copy in DepthwiseConv2D — conv.go:1247, 1311-1318 #p3
- [ ] Add backward (input/kernel grads) for depthwise/group/dilated if training is in scope — conv.go #p2
- [ ] Add depth multiplier >1 and an NHWC/int8 depthwise path for TFLite MobileNets — fp32/tensor.go:1429 #p2

## Links

- Module: [[tensor]]
- Related: [[conv2d-kernel]], [[conv2d-transposed-kernel]], [[tflite]]
