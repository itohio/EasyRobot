---
type: algorithm
status: tested
tags: [easyrobot, nn, tensor, conv]
workstream: [W9]
module: "[[tensor]]"
summary: "adjoint of Conv2D by probe"
source: [x/math/tensor/eager/conv.go, x/math/primitive/fp32/tensor.go]
tests: [x/math/primitive/fp32/tensor_test.go, x/math/tensor/eager/conv_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → fp32 TestConv2DTransposed (InDeltaSlice) pass; probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): adjoint <Cx,y>=-2.449117 vs <x,Cᵀy>=-2.449115"
---
# Conv2DTransposed (deconvolution)

**What it does:** Transposed 2D convolution (the gradient of Conv2D with respect to its input), with an optional output padding.

**Where:**
- Eager `x/math/tensor/eager/conv.go:154-337` (`Conv2DTransposed`).
- Primitives `x/math/primitive/fp32/tensor.go`: `Conv2DTransposed` 378-467 (scatter form) and `Conv2DTransposedWithOutputPadding` 1777-1865.
- `Conv3DTransposed` (tensor.go:1944-2038) and `SeparableConv2D` (1877-1936) also exist as primitives with no eager wrapper.

**Inputs → outputs:**
- input `[B,Cin,H,W]`, kernel `[Cin,Cout,kH,kW]` (PyTorch ConvTranspose layout), bias `[Cout]` or scalar.
- Output `[B,Cout,(H−1)sH−2pH+kH (+outPad), …]`.
- Output padding is inferred from the dst shape: `0 ≤ outPad < stride`, else panic (conv.go:213-236).
- FP32 only.

## Algorithm

Zero the output, then scatter: for each (b, ic, oc, ih, iw, kh, kw) do `out[ih·s−p+kh, iw·s−p+kw] += in·w`, with bounds checks. Bias is added in a separate pass. No dilation or groups.

## Completeness review

- Complete: yes, for stride/pad/outPad/bias.
- Tested:
  - fp32 `TestConv2DTransposed` (tensor_test.go:347-418): output values, `InDeltaSlice 1e-5`.
  - Eager `TestConv2DTransposed` (conv_test.go:94-118): shape only.
  - The output-padding path has no test.
  - Pass (2026-11-09).
- Correct: yes for contiguous inputs. In the probe (B1, Conv: Cin 2→Cout 3, 5×5, k3, s2, p1), the transposed op with the same weight buffer is the exact adjoint of `fp32.Conv2D`: −2.449117 vs −2.449115. View offsets are ignored, as in all eager ops.

## Embedded suitability

The kernel does not allocate; the eager op allocates the output when dst is nil. float32 only. The scatter writes are cache-unfriendly; the loop order is ic, oc, ih, iw, kh, kw (tensor.go:398-461).

## TODOs

- [ ] Add a test for the output-padding path and an eager value test — x/math/tensor/eager/conv_test.go:94-118; fp32/tensor.go:1777 #p2
- [ ] Expose Conv3DTransposed/SeparableConv2D in eager or delete them (no callers in eager) — x/math/primitive/fp32/tensor.go:1877-2038 #p3

## Links

- Module: [[tensor]]
- Related: [[conv2d-kernel]], [[conv1d-kernel]], [[im2col-col2im]]
