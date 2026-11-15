---
type: algorithm
status: tested
tags: [easyrobot, nn, tensor, conv]
workstream: [W9, W2]
module: "[[tensor]]"
summary: "—"
source: [x/math/tensor/eager/conv.go, x/math/primitive/fp32/tensor.go]
tests: [x/math/primitive/fp32/tensor_test.go, x/math/tensor/eager/conv_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → fp32 TestIm2Col/TestCol2Im/TestCol2ImClearsDestination (InDeltaSlice) pass; probe: im2col+Gemm_NT vs Conv2D 9.5e-7, col2im adjoint exact"
---
# Im2Col / Col2Im

**What it does:** `Im2Col` unrolls NCHW image patches into a row matrix so that convolution becomes a GEMM. `Col2Im` is its adjoint: it scatter-adds columns back into an image.

**Where:**
- Eager `x/math/tensor/eager/conv.go`: `Im2Col` 1687-1766, `Col2Im` 1774-1859.
- Primitives `x/math/primitive/fp32/tensor.go`: `Im2Col` 8-89, `Col2Im` 96-179.

**Inputs → outputs:**
- Eager `Im2Col(dst, kernelSize, stride, padding)`: `[B,C,H,W]` → `[B·OH·OW, C·kH·kW]`. Rows are output positions; columns are ordered (c, kh, kw).
- Eager `Col2Im(dst, outputShape[4], kernelSize, stride, padding)` does the reverse. It **zeroes dst first**, then accumulates.
- The primitive argument order is `(…, kernelH, kernelW, padH, padW, strideH, strideW)`, with **pad before stride**. Every other fp32 conv/pool kernel takes stride then pad, so this is an easy argument swap to make.
- FP32 only.

## Completeness review

- Complete: yes. A fast path copies whole kernel rows when the window is fully inside the image (tensor.go:42-55). Dilation is not supported.
- Tested: fp32 `TestIm2Col` (tensor_test.go:10-135, `InDeltaSlice`), `TestCol2Im`, `TestCol2ImClearsDestination`; eager `TestIm2Col` and `TestCol2Im` (conv_test.go:267-309). Pass (2026-11-09).
- Correct: yes, by probe (p2):
  - Im2Col followed by `Gemm_NT(W[Cout,CkK], col)` plus bias reproduces `fp32.Conv2D` with error 9.5e-7.
  - ⟨im2col(x), c⟩ = ⟨x, col2im(c)⟩ = −16.77843.
  - View offsets are ignored, as in all eager ops.
- No conv path in the repo uses im2col anymore: `Conv2D` is direct. The only remaining user is the broken `Conv1DKernelGrad` ([[conv1d-kernel]]).

## Embedded suitability

The eager op allocates a `B·OH·OW·C·kH·kW` matrix when dst is nil, which is k²× the input size. That is prohibitive on ESP32 for large layers, but it is the standard route to a fast GEMM on K230/desktop. The kernel itself does not allocate; it is float32 only, uses `copy()` in the fast path, and has no generics.

## TODOs

- [ ] Use the documented/consistent argument order (stride before pad) or a params struct — x/math/primitive/fp32/tensor.go:8-15,96-103 #p3
- [ ] Optionally route large-Cin Conv2D through Im2Col+Gemm_NT (≈3–6× faster on M5 per [[conv2d-kernel]] bench) with a tiled column buffer — fp32/tensor.go:187 #p2

## Links

- Module: [[tensor]]
- Related: [[conv2d-kernel]], [[fp32-gemm]], [[conv1d-kernel]]
