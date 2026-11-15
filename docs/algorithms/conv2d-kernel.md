---
type: algorithm
status: tested
tags: [easyrobot, nn, tensor, conv]
workstream: [W9, W2]
module: "[[tensor]]"
summary: "NCHW, matches float64 naive (1.9e-6) and adjoint test; eager test checks shapes only"
source: [x/math/tensor/eager/conv.go, x/math/primitive/fp32/tensor.go]
tests: [x/math/primitive/fp32/tensor_test.go, x/math/tensor/eager/conv_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → fp32 TestConv2D (InDeltaSlice) pass; probe (2026-11-09): Conv2D vs naive ref 1.9e-6, KernelGrad vs ref 0; bench darwin/arm64 M5"
---
# Conv2D forward and Conv2DKernelGrad

**What it does:** NCHW 2D cross-correlation (deep-learning "convolution") with symmetric zero padding, stride, and an optional bias. `Conv2DKernelGrad` computes dL/dW.

**Where:**
- Eager `x/math/tensor/eager/conv.go`: `Conv2D` 19-146, `Conv2DKernelGrad` 475-576.
- Primitive `x/math/primitive/fp32/tensor.go`: `Conv2D` 187-250 (direct convolution), `Conv2DKernelGrad` 256-316 (direct).

**Inputs → outputs:**
- input `[B,Cin,H,W]`, kernel `[Cout,Cin,kH,kW]`, bias `[Cout]` or scalar or nil, `stride=[sH,sW]`, `padding=[pH,pW]`.
- Output `[B,Cout,OH,OW]` with `OH=(H+2pH−kH)/sH+1`.
- `Conv2DKernelGrad(dst, outputGrad[B,Cout,OH,OW], kernel, stride, padding)` returns a gradient shaped like the kernel.
- FP32 only (conv.go:144, 574).
- No dilation or groups here; see [[conv-variants-depthwise-group-dilated-3d]]. Input-gradient (dX) is done via `Conv2DTransposed` ([[conv2d-transposed-kernel]]).

## Algorithm

- Forward: for each output pixel, loop over (kh, kw) with the input-channel loop innermost. Index math is computed per element, with bounds checks for padding and no im2col.
  - The repo's `BENCHMARK_REPORT.md` declares that an earlier im2col+GEMM version was replaced by this one.
  - The `ic` loop is innermost, so the input stride is `H·W` floats, which is cache-hostile.
- KernelGrad: zeroes dW, then accumulates `in·dy` with the same loop nest.

## Completeness review

- Complete: yes, for its stated NCHW/symmetric-pad scope.
  - Nothing validates `stride>0`, `OH>0`, or that dst/input are contiguous.
  - There is no NHWC layout. TFLite/K230 models are NHWC; see [[tflite]].
- Tested: fp32 `TestConv2D` (tensor_test.go:249-346) checks output values with `InDeltaSlice 1e-5`. Eager `TestConv2D` (conv_test.go:12-92) checks **shapes only**. There is no test for `Conv2DKernelGrad`. Pass (2026-11-09).
- Correct: yes for contiguous offset-0 tensors.
  - Probe B=2, Cin=4, Cout=6, 6×5, k3, s2, p1 with bias vs an independent float64 naive loop: maxAbsErr 1.9e-6.
  - `Conv2DKernelGrad` vs an analytic reference: 0.
  - Adjoint with Conv2DTransposed: ⟨Cx,y⟩=−2.449117 vs ⟨x,Cᵀy⟩=−2.449115.
  - Like all eager kernels it ignores view offset and strides (passes `Data()`, conv.go:75-76, 556-557); see [[tensor-views-strides]].

## Performance (measured)

`BenchmarkTensorConv2D_Destination -benchtime 3x` on darwin/arm64 Apple M5, go1.27.2:

| Case | ns/op | Throughput |
|---|---|---|
| B32 64→128 64×64 k3 | 23,647,091,736 | ≈0.82 GFLOP/s |
| B16 32→64 128×128 k5 s2 | 4,223,706,069 | — |
| B8 16→32 256×256 k7 | 18,320,566,681 | — |
| B1 3→64 224×224 k3 | 299,066,778 | ≈0.58 GFLOP/s |

That is 3–6× below the same machine's `Gemm_NN` rate, so im2col+`Gemm_NT` would be faster here [INFERENCE; see [[im2col-col2im]]].

Declared, not reproduced: `x/math/primitive/fp32/BENCHMARK_REPORT.md` claims "Large Conv2D (8×64×32×32→8×128×32×32) 3.8 ms". That is ≈320 GFLOP/s, implausible for this scalar loop. `PERFORMANCE_REPORT.md` describes Conv2D as "naive Im2Col+GEMM", which contradicts the current code.

## Embedded suitability

- Kernel: zero allocation; float32 only; no generics or goroutines.
- Eager wrapper: allocates the output when dst is nil. With scalar bias it calls generic `ElemAddScalar` (closure per element).
- Integer index arithmetic per MAC (4 multiplies) dominates on Xtensa.
- No int8 conv for quantised TFLite models.

## TODOs

- [ ] Add value tests for eager Conv2D (stride/pad/bias/multi-channel) and Conv2DKernelGrad (numerical check) — x/math/tensor/eager/conv_test.go:12-92 #p2
- [ ] Validate stride>0 and OH/OW>0 and contiguous inputs; honour view offset — conv.go:63-66,75-76 #p2
- [ ] Reorder loops (ic outer, kh/kw inner, hoist row pointers) or switch to im2col+Gemm_NT for large Cin — x/math/primitive/fp32/tensor.go:205-249; bench 0.6–0.8 GFLOP/s #p2
- [ ] Add NHWC and int8 conv paths for TFLite/K230 models — fp32/tensor.go #p2
- [ ] Correct stale BENCHMARK_REPORT/PERFORMANCE_REPORT Conv2D claims — x/math/primitive/fp32/BENCHMARK_REPORT.md:49-50, PERFORMANCE_REPORT.md:23 #p3

## Links

- Module: [[tensor]]
- Related: [[conv2d-transposed-kernel]], [[im2col-col2im]], [[conv1d-kernel]], [[conv-variants-depthwise-group-dilated-3d]], [[fp32-gemm]], [[nn]]
