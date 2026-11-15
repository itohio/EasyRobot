---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, conv]
workstream: [W9]
module: "[[tensor]]"
summary: "forward correct; `Conv1DKernelGrad` wrong for Cout > 1 (`primitive/fp32/tensor.go:349-370`…"
source: [x/math/tensor/eager/conv.go, x/math/primitive/fp32/tensor.go]
tests: [x/math/tensor/eager/conv_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestConv1D (shape-only) pass; probe (2026-11-09): Conv1DKernelGrad maxAbsErr 4.45 vs reference (Cout=3,B=2)"
---
# Conv1D, Conv1DKernelGrad, Conv1DTransposed

**What it does:** 1D convolution over `[B,C,L]` or `[C,L]`. All three operations reuse the 2D kernels with width 1.

**Where:** `x/math/tensor/eager/conv.go`:
- `Conv1D` 347-470: calls `fp32.Conv2D` with `W=1`.
- `Conv1DKernelGrad` 581-681: calls `fp32.Conv1DKernelGrad`, which is defined at `x/math/primitive/fp32/tensor.go:322-371` and uses im2col + `Gemm_NN`.
- `Conv1DTransposed` 688-802: calls `fp32.Conv2DTransposed` with `W=1`.

**Inputs → outputs:**
- `Conv1D(dst, kernel[Cout,Cin,K], bias, stride, padding int)`: `[B,Cin,L]→[B,Cout,(L+2p−K)/s+1]`. A 2D input `[Cin,L]` gives a 2D output.
- `Conv1DKernelGrad(dst, dy, kernel, stride, padding)` returns a `[Cout,Cin,K]` gradient.
- `Conv1DTransposed(kernel, bias, stride, padding)` takes the forward-format kernel `[Cout_fwd, Cin_fwd, K]` and needs input channels = `kernel[0]`. Output is `[B, kernel[1], (L−1)s−2p+K]`. It has no dst parameter and always allocates.
- FP32 only.

## Completeness review

- Complete: yes; all three entry points exist. No dilation, groups, causal or asymmetric padding (needed for audio/TCN streaming).
- Tested: eager `TestConv1D` (conv_test.go:120-164) checks **shapes only**. No test for `Conv1DKernelGrad` or `Conv1DTransposed`. Pass (2026-11-09).
- Correct:
  - `Conv1D`: yes. The probe of `fp32.Conv2D` in the 1D configuration vs a naive reference gave maxAbsErr 4.8e-7.
  - `Conv1DKernelGrad`: **no**.
    - `outputGradFlat` is laid out `[B·Lout, Cout]` (tensor.go:349-359) but is passed to `Gemm_NN` as an `M=Cout × K=B·Lout` matrix with `ldA=im2colSize` (tensor.go:363-370). That is a transposed layout, so the result is wrong unless `Cout==1` or `B·Lout==1`. `Gemm_TN` with `ldA=Cout` is the fix.
    - Probe B=2, Cin=2, Cout=3, L=6, K=3: maxAbsErr 4.45. `fp32.Conv2DKernelGrad` on the same data: 0.
  - `Conv1DTransposed`: correct by derivation. The `[Cout_fwd,Cin_fwd,K]` buffer is already the `[inC,outC,K]` layout that `Conv2DTransposed` expects, so the doc's "internally transposes" means no data movement. The 2D kernel itself passed an adjoint probe ([[conv2d-transposed-kernel]]). There is no 1D-specific value check.

## Embedded suitability

`Conv1DKernelGrad` takes two buffers from `fp32.Pool` per call: `B·Lout·Cin·K` and `B·Lout·Cout` floats (tensor.go:340-343). On ESP32 that is a large transient buffer. The direct `Conv2DKernelGrad` with W=1 needs none and is correct. Otherwise no allocations beyond the output, and float32 only.

## TODOs

- [ ] Fix Conv1DKernelGrad GEMM layout (use Gemm_TN with ldA=Cout, or delegate to Conv2DKernelGrad with W=1) — x/math/primitive/fp32/tensor.go:349-370; probe maxAbsErr 4.45 #p1 #ER-020
- [ ] Add value tests for Conv1D, Conv1DKernelGrad (numerical) and Conv1DTransposed — x/math/tensor/eager/conv_test.go:120-164 #p2
- [ ] Give Conv1DTransposed a dst parameter like the other ops, and fix its doc comment about output channels/"transposes" — conv.go:683-688 #p3
- [ ] Add causal/dilated 1D conv for streaming audio models — conv.go #p2

## Links

- Module: [[tensor]]
- Related: [[conv2d-kernel]], [[conv2d-transposed-kernel]], [[im2col-col2im]], [[fp32-gemm]]
