---
type: algorithm
status: partial
tags: [easyrobot, nn, layers, conv]
workstream: [W9, W2]
module: "[[nn]]"
summary: "input gradient wrong when `(H+2p-k) % s ≠ 0` (maxAbs 1.52); tests only 1→1 channel, stride 1"
source: [x/math/nn/layers/conv2d.go]
tests: [x/math/nn/layers/conv2d_test.go, x/math/nn/layers/conv2d_weight_bias_test.go, x/math/nn/layers/pooling_grad_test.go, x/math/learn/conv2d_layer_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → all 13 TestConv2D_* pass; probe tmp/easyrobot/tensor/layernotes/gradcheck (2026-11-09): dInput wrong (maxAbs 1.52) when (H+2p-k)%s≠0"
---
# Conv2D layer (NCHW)

**What it does:** 2D cross-correlation with optional bias, plus backward for input, kernel and bias.

**Where:** `x/math/nn/layers/conv2d.go:29-101` (`NewConv2D`), `:104-152` (`Init`), `:155-190` (`Forward`), `:195-294` (`Backward`), `:297-319` (`OutputShape`), `:322-400` (accessors).

**Inputs → outputs:** input `[B,Cin,H,W]` FP32 → `[B,Cout,(H+2p-kh)/sh+1,(W+2p-kw)/sw+1]`. Kernel `[Cout,Cin,kh,kw]` (`ParamKernels`), bias `[Cout]` (`ParamBiases`), created only with `UseBias(true)` (:59-62). Symmetric padding per axis; no dilation and no groups. Kernels ([[tensor]]): `Conv2D` (:185), `Conv2DTransposed` for dX (:238), `Conv2DKernelGrad` for dK (:287), `Sum(axes 0,2,3)` for db (:271).

## Completeness review

- Complete: no. Forward and backward exist. dX is only right when the transposed conv restores the input size exactly.
- Tested: `TestConv2D_ComputeOutput` checks output values with InDelta 1e-5. `TestConv2D_BackwardAccuracy` checks dW/db/dX values, but only for 1→1 channel, 2×2 all-ones kernel, stride 1 and pad 0 (conv2d_test.go:650-771). An all-ones kernel cannot detect a missing kernel flip, and there is no multi-channel or stride case. `TestMaxPoolLayerBackwardMatchesExpected` uses a 1×1 kernel. Run: all pass. learn `TestConv2D_Layer_Learning` and `TestDenseConv2D_CombinedLearning` pass (`go test -count=1 -short ./x/math/learn/`, 2026-11-09). These only check that loss decreases.
- Correct: no. Probe `gradcheck` (central difference, eps 1e-2, loss=Σout·R):
  - 2→3 ch, 3×3, s2 p1, input 5×5: dX maxAbs 1.2e-5, dK 8.2e-6, db 1.7e-6 ✓
  - 2→3 ch, 3×2, s1, pad(0,1), B=2: dX maxAbs 1.7e-5, dK 2.1e-5 ✓
  - 1→1, 3×3, s2 p1, **input 6×6**: dK ✓, **dX maxAbs 1.52 (rel 15.5) ✗**. With (6+2-3)%2=1, `Conv2DTransposed` produces 5×5 data, which is then reinterpreted into the 6×6 buffer.

## Defects

- dX is wrong whenever `(H+2p-kh) % sh ≠ 0` or `(W+2p-kw) % sw ≠ 0`. The transposed conv drops the tail rows/cols. `Backward` has two identical branches "if sizes don't match, reshape and copy what we can" (conv2d.go:247-257), so nothing pads or crops. Typical trigger: stride-2 downsampling on even inputs (MNIST 28×28 with k3 s2 p1).
- TODO comment "Replace manual gradient computation" (:193) is stale: the code already uses tensor ops.
- dK/db overwrite instead of accumulate (:271-273, :287). Same semantics as [[dense-layer]].
- Bias default differs from Dense: Conv defaults to no bias (:59), Dense always has one.
- `inputGradTmpTensor` is reallocated whenever the shape changes (:235-237). Fixed batch, same as Dense.

## TODOs

- [ ] Fix dX for non-divisible stride: compute transposed conv with output_padding = (H+2p-kh)%sh (or col2im into an exact-size buffer); add multi-channel, stride-2, even-size, asymmetric-kernel gradient tests — conv2d.go:238-257 #p1 #ER-029
- [ ] Replace the BackwardAccuracy cases with a finite-difference checker (random asymmetric kernels, Cin/Cout>1, stride 2, padding) — conv2d_test.go:629-843 #p2
- [ ] Remove the stale TODO and the duplicated if/else — conv2d.go:193,247-257 #p3
- [ ] Add dilation/groups (depthwise) if needed for K230/ESP32 mobile nets — conv2d.go:13-27 #p2

## Embedded suitability

Forward writes into the preallocated output. Backward reuses `inputGradTmpTensor` and `Base.Grad()`. The kernels use im2col/GEMM in [[primitive-kernels]], with a `Pool` scratch buffer. Slices are passed as `[]int{stride…}` literals, which costs a small heap allocation per call. FP32 only. No goroutines in the layer.

## Links

- Module: [[nn]] · kernels: [[tensor]], [[primitive-kernels]] · related: [[conv1d-layer]], [[pooling-layers]], [[batchnorm2d-layer]], [[sequential-model-and-backprop]]
