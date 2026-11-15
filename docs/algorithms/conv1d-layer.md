---
type: algorithm
status: partial
tags: [easyrobot, nn, layers, conv]
workstream: [W9, W2]
module: "[[nn]]"
summary: "kernel gradient wrong for >1 output channel (probe `micro`: got `[0 -0.5 8 11.5]` vs ref `[5 6.5 5…"
source: [x/math/nn/layers/conv1d.go, x/math/primitive/fp32/tensor.go]
tests: [x/math/nn/layers/conv1d_test.go, x/math/nn/layers/conv1d_weight_bias_test.go, x/math/learn/conv1d_layer_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → all 13 TestConv1D_* pass; probes (2026-11-09): kernel grad wrong whenever outChannels>1"
---
# Conv1D layer (NCL)

**What it does:** 1D cross-correlation with optional bias, plus backward for input, kernel and bias.

**Where:** `x/math/nn/layers/conv1d.go:24-87` (`NewConv1D`), `:90-125` (`Init`), `:128-163` (`Forward`), `:166-249` (`Backward`), `:252-272` (`OutputShape`). Kernel-gradient root cause: `x/math/primitive/fp32/tensor.go:322-373` (`Conv1DKernelGrad`).

**Inputs → outputs:** `[B,Cin,L]` FP32 → `[B,Cout,(L+2p-k)/s+1]`. Kernel `[Cout,Cin,k]` (`ParamKernels`), bias `[Cout]` only with `UseBias(true)` (:48-51). Kernels ([[tensor]]): `Conv1D` (:158), `Conv1DKernelGrad` (:219), `Conv2DTransposed` on a width-1 4D view for dX (:232-245), `Sum(axes 0,2)` for db (:203).

## Completeness review

- Complete: yes. Forward and backward exist for the stated scope (stride, symmetric pad; no dilation or groups).
- Tested: `TestConv1D_ComputeOutput` checks forward values. `TestConv1D_BackwardAccuracy` checks dK/db/dX values, but **only for Cin=Cout=1, stride 1** (conv1d_test.go cases `simple_1x1_kernel2`, `no_bias`). The remaining tests cover construction, shapes and flags. Run: all pass. learn `TestConv1D_Layer_Learning` and `TestDenseConv1D_CombinedLearning` pass (loss decreases only).
- Correct: **no**.
  - Probe `gradcheck`: dX and db match central difference (≤4.8e-6), but **dK maxAbs 2.36 (rel 15.6)** for 2→3 ch s2 p1 and 0.77 for 2→2 ch s1.
  - Probe `micro` (reference loop `dK[o,c,k]=Σ g[b,o,t]·x[b,c,ts+k-p]`): Cout=1 cases (C=1/2, B=1/2, pad 1) are exact. **Cout=2 fails**: C=1 L=4 K=2 got `[0 -0.5 8 11.5]`, reference `[5 6.5 5 6.5]`.
  - Cause: `fp32.Conv1DKernelGrad` lays out `outputGradFlat` as `[B·T, Cout]` (tensor.go:345-354) and then calls `Gemm_NN` treating it as `[Cout, B·T]` with ldA=B·T (:362-372). It needs a transposed-A GEMM (`Gemm_TN`) or a `[Cout, B·T]` layout. Correct only when Cout=1.

## Defects

- Kernel gradient is wrong for Cout>1 (primitive-kernels: fp32/tensor.go:345-372). Any multi-filter Conv1D trains on garbage kernel gradients. The learn tests still "converge" because Dense layers compensate.
- Backward allocates on every call: `Conv1DKernelGrad(nil, …)` (:219) creates a new tensor and then copies it (:221), and `tensor.New` for `gradInput4DTmp` (:240).
- dK/db overwrite instead of accumulate (:205, :221).
- dX via Conv2DTransposed has the same tail-size risk as [[conv2d-layer]] when `(L+2p-k)%s≠0` [INFERENCE: not hit in probe because the s2 case used L=7].

## TODOs

- [ ] Fix fp32.Conv1DKernelGrad GEMM operand layout (use Gemm_TN or transpose outputGradFlat to [Cout,B·T]) — x/math/primitive/fp32/tensor.go:345-372 #p1 #ER-029
- [ ] Add a multi-channel, stride-2, finite-difference gradient test for Conv1D — conv1d_test.go (BackwardAccuracy only covers 1→1) #p1 #ER-029
- [ ] Preallocate the kernel-grad and gradInput buffers in Init — conv1d.go:219,240 #p2
- [ ] Check dX when (L+2p-k)%s≠0 (same root cause as Conv2D) — conv1d.go:239-245 #p2

## Embedded suitability

Forward uses no extra allocation beyond `[]int` literals. Backward allocates twice per call (see above) and uses a `Pool` scratch for im2col. Reshape views are created on every call (:232, :235, :245). FP32 only. No goroutines.

## Links

- Module: [[nn]] · kernels: [[tensor]], [[primitive-kernels]] · related: [[conv2d-layer]], [[sequential-model-and-backprop]], [[learn]]
