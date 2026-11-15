---
type: algorithm
status: tested
tags: [easyrobot, nn, layers, activation]
workstream: W9
module: "[[nn]]"
summary: "gradients match finite differences"
source: [x/math/nn/layers/activations.go]
tests: [x/math/nn/layers/activations_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → TestReLU, TestSigmoid, TestTanh, TestSoftmax (+_EdgeCases) pass; probe (2026-11-09) dInput vs central diff ≤2.2e-5"
---
# Activation layers: ReLU, Sigmoid, Tanh, Softmax

**What it does:** element-wise activations (and softmax along one axis) wrapped as `types.Layer`, with analytic backward.

**Where:** `x/math/nn/layers/activations.go`
- `ReLU` :17-103: backward uses `ReLUGrad` (:92)
- `Sigmoid` :106-197: backward uses `SigmoidGrad(output)` (:186)
- `Tanh` :200-291: backward uses `TanhGrad(output)` (:280)
- `Softmax` :294-391: `Softmax(dim)` (:349), backward uses `SoftmaxGrad(gradInput, gradOutput, dim)` (:380)

Dropout and BatchNorm2D live in the same file; see [[shape-utility-layers]] and [[batchnorm2d-layer]].

**Inputs → outputs:** any shape → same shape. FP32. The Softmax `dim` must lie in `[0, rank)`; this is checked in `Init` only (:315). Kernels in [[tensor]], down to fp32 primitives.

## Completeness review

- Complete: yes for these four.
- Missing for a modern decoder: **GELU, SiLU/Swish (needed for SwiGLU), LeakyReLU/ELU, LogSoftmax** layers. [INFERENCE] The tensor API may have some ops, but no Layer wrappers exist here.
- Tested:
  - `TestReLU`: exact forward and dX for three vectors.
  - `TestSigmoid`, `TestTanh`: output and derivative values, InDelta 1e-6.
  - `TestSoftmax`: only checks sum=1 and non-negativity; **backward only checks length** (activations_test.go:299-303).
  - `_EdgeCases`: nil receiver, Backward before Forward, empty grad.
  - All pass.
- Correct: yes. Probe `gradcheck` (central diff): Softmax dim1 [2,4] maxAbs 3.0e-6, Softmax dim0 [5] 1.5e-6, Sigmoid 2.0e-6, Tanh 2.2e-5, ReLU 4.9e-6.

## Defects

- `gradInput` is allocated lazily on the first Backward, because `Init` never preallocates `Base.Grad()` (:88-91, :182-185, :276-279, :376-379). Later calls reuse it via `StoreGrad`. There is no shape check before reuse [INFERENCE: a changed batch shape panics inside the kernel].
- Constructors take `name string` instead of the `Option` pattern used by Dense/Conv. `NewReLU("")` therefore sets an empty explicit name (:22-28, `WithName(name)` sets `nameSet`). Inconsistent API.
- `Sigmoid`/`Tanh` backward use the stored *output* reference, which is the layer's own output buffer, so they are safe. ReLU uses the stored *input*, which is the previous layer's output buffer. That is only valid if nothing mutates it between Forward and Backward. GPTEncoding does mutate its input (see [[gpt-positional-encoding-layer]]).

## TODOs

- [ ] Assert Softmax backward values (Jacobian-vector product) in TestSoftmax — activations_test.go:299-303 #p2
- [ ] Add SiLU/GELU layers (Qwen2 SwiGLU MLP needs SiLU) — activations.go #p2
- [ ] Preallocate gradInput in Init — activations.go:88-91,182-185,276-279,376-379 #p3

## Embedded suitability

Forward writes into the preallocated output. Backward allocates once. Pure FP32 loops in fp32 primitives (`exp` via math32 [INFERENCE]). No generics or reflection on the hot path beyond a `Data().(type)` switch per call in eager. No goroutines.

## Links

- Module: [[nn]] · kernels: [[tensor]], [[primitive-kernels]] · related: [[shape-utility-layers]], [[nn-loss-functions]]
