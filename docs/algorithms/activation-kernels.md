---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, activations]
workstream: [W9]
module: "[[tensor]]"
summary: "Softplus(100)=+Inf; no grads for ReLU6/Leaky/ELU/Softplus/Swish/GELU; GELU tanh-approx only; flat…"
source: [x/math/tensor/eager/activations.go, x/math/primitive/fp32/activations.go]
tests: [x/math/tensor/eager/activations_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestReLU/TestSigmoid/TestTanh pass (values); probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): Softplus(100)=+Inf; GELU/Swish/ELU/ReLU6/LeakyReLU values OK"
---
# Activation kernels and gradients

**What it does:** Element-wise activations on eager tensors. ReLU, Sigmoid and Tanh have fp32 kernels and gradients; ReLU6, LeakyReLU, ELU, Softplus, Swish/SiLU and GELU (tanh approximation) are composed from other tensor ops and have no gradient.

**Where:**
- Eager `x/math/tensor/eager/activations.go`: `ReLU` 13-43, `Sigmoid` 47-77, `Tanh` 81-111, `ReLUGrad` 115-162, `SigmoidGrad` 166-213, `TanhGrad` 217-264, `ReLU6` 510-547, `LeakyReLU` 551-583, `ELU` 587-625, `Softplus` 629-660, `Swish` 664-695, `GELU` 700-750.
- fp32 `x/math/primitive/fp32/activations.go`: `ReLU` 13-25, `ReLUGrad` 29-41, `Sigmoid` 45-60, `SigmoidGrad` 64-72, `Tanh` 76-85, `TanhGrad` 89-97, and the `*GradStride` variants 280-426.

**Inputs → outputs:**
- Same shape in and out.
- `ReLU`/`Sigmoid`/`Tanh` with nil dst work **in place on the receiver** and return it. The doc comment says "creates a new tensor".
- Composite ops with nil dst `Clone()` first.
- The `*Grad` ops take the forward input (ReLU) or output (Sigmoid/Tanh) as the receiver.
- FP32 only.

## Completeness review

- Complete: no.
  - No gradients for ReLU6, LeakyReLU, ELU, Softplus, Swish or GELU.
  - No exact-erf GELU; the tanh approximation is used (activations.go:728-746).
  - No HardSwish/HardSigmoid (MobileNetV3), no `Clip`.
  - SiLU exists only under the name `Swish`; Qwen's MLP uses SiLU.
- Tested: `TestReLU`, `TestSigmoid`, `TestTanh` (activations_test.go:12-115) check values with InDelta. Pass (2026-11-09). There are no tests for the grads or for any composite activation.
- Correct: **no** (Softplus overflow); otherwise yes by probe.

  | Probe input | Result | Reference |
  |---|---|---|
  | GELU [-1, .5, 3] | [-0.15881 0.34571 2.99636] | erf-exact [-0.15866 0.34573 2.99595]; tanh-approx error ≤4e-4 |
  | Swish [-2, 2] | matches x·σ(x) | — |
  | ELU [-1, 2] | [-0.63212 2] | correct |
  | ReLU6 [-1, 3, 9] | [0 3 6] | correct |
  | LeakyReLU(0.1) [-2, 2] | [-0.2 2] | correct |
  | **Softplus(100)** | **+Inf** | 100; naive `log(1+exp(x))`, and `ElemExp` returns +Inf for x>88 (activations.go:653-656, tensor_elementwise.go:171-181) |

  - `ReLU`/`Sigmoid`/`Tanh` loop over `Size()` flat elements of `Data()`, so they ignore strides and offset (activations.go:22-39). Probe on a Slice view gave [1 2 3 4]; the correct result is [5 6 7 8].
  - The grads have a stride path, but it ignores offset.

## Embedded suitability

- fp32 `Sigmoid` and `Tanh` call float64 `math.Exp` and `math.Tanh` per element (activations.go:57, 83), which is soft-float double on ESP32. `math32` is used elsewhere.
- The composite activations allocate 2–5 full-size temporaries per call:
  - GELU: Clone ×2 plus Pow/Add/Tanh passes (activations.go:728-746).
  - ReLU6: Clone + Fill + Less + Where (536-539).
  - LeakyReLU and ELU: 3 clones each.
- Each composite makes 5–8 memory passes through generic strided iterators. That is poor for MCU RAM and speed; fused fp32 kernels are needed.

## TODOs

- [ ] Softplus: use the stable form max(x,0)+log1p(exp(-|x|)) — x/math/tensor/eager/activations.go:653-656; probe Softplus(100)=+Inf #p1 #ER-020
- [ ] Add fused fp32 kernels (ReLU6, LeakyReLU, ELU, SiLU, GELU tanh/erf, HardSwish) with zero temporaries — activations.go:510-750 #p2
- [ ] Add gradients for ReLU6/LeakyReLU/ELU/Softplus/Swish/GELU — activations.go #p2
- [ ] Replace float64 math.Exp/math.Tanh with math32 in Sigmoid/Tanh/Softmax — x/math/primitive/fp32/activations.go:57,83,121,157,196 #p2
- [ ] ReLU/Sigmoid/Tanh: honour strides and offset (use Elem* strided kernels); fix the "creates a new tensor" doc — activations.go:13-111 #p1 #ER-020
- [ ] Add value tests for composite activations and all *Grad ops (numerical check) — x/math/tensor/eager/activations_test.go #p2

## Links

- Module: [[tensor]]
- Related: [[softmax-kernel]], [[tensor-elementwise-broadcast-reduce]], [[nn]]
