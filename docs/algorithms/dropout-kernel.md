---
type: algorithm
status: tested
tags: [easyrobot, nn, tensor, dropout]
workstream: [W9]
module: "[[tensor]]"
summary: "mask statistics verified by probe"
source: [x/math/tensor/eager/activations.go, x/math/tensor/types/dropout.go]
tests: [x/math/tensor/eager/activations_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → TestDropoutForward pass (values); probe (2026-11-09): mask p=0.3 → zero fraction 0.3008, mean 0.9989"
---
# Dropout (mask, forward, backward)

**What it does:** Inverted dropout. `DropoutMask` fills a tensor with 0 (with probability p) or `scale`. Forward and backward are element-wise multiplies by that mask.

**Where:**
- `x/math/tensor/eager/activations.go`: `DropoutForward(dst, mask)` 396-440, `DropoutMask(p, scale, rng)` 446-469, `DropoutBackward(dst, gradOutput, mask)` 476-506.
- Interface: `x/math/tensor/types/dropout.go`; `RNG` in `types/tensor.go:5-8`, which needs `Float64()` and `NormFloat64()`.

**Inputs → outputs:**
- Mask: the receiver's shape. The caller passes `scale` (normally `1/(1−p)`); it is not derived from p.
- `DropoutMask` **overwrites the receiver** and returns it (via `FillFunc`). The interface comment says "Returns a new tensor with the mask" (types/dropout.go:15). The probe confirmed the mask aliases the receiver.
- Forward with nil dst is in place.
- FP32 for forward/backward (ElemMul); `FillFunc` is generic over dtypes.

## Completeness review

- Complete: yes. No train/eval flag at this level; the caller skips dropout in eval.
- Tested: `TestDropoutForward` (activations_test.go:186-221) checks values for 2 masks. No test of the mask statistics or of `DropoutBackward`. Pass (2026-11-09).
- Correct: yes.
  - Forward by test.
  - Mask statistics by probe: p=0.3, scale 1/0.7, N=1e5 gives zero fraction 0.3008 and mean 0.9989.
  - Backward is `Clone(dy)·mask` (derivation).
  - Offsets are ignored, as in all eager ops.

## Embedded suitability

- Mask generation calls `rng.Float64()` per element through an interface, plus a float64 closure in `FillFunc`. Slow on MCU, but dropout is training-only.
- `DropoutBackward` with nil dst `Clone`s dy (one allocation).
- A float32 mask costs 4 B per element; a bitmask would be 32× smaller.

## TODOs

- [ ] Fix the interface doc (mask is written into the receiver, not a new tensor) or allocate a new tensor — x/math/tensor/types/dropout.go:14-16, eager/activations.go:446-469 #p3
- [ ] Add tests for DropoutMask statistics and DropoutBackward — x/math/tensor/eager/activations_test.go #p2

## Links

- Module: [[tensor]]
- Related: [[activation-kernels]], [[learn]], [[nn]]
