---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, normalization]
workstream: [W9]
module: "[[tensor]]"
summary: "per-element not per-channel stats; no running stats/inference mode; dx wrong for γ ≠ 1; nil-dst…"
source: [x/math/tensor/eager/normalizations.go, x/math/primitive/fp32/normalization.go]
tests: [x/math/tensor/eager/normalizations_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestBatchNormForward pass (no value checks); probe (2026-11-09): BatchNormGrad dx wrong for gamma≠1 (maxAbsErr 5.6); per-feature not per-channel stats on NCHW; batch=1 → all zeros"
---
# Batch normalization (forward + grad)

**What it does:** Computes `y = (x − μ)/√(σ² + ε)·γ + β`, where μ and σ² are taken over the **batch axis only**, separately for every non-batch element.

**Where:**
- Eager `x/math/tensor/eager/normalizations.go`: `BatchNormForward` 16-52, `BatchNormGrad` 63-129.
- fp32 `x/math/primitive/fp32/normalization.go`: `BatchNormForward` 12-75, `BatchNormGrad` 378-464.

**Inputs → outputs:**
- x `[B, …]`; γ and β have shape `[prod(shape[1:])]`, i.e. per feature element, not per channel. γ and β may be nil.
- ε is added inside the square root. The variance is biased (/B).
- Grad returns `(dx, dγ, dβ)`. dγ and dβ are **assigned** (`=`); the LayerNorm, InstanceNorm and GroupNorm grads accumulate with `+=`.
- FP32 only.

## Completeness review

- Complete: no.
  - **No running mean/var and no inference mode.** With batch 1 the output is always β: probe `[5,-2,7]` with B=1 gave [0 0 0].
  - On NCHW, statistics are per (c,h,w) element, not per channel as in `BatchNorm2d`/TF conv BN. Probe vs a per-channel reference on `[2,3,2,2]`: maxAbsErr 1.59.
  - Imported TFLite/ONNX BN layers cannot be expressed: they need folded per-channel scale and shift.
- Tested: `TestBatchNormForward` (normalizations_test.go:11-66) asserts the shape, that values are non-zero, and the rank-1 copy case. **No value checks.** No grad test. Pass (2026-11-09).
- Correct: **no**.
  - Forward: matches its own per-feature definition.
  - `BatchNormGrad` dx uses `Σdy` and `Σdy·x̂` where it needs `γ·Σdy` and `γ·Σdy·x̂` (normalization.go:445-461), so it is wrong whenever γ≠1. Probe: γ≠1 gives maxAbsErr 5.63 vs central differences; γ=1 gives 1.6e-4.
  - Eager forward with nil dst calls `NewAs(t)`, which **aliases the input buffer**. The input is overwritten, so a later `BatchNormGrad(input=t)` sees normalised data (normalizations.go:20-22; tensor.go:42-44).
  - The rank<2 branch calls `t.Copy(dst)`, which copies dst into t, the wrong direction (normalizations.go:31).
  - Offsets and strides are ignored.

## Embedded suitability

Two `Pool` buffers of featureSize floats per call (normalization.go:23-26). It divides by `Sqrt` per element instead of using a hoisted invStd (normalization.go:61). float32 only.

## TODOs

- [ ] Fix BatchNormGrad dx for γ≠1 (multiply the reduction sums by γ[f]) — x/math/primitive/fp32/normalization.go:445-461; probe maxAbsErr 5.63 #p1 #ER-020
- [ ] Add running mean/var and an inference path (scale/shift with stored stats) — normalizations.go:16-52 #p1 #ER-020
- [ ] Add per-channel (NCHW, reduce over B,H,W) BatchNorm2d semantics — fp32/normalization.go:12-75 #p2
- [ ] NewAs aliasing: allocate a new dst when nil; fix t.Copy(dst) direction — x/math/tensor/eager/normalizations.go:20-22,31 #p1 #ER-020
- [ ] Add value tests for forward (vs reference) and grad (numerical) — x/math/tensor/eager/normalizations_test.go:11-66 #p2

## Links

- Module: [[tensor]]
- Related: [[layer-norm]], [[rms-norm]], [[instance-and-group-norm]], [[nn]]
