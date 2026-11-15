---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, normalization]
workstream: [W9]
module: "[[tensor]]"
summary: "forward correct; dx wrong for γ ≠ 1 (maxAbsErr 3.43)"
source: [x/math/tensor/eager/normalizations.go, x/math/primitive/fp32/normalization.go]
tests: [x/math/tensor/eager/normalizations_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestLayerNormForward pass (shape only); probe (2026-11-09): forward maxAbsErr 2.4e-7 vs float64 ref; LayerNormGrad dx maxAbsErr 3.43 for gamma≠1 (1.1e-4 for gamma=1)"
---
# Layer normalization (forward + grad)

**What it does:** Computes `y = (x − μ)/√(σ² + ε)·γ + β`, with μ and σ² taken over the **last dimension** of each row.

**Where:**
- Eager `x/math/tensor/eager/normalizations.go`: `LayerNormForward` 136-166, `LayerNormGrad` 177-231.
- fp32 `x/math/primitive/fp32/normalization.go`: `LayerNormForward` 81-131, `LayerNormGrad` 473-543.

**Inputs → outputs:**
- x `[..., D]`; γ and β are `[D]` or nil.
- ε is inside the square root; the variance is biased (/D); two-pass mean then variance.
- `LayerNormGrad(dxDst, dγDst, dβDst, dy, input, γ, ε)`: dγ and dβ **accumulate** (`+=`). When nil they are freshly zeroed; a caller-provided buffer must be zeroed by the caller.
- FP32 only. Assumes a contiguous buffer.

## Completeness review

- Complete: yes for last-axis LN. There is no normalized-shape over several trailing dims.
- Tested: `TestLayerNormForward` (normalizations_test.go:68-104) checks **shape and length only**. No grad test. Pass (2026-11-09).
- Correct: **no**.
  - Forward: correct. Probe `[2,4]` with γ, β: maxAbsErr 2.4e-7 vs a float64 reference.
  - `LayerNormGrad` dx uses `Σdy` and `Σdy·x̂` (normalization.go:513-541). The correct form uses `Σγᵢdyᵢ` and `Σγᵢdyᵢx̂ᵢ`, so the result is wrong whenever γ is not all ones. Probe: γ=[0.5,2,-1,1.5] gives maxAbsErr 3.43 (|dx|≈4.8) vs central differences; γ=1 gives 1.1e-4.
  - The dγ and dβ formulas are correct.
  - Eager forward with nil dst **aliases the input**: `NewAs` shares data (normalizations.go:140-142, tensor.go:42-44). Probe: the input [1 2 3] became [-1.22 0 1.22].

## Embedded suitability

No allocation in the kernel; the eager op uses no allocation even when dst is nil, because of the aliasing bug. float32 with `math32`. Three passes per row (mean, var, apply). A Welford single pass would save one pass, which matters on MCU.

## TODOs

- [ ] Fix LayerNormGrad dx for γ≠1 (use γ-weighted sums) — x/math/primitive/fp32/normalization.go:513-541; probe maxAbsErr 3.43 #p1 #ER-020
- [ ] NewAs aliasing: allocate a dst when nil (input currently overwritten) — x/math/tensor/eager/normalizations.go:140-142 #p1 #ER-020
- [ ] Add forward value tests and a numerical grad test — x/math/tensor/eager/normalizations_test.go:68-104 #p2
- [ ] Document dγ/dβ accumulate semantics (BatchNormGrad assigns instead) — normalization.go:507-511 #p3

## Links

- Module: [[tensor]]
- Related: [[rms-norm]], [[batch-norm]], [[instance-and-group-norm]], [[nn]]
