---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, normalization]
workstream: [W9]
module: "[[tensor]]"
summary: "forward correct; dx and GN dγ/dβ wrong"
source: [x/math/tensor/eager/normalizations.go, x/math/primitive/fp32/normalization.go]
tests: [x/math/tensor/eager/normalizations_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestInstanceNorm2D/TestGroupNormForward pass (shape/fallback only); probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): IN forward 2.4e-7 vs ref, GN(groups=C)==IN; IN grad dx maxAbsErr 4.18 and GN grad dx 2.15 for gamma≠1; GN dgamma/dbeta wrong"
---
# Instance normalization and group normalization (forward + grad)

**What it does:**
- InstanceNorm2D normalises each (batch, channel) plane over H·W.
- GroupNorm normalises each (batch, group of C/G channels) over channels×spatial.
- Both apply a per-channel γ/β afterwards.

**Where:**
- Eager `x/math/tensor/eager/normalizations.go`: `InstanceNorm2D` 328-363, `InstanceNorm2DGrad` 374-432, `GroupNormForward` 439-472, `GroupNormGrad` 484-541.
- fp32 `x/math/primitive/fp32/normalization.go`: `InstanceNorm2D` 249-290, `GroupNormForward` 296-369, `InstanceNorm2DGrad` 608-666, `GroupNormGrad` 676-774.

**Inputs → outputs:**
- IN: x `[B,C,H,W]` (anything else panics); γ and β are `[C]`.
- GN: x `[B,C,…]`; γ and β are `[C]`. If `C % G != 0` it **silently copies the input** (normalization.go:305-308); the test asserts that behaviour.
- ε inside the square root; biased variance.
- Grads accumulate dγ and dβ (`+=`).
- FP32 only.

## Completeness review

- Complete: yes for forward and backward signatures. There are no affine=false or running-stats options for IN.
- Tested: `TestInstanceNorm2D` checks the shape and the invalid-shape panic. `TestGroupNormForward` checks the shape and the invalid-group copy fallback (normalizations_test.go:135-196). **No value checks; no grad tests.** Pass (2026-11-09).
- Correct: **no** (gradients). Forward is correct:
  - IN forward vs a float64 reference on `[2,3,2,2]` with γ, β: 2.4e-7.
  - GN with groups=C equals IN exactly (probe p2).
- Gradients:
  - IN and GN dx omit γ from the reduction sums (normalization.go:662, 769). Probe vs central differences: IN γ≠1 gives maxAbsErr 4.18; GN (2 groups) γ≠1 gives 2.15; GN γ=1 gives 1.7e-5.
  - GN dγ/dβ add the **whole group's** Σdy·x̂ and Σdy to every channel in the group (normalization.go:749-755). Per-channel sums are correct. Probe: dγ maxAbsErr 2.17, dβ 0.96.
  - Eager forwards with nil dst alias the input via `NewAs` (normalizations.go:332-334, 443-445).

## Embedded suitability

No allocations in the kernels. float32 `math32`. Three passes per group. Silent fallbacks: GN copies the input when G does not divide C or G ≤ 0, and GN grad returns without writing anything.

## TODOs

- [ ] Fix IN/GN dx for γ≠1 (γ-weighted sums) — x/math/primitive/fp32/normalization.go:640-664,733-771; probe maxAbsErr 4.18/2.15 #p1 #ER-020
- [ ] Fix GroupNormGrad dγ/dβ to per-channel sums — normalization.go:749-755; probe dγ err 2.17 #p1 #ER-020
- [ ] Panic or return an error instead of silently copying when C%G≠0 — normalization.go:297-308,682-684 #p2
- [ ] NewAs aliasing in eager forwards — x/math/tensor/eager/normalizations.go:332-334,443-445 #p1 #ER-020
- [ ] Add value and numerical-grad tests — x/math/tensor/eager/normalizations_test.go:135-196 #p2

## Links

- Module: [[tensor]]
- Related: [[batch-norm]], [[layer-norm]], [[rms-norm]]
