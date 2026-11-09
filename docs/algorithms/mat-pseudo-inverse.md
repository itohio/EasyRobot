---
type: algorithm
status: partial
tags: [easyrobot, mat, pseudo-inverse, dls, kinematics]
workstream: W9
module: "[[mat-vec]]"
summary: "fails for wide matrices (rows < cols, `la.go:760` Gesvd rejection) → DH IK with > 3 joints falls…"
source: [x/math/mat/pseudo_inverse.go]
tests: [x/math/mat/pseudo_inverse_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09): TestMatrix_PseudoInverse (4 subtests: square ≈ inverse, overdetermined, IK-style 12×6 Penrose identities, M<N expects error), TestMatrix_DampedLeastSquares (3), TestMatrix_DampedLeastSquares_LambdaComparison pass; probe InfraNotes/matprobe2: 3×2 pinv matches closed form; rank-1 (1 1; 1 1) → all 0.25 ✔; 3×4 → error"
---
# Moore–Penrose pseudo-inverse and damped least squares

**What it does:** `PseudoInverse`: A⁺ = V·Σ⁺·Uᵀ via `fp32.Gepseu` (SVD-based, small singular values zeroed). `DampedLeastSquares`: J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹ (Levenberg–Marquardt style) for IK near singularities.

**Where:** `x/math/mat/pseudo_inverse.go:18-42` `Matrix.PseudoInverse(dst)`, `:48-84` `Matrix.DampedLeastSquares(lambda, dst)` (allocates Jᵀ, JJᵀ, (JJᵀ+λ²I)⁻¹; inverse via pivoted dense `Matrix.Inverse`, see [[matrix-inverse-and-lu]]).

**Inputs → outputs:** m×n float32 → n×m. `PseudoInverse` requires **m ≥ n**; for m < n returns `ErrPseudoInverseFailed` (expected by the test `pseudo_inverse_test.go:110-120`, "not implemented in PseudoInverse"). DLS works for any shape (probe 3×4, λ=0.01 → ≈[I₃;0]).

**Consumers:** `x/math/control/kinematics/joints/dh/denavithartenberg.go:216` (J 6×N — fails for redundant arms N>6), `x/math/control/kinematics/thrusters/inverse.go:75-76` (falls back to its own damped pinv on error), `x/math/gps/trilateration.go:131`, `x/math/filter/savgol/savgol.go:149`, `x/math/colorscience/spd.go:479`.

## Completeness review

- Complete: **no** — underdetermined (wide) matrices unsupported in `PseudoInverse`; trivial fix A⁺ = ((Aᵀ)⁺)ᵀ.
- Tested: Penrose identities (A·A⁺·A ≈ A, A⁺·A·A⁺ ≈ A⁺, 5e-4) on square/overdetermined/12×6; DLS on square, overdetermined, singular; M<N error path.
- Correct: yes for m ≥ n (tests + probe incl. rank-deficient case).

**Embedded suitability:** float32; `PseudoInverse` allocations happen inside `fp32.Gepseu` (SVD workspace) [INFERENCE]; DLS allocates 4 matrices per call — per IK iteration on ESP32 this is GC pressure; provide workspace.

## TODOs

- [ ] Handle m < n by transposing (A⁺ = ((Aᵀ)⁺)ᵀ) and flip the test expectation — x/math/mat/pseudo_inverse.go:18, x/math/mat/pseudo_inverse_test.go:120 #p2
- [ ] Workspace-reusing variants for IK loops — x/math/mat/pseudo_inverse.go:61-75 #p3
- [ ] Expose singular-value cutoff (rcond) parameter — x/math/mat/pseudo_inverse.go:37 #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-svd]], [[matrix-inverse-and-lu]], [[mat-numeric-jacobian]], [[kinematics]]
