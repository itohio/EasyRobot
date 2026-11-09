---
type: algorithm
status: tested
tags: [easyrobot, mat, cholesky, linear-algebra]
workstream: W9
module: "[[mat-vec]]"
summary: "correct; re-factors and allocates on every solve"
source: [x/math/mat/cholesky.go]
tests: [x/math/mat/cholesky_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09): TestMatrix_Cholesky (5 subtests: identity → L=I, 2×2 PD, not-PD error, non-square, empty) and TestMatrix_CholeskySolve (1) pass; probe InfraNotes/matprobe2: A=(4 2; 2 3), b=(2 1) → x=(0.5 0) ✔"
---
# Cholesky decomposition and solve

**What it does:** `A = L·Lᵀ` for symmetric positive-definite A (Cholesky–Banachiewicz, row by row with `fp32.Dot`), and `CholeskySolve` = factor + forward + back substitution.

**Where:** `x/math/mat/cholesky.go:21-72` `Matrix.Cholesky(dst)` (zeros dst, error if pivot ≤ 0), `:77-134` `Matrix.CholeskySolve(b, dst)`.

**Inputs → outputs:** n×n float32 (only lower triangle read; symmetry not checked); dst n×n lower-triangular; `b`, `dst` vectors length n.

## Completeness review

- Complete: yes (factor + solve).
- Tested: value checks on L for identity and a 2×2 PD matrix, error paths, one 2×2 solve; probe solve correct.
- Correct: yes within tests.

**Notes:** `CholeskySolve` re-factors every call (no reuse of L), allocates L (`:90`) and y (`:98`), and calls `L.Flat()` inside the back-substitution loop (`:121`, cheap when contiguous). Symmetry of A is assumed, not validated.

**Embedded suitability:** float32, O(n³/6), allocation per solve (2 buffers); no goroutines; good fit for small SPD systems (EKF covariance, least-squares normal equations).

## TODOs

- [ ] Solve-with-precomputed-L API and caller buffers — x/math/mat/cholesky.go:77-98 #p3
- [ ] Larger/ill-conditioned tests (e.g. 6×6 normal equations) — x/math/mat/cholesky_test.go #p3

## Links

- Module: [[mat-vec]]
- Related: [[matrix-inverse-and-lu]], [[mat-qr]]
