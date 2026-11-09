---
type: algorithm
status: partial
tags: [easyrobot, linear-algebra, fp32]
workstream: [W9]
module: "[[primitive-kernels]]"
summary: "QR does not reconstruct A (error 3.1, `la.go:542-660`); `Gesvd` returns V not Vᵀ, unsorted; pinv…"
source: [x/math/primitive/fp32/la.go]
tests: [x/math/primitive/fp32/la_test.go, x/math/primitive/fp32/householder_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestGetrf_IP/Getri/Geqrf/Orgqr/Gesvd/Gepseu/Gnnls/H1-3/G1/G2 pass; probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): Geqrf+Orgqr do not reconstruct A (err 2.3–3.1), Gesvd returns V not Vᵀ and unsorted s, Gepseu fails for M<N"
---
# fp32 dense linear algebra: LU, inverse, QR, SVD, pseudo-inverse, NNLS

**What it does:** LAPACK-named float32 routines on row-major matrices. They come from Numerical Recipes (Golub–Reinsch SVD) and Lawson–Hanson (H1/H2/H3 Householder, NNLS).

**Where:** `x/math/primitive/fp32/la.go`:

| Routine | Lines |
|---|---|
| helpers `pytag` | 60-70 |
| Givens `G1`/`G2` | 78-109 |
| Householder `H1`/`H2`/`H3` | 140-324 |
| `Getrf_IP` (LU, partial pivoting) | 331-390 |
| `Getrf` (L and U split) | 399-448 |
| `Getri` (inverse from LU) | 454-519 |
| `Geqrf` (QR) | 526-598 |
| `Orgqr` (form Q) | 603-664 |
| `Gepseu` (pinv via SVD) | 671-735 |
| `Gesvd` | 755-1101 |
| `Gnnls` | 1112-1452 |

**Inputs → outputs:** row-major `a` with `ldA`. Errors are returned (`ErrSingularMatrix`, `ErrBadDimensions`, `ErrMaxIterations`, `ErrNotSquare`).

## Completeness review

- Complete: no.
  - There is no linear solve (`Gesv`/`Getrs`), no Cholesky (`Potrf`), no symmetric eigen-decomposition, and no least-squares QR solve.
  - `Gesvd` requires M ≥ N (la.go:760-762), so `Gepseu` fails for wide matrices. Probe 3×4 returned `bad matrix dimensions`.
  - `Getri` requires `ldA==ldInv==N` (la.go:458-460).
- Tested: la_test.go and householder_test.go pass (2026-11-09), but the tests are weak:
  - `TestGeqrf` checks the identity case plus "no error" on 3×2 (la_test.go).
  - `TestOrgqr` checks only QᵀQ≈I with tolerance 0.1.
  - `TestGesvd` reconstructs only the identity (where V = Vᵀ).
  - `TestGnnls` checks only x ≥ 0 and residual ≤ 1.
- Correct: **no**.
  - **QR is broken.** `Geqrf` overwrites the Householder vector's head `a[k][k]` with R's diagonal `−scale·σ` (la.go:588). `Orgqr` then builds reflectors from `a[i][k]`, `i≥k`, including that overwritten diagonal (la.go:636-660). Probe on a fixed 3×3: Q is orthogonal (err 4.8e-7) but |QR−A|=3.14 and |QᵀR−A|=1.76. Random 4×3 gives err 2.31.
  - For M>N, `Geqrf` skips the last column's reflector (`k < minMN−1`, la.go:542; `tau[N−1]=0` at la.go:593-595), so R(N−1,N−1) is not the column norm.
  - **`Gesvd` returns V in `vt`, not Vᵀ** as documented (la.go:742). Probe on random 4×3: reconstruction with vt as Vᵀ has err 2.83; as V, 7.2e-7. `Gepseu` works around this by transposing (la.go:697-707, comment "Gesvd returns V").
  - **The singular values are not sorted** despite the "sorted descending" doc: probe s=[2.35 1.12 1.66].
  - `Gepseu` 4×3: ‖APA−A‖ = 4.2e-7, correct.
  - `Getrf_IP`/`Getri` use an absolute pivot threshold `|a_kk| < 1e-6` (la.go:372, 511). This is scale-dependent: well-conditioned matrices with tiny entries are reported singular, and nearly singular large matrices pass. Values are verified by `TestGetri` (M·M⁻¹ = I on 2×2).
  - The `Gepseu` tolerance is a fixed `1e-10` (la.go:712), below float32 eps·σmax, so tiny noisy singular values get inverted.

## Embedded suitability

`Pool` buffers per call: `Getrf` (M·ldA), `Gesvd` (M·N + N), `Gepseu` (M² + 2N² + N). `Gnnls` allocates `make([]int, N)` (la.go:1137). float32 `math32` throughout. SVD has up to 30 iterations per singular value. No generics or goroutines. Fine for small (≤12×12) kinematics/calibration matrices on MCU.

## TODOs

- [ ] Fix QR: store the Householder vector head separately (or follow LAPACK's tau/v convention) so Orgqr can rebuild Q; apply the last reflector for M>N; add a Q·R=A reconstruction test — x/math/primitive/fp32/la.go:542-595,636-660; probe |QR−A|=3.14 #p1 #ER-020
- [ ] Gesvd: return Vᵀ as documented (or rename the parameter to v) and sort singular values descending with matching U/V columns; add a random-matrix reconstruction test — la.go:743-1101; probe recon err 2.83 #p1 #ER-020
- [ ] Support M<N in Gesvd/Gepseu (transpose internally) — la.go:760-762 #p2
- [ ] Use relative pivot/singular tolerances (eps·‖A‖, eps·max(M,N)·σmax) — la.go:372,511,712 #p2
- [ ] Add Getrs/Gesv, Potrf (Cholesky) and a symmetric eigensolver for control/estimation users — la.go #p2
- [ ] Strengthen TestGnnls (compare with known NNLS solutions) — la_test.go #p2

## Links

- Module: [[primitive-kernels]]
- Related: [[fp32-blas-level1-2]], [[fp32-gemm]], [[mat-vec]], [[kinematics]]
