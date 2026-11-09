---
type: algorithm
status: partial
tags: [easyrobot, mat, qr, householder]
workstream: W9
module: "[[mat-vec]]"
summary: "Q orthonormal but Q·R ≠ A (error 3–12, `qr.go:57,109`); tests check lengths only"
source: [x/math/mat/qr.go, x/math/mat/types/types.go]
tests: [x/math/mat/qr_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): TestMatrix_QRDecompose pass (5 subtests; verify helper checks only lengths of Q/C/D); probe InfraNotes/matprobe2: Q orthonormal (|QᵀQ−I|≈1e-7) but |Q·R−A| = 8.45 (2×2), 3.18 (3×3), 12.3 (3×2), 4.71 (4×2)"
---
# QR decomposition (Matrix.QRDecompose / Matrix.QR)

**What it does:** Householder QR via `fp32.Geqrf` (in place), storing Householder constants in `C` and R-diagonal in `D` ("compatibility" with an older C/D format), then `QR()` rebuilds explicit `Q (m×m)` with `fp32.Orgqr` and `R (n×n)`.

**Where:** `x/math/mat/qr.go:20-84` `QRDecompose(dst *QRResult)` (modifies the receiver; `dst.Q = m`), `:88-143` `QR(dst)`. Result type `x/math/mat/types/types.go:103` (`QRResult{Q, R, C, D, Singular}`).

**Inputs → outputs:** m×n float32, m ≥ n (error otherwise `:28-30`); outputs Q, R, `Singular` flag (|D[k]| < 1e-10).

## Completeness review

- Complete: **no** — no solve/least-squares helper; m < n rejected.
- Tested: `TestMatrix_QRDecompose` — identity, 2×2, 3×2, empty, rows<cols. `verifyQRDecomposition` (`qr_test.go`) checks only that Q/C/D exist and have the right lengths (`_ = Q // … if additional checks are added later`). **No value check.**
- Correct: **no** — probe: Q is orthonormal but `Q[:, :n]·R` does not reproduce A for any tested shape (errors 3–12). Contributing causes visible in code: tau for the last reflector is dropped (`qr.go:57,109-112`, `C[k]` only for `k < minMN-1`, `tau[minMN-1] = 0`) which loses a reflector for m > n; R diagonal is taken from `D` captured before/after Geqrf convention mismatch [INFERENCE]. No production caller (`grep` 2026-11-09).

## TODOs

- [ ] Fix tau/C handling and R extraction; add Q·R ≈ A and QᵀQ ≈ I assertions to verifyQRDecomposition — x/math/mat/qr.go:57,109, x/math/mat/qr_test.go #p1 #ER-021
- [ ] Do not mutate the receiver silently (document or copy) — x/math/mat/qr.go:33 #p3
- [ ] Add least-squares solve (Rx = Qᵀb) for overdetermined fits — proposal #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-svd]], [[mat-nnls]], [[primitive-kernels]]
