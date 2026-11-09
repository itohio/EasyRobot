---
type: algorithm
status: tested
tags: [easyrobot, mat, svd, linear-algebra]
workstream: W9
module: "[[mat-vec]]"
summary: "reconstruction value-tested; rows < cols unsupported"
source: [x/math/mat/svd.go, x/math/mat/types/types.go]
tests: [x/math/mat/svd_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09): TestMatrix_SVD pass, 8 subtests incl. reconstruction U·Σ·Vᵀ≈M (1e-4) for identity, diagonal, 4×3; M<N cases assert error; probe InfraNotes/matprobe: 4×3 S=[5.964 3.529 2.445] descending"
---
# Singular value decomposition (Matrix.SVD)

**What it does:** Thin wrapper over `fp32.Gesvd` (Golub–Reinsch: Householder bidiagonalisation + implicit QR, per Numerical Recipes) returning `U (m×m)`, `S (n)`, `Vt (n×n)` with `M = U·Σ·Vᵀ`.

**Where:** `x/math/mat/svd.go:31-92` `Matrix.SVD(dst *SVDResult)`; result type `x/math/mat/types/types.go:114` (`SVDResult{U, S, Vt}`), alias `types_alias.go:7`. Kernel `fp32.Gesvd` in [[primitive-kernels]].

**Inputs → outputs:** m×n float32 with **m ≥ n**; error for empty or m < n (`:40-42`, "transpose input first"). Allocates working copy (`:47`), `sVec`, `S`, `U` m×m, `Vt` n×n, plus a transposed copy of V (`:83-89`) — 5 allocations; Gesvd returns V, which is transposed into a new matrix.

## Completeness review

- Complete: **no** — m < n unsupported (no internal transpose), so callers with wide Jacobians (6×N, N>6) fail; see [[mat-pseudo-inverse]].
- Tested: `TestMatrix_SVD` — identity (S=1), diagonal, 4×3 overdetermined, zero matrix (S≈0), empty → error, three M<N cases → error; reconstruction checked with tolerance 1e-4 (`svd_test.go:179-210`).
- Correct: yes for m ≥ n within tests; probe 4×3 gave descending singular values.

**Embedded suitability:** float32; iterative (max-iteration error path); full U (m×m) is wasteful for tall matrices (thin U m×n would do); 5 heap allocations per call.

## TODOs

- [ ] Support m < n by transposing internally — x/math/mat/svd.go:40 #p2
- [ ] Thin-U option and caller-provided buffers — x/math/mat/svd.go:60 #p3
- [ ] Remove extra V transpose (have Gesvd return Vᵀ) — x/math/mat/svd.go:81-89 #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-pseudo-inverse]], [[mat-qr]], [[primitive-kernels]]
