---
type: algorithm
status: partial
tags: [easyrobot, nn, blas, fp32]
workstream: [W9, W2]
module: "[[primitive-kernels]]"
summary: "`Trmv` transpose paths wrong and `TestTrmv` encodes the wrong outputs (`level2.go:211-279`)"
source: [x/math/primitive/fp32/level1.go, x/math/primitive/fp32/level2.go, x/math/primitive/fp32/vector.go, x/math/primitive/fp32/array.go]
tests: [x/math/primitive/fp32/level1_test.go, x/math/primitive/fp32/level2_test.go, x/math/primitive/fp32/vector_test.go, x/math/primitive/fp32/array_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → fp32 304/304 pass; probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): Trmv transpose paths wrong"
---
# fp32 BLAS level 1 and 2

**What it does:** Plain-Go, scalar, strided BLAS-style routines for `float32`: level 1 (vector) and level 2 (matrix–vector), plus some array helpers. Matrices are row-major with a leading dimension `ldA`.

**Where:**
- Level 1 `x/math/primitive/fp32/level1.go`: `Axpy` 9-22, `Dot` 26-42, `Nrm2` 46-61, `Asum` 65-84, `Scal` 88-99, `Copy` 103-116, `Swap` 120-133, `Iamax` 137-159.
- Level 2 `level2.go`: `Gemv_N` 9-48, `Gemv_T` 57-97, `Ger` 106-123, `Symv` 133-193, `Trmv` 203-305.
- Helpers: `vector.go` (`HadamardProduct[Add]`, `NormalizeVec[InPlace]`, `SumArrScalar`, `DotProduct2D`) and `array.go` (`Sum`, `SqrSum`, `StatsArr`, `PercentileArr`, `DiffArr*`).

**Inputs → outputs:** slices plus `stride`/`n` (level 1) or `ldA, M, N, alpha, beta` (level 2). `y = αAx + βy` (`Gemv_N`, x has length N), `y = αAᵀx + βy` (`Gemv_T`), `A += αxyᵀ` (`Ger`). `Trmv(y, a, x, ldA, N, uplo, trans, diag)` computes y = op(A)x.

## Completeness review

- Complete: yes. All routines listed above exist. There is no `Rot`/`Rotg`, `Trsv`, `Gbmv` or `Spmv`. Strides must be positive; nothing checks lengths, so a bad length panics on an out-of-range index.
- Tested: `TestAxpy`, `TestDot`, `TestNrm2`, `TestAsum`, `TestScal`, `TestCopy`, `TestSwap`, `TestIamax`, `TestGemv_N`, `TestGemv_T`, `TestGer`, `TestSymv`, `TestTrmv`, `TestLevel2Empty` and the vector/array tests compare values with `InDelta`. All pass in `go test -count=1 ./x/... (2026-11-09)`.
- Correct: **no** for `Trmv`; yes (by test) for the rest.
  - `Trmv` with uplo U and trans T (level2.go:211-231) computes `A·x`, not `Aᵀx`. It reads row `i`, `j≥i`. Probe with `A=[[1,2,3],[0,4,5],[0,0,6]]`, x=1: returned [6 9 6]; Aᵀx is [1 6 14].
  - `Trmv` with uplo L and trans T (level2.go:262-279) reads `a[j*ldA+i]` for `j≤i`, which is the zero upper triangle. Probe returned [1 4 6]; the correct result is [6 9 6].
  - `TestTrmv` (level2_test.go:458-535) encodes these wrong outputs. Its comments derive the correct values and then set `wantY` to the implementation's output ("Corrected based on implementation").
  - The doc comment says in-place `y==x` is allowed (level2.go:198). With uplo L and trans N, the loop runs `i` upward and reads `x[j<i]` after those entries are overwritten. Probe in-place returned [1 6 39]; the correct result is [1 6 14].
  - Minor: `Iamax` returns 0 for an all-NaN vector, and -1 for n=0, where BLAS returns 0 (level1.go:137-159). `Scal` skips the multiply when α==1, so NaN/Inf are not propagated.

## Embedded suitability

No allocation in level 1/2. `PercentileArr` takes a buffer from `Pool` and calls `sort.Slice` (array.go:125-133), which uses reflection and allocates a closure. Everything is float32 and goes through `math32`. No generics, no goroutines, no SIMD/unrolling. These loops are where a TinyGo/Xtensa build would get its speed from, because there is no assembly backend.

## TODOs

- [ ] Fix `Trmv` transpose branches (U/T reads A, L/T reads the wrong triangle) and the in-place ordering for L/N; fix `TestTrmv` expectations — x/math/primitive/fp32/level2.go:211-279, level2_test.go:458-535 #p1 #ER-020
- [ ] Return 0 for n=0 (not -1) and handle NaN in `Iamax` — level1.go:137-159 #p3
- [ ] Add `Trsv` (needed for triangular solves after LU/Cholesky) — level2.go #p2

## Links

- Module: [[primitive-kernels]]
- Related: [[fp32-gemm]], [[fp32-linear-algebra-decompositions]], [[tensor-matmul]]
