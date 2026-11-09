---
type: algorithm
status: tested
tags: [easyrobot, nn, blas, gemm, fp32]
workstream: [W9, W2]
module: "[[primitive-kernels]]"
summary: "correct; i-j-k loop with column-strided B, no blocking/SIMD (2.3–4.8 GFLOP/s on M5 by the kernel…"
source: [x/math/primitive/fp32/level3.go, x/math/primitive/fp32/batched.go]
tests: [x/math/primitive/fp32/level3_test.go, x/math/primitive/fp32/batched_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → TestGemm_NN/NT/TN/TT, TestGemmBatched, TestGemmStrided pass; bench darwin/arm64 Apple M5 go1.27.2 (2026-11-09)"
---
# fp32 GEMM (level 3) and batched GEMM

**What it does:** Computes `C = α·op(A)·op(B) + β·C` on row-major float32 matrices in plain Go, plus a batch loop over that.

**Where:** `x/math/primitive/fp32/level3.go`: `Gemm_NN` 9-66, `Gemm_NT` 74-130, `Gemm_TN` 138-195, `Gemm_TT` 203-257 (`Syrk` 265-343 and `Trmm` 354-612 are in the same file; see below). `batched.go`: `GemmBatched` 10-24, `GemmStrided` 34-48, `GemvBatched` 57-71.

**Inputs → outputs:** `Gemm_XY(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)`.
- NN: A is M×K, B is K×N.
- NT: B is stored N×K.
- TN: A is stored K×M.
- C is M×N.
- Batched variants take `batchCount` and element strides `stridea/b/c` between matrices.

## Algorithm / loop order

- Step 1: a separate beta pass scales or zeroes C. β==0 zeroes C, so NaN in C does not leak through.
- Step 2: for every `(i, j)` a scalar dot product over k, with a 4-way manual unroll.
  - `Gemm_NN` (level3.go:38-65) uses i-j-k order and walks B down a column with stride `ldB`, so every k step lands on a new cache line.
  - `Gemm_NT` is the cache-friendly one: both operands are read along rows.
  - `Gemm_TN` and `Gemm_TT` read A down a column.
- No blocking, no packing, no register tiling, no SIMD, no goroutines.
- `GemmBatched` and `GemmStrided` are byte-for-byte the same loop over `Gemm_NN` (batched.go:15-23 and 39-47). The names suggest a pointer-array variant versus a strided variant, but there is no difference.

## Completeness review

- Complete: yes, for the 4 transpose combinations and the strided batch. There is no batched NT/TN.
  - `K==0` returns early without applying β (level3.go:10-12). BLAS gives C=βC. Probe: C=[5 5] with β=0 stayed [5 5].
- Tested: `TestGemm_NN`, `TestGemm_NT`, `TestGemm_TN`, `TestGemm_TT` cover hand-computed 2×2/2×3/3×2 cases with α/β and ld padding (`InDeltaSlice`). `TestGemmBatched`, `TestGemmStrided`, `TestGemvBatched` and `TestLevel3Empty` also run. All pass (2026-11-09).
- Correct: yes for Gemm/batched, by those value tests. The eager `MatMulTransposed` TN/NT probe matched a reference (see [[tensor-matmul]]).

## Performance (measured)

`go test -run '^$' -bench BenchmarkTensorMatMul_Destination -benchtime 3x ./x/math/tensor/eager/` on darwin/arm64 Apple M5, go1.27.2, which calls `Gemm_NN`:

| Case | ns/op | Throughput |
|---|---|---|
| 100×50·50×100 | 206,972 | ≈4.8 GFLOP/s |
| 256×128·128×256 | 4,401,861 | ≈3.8 GFLOP/s |
| 512×256·256×512 | 57,875,569 | ≈2.3 GFLOP/s, falling as B leaves cache |
| batched 32×(64×128·128×64) | 8,851,917 | ≈3.0 GFLOP/s |

Every case does 4 allocs/op (256 B, wrapper only).

Reports in the repo state other numbers (declared, not re-checked): `x/math/primitive/fp32/PERFORMANCE_REPORT.md` (Go vs "assembly" MatMul ratios; no assembly exists for darwin/arm64), `BENCHMARK_REPORT.md`, `x/math/tensor/BENCHMARKS.md`.

A Qwen-0.5B decode step is dominated by GEMV/GEMM over ~0.5 G weights. At ~3 GFLOP/s that is roughly 0.3 s/token on M5 [INFERENCE]. An ESP32 will be orders of magnitude slower.

## Other level 3 in the same file

- `Syrk`: tested by `TestSyrk` (values).
- `Trmm`: **wrong outside the left/no-transpose case**. Only `side=L, trans=N` is tested (`TestTrmm`).
  - The left/transpose branches read `c[k*ldC+j]` instead of `b` (level3.go:396, 408, 452, 463). Probe L,U,T gave [0 0 0]; the correct result is [1 6 14].
  - The right-side no-transpose branches compute B·Aᵀ (`a[j*ldA+k]`, level3.go:539, 550, 592, 603). Probe R,U,N gave [6 9 6]; the correct result is [1 6 14].

## Embedded suitability

No allocation in the kernels. float32 only, no `any` dispatch, no generics, no goroutines. On ESP32-S3 the missing pieces are: no PIE/SIMD use, no int8/fixed-point GEMM (the qi8/qu8 packages are not wired in), and column-strided B in `Gemm_NN` thrashes the small cache. For inference, pre-transposing weights and calling `Gemm_NT` / `Gemv_N` is the cheap win [INFERENCE].

## TODOs

- [ ] Fix `Trmm` transpose (reads C instead of B) and right-side (computes B·Aᵀ) branches; add tests for all side/uplo/trans combos — x/math/primitive/fp32/level3.go:383-612 #p1 #ER-020
- [ ] Apply β when K==0 (BLAS semantics) — level3.go:10-12 (and NT/TN/TT equivalents) #p3
- [ ] Add cache blocking/packing (or k-inner over contiguous B rows, i-k-j order) to `Gemm_NN` — level3.go:38-65; bench above #p2
- [ ] Merge or differentiate `GemmBatched` and `GemmStrided` (identical code) — batched.go:10-48 #p3
- [ ] Add int8/fp16 GEMM paths for embedded/LLM weights — no implementation in fp32/ #p2

## Links

- Module: [[primitive-kernels]]
- Related: [[fp32-blas-level1-2]], [[tensor-matmul]], [[im2col-col2im]], [[conv2d-kernel]]
