---
type: algorithm
status: partial
tags: [easyrobot, nn, tensor, matmul]
workstream: [W9, W2]
module: "[[tensor]]"
summary: "contiguous correct; views ignored (maxAbsErr 72 on a slice view, 16 on a transpose view…"
source: [x/math/tensor/eager/linalg.go, x/math/tensor/eager/linalg_helpers.go]
tests: [x/math/tensor/eager/linalg_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → TestMatMul2D/Batched/To, TestDot, TestNorm, TestNormalize pass; probe tmp/easyrobot/tensor/kernelnotes (2026-11-09): wrong results on Slice/Transpose views"
---
# Eager MatMul, MatMulTransposed, MatVecMulTransposed, Dot, Norm

**What it does:** These are the eager tensor wrappers over the fp32 BLAS: matrix multiply (2D, batched, one-side broadcast), multiply with transpose flags, transposed mat-vec, dot/Frobenius product, L1/L2 norm and L2 normalise.

**Where:**
- `x/math/tensor/eager/linalg.go`:
  - `MatMul` 19-43 (dispatches to `matMul2D` 46-78, `matMulBatched` 81-124, `matMulSameBatch` 127-174, `matMulBroadcastFirst` 177-201, `matMulBroadcastSecond` 204-234)
  - `prepareMatMulTarget`/`finalizeMatMulTarget` 236-272
  - `Dot` 441-477, `Norm` 517-555, `L2Normalize` 644-687
- `linalg_helpers.go`: `MatVecMulTransposed` 19-77, `MatMulTransposed` 86-330, `AddScaled` 339-404.

**Inputs → outputs:**
- `MatMul`: `[M,K]×[K,N]→[M,N]`, `[...B,M,K]×[...B,K,N]`, or broadcast of one side with batch 1. The broadcast-first output is always rank 3 `[B,M,N]`, so leading batch dims are flattened (linalg.go:178).
- `MatMulTransposed(dst, other, tA, tB)` calls `Gemm_{NN,NT,TN,TT}`. In the batched case it requires equal ranks and equal batch dims.
- `MatVecMulTransposed(dst, A[M,N], x[M], α, β)` → `[N]`.
- `Dot` and `Norm` return `float64`. `Norm` uses `ord` 0=L1, 1=L2, 2=Frobenius (non-standard numbering).
- FP32 only: every entry point panics `unsupported data type` for other dtypes (linalg.go:40, linalg_helpers.go:74, 325).

## Completeness review

- Complete: no.
  - No general batch broadcasting; batch configurations other than equal / 1×B / B×1 panic (linalg.go:123).
  - No matmul of a vector with a matrix.
  - `MatMulTransposed` with different ranks panics (linalg_helpers.go:329).
  - No fp16/int8.
- Tested: `TestMatMul2D`, `TestMatMulBatched`, `TestMatMulTo`, `TestDot`, `TestNorm`, `TestNormalize`, `TestTranspose2D`, `TestPermute`. All use contiguous, offset-0 tensors and check values. All pass (2026-11-09).
  - No test exists for `MatMulTransposed`, `MatVecMulTransposed` or `AddScaled` (benches only, `bench_test.go:1617-1660`).
- Correct: **no** for views. **The view metadata is ignored.** Every path passes `types.GetTensorData` (the full backing slice; `Data()` at tensor.go:103-111 ignores `offset`) to Gemm with `ld=K/N`, so offset and strides are dropped (linalg.go:65-66, 137-138, 184-185, 219-220; linalg_helpers.go:56-58, 199-201).

  | Probe (vs `At()`-based reference) | Result |
  |---|---|
  | Contiguous `MatMul` | err 0 |
  | `MatMulTransposed` TN/NT | err 0 |
  | `MatVecMulTransposed` | [5 7 9] correct |
  | `Slice(nil,0,2,2)` view (offset 6) · B | maxAbsErr 72 (computes rows 0-1 instead of 2-3) |
  | `Transpose(nil,nil)` view · B | maxAbsErr 16 |
  | `Dot(slice[4 5 6], self)` | 14; correct 77 |
  | `Norm(1)` of that slice | 3.74; correct 8.77 |

  - The `IsContiguous` branch in `matMulSameBatch` (linalg.go:140-171) picks `GemmBatched` for non-contiguous inputs, but that is the same strided loop, so nothing changes.
  - `dotStrided` and `dotFrobenius` ignore the offset; Frobenius also ignores strides (linalg.go:485-511).
  - `AddScaled`'s "strided case" is a flat `i` loop (linalg_helpers.go:372-375, 397-400).
  - `MatMulTransposed` accepts a `dst` that is an eager value view and writes the raw buffer (linalg_helpers.go:146-151), ignoring the dst offset.
  - `L2Normalize(dim=0)` normalises each row (linalg.go:728-737). `tf.nn.l2_normalize(axis=0)` normalises columns, so the comment "matches tf" is wrong.

## Embedded suitability

`MatMul` with nil dst allocates the result. With a dst that is non-contiguous or has an offset it allocates a temp and copies back (linalg.go:250-256). 4 allocs/op even with dst (bench). `MatMulTransposed` with nil dst allocates; `normalizeMatrixDim(1)` allocates once per column (linalg.go:743). Type switch on `Data() any` per call. Generic strided copy goes through `x/types`. `Dot`/`Norm` return float64 and the strided norms accumulate in float64 (linalg.go:559, 588), which is soft-float on ESP32.

## TODOs

- [ ] Honour `offset` and strides in MatMul/MatMulTransposed/MatVecMulTransposed/Dot/Norm/AddScaled: use `DataWithOffset()` and pass ld from strides, or materialise contiguous copies — x/math/tensor/eager/linalg.go:65-66, linalg_helpers.go:56-58,199-201,372-375; probe maxAbsErr 72/16 #p1 #ER-020
- [ ] Add value tests for MatMulTransposed (4 combos, 2D and batched), MatVecMulTransposed, AddScaled, and view inputs — x/math/tensor/eager/linalg_test.go #p2
- [ ] Support general batch broadcasting `[B1,1,M,K]×[1,B2,K,N]` (attention with GQA heads for Qwen) — linalg.go:116-123 #p2
- [ ] Fix L2Normalize axis semantics vs tf (dim 0 currently normalises rows) — linalg.go:728-751 #p3
- [ ] Drop the dead contiguity branch (GemmBatched == GemmStrided) — linalg.go:140-171 #p3

## Links

- Module: [[tensor]]
- Related: [[fp32-gemm]], [[fp32-blas-level1-2]], [[tensor-views-strides]]
