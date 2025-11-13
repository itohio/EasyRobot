# BLAS Function Implementation Map

This document maps BLAS function names to their implementations in the `primitive` package.

## BLAS Level 1: Vector Operations

All Level 1 functions are implemented in `level1.go`.

| BLAS Function | Our Function | Implementation | Status |
|---------------|--------------|----------------|--------|
| **AXPY** | `Axpy(y, x, strideY, strideX, n, alpha)` | `level1.go` | ✅ |
| **DOT** | `Dot(x, y, strideX, strideY, n)` | `level1.go` | ✅ |
| **NRM2** | `Nrm2(x, stride, n)` | `level1.go` | ✅ |
| **ASUM** | `Asum(x, stride, n)` | `level1.go` | ✅ |
| **SCAL** | `Scal(x, stride, n, alpha)` | `level1.go` | ✅ |
| **COPY** | `Copy(y, x, strideY, strideX, n)` | `level1.go` | ✅ |
| **SWAP** | `Swap(x, y, strideX, strideY, n)` | `level1.go` | ✅ |
| **IAMAX** | `Iamax(x, stride, n)` | `level1.go` | ✅ |

## BLAS Level 2: Matrix-Vector Operations

All matrix operations are row-major.

All Level 2 functions are implemented in `level2.go`.

| BLAS Function | Our Function | Implementation | Status |
|---------------|--------------|----------------|--------|
| **GEMV_N** | `Gemv_N(y, a, x, ldA, M, N, alpha, beta)` | `level2.go` | ✅ |
| **GEMV_T** | `Gemv_T(y, a, x, ldA, M, N, alpha, beta)` | `level2.go` | ✅ |
| **GER** | `Ger(a, x, y, ldA, M, N, alpha)` | `level2.go` | ✅ |
| **SYMV** | `Symv(y, a, x, ldA, N, alpha, beta, uplo)` | `level2.go` | ✅ |
| **TRMV** | `Trmv(y, a, x, ldA, N, uplo, trans, diag)` | `level2.go` | ✅ |

## BLAS Level 3: Matrix-Matrix Operations

All matrix operations are row-major.

All Level 3 functions are implemented in `level3.go`.

| BLAS Function | Our Function | Implementation | Status |
|---------------|--------------|----------------|--------|
| **GEMM_NN** | `Gemm_NN(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)` | `level3.go` | ✅ |
| **GEMM_NT** | `Gemm_NT(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)` | `level3.go` | ✅ |
| **GEMM_TN** | `Gemm_TN(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)` | `level3.go` | ✅ |
| **GEMM_TT** | `Gemm_TT(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)` | `level3.go` | ✅ |
| **SYRK** | `Syrk(c, a, ldC, ldA, N, K, alpha, beta, uplo)` | `level3.go` | ✅ |
| **TRMM** | `Trmm(c, a, b, ldC, ldA, ldB, M, N, alpha, beta, side, uplo, trans, diag)` | `level3.go` | ✅ |

## Batched BLAS Operations

All batched operations are implemented in `batched.go`.

| BLAS Function | Our Function | Implementation | Status |
|---------------|--------------|----------------|--------|
| **Batched GEMM** | `GemmBatched(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta, batchCount, stridea, strideb, stridec)` | `batched.go` | ✅ |
| **Strided GEMM** | `GemmStrided(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta, batchCount, stridea, strideb, stridec)` | `batched.go` | ✅ |
| **Batched GEMV** | `GemvBatched(y, a, x, ldA, M, N, alpha, beta, batchCount, strideA, strideX, strideY)` | `batched.go` | ✅ |

## Quantized Operations

All quantized INT8 operations are implemented in `quantized.go`.

| Operation | Our Function | Implementation | Status |
|-----------|--------------|----------------|--------|
| **Copy_Q8** | `Copy_Q8(y, x, strideY, strideX, n)` | `quantized.go` | ✅ |
| **Gemm_NN_Q8** | `Gemm_NN_Q8(output, input, weight, ldOutput, ldInput, ldWeight, M, N, K, inputScale, weightScale, outputScale, inputZero, weightZero, outputZero)` | `quantized.go` | ✅ |
| **Gemm_NN_Q8_Accum** | `Gemm_NN_Q8_Accum(output, input, weight, ldOutput, ldInput, ldWeight, M, N, K, inputZero, weightZero)` | `quantized.go` | ✅ |
| **Conv2D_Q8** | `Conv2D_Q8(output, input, weights, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, bias, inputScale, weightScale, outputScale, inputZero, weightZero, outputZero)` | `quantized.go` | ✅ |
| **Im2Col_Q8** | `Im2Col_Q8(col, im, batchSize, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW)` | `quantized.go` | ✅ |
| **Col2Im_Q8** | `Col2Im_Q8(im, col, batchSize, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW)` | `quantized.go` | ✅ |
| **GemmBatched_Q8** | `GemmBatched_Q8(output, input, weight, ldOutput, ldInput, ldWeight, M, N, K, inputScale, weightScale, outputScale, inputZero, weightZero, outputZero, batchCount, strideOutput, strideInput, strideWeight)` | `quantized.go` | ✅ |

## Tensor Operations

All tensor operations are implemented in `tensor.go`.

| Operation | Our Function | Implementation | Status |
|-----------|--------------|----------------|--------|
| **Conv2D** | `Conv2D(output, input, weights, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, bias)` | `tensor.go` | ✅ |
| **Conv2DTransposed** | `Conv2DTransposed(output, input, weights, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, bias)` | `tensor.go` | ✅ |
| **Im2Col** | `Im2Col(col, im, batchSize, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW)` | `tensor.go` | ✅ |
| **Col2Im** | `Col2Im(im, col, batchSize, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW)` | `tensor.go` | ✅ |

## Convolution Operations

Convolution operations are implemented in `conv.go` (legacy, to be merged).

| Function | Our Function | Implementation | Status |
|----------|--------------|----------------|--------|
| **Convolve1D** | `Convolve1DAdd(dst, vec, kernel, N, M, stride, transposed)` | `conv.go` | ✅ |
| **Convolve2D** | `Convolve2DAdd(dst, mat, kernel, N, M, K, L, stride, transposed)` | `conv.go` | ❌ **NOT IMPLEMENTED** |

**Note**: `Convolve1DAdd` performs accumulation: `dst += conv(vec, kernel)`. For tensor API consistency, a dst-based version `Convolve1D(dst, vec, kernel, ...)` is planned (see `ALIGN_WITH_TF_PLAN.md` Phase 6).

## Non-BLAS Utility Functions

These functions exist in `array.go` and `vector.go` for tensor operations and statistics:

### From `array.go`:
- `SumArr`, `DiffArr`, `MulArr`, `DivArr` - Element-wise operations (for tensor ops)
- `Sum`, `SqrSum` - Utility reductions for statistics
- `StatsArr` - Computes min, max, mean, and standard deviation in one pass
- `PercentileArr` - Computes percentile value and sum of values above percentile
- `DiffArrInPlace(dst, c, num)` - In-place scalar subtraction: `dst[i] -= c`
  - **Note**: For tensor API consistency, a dst-based version `DiffArrScalar(dst, src, c, ...)` is planned (see `ALIGN_WITH_TF_PLAN.md` Phase 6)
- `MulArrInPlace` - **DEPRECATED**: Use `Scal` from level1.go instead, kept for backward compatibility

**Removed (replaced by BLAS operations):**
- `SumArrConst`, `DiffArrConst`, `MulArrConst`, `DivArrConst` → Use `Axpy`, `Scal` from level1.go
- `MinArr`, `MaxArr`, `MeanArr`, `MomentsArr` → Use `StatsArr`
- `WeightedMomentsArr` → Removed (not needed)

### From `vector.go`:
- `HadamardProductAdd(dst, a, b, num, strideA, strideB)` - Element-wise product and add: `dst[i] += a[i] * b[i]` (accumulation pattern)
  - **Note**: For tensor API consistency, a dst-based version `HadamardProduct(dst, a, b, ...)` is planned (see `ALIGN_WITH_TF_PLAN.md` Phase 6)
- `DotProduct2D(a, b, N, M, K, L)` - 2D matrix dot product (specialized, not BLAS)
- `NormalizeVec(dst, num, stride)` - Vector normalization in-place: `dst = dst / ||dst||` (uses `Nrm2` from level1.go)
  - **Note**: For tensor API consistency, a dst-based version `NormalizeVecTo(dst, src, ...)` is planned (see `ALIGN_WITH_TF_PLAN.md` Phase 6)
- `SumArrInPlace(dst, c, num)` - In-place scalar addition: `dst[i] += c`
  - **Note**: For tensor API consistency, a dst-based version `SumArrScalar(dst, src, c, ...)` is planned (see `ALIGN_WITH_TF_PLAN.md` Phase 6)
- `DotProduct` - **DEPRECATED**: Use `Dot` from level1.go, kept for backward compatibility

**Removed (replaced by BLAS operations):**
- `OuterProduct`, `OuterProductConst`, `OuterProductAddConst` → Use `Ger` from level2.go

## Status Legend

- ✅ **Implemented**: Function is complete and tested
- ⏳ **In Progress**: Function is being implemented
- 🔮 **Planned**: Function is planned but not yet started
- ❌ **Not Implemented**: Function is not implemented (may be mentioned but doesn't exist)

## Operation Patterns

**BLAS Operations**: Maintain standard in-place patterns for BLAS compatibility:
- `Axpy(y, x, ...)` - `y = alpha*x + y` (modifies y in-place) ✅ **KEEP AS-IS**
- `Scal(x, ...)` - `x = alpha*x` (modifies x in-place) ✅ **KEEP AS-IS**

**Tensor Operations**: Should follow `Operation(dst, src, ...)` pattern (see `ALIGN_WITH_TF_PLAN.md` for details):
- Most tensor operations already follow this pattern
- Some operations (e.g., `HadamardProductAdd`, `Convolve1DAdd`) use accumulation pattern and need dst-based alternatives

## Migration Path

1. Replace `DotProduct` → `Dot` from level1.go
2. Replace manual norms → `Nrm2`, `Asum` from level1.go
3. Replace `SumArrConst` variants → `Axpy`, `Scal` from level1.go
4. Consolidate outer product → `Ger` from level2.go (when implemented)

