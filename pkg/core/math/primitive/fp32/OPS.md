# FP32 Primitive Operations Reference

This document catalogs all primitive operations implemented in the `fp32` package, organized by category. **Deprecated functions are marked for removal** in v2.0.0 as part of the tensor/fp32 consolidation effort.

**⚠️ IMPORTANT**: 11 functions are deprecated and will be removed in v2.0.0. See `tensor/CONSOLIDATION_PLAN.md` for migration guidance.

## Table of Contents

1. [BLAS Level 1: Vector Operations](#blas-level-1-vector-operations)
2. [BLAS Level 2: Matrix-Vector Operations](#blas-level-2-matrix-vector-operations)
3. [BLAS Level 3: Matrix-Matrix Operations](#blas-level-3-matrix-matrix-operations)
4. [Batched BLAS Operations](#batched-blas-operations)
5. [Element-wise Operations](#element-wise-operations)
6. [Reduction Operations](#reduction-operations)
7. [Tensor Operations](#tensor-operations)
8. [Convolution Operations](#convolution-operations)
9. [Activation Functions](#activation-functions)
10. [Linear Algebra Operations](#linear-algebra-operations)
11. [Utility Operations](#utility-operations)
12. [Duplicates Summary](#duplicates-summary)

## BLAS Level 1: Vector Operations

Located in `level1.go`. All operations follow BLAS standard.

| Operation | Function | Description |
|-----------|----------|-------------|
| AXPY | `Axpy(y, x, strideY, strideX, n, alpha)` | `y = alpha*x + y` |
| DOT | `Dot(x, y, strideX, strideY, n)` | `dot = x^T * y` |
| NRM2 | `Nrm2(x, stride, n)` | `norm = ||x||_2` (Euclidean norm) |
| ASUM | `Asum(x, stride, n)` | `sum = ||x||_1` (L1 norm) |
| SCAL | `Scal(x, stride, n, alpha)` | `x = alpha*x` (in-place) |
| COPY | `Copy(y, x, strideY, strideX, n)` | `y = x` |
| SWAP | `Swap(x, y, strideX, strideY, n)` | `x ↔ y` |
| IAMAX | `Iamax(x, stride, n)` | Index of element with maximum absolute value |

## BLAS Level 2: Matrix-Vector Operations

Located in `level2.go`. All operations follow BLAS standard.

| Operation | Function | Description |
|-----------|----------|-------------|
| GEMV_N | `Gemv_N(y, a, x, ldA, M, N, alpha, beta)` | `y = alpha*A*x + beta*y` (no transpose) |
| GEMV_T | `Gemv_T(y, a, x, ldA, M, N, alpha, beta)` | `y = alpha*A^T*x + beta*y` (transpose) |
| GER | `Ger(a, x, y, ldA, M, N, alpha)` | `A = alpha*x*y^T + A` (rank-1 update) |
| SYMV | `Symv(y, a, x, ldA, N, alpha, beta, uplo)` | `y = alpha*A*x + beta*y` (symmetric) |
| TRMV | `Trmv(y, a, x, ldA, N, uplo, trans, diag)` | `y = A*x` (triangular) |

## BLAS Level 3: Matrix-Matrix Operations

Located in `level3.go`. All operations follow BLAS standard.

| Operation | Function | Description |
|-----------|----------|-------------|
| GEMM_NN | `Gemm_NN(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)` | `C = alpha*A*B + beta*C` (neither transposed) |
| GEMM_NT | `Gemm_NT(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)` | `C = alpha*A*B^T + beta*C` (B transposed) |
| GEMM_TN | `Gemm_TN(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)` | `C = alpha*A^T*B + beta*C` (A transposed) |
| GEMM_TT | `Gemm_TT(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta)` | `C = alpha*A^T*B^T + beta*C` (both transposed) |
| SYRK | `Syrk(c, a, ldC, ldA, N, K, alpha, beta, uplo)` | `C = alpha*A*A^T + beta*C` (symmetric rank-k) |
| TRMM | `Trmm(c, a, b, ldC, ldA, ldB, M, N, alpha, beta, side, uplo, trans, diag)` | `C = alpha*A*B + beta*C` (triangular) |

## Batched BLAS Operations

Located in `batched.go`. Extensions of BLAS operations for batched processing.

| Operation | Function | Description |
|-----------|----------|-------------|
| Batched GEMM | `GemmBatched(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta, batchCount, stridea, strideb, stridec)` | Batched matrix multiplication |
| Strided GEMM | `GemmStrided(c, a, b, ldC, ldA, ldB, M, N, K, alpha, beta, batchCount, stridea, strideb, stridec)` | Strided batched matrix multiplication |
| Batched GEMV | `GemvBatched(y, a, x, ldA, M, N, alpha, beta, batchCount, strideA, strideX, strideY)` | Batched matrix-vector multiplication |

## Element-wise Operations


### Tensor Element-wise Operations (tensor_elementwise.go)

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Element-wise Add | `ElemAdd(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = a[i] + b[i]` | ✅ **RECOMMENDED** |
| Element-wise Subtract | `ElemSub(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = a[i] - b[i]` | ✅ **RECOMMENDED** |
| Element-wise Multiply | `ElemMul(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = a[i] * b[i]` | ✅ **RECOMMENDED** |
| Element-wise Divide | `ElemDiv(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = a[i] / b[i]` | ✅ **RECOMMENDED** |
| Element-wise Scale | `ElemScale(dst, scalar, shape, stridesDst)` | `dst[i] *= scalar` | |
| Element-wise Copy | `ElemCopy(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = src[i]` | |

### Vector Element-wise Operations (vector.go)

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Hadamard Product Add | `HadamardProductAdd(dst, a, b, num, strideA, strideB)` | `dst[i] += a[i] * b[i]` | |

## Reduction Operations

Located in `tensor_reduction.go`.

| Operation | Function | Description |
|-----------|----------|-------------|
| Reduce Sum | `ReduceSum(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)` | Sum reduction along axes |
| Reduce Mean | `ReduceMean(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)` | Mean reduction along axes |
| Reduce Max | `ReduceMax(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)` | Max reduction along axes |
| Reduce Min | `ReduceMin(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)` | Min reduction along axes |
| Argmax | `Argmax(dst, dstShape, dstStrides, src, srcShape, srcStrides, axis)` | Argmax along specified axis |

## Tensor Operations

Located in `tensor.go`.

| Operation | Function | Description |
|-----------|----------|-------------|
| Im2Col | `Im2Col(col, im, batchSize, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW)` | Image to column conversion for GEMM-based convolution |
| Col2Im | `Col2Im(im, col, batchSize, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW)` | Column to image conversion |
| Conv2D | `Conv2D(output, input, weights, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, bias)` | 2D convolution using Im2Col + GEMM |
| Conv2DKernelGrad | `Conv2DKernelGrad(kernelGrad, input, outputGrad, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW)` | **DEPRECATED**: 2D convolution kernel gradients - compose from primitives in layer implementations |
| Conv1DKernelGrad | `Conv1DKernelGrad(kernelGrad, input, outputGrad, batchSize, inChannels, outChannels, inLength, outLength, kernelLen, stride, padding)` | **DEPRECATED**: 1D convolution kernel gradients - compose from primitives in layer implementations |
| Conv2DTransposed | `Conv2DTransposed(output, input, weights, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, bias)` | Transposed 2D convolution (deconvolution) |
| MaxPool2D | `MaxPool2D(dst, src, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)` | 2D max pooling |
| AvgPool2D | `AvgPool2D(dst, src, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)` | 2D average pooling |
| GlobalAvgPool2D | `GlobalAvgPool2D(dst, src, batchSize, channels, height, width)` | Global average pooling |
| AdaptiveAvgPool2D | `AdaptiveAvgPool2D(dst, src, batchSize, channels, height, width, outHeight, outWidth)` | Adaptive average pooling to fixed size |
| DepthwiseConv2D | `DepthwiseConv2D(dst, src, kernel, bias, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)` | Depthwise 2D convolution |
| GroupConv2D | `GroupConv2D(dst, src, kernel, bias, batchSize, inChannels, outChannels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, groups)` | Grouped 2D convolution |
| DilatedConv2D | `DilatedConv2D(dst, src, kernel, bias, batchSize, inChannels, outChannels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW)` | Dilated 2D convolution |
| Conv3D | `Conv3D(dst, src, kernel, bias, batchSize, inChannels, outChannels, depth, height, width, kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW)` | 3D convolution |

## Convolution Operations

Located in `conv.go`.

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| 1D Convolution | `Convolve1DAdd(dst, vec, kernel, N, M, stride, transposed)` | 1D convolution with add | |

## Activation Functions

Located in `activations.go`.

| Operation | Function | Description |
|-----------|----------|-------------|
| ReLU | `ReLU(dst, src, size)` | Rectified Linear Unit: `max(0, x)` |
| ReLU Gradient | `ReLUGrad(dst, gradOutput, input, size)` | **DEPRECATED**: ReLU gradient - compose from conditional primitives in layer implementations |
| Sigmoid | `Sigmoid(dst, src, size)` | Sigmoid activation: `1/(1+exp(-x))` |
| Sigmoid Gradient | `SigmoidGrad(dst, gradOutput, output, size)` | **DEPRECATED**: Sigmoid gradient - compose from primitives in layer implementations |
| Tanh | `Tanh(dst, src, size)` | Hyperbolic tangent |
| Tanh Gradient | `TanhGrad(dst, gradOutput, output, size)` | **DEPRECATED**: Tanh gradient - compose from primitives in layer implementations |
| Softmax 1D | `Softmax1D(dst, size)` | 1D softmax |
| Softmax 2D Rows | `Softmax2DRows(dst, rows, cols)` | Softmax along rows |
| Softmax 2D Columns | `Softmax2DCols(dst, rows, cols)` | Softmax along columns |
| Softmax 1D Gradient | `Softmax1DGrad(dst, gradOutput, output, size)` | **DEPRECATED**: 1D softmax gradient - compose from primitives in layer implementations |
| Softmax 2D Rows Gradient | `Softmax2DRowsGrad(dst, gradOutput, output, rows, cols)` | **DEPRECATED**: Softmax gradient along rows - compose from primitives in layer implementations |
| Softmax 2D Columns Gradient | `Softmax2DColsGrad(dst, gradOutput, output, rows, cols)` | **DEPRECATED**: Softmax gradient along columns - compose from primitives in layer implementations |

## Linear Algebra Operations

Located in `la.go`. LAPACK-style operations.

| Operation | Function | Description |
|-----------|----------|-------------|
| Givens Rotation | `G1(a, b)` | Generate Givens rotation parameters |
| Apply Givens Rotation | `G2(cs, sn, x, y)` | Apply Givens rotation to vector elements |
| Householder H1 | `H1(a, col0, lpivot, l1, ldA, rangeVal)` | Construct Householder transformation |
| Householder H2 | `H2(a, zz, col0, lpivot, l1, up, ldA, rangeVal)` | Apply Householder to vector |
| Householder H3 | `H3(a, col0, lpivot, l1, up, col1, ldA, rangeVal)` | Apply Householder to matrix column |
| LU Factorization (in-place) | `Getrf_IP(a, ipiv, ldA, M, N)` | LU decomposition with partial pivoting |
| LU Factorization | `Getrf(a, l, u, ipiv, ldA, ldL, ldU, M, N)` | LU decomposition with separate outputs |
| Matrix Inverse | `Getri(aInv, a, ldA, ldInv, N, ipiv)` | Matrix inversion using LU |
| QR Factorization | `Geqrf(a, tau, ldA, M, N)` | QR decomposition using Householder |
| Generate Q from QR | `Orgqr(q, a, tau, ldA, ldQ, M, N, K)` | Generate Q matrix from QR decomposition |
| Pseudo-Inverse | `Gepseu(aPinv, a, ldA, ldApinv, M, N)` | Moore-Penrose pseudo-inverse |
| SVD | `Gesvd(u, s, vt, a, ldA, ldU, ldVt, M, N)` | Singular value decomposition |
| NNLS | `Gnnls(x, a, b, ldA, M, N)` | Non-negative least squares |

## Utility Operations

### Array Utilities (array.go)

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Sum | `Sum(a, num, stride)` | Sum of array elements | |
| Sum of Squares | `SqrSum(a, num, stride)` | Sum of squares | |
| Statistics | `StatsArr(min, max, mean, stddev, a, num, stride)` | Min, max, mean, std dev in one pass | |
| Percentile | `PercentileArr(p, sumAboveP, a, num, stride)` | Percentile value and sum above percentile | |
| Diff In-Place | `DiffArrInPlace(dst, c, num)` | `dst -= c` | |

### Vector Utilities (vector.go)

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| 2D Dot Product | `DotProduct2D(a, b, N, M, K, L)` | Specialized 2D dot product | |
| Vector Normalization | `NormalizeVec(dst, num, stride)` | In-place vector normalization | |
| Sum In-Place | `SumArrInPlace(dst, c, num)` | `dst += c` | |

### Tensor Utilities (tensor_broadcast.go, tensor_helpers.go)

| Operation | Function | Description |
|-----------|----------|-------------|
| Broadcast Strides | `BroadcastStrides(shape, strides, target)` | Compute strides for broadcasting |
| Expand To | `ExpandTo(dst, src, dstShape, srcShape, dstStrides, srcStrides)` | Broadcast tensor to target shape |
| Compute Strides | `ComputeStrides(shape)` | Compute strides from shape |
| Size From Shape | `SizeFromShape(shape)` | Compute total size from shape |
| Ensure Strides | `EnsureStrides(strides, shape)` | Ensure strides are valid |
| Is Contiguous | `IsContiguous(strides, shape)` | Check if layout is contiguous |
| Index Linear | `IndexLinear(indices, strides)` | Convert multi-dimensional indices to linear |
| Validate Axes | `ValidateAxes(shape, axes)` | Validate reduction axes |

## Duplicates Summary

### ✅ Consolidation Complete

All deprecated functions have been removed as of this version. The fp32 package now contains only the recommended implementations:

- **Element-wise operations**: Use `ElemAdd`, `ElemSub`, `ElemMul`, `ElemDiv` from `tensor_elementwise.go`
- **BLAS operations**: Use Level 1-3 operations (`Axpy`, `Dot`, `Scal`, `Gemm`, etc.)
- **Convolution**: Use `Conv2D` from `tensor.go` for modern GEMM-based implementation

### 💡 Best Practices

1. **Use BLAS Level 1-3 operations** as the foundation for all mathematical computations
2. **Use tensor_elementwise.go functions** for element-wise operations with stride/shape support
3. **Tensor package uses fp32 primitives** - fp32 does not depend on tensor (maintains proper dependency hierarchy)
4. **All operations are zero-allocation** in hot paths where possible

### 📊 Operation Counts by Category

- **BLAS Operations**: 18 (Level 1: 8, Level 2: 5, Level 3: 5, Batched: 3) ✓ **CORE - KEEP**
- **Tensor Operations**: 16 (Element-wise: 6, Reduction: 5, Convolution/Pooling: 10) ✓ **PRIMARY API**
- **Activation Functions**: 13 (Forward: 6, Gradient: 7) ⚠️ **GRADIENT FUNCTIONS DEPRECATED** - compose from primitives in layer implementations
- **Linear Algebra**: 13 (LAPACK-style operations) ✓ **CORE - KEEP**
- **Utilities**: 7 (Array: 5, Vector: 2, Tensor: 4) ✓ **CLEANED**

**Total Operations**: 67 functions across 13 categories
**Consolidation Result**: Removed 11 deprecated functions, eliminated duplicates
