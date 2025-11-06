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

#### Binary Operations

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Element-wise Add | `ElemAdd(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = a[i] + b[i]` | ✅ **RECOMMENDED** |
| Element-wise Subtract | `ElemSub(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = a[i] - b[i]` | ✅ **RECOMMENDED** |
| Element-wise Multiply | `ElemMul(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = a[i] * b[i]` | ✅ **RECOMMENDED** |
| Element-wise Divide | `ElemDiv(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = a[i] / b[i]` | ✅ **RECOMMENDED** |

#### Unary Operations

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Element-wise Copy | `ElemCopy(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = src[i]` | ✅ **RECOMMENDED** |
| Element-wise Scale In-Place | `ElemScaleInPlace(dst, scalar, shape, stridesDst)` | `dst[i] *= scalar` (in-place) | ✅ **FOR IN-PLACE USE** |
| Element-wise Scale | `ElemScale(dst, src, scalar, shape, stridesDst, stridesSrc)` | `dst[i] = src[i] * scalar` (dst-based) | ✅ **RECOMMENDED** |
| Element-wise Square | `ElemSquare(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = src[i]^2` | ✅ **RECOMMENDED** |
| Element-wise Square Root | `ElemSqrt(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = sqrt(src[i])` | ✅ **RECOMMENDED** |
| Element-wise Exponential | `ElemExp(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = exp(src[i])` | ✅ **RECOMMENDED** |
| Element-wise Logarithm | `ElemLog(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = log(src[i])` | ✅ **RECOMMENDED** |
| Element-wise Power | `ElemPow(dst, src, power, shape, stridesDst, stridesSrc)` | `dst[i] = src[i]^power` | ✅ **RECOMMENDED** |
| Element-wise Absolute | `ElemAbs(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = abs(src[i])` | ✅ **RECOMMENDED** |
| Element-wise Sign | `ElemSign(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = sign(src[i])` (-1, 0, or 1) | ✅ **RECOMMENDED** |
| Element-wise Cosine | `ElemCos(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = cos(src[i])` | ✅ **RECOMMENDED** |
| Element-wise Sine | `ElemSin(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = sin(src[i])` | ✅ **RECOMMENDED** |
| Element-wise Tanh | `ElemTanh(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = tanh(src[i])` | ✅ **RECOMMENDED** |
| Element-wise Negation | `ElemNegative(dst, src, shape, stridesDst, stridesSrc)` | `dst[i] = -src[i]` | ✅ **RECOMMENDED** |

#### Scalar Operations

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Element-wise Fill | `ElemFill(dst, value, shape, stridesDst)` | `dst[i] = value` | ✅ **RECOMMENDED** |
| Add Scalar | `ElemAddScalar(dst, src, scalar, shape, stridesDst, stridesSrc)` | `dst[i] = src[i] + scalar` | ✅ **RECOMMENDED** |
| Subtract Scalar | `ElemSubScalar(dst, src, scalar, shape, stridesDst, stridesSrc)` | `dst[i] = src[i] - scalar` | ✅ **RECOMMENDED** |
| Divide Scalar | `ElemDivScalar(dst, src, scalar, shape, stridesDst, stridesSrc)` | `dst[i] = src[i] / scalar` | ✅ **RECOMMENDED** |

#### Comparison Operations

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Element-wise Greater Than | `ElemGreaterThan(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = 1.0 if a[i] > b[i], else 0.0` | ✅ **RECOMMENDED** |
| Element-wise Equal | `ElemEqual(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = 1.0 if a[i] == b[i], else 0.0` | ✅ **RECOMMENDED** |
| Element-wise Less | `ElemLess(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = 1.0 if a[i] < b[i], else 0.0` | ✅ **RECOMMENDED** |
| Element-wise Not Equal | `ElemNotEqual(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = 1.0 if a[i] != b[i], else 0.0` | ✅ **RECOMMENDED** |
| Element-wise Less Equal | `ElemLessEqual(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = 1.0 if a[i] <= b[i], else 0.0` | ✅ **RECOMMENDED** |
| Element-wise Greater Equal | `ElemGreaterEqual(dst, a, b, shape, stridesDst, stridesA, stridesB)` | `dst[i] = 1.0 if a[i] >= b[i], else 0.0` | ✅ **RECOMMENDED** |

#### Ternary Operations

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Element-wise Where | `ElemWhere(dst, condition, a, b, shape, stridesDst, stridesCond, stridesA, stridesB)` | `dst[i] = a[i] if condition[i] > 0, else b[i]` | ✅ **RECOMMENDED** |

#### Scaled Operations (Optimized Composite)

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Add Scaled Multiply | `ElemAddScaledMul(dst, other, scalar, shape, stridesDst, stridesOther)` | `dst[i] = (1 + scalar) * other[i]` | ✅ **RECOMMENDED** |
| Add Scaled Square Multiply | `ElemAddScaledSquareMul(dst, other, scalar, shape, stridesDst, stridesOther)` | `dst[i] = (1 + scalar * other[i]^2) * other[i]` | ✅ **RECOMMENDED** |

### Vector Element-wise Operations (vector.go)

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Hadamard Product | `HadamardProduct(dst, a, b, num, strideDst, strideA, strideB)` | `dst[i] = a[i] * b[i]` (dst-based) | ✅ **RECOMMENDED** |
| Hadamard Product Add | `HadamardProductAdd(dst, a, b, num, strideA, strideB)` | `dst[i] += a[i] * b[i]` (accumulation) | ✅ **FOR ACCUMULATION** |

## Reduction Operations

Located in `tensor_reduction.go`.

| Operation | Function | Description |
|-----------|----------|-------------|
| Reduce Sum | `ReduceSum(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)` | Sum reduction along axes |
| Reduce Mean | `ReduceMean(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)` | Mean reduction along axes |
| Reduce Max | `ReduceMax(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)` | Max reduction along axes |
| Reduce Min | `ReduceMin(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)` | Min reduction along axes |
| Argmax | `Argmax(dst, dstShape, dstStrides, src, srcShape, srcStrides, axis)` | Argmax along specified axis (returns float32 indices) |
| Argmin | `Argmin(dst, dstShape, dstStrides, src, srcShape, srcStrides, axis)` | Argmin along specified axis (returns int32 indices) |

## Tensor Operations

Located in `tensor.go`.

| Operation | Function | Description |
|-----------|----------|-------------|
| Im2Col | `Im2Col(col, im, batchSize, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW)` | Image to column conversion for GEMM-based convolution |
| Col2Im | `Col2Im(im, col, batchSize, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW)` | Column to image conversion |
| Conv2D | `Conv2D(output, input, weights, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, bias)` | 2D convolution using Im2Col + GEMM |
| Conv2DKernelGrad | `Conv2DKernelGrad(kernelGrad, input, outputGrad, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW)` | 2D convolution kernel gradients - computes gradient with respect to kernel weights |
| Conv1DKernelGrad | `Conv1DKernelGrad(kernelGrad, input, outputGrad, batchSize, inChannels, outChannels, inLength, outLength, kernelLen, stride, padding)` | 1D convolution kernel gradients - computes gradient with respect to kernel weights |
| Conv2DTransposed | `Conv2DTransposed(output, input, weights, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, bias)` | Transposed 2D convolution (deconvolution) |
| Conv2DTransposedWithOutputPadding | `Conv2DTransposedWithOutputPadding(output, input, weights, batchSize, inChannels, outChannels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, outputPadH, outputPadW, bias)` | Transposed 2D convolution with output padding (for GAN architectures) |
| SeparableConv2D | `SeparableConv2D(dst, src, depthwiseKernel, pointwiseKernel, bias, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)` | Separable 2D convolution (depthwise + pointwise, optimized) |
| Conv3DTransposed | `Conv3DTransposed(dst, src, kernel, bias, batchSize, inChannels, outChannels, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth, kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW)` | Transposed 3D convolution (deconvolution) |
| MaxPool1D | `MaxPool1D(dst, src, batchSize, channels, length, kernelLen, stride, padding)` | 1D max pooling |
| MaxPool2D | `MaxPool2D(dst, src, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)` | 2D max pooling |
| MaxPool3D | `MaxPool3D(dst, src, batchSize, channels, depth, height, width, kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW)` | 3D max pooling |
| MaxPool2DWithIndices | `MaxPool2DWithIndices(dst, src, indices, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)` | 2D max pooling with indices |
| MaxPool2DBackward | `MaxPool2DBackward(gradInput, gradOutput, indices, src, batchSize, channels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW)` | Max pooling backward pass |
| AvgPool1D | `AvgPool1D(dst, src, batchSize, channels, length, kernelLen, stride, padding)` | 1D average pooling |
| AvgPool2D | `AvgPool2D(dst, src, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)` | 2D average pooling |
| AvgPool3D | `AvgPool3D(dst, src, batchSize, channels, depth, height, width, kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW)` | 3D average pooling |
| AvgPool2DBackward | `AvgPool2DBackward(gradInput, gradOutput, batchSize, channels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW)` | Average pooling backward pass |
| GlobalMaxPool1D | `GlobalMaxPool1D(dst, src, batchSize, channels, length)` | Global max pooling 1D |
| GlobalMaxPool2D | `GlobalMaxPool2D(dst, src, batchSize, channels, height, width)` | Global max pooling 2D |
| GlobalMaxPool3D | `GlobalMaxPool3D(dst, src, batchSize, channels, depth, height, width)` | Global max pooling 3D |
| GlobalAvgPool2D | `GlobalAvgPool2D(dst, src, batchSize, channels, height, width)` | Global average pooling 2D |
| AdaptiveMaxPool1D | `AdaptiveMaxPool1D(dst, src, batchSize, channels, inLength, outLength)` | Adaptive max pooling 1D |
| AdaptiveMaxPool2D | `AdaptiveMaxPool2D(dst, src, batchSize, channels, inHeight, inWidth, outHeight, outWidth)` | Adaptive max pooling 2D |
| AdaptiveMaxPool3D | `AdaptiveMaxPool3D(dst, src, batchSize, channels, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth)` | Adaptive max pooling 3D |
| AdaptiveAvgPool2D | `AdaptiveAvgPool2D(dst, src, batchSize, channels, height, width, outHeight, outWidth)` | Adaptive average pooling 2D to fixed size |
| DepthwiseConv2D | `DepthwiseConv2D(dst, src, kernel, bias, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)` | Depthwise 2D convolution |
| GroupConv2D | `GroupConv2D(dst, src, kernel, bias, batchSize, inChannels, outChannels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, groups)` | Grouped 2D convolution |
| DilatedConv2D | `DilatedConv2D(dst, src, kernel, bias, batchSize, inChannels, outChannels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW)` | Dilated 2D convolution |
| Conv3D | `Conv3D(dst, src, kernel, bias, batchSize, inChannels, outChannels, depth, height, width, kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW)` | 3D convolution |

## Convolution Operations

Located in `conv.go`.

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| 1D Convolution | `Convolve1D(dst, vec, kernel, N, M, stride, transposed)` | 1D convolution: `dst = conv(vec, kernel)` (dst-based) | ✅ **RECOMMENDED** |
| 1D Convolution Add | `Convolve1DAdd(dst, vec, kernel, N, M, stride, transposed)` | 1D convolution with add: `dst += conv(...)` (accumulation) | ✅ **FOR ACCUMULATION** |

## Activation Functions

Located in `activations.go`.

| Operation | Function | Description |
|-----------|----------|-------------|
| ReLU | `ReLU(dst, src, size)` | Rectified Linear Unit: `max(0, x)` |
| ReLU Gradient | `ReLUGrad(dst, gradOutput, input, size)` | ReLU gradient: `gradOutput * (input > 0 ? 1 : 0)` |
| ReLU Gradient Stride | `ReLUGradStride(dst, gradOutput, input, shape, stridesDst, stridesGrad, stridesInput)` | ReLU gradient with stride support |
| Sigmoid | `Sigmoid(dst, src, size)` | Sigmoid activation: `1/(1+exp(-x))` |
| Sigmoid Gradient | `SigmoidGrad(dst, gradOutput, output, size)` | Sigmoid gradient: `gradOutput * output * (1 - output)` |
| Sigmoid Gradient Stride | `SigmoidGradStride(dst, gradOutput, output, shape, stridesDst, stridesGrad, stridesOutput)` | Sigmoid gradient with stride support |
| Tanh | `Tanh(dst, src, size)` | Hyperbolic tangent |
| Tanh Gradient | `TanhGrad(dst, gradOutput, output, size)` | Tanh gradient: `gradOutput * (1 - output^2)` |
| Tanh Gradient Stride | `TanhGradStride(dst, gradOutput, output, shape, stridesDst, stridesGrad, stridesOutput)` | Tanh gradient with stride support |
| Softmax 1D | `Softmax1D(dst, size)` | 1D softmax |
| Softmax 2D Rows | `Softmax2DRows(dst, rows, cols)` | Softmax along rows |
| Softmax 2D Columns | `Softmax2DCols(dst, rows, cols)` | Softmax along columns |
| Softmax 1D Gradient | `Softmax1DGrad(dst, gradOutput, output, size)` | 1D softmax gradient: `output * (gradOutput - sum(gradOutput * output))` |
| Softmax 2D Rows Gradient | `Softmax2DRowsGrad(dst, gradOutput, output, rows, cols)` | Softmax gradient along rows |
| Softmax 2D Columns Gradient | `Softmax2DColsGrad(dst, gradOutput, output, rows, cols)` | Softmax gradient along columns |

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
| Diff Scalar | `DiffArrScalar(dst, src, c, num, strideDst, strideSrc)` | `dst[i] = src[i] - c` (dst-based) | ✅ **RECOMMENDED** |
| Diff In-Place | `DiffArrInPlace(dst, c, num)` | `dst[i] -= c` (in-place accumulation) | ✅ **FOR ACCUMULATION** |

### Vector Utilities (vector.go)

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| 2D Dot Product | `DotProduct2D(a, b, N, M, K, L)` | Specialized 2D dot product | |
| Vector Normalization | `NormalizeVec(dst, src, num, strideDst, strideSrc)` | Vector normalization: `dst = src / ||src||` (dst-based) | ✅ **RECOMMENDED** |
| Vector Normalization In-Place | `NormalizeVecInPlace(dst, num, stride)` | In-place vector normalization: `dst = dst / ||dst||` | ✅ **FOR IN-PLACE USE** |
| Sum Scalar | `SumArrScalar(dst, src, c, num, strideDst, strideSrc)` | `dst[i] = src[i] + c` (dst-based) | ✅ **RECOMMENDED** |
| Sum In-Place | `SumArrInPlace(dst, c, num)` | `dst[i] += c` (in-place accumulation) | ✅ **FOR ACCUMULATION** |

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
- **Tensor Element-wise Operations**: 31 (Binary: 4, Unary: 13, Scalar: 5, Comparison: 6, Ternary: 1, Scaled: 2) ✓ **PRIMARY API**
- **Reduction Operations**: 6 (Sum, Mean, Max, Min, Argmax, Argmin) ✓ **PRIMARY API**
- **Tensor Operations**: 29 (Pooling: 16, Convolution: 13) ✓ **PRIMARY API**
- **Activation Functions**: 16 (Forward: 6, Gradient: 7, Gradient Stride: 3) ✓ **RECOMMENDED FOR EMBEDDED** - dedicated gradient functions provide better performance
- **Linear Algebra**: 13 (LAPACK-style operations) ✓ **CORE - KEEP**
- **Utilities**: 12 (Array: 7, Vector: 5, Tensor: 4) ✓ **CLEANED** (includes dst-based versions)

**Total Operations**: 125 functions across 13 categories
**Consolidation Result**: Removed 11 deprecated functions, eliminated duplicates, added 31 new element-wise operations, added 16 new pooling operations, added 3 stride-based gradient functions, added 3 convolution extension operations, renamed 3 operations for naming consistency, added 5 dst-based operation versions

### Operation Patterns

**Naming Convention**: All in-place operations are suffixed with `InPlace` (e.g., `ElemScaleInPlace`, `NormalizeVecInPlace`). Base operation names (without `InPlace`) are reserved for dst-based versions following `Operation(dst, src, ...)` pattern.

**In-Place Operations**: Some operations modify their input arrays in-place for memory efficiency:
- BLAS: `Axpy`, `Scal` (standard BLAS in-place pattern, kept for BLAS compatibility)
- Element-wise: `ElemScaleInPlace` (use `ElemScale` for dst-based version)
- Accumulation: `HadamardProductAdd`, `Convolve1DAdd`, `SumArrInPlace`, `DiffArrInPlace` (accumulation pattern, different from in-place)
- Utility: `NormalizeVecInPlace`

**Destination-Based Operations**: All tensor operations write to separate `dst` parameter:
- All `Elem*` operations use `dst = operation(src, ...)` pattern
- All pooling and convolution operations write to separate `dst`
- All new dst-based versions: `HadamardProduct`, `Convolve1D`, `NormalizeVec`, `SumArrScalar`, `DiffArrScalar`
- Activation functions support in-place (dst and src can be same slice) but default to separate dst

**Recommendation**: For tensor API consistency, prefer destination-based operations (base names without `InPlace` suffix). In-place operations (`*InPlace`) are useful for memory-constrained embedded systems. Accumulation operations (`*Add`) are useful for gradient updates.
