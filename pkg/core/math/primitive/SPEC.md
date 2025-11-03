# math/primitive - BLAS/LAPACK Implementation Specification

## Overview

Pure Go implementation of Basic Linear Algebra Subprograms (BLAS) levels 1, 2, and 3, optimized for embedded systems with zero allocations, stride-based access, and no internal copying.

## Design Principles

1. **Row-Major Storage**: All matrices stored in row-major order (Go nested arrays layout)
2. **Zero Allocations**: No internal memory allocations in hot paths
3. **Stride-Based**: All operations accept leading dimension (ld) parameters
4. **No Boolean Flags**: Separate functions for transpose variants
5. **Explicit Dimensions**: Always specify dimensions explicitly
6. **In-Place OK**: Many operations support in-place computation

## Matrix Storage Layout

All matrices use **row-major** storage (matching Go nested arrays layout):

```go
// Matrix A: M rows x N columns
// Element A[i][j] stored at index: i*ldA + j
// where ldA >= N (leading dimension, number of columns)

// Example: 3x2 matrix A
A = [a00 a01
     a10 a11  
     a20 a21]

// Row-major storage (ldA = 2):
storage = [a00, a01, a10, a11, a20, a21]
//          row0    row1    row2

// Access formula:
A[i][j] = storage[i*ldA + j]
```

**Why Row-Major?**
- Matches Go nested arrays layout: `[][]float32` is row-major
- Consistent with Go's memory layout conventions
- Tensors are batch-major then row-major (Go nested arrays)

## BLAS Level 1: Vector Operations

### File: `level1.go`

Vector operations on one or two vectors.

| BLAS | Function | Description | Status |
|------|----------|-------------|--------|
| AXPY | Axpy | y = alpha*x + y | ✅ |
| DOT | Dot | dot = x^T * y | ✅ |
| NRM2 | Nrm2 | norm = \|\|x\|\|_2 | ✅ |
| ASUM | Asum | sum = \|\|x\|\|_1 | ✅ |
| SCAL | Scal | x = alpha*x | ✅ |
| COPY | Copy | y = x | ✅ |
| SWAP | Swap | x ↔ y | ✅ |
| IAMAX | Iamax | index = argmax\|x_i\| | ✅ |

### Function Signatures

```go
// AXPY: y = alpha*x + y
func Axpy(y, x []float32, strideY, strideX, n int, alpha float32)

// DOT: dot = x^T * y
func Dot(x, y []float32, strideX, strideY, n int) float32

// NRM2: norm = ||x||_2 (Euclidean norm)
func Nrm2(x []float32, stride, n int) float32

// ASUM: sum = ||x||_1 (L1 norm)
func Asum(x []float32, stride, n int) float32

// SCAL: x = alpha*x
func Scal(x []float32, stride, n int, alpha float32)

// COPY: y = x
func Copy(y, x []float32, strideY, strideX, n int)

// SWAP: x ↔ y (swap x and y)
func Swap(x, y []float32, strideX, strideY, n int)

// IAMAX: return index of element with max absolute value
func Iamax(x []float32, stride, n int) int
```

### Parameters
- `strideX, strideY`: Access stride (usually 1 for contiguous)
- `n`: Vector length
- `alpha`: Scalar multiplier

## BLAS Level 2: Matrix-Vector Operations

### File: `level2.go`

Matrix-vector operations.

| BLAS | Function | Description | Status |
|------|----------|-------------|--------|
| GEMV | Gemv | General matrix-vector multiply | ✅ |
| GER | Ger | Rank-1 update | ✅ |
| SYMV | Symv | Symmetric matrix-vector | ✅ |
| TRMV | Trmv | Triangular matrix-vector | ✅ |

### GEMV: General Matrix-Vector Multiply

**Operations:**
```go
// GEMV_N: y = alpha*A*x + beta*y  (no transpose)
func Gemv_N(y []float32, a, x []float32, ldA, M, N int, alpha, beta float32)

// GEMV_T: y = alpha*A^T*x + beta*y  (transpose)
func Gemv_T(y []float32, a, x []float32, ldA, M, N int, alpha, beta float32)
```

**Dimensions:**
- A: M × N matrix (row-major, ldA ≥ N)
- x: N × 1 vector
- y: M × 1 vector
- Result: y = alpha*A*x + beta*y

**GEMV_T (Transpose):**
- A^T: N × M matrix
- x: M × 1 vector
- y: N × 1 vector
- Result: y = alpha*A^T*x + beta*y

### GER: General Rank-1 Update

```go
// GER: A = alpha*x*y^T + A
func Ger(a []float32, x, y []float32, ldA, M, N int, alpha float32)
```

**Dimensions:**
- A: M × N matrix (updated in-place)
- x: M × 1 vector
- y: N × 1 vector
- alpha: scalar

**Operation:** A += alpha * x * y^T

## BLAS Level 3: Matrix-Matrix Operations

### File: `level3.go`

Matrix-matrix operations.

| BLAS | Function | Description | Status |
|------|----------|-------------|--------|
| GEMM | Gemm | General matrix-matrix multiply | ✅ |
| SYRK | Syrk | Symmetric rank-k update | ✅ |
| TRMM | Trmm | Triangular matrix-matrix | ✅ |

### GEMM: General Matrix-Matrix Multiply

**All Variants:**
```go
// GEMM_NN: C = alpha*A*B + beta*C  (neither transposed)
func Gemm_NN(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32)

// GEMM_NT: C = alpha*A*B^T + beta*C  (B transposed)
func Gemm_NT(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32)

// GEMM_TN: C = alpha*A^T*B + beta*C  (A transposed)
func Gemm_TN(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32)

// GEMM_TT: C = alpha*A^T*B^T + beta*C  (both transposed)
func Gemm_TT(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32)
```

**Dimensions:**
- A: M × K (row-major, ldA ≥ K)
- B: K × N (row-major, ldB ≥ N)
- C: M × N (row-major, ldC ≥ N)
- ldA, ldB, ldC: leading dimensions (≥ number of columns)

**Operation:** C = alpha * op(A) * op(B) + beta * C
- op(X) = X or X^T depending on variant
- alpha, beta: scalars

### SYRK: Symmetric Rank-K Update

```go
// SYRK: C = alpha*A*A^T + beta*C  (C symmetric)
func Syrk(c, a []float32, ldC, ldA, N, K int, alpha, beta float32)
```

**Dimensions:**
- A: N × K
- C: N × N (symmetric, only upper or lower stored)
- Operation: C += alpha * A * A^T

## LAPACK Operations

### File: `la.go`

Linear Algebra Package operations for matrix factorizations and decompositions. All operations use row-major storage. See [LA.md](LA.md) for detailed function specifications.

| LAPACK | Function | Description | Status |
|--------|----------|-------------|--------|
| GETRF | Getrf | LU decomposition with pivoting | 🔮 |
| GETRI | Getri | Matrix inversion using LU | 🔮 |
| H1 | H1 | Construct Householder transformation | 🔮 |
| H2 | H2 | Apply Householder to vector | 🔮 |
| H3 | H3 | Apply Householder to matrix column | 🔮 |
| GEQRF | Geqrf | QR decomposition | 🔮 |
| ORGQR | Orgqr | Generate Q from QR | 🔮 |
| GESVD | Gesvd | Singular value decomposition | 🔮 |
| GEPSEU | Gepseu | Moore-Penrose pseudo-inverse | 🔮 |
| GNNLS | Gnnls | Non-negative least squares | 🔮 |

### LU Decomposition

```go
// GETRF: A = P * L * U (LU decomposition with partial pivoting)
func Getrf(a, l, u []float32, ipiv []int, ldA, ldL, ldU, M, N int) error

// GETRF_IP: In-place LU decomposition (stores L and U in A)
func Getrf_IP(a []float32, ipiv []int, ldA, M, N int) error
```

### Matrix Inversion

```go
// GETRI: A^(-1) = U^(-1) * L^(-1) * P^T (uses LU from GETRF)
func Getri(aInv, a []float32, ldA, N int, ipiv []int) error
```

### Householder Transformations

Primitive Householder transformation functions used by QR decomposition and NNLS.

```go
// H1: Construct Householder transformation
// Returns transformation parameter 'up'
func H1(a []float32, col0, lpivot, l1 int, ldA int, rangeVal float32) (up float32, err error)

// H2: Apply Householder transformation to vector
// Applies transformation to vector zz in-place
func H2(a, zz []float32, col0, lpivot, l1 int, up float32, ldA int, rangeVal float32) error

// H3: Apply Householder transformation to matrix column
// Applies transformation to column col1 of matrix a in-place
func H3(a []float32, col0, lpivot, l1 int, up float32, col1 int, ldA int, rangeVal float32) error
```

### QR Decomposition

```go
// GEQRF: A = Q * R (QR decomposition using Householder reflections)
func Geqrf(a []float32, tau []float32, ldA, M, N int) error

// ORGQR: Generate Q from QR decomposition
func Orgqr(q, a []float32, tau []float32, ldA, ldQ, M, N, K int) error
```

### SVD Decomposition

```go
// GESVD: A = U * Σ * V^T (singular value decomposition)
func Gesvd(u, s, vt, a []float32, ldA, ldU, ldVt, M, N int) error
```

### Pseudo-Inverse

```go
// GEPSEU: A^+ = V * Σ^+ * U^T (Moore-Penrose pseudo-inverse)
func Gepseu(aPinv, a []float32, ldA, ldApinv, M, N int) error
```

### Non-Negative Least Squares

```go
// GNNLS: Solve min ||AX - B|| subject to X >= 0
func Gnnls(x, a, b []float32, ldA, M, N int) (rNorm float32, err error)
```

For detailed specifications, see [LA.md](LA.md).

## Stride Parameter Rules

### Vector Stride
- Usually `stride = 1` for contiguous vectors
- `stride > 1` for strided access
- Negative strides not supported

### Matrix Leading Dimension (ld)
- **Row-major**: `ldA ≥ N` (number of columns)
- **Distance between rows**: ldA elements
- Must be ≥ actual dimension to avoid bounds errors

### Example
```go
// 5x3 matrix (M=5, N=3) with ldA=4 (padded)
// Storage: [m00, m01, m02, _, m10, m11, m12, _, m20, ...]
//          row0 (4 elems)    row1 (4 elems)    row2
// Element m[i][j] = storage[i*ldA + j] = storage[i*4 + j]
```

## Implementation Status

- ✅ **Level 1**: Complete (all core vector operations)
- ✅ **Level 2**: Complete (GEMV_N, GEMV_T, GER, SYMV, TRMV)
- ✅ **Level 3**: Complete (GEMM_NN, GEMM_NT, GEMM_TN, GEMM_TT, SYRK, TRMM)
- ✅ **Batched**: Complete (GemmBatched, GemmStrided, GemvBatched)
- ✅ **Tensor**: Complete (Conv2D, Conv2DTransposed, Im2Col, Col2Im)
- 🔮 **LAPACK**: Planned (GETRF/GETRI, GEQRF/ORGQR, GESVD, GEPSEU, GNNLS)
- 🔮 **Future**: Symmetric, triangular, banded matrices

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Level 1 | Cache-friendly | Optimize stride access |
| Level 2 | Cache-blocking | Consider loop tiling |
| Level 3 | Cache-blocking | Must tile for performance |

## Testing Requirements

1. **Unit Tests**: Each function tested independently
2. **Equivalence Tests**: Verify same results as reference
3. **Stride Tests**: Test non-unit strides
4. **Edge Cases**: Empty vectors, zero dimensions
5. **Performance Tests**: Benchmark against baseline

## Migration from Current Implementation

### Current State
- Vector operations in `vector.go`, `array.go`
- Uses row-major storage
- Boolean transpose flags
- Some internal copying

### Target State
- Level 1 in `level1.go`
- Level 2 in `level2.go`
- Level 3 in `level3.go`
- Row-major storage (consistent with Go nested arrays)
- Separate transpose functions
- Zero allocations

### Breaking Changes
1. Function names: add _N, _T suffixes
2. Parameters: add ld (leading dimension)
3. Remove boolean flags

## BLAS Batched Operations

### File: `batched.go`

Batched operations process multiple matrices simultaneously, critical for:
- Neural network inference (batch processing)
- Convolutions (multiple kernels/filters)
- Tensor operations (parallel matrix operations)

| BLAS | Function | Description | Status |
|------|----------|-------------|--------|
| Batched GEMM | GemmBatched | General batched matrix multiply | ✅ |
| Batched GEMV | GemvBatched | General batched matrix-vector | ✅ |
| Strided GEMM | GemmStrided | Strided batch GEMM | ✅ |

### GemmBatched: Batched General Matrix Multiply

```go
// GemmBatched: C[k] = alpha*A[k]*B[k] + beta*C[k] for k=0..batchCount-1
func GemmBatched(
    c, a, b []float32,                // Arrays of batchCount matrices
    ldC, ldA, ldB int,                // Leading dimensions
    M, N, K int,                      // Matrix dimensions
    alpha, beta float32,              // Scalars
    batchCount int,                   // Number of matrices in batch
    stridea, strideb, stridec int,    // Strides between matrices
)
```

**Use Case:** Process multiple independent matrix multiplications
- Neural network: batch of samples through linear layer
- Convolution: multiple filters applied to same input

### GemmStrided: Strided Batch GEMM

```go
// GemmStrided: C[k] = alpha*A[k]*B[k] + beta*C[k] with strided access
func GemmStrided(
    c, a, b []float32,                // Flattened arrays
    ldC, ldA, ldB int,                // Leading dimensions
    M, N, K int,                      // Matrix dimensions
    alpha, beta float32,              // Scalars
    batchCount int,                   // Number of matrices
    stridea, strideb, stridec int,    // Strides
)
```

**Advantages:**
- All matrices in contiguous memory with fixed stride
- Better cache locality than separate arrays
- Easier to parallelize

### GemvBatched: Batched General Matrix-Vector Multiply

```go
// GemvBatched: y[k] = alpha*A[k]*x[k] + beta*y[k] for k=0..batchCount-1
func GemvBatched(
    y, a, x []float32,                // Arrays of vectors/matrices
    ldA int,                          // Leading dimension
    M, N int,                         // Matrix dimensions
    alpha, beta float32,              // Scalars
    batchCount int,                   // Number of operations
    strideA, strideX, strideY int,    // Strides
)
```

**Use Case:** Batch of feed-forward operations

## Tensor-Specific Operations

### File: `tensor.go`

Operations optimized for multi-dimensional tensor operations.

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Conv2D | Conv2D | 2D convolution | ✅ |
| Conv2DTransposed | Conv2DTransposed | Transposed convolution | ✅ |
| Im2Col | Im2Col | Image to column conversion | ✅ |
| Col2Im | Col2Im | Column to image conversion | ✅ |
| Pooling | MaxPool, AvgPool | Pooling operations | 🔮 |

### Conv2D: 2D Convolution

```go
// Conv2D: Perform 2D convolution with batched input
func Conv2D(
    output, input, weights []float32,
    batchSize, inChannels, outChannels int,
    inHeight, inWidth, outHeight, outWidth int,
    kernelH, kernelW, strideH, strideW, padH, padW int,
)
```

**Dimensions:**
- input: [batch, inChannels, inHeight, inWidth]
- weights: [outChannels, inChannels, kernelH, kernelW]
- output: [batch, outChannels, outHeight, outWidth]

**Optimizations:**
- Use Im2Col for GEMM-based convolution
- Direct convolution for small kernels
- Winograd algorithm for 3x3 kernels

### Im2Col: Image to Column Conversion

```go
// Im2Col: Convert image patches to columns (for GEMM-based conv)
func Im2Col(
    col, im []float32,                // Output and input
    batchSize, channels int,          // Batch and channel dimensions
    height, width, kernelH, kernelW int,
    padH, padW, strideH, strideW int,
)
```

**Converts:** Image patches → columns for matrix multiplication
**Enables:** Using GEMM for convolution (often faster for large kernels)

## File Organization

```
pkg/core/math/primitive/
├── level1.go          # Vector operations (BLAS level 1)
├── level1_test.go
├── level2.go          # Matrix-vector operations (BLAS level 2)
├── level2_test.go
├── level3.go          # Matrix-matrix operations (BLAS level 3)
├── level3_test.go
├── batched.go         # Batched BLAS operations
├── batched_test.go
├── la.go              # LAPACK operations (factorizations, decompositions)
├── la_test.go
├── tensor.go          # Tensor operations (conv, pooling, etc.)
├── tensor_test.go
├── conv.go            # Convolution operations (legacy, to be merged)
├── BLAS.md            # BLAS function mapping
├── LA.md              # LAPACK function mapping
├── SPEC.md            # This file
```

## Batched Operation Details

### Memory Layout for Batched Operations

```go
// Strided batch layout: all matrices in one array
// Example: batchCount=3, each matrix is M x N

// Stride: distance from start of matrix k to start of matrix k+1
stride = M * ldA  // For row-major layout (M rows, ldA columns per row)

// Access element A[i][j] in matrix k:
// storage[k*stride + i*ldA + j]

// Separate arrays: each matrix in its own array
// [][]float32 where [batchIndex] is a matrix
// Each can have different leading dimensions
```

### Im2Col Optimization for Convolution

Im2Col converts convolutions to GEMM operations:

**Original Convolution:**
```
Input: [batch, channels, height, width] = [B, C, H, W]
Kernel: [outChannels, inChannels, kH, kW] = [K, C, 3, 3]
Output: [batch, outChannels, outH, outW]

Operation: For each output position (i,j):
  Sum over kernel positions (ki,kj):
    output[b,k,i,j] += input[b,c,i+ki,j+kj] * weight[k,c,ki,kj]
```

**Im2Col + GEMM:**
```
1. Im2Col: Convert input to [B*outH*outW, C*kH*kW]
2. Reshape weights to [K, C*kH*kW]
3. GEMM: output = weights * im2col^T  → [K, B*outH*outW]
4. Reshape: [B, K, outH, outW]
```

**Advantages:** Exploits optimized GEMM code

## Next Steps

1. **Phase 1**: Implement Level 1 operations with stride support ✅
2. **Phase 2**: Implement Level 2 operations (GEMV, GER) ✅
3. **Phase 3**: Implement Level 3 operations (GEMM) ✅
4. **Phase 4**: Implement Batched operations (GemmBatched, GemvBatched) ✅
5. **Phase 5**: Implement Tensor operations (Conv2D, Im2Col) ✅
6. **Phase 6**: Implement LAPACK operations (GETRF/GETRI, GEQRF/ORGQR, GESVD, GEPSEU, GNNLS) 🔮
7. **Phase 7**: Update mat/tensor packages to use new primitives
8. **Phase 8**: Performance optimization and tiling

