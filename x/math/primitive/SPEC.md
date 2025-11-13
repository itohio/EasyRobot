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

### Non-BLAS Utility Functions

These functions exist in `array.go` and `vector.go` for tensor operations and statistics:

#### From `array.go`:
- `SumArr`, `DiffArr`, `MulArr`, `DivArr` - Element-wise operations (for tensor ops)
- `Sum`, `SqrSum` - Utility reductions for statistics
- `StatsArr` - Computes min, max, mean, and standard deviation in one pass
- `PercentileArr` - Computes percentile value and sum of values above percentile
- `SumArrInPlace` - In-place scalar addition (utility)
- `MulArrInPlace` - **DEPRECATED**: Use `Scal` from level1.go instead, kept for backward compatibility

**Removed (replaced by BLAS operations):**
- `SumArrConst`, `DiffArrConst`, `MulArrConst`, `DivArrConst` → Use `Axpy`, `Scal` from level1.go
- `MinArr`, `MaxArr`, `MeanArr`, `MomentsArr` → Use `StatsArr`
- `WeightedMomentsArr` → Removed (not needed)

#### From `vector.go`:
- `HadamardProduct` - Element-wise product (for tensor ops)
- `HadamardProductAdd` - Element-wise product and add (for tensor ops)
- `DotProduct` - **DEPRECATED**: Use `Dot` from level1.go, kept for backward compatibility
- `DotProduct2D` - 2D matrix dot product (specialized, not BLAS)
- `NormalizeVec` - Vector normalization (uses `Nrm2` from level1.go)

**Removed (replaced by BLAS operations):**
- `OuterProduct`, `OuterProductConst`, `OuterProductAddConst` → Use `Ger` from level2.go

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

Linear Algebra Package operations for matrix factorizations and decompositions. All operations use row-major storage.

| LAPACK | Function | Description | Status |
|--------|----------|-------------|--------|
| GETRF | Getrf | LU decomposition with pivoting | ✅ |
| GETRI | Getri | Matrix inversion using LU | ✅ |
| H1 | H1 | Construct Householder transformation | ✅ |
| H2 | H2 | Apply Householder to vector | ✅ |
| H3 | H3 | Apply Householder to matrix column | ✅ |
| GEQRF | Geqrf | QR decomposition | ✅ |
| ORGQR | Orgqr | Generate Q from QR | ✅ |
| GESVD | Gesvd | Singular value decomposition | ✅ |
| GEPSEU | Gepseu | Moore-Penrose pseudo-inverse | ✅ |
| GNNLS | Gnnls | Non-negative least squares | ✅ |

### Design Principles for LAPACK Operations

1. **Row-Major Storage**: All matrices stored in row-major order (Go nested arrays layout)
2. **Zero Allocations**: No internal memory allocations in hot paths
3. **Stride-Based**: All operations accept leading dimension (ld) parameters
4. **LAPACK Naming**: Follow LAPACK naming conventions where applicable
5. **Explicit Dimensions**: Always specify dimensions explicitly
6. **In-Place Support**: Many operations support in-place computation

### Matrix Inversion

#### GETRI: General Matrix Inversion

Computes the inverse of a matrix using LU decomposition with partial pivoting.

```go
// GETRF: Compute LU decomposition with partial pivoting
// A = P * L * U
// Returns pivot indices in ipiv (length min(M,N))
func Getrf(a []float32, ldA, M, N int, ipiv []int) error

// GETRI: Compute inverse using LU decomposition
// A^(-1) = U^(-1) * L^(-1) * P^T
// Input: a contains LU decomposition from GETRF, ipiv contains pivots
// Output: aInv contains the inverse
func Getri(aInv, a []float32, ldA, N int, ipiv []int) error
```

**Dimensions:**
- A: N × N square matrix (row-major, ldA ≥ N)
- ipiv: pivot indices (length N)
- Result: A^(-1) in aInv

**Note:** GETRF modifies input matrix A in-place. GETRI uses the LU decomposition from GETRF.

### LU Decomposition

#### GETRF: LU Factorization with Partial Pivoting

Computes LU decomposition with partial pivoting: A = P * L * U.

```go
// GETRF: Compute LU decomposition with partial pivoting
// A = P * L * U
// On input: a contains M × N matrix (row-major, ldA ≥ N)
// On output: l contains M × min(M,N) lower triangular matrix (unit diagonal)
//           u contains min(M,N) × N upper triangular matrix
//           ipiv contains pivot indices (length min(M,N))
func Getrf(a, l, u []float32, ipiv []int, ldA, ldL, ldU, M, N int) error

// Alternative in-place version (stores LU in A):
// GETRF_IP: Compute LU decomposition with partial pivoting (in-place)
// On input: a contains M × N matrix
// On output: a contains L (below diagonal) and U (on/above diagonal)
//           ipiv contains pivot indices
func Getrf_IP(a []float32, ipiv []int, ldA, M, N int) error
```

**Dimensions:**
- A: M × N matrix (row-major, ldA ≥ N)
- L: M × min(M,N) lower triangular (unit diagonal, row-major, ldL ≥ min(M,N))
- U: min(M,N) × N upper triangular (row-major, ldU ≥ N)
- ipiv: pivot indices (length min(M,N))
- Result: A = P * L * U where P is permutation matrix from ipiv

**Note:** GETRF can work in-place (storing L and U in A) or with separate output matrices. The in-place version modifies A to contain L (below diagonal) and U (on/above diagonal).

### Householder Transformations

#### H1, H2, H3: Householder Transform Primitives

Low-level Householder transformation functions used by QR decomposition and NNLS. Reference: C. L. Lawson and R. J. Hanson, 'Solving Least Squares Problems'.

```go
// H1: Construct Householder transformation
// Computes transformation parameter 'up' for Householder reflector
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//        col0: column index of the pivot vector
//        lpivot: pivot row index
//        l1: starting row index for transformation
//        rangeVal: regularization parameter for numerical stability (typically 1e30)
// Output: up: transformation parameter (returned)
//         a: modified (pivot element contains transformation value)
func H1(a []float32, col0, lpivot, l1 int, ldA int, rangeVal float32) (up float32, err error)

// H2: Apply Householder transformation to vector
// Applies transformation I + u*u^T/b to vector zz
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//        zz: vector to transform (modified in-place)
//        col0: column index of the pivot vector in a
//        lpivot: pivot row index
//        l1: starting row index for transformation
//        up: transformation parameter from H1
//        rangeVal: regularization parameter
func H2(a, zz []float32, col0, lpivot, l1 int, up float32, ldA int, rangeVal float32) error

// H3: Apply Householder transformation to matrix column
// Applies transformation I + u*u^T/b to column col1 of matrix a
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//        col0: column index of the pivot vector (contains Householder vector)
//        lpivot: pivot row index
//        l1: starting row index for transformation
//        up: transformation parameter from H1
//        col1: column index to transform
//        rangeVal: regularization parameter
// Output: a: column col1 is transformed in-place
func H3(a []float32, col0, lpivot, l1 int, up float32, col1 int, ldA int, rangeVal float32) error
```

**Dimensions:**
- A: M × N matrix (row-major, ldA ≥ N)
- zz: vector of length M
- col0, col1: column indices (0 ≤ col0, col1 < N)
- lpivot: pivot row index (0 ≤ lpivot < l1 < M)
- l1: starting row index for transformation
- rangeVal: regularization parameter (typically 1e30 for float32)

**Operation:**
- H1: Constructs Householder reflector `H = I - 2*u*u^T/(u^T*u)` where `u` is stored in column `col0` starting from row `lpivot`
- H2: Applies transformation to vector: `zz = H * zz` where H uses reflector from H1
- H3: Applies transformation to matrix column: `a[:,col1] = H * a[:,col1]` where H uses reflector from H1

**Usage:** These functions are building blocks for QR decomposition (GEQRF) and NNLS (GNNLS). H1 constructs the transformation, H2 and H3 apply it to vectors and matrix columns respectively.

### QR Decomposition

#### GEQRF: QR Factorization

Computes QR decomposition using Householder reflections.

```go
// GEQRF: Compute QR decomposition A = Q * R
// On input: a contains M × N matrix
// On output: a contains R in upper triangular part and Householder vectors below diagonal
//           tau contains scalar factors for Householder vectors (length min(M,N))
func Geqrf(a []float32, tau []float32, ldA, M, N int) error

// ORGQR: Generate orthogonal matrix Q from QR decomposition
// Input: a and tau from GEQRF
// Output: q contains M × M orthogonal matrix Q
func Orgqr(q, a []float32, tau []float32, ldA, ldQ, M, N, K int) error
```

**Dimensions:**
- A: M × N matrix (row-major, ldA ≥ N)
- tau: scalar factors (length min(M,N))
- Q: M × M orthogonal matrix (row-major, ldQ ≥ M)
- R: N × N upper triangular (stored in upper part of a)

**Note:** GEQRF modifies input matrix A in-place. Lower triangular part contains Householder vectors.

### SVD Decomposition

#### GESVD: Singular Value Decomposition

Computes singular value decomposition using bidiagonalization and QR iteration.

```go
// GESVD: Compute SVD decomposition A = U * Σ * V^T
// On input: a contains M × N matrix
// On output: u contains M × M left singular vectors (row-major, ldU ≥ M)
//           s contains min(M,N) singular values (sorted descending)
//           vt contains N × N right singular vectors transposed (row-major, ldVt ≥ N)
func Gesvd(u, s, vt, a []float32, ldA, ldU, ldVt, M, N int) error
```

**Dimensions:**
- A: M × N matrix (row-major, ldA ≥ N)
- U: M × M left singular vectors (row-major, ldU ≥ M)
- s: singular values (length min(M,N), sorted descending)
- Vt: N × N right singular vectors transposed (row-major, ldVt ≥ N)
- Result: A = U * diag(s) * Vt

**Note:** Input matrix A may be modified during computation.

### Pseudo-Inverse

#### GEPSEU: Moore-Penrose Pseudo-Inverse

Computes the pseudo-inverse using SVD decomposition.

```go
// GEPSEU: Compute Moore-Penrose pseudo-inverse A^+
// A^+ = V * Σ^+ * U^T where Σ^+ is pseudo-inverse of diagonal matrix
// Input: a contains M × N matrix
// Output: aPinv contains N × M pseudo-inverse (row-major, ldApinv ≥ M)
func Gepseu(aPinv, a []float32, ldA, ldApinv, M, N int) error
```

**Dimensions:**
- A: M × N matrix (row-major, ldA ≥ N)
- A^+: N × M pseudo-inverse (row-major, ldApinv ≥ M)
- Result: A * A^+ = I_M (if M ≥ N) or A^+ * A = I_N (if N ≥ M)

**Algorithm:** Uses GESVD internally and computes A^+ = V * diag(s^+) * U^T where s^+[i] = 1/s[i] if |s[i]| > tolerance, else 0.

### Non-Negative Least Squares (NNLS)

#### GNNLS: Non-Negative Least Squares

Solves constrained least squares problem: min ||AX - B|| subject to X ≥ 0.

```go
// GNNLS: Solve non-negative least squares min ||AX - B|| subject to X >= 0
// Input: a contains M × N matrix (row-major, ldA ≥ N)
//        b contains M × 1 right-hand side vector
// Output: x contains N × 1 solution vector (non-negative)
// Returns: residual norm ||AX - B||
func Gnnls(x, a, b []float32, ldA, M, N int) (rNorm float32, err error)
```

**Dimensions:**
- A: M × N matrix (row-major, ldA ≥ N)
- B: M × 1 right-hand side vector
- X: N × 1 solution vector (all elements ≥ 0)
- Result: minimizes ||AX - B||_2 subject to X ≥ 0

**Algorithm:** Lawson-Hanson active set method with Householder QR decomposition.

**Note:** Input matrix A and vector B may be modified during computation.

### Row-Major Considerations

Since all operations use row-major storage (matching Go's `[][]float32` layout), the function signatures differ from standard LAPACK (which uses column-major):

1. **Leading Dimension**: `ldA ≥ N` (number of columns) for row-major, vs `ldA ≥ M` (number of rows) for column-major
2. **Transpose Operations**: May need special handling for operations that transpose internally
3. **Householder Vectors**: Stored below diagonal in row-major matrices (vs above diagonal in column-major)

## Activation Functions

Activation functions are implemented as neural network layers in the `math/nn/layers` package. These provide both forward and backward operations for automatic differentiation and are not part of the primitive linear algebra operations.

### Implemented Activation Functions

| Function | Description | Location | Status |
|----------|-------------|----------|--------|
| **ReLU** | Rectified Linear Unit | `math/nn/layers/activations.go` | ✅ |
| **Sigmoid** | Logistic function | `math/nn/layers/activations.go` | ✅ |
| **Tanh** | Hyperbolic tangent | `math/nn/layers/activations.go` | ✅ |
| **Softmax** | Normalized exponential | `math/nn/layers/activations.go` | ✅ |
| **Dropout** | Random dropout | `math/nn/layers/activations.go` | ✅ |

See `math/nn/layers/activations.go` and the tensor package SPEC.md for detailed documentation of activation function implementations.

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
- ✅ **LAPACK**: Complete (GETRF/GETRI, GEQRF/ORGQR, GESVD, GEPSEU, GNNLS)
- ✅ **Quantized**: Complete (INT8 quantized operations for neural network inference)
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

## Quantized Operations

### File: `quantized.go`

INT8 quantized operations for efficient neural network inference.

**Quantization Scheme:**
- Asymmetric quantization: `real_value = scale * (quantized_value - zero_point)`
- Uses uint8 storage for quantized values
- Intermediate calculations use int32 to avoid overflow

| Operation | Function | Description | Status |
|-----------|----------|-------------|--------|
| Copy | Copy_Q8 | Copy quantized vector | ✅ |
| GEMM | Gemm_NN_Q8 | Quantized matrix multiply with requantization | ✅ |
| GEMM Accum | Gemm_NN_Q8_Accum | Quantized GEMM with int32 accumulator | ✅ |
| Conv2D | Conv2D_Q8 | Quantized 2D convolution | ✅ |
| Im2Col | Im2Col_Q8 | Image to column for quantized conv | ✅ |
| Col2Im | Col2Im_Q8 | Column to image for quantized conv | ✅ |
| Batched GEMM | GemmBatched_Q8 | Batched quantized GEMM | ✅ |

### Gemm_NN_Q8: Quantized Matrix Multiplication

```go
// Gemm_NN_Q8: Quantized GEMM with zero-point corrections
func Gemm_NN_Q8(
    output, input, weight []uint8,
    ldOutput, ldInput, ldWeight, M, N, K int,
    inputScale, weightScale, outputScale float32,
    inputZero, weightZero, outputZero int32,
)
```

**Zero-Point Correction:**
```
C_int[i,j] = (sum(A_int[i,k] * B_int[k,j]) 
             - inputZero * sum(B_int[k,j])
             - weightZero * sum(A_int[i,k])
             + inputZero * weightZero * K) * scale
where scale = inputScale * weightScale / outputScale
```

### Conv2D_Q8: Quantized Convolution

Uses Im2Col + quantized GEMM approach with int32 accumulator for bias.

```go
// Conv2D_Q8: Quantized 2D convolution
func Conv2D_Q8(
    output, input, weights []uint8,
    batchSize, inChannels, outChannels int,
    inHeight, inWidth, outHeight, outWidth int,
    kernelH, kernelW, strideH, strideW, padH, padW int,
    bias []int32,
    inputScale, weightScale, outputScale float32,
    inputZero, weightZero, outputZero int32,
)
```

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
├── quantized.go       # Quantized INT8 operations for neural networks
├── quantized_test.go
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
6. **Phase 6**: Implement LAPACK operations (GETRF/GETRI, GEQRF/ORGQR, GESVD, GEPSEU, GNNLS) ✅
7. **Phase 7**: Implement Quantized operations (INT8 quantization for neural networks) ✅
8. **Phase 8**: Update mat/tensor packages to use new primitives
9. **Phase 9**: Performance optimization and tiling

