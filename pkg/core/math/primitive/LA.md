# LAPACK Function Implementation Map

This document maps LAPACK function names to their implementations in the `primitive` package. All operations use row-major storage (matching Go nested arrays layout).

## Design Principles

1. **Row-Major Storage**: All matrices stored in row-major order (Go nested arrays layout)
2. **Zero Allocations**: No internal memory allocations in hot paths
3. **Stride-Based**: All operations accept leading dimension (ld) parameters
4. **LAPACK Naming**: Follow LAPACK naming conventions where applicable
5. **Explicit Dimensions**: Always specify dimensions explicitly
6. **In-Place Support**: Many operations support in-place computation

## Matrix Inversion

### GETRI: General Matrix Inversion

Computes the inverse of a matrix using LU decomposition with partial pivoting.

| LAPACK Function | Our Function | Implementation | Status |
|----------------|--------------|----------------|--------|
| **GETRF** | `Getrf(a, ldA, ipiv, M, N)` | `la.go` | 🔮 |
| **GETRI** | `Getri(aInv, a, ldA, ipiv, N)` | `la.go` | 🔮 |

**Function Signatures:**
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

## LU Decomposition

### GETRF: LU Factorization with Partial Pivoting

Computes LU decomposition with partial pivoting: A = P * L * U.

| LAPACK Function | Our Function | Implementation | Status |
|----------------|--------------|----------------|--------|
| **GETRF** | `Getrf(a, l, u, ipiv, ldA, M, N)` | `la.go` | 🔮 |

**Function Signatures:**
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

## Householder Transformations

### H1, H2, H3: Householder Transform Primitives

Low-level Householder transformation functions used by QR decomposition and NNLS. Reference: C. L. Lawson and R. J. Hanson, 'Solving Least Squares Problems'.

| Function | Our Function | Implementation | Status |
|----------|--------------|----------------|--------|
| **H1** | `H1(a, up, col0, lpivot, l1, ldA, rangeVal)` | `la.go` | 🔮 |
| **H2** | `H2(a, zz, col0, lpivot, l1, up, ldA, rangeVal)` | `la.go` | 🔮 |
| **H3** | `H3(a, col0, lpivot, l1, up, col1, ldA, rangeVal)` | `la.go` | 🔮 |

**Function Signatures:**
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

## QR Decomposition

### GEQRF: QR Factorization

Computes QR decomposition using Householder reflections.

| LAPACK Function | Our Function | Implementation | Status |
|----------------|--------------|----------------|--------|
| **GEQRF** | `Geqrf(a, tau, ldA, M, N)` | `la.go` | 🔮 |
| **ORGQR** | `Orgqr(q, a, tau, ldA, ldQ, M, N, K)` | `la.go` | 🔮 |

**Function Signatures:**
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

## SVD Decomposition

### GESVD: Singular Value Decomposition

Computes singular value decomposition using bidiagonalization and QR iteration.

| LAPACK Function | Our Function | Implementation | Status |
|----------------|--------------|----------------|--------|
| **GESVD** | `Gesvd(u, s, vt, a, ldA, ldU, ldVt, M, N)` | `la.go` | 🔮 |

**Function Signatures:**
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

## Pseudo-Inverse

### GEPSEU: Moore-Penrose Pseudo-Inverse

Computes the pseudo-inverse using SVD decomposition.

| Operation | Our Function | Implementation | Status |
|-----------|--------------|----------------|--------|
| **Pseudo-Inverse** | `Gepseu(aPinv, a, ldA, ldApinv, M, N)` | `la.go` | 🔮 |

**Function Signatures:**
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

## Non-Negative Least Squares (NNLS)

### GNNLS: Non-Negative Least Squares

Solves constrained least squares problem: min ||AX - B|| subject to X ≥ 0.

| Operation | Our Function | Implementation | Status |
|-----------|--------------|----------------|--------|
| **NNLS** | `Gnnls(x, a, b, ldA, M, N)` | `la.go` | 🔮 |

**Function Signatures:**
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

## Matrix Factorization Primitives

### Helper Functions for Factorizations

These are internal functions used by the main LAPACK-style functions.

| Function | Description | Status |
|----------|-------------|--------|
| `Trsm_L` | Solve triangular system L * X = B (lower triangular) | 🔮 |
| `Trsm_U` | Solve triangular system U * X = B (upper triangular) | 🔮 |
| `H1` | Construct Householder transformation | 🔮 |
| `H2` | Apply Householder transformation to vector | 🔮 |
| `H3` | Apply Householder transformation to matrix column | 🔮 |
| `HouseholderQR` | Householder QR factorization (helper for GEQRF) | 🔮 |
| `BidiagonalSVD` | Bidiagonal SVD (helper for GESVD) | 🔮 |

## Status Legend

- ✅ **Implemented**: Function is complete and tested
- ⏳ **In Progress**: Function is being implemented
- 🔮 **Planned**: Function is planned but not yet started

## Row-Major Considerations

Since all operations use row-major storage (matching Go's `[][]float32` layout), the function signatures differ from standard LAPACK (which uses column-major):

1. **Leading Dimension**: `ldA ≥ N` (number of columns) for row-major, vs `ldA ≥ M` (number of rows) for column-major
2. **Transpose Operations**: May need special handling for operations that transpose internally
3. **Householder Vectors**: Stored below diagonal in row-major matrices (vs above diagonal in column-major)

## Migration Path

1. Implement H1, H2, H3 for Householder transformations (primitives for QR and NNLS)
2. Implement GETRF for LU decomposition (replace LU decomposition in `mat` package)
3. Implement GETRF/GETRI for matrix inversion (replace Inverse in `mat` package)
4. Implement GEQRF/ORGQR for QR decomposition (replace QR in `mat` package)
5. Implement GESVD for SVD decomposition (replace SVD in `mat` package)
6. Implement GEPSEU for pseudo-inverse (replace PseudoInverse in `mat` package)
7. Implement GNNLS for non-negative least squares (replace NNLS in `mat` package)

