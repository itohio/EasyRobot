# math/mat - Matrix Operations Specification

## Overview

The `mat` package provides matrix mathematics for arbitrary and fixed-size matrices, optimized for robotics applications with emphasis on embedded systems compatibility and performance. All matrices use **row-major** storage layout.

## Design Principles

1. **Row-Major Storage**: All matrices stored in row-major order (matching Go nested arrays layout)
2. **In-Place Operations**: Most operations modify the matrix in-place to minimize allocations
3. **Zero-Copy Flattening**: Matrix flattening uses copy-free base array recovery when contiguous
4. **Method Chaining**: Operations return the matrix to enable method chaining
5. **Zero Allocations**: Critical paths avoid memory allocations
6. **Float32 Precision**: Uses `float32` for embedded-friendly precision
7. **Type Safety**: Fixed-size types (Matrix2x2, Matrix3x3, Matrix4x4, etc.) for compile-time size guarantees
8. **Direct Computation**: Fixed-size matrix operations use direct math computations unless primitives offer better performance

## Matrix Storage Layout

### Row-Major Storage

All matrices use **row-major** storage (matching Go nested arrays layout):

```go
// Matrix A: M rows x N columns
// Element A[i][j] stored at index: i*cols + j
// where cols = N (number of columns)

// Example: 3x2 matrix A
A = [a00 a01
     a10 a11  
     a20 a21]

// Row-major storage (contiguous):
storage = [a00, a01, a10, a11, a20, a21]
//          row0    row1    row2

// Access formula:
A[i][j] = storage[i*cols + j]
```

**Why Row-Major?**
- Matches Go nested arrays layout: `[][]float32` is row-major
- Consistent with Go's memory layout conventions
- Compatible with BLAS/LAPACK operations (which support row-major via stride parameters)

## Types

### Matrix
Generic variable-size matrix type (slice of slices of float32).

```go
type Matrix [][]float32
```

**Characteristics:**
- Variable dimensions (runtime-determined)
- Slice-based (allows sub-slicing, views)
- Backed by `[][]float32`
- Can be non-contiguous (rows may not be adjacent in memory)
- Supports `Flat()` method for flattening (zero-copy when contiguous)
- Provides structural queries (`Rows()`, `Cols()`) and `Rank()` derived via row-reduction

### Fixed-Size Matrices

#### Matrix2x2
Fixed-size 2x2 matrix.

```go
type Matrix2x2 [2][2]float32
```

**Characteristics:**
- Fixed size at compile time (2x2 = 4 elements)
- Array-based (value semantics)
- Contiguous in memory
- Direct computation for all operations (no primitives needed for small size)

#### Matrix3x3
Fixed-size 3x3 matrix.

```go
type Matrix3x3 [3][3]float32
```

**Characteristics:**
- Fixed size at compile time (3x3 = 9 elements)
- Array-based (value semantics)
- Contiguous in memory
- Direct computation for most operations
- May use primitives for complex operations (matrix multiplication)

#### Matrix4x4
Fixed-size 4x4 matrix.

```go
type Matrix4x4 [4][4]float32
```

**Characteristics:**
- Fixed size at compile time (4x4 = 16 elements)
- Array-based (value semantics)
- Contiguous in memory
- Direct computation for most operations
- May use primitives for complex operations (matrix multiplication)

#### Matrix3x4 and Matrix4x3
Fixed-size rectangular matrices for homogeneous transformations.

```go
type Matrix3x4 [3][4]float32  // 3 rows, 4 columns
type Matrix4x3 [4][3]float32  // 4 rows, 3 columns
```

**Characteristics:**
- Fixed size at compile time
- Array-based (value semantics)
- Used for homogeneous transformations in robotics

## Matrix Flattening

### Flat() Method - Copy-Free When Possible

The `Flat()` method returns a flat `[]float32` representation of the matrix.

**For Matrix (variable-size):**
```go
func (m Matrix) Flat() []float32
```

**Behavior:**
1. **Fast Path (Zero-Copy)**: If matrix is contiguous (`IsContiguous() == true`), returns a zero-copy slice using `unsafe.Slice()` pointing to the base array
2. **Slow Path (Copy)**: If matrix is non-contiguous, creates a new array and copies all elements

**For Fixed-Size Matrices:**
```go
func (m *Matrix2x2) Flat(v vec.Vector) vec.Vector
func (m *Matrix3x3) Flat(v vec.Vector) vec.Vector
func (m *Matrix4x4) Flat(v vec.Vector) vec.Vector
```

**Behavior:**
- Always copies data into provided vector (no zero-copy, but no allocation)
- Fixed-size matrices are always contiguous, but use explicit copy for safety

### IsContiguous() Method

Checks if matrix rows are adjacent in memory (contiguous).

```go
func (m Matrix) IsContiguous() bool
```

**Usage:**
- Used internally by `Flat()` to determine if zero-copy flattening is possible
- Checks if end of row `i` is adjacent to start of row `i+1` using pointer arithmetic

## Operations

### Construction and Access

| Method | Type | Description |
|--------|------|-------------|
| `New(rows, cols int, backing ...float32) Matrix` | Matrix | Creates new matrix with optional backing array |
| `New2x2(arr ...float32) Matrix2x2` | Matrix2x2 | Creates 2x2 matrix from array |
| `New3x3(arr ...float32) Matrix3x3` | Matrix3x3 | Creates 3x3 matrix from array |
| `New4x4(arr ...float32) Matrix4x4` | Matrix4x4 | Creates 4x4 matrix from array |
| `Flat() []float32` | Matrix | Returns flat representation (zero-copy if contiguous) |
| `Flat(v vec.Vector) vec.Vector` | Fixed-size | Copies matrix into provided vector |
| `Matrix() Matrix` | Fixed-size | Returns Matrix view of fixed-size matrix |
| `Clone() Matrix/*Matrix` | All | Creates deep copy of matrix |
| `Row(row int) vec.Vector` | All | Returns slice view of row |
| `Col(col int, v vec.Vector) vec.Vector` | All | Copies column into vector |
| `SetRow(row int, v vec.Vector)` | All | Sets row from vector |
| `SetCol(col int, v vec.Vector)` | All | Sets column from vector |
| `Diagonal(dst vec.Vector) vec.Vector` | All | Extracts diagonal elements |
| `SetDiagonal(v vec.Vector)` | All | Sets diagonal elements |
| `Rows() int` | All | Returns number of matrix rows |
| `Cols() int` | All | Returns number of matrix columns |
| `Rank() int` | All | Returns numerical rank (epsilon-based row-reduction) |

### Arithmetic Operations

All arithmetic operations are **in-place** and return the matrix for chaining.

| Method | Type | Description | BLAS Equivalent |
|--------|------|-------------|-----------------|
| `Add(m1) Matrix/*Matrix` | All | `m = m + m1` (element-wise) | Row-wise Axpy |
| `Sub(m1) Matrix/*Matrix` | All | `m = m - m1` (element-wise) | Row-wise Axpy (alpha=-1) |
| `MulC(c float32) Matrix/*Matrix` | All | `m = m * c` (scalar) | Row-wise Scal |
| `DivC(c float32) Matrix/*Matrix` | All | `m = m / c` (scalar) | Row-wise Scal (alpha=1/c) |

### Matrix-Matrix Operations

| Method | Type | Description | BLAS Equivalent |
|--------|------|-------------|-----------------|
| `Mul(a, b) Matrix/*Matrix` | All | `m = a * b` | Gemm_NN |
| `Transpose(m1) Matrix/*Matrix` | All | `m = m1^T` | Custom (copy + transpose) |
| `MulDiag(a, b) Matrix/*Matrix` | All | Element-wise multiply rows by vector | Row-wise Hadamard |

### Matrix-Vector Operations

| Method | Type | Description | BLAS Equivalent |
|--------|------|-------------|-----------------|
| `MulVec(v, dst) vec.Vector` | All | `dst = m * v` | Gemv_N |
| `MulVecT(v, dst) vec.Vector` | All | `dst = m^T * v` | Gemv_T |

### Geometric Operations

| Method | Type | Description |
|--------|------|-------------|
| `Rotation2D(a float32)` | Matrix, Matrix2x2 | 2D rotation matrix |
| `RotationX(a float32)` | Matrix, Matrix3x3, Matrix4x4 | Rotation around X axis |
| `RotationY(a float32)` | Matrix, Matrix3x3, Matrix4x4 | Rotation around Y axis |
| `RotationZ(a float32)` | Matrix, Matrix3x3, Matrix4x4 | Rotation around Z axis |
| `Orientation(q vec.Quaternion)` | Matrix, Matrix3x3, Matrix4x4 | Rotation from quaternion |
| `Eye() Matrix/*Matrix` | All | Identity matrix |
| `Det() float32` | All | Determinant (square matrices only) |

### Linear Algebra Operations

All linear algebra operations use primitives from the `primitive` package for optimized implementations.

| Method | Type | Description | Primitive Used |
|--------|------|-------------|----------------|
| `LU(L, U) Matrix/*Matrix` | All | LU decomposition | Getrf |
| `Inverse(dst) error` | All | Matrix inversion | Getrf_IP, Getri |
| `QRDecompose(dst *QRResult) error` | Matrix | QR decomposition | Geqrf |
| `QR(dst *QRResult) error` | Matrix | Reconstruct Q from QR | Orgqr |
| `SVD(dst *SVDResult) error` | Matrix | Singular value decomposition | Gesvd |
| `PseudoInverse(dst) error` | Matrix | Moore-Penrose pseudo-inverse | Gepseu |
| `Cholesky(dst) error` | Matrix | Cholesky decomposition | (manual with Dot) |
| `CholeskySolve(b, dst) error` | Matrix | Solve using Cholesky | (manual with Dot) |
| `NNLS(A, B, dst, rangeVal) error` | Function | Non-negative least squares | Gnnls |
| `LDP(G, H, dst, rangeVal) error` | Function | Least distance programming | (uses NNLS) |
| `DampedLeastSquares(lambda, dst) error` | Matrix | Damped least squares | (uses Inverse) |

### Submatrix Operations

| Method | Type | Description |
|--------|------|-------------|
| `Submatrix(row, col int, m1 Matrix) Matrix` | All | Extract submatrix into m1 |
| `SetSubmatrix(row, col int, m1 Matrix)` | All | Set submatrix from m1 |
| `SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32)` | All | Set submatrix from raw values |

## Current Implementation Details

### Matrix Type (Variable Size)

**Element-Wise Operations:**
- `Add()`, `Sub()`, `MulC()`, `DivC()`: Use `primitive.SumArr`, `primitive.DiffArr`, `primitive.Scal` on flattened matrices
- Zero-copy flattening when matrices are contiguous

**Matrix-Vector Operations:**
- `MulVec()`: Uses `primitive.Gemv_N` on flattened matrix (zero-copy if contiguous)
- `MulVecT()`: Uses `primitive.Gemv_T` on flattened matrix (zero-copy if contiguous)

**Matrix-Matrix Operations:**
- `Mul()`: Uses `primitive.Gemm_NN` on flattened matrices (zero-copy if contiguous)
- `Transpose()`: Uses direct loops for transpose (no primitive, but optimized)
- `MulDiag()`: Uses `primitive.HadamardProduct()` per row

**Linear Algebra Operations:**
- `Inverse()`: Uses `primitive.Getrf_IP` and `primitive.Getri`
- `LU()`: Uses `primitive.Getrf`
- `QRDecompose()`: Uses `primitive.Geqrf`
- `QR()`: Uses `primitive.Orgqr`
- `SVD()`: Uses `primitive.Gesvd`
- `PseudoInverse()`: Uses `primitive.Gepseu` (which internally uses `Gesvd`)
- `Cholesky()` and `CholeskySolve()`: Use `primitive.Dot` for inner products
- `Det()`: Uses `primitive.Axpy` for row operations
- `NNLS()`: Uses `primitive.Gnnls`

### Fixed-Size Matrices

**Current Implementation:**
- All operations use **inline loops** or **direct computations**
- No flattening before operations (direct element access)
- No use of primitives (except `HadamardProduct` in `MulDiag` for Matrix type)

**Rationale:**
- Small fixed sizes (4-16 elements) benefit from direct computation
- Function call overhead > direct computation for small matrices
- Compiler can optimize inline loops better (unrolling, SIMD)

## Performance Characteristics

### Matrix (Variable Size)
- Can be non-contiguous (performance penalty)
- Uses `Flat()` for primitives (zero-copy if contiguous, copy otherwise)
- Row-wise operations have function call overhead per row
- Better performance for larger matrices (> 100 elements)

### Fixed-Size Matrices
- Always contiguous in memory
- Direct computation (zero function call overhead)
- Compiler optimizations (loop unrolling, SIMD)
- Optimal for small fixed-size matrices

## Memory Layout

### Matrix
- Backed by `[][]float32` slice
- Rows may not be contiguous (each row is a separate slice)
- Requires `IsContiguous()` check before zero-copy flattening

### Fixed-Size Matrices
- Backed by arrays `[N][M]float32`
- Always contiguous in memory
- Stack-allocated (when not referenced)
- Fixed-size guarantees

## Zero-Copy Flattening Strategy

### For Matrix Type

```go
func (m Matrix) Flat() []float32 {
    // Fast path: zero-copy if contiguous
    if m.IsContiguous() {
        return unsafe.Slice((*float32)(unsafe.Pointer(&m[0][0])), rows*cols)
    }
    
    // Slow path: copy if not contiguous
    // ... copy implementation
}
```

**Benefits:**
- Zero allocation for contiguous matrices
- Works with BLAS primitives that expect flat arrays
- No performance penalty for normal operation

### For Fixed-Size Matrices

Fixed-size matrices use explicit copy to provided vector:
- No allocation (user provides vector)
- Explicit control over memory
- Safe (no unsafe operations needed)

## Dependencies

- `github.com/chewxy/math32` - Math functions (Cos, Sin, etc.)
- `github.com/itohio/EasyRobot/pkg/core/math/vec` - Vector operations
- `github.com/itohio/EasyRobot/pkg/core/math/primitive` - BLAS/LAPACK primitives

## Testing

- Unit tests for all operations
- Benchmark tests for performance critical paths
- Edge case tests (zero rows, zero columns, singular matrices)
- Fixed-size vs. variable-size performance comparisons
- Behavioural parity tests verifying slice-backed `Matrix` operations mutate receivers in-place while fixed-size variants (`Matrix2x2`, `Matrix3x3`, `Matrix4x4`, `Matrix3x4`, `Matrix4x3`) return updated copies
- Contiguous vs. non-contiguous matrix tests

