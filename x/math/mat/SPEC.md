# math/mat – Matrix Operations Specification

## Overview

The `mat` package provides matrix mathematics for robotics-oriented systems. Two families of types are exposed:

- `Matrix`: a runtime-sized, slice-backed, row-major matrix optimised for interoperability with BLAS-like primitives in `primitive/fp32`.
- Fixed-size array types (`Matrix2x2`, `Matrix3x3`, `Matrix4x4`, `Matrix3x4`, `Matrix4x3`) that trade heap allocations for value semantics suited to small, frequently-copied transforms.

All APIs operate on `float32` to match embedded/real-time constraints.

## Design Principles

1. **Row-Major Layout** – matches Go’s slice/array memory layout and keeps interoperability with the fp32 primitive layer via leading dimension control.
2. **Explicit Semantics** – variable-size matrices mutate receivers; fixed-size matrices return new values (Go arrays copy-on-return).
3. **Zero or Bounded Allocation** – hot paths reuse backing storage or rely on stack-allocated arrays.
4. **Unsafe When Worthwhile** – `Matrix.Flat()` falls back to `unsafe.Slice` only when contiguity is guaranteed.
5. **Contextual Algorithms** – direct formulas for very small systems, primitives (Getrf, Gemm, …) for larger ones.

## Type Catalogue

### `Matrix`

`type Matrix [][]float32`

- Created with `New(rows, cols int, backing ...float32)`. Supplying a `backing` slice yields a contiguous layout; otherwise storage is allocated.
- `View()` returns a shallow copy of the row slices. Use against the `types.Matrix` interface.
- `IsContiguous()` checks whether rows are adjacent in memory; required before exposing the matrix to primitives without copying.
- `Flat()` returns a `[]float32` in row-major order. Zero-copy when contiguous, copy otherwise.
- `Rank()` performs Gaussian elimination with partial pivoting; small matrices (≤4×4) run in a stack buffer.
- `Clone()` re-allocates a contiguous backing array.
- `Eye()` zeroes and writes an identity (square assumption is on the caller).
- Reference semantics: methods mutate receiver storage in place; callers should clone first when they need to preserve inputs.

### Fixed-Size Arrays

| Type         | Storage             | Notes                                                                 |
|--------------|---------------------|-----------------------------------------------------------------------|
| `Matrix2x2`  | `[2][2]float32`     | Direct formulas for `Det()`/`Inverse()`. Rotation limited to 2D.      |
| `Matrix3x3`  | `[3][3]float32`     | Implements SO(3) rotations, quaternion conversions, direct inverse.   |
| `Matrix4x4`  | `[4][4]float32`     | Homogeneous transforms support; `LU()` dedicated implementation.      |
| `Matrix3x4`  | `[3][4]float32`     | Intended for pose matrices; non-square factorisations panic.          |
| `Matrix4x3`  | `[4][3]float32`     | Complement to `Matrix3x4`; non-square factorisations panic.           |

Characteristics:

- `Flat()` always allocates a fresh slice (copy-on-return). Copy semantics are fine because array receivers are passed by value.
- Value semantics: operations on these array types return a modified copy. Callers must reassign the result (`m = m.Add(...)`).
- `CopyFrom` on fixed matrices is intentionally a no-op—the types are used by value.
- Submatrix helpers only accept full writes (`row == 0 && col == 0`). Alternate placements panic by design.
- Unsupported factorisations (`QR`, `SVD`, `PseudoInverse`, `Cholesky`, etc.) panic immediately to surface improper use.

### Shared Interface (`types.Matrix`)

The `types.Matrix` interface is the contract implemented by both `Matrix` and the fixed-size array types. It is composed of five smaller interfaces; every method listed below must exist on conforming types:

- **Core Manipulation**
  - `IsContiguous() bool`
  - `Flat() []float32`
  - `View() Matrix`
  - `Rows() int`
  - `Cols() int`
  - `Rank() int`
  - `Eye() Matrix`
  - `Clone() Matrix`
  - `CopyFrom(src Matrix)`
  - `Row(row int) vec.Vector`
  - `Col(col int, v vec.Vector) vec.Vector`
  - `SetRow(row int, v vec.Vector) Matrix`
  - `SetCol(col int, v vec.Vector) Matrix`
  - `SetColFromRow(col int, rowStart int, v vec.Vector) Matrix`
  - `GetCol(col int, dst vec.Vector) vec.Vector`
  - `Diagonal(dst vec.Vector) vec.Vector`
  - `SetDiagonal(v vec.Vector) Matrix`
  - `Submatrix(row, col int, m1 Matrix) Matrix`
  - `SetSubmatrix(row, col int, m1 Matrix) Matrix`
  - `SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) Matrix`
  - `Transpose(m1 Matrix) Matrix`

- **Rotation Constructors**
  - `Rotation2D(a float32) Matrix`
  - `RotationX(a float32) Matrix`
  - `RotationY(a float32) Matrix`
  - `RotationZ(a float32) Matrix`
  - `Orientation(q vec.Quaternion) Matrix`

- **Arithmetic**
  - `Add(m1 Matrix) Matrix`
  - `Sub(m1 Matrix) Matrix`
  - `MulC(c float32) Matrix`
  - `DivC(c float32) Matrix`

- **Multiplication**
  - `Mul(a Matrix, b Matrix) Matrix`
  - `MulDiag(a Matrix, b vec.Vector) Matrix`
  - `MulVec(v vec.Vector, dst vec.Vector) vec.Vector`
  - `MulVecT(v vec.Vector, dst vec.Vector) vec.Vector`

- **Factorisation & Advanced Ops**
  - `Det() float32`
  - `LU(L, U Matrix)`
  - `Quaternion() vec.Quaternion`
  - `SVD(dst *SVDResult) error`
  - `Cholesky(dst Matrix) error`
  - `CholeskySolve(b vec.Vector, dst vec.Vector) error`
  - `QRDecompose(dst *QRResult) error`
  - `QR(dst *QRResult) error`
  - `Inverse(dst Matrix) error`
  - `PseudoInverse(dst Matrix) error`
  - `DampedLeastSquares(lambda float32, dst Matrix) error`

Each fixed-size type only implements the methods that are meaningful for its dimensions; methods that cannot be satisfied (e.g. `SVD` on rectangular matrices) panic explicitly to surface misuse.

## Core Behaviour

### Construction & Accessors

- `New`, `New2x2`, `New3x3`, `New4x4`, `New3x4`, `New4x3` allow optional initializer slices (excess data ignored).
- `FromDiagonal`/`FromVector` produce square diagonal matrices (`Matrix`, `Matrix3x3`, `Matrix4x4` helpers exist).
- Row/column functions on array types bound-check and panic on invalid indices; `Matrix` assumes caller-provided bounds.
- `SetSubmatrixRaw` performs contiguous copy writes; new data length is validated implicitly by slicing.

### Arithmetic & Multiplication

- `Matrix` delegates to fp32 primitives (`ElemAdd`, `ElemSub`, `Scal`, `Gemv`, `Gemm`, `ElemMul`) on flattened data. Contiguous matrices reuse backing storage; non-contiguous matrices incur an intermediate copy.
- Fixed-size matrices compute inline with explicit summations; this preserves stack allocation and minimises interface dispatch.
- `MulDiag` scales rows by a vector; column-major variants are not provided.

### Transform Constructors

- `Rotation2D`, `RotationX/Y/Z`, and `Orientation` (quaternion) fill the rotational block using trigonometric identities.
- `Matrix` variants use `SetSubmatrixRaw`, which requires the receiver to be pre-sized appropriately.
- Quaternion conversion on `Matrix` returns an axis-angle derived quaternion (see `Quaternion()` in `mat.go`).
- Homogeneous helpers on `*Matrix4x4` (`Homogenous`, `HomogenousFromQuaternion`, `HomogenousFromEuler`, `SetRotation`, `SetTranslation`, `GetRotation`, `GetTranslation`, `Col3D`) combine rotation/translation blocks and normalise the fourth row/column.

### Linear Algebra & Solvers (Variable-Size)

- `Det()` clones the matrix and performs row-reduction with `Axpy`.
- `LU(L, U)` tries the fp32 `Getrf` path; on failure it falls back to manual Doolittle decomposition.
- `Inverse(dst)` (square matrices) uses `Getrf_IP` + `Getri`. Errors returned: `ErrNotSquare`, `ErrSingular`.
- `Cholesky(dst)` implements lower-triangular decomposition using `fp32.Dot`; `CholeskySolve` performs manual forward/back substitution with dot products for cache-friendliness.
- `QRDecompose`/`QR` wrap `Geqrf` and `Orgqr`, storing Householder coefficients in `QRResult.C/D`.
- `SVD` is a thin wrapper around `Gesvd`. Requires `rows >= cols`; otherwise returns an error instructing callers to transpose first.
- `PseudoInverse(dst)` uses `Gepseu` (SVD-based) and writes a `cols × rows` result.
- `DampedLeastSquares(lambda, dst)` computes `J^T (J J^T + λ² I)^{-1}` using the above building blocks.
- `NNLS` (Lawson-Hanson) and `LDP` (Least Distance Programming) live in this package but operate on `Matrix`/`vec.Vector`. They preserve original matrices by copying inputs that BLAS routines mutate.

### Fixed-Size Factorisations

- `Matrix2x2` and `Matrix3x3` provide direct `Inverse()` implementations returning errors when determinants fall below `SingularityTolerance`.
- `Matrix4x4.Inverse` relies on its bespoke `LU` and forward/back substitution. All other fixed-size types panic for unsupported factorisations to direct users to `Matrix`.

### Jacobian Utilities

- `CalculateJacobianColumn` constructs a combined linear/angular column for geometric Jacobians. Revolute joints compute `jointAxis × (p_ee - p_i)` for the translational part; prismatic joints copy the axis into the linear component and zero the angular part.

## Error & Panic Strategy

- Dimension mismatches, non-square requirements, and unsupported operations surface as panics for fixed-size types (programmer errors).
- Numerical issues return sentinel errors (`ErrNotSquare`, `ErrSingular`, `ErrNNLSBadDimensions`, etc.) on variable-size functions.
- Homogeneous helpers assume the receiver is at least 4×4; caller responsibility is documented via function comments.

## External Dependencies

- `github.com/chewxy/math32` – trigonometric and elementary functions.
- `github.com/itohio/EasyRobot/x/math/vec` – typed vector helpers (`Vector2D`, `Vector3D`, `Vector4D`, `Quaternion`).
- `github.com/itohio/EasyRobot/x/math/primitive/fp32` – BLAS/LAPACK-like routines (`Gemm`, `Gemv`, `Getrf`, `Gesvd`, `Gepseu`, …) and dot-product utilities.

## Testing Expectations

- Substantial table-driven coverage in `*_test.go` files ensures parity across variable-size and fixed-size types, including Jacobian, homogeneous transforms, inverse computations, and numerical edge cases (singular matrices, non-contiguous backing arrays).
- Benchmarks (`benchmark_inverse.go`, `benchmark_test.go`) emphasise regression detection on critical operations such as inversion and homogeneous transformations.

