package types

import (
	vec "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

// Core groups fundamental matrix access and manipulation operations.
type Core interface {
	IsContiguous() bool
	Flat() []float32
	Matrix() Matrix
	Eye() Matrix
	Clone() Matrix

	Row(row int) vec.Vector
	Col(col int, v vec.Vector) vec.Vector
	SetRow(row int, v vec.Vector) Matrix
	SetCol(col int, v vec.Vector) Matrix
	SetColFromRow(col int, rowStart int, v vec.Vector) Matrix
	GetCol(col int, dst vec.Vector) vec.Vector
	Diagonal(dst vec.Vector) vec.Vector
	SetDiagonal(v vec.Vector) Matrix
	Submatrix(row, col int, m1 Matrix) Matrix
	SetSubmatrix(row, col int, m1 Matrix) Matrix
	SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) Matrix
	Transpose(m1 Matrix) Matrix
}

// Rotations enumerates routines for constructing rotation matrices.
type Rotations interface {
	Rotation2D(a float32) Matrix
	RotationX(a float32) Matrix
	RotationY(a float32) Matrix
	RotationZ(a float32) Matrix
	Orientation(q vec.Quaternion) Matrix
}

// Arithmetic encapsulates scalar and element-wise matrix operations.
type Arithmetic interface {
	Add(m1 Matrix) Matrix
	Sub(m1 Matrix) Matrix
	MulC(c float32) Matrix
	DivC(c float32) Matrix
}

// Multiplication captures matrix-matrix and matrix-vector multiplication.
type Multiplication interface {
	Mul(a Matrix, b Matrix) Matrix
	MulDiag(a Matrix, b vec.Vector) Matrix
	MulVec(v vec.Vector, dst vec.Vector) vec.Vector
	MulVecT(v vec.Vector, dst vec.Vector) vec.Vector
}

// Factorization exposes higher-level decomposition and inversion routines.
type Factorization interface {
	Det() float32
	LU(L, U Matrix)
	Quaternion() vec.Quaternion
	SVD(dst *SVDResult) error
	Cholesky(dst Matrix) error
	CholeskySolve(b vec.Vector, dst vec.Vector) error
	QRDecompose(dst *QRResult) error
	QR(dst *QRResult) error
	Inverse(dst Matrix) error
	PseudoInverse(dst Matrix) error
	DampedLeastSquares(lambda float32, dst Matrix) error
}

// Matrix composes all matrix capabilities provided by the mat package.
type Matrix interface {
	Core
	Rotations
	Arithmetic
	Multiplication
	Factorization
}

// QRResult holds the result of QR decomposition.
// M = Q * R
// Note: Input matrix M is modified (contains Q via Householder vec.vectors)
type QRResult struct {
	Q        Matrix     // Orthogonal matrix (row x row) - stored in input matrix
	R        Matrix     // Upper triangular matrix (row x col)
	C        vec.Vector // Householder constants (col length)
	D        vec.Vector // Diagonal of R (col length)
	Singular bool       // True if matrix is singular
}

// SVDResult holds the result of Singular Value Decomposition.
// M = U * Σ * V^T
// Note: Input matrix M is modified (contains U on output)
type SVDResult struct {
	U  Matrix     // Left singular vec.vectors (row x row) - stored in input matrix
	S  vec.Vector // Singular values (col length)
	Vt Matrix     // Right singular vec.vectors transposed (col x col)
}
