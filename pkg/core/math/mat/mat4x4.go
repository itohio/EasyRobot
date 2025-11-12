// Generated code. DO NOT EDIT

package mat

import (
	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

type Matrix4x4 [4][4]float32

var _ matTypes.Matrix = (*Matrix4x4)(nil)

func (m *Matrix4x4) view() Matrix {
	return Matrix{
		m[0][:],
		m[1][:],
		m[2][:],
		m[3][:],
	}
}

func (m *Matrix4x4) IsContiguous() bool {
	return true
}

func New4x4(arr ...float32) Matrix4x4 {
	m := Matrix4x4{}
	if arr != nil {
		for i := range m {
			copy(m[i][:], arr[i*4 : i*4+4][:])
		}
	}
	return m
}

// Returns a Matrix view of this matrix.
// The view actually contains slices of original matrix rows.
// This way original matrix can be modified.
func (m *Matrix4x4) Matrix() matTypes.Matrix {
	return m.view()
}

// Returns a flat representation of this matrix.
func (m *Matrix4x4) Flat() []float32 {
	return cloneFlat(m.view().Flat())
}

// Fills destination matrix with a rotation around X axis
// Matrix size must be at least 3x3
func (m *Matrix4x4) RotationX(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		1, 0, 0,
		0, c, -s,
		0, s, c,
	)
}

// Fills destination matrix with a rotation around Y axis
// Matrix size must be at least 3x3
func (m *Matrix4x4) RotationY(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, 0, s,
		0, 1, 0,
		-s, 0, c,
	)
}

// Fills destination matrix with a rotation around Z axis
// Matrix size must be at least 3x3
func (m *Matrix4x4) RotationZ(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, -s, 0,
		s, c, 0,
		0, 0, 1,
	)
}

func (m *Matrix4x4) Rotation2D(a float32) matTypes.Matrix {
	m.view().Rotation2D(a)
	return m
}

// Build orientation matrix from quaternion
// Matrix size must be at least 3x3
// Quaternion axis must be unit vector
func (m *Matrix4x4) Orientation(q vecTypes.Quaternion) matTypes.Matrix {
	m.view().Orientation(q)
	return m
}

// Fills destination matrix with identity matrix.
func (m *Matrix4x4) Eye() matTypes.Matrix {
	for i := range m {
		row := m[i][:]
		for j := range row {
			row[j] = 0
		}
	}
	for i := range m {
		m[i][i] = 1
	}
	return m
}

// Returns a slice to the row.
func (m *Matrix4x4) Row(row int) vecTypes.Vector {
	return vec.Vector(m[row][:])
}

// Returns a copy of the matrix column.
func (m *Matrix4x4) Col(col int, v vecTypes.Vector) vecTypes.Vector {
	return m.view().Col(col, v)
}

func (m *Matrix4x4) SetRow(row int, v vecTypes.Vector) matTypes.Matrix {
	m.view().SetRow(row, v)
	return m
}

func (m *Matrix4x4) SetCol(col int, v vecTypes.Vector) matTypes.Matrix {
	m.view().SetCol(col, v)
	return m
}

func (m *Matrix4x4) SetColFromRow(col int, rowStart int, v vecTypes.Vector) matTypes.Matrix {
	m.view().SetColFromRow(col, rowStart, v)
	return m
}

// Size of the destination vector must equal to number of rows
func (m *Matrix4x4) Diagonal(dst vecTypes.Vector) vecTypes.Vector {
	return m.view().Diagonal(dst)
}

// Size of the vector must equal to number of rows
func (m *Matrix4x4) SetDiagonal(v vecTypes.Vector) matTypes.Matrix {
	m.view().SetDiagonal(v)
	return m
}

// FromDiagonal4x4 creates a 4x4 diagonal matrix from diagonal values.
// Returns a matrix with zeros everywhere except the diagonal.
func FromDiagonal4x4(d0, d1, d2, d3 float32) Matrix4x4 {
	m := Matrix4x4{}
	m[0][0] = d0
	m[1][1] = d1
	m[2][2] = d2
	m[3][3] = d3
	return m
}

// FromVector4x4 creates a 4x4 diagonal matrix from a 4D vector.
// The vector elements become the diagonal elements of the matrix.
func FromVector4x4(v vec.Vector4D) Matrix4x4 {
	m := Matrix4x4{}
	m[0][0] = v[0]
	m[1][1] = v[1]
	m[2][2] = v[2]
	m[3][3] = v[3]
	return m
}

func (m *Matrix4x4) Submatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	return m.view().Submatrix(row, col, m1)
}

func (m *Matrix4x4) SetSubmatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	m.view().SetSubmatrix(row, col, m1)
	return m
}

func (m *Matrix4x4) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) matTypes.Matrix {
	m.view().SetSubmatrixRaw(row, col, rows1, cols1, m1...)
	return m
}

func (m *Matrix4x4) Clone() matTypes.Matrix {
	clone := &Matrix4x4{}
	for i, row := range m {
		copy(clone[i][:], row[:])
	}
	return clone
}

// Transposes matrix m1 and stores the result in the destination matrix
// destination matrix must be of appropriate size.
// NOTE: Does not support in place transpose
func (m *Matrix4x4) Transpose(m1 matTypes.Matrix) matTypes.Matrix {
	m.view().Transpose(m1)
	return m
}

func (m *Matrix4x4) Add(m1 matTypes.Matrix) matTypes.Matrix {
	m.view().Add(m1)
	return m
}

func (m *Matrix4x4) Sub(m1 matTypes.Matrix) matTypes.Matrix {
	m.view().Sub(m1)
	return m
}

func (m *Matrix4x4) MulC(c float32) matTypes.Matrix {
	m.view().MulC(c)
	return m
}

func (m *Matrix4x4) DivC(c float32) matTypes.Matrix {
	m.view().DivC(c)
	return m
}

// Destination matrix must be properly sized.
// given that a is MxN and b is NxK
// then destinatiom matrix must be MxK
func (m *Matrix4x4) Mul(a matTypes.Matrix, b matTypes.Matrix) matTypes.Matrix {
	m.view().Mul(a, b)
	return m
}

// Only makes sense for square matrices.
// Vector size must be equal to number of rows/cols
func (m *Matrix4x4) MulDiag(a matTypes.Matrix, b vecTypes.Vector) matTypes.Matrix {
	m.view().MulDiag(a, b)
	return m
}

// Vector must have a size equal to number of cols.
// Destination vector must have a size equal to number of rows.
func (m *Matrix4x4) MulVec(v vecTypes.Vector, dst vecTypes.Vector) vecTypes.Vector {
	return m.view().MulVec(v, dst)
}

// Vector must have a size equal to number of rows.
// Destination vector must have a size equal to number of cols.
func (m *Matrix4x4) MulVecT(v vecTypes.Vector, dst vecTypes.Vector) vecTypes.Vector {
	return m.view().MulVecT(v, dst)
}

// Determinant only valid for square matrix
// Undefined behavior for non square matrices
func (m *Matrix4x4) Det() float32 {
	return m.view().Det()
}

// LU decomposition into two triangular matrices
// NOTE: Assume, that l&u matrices are set to zero
// Matrix must be square and M, L and U matrix sizes must be equal
func (m *Matrix4x4) LU(L, U matTypes.Matrix) {
	m.view().LU(L, U)
}

// / https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
// / Must be at least 3x3 matrix
func (m *Matrix4x4) Quaternion() vecTypes.Quaternion {
	return m.view().Quaternion()
}

func (m *Matrix4x4) Cholesky(dst matTypes.Matrix) error {
	return m.view().Cholesky(dst)
}

func (m *Matrix4x4) CholeskySolve(b vecTypes.Vector, dst vecTypes.Vector) error {
	return m.view().CholeskySolve(b, dst)
}

func (m *Matrix4x4) QRDecompose(dst *matTypes.QRResult) error {
	return m.view().QRDecompose(dst)
}

func (m *Matrix4x4) QR(dst *matTypes.QRResult) error {
	return m.view().QR(dst)
}

func (m *Matrix4x4) PseudoInverse(dst matTypes.Matrix) error {
	return m.view().PseudoInverse(dst)
}

func (m *Matrix4x4) DampedLeastSquares(lambda float32, dst matTypes.Matrix) error {
	return m.view().DampedLeastSquares(lambda, dst)
}

func (m *Matrix4x4) SVD(dst *matTypes.SVDResult) error {
	return m.view().SVD(dst)
}

func (m *Matrix4x4) GetCol(col int, dst vecTypes.Vector) vecTypes.Vector {
	return m.view().GetCol(col, dst)
}
