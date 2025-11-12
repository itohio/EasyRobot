// Generated code. DO NOT EDIT

package mat

import (
	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

type Matrix2x2 [2][2]float32

var _ matTypes.Matrix = (*Matrix2x2)(nil)

func (m *Matrix2x2) view() Matrix {
	return Matrix{
		m[0][:],
		m[1][:],
	}
}

func (m *Matrix2x2) IsContiguous() bool {
	return true
}

func New2x2(arr ...float32) Matrix2x2 {
	m := Matrix2x2{}
	if arr != nil {
		for i := range m {
			copy(m[i][:], arr[i*2 : i*2+2][:])
		}
	}
	return m
}

// Returns a flat representation of this matrix.
func (m *Matrix2x2) Flat() []float32 {
	return cloneFlat(m.view().Flat())
}

// Returns a Matrix view of this matrix.
// The view actually contains slices of original matrix rows.
// This way original matrix can be modified.
func (m *Matrix2x2) Matrix() matTypes.Matrix {
	m1 := make(Matrix, len(m))
	for i := range m {
		m1[i] = m[i][:]
	}
	return m1
}

// Fills destination matrix with a 2D rotation
// Matrix size must be at least 2x2
func (m *Matrix2x2) Rotation2D(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 2, 2,
		c, -s,
		s, c,
	)
}

// Fills destination matrix with identity matrix.
func (m *Matrix2x2) Eye() matTypes.Matrix {
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
func (m *Matrix2x2) Row(row int) vecTypes.Vector {
	return vec.Vector(m[row][:])
}

// Returns a copy of the matrix column.
func (m *Matrix2x2) Col(col int, v vecTypes.Vector) vecTypes.Vector {
	dst := ensureVector(v, "Matrix2x2.Col")
	for i, row := range m {
		dst[i] = row[col]
	}
	return dst
}

func (m *Matrix2x2) SetRow(row int, v vecTypes.Vector) matTypes.Matrix {
	src := ensureVector(v, "Matrix2x2.SetRow")
	copy(m[row][:], src)
	return m
}

func (m *Matrix2x2) SetCol(col int, v vecTypes.Vector) matTypes.Matrix {
	src := ensureVector(v, "Matrix2x2.SetCol")
	for i, val := range src {
		m[i][col] = val
	}
	return m
}

// Size of the destination vector must equal to number of rows
func (m *Matrix2x2) Diagonal(dst vecTypes.Vector) vecTypes.Vector {
	out := ensureVector(dst, "Matrix2x2.Diagonal")
	for i, row := range m {
		out[i] = row[i]
	}
	return out
}

// Size of the vector must equal to number of rows
func (m *Matrix2x2) SetDiagonal(v vecTypes.Vector) matTypes.Matrix {
	src := ensureVector(v, "Matrix2x2.SetDiagonal")
	for i, val := range src {
		m[i][i] = val
	}
	return m
}

// FromDiagonal2x2 creates a 2x2 diagonal matrix from diagonal values.
// Returns a matrix with zeros everywhere except the diagonal.
func FromDiagonal2x2(d0, d1 float32) Matrix2x2 {
	m := Matrix2x2{}
	m[0][0] = d0
	m[1][1] = d1
	return m
}

// FromVector2x2 creates a 2x2 diagonal matrix from a 2D vector.
// The vector elements become the diagonal elements of the matrix.
func FromVector2x2(v vec.Vector2D) Matrix2x2 {
	m := Matrix2x2{}
	m[0][0] = v[0]
	m[1][1] = v[1]
	return m
}

func (m *Matrix2x2) Submatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	dst := ensureMatrix(m1, "Matrix2x2.Submatrix")
	cols := len(dst[0])
	for i := range dst {
		copy(dst[i], m[row+i][col:col+cols])
	}
	return dst
}

func (m *Matrix2x2) SetSubmatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	src := ensureMatrix(m1, "Matrix2x2.SetSubmatrix")
	for i := range src {
		copy(m[row+i][col:col+len(src[i])], src[i])
	}
	return m
}

func (m *Matrix2x2) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) matTypes.Matrix {
	for i := 0; i < rows1; i++ {
		copy(m[row+i][col:col+cols1], m1[i*cols1:i*cols1+cols1])
	}
	return m
}

func (m *Matrix2x2) Clone() matTypes.Matrix {
	clone := &Matrix2x2{}
	for i, row := range m {
		copy(clone[i][:], row[:])
	}
	return clone
}

// Transposes matrix m1 and stores the result in the destination matrix
// destination matrix must be of appropriate size.
// NOTE: Does not support in place transpose
func (m *Matrix2x2) Transpose(m1 matTypes.Matrix) matTypes.Matrix {
	src := ensureMatrix(m1, "Matrix2x2.Transpose")
	for i, row := range src {
		for j, val := range row {
			m[j][i] = val
		}
	}
	return m
}

func (m *Matrix2x2) Add(m1 matTypes.Matrix) matTypes.Matrix {
	other := ensureMatrix(m1, "Matrix2x2.Add")
	for i := 0; i < 2; i++ {
		m[i][0] += other[i][0]
		m[i][1] += other[i][1]
	}
	return m
}

func (m *Matrix2x2) Sub(m1 matTypes.Matrix) matTypes.Matrix {
	other := ensureMatrix(m1, "Matrix2x2.Sub")
	for i := 0; i < 2; i++ {
		m[i][0] -= other[i][0]
		m[i][1] -= other[i][1]
	}
	return m
}

func (m *Matrix2x2) MulC(c float32) matTypes.Matrix {
	for i := range m {
		vec.Vector(m[i][:]).MulC(c)
	}
	return m
}

func (m *Matrix2x2) DivC(c float32) matTypes.Matrix {
	for i := range m {
		vec.Vector(m[i][:]).DivC(c)
	}
	return m
}

// Destination matrix must be properly sized.
// given that a is MxN and b is NxK
// then destinatiom matrix must be MxK
func (m *Matrix2x2) Mul(a matTypes.Matrix, b matTypes.Matrix) matTypes.Matrix {
	aMat := ensureMatrix(a, "Matrix2x2.Mul.a")
	bMat := ensureMatrix(b, "Matrix2x2.Mul.b")

	m00 := aMat[0][0]*bMat[0][0] + aMat[0][1]*bMat[1][0]
	m01 := aMat[0][0]*bMat[0][1] + aMat[0][1]*bMat[1][1]
	m10 := aMat[1][0]*bMat[0][0] + aMat[1][1]*bMat[1][0]
	m11 := aMat[1][0]*bMat[0][1] + aMat[1][1]*bMat[1][1]

	m[0][0] = m00
	m[0][1] = m01
	m[1][0] = m10
	m[1][1] = m11

	return m
}

// Only makes sense for square matrices.
// Vector size must be equal to number of rows/cols
func (m *Matrix2x2) MulDiag(a matTypes.Matrix, b vecTypes.Vector) matTypes.Matrix {
	aMat := ensureMatrix(a, "Matrix2x2.MulDiag")
	bVec := ensureVector(b, "Matrix2x2.MulDiag.b")
	for i := 0; i < 2; i++ {
		m[i][0] = aMat[i][0] * bVec[0]
		m[i][1] = aMat[i][1] * bVec[1]
	}

	return m
}

// Vector must have a size equal to number of cols.
// Destination vector must have a size equal to number of rows.
func (m *Matrix2x2) MulVec(v vecTypes.Vector, dst vecTypes.Vector) vecTypes.Vector {
	src := ensureVector(v, "Matrix2x2.MulVec.v")
	dstVec := ensureVector(dst, "Matrix2x2.MulVec.dst")
	dstVec[0] = src[0]*m[0][0] + src[1]*m[0][1]
	dstVec[1] = src[0]*m[1][0] + src[1]*m[1][1]
	return dstVec
}

// Vector must have a size equal to number of rows.
// Destination vector must have a size equal to number of cols.
func (m *Matrix2x2) MulVecT(v vecTypes.Vector, dst vecTypes.Vector) vecTypes.Vector {
	src := ensureVector(v, "Matrix2x2.MulVecT.v")
	dstVec := ensureVector(dst, "Matrix2x2.MulVecT.dst")
	dstVec[0] = src[0]*m[0][0] + src[1]*m[1][0]
	dstVec[1] = src[0]*m[0][1] + src[1]*m[1][1]
	return dstVec
}

// Determinant only valid for square matrix
// Undefined behavior for non square matrices
func (m *Matrix2x2) Det() float32 {
	return m.view().Det()
}

// LU decomposition into two triangular matrices
// NOTE: Assume, that l&u matrices are set to zero
// Matrix must be square and M, L and U matrix sizes must be equal
func (m *Matrix2x2) LU(L, U matTypes.Matrix) {
	LMat := ensureMatrix(L, "Matrix2x2.LU.L")
	UMat := ensureMatrix(U, "Matrix2x2.LU.U")
	for i := range m {
		for k := i; k < len(m); k++ {
			var sum float32
			for j := 0; j < i; j++ {
				sum += LMat[i][j] * UMat[j][k]
			}
			UMat[i][k] = m[i][k] - sum
		}

		for k := i; k < len(m); k++ {
			if i == k {
				LMat[i][i] = 1
			} else {
				var sum float32
				for j := 0; j < i; j++ {
					sum += LMat[k][j] * UMat[j][i]
				}
				LMat[k][i] = (m[k][i] - sum) / UMat[i][i]
			}
		}
	}
}

func (m *Matrix2x2) SetColFromRow(col int, rowStart int, v vecTypes.Vector) matTypes.Matrix {
	src := ensureVector(v, "Matrix2x2.SetColFromRow")
	for i, val := range src {
		if rowStart+i < len(m) {
			m[rowStart+i][col] = val
		}
	}
	return m
}

// GetCol extracts a column from the matrix as a vector.
// dst must be at least as long as the number of rows.
func (m *Matrix2x2) GetCol(col int, dst vecTypes.Vector) vecTypes.Vector {
	out := ensureVector(dst, "Matrix2x2.GetCol")
	for i := range m {
		if i < len(out) {
			out[i] = m[i][col]
		}
	}
	return out
}

func (m *Matrix2x2) Cholesky(dst matTypes.Matrix) error {
	return m.view().Cholesky(dst)
}

func (m *Matrix2x2) CholeskySolve(b vecTypes.Vector, dst vecTypes.Vector) error {
	return m.view().CholeskySolve(b, dst)
}

func (m *Matrix2x2) QRDecompose(dst *matTypes.QRResult) error {
	return m.view().QRDecompose(dst)
}

func (m *Matrix2x2) QR(dst *matTypes.QRResult) error {
	return m.view().QR(dst)
}

func (m *Matrix2x2) PseudoInverse(dst matTypes.Matrix) error {
	return m.view().PseudoInverse(dst)
}

func (m *Matrix2x2) DampedLeastSquares(lambda float32, dst matTypes.Matrix) error {
	return m.view().DampedLeastSquares(lambda, dst)
}

func (m *Matrix2x2) SVD(dst *matTypes.SVDResult) error {
	return m.view().SVD(dst)
}

// Fills destination matrix with a rotation around X axis
// Matrix size must be at least 3x3
func (m *Matrix2x2) RotationX(a float32) matTypes.Matrix {
	panic("Matrix2x2.RotationX: unsupported operation")
}

func (m *Matrix2x2) RotationY(a float32) matTypes.Matrix {
	panic("Matrix2x2.RotationY: unsupported operation")
}

func (m *Matrix2x2) RotationZ(a float32) matTypes.Matrix {
	panic("Matrix2x2.RotationZ: unsupported operation")
}

func (m *Matrix2x2) Orientation(q vecTypes.Quaternion) matTypes.Matrix {
	panic("Matrix2x2.Orientation: unsupported operation")
}

func (m *Matrix2x2) Quaternion() vecTypes.Quaternion {
	panic("Matrix2x2.Quaternion: unsupported operation")
}
