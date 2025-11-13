package mat

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

const (
	// SingularityTolerance is the tolerance for detecting singular matrices
	SingularityTolerance = 1e-6
)

var (
	// ErrNotSquare is returned when trying to invert a non-square matrix
	ErrNotSquare = errors.New("matrix must be square for inverse")
	// ErrSingular is returned when trying to invert a singular matrix
	ErrSingular = errors.New("matrix is singular (determinant near zero)")
)

// Inverse calculates the inverse of a square matrix using LU decomposition.
// Returns error if matrix is not square or singular.
// Destination matrix must be properly sized (same as source).
func (m Matrix) Inverse(dst matTypes.Matrix) error {
	rows := len(m)
	if rows == 0 {
		return ErrNotSquare
	}
	cols := len(m[0])
	if rows != cols {
		return ErrNotSquare
	}

	var dstMat Matrix
	switch out := dst.(type) {
	case *Matrix2x2:
		dstMat = out.View().(Matrix)
	case *Matrix3x3:
		dstMat = out.View().(Matrix)
	case *Matrix4x4:
		dstMat = out.View().(Matrix)
	case *Matrix4x3:
		dstMat = out.View().(Matrix)
	case *Matrix3x4:
		dstMat = out.View().(Matrix)
	case Matrix:
		dstMat = out
	default:
		panic(fmt.Sprintf("Matrix2x2.Inverse: unsupported destination type %T", dst))
	}

	mFlat := m.Flat()
	dstFlat := dstMat.Flat()
	ldA := len(m[0])
	ldInv := len(dstMat[0])

	work := make([]float32, len(mFlat))
	copy(work, mFlat)
	ipiv := make([]int, rows)
	if err := fp32.Getrf_IP(work, ipiv, ldA, rows, cols); err != nil {
		return ErrSingular
	}

	// Use Getri to compute inverse from LU decomposition
	if err := fp32.Getri(dstFlat, work, ldA, ldInv, rows, ipiv); err != nil {
		return ErrSingular
	}

	return nil
}

// Inverse calculates the inverse of a Matrix2x2 using direct formula.
func (m Matrix2x2) Inverse(dst matTypes.Matrix) error {
	det := m.Det()
	if math32.Abs(det) < SingularityTolerance {
		return ErrSingular
	}

	inv := Matrix2x2{
		{m[1][1] / det, -m[0][1] / det},
		{-m[1][0] / det, m[0][0] / det},
	}

	switch out := dst.(type) {
	case *Matrix2x2:
		*out = inv
	case Matrix2x2:
		panic("Matrix2x2.Inverse: destination must be *Matrix2x2")
	default:
		panic(fmt.Sprintf("Matrix2x2.Inverse: unsupported destination type %T", dst))
	}

	return nil
}

// Inverse calculates the inverse of a Matrix3x3 using direct formula.
func (m Matrix3x3) Inverse(dst matTypes.Matrix) error {
	det := m.Det()
	if math32.Abs(det) < SingularityTolerance {
		return ErrSingular
	}

	invDet := 1.0 / det

	inv := Matrix3x3{
		{(m[1][1]*m[2][2] - m[1][2]*m[2][1]) * invDet, (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet, (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet},
		{(m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet, (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet, (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invDet},
		{(m[1][0]*m[2][1] - m[1][1]*m[2][0]) * invDet, (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invDet, (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet},
	}

	switch out := dst.(type) {
	case *Matrix3x3:
		*out = inv
	case Matrix:
		out.CopyFrom(inv.View())
	case Matrix3x3:
		panic("Matrix3x3.Inverse: destination must be *Matrix3x3")
	default:
		panic(fmt.Sprintf("Matrix3x3.Inverse: unsupported destination type %T", dst))
	}

	return nil
}

// Inverse calculates the inverse of a Matrix4x4 using LU decomposition.
func (m Matrix4x4) Inverse(dst matTypes.Matrix) error {
	det := m.Det()
	if math32.Abs(det) < SingularityTolerance {
		return ErrSingular
	}

	out := dst.(*Matrix4x4)

	// LU decomposition
	var L, U Matrix4x4
	m.LU(&L, &U)

	// Identity matrix
	I := Matrix4x4{}.Eye().(Matrix4x4)

	// Solve L * Y = I for Y (forward substitution)
	var Y Matrix4x4
	for col := 0; col < 4; col++ {
		for row := 0; row < 4; row++ {
			sum := I[row][col]
			for k := 0; k < row; k++ {
				sum -= L[row][k] * Y[k][col]
			}
			Y[row][col] = sum / L[row][row]
		}
	}

	// Solve U * X = Y for X (back substitution)
	for col := 0; col < 4; col++ {
		for row := 3; row >= 0; row-- {
			sum := Y[row][col]
			for k := row + 1; k < 4; k++ {
				sum -= U[row][k] * out[k][col]
			}
			out[row][col] = sum / U[row][row]
		}
	}

	return nil
}
