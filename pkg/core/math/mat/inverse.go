package mat

import (
	"errors"
	"github.com/chewxy/math32"
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
func (m Matrix) Inverse(dst Matrix) error {
	rows := len(m)
	if rows == 0 {
		return ErrNotSquare
	}
	cols := len(m[0])
	if rows != cols {
		return ErrNotSquare
	}

	// Check if matrix is singular
	det := m.Det()
	if math32.Abs(det) < SingularityTolerance {
		return ErrSingular
	}

	// LU decomposition
	L := New(rows, cols)
	U := New(rows, cols)
	m.LU(L, U)

	// Identity matrix
	I := New(rows, cols)
	I.Eye()

	// Solve L * Y = I for Y (forward substitution)
	Y := New(rows, cols)
	for col := 0; col < cols; col++ {
		for row := 0; row < rows; row++ {
			sum := I[row][col]
			for k := 0; k < row; k++ {
				sum -= L[row][k] * Y[k][col]
			}
			Y[row][col] = sum / L[row][row]
		}
	}

	// Solve U * X = Y for X (back substitution)
	// X is the inverse
	for col := 0; col < cols; col++ {
		for row := rows - 1; row >= 0; row-- {
			sum := Y[row][col]
			for k := row + 1; k < rows; k++ {
				sum -= U[row][k] * dst[k][col]
			}
			dst[row][col] = sum / U[row][row]
		}
	}

	return nil
}

// Inverse calculates the inverse of a Matrix2x2 using direct formula.
func (m *Matrix2x2) Inverse(dst *Matrix2x2) error {
	det := m.Det()
	if math32.Abs(det) < SingularityTolerance {
		return ErrSingular
	}

	invDet := 1.0 / det
	dst[0][0] = m[1][1] * invDet
	dst[0][1] = -m[0][1] * invDet
	dst[1][0] = -m[1][0] * invDet
	dst[1][1] = m[0][0] * invDet

	return nil
}

// Inverse calculates the inverse of a Matrix3x3 using direct formula.
func (m *Matrix3x3) Inverse(dst *Matrix3x3) error {
	det := m.Det()
	if math32.Abs(det) < SingularityTolerance {
		return ErrSingular
	}

	invDet := 1.0 / det

	// Cofactor matrix (adjugate transpose)
	dst[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * invDet
	dst[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet
	dst[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet

	dst[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet
	dst[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet
	dst[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invDet

	dst[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * invDet
	dst[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invDet
	dst[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet

	return nil
}

// Inverse calculates the inverse of a Matrix4x4 using LU decomposition.
func (m *Matrix4x4) Inverse(dst *Matrix4x4) error {
	det := m.Det()
	if math32.Abs(det) < SingularityTolerance {
		return ErrSingular
	}

	// LU decomposition
	var L, U Matrix4x4
	m.LU(&L, &U)

	// Identity matrix
	I := Matrix4x4{}
	I.Eye()

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
				sum -= U[row][k] * dst[k][col]
			}
			dst[row][col] = sum / U[row][row]
		}
	}

	return nil
}
