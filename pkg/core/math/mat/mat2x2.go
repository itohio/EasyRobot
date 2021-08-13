package mat

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type Matrix2x2 [2][2]float32

func New2x2(arr ...float32) Matrix2x2 {
	m := Matrix2x2{}
	if arr != nil {
		for i := range m {
			copy(m[i][:], arr[i*2 : i*2+2][:])
		}
	}
	return m
}

func (m *Matrix2x2) Flat(v vec.Vector) vec.Vector {
	N := len(m[0])
	for i, row := range m {
		copy(v[i*N:i*N+N], row[:])
	}
	return v
}

func (m *Matrix2x2) Matrix() Matrix {
	m1 := make(Matrix, len(m))
	for i := range m {
		m1[i] = m[i][:]
	}
	return m1
}

// Fills destination matrix with a 2D rotation
// Matrix size must be at least 2x2
func (m *Matrix2x2) Rotation2D(a float32) *Matrix2x2 {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 2, 2,
		c, -s,
		s, c,
	)
}

// Fills destination matrix with identity matrix
func (m *Matrix2x2) Eye() *Matrix2x2 {
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

func (m *Matrix2x2) Row(row int) vec.Vector {
	return m[row][:]
}

func (m *Matrix2x2) Col(col int, v vec.Vector) vec.Vector {
	for i, row := range m {
		v[i] = row[col]
	}
	return v
}

func (m *Matrix2x2) SetRow(row int, v vec.Vector) *Matrix2x2 {
	copy(m[row][:], v[:])
	return m
}

func (m *Matrix2x2) SetCol(col int, v vec.Vector) *Matrix2x2 {
	for i, v := range v {
		m[i][col] = v
	}
	return m
}

// Size of the destination vector must equal to number of rows
func (m *Matrix2x2) Diagonal(dst vec.Vector) vec.Vector {
	for i, row := range m {
		dst[i] = row[i]
	}
	return dst
}

func (m *Matrix2x2) SetDiagonal(v vec.Vector2D) *Matrix2x2 {
	for i, v := range v {
		m[i][i] = v
	}
	return m
}

func (m *Matrix2x2) Submatrix(row, col int, m1 Matrix) Matrix {
	cols := len(m1[0])
	for i, m1row := range m1 {
		copy(m1row, m[row+i][col : cols+col][:])
	}
	return m1
}

func (m *Matrix2x2) SetSubmatrix(row, col int, m1 Matrix) *Matrix2x2 {
	for i := range m[row : row+len(m1)] {
		copy(m[row+i][col : col+len(m1[i])][:], m1[i][:])
	}
	return m
}

func (m *Matrix2x2) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) *Matrix2x2 {
	for i := 0; i < rows1; i++ {
		copy(m[row+i][col : col+cols1][:], m1[i*cols1:i*cols1+cols1])
	}
	return m
}

func (m *Matrix2x2) Clone() *Matrix2x2 {

	m1 := &Matrix2x2{}

	for i, row := range m {
		copy(m1[i][:], row[:])
	}
	return m1
}

// Transposes matrix m1 and stores the result in the destination matrix
// destination matrix must be of appropriate size.
// NOTE: Does not support in place transpose
func (m *Matrix2x2) Transpose(m1 Matrix2x2) *Matrix2x2 {
	for i, row := range m1 {
		for j, val := range row {
			m[j][i] = val
		}
	}
	return m
}

func (m *Matrix2x2) Add(m1 Matrix2x2) *Matrix2x2 {
	for i := range m {
		vec.Vector(m[i][:]).Add(m1[i][:])
	}
	return m
}

func (m *Matrix2x2) Sub(m1 Matrix2x2) *Matrix2x2 {
	for i := range m {
		vec.Vector(m[i][:]).Sub(m1[i][:])
	}
	return m
}

func (m *Matrix2x2) MulC(c float32) *Matrix2x2 {
	for i := range m {
		vec.Vector(m[i][:]).MulC(c)
	}
	return m
}

func (m *Matrix2x2) DivC(c float32) *Matrix2x2 {
	for i := range m {
		vec.Vector(m[i][:]).DivC(c)
	}
	return m
}

// Destination matrix must be properly sized.
// given that a is MxN and b is NxK
// then destinatiom matrix must be MxK
func (m *Matrix2x2) Mul(a Matrix2x2, b Matrix2x2) *Matrix2x2 {
	for i, row := range a {
		mrow := m[i][:]
		for j := range mrow {
			var sum float32
			for k, brow := range b {
				sum += row[k] * brow[j]
			}
			mrow[j] = sum
		}
	}
	return m
}

// Only makes sense for square matrices.
// Vector size must be equal to number of rows/cols
func (m *Matrix2x2) MulDiag(a Matrix2x2, b vec.Vector2D) *Matrix2x2 {
	for i, row := range a {
		mrow := m[i][:]
		for j := range row {
			mrow[j] = row[j] * b[j]
		}
	}

	return m
}

// Vector must have a size equal to number of cols.
// Destination vector must have a size equal to number of rows.
func (m *Matrix2x2) MulVec(v vec.Vector2D, dst vec.Vector) vec.Vector {
	for i, row := range m {
		var sum float32
		for j, val := range row {
			sum += v[j] * val
		}
		dst[i] = sum
	}
	return dst
}

// Vector must have a size equal to number of rows.
// Destination vector must have a size equal to number of cols.
func (m *Matrix2x2) MulVecT(v vec.Vector2D, dst vec.Vector) vec.Vector {
	for i := range m[0] {
		var sum float32
		for j, val := range m {
			sum += v[j] * val[i]
		}
		dst[i] = sum
	}
	return dst
}

// Determinant only valid for square matrix
// Undefined behavior for non square matrices
func (m *Matrix2x2) Det() float32 {
	tmp := m.Clone()

	var ratio float32
	var det float32 = 1

	// upper triangular
	for i, row := range tmp {
		for j := range row {
			if j > i {
				tmpj := tmp[j][:]
				ratio = tmpj[i] / row[i]
				for k := range tmp {
					tmpj[k] -= ratio * row[k]
				}
			}
		}
	}

	for i, row := range tmp {
		det *= row[i]
	}

	return det
}

//
// LU decomposition into two triangular matrices
// NOTE: Assume, that l&u matrices are set to zero
// Matrix must be square and M, L and U matrix sizes must be equal
func (m *Matrix2x2) LU(L, U *Matrix2x2) {
	for i := range m {
		// Upper Triangular
		for k := i; k < len(m); k++ {
			// Summation of L(i, j) * U(j, k)
			var sum float32
			for j := 0; j < i; j++ {
				sum += L[i][j] * U[j][k]
			}

			// Evaluating U(i, k)
			U[i][k] = m[i][k] - sum
		}

		// Lower Triangular
		for k := i; k < len(m); k++ {
			if i == k {
				L[i][i] = 1 // Diagonal as 1
			} else {
				// Summation of L(k, j) * U(j, i)
				var sum float32
				for j := 0; j < i; j++ {
					sum += L[k][j] * U[j][i]
				}

				// Evaluating L(k, i)
				L[k][i] = (m[k][i] - sum) / U[i][i]
			}
		}
	}
}
