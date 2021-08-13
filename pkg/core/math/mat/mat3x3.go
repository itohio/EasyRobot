// Generated code. DO NOT EDIT

package mat

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type Matrix3x3 [3][3]float32

func New3x3(arr ...float32) Matrix3x3 {
	m := Matrix3x3{}
	if arr != nil {
		for i := range m {
			copy(m[i][:], arr[i*3 : i*3+3][:])
		}
	}
	return m
}

// Returns a flat representation of this matrix.
func (m *Matrix3x3) Flat(v vec.Vector) vec.Vector {
	N := len(m[0])
	for i, row := range m {
		copy(v[i*N:i*N+N], row[:])
	}
	return v
}

// Returns a Matrix view of this matrix.
// The view actually contains slices of original matrix rows.
// This way original matrix can be modified.
func (m *Matrix3x3) Matrix() Matrix {
	m1 := make(Matrix, len(m))
	for i := range m {
		m1[i] = m[i][:]
	}
	return m1
}

// Fills destination matrix with a rotation around X axis
// Matrix size must be at least 3x3
func (m *Matrix3x3) RotationX(a float32) *Matrix3x3 {
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
func (m *Matrix3x3) RotationY(a float32) *Matrix3x3 {
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
func (m *Matrix3x3) RotationZ(a float32) *Matrix3x3 {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, -s, 0,
		s, c, 0,
		0, 0, 1,
	)
}

// Build orientation matrix from quaternion
// Matrix size must be at least 3x3
// Quaternion axis must be unit vector
func (m *Matrix3x3) Orientation(q vec.Quaternion) *Matrix3x3 {
	theta := q.Theta() / 2

	qr := math32.Cos(theta)
	s := math32.Sin(theta)
	qi := q[0] * s
	qj := q[1] * s
	qk := q[2] * s

	// calculate quaternion rotation matrix
	qjqj := qj * qj
	qiqi := qi * qi
	qkqk := qk * qk
	qiqj := qi * qj
	qjqr := qj * qr
	qiqk := qi * qk
	qiqr := qi * qr
	qkqr := qk * qr
	qjqk := qj * qk
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		1.0-2.0*(qjqj+qkqk),
		2.0*(qiqj+qkqr),
		2.0*(qiqk+qjqr),
		2.0*(qiqj+qkqr),
		1.0-2.0*(qiqi+qkqk),
		2.0*(qjqk+qiqr),
		2.0*(qiqk+qjqr),
		2.0*(qjqk+qiqr),
		1.0-2.0*(qiqi+qjqj),
	)
}

// Fills destination matrix with identity matrix.
func (m *Matrix3x3) Eye() *Matrix3x3 {
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
func (m *Matrix3x3) Row(row int) vec.Vector {
	return m[row][:]
}

// Returns a copy of the matrix column.
func (m *Matrix3x3) Col(col int, v vec.Vector) vec.Vector {
	for i, row := range m {
		v[i] = row[col]
	}
	return v
}

func (m *Matrix3x3) SetRow(row int, v vec.Vector) *Matrix3x3 {
	copy(m[row][:], v[:])
	return m
}

func (m *Matrix3x3) SetCol(col int, v vec.Vector) *Matrix3x3 {
	for i, v := range v {
		m[i][col] = v
	}
	return m
}

// Size of the destination vector must equal to number of rows
func (m *Matrix3x3) Diagonal(dst vec.Vector) vec.Vector {
	for i, row := range m {
		dst[i] = row[i]
	}
	return dst
}

// Size of the vector must equal to number of rows
func (m *Matrix3x3) SetDiagonal(v vec.Vector3D) *Matrix3x3 {
	for i, v := range v {
		m[i][i] = v
	}
	return m
}

func (m *Matrix3x3) Submatrix(row, col int, m1 Matrix) Matrix {
	cols := len(m1[0])
	for i, m1row := range m1 {
		copy(m1row, m[row+i][col : cols+col][:])
	}
	return m1
}

func (m *Matrix3x3) SetSubmatrix(row, col int, m1 Matrix) *Matrix3x3 {
	for i := range m[row : row+len(m1)] {
		copy(m[row+i][col : col+len(m1[i])][:], m1[i][:])
	}
	return m
}

func (m *Matrix3x3) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) *Matrix3x3 {
	for i := 0; i < rows1; i++ {
		copy(m[row+i][col : col+cols1][:], m1[i*cols1:i*cols1+cols1])
	}
	return m
}

func (m *Matrix3x3) Clone() *Matrix3x3 {

	m1 := &Matrix3x3{}

	for i, row := range m {
		copy(m1[i][:], row[:])
	}
	return m1
}

// Transposes matrix m1 and stores the result in the destination matrix
// destination matrix must be of appropriate size.
// NOTE: Does not support in place transpose
func (m *Matrix3x3) Transpose(m1 Matrix3x3) *Matrix3x3 {
	for i, row := range m1 {
		for j, val := range row {
			m[j][i] = val
		}
	}
	return m
}

func (m *Matrix3x3) Add(m1 Matrix3x3) *Matrix3x3 {
	for i := range m {
		vec.Vector(m[i][:]).Add(m1[i][:])
	}
	return m
}

func (m *Matrix3x3) Sub(m1 Matrix3x3) *Matrix3x3 {
	for i := range m {
		vec.Vector(m[i][:]).Sub(m1[i][:])
	}
	return m
}

func (m *Matrix3x3) MulC(c float32) *Matrix3x3 {
	for i := range m {
		vec.Vector(m[i][:]).MulC(c)
	}
	return m
}

func (m *Matrix3x3) DivC(c float32) *Matrix3x3 {
	for i := range m {
		vec.Vector(m[i][:]).DivC(c)
	}
	return m
}

// Destination matrix must be properly sized.
// given that a is MxN and b is NxK
// then destinatiom matrix must be MxK
func (m *Matrix3x3) Mul(a Matrix3x3, b Matrix3x3) *Matrix3x3 {
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
func (m *Matrix3x3) MulDiag(a Matrix3x3, b vec.Vector3D) *Matrix3x3 {
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
func (m *Matrix3x3) MulVec(v vec.Vector3D, dst vec.Vector) vec.Vector {
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
func (m *Matrix3x3) MulVecT(v vec.Vector3D, dst vec.Vector) vec.Vector {
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
func (m *Matrix3x3) Det() float32 {
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
func (m *Matrix3x3) LU(L, U *Matrix3x3) {
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

/// https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
/// Must be at least 3x3 matrix
func (m *Matrix3x3) Quaternion() (q *vec.Quaternion) {
	var t float32
	if m[2][2] < 0 {
		if m[0][0] > m[1][1] {
			t = 1 + m[0][0] - m[1][1] - m[2][2]
			q = &vec.Quaternion{t, m[0][1] + m[1][0], m[2][0] + m[0][2], m[1][2] - m[2][1]}
		} else {
			t = 1 - m[0][0] + m[1][1] - m[2][2]
			q = &vec.Quaternion{m[0][1] + m[1][0], t, m[1][2] + m[2][1], m[2][0] - m[0][2]}
		}
	} else {
		if m[0][0] < -m[1][1] {
			t = 1 - m[0][0] - m[1][1] + m[2][2]
			q = &vec.Quaternion{m[2][0] + m[0][2], m[1][2] + m[2][1], t, m[0][1] - m[1][0]}
		} else {
			t = 1 + m[0][0] + m[1][1] + m[2][2]
			q = &vec.Quaternion{m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0], t}
		}
	}
	q.Vector().MulC(0.5 / math32.Sqrt(t))
	return
}
