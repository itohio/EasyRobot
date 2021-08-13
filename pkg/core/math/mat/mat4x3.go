// Generated code. DO NOT EDIT

package mat

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type Matrix4x3 [4][3]float32

func New4x3(arr ...float32) Matrix4x3 {
	m := Matrix4x3{}
	if arr != nil {
		for i := range m {
			copy(m[i][:], arr[i*3 : i*3+3][:])
		}
	}
	return m
}

// Returns a flat representation of this matrix.
func (m *Matrix4x3) Flat(v vec.Vector) vec.Vector {
	N := len(m[0])
	for i, row := range m {
		copy(v[i*N:i*N+N], row[:])
	}
	return v
}

// Returns a Matrix view of this matrix.
// The view actually contains slices of original matrix rows.
// This way original matrix can be modified.
func (m *Matrix4x3) Matrix() Matrix {
	m1 := make(Matrix, len(m))
	for i := range m {
		m1[i] = m[i][:]
	}
	return m1
}

// Fills destination matrix with a rotation around X axis
// Matrix size must be at least 3x3
func (m *Matrix4x3) RotationX(a float32) *Matrix4x3 {
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
func (m *Matrix4x3) RotationY(a float32) *Matrix4x3 {
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
func (m *Matrix4x3) RotationZ(a float32) *Matrix4x3 {
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
func (m *Matrix4x3) Orientation(q vec.Quaternion) *Matrix4x3 {
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

// Returns a slice to the row.
func (m *Matrix4x3) Row(row int) vec.Vector {
	return m[row][:]
}

// Returns a copy of the matrix column.
func (m *Matrix4x3) Col(col int, v vec.Vector) vec.Vector {
	for i, row := range m {
		v[i] = row[col]
	}
	return v
}

func (m *Matrix4x3) SetRow(row int, v vec.Vector) *Matrix4x3 {
	copy(m[row][:], v[:])
	return m
}

func (m *Matrix4x3) SetCol(col int, v vec.Vector) *Matrix4x3 {
	for i, v := range v {
		m[i][col] = v
	}
	return m
}

func (m *Matrix4x3) Submatrix(row, col int, m1 Matrix) Matrix {
	cols := len(m1[0])
	for i, m1row := range m1 {
		copy(m1row, m[row+i][col : cols+col][:])
	}
	return m1
}

func (m *Matrix4x3) SetSubmatrix(row, col int, m1 Matrix) *Matrix4x3 {
	for i := range m[row : row+len(m1)] {
		copy(m[row+i][col : col+len(m1[i])][:], m1[i][:])
	}
	return m
}

func (m *Matrix4x3) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) *Matrix4x3 {
	for i := 0; i < rows1; i++ {
		copy(m[row+i][col : col+cols1][:], m1[i*cols1:i*cols1+cols1])
	}
	return m
}

func (m *Matrix4x3) Clone() *Matrix4x3 {

	m1 := &Matrix4x3{}

	for i, row := range m {
		copy(m1[i][:], row[:])
	}
	return m1
}

// Transposes matrix m1 and stores the result in the destination matrix
// destination matrix must be of appropriate size.
// NOTE: Does not support in place transpose
func (m *Matrix4x3) Transpose(m1 Matrix3x4) *Matrix4x3 {
	for i, row := range m1 {
		for j, val := range row {
			m[j][i] = val
		}
	}
	return m
}

func (m *Matrix4x3) Add(m1 Matrix4x3) *Matrix4x3 {
	for i := range m {
		vec.Vector(m[i][:]).Add(m1[i][:])
	}
	return m
}

func (m *Matrix4x3) Sub(m1 Matrix4x3) *Matrix4x3 {
	for i := range m {
		vec.Vector(m[i][:]).Sub(m1[i][:])
	}
	return m
}

func (m *Matrix4x3) MulC(c float32) *Matrix4x3 {
	for i := range m {
		vec.Vector(m[i][:]).MulC(c)
	}
	return m
}

func (m *Matrix4x3) DivC(c float32) *Matrix4x3 {
	for i := range m {
		vec.Vector(m[i][:]).DivC(c)
	}
	return m
}

// Destination matrix must be properly sized.
// given that a is MxN and b is NxK
// then destinatiom matrix must be MxK
func (m *Matrix4x3) Mul(a Matrix, b Matrix3x4) *Matrix4x3 {
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

// Vector must have a size equal to number of cols.
// Destination vector must have a size equal to number of rows.
func (m *Matrix4x3) MulVec(v vec.Vector3D, dst vec.Vector) vec.Vector {
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
func (m *Matrix4x3) MulVecT(v vec.Vector4D, dst vec.Vector) vec.Vector {
	for i := range m[0] {
		var sum float32
		for j, val := range m {
			sum += v[j] * val[i]
		}
		dst[i] = sum
	}
	return dst
}

/// https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
/// Must be at least 3x3 matrix
func (m *Matrix4x3) Quaternion() (q *vec.Quaternion) {
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
