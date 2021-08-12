package mat

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type Matrix4x4 [16]float32

func New4x4From(arr ...float32) Matrix4x4 {
	if len(arr) != 4 {
		panic(-1)
	}
	m := Matrix4x4{}
	copy(m[:], arr)
	return m
}

func (m *Matrix4x4) RotationX(a float32) *Matrix4x4 {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		1, 0, 0,
		0, c, -s,
		0, s, c,
	)
}

func (m *Matrix4x4) RotationY(a float32) *Matrix4x4 {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, 0, s,
		0, 1, 0,
		-s, 0, c,
	)
}

func (m *Matrix4x4) RotationZ(a float32) *Matrix4x4 {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, -s, 0,
		s, c, 0,
		0, 0, 1,
	)
}

func (m *Matrix4x4) Rotation(x, y, z float32) *Matrix4x4 {
	m.RotationX(x)
	Y := m.Clone()
	Y.RotationY(y)
	Z := m.Clone()
	Z.RotationZ(z)
	return m.Mul(*Y).Mul(*Z)
}

/// Build orientation matrix from quaternion
/// NOTE: axis must be unit vector
func (m *Matrix4x4) Orientation(q vec.Quaternion) *Matrix4x4 {
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

func (m *Matrix4x4) Eye() *Matrix4x4 {
	idx := 0
	for i := range m {
		m[i] = 0
	}
	for i := 0; i < 4; i++ {
		m[idx] = 1
		idx += 4 + 1
	}
	return m
}

func (m *Matrix4x4) RowIdx(row int) int {
	return row * 4
}

func (m *Matrix4x4) Row(row int) vec.Vector {
	i := m.RowIdx(row)
	return m[i : i+4]
}

func (m *Matrix4x4) Col(col int) vec.Vector {
	v := vec.New(4)
	for i := range v {
		v[i] = m[col]
		col += 4
	}
	return v
}

func (m *Matrix4x4) SetRow(row int, v vec.Vector) *Matrix4x4 {
	i := m.RowIdx(row)
	copy(m[i:i+4], v)
	return m
}

func (m *Matrix4x4) SetCol(col int, v vec.Vector) *Matrix4x4 {
	for _, v := range v {
		m[col] = v
		col += 4
	}
	return m
}

func (m *Matrix4x4) Diagonal(v vec.Vector) vec.Vector {
	j := 0
	for i := range v {
		v[i] = m[j]
		j += 4 + 1
	}
	return v
}

func (m *Matrix4x4) SetDiagonal(v vec.Vector) *Matrix4x4 {
	j := 0
	for i := range v {
		m[j] = v[i]
		j += 4 + 1
	}
	return m
}

func (m *Matrix4x4) Submatrix(row, col int, m1 Matrix4x4) *Matrix4x4 {
	dst := 0
	for i := 0; i < 4; i++ {
		src := m.RowIdx(row+i) + col
		for j := 0; j < 4; j++ {
			m1[dst] = m[src]
			dst++
			src++
		}
	}
	return &m1
}

func (m *Matrix4x4) SetSubmatrix(row, col int, m1 Matrix4x4) *Matrix4x4 {
	src := 0
	for i := 0; i < 4; i++ {
		dst := m.RowIdx(row+i) + col
		for j := 0; j < 4; j++ {
			m[dst] = m1[src]
			dst++
			src++
		}
	}
	return m
}

func (m *Matrix4x4) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) *Matrix4x4 {
	src := 0
	for i := 0; i < rows1; i++ {
		dst := m.RowIdx(row+i) + col
		for j := 0; j < cols1; j++ {
			m[dst] = m1[src]
			dst++
			src++
		}
	}
	return m
}

func (m *Matrix4x4) Set(row, col int, val float32) {
	m[m.RowIdx(row)+col] = val
}

func (m *Matrix4x4) Get(row, col int) float32 {
	return m[m.RowIdx(row)+col]
}

func (m *Matrix4x4) Clone() *Matrix4x4 {
	m1 := Matrix4x4{}
	copy(m1[:], m[:])
	return &m1
}

func (m *Matrix4x4) Transpose(m1 Matrix4x4) *Matrix4x4 {
	if 4 != 4 || 4 != 4 {
		panic(-1)
	}
	src := 0
	for i := 0; i < 4; i++ {
		dst := i
		for j := 0; j < 4; j++ {
			m1[dst] = m[src]
			src++
			dst += 4
		}
	}
	return &m1
}

func (m *Matrix4x4) Add(m1 Matrix4x4) *Matrix4x4 {
	for i, v := range m1 {
		m[i] += v
	}
	return m
}

func (m *Matrix4x4) Sub(m1 Matrix4x4) *Matrix4x4 {
	for i, v := range m1 {
		m[i] -= v
	}
	return m
}

func (m *Matrix4x4) MulC(c float32) *Matrix4x4 {
	for i := range m {
		m[i] *= c
	}
	return m
}

func (m *Matrix4x4) DivC(c float32) *Matrix4x4 {
	for i := range m {
		m[i] *= c
	}
	return m
}

func (m *Matrix4x4) Mul(m1 Matrix4x4) *Matrix4x4 {
	tmp := m.Clone()
	return tmp.MulTo(m1, tmp)
}

func (m *Matrix4x4) MulTo(m1 Matrix4x4, dst *Matrix4x4) *Matrix4x4 {
	multo(2, 2, 2, 2, m[:], m1[:], dst[:])
	return dst
}

func (m *Matrix4x4) MulV(v vec.Vector4D) *vec.Vector4D {
	dst := vec.Vector4D{}
	return m.MulVTo(v, &dst)
}

func (m *Matrix4x4) MulVTo(v vec.Vector4D, dst *vec.Vector4D) *vec.Vector4D {
	mulvto(m[:], v[:], dst[:])
	return dst
}

func (m *Matrix4x4) MulVT(v vec.Vector4D) *vec.Vector4D {
	dst := vec.Vector4D{}
	return m.MulVTTo(v, &dst)
}

func (m *Matrix4x4) MulVTTo(v vec.Vector4D, dst *vec.Vector4D) *vec.Vector4D {
	mulvtto(m[:], v[:], dst[:])
	return dst
}

/// Determinant only valid for square matrix
/// Undefined behavior for non square matrices

///
/// LU decomposition into two triangular matrices
/// NOTE: Assume, that l&u matrices are set to zero

/// https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
/// Must be at least 3x3 matrix
func (m *Matrix4x4) Quaternion() (q *vec.Quaternion) {
	var t float32
	if m.Get(2, 2) < 0 {
		if m.Get(0, 0) > m.Get(1, 1) {
			t = 1 + m.Get(0, 0) - m.Get(1, 1) - m.Get(2, 2)
			q = &vec.Quaternion{t, m.Get(0, 1) + m.Get(1, 0), m.Get(2, 0) + m.Get(0, 2), m.Get(1, 2) - m.Get(2, 1)}
		} else {
			t = 1 - m.Get(0, 0) + m.Get(1, 1) - m.Get(2, 2)
			q = &vec.Quaternion{m.Get(0, 1) + m.Get(1, 0), t, m.Get(1, 2) + m.Get(2, 1), m.Get(2, 0) - m.Get(0, 2)}
		}
	} else {
		if m.Get(0, 0) < -m.Get(1, 1) {
			t = 1 - m.Get(0, 0) - m.Get(1, 1) + m.Get(2, 2)
			q = &vec.Quaternion{m.Get(2, 0) + m.Get(0, 2), m.Get(1, 2) + m.Get(2, 1), t, m.Get(0, 1) - m.Get(1, 0)}
		} else {
			t = 1 + m.Get(0, 0) + m.Get(1, 1) + m.Get(2, 2)
			q = &vec.Quaternion{m.Get(1, 2) - m.Get(2, 1), m.Get(2, 0) - m.Get(0, 2), m.Get(0, 1) - m.Get(1, 0), t}
		}
	}
	q.Vector().MulC(0.5 / math32.Sqrt(t))
	return
}
