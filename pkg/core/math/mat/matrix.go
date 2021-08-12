package mat

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type Matrix struct {
	Rows int
	Cols int
	Data []float32
}

func New(rows, cols int) Matrix {
	return Matrix{
		Data: make([]float32, rows*cols),
		Rows: rows,
		Cols: cols,
	}
}

func NewFromRaw(rows, cols int, arr []float32) Matrix {
	if len(arr) != rows*cols {
		panic(-1)
	}
	return Matrix{
		Data: arr,
		Rows: rows,
		Cols: cols,
	}
}

func NewFrom(rows, cols int, arr ...float32) Matrix {
	if len(arr) != rows*cols {
		panic(-1)
	}
	return Matrix{
		Data: arr,
		Rows: rows,
		Cols: cols,
	}
}

func (m Matrix) Rotation2D(a float32) Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 2, 2,
		c, -s,
		s, c,
	)
}

func (m Matrix) RotationX(a float32) Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		1, 0, 0,
		0, c, -s,
		0, s, c,
	)
}

func (m Matrix) RotationY(a float32) Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, 0, s,
		0, 1, 0,
		-s, 0, c,
	)
}

func (m Matrix) RotationZ(a float32) Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, -s, 0,
		s, c, 0,
		0, 0, 1,
	)
}

func (m Matrix) Rotation(x, y, z float32) Matrix {
	m.RotationX(x)
	Y := m.Clone()
	Y.RotationY(y)
	Z := m.Clone()
	Z.RotationZ(z)
	return m.Mul(Y).Mul(Z)
}

/// Build orientation matrix from quaternion
/// NOTE: axis must be unit vector
func (m Matrix) Orientation(q vec.Quaternion) Matrix {
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

func (m Matrix) Eye() Matrix {
	idx := 0
	for i := range m.Data {
		m.Data[i] = 0
	}
	for i := 0; i < m.Rows; i++ {
		m.Data[idx] = 1
		idx += m.Cols + 1
	}
	return m
}

func (m Matrix) RowIdx(row int) int {
	return row * m.Cols
}

func (m Matrix) Row(row int) vec.Vector {
	i := m.RowIdx(row)
	return m.Data[i : i+m.Cols]
}

func (m Matrix) Col(col int) vec.Vector {
	v := vec.New(m.Rows)
	for i := range v {
		v[i] = m.Data[col]
		col += m.Cols
	}
	return v
}

func (m Matrix) SetRow(row int, v vec.Vector) Matrix {
	i := m.RowIdx(row)
	copy(m.Data[i:i+m.Cols], v)
	return m
}

func (m Matrix) SetCol(col int, v vec.Vector) Matrix {
	for _, v := range v {
		m.Data[col] = v
		col += m.Cols
	}
	return m
}

func (m Matrix) Diagonal(v vec.Vector) vec.Vector {
	j := 0
	for i := range v {
		v[i] = m.Data[j]
		j += m.Cols + 1
	}
	return v
}

func (m Matrix) SetDiagonal(v vec.Vector) Matrix {
	j := 0
	for i := range v {
		m.Data[j] = v[i]
		j += m.Cols + 1
	}
	return m
}

func (m Matrix) Submatrix(row, col int, m1 Matrix) Matrix {
	dst := 0
	for i := 0; i < m1.Rows; i++ {
		src := m.RowIdx(row+i) + col
		for j := 0; j < m1.Cols; j++ {
			m1.Data[dst] = m.Data[src]
			dst++
			src++
		}
	}
	return m1
}

func (m Matrix) SetSubmatrix(row, col int, m1 Matrix) Matrix {
	src := 0
	for i := 0; i < m1.Rows; i++ {
		dst := m.RowIdx(row+i) + col
		for j := 0; j < m1.Cols; j++ {
			m.Data[dst] = m1.Data[src]
			dst++
			src++
		}
	}
	return m
}

func (m Matrix) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) Matrix {
	src := 0
	for i := 0; i < rows1; i++ {
		dst := m.RowIdx(row+i) + col
		for j := 0; j < cols1; j++ {
			m.Data[dst] = m1[src]
			dst++
			src++
		}
	}
	return m
}

func (m Matrix) Set(row, col int, val float32) {
	m.Data[m.RowIdx(row)+col] = val
}

func (m Matrix) Get(row, col int) float32 {
	return m.Data[m.RowIdx(row)+col]
}

func (m Matrix) Clone() Matrix {
	m1 := New(m.Rows, m.Cols)
	copy(m1.Data, m.Data)
	return m1
}

func (m Matrix) Transpose(m1 Matrix) Matrix {
	src := 0
	for i := 0; i < m.Rows; i++ {
		dst := i
		for j := 0; j < m.Cols; j++ {
			m1.Data[dst] = m.Data[src]
			src++
			dst += m1.Cols
		}
	}
	return m1
}

func (m Matrix) Add(m1 Matrix) Matrix {
	for i, v := range m1.Data {
		m.Data[i] += v
	}
	return m
}

func (m Matrix) Sub(m1 Matrix) Matrix {
	for i, v := range m1.Data {
		m.Data[i] -= v
	}
	return m
}

func (m Matrix) MulC(c float32) Matrix {
	for i := range m.Data {
		m.Data[i] *= c
	}
	return m
}

func (m Matrix) DivC(c float32) Matrix {
	for i := range m.Data {
		m.Data[i] *= c
	}
	return m
}

func (m Matrix) Mul(m1 Matrix) Matrix {
	tmp := m.Clone()
	return tmp.MulTo(m1, tmp)
}

func (m Matrix) MulTo(m1 Matrix, dst Matrix) Matrix {
	multo(m.Rows, m.Cols, m1.Rows, m1.Cols, m.Data, m1.Data, dst.Data)
	return dst
}

func (m Matrix) MulDiag(v vec.Vector) Matrix {
	dst := New(len(v), len(v))
	return m.MulDiagTo(v, dst)
}

func (m Matrix) MulDiagTo(v vec.Vector, dst Matrix) Matrix {
	muldiagto(m.Rows, m.Cols, v, m.Data, dst.Data)
	return dst
}

func (m Matrix) MulV(v vec.Vector) vec.Vector {
	dst := vec.New(m.Rows)
	return m.MulVTo(v, dst)
}

func (m Matrix) MulVTo(v vec.Vector, dst vec.Vector) vec.Vector {
	mulvto(m.Data, v, dst)
	return dst
}

func (m Matrix) MulVT(v vec.Vector) vec.Vector {
	dst := vec.New(m.Cols)
	return m.MulVTTo(v, dst)
}

func (m Matrix) MulVTTo(v vec.Vector, dst vec.Vector) vec.Vector {
	mulvtto(m.Data, v, dst)
	return dst
}

/// Determinant only valid for square matrix
/// Undefined behavior for non square matrices
func (m Matrix) Det() float32 {
	tmp := m.Clone()

	var ratio float32
	var det float32 = 1

	// upper triangular
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Rows; j++ {
			if j > i {
				ratio = tmp.Get(j, i) / tmp.Get(i, i)
				for k := 0; k < m.Rows; k++ {
					tmp.Set(j, k, tmp.Get(j, k)-ratio*tmp.Get(i, k))
				}
			}
		}
	}

	pd := 0
	for i := 0; i < m.Rows; i++ {
		det *= tmp.Data[pd]
		pd += m.Cols + 1
	}

	return det
}

///
/// LU decomposition into two triangular matrices
/// NOTE: Assume, that l&u matrices are set to zero
func (m Matrix) LU(L, U Matrix) {
	if m.Cols != m.Rows {
		panic("The matrix must be square")
	}
	if m.Cols != L.Cols || m.Cols != U.Cols {
		panic("Invalid L/U matrices")
	}
	for i := 0; i < m.Rows; i++ {
		// Upper Triangular
		for k := i; k < m.Rows; k++ {
			// Summation of L(i, j) * U(j, k)
			var sum float32 = 0
			for j := 0; j < i; j++ {
				sum += L.Get(i, j) * U.Get(j, k)
			}

			// Evaluating U(i, k)
			U.Set(i, k, m.Get(i, k)-sum)
		}

		// Lower Triangular
		for k := i; k < m.Rows; k++ {
			if i == k {
				L.Set(i, i, 1) // Diagonal as 1
			} else {
				// Summation of L(k, j) * U(j, i)
				var sum float32 = 0
				for j := 0; j < i; j++ {
					sum += L.Get(k, j) * U.Get(j, i)
				}

				// Evaluating L(k, i)
				L.Set(k, i, (m.Get(k, i)-sum)/U.Get(i, i))
			}
		}
	}
}

/// https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
/// Must be at least 3x3 matrix
func (m Matrix) Quaternion() (q *vec.Quaternion) {
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
