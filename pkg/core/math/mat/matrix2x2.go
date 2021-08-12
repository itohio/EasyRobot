package mat

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type Matrix2x2 [4]float32

func New2x2From(arr ...float32) Matrix2x2 {
	if len(arr) != 4 {
		panic(-1)
	}
	m := Matrix2x2{}
	copy(m[:], arr)
	return m
}

func (m *Matrix2x2) Rotation2D(a float32) *Matrix2x2 {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 2, 2,
		c, -s,
		s, c,
	)
}

/// Build orientation matrix from quaternion
/// NOTE: axis must be unit vector

func (m *Matrix2x2) Eye() *Matrix2x2 {
	idx := 0
	for i := range m {
		m[i] = 0
	}
	for i := 0; i < 2; i++ {
		m[idx] = 1
		idx += 2 + 1
	}
	return m
}

func (m *Matrix2x2) RowIdx(row int) int {
	return row * 2
}

func (m *Matrix2x2) Row(row int) vec.Vector {
	i := m.RowIdx(row)
	return m[i : i+2]
}

func (m *Matrix2x2) Col(col int) vec.Vector {
	v := vec.New(2)
	for i := range v {
		v[i] = m[col]
		col += 2
	}
	return v
}

func (m *Matrix2x2) SetRow(row int, v vec.Vector) *Matrix2x2 {
	i := m.RowIdx(row)
	copy(m[i:i+2], v)
	return m
}

func (m *Matrix2x2) SetCol(col int, v vec.Vector) *Matrix2x2 {
	for _, v := range v {
		m[col] = v
		col += 2
	}
	return m
}

func (m *Matrix2x2) Diagonal(v vec.Vector) vec.Vector {
	j := 0
	for i := range v {
		v[i] = m[j]
		j += 2 + 1
	}
	return v
}

func (m *Matrix2x2) SetDiagonal(v vec.Vector) *Matrix2x2 {
	j := 0
	for i := range v {
		m[j] = v[i]
		j += 2 + 1
	}
	return m
}

func (m *Matrix2x2) Submatrix(row, col int, m1 Matrix2x2) *Matrix2x2 {
	dst := 0
	for i := 0; i < 2; i++ {
		src := m.RowIdx(row+i) + col
		for j := 0; j < 2; j++ {
			m1[dst] = m[src]
			dst++
			src++
		}
	}
	return &m1
}

func (m *Matrix2x2) SetSubmatrix(row, col int, m1 Matrix2x2) *Matrix2x2 {
	src := 0
	for i := 0; i < 2; i++ {
		dst := m.RowIdx(row+i) + col
		for j := 0; j < 2; j++ {
			m[dst] = m1[src]
			dst++
			src++
		}
	}
	return m
}

func (m *Matrix2x2) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) *Matrix2x2 {
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

func (m *Matrix2x2) Set(row, col int, val float32) {
	m[m.RowIdx(row)+col] = val
}

func (m *Matrix2x2) Get(row, col int) float32 {
	return m[m.RowIdx(row)+col]
}

func (m *Matrix2x2) Clone() *Matrix2x2 {
	m1 := Matrix2x2{}
	copy(m1[:], m[:])
	return &m1
}

func (m *Matrix2x2) Transpose(m1 Matrix2x2) *Matrix2x2 {
	src := 0
	for i := 0; i < 2; i++ {
		dst := i
		for j := 0; j < 2; j++ {
			m1[dst] = m[src]
			src++
			dst += 2
		}
	}
	return &m1
}

func (m *Matrix2x2) Add(m1 Matrix2x2) *Matrix2x2 {
	for i, v := range m1 {
		m[i] += v
	}
	return m
}

func (m *Matrix2x2) Sub(m1 Matrix2x2) *Matrix2x2 {
	for i, v := range m1 {
		m[i] -= v
	}
	return m
}

func (m *Matrix2x2) MulC(c float32) *Matrix2x2 {
	for i := range m {
		m[i] *= c
	}
	return m
}

func (m *Matrix2x2) DivC(c float32) *Matrix2x2 {
	for i := range m {
		m[i] *= c
	}
	return m
}

func (m Matrix2x2) Mul(m1 Matrix2x2) *Matrix2x2 {
	tmp := m.Clone()
	return tmp.MulTo(m1, tmp)
}

func (m *Matrix2x2) MulTo(m1 Matrix2x2, dst *Matrix2x2) *Matrix2x2 {
	multo(2, 2, 2, 2, m[:], m1[:], dst[:])
	return dst
}

func (m *Matrix2x2) MulV(v vec.Vector2D) *vec.Vector2D {
	dst := vec.Vector2D{}
	return m.MulVTo(v, &dst)
}

func (m *Matrix2x2) MulVTo(v vec.Vector2D, dst *vec.Vector2D) *vec.Vector2D {
	mulvto(m[:], v[:], dst[:])
	return dst
}

func (m *Matrix2x2) MulVT(v vec.Vector2D) *vec.Vector2D {
	dst := vec.Vector2D{}
	return m.MulVTTo(v, &dst)
}

func (m *Matrix2x2) MulVTTo(v vec.Vector2D, dst *vec.Vector2D) *vec.Vector2D {
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
