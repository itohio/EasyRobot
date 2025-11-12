package mat

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

// Matrix layout is row-major.
type Matrix [][]float32

var _ matTypes.Matrix = Matrix(nil)

type vectorView interface {
	Vector() vecTypes.Vector
}

func ensureMatrix(arg matTypes.Matrix, op string) Matrix {
	if arg == nil {
		panic("mat.Matrix." + op + ": nil matrix")
	}
	if concrete, ok := arg.(Matrix); ok {
		return concrete
	}
	if view := arg.Matrix(); view != nil {
		if concrete, ok := view.(Matrix); ok {
			return concrete
		}
	}
	panic(fmt.Sprintf("mat.Matrix.%s: unsupported matrix type %T", op, arg))
}

func ensureVector(arg vecTypes.Vector, op string) vec.Vector {
	if arg == nil {
		panic("mat.Matrix." + op + ": nil vector")
	}
	if concrete, ok := arg.(vec.Vector); ok {
		return concrete
	}
	if view, ok := arg.(vectorView); ok {
		provided := view.Vector()
		if provided == nil {
			panic(fmt.Sprintf("mat.Matrix.%s: vector provider returned nil", op))
		}
		if concrete, ok := provided.(vec.Vector); ok {
			return concrete
		}
		return ensureVector(provided, op)
	}
	if clone := arg.Clone(); clone != nil {
		if concrete, ok := clone.(vec.Vector); ok {
			return concrete
		}
		if view, ok := clone.(vectorView); ok {
			provided := view.Vector()
			if provided == nil {
				panic(fmt.Sprintf("mat.Matrix.%s: cloned vector provider returned nil", op))
			}
			if concrete, ok := provided.(vec.Vector); ok {
				return concrete
			}
			return ensureVector(provided, op)
		}
	}
	panic(fmt.Sprintf("mat.Matrix.%s: unsupported vector type %T", op, arg))
}

func ensureQuaternion(arg vecTypes.Quaternion, op string) *vec.Quaternion {
	if arg == nil {
		panic("mat.Matrix." + op + ": nil quaternion")
	}
	if concrete, ok := arg.(*vec.Quaternion); ok {
		return concrete
	}
	panic(fmt.Sprintf("mat.Matrix.%s: unsupported quaternion type %T", op, arg))
}

func cloneFlat(flat []float32) []float32 {
	if flat == nil {
		return nil
	}
	dup := make([]float32, len(flat))
	copy(dup, flat)
	return dup
}

// New creates a new matrix. Matrix is backed by a flat array. If backing is provided, it must be of length rows*cols, otherwise new backing array is created.
// Matrix layout is row-major.
func New(rows, cols int, backing ...float32) Matrix {
	m := make([][]float32, rows)
	if len(backing) != cols*rows {
		backing = make([]float32, rows*cols)
	}
	s := 0
	for i := range m {
		m[i] = backing[s : s+cols]
		s += cols
	}
	return m
}

func (m Matrix) IsContiguous() bool {
	if len(m) <= 1 {
		return true
	}

	// Check if rows are adjacent in memory
	for i := 0; i < len(m)-1; i++ {
		if len(m[i]) == 0 || len(m[i+1]) == 0 {
			return false
		}

		// Check if end of row i is adjacent to start of row i+1
		ptr1 := uintptr(unsafe.Pointer(&m[i][len(m[i])-1]))
		ptr2 := uintptr(unsafe.Pointer(&m[i+1][0]))

		if ptr2-ptr1 != unsafe.Sizeof(float32(0)) {
			return false
		}
	}
	return true
}

// Flat returns flat representation of this matrix
func (m Matrix) Flat() []float32 {
	if len(m) == 0 {
		return nil
	}

	if len(m[0]) == 0 {
		return []float32{}
	}

	rows := len(m)
	cols := len(m[0])

	// Fast path: zero-copy if contiguous
	if m.IsContiguous() {
		return unsafe.Slice((*float32)(unsafe.Pointer(&m[0][0])), rows*cols)
	}

	// Slow path: copy if not contiguous
	result := make([]float32, rows*cols)
	idx := 0
	for i := range m {
		for j := range m[i] {
			result[idx] = m[i][j]
			idx++
		}
	}
	return result
}

// Returns a Matrix view of this matrix.
// The view actually contains slices of original matrix rows.
// This way original matrix can be modified.
func (m Matrix) Matrix() matTypes.Matrix {
	m1 := make(Matrix, len(m))
	for i := range m {
		m1[i] = m[i][:]
	}
	return m1
}

func (m Matrix) Rotation2D(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 2, 2,
		c, -s,
		s, c,
	)
}

func (m Matrix) RotationX(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		1, 0, 0,
		0, c, -s,
		0, s, c,
	)
}

func (m Matrix) RotationY(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, 0, s,
		0, 1, 0,
		-s, 0, c,
	)
}

func (m Matrix) RotationZ(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 3, 3,
		c, -s, 0,
		s, c, 0,
		0, 0, 1,
	)
}

func (m Matrix) Orientation(q vecTypes.Quaternion) matTypes.Matrix {
	quat := ensureQuaternion(q, "Orientation")
	theta := quat.Theta() / 2

	qr := math32.Cos(theta)
	s := math32.Sin(theta)
	qVal := *quat
	qi := qVal[0] * s
	qj := qVal[1] * s
	qk := qVal[2] * s

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

func (m Matrix) Eye() matTypes.Matrix {
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

func (m Matrix) Row(row int) vecTypes.Vector {
	return vec.Vector(m[row][:])
}

func (m Matrix) Col(col int, v vecTypes.Vector) vecTypes.Vector {
	dst := ensureVector(v, "Col")
	for i, row := range m {
		dst[i] = row[col]
	}
	return dst
}

func (m Matrix) SetRow(row int, v vecTypes.Vector) matTypes.Matrix {
	src := ensureVector(v, "SetRow")
	copy(m[row][:], src)
	return m
}

func (m Matrix) SetCol(col int, v vecTypes.Vector) matTypes.Matrix {
	src := ensureVector(v, "SetCol")
	for i, val := range src {
		m[i][col] = val
	}
	return m
}

func (m Matrix) SetColFromRow(col int, rowStart int, v vecTypes.Vector) matTypes.Matrix {
	src := ensureVector(v, "SetColFromRow")
	for i, val := range src {
		if rowStart+i < len(m) {
			m[rowStart+i][col] = val
		}
	}
	return m
}

func (m Matrix) GetCol(col int, dst vecTypes.Vector) vecTypes.Vector {
	out := ensureVector(dst, "GetCol")
	for i := range m {
		if i < len(out) {
			out[i] = m[i][col]
		}
	}
	return out
}

func (m Matrix) Diagonal(dst vecTypes.Vector) vecTypes.Vector {
	out := ensureVector(dst, "Diagonal")
	for i, row := range m {
		out[i] = row[i]
	}
	return out
}

func (m Matrix) SetDiagonal(v vecTypes.Vector) matTypes.Matrix {
	src := ensureVector(v, "SetDiagonal")
	for i, val := range src {
		m[i][i] = val
	}
	return m
}

// FromDiagonal creates a square diagonal matrix from diagonal values.
// Size is determined by the number of values provided.
// Returns a matrix with zeros everywhere except the diagonal.
func FromDiagonal(diagonal ...float32) Matrix {
	if len(diagonal) == 0 {
		return nil
	}
	size := len(diagonal)
	m := New(size, size)
	for i := range diagonal {
		m[i][i] = diagonal[i]
	}
	return m
}

// FromVector creates a square diagonal matrix from a vector.
// The vector elements become the diagonal elements of the matrix.
// Returns a matrix with zeros everywhere except the diagonal.
func FromVector(v vecTypes.Vector) Matrix {
	vecSrc := ensureVector(v, "FromVector")
	if len(vecSrc) == 0 {
		return nil
	}
	size := len(vecSrc)
	m := New(size, size)
	for i := range vecSrc {
		m[i][i] = vecSrc[i]
	}
	return m
}

func (m Matrix) Submatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	dst := ensureMatrix(m1, "Submatrix")
	cols := len(dst[0])
	for i := range dst {
		copy(dst[i], m[row+i][col:cols+col])
	}
	return dst
}

func (m Matrix) SetSubmatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	src := ensureMatrix(m1, "SetSubmatrix")
	for i := range m[row : row+len(src)] {
		copy(m[row+i][col:col+len(src[i])], src[i])
	}
	return m
}

func (m Matrix) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) matTypes.Matrix {
	for i := 0; i < rows1; i++ {
		copy(m[row+i][col:col+cols1], m1[i*cols1:i*cols1+cols1])
	}
	return m
}

func (m Matrix) Clone() matTypes.Matrix {
	if len(m) == 0 {
		return nil
	}
	m1 := New(len(m), len(m[0]))
	for i, row := range m {
		copy(m1[i], row)
	}
	return m1
}

func (m Matrix) Transpose(m1 matTypes.Matrix) matTypes.Matrix {
	src := ensureMatrix(m1, "Transpose")
	rows := len(src)
	cols := len(src[0])
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[j][i] = src[i][j]
		}
	}
	return m
}

func (m Matrix) Add(m1 matTypes.Matrix) matTypes.Matrix {
	rows := len(m)
	if rows == 0 {
		return m
	}
	cols := len(m[0])
	total := rows * cols
	mFlat := m.Flat()
	m1Flat := ensureMatrix(m1, "Add").Flat()
	fp32.ElemAdd(mFlat, mFlat, m1Flat, []int{total}, []int{1}, []int{1}, []int{1})
	return m
}

func (m Matrix) Sub(m1 matTypes.Matrix) matTypes.Matrix {
	rows := len(m)
	if rows == 0 {
		return m
	}
	cols := len(m[0])
	total := rows * cols
	mFlat := m.Flat()
	m1Flat := ensureMatrix(m1, "Sub").Flat()
	fp32.ElemSub(mFlat, mFlat, m1Flat, []int{total}, []int{1}, []int{1}, []int{1})
	return m
}

func (m Matrix) MulC(c float32) matTypes.Matrix {
	rows := len(m)
	if rows == 0 {
		return m
	}
	cols := len(m[0])
	total := rows * cols
	mFlat := m.Flat()
	fp32.Scal(mFlat, 1, total, c)
	return m
}

func (m Matrix) DivC(c float32) matTypes.Matrix {
	rows := len(m)
	if rows == 0 {
		return m
	}
	cols := len(m[0])
	total := rows * cols
	mFlat := m.Flat()
	fp32.Scal(mFlat, 1, total, 1.0/c)
	return m
}

func (m Matrix) Mul(a matTypes.Matrix, b matTypes.Matrix) matTypes.Matrix {
	aMat := ensureMatrix(a, "Mul.left")
	bMat := ensureMatrix(b, "Mul.right")
	M := len(aMat)
	if M == 0 {
		return m
	}
	N := len(bMat[0])
	K := len(bMat)
	if len(m) < M || len(m[0]) < N {
		panic("mat.Matrix.Mul: destination too small")
	}
	aFlat := aMat.Flat()
	bFlat := bMat.Flat()
	mFlat := m.Flat()
	ldC := N
	ldA := K
	ldB := N
	fp32.Gemm_NN(mFlat, aFlat, bFlat, ldC, ldA, ldB, M, N, K, 1.0, 0.0)
	return m
}

func (m Matrix) MulDiag(a matTypes.Matrix, b vecTypes.Vector) matTypes.Matrix {
	aMat := ensureMatrix(a, "MulDiag")
	vecB := ensureVector(b, "MulDiag")
	rows := len(aMat)
	cols := len(aMat[0])
	for i := 0; i < rows; i++ {
		fp32.ElemMul(m[i], aMat[i], vecB, []int{cols}, []int{1}, []int{1}, []int{1})
	}
	return m
}

func (m Matrix) MulVec(v vecTypes.Vector, dst vecTypes.Vector) vecTypes.Vector {
	rows := len(m)
	if rows == 0 {
		return dst
	}
	cols := len(m[0])
	matFlat := m.Flat()
	srcVec := ensureVector(v, "MulVec")
	dstVec := ensureVector(dst, "MulVec.dst")
	fp32.Gemv_N(dstVec, matFlat, srcVec, cols, rows, cols, 1.0, 0.0)
	return dstVec
}

func (m Matrix) MulVecT(v vecTypes.Vector, dst vecTypes.Vector) vecTypes.Vector {
	rows := len(m)
	if rows == 0 {
		return dst
	}
	cols := len(m[0])
	matFlat := m.Flat()
	srcVec := ensureVector(v, "MulVecT")
	dstVec := ensureVector(dst, "MulVecT.dst")
	fp32.Gemv_T(dstVec, matFlat, srcVec, cols, rows, cols, 1.0, 0.0)
	return dstVec
}

// Determinant only valid for square matrix
// Undefined behavior for non square matrices
func (m Matrix) Det() float32 {
	tmp := ensureMatrix(m.Clone(), "Det.clone")

	var ratio float32
	var det float32 = 1
	size := len(tmp)

	for i := 0; i < size; i++ {
		row := tmp[i][:]
		for j := i + 1; j < size; j++ {
			tmpj := tmp[j][:]
			ratio = tmpj[i] / row[i]
			fp32.Axpy(tmpj, row, 1, 1, size, -ratio)
		}
	}

	for i := range tmp {
		det *= tmp[i][i]
	}

	return det
}

// LU decomposition into two triangular matrices
// NOTE: Assume, that l&u matrices are set to zero
// Matrix must be square and M, L and U matrix sizes must be equal
func (m Matrix) LU(L, U matTypes.Matrix) {
	LMat := ensureMatrix(L, "LU.L")
	UMat := ensureMatrix(U, "LU.U")
	size := len(m)

	mFlat := m.Flat()
	LFlat := LMat.Flat()
	UFlat := UMat.Flat()
	ldA := len(m[0])
	ldL := len(LMat[0])
	ldU := len(UMat[0])

	ipiv := make([]int, size)
	if err := fp32.Getrf(mFlat, LFlat, UFlat, ipiv, ldA, ldL, ldU, size, size); err != nil {
		for i := 0; i < size; i++ {
			for k := i; k < size; k++ {
				var sum float32
				for j := 0; j < i; j++ {
					sum += LMat[i][j] * UMat[j][k]
				}
				UMat[i][k] = m[i][k] - sum
			}
			for k := i; k < size; k++ {
				if i == k {
					LMat[i][i] = 1
				} else {
					var sum float32
					for j := 0; j < i; j++ {
						sum += LMat[k][j] * UMat[j][i]
					}
					if UMat[i][i] != 0 {
						LMat[k][i] = (m[k][i] - sum) / UMat[i][i]
					} else {
						LMat[k][i] = 0
					}
				}
			}
		}
	}
}

// / https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
// / Must be at least 3x3 matrix
func (m Matrix) Quaternion() vecTypes.Quaternion {
	var t float32
	var q vec.Quaternion
	if m[2][2] < 0 {
		if m[0][0] > m[1][1] {
			t = 1 + m[0][0] - m[1][1] - m[2][2]
			q = vec.Quaternion{t, m[0][1] + m[1][0], m[2][0] + m[0][2], m[1][2] - m[2][1]}
		} else {
			t = 1 - m[0][0] + m[1][1] - m[2][2]
			q = vec.Quaternion{m[0][1] + m[1][0], t, m[1][2] + m[2][1], m[2][0] - m[0][2]}
		}
	} else {
		if m[0][0] < -m[1][1] {
			t = 1 - m[0][0] - m[1][1] + m[2][2]
			q = vec.Quaternion{m[2][0] + m[0][2], m[1][2] + m[2][1], t, m[0][1] - m[1][0]}
		} else {
			t = 1 + m[0][0] + m[1][1] + m[2][2]
			q = vec.Quaternion{m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0], t}
		}
	}
	vec.Vector(q[:]).MulC(0.5 / math32.Sqrt(t))
	return &q
}
