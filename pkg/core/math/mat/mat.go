package mat

import (
	"unsafe"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// Matrix layout is row-major.
type Matrix [][]float32

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
func (m Matrix) Matrix() Matrix {
	m1 := make(Matrix, len(m))
	for i := range m {
		m1[i] = m[i][:]
	}
	return m1
}

// Fills destination matrix with a 2D rotation
// Matrix size must be at least 2x2
func (m Matrix) Rotation2D(a float32) Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return m.SetSubmatrixRaw(0, 0, 2, 2,
		c, -s,
		s, c,
	)
}

// Fills destination matrix with a rotation around X axis
// Matrix size must be at least 3x3
func (m Matrix) RotationX(a float32) Matrix {
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
func (m Matrix) RotationY(a float32) Matrix {
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
func (m Matrix) RotationZ(a float32) Matrix {
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

// Fills destination matrix with identity matrix.
func (m Matrix) Eye() Matrix {
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
func (m Matrix) Row(row int) vec.Vector {
	return m[row][:]
}

// Returns a copy of the matrix column.
func (m Matrix) Col(col int, v vec.Vector) vec.Vector {
	for i, row := range m {
		v[i] = row[col]
	}
	return v
}

func (m Matrix) SetRow(row int, v vec.Vector) Matrix {
	copy(m[row][:], v[:])
	return m
}

func (m Matrix) SetCol(col int, v vec.Vector) Matrix {
	for i, v := range v {
		m[i][col] = v
	}
	return m
}

// SetColFromRow sets a partial column starting at rowStart.
// Copies elements from v into column col starting at rowStart.
// This is needed for IK solvers that build Jacobian matrices.
func (m Matrix) SetColFromRow(col int, rowStart int, v vec.Vector) Matrix {
	for i, val := range v {
		if rowStart+i < len(m) {
			m[rowStart+i][col] = val
		}
	}
	return m
}

// GetCol extracts a column from the matrix as a vector.
// dst must be at least as long as the number of rows.
func (m Matrix) GetCol(col int, dst vec.Vector) vec.Vector {
	for i := range m {
		if i < len(dst) {
			dst[i] = m[i][col]
		}
	}
	return dst
}

// Size of the destination vector must equal to number of rows
func (m Matrix) Diagonal(dst vec.Vector) vec.Vector {
	for i, row := range m {
		dst[i] = row[i]
	}
	return dst
}

// Size of the vector must equal to number of rows
func (m Matrix) SetDiagonal(v vec.Vector) Matrix {
	for i, v := range v {
		m[i][i] = v
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
func FromVector(v vec.Vector) Matrix {
	if len(v) == 0 {
		return nil
	}
	size := len(v)
	m := New(size, size)
	for i := range v {
		m[i][i] = v[i]
	}
	return m
}

func (m Matrix) Submatrix(row, col int, m1 Matrix) Matrix {
	cols := len(m1[0])
	for i, m1row := range m1 {
		copy(m1row, m[row+i][col : cols+col][:])
	}
	return m1
}

func (m Matrix) SetSubmatrix(row, col int, m1 Matrix) Matrix {
	for i := range m[row : row+len(m1)] {
		copy(m[row+i][col : col+len(m1[i])][:], m1[i][:])
	}
	return m
}

func (m Matrix) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) Matrix {
	for i := 0; i < rows1; i++ {
		copy(m[row+i][col : col+cols1][:], m1[i*cols1:i*cols1+cols1])
	}
	return m
}

func (m Matrix) Clone() Matrix {

	m1 := New(len(m), len(m[0]))

	for i, row := range m {
		copy(m1[i][:], row[:])
	}
	return m1
}

// Transposes matrix m1 and stores the result in the destination matrix
// destination matrix must be of appropriate size.
// NOTE: Does not support in place transpose
func (m Matrix) Transpose(m1 Matrix) Matrix {
	rows := len(m1)
	cols := len(m1[0])

	// Direct transpose using loops (no BLAS equivalent for pure transpose)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[j][i] = m1[i][j]
		}
	}

	return m
}

func (m Matrix) Add(m1 Matrix) Matrix {
	rows := len(m)
	if rows == 0 {
		return m
	}
	cols := len(m[0])
	total := rows * cols

	// Flatten both matrices (zero-copy if contiguous)
	mFlat := m.Flat()
	m1Flat := m1.Flat()

	// Use ElemAdd for element-wise addition
	fp32.ElemAdd(mFlat, mFlat, m1Flat, []int{total}, []int{1}, []int{1}, []int{1})

	return m
}

func (m Matrix) Sub(m1 Matrix) Matrix {
	rows := len(m)
	if rows == 0 {
		return m
	}
	cols := len(m[0])
	total := rows * cols

	// Flatten both matrices (zero-copy if contiguous)
	mFlat := m.Flat()
	m1Flat := m1.Flat()

	// Use ElemSub for element-wise subtraction
	fp32.ElemSub(mFlat, mFlat, m1Flat, []int{total}, []int{1}, []int{1}, []int{1})

	return m
}

func (m Matrix) MulC(c float32) Matrix {
	rows := len(m)
	if rows == 0 {
		return m
	}
	cols := len(m[0])
	total := rows * cols

	// Flatten matrix (zero-copy if contiguous)
	mFlat := m.Flat()

	// Use Scal for scalar multiplication
	fp32.Scal(mFlat, 1, total, c)

	return m
}

func (m Matrix) DivC(c float32) Matrix {
	rows := len(m)
	if rows == 0 {
		return m
	}
	cols := len(m[0])
	total := rows * cols

	// Flatten matrix (zero-copy if contiguous)
	mFlat := m.Flat()

	// Use Scal for scalar division (multiply by 1/c)
	fp32.Scal(mFlat, 1, total, 1.0/c)

	return m
}

// Destination matrix must be properly sized.
// given that a is MxK and b is KxN
// then destination matrix must be MxN
func (m Matrix) Mul(a Matrix, b Matrix) Matrix {
	M := len(a)
	if M == 0 {
		return m
	}
	N := len(b[0])
	K := len(b)

	// Validate destination matrix size
	if len(m) < M || len(m[0]) < N {
		// Matrix too small - this should not happen in normal usage
		// but we need to ensure we don't write beyond bounds
		return m
	}

	// Flatten matrices (zero-copy if contiguous)
	aFlat := a.Flat()
	bFlat := b.Flat()
	mFlat := m.Flat()

	// Use BLAS Level 3 Gemm_NN: m = 1.0 * a * b + 0.0 * m
	// A: M×K (ldA = K), B: K×N (ldB = N), C: M×N (ldC = N)
	// Leading dimensions: number of columns for each matrix
	// Note: ldC = N (result columns), not len(m[0]) - this assumes m is exactly M×N
	ldC := N
	ldA := K
	ldB := N
	fp32.Gemm_NN(mFlat, aFlat, bFlat, ldC, ldA, ldB, M, N, K, 1.0, 0.0)

	return m
}

// Only makes sense for square matrices.
// Vector size must be equal to number of rows/cols
func (m Matrix) MulDiag(a Matrix, b vec.Vector) Matrix {
	rows := len(a)
	cols := len(a[0])

	// Use Hadamard product for element-wise multiplication
	// For each row, multiply by corresponding b element
	for i := 0; i < rows; i++ {
		fp32.ElemMul(m[i], a[i], b, []int{cols}, []int{1}, []int{1}, []int{1})
	}

	return m
}

// Vector must have a size equal to number of cols.
// Destination vector must have a size equal to number of rows.
func (m Matrix) MulVec(v vec.Vector, dst vec.Vector) vec.Vector {
	rows := len(m)
	if rows == 0 {
		return dst
	}
	cols := len(m[0])

	// Flatten matrix (zero-copy if contiguous)
	matFlat := m.Flat()

	// Use BLAS Level 2 Gemv_N: dst = 1.0 * mat * v + 0.0 * dst
	fp32.Gemv_N(dst, matFlat, v, cols, rows, cols, 1.0, 0.0)

	return dst
}

// Vector must have a size equal to number of rows.
// Destination vector must have a size equal to number of cols.
func (m Matrix) MulVecT(v vec.Vector, dst vec.Vector) vec.Vector {
	rows := len(m)
	if rows == 0 {
		return dst
	}
	cols := len(m[0])

	// Flatten matrix (zero-copy if contiguous)
	matFlat := m.Flat()

	// Use BLAS Level 2 Gemv_T: dst = 1.0 * mat^T * v + 0.0 * dst
	fp32.Gemv_T(dst, matFlat, v, cols, rows, cols, 1.0, 0.0)

	return dst
}

// Determinant only valid for square matrix
// Undefined behavior for non square matrices
func (m Matrix) Det() float32 {
	tmp := m.Clone()

	var ratio float32
	var det float32 = 1
	size := len(tmp)

	// Upper triangular form using Gaussian elimination
	for i := 0; i < size; i++ {
		row := tmp[i][:]
		for j := i + 1; j < size; j++ {
			tmpj := tmp[j][:]
			ratio = tmpj[i] / row[i]
			// Use Axpy for row operation: tmpj = tmpj - ratio * row
			// Axpy computes: y = alpha*x + y, so we use -ratio to get subtraction
			fp32.Axpy(tmpj, row, 1, 1, size, -ratio)
		}
	}

	// Compute determinant as product of diagonal elements
	for i := range tmp {
		det *= tmp[i][i]
	}

	return det
}

// LU decomposition into two triangular matrices
// NOTE: Assume, that l&u matrices are set to zero
// Matrix must be square and M, L and U matrix sizes must be equal
func (m Matrix) LU(L, U Matrix) {
	size := len(m)

	// Flatten matrices (zero-copy if contiguous)
	mFlat := m.Flat()
	LFlat := L.Flat()
	UFlat := U.Flat()
	ldA := len(m[0])
	ldL := len(L[0])
	ldU := len(U[0])

	// Use Getrf for LU decomposition
	ipiv := make([]int, size)
	if err := fp32.Getrf(mFlat, LFlat, UFlat, ipiv, ldA, ldL, ldU, size, size); err != nil {
		// Fall back to manual implementation if Getrf fails
		// This maintains compatibility
		for i := 0; i < size; i++ {
			// Upper Triangular
			for k := i; k < size; k++ {
				var sum float32
				for j := 0; j < i; j++ {
					sum += L[i][j] * U[j][k]
				}
				U[i][k] = m[i][k] - sum
			}

			// Lower Triangular
			for k := i; k < size; k++ {
				if i == k {
					L[i][i] = 1 // Diagonal as 1
				} else {
					var sum float32
					for j := 0; j < i; j++ {
						sum += L[k][j] * U[j][i]
					}
					if i < len(U) && i < len(U[i]) && U[i][i] != 0 {
						L[k][i] = (m[k][i] - sum) / U[i][i]
					} else if k < len(L) && i < len(L[k]) {
						L[k][i] = 0
					}
				}
			}
		}
	}
}

// / https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
// / Must be at least 3x3 matrix
func (m Matrix) Quaternion() (q *vec.Quaternion) {
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
