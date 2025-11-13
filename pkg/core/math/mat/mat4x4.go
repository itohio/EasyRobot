package mat

import (
	"fmt"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

type Matrix4x4 [4][4]float32

var _ matTypes.Matrix = Matrix4x4{}

func (m Matrix4x4) View() matTypes.Matrix {
	return Matrix{
		[]float32{m[0][0], m[0][1], m[0][2], m[0][3]},
		[]float32{m[1][0], m[1][1], m[1][2], m[1][3]},
		[]float32{m[2][0], m[2][1], m[2][2], m[2][3]},
		[]float32{m[3][0], m[3][1], m[3][2], m[3][3]},
	}
}

func (m Matrix4x4) Rows() int { return 4 }
func (m Matrix4x4) Cols() int { return 4 }
func (m Matrix4x4) Rank() int {
	return rankSmall(4, 4, func(i, j int) float32 { return m[i][j] })
}
func (m Matrix4x4) IsContiguous() bool { return true }

func New4x4(arr ...float32) Matrix4x4 {
	var out Matrix4x4
	if len(arr) >= 16 {
		for i := 0; i < 4; i++ {
			out[i][0], out[i][1], out[i][2], out[i][3] = arr[i*4], arr[i*4+1], arr[i*4+2], arr[i*4+3]
		}
	}
	return out
}

func (m Matrix4x4) CopyFrom(src matTypes.Matrix) {
	// noop
}

func (m Matrix4x4) Flat() []float32 {
	return []float32{
		m[0][0], m[0][1], m[0][2], m[0][3],
		m[1][0], m[1][1], m[1][2], m[1][3],
		m[2][0], m[2][1], m[2][2], m[2][3],
		m[3][0], m[3][1], m[3][2], m[3][3],
	}
}

func (m Matrix4x4) RotationX(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = 1, 0, 0
	res[1][0], res[1][1], res[1][2] = 0, c, -s
	res[2][0], res[2][1], res[2][2] = 0, s, c
	return res
}

func (m Matrix4x4) RotationY(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = c, 0, s
	res[1][0], res[1][1], res[1][2] = 0, 1, 0
	res[2][0], res[2][1], res[2][2] = -s, 0, c
	return res
}

func (m Matrix4x4) RotationZ(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = c, -s, 0
	res[1][0], res[1][1], res[1][2] = s, c, 0
	res[2][0], res[2][1], res[2][2] = 0, 0, 1
	return res
}

func (m Matrix4x4) Rotation2D(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1] = c, -s
	res[1][0], res[1][1] = s, c
	return res
}

func (m Matrix4x4) Orientation(q vecTypes.Quaternion) matTypes.Matrix {
	var quat vec.Quaternion
	switch v := q.(type) {
	case vec.Quaternion:
		quat = v
	case *vec.Quaternion:
		quat = *v
	default:
		panic(fmt.Sprintf("Matrix4x4.Orientation: unsupported quaternion type %T", q))
	}
	rot := Matrix3x3{}.Orientation(&quat).(Matrix3x3)
	res := m
	for i := 0; i < 3; i++ {
		res[i][0], res[i][1], res[i][2] = rot[i][0], rot[i][1], rot[i][2]
	}
	return res
}

func (m Matrix4x4) Eye() matTypes.Matrix {
	return Matrix4x4{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
}

func (m Matrix4x4) Row(row int) vecTypes.Vector {
	if row < 0 || row >= 4 {
		panic(fmt.Sprintf("Matrix4x4.Row: index %d out of range", row))
	}
	return vec.Vector4D{m[row][0], m[row][1], m[row][2], m[row][3]}
}

func (m Matrix4x4) Col(col int, _ vecTypes.Vector) vecTypes.Vector {
	if col < 0 || col >= 4 {
		panic(fmt.Sprintf("Matrix4x4.Col: index %d out of range", col))
	}
	return vec.Vector4D{m[0][col], m[1][col], m[2][col], m[3][col]}
}

func (m Matrix4x4) SetRow(row int, v vecTypes.Vector) matTypes.Matrix {
	if row < 0 || row >= 4 {
		panic(fmt.Sprintf("Matrix4x4.SetRow: index %d out of range", row))
	}
	vecVal := v.(vec.Vector4D)
	res := m
	res[row][0], res[row][1], res[row][2], res[row][3] = vecVal[0], vecVal[1], vecVal[2], vecVal[3]
	return res
}

func (m Matrix4x4) SetCol(col int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 4 {
		panic(fmt.Sprintf("Matrix4x4.SetCol: index %d out of range", col))
	}
	vecVal := v.(vec.Vector4D)
	res := m
	res[0][col], res[1][col], res[2][col], res[3][col] = vecVal[0], vecVal[1], vecVal[2], vecVal[3]
	return res
}

func (m Matrix4x4) SetColFromRow(col int, rowStart int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 4 {
		panic(fmt.Sprintf("Matrix4x4.SetColFromRow: column %d out of range", col))
	}
	if rowStart < 0 || rowStart > 3 {
		panic(fmt.Sprintf("Matrix4x4.SetColFromRow: rowStart %d out of range", rowStart))
	}
	vecVal := v.(vec.Vector4D)
	res := m
	for i := 0; i+rowStart < 4 && i < len(vecVal); i++ {
		res[rowStart+i][col] = vecVal[i]
	}
	return res
}

func (m Matrix4x4) Diagonal(_ vecTypes.Vector) vecTypes.Vector {
	return vec.Vector4D{m[0][0], m[1][1], m[2][2], m[3][3]}
}

func (m Matrix4x4) SetDiagonal(v vecTypes.Vector) matTypes.Matrix {
	vecVal := v.(vec.Vector4D)
	res := m
	res[0][0], res[1][1], res[2][2], res[3][3] = vecVal[0], vecVal[1], vecVal[2], vecVal[3]
	return res
}

func FromDiagonal4x4(d0, d1, d2, d3 float32) Matrix4x4 {
	return Matrix4x4{{d0, 0, 0, 0}, {0, d1, 0, 0}, {0, 0, d2, 0}, {0, 0, 0, d3}}
}

func FromVector4x4(v vec.Vector4D) Matrix4x4 {
	return Matrix4x4{{v[0], 0, 0, 0}, {0, v[1], 0, 0}, {0, 0, v[2], 0}, {0, 0, 0, v[3]}}
}

func (m Matrix4x4) Submatrix(row, col int, _ matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return m
	}
	panic("Matrix4x4.Submatrix: unsupported submatrix extraction")
}

func (m Matrix4x4) SetSubmatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return m1.(Matrix4x4)
	}
	panic("Matrix4x4.SetSubmatrix: unsupported submatrix placement")
}

func (m Matrix4x4) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) matTypes.Matrix {
	if row == 0 && col == 0 && rows1 == 4 && cols1 == 4 && len(m1) >= 16 {
		return New4x4(m1...)
	}
	panic("Matrix4x4.SetSubmatrixRaw: unsupported parameters")
}

func (m Matrix4x4) Clone() matTypes.Matrix { return m }

func (m Matrix4x4) Transpose(m1 matTypes.Matrix) matTypes.Matrix {
	src := m1.(Matrix4x4)
	return Matrix4x4{{src[0][0], src[1][0], src[2][0], src[3][0]}, {src[0][1], src[1][1], src[2][1], src[3][1]}, {src[0][2], src[1][2], src[2][2], src[3][2]}, {src[0][3], src[1][3], src[2][3], src[3][3]}}
}

func (m Matrix4x4) Add(m1 matTypes.Matrix) matTypes.Matrix {
	other := m1.(Matrix4x4)
	res := m
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			res[i][j] += other[i][j]
		}
	}
	return res
}

func (m Matrix4x4) Sub(m1 matTypes.Matrix) matTypes.Matrix {
	other := m1.(Matrix4x4)
	res := m
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			res[i][j] -= other[i][j]
		}
	}
	return res
}

func (m Matrix4x4) MulC(c float32) matTypes.Matrix {
	res := m
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			res[i][j] *= c
		}
	}
	return res
}

func (m Matrix4x4) DivC(c float32) matTypes.Matrix {
	if c == 0 {
		panic("Matrix4x4.DivC: divide by zero")
	}
	inv := 1 / c
	res := m
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			res[i][j] *= inv
		}
	}
	return res
}

func (m Matrix4x4) Mul(a matTypes.Matrix, b matTypes.Matrix) matTypes.Matrix {
	left := a.(Matrix4x4)
	right := b.(Matrix4x4)
	var res Matrix4x4
	res[0][0] = left[0][0]*right[0][0] + left[0][1]*right[1][0] + left[0][2]*right[2][0] + left[0][3]*right[3][0]
	res[0][1] = left[0][0]*right[0][1] + left[0][1]*right[1][1] + left[0][2]*right[2][1] + left[0][3]*right[3][1]
	res[0][2] = left[0][0]*right[0][2] + left[0][1]*right[1][2] + left[0][2]*right[2][2] + left[0][3]*right[3][2]
	res[0][3] = left[0][0]*right[0][3] + left[0][1]*right[1][3] + left[0][2]*right[2][3] + left[0][3]*right[3][3]

	res[1][0] = left[1][0]*right[0][0] + left[1][1]*right[1][0] + left[1][2]*right[2][0] + left[1][3]*right[3][0]
	res[1][1] = left[1][0]*right[0][1] + left[1][1]*right[1][1] + left[1][2]*right[2][1] + left[1][3]*right[3][1]
	res[1][2] = left[1][0]*right[0][2] + left[1][1]*right[1][2] + left[1][2]*right[2][2] + left[1][3]*right[3][2]
	res[1][3] = left[1][0]*right[0][3] + left[1][1]*right[1][3] + left[1][2]*right[2][3] + left[1][3]*right[3][3]

	res[2][0] = left[2][0]*right[0][0] + left[2][1]*right[1][0] + left[2][2]*right[2][0] + left[2][3]*right[3][0]
	res[2][1] = left[2][0]*right[0][1] + left[2][1]*right[1][1] + left[2][2]*right[2][1] + left[2][3]*right[3][1]
	res[2][2] = left[2][0]*right[0][2] + left[2][1]*right[1][2] + left[2][2]*right[2][2] + left[2][3]*right[3][2]
	res[2][3] = left[2][0]*right[0][3] + left[2][1]*right[1][3] + left[2][2]*right[2][3] + left[2][3]*right[3][3]

	res[3][0] = left[3][0]*right[0][0] + left[3][1]*right[1][0] + left[3][2]*right[2][0] + left[3][3]*right[3][0]
	res[3][1] = left[3][0]*right[0][1] + left[3][1]*right[1][1] + left[3][2]*right[2][1] + left[3][3]*right[3][1]
	res[3][2] = left[3][0]*right[0][2] + left[3][1]*right[1][2] + left[3][2]*right[2][2] + left[3][3]*right[3][2]
	res[3][3] = left[3][0]*right[0][3] + left[3][1]*right[1][3] + left[3][2]*right[2][3] + left[3][3]*right[3][3]
	return res
}

func (m Matrix4x4) MulDiag(a matTypes.Matrix, b vecTypes.Vector) matTypes.Matrix {
	var src Matrix4x4
	switch v := a.(type) {
	case Matrix4x4:
		src = v
	case *Matrix4x4:
		src = *v
	default:
		panic(fmt.Sprintf("Matrix4x4.MulDiag: unsupported matrix type %T", a))
	}
	diag := b.(vec.Vector4D)
	var res Matrix4x4
	for i := 0; i < 4; i++ {
		res[i][0] = src[i][0] * diag[0]
		res[i][1] = src[i][1] * diag[1]
		res[i][2] = src[i][2] * diag[2]
		res[i][3] = src[i][3] * diag[3]
	}
	return res
}

func (m Matrix4x4) MulVec(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector4D)
	return vec.Vector4D{
		m[0][0]*vecVal[0] + m[0][1]*vecVal[1] + m[0][2]*vecVal[2] + m[0][3]*vecVal[3],
		m[1][0]*vecVal[0] + m[1][1]*vecVal[1] + m[1][2]*vecVal[2] + m[1][3]*vecVal[3],
		m[2][0]*vecVal[0] + m[2][1]*vecVal[1] + m[2][2]*vecVal[2] + m[2][3]*vecVal[3],
		m[3][0]*vecVal[0] + m[3][1]*vecVal[1] + m[3][2]*vecVal[2] + m[3][3]*vecVal[3],
	}
}

func (m Matrix4x4) MulVecT(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector4D)
	return vec.Vector4D{
		m[0][0]*vecVal[0] + m[1][0]*vecVal[1] + m[2][0]*vecVal[2] + m[3][0]*vecVal[3],
		m[0][1]*vecVal[0] + m[1][1]*vecVal[1] + m[2][1]*vecVal[2] + m[3][1]*vecVal[3],
		m[0][2]*vecVal[0] + m[1][2]*vecVal[1] + m[2][2]*vecVal[2] + m[3][2]*vecVal[3],
		m[0][3]*vecVal[0] + m[1][3]*vecVal[1] + m[2][3]*vecVal[2] + m[3][3]*vecVal[3],
	}
}

func (m Matrix4x4) Det() float32 {
	data := [4][4]float32{}
	for i := 0; i < 4; i++ {
		copy(data[i][:], m[i][:])
	}
	det := float32(1)
	sign := float32(1)
	for i := 0; i < 4; i++ {
		pivot := i
		maxVal := math32.Abs(data[i][i])
		for r := i + 1; r < 4; r++ {
			if v := math32.Abs(data[r][i]); v > maxVal {
				maxVal = v
				pivot = r
			}
		}
		if maxVal == 0 {
			return 0
		}
		if pivot != i {
			data[i], data[pivot] = data[pivot], data[i]
			sign = -sign
		}
		pivotVal := data[i][i]
		det *= pivotVal
		invPivot := 1 / pivotVal
		for r := i + 1; r < 4; r++ {
			factor := data[r][i] * invPivot
			for c := i + 1; c < 4; c++ {
				data[r][c] -= factor * data[i][c]
			}
		}
	}
	return det * sign
}

func (m Matrix4x4) LU(L, U matTypes.Matrix) {
	lMat, ok := L.(*Matrix4x4)
	if !ok {
		panic("Matrix4x4.LU: destination L must be *Matrix4x4")
	}
	uMat, ok := U.(*Matrix4x4)
	if !ok {
		panic("Matrix4x4.LU: destination U must be *Matrix4x4")
	}
	for i := 0; i < 4; i++ {
		for k := i; k < 4; k++ {
			sum := float32(0)
			for j := 0; j < i; j++ {
				sum += lMat[i][j] * uMat[j][k]
			}
			uMat[i][k] = m[i][k] - sum
		}
		for k := i; k < 4; k++ {
			if i == k {
				lMat[i][i] = 1
			} else {
				sum := float32(0)
				for j := 0; j < i; j++ {
					sum += lMat[k][j] * uMat[j][i]
				}
				lMat[k][i] = (m[k][i] - sum) / uMat[i][i]
			}
		}
	}
}

func (m Matrix4x4) Quaternion() vecTypes.Quaternion {
	rot := Matrix3x3{
		{m[0][0], m[0][1], m[0][2]},
		{m[1][0], m[1][1], m[1][2]},
		{m[2][0], m[2][1], m[2][2]},
	}
	return rot.Quaternion()
}

func (m Matrix4x4) Cholesky(matTypes.Matrix) error {
	panic("Matrix4x4.Cholesky: unsupported in fixed matrix implementation")
}

func (m Matrix4x4) CholeskySolve(vecTypes.Vector, vecTypes.Vector) error {
	panic("Matrix4x4.CholeskySolve: unsupported in fixed matrix implementation")
}

func (m Matrix4x4) QRDecompose(*matTypes.QRResult) error {
	panic("Matrix4x4.QRDecompose: unsupported in fixed matrix implementation")
}

func (m Matrix4x4) QR(*matTypes.QRResult) error {
	panic("Matrix4x4.QR: unsupported in fixed matrix implementation")
}

func (m Matrix4x4) PseudoInverse(matTypes.Matrix) error {
	panic("Matrix4x4.PseudoInverse: unsupported in fixed matrix implementation")
}

func (m Matrix4x4) DampedLeastSquares(float32, matTypes.Matrix) error {
	panic("Matrix4x4.DampedLeastSquares: unsupported in fixed matrix implementation")
}

func (m Matrix4x4) SVD(*matTypes.SVDResult) error {
	panic("Matrix4x4.SVD: unsupported in fixed matrix implementation")
}

func (m Matrix4x4) GetCol(col int, _ vecTypes.Vector) vecTypes.Vector {
	return m.Col(col, nil)
}
