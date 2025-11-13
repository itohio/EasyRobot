package mat

import (
	"fmt"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

type Matrix3x4 [3][4]float32

var _ matTypes.Matrix = Matrix3x4{}

func (m Matrix3x4) View() matTypes.Matrix {
	return Matrix{
		[]float32{m[0][0], m[0][1], m[0][2], m[0][3]},
		[]float32{m[1][0], m[1][1], m[1][2], m[1][3]},
		[]float32{m[2][0], m[2][1], m[2][2], m[2][3]},
	}
}

func (m Matrix3x4) Rows() int { return 3 }
func (m Matrix3x4) Cols() int { return 4 }
func (m Matrix3x4) Rank() int {
	return rankSmall(3, 4, func(i, j int) float32 { return m[i][j] })
}
func (m Matrix3x4) IsContiguous() bool { return true }

func New3x4(arr ...float32) Matrix3x4 {
	var out Matrix3x4
	if len(arr) >= 12 {
		out[0][0], out[0][1], out[0][2], out[0][3] = arr[0], arr[1], arr[2], arr[3]
		out[1][0], out[1][1], out[1][2], out[1][3] = arr[4], arr[5], arr[6], arr[7]
		out[2][0], out[2][1], out[2][2], out[2][3] = arr[8], arr[9], arr[10], arr[11]
	}
	return out
}

func (m Matrix3x4) CopyFrom(src matTypes.Matrix) {
	// noop
}

func (m Matrix3x4) Flat() []float32 {
	return []float32{
		m[0][0], m[0][1], m[0][2], m[0][3],
		m[1][0], m[1][1], m[1][2], m[1][3],
		m[2][0], m[2][1], m[2][2], m[2][3],
	}
}

func (m Matrix3x4) RotationX(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = 1, 0, 0
	res[1][0], res[1][1], res[1][2] = 0, c, -s
	res[2][0], res[2][1], res[2][2] = 0, s, c
	return res
}

func (m Matrix3x4) RotationY(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = c, 0, s
	res[1][0], res[1][1], res[1][2] = 0, 1, 0
	res[2][0], res[2][1], res[2][2] = -s, 0, c
	return res
}

func (m Matrix3x4) RotationZ(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = c, -s, 0
	res[1][0], res[1][1], res[1][2] = s, c, 0
	res[2][0], res[2][1], res[2][2] = 0, 0, 1
	return res
}

func (m Matrix3x4) Rotation2D(a float32) matTypes.Matrix {
	return m.RotationZ(a)
}

func (m Matrix3x4) Orientation(q vecTypes.Quaternion) matTypes.Matrix {
	rot := Matrix3x3{}.Orientation(q).(Matrix3x3)
	res := m
	for i := 0; i < 3; i++ {
		res[i][0], res[i][1], res[i][2] = rot[i][0], rot[i][1], rot[i][2]
	}
	return res
}

func (m Matrix3x4) Eye() matTypes.Matrix {
	return Matrix3x4{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
}

func (m Matrix3x4) Row(row int) vecTypes.Vector {
	if row < 0 || row >= 3 {
		panic(fmt.Sprintf("Matrix3x4.Row: index %d out of range", row))
	}
	return vec.Vector4D{m[row][0], m[row][1], m[row][2], m[row][3]}
}

func (m Matrix3x4) Col(col int, _ vecTypes.Vector) vecTypes.Vector {
	if col < 0 || col >= 4 {
		panic(fmt.Sprintf("Matrix3x4.Col: index %d out of range", col))
	}
	return vec.Vector3D{m[0][col], m[1][col], m[2][col]}
}

func (m Matrix3x4) SetRow(row int, v vecTypes.Vector) matTypes.Matrix {
	if row < 0 || row >= 3 {
		panic(fmt.Sprintf("Matrix3x4.SetRow: index %d out of range", row))
	}
	vecVal := v.(vec.Vector4D)
	res := m
	res[row][0], res[row][1], res[row][2], res[row][3] = vecVal[0], vecVal[1], vecVal[2], vecVal[3]
	return res
}

func (m Matrix3x4) SetCol(col int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 4 {
		panic(fmt.Sprintf("Matrix3x4.SetCol: index %d out of range", col))
	}
	vecVal := v.(vec.Vector3D)
	res := m
	res[0][col], res[1][col], res[2][col] = vecVal[0], vecVal[1], vecVal[2]
	return res
}

func (m Matrix3x4) SetColFromRow(col int, rowStart int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 4 {
		panic(fmt.Sprintf("Matrix3x4.SetColFromRow: column %d out of range", col))
	}
	if rowStart < 0 || rowStart > 2 {
		panic(fmt.Sprintf("Matrix3x4.SetColFromRow: rowStart %d out of range", rowStart))
	}
	vecVal := v.(vec.Vector3D)
	res := m
	for i := 0; i+rowStart < 3 && i < len(vecVal); i++ {
		res[rowStart+i][col] = vecVal[i]
	}
	return res
}

func (m Matrix3x4) Submatrix(row, col int, _ matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return m
	}
	panic("Matrix3x4.Submatrix: unsupported submatrix extraction")
}

func (m Matrix3x4) SetSubmatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return m1.(Matrix3x4)
	}
	panic("Matrix3x4.SetSubmatrix: unsupported placement")
}

func (m Matrix3x4) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) matTypes.Matrix {
	if row == 0 && col == 0 && rows1 == 3 && cols1 == 3 && len(m1) >= 9 {
		res := m
		for i := 0; i < 3; i++ {
			res[i][0], res[i][1], res[i][2] = m1[i*3], m1[i*3+1], m1[i*3+2]
		}
		return res
	}
	panic("Matrix3x4.SetSubmatrixRaw: unsupported parameters")
}

func (m Matrix3x4) Clone() matTypes.Matrix { return m }

func (m Matrix3x4) Transpose(m1 matTypes.Matrix) matTypes.Matrix {
	src := m1.(Matrix3x4)
	var res Matrix4x3
	for i := 0; i < 3; i++ {
		res[0][i] = src[i][0]
		res[1][i] = src[i][1]
		res[2][i] = src[i][2]
		res[3][i] = src[i][3]
	}
	return res
}

func (m Matrix3x4) Add(m1 matTypes.Matrix) matTypes.Matrix {
	other := m1.(Matrix3x4)
	res := m
	for i := 0; i < 3; i++ {
		res[i][0] += other[i][0]
		res[i][1] += other[i][1]
		res[i][2] += other[i][2]
		res[i][3] += other[i][3]
	}
	return res
}

func (m Matrix3x4) Sub(m1 matTypes.Matrix) matTypes.Matrix {
	other := m1.(Matrix3x4)
	res := m
	for i := 0; i < 3; i++ {
		res[i][0] -= other[i][0]
		res[i][1] -= other[i][1]
		res[i][2] -= other[i][2]
		res[i][3] -= other[i][3]
	}
	return res
}

func (m Matrix3x4) MulC(c float32) matTypes.Matrix {
	res := m
	for i := 0; i < 3; i++ {
		res[i][0] *= c
		res[i][1] *= c
		res[i][2] *= c
		res[i][3] *= c
	}
	return res
}

func (m Matrix3x4) DivC(c float32) matTypes.Matrix {
	if c == 0 {
		panic("Matrix3x4.DivC: divide by zero")
	}
	inv := 1 / c
	res := m
	for i := 0; i < 3; i++ {
		res[i][0] *= inv
		res[i][1] *= inv
		res[i][2] *= inv
		res[i][3] *= inv
	}
	return res
}

func (m Matrix3x4) Mul(a matTypes.Matrix, b matTypes.Matrix) matTypes.Matrix {
	left := a.(Matrix3x4)
	right := b.(Matrix4x4)
	var res Matrix3x4
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			res[i][j] = left[i][0]*right[0][j] + left[i][1]*right[1][j] + left[i][2]*right[2][j] + left[i][3]*right[3][j]
		}
	}
	return res
}

func (m Matrix3x4) MulDiag(a matTypes.Matrix, b vecTypes.Vector) matTypes.Matrix {
	src := a.(Matrix3x4)
	diag := b.(vec.Vector4D)
	var res Matrix3x4
	for i := 0; i < 3; i++ {
		res[i][0] = src[i][0] * diag[0]
		res[i][1] = src[i][1] * diag[1]
		res[i][2] = src[i][2] * diag[2]
		res[i][3] = src[i][3] * diag[3]
	}
	return res
}

func (m Matrix3x4) MulVec(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector4D)
	return vec.Vector3D{
		m[0][0]*vecVal[0] + m[0][1]*vecVal[1] + m[0][2]*vecVal[2] + m[0][3]*vecVal[3],
		m[1][0]*vecVal[0] + m[1][1]*vecVal[1] + m[1][2]*vecVal[2] + m[1][3]*vecVal[3],
		m[2][0]*vecVal[0] + m[2][1]*vecVal[1] + m[2][2]*vecVal[2] + m[2][3]*vecVal[3],
	}
}

func (m Matrix3x4) MulVecT(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector3D)
	return vec.Vector4D{
		m[0][0]*vecVal[0] + m[1][0]*vecVal[1] + m[2][0]*vecVal[2],
		m[0][1]*vecVal[0] + m[1][1]*vecVal[1] + m[2][1]*vecVal[2],
		m[0][2]*vecVal[0] + m[1][2]*vecVal[1] + m[2][2]*vecVal[2],
		m[0][3]*vecVal[0] + m[1][3]*vecVal[1] + m[2][3]*vecVal[2],
	}
}

func (m Matrix3x4) Det() float32 {
	panic("Matrix3x4.Det: undefined for non-square matrices")
}

func (m Matrix3x4) LU(matTypes.Matrix, matTypes.Matrix) {
	panic("Matrix3x4.LU: unsupported for rectangular matrices")
}

func (m Matrix3x4) Quaternion() vecTypes.Quaternion {
	rot := Matrix3x3{
		{m[0][0], m[0][1], m[0][2]},
		{m[1][0], m[1][1], m[1][2]},
		{m[2][0], m[2][1], m[2][2]},
	}
	return rot.Quaternion()
}

func (m Matrix3x4) Cholesky(matTypes.Matrix) error {
	panic("Matrix3x4.Cholesky: unsupported for rectangular matrices")
}

func (m Matrix3x4) CholeskySolve(vecTypes.Vector, vecTypes.Vector) error {
	panic("Matrix3x4.CholeskySolve: unsupported for rectangular matrices")
}

func (m Matrix3x4) QRDecompose(*matTypes.QRResult) error {
	panic("Matrix3x4.QRDecompose: unsupported for rectangular matrices")
}

func (m Matrix3x4) QR(*matTypes.QRResult) error {
	panic("Matrix3x4.QR: unsupported for rectangular matrices")
}

func (m Matrix3x4) Inverse(matTypes.Matrix) error {
	panic("Matrix3x4.Inverse: unsupported for rectangular matrices")
}

func (m Matrix3x4) PseudoInverse(matTypes.Matrix) error {
	panic("Matrix3x4.PseudoInverse: unsupported for rectangular matrices")
}

func (m Matrix3x4) DampedLeastSquares(float32, matTypes.Matrix) error {
	panic("Matrix3x4.DampedLeastSquares: unsupported for rectangular matrices")
}

func (m Matrix3x4) SVD(*matTypes.SVDResult) error {
	panic("Matrix3x4.SVD: unsupported for rectangular matrices")
}

func (m Matrix3x4) GetCol(col int, _ vecTypes.Vector) vecTypes.Vector {
	return m.Col(col, nil)
}

func (m Matrix3x4) Diagonal(_ vecTypes.Vector) vecTypes.Vector {
	return vec.Vector3D{m[0][0], m[1][1], m[2][2]}
}

func (m Matrix3x4) SetDiagonal(v vecTypes.Vector) matTypes.Matrix {
	vecVal := v.(vec.Vector3D)
	res := m
	res[0][0], res[1][1], res[2][2] = vecVal[0], vecVal[1], vecVal[2]
	return res
}
