package mat

import (
	"fmt"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

type Matrix4x3 [4][3]float32

var _ matTypes.Matrix = Matrix4x3{}

func (m Matrix4x3) View() matTypes.Matrix {
	return Matrix{
		[]float32{m[0][0], m[0][1], m[0][2]},
		[]float32{m[1][0], m[1][1], m[1][2]},
		[]float32{m[2][0], m[2][1], m[2][2]},
		[]float32{m[3][0], m[3][1], m[3][2]},
	}
}

func (m Matrix4x3) Rows() int { return 4 }
func (m Matrix4x3) Cols() int { return 3 }
func (m Matrix4x3) Rank() int {
	return rankSmall(4, 3, func(i, j int) float32 { return m[i][j] })
}
func (m Matrix4x3) IsContiguous() bool { return true }

func New4x3(arr ...float32) Matrix4x3 {
	var out Matrix4x3
	if len(arr) >= 12 {
		out[0][0], out[0][1], out[0][2] = arr[0], arr[1], arr[2]
		out[1][0], out[1][1], out[1][2] = arr[3], arr[4], arr[5]
		out[2][0], out[2][1], out[2][2] = arr[6], arr[7], arr[8]
		out[3][0], out[3][1], out[3][2] = arr[9], arr[10], arr[11]
	}
	return out
}

func (m Matrix4x3) CopyFrom(src matTypes.Matrix) {
	// noop
}

func (m Matrix4x3) Flat() []float32 {
	return []float32{
		m[0][0], m[0][1], m[0][2],
		m[1][0], m[1][1], m[1][2],
		m[2][0], m[2][1], m[2][2],
		m[3][0], m[3][1], m[3][2],
	}
}

func (m Matrix4x3) RotationX(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = 1, 0, 0
	res[1][0], res[1][1], res[1][2] = 0, c, -s
	res[2][0], res[2][1], res[2][2] = 0, s, c
	return res
}

func (m Matrix4x3) RotationY(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = c, 0, s
	res[1][0], res[1][1], res[1][2] = 0, 1, 0
	res[2][0], res[2][1], res[2][2] = -s, 0, c
	return res
}

func (m Matrix4x3) RotationZ(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	res := m
	res[0][0], res[0][1], res[0][2] = c, -s, 0
	res[1][0], res[1][1], res[1][2] = s, c, 0
	res[2][0], res[2][1], res[2][2] = 0, 0, 1
	return res
}

func (m Matrix4x3) Rotation2D(a float32) matTypes.Matrix {
	return m.RotationZ(a)
}

func (m Matrix4x3) Orientation(q vecTypes.Quaternion) matTypes.Matrix {
	rot := Matrix3x3{}.Orientation(q).(Matrix3x3)
	res := m
	for i := 0; i < 3; i++ {
		res[i][0], res[i][1], res[i][2] = rot[i][0], rot[i][1], rot[i][2]
	}
	return res
}

func (m Matrix4x3) Eye() matTypes.Matrix {
	return Matrix4x3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}}
}

func (m Matrix4x3) Row(row int) vecTypes.Vector {
	if row < 0 || row >= 4 {
		panic(fmt.Sprintf("Matrix4x3.Row: index %d out of range", row))
	}
	return vec.Vector3D{m[row][0], m[row][1], m[row][2]}
}

func (m Matrix4x3) Col(col int, _ vecTypes.Vector) vecTypes.Vector {
	if col < 0 || col >= 3 {
		panic(fmt.Sprintf("Matrix4x3.Col: index %d out of range", col))
	}
	return vec.Vector4D{m[0][col], m[1][col], m[2][col], m[3][col]}
}

func (m Matrix4x3) SetRow(row int, v vecTypes.Vector) matTypes.Matrix {
	if row < 0 || row >= 4 {
		panic(fmt.Sprintf("Matrix4x3.SetRow: index %d out of range", row))
	}
	vecVal := v.(vec.Vector3D)
	res := m
	res[row][0], res[row][1], res[row][2] = vecVal[0], vecVal[1], vecVal[2]
	return res
}

func (m Matrix4x3) SetCol(col int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 3 {
		panic(fmt.Sprintf("Matrix4x3.SetCol: index %d out of range", col))
	}
	vecVal := v.(vec.Vector4D)
	res := m
	res[0][col], res[1][col], res[2][col], res[3][col] = vecVal[0], vecVal[1], vecVal[2], vecVal[3]
	return res
}

func (m Matrix4x3) SetColFromRow(col int, rowStart int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 3 {
		panic(fmt.Sprintf("Matrix4x3.SetColFromRow: column %d out of range", col))
	}
	if rowStart < 0 || rowStart > 3 {
		panic(fmt.Sprintf("Matrix4x3.SetColFromRow: rowStart %d out of range", rowStart))
	}
	vecVal := v.(vec.Vector4D)
	res := m
	for i := 0; i+rowStart < 4 && i < len(vecVal); i++ {
		res[rowStart+i][col] = vecVal[i]
	}
	return res
}

func (m Matrix4x3) Submatrix(row, col int, _ matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return m
	}
	panic("Matrix4x3.Submatrix: unsupported submatrix extraction")
}

func (m Matrix4x3) SetSubmatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return m1.(Matrix4x3)
	}
	panic("Matrix4x3.SetSubmatrix: unsupported placement")
}

func (m Matrix4x3) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) matTypes.Matrix {
	if row == 0 && col == 0 && rows1 == 3 && cols1 == 3 && len(m1) >= 9 {
		res := m
		for i := 0; i < 3; i++ {
			res[i][0], res[i][1], res[i][2] = m1[i*3], m1[i*3+1], m1[i*3+2]
		}
		return res
	}
	panic("Matrix4x3.SetSubmatrixRaw: unsupported parameters")
}

func (m Matrix4x3) Clone() matTypes.Matrix { return m }

func (m Matrix4x3) Transpose(m1 matTypes.Matrix) matTypes.Matrix {
	src := m1.(Matrix4x3)
	var res Matrix3x4
	for i := 0; i < 4; i++ {
		res[0][i] = src[i][0]
		res[1][i] = src[i][1]
		res[2][i] = src[i][2]
	}
	return res
}

func (m Matrix4x3) Add(m1 matTypes.Matrix) matTypes.Matrix {
	other := m1.(Matrix4x3)
	res := m
	for i := 0; i < 4; i++ {
		res[i][0] += other[i][0]
		res[i][1] += other[i][1]
		res[i][2] += other[i][2]
	}
	return res
}

func (m Matrix4x3) Sub(m1 matTypes.Matrix) matTypes.Matrix {
	other := m1.(Matrix4x3)
	res := m
	for i := 0; i < 4; i++ {
		res[i][0] -= other[i][0]
		res[i][1] -= other[i][1]
		res[i][2] -= other[i][2]
	}
	return res
}

func (m Matrix4x3) MulC(c float32) matTypes.Matrix {
	res := m
	for i := 0; i < 4; i++ {
		res[i][0] *= c
		res[i][1] *= c
		res[i][2] *= c
	}
	return res
}

func (m Matrix4x3) DivC(c float32) matTypes.Matrix {
	if c == 0 {
		panic("Matrix4x3.DivC: divide by zero")
	}
	inv := 1 / c
	res := m
	for i := 0; i < 4; i++ {
		res[i][0] *= inv
		res[i][1] *= inv
		res[i][2] *= inv
	}
	return res
}

func (m Matrix4x3) Mul(a matTypes.Matrix, b matTypes.Matrix) matTypes.Matrix {
	left := a.(Matrix3x4)
	right := b.(Matrix4x3)
	var res Matrix4x3
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			res[i][j] = left[i][0]*right[0][j] + left[i][1]*right[1][j] + left[i][2]*right[2][j] + left[i][3]*right[3][j]
		}
	}
	return res
}

func (m Matrix4x3) MulDiag(a matTypes.Matrix, b vecTypes.Vector) matTypes.Matrix {
	src := a.(Matrix4x3)
	diag := b.(vec.Vector3D)
	var res Matrix4x3
	for i := 0; i < 4; i++ {
		res[i][0] = src[i][0] * diag[0]
		res[i][1] = src[i][1] * diag[1]
		res[i][2] = src[i][2] * diag[2]
	}
	return res
}

func (m Matrix4x3) MulVec(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector3D)
	return vec.Vector4D{
		m[0][0]*vecVal[0] + m[0][1]*vecVal[1] + m[0][2]*vecVal[2],
		m[1][0]*vecVal[0] + m[1][1]*vecVal[1] + m[1][2]*vecVal[2],
		m[2][0]*vecVal[0] + m[2][1]*vecVal[1] + m[2][2]*vecVal[2],
		m[3][0]*vecVal[0] + m[3][1]*vecVal[1] + m[3][2]*vecVal[2],
	}
}

func (m Matrix4x3) MulVecT(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector4D)
	return vec.Vector3D{
		m[0][0]*vecVal[0] + m[1][0]*vecVal[1] + m[2][0]*vecVal[2] + m[3][0]*vecVal[3],
		m[0][1]*vecVal[0] + m[1][1]*vecVal[1] + m[2][1]*vecVal[2] + m[3][1]*vecVal[3],
		m[0][2]*vecVal[0] + m[1][2]*vecVal[1] + m[2][2]*vecVal[2] + m[3][2]*vecVal[3],
	}
}

func (m Matrix4x3) Det() float32 {
	panic("Matrix4x3.Det: undefined for non-square matrices")
}

func (m Matrix4x3) LU(matTypes.Matrix, matTypes.Matrix) {
	panic("Matrix4x3.LU: unsupported for rectangular matrices")
}

func (m Matrix4x3) Quaternion() vecTypes.Quaternion {
	rot := Matrix3x3{
		{m[0][0], m[0][1], m[0][2]},
		{m[1][0], m[1][1], m[1][2]},
		{m[2][0], m[2][1], m[2][2]},
	}
	return rot.Quaternion()
}

func (m Matrix4x3) Cholesky(matTypes.Matrix) error {
	panic("Matrix4x3.Cholesky: unsupported for rectangular matrices")
}

func (m Matrix4x3) CholeskySolve(vecTypes.Vector, vecTypes.Vector) error {
	panic("Matrix4x3.CholeskySolve: unsupported for rectangular matrices")
}

func (m Matrix4x3) QRDecompose(*matTypes.QRResult) error {
	panic("Matrix4x3.QRDecompose: unsupported for rectangular matrices")
}

func (m Matrix4x3) QR(*matTypes.QRResult) error {
	panic("Matrix4x3.QR: unsupported for rectangular matrices")
}

func (m Matrix4x3) Inverse(matTypes.Matrix) error {
	panic("Matrix4x3.Inverse: unsupported for rectangular matrices")
}

func (m Matrix4x3) PseudoInverse(matTypes.Matrix) error {
	panic("Matrix4x3.PseudoInverse: unsupported for rectangular matrices")
}

func (m Matrix4x3) DampedLeastSquares(float32, matTypes.Matrix) error {
	panic("Matrix4x3.DampedLeastSquares: unsupported for rectangular matrices")
}

func (m Matrix4x3) SVD(*matTypes.SVDResult) error {
	panic("Matrix4x3.SVD: unsupported for rectangular matrices")
}

func (m Matrix4x3) GetCol(col int, _ vecTypes.Vector) vecTypes.Vector {
	return m.Col(col, nil)
}

func (m Matrix4x3) Diagonal(_ vecTypes.Vector) vecTypes.Vector {
	return vec.Vector3D{m[0][0], m[1][1], m[2][2]}
}

func (m Matrix4x3) SetDiagonal(v vecTypes.Vector) matTypes.Matrix {
	vecVal := v.(vec.Vector3D)
	res := m
	res[0][0], res[1][1], res[2][2] = vecVal[0], vecVal[1], vecVal[2]
	return res
}
