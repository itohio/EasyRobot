package mat

import (
	"fmt"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

type Matrix3x3 [3][3]float32

var _ matTypes.Matrix = Matrix3x3{}

func asMatrix3x3(arg matTypes.Matrix, op string) Matrix3x3 {
	switch v := arg.(type) {
	case Matrix3x3:
		return v
	case *Matrix3x3:
		return *v
	default:
		panic(fmt.Sprintf("Matrix3x3.%s: unsupported matrix type %T", op, arg))
	}
}

func (m Matrix3x3) Release() {
}

func (m Matrix3x3) View() matTypes.Matrix {
	return Matrix{
		[]float32{m[0][0], m[0][1], m[0][2]},
		[]float32{m[1][0], m[1][1], m[1][2]},
		[]float32{m[2][0], m[2][1], m[2][2]},
	}
}

func (m Matrix3x3) Rows() int { return 3 }
func (m Matrix3x3) Cols() int { return 3 }
func (m Matrix3x3) Rank() int {
	return rankSmall(3, 3, func(i, j int) float32 { return m[i][j] })
}
func (m Matrix3x3) IsContiguous() bool { return true }

func New3x3(arr ...float32) Matrix3x3 {
	var out Matrix3x3
	if len(arr) >= 9 {
		out[0][0], out[0][1], out[0][2] = arr[0], arr[1], arr[2]
		out[1][0], out[1][1], out[1][2] = arr[3], arr[4], arr[5]
		out[2][0], out[2][1], out[2][2] = arr[6], arr[7], arr[8]
	}
	return out
}

func (m Matrix3x3) Flat() []float32 {
	return []float32{
		m[0][0], m[0][1], m[0][2],
		m[1][0], m[1][1], m[1][2],
		m[2][0], m[2][1], m[2][2],
	}
}

func (m Matrix3x3) RotationX(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return Matrix3x3{{1, 0, 0}, {0, c, -s}, {0, s, c}}
}

func (m Matrix3x3) RotationY(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return Matrix3x3{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}
}

func (m Matrix3x3) RotationZ(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return Matrix3x3{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}
}

func (m Matrix3x3) Rotation2D(a float32) matTypes.Matrix {
	return m.RotationZ(a)
}

func (m Matrix3x3) Orientation(q vecTypes.Quaternion) matTypes.Matrix {
	var quat vec.Quaternion
	switch v := q.(type) {
	case vec.Quaternion:
		quat = v
	case *vec.Quaternion:
		quat = *v
	default:
		panic("Matrix3x3.Orientation: expected vec.Quaternion")
	}

	x, y, z, w := quat[0], quat[1], quat[2], quat[3]
	xx := x * x
	yy := y * y
	zz := z * z
	xy := x * y
	xz := x * z
	yz := y * z
	wx := w * x
	wy := w * y
	wz := w * z

	return Matrix3x3{
		{1 - 2*(yy+zz), 2 * (xy - wz), 2 * (xz + wy)},
		{2 * (xy + wz), 1 - 2*(xx+zz), 2 * (yz - wx)},
		{2 * (xz - wy), 2 * (yz + wx), 1 - 2*(xx+yy)},
	}
}

func (m Matrix3x3) CopyFrom(src matTypes.Matrix) {
	// noop
}

func (m Matrix3x3) Eye() matTypes.Matrix {
	return Matrix3x3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
}

func (m Matrix3x3) Row(row int) vecTypes.Vector {
	if row < 0 || row >= 3 {
		panic(fmt.Sprintf("Matrix3x3.Row: index %d out of range", row))
	}
	return vec.Vector3D{m[row][0], m[row][1], m[row][2]}
}

func (m Matrix3x3) Col(col int, _ vecTypes.Vector) vecTypes.Vector {
	if col < 0 || col >= 3 {
		panic(fmt.Sprintf("Matrix3x3.Col: index %d out of range", col))
	}
	return vec.Vector3D{m[0][col], m[1][col], m[2][col]}
}

func (m Matrix3x3) SetRow(row int, v vecTypes.Vector) matTypes.Matrix {
	if row < 0 || row >= 3 {
		panic(fmt.Sprintf("Matrix3x3.SetRow: index %d out of range", row))
	}
	vecVal := v.(vec.Vector3D)
	res := m
	res[row][0], res[row][1], res[row][2] = vecVal[0], vecVal[1], vecVal[2]
	return res
}

func (m Matrix3x3) SetCol(col int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 3 {
		panic(fmt.Sprintf("Matrix3x3.SetCol: index %d out of range", col))
	}
	vecVal := v.(vec.Vector3D)
	res := m
	res[0][col], res[1][col], res[2][col] = vecVal[0], vecVal[1], vecVal[2]
	return res
}

func (m Matrix3x3) SetColFromRow(col int, rowStart int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 3 {
		panic(fmt.Sprintf("Matrix3x3.SetColFromRow: column %d out of range", col))
	}
	if rowStart < 0 || rowStart > 2 {
		panic(fmt.Sprintf("Matrix3x3.SetColFromRow: rowStart %d out of range", rowStart))
	}
	vecVal := v.(vec.Vector3D)
	res := m
	for i := 0; i+rowStart < 3 && i < len(vecVal); i++ {
		res[rowStart+i][col] = vecVal[i]
	}
	return res
}

func (m Matrix3x3) GetCol(col int, _ vecTypes.Vector) vecTypes.Vector {
	if col < 0 || col >= 3 {
		panic(fmt.Sprintf("Matrix3x3.GetCol: index %d out of range", col))
	}
	return vec.Vector3D{m[0][col], m[1][col], m[2][col]}
}

func (m Matrix3x3) Diagonal(_ vecTypes.Vector) vecTypes.Vector {
	return vec.Vector3D{m[0][0], m[1][1], m[2][2]}
}

func (m Matrix3x3) SetDiagonal(v vecTypes.Vector) matTypes.Matrix {
	vecVal := v.(vec.Vector3D)
	res := m
	res[0][0], res[1][1], res[2][2] = vecVal[0], vecVal[1], vecVal[2]
	return res
}

func (m Matrix3x3) Submatrix(row, col int, _ matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return m
	}
	panic("Matrix3x3.Submatrix: unsupported submatrix extraction")
}

func (m Matrix3x3) SetSubmatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return asMatrix3x3(m1, "SetSubmatrix")
	}
	panic("Matrix3x3.SetSubmatrix: unsupported submatrix placement")
}

func (m Matrix3x3) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) matTypes.Matrix {
	if rows1 == 3 && cols1 == 3 && row == 0 && col == 0 && len(m1) >= 9 {
		return New3x3(m1...)
	}
	panic("Matrix3x3.SetSubmatrixRaw: unsupported parameters")
}

func (m Matrix3x3) Clone() matTypes.Matrix { return m }

func (m Matrix3x3) Transpose(m1 matTypes.Matrix) matTypes.Matrix {
	src := asMatrix3x3(m1, "Transpose")
	return Matrix3x3{{src[0][0], src[1][0], src[2][0]}, {src[0][1], src[1][1], src[2][1]}, {src[0][2], src[1][2], src[2][2]}}
}

func (m Matrix3x3) Add(m1 matTypes.Matrix) matTypes.Matrix {
	other := asMatrix3x3(m1, "Add")
	return Matrix3x3{
		{m[0][0] + other[0][0], m[0][1] + other[0][1], m[0][2] + other[0][2]},
		{m[1][0] + other[1][0], m[1][1] + other[1][1], m[1][2] + other[1][2]},
		{m[2][0] + other[2][0], m[2][1] + other[2][1], m[2][2] + other[2][2]},
	}
}

func (m Matrix3x3) Sub(m1 matTypes.Matrix) matTypes.Matrix {
	other := asMatrix3x3(m1, "Sub")
	return Matrix3x3{
		{m[0][0] - other[0][0], m[0][1] - other[0][1], m[0][2] - other[0][2]},
		{m[1][0] - other[1][0], m[1][1] - other[1][1], m[1][2] - other[1][2]},
		{m[2][0] - other[2][0], m[2][1] - other[2][1], m[2][2] - other[2][2]},
	}
}

func (m Matrix3x3) MulC(c float32) matTypes.Matrix {
	return Matrix3x3{
		{m[0][0] * c, m[0][1] * c, m[0][2] * c},
		{m[1][0] * c, m[1][1] * c, m[1][2] * c},
		{m[2][0] * c, m[2][1] * c, m[2][2] * c},
	}
}

func (m Matrix3x3) DivC(c float32) matTypes.Matrix {
	if c == 0 {
		panic("Matrix3x3.DivC: divide by zero")
	}
	inv := 1 / c
	return Matrix3x3{
		{m[0][0] * inv, m[0][1] * inv, m[0][2] * inv},
		{m[1][0] * inv, m[1][1] * inv, m[1][2] * inv},
		{m[2][0] * inv, m[2][1] * inv, m[2][2] * inv},
	}
}

func (m Matrix3x3) Mul(a matTypes.Matrix, b matTypes.Matrix) matTypes.Matrix {
	left := asMatrix3x3(a, "Mul.left")
	right := asMatrix3x3(b, "Mul.right")
	var res Matrix3x3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			res[i][j] = left[i][0]*right[0][j] + left[i][1]*right[1][j] + left[i][2]*right[2][j]
		}
	}
	return res
}

func (m Matrix3x3) MulDiag(a matTypes.Matrix, b vecTypes.Vector) matTypes.Matrix {
	src := asMatrix3x3(a, "MulDiag")
	diag := b.(vec.Vector3D)
	return Matrix3x3{
		{src[0][0] * diag[0], src[0][1] * diag[1], src[0][2] * diag[2]},
		{src[1][0] * diag[0], src[1][1] * diag[1], src[1][2] * diag[2]},
		{src[2][0] * diag[0], src[2][1] * diag[1], src[2][2] * diag[2]},
	}
}

func (m Matrix3x3) MulVec(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector3D)
	return vec.Vector3D{
		m[0][0]*vecVal[0] + m[0][1]*vecVal[1] + m[0][2]*vecVal[2],
		m[1][0]*vecVal[0] + m[1][1]*vecVal[1] + m[1][2]*vecVal[2],
		m[2][0]*vecVal[0] + m[2][1]*vecVal[1] + m[2][2]*vecVal[2],
	}
}

func (m Matrix3x3) MulVecT(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector3D)
	return vec.Vector3D{
		m[0][0]*vecVal[0] + m[1][0]*vecVal[1] + m[2][0]*vecVal[2],
		m[0][1]*vecVal[0] + m[1][1]*vecVal[1] + m[2][1]*vecVal[2],
		m[0][2]*vecVal[0] + m[1][2]*vecVal[1] + m[2][2]*vecVal[2],
	}
}

func (m Matrix3x3) Det() float32 {
	return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
}

func (m Matrix3x3) LU(_, _ matTypes.Matrix) {
	panic("Matrix3x3.LU: not supported in fixed matrix implementation")
}

func (m Matrix3x3) Cholesky(matTypes.Matrix) error {
	panic("Matrix3x3.Cholesky: not supported in fixed matrix implementation")
}

func (m Matrix3x3) CholeskySolve(vecTypes.Vector, vecTypes.Vector) error {
	panic("Matrix3x3.CholeskySolve: not supported in fixed matrix implementation")
}

func (m Matrix3x3) QRDecompose(*matTypes.QRResult) error {
	panic("Matrix3x3.QRDecompose: not supported in fixed matrix implementation")
}

func (m Matrix3x3) QR(*matTypes.QRResult) error {
	panic("Matrix3x3.QR: not supported in fixed matrix implementation")
}

func (m Matrix3x3) PseudoInverse(matTypes.Matrix) error {
	panic("Matrix3x3.PseudoInverse: not supported in fixed matrix implementation")
}

func (m Matrix3x3) DampedLeastSquares(float32, matTypes.Matrix) error {
	panic("Matrix3x3.DampedLeastSquares: not supported in fixed matrix implementation")
}

func (m Matrix3x3) SVD(*matTypes.SVDResult) error {
	panic("Matrix3x3.SVD: not supported in fixed matrix implementation")
}

func (m Matrix3x3) Quaternion() vecTypes.Quaternion {
	panic("Matrix3x3.Quaternion: unsupported operation")
}

func FromDiagonal3x3(d0, d1, d2 float32) Matrix3x3 {
	return Matrix3x3{{d0, 0, 0}, {0, d1, 0}, {0, 0, d2}}
}
