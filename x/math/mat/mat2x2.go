package mat

import (
	"fmt"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

type Matrix2x2 [2][2]float32

var _ matTypes.Matrix = Matrix2x2{}

func asMatrix2x2(arg matTypes.Matrix, op string) Matrix2x2 {
	switch v := arg.(type) {
	case Matrix2x2:
		return v
	case *Matrix2x2:
		return *v
	default:
		panic(fmt.Sprintf("Matrix2x2.%s: unsupported matrix type %T", op, arg))
	}
}

func (m Matrix2x2) View() matTypes.Matrix {
	return Matrix{
		[]float32{m[0][0], m[0][1]},
		[]float32{m[1][0], m[1][1]},
	}
}

func (m Matrix2x2) Rows() int { return 2 }
func (m Matrix2x2) Cols() int { return 2 }

func (m Matrix2x2) Rank() int {
	if m[0][0] == 0 && m[0][1] == 0 {
		if m[1][0] == 0 && m[1][1] == 0 {
			return 0
		}
		return 1
	}
	if m[1][0] == 0 && m[1][1] == 0 {
		return 1
	}
	if m.Det() == 0 {
		return 1
	}
	return 2
}

func (m Matrix2x2) CopyFrom(src matTypes.Matrix) {
	// noop
}
func (m Matrix2x2) IsContiguous() bool { return true }

func New2x2(arr ...float32) Matrix2x2 {
	var out Matrix2x2
	if len(arr) >= 4 {
		out[0][0], out[0][1] = arr[0], arr[1]
		out[1][0], out[1][1] = arr[2], arr[3]
	}
	return out
}

func (m Matrix2x2) Flat() []float32 {
	return []float32{m[0][0], m[0][1], m[1][0], m[1][1]}
}

func (m Matrix2x2) Rotation2D(a float32) matTypes.Matrix {
	c := math32.Cos(a)
	s := math32.Sin(a)
	return Matrix2x2{{c, -s}, {s, c}}
}

func (m Matrix2x2) Eye() matTypes.Matrix {
	return Matrix2x2{{1, 0}, {0, 1}}
}

func (m Matrix2x2) Row(row int) vecTypes.Vector {
	if row < 0 || row >= 2 {
		panic(fmt.Sprintf("Matrix2x2.Row: index %d out of range", row))
	}
	return vec.Vector2D{m[row][0], m[row][1]}
}

func (m Matrix2x2) Col(col int, _ vecTypes.Vector) vecTypes.Vector {
	if col < 0 || col >= 2 {
		panic(fmt.Sprintf("Matrix2x2.Col: index %d out of range", col))
	}
	return vec.Vector2D{m[0][col], m[1][col]}
}

func (m Matrix2x2) SetRow(row int, v vecTypes.Vector) matTypes.Matrix {
	if row < 0 || row >= 2 {
		panic(fmt.Sprintf("Matrix2x2.SetRow: index %d out of range", row))
	}
	vecVal := v.(vec.Vector2D)
	res := m
	res[row][0], res[row][1] = vecVal[0], vecVal[1]
	return res
}

func (m Matrix2x2) SetCol(col int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 2 {
		panic(fmt.Sprintf("Matrix2x2.SetCol: index %d out of range", col))
	}
	vecVal := v.(vec.Vector2D)
	res := m
	res[0][col], res[1][col] = vecVal[0], vecVal[1]
	return res
}

func (m Matrix2x2) SetColFromRow(col int, rowStart int, v vecTypes.Vector) matTypes.Matrix {
	if col < 0 || col >= 2 {
		panic(fmt.Sprintf("Matrix2x2.SetColFromRow: column %d out of range", col))
	}
	if rowStart < 0 || rowStart > 1 {
		panic(fmt.Sprintf("Matrix2x2.SetColFromRow: rowStart %d out of range", rowStart))
	}
	vecVal := v.(vec.Vector2D)
	res := m
	for i := 0; i+rowStart < 2 && i < len(vecVal); i++ {
		res[rowStart+i][col] = vecVal[i]
	}
	return res
}

func (m Matrix2x2) GetCol(col int, _ vecTypes.Vector) vecTypes.Vector {
	if col < 0 || col >= 2 {
		panic(fmt.Sprintf("Matrix2x2.GetCol: index %d out of range", col))
	}
	return vec.Vector2D{m[0][col], m[1][col]}
}

func (m Matrix2x2) Diagonal(_ vecTypes.Vector) vecTypes.Vector {
	return vec.Vector2D{m[0][0], m[1][1]}
}

func (m Matrix2x2) SetDiagonal(v vecTypes.Vector) matTypes.Matrix {
	vecVal := v.(vec.Vector2D)
	res := m
	res[0][0], res[1][1] = vecVal[0], vecVal[1]
	return res
}

func (m Matrix2x2) Submatrix(row, col int, _ matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return m
	}
	panic("Matrix2x2.Submatrix: unsupported submatrix dimensions")
}

func (m Matrix2x2) SetSubmatrix(row, col int, m1 matTypes.Matrix) matTypes.Matrix {
	if row == 0 && col == 0 {
		return asMatrix2x2(m1, "SetSubmatrix")
	}
	panic("Matrix2x2.SetSubmatrix: unsupported submatrix placement")
}

func (m Matrix2x2) SetSubmatrixRaw(row, col, rows1, cols1 int, m1 ...float32) matTypes.Matrix {
	if rows1 == 2 && cols1 == 2 && row == 0 && col == 0 && len(m1) >= 4 {
		return Matrix2x2{{m1[0], m1[1]}, {m1[2], m1[3]}}
	}
	panic("Matrix2x2.SetSubmatrixRaw: unsupported parameters")
}

func (m Matrix2x2) Clone() matTypes.Matrix {
	return m
}

func (m Matrix2x2) Transpose(m1 matTypes.Matrix) matTypes.Matrix {
	src := asMatrix2x2(m1, "Transpose")
	return Matrix2x2{{src[0][0], src[1][0]}, {src[0][1], src[1][1]}}
}

func (m Matrix2x2) Add(m1 matTypes.Matrix) matTypes.Matrix {
	other := asMatrix2x2(m1, "Add")
	return Matrix2x2{{m[0][0] + other[0][0], m[0][1] + other[0][1]}, {m[1][0] + other[1][0], m[1][1] + other[1][1]}}
}

func (m Matrix2x2) Sub(m1 matTypes.Matrix) matTypes.Matrix {
	other := asMatrix2x2(m1, "Sub")
	return Matrix2x2{{m[0][0] - other[0][0], m[0][1] - other[0][1]}, {m[1][0] - other[1][0], m[1][1] - other[1][1]}}
}

func (m Matrix2x2) MulC(c float32) matTypes.Matrix {
	return Matrix2x2{{m[0][0] * c, m[0][1] * c}, {m[1][0] * c, m[1][1] * c}}
}

func (m Matrix2x2) DivC(c float32) matTypes.Matrix {
	if c == 0 {
		panic("Matrix2x2.DivC: divide by zero")
	}
	inv := 1 / c
	return Matrix2x2{{m[0][0] * inv, m[0][1] * inv}, {m[1][0] * inv, m[1][1] * inv}}
}

func (m Matrix2x2) Mul(a matTypes.Matrix, b matTypes.Matrix) matTypes.Matrix {
	left := asMatrix2x2(a, "Mul.left")
	right := asMatrix2x2(b, "Mul.right")
	return Matrix2x2{
		{
			left[0][0]*right[0][0] + left[0][1]*right[1][0],
			left[0][0]*right[0][1] + left[0][1]*right[1][1],
		},
		{
			left[1][0]*right[0][0] + left[1][1]*right[1][0],
			left[1][0]*right[0][1] + left[1][1]*right[1][1],
		},
	}
}

func (m Matrix2x2) MulDiag(a matTypes.Matrix, b vecTypes.Vector) matTypes.Matrix {
	src := asMatrix2x2(a, "MulDiag")
	diag := b.(vec.Vector2D)
	return Matrix2x2{
		{src[0][0] * diag[0], src[0][1] * diag[1]},
		{src[1][0] * diag[0], src[1][1] * diag[1]},
	}
}

func (m Matrix2x2) MulVec(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector2D)
	return vec.Vector2D{
		m[0][0]*vecVal[0] + m[0][1]*vecVal[1],
		m[1][0]*vecVal[0] + m[1][1]*vecVal[1],
	}
}

func (m Matrix2x2) MulVecT(v vecTypes.Vector, _ vecTypes.Vector) vecTypes.Vector {
	vecVal := v.(vec.Vector2D)
	return vec.Vector2D{
		m[0][0]*vecVal[0] + m[1][0]*vecVal[1],
		m[0][1]*vecVal[0] + m[1][1]*vecVal[1],
	}
}

func (m Matrix2x2) Det() float32 {
	return m[0][0]*m[1][1] - m[0][1]*m[1][0]
}

func (m Matrix2x2) LU(_, _ matTypes.Matrix) {
	panic("Matrix2x2.LU: not supported in fixed matrix implementation")
}

func (m Matrix2x2) Cholesky(matTypes.Matrix) error {
	panic("Matrix2x2.Cholesky: not supported in fixed matrix implementation")
}

func (m Matrix2x2) CholeskySolve(vecTypes.Vector, vecTypes.Vector) error {
	panic("Matrix2x2.CholeskySolve: not supported in fixed matrix implementation")
}

func (m Matrix2x2) QRDecompose(*matTypes.QRResult) error {
	panic("Matrix2x2.QRDecompose: not supported in fixed matrix implementation")
}

func (m Matrix2x2) QR(*matTypes.QRResult) error {
	panic("Matrix2x2.QR: not supported in fixed matrix implementation")
}

func (m Matrix2x2) PseudoInverse(matTypes.Matrix) error {
	panic("Matrix2x2.PseudoInverse: not supported in fixed matrix implementation")
}

func (m Matrix2x2) DampedLeastSquares(float32, matTypes.Matrix) error {
	panic("Matrix2x2.DampedLeastSquares: not supported in fixed matrix implementation")
}

func (m Matrix2x2) SVD(*matTypes.SVDResult) error {
	panic("Matrix2x2.SVD: not supported in fixed matrix implementation")
}

func (m Matrix2x2) RotationX(float32) matTypes.Matrix {
	panic("Matrix2x2.RotationX: unsupported operation")
}

func (m Matrix2x2) RotationY(float32) matTypes.Matrix {
	panic("Matrix2x2.RotationY: unsupported operation")
}

func (m Matrix2x2) RotationZ(float32) matTypes.Matrix {
	panic("Matrix2x2.RotationZ: unsupported operation")
}

func (m Matrix2x2) Orientation(vecTypes.Quaternion) matTypes.Matrix {
	panic("Matrix2x2.Orientation: unsupported operation")
}

func (m Matrix2x2) Quaternion() vecTypes.Quaternion {
	panic("Matrix2x2.Quaternion: unsupported operation")
}
