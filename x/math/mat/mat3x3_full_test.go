package mat

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	"github.com/stretchr/testify/require"
)

func TestMatrix3x3_AddDoesNotMutateReceiver(t *testing.T) {
	a := New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
	b := Matrix3x3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}

	result, ok := a.Add(b).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, Matrix3x3{{2, 2, 3}, {4, 6, 6}, {7, 8, 10}}, result)
	require.Equal(t, New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9), a)
}

func TestMatrix3x3_SubDoesNotMutateReceiver(t *testing.T) {
	a := New3x3(5, 6, 7, 8, 9, 10, 11, 12, 13)
	b := Matrix3x3{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}

	result, ok := a.Sub(b).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, Matrix3x3{{4, 5, 6}, {6, 7, 8}, {8, 9, 10}}, result)
	require.Equal(t, New3x3(5, 6, 7, 8, 9, 10, 11, 12, 13), a)
}

func TestMatrix3x3_ScalarOps(t *testing.T) {
	a := New3x3(1, -2, 3, -4, 5, -6, 7, -8, 9)

	mul, ok := a.MulC(3).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, Matrix3x3{{3, -6, 9}, {-12, 15, -18}, {21, -24, 27}}, mul)
	require.Equal(t, a, New3x3(1, -2, 3, -4, 5, -6, 7, -8, 9))

	div, ok := a.DivC(-2).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, Matrix3x3{{-0.5, 1, -1.5}, {2, -2.5, 3}, {-3.5, 4, -4.5}}, div)
	require.Panics(t, func() { _ = a.DivC(0) })
}

func TestMatrix3x3_Mul(t *testing.T) {
	left := New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
	right := Matrix3x3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}

	result, ok := Matrix3x3{}.Mul(left, right).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, left, result)
}

func TestMatrix3x3_Rotation(t *testing.T) {
	rotX := Matrix3x3{}.RotationX(math.Pi / 2).(Matrix3x3)
	require.InDelta(t, 1, rotX[0][0], 1e-6)
	require.InDelta(t, 0, rotX[1][1], 1e-6)

	rotY := Matrix3x3{}.RotationY(math.Pi / 2).(Matrix3x3)
	require.InDelta(t, 0, rotY[0][0], 1e-6)

	rotZ := Matrix3x3{}.RotationZ(math.Pi / 2).(Matrix3x3)
	require.InDelta(t, 0, rotZ[0][0], 1e-6)
	require.InDelta(t, -1, rotZ[0][1], 1e-6)
	require.InDelta(t, 1, rotZ[1][0], 1e-6)
}

func TestMatrix3x3_RowAndColReturnCopies(t *testing.T) {
	m := New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
	row := m.Row(1).(vec.Vector3D)
	col := m.Col(2, nil).(vec.Vector3D)

	require.Equal(t, vec.Vector3D{4, 5, 6}, row)
	require.Equal(t, vec.Vector3D{3, 6, 9}, col)

	row[0] = 100
	col[1] = 200
	require.Equal(t, New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9), m)
}

func TestMatrix3x3_SettersReturnNewValues(t *testing.T) {
	m := New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)

	setRow, ok := m.SetRow(0, vec.Vector3D{9, 8, 7}).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, Matrix3x3{{9, 8, 7}, {4, 5, 6}, {7, 8, 9}}, setRow)

	setCol, ok := m.SetCol(2, vec.Vector3D{1, 2, 3}).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, Matrix3x3{{1, 2, 1}, {4, 5, 2}, {7, 8, 3}}, setCol)

	setDiag, ok := m.SetDiagonal(vec.Vector3D{3, 2, 1}).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, Matrix3x3{{3, 2, 3}, {4, 2, 6}, {7, 8, 1}}, setDiag)
}

func TestMatrix3x3_Diagonal(t *testing.T) {
	m := New3x3(5, 6, 7, 8, 9, 10, 11, 12, 13)
	diag := m.Diagonal(nil).(vec.Vector3D)
	require.Equal(t, vec.Vector3D{5, 9, 13}, diag)
}

func TestMatrix3x3_MulVec(t *testing.T) {
	m := New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
	v := vec.Vector3D{1, 0, -1}

	res := m.MulVec(v, nil).(vec.Vector3D)
	require.Equal(t, vec.Vector3D{-2, -2, -2}, res)

	resT := m.MulVecT(v, nil).(vec.Vector3D)
	require.Equal(t, vec.Vector3D{-6, -6, -6}, resT)
}

func TestMatrix3x3_DetAndRank(t *testing.T) {
	ident := Matrix3x3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	require.Equal(t, float32(1), ident.Det())
	require.Equal(t, 3, ident.Rank())

	singular := New3x3(1, 2, 3, 2, 4, 6, 3, 6, 9)
	require.Equal(t, float32(0), singular.Det())
	require.Equal(t, 1, singular.Rank())
}

func TestMatrix3x3_CloneValue(t *testing.T) {
	m := New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
	clone, ok := m.Clone().(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, m, clone)

	clone[0][0] = 99
	require.Equal(t, New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9), m)
}

func TestMatrix3x3_Transpose(t *testing.T) {
	src := New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
	res, ok := Matrix3x3{}.Transpose(src).(Matrix3x3)
	require.True(t, ok)
	require.Equal(t, Matrix3x3{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}, res)
}

func TestMatrix3x3_Orientation(t *testing.T) {
	q := vec.Quaternion{0, 0, math32.Sin(math32.Pi / 4), math32.Cos(math32.Pi / 4)}
	rot := Matrix3x3{}.Orientation(q).(Matrix3x3)
	require.InDelta(t, 0, rot[0][2], 1e-6)
	require.InDelta(t, 0, rot[1][2], 1e-6)
	require.InDelta(t, 1, rot[2][2], 1e-6)
}
