package mat

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/require"
)

func TestMatrix2x2_AddDoesNotModifyReceiver(t *testing.T) {
	a := New2x2(1, 2, 3, 4)
	b := Matrix2x2{{5, 6}, {7, 8}}

	result, ok := a.Add(b).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{6, 8}, {10, 12}}, result)
	require.Equal(t, New2x2(1, 2, 3, 4), a)
}

func TestMatrix2x2_SubDoesNotModifyReceiver(t *testing.T) {
	a := New2x2(5, 7, 9, 11)
	b := Matrix2x2{{1, 2}, {3, 4}}

	result, ok := a.Sub(b).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{4, 5}, {6, 7}}, result)
	require.Equal(t, New2x2(5, 7, 9, 11), a)
}

func TestMatrix2x2_ScalarOps(t *testing.T) {
	a := New2x2(1, -2, 3, -4)

	mul, ok := a.MulC(2).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{2, -4}, {6, -8}}, mul)
	require.Equal(t, a, New2x2(1, -2, 3, -4))

	div, ok := a.DivC(-2).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{-0.5, 1}, {-1.5, 2}}, div)

	require.Panics(t, func() { _ = a.DivC(0) })
}

func TestMatrix2x2_Mul(t *testing.T) {
	a := New2x2(1, 2, 3, 4)
	b := Matrix2x2{{2, 0}, {1, 2}}

	result, ok := Matrix2x2{}.Mul(a, b).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{4, 4}, {10, 8}}, result)
}

func TestMatrix2x2_Rotation2D(t *testing.T) {
	angle := float32(math.Pi / 2)
	rot, ok := Matrix2x2{}.Rotation2D(angle).(Matrix2x2)
	require.True(t, ok)
	require.InDelta(t, 0, rot[0][0], 1e-6)
	require.InDelta(t, -1, rot[0][1], 1e-6)
	require.InDelta(t, 1, rot[1][0], 1e-6)
	require.InDelta(t, 0, rot[1][1], 1e-6)
}

func TestMatrix2x2_RowColReturnCopies(t *testing.T) {
	m := New2x2(1, 2, 3, 4)

	row := m.Row(1).(vec.Vector2D)
	col := m.Col(0, nil).(vec.Vector2D)

	require.Equal(t, vec.Vector2D{3, 4}, row)
	require.Equal(t, vec.Vector2D{1, 3}, col)

	// Mutating the returned vectors must not touch the matrix
	row[0] = 100
	col[1] = 200
	require.Equal(t, New2x2(1, 2, 3, 4), m)
}

func TestMatrix2x2_SetRowCol(t *testing.T) {
	m := New2x2(1, 2, 3, 4)

	updatedRow, ok := m.SetRow(1, vec.Vector2D{9, 8}).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{1, 2}, {9, 8}}, updatedRow)
	require.Equal(t, New2x2(1, 2, 3, 4), m)

	updatedCol, ok := m.SetCol(0, vec.Vector2D{7, 6}).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{7, 2}, {6, 4}}, updatedCol)
}

func TestMatrix2x2_Diagonal(t *testing.T) {
	m := New2x2(5, 6, 7, 8)
	diag := m.Diagonal(nil).(vec.Vector2D)
	require.Equal(t, vec.Vector2D{5, 8}, diag)

	updated, ok := m.SetDiagonal(vec.Vector2D{1, 2}).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{1, 6}, {7, 2}}, updated)
}

func TestMatrix2x2_MulVec(t *testing.T) {
	m := New2x2(1, 2, 3, 4)
	v := vec.Vector2D{2, 3}

	res := m.MulVec(v, nil).(vec.Vector2D)
	require.Equal(t, vec.Vector2D{8, 18}, res)

	resT := m.MulVecT(v, nil).(vec.Vector2D)
	require.Equal(t, vec.Vector2D{11, 16}, resT)
}

func TestMatrix2x2_DetAndRank(t *testing.T) {
	m := New2x2(1, 2, 2, 4)
	require.Equal(t, float32(0), m.Det())
	require.Equal(t, 1, m.Rank())

	id := Matrix2x2{{1, 0}, {0, 1}}
	require.Equal(t, float32(1), id.Det())
	require.Equal(t, 2, id.Rank())
}

func TestMatrix2x2_CloneValue(t *testing.T) {
	m := New2x2(1, 2, 3, 4)
	clone, ok := m.Clone().(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, m, clone)

	clone[0][0] = 100
	require.Equal(t, New2x2(1, 2, 3, 4), m)
}

func TestMatrix2x2_Transpose(t *testing.T) {
	src := New2x2(1, 2, 3, 4)
	res, ok := Matrix2x2{}.Transpose(src).(Matrix2x2)
	require.True(t, ok)
	require.Equal(t, Matrix2x2{{1, 3}, {2, 4}}, res)
}
