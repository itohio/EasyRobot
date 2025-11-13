package mat

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	"github.com/stretchr/testify/require"
)

func TestMatrix4x4_AddDoesNotMutateReceiver(t *testing.T) {
	a := New4x4(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	)
	b := Matrix4x4{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}

	result, ok := a.Add(b).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{2, 2, 3, 4}, {5, 7, 7, 8}, {9, 10, 12, 12}, {13, 14, 15, 17}}, result)
	require.Equal(t, New4x4(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	), a)
}

func TestMatrix4x4_SubDoesNotMutateReceiver(t *testing.T) {
	a := New4x4(
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
		17, 18, 19, 20,
	)
	b := Matrix4x4{{1, 2, 3, 4}, {4, 3, 2, 1}, {0, 1, 0, 1}, {1, 0, 1, 0}}

	result, ok := a.Sub(b).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{4, 4, 4, 4}, {5, 7, 9, 11}, {13, 13, 15, 15}, {16, 18, 18, 20}}, result)
	require.Equal(t, New4x4(
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
		17, 18, 19, 20,
	), a)
}

func TestMatrix4x4_ScalarOps(t *testing.T) {
	a := New4x4(
		1, -2, 3, -4,
		5, -6, 7, -8,
		9, -10, 11, -12,
		13, -14, 15, -16,
	)

	mul, ok := a.MulC(2).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{2, -4, 6, -8}, {10, -12, 14, -16}, {18, -20, 22, -24}, {26, -28, 30, -32}}, mul)

	div, ok := a.DivC(-2).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{-0.5, 1, -1.5, 2}, {-2.5, 3, -3.5, 4}, {-4.5, 5, -5.5, 6}, {-6.5, 7, -7.5, 8}}, div)

	require.Panics(t, func() { _ = a.DivC(0) })
}

func TestMatrix4x4_Mul(t *testing.T) {
	left := New4x4(
		1, 2, 3, 4,
		0, 1, 2, 3,
		0, 0, 1, 2,
		0, 0, 0, 1,
	)
	right := Matrix4x4{{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 2}}

	result, ok := Matrix4x4{}.Mul(left, right).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{2, 4, 6, 8}, {0, 2, 4, 6}, {0, 0, 2, 4}, {0, 0, 0, 2}}, result)
}

func TestMatrix4x4_RowColReturnCopies(t *testing.T) {
	m := New4x4(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	)

	row := m.Row(2).(vec.Vector4D)
	col := m.Col(1, nil).(vec.Vector4D)
	require.Equal(t, vec.Vector4D{9, 10, 11, 12}, row)
	require.Equal(t, vec.Vector4D{2, 6, 10, 14}, col)

	row[0] = 100
	col[1] = 200
	require.Equal(t, New4x4(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	), m)
}

func TestMatrix4x4_SettersReturnNewValues(t *testing.T) {
	m := New4x4(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	)

	setRow, ok := m.SetRow(0, vec.Vector4D{16, 15, 14, 13}).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{16, 15, 14, 13}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, setRow)

	setCol, ok := m.SetCol(3, vec.Vector4D{1, 2, 3, 4}).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{1, 2, 3, 1}, {5, 6, 7, 2}, {9, 10, 11, 3}, {13, 14, 15, 4}}, setCol)

	setDiag, ok := m.SetDiagonal(vec.Vector4D{4, 3, 2, 1}).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{4, 2, 3, 4}, {5, 3, 7, 8}, {9, 10, 2, 12}, {13, 14, 15, 1}}, setDiag)
}

func TestMatrix4x4_Diagonal(t *testing.T) {
	m := New4x4(
		5, 6, 7, 8,
		1, 2, 3, 4,
		9, 10, 11, 12,
		13, 14, 15, 16,
	)
	diag := m.Diagonal(nil).(vec.Vector4D)
	require.Equal(t, vec.Vector4D{5, 2, 11, 16}, diag)
}

func TestMatrix4x4_MulVec(t *testing.T) {
	m := New4x4(
		1, 2, 3, 4,
		0, 1, 0, 1,
		1, 0, 1, 0,
		0, 0, 0, 1,
	)
	v := vec.Vector4D{1, 2, 3, 4}

	res := m.MulVec(v, nil).(vec.Vector4D)
	require.Equal(t, vec.Vector4D{30, 6, 4, 4}, res)

	resT := m.MulVecT(v, nil).(vec.Vector4D)
	require.Equal(t, vec.Vector4D{4, 4, 6, 10}, resT)
}

func TestMatrix4x4_DetAndRank(t *testing.T) {
	ident := Matrix4x4{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
	require.Equal(t, float32(1), ident.Det())
	require.Equal(t, 4, ident.Rank())

	upper := New4x4(
		1, 2, 3, 4,
		0, 1, 2, 3,
		0, 0, 1, 2,
		0, 0, 0, 1,
	)
	require.Equal(t, float32(1), upper.Det())
	require.Equal(t, 4, upper.Rank())

	singular := New4x4(
		1, 2, 3, 4,
		2, 4, 6, 8,
		3, 6, 9, 12,
		4, 8, 12, 16,
	)
	require.Equal(t, float32(0), singular.Det())
	require.LessOrEqual(t, singular.Rank(), 1)
}

func TestMatrix4x4_LUProducesFactors(t *testing.T) {
	m := New4x4(
		4, 3, 2, 1,
		3, 3, 2, 1,
		2, 2, 2, 1,
		1, 1, 1, 1,
	)
	var L, U Matrix4x4
	m.LU(&L, &U)

	product, ok := Matrix4x4{}.Mul(L, U).(Matrix4x4)
	require.True(t, ok)
	require.InDeltaSlice(t, m.Flat(), product.Flat(), 1e-4)
	for i := 0; i < 4; i++ {
		require.InDelta(t, 1.0, L[i][i], 1e-5)
	}
}

func TestMatrix4x4_CloneValue(t *testing.T) {
	m := New4x4(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	)
	clone, ok := m.Clone().(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, m, clone)

	clone[0][0] = 99
	require.NotEqual(t, clone, m)
}

func TestMatrix4x4_Transpose(t *testing.T) {
	src := New4x4(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	)
	transposed, ok := Matrix4x4{}.Transpose(src).(Matrix4x4)
	require.True(t, ok)
	require.Equal(t, Matrix4x4{{1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}, {4, 8, 12, 16}}, transposed)
}

func TestMatrix4x4_RotationZ(t *testing.T) {
	angle := math.Pi / 2
	rot, ok := Matrix4x4{}.RotationZ(float32(angle)).(Matrix4x4)
	require.True(t, ok)
	require.InDelta(t, 0, rot[0][0], 1e-6)
	require.InDelta(t, -1, rot[0][1], 1e-6)
	require.InDelta(t, 1, rot[1][0], 1e-6)
	require.InDelta(t, 0, rot[1][1], 1e-6)
}

func TestMatrix4x4_Orientation(t *testing.T) {
	q := vec.Quaternion{0, 0, math32.Sin(math32.Pi / 4), math32.Cos(math32.Pi / 4)}
	oriented, ok := Matrix4x4{}.Orientation(q).(Matrix4x4)
	require.True(t, ok)

	rot := Matrix3x3{}.RotationZ(math32.Pi / 2).(Matrix3x3)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			require.InDelta(t, rot[i][j], oriented[i][j], 1e-5)
		}
	}
}
