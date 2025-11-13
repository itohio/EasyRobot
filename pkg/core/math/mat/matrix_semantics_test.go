package mat

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMatrix_AddMutatesBacking(t *testing.T) {
	backing := []float32{1, 2, 3, 4}
	m := New(2, 2, backing...)
	addend := New(2, 2, 1, 0, 0, 1)

	got := m.Add(addend)

	result, ok := got.(Matrix)
	if assert.True(t, ok, "Add should return a Matrix") {
		assert.Equal(t, Matrix{{2, 2}, {3, 5}}, result)
	}
	assert.Equal(t, []float32{2, 2, 3, 5}, backing)
}

func TestMatrix_CloneIsIndependent(t *testing.T) {
	m := Matrix{
		{1, 2},
		{3, 4},
	}

	clonedAny := m.Clone()

	cloned, ok := clonedAny.(Matrix)
	if assert.True(t, ok, "Clone should return a Matrix") {
		cloned[0][0] = 99
		assert.Equal(t, float32(1), m[0][0])
		assert.Equal(t, float32(99), cloned[0][0])
	}
}

func TestMatrix_DimensionsAndRank(t *testing.T) {
	generic := Matrix{
		{1, 2, 3},
		{2, 4, 6},
		{0, 0, 1},
	}

	assert.Equal(t, 3, generic.Rows())
	assert.Equal(t, 3, generic.Cols())
	assert.Equal(t, 2, generic.Rank())

	rect := Matrix{
		{1, 2, 3, 4},
		{2, 4, 6, 8},
	}
	assert.Equal(t, 2, rect.Rows())
	assert.Equal(t, 4, rect.Cols())
	assert.Equal(t, 1, rect.Rank())

	zero := Matrix{
		{0, 0},
		{0, 0},
	}
	assert.Equal(t, 0, zero.Rank())
}
