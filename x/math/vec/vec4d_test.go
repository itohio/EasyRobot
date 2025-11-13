package vec

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVector4D_AddDoesNotMutateReceiver(t *testing.T) {
	receiver := Vector4D{1, 2, 3, 4}
	operand := Vector4D{4, 5, 6, 7}

	got := receiver.Add(operand)

	result, ok := got.(Vector4D)
	if assert.True(t, ok, "Add should return a Vector4D") {
		assert.Equal(t, Vector4D{5, 7, 9, 11}, result)
	}
	assert.Equal(t, Vector4D{1, 2, 3, 4}, receiver)
}

func TestVector4D_NormalProducesUnitVector(t *testing.T) {
	receiver := Vector4D{0, 3, 0, 4}
	before := receiver

	got := receiver.Normal()

	result, ok := got.(Vector4D)
	if assert.True(t, ok, "Normal should return a Vector4D") {
		length := math.Sqrt(float64(result[0]*result[0] + result[1]*result[1] + result[2]*result[2] + result[3]*result[3]))
		assert.InDelta(t, 1.0, length, 1e-6)
	}
	assert.Equal(t, before, receiver)
}

func TestVector4D_AxisReturnsFirstThreeComponents(t *testing.T) {
	receiver := Vector4D{1, 2, 3, 4}

	axis := receiver.Axis()

	axisVec, ok := axis.(Vector)
	if assert.True(t, ok, "Axis should return a Vector slice") {
		assert.Equal(t, Vector{1, 2, 3}, axisVec)
		axisVec[0] = 99
	}
	assert.Equal(t, Vector4D{1, 2, 3, 4}, receiver)
}
