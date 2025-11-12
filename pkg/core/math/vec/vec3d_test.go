package vec

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVector3D_AddDoesNotMutateReceiver(t *testing.T) {
	receiver := Vector3D{1, 2, 3}
	operand := NewFrom(4, 5, 6)

	got := receiver.Add(operand)

	result, ok := got.(Vector3D)
	if assert.True(t, ok, "Add should return a Vector3D") {
		assert.Equal(t, Vector3D{5, 7, 9}, result)
	}
	assert.Equal(t, Vector3D{1, 2, 3}, receiver)
}

func TestVector3D_CrossDoesNotMutatePointerReceiver(t *testing.T) {
	receiver := &Vector3D{1, 0, 0}
	operand := Vector3D{0, 1, 0}
	before := *receiver

	got := receiver.Cross(operand)

	result, ok := got.(Vector3D)
	if assert.True(t, ok, "Cross should return a Vector3D") {
		assert.Equal(t, Vector3D{0, 0, 1}, result)
	}
	assert.Equal(t, before, *receiver)
}

func TestVector3D_NormalProducesUnitVector(t *testing.T) {
	receiver := Vector3D{3, 0, 4}
	before := receiver

	got := receiver.Normal()

	result, ok := got.(Vector3D)
	if assert.True(t, ok, "Normal should return a Vector3D") {
		length := math.Sqrt(float64(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]))
		assert.InDelta(t, 1.0, length, 1e-6)
	}
	assert.Equal(t, before, receiver)
}
