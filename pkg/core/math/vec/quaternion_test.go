package vec

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQuaternion_AddDoesNotMutatePointerReceiver(t *testing.T) {
	receiver := &Quaternion{1, 2, 3, 4}
	operand := NewFrom(1, 1, 1, 1)
	before := *receiver

	got := receiver.Add(operand)

	result, ok := got.(Quaternion)
	if assert.True(t, ok, "Add should return a Quaternion") {
		assert.Equal(t, Quaternion{2, 3, 4, 5}, result)
	}
	assert.Equal(t, before, *receiver)
}

func TestQuaternion_ProductIdentity(t *testing.T) {
	receiver := Quaternion{0, 0, 0, 1}
	operand := Quaternion{1, 2, 3, 4}

	got := receiver.Product(operand)

	result, ok := got.(Quaternion)
	if assert.True(t, ok, "Product should return a Quaternion") {
		assert.Equal(t, operand, result)
	}
	assert.Equal(t, Quaternion{0, 0, 0, 1}, receiver)
}

func TestQuaternion_NormalProducesUnitQuaternion(t *testing.T) {
	receiver := Quaternion{0, 3, 0, 4}
	before := receiver

	got := receiver.Normal()

	result, ok := got.(Quaternion)
	if assert.True(t, ok, "Normal should return a Quaternion") {
		length := math.Sqrt(float64(result[0]*result[0] + result[1]*result[1] + result[2]*result[2] + result[3]*result[3]))
		assert.InDelta(t, 1.0, length, 1e-6)
	}
	assert.Equal(t, before, receiver)
}
