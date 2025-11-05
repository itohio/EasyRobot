package layers

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMaxPool2D_EdgeCases tests edge cases for MaxPool2D layer
func TestMaxPool2D_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilPool *MaxPool2D
	_, err := nilPool.Forward(tensor.FromFloat32(tensor.NewShape(1, 1, 4, 4), make([]float32, 16)))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	pool, err := NewMaxPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err)
	err = pool.Init([]int{1, 1, 4, 4})
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = pool.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	pool2, err := NewMaxPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err)
	input := tensor.FromFloat32(tensor.NewShape(1, 1, 4, 4), make([]float32, 16))
	_, err = pool2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward without Forward
	pool3, err := NewMaxPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err)
	err = pool3.Init([]int{1, 1, 4, 4})
	require.NoError(t, err)
	gradOutput := tensor.FromFloat32(tensor.NewShape(1, 1, 2, 2), make([]float32, 4))
	_, err = pool3.Backward(gradOutput)
	assert.Error(t, err, "Backward should error if Forward not called")

	// Test Backward with empty gradOutput
	pool4, err := NewMaxPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err)
	err = pool4.Init([]int{1, 1, 4, 4})
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(1, 1, 4, 4), make([]float32, 16))
	_, err = pool4.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = pool4.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
}

// TestAvgPool2D_EdgeCases tests edge cases for AvgPool2D layer
func TestAvgPool2D_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilPool *AvgPool2D
	_, err := nilPool.Forward(tensor.FromFloat32(tensor.NewShape(1, 1, 4, 4), make([]float32, 16)))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	pool, err := NewAvgPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err)
	err = pool.Init([]int{1, 1, 4, 4})
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = pool.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	pool2, err := NewAvgPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err)
	input := tensor.FromFloat32(tensor.NewShape(1, 1, 4, 4), make([]float32, 16))
	_, err = pool2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward without Forward
	pool3, err := NewAvgPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err)
	err = pool3.Init([]int{1, 1, 4, 4})
	require.NoError(t, err)
	gradOutput2 := tensor.FromFloat32(tensor.NewShape(1, 1, 2, 2), make([]float32, 4))
	_, err = pool3.Backward(gradOutput2)
	assert.Error(t, err, "Backward should error if Forward not called")

	// Test Backward with empty gradOutput
	pool4, err := NewAvgPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err)
	err = pool4.Init([]int{1, 1, 4, 4})
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(1, 1, 4, 4), make([]float32, 16))
	_, err = pool4.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = pool4.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
}

// TestGlobalAvgPool2D_EdgeCases tests edge cases for GlobalAvgPool2D layer
func TestGlobalAvgPool2D_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilPool *GlobalAvgPool2D
	_, err := nilPool.Forward(tensor.FromFloat32(tensor.NewShape(1, 2, 3, 3), make([]float32, 18)))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	pool := NewGlobalAvgPool2D()
	err = pool.Init([]int{1, 2, 3, 3})
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = pool.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	pool2 := NewGlobalAvgPool2D()
	input := tensor.FromFloat32(tensor.NewShape(1, 2, 3, 3), make([]float32, 18))
	_, err = pool2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward without Forward
	pool3 := NewGlobalAvgPool2D()
	err = pool3.Init([]int{1, 2, 3, 3})
	require.NoError(t, err)
	gradOutput3 := tensor.FromFloat32(tensor.NewShape(1, 2), make([]float32, 2))
	_, err = pool3.Backward(gradOutput3)
	assert.Error(t, err, "Backward should error if Forward not called")

	// Test Backward with empty gradOutput
	pool4 := NewGlobalAvgPool2D()
	err = pool4.Init([]int{1, 2, 3, 3})
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(1, 2, 3, 3), make([]float32, 18))
	_, err = pool4.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = pool4.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
}
