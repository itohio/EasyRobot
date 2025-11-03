package layers

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMaxPool2D tests the MaxPool2D layer
func TestMaxPool2D(t *testing.T) {
	pool, err := NewMaxPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err, "Should create MaxPool2D layer")

	inputTensor := tensor.Tensor{
		Dim: []int{1, 1, 4, 4},
		Data: []float32{
			1.0, 2.0, 3.0, 4.0,
			5.0, 6.0, 7.0, 8.0,
			9.0, 10.0, 11.0, 12.0,
			13.0, 14.0, 15.0, 16.0,
		},
	}

	err = pool.Init([]int{1, 1, 4, 4})
	require.NoError(t, err, "Init should succeed")

	output, err := pool.Forward(inputTensor)
	require.NoError(t, err, "Forward should succeed")

	// Expected output: max pooling 2x2 on 4x4 input should give 2x2 output
	expectedOutput := []float32{6.0, 8.0, 14.0, 16.0}
	assert.Equal(t, expectedOutput, output.Data, "Output should match expected")
}

// TestAvgPool2D tests the AvgPool2D layer
func TestAvgPool2D(t *testing.T) {
	pool, err := NewAvgPool2D(2, 2, 2, 2, 0, 0)
	require.NoError(t, err, "Should create AvgPool2D layer")

	inputTensor := tensor.Tensor{
		Dim: []int{1, 1, 4, 4},
		Data: []float32{
			1.0, 2.0, 3.0, 4.0,
			5.0, 6.0, 7.0, 8.0,
			9.0, 10.0, 11.0, 12.0,
			13.0, 14.0, 15.0, 16.0,
		},
	}

	err = pool.Init([]int{1, 1, 4, 4})
	require.NoError(t, err, "Init should succeed")

	output, err := pool.Forward(inputTensor)
	require.NoError(t, err, "Forward should succeed")

	// Expected output: avg pooling 2x2 on 4x4 input should give 2x2 output
	expectedOutput := []float32{3.5, 5.5, 11.5, 13.5}
	require.Len(t, output.Data, len(expectedOutput), "Output size should match")
	for i := range expectedOutput {
		assert.InDelta(t, expectedOutput[i], output.Data[i], 1e-6, "Output[%d] should match", i)
	}
}

// TestGlobalAvgPool2D tests the GlobalAvgPool2D layer
func TestGlobalAvgPool2D(t *testing.T) {
	pool := NewGlobalAvgPool2D()

	inputTensor := tensor.Tensor{
		Dim: []int{1, 2, 3, 3},
		Data: []float32{
			// Channel 1: 3x3
			1.0, 2.0, 3.0,
			4.0, 5.0, 6.0,
			7.0, 8.0, 9.0,
			// Channel 2: 3x3
			10.0, 11.0, 12.0,
			13.0, 14.0, 15.0,
			16.0, 17.0, 18.0,
		},
	}

	err := pool.Init([]int{1, 2, 3, 3})
	require.NoError(t, err, "Init should succeed")

	output, err := pool.Forward(inputTensor)
	require.NoError(t, err, "Forward should succeed")

	// Expected output: global average of each channel [1, 2] -> [5.0, 14.0]
	expectedOutput := []float32{5.0, 14.0}
	require.Len(t, output.Data, len(expectedOutput), "Output size should match")
	for i := range expectedOutput {
		assert.InDelta(t, expectedOutput[i], output.Data[i], 1e-6, "Output[%d] should match", i)
	}
}
