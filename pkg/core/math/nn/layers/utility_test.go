package layers

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFlatten tests the Flatten layer
func TestFlatten(t *testing.T) {
	tests := []struct {
		name         string
		input        tensor.Tensor
		startDim     int
		endDim       int
		expectedSize int
	}{
		{
			name: "2d_to_1d",
			input: tensor.Tensor{
				Dim:  []int{2, 3},
				Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			},
			startDim:     0,
			endDim:       2,
			expectedSize: 6,
		},
		{
			name: "4d_to_2d",
			input: tensor.Tensor{
				Dim:  []int{1, 2, 2, 2},
				Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			},
			startDim:     1,
			endDim:       4,
			expectedSize: 8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flatten := NewFlatten(tt.startDim, tt.endDim)
			err := flatten.Init(tt.input.Shape())
			require.NoError(t, err, "Init should succeed")

			output, err := flatten.Forward(tt.input)
			require.NoError(t, err, "Forward should succeed")
			assert.Len(t, output.Data, tt.expectedSize, "Output size should match")
			assert.Equal(t, tt.input.Data, output.Data, "Data should be preserved")
		})
	}
}

// TestReshape tests the Reshape layer
func TestReshape(t *testing.T) {
	reshape := NewReshape([]int{2, 4})

	inputTensor := tensor.Tensor{
		Dim:  []int{1, 8},
		Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
	}

	err := reshape.Init([]int{1, 8})
	require.NoError(t, err, "Init should succeed")

	output, err := reshape.Forward(inputTensor)
	require.NoError(t, err, "Forward should succeed")

	// Check output shape
	expectedShape := []int{2, 4}
	assert.Equal(t, expectedShape, output.Dim, "Output shape should match")

	// Check that data is preserved
	assert.Equal(t, inputTensor.Data, output.Data, "Data should be preserved")
}
