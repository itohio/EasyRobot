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

// TestUnsqueeze tests the Unsqueeze layer
func TestUnsqueeze(t *testing.T) {
	t.Run("add_dim_at_beginning", func(t *testing.T) {
		unsqueeze := NewUnsqueeze(0)
		input := tensor.Tensor{
			Dim:  []int{3, 4},
			Data: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		}

		err := unsqueeze.Init([]int{3, 4})
		require.NoError(t, err)

		output, err := unsqueeze.Forward(input)
		require.NoError(t, err)
		assert.Equal(t, []int{1, 3, 4}, output.Dim)
		assert.Equal(t, input.Data, output.Data)
	})

	t.Run("add_dim_at_end", func(t *testing.T) {
		unsqueeze := NewUnsqueeze(-1)
		err := unsqueeze.Init([]int{3, 4})
		require.NoError(t, err)

		outputShape, err := unsqueeze.OutputShape([]int{3, 4})
		require.NoError(t, err)
		assert.Equal(t, []int{3, 4, 1}, outputShape)
	})
}

// TestSqueeze tests the Squeeze layer
func TestSqueeze(t *testing.T) {
	t.Run("squeeze_all", func(t *testing.T) {
		squeeze := NewSqueeze()
		input := tensor.Tensor{
			Dim:  []int{1, 3, 1, 4},
			Data: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		}

		err := squeeze.Init([]int{1, 3, 1, 4})
		require.NoError(t, err)

		output, err := squeeze.Forward(input)
		require.NoError(t, err)
		assert.Equal(t, []int{3, 4}, output.Dim)
		assert.Equal(t, input.Data, output.Data)
	})

	t.Run("squeeze_specific_dim", func(t *testing.T) {
		squeeze := NewSqueezeDims(0, 2)
		err := squeeze.Init([]int{1, 3, 1, 4})
		require.NoError(t, err)

		outputShape, err := squeeze.OutputShape([]int{1, 3, 1, 4})
		require.NoError(t, err)
		assert.Equal(t, []int{3, 4}, outputShape)
	})
}

// TestTranspose tests the Transpose layer
func TestTranspose(t *testing.T) {
	transpose := NewTranspose()
	input := tensor.Tensor{
		Dim:  []int{2, 3},
		Data: []float32{1, 2, 3, 4, 5, 6},
	}

	err := transpose.Init([]int{2, 3})
	require.NoError(t, err)

	output, err := transpose.Forward(input)
	require.NoError(t, err)
	assert.Equal(t, []int{3, 2}, output.Dim)
	assert.Equal(t, []float32{1, 4, 2, 5, 3, 6}, output.Data)

	// Test backward
	gradOutput := tensor.Tensor{
		Dim:  []int{3, 2},
		Data: []float32{1, 1, 1, 1, 1, 1},
	}
	gradInput, err := transpose.Backward(gradOutput)
	require.NoError(t, err)
	assert.Equal(t, []int{2, 3}, gradInput.Dim)
}
