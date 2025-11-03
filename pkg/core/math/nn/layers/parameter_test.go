package layers_test

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDenseParameterAccess(t *testing.T) {
	// Create a Dense layer
	dense, err := layers.NewDense(4, 2, layers.WithCanLearn(true))
	require.NoError(t, err, "Should create Dense layer")

	// Test getting initial weights
	weight := dense.Weight()
	assert.Len(t, weight.Dim, 2, "Weight should be 2D")
	assert.Equal(t, []int{4, 2}, weight.Dim, "Weight shape should be [4, 2]")

	// Test getting initial bias
	bias := dense.Bias()
	assert.Len(t, bias.Dim, 1, "Bias should be 1D")
	assert.Equal(t, []int{2}, bias.Dim, "Bias shape should be [2]")

	// Test setting new weight
	newWeight := tensor.Tensor{
		Dim:  []int{4, 2},
		Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
	}
	err = dense.SetWeight(newWeight)
	require.NoError(t, err, "Should set weight successfully")

	// Verify weight was set
	weight = dense.Weight()
	assert.Equal(t, newWeight.Data, weight.Data, "Weight data should match")

	// Test setting new bias
	newBias := tensor.Tensor{
		Dim:  []int{2},
		Data: []float32{0.5, -0.5},
	}
	err = dense.SetBias(newBias)
	require.NoError(t, err, "Should set bias successfully")

	// Verify bias was set
	bias = dense.Bias()
	assert.Equal(t, newBias.Data, bias.Data, "Bias data should match")

	// Test validation errors
	err = dense.SetWeight(tensor.Tensor{Dim: []int{4, 3}, Data: make([]float32, 12)})
	assert.Error(t, err, "Should return error for wrong weight shape")

	err = dense.SetBias(tensor.Tensor{Dim: []int{3}, Data: []float32{0, 0, 0}})
	assert.Error(t, err, "Should return error for wrong bias shape")
}

func TestConv2DParameterAccess(t *testing.T) {
	// Create a Conv2D layer
	conv2d, err := layers.NewConv2D(3, 16, 3, 3, 1, 1, 1, 1, layers.WithCanLearn(true))
	require.NoError(t, err, "Should create Conv2D layer")

	// Test getting initial weights
	weight := conv2d.Weight()
	assert.Len(t, weight.Dim, 4, "Weight should be 4D")
	assert.Equal(t, []int{16, 3, 3, 3}, weight.Dim, "Weight shape should match")

	// Test getting initial bias
	bias := conv2d.Bias()
	assert.Len(t, bias.Dim, 1, "Bias should be 1D")
	assert.Equal(t, []int{16}, bias.Dim, "Bias shape should be [16]")

	// Test setting new weight
	newWeight := tensor.Tensor{
		Dim:  []int{16, 3, 3, 3},
		Data: make([]float32, 16*3*3*3),
	}
	err = conv2d.SetWeight(newWeight)
	require.NoError(t, err, "Should set weight successfully")

	// Test validation errors
	err = conv2d.SetWeight(tensor.Tensor{Dim: []int{16, 4, 3, 3}, Data: make([]float32, 576)})
	assert.Error(t, err, "Should return error for wrong weight shape")
}

func TestConv1DParameterAccess(t *testing.T) {
	// Create a Conv1D layer
	conv1d, err := layers.NewConv1D(3, 16, 3, 1, 1, layers.WithCanLearn(true))
	require.NoError(t, err, "Should create Conv1D layer")

	// Test getting initial weights
	weight := conv1d.Weight()
	assert.Len(t, weight.Dim, 3, "Weight should be 3D")
	assert.Equal(t, []int{16, 3, 3}, weight.Dim, "Weight shape should match")

	// Test getting initial bias
	bias := conv1d.Bias()
	assert.Len(t, bias.Dim, 1, "Bias should be 1D")
	assert.Equal(t, []int{16}, bias.Dim, "Bias shape should be [16]")

	// Test setting new weight
	newWeight := tensor.Tensor{
		Dim:  []int{16, 3, 3},
		Data: make([]float32, 16*3*3),
	}
	err = conv1d.SetWeight(newWeight)
	require.NoError(t, err, "Should set weight successfully")

	// Test validation errors
	err = conv1d.SetWeight(tensor.Tensor{Dim: []int{16, 4, 3}, Data: make([]float32, 192)})
	assert.Error(t, err, "Should return error for wrong weight shape")
}

func TestParameterAccessWithBias(t *testing.T) {
	// Create a Dense layer (bias is always created)
	dense, err := layers.NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	// Test getting bias should return non-empty tensor
	bias := dense.Bias()
	assert.Len(t, bias.Dim, 1, "Bias should have dimensions")
	assert.Equal(t, []int{2}, bias.Dim, "Bias shape should match")

	// Test setting bias should succeed
	newBias := tensor.Tensor{
		Dim:  []int{2},
		Data: []float32{0.5, -0.5},
	}
	err = dense.SetBias(newBias)
	require.NoError(t, err, "Should succeed setting bias")

	// Verify bias was set
	bias = dense.Bias()
	assert.Equal(t, newBias.Data, bias.Data, "Bias data should match")
}
