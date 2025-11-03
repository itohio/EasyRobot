package layers

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestReLU tests the ReLU activation layer
func TestReLU(t *testing.T) {
	tests := []struct {
		name      string
		input     []float32
		output    []float32
		gradInput []float32
	}{
		{
			name:      "basic_positive",
			input:     []float32{1.0, 2.0, 3.0},
			output:    []float32{1.0, 2.0, 3.0},
			gradInput: []float32{1.0, 1.0, 1.0},
		},
		{
			name:      "basic_negative",
			input:     []float32{-1.0, -2.0, -3.0},
			output:    []float32{0.0, 0.0, 0.0},
			gradInput: []float32{0.0, 0.0, 0.0},
		},
		{
			name:      "mixed",
			input:     []float32{-1.0, 0.0, 1.0, -2.0, 2.0},
			output:    []float32{0.0, 0.0, 1.0, 0.0, 2.0},
			gradInput: []float32{0.0, 0.0, 1.0, 0.0, 1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			relu := NewReLU("relu")
			inputTensor := tensor.Tensor{
				Dim:  []int{len(tt.input)},
				Data: tt.input,
			}

			// Test Init
			err := relu.Init([]int{len(tt.input)})
			require.NoError(t, err, "Init should succeed")

			// Test Forward
			output, err := relu.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			assert.Equal(t, tt.output, output.Data, "Output should match expected")

			// Test Backward
			gradOutput := tensor.Tensor{
				Dim:  []int{len(tt.input)},
				Data: []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
			}
			gradInput, err := relu.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")
			assert.Equal(t, tt.gradInput, gradInput.Data, "GradInput should match expected")
		})
	}
}

// TestSigmoid tests the Sigmoid activation layer
func TestSigmoid(t *testing.T) {
	tests := []struct {
		name   string
		input  float32
		output float32
	}{
		{"zero", 0.0, 0.5},
		{"positive", 1.0, 0.7310585786300049},
		{"negative", -1.0, 0.2689414213699951},
		{"large_positive", 10.0, 0.9999546021312976},
		{"large_negative", -10.0, 0.0000453978687024234},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sigmoid := NewSigmoid("sigmoid")
			inputTensor := tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{tt.input},
			}

			err := sigmoid.Init([]int{1})
			require.NoError(t, err, "Init should succeed")

			output, err := sigmoid.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			assert.InDelta(t, tt.output, output.Data[0], 1e-6, "Output should match expected")

			// Test Backward
			gradOutput := tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{1.0},
			}
			gradInput, err := sigmoid.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")

			// Sigmoid derivative: output * (1 - output)
			expectedGrad := output.Data[0] * (1 - output.Data[0])
			assert.InDelta(t, expectedGrad, gradInput.Data[0], 1e-6, "GradInput should match expected derivative")
		})
	}
}

// TestTanh tests the Tanh activation layer
func TestTanh(t *testing.T) {
	tests := []struct {
		name   string
		input  float32
		output float32
	}{
		{"zero", 0.0, 0.0},
		{"positive", 1.0, 0.7615941559557649},
		{"negative", -1.0, -0.7615941559557649},
		{"large_positive", 10.0, 0.999999999},
		{"large_negative", -10.0, -0.999999999},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tanh := NewTanh("tanh")
			inputTensor := tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{tt.input},
			}

			err := tanh.Init([]int{1})
			require.NoError(t, err, "Init should succeed")

			output, err := tanh.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			assert.InDelta(t, tt.output, output.Data[0], 1e-6, "Output should match expected")

			// Test Backward
			gradOutput := tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{1.0},
			}
			gradInput, err := tanh.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")

			// Tanh derivative: 1 - output^2
			expectedGrad := 1 - output.Data[0]*output.Data[0]
			assert.InDelta(t, expectedGrad, gradInput.Data[0], 1e-6, "GradInput should match expected derivative")
		})
	}
}

// TestSoftmax tests the Softmax activation layer
func TestSoftmax(t *testing.T) {
	softmax := NewSoftmax("softmax", 0)
	inputTensor := tensor.Tensor{
		Dim:  []int{3},
		Data: []float32{1.0, 2.0, 3.0},
	}

	err := softmax.Init([]int{3})
	require.NoError(t, err, "Init should succeed")

	output, err := softmax.Forward(inputTensor)
	require.NoError(t, err, "Forward should succeed")

	// Check that outputs sum to 1.0
	sum := float32(0)
	for _, v := range output.Data {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 1e-6, "Softmax outputs should sum to 1")

	// Check that all outputs are non-negative
	for i, v := range output.Data {
		assert.GreaterOrEqual(t, v, float32(0), "Output[%d] should be non-negative", i)
	}

	// Test Backward
	gradOutput := tensor.Tensor{
		Dim:  []int{3},
		Data: []float32{0.1, 0.2, 0.3},
	}
	gradInput, err := softmax.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed")
	assert.Len(t, gradInput.Data, 3, "GradInput should have size 3")
}
