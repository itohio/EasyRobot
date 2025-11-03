package layers

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewDense(t *testing.T) {
	tests := []struct {
		name           string
		inFeatures     int
		outFeatures    int
		opts           []interface{}
		expectError    bool
		expectBias     bool
		expectCanLearn bool
	}{
		{
			name:           "basic",
			inFeatures:     4,
			outFeatures:    2,
			expectError:    false,
			expectBias:     true,
			expectCanLearn: false,
		},
		{
			name:           "no_bias",
			inFeatures:     4,
			outFeatures:    2,
			opts:           []interface{}{WithDenseBias(false)},
			expectError:    false,
			expectBias:     false,
			expectCanLearn: false,
		},
		{
			name:           "with_can_learn",
			inFeatures:     4,
			outFeatures:    2,
			opts:           []interface{}{WithCanLearn(true)},
			expectError:    false,
			expectBias:     true,
			expectCanLearn: true,
		},
		{
			name:        "with_name",
			inFeatures:  4,
			outFeatures: 2,
			opts:        []interface{}{WithName("my_dense")},
			expectError: false,
			expectBias:  true,
		},
		{
			name:        "invalid_inFeatures",
			inFeatures:  0,
			outFeatures: 2,
			expectError: true,
		},
		{
			name:        "invalid_outFeatures",
			inFeatures:  4,
			outFeatures: 0,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dense, err := NewDense(tt.inFeatures, tt.outFeatures, tt.opts...)
			if tt.expectError {
				assert.Error(t, err, "Should return error")
				assert.Nil(t, dense, "Should return nil on error")
			} else {
				require.NoError(t, err, "Should create Dense layer")
				require.NotNil(t, dense, "Dense should not be nil")

				weight := dense.Weight()
				assert.Equal(t, []int{tt.inFeatures, tt.outFeatures}, weight.Dim, "Weight shape should match")

				if tt.expectBias {
					bias := dense.Bias()
					assert.Equal(t, []int{tt.outFeatures}, bias.Dim, "Bias shape should match")
				} else {
					bias := dense.Bias()
					assert.Len(t, bias.Dim, 0, "Bias should be empty when disabled")
				}

				if tt.name == "with_name" {
					assert.Equal(t, "my_dense", dense.Name(), "Name should be set")
				}

				if tt.expectCanLearn {
					assert.True(t, dense.CanLearn(), "CanLearn should be true")
				}
			}
		})
	}
}

func TestDense_Init(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		expectError bool
	}{
		{
			name:        "valid_1d",
			inputShape:  []int{4},
			expectError: false,
		},
		{
			name:        "valid_2d",
			inputShape:  []int{2, 4},
			expectError: false,
		},
		{
			name:        "invalid_1d_shape",
			inputShape:  []int{3},
			expectError: true,
		},
		{
			name:        "invalid_2d_shape",
			inputShape:  []int{2, 3},
			expectError: true,
		},
		{
			name:        "invalid_dimension",
			inputShape:  []int{2, 3, 4},
			expectError: true,
		},
	}

	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := dense.Init(tt.inputShape)
			if tt.expectError {
				assert.Error(t, err, "Should return error")
			} else {
				require.NoError(t, err, "Init should succeed")
				output := dense.Output()
				assert.NotEmpty(t, output.Dim, "Output should be allocated")
			}
		})
	}

	// Test nil receiver
	var nilDense *Dense
	err = nilDense.Init([]int{4})
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestDense_Forward(t *testing.T) {
	tests := []struct {
		name       string
		inputShape []int
		inputData  []float32
		weightData []float32
		biasData   []float32
	}{
		{
			name:       "1d_input",
			inputShape: []int{4},
			inputData:  []float32{1.0, 2.0, 3.0, 4.0},
			weightData: []float32{1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0},
			biasData:   []float32{0.0, 0.0},
		},
		{
			name:       "2d_input",
			inputShape: []int{2, 4},
			inputData:  []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			weightData: []float32{1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0},
			biasData:   []float32{0.0, 0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dense, err := NewDense(4, 2)
			require.NoError(t, err, "Should create Dense layer")

			// Set weights and bias
			weight := tensor.Tensor{
				Dim:  []int{4, 2},
				Data: tt.weightData,
			}
			err = dense.SetWeight(weight)
			require.NoError(t, err, "SetWeight should succeed")

			bias := tensor.Tensor{
				Dim:  []int{2},
				Data: tt.biasData,
			}
			err = dense.SetBias(bias)
			require.NoError(t, err, "SetBias should succeed")

			// Initialize layer
			err = dense.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			// Forward pass
			input := tensor.Tensor{
				Dim:  tt.inputShape,
				Data: tt.inputData,
			}

			output, err := dense.Forward(input)
			require.NoError(t, err, "Forward should succeed")
			assert.NotEmpty(t, output.Dim, "Output should have dimensions")
			assert.NotEmpty(t, output.Data, "Output should have data")

			// Verify output shape
			expectedShape, err := dense.OutputShape(tt.inputShape)
			require.NoError(t, err, "OutputShape should succeed")
			assert.Equal(t, expectedShape, output.Dim, "Output shape should match")
		})
	}

	// Test error cases
	var nilDense *Dense
	input := tensor.Tensor{
		Dim:  []int{4},
		Data: []float32{1.0, 2.0, 3.0, 4.0},
	}
	_, err := nilDense.Forward(input)
	assert.Error(t, err, "Should return error for nil receiver")

	dense, _ := NewDense(4, 2)
	emptyInput := tensor.Tensor{}
	_, err = dense.Forward(emptyInput)
	assert.Error(t, err, "Should return error for empty input")

	// Test without Init
	dense2, _ := NewDense(4, 2)
	_, err = dense2.Forward(input)
	assert.Error(t, err, "Should return error if Init not called")
}

func TestDense_Backward(t *testing.T) {
	tests := []struct {
		name       string
		inputShape []int
	}{
		{
			name:       "1d_input",
			inputShape: []int{4},
		},
		{
			name:       "2d_input",
			inputShape: []int{2, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test with CanLearn = false (inference-only)
			dense, err := NewDense(4, 2)
			require.NoError(t, err, "Should create Dense layer")

			err = dense.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			inputSize := 1
			for _, dim := range tt.inputShape {
				inputSize *= dim
			}
			input := tensor.Tensor{
				Dim:  tt.inputShape,
				Data: make([]float32, inputSize),
			}
			for i := range input.Data {
				input.Data[i] = float32(i) * 0.1
			}

			_, err = dense.Forward(input)
			require.NoError(t, err, "Forward should succeed")

			outputShape, _ := dense.OutputShape(tt.inputShape)
			outputSize := 1
			for _, dim := range outputShape {
				outputSize *= dim
			}
			gradOutput := tensor.Tensor{
				Dim:  outputShape,
				Data: make([]float32, outputSize),
			}
			for i := range gradOutput.Data {
				gradOutput.Data[i] = float32(i) * 0.01
			}

			gradInput, err := dense.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")
			assert.NotEmpty(t, gradInput.Dim, "GradInput should have dimensions")

			// Test with CanLearn = true (training mode)
			dense2, err := NewDense(4, 2, WithCanLearn(true))
			require.NoError(t, err, "Should create Dense layer")

			// Set weights and bias
			weight := tensor.Tensor{
				Dim:  []int{4, 2},
				Data: make([]float32, 8),
			}
			dense2.SetWeight(weight)

			bias := tensor.Tensor{
				Dim:  []int{2},
				Data: make([]float32, 2),
			}
			dense2.SetBias(bias)

			err = dense2.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			_, err = dense2.Forward(input)
			require.NoError(t, err, "Forward should succeed")

			gradInput2, err := dense2.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")
			assert.NotEmpty(t, gradInput2.Dim, "GradInput should have dimensions")

			// Verify gradients were computed
			weightParam := dense2.Parameter(ParamWeight)
			assert.NotEmpty(t, weightParam.Grad.Dim, "Weight grad should be allocated")

			if tt.name == "1d_input" || tt.name == "2d_input" {
				biasParam := dense2.Parameter(ParamBias)
				assert.NotEmpty(t, biasParam.Grad.Dim, "Bias grad should be allocated")
			}
		})
	}

	// Test error cases
	var nilDense *Dense
	gradOutput := tensor.Tensor{
		Dim:  []int{2},
		Data: []float32{1.0, 1.0},
	}
	_, err := nilDense.Backward(gradOutput)
	assert.Error(t, err, "Should return error for nil receiver")

	dense, _ := NewDense(4, 2)
	dense.Init([]int{4})
	dense.Forward(tensor.Tensor{Dim: []int{4}, Data: make([]float32, 4)})

	emptyGrad := tensor.Tensor{}
	_, err = dense.Backward(emptyGrad)
	assert.Error(t, err, "Should return error for empty gradOutput")

	// Test without Forward
	dense3, _ := NewDense(4, 2)
	dense3.Init([]int{4})
	_, err = dense3.Backward(gradOutput)
	assert.Error(t, err, "Should return error if Forward not called")
}

func TestDense_OutputShape(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		expectError bool
		expected    []int
	}{
		{
			name:        "1d_input",
			inputShape:  []int{4},
			expectError: false,
			expected:    []int{2},
		},
		{
			name:        "2d_input",
			inputShape:  []int{2, 4},
			expectError: false,
			expected:    []int{2, 2},
		},
		{
			name:        "invalid_1d_shape",
			inputShape:  []int{3},
			expectError: true,
		},
		{
			name:        "invalid_2d_shape",
			inputShape:  []int{2, 3},
			expectError: true,
		},
		{
			name:        "invalid_dimension",
			inputShape:  []int{2, 3, 4},
			expectError: true,
		},
	}

	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputShape, err := dense.OutputShape(tt.inputShape)
			if tt.expectError {
				assert.Error(t, err, "Should return error")
			} else {
				require.NoError(t, err, "OutputShape should succeed")
				assert.Equal(t, tt.expected, outputShape, "Output shape should match")
			}
		})
	}

	// Test nil receiver
	var nilDense *Dense
	_, err = nilDense.OutputShape([]int{4})
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestDense_Weight(t *testing.T) {
	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	weight := dense.Weight()
	assert.Equal(t, []int{4, 2}, weight.Dim, "Weight shape should match")
	assert.Len(t, weight.Data, 8, "Weight data size should match")

	// Test nil receiver
	var nilDense *Dense
	weight = nilDense.Weight()
	assert.Len(t, weight.Dim, 0, "Should return empty tensor for nil receiver")
}

func TestDense_Bias(t *testing.T) {
	// Test with bias
	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	bias := dense.Bias()
	assert.Equal(t, []int{2}, bias.Dim, "Bias shape should match")
	assert.Len(t, bias.Data, 2, "Bias data size should match")

	// Test without bias
	dense2, err := NewDense(4, 2, WithDenseBias(false))
	require.NoError(t, err, "Should create Dense layer")

	bias = dense2.Bias()
	assert.Len(t, bias.Dim, 0, "Bias should be empty when disabled")

	// Test nil receiver
	var nilDense *Dense
	bias = nilDense.Bias()
	assert.Len(t, bias.Dim, 0, "Should return empty tensor for nil receiver")
}

func TestDense_SetWeight(t *testing.T) {
	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	newWeight := tensor.Tensor{
		Dim:  []int{4, 2},
		Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
	}

	err = dense.SetWeight(newWeight)
	require.NoError(t, err, "SetWeight should succeed")

	weight := dense.Weight()
	assert.Equal(t, newWeight.Data, weight.Data, "Weight data should match")

	// Test error cases
	err = dense.SetWeight(tensor.Tensor{Dim: []int{4, 3}, Data: make([]float32, 12)})
	assert.Error(t, err, "Should return error for wrong shape")

	err = dense.SetWeight(tensor.Tensor{})
	assert.Error(t, err, "Should return error for empty tensor")

	var nilDense *Dense
	err = nilDense.SetWeight(newWeight)
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestDense_SetBias(t *testing.T) {
	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	newBias := tensor.Tensor{
		Dim:  []int{2},
		Data: []float32{0.5, -0.5},
	}

	err = dense.SetBias(newBias)
	require.NoError(t, err, "SetBias should succeed")

	bias := dense.Bias()
	assert.Equal(t, newBias.Data, bias.Data, "Bias data should match")

	// Test error cases
	err = dense.SetBias(tensor.Tensor{Dim: []int{3}, Data: []float32{0, 0, 0}})
	assert.Error(t, err, "Should return error for wrong shape")

	err = dense.SetBias(tensor.Tensor{})
	assert.Error(t, err, "Should return error for empty tensor")

	// Test without bias
	dense2, err := NewDense(4, 2, WithDenseBias(false))
	require.NoError(t, err, "Should create Dense layer")

	err = dense2.SetBias(newBias)
	assert.Error(t, err, "Should return error for layer without bias")

	var nilDense *Dense
	err = nilDense.SetBias(newBias)
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestDense_BackwardGradientComputation(t *testing.T) {
	// Test gradient computation with a simple case
	dense, err := NewDense(2, 1, WithCanLearn(true))
	require.NoError(t, err, "Should create Dense layer")

	// Set simple weights: [[1.0], [1.0]]
	weight := tensor.Tensor{
		Dim:  []int{2, 1},
		Data: []float32{1.0, 1.0},
	}
	dense.SetWeight(weight)

	// Set bias: [0.0]
	bias := tensor.Tensor{
		Dim:  []int{1},
		Data: []float32{0.0},
	}
	dense.SetBias(bias)

	// Initialize and forward
	err = dense.Init([]int{2})
	require.NoError(t, err, "Init should succeed")

	input := tensor.Tensor{
		Dim:  []int{2},
		Data: []float32{1.0, 2.0},
	}
	output, err := dense.Forward(input)
	require.NoError(t, err, "Forward should succeed")

	// Expected output: 1.0*1.0 + 2.0*1.0 + 0.0 = 3.0
	assert.InDelta(t, 3.0, output.Data[0], 1e-6, "Output should be 3.0")

	// Backward pass
	gradOutput := tensor.Tensor{
		Dim:  []int{1},
		Data: []float32{1.0},
	}
	gradInput, err := dense.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed")

	// Expected gradInput: [1.0, 1.0] (weight transpose @ gradOutput)
	assert.InDelta(t, 1.0, gradInput.Data[0], 1e-6, "GradInput[0] should be 1.0")
	assert.InDelta(t, 1.0, gradInput.Data[1], 1e-6, "GradInput[1] should be 1.0")

	// Check weight gradient: should be [1.0, 2.0] (input @ gradOutput)
	weightParam := dense.Parameter(ParamWeight)
	assert.NotEmpty(t, weightParam.Grad.Dim, "Weight grad should be allocated")
	assert.InDelta(t, 1.0, weightParam.Grad.Data[0], 1e-6, "Weight grad[0] should be 1.0")
	assert.InDelta(t, 2.0, weightParam.Grad.Data[1], 1e-6, "Weight grad[1] should be 2.0")

	// Check bias gradient: should be [1.0]
	biasParam := dense.Parameter(ParamBias)
	assert.NotEmpty(t, biasParam.Grad.Dim, "Bias grad should be allocated")
	assert.InDelta(t, 1.0, biasParam.Grad.Data[0], 1e-6, "Bias grad[0] should be 1.0")
}

// TestDense_ComputeOutput tests that dense layers compute outputs correctly
func TestDense_ComputeOutput(t *testing.T) {
	tests := []struct {
		name           string
		inFeatures     int
		outFeatures    int
		inputShape     []int
		input          tensor.Tensor
		weight         tensor.Tensor
		bias           tensor.Tensor
		expectedShape  []int
		expectedOutput []float32
	}{
		{
			name:        "1d_3x3_identity",
			inFeatures:  3,
			outFeatures: 3,
			inputShape:  []int{3},
			input: tensor.Tensor{
				Dim:  []int{3},
				Data: []float32{1.0, 2.0, 3.0},
			},
			weight: tensor.Tensor{
				Dim:  []int{3, 3},
				Data: []float32{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},
			},
			bias: tensor.Tensor{
				Dim:  []int{3},
				Data: []float32{0.0, 0.0, 0.0},
			},
			expectedShape:  []int{3},
			expectedOutput: []float32{1.0, 2.0, 3.0},
		},
		{
			name:        "1d_3x3_custom",
			inFeatures:  3,
			outFeatures: 3,
			inputShape:  []int{3},
			input: tensor.Tensor{
				Dim:  []int{3},
				Data: []float32{1.0, 2.0, 3.0},
			},
			weight: tensor.Tensor{
				Dim:  []int{3, 3},
				Data: []float32{1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0},
			},
			bias: tensor.Tensor{
				Dim:  []int{3},
				Data: []float32{0.0, 0.0, 0.0},
			},
			expectedShape:  []int{3},
			expectedOutput: []float32{4.0, 3.0, 5.0},
		},
		{
			name:        "2d_2x2_4_outputs",
			inFeatures:  2,
			outFeatures: 4,
			inputShape:  []int{2, 2},
			input: tensor.Tensor{
				Dim:  []int{2, 2},
				Data: []float32{1.0, 2.0, 3.0, 4.0},
			},
			weight: tensor.Tensor{
				Dim:  []int{2, 4},
				Data: []float32{1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0},
			},
			bias: tensor.Tensor{
				Dim:  []int{4},
				Data: []float32{0.0, 0.0, 0.0, 0.0},
			},
			expectedShape:  []int{2, 4},
			expectedOutput: []float32{3.0, -1.0, 3.0, -1.0, 7.0, -1.0, 7.0, -1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dense, err := NewDense(tt.inFeatures, tt.outFeatures)
			require.NoError(t, err, "Should create Dense layer")

			err = dense.SetWeight(tt.weight)
			require.NoError(t, err, "SetWeight should succeed")

			err = dense.SetBias(tt.bias)
			require.NoError(t, err, "SetBias should succeed")

			err = dense.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			output, err := dense.Forward(tt.input)
			require.NoError(t, err, "Forward should succeed")

			assert.Equal(t, tt.expectedShape, output.Dim, "Output shape should match")
			require.Len(t, output.Data, len(tt.expectedOutput), "Output data length should match")
			for i := range tt.expectedOutput {
				assert.InDelta(t, tt.expectedOutput[i], output.Data[i], 1e-6, "Output[%d] should match expected", i)
			}
		})
	}
}

// TestDense_BackwardAccuracy tests that dense layer backward pass computes parameter gradients correctly
func TestDense_BackwardAccuracy(t *testing.T) {
	tests := []struct {
		name               string
		inFeatures         int
		outFeatures        int
		hasBias            bool
		inputShape         []int
		input              tensor.Tensor
		weight             tensor.Tensor
		bias               tensor.Tensor
		gradOutput         tensor.Tensor
		expectedWeightGrad []float32
		expectedBiasGrad   []float32
		expectedInputGrad  []float32
	}{
		{
			name:        "1d_single_sample",
			inFeatures:  2,
			outFeatures: 1,
			hasBias:     true,
			inputShape:  []int{2},
			input: tensor.Tensor{
				Dim:  []int{2},
				Data: []float32{1.0, 2.0},
			},
			weight: tensor.Tensor{
				Dim:  []int{2, 1},
				Data: []float32{1.0, 1.0},
			},
			bias: tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{0.0},
			},
			gradOutput: tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{1.0},
			},
			expectedWeightGrad: []float32{1.0, 2.0}, // input^T @ gradOutput = [1, 2]^T @ [1] = [1, 2]
			expectedBiasGrad:   []float32{1.0},      // gradOutput = [1]
			expectedInputGrad:  []float32{1.0, 1.0}, // gradOutput @ weight^T = [1] @ [[1],[1]]^T = [1, 1]
		},
		{
			name:        "1d_no_bias",
			inFeatures:  2,
			outFeatures: 1,
			hasBias:     false,
			inputShape:  []int{2},
			input: tensor.Tensor{
				Dim:  []int{2},
				Data: []float32{1.0, 2.0},
			},
			weight: tensor.Tensor{
				Dim:  []int{2, 1},
				Data: []float32{1.0, 1.0},
			},
			bias: tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{0.0},
			},
			gradOutput: tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{1.0},
			},
			expectedWeightGrad: []float32{1.0, 2.0},
			expectedBiasGrad:   nil, // No bias
			expectedInputGrad:  []float32{1.0, 1.0},
		},
		{
			name:        "1d_multi_output",
			inFeatures:  3,
			outFeatures: 2,
			hasBias:     true,
			inputShape:  []int{3},
			input: tensor.Tensor{
				Dim:  []int{3},
				Data: []float32{1.0, 2.0, 3.0},
			},
			weight: tensor.Tensor{
				Dim:  []int{3, 2},
				Data: []float32{1.0, 0.0, 0.0, 1.0, 1.0, 1.0},
			},
			bias: tensor.Tensor{
				Dim:  []int{2},
				Data: []float32{0.0, 0.0},
			},
			gradOutput: tensor.Tensor{
				Dim:  []int{2},
				Data: []float32{1.0, 2.0},
			},
			expectedWeightGrad: []float32{1.0, 2.0, 2.0, 4.0, 3.0, 6.0}, // [1,2,3]^T @ [1,2] = [[1,2],[2,4],[3,6]]
			expectedBiasGrad:   []float32{1.0, 2.0},
			expectedInputGrad:  []float32{1.0, 2.0, 3.0}, // gradInput[i] = sum_j(gradOutput[j] * weight[i,j])
		},
		{
			name:        "2d_batch",
			inFeatures:  2,
			outFeatures: 1,
			hasBias:     true,
			inputShape:  []int{2, 2},
			input: tensor.Tensor{
				Dim:  []int{2, 2},
				Data: []float32{1.0, 2.0, 3.0, 4.0},
			},
			weight: tensor.Tensor{
				Dim:  []int{2, 1},
				Data: []float32{1.0, 1.0},
			},
			bias: tensor.Tensor{
				Dim:  []int{1},
				Data: []float32{0.0},
			},
			gradOutput: tensor.Tensor{
				Dim:  []int{2, 1},
				Data: []float32{1.0, 2.0},
			},
			expectedWeightGrad: []float32{7.0, 10.0},          // sum over batch: [1,2]^T @ [1] + [3,4]^T @ [2] = [1,2] + [6,8] = [7,10]
			expectedBiasGrad:   []float32{3.0},                // sum([1, 2]) = 3
			expectedInputGrad:  []float32{1.0, 1.0, 2.0, 2.0}, // [1] @ [1,1]^T = [1,1], [2] @ [1,1]^T = [2,2]
		},
		{
			name:        "2d_batch_multi_output",
			inFeatures:  2,
			outFeatures: 2,
			hasBias:     true,
			inputShape:  []int{2, 2},
			input: tensor.Tensor{
				Dim:  []int{2, 2},
				Data: []float32{1.0, 2.0, 3.0, 4.0},
			},
			weight: tensor.Tensor{
				Dim:  []int{2, 2},
				Data: []float32{1.0, 0.0, 0.0, 1.0},
			},
			bias: tensor.Tensor{
				Dim:  []int{2},
				Data: []float32{0.0, 0.0},
			},
			gradOutput: tensor.Tensor{
				Dim:  []int{2, 2},
				Data: []float32{1.0, 2.0, 3.0, 4.0},
			},
			expectedWeightGrad: []float32{
				10.0, 14.0, // weight grad[i,j] = sum_batch(input[b,i] * gradOutput[b,j])
				14.0, 20.0, // i=0,j=0: 1*1 + 3*3 = 10; i=0,j=1: 1*2 + 3*4 = 14; etc.
			},
			expectedBiasGrad:  []float32{4.0, 6.0},           // sum([1,2]) = [4], sum([2,4]) = [6]
			expectedInputGrad: []float32{1.0, 2.0, 3.0, 4.0}, // batch 0: [1,2] @ [[1,0],[0,1]]^T = [1,2]; batch 1: [3,4] @ [[1,0],[0,1]]^T = [3,4]
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := []interface{}{WithCanLearn(true)}
			if !tt.hasBias {
				opts = append(opts, WithDenseBias(false))
			}
			dense, err := NewDense(tt.inFeatures, tt.outFeatures, opts...)
			require.NoError(t, err, "Should create Dense layer")

			err = dense.SetWeight(tt.weight)
			require.NoError(t, err, "SetWeight should succeed")

			if tt.hasBias {
				err = dense.SetBias(tt.bias)
				require.NoError(t, err, "SetBias should succeed")
			}

			err = dense.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			// Forward pass to store input and output
			_, err = dense.Forward(tt.input)
			require.NoError(t, err, "Forward should succeed")

			// Zero gradients first
			dense.ZeroGrad()

			// Backward pass
			gradInput, err := dense.Backward(tt.gradOutput)
			require.NoError(t, err, "Backward should succeed")

			// Verify weight gradient
			weightParam := dense.Parameter(ParamWeight)
			require.NotNil(t, weightParam, "Weight parameter should exist")
			require.NotEmpty(t, weightParam.Grad.Dim, "Weight grad should be allocated")
			require.Len(t, weightParam.Grad.Data, len(tt.expectedWeightGrad), "Weight grad length should match")
			for i := range tt.expectedWeightGrad {
				assert.InDelta(t, tt.expectedWeightGrad[i], weightParam.Grad.Data[i], 1e-6,
					"Weight grad[%d] should match expected", i)
			}

			// Verify bias gradient
			if tt.hasBias {
				biasParam := dense.Parameter(ParamBias)
				require.NotNil(t, biasParam, "Bias parameter should exist")
				require.NotEmpty(t, biasParam.Grad.Dim, "Bias grad should be allocated")
				require.Len(t, biasParam.Grad.Data, len(tt.expectedBiasGrad), "Bias grad length should match")
				for i := range tt.expectedBiasGrad {
					assert.InDelta(t, tt.expectedBiasGrad[i], biasParam.Grad.Data[i], 1e-6,
						"Bias grad[%d] should match expected", i)
				}
			}

			// Verify input gradient
			require.Len(t, gradInput.Data, len(tt.expectedInputGrad), "GradInput length should match")
			for i := range tt.expectedInputGrad {
				assert.InDelta(t, tt.expectedInputGrad[i], gradInput.Data[i], 1e-6,
					"GradInput[%d] should match expected", i)
			}
		})
	}
}
