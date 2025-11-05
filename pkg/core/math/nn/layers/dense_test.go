package layers

import (
	"testing"

	nntypes "github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewDense(t *testing.T) {
	tests := []struct {
		name           string
		inFeatures     int
		outFeatures    int
		opts           []Option
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
			name:           "with_can_learn",
			inFeatures:     4,
			outFeatures:    2,
			opts:           []Option{WithCanLearn(true)},
			expectError:    false,
			expectBias:     true,
			expectCanLearn: true,
		},
		{
			name:        "with_name",
			inFeatures:  4,
			outFeatures: 2,
			opts:        []Option{WithName("my_dense")},
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
				assert.Equal(t, []int{tt.inFeatures, tt.outFeatures}, weight.Shape().ToSlice(), "Weight shape should match")

				if tt.expectBias {
					bias := dense.Bias()
					assert.Equal(t, []int{tt.outFeatures}, bias.Shape().ToSlice(), "Bias shape should match")
				} else {
					bias := dense.Bias()
					assert.Len(t, bias.Shape().ToSlice(), 0, "Bias should be empty when disabled")
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
				assert.NotEmpty(t, output.Shape().ToSlice(), "Output should be allocated")
			}
		})
	}

	// Test nil receiver
	var nilDense *Dense
	err = nilDense.Init(tensor.NewShape(4))
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
			weight := tensor.FromFloat32(tensor.NewShape(4, 2), tt.weightData)
			err = dense.SetWeight(weight)
			require.NoError(t, err, "SetWeight should succeed")

			bias := tensor.FromFloat32(tensor.NewShape(2), tt.biasData)
			err = dense.SetBias(bias)
			require.NoError(t, err, "SetBias should succeed")

			// Initialize layer
			err = dense.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			// Forward pass
			input := tensor.FromFloat32(tensor.NewShape(tt.inputShape...), tt.inputData)

			output, err := dense.Forward(input)
			require.NoError(t, err, "Forward should succeed")
			assert.NotEmpty(t, output.Shape().ToSlice(), "Output should have dimensions")
			assert.NotEmpty(t, output.Data(), "Output should have data")

			// Verify output shape
			expectedShape, err := dense.OutputShape(tt.inputShape)
			require.NoError(t, err, "OutputShape should succeed")
			assert.Equal(t, expectedShape, output.Shape().ToSlice(), "Output shape should match")
		})
	}

	// Test error cases
	var nilDense *Dense
	input := tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 2.0, 3.0, 4.0})
	_, err := nilDense.Forward(input)
	assert.Error(t, err, "Should return error for nil receiver")

	dense, _ := NewDense(4, 2)
	var emptyInput types.Tensor
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
			input := tensor.FromFloat32(tensor.NewShape(tt.inputShape...), make([]float32, inputSize))
			inputData := input.Data().([]float32)
			for i := range inputData {
				inputData[i] = float32(i) * 0.1
			}

			_, err = dense.Forward(input)
			require.NoError(t, err, "Forward should succeed")

			outputShape, _ := dense.OutputShape(tt.inputShape)
			outputSize := 1
			for _, dim := range outputShape {
				outputSize *= dim
			}
			gradOutput := tensor.FromFloat32(tensor.NewShape(outputShape...), make([]float32, outputSize))
			gradOutputData := gradOutput.Data().([]float32)
			for i := range gradOutputData {
				gradOutputData[i] = float32(i) * 0.01
			}

			gradInput, err := dense.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")
			assert.NotEmpty(t, gradInput.Shape().ToSlice(), "GradInput should have dimensions")

			// Test with CanLearn = true (training mode)
			dense2, err := NewDense(4, 2, WithCanLearn(true))
			require.NoError(t, err, "Should create Dense layer")

			// Set weights and bias
			weight := tensor.FromFloat32(tensor.NewShape(4, 2), make([]float32, 8))
			dense2.SetWeight(weight)

			bias := tensor.FromFloat32(tensor.NewShape(2), make([]float32, 2))
			dense2.SetBias(bias)

			err = dense2.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			_, err = dense2.Forward(input)
			require.NoError(t, err, "Forward should succeed")

			gradInput2, err := dense2.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")
			assert.NotEmpty(t, gradInput2.Shape().ToSlice(), "GradInput should have dimensions")

			// Verify gradients were computed
			weightParam, ok := dense2.Base.Parameter(nntypes.ParamWeights)
			require.True(t, ok, "Weight parameter should exist")
			require.NotNil(t, weightParam.Grad, "Weight grad should be allocated")
			assert.NotEmpty(t, weightParam.Grad.Shape().ToSlice(), "Weight grad should have dimensions")

			if tt.name == "1d_input" || tt.name == "2d_input" {
				biasParam, ok := dense2.Base.Parameter(nntypes.ParamBiases)
				require.True(t, ok, "Bias parameter should exist")
				require.NotNil(t, biasParam.Grad, "Bias grad should be allocated")
				assert.NotEmpty(t, biasParam.Grad.Shape().ToSlice(), "Bias grad should have dimensions")
			}
		})
	}

	// Test error cases
	var nilDense *Dense
	gradOutput := tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 1.0})
	_, err := nilDense.Backward(gradOutput)
	assert.Error(t, err, "Should return error for nil receiver")

	dense, _ := NewDense(4, 2)
	dense.Init(tensor.NewShape(4))
	dense.Forward(tensor.FromFloat32(tensor.NewShape(4), make([]float32, 4)))

	var emptyGrad types.Tensor
	_, err = dense.Backward(emptyGrad)
	assert.Error(t, err, "Should return error for empty gradOutput")

	// Test without Forward
	dense3, _ := NewDense(4, 2)
	dense3.Init(tensor.NewShape(4))
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
	assert.Equal(t, []int{4, 2}, weight.Shape().ToSlice(), "Weight shape should match")
	assert.Len(t, weight.Data(), 8, "Weight data size should match")

	// Test nil receiver
	var nilDense *Dense
	weight = nilDense.Weight()
	assert.Len(t, weight.Shape().ToSlice(), 0, "Should return empty tensor for nil receiver")
}

func TestDense_Bias(t *testing.T) {
	// Test with bias
	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	bias := dense.Bias()
	assert.Equal(t, []int{2}, bias.Shape().ToSlice(), "Bias shape should match")
	assert.Len(t, bias.Data(), 2, "Bias data size should match")

	// Test nil receiver
	var nilDense *Dense
	bias = nilDense.Bias()
	assert.Len(t, bias.Shape().ToSlice(), 0, "Should return empty tensor for nil receiver")
}

func TestDense_SetWeight(t *testing.T) {
	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	newWeight := tensor.FromFloat32(tensor.NewShape(4, 2), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})

	err = dense.SetWeight(newWeight)
	require.NoError(t, err, "SetWeight should succeed")

	weight := dense.Weight()
	newWeightData := newWeight.Data().([]float32)
	weightData := weight.Data().([]float32)
	assert.Equal(t, newWeightData, weightData, "Weight data should match")

	// Test error cases
	err = dense.SetWeight(tensor.FromFloat32(tensor.NewShape(4, 3), make([]float32, 12)))
	assert.Error(t, err, "Should return error for wrong shape")

	var emptyTensor types.Tensor
	err = dense.SetWeight(emptyTensor)
	assert.Error(t, err, "Should return error for empty tensor")

	var nilDense *Dense
	err = nilDense.SetWeight(newWeight)
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestDense_SetBias(t *testing.T) {
	dense, err := NewDense(4, 2)
	require.NoError(t, err, "Should create Dense layer")

	newBias := tensor.FromFloat32(tensor.NewShape(2), []float32{0.5, -0.5})

	err = dense.SetBias(newBias)
	require.NoError(t, err, "SetBias should succeed")

	bias := dense.Bias()
	assert.Equal(t, newBias.Data(), bias.Data(), "Bias data should match")

	// Test error cases
	err = dense.SetBias(tensor.FromFloat32(tensor.NewShape(3), []float32{0, 0, 0}))
	assert.Error(t, err, "Should return error for wrong shape")

	var emptyTensor types.Tensor
	err = dense.SetBias(emptyTensor)
	assert.Error(t, err, "Should return error for empty tensor")

	var nilDense *Dense
	err = nilDense.SetBias(newBias)
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestDense_BackwardGradientComputation(t *testing.T) {
	// Test gradient computation with a simple case
	dense, err := NewDense(2, 1, WithCanLearn(true))
	require.NoError(t, err, "Should create Dense layer")

	// Set simple weights: [[1.0], [1.0]]
	weight := tensor.FromFloat32(tensor.NewShape(2, 1), []float32{1.0, 1.0})
	dense.SetWeight(weight)

	// Set bias: [0.0]
	bias := tensor.FromFloat32(tensor.NewShape(1), []float32{0.0})
	dense.SetBias(bias)

	// Initialize and forward
	err = dense.Init(tensor.NewShape(2))
	require.NoError(t, err, "Init should succeed")

	input := tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0})
	output, err := dense.Forward(input)
	require.NoError(t, err, "Forward should succeed")

	// Expected output: 1.0*1.0 + 2.0*1.0 + 0.0 = 3.0
	outputData := output.Data().([]float32)
	assert.InDelta(t, 3.0, outputData[0], 1e-6, "Output should be 3.0")

	// Backward pass
	gradOutput := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
	gradInput, err := dense.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed")

	// Expected gradInput: [1.0, 1.0] (weight transpose @ gradOutput)
	gradInputData := gradInput.Data().([]float32)
	assert.InDelta(t, 1.0, gradInputData[0], 1e-6, "GradInput[0] should be 1.0")
	assert.InDelta(t, 1.0, gradInputData[1], 1e-6, "GradInput[1] should be 1.0")

	// Check weight gradient: should be [1.0, 2.0] (input @ gradOutput)
	weightParam, ok := dense.Base.Parameter(nntypes.ParamWeights)
	require.True(t, ok, "Weight parameter should exist")
	require.NotNil(t, weightParam.Grad, "Weight grad should be allocated")
	assert.NotEmpty(t, weightParam.Grad.Shape().ToSlice(), "Weight grad should have dimensions")
	weightGradData := weightParam.Grad.Data().([]float32)
	assert.InDelta(t, 1.0, weightGradData[0], 1e-6, "Weight grad[0] should be 1.0")
	assert.InDelta(t, 2.0, weightGradData[1], 1e-6, "Weight grad[1] should be 2.0")

	// Check bias gradient: should be [1.0]
	biasParam, ok := dense.Base.Parameter(nntypes.ParamBiases)
	require.True(t, ok, "Bias parameter should exist")
	require.NotNil(t, biasParam.Grad, "Bias grad should be allocated")
	assert.NotEmpty(t, biasParam.Grad.Shape().ToSlice(), "Bias grad should have dimensions")
	biasGradData := biasParam.Grad.Data().([]float32)
	assert.InDelta(t, 1.0, biasGradData[0], 1e-6, "Bias grad[0] should be 1.0")
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
			name:           "1d_3x3_identity",
			inFeatures:     3,
			outFeatures:    3,
			inputShape:     []int{3},
			input:          tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0}),
			weight:         tensor.FromFloat32(tensor.NewShape(3, 3), []float32{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
			bias:           tensor.FromFloat32(tensor.NewShape(3), []float32{0.0, 0.0, 0.0}),
			expectedShape:  []int{3},
			expectedOutput: []float32{1.0, 2.0, 3.0},
		},
		{
			name:           "1d_3x3_custom",
			inFeatures:     3,
			outFeatures:    3,
			inputShape:     []int{3},
			input:          tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0}),
			weight:         tensor.FromFloat32(tensor.NewShape(3, 3), []float32{1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0}),
			bias:           tensor.FromFloat32(tensor.NewShape(3), []float32{0.0, 0.0, 0.0}),
			expectedShape:  []int{3},
			expectedOutput: []float32{4.0, 3.0, 5.0},
		},
		{
			name:           "2d_2x2_4_outputs",
			inFeatures:     2,
			outFeatures:    4,
			inputShape:     []int{2, 2},
			input:          tensor.FromFloat32(tensor.NewShape(2, 2), []float32{1.0, 2.0, 3.0, 4.0}),
			weight:         tensor.FromFloat32(tensor.NewShape(2, 4), []float32{1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0}),
			bias:           tensor.FromFloat32(tensor.NewShape(4), []float32{0.0, 0.0, 0.0, 0.0}),
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

			assert.Equal(t, tt.expectedShape, output.Shape().ToSlice(), "Output shape should match")
			outputData := output.Data().([]float32)
			require.Len(t, outputData, len(tt.expectedOutput), "Output data length should match")
			for i := range tt.expectedOutput {
				assert.InDelta(t, tt.expectedOutput[i], outputData[i], 1e-6, "Output[%d] should match expected", i)
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
			name:               "1d_single_sample",
			inFeatures:         2,
			outFeatures:        1,
			hasBias:            true,
			inputShape:         []int{2},
			input:              tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}),
			weight:             tensor.FromFloat32(tensor.NewShape(2, 1), []float32{1.0, 1.0}),
			bias:               tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
			gradOutput:         tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}),
			expectedWeightGrad: []float32{1.0, 2.0}, // input^T @ gradOutput = [1, 2]^T @ [1] = [1, 2]
			expectedBiasGrad:   []float32{1.0},      // gradOutput = [1]
			expectedInputGrad:  []float32{1.0, 1.0}, // gradOutput @ weight^T = [1] @ [[1],[1]]^T = [1, 1]
		},
		{
			name:               "1d_multi_output",
			inFeatures:         3,
			outFeatures:        2,
			hasBias:            true,
			inputShape:         []int{3},
			input:              tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0}),
			weight:             tensor.FromFloat32(tensor.NewShape(3, 2), []float32{1.0, 0.0, 0.0, 1.0, 1.0, 1.0}),
			bias:               tensor.FromFloat32(tensor.NewShape(2), []float32{0.0, 0.0}),
			gradOutput:         tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}),
			expectedWeightGrad: []float32{1.0, 2.0, 2.0, 4.0, 3.0, 6.0}, // [1,2,3]^T @ [1,2] = [[1,2],[2,4],[3,6]]
			expectedBiasGrad:   []float32{1.0, 2.0},
			expectedInputGrad:  []float32{1.0, 2.0, 3.0}, // gradInput[i] = sum_j(gradOutput[j] * weight[i,j])
		},
		{
			name:               "2d_batch",
			inFeatures:         2,
			outFeatures:        1,
			hasBias:            true,
			inputShape:         []int{2, 2},
			input:              tensor.FromFloat32(tensor.NewShape(2, 2), []float32{1.0, 2.0, 3.0, 4.0}),
			weight:             tensor.FromFloat32(tensor.NewShape(2, 1), []float32{1.0, 1.0}),
			bias:               tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
			gradOutput:         tensor.FromFloat32(tensor.NewShape(2, 1), []float32{1.0, 2.0}),
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
			input:       tensor.FromFloat32(tensor.NewShape(2, 2), []float32{1.0, 2.0, 3.0, 4.0}),
			weight:      tensor.FromFloat32(tensor.NewShape(2, 2), []float32{1.0, 0.0, 0.0, 1.0}),
			bias:        tensor.FromFloat32(tensor.NewShape(2), []float32{0.0, 0.0}),
			gradOutput:  tensor.FromFloat32(tensor.NewShape(2, 2), []float32{1.0, 2.0, 3.0, 4.0}),
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
			opts := []Option{WithCanLearn(true)}
			dense, err := NewDense(tt.inFeatures, tt.outFeatures, opts...)
			require.NoError(t, err, "Should create Dense layer")

			err = dense.SetWeight(tt.weight)
			require.NoError(t, err, "SetWeight should succeed")

			// Bias is always created, so always set it
			err = dense.SetBias(tt.bias)
			require.NoError(t, err, "SetBias should succeed")

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
			weightParam, _ := dense.Base.Parameter(nntypes.ParamWeights)
			require.NotNil(t, weightParam, "Weight parameter should exist")
			require.NotEmpty(t, weightParam.Grad.Shape().ToSlice(), "Weight grad should be allocated")
			weightGradData := weightParam.Grad.Data().([]float32)
			require.Len(t, weightGradData, len(tt.expectedWeightGrad), "Weight grad length should match")
			for i := range tt.expectedWeightGrad {
				assert.InDelta(t, tt.expectedWeightGrad[i], weightGradData[i], 1e-6,
					"Weight grad[%d] should match expected", i)
			}

			// Verify bias gradient (bias is always created)
			biasParam, _ := dense.Base.Parameter(nntypes.ParamBiases)
			require.NotNil(t, biasParam, "Bias parameter should exist")
			if tt.expectedBiasGrad != nil {
				require.NotEmpty(t, biasParam.Grad.Shape().ToSlice(), "Bias grad should be allocated")
				biasGradData := biasParam.Grad.Data().([]float32)
				require.Len(t, biasGradData, len(tt.expectedBiasGrad), "Bias grad length should match")
				for i := range tt.expectedBiasGrad {
					assert.InDelta(t, tt.expectedBiasGrad[i], biasGradData[i], 1e-6,
						"Bias grad[%d] should match expected", i)
				}
			}

			// Verify input gradient
			gradInputData := gradInput.Data().([]float32)
			require.Len(t, gradInputData, len(tt.expectedInputGrad), "GradInput length should match")
			for i := range tt.expectedInputGrad {
				assert.InDelta(t, tt.expectedInputGrad[i], gradInputData[i], 1e-6,
					"GradInput[%d] should match expected", i)
			}
		})
	}
}

// TestDense_EdgeCases tests edge cases for Dense layer
func TestDense_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilDense *Dense
	_, err := nilDense.Forward(tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	dense, err := NewDense(2, 3)
	require.NoError(t, err)
	err = dense.Init(tensor.NewShape(2))
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = dense.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	dense2, err := NewDense(2, 3)
	require.NoError(t, err)
	input := tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0})
	_, err = dense2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward without Forward
	dense3, err := NewDense(2, 3)
	require.NoError(t, err)
	err = dense3.Init(tensor.NewShape(2))
	require.NoError(t, err)
	gradOutput := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 1.0, 1.0})
	_, err = dense3.Backward(gradOutput)
	assert.Error(t, err, "Backward should error if Forward not called")

	// Test Backward with empty gradOutput
	dense4, err := NewDense(2, 3)
	require.NoError(t, err)
	err = dense4.Init(tensor.NewShape(2))
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0})
	_, err = dense4.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = dense4.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
}
