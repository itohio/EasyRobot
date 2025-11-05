package layers

import (
	"testing"

	nntypes "github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewConv1D(t *testing.T) {
	tests := []struct {
		name           string
		inChannels     int
		outChannels    int
		kernelLen      int
		stride         int
		pad            int
		opts           []Option
		expectError    bool
		expectBias     bool
		expectCanLearn bool
	}{
		{
			name:           "basic",
			inChannels:     3,
			outChannels:    16,
			kernelLen:      3,
			stride:         1,
			pad:            1,
			expectError:    false,
			expectBias:     false,
			expectCanLearn: false,
		},
		{
			name:           "no_bias",
			inChannels:     3,
			outChannels:    16,
			kernelLen:      3,
			stride:         1,
			pad:            1,
			opts:           []Option{UseBias(false)},
			expectError:    false,
			expectBias:     false,
			expectCanLearn: false,
		},
		{
			name:           "with_can_learn",
			inChannels:     3,
			outChannels:    16,
			kernelLen:      3,
			stride:         1,
			pad:            1,
			opts:           []Option{WithCanLearn(true), UseBias(true)},
			expectError:    false,
			expectBias:     true,
			expectCanLearn: true,
		},
		{
			name:        "with_name",
			inChannels:  3,
			outChannels: 16,
			kernelLen:   3,
			stride:      1,
			pad:         1,
			opts:        []Option{WithName("my_conv1d"), UseBias(true)},
			expectError: false,
			expectBias:  true,
		},
		{
			name:        "invalid_inChannels",
			inChannels:  0,
			outChannels: 16,
			kernelLen:   3,
			stride:      1,
			pad:         1,
			expectError: true,
		},
		{
			name:        "invalid_outChannels",
			inChannels:  3,
			outChannels: 0,
			kernelLen:   3,
			stride:      1,
			pad:         1,
			expectError: true,
		},
		{
			name:        "invalid_kernelLen",
			inChannels:  3,
			outChannels: 16,
			kernelLen:   0,
			stride:      1,
			pad:         1,
			expectError: true,
		},
		{
			name:        "invalid_stride",
			inChannels:  3,
			outChannels: 16,
			kernelLen:   3,
			stride:      0,
			pad:         1,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv, err := NewConv1D(tt.inChannels, tt.outChannels, tt.kernelLen, tt.stride, tt.pad, tt.opts...)
			if tt.expectError {
				assert.Error(t, err, "Should return error")
				assert.Nil(t, conv, "Should return nil on error")
			} else {
				require.NoError(t, err, "Should create Conv1D layer")
				require.NotNil(t, conv, "Conv1D should not be nil")

				weight := conv.Weight()
				assert.Equal(t, []int{tt.outChannels, tt.inChannels, tt.kernelLen}, weight.Shape().ToSlice(), "Weight shape should match")

				if tt.expectBias {
					bias := conv.Bias()
					assert.Equal(t, []int{tt.outChannels}, bias.Shape().ToSlice(), "Bias shape should match")
				} else {
					bias := conv.Bias()
					if tensor.IsNil(bias) {
						assert.True(t, tensor.IsNil(bias), "Bias should be nil or empty when disabled")
					} else {
						assert.Equal(t, 0, bias.Shape().Rank(), "Bias should be empty when disabled")
					}
				}

				if tt.name == "with_name" {
					assert.Equal(t, "my_conv1d", conv.Name(), "Name should be set")
				}

				if tt.expectCanLearn {
					assert.True(t, conv.CanLearn(), "CanLearn should be true")
				}
			}
		})
	}
}

func TestConv1D_Init(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		expectError bool
	}{
		{
			name:        "valid_3d",
			inputShape:  []int{1, 3, 10},
			expectError: false,
		},
		{
			name:        "valid_batch",
			inputShape:  []int{2, 3, 10},
			expectError: false,
		},
		{
			name:        "invalid_dimension",
			inputShape:  []int{3, 10},
			expectError: true,
		},
		{
			name:        "invalid_channels",
			inputShape:  []int{1, 2, 10},
			expectError: true,
		},
		{
			name:        "invalid_batch",
			inputShape:  []int{0, 3, 10},
			expectError: true,
		},
		{
			name:        "invalid_output_length",
			inputShape:  []int{1, 3, 1},
			expectError: true,
		},
	}

	conv, err := NewConv1D(3, 16, 3, 1, 0)
	require.NoError(t, err, "Should create Conv1D layer")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := conv.Init(tt.inputShape)
			if tt.expectError {
				assert.Error(t, err, "Should return error")
			} else {
				require.NoError(t, err, "Init should succeed")
				output := conv.Output()
				assert.True(t, output.Shape().Rank() > 0, "Output should be allocated")
			}
		})
	}

	// Test nil receiver
	var nilConv *Conv1D
	err = nilConv.Init(tensor.NewShape(1, 3, 10))
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestConv1D_Forward(t *testing.T) {
	conv, err := NewConv1D(3, 16, 3, 1, 1)
	require.NoError(t, err, "Should create Conv1D layer")

	inputShape := []int{1, 3, 10}
	err = conv.Init(inputShape)
	require.NoError(t, err, "Init should succeed")

	input := tensor.New(tensor.DTFP32, tensor.NewShape(inputShape...))
	inputData := input.Data().([]float32)
	for i := range inputData {
		inputData[i] = float32(i) * 0.1
	}

	output, err := conv.Forward(input)
	require.NoError(t, err, "Forward should succeed")
	assert.True(t, output.Shape().Rank() > 0, "Output should have dimensions")
	assert.True(t, output.Size() > 0, "Output should have data")

	// Verify output shape
	expectedShape, err := conv.OutputShape(inputShape)
	require.NoError(t, err, "OutputShape should succeed")
	assert.Equal(t, expectedShape, output.Shape().ToSlice(), "Output shape should match")

	// Test error cases
	var nilConv *Conv1D
	_, err = nilConv.Forward(input)
	assert.Error(t, err, "Should return error for nil receiver")

	var emptyInput types.Tensor
	_, err = conv.Forward(emptyInput)
	assert.Error(t, err, "Should return error for empty input")

	// Test without Init
	conv2, _ := NewConv1D(3, 16, 3, 1, 1)
	_, err = conv2.Forward(input)
	assert.Error(t, err, "Should return error if Init not called")
}

func TestConv1D_Backward(t *testing.T) {
	conv, err := NewConv1D(3, 16, 3, 1, 1)
	require.NoError(t, err, "Should create Conv1D layer")

	inputShape := []int{1, 3, 10}
	err = conv.Init(inputShape)
	require.NoError(t, err, "Init should succeed")

	input := tensor.New(tensor.DTFP32, tensor.NewShape(inputShape...))
	_, err = conv.Forward(input)
	require.NoError(t, err, "Forward should succeed")

	outputShape, _ := conv.OutputShape(inputShape)
	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}
	gradOutput := tensor.New(tensor.DTFP32, tensor.NewShape(outputShape...))

	// Test with CanLearn = false (inference-only)
	gradInput, err := conv.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed for inference-only")
	assert.True(t, gradInput.Shape().Rank() > 0, "GradInput should have dimensions")
	assert.Equal(t, inputShape, gradInput.Shape().ToSlice(), "GradInput shape should match input shape")

	// Test with CanLearn = true (backward is now implemented)
	conv2, _ := NewConv1D(3, 16, 3, 1, 1, WithCanLearn(true))
	conv2.Init(inputShape)
	conv2.Forward(input)
	gradInput2, err := conv2.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed when CanLearn is true")
	assert.True(t, gradInput2.Shape().Rank() > 0, "GradInput should have dimensions")
	assert.Equal(t, inputShape, gradInput2.Shape().ToSlice(), "GradInput shape should match input shape")

	// Test error cases
	var nilConv *Conv1D
	_, err = nilConv.Backward(gradOutput)
	assert.Error(t, err, "Should return error for nil receiver")

	var emptyGrad types.Tensor
	_, err = conv.Backward(emptyGrad)
	assert.Error(t, err, "Should return error for empty gradOutput")

	// Test without Forward
	conv3, _ := NewConv1D(3, 16, 3, 1, 1)
	conv3.Init(inputShape)
	_, err = conv3.Backward(gradOutput)
	assert.Error(t, err, "Should return error if Forward not called")
}

func TestConv1D_OutputShape(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		expectError bool
		expected    []int
	}{
		{
			name:        "valid",
			inputShape:  []int{1, 3, 10},
			expectError: false,
			expected:    []int{1, 16, 8},
		},
		{
			name:        "with_padding",
			inputShape:  []int{1, 3, 10},
			expectError: false,
			expected:    []int{1, 16, 8},
		},
		{
			name:        "invalid_dimension",
			inputShape:  []int{3, 10},
			expectError: true,
		},
		{
			name:        "invalid_channels",
			inputShape:  []int{1, 2, 10},
			expectError: true,
		},
	}

	conv, err := NewConv1D(3, 16, 3, 1, 0)
	require.NoError(t, err, "Should create Conv1D layer")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputShape, err := conv.OutputShape(tt.inputShape)
			if tt.expectError {
				assert.Error(t, err, "Should return error")
			} else {
				require.NoError(t, err, "OutputShape should succeed")
				assert.Equal(t, tt.expected, outputShape, "Output shape should match")
			}
		})
	}

	// Test nil receiver
	var nilConv *Conv1D
	_, err = nilConv.OutputShape([]int{1, 3, 10})
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestConv1D_Weight(t *testing.T) {
	conv, err := NewConv1D(3, 16, 3, 1, 1)
	require.NoError(t, err, "Should create Conv1D layer")

	weight := conv.Weight()
	assert.Equal(t, []int{16, 3, 3}, weight.Shape().ToSlice(), "Weight shape should match")
	assert.Len(t, weight.Data(), 16*3*3, "Weight data size should match")

	// Test nil receiver
	var nilConv *Conv1D
	weight = nilConv.Weight()
	assert.Len(t, weight.Shape().ToSlice(), 0, "Should return empty tensor for nil receiver")
}

func TestConv1D_Bias(t *testing.T) {
	// Test with bias
	conv, err := NewConv1D(3, 16, 3, 1, 1, UseBias(true))
	require.NoError(t, err, "Should create Conv1D layer")

	bias := conv.Bias()
	assert.Equal(t, []int{16}, bias.Shape().ToSlice(), "Bias shape should match")
	assert.Len(t, bias.Data(), 16, "Bias data size should match")

	// Test without bias
	conv2, err := NewConv1D(3, 16, 3, 1, 1, UseBias(false))
	require.NoError(t, err, "Should create Conv1D layer")

	bias = conv2.Bias()
	assert.Len(t, bias.Shape().ToSlice(), 0, "Bias should be empty when disabled")

	// Test nil receiver
	var nilConv *Conv1D
	bias = nilConv.Bias()
	assert.Len(t, bias.Shape().ToSlice(), 0, "Should return empty tensor for nil receiver")
}

func TestConv1D_SetWeight(t *testing.T) {
	conv, err := NewConv1D(3, 16, 3, 1, 1)
	require.NoError(t, err, "Should create Conv1D layer")

	newWeight := tensor.New(tensor.DTFP32, tensor.NewShape(16, 3, 3))
	newWeightData := newWeight.Data().([]float32)
	for i := range newWeightData {
		newWeightData[i] = float32(i) * 0.01
	}

	err = conv.SetWeight(newWeight)
	require.NoError(t, err, "SetWeight should succeed")

	weight := conv.Weight()
	newWeightData2 := newWeight.Data().([]float32)
	weightData := weight.Data().([]float32)
	assert.Equal(t, newWeightData2, weightData, "Weight data should match")

	// Test error cases
	err = conv.SetWeight(tensor.New(tensor.DTFP32, tensor.NewShape(16, 4, 3)))
	assert.Error(t, err, "Should return error for wrong shape")

	var emptyTensor types.Tensor
	err = conv.SetWeight(emptyTensor)
	assert.Error(t, err, "Should return error for empty tensor")

	var nilConv *Conv1D
	err = nilConv.SetWeight(newWeight)
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestConv1D_SetBias(t *testing.T) {
	conv, err := NewConv1D(3, 16, 3, 1, 1, UseBias(true))
	require.NoError(t, err, "Should create Conv1D layer")

	newBias := tensor.New(tensor.DTFP32, tensor.NewShape(16))
	newBiasData := newBias.Data().([]float32)
	for i := range newBiasData {
		newBiasData[i] = float32(i) * 0.01
	}

	err = conv.SetBias(newBias)
	require.NoError(t, err, "SetBias should succeed")

	bias := conv.Bias()
	newBiasData2 := newBias.Data().([]float32)
	biasData := bias.Data().([]float32)
	assert.Equal(t, newBiasData2, biasData, "Bias data should match")

	// Test error cases
	err = conv.SetBias(tensor.New(tensor.DTFP32, tensor.NewShape(17)))
	assert.Error(t, err, "Should return error for wrong shape")

	var emptyTensor types.Tensor
	err = conv.SetBias(emptyTensor)
	assert.Error(t, err, "Should return error for empty tensor")

	// Test without bias
	conv2, err := NewConv1D(3, 16, 3, 1, 1)
	require.NoError(t, err, "Should create Conv1D layer")

	err = conv2.SetBias(newBias)
	assert.Error(t, err, "Should return error for layer without bias")

	var nilConv *Conv1D
	err = nilConv.SetBias(newBias)
	assert.Error(t, err, "Should return error for nil receiver")
}

// TestConv1D_ComputeOutput tests that Conv1D layers compute outputs correctly
func TestConv1D_ComputeOutput(t *testing.T) {
	// Helper to create weight tensor for Conv1D
	createConv1DWeight := func(outChannels, inChannels, kernelLen int, weightData []float32) types.Tensor {
		return tensor.FromFloat32(tensor.NewShape(outChannels, inChannels, kernelLen), weightData)
	}

	tests := []struct {
		name           string
		inChannels     int
		outChannels    int
		kernelLen      int
		stride         int
		pad            int
		inputShape     []int
		input          tensor.Tensor
		weight         tensor.Tensor
		bias           tensor.Tensor
		expectedShape  []int
		expectedOutput []float32
	}{
		{
			name:        "5_inputs_3_outputs_kernel2",
			inChannels:  5,
			outChannels: 3,
			kernelLen:   2,
			stride:      1,
			pad:         0,
			inputShape:  []int{1, 5, 4},
			input: tensor.FromFloat32(tensor.NewShape(1, 5, 4), []float32{
				1.0, 2.0, 3.0, 4.0, // channel 0
				2.0, 4.0, 6.0, 8.0, // channel 1
				3.0, 6.0, 9.0, 12.0, // channel 2
				4.0, 8.0, 12.0, 16.0, // channel 3
				5.0, 10.0, 15.0, 20.0, // channel 4
			}),
			weight: createConv1DWeight(3, 5, 2, func() []float32 {
				// Weight shape: [outChannels=3, inChannels=5, kernelLen=2]
				// Each output channel uses one input channel at kernel position 0
				weightData := make([]float32, 3*5*2)
				weightData[0*5*2+0*2+0] = 1.0 // out ch 0 -> in ch 0, pos 0
				weightData[1*5*2+1*2+0] = 1.0 // out ch 1 -> in ch 1, pos 0
				weightData[2*5*2+2*2+0] = 1.0 // out ch 2 -> in ch 2, pos 0
				return weightData
			}()),
			bias:          tensor.FromFloat32(tensor.NewShape(3), []float32{0.0, 0.0, 0.0}),
			expectedShape: []int{1, 3, 3},
			expectedOutput: []float32{
				1.0, 2.0, 3.0, // output channel 0
				2.0, 4.0, 6.0, // output channel 1
				3.0, 6.0, 9.0, // output channel 2
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv, err := NewConv1D(tt.inChannels, tt.outChannels, tt.kernelLen, tt.stride, tt.pad, UseBias(true))
			require.NoError(t, err, "Should create Conv1D layer")

			err = conv.SetWeight(tt.weight)
			require.NoError(t, err, "SetWeight should succeed")

			err = conv.SetBias(tt.bias)
			require.NoError(t, err, "SetBias should succeed")

			err = conv.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			output, err := conv.Forward(tt.input)
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

// TestConv1D_BackwardAccuracy tests that conv1d layer backward pass computes parameter gradients correctly
func TestConv1D_BackwardAccuracy(t *testing.T) {
	tests := []struct {
		name               string
		inChannels         int
		outChannels        int
		kernelLen          int
		stride             int
		pad                int
		hasBias            bool
		inputShape         []int
		input              types.Tensor
		weight             types.Tensor
		bias               types.Tensor
		gradOutput         types.Tensor
		expectedWeightGrad []float32
		expectedBiasGrad   []float32
		expectedInputGrad  []float32
	}{
		{
			name:        "simple_1x1_kernel2",
			inChannels:  1,
			outChannels: 1,
			kernelLen:   2,
			stride:      1,
			pad:         0,
			hasBias:     true,
			inputShape:  []int{1, 1, 4},
			input:       tensor.FromFloat32(tensor.NewShape(1, 1, 4), []float32{1.0, 2.0, 3.0, 4.0}),
			weight:      tensor.FromFloat32(tensor.NewShape(1, 1, 2), []float32{1.0, 1.0}),
			bias:        tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
			gradOutput:  tensor.FromFloat32(tensor.NewShape(1, 1, 3), []float32{1.0, 1.0, 1.0}),
			// Weight grad: correlation of input with gradOutput
			// weight[0][0][0]: sum(input[0][0][pos] * gradOutput[0][0][outPos]) where pos=outPos*stride+0-pad
			// outPos=0: pos=0, input[0]=1.0, grad=1.0 → 1.0
			// outPos=1: pos=1, input[1]=2.0, grad=1.0 → 2.0
			// outPos=2: pos=2, input[2]=3.0, grad=1.0 → 3.0
			// Total: 6.0
			// weight[0][0][1]: sum(input[0][0][pos+1] * gradOutput[0][0][outPos])
			// outPos=0: pos=1, input[1]=2.0, grad=1.0 → 2.0
			// outPos=1: pos=2, input[2]=3.0, grad=1.0 → 3.0
			// outPos=2: pos=3, input[3]=4.0, grad=1.0 → 4.0
			// Total: 9.0
			expectedWeightGrad: []float32{6.0, 9.0},
			expectedBiasGrad:   []float32{3.0}, // sum(gradOutput) = 1.0+1.0+1.0
			// Input grad: transposed conv with flipped weights
			// weight flipped: [1.0, 1.0] → [1.0, 1.0] (symmetric, so same)
			// gradInput[0] = weight[1-1-0]=weight[0]=1.0 * gradOutput[0]=1.0 = 1.0
			// gradInput[1] = weight[1-1-1]=weight[0]=1.0 * gradOutput[0]=1.0 + weight[1-1-0]=weight[0]=1.0 * gradOutput[1]=1.0 = 2.0
			// gradInput[2] = weight[1-1-1]=weight[0]=1.0 * gradOutput[1]=1.0 + weight[1-1-0]=weight[0]=1.0 * gradOutput[2]=1.0 = 2.0
			// gradInput[3] = weight[1-1-1]=weight[0]=1.0 * gradOutput[2]=1.0 = 1.0
			// Actually, let me recalculate more carefully:
			// For each outPos and kernel position k:
			//   inPos = outPos*stride + k - pad
			//   gradInput[inPos] += weight[kernelLen-1-k] * gradOutput[outPos]
			// outPos=0, k=0: inPos=0+0-0=0, weight[2-1-0]=weight[1]=1.0, gradInput[0]+=1.0*1.0=1.0
			// outPos=0, k=1: inPos=0+1-0=1, weight[2-1-1]=weight[0]=1.0, gradInput[1]+=1.0*1.0=1.0
			// outPos=1, k=0: inPos=1+0-0=1, weight[1]=1.0, gradInput[1]+=1.0*1.0=1.0
			// outPos=1, k=1: inPos=1+1-0=2, weight[0]=1.0, gradInput[2]+=1.0*1.0=1.0
			// outPos=2, k=0: inPos=2+0-0=2, weight[1]=1.0, gradInput[2]+=1.0*1.0=1.0
			// outPos=2, k=1: inPos=2+1-0=3, weight[0]=1.0, gradInput[3]+=1.0*1.0=1.0
			expectedInputGrad: []float32{1.0, 2.0, 2.0, 1.0},
		},
		{
			name:               "no_bias",
			inChannels:         1,
			outChannels:        1,
			kernelLen:          2,
			stride:             1,
			pad:                0,
			hasBias:            false,
			inputShape:         []int{1, 1, 4},
			input:              tensor.FromFloat32(tensor.NewShape(1, 1, 4), []float32{1.0, 2.0, 3.0, 4.0}),
			weight:             tensor.FromFloat32(tensor.NewShape(1, 1, 2), []float32{1.0, 1.0}),
			bias:               tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
			gradOutput:         tensor.FromFloat32(tensor.NewShape(1, 1, 3), []float32{1.0, 1.0, 1.0}),
			expectedWeightGrad: []float32{6.0, 9.0},
			expectedBiasGrad:   nil, // No bias
			expectedInputGrad:  []float32{1.0, 2.0, 2.0, 1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := []Option{WithCanLearn(true)}
			if tt.hasBias {
				opts = append(opts, UseBias(true))
			}

			conv, err := NewConv1D(tt.inChannels, tt.outChannels, tt.kernelLen, tt.stride, tt.pad, opts...)
			require.NoError(t, err, "Should create Conv1D layer")

			err = conv.SetWeight(tt.weight)
			require.NoError(t, err, "SetWeight should succeed")

			if tt.hasBias {
				err = conv.SetBias(tt.bias)
				require.NoError(t, err, "SetBias should succeed")
			}

			err = conv.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			// Forward pass to store input and output
			_, err = conv.Forward(tt.input)
			require.NoError(t, err, "Forward should succeed")

			// Zero gradients first
			conv.ZeroGrad()

			// Backward pass
			gradInput, err := conv.Backward(tt.gradOutput)
			require.NoError(t, err, "Backward should succeed")

			// Verify weight gradient
			if tt.expectedWeightGrad != nil {
				weightParam, ok := conv.Base.Parameter(nntypes.ParamKernels)
				require.True(t, ok, "Weight parameter should exist")
				require.NotEmpty(t, weightParam.Grad.Shape().ToSlice(), "Weight grad should be allocated")
				weightGradData := weightParam.Grad.Data().([]float32)
				require.Len(t, weightGradData, len(tt.expectedWeightGrad), "Weight grad length should match")
				for i := range tt.expectedWeightGrad {
					assert.InDelta(t, tt.expectedWeightGrad[i], weightGradData[i], 1e-5,
						"Weight grad[%d] should match expected", i)
				}
			}

			// Verify bias gradient
			if tt.hasBias && tt.expectedBiasGrad != nil {
				biasParam, ok := conv.Base.Parameter(nntypes.ParamBiases)
				require.True(t, ok, "Bias parameter should exist")
				require.NotEmpty(t, biasParam.Grad.Shape().ToSlice(), "Bias grad should be allocated")
				biasGradData := biasParam.Grad.Data().([]float32)
				require.Len(t, biasGradData, len(tt.expectedBiasGrad), "Bias grad length should match")
				for i := range tt.expectedBiasGrad {
					assert.InDelta(t, tt.expectedBiasGrad[i], biasGradData[i], 1e-5,
						"Bias grad[%d] should match expected", i)
				}
			}

			// Verify input gradient
			if tt.expectedInputGrad != nil {
				gradInputData := gradInput.Data().([]float32)
				require.Len(t, gradInputData, len(tt.expectedInputGrad), "GradInput length should match")
				for i := range tt.expectedInputGrad {
					assert.InDelta(t, tt.expectedInputGrad[i], gradInputData[i], 1e-5,
						"GradInput[%d] should match expected", i)
				}
			}
		})
	}
}
