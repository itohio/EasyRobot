package layers

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewConv2D(t *testing.T) {
	tests := []struct {
		name           string
		inChannels     int
		outChannels    int
		kernelH        int
		kernelW        int
		strideH        int
		strideW        int
		padH           int
		padW           int
		opts           []Option
		expectError    bool
		expectBias     bool
		expectCanLearn bool
	}{
		{
			name:           "basic",
			inChannels:     3,
			outChannels:    16,
			kernelH:        3,
			kernelW:        3,
			strideH:        1,
			strideW:        1,
			padH:           1,
			padW:           1,
			expectError:    false,
			expectBias:     false,
			expectCanLearn: false,
		},
		{
			name:           "no_bias",
			inChannels:     3,
			outChannels:    16,
			kernelH:        3,
			kernelW:        3,
			strideH:        1,
			strideW:        1,
			padH:           1,
			padW:           1,
			opts:           []Option{WithBias(false)},
			expectError:    false,
			expectBias:     false,
			expectCanLearn: false,
		},
		{
			name:           "with_can_learn",
			inChannels:     3,
			outChannels:    16,
			kernelH:        3,
			kernelW:        3,
			strideH:        1,
			strideW:        1,
			padH:           1,
			padW:           1,
			opts:           []Option{WithCanLearn(true), WithBias(true)},
			expectError:    false,
			expectBias:     true,
			expectCanLearn: true,
		},
		{
			name:        "with_name",
			inChannels:  3,
			outChannels: 16,
			kernelH:     3,
			kernelW:     3,
			strideH:     1,
			strideW:     1,
			padH:        1,
			padW:        1,
			opts:        []Option{WithName("my_conv2d"), WithBias(true)},
			expectError: false,
			expectBias:  true,
		},
		{
			name:        "invalid_inChannels",
			inChannels:  0,
			outChannels: 16,
			kernelH:     3,
			kernelW:     3,
			strideH:     1,
			strideW:     1,
			padH:        1,
			padW:        1,
			expectError: true,
		},
		{
			name:        "invalid_outChannels",
			inChannels:  3,
			outChannels: 0,
			kernelH:     3,
			kernelW:     3,
			strideH:     1,
			strideW:     1,
			padH:        1,
			padW:        1,
			expectError: true,
		},
		{
			name:        "invalid_kernelH",
			inChannels:  3,
			outChannels: 16,
			kernelH:     0,
			kernelW:     3,
			strideH:     1,
			strideW:     1,
			padH:        1,
			padW:        1,
			expectError: true,
		},
		{
			name:        "invalid_kernelW",
			inChannels:  3,
			outChannels: 16,
			kernelH:     3,
			kernelW:     0,
			strideH:     1,
			strideW:     1,
			padH:        1,
			padW:        1,
			expectError: true,
		},
		{
			name:        "invalid_strideH",
			inChannels:  3,
			outChannels: 16,
			kernelH:     3,
			kernelW:     3,
			strideH:     0,
			strideW:     1,
			padH:        1,
			padW:        1,
			expectError: true,
		},
		{
			name:        "invalid_strideW",
			inChannels:  3,
			outChannels: 16,
			kernelH:     3,
			kernelW:     3,
			strideH:     1,
			strideW:     0,
			padH:        1,
			padW:        1,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv, err := NewConv2D(tt.inChannels, tt.outChannels, tt.kernelH, tt.kernelW, tt.strideH, tt.strideW, tt.padH, tt.padW, tt.opts...)
			if tt.expectError {
				assert.Error(t, err, "Should return error")
				assert.Nil(t, conv, "Should return nil on error")
			} else {
				require.NoError(t, err, "Should create Conv2D layer")
				require.NotNil(t, conv, "Conv2D should not be nil")

				weight := conv.Weight()
				assert.Equal(t, []int{tt.outChannels, tt.inChannels, tt.kernelH, tt.kernelW}, []int(weight.Shape()), "Weight shape should match")

				if tt.expectBias {
					bias := conv.Bias()
					assert.Equal(t, []int{tt.outChannels}, []int(bias.Shape()), "Bias shape should match")
				} else {
					bias := conv.Bias()
					assert.Len(t, bias.Shape(), 0, "Bias should be empty when disabled")
				}

				if tt.name == "with_name" {
					assert.Equal(t, "my_conv2d", conv.Name(), "Name should be set")
				}

				if tt.expectCanLearn {
					assert.True(t, conv.CanLearn(), "CanLearn should be true")
				}
			}
		})
	}
}

func TestConv2D_Init(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		expectError bool
	}{
		{
			name:        "valid_4d",
			inputShape:  []int{1, 3, 32, 32},
			expectError: false,
		},
		{
			name:        "valid_batch",
			inputShape:  []int{2, 3, 32, 32},
			expectError: false,
		},
		{
			name:        "invalid_dimension",
			inputShape:  []int{3, 32, 32},
			expectError: true,
		},
		{
			name:        "invalid_channels",
			inputShape:  []int{1, 2, 32, 32},
			expectError: true,
		},
		{
			name:        "invalid_batch",
			inputShape:  []int{0, 3, 32, 32},
			expectError: true,
		},
		{
			name:        "small_height_with_padding",
			inputShape:  []int{1, 3, 1, 32},
			expectError: false, // Valid: (1 + 2*1 - 3)/1 + 1 = 1
		},
		{
			name:        "small_width_with_padding",
			inputShape:  []int{1, 3, 32, 1},
			expectError: false, // Valid: (1 + 2*1 - 3)/1 + 1 = 1
		},
	}

	conv, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	require.NoError(t, err, "Should create Conv2D layer")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := conv.Init(tt.inputShape)
			if tt.expectError {
				assert.Error(t, err, "Should return error")
			} else {
				require.NoError(t, err, "Init should succeed")
				output := conv.Output()
				assert.NotEmpty(t, output.Shape(), "Output should be allocated")
			}
		})
	}

	// Test nil receiver
	var nilConv *Conv2D
	err = nilConv.Init([]int{1, 3, 32, 32})
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestConv2D_Forward(t *testing.T) {
	conv, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	require.NoError(t, err, "Should create Conv2D layer")

	inputShape := []int{1, 3, 32, 32}
	err = conv.Init(inputShape)
	require.NoError(t, err, "Init should succeed")

	input := *tensor.FromFloat32(tensor.NewShape(inputShape...), make([]float32, 1*3*32*32))
	inputData := input.Data()
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}

	output, err := conv.Forward(input)
	require.NoError(t, err, "Forward should succeed")
	assert.NotEmpty(t, output.Shape(), "Output should have dimensions")
	assert.NotEmpty(t, output.Data(), "Output should have data")

	// Verify output shape
	expectedShape, err := conv.OutputShape(inputShape)
	require.NoError(t, err, "OutputShape should succeed")
	assert.Equal(t, expectedShape, []int(output.Shape()), "Output shape should match")

	// Test error cases
	var nilConv *Conv2D
	_, err = nilConv.Forward(input)
	assert.Error(t, err, "Should return error for nil receiver")

	emptyInput := tensor.Tensor{}
	_, err = conv.Forward(emptyInput)
	assert.Error(t, err, "Should return error for empty input")

	// Test without Init
	conv2, _ := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	_, err = conv2.Forward(input)
	assert.Error(t, err, "Should return error if Init not called")
}

func TestConv2D_Backward(t *testing.T) {
	conv, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	require.NoError(t, err, "Should create Conv2D layer")

	inputShape := []int{1, 3, 32, 32}
	err = conv.Init(inputShape)
	require.NoError(t, err, "Init should succeed")

	input := *tensor.FromFloat32(tensor.NewShape(inputShape...), make([]float32, 1*3*32*32))
	_, err = conv.Forward(input)
	require.NoError(t, err, "Forward should succeed")

	outputShape, _ := conv.OutputShape(inputShape)
	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}
	gradOutput := *tensor.FromFloat32(tensor.NewShape(outputShape...), make([]float32, outputSize))

	// Test with CanLearn = false (inference-only)
	gradInput, err := conv.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed for inference-only")
	assert.NotEmpty(t, gradInput.Shape(), "GradInput should have dimensions")
	assert.Equal(t, inputShape, []int(gradInput.Shape()), "GradInput shape should match input shape")

	// Test with CanLearn = true (backward is now implemented)
	conv2, _ := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1, WithCanLearn(true))
	conv2.Init(inputShape)
	conv2.Forward(input)
	gradInput2, err := conv2.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed when CanLearn is true")
	assert.NotEmpty(t, gradInput2.Shape(), "GradInput should have dimensions")
	assert.Equal(t, inputShape, gradInput2.Shape().ToSlice(), "GradInput shape should match input shape")

	// Test error cases
	var nilConv *Conv2D
	_, err = nilConv.Backward(gradOutput)
	assert.Error(t, err, "Should return error for nil receiver")

	emptyGrad := tensor.Tensor{}
	_, err = conv.Backward(emptyGrad)
	assert.Error(t, err, "Should return error for empty gradOutput")

	// Test without Forward
	conv3, _ := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	conv3.Init(inputShape)
	_, err = conv3.Backward(gradOutput)
	assert.Error(t, err, "Should return error if Forward not called")
}

func TestConv2D_OutputShape(t *testing.T) {
	tests := []struct {
		name        string
		inputShape  []int
		expectError bool
		expected    []int
	}{
		{
			name:        "valid",
			inputShape:  []int{1, 3, 32, 32},
			expectError: false,
			expected:    []int{1, 16, 32, 32},
		},
		{
			name:        "with_padding",
			inputShape:  []int{1, 3, 32, 32},
			expectError: false,
			expected:    []int{1, 16, 32, 32},
		},
		{
			name:        "invalid_dimension",
			inputShape:  []int{3, 32, 32},
			expectError: true,
		},
		{
			name:        "invalid_channels",
			inputShape:  []int{1, 2, 32, 32},
			expectError: true,
		},
	}

	conv, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	require.NoError(t, err, "Should create Conv2D layer")

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
	var nilConv *Conv2D
	_, err = nilConv.OutputShape([]int{1, 3, 32, 32})
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestConv2D_Weight(t *testing.T) {
	conv, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	require.NoError(t, err, "Should create Conv2D layer")

	weight := conv.Weight()
	assert.Equal(t, []int{16, 3, 3, 3}, []int(weight.Shape()), "Weight shape should match")
	assert.Len(t, weight.Data(), 16*3*3*3, "Weight data size should match")

	// Test nil receiver
	var nilConv *Conv2D
	weight = nilConv.Weight()
	assert.Len(t, weight.Shape(), 0, "Should return empty tensor for nil receiver")
}

func TestConv2D_Bias(t *testing.T) {
	// Test with bias
	conv, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1, WithBias(true))
	require.NoError(t, err, "Should create Conv2D layer")

	bias := conv.Bias()
	assert.Equal(t, []int{16}, []int(bias.Shape()), "Bias shape should match")
	assert.Len(t, bias.Data(), 16, "Bias data size should match")

	// Test without bias
	conv2, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1, WithBias(false))
	require.NoError(t, err, "Should create Conv2D layer")

	bias = conv2.Bias()
	assert.Len(t, bias.Shape(), 0, "Bias should be empty when disabled")

	// Test nil receiver
	var nilConv *Conv2D
	bias = nilConv.Bias()
	assert.Len(t, bias.Shape(), 0, "Should return empty tensor for nil receiver")
}

func TestConv2D_SetWeight(t *testing.T) {
	conv, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	require.NoError(t, err, "Should create Conv2D layer")

	newWeight := *tensor.FromFloat32(tensor.NewShape(16, 3, 3, 3), make([]float32, 16*3*3*3))
	newWeightData := newWeight.Data()
	for i := range newWeightData {
		newWeightData[i] = float32(i) * 0.001
	}

	err = conv.SetWeight(newWeight)
	require.NoError(t, err, "SetWeight should succeed")

	weight := conv.Weight()
	assert.Equal(t, newWeight.Data(), weight.Data(), "Weight data should match")

	// Test error cases
	err = conv.SetWeight(*tensor.FromFloat32(tensor.NewShape(16, 4, 3, 3), make([]float32, 576)))
	assert.Error(t, err, "Should return error for wrong shape")

	err = conv.SetWeight(tensor.Tensor{})
	assert.Error(t, err, "Should return error for empty tensor")

	var nilConv *Conv2D
	err = nilConv.SetWeight(newWeight)
	assert.Error(t, err, "Should return error for nil receiver")
}

func TestConv2D_SetBias(t *testing.T) {
	conv, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1, WithBias(true))
	require.NoError(t, err, "Should create Conv2D layer")

	newBias := *tensor.FromFloat32(tensor.NewShape(16), make([]float32, 16))
	newBiasData := newBias.Data()
	for i := range newBiasData {
		newBiasData[i] = float32(i) * 0.01
	}

	err = conv.SetBias(newBias)
	require.NoError(t, err, "SetBias should succeed")

	bias := conv.Bias()
	assert.Equal(t, newBias.Data(), bias.Data(), "Bias data should match")

	// Test error cases
	err = conv.SetBias(*tensor.FromFloat32(tensor.NewShape(17), make([]float32, 17)))
	assert.Error(t, err, "Should return error for wrong shape")

	err = conv.SetBias(tensor.Tensor{})
	assert.Error(t, err, "Should return error for empty tensor")

	// Test without bias
	conv2, err := NewConv2D(3, 16, 3, 3, 1, 1, 1, 1)
	require.NoError(t, err, "Should create Conv2D layer")

	err = conv2.SetBias(newBias)
	assert.Error(t, err, "Should return error for layer without bias")

	var nilConv *Conv2D
	err = nilConv.SetBias(newBias)
	assert.Error(t, err, "Should return error for nil receiver")
}

// TestConv2D_ComputeOutput tests that Conv2D layers compute outputs correctly
func TestConv2D_ComputeOutput(t *testing.T) {
	tests := []struct {
		name           string
		inChannels     int
		outChannels    int
		kernelH        int
		kernelW        int
		strideH        int
		strideW        int
		padH           int
		padW           int
		inputShape     []int
		input          tensor.Tensor
		weight         tensor.Tensor
		bias           tensor.Tensor
		expectedShape  []int
		expectedOutput []float32
	}{
		{
			name:        "5x5_input_2x2_kernel_4x4_output",
			inChannels:  1,
			outChannels: 1,
			kernelH:     2,
			kernelW:     2,
			strideH:     1,
			strideW:     1,
			padH:        0,
			padW:        0,
			inputShape:  []int{1, 1, 5, 5},
			input: *tensor.FromFloat32(
				tensor.NewShape(1, 1, 5, 5),
				[]float32{
					1.0, 2.0, 3.0, 4.0, 5.0,
					6.0, 7.0, 8.0, 9.0, 10.0,
					11.0, 12.0, 13.0, 14.0, 15.0,
					16.0, 17.0, 18.0, 19.0, 20.0,
					21.0, 22.0, 23.0, 24.0, 25.0,
				},
			),
			weight: *tensor.FromFloat32(
				tensor.NewShape(1, 1, 2, 2),
				[]float32{1.0, 1.0, 1.0, 1.0},
			),
			bias: *tensor.FromFloat32(
				tensor.NewShape(1),
				[]float32{0.0},
			),
			expectedShape: []int{1, 1, 4, 4},
			expectedOutput: []float32{
				16.0, 20.0, 24.0, 28.0,
				36.0, 40.0, 44.0, 48.0,
				56.0, 60.0, 64.0, 68.0,
				76.0, 80.0, 84.0, 88.0,
			},
		},
		{
			name:        "6x6_input_2x2_kernel_3x3_output",
			inChannels:  1,
			outChannels: 1,
			kernelH:     2,
			kernelW:     2,
			strideH:     2,
			strideW:     2,
			padH:        0,
			padW:        0,
			inputShape:  []int{1, 1, 6, 6},
			input: *tensor.FromFloat32(
				tensor.NewShape(1, 1, 6, 6),
				[]float32{
					1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
					7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
					13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
					19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
					25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
					31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
				},
			),
			weight: *tensor.FromFloat32(
				tensor.NewShape(1, 1, 2, 2),
				[]float32{1.0, 1.0, 1.0, 1.0},
			),
			bias: *tensor.FromFloat32(
				tensor.NewShape(1),
				[]float32{0.0},
			),
			expectedShape: []int{1, 1, 3, 3},
			expectedOutput: []float32{
				18.0, 26.0, 34.0,
				66.0, 74.0, 82.0,
				114.0, 122.0, 130.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv, err := NewConv2D(tt.inChannels, tt.outChannels, tt.kernelH, tt.kernelW, tt.strideH, tt.strideW, tt.padH, tt.padW, WithBias(true))
			require.NoError(t, err, "Should create Conv2D layer")

			err = conv.SetWeight(tt.weight)
			require.NoError(t, err, "SetWeight should succeed")

			err = conv.SetBias(tt.bias)
			require.NoError(t, err, "SetBias should succeed")

			err = conv.Init(tt.inputShape)
			require.NoError(t, err, "Init should succeed")

			output, err := conv.Forward(tt.input)
			require.NoError(t, err, "Forward should succeed")

			assert.Equal(t, tt.expectedShape, []int(output.Shape()), "Output shape should match")
			require.Len(t, output.Data(), len(tt.expectedOutput), "Output data length should match")
			for i := range tt.expectedOutput {
				assert.InDelta(t, tt.expectedOutput[i], output.Data()[i], 1e-5, "Output[%d] should match expected", i)
			}
		})
	}
}

// TestConv2D_BackwardAccuracy tests that conv2d layer backward pass computes parameter gradients correctly
func TestConv2D_BackwardAccuracy(t *testing.T) {
	tests := []struct {
		name               string
		inChannels         int
		outChannels        int
		kernelH            int
		kernelW            int
		strideH            int
		strideW            int
		padH               int
		padW               int
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
			name:        "simple_1x1_kernel2x2",
			inChannels:  1,
			outChannels: 1,
			kernelH:     2,
			kernelW:     2,
			strideH:     1,
			strideW:     1,
			padH:        0,
			padW:        0,
			hasBias:     true,
			inputShape:  []int{1, 1, 3, 3},
			input: *tensor.FromFloat32(
				tensor.Shape([]int{1, 1, 3, 3}),
				[]float32{
					// Input layout: [batch=1, channels=1, height=3, width=3]
					// Row 0: 1.0, 2.0, 3.0
					// Row 1: 4.0, 5.0, 6.0
					// Row 2: 7.0, 8.0, 9.0
					1.0, 2.0, 3.0,
					4.0, 5.0, 6.0,
					7.0, 8.0, 9.0,
				},
			),
			weight: *tensor.FromFloat32(
				tensor.Shape([]int{1, 1, 2, 2}),
				[]float32{1.0, 1.0, 1.0, 1.0}, // All weights are 1.0
			),
			bias: *tensor.FromFloat32(
				tensor.Shape([]int{1}),
				[]float32{0.0},
			),
			gradOutput: *tensor.FromFloat32(
				tensor.Shape([]int{1, 1, 2, 2}),
				[]float32{
					// Output is 2x2, all values are 1.0
					1.0, 1.0,
					1.0, 1.0,
				},
			),
			// Weight grad: correlation of input with gradOutput
			// weight[0][0][kh][kw] = sum over batch and output positions of input[inH][inW] * gradOutput[outH][outW]
			// where inH = outH*strideH + kh - padH, inW = outW*strideW + kw - padW
			// weight[0][0][0][0]:
			//   outH=0,outW=0: inH=0,inW=0, input[0][0]=1.0, grad=1.0 → 1.0
			//   outH=0,outW=1: inH=0,inW=1, input[0][1]=2.0, grad=1.0 → 2.0
			//   outH=1,outW=0: inH=1,inW=0, input[1][0]=4.0, grad=1.0 → 4.0
			//   outH=1,outW=1: inH=1,inW=1, input[1][1]=5.0, grad=1.0 → 5.0
			//   Total: 12.0
			// weight[0][0][0][1]:
			//   outH=0,outW=0: inH=0,inW=1, input[0][1]=2.0, grad=1.0 → 2.0
			//   outH=0,outW=1: inH=0,inW=2, input[0][2]=3.0, grad=1.0 → 3.0
			//   outH=1,outW=0: inH=1,inW=1, input[1][1]=5.0, grad=1.0 → 5.0
			//   outH=1,outW=1: inH=1,inW=2, input[1][2]=6.0, grad=1.0 → 6.0
			//   Total: 16.0
			// weight[0][0][1][0]:
			//   outH=0,outW=0: inH=1,inW=0, input[1][0]=4.0, grad=1.0 → 4.0
			//   outH=0,outW=1: inH=1,inW=1, input[1][1]=5.0, grad=1.0 → 5.0
			//   outH=1,outW=0: inH=2,inW=0, input[2][0]=7.0, grad=1.0 → 7.0
			//   outH=1,outW=1: inH=2,inW=1, input[2][1]=8.0, grad=1.0 → 8.0
			//   Total: 24.0
			// weight[0][0][1][1]:
			//   outH=0,outW=0: inH=1,inW=1, input[1][1]=5.0, grad=1.0 → 5.0
			//   outH=0,outW=1: inH=1,inW=2, input[1][2]=6.0, grad=1.0 → 6.0
			//   outH=1,outW=0: inH=2,inW=1, input[2][1]=8.0, grad=1.0 → 8.0
			//   outH=1,outW=1: inH=2,inW=2, input[2][2]=9.0, grad=1.0 → 9.0
			//   Total: 28.0
			expectedWeightGrad: []float32{12.0, 16.0, 24.0, 28.0},
			expectedBiasGrad:   []float32{4.0}, // sum(gradOutput) = 1.0+1.0+1.0+1.0
			// Input grad: transposed conv with flipped weights
			// Flipped weight: [1.0, 1.0, 1.0, 1.0] → [1.0, 1.0, 1.0, 1.0] (symmetric for all 1s)
			// For each output position and kernel position, accumulate into input gradient
			// gradInput[h][w] = sum over (outH,outW,kh,kw) where h=outH*strideH+kh-padH, w=outW*strideW+kw-padW
			//   of weight[kernelH-1-kh][kernelW-1-kw] * gradOutput[outH][outW]
			// Simplified calculation:
			// Position (0,0): receives from outH=0,outW=0 with kh=0,kw=0 (weight[1][1]=1.0*1.0=1.0)
			// Position (0,1): receives from outH=0,outW=0 with kh=0,kw=1 (weight[1][0]=1.0*1.0=1.0)
			//                   and outH=0,outW=1 with kh=0,kw=0 (weight[1][1]=1.0*1.0=1.0) = 2.0
			// Position (0,2): receives from outH=0,outW=1 with kh=0,kw=1 (weight[1][0]=1.0*1.0=1.0) = 1.0
			// Position (1,0): receives from outH=0,outW=0 with kh=1,kw=0 (weight[0][1]=1.0*1.0=1.0)
			//                   and outH=1,outW=0 with kh=0,kw=0 (weight[1][1]=1.0*1.0=1.0) = 2.0
			// Position (1,1): receives from all 4 output positions = 4.0
			// Position (1,2): receives from outH=0,outW=1 with kh=1,kw=1 (weight[0][0]=1.0*1.0=1.0)
			//                   and outH=1,outW=1 with kh=0,kw=1 (weight[1][0]=1.0*1.0=1.0) = 2.0
			// Position (2,0): receives from outH=1,outW=0 with kh=1,kw=0 (weight[0][1]=1.0*1.0=1.0) = 1.0
			// Position (2,1): receives from outH=1,outW=0 with kh=1,kw=1 (weight[0][0]=1.0*1.0=1.0)
			//                   and outH=1,outW=1 with kh=1,kw=0 (weight[0][1]=1.0*1.0=1.0) = 2.0
			// Position (2,2): receives from outH=1,outW=1 with kh=1,kw=1 (weight[0][0]=1.0*1.0=1.0) = 1.0
			expectedInputGrad: []float32{1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0},
		},
		{
			name:        "no_bias",
			inChannels:  1,
			outChannels: 1,
			kernelH:     2,
			kernelW:     2,
			strideH:     1,
			strideW:     1,
			padH:        0,
			padW:        0,
			hasBias:     false,
			inputShape:  []int{1, 1, 3, 3},
			input: *tensor.FromFloat32(
				tensor.Shape{1, 1, 3, 3},
				[]float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
			),
			weight: *tensor.FromFloat32(
				tensor.Shape{1, 1, 2, 2},
				[]float32{1.0, 1.0, 1.0, 1.0},
			),
			bias: *tensor.FromFloat32(
				tensor.Shape{1},
				[]float32{0.0},
			),
			gradOutput: *tensor.FromFloat32(
				tensor.Shape{1, 1, 2, 2},
				[]float32{1.0, 1.0, 1.0, 1.0},
			),
			expectedWeightGrad: []float32{12.0, 16.0, 24.0, 28.0},
			expectedBiasGrad:   nil, // No bias
			expectedInputGrad:  []float32{1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := []Option{WithCanLearn(true)}
			if tt.hasBias {
				opts = append(opts, WithBias(true))
			}

			conv, err := NewConv2D(tt.inChannels, tt.outChannels, tt.kernelH, tt.kernelW, tt.strideH, tt.strideW, tt.padH, tt.padW, opts...)
			require.NoError(t, err, "Should create Conv2D layer")

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
				weightParam, ok := conv.Base.Parameter(ParamKernels)
				require.True(t, ok, "Weight parameter should exist")
				require.NotEmpty(t, weightParam.Grad.Shape(), "Weight grad should be allocated")
				require.Len(t, weightParam.Grad.Data(), len(tt.expectedWeightGrad), "Weight grad length should match")
				for i := range tt.expectedWeightGrad {
					assert.InDelta(t, tt.expectedWeightGrad[i], weightParam.Grad.Data()[i], 1e-5,
						"Weight grad[%d] should match expected", i)
				}
			}

			// Verify bias gradient
			if tt.hasBias && tt.expectedBiasGrad != nil {
				biasParam, ok := conv.Base.Parameter(ParamBiases)
				require.True(t, ok, "Bias parameter should exist")
				require.NotEmpty(t, biasParam.Grad.Shape(), "Bias grad should be allocated")
				require.Len(t, biasParam.Grad.Data(), len(tt.expectedBiasGrad), "Bias grad length should match")
				for i := range tt.expectedBiasGrad {
					assert.InDelta(t, tt.expectedBiasGrad[i], biasParam.Grad.Data()[i], 1e-5,
						"Bias grad[%d] should match expected", i)
				}
			}

			// Verify input gradient
			if tt.expectedInputGrad != nil {
				require.Len(t, gradInput.Data(), len(tt.expectedInputGrad), "GradInput length should match")
				for i := range tt.expectedInputGrad {
					assert.InDelta(t, tt.expectedInputGrad[i], gradInput.Data()[i], 1e-5,
						"GradInput[%d] should match expected", i)
				}
			}
		})
	}
}
