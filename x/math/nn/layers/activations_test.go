package layers

import (
	"math/rand"
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
			inputTensor := tensor.FromFloat32(tensor.NewShape(len(tt.input)), tt.input)

			// Test Init
			err := relu.Init(tensor.NewShape(len(tt.input)))
			require.NoError(t, err, "Init should succeed")

			// Test Forward
			output, err := relu.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			outputData := output.Data().([]float32)
			assert.Equal(t, tt.output, outputData, "Output should match expected")

			// Test Backward - create gradOutput with ones matching input size
			gradOutputData := make([]float32, len(tt.input))
			for i := range gradOutputData {
				gradOutputData[i] = 1.0
			}
			gradOutput := tensor.FromFloat32(tensor.NewShape(len(tt.input)), gradOutputData)
			gradInput, err := relu.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")
			gradInputData := gradInput.Data().([]float32)
			assert.Equal(t, tt.gradInput, gradInputData, "GradInput should match expected")
		})
	}
}

// TestReLU_EdgeCases tests edge cases for ReLU layer
func TestReLU_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilReLU *ReLU
	_, err := nilReLU.Forward(tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0}))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	relu := NewReLU("relu")
	err = relu.Init(tensor.NewShape(3))
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = relu.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	relu2 := NewReLU("relu2")
	input := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0})
	_, err = relu2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward without Forward
	relu3 := NewReLU("relu3")
	err = relu3.Init(tensor.NewShape(3))
	require.NoError(t, err)
	gradOutput := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 1.0, 1.0})
	_, err = relu3.Backward(gradOutput)
	assert.Error(t, err, "Backward should error if Forward not called")

	// Test Backward with empty gradOutput
	relu4 := NewReLU("relu4")
	err = relu4.Init(tensor.NewShape(3))
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0})
	_, err = relu4.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = relu4.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
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
			inputTensor := tensor.FromFloat32(tensor.NewShape(1), []float32{tt.input})

			err := sigmoid.Init(tensor.NewShape(1))
			require.NoError(t, err, "Init should succeed")

			output, err := sigmoid.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			outputData := output.Data().([]float32)
			assert.InDelta(t, tt.output, outputData[0], 1e-6, "Output should match expected")

			// Test Backward
			gradOutput := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
			gradInput, err := sigmoid.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")

			// Sigmoid derivative: output * (1 - output)
			gradInputData := gradInput.Data().([]float32)
			expectedGrad := outputData[0] * (1 - outputData[0])
			assert.InDelta(t, expectedGrad, gradInputData[0], 1e-6, "GradInput should match expected derivative")
		})
	}
}

// TestSigmoid_EdgeCases tests edge cases for Sigmoid layer
func TestSigmoid_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilSigmoid *Sigmoid
	_, err := nilSigmoid.Forward(tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	sigmoid := NewSigmoid("sigmoid")
	err = sigmoid.Init(tensor.NewShape(1))
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = sigmoid.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	sigmoid2 := NewSigmoid("sigmoid2")
	input := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
	_, err = sigmoid2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward without Forward
	sigmoid3 := NewSigmoid("sigmoid3")
	err = sigmoid3.Init(tensor.NewShape(1))
	require.NoError(t, err)
	gradOutput := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
	_, err = sigmoid3.Backward(gradOutput)
	assert.Error(t, err, "Backward should error if Forward not called")

	// Test Backward with empty gradOutput
	sigmoid4 := NewSigmoid("sigmoid4")
	err = sigmoid4.Init(tensor.NewShape(1))
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
	_, err = sigmoid4.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = sigmoid4.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
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
			inputTensor := tensor.FromFloat32(tensor.NewShape(1), []float32{tt.input})

			err := tanh.Init(tensor.NewShape(1))
			require.NoError(t, err, "Init should succeed")

			output, err := tanh.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			outputData := output.Data().([]float32)
			assert.InDelta(t, tt.output, outputData[0], 1e-6, "Output should match expected")

			// Test Backward
			gradOutput := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
			gradInput, err := tanh.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")

			// Tanh derivative: 1 - output^2
			gradInputData := gradInput.Data().([]float32)
			expectedGrad := 1 - outputData[0]*outputData[0]
			assert.InDelta(t, expectedGrad, gradInputData[0], 1e-6, "GradInput should match expected derivative")
		})
	}
}

// TestTanh_EdgeCases tests edge cases for Tanh layer
func TestTanh_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilTanh *Tanh
	_, err := nilTanh.Forward(tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	tanh := NewTanh("tanh")
	err = tanh.Init(tensor.NewShape(1))
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = tanh.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	tanh2 := NewTanh("tanh2")
	input := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
	_, err = tanh2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward without Forward
	tanh3 := NewTanh("tanh3")
	err = tanh3.Init(tensor.NewShape(1))
	require.NoError(t, err)
	gradOutput := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
	_, err = tanh3.Backward(gradOutput)
	assert.Error(t, err, "Backward should error if Forward not called")

	// Test Backward with empty gradOutput
	tanh4 := NewTanh("tanh4")
	err = tanh4.Init(tensor.NewShape(1))
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
	_, err = tanh4.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = tanh4.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
}

// TestSoftmax tests the Softmax activation layer
func TestSoftmax(t *testing.T) {
	softmax := NewSoftmax("softmax", 0)
	inputTensor := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0})

	err := softmax.Init(tensor.NewShape(3))
	require.NoError(t, err, "Init should succeed")

	output, err := softmax.Forward(inputTensor)
	require.NoError(t, err, "Forward should succeed")

	// Check that outputs sum to 1.0
	outputData := output.Data().([]float32)
	sum := float32(0)
	for _, v := range outputData {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 1e-6, "Softmax outputs should sum to 1")

	// Check that all outputs are non-negative
	for i, v := range outputData {
		assert.GreaterOrEqual(t, v, float32(0), "Output[%d] should be non-negative", i)
	}

	// Test Backward
	gradOutput := tensor.FromFloat32(tensor.NewShape(3), []float32{0.1, 0.2, 0.3})
	gradInput, err := softmax.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed")
	gradInputData := gradInput.Data().([]float32)
	assert.Len(t, gradInputData, 3, "GradInput should have size 3")
}

// TestSoftmax_EdgeCases tests edge cases for Softmax layer
func TestSoftmax_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilSoftmax *Softmax
	_, err := nilSoftmax.Forward(tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0}))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	softmax := NewSoftmax("softmax", 0)
	err = softmax.Init(tensor.NewShape(3))
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = softmax.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	softmax2 := NewSoftmax("softmax2", 0)
	input := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0})
	_, err = softmax2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward without Forward
	softmax3 := NewSoftmax("softmax3", 0)
	err = softmax3.Init(tensor.NewShape(3))
	require.NoError(t, err)
	gradOutput := tensor.FromFloat32(tensor.NewShape(3), []float32{0.1, 0.2, 0.3})
	_, err = softmax3.Backward(gradOutput)
	assert.Error(t, err, "Backward should error if Forward not called")

	// Test Backward with empty gradOutput
	softmax4 := NewSoftmax("softmax4", 0)
	err = softmax4.Init(tensor.NewShape(3))
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0})
	_, err = softmax4.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = softmax4.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
}

// TestDropout tests the Dropout layer
func TestDropout(t *testing.T) {
	t.Run("inference_mode_passthrough", func(t *testing.T) {
		dropout := NewDropout("dropout", WithTrainingMode(false))
		input := tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 2.0, 3.0, 4.0})

		err := dropout.Init(tensor.NewShape(4))
		require.NoError(t, err)

		output, err := dropout.Forward(input)
		require.NoError(t, err)
		inputData := input.Data().([]float32)
		outputData := output.Data().([]float32)
		assert.Equal(t, inputData, outputData, "Inference mode should pass through unchanged")

		// Test backward
		gradOutput := tensor.FromFloat32(tensor.NewShape(4), []float32{0.1, 0.2, 0.3, 0.4})
		gradInput, err := dropout.Backward(gradOutput)
		require.NoError(t, err)
		gradOutputData := gradOutput.Data().([]float32)
		gradInputData := gradInput.Data().([]float32)
		assert.Equal(t, gradOutputData, gradInputData, "Inference mode backward should pass through unchanged")
	})

	t.Run("training_mode_p_zero", func(t *testing.T) {
		dropout := NewDropout("dropout", WithDropoutRate(0.0), WithTrainingMode(true))
		input := tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 2.0, 3.0, 4.0})

		err := dropout.Init(tensor.NewShape(4))
		require.NoError(t, err)

		output, err := dropout.Forward(input)
		require.NoError(t, err)
		inputData := input.Data().([]float32)
		outputData := output.Data().([]float32)
		assert.Equal(t, inputData, outputData, "p=0 should pass through unchanged even in training")

		gradOutput := tensor.FromFloat32(tensor.NewShape(4), []float32{0.1, 0.2, 0.3, 0.4})
		gradInput, err := dropout.Backward(gradOutput)
		require.NoError(t, err)
		gradOutputData := gradOutput.Data().([]float32)
		gradInputData := gradInput.Data().([]float32)
		assert.Equal(t, gradOutputData, gradInputData, "p=0 backward should pass through unchanged")
	})

	t.Run("training_mode_deterministic", func(t *testing.T) {
		// Use deterministic RNG for reproducible test
		rng := rand.New(rand.NewSource(42))
		dropout := NewDropout("dropout",
			WithDropoutRate(0.5),
			WithTrainingMode(true),
			WithDropoutRNG(rng),
		)

		input := tensor.FromFloat32(tensor.NewShape(10), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0})

		err := dropout.Init(tensor.NewShape(10))
		require.NoError(t, err)

		output, err := dropout.Forward(input)
		require.NoError(t, err)

		// With p=0.5, some elements should be zero, others scaled by 2.0
		scale := float32(1.0) / (1.0 - 0.5) // = 2.0
		zeroCount := 0
		scaledCount := 0

		outputData := output.Data().([]float32)
		inputData := input.Data().([]float32)
		for i, val := range outputData {
			if val == 0 {
				zeroCount++
				assert.Equal(t, float32(0), val, "Dropped element should be zero")
			} else {
				scaledCount++
				expected := inputData[i] * scale
				assert.InDelta(t, expected, val, 1e-6, "Non-dropped element should be scaled")
			}
		}

		// With deterministic seed, we should get consistent results
		assert.Greater(t, zeroCount, 0, "Some elements should be dropped")
		assert.Greater(t, scaledCount, 0, "Some elements should be kept and scaled")

		// Test backward
		inputShape := input.Shape().ToSlice()
		inputSize := input.Size()
		gradData := make([]float32, inputSize)
		for i := range gradData {
			gradData[i] = 1.0
		}
		gradOutput := tensor.FromFloat32(tensor.NewShape(inputShape...), gradData)

		gradInput, err := dropout.Backward(gradOutput)
		require.NoError(t, err)

		// Gradient should be multiplied by mask (0 or scale)
		gradInputData := gradInput.Data().([]float32)
		for i := range gradInputData {
			if outputData[i] == 0 {
				assert.Equal(t, float32(0), gradInputData[i], "Gradient for dropped element should be zero")
			} else {
				assert.InDelta(t, scale, gradInputData[i], 1e-6, "Gradient for kept element should be scaled")
			}
		}
	})

	t.Run("shape_preservation", func(t *testing.T) {
		shapes := [][]int{
			{10},
			{5, 4},
			{2, 3, 4},
		}

		for _, shape := range shapes {
			shapeTensor := tensor.NewShape(shape...)
			dropout := NewDropout("dropout", WithTrainingMode(false))
			err := dropout.Init(shapeTensor)
			require.NoError(t, err)

			outputShape, err := dropout.OutputShape(shapeTensor)
			require.NoError(t, err)
			assert.Equal(t, shape, outputShape.ToSlice(), "Output shape should match input shape")
		}
	})

	t.Run("training_mode_setter", func(t *testing.T) {
		dropout := NewDropout("dropout")
		assert.False(t, dropout.TrainingMode(), "Default should be inference mode")

		dropout.SetTrainingMode(true)
		assert.True(t, dropout.TrainingMode(), "Should be in training mode after setting")

		dropout.SetTrainingMode(false)
		assert.False(t, dropout.TrainingMode(), "Should be in inference mode after setting")
	})

	t.Run("rate_getter", func(t *testing.T) {
		dropout := NewDropout("dropout", WithDropoutRate(0.3))
		assert.InDelta(t, 0.3, dropout.Rate(), 1e-6, "Rate should match set value")

		dropout2 := NewDropout("dropout2") // Default rate
		assert.InDelta(t, 0.5, dropout2.Rate(), 1e-6, "Default rate should be 0.5")
	})

	t.Run("invalid_rate_ignored", func(t *testing.T) {
		// Invalid rates should be ignored, using default
		dropout := NewDropout("dropout", WithDropoutRate(-0.1))
		assert.InDelta(t, 0.5, dropout.Rate(), 1e-6, "Invalid negative rate should be ignored")

		dropout2 := NewDropout("dropout2", WithDropoutRate(1.0))
		assert.InDelta(t, 0.5, dropout2.Rate(), 1e-6, "Invalid rate >= 1 should be ignored")
	})

	t.Run("empty_input_error", func(t *testing.T) {
		dropout := NewDropout("dropout")
		err := dropout.Init(tensor.NewShape())
		require.Error(t, err, "Init with empty shape should error")

		// Use Empty() to create a truly empty tensor (nil shape and data)
		input := tensor.Empty(tensor.DTFP32)
		_, err = dropout.Forward(input)
		require.Error(t, err, "Forward with empty input should error")
	})

	t.Run("backward_without_forward_error", func(t *testing.T) {
		dropout := NewDropout("dropout")
		err := dropout.Init(tensor.NewShape(4))
		require.NoError(t, err)

		gradOutput := tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 1.0, 1.0, 1.0})
		_, err = dropout.Backward(gradOutput)
		require.Error(t, err, "Backward without Forward should error")
	})
}

// TestDropout_EdgeCases tests edge cases for Dropout layer
func TestDropout_EdgeCases(t *testing.T) {
	// Test nil receiver
	var nilDropout *Dropout
	_, err := nilDropout.Forward(tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 2.0, 3.0, 4.0}))
	assert.Error(t, err, "Forward should error on nil receiver")

	// Test empty input
	dropout := NewDropout("dropout")
	err = dropout.Init(tensor.NewShape(4))
	require.NoError(t, err)

	emptyInput := tensor.Empty(tensor.DTFP32)
	_, err = dropout.Forward(emptyInput)
	assert.Error(t, err, "Forward should error on empty input")

	// Test Forward without Init
	dropout2 := NewDropout("dropout2")
	input := tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 2.0, 3.0, 4.0})
	_, err = dropout2.Forward(input)
	assert.Error(t, err, "Forward should error if Init not called")

	// Test Backward with empty gradOutput
	dropout3 := NewDropout("dropout3")
	err = dropout3.Init(tensor.NewShape(4))
	require.NoError(t, err)
	input2 := tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 2.0, 3.0, 4.0})
	_, err = dropout3.Forward(input2)
	require.NoError(t, err)
	emptyGrad := tensor.Empty(tensor.DTFP32)
	_, err = dropout3.Backward(emptyGrad)
	assert.Error(t, err, "Backward should error on empty gradOutput")
}
