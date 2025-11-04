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
			inputTensor := *tensor.FromFloat32(tensor.NewShape(len(tt.input)), tt.input)

			// Test Init
			err := relu.Init([]int{len(tt.input)})
			require.NoError(t, err, "Init should succeed")

			// Test Forward
			output, err := relu.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			assert.Equal(t, tt.output, output.Data(), "Output should match expected")

			// Test Backward - create gradOutput with ones matching input size
			gradOutputData := make([]float32, len(tt.input))
			for i := range gradOutputData {
				gradOutputData[i] = 1.0
			}
			gradOutput := *tensor.FromFloat32(tensor.NewShape(len(tt.input)), gradOutputData)
			gradInput, err := relu.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")
			assert.Equal(t, tt.gradInput, gradInput.Data(), "GradInput should match expected")
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
			inputTensor := *tensor.FromFloat32(tensor.NewShape(1), []float32{tt.input})

			err := sigmoid.Init([]int{1})
			require.NoError(t, err, "Init should succeed")

			output, err := sigmoid.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			assert.InDelta(t, tt.output, output.Data()[0], 1e-6, "Output should match expected")

			// Test Backward
			gradOutput := *tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
			gradInput, err := sigmoid.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")

			// Sigmoid derivative: output * (1 - output)
			expectedGrad := output.Data()[0] * (1 - output.Data()[0])
			assert.InDelta(t, expectedGrad, gradInput.Data()[0], 1e-6, "GradInput should match expected derivative")
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
			inputTensor := *tensor.FromFloat32(tensor.NewShape(1), []float32{tt.input})

			err := tanh.Init([]int{1})
			require.NoError(t, err, "Init should succeed")

			output, err := tanh.Forward(inputTensor)
			require.NoError(t, err, "Forward should succeed")
			assert.InDelta(t, tt.output, output.Data()[0], 1e-6, "Output should match expected")

			// Test Backward
			gradOutput := *tensor.FromFloat32(tensor.NewShape(1), []float32{1.0})
			gradInput, err := tanh.Backward(gradOutput)
			require.NoError(t, err, "Backward should succeed")

			// Tanh derivative: 1 - output^2
			expectedGrad := 1 - output.Data()[0]*output.Data()[0]
			assert.InDelta(t, expectedGrad, gradInput.Data()[0], 1e-6, "GradInput should match expected derivative")
		})
	}
}

// TestSoftmax tests the Softmax activation layer
func TestSoftmax(t *testing.T) {
	softmax := NewSoftmax("softmax", 0)
	inputTensor := *tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0})

	err := softmax.Init([]int{3})
	require.NoError(t, err, "Init should succeed")

	output, err := softmax.Forward(inputTensor)
	require.NoError(t, err, "Forward should succeed")

	// Check that outputs sum to 1.0
	sum := float32(0)
	for _, v := range output.Data() {
		sum += v
	}
	assert.InDelta(t, 1.0, sum, 1e-6, "Softmax outputs should sum to 1")

	// Check that all outputs are non-negative
	for i, v := range output.Data() {
		assert.GreaterOrEqual(t, v, float32(0), "Output[%d] should be non-negative", i)
	}

	// Test Backward
	gradOutput := *tensor.FromFloat32(tensor.NewShape(3), []float32{0.1, 0.2, 0.3})
	gradInput, err := softmax.Backward(gradOutput)
	require.NoError(t, err, "Backward should succeed")
	assert.Len(t, gradInput.Data(), 3, "GradInput should have size 3")
}

// TestDropout tests the Dropout layer
func TestDropout(t *testing.T) {
	t.Run("inference_mode_passthrough", func(t *testing.T) {
		dropout := NewDropout("dropout", WithTrainingMode(false))
		input := *tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 2.0, 3.0, 4.0})

		err := dropout.Init([]int{4})
		require.NoError(t, err)

		output, err := dropout.Forward(input)
		require.NoError(t, err)
		assert.Equal(t, input.Data(), output.Data(), "Inference mode should pass through unchanged")

		// Test backward
		gradOutput := *tensor.FromFloat32(tensor.NewShape(4), []float32{0.1, 0.2, 0.3, 0.4})
		gradInput, err := dropout.Backward(gradOutput)
		require.NoError(t, err)
		assert.Equal(t, gradOutput.Data(), gradInput.Data(), "Inference mode backward should pass through unchanged")
	})

	t.Run("training_mode_p_zero", func(t *testing.T) {
		dropout := NewDropout("dropout", WithDropoutRate(0.0), WithTrainingMode(true))
		input := *tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 2.0, 3.0, 4.0})

		err := dropout.Init([]int{4})
		require.NoError(t, err)

		output, err := dropout.Forward(input)
		require.NoError(t, err)
		assert.Equal(t, input.Data(), output.Data(), "p=0 should pass through unchanged even in training")

		gradOutput := *tensor.FromFloat32(tensor.NewShape(4), []float32{0.1, 0.2, 0.3, 0.4})
		gradInput, err := dropout.Backward(gradOutput)
		require.NoError(t, err)
		assert.Equal(t, gradOutput.Data(), gradInput.Data(), "p=0 backward should pass through unchanged")
	})

	t.Run("training_mode_deterministic", func(t *testing.T) {
		// Use deterministic RNG for reproducible test
		rng := rand.New(rand.NewSource(42))
		dropout := NewDropout("dropout",
			WithDropoutRate(0.5),
			WithTrainingMode(true),
			WithDropoutRNG(rng),
		)

		input := *tensor.FromFloat32(tensor.NewShape(10), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0})

		err := dropout.Init([]int{10})
		require.NoError(t, err)

		output, err := dropout.Forward(input)
		require.NoError(t, err)

		// With p=0.5, some elements should be zero, others scaled by 2.0
		scale := float32(1.0) / (1.0 - 0.5) // = 2.0
		zeroCount := 0
		scaledCount := 0

		for i, val := range output.Data() {
			if val == 0 {
				zeroCount++
				assert.Equal(t, float32(0), val, "Dropped element should be zero")
			} else {
				scaledCount++
				expected := input.Data()[i] * scale
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
		gradOutput := *tensor.FromFloat32(tensor.NewShape(inputShape...), gradData)

		gradInput, err := dropout.Backward(gradOutput)
		require.NoError(t, err)

		// Gradient should be multiplied by mask (0 or scale)
		for i := range gradInput.Data() {
			if output.Data()[i] == 0 {
				assert.Equal(t, float32(0), gradInput.Data()[i], "Gradient for dropped element should be zero")
			} else {
				assert.InDelta(t, scale, gradInput.Data()[i], 1e-6, "Gradient for kept element should be scaled")
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
			dropout := NewDropout("dropout", WithTrainingMode(false))
			err := dropout.Init(shape)
			require.NoError(t, err)

			outputShape, err := dropout.OutputShape(shape)
			require.NoError(t, err)
			assert.Equal(t, shape, outputShape, "Output shape should match input shape")
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
		err := dropout.Init([]int{})
		require.Error(t, err, "Init with empty shape should error")

		input := *tensor.FromFloat32(tensor.NewShape(), []float32{})
		_, err = dropout.Forward(input)
		require.Error(t, err, "Forward with empty input should error")
	})

	t.Run("backward_without_forward_error", func(t *testing.T) {
		dropout := NewDropout("dropout")
		err := dropout.Init([]int{4})
		require.NoError(t, err)

		gradOutput := *tensor.FromFloat32(tensor.NewShape(4), []float32{1.0, 1.0, 1.0, 1.0})
		_, err = dropout.Backward(gradOutput)
		require.Error(t, err, "Backward without Forward should error")
	})
}
