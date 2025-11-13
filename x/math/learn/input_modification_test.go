package learn

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn"
	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestInputNotModifiedDuringTraining verifies that input tensors are NOT modified
// during training (forward and backward passes).
func TestInputNotModifiedDuringTraining(t *testing.T) {
	// Create a simple dense layer
	dense, err := layers.NewDense(2, 1, layers.WithCanLearn(true))
	require.NoError(t, err)
	require.NoError(t, dense.Init(tensor.NewShape(2)))

	// Create optimizer and loss function
	optimizer := NewSGD(0.1)
	lossFn := nn.NewMSE()

	// Create input tensor with known values
	originalInput := tensor.FromFloat32(tensor.NewShape(2), []float32{0.5, 0.3})
	target := tensor.FromFloat32(tensor.NewShape(1), []float32{0.8})

	// Clone the input to preserve original values for comparison
	inputCopy := originalInput.Clone()
	originalValues := make([]float32, originalInput.Size())
	for i := 0; i < originalInput.Size(); i++ {
		originalValues[i] = float32(originalInput.At(i))
	}

	t.Logf("Original input values: [%.6f, %.6f]", originalValues[0], originalValues[1])

	// Perform forward pass
	output, err := dense.Forward(inputCopy)
	require.NoError(t, err)
	t.Logf("Output after forward: %.6f", output.At(0))

	// Verify input is still unchanged after forward pass
	for i := 0; i < inputCopy.Size(); i++ {
		assert.InDelta(t, float64(originalValues[i]), float64(inputCopy.At(i)), 1e-6,
			"Input was modified during forward pass at index %d", i)
	}

	// Compute loss
	loss, err := lossFn.Compute(output, target)
	require.NoError(t, err)
	t.Logf("Loss: %.6f", loss)

	// Get loss gradient
	gradOutput, err := lossFn.Gradient(output, target)
	require.NoError(t, err)

	// Perform backward pass
	dense.ZeroGrad()
	gradInput, err := dense.Backward(gradOutput)
	require.NoError(t, err)
	require.NotNil(t, gradInput)

	// Verify input is still unchanged after backward pass
	for i := 0; i < inputCopy.Size(); i++ {
		assert.InDelta(t, float64(originalValues[i]), float64(inputCopy.At(i)), 1e-6,
			"Input was modified during backward pass at index %d", i)
	}

	// Update parameters
	err = dense.Update(optimizer)
	require.NoError(t, err)

	// Verify input is still unchanged after parameter update
	for i := 0; i < inputCopy.Size(); i++ {
		assert.InDelta(t, float64(originalValues[i]), float64(inputCopy.At(i)), 1e-6,
			"Input was modified during parameter update at index %d", i)
	}

	// Perform a full training step
	trainingInput := originalInput.Clone()
	trainingInputValues := make([]float32, trainingInput.Size())
	for i := 0; i < trainingInput.Size(); i++ {
		trainingInputValues[i] = float32(trainingInput.At(i))
	}

	_, err = TrainStep(dense, optimizer, lossFn, trainingInput, target)
	require.NoError(t, err)

	// Verify input is still unchanged after full training step
	for i := 0; i < trainingInput.Size(); i++ {
		assert.InDelta(t, float64(trainingInputValues[i]), float64(trainingInput.At(i)), 1e-6,
			"Input was modified during TrainStep at index %d", i)
	}
}

// TestInputNotModifiedDuringInference verifies that input tensors are NOT modified
// during inference (forward pass only).
func TestInputNotModifiedDuringInference(t *testing.T) {
	// Create a simple dense layer
	dense, err := layers.NewDense(2, 1, layers.WithCanLearn(false))
	require.NoError(t, err)
	require.NoError(t, dense.Init(tensor.NewShape(2)))

	// Create input tensor with known values
	originalInput := tensor.FromFloat32(tensor.NewShape(2), []float32{0.7, 0.4})
	inputCopy := originalInput.Clone()
	originalValues := make([]float32, originalInput.Size())
	for i := 0; i < originalInput.Size(); i++ {
		originalValues[i] = float32(originalInput.At(i))
	}

	t.Logf("Original input values: [%.6f, %.6f]", originalValues[0], originalValues[1])

	// Perform multiple forward passes (simulating inference)
	for i := 0; i < 5; i++ {
		output, err := dense.Forward(inputCopy)
		require.NoError(t, err)
		t.Logf("Forward pass %d: output = %.6f", i+1, output.At(0))

		// Verify input is still unchanged after each forward pass
		for j := 0; j < inputCopy.Size(); j++ {
			assert.InDelta(t, float64(originalValues[j]), float64(inputCopy.At(j)), 1e-6,
				"Input was modified during forward pass %d at index %d", i+1, j)
		}
	}
}

// TestInputNotModifiedWithSequentialModel verifies that input tensors are NOT modified
// when using a Sequential model with multiple layers.
func TestInputNotModifiedWithSequentialModel(t *testing.T) {
	// Create a Sequential model with multiple layers
	dense1, err := layers.NewDense(2, 4, layers.WithCanLearn(true))
	require.NoError(t, err)
	relu := layers.NewReLU("relu")
	dense2, err := layers.NewDense(4, 1, layers.WithCanLearn(true))
	require.NoError(t, err)
	sigmoid := layers.NewSigmoid("sigmoid")

	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(2)).
		AddLayer(dense1).
		AddLayer(relu).
		AddLayer(dense2).
		AddLayer(sigmoid).
		Build()
	require.NoError(t, err)
	require.NoError(t, model.Init(tensor.NewShape(2)))

	// Create optimizer and loss function
	optimizer := NewSGD(0.1)
	lossFn := nn.NewMSE()

	// Create input tensor with known values
	originalInput := tensor.FromFloat32(tensor.NewShape(2), []float32{0.6, 0.2})
	target := tensor.FromFloat32(tensor.NewShape(1), []float32{0.9})

	// Clone the input to preserve original values
	inputCopy := originalInput.Clone()
	originalValues := make([]float32, originalInput.Size())
	for i := 0; i < originalInput.Size(); i++ {
		originalValues[i] = float32(originalInput.At(i))
	}

	t.Logf("Original input values: [%.6f, %.6f]", originalValues[0], originalValues[1])

	// Perform a full training step
	_, err = TrainStep(model, optimizer, lossFn, inputCopy, target)
	require.NoError(t, err)

	// Verify input is still unchanged after training step
	for i := 0; i < inputCopy.Size(); i++ {
		assert.InDelta(t, float64(originalValues[i]), float64(inputCopy.At(i)), 1e-6,
			"Input was modified during TrainStep with Sequential model at index %d", i)
	}

	// Perform multiple inference passes
	for i := 0; i < 3; i++ {
		output, err := model.Forward(inputCopy)
		require.NoError(t, err)
		t.Logf("Inference pass %d: output = %.6f", i+1, output.At(0))

		// Verify input is still unchanged after each inference pass
		for j := 0; j < inputCopy.Size(); j++ {
			assert.InDelta(t, float64(originalValues[j]), float64(inputCopy.At(j)), 1e-6,
				"Input was modified during inference pass %d at index %d", i+1, j)
		}
	}
}
