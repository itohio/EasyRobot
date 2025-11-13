package learn

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn"
	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/nn/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/stretchr/testify/assert"
)

// TestDenseLayer_SimpleLinearRegression tests that a single Dense layer can learn
// a simple linear function: y = 2*x + 1 (with bias)
// This verifies that optimizer and parameter updates work correctly
func TestDenseLayer_SimpleLinearRegression(t *testing.T) {
	// Create a single dense layer: 1 input -> 1 output with bias
	// This should be able to learn y = 2*x + 1
	dense, err := layers.NewDense(1, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create dense layer: %v", err)
	}

	// Initialize the layer
	if err := dense.Init(tensor.NewShape(1)); err != nil {
		t.Fatalf("Failed to initialize layer: %v", err)
	}

	// Set initial weights and bias to small random values
	// We'll start with weights close to 0 and bias close to 0
	weightParam := dense.Base.Weights()
	weightParam.Data.SetAt(0.1, 0, 0) // Set weight to 0.1
	dense.Base.SetParam(types.ParamWeights, weightParam)

	biasParam := dense.Base.Biases()
	biasParam.Data.SetAt(0.1, 0) // Set bias to 0.1
	dense.Base.SetParam(types.ParamBiases, biasParam)

	// Create training data: y = 2*x + 1
	// We'll use a few points: (0, 1), (1, 3), (2, 5), (-1, -1)
	inputs := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{1}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{2}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{-1}),
	}
	targets := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{1}),  // 2*0 + 1 = 1
		tensor.FromFloat32(tensor.NewShape(1), []float32{3}),  // 2*1 + 1 = 3
		tensor.FromFloat32(tensor.NewShape(1), []float32{5}),  // 2*2 + 1 = 5
		tensor.FromFloat32(tensor.NewShape(1), []float32{-1}), // 2*(-1) + 1 = -1
	}

	// Create loss and optimizer
	lossFn := nn.NewMSE()
	optimizer := NewSGD(0.1) // Learning rate

	// Track initial loss
	initialLoss := 0.0
	for i := range inputs {
		output, err := dense.Forward(inputs[i])
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}
		loss, err := lossFn.Compute(output, targets[i])
		if err != nil {
			t.Fatalf("Loss computation failed: %v", err)
		}
		initialLoss += float64(loss)
	}
	initialLoss /= float64(len(inputs))
	t.Logf("Initial average loss: %.6f", initialLoss)

	// Get initial weight and bias values
	initialWeight := dense.Base.Weights().Data.At(0, 0)
	initialBias := dense.Base.Biases().Data.At(0)
	t.Logf("Initial weight: %.6f, bias: %.6f", initialWeight, initialBias)

	// Create a reusable input tensor for training
	trainingInput := tensor.New(inputs[0].DataType(), inputs[0].Shape())

	// Train for multiple epochs
	epochs := 100
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i := range inputs {
			// Copy original input into training tensor
			trainingInput.Copy(inputs[i])

			loss, err := TrainStep(dense, optimizer, lossFn, trainingInput, targets[i])
			if err != nil {
				t.Fatalf("TrainStep failed at epoch %d, sample %d: %v", epoch, i, err)
			}
			totalLoss += loss
		}

		avgLoss := totalLoss / float64(len(inputs))
		if (epoch+1)%20 == 0 {
			t.Logf("Epoch %d: Average loss = %.6f", epoch+1, avgLoss)
		}

		// Check if converged
		if avgLoss < 0.001 {
			t.Logf("Converged at epoch %d with loss %.6f", epoch+1, avgLoss)
			break
		}
	}

	// Get final weight and bias values
	finalWeight := dense.Base.Weights().Data.At(0, 0)
	finalBias := dense.Base.Biases().Data.At(0)
	t.Logf("Final weight: %.6f, bias: %.6f", finalWeight, finalBias)
	t.Logf("Expected weight: ~2.0, bias: ~1.0")

	// Check final loss
	finalLoss := 0.0
	for i := range inputs {
		output, err := dense.Forward(inputs[i])
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}
		loss, err := lossFn.Compute(output, targets[i])
		if err != nil {
			t.Fatalf("Loss computation failed: %v", err)
		}
		finalLoss += float64(loss)
	}
	finalLoss /= float64(len(inputs))
	t.Logf("Final average loss: %.6f", finalLoss)

	// Verify that loss decreased
	assert.True(t, finalLoss < initialLoss, "Loss should decrease during training: initial=%.6f, final=%.6f", initialLoss, finalLoss)

	// Verify that parameters changed
	assert.NotEqual(t, initialWeight, finalWeight, "Weight should change during training")
	assert.NotEqual(t, initialBias, finalBias, "Bias should change during training")

	// Verify that the learned function is close to y = 2*x + 1
	// Weight should be close to 2.0, bias should be close to 1.0
	assert.InDelta(t, 2.0, finalWeight, 0.5, "Weight should be close to 2.0 (learned function is y = 2*x + 1)")
	assert.InDelta(t, 1.0, finalBias, 0.5, "Bias should be close to 1.0 (learned function is y = 2*x + 1)")

	// Test predictions on training data
	for i := range inputs {
		output, err := dense.Forward(inputs[i])
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}
		predicted := output.At(0)
		expected := targets[i].At(0)
		error := predicted - expected
		t.Logf("Input: %.1f -> Predicted: %.6f, Expected: %.1f, Error: %.6f", inputs[i].At(0), predicted, expected, error)
		assert.InDelta(t, expected, predicted, 0.5, "Prediction should be close to expected value")
	}
}
