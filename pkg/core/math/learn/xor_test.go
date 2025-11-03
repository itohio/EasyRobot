package learn_test

import (
	"math/rand"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/learn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// TestXOR trains a neural network to learn the XOR function.
// XOR is a classic non-linearly separable problem that requires a hidden layer.
func TestXOR(t *testing.T) {
	// XOR truth table:
	// Input: [0, 0] -> Output: 0
	// Input: [0, 1] -> Output: 1
	// Input: [1, 0] -> Output: 1
	// Input: [1, 1] -> Output: 0

	// Create training data
	inputs := []tensor.Tensor{
		{Dim: []int{2}, Data: []float32{0, 0}},
		{Dim: []int{2}, Data: []float32{0, 1}},
		{Dim: []int{2}, Data: []float32{1, 0}},
		{Dim: []int{2}, Data: []float32{1, 1}},
	}
	targets := []tensor.Tensor{
		{Dim: []int{1}, Data: []float32{0}},
		{Dim: []int{1}, Data: []float32{1}},
		{Dim: []int{1}, Data: []float32{1}},
		{Dim: []int{1}, Data: []float32{0}},
	}

	// Build model: 2 inputs -> 2 hidden (ReLU) -> 1 output (Sigmoid)
	hiddenLayer, err := layers.NewDense(2, 2, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create hidden layer: %v", err)
	}

	relu := layers.NewReLU("relu")

	outputLayer, err := layers.NewDense(2, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create output layer: %v", err)
	}

	sigmoid := layers.NewSigmoid("sigmoid")

	model, err := nn.NewModelBuilder([]int{2}).
		AddLayer(hiddenLayer).
		AddLayer(relu).
		AddLayer(outputLayer).
		AddLayer(sigmoid).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize model (allocates output tensors for layers)
	if err := model.Init(); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Initialize weights with small random values
	// Use seed that gives positive values
	rng := rand.New(rand.NewSource(123))
	params := model.Parameters()
	for _, param := range params {
		for i := range param.Data.Data {
			// Generate positive small random values
			param.Data.Data[i] = rng.Float32() * 0.1
		}
	}

	// Create loss and optimizer
	lossFn := nn.NewMSE()
	optimizer := learn.NewSGD(0.5) // Learning rate

	// Train for multiple epochs
	epochs := 2000
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0)

		// Train on all 4 examples
		for i := range inputs {
			loss, err := learn.TrainStep(model, optimizer, lossFn, inputs[i], targets[i])
			if err != nil {
				t.Fatalf("TrainStep failed at epoch %d, sample %d: %v", epoch, i, err)
			}
			totalLoss += loss
		}

		// Every 100 epochs, check progress
		if (epoch+1)%100 == 0 {
			avgLoss := totalLoss / float32(len(inputs))
			t.Logf("Epoch %d: Average loss = %.6f", epoch+1, avgLoss)

			// Check if converged
			if avgLoss < 0.01 {
				t.Logf("Converged at epoch %d with loss %.6f", epoch+1, avgLoss)
				break
			}
		}
	}

	// Test the trained model
	t.Log("\nTesting trained model:")
	allCorrect := true
	for i, input := range inputs {
		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("Forward failed for input %d: %v", i, err)
		}

		expected := targets[i].Data[0]
		predicted := output.Data[0]
		error := abs(predicted - expected)

		t.Logf("Input: [%.0f, %.0f] -> Predicted: %.4f, Expected: %.0f, Error: %.4f",
			input.Data[0], input.Data[1], predicted, expected, error)

		// Consider correct if error < 0.2 (XOR is binary, but we use sigmoid so values are in [0,1])
		if error > 0.2 {
			allCorrect = false
		}
	}

	if !allCorrect {
		t.Errorf("Model did not learn XOR function correctly")
	} else {
		t.Log("\n✓ Model successfully learned XOR function!")
	}
}

// abs returns absolute value of float32
func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
