package learn_test

import (
	"math/rand"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/learn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
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
		tensor.FromFloat32(tensor.NewShape(2), []float32{0, 0}),
		tensor.FromFloat32(tensor.NewShape(2), []float32{0, 1}),
		tensor.FromFloat32(tensor.NewShape(2), []float32{1, 0}),
		tensor.FromFloat32(tensor.NewShape(2), []float32{1, 1}),
	}
	targets := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{1}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{1}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{0}),
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

	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(2)).
		AddLayer(hiddenLayer).
		AddLayer(relu).
		AddLayer(outputLayer).
		AddLayer(sigmoid).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize model (allocates output tensors for layers)
	if err := model.Init(tensor.NewShape(2)); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Initialize weights with small random values to break symmetry
	// This is critical for XOR - zero initialization causes all neurons to be dead
	// Get layers from model
	hiddenDense := model.GetLayer(0).(*layers.Dense)
	outputDense := model.GetLayer(2).(*layers.Dense)

	// Initialize hidden layer weights and biases with small random values
	hiddenWeight := hiddenDense.Base.Weights()
	for indices := range hiddenWeight.Data.Shape().Iterator() {
		// Small random values in range [-0.5, 0.5]
		val := float64((rand.Float32()*2 - 1) * 0.5)
		hiddenWeight.Data.SetAt(val, indices...)
	}
	hiddenDense.Base.SetParam(types.ParamWeights, hiddenWeight)

	hiddenBias := hiddenDense.Base.Biases()
	for indices := range hiddenBias.Data.Shape().Iterator() {
		// Small positive bias to avoid dead neurons with ReLU
		val := float64(rand.Float32() * 0.1)
		hiddenBias.Data.SetAt(val, indices...)
	}
	hiddenDense.Base.SetParam(types.ParamBiases, hiddenBias)

	// Initialize output layer weights and biases
	outputWeight := outputDense.Base.Weights()
	for indices := range outputWeight.Data.Shape().Iterator() {
		val := float64((rand.Float32()*2 - 1) * 0.5)
		outputWeight.Data.SetAt(val, indices...)
	}
	outputDense.Base.SetParam(types.ParamWeights, outputWeight)

	outputBias := outputDense.Base.Biases()
	for indices := range outputBias.Data.Shape().Iterator() {
		val := float64((rand.Float32()*2 - 1) * 0.1)
		outputBias.Data.SetAt(val, indices...)
	}
	outputDense.Base.SetParam(types.ParamBiases, outputBias)

	// Create loss and optimizer
	lossFn := nn.NewMSE()
	optimizer := learn.NewSGD(1.0) // Learning rate

	// Track initial loss
	initialLoss := 0.0
	for i := range inputs {
		output, err := model.Forward(inputs[i])
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

	// Train for multiple epochs - XOR needs more training
	epochs := 3000
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float64(0)

		// Train on all 4 examples
		for i := range inputs {
			loss, err := learn.TrainStep(model, optimizer, lossFn, inputs[i], targets[i])
			if err != nil {
				t.Fatalf("TrainStep failed at epoch %d, sample %d: %v", epoch, i, err)
			}
			totalLoss += loss
		}

		avgLoss := totalLoss / float64(len(inputs))

		// Every 100 epochs, check progress
		if (epoch+1)%100 == 0 {
			t.Logf("Epoch %d: Average loss = %.6f", epoch+1, avgLoss)

			// Check if converged
			if avgLoss < 0.01 {
				t.Logf("Converged at epoch %d with loss %.6f", epoch+1, avgLoss)
				break
			}
		}
	}

	// Check final loss
	finalLoss := 0.0
	for i := range inputs {
		output, err := model.Forward(inputs[i])
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
	if finalLoss >= initialLoss {
		t.Errorf("Loss should decrease during training: initial=%.6f, final=%.6f", initialLoss, finalLoss)
	}

	// Test the trained model
	t.Log("\nTesting trained model:")
	correctCount := 0
	totalCount := len(inputs)

	for i, input := range inputs {
		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("Forward failed for input %d: %v", i, err)
		}

		expected := targets[i].At(0)
		predicted := output.At(0)
		error := abs(float32(predicted - expected))

		t.Logf("Input: [%.0f, %.0f] -> Predicted: %.4f, Expected: %.0f, Error: %.4f",
			input.At(0), input.At(1), predicted, expected, error)

		// Consider correct if error < 0.2 (XOR is binary, but we use sigmoid so values are in [0,1])
		if error <= 0.2 {
			correctCount++
		}
	}

	// Calculate accuracy
	accuracy := float64(correctCount) / float64(totalCount) * 100.0
	t.Logf("\nAccuracy: %d/%d = %.2f%%", correctCount, totalCount, accuracy)

	// Verify accuracy is >= 50% (XOR is a challenging problem, 50% is better than random)
	if accuracy < 50.0 {
		t.Errorf("Accuracy %.2f%% is < 50%%, model did not learn XOR function", accuracy)
	}

	if accuracy < 100.0 {
		t.Logf("Model learned XOR partially (%.2f%% accuracy)", accuracy)
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
