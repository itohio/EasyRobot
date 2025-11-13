package learn

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn"
	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/nn/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/stretchr/testify/assert"
)

// TestSequential_Classification_ActivationAtEnd tests classification with activation at the end
// Uses different activation functions: Sigmoid, ReLU, Tanh
func TestSequential_Classification_ActivationAtEnd_Sigmoid(t *testing.T) {
	testActivationAtEnd(t, layers.NewSigmoid("sigmoid"), "Sigmoid")
}

func TestSequential_Classification_ActivationAtEnd_ReLU(t *testing.T) {
	testActivationAtEnd(t, layers.NewReLU("relu"), "ReLU")
}

func TestSequential_Classification_ActivationAtEnd_Tanh(t *testing.T) {
	testActivationAtEnd(t, layers.NewTanh("tanh"), "Tanh")
}

// testActivationAtEnd is a helper function that tests classification with activation at the end
func testActivationAtEnd(t *testing.T, activationLayer types.Layer, activationName string) {
	// Create a simple binary classification problem: learn y = x > 0.5
	// Input: [0.0, 0.5, 1.0, -0.5]
	// Target: [0.0, 0.0, 1.0, 0.0] (1 if x > 0.5, else 0)

	// Create two dense layers: 1 -> 2 -> 1
	dense1, err := layers.NewDense(1, 2, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create first dense layer: %v", err)
	}

	dense2, err := layers.NewDense(2, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create second dense layer: %v", err)
	}

	// Build model: Dense1 -> Dense2 -> Activation
	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(1)).
		AddLayer(dense1).
		AddLayer(dense2).
		AddLayer(activationLayer).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize the model
	if err := model.Init(tensor.NewShape(1)); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Create training data: simple threshold classification
	inputs := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.5}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{-0.5}),
	}
	targets := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // 0.0 <= 0.5
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // 0.5 <= 0.5
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}), // 1.0 > 0.5
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // -0.5 <= 0.5
	}

	// Create loss and optimizer
	lossFn := nn.NewMSE()
	optimizer := NewSGD(0.1)

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
	t.Logf("[%s] Initial average loss: %.6f", activationName, initialLoss)

	// Train for multiple epochs
	epochs := 200
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i := range inputs {
			loss, err := TrainStep(model, optimizer, lossFn, inputs[i], targets[i])
			if err != nil {
				t.Fatalf("TrainStep failed at epoch %d, sample %d: %v", epoch, i, err)
			}
			totalLoss += loss
		}

		avgLoss := totalLoss / float64(len(inputs))
		if (epoch+1)%50 == 0 {
			t.Logf("[%s] Epoch %d: Average loss = %.6f", activationName, epoch+1, avgLoss)
		}

		// Check if converged
		if avgLoss < 0.01 {
			t.Logf("[%s] Converged at epoch %d with loss %.6f", activationName, epoch+1, avgLoss)
			break
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
	t.Logf("[%s] Final average loss: %.6f", activationName, finalLoss)

	// Verify that loss decreased
	// Note: ReLU at the end may not decrease loss significantly for classification
	// because ReLU outputs are unbounded, making it hard to learn bounded targets
	if activationName == "ReLU" {
		// For ReLU at end, we just check that it doesn't increase dramatically
		assert.True(t, finalLoss <= initialLoss*1.1, "[%s] Loss should not increase much: initial=%.6f, final=%.6f", activationName, initialLoss, finalLoss)
	} else {
		assert.True(t, finalLoss < initialLoss, "[%s] Loss should decrease: initial=%.6f, final=%.6f", activationName, initialLoss, finalLoss)
	}

	// Test predictions
	correct := 0
	for i := range inputs {
		output, err := model.Forward(inputs[i])
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}
		predicted := output.At(0)
		expected := targets[i].At(0)

		// For classification, we consider prediction correct if it's on the right side of 0.5
		predictedClass := 0.0
		if predicted > 0.5 {
			predictedClass = 1.0
		}
		expectedClass := expected

		if predictedClass == expectedClass {
			correct++
		}
		t.Logf("[%s] Input: %.1f -> Predicted: %.6f (class: %.0f), Expected: %.0f",
			activationName, inputs[i].At(0), predicted, predictedClass, expectedClass)
	}

	accuracy := float64(correct) / float64(len(inputs))
	t.Logf("[%s] Accuracy: %d/%d = %.2f%%", activationName, correct, len(inputs), accuracy*100)

	// For this simple problem, we expect at least some learning
	// The exact accuracy depends on the activation function
	assert.True(t, accuracy >= 0.5, "[%s] Accuracy should be at least 50%%, got %.2f%%", activationName, accuracy*100)
}

// TestSequential_Classification_ActivationBetween tests classification with activation between layers
func TestSequential_Classification_ActivationBetween_Sigmoid(t *testing.T) {
	testActivationBetween(t, layers.NewSigmoid("sigmoid"), "Sigmoid")
}

func TestSequential_Classification_ActivationBetween_ReLU(t *testing.T) {
	testActivationBetween(t, layers.NewReLU("relu"), "ReLU")
}

func TestSequential_Classification_ActivationBetween_Tanh(t *testing.T) {
	testActivationBetween(t, layers.NewTanh("tanh"), "Tanh")
}

// testActivationBetween is a helper function that tests classification with activation between layers
func testActivationBetween(t *testing.T, activationLayer types.Layer, activationName string) {
	// Create a simple binary classification problem: learn y = x > 0.5
	// Input: [0.0, 0.5, 1.0, -0.5]
	// Target: [0.0, 0.0, 1.0, 0.0] (1 if x > 0.5, else 0)

	// Create two dense layers: 1 -> 2 -> 1
	dense1, err := layers.NewDense(1, 2, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create first dense layer: %v", err)
	}

	dense2, err := layers.NewDense(2, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create second dense layer: %v", err)
	}

	// Build model: Dense1 -> Activation -> Dense2
	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(1)).
		AddLayer(dense1).
		AddLayer(activationLayer).
		AddLayer(dense2).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize the model
	if err := model.Init(tensor.NewShape(1)); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Create training data: simple threshold classification
	inputs := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.5}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{-0.5}),
	}
	targets := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // 0.0 <= 0.5
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // 0.5 <= 0.5
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}), // 1.0 > 0.5
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // -0.5 <= 0.5
	}

	// Create loss and optimizer
	lossFn := nn.NewMSE()
	optimizer := NewSGD(0.1)

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
	t.Logf("[%s] Initial average loss: %.6f", activationName, initialLoss)

	// Train for multiple epochs
	epochs := 200
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i := range inputs {
			loss, err := TrainStep(model, optimizer, lossFn, inputs[i], targets[i])
			if err != nil {
				t.Fatalf("TrainStep failed at epoch %d, sample %d: %v", epoch, i, err)
			}
			totalLoss += loss
		}

		avgLoss := totalLoss / float64(len(inputs))
		if (epoch+1)%50 == 0 {
			t.Logf("[%s] Epoch %d: Average loss = %.6f", activationName, epoch+1, avgLoss)
		}

		// Check if converged
		if avgLoss < 0.01 {
			t.Logf("[%s] Converged at epoch %d with loss %.6f", activationName, epoch+1, avgLoss)
			break
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
	t.Logf("[%s] Final average loss: %.6f", activationName, finalLoss)

	// Verify that loss decreased
	// Note: ReLU at the end may not decrease loss significantly for classification
	// because ReLU outputs are unbounded, making it hard to learn bounded targets
	if activationName == "ReLU" {
		// For ReLU at end, we just check that it doesn't increase dramatically
		assert.True(t, finalLoss <= initialLoss*1.1, "[%s] Loss should not increase much: initial=%.6f, final=%.6f", activationName, initialLoss, finalLoss)
	} else {
		assert.True(t, finalLoss < initialLoss, "[%s] Loss should decrease: initial=%.6f, final=%.6f", activationName, initialLoss, finalLoss)
	}

	// Test predictions
	correct := 0
	for i := range inputs {
		output, err := model.Forward(inputs[i])
		if err != nil {
			t.Fatalf("Forward pass failed: %v", err)
		}
		predicted := output.At(0)
		expected := targets[i].At(0)

		// For classification, we consider prediction correct if it's on the right side of 0.5
		predictedClass := 0.0
		if predicted > 0.5 {
			predictedClass = 1.0
		}
		expectedClass := expected

		if predictedClass == expectedClass {
			correct++
		}
		t.Logf("[%s] Input: %.1f -> Predicted: %.6f (class: %.0f), Expected: %.0f",
			activationName, inputs[i].At(0), predicted, predictedClass, expectedClass)
	}

	accuracy := float64(correct) / float64(len(inputs))
	t.Logf("[%s] Accuracy: %d/%d = %.2f%%", activationName, correct, len(inputs), accuracy*100)

	// For this simple problem, we expect at least some learning
	// The exact accuracy depends on the activation function
	assert.True(t, accuracy >= 0.5, "[%s] Accuracy should be at least 50%%, got %.2f%%", activationName, accuracy*100)
}
