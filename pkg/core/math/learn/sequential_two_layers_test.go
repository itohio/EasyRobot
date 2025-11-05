package learn

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
)

// TestSequential_TwoLayers_SimpleLinearRegression tests that a Sequential model
// with two Dense layers can learn a simple linear function: y = 2*x + 1 (with bias)
// Since composition of linear functions is still linear, this should work.
// This verifies that Sequential model correctly handles multiple layers during training
func TestSequential_TwoLayers_SimpleLinearRegression(t *testing.T) {
	// Create two dense layers: 1 -> 2 -> 1
	// This should be able to learn y = 2*x + 1 (composition of linear functions is linear)
	dense1, err := layers.NewDense(1, 2, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create first dense layer: %v", err)
	}

	dense2, err := layers.NewDense(2, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create second dense layer: %v", err)
	}

	// Build a Sequential model with both layers
	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(1)).
		AddLayer(dense1).
		AddLayer(dense2).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize the model
	if err := model.Init(tensor.NewShape(1)); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Set initial weights and biases to small random values
	// We'll start with weights close to 0 and bias close to 0
	// Get the layers from the model
	layer1 := model.GetLayer(0).(*layers.Dense)
	layer2 := model.GetLayer(1).(*layers.Dense)

	// Initialize layer1 weights (1 -> 2): [2x1] weight matrix
	weight1 := layer1.Base.Weights()
	weight1.Data.SetAt(0.1, 0, 0) // weight[0][0] = 0.1
	weight1.Data.SetAt(0.1, 0, 1) // weight[0][1] = 0.1
	layer1.Base.SetParam(types.ParamWeights, weight1)

	bias1 := layer1.Base.Biases()
	bias1.Data.SetAt(0.1, 0) // bias[0] = 0.1
	bias1.Data.SetAt(0.1, 1) // bias[1] = 0.1
	layer1.Base.SetParam(types.ParamBiases, bias1)

	// Initialize layer2 weights (2 -> 1): [2x1] weight matrix
	weight2 := layer2.Base.Weights()
	weight2.Data.SetAt(0.1, 0, 0) // weight[0][0] = 0.1
	weight2.Data.SetAt(0.1, 1, 0) // weight[1][0] = 0.1
	layer2.Base.SetParam(types.ParamWeights, weight2)

	bias2 := layer2.Base.Biases()
	bias2.Data.SetAt(0.1, 0) // bias[0] = 0.1
	layer2.Base.SetParam(types.ParamBiases, bias2)

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

	// Get initial parameter values
	initialWeight1_00 := layer1.Base.Weights().Data.At(0, 0)
	initialBias1_0 := layer1.Base.Biases().Data.At(0)
	initialWeight2_00 := layer2.Base.Weights().Data.At(0, 0)
	initialBias2_0 := layer2.Base.Biases().Data.At(0)
	t.Logf("Initial layer1: weight[0][0]=%.6f, bias[0]=%.6f", initialWeight1_00, initialBias1_0)
	t.Logf("Initial layer2: weight[0][0]=%.6f, bias[0]=%.6f", initialWeight2_00, initialBias2_0)

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
		if (epoch+1)%40 == 0 {
			t.Logf("Epoch %d: Average loss = %.6f", epoch+1, avgLoss)
		}

		// Check if converged
		if avgLoss < 0.001 {
			t.Logf("Converged at epoch %d with loss %.6f", epoch+1, avgLoss)
			break
		}
	}

	// Get final parameter values
	finalWeight1_00 := layer1.Base.Weights().Data.At(0, 0)
	finalBias1_0 := layer1.Base.Biases().Data.At(0)
	finalWeight2_00 := layer2.Base.Weights().Data.At(0, 0)
	finalBias2_0 := layer2.Base.Biases().Data.At(0)
	t.Logf("Final layer1: weight[0][0]=%.6f, bias[0]=%.6f", finalWeight1_00, finalBias1_0)
	t.Logf("Final layer2: weight[0][0]=%.6f, bias[0]=%.6f", finalWeight2_00, finalBias2_0)

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
	assert.True(t, finalLoss < initialLoss, "Loss should decrease during training: initial=%.6f, final=%.6f", initialLoss, finalLoss)

	// Verify that parameters changed
	assert.NotEqual(t, initialWeight1_00, finalWeight1_00, "Layer1 weight should change during training")
	assert.NotEqual(t, initialBias1_0, finalBias1_0, "Layer1 bias should change during training")
	assert.NotEqual(t, initialWeight2_00, finalWeight2_00, "Layer2 weight should change during training")
	assert.NotEqual(t, initialBias2_0, finalBias2_0, "Layer2 bias should change during training")

	// Test predictions on training data
	// The model should learn to approximate y = 2*x + 1
	for i := range inputs {
		output, err := model.Forward(inputs[i])
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
