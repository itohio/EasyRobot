package learn_test

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/learn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// TestConv1D_Layer_Learning tests if a single Conv1D layer can learn a simple pattern.
// Task: Detect if a sequence contains a spike (high value) at position 3.
// Input: Sequences of length 5, with values mostly low (0.1) but one sequence has a spike (1.0) at position 3.
// Output: Conv1D output is flattened and compared to target.
func TestConv1D_Layer_Learning(t *testing.T) {
	// Create training data
	// Input shape: [batch=1, channels=1, length=5]
	trainingInputs := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 0.1, 1.0, 0.1}), // Spike at pos 3 -> target high
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 0.1, 0.1, 0.1}), // No spike -> target low
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 1.0, 0.1, 0.1, 0.1}), // Spike at pos 1 -> target low
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 1.0, 0.1, 0.1}), // Spike at pos 2 -> target low
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 0.1, 1.0, 0.1}), // Spike at pos 3 -> target high
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 0.1, 0.1, 0.1}), // No spike -> target low
	}

	trainingTargets := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}), // Spike at pos 3
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // No spike
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // Wrong position
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // Wrong position
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}), // Spike at pos 3
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // No spike
	}

	// Test data
	testInputs := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 0.1, 1.0, 0.1}), // Spike at pos 3 -> target 1
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 0.1, 0.1, 0.1}), // No spike -> target 0
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 0.1, 1.0, 0.1}), // Spike at pos 3 -> target 1
		tensor.FromFloat32(tensor.NewShape(1, 1, 5), []float32{0.1, 0.1, 0.1, 0.1, 0.1}), // No spike -> target 0
	}

	testTargets := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
	}

	// Create Conv1D layer: 1 input channel, 4 output channels, kernel length 3
	conv1d, err := layers.NewConv1D(1, 4, 3, 1, 0, layers.WithCanLearn(true), layers.UseBias(true))
	if err != nil {
		t.Fatalf("Failed to create Conv1D layer: %v", err)
	}

	// Initialize layer
	if err := conv1d.Init(tensor.NewShape(1, 1, 5)); err != nil {
		t.Fatalf("Failed to initialize Conv1D layer: %v", err)
	}

	// Create loss and optimizer
	lossFn := nn.NewMSE()
	optimizer := learn.NewAdam(0.1, 0.9, 0.999, 1e-8) // Higher learning rate

	// Train for multiple epochs
	epochs := 1000
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float64(0)

		for i := range trainingInputs {
			// Forward pass
			output, err := conv1d.Forward(trainingInputs[i])
			if err != nil {
				t.Fatalf("Forward failed at epoch %d, sample %d: %v", epoch, i, err)
			}

			// Mean output to get scalar: output shape [1, 4, 3] -> mean all elements -> scalar
			// This normalizes the output to a reasonable range
			outputMean := output.Mean() // Returns [1] tensor with mean

			// Scale target to match expected output range (since we're using mean)
			// Mean of [1, 4, 3] = 12 elements, so scale target by 12 to match sum
			scaledTarget := tensor.FromFloat32(tensor.NewShape(1), []float32{float32(trainingTargets[i].At(0)) * 12.0})

			// Compute loss
			loss, err := lossFn.Compute(outputMean, scaledTarget)
			if err != nil {
				t.Fatalf("Loss computation failed: %v", err)
			}

			// Get gradient from loss function
			lossGrad, err := lossFn.Gradient(outputMean, scaledTarget)
			if err != nil {
				t.Fatalf("Gradient computation failed: %v", err)
			}

			// Propagate gradient through mean: mean gradient is broadcast to all output elements
			// For mean operation: if outputMean = mean(output), then dLoss/dOutput = dLoss/dOutputMean / size (broadcast)
			outputShape := output.Shape()
			outputSize := float32(outputShape.Size())
			gradData := make([]float32, outputShape.Size())
			gradVal := float32(lossGrad.At(0)) / outputSize // Divide by size because mean = sum/size
			for j := range gradData {
				gradData[j] = gradVal
			}
			lossGradBroadcast := tensor.FromFloat32(outputShape, gradData)

			_, err = conv1d.Backward(lossGradBroadcast)
			if err != nil {
				t.Fatalf("Backward failed at epoch %d, sample %d: %v", epoch, i, err)
			}

			// Update parameters
			if err := conv1d.Update(optimizer); err != nil {
				t.Fatalf("Update failed at epoch %d, sample %d: %v", epoch, i, err)
			}

			// Zero gradients
			conv1d.ZeroGrad()

			totalLoss += float64(loss)
		}

		avgLoss := totalLoss / float64(len(trainingInputs))
		if avgLoss < 0.01 {
			break
		}
	}

	// Test the trained model
	correctCount := 0
	totalCount := len(testInputs)

	for i, input := range testInputs {
		output, err := conv1d.Forward(input)
		if err != nil {
			t.Fatalf("Forward failed for test input %d: %v", i, err)
		}

		// Mean output to get scalar
		outputMean := output.Mean()
		predicted := float32(outputMean.At(0))

		// Scale target to match (since we're using mean of 12 elements)
		expected := float32(testTargets[i].At(0)) * 12.0
		error := absFloat32(predicted - expected)

		// Consider correct if error < 1.0 (since we scaled by 12)
		if error <= 1.0 {
			correctCount++
		}
	}

	// Calculate accuracy
	accuracy := float64(correctCount) / float64(totalCount) * 100.0
	t.Logf("Test accuracy: %.2f%% (%d/%d correct)", accuracy, correctCount, totalCount)

	// Log predictions
	for i, input := range testInputs {
		output, err := conv1d.Forward(input)
		if err == nil {
			outputMean := output.Mean()
			predicted := float32(outputMean.At(0))
			expected := float32(testTargets[i].At(0)) * 12.0
			error := absFloat32(predicted - expected)
			t.Logf("Test %d: Expected: %.1f, Predicted: %.4f, Error: %.4f", i, expected, predicted, error)
		}
	}

	// Require at least 75% accuracy
	if accuracy < 75.0 {
		t.Errorf("Model did not learn well enough. Accuracy: %.2f%% < 75%%", accuracy)
	}
}

// TestDenseConv1D_CombinedLearning tests if Dense + Conv1D layers can learn together using Sequential model.
// Task: Detect if a sequence contains a spike (high value) at position 3.
// Architecture: Input [5] -> Dense -> Reshape -> Conv1D -> Mean -> scalar
func TestDenseConv1D_CombinedLearning(t *testing.T) {
	// Create training data - flatten input for Dense layer
	// Input shape for Dense: [5] (flattened)
	trainingInputs := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 0.1, 1.0, 0.1}), // Spike at pos 3 -> target high
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 0.1, 0.1, 0.1}), // No spike -> target low
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 1.0, 0.1, 0.1, 0.1}), // Spike at pos 1 -> target low
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 1.0, 0.1, 0.1}), // Spike at pos 2 -> target low
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 0.1, 1.0, 0.1}), // Spike at pos 3 -> target high
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 0.1, 0.1, 0.1}), // No spike -> target low
	}

	trainingTargets := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}), // Spike at pos 3
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // No spike
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // Wrong position
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // Wrong position
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}), // Spike at pos 3
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}), // No spike
	}

	// Test data
	testInputs := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 0.1, 1.0, 0.1}), // Spike at pos 3 -> target 1
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 0.1, 0.1, 0.1}), // No spike -> target 0
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 0.1, 1.0, 0.1}), // Spike at pos 3 -> target 1
		tensor.FromFloat32(tensor.NewShape(5), []float32{0.1, 0.1, 0.1, 0.1, 0.1}), // No spike -> target 0
	}

	testTargets := []tensor.Tensor{
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{1.0}),
		tensor.FromFloat32(tensor.NewShape(1), []float32{0.0}),
	}

	// Build Sequential model: Dense -> Reshape -> Conv1D
	dense, err := layers.NewDense(5, 8, layers.WithCanLearn(true), layers.UseBias(true))
	if err != nil {
		t.Fatalf("Failed to create Dense layer: %v", err)
	}

	reshape := layers.NewReshape([]int{1, 1, 8})

	conv1d, err := layers.NewConv1D(1, 4, 3, 1, 0, layers.WithCanLearn(true), layers.UseBias(true))
	if err != nil {
		t.Fatalf("Failed to create Conv1D layer: %v", err)
	}

	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(5)).
		AddLayer(dense).
		AddLayer(reshape).
		AddLayer(conv1d).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize model
	if err := model.Init(tensor.NewShape(5)); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Create loss and optimizer
	lossFn := nn.NewMSE()
	optimizer := learn.NewAdam(0.1, 0.9, 0.999, 1e-8)

	// Train for multiple epochs
	epochs := 1000
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float64(0)

		for i := range trainingInputs {
			// Forward pass through model
			output, err := model.Forward(trainingInputs[i])
			if err != nil {
				t.Fatalf("Forward failed at epoch %d, sample %d: %v", epoch, i, err)
			}

			// Mean output to get scalar
			outputMean := output.Mean()
			scaledTarget := tensor.FromFloat32(tensor.NewShape(1), []float32{float32(trainingTargets[i].At(0)) * 12.0})

			// Compute loss
			loss, err := lossFn.Compute(outputMean, scaledTarget)
			if err != nil {
				t.Fatalf("Loss computation failed: %v", err)
			}

			// Get gradient from loss function
			lossGrad, err := lossFn.Gradient(outputMean, scaledTarget)
			if err != nil {
				t.Fatalf("Gradient computation failed: %v", err)
			}

			// Backward pass through model
			outputShape := output.Shape()
			outputSize := float32(outputShape.Size())
			gradData := make([]float32, outputShape.Size())
			gradVal := float32(lossGrad.At(0)) / outputSize
			for j := range gradData {
				gradData[j] = gradVal
			}
			lossGradBroadcast := tensor.FromFloat32(outputShape, gradData)

			if _, err := model.Backward(lossGradBroadcast); err != nil {
				t.Fatalf("Backward failed at epoch %d, sample %d: %v", epoch, i, err)
			}

			// Update parameters
			if err := model.Update(optimizer); err != nil {
				t.Fatalf("Update failed at epoch %d, sample %d: %v", epoch, i, err)
			}

			// Zero gradients
			model.ZeroGrad()

			totalLoss += float64(loss)
		}

		avgLoss := totalLoss / float64(len(trainingInputs))
		if avgLoss < 0.01 {
			break
		}
	}

	// Test the trained model
	correctCount := 0
	totalCount := len(testInputs)

	for i, input := range testInputs {
		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("Forward failed for test input %d: %v", i, err)
		}

		outputMean := output.Mean()
		predicted := float32(outputMean.At(0))
		expected := float32(testTargets[i].At(0)) * 12.0
		error := absFloat32(predicted - expected)

		if error <= 1.0 {
			correctCount++
		}
	}

	accuracy := float64(correctCount) / float64(totalCount) * 100.0
	t.Logf("Test accuracy: %.2f%% (%d/%d correct)", accuracy, correctCount, totalCount)

	if accuracy < 75.0 {
		t.Errorf("Model did not learn well enough. Accuracy: %.2f%% < 75%%", accuracy)
	}
}

// absFloat32 returns absolute value of float32
func absFloat32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
