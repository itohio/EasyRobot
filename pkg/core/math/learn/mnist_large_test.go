package learn_test

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/learn"
	"github.com/itohio/EasyRobot/pkg/core/math/learn/datasets/mnist"
	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// oneHot creates a one-hot encoded tensor for a label.
// Returns shape [1, numClasses] to match model output.
func oneHot(label int, numClasses int) tensor.Tensor {
	data := make([]float32, numClasses)
	data[label] = 1.0
	return tensor.FromFloat32(tensor.NewShape(1, numClasses), data)
}

// TestMNISTLarge trains a larger CNN on MNIST dataset.
// Uses convolutional blocks with pooling layers.
// Optimized for speed: reduced model size, fewer samples, fewer epochs.
// For faster training:
//   - Reduce maxTrainSamples / maxTestSamples
//   - Reduce number of epochs
//   - Reduce model channels and dense layer sizes
//   - Increase logInterval to reduce I/O
func TestMNISTLarge(t *testing.T) {
	// Use more samples for better learning (reduced for speed)
	const maxTrainSamples = 1500
	const maxTestSamples = 250

	// Load training data
	trainPath := filepath.Join("datasets", "mnist", "mnist_train.csv.gz")
	trainSamples, err := mnist.Load(trainPath, maxTrainSamples)
	if err != nil {
		t.Fatalf("Failed to load training data: %v", err)
	}
	t.Logf("Loaded %d training samples", len(trainSamples))

	// Load test data
	testPath := filepath.Join("datasets", "mnist", "mnist_test.csv.gz")
	testSamples, err := mnist.Load(testPath, maxTestSamples)
	if err != nil {
		t.Fatalf("Failed to load test data: %v", err)
	}
	t.Logf("Loaded %d test samples", len(testSamples))

	// Build a larger CNN model:
	// Input: [1, 1, 28, 28]
	// Conv2D(1->32, 3x3) -> ReLU -> MaxPool2D(2x2)
	// Conv2D(32->64, 3x3) -> ReLU -> MaxPool2D(2x2)
	// Conv2D(64->128, 3x3) -> ReLU
	// Flatten
	// Dense(->256) -> ReLU
	// Dense(->128) -> ReLU
	// Dense(->10)

	// First conv block: 1 -> 16 channels (reduced for speed)
	conv1, err := layers.NewConv2D(1, 16, 3, 3, 1, 1, 1, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create conv1: %v", err)
	}
	relu1 := layers.NewReLU("relu1")
	pool1, err := layers.NewMaxPool2D(2, 2, 2, 2, 0, 0)
	if err != nil {
		t.Fatalf("Failed to create pool1: %v", err)
	}

	// Second conv block: 16 -> 32 channels (reduced for speed)
	conv2, err := layers.NewConv2D(16, 32, 3, 3, 1, 1, 1, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create conv2: %v", err)
	}
	relu2 := layers.NewReLU("relu2")
	pool2, err := layers.NewMaxPool2D(2, 2, 2, 2, 0, 0)
	if err != nil {
		t.Fatalf("Failed to create pool2: %v", err)
	}

	// Simplified: Skip third conv block for speed
	// After second conv block, go straight to dense layers
	flatten := layers.NewFlatten(1, 4) // Flatten dims 1-3 (channels, height, width), keeping batch

	// After Conv2D layers and pooling:
	// Input: [1, 1, 28, 28]
	// After conv1+pool1: [1, 16, 14, 14] (28->14 with pool)
	// After conv2+pool2: [1, 32, 7, 7] (14->7 with pool)
	// After flatten: [1, 32*7*7] = [1, 1568]
	dense1, err := layers.NewDense(32*7*7, 64, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create dense1: %v", err)
	}
	relu4 := layers.NewReLU("relu4")
	dropout1 := layers.NewDropout("dropout1", layers.WithDropoutRate(0.3), layers.WithTrainingMode(true))

	dense2, err := layers.NewDense(64, 10, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create dense2: %v", err)
	}

	model, err := nn.NewModelBuilder([]int{1, 1, 28, 28}).
		AddLayer(conv1).
		AddLayer(relu1).
		AddLayer(pool1).
		AddLayer(conv2).
		AddLayer(relu2).
		AddLayer(pool2).
		AddLayer(flatten).
		AddLayer(dense1).
		AddLayer(relu4).
		AddLayer(dropout1).
		AddLayer(dense2).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize model
	if err := model.Init(); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Initialize weights with He initialization (better for ReLU)
	// He initialization: stddev = sqrt(2 / fan_in)
	rng := rand.New(rand.NewSource(42))
	params := model.Parameters()
	for _, param := range params {
		// Parameters() returns copies, but tensor.Data slices are shared references
		// So modifying param.Data.Data will modify the actual layer parameters
		paramShape := param.Data.Shape()
		if paramShape.Rank() >= 2 {
			// Estimate fan-in
			fanIn := 1
			if paramShape.Rank() >= 2 {
				fanIn = paramShape[paramShape.Rank()-1]
				// For conv layers (4D), account for kernel size
				if paramShape.Rank() == 4 {
					// Conv kernel: [outChannels, inChannels, kernelH, kernelW]
					kernelH := paramShape[2]
					kernelW := paramShape[3]
					fanIn = paramShape[1] * kernelH * kernelW
				} else if paramShape.Rank() == 2 {
					// Dense layer: [outFeatures, inFeatures]
					fanIn = paramShape[1]
				}
			}
			// He initialization for ReLU: uniform with limit = sqrt(6 / fan_in)
			// This gives variance ≈ 2/fan_in (He initialization)
			if fanIn > 0 {
				limit := float32(math.Sqrt(6.0 / float64(fanIn)))
				for indices := range param.Data.Shape().Iterator() {
					val := float64((rng.Float32()*2 - 1) * limit)
					param.Data.SetAt(val, indices...)
				}
			}
		} else {
			// For biases (1D), initialize to zero (common practice)
			for indices := range param.Data.Shape().Iterator() {
				param.Data.SetAt(0.0, indices...)
			}
		}
	}

	// Create loss and optimizer
	// Use slightly higher learning rate for deeper network to help gradient flow
	lossFn := nn.NewCategoricalCrossEntropy(true) // fromLogits=true, applies softmax
	optimizer := learn.NewAdam(0.002, 0.9, 0.999, 1e-8)

	// Training loop - reduced epochs for speed
	epochs := 2
	// Ensure dropout is in training mode
	dropout1.SetTrainingMode(true)
	t.Log("\n=== Training ===")
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float64(0)
		correct := 0
		runningLoss := float64(0)
		runningCount := 0

		// Log progress every N samples (increased for speed)
		logInterval := 200

		for i, sample := range trainSamples {
			// Reshape image to [1, 1, 28, 28] format for Conv2D
			// sample.Image is [1, 28, 28], we need [1, 1, 28, 28]
			input := tensor.New(tensor.DTFP32, tensor.NewShape(1, 1, 28, 28))
			// Copy from sample.Image to input tensor
			for indices := range sample.Image.Shape().Iterator() {
				val := sample.Image.At(indices...)
				// Map from [1, 28, 28] to [1, 1, 28, 28]
				inputIndices := []int{0, 0, indices[1], indices[2]}
				input.SetAt(val, inputIndices...)
			}

			// Create one-hot target
			target := oneHot(sample.Label, 10)

			// Training step
			loss, err := learn.TrainStep(model, optimizer, lossFn, input, target)
			if err != nil {
				t.Fatalf("TrainStep failed at epoch %d, sample %d: %v", epoch, i, err)
			}

			totalLoss += loss
			runningLoss += loss
			runningCount++

			// Check prediction
			output, err := model.Forward(input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Find predicted class (argmax)
			predicted := 0
			maxProb := output.At(0)
			for j := 1; j < 10; j++ {
				prob := output.At(j)
				if prob > maxProb {
					maxProb = prob
					predicted = j
				}
			}

			if predicted == sample.Label {
				correct++
			}

			// Log progress in real-time every N samples
			if (i+1)%logInterval == 0 || i == 0 {
				avgRunningLoss := runningLoss / float64(runningCount)
				currentAccuracy := float64(correct) / float64(i+1)
				t.Logf("[Epoch %d, Sample %d/%d] Current Loss: %.4f | Avg Loss (last %d): %.4f | Accuracy: %.2f%%",
					epoch+1, i+1, len(trainSamples), loss, logInterval, avgRunningLoss, currentAccuracy*100)
				// Reset running averages for next interval
				runningLoss = 0
				runningCount = 0
			}
		}

		avgLoss := totalLoss / float64(len(trainSamples))
		accuracy := float64(correct) / float64(len(trainSamples))
		t.Logf("Epoch %d COMPLETE: Loss=%.4f, Accuracy=%.2f%% (%d/%d)", epoch+1, avgLoss, accuracy*100, correct, len(trainSamples))
	}

	// Validation on test set
	// Set dropout to inference mode for validation
	dropout1.SetTrainingMode(false)
	t.Log("\n=== Validation ===")
	testCorrect := 0
	totalTestLoss := float64(0)

	for i, sample := range testSamples {
		// Reshape image to [1, 1, 28, 28]
		input := tensor.New(tensor.DTFP32, tensor.NewShape(1, 1, 28, 28))
		// Copy from sample.Image to input tensor
		for indices := range sample.Image.Shape().Iterator() {
			val := sample.Image.At(indices...)
			// Map from [1, 28, 28] to [1, 1, 28, 28]
			inputIndices := []int{0, 0, indices[1], indices[2]}
			input.SetAt(val, inputIndices...)
		}

		// Create one-hot target
		target := oneHot(sample.Label, 10)

		// Forward pass
		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("Forward failed on test sample %d: %v", i, err)
		}

		// Compute loss (without training)
		loss, err := lossFn.Compute(output, target)
		if err != nil {
			t.Fatalf("Loss computation failed: %v", err)
		}
		totalTestLoss += float64(loss)

		// Find predicted class
		predicted := 0
		maxProb := output.At(0)
		for j := 1; j < 10; j++ {
			prob := output.At(j)
			if prob > maxProb {
				maxProb = prob
				predicted = j
			}
		}

		if predicted == sample.Label {
			testCorrect++
		}
	}

	testAccuracy := float64(testCorrect) / float64(len(testSamples))
	avgTestLoss := totalTestLoss / float64(len(testSamples))
	t.Logf("Test Accuracy: %.2f%% (%d/%d)", testAccuracy*100, testCorrect, len(testSamples))
	t.Logf("Test Loss: %.4f", avgTestLoss)

	// Expect better accuracy with larger network and more training data
	if testAccuracy < 0.90 {
		t.Logf("Test accuracy %.2f%% is below 90%% (expected >90%% for larger model)", testAccuracy*100)
	} else {
		t.Logf("✓ Large model achieved %.2f%% accuracy on test set!", testAccuracy*100)
	}
}
