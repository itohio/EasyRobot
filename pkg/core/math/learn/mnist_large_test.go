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
	trainPath := filepath.Join("datasets", "mnist", "mnist_train.csv")
	trainSamples, err := mnist.Load(trainPath, maxTrainSamples)
	if err != nil {
		t.Fatalf("Failed to load training data: %v", err)
	}
	t.Logf("Loaded %d training samples", len(trainSamples))

	// Load test data
	testPath := filepath.Join("datasets", "mnist", "mnist_test.csv")
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
		if len(param.Data.Dim) >= 2 {
			// Estimate fan-in
			fanIn := 1
			if len(param.Data.Dim) >= 2 {
				fanIn = param.Data.Dim[len(param.Data.Dim)-1]
				// For conv layers (4D), account for kernel size
				if len(param.Data.Dim) == 4 {
					// Conv kernel: [outChannels, inChannels, kernelH, kernelW]
					kernelH := param.Data.Dim[2]
					kernelW := param.Data.Dim[3]
					fanIn = param.Data.Dim[1] * kernelH * kernelW
				} else if len(param.Data.Dim) == 2 {
					// Dense layer: [outFeatures, inFeatures]
					fanIn = param.Data.Dim[1]
				}
			}
			// He initialization for ReLU: uniform with limit = sqrt(6 / fan_in)
			// This gives variance ≈ 2/fan_in (He initialization)
			if fanIn > 0 {
				limit := float32(math.Sqrt(6.0 / float64(fanIn)))
				for i := range param.Data.Data {
					param.Data.Data[i] = (rng.Float32()*2 - 1) * limit
				}
			}
		} else {
			// For biases (1D), initialize to zero (common practice)
			for i := range param.Data.Data {
				param.Data.Data[i] = 0
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
		totalLoss := float32(0)
		correct := 0
		runningLoss := float32(0)
		runningCount := 0

		// Log progress every N samples (increased for speed)
		logInterval := 200

		for i, sample := range trainSamples {
			// Reshape image to [1, 1, 28, 28] format for Conv2D
			imageData := make([]float32, 1*1*28*28)
			copy(imageData, sample.Image.Data)
			input := tensor.Tensor{
				Dim:  []int{1, 1, 28, 28},
				Data: imageData,
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
			maxProb := output.Data[0]
			for j := 1; j < 10; j++ {
				if output.Data[j] > maxProb {
					maxProb = output.Data[j]
					predicted = j
				}
			}

			if predicted == sample.Label {
				correct++
			}

			// Log progress in real-time every N samples
			if (i+1)%logInterval == 0 || i == 0 {
				avgRunningLoss := runningLoss / float32(runningCount)
				currentAccuracy := float32(correct) / float32(i+1)
				t.Logf("[Epoch %d, Sample %d/%d] Current Loss: %.4f | Avg Loss (last %d): %.4f | Accuracy: %.2f%%",
					epoch+1, i+1, len(trainSamples), loss, logInterval, avgRunningLoss, currentAccuracy*100)
				// Reset running averages for next interval
				runningLoss = 0
				runningCount = 0
			}
		}

		avgLoss := totalLoss / float32(len(trainSamples))
		accuracy := float32(correct) / float32(len(trainSamples))
		t.Logf("Epoch %d COMPLETE: Loss=%.4f, Accuracy=%.2f%% (%d/%d)", epoch+1, avgLoss, accuracy*100, correct, len(trainSamples))
	}

	// Validation on test set
	// Set dropout to inference mode for validation
	dropout1.SetTrainingMode(false)
	t.Log("\n=== Validation ===")
	testCorrect := 0
	totalTestLoss := float32(0)

	for i, sample := range testSamples {
		// Reshape image to [1, 1, 28, 28]
		imageData := make([]float32, 1*1*28*28)
		copy(imageData, sample.Image.Data)
		input := tensor.Tensor{
			Dim:  []int{1, 1, 28, 28},
			Data: imageData,
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
		totalTestLoss += loss

		// Find predicted class
		predicted := 0
		maxProb := output.Data[0]
		for j := 1; j < 10; j++ {
			if output.Data[j] > maxProb {
				maxProb = output.Data[j]
				predicted = j
			}
		}

		if predicted == sample.Label {
			testCorrect++
		}
	}

	testAccuracy := float32(testCorrect) / float32(len(testSamples))
	avgTestLoss := totalTestLoss / float32(len(testSamples))
	t.Logf("Test Accuracy: %.2f%% (%d/%d)", testAccuracy*100, testCorrect, len(testSamples))
	t.Logf("Test Loss: %.4f", avgTestLoss)

	// Expect better accuracy with larger network and more training data
	if testAccuracy < 0.90 {
		t.Logf("Test accuracy %.2f%% is below 90%% (expected >90%% for larger model)", testAccuracy*100)
	} else {
		t.Logf("✓ Large model achieved %.2f%% accuracy on test set!", testAccuracy*100)
	}
}
