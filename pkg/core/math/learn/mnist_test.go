package learn_test

import (
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
	oneHot := tensor.Tensor{
		Dim:  []int{1, numClasses},
		Data: make([]float32, numClasses),
	}
	oneHot.Data[label] = 1.0
	return oneHot
}

// TestMNIST trains a CNN on MNIST dataset for digit classification.
func TestMNIST(t *testing.T) {
	// Limit samples for speed (focus on digits 0-9 as requested)
	const maxTrainSamples = 1000
	const maxTestSamples = 200

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

	// Build CNN model:
	// Input: [1, 1, 28, 28] (batch=1, channels=1, height=28, width=28)
	// Using stride 2 in conv layers for simplicity and speed
	// Conv2D(1->16, 3x3, stride=2) -> ReLU
	// Conv2D(16->32, 3x3, stride=2) -> ReLU
	// Flatten -> Dense(->128) -> ReLU -> Dense(->10)

	conv1, err := layers.NewConv2D(1, 16, 3, 3, 2, 2, 1, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create conv1: %v", err)
	}
	relu1 := layers.NewReLU("relu1")

	conv2, err := layers.NewConv2D(16, 32, 3, 3, 2, 2, 1, 1, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create conv2: %v", err)
	}
	relu2 := layers.NewReLU("relu2")

	flatten := layers.NewFlatten(1, 4) // Flatten dims 1-3 (channels, height, width), keeping batch

	// After Conv2D layers with stride 2:
	// Input: [1, 1, 28, 28]
	// After conv1 (stride=2): [1, 16, 14, 14] (28/2 = 14)
	// After conv2 (stride=2): [1, 32, 7, 7] (14/2 = 7)
	// After flatten: [1, 32*7*7] = [1, 1568]
	dense1, err := layers.NewDense(32*7*7, 128, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create dense1: %v", err)
	}
	relu3 := layers.NewReLU("relu3")

	dense2, err := layers.NewDense(128, 10, layers.WithCanLearn(true))
	if err != nil {
		t.Fatalf("Failed to create dense2: %v", err)
	}

	// Note: CategoricalCrossEntropy with fromLogits=true will apply softmax internally
	// So we don't need an explicit softmax layer

	model, err := nn.NewModelBuilder([]int{1, 1, 28, 28}).
		AddLayer(conv1).
		AddLayer(relu1).
		AddLayer(conv2).
		AddLayer(relu2).
		AddLayer(flatten).
		AddLayer(dense1).
		AddLayer(relu3).
		AddLayer(dense2).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize model
	if err := model.Init(); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Initialize weights with Xavier initialization
	rng := rand.New(rand.NewSource(42))
	params := model.Parameters()
	for _, param := range params {
		if len(param.Data.Dim) >= 2 {
			// Estimate fan-in and fan-out
			fanIn := 1
			fanOut := 1
			if len(param.Data.Dim) >= 2 {
				fanIn = param.Data.Dim[len(param.Data.Dim)-1]
				fanOut = param.Data.Dim[0]
				// For higher dimensions, multiply intermediate dims
				for i := 1; i < len(param.Data.Dim)-1; i++ {
					fanOut *= param.Data.Dim[i]
				}
			}
			// Xavier uniform initialization
			limit := float32(1.0 / float64(fanIn+fanOut))
			for i := range param.Data.Data {
				param.Data.Data[i] = (rng.Float32()*2 - 1) * limit
			}
		} else {
			// For biases or 1D tensors, use small random values
			for i := range param.Data.Data {
				param.Data.Data[i] = (rng.Float32()*2 - 1) * 0.1
			}
		}
	}

	// Create loss and optimizer
	lossFn := nn.NewCategoricalCrossEntropy(true) // fromLogits=true, applies softmax
	optimizer := learn.NewAdam(0.001, 0.9, 0.999, 1e-8)

	// Training loop
	epochs := 5
	t.Log("\n=== Training ===")
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0)
		correct := 0

		for i, sample := range trainSamples {
			// Reshape image to [1, 1, 28, 28] format for Conv2D
			// sample.Image is [1, 28, 28], we need [1, 1, 28, 28]
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
		}

		avgLoss := totalLoss / float32(len(trainSamples))
		accuracy := float32(correct) / float32(len(trainSamples))
		t.Logf("Epoch %d: Loss=%.4f, Accuracy=%.2f%% (%d/%d)", epoch+1, avgLoss, accuracy*100, correct, len(trainSamples))
	}

	// Validation on test set
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

	// Basic sanity check: accuracy should be better than random (10% for 10 classes)
	if testAccuracy < 0.15 {
		t.Errorf("Test accuracy %.2f%% is too low (expected >15%% for a trained model)", testAccuracy*100)
	} else {
		t.Logf("✓ Model achieved %.2f%% accuracy on test set", testAccuracy*100)
	}
}
