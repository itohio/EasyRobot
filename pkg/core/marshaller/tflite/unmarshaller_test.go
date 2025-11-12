//go:build tflite
// +build tflite

package tflite

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/learn/datasets/mnist"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

func isTFLiteUnavailable(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	for _, fragment := range []string{"libtensorflowlite", "dlopen", "failed to create interpreter"} {
		if strings.Contains(msg, fragment) {
			return true
		}
	}
	return false
}

func loadTestModel(t *testing.T) *Model {
	t.Helper()

	modelPath := filepath.Join("..", "testdata", "mnist_model.tflite")
	loader := NewLoader()

	model, err := loader.LoadFromFile(modelPath)
	if err != nil {
		if isTFLiteUnavailable(err) {
			t.Skipf("skipping TFLite tests: %v", err)
		}
		t.Fatalf("failed to load TFLite model: %v", err)
	}

	t.Cleanup(func() {
		model.Close()
	})

	return model
}

func TestLoadMNISTModel(t *testing.T) {
	model := loadTestModel(t)

	inputShape := model.InputShape()
	if inputShape.Rank() == 0 {
		t.Fatalf("unexpected empty input shape")
	}

	if err := model.Init(inputShape.Clone()); err != nil {
		t.Fatalf("Init failed: %v", err)
	}

	input := tensor.New(tensor.DTFP32, inputShape.Clone())
	output, err := model.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	expectedOutputShape := model.OutputShapeValue()
	if !output.Shape().Equal(expectedOutputShape) {
		t.Fatalf("unexpected output shape. want %v, got %v", expectedOutputShape, output.Shape())
	}
}

func TestMNISTInference(t *testing.T) {
	model := loadTestModel(t)

	if err := model.Init(model.InputShape().Clone()); err != nil {
		t.Fatalf("Init failed: %v", err)
	}

	dataPath := filepath.Join("..", "..", "math", "learn", "datasets", "mnist", "mnist_test.csv.gz")
	samples, err := mnist.Load(dataPath, 50)
	if err != nil {
		t.Skipf("skipping inference test: %v", err)
	}
	if len(samples) == 0 {
		t.Skip("no MNIST samples available")
	}

	inputShape := model.InputShape()
	sampleSize := samples[0].Image.Shape().Size()
	if inputShape.Size() != sampleSize {
		t.Skipf("model input size %d does not match sample size %d", inputShape.Size(), sampleSize)
	}

	correct := 0
	total := len(samples)

	for idx, sample := range samples {
		input := tensor.New(tensor.DTFP32, inputShape.Clone())
		inputData := input.Data().([]float32)
		sampleData := sample.Image.Data().([]float32)
		if len(sampleData) != len(inputData) {
			t.Fatalf("sample %d: data length %d does not match input length %d", idx, len(sampleData), len(inputData))
		}
		copy(inputData, sampleData)

		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("forward pass failed for sample %d: %v", idx, err)
		}

		predicted := 0
		maxVal := output.At(0)
		for i := 1; i < output.Size(); i++ {
			if val := output.At(i); val > maxVal {
				maxVal = val
				predicted = i
			}
		}
		if predicted == sample.Label {
			correct++
		}
	}

	accuracy := float64(correct) / float64(total)
	if accuracy < 0.8 {
		t.Fatalf("accuracy too low: %.2f (expected >= 0.80)", accuracy)
	}
}
