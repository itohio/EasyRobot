//go:build tflite
// +build tflite

package marshaller_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/marshaller"
	tflitepkg "github.com/itohio/EasyRobot/pkg/core/marshaller/tflite"
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

func TestTFLiteUnmarshallerMNISTInference(t *testing.T) {
	u, err := marshaller.NewUnmarshaller("tflite")
	if err != nil {
		t.Fatalf("failed to create tflite unmarshaller: %v", err)
	}

	modelPath := filepath.Join("testdata", "mnist_model.tflite")
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatalf("failed to open model file: %v", err)
	}
	defer modelFile.Close()

	var model *tflitepkg.Model
	if err := u.Unmarshal(modelFile, &model); err != nil {
		if isTFLiteUnavailable(err) {
			t.Skipf("skipping TFLite tests: %v", err)
		}
		t.Fatalf("failed to unmarshal TFLite model: %v", err)
	}
	if model == nil {
		t.Fatal("unmarshal succeeded but model is nil")
	}
	t.Cleanup(func() {
		model.Close()
	})

	inputShape := model.InputShape()
	if inputShape == nil || inputShape.Rank() == 0 {
		t.Fatalf("unexpected input shape: %v", inputShape)
	}
	if err := model.Init(inputShape.Clone()); err != nil {
		t.Fatalf("Init failed: %v", err)
	}

	const sampleLimit = 500

	dataPath := filepath.Join("..", "math", "learn", "datasets", "mnist", "mnist_test.csv.gz")
	samples, err := mnist.Load(dataPath, sampleLimit)
	if err != nil {
		t.Skipf("skipping inference test due to dataset load error: %v", err)
	}
	if len(samples) == 0 {
		t.Skip("no MNIST samples available")
	}

	inputSize := inputShape.Size()
	if sampleSize := samples[0].Image.Shape().Size(); sampleSize != inputSize {
		t.Skipf("sample size %d does not match model input size %d", sampleSize, inputSize)
	}

	const (
		requiredAccuracy = 0.80
		maxLoggedSamples = 5
	)

	correct := 0
	total := len(samples)

	for idx, sample := range samples {
		input := tensor.New(tensor.DTFP32, inputShape.Clone())
		inputData, ok := input.Data().([]float32)
		if !ok {
			t.Fatalf("input tensor data is not []float32 (got %T)", input.Data())
		}

		sampleData, ok := sample.Image.Data().([]float32)
		if !ok {
			t.Fatalf("sample %d image data is not []float32 (got %T)", idx, sample.Image.Data())
		}
		if len(sampleData) != len(inputData) {
			t.Fatalf("sample %d data length %d does not match input length %d", idx, len(sampleData), len(inputData))
		}
		copy(inputData, sampleData)

		output, err := model.Forward(input)
		if err != nil {
			t.Fatalf("forward pass failed for sample %d: %v", idx, err)
		}

		if output.Size() == 0 {
			t.Fatalf("sample %d: empty output tensor", idx)
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

		if idx < maxLoggedSamples {
			t.Logf("sample %d: label=%d predicted=%d confidence=%.4f", idx, sample.Label, predicted, maxVal)
		}
	}

	accuracy := float64(correct) / float64(total)
	t.Logf("TFLite MNIST inference: samples=%d correct=%d accuracy=%.4f", total, correct, accuracy)

	if accuracy < requiredAccuracy {
		t.Fatalf("accuracy %.2f below expected threshold %.2f", accuracy, requiredAccuracy)
	}
}
