package eager_tensor

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

func TestTensorAvgPool2DBackwardMatchesPrimitive(t *testing.T) {
	inputData := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	input := FromFloat32(types.NewShape(1, 1, 3, 3), inputData)

	kernel := []int{2, 2}
	stride := []int{2, 2}
	padding := []int{1, 1}

	gradOutput := FromFloat32(types.NewShape(1, 1, 2, 2), []float32{1, 1, 1, 1})

	gradInput := input.AvgPool2DBackward(nil, gradOutput, kernel, stride, padding)

	expected := []float32{
		1, 0.5, 0.5,
		0.5, 0.25, 0.25,
		0.5, 0.25, 0.25,
	}

	assertTensorClose(t, gradInput, expected, 1e-6)
}

func TestTensorMaxPool2DBackwardMatchesPrimitive(t *testing.T) {
	inputData := []float32{
		1, 2, 3, 4,
		5, 9, 8, 7,
		0, 6, 5, 4,
		3, 2, 1, 0,
	}
	input := FromFloat32(types.NewShape(1, 1, 4, 4), inputData)

	kernel := []int{2, 2}
	stride := []int{2, 2}
	padding := []int{0, 0}

	output, indices := input.MaxPool2DWithIndices(nil, nil, kernel, stride, padding)
	if output == nil || indices == nil {
		t.Fatalf("maxpool forward returned nil")
	}

	gradOutput := FromFloat32(types.NewShape(1, 1, 2, 2), []float32{1, 1, 1, 1})

	gradInput := input.MaxPool2DBackward(nil, gradOutput, indices, kernel, stride, padding)

	expected := []float32{
		0, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 0, 0, 0,
	}

	assertTensorClose(t, gradInput, expected, 1e-6)
}

func TestTensorAvgPool2DBackwardClearsDestination(t *testing.T) {
	inputData := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	input := FromFloat32(types.NewShape(1, 1, 3, 3), inputData)

	kernel := []int{2, 2}
	stride := []int{2, 2}
	padding := []int{1, 1}

	gradOutput := FromFloat32(types.NewShape(1, 1, 2, 2), []float32{1, 1, 1, 1})

	preFilled := make([]float32, 9)
	for i := range preFilled {
		preFilled[i] = float32(100 + i)
	}
	dst := FromFloat32(types.NewShape(1, 1, 3, 3), preFilled)

	result := input.AvgPool2DBackward(dst, gradOutput, kernel, stride, padding)

	expected := []float32{
		1, 0.5, 0.5,
		0.5, 0.25, 0.25,
		0.5, 0.25, 0.25,
	}

	assertTensorClose(t, result, expected, 1e-6)
}

func TestTensorMaxPool2DBackwardClearsDestination(t *testing.T) {
	inputData := []float32{
		1, 2, 3, 4,
		5, 9, 8, 7,
		0, 6, 5, 4,
		3, 2, 1, 0,
	}
	input := FromFloat32(types.NewShape(1, 1, 4, 4), inputData)

	kernel := []int{2, 2}
	stride := []int{2, 2}
	padding := []int{0, 0}

	_, indices := input.MaxPool2DWithIndices(nil, nil, kernel, stride, padding)

	gradOutput := FromFloat32(types.NewShape(1, 1, 2, 2), []float32{1, 1, 1, 1})

	preFilled := make([]float32, 16)
	for i := range preFilled {
		preFilled[i] = float32(50 + i)
	}
	dst := FromFloat32(types.NewShape(1, 1, 4, 4), preFilled)

	result := input.MaxPool2DBackward(dst, gradOutput, indices, kernel, stride, padding)

	expected := []float32{
		0, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 0, 0, 0,
	}

	assertTensorClose(t, result, expected, 1e-6)
}

func assertTensorClose(t *testing.T, tensor types.Tensor, expected []float32, tol float64) {
	t.Helper()
	if tensor == nil {
		t.Fatalf("tensor is nil")
	}

	data, ok := tensor.Data().([]float32)
	if !ok {
		t.Fatalf("unexpected tensor data type %T", tensor.Data())
	}

	if len(data) != len(expected) {
		t.Fatalf("data length %d does not match expected %d", len(data), len(expected))
	}

	for i, got := range data {
		want := expected[i]
		if absDiff(float64(got), float64(want)) > tol {
			t.Fatalf("value mismatch at %d: got %f want %f", i, got, want)
		}
	}
}

func absDiff(a, b float64) float64 {
	return math.Abs(a - b)
}
