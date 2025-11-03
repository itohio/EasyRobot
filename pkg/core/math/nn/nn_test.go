package nn

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

func TestLinear(t *testing.T) {
	t.Run("single sample", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}
		weight := &tensor.Tensor{Dim: []int{3, 4}, Data: []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
		}}
		bias := &tensor.Tensor{Dim: []int{4}, Data: []float32{1, 1, 1, 1}}

		result := Linear(input, weight, bias)

		expectedShape := []int{4}
		if len(result.Dim) != len(expectedShape) {
			t.Fatalf("Shape mismatch: got %v, expected %v", result.Dim, expectedShape)
		}

		if result.Dim[0] != 4 {
			t.Errorf("Output size = %d, expected 4", result.Dim[0])
		}

		// Check first output: 1*1 + 2*5 + 3*9 + 1 = 1 + 10 + 27 + 1 = 39
		expected := float32(39.0)
		if !floatEqual(result.Data[0], expected) {
			t.Errorf("result[0] = %f, expected %f", result.Data[0], expected)
		}
	})

	t.Run("batch", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{2, 3}, Data: []float32{
			1, 2, 3,
			4, 5, 6,
		}}
		weight := &tensor.Tensor{Dim: []int{3, 2}, Data: []float32{
			1, 2,
			3, 4,
			5, 6,
		}}
		bias := &tensor.Tensor{Dim: []int{2}, Data: []float32{1, 1}}

		result := Linear(input, weight, bias)

		expectedShape := []int{2, 2}
		if len(result.Dim) != len(expectedShape) {
			t.Fatalf("Shape mismatch: got %v, expected %v", result.Dim, expectedShape)
		}

		// First sample: [1,2,3] × weight + bias
		// [1*1+2*3+3*5, 1*2+2*4+3*6] + [1,1]
		// [22, 28] + [1,1] = [23, 29]
		if !floatEqual(result.Data[0], 23.0) {
			t.Errorf("result[0] = %f, expected 23.0", result.Data[0])
		}
		if !floatEqual(result.Data[1], 29.0) {
			t.Errorf("result[1] = %f, expected 29.0", result.Data[1])
		}
	})

	t.Run("no bias", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{2}, Data: []float32{1, 2}}
		weight := &tensor.Tensor{Dim: []int{2, 3}, Data: []float32{
			1, 2, 3,
			4, 5, 6,
		}}

		result := Linear(input, weight, nil)

		expectedShape := []int{3}
		if len(result.Dim) != len(expectedShape) {
			t.Fatalf("Shape mismatch: got %v, expected %v", result.Dim, expectedShape)
		}

		// [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
		expected := []float32{9, 12, 15}
		for i := range expected {
			if !floatEqual(result.Data[i], expected[i]) {
				t.Errorf("result[%d] = %f, expected %f", i, result.Data[i], expected[i])
			}
		}
	})
}

func TestLinearTo(t *testing.T) {
	t.Run("create new tensor", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}
		weight := &tensor.Tensor{Dim: []int{3, 2}, Data: []float32{1, 2, 3, 4, 5, 6}}

		result := Linear(input, weight, nil)

		if result == nil {
			t.Fatal("LinearTo returned nil")
		}

		expectedShape := []int{2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})

	t.Run("verify correct shape", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{2}, Data: []float32{1, 2}}
		weight := &tensor.Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
		result := Linear(input, weight, nil)

		// Test that result has correct shape
		if len(result.Shape()) != 1 || result.Shape()[0] != 3 {
			t.Errorf("Result shape should be [3], got %v", result.Shape())
		}
	})
}

func TestRelu(t *testing.T) {
	t.Run("basic ReLU", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{4}, Data: []float32{-2, -1, 0, 1}}
		originalData := make([]float32, len(input.Data))
		copy(originalData, input.Data)

		result := Relu(input)

		if result != input {
			t.Errorf("Relu should return self for in-place operation")
		}

		expected := []float32{0, 0, 0, 1}
		for i := range expected {
			if input.Data[i] != expected[i] {
				t.Errorf("Data[%d] = %f, expected %f", i, input.Data[i], expected[i])
			}
		}

		// Original data should be modified (in-place)
		// Note: input and result are the same, so input.Data has been modified
		if input.Data[0] == originalData[0] {
			t.Error("Tensor should be modified in-place by Relu")
		}
	})

	t.Run("all positive", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}
		result := Relu(input)

		expected := []float32{1, 2, 3}
		for i := range expected {
			if result.Data[i] != expected[i] {
				t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], expected[i])
			}
		}
	})
}

func TestSigmoid(t *testing.T) {
	t.Run("basic sigmoid", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{3}, Data: []float32{0, 1, -1}}
		result := Sigmoid(input)

		// Sigmoid(0) = 0.5
		if !floatEqual(result.Data[0], 0.5) {
			t.Errorf("Sigmoid(0) = %f, expected 0.5", result.Data[0])
		}

		// Sigmoid(1) ≈ 0.731
		expected1 := float32(1.0 / (1.0 + math.Exp(-1.0)))
		if !floatEqual(result.Data[1], expected1) {
			t.Errorf("Sigmoid(1) = %f, expected %f", result.Data[1], expected1)
		}

		// Sigmoid(-1) ≈ 0.269
		expectedNeg1 := float32(1.0 / (1.0 + math.Exp(1.0)))
		if !floatEqual(result.Data[2], expectedNeg1) {
			t.Errorf("Sigmoid(-1) = %f, expected %f", result.Data[2], expectedNeg1)
		}
	})

	t.Run("extreme values", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{2}, Data: []float32{20, -20}}
		result := Sigmoid(input)

		// Large positive should saturate to 1
		if !floatEqual(result.Data[0], 1.0) {
			t.Errorf("Sigmoid(20) = %f, expected ~1.0", result.Data[0])
		}

		// Large negative should saturate to 0
		if !floatEqual(result.Data[1], 0.0) {
			t.Errorf("Sigmoid(-20) = %f, expected ~0.0", result.Data[1])
		}
	})

	t.Run("original unchanged", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{2}, Data: []float32{1, 2}}
		original := input.Clone()
		result := Sigmoid(input)

		// Original should be unchanged (Sigmoid creates new tensor)
		if input.Data[0] != original.Data[0] {
			t.Error("Original tensor should be unchanged")
		}

		// Result should be different
		if result.Data[0] == input.Data[0] {
			t.Error("Result should be different from input")
		}
	})
}

func TestTanh(t *testing.T) {
	t.Run("basic tanh", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{3}, Data: []float32{0, 1, -1}}
		result := Tanh(input)

		// Tanh(0) = 0
		if !floatEqual(result.Data[0], 0.0) {
			t.Errorf("Tanh(0) = %f, expected 0.0", result.Data[0])
		}

		// Tanh(1) ≈ 0.762
		expected1 := float32(math.Tanh(1.0))
		if !floatEqual(result.Data[1], expected1) {
			t.Errorf("Tanh(1) = %f, expected %f", result.Data[1], expected1)
		}

		// Tanh(-1) ≈ -0.762
		expectedNeg1 := float32(math.Tanh(-1.0))
		if !floatEqual(result.Data[2], expectedNeg1) {
			t.Errorf("Tanh(-1) = %f, expected %f", result.Data[2], expectedNeg1)
		}
	})

	t.Run("original unchanged", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{2}, Data: []float32{1, 2}}
		original := input.Clone()
		result := Tanh(input)

		// Original should be unchanged
		if input.Data[0] != original.Data[0] {
			t.Error("Original tensor should be unchanged")
		}

		// Result should be different
		if result.Data[0] == input.Data[0] {
			t.Error("Result should be different from input")
		}
	})
}

func TestSoftmax(t *testing.T) {
	t.Run("1D tensor", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}
		result := Softmax(input, 0)

		// Check that sum equals 1
		var sum float32
		for i := range result.Data {
			sum += result.Data[i]
		}

		if !floatEqual(sum, 1.0) {
			t.Errorf("Softmax sum = %f, expected 1.0", sum)
		}

		// Check that all values are positive
		for i := range result.Data {
			if result.Data[i] < 0 {
				t.Errorf("Softmax[%d] = %f, expected non-negative", i, result.Data[i])
			}
		}

		// Largest input should have largest probability
		if result.Data[2] <= result.Data[0] || result.Data[2] <= result.Data[1] {
			t.Error("Largest input should have largest softmax value")
		}
	})

	t.Run("2D tensor along rows", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{2, 3}, Data: []float32{
			1, 2, 3,
			4, 5, 6,
		}}
		result := Softmax(input, 0)

		// Sum along rows (dim 0) should equal 1 for each column
		for j := 0; j < 3; j++ {
			sum := result.Data[j] + result.Data[3+j]
			if !floatEqual(sum, 1.0) {
				t.Errorf("Column %d sum = %f, expected 1.0", j, sum)
			}
		}
	})

	t.Run("2D tensor along columns", func(t *testing.T) {
		input := &tensor.Tensor{Dim: []int{2, 3}, Data: []float32{
			1, 2, 3,
			4, 5, 6,
		}}
		result := Softmax(input, 1)

		// Sum along columns (dim 1) should equal 1 for each row
		for i := 0; i < 2; i++ {
			sum := result.Data[i*3] + result.Data[i*3+1] + result.Data[i*3+2]
			if !floatEqual(sum, 1.0) {
				t.Errorf("Row %d sum = %f, expected 1.0", i, sum)
			}
		}
	})
}

func TestMSE(t *testing.T) {
	t.Run("basic MSE", func(t *testing.T) {
		pred := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}
		target := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}

		loss := MSE(pred, target)

		if !floatEqual(loss, 0.0) {
			t.Errorf("MSE = %f, expected 0.0", loss)
		}
	})

	t.Run("non-zero MSE", func(t *testing.T) {
		pred := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}
		target := &tensor.Tensor{Dim: []int{3}, Data: []float32{2, 3, 4}}

		loss := MSE(pred, target)

		// MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2) = mean(1 + 1 + 1) = 1
		expected := float32(1.0)
		if !floatEqual(loss, expected) {
			t.Errorf("MSE = %f, expected %f", loss, expected)
		}
	})

	t.Run("2D tensor", func(t *testing.T) {
		pred := &tensor.Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}}
		target := &tensor.Tensor{Dim: []int{2, 2}, Data: []float32{0, 0, 0, 0}}

		loss := MSE(pred, target)

		// MSE = mean(1^2 + 2^2 + 3^2 + 4^2) = mean(1+4+9+16) = 30/4 = 7.5
		expected := float32(7.5)
		if !floatEqual(loss, expected) {
			t.Errorf("MSE = %f, expected %f", loss, expected)
		}
	})
}

func TestCrossEntropy(t *testing.T) {
	t.Run("basic cross entropy", func(t *testing.T) {
		// Predictions (softmax probabilities)
		pred := &tensor.Tensor{Dim: []int{3}, Data: []float32{0.8, 0.1, 0.1}}
		// Target (one-hot: class 0)
		target := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 0, 0}}

		loss := CrossEntropy(pred, target)

		// CE = -log(0.8) ≈ 0.223
		expected := float32(-math.Log(0.8))
		if !floatEqual(loss, expected) {
			t.Errorf("CrossEntropy = %f, expected %f", loss, expected)
		}

		// Loss should be positive
		if loss < 0 {
			t.Error("CrossEntropy should be non-negative")
		}
	})

	t.Run("perfect prediction", func(t *testing.T) {
		pred := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 0, 0}}
		target := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 0, 0}}

		loss := CrossEntropy(pred, target)

		// CE = -log(1) = 0
		if !floatEqual(loss, 0.0) {
			t.Errorf("CrossEntropy for perfect prediction = %f, expected 0.0", loss)
		}
	})
}

func TestDetach(t *testing.T) {
	t.Run("detach creates clone", func(t *testing.T) {
		ten := &tensor.Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}
		result := ten.Clone() // Detach is now Clone in tensor package

		if result == ten {
			t.Error("Detach should create new tensor")
		}

		// Data should be same
		for i := range ten.Data {
			if result.Data[i] != ten.Data[i] {
				t.Errorf("Detached data[%d] = %f, expected %f", i, result.Data[i], ten.Data[i])
			}
		}

		// Modifying result should not affect original
		result.Data[0] = 999
		if ten.Data[0] == 999 {
			t.Error("Modifying detached tensor should not affect original")
		}
	})
}

// Helper function for float32 comparison with epsilon
func floatEqual(a, b float32) bool {
	const epsilon = 1e-5
	return float32(math.Abs(float64(a-b))) < epsilon
}
