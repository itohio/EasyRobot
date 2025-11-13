package gorgonia

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	// Test creating a new tensor
	tensor := New(types.FP32, 2, 3, 4)

	assert.Equal(t, 3, tensor.Rank(), "Rank should be 3")
	assert.Equal(t, 24, tensor.Size(), "Size should be 24 (2*3*4)")
	assert.False(t, tensor.Empty(), "Tensor should not be empty")

	shape := tensor.Shape()
	assert.Equal(t, 3, len(shape), "Shape should have 3 dimensions")
	assert.Equal(t, 2, shape[0], "First dimension should be 2")
	assert.Equal(t, 3, shape[1], "Second dimension should be 3")
	assert.Equal(t, 4, shape[2], "Third dimension should be 4")
}

func TestFill(t *testing.T) {
	// Test filling a tensor with a value
	tensor := New(types.FP32, 2, 3)
	tensor = tensor.Fill(nil, 5.0).(Tensor)

	// Check all elements are 5.0
	for i := 0; i < tensor.Size(); i++ {
		assert.Equal(t, 5.0, tensor.At(i), "Element should be 5.0")
	}
}

func TestClone(t *testing.T) {
	// Test cloning a tensor
	original := New(types.FP32, 2, 3)
	original.Fill(nil, 1.0)

	cloned := original.Clone().(Tensor)

	// Verify shapes match
	assert.True(t, original.Shape().Equal(cloned.Shape()), "Shapes should match")

	// Verify data matches
	for i := 0; i < original.Size(); i++ {
		assert.Equal(t, original.At(i), cloned.At(i), "Elements should match")
	}

	// Modify original and ensure clone is independent
	original.SetAt(99.0, 0, 0)
	assert.NotEqual(t, original.At(0, 0), cloned.At(0, 0), "Clone should be independent")
}

func TestAdd(t *testing.T) {
	// Test element-wise addition
	a := New(types.FP32, 2, 3)
	a.Fill(nil, 1.0)

	b := New(types.FP32, 2, 3)
	b.Fill(nil, 2.0)

	result := a.Add(nil, b).(Tensor)

	// Check all elements are 3.0
	for i := 0; i < result.Size(); i++ {
		assert.InDelta(t, 3.0, result.At(i), 0.001, "Element should be 3.0")
	}
}

func TestMultiply(t *testing.T) {
	// Test element-wise multiplication
	a := New(types.FP32, 2, 3)
	a.Fill(nil, 2.0)

	b := New(types.FP32, 2, 3)
	b.Fill(nil, 3.0)

	result := a.Multiply(nil, b).(Tensor)

	// Check all elements are 6.0
	for i := 0; i < result.Size(); i++ {
		assert.InDelta(t, 6.0, result.At(i), 0.001, "Element should be 6.0")
	}
}

func TestScalarMul(t *testing.T) {
	// Test scalar multiplication
	tensor := New(types.FP32, 2, 3)
	tensor.Fill(nil, 2.0)

	result := tensor.ScalarMul(nil, 3.0).(Tensor)

	// Check all elements are 6.0
	for i := 0; i < result.Size(); i++ {
		assert.InDelta(t, 6.0, result.At(i), 0.001, "Element should be 6.0")
	}
}

func TestReshape(t *testing.T) {
	// Test reshaping
	tensor := New(types.FP32, 2, 3, 4)

	// Fill with sequential values
	for i := 0; i < tensor.Size(); i++ {
		tensor.SetAt(float64(i), i)
	}

	// Reshape to [6, 4]
	reshaped := tensor.Reshape(nil, []int{6, 4}).(Tensor)

	assert.Equal(t, 2, reshaped.Rank(), "Rank should be 2")
	assert.Equal(t, 24, reshaped.Size(), "Size should still be 24")

	// Verify data is preserved
	for i := 0; i < reshaped.Size(); i++ {
		assert.Equal(t, float64(i), reshaped.At(i), "Data should be preserved")
	}
}

func TestMatMul(t *testing.T) {
	// Test matrix multiplication
	// Create a 2x3 matrix
	a := New(types.FP32, 2, 3)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a.SetAt(float64(i*3+j+1), i, j)
		}
	}

	// Create a 3x2 matrix
	b := New(types.FP32, 3, 2)
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			b.SetAt(float64(i*2+j+1), i, j)
		}
	}

	// Multiply: [2,3] x [3,2] = [2,2]
	result := a.MatMul(nil, b).(Tensor)

	assert.Equal(t, 2, result.Rank(), "Result rank should be 2")
	assert.Equal(t, 2, result.Shape()[0], "Result first dimension should be 2")
	assert.Equal(t, 2, result.Shape()[1], "Result second dimension should be 2")
}

func TestSum(t *testing.T) {
	// Test sum reduction
	tensor := New(types.FP32, 2, 3)
	tensor.Fill(nil, 2.0)

	// Sum all elements
	sum := tensor.Sum(nil, nil).(Tensor)

	assert.Equal(t, 1, sum.Size(), "Sum should be a scalar")
	assert.InDelta(t, 12.0, sum.At(0), 0.001, "Sum should be 12.0 (2*3*2)")
}

func TestTranspose(t *testing.T) {
	// Test transpose
	tensor := New(types.FP32, 2, 3)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			tensor.SetAt(float64(i*3+j), i, j)
		}
	}

	transposed := tensor.Transpose(nil, nil).(Tensor)

	shape := transposed.Shape()
	assert.Equal(t, 2, len(shape), "Transposed should have 2 dimensions")
	assert.Equal(t, 3, shape[0], "First dimension should be 3")
	assert.Equal(t, 2, shape[1], "Second dimension should be 2")
}
