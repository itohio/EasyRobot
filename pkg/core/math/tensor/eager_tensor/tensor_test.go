package eager_tensor

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestFlat(t *testing.T) {
	tests := []struct {
		name     string
		tensor   Tensor
		expected []float32
	}{
		{
			name:     "2x2 tensor",
			tensor:   FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			expected: []float32{1, 2, 3, 4},
		},
		{
			name:     "1D tensor",
			tensor:   FromFloat32(types.NewShape(5), []float32{1, 2, 3, 4, 5}),
			expected: []float32{1, 2, 3, 4, 5},
		},
		{
			name:     "3D tensor",
			tensor:   FromFloat32(types.NewShape(2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			expected: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:     "nil tensor",
			tensor:   Tensor{},
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.tensor.Data().([]float32)

			if tt.tensor.Empty() {
				if result != nil {
					t.Errorf("Flat() of nil tensor should return nil, got %v", result)
				}
				return
			}

			// Check that Flat returns the same slice (zero-copy)
			if len(result) != len(tt.expected) {
				t.Errorf("Flat() length = %d, expected %d", len(result), len(tt.expected))
			}

			// Verify it's the same underlying slice (zero-copy)
			tensorData := tt.tensor.Data().([]float32)
			if len(tensorData) > 0 {
				originalFirst := tensorData[0]
				result[0] = 999.0
				if tt.tensor.Data().([]float32)[0] != 999.0 {
					t.Errorf("Flat() should return same slice (zero-copy), but modifying result didn't modify original")
				}
				// Restore
				result[0] = originalFirst
			}

			for i := range tt.expected {
				assert.InDeltaf(t, tt.expected[i], result[i], 1e-6, "Flat()[%d] = %f, expected %f", i, result[i], tt.expected[i])
			}
		})
	}
}

func TestAt(t *testing.T) {
	tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	tests := []struct {
		name        string
		indices     []int
		expected    float32
		shouldPanic bool
	}{
		{
			name:     "access [0, 0]",
			indices:  []int{0, 0},
			expected: 1.0,
		},
		{
			name:     "access [0, 1]",
			indices:  []int{0, 1},
			expected: 2.0,
		},
		{
			name:     "access [1, 2]",
			indices:  []int{1, 2},
			expected: 6.0,
		},
		{
			name:     "access [1, 0]",
			indices:  []int{1, 0},
			expected: 4.0,
		},
		{
			name:        "too few indices",
			indices:     []int{0},
			shouldPanic: false, // Single index uses linear indexing for rank > 1
			expected:    1.0,   // Linear index 0
		},
		{
			name:     "linear indexing - single index",
			indices:  []int{3},
			expected: 4.0, // Linear index 3 (row-major: [1,2,3,4,5,6], index 3 is 4)
		},
		{
			name:     "linear indexing - last element",
			indices:  []int{5},
			expected: 6.0, // Last element
		},
		{
			name:        "linear indexing - out of bounds",
			indices:     []int{6},
			shouldPanic: true,
		},
		{
			name:        "too many indices",
			indices:     []int{0, 0, 0},
			shouldPanic: true,
		},
		{
			name:        "negative index",
			indices:     []int{-1, 0},
			shouldPanic: true,
		},
		{
			name:        "index out of bounds",
			indices:     []int{2, 0},
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if tt.shouldPanic {
					if r == nil {
						t.Errorf("At() should have panicked")
					}
				} else {
					if r != nil {
						t.Errorf("At() panicked unexpectedly: %v", r)
					}
				}
			}()

			result := tensor.At(tt.indices...)
			if !tt.shouldPanic {
				assert.InDelta(t, tt.expected, result, 1e-6, "At(%v) = %f, expected %f", tt.indices, result, tt.expected)
			}
		})
	}

	// Test 3D tensor
	tensor3D := FromFloat32(types.NewShape(2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8})
	result := tensor3D.At(1, 0, 1)
	assert.InDelta(t, 6.0, result, 1e-6, "At(1, 0, 1) for 3D tensor = %f, expected 6.0", result)

	// Test 3D tensor with linear indexing
	resultLinear := tensor3D.At(3)
	assert.InDelta(t, 4.0, resultLinear, 1e-6, "At(3) for 3D tensor (linear) = %f, expected 4.0", resultLinear)

	// Test 1D tensor - single index should still use normal indexing (not linear)
	tensor1D := FromFloat32(types.NewShape(3), []float32{10, 20, 30})
	result1D := tensor1D.At(1)
	assert.InDelta(t, 20.0, result1D, 1e-6, "At(1) for 1D tensor = %f, expected 20.0", result1D)
}

func TestSetAt(t *testing.T) {
	tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	tests := []struct {
		name        string
		indices     []int
		value       float64
		expected    float64
		shouldPanic bool
	}{
		{
			name:     "set [0, 0]",
			indices:  []int{0, 0},
			value:    10.0,
			expected: 10.0,
		},
		{
			name:     "set [1, 2]",
			indices:  []int{1, 2},
			value:    20.0,
			expected: 20.0,
		},
		{
			name:        "linear indexing - single index",
			indices:     []int{3},
			value:       99.0,
			expected:    99.0, // Linear index 3
			shouldPanic: false,
		},
		{
			name:        "linear indexing - out of bounds",
			indices:     []int{10},
			value:       5.0,
			shouldPanic: true,
		},
		{
			name:        "negative index",
			indices:     []int{-1, 0},
			value:       5.0,
			shouldPanic: true,
		},
		{
			name:        "index out of bounds",
			indices:     []int{0, 10},
			value:       5.0,
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a fresh copy for each test
			tensorCopy := tensor.Clone()

			defer func() {
				r := recover()
				if tt.shouldPanic {
					if r == nil {
						t.Errorf("SetAt() should have panicked")
					}
				} else {
					if r != nil {
						t.Errorf("SetAt() panicked unexpectedly: %v", r)
					} else {
						// Verify the value was set
						result := tensorCopy.At(tt.indices...)
						assert.InDeltaf(t, tt.expected, result, 1e-6, "After SetAt(%f, %v...), At(%v...) = %f, expected %f",
							tt.value, tt.indices, tt.indices, result, tt.expected)
					}
				}
			}()

			tensorCopy.SetAt(tt.value, tt.indices...)
		})
	}
}

func TestReshape(t *testing.T) {
	tests := []struct {
		name        string
		tensor      Tensor
		shape       []int
		expected    []float32
		shouldPanic bool
	}{
		{
			name:     "2x2 to 1x4",
			tensor:   FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			shape:    []int{1, 4},
			expected: []float32{1, 2, 3, 4},
		},
		{
			name:     "2x2 to 4",
			tensor:   FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			shape:    []int{4},
			expected: []float32{1, 2, 3, 4},
		},
		{
			name:     "6 to 2x3",
			tensor:   FromFloat32(types.NewShape(6), []float32{1, 2, 3, 4, 5, 6}),
			shape:    []int{2, 3},
			expected: []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name:     "2x2x2 to 8",
			tensor:   FromFloat32(types.NewShape(2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			shape:    []int{8},
			expected: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:     "2x2x2 to 4x2",
			tensor:   FromFloat32(types.NewShape(2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			shape:    []int{4, 2},
			expected: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:        "size mismatch",
			tensor:      FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			shape:       []int{5},
			shouldPanic: true,
		},
		{
			name:        "zero dimension",
			tensor:      FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			shape:       []int{0, 4},
			shouldPanic: true,
		},
		{
			name:        "negative dimension",
			tensor:      FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			shape:       []int{-1, 4},
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if tt.shouldPanic {
					if r == nil {
						t.Errorf("Reshape() should have panicked")
					}
				} else {
					if r != nil {
						t.Errorf("Reshape() panicked unexpectedly: %v", r)
					}
				}
			}()

			result := tt.tensor.Reshape(nil, tt.shape)

			if tt.shouldPanic {
				return
			}

			if result == nil {
				t.Fatalf("Reshape() returned nil")
			}

			// Verify shape
			resultShape := result.Shape()
			if len(resultShape) != len(tt.shape) {
				t.Errorf("Reshape() shape length = %d, expected %d", len(resultShape), len(tt.shape))
			}

			for i := range tt.shape {
				if resultShape[i] != tt.shape[i] {
					t.Errorf("Reshape() Shape()[%d] = %d, expected %d", i, resultShape[i], tt.shape[i])
				}
			}

			// Verify data is the same (zero-copy)
			resultData := result.Data().([]float32)
			if len(resultData) != len(tt.expected) {
				t.Errorf("Reshape() Data length = %d, expected %d", len(resultData), len(tt.expected))
			}

			// Verify it's the same underlying slice (zero-copy)
			tensorData := tt.tensor.Data().([]float32)
			if len(tensorData) > 0 {
				originalFirst := tensorData[0]
				resultData[0] = 999.0
				if tt.tensor.Data().([]float32)[0] != 999.0 {
					t.Errorf("Reshape() should return same slice (zero-copy), but modifying result didn't modify original")
				}
				// Restore
				resultData[0] = originalFirst
			}

			for i := range tt.expected {
				assert.InDeltaf(t, tt.expected[i], resultData[i], 1e-5, "Reshape() Data[%d] = %f, expected %f", i, resultData[i], tt.expected[i])
			}
		})
	}
}

func TestReshapeWithDst(t *testing.T) {
	t.Run("with dst parameter", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4})
		newShape := types.NewShape(4)

		// Test with dst parameter
		dst := New(types.FP32, newShape)
		result := tensor.Reshape(dst, newShape)

		if result.ID() != dst.ID() {
			t.Errorf("Reshape() with dst should return dst (same ID), got different tensor")
		}

		resultData := result.Data().([]float32)
		expected := []float32{1, 2, 3, 4}
		for i := range expected {
			assert.InDeltaf(t, expected[i], resultData[i], 1e-5, "Reshape() with dst Data[%d] = %f, expected %f", i, resultData[i], expected[i])
		}
	})

	t.Run("dst shape mismatch", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4})
		newShape := types.NewShape(4)
		dst := New(types.FP32, types.NewShape(5)) // Wrong shape

		defer func() {
			r := recover()
			if r == nil {
				t.Errorf("Reshape() with mismatched dst shape should panic")
			}
		}()

		tensor.Reshape(dst, newShape)
	})
}

func TestTensorIterator(t *testing.T) {
	t.Run("iterate all elements", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

		expected := []float32{1, 2, 3, 4, 5, 6}
		var got []float32

		for elem := range tensor.Elements() {
			got = append(got, float32(elem.Get()))
		}

		assert.Equal(t, expected, got)
	})

	t.Run("set all elements", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

		value := float32(10.0)
		for elem := range tensor.Elements() {
			elem.Set(float64(value))
			value++
		}

		// Verify all values were set
		value = float32(10.0)
		data := tensor.Data().([]float32)
		for i := range data {
			assert.Equal(t, value, data[i])
			value++
		}
	})

	t.Run("get and set operations", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

		// Double all values
		for elem := range tensor.Elements() {
			val := elem.Get()
			elem.Set(val * 2.0)
		}

		expected := []float32{2, 4, 6, 8, 10, 12}
		data := tensor.Data().([]float32)
		for i := range expected {
			assert.Equal(t, expected[i], data[i])
		}
	})

	t.Run("fix first dimension", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3, 4), make([]float32, 2*3*4))

		// Initialize with index values
		idx := 0
		for elem := range tensor.Elements() {
			elem.Set(float64(idx))
			idx++
		}

		// Fix dimension 0 at index 1, iterate over remaining
		count := 0
		for elem := range tensor.Elements(0, 1) {
			// Should only iterate over dimensions 1 and 2 (12 elements)
			_ = elem.Get()
			count++
		}

		assert.Equal(t, 3*4, count) // 3 * 4 = 12 combinations
	})

	t.Run("fix multiple dimensions", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3, 4, 5), make([]float32, 2*3*4*5))

		// Fix dimensions 0 and 2, iterate over 1 and 3
		count := 0
		for elem := range tensor.Elements(0, 1, 2, 3) {
			// Should iterate over dimensions 1 and 3 (15 elements)
			elem.Set(float64(count))
			count++
		}

		assert.Equal(t, 3*5, count) // 3 * 5 = 15 combinations
	})

	t.Run("fix all dimensions", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24})

		// Fix all dimensions - should iterate once
		count := 0
		var value float64
		for elem := range tensor.Elements(0, 1, 1, 2, 2, 3) {
			value = elem.Get()
			count++
		}

		assert.Equal(t, 1, count)
		// Value at [1, 2, 3] should be at index: 1*12 + 2*4 + 3 = 12 + 8 + 3 = 23
		assert.Equal(t, float64(24), value) // 0-indexed, so 24th element
	})

	t.Run("row-major order", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

		var got []float32
		for elem := range tensor.Elements() {
			got = append(got, float32(elem.Get()))
		}

		// Row-major: last dimension changes fastest
		expected := []float32{1, 2, 3, 4, 5, 6}
		assert.Equal(t, expected, got)
	})

	t.Run("empty tensor", func(t *testing.T) {
		// Empty shape has size 1 (scalar)
		tensor := FromFloat32(types.NewShape(), []float32{42})

		count := 0
		var value float64
		for elem := range tensor.Elements() {
			value = elem.Get()
			count++
		}

		assert.Equal(t, 1, count)
		assert.Equal(t, float64(42), value)
	})

	t.Run("modify during iteration", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

		// Modify elements as we iterate
		for elem := range tensor.Elements() {
			val := elem.Get()
			elem.Set(val + 10.0)
		}

		expected := []float32{11, 12, 13, 14, 15, 16}
		data := tensor.Data().([]float32)
		for i := range expected {
			assert.Equal(t, expected[i], data[i])
		}
	})

	t.Run("single dimension", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(3), []float32{10, 20, 30})

		var got []float32
		for elem := range tensor.Elements() {
			got = append(got, float32(elem.Get()))
		}

		assert.Equal(t, []float32{10, 20, 30}, got)
	})
}
