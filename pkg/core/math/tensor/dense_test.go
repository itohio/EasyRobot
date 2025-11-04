package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFlat(t *testing.T) {
	tests := []struct {
		name     string
		tensor   *Tensor
		expected []float32
	}{
		{
			name:     "2x2 tensor",
			tensor:   FromFloat32(NewShape(2, 2), []float32{1, 2, 3, 4}),
			expected: []float32{1, 2, 3, 4},
		},
		{
			name:     "1D tensor",
			tensor:   FromFloat32(NewShape(5), []float32{1, 2, 3, 4, 5}),
			expected: []float32{1, 2, 3, 4, 5},
		},
		{
			name:     "3D tensor",
			tensor:   FromFloat32(NewShape(2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			expected: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:     "nil tensor",
			tensor:   nil,
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.tensor.Flat()

			if tt.tensor == nil {
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
			tensorData := tt.tensor.Data()
			if len(tensorData) > 0 {
				originalFirst := tensorData[0]
				result[0] = 999.0
				if tt.tensor.Data()[0] != 999.0 {
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
	tensor := FromFloat32(NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

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
	tensor3D := FromFloat32(NewShape(2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8})
	result := tensor3D.At(1, 0, 1)
	assert.InDelta(t, 6.0, result, 1e-6, "At(1, 0, 1) for 3D tensor = %f, expected 6.0", result)
}

func TestSetAt(t *testing.T) {
	tensor := FromFloat32(NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	tests := []struct {
		name        string
		indices     []int
		value       float32
		expected    float32
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
			name:        "too few indices",
			indices:     []int{0},
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
						assert.InDeltaf(t, tt.expected, result, 1e-6, "After SetAt(%v, %f), At(%v) = %f, expected %f",
							tt.indices, tt.value, tt.indices, result, tt.expected)
					}
				}
			}()

			tensorCopy.SetAt(tt.indices, tt.value)
		})
	}
}

func TestReshape(t *testing.T) {
	tests := []struct {
		name        string
		tensor      *Tensor
		newShape    []int
		expected    []float32
		shouldPanic bool
	}{
		{
			name:     "2x2 to 1x4",
			tensor:   FromFloat32(NewShape(2, 2), []float32{1, 2, 3, 4}),
			newShape: []int{1, 4},
			expected: []float32{1, 2, 3, 4},
		},
		{
			name:     "2x2 to 4",
			tensor:   FromFloat32(NewShape(2, 2), []float32{1, 2, 3, 4}),
			newShape: []int{4},
			expected: []float32{1, 2, 3, 4},
		},
		{
			name:     "6 to 2x3",
			tensor:   FromFloat32(NewShape(6), []float32{1, 2, 3, 4, 5, 6}),
			newShape: []int{2, 3},
			expected: []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name:     "2x2x2 to 8",
			tensor:   FromFloat32(NewShape(2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			newShape: []int{8},
			expected: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:     "2x2x2 to 4x2",
			tensor:   FromFloat32(NewShape(2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			newShape: []int{4, 2},
			expected: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:        "size mismatch",
			tensor:      FromFloat32(NewShape(2, 2), []float32{1, 2, 3, 4}),
			newShape:    []int{5},
			shouldPanic: true,
		},
		{
			name:        "zero dimension",
			tensor:      FromFloat32(NewShape(2, 2), []float32{1, 2, 3, 4}),
			newShape:    []int{0, 4},
			shouldPanic: true,
		},
		{
			name:        "negative dimension",
			tensor:      FromFloat32(NewShape(2, 2), []float32{1, 2, 3, 4}),
			newShape:    []int{-1, 4},
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

			result := tt.tensor.Reshape(tt.newShape)

			if tt.shouldPanic {
				return
			}

			if result == nil {
				t.Fatalf("Reshape() returned nil")
			}

			// Verify shape
			resultShape := result.Shape()
			if len(resultShape) != len(tt.newShape) {
				t.Errorf("Reshape() shape length = %d, expected %d", len(resultShape), len(tt.newShape))
			}

			for i := range tt.newShape {
				if resultShape[i] != tt.newShape[i] {
					t.Errorf("Reshape() Shape()[%d] = %d, expected %d", i, resultShape[i], tt.newShape[i])
				}
			}

			// Verify data is the same (zero-copy)
			resultData := result.Data()
			if len(resultData) != len(tt.expected) {
				t.Errorf("Reshape() Data length = %d, expected %d", len(resultData), len(tt.expected))
			}

			// Verify it's the same underlying slice (zero-copy)
			tensorData := tt.tensor.Data()
			if len(tensorData) > 0 {
				originalFirst := tensorData[0]
				resultData[0] = 999.0
				if tt.tensor.Data()[0] != 999.0 {
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

	// Test nil tensor
	result := (*Tensor)(nil).Reshape([]int{1})
	assert.Nil(t, result, "Reshape() of nil tensor should return nil")
}
