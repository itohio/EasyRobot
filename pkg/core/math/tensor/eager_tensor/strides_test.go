package eager_tensor

import (
	"testing"
	"unsafe"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestStrides(t *testing.T) {
	tests := []struct {
		name            string
		tensor          Tensor
		expectedStrides []int
		expectedNil     bool
	}{
		{
			name:            "contiguous 2x3 tensor",
			tensor:          FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6}),
			expectedStrides: []int{3, 1},
			expectedNil:     false,
		},
		{
			name:            "contiguous 2x3x4 tensor",
			tensor:          FromFloat32(types.NewShape(2, 3, 4), make([]float32, 24)),
			expectedStrides: []int{12, 4, 1},
			expectedNil:     false,
		},
		{
			name:            "1D tensor",
			tensor:          FromFloat32(types.NewShape(5), []float32{1, 2, 3, 4, 5}),
			expectedStrides: []int{1},
			expectedNil:     false,
		},
		{
			name:            "empty tensor",
			tensor:          Tensor{},
			expectedStrides: nil,
			expectedNil:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var dstStatic [helpers.MAX_DIMS]int
			strides := tt.tensor.Strides(dstStatic[:tt.tensor.Rank()])

			if tt.expectedNil {
				assert.Nil(t, strides)
				return
			}

			assert.NotNil(t, strides)
			assert.Equal(t, len(tt.expectedStrides), len(strides))
			for i := range tt.expectedStrides {
				assert.Equal(t, tt.expectedStrides[i], strides[i], "strides[%d]", i)
			}
		})
	}
}

func TestStrides_WithDestination(t *testing.T) {
	tensor := FromFloat32(types.NewShape(2, 3, 4), make([]float32, 24))

	// Test with destination buffer
	var dstStatic [helpers.MAX_DIMS]int
	dst := dstStatic[:3]
	strides := tensor.Strides(dst)

	assert.NotNil(t, strides)
	assert.Equal(t, []int{12, 4, 1}, strides)
	assert.Equal(t, dst, strides[:3]) // Should use provided destination
}

func TestStrides_StoredStrides(t *testing.T) {
	// Create tensor with explicit strides (non-contiguous)
	shape := types.NewShape(2, 3)
	explicitStrides := []int{6, 1} // Non-contiguous: ld = 6 instead of 3
	tensor := Tensor{
		shape:   shape,
		data:    make([]float32, 12),
		strides: explicitStrides,
		offset:  0,
	}

	var dstStatic [helpers.MAX_DIMS]int
	strides := tensor.Strides(dstStatic[:2])

	assert.NotNil(t, strides)
	assert.Equal(t, explicitStrides, strides)
}

func TestIsContiguous(t *testing.T) {
	tests := []struct {
		name        string
		tensor      Tensor
		expected    bool
		description string
	}{
		{
			name:        "contiguous tensor (nil strides)",
			tensor:      FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6}),
			expected:    true,
			description: "New tensors have nil strides (contiguous)",
		},
		{
			name: "non-contiguous tensor (explicit strides)",
			tensor: Tensor{
				shape:   types.NewShape(2, 3),
				data:    make([]float32, 12),
				strides: []int{6, 1}, // ld = 6, not 3 (non-contiguous)
				offset:  0,
			},
			expected:    false,
			description: "Explicit strides that don't match canonical",
		},
		{
			name: "contiguous tensor (explicit matching strides)",
			tensor: Tensor{
				shape:   types.NewShape(2, 3),
				data:    make([]float32, 6),
				strides: []int{3, 1}, // Matches canonical
				offset:  0,
			},
			expected:    true,
			description: "Explicit strides that match canonical",
		},
		{
			name:        "empty tensor",
			tensor:      Tensor{},
			expected:    true,
			description: "Empty tensors are considered contiguous",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.tensor.IsContiguous()
			assert.Equal(t, tt.expected, result, tt.description)
		})
	}
}

func TestOffset(t *testing.T) {
	tests := []struct {
		name     string
		tensor   Tensor
		expected int
	}{
		{
			name:     "new tensor (offset 0)",
			tensor:   FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6}),
			expected: 0,
		},
		{
			name: "tensor with offset",
			tensor: Tensor{
				shape:   types.NewShape(2, 3),
				data:    make([]float32, 20),
				strides: nil,
				offset:  5,
			},
			expected: 5,
		},
		{
			name:     "empty tensor",
			tensor:   Tensor{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.tensor.Offset()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDataWithOffset(t *testing.T) {
	tests := []struct {
		name     string
		tensor   Tensor
		expected []float32
	}{
		{
			name:     "tensor with offset 0",
			tensor:   FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6}),
			expected: []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name: "tensor with offset 2",
			tensor: Tensor{
				shape:   types.NewShape(2, 3),
				data:    []float32{0, 0, 1, 2, 3, 4, 5, 6},
				strides: nil,
				offset:  2,
			},
			expected: []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name:     "empty tensor",
			tensor:   Tensor{},
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.tensor.DataWithOffset()
			if tt.expected == nil {
				assert.Nil(t, result)
				return
			}

			data, ok := result.([]float32)
			assert.True(t, ok, "DataWithOffset should return []float32")
			assert.Equal(t, len(tt.expected), len(data))
			for i := range tt.expected {
				assert.Equal(t, tt.expected[i], data[i], "data[%d]", i)
			}
		})
	}
}

func TestDataWithOffset_OffsetOutOfBounds(t *testing.T) {
	tensor := Tensor{
		shape:   types.NewShape(2, 3),
		data:    []float32{1, 2, 3},
		strides: nil,
		offset:  10, // Out of bounds
	}

	result := tensor.DataWithOffset()
	data, ok := result.([]float32)
	assert.True(t, ok)
	assert.Nil(t, data) // Should return nil slice when offset >= len(data)
}

func TestClone_StridesAndOffset(t *testing.T) {
	// Create tensor with explicit strides and offset
	original := Tensor{
		shape:   types.NewShape(2, 3),
		data:    []float32{0, 0, 1, 2, 3, 4, 5, 6},
		strides: []int{6, 1}, // Non-contiguous
		offset:  2,
	}

	cloned := original.Clone().(Tensor)

	// Verify shape and data are cloned
	assert.Equal(t, original.Shape(), cloned.Shape())
	// Data should be different slices (deep copy)
	originalData := original.Data().([]float32)
	clonedData := cloned.Data().([]float32)
	assert.NotEqual(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&clonedData[0])))

	// Verify strides are cloned
	assert.NotNil(t, cloned.strides)
	assert.Equal(t, original.strides, cloned.strides)
	// Strides should be different slices (deep copy)
	assert.NotEqual(t, uintptr(unsafe.Pointer(&original.strides[0])), uintptr(unsafe.Pointer(&cloned.strides[0])))

	// Verify offset is copied
	assert.Equal(t, original.offset, cloned.offset)

	// Verify modifying cloned strides doesn't affect original
	originalStride0 := original.strides[0]
	cloned.strides[0] = 999
	assert.Equal(t, originalStride0, original.strides[0]) // Original unchanged
	assert.Equal(t, 999, cloned.strides[0])               // Cloned changed
}

func TestClone_ContiguousTensor(t *testing.T) {
	// Create contiguous tensor (nil strides)
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	cloned := original.Clone().(Tensor)

	// Verify strides remain nil (contiguous)
	assert.Nil(t, cloned.strides)
	assert.Equal(t, 0, cloned.offset)
}

func TestNew_InitializesStridesAndOffset(t *testing.T) {
	tensor := New(types.FP32, types.NewShape(2, 3))

	// New tensors should have nil strides (contiguous) and offset 0
	assert.Nil(t, tensor.strides)
	assert.Equal(t, 0, tensor.offset)
	assert.True(t, tensor.IsContiguous())
}

func TestFromArray_InitializesStridesAndOffset(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	tensor := FromArray(types.NewShape(2, 3), data)

	// FromArray tensors should have nil strides (contiguous) and offset 0
	assert.Nil(t, tensor.strides)
	assert.Equal(t, 0, tensor.offset)
	assert.True(t, tensor.IsContiguous())
}

func TestStrides_StackAllocation(t *testing.T) {
	// Test that Strides uses stack allocation when dst is provided
	tensor := FromFloat32(types.NewShape(2, 3, 4, 5), make([]float32, 120))

	var dstStatic [helpers.MAX_DIMS]int
	dst := dstStatic[:4]
	strides := tensor.Strides(dst)

	// Should use provided destination (stack-allocated)
	assert.Equal(t, dst, strides[:4])
	assert.Equal(t, []int{60, 20, 5, 1}, strides)
}

func TestAt_WithOffset(t *testing.T) {
	// Create tensor with offset
	data := []float32{0, 0, 1, 2, 3, 4, 5, 6}
	tensor := Tensor{
		shape:   types.NewShape(2, 3),
		data:    data,
		strides: nil,
		offset:  2,
	}

	// Test accessing elements (should account for offset)
	assert.Equal(t, float64(1), tensor.At(0, 0))
	assert.Equal(t, float64(2), tensor.At(0, 1))
	assert.Equal(t, float64(4), tensor.At(1, 0))
	assert.Equal(t, float64(6), tensor.At(1, 2))
}

func TestSetAt_WithOffset(t *testing.T) {
	// Create tensor with offset
	data := []float32{0, 0, 1, 2, 3, 4, 5, 6}
	tensor := Tensor{
		shape:   types.NewShape(2, 3),
		data:    data,
		strides: nil,
		offset:  2,
	}

	// Set element and verify it's written at correct offset
	tensor.SetAt(99.0, 0, 0)
	assert.Equal(t, float32(99), data[2]) // offset + (0*3 + 0) = 2

	tensor.SetAt(88.0, 1, 2)
	assert.Equal(t, float32(88), data[7]) // offset + (1*3 + 2) = 2 + 5 = 7
}

func TestElements_WithOffset(t *testing.T) {
	// Create tensor with offset
	data := []float32{0, 0, 1, 2, 3, 4, 5, 6}
	tensor := Tensor{
		shape:   types.NewShape(2, 3),
		data:    data,
		strides: nil,
		offset:  2,
	}

	// Iterate and verify elements are accessed correctly
	expected := []float32{1, 2, 3, 4, 5, 6}
	var got []float32
	for elem := range tensor.Elements() {
		got = append(got, float32(elem.Get()))
	}

	assert.Equal(t, expected, got)
}
