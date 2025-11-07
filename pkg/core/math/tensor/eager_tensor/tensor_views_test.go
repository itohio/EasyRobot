package eager_tensor

import (
	"testing"
	"unsafe"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestSlice_ZeroCopyView(t *testing.T) {
	// Create original tensor
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	// Create slice view (zero-copy)
	sliced := original.Slice(nil, 0, 1, 1).(Tensor) // Slice first dimension: [1, 1:3] -> [4, 5, 6]

	// Verify it's a view (same backing array)
	originalData := original.Data().([]float32)
	slicedData := sliced.Data().([]float32)
	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&slicedData[0])))

	// Verify offset is adjusted
	assert.Equal(t, 3, sliced.offset) // start=1, stride[0]=3, so offset = 0 + 1*3 = 3

	// Verify strides are preserved (original was contiguous, so strides were computed and stored)
	var origStridesStatic [16]int
	origStrides := original.Strides(origStridesStatic[:2])
	var slicedStridesStatic [16]int
	slicedStrides := sliced.Strides(slicedStridesStatic[:2])
	assert.Equal(t, origStrides, slicedStrides)

	// Verify modifying view modifies original
	// Use SetAt/At methods to account for offset
	originalValue := original.At(1, 0) // Second row, first column
	sliced.SetAt(99.0, 0, 0)
	assert.Equal(t, float64(99), original.At(1, 0), "Modifying view should modify original")
	
	// Restore
	sliced.SetAt(originalValue, 0, 0)
}

func TestSlice_ZeroCopyView_WithOffset(t *testing.T) {
	// Create tensor with offset
	data := []float32{0, 0, 0, 1, 2, 3, 4, 5, 6}
	original := Tensor{
		shape:   types.NewShape(2, 3),
		data:    data,
		strides: nil,
		offset:  3,
	}

	// Create slice view (slice dimension 0, start=1, length=1)
	// Original shape: [2, 3], data starts at offset 3: [1, 2, 3, 4, 5, 6]
	// After slice: shape [1, 3], should point to second row: [4, 5, 6]
	sliced := original.Slice(nil, 0, 1, 1).(Tensor)

	// Verify offset is correctly adjusted
	// Original offset: 3, start: 1, stride[0]: 3
	// New offset: 3 + 1*3 = 6
	assert.Equal(t, 6, sliced.offset)

	// Verify data access is correct
	// sliced.At(0, 0) should access data[6] = 4 (second row, first element)
	assert.Equal(t, float64(4), sliced.At(0, 0))
	assert.Equal(t, float64(5), sliced.At(0, 1))
	assert.Equal(t, float64(6), sliced.At(0, 2))
}

func TestSlice_ZeroCopyView_NonContiguous(t *testing.T) {
	// Create non-contiguous tensor (with explicit strides)
	original := Tensor{
		shape:   types.NewShape(2, 3),
		data:    make([]float32, 12),
		strides: []int{6, 1}, // Non-contiguous: ld = 6
		offset:  0,
	}

	// Initialize data
	for i := range original.Data().([]float32) {
		original.Data().([]float32)[i] = float32(i)
	}

	// Create slice view
	sliced := original.Slice(nil, 0, 1, 1).(Tensor)

	// Verify strides are cloned (not shared)
	assert.NotNil(t, sliced.strides)
	assert.Equal(t, original.strides, sliced.strides)
	assert.NotEqual(t, uintptr(unsafe.Pointer(&original.strides[0])), uintptr(unsafe.Pointer(&sliced.strides[0])))

	// Verify offset is adjusted
	assert.Equal(t, 6, sliced.offset) // start=1, stride[0]=6, so offset = 0 + 1*6 = 6
}

func TestSlice_CopyToDst(t *testing.T) {
	// Create original tensor
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	// Create destination tensor
	dst := New(types.FP32, types.NewShape(1, 3))

	// Slice to dst (should copy)
	sliced := original.Slice(dst, 0, 1, 1).(Tensor)

	// Verify it's a copy (different backing array)
	originalData := original.Data().([]float32)
	slicedData := sliced.Data().([]float32)
	assert.NotEqual(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&slicedData[0])))

	// Verify data is copied correctly
	assert.Equal(t, []float32{4, 5, 6}, slicedData[:3])

	// Verify modifying copy doesn't affect original
	originalValue := originalData[3]
	slicedData[0] = 99.0
	assert.Equal(t, originalValue, originalData[3], "Modifying copy should not modify original")
}

func TestPermute_ZeroCopyView(t *testing.T) {
	// Create original tensor
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	// Create transpose view (zero-copy)
	transposed := original.Permute(nil, []int{1, 0}).(Tensor) // Transpose: [2,3] -> [3,2]

	// Verify it's a view (same backing array)
	originalData := original.Data().([]float32)
	transposedData := transposed.Data().([]float32)
	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&transposedData[0])))

	// Verify offset is preserved
	assert.Equal(t, original.offset, transposed.offset)

	// Verify strides are permuted
	// Original strides: [3, 1] for shape [2, 3]
	// Permuted strides: [1, 3] for shape [3, 2]
	var origStridesStatic [16]int
	origStrides := original.Strides(origStridesStatic[:2])
	var transStridesStatic [16]int
	transStrides := transposed.Strides(transStridesStatic[:2])
	assert.Equal(t, origStrides[1], transStrides[0]) // stride[1] -> stride[0]
	assert.Equal(t, origStrides[0], transStrides[1]) // stride[0] -> stride[1]

	// Verify modifying view modifies original
	originalValue := float64(originalData[0])
	transposed.SetAt(99.0, 0, 0)
	assert.Equal(t, float32(99), originalData[0], "Modifying view should modify original")
	
	// Restore
	transposed.SetAt(originalValue, 0, 0)
}

func TestPermute_ZeroCopyView_ContiguousResult(t *testing.T) {
	// Create a tensor that when permuted results in contiguous layout
	// For example: [1, 2, 3] -> [3, 2, 1] might result in contiguous if original was contiguous
	original := FromFloat32(types.NewShape(1, 2, 3), make([]float32, 6))

	// Permute to same shape (identity permutation) - should remain contiguous
	permuted := original.Permute(nil, []int{0, 1, 2}).(Tensor)

	// Verify strides are nil (contiguous)
	assert.Nil(t, permuted.strides)
	assert.True(t, permuted.IsContiguous())
}

func TestPermute_ZeroCopyView_NonContiguousResult(t *testing.T) {
	// Create original tensor
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	// Transpose creates non-contiguous view
	transposed := original.Permute(nil, []int{1, 0}).(Tensor)

	// Verify strides are stored (non-contiguous)
	assert.NotNil(t, transposed.strides)
	assert.False(t, transposed.IsContiguous())

	// Verify strides are correct
	// Original: shape [2,3], strides [3,1]
	// Transposed: shape [3,2], strides [1,3]
	var transStridesStatic [16]int
	transStrides := transposed.Strides(transStridesStatic[:2])
	assert.Equal(t, []int{1, 3}, transStrides)
}

func TestPermute_CopyToDst(t *testing.T) {
	// Create original tensor
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	// Create destination tensor
	dst := New(types.FP32, types.NewShape(3, 2))

	// Permute to dst (should copy)
	permuted := original.Permute(dst, []int{1, 0}).(Tensor)

	// Verify it's a copy (different backing array)
	originalData := original.Data().([]float32)
	permutedData := permuted.Data().([]float32)
	assert.NotEqual(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&permutedData[0])))

	// Verify data is permuted correctly
	// Original: [[1,2,3], [4,5,6]]
	// Transposed: [[1,4], [2,5], [3,6]]
	assert.Equal(t, float32(1), permutedData[0])
	assert.Equal(t, float32(4), permutedData[1])
	assert.Equal(t, float32(2), permutedData[2])
	assert.Equal(t, float32(5), permutedData[3])
	assert.Equal(t, float32(3), permutedData[4])
	assert.Equal(t, float32(6), permutedData[5])
}

func TestTranspose_ZeroCopyView(t *testing.T) {
	// Transpose uses Permute internally, so test that it works
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	// Transpose (default: swap last two dimensions)
	transposed := original.Transpose(nil, nil).(Tensor)

	// Verify it's a view
	originalData := original.Data().([]float32)
	transposedData := transposed.Data().([]float32)
	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&transposedData[0])))

	// Verify shape is transposed
	assert.Equal(t, types.NewShape(3, 2), transposed.Shape())
}

func TestReshape_ZeroCopyView(t *testing.T) {
	// Reshape was already updated in Phase 1, but verify it works
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	// Reshape to 1D
	reshaped := original.Reshape(nil, types.NewShape(6)).(Tensor)

	// Verify it's a view (same backing array)
	originalData := original.Data().([]float32)
	reshapedData := reshaped.Data().([]float32)
	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&reshapedData[0])))

	// Verify strides and offset are preserved
	assert.Equal(t, original.strides, reshaped.strides)
	assert.Equal(t, original.offset, reshaped.offset)

	// Verify modifying view modifies original
	originalValue := float64(originalData[0])
	reshaped.SetAt(99.0, 0)
	assert.Equal(t, float32(99), originalData[0], "Modifying view should modify original")
	
	// Restore
	reshaped.SetAt(originalValue, 0)
}

func TestNestedViews(t *testing.T) {
	// Test chaining view operations
	original := FromFloat32(types.NewShape(2, 3, 4), make([]float32, 24))

	// Initialize with index values
	for i := range original.Data().([]float32) {
		original.Data().([]float32)[i] = float32(i)
	}

	// Create slice view (slices dimension 0, so result has shape [1, 3, 4] -> rank 3)
	sliced := original.Slice(nil, 0, 1, 1).(Tensor)

	// Create transpose view of slice (swap last two dimensions: [1, 3, 4] -> [1, 4, 3])
	transposed := sliced.Permute(nil, []int{0, 2, 1}).(Tensor)

	// Verify all share same backing array
	originalData := original.Data().([]float32)
	slicedData := sliced.Data().([]float32)
	transposedData := transposed.Data().([]float32)
	
	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&slicedData[0])))
	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&transposedData[0])))

	// Verify modifying nested view modifies original
	// transposed has shape [1, 4, 3], so we need 3 indices
	originalValue := float64(originalData[12]) // Offset for slice
	transposed.SetAt(99.0, 0, 0, 0)
	assert.Equal(t, float32(99), originalData[12], "Modifying nested view should modify original")
	
	// Restore
	transposed.SetAt(originalValue, 0, 0, 0)
}

func TestViewMemorySharing(t *testing.T) {
	// Comprehensive test for memory sharing
	original := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

	// Create multiple views
	view1 := original.Slice(nil, 0, 0, 1).(Tensor)
	view2 := original.Permute(nil, []int{1, 0}).(Tensor)
	view3 := original.Reshape(nil, types.NewShape(6)).(Tensor)

	// All should share same backing array
	originalData := original.Data().([]float32)
	view1Data := view1.Data().([]float32)
	view2Data := view2.Data().([]float32)
	view3Data := view3.Data().([]float32)

	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&view1Data[0])))
	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&view2Data[0])))
	assert.Equal(t, uintptr(unsafe.Pointer(&originalData[0])), uintptr(unsafe.Pointer(&view3Data[0])))

	// Modify through one view
	view1.SetAt(99.0, 0, 0)

	// All should see the change
	assert.Equal(t, float32(99), originalData[0])
	assert.Equal(t, float64(99), view1.At(0, 0))
	assert.Equal(t, float64(99), view2.At(0, 0))
	assert.Equal(t, float64(99), view3.At(0))
}

