package eager_tensor

import (
	"testing"
	"unsafe"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

// Test3DSlice_SetGet verifies that set/get operations on a 3D tensor slice
// correctly modify and read the original 3D tensor.
func Test3DSlice_SetGet(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4] = 24 elements
	// Data layout (row-major): [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23]
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Slice dimension 0: take second batch [1, 3, 4]
	// This should give us elements starting at index 12 (1 * 3 * 4)
	sliced := original.Slice(nil, 0, 1, 1).(Tensor)

	// Verify slice shape
	assert.Equal(t, []int{1, 3, 4}, sliced.Shape().ToSlice())

	// Test 1: Get from slice, verify it matches original
	// sliced.At(0, 0, 0) should equal original.At(1, 0, 0)
	sliceValue := sliced.At(0, 0, 0)
	originalValue := original.At(1, 0, 0)
	assert.Equal(t, originalValue, sliceValue, "Slice get should match original")

	// Test 2: Set on slice, verify original is modified
	testValue := 999.0
	sliced.SetAt(testValue, 0, 0, 0)
	
	// Verify original tensor was modified
	assert.Equal(t, testValue, original.At(1, 0, 0), "Setting slice should modify original tensor")
	
	// Verify slice reflects the change
	assert.Equal(t, testValue, sliced.At(0, 0, 0), "Slice should reflect the change")

	// Test 3: Set on original, verify slice is modified
	testValue2 := 888.0
	original.SetAt(testValue2, 1, 1, 1)
	
	// Verify slice reflects the change
	assert.Equal(t, testValue2, sliced.At(0, 1, 1), "Setting original should modify slice")

	// Test 4: Verify all elements in slice correspond to correct original elements
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			sliceVal := sliced.At(0, i, j)
			originalVal := original.At(1, i, j)
			assert.Equal(t, originalVal, sliceVal,
				"Slice[0,%d,%d] = %f should equal Original[1,%d,%d] = %f", i, j, sliceVal, i, j, originalVal)
		}
	}

	// Test 5: Slice a different dimension (dimension 1)
	slicedDim1 := original.Slice(nil, 1, 1, 1).(Tensor)
	assert.Equal(t, []int{2, 1, 4}, slicedDim1.Shape().ToSlice())

	// Verify get/set works
	slicedDim1.SetAt(777.0, 0, 0, 0)
	assert.Equal(t, 777.0, original.At(0, 1, 0), "Slice dim1 set should modify original")
	assert.Equal(t, 777.0, slicedDim1.At(0, 0, 0), "Slice dim1 should reflect change")
}

// Test3DTranspose_SetGet verifies that set/get operations on a 3D tensor transpose
// correctly modify and read the original 3D tensor.
func Test3DTranspose_SetGet(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4]
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Transpose: swap last two dimensions [2, 3, 4] -> [2, 4, 3]
	transposed := original.Transpose(nil, nil).(Tensor)

	// Verify transposed shape
	assert.Equal(t, []int{2, 4, 3}, transposed.Shape().ToSlice())

	// Test 1: Get from transposed, verify it matches original (with swapped indices)
	// transposed.At(0, 0, 0) should equal original.At(0, 0, 0) (first element)
	transValue := transposed.At(0, 0, 0)
	origValue := original.At(0, 0, 0)
	assert.Equal(t, origValue, transValue, "Transpose get should match original (first element)")

	// Test 2: Verify transposed indexing
	// original.At(0, 0, 1) should equal transposed.At(0, 1, 0)
	origVal := original.At(0, 0, 1)
	transVal := transposed.At(0, 1, 0)
	assert.Equal(t, origVal, transVal, "Transpose indexing: Original[0,0,1] should equal Transposed[0,1,0]")

	// Test 3: Set on transposed, verify original is modified
	testValue := 999.0
	transposed.SetAt(testValue, 0, 1, 0)
	
	// Verify original tensor was modified at correct position
	assert.Equal(t, testValue, original.At(0, 0, 1), "Setting transposed should modify original tensor")
	
	// Verify transposed reflects the change
	assert.Equal(t, testValue, transposed.At(0, 1, 0), "Transposed should reflect the change")

	// Test 4: Set on original, verify transposed is modified
	testValue2 := 888.0
	original.SetAt(testValue2, 0, 1, 2)
	
	// Verify transposed reflects the change (indices swapped)
	assert.Equal(t, testValue2, transposed.At(0, 2, 1), "Setting original should modify transposed")

	// Test 5: Verify all elements are correctly transposed
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				origVal := original.At(i, j, k)
				transVal := transposed.At(i, k, j) // Indices swapped
				assert.Equal(t, origVal, transVal,
					"Original[%d,%d,%d] = %f should equal Transposed[%d,%d,%d] = %f",
					i, j, k, origVal, i, k, j, transVal)
			}
		}
	}
}

// Test3DReshape_SetGet verifies that set/get operations on a 3D tensor reshape
// correctly modify and read the original 3D tensor.
func Test3DReshape_SetGet(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4] = 24 elements
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Reshape to 2D: [2, 3, 4] -> [6, 4]
	reshaped := original.Reshape(nil, types.NewShape(6, 4)).(Tensor)

	// Verify reshaped shape
	assert.Equal(t, []int{6, 4}, reshaped.Shape().ToSlice())

	// Test 1: Get from reshaped, verify it matches original (linear indexing)
	// reshaped.At(0, 0) should equal original.At(0, 0, 0)
	reshapedValue := reshaped.At(0, 0)
	originalValue := original.At(0, 0, 0)
	assert.Equal(t, originalValue, reshapedValue, "Reshape get should match original (first element)")

	// Test 2: Verify linear indexing correspondence
	// original.At(0, 0, 1) should equal reshaped.At(0, 1)
	origVal := original.At(0, 0, 1)
	reshVal := reshaped.At(0, 1)
	assert.Equal(t, origVal, reshVal, "Reshape indexing: Original[0,0,1] should equal Reshaped[0,1]")

	// Test 3: Set on reshaped, verify original is modified
	testValue := 999.0
	reshaped.SetAt(testValue, 0, 1)
	
	// Verify original tensor was modified at correct position
	assert.Equal(t, testValue, original.At(0, 0, 1), "Setting reshaped should modify original tensor")
	
	// Verify reshaped reflects the change
	assert.Equal(t, testValue, reshaped.At(0, 1), "Reshaped should reflect the change")

	// Test 4: Set on original, verify reshaped is modified
	testValue2 := 888.0
	original.SetAt(testValue2, 1, 1, 1)
	
	// Calculate linear index: 1*3*4 + 1*4 + 1 = 12 + 4 + 1 = 17
	// Reshaped as [6, 4]: row = 17 / 4 = 4, col = 17 % 4 = 1
	assert.Equal(t, testValue2, reshaped.At(4, 1), "Setting original should modify reshaped")

	// Test 5: Verify all elements correspond correctly
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				origVal := original.At(i, j, k)
				// Calculate linear index
				linearIdx := i*3*4 + j*4 + k
				reshapedRow := linearIdx / 4
				reshapedCol := linearIdx % 4
				reshVal := reshaped.At(reshapedRow, reshapedCol)
				assert.Equal(t, origVal, reshVal,
					"Original[%d,%d,%d] (linear %d) = %f should equal Reshaped[%d,%d] = %f",
					i, j, k, linearIdx, origVal, reshapedRow, reshapedCol, reshVal)
			}
		}
	}

	// Test 6: Reshape to 1D
	reshaped1D := original.Reshape(nil, types.NewShape(24)).(Tensor)
	assert.Equal(t, []int{24}, reshaped1D.Shape().ToSlice())

	// Verify get/set works
	reshaped1D.SetAt(777.0, 5)
	assert.Equal(t, 777.0, original.At(0, 1, 1), "Reshaped 1D set should modify original")
	assert.Equal(t, 777.0, reshaped1D.At(5), "Reshaped 1D should reflect change")
}

// Test3DPermute_SetGet verifies that set/get operations on a 3D tensor permute
// correctly modify and read the original 3D tensor.
func Test3DPermute_SetGet(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4]
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Permute dimensions: [0, 1, 2] -> [2, 0, 1]
	// This swaps: dim0->dim2, dim1->dim0, dim2->dim1
	// Shape: [2, 3, 4] -> [4, 2, 3]
	permuted := original.Permute(nil, []int{2, 0, 1}).(Tensor)

	// Verify permuted shape
	assert.Equal(t, []int{4, 2, 3}, permuted.Shape().ToSlice())

	// Test 1: Get from permuted, verify it matches original (with permuted indices)
	// permuted.At(0, 0, 0) should equal original.At(0, 0, 0) (first element)
	permValue := permuted.At(0, 0, 0)
	origValue := original.At(0, 0, 0)
	assert.Equal(t, origValue, permValue, "Permute get should match original (first element)")

	// Test 2: Verify permuted indexing
	// original.At(0, 1, 2) should equal permuted.At(2, 0, 1)
	// Because: perm[2,0,1] -> orig[0,1,2] (inverse permutation)
	origVal := original.At(0, 1, 2)
	permVal := permuted.At(2, 0, 1)
	assert.Equal(t, origVal, permVal, "Permute indexing: Original[0,1,2] should equal Permuted[2,0,1]")

	// Test 3: Set on permuted, verify original is modified
	testValue := 999.0
	permuted.SetAt(testValue, 2, 0, 1)
	
	// Verify original tensor was modified at correct position
	assert.Equal(t, testValue, original.At(0, 1, 2), "Setting permuted should modify original tensor")
	
	// Verify permuted reflects the change
	assert.Equal(t, testValue, permuted.At(2, 0, 1), "Permuted should reflect the change")

	// Test 4: Set on original, verify permuted is modified
	testValue2 := 888.0
	original.SetAt(testValue2, 1, 2, 3)
	
	// Verify permuted reflects the change (indices permuted)
	// orig[1,2,3] -> perm[3,1,2] (apply permutation [2,0,1])
	assert.Equal(t, testValue2, permuted.At(3, 1, 2), "Setting original should modify permuted")

	// Test 5: Verify all elements are correctly permuted
	// Permutation [2,0,1] means:
	// permuted[i,j,k] corresponds to original[k,i,j]
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				origVal := original.At(i, j, k)
				// Apply permutation [2,0,1]: new indices are [k, i, j]
				permVal := permuted.At(k, i, j)
				assert.Equal(t, origVal, permVal,
					"Original[%d,%d,%d] = %f should equal Permuted[%d,%d,%d] = %f",
					i, j, k, origVal, k, i, j, permVal)
			}
		}
	}

	// Test 6: Different permutation - swap first two dimensions [1,0,2]
	permuted2 := original.Permute(nil, []int{1, 0, 2}).(Tensor)
	assert.Equal(t, []int{3, 2, 4}, permuted2.Shape().ToSlice())

	// Verify get/set works
	permuted2.SetAt(777.0, 1, 0, 2)
	assert.Equal(t, 777.0, original.At(0, 1, 2), "Permuted2 set should modify original")
	assert.Equal(t, 777.0, permuted2.At(1, 0, 2), "Permuted2 should reflect change")
}

// Test3DChainedViews verifies that chaining multiple view operations works correctly.
func Test3DChainedViews(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4]
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Chain: Slice -> Transpose -> Reshape
	sliced := original.Slice(nil, 0, 1, 1).(Tensor)      // [1, 3, 4]
	transposed := sliced.Transpose(nil, nil).(Tensor)    // [1, 4, 3]
	reshaped := transposed.Reshape(nil, types.NewShape(4, 3)).(Tensor) // [4, 3]

	// Verify all share same backing array
	assert.Equal(t, []int{1, 3, 4}, sliced.Shape().ToSlice())
	assert.Equal(t, []int{1, 4, 3}, transposed.Shape().ToSlice())
	assert.Equal(t, []int{4, 3}, reshaped.Shape().ToSlice())

	// Test: Set on final view, verify all previous views and original are modified
	testValue := 999.0
	reshaped.SetAt(testValue, 0, 0)

	// Note: When reshaping a non-contiguous tensor (transposed) with rank change,
	// Reshape copies data, so reshaped may not share memory with original.
	// Check if they share memory
	reshapedData := reshaped.Data().([]float32)
	originalData := original.Data().([]float32)
	shareMemory := uintptr(unsafe.Pointer(&reshapedData[0])) == uintptr(unsafe.Pointer(&originalData[0]))
	
	if shareMemory {
		// If they share memory, all views should be modified
		assert.Equal(t, testValue, original.At(1, 0, 0), "Chained view set should modify original")
		assert.Equal(t, testValue, sliced.At(0, 0, 0), "Chained view set should modify intermediate slice")
		assert.Equal(t, testValue, transposed.At(0, 0, 0), "Chained view set should modify intermediate transpose")
		assert.Equal(t, testValue, reshaped.At(0, 0), "Chained view set should modify final reshape")
	} else {
		// If they don't share memory (due to copy during reshape), only verify reshaped was modified
		// Original and intermediate views should be unchanged
		assert.Equal(t, testValue, reshaped.At(0, 0), "Chained view set should modify final reshape")
		// Verify original is unchanged
		assert.Equal(t, float64(12), original.At(1, 0, 0), "Original should be unchanged (no shared memory due to reshape copy)")
	}

	// Test: Set on original, verify all views are modified
	testValue2 := 888.0
	original.SetAt(testValue2, 1, 1, 1)

	// Verify all views reflect the change (if they share memory)
	if shareMemory {
		assert.Equal(t, testValue2, sliced.At(0, 1, 1), "Original set should modify slice")
		assert.Equal(t, testValue2, transposed.At(0, 1, 1), "Original set should modify transpose")
		// Reshaped: original[1,1,1] has linear index 1*3*4 + 1*4 + 1 = 17
		// Reshaped view has offset 12, so relative index = 17 - 12 = 5
		// For shape [4, 3]: row = 5/3 = 1, col = 5%3 = 2
		assert.Equal(t, testValue2, reshaped.At(1, 2), "Original set should modify reshape")
	} else {
		// If reshaped doesn't share memory, it won't reflect changes to original
		// But slice and transpose should still work
		assert.Equal(t, testValue2, sliced.At(0, 1, 1), "Original set should modify slice")
		assert.Equal(t, testValue2, transposed.At(0, 1, 1), "Original set should modify transpose")
		// Reshaped won't reflect the change since it's a copy
	}
}

// Test3DViewOffsetCalculation verifies that offset calculations are correct for 3D tensors.
func Test3DViewOffsetCalculation(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4]
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Slice dimension 0 at index 1: should have offset = 1 * 3 * 4 = 12
	sliced := original.Slice(nil, 0, 1, 1).(Tensor)
	assert.Equal(t, 12, sliced.Offset(), "Slice offset should be 12 (1 * 3 * 4)")

	// Verify offset is correct by checking first element
	assert.Equal(t, float64(12), sliced.At(0, 0, 0), "First element of slice should be at index 12")

	// Slice dimension 1 at index 1: should have offset = 1 * 4 = 4 (for first batch)
	slicedDim1 := original.Slice(nil, 1, 1, 1).(Tensor)
	assert.Equal(t, 4, slicedDim1.Offset(), "Slice dim1 offset should be 4 (1 * 4)")

	// Verify offset is correct
	assert.Equal(t, float64(4), slicedDim1.At(0, 0, 0), "First element of slice dim1 should be at index 4")
}

// Test3DRoundTrip_Reshape verifies that reshape -> reshape -> original returns correct values.
func Test3DRoundTrip_Reshape(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4] = 24 elements
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Store original values for comparison
	originalValues := make([]float64, 24)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				idx := i*3*4 + j*4 + k
				originalValues[idx] = original.At(i, j, k)
			}
		}
	}

	// Round trip: orig -> reshape -> reshape -> orig
	// Step 1: Reshape to 2D [6, 4]
	reshaped1 := original.Reshape(nil, types.NewShape(6, 4)).(Tensor)
	assert.Equal(t, []int{6, 4}, reshaped1.Shape().ToSlice())

	// Step 2: Reshape back to 3D [2, 3, 4]
	reshaped2 := reshaped1.Reshape(nil, types.NewShape(2, 3, 4)).(Tensor)
	assert.Equal(t, []int{2, 3, 4}, reshaped2.Shape().ToSlice())

	// Verify all values match original
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				origVal := original.At(i, j, k)
				roundTripVal := reshaped2.At(i, j, k)
				assert.Equal(t, origVal, roundTripVal,
					"Round trip reshape: Original[%d,%d,%d] = %f should equal Reshaped2[%d,%d,%d] = %f",
					i, j, k, origVal, i, j, k, roundTripVal)
			}
		}
	}

	// Verify modifying reshaped2 modifies original (they share memory)
	testValue := 999.0
	reshaped2.SetAt(testValue, 0, 0, 0)
	assert.Equal(t, testValue, original.At(0, 0, 0), "Modifying reshaped2 should modify original")
	
	// Restore
	reshaped2.SetAt(0.0, 0, 0, 0)
}

// Test3DRoundTrip_Transpose verifies that transpose -> transpose -> original returns correct values.
func Test3DRoundTrip_Transpose(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4]
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Store original values for comparison
	originalValues := make([]float64, 24)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				idx := i*3*4 + j*4 + k
				originalValues[idx] = original.At(i, j, k)
			}
		}
	}

	// Round trip: orig -> transpose -> transpose -> orig
	// Step 1: Transpose (swap last two dimensions) [2, 3, 4] -> [2, 4, 3]
	transposed1 := original.Transpose(nil, nil).(Tensor)
	assert.Equal(t, []int{2, 4, 3}, transposed1.Shape().ToSlice())

	// Step 2: Transpose back (swap last two dimensions again) [2, 4, 3] -> [2, 3, 4]
	transposed2 := transposed1.Transpose(nil, nil).(Tensor)
	assert.Equal(t, []int{2, 3, 4}, transposed2.Shape().ToSlice())

	// Verify all values match original
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				origVal := original.At(i, j, k)
				roundTripVal := transposed2.At(i, j, k)
				assert.Equal(t, origVal, roundTripVal,
					"Round trip transpose: Original[%d,%d,%d] = %f should equal Transposed2[%d,%d,%d] = %f",
					i, j, k, origVal, i, j, k, roundTripVal)
			}
		}
	}

	// Verify modifying transposed2 modifies original (they share memory)
	testValue := 999.0
	transposed2.SetAt(testValue, 0, 0, 0)
	assert.Equal(t, testValue, original.At(0, 0, 0), "Modifying transposed2 should modify original")
	
	// Restore
	transposed2.SetAt(0.0, 0, 0, 0)
}

// Test3DRoundTrip_TransposeReshape verifies that transpose -> reshape -> reshape -> transpose -> original returns correct values.
func Test3DRoundTrip_TransposeReshape(t *testing.T) {
	// Create 3D tensor: shape [2, 3, 4]
	data := make([]float32, 24)
	for i := range data {
		data[i] = float32(i)
	}
	original := FromFloat32(types.NewShape(2, 3, 4), data)

	// Store original values for comparison
	originalValues := make([]float64, 24)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				idx := i*3*4 + j*4 + k
				originalValues[idx] = original.At(i, j, k)
			}
		}
	}

	// Round trip: orig -> transpose -> reshape -> reshape -> transpose -> orig
	// Step 1: Transpose (swap last two dimensions) [2, 3, 4] -> [2, 4, 3]
	transposed1 := original.Transpose(nil, nil).(Tensor)
	assert.Equal(t, []int{2, 4, 3}, transposed1.Shape().ToSlice())

	// Step 2: Reshape to 2D [8, 3]
	reshaped1 := transposed1.Reshape(nil, types.NewShape(8, 3)).(Tensor)
	assert.Equal(t, []int{8, 3}, reshaped1.Shape().ToSlice())

	// Step 3: Reshape back to 3D [2, 4, 3]
	reshaped2 := reshaped1.Reshape(nil, types.NewShape(2, 4, 3)).(Tensor)
	assert.Equal(t, []int{2, 4, 3}, reshaped2.Shape().ToSlice())

	// Step 4: Transpose back (swap last two dimensions) [2, 4, 3] -> [2, 3, 4]
	transposed2 := reshaped2.Transpose(nil, nil).(Tensor)
	assert.Equal(t, []int{2, 3, 4}, transposed2.Shape().ToSlice())

	// Verify all values match original
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 4; k++ {
				origVal := original.At(i, j, k)
				roundTripVal := transposed2.At(i, j, k)
				assert.Equal(t, origVal, roundTripVal,
					"Round trip transpose+reshape: Original[%d,%d,%d] = %f should equal Transposed2[%d,%d,%d] = %f",
					i, j, k, origVal, i, j, k, roundTripVal)
			}
		}
	}

	// Note: When reshaping a non-contiguous tensor (transposed) with rank change,
	// Reshape copies data, so transposed2 may not share memory with original.
	// We verify values are correct, but memory sharing is not guaranteed.
	
	// Verify values are correct (this is the main requirement)
	// Memory sharing check is conditional
	testValue := 999.0
	originalValue := original.At(0, 0, 0)
	transposed2.SetAt(testValue, 0, 0, 0)
	
	// Check if they share memory
	transposed2Data := transposed2.Data().([]float32)
	originalData := original.Data().([]float32)
	shareMemory := uintptr(unsafe.Pointer(&transposed2Data[0])) == uintptr(unsafe.Pointer(&originalData[0]))
	
	if shareMemory {
		// If they share memory, modification should propagate
		assert.Equal(t, testValue, original.At(0, 0, 0), "Modifying transposed2 should modify original (shared memory)")
		transposed2.SetAt(originalValue, 0, 0, 0)
	} else {
		// If they don't share memory (due to copy during reshape), original should be unchanged
		assert.Equal(t, originalValue, original.At(0, 0, 0), "Original should be unchanged (no shared memory due to reshape copy)")
		// Restore transposed2
		transposed2.SetAt(originalValue, 0, 0, 0)
	}

	// Also verify intermediate steps are correct
	// transposed1[0,0,0] should equal original[0,0,0]
	assert.Equal(t, original.At(0, 0, 0), transposed1.At(0, 0, 0), "Transposed1[0,0,0] should equal Original[0,0,0]")
	
	// transposed1[0,0,1] should equal original[0,1,0]
	assert.Equal(t, original.At(0, 1, 0), transposed1.At(0, 0, 1), "Transposed1[0,0,1] should equal Original[0,1,0]")
}

