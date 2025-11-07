package generics

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// ElemWhere writes elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes/strides.
// condition > 0 means true, otherwise false.
func ElemWhere[T Numeric](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var condStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesCond = EnsureStrides(condStridesStatic[:len(shape)], stridesCond, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesCond, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = condition[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			if condition[i] > 0 {
				dst[i] = a[i]
			} else {
				dst[i] = b[i]
			}
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	// Maintain offsets incrementally (like AdvanceOffsets but for 4 arrays)
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [4]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:4]
	for {
		dIdx := offsets[0]
		cIdx := offsets[1]
		aIdx := offsets[2]
		bIdx := offsets[3]
		if condition[cIdx] > 0 {
			dst[dIdx] = a[aIdx]
		} else {
			dst[dIdx] = b[bIdx]
		}
		// Advance offsets incrementally for 4 arrays
		if !AdvanceOffsets4(shape, indices, offsets, stridesDst, stridesCond, stridesA, stridesB) {
			break
		}
	}
}
