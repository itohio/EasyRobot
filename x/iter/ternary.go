package iter

import (
	shp "github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/types"
)

// ElemWhere writes elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes/strides.
// condition > 0 means true, otherwise false.
func ElemWhere[T types.NumberReal](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int) {
	size := shp.SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [shp.MAX_DIMS]int
	var condStridesStatic [shp.MAX_DIMS]int
	var aStridesStatic [shp.MAX_DIMS]int
	var bStridesStatic [shp.MAX_DIMS]int
	stridesDst = shp.EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesCond = shp.EnsureStrides(condStridesStatic[:len(shape)], stridesCond, shape)
	stridesA = shp.EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = shp.EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if shp.IsContiguous(stridesDst, shape) && shp.IsContiguous(stridesCond, shape) && shp.IsContiguous(stridesA, shape) && shp.IsContiguous(stridesB, shape) {
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
	var indicesStatic [shp.MAX_DIMS]int
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
		if !shp.AdvanceOffsets4(shape, indices, offsets, stridesDst, stridesCond, stridesA, stridesB) {
			break
		}
	}
}
