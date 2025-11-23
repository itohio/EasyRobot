package iter

import (
	shp "github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/types"
)

// ElemSign writes the sign of src into dst: -1 if src < 0, 0 if src == 0, 1 if src > 0 for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemSign[T types.NumberReal](dst, src []T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := range n {
		v := src[i]
		if v > 0 {
			dst[i] = 1
		} else if v < 0 {
			dst[i] = -1
		} else {
			dst[i] = 0
		}
	}
}

// ElemSignStrided writes the sign of src into dst with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemSignStrided[T types.NumberReal](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
	size := shp.SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [shp.MAX_DIMS]int
	var srcStridesStatic [shp.MAX_DIMS]int
	stridesDst = shp.EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = shp.EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)

	if shp.IsContiguous(stridesDst, shape) && shp.IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			v := src[i]
			if v > 0 {
				dst[i] = 1
			} else if v < 0 {
				dst[i] = -1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	rank := len(shape)
	var indicesStatic [shp.MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		v := src[sIdx]
		if v > 0 {
			dst[dIdx] = 1
		} else if v < 0 {
			dst[dIdx] = -1
		} else {
			dst[dIdx] = 0
		}
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemNegative writes the negation of src into dst: dst[i] = -src[i] for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemNegative[T types.Number](dst, src []T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := range n {
		dst[i] = -src[i]
	}
}

// ElemNegativeStrided writes the negation of src into dst with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemNegativeStrided[T types.Number](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
	size := shp.SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [shp.MAX_DIMS]int
	var srcStridesStatic [shp.MAX_DIMS]int
	stridesDst = shp.EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = shp.EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)

	if shp.IsContiguous(stridesDst, shape) && shp.IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			dst[i] = -src[i]
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	rank := len(shape)
	var indicesStatic [shp.MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = -src[sIdx]
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}
