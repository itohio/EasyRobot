package iter

import (
	shp "github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/types"
)

// ElemFill writes constant value to dst for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemFill[T types.Number](dst []T, value T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
	}
	for i := range n {
		dst[i] = value
	}
}

// ElemFillStrided writes constant value to dst with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemFillStrided[T types.Number](dst []T, value T, shape []int, stridesDst []int) {
	size := shp.SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [shp.MAX_DIMS]int
	stridesDst = shp.EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)

	if shp.IsContiguous(stridesDst, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
		}
		for i := 0; i < size; i++ {
			dst[i] = value
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	// Use shp.AdvanceOffsets pattern with same stride for both (we only use offsets[0])
	rank := len(shape)
	var indicesStatic [shp.MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dst[offsets[0]] = value
		// Use shp.AdvanceOffsets with same stride for both (we only use offsets[0])
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesDst) {
			break
		}
	}
}

// ElemEqualScalar writes 1 where src == scalar, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemEqualScalar[T types.Number](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := range n {
		if src[i] == scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemEqualScalarStrided writes 1 where src == scalar, 0 otherwise with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemEqualScalarStrided[T types.Number](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
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
			if src[i] == scalar {
				dst[i] = 1
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
		if src[sIdx] == scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemGreaterScalar writes 1 where src > scalar, 0 otherwise for contiguous arrays.
func ElemGreaterScalar[T types.NumberReal](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := range n {
		if src[i] > scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemGreaterScalarStrided writes 1 where src > scalar, 0 otherwise with strides.
func ElemGreaterScalarStrided[T types.NumberReal](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
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
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			if src[i] > scalar {
				dst[i] = 1
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
		if src[sIdx] > scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemLessScalar writes 1 where src < scalar, 0 otherwise for contiguous arrays.
func ElemLessScalar[T types.NumberReal](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := range n {
		if src[i] < scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemLessScalarStrided writes 1 where src < scalar, 0 otherwise with strides.
func ElemLessScalarStrided[T types.NumberReal](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
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
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			if src[i] < scalar {
				dst[i] = 1
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
		if src[sIdx] < scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemNotEqualScalar writes 1 where src != scalar, 0 otherwise for contiguous arrays.
func ElemNotEqualScalar[T types.NumberReal](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := range n {
		if src[i] != scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemNotEqualScalarStrided writes 1 where src != scalar, 0 otherwise with strides.
func ElemNotEqualScalarStrided[T types.NumberReal](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
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
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			if src[i] != scalar {
				dst[i] = 1
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
		if src[sIdx] != scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemLessEqualScalar writes 1 where src <= scalar, 0 otherwise for contiguous arrays.
func ElemLessEqualScalar[T types.NumberReal](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := range n {
		if src[i] <= scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemLessEqualScalarStrided writes 1 where src <= scalar, 0 otherwise with strides.
func ElemLessEqualScalarStrided[T types.NumberReal](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
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
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			if src[i] <= scalar {
				dst[i] = 1
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
		if src[sIdx] <= scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemGreaterEqualScalar writes 1 where src >= scalar, 0 otherwise for contiguous arrays.
func ElemGreaterEqualScalar[T types.NumberReal](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := range n {
		if src[i] >= scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemGreaterEqualScalarStrided writes 1 where src >= scalar, 0 otherwise with strides.
func ElemGreaterEqualScalarStrided[T types.NumberReal](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
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
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			if src[i] >= scalar {
				dst[i] = 1
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
		if src[sIdx] >= scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !shp.AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}
