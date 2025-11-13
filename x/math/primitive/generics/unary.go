package generics

// ElemSign writes the sign of src into dst: -1 if src < 0, 0 if src == 0, 1 if src > 0 for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemSign[T Numeric](dst, src []T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := 0; i < n; i++ {
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
func ElemSignStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
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
	var indicesStatic [MAX_DIMS]int
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
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemNegative writes the negation of src into dst: dst[i] = -src[i] for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemNegative[T Numeric](dst, src []T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := 0; i < n; i++ {
		dst[i] = -src[i]
	}
}

// ElemNegativeStrided writes the negation of src into dst with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemNegativeStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
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
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = -src[sIdx]
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}
