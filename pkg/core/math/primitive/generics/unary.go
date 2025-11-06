package generics

// ElemSign writes the sign of src into dst: -1 if src < 0, 0 if src == 0, 1 if src > 0 for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemSign[T Numeric](dst, src []T, n int) {
	if n == 0 {
		return
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

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays
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

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
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
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
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

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			dst[i] = -src[i]
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = -src[sIdx]
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}
