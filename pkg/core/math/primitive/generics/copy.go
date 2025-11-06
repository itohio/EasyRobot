package generics

// Numeric types that can be used with generic operations
type Numeric interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32 | ~int16 | ~int8
}

// ElemCopy copies src into dst for contiguous arrays.
// This is a simple, optimized version for the common case of contiguous memory.
// For strided or non-contiguous arrays, use ElemCopyStrided.
func ElemCopy[T Numeric](dst, src []T, n int) {
	if n == 0 {
		return
	}
	copy(dst[:n], src[:n])
}

// ElemCopyStrided copies src into dst respecting the supplied shape/strides.
// This function handles both contiguous and strided cases with optimized paths.
func ElemCopyStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays - use direct copy
		copy(dst[:size], src[:size])
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = src[sIdx]
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemSwap swaps elements between dst and src for contiguous arrays.
// This is a simple, optimized version for the common case of contiguous memory.
// For strided or non-contiguous arrays, use ElemSwapStrided.
func ElemSwap[T Numeric](dst, src []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i], src[i] = src[i], dst[i]
	}
}

// ElemSwapStrided swaps elements between dst and src respecting the supplied shape/strides.
// This function handles both contiguous and strided cases with optimized paths.
func ElemSwapStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays - use direct swap
		for i := 0; i < size; i++ {
			dst[i], src[i] = src[i], dst[i]
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
		dst[dIdx], src[sIdx] = src[sIdx], dst[dIdx]
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}
