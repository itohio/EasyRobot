package generics

// ElemGreaterThan writes 1 where a > b, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemGreaterThan[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] > b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemGreaterThanStrided writes 1 where a > b, 0 otherwise with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemGreaterThanStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			if a[i] > b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] > b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemEqual writes 1 where a == b, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemEqual[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] == b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemEqualStrided writes 1 where a == b, 0 otherwise with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			if a[i] == b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] == b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemLess writes 1 where a < b, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemLess[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] < b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemLessStrided writes 1 where a < b, 0 otherwise with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemLessStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			if a[i] < b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] < b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemNotEqual writes 1 where a != b, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemNotEqual[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemNotEqualStrided writes 1 where a != b, 0 otherwise with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemNotEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			if a[i] != b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] != b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemLessEqual writes 1 where a <= b, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemLessEqual[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] <= b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemLessEqualStrided writes 1 where a <= b, 0 otherwise with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemLessEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			if a[i] <= b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] <= b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemGreaterEqual writes 1 where a >= b, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemGreaterEqual[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] >= b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemGreaterEqualStrided writes 1 where a >= b, 0 otherwise with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemGreaterEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			if a[i] >= b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] >= b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

