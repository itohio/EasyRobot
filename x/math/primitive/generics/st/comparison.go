package st

import (
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

// ElemGreaterThan writes 1 where a > b, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemGreaterThan[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = a[n-1]
		_ = b[n-1]
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

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			if a[i] > b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	// Maintain offsets incrementally (like AdvanceOffsets but for 3 arrays)
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [3]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:3]
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] > b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		// Advance offsets incrementally for 3 arrays
		if !AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesA, stridesB) {
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
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = a[n-1]
		_ = b[n-1]
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

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			if a[i] == b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	// Maintain offsets incrementally (like AdvanceOffsets but for 3 arrays)
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [3]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:3]
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] == b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		// Advance offsets incrementally for 3 arrays
		if !AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesA, stridesB) {
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
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = a[n-1]
		_ = b[n-1]
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

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			if a[i] < b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	// Maintain offsets incrementally (like AdvanceOffsets but for 3 arrays)
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [3]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:3]
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] < b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		// Advance offsets incrementally for 3 arrays
		if !AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesA, stridesB) {
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
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = a[n-1]
		_ = b[n-1]
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

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			if a[i] != b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	// Maintain offsets incrementally (like AdvanceOffsets but for 3 arrays)
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [3]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:3]
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] != b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		// Advance offsets incrementally for 3 arrays
		if !AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesA, stridesB) {
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
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = a[n-1]
		_ = b[n-1]
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

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			if a[i] <= b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	// Maintain offsets incrementally (like AdvanceOffsets but for 3 arrays)
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [3]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:3]
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] <= b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		// Advance offsets incrementally for 3 arrays
		if !AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesA, stridesB) {
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
	// Boundary check elimination hint
	if n > 0 {
		_ = dst[n-1]
		_ = a[n-1]
		_ = b[n-1]
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

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			if a[i] >= b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	// Maintain offsets incrementally (like AdvanceOffsets but for 3 arrays)
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [3]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:3]
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		if a[aIdx] >= b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		// Advance offsets incrementally for 3 arrays
		if !AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesA, stridesB) {
			break
		}
	}
}
