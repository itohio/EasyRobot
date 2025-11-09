package mt

import (
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
)

// ElemGreaterThan writes 1 where a > b, 0 otherwise for contiguous arrays.
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemGreaterThan[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemGreaterThan(dst, a, b, n)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemGreaterThan(dst[start:end], a[start:end], b[start:end], end-start)
	})
}

// ElemGreaterThanStrided writes 1 where a > b, 0 otherwise with strides.
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
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

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemGreaterThanStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemGreaterThan(dst[start:end], a[start:end], b[start:end], end-start)
		})
		return
	}

	// Parallelize along first dimension - reuse st function for each chunk
	parallelTensorChunks(shape, func(startDim0, endDim0 int) {
		// Create adjusted shape for this chunk using stack-allocated array
		var chunkShapeStatic [MAX_DIMS]int
		chunkShape := chunkShapeStatic[:len(shape)]
		copy(chunkShape, shape)
		chunkShape[0] = endDim0 - startDim0

		// Calculate base offsets for this chunk
		baseOffsetDst := startDim0 * stridesDst[0]
		baseOffsetA := startDim0 * stridesA[0]
		baseOffsetB := startDim0 * stridesB[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemGreaterThanStrided(dst[baseOffsetDst:], a[baseOffsetA:], b[baseOffsetB:], chunkShape, stridesDst, stridesA, stridesB)
	})
}

// ElemEqual writes 1 where a == b, 0 otherwise for contiguous arrays.
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemEqual[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemEqual(dst, a, b, n)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemEqual(dst[start:end], a[start:end], b[start:end], end-start)
	})
}

// ElemEqualStrided writes 1 where a == b, 0 otherwise with strides.
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
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

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemEqual(dst[start:end], a[start:end], b[start:end], end-start)
		})
		return
	}

	// Parallelize along first dimension - reuse st function for each chunk
	parallelTensorChunks(shape, func(startDim0, endDim0 int) {
		// Create adjusted shape for this chunk using stack-allocated array
		var chunkShapeStatic [MAX_DIMS]int
		chunkShape := chunkShapeStatic[:len(shape)]
		copy(chunkShape, shape)
		chunkShape[0] = endDim0 - startDim0

		// Calculate base offsets for this chunk
		baseOffsetDst := startDim0 * stridesDst[0]
		baseOffsetA := startDim0 * stridesA[0]
		baseOffsetB := startDim0 * stridesB[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemEqualStrided(dst[baseOffsetDst:], a[baseOffsetA:], b[baseOffsetB:], chunkShape, stridesDst, stridesA, stridesB)
	})
}

// ElemLess writes 1 where a < b, 0 otherwise for contiguous arrays.
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemLess[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemLess(dst, a, b, n)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemLess(dst[start:end], a[start:end], b[start:end], end-start)
	})
}

// ElemLessStrided writes 1 where a < b, 0 otherwise with strides.
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
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

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemLessStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemLess(dst[start:end], a[start:end], b[start:end], end-start)
		})
		return
	}

	// Parallelize along first dimension - reuse st function for each chunk
	parallelTensorChunks(shape, func(startDim0, endDim0 int) {
		// Create adjusted shape for this chunk using stack-allocated array
		var chunkShapeStatic [MAX_DIMS]int
		chunkShape := chunkShapeStatic[:len(shape)]
		copy(chunkShape, shape)
		chunkShape[0] = endDim0 - startDim0

		// Calculate base offsets for this chunk
		baseOffsetDst := startDim0 * stridesDst[0]
		baseOffsetA := startDim0 * stridesA[0]
		baseOffsetB := startDim0 * stridesB[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemLessStrided(dst[baseOffsetDst:], a[baseOffsetA:], b[baseOffsetB:], chunkShape, stridesDst, stridesA, stridesB)
	})
}

// ElemNotEqual writes 1 where a != b, 0 otherwise for contiguous arrays.
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemNotEqual[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemNotEqual(dst, a, b, n)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemNotEqual(dst[start:end], a[start:end], b[start:end], end-start)
	})
}

// ElemNotEqualStrided writes 1 where a != b, 0 otherwise with strides.
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
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

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemNotEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemNotEqual(dst[start:end], a[start:end], b[start:end], end-start)
		})
		return
	}

	// Parallelize along first dimension - reuse st function for each chunk
	parallelTensorChunks(shape, func(startDim0, endDim0 int) {
		// Create adjusted shape for this chunk using stack-allocated array
		var chunkShapeStatic [MAX_DIMS]int
		chunkShape := chunkShapeStatic[:len(shape)]
		copy(chunkShape, shape)
		chunkShape[0] = endDim0 - startDim0

		// Calculate base offsets for this chunk
		baseOffsetDst := startDim0 * stridesDst[0]
		baseOffsetA := startDim0 * stridesA[0]
		baseOffsetB := startDim0 * stridesB[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemNotEqualStrided(dst[baseOffsetDst:], a[baseOffsetA:], b[baseOffsetB:], chunkShape, stridesDst, stridesA, stridesB)
	})
}

// ElemLessEqual writes 1 where a <= b, 0 otherwise for contiguous arrays.
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemLessEqual[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemLessEqual(dst, a, b, n)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemLessEqual(dst[start:end], a[start:end], b[start:end], end-start)
	})
}

// ElemLessEqualStrided writes 1 where a <= b, 0 otherwise with strides.
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
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

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemLessEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemLessEqual(dst[start:end], a[start:end], b[start:end], end-start)
		})
		return
	}

	// Parallelize along first dimension - reuse st function for each chunk
	parallelTensorChunks(shape, func(startDim0, endDim0 int) {
		// Create adjusted shape for this chunk using stack-allocated array
		var chunkShapeStatic [MAX_DIMS]int
		chunkShape := chunkShapeStatic[:len(shape)]
		copy(chunkShape, shape)
		chunkShape[0] = endDim0 - startDim0

		// Calculate base offsets for this chunk
		baseOffsetDst := startDim0 * stridesDst[0]
		baseOffsetA := startDim0 * stridesA[0]
		baseOffsetB := startDim0 * stridesB[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemLessEqualStrided(dst[baseOffsetDst:], a[baseOffsetA:], b[baseOffsetB:], chunkShape, stridesDst, stridesA, stridesB)
	})
}

// ElemGreaterEqual writes 1 where a >= b, 0 otherwise for contiguous arrays.
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemGreaterEqual[T Numeric](dst, a, b []T, n int) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemGreaterEqual(dst, a, b, n)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemGreaterEqual(dst[start:end], a[start:end], b[start:end], end-start)
	})
}

// ElemGreaterEqualStrided writes 1 where a >= b, 0 otherwise with strides.
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
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

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemGreaterEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemGreaterEqual(dst[start:end], a[start:end], b[start:end], end-start)
		})
		return
	}

	// Parallelize along first dimension - reuse st function for each chunk
	parallelTensorChunks(shape, func(startDim0, endDim0 int) {
		// Create adjusted shape for this chunk using stack-allocated array
		var chunkShapeStatic [MAX_DIMS]int
		chunkShape := chunkShapeStatic[:len(shape)]
		copy(chunkShape, shape)
		chunkShape[0] = endDim0 - startDim0

		// Calculate base offsets for this chunk
		baseOffsetDst := startDim0 * stridesDst[0]
		baseOffsetA := startDim0 * stridesA[0]
		baseOffsetB := startDim0 * stridesB[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemGreaterEqualStrided(dst[baseOffsetDst:], a[baseOffsetA:], b[baseOffsetB:], chunkShape, stridesDst, stridesA, stridesB)
	})
}
