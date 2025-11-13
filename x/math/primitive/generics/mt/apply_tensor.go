package mt

import (
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
)

// ElemApplyBinary applies a binary function element-wise to contiguous arrays: dst[i] = op(a[i], b[i]).
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyBinary[T Numeric](dst, a, b []T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemApplyBinary(dst, a, b, n, op)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemApplyBinary(dst[start:end], a[start:end], b[start:end], end-start, op)
	})
}

// ElemApplyBinaryStrided applies a binary function element-wise with strides: dst[i] = op(a[i], b[i]).
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyBinaryStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int, op func(T, T) T) {
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
		st.ElemApplyBinaryStrided(dst, a, b, shape, stridesDst, stridesA, stridesB, op)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemApplyBinary(dst[start:end], a[start:end], b[start:end], end-start, op)
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
		st.ElemApplyBinaryStrided(dst[baseOffsetDst:], a[baseOffsetA:], b[baseOffsetB:], chunkShape, stridesDst, stridesA, stridesB, op)
	})
}

// ElemApplyUnary applies a unary function element-wise to contiguous arrays: dst[i] = op(src[i]).
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyUnary[T Numeric](dst, src []T, n int, op func(T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemApplyUnary(dst, src, n, op)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemApplyUnary(dst[start:end], src[start:end], end-start, op)
	})
}

// ElemApplyUnaryStrided applies a unary function element-wise with strides: dst[i] = op(src[i]).
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyUnaryStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int, op func(T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemApplyUnaryStrided(dst, src, shape, stridesDst, stridesSrc, op)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemApplyUnary(dst[start:end], src[start:end], end-start, op)
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
		baseOffsetSrc := startDim0 * stridesSrc[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemApplyUnaryStrided(dst[baseOffsetDst:], src[baseOffsetSrc:], chunkShape, stridesDst, stridesSrc, op)
	})
}

// ElemApplyTernary applies a ternary function element-wise to contiguous arrays: dst[i] = op(condition[i], a[i], b[i]).
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyTernary[T Numeric](dst, condition, a, b []T, n int, op func(T, T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemApplyTernary(dst, condition, a, b, n, op)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemApplyTernary(dst[start:end], condition[start:end], a[start:end], b[start:end], end-start, op)
	})
}

// ElemApplyTernaryStrided applies a ternary function element-wise with strides: dst[i] = op(condition[i], a[i], b[i]).
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyTernaryStrided[T Numeric](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int, op func(T, T, T) T) {
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

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemApplyTernaryStrided(dst, condition, a, b, shape, stridesDst, stridesCond, stridesA, stridesB, op)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesCond, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemApplyTernary(dst[start:end], condition[start:end], a[start:end], b[start:end], end-start, op)
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
		baseOffsetCond := startDim0 * stridesCond[0]
		baseOffsetA := startDim0 * stridesA[0]
		baseOffsetB := startDim0 * stridesB[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemApplyTernaryStrided(dst[baseOffsetDst:], condition[baseOffsetCond:], a[baseOffsetA:], b[baseOffsetB:], chunkShape, stridesDst, stridesCond, stridesA, stridesB, op)
	})
}

// ElemApplyUnaryScalar applies a unary function with a scalar parameter to contiguous arrays: dst[i] = op(src[i], scalar).
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyUnaryScalar[T Numeric](dst, src []T, scalar T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemApplyUnaryScalar(dst, src, scalar, n, op)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemApplyUnaryScalar(dst[start:end], src[start:end], scalar, end-start, op)
	})
}

// ElemApplyUnaryScalarStrided applies a unary function with a scalar parameter with strides: dst[i] = op(src[i], scalar).
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int, op func(T, T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemApplyUnaryScalarStrided(dst, src, scalar, shape, stridesDst, stridesSrc, op)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemApplyUnaryScalar(dst[start:end], src[start:end], scalar, end-start, op)
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
		baseOffsetSrc := startDim0 * stridesSrc[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemApplyUnaryScalarStrided(dst[baseOffsetDst:], src[baseOffsetSrc:], scalar, chunkShape, stridesDst, stridesSrc, op)
	})
}

// ElemApplyBinaryScalar applies a binary function with a scalar parameter to contiguous arrays: dst[i] = op(a[i], scalar).
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyBinaryScalar[T Numeric](dst, a []T, scalar T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemApplyBinaryScalar(dst, a, scalar, n, op)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemApplyBinaryScalar(dst[start:end], a[start:end], scalar, end-start, op)
	})
}

// ElemApplyBinaryScalarStrided applies a binary function with a scalar parameter with strides: dst[i] = op(a[i], scalar).
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, shape []int, stridesDst, stridesA []int, op func(T, T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemApplyBinaryScalarStrided(dst, a, scalar, shape, stridesDst, stridesA, op)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemApplyBinaryScalar(dst[start:end], a[start:end], scalar, end-start, op)
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

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemApplyBinaryScalarStrided(dst[baseOffsetDst:], a[baseOffsetA:], scalar, chunkShape, stridesDst, stridesA, op)
	})
}

// ElemApplyTernaryScalar applies a ternary function with a scalar parameter to contiguous arrays: dst[i] = op(condition[i], a[i], scalar).
// Multi-threaded version that parallelizes the operation across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyTernaryScalar[T Numeric](dst, condition, a []T, scalar T, n int, op func(T, T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemApplyTernaryScalar(dst, condition, a, scalar, n, op)
		return
	}

	// Parallelize using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemApplyTernaryScalar(dst[start:end], condition[start:end], a[start:end], scalar, end-start, op)
	})
}

// ElemApplyTernaryScalarStrided applies a ternary function with a scalar parameter with strides: dst[i] = op(condition[i], a[i], scalar).
// Multi-threaded version that parallelizes strided operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, shape []int, stridesDst, stridesCond, stridesA []int, op func(T, T, T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var condStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesCond = EnsureStrides(condStridesStatic[:len(shape)], stridesCond, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) {
		st.ElemApplyTernaryScalarStrided(dst, condition, a, scalar, shape, stridesDst, stridesCond, stridesA, op)
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesCond, shape) && IsContiguous(stridesA, shape) {
		// Fast path: contiguous arrays - reuse st function for each chunk
		parallelChunks(size, func(start, end int) {
			st.ElemApplyTernaryScalar(dst[start:end], condition[start:end], a[start:end], scalar, end-start, op)
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
		baseOffsetCond := startDim0 * stridesCond[0]
		baseOffsetA := startDim0 * stridesA[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemApplyTernaryScalarStrided(dst[baseOffsetDst:], condition[baseOffsetCond:], a[baseOffsetA:], scalar, chunkShape, stridesDst, stridesCond, stridesA, op)
	})
}
