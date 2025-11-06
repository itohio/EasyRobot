package mt

import (
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
)

// ElemVecApplyUnaryStrided applies a unary function to a vector: dst[i] = op(src[i]).
// Multi-threaded version that parallelizes vector operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemVecApplyUnaryStrided[T Numeric](dst, src []T, n int, strideDst, strideSrc int, op func(T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemVecApplyUnaryStrided(dst, src, n, strideDst, strideSrc, op)
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors - reuse st function for each chunk
		parallelChunks(n, func(start, end int) {
			st.ElemVecApplyUnaryStrided(dst[start:end], src[start:end], end-start, 1, 1, op)
		})
		return
	}

	// Strided path - parallelize by splitting the range
	parallelIteratorChunks(n, func(start, end int) {
		// Calculate starting indices for this chunk
		dIdx := start * strideDst
		sIdx := start * strideSrc
		chunkSize := end - start

		// Reuse st function for this chunk
		st.ElemVecApplyUnaryStrided(dst[dIdx:], src[sIdx:], chunkSize, strideDst, strideSrc, op)
	})
}

// ElemVecApplyBinaryStrided applies a binary function to vectors: dst[i] = op(a[i], b[i]).
// Multi-threaded version that parallelizes vector operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemVecApplyBinaryStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int, op func(T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemVecApplyBinaryStrided(dst, a, b, n, strideDst, strideA, strideB, op)
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		// Fast path: contiguous vectors - reuse st function for each chunk
		parallelChunks(n, func(start, end int) {
			st.ElemVecApplyBinaryStrided(dst[start:end], a[start:end], b[start:end], end-start, 1, 1, 1, op)
		})
		return
	}

	// Strided path - parallelize by splitting the range
	parallelIteratorChunks(n, func(start, end int) {
		// Calculate starting indices for this chunk
		dIdx := start * strideDst
		aIdx := start * strideA
		bIdx := start * strideB
		chunkSize := end - start

		// Reuse st function for this chunk
		st.ElemVecApplyBinaryStrided(dst[dIdx:], a[aIdx:], b[bIdx:], chunkSize, strideDst, strideA, strideB, op)
	})
}

// ElemVecApplyTernaryStrided applies a ternary function to vectors: dst[i] = op(condition[i], a[i], b[i]).
// Multi-threaded version that parallelizes vector operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemVecApplyTernaryStrided[T Numeric](dst, condition, a, b []T, n int, strideDst, strideCond, strideA, strideB int, op func(T, T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemVecApplyTernaryStrided(dst, condition, a, b, n, strideDst, strideCond, strideA, strideB, op)
		return
	}

	if strideDst == 1 && strideCond == 1 && strideA == 1 && strideB == 1 {
		// Fast path: contiguous vectors - reuse st function for each chunk
		parallelChunks(n, func(start, end int) {
			st.ElemVecApplyTernaryStrided(dst[start:end], condition[start:end], a[start:end], b[start:end], end-start, 1, 1, 1, 1, op)
		})
		return
	}

	// Strided path - parallelize by splitting the range
	parallelIteratorChunks(n, func(start, end int) {
		// Calculate starting indices for this chunk
		dIdx := start * strideDst
		cIdx := start * strideCond
		aIdx := start * strideA
		bIdx := start * strideB
		chunkSize := end - start

		// Reuse st function for this chunk
		st.ElemVecApplyTernaryStrided(dst[dIdx:], condition[cIdx:], a[aIdx:], b[bIdx:], chunkSize, strideDst, strideCond, strideA, strideB, op)
	})
}

// ElemVecApplyUnaryScalarStrided applies a unary function with a scalar to a vector: dst[i] = op(src[i], scalar).
// Multi-threaded version that parallelizes vector operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemVecApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, n int, strideDst, strideSrc int, op func(T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemVecApplyUnaryScalarStrided(dst, src, scalar, n, strideDst, strideSrc, op)
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors - reuse st function for each chunk
		parallelChunks(n, func(start, end int) {
			st.ElemVecApplyUnaryScalarStrided(dst[start:end], src[start:end], scalar, end-start, 1, 1, op)
		})
		return
	}

	// Strided path - parallelize by splitting the range
	parallelIteratorChunks(n, func(start, end int) {
		// Calculate starting indices for this chunk
		dIdx := start * strideDst
		sIdx := start * strideSrc
		chunkSize := end - start

		// Reuse st function for this chunk
		st.ElemVecApplyUnaryScalarStrided(dst[dIdx:], src[sIdx:], scalar, chunkSize, strideDst, strideSrc, op)
	})
}

// ElemVecApplyBinaryScalarStrided applies a binary function with a scalar to a vector: dst[i] = op(a[i], scalar).
// Multi-threaded version that parallelizes vector operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemVecApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, n int, strideDst, strideA int, op func(T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemVecApplyBinaryScalarStrided(dst, a, scalar, n, strideDst, strideA, op)
		return
	}

	if strideDst == 1 && strideA == 1 {
		// Fast path: contiguous vectors - reuse st function for each chunk
		parallelChunks(n, func(start, end int) {
			st.ElemVecApplyBinaryScalarStrided(dst[start:end], a[start:end], scalar, end-start, 1, 1, op)
		})
		return
	}

	// Strided path - parallelize by splitting the range
	parallelIteratorChunks(n, func(start, end int) {
		// Calculate starting indices for this chunk
		dIdx := start * strideDst
		aIdx := start * strideA
		chunkSize := end - start

		// Reuse st function for this chunk
		st.ElemVecApplyBinaryScalarStrided(dst[dIdx:], a[aIdx:], scalar, chunkSize, strideDst, strideA, op)
	})
}

// ElemVecApplyTernaryScalarStrided applies a ternary function with a scalar to a vector: dst[i] = op(condition[i], a[i], scalar).
// Multi-threaded version that parallelizes vector operations across CPU cores.
// Falls back to single-threaded implementation for small arrays.
func ElemVecApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, n int, strideDst, strideCond, strideA int, op func(T, T, T) T) {
	if n == 0 {
		return
	}

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemVecApplyTernaryScalarStrided(dst, condition, a, scalar, n, strideDst, strideCond, strideA, op)
		return
	}

	if strideDst == 1 && strideCond == 1 && strideA == 1 {
		// Fast path: contiguous vectors - reuse st function for each chunk
		parallelChunks(n, func(start, end int) {
			st.ElemVecApplyTernaryScalarStrided(dst[start:end], condition[start:end], a[start:end], scalar, end-start, 1, 1, 1, op)
		})
		return
	}

	// Strided path - parallelize by splitting the range
	parallelIteratorChunks(n, func(start, end int) {
		// Calculate starting indices for this chunk
		dIdx := start * strideDst
		cIdx := start * strideCond
		aIdx := start * strideA
		chunkSize := end - start

		// Reuse st function for this chunk
		st.ElemVecApplyTernaryScalarStrided(dst[dIdx:], condition[cIdx:], a[aIdx:], scalar, chunkSize, strideDst, strideCond, strideA, op)
	})
}
