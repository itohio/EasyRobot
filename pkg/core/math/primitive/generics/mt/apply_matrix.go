package mt

import (
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
)

// ElemMatApplyUnaryStrided applies a unary function to a matrix: dst[i,j] = op(src[i,j]).
// Multi-threaded version that parallelizes matrix operations across CPU cores using row-based parallelization.
// Falls back to single-threaded implementation for small matrices.
func ElemMatApplyUnaryStrided[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int, op func(T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	// For small matrices, fallback to single-threaded implementation
	if !shouldParallelize(rows * cols) {
		st.ElemMatApplyUnaryStrided(dst, src, rows, cols, ldDst, ldSrc, op)
		return
	}

	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices - reuse st function for each chunk
		size := rows * cols
		parallelChunks(size, func(start, end int) {
			st.ElemMatApplyUnaryStrided(dst[start:end], src[start:end], 1, end-start, cols, cols, op)
		})
		return
	}

	// Strided path: parallelize by rows
	parallelRows(rows, func(startRow, endRow int) {
		// Reuse st function for this row range
		for i := startRow; i < endRow; i++ {
			dstRow := dst[i*ldDst:]
			srcRow := src[i*ldSrc:]
			st.ElemMatApplyUnaryStrided(dstRow, srcRow, 1, cols, ldDst, ldSrc, op)
		}
	})
}

// ElemMatApplyBinaryStrided applies a binary function to matrices: dst[i,j] = op(a[i,j], b[i,j]).
// Multi-threaded version that parallelizes matrix operations across CPU cores using row-based parallelization.
// Falls back to single-threaded implementation for small matrices.
func ElemMatApplyBinaryStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	// For small matrices, fallback to single-threaded implementation
	if !shouldParallelize(rows * cols) {
		st.ElemMatApplyBinaryStrided(dst, a, b, rows, cols, ldDst, ldA, ldB, op)
		return
	}

	if ldDst == cols && ldA == cols && ldB == cols {
		// Fast path: contiguous matrices - reuse st function for each chunk
		size := rows * cols
		parallelChunks(size, func(start, end int) {
			st.ElemMatApplyBinaryStrided(dst[start:end], a[start:end], b[start:end], 1, end-start, cols, cols, cols, op)
		})
		return
	}

	// Strided path: parallelize by rows
	parallelRows(rows, func(startRow, endRow int) {
		// Reuse st function for this row range
		for i := startRow; i < endRow; i++ {
			dstRow := dst[i*ldDst:]
			aRow := a[i*ldA:]
			bRow := b[i*ldB:]
			st.ElemMatApplyBinaryStrided(dstRow, aRow, bRow, 1, cols, ldDst, ldA, ldB, op)
		}
	})
}

// ElemMatApplyTernaryStrided applies a ternary function to matrices: dst[i,j] = op(condition[i,j], a[i,j], b[i,j]).
// Multi-threaded version that parallelizes matrix operations across CPU cores using row-based parallelization.
// Falls back to single-threaded implementation for small matrices.
func ElemMatApplyTernaryStrided[T Numeric](dst, condition, a, b []T, rows, cols int, ldDst, ldCond, ldA, ldB int, op func(T, T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	// For small matrices, fallback to single-threaded implementation
	if !shouldParallelize(rows * cols) {
		st.ElemMatApplyTernaryStrided(dst, condition, a, b, rows, cols, ldDst, ldCond, ldA, ldB, op)
		return
	}

	if ldDst == cols && ldCond == cols && ldA == cols && ldB == cols {
		// Fast path: contiguous matrices - reuse st function for each chunk
		size := rows * cols
		parallelChunks(size, func(start, end int) {
			st.ElemMatApplyTernaryStrided(dst[start:end], condition[start:end], a[start:end], b[start:end], 1, end-start, cols, cols, cols, cols, op)
		})
		return
	}

	// Strided path: parallelize by rows
	parallelRows(rows, func(startRow, endRow int) {
		// Reuse st function for this row range
		for i := startRow; i < endRow; i++ {
			dstRow := dst[i*ldDst:]
			condRow := condition[i*ldCond:]
			aRow := a[i*ldA:]
			bRow := b[i*ldB:]
			st.ElemMatApplyTernaryStrided(dstRow, condRow, aRow, bRow, 1, cols, ldDst, ldCond, ldA, ldB, op)
		}
	})
}

// ElemMatApplyUnaryScalarStrided applies a unary function with a scalar to a matrix: dst[i,j] = op(src[i,j], scalar).
// Multi-threaded version that parallelizes matrix operations across CPU cores using row-based parallelization.
// Falls back to single-threaded implementation for small matrices.
func ElemMatApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, rows, cols int, ldDst, ldSrc int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	// For small matrices, fallback to single-threaded implementation
	if !shouldParallelize(rows * cols) {
		st.ElemMatApplyUnaryScalarStrided(dst, src, scalar, rows, cols, ldDst, ldSrc, op)
		return
	}

	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices - reuse st function for each chunk
		size := rows * cols
		parallelChunks(size, func(start, end int) {
			st.ElemMatApplyUnaryScalarStrided(dst[start:end], src[start:end], scalar, 1, end-start, cols, cols, op)
		})
		return
	}

	// Strided path: parallelize by rows
	parallelRows(rows, func(startRow, endRow int) {
		// Reuse st function for this row range
		for i := startRow; i < endRow; i++ {
			dstRow := dst[i*ldDst:]
			srcRow := src[i*ldSrc:]
			st.ElemMatApplyUnaryScalarStrided(dstRow, srcRow, scalar, 1, cols, ldDst, ldSrc, op)
		}
	})
}

// ElemMatApplyBinaryScalarStrided applies a binary function with a scalar to a matrix: dst[i,j] = op(a[i,j], scalar).
// Multi-threaded version that parallelizes matrix operations across CPU cores using row-based parallelization.
// Falls back to single-threaded implementation for small matrices.
func ElemMatApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, rows, cols int, ldDst, ldA int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	// For small matrices, fallback to single-threaded implementation
	if !shouldParallelize(rows * cols) {
		st.ElemMatApplyBinaryScalarStrided(dst, a, scalar, rows, cols, ldDst, ldA, op)
		return
	}

	if ldDst == cols && ldA == cols {
		// Fast path: contiguous matrices - reuse st function for each chunk
		size := rows * cols
		parallelChunks(size, func(start, end int) {
			st.ElemMatApplyBinaryScalarStrided(dst[start:end], a[start:end], scalar, 1, end-start, cols, cols, op)
		})
		return
	}

	// Strided path: parallelize by rows
	parallelRows(rows, func(startRow, endRow int) {
		// Reuse st function for this row range
		for i := startRow; i < endRow; i++ {
			dstRow := dst[i*ldDst:]
			aRow := a[i*ldA:]
			st.ElemMatApplyBinaryScalarStrided(dstRow, aRow, scalar, 1, cols, ldDst, ldA, op)
		}
	})
}

// ElemMatApplyTernaryScalarStrided applies a ternary function with a scalar to a matrix: dst[i,j] = op(condition[i,j], a[i,j], scalar).
// Multi-threaded version that parallelizes matrix operations across CPU cores using row-based parallelization.
// Falls back to single-threaded implementation for small matrices.
func ElemMatApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, rows, cols int, ldDst, ldCond, ldA int, op func(T, T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	// For small matrices, fallback to single-threaded implementation
	if !shouldParallelize(rows * cols) {
		st.ElemMatApplyTernaryScalarStrided(dst, condition, a, scalar, rows, cols, ldDst, ldCond, ldA, op)
		return
	}

	if ldDst == cols && ldCond == cols && ldA == cols {
		// Fast path: contiguous matrices - reuse st function for each chunk
		size := rows * cols
		parallelChunks(size, func(start, end int) {
			st.ElemMatApplyTernaryScalarStrided(dst[start:end], condition[start:end], a[start:end], scalar, 1, end-start, cols, cols, cols, op)
		})
		return
	}

	// Strided path: parallelize by rows
	parallelRows(rows, func(startRow, endRow int) {
		// Reuse st function for this row range
		for i := startRow; i < endRow; i++ {
			dstRow := dst[i*ldDst:]
			condRow := condition[i*ldCond:]
			aRow := a[i*ldA:]
			st.ElemMatApplyTernaryScalarStrided(dstRow, condRow, aRow, scalar, 1, cols, ldDst, ldCond, ldA, op)
		}
	})
}
