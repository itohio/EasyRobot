package mt

import (
	. "github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"

	st "github.com/itohio/EasyRobot/x/math/primitive/generics/st"
)

// ElemVecConvertStrided converts src into dst for a vector with stride support.
// Multi-threaded version that parallelizes vector conversion across CPU cores.
// Falls back to single-threaded implementation for small vectors.
func ElemVecConvertStrided[T, U Numeric](dst []T, src []U, n int, strideDst, strideSrc int) {
	if n == 0 {
		return
	}

	// For small vectors, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemVecConvertStrided(dst, src, n, strideDst, strideSrc)
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors - reuse st function for each chunk
		parallelChunks(n, func(start, end int) {
			st.ElemVecConvertStrided(dst[start:end], src[start:end], end-start, 1, 1)
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
		st.ElemVecConvertStrided(dst[dIdx:], src[sIdx:], chunkSize, strideDst, strideSrc)
	})
}

// ElemMatConvertStrided converts src into dst for a matrix with leading dimension support.
// Multi-threaded version that parallelizes matrix conversion across CPU cores using row-based parallelization.
// Falls back to single-threaded implementation for small matrices.
func ElemMatConvertStrided[T, U Numeric](dst []T, src []U, rows, cols int, ldDst, ldSrc int) {
	if rows == 0 || cols == 0 {
		return
	}

	// For small matrices, fallback to single-threaded implementation
	if !shouldParallelize(rows * cols) {
		st.ElemMatConvertStrided(dst, src, rows, cols, ldDst, ldSrc)
		return
	}

	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices - reuse st function for each chunk
		size := rows * cols
		parallelChunks(size, func(start, end int) {
			st.ElemMatConvertStrided(dst[start:end], src[start:end], 1, end-start, cols, cols)
		})
		return
	}

	// Strided path: parallelize by rows
	parallelRows(rows, func(startRow, endRow int) {
		// Reuse st function for this row range
		for i := startRow; i < endRow; i++ {
			dstRow := dst[i*ldDst:]
			srcRow := src[i*ldSrc:]
			st.ElemMatConvertStrided(dstRow, srcRow, 1, cols, ldDst, ldSrc)
		}
	})
}
