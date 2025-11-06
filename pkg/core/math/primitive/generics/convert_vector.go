package generics

// ElemVecConvertStrided converts src into dst for a vector with stride support.
// Optimized for 1D vector operations.
// n is the vector length, strideDst and strideSrc are the strides.
func ElemVecConvertStrided[T, U Numeric](dst []T, src []U, n int, strideDst, strideSrc int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors
		ElemConvert(dst, src, n)
		return
	}

	// Strided path: use ValueConvert for each element to handle clamping correctly
	dIdx := 0
	sIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = ValueConvert[U, T](src[sIdx])
		dIdx += strideDst
		sIdx += strideSrc
	}
}

// ElemMatConvertStrided converts src into dst for a matrix with leading dimension support.
// Optimized for 2D matrix operations.
// rows and cols are the matrix dimensions, ldDst and ldSrc are leading dimensions (typically cols for row-major).
func ElemMatConvertStrided[T, U Numeric](dst []T, src []U, rows, cols int, ldDst, ldSrc int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices
		size := rows * cols
		ElemConvert(dst, src, size)
		return
	}

	// Strided path: iterate row by row, use ValueConvert for each element to handle clamping correctly
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		srcRow := src[i*ldSrc:]
		for j := 0; j < cols; j++ {
			dstRow[j] = ValueConvert[U, T](srcRow[j])
		}
	}
}

