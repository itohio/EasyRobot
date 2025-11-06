package generics

// ElemVecSignStrided writes the sign of src into dst for a vector with stride support.
// Optimized for 1D vector operations.
// n is the vector length, strideDst and strideSrc are the strides.
func ElemVecSignStrided[T Numeric](dst, src []T, n int, strideDst, strideSrc int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors
		ElemSign(dst, src, n)
		return
	}

	// Strided path
	dIdx := 0
	sIdx := 0
	for i := 0; i < n; i++ {
		v := src[sIdx]
		if v > 0 {
			dst[dIdx] = 1
		} else if v < 0 {
			dst[dIdx] = -1
		} else {
			dst[dIdx] = 0
		}
		dIdx += strideDst
		sIdx += strideSrc
	}
}

// ElemMatSignStrided writes the sign of src into dst for a matrix with leading dimension support.
// Optimized for 2D matrix operations.
// rows and cols are the matrix dimensions, ldDst and ldSrc are leading dimensions (typically cols for row-major).
func ElemMatSignStrided[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices
		size := rows * cols
		ElemSign(dst, src, size)
		return
	}

	// Strided path: iterate row by row
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		srcRow := src[i*ldSrc:]
		for j := 0; j < cols; j++ {
			v := srcRow[j]
			if v > 0 {
				dstRow[j] = 1
			} else if v < 0 {
				dstRow[j] = -1
			} else {
				dstRow[j] = 0
			}
		}
	}
}

// ElemVecNegativeStrided writes the negation of src into dst for a vector with stride support.
// Optimized for 1D vector operations.
// n is the vector length, strideDst and strideSrc are the strides.
func ElemVecNegativeStrided[T Numeric](dst, src []T, n int, strideDst, strideSrc int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors
		ElemNegative(dst, src, n)
		return
	}

	// Strided path
	dIdx := 0
	sIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = -src[sIdx]
		dIdx += strideDst
		sIdx += strideSrc
	}
}

// ElemMatNegativeStrided writes the negation of src into dst for a matrix with leading dimension support.
// Optimized for 2D matrix operations.
// rows and cols are the matrix dimensions, ldDst and ldSrc are leading dimensions (typically cols for row-major).
func ElemMatNegativeStrided[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices
		size := rows * cols
		ElemNegative(dst, src, size)
		return
	}

	// Strided path: iterate row by row
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		srcRow := src[i*ldSrc:]
		for j := 0; j < cols; j++ {
			dstRow[j] = -srcRow[j]
		}
	}
}

