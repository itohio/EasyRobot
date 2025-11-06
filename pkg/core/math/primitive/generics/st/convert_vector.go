package st

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// ElemVecConvertStrided converts src into dst for a vector with stride support.
// Optimized for 1D vector operations.
// n is the vector length, strideDst and strideSrc are the strides.
func ElemVecConvertStrided[T, U Numeric](dst []T, src []U, n int, strideDst, strideSrc int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors - use elemConvertNumeric directly
		_ = elemConvertNumeric(dst[:n], src[:n])
		return
	}

	// Strided path: convert each element individually
	dIdx := 0
	sIdx := 0
	for i := 0; i < n; i++ {
		// Convert single element using the same logic as ElemConvert
		val := src[sIdx]
		dst[dIdx] = T(val)
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
		// Fast path: contiguous matrices - use elemConvertNumeric directly
		size := rows * cols
		_ = elemConvertNumeric(dst[:size], src[:size])
		return
	}

	// Strided path: iterate row by row, convert each element
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		srcRow := src[i*ldSrc:]
		// Convert row using elemConvertNumeric
		_ = elemConvertNumeric(dstRow[:cols], srcRow[:cols])
	}
}

