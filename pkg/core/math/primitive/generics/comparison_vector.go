package generics

// ElemVecGreaterThanStrided writes 1 where a > b, 0 otherwise for vectors with stride support.
// Optimized for 1D vector operations.
// n is the vector length, strideDst, strideA, and strideB are the strides.
func ElemVecGreaterThanStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		// Fast path: contiguous vectors
		ElemGreaterThan(dst, a, b, n)
		return
	}

	// Strided path
	dIdx := 0
	aIdx := 0
	bIdx := 0
	for i := 0; i < n; i++ {
		if a[aIdx] > b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
}

// ElemMatGreaterThanStrided writes 1 where a > b, 0 otherwise for matrices with leading dimension support.
// Optimized for 2D matrix operations.
// rows and cols are the matrix dimensions, ldDst, ldA, and ldB are leading dimensions (typically cols for row-major).
func ElemMatGreaterThanStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldA == cols && ldB == cols {
		// Fast path: contiguous matrices
		size := rows * cols
		ElemGreaterThan(dst, a, b, size)
		return
	}

	// Strided path: iterate row by row
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		aRow := a[i*ldA:]
		bRow := b[i*ldB:]
		for j := 0; j < cols; j++ {
			if aRow[j] > bRow[j] {
				dstRow[j] = 1
			} else {
				dstRow[j] = 0
			}
		}
	}
}

// ElemVecEqualStrided writes 1 where a == b, 0 otherwise for vectors with stride support.
// Optimized for 1D vector operations.
func ElemVecEqualStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		ElemEqual(dst, a, b, n)
		return
	}

	dIdx := 0
	aIdx := 0
	bIdx := 0
	for i := 0; i < n; i++ {
		if a[aIdx] == b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
}

// ElemMatEqualStrided writes 1 where a == b, 0 otherwise for matrices with leading dimension support.
// Optimized for 2D matrix operations.
func ElemMatEqualStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldA == cols && ldB == cols {
		size := rows * cols
		ElemEqual(dst, a, b, size)
		return
	}

	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		aRow := a[i*ldA:]
		bRow := b[i*ldB:]
		for j := 0; j < cols; j++ {
			if aRow[j] == bRow[j] {
				dstRow[j] = 1
			} else {
				dstRow[j] = 0
			}
		}
	}
}

// ElemVecLessStrided writes 1 where a < b, 0 otherwise for vectors with stride support.
// Optimized for 1D vector operations.
func ElemVecLessStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		ElemLess(dst, a, b, n)
		return
	}

	dIdx := 0
	aIdx := 0
	bIdx := 0
	for i := 0; i < n; i++ {
		if a[aIdx] < b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
}

// ElemMatLessStrided writes 1 where a < b, 0 otherwise for matrices with leading dimension support.
// Optimized for 2D matrix operations.
func ElemMatLessStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldA == cols && ldB == cols {
		size := rows * cols
		ElemLess(dst, a, b, size)
		return
	}

	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		aRow := a[i*ldA:]
		bRow := b[i*ldB:]
		for j := 0; j < cols; j++ {
			if aRow[j] < bRow[j] {
				dstRow[j] = 1
			} else {
				dstRow[j] = 0
			}
		}
	}
}

// ElemVecNotEqualStrided writes 1 where a != b, 0 otherwise for vectors with stride support.
// Optimized for 1D vector operations.
func ElemVecNotEqualStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		ElemNotEqual(dst, a, b, n)
		return
	}

	dIdx := 0
	aIdx := 0
	bIdx := 0
	for i := 0; i < n; i++ {
		if a[aIdx] != b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
}

// ElemMatNotEqualStrided writes 1 where a != b, 0 otherwise for matrices with leading dimension support.
// Optimized for 2D matrix operations.
func ElemMatNotEqualStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldA == cols && ldB == cols {
		size := rows * cols
		ElemNotEqual(dst, a, b, size)
		return
	}

	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		aRow := a[i*ldA:]
		bRow := b[i*ldB:]
		for j := 0; j < cols; j++ {
			if aRow[j] != bRow[j] {
				dstRow[j] = 1
			} else {
				dstRow[j] = 0
			}
		}
	}
}

// ElemVecLessEqualStrided writes 1 where a <= b, 0 otherwise for vectors with stride support.
// Optimized for 1D vector operations.
func ElemVecLessEqualStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		ElemLessEqual(dst, a, b, n)
		return
	}

	dIdx := 0
	aIdx := 0
	bIdx := 0
	for i := 0; i < n; i++ {
		if a[aIdx] <= b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
}

// ElemMatLessEqualStrided writes 1 where a <= b, 0 otherwise for matrices with leading dimension support.
// Optimized for 2D matrix operations.
func ElemMatLessEqualStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldA == cols && ldB == cols {
		size := rows * cols
		ElemLessEqual(dst, a, b, size)
		return
	}

	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		aRow := a[i*ldA:]
		bRow := b[i*ldB:]
		for j := 0; j < cols; j++ {
			if aRow[j] <= bRow[j] {
				dstRow[j] = 1
			} else {
				dstRow[j] = 0
			}
		}
	}
}

// ElemVecGreaterEqualStrided writes 1 where a >= b, 0 otherwise for vectors with stride support.
// Optimized for 1D vector operations.
func ElemVecGreaterEqualStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		ElemGreaterEqual(dst, a, b, n)
		return
	}

	dIdx := 0
	aIdx := 0
	bIdx := 0
	for i := 0; i < n; i++ {
		if a[aIdx] >= b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
}

// ElemMatGreaterEqualStrided writes 1 where a >= b, 0 otherwise for matrices with leading dimension support.
// Optimized for 2D matrix operations.
func ElemMatGreaterEqualStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldA == cols && ldB == cols {
		size := rows * cols
		ElemGreaterEqual(dst, a, b, size)
		return
	}

	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		aRow := a[i*ldA:]
		bRow := b[i*ldB:]
		for j := 0; j < cols; j++ {
			if aRow[j] >= bRow[j] {
				dstRow[j] = 1
			} else {
				dstRow[j] = 0
			}
		}
	}
}

