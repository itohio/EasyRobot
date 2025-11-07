package st

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// ElemMatApplyUnaryStrided applies a unary function to a matrix: dst[i,j] = op(src[i,j]).
// Optimized for 2D matrix operations with leading dimension support.
// rows and cols are the matrix dimensions, ldDst and ldSrc are leading dimensions (typically cols for row-major).
func ElemMatApplyUnaryStrided[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int, op func(T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices
		// Boundary check elimination hint
		size := rows * cols
		dst = dst[:size]
		src = src[:size]
		for i := range size {
			dst[i] = op(src[i])
		}
		return
	}

	// Strided path: iterate row by row
	for i := range rows {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		srcRow := src[i*ldSrc : i*ldSrc+cols]
		dstRow = dstRow[:cols] // NOOP, but hints the compiler that we are not writing beyond the bounds
		srcRow = srcRow[:cols]
		for j := range cols {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// ElemMatApplyBinaryStrided applies a binary function to matrices: dst[i,j] = op(a[i,j], b[i,j]).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyBinaryStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	size := rows * cols
	dst = dst[:size]
	a = a[:size]
	b = b[:size]
	if ldDst == cols && ldA == cols && ldB == cols {
		// Fast path: contiguous matrices
		// Boundary check elimination hint
		for i := range size {
			dst[i] = op(a[i], b[i])
		}
		return
	}

	// Strided path: iterate row by row
	for i := range rows {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		aRow := a[i*ldA : i*ldA+cols]
		bRow := b[i*ldB : i*ldB+cols]
		dstRow = dstRow[:cols] // NOOP, but hints the compiler that we are not writing beyond the bounds
		aRow = aRow[:cols]
		bRow = bRow[:cols]
		for j := range cols {
			dstRow[j] = op(aRow[j], bRow[j])
		}
	}
}

// ElemMatApplyTernaryStrided applies a ternary function to matrices: dst[i,j] = op(condition[i,j], a[i,j], b[i,j]).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyTernaryStrided[T Numeric](dst, condition, a, b []T, rows, cols int, ldDst, ldCond, ldA, ldB int, op func(T, T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	size := rows * cols
	dst = dst[:size]
	condition = condition[:size]
	a = a[:size]
	b = b[:size]
	if ldDst == cols && ldCond == cols && ldA == cols && ldB == cols {
		// Fast path: contiguous matrices
		// Boundary check elimination hint
		for i := range size {
			dst[i] = op(condition[i], a[i], b[i])
		}
		return
	}

	// Strided path: iterate row by row
	for i := range rows {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		condRow := condition[i*ldCond : i*ldCond+cols]
		aRow := a[i*ldA : i*ldA+cols]
		bRow := b[i*ldB : i*ldB+cols]
		dstRow = dstRow[:cols] // NOOP, but hints the compiler that we are not writing beyond the bounds
		condRow = condRow[:cols]
		aRow = aRow[:cols]
		bRow = bRow[:cols]
		for j := range cols {
			dstRow[j] = op(condRow[j], aRow[j], bRow[j])
		}
	}
}

// ElemMatApplyUnaryScalarStrided applies a unary function with a scalar to a matrix: dst[i,j] = op(src[i,j], scalar).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, rows, cols int, ldDst, ldSrc int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	size := rows * cols
	dst = dst[:size]
	src = src[:size]
	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices
		// Boundary check elimination hint
		for i := range size {
			dst[i] = op(src[i], scalar)
		}
		return
	}

	// Strided path: iterate row by row
	for i := range rows {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		srcRow := src[i*ldSrc : i*ldSrc+cols]
		dstRow = dstRow[:cols] // NOOP, but hints the compiler that we are not writing beyond the bounds
		srcRow = srcRow[:cols]
		for j := range cols {
			dstRow[j] = op(srcRow[j], scalar)
		}
	}
}

// ElemMatApplyBinaryScalarStrided applies a binary function with a scalar to a matrix: dst[i,j] = op(a[i,j], scalar).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, rows, cols int, ldDst, ldA int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	size := rows * cols
	dst = dst[:size]
	a = a[:size]
	if ldDst == cols && ldA == cols {
		// Fast path: contiguous matrices
		// Boundary check elimination hint
		for i := range size {
			dst[i] = op(a[i], scalar)
		}
		return
	}

	// Strided path: iterate row by row
	for i := range rows {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		aRow := a[i*ldA : i*ldA+cols]
		dstRow = dstRow[:cols] // NOOP, but hints the compiler that we are not writing beyond the bounds
		aRow = aRow[:cols]
		for j := range cols {
			dstRow[j] = op(aRow[j], scalar)
		}
	}
}

// ElemMatApplyTernaryScalarStrided applies a ternary function with a scalar to a matrix: dst[i,j] = op(condition[i,j], a[i,j], scalar).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, rows, cols int, ldDst, ldCond, ldA int, op func(T, T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	size := rows * cols
	dst = dst[:size]
	condition = condition[:size]
	a = a[:size]
	if ldDst == cols && ldCond == cols && ldA == cols {
		// Fast path: contiguous matrices
		// Boundary check elimination hint
		for i := range size {
			dst[i] = op(condition[i], a[i], scalar)
		}
		return
	}

	// Strided path: iterate row by row
	for i := range rows {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		condRow := condition[i*ldCond : i*ldCond+cols]
		aRow := a[i*ldA : i*ldA+cols]
		dstRow = dstRow[:cols] // NOOP, but hints the compiler that we are not writing beyond the bounds
		condRow = condRow[:cols]
		aRow = aRow[:cols]
		for j := range cols {
			dstRow[j] = op(condRow[j], aRow[j], scalar)
		}
	}
}
