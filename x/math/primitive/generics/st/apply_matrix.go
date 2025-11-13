package st

import (
	. "github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
)

// ElemMatApplyUnaryStrided applies a unary function to a matrix: dst[i,j] = op(src[i,j]).
// Optimized for 2D matrix operations with leading dimension support.
// rows and cols are the matrix dimensions, ldDst and ldSrc are leading dimensions (typically cols for row-major).
func ElemMatApplyUnaryStrided[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int, op func(T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldSrc == cols {
		size := rows * cols
		applyUnaryContiguous(dst, src, size, op)
		return
	}

	for i := 0; i < rows; i++ {
		dRow := dst[i*ldDst : i*ldDst+cols]
		sRow := src[i*ldSrc : i*ldSrc+cols]
		applyUnaryContiguous(dRow, sRow, cols, op)
	}
}

// ElemMatApplyBinaryStrided applies a binary function to matrices: dst[i,j] = op(a[i,j], b[i,j]).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyBinaryStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldA == cols && ldB == cols {
		size := rows * cols
		applyBinaryContiguous(dst, a, b, size, op)
		return
	}

	for i := 0; i < rows; i++ {
		dRow := dst[i*ldDst : i*ldDst+cols]
		aRow := a[i*ldA : i*ldA+cols]
		bRow := b[i*ldB : i*ldB+cols]
		applyBinaryContiguous(dRow, aRow, bRow, cols, op)
	}
}

// ElemMatApplyTernaryStrided applies a ternary function to matrices: dst[i,j] = op(condition[i,j], a[i,j], b[i,j]).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyTernaryStrided[T Numeric](dst, condition, a, b []T, rows, cols int, ldDst, ldCond, ldA, ldB int, op func(T, T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldCond == cols && ldA == cols && ldB == cols {
		size := rows * cols
		applyTernaryContiguous(dst, condition, a, b, size, op)
		return
	}

	for i := 0; i < rows; i++ {
		dRow := dst[i*ldDst : i*ldDst+cols]
		cRow := condition[i*ldCond : i*ldCond+cols]
		aRow := a[i*ldA : i*ldA+cols]
		bRow := b[i*ldB : i*ldB+cols]
		applyTernaryContiguous(dRow, cRow, aRow, bRow, cols, op)
	}
}

// ElemMatApplyUnaryScalarStrided applies a unary function with a scalar to a matrix: dst[i,j] = op(src[i,j], scalar).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, rows, cols int, ldDst, ldSrc int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldSrc == cols {
		size := rows * cols
		applyUnaryScalarContiguous(dst, src, scalar, size, op)
		return
	}

	for i := 0; i < rows; i++ {
		dRow := dst[i*ldDst : i*ldDst+cols]
		sRow := src[i*ldSrc : i*ldSrc+cols]
		applyUnaryScalarContiguous(dRow, sRow, scalar, cols, op)
	}
}

// ElemMatApplyBinaryScalarStrided applies a binary function with a scalar to a matrix: dst[i,j] = op(a[i,j], scalar).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, rows, cols int, ldDst, ldA int, op func(T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldA == cols {
		size := rows * cols
		applyBinaryScalarContiguous(dst, a, scalar, size, op)
		return
	}

	for i := 0; i < rows; i++ {
		dRow := dst[i*ldDst : i*ldDst+cols]
		aRow := a[i*ldA : i*ldA+cols]
		applyBinaryScalarContiguous(dRow, aRow, scalar, cols, op)
	}
}

// ElemMatApplyTernaryScalarStrided applies a ternary function with a scalar to a matrix: dst[i,j] = op(condition[i,j], a[i,j], scalar).
// Optimized for 2D matrix operations with leading dimension support.
func ElemMatApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, rows, cols int, ldDst, ldCond, ldA int, op func(T, T, T) T) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldCond == cols && ldA == cols {
		size := rows * cols
		applyTernaryScalarContiguous(dst, condition, a, scalar, size, op)
		return
	}

	for i := 0; i < rows; i++ {
		dRow := dst[i*ldDst : i*ldDst+cols]
		cRow := condition[i*ldCond : i*ldCond+cols]
		aRow := a[i*ldA : i*ldA+cols]
		applyTernaryScalarContiguous(dRow, cRow, aRow, scalar, cols, op)
	}
}
