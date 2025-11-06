package generics

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// ElemVecApply applies a unary function to a vector: dst[i] = op(src[i]).
// Optimized for 1D vector operations with stride support.
// Deprecated: Use ElemVecApplyUnaryStrided instead. This function is kept for backward compatibility.
func ElemVecApply[T Numeric](dst, src []T, n int, strideDst, strideSrc int, op func(T) T) {
	ElemVecApplyUnaryStrided(dst, src, n, strideDst, strideSrc, op)
}

// ElemMatApply applies a unary function to a matrix: dst[i,j] = op(src[i,j]).
// Optimized for 2D matrix operations with leading dimension support.
// rows and cols are the matrix dimensions, ldDst and ldSrc are leading dimensions (typically cols for row-major).
// Deprecated: Use ElemMatApplyUnaryStrided instead. This function is kept for backward compatibility.
func ElemMatApply[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int, op func(T) T) {
	ElemMatApplyUnaryStrided(dst, src, rows, cols, ldDst, ldSrc, op)
}
