//go:build !use_mt

package generics

import (
	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

// Re-export matrix apply functions from single-threaded implementation
func ElemMatApplyUnaryStrided[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int, op func(T) T) {
	st.ElemMatApplyUnaryStrided(dst, src, rows, cols, ldDst, ldSrc, op)
}

func ElemMatApplyBinaryStrided[T Numeric](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int, op func(T, T) T) {
	st.ElemMatApplyBinaryStrided(dst, a, b, rows, cols, ldDst, ldA, ldB, op)
}

func ElemMatApplyTernaryStrided[T Numeric](dst, condition, a, b []T, rows, cols int, ldDst, ldCond, ldA, ldB int, op func(T, T, T) T) {
	st.ElemMatApplyTernaryStrided(dst, condition, a, b, rows, cols, ldDst, ldCond, ldA, ldB, op)
}

func ElemMatApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, rows, cols int, ldDst, ldSrc int, op func(T, T) T) {
	st.ElemMatApplyUnaryScalarStrided(dst, src, scalar, rows, cols, ldDst, ldSrc, op)
}

func ElemMatApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, rows, cols int, ldDst, ldA int, op func(T, T) T) {
	st.ElemMatApplyBinaryScalarStrided(dst, a, scalar, rows, cols, ldDst, ldA, op)
}

func ElemMatApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, rows, cols int, ldDst, ldCond, ldA int, op func(T, T, T) T) {
	st.ElemMatApplyTernaryScalarStrided(dst, condition, a, scalar, rows, cols, ldDst, ldCond, ldA, op)
}
