//go:build !use_mt

package generics

import (
	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
)

// Re-export apply functions from single-threaded implementation
func ElemApplyBinary[T Numeric](dst, a, b []T, n int, op func(T, T) T) {
	st.ElemApplyBinary(dst, a, b, n, op)
}

func ElemApplyBinaryStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int, op func(T, T) T) {
	st.ElemApplyBinaryStrided(dst, a, b, shape, stridesDst, stridesA, stridesB, op)
}

func ElemApplyUnary[T Numeric](dst, src []T, n int, op func(T) T) {
	st.ElemApplyUnary(dst, src, n, op)
}

func ElemApplyUnaryStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int, op func(T) T) {
	st.ElemApplyUnaryStrided(dst, src, shape, stridesDst, stridesSrc, op)
}

func ElemApplyTernary[T Numeric](dst, condition, a, b []T, n int, op func(T, T, T) T) {
	st.ElemApplyTernary(dst, condition, a, b, n, op)
}

func ElemApplyTernaryStrided[T Numeric](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int, op func(T, T, T) T) {
	st.ElemApplyTernaryStrided(dst, condition, a, b, shape, stridesDst, stridesCond, stridesA, stridesB, op)
}

func ElemApplyUnaryScalar[T Numeric](dst, src []T, scalar T, n int, op func(T, T) T) {
	st.ElemApplyUnaryScalar(dst, src, scalar, n, op)
}

func ElemApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int, op func(T, T) T) {
	st.ElemApplyUnaryScalarStrided(dst, src, scalar, shape, stridesDst, stridesSrc, op)
}

func ElemApplyBinaryScalar[T Numeric](dst, a []T, scalar T, n int, op func(T, T) T) {
	st.ElemApplyBinaryScalar(dst, a, scalar, n, op)
}

func ElemApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, shape []int, stridesDst, stridesA []int, op func(T, T) T) {
	st.ElemApplyBinaryScalarStrided(dst, a, scalar, shape, stridesDst, stridesA, op)
}

func ElemApplyTernaryScalar[T Numeric](dst, condition, a []T, scalar T, n int, op func(T, T, T) T) {
	st.ElemApplyTernaryScalar(dst, condition, a, scalar, n, op)
}

func ElemApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, shape []int, stridesDst, stridesCond, stridesA []int, op func(T, T, T) T) {
	st.ElemApplyTernaryScalarStrided(dst, condition, a, scalar, shape, stridesDst, stridesCond, stridesA, op)
}
