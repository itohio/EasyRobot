//go:build use_mt

package generics

import (
	mt "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/mt"
)

// Re-export apply functions from multi-threaded implementation
func ElemApplyBinary[T Numeric](dst, a, b []T, n int, op func(T, T) T) {
	mt.ElemApplyBinary(dst, a, b, n, op)
}

func ElemApplyBinaryStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int, op func(T, T) T) {
	mt.ElemApplyBinaryStrided(dst, a, b, shape, stridesDst, stridesA, stridesB, op)
}

func ElemApplyUnary[T Numeric](dst, src []T, n int, op func(T) T) {
	mt.ElemApplyUnary(dst, src, n, op)
}

func ElemApplyUnaryStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int, op func(T) T) {
	mt.ElemApplyUnaryStrided(dst, src, shape, stridesDst, stridesSrc, op)
}

func ElemApplyTernary[T Numeric](dst, condition, a, b []T, n int, op func(T, T, T) T) {
	mt.ElemApplyTernary(dst, condition, a, b, n, op)
}

func ElemApplyTernaryStrided[T Numeric](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int, op func(T, T, T) T) {
	mt.ElemApplyTernaryStrided(dst, condition, a, b, shape, stridesDst, stridesCond, stridesA, stridesB, op)
}

func ElemApplyUnaryScalar[T Numeric](dst, src []T, scalar T, n int, op func(T, T) T) {
	mt.ElemApplyUnaryScalar(dst, src, scalar, n, op)
}

func ElemApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int, op func(T, T) T) {
	mt.ElemApplyUnaryScalarStrided(dst, src, scalar, shape, stridesDst, stridesSrc, op)
}

func ElemApplyBinaryScalar[T Numeric](dst, a []T, scalar T, n int, op func(T, T) T) {
	mt.ElemApplyBinaryScalar(dst, a, scalar, n, op)
}

func ElemApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, shape []int, stridesDst, stridesA []int, op func(T, T) T) {
	mt.ElemApplyBinaryScalarStrided(dst, a, scalar, shape, stridesDst, stridesA, op)
}

func ElemApplyTernaryScalar[T Numeric](dst, condition, a []T, scalar T, n int, op func(T, T, T) T) {
	mt.ElemApplyTernaryScalar(dst, condition, a, scalar, n, op)
}

func ElemApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, shape []int, stridesDst, stridesCond, stridesA []int, op func(T, T, T) T) {
	mt.ElemApplyTernaryScalarStrided(dst, condition, a, scalar, shape, stridesDst, stridesCond, stridesA, op)
}
