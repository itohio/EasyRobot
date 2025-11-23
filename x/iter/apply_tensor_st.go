//go:build !use_mt

package iter

import (
	st "github.com/itohio/EasyRobot/x/iter/st"
	"github.com/itohio/EasyRobot/x/types"
)

// Re-export apply functions from single-threaded implementation
func ElemApplyBinary[T types.Number](dst, a, b []T, n int, op func(T, T) T) {
	st.ElemApplyBinary(dst, a, b, n, op)
}

func ElemApplyBinaryStrided[T types.Number](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int, op func(T, T) T) {
	st.ElemApplyBinaryStrided(dst, a, b, shape, stridesDst, stridesA, stridesB, op)
}

func ElemApplyUnary[T types.Number](dst, src []T, n int, op func(T) T) {
	st.ElemApplyUnary(dst, src, n, op)
}

func ElemApplyUnaryStrided[T types.Number](dst, src []T, shape []int, stridesDst, stridesSrc []int, op func(T) T) {
	st.ElemApplyUnaryStrided(dst, src, shape, stridesDst, stridesSrc, op)
}

func ElemApplyTernary[T types.Number](dst, condition, a, b []T, n int, op func(T, T, T) T) {
	st.ElemApplyTernary(dst, condition, a, b, n, op)
}

func ElemApplyTernaryStrided[T types.Number](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int, op func(T, T, T) T) {
	st.ElemApplyTernaryStrided(dst, condition, a, b, shape, stridesDst, stridesCond, stridesA, stridesB, op)
}

func ElemApplyUnaryScalar[T types.Number](dst, src []T, scalar T, n int, op func(T, T) T) {
	st.ElemApplyUnaryScalar(dst, src, scalar, n, op)
}

func ElemApplyUnaryScalarStrided[T types.Number](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int, op func(T, T) T) {
	st.ElemApplyUnaryScalarStrided(dst, src, scalar, shape, stridesDst, stridesSrc, op)
}

func ElemApplyBinaryScalar[T types.Number](dst, a []T, scalar T, n int, op func(T, T) T) {
	st.ElemApplyBinaryScalar(dst, a, scalar, n, op)
}

func ElemApplyBinaryScalarStrided[T types.Number](dst, a []T, scalar T, shape []int, stridesDst, stridesA []int, op func(T, T) T) {
	st.ElemApplyBinaryScalarStrided(dst, a, scalar, shape, stridesDst, stridesA, op)
}

func ElemApplyTernaryScalar[T types.Number](dst, condition, a []T, scalar T, n int, op func(T, T, T) T) {
	st.ElemApplyTernaryScalar(dst, condition, a, scalar, n, op)
}

func ElemApplyTernaryScalarStrided[T types.Number](dst, condition, a []T, scalar T, shape []int, stridesDst, stridesCond, stridesA []int, op func(T, T, T) T) {
	st.ElemApplyTernaryScalarStrided(dst, condition, a, scalar, shape, stridesDst, stridesCond, stridesA, op)
}
