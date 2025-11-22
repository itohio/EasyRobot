//go:build !use_mt

package generics

import (
	st "github.com/itohio/EasyRobot/x/math/primitive/generics/st"
)

// Re-export comparison functions from single-threaded implementation
func ElemGreaterThan[T NumericReal](dst, a, b []T, n int) {
	st.ElemGreaterThan(dst, a, b, n)
}

func ElemGreaterThanStrided[T NumericReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemGreaterThanStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemEqual[T Numeric](dst, a, b []T, n int) {
	st.ElemEqual(dst, a, b, n)
}

func ElemEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemLess[T NumericReal](dst, a, b []T, n int) {
	st.ElemLess(dst, a, b, n)
}

func ElemLessStrided[T NumericReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemLessStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemNotEqual[T Numeric](dst, a, b []T, n int) {
	st.ElemNotEqual(dst, a, b, n)
}

func ElemNotEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemNotEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemLessEqual[T NumericReal](dst, a, b []T, n int) {
	st.ElemLessEqual(dst, a, b, n)
}

func ElemLessEqualStrided[T NumericReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemLessEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemGreaterEqual[T NumericReal](dst, a, b []T, n int) {
	st.ElemGreaterEqual(dst, a, b, n)
}

func ElemGreaterEqualStrided[T NumericReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemGreaterEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}
