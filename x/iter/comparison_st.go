//go:build !use_mt

package iter

import (
	st "github.com/itohio/EasyRobot/x/iter/st"
	"github.com/itohio/EasyRobot/x/types"
)

// Re-export comparison functions from single-threaded implementation
func ElemGreaterThan[T types.NumberReal](dst, a, b []T, n int) {
	st.ElemGreaterThan(dst, a, b, n)
}

func ElemGreaterThanStrided[T types.NumberReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemGreaterThanStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemEqual[T types.Number](dst, a, b []T, n int) {
	st.ElemEqual(dst, a, b, n)
}

func ElemEqualStrided[T types.Number](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemLess[T types.NumberReal](dst, a, b []T, n int) {
	st.ElemLess(dst, a, b, n)
}

func ElemLessStrided[T types.NumberReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemLessStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemNotEqual[T types.Number](dst, a, b []T, n int) {
	st.ElemNotEqual(dst, a, b, n)
}

func ElemNotEqualStrided[T types.Number](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemNotEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemLessEqual[T types.NumberReal](dst, a, b []T, n int) {
	st.ElemLessEqual(dst, a, b, n)
}

func ElemLessEqualStrided[T types.NumberReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemLessEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemGreaterEqual[T types.NumberReal](dst, a, b []T, n int) {
	st.ElemGreaterEqual(dst, a, b, n)
}

func ElemGreaterEqualStrided[T types.NumberReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	st.ElemGreaterEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}
