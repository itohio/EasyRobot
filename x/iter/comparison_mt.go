//go:build use_mt

package iter

import (
	mt "github.com/itohio/EasyRobot/x/iter/mt"
	"github.com/itohio/EasyRobot/x/types"
)

// Re-export comparison functions from multi-threaded implementation
func ElemGreaterThan[T types.NumericReal](dst, a, b []T, n int) {
	mt.ElemGreaterThan(dst, a, b, n)
}

func ElemGreaterThanStrided[T types.NumericReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemGreaterThanStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemEqual[T types.Numeric](dst, a, b []T, n int) {
	mt.ElemEqual(dst, a, b, n)
}

func ElemEqualStrided[T types.Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemLess[T types.NumericReal](dst, a, b []T, n int) {
	mt.ElemLess(dst, a, b, n)
}

func ElemLessStrided[T types.NumericReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemLessStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemNotEqual[T types.Numeric](dst, a, b []T, n int) {
	mt.ElemNotEqual(dst, a, b, n)
}

func ElemNotEqualStrided[T types.Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemNotEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemLessEqual[T types.NumericReal](dst, a, b []T, n int) {
	mt.ElemLessEqual(dst, a, b, n)
}

func ElemLessEqualStrided[T types.NumericReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemLessEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemGreaterEqual[T types.NumericReal](dst, a, b []T, n int) {
	mt.ElemGreaterEqual(dst, a, b, n)
}

func ElemGreaterEqualStrided[T types.NumericReal](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemGreaterEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}
