//go:build use_mt

package generics

import (
	mt "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/mt"
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

// Re-export comparison functions from multi-threaded implementation
func ElemGreaterThan[T Numeric](dst, a, b []T, n int) {
	mt.ElemGreaterThan(dst, a, b, n)
}

func ElemGreaterThanStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemGreaterThanStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemEqual[T Numeric](dst, a, b []T, n int) {
	mt.ElemEqual(dst, a, b, n)
}

func ElemEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemLess[T Numeric](dst, a, b []T, n int) {
	mt.ElemLess(dst, a, b, n)
}

func ElemLessStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemLessStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemNotEqual[T Numeric](dst, a, b []T, n int) {
	mt.ElemNotEqual(dst, a, b, n)
}

func ElemNotEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemNotEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemLessEqual[T Numeric](dst, a, b []T, n int) {
	mt.ElemLessEqual(dst, a, b, n)
}

func ElemLessEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemLessEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

func ElemGreaterEqual[T Numeric](dst, a, b []T, n int) {
	mt.ElemGreaterEqual(dst, a, b, n)
}

func ElemGreaterEqualStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int) {
	mt.ElemGreaterEqualStrided(dst, a, b, shape, stridesDst, stridesA, stridesB)
}

