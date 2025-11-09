//go:build use_mt

package generics

import (
	mt "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/mt"
)

// Re-export conversion functions from multi-threaded implementation
func ElemConvert[T, U Numeric](dst []T, src []U, n int) {
	mt.ElemConvert(dst, src, n)
}

func ElemConvertStrided[T, U Numeric](dst []T, src []U, shape []int, stridesDst, stridesSrc []int) {
	mt.ElemConvertStrided(dst, src, shape, stridesDst, stridesSrc)
}
