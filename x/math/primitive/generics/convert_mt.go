//go:build use_mt

package generics

import (
	// TODO: Implement multi-threaded conversion functions that use the st implementation under the hood
	mt "github.com/itohio/EasyRobot/x/math/primitive/generics/st"
)

// Re-export conversion functions from multi-threaded implementation
func ElemConvert[T, U Numeric](dst []T, src []U, n int) {
	mt.ElemConvert(dst, src, n)
}

func ElemConvertStrided[T, U Numeric](dst []T, src []U, shape []int, stridesDst, stridesSrc []int) {
	mt.ElemConvertStrided(dst, src, shape, stridesDst, stridesSrc)
}
