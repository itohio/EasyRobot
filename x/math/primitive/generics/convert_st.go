//go:build !use_mt

package generics

import (
	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
)

// Re-export conversion functions from single-threaded implementation
func ElemConvert[T, U Numeric](dst []T, src []U, n int) {
	st.ElemConvert(dst, src, n)
}

func ElemConvertStrided[T, U Numeric](dst []T, src []U, shape []int, stridesDst, stridesSrc []int) {
	st.ElemConvertStrided(dst, src, shape, stridesDst, stridesSrc)
}
