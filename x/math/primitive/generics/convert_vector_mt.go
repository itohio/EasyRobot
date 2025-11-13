//go:build use_mt

package generics

import (
	mt "github.com/itohio/EasyRobot/x/math/primitive/generics/mt"
)

// Re-export vector/matrix conversion functions from multi-threaded implementation
func ElemVecConvertStrided[T, U Numeric](dst []T, src []U, n int, strideDst, strideSrc int) {
	mt.ElemVecConvertStrided(dst, src, n, strideDst, strideSrc)
}

func ElemMatConvertStrided[T, U Numeric](dst []T, src []U, rows, cols int, ldDst, ldSrc int) {
	mt.ElemMatConvertStrided(dst, src, rows, cols, ldDst, ldSrc)
}
