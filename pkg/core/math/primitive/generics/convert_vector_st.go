//go:build !use_mt

package generics

import (
	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

// Re-export vector/matrix conversion functions from single-threaded implementation
func ElemVecConvertStrided[T, U Numeric](dst []T, src []U, n int, strideDst, strideSrc int) {
	st.ElemVecConvertStrided(dst, src, n, strideDst, strideSrc)
}

func ElemMatConvertStrided[T, U Numeric](dst []T, src []U, rows, cols int, ldDst, ldSrc int) {
	st.ElemMatConvertStrided(dst, src, rows, cols, ldDst, ldSrc)
}
